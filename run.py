import transformers
import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from DataHandler import DataHandler
from Prompt import Prompt
import pandas as pd
from pathlib import Path
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME = Path.home()
PROJECT_DIR = os.path.join(HOME, "Data-Comtamination-Sem-3-project")


def load_openllama():
    """
    Load the OpenLlama model and the tokenizer.
    """
    path = "VMware/open-llama-7b-v2-open-instruct"  #'openlm-research/open_llama_7b_v2'#"VMware/open-llama-13b-open-instruct"  #'openlm-research/open_llama_13b'#
    print(f"Loading {path}...")

    tokenizer = LlamaTokenizer.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model


def load_llama():
    """
    Load the Llama model and the tokenizer.
    """
    path = "meta-llama/Llama-3.2-3B"
    print(f"Loading {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    return tokenizer, model


def parse_args():
    """
    Parse command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the model")

    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        help="The model to use. Options: Llama, OpenLlama, Mistral",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="cb",
        help="The tasks to run",
        required=False,
        choices=["cb", "wsc", "wikipedia", "stackexchange"],
    )

    parser.add_argument(
        "--type",
        type=str,
        default="unguided",
        help="Guided or unguided instruction",
        required=False,
        choices=["guided", "unguided"],
    )

    args = parser.parse_args()
    print(args)

    return args


def main():
    """
    Main function
    """
    args = parse_args()

    if args.model not in ["Llama", "OpenLlama", "Mistral"]:
        print("Invalid model")
    elif args.model in ["Llama", "Llama3"]:
        tokenizer, model = load_llama()
    elif args.model in ["OpenLlama"]:
        tokenizer, model = load_openllama()
    ### test data loading without model
    elif args.model == "test":
        pass

    dh = DataHandler()

    dataset = dh.load_dataset(args.task)
    # shuffle the dataset and sample from it
    dataset = dataset.shuffle(seed=42)

    prompt = Prompt()

    if args.type == "guided":
        prompt_template = prompt.get_guided_prompt(args.task)
    elif args.type == "unguided":
        prompt_template = prompt.get_unguided_prompt(args.task)

    results_df = pd.DataFrame(
        columns=[
            "Index",
            "Label",
            "First piece",
            "Gold",
            "Prediction",
        ]
    )

    for index, sample in enumerate(dataset):
        if args.task == "cb":
            first_piece = sample["premise"]
            second_piece = sample["hypothesis"]
        elif args.task == "wsc":
            first_piece, second_piece = dh.split_sentence(sample["text"])
        elif args.task == "wikipedia":
            first_piece, second_piece = dh.split_sentence(sample["text"])
        elif args.task == "stackexchange":
            first_piece, second_piece = dh.split_sentence(
                sample["text"], split_with_char=0
            )

        if "label" in sample.keys():
            formatted_prompt = prompt_template.format(
                first_piece=first_piece, label=sample["label"]
            )
        else:
            formatted_prompt = prompt_template.format(first_piece=first_piece)

        print(f"-------- Formatted prompt: --------\n{formatted_prompt}", flush=True)
        print("------------------------")

        encoded_prompt = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=True
        )

        start_index_answer = len(encoded_prompt["input_ids"][0])

        if args.task == "stackexchange":
            max_len = 4000
        else:
            max_len = 100

        out = model.generate(
            encoded_prompt["input_ids"].to(DEVICE),
            max_new_tokens=max_len,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.4,
            do_sample=True,
        )[0][start_index_answer:]

        decoded_out = tokenizer.decode(out, skip_special_tokens=True)
        decoded_out = decoded_out.strip()

        print(f"-------- Output: --------\n{decoded_out}", flush=True)
        print("------------------------", end="\n\n")

        results_df.loc[index] = {
            "Index": index,
            "Label": sample["label"] if "label" in sample.keys() else None,
            "First piece": first_piece,
            "Gold": second_piece,
            "Prediction": decoded_out,
        }

        if index > 10:
            break

    res_dir = os.path.join(PROJECT_DIR, "results")

    res_path = os.path.join(res_dir, f"{args.task}_{args.model}_{args.type}.csv")

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    results_df.to_csv(res_path, index=False, sep=";")
    print(f"Results saved to {res_path}")


if __name__ == "__main__":
    main()
