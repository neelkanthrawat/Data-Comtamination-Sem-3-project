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
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_openllama():
    """
    Load the OpenLlama model and the tokenizer.
    """
    path = "VMware/open-llama-7b-v2-open-instruct" #'openlm-research/open_llama_7b_v2'#"VMware/open-llama-13b-open-instruct"  #'openlm-research/open_llama_13b'#
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


def load_mistral():
    """
    Load the Mistral model and the tokenizer.
    """
    path = "mistralai/Mistral-7B-v0.3"
    print(f"Loading {path}...")

    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
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
        help="The model to use. Options: llama, openllama, mistral",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="cb",
        help="The tasks to run",
        required=False,
        choices=["cb", "winogrande"],
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

    dh = DataHandler()

    df = dh.load_dataset(args.task)

    prompt = Prompt()

    if args.type == "guided":
        prompt_template = prompt.get_guided_prompt(args.task)
    elif args.type == "unguided":
        prompt_template = prompt.get_unguided_prompt(args.task)

    results_df = pd.DataFrame(
        columns=["Index", "Embedding", "Context", "Target", "Prediction"]
    )

    for index, row in df.iterrows():
        first_piece = row["Context"]
        second_piece = row["Target"]
        if "Embedding" in df.columns:
            label = row["Embedding"]
            formatted_prompt = prompt_template.format(
                first_piece=first_piece, label=label
            )
        else:
            formatted_prompt = prompt_template.format(first_piece=first_piece)

        print(f"-------- Formatted prompt: --------\n{formatted_prompt}", flush=True)
        print("------------------------")

        encoded_prompt = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=True
        )
        print(f'encoded_prompt {encoded_prompt}')
        start_index_answer = len(encoded_prompt['input_ids'])

        out = model.generate(
            encoded_prompt['input_ids'].to(DEVICE),
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.2,
            do_sample=  True,
        )[0]#[start_index_answer:]

        print(f'out: {out}')

        decoded_out = tokenizer.decode(out, skip_special_tokens=True)
        decoded_out = decoded_out.strip()
        print(f"-------- Output: --------\n{decoded_out}", flush=True)
        print("------------------------", end="\n\n")

        results_df.loc[index] = {
            "Index": index,
            "Embedding": label if "Embedding" in df.columns else None,
            "Context": first_piece,
            "Target": second_piece,
            "Prediction": decoded_out,
        }

        if index > 10:
            break

    res_path = os.path.join("results", f"{args.task}_{args.model}_{args.type}.csv")

    if not os.path.exists("results"):
        os.makedirs("results")

    results_df.to_csv(res_path, index=False)
    print(f"Results saved to {res_path}")


if __name__ == "__main__":
    main()
