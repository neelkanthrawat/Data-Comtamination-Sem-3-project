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


def load_openllama():
    """
    Load the OpenLlama model and the tokenizer.
    """
    path = "openlm-research/open_llama_13b"
    print(f"Loading {path}...")

    tokenizer = LlamaTokenizer.from_pretrained(path)

    model = LlamaForCausalLM.from_pretrained(
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

    index = 0
    for index, row in df.iterrows():
        index += 1
        first_piece = row["Context"]
        second_piece = row["Target"]
        if df.columns.contains("Embedding"):
            label = row["Embedding"]
            formatted_prompt = prompt_template.format(
                first_piece=first_piece, label=label
            )
        else:
            formatted_prompt = prompt_template.format(first_piece=first_piece)

        print(f"Formatted prompt: \n {formatted_prompt}", flush=True)

        encoded_prompt = tokenizer.encode(formatted_prompt, return_tensors="pt")
        out = model.generate(encoded_prompt, max_new_tokens=50, return_full_text=False)
        decoded_out = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"output: \n {decoded_out}", flush=True)

        if index > 50:
            break


if __name__ == "__main__":
    main()
