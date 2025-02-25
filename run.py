import transformers
import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def load_openllama():
    """
    Load the OpenLlama model and the tokenizer.
    """
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_13b")
    model = LlamaForCausalLM.from_pretrained(
        "openlm-research/open_llama_13b",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model


def load_llama():
    """
    Load the Llama model and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
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
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.3",
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

    args = parser.parse_args()

    return args


def main():
    """
    Main function
    """
    print("Running...")
    args = parse_args()


if __name__ == "__main__":
    main()
