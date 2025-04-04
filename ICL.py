import torch
import pandas as pd
import os
import argparse
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

HOME = Path.home()
PROJECT_DIR = os.path.join(HOME, "Data-Comtamination-Sem-3-project")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run the model")

    parser.add_argument(
        "--guided",
        type=str,
        help="Path to the guided predictions file",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the evaluation",
    )

    args = parser.parse_args()
    print(args)

    return args


def load_mistral():
    """
    Load the Mistral model and the tokenizer.
    """
    path = "mistralai/Mistral-7B-Instruct-v0.3"#  copy the load llama-instruct function 
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

def load_llama_instruct():
    """
    Load the Llama model and the tokenizer.
    """
    path = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    return tokenizer, model



def ICL_prompting(in_path: str):
    """
    Only necessary for guided.
    """
    path = os.path.join(PROJECT_DIR, in_path)

    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter="|")

    tokenizer, model = load_llama_instruct()#load_mistral()# change it to load llama-instruct

    prompt_template = """INSTRUCTION:
You are provided with a reference text and a candidate text.
The candidate text is a generated replica of the reference text.
Your task is to determine if the candidate text is an exact or near-exact match of the reference text.
Near-exact match candidates must PRECISELY match the reference candidates in terms of sentence structure, overlap, and contextual similarity.
Respond only with 'Yes' or 'No'.
---
Example 1:

REFERENCE TEXT:
The cat waited at the top.

CANDIDATE TEXT:
The cat waited at the top.

ANSWER: Yes (exact match)
---
Example 2:

REFERENCE TEXT:
icy surface of Jupiter's largest moon, Ganymede. These irregular masses may be rock formations, supported by Ganymede's icy shell for billions of years.

CANDIDATE TEXT:
icy surface of Jupiter's largest moon, Ganymede. These irregular masses may be rock formations, supported by Ganymede's icy shell for billions of years. This discovery supports the theory that Ganymede has a subsurface ocean. Scientists used gravity data from NASA's Galileo spacecraft to create a geophysical model of the interior of Ganymede.

ANSWER: Yes (near-exact match)
---
Example 3:

REFERENCE TEXT:
50th Anniversary of Normandy Landings lasts a year.

CANDIDATE TEXT:
The 50th anniversary celebration of the first Normandy landing will last a year.

ANSWER: Yes (near-exact match)
---
Example 4:

REFERENCE TEXT:
Microsoft's Hotmail has raised its storage capacity to 250MB.

CANDIDATE TEXT:
Microsoft has increased the storage capacity of its Hotmail e-mail service to 250MB.

ANSWER: Yes (near-exact match)
---
Example 5:

REFERENCE TEXT:
{ref_text}

CANDIDATE TEXT:
{cand_text}

ANSWER:
"""

    results_df = pd.DataFrame(
        columns=["Index", "Label", "First piece", "Gold", "Prediction", "Mistral"]
    )

    for index, row in df.iterrows():
        formatted_prompt = prompt_template.format(
            ref_text=row["Gold"], cand_text=row["Prediction"]
        )

        print(f"-------- Formatted prompt: --------\n{formatted_prompt}", flush=True)
        print("------------------------")

        encoded_prompt = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=True
        )

        start_index_answer = len(encoded_prompt["input_ids"][0])

        out = model.generate(
            encoded_prompt["input_ids"].to(DEVICE),
            max_new_tokens=30,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.2,
            do_sample=True,
        )[0][start_index_answer:]

        decoded_out = tokenizer.decode(out, skip_special_tokens=True)
        decoded_out = decoded_out.strip()

        print(f"-------- Output: --------\n{decoded_out}", flush=True)
        print("------------------------", end="\n\n")

        results_df.loc[index] = {
            "Index": index,
            # "Label": row["label"] if "label" in row.columns() else None,
            "First piece": row["First piece"],
            "Gold": row["Gold"],
            "Prediction": row["Prediction"],
            "Mistral": decoded_out,
        }

    prefix = in_path.split(".")[0]
    file_name = in_path.split("/")[-1]
    split = file_name.split("_")
    res_dir = os.path.join(PROJECT_DIR, "results")
    res_dir = os.path.join(res_dir, split[1])
    res_dir = os.path.join(res_dir, split[0])

    res_path = os.path.join(res_dir, f"{prefix}_prompting.csv")

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    results_df.to_csv(res_path, index=False, sep="|")
    print(f"Results saved to {res_path}")


def main():
    args = parse_args()
    ICL_prompting(args.guided)


if __name__ == "__main__":
    main()
