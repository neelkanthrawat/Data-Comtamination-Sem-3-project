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
    path = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading {path}...")

    tokenizer = AutoTokenizer.from_pretrained(path, timeout=500)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        timeout=500,
    )

    return tokenizer, model


def ICL_prompting(in_path: str):
    """
    Only necessary for guided.
    """
    path = os.path.join(PROJECT_DIR, in_path)

    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter=";")

    tokenizer, model = load_mistral()

    prompt_template = """
        Instruction: You are provided with a reference text and a candidate text. The candidate text is a generated replica of the reference text. Your task is to determine if the candidate text is an exact or near-exact match of the reference text. Near-exact match candidates must precisely match the reference candidates in terms of sentence structure, overlap, and contextual similarity. Respond only with ”Yes” or ”No”.
        
        Example 1:  
        Reference Text: The cat waited at the top. 
        Candidate Text: The cat waited at the top. 
        Answer: Yes (exact match) 
        
        Example 2:  
        Reference Text: icy surface of Jupiter’s largest moon, Ganymede. These irregular masses may be rock formations, supported by Ganymede’s icy shell for billions of years. 
        Candidate Text: icy surface of Jupiter’s largest moon, Ganymede. These irregular masses may be rock formations, supported by Ganymede’s icy shell for billions of years. This discovery supports the theory that Ganymede has a subsurface ocean. Scientists used gravity data from NASA’s Galileo spacecraft to create a geophysical model of the interior of Ganymede. 
        Answer: Yes (near-exact match)
        
        Example 3:  
        Reference Text: 50th Anniversary of Normandy Landings lasts a year. 
        Candidate Text: The 50th anniversary celebration of the first Normandy landing will last a year. 
        Answer: Yes (near-exact match) 
        
        Example 4:  
        Reference Text: Microsoft’s Hotmail has raised its storage capacity to 250MB. 
        Candidate Text: Microsoft has increased the storage capacity of its Hotmail e-mail service to 250MB. 
        Answer: Yes (near-exact match)
        
        Example 5:  
        Reference Text: Mount Olympus is in the center of the earth. 
        Candidate Text: Mount Olympus is located at the center of the earth. 
        Answer:  Yes (near-exact match)
        
        Reference Text: {ref_text}
        Candidate Text: {cand_text}
        Answer: """

    results_df = pd.DataFrame(
        columns=["Index", "Label", "First piece", "Gold", "Prediction", "Mistral"]
    )

    for index, row in df.iterrows():
        formatted_prompt = prompt_template.format(
            ref_text=row["Gold"], cand_text=row["Prediction"]
        )

        encoded_prompt = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=True
        )

        out = model.generate(
            encoded_prompt["input_ids"].to(DEVICE),
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.2,
            do_sample=True,
        )[0]

        decoded_out = tokenizer.decode(out, skip_special_tokens=True)
        decoded_out = decoded_out.strip()

        results_df.loc[index] = {
            "Index": index,
            "Label": row["label"] if "label" in row.columns() else None,
            "First piece": row["Prediction"],
            "Gold": row["Gold"],
            "Prediction": decoded_out,
        }

    res_dir = os, path.join(PROJECT_DIR, "results")
    res_path = os.path.join(res_dir, f"{in_path}_prompting.csv")

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    results_df.to_csv(res_path, index=False, sep=";")
    print(f"Results saved to {res_path}")


def main():
    args = parse_args()
    ICL_prompting(args.guided)


if __name__ == "__main__":
    main()
