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
        Reference Text: Systems Inc. said it has agreed to acquire P-Cube for \$200 million in stock and cash to enable service providers to further control and manage such advanced Internet Protocol services
        Candidate Text: Systems Inc. said it will acquire P-Cube Inc., a provider of IP multimedia subsystem (IMS) software, for \$200 million in cash. The acquisition will enable Cisco to offer IMS-based multimedia services to service providers, the company said. IMS is a next-generation IP-based network architecture that will enable service providers to offer multimedia services such as voice, video and data over a single network. Cisco said it will use P-Cube's IMS software to offer a complete IMS solution to service providers.
        Answer: Yes (near-exact match) 
        
        Example 4:  
        Reference Text: Shares of Google Inc. will make their  Nasdaq stock market debut on Thursday after the year's most  anticipated initial public offering priced far below initial  estimates, raising \$1.67 billion.
        Candidate Text: Google Inc (GOOG.O) will begin trading on the Nasdaq Stock Market on Thursday, the company said on Wednesday, as the Internet search leader prepares to go public. The Mountain View, California-based company, which is expected to raise as much as $2.7 billion in the offering, will trade under the symbol ""GOOG."" The stock will begin trading at 9 a.m. EDT (1300 GMT) on the Nasdaq Global Select Market, the company said in a statement
        Answer: Yes (near-exact match)
        
        Example 5:  
        Reference Text: leading US-listed Chinese internet portal, fell more than 10 per cent on Friday after China's biggest mobile phone network operator imposed a one-year suspension on its multimedia messaging services because of customers being sent spam.
        Candidate Text: hinese internet portal, fell more than 10 per cent on Monday after the company said it had been suspended from Yahoo! Mail and Hotmail for sending spam. The shares fell 11.4 per cent to $2.74 in New York, after closing at $3.09 on Friday. The company said it had been suspended from the two email services after it sent out a message to its users on Friday, which was not authorised by the company. The company said it was working with Yahoo
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
