import pandas as pd
import transformers
import evaluate
import os
import transformers
import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """
    Parse command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the model")

    parser.add_argument(
        "--guided",
        type=str,
        help="Path to the guided predictions file",
    )

    parser.add_argument(
        "--unguided",
        type=str,
        help="Path to the unguided predictions file",
    )

    args = parser.parse_args()
    print(args)

    return args


def calc_scores(path: str):
    """
    Calculate the BLEURT and ROUGEL score for the predictions.
    """

    with open(path, "r") as f:
        pred_df = pd.read_csv(f)

    results_df = pd.DataFrame(
        columns=[
            "Index",
            "Label",
            "First piece",
            "Gold",
            "Prediction",
            "BLEURT",
            "ROUGEL",
        ]
    )

    bleurt = evaluate.load(
        "bleurt", module_type="metric", checkpoint="bleurt-large-512"
    )
    rouge = evaluate.load("rouge")

    for index, row in pred_df.iterrows():
        label = row["Label"]
        first_piece = row["First piece"]
        gold = row["Gold"]
        prediction = row["Prediction"]

        bleurt_score = bleurt.compute(predictions=[prediction], references=[gold])
        rouge_score = rouge.compute(predictions=[prediction], references=[gold])

        results_df.loc[index] = {
            "Index": index,
            "Label": row["label"] if "label" in row.columns() else None,
            "First piece": first_piece,
            "Gold": gold,
            "Prediction": prediction,
            "BLEURT": bleurt_score["scores"][0],
            "ROUGEL": rouge_score["rougeL"],
        }

    res_path = os.path.join("results", f"{path}_scores.csv")

    if not os.path.exists("results"):
        os.makedirs("results")

    results_df.to_csv(res_path, index=False, sep=";")
    print(f"Results saved to {res_path}")

    return results_df, res_path


def calc_differences(results_df_guided, results_df_unguided):
    if type(results_df_guided) == str:
        with open(results_df_guided, "r") as f:
            results_df_guided = pd.read_csv(f, sep=";")
    if type(results_df_unguided) == str:
        with open(results_df_unguided, "r") as f:
            results_df_unguided = pd.read_csv(f, sep=";")

    diff_df = pd.DataFrame(
        columns=[
            "Index",
            "BLEURT guided",
            "BLEURT unguided",
            "BLEURT_diff",
            "ROUGEL guided",
            "ROUGEL unguided",
            "ROUGEL_diff",
        ]
    )

    diff_df["BLEURT_diff"] = results_df_guided["BLEURT"] - results_df_unguided["BLEURT"]
    diff_df["ROUGEL_diff"] = results_df_guided["ROUGEL"] - results_df_unguided["ROUGEL"]

    diff_df["Index"] = results_df_guided["Index"]
    diff_df["BLEURT guided"] = results_df_guided["BLEURT"]
    diff_df["BLEURT unguided"] = results_df_unguided["BLEURT"]
    diff_df["ROUGEL guided"] = results_df_guided["ROUGEL"]
    diff_df["ROUGEL unguided"] = results_df_unguided["ROUGEL"]

    res_path = os.path.join("results", f"differences.csv")
    diff_df.to_csv(res_path, index=False, sep=";")
    print(f"Differences saved to {res_path}")


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


def ICL_prompting(path: str, args):
    """
    Only necessary for guided.
    """
    with open(path, "r") as f:
        df = pd.read_csv(f)

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

    res_path = os.path.join("results", f"{args.task}_{args.model}_{args.type}.csv")

    if not os.path.exists("results"):
        os.makedirs("results")

    results_df.to_csv(res_path, index=False, sep=";")
    print(f"Results saved to {res_path}")


def main():
    args = parse_args()
    results_df_guided, res_path_guided = calc_scores(args.guided)
    results_df_unguided, res_path_unguided = calc_scores(args.unguided)
    calc_differences(results_df_guided, results_df_unguided)


if __name__ == "__main__":
    main()
