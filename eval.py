import pandas as pd
import evaluate
import os
from pathlib import Path

HOME = Path.home()
PROJECT_DIR = os.path.join(HOME, "Data-Comtamination-Sem-3-project")


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

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the evaluation",
    )

    args = parser.parse_args()
    print(args)

    return args


def calc_scores(in_path: str):
    """
    Calculate the BLEURT and ROUGEL score for the predictions.
    """
    path = os.path.join(in_path.split("/"))
    file_in = os.path.join(PROJECT_DIR, path)

    with open(file_in, "r") as f:
        pred_df = pd.read_csv(f, delimiter=';')
    
    print('_______I HAVE READ THE FILE________')

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

    res_path = os.path.join(PROJECT_DIR, "results", f"{in_path}_scores.csv")

    if not os.path.exists("results"):
        os.makedirs("results")

    results_df.to_csv(res_path, index=False, sep=";")
    print(f"Results saved to {res_path}")

    return results_df, res_path


def calc_differences(results_df_guided, results_df_unguided, eval_name=None):
    if type(results_df_guided) == str:
        # path = os.path.join(results_df_guided.split("/"))
        # file_in = os.path.join(PROJECT_DIR, path)
        file_in = os.path.normpath(os.path.join(PROJECT_DIR, results_df_guided))
        with open(file_in, "r") as f:
            results_df_guided = pd.read_csv(f, sep=";")

    if type(results_df_unguided) == str:
        # path = os.path.join(results_df_unguided.split("/"))
        # file_in = os.path.join(PROJECT_DIR, path)
        file_in = os.path.normpath(os.path.join(PROJECT_DIR, results_df_unguided))
        with open(file_in, "r") as f:
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

    if eval_name is None:
        res_path = os.path.join(PROJECT_DIR, "results", f"differences.csv")
    else:
        res_path = os.path.join(PROJECT_DIR, "results", f"{eval_name}_differences.csv")
    diff_df.to_csv(res_path, index=False, sep=";")
    print(f"Differences saved to {res_path}")


def main():
    args = parse_args()
    results_df_guided, res_path_guided = calc_scores(args.guided)
    results_df_unguided, res_path_unguided = calc_scores(args.unguided)
    calc_differences(results_df_guided, results_df_unguided, eval_name=args.name)
    print('_BKLA BLA BLA BLA BLAB BLAB')


if __name__ == "__main__":
    main()
