import pandas as pd
import transformers
import evaluate
import os


def parse_args():
    """
    Parse command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the model")

    parser.add_argument(
        "--pred",
        type=str,
        help="Path to the predictions file",
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

    bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="BLEURT-20")
    rouge = evaluate.load("rouge")

    for index, row in enumerate(pred_df.iterrows()):
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


def main():
    args = parse_args()
    calc_scores(args.pred)


if __name__ == "__main__":
    main()
