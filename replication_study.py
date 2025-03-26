import pandas as pd
import os
import evaluate
from pathlib import Path

HOME = Path.home()
PROJECT_DIR = os.path.join(HOME, "Data-Comtamination-Sem-3-project")

def calculate_bleurt(preds, refs):
    bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="BLEURT-20")
    bleurt_score = bleurt.compute(predictions=preds, references=refs)
    return bleurt_score


def calculate_rouge(preds, refs):
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=preds, references=refs)
    return rouge_score


def main():
    results_dir = os.path.join(PROJECT_DIR, "time-travel-in-llms-main/results/")
    paths = []
    print(results_dir)
    for dirpath, _, filenames in os.walk(results_dir):
        for file in filenames:
            if file.endswith(".csv"):
                paths.append(os.path.join(dirpath, file))

    print(f"Found {len(paths)} files: {paths}")

    for file_path in paths:
        with open(file_path) as f:
            df = pd.read_csv(f)
            if "generated_guided_completion" in df.columns:
                guided_completions = df["generated_guided_completion"].tolist()
            if "generated_general_completion" in df.columns:
                unguided_completions = df["generated_general_completion"].tolist()
            if "first_piece" in df.columns:
                first_pieces = df["first_piece"].tolist()
            else:
                first_pieces = df["sentence1"].tolist()
            if "second_piece" in df.columns:
                second_pieces = df["second_piece"].tolist()
            else:
                second_pieces = df["sentence2"].tolist()

            bleurt_score_guided = calculate_bleurt(
                preds=guided_completions,
                refs=second_pieces,
            )
            rouge_score_guided = calculate_rouge(
                preds=guided_completions,
                refs=second_pieces,
            )

            bleurt_score_unguided = calculate_bleurt(
                preds=unguided_completions,
                refs=second_pieces,
            )
            bleurt_score_guided = calculate_rouge(
                preds=unguided_completions,
                refs=second_pieces,
            )

            print(f"File: {file_path}")
            print(
                f"Our recalculated BLEURT Score Guided: {bleurt_score_guided}, theirs: {df['bleurt_score_for_guided_completion'].tolist()}"
            )
            print(
                f"Our recalculated ROUGE Score Guided: {rouge_score_guided}, theirs: {df['rouge_score_for_guided_completion'].tolist()}"
            )
            print(
                f"Our recalculated BLEURT Score Unguided: {bleurt_score_unguided}, theirs: {df['bleurt_score_for_general_completion'].tolist()}"
            )
            print(
                f"Our recalculated ROUGE Score Unguided: {rouge_score_guided}, theirs: {df['rouge_score_for_general_completion'].tolist()}"
            )


if __name__ == "__main__":
    main()
