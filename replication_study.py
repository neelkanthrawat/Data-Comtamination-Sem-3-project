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


def resample_scores(scores, num_resample, num_samples):
    means = []
    for _ in range(num_resample):
        sample = scores.sample(n=num_samples, replace=True)
        means.append(sample.mean())

    return means


def calculate_p_value(scores, num_resample, num_samples, guided, unguided):
    guided_means = resample_scores(
        scores[guided], num_resample=num_resample, num_samples=num_samples
    )
    unguided_means = resample_scores(
        scores[unguided], num_resample=num_resample, num_samples=num_samples
    )

    count = 0

    for avg_guided, avg_unguided in zip(guided_means, unguided_means):
        if avg_guided > avg_unguided:
            count += 1

    p_val = 1 - (count / num_resample)

    return p_val


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
            if "second_piece" in df.columns:
                second_pieces = df["second_piece"].tolist()
            else:
                second_pieces = df["sentence2"].tolist()

            bleurt_score_guided = calculate_bleurt(
                preds=guided_completions,
                refs=second_pieces,
            )

            bleurt_score_unguided = calculate_bleurt(
                preds=unguided_completions,
                refs=second_pieces,
            )

            rouges_guided = []
            for guided, ref in zip(guided_completions, second_pieces):
                rouge_score = calculate_rouge(preds=[guided], refs=[ref])
                rouges_guided.append(rouge_score["rougeL"])

            rouges_unguided = []
            for unguided, ref in zip(unguided_completions, second_pieces):
                rouge_score = calculate_rouge(preds=[unguided], refs=[ref])
                rouges_unguided.append(rouge_score["rougeL"])


            rep_path = os.path.join(PROJECT_DIR, "replication_results.txt")
            with open(rep_path, "a") as f:
                f.writelines(f"\n\n\nFile: {file_path}\n")
                f.writelines(
                    f"Our recalculated BLEURT Score Guided: {bleurt_score_guided},\n theirs: {df['bleurt_score_for_guided_completion'].tolist()}"
                )
                f.writelines(
                    f"Our recalculated ROUGE Score Guided: {rouges_guided},\n theirs: {df['rouge_score_for_guided_completion'].tolist()}"
                )
                f.writelines(
                    f"Our recalculated BLEURT Score Unguided: {bleurt_score_unguided},\n theirs: {df['bleurt_score_for_general_completion'].tolist()}"
                )
                f.writelines(
                    f"Our recalculated ROUGE Score Unguided: {rouges_unguided},\n theirs: {df['rouge_score_for_general_completion'].tolist()}"
                )


if __name__ == "__main__":
    main()
