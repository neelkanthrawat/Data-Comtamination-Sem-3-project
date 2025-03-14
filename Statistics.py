import pandas as pd
import Path
import os

HOME = Path.home()
PROJECT_DIR = os.path.join(HOME, "Data-Comtamination-Sem-3-project")


def calculate_statistics(scores, icl):
    with open(scores, "r") as f:
        scores = pd.read_csv(f)

    with open(icl, "r") as f:
        icl = pd.read_csv(f)

    res_dir = os.path.join(PROJECT_DIR, "results")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Calculate and save some descriptive statistics about the scores and ICL
    # res_path = os.path.join(res_dir, scores.split("."), "_statistics.csv")
    # scores.describe().to_csv(res_path)
    # res_path = os.path.join(res_dir, scores.split("."), "_statistics.csv")
    # icl.describe().to_csv(res_path)

    # Calculate the p-value for BLEURT and ROUGE-L
    p_val_bleu = calculate_p_value(
        scores, 1000, guided="BLEURT guided", unguided="BLEURT unguided"
    )
    print(f"The p-value is {p_val_bleu}")

    p_val_rouge = calculate_p_value(
        icl, 1000, guided="ROUGE-L guided", unguided="ROUGE-L unguided"
    )
    print(f"The p-value is {p_val_rouge}")

    res_path = os.path.join(res_dir, scores.split("."), "p_values.txt")
    with open(res_path, "w") as f:
        f.write(
            f"BLEURT p-value, {p_val_bleu} \t {'Significant' if p_val_bleu <= 0.05 else 'Not Significant'}\n"
        )
        f.write(
            f"ROUGE-L p-value, {p_val_rouge} \t {'Significant' if p_val_rouge <= 0.05 else 'Not Significant'}\n"
        )


def resample_scores(scores, num_resample):
    means = []
    for i in range(num_resample):
        sample = scores.sample(n=10, replace=True)
        means.append(sample.mean())

    return means


def calculate_p_value(
    scores, num_resample, guided="BLEURT guided", unguided="BLEURT unguided"
):
    guided_means = resample_scores(scores[guided], 1000)
    unguided_means = resample_scores(scores[unguided], 1000)

    count = 0

    for avg_guided, avg_unguided in zip(guided_means, unguided_means):
        if avg_guided > avg_unguided:
            count += 1

    p_val = 1 - (count / num_resample)

    return p_val


def parse_args():
    """
    Parse command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the model")

    parser.add_argument(
        "--diffs",
        type=str,
        help="Path to the file containing the differences in scores",
    )

    parser.add_argument(
        "--icl",
        type=str,
        help="Path to the file containing the results of the ICL",
    )

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()
    calculate_statistics(args.scores, args.icl)


if __name__ == "__main__":
    main()
