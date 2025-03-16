import pandas as pd
from pathlib import Path
import os

HOME = Path.home()
PROJECT_DIR = os.path.join(HOME, "Data-Comtamination-Sem-3-project")


def calculate_statistics(scores_path, icl_path):
    num_resamples_list = [10000, 50000, 100000]
    num_samples_list = [10, 100, 1000]
    
    for num_resamples, num_samples in zip(num_resamples_list, num_samples_list):
        print(f'case: num_samples = {num_samples} and num_resamples = {num_resamples}')
        with open(scores_path, "r") as f:
            scores = pd.read_csv(f, delimiter="|")

        with open(icl_path, "r") as f:
            icl = pd.read_csv(f, delimiter="|")

        res_dir = os.path.join(PROJECT_DIR, "results")
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Calculate and save some descriptive statistics about the scores and ICL
        res_path = os.path.join(res_dir, f"{scores_path.split(".")[0]}_stats.txt")
        scores.describe().to_csv(res_path)
        res_path = os.path.join(res_dir, f"{scores_path.split(".")[0]}_stats.txt")
        icl.describe().to_csv(res_path)

        # do the resampling from a dataframe that contains num_samples of instances
        sample_df = scores.sample(n=num_samples, seed=42)

        # Calculate the p-value for BLEURT and ROUGE-L
        p_val_bleu = calculate_p_value(
            sample_df, num_resample=num_resamples, num_samples=num_samples, guided="BLEURT guided", unguided="BLEURT unguided"
        )
        print(f"The p-value is {p_val_bleu}")

        p_val_rouge = calculate_p_value(
            sample_df, num_resample=num_resamples, num_samples=num_samples, guided="ROUGEL guided", unguided="ROUGEL unguided"
        )
        print(f"The p-value is {p_val_rouge}")

        res_path = os.path.join(res_dir, f"{scores_path.split(".")[0]}_p_values_{num_resamples}_{num_samples}.txt")
        with open(res_path, "w") as f:
            f.write(f"Results of bootstrapping\n")
            f.write(f"Number of resamples: {num_resamples}, number of samples: {num_samples}")
            f.write(
                f"BLEURT p-value, {p_val_bleu} \t {'Significant' if p_val_bleu <= 0.05 else 'Not Significant'}\n"
            )
            f.write(
                f"ROUGE-L p-value, {p_val_rouge} \t {'Significant' if p_val_rouge <= 0.05 else 'Not Significant'}\n"
            )


def resample_scores(scores, num_resample, num_samples):
    means = []
    for _ in range(num_resample):
        sample = scores.sample(n=num_samples, replace=True)
        means.append(sample.mean())

    return means


def calculate_p_value(scores, num_resample, num_samples, guided, unguided):
    guided_means = resample_scores(scores[guided], num_resample=num_resample, num_samples=num_samples)
    unguided_means = resample_scores(scores[unguided], num_resample=num_resample, num_samples=num_samples)

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
    calculate_statistics(args.diffs, args.icl)


if __name__ == "__main__":
    main()
