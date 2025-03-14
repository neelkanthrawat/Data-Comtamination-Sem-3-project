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

    res_path = os.path.join(res_dir, scores.split("."), "_statistics.csv")
    scores.describe().to_csv(res_path)
    res_path = os.path.join(res_dir, scores.split("."), "_statistics.csv")
    icl.describe().to_csv(res_path)


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
