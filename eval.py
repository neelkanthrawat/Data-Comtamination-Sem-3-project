import pandas as pd
import transformers


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

    parser.add_argument(
        "--gold",
        type=str,
        help="Path to the golden file",
    )

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    with open(args.pred, "r") as f:
        pred_df = pd.read_csv(f)

    # TODO: implement BLEURT and ROUGE-L


if __name__ == "__main__":
    main()
