def calculate_statistics(scores, icl):
    pass


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
