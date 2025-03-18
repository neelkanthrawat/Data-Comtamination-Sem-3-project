import pandas as pd
import os


def read_df(path: str):
    """
    Read a CSV file into a DataFrame.
    """
    with open(path, "r") as f:
        df = pd.read_csv(f, sep="|")
    return df


def get_highest_score(df: pd.DataFrame, column: str):
    """
    Get the highest score from a DataFrame.
    """
    return df[df[column] == df[column].min()]


def get_lowest_score(df: pd.DataFrame, column: str):
    """
    Get the lowest score from a DataFrame.
    """
    return df[df[column] == df[column].max()]


def print_min_max(df, metric):
    """
    Print the minimum and maximum values of a metric in a DataFrame.
    """
    for type in ["guided", "unguided"]:
        highest = get_highest_score(df, f"{metric} {type}")
        print(f"Highest {metric} {type} score: {highest}")
        lowest = get_lowest_score(df, f"{metric} {type}")
        print(f"Lowest {metric} {type} score: {lowest}")


def main():
    path = os.path.join("results_llama", "ag_news_Llama_differences.csv")
    print(f"Reading {path}")
    df = read_df(path=path)
    print_min_max(df, "BLEURT")
    print_min_max(df, "ROUGEL")

    path = os.path.join("results_llama", "imdb_Llama_differences.csv")
    print(f"Reading {path}")
    df = read_df(path=path)
    print_min_max(df, "BLEURT")
    print_min_max(df, "ROUGEL")


if __name__ == "__name__":
    main()
