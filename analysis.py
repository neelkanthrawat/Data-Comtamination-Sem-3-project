import pandas as pd
import os
import matplotlib.pyplot as plt


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


def calculate_correlation(df: pd.DataFrame):
    """
    Calculate the correlation between BLEURT and ROUGEL.
    """
    unguided_corr = df["BLEURT unguided"].corr(df["ROUGEL unguided"])
    guided_corr = df["BLEURT guided"].corr(df["ROUGEL guided"])
    diff_corr = df["BLEURT_diff"].corr(df["ROUGEL_diff"])
    print(f"Unguided correlation: {unguided_corr}")
    print(f"Guided correlation: {guided_corr}")
    print(f"Difference correlation: {diff_corr}")


def plot_corr(df: pd.DataFrame, task: str):
    """
    Plot the correlation of the dataframe.
    """
    plt.figure()
    plt.scatter(df["BLEURT guided"], df["ROUGEL guided"], label="Guided")
    plt.legend()
    plt.savefig(f"{task}_guided_scores.png")
    plt.close()

    plt.figure()
    plt.scatter(df["BLEURT unguided"], df["ROUGEL unguided"], label="Unguided")
    plt.legend()
    plt.savefig(f"{task}_unguided_scores.png")
    plt.close()

    plt.figure()
    plt.scatter(df["BLEURT_diff"], df["ROUGEL_diff"], label="Difference")
    plt.legend()
    plt.savefig(f"{task}_difference_scores.png")
    plt.close()


def plot_scores(df: pd.DataFrame, metric: str, task: str):
    """
    Plot the scores of the dataframe.
    """
    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(df[f"{metric} guided"], label=f"Guided {metric}")
    plt.plot(df[f"{metric} unguided"], label=f"Unguided {metric}")
    plt.legend()
    plt.savefig(f"{task}_{metric}_scores.png")


def main():
    for task in ["ag_news", "imdb"]:
        path = os.path.join("results_llama", f"{task}_Llama_differences.csv")
        print(f"Reading {path}")
        df = read_df(path=path)
        print_min_max(df, "BLEURT")
        print_min_max(df, "ROUGEL")
        calculate_correlation(df)
        plot_corr(df, task=task)
        plot_scores(df, metric="BLEURT", task=task)
        plot_scores(df, metric="ROUGEL", task=task)


if __name__ == "__main__":
    main()
