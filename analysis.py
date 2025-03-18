import pandas as pd
import os
import matplotlib.pyplot as plt


def verify_or_create_dir(path: str):
    """
    Verify if a directory exists, if not create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)


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


def print_min_max(df, metric, res_dir):
    """
    Print the minimum and maximum values of a metric in a DataFrame.
    """
    res_path = os.join(res_dir, f"{metric}_{type}_min_max.txt")
    for type in ["guided", "unguided"]:
        highest = get_highest_score(df, f"{metric} {type}")
        print(f"Highest {metric} {type} score: {highest}")
        lowest = get_lowest_score(df, f"{metric} {type}")
        print(f"Lowest {metric} {type} score: {lowest}")

        with open(res_path, "w") as f:
            f.write(f"Highest {metric} {type} score: {highest.to_string()} \n")
            f.write(f"Lowest {metric} {type} score: {lowest.to_string()} \n")


def calculate_correlation(df: pd.DataFrame, task: str, res_dir: str):
    """
    Calculate the correlation between BLEURT and ROUGEL.
    """
    res_path = os.join(res_dir, f"{task}_correlation.txt")
    unguided_corr = df["BLEURT unguided"].corr(df["ROUGEL unguided"], method="spearman")
    guided_corr = df["BLEURT guided"].corr(df["ROUGEL guided"], method="spearman")
    diff_corr = df["BLEURT_diff"].corr(df["ROUGEL_diff"], method="spearman")
    print(f"Unguided correlation: {unguided_corr}")
    print(f"Guided correlation: {guided_corr}")
    print(f"Difference correlation: {diff_corr}")

    with open(res_path, "w") as f:
        f.write(f"Correclation between BLEURT and ROUGEL unguided: {unguided_corr} \n")
        f.write(f"Correclation between BLEURT and ROUGEL guided: {guided_corr} \n")
        f.write(f"Correclation between BLEURT and ROUGEL differences: {diff_corr} \n")


def plot_corr(df: pd.DataFrame, task: str, res_dir: str):
    """
    Plot the correlation of the dataframe.
    """
    res_path = os.join(res_dir, f"{task}_guided_scores.png")
    unguided_corr = df["BLEURT unguided"].corr(df["ROUGEL unguided"], method="spearman")
    plt.figure()
    plt.scatter(df["BLEURT guided"], df["ROUGEL guided"], label="Guided")
    plt.xlabel("BLEURT guided")
    plt.ylabel("ROUGEL unguided")
    plt.title(f"Spearman correlation: {unguided_corr}")
    plt.legend()
    plt.savefig(res_path)
    plt.close()

    res_path = os.join(res_dir, f"{task}_unguided_scores.png")
    guided_corr = df["BLEURT guided"].corr(df["ROUGEL guided"], method="spearman")
    plt.figure()
    plt.scatter(df["BLEURT unguided"], df["ROUGEL unguided"], label="Unguided")
    plt.xlabel("BLEURT unguided")
    plt.ylabel("ROUGEL unguided")
    plt.title(f"Spearman correlation: {guided_corr}")
    plt.legend()
    plt.savefig(res_path)
    plt.close()

    res_path = os.join(res_dir, f"{task}_differences_scores.png")
    diff_corr = df["BLEURT_diff"].corr(df["ROUGEL_diff"], method="spearman")
    plt.figure()
    plt.scatter(df["BLEURT_diff"], df["ROUGEL_diff"], label="Difference")
    plt.xlabel("BLEURT_diff")
    plt.ylabel("ROUGEL_diff")
    plt.title(f"Spearman correlation: {diff_corr}")
    plt.legend()
    plt.savefig(res_path)
    plt.close()


def plot_scores(df: pd.DataFrame, metric: str, task: str, res_dir: str):
    """
    Plot the scores of the dataframe.
    """
    res_path = os.join(res_dir, f"{task}_{metric}_scores.png")
    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(df[f"{metric} guided"], label=f"Guided {metric}", linestyle="o")
    plt.plot(df[f"{metric} unguided"], label=f"Unguided {metric}", linestyle="+")
    plt.xlabel("Index")
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.savefig(res_path)


def main():
    verify_or_create_dir("results_llama")
    res_dir = "results_llama"

    for task in ["ag_news", "imdb"]:
        path = os.path.join("results_llama", f"{task}_Llama_differences.csv")
        print(f"Reading {path}")
        df = read_df(path=path)
        num_samples_list = [10, 100, 1000, df.shape[0]]
        for num_samples in num_samples_list:
            task_num_samples = f"{task}_{num_samples}"
            sample_df = df.sample(n=num_samples, random_state=42)
            print_min_max(df=sample_df, metric="BLEURT", res_dir=res_dir)
            print_min_max(df=sample_df, metric="ROUGEL", res_dir=res_dir)
            calculate_correlation(df=sample_df, task=task_num_samples, res_dir=res_dir)

            plot_corr(df=sample_df, task=task_num_samples, res_dir=res_dir)
            plot_scores(
                df=sample_df, metric="BLEURT", task=task_num_samples, res_dir=res_dir
            )
            plot_scores(
                df=sample_df, metric="ROUGEL", task=task_num_samples, res_dir=res_dir
            )


if __name__ == "__main__":
    main()
