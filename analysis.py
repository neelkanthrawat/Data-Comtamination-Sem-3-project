import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns
import random


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
    return df[df[column] == df[column].max()]


def get_lowest_score(df: pd.DataFrame, column: str):
    """
    Get the lowest score from a DataFrame.
    """
    return df[df[column] == df[column].min()]


def print_min_max(df, metric, res_dir):
    """
    Print the minimum and maximum values of a metric in a DataFrame.
    """
    
    for type in ["guided", "unguided"]:
        res_path = os.path.join(res_dir, f"{metric}_{type}_min_max.txt")
        highest = get_highest_score(df, f"{metric} {type}")
        print(f"Highest {metric} {type} score: {highest}")
        lowest = get_lowest_score(df, f"{metric} {type}")
        print(f"Lowest {metric} {type} score: {lowest}")

        with open(res_path, "w") as f:
            f.write(f"Highest {metric} {type} score: {highest.to_string()} \n")
            f.write(f"Lowest {metric} {type} score: {lowest.to_string()} \n")

def resample_scores(scores, guided, unguided, num_resample, num_samples):
    unguided_means = []
    guided_means = []
    for _ in range(num_resample):
        sample_df = scores.sample(n=num_samples, replace=True, random_state=random.randint(0, 10000))
        unguided_means.append(sample_df[unguided].mean())
        guided_means.append(sample_df[guided].mean())

    return unguided_means, guided_means


def calculate_p_value(scores, num_resample, num_samples, guided, unguided):
    unguided_means, guided_means = resample_scores(
        scores, guided, unguided, num_resample=num_resample, num_samples=num_samples
    )

    count = 0

    for avg_guided, avg_unguided in zip(guided_means, unguided_means):
        if avg_guided > avg_unguided:
            count += 1

    p_val = 1 - (count / num_resample)

    return p_val

def calculate_correlation(df: pd.DataFrame, task: str, res_dir: str, write_in_file:bool = 1, return_scores:bool=0):
    """
    Calculate the correlation between BLEURT and ROUGEL.
    """
    res_path = os.path.join(res_dir, f"{task}_correlation.txt")
    unguided_corr = df["BLEURT unguided"].corr(df["ROUGEL unguided"], method="spearman")
    guided_corr = df["BLEURT guided"].corr(df["ROUGEL guided"], method="spearman")
    diff_corr = df["BLEURT_diff"].corr(df["ROUGEL_diff"], method="spearman")
    print(f"Unguided correlation: {unguided_corr}")
    print(f"Guided correlation: {guided_corr}")
    print(f"Difference correlation: {diff_corr}")

    if write_in_file:
        with open(res_path, "w") as f:
            f.write(f"Correclation between BLEURT and ROUGEL unguided: {unguided_corr} \n")
            f.write(f"Correclation between BLEURT and ROUGEL guided: {guided_corr} \n")
            f.write(f"Correclation between BLEURT and ROUGEL differences: {diff_corr} \n")
    if return_scores:
        return unguided_corr, guided_corr, diff_corr


def plot_corr(df: pd.DataFrame, task: str, res_dir: str):
    """
    Plot the correlation of the dataframe.
    """
    res_path = os.path.join(res_dir, f"{task}_unguided_scores.png")
    unguided_corr = df["BLEURT unguided"].corr(df["ROUGEL unguided"], method="spearman")
    plt.figure()
    plt.scatter(df["BLEURT unguided"], df["ROUGEL unguided"], label="unguided")
    plt.xlabel("BLEURT unguided")
    plt.ylabel("ROUGEL unguided")
    plt.title(f"Spearman correlation: {unguided_corr}")
    plt.legend()
    plt.savefig(res_path)
    plt.close()

    res_path = os.path.join(res_dir, f"{task}_guided_scores.png")
    guided_corr = df["BLEURT guided"].corr(df["ROUGEL guided"], method="spearman")
    plt.figure()
    plt.scatter(df["BLEURT guided"], df["ROUGEL guided"], label="guided")
    plt.xlabel("BLEURT guided")
    plt.ylabel("ROUGEL guided")
    plt.title(f"Spearman correlation: {guided_corr}")
    plt.legend()
    plt.savefig(res_path)
    plt.close()

    res_path = os.path.join(res_dir, f"{task}_differences_scores.png")
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
    res_path = os.path.join(res_dir, f"{task}_{metric}_scores.png")
    plt.figure(figsize=(12, 6), dpi=200)
    plt.plot(df[f"{metric} guided"], "o", label=f"Guided {metric}")
    plt.plot(df[f"{metric} unguided"], "+", label=f"Unguided {metric}")
    plt.xlabel("Index")
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.savefig(res_path)
    plt.close()

def filter_and_count(df, col1, col2):
    """
    Filters rows in a DataFrame where values in col1 > col2.
    Returns: the filtered list and the count of such rows.
    """
    filtered_df = df[df[col1] > df[col2]]
    return filtered_df.to_dict(orient='records'), len(filtered_df)

def compute_correlations(df, task, num_samples_list, analysis_dir):
    """
    Computes and prints Spearman correlations for different sample sizes.
    """
    for num_samples in num_samples_list:
        if num_samples < 1000:
            mean_list_unguided, mean_list_guided = [], []
            for i in range(10):
                sample_df = df.sample(n=num_samples, random_state=i)
                mean_list_unguided.append(sample_df["BLEURT unguided"].corr(sample_df["ROUGEL unguided"], method="spearman"))
                mean_list_guided.append(sample_df["BLEURT guided"].corr(sample_df["ROUGEL guided"], method="spearman"))
            print(f'for task: {task} and num_samples: {num_samples}')
            print(f'unguided correlation: {np.mean(mean_list_unguided)} ± {np.std(mean_list_unguided)}')
            print(f'guided correlation: {np.mean(mean_list_guided)} ± {np.std(mean_list_guided)}')
        else:
            sample_df = df.sample(n=num_samples, random_state=42)
            print(f'for task: {task} and num_samples: {num_samples}')
            print(f'unguided correlation: {sample_df["BLEURT unguided"].corr(sample_df["ROUGEL unguided"], method="spearman")}')
            print(f'guided correlation: {sample_df["BLEURT guided"].corr(sample_df["ROUGEL guided"], method="spearman")}')

def parse_p_values(path_list: list, num_samples_list, num_resample_list):
    """
    Reads p-values from files and creates DataFrames for ROUGE-L and BLEURT.
    """
    df2_rouge = pd.DataFrame(index=num_samples_list, columns=num_resample_list, dtype=float)
    df2_bl = pd.DataFrame(index=num_samples_list, columns=num_resample_list, dtype=float)
    
    for path in path_list:
        num_samples = int(path.split("_")[-1].split(".")[0])
        num_resample = int(path.split("_")[-2])
        with open(path, "r") as file:
            content = file.read()

        matches = re.findall(r"([A-Z\-]+) p-value, ([\d\.e-]+)", content)
        p_values = {}
        for metric, value in matches:
            print(value)
            if "e" not in value:
                p_values[metric] = float(value)
            else:
                p_values[metric] = float(value)
        df2_rouge.at[num_samples, num_resample] = p_values['ROUGE-L']
        df2_bl.at[num_samples, num_resample] = p_values['BLEURT']
    
    return df2_rouge, df2_bl

def calculate_p_values(scores, num_resamples_list, num_samples_list, analysis_dir):
    rouge_df = pd.DataFrame(index=num_samples_list, columns=num_resamples_list, dtype=float)
    bleurt_df = pd.DataFrame(index=num_samples_list, columns=num_resamples_list, dtype=float)

    for num_resamples in num_resamples_list:
        for num_samples in num_samples_list:
            print(f"Calculating p-values for {num_resamples} resamples and {num_samples} samples")
            task = analysis_dir.split("/")[-2]
            model = analysis_dir.split("/")[-3]

            res_path = os.path.join(analysis_dir, f"{model}_{task}_p_values_{num_resamples}_{num_samples}.txt")

            # Calculate the p-value for BLEURT and ROUGE-L
            p_val_bleu = calculate_p_value(
                scores,
                num_resample=num_resamples,
                num_samples=num_samples,
                guided="BLEURT guided",
                unguided="BLEURT unguided",
            )

            p_val_rouge = calculate_p_value(
                scores,
                num_resample=num_resamples,
                num_samples=num_samples,
                guided="ROUGEL guided",
                unguided="ROUGEL unguided",
            )

            with open(res_path, "w") as f:
                f.write(f"Results of bootstrapping\n")
                f.write(
                    f"Number of resamples: {num_resamples}, number of samples: {num_samples}\n"
                )
                f.write(
                    f"BLEURT p-value, {p_val_bleu} \t {'Significant' if p_val_bleu <= 0.05 else 'Not Significant'}\n"
                )
                f.write(
                    f"ROUGE-L p-value, {p_val_rouge} \t {'Significant' if p_val_rouge <= 0.05 else 'Not Significant'}\n"
                )
            
            rouge_df.at[num_samples, num_resamples] = p_val_rouge
            bleurt_df.at[num_samples, num_resamples] = p_val_bleu

    return rouge_df, bleurt_df

def plot_p_values_heatmap(df, title, analysis_dir):
    """
    Plots a heatmap of p-values.
    """
    print(df)
    # Set up the figure size
    plt.figure(figsize=(8, 6))

    # Create the heatmap
    sns.heatmap(df, cmap="coolwarm", annot=True, linewidths=0.5, cbar=True, fmt=".5f")
    
    # Improve readability of axis labels
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Bootstrap resampling rounds")
    plt.yticks(rotation=0)
    plt.ylabel("Number of samples")

    # Set title
    plt.title(f"{title}")

    # Save the heatmap
    task = analysis_dir.split("/")[-2]
    model = analysis_dir.split("/")[-3]
    metric = title.split(" ")[0]

    save_path = os.path.join(analysis_dir, f"{model}_{task}_{metric}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    for result_dir in ["llama_results_3.2_3b", "openllama_results_7b_v2"]: #, "openllama_results_13b"]:
        for path, directories, files in os.walk(result_dir):
            if files != [] and not path.endswith("analysis"):
                analysis_dir = os.path.join(path, "analysis")
                verify_or_create_dir(analysis_dir)

                diff_file = ""
                p_val_files = []

                for file in files:
                    if file.endswith("_differences.csv"):
                        diff_file = file
                    if "differences_p_values" in file:
                        p_val_files.append(os.path.join(path, file))

                differences_path = os.path.join(path, diff_file)
                print(f"Reading {differences_path}")
                df = read_df(path=differences_path)
            
                guided_more_bl_list, num_guided_more_bl = filter_and_count(df, "BLEURT guided", "BLEURT unguided")
                guided_more_rouge_list, num_guided_more_rouge = filter_and_count(df, "ROUGEL guided", "ROUGEL unguided")
                
                num_samples_list = [10, 100]
                num_resamples_list = [10000, 50000, 100000]
                
                if df.shape[0] > 500:
                    num_samples_list.append(500)
                    if df.shape[0] > 1000:
                        num_samples_list.append(1000)
                    else:
                        num_samples_list.append(df.shape[0])
                else:
                    num_samples_list.append(df.shape[0])

                print('seeing what is in the data frame:')
                print(df.columns.tolist())
                
                task = file.split("_")[0]
                compute_correlations(df, task, num_samples_list, analysis_dir)
                print('_' * 100)

                # df2_rouge, df2_bl = parse_p_values(p_val_files, num_samples_list, num_resamples_list)
                df2_rouge, df2_bl = calculate_p_values(scores=df, num_resamples_list=num_resamples_list, num_samples_list=num_samples_list, analysis_dir=analysis_dir)
                
                print('ROUGE-L p-values:')
                print(df2_rouge)
                print('BLEURT p-values:')
                print(df2_bl)
                
                task = analysis_dir.split("/")[-2]
                model = analysis_dir.split("/")[-3]
                plot_p_values_heatmap(df2_bl, f"BLEURT p-values: {model}, {task}", analysis_dir)
                plot_p_values_heatmap(df2_rouge, f"ROUGE-L p-values: {model}, {task}", analysis_dir)

if __name__ == "__main__":
    main()
