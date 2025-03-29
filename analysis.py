import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np


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
    
    for type in ["guided", "unguided"]:
        res_path = os.path.join(res_dir, f"{metric}_{type}_min_max.txt")
        highest = get_highest_score(df, f"{metric} {type}")
        print(f"Highest {metric} {type} score: {highest}")
        lowest = get_lowest_score(df, f"{metric} {type}")
        print(f"Lowest {metric} {type} score: {lowest}")

        with open(res_path, "w") as f:
            f.write(f"Highest {metric} {type} score: {highest.to_string()} \n")
            f.write(f"Lowest {metric} {type} score: {lowest.to_string()} \n")


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
    res_path = os.path.join(res_dir, f"{task}_guided_scores.png")
    unguided_corr = df["BLEURT unguided"].corr(df["ROUGEL unguided"], method="spearman")
    plt.figure()
    plt.scatter(df["BLEURT guided"], df["ROUGEL guided"], label="Guided")
    plt.xlabel("BLEURT guided")
    plt.ylabel("ROUGEL unguided")
    plt.title(f"Spearman correlation: {unguided_corr}")
    plt.legend()
    plt.savefig(res_path)
    plt.close()

    res_path = os.path.join(res_dir, f"{task}_unguided_scores.png")
    guided_corr = df["BLEURT guided"].corr(df["ROUGEL guided"], method="spearman")
    plt.figure()
    plt.scatter(df["BLEURT unguided"], df["ROUGEL unguided"], label="Unguided")
    plt.xlabel("BLEURT unguided")
    plt.ylabel("ROUGEL unguided")
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

def filter_and_count(df, col1, col2):
    """
    Filters rows in a DataFrame where values in col1 > col2.
    Returns: the filtered list and the count of such rows.
    """
    filtered_df = df[df[col1] > df[col2]]
    return filtered_df.to_dict(orient='records'), len(filtered_df)

def compute_correlations(df, task, num_samples_list):
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

def parse_p_values(base_dir, task, num_samples_list, num_resample_list):
    """
    Reads p-values from files and creates DataFrames for ROUGE-L and BLEURT.
    """
    df2_rouge = pd.DataFrame(index=num_samples_list, columns=num_resample_list)
    df2_bl = pd.DataFrame(index=num_samples_list, columns=num_resample_list)
    
    for num_samples in num_samples_list:
        for num_resample in num_resample_list:
            path_p_vals = os.path.join(base_dir, "Data-Comtamination-Sem-3-project/llama_results_3.2_3b/Llama", task, f"{task}_Llama_differences_p_values_{num_resample}_{num_samples}.txt")
            with open(path_p_vals, "r") as file:
                content = file.read()
            matches = re.findall(r"([A-Z\-]+) p-value, ([\d\.]+)", content)
            p_values = {metric: float(value) for metric, value in matches}
            df2_rouge.at[num_samples, num_resample] = p_values['ROUGE-L']
            df2_bl.at[num_samples, num_resample] = p_values['BLEURT']
    
    return df2_rouge, df2_bl

def plot_p_values_heatmap(df, title):
    """
    Plots a heatmap of p-values.
    """
    plt.imshow(df, cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns)
    plt.yticks(ticks=np.arange(len(df.index)), labels=df.index)
    plt.title(title)
    plt.show()


def main():
    verify_or_create_dir("results_llama")
    res_dir = "results_llama"

    for task in ["agnews", "imdb"]:
        # /home/neel/Desktop/results_llama
        #/home/neel/Desktop/Data-Comtamination-Sem-3-project/results_complete_Llama_25_mar/results/Llama/agnews/agnews_Llama_differences.csv
        path_correlation = os.path.join(
            f"/home/neel/Desktop/Data-Comtamination-Sem-3-project/llama_results_3.2_3b/Llama/{task}", f"{task}_Llama_differences.csv"
        )
        print(f"Reading {path_correlation}")
        df = read_df(path=path_correlation)
        num_samples_list = [10, 100, 1000, df.shape[0]]
        num_resample_list= [10000, 50000, 100000]
        print('seeing what is in the data frame:')
        headers = df.columns.tolist()
        print(headers)
        for num_samples in num_samples_list:
            
            task_num_samples = f"{task}_{num_samples}"
            if num_samples <1000: 
                mean_list_unguided, mean_list_guided=[],[]
                for i in range(0,10):
                    sample_df = df.sample(n=num_samples, random_state=i)
                    #bleurt scores
                    unguided_corr = sample_df["BLEURT unguided"].corr(sample_df["ROUGEL unguided"], method="spearman")
                    guided_corr = sample_df["BLEURT guided"].corr(sample_df["ROUGEL guided"], method="spearman")
                    mean_list_unguided.append(unguided_corr)
                    mean_list_guided.append(guided_corr)

                    print(f'for task: {task} and num_samples: {num_samples}')
                    print(f'unguided correlation is: {np.mean(mean_list_unguided)} +- {np.std(mean_list_unguided)}')
                    print(f'guided correlation is: {np.mean(mean_list_guided)} +- {np.std(mean_list_guided)}')
            else: 
                sample_df = df.sample(n=num_samples, random_state=42)
                #bleurt scores
                unguided_corr = sample_df["BLEURT unguided"].corr(sample_df["ROUGEL unguided"], method="spearman")
                guided_corr = sample_df["BLEURT guided"].corr(sample_df["ROUGEL guided"], method="spearman")

                print(f'for task: {task} and num_samples: {num_samples}')
                print(f'unguided correlation is: {unguided_corr}')
                print(f'guided correlation is: {guided_corr}')
                
        print('_'*100) 

        # row labels would be number of samples and colm labels would be number of resamplings
        df2_rouge= pd.DataFrame(index= num_samples_list, columns= num_resample_list)
        df2_bl= pd.DataFrame(index= num_samples_list, columns= num_resample_list)
        for num_samples in num_samples_list:
            for num_resample  in num_resample_list:
                # set the file name
                path_p_vals = os.path.join(
                    f"/home/neel/Desktop/Data-Comtamination-Sem-3-project/llama_results_3.2_3b/Llama/{task}", f"{task}_Llama_differences_p_values_{num_resample}_{num_samples}.txt"
                )
                # read the file
                with open(path_p_vals, "r") as file:
                    content = file.read()

                # get the p values for ROUGE-L and BLEURT model
                # Regular expression pattern to capture metric and p-value
                pattern = r"([A-Z\-]+) p-value, ([\d\.]+)"

                # Extract matches
                matches = re.findall(pattern, content)

                # Convert to dictionary
                p_values = {metric: float(value) for metric, value in matches}

                df2_rouge.at[num_samples, num_resample] = p_values['ROUGE-L']
                df2_bl.at[num_samples, num_resample] = p_values['BLEURT']

        print('rouge'); print(df2_rouge)
        print('BLEURT'); print(df2_bl)

        plt.imshow(df2_bl, cmap="coolwarm", aspect="auto")
        plt.colorbar()  # Add color scale
        plt.xticks(ticks=np.arange(len(df2_bl.columns)), labels=df2_bl.columns)
        plt.yticks(ticks=np.arange(len(df2_bl.index)), labels=df2_bl.index)
        plt.title("Matrix Representation of DataFrame")
        plt.show()

def main_p_vals():
    base_dir = os.path.expanduser("~")  # Get user's home directory
    verify_or_create_dir("results_llama")
    res_dir = "results_llama"
    
    for task in ["agnews", "imdb"]:
        path_correlation = os.path.join(base_dir, "Data-Comtamination-Sem-3-project/llama_results_3.2_3b/Llama", task, f"{task}_Llama_differences.csv")
        print(f"Reading {path_correlation}")
        df = read_df(path=path_correlation)
        
        guided_more_bl_list, num_guided_more_bl = filter_and_count(df, "BLEURT guided", "BLEURT unguided")
        guided_more_rouge_list, num_guided_more_rouge = filter_and_count(df, "ROUGEL guided", "ROUGEL unguided")
        
        num_samples_list = [10, 100, 1000, df.shape[0]]
        num_resample_list = [10000, 50000, 100000]
        
        print('seeing what is in the data frame:')
        print(df.columns.tolist())
        
        compute_correlations(df, task, num_samples_list)
        print('_' * 100)
        
        df2_rouge, df2_bl = parse_p_values(base_dir, task, num_samples_list, num_resample_list)
        
        print('ROUGE-L p-values:')
        print(df2_rouge)
        print('BLEURT p-values:')
        print(df2_bl)
        
        plot_p_values_heatmap(df2_bl, "BLEURT P-Values Heatmap")

if __name__ == "__main__":
    main()
    main_p_vals()
