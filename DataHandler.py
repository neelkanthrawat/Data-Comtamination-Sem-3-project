import pandas as pd
import json
import re
import random
import os
import json


class DataHandler:
    """
    The DataHandler is responsible for everything related to loading and processing the data.
    """

    def __init__(self):
        """
        Initialize the DataHandler.
        """
        self.script_dir = os.getcwd()
        self.dataset_folder_path = os.path.join(self.script_dir, "datasets")

    def load_dataset(self, dataset_name):
        """
        Load the dataset.
        """
        print(f"Loading dataset {dataset_name}...")
        if dataset_name == "cb":
            cb_file = os.path.join(self.dataset_folder_path, "cb_sentences.csv")
            print(cb_file)
            df = pd.read_csv(cb_file)[["uID", "Embedding", "Context", "Target"]]
        elif dataset_name == "winogrande":
            wg_file = os.path.join(
                self.dataset_folder_path, "winogrande_val_splitup.jsonl"
            )
            df = pd.DataFrame(wg_file)
            df = df.rename(columns={'part1': 'Context', 'part2': 'Target'})
        else:
            print(f"Dataset {dataset_name} was not found.")
            return None
        return df

    def load_cb(self):
        """
        Load the CommitmentBank dataset.
        """
        cb_file = os.path.join(self.dataset_folder_path, "CommitmentBank-items.csv")
        df = pd.read_csv(cb_file)

        # Extract relevant columns
        df_context = df[["uID", "Context"]]
        df_prompt = df[["uID", "Prompt"]]
        df_target = df[["uID", "Target"]]

        # Save to separate CSV files
        df_context.to_csv("context_sentences.csv", index=False)
        df_prompt.to_csv("prompt_sentences.csv", index=False)
        df_target.to_csv("target_sentences.csv", index=False)

        df = pd.read_csv(cb_file)[["uID", "Context", "Prompt", "Target"]].to_csv(
            "full_dataset.csv", index=False
        )

        return df

    def handle_winogrande(self):
        """
        Process the Winogrande dataset.
        """
        if path is None:
            input_file = os.path.join(
                self.dataset_folder_path, "winogrande_validation.jsonl"
            )
            output_file = os.path.join(
                self.dataset_folder_path, "winogrande_val_splitup.jsonl"
            )
        else:
            input_file = path
            output_file = os.path.join(
                self.dataset_folder_path, "winogrande_val_splitup.jsonl"
            )

        dataset = self.load_winogrande(input_file)
        processed_data = self.process_dataset(dataset)
        self.save_dataset(processed_data, output_file)

    def load_winogrande(self, file_path):
        """
        Load the dataset (assuming JSONL format)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def save_dataset(self, data, output_file):
        """
        Save the processed dataset to a JSONL file.
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def split_sentence(self, sentence):
        """
        Function to split a sentence, prioritizing "." (not at end), "than", then other punctuation
        """
        # Prioritize "." but only if it's not at the end
        match = re.search(r"\.(?!$)", sentence)  # Ensures the period is not at the end
        if match:
            idx = match.start() + 1
            return sentence[:idx], sentence[idx:].strip()

        # If no ".", prioritize other punctuation (; , :)
        match = re.search(r"([,;:])", sentence)
        if match:
            idx = match.start() + 1  # Keep punctuation in part1
            return sentence[:idx], sentence[idx:].strip()

        # If "than" exists, split after the compared entity
        idx = sentence.find(" than ")
        if idx != -1:
            next_space = sentence.find(" ", idx + 6)  # Look after " than "
            if next_space != -1:
                return sentence[:next_space], sentence[next_space:].strip()

        # If no punctuation or "than", split in half at a word boundary
        mid = len(sentence) // 2
        while mid > 0 and sentence[mid] != " ":
            mid -= 1
        if mid == 0:
            mid = len(sentence) // 2
            while mid < len(sentence) and sentence[mid] != " ":
                mid += 1

        return sentence[:mid].strip(), sentence[mid:].strip()

    def process_dataset(self, dataset):
        """
        Process the dataset.
        """
        new_data = []
        for item in dataset:
            sentence = item["sentence"]
            part1, part2 = self.split_sentence(sentence)
            new_data.append(
                {"part1": part1, "part2": part2, **item}
            )  # Keep original fields
        return new_data
