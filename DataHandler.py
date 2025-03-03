import pandas as pd
import json
import re
import random
import os
import json
from datasets import load_dataset


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
            dataset = load_dataset(
                "super_glue", name="cb", split="test", trust_remote_code=1
            )
        elif dataset_name == "wsc":
            dataset = load_dataset(
                "super_glue", name="wsc", split="test", trust_remote_code=1
            )
        elif dataset_name == "wikipedia":
            dataset = load_dataset(
                "togethercomputer/RedPajama-Data-1T",
                name="wikipedia",
                split="train",
                streaming=True,
                trust_remote_code=1,
            )
            dataset = dataset.filter(
                lambda example: json.loads(example["meta"]).get("language", "") == "en"
            )
        else:
            print(f"Dataset {dataset_name} was not found.")
            return None
        return dataset

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
