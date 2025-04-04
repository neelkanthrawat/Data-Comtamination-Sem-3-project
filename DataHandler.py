import pandas as pd
import json
import re
import random
import os
import json
from datasets import load_dataset
import numpy as np


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
                trust_remote_code=True,
            )
            dataset = dataset.filter(
                lambda example: json.loads(example["meta"].replace('"', "'")).get(
                    "language", ""
                )
                == "en"
            )
        elif dataset_name == "stackexchange":
            dataset = load_dataset(
                "togethercomputer/RedPajama-Data-1T",
                name="stackexchange",
                streaming=True,
                trust_remote_code=True,
                split="train",
            )
        elif dataset_name == "agnews":
            dataset = load_dataset(
                "fancyzhx/ag_news", split="test", streaming=True, trust_remote_code=True
            )
        elif dataset_name == "imdb":
            dataset = load_dataset(
                "stanfordnlp/imdb", split="test", streaming=True, trust_remote_code=True
            )
        else:
            print(f"Dataset {dataset_name} was not found.")
            return None
        return dataset

    def split_sentence(self, sentence, split_with_char=False):
        """
        Function to split a sentence, prioritizing "." (not at end), "than", then other punctuation
        """
        np.random.seed(42)

        if split_with_char:
            # Prioritize "." but only if it's not at the end
            match = re.search(
                r"\.(?!$)", sentence
            )  # Ensures the period is not at the end
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
            x = 2
        else:
            x = np.random.randint(
                2,
                5,
            )
        # If no punctuation or "than", split in half at a word boundary
        mid = len(sentence) // x
        while mid > 0 and sentence[mid] != " ":
            mid -= 1
        if mid == 0:
            mid = len(sentence) // x
            while mid < len(sentence) and sentence[mid] != " ":
                mid += 1
        return sentence[:mid].strip(), sentence[mid:].strip()
