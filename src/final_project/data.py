import torch
from pytorch_lightning import LightningDataModule
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from transformers import AutoTokenizer
import typer
from torch.utils.data import random_split, Dataset, DataLoader

RAW_DATA_PATH = Path("data/raw/mental_disorders_reddit.csv")
OUTPUT_FOLDER = Path("data/processed").resolve()
TRAINING_DATA_PATH = OUTPUT_FOLDER / "training_data.csv"
TESTING_DATA_PATH = OUTPUT_FOLDER / "testing_data.csv"


class MentalDisordersDataset(Dataset):
    """My custom dataset."""

    def __init__(self, train: bool, data_dir: Path = OUTPUT_FOLDER) -> None:
        self.data_dir = data_dir
        try:
            print("Loading data from ", self.data_dir)
            if train:
                self.data_path = TRAINING_DATA_PATH
            else:
                self.data_path = TESTING_DATA_PATH
            # Open and read the JSON file
            # with open(self.data_path, 'r') as file:
            #     self.df = json.load(file)
            self.df = pd.read_csv(self.data_path)

        except FileNotFoundError:
            print("File not found. Did you preprocess the data?")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        row = self.df.iloc[index]

        input_ids = torch.tensor(eval(row["input_ids"]), dtype=torch.long)
        attention_mask = torch.tensor(
            eval(row["attention_mask"]), dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


def preprocess(raw_data_path: Path = RAW_DATA_PATH, output_folder: Path = OUTPUT_FOLDER,  tokenizer_name: str = "bert-base-uncased", max_length: int = 512) -> None:
    print("Preprocessing data...")
    raw_data = pd.read_csv(raw_data_path)
    preprocessed_data = raw_data.copy()
    preprocessed_data.dropna(
        subset=["title", "selftext", "subreddit"], inplace=True)
    # Remove any rows where the selftext is [removed] or [deleted]
    preprocessed_data = preprocessed_data[preprocessed_data["selftext"] != "[removed]"]
    preprocessed_data = preprocessed_data[preprocessed_data["selftext"] != "[deleted]"]
    # Remove any rows from 'mentalillness' subreddit
    preprocessed_data = preprocessed_data[preprocessed_data["subreddit"]
                                          != "mentalillness"]
    # Remove low effort posts under 20 characters
    preprocessed_data = preprocessed_data[preprocessed_data["selftext"].apply(
        len) > 20]

    preprocessed_data["text"] = preprocessed_data["title"] + \
        "\n" + preprocessed_data["selftext"]

    # preprocessed_data.rename(
    #     columns={"subreddit": "label"}, inplace=True)

    # Map string labels to integers
    label_mapping = {label: idx for idx, label in enumerate(
        preprocessed_data["subreddit"].unique())}
    preprocessed_data["label"] = preprocessed_data["subreddit"].map(
        label_mapping)

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer(
        preprocessed_data["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Add tokenized data to DataFrame
    preprocessed_data["input_ids"] = tokens["input_ids"].tolist()
    preprocessed_data["attention_mask"] = tokens["attention_mask"].tolist()

    # Calculate percentages for each unique value
    label_percentage = preprocessed_data['label'].value_counts(
        normalize=True) * 100
    label_percentage = label_percentage.round(2)
    print("Label percentages:")
    print(label_percentage)

    # Save the data
    df_train, df_test = train_test_split(
        preprocessed_data, test_size=0.1, random_state=42)

    # Convert to csv
    with open(output_folder / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f)
    df_train.to_csv(TRAINING_DATA_PATH, index=False)

    df_test.to_csv(TESTING_DATA_PATH, index=False)

    print(f"Saved train and test splits at {OUTPUT_FOLDER}")
    print(f"Label mapping: {label_mapping}")


if __name__ == "__main__":
    typer.run(preprocess)
