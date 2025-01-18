from pytorch_lightning import LightningDataModule
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import json

import typer
from torch.utils.data import random_split, Dataset, DataLoader

RAW_DATA_PATH = Path("data/raw/mental_disorders_reddit.csv")
OUTPUT_FOLDER = Path("data/processed")
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
        return row["text"], row["label"]


def preprocess(raw_data_path: Path = RAW_DATA_PATH, output_folder: Path = OUTPUT_FOLDER) -> None:
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

    preprocessed_data.rename(
        columns={"subreddit": "label"}, inplace=True)

    # Calculate percentages for each unique value
    label_percentage = preprocessed_data['label'].value_counts(
        normalize=True) * 100
    label_percentage = label_percentage.round(2)
    print("Label percentages:")
    print(label_percentage)

    # Save the data
    # output_folder.mkdir(parents=True, exist_ok=True)
    # preprocessed_data.to_csv(
    #     output_folder / "processed_data.csv", index=False)

    df_train, df_test = train_test_split(
        preprocessed_data, test_size=0.1, random_state=42)

    # Convert to JSON Lines (one record per line)ue)
    df_train.to_csv(TRAINING_DATA_PATH, index=False)

    df_test.to_csv(TESTING_DATA_PATH, index=False)

    print(f"Saved train and test splits at {OUTPUT_FOLDER}")


if __name__ == "__main__":
    typer.run(preprocess)
