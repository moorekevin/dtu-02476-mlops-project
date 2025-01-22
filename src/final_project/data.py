import torch
from pytorch_lightning import seed_everything
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import hydra
import logging
import random
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

log = logging.getLogger(__name__)


class MentalDisordersDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_percentage: float, seed: int, train: bool, training_data_path: str, testing_data_path: str) -> None:
        self.data_percentage = data_percentage
        self.seed = seed
        try:
            if train:
                self.data_path = Path(training_data_path).resolve()
                log.info(f"Loading training data from {self.data_path}")
            else:
                self.data_path = Path(testing_data_path).resolve()
                log.info(f"Loading testing data from {self.data_path}")

            data_dict = torch.load(self.data_path, weights_only=False)
            self.input_ids = data_dict["input_ids"]
            self.attention_masks = data_dict["attention_mask"]
            self.labels = data_dict["labels"]
            # Apply subset logic if data_percentage < 1.0
            if self.data_percentage < 1.0:
                self._sample_data()

        except FileNotFoundError:
            logging.error("File not found. Did you preprocess the data?")

    def _sample_data(self):
        """Samples a subset of the data based on the specified percentage."""
        subset_size = int(len(self.labels) * self.data_percentage)
        seed_everything(self.seed)
        indices = random.sample(range(len(self.labels)), subset_size)
        self.input_ids = [self.input_ids[i] for i in indices]
        self.attention_masks = [self.attention_masks[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index],
        }


def preprocess(raw_data_path: str, training_data_path: str, testing_data_path,  tokenizer_name: str = "distilbert-base-uncased", max_length: int = 512) -> None:
    log.info("Preprocessing data...")
    # raw_data = pd.read_csv(Path(raw_data_path).resolve())# Access the public GCS bucket without authentication
    raw_data = pd.read_csv(
        'gs://mlops-bucket-1999/data/raw/mental_disorders_reddit.csv',
        storage_options={'anon': True}
    )
    preprocessed_data = raw_data.copy()
    ##################
    # CLEANING LOGIC #
    ##################
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
    # Combine title and selftext into text column
    preprocessed_data["text"] = preprocessed_data["title"] + \
        "\n" + preprocessed_data["selftext"]
    # Map subreddit to label
    label_mapping = {label: idx for idx, label in enumerate(
        preprocessed_data["subreddit"].unique())}
    preprocessed_data["labels"] = preprocessed_data["subreddit"].map(
        label_mapping)
    
    logging.info(f"Old shape: {preprocessed_data.shape}")
    
    ##########################
    # UNDERSAMPLING LOGIC    #
    ##########################
    # 1. Group by labels
    grouped = preprocessed_data.groupby("labels")
    
    # 2. Find the smallest class size
    min_count = grouped.size().min()
    
    # 3. Undersample each group to that smallest size
    preprocessed_data = grouped.apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

    logging.info(f"New shape: {preprocessed_data.shape}")
    
    ######################
    # TOKENIZATION LOGIC #
    ######################

    # Tokenize text
    log.info("Tokenizing text...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    log.info("Tokenizer created")

    df_train, df_test = train_test_split(
        preprocessed_data, test_size=0.1, random_state=42)
    
    log.info("Data split completed")

    train_tokens = tokenizer(
        df_train["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    log.info("Trained data tokenized")

    train_dict = {
        "input_ids": train_tokens["input_ids"],
        "attention_mask": train_tokens["attention_mask"],
        "labels": torch.tensor(df_train["labels"].tolist(), dtype=torch.long),
    }

    test_tokens = tokenizer(
        df_test["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    log.info("Test data tokenized")

    test_dict = {
        "input_ids": test_tokens["input_ids"],
        "attention_mask": test_tokens["attention_mask"],
        "labels": torch.tensor(df_test["labels"].tolist(), dtype=torch.long),
    }

    log.info("Saving...")

    # Save both dictionaries with torch.save
    # torch.save(train_dict, Path(training_data_path).resolve())
    # torch.save(test_dict, Path(testing_data_path).resolve())
    torch.save(train_dict, 'gs://mlops-bucket-1999/data/processed/training_data.pt')
    torch.save(test_dict, 'gs://mlops-bucket-1999/data/processed/test_data.pt')


    log.info(
        f"Saved train and test splits at {training_data_path} and {testing_data_path}")
    log.info(f"Label mapping: {label_mapping}")


@hydra.main(version_base="1.1", config_path="config", config_name="data.yaml")
def main(cfg):
    preprocess(raw_data_path=cfg.raw_data_path, training_data_path=cfg.training_data_path,
               testing_data_path=cfg.testing_data_path, tokenizer_name=cfg.tokenizer_name, max_length=cfg.max_length)


if __name__ == "__main__":
    main()
