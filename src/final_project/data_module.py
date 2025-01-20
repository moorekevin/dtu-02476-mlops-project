from src.final_project.data import MentalDisordersDataset, OUTPUT_FOLDER
from pathlib import Path
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
import torch
import random


class MentalDisordersDataModule(LightningDataModule):
    def __init__(self, data_dir: Path = OUTPUT_FOLDER, batch_size: int = 32, num_workers: int = 4, data_percentage: float = 1.0):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_percentage = data_percentage

    def sample_dataset(self, dataset):
        """Samples a subset of the dataset based on the specified percentage."""
        if self.data_percentage < 1.0:
            subset_size = int(len(dataset) * self.data_percentage)
            indices = random.sample(range(len(dataset)), subset_size)
            return Subset(dataset, indices)
        return dataset

    def setup(self, stage: str):
        # trainFullDataset = MentalDisordersDataset(
        #     train=True, data_dir=self.data_dir)
        # # Sample the dataset if a smaller percentage is specified
        # trainFullDataset = self.sample_dataset(trainFullDataset)

        # self.train, self.val = random_split(
        #     trainFullDataset,
        #     [int(0.9 * len(trainFullDataset)),
        #      int(0.1 * len(trainFullDataset))],
        #     generator=torch.Generator().manual_seed(42))

        # self.test = MentalDisordersDataset(
        #     train=False, data_dir=self.data_dir)
        # Load the full training dataset
        trainFullDataset = MentalDisordersDataset(
            train=True, data_dir=self.data_dir)

        # Apply sampling if data_percentage is less than 1.0
        if self.data_percentage < 1.0:
            subset_size = int(len(trainFullDataset) * self.data_percentage)
            indices = random.sample(range(len(trainFullDataset)), subset_size)
            trainFullDataset = Subset(trainFullDataset, indices)

        # Calculate split sizes based on the sampled dataset
        train_size = int(0.9 * len(trainFullDataset))
        val_size = len(trainFullDataset) - train_size
        print(f"Sampled dataset length: {len(trainFullDataset)}")
        print(f"Train size: {train_size}, Validation size: {val_size}")

        # Perform the split
        self.train, self.val = random_split(
            trainFullDataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Load and sample the test dataset (if needed)
        self.test = MentalDisordersDataset(train=False, data_dir=self.data_dir)
        if self.data_percentage < 1.0:
            subset_size = int(len(self.test) * self.data_percentage)
            indices = random.sample(range(len(self.test)), subset_size)
            self.test = Subset(self.test, indices)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    data_module = MentalDisordersDataModule()
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print("DataModule setup complete.")
    print("Train loader length: ", len(train_loader))
    print("Val loader length: ", len(val_loader))
    print("Test loader length: ", len(test_loader))
    # print("Example batch: ", next(iter(train_loader)))
    # print("Example batch: ", next(iter(val_loader)))
    # print("Example batch: ", next(iter(test_loader)))
