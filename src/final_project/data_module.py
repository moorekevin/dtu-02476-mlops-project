from src.final_project.data import MentalDisordersDataset, OUTPUT_FOLDER
from pathlib import Path
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import torch


class MentalDisordersDataModule(LightningDataModule):
    def __init__(self, data_dir: Path = OUTPUT_FOLDER, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.test = MentalDisordersDataset(train=False, data_dir=self.data_dir)
        trainFullDataset = MentalDisordersDataset(
            train=True, data_dir=self.data_dir)
        self.train, self.val = random_split(
            trainFullDataset,
            [int(0.9 * len(trainFullDataset)),
             int(0.1 * len(trainFullDataset))],
            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


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
