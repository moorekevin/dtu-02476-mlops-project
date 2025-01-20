from data import MentalDisordersDataset
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, random_split
import torch
import hydra
import logging
log = logging.getLogger(__name__)


class MentalDisordersDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.data_percentage = cfg.data_percentage
        self.seed = cfg.seed
        self.training_data_path = cfg.training_data_path
        self.testing_data_path = cfg.testing_data_path

    def setup(self, stage: str):
        seed_everything(self.seed)
        if stage == "fit":
            trainFullDataset = MentalDisordersDataset(data_percentage=self.data_percentage, seed=self.seed,
                                                      train=True, training_data_path=self.training_data_path, testing_data_path=self.testing_data_path)

            # Calculate split sizes based on the sampled dataset
            train_size = int(0.9 * len(trainFullDataset))
            val_size = len(trainFullDataset) - train_size
            print(f"Sampled dataset length: {len(trainFullDataset)}")
            print(f"Train size: {train_size}, Validation size: {val_size}")

            # Perform the split
            self.train, self.val = random_split(
                trainFullDataset,
                [train_size, val_size],
                generator=torch.Generator()
            )

        if stage == "test":
            # Load and sample the test dataset (if needed)
            self.test = MentalDisordersDataset(data_percentage=self.data_percentage, seed=self.seed,
                                               train=False, training_data_path=self.training_data_path, testing_data_path=self.testing_data_path)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


@hydra.main(version_base="1.1", config_path="config", config_name="data.yaml")
def main(cfg):
    data_module = MentalDisordersDataModule(cfg)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    log.info(f"Train loader length: {len(train_loader)}")
    val_loader = data_module.val_dataloader()
    log.info(f"Val loader length: {len(val_loader)}")
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    log.info(f"Test loader length: {len(test_loader)}")


if __name__ == "__main__":
    main()
