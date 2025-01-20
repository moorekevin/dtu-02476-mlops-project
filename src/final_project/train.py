# train.py
import pytorch_lightning as pl
import torch
from data_module import MentalDisordersDataModule
from model import AwesomeModel
import hydra
from src.final_project.model import AwesomeModel
import logging
import os
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")


def train(model, batch_size: int = 32, num_workers: int = 4, epochs: int = 3, learning_rate: float = 1e-5, data_percentage: float = 1.0):
    log.info("Starting training")
    log.info(f"{learning_rate=}, {batch_size=}, {epochs=}")
    # Instantiate DataModule
    dm = MentalDisordersDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        data_percentage=data_percentage,
    )

    # Make sure to call prepare_data() + setup() to figure out #labels if needed
    dm.setup("fit")  # get train/validation split

    # We can refine total_training_steps if we want a scheduler
    # total_steps = (
    #     int(0.01 * len(dm.train_dataloader()) * epochs)
    # )
    # model.hparams.total_training_steps = total_steps

    # Create a trainer
    # Inside main() function after setting up the model
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=True,
        log_every_n_steps=10,
    )

    # Fit
    trainer.fit(model, dm)
    log.info("Training complete")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")


@hydra.main(version_base="1.1", config_path="config", config_name="train.yaml")
def main(cfg):
    model = AwesomeModel(cfg).to(DEVICE)
    model.train()
    train(model, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
          epochs=cfg.epochs, learning_rate=cfg.learning_rate, data_percentage=cfg.data_percentage)


if __name__ == "__main__":
    main()
