# evaluate.py
import logging
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import hydra

from data_module import MentalDisordersDataModule
from model import AwesomeModel

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base="1.1", config_path="config", config_name="eval.yaml")
def main(cfg):
    """
    Evaluate the trained model on the test set.

    This script assumes:
      - There is a 'models/model.pth' file containing the state dict
        of a trained AwesomeModel.
      - The data paths and other hyperparameters are provided via Hydra config (eval.yaml).
    """
    log.info("Starting evaluation...")

    # For reproducibility
    seed_everything(cfg.seed)

    # 1) Load the test data
    log.info("Initializing DataModule for test set.")
    dm = MentalDisordersDataModule(cfg)
    dm.setup("test")

    # 2) Initialize the model
    log.info("Initializing AwesomeModel from config.")
    model = AwesomeModel(cfg).to(DEVICE)

    # 3) Load model weights saved during training
    model_path = cfg.get("model_path", "models/model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find model weights at {model_path}. "
            f"Please ensure you've trained and saved the model."
        )
    log.info(f"Loading model state_dict from {model_path}")
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # 4) Setup the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=True,
        log_every_n_steps=10,
    )

    # 5) Run testing
    log.info("Running evaluation on test set...")
    test_results = trainer.test(model, datamodule=dm, verbose=True)
    log.info(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
