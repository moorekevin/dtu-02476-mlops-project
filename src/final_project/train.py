import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from final_project import MentalDisordersDataModule
from final_project import AwesomeModel
import hydra
import logging
import os
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")


def train(model, cfg):
    wandb_logger = WandbLogger(
        project="dtu-02476-final-project",
        entity="moorekevin-",
        log_model=True
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,                       # Number of best checkpoints to keep
        monitor="val_loss",                 # Metric to monitor
        mode="min",
    )
    log.info("Starting training")
    log.info(f"{cfg.learning_rate=}, {cfg.batch_size=}, {cfg.epochs=}")

    # Instantiate DataModule
    dm = MentalDisordersDataModule(
        cfg=cfg
    )

    # Make sure to call prepare_data() + setup() to figure out #labels if needed
    dm.setup("fit")  # get train/validation split

    # Create a trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
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
    train(model, cfg)


if __name__ == "__main__":
    main()
