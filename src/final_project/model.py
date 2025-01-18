from pytorch_lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from omegaconf import OmegaConf
import hydra
import logging
import torch
from torch import nn, optim
log = logging.getLogger(__name__)


class AwesomeModel(LightningModule):
    """Transformer model for sequence classification."""

    def __init__(self, cfg) -> None:
        super(AwesomeModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        log.info(
            f"Initialized model and tokenizer from {cfg.model_name_or_path}")

        self.criterium = nn.CrossEntropyLoss()
        self.lr = cfg.lr

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
