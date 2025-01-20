from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from torch.optim import AdamW

from torchmetrics.classification import Accuracy
from pytorch_lightning import LightningModule, seed_everything, Trainer
from transformers import AutoModelForSequenceClassification
from omegaconf import OmegaConf
import hydra
import logging
import torch
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")


class AwesomeModel(LightningModule):
    """Transformer model for sequence classification."""

    def __init__(self, cfg) -> None:
        super(AwesomeModel, self).__init__()
        self.save_hyperparameters(cfg)

        # Load config with correct number of labels
        self.config = AutoConfig.from_pretrained(
            cfg.model_name_or_path, num_labels=cfg.num_labels
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name_or_path, config=self.config
        )
        # Metrics
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=cfg.num_labels)
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=cfg.num_labels)

        log.info(
            f"Initialized model and tokenizer from {cfg.model_name_or_path}")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, batch["labels"])

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, batch["labels"])

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Same as val, or adapt as needed
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, batch["labels"])
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Prepare optimizer and scheduler. If you know the total training steps,
        you can set up a linear schedule with warmup.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        # If you want a linear schedule over total steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_training_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_start(self):
        # Reset metrics
        self.train_accuracy.reset()
        self.val_accuracy.reset()


@hydra.main(version_base="1.1", config_path="config", config_name="model.yaml")
def main(cfg):
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    model = AwesomeModel(cfg).to(DEVICE)
    log.info(f"Model architecture: {model}")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Seed everything for reproducibility
    seed_everything(cfg.seed)

    # Set model to evaluation mode
    model.eval()

    # Sample sentence
    sample_text = "I have been feeling very anxious lately and can't sleep."

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    # Tokenize the sample sentence
    token = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Prepare inputs for the model
    input_ids = token["input_ids"].to(DEVICE)
    attention_mask = token["attention_mask"].to(DEVICE)
    label = torch.tensor([1], device=DEVICE)  # Dummy label tensor

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=label)

    # Print results
    print("Loss:", outputs.loss.item())
    print("Logits:", outputs.logits)

    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Get the predicted class (index of the highest probability)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    print("Predicted class: " + str(predicted_class))


if __name__ == "__main__":
    main()
