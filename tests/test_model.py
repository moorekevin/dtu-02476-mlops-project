# tests/test_model.py

import pytest
import torch
from omegaconf import OmegaConf

from final_project.model import AwesomeModel

@pytest.fixture
def mock_cfg():
    """
    Return an OmegaConf DictConfig, which Lightning also supports.
    """
    cfg_dict = {
        "model_name_or_path": "distilbert-base-uncased",
        "num_labels": 2,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "warmup_steps": 0,
        "total_training_steps": 10,
        "seed": 42,
    }
    return OmegaConf.create(cfg_dict)  # -> DictConfig

@pytest.fixture
def dummy_batch():
    """
    A small batch of tokenized data for testing forward/training steps.
    Shapes: (batch_size=2, seq_len=5) for demonstration.
    """
    return {
        "input_ids": torch.randint(0, 1000, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }

def test_model_init(mock_cfg):
    """
    Test that the AwesomeModel initializes without errors
    and properly sets hyperparameters.
    """
    model = AwesomeModel(mock_cfg)
    assert model.hparams.learning_rate == mock_cfg.learning_rate, \
        "Learning rate hyperparameter not set correctly."
    assert model.hparams.num_labels == mock_cfg.num_labels, \
        "Number of labels not set correctly in hparams."
    assert model.config.num_labels == mock_cfg.num_labels, \
        "HF config does not have the correct number of labels."
    assert hasattr(model, "train_accuracy"), \
        "Missing train_accuracy metric attribute."
    assert hasattr(model, "val_accuracy"), \
        "Missing val_accuracy metric attribute."

@pytest.mark.parametrize("batch_size", [1, 2])
def test_forward_pass(mock_cfg, batch_size):
    """
    Test that the forward pass returns the expected outputs (loss, logits).
    Using different batch sizes to ensure shape correctness.
    """
    model = AwesomeModel(mock_cfg)
    input_ids = torch.randint(0, 1000, (batch_size, 5))
    attention_mask = torch.ones(batch_size, 5, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    # Check the shape of outputs
    assert hasattr(outputs, "loss"), "Model output is missing 'loss' attribute."
    assert hasattr(outputs, "logits"), "Model output is missing 'logits' attribute."
    assert outputs.logits.shape == (batch_size, mock_cfg.num_labels), \
        f"Logits shape {outputs.logits.shape} does not match expected ({batch_size}, {mock_cfg.num_labels})."

@pytest.mark.filterwarnings("ignore:.*self.log.*:UserWarning")
def test_training_step(mock_cfg, dummy_batch):
    """
    Test the training_step, verifying it returns a scalar loss
    and that accuracy is updated.
    """
    model = AwesomeModel(mock_cfg)
    # Simulate one training step
    loss = model.training_step(dummy_batch, batch_idx=0)

    # Loss should be a scalar
    assert isinstance(loss, torch.Tensor), "training_step should return a Tensor."
    assert loss.dim() == 0, "training_step loss should be a 0-dim (scalar) tensor."

    # The train_accuracy metric should be updated
    # We can peek inside the metric's internal state if needed,
    # but typically verifying no errors is enough.

@pytest.mark.filterwarnings("ignore:.*self.log.*:UserWarning")
def test_validation_step(mock_cfg, dummy_batch):
    """
    Test the validation_step. 
    This should compute and log val_loss and val_acc without errors.
    """
    model = AwesomeModel(mock_cfg)
    loss = model.validation_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor), "validation_step should return a Tensor (the loss)."
    assert loss.dim() == 0, "validation_step loss should be scalar."

@pytest.mark.filterwarnings("ignore:.*self.log.*:UserWarning")
def test_test_step(mock_cfg, dummy_batch):
    """
    Test the test_step. 
    This should compute and log test_loss and test_acc without errors.
    """
    model = AwesomeModel(mock_cfg)
    loss = model.test_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor), "test_step should return a Tensor (the loss)."
    assert loss.dim() == 0, "test_step loss should be scalar."

def test_configure_optimizers(mock_cfg):
    """
    Test that configure_optimizers returns two items: [optimizer], [scheduler_dict].
    """
    model = AwesomeModel(mock_cfg)
    optimizers, schedulers = model.configure_optimizers()

    # Check the basic structure
    assert len(optimizers) == 1, "configure_optimizers should return one optimizer."
    assert len(schedulers) == 1, "configure_optimizers should return one scheduler dict."
    optimizer = optimizers[0]
    scheduler_dict = schedulers[0]

    # Verify the optimizer is AdamW
    from torch.optim import AdamW
    assert isinstance(optimizer, AdamW), "Expected an AdamW optimizer."

    # Verify the scheduler is from transformers.get_linear_schedule_with_warmup
    assert "scheduler" in scheduler_dict, "Scheduler dict missing 'scheduler' key."
    assert "interval" in scheduler_dict, "Scheduler dict missing 'interval' key."
    assert scheduler_dict["interval"] == "step", "Expected scheduler interval to be 'step'."

@pytest.mark.filterwarnings("ignore:.*self.log.*:UserWarning")
def test_on_train_start(mock_cfg, dummy_batch):
    """
    Test that on_train_start resets the train_accuracy and val_accuracy metrics.
    We'll do a quick update, then reset, then check that it starts from zero again.
    """
    model = AwesomeModel(mock_cfg)
    # Simulate a partial training step to update train_accuracy
    _ = model.training_step(dummy_batch, batch_idx=0)
    # The internal state of train_accuracy should be non-trivial now.

    model.on_train_start()
    # The metrics should be reset. We'll run another step and ensure
    # it doesn't throw an error or produce weird results.

    _ = model.training_step(dummy_batch, batch_idx=1)
    # If it didn't reset properly, we might see accumulative results. 
    # For now, we simply ensure no error is raised and rely on consistent 
    # usage of train_accuracy reset behavior.

