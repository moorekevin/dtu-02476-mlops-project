import pytest
from unittest.mock import patch, MagicMock
import torch

from omegaconf import OmegaConf

# Import the train function and main entry point
from final_project.train import train, main as train_main
# For creating a mock model
from final_project.model import AwesomeModel

@pytest.fixture
def mock_cfg(tmp_path):
    """Minimal config as a DictConfig that Hydra/Lightning can accept."""
    cfg_dict = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs": 1,
        "num_workers": 0,
        "data_percentage": 1.0,
        "seed": 42,
        # Paths that won't actually be used if we mock the data module
        "training_data_path": str(tmp_path / "train_data.pt"),
        "testing_data_path": str(tmp_path / "test_data.pt"),
        "model_name_or_path": "distilbert-base-uncased",
        "num_labels": 2,
        "adam_epsilon": 1e-8,
        "warmup_steps": 0,
        "total_training_steps": 10,
    }
    return OmegaConf.create(cfg_dict)

@patch("final_project.train.MentalDisordersDataModule")
@patch("final_project.train.pl.Trainer")
@patch("final_project.train.torch.save")
def test_train_function(mock_torch_save, mock_trainer_cls, mock_data_module_cls, mock_cfg):
    """
    Test the train function to ensure:
      - DataModule is instantiated with the correct cfg
      - trainer.fit is called
      - model is saved to 'models/model.pth'
    """
    # 1) Mock Trainer
    mock_trainer = MagicMock()
    mock_trainer_cls.return_value = mock_trainer

    # 2) Mock DataModule
    mock_dm_instance = MagicMock()
    mock_data_module_cls.return_value = mock_dm_instance

    # 3) Create a mock model
    model = MagicMock()
    model.state_dict.return_value = {"weights": "some_dummy_weights"}

    # 4) Call the train function
    train(model, mock_cfg)

    # 5) Assertions
    mock_data_module_cls.assert_called_once_with(cfg=mock_cfg)
    mock_dm_instance.setup.assert_called_once_with("fit")

    mock_trainer_cls.assert_called_once_with(
        max_epochs=mock_cfg.epochs,
        accelerator="auto",
        devices="auto",
        logger=True,
        log_every_n_steps=10,
    )
    mock_trainer.fit.assert_called_once_with(model, mock_dm_instance)

    mock_torch_save.assert_called_once()
    # Check the arguments passed to torch.save:
    args, kwargs = mock_torch_save.call_args
    assert args[0] == {"weights": "some_dummy_weights"}
    assert args[1] == "models/model.pth"

@patch("final_project.train.train")
@patch("final_project.train.DEVICE", torch.device("cpu"))
def test_train_main(mock_train, mock_cfg):
    """
    Test the 'main' entry point to ensure it calls train(...) with 
    the appropriate arguments. We'll mock 'train' itself 
    to avoid running real training.
    """
    # We also mock the Hydra decorator usage by just calling main(cfg)
    # but typically you'd call main() with Hydra handles if needed.
    # For a pure unit test, let's just check that main calls 'train(...)'.

    # Call main as if we had Hydra compose a config
    train_main(mock_cfg)

    mock_train.assert_called_once()
    # We can check that the first argument was an AwesomeModel instance:
    args, kwargs = mock_train.call_args
    model_arg = args[0]
    passed_cfg = args[1]
    assert isinstance(model_arg, AwesomeModel), "Expected an AwesomeModel to be constructed."
    assert passed_cfg == mock_cfg, "Expected the same cfg to be passed to train()."
