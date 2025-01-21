import pytest
import os
import torch
from unittest.mock import patch, MagicMock, ANY
from omegaconf import OmegaConf

from final_project.evaluate import main as eval_main

@pytest.fixture
def mock_eval_cfg(tmp_path):
    """Minimal config for evaluation that satisfies DataModule needs."""
    cfg_dict = {
        "seed": 42,
        "model_name_or_path": "distilbert-base-uncased",
        "num_labels": 2,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "warmup_steps": 0,
        "total_training_steps": 10,
        "batch_size": 4,
        "num_workers": 0,
        "data_percentage": 1.0,
        "training_data_path": str(tmp_path / "train_data.pt"),
        "testing_data_path": str(tmp_path / "test_data.pt"),
        "model_path": str(tmp_path / "models" / "model.pth"),
    }
    return OmegaConf.create(cfg_dict)

@patch("final_project.evaluate.DEVICE", torch.device("cpu"))
def test_evaluate_file_not_found(mock_eval_cfg):
    """
    Test that evaluate raises FileNotFoundError
    if the model file doesn't exist.
    """
    # model_path doesn't exist yet
    assert not os.path.exists(mock_eval_cfg.model_path)
    with pytest.raises(FileNotFoundError, match="Could not find model weights"):
        eval_main(mock_eval_cfg)

@patch("final_project.evaluate.AwesomeModel.load_state_dict", return_value=None)
@patch("final_project.evaluate.MentalDisordersDataModule")
@patch("final_project.evaluate.pl.Trainer")
@patch("final_project.evaluate.os.path.exists", return_value=True)
@patch("final_project.evaluate.torch.load")
@patch("final_project.evaluate.DEVICE", torch.device("cpu"))
def test_evaluate_main(
    mock_torch_load,
    mock_os_path_exists,
    mock_trainer_cls,
    mock_data_module_cls,
    mock_load_state_dict,
    mock_eval_cfg,
    tmp_path
):
    """
    Test the main evaluation flow if the model file exists.
    """
    # 1) "File" is found
    mock_os_path_exists.return_value = True
    # 2) Fake state_dict
    mock_torch_load.return_value = {"weights": "dummy_state_dict"}
    # 3) Mock Trainer
    mock_trainer = MagicMock()
    mock_trainer.test.return_value = [{"test_acc": 0.9, "test_loss": 0.2}]
    mock_trainer_cls.return_value = mock_trainer
    # 4) Mock DataModule
    mock_dm_instance = MagicMock()
    mock_data_module_cls.return_value = mock_dm_instance

    # 5) Run evaluation
    eval_main(mock_eval_cfg)

    # 6) Check calls
    mock_data_module_cls.assert_called_once_with(mock_eval_cfg)
    mock_dm_instance.setup.assert_called_once_with("test")

    mock_trainer_cls.assert_called_once()
    mock_trainer.test.assert_called_once()

    mock_torch_load.assert_called_once_with(mock_eval_cfg.model_path, map_location=ANY)
    mock_load_state_dict.assert_called_once_with({"weights": "dummy_state_dict"})

    # 7) Confirm test results
    assert mock_trainer.test.return_value == [{"test_acc": 0.9, "test_loss": 0.2}]
