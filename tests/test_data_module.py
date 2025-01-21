import pytest
import torch
from unittest.mock import patch, MagicMock

from torch.utils.data import DataLoader

from final_project import MentalDisordersDataModule

@pytest.fixture
def mock_cfg(tmp_path):
    """
    Return a mock config object with the necessary attributes.
    """
    class MockCfg:
        batch_size = 4
        num_workers = 2
        data_percentage = 1.0
        seed = 123
        training_data_path = str(tmp_path / "train_data.pt")
        testing_data_path = str(tmp_path / "test_data.pt")
    return MockCfg()


@pytest.fixture
def mock_dataset():
    """
    Return a mock dataset that we can control the length of.
    """
    mock_ds = MagicMock()
    # We'll say the dataset length is 100 for testing
    mock_ds.__len__.return_value = 100  
    return mock_ds


def test_init_data_module(mock_cfg):
    """
    Basic test to ensure the DataModule correctly sets internal fields.
    """
    dm = MentalDisordersDataModule(mock_cfg)
    assert dm.batch_size == mock_cfg.batch_size, "batch_size not set correctly."
    assert dm.num_workers == mock_cfg.num_workers, "num_workers not set correctly."
    assert dm.data_percentage == mock_cfg.data_percentage, "data_percentage not set correctly."
    assert dm.seed == mock_cfg.seed, "seed not set correctly."
    assert dm.training_data_path == mock_cfg.training_data_path, "training_data_path not set correctly."
    assert dm.testing_data_path == mock_cfg.testing_data_path, "testing_data_path not set correctly."


@patch("final_project.data_module.MentalDisordersDataset")
def test_setup_fit(mock_data_class, mock_dataset, mock_cfg):
    """
    Test the 'fit' stage of the DataModule.
    We patch the dataset so it doesn't rely on actual .pt files.
    """
    mock_data_class.return_value = mock_dataset
    dm = MentalDisordersDataModule(mock_cfg)

    dm.setup("fit")
    
    # We expect the dataset length to be 100 (mock_dataset).
    # 90% -> train=90, val=10
    assert len(dm.train) == 90, f"Expected train split of 90, got {len(dm.train)}"
    assert len(dm.val) == 10, f"Expected val split of 10, got {len(dm.val)}"

    # Confirm data loaders are created
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert isinstance(train_loader, DataLoader), "train_dataloader did not return a DataLoader."
    assert isinstance(val_loader, DataLoader), "val_dataloader did not return a DataLoader."
    
    # Check batch size
    assert train_loader.batch_size == mock_cfg.batch_size, "Train loader has incorrect batch size."
    assert val_loader.batch_size == mock_cfg.batch_size, "Val loader has incorrect batch size."


@patch("final_project.data_module.MentalDisordersDataset")
def test_setup_test(mock_data_class, mock_dataset, mock_cfg):
    """
    Test the 'test' stage of the DataModule.
    """
    mock_data_class.return_value = mock_dataset
    dm = MentalDisordersDataModule(mock_cfg)
    dm.setup("test")

    # The test dataset is stored in dm.test
    assert dm.test is not None, "dm.test was never set."
    assert len(dm.test) == 100, f"Expected length of 100, got {len(dm.test)}"

    test_loader = dm.test_dataloader()
    assert isinstance(test_loader, DataLoader), "test_dataloader did not return a DataLoader."
    assert test_loader.batch_size == mock_cfg.batch_size, "Test loader has incorrect batch size."


@pytest.mark.parametrize("stage", ["fit", "test", "bogus"])
def test_setup_stages(mock_dataset, mock_cfg, stage):
    """
    Check that calling setup with various stages doesn't raise an error
    (even if the stage is not recognized).
    """
    # Instead of fully patching, let's do a partial approach:
    # We'll patch only when stage is 'fit' or 'test', since those create a dataset.
    with patch("final_project.data_module.MentalDisordersDataset", return_value=mock_dataset) as mock_data_class:
        dm = MentalDisordersDataModule(mock_cfg)

        dm.setup(stage)

        if stage == "fit":
            # Expect train & val
            assert hasattr(dm, "train"), "dm.train not defined for stage=fit"
            assert hasattr(dm, "val"), "dm.val not defined for stage=fit"
        elif stage == "test":
            # Expect test
            assert hasattr(dm, "test"), "dm.test not defined for stage=test"
        else:
            # Some arbitrary stage that does nothing
            # Just ensure no crash
            pass


@patch("final_project.data_module.MentalDisordersDataset")
def test_seed_effect(mock_data_class, mock_dataset, mock_cfg):
    """
    Test that the seed is actually being set by comparing random values.
    This is more of an integration-style test.
    """
    mock_data_class.return_value = mock_dataset
    dm = MentalDisordersDataModule(mock_cfg)
    
    # We call setup('fit'), which calls seed_everything(self.seed).
    dm.setup("fit")
    
    # Next random number should be deterministically the same if the seed was set properly
    rand_val1 = torch.rand(1)
    
    # Re-run the entire sequence
    dm2 = MentalDisordersDataModule(mock_cfg)
    dm2.setup("fit")
    rand_val2 = torch.rand(1)
    
    # If seeds are set consistently, these two should match exactly
    assert torch.equal(rand_val1, rand_val2), "Random values differ, seed_everything might not be called correctly."