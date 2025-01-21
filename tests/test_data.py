import pytest
import torch
import pandas as pd

from pathlib import Path
from unittest.mock import patch
from typing import Dict, Any

# Import from your package:
from final_project.data import MentalDisordersDataset, preprocess


@pytest.fixture
def minimal_csv(tmp_path) -> Path:
    """
    Create a minimal CSV file with columns [title, selftext, subreddit].
    Returns the path to the CSV file.
    """
    data = {
        "title": ["My anxious thoughts", "Feeling depressed"],
        "selftext": [
            "I am very anxious about everything lately.",
            "Can't get out of bed due to depression."
        ],
        "subreddit": ["anxiety", "depression"]
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "minimal_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def small_tensor_dict() -> Dict[str, torch.Tensor]:
    """
    Return a tiny dictionary of tensors that mimic the structure
    expected by MentalDisordersDataset.
    """
    return {
        "input_ids": torch.tensor([[101, 102], [103, 104]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "labels": torch.tensor([0, 1])
    }


@pytest.fixture
def saved_train_test_files(tmp_path, small_tensor_dict) -> (Path, Path):
    """
    Create minimal training and testing .pt files that can be loaded by MentalDisordersDataset.
    Returns (train_path, test_path).
    """
    train_path = tmp_path / "train_data.pt"
    test_path = tmp_path / "test_data.pt"

    # For demonstration, we use the same small_tensor_dict for train and test
    torch.save(small_tensor_dict, train_path)
    torch.save(small_tensor_dict, test_path)

    return train_path, test_path


def test_preprocess_creates_files(minimal_csv, tmp_path):
    """
    Test that `preprocess()` successfully reads a CSV and
    writes two .pt files (train and test).
    """
    # Where we will save the train/test splits
    train_file = tmp_path / "train_output.pt"
    test_file = tmp_path / "test_output.pt"

    preprocess(
        raw_data_path=str(minimal_csv),
        training_data_path=str(train_file),
        testing_data_path=str(test_file),
        tokenizer_name="distilbert-base-uncased",
        max_length=64,  # keep it small for tests
    )

    # Check that files are created
    assert train_file.exists(), "Training file was not created by preprocess."
    assert test_file.exists(), "Test file was not created by preprocess."

    # Load them to verify structure
    train_dict = torch.load(train_file)
    test_dict = torch.load(test_file)
    for dset in [train_dict, test_dict]:
        assert isinstance(dset, dict), "Saved data is not a dictionary."
        assert "input_ids" in dset, "Saved dict missing 'input_ids'."
        assert "attention_mask" in dset, "Saved dict missing 'attention_mask'."
        assert "labels" in dset, "Saved dict missing 'labels'."


@pytest.mark.parametrize("percentage", [1.0, 0.5])
def test_mental_disorders_dataset_loading(saved_train_test_files, percentage):
    """
    Test that MentalDisordersDataset loads data from a .pt file
    and properly samples it if data_percentage < 1.0.
    """
    train_path, test_path = saved_train_test_files

    # Instantiate with train=True
    dataset = MentalDisordersDataset(
        data_percentage=percentage,
        seed=123,
        train=True,
        training_data_path=str(train_path),
        testing_data_path=str(test_path),
    )
    # If percentage=1.0 and we started with 2 samples in fixture => length=2
    # If percentage=0.5 => length=1
    expected_len = int(2 * percentage)
    assert len(dataset) == expected_len, (
        f"Dataset length {len(dataset)} != expected {expected_len} with data_percentage={percentage}"
    )

    # Confirm data structure:
    sample = dataset[0]
    assert "input_ids" in sample, "Dataset sample missing 'input_ids'."
    assert "attention_mask" in sample, "Dataset sample missing 'attention_mask'."
    assert "labels" in sample, "Dataset sample missing 'labels'."


def test_mental_disorders_dataset_file_not_found(tmp_path):
    """
    Test that we gracefully handle a missing file, or at least
    do not crash. The code prints 'File not found' so let's just
    verify it doesn't raise an unexpected exception.
    """
    random_file = tmp_path / "non_existent.pt"
    # We expect no crash, but a message. We'll just confirm no exception is raised.
    try:
        dataset = MentalDisordersDataset(
            data_percentage=1.0,
            seed=123,
            train=True,
            training_data_path=str(random_file),
            testing_data_path=str(random_file),
        )
    except Exception as e:
        pytest.fail(f"Dataset init unexpectedly raised {e} for missing file.")
