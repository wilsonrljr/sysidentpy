import pytest
import numpy as np
from sysidentpy.utils import get_siso_data, get_miso_data


def test_get_siso_data_default():
    """Test get_siso_data with default parameters."""
    x_train, x_valid, y_train, y_valid = get_siso_data()

    assert x_train.shape[0] == 4500, "Train set should have 4500 samples"
    assert x_valid.shape[0] == 500, "Validation set should have 500 samples"
    assert x_train.shape[1] == 1, "Input should have one feature"
    assert y_train.shape == x_train.shape, "Target shape should match input"


def test_get_siso_data_colored_noise():
    """Test get_siso_data with colored noise enabled."""
    x_train, x_valid, y_train, y_valid = get_siso_data(colored_noise=True)

    assert x_train.shape[0] == 4500, "Train set should have 4500 samples"
    assert x_valid.shape[0] == 500, "Validation set should have 500 samples"


def test_get_siso_data_custom_train_percentage():
    """Test get_siso_data with a different train percentage."""
    x_train, x_valid, y_train, y_valid = get_siso_data(train_percentage=80)

    assert x_train.shape[0] == 4000, "Train set should have 4000 samples"
    assert x_valid.shape[0] == 1000, "Validation set should have 1000 samples"


def test_get_miso_data_default():
    """Test get_miso_data with default parameters."""
    x_train, x_valid, y_train, y_valid = get_miso_data()

    assert x_train.shape[0] == 4500, "Train set should have 4500 samples"
    assert x_valid.shape[0] == 500, "Validation set should have 500 samples"
    assert x_train.shape[1] == 2, "Input should have two features"
    assert y_train.shape[0] == x_train.shape[0], "Target should match input"


def test_get_miso_data_colored_noise():
    """Test get_miso_data with colored noise enabled."""
    x_train, x_valid, y_train, y_valid = get_miso_data(colored_noise=True)

    assert x_train.shape[0] == 4500, "Train set should have 4500 samples"
    assert x_valid.shape[0] == 500, "Validation set should have 500 samples"


def test_get_miso_data_custom_train_percentage():
    """Test get_miso_data with a different train percentage."""
    x_train, x_valid, y_train, y_valid = get_miso_data(train_percentage=80)

    assert x_train.shape[0] == 4000, "Train set should have 4000 samples"
    assert x_valid.shape[0] == 1000, "Validation set should have 1000 samples"


def test_get_siso_data_invalid_train_percentage():
    """Test get_siso_data with invalid train percentage."""
    with pytest.raises(ValueError, match="train_percentage must be smaller than 100"):
        get_siso_data(train_percentage=150)


def test_get_miso_data_invalid_train_percentage():
    """Test get_miso_data with invalid train percentage."""
    with pytest.raises(ValueError, match="train_percentage must be smaller than 100"):
        get_miso_data(train_percentage=-10)
