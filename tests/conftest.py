"""
Pytest configuration and fixtures for the entire test suite.

This file is automatically discovered and loaded by pytest.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root and directories to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "app"))


@pytest.fixture(scope="session")
def pytorch_device():
    """Fixture that returns the appropriate device (CPU or CUDA)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    return device


@pytest.fixture(scope="session")
def random_seed():
    """Fixture for reproducible randomness."""
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return SEED


@pytest.fixture
def lstm_model():
    """Fixture that provides a basic LSTM model."""
    from src.lstm_model import LSTMModel
    return LSTMModel()


@pytest.fixture
def lstm_model_custom():
    """Fixture that provides a customized LSTM model."""
    from src.lstm_model import LSTMModel
    return LSTMModel(input_size=5, hidden_layer_size=100, output_size=3)


@pytest.fixture
def sample_tensor_batch():
    """Fixture that provides a sample batch of tensors."""
    batch_size, seq_length, input_size = 32, 10, 1
    return torch.randn(batch_size, seq_length, input_size)


@pytest.fixture
def sample_labels():
    """Fixture that provides sample labels."""
    batch_size = 32
    return torch.randn(batch_size, 1)


@pytest.fixture
def minmax_scaler():
    """Fixture that provides a fitted MinMaxScaler."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0], [1]]))
    return scaler


@pytest.fixture
def sample_dataloader(sample_tensor_batch, sample_labels):
    """Fixture that provides a sample DataLoader."""
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(sample_tensor_batch, sample_labels)
    return DataLoader(dataset, batch_size=16)


@pytest.fixture
def temp_model_path(tmp_path):
    """Fixture that provides a temporary path for saving models."""
    path = tmp_path / "model.pth"
    return str(path)


# Hooks for test execution

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set up custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add unit marker to tests without explicit marker
        if not any(marker.name in ["integration", "slow", "gpu"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Automatically reset random seeds for each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup after test


def pytest_runtest_logreport(report):
    """
    Generate custom test report.
    """
    if report.when == "call":
        if report.outcome == "passed":
            print(f"\n[PASS] {report.nodeid}")
        elif report.outcome == "failed":
            print(f"\n[FAIL] {report.nodeid}")
