import pytest
import pandas as pd
import numpy as np
import torch
from src.preprocessing import create_sequences, prepare_data

def test_create_sequences():
    """Testa a criação de sequências temporais."""
    data = np.array([[i] for i in range(100)])
    seq_length = 10
    X, y = create_sequences(data, seq_length)
    expected_samples = 100 - seq_length
    assert X.shape == (expected_samples, seq_length)
    assert y.shape == (expected_samples,)
    assert np.array_equal(X[0], np.arange(10))
    assert y[0] == 10

def test_prepare_data_success():
    """Testa o fluxo completo de preparação de dados com sucesso."""
    dates = pd.date_range(start='2020-01-01', periods=100)
    df = pd.DataFrame({
        'Close': np.linspace(100, 200, 100)
    }, index=dates)
    
    train_loader, test_loader, scaler = prepare_data(
        df, test_size=0.2, sequence_length=10, batch_size=5
    )
    
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    assert scaler.data_min_ is not None

def test_prepare_data_empty_df():
    """Testa erro ao passar DataFrame vazio."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame está vazio"):
        prepare_data(df)

def test_prepare_data_missing_column():
    """Testa erro ao passar DataFrame sem coluna Close."""
    df = pd.DataFrame({'Open': [1, 2, 3]})
    with pytest.raises(ValueError, match="DataFrame deve conter coluna 'Close'"):
        prepare_data(df)
