import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from src.data_loader import DataProcessor, load_data

@pytest.fixture
def mock_yf_download():
    with patch("src.data_loader.yf.download") as mock:
        yield mock

def test_load_data_success(mock_yf_download):
    """Testa carregamento de dados com sucesso."""
    dates = pd.date_range(start='2020-01-01', periods=5)
    mock_df = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0, 103.0, 104.0],
        'Open': [99.0] * 5
    }, index=dates)
    mock_yf_download.return_value = mock_df
    
    df = load_data('AAPL', '2020-01-01', '2020-01-05')
    
    assert isinstance(df, pd.DataFrame)
    assert 'Close' in df.columns
    assert len(df) == 5
    assert 'Open' not in df.columns
    mock_yf_download.assert_called_once()

def test_load_data_empty(mock_yf_download):
    """Testa erro quando yfinance não retorna dados."""
    mock_yf_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="No data found"):
        load_data('AAPL', '2020-01-01', '2020-01-05')

def test_data_processor_init():
    """Testa inicialização do DataProcessor."""
    processor = DataProcessor(symbol='MSFT', sequence_length=30)
    assert processor.symbol == 'MSFT'
    assert processor.sequence_length == 30
    assert processor.scaler is not None

def test_data_processor_workflow(mock_yf_download):
    """Testa o fluxo completo do DataProcessor."""
    seq_length = 10
    total_points = 100
    dates = pd.date_range(start='2020-01-01', periods=total_points)
    mock_df = pd.DataFrame({
        'Close': np.linspace(100, 200, total_points)
    }, index=dates)
    mock_yf_download.return_value = mock_df
    
    processor = DataProcessor(sequence_length=seq_length)
    X_train, y_train, X_test, y_test = processor.get_train_test_data(split_ratio=0.8)
    
    assert isinstance(X_train, torch.Tensor)
    assert X_train.shape[1] == seq_length
    assert X_train.shape[2] == 1
    mock_yf_download.assert_called()

def test_data_processor_download_error(mock_yf_download):
    """Testa erro de download no processador."""
    mock_yf_download.return_value = pd.DataFrame()
    processor = DataProcessor()
    with pytest.raises(ValueError, match="No data found"):
        processor.download_data()
