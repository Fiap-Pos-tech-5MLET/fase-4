import pytest
import torch
from unittest.mock import MagicMock, patch
from src.train import ModelTrainer, run_training_pipeline
from src.lstm_model import LSTMModel
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def mock_model():
    model = MagicMock(spec=LSTMModel)
    model.parameters.return_value = [torch.randn(1, requires_grad=True)]
    model.to.return_value = model
    return model

@pytest.fixture
def mock_dataloader():
    X = torch.randn(10, 5, 1)
    y = torch.randn(10, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=2)

def test_model_trainer_init(mock_model):
    """Testa inicialização do ModelTrainer."""
    trainer = ModelTrainer(mock_model, lr=0.01)
    assert trainer.model == mock_model
    assert isinstance(trainer.criterion, torch.nn.MSELoss)
    assert isinstance(trainer.optimizer, torch.optim.Adam)

def test_model_trainer_train_step(mock_model, mock_dataloader):
    """Testa o loop de treinamento (simulado)."""
    # Setup mock returns
    mock_model.return_value = torch.randn(2, 1) # Batch size 2
    
    trainer = ModelTrainer(mock_model)
    # Mock criterion to avoid shape errors during test
    trainer.criterion = MagicMock(return_value=torch.tensor(0.1, requires_grad=True))
    
    # Mock optimizer step
    trainer.optimizer.zero_grad = MagicMock()
    trainer.optimizer.step = MagicMock()
    
    loss_history = trainer.train(mock_dataloader, epochs=1)
    
    assert len(loss_history) == 1
    assert isinstance(loss_history[0], float)
    trainer.model.train.assert_called()

@patch("src.train.mlflow")
@patch("src.train.DataProcessor")
@patch("src.train.LSTMModel")
@patch("src.train.ModelTrainer")
@patch("src.train.evaluate_model")
@patch("src.train.save_model")
@patch("src.train.joblib.dump")
@patch("src.train.os.makedirs")
def test_run_training_pipeline_success(
    mock_makedirs, mock_joblib, mock_save, mock_evaluate, 
    mock_trainer_cls, mock_lstm_cls, mock_processor_cls, mock_mlflow
):
    """Testa o pipeline de treinamento com sucesso."""
    
    # Mock DataProcessor
    mock_processor = mock_processor_cls.return_value
    mock_processor.get_train_test_data.return_value = (
        torch.randn(10, 5, 1), torch.randn(10, 1),
        torch.randn(5, 5, 1), torch.randn(5, 1)
    )
    mock_processor.scaler = MagicMock()
    
    # Mock Trainer
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.train.return_value = [0.1, 0.05]
    mock_trainer.device = torch.device('cpu')
    mock_trainer.model = MagicMock()
    
    # Mock Evaluate return numpy arrays to allow np.mean calculation
    import numpy as np
    mock_evaluate.return_value = (np.array([1.0, 2.0]), np.array([1.1, 2.1]))
    
    result = run_training_pipeline(
        symbol='TEST',
        epochs=1,
        batch_size=2,
        learning_rate=0.01
    )
    
    assert "mae" in result
    assert "rmse" in result
    assert result["symbol"] == "TEST"
    
    # Verify calls
    mock_mlflow.start_run.assert_called()
    mock_mlflow.log_params.assert_called()
    mock_mlflow.log_metric.assert_called()
    mock_save.assert_called()

@patch("src.train.DataProcessor")
def test_run_training_pipeline_error(mock_processor_cls):
    """Testa o pipeline quando ocorre erro no processamento de dados."""
    mock_processor = mock_processor_cls.return_value
    mock_processor.get_train_test_data.side_effect = ValueError("Erro de dados")
    
    result = run_training_pipeline(symbol='TEST')
    
    assert "error" in result
    assert result["error"] == "Erro de dados"
