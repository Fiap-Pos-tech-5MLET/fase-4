import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app
from app.routes.train_route import JOBS, train_model_task
from app.schemas import TrainRequest

client = TestClient(app)


def test_import_error_fallback():
    """Testa o fallback de ImportError para run_training_pipeline."""
    with patch("app.routes.train_route.run_training_pipeline", side_effect=ImportError("Test import error")):
        payload = {
            "symbol": "TEST",
            "start_date": "2023-01-01",
            "end_date": "2023-02-01",
            "epochs": 1
        }
        
        response = client.post("/train", json=payload)
        assert response.status_code == 202


def test_train_model_task_with_error_result():
    """Testa train_model_task quando o treinamento retorna um erro."""
    job_id = "test-error-job"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    request = TrainRequest(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-02-01",
        epochs=1
    )
    
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline:
        mock_pipeline.return_value = {"error": "O treinamento falhou devido à falta de dados."}
        
        train_model_task(job_id, request)
        
        assert JOBS[job_id]["status"] == "failed"
        assert "error" in JOBS[job_id]
        assert JOBS[job_id]["error"] == "O treinamento falhou devido à falta de dados."


def test_train_model_task_with_exception():
    """Testa train_model_task quando uma exceção é lançada."""
    job_id = "test-exception-job"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    request = TrainRequest(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-02-01",
        epochs=1
    )
    
    with patch("app.routes.train_route.run_training_pipeline", side_effect=Exception("Unexpected error")):
        train_model_task(job_id, request)
        
        assert JOBS[job_id]["status"] == "failed"
        assert "error" in JOBS[job_id]
        assert "Unexpected error" in JOBS[job_id]["error"]


def test_train_model_task_hot_reload_success():
    """Testa o hot-reload bem-sucedido do modelo e do scaler após o treinamento."""
    job_id = "test-hotreload-job"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    request = TrainRequest(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-02-01",
        epochs=1,
        hidden_layer_size=64,
        num_layers=2,
        dropout=0.2
    )
    
    training_result = {
        "mae": 0.1,
        "rmse": 0.15,
        "mape": 5.0,
        "is_best_model": True
    }
    
    mock_model_state = {"layer1.weight": "fake_weights"}
    mock_scaler = MagicMock()
    
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline, \
         patch("app.routes.train_route.os.path.exists") as mock_exists, \
         patch("torch.load") as mock_torch_load, \
         patch("joblib.load") as mock_joblib_load, \
         patch("src.lstm_model.LSTMModel") as MockModel, \
         patch("app.config.get_settings") as mock_get_settings:
        
        mock_pipeline.return_value = training_result
        mock_exists.return_value = True
        mock_torch_load.return_value = mock_model_state
        mock_joblib_load.return_value = mock_scaler
        
        mock_model_instance = MagicMock()
        MockModel.return_value = mock_model_instance
        
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        
        train_model_task(job_id, request)
        
        assert JOBS[job_id]["status"] == "completed"
        assert JOBS[job_id]["result"] == training_result
        
        # Verify hot-reload was attempted
        mock_joblib_load.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_model_instance.load_state_dict.assert_called_once_with(mock_model_state)


def test_train_model_task_hot_reload_not_best_model():
    """Testa que o hot-reload é ignorado quando o modelo não é o melhor."""
    job_id = "test-not-best-job"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    request = TrainRequest(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-02-01",
        epochs=1
    )
    
    training_result = {
        "mae": 0.5,
        "rmse": 0.6,
        "mape": 10.0,
        "is_best_model": False
    }
    
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline, \
         patch("joblib.load") as mock_joblib_load:
        
        mock_pipeline.return_value = training_result
        
        train_model_task(job_id, request)
        
        assert JOBS[job_id]["status"] == "completed"
        mock_joblib_load.assert_not_called()


def test_train_model_task_hot_reload_error():
    """Testa que os erros de hot-reload não falham a tarefa."""
    job_id = "test-hotreload-error-job"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    request = TrainRequest(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-02-01",
        epochs=1,
        hidden_layer_size=64,
        num_layers=2,
        dropout=0.2
    )
    
    training_result = {
        "mae": 0.1,
        "is_best_model": True
    }
    
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline, \
         patch("app.routes.train_route.os.path.exists") as mock_exists, \
         patch("joblib.load", side_effect=Exception("Failed to load scaler")):
        
        mock_pipeline.return_value = training_result
        mock_exists.return_value = True
        
        train_model_task(job_id, request)
        
        assert JOBS[job_id]["status"] == "completed"
        assert JOBS[job_id]["result"] == training_result


def test_train_model_task_missing_scaler():
    """Testa o hot-reload quando o arquivo do scaler não existe."""
    job_id = "test-missing-scaler-job"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    request = TrainRequest(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-02-01",
        epochs=1,
        hidden_layer_size=64,
        num_layers=2,
        dropout=0.2
    )
    
    training_result = {
        "mae": 0.1,
        "is_best_model": True
    }
    
    def exists_side_effect(path):
        if "scaler.pkl" in path:
            return False
        return True
    
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline, \
         patch("app.routes.train_route.os.path.exists", side_effect=exists_side_effect), \
         patch("torch.load") as mock_torch_load, \
         patch("src.lstm_model.LSTMModel") as MockModel, \
         patch("app.config.get_settings") as mock_get_settings:
        
        mock_pipeline.return_value = training_result
        mock_torch_load.return_value = {"state": "dict"}
        
        mock_model_instance = MagicMock()
        MockModel.return_value = mock_model_instance
        
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        
        train_model_task(job_id, request)
        
        assert JOBS[job_id]["status"] == "completed"


def test_train_model_task_missing_model():
    """Testa o hot-reload quando o arquivo do modelo não existe."""
    job_id = "test-missing-model-job"
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    request = TrainRequest(
        symbol="TEST",
        start_date="2023-01-01",
        end_date="2023-02-01",
        epochs=1
    )
    
    training_result = {
        "mae": 0.1,
        "is_best_model": True
    }
    
    def exists_side_effect(path):
        if "lstm_model.pth" in path:
            return False
        return True
    
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline, \
         patch("app.routes.train_route.os.path.exists", side_effect=exists_side_effect), \
         patch("joblib.load") as mock_joblib_load, \
         patch("app.config.get_settings") as mock_get_settings:
        
        mock_pipeline.return_value = training_result
        mock_joblib_load.return_value = MagicMock()
        
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        
        train_model_task(job_id, request)
        
        assert JOBS[job_id]["status"] == "completed"
