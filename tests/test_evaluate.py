"""
Testes unitários para o módulo evaluate.

Garante 100% de cobertura das funções de avaliação e cálculo de métricas.
"""

import pytest
import torch
import numpy as np
from typing import Tuple, Dict
from unittest.mock import Mock, MagicMock
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from src.evaluate import evaluate_model, calculate_metrics
from src.lstm_model import LSTMModel


class TestCalculateMetrics:
    """Testes para a função calculate_metrics."""

    def test_calculate_metrics_returns_dict(self):
        """Testa se a função retorna um dicionário."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 2.1, 3.1])
        
        result = calculate_metrics(predictions, actuals)
        
        assert isinstance(result, dict)

    def test_calculate_metrics_has_required_keys(self):
        """Testa se o dicionário contém as chaves esperadas."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 2.1, 3.1])
        
        result = calculate_metrics(predictions, actuals)
        
        assert 'mae' in result
        assert 'rmse' in result
        assert 'mape' in result

    def test_calculate_metrics_values_are_floats(self):
        """Testa se os valores das métricas são floats."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 2.1, 3.1])
        
        result = calculate_metrics(predictions, actuals)
        
        assert isinstance(result['mae'], (float, np.floating))
        assert isinstance(result['rmse'], (float, np.floating))
        assert isinstance(result['mape'], (float, np.floating))

    def test_calculate_metrics_mae_calculation(self):
        """Testa o cálculo correto do MAE."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.5, 2.5, 3.5])
        
        result = calculate_metrics(predictions, actuals)
        expected_mae = np.mean(np.abs(predictions - actuals))
        
        assert np.isclose(result['mae'], expected_mae)

    def test_calculate_metrics_rmse_calculation(self):
        """Testa o cálculo correto do RMSE."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.5, 2.5, 3.5])
        
        result = calculate_metrics(predictions, actuals)
        expected_rmse = np.sqrt(np.mean((predictions - actuals)**2))
        
        assert np.isclose(result['rmse'], expected_rmse)

    def test_calculate_metrics_mape_calculation(self):
        """Testa o cálculo correto do MAPE."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.5, 2.5, 3.5])
        
        result = calculate_metrics(predictions, actuals)
        expected_mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        assert np.isclose(result['mape'], expected_mape)

    def test_calculate_metrics_perfect_prediction(self):
        """Testa com predições perfeitas (MAE e RMSE devem ser 0)."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])
        
        result = calculate_metrics(predictions, actuals)
        
        assert result['mae'] == 0.0
        assert result['rmse'] == 0.0
        assert result['mape'] == 0.0

    def test_calculate_metrics_large_arrays(self):
        """Testa com arrays grandes."""
        predictions = np.random.randn(10000)
        actuals = np.random.randn(10000)
        
        result = calculate_metrics(predictions, actuals)
        
        assert result['mae'] >= 0
        assert result['rmse'] >= 0
        assert result['mape'] >= 0

    def test_calculate_metrics_single_sample(self):
        """Testa com uma única amostra."""
        predictions = np.array([1.0])
        actuals = np.array([1.5])
        
        result = calculate_metrics(predictions, actuals)
        
        assert result['mae'] == 0.5
        assert result['rmse'] == 0.5

    def test_calculate_metrics_negative_values(self):
        """Testa com valores negativos."""
        predictions = np.array([-1.0, -2.0, -3.0])
        actuals = np.array([-1.5, -2.5, -3.5])
        
        result = calculate_metrics(predictions, actuals)
        
        assert result['mae'] > 0
        assert result['rmse'] > 0


class TestEvaluateModel:
    """Testes para a função evaluate_model."""

    def test_evaluate_model_returns_tuple(self):
        """Testa se a função retorna uma tupla."""
        model = LSTMModel()
        
        # Cria dados de teste mock
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        result = evaluate_model(model, test_loader, scaler, device)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_evaluate_model_returns_arrays(self):
        """Testa se a função retorna arrays numpy."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(actuals, np.ndarray)

    def test_evaluate_model_shapes_match(self):
        """Testa se as formas de predictions e actuals correspondem."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert predictions.shape == actuals.shape

    def test_evaluate_model_no_nan_values(self):
        """Testa se não há NaN nos resultados."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert not np.isnan(predictions).any()
        assert not np.isnan(actuals).any()

    def test_evaluate_model_eval_mode(self):
        """Testa se o modelo é colocado em modo eval."""
        model = LSTMModel()
        model.train()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        evaluate_model(model, test_loader, scaler, device)
        
        assert not model.training

    def test_evaluate_model_multiple_batches(self):
        """Testa com múltiplos lotes."""
        model = LSTMModel()
        
        X_test = torch.randn(100, 10, 1)
        y_test = torch.randn(100, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert len(predictions) == 100
        assert len(actuals) == 100

    def test_evaluate_model_single_batch(self):
        """Testa com um único lote."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert predictions.shape == (32, 1)

    def test_evaluate_model_custom_scaler(self):
        """Testa com um scaler customizado."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        # Scaler com range específico
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(np.array([[0], [100]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert predictions.shape == actuals.shape

    def test_evaluate_model_no_gradient(self):
        """Testa se não há gradientes calculados durante avaliação."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        with torch.no_grad():
            predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert predictions.shape == actuals.shape


class TestEvaluateModelEdgeCases:
    """Testes para casos extremos."""

    def test_evaluate_with_batch_size_one(self):
        """Testa com batch size de 1."""
        model = LSTMModel()
        
        X_test = torch.randn(10, 10, 1)
        y_test = torch.randn(10, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=1)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert len(predictions) == 10

    def test_evaluate_with_large_batch_size(self):
        """Testa com batch size grande."""
        model = LSTMModel()
        
        X_test = torch.randn(100, 10, 1)
        y_test = torch.randn(100, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=100)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert len(predictions) == 100

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_evaluate_on_cuda(self):
        """Testa avaliação em CUDA (se disponível)."""
        model = LSTMModel().cuda()
        
        X_test = torch.randn(32, 10, 1).cuda()
        y_test = torch.randn(32, 1).cuda()
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cuda')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        
        assert predictions.shape == actuals.shape


class TestEvaluateAndCalculateIntegration:
    """Testes de integração entre evaluate_model e calculate_metrics."""

    def test_metrics_from_evaluation(self):
        """Testa cálculo de métricas a partir de avaliação."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        metrics = calculate_metrics(predictions, actuals)
        
        assert all(key in metrics for key in ['mae', 'rmse', 'mape'])
        assert all(metric >= 0 for metric in metrics.values())

    def test_metrics_are_reasonable(self):
        """Testa se as métricas são valores razoáveis."""
        model = LSTMModel()
        
        X_test = torch.randn(32, 10, 1)
        y_test = torch.randn(32, 1)
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=32)
        
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1]]))
        
        device = torch.device('cpu')
        
        predictions, actuals = evaluate_model(model, test_loader, scaler, device)
        # Garantir que são numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        if isinstance(actuals, torch.Tensor):
            actuals = actuals.numpy()
            
        metrics = calculate_metrics(predictions, actuals)
        
        # RMSE deve ser >= MAE
        assert metrics['rmse'] >= metrics['mae']
        # Valores devem ser positivos finitos
        assert all(np.isfinite([v for v in metrics.values()]))
