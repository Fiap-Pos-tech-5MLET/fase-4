"""
Testes unitários para o módulo lstm_model.

Garante 100% de cobertura do código da classe LSTMModel,
incluindo inicialização, forward pass e representação em string.
"""

import pytest
import torch
import torch.nn as nn
from src.lstm_model import LSTMModel


class TestLSTMModelInit:
    """Testes para inicialização do modelo LSTM."""

    def test_init_default_parameters(self):
        """Testa inicialização com parâmetros padrão."""
        model = LSTMModel()
        
        assert model.hidden_layer_size == 50
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.linear, nn.Linear)

    def test_init_custom_parameters(self):
        """Testa inicialização com parâmetros customizados."""
        model = LSTMModel(input_size=5, hidden_layer_size=100, output_size=3, num_layers=3, dropout=0.5)
        
        assert model.hidden_layer_size == 100
        assert model.lstm.input_size == 5
        assert model.lstm.hidden_size == 100
        assert model.lstm.num_layers == 3
        assert model.lstm.dropout == 0.5
        assert model.linear.out_features == 3

    def test_init_lstm_batch_first(self):
        """Testa se LSTM foi configurado com batch_first=True."""
        model = LSTMModel()
        assert model.lstm.batch_first is True
        assert model.lstm.num_layers == 2 # Default updated

    def test_model_is_nn_module(self):
        """Testa se o modelo é uma instância de nn.Module."""
        model = LSTMModel()
        assert isinstance(model, nn.Module)


class TestLSTMModelForward:
    """Testes para o forward pass do modelo."""

    def test_forward_default_shapes(self):
        """Testa o forward pass com shapes padrão."""
        model = LSTMModel()
        batch_size, seq_length, input_size = 32, 10, 1
        
        x = torch.randn(batch_size, seq_length, input_size)
        output = model.forward(x)
        
        assert output.shape == (batch_size, 1)

    def test_forward_custom_shapes(self):
        """Testa o forward pass com dimensões customizadas."""
        model = LSTMModel(input_size=5, hidden_layer_size=100, output_size=3)
        batch_size, seq_length = 16, 20
        
        x = torch.randn(batch_size, seq_length, 5)
        output = model.forward(x)
        
        assert output.shape == (batch_size, 3)

    def test_forward_single_sample(self):
        """Testa o forward pass com uma única amostra."""
        model = LSTMModel()
        x = torch.randn(1, 10, 1)
        output = model.forward(x)
        
        assert output.shape == (1, 1)

    def test_forward_large_batch(self):
        """Testa o forward pass com lote grande."""
        model = LSTMModel()
        x = torch.randn(256, 10, 1)
        output = model.forward(x)
        
        assert output.shape == (256, 1)

    def test_forward_output_type(self):
        """Testa se o output é um tensor PyTorch."""
        model = LSTMModel()
        x = torch.randn(32, 10, 1)
        output = model.forward(x)
        
        assert isinstance(output, torch.Tensor)

    def test_forward_output_gradient(self):
        """Testa se o output retem informações de gradiente."""
        model = LSTMModel()
        x = torch.randn(32, 10, 1, requires_grad=True)
        output = model.forward(x)
        
        assert output.requires_grad is True

    def test_forward_no_nan_output(self):
        """Testa se o output não contém NaN."""
        model = LSTMModel()
        x = torch.randn(32, 10, 1)
        output = model.forward(x)
        
        assert not torch.isnan(output).any()

    def test_forward_backward_pass(self):
        """Testa se o backward pass funciona corretamente."""
        model = LSTMModel()
        x = torch.randn(32, 10, 1, requires_grad=True)
        output = model.forward(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


class TestLSTMModelString:
    """Testes para a representação em string do modelo."""

    def test_str_method_returns_string(self):
        """Testa se __str__ retorna uma string."""
        model = LSTMModel()
        result = str(model)
        
        assert isinstance(result, str)

    def test_str_contains_model_name(self):
        """Testa se a string contém o nome do modelo."""
        model = LSTMModel()
        result = str(model)
        
        assert "LSTMModel" in result

    def test_str_contains_hidden_layer_size(self):
        """Testa se a string contém o tamanho da camada oculta."""
        model = LSTMModel(hidden_layer_size=75)
        result = str(model)
        
        assert "75" in result

    def test_str_contains_lstm_info(self):
        """Testa se a string contém informações sobre LSTM."""
        model = LSTMModel()
        result = str(model)
        
        assert "lstm" in result.lower()

    def test_str_contains_linear_info(self):
        """Testa se a string contém informações sobre a camada linear."""
        model = LSTMModel()
        result = str(model)
        
        assert "linear" in result.lower()

    def test_str_is_formatted(self):
        """Testa se a string está bem formatada com quebras de linha."""
        model = LSTMModel()
        result = str(model)
        
        assert "\n" in result


class TestLSTMModelDeviceCompatibility:
    """Testes para compatibilidade com diferentes dispositivos."""

    def test_model_cpu(self):
        """Testa o modelo em CPU."""
        model = LSTMModel()
        model = model.cpu()
        
        x = torch.randn(32, 10, 1)
        output = model.forward(x)
        
        assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda(self):
        """Testa o modelo em CUDA (se disponível)."""
        model = LSTMModel()
        model = model.cuda()
        
        x = torch.randn(32, 10, 1).cuda()
        output = model.forward(x)
        
        assert output.device.type == 'cuda'


class TestLSTMModelStateDict:
    """Testes para salvar e carregar estado do modelo."""

    def test_state_dict_keys(self):
        """Testa se o state_dict contém as chaves esperadas."""
        model = LSTMModel()
        state_dict = model.state_dict()
        
        assert 'lstm.weight_ih_l0' in state_dict
        assert 'lstm.weight_hh_l0' in state_dict
        assert 'linear.weight' in state_dict
        assert 'linear.bias' in state_dict

    def test_load_state_dict(self):
        """Testa se é possível carregar um state_dict."""
        model1 = LSTMModel()
        state_dict = model1.state_dict()
        
        model2 = LSTMModel()
        model2.load_state_dict(state_dict)
        
        model1.eval()
        model2.eval()
        
        x = torch.randn(32, 10, 1)
        out1 = model1.forward(x)
        out2 = model2.forward(x)
        
        assert torch.allclose(out1, out2)


class TestLSTMModelEdgeCases:
    """Testes para casos extremos."""

    def test_zero_input(self):
        """Testa com entrada zero."""
        model = LSTMModel()
        x = torch.zeros(32, 10, 1)
        output = model.forward(x)
        
        assert output.shape == (32, 1)

    def test_very_small_input(self):
        """Testa com valores muito pequenos."""
        model = LSTMModel()
        x = torch.randn(32, 10, 1) * 1e-6
        output = model.forward(x)
        
        assert output.shape == (32, 1)

    def test_very_large_input(self):
        """Testa com valores muito grandes."""
        model = LSTMModel()
        x = torch.randn(32, 10, 1) * 1e6
        output = model.forward(x)
        
        assert output.shape == (32, 1)

    def test_minimum_sequence_length(self):
        """Testa com comprimento de sequência mínimo."""
        model = LSTMModel()
        x = torch.randn(32, 1, 1)
        output = model.forward(x)
        
        assert output.shape == (32, 1)
