"""
Testes unitários para o módulo utils.

Garante 100% de cobertura das funções de salvamento e carregamento de modelos.
"""

import pytest
import torch
import torch.nn as nn
import os
import tempfile
from pathlib import Path
from src.utils import save_model, load_model
from src.lstm_model import LSTMModel


class TestSaveModel:
    """Testes para a função save_model."""

    def test_save_model_creates_file(self):
        """Testa se o arquivo de modelo é criado."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LSTMModel()
            path = os.path.join(tmpdir, "test_model.pth")
            
            save_model(model, path)
            
            assert os.path.exists(path)

    def test_save_model_file_size(self):
        """Testa se o arquivo salvo tem tamanho positivo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LSTMModel()
            path = os.path.join(tmpdir, "test_model.pth")
            
            save_model(model, path)
            
            file_size = os.path.getsize(path)
            assert file_size > 0

    def test_save_model_default_path(self):
        """Testa salvamento com caminho padrão."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                model = LSTMModel()
                
                save_model(model)
                
                assert os.path.exists("model.pth")
            finally:
                os.chdir(original_dir)

    def test_save_model_overwrites_existing(self):
        """Testa se o arquivo existente é sobrescrito."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            
            model1 = LSTMModel()
            save_model(model1, path)
            first_size = os.path.getsize(path)
            
            model2 = LSTMModel()
            save_model(model2, path)
            second_size = os.path.getsize(path)
            
            assert first_size == second_size

    def test_save_model_with_different_architectures(self):
        """Testa salvamento com arquiteturas diferentes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model1 = LSTMModel(input_size=5, hidden_layer_size=100, output_size=3)
            path1 = os.path.join(tmpdir, "model1.pth")
            save_model(model1, path1)
            
            model2 = LSTMModel(input_size=10, hidden_layer_size=200, output_size=5)
            path2 = os.path.join(tmpdir, "model2.pth")
            save_model(model2, path2)
            
            assert os.path.exists(path1)
            assert os.path.exists(path2)
            assert os.path.getsize(path1) != os.path.getsize(path2)

    def test_save_model_is_state_dict(self):
        """Testa se o arquivo contém um state_dict válido."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LSTMModel()
            path = os.path.join(tmpdir, "model.pth")
            save_model(model, path)
            
            # Tenta carregar como state_dict
            state_dict = torch.load(path)
            assert isinstance(state_dict, dict)
            assert 'lstm.weight_ih_l0' in state_dict


class TestLoadModel:
    """Testes para a função load_model."""

    def test_load_model_returns_model(self):
        """Testa se load_model retorna o modelo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LSTMModel()
            path = os.path.join(tmpdir, "model.pth")
            save_model(model, path)
            
            loaded_model = load_model(model, path)
            
            assert isinstance(loaded_model, LSTMModel)

    def test_load_model_default_path(self):
        """Testa carregamento com caminho padrão."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                model = LSTMModel()
                save_model(model)
                
                loaded_model = LSTMModel()
                load_model(loaded_model)
                
                assert isinstance(loaded_model, LSTMModel)
            finally:
                os.chdir(original_dir)

    def test_load_model_weights_match(self):
        """Testa se os pesos carregados correspondem aos salvos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model1 = LSTMModel()
            model1.eval()  # Colocar em eval para desativar dropout
            path = os.path.join(tmpdir, "model.pth")
            save_model(model1, path)
            
            model2 = LSTMModel()
            load_model(model2, path)
            model2.eval()  # Garantir que está em eval
            
            # Usar mesmo input e comparar outputs
            x = torch.randn(32, 10, 1)
            
            # Garantir que ambos estão em eval mode
            with torch.no_grad():
                out1 = model1.forward(x)
                out2 = model2.forward(x)
            
            assert torch.allclose(out1, out2)

    def test_load_model_sets_eval_mode(self):
        """Testa se o modelo é colocado em modo eval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LSTMModel()
            path = os.path.join(tmpdir, "model.pth")
            save_model(model, path)
            
            model.train()  # Coloca em modo train
            load_model(model, path)
            
            assert not model.training

    def test_load_model_nonexistent_file(self):
        """Testa carregamento de arquivo inexistente."""
        model = LSTMModel()
        
        with pytest.raises(FileNotFoundError):
            load_model(model, "nonexistent_model.pth")

    def test_load_model_corrupted_file(self):
        """Testa carregamento de arquivo corrompido."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupted.pth")
            with open(path, 'w') as f:
                f.write("corrupted data")
            
            model = LSTMModel()
            
            with pytest.raises(Exception):  # torch.load lança exceção
                load_model(model, path)

    def test_load_model_different_architecture(self):
        """Testa carregamento com arquitetura diferente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model1 = LSTMModel(input_size=5, hidden_layer_size=100, output_size=3)
            path = os.path.join(tmpdir, "model.pth")
            save_model(model1, path)
            
            model2 = LSTMModel(input_size=10, hidden_layer_size=200, output_size=5)
            
            with pytest.raises(RuntimeError):
                load_model(model2, path)


class TestSaveLoadIntegration:
    """Testes de integração entre save e load."""

    def test_save_load_cycle(self):
        """Testa um ciclo completo de save e load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            
            # Salva modelo
            original_model = LSTMModel()
            original_state = {k: v.clone() for k, v in original_model.state_dict().items()}
            save_model(original_model, path)
            
            # Carrega modelo
            loaded_model = LSTMModel()
            load_model(loaded_model, path)
            loaded_state = loaded_model.state_dict()
            
            # Verifica se os estados são iguais
            for key in original_state:
                assert torch.allclose(original_state[key], loaded_state[key])

    def test_multiple_save_load_cycles(self):
        """Testa múltiplos ciclos de save e load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LSTMModel()
            
            for i in range(5):
                path = os.path.join(tmpdir, f"model_{i}.pth")
                save_model(model, path)
                load_model(model, path)
                
                assert os.path.exists(path)

    def test_save_load_with_inference(self):
        """Testa save/load e depois faz inferência."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            
            model = LSTMModel()
            save_model(model, path)
            
            loaded_model = LSTMModel()
            load_model(loaded_model, path)
            
            x = torch.randn(32, 10, 1)
            output = loaded_model.forward(x)
            
            assert output.shape == (32, 1)
            assert not torch.isnan(output).any()


class TestModelUtilsEdgeCases:
    """Testes para casos extremos."""

    def test_save_with_special_characters_in_path(self):
        """Testa salvamento com caracteres especiais no caminho."""
        with tempfile.TemporaryDirectory() as tmpdir:
            special_dir = os.path.join(tmpdir, "model_2024-12-14")
            os.makedirs(special_dir)
            path = os.path.join(special_dir, "model.pth")
            
            model = LSTMModel()
            save_model(model, path)
            
            assert os.path.exists(path)

    def test_save_model_permissions(self):
        """Testa se o arquivo salvo tem permissões de leitura."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            model = LSTMModel()
            save_model(model, path)
            
            assert os.access(path, os.R_OK)

    def test_load_after_model_modification(self):
        """Testa carregamento após modificação manual do modelo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            
            model = LSTMModel()
            save_model(model, path)
            
            # Modifica parâmetros
            for param in model.parameters():
                param.data.fill_(0)
            
            # Carrega de volta
            load_model(model, path)
            
            # Verifica se foi restaurado (não todos zeros)
            has_nonzero = False
            for param in model.parameters():
                if (param.data != 0).any():
                    has_nonzero = True
                    break
            
            assert has_nonzero
