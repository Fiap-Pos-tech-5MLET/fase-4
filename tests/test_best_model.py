"""
Testes para o sistema de checkpoint de melhor modelo.
"""

import pytest
import os
import torch
from src.train import run_training_pipeline
from src.model_loader import load_best_model, get_best_model_info


class TestBestModelCheckpoint:
    """Testes para o sistema de salvamento do melhor modelo."""
    
    def setup_method(self):
        """Limpar arquivos de teste antes de cada teste."""
        best_model_path = "app/artifacts/best_stock_lstm_model.pth"
        best_loss_file = "app/artifacts/best_test_loss.txt"
        
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        if os.path.exists(best_loss_file):
            os.remove(best_loss_file)
    
    @pytest.mark.slow
    def test_first_training_saves_best_model(self):
        """Verifica se o primeiro treinamento salva o melhor modelo."""
        # Primeiro treinamento
        result = run_training_pipeline(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            epochs=3,
            batch_size=32,
            hidden_layer_size=32,
            num_layers=1,
            seed=42
        )
        
        # Verificar que não houve erro
        assert "error" not in result
        
        # Verificar que é o melhor modelo (primeiro sempre é)
        assert result["is_best_model"] == True
        
        # Verificar que arquivos foram criados
        assert os.path.exists("app/artifacts/best_stock_lstm_model.pth")
        assert os.path.exists("app/artifacts/best_test_loss.txt")
        
        # Verificar test_loss salvo
        with open("app/artifacts/best_test_loss.txt", 'r') as f:
            saved_loss = float(f.read().strip())
        
        assert saved_loss == result["test_loss"]
        print(f"✅ Primeiro modelo salvo com test_loss: {saved_loss:.5f}")
    
    @pytest.mark.slow
    def test_better_model_replaces_previous(self):
        """Verifica se um modelo melhor substitui o anterior."""
        # Primeiro treinamento (pior)
        result1 = run_training_pipeline(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            epochs=2,  # Menos épocas = pior modelo
            batch_size=32,
            hidden_layer_size=16,
            num_layers=1,
            seed=42
        )
        
        first_loss = result1["test_loss"]
        
        # Segundo treinamento (melhor)
        result2 = run_training_pipeline(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            epochs=5,  # Mais épocas = melhor modelo
            batch_size=32,
            hidden_layer_size=32,
            num_layers=1,
            seed=42
        )
        
        second_loss = result2["test_loss"]
        
        # Verificar que segundo modelo é melhor
        if second_loss < first_loss:
            assert result2["is_best_model"] == True
            
            # Verificar que test_loss salvo é do segundo modelo
            with open("app/artifacts/best_test_loss.txt", 'r') as f:
                saved_loss = float(f.read().strip())
            
            assert saved_loss == second_loss
            print(f"✅ Modelo melhor substituiu anterior: {second_loss:.5f} < {first_loss:.5f}")
        else:
            print(f"ℹ️  Segundo modelo não foi melhor: {second_loss:.5f} >= {first_loss:.5f}")
    
    @pytest.mark.slow
    def test_worse_model_does_not_replace(self):
        """Verifica se um modelo pior NÃO substitui o melhor."""
        # Primeiro treinamento (melhor)
        result1 = run_training_pipeline(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            epochs=5,
            batch_size=32,
            hidden_layer_size=32,
            num_layers=1,
            seed=42
        )
        
        first_loss = result1["test_loss"]
        
        # Segundo treinamento (pior - seed diferente)
        result2 = run_training_pipeline(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            epochs=2,  # Menos épocas
            batch_size=32,
            hidden_layer_size=16,  # Menor arquitetura
            num_layers=1,
            seed=123  # Seed diferente
        )
        
        second_loss = result2["test_loss"]
        
        # Verificar test_loss salvo ainda é do primeiro modelo
        with open("app/artifacts/best_test_loss.txt", 'r') as f:
            saved_loss = float(f.read().strip())
        
        # Deve ser o menor dos dois
        assert saved_loss == min(first_loss, second_loss)
        
        if second_loss >= first_loss:
            assert result2["is_best_model"] == False
            assert saved_loss == first_loss
            print(f"✅ Modelo pior não substituiu melhor: {second_loss:.5f} >= {first_loss:.5f}")
    
    def test_get_best_model_info_no_model(self):
        """Verifica get_best_model_info quando não há modelo."""
        info = get_best_model_info()
        assert info is None
    
    @pytest.mark.slow
    def test_load_best_model(self):
        """Verifica se load_best_model funciona corretamente."""
        # Treinar primeiro
        result = run_training_pipeline(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            epochs=3,
            batch_size=32,
            hidden_layer_size=32,
            num_layers=1,
            seed=42
        )
        
        # Carregar melhor modelo
        model, best_loss = load_best_model(
            input_size=1,
            hidden_layer_size=32,
            output_size=1,
            num_layers=1,
            dropout=0.0
        )
        
        assert model is not None
        assert best_loss == result["test_loss"]
        assert isinstance(model, torch.nn.Module)
        print(f"✅ Melhor modelo carregado com test_loss: {best_loss:.5f}")


if __name__ == "__main__":
    print("Para executar os testes:")
    print("  pytest tests/test_best_model.py -v")
    print("  pytest tests/test_best_model.py -v -m slow")
