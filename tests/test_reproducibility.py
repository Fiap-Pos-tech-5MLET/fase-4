"""
Testes de reprodutibilidade do treinamento.

Verifica se o sistema de seeds garante resultados idênticos
quando o treinamento é executado com os mesmos parâmetros.
"""

import pytest
import torch
import numpy as np
from src.seed_manager import set_seed, worker_init_fn, get_worker_seed
from src.train import run_training_pipeline
from src.lstm_model import LSTMModel


class TestSeedManager:
    """Testes para o módulo seed_manager."""
    
    def test_set_seed_python_random(self):
        """Verifica se set_seed torna random do Python reproduzível."""
        import random
        
        set_seed(42)
        values1 = [random.random() for _ in range(10)]
        
        set_seed(42)
        values2 = [random.random() for _ in range(10)]
        
        assert values1 == values2, "Python random não é reproduzível"
    
    def test_set_seed_numpy(self):
        """Verifica se set_seed torna NumPy reproduzível."""
        set_seed(42)
        values1 = np.random.rand(10)
        
        set_seed(42)
        values2 = np.random.rand(10)
        
        assert np.allclose(values1, values2), "NumPy random não é reproduzível"
    
    def test_set_seed_pytorch(self):
        """Verifica se set_seed torna PyTorch reproduzível."""
        set_seed(42)
        values1 = torch.rand(10)
        
        set_seed(42)
        values2 = torch.rand(10)
        
        assert torch.allclose(values1, values2), "PyTorch random não é reproduzível"
    
    def test_set_seed_model_initialization(self):
        """Verifica se pesos do modelo são reproduzíveis."""
        set_seed(42)
        model1 = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
        weights1 = model1.lstm.weight_ih_l0.data.clone()
        
        set_seed(42)
        model2 = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
        weights2 = model2.lstm.weight_ih_l0.data.clone()
        
        assert torch.allclose(weights1, weights2), "Pesos do modelo não são reproduzíveis"
    
    def test_worker_seed_generation(self):
        """Verifica se seeds de workers são únicas mas determinísticas."""
        base_seed = 42
        
        # Gerar seeds para 4 workers
        seeds1 = [get_worker_seed(i, base_seed) for i in range(4)]
        seeds2 = [get_worker_seed(i, base_seed) for i in range(4)]
        
        # Devem ser idênticas entre execuções
        assert seeds1 == seeds2, "Seeds de workers não são determinísticas"
        
        # Devem ser únicas
        assert len(set(seeds1)) == len(seeds1), "Seeds de workers não são únicas"
    
    def test_worker_init_fn(self):
        """Verifica se worker_init_fn configura seeds corretamente."""
        import random
        
        # Inicializar worker 0
        worker_init_fn(0, seed=42)
        values1 = [random.random() for _ in range(5)]
        
        # Reinicializar worker 0
        worker_init_fn(0, seed=42)
        values2 = [random.random() for _ in range(5)]
        
        assert values1 == values2, "worker_init_fn não é reproduzível"


class TestTrainingReproducibility:
    """Testes de reprodutibilidade do pipeline de treinamento."""
    
    @pytest.mark.slow
    def test_training_reproducibility_same_seed(self):
        """
        Verifica se dois treinamentos com mesma seed produzem resultados idênticos.
        
        IMPORTANTE: Este teste pode demorar alguns minutos para executar.
        """
        # Parâmetros de teste (reduzidos para velocidade)
        params = {
            "symbol": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
            "epochs": 5,  # Reduzido para teste rápido
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_layers": 1,  # Reduzido para teste rápido
            "dropout": 0.0,
            "hidden_layer_size": 32,  # Reduzido para teste rápido
            "seed": 42
        }
        
        # Primeiro treinamento
        result1 = run_training_pipeline(**params)
        
        # Segundo treinamento com mesma seed
        result2 = run_training_pipeline(**params)
        
        # Verificar se não houve erros
        assert "error" not in result1, f"Primeiro treinamento falhou: {result1.get('error')}"
        assert "error" not in result2, f"Segundo treinamento falhou: {result2.get('error')}"
        
        # Verificar reprodutibilidade das métricas
        # Usamos tolerância muito pequena pois devem ser idênticas
        assert abs(result1["mae"] - result2["mae"]) < 1e-6, \
            f"MAE não reproduzível: {result1['mae']} vs {result2['mae']}"
        
        assert abs(result1["rmse"] - result2["rmse"]) < 1e-6, \
            f"RMSE não reproduzível: {result1['rmse']} vs {result2['rmse']}"
        
        assert abs(result1["mape"] - result2["mape"]) < 1e-6, \
            f"MAPE não reproduzível: {result1['mape']} vs {result2['mape']}"
        
        print(f"✅ Treinamento reproduzível!")
        print(f"   MAE: {result1['mae']:.6f} == {result2['mae']:.6f}")
        print(f"   RMSE: {result1['rmse']:.6f} == {result2['rmse']:.6f}")
        print(f"   MAPE: {result1['mape']:.6f} == {result2['mape']:.6f}")
    
    @pytest.mark.slow
    def test_training_different_seeds_produce_different_results(self):
        """
        Verifica se seeds diferentes produzem resultados diferentes.
        
        Isso garante que o sistema de seeds está realmente funcionando.
        """
        # Parâmetros de teste
        params_base = {
            "symbol": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_layers": 1,
            "dropout": 0.0,
            "hidden_layer_size": 32,
        }
        
        # Treinamento com seed 42
        result1 = run_training_pipeline(**params_base, seed=42)
        
        # Treinamento com seed 123
        result2 = run_training_pipeline(**params_base, seed=123)
        
        # Verificar se não houve erros
        assert "error" not in result1, f"Primeiro treinamento falhou: {result1.get('error')}"
        assert "error" not in result2, f"Segundo treinamento falhou: {result2.get('error')}"
        
        # Verificar que os resultados são DIFERENTES
        # (pelo menos uma métrica deve ser diferente)
        mae_different = abs(result1["mae"] - result2["mae"]) > 1e-3
        rmse_different = abs(result1["rmse"] - result2["rmse"]) > 1e-3
        mape_different = abs(result1["mape"] - result2["mape"]) > 1e-3
        
        assert mae_different or rmse_different or mape_different, \
            "Seeds diferentes não produziram resultados diferentes"
        
        print(f"✅ Seeds diferentes produzem resultados diferentes!")
        print(f"   Seed 42  - MAE: {result1['mae']:.6f}, RMSE: {result1['rmse']:.6f}")
        print(f"   Seed 123 - MAE: {result2['mae']:.6f}, RMSE: {result2['rmse']:.6f}")


if __name__ == "__main__":
    # Executar testes básicos
    print("Executando testes de seed_manager...")
    
    test_sm = TestSeedManager()
    test_sm.test_set_seed_python_random()
    print("✅ Python random reproduzível")
    
    test_sm.test_set_seed_numpy()
    print("✅ NumPy reproduzível")
    
    test_sm.test_set_seed_pytorch()
    print("✅ PyTorch reproduzível")
    
    test_sm.test_set_seed_model_initialization()
    print("✅ Inicialização de modelo reproduzível")
    
    print("\n✅ Todos os testes básicos passaram!")
    print("\nPara executar testes completos (incluindo treinamento):")
    print("  pytest tests/test_reproducibility.py -v")
    print("  pytest tests/test_reproducibility.py -v -m slow  # Apenas testes lentos")
