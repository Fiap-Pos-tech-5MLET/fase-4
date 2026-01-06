"""
Módulo para gerenciamento de seeds e reprodutibilidade.

Este módulo fornece funções para configurar seeds em todas as bibliotecas
relevantes (Python, NumPy, PyTorch) para garantir reprodutibilidade completa
nos experimentos de machine learning.
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Configura seed para reprodutibilidade em Python, NumPy e PyTorch.
    
    Esta função garante que todos os geradores de números aleatórios usem
    a mesma seed, tornando os experimentos reproduzíveis. Isso inclui:
    - Random do Python
    - NumPy random
    - PyTorch (CPU e CUDA)
    - Configurações de determinismo do CUDA
    
    Args:
        seed (int): Valor da seed a ser configurada. Padrão: 42
    
    Example:
        >>> from src.seed_manager import set_seed
        >>> set_seed(42)
        >>> # Agora todos os experimentos serão reproduzíveis
    
    Note:
        - Configurar determinismo no CUDA pode reduzir performance
        - Algumas operações do PyTorch podem não ser totalmente determinísticas
          mesmo com estas configurações em versões antigas
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random (CPU)
    torch.manual_seed(seed)
    
    # PyTorch random (CUDA/GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Para multi-GPU
    
    # Configurações de determinismo do PyTorch
    # Nota: Isso pode reduzir performance, mas garante reprodutibilidade
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Variável de ambiente para operações hash do Python
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_worker_seed(worker_id: int, base_seed: int = 42) -> int:
    """
    Gera seed única para workers do DataLoader.
    
    Quando usando múltiplos workers no DataLoader, cada worker precisa
    de uma seed diferente mas determinística para manter reprodutibilidade.
    
    Args:
        worker_id (int): ID do worker (fornecido pelo DataLoader)
        base_seed (int): Seed base para gerar seeds dos workers. Padrão: 42
    
    Returns:
        int: Seed única para o worker
    
    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
    """
    return base_seed + worker_id


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """
    Função de inicialização para workers do DataLoader.
    
    Esta função deve ser passada como `worker_init_fn` ao criar um DataLoader
    para garantir que cada worker tenha uma seed determinística.
    
    Args:
        worker_id (int): ID do worker (fornecido automaticamente pelo DataLoader)
        seed (int, optional): Seed base. Se None, usa 42. Padrão: None
    
    Example:
        >>> from functools import partial
        >>> from torch.utils.data import DataLoader
        >>> 
        >>> init_fn = partial(worker_init_fn, seed=42)
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=init_fn)
    """
    if seed is None:
        seed = 42
    
    worker_seed = get_worker_seed(worker_id, seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    # Teste básico
    print("Testando seed_manager...")
    
    # Teste 1: Verificar se seeds produzem mesmos resultados
    set_seed(42)
    random_values_1 = [random.random() for _ in range(5)]
    numpy_values_1 = np.random.rand(5)
    torch_values_1 = torch.rand(5)
    
    set_seed(42)
    random_values_2 = [random.random() for _ in range(5)]
    numpy_values_2 = np.random.rand(5)
    torch_values_2 = torch.rand(5)
    
    assert random_values_1 == random_values_2, "Python random não reproduzível"
    assert np.allclose(numpy_values_1, numpy_values_2), "NumPy random não reproduzível"
    assert torch.allclose(torch_values_1, torch_values_2), "PyTorch random não reproduzível"
    
    print("✅ Todos os testes passaram!")
    print(f"Python random: {random_values_1[:3]}")
    print(f"NumPy random: {numpy_values_1[:3]}")
    print(f"PyTorch random: {torch_values_1[:3]}")
