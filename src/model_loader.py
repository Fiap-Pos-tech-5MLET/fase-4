"""
Funções utilitárias para carregar o melhor modelo treinado.

Este módulo fornece funções para carregar o melhor modelo salvo
durante o treinamento, baseado na menor test_loss.
"""

import torch
import os
from typing import Optional, Tuple
from src.lstm_model import LSTMModel


def load_best_model(
    model_path: str = "app/artifacts/best_stock_lstm_model.pth",
    device: Optional[torch.device] = None,
    **model_params
) -> Tuple[LSTMModel, float]:
    """
    Carrega o melhor modelo salvo durante o treinamento.
    
    Args:
        model_path (str): Caminho para o arquivo do melhor modelo
        device (torch.device, optional): Dispositivo para carregar o modelo
        **model_params: Parâmetros do modelo (input_size, hidden_layer_size, etc.)
    
    Returns:
        Tuple[LSTMModel, float]: Modelo carregado e melhor test_loss
    
    Raises:
        FileNotFoundError: Se o modelo não existir
    
    Example:
        >>> model, best_loss = load_best_model(
        ...     input_size=1,
        ...     hidden_layer_size=64,
        ...     output_size=1,
        ...     num_layers=2,
        ...     dropout=0.2
        ... )
        >>> print(f"Melhor test_loss: {best_loss:.5f}")
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Melhor modelo não encontrado em {model_path}. "
            "Execute o treinamento primeiro."
        )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar melhor test_loss
    best_loss_file = model_path.replace('.pth', '_loss.txt')
    if model_path == "app/artifacts/best_stock_lstm_model.pth":
        best_loss_file = "app/artifacts/best_test_loss.txt"
    
    best_test_loss = None
    if os.path.exists(best_loss_file):
        with open(best_loss_file, 'r') as f:
            best_test_loss = float(f.read().strip())
    
    # Criar modelo com parâmetros fornecidos
    model = LSTMModel(**model_params)
    
    # Carregar state_dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Melhor modelo carregado de: {model_path}")
    if best_test_loss is not None:
        print(f"   Test Loss: {best_test_loss:.5f}")
    
    return model, best_test_loss


def get_best_model_info() -> Optional[dict]:
    """
    Retorna informações sobre o melhor modelo salvo.
    
    Returns:
        dict: Informações do melhor modelo ou None se não existir
    
    Example:
        >>> info = get_best_model_info()
        >>> if info:
        ...     print(f"Melhor test_loss: {info['test_loss']}")
    """
    best_model_path = "app/artifacts/best_stock_lstm_model.pth"
    best_loss_file = "app/artifacts/best_test_loss.txt"
    
    if not os.path.exists(best_model_path):
        return None
    
    info = {
        "model_path": best_model_path,
        "exists": True,
        "test_loss": None
    }
    
    if os.path.exists(best_loss_file):
        try:
            with open(best_loss_file, 'r') as f:
                info["test_loss"] = float(f.read().strip())
        except:
            pass
    
    return info


if __name__ == "__main__":
    # Teste
    info = get_best_model_info()
    if info:
        print("Informações do melhor modelo:")
        print(f"  Caminho: {info['model_path']}")
        print(f"  Test Loss: {info['test_loss']}")
    else:
        print("Nenhum melhor modelo encontrado. Execute o treinamento primeiro.")
