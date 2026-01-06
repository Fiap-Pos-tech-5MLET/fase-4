import torch
import numpy as np
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Avalia o modelo LSTM em dados de teste.
    
    Realiza a predição do modelo em dados de teste e rescala os valores
    para a escala original usando o scaler fornecido.
    
    Args:
        model (torch.nn.Module): Modelo LSTM treinado.
        test_loader (DataLoader): DataLoader com dados de teste.
        scaler (MinMaxScaler): Scaler para inverter a normalização das predições.
        device (torch.device): Dispositivo de computação (CPU ou CUDA).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tupla contendo:
            - predictions_rescaled: Predições do modelo rescaladas para escala original.
            - actuals_rescaled: Valores reais rescalados para escala original.
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for seq, labels in test_loader:
            seq = seq.to(device)
            labels = labels.to(device)  # Mover labels para o mesmo device
            y_pred = model(seq)
            predictions.append(y_pred.cpu().numpy().flatten())
            actuals.append(labels.cpu().numpy().flatten())
    
    # Concatenar e inverter a escala
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    actuals_rescaled = scaler.inverse_transform(actuals.reshape(-1, 1))
    
    return predictions_rescaled, actuals_rescaled


def evaluate_with_loss(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device,
    criterion: torch.nn.Module
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Avalia o modelo LSTM em dados de teste e calcula a perda.
    
    Realiza a predição do modelo em dados de teste, calcula a perda média
    e rescala os valores para a escala original usando o scaler fornecido.
    
    Args:
        model (torch.nn.Module): Modelo LSTM treinado.
        test_loader (DataLoader): DataLoader com dados de teste.
        scaler (MinMaxScaler): Scaler para inverter a normalização das predições.
        device (torch.device): Dispositivo de computação (CPU ou CUDA).
        criterion (torch.nn.Module): Função de perda (ex: MSELoss).
    
    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Tupla contendo:
            - predictions_rescaled: Predições do modelo rescaladas para escala original.
            - actuals_rescaled: Valores reais rescalados para escala original.
            - avg_loss: Perda média no conjunto de teste.
    """
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for seq, labels in test_loader:
            seq = seq.to(device)
            labels = labels.to(device)
            
            y_pred = model(seq)
            
            # Calcular perda
            loss = criterion(y_pred.squeeze(), labels)
            total_loss += loss.item()
            num_batches += 1
            
            predictions.append(y_pred.cpu().numpy().flatten())
            actuals.append(labels.cpu().numpy().flatten())
    
    # Calcular perda média
    avg_loss = total_loss / num_batches
    
    # Concatenar e inverter a escala
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    actuals_rescaled = scaler.inverse_transform(actuals.reshape(-1, 1))
    
    return predictions_rescaled, actuals_rescaled, avg_loss



def calculate_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, float]:
    """
    Calcula as métricas de avaliação do modelo.
    
    Computa as seguintes métricas:
    - MAE (Mean Absolute Error): Erro Absoluto Médio
    - RMSE (Root Mean Squared Error): Raiz do Erro Quadrático Médio
    - MAPE (Mean Absolute Percentage Error): Erro Percentual Absoluto Médio
    
    Args:
        predictions (np.ndarray): Array com as predições do modelo.
        actuals (np.ndarray): Array com os valores reais.
    
    Returns:
        Dict[str, float]: Dicionário contendo as métricas calculadas:
            - mae: Erro Absoluto Médio
            - rmse: Raiz do Erro Quadrático Médio
            - mape: Erro Percentual Absoluto Médio (%)
    """
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
