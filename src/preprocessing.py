"""
Módulo de pré-processamento de dados para o modelo LSTM.

Fornece funções para preparar dados de entrada, normalização e criação
de DataLoaders para treinamento e teste.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências temporais para treinamento LSTM.
    
    Args:
        data (np.ndarray): Dados normalizados
        sequence_length (int): Tamanho da janela de sequência. Padrão: 60
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) onde X são as sequências e y são os targets
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    sequence_length: int = 60,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, MinMaxScaler]:
    """
    Prepara dados para treinamento do modelo LSTM.
    
    Normaliza os dados, cria sequências temporais e retorna DataLoaders.
    
    Args:
        df (pd.DataFrame): DataFrame com coluna 'Close' contendo preços
        test_size (float): Proporção de dados para teste (0.0 a 1.0). Padrão: 0.2
        sequence_length (int): Tamanho da janela temporal. Padrão: 60
        batch_size (int): Tamanho do batch para DataLoader. Padrão: 32
    
    Returns:
        Tuple[DataLoader, DataLoader, MinMaxScaler]: (train_loader, test_loader, scaler)
    
    Raises:
        ValueError: Se DataFrame estiver vazio ou não conter coluna 'Close'
    """
    if df.empty:
        raise ValueError("DataFrame está vazio")
    
    if 'Close' not in df.columns:
        raise ValueError("DataFrame deve conter coluna 'Close'")
    
    # Normalizar dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)
    
    # Dividir em treino e teste
    training_data_len = int(np.ceil(len(scaled_data) * (1 - test_size)))
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - sequence_length:, :]
    
    # Criar sequências
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    
    # Reshape para LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Converter para tensores PyTorch
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()
    
    # Criar Datasets e DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler
