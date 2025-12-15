import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

class DataProcessor:
    def __init__(self, symbol='AAPL', start_date='2018-01-01', end_date='2024-07-20', sequence_length=60):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None

    def download_data(self):
        """Downloads historical data from Yahoo Finance."""
        print(f"Downloading data for {self.symbol}...")
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        if self.data.empty:
            raise ValueError("No data found for the given symbol/date range.")
        # Ensure we only use the 'Close' column
        self.data = self.data[['Close']]
        print(f"Data downloaded: {len(self.data)} rows.")

    def preprocess_data(self):
        """Scales the data using MinMaxScaler."""
        if self.data is None:
            self.download_data()
        
        dataset = self.data.values
        self.scaled_data = self.scaler.fit_transform(dataset)
        return self.scaled_data

    def create_sequences(self, data):
        """Creates sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def get_train_test_data(self, split_ratio=0.8):
        """Splits data into train and test sets and converts to PyTorch tensors."""
        if self.scaled_data is None:
            self.preprocess_data()

        training_data_len = int(np.ceil(len(self.scaled_data) * split_ratio))
        train_data = self.scaled_data[0:training_data_len, :]
        test_data = self.scaled_data[training_data_len - self.sequence_length:, :]

        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)

        # Reshape for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def load_data(
    symbol: str = 'AAPL',
    start_date: str = '2018-01-01',
    end_date: str = '2024-07-20'
) -> pd.DataFrame:
    """
    Carrega dados históricos de ações do Yahoo Finance.
    
    Args:
        symbol (str): Símbolo da ação (ex: AAPL, GOOGL, MSFT)
        start_date (str): Data inicial no formato YYYY-MM-DD
        end_date (str): Data final no formato YYYY-MM-DD
    
    Returns:
        pd.DataFrame: DataFrame com dados históricos (coluna 'Close')
    
    Raises:
        ValueError: Se não houver dados para o símbolo/período especificado
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for symbol {symbol} in date range {start_date} to {end_date}")
    return data[['Close']]


if __name__ == "__main__":
    # Simple test
    processor = DataProcessor()
    X_train, y_train, X_test, y_test = processor.get_train_test_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
