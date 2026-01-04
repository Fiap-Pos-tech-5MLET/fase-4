import torch
import torch.nn as nn
import os
import joblib
import numpy as np
import mlflow
import mlflow.pytorch
from typing import Dict, Tuple, List

from src.data_loader import DataProcessor
from src.lstm_model import LSTMModel
from src.utils import save_model
from src.evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset


class ModelTrainer:
    """
    Classe treinadora para modelos LSTM.
    
    Responsável pelo treinamento do modelo, gerenciamento do otimizador,
    critério de perda e dispositivo de computação (CPU/CUDA).
    
    Atributos:
        model (LSTMModel): Modelo LSTM a ser treinado.
        criterion (nn.MSELoss): Função de perda (Mean Squared Error).
        optimizer (torch.optim.Adam): Otimizador Adam.
        device (torch.device): Dispositivo de computação (CPU ou CUDA).
    """

    def __init__(self, model: LSTMModel, lr: float = 0.001) -> None:
        """
        Inicializa o treinador do modelo.
        
        Args:
            model (LSTMModel): Modelo LSTM a ser treinado.
            lr (float): Taxa de aprendizado. Padrão: 0.001
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, epochs: int = 10) -> List[float]:
        """
        Treina o modelo LSTM.
        
        Args:
            train_loader (DataLoader): DataLoader com dados de treinamento.
            epochs (int): Número de épocas de treinamento. Padrão: 10
        
        Returns:
            List[float]: Lista com o histórico de perdas por época.
        """
        self.model.train()
        loss_history = []
        for i in range(epochs):
            batch_loss = 0
            for seq, labels in train_loader:
                seq = seq.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(seq)

                single_loss = self.criterion(y_pred.squeeze(), labels)
                single_loss.backward()
                self.optimizer.step()
                batch_loss = single_loss.item()

            loss_history.append(batch_loss)

            if i % 5 == 0:
                print(f'Época: {i} Perda: {batch_loss:.5f}')
        return loss_history


def run_training_pipeline(
    symbol: str = 'AAPL',
    start_date: str = '2018-01-01',
    end_date: str = '2024-07-20',
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001
) -> Dict[str, float]:
    """
    Executa o pipeline completo de treinamento do modelo LSTM.
    
    Realiza as seguintes etapas:
    1. Carregamento e processamento de dados
    2. Criação de DataLoaders
    3. Inicialização e treinamento do modelo
    4. Avaliação do modelo
    5. Salvamento de artefatos
    6. Logging com MLflow
    
    Args:
        symbol (str): Símbolo da ação. Padrão: 'AAPL'
        start_date (str): Data de início (formato: YYYY-MM-DD). Padrão: '2018-01-01'
        end_date (str): Data de término (formato: YYYY-MM-DD). Padrão: '2024-07-20'
        epochs (int): Número de épocas de treinamento. Padrão: 50
        batch_size (int): Tamanho do lote. Padrão: 64
        learning_rate (float): Taxa de aprendizado. Padrão: 0.001
    
    Returns:
        Dict[str, float]: Dicionário com símbolo e métricas (MAE, RMSE, MAPE).
                         Em caso de erro, retorna dicionário com mensagem de erro.
    """
    mlflow.set_experiment("Stock_Price_Prediction")
    
    with mlflow.start_run():
        # Log de Parâmetros
        mlflow.log_params({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_units": 50
        })

        # 1. Carregamento e Processamento de Dados
        processor = DataProcessor(symbol=symbol, start_date=start_date, end_date=end_date)
        try:
            X_train, y_train, X_test, y_test = processor.get_train_test_data()
        except ValueError as e:
            return {"error": str(e)}
        
        # Criação de DataLoaders
        # batch_size usará o argumento
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        
        # 2. Inicialização do Modelo
        model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
        trainer = ModelTrainer(model, lr=learning_rate)
        
        # 3. Treinamento do Modelo
        print(f"Iniciando Treinamento para {symbol}...")
        loss_history = trainer.train(train_loader, epochs=epochs)
        
        # Log de perda de treinamento
        for epoch, loss in enumerate(loss_history):
            mlflow.log_metric("train_loss", loss, step=epoch)
        
        # 4. Avaliação do Modelo
        print("Avaliando Modelo...")
        predictions, actuals = evaluate_model(trainer.model, test_loader, processor.scaler, trainer.device)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Log de Métricas
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })
        
        # 5. Salvamento de Artefatos
        os.makedirs("app/artifacts", exist_ok=True)
        save_model(model, "app/artifacts/lstm_model.pth")
        joblib.dump(processor.scaler, "app/artifacts/scaler.pkl")
        print("Modelo e Scaler salvos.")
        
        # Log do Modelo no MLflow
        mlflow.pytorch.log_model(model, "lstm_model")
        
        return {
            "symbol": symbol,
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }

if __name__ == "__main__":
    run_training_pipeline()