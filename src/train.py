import torch
import torch.nn as nn
import os
import joblib
import numpy as np
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from src.data_loader import DataProcessor
from src.lstm_model import LSTMModel
from src.utils import save_model
from src.evaluate import evaluate_model, calculate_metrics, evaluate_with_loss
from src.seed_manager import set_seed
from torch.utils.data import DataLoader, TensorDataset


def plot_losses(train_losses: List[float], test_loss: float, save_path: str = "app/artifacts/loss_curves.png") -> str:
    """
    Plota as curvas de loss de treino e teste.
    
    Args:
        train_losses: Lista com loss de cada época de treinamento
        test_loss: Loss no conjunto de teste
        save_path: Caminho para salvar o gráfico
    
    Returns:
        Caminho do arquivo salvo
    """
    plt.figure(figsize=(10, 6))
    
    # Plotar train loss
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2)
    
    # Plotar test loss como linha horizontal
    plt.axhline(y=test_loss, color='r', linestyle='--', 
               label=f'Test Loss ({test_loss:.5f})', linewidth=2)
    
    # Configurações do gráfico
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Curvas de Loss - Treinamento LSTM', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Garantir que o diretório existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Salvar figura
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Fechar para liberar memória
    
    return save_path


def plot_predictions(actuals: np.ndarray, predictions: np.ndarray, symbol: str = "AAPL", num_points: int = 200, save_path: str = "app/artifacts/predictions.png") -> str:
    """
    Plota os valores reais vs previsões.
    
    Args:
        actuals: Array com valores reais
        predictions: Array com valores previstos
        symbol: Símbolo da ação (ex: AAPL, PETR4). Padrão: 'AAPL'
        num_points: Número de pontos a plotar (últimos N pontos). Padrão: 200
        save_path: Caminho para salvar o gráfico
    
    Returns:
        Caminho do arquivo salvo
    """
    # Plotar apenas os últimos num_points para melhor visualização
    if len(actuals) > num_points:
        actuals = actuals[-num_points:]
        predictions = predictions[-num_points:]
    
    plt.figure(figsize=(15, 6))
    
    # Plotar valores reais e previsões
    plt.plot(actuals, label='Preço Real', linewidth=2)
    plt.plot(predictions, label='Previsão', linewidth=2, alpha=0.7)
    
    # Configurações do gráfico
    plt.xlabel('Dias', fontsize=12)
    plt.ylabel('Preço (R$)', fontsize=12)
    plt.title(f'Previsão vs Preço Real - {symbol}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Garantir que o diretório existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Salvar figura
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Fechar para liberar memória
    
    return save_path




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
        import sys
        print(f"--- DEBUG INFO ---")
        print(f"Python Executable: {sys.executable}")
        print(f"Torch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"--- DEBUG INFO ---")
        print(f"ModelTrainer configurado para usar: {self.device}")
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, epochs: int = 10) -> List[float]:
        """
        Treina o modelo LSTM.
        
        Args:
            train_loader (DataLoader): DataLoader com dados de treinamento.
            epochs (int): Número de épocas de treinamento. Padrão: 10
        
        Returns:
            List[float]: Lista com o histórico de perdas médias por época.
        """
        self.model.train()
        loss_history = []
        
        for i in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for seq, labels in train_loader:
                seq = seq.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(seq)

                single_loss = self.criterion(y_pred.squeeze(), labels)
                single_loss.backward()
                self.optimizer.step()
                
                epoch_loss += single_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)

            print(f'Época: {i}/{epochs} Perda Média: {avg_loss:.5f}')   
        
        return loss_history


def run_training_pipeline(
    symbol: str = 'AAPL',
    start_date: str = '2018-01-01',
    end_date: str = '2026-01-05',
    epochs: int = 128,
    batch_size: int = 15,
    learning_rate: float = 0.001,
    num_layers: int = 1,
    dropout: float = 0.3,
    hidden_layer_size: int = 16,
    seed: int = 42
) -> Dict[str, float]:
    """
    Executa o pipeline completo de treinamento do modelo LSTM.
    
    Realiza as seguintes etapas:
    0. Configuração de seed para reprodutibilidade
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
        num_layers (int): Número de camadas LSTM. Padrão: 2
        dropout (float): Taxa de dropout. Padrão: 0.2
        hidden_layer_size (int): Tamanho da camada oculta. Padrão: 64
        seed (int): Seed para reprodutibilidade. Padrão: 42
    
    Returns:
        Dict[str, float]: Dicionário com símbolo e métricas (MAE, RMSE, MAPE).
                         Em caso de erro, retorna dicionário com mensagem de erro.
    """
    # 0. Configurar seed para reprodutibilidade
    set_seed(seed)
    print(f"Seed configurada para: {seed}")
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
            "hidden_layer": hidden_layer_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "seed": seed
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
        model = LSTMModel(input_size=1, hidden_layer_size=hidden_layer_size, output_size=1, num_layers=num_layers, dropout=dropout)
        trainer = ModelTrainer(model, lr=learning_rate)
        
        # 3. Treinamento do Modelo
        print(f"Iniciando Treinamento para {symbol}...")
        loss_history = trainer.train(train_loader, epochs=epochs)
        
        # Log de train_loss por época
        for epoch, loss in enumerate(loss_history):
            mlflow.log_metric("train_loss", loss, step=epoch)
        
        # 4. Avaliação do Modelo
        print("Avaliando Modelo...")
        predictions, actuals, test_loss = evaluate_with_loss(
            trainer.model, 
            test_loader, 
            processor.scaler, 
            trainer.device,
            trainer.criterion
        )
        
        print(f"Test Loss (MSE): {test_loss:.5f}")
        
        # Calcular métricas usando a função existente
        metrics = calculate_metrics(predictions, actuals)
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        mape = metrics["mape"]
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Log de Métricas
        mlflow.log_metrics({
            "test_loss": test_loss,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })
        
        # Plotar curvas de loss
        print("Gerando gráfico de curvas de loss...")
        loss_plot_path = plot_losses(
            train_losses=loss_history,
            test_loss=test_loss,
            save_path="app/artifacts/loss_curves.png"
        )
        print(f"Gráfico de loss salvo em: {loss_plot_path}")
        
        # Log do gráfico no MLflow
        mlflow.log_artifact(loss_plot_path)
        
        # Plotar predições vs valores reais
        print("Gerando gráfico de predições...")
        predictions_plot_path = plot_predictions(
            actuals=actuals.flatten(),
            predictions=predictions.flatten(),
            symbol=symbol,
            num_points=200,
            save_path="app/artifacts/predictions.png"
        )
        print(f"Gráfico de predições salvo em: {predictions_plot_path}")
        
        # Log do gráfico de predições no MLflow
        mlflow.log_artifact(predictions_plot_path)
        
        # 5. Salvamento de Artefatos
        os.makedirs("app/artifacts", exist_ok=True)
        
        # Verificar se é o melhor modelo baseado em test_loss
        best_model_path = "app/artifacts/best_stock_lstm_model.pth"
        best_loss_file = "app/artifacts/best_test_loss.txt"
        
        # Verificar se já existe um melhor modelo
        best_test_loss = float('inf')
        if os.path.exists(best_loss_file):
            try:
                with open(best_loss_file, 'r') as f:
                    best_test_loss = float(f.read().strip())
                print(f"Melhor test_loss anterior: {best_test_loss:.5f}")
            except:
                pass
        
        # Salvar APENAS se for o melhor modelo
        if test_loss < best_test_loss:
            print(f"✅ Novo melhor modelo! Test Loss: {test_loss:.5f} < {best_test_loss:.5f}")
            
            # Salvar como melhor modelo (backup)
            torch.save(model.state_dict(), best_model_path)
            with open(best_loss_file, 'w') as f:
                f.write(str(test_loss))
            
            # Salvar configuração do modelo (para carregar corretamente depois)
            model_config = {
                "input_size": 1,
                "hidden_layer_size": hidden_layer_size,
                "output_size": 1,
                "num_layers": num_layers,
                "dropout": dropout
            }
            import json
            with open("app/artifacts/model_config.json", 'w') as f:
                json.dump(model_config, f, indent=2)
            print("Configuração do modelo salva em: app/artifacts/model_config.json")
            
            # Salvar como modelo de produção (usado pela API)
            save_model(model, "app/artifacts/lstm_model.pth")
            joblib.dump(processor.scaler, "app/artifacts/scaler.pkl")
            print(f"Modelo de produção atualizado: app/artifacts/lstm_model.pth")
            print(f"Melhor modelo salvo em: {best_model_path}")
            
            # Log no MLflow que este é o melhor modelo
            mlflow.log_metric("is_best_model", 1.0)
            mlflow.log_artifact(best_model_path)
            mlflow.log_artifact("app/artifacts/model_config.json")
        else:
            print(f"ℹ️  Modelo atual não é o melhor. Test Loss: {test_loss:.5f} >= {best_test_loss:.5f}")
            print(f"   Mantendo modelo anterior em produção (test_loss: {best_test_loss:.5f})")
            mlflow.log_metric("is_best_model", 0.0)
        
        # Log do Modelo no MLflow (sempre loga o modelo atual, mesmo que não seja o melhor)
        mlflow.pytorch.log_model(model, "lstm_model")
        
        return {
            "symbol": symbol,
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "test_loss": float(test_loss),
            "is_best_model": test_loss < best_test_loss
        }

if __name__ == "__main__":
    run_training_pipeline()