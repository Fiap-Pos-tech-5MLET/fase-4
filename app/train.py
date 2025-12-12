from app.data.data_loader import DataProcessor
from app.model.lstm_model import LSTMModel, ModelTrainer
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
import os
import numpy as np

import mlflow
import mlflow.pytorch

def run_training_pipeline(symbol='AAPL', start_date='2018-01-01', end_date='2024-07-20', epochs=50):
    mlflow.set_experiment("Stock_Price_Prediction")
    
    with mlflow.start_run():
        # Log Parameters
        mlflow.log_params({
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "epochs": epochs,
            "batch_size": 64,
            "learning_rate": 0.001,
            "hidden_units": 50
        })

        # 1. Load and Process Data
        processor = DataProcessor(symbol=symbol, start_date=start_date, end_date=end_date)
        try:
            X_train, y_train, X_test, y_test = processor.get_train_test_data()
        except ValueError as e:
            return {"error": str(e)}
        
        # Create DataLoaders
        batch_size = 64
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        
        # 2. Initialize Model
        model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
        trainer = ModelTrainer(model, lr=0.001)
        
        # 3. Train Model
        print(f"Starting Training for {symbol}...")
        loss_history = trainer.train(train_loader, epochs=epochs)
        
        # Log training loss
        for epoch, loss in enumerate(loss_history):
            mlflow.log_metric("train_loss", loss, step=epoch)
        
        # 4. Evaluate Model
        print("Evaluating Model...")
        predictions, actuals = trainer.evaluate(test_loader, processor.scaler)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Log Metrics
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })
        
        # 5. Save Artifacts
        os.makedirs("app/artifacts", exist_ok=True)
        trainer.save_model("app/artifacts/lstm_model.pth")
        joblib.dump(processor.scaler, "app/artifacts/scaler.pkl")
        print("Model and Scaler saved.")
        
        # Log Model to MLflow
        mlflow.pytorch.log_model(model, "lstm_model")
        
        return {
            "symbol": symbol,
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }

if __name__ == "__main__":
    run_training_pipeline()
