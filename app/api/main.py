from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import numpy as np
import os
from app.model.lstm_model import LSTMModel

from app.train import run_training_pipeline
from fastapi import BackgroundTasks

app = FastAPI(title="Stock Price Prediction API", version="1.0")

# Global variables for model and scaler
model = None
scaler = None
model_path = "app/artifacts/lstm_model.pth"
scaler_path = "app/artifacts/scaler.pkl"

class PredictionRequest(BaseModel):
    last_60_days_prices: list[float]

class PredictionResponse(BaseModel):
    predicted_price: float

class TrainRequest(BaseModel):
    symbol: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: str = "2024-07-20"
    epochs: int = 50

@app.on_event("startup")
async def load_artifacts():
    global model, scaler
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        # Load Scaler
        scaler = joblib.load(scaler_path)
        
        # Load Model
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model and Scaler loaded successfully.")
    else:
        print("Artifacts not found. Proceeding without model (endpoints may fail).")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/train")
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    def training_task(symbol, start, end, epochs):
        result = run_training_pipeline(symbol, start, end, epochs)
        print(f"Training completed: {result}")
        # Reload model after training
        global model, scaler
        scaler = joblib.load(scaler_path)
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model reloaded.")

    background_tasks.add_task(training_task, request.symbol, request.start_date, request.end_date, request.epochs)
    return {"message": "Training started in background", "params": request}
    
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or scaler is None:
         raise HTTPException(status_code=503, detail="Model is not trained yet.")
         
    if len(request.last_60_days_prices) != 60:
        raise HTTPException(status_code=400, detail="Must provide exactly 60 days of closing prices.")
    
    try:
        # Preprocess
        input_data = np.array(request.last_60_days_prices).reshape(-1, 1)
        scaled_data = scaler.transform(input_data)
        
        # Create sequence [1, 60, 1]
        seq = torch.from_numpy(scaled_data).float().unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prediction = model(seq)
            
        # Inverse transform
        predicted_price_scaled = prediction.numpy().reshape(-1, 1)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
        
        return {"predicted_price": float(predicted_price)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
