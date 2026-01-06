from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from app.schemas import TrainRequest, TrainResponse, TrainingJobStatus
import sys
import os
import uuid
from typing import Dict

# Store em memória para os jobs (Global)
# Em produção, use um banco de dados (Redis/Postgres)
JOBS: Dict[str, Dict] = {}

sys.path.append(os.path.abspath("src"))

try:
    from src.train import run_training_pipeline
except ImportError as e:
    print(f"Erro ao importar modulo de treino: {e}")

    def run_training_pipeline(**kwargs):
        raise ImportError("Não foi possível importar run_training_pipeline de src.train")

router = APIRouter(
    tags=["Treinamento"]
)

def train_model_task(job_id: str, request: TrainRequest):
    """
    Função wrapper para rodar o pipeline de treino e atualizar o status.
    """
    try:
        print(f"Iniciando job de treino {job_id} para {request.symbol}")
        # Atualiza status para rodando
        if job_id in JOBS:
            JOBS[job_id]["status"] = "running"
        
        result = run_training_pipeline(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            num_layers=request.num_layers,
            dropout=request.dropout,
            hidden_layer_size=request.hidden_layer_size
        )
        
        if job_id in JOBS:
            if "error" in result:
                 print(f"Job {job_id} falhou: {result['error']}")
                 JOBS[job_id]["status"] = "failed"
                 JOBS[job_id]["error"] = result["error"]
            else:
                 print(f"Job {job_id} completado com sucesso. Métricas: {result}")
                 JOBS[job_id]["status"] = "completed"
                 JOBS[job_id]["result"] = result

                 # === HOT RELOAD ===
                 try:
                     print("Iniciando Hot-Reload do modelo e scaler...")
                     import torch
                     import joblib
                     from app.config import get_settings
                     from src.lstm_model import LSTMModel
                     
                     settings = get_settings()
                     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                     
                     # Carrega Scaler
                     if os.path.exists("app/artifacts/scaler.pkl"):
                         settings.SCALER = joblib.load("app/artifacts/scaler.pkl")
                         print("Scaler recarregado.")
                     
                     # Carrega Modelo
                     if os.path.exists("app/artifacts/lstm_model.pth"):
                         # Instancia com os mesmos parâmetros do treino
                         new_model = LSTMModel(
                             input_size=1, 
                             hidden_layer_size=request.hidden_layer_size, 
                             output_size=1, 
                             num_layers=request.num_layers, 
                             dropout=request.dropout
                         )
                         new_model.to(device)
                         new_model.load_state_dict(torch.load("app/artifacts/lstm_model.pth", map_location=device))
                         new_model.eval()
                         settings.MODEL = new_model
                         print(f"Modelo recarregado com sucesso no dispositivo {device}.")
                         
                 except Exception as reload_error:
                     print(f"Erro no Hot-Reload: {reload_error}")
                     # Não falha o job, mas avisa
                     pass
             
    except Exception as e:
        print(f"Job {job_id} falhou com exceção: {e}")
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)

@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Dispara o treinamento do modelo LSTM em background.
    
    Este endpoint aceita parâmetros de treinamento, inicia o processo em segundo plano
    e retorna imediatamente um ID de job.
    """
    job_id = f"train-{uuid.uuid4()}"
    
    # Registra o job inicial
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(train_model_task, job_id, request)
    
    return {
        "message": "Treinamento iniciado em background",
        "job_id": job_id,
        "status": "pending"
    }

@router.get("/train/status/{job_id}", response_model=TrainingJobStatus)
async def get_training_status(job_id: str):
    """
    Retorna o status atual de um job de treinamento.
    """
    if job_id not in JOBS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} não encontrado."
        )
    
    return JOBS[job_id]
