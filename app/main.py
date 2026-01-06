# ===========================
# app/main.py
# ===========================
import joblib
import os
import joblib
import torch
import sys
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
# Importa as rotas de pacientes e auditoria
from app.routes.audit_route import router as audit_router
from app.routes.train_route import router as train_router
from app.routes.predict_route import router as predict_router

import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega as configurações do projeto, incluindo as variáveis do modelo
load_dotenv()
__SETTINGS__ = get_settings()

# Variáveis globais para armazenar o modelo e o scaler
__model__ = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador de contexto para a API.
    Carrega o modelo na inicialização e pode liberar recursos no desligamento.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("INICIALIZANDO API DE PREDIÇÃO DE PREÇOS - TECH CHALLENGE FASE 4")
    logger.info("=" * 80)
    
    environment = os.getenv("ENVIRONMENT", "development")
    logger.info(f"Ambiente: {environment.upper()}")
    
    if environment == "production":
        logger.info("URLs em Produção:")
        logger.info("  - API Docs: /api/docs")
        logger.info("  - Landing Page: /")
    else:
        logger.info("URLs em Desenvolvimento:")
        logger.info("  - API: http://localhost:8000")
        logger.info("  - API Docs: http://localhost:8000/docs")
        logger.info("  - Streamlit (local opcional): http://localhost:8501")
    
    try:
        # Import necessário para instanciar o modelo
        sys.path.append(os.path.abspath("src"))
        from src.lstm_model import LSTMModel

        # Tenta baixar/carregar modelo do HuggingFace ou local
        # Se não existir, a API deve subir mesmo assim para permitir o treino
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Dispositivo de inferência selecionado: {device}")

            # Carregar configuração do modelo (se existir)
            model_config_path = "app/artifacts/model_config.json"
            if os.path.exists(model_config_path):
                import json
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                print(f"Configuração do modelo carregada: {model_config}")
            else:
                # Fallback para configuração padrão
                model_config = {
                    "input_size": 1,
                    "hidden_layer_size": 64,  # Atualizado para 64 (padrão do train.py)
                    "output_size": 1,
                    "num_layers": 2,
                    "dropout": 0.2
                }
                print("Usando configuração padrão do modelo")
            
            # Instancia o modelo com a configuração correta
            model = LSTMModel(
                input_size=model_config["input_size"],
                hidden_layer_size=model_config["hidden_layer_size"],
                output_size=model_config["output_size"],
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"]
            )
            model.to(device)
            
            # Primeiro tenta local
            local_model_path = "app/artifacts/lstm_model.pth"
            
            state_dict = None
            if os.path.exists(local_model_path):
                 print(f"Carregando modelo local de {local_model_path}...")
                 state_dict = torch.load(local_model_path, map_location=device)
            else:
                # Fallback para HuggingFace se configurado
                 print("Modelo local não encontrado. Tentando HuggingFace...")
                 model_path = hf_hub_download(repo_id=__SETTINGS__.MODEL_REPO_ID, filename=__SETTINGS__.MODEL_FILENAME)

                 loaded = joblib.load(model_path)
                 if isinstance(loaded, dict):
                     state_dict = loaded
                 else:
                     model = loaded
                     model.to(device)
            
            if state_dict:
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError as re:
                    print(f"ERRO DE COMPATIBILIDADE: O modelo salvo não corresponde à arquitetura atual ({re}). \n"
                          "Isso é esperado se você mudou a arquitetura (ex: adicionou camadas). \n"
                          "O modelo foi inicializado com pesos aleatórios. TREINE O MODELO NOVAMENTE VIA /train.")
            
            model.eval() # Coloca em modo de inferência
            __SETTINGS__.MODEL = model
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar o modelo ({e}). A API funcionará, mas /predict retornará erro até que o modelo seja treinado.")
            __SETTINGS__.MODEL = None

        # Tenta carregar o scaler localmente
        scaler_path = "app/artifacts/scaler.pkl"
        if os.path.exists(scaler_path):
            __SETTINGS__.SCALER = joblib.load(scaler_path)
            print("Scaler carregado com sucesso!")
        else:
            print(f"Aviso: Scaler não encontrado em {scaler_path}. Predições podem falhar.")
            __SETTINGS__.SCALER = None

    except Exception as e:
        print(f"Erro crítico no lifespan: {e}")
        __SETTINGS__.MODEL = None
        __SETTINGS__.SCALER = None
    
    yield
    print("API desligada. Recursos liberados.")


app = FastAPI(
    title=__SETTINGS__.PROJECT_NAME,
    description=(
        "## Tech Challenge Fase 4 - Predição de Preços de Ações com LSTM\n\n"
        "Bem-vindo ao desafio! Você aprendeu sobre aprendizado profundo com redes neurais "
        "Long Short Term Memory (LSTM) e chegou o momento de colocar isso em prática.\n\n"
        "### Objetivo\n"
        "Criar um modelo preditivo de redes neurais LSTM para predizer o valor de fechamento "
        "da bolsa de valores de uma empresa à sua escolha. Este projeto abrange toda a pipeline "
        "de desenvolvimento, desde a criação do modelo até o deploy em API.\n\n"
        "### Requisitos do Tech Challenge\n\n"
        "1. **Coleta e Pré-processamento dos Dados**\n"
        "   - Utiliza dataset de preços históricos de ações (Yahoo Finance via yfinance)\n"
        "   - Normalização e tratamento dos dados\n\n"
        "2. **Desenvolvimento do Modelo LSTM**\n"
        "   - Arquitetura com camada LSTM e camada linear\n"
        "   - Treinamento com otimização Adam\n"
        "   - Validação e avaliação com métricas (MAE, RMSE, MAPE)\n\n"
        "3. **API REST para Predições**\n"
        "   - Endpoints para predição de preços de ações\n"
        "   - Autenticação JWT para segurança\n"
        "   - Documentação interativa com Swagger UI\n\n"
        "### Tecnologias\n"
        "- **FastAPI**: Framework web moderno\n"
        "- **PyTorch**: Implementação do modelo LSTM\n"
        "- **MLflow**: Rastreamento de experimentos\n"
        "- **yfinance**: Coleta de dados de ações\n"
    ),
    version="1.0.0",
    root_path="/api" if os.getenv("ENVIRONMENT") == "production" else "",
    lifespan=lifespan
)


# Configurar CORS (mantido como estava)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audit_router)
app.include_router(train_router)
app.include_router(predict_router)

@app.get("/", tags=["Home"])
async def root():
    """
    Endpoint inicial da API de Predição de Preços de Ações.
    
    Fornece informações sobre o Tech Challenge e como acessar a documentação.
    """
    return {
        "message": "Bem-vindo ao Tech Challenge Fase 4 - Predição de Preços de Ações com LSTM!",
        "desafio": "Prever o valor de fechamento da bolsa de valores usando redes neurais LSTM",
        "info": "Registre-se e faça login para obter um token JWT e acessar os endpoints de predição",
        "documentação": "/docs",
        "versão": "1.0.0"
    }
