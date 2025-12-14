# ===========================
# app/main.py
# ===========================
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
# Importa as rotas de pacientes e auditoria
from app.routes.audit_route import router as audit_router
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
    try:
        model_path = hf_hub_download(repo_id=__SETTINGS__.MODEL_REPO_ID, filename=__SETTINGS__.MODEL_FILENAME)
        __SETTINGS__.MODEL = joblib.load(model_path)
        print("Modelo carregado com sucesso na inicialização da API!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        __SETTINGS__.MODEL = None
    
    # O 'yield' é crucial! Ele permite que a API inicie e comece a processar requisições.
    yield
    # O código abaixo será executado quando a API for desligada.
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
