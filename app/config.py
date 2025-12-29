# ===========================
#  app/config.py
# ===========================
import os
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Any

# A classe Settings agora herda de BaseSettings.
# Pydantic lerá automaticamente as variáveis de ambiente com o mesmo nome.
class Settings(BaseSettings):
    """
    Configurações do projeto lidas a partir de variáveis de ambiente.
    """
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Stock Prediction API - LSTM")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    HTML_CACHE_DIR: str =  os.getenv("HTML_CACHE_DIR", "./cache")
    # Variáveis do modelo
    MODEL_REPO_ID: str =  os.getenv("MODEL_REPO_ID", "default-repo")
    MODEL_FILENAME: str =  os.getenv("MODEL_FILENAME", "lstm_model.pth")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    
    # Objetos em memória (não são carregados de env, mas setados na inicialização)
    MODEL: Any = None
    SCALER: Any = None

@lru_cache()
def get_settings() -> Settings:
    """
    Retorna uma única instância da classe Settings.
    O lru_cache garante que a função só será executada uma vez,
    tornando-a eficiente.
    """
    return Settings()