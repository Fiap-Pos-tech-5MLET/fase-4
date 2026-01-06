from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional

class TrainRequest(BaseModel):
    symbol: str = Field(default="AAPL", description="Símbolo da ação para treinamento (ex: AAPL, MSFT)")
    start_date: str = Field(default="2018-01-01", description="Data de início dos dados (YYYY-MM-DD)")
    end_date: str = Field(default="2024-07-20", description="Data de fim dos dados (YYYY-MM-DD)")
    epochs: int = Field(default=50, ge=1, description="Número de épocas de treinamento")
    batch_size: int = Field(default=64, ge=1, description="Tamanho do lote (batch size)")
    learning_rate: float = Field(default=0.001, gt=0, description="Taxa de aprendizado (learning rate)")
    num_layers: int = Field(default=2, ge=1, description="Número de camadas LSTM")
    dropout: float = Field(default=0.2, ge=0.0, le=1.0, description="Taxa de dropout")
    hidden_layer_size: int = Field(default=64, ge=0, description="Tamanho da camada oculta")

class TrainResponse(BaseModel):
    message: str
    job_id: str
    status: str

from typing import Dict, Any
class TrainingJobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="Status atual do job: pending, running, completed, failed")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

class PredictRequest(BaseModel):
    symbol: Optional[str] = Field(None, description="Símbolo da ação (ex: AAPL). Se fornecido, busca dados automaticamente.")
    last_60_days_prices: Optional[List[float]] = Field(None, min_items=60, max_items=60, description="Lista com exatamente 60 preços de fechamento. Opcional se symbol for fornecido.")
    start_date: Optional[str] = Field(None, description="Data inicial para busca automática (YYYY-MM-DD). Opcional.")
    end_date: Optional[str] = Field(None, description="Data final para busca automática (YYYY-MM-DD). Opcional.")

    @model_validator(mode='after')
    def check_data_source(self):
        if not self.symbol and not self.last_60_days_prices:
            raise ValueError('Deve fornecer ou "symbol" (para busca automática) ou "last_60_days_prices" (manual).')
        return self

class PredictResponse(BaseModel):
    predicted_price: float
    timestamp: str
