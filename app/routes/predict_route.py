from fastapi import APIRouter, HTTPException, status
from app.schemas import PredictRequest, PredictResponse
from app.config import get_settings
from datetime import datetime
import torch
import numpy as np

router = APIRouter(
    tags=["Predição"]
)

__SETTINGS__ = get_settings()

@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_stock_price(request: PredictRequest):
    """
    Prevê o próximo preço de fechamento com base nos últimos 60 dias.
    """
    model = __SETTINGS__.MODEL
    scaler = getattr(__SETTINGS__, "SCALER", None)
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo ainda não foi carregado. Tente novamente mais tarde."
        )
    
    if scaler is None:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scaler de normalização não está disponível. Realize o treinamento primeiro."
        )

    try:
        # Lógica Híbrida: Busca automática ou Uso de dados manuais
        prices_data = []

        if request.symbol:
            import yfinance as yf
            print(f"Buscando dados automáticos para: {request.symbol}")
            
            # Definição do período de busca
            if request.start_date and request.end_date:
                # Se ambas as datas fornecidas, usa intervalo explícito
                # Adicionamos uma margem de segurança no start_date se o usuário não der
                # Mas aqui confiamos que o usuário quer este intervalo.
                # Para garantir 60 dias, o start_date precisaria ser bem anterior ao end_date.
                # O ideal é usar o end_date como referência e pegar para trás.
                # Se o usuário passar start/end fixos, pode não ter 60 dias.
                # ESTRETÉGIA: Usar download com start/end fornecidos.
                data = yf.download(request.symbol, start=request.start_date, end=request.end_date, progress=False)
            elif request.end_date:
                # Se só end_date, pegamos um periodo longo para trás
                import pandas as pd
                end_dt = pd.to_datetime(request.end_date)
                start_dt = end_dt - pd.Timedelta(days=150) # ~5 meses para garantir 60 dias úteis
                data = yf.download(request.symbol, start=start_dt, end=end_dt, progress=False)
            else:
                 # Default: últimos 6 meses até hoje
                 data = yf.download(request.symbol, period="6mo", progress=False)
            
            if len(data) < 60:
                 raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Não há dados históricos suficientes para {request.symbol} no período solicitado. Encontrado: {len(data)}, Necessário: 60."
                )
            
            # Pega os últimos 60 fechamentos
            # O yfinance pode retornar MultiIndex ou Series, garantindo pegar 'Close'
            if 'Close' in data.columns:
                prices_data = data['Close'].iloc[-60:].values.flatten().tolist()
            else:
                 # Fallback se estrutura for diferente (ex: apenas series)
                 prices_data = data.iloc[-60:].values.flatten().tolist()
                 
            print(f"Dados recuperados: {len(prices_data)} registros.")

        elif request.last_60_days_prices:
            prices_data = request.last_60_days_prices
        
        # Garante que temos dados
        if not prices_data or len(prices_data) != 60:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dados de entrada inválidos. Esperados 60 preços, obtidos {len(prices_data)}."
            )

        # Prepara os dados
        input_data = np.array(prices_data).reshape(-1, 1)
        
        # Normaliza
        normalized_data = scaler.transform(input_data)
        
        # Cria tensor (Batch size 1, Sequence Length 60, Features 1)
        input_tensor = torch.FloatTensor(normalized_data).view(1, 60, 1)
        
        # Predição
        model.eval()
        with torch.no_grad():
            predicted_scaled = model(input_tensor)
            
        # Desnormaliza
        predicted_price = scaler.inverse_transform(predicted_scaled.numpy())[0][0]
        
        return {
            "predicted_price": float(predicted_price),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar predição: {str(e)}"
        )
