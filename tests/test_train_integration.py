
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)

def test_train_route_triggers_background_task():
    """
    Testa se a rota /train aceita a requisição e retorna 202.
    Mocka a função de treino real para evitar execução pesada.
    """
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline:
        mock_pipeline.return_value = {"mae": 0.1, "rmse": 0.1, "mape": 0.1}
        
        payload = {
            "symbol": "TEST",
            "start_date": "2023-01-01",
            "end_date": "2023-02-01",
            "epochs": 1
        }
        
        response = client.post("/train", json=payload)
        
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

def test_predict_route_requires_scaler_and_model():
    """
    Testa a rota /predict.
    Como o ambiente de teste pode não ter o modelo carregado (depende do lifespan),
    esperamos sucesso (se mockado) ou erro 503 (se não mockado e sem arquivo).
    Vamos mockar o settings para garantir teste de sucesso.
    """
    # Create fake model and scaler
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(detach=lambda: MagicMock(numpy=lambda: [[0.5]], cpu=lambda: MagicMock(numpy=lambda: [[0.5]]))) # Mock output tensor
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_model.parameters.return_value = iter([mock_param])
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.5]] * 60
    mock_scaler.inverse_transform.return_value = [[150.0]]

    # We need to patch the __SETTINGS__ in predict_route module
    with patch("app.routes.predict_route.__SETTINGS__") as mock_settings:
        mock_settings.MODEL = mock_model
        mock_settings.SCALER = mock_scaler
        
        fake_prices = [100.0 + i for i in range(60)]
        payload = {"last_60_days_prices": fake_prices}
        
        response = client.post("/predict", json=payload)
        
        if response.status_code != 200:
            print(f"Predict Failed: {response.json()}")

        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        data = response.json()
        assert "predicted_price" in data
        assert data["predicted_price"] == 150.0

def test_predict_route_with_symbol():
    """
    Testa a rota /predict usando 'symbol'.
    Mockamos o yfinance para não depender da rede/API externa no teste unitário.
    """
    # Create fake model and scaler (same as above)
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(detach=lambda: MagicMock(numpy=lambda: [[0.5]], cpu=lambda: MagicMock(numpy=lambda: [[0.5]])))
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_model.parameters.return_value = iter([mock_param]) 
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.5]] * 60
    mock_scaler.inverse_transform.return_value = [[200.0]]

    # Mock yfinance
    with patch("yfinance.Ticker") as mock_ticker, \
         patch("yfinance.download") as mock_download, \
         patch("app.routes.predict_route.__SETTINGS__") as mock_settings:
        
        mock_settings.MODEL = mock_model
        mock_settings.SCALER = mock_scaler
        
        # Mock dataframe returned by yfinance
        import pandas as pd
        mock_df = pd.DataFrame({'Close': [100.0] * 70}) # More than 60
        mock_download.return_value = mock_df

        payload = {"symbol": "MOCK_AAPL"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_price"] == 200.0

        data = response.json()
        assert data["predicted_price"] == 200.0

def test_predict_route_with_dates():
    """
    Testa a rota /predict usando 'symbol' e 'end_date'.
    """
    # Reuse mocks setup logic or duplicate for simplicity in this script
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(detach=lambda: MagicMock(numpy=lambda: [[0.5]], cpu=lambda: MagicMock(numpy=lambda: [[0.5]])))
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_model.parameters.return_value = iter([mock_param]) 
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.5]] * 60
    mock_scaler.inverse_transform.return_value = [[200.0]]

    with patch("yfinance.download") as mock_download, \
         patch("app.routes.predict_route.__SETTINGS__") as mock_settings:
        
        mock_settings.MODEL = mock_model
        mock_settings.SCALER = mock_scaler
        
        import pandas as pd
        mock_df = pd.DataFrame({'Close': [100.0] * 100}) 
        mock_download.return_value = mock_df

        # Test request with end_date
        payload = {"symbol": "MOCK_DATE", "end_date": "2023-01-01"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        # Verify if yfinance was called with data range logic (start calculated)
        args, kwargs = mock_download.call_args
        assert kwargs['end'] == pd.to_datetime("2023-01-01") # or string dep on pd ver
        
        print("Date range logic verified in mock.")

        print("Date range logic verified in mock.")

def test_training_status_flow():
    """
    Testa o fluxo completo: Disparar treino -> Pegar ID -> Consultar Status.
    """
    """
    Testa o fluxo completo: Disparar treino -> Pegar ID -> Consultar Status.
    """
    with patch("app.routes.train_route.run_training_pipeline") as mock_pipeline:
        mock_pipeline.return_value = {"mae": 0.1}
        
        # 1. Trigger
        payload = {"symbol": "TEST_STATUS", "epochs": 1}
        res_train = client.post("/train", json=payload)
        assert res_train.status_code == 202
        job_id = res_train.json()["job_id"]
        
        # 2. Check Status (Initial/Running)
        # Nota: Como é background task, pode já ter acabado ou estar rodando.
        # No ambiente de teste do TestClient, as background tasks rodam síncronas após retorno?
        # NÃO, TestClient roda background tasks DEPOIS do retorno da resposta.
        # Então aqui deve estar pending ou já completed dependendo da implementação do StarletteTestClient.
        # Vamos checar se existe.
        res_status = client.get(f"/train/status/{job_id}")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert status_data["job_id"] == job_id
        # Deve ter completado pois é síncrono no mock simples ou TestClient espera
        print(f"Job Status: {status_data['status']}")

def test_training_status_not_found():
    """
    Testa erro 404 para job inexistente.
    """
    res = client.get("/train/status/fake-id-123")
    assert res.status_code == 404


if __name__ == "__main__":
    try:
        test_train_route_triggers_background_task()
        print("[PASS] test_train_route_triggers_background_task")
        test_predict_route_requires_scaler_and_model()
        print("[PASS] test_predict_route_requires_scaler_and_model")
        test_predict_route_with_symbol()
        print("[PASS] test_predict_route_with_symbol")
        test_predict_route_with_dates()
        print("[PASS] test_predict_route_with_dates")
        test_training_status_flow()
        print("[PASS] test_training_status_flow")
        test_training_status_not_found()
        print("[PASS] test_training_status_not_found")
    except AssertionError as e:
        print(f"[FAIL] TEST FAILED: {e}")
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
