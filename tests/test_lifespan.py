import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from app.main import lifespan
from app.config import Settings

@pytest.fixture
def mock_app():
    return FastAPI()

@pytest.mark.asyncio
async def test_lifespan_load_local_success(mock_app):
    """Testa carregamento bem-sucedido de modelo e scaler locais."""
    from unittest.mock import mock_open
    import json
    
    # Mock config data
    mock_config = {
        "input_size": 1,
        "hidden_layer_size": 64,
        "output_size": 1,
        "num_layers": 2,
        "dropout": 0.2
    }
    
    with patch("app.main.os.path.exists") as mock_exists, \
         patch("app.main.torch.load") as mock_torch_load, \
         patch("app.main.joblib.load") as mock_joblib_load, \
         patch("app.main.hf_hub_download") as mock_hf_download, \
         patch("builtins.open", mock_open(read_data=json.dumps(mock_config))), \
         patch("app.main.__SETTINGS__") as mock_settings:
        
        # Simula existência de arquivos locais (config, modelo E scaler)
        # Retorna True para todos os arquivos
        mock_exists.return_value = True
        
        # Simula state_dict carregado via torch.load
        mock_dict = {"state": "dict"}
        mock_torch_load.return_value = mock_dict
        
        # Simula scaler carregado via joblib.load
        mock_scaler = MagicMock()
        mock_joblib_load.return_value = mock_scaler
        
        # Simula instanciacao do modelo
        with patch("src.lstm_model.LSTMModel") as MockModel:
            mock_instance = MockModel.return_value
            
            async with lifespan(mock_app):
                # Verifica se tentou carregar model.pth via torch
                mock_torch_load.assert_called_once()
                
                # Verifica se carregou state_dict no modelo
                mock_instance.load_state_dict.assert_called_once_with(mock_dict)
                
                # Verifica se scaler foi carregado via joblib
                mock_joblib_load.assert_called_once()
                
                # Não deve ter tentado baixar do HF
                mock_hf_download.assert_not_called()
                
@pytest.mark.asyncio
async def test_lifespan_load_hf_fallback(mock_app):
    """Testa fallback para Hugging Face quando local não existe."""
    with patch("app.main.os.path.exists") as mock_exists, \
         patch("app.main.hf_hub_download") as mock_hf, \
         patch("app.main.joblib.load") as mock_joblib_load, \
         patch("app.main.__SETTINGS__") as mock_settings:
        
        # path.exists retorna False para modelo (primeira chamada) e True para scaler
        def side_effect(path):
            if "lstm_model.pth" in path: return False
            if "scaler.pkl" in path: return True
            return False
        mock_exists.side_effect = side_effect
        
        # Mock HF download
        mock_hf.return_value = "downloaded_path.pth"
        
        # Mock load returning dict (state_dict) via joblib (legacy path)
        mock_joblib_load.return_value = {"state": "dict"}
        
        with patch("src.lstm_model.LSTMModel") as MockModel:
            mock_instance = MockModel.return_value
            
            async with lifespan(mock_app):
                mock_hf.assert_called()
                mock_instance.load_state_dict.assert_called()

@pytest.mark.asyncio
async def test_lifespan_load_fail_gracefully(mock_app):
    """Testa inicialização sem falha quando modelos não são encontrados."""
    with patch("app.main.os.path.exists", return_value=False), \
         patch("app.main.hf_hub_download", side_effect=Exception("HF Error")), \
         patch("app.main.__SETTINGS__") as mock_settings:
         
         with patch("src.lstm_model.LSTMModel"): # Mock para não falhar na instanciação
             async with lifespan(mock_app):
                 # Deve capturar exceção e setar None
                 assert mock_settings.MODEL is None
                 assert mock_settings.SCALER is None
