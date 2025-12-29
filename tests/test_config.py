"""
Testes unitários para o módulo app/config.py.

Garante 100% de cobertura das configurações do projeto.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from app.config import Settings, get_settings


class TestSettingsClass:
    """Testes para inicialização da classe Settings."""

    def test_settings_with_env_variables(self):
        """Testa inicialização com variáveis de ambiente."""
        with patch.dict(os.environ, {
            'PROJECT_NAME': 'Test Project',
            'SECRET_KEY': 'test-secret-key',
            'ACCESS_TOKEN_EXPIRE_MINUTES': '120'
        }):
            settings = Settings()
            
            assert settings.PROJECT_NAME == 'Test Project'
            assert settings.SECRET_KEY == 'test-secret-key'
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 120

    def test_settings_default_values(self):
        """Testa valores padrão quando variáveis não estão definidas."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            assert settings.PROJECT_NAME == "Stock Prediction API - LSTM"
            assert settings.SECRET_KEY == "development-secret-key-change-in-production"
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 60
            assert settings.HTML_CACHE_DIR == "./cache"

    def test_settings_partial_env_variables(self):
        """Testa com apenas algumas variáveis definidas."""
        with patch.dict(os.environ, {
            'PROJECT_NAME': 'Partial Project',
            'ACCESS_TOKEN_EXPIRE_MINUTES': '90'
        }, clear=True):
            settings = Settings()
            
            assert settings.PROJECT_NAME == 'Partial Project'
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 90
            assert settings.SECRET_KEY == "development-secret-key-change-in-production"

    def test_settings_model_initialization(self):
        """Testa se Settings é instância de BaseSettings."""
        settings = Settings()
        assert isinstance(settings, Settings)

    def test_settings_attributes_exist(self):
        """Testa se todos os atributos esperados existem."""
        settings = Settings()
        
        assert hasattr(settings, 'PROJECT_NAME')
        assert hasattr(settings, 'SECRET_KEY')
        assert hasattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES')
        assert hasattr(settings, 'HTML_CACHE_DIR')
        assert hasattr(settings, 'MODEL_REPO_ID')
        assert hasattr(settings, 'MODEL_FILENAME')
        assert hasattr(settings, 'ALGORITHM')
        assert hasattr(settings, 'MODEL')

    def test_settings_access_token_expire_is_int(self):
        """Testa se ACCESS_TOKEN_EXPIRE_MINUTES é inteiro."""
        with patch.dict(os.environ, {
            'ACCESS_TOKEN_EXPIRE_MINUTES': '150'
        }):
            settings = Settings()
            assert isinstance(settings.ACCESS_TOKEN_EXPIRE_MINUTES, int)
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 150

    def test_settings_model_is_empty_string(self):
        """Testa se MODEL inicializa como string vazia."""
        settings = Settings()
        assert settings.MODEL == None

    def test_settings_with_special_characters(self):
        """Testa com caracteres especiais nas variáveis."""
        with patch.dict(os.environ, {
            'PROJECT_NAME': 'Test-Project_2024',
            'SECRET_KEY': 'secret!@#$%^&*()',
            'ALGORITHM': 'HS256'
        }):
            settings = Settings()
            
            assert settings.PROJECT_NAME == 'Test-Project_2024'
            assert settings.SECRET_KEY == 'secret!@#$%^&*()'
            assert settings.ALGORITHM == 'HS256'

    def test_settings_environment_priority(self):
        """Testa se variáveis de ambiente têm prioridade sobre padrões."""
        with patch.dict(os.environ, {
            'PROJECT_NAME': 'Env Project',
            'ACCESS_TOKEN_EXPIRE_MINUTES': '180'
        }):
            settings = Settings()
            
            # Env deve sobrescrever padrão
            assert settings.PROJECT_NAME == 'Env Project'
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 180


class TestGetSettings:
    """Testes para a função get_settings com cache."""

    def test_get_settings_returns_settings(self):
        """Testa se get_settings retorna instância de Settings."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_same_instance(self):
        """Testa se get_settings retorna mesma instância (cache)."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_get_settings_is_callable(self):
        """Testa se get_settings é função/callable."""
        assert callable(get_settings)

    def test_get_settings_no_arguments(self):
        """Testa se get_settings pode ser chamada sem argumentos."""
        settings = get_settings()
        assert settings is not None

    def test_get_settings_cached_behavior(self):
        """Testa comportamento de cache da função."""
        # Primeira chamada
        settings1 = get_settings()
        project1 = settings1.PROJECT_NAME
        
        # Segunda chamada deve retornar exatamente o mesmo objeto
        settings2 = get_settings()
        
        assert id(settings1) == id(settings2)
        assert settings1.PROJECT_NAME == settings2.PROJECT_NAME

    def test_get_settings_returns_type(self):
        """Testa tipo de retorno de get_settings."""
        result = get_settings()
        assert type(result).__name__ == 'Settings'


class TestSettingsDefaults:
    """Testes para valores padrão das configurações."""

    def test_default_algorithm(self):
        """Testa algoritmo padrão."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.ALGORITHM == "HS256"

    def test_default_model_repo_id(self):
        """Testa MODEL_REPO_ID padrão."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.MODEL_REPO_ID == "default-repo"

    def test_default_model_filename(self):
        """Testa MODEL_FILENAME padrão."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.MODEL_FILENAME == "lstm_model.pth"

    def test_access_token_default(self):
        """Testa ACCESS_TOKEN_EXPIRE_MINUTES padrão."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 60


class TestSettingsEnvironmentVariables:
    """Testes com diferentes variáveis de ambiente."""

    def test_all_env_variables_set(self):
        """Testa com todas as variáveis de ambiente definidas."""
        env_vars = {
            'PROJECT_NAME': 'Stock API',
            'SECRET_KEY': 'super-secret',
            'ACCESS_TOKEN_EXPIRE_MINUTES': '120',
            'HTML_CACHE_DIR': '/cache',
            'MODEL_REPO_ID': 'user/model-repo',
            'MODEL_FILENAME': 'lstm_model.pth',
            'ALGORITHM': 'HS256'
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.PROJECT_NAME == 'Stock API'
            assert settings.SECRET_KEY == 'super-secret'
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 120
            assert settings.HTML_CACHE_DIR == '/cache'
            assert settings.MODEL_REPO_ID == 'user/model-repo'
            assert settings.MODEL_FILENAME == 'lstm_model.pth'
            assert settings.ALGORITHM == 'HS256'

    def test_env_variable_types(self):
        """Testa conversão de tipos das variáveis."""
        with patch.dict(os.environ, {
            'ACCESS_TOKEN_EXPIRE_MINUTES': '90'
        }):
            settings = Settings()
            
            assert isinstance(settings.ACCESS_TOKEN_EXPIRE_MINUTES, int)
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 90

    def test_env_empty_string_vs_none(self):
        """Testa diferença entre string vazia e None."""
        with patch.dict(os.environ, {
            'PROJECT_NAME': ''
        }):
            settings = Settings()
            
            # String vazia deve ser preservada
            assert settings.PROJECT_NAME == ''
            assert isinstance(settings.PROJECT_NAME, str)


class TestSettingsEdgeCases:
    """Testes para casos extremos."""

    def test_very_long_project_name(self):
        """Testa com nome de projeto muito longo."""
        long_name = 'A' * 1000
        with patch.dict(os.environ, {
            'PROJECT_NAME': long_name
        }):
            settings = Settings()
            assert settings.PROJECT_NAME == long_name

    def test_very_long_secret_key(self):
        """Testa com secret key muito longa."""
        long_key = 'K' * 1000
        with patch.dict(os.environ, {
            'SECRET_KEY': long_key
        }):
            settings = Settings()
            assert settings.SECRET_KEY == long_key

    def test_zero_access_token_expire(self):
        """Testa com tempo de expiração zero."""
        with patch.dict(os.environ, {
            'ACCESS_TOKEN_EXPIRE_MINUTES': '0'
        }):
            settings = Settings()
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 0

    def test_negative_access_token_expire(self):
        """Testa com tempo de expiração negativo."""
        with patch.dict(os.environ, {
            'ACCESS_TOKEN_EXPIRE_MINUTES': '-1'
        }):
            settings = Settings()
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == -1

    def test_very_large_access_token_expire(self):
        """Testa com tempo de expiração muito grande."""
        with patch.dict(os.environ, {
            'ACCESS_TOKEN_EXPIRE_MINUTES': '999999'
        }):
            settings = Settings()
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 999999

    def test_unicode_in_variables(self):
        """Testa com caracteres Unicode."""
        with patch.dict(os.environ, {
            'PROJECT_NAME': 'Projeto 🚀 LSTM',
            'SECRET_KEY': 'chave-secreta-🔐'
        }):
            settings = Settings()
            assert settings.PROJECT_NAME == 'Projeto 🚀 LSTM'
            assert settings.SECRET_KEY == 'chave-secreta-🔐'


class TestSettingsModelInitialization:
    """Testes para inicialização do modelo."""

    def test_model_attribute_initial_value(self):
        """Testa valor inicial do atributo MODEL."""
        settings = Settings()
        assert settings.MODEL == None

    def test_model_can_be_modified(self):
        """Testa se MODEL pode ser modificado."""
        settings = Settings()
        settings.MODEL = "test_model_object"
        assert settings.MODEL == "test_model_object"

    def test_model_type_flexibility(self):
        """Testa que MODEL pode armazenar diferentes tipos."""
        settings = Settings()
        
        # String
        settings.MODEL = "model.pth"
        assert isinstance(settings.MODEL, str)
        
        # None
        settings.MODEL = None
        assert settings.MODEL is None

    def test_settings_immutability_outside_model(self):
        """Testa se outras propriedades podem ser acessadas."""
        with patch.dict(os.environ, {
            'PROJECT_NAME': 'Test'
        }):
            settings = Settings()
            name = settings.PROJECT_NAME
            assert name == 'Test'
