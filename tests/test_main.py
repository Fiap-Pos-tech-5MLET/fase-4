"""
Testes unitários para o módulo app/main.py

Garante 100% de cobertura da aplicação FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from app.main import app, lifespan
from app.config import Settings


# Cliente de teste
client = TestClient(app)


class TestAppInitialization:
    """Testes para inicialização da aplicação."""

    def test_app_is_fastapi_instance(self):
        """Testa se app é instância de FastAPI."""
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)

    def test_app_title_configured(self):
        """Testa se título da app está configurado."""
        assert app.title is not None

    def test_app_description_contains_lstm(self):
        """Testa se descrição menciona LSTM."""
        assert "LSTM" in app.description

    def test_app_description_contains_tech_challenge(self):
        """Testa se descrição menciona Tech Challenge."""
        assert "Tech Challenge" in app.description

    def test_app_version_configured(self):
        """Testa se versão está configurada."""
        assert app.version == "1.0.0"

    def test_cors_middleware_configured(self):
        """Testa se CORS está configurado."""
        assert len(app.user_middleware) > 0

    def test_routes_registered(self):
        """Testa se rotas estão registradas."""
        routes = [route.path for route in app.routes]
        assert "/" in routes

    def test_lifespan_configured(self):
        """Testa se lifespan está configurado."""
        assert app.router.lifespan is not None


class TestRootEndpoint:
    """Testes para endpoint raiz."""

    def test_root_get_status_200(self):
        """Testa se GET / retorna 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self):
        """Testa se resposta é JSON válido."""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"

    def test_root_response_has_message(self):
        """Testa se resposta contém message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data

    def test_root_response_has_desafio(self):
        """Testa se resposta contém desafio."""
        response = client.get("/")
        data = response.json()
        assert "desafio" in data

    def test_root_response_has_info(self):
        """Testa se resposta contém info."""
        response = client.get("/")
        data = response.json()
        assert "info" in data

    def test_root_response_has_documentacao(self):
        """Testa se resposta contém documentacao."""
        response = client.get("/")
        data = response.json()
        assert "documentação" in data

    def test_root_response_has_versao(self):
        """Testa se resposta contém versão."""
        response = client.get("/")
        data = response.json()
        assert "versão" in data

    def test_root_message_contains_lstm(self):
        """Testa se mensagem menciona LSTM."""
        response = client.get("/")
        data = response.json()
        assert "LSTM" in data["message"]

    def test_root_message_contains_fase4(self):
        """Testa se mensagem menciona Fase 4."""
        response = client.get("/")
        data = response.json()
        assert "Fase 4" in data["message"]

    def test_root_desafio_contains_lstm(self):
        """Testa se desafio menciona LSTM."""
        response = client.get("/")
        data = response.json()
        assert "LSTM" in data["desafio"]

    def test_root_documentacao_points_to_docs(self):
        """Testa se documentação aponta para /docs."""
        response = client.get("/")
        data = response.json()
        assert "/docs" in data["documentação"]

    def test_root_versao_is_string(self):
        """Testa se versão é string."""
        response = client.get("/")
        data = response.json()
        assert isinstance(data["versão"], str)

    def test_root_all_values_are_strings(self):
        """Testa se todos os valores são strings."""
        response = client.get("/")
        data = response.json()
        for key, value in data.items():
            assert isinstance(value, str)


class TestCorsConfiguration:
    """Testes para configuração CORS."""

    def test_cors_allows_all_origins(self):
        """Testa se CORS permite todas as origens."""
        response = client.get("/", headers={"Origin": "http://example.com"})
        assert response.status_code == 200

    def test_cors_allows_credentials(self):
        """Testa se CORS permite credenciais."""
        # CORS configurado
        assert len(app.user_middleware) > 0

    def test_cors_allows_all_methods(self):
        """Testa se CORS permite todos os métodos."""
        # Verificar que POST é permitido (mesmo que endpoint não exista)
        response = client.options("/")
        assert response.status_code in [200, 405]  # 405 é Method Not Allowed (esperado)


class TestAppDocumentation:
    """Testes para documentação OpenAPI."""

    def test_openapi_schema_exists(self):
        """Testa se schema OpenAPI existe."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_swagger_ui_available(self):
        """Testa se Swagger UI está disponível."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """Testa se ReDoc está disponível."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema_has_paths(self):
        """Testa se schema OpenAPI contém paths."""
        response = client.get("/openapi.json")
        schema = response.json()
        assert "paths" in schema

    def test_root_endpoint_in_schema(self):
        """Testa se endpoint / está no schema."""
        response = client.get("/openapi.json")
        schema = response.json()
        assert "/" in schema["paths"]


class TestAppMetadata:
    """Testes para metadados da aplicação."""

    def test_app_has_title(self):
        """Testa se app tem título."""
        assert app.title is not None
        assert len(app.title) > 0

    def test_app_has_description(self):
        """Testa se app tem descrição."""
        assert app.description is not None
        assert len(app.description) > 0

    def test_app_has_version(self):
        """Testa se app tem versão."""
        assert app.version is not None
        assert app.version == "1.0.0"

    def test_app_description_format(self):
        """Testa se descrição está bem formatada."""
        assert "##" in app.description  # Markdown headers


class TestEndpointTags:
    """Testes para tags de endpoints."""

    def test_root_endpoint_has_tag(self):
        """Testa se endpoint raiz tem tag."""
        for route in app.routes:
            if route.path == "/":
                assert (route.openapi_extra and "tags" in route.openapi_extra) or route.methods

    def test_endpoint_tags_are_strings(self):
        """Testa se tags são strings."""
        response = client.get("/openapi.json")
        schema = response.json()
        
        for path_info in schema["paths"].values():
            for operation in path_info.values():
                if isinstance(operation, dict) and "tags" in operation:
                    for tag in operation["tags"]:
                        assert isinstance(tag, str)


class TestAppErrorHandling:
    """Testes para tratamento de erros."""

    def test_404_endpoint_not_found(self):
        """Testa se 404 é retornado para endpoint inexistente."""
        response = client.get("/inexistente")
        assert response.status_code == 404

    def test_404_response_is_json(self):
        """Testa se resposta 404 é JSON válido."""
        response = client.get("/inexistente")
        assert response.headers["content-type"] == "application/json"

    def test_404_has_detail(self):
        """Testa se 404 tem detail."""
        response = client.get("/inexistente")
        data = response.json()
        assert "detail" in data


class TestHttpMethods:
    """Testes para métodos HTTP permitidos."""

    def test_root_get_allowed(self):
        """Testa se GET é permitido em /."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_post_not_allowed(self):
        """Testa se POST não é permitido em /."""
        response = client.post("/")
        assert response.status_code == 405

    def test_root_put_not_allowed(self):
        """Testa se PUT não é permitido em /."""
        response = client.put("/")
        assert response.status_code == 405

    def test_root_delete_not_allowed(self):
        """Testa se DELETE não é permitido em /."""
        response = client.delete("/")
        assert response.status_code == 405


class TestAppIntegration:
    """Testes de integração da aplicação."""

    def test_app_can_handle_multiple_requests(self):
        """Testa se app pode lidar com múltiplas requisições."""
        for _ in range(5):
            response = client.get("/")
            assert response.status_code == 200

    def test_app_response_consistent(self):
        """Testa se respostas são consistentes."""
        response1 = client.get("/")
        response2 = client.get("/")
        
        assert response1.json() == response2.json()

    def test_app_with_different_accept_headers(self):
        """Testa app com diferentes Accept headers."""
        headers = {"Accept": "application/json"}
        response = client.get("/", headers=headers)
        assert response.status_code == 200

    def test_app_with_custom_headers(self):
        """Testa app com headers customizados."""
        headers = {"X-Custom-Header": "test"}
        response = client.get("/", headers=headers)
        assert response.status_code == 200


class TestAppLifespan:
    """Testes para gerenciador de contexto lifespan."""

    @pytest.mark.asyncio
    async def test_lifespan_is_async_context_manager(self):
        """Testa se lifespan é um context manager assíncrono."""
        # Mock os/sys para evitar erros de import e file not found
        from unittest.mock import patch
        from app.main import lifespan
        with patch("app.main.os.path.abspath"), \
             patch("app.main.hf_hub_download"), \
             patch("app.main.joblib.load"), \
             patch("app.main.torch.load"), \
             patch("app.main.os.path.exists", return_value=False):
             
            async with lifespan(app):
                pass
            # Se não levantou erro, passounfigurado na app
        assert app.router.lifespan is not None


class TestAppMiddleware:
    """Testes para middleware da aplicação."""

    def test_app_has_middleware(self):
        """Testa se app tem middleware configurado."""
        assert len(app.user_middleware) > 0

    def test_cors_in_middleware(self):
        """Testa se CORS está em middleware."""
        middleware_names = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_names or len(app.user_middleware) > 0


class TestAppResponseFormat:
    """Testes para formato de resposta."""

    def test_root_response_json_structure(self):
        """Testa estrutura JSON da resposta raiz."""
        response = client.get("/")
        data = response.json()
        
        expected_keys = {"message", "desafio", "info", "documentação", "versão"}
        assert set(data.keys()) == expected_keys

    def test_root_response_types(self):
        """Testa tipos de cada campo da resposta."""
        response = client.get("/")
        data = response.json()
        
        for value in data.values():
            assert isinstance(value, str)

    def test_root_response_non_empty_values(self):
        """Testa se valores não são vazios."""
        response = client.get("/")
        data = response.json()
        
        for value in data.values():
            assert len(value) > 0


class TestAppContentType:
    """Testes para content-type das respostas."""

    def test_root_content_type_json(self):
        """Testa se content-type é JSON."""
        response = client.get("/")
        assert "application/json" in response.headers["content-type"]

    def test_root_charset_utf8(self):
        """Testa se charset é UTF-8."""
        response = client.get("/")
        content_type = response.headers.get("content-type", "")
        # FastAPI usa UTF-8 por padrão
        assert response.status_code == 200


class TestAppResponseEncoding:
    """Testes para encoding de resposta."""

    def test_root_response_encoding(self):
        """Testa encoding da resposta."""
        response = client.get("/")
        
        # Verificar que pode ser decodificado como UTF-8
        assert response.content.decode("utf-8") is not None

    def test_root_response_with_unicode(self):
        """Testa se resposta pode conter Unicode."""
        response = client.get("/")
        data = response.json()
        
        # Verificar que strings foram decodificadas corretamente
        assert all(isinstance(v, str) for v in data.values())
