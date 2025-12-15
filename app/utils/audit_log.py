"""
Módulo de utilitários para gerenciamento de logs de auditoria.

Fornece funções para armazenamento e recuperação de logs de auditoria
de requisições da API.
"""

from typing import List, Dict, Optional
from datetime import datetime


# Armazenamento em memória (em produção, usar banco de dados)
_audit_logs: List[Dict] = []


def get_audit_logs(
    date_filter: Optional[str] = None,
    route_filter: Optional[str] = None
) -> List[Dict]:
    """
    Retorna logs de auditoria filtrados por data e/ou rota.
    
    Args:
        date_filter (str, opcional): Data no formato YYYY-MM-DD
        route_filter (str, opcional): Nome da rota (ex: /api/prediction/predict)
    
    Returns:
        List[Dict]: Lista de logs de auditoria filtrados
    """
    filtered_logs = _audit_logs.copy()
    
    if date_filter:
        filtered_logs = [
            log for log in filtered_logs
            if log.get("date", "")[:10] == date_filter
        ]
    
    if route_filter:
        filtered_logs = [
            log for log in filtered_logs
            if log.get("route") == route_filter
        ]
    
    return filtered_logs


def get_audit_log_by_id(request_id: str) -> Optional[Dict]:
    """
    Retorna um log de auditoria específico pelo ID.
    
    Args:
        request_id (str): ID único da requisição
    
    Returns:
        Optional[Dict]: Log de auditoria ou None se não encontrado
    """
    for log in _audit_logs:
        if log.get("request_id") == request_id:
            return log
    
    return None


def add_audit_log(
    request_id: str,
    route: str,
    method: str = "GET",
    status_code: int = 200,
    **kwargs
) -> Dict:
    """
    Adiciona um novo log de auditoria.
    
    Args:
        request_id (str): ID único da requisição
        route (str): Rota acessada
        method (str): Método HTTP (GET, POST, etc)
        status_code (int): Código de status da resposta
        **kwargs: Dados adicionais para o log
    
    Returns:
        Dict: Log de auditoria criado
    """
    log_entry = {
        "request_id": request_id,
        "route": route,
        "method": method,
        "status_code": status_code,
        "date": datetime.now().isoformat(),
        **kwargs
    }
    
    _audit_logs.append(log_entry)
    
    return log_entry


def clear_audit_logs() -> None:
    """
    Limpa todos os logs de auditoria.
    
    Útil para testes ou reset do sistema.
    """
    global _audit_logs
    _audit_logs = []
