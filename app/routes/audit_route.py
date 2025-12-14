from fastapi import APIRouter, Depends, HTTPException, Path, status, Query
from typing import Optional, List, Dict
from app.utils.audit_log import get_audit_log_by_id, get_audit_logs
from datetime import datetime

router = APIRouter(
    prefix="/api/audit",
    tags=["Auditoria"]
    )

@router.get("/audit", status_code=status.HTTP_200_OK, response_model=List[Dict])
def get_audit(
    date: Optional[str] = Query(description="Filtrar por data (YYYY-MM-DD)",default=datetime.now().isoformat()[:10]),
    route: Optional[str] = Query(None, description="Filtrar por nome da rota, ex: /api/prediction/predict")
) -> List[Dict]:
    """
    Retorna os logs de auditoria das requisições de predição.

    É possível filtrar os logs por data e/ou nome da rota.

    Args:
        date (str, opcional): Filtra os logs por data de processamento.
        route (str, opcional): Filtra os logs por nome da rota.

    Returns:
        List[Dict]: Uma lista de logs de auditoria correspondentes aos filtros.
    """
    return get_audit_logs(date_filter=date, route_filter=route)

@router.get("/audit/{request_id}", status_code=status.HTTP_200_OK)
def get_audit_by_id(
    request_id: str = Path(..., description="O ID único da requisição de auditoria")
) -> Dict:
    """
    Retorna um log de auditoria específico com base no seu ID único.

    Args:
        request_id (str): O ID da requisição de auditoria.

    Returns:
        dict: O log de auditoria completo.

    Raises:
        HTTPException: 404 Not Found se o ID não for encontrado.
    """
    log_entry = get_audit_log_by_id(request_id)
    if not log_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Log de auditoria com ID '{request_id}' não encontrado."
        )
    return log_entry