import pandas as pd
import pytest

from src.feature_engineering import create_features, select_features


def test_create_features_adds_growth_and_history():
    """Testa a criação das features de crescimento e histórico."""
    df = pd.DataFrame({
        'INDE_2024': [10.0, 5.0],
        'INDE_23': [8.0, 0.0]
    })

    result = create_features(df)

    assert 'HAS_HISTORY_23' in result.columns
    assert 'INDE_GROWTH' in result.columns
    assert result['HAS_HISTORY_23'].tolist() == [1, 0]
    assert result['INDE_GROWTH'].tolist() == [2.0, 5.0]


def test_create_features_without_required_columns():
    """Testa o comportamento quando as colunas necessárias não existem."""
    df = pd.DataFrame({
        'FASE': ['1A', '2B'],
        'OUTRA_COLUNA': [1, 2]
    })

    result = create_features(df)

    assert set(result.columns) == {'FASE', 'OUTRA_COLUNA'}


def test_select_features_raises_without_target():
    """Testa erro quando a coluna TARGET não existe."""
    df = pd.DataFrame({'INDE_23': [1, 2]})

    with pytest.raises(ValueError, match="Target column not found"):
        select_features(df)


def test_select_features_drops_leakage_and_metadata():
    """Testa remoção de colunas com vazamento e metadados."""
    df = pd.DataFrame({
        'TARGET': [0, 1],
        'DEFASAGEM': [0, 1],
        'RA': ['1', '2'],
        'NOME_ANONIMIZADO': ['A', 'B'],
        'PEDRA_2024': ['X', 'Y'],
        'INDE_2024': [10.0, 12.0],
        'IAA': [1, 2],
        'IEG': [1, 2],
        'IPS': [1, 2],
        'IPP': [1, 2],
        'IDA': [1, 2],
        'MAT': [1, 2],
        'POR': [1, 2],
        'ING': [1, 2],
        'IPV': [1, 2],
        'IAN': [1, 2],
        'DESTAQUE_IEG': [0, 1],
        'DESTAQUE_IDA': [0, 1],
        'DESTAQUE_IPV': [0, 1],
        'ATINGIU_PV': [0, 1],
        'INDICADO': [0, 1],
        'REC_AV1': [0, 1],
        'REC_AV2': [0, 1],
        'REC_PSICOLOGIA': [0, 1],
        'INDE_GROWTH': [2.0, 3.0],
        'INDE_23': [8.0, 9.0],
        'IDADE': [10, 11]
    })

    X, y = select_features(df)

    assert y.tolist() == [0, 1]
    assert set(X.columns) == {'INDE_23', 'IDADE'}
    assert 'TARGET' not in X.columns
    assert 'INDE_GROWTH' not in X.columns
