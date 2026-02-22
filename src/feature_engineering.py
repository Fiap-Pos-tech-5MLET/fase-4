import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features para o modelo.

    Args:
        df (pd.DataFrame): DataFrame com dados de entrada.

    Returns:
        pd.DataFrame: DataFrame com novas features adicionadas.
    """
    # 1. Feature de evolução: crescimento do INDE (INDE_2024 - INDE_23)
    # É necessário tratar casos em que INDE_23 é 0 (ausente).
    if 'INDE_2024' in df.columns and 'INDE_23' in df.columns:
        # Flag para indicar histórico disponível
        df['HAS_HISTORY_23'] = (df['INDE_23'] > 0).astype(int)

        # Crescimento do INDE
        df['INDE_GROWTH'] = df['INDE_2024'] - df['INDE_23']
        # Sem histórico, crescimento será 0 (neutro)

    # 2. Tratamento de features categóricas
    # A pipeline/modelo pode lidar com codificação (ex.: OneHotEncoder).

    # Exemplo: manter FASE como categórica caso exista.
    if 'FASE' in df.columns:
        # Sem transformação direta neste momento.
        pass

    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Seleciona as colunas finais para treinamento.

    Args:
        df (pd.DataFrame): DataFrame com dados de entrada.

    Returns:
        tuple[pd.DataFrame, pd.Series]: X (features) e y (target).

    Raises:
        ValueError: Se a coluna alvo não existir.
    """
    target_col = 'TARGET'
    if target_col not in df.columns:
        raise ValueError("Target column not found. Run create_target first.")
        
    y = df[target_col]
    
    # Remover colunas que podem causar vazamento de informação
    # e/ou que fazem parte do cálculo do target.
    leakage_cols = [
        'INDE_2024', 'PEDRA_2024', 'IAA', 'IEG', 'IPS', 'IPP', 'IDA', 'MAT', 'POR', 'ING', 
        'IPV', 'IAN', 'DESTAQUE_IEG', 'DESTAQUE_IDA', 'DESTAQUE_IPV', 'ATINGIU_PV', 
        'INDICADO', 'REC_AV1', 'REC_AV2', 'REC_PSICOLOGIA'
    ]
    
    # Remover INDE_GROWTH se estiver presente, pois usa dados de 2024.
    if 'INDE_GROWTH' in df.columns:
         leakage_cols.append('INDE_GROWTH')

    drop_cols = [target_col, 'DEFASAGEM', 'RA', 'NOME_ANONIMIZADO', 'PEDRA_2024'] + leakage_cols
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Manter apenas o conjunto mínimo de features limpas.
    
    return X, y
