"""Leitura do CSV da PRF e construção das tabelas por pessoa e por acidente."""
import numpy as np
import pandas as pd

from config import CSV


def carregar_bruto() -> pd.DataFrame:
    """CSV da PRF: separador ';', decimal ',', encoding latin-1 (não é UTF-8)."""
    df = pd.read_csv(CSV, sep=";", encoding="latin-1", decimal=",", low_memory=False)
    df["id"] = df["id"].astype("int64")
    df["data_inversa"] = pd.to_datetime(df["data_inversa"])
    return df


def construir_pessoas(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pessoa (id, pesid), com atributos do acidente agregados.

    O CSV repete a pessoa para cada combinação causa x tipo de acidente; aqui isso
    é colapsado para evitar contar a mesma pessoa várias vezes.
    """
    acid = df.groupby("id").agg(n_tipos_acidente=("tipo_acidente", "nunique"),
                                n_causas=("causa_acidente", "nunique"),
                                n_pessoas_acidente=("pesid", "nunique"))
    tipo1 = df[df.ordem_tipo_acidente == 1].drop_duplicates("id").set_index("id")["tipo_acidente"]
    causa1 = df[df.causa_principal == "Sim"].drop_duplicates("id").set_index("id")["causa_acidente"]
    acid["tipo_acidente_principal"] = tipo1
    acid["causa_acidente_principal"] = causa1

    p = df[df.pesid.notna()].drop_duplicates(["id", "pesid"]).copy()
    p = p.join(acid, on="id")
    p["hora"] = pd.to_datetime(p["horario"], format="%H:%M:%S").dt.hour
    p["mes"] = p["data_inversa"].dt.month
    p["idade_veiculo"] = 2026 - p["ano_fabricacao_veiculo"]
    # Idade e ano de fabricação impossíveis são erros de registro (ver EDA): viram ausentes.
    p.loc[(p.idade < 0) | (p.idade > 110), "idade"] = np.nan
    p.loc[(p.idade_veiculo < 0) | (p.idade_veiculo > 80), "idade_veiculo"] = np.nan
    p["br"] = p["br"].astype("Int64").astype("string")
    return p.reset_index(drop=True)


def construir_acidentes(df: pd.DataFrame, pessoas: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por acidente (id); as colunas do acidente são constantes dentro do id."""
    acid = df.sort_values("ordem_tipo_acidente").drop_duplicates("id").copy()
    acid["hora"] = pd.to_datetime(acid.horario, format="%H:%M:%S").dt.hour
    acid["mes"] = acid.data_inversa.dt.month
    return acid.merge(pessoas.groupby("id").size().rename("n_pessoas"), on="id", how="left")
