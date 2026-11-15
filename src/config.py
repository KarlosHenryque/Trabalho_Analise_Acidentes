"""Caminhos, semente e definição do alvo, compartilhados por todos os módulos."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "data" / "raw" / "acidentes2026.csv"
GRAFICOS = RAIZ / "outputs" / "graficos"
RESULTADOS = RAIZ / "outputs" / "resultados"
MODELOS = RAIZ / "outputs" / "modelos"

SEED = 42

ALVO = "estado_fisico"
CLASSES = ["Ileso", "Lesões Leves", "Lesões Graves", "Óbito"]  # ordem de gravidade
