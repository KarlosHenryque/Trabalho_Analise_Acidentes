"""Estilo, cores e salvamento padronizado dos gráficos."""
import matplotlib.pyplot as plt

from config import GRAFICOS

# Paleta amigável a daltônicos, com cor fixa por classe (guia em .claude/skills/data-visualization).
AZUL = "#4C72B0"
COR = {"Ileso": "#4C72B0", "Lesões Leves": "#F2C14E", "Lesões Graves": "#DD8452",
       "Óbito": "#9B2226", "Não Informado": "#B0B0B0"}
FONTE = "Fonte: PRF – Dados Abertos, Acidentes 2026 (jan–set)"


def aplicar_estilo():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 100, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "axes.grid": True, "grid.alpha": 0.35,
                         "axes.prop_cycle": plt.cycler(color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"])})


def rotular(ax, fmt="{:,.0f}"):
    """Rótulos de valor nas barras (apenas nos gráficos de barras simples)."""
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=3, fontsize=9)


def salvar(nome: str):
    """Despe bordas, adiciona a fonte, salva em outputs/graficos e exibe a figura corrente."""
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    for ax in plt.gcf().axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.gcf().text(0.01, -0.03, FONTE, fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(GRAFICOS / f"{nome}.png", dpi=130, bbox_inches="tight")
    plt.show()
