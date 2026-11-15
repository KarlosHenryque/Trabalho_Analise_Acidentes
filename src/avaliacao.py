"""Métricas e gráficos de avaliação, idênticos para os três modelos."""
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from config import CLASSES


def avaliar(modelo, X, y) -> tuple[dict, pd.Series]:
    """Métricas globais no conjunto (X, y) e as previsões do modelo."""
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)
    cls = list(modelo.classes_)
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=cls)
    graves = proba[:, [cls.index("Lesões Graves"), cls.index("Óbito")]].sum(1)
    auc_bin = roc_auc_score(y.isin(["Lesões Graves", "Óbito"]), graves)
    return {"Acurácia": accuracy_score(y, pred), "Balanced Accuracy": balanced_accuracy_score(y, pred),
            "Precisão macro": precision_score(y, pred, average="macro", zero_division=0),
            "Recall macro": recall_score(y, pred, average="macro", zero_division=0),
            "F1 macro": f1_score(y, pred, average="macro"), "F1 ponderado": f1_score(y, pred, average="weighted"),
            "AUC OvR macro": auc, "AUC grave/óbito": auc_bin}, pred


def por_classe(y, pred) -> pd.DataFrame:
    r = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    return pd.DataFrame(r).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]]


def matriz(y, pred, ax, titulo):
    """Matriz de confusão normalizada por linha (diagonal = recall de cada classe)."""
    m = confusion_matrix(y, pred, labels=CLASSES, normalize="true")
    sns.heatmap(m, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                vmin=0, vmax=1, cbar=False)
    ax.set(title=titulo, xlabel="Previsto", ylabel="Real (proporção da linha = recall)")
    ax.tick_params(axis="x", rotation=30)
