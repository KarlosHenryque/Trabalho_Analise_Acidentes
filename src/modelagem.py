"""Seleção de preditoras, divisão treino/teste, pré-processamento e ajuste de hiperparâmetros."""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import ALVO, CLASSES, MODELOS, RESULTADOS, SEED

# Fora das preditoras por vazamento: ilesos, feridos_leves, feridos_graves, mortos (o próprio alvo
# recodificado) e classificacao_acidente (pior desfecho entre os envolvidos).
NUMERICAS = ["idade", "km", "hora", "mes", "idade_veiculo", "n_tipos_acidente", "n_causas"]
CATEGORICAS = ["dia_semana", "uf", "br", "fase_dia", "sentido_via", "condicao_metereologica",
               "tipo_pista", "tracado_via", "uso_solo", "tipo_veiculo", "tipo_envolvido", "sexo",
               "tipo_acidente_principal", "causa_acidente_principal"]


def dados_modelagem(pessoas: pd.DataFrame):
    """Remove 'Não Informado' (rótulo desconhecido) e separa X, y e grupos (id do acidente)."""
    d = pessoas[pessoas[ALVO].isin(CLASSES)].reset_index(drop=True)
    X = d[NUMERICAS + CATEGORICAS].copy()
    for c in CATEGORICAS:
        X[c] = X[c].astype("object").where(X[c].notna(), "Ausente")
    return X, d[ALVO], d["id"]


def dividir(X, y, grupos, n_splits=5):
    """Teste ~20%, estratificado por classe e agrupado por acidente (sem vazamento entre id)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return next(sgkf.split(X, y, grupos))


def preprocessador(escalar: bool) -> ColumnTransformer:
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        num.append(("sc", StandardScaler()))
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)
    return ColumnTransformer([("num", Pipeline(num), NUMERICAS), ("cat", cat, CATEGORICAS)])


def subamostra_busca(Xtr, ytr, gtr, frac=0.4, n_splits=3):
    """Sorteia `frac` dos acidentes do treino e gera dobras agrupadas para a busca de hiperparâmetros."""
    rng = np.random.RandomState(SEED)
    ids = gtr.unique()
    msk = gtr.isin(rng.choice(ids, size=int(frac * len(ids)), replace=False)).values
    Xs, ys, gs = Xtr[msk], ytr[msk], gtr[msk]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return Xs, ys, list(cv.split(Xs, ys, gs))


def ajustar(nome, estimador, espaco, n_iter, escalar, busca, treino, peso=False):
    """RandomizedSearchCV (F1 macro) na subamostra `busca`=(Xs, ys, cv); reajusta o melhor em `treino`=(Xtr, ytr).

    Salva o modelo em outputs/modelos e o ranking da busca em outputs/resultados.
    `peso=True` passa sample_weight balanceado (para estimadores sem class_weight, como o MLP).
    """
    Xs, ys, cv = busca
    Xtr, ytr = treino
    pipe = Pipeline([("prep", preprocessador(escalar)), ("modelo", estimador)])
    bs = RandomizedSearchCV(pipe, {f"modelo__{k}": v for k, v in espaco.items()}, n_iter=n_iter, cv=cv,
                            scoring="f1_macro", random_state=SEED, n_jobs=-1, refit=False)
    t0 = time.time()
    bs.fit(Xs, ys, **({"modelo__sample_weight": compute_sample_weight("balanced", ys)} if peso else {}))
    melhor = {k.replace("modelo__", ""): v for k, v in bs.best_params_.items()}
    final = Pipeline([("prep", preprocessador(escalar)), ("modelo", estimador.set_params(**melhor))])
    final.fit(Xtr, ytr, **({"modelo__sample_weight": compute_sample_weight("balanced", ytr)} if peso else {}))
    r = pd.DataFrame(bs.cv_results_).sort_values("rank_test_score")
    r["parametros"] = r.params.apply(lambda d: {k.replace("modelo__", ""): v for k, v in d.items()})
    print(f"{nome}: ajuste+treino em {time.time()-t0:,.0f}s | melhores parâmetros: {melhor}")
    print(f"   F1 macro (validação cruzada): {r.mean_test_score.iloc[0]:.3f} ± {r.std_test_score.iloc[0]:.3f}")
    MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELOS / f"{nome}.joblib")
    r[["parametros", "mean_test_score", "std_test_score"]].to_csv(RESULTADOS / f"busca_{nome}.csv", index=False)
    return final, r
