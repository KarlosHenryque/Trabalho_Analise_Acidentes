# Acidentes PRF 2026: análise exploratória e Machine Learning

**Integrantes:** Karlos Henryque e Paula Krupinski
**Dados:** Polícia Rodoviária Federal – Dados Abertos, *Acidentes 2026 (jan–set), agrupados por pessoa*
<https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf>

O projeto faz duas coisas:

1. **Análise exploratória (EDA):** quando, onde e em que condições os acidentes acontecem e o que se associa à gravidade das lesões.
2. **Classificação:** prevê o **estado físico** de cada pessoa envolvida (Ileso, Lesões Leves, Lesões Graves, Óbito) com três modelos (Árvore de Decisão, Regressão Logística e rede neural MLP) e compara os resultados.

Tudo roda localmente; nenhum dado sai do computador.

---

## 1. Estrutura do projeto

```
trabalho/
├── data/
│   └── raw/acidentes2026.csv          # CSV original da PRF (não é alterado)
├── notebooks/
│   └── analise_acidentes_prf_2026.ipynb  # relatório completo: texto, código, gráficos e conclusões
├── src/                               # código reutilizável, importado pelo notebook
│   ├── config.py                      # caminhos, semente (42), alvo e classes
│   ├── dados.py                       # leitura do CSV e tabelas por pessoa e por acidente
│   ├── estatistica.py                 # V de Cramér, variabilidade, outliers (IQR)
│   ├── visualizacao.py                # estilo, cores e salvamento dos gráficos
│   ├── modelagem.py                   # preditoras, divisão treino/teste, pipeline, busca de hiperparâmetros
│   └── avaliacao.py                   # métricas, relatório por classe e matriz de confusão
├── outputs/                           # tudo aqui é GERADO pelo notebook (pode apagar e rodar de novo)
│   ├── graficos/    01..15 *.png
│   ├── resultados/  tabelas *.csv
│   └── modelos/     modelos treinados *.joblib
├── .claude/skills/data-visualization/ # guia de estilo de gráficos usado pelo Claude Code (não afeta a execução)
├── requirements.txt
└── README.md
```

### Por que essa separação

Cada arquivo tem **uma** responsabilidade, e as dependências seguem uma única direção:

```
config.py  ←  dados.py
           ←  estatistica.py      (não depende de config)
           ←  visualizacao.py
           ←  modelagem.py
           ←  avaliacao.py
                     ↑
              notebook (orquestra tudo, escreve o texto e as interpretações)
```

- **`config.py`** é o único lugar com caminhos e constantes do problema. Mudar a pasta de saída ou a semente é uma linha.
- **`dados.py`** só sabe ler e reorganizar o CSV. Não sabe nada de modelos nem de gráficos.
- **`estatistica.py`** e **`visualizacao.py`** são as ferramentas da EDA, separadas entre "calcular" e "desenhar".
- **`modelagem.py`** concentra tudo o que define o experimento de ML (quais colunas entram, como dividir, como pré-processar, como buscar hiperparâmetros), para que os três modelos passem exatamente pelo mesmo processo.
- **`avaliacao.py`** aplica as mesmas métricas aos três modelos, então a comparação é justa.
- **O notebook** é o relatório: chama as funções, mostra os resultados e explica cada decisão.

---

## 2. Como executar (Windows, PowerShell, Python 3.14)

Na pasta `trabalho`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m jupyter lab notebooks/analise_acidentes_prf_2026.ipynb
```

No Jupyter: **Run > Run All Cells** (kernel "Python 3"). A execução completa leva alguns minutos (a busca de hiperparâmetros do MLP é a parte mais lenta).

Para executar sem abrir o Jupyter (atualiza o notebook e todos os arquivos de `outputs/`):

```powershell
cd notebooks
..\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute analise_acidentes_prf_2026.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

O notebook encontra o código de `src/` pela linha `sys.path.append("../src")`, por isso deve ser executado de dentro de `notebooks/`.

---

## 3. O que cada módulo faz

### `config.py`
| Nome | O que é |
|---|---|
| `RAIZ`, `CSV`, `GRAFICOS`, `RESULTADOS`, `MODELOS` | caminhos do projeto |
| `SEED = 42` | semente usada em toda aleatoriedade (divisão, busca, modelos), garante resultados reproduzíveis |
| `ALVO`, `CLASSES` | coluna prevista (`estado_fisico`) e as 4 classes em ordem de gravidade |

### `dados.py`
| Função | O que faz |
|---|---|
| `carregar_bruto()` | lê o CSV (separador `;`, decimal `,`, encoding latin-1) e converte `id` e data |
| `construir_pessoas(df)` | gera **uma linha por pessoa** (`id`, `pesid`), com dados agregados do acidente (tipo e causa principais, nº de tipos e causas), hora, mês e idade do veículo. Idades > 110 e idade de veículo fora de 0–80 viram ausentes (erros de registro) |
| `construir_acidentes(df, pessoas)` | gera **uma linha por acidente** (`id`), com hora, mês e nº de pessoas |

### `estatistica.py`
| Função | O que faz |
|---|---|
| `cramers_v(a, b)` | qui-quadrado, p-valor e **V de Cramér** (tamanho do efeito entre duas categóricas, de 0 a 1) |
| `variabilidade(df, cols)` | média, mediana, desvio, variância, quartis, amplitude, **coeficiente de variação** e assimetria |
| `outliers_iqr(s)` | limites de Tukey (Q1 − 1,5·IQR, Q3 + 1,5·IQR) e quantos valores ficam fora |

### `visualizacao.py`
| Nome | O que faz |
|---|---|
| `AZUL`, `COR` | paleta amigável a daltônicos; cada classe tem sempre a mesma cor (Óbito = vinho etc.) |
| `aplicar_estilo()` | tema padrão do matplotlib para todos os gráficos |
| `rotular(ax)` | escreve o valor em cima de cada barra |
| `salvar(nome)` | remove bordas, adiciona a fonte dos dados, salva em `outputs/graficos/nome.png` e exibe |

### `modelagem.py`
| Nome | O que faz |
|---|---|
| `NUMERICAS`, `CATEGORICAS` | as 21 preditoras usadas pelos modelos |
| `dados_modelagem(pessoas)` | remove "Não Informado" e devolve `X`, `y` e o grupo (`id` do acidente) |
| `dividir(X, y, grupos)` | separa ~20% para teste, **estratificado** pela classe e **agrupado** por acidente |
| `preprocessador(escalar)` | `ColumnTransformer`: imputação pela mediana (+ indicador de ausência) e one-hot; padronização só se `escalar=True` (MLP) |
| `subamostra_busca(Xtr, ytr, gtr)` | sorteia 40% dos acidentes do treino e cria 3 dobras agrupadas para a busca |
| `ajustar(...)` | `RandomizedSearchCV` com F1 macro na subamostra, reajusta o melhor no treino inteiro, salva o modelo (`.joblib`) e o ranking da busca (`busca_*.csv`) |

### `avaliacao.py`
| Função | O que faz |
|---|---|
| `avaliar(modelo, X, y)` | acurácia, balanced accuracy, precisão/recall/F1 macro, F1 ponderado, AUC OvR e AUC "grave ou óbito"; devolve também as previsões |
| `por_classe(y, pred)` | precisão, recall, F1 e suporte de cada classe |
| `matriz(y, pred, ax, titulo)` | matriz de confusão normalizada por linha (diagonal = recall) |

---

## 4. Roteiro do notebook

| Seção | Conteúdo |
|---|---|
| 1–2 | Introdução e objetivos |
| 3 | Descrição da base e **granularidade**: o CSV tem 457.622 linhas, mas só 135.786 pessoas e 54.895 acidentes |
| 4 | Metodologia |
| 5.1 | Tendências temporais e espaciais (mês, dia da semana, hora, UF, rodovia, tipo de acidente, clima) |
| 5.2 | Perfil das pessoas (estado físico, idade, sexo) |
| 5.3 | Análise conjunta (gravidade × hora, dia, idade, veículo; clima e pista × tipo de acidente; V de Cramér; Kruskal-Wallis; taxa de óbito por UF) |
| 5.4 | Variabilidade (coeficiente de variação, boxplots) |
| 5.5 | Outliers (regra IQR e decisão caso a caso) |
| 6 | Preparação: definição do problema, vazamento de dados, divisão treino/teste |
| 7 | Os três modelos: explicação, busca de hiperparâmetros, árvore desenhada, curvas do MLP |
| 8 | Avaliação no teste: métricas, matrizes de confusão, ROC, importância das variáveis, estabilidade |
| 9–11 | Discussão, conclusão e referências |

---

## 5. Decisões metodológicas importantes

1. **Uma linha por pessoa.** O CSV repete cada pessoa para cada combinação causa × tipo de acidente (em média 3 vezes, até 60). Contar linhas inflaria tudo. A EDA de acidentes usa uma linha por `id`; a de pessoas e a modelagem usam uma linha por (`id`, `pesid`).
2. **"Não Informado" fica fora da modelagem.** Não é um estado clínico, é falha de registro (6,5% das pessoas).
3. **Vazamento de dados (data leakage).** `ilesos`, `feridos_leves`, `feridos_graves` e `mortos` são o próprio alvo recodificado; `classificacao_acidente` resume o pior desfecho do acidente. As cinco foram excluídas, senão o modelo "colaria" a resposta.
4. **Divisão agrupada por acidente.** Pessoas do mesmo acidente ficam todas no treino ou todas no teste; caso contrário o modelo veria o mesmo acidente nos dois lados.
5. **Desbalanceamento.** Óbito é só 3,6% das pessoas. Por isso os modelos usam pesos por classe e o critério de escolha é o **F1 macro** (média do F1 das 4 classes), não a acurácia.
6. **O teste só é usado no final.** Toda a busca de hiperparâmetros acontece dentro do treino.
7. **Outliers não são removidos automaticamente.** Só valores impossíveis (idade > 110, ano de fabricação 0) viram ausentes; extremos plausíveis (ônibus com 87 pessoas, km 1.260) ficam.

---

## 6. Principais resultados

**EDA**
- Mais acidentes no sábado e domingo (~25% acima da terça), pico às 18h; BR-101 e BR-116 somam 32% dos acidentes.
- A gravidade é mais associada ao **tipo de veículo** (V de Cramér 0,31; moto e motoneta ~26–29% com lesão grave ou óbito) e ao **tipo de envolvido** (pedestres 65%). Idade, sexo, hora e dia da semana têm efeito pequeno.
- Óbito chega a ~8% das pessoas às 3–4h da madrugada, contra 2–3% durante o dia.

**Modelos (conjunto de teste, 25.399 pessoas)**

| Modelo | Acurácia | F1 macro | AUC OvR | AUC grave/óbito |
|---|---|---|---|---|
| Árvore de Decisão | 0,574 | 0,437 | 0,757 | 0,772 |
| **Regressão Logística** | **0,584** | **0,462** | **0,801** | **0,804** |
| MLP | 0,571 | 0,454 | 0,785 | 0,800 |

A Regressão Logística é a melhor nas métricas globais, mas com pouca vantagem sobre o MLP. O desempenho é moderado porque faltam na base as variáveis que mais determinam a gravidade (velocidade, cinto/capacete, ponto de impacto). Os resultados indicam **associação, não causalidade**.

---

## 7. Arquivos gerados em `outputs/`

| Arquivo | Conteúdo |
|---|---|
| `graficos/01`–`08` | gráficos da EDA |
| `graficos/09`–`15` | classes, árvore, curvas do MLP, matrizes de confusão, comparação, ROC, importância |
| `resultados/diagnostico_colunas.csv` | tipo, nulos e valores únicos de cada coluna do CSV |
| `resultados/associacoes_cramer.csv` | qui-quadrado e V de Cramér dos pares analisados |
| `resultados/variabilidade.csv`, `outliers_iqr.csv` | tabelas das seções 5.4 e 5.5 |
| `resultados/distribuicao_classes.csv` | proporção das classes no treino e no teste |
| `resultados/busca_*.csv` | todas as combinações testadas na busca de cada modelo |
| `resultados/comparacao_modelos.csv`, `metricas_por_classe.csv` | métricas no teste |
| `resultados/importancia_permutacao.csv`, `estabilidade.csv` | importância das variáveis e estabilidade entre dobras |
| `modelos/*.joblib` | modelos treinados (pipeline completo: pré-processamento + modelo) |

Para usar um modelo salvo:

```python
import joblib
modelo = joblib.load("outputs/modelos/regressao_logistica.joblib")
modelo.predict(X)  # X com as colunas de modelagem.NUMERICAS + modelagem.CATEGORICAS
```
