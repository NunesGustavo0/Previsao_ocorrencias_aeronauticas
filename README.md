# Ocorrencias Aeronauticas

Projeto do Desafio 01 IFAM / AxAcademy para classificar a seriedade de ocorrencias aeronauticas brasileiras a partir de dados publicos.

## Problema e metrica de sucesso

O objetivo e prever a coluna `Classificacao_da_Ocorrencia`, separando classes como `Acidente`, `Incidente` e `Incidente Grave` a partir de informacoes operacionais, geograficas e tecnicas da ocorrencia e da aeronave.

A metrica principal e `F1-macro`, porque o problema tem classes desbalanceadas e a avaliacao precisa considerar desempenho medio entre as classes. A meta definida para sucesso do projeto e atingir pelo menos `0,90` de F1-macro; no experimento atual, o melhor modelo chegou a `0,7178`, portanto ainda ha espaco claro de melhoria.

## Dataset

Fonte: conjunto publico "Ocorrencias Aeronauticas" da ANAC.

- URL dos metadados: https://www.anac.gov.br/acesso-a-informacao/dados-abertos/areas-de-atuacao/seguranca-operacional/ocorrencias-aeronauticas/metadados-do-conjunto-de-dados-ocorrencias-aeronauticas
- URL alternativa no portal dados.gov.br: https://dados.gov.br/dados/conjuntos-dados/ocorrencias-aeronauticas-da-aviacao-civil-brasileira
- Arquivo usado no projeto: `V_OCORRENCIA_AMPLA.csv`
- Data de download registrada para reproducao do pacote: 2026-05-29
- Licenca: dados abertos governamentais; no portal dados.gov.br, o conjunto relacionado a ocorrencias aeronauticas consta com Open Data Commons Open Database License (ODbL).

Os CSVs nao ficam versionados diretamente no Git. Eles sao controlados pelo DVC em arquivos `.dvc`.

## Estrutura

```text
.
+-- data/
|   +-- raw/V_OCORRENCIA_AMPLA.csv.dvc
|   +-- processed/ocorrencia_pre_tratado.csv.dvc
+-- notebooks/
|   +-- 01_eda.ipynb
+-- src/
|   +-- treino.py
|   +-- promover.py
|   +-- api.py
+-- eda/
+-- pre_processamento/
|   +-- treino.py
|   +-- mlruns/
+-- api/
|   +-- main.py
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- REPOSITORIO.md
```

As pastas `data/`, `notebooks/` e `src/` foram adicionadas para seguir a estrutura solicitada no desafio. O codigo historico usado no desenvolvimento permanece em `eda/`, `pre_processamento/` e `api/`.

## Pre-requisitos

- Python 3.12
- Docker e Docker Compose
- Git
- DVC

Instalacao local das dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducao passo a passo

Clone o repositorio:

```bash
git clone git@github.com:NunesGustavo0/Previsao_ocorrencias_aeronauticas.git
cd Previsao_ocorrencias_aeronauticas
```

Instale as dependencias e recupere os dados pelo DVC:

```bash
python -m pip install -r requirements.txt
dvc pull
```

O remote DVC local deste pacote esta configurado como:

```bash
/tmp/ocorrencias_aeronauticas_dvcstore
```

Se o `dvcstore` estiver em outro caminho na maquina de correcao, ajuste antes do `dvc pull`:

```bash
dvc remote modify local url /caminho/local/dvcstore
dvc pull
```

Execute a EDA em `notebooks/01_eda.ipynb` ou na copia historica `eda/01_eda.ipynb`.

Treine os modelos e registre o melhor no MLflow:

```bash
python src/treino.py
```

Promova para o alias `@production` o melhor run registrado:

```bash
python src/promover.py
```

Suba a API:

```bash
docker compose up --build
```

A API ficara disponivel em:

- Swagger: http://localhost:8000/docs
- Saude: http://localhost:8000/saude
- Predicao: `POST http://localhost:8000/predict`

## Resultados

Validacao cruzada estratificada com 5 folds, usando `F1-macro`.

| Modelo | F1-macro medio CV | Desvio padrao |
|---|---:|---:|
| GradientBoosting | 0.7178 | 0.0238 |
| LogisticRegression | 0.7065 | 0.0256 |
| RandomForest | 0.7026 | 0.0235 |

Modelo promovido para producao: `previsor_classificacao_ocorrencia@production`, usando a URI MLflow `models:/previsor_classificacao_ocorrencia@production`.

## Decisoes de modelagem

Foram removidas ou evitadas features de alto risco de vazamento, identificacao direta ou baixa utilidade operacional para inferencia, como numero/historico textual da ocorrencia e campos que nao fazem parte do contrato final da API.

Features derivadas:

- `Ano_da_Ocorrencia`, `Mes_da_Ocorrencia` e `Dia_Semana_da_Ocorrencia` a partir da data.
- Indicadores de ausencia e valor zero para `PMD`, `Numero_de_Assentos`, `ICAO`, `Tipo_de_Aerodromo` e UF indeterminada.
- `PMD_faixa` por faixas de peso, para capturar relacoes nao lineares simples.

Pre-processamento:

- Numericas com mediana, `StandardScaler` e `log1p` em `PMD` e `Numero_de_Assentos`.
- Categoricas com imputacao por `Desconhecido`.
- `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)` para reduzir explosao de cardinalidade e lidar com categorias raras ou novas.
- `compute_sample_weight(class_weight="balanced")` para reduzir impacto do desbalanceamento.

Modelos comparados:

- `LogisticRegression`: baseline linear interpretavel.
- `RandomForestClassifier`: modelo de arvores com bagging para capturar nao linearidades.
- `GradientBoostingClassifier`: boosting sequencial, melhor resultado atual.

## MLflow

O tracking local esta em `pre_processamento/mlruns`. A API carrega o modelo registrado no MLflow Registry durante o `lifespan`, usando:

```text
models:/previsor_classificacao_ocorrencia@production
```

## DVC

Arquivos versionados pelo DVC:

- `data/raw/V_OCORRENCIA_AMPLA.csv.dvc`
- `data/processed/ocorrencia_pre_tratado.csv.dvc`
- `eda/V_OCORRENCIA_AMPLA.csv.dvc`
- `eda/ocorrencia_pre_tratado.csv.dvc`
- `pre_processamento/ocorrencia_pre_tratado.csv.dvc`

## Limitacoes e proximos passos

- A meta de `0,90` em F1-macro ainda nao foi atingida.
- Ha forte dependencia de categorias de alta cardinalidade, como municipio, fabricante e tipo ICAO.
- O dataset possui classes desbalanceadas; tecnicas adicionais de reamostragem ou ajuste de limiar podem melhorar recall das classes minoritarias.
- O contrato da API espera features ja derivadas; um proximo passo e aceitar campos mais brutos e aplicar a engenharia de features dentro do endpoint.
- Testes automatizados da API e do pipeline de treinamento ainda devem ser ampliados.
