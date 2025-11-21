# DM_Project_Final

# TODO

## Step 1: One-Hot Encoding (FEITO)
- Converter variáveis categóricas (`Gender`, `Education`, `Marital Status`, `LoyaltyStatus`, `EnrollmentType`) em variáveis numéricas.
- Usar `pd.get_dummies` ou `OneHotEncoder` do scikit-learn.
- Objetivo: permitir que os algoritmos de clustering processem dados categóricos.

---

## Step 2: Feature Engineering (Em Andamento)
- Criar novas variáveis derivadas (já comecei a fazer rever/adicionar)
  - `PointsUtilizationRate = PointsRedeemed / (PointsAccumulated + 1)`
  - `CLVperFlight = CustomerLifetimeValue / (NumFlights + 1)` (Feito)
  - `Tenure = tempo desde EnrollmentDateOpening` (Feito)
  - `EngagementScore = (NumFlights * PointsAccumulated) / (recency_months + 1)`
- Objetivo: enriquecer os dados com métricas que capturam comportamento e valor.

---

## Step 3: Divisão de variáveis (Feito/para revisão)
- Separar colunas em:
  - **Métricas úteis** (Income, CLV, NumFlights, Points, Tenure, etc.)
  - **Não métricas** (IDs, nomes, latitude/longitude)
  - **Unused** (redundantes ou irrelevantes)
- Objetivo: manter apenas variáveis relevantes para clustering.

---

## Step 4: Outlier Removal (DBSCAN) 
- Aplicar DBSCAN para identificar e remover outliers.
- Objetivo: evitar que valores extremos distorçam os clusters.

---

## Step 5: Definição de Perspectivas
- Criar subconjuntos de features com o pbjetivo de analisar diferentes dimensões do cliente separadamente.

---

## Step 6: Clustering por Perspectiva
- Aplicar K-Means em cada subconjunto de features.
- Objetivo: obter clusters específicos para cada dimensão.

---

## Step 7: Merge de Clusters com Centroides
- Combinar resultados dos diferentes clusters através da média dos centroides.
- Objetivo: criar uma visão integrada dos segmentos.

---

## Step 8: Hierarchical Clustering
- Usar dendograma para definir número ótimo de clusters.
- Objetivo: validar e complementar os clusters obtidos.

---

## Step 9: Profiling & Feature Importance
- Calcular médias por cluster e visualizar com heatmaps.
- Objetivo: interpretar os clusters e dar significado de negócio.

