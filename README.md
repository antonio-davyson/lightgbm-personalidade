# Classificação de personalidade com LightGBM

Desafio Rocketseat: prever o tipo de personalidade (**Introvert**, **Extrovert** ou **Ambivert**) a partir de 29 traços psicológicos (escala 0–10). Dataset sintético do Kaggle "Introvert, Extrovert & Ambivert Classification" com 20.000 registros, sem valores ausentes e com classes balanceadas (~33% cada).

## Abordagem
1. Análise exploratória: distribuição das variáveis e relação dos traços com o alvo
2. Preparação: `LabelEncoder` no alvo e split estratificado 80/20 (árvores dispensam normalização)
3. Baseline: `LGBMClassifier` padrão com validação cruzada estratificada
4. Otimização: `RandomizedSearchCV` (F1 macro) em `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree` e `reg_lambda`
5. Avaliação no teste: acurácia, F1 macro, log loss, ROC AUC (one-vs-rest), matriz de confusão e importância das variáveis

## Resultados
> Preencha após rodar o notebook (métricas, gráfico e principais descobertas).

## Como rodar
```bash
pip install -r requirements.txt
jupyter notebook lightgbm_personalidade.ipynb   # ou: python lightgbm_personalidade.py
```
