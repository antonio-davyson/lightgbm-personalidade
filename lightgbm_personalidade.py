"""Classificação multiclasse de personalidade (Introvert / Extrovert / Ambivert) com LightGBM."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score, f1_score, log_loss,
                             roc_auc_score, ConfusionMatrixDisplay)

# 1. Carregamento e exploração
df = pd.read_csv("personality_synthetic_dataset.csv")
print(df.shape, "| nulos:", df.isna().sum().sum(), "| duplicados:", df.duplicated().sum())
print(df.personality_type.value_counts(normalize=True).round(3))

# 2. Preparação (sem nulos; árvores não precisam de normalização)
le = LabelEncoder()
y = le.fit_transform(df["personality_type"])
X = df.drop(columns="personality_type")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Baseline e validação cruzada
cv = StratifiedKFold(5, shuffle=True, random_state=42)
base = LGBMClassifier(objective="multiclass", random_state=42, verbose=-1)
print("Baseline CV accuracy:", cross_val_score(base, Xtr, ytr, cv=cv, scoring="accuracy").mean().round(4))

# 4. Otimização de hiperparâmetros
space = {
    "n_estimators": randint(100, 600),
    "learning_rate": uniform(0.01, 0.15),
    "num_leaves": randint(8, 64),
    "max_depth": [-1, 4, 6, 8],
    "min_child_samples": randint(10, 60),
    "subsample": uniform(0.6, 0.4),
    "subsample_freq": [1],
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": uniform(0, 5),
}
search = RandomizedSearchCV(base, space, n_iter=20, scoring="f1_macro", cv=3,
                            random_state=42, n_jobs=-1, verbose=1)
search.fit(Xtr, ytr)
print("Melhores parâmetros:", search.best_params_)
print("CV F1 macro (otimizado):", round(search.best_score_, 4))
best = search.best_estimator_

# 5. Avaliação no teste
pred = best.predict(Xte); proba = best.predict_proba(Xte)
print(classification_report(yte, pred, target_names=le.classes_))
print("Acurácia:", round(accuracy_score(yte, pred), 4), "| F1 macro:", round(f1_score(yte, pred, average="macro"), 4))
print("Log loss:", round(log_loss(yte, proba), 4), "| ROC AUC (ovr):", round(roc_auc_score(yte, proba, multi_class="ovr"), 4))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=le.classes_, ax=ax[0], colorbar=False)
ax[0].set_title("Matriz de confusão")
imp = pd.Series(best.feature_importances_, index=X.columns).sort_values().tail(15)
imp.plot.barh(ax=ax[1], title="Top 15 importâncias")
plt.tight_layout(); plt.savefig("avaliacao.png", dpi=110)
