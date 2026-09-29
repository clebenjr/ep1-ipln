"""
Baseline oficial do EP1 (ACH2118): TF-IDF + Regressão Logística.

Tarefa: classificação ternária da clareza de respostas do e-SIC em {c1, c234, c5}.

O script:
  1. Carrega train.xlsx (colunas resp_text, clarity).
  2. Avalia com validação cruzada estratificada (5 folds):
       - baseline de classe majoritária (piso mínimo exigido);
       - TF-IDF + Regressão Logística (baseline a ser superado).
  3. Treina o TF-IDF + RegLog em todo o treino e rotula test1.xlsx,
     gerando uma planilha no mesmo formato (mesmas linhas/colunas).

Uso:
    pip install pandas openpyxl scikit-learn
    python baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.xlsx"
TEST_PATH = DATA_DIR / "test1.xlsx"
OUTPUT_PATH = DATA_DIR / "test1_baseline.xlsx"

TEXT_COL = "resp_text"
LABEL_COL = "clarity"
LABELS = ["c1", "c234", "c5"]

SEED = 42
N_FOLDS = 5


def load_data():
    train = pd.read_excel(TRAIN_PATH)
    test = pd.read_excel(TEST_PATH)

    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    test[TEXT_COL] = test[TEXT_COL].fillna("").astype(str)
    train[LABEL_COL] = train[LABEL_COL].astype(str).str.strip()

    unknown = set(train[LABEL_COL]) - set(LABELS)
    if unknown:
        raise ValueError(f"Rótulos inesperados no treino: {unknown}")

    return train, test


def build_baseline():
    """TF-IDF (unigramas) + Regressão Logística com parâmetros padrão."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents=None)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def evaluate(name, model, X, y, cv):
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True, n_jobs=-1,
    )
    acc, f1 = scores["test_accuracy"], scores["test_f1_macro"]
    train_acc = scores["train_accuracy"]
    print(f"\n=== {name} ===")
    print(f"Acurácia (CV):  {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"F1-macro (CV):  {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"Acurácia treino: {train_acc.mean():.4f}  (gap = {train_acc.mean() - acc.mean():.4f})")
    return acc.mean()


def main():
    train, test = load_data()
    X, y = train[TEXT_COL], train[LABEL_COL]

    print(f"Treino: {len(train)} exemplos | Teste: {len(test)} exemplos")
    print("Distribuição de classes no treino:")
    print(y.value_counts(normalize=True).reindex(LABELS).round(4).to_string())

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    evaluate("Classe majoritária", DummyClassifier(strategy="most_frequent"), X, y, cv)
    evaluate("TF-IDF + Regressão Logística", build_baseline(), X, y, cv)

    # Relatório detalhado por classe a partir das predições out-of-fold
    y_pred = cross_val_predict(build_baseline(), X, y, cv=cv, n_jobs=-1)
    print("\nRelatório por classe (out-of-fold):")
    print(classification_report(y, y_pred, labels=LABELS, digits=4))
    print("Matriz de confusão (linhas = real, colunas = predito):")
    print(pd.DataFrame(confusion_matrix(y, y_pred, labels=LABELS), index=LABELS, columns=LABELS))

    # Treino final em todo o conjunto e rotulação do teste
    model = build_baseline().fit(X, y)
    out = test.copy()
    out[LABEL_COL] = model.predict(test[TEXT_COL])
    out.to_excel(OUTPUT_PATH, index=False)

    assert len(out) == len(test) and list(out.columns) == list(test.columns)
    print(f"\nTeste rotulado salvo em: {OUTPUT_PATH.name}")
    print(out[LABEL_COL].value_counts(normalize=True).reindex(LABELS).round(4).to_string())


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
