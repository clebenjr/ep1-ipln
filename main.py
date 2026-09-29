"""
EP1 (ACH2118): avaliação e seleção de classificadores.

Protocolo (Holdout + validação cruzada):
  1. train.xlsx é dividido (estratificado) em dev (80%) e holdout (20%).
     O holdout fica intocado até o final.
  2. Cada classificador do registro MODELS é avaliado com validação cruzada
     estratificada no dev; o ranking é feito pela acurácia média.
  3. O melhor modelo e o baseline oficial são re-treinados no dev inteiro e
     avaliados uma única vez no holdout (estimativa honesta pós-seleção).
  4. Com --predict, o melhor modelo é re-treinado em todo o treino e rotula
     test1.xlsx.

Uso:
    python main.py --list
    python main.py                                  # todos os modelos
    python main.py --models tfidf_linearsvc tfidf_word_char_svc
    python main.py --predict                        # também rotula o teste
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from baseline import DATA_DIR, LABEL_COL, LABELS, SEED, TEST_PATH, TEXT_COL, build_baseline, load_data

RESULTS_DIR = DATA_DIR / "results"

MAJORITY = "majority_class"
BASELINE = "baseline_tfidf_logreg"


# ---------------------------------------------------------------------------
# Registro de modelos
#
# Para adicionar um classificador, registre uma factory sem argumentos que
# devolva um estimador compatível com sklearn e que receba TEXTO CRU:
#     fit(list[str], y) / predict(list[str])
# Assim todo pré-processamento (vetorização, embeddings...) acontece dentro
# de cada fold, sem vazamento de informação.
#
# Modelos que usam GPU (embeddings, fine-tuning de BERT) devem ser
# implementados como uma classe BaseEstimator + ClassifierMixin com fit/predict
# e registrados com parallel=False, para que os folds rodem em sequência.
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    factory: Callable[[], BaseEstimator]
    parallel: bool = True
    description: str = ""


def word_tfidf(**kwargs):
    params = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)
    params.update(kwargs)
    return TfidfVectorizer(**params)


def char_tfidf(**kwargs):
    params = dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=300_000)
    params.update(kwargs)
    return TfidfVectorizer(**params)


MODELS: dict[str, ModelSpec] = {
    MAJORITY: ModelSpec(
        lambda: DummyClassifier(strategy="most_frequent"),
        description="Piso mínimo: sempre prevê a classe mais frequente",
    ),
    BASELINE: ModelSpec(
        build_baseline,
        description="Baseline oficial (baseline.py): TF-IDF unigramas + RegLog",
    ),
    "tfidf_bigram_logreg": ModelSpec(
        lambda: Pipeline([
            ("tfidf", word_tfidf()),
            ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=SEED)),
        ]),
        description="TF-IDF uni+bigramas (sublinear) + RegLog",
    ),
    "tfidf_linearsvc": ModelSpec(
        lambda: Pipeline([
            ("tfidf", word_tfidf()),
            ("clf", LinearSVC(C=0.5, random_state=SEED)),
        ]),
        description="TF-IDF uni+bigramas + LinearSVC",
    ),
    "tfidf_char_linearsvc": ModelSpec(
        lambda: Pipeline([
            ("tfidf", char_tfidf()),
            ("clf", LinearSVC(C=0.5, random_state=SEED)),
        ]),
        description="TF-IDF char n-grams (3-5) + LinearSVC",
    ),
    "tfidf_word_char_svc": ModelSpec(
        lambda: Pipeline([
            ("features", FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])),
            ("clf", LinearSVC(C=0.3, random_state=SEED)),
        ]),
        description="TF-IDF palavras + char n-grams + LinearSVC",
    ),
    "tfidf_complementnb": ModelSpec(
        lambda: Pipeline([
            ("tfidf", word_tfidf()),
            ("clf", ComplementNB(alpha=0.3)),
        ]),
        description="TF-IDF uni+bigramas + Complement Naive Bayes",
    ),
    "tfidf_sgd": ModelSpec(
        lambda: Pipeline([
            ("tfidf", word_tfidf()),
            ("clf", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=50, random_state=SEED)),
        ]),
        description="TF-IDF uni+bigramas + SGD (modified huber)",
    ),
}

# Sempre avaliados, mesmo quando --models filtra a lista
REFERENCE_MODELS = [MAJORITY, BASELINE]


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def cross_validate_model(name, spec, X, y, cv):
    start = time.perf_counter()
    scores = cross_validate(
        spec.factory(), X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
        n_jobs=-1 if spec.parallel else 1,
    )
    elapsed = time.perf_counter() - start
    return {
        "model": name,
        "acc_mean": scores["test_accuracy"].mean(),
        "acc_std": scores["test_accuracy"].std(),
        "f1_macro": scores["test_f1_macro"].mean(),
        "acc_train": scores["train_accuracy"].mean(),
        "gap": scores["train_accuracy"].mean() - scores["test_accuracy"].mean(),
        "time_s": elapsed,
    }


def print_ranking(results):
    base_acc = results.loc[results["model"] == BASELINE, "acc_mean"].iloc[0]
    table = results.assign(
        acc_cv=results.apply(lambda r: f"{r.acc_mean:.4f} ± {r.acc_std:.4f}", axis=1),
        vs_baseline=results["acc_mean"] - base_acc,
    )[["model", "acc_cv", "vs_baseline", "f1_macro", "acc_train", "gap", "time_s"]]
    print("\n=== Ranking (validação cruzada no dev) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def evaluate_on_holdout(name, X_dev, y_dev, X_hold, y_hold):
    model = MODELS[name].factory().fit(X_dev, y_dev)
    y_pred = model.predict(X_hold)
    acc = accuracy_score(y_hold, y_pred)
    print(f"\n--- {name} no holdout ---")
    print(f"Acurácia: {acc:.4f} | F1-macro: {f1_score(y_hold, y_pred, average='macro'):.4f}")
    print(classification_report(y_hold, y_pred, labels=LABELS, digits=4))
    print("Matriz de confusão (linhas = real, colunas = predito):")
    print(pd.DataFrame(confusion_matrix(y_hold, y_pred, labels=LABELS), index=LABELS, columns=LABELS))
    return acc


def predict_test(name, X, y):
    test = pd.read_excel(TEST_PATH)
    test[TEXT_COL] = test[TEXT_COL].fillna("").astype(str)

    model = MODELS[name].factory().fit(X, y)
    out = test.copy()
    out[LABEL_COL] = model.predict(test[TEXT_COL])

    assert len(out) == len(test), "número de linhas diferente do teste original"
    assert list(out.columns) == list(test.columns), "colunas diferentes do teste original"
    assert set(out[LABEL_COL]) <= set(LABELS), "rótulos fora de {c1, c234, c5}"

    path = DATA_DIR / f"test1_{name}.xlsx"
    out.to_excel(path, index=False)
    print(f"\nTeste rotulado com '{name}' salvo em: {path.name}")
    print(out[LABEL_COL].value_counts(normalize=True).reindex(LABELS).round(4).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Avaliação e seleção de classificadores (EP1)")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), help="modelos a avaliar (padrão: todos)")
    parser.add_argument("--folds", type=int, default=5, help="número de folds da validação cruzada")
    parser.add_argument("--holdout", type=float, default=0.2, help="fração do treino reservada como holdout")
    parser.add_argument("--predict", action="store_true", help="re-treina o melhor em todo o treino e rotula o teste")
    parser.add_argument("--list", action="store_true", help="lista os modelos registrados e sai")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        for name, spec in MODELS.items():
            print(f"{name:<24} {spec.description}")
        return

    selected = list(dict.fromkeys(REFERENCE_MODELS + (args.models or list(MODELS))))

    train, _ = load_data()
    X, y = train[TEXT_COL], train[LABEL_COL]
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=args.holdout, stratify=y, random_state=SEED,
    )
    print(f"Dev: {len(X_dev)} | Holdout: {len(X_hold)} | Folds: {args.folds}")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    rows = []
    for name in selected:
        print(f"Avaliando {name}...", flush=True)
        rows.append(cross_validate_model(name, MODELS[name], X_dev, y_dev, cv))

    results = pd.DataFrame(rows).sort_values("acc_mean", ascending=False, ignore_index=True)
    print_ranking(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"cv_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path.relative_to(DATA_DIR)}")

    candidates = results[~results["model"].isin([MAJORITY])]
    best = candidates.iloc[0]["model"]
    best_cv = candidates.iloc[0]["acc_mean"]
    print(f"\nMelhor modelo na CV: {best} ({best_cv:.4f})")

    best_hold = evaluate_on_holdout(best, X_dev, y_dev, X_hold, y_hold)
    if best != BASELINE:
        base_hold = evaluate_on_holdout(BASELINE, X_dev, y_dev, X_hold, y_hold)
        print(f"\nHoldout: {best} = {best_hold:.4f} vs baseline = {base_hold:.4f} "
              f"(Δ = {best_hold - base_hold:+.4f})")
        if best_hold <= base_hold:
            print("ATENÇÃO: o melhor modelo da CV não superou o baseline no holdout.")
    if best_cv - best_hold > 0.02:
        print(f"ATENÇÃO: holdout {best_cv - best_hold:.4f} abaixo da CV — possível overfitting de seleção.")

    if args.predict:
        predict_test(best, X, y)


if __name__ == "__main__":
    main()
