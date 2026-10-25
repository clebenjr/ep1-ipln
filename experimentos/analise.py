"""Análises da seção 9 do relatório do EP2.

(i) interpretação global: termos de maior peso por classe
(ii) interpretação local: contribuição por termo em exemplos
(iii) análise de erros: distribuição, estatísticas agregadas e casos manuais

A interpretação usa os coeficientes de uma regressão logística sobre TF-IDF de
palavras, equivalente ao que a eli5 extrai para modelos lineares.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

LABELS = ["c1", "c234", "c5"]
SEEDS = [42, 1, 2]

train = pd.read_excel("train.xlsx").dropna(subset=["resp_text", "clarity"])
train["resp_text"] = train["resp_text"].astype(str)
train["clarity"] = train["clarity"].astype(str).str.strip()
train = train[train["clarity"].isin(LABELS)]
X = np.array([str(t) for t in train["resp_text"]], dtype=object)
y = np.array([str(v) for v in train["clarity"]], dtype=object)

pipe = Pipeline([
    ("v", TfidfVectorizer(min_df=5, ngram_range=(1, 2))),
    ("c", LogisticRegression(max_iter=2000, random_state=42)),
]).fit(X, y)
vec, clf = pipe["v"], pipe["c"]
feat = np.array(vec.get_feature_names_out())
cls = list(clf.classes_)

print("=== (i) interpretacao global: top-12 por classe ===")
for k in LABELS:
    w = clf.coef_[cls.index(k)]
    top = np.argsort(w)[::-1][:12]
    print(f"\n{k} a favor : " + " | ".join(f"{feat[i]} {w[i]:+.2f}" for i in top))
    print(f"{k} contra  : " + " | ".join(f"{feat[i]} {w[i]:+.2f}" for i in np.argsort(w)[:6]))

# predicoes do ensemble final no teste1
t1 = pd.read_excel("test1.xlsx")
t1["resp_text"] = t1["resp_text"].fillna("").astype(str)
gold = np.array([str(v) for v in pd.read_excel("test1-rotulado-para-teste-estatistico.xlsx")["clarity"]], dtype=object)
bert = np.mean([np.load(f"probs_final_test1_seed{s}.npz")["probs"] for s in SEEDS], axis=0)
cnb = Pipeline([
    ("v", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)),
    ("c", ComplementNB(alpha=0.3)),
]).fit(X, y)
nb = cnb.predict_proba(np.array([str(x) for x in t1["resp_text"]], dtype=object))[:, [list(cnb.classes_).index(k) for k in LABELS]]
E = (bert + nb) / 2
pred = np.array([LABELS[i] for i in E.argmax(1)], dtype=object)


def explica(i, n=8):
    txt = t1["resp_text"].iloc[i]
    x = vec.transform([txt])
    k = cls.index(pred[i])
    contrib = sorted(((feat[j], x[0, j] * clf.coef_[k, j]) for j in x.nonzero()[1]),
                     key=lambda z: -abs(z[1]))
    print(f"\n[linha {i}] gold={gold[i]} predito={pred[i]} probs={dict(zip(LABELS, E[i].round(3)))}")
    print(f"  texto: {txt[:150]}".replace("\n", " "))
    print(f"  termos: " + " | ".join(f"{w} {v:+.3f}" for w, v in contrib[:n]))


ok = np.where(pred == gold)[0]
err = np.where(pred != gold)[0]

print("\n\n=== (ii) interpretacao local (acertos) ===")
for i in ok[:2]:
    explica(i)

print("\n\n=== (iii) analise de erros ===")
print(f"erros: {len(err)}/{len(gold)}")
print("\npor par (linha=gold, coluna=predito):")
print(pd.crosstab(gold[err], pred[err]).to_string())
lens = t1["resp_text"].str.len()
print(f"\ncomprimento medio: acertos={lens.iloc[ok].mean():.0f} erros={lens.iloc[err].mean():.0f}")
conf = E.max(1)
print(f"confianca media  : acertos={conf[ok].mean():.3f} erros={conf[err].mean():.3f}")
print("\n--- casos para discussao manual ---")
for i in err[:4]:
    explica(i, 6)
