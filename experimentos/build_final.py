"""Monta o ensemble final do EP2, roda o McNemar no teste1 e grava test2_ep2.xlsx.

Requer os probs_final_{test1,test2}_seed{42,1,2}.npz na raiz do projeto
(gerados por colab-membros-finais.ipynb ou final_member.py).
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

cnb = Pipeline([
    ("v", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)),
    ("c", ComplementNB(alpha=0.3)),
]).fit(X, y)
order = [list(cnb.classes_).index(k) for k in LABELS]


def ensemble(name):
    t = pd.read_excel(f"{name}.xlsx")
    t["resp_text"] = t["resp_text"].fillna("").astype(str)
    bert = np.mean([np.load(f"probs_final_{name}_seed{s}.npz")["probs"] for s in SEEDS], axis=0)
    nb = cnb.predict_proba(np.array([str(x) for x in t["resp_text"]], dtype=object))[:, order]
    return t, (bert + nb) / 2


# teste1: avaliacao e comparacao com o EP1
t1, E1 = ensemble("test1")
gold = np.array([str(v) for v in pd.read_excel("test1-rotulado-para-teste-estatistico.xlsx")["clarity"]], dtype=object)
ep1 = np.array([str(v) for v in pd.read_excel("test1_bert.xlsx")["clarity"]], dtype=object)
ep2 = np.array([LABELS[i] for i in E1.argmax(1)], dtype=object)

print("=== teste1 (900) ===")
print(f"EP1 : acc={accuracy_score(gold, ep1):.4f}")
print(f"EP2 : acc={accuracy_score(gold, ep2):.4f} f1m={f1_score(gold, ep2, average='macro', labels=LABELS):.4f}")

b = ((ep1 == gold) & (ep2 != gold)).sum()
c = ((ep1 != gold) & (ep2 == gold)).sum()
stat = (abs(b - c) - 1) ** 2 / (b + c)
print(f"\n=== McNemar ===")
print(f"so EP1 certo={b}  so EP2 certo={c}  discordantes={b + c}")
print(f"chi2={stat:.4f} (1 gl)  p={1 - chi2.cdf(stat, 1):.4f}  "
      f"binomial exato p={binomtest(c, b + c, 0.5).pvalue:.4f}")
print("\n" + classification_report(gold, ep2, labels=LABELS, digits=4))
print(pd.DataFrame(confusion_matrix(gold, ep2, labels=LABELS), index=LABELS, columns=LABELS))

# teste2: planilha de entrega
t2, E2 = ensemble("test2")
cols = list(pd.read_excel("test2.xlsx").columns)
out = t2.copy()
out["clarity"] = [LABELS[i] for i in E2.argmax(1)]
assert len(out) == 897 and list(out.columns) == cols
assert set(out["clarity"]) <= set(LABELS) and not out["clarity"].isna().any()
out.to_excel("test2_ep2.xlsx", index=False)
print(f"\n=== test2_ep2.xlsx ({len(out)} linhas) ===")
print(out["clarity"].value_counts(normalize=True).reindex(LABELS).round(4).to_string())
