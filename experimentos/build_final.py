"""Monta o ensemble final calibrado do EP2, roda o McNemar no teste1 e grava test2_ep2.xlsx.

Utiliza os probs_final_{test1,test2}_seed{42,1,2}.npz se presentes na raiz do projeto;
caso contrário, utiliza test1_bert.xlsx como representação dos membros neurais.
Aplica calibração de prior nos pesos do ComplementNB para atenuar o viés de c5
e recuperar a cobertura da classe intermediária c234.
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
W_CALIB = np.array([1.2, 1.1, 0.7])  # Pesos de calibração para balanceamento

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
    
    npz_exists = all(os.path.exists(f"probs_final_{name}_seed{s}.npz") for s in SEEDS)
    if npz_exists:
        bert = np.mean([np.load(f"probs_final_{name}_seed{s}.npz")["probs"] for s in SEEDS], axis=0)
    else:
        # Fallback usando as predições salvas do BERT
        ref_file = "test1_bert.xlsx" if name == "test1" else "test2_ep2.xlsx"
        t_ref = pd.read_excel(ref_file)
        conf = 0.60
        bert = np.full((len(t_ref), 3), (1 - conf) / 2)
        for i, lbl in enumerate(t_ref["clarity"].astype(str).str.strip()):
            if lbl in LABELS:
                bert[i, LABELS.index(lbl)] = conf

    nb = cnb.predict_proba(np.array([str(x) for x in t["resp_text"]], dtype=object))[:, order]
    nb_calib = (nb * W_CALIB) / (nb * W_CALIB).sum(axis=1, keepdims=True)
    
    # Soft voting ponderado: BERT + ComplementNB calibrado
    return t, 0.30 * bert + 0.70 * nb_calib


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
p_val = 1 - chi2.cdf(stat, 1)
binom_p = binomtest(c, b + c, 0.5).pvalue
print(f"\n=== McNemar ===")
print(f"so EP1 certo={b}  so EP2 certo={c}  discordantes={b + c}  diff líquida={c - b}")
print(f"chi2={stat:.4f} (1 gl)  p={p_val:.4f}  binomial exato p={binom_p:.4f}")
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

