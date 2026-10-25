"""Ensemble das 3 sementes do BERT ja treinadas, sem retreinar.

Carrega os checkpoints de out_42/out_1/out_2, gera as probabilidades no holdout
e compara soft voting, hard voting e as sementes isoladas. Salva
probs_holdout.npz para reaproveitar em outros experimentos.
"""

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DATA_DIR = Path(__file__).resolve().parent.parent
os.chdir(DATA_DIR)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
LABELS = ["c1", "c234", "c5"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 512
SEEDS = [42, 1, 2]
OUT_NPZ = "probs_holdout.npz"


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def find_checkpoint(seed):
    d = Path(f"out_{seed}")
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not cks:
        raise SystemExit(f"nenhum checkpoint em {d}")
    return cks[-1]


df = pd.read_excel("train.xlsx").dropna(subset=["resp_text", "clarity"])
df["resp_text"] = df["resp_text"].astype(str)
df["clarity"] = df["clarity"].astype(str).str.strip()
df = df[df["clarity"].isin(LABELS)].copy()
df["label"] = df["clarity"].map(LABEL_MAP)
_, hold = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
y = hold["label"].to_numpy()
texts = hold["resp_text"].tolist()
print(f"holdout: {len(texts)} exemplos")

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_probs = {}
for seed in SEEDS:
    ck = find_checkpoint(seed)
    model = AutoModelForSequenceClassification.from_pretrained(ck).to(device).eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=MAX_LEN, return_tensors="pt").to(device)
            logits.append(model(**enc).logits.float().cpu().numpy())
            if i % 960 == 0:
                print(f"  seed {seed}: {i}/{len(texts)}", flush=True)
    p = softmax(np.vstack(logits))
    all_probs[f"seed_{seed}"] = p
    print(f"seed {seed} ({ck.name}): acc={accuracy_score(y, p.argmax(1)):.4f}")
    del model

np.savez(OUT_NPZ, y=y, **all_probs)
print(f"\nprobabilidades salvas em {OUT_NPZ}")

P = np.stack([all_probs[f"seed_{s}"] for s in SEEDS])

soft = P.mean(axis=0).argmax(1)
print(f"\n=== ensemble no holdout ({len(y)}) ===")
print(f"soft voting (media das probs): acc={accuracy_score(y, soft):.4f} "
      f"f1m={f1_score(y, soft, average='macro'):.4f}")

votes = P.argmax(2)
hard = np.array([np.bincount(votes[:, i], minlength=3).argmax() for i in range(len(y))])
print(f"hard voting (voto maioria):    acc={accuracy_score(y, hard):.4f} "
      f"f1m={f1_score(y, hard, average='macro'):.4f}")

best_single = max(accuracy_score(y, all_probs[f"seed_{s}"].argmax(1)) for s in SEEDS)
print(f"\nmelhor semente isolada:        acc={best_single:.4f}")
print(f"ganho do soft voting:          {accuracy_score(y, soft) - best_single:+.4f}")
