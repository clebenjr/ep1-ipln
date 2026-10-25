"""Treina um membro BERT do ensemble final em todo o train e prediz test1/test2.

Salva probs_final_{test1,test2}_seed<N>.npz. Hiperparametros identicos aos
membros medidos no holdout (lr 2e-5, 4 epocas, batch 8); 10% do train e usado
apenas para early stopping.

Uso:
  python experimentos/final_member.py --seed 1
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

MODEL = "neuralmind/bert-base-portuguese-cased"
LABELS = ["c1", "c234", "c5"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 512

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
args = ap.parse_args()

train = pd.read_excel("train.xlsx").dropna(subset=["resp_text", "clarity"])
train["resp_text"] = train["resp_text"].astype(str)
train["clarity"] = train["clarity"].astype(str).str.strip()
train = train[train["clarity"].isin(LABELS)].copy()
train["label"] = train["clarity"].map(LABEL_MAP)

tr, va = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=args.seed)
print(f"seed={args.seed} | treino={len(tr)} val={len(va)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)


class DS(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = list(labels) if labels is not None else None

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            d["labels"] = self.labels[i]
        return d

    def __len__(self):
        return len(self.enc["input_ids"])


def compute_metrics(p):
    return {"f1_macro": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
targs = TrainingArguments(
    output_dir=f"./out_final_s{args.seed}",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=32,
    learning_rate=args.lr,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=1,
    seed=args.seed,
    report_to="none",
)
trainer = Trainer(
    model=model, args=targs,
    train_dataset=DS(tr["resp_text"], tr["label"]),
    eval_dataset=DS(va["resp_text"], va["label"]),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()

for name, path in [("test1", "test1.xlsx"), ("test2", "test2.xlsx")]:
    t = pd.read_excel(path)
    t["resp_text"] = t["resp_text"].fillna("").astype(str)
    pr = trainer.predict(DS(t["resp_text"]))
    probs = torch.softmax(torch.tensor(pr.predictions.astype("float32")), dim=-1).numpy()
    out = f"probs_final_{name}_seed{args.seed}.npz"
    np.savez(out, probs=probs)
    dist = pd.Series([LABELS[i] for i in probs.argmax(1)]).value_counts(normalize=True)
    print(f"\n{name}: {len(t)} linhas -> {out}")
    print(dist.reindex(LABELS).round(4).to_string())
