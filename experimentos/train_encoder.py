"""Fine-tuning de um encoder no protocolo do EP1, para comparar arquiteturas.

Holdout 20% (seed 42) intocado, validacao do dev para early stopping, avaliacao
unica no holdout. Padding dinamico por batch. Salva probs_<tag>.npz.

Uso:
  python experimentos/train_encoder.py --model <hf_id> --tag <nome> --lr 1e-5 --epochs 3
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

LABELS = ["c1", "c234", "c5"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
ap.add_argument("--tag", required=True, help="sufixo dos arquivos de saída")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=float, default=4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-len", type=int, default=384)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df = pd.read_excel("train.xlsx").dropna(subset=["resp_text", "clarity"])
df["resp_text"] = df["resp_text"].astype(str)
df["clarity"] = df["clarity"].astype(str).str.strip()
df = df[df["clarity"].isin(LABELS)].copy()
df["label"] = df["clarity"].map(LABEL_MAP)

dev_df, hold_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
tr, va = train_test_split(dev_df, test_size=0.1, stratify=dev_df["label"], random_state=args.seed)
hold_labels = hold_df["label"].to_numpy()
print(f"modelo={args.model} tag={args.tag} lr={args.lr} epochs={args.epochs}")
print(f"treino={len(tr)} val={len(va)} holdout={len(hold_df)}")

tokenizer = AutoTokenizer.from_pretrained(args.model)


class DS(torch.utils.data.Dataset):

    def __init__(self, texts, labels=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=args.max_len)
        self.labels = list(labels) if labels is not None else None

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item

    def __len__(self):
        return len(self.enc["input_ids"])


def compute_metrics(pred):
    return {"f1_macro": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="macro")}


model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
targs = TrainingArguments(
    output_dir=f"./out_{args.tag}",
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

pred = trainer.predict(DS(hold_df["resp_text"], hold_df["label"]))
logits = pred.predictions
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
y = logits.argmax(-1)

np.savez(f"probs_{args.tag}.npz", y=hold_labels, probs=probs)
acc = accuracy_score(hold_labels, y)
f1 = f1_score(hold_labels, y, average="macro")
print(f"\n=== {args.tag} no holdout ===")
print(f"acuracia={acc:.4f}  f1_macro={f1:.4f}")
print(classification_report(hold_labels, y, target_names=LABELS, digits=4))
print(pd.DataFrame(confusion_matrix(hold_labels, y), index=LABELS, columns=LABELS))
pd.DataFrame([{"tag": args.tag, "model": args.model, "lr": args.lr,
               "epochs": args.epochs, "acc": acc, "f1_macro": f1}]
             ).to_csv(f"result_{args.tag}.csv", index=False)
