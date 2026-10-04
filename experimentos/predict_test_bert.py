"""Rotula test1.xlsx com o modelo final (BERTimbau), para a entrega.

Treina em TODO o train.xlsx (uma validação de 5% é reservada apenas para o early
stopping) e prediz test1.xlsx, preservando o número de linhas e as colunas do
arquivo original, com rótulos restritos a {c1, c234, c5}.

Rode da raiz do projeto (ou de qualquer lugar; o script entra na raiz):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python experimentos/predict_test_bert.py

Saída: test1_bert.xlsx na raiz do projeto.
"""

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.chdir(Path(__file__).resolve().parent.parent)  # raiz do projeto

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
LABELS = ["c1", "c234", "c5"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
INV = {i: l for l, i in LABEL_MAP.items()}
MAX_LEN = 512
SEED = 42
OUT_PATH = "test1_bert.xlsx"

train = pd.read_excel("train.xlsx").dropna(subset=["resp_text", "clarity"])
train["resp_text"] = train["resp_text"].astype(str)
train["clarity"] = train["clarity"].astype(str).str.strip()
train = train[train["clarity"].isin(LABELS)].copy()
train["label"] = train["clarity"].map(LABEL_MAP)

test = pd.read_excel("test1.xlsx")
test_cols = list(test.columns)
test["resp_text"] = test["resp_text"].fillna("").astype(str)

# treina em TODO o treino; 5% reservado apenas para early stopping
tr, va = train_test_split(train, test_size=0.05, stratify=train["label"], random_state=SEED)
print(f"treino={len(tr)}  val(early stopping)={len(va)}  teste={len(test)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = list(labels) if labels is not None else None

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.enc["input_ids"])

def compute_metrics(pred):
    return {"f1_macro": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="macro")}

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
args = TrainingArguments(
    output_dir="./out_final",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=1,
    seed=SEED,
    report_to="none",
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=TextDataset(tr["resp_text"], tr["label"]),
    eval_dataset=TextDataset(va["resp_text"], va["label"]),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()

pred = trainer.predict(TextDataset(test["resp_text"]))
test["clarity"] = [INV[i] for i in pred.predictions.argmax(-1)]

# formato: mesmas linhas, mesmas colunas (ordem original), rótulos válidos
assert len(test) == len(pd.read_excel("test1.xlsx")), "número de linhas mudou"
assert list(test.columns) == test_cols, "colunas mudaram de ordem/nome"
assert set(test["clarity"]) <= set(LABELS), "rótulo fora de {c1, c234, c5}"

test.to_excel(OUT_PATH, index=False)
print(f"\nsalvo: {OUT_PATH} | linhas: {len(test)}")
print(test["clarity"].value_counts(normalize=True).reindex(LABELS).round(4).to_string())
