"""Runner do notebook experimento-codigo-bert.ipynb (código idêntico às células).
Gerado para execução longa/não supervisionada. Rode da raiz do projeto:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python experimentos/run_bert.py
Resultados por seed em bert_results.csv (persistidos incrementalmente)."""
import os
from pathlib import Path
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.chdir(Path(__file__).resolve().parent.parent)  # raiz do projeto (train.xlsx, bert_results.csv)

# BERTimbau fine-tuning (versão revisada)
# Melhorias sobre a 1a versão:
#  - split em 3 vias: holdout (20%, seed 42) INTOCADO ate a avaliacao final;
#    validacao separada (tirada do dev) para early stopping -> sem "peek" no holdout;
#  - learning rate 2e-5 e early stopping pelo F1-macro da validacao;
#  - repeticao em varias seeds, reportando media +/- desvio no holdout.
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
LABELS = ["c1", "c234", "c5"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
SEEDS = [42, 1, 2]        # media +/- desvio sobre estas sementes
MAX_LEN = 512

df = pd.read_excel("train.xlsx").dropna(subset=["resp_text", "clarity"])
df["resp_text"] = df["resp_text"].astype(str)
df["clarity"] = df["clarity"].astype(str).str.strip()
df = df[df["clarity"].isin(LABELS)].copy()
df["label"] = df["clarity"].map(LABEL_MAP)

# MESMO holdout dos modelos finais (main.py): 20% estratificado, seed 42, intocado.
dev_df, hold_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
hold_labels = hold_df["label"].to_numpy()
print(f"dev={len(dev_df)}  holdout={len(hold_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = list(labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"f1_macro": f1_score(pred.label_ids, preds, average="macro")}

def run_seed(seed):
    # validacao tirada do DEV (nao do holdout) -> early stopping honesto
    tr, va = train_test_split(dev_df, test_size=0.1, stratify=dev_df["label"], random_state=seed)
    ds_tr = TextDataset(tr["resp_text"], tr["label"])
    ds_va = TextDataset(va["resp_text"], va["label"])
    ds_ho = TextDataset(hold_df["resp_text"], hold_df["label"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    args = TrainingArguments(
        output_dir=f"./out_{seed}",
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
        seed=seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_va,   # seleciona o melhor checkpoint pela VALIDACAO
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    pred = trainer.predict(ds_ho)                   # avaliacao UNICA no holdout
    y = pred.predictions.argmax(-1)
    return accuracy_score(hold_labels, y), f1_score(hold_labels, y, average="macro"), y

import os
RESULTS_CSV = "bert_results.csv"   # persiste a cada seed (sobrevive a falhas)

accs, f1s, last_pred, rows = [], [], None, []
for s in SEEDS:
    acc, f1, y = run_seed(s)
    accs.append(acc); f1s.append(f1); last_pred = y
    rows.append({"seed": s, "acc": acc, "f1_macro": f1})
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"seed {s}: acc={acc:.4f}  f1_macro={f1:.4f}  (salvo em {RESULTS_CSV})")

accs, f1s = np.array(accs), np.array(f1s)
print(f"\nHoldout em {len(SEEDS)} seeds:")
print(f"  acuracia : {accs.mean():.4f} +/- {accs.std():.4f}")
print(f"  f1_macro : {f1s.mean():.4f} +/- {f1s.std():.4f}")

print("\nRelatorio da ultima seed:")
print(classification_report(hold_labels, last_pred, target_names=LABELS, digits=4))
print("Matriz de confusao (linhas=real, colunas=predito):")
print(pd.DataFrame(confusion_matrix(hold_labels, last_pred), index=LABELS, columns=LABELS))
