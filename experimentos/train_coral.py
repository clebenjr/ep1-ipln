"""CORAL (COnsistent RAnk Logits) sobre o BERTimbau: abordagem ordinal.

Para K=3 classes, prediz K-1=2 limiares cumulativos ("y > c1?", "y > c234?")
com um vetor de pesos compartilhado e um vies por limiar, o que garante
consistencia de ranque. Probabilidades por classe saem por diferenca dos
cumulativos, permitindo usar o modelo como membro de ensemble.

Mesmo protocolo do EP1. Salva probs_<tag>.npz.

Uso:
  python experimentos/train_coral.py --tag coral --epochs 3
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["c1", "c234", "c5"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
K = 3

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
ap.add_argument("--tag", default="coral")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-len", type=int, default=384)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "mps" if torch.backends.mps.is_available() else "cpu"

df = pd.read_excel("train.xlsx").dropna(subset=["resp_text", "clarity"])
df["resp_text"] = df["resp_text"].astype(str)
df["clarity"] = df["clarity"].astype(str).str.strip()
df = df[df["clarity"].isin(LABELS)].copy()
df["label"] = df["clarity"].map(LABEL_MAP)

dev_df, hold_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
tr, va = train_test_split(dev_df, test_size=0.1, stratify=dev_df["label"], random_state=args.seed)
print(f"treino={len(tr)} val={len(va)} holdout={len(hold_df)} | device={device}")

tokenizer = AutoTokenizer.from_pretrained(args.model)


class DS(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def collate(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), truncation=True, max_length=args.max_len,
                    padding=True, return_tensors="pt")
    return enc, torch.tensor(labels)


class CoralBert(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(h, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(K - 1))

    def forward(self, enc):
        pooled = self.bert(**enc).pooler_output
        g = self.fc(self.dropout(pooled))               # (B, 1)
        return g + self.biases                          # (B, K-1)


def coral_loss(logits, labels):
    # levels[i,k] = 1 se y_i > k  (alvos cumulativos)
    levels = (labels.unsqueeze(1) > torch.arange(K - 1, device=labels.device)).float()
    # log(1-sigmoid(x)) = logsigmoid(x) - x  (forma estável)
    return -(F.logsigmoid(logits) * levels
             + (F.logsigmoid(logits) - logits) * (1 - levels)).sum(1).mean()


def to_class_probs(logits):
    """Cumulativos -> distribuicao por classe."""
    cum = torch.sigmoid(logits)
    p0 = 1 - cum[:, 0]
    p1 = cum[:, 0] - cum[:, 1]
    p2 = cum[:, 1]
    p = torch.stack([p0, p1, p2], dim=1)
    return (p.clamp_min(0) / p.clamp_min(0).sum(1, keepdim=True)).cpu().numpy()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    P, Y = [], []
    for enc, lab in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        P.append(to_class_probs(model(enc)))
        Y.append(lab.numpy())
    return np.vstack(P), np.concatenate(Y)


tr_dl = DataLoader(DS(tr["resp_text"], tr["label"]), batch_size=args.batch,
                   shuffle=True, collate_fn=collate)
va_dl = DataLoader(DS(va["resp_text"], va["label"]), batch_size=32, collate_fn=collate)
ho_dl = DataLoader(DS(hold_df["resp_text"], hold_df["label"]), batch_size=32, collate_fn=collate)

model = CoralBert(args.model).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total = len(tr_dl) * args.epochs
sched = get_linear_schedule_with_warmup(opt, 500, total)
print(f"passos totais: {total}")

best_f1, best_state = -1.0, None
for ep in range(1, args.epochs + 1):
    model.train()
    for i, (enc, lab) in enumerate(tr_dl, 1):
        enc = {k: v.to(device) for k, v in enc.items()}
        loss = coral_loss(model(enc), lab.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if i % 200 == 0:
            print(f"  ep{ep} passo {i}/{len(tr_dl)} loss={loss.item():.4f}", flush=True)
    Pv, Yv = evaluate(model, va_dl)
    f1 = f1_score(Yv, Pv.argmax(1), average="macro")
    # se os vieses colapsam, a faixa do meio fecha e P(c234) -> 0
    b = model.biases.detach().cpu().numpy()
    print(f"epoca {ep}: val f1_macro={f1:.4f} acc={accuracy_score(Yv, Pv.argmax(1)):.4f} "
          f"| vieses={b.round(3)} (gap={b[0]-b[1]:.3f}) "
          f"| preve c234 em {(Pv.argmax(1) == 1).mean():.1%}")
    if f1 > best_f1:
        best_f1 = f1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print("   (melhor ate agora, guardado)")

model.load_state_dict(best_state)
Ph, Yh = evaluate(model, ho_dl)
y = Ph.argmax(1)
acc, f1m = accuracy_score(Yh, y), f1_score(Yh, y, average="macro")

np.savez(f"probs_{args.tag}.npz", y=Yh, probs=Ph)
print(f"\n=== CORAL ({args.tag}) no holdout ===")
print(f"acuracia={acc:.4f}  f1_macro={f1m:.4f}  (val f1 do checkpoint={best_f1:.4f})")
print(classification_report(Yh, y, target_names=LABELS, digits=4))
print(pd.DataFrame(confusion_matrix(Yh, y), index=LABELS, columns=LABELS))
pd.DataFrame([{"tag": args.tag, "model": args.model, "lr": args.lr,
               "epochs": args.epochs, "acc": acc, "f1_macro": f1m}]
             ).to_csv(f"result_{args.tag}.csv", index=False)
