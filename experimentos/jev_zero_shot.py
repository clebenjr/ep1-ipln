"""
Experimento: estimativa do teto da tarefa via JEV (TypeSafe System One), zero-shot.

Ideia: se um modelo externo forte, SEM treino na nossa base, só concordar com o
rótulo humano em ~0,45-0,50, isso é evidência de que o teto do problema está no
rótulo (subjetivo/ruidoso), não na capacidade dos nossos classificadores.

O JEV é um classificador: entrada não estruturada -> opção tipada de um conjunto
fixo, com distribuição de probabilidades e confiança calibrada. Usamos a primitiva
`Choice` com as 3 classes {c1, c234, c5}, via SDK oficial typesafe-sdk.

Saídas:
  - acurácia e F1-macro contra o rótulo humano (o "teto" estimado);
  - matriz de confusão;
  - confiança média por classe real (esperamos que c234 seja a mais incerta).

Pré-requisitos:
  pip install pandas openpyxl scikit-learn python-dotenv typesafe-sdk
  # crie um arquivo .env na raiz do projeto (já no .gitignore) com:
  #   TYPESAFE_API_KEY=sua-chave        # chave do console.typesafe.ai

Uso:
  python jev_zero_shot.py --n 100                 # amostra estratificada de 100
  python jev_zero_shot.py --n 300 --model jev     # modelo específico (padrão: jev-latest)

O script é resumível: cada resposta é gravada em jev_cache.csv e reaproveitada
numa nova execução. Interromper e retomar é seguro.
"""

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typesafe_sdk import Choice, TypeSafeAPIError, TypeSafeClient

DATA_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = DATA_DIR / "train.xlsx"
CACHE_PATH = Path(__file__).resolve().parent / "jev_cache.csv"

load_dotenv(DATA_DIR / ".env")

TEXT_COL = "resp_text"
LABEL_COL = "clarity"
LABELS = ["c1", "c234", "c5"]
SEED = 42

INSTRUCTIONS = (
    "Classifique a clareza da resposta de um órgão público a um pedido de acesso à "
    "informação (e-SIC), do ponto de vista do cidadão que a recebeu."
)

# A clareza foi avaliada por usuários numa escala 1..5; c234 agrega os escores
# intermediários (2, 3 e 4) em uma única classe.
CRITERIA = {
    "c1": "Resposta pouco clara ou confusa. Menor grau de clareza (nota 1 de 5).",
    "c234": "Clareza intermediária: nem muito clara nem muito confusa. Agrega as notas 2, 3 e 4 de 5.",
    "c5": "Resposta muito clara e completa. Maior grau de clareza (nota 5 de 5).",
}


def sample(n: int) -> pd.DataFrame:
    train = pd.read_excel(TRAIN_PATH)
    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    train[LABEL_COL] = train[LABEL_COL].astype(str).str.strip()
    train = train[train[LABEL_COL].isin(LABELS)].reset_index(drop=True)
    train = train.reset_index().rename(columns={"index": "idx"})  # idx: id estável da linha

    # amostra estratificada, proporcional por classe, com semente fixa
    parts = []
    for _, g in train.groupby(LABEL_COL):
        k = max(1, round(n * len(g) / len(train)))
        parts.append(g.sample(min(len(g), k), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def load_cache() -> dict[int, dict]:
    if not CACHE_PATH.exists():
        return {}
    df = pd.read_csv(CACHE_PATH)
    return {int(r["idx"]): r.to_dict() for _, r in df.iterrows()}


def save_cache(cache: dict[int, dict]) -> None:
    pd.DataFrame(list(cache.values())).sort_values("idx").to_csv(CACHE_PATH, index=False)


def report(cache: dict[int, dict], df: pd.DataFrame) -> None:
    done = pd.DataFrame([c for c in cache.values() if c.get("pred") in LABELS])
    done = done[done["idx"].isin(df["idx"])] if not done.empty else done
    if done.empty:
        print("Nenhuma predição válida ainda.")
        return

    y_true, y_pred = done["gold"], done["pred"]
    print(f"\n=== JEV zero-shot ({len(done)} exemplos) ===")
    print(f"Acurácia (concordância com o rótulo humano): {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_true, y_pred, average='macro', labels=LABELS):.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print("Matriz de confusão (linhas = humano, colunas = JEV):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS), index=LABELS, columns=LABELS))
    print("\nConfiança média do JEV por classe real (menor = mais incerto):")
    print(done.groupby("gold")["conf"].mean().reindex(LABELS).round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description="JEV zero-shot: estimativa do teto da tarefa")
    ap.add_argument("--n", type=int, default=100, help="tamanho da amostra estratificada")
    ap.add_argument("--model", default=None, help="modelo (padrão: jev-latest do SDK)")
    args = ap.parse_args()

    df = sample(args.n)
    print(f"Amostra: {len(df)} exemplos | modelo: {args.model or 'jev-latest'}")
    print(df[LABEL_COL].value_counts().reindex(LABELS).to_string())

    cache = load_cache()
    client_kwargs = {"model": args.model} if args.model else {}
    with TypeSafeClient(**client_kwargs) as client:
        for i, row in enumerate(df.itertuples(), 1):
            idx = int(row.idx)
            if idx in cache and cache[idx].get("pred") in LABELS:
                continue
            try:
                r = client.system_one(
                    state=getattr(row, TEXT_COL),
                    questions={"clarity": Choice(instructions=INSTRUCTIONS, criteria=CRITERIA)},
                )
                ans = r.choices["clarity"]
            except TypeSafeAPIError as e:
                print(f"[{i}/{len(df)}] idx={idx} ERRO {getattr(e, 'status', '?')}: {e}")
                continue
            cache[idx] = {
                "idx": idx,
                "gold": getattr(row, LABEL_COL),
                "pred": ans.choice,
                "conf": ans.confidence,
            }
            print(f"[{i}/{len(df)}] idx={idx} gold={cache[idx]['gold']} "
                  f"pred={ans.choice} conf={ans.confidence}")
            save_cache(cache)

    report(cache, df)


if __name__ == "__main__":
    main()
