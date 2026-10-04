"""
Experimento: JEV (TypeSafe System One) com few-shot + contexto enriquecido.

Complementa jev_zero_shot.py. A diferença é o request:
  - instructions como objeto, com contexto do e-SIC e da escala 1..5;
  - criteria de cada classe como objeto {what, examples}, onde `examples` são
    RESPOSTAS REAIS ROTULADAS do treino (os exemplos few-shot);
  - state como campo nomeado.

Os exemplos few-shot são sorteados do treino EXCLUINDO o conjunto de avaliação,
para não haver vazamento. A avaliação usa a MESMA amostra do zero-shot (mesma
semente e mesmo --n), de modo que os dois números são diretamente comparáveis.

Objetivo: separar "tarefa difícil" de "modelo não conhece a convenção de rótulo".
Se o número subir para perto dos ~0,45 supervisionados, a convenção é aprendível
e o teto é esse; se continuar baixo, aponta ruído irredutível no rótulo.

Pré-requisitos:
  pip install pandas openpyxl scikit-learn python-dotenv typesafe-sdk
  # .env na raiz (no .gitignore) com: TYPESAFE_API_KEY=sua-chave

Uso:
  python jev_few_shot.py --n 400 --k 3        # 3 exemplos por classe no prompt
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typesafe_sdk import Choice, TypeSafeAPIError, TypeSafeClient

DATA_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = DATA_DIR / "train.xlsx"
CACHE_PATH = Path(__file__).resolve().parent / "jev_fewshot_cache.csv"

load_dotenv(DATA_DIR / ".env")

TEXT_COL = "resp_text"
LABEL_COL = "clarity"
LABELS = ["c1", "c234", "c5"]
SEED = 42

# contexto da tarefa, passado como instructions estruturado
INSTRUCTIONS = {
    "pergunta": "Classifique o grau de clareza desta resposta de um órgão público.",
    "contexto": (
        "No e-SIC, o cidadão faz um pedido de acesso à informação e um órgão público responde. "
        "A clareza da resposta foi avaliada pelo próprio cidadão numa escala de 1 (nada clara) "
        "a 5 (muito clara). As notas intermediárias 2, 3 e 4 foram agrupadas em uma única classe (c234)."
    ),
    "perspectiva": "Avalie do ponto de vista do cidadão que recebeu a resposta, não do órgão.",
}

WHAT = {
    "c1": "Resposta pouco clara, incompleta ou confusa. Menor grau de clareza (nota 1 de 5).",
    "c234": "Clareza intermediária: nem muito clara nem muito confusa. Agrega as notas 2, 3 e 4 de 5.",
    "c5": "Resposta muito clara, direta e completa. Maior grau de clareza (nota 5 de 5).",
}


def load_train() -> pd.DataFrame:
    train = pd.read_excel(TRAIN_PATH)
    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    train[LABEL_COL] = train[LABEL_COL].astype(str).str.strip()
    train = train[train[LABEL_COL].isin(LABELS)].reset_index(drop=True)
    return train.reset_index().rename(columns={"index": "idx"})  # idx: id estável


def eval_sample(train: pd.DataFrame, n: int) -> pd.DataFrame:
    # idêntico ao sample() do zero-shot: estratificado, proporcional, semente fixa
    parts = []
    for _, g in train.groupby(LABEL_COL):
        k = max(1, round(n * len(g) / len(train)))
        parts.append(g.sample(min(len(g), k), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def pick_fewshot(train: pd.DataFrame, exclude_idx: set[int], k: int,
                 max_chars: int = 400) -> tuple[set[int], dict[str, list[str]]]:
    """k exemplos por classe, fora de exclude_idx, curtos e representativos.

    Devolve (índices escolhidos, {classe: [textos]}). Os índices permitem excluir
    os próprios exemplos few-shot da avaliação, evitando vazamento.
    """
    pool = train[~train["idx"].isin(exclude_idx)].copy()
    pool = pool[pool[TEXT_COL].str.len().between(100, 600)]  # evita outliers e textos triviais
    idx, texts = set(), {}
    for lbl in LABELS:
        g = pool[pool[LABEL_COL] == lbl].sample(k, random_state=SEED)
        idx |= set(g["idx"])
        texts[lbl] = [t[:max_chars] for t in g[TEXT_COL]]
    return idx, texts


def build_criteria(fewshot: dict[str, list[str]]) -> dict:
    return {lbl: {"what": WHAT[lbl], "examples": fewshot[lbl]} for lbl in LABELS}


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
    print(f"\n=== JEV few-shot ({len(done)} exemplos) ===")
    print(f"Acurácia (concordância com o rótulo humano): {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_true, y_pred, average='macro', labels=LABELS):.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4))
    print("Matriz de confusão (linhas = humano, colunas = JEV):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS), index=LABELS, columns=LABELS))
    print("\nConfiança média do JEV por classe real (menor = mais incerto):")
    print(done.groupby("gold")["conf"].mean().reindex(LABELS).round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description="JEV few-shot + contexto enriquecido")
    ap.add_argument("--n", type=int, default=400, help="tamanho da amostra de avaliação (igual ao zero-shot)")
    ap.add_argument("--k", type=int, default=3, help="exemplos few-shot por classe")
    ap.add_argument("--full", action="store_true", help="avalia TODO o treino (menos os exemplos few-shot)")
    ap.add_argument("--holdout", action="store_true", help="avalia no MESMO holdout dos modelos finais (main.py)")
    ap.add_argument("--workers", type=int, default=16, help="chamadas concorrentes (tier pago, sem rate limit)")
    ap.add_argument("--model", default=None, help="modelo (padrão: jev-latest do SDK)")
    args = ap.parse_args()

    train = load_train()
    if args.holdout:
        # MESMO holdout dos modelos finais (main.py): split estratificado 20%, seed 42.
        # few-shot sorteado só do dev (80%), espelhando o que os supervisionados viram.
        from sklearn.model_selection import train_test_split
        dev, hold = train_test_split(train, test_size=0.2, stratify=train[LABEL_COL], random_state=SEED)
        df = hold.reset_index(drop=True)
        fs_idx, fewshot = pick_fewshot(dev, set(), args.k)
        tag = f"holdout_k{args.k}"
    elif args.full:
        # few-shot primeiro; avalia em todo o resto do treino (sem vazamento)
        fs_idx, fewshot = pick_fewshot(train, set(), args.k)
        df = train[~train["idx"].isin(fs_idx)].reset_index(drop=True)
        tag = f"full_k{args.k}"
    else:
        df = eval_sample(train, args.n)
        fs_idx, fewshot = pick_fewshot(train, set(df["idx"]), args.k)
        tag = f"k{args.k}"
    criteria = build_criteria(fewshot)

    global CACHE_PATH
    CACHE_PATH = CACHE_PATH.with_name(f"jev_fewshot_cache_{tag}.csv")

    print(f"Avaliação: {len(df)} exemplos | few-shot: {args.k}/classe | modelo: {args.model or 'jev-latest'}")
    print(df[LABEL_COL].value_counts().reindex(LABELS).to_string())

    cache = load_cache()
    pending = [row for row in df.itertuples()
               if int(row.idx) not in cache or cache[int(row.idx)].get("pred") not in LABELS]
    print(f"A processar: {len(pending)} ({len(df) - len(pending)} já em cache) | workers: {args.workers}")

    question = {"clarity": Choice(instructions=INSTRUCTIONS, criteria=criteria)}
    client_kwargs = {"model": args.model} if args.model else {}
    lock = threading.Lock()
    done = 0

    def work(row):
        try:
            r = client.system_one(state={"resposta_do_orgao": getattr(row, TEXT_COL)}, questions=question)
            ans = r.choices["clarity"]
            return int(row.idx), getattr(row, LABEL_COL), ans.choice, ans.confidence
        except TypeSafeAPIError as e:
            return int(row.idx), getattr(row, LABEL_COL), None, getattr(e, "status", "?")

    with TypeSafeClient(**client_kwargs) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for idx, gold, pred, info in ex.map(work, pending):
                with lock:
                    done += 1
                    if pred in LABELS:
                        cache[idx] = {"idx": idx, "gold": gold, "pred": pred, "conf": info}
                    else:
                        print(f"idx={idx} ERRO {info}")
                    if done % 500 == 0:
                        save_cache(cache)
                        print(f"  ... {done}/{len(pending)}")
    save_cache(cache)
    report(cache, df)


if __name__ == "__main__":
    main()
