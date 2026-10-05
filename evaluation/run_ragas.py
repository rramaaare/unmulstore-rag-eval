"""Nilai responses.jsonl dengan RAGAS. Juri = model OpenAI-compatible (Cerebras/OpenRouter/Gemini) atau claude -p.

Berjalan per batch dan bisa dilanjutkan: results.csv ditulis setelah tiap batch, baris yang sudah
terisi lengkap dilewati, baris yang masih NaN dinilai ulang.
"""
import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
    SemanticSimilarity,
)
from ragas.run_config import RunConfig

AVAILABLE = {
    "faithfulness": Faithfulness, "answer_relevancy": ResponseRelevancy,
    "context_precision": LLMContextPrecisionWithReference, "context_recall": LLMContextRecall,
    "factual_correctness": FactualCorrectness, "semantic_similarity": SemanticSimilarity,
}
META_COLS = ["id", "type", "domain", "user_input", "response"]


def load_records(path):
    """Catatan terakhir per id, hanya yang punya jawaban."""
    last = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                last[r["id"]] = r
    return [r for r in last.values() if r.get("response")]


def build_judge():
    if os.environ.get("JUDGE_PROVIDER") == "claude-cli":
        from judge_claude_cli import ClaudeCLI
        return ClaudeCLI(model=os.environ.get("JUDGE_CLI_MODEL", "claude-sonnet-5-5"),
                         concurrency=int(os.environ.get("RAGAS_WORKERS", "2")))
    rpm = float(os.environ.get("RAGAS_RPM", "0"))
    limiter = InMemoryRateLimiter(requests_per_second=rpm / 60, max_bucket_size=1) if rpm else None
    return ChatOpenAI(
        rate_limiter=limiter,
        model=os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-5.5"),
        # default OpenRouter; Cerebras: JUDGE_BASE_URL=https://api.cerebras.ai/v1
        base_url=os.environ.get("JUDGE_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "1024")),
    )


def summarize(results_path, responses_path, summary_path):
    df = pd.read_csv(results_path)
    metric_cols = [c for c in df.columns if c not in META_COLS]
    recs = {r["id"]: r for r in load_records(responses_path)}
    df["contexts_kosong"] = [0 if recs.get(i, {}).get("retrieved_contexts") else 1 for i in df["id"]]

    def agg(g, label, name):
        row = {"group": label, "name": name, "n": len(g)}
        row.update({c: round(g[c].mean(), 3) for c in metric_cols})
        row["persen_contexts_kosong"] = round(100 * g["contexts_kosong"].mean(), 1)
        row["baris_dengan_nan"] = int(g[metric_cols].isna().any(axis=1).sum())
        return row

    rows = [agg(df, "semua", "semua")]
    rows += [agg(g, "type", k) for k, g in df.groupby("type")]
    rows += [agg(g, "domain", k) for k, g in df.groupby("domain")]
    out = pd.DataFrame(rows)
    out.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n=== Ringkasan ===")
    print(out.to_string(index=False))

    nan_per_metric = df[metric_cols].isna().sum()
    print("\n=== Jumlah NaN per metrik (dari", len(df), "baris) ===")
    print(nan_per_metric.to_string())
    worst = 100 * nan_per_metric.max() / len(df)
    if worst > 5:
        print(f"\n!!! PERINGATAN: sampai {worst:.0f}% baris kosong pada satu metrik. Rata-rata di atas dihitung dari "
              "baris yang berhasil saja dan BELUM valid. Jalankan perintah yang sama lagi (baris NaN dinilai ulang) "
              "atau pakai --debug --ids <id> untuk melihat penyebabnya.")
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="responses.jsonl")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--summary", default="summary.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = semua; pakai 5 untuk uji awal")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--in-scope-only", action="store_true")
    ap.add_argument("--ids", default="", help="nilai hanya id ini, pisah koma, mis. Q124,Q125")
    ap.add_argument("--redo", action="store_true", help="nilai ulang semua, abaikan results.csv lama")
    ap.add_argument("--debug", action="store_true", help="tampilkan error asli (raise_exceptions=True)")
    ap.add_argument("--metrics", default="", help="pilih sebagian, pisah koma, mis. faithfulness,context_recall")
    a = ap.parse_args()

    names = [m.strip() for m in a.metrics.split(",") if m.strip()] or list(AVAILABLE)
    rows = load_records(a.responses)
    if a.in_scope_only:
        rows = [r for r in rows if r["type"] == "in_scope"]
    if a.ids:
        wanted = {i.strip() for i in a.ids.split(",") if i.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    if a.limit:
        rows = rows[: a.limit]

    # baris yang sudah lengkap di results.csv dilewati; yang NaN dinilai ulang
    prev = pd.DataFrame()
    if os.path.exists(a.out) and not a.redo:
        prev = pd.read_csv(a.out)
        metric_cols = [c for c in prev.columns if c not in META_COLS]
        complete = set(prev.loc[~prev[metric_cols].isna().any(axis=1), "id"]) if metric_cols else set()
        rows_todo = [r for r in rows if r["id"] not in complete]
        print(f"{len(rows)} baris, {len(rows) - len(rows_todo)} sudah lengkap, {len(rows_todo)} dinilai")
    else:
        rows_todo = rows
    if not rows_todo:
        print("Tidak ada yang perlu dinilai.")
        summarize(a.out, a.responses, a.summary)
        return

    # Cerebras/Groq tidak mendukung parameter n>1; bypass_n membuat RAGAS memanggil berkali-kali
    llm = LangchainLLMWrapper(build_judge(), bypass_n=os.environ.get("JUDGE_BYPASS_N", "1") == "1")
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")))
    run_config = RunConfig(timeout=int(os.environ.get("RAGAS_TIMEOUT", "600")),
                           max_workers=int(os.environ.get("RAGAS_WORKERS", "4")),
                           max_retries=int(os.environ.get("RAGAS_RETRIES", "5")))

    results = prev
    for start in range(0, len(rows_todo), a.batch_size):
        batch = rows_todo[start: start + a.batch_size]
        print(f"\n--- Batch {start // a.batch_size + 1}: baris {start + 1}-{start + len(batch)} dari {len(rows_todo)} ---")
        ds = EvaluationDataset.from_list([
            {k: r[k] for k in ("user_input", "retrieved_contexts", "response", "reference")} for r in batch])
        res = evaluate(dataset=ds, metrics=[AVAILABLE[n]() for n in names], llm=llm, embeddings=emb,
                       raise_exceptions=a.debug, run_config=run_config, show_progress=True)
        scored = res.to_pandas()
        meta = pd.DataFrame(batch)[["id", "type", "domain"]]
        part = pd.concat([meta, scored.drop(columns=["retrieved_contexts", "reference"], errors="ignore")], axis=1)
        part = part[[c for c in part.columns if c in META_COLS or c not in ("user_input", "response")]]
        # user_input & response ikut disimpan agar hasil mudah diperiksa
        part["user_input"] = [r["user_input"] for r in batch]
        part["response"] = [r["response"] for r in batch]
        if len(results):
            results = results[~results["id"].isin(part["id"])]
        results = pd.concat([results, part], ignore_index=True)
        results.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"Tersimpan: {a.out} ({len(results)} baris)")

    summarize(a.out, a.responses, a.summary)


if __name__ == "__main__":
    main()
