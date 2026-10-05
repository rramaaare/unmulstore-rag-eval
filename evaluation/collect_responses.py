"""Kirim tiap pertanyaan di testset ke endpoint website (/api/chat), simpan jawaban + konteks ke responses.jsonl.

Syarat:
- Website jalan (`npm run dev` di web/) dengan env EVAL_MODE=true.
- Node AI Agent di n8n mengaktifkan "Return Intermediate Steps".
Respons yang diharapkan: {"reply": "...", "contexts": ["chunk1", ...]}

Bisa dilanjutkan (resume): baris yang sudah punya jawaban di responses.jsonl dilewati, jadi aman
dihentikan (Ctrl+C) dan dijalankan lagi, mis. saat kuota harian model chatbot habis.
"""
import argparse
import csv
import json
import os
import time

import requests
from dotenv import load_dotenv


def expand_context(ctx):
    """Ubah satu 'konteks' mentah dari n8n menjadi daftar teks chunk (hanya pageContent).

    Menangani tiga bentuk: string biasa, string JSON {pageContent, metadata}, dan
    bungkus Agent baru {"response": [{"type": "text", "text": "<json chunk>"}, ...]}.
    """
    if not isinstance(ctx, str):
        ctx = json.dumps(ctx, ensure_ascii=False)
    s = ctx.lstrip()
    if s.startswith(("{", "[")):
        try:
            return list(_walk(json.loads(s))) or [ctx]
        except json.JSONDecodeError:
            return [ctx]
    return [ctx]


def _walk(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from _walk(item)
    elif isinstance(obj, dict):
        if obj.get("pageContent"):
            yield str(obj["pageContent"])
        elif "response" in obj:
            yield from _walk(obj["response"])
        elif "text" in obj:
            yield from expand_context(obj["text"])
        else:
            yield json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, str):
        yield from expand_context(obj)


def ask(url, question, retries=4):
    """Kembalikan (jawaban, konteks). Jawaban kosong = gagal setelah semua percobaan."""
    ask.last_queries = []
    for attempt in range(retries):
        try:
            r = requests.post(url, json={"message": question, "includeContexts": True}, timeout=180)
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            contexts = [c for raw in data.get("contexts") or [] for c in expand_context(raw)]
            ask.last_queries = data.get("searchQueries") or []
            return data.get("reply", ""), contexts
        except Exception as e:  # noqa: BLE001
            print(f"  gagal ({attempt + 1}/{retries}): {e}")
            time.sleep(20 * (attempt + 1))  # tunggu cukup lama agar batas per menit pulih
    return "", []


def load_done(path):
    """id yang sudah punya jawaban tidak kosong (catatan terakhir per id yang dipakai)."""
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = bool(r.get("response"))
    return {i for i, ok in done.items() if ok}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="testset.csv")
    ap.add_argument("--out", default="responses.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = semua")
    ap.add_argument("--start", type=int, default=0, help="mulai dari baris ke-N (0-based)")
    ap.add_argument("--end", type=int, default=0, help="berhenti sebelum baris ke-N; 0 = sampai akhir")
    ap.add_argument("--fresh", action="store_true", help="hapus responses lama dan mulai dari awal")
    ap.add_argument("--delay", type=float, default=float(os.environ.get("EVAL_DELAY", "1")),
                    help="jeda antar pertanyaan (detik); default dari env EVAL_DELAY")
    a = ap.parse_args()

    url = os.environ.get("EVAL_ENDPOINT", "http://localhost:3000/api/chat")
    with open(a.testset, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = rows[a.start: a.end or None]
    if a.limit:
        rows = rows[: a.limit]

    if a.fresh and os.path.exists(a.out):
        os.remove(a.out)
    done = load_done(a.out)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} baris dipilih, {len(rows) - len(todo)} sudah selesai, {len(todo)} akan diproses")

    with open(a.out, "a", encoding="utf-8") as out:
        for i, r in enumerate(todo, 1):
            answer, contexts = ask(url, r["question"])
            print(f"[{i}/{len(todo)}] {r['id']} contexts={len(contexts)}" + ("" if answer else "  (GAGAL, akan diulang saat dijalankan lagi)"))
            out.write(json.dumps({
                "id": r["id"], "type": r["type"], "domain": r["domain"],
                "user_input": r["question"], "retrieved_contexts": contexts,
                "response": answer, "reference": r["ground_truth"],
                "search_queries": getattr(ask, "last_queries", []),
            }, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(a.delay)


if __name__ == "__main__":
    main()
