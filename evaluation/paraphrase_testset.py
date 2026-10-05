"""Parafrase kolom `question` di testset agar tidak identik dengan judul/Varian di knowledge base.

Memakai model juri yang sama (JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY di .env).
Kolom asli disimpan di `question_original`. Baris `out_of_scope` tidak diubah.
Hasil WAJIB disaring manual (mis. 20 baris acak) untuk memastikan makna tidak bergeser.
"""
import argparse
import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

PROMPT = """Tulis ulang pertanyaan berikut sebagai satu pertanyaan baru dalam bahasa Indonesia santai, seperti ditulis mahasiswa di chat.

Aturan:
- Makna dan hal yang ditanyakan HARUS sama persis; jangan menambah atau mengurangi informasi.
- Ganti susunan kalimat dan pilihan kata; hindari menyalin frasa khas dari pertanyaan asli.
- Boleh memakai bahasa sehari-hari, singkatan wajar (mis. "gimana", "nggak"), tetapi tetap jelas.
- Nama aplikasi "UnmulStore" boleh tetap ditulis bila perlu.
- Jawab HANYA dengan satu pertanyaan baru, tanpa tanda kutip, tanpa penjelasan.

Pertanyaan asli: {q}"""


def rewrite(client, model, q, max_tokens):
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": PROMPT.format(q=q)}],
                max_tokens=max_tokens, temperature=0.7)
            out = (r.choices[0].message.content or "").strip().strip('"').splitlines()
            out = next((l.strip() for l in out if l.strip()), "")
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            print("  gagal:", str(e)[:120])
    return q  # gagal: pakai pertanyaan asli


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="testset_full.csv")
    ap.add_argument("--out", default="testset_full_para.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    client = OpenAI(base_url=os.environ["JUDGE_BASE_URL"], api_key=os.environ["JUDGE_API_KEY"])
    model = os.environ["JUDGE_MODEL"]
    max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))

    with open(a.inp, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[: a.limit]

    def work(r):
        r = dict(r)
        r["question_original"] = r["question"]
        if r["type"] == "in_scope":
            r["question"] = rewrite(client, model, r["question"], max_tokens)
        return r

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        out = list(ex.map(work, rows))

    fields = list(out[0])
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    same = sum(1 for r in out if r["type"] == "in_scope" and r["question"] == r["question_original"])
    print(f"{len(out)} baris ditulis ke {a.out}; {same} baris gagal diparafrase (pakai asli)")
    print("\nContoh untuk disaring manual:")
    for r in random.Random(1).sample([r for r in out if r["type"] == "in_scope"], min(8, len(out))):
        print(f"- ASLI : {r['question_original']}\n  BARU : {r['question']}")


if __name__ == "__main__":
    main()
