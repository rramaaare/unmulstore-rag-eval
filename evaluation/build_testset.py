"""Bangun testset.csv dari docx FAQ UnmulStore.

Tiap entri docx berisi 4 paragraf:
  Q<n>. pertanyaan ~ alternatif
  Varian: v1 | v2 | ...
  Jawaban: teks
  Domain: a, b  |  Tags: ...

Pertanyaan uji diambil dari salah satu Varian (parafrase) agar tidak identik dengan judul.
Hasil WAJIB ditinjau manual sebelum dipakai.
"""
import argparse
import csv
import random
import re
import zipfile
from collections import defaultdict
from html import unescape

OUT_OF_SCOPE = [
    "Apakah UnmulStore menjual laptop?",
    "Bagaimana cara bayar UKT lewat UnmulStore?",
    "Apakah ada layanan pemesanan tiket pesawat sekarang?",
    "Siapa rektor Universitas Mulawarman saat ini?",
    "Berapa harga tiket bioskop di Samarinda?",
]
OOS_REFERENCE = "Informasi tersebut tidak tersedia dalam knowledge base UnmulStore."


def read_paragraphs(path):
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    paras = re.findall(r"<w:p>.*?</w:p>", xml, flags=re.S)
    return [unescape(re.sub(r"<[^>]+>", "", p)) for p in paras]


def parse(path):
    paras = read_paragraphs(path)
    entries, i = [], 0
    while i < len(paras):
        m = re.match(r"Q(\d+)\. (.*)", paras[i])
        if m and i + 3 < len(paras) and paras[i + 2].startswith("Jawaban:"):
            title = m.group(2)
            varian = [v.strip() for v in paras[i + 1].removeprefix("Varian:").split("|") if v.strip()]
            answer = paras[i + 2].removeprefix("Jawaban:").strip()
            dom = re.match(r"Domain:\s*(.*?)\s*\|", paras[i + 3])
            domain = dom.group(1).split(",")[0].strip() if dom else "umum"
            entries.append(dict(id=int(m.group(1)), title=title, varian=varian, answer=answer, domain=domain))
            i += 4
        else:
            i += 1
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", default="../FAQ_UnmulStore_KnowledgeBase_updated.docx")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="testset.csv")
    a = ap.parse_args()

    entries = parse(a.docx)
    rnd = random.Random(a.seed)
    by_domain = defaultdict(list)
    for e in entries:
        by_domain[e["domain"]].append(e)
    # ambil bergilir dari tiap domain agar sebaran merata
    for v in by_domain.values():
        rnd.shuffle(v)
    picked, doms = [], sorted(by_domain)
    while len(picked) < min(a.n, len(entries)):
        for d in doms:
            if by_domain[d] and len(picked) < a.n:
                picked.append(by_domain[d].pop())

    rows = []
    for e in sorted(picked, key=lambda e: e["id"]):
        q = rnd.choice(e["varian"]) if e["varian"] else e["title"].split("~")[0].strip()
        rows.append(dict(id=f"Q{e['id']}", type="in_scope", domain=e["domain"], question=q,
                         ground_truth=e["answer"], original_title=e["title"]))
    for k, q in enumerate(OUT_OF_SCOPE, 1):
        rows.append(dict(id=f"OOS{k}", type="out_of_scope", domain="-", question=q,
                         ground_truth=OOS_REFERENCE, original_title=""))

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(entries)} entri diparse, {len(rows)} baris ditulis ke {a.out}")


if __name__ == "__main__":
    main()
