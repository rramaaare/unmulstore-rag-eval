"""Perbandingan v1 vs v2 (konfigurasi akhir) pada seluruh pertanyaan, untuk laporan.

Angka utama: baris dalam cakupan pada v1 dan v3 (perbandingan berpasangan).
Tabel lain: ringkasan seluruh 212 baris, per domain, distribusi skor, entri sumber terambil.
Keluaran (di folder ini): comparison_full.csv, comparison_domain.csv, comparison_juri.csv, comparison_dist.csv,
comparison_hit.csv, appendix_rows.csv, comparison_full.md
"""
import json
import re

import numpy as np
import pandas as pd

M = ["faithfulness", "llm_context_precision_with_reference", "context_recall",
     "semantic_similarity", "factual_correctness(mode=f1)", "answer_relevancy"]
NAME = {"faithfulness": "Faithfulness", "llm_context_precision_with_reference": "Context precision",
        "context_recall": "Context recall", "semantic_similarity": "Semantic similarity",
        "factual_correctness(mode=f1)": "Factual correctness", "answer_relevancy": "Answer relevancy"}

v1 = pd.read_csv("results_v1.csv").set_index("id")
F = pd.read_csv("results_v3_full.csv").set_index("id")
rng = np.random.default_rng(0)

# ---------- 1. v1 vs v3, baris dalam cakupan (tanpa kolom juri; kanonik v3 lengkap) ----------
qwen = F[F["type"] == "in_scope"]
ids = [i for i in qwen.index if i in v1.index]
rows = []
for c in M:
    a, b = v1.loc[ids, c], qwen.loc[ids, c]
    d = (b - a).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    rows.append({"metrik": NAME[c], "v1": a.mean(), "v2": b.mean(), "selisih": d.mean(), "CI95_bawah": lo, "CI95_atas": hi,
                 "naik": int((b - a > 0.05).sum()), "turun": int((b - a < -0.05).sum()), "sama": int(((b - a).abs() <= 0.05).sum()),
                 "n": len(d), "bermakna": "ya" if (lo > 0 or hi < 0) else "belum terbukti"})
T = pd.DataFrame(rows).round(3)
T.to_csv("comparison_full.csv", index=False, encoding="utf-8-sig")

# ---------- 1b. sensitivitas: tanpa baris pengembangan (prompt disetel pada baris-baris itu) ----------
import csv as _csv
dev_ids = {r["id"] for r in _csv.DictReader(open("testset_dev.csv", encoding="utf-8-sig"))}
ids_nd = [i for i in ids if i not in dev_ids]
rows_nd = []
for c in M:
    a, b = v1.loc[ids_nd, c], qwen.loc[ids_nd, c]
    d = (b - a).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    rows_nd.append({"metrik": NAME[c], "v1": a.mean(), "v2": b.mean(), "selisih": d.mean(), "CI95_bawah": lo, "CI95_atas": hi,
                    "naik": int((b - a > 0.05).sum()), "turun": int((b - a < -0.05).sum()), "sama": int(((b - a).abs() <= 0.05).sum()),
                    "n": len(d), "bermakna": "ya" if (lo > 0 or hi < 0) else "belum terbukti"})
pd.DataFrame(rows_nd).round(3).to_csv("comparison_nondev.csv", index=False, encoding="utf-8-sig")
print("\nTANPA BARIS PENGEMBANGAN (n=%d):" % len(ids_nd)); print(pd.DataFrame(rows_nd).round(3).to_string(index=False))

# ---------- 2. seluruh 212 baris (rata-rata keseluruhan; tanpa kolom juri) ----------
J = F[M].mean().to_frame("semua_212").rename(index=NAME).round(3)
J.to_csv("comparison_juri.csv", encoding="utf-8-sig")

# ---------- 3. per domain (v1 vs v2, Qwen, dalam cakupan) ----------
dom = []
for d_, g in qwen.loc[ids].groupby("domain"):
    gi = list(g.index)
    dom.append({"domain": d_, "n": len(gi),
                "faith_v1": v1.loc[gi, "faithfulness"].mean(), "faith_v2": g["faithfulness"].mean(),
                "prec_v1": v1.loc[gi, "llm_context_precision_with_reference"].mean(), "prec_v2": g["llm_context_precision_with_reference"].mean(),
                "recall_v1": v1.loc[gi, "context_recall"].mean(), "recall_v2": g["context_recall"].mean()})
D = pd.DataFrame(dom).round(3).sort_values("faith_v2")
D.to_csv("comparison_domain.csv", index=False, encoding="utf-8-sig")

# ---------- 4. distribusi skor (Qwen, dalam cakupan) ----------
dist = []
for c in ("faithfulness", "context_recall", "llm_context_precision_with_reference"):
    for ver, s in (("v1", v1.loc[ids, c]), ("v2", qwen.loc[ids, c])):
        dist.append({"metrik": NAME[c], "versi": ver, "persen_sempurna_1.0": round(100 * (s >= 0.999).mean(), 1),
                     "persen_di_bawah_0.5": round(100 * (s < 0.5).mean(), 1), "median": round(s.median(), 2)})
pd.DataFrame(dist).to_csv("comparison_dist.csv", index=False, encoding="utf-8-sig")


# ---------- 5. entri sumber terambil (tanpa juri), per domain, seluruh baris dalam cakupan ----------
def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["id"]] = r
    return d


R1, R2 = load("responses_v1.jsonl"), load("responses_v3_full.jsonl")


def hit(R, i):
    m = re.match(r"Q(\d+)$", i)
    pat = re.compile(r"(?:^|\n)(?:content: )?Q%s\. " % m.group(1))
    return any(pat.search(c) for c in R[i]["retrieved_contexts"])


common = [i for i in R2 if re.match(r"Q\d+$", i) and i in R1]
dom_of = F["domain"].to_dict()
hits = pd.DataFrame({"id": common, "domain": [dom_of.get(i, "?") for i in common],
                     "v1": [hit(R1, i) for i in common], "v2": [hit(R2, i) for i in common]})
H = hits.groupby("domain")[["v1", "v2"]].agg(["sum", "count"])
H.columns = ["v1_terambil", "n", "v2_terambil", "n2"]
H = H.drop(columns="n2").reset_index()
tot = pd.DataFrame([{"domain": "SEMUA", "v1_terambil": int(hits.v1.sum()), "n": len(hits), "v2_terambil": int(hits.v2.sum())}])
H = pd.concat([H, tot], ignore_index=True)
H.to_csv("comparison_hit.csv", index=False, encoding="utf-8-sig")

# ---------- 6. lampiran per baris ----------
A = F.reset_index()[["id", "domain", "type"] + M].copy()
A.to_csv("appendix_rows.csv", index=False, encoding="utf-8-sig")

# ---------- ringkasan md ----------
lines = ["| Metrik | v1 | v2 | Selisih | CI 95% | naik/turun/sama | Bermakna |", "|---|---|---|---|---|---|---|"]
for _, r in T.iterrows():
    lines.append(f"| {r['metrik']} | {r['v1']:.2f} | {r['v2']:.2f} | {r['selisih']:+.2f} | [{r['CI95_bawah']:+.2f}, {r['CI95_atas']:+.2f}] | {r['naik']}/{r['turun']}/{r['sama']} | {r['bermakna']} |")
lines.append(f"\nBaris berpasangan (dalam cakupan, dinilai Qwen di kedua sisi): {len(ids)}.")
open("comparison_full.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nkeseluruhan:\n", J.to_string())
print("\ndomain (diurutkan faith_v2):\n", D.to_string(index=False))
print("\ndistribusi:\n", pd.DataFrame(dist).to_string(index=False))
print("\nhit:\n", H.to_string(index=False))
print("\nlampiran:", len(A), "baris")
