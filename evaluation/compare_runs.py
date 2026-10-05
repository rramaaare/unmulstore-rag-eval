"""Bandingkan tiga konfigurasi pada baris yang sama dan simpan tabel untuk laporan.

v1 = prompt lama, potongan lama | A = prompt v2.1 + Temperature 0, potongan lama | B = A + satu entri per potongan.
Keluaran: comparison_dev.csv, comparison_dev.md (tabel siap tempel), dan ringkasan di layar.
Yang dihitung selain skor RAGAS: selisih B-A dengan interval kepercayaan 95% (bootstrap per baris),
jumlah baris naik/turun/sama (ambang 0,05), dan ukuran retrieval tanpa juri (entri sumber ikut terambil).
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

METRICS = ["faithfulness", "llm_context_precision_with_reference", "context_recall",
           "semantic_similarity", "factual_correctness(mode=f1)", "answer_relevancy"]
NAMES = {"faithfulness": "Faithfulness", "llm_context_precision_with_reference": "Context precision",
         "context_recall": "Context recall", "semantic_similarity": "Semantic similarity",
         "factual_correctness(mode=f1)": "Factual correctness", "answer_relevancy": "Answer relevancy"}


def load_jsonl(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            d[r["id"]] = r
    return d


def source_hit(resp, rid):
    m = re.match(r"Q(\d+)$", rid)
    if not m:
        return None
    pat = re.compile(r"(?:^|\n)(?:content: )?Q%s\. " % m.group(1))
    return any(pat.search(c) for c in resp[rid]["retrieved_contexts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="results_v1.csv")
    ap.add_argument("--a", default="results_v2_dev_t0.csv")
    ap.add_argument("--b", default="results_v3_dev.csv")
    ap.add_argument("--v1-resp", default="responses_v1.jsonl")
    ap.add_argument("--a-resp", default="responses_v2_dev_t0.jsonl")
    ap.add_argument("--b-resp", default="responses_v3_dev.jsonl")
    ap.add_argument("--out", default="comparison_dev")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    v1, A, B = (pd.read_csv(p).set_index("id") for p in (a.v1, a.a, a.b))
    ids = [i for i in B.index if i in A.index and i in v1.index]
    rng = np.random.default_rng(a.seed)

    rows = []
    for c in METRICS:
        d = (B.loc[ids, c] - A.loc[ids, c]).dropna().values
        boot = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        diff = B.loc[ids, c] - A.loc[ids, c]
        rows.append({
            "metrik": NAMES[c], "v1": v1.loc[ids, c].mean(), "A": A.loc[ids, c].mean(), "B": B.loc[ids, c].mean(),
            "B-A": d.mean(), "CI95_bawah": np.percentile(boot, 2.5), "CI95_atas": np.percentile(boot, 97.5),
            "naik": int((diff > 0.05).sum()), "turun": int((diff < -0.05).sum()), "sama": int((diff.abs() <= 0.05).sum()),
            "n": len(d),
            "bermakna": "ya" if (np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0) else "belum terbukti",
        })
    T = pd.DataFrame(rows).round(3)
    T.to_csv(a.out + ".csv", index=False, encoding="utf-8-sig")

    # ukuran retrieval tanpa juri
    R = {k: load_jsonl(p) for k, p in (("v1", a.v1_resp), ("A", a.a_resp), ("B", a.b_resp))}
    q = [i for i in ids if re.match(r"Q\d+$", i)]
    hit = {k: sum(bool(source_hit(R[k], i)) for i in q) for k in R}
    norag = {k: sum(1 for i in ids if not R[k][i]["retrieved_contexts"]) for k in R}

    lines = ["| Metrik | v1 | A | B | B - A | 95% CI | naik/turun/sama | bermakna |", "|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        lines.append(f"| {r['metrik']} | {r['v1']:.2f} | {r['A']:.2f} | {r['B']:.2f} | {r['B-A']:+.2f} | "
                     f"[{r['CI95_bawah']:+.2f}, {r['CI95_atas']:+.2f}] | {r['naik']}/{r['turun']}/{r['sama']} | {r['bermakna']} |")
    lines += ["", f"Baris dibandingkan: {len(ids)}. Entri sumber terambil (dari {len(q)} baris): "
                  f"v1 {hit['v1']}, A {hit['A']}, B {hit['B']}. Baris tanpa retrieval: v1 {norag['v1']}, A {norag['A']}, B {norag['B']}."]
    md = "\n".join(lines)
    open(a.out + ".md", "w", encoding="utf-8").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
