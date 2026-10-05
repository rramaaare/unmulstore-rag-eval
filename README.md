# UnmulStore RAG Evaluation

Reproducibility package for: **Two Single-Variable Fixes Elevate RAG Faithfulness to 0.88: A Controlled RAGAS Evaluation of Chunking and Prompt Interventions** (submitted to *Trends in Sciences*).

A Retrieval-Augmented Generation (RAG) chatbot for UnmulStore (campus store and toga-rental service, Universitas Mulawarman), evaluated with the RAGAS framework (six metrics, Qwen 3.8 27B judge).

## Key result

| Metric | v1 (baseline) | v3 (final) | Gain |
|---|---|---|---|
| Faithfulness | 0.49 | 0.88 | +0.39 |
| Context precision | 0.76 | 0.85 | +0.09 |
| Context recall | 0.84 | 0.95 | +0.11 |
| Semantic similarity | 0.62 | 0.85 | +0.23 |
| Factual correctness | 0.27 | 0.67 | +0.40 |

207 paired in-scope questions, same judge. Judge-free check: source-entry retrieval 71% → 91%.

## Contents

- `evaluation/` — collection, scoring, and comparison scripts
  - `prompts/` — v1, v2, v2.1 system prompts + tool description
  - `results_v1.csv` — v1 scores (212 rows, no judge column)
  - `results_v3_full.csv` — v3 scores, canonical (212 rows, no judge column)
  - `comparison_*.csv/.md`, `summary_v1.csv` — frozen paper outputs
  - `testset*.csv`, `FAQ_per_entry.csv` — inputs
  - `responses_*.jsonl` — raw responses (needed to regenerate `comparison_hit.csv`)
- `n8n-workflow.json` — n8n workflow export (AI Agent + Supabase Vector Store)
- `evaluation/requirements.txt` — Python dependencies

## Reproduce

```bash
cd evaluation
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
python collect_responses.py --testset testset_full.csv
python run_ragas.py
python compare_final.py
```

See `evaluation/README.md` and `evaluation/prompts/README.md` for details.

## Models

- Chatbot: `gpt-oss-120b` via Cerebras API
- Judge: `qwen-3.8-27b` via Cerebras API (Temperature 0, max tokens 16000)
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Judge prompt: RAGAS v0.4.3 default (no custom prompt)

## Citation

See `CITATION.cff`. If you use this work, please cite the paper (DOI to be added after publication).
