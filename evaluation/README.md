# Evaluasi RAG UnmulStore dengan RAGAS

Mengikuti pendekatan VectorHub (RAG eval + RAGAS), disesuaikan dengan API RAGAS terbaru
(`EvaluationDataset`, kolom `user_input / retrieved_contexts / response / reference`).

## Persiapan sekali saja
1. **n8n:** buka node **AI Agent** > Options > **Add Option** > **Return Intermediate Steps** > aktif. Simpan workflow.
   Dengan ini respons webhook memuat hasil tool Supabase Vector Store (chunk yang diambil).
2. **Website:** di `web/.env.local` tambahkan `EVAL_MODE=true`, lalu jalankan `npm run dev` (port 3000).
   Tanpa `EVAL_MODE`, konteks tidak pernah dikirim ke browser.
3. **Evaluasi:** salin `.env.example` menjadi `.env`, isi `OPENROUTER_API_KEY`.
4. Aktifkan venv (PowerShell): `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\.venv\Scripts\Activate.ps1`

## Alur
1. `python build_testset.py` -> `testset.csv` (tinjau dan saring manual; ubah pertanyaan menjadi parafrase).
2. `python collect_responses.py --limit 5` (uji), lalu tanpa `--limit` -> `responses.jsonl`. Cek `contexts=` tidak 0.
3. `python run_ragas.py --limit 5` (uji), lalu penuh -> `results.csv`.

## Metrik
| Metrik | Mengukur |
|---|---|
| Faithfulness | jawaban konsisten dengan konteks (tidak mengarang) |
| ResponseRelevancy | jawaban relevan dengan pertanyaan |
| LLMContextPrecisionWithReference | chunk relevan berada di peringkat atas |
| LLMContextRecall | konteks memuat informasi dari ground truth |
| FactualCorrectness | kebenaran fakta vs ground truth |
| SemanticSimilarity | kemiripan makna jawaban vs ground truth |

Catatan: `context_relevancy` di artikel sudah tidak ada di RAGAS baru.

## Konfigurasi pipeline yang diuji (dari `Demo_ RAG in n8n.json`)
- Ingest: PDF loader (Default Data Loader) -> Supabase Vector Store (tabel `documents`)
- Embedding: Cohere `embed-multilingual-v3.0`
- Retrieval: Supabase Vector Store sebagai tool Agent (`retrieve-as-tool`), top-k default node
- LLM chatbot: `anthropic/claude-sonnet-5.5` via OpenRouter, maxTokens 4096
- Catatan: Agent memutuskan sendiri kapan memanggil tool. Jika tool tidak dipanggil, `contexts` kosong
  dan skor Faithfulness / Context* tidak bermakna untuk baris itu.
- Juri RAGAS memakai model yang sama dengan chatbot, sehingga penilaian bisa condong menguntungkan.

## Uji penuh (semua entri FAQ)
1. `python build_testset.py --n 999 --out testset_full.csv`
2. `python paraphrase_testset.py` -> `testset_full_para.csv` (saring beberapa baris secara manual; kolom `question_original` menyimpan aslinya)
3. `python collect_responses.py --testset testset_full_para.csv --out responses_v1.jsonl`
   - Bisa dihentikan (Ctrl+C) dan dijalankan lagi; baris yang sudah selesai dilewati.
   - `--start/--end` untuk mencicil, `--fresh` untuk mengulang dari awal.
4. `python run_ragas.py --responses responses_v1.jsonl --out results_v1.csv --summary summary_v1.csv`
   - Per batch (`--batch-size 20`), hasil tersimpan tiap batch; baris NaN dinilai ulang saat dijalankan lagi.
