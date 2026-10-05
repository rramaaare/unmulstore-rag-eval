# Eksperimen system prompt Agent

| Versi | File | Catatan |
|---|---|---|
| v1 (baseline) | `system_prompt_v1.txt` | prompt awal di workflow `Demo_ RAG in n8n.json`; hasil: `results_v1.csv`, `responses_v1.jsonl` |
| v2.1 | `system_prompt_v2_1.txt` | v2 + kueri pencarian pendek (2-5 kata inti, tanpa "UnmulStore", kata tanya, atau bahasa Inggris); alasan: uji "kapan restok" 6x, Q261 terambil 5/6, gagal saat kueri ditambah "UnmulStore" dan "restocking" |
| v2 | `system_prompt_v2.txt` + `tool_description_v2.txt` | wajib mencari dulu, jawab hanya dari hasil, ringkas, tanpa emoji/markdown |

Yang diubah di n8n untuk v2:
1. Node **AI Agent** > Options > **System Message** = isi `system_prompt_v2.txt`.
2. Node **Supabase Vector Store** (yang terhubung ke Agent) > **Description** = isi `tool_description_v2.txt`.
3. Node model chat: **Temperature** 0 (atau serendah yang tersedia) agar hasil lebih konsisten.
Simpan workflow, lalu kumpulkan jawaban ke file baru (`responses_v2.jsonl`), jangan menimpa v1.

Faktor lain sengaja TIDAK diubah (model chatbot, knowledge base, chunking, juri) agar perbedaan hasil bisa dikaitkan ke prompt.
