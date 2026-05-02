

Summary
-------
This repo is an VERY early (none of these files have been touched since December 2024) project that:
- extracts text from the included book (`worldviews.txt`) and attempts to structure it into JSON (`textbook.json`) using a Jsonformer + LLM approach ([`preprocessing.py`](preprocessing.py))
- builds a semantic index over extracted chunks and answers natural-language queries with a lightweight RAG-style pipeline ([`semanticsearch.py`](semanticsearch.py))
- ships local model artifacts and an instruction-style pipeline for generation in `dolly-v2-12b` ([`dolly-v2-12b/instruct_pipeline.py`](dolly-v2-12b/instruct_pipeline.py))

This was essentially a crude attempt at a precursor to NotebookLM, it includes document ingestion + structure extraction, semantic embeddings, and an answer-generation step into a single code-driven workflow that you could run and iterate on locally before NotebookLM was usable or widely known


How the pipeline is supposed to work
-----------------------------------
1. `preprocess_text` reads the plaintext book and splits pages by form-feed (`\f`). See [`preprocess_text`](preprocessing.py)
2. `generate_json` builds a big prompt from all pages and calls Jsonformer (backed by the local Dolly model/tokenizer) to produce a schema-constrained JSON structure for chapters. See [`generate_json`](preprocessing.py)
3. The semantic index (`semanticsearch.py`) loads preprocessed chunks (from `output.json`), embeds them with `SentenceTransformer("all-MiniLM-L6-v2")`, upserts to Pinecone, and answers queries by retrieving top chunks and conditioning a T5 generator (`t5-small`) on the retrieved context
4. The custom `InstructionTextGenerationPipeline` in `dolly-v2-12b` formats instruction/response prompts using special tokens (e.g., `### Instruction:`, `### Response:`, `### End`) and has logic to find those tokens in generated token sequences

ISSUES (there's a lot of them)
--------------------------------

1) Preprocessing produced trivial/incorrect JSON and many empty/placeholder records
   - Symptom: `textbook.json` contains a single minimal record and `output.json` / `processed_worldviews.json` contain many "Unknown Topic" or partial/garbled items
   - Causes:
     - The notebook/script dumped a raw PDF binary into the editor earlier — ensure `preprocessing.py` reads the plaintext file: [`worldviews.txt`](worldviews.txt), not the PDF binary [`worldviews.pdf`](worldviews.pdf)
     - `preprocessing.py` contains two different code blocks (one commented and one active) and builds a single enormous prompt. Large prompts will cause truncation or failed structured output
     - The chapter-detection regex in `generate_json` expects `Chapter (\d+): (.+)` but the text uses headings like `Chapter 1: Worldviews` and other formatting, so many pages don't match and produce `None`
   - Fixes:
     - Clean `preprocessing.py`: keep one coherent flow, chunk pages (e.g., 1–5 pages per prompt) rather than send the whole book at once, and use robust heading parsing. Re-run to regenerate `textbook.json`
     - Use a deterministic chunk id when upserting embeddings (see semanticsearch point 3)

2) Answer generation and retrieval filtering
   - The semantic search filters results with `if res.score > 0.8`. Many cosine similarities from `all-MiniLM-L6-v2` are < 0.8 even for good matches; the threshold is too strict
   - Fix:
     - Lower the threshold (e.g., 0.2–0.35 for cosine) or use `top_k` without hard thresholding, then sort by score
       
3) Local Dolly pipeline token handling
     - `get_special_token_id` uses `tokenizer.encode(key)` and raises if multiple tokens are returned. Some tokenizers map strings like `"### Response:\n"` to multiple tokens if whitespace/newline differs
     - In `postprocess`, `generated_sequence.numpy()` assumes a CPU tensor; for PyTorch tensors you should call `.cpu().numpy()` first
   - Fixes:
     - Prefer `tokenizer.encode(key, add_special_tokens=False)` or use `tokenizer.convert_tokens_to_ids` on the added special token, or inspect `tokenizer.additional_special_tokens` mapping directly
     - Replace `logger.warn` with `logger.warning`
     - Convert torch tensors: `generated_sequence = generated_sequence.cpu().numpy().tolist()`

4) Mixed / duplicate code blocks and commented-out sections
   - Consolidate into a single, well-documented flow; add CLI arguments (input path, chunk size, output path) and unit tests

