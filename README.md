# RAG_Paper_Assistant
Retrieval-Augmented QA system for research papers with grounded answers and source citations.

## Run UI

```bash
cd "/Users/moulya.r.b/Desktop/Final Project"
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

## Evaluation (Professor feedback alignment)

This repo now includes a lightweight evaluation runner for:
- Retrieval quality: `Precision@k`, `Recall@k`
- End-to-end QA proxies: citation count, citation hit rate, refusal correctness

### 1) Prepare a gold set

Edit `evaluation/gold_qa_template.jsonl` and replace:
- `doc_id`: set to your indexed document id (or `null` to search all)
- `relevant_pages` and/or `relevant_chunk_ids`
- `expected_refusal` for questions that should be declined

### 2) Run evaluation

```bash
cd "/Users/moulya.r.b/Desktop/Final Project"
source .venv/bin/activate
python evaluation/run_eval.py --gold evaluation/gold_qa_template.jsonl --k 5 --out evaluation/eval_report.json
```

The command prints summary metrics and writes full per-question results to `evaluation/eval_report.json`.
