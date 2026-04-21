from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag import answer_question, gather_contexts


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float


@dataclass
class AnswerMetrics:
    citation_count: int
    refusal_correct: bool | None
    citation_hit_rate: float | None


def _load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _retrieval_metrics(
    retrieved: list[tuple[str, dict, str, float]],
    relevant_chunk_ids: set[str],
    relevant_pages: set[int],
    k: int,
) -> RetrievalMetrics:
    top = retrieved[:k]
    if not top:
        return RetrievalMetrics(precision_at_k=0.0, recall_at_k=0.0)

    hits = 0
    for chunk_id, meta, _text, _sim in top:
        page = int(meta.get("page") or 0)
        by_id = chunk_id in relevant_chunk_ids if relevant_chunk_ids else False
        by_page = page in relevant_pages if relevant_pages else False
        if by_id or by_page:
            hits += 1

    precision = hits / len(top)
    denom = len(relevant_chunk_ids) if relevant_chunk_ids else len(relevant_pages)
    recall = (hits / denom) if denom > 0 else 0.0
    return RetrievalMetrics(precision_at_k=precision, recall_at_k=recall)


def _answer_metrics(
    answer: str,
    citations: list[Any],
    relevant_chunk_ids: set[str],
    relevant_pages: set[int],
    expected_refusal: bool | None,
) -> AnswerMetrics:
    citation_count = len(re.findall(r"\[\d+\]", answer))
    refusal_text = "The document does not provide enough information in the retrieved context."
    refusal_correct = None
    if expected_refusal is not None:
        refusal_correct = (refusal_text in answer) == expected_refusal

    citation_hit_rate: float | None = None
    if citations and (relevant_chunk_ids or relevant_pages):
        matched = 0
        for c in citations:
            cid = str(getattr(c, "chunk_id", ""))
            page = int(getattr(c, "page", 0))
            by_id = cid in relevant_chunk_ids if relevant_chunk_ids else False
            by_page = page in relevant_pages if relevant_pages else False
            if by_id or by_page:
                matched += 1
        citation_hit_rate = matched / len(citations)

    return AnswerMetrics(
        citation_count=citation_count,
        refusal_correct=refusal_correct,
        citation_hit_rate=citation_hit_rate,
    )


def run(gold_path: Path, k: int) -> dict[str, Any]:
    gold = _load_gold(gold_path)
    retrieval_ps: list[float] = []
    retrieval_rs: list[float] = []
    citation_counts: list[float] = []
    citation_hits: list[float] = []
    refusal_scores: list[float] = []
    per_question: list[dict[str, Any]] = []

    for item in gold:
        qid = str(item.get("id") or "")
        question = str(item["question"])
        doc_id = item.get("doc_id")
        relevant_chunk_ids = {str(x) for x in item.get("relevant_chunk_ids", [])}
        relevant_pages = {int(x) for x in item.get("relevant_pages", [])}
        expected_refusal = item.get("expected_refusal")

        retrieved = gather_contexts(question, doc_id)
        rm = _retrieval_metrics(retrieved, relevant_chunk_ids, relevant_pages, k=k)
        retrieval_ps.append(rm.precision_at_k)
        retrieval_rs.append(rm.recall_at_k)

        rag = answer_question(question, doc_id)
        am = _answer_metrics(
            rag.answer,
            rag.citations,
            relevant_chunk_ids,
            relevant_pages,
            expected_refusal if isinstance(expected_refusal, bool) else None,
        )
        citation_counts.append(float(am.citation_count))
        if am.citation_hit_rate is not None:
            citation_hits.append(am.citation_hit_rate)
        if am.refusal_correct is not None:
            refusal_scores.append(1.0 if am.refusal_correct else 0.0)

        per_question.append(
            {
                "id": qid,
                "question": question,
                "precision_at_k": rm.precision_at_k,
                "recall_at_k": rm.recall_at_k,
                "citation_count": am.citation_count,
                "citation_hit_rate": am.citation_hit_rate,
                "refusal_correct": am.refusal_correct,
            }
        )

    out: dict[str, Any] = {
        "n_questions": len(gold),
        "retrieval_precision_at_k_mean": mean(retrieval_ps) if retrieval_ps else 0.0,
        "retrieval_recall_at_k_mean": mean(retrieval_rs) if retrieval_rs else 0.0,
        "answer_citation_count_mean": mean(citation_counts) if citation_counts else 0.0,
        "answer_citation_hit_rate_mean": mean(citation_hits) if citation_hits else None,
        "refusal_correctness_mean": mean(refusal_scores) if refusal_scores else None,
        "per_question": per_question,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + RAG metrics from gold JSONL.")
    parser.add_argument(
        "--gold",
        default="evaluation/gold_qa_template.jsonl",
        help="Path to JSONL with fields: question, doc_id, relevant_chunk_ids/relevant_pages, expected_refusal.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics.")
    parser.add_argument("--out", default="evaluation/eval_report.json", help="Output JSON path.")
    args = parser.parse_args()

    report = run(Path(args.gold), args.k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    print(f"\nSaved full report: {out_path}")


if __name__ == "__main__":
    main()
