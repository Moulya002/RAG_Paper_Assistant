"""
Streamlit UI: upload PDFs, index into Chroma, ask questions via RAG (retrieve + one LLM call).
Not agentic AI: no autonomous tool use or multi-step agent loop.
Run from project root: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import string
import uuid
from pathlib import Path
import streamlit as st
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHROMA_DIR, DATA_DIR, UPLOADS_DIR, LLM_CONFIGURED
from src.parse_pdf import ingest_pdf
from src.vector_store import add_chunks, get_collection
from src.rag import answer_question


QUICK_ACTIONS: list[tuple[str, str]] = [
    (
        "📄 Summarize Paper",
        "Summarize the paper with:\n- main idea\n- key methods\n- results / conclusions",
    ),
    ("🧠 Main Contribution", "What is the main contribution of this paper?"),
    ("⚙️ Methodology", "Explain the methodology used in this paper."),
    (
        "📉 Limitations & Future Work",
        "What limitations does this paper mention, and what future work directions are suggested?",
    ),
    ("❓ Key Concepts", "List and briefly explain the key concepts in this paper."),
    (
        "🧒 Explain Like I'm 12",
        "Explain the main idea of this paper in very simple terms, as if explaining to a 12-year-old. "
        "Avoid technical jargon. Keep it short.",
    ),
]

SUGGESTED_QUESTIONS: list[str] = [
    "What problem does this paper solve?",
    "What dataset is used?",
    "What are the limitations?",
    "What is the main contribution?",
]


def _normalize_text(s: str) -> str:
    t = (s or "").lower().strip()
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t)
    return t


def _token_f1(pred: str, gold: str) -> float:
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = {}
    for tok in p_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in g_tokens:
        c = common.get(tok, 0)
        if c > 0:
            overlap += 1
            common[tok] = c - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_live_doc_scores(
    answer: str,
    citations: list,
    retrieval_confidence: float | None,
) -> dict:
    """
    Always-available live score per answer/document.
    This is a groundedness/support score (not benchmark exact-match).
    """
    n_citations = len(citations or [])
    citation_coverage = min(1.0, n_citations / 4.0)  # saturates at 4 citations
    has_refusal = "not provide enough information in the retrieved context" in answer.lower()
    refusal_penalty = 0.85 if has_refusal else 1.0
    retr = retrieval_confidence if retrieval_confidence is not None else 0.0
    # Weighted live score: retrieval quality + citation support.
    live_accuracy_estimate = (0.7 * retr + 0.3 * citation_coverage) * 100.0 * refusal_penalty
    return {
        "live_accuracy_estimate": max(0.0, min(100.0, live_accuracy_estimate)),
        "retrieval_confidence": retr * 100.0,
        "citation_coverage": citation_coverage * 100.0,
        "citation_count": n_citations,
    }


def _format_answer_for_display(answer: str) -> str:
    """
    Improve readability of structured summary headings without changing model logic.
    """
    if not answer:
        return answer
    text = answer
    # Normalize common "Step N: Heading" variants into markdown headings.
    text = re.sub(
        r"(?im)^\s*step\s*\d+\s*:\s*main idea\s*:?\s*$",
        "### Main idea:",
        text,
    )
    text = re.sub(
        r"(?im)^\s*step\s*\d+\s*:\s*key methods?\s*:?\s*$",
        "### Key methods:",
        text,
    )
    text = re.sub(
        r"(?im)^\s*step\s*\d+\s*:\s*results?\s*/\s*conclusions?\s*:?\s*$",
        "### Results / conclusions:",
        text,
    )
    # Also handle plain heading lines without "Step".
    text = re.sub(r"(?im)^\s*main idea\s*:?\s*$", "### Main idea:", text)
    text = re.sub(r"(?im)^\s*key methods?\s*:?\s*$", "### Key methods:", text)
    text = re.sub(
        r"(?im)^\s*results?\s*/\s*conclusions?\s*:?\s*$",
        "### Results / conclusions:",
        text,
    )
    # Handle inline numbered forms like:
    # "1. Main idea The main idea is ..."
    # "2. Key methods ..."
    # "3. Results / conclusions ..."
    text = re.sub(
        r"(?is)\b1\.\s*Main idea\s*",
        "\n### Main idea:\n",
        text,
    )
    text = re.sub(
        r"(?is)\b2\.\s*Key methods?\s*",
        "\n### Key methods:\n",
        text,
    )
    text = re.sub(
        r"(?is)\b3\.\s*Results?\s*/\s*conclusions?\s*",
        "\n### Results / conclusions:\n",
        text,
    )
    # Clean up excessive blank lines introduced by formatting.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _normalize_citation(c: object) -> dict:
    if isinstance(c, dict):
        return {
            "index": int(c.get("index", 0)),
            "page": int(c.get("page", 0)),
            "section": str(c.get("section") or "—"),
            "quoted_text": str(c.get("quoted_text") or c.get("excerpt") or ""),
            "title": str(c.get("title") or ""),
            "chunk_id": str(c.get("chunk_id") or ""),
        }
    return {
        "index": int(getattr(c, "index", 0)),
        "page": int(getattr(c, "page", 0)),
        "section": str(getattr(c, "section", None) or "—"),
        "quoted_text": str(getattr(c, "quoted_text", None) or getattr(c, "excerpt", None) or ""),
        "title": str(getattr(c, "title", None) or ""),
        "chunk_id": str(getattr(c, "chunk_id", None) or ""),
    }


def _render_sources_section(citations: list) -> None:
    if not citations:
        return
    st.markdown("#### Sources")
    st.caption("Each row matches one retrieved chunk; citation numbers align with inline [1], [2], … in the answer.")
    rows = [_normalize_citation(c) for c in citations]
    rows.sort(key=lambda r: r["index"])
    for j, row in enumerate(rows):
        sec = row["section"] if row["section"] and row["section"] != "—" else "(section not detected)"
        st.markdown(f"**[{row['index']}]** — page **{row['page']}** — _{sec}_")
        if row.get("title"):
            st.caption(row["title"])
        quote = row["quoted_text"]
        if quote:
            lines = [ln.strip() for ln in quote.splitlines() if ln.strip()]
            preview_lines = lines[:3]
            preview_text = "\n".join(preview_lines).strip()
            if len(preview_text) > 280:
                preview_text = preview_text[:280].rstrip() + "..."
            st.markdown(preview_text)
            with st.expander("Expand full source"):
                st.markdown(f"> {quote}")
        else:
            st.caption("_Empty chunk text_")
        if j < len(rows) - 1:
            st.markdown("---")


def _render_logprob_summary(summary: dict, download_key: str) -> None:
    st.caption(
        "Next-token **log-probabilities** from the generator (natural log of the probability of each "
        "chosen token). Useful as a *secondary* signal: higher average logprob / lower perplexity often "
        "means more decisive wording, but this does **not** measure factual grounding to the PDF."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Output tokens", summary.get("num_output_tokens", 0))
    c2.metric("Mean logprob", f"{summary.get('mean_logprob', 0):.4f}")
    perp = summary.get("perplexity")
    c3.metric("Perplexity", f"{perp:.2f}" if perp is not None else "—")
    c4.metric("Geom. mean p", f"{summary.get('geometric_mean_prob', 0):.5f}")
    st.write(f"**Finish reason:** `{summary.get('finish_reason', '')}`")
    st.write(f"**Min logprob token:** `{summary.get('min_token', '')}` ({summary.get('min_logprob', 0):.4f})")

    lows = summary.get("lowest_logprob_tokens") or []
    if lows:
        st.markdown("**Lowest-probability tokens** (often punctuation / rare words; interpret carefully)")
        st.dataframe(lows, use_container_width=True, hide_index=True)

    details = summary.get("token_details") or []
    if details:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["token", "logprob", "probability"])
        writer.writeheader()
        writer.writerows(details)
        st.download_button(
            label="Download all output-token logprobs (CSV)",
            data=buf.getvalue(),
            file_name="rag_output_token_logprobs.csv",
            mime="text/csv",
            key=download_key,
        )


def _render_correctness_panel(correctness_score: float | None, retrieval_confidence: float | None) -> None:
    c1, c2 = st.columns(2)
    if correctness_score is not None:
        c1.metric("Retrieval confidence (proxy)", f"{correctness_score:.1f}/100")
    else:
        c1.metric("Retrieval confidence (proxy)", "—")
    if retrieval_confidence is not None:
        c2.metric("Avg semantic similarity", f"{retrieval_confidence * 100:.1f}%")
    else:
        c2.metric("Avg semantic similarity", "—")
    st.caption(
        "This is a proxy derived from token probability (when available) and/or retrieval similarity. "
        "It is not a ground-truth correctness score."
    )


def _render_retrieval_debug(chunks: list[dict] | None) -> None:
    if not chunks:
        st.info("No retrieved chunk debug info.")
        return
    st.caption("Retrieved chunks (preview + similarity) before LLM generation.")
    for c in chunks:
        st.markdown(
            f"**#{c.get('rank', '?')}** | sim={float(c.get('similarity', 0.0)):.3f} | "
            f"page={c.get('page', '?')} | section={c.get('section', '—')}"
        )
        st.code(str(c.get("preview", "")))


def _render_live_doc_scores(scores: dict | None) -> None:
    if not scores:
        st.info("No live scores available for this answer.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live accuracy estimate", f"{float(scores.get('live_accuracy_estimate', 0.0)):.1f}%")
    c2.metric("Retrieval confidence", f"{float(scores.get('retrieval_confidence', 0.0)):.1f}%")
    c3.metric("Citation coverage", f"{float(scores.get('citation_coverage', 0.0)):.1f}%")
    c4.metric("Citation count", f"{int(scores.get('citation_count', 0))}")
    st.caption("Live score is computed per answer from retrieval quality + citation support for this document.")


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    st.set_page_config(page_title="Research paper RAG", layout="wide")
    ensure_data_dirs()

    st.title("Research paper QA (RAG)")
    st.caption(
        "Retrieval-Augmented Generation: embed PDF chunks, retrieve relevant passages (with MMR), "
        "then generate one answer conditioned on those passages—no agent loop, tools, or autonomous planning."
    )

    with st.sidebar:
        st.subheader("Index a paper")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded and st.button("Parse & index"):
            doc_id = str(uuid.uuid4())
            dest = UPLOADS_DIR / f"{doc_id}.pdf"
            dest.write_bytes(uploaded.getvalue())
            with st.spinner("Extracting text and building chunks…"):
                doc_id, title, chunks = ingest_pdf(str(dest), doc_id=doc_id)
                col = get_collection()
                add_chunks(col, chunks)
            st.success(f"Indexed: **{title}** ({len(chunks)} chunks)")
            st.session_state.setdefault("indexed", []).append({"doc_id": doc_id, "title": title})

        st.subheader("Scope")
        col = get_collection()
        try:
            count = col.count()
        except Exception:
            count = 0
        st.write(f"Chunks in store: **{count}**")

        doc_filter = st.radio(
            "Retrieve from",
            options=["All uploaded papers", "Selected paper only"],
            index=0,
        )
        titles_meta = col.get(include=["metadatas"], limit=min(500, max(1, count or 1)))
        metas = (titles_meta.get("metadatas") or []) if count else []
        seen: dict[str, str] = {}
        for m in metas:
            if not m:
                continue
            did = m.get("doc_id")
            ttl = m.get("title")
            if did and ttl and did not in seen:
                seen[str(did)] = str(ttl)
        doc_options = list(seen.items())
        selected_doc = None
        if doc_filter == "Selected paper only":
            if not doc_options:
                st.info("Index at least one PDF to filter by paper.")
            else:
                labels = [f"{t} ({d[:8]}…)" for d, t in doc_options]
                pick = st.selectbox("Paper", range(len(labels)), format_func=lambda i: labels[i])
                selected_doc = doc_options[pick][0]

        if st.button("Clear all indexed chunks"):
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            st.success("Vector store cleared. Reload the page.")
            st.stop()

        if not LLM_CONFIGURED:
            st.error("Set `GROQ_API_KEY` in `.env` (project root) for chat answers.")

        if count > 0:
            st.markdown("### Suggested Questions")
            for i, q in enumerate(SUGGESTED_QUESTIONS):
                if st.button(q, key=f"suggested_q_{i}", use_container_width=True):
                    st.session_state["pending_query"] = q

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant":
                content = _format_answer_for_display(content)
            st.markdown(content)
            if msg["role"] == "assistant" and msg.get("citations"):
                _render_sources_section(msg["citations"])
            if msg["role"] == "assistant":
                with st.expander("Confidence", expanded=False):
                    _render_correctness_panel(
                        msg.get("correctness_score"),
                        msg.get("retrieval_confidence"),
                    )
                with st.expander("Live scores (per-document)", expanded=False):
                    _render_live_doc_scores(msg.get("live_doc_scores"))
                with st.expander("Retrieval debug (chunks + scores)", expanded=False):
                    _render_retrieval_debug(msg.get("retrieved_chunks_debug"))
            if msg["role"] == "assistant" and msg.get("logprob_summary"):
                with st.expander("Evaluation: output-token log-probabilities"):
                    _render_logprob_summary(msg["logprob_summary"], download_key=f"logprob_hist_{i}")

    st.markdown("### Quick Actions")
    qa_cols = st.columns(3)
    for i, (label, query) in enumerate(QUICK_ACTIONS):
        with qa_cols[i % 3]:
            if st.button(label, key=f"quick_action_{i}", use_container_width=True):
                st.session_state["pending_query"] = query

    prompt = st.chat_input("Ask about your indexed papers…")
    pending_query = st.session_state.pop("pending_query", None)
    run_query = pending_query or prompt
    if not run_query:
        return

    st.session_state.messages.append({"role": "user", "content": run_query})
    with st.chat_message("user"):
        st.markdown(run_query)

    doc_id = None
    if doc_filter == "Selected paper only":
        doc_id = selected_doc

    with st.chat_message("assistant"):
        if not LLM_CONFIGURED:
            st.error("Missing `GROQ_API_KEY` in `.env` (project folder).")
            return
        try:
            with st.spinner("Analyzing paper..."):
                result = answer_question(run_query, doc_id=doc_id)
        except Exception as e:
            st.error(str(e))
            return
        if not result.citations:
            formatted_answer = "No relevant information found in the document."
        else:
            formatted_answer = _format_answer_for_display(result.answer)
        st.markdown(formatted_answer)
        with st.expander("Confidence", expanded=False):
            _render_correctness_panel(
                result.correctness_score,
                result.retrieval_confidence,
            )
        live_doc_scores = _compute_live_doc_scores(
            result.answer,
            result.citations,
            result.retrieval_confidence,
        )
        with st.expander("Live scores (per-document)", expanded=False):
            _render_live_doc_scores(live_doc_scores)
        with st.expander("Retrieval debug (chunks + scores)", expanded=False):
            _render_retrieval_debug(result.retrieved_chunks_debug)
        if result.citations:
            _render_sources_section(result.citations)
        logprob_dict = None
        if result.logprob_summary is not None:
            logprob_dict = result.logprob_summary.to_dict()
            with st.expander("Evaluation: output-token log-probabilities"):
                _render_logprob_summary(logprob_dict, download_key="logprob_latest")

        payload = {
            "role": "assistant",
            "content": formatted_answer,
            "citations": [c.to_dict() for c in result.citations],
            "correctness_score": result.correctness_score,
            "retrieval_confidence": result.retrieval_confidence,
            "retrieved_chunks_debug": result.retrieved_chunks_debug,
            "live_doc_scores": live_doc_scores,
        }
        if logprob_dict is not None:
            payload["logprob_summary"] = logprob_dict
        st.session_state.messages.append(payload)


if __name__ == "__main__":
    main()
