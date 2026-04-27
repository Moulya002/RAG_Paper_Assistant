"""
Citation-aware QA via RAG: retrieve chunks from the vector store, then a single LLM call.
This is not an agentic system (no tool use, ReAct loop, or planner).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from openai import BadRequestError, OpenAI

from src.config import (
    CHAT_LOGPROBS,
    CHAT_TOP_LOGPROBS,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_CHAT_MODEL,
    LLM_CONFIGURED,
    MMR_LAMBDA,
    MMR_SELECT,
    MIN_RELEVANCE_SIM,
    RETRIEVAL_TOP_K,
    SUMMARY_MMR_SELECT,
    SUMMARY_RETRIEVAL_K,
)
from src.token_logprobs import TokenLogprobSummary, summarize_choice_logprobs
from src.retrieve import retrieve_with_mmr
from src.vector_store import get_collection, query_raw

logger = logging.getLogger(__name__)

def _groq_client() -> OpenAI:
    """OpenAI-compatible client targeting Groq chat completions."""
    if not GROQ_API_KEY:
        raise RuntimeError("Set GROQ_API_KEY in the project `.env` file.")
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


SYSTEM_PROMPT = """You are the generation step of a retrieval-augmented (RAG) system.

You will receive numbered contexts. Each context is the ONLY evidence you may use. Each includes:
- Citation number (use this exact number in brackets)
- Document title, section name (if known), page number
- The exact chunk text from the PDF (verbatim excerpt)

STRICT RULES:
1. Answer ONLY using information supported by those numbered contexts. Do not use outside knowledge, guesses, or unstated assumptions.
2. Every substantive sentence or clause must carry inline citations using the context numbers exactly as given, e.g. [1], [2], [1][3]. Place citations immediately after the supported text.
3. If the contexts are sparse, provide the best possible answer from available evidence and clearly state what is missing.
4. You may paraphrase, but any factual claim must still be tied to the cited chunk(s). Prefer short quotes from the chunk text when precision matters, and cite them.
5. Do not invent section names, page numbers, authors, or citations beyond [1]..[n] for the contexts supplied.
6. Avoid vague wording such as "appears to be", "seems", "likely", or speculative language.
7. Prefer short, direct, evidence-based statements over generic textbook phrasing.
8. Only refuse when there is genuinely no relevant context at all."""


@dataclass
class Citation:
    """One retrieved chunk, numbered for inline [n] references in the answer."""

    index: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    page: int
    quoted_text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagResult:
    answer: str
    citations: list[Citation]
    logprob_summary: TokenLogprobSummary | None = None
    retrieval_confidence: float | None = None
    correctness_score: float | None = None
    retrieved_chunks_debug: list[dict] | None = None


def _flatten_chroma(res: dict) -> tuple[list[str], list[str], list[dict]]:
    ids = (res.get("ids") or [[]])[0] or []
    docs = (res.get("documents") or [[]])[0] or []
    metas = (res.get("metadatas") or [[]])[0] or []
    return ids, docs, metas


def gather_contexts(
    query: str,
    doc_id: str | None,
    retrieval_k: int = RETRIEVAL_TOP_K,
    mmr_select: int = MMR_SELECT,
    min_relevance_sim: float = MIN_RELEVANCE_SIM,
    prefer_sections: set[str] | None = None,
) -> list[tuple[str, dict, str, float]]:
    """Retrieve candidates then MMR; returns (chunk_id, metadata, chunk_text, sim)."""
    col = get_collection()
    wide_k = max(retrieval_k * 3, 20)
    raw = query_raw(col, query, k=wide_k, doc_id=doc_id)
    ids_list, texts, metas = _flatten_chroma(raw)
    n = min(len(ids_list), len(texts), len(metas))
    ids_list = [str(ids_list[i]) for i in range(n)]
    texts = [texts[i] or "" for i in range(n)]
    metas = [metas[i] or {} for i in range(n)]

    chunk_ids: list[str] = []
    for i in range(n):
        cid = ids_list[i] if i < len(ids_list) else ""
        if not cid and metas[i].get("chunk_id"):
            cid = str(metas[i]["chunk_id"])
        chunk_ids.append(cid)

    pairs = retrieve_with_mmr(
        query,
        texts,
        metas,
        chunk_ids,
        lambda_mult=MMR_LAMBDA,
        select_n=min(mmr_select, len(texts)) if texts else 0,
    )
    filtered = [p for p in pairs if p[3] >= min_relevance_sim]
    shortlist = filtered if filtered else pairs[: max(1, min(5, len(pairs)))]

    if prefer_sections:
        prefer_lower = {s.lower() for s in prefer_sections}

        def _rank_key(item: tuple[str, dict, str, float]) -> tuple[int, float]:
            _cid, meta, _txt, sim = item
            section = str(meta.get("section") or "").lower()
            preferred = 1 if any(ps in section for ps in prefer_lower) else 0
            return (preferred, sim)

        shortlist = sorted(shortlist, key=_rank_key, reverse=True)
    return shortlist


def _merge_context_lists(
    groups: list[list[tuple[str, dict, str, float]]], limit: int
) -> list[tuple[str, dict, str, float]]:
    seen: set[str] = set()
    merged: list[tuple[str, dict, str, float]] = []
    for group in groups:
        for item in group:
            cid = item[0] or f"noid-{len(merged)}"
            if cid in seen:
                continue
            seen.add(cid)
            merged.append(item)
    merged.sort(key=lambda x: x[3], reverse=True)
    return merged[:limit]


def _is_summary_query(query: str) -> bool:
    q = query.lower()
    return any(
        kw in q
        for kw in (
            "summarize",
            "summary",
            "summarise",
            "overview",
            "high level",
            "main idea",
        )
    )


def _is_results_query(query: str) -> bool:
    q = query.lower()
    return any(
        kw in q
        for kw in (
            "result",
            "results",
            "conclusion",
            "conclusions",
            "findings",
            "performance",
            "outcome",
            "outcomes",
        )
    )


def _query_expansion_terms(query: str) -> str:
    q = query.lower()
    terms = ["results", "findings", "experiments", "evaluation", "conclusions"]
    if any(x in q for x in ("method", "methodology", "approach", "model")):
        terms.extend(["methodology", "approach", "architecture", "training setup"])
    if any(x in q for x in ("conclusion", "limitation", "future work")):
        terms.extend(["discussion", "limitations", "future work", "conclusion"])
    return " ".join(dict.fromkeys(terms))


def answer_question(query: str, doc_id: str | None = None) -> RagResult:
    if not LLM_CONFIGURED:
        raise RuntimeError("Set GROQ_API_KEY in the project `.env` file.")

    summary_mode = _is_summary_query(query)
    results_mode = _is_results_query(query)
    retrieval_k = SUMMARY_RETRIEVAL_K if summary_mode else max(RETRIEVAL_TOP_K, 6)
    mmr_select = SUMMARY_MMR_SELECT if summary_mode else max(MMR_SELECT, 5)
    min_sim = MIN_RELEVANCE_SIM
    retrieval_query = query
    prefer_sections: set[str] | None = None
    if results_mode:
        retrieval_k = max(retrieval_k, 14)
        mmr_select = max(mmr_select, 10)
        min_sim = min(MIN_RELEVANCE_SIM, 0.12)
        retrieval_query = (
            query
            + "\nFocus on experimental results, findings, metrics, comparisons, discussion, and conclusions."
        )
        prefer_sections = {"results", "conclusion", "discussion", "experiment"}
    query_variants = [retrieval_query]
    if summary_mode:
        query_variants.append(
            "main idea key methods findings contributions limitations conclusion"
        )
    if results_mode:
        query_variants.append(
            "reported results findings outcomes metrics benchmark comparison conclusion"
        )
    query_variants.append(query + "\n" + _query_expansion_terms(query))
    query_variants.append(query)

    context_groups: list[list[tuple[str, dict, str, float]]] = []
    for qv in query_variants:
        context_groups.append(
            gather_contexts(
                qv,
                doc_id,
                retrieval_k=retrieval_k,
                mmr_select=mmr_select,
                min_relevance_sim=min_sim,
                prefer_sections=prefer_sections,
            )
        )
    retrieved = _merge_context_lists(context_groups, limit=max(mmr_select, 6))
    if not retrieved:
        # Last-resort fallback: very permissive pass, then pick top chunks.
        retrieved = gather_contexts(
            query,
            doc_id,
            retrieval_k=max(retrieval_k, 16),
            mmr_select=max(mmr_select, 8),
            min_relevance_sim=-1.0,
            prefer_sections=prefer_sections,
        )

    if not retrieved:
        return RagResult(
            answer="No indexed passages were found. Upload and index a PDF first, or broaden your question.",
            citations=[],
            logprob_summary=None,
            retrieval_confidence=None,
            correctness_score=None,
            retrieved_chunks_debug=[],
        )

    blocks: list[str] = []
    citations: list[Citation] = []
    for i, (chunk_id, meta, text, _sim) in enumerate(retrieved, start=1):
        title = str(meta.get("title") or "Unknown title")
        section = str(meta.get("section") or "Unknown section")
        page = int(meta.get("page") or 0)
        doc = str(meta.get("doc_id") or "")
        cid = chunk_id or str(meta.get("chunk_id") or "")
        quoted = (text or "").strip()
        citations.append(
            Citation(
                index=i,
                chunk_id=cid,
                doc_id=doc,
                title=title,
                section=section,
                page=page,
                quoted_text=quoted,
            )
        )
        blocks.append(
            f"### Context [{i}]\n"
            f"- **Title:** {title}\n"
            f"- **Section:** {section}\n"
            f"- **Page:** {page}\n"
            f"- **Chunk id:** {cid}\n"
            f"- **Chunk text (verbatim, cite with [{i}]):**\n{quoted}\n"
        )

    if summary_mode:
        task_instruction = (
            "Task mode: SUMMARY.\n"
            "Provide a concise structured summary with exactly these headings:\n"
            "1) Main idea\n2) Key methods\n3) Results / conclusions\n"
            "Every line with factual content must include citations [n].\n"
            "Do not output generic background information that is not explicitly present in context.\n"
        )
    elif results_mode:
        task_instruction = (
            "Task mode: RESULTS-FOCUSED QA.\n"
            "Prioritize evidence from results/discussion/conclusion style content when present.\n"
            "Answer with concrete findings from the retrieved context and cite each claim with [n].\n"
            "If no concrete findings are present, output exactly: "
            "\"The document does not provide enough information in the retrieved context.\"\n"
        )
    else:
        task_instruction = (
            "Task mode: QA.\n"
            "Provide a short direct factual answer with inline [n] citations.\n"
            "If evidence is partial, answer what is supported and explicitly mention what is missing.\n"
        )

    user_content = (
        task_instruction
        + "Answer using ONLY the contexts below. Citation numbers [1]..[n] must match these blocks.\n\n"
        + f"**Question:**\n{query.strip()}\n\n"
        + f"**Retrieved contexts:**\n"
        + "\n".join(blocks)
    )

    client = _groq_client()
    base_kwargs = dict(
        model=GROQ_CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    logprob_summary: TokenLogprobSummary | None = None
    use_logprobs = CHAT_LOGPROBS
    if use_logprobs:
        try:
            completion = client.chat.completions.create(
                **base_kwargs,
                logprobs=True,
                top_logprobs=max(0, min(20, CHAT_TOP_LOGPROBS)),
            )
        except BadRequestError:
            completion = client.chat.completions.create(**base_kwargs)
    else:
        completion = client.chat.completions.create(**base_kwargs)

    choice = completion.choices[0]
    answer = (choice.message.content or "").strip()
    if use_logprobs and choice.logprobs is not None:
        logprob_summary = summarize_choice_logprobs(choice)
    sims = [float(p[3]) for p in retrieved]
    retrieval_conf = max(0.0, min(1.0, ((sum(sims) / len(sims)) + 1.0) / 2.0)) if sims else None
    if logprob_summary is not None:
        correctness = max(0.0, min(100.0, logprob_summary.geometric_mean_prob * 100.0))
    elif retrieval_conf is not None:
        correctness = retrieval_conf * 100.0
    else:
        correctness = None

    retrieved_debug: list[dict] = []
    for i, (cid, meta, txt, sim) in enumerate(retrieved, start=1):
        preview = (txt or "").replace("\n", " ").strip()[:200]
        retrieved_debug.append(
            {
                "rank": i,
                "chunk_id": cid,
                "section": str(meta.get("section") or ""),
                "page": int(meta.get("page") or 0),
                "similarity": float(sim),
                "preview": preview,
            }
        )
        logger.info(
            "RAG retrieved #%s | sim=%.4f | section=%s | page=%s | preview=%s",
            i,
            float(sim),
            str(meta.get("section") or ""),
            int(meta.get("page") or 0),
            preview,
        )

    return RagResult(
        answer=answer,
        citations=citations,
        logprob_summary=logprob_summary,
        retrieval_confidence=retrieval_conf,
        correctness_score=correctness,
        retrieved_chunks_debug=retrieved_debug,
    )
