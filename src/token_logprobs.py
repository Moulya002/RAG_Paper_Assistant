"""
Summarize output-token log-probabilities from chat completions (for RAG evaluation).

Log-probs are natural-log probabilities of the sampled token at each step.
Perplexity uses the standard length-normalized definition: exp(-mean(log p)).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TokenLogprobSummary:
    """Aggregate stats over the assistant message content tokens."""

    num_output_tokens: int
    mean_logprob: float
    std_logprob: float
    min_logprob: float
    min_token: str
    perplexity: float
    geometric_mean_prob: float
    finish_reason: str
    token_details: list[dict[str, Any]]
    lowest_logprob_tokens: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        p = d.get("perplexity")
        if isinstance(p, float) and (math.isinf(p) or math.isnan(p)):
            d["perplexity"] = None
        return d


def _visible_token(t: str) -> str:
    return t.replace("\n", "↵").replace("\t", "→")


def summarize_choice_logprobs(choice) -> TokenLogprobSummary | None:
    """Build summary from OpenAI `Choice` (expects `logprobs.content` when requested)."""
    logprobs_obj = getattr(choice, "logprobs", None)
    if logprobs_obj is None:
        return None
    content = getattr(logprobs_obj, "content", None) or []
    rows: list[tuple[str, float]] = []
    for item in content:
        token = getattr(item, "token", None)
        lp = getattr(item, "logprob", None)
        if token is None or lp is None:
            continue
        rows.append((str(token), float(lp)))
    if not rows:
        return None

    lps = [r[1] for r in rows]
    mean = float(statistics.mean(lps))
    std = float(statistics.pstdev(lps)) if len(lps) > 1 else 0.0
    min_i = min(range(len(lps)), key=lambda i: lps[i])
    min_lp, min_tok = lps[min_i], rows[min_i][0]
    perplexity = float(math.exp(-mean)) if mean < 100 else float("inf")
    geom_p = float(math.exp(mean))
    finish = str(getattr(choice, "finish_reason", "") or "")

    token_details: list[dict[str, Any]] = []
    for tok, lp in rows:
        token_details.append(
            {
                "token": _visible_token(tok),
                "logprob": lp,
                "probability": float(math.exp(lp)),
            }
        )

    sorted_idx = sorted(range(len(lps)), key=lambda i: lps[i])
    lowest: list[dict[str, Any]] = []
    for i in sorted_idx[:15]:
        tok, lp = rows[i]
        lowest.append(
            {
                "token": _visible_token(tok),
                "logprob": lp,
                "probability": float(math.exp(lp)),
            }
        )

    return TokenLogprobSummary(
        num_output_tokens=len(rows),
        mean_logprob=mean,
        std_logprob=std,
        min_logprob=min_lp,
        min_token=_visible_token(min_tok),
        perplexity=perplexity,
        geometric_mean_prob=geom_p,
        finish_reason=finish,
        token_details=token_details,
        lowest_logprob_tokens=lowest,
    )
