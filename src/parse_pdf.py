"""
PDF text extraction with section heuristics and hybrid chunking
(section boundaries + sliding windows), per project proposal.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from src.config import CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS


@dataclass
class TextBlock:
    page: int
    text: str
    section_hint: str | None


SECTION_PATTERNS = [
    (re.compile(r"^\s*(abstract)\s*$", re.I), "Abstract"),
    (re.compile(r"^\s*(1\.?\s*)?introduction\b", re.I), "Introduction"),
    (re.compile(r"^\s*(2\.?\s*)?(related work|background)\b", re.I), "Related Work"),
    (re.compile(r"^\s*(3\.?\s*)?(method|methodology|materials and methods)\b", re.I), "Methodology"),
    (re.compile(r"^\s*(4\.?\s*)?(experiment|experiments|results?)\b", re.I), "Results"),
    (re.compile(r"^\s*(5\.?\s*)?discussion\b", re.I), "Discussion"),
    (re.compile(r"^\s*(6\.?\s*)?conclusion\b", re.I), "Conclusion"),
    (re.compile(r"^\s*references?\b", re.I), "References"),
    (re.compile(r"^\s*acknowledg(e)?ments?\b", re.I), "Acknowledgements"),
]


def _guess_section(line: str, current: str) -> str:
    stripped = line.strip()
    if len(stripped) > 120:
        return current
    for pat, name in SECTION_PATTERNS:
        if pat.search(stripped):
            return name
    return current


EXCLUDED_EMBED_SECTIONS = {"References", "Acknowledgements"}


def _clean_text_artifacts(text: str) -> str:
    t = text
    t = t.replace("<EOS>", " ").replace("<eos>", " ")
    t = t.replace("<pad>", " ").replace("[PAD]", " ")
    t = t.replace("\x00", " ")
    # Remove numeric citation markers in prose, e.g. [12], [2, 5], [3][4]
    t = re.sub(r"\[(?:\d{1,3}\s*(?:,\s*\d{1,3})*)\]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _looks_like_reference_entry(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    # Common bibliography entry patterns
    if re.match(r"^\[\d{1,3}\]\s+", s):
        return True
    if re.match(r"^\d{1,3}\.\s+[A-Z][a-z]+", s):
        return True
    if re.search(r"\b(Proceedings|Journal|vol\.|doi:|arXiv:)\b", s, re.I):
        if len(s) > 120:
            return True
    # Author-heavy bibliography style lines with years
    if re.search(r"\b(19|20)\d{2}\b", s) and s.count(",") >= 4:
        return True
    return False


def _is_noise_paragraph(text: str) -> bool:
    s = text.strip()
    if len(s) < 40:
        return True
    if re.fullmatch(r"[\d\W_]+", s):
        return True
    if _looks_like_reference_entry(s):
        return True
    return False


def _infer_section_by_position(page_no: int, total_pages: int) -> str:
    if total_pages <= 1:
        return "Body"
    ratio = page_no / total_pages
    if ratio <= 0.2:
        return "Introduction"
    if ratio <= 0.7:
        return "Methodology"
    if ratio <= 0.9:
        return "Results"
    return "Conclusion"


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(ch.isalnum() for ch in text)
    return alnum / max(1, len(text))


def _is_low_information_text(text: str) -> bool:
    # Reject symbol-heavy / low-signal strings.
    if _alnum_ratio(text) < 0.55:
        return True
    # Reject repetitive separators and equation-like remnants.
    if re.search(r"[=_\-]{8,}", text):
        return True
    return False


def _collect_repeated_margin_lines(doc: fitz.Document) -> set[str]:
    """
    Detect repeated short lines that appear on many pages (likely headers/footers).
    These are removed before paragraph chunking.
    """
    first_last_counter: Counter[str] = Counter()
    total_pages = len(doc)
    for i in range(total_pages):
        page_text = doc[i].get_text("text") or ""
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        if not lines:
            continue
        candidates = []
        if lines:
            candidates.append(lines[0])
        if len(lines) > 1:
            candidates.append(lines[-1])
        for ln in candidates:
            if 3 <= len(ln) <= 120:
                first_last_counter[ln] += 1
    # Mark as margin noise if repeated on at least 3 pages and >=20% of document.
    threshold = max(3, int(total_pages * 0.2))
    return {ln for ln, c in first_last_counter.items() if c >= threshold}


def extract_blocks(pdf_path: str) -> tuple[str, list[TextBlock]]:
    doc = fitz.open(pdf_path)
    title = (doc.metadata or {}).get("title") or ""
    if not title.strip():
        title = Path(pdf_path).stem.replace("_", " ")

    blocks: list[TextBlock] = []
    total_pages = len(doc)
    current_section = "Unknown"
    repeated_margin_lines = _collect_repeated_margin_lines(doc)

    for page_index in range(total_pages):
        page = doc[page_index]
        page_no = page_index + 1
        text = page.get_text("text") or ""
        if repeated_margin_lines:
            kept_lines = []
            for ln in text.splitlines():
                stripped = ln.strip()
                if stripped and stripped in repeated_margin_lines:
                    continue
                kept_lines.append(ln)
            text = "\n".join(kept_lines)
        for para in _split_paragraphs(text):
            line = para.split("\n", 1)[0] if para else ""
            current_section = _guess_section(line, current_section)
            normalized = _normalize_noise(para)
            if current_section in EXCLUDED_EMBED_SECTIONS:
                continue
            if current_section == "Unknown":
                current_section = _infer_section_by_position(page_no, total_pages)
            # Noise filtering must run before citation marker stripping so
            # bibliography patterns like "[25] Author..." are still detectable.
            if _is_noise_paragraph(normalized):
                continue
            cleaned = _clean_text_artifacts(normalized)
            if len(cleaned) < 40:
                continue
            if _is_low_information_text(cleaned):
                continue
            blocks.append(TextBlock(page=page_no, text=cleaned, section_hint=current_section))

    doc.close()
    return title.strip() or "Untitled document", blocks


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _normalize_noise(text: str) -> str:
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def char_windows(text: str, target: int, overlap: int) -> Iterator[str]:
    clean = text.strip()
    if not clean:
        return
    n = len(clean)
    start = 0
    while start < n:
        end = min(start + target, n)
        yield clean[start:end].strip()
        if end >= n:
            break
        start = max(end - overlap, start + 1)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    page: int
    text: str


def build_chunks(
    doc_id: str,
    title: str,
    blocks: list[TextBlock],
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    section_buffers: dict[str, list[TextBlock]] = {}
    order: list[str] = []
    for block in blocks:
        section = block.section_hint or "Body"
        if section not in section_buffers:
            section_buffers[section] = []
            order.append(section)
        section_buffers[section].append(block)

    for section in order:
        sec_blocks = section_buffers[section]
        # Concatenate section text so headings like Results/Conclusion are preserved as semantic units.
        full_text = "\n\n".join(b.text for b in sec_blocks).strip()
        first_page = sec_blocks[0].page if sec_blocks else 0
        for window in char_windows(full_text, target_chars, overlap_chars):
            if len(window.strip()) < 80:
                continue
            records.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    page=first_page,
                    text=window.strip(),
                )
            )
    return records


def ingest_pdf(pdf_path: str, doc_id: str | None = None) -> tuple[str, str, list[ChunkRecord]]:
    import uuid as uu

    doc_id = doc_id or str(uu.uuid4())
    title, blocks = extract_blocks(pdf_path)
    chunks = build_chunks(doc_id, title, blocks)
    return doc_id, title, chunks
