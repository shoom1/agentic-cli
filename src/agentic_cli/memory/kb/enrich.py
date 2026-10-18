"""Enriching a document: the one summarizer call that writes its summary,
claims and entities.

Ingestion asks a knowledge base's summarizer once per document, with the
sidecar prompt: the answer's SUMMARY (a 3–5 paragraph synthesis) becomes
the document's summary, and its CLAIMS and ENTITIES the sidecar's. Without
a summarizer, or when the call fails or the answer has none of the three
sections, the summary is the first ~500 characters and there are no
claims. ``summarize`` (the 2–3 sentence prompt) is what
``KnowledgeBaseManager.generate_summary`` returns.
"""

from __future__ import annotations

import re
from typing import Any

from agentic_cli.memory._core.log import get_logger

logger = get_logger("agentic_cli.memory.kb")

# Long documents can exceed sensible prompt budgets, and the summary only
# needs the lead-in to be useful.
SUMMARY_INPUT_CHAR_LIMIT = 12_000

_PREVIEW_CHARS = 500

SUMMARY_PROMPT_TEMPLATE = (
    "Summarize the following document in 2-3 sentences (max ~500 "
    "characters). Focus on the main topic, key contributions, and "
    "why the document is relevant. Return only the summary text "
    "with no preamble or markdown.\n\n"
    "Title: {title}\n\n"
    "Content:\n---\n{content}\n---"
)

SIDECAR_PROMPT_TEMPLATE = (
    "Extract a structured payload from the document below. "
    "Return PLAIN TEXT in exactly this layout, with section headers "
    "in ALL CAPS followed by a colon:\n\n"
    "SUMMARY:\n"
    "<3-5 paragraph synthesis of the document, roughly 1500-3000 "
    "chars total. Use blank lines between paragraphs. Cover, in "
    "order: (1) the problem the work addresses and why it matters, "
    "(2) the technical approach / methodology with enough detail "
    "to convey how it works, (3) the main results with concrete "
    "numbers or outcomes, (4) limitations or open questions, and "
    "(5) how this fits into the broader landscape. Write for a "
    "reader who will use this instead of re-reading the paper.>\n\n"
    "CLAIMS:\n"
    "- <one specific claim with numbers or concrete evidence>\n"
    "- <one specific claim with numbers or concrete evidence>\n"
    "- <aim for 5-10 claims, each standalone and specific>\n\n"
    "ENTITIES:\n"
    "<KindLabel>: <comma-separated names>\n"
    "<KindLabel>: <comma-separated names>\n"
    "Use kind labels like Models, Datasets, Methods, Authors, Tools, "
    "Organizations as relevant; be exhaustive within each kind. "
    "Omit any section that has no real content.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- No preamble, no meta-commentary, no thinking trace.\n"
    "- No markdown bold/italic on the SUMMARY/CLAIMS/ENTITIES headers.\n"
    "- Start your response directly with 'SUMMARY:'.\n"
    "- Always emit all three sections when content supports them.\n"
    "- The summary is prose (paragraphs), not a bullet list.\n\n"
    "Title: {title}\n\n"
    "Content:\n---\n{content}\n---"
)

# Matches section headers like "SUMMARY:", "**SUMMARY:**", "  Claims :",
# tolerating leading whitespace and optional markdown bold wrappers
# (including a trailing ``**`` immediately after the colon).
_SECTION_HEADER_RE = re.compile(
    r"^\s*\**\s*(SUMMARY|CLAIMS|ENTITIES)\s*\**\s*:\**",
    re.MULTILINE | re.IGNORECASE,
)


def preview(content: str) -> str:
    """The fallback summary: the first ~500 characters."""
    if not content:
        return ""
    if len(content) > _PREVIEW_CHARS:
        return content[:_PREVIEW_CHARS] + "..."
    return content


def _prompt(template: str, content: str, title: str) -> tuple[str, str]:
    capped = content[:SUMMARY_INPUT_CHAR_LIMIT]
    return capped, template.format(title=title or "(untitled)", content=capped)


async def summarize(summarizer: Any, content: str, title: str = "") -> str:
    """A 2–3 sentence summary from ``summarizer``, else the preview."""
    if not content:
        return ""
    try:
        if summarizer is not None:
            capped, prompt = _prompt(SUMMARY_PROMPT_TEMPLATE, content, title)
            summary = await summarizer.summarize(capped, prompt)
            if summary:
                return summary.strip()
    except Exception:
        logger.debug("kb_summary_generation_failed", exc_info=True)
    return preview(content)


async def sidecar_payload(summarizer: Any, content: str, title: str = "") -> dict[str, Any]:
    """The summary, claims and entities from one ``summarizer`` call.

    Falls back to ``{summary: preview(content), claims: [], entities: {}}``
    without a summarizer, when the call raises, or when the answer has none
    of the three sections.
    """
    fallback = {"summary": preview(content), "claims": [], "entities": {}}
    if not content or summarizer is None:
        return fallback
    try:
        capped, prompt = _prompt(SIDECAR_PROMPT_TEMPLATE, content, title)
        raw = await summarizer.summarize(capped, prompt)
    except Exception:
        logger.debug("kb_sidecar_payload_failed", exc_info=True)
        return fallback
    return parse_sidecar_response(raw) or fallback


def parse_sidecar_response(raw: str) -> dict[str, Any] | None:
    """Parse the SUMMARY/CLAIMS/ENTITIES blocks. Returns None on garbage.

    Sections are split by regex so multi-paragraph summaries preserve
    their blank-line breaks verbatim. CLAIMS is parsed as a bullet
    list; ENTITIES as ``Kind: comma, separated, names`` lines.
    """
    if not raw:
        return None

    matches = list(_SECTION_HEADER_RE.finditer(raw))
    if not matches:
        return None

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections[key] = raw[start:end].strip()

    summary = sections.get("summary", "")

    claims: list[str] = []
    for line in sections.get("claims", "").splitlines():
        stripped = line.strip().lstrip("-*").strip()
        if stripped:
            claims.append(stripped)

    entities: dict[str, list[str]] = {}
    for line in sections.get("entities", "").splitlines():
        stripped = line.strip().lstrip("-*").strip()
        if ":" in stripped:
            kind, _, names = stripped.partition(":")
            items = [n.strip() for n in names.split(",") if n.strip()]
            if items:
                entities[kind.strip()] = items

    if not summary and not claims and not entities:
        return None
    return {"summary": summary, "claims": claims, "entities": entities}
