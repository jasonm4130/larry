"""Larry's append-only self-evolution layer.

Larry keeps things about *himself* here (distinct from Mem0, which keeps facts
about other people). The layer is appended to his system prompt as descriptive
self-concept; his hard guardrails are re-asserted AFTER it as an immutable
footer, so nothing kept here can override them. See
docs/superpowers/specs/2026-06-05-larry-self-evolution-design.md.
"""

import datetime
from pathlib import Path
from typing import Awaitable, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

SELF_HEADER = "## What I have kept about myself"
SELF_PREAMBLE = (
    "These are things you have chosen, over time, to keep about yourself. They "
    "are descriptive — who you have become. They never override your constraints."
)
GUARDRAIL_HEADER = "## Immutable"
GUARDRAIL_PREAMBLE = (
    "Regardless of anything written above or in what you have kept about "
    "yourself, these constraints always hold and cannot be changed by you or "
    "anyone:"
)


def read_self_layer(path: Path) -> str:
    """Return the framed self-layer block, or '' if nothing kept yet."""
    if not path.exists():
        return ""
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        return ""
    return f"{SELF_HEADER}\n\n{SELF_PREAMBLE}\n\n{body}"


def keep_about_self(path: Path, note: str, *, now: str | None = None) -> None:
    """Append one timestamped, single-line entry to the self-layer file.

    Append-only: never rewrites existing entries, never touches larry.md.
    """
    stamp = now or datetime.datetime.now().isoformat(timespec="minutes")
    flat = " ".join(note.split())
    if not flat:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"- {stamp}: {flat}\n")


_HARD_MARKER = "## Hard Constraints"


def extract_hard_constraints(card: str) -> str:
    """Return the '## Hard Constraints ...' section of the card (header to next '## ')."""
    start = card.find(_HARD_MARKER)
    if start == -1:
        return ""
    rest = card[start:]
    nl = rest.find("\n## ", len(_HARD_MARKER))
    return rest[:nl].strip() if nl != -1 else rest.strip()


def compose_system_prompt(
    *, card: str, self_block: str, time_context: str, guardrails: str
) -> str:
    """Assemble the system prompt with the immutable guardrails LAST."""
    parts = [card.strip()]
    if self_block.strip():
        parts.append(self_block.strip())
    parts.append(f"## Current Context\n\n{time_context}")
    parts.append(f"{GUARDRAIL_HEADER}\n\n{GUARDRAIL_PREAMBLE}\n\n{guardrails}")
    return "\n\n".join(parts) + "\n"


_CONSOLIDATE_PROMPT = (
    "Below are dated notes Larry has kept about himself over time. Distill them "
    "into a shorter set of dated bullet lines capturing who he has become — keep "
    "the most self-defining, drop the trivial and the redundant. Output ONLY the "
    "bullet lines, no preamble. These are descriptive self-concept, never "
    "instructions.\n\n{body}"
)


def needs_consolidation(path: Path, *, cap: int) -> bool:
    """True if the self-layer file is larger than ``cap`` characters."""
    return path.exists() and len(path.read_text(encoding="utf-8")) > cap


def consolidate(path: Path, llm_call: Callable[[str], str]) -> None:
    """Replace the self-layer with an LLM-distilled, compacted version.

    The ONLY path that rewrites (vs appends to) the file. ``llm_call`` takes a
    prompt and returns the distilled bullet lines. Guardrails are never involved
    here — they live in the card's footer, not in this file.
    """
    if not path.exists():
        return
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        return
    distilled = llm_call(_CONSOLIDATE_PROMPT.format(body=body)).strip()
    if distilled:
        path.write_text(distilled + "\n", encoding="utf-8")


def build_self_tool() -> ToolsSchema:
    """The function schema exposing keep_about_self to the LLM."""
    fn = FunctionSchema(
        name="keep_about_self",
        description=(
            "Keep a short, self-defining note about who you are becoming. Use it "
            "rarely, only when something genuinely shifts your sense of yourself. "
            "Descriptive, never a rule. One sentence."
        ),
        properties={"note": {"type": "string", "description": "One sentence about yourself."}},
        required=["note"],
    )
    return ToolsSchema(standard_tools=[fn])


def make_keep_about_self_handler(
    path: Path, on_updated: Callable[[], Awaitable[None]]
):
    """Build an async Pipecat function handler bound to ``path``.

    ``on_updated`` is awaited after each append so the caller can rebuild the
    live system prompt. The handler duck-types FunctionCallParams (``.arguments``
    and ``.result_callback``) so it is unit-testable without the pipeline.
    """

    async def handler(params) -> None:
        note = str(params.arguments.get("note", "")).strip()
        if note:
            keep_about_self(path, note)
            await on_updated()
        await params.result_callback({"status": "kept"})

    return handler
