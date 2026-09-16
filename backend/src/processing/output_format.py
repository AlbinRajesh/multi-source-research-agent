"""Shared parsing and validation for user-requested answer formats."""
import re
from typing import Any, Dict, List, Optional

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUMBER = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_COMPARE_KEYWORD = r"(?:compare|comparison(?:\s+of)?|difference(?:s)?\s+between)"
_VS_SEPARATOR = r"(?:vs\.?|versus)"


def _to_number(value: str) -> int:
    word_value = _WORD_NUMBERS.get(value.lower())
    return word_value if word_value is not None else int(value)


def _extract_comparison_entities(text: str) -> List[str]:
    """Best-effort extraction of the items being compared, so the
    synthesizer can be told explicitly which entities must each get their
    own section. This is heuristic, not guaranteed — if it can't confidently
    find 2+ distinct entities, it returns [] and the caller falls back to
    letting the LLM infer entities from the raw query, same as before this
    change. Never silently invents entities."""
    query = text.strip()

    candidate = None
    m = re.search(rf"{_COMPARE_KEYWORD}\s+(.+)", query, re.IGNORECASE)
    if m:
        candidate = m.group(1)
    else:
        m2 = re.search(rf"(.+?)\s+{_VS_SEPARATOR}\s+(.+)", query, re.IGNORECASE)
        if m2:
            candidate = f"{m2.group(1)} and {m2.group(2)}"

    if not candidate:
        return []

    # Strip trailing instruction fragments like "in 3 points each"
    candidate = re.split(
        rf"\b(?:in|with|using)\s+{_NUMBER}\b", candidate, flags=re.IGNORECASE
    )[0]

    parts = re.split(
        r"\s*(?:,|\band\b|\bvs\.?\b|\bversus\b)\s*", candidate, flags=re.IGNORECASE
    )
    entities = [p.strip(" .?!") for p in parts if p.strip(" .?!")]
    entities = [e for e in entities if 0 < len(e) <= 60]

    if len(entities) < 2:
        return []
    return entities


def parse_output_format(text: str) -> Dict[str, Any]:
    """Extract explicit answer-shape requirements without using an LLM."""
    query = text.strip().lower()

    comparison = bool(re.search(
        r"\b(?:compare|comparison|versus|vs\.?|difference(?:s)? between)\b",
        query,
    ))
    each = bool(re.search(r"\b(?:each|per\s+(?:item|entity|side))\b", query))

    match = re.search(
        rf"\b(?:in|with|using|exactly|give me)?\s*({_NUMBER})\s+"
        r"(?:bullet\s*)?(?:points?|takeaways?|key\s+points?)\b",
        query,
    )
    if match:
        count = _to_number(match.group(1))
        per_entity = comparison and each
        entities = _extract_comparison_entities(text) if per_entity else []

        if per_entity and entities:
            entity_list_str = ", ".join(entities)
            instruction = (
                f"This is a comparison between exactly these items: "
                f"{entity_list_str}. Do NOT use a markdown table for this "
                f"response. Instead, write one clearly labeled subsection "
                f"per item ({entity_list_str}), and give exactly {count} "
                f"bullet points under each. Every listed item must appear "
                f"— do not omit any of them."
            )
        elif per_entity:
            instruction = f"Use exactly {count} points for each compared item."
        else:
            instruction = f"Use exactly {count} bullet points."

        return {
            "style": "comparison" if comparison else "bullets",
            "count": count,
            "per_entity": per_entity,
            "entities": entities,
            "exact": True,
            "instruction": instruction,
        }

    match = re.search(rf"\b(?:in|with|using|exactly)\s+({_NUMBER})\s+sentences?\b", query)
    if match:
        count = _to_number(match.group(1))
        return {
            "style": "sentences", "count": count, "exact": True,
            "instruction": f"Write exactly {count} sentence(s), no more and no less.",
        }

    match = re.search(
        r"\b(?:under|below|within|in|with|using|exactly)\s+(\d+)\s+words?\b",
        query,
    )
    if match:
        count = int(match.group(1))
        return {
            "style": "words", "max_words": count, "exact": False,
            "instruction": f"Use at most {count} words.",
        }

    if re.search(r"\b(?:one|a single|1)\s+paragraph\b", query):
        return {
            "style": "paragraph", "exact": True,
            "instruction": "Write exactly one paragraph.",
        }

    if re.search(r"\b(?:one[- ]liner|tl;?dr|very short|super short)\b", query):
        return {
            "style": "sentences", "max_sentences": 2, "exact": False,
            "instruction": "Write no more than two sentences.",
        }

    if re.search(r"\b(?:brief|short|concise|quick)\b", query):
        return {
            "style": "sentences", "max_sentences": 5, "exact": False,
            "instruction": "Write a brief answer of no more than five sentences.",
        }

    if re.search(r"\b(?:detailed|long|comprehensive|in-depth|thorough)\b", query):
        return {
            "style": "paragraph", "exact": False,
            "instruction": "Write a detailed answer covering the major points.",
        }

    return {"style": "default", "exact": False, "instruction": "Use a clear answer format."}


def format_instruction(output_format: Dict[str, Any]) -> str:
    return output_format.get("instruction", "Use a clear answer format.")


def _enforce_per_entity_format(
    answer: str, entities: List[str], expected: Optional[int]
) -> Dict[str, Any]:
    """Validate (and lightly repair) a comparison answer that should have
    one section per entity, each with `expected` bullet points.

    Never fabricates missing content — a missing entity or a short section
    is reported as invalid/missing rather than papered over, so upstream
    logging/retry logic can see the real state instead of a false pass.
    """
    per_entity_counts: Dict[str, int] = {}
    missing_entities: List[str] = []
    repaired = False
    sections: Dict[str, str] = {}

    for entity in entities:
        heading_pattern = re.compile(rf"(?im)^.*\b{re.escape(entity)}\b.*$")
        match = heading_pattern.search(answer)
        if not match:
            missing_entities.append(entity)
            per_entity_counts[entity] = 0
            continue

        start = match.end()
        next_positions = []
        for other in entities:
            if other == entity:
                continue
            other_match = re.search(
                rf"(?im)^.*\b{re.escape(other)}\b.*$", answer[start:]
            )
            if other_match:
                next_positions.append(start + other_match.start())
        end = min(next_positions) if next_positions else len(answer)

        section = answer[start:end]
        bullet_lines = [
            line.strip() for line in section.splitlines()
            if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line)
        ]

        if expected and len(bullet_lines) > expected:
            bullet_lines = bullet_lines[:expected]
            repaired = True

        sections[entity] = "\n".join(bullet_lines)
        per_entity_counts[entity] = len(bullet_lines)

    valid = (
        not missing_entities
        and (expected is None or all(c == expected for c in per_entity_counts.values()))
    )

    if repaired:
        rebuilt_parts = []
        for entity in entities:
            if entity in sections:
                rebuilt_parts.append(f"### {entity}\n{sections[entity]}")
        answer = "\n\n".join(rebuilt_parts) if rebuilt_parts else answer

    return {
        "text": answer,
        "valid": valid,
        "repaired": repaired,
        "style": "comparison",
        "requested_count": expected,
        "actual_count": None,
        "per_entity_counts": per_entity_counts,
        "missing_entities": missing_entities,
    }


def enforce_output_format(text: str, output_format: Dict[str, Any]) -> Dict[str, Any]:
    """Apply safe deterministic limits and report whether the shape is valid."""
    answer = (text or "").strip()
    style = output_format.get("style", "default")
    expected = output_format.get("count")
    per_entity = output_format.get("per_entity", False)
    entities = output_format.get("entities") or []
    repaired = False

    if style == "words" and output_format.get("max_words"):
        words = answer.split()
        if len(words) > output_format["max_words"]:
            answer = " ".join(words[:output_format["max_words"]]).rstrip(" .,;:") + "..."
            repaired = True

    if style == "sentences":
        maximum = output_format.get("max_sentences") or expected
        if maximum:
            parts = re.split(r"(?<=[.!?])\s+", answer)
            if len(parts) > maximum:
                answer = " ".join(parts[:maximum]).strip()
                repaired = True

    if style in {"bullets", "comparison"} and per_entity and entities:
        return _enforce_per_entity_format(answer, entities, expected)

    if style in {"bullets", "comparison"}:
        lines = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            for line in answer.splitlines()
        ]
        lines = [line for line in lines if line]
        if len(lines) == 1 and expected and expected > 1:
            lines = [p.strip() for p in re.split(r"(?<=[.!?])\s+", lines[0]) if p.strip()]
        if expected and len(lines) > expected and not output_format.get("per_entity"):
            lines = lines[:expected]
            repaired = True
        answer = "\n".join(f"- {line}" for line in lines)

    actual_lines = [line for line in answer.splitlines() if line.strip()]
    if style in {"bullets", "comparison"} and expected:
        valid = len(actual_lines) == expected
    elif style == "sentences" and expected:
        valid = len(re.split(r"(?<=[.!?])\s+", answer)) == expected
    elif style == "words" and output_format.get("max_words"):
        valid = len(answer.split()) <= output_format["max_words"]
    else:
        valid = True

    return {
        "text": answer,
        "valid": valid,
        "repaired": repaired,
        "style": style,
        "requested_count": expected,
        "actual_count": len(actual_lines) if style in {"bullets", "comparison"} else None,
    }