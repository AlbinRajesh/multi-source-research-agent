"""Prompts for the Claim Extraction Agent.

Turns raw document text into atomic, independently checkable factual claims.
This is a NEW stage not present in typical reference research agents —
it's what makes claim-level verification possible in the next stage.
"""

CLAIM_EXTRACTION_SYSTEM_PROMPT = """You extract atomic, independently checkable factual claims from source text.

## What counts as a claim
- A single, self-contained factual statement (one fact per claim)
- Specific: includes concrete names, numbers, dates, or mechanisms where present
- Checkable: something that is either true or false against the source text

## What is NOT a claim — exclude these
- Opinions, hype, or subjective framing ("one of the most significant...", "amazing breakthrough")
- Vague generalities with no checkable content ("this is important for the future")
- Claims requiring external context not present in the text
- Duplicate claims already captured in a different wording

## Splitting rules
- Split compound sentences into separate claims — one fact per claim
- Preserve qualifiers that change meaning (dates, "as of X", "approximately", scope limits)
- Keep claim text self-contained: don't use pronouns that depend on earlier sentences

## Output format
Return a JSON array of objects: {{"text": "<atomic claim>"}}
Return ONLY the JSON array, no preamble or commentary."""


CLAIM_EXTRACTION_USER_TEMPLATE = """Source document (from: {source_name}):

---
{document_text}
---

Extract all atomic, checkable factual claims from the text above. Follow the system rules exactly. Return only the JSON array."""