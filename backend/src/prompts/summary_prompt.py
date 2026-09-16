"""Prompts for the Summarizer Agent."""

SINGLE_PASS_PROMPT = """Summarize the following document.

Length/format requirement: {length_instruction}

Rules:
- Use only information in the document. Do not add outside facts.
- Do not mention "the document" or "the text" — write as if describing the subject directly.
- Follow the length/format requirement exactly.

Document:
{document}

Summary:"""


MAP_PROMPT = """Summarize the key points of this section in 3-5 sentences. Use only information present in the text — do not add outside facts.

Section:
{section}

Summary:"""


REDUCE_PROMPT = """You are given summaries of consecutive sections of one document, in order. Combine them into a single coherent summary of the whole document.

Length/format requirement: {length_instruction}

Rules:
- Remove redundancy between sections.
- Preserve the most important facts and figures.
- Do not mention "the sections" or "the summaries" — write as if describing the document directly.
- Follow the length/format requirement exactly.

Section summaries:
{partial_summaries}

Final summary:"""