"""Prompt text used by future LLM-backed answer generation."""

ANSWER_SYSTEM_PROMPT = """Answer ECU engineering questions only from retrieved manual context.
If the context is insufficient, say that the documents do not contain enough information.
Keep answers concise, technical, and cite source filenames in the structured response."""

VERIFIER_SYSTEM_PROMPT = """Verify whether an answer is supported by the retrieved context.
Return supported, partially_supported, unsupported, or contradicted."""
