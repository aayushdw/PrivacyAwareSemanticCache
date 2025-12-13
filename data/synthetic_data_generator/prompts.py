"""Prompt templates for synthetic question generation."""

SIMILAR_QUESTIONS_PROMPT = """Generate exactly {count} semantically similar questions that convey the SAME intent and would expect the SAME answer as the original question.

Requirements:
- Preserve the exact meaning and intent
- Use different words and sentence structure
- Keep the same level of formality
- Do not add or remove information

Original Question: {question}

Output exactly {count} questions, one per line, without numbering or explanation:"""


DIFFERENT_INTENT_PROMPT = """Generate exactly {count} questions that look similar to the original (share some keywords or topic) but have DIFFERENT intent and would require DIFFERENT answers.

Requirements:
- Questions should share some vocabulary with the original
- Questions must have clearly different intent/meaning
- Questions should be realistic and coherent
- Avoid trivially different questions (like just negating)

Original Question: {question}

Output exactly {count} questions, one per line, without numbering or explanation:"""
