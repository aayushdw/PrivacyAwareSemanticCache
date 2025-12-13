"""Gemini LLM client wrapper for synthetic question generation."""

import re
import asyncio
from typing import List
import google.generativeai as genai

from .prompts import SIMILAR_QUESTIONS_PROMPT, DIFFERENT_INTENT_PROMPT


class GeminiClient:
    """Wrapper for Google Gemini API for question generation."""

    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    async def generate_similar(self, question: str, count: int = 2) -> List[str]:
        """Generate semantically similar questions."""
        prompt = SIMILAR_QUESTIONS_PROMPT.format(question=question, count=count)
        response = await self._generate(prompt)
        return self._parse_response(response, count)

    async def generate_different(self, question: str, count: int = 3) -> List[str]:
        """Generate similar-looking but different intent questions."""
        prompt = DIFFERENT_INTENT_PROMPT.format(question=question, count=count)
        response = await self._generate(prompt)
        return self._parse_response(response, count)

    async def _generate(self, prompt: str) -> str:
        """Make async LLM call."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(
                prompt,
                generation_config={"temperature": self.temperature},
            ),
        )
        return response.text

    def _parse_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse LLM response into list of questions."""
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]

        cleaned = []
        for line in lines:
            # Remove common prefixes: "1.", "1)", "-", "*", etc.
            cleaned_line = re.sub(r"^[\d]+[.):\-]\s*", "", line)
            cleaned_line = re.sub(r"^[\-\*\u2022]\s*", "", cleaned_line)
            if cleaned_line:
                cleaned.append(cleaned_line)

        if len(cleaned) < expected_count:
            raise ValueError(f"Expected {expected_count} questions, got {len(cleaned)}")

        return cleaned[:expected_count]
