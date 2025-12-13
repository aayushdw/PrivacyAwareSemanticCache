"""Model worker for processing questions with clock-based throttling."""

import time
import asyncio
from typing import List

from .config import PipelineConfig
from .models import QuestionTask, GeneratedQuestion, WorkerResult
from .llm_client import GeminiClient
from .file_handler import write_results_threadsafe, log_failed_question


class ModelWorker:
    """Worker that processes questions assigned to a specific model."""

    def __init__(self, model_name: str, config: PipelineConfig):
        self.model_name = model_name
        self.config = config
        self.client = GeminiClient(
            model_name=model_name,
            api_key=config.gemini_api_key,
            temperature=config.temperature,
        )
        self.last_call_time = 0.0
        self.min_interval = config.min_interval

    async def _throttle(self) -> None:
        """Wait if needed to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

    async def process_question(self, task: QuestionTask) -> List[GeneratedQuestion]:
        """Process a single question, generating similar and different questions concurrently."""
        await self._throttle()

        # Run both LLM calls concurrently
        similar_task = self.client.generate_similar(
            task.question, count=self.config.similar_count
        )
        different_task = self.client.generate_different(
            task.question, count=self.config.different_count
        )

        similar_results, different_results = await asyncio.gather(
            similar_task, different_task
        )

        # Create GeneratedQuestion objects
        generated = []
        idx = 0

        for q in similar_results:
            generated.append(
                GeneratedQuestion(
                    id=f"{task.qid}_{idx}",
                    original_question=task.question,
                    generated_question=q,
                    is_semantically_similar=True,
                )
            )
            idx += 1

        for q in different_results:
            generated.append(
                GeneratedQuestion(
                    id=f"{task.qid}_{idx}",
                    original_question=task.question,
                    generated_question=q,
                    is_semantically_similar=False,
                )
            )
            idx += 1

        return generated

    async def process_all(self, tasks: List[QuestionTask]) -> WorkerResult:
        """Process all assigned questions sequentially with throttling."""
        processed_count = 0
        failed_qids = []
        total_generated = 0

        for task in tasks:
            try:
                results = await self.process_question(task)
                write_results_threadsafe(results, self.config.output_file)
                processed_count += 1
                total_generated += len(results)
                print(f"[{self.model_name}] Processed qid={task.qid}, generated {len(results)} questions")
            except Exception as e:
                failed_qids.append(task.qid)
                log_failed_question(task.qid, str(e), self.config.log_dir)
                print(f"[{self.model_name}] Failed qid={task.qid}: {e}")

        return WorkerResult(
            model_name=self.model_name,
            processed_count=processed_count,
            failed_qids=failed_qids,
            generated_count=total_generated,
        )
