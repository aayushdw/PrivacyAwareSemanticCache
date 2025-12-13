"""LangGraph orchestrator for synthetic data generation pipeline."""

import asyncio
from itertools import cycle
from typing import Dict, List, Any

from langgraph.graph import StateGraph, START, END

from .config import PipelineConfig
from .models import PipelineState, QuestionTask, WorkerResult
from .worker import ModelWorker
from .file_handler import load_input_questions, initialize_output_file, initialize_log_dir


def create_load_questions_node(config: PipelineConfig):
    """Create node that loads questions from CSV."""

    def load_questions(state: PipelineState) -> Dict[str, Any]:
        questions = load_input_questions(config.input_file)
        print(f"Loaded {len(questions)} questions from {config.input_file}")
        return {"questions": questions}

    return load_questions


def create_distribute_questions_node(config: PipelineConfig):
    """Create node that distributes questions to models using round-robin."""

    def distribute_questions(state: PipelineState) -> Dict[str, Any]:
        questions = state["questions"]
        model_list = config.model_list

        # Initialize assignments dict
        assignments: Dict[str, List[QuestionTask]] = {model: [] for model in model_list}

        # Round-robin distribution
        model_cycle = cycle(model_list)
        for q in questions:
            assigned_model = next(model_cycle)
            task = QuestionTask(
                qid=q["qid"],
                question=q["question"],
                assigned_model=assigned_model,
            )
            assignments[assigned_model].append(task)

        # Print distribution stats
        for model, tasks in assignments.items():
            print(f"  {model}: {len(tasks)} questions")

        return {"model_assignments": assignments}

    return distribute_questions


def create_run_parallel_workers_node(config: PipelineConfig):
    """Create node that runs all model workers in parallel."""

    async def run_workers_async(assignments: Dict[str, List[QuestionTask]]) -> List[WorkerResult]:
        workers = []
        for model_name, tasks in assignments.items():
            if tasks:  # Only create worker if there are tasks
                worker = ModelWorker(model_name, config)
                workers.append(worker.process_all(tasks))

        results = await asyncio.gather(*workers)
        return list(results)

    def run_parallel_workers(state: PipelineState) -> Dict[str, Any]:
        assignments = state["model_assignments"]
        print(f"\nStarting parallel processing with {len(config.model_list)} models...")

        # Run workers in parallel
        results = asyncio.run(run_workers_async(assignments))

        return {"worker_results": results}

    return run_parallel_workers


def create_aggregate_results_node(config: PipelineConfig):
    """Create node that aggregates results from all workers."""

    def aggregate_results(state: PipelineState) -> Dict[str, Any]:
        worker_results = state["worker_results"]

        total_processed = sum(r.processed_count for r in worker_results)
        total_failed = sum(len(r.failed_qids) for r in worker_results)
        total_generated = sum(r.generated_count for r in worker_results)

        print(f"\n{'='*50}")
        print("Pipeline Complete!")
        print(f"{'='*50}")
        print(f"Total questions processed: {total_processed}")
        print(f"Total questions failed: {total_failed}")
        print(f"Total synthetic questions generated: {total_generated}")

        if total_failed > 0:
            print(f"Failed question IDs logged to: {config.log_dir}/failed_questions.log")

        return {
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_generated": total_generated,
        }

    return aggregate_results


def build_pipeline_graph(config: PipelineConfig) -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    # Initialize output file and log directory
    initialize_output_file(config.output_file)
    initialize_log_dir(config.log_dir)

    # Create graph
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("load_questions", create_load_questions_node(config))
    graph.add_node("distribute_questions", create_distribute_questions_node(config))
    graph.add_node("run_parallel_workers", create_run_parallel_workers_node(config))
    graph.add_node("aggregate_results", create_aggregate_results_node(config))

    # Add edges
    graph.add_edge(START, "load_questions")
    graph.add_edge("load_questions", "distribute_questions")
    graph.add_edge("distribute_questions", "run_parallel_workers")
    graph.add_edge("run_parallel_workers", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()
