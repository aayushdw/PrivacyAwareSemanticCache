"""Entry point for synthetic data generation pipeline."""

from .config import PipelineConfig
from .graph import build_pipeline_graph


def main():
    """Run the synthetic data generation pipeline."""
    config = PipelineConfig()

    print(f"\n{'='*50}")
    print("Synthetic Data Generation Pipeline")
    print(f"{'='*50}")
    print(f"Input file: {config.input_file}")
    print(f"Output file: {config.output_file}")
    print(f"Models: {config.model_list}")
    print(f"RPM limit per model: {config.llm_rpm_limit}")
    print(f"Min interval between questions: {config.min_interval}s")
    print(f"{'='*50}\n")

    # Build and run pipeline
    pipeline = build_pipeline_graph(config)

    # Initial state
    initial_state = {
        "questions": [],
        "model_assignments": {},
        "worker_results": [],
        "total_processed": 0,
        "total_failed": 0,
        "total_generated": 0,
    }

    # Execute pipeline
    pipeline.invoke(initial_state)

    print(f"\nOutput written to: {config.output_file}")


if __name__ == "__main__":
    main()
