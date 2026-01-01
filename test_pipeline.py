import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from embedding_pipeline.flows.main_flow import quick_test_pipeline

if __name__ == "__main__":
    print("Running quick test pipeline...")
    try:
        result = quick_test_pipeline()
        print("\nSUCCESS: Pipeline finished without errors.")
        print(f"Summary: {result.get('summary')}")
    except Exception as e:
        print(f"\nFAILURE: Pipeline failed with error: {e}")
        sys.exit(1)
