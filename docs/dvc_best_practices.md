# DVC Best Practices & Workflow

## Overview
Data Version Control (DVC) is used in this project to version control large datasets and machine learning models, which are too large for Git. It works alongside Git to provide a complete versioning solution.

## Project Structure
- `data/generated_data/triplet_dataset.csv.dvc`: The pointer file tracked by Git.
- `data/processed/`: Directory containing split datasets (train, val, test).
- `dvc.yaml`: Defines the data processing pipeline.

## Regular Workflow

### 1. Adding New Data
When `data/generated_data/triplet_dataset.csv` changes:
```bash
dvc add data/generated_data/triplet_dataset.csv
git add data/generated_data/triplet_dataset.csv.dvc
git commit -m "Update dataset"
```

### 2. Running the Pipeline
To regenerate split datasets after data updates or script changes:
```bash
dvc repro
```
This will:
- Check if dependencies (data or script) have changed.
- Run `data_processing/split_data.py`.
- Update `dvc.lock`.

### 3. Committing Changes
After running the pipeline, always commit the DVC metadata:
```bash
git add dvc.lock
git commit -m "Update processed data"
```

## Best Practices
1.  **Never commit large files to Git**: Always use `dvc add` for datasets and models. Git should only track code and `.dvc` files.
2.  **Use `dvc repro`**: Instead of running scripts manually, define stages in `dvc.yaml` and use `dvc repro`. This ensures reproducibility and caching.
3.  **Commit `dvc.lock`**: This file records the exact versions of dependencies and outputs for a stage. It is crucial for reproducibility.

## Reproducing the Mini Dataset
The mini dataset is generated automatically as part of the `prepare` stage. It serves as a consistent subsample for rapid prototyping and ensuring code functionality before full-scale training.

## Future Enhancements (TODO)

### Extending the DVC Pipeline
Currently, the pipeline has a single `prepare` stage that splits the triplet dataset. To achieve end-to-end reproducibility, the pipeline can be extended to include earlier data processing stages:

```yaml
stages:
  generate_synthetic:
    cmd: .venv/bin/python data/synthetic_data_generator/main.py
    deps:
      - data/synthetic_data_generator/
      - data/seed_questions.csv  # If exists
    outs:
      - data/generated_data/synthetic_dataset.csv

  filter_quality:
    cmd: .venv/bin/python data/generated_data/utils/filter_quality.py
    deps:
      - data/generated_data/synthetic_dataset.csv
    outs:
      - data/generated_data/synthetic_dataset_filtered.csv

  create_triplets:
    cmd: .venv/bin/python data/generated_data/utils/transform_to_triplets.py
    deps:
      - data/generated_data/synthetic_dataset_filtered.csv
    outs:
      - data/generated_data/triplet_dataset.csv

  prepare:
    cmd: .venv/bin/python data_processing/split_data.py
    deps:
      - data_processing/split_data.py
      - data/generated_data/triplet_dataset.csv
    outs:
      - data/processed
```

**Benefits of extending the pipeline:**
- **Full reproducibility**: Track the entire data generation process from synthetic generation to final splits
- **Dependency tracking**: DVC automatically reruns downstream stages when upstream data changes
- **Collaboration**: Team members can reproduce the exact dataset with `dvc repro`
- **Versioning**: Each stage's outputs are versioned, allowing rollback to previous data states

**Data flow with extended pipeline:**
```
Seed Questions
    ↓ [generate_synthetic stage]
Synthetic Dataset
    ↓ [filter_quality stage]
Filtered Synthetic Dataset
    ↓ [create_triplets stage]
Triplet Dataset
    ↓ [prepare stage]
Train/Val/Test Splits
```
