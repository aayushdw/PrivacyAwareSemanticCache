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
