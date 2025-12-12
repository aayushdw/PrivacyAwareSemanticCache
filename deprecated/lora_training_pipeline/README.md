# LoRA Training Pipeline for Question Similarity

This directory contains scripts for fine-tuning MiniLM with LoRA on the Quota Question Similarity dataset.

## Overview

The pipeline consists of three main components:

1. **Baseline Evaluation** (`evaluate_roberta.py`) - Evaluates the base MiniLM model
2. **LoRA Fine-tuning** (`train_minilm_lora.py`) - Fine-tunes MiniLM with LoRA
3. **Fine-tuned Model Evaluation** (`evaluate_lora_model.py`) - Evaluates the LoRA fine-tuned model

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The pipeline uses the Quota Question Similarity dataset:
- Training data: `data/small/questions_train.csv`
- Validation data: `data/small/questions_val.csv`

Each CSV contains:
- `question1`: First question
- `question2`: Second question
- `is_duplicate`: Label (1.0 for duplicate, 0.0 for not duplicate)

## Usage

### Step 1: Evaluate Baseline Model

Evaluate the base MiniLM model without fine-tuning:

```bash
python evaluate_roberta.py
```

This will:
- Load the pre-trained MiniLM model
- Find optimal threshold on training data
- Display confusion matrix and metrics
- Evaluate on validation data
- Save visualizations to `evaluation_results/`

### Step 2: Fine-tune with LoRA

Train the model using LoRA:

```bash
python train_minilm_lora.py
```

This will:
- Load the base MiniLM model
- Apply LoRA adapters (rank=8, alpha=16)
- Train for 10 epochs on the training data
- Find optimal threshold after each epoch
- Save the best model to `models/best_model/`
- Save the final model to `models/final_model/`

**Training Configuration:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- LoRA rank: 8
- LoRA alpha: 16
- Learning rate: 2e-4
- Batch size: 32
- Epochs: 10
- Loss: Cosine Embedding Loss

### Step 3: Evaluate Fine-tuned Model

Evaluate the LoRA fine-tuned model:

```bash
python evaluate_lora_model.py
```

This will:
- Load the fine-tuned LoRA model
- Find optimal threshold on training data
- Display confusion matrix and metrics
- Evaluate on validation data
- Save visualizations to `lora_evaluation_results/`

## Output

### Model Checkpoints

- `models/best_model/` - Best model based on F1 score
- `models/final_model/` - Final model after all epochs
- `models/*/metadata.json` - Training metadata and optimal threshold

### Visualizations

**Baseline Model** (`evaluation_results/`):
- `confusion_matrix_train.png` - Training confusion matrix
- `confusion_matrix_val.png` - Validation confusion matrix
- `threshold_analysis.png` - Precision/Recall/F1 vs threshold
- `similarity_distribution_train.png` - Similarity distribution (training)
- `similarity_distribution_val.png` - Similarity distribution (validation)

**LoRA Fine-tuned Model** (`lora_evaluation_results/`):
- `confusion_matrix_train_lora.png` - Training confusion matrix
- `confusion_matrix_val_lora.png` - Validation confusion matrix
- `threshold_analysis_lora.png` - Precision/Recall/F1 vs threshold
- `similarity_distribution_train_lora.png` - Similarity distribution (training)
- `similarity_distribution_val_lora.png` - Similarity distribution (validation)

## Metrics

The evaluation scripts report:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted duplicates, how many are actually duplicates
- **Recall**: Of actual duplicates, how many are detected
- **F1 Score**: Harmonic mean of precision and recall
- **Optimal Threshold**: Threshold that maximizes F1 score

## Offline Mode

All scripts support offline mode by loading models from the local cache. This is useful when working without internet connection (e.g., on a flight). The models will be loaded from `~/.cache/huggingface/` if available.

## Notes

- The training script automatically evaluates on training data after each epoch to find the optimal threshold
- The optimal threshold is saved with the model and can be used for inference
- Adjust hyperparameters in the `main()` function of each script as needed
- GPU is automatically used if available, otherwise falls back to CPU
