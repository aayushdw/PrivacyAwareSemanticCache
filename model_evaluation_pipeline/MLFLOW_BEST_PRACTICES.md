# MLflow Best Practices Guide for Embedding Model Evaluation

**Last Updated:** December 7, 2025
**Project:** Privacy-Aware Semantic Cache
**Purpose:** Reference guide for industry-standard MLflow usage in ML experimentation

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [MLflow Best Practices](#2-mlflow-best-practices)
3. [Recommended Implementation Roadmap](#3-recommended-implementation-roadmap)
4. [Specific Recommendations for 20-Model Evaluation](#4-specific-recommendations-for-20-model-evaluation)
5. [Key Takeaways](#5-key-takeaways)
6. [Additional Resources](#6-additional-resources)

---

## 1. Current State Analysis

### What You're Doing Well ✓

1. **Proper Experiment Organization**: Using named experiments (`Semantic_Cache_Evaluation_v2`)
2. **Comprehensive Metrics Tracking**: Logging 15+ metrics covering performance, latency, and classification quality
3. **Artifact Logging**: Generating and storing visualization plots (PR curves, ROC curves, confusion matrices)
4. **Infrastructure**: Self-hosted MLflow with proper backend (SQLite) and artifact storage
5. **Integration**: Using Prefect for orchestration alongside MLflow for tracking

### Current Gaps

1. **No Model Comparison**: Can't easily compare multiple models side-by-side
2. **No Model Registry**: Models aren't versioned or promoted for production
3. **Missing Tags**: No way to categorize/filter runs (by model family, dataset size, etc.)
4. **No Interactive Dashboard**: Must use MLflow UI, which has limited comparison features
5. **Limited Metadata**: No run descriptions, dataset versioning, or code snapshots
6. **Manual Model Selection**: No automated "best model" identification based on your criteria

---

## 2. MLflow Best Practices

### 2.1 Experiment Organization Strategy

**Corporate Pattern: Hierarchical Experiments**

```
Semantic_Cache_Evaluation/
├── v1_baseline_models/          # Initial model screening
├── v2_optimized_threshold/      # Threshold tuning experiments
├── v3_production_candidates/    # Final model selection
└── v4_ablation_studies/         # Model architecture analysis
```

**Why This Matters:**
- Keeps experiments organized by phase
- Makes it easy to track evolution over time
- Allows team members to understand project timeline

**For Your Use Case:**
- Create separate experiments for different dataset sizes (small/medium/large)
- Use experiment names to indicate evaluation goals
- Archive old experiments when no longer relevant

### 2.2 Run Tagging and Metadata

**Corporate Pattern: Rich Metadata**

Every MLflow run should include:

#### 1. Tags (for filtering/grouping):

```python
mlflow.set_tag("model_family", "sentence-transformers")
mlflow.set_tag("model_size", "small")  # small/medium/large
mlflow.set_tag("dataset_size", "1000")
mlflow.set_tag("evaluation_phase", "initial_screening")
mlflow.set_tag("candidate_for_production", "true")
```

#### 2. Run Description (for context):

```python
mlflow.set_tag("mlflow.note.content",
    "Evaluating all-MiniLM-L6-v2 on small dataset. "
    "This model is a candidate for mobile deployment due to small size.")
```

#### 3. Git Information (for reproducibility):

```python
import subprocess

mlflow.set_tag("git_commit", subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip())
mlflow.set_tag("git_branch", subprocess.check_output(['git', 'branch', '--show-current']).decode().strip())
```

**Why This Matters:**
- Enables powerful filtering in MLflow UI
- Provides context months later when reviewing experiments
- Essential for reproducibility and debugging

### 2.3 Model Registry for Production Management

**Corporate Pattern: Model Lifecycle Stages**

```
Development → Staging → Production → Archived
```

**How It Works:**

#### 1. Register Model After Training:

```python
# Log the model
mlflow.sentence_transformers.log_model(
    model,
    artifact_path="model",
    registered_model_name="semantic-cache-embedder"
)
```

#### 2. Promote Models Through Stages:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Promote to staging
client.transition_model_version_stage(
    name="semantic-cache-embedder",
    version=3,
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="semantic-cache-embedder",
    version=3,
    stage="Production"
)
```

#### 3. Load Production Model in Application:

```python
# Always loads the current production model
model = mlflow.sentence_transformers.load_model(
    "models:/semantic-cache-embedder/Production"
)
```

**Why This Matters:**
- Decouples model selection from application code
- Enables A/B testing (run Staging and Production in parallel)
- Provides audit trail of model changes
- Allows rollback by simply changing stage assignment

**For Your Use Case:**
- Register top 3 models from your evaluation
- Put best model in "Staging" for testing
- Only promote to "Production" after validation
- Keep previous production model as fallback

### 2.4 Multi-Objective Model Selection

**Your Criteria:**
1. Primary: Maximize recall at minimum precision threshold (0.75-0.80)
2. Secondary: Minimize latency
3. Tertiary: Minimize model size

**Corporate Pattern: Composite Scoring Function**

```python
def calculate_model_score(metrics, weights):
    """
    Calculate weighted score for model ranking.

    Higher recall_at_target is better (maximize)
    Lower latency is better (minimize, so we invert)
    Lower model_size is better (minimize, so we invert)
    """
    # Normalize metrics to 0-1 scale
    recall_score = metrics['recall_at_target']  # Already 0-1

    # Invert latency (lower is better)
    # Assume max latency is 100ms for normalization
    latency_score = 1 - (metrics['latency_ms_per_pair'] / 100.0)
    latency_score = max(0, min(1, latency_score))

    # Invert model size (lower is better)
    # Assume max size is 1000MB
    size_score = 1 - (metrics['model_size_mb'] / 1000.0)
    size_score = max(0, min(1, size_score))

    # Weighted combination
    composite_score = (
        weights['recall'] * recall_score +
        weights['latency'] * latency_score +
        weights['size'] * size_score
    )

    return composite_score

# Log composite score in pipeline
weights = {'recall': 0.60, 'latency': 0.25, 'size': 0.15}
composite_score = calculate_model_score(metrics, weights)
mlflow.log_metric("composite_score", composite_score)
mlflow.log_param("score_weights", str(weights))
```

**Why This Matters:**
- Enables objective model ranking
- Makes tradeoffs explicit and tunable
- Can adjust weights based on deployment context (mobile vs server)

### 2.5 Interactive Model Comparison Dashboard

**Corporate Standard: MLflow UI + Custom Dashboard**

MLflow provides built-in comparison features, but corporations typically augment with custom dashboards.

#### Option 1: MLflow UI Native Comparison (Free, Built-in)

**Steps:**
1. Navigate to your experiment in MLflow UI (http://localhost:5001)
2. Select multiple runs (checkbox on left)
3. Click "Compare" button
4. View side-by-side:
   - Parallel coordinates plot (see metric tradeoffs)
   - Scatter plots (latency vs recall)
   - Difference view (metric delta between runs)

#### Option 2: Custom Streamlit Dashboard (Recommended for Corporations)

```python
# dashboard.py
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px

st.title("Embedding Model Comparison Dashboard")

# Fetch all runs from experiment
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Semantic_Cache_Evaluation_v2")
runs = client.search_runs(experiment.experiment_id)

# Convert to DataFrame
df = pd.DataFrame([
    {
        'model': run.data.params.get('model_name'),
        'recall': run.data.metrics.get('recall_at_target_precision'),
        'latency_ms': run.data.metrics.get('latency_ms_per_pair'),
        'model_size_mb': run.data.metrics.get('model_size_mb'),
        'f1_score': run.data.metrics.get('f1_score'),
        'run_id': run.info.run_id
    }
    for run in runs
])

# Interactive scatter plot: Recall vs Latency (sized by model size)
fig = px.scatter(df,
    x='latency_ms',
    y='recall',
    size='model_size_mb',
    color='model_size_mb',
    hover_data=['model', 'f1_score'],
    title='Model Performance: Recall vs Latency (bubble size = model size)'
)
st.plotly_chart(fig)

# Model comparison table
st.dataframe(df.sort_values('recall', ascending=False))

# Side-by-side model comparison
col1, col2 = st.columns(2)
model1 = col1.selectbox("Model 1", df['model'].unique())
model2 = col2.selectbox("Model 2", df['model'].unique())

# Display artifacts (plots) side by side
run1 = df[df['model'] == model1].iloc[0]['run_id']
run2 = df[df['model'] == model2].iloc[0]['run_id']

# Download and display PR curves
with col1:
    st.image(mlflow.artifacts.download_artifacts(f"runs:/{run1}/plots/pr_curve.png"))
with col2:
    st.image(mlflow.artifacts.download_artifacts(f"runs:/{run2}/plots/pr_curve.png"))
```

**Deployment:**
```bash
streamlit run dashboard.py
# Access at http://localhost:8501
```

**Why This Matters:**
- Interactive exploration of model tradeoffs
- Easy to share with stakeholders (non-technical users)
- Can embed business logic (cost calculations, deployment constraints)
- Extensible for future needs

### 2.6 Dataset Versioning and Reproducibility

**Corporate Pattern: DVC + MLflow Integration**

You already have DVC configured! Corporations use DVC for dataset versioning alongside MLflow for experiment tracking.

**Best Practice Workflow:**

#### 1. Version Your Dataset:

```bash
dvc add data/questions_large.csv
git add data/questions_large.csv.dvc
git commit -m "Add large evaluation dataset v2"
git tag dataset-v2
```

#### 2. Log Dataset Version in MLflow:

```python
# Get current dataset version from DVC
import subprocess
dataset_hash = subprocess.check_output(
    ['dvc', 'get-url', 'data/questions_large.csv', '--show-url']
).decode().strip()

mlflow.log_param("dataset_version", "v2")
mlflow.log_param("dataset_dvc_hash", dataset_hash)
mlflow.log_param("dataset_rows", len(labels))
```

#### 3. Log Dataset as Artifact (Small Files Only):

```python
# For small datasets, log directly to MLflow
mlflow.log_artifact(dataset_path, "dataset")
```

**Why This Matters:**
- Complete reproducibility (can recreate exact run months later)
- Track when dataset changes affect model performance
- Essential for debugging distribution shift issues

### 2.7 Automated Model Selection and Reporting

**Corporate Pattern: Post-Evaluation Analysis Script**

```python
# select_best_model.py
from mlflow.tracking import MlflowClient
import pandas as pd

def select_best_models(experiment_name, min_precision=0.75, top_k=3):
    """
    Automatically select top K models based on composite score.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string=f"metrics.precision_at_target >= {min_precision}",
        order_by=["metrics.composite_score DESC"],
        max_results=top_k
    )

    results = []
    for run in runs:
        results.append({
            'model': run.data.params['model_name'],
            'recall': run.data.metrics['recall_at_target_precision'],
            'latency_ms': run.data.metrics['latency_ms_per_pair'],
            'model_size_mb': run.data.metrics['model_size_mb'],
            'composite_score': run.data.metrics['composite_score'],
            'run_id': run.info.run_id
        })

    df = pd.DataFrame(results)

    # Generate markdown report
    report = f"""
# Model Selection Report

**Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Minimum Precision Threshold:** {min_precision}
**Models Evaluated:** {len(runs)} models

## Top {top_k} Models

{df.to_markdown(index=False)}

## Recommendation

**Selected Model:** {df.iloc[0]['model']}
**Rationale:** Highest composite score balancing recall ({df.iloc[0]['recall']:.3f}),
latency ({df.iloc[0]['latency_ms']:.2f}ms), and model size ({df.iloc[0]['model_size_mb']:.1f}MB).

## Next Steps

1. Register model in MLflow Model Registry
2. Deploy to staging environment for integration testing
3. Run A/B test against current production model
4. Monitor performance metrics in production
"""

    # Save report as MLflow artifact
    with open("model_selection_report.md", "w") as f:
        f.write(report)

    return df

# Run after all models evaluated
best_models = select_best_models("Semantic_Cache_Evaluation_v2", min_precision=0.8)
print(best_models)
```

**Why This Matters:**
- Removes human bias from model selection
- Documents decision rationale
- Repeatable process for future evaluations

### 2.8 CI/CD Integration Pattern

**Corporate Standard: Automated Model Evaluation Pipeline**

```yaml
# .github/workflows/model-evaluation.yml
name: Model Evaluation Pipeline

on:
  push:
    paths:
      - 'model_evaluation_pipeline/**'
      - 'data/**'
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday

jobs:
  evaluate-models:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Pull dataset from DVC
      run: |
        dvc pull data/questions_large.csv.dvc

    - name: Run MLflow server
      run: |
        mlflow server --backend-store-uri sqlite:///mlflow.db \
                      --default-artifact-root ./artifacts \
                      --port 5001 &
        sleep 10

    - name: Evaluate models
      env:
        MLFLOW_TRACKING_URI: http://localhost:5001
      run: |
        python model_evaluation_pipeline/src/pipeline.py

    - name: Select best model
      run: |
        python scripts/select_best_model.py

    - name: Register model in registry
      if: github.ref == 'refs/heads/main'
      run: |
        python scripts/register_model.py

    - name: Upload MLflow artifacts
      uses: actions/upload-artifact@v3
      with:
        name: mlflow-results
        path: |
          model_selection_report.md
          artifacts/
```

**Why This Matters:**
- Ensures every code change is evaluated
- Catches performance regressions automatically
- Enables continuous model improvement
- Creates audit trail for compliance

---

## 3. Recommended Implementation Roadmap

### Phase 1: Enhance Current Setup (Week 1)
1. Add comprehensive tags to runs (model_family, size category, phase)
2. Add run descriptions with context
3. Implement composite scoring function
4. Create model selection script

### Phase 2: Model Registry (Week 2)
1. Set up MLflow Model Registry
2. Register top 3 models from current evaluation
3. Create model promotion workflow (Dev → Staging → Prod)
4. Update inference code to load from registry

### Phase 3: Visualization Dashboard (Week 2-3)
1. Build Streamlit dashboard for model comparison
2. Add interactive plots (scatter, parallel coordinates)
3. Implement side-by-side artifact comparison
4. Add export functionality (PDF reports)

### Phase 4: Dataset Versioning Integration (Week 3)
1. Link DVC dataset versions to MLflow runs
2. Log dataset metadata in every run
3. Create dataset changelog

### Phase 5: CI/CD Automation (Week 4)
1. Set up GitHub Actions workflow
2. Automate model evaluation on code changes
3. Add automated model registration for qualifying models
4. Set up notifications for evaluation failures

---

## 4. Specific Recommendations for 20-Model Evaluation

### Evaluation Strategy

**Batch 1: Initial Screening (Small Dataset)**
- Run all 20 models on small dataset (1000 samples)
- Goal: Quickly eliminate poor performers
- Filter: Keep models with `recall_at_target >= 0.70`
- Tag: `evaluation_phase=initial_screening`

**Batch 2: Deep Evaluation (Medium Dataset)**
- Run top 10 models on medium dataset (5000 samples)
- Goal: Assess generalization and stability
- Filter: Keep models with `composite_score >= 0.75`
- Tag: `evaluation_phase=deep_evaluation`

**Batch 3: Production Validation (Large Dataset)**
- Run top 5 models on large dataset (20000 samples)
- Goal: Final performance measurement
- Select top 3 for production candidates
- Tag: `evaluation_phase=production_validation`

### MLflow Experiment Structure

```
Semantic_Cache_Evaluation/
├── 01_initial_screening_small/     # 20 models, small dataset
├── 02_deep_eval_medium/            # 10 models, medium dataset
├── 03_production_validation_large/ # 5 models, large dataset
└── 04_threshold_optimization/      # Top 3 models, threshold tuning
```

### Run Naming Convention

```python
run_name = f"{model_family}-{model_size}-{dataset_size}-{timestamp}"
# Example: "bge-small-1k-20250107"
```

### Comparison Queries in MLflow UI

```python
# Find all small models with good recall
filter_string = "tags.model_size = 'small' AND metrics.recall_at_target >= 0.75"

# Find production candidates
filter_string = "tags.candidate_for_production = 'true'"

# Find models faster than 10ms
filter_string = "metrics.latency_ms_per_pair < 10"
```

---

## 5. Key Takeaways

### What Corporations Do Differently

1. **Everything is Tagged and Documented**: Every run has rich metadata
2. **Model Registry is Central**: Production models always loaded from registry, never by path
3. **Automated Model Selection**: Composite scoring removes bias
4. **Version Everything**: Code (git), data (DVC), models (MLflow registry)
5. **CI/CD Integration**: Model evaluation runs automatically on every change
6. **Custom Dashboards**: MLflow UI is augmented with domain-specific visualizations
7. **Formal Promotion Process**: Models go through stages (Dev → Staging → Prod)

### Immediate Actions You Can Take

#### Today: Add tags to your next run
```python
mlflow.set_tag("model_family", "sentence-transformers")
mlflow.set_tag("model_size", "small")
```

#### This Week: Implement composite scoring
```python
composite_score = calculate_model_score(metrics, weights)
mlflow.log_metric("composite_score", composite_score)
```

#### Next Week: Build simple Streamlit dashboard for comparison

#### This Month: Set up Model Registry and register your best model

---

## 6. Additional Resources

### MLflow Documentation
- [Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
- [Tracking Queries](https://mlflow.org/docs/latest/search-runs.html)
- [Experiment Management](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments)

### Industry Best Practices
- [Uber's MLflow Setup](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- [Netflix Model Management](https://netflixtechblog.com/notebook-innovation-591ee3221233)
- [DoorDash ML Platform](https://doordash.engineering/2020/04/23/building-scalable-ml-platform/)

### Tools to Consider
- **Streamlit**: Interactive dashboards
- **Great Expectations**: Data validation before model evaluation
- **Optuna**: Hyperparameter optimization with MLflow integration
- **W&B**: Alternative to MLflow with better collaboration features (paid)

---

## Conclusion

Your current MLflow setup is solid for individual experimentation. To scale to corporate standards:

1. **Add rich metadata** (tags, descriptions, git info)
2. **Use Model Registry** for production model management
3. **Build custom dashboards** for model comparison
4. **Automate model selection** with composite scoring
5. **Integrate with CI/CD** for continuous evaluation

Focus on these 5 areas and you'll have a production-grade ML evaluation pipeline that matches what companies like Uber, Netflix, and DoorDash use internally.

**Next Steps:**
1. Review this document and identify which practices are most valuable for your use case
2. Start with Phase 1 (adding tags and composite scoring) as quick wins
3. Request implementation help for specific sections when ready to proceed
