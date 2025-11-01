# Privacy-Preserving Semantic Cache with Federated Learning

A privacy-preserving semantic cache for Large Language Model (LLM) queries using Federated Learning (FL) and Differential Privacy (DP). The system enables users to cache LLM responses locally while collaboratively improving the embedding model across all users without exposing individual queries to any central server.

## Project Status

**Current Phase:** Day 1-3 Foundation & Model Evaluation ✅
**Next Phase:** Federated Learning Implementation (Day 4-5)

### Completed Components

#### ✅ Embedding Engine & Evaluation Framework
- **20 embedding models** evaluated and benchmarked
- Automatic threshold tuning with precision constraints
- Comprehensive evaluation metrics (Precision, Recall, F1, Confusion Matrix)
- Visualization and reporting system

#### ✅ Vector Database Integration
- ChromaDB-based vector store implementation
- Efficient semantic similarity search
- Persistent local storage

#### ✅ Dataset Preparation
- Question similarity dataset (Quora Question Pairs)
- Training/test split functionality
- Synthetic data generation capabilities

### Model Evaluation Results

**Best Performing Models** (with optimal thresholds, min precision 0.80):

| Rank | Model | Category | Threshold | Precision | Recall | F1 Score |
|------|-------|----------|-----------|-----------|--------|----------|
| 1 | `roberta-large` | Quality | 0.880 | 0.8084 | 0.6187 | **0.7009** |
| 2 | `mpnet-base` | Balanced | 0.870 | 0.8059 | 0.5813 | **0.6754** |
| 3 | `instructor-large` | Quality | 0.970 | 0.8113 | 0.5560 | **0.6598** |
| 4 | `instructor-xl` | Quality | 0.940 | 0.8267 | 0.4960 | **0.6200** |
| 5 | `distilroberta` | Balanced | 0.890 | 0.8077 | 0.4760 | **0.5990** |

**Full evaluation report:** [embeddings/eval/results/results.md](embeddings/eval/results/results.md)

**Key Findings:**
- RoBERTa-Large achieves the best F1 score (0.7009) with strong balance
- MPNet-Base offers excellent performance (F1: 0.6754) with 768-dim embeddings
- Optimal thresholds range from 0.87-0.97 (models vary significantly)
- Precision constraint (≥0.80) ensures low false positive rates

## Architecture

### Current Implementation

```
┌─────────────────────────────────────────────────┐
│                User (CLI)                        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         Semantic Cache Manager                  │
│  ┌───────────────────────────────────────────┐  │
│  │  Query → Embedding Engine → Vector Search │  │
│  │  (sentence-transformers + ChromaDB)       │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   Local Cache Hit?    │
       └───┬───────────────┬───┘
           │ Yes           │ No
           ▼               ▼
    Return Cached    Forward to LLM API
       Response      Store Response + Embed
```

### Planned Architecture (with FL+DP)

```
Client Side:                     Server Side:
┌──────────────────────┐         ┌────────────────────┐
│  CLI Query Interface │         │   FL Server        │
└──────────┬───────────┘         │  (Flower/Flwr)     │
           │                     └─────────┬──────────┘
┌──────────▼───────────┐                   │
│  Semantic Cache      │                   │
│  (ChromaDB Vector DB)│         ┌─────────▼──────────┐
└──────────┬───────────┘         │ Model Aggregation  │
           │                     │  (FedAvg)          │
┌──────────▼───────────┐         └─────────┬──────────┘
│  Embedding Engine    │                   │
│ (sentence-transformers) ◄────────────────┘
│  + LoRA Adapters     │         Broadcast Global Model
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  FL Client (DP)      │
│  • Local Training    │
│  • DP-SGD (Opacus)   │
│  • Upload Δ Weights  │
└──────────────────────┘
```

### Dependencies

Core libraries:
- `sentence-transformers` - Embedding generation
- `chromadb` - Vector database
- `torch` - PyTorch for model training
- `scikit-learn` - Evaluation metrics
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization

Planned dependencies (for FL+DP):
- `flwr` (Flower) - Federated learning framework
- `opacus` - Differential privacy
- `peft` - LoRA adapters for efficient fine-tuning

## Usage

### 1. Embedding Engine

```python
from embeddings import EmbeddingEngine

# Initialize with best performing model
engine = EmbeddingEngine(model_name='sentence-transformers/all-mpnet-base-v2')

# Generate embedding for a query
query = "How do I reset my password?"
embedding = engine.encode(query)
print(f"Embedding dimension: {embedding.shape}")  # (768,)
```

### 2. Embedding Model Evaluation

### 3. Vector Database

### 4. CLI Interface (Coming Soon)

## TODO List

### 🔲 Phase 1: Core Cache Implementation

- [x] Embedding engine with multiple models
- [x] Model evaluation and comparison
- [x] Automatic threshold tuning
- [x] Vector database (ChromaDB) integration
- [x] Dataset preparation and splitting
- [ ] **Cache Manager**
  - Integrate embedding engine + vector store
  - Implement cache hit/miss logic
  - Add cache statistics tracking
  - Response storage and retrieval
- [ ] **CLI Query Interface**
  - Command-line tool for queries
  - Cache hit/miss reporting
  - Statistics dashboard
- [ ] **Basic Integration Tests**
  - End-to-end cache flow
  - Performance benchmarks

### 🔲 Phase 2: Model Fine-tuning

- [ ] **Training Pipeline**
  - Load training data from cache
  - Fine-tune embedding model
  - Use CosineSimilarityLoss
  - Save fine-tuned model
- [ ] **Evaluation on Fine-tuned Model**
  - Compare before/after metrics
  - Validate improvement

### 🔲 Phase 3: Federated Learning

- [ ] **FL Server**
  - Flower server setup
  - FedAvg aggregation strategy
  - Model versioning
  - Convergence tracking
- [ ] **FL Client** 
  - Flower client implementation
  - Local training on cache data
  - Model parameter extraction/update
  - Communication with server
- [ ] **FL Client with DP** 
  - Integrate Opacus PrivacyEngine
  - DP-SGD implementation
  - Gradient clipping
  - Noise addition
- [ ] **Privacy Tracker** 
  - Privacy budget accounting
  - Track (ε, δ) across rounds
  - Privacy guarantees validation
- [ ] **Multi-client Simulation**
  - Simulate 5-10 clients
  - Test FL convergence
  - Measure communication overhead

### 🔲 Phase 4: Privacy Validation

- [ ] **Membership Inference Attack**
  - Test privacy guarantees
  - Validate DP effectiveness
  - Target: ~52% attack success (random baseline: 50%)
- [ ] **Privacy-Utility Tradeoff Analysis**
  - Test different ε values
  - Plot F1 score vs privacy budget
  - Optimize for (ε=3, δ=10⁻⁵)

### 🔲 Phase 5: Optimization & Production

- [ ] **LoRA Integration**
  - Parameter-efficient fine-tuning
  - Reduce FL communication overhead
  - Target: ~6MB per round (vs 316MB full model)
- [ ] **Model Compression**
  - Quantization exploration
  - Model size optimization
  - Latency benchmarking
- [ ] **Comprehensive Benchmarks**
  - Cache hit rate measurement
  - Latency analysis
  - Privacy budget validation
  - FL convergence metrics
- [ ] **Documentation**
  - Architecture documentation
  - API reference
  - Deployment guide
  - Demo video/GIF

### 🔲 Phase 6: Advanced Features (Optional)

- [ ] Multi-domain support (code, math, general QA)
- [ ] Adaptive threshold tuning
- [ ] Cache eviction policies
- [ ] Distributed deployment support
- [ ] Monitoring dashboard (Streamlit)


## Development Approach

This project follows a **command-line first** approach.:

1. **CLI-based query interface** for testing and demonstration
2. **Local cache storage** using ChromaDB
3. **Python backend** for all ML operations
4. **Simulated FL** using Flower's simulation mode (5-10 clients on localhost)
5. **Iterative development** with evaluation at each phase

## Key Design Decisions

### Why ChromaDB over FAISS?
- Persistent storage with metadata support
- Built-in similarity search with filtering
- Easy integration and API
- Production-ready with minimal setup

### Why Command-Line Interface?
- Faster development and testing
- Focus on core ML/privacy functionality
- Easier to demonstrate FL concepts
- Can add web UI later if needed

### Why Threshold Tuning?
- Each model has optimal threshold (0.87-0.97 range observed)
- Significant F1 improvement vs fixed threshold
- Precision constraints ensure cache quality
- Model-specific optimization crucial for fair comparison

## References

### Research Papers
- [MeanCache (arXiv:2403.02694)](https://arxiv.org/abs/2403.02694) - Privacy-preserving cache
- [Semantic Caching for LLMs (arXiv:2504.02268)](https://arxiv.org/abs/2504.02268) - Domain-specific embeddings
- [DP-LoRA (arXiv:2312.17493)](https://arxiv.org/abs/2312.17493) - Differential privacy with LoRA
- [Flower Framework (arXiv:2007.14390)](https://arxiv.org/abs/2007.14390) - FL system design

### Documentation
- [sentence-transformers](https://sbert.net)
- [Flower FL Framework](https://flower.ai/docs)
- [ChromaDB](https://docs.trychroma.com)

### Datasets
- [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) - Question similarity dataset

---

*Last Updated: November 1, 2025*
*Project Plan Reference: [Privacy_Cache_FL_Project_Plan.pdf](Privacy_Cache_FL_Project_Plan.pdf)*
