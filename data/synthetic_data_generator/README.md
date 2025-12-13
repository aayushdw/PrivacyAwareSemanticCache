# Synthetic Data Generator

A high-performance, asynchronous pipeline for generating synthetic datasets using Google Gemini models. This tool mimics real-world user intent variations to robustly train semantic search and cache systems.

## Overview & Data Strategy

High-quality semantic search requires more than just keyword matching. This pipeline automates the creation of a sophisticated training dataset by generating specific semantic variants for each seed question.

| Data Type | Definition | Architectural Purpose |
| :--- | :--- | :--- |
| **Similar Intent** | Different phrasing, same meaning. | **Positive Pairs**: Teaches the model that "How much is this?" and "What's the price?" are identical requests, improving recall. |
| **Different Intent** | Shared keywords, different meaning. | **Hard Negatives**: Teaches the model to distinguish subtler nuances (e.g., "Install python" vs "Uninstall python"), reducing false positives. |

It employs a **Map-Reduce** style architecture orchestrated by **LangGraph**, utilizing true parallelism across multiple model workers to maximize throughput while adhering to strict rate limits.

---

## System Architecture

The system is built on a distributed worker model managed by a central graph orchestrator.

### High-Level Components

```mermaid
graph TD
    classDef storage fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef external fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    InputFile[(Input CSV)]:::storage --> Load[Load Questions]:::process
    Load --> Distribute[Distribute Workload<br/>Round-Robin]:::process
    
    subgraph ExecutionPlane [Parallel Execution Plane]
        direction TB
        Distribute --> W1[Worker 1<br/>Model: gemma-2-1b]:::process
        Distribute --> W2[Worker 2<br/>Model: gemma-2-12b]:::process
        Distribute --> WN[Worker N<br/>Model: gemma-2-27b]:::process
    end

    subgraph ExternalServices [External Services]
        W1 -.->|Async API Calls| Gemini[Google Gemini API]:::external
        W2 -.->|Async API Calls| Gemini
        WN -.->|Async API Calls| Gemini
    end

    W1 -->|Thread-Safe Write| Output[(Output CSV)]:::storage
    W2 -->|Thread-Safe Write| Output
    WN -->|Thread-Safe Write| Output

    W1 -.->|Log Failures| ErrorLogs[(Error Logs)]:::storage
    
    ExecutionPlane --> Aggregate[Aggregate Stats]:::process
    Aggregate --> End((End))
```

---

## Logical Flow

### 1. The Worker Lifecycle
The architecture isolates execution into independent workers. Each worker manages its own state, ensuring that failures or delays in one model do not impact the overall pipeline thoughtput.

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant W as Worker
    participant G as Gemini API
    participant DB as Storage

    O->>W: Assign Batch of Questions
    
    loop Process Question
        W->>W: Distributed Rate Limit Check
        
        par Parallel Generation
            W->>G: Generate Similar Intent
            W->>G: Generate Different Intent
        end
        
        G-->>W: Return Synthetic Data
        W->>DB: Persist Results
    end
    
    W->>O: Report Completion Stats
```

### 2. Concurrency Model

The pipeline utilizes a hybrid concurrency model to balance I/O binding and API limits:

*   **Process Level**: The main Python process runs the LangGraph orchestrator.
*   **Worker Level (Multi-Tasking)**: Multiple `ModelWorker` instances run concurrently using `asyncio`.
*   **Request Level (Fan-Out)**: For every single question, we launch **2 parallel LLM requests** (one for "similar", one for "different"). This doubles the effective IOPS per worker.

### 3. Distributed Throttling
Unlike traditional centralized rate limiters, this architecture uses a simplified Clock-Based Throttling within each worker.

$$ \large T_{wait} = \max(0, \frac{60}{RPM_{limit}} - (T_{now} - T_{last})) $$

This ensures that every model stays mathematically within its API quotas (Requests Per Minute) regardless of network latency or jitter, without the need for complex locking mechanisms.