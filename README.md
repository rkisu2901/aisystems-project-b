# Project B — Customer Support Agent with Tool Use & Human Escalation

Part of **AI Systems in Production** — AI Classroom Cohort 3

A LangGraph-based agent that receives customer queries, reasons about which tools to use, executes a multi-step plan, and produces structured responses with citations and confidence-based human escalation. Evolves from a naive pipeline in Week 1 into a full production agent by Week 4.

## Setup

### 1. Prerequisites
- Python 3.11+
- Docker Desktop
- OpenAI API key (with credits)
- LangFuse account (cloud.langfuse.com — free tier)

### 2. Environment

```bash
cp .env.example .env
# Fill in your API keys in .env

docker-compose up -d

pip install -r requirements.txt
```

### 3. Run

```bash
# Set up the database (retrieval over policy corpus)
python scripts/setup_db.py

# Ingest policy documents
python scripts/ingest.py

# Test the support pipeline
python scripts/support_pipeline.py

# Eval harness
python scripts/eval_harness.py
```

## Repo Structure

```
project-b/
├── corpus/                 # Acmera policy documents (19 files, same as Project A)
├── mock_data/
│   ├── customers.json      # Simulated customer DB (order history, tier, account status)
│   └── orders.json         # Simulated order status + shipping data
├── scripts/
│   ├── setup_db.py         # Create pgvector table for policy retrieval
│   ├── ingest.py           # Chunk + embed + store policy corpus
│   ├── retrieval.py        # Self-contained retrieval layer (embed + retrieve + assemble)
│   ├── support_pipeline.py # Naive pipeline: classify → retrieve → respond
│   └── eval_harness.py     # Multi-dimensional eval skeleton (built in Session 1)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## What We Build (Week by Week)

| Week | Layer | What Gets Added |
|------|-------|-----------------|
| 1 | Evaluate | Multi-dimensional eval: classification + retrieval + response + routing accuracy |
| 2 | Retrieve | Query classifier for tool selection, proto-reasoning layer |
| 3 | Optimize & Observe | Full LangGraph agent, model routing, agent decision tracing |
| 4 | Harden & Deploy | Output guardrails, loop detection, confidence-based escalation, AWS ECS Fargate |

## The Agent's Tool Set (Week 3+)

- **Policy & FAQ KB** — RAG retrieval over corpus (this repo's retrieval.py)
- **Customer Record Lookup** — simulated Postgres for order history, account status, tier
- **Order Status Tracker** — simulated API (mock_data/orders.json)
- **Human Escalation** — structured handoff: what was tried, why escalating
- **Response Generator** — structured, cited responses with confidence scoring
