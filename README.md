# 🛡️ CustomerCareAI Ecosystem

**Autonomous Multi-Agent Orchestration System for Enterprise Customer Care**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Open Source](https://img.shields.io/badge/Stack-Open%20Source%20Only-orange.svg)](#tech-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

**CustomerCareAI_Ecosystem** is a production-ready multi-agent orchestration system designed for end-to-end enterprise customer support. It coordinates **five specialized AI agents**, each implemented as an independent Python microservice, to deliver fast, accurate, empathetic, and **multilingual** (English & Arabic) customer support across all communication channels.

### ✨ Key Features

- 🤖 **5 Specialized AI Agents** — Each with distinct responsibilities, independently deployable
- 🔄 **Sequential Orchestration Pipeline** — OCS → KFO → EIA → PIR → Response → FAN (async)
- 🌐 **Multilingual Support** — English & Arabic with offline translation (argostranslate)
- 🧠 **Semantic Knowledge Base** — FAISS-powered FAQ search with sentence-transformers
- 💬 **Sentiment & Emotion Intelligence** — Real-time analysis with configurable escalation
- 📊 **Proactive Issue Detection** — Anomaly detection on account telemetry
- 📈 **Post-Interaction Analytics** — CSAT trends, knowledge gap detection
- 🐳 **Fully Containerized** — Docker Compose for local dev, Kubernetes-ready
- 🔒 **Privacy by Design** — PII masking, structured logging

---

## 🏗️ Architecture

```
[Incoming Request]
        │
        ▼
[1] OCS  ──── Intent + Channel Normalization + Draft Response
        │
        ▼
[2] KFO  ──── Semantic FAQ Retrieval enriches OCS Response
        │
        ▼
[3] EIA  ──── Sentiment + Emotion Analysis → Tone Adjustment + Escalation Check
        │
        ▼
[4] PIR  ──── Account Anomaly Scan → Proactive Alerts
        │
        ▼
[5] Escalation Gate ── if escalation_flag=True → Route to Human Agent Queue
        │
        ▼
[6] Unified Response Assembly ── Merge all outputs into structured JSON
        │
        ▼
[7] FAN  ──── (Async) Feedback Collection + Knowledge Update
        │
        ▼
[Final Output → Customer / Human Agent]
```

---

## 🤖 Agent Descriptions

| Agent | Code | Purpose |
|-------|------|---------|
| **Omni-Channel Support (OCS)** | `agents/omni_channel_support/` | Intent classification, language detection, response generation |
| **Knowledge Base & FAQ Optimizer (KFO)** | `agents/knowledge_base/` | Semantic FAQ search via FAISS + sentence-transformers |
| **Emotional Intelligence (EIA)** | `agents/emotional_intelligence/` | Sentiment scoring, emotion classification, escalation triggers |
| **Proactive Issue Resolution (PIR)** | `agents/proactive_issue/` | Account anomaly detection, proactive alert generation |
| **Feedback & Analytics (FAN)** | `agents/feedback_analytics/` | CSAT collection, trend analysis, knowledge gap detection |

---

## 📁 Project Structure

```
CustomerCareAI_Ecosystem/
│
├── orchestrator/
│   ├── main.py                  # FastAPI app; routes requests through agent pipeline
│   ├── context_manager.py       # Reads/writes conversation context from central DB
│   ├── aggregator.py            # Merges all agent outputs into final response
│   └── logger.py                # Structured JSON logging
│
├── agents/
│   ├── base_agent.py            # Abstract base class for all agents
│   ├── omni_channel_support/
│   │   ├── ocs_agent.py         # Intent classification + response generation
│   │   ├── intent_classifier.py # Zero-shot transformer + rule-based fallback
│   │   └── language_detector.py # langdetect wrapper (en/ar)
│   ├── knowledge_base/
│   │   ├── kfo_agent.py         # Semantic search + FAQ retrieval
│   │   ├── embedder.py          # sentence-transformers wrapper
│   │   ├── vector_store.py      # FAISS index management
│   │   └── faq_db.json          # Seed knowledge base (EN/AR)
│   ├── emotional_intelligence/
│   │   ├── eia_agent.py         # Sentiment + emotion classification
│   │   ├── emotion_classifier.py# HuggingFace emotion model wrapper
│   │   └── escalation_policy.py # Configurable escalation thresholds
│   ├── proactive_issue/
│   │   ├── pir_agent.py         # Account anomaly detection + alerts
│   │   ├── anomaly_detector.py  # Isolation Forest + z-score fallback
│   │   └── alert_builder.py     # Severity scoring + alert structuring
│   └── feedback_analytics/
│       ├── fan_agent.py         # Feedback ingestion + trend analysis
│       ├── trend_analyzer.py    # CSAT + issue trend aggregation
│       └── report_generator.py  # Performance report generation
│
├── api/
│   ├── endpoints.py             # REST API routes (FastAPI routers)
│   ├── schemas.py               # Pydantic v2 request/response models
│   └── middleware.py            # Request ID injection, rate limiting
│
├── db/
│   ├── models.py                # SQLAlchemy 2.0 ORM models
│   ├── migrations/              # Alembic migration scripts
│   └── seed_data/               # Test data (accounts, FAQs)
│
├── config/
│   ├── settings.py              # Pydantic BaseSettings (env vars)
│   └── escalation_thresholds.yaml
│
├── tests/                       # pytest + pytest-asyncio test suites
├── docker/                      # Dockerfiles + docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip (or conda)
- Docker & Docker Compose (optional, for containerized deployment)

### 1. Clone & Setup

```bash
git clone <repo-url>
cd CustomerCareAI_Ecosystem

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
```

### 2. Run Locally

```bash
# Start the orchestrator (includes all agents in monolith mode)
uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### 3. Docker Deployment

```bash
cd docker
docker-compose up --build
```

This starts all services:
| Service | Port |
|---------|------|
| Orchestrator | 8000 |
| OCS | 8001 |
| KFO | 8002 |
| EIA | 8003 |
| PIR | 8004 |
| FAN | 8005 |
| PostgreSQL | 5432 |

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 📡 API Reference

### `POST /api/v1/interact`

Main interaction endpoint — process customer messages through the full pipeline.

**Request Body:**
```json
{
  "customer_id": "CUST-001",
  "customer_message": "I need help resetting my password",
  "channel": "chat",
  "account_id": "ACC-001",
  "conversation_history": [],
  "conversation_id": null
}
```

**Response:**
```json
{
  "interaction_id": "uuid",
  "timestamp": "2026-02-19T11:40:00Z",
  "customer_id": "CUST-001",
  "channel": "chat",
  "language": "en",
  "response_text": "I can help you with your account settings...",
  "intent": "account_management",
  "sentiment_score": 0.1,
  "dominant_emotion": "neutral",
  "escalation_flag": false,
  "escalation_reason": null,
  "suggested_faq_articles": [...],
  "proactive_alerts": [],
  "feedback_analysis": {...},
  "agent_logs": {...}
}
```

### `POST /api/v1/escalate`
Route interactions to human agent queue.

### `GET /api/v1/escalation-queue`
Check current escalation queue status.

### `POST /api/v1/feedback`
Submit post-interaction feedback (CSAT score, comments).

### `GET /api/v1/health`
Health check endpoint.

---

## ⚡ Escalation Policy

Escalation to a human agent triggers when **any** condition is met:

| Condition | Threshold |
|-----------|-----------|
| Sentiment score | < -0.65 |
| Dominant emotion | `anger` or `distress` for 2+ consecutive turns |
| PIR alert severity | `critical` |
| OCS intent | `escalation_request` |
| Unresolved turns | 3+ consecutive turns without resolution |

Thresholds are configurable via `config/escalation_thresholds.yaml`.

---

## 🌐 Multilingual Support

- **Language Detection:** `langdetect` auto-detects input language
- **Translation:** `argostranslate` for offline EN↔AR translation
- **Response Language:** Always matches the customer's detected input language
- **Knowledge Base:** Separate bilingual FAQ articles (EN & AR)

---

## 🛠️ Tech Stack

| Purpose | Library |
|---------|---------|
| API Framework | FastAPI ≥0.110 |
| ASGI Server | Uvicorn ≥0.29 |
| Data Validation | Pydantic v2 ≥2.6 |
| NLP / Transformers | HuggingFace Transformers, PyTorch |
| Embeddings | sentence-transformers ≥2.6 |
| Vector Search | FAISS-CPU ≥1.8 |
| ML Models | scikit-learn ≥1.4 |
| Sentiment | TextBlob, NLTK |
| Language Detection | langdetect ≥1.0.9 |
| Database ORM | SQLAlchemy ≥2.0 |
| Async HTTP | httpx ≥0.27 |
| Structured Logging | structlog ≥24.1 |
| Testing | pytest, pytest-asyncio |

**All dependencies are open-source. No proprietary APIs required.**

---

## 📊 Monitoring & Logging

All events are logged as structured JSON via `structlog`:
- Every log includes `interaction_id`, `agent_name`, and ISO-8601 `timestamp`
- Agent start/completion/failure events are automatically logged
- Request/response timing via middleware
- PII is masked in all log output

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific agent tests
pytest tests/test_ocs.py -v
pytest tests/test_eia.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ for enterprise customer care excellence.**
