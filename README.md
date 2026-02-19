# Agentic Document Processor

A **local, production-grade agentic pipeline** that ingests any document, routes it through 6 specialised AI agents orchestrated by **LangGraph**, and produces validated JSON + redacted output with a full **Responsible AI audit trail**.

> **Prototype 2** — Classify · Extract · Validate · Self-Repair · Redact · Report

---

## Evaluation Results

| Metric | Target | Achieved |
|---|---|---|
| Extraction Accuracy | ≥ 90% | **92.3%** |
| PII Recall | ≥ 95% | **98.1%** |
| PII Precision | ≥ 90% | **94.6%** |
| Workflow Success | ≥ 90% | **95.2%** |
| P95 Latency | ≤ 4 s | **1.8 s** |

Evaluated on 28 documents across 6 types (Financial, Resume, Job Offer, Medical, Identity, Academic) with US, India, UK, Germany, and Canada coverage.

---

## Pipeline Architecture

```
Document Input (PDF / DOCX / TXT / Image)
    │
    ▼
[Document Loader]  ← PyPDFLoader / TextLoader / Tesseract OCR
    │
    ▼
[Classifier Agent]  ← LLM: decision-tree prompt, 600-char window
    │
    ▼
[Extractor Agent]  ← LLM: schema-driven JSON extraction, chunked
    │
    ▼
[Validator Agent]  ← Rule-based (60+ aliases) + LLM semantic check
    │
    ├── needs_repair=True ──► [Self-Repair Node] ──► [Validator Agent]
    │                                                (max 1 attempt)
    ▼
[Redactor Agent]  ← Presidio + custom Indian recognisers + LLM
    │
    ▼
[Reporter Agent]  ← JSON report + Responsible AI CSV + metrics
    │
    ▼
Output: report_{ts}.json  +  responsible_ai_{ts}.csv
```

### LLM Fallback Chain

```
Groq (llama-3.1-8b-instant)       ← Primary, ~1.5 s, 300+ tokens/sec
    ↓ rate-limit / error
Groq (backup key)                  ← Rate-limit escape
    ↓ both keys fail
Amazon Bedrock Claude 3.5 Haiku   ← Reliable fallback, AWS SLA
    ↓ timeout / error
HuggingFace API                    ← Tertiary
    ↓ unavailable
Ollama (llama3.1, local server)   ← Local fallback via ollama serve
    ↓ server not running
Local Llama (GGUF / Transformers) ← Offline last resort, CPU-only
```

Tenacity retries each provider 3× with exponential backoff (2 s → 4 s → 8 s) before advancing.

---

## Features

- **LangGraph stateful graph** — `DocumentState` TypedDict shared across all nodes; `MemorySaver` checkpointing for crash recovery
- **Conditional self-repair** — fires only when extraction accuracy < 80%; max 1 attempt; merges without overwriting good values
- **Hybrid PII redaction** — Microsoft Presidio baseline + LLM enhancement; custom regex recognisers for India Aadhaar, PAN, GSTIN, Passport, Voter ID, UPI, IFSC
- **Universal document ingestion** — PDF (digital + scanned), DOCX, TXT, PPTX, XLSX, PNG, JPG, TIFF; Tesseract OCR fallback for image-only files
- **5-tier LLM fallback** — Groq → Bedrock → HuggingFace → Ollama → Local Llama; zero downtime on any single provider failure
- **Responsible AI logging** — every agent decision (input summary, output summary, LLM provider, latency, status) exported as structured CSV; meets GDPR Art. 22 explainability requirements
- **FastAPI REST API** — `/process`, `/upload`, `/health` endpoints with Swagger UI
- **Streamlit dashboard** — upload, sample selector, tabbed results (classification, extraction, validation, redaction, metrics, trace log), JSON/CSV download

---

## Document Types & Schemas

| Type | Pydantic Schema | Fields |
|---|---|---|
| Financial | `FinancialDocumentFields` | 12 |
| Resume | `ResumeFields` | 11 (incl. nested lists) |
| Job Offer | `JobOfferFields` | 13 |
| Medical Record | `MedicalRecordFields` | 11 |
| Identity Document | `IdDocumentFields` | 11 |
| Academic | `AcademicFields` | 10 |

---

## Installation

```bash
# Clone
git clone https://github.com/Charan-L574/agentic-doc-processor.git
cd agentic-doc-processor

# Create virtual environment (Python 3.11 recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Install spaCy model (required by Presidio)
python -m spacy download en_core_web_lg
```

---

## Configuration

Create a `.env` file in the project root:

```env
# ── Groq (Primary LLM) ──────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY_BACKUP=your_groq_backup_key   # optional
GROQ_MODEL=llama-3.1-8b-instant

# ── Amazon Bedrock (Fallback) ────────────────────────────────────
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
BEDROCK_MODEL_ID=anthropic.claude-3-5-haiku-20241022-v1:0

# ── HuggingFace (Tertiary Fallback) ─────────────────────────────
HF_API_KEY=your_hf_token
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

# ── Ollama (Local server fallback) ─────────────────────────────
# Start server with: ollama serve
# Pull model with:   ollama pull llama3.1
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# ── Local Llama (GGUF fallback, optional) ───────────────────────
# Set path to a local .gguf file for llama-cpp-python inference
# LLAMA_MODEL_PATH=C:/models/llama-3.1-8b-q4.gguf

# ── App Settings ─────────────────────────────────────────────────
LOG_LEVEL=INFO
MAX_REPAIR_ATTEMPTS=1
MIN_EXTRACTION_ACCURACY=0.80
```

---

## Running the Application

### Streamlit UI (port 8501)

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) — upload any document or pick from 34 pre-loaded samples.

### FastAPI Server (port 8000)

```bash
uvicorn api.main:app --reload
```

Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).

#### Key endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | LLM provider availability |
| `POST` | `/process` | Process by file path `{"file_path": "data/samples/resume.txt"}` |
| `POST` | `/upload` | Multipart file upload |

---

## Running Tests

```bash
pytest tests/test_agents.py -v
```

---

## Project Structure

```
agentic-doc-processor/
├── agents/
│   ├── classifier_agent.py     # LLM-based doc type classification
│   ├── extractor_agent.py      # Schema-driven JSON field extraction
│   ├── validator_agent.py      # Rule-based + LLM validation
│   ├── self_repair_node.py     # Re-extraction / field repair
│   ├── redactor_agent.py       # Presidio + LLM PII redaction
│   └── reporter_agent.py       # JSON report + Responsible AI CSV
├── api/
│   └── main.py                 # FastAPI application
├── graph/
│   ├── state.py                # DocumentState TypedDict
│   └── workflow.py             # LangGraph pipeline definition
├── ocr/
│   └── processor.py            # Tesseract OCR wrapper
├── schemas/
│   └── document_schemas.py     # Pydantic schemas for all doc types
├── utils/
│   ├── llm_client.py           # 4-tier LLM fallback client
│   ├── document_loader.py      # Multi-format document ingestion
│   ├── retry_decorator.py      # Tenacity retry wrapper
│   ├── faiss_manager.py        # Optional FAISS vector lookup
│   ├── graph_visualizer.py     # LangGraph Mermaid diagram export
│   └── logger.py               # structlog structured logging
├── data/
│   ├── samples/                # 34 sample documents (6 types)
│   ├── evaluation_dataset_v2.csv   # 28-doc evaluation set
│   └── sample_dataset.csv      # Demo sample set
├── evaluation/
│   ├── metrics_calculation_report.xlsx
│   └── test_execution_log.xlsx
├── tests/
│   └── test_agents.py
├── config.py                   # Centralised settings via Pydantic
├── streamlit_app.py            # Streamlit dashboard
└── requirements.txt
```

---

## Responsible AI

Every agent appends a `ResponsibleAILog` entry containing:

- `agent_name`, `action`, `timestamp` (UTC)
- `input_summary` (first 200 chars), `output_summary`
- `llm_provider` (groq / bedrock_claude / huggingface / ollama / local_llama)
- `latency_ms`, `status`, `error_message`

Logs are exported to `reports/responsible_ai_{ts}.csv` and surfaced in the **Responsible AI** tab of the Streamlit dashboard.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| Primary LLM | Groq — llama-3.1-8b-instant |
| Fallback LLM | Amazon Bedrock — Claude 3.5 Haiku |
| Local LLM | Ollama (llama3.1) + Local Llama (GGUF/Transformers) |
| PII detection | Microsoft Presidio + spaCy + custom regex |
| OCR | Tesseract + pytesseract + pdf2image |
| Schema validation | Pydantic v2 |
| REST API | FastAPI |
| UI | Streamlit |
| Retries | Tenacity |
| Logging | structlog |
| Testing | Pytest |
| Python | 3.11 |

---

## License

MIT
