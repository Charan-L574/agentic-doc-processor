# Agentic Document Processor

Intelligent document processing system with multi-agent workflow, LLM fallback chain, and PII redaction.

## Features

- **Multi-Agent Workflow**: Classifier → Extractor → Validator → Self-Repair → Redactor → Reporter
- **LLM Fallback Chain**: AWS Bedrock → HuggingFace API → Local Llama
- **Document Support**: PDF, DOCX, TXT, images with OCR
- **PII Redaction**: Automatic detection and redaction of sensitive information
- **Validation & Repair**: Self-healing extraction with confidence scoring
- **LangGraph Integration**: State management and checkpointing

## Installation

```bash
# Clone repository
git clone https://github.com/Charan-L574/agentic-doc-processor.git
cd agentic-doc-processor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create `.env` file:

```env
# AWS Bedrock (Primary)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

# HuggingFace (Secondary Fallback)
HF_API_KEY=your_hf_token
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Local Llama (Tertiary Fallback - Optional)
LLAMA_MODEL_PATH=./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LLAMA_N_GPU_LAYERS=0  # Set to 33+ for GPU acceleration
```

## Usage

### FastAPI Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

### API Request

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"file_path": "document.pdf"}'
```

## Tech Stack

- **Framework**: LangGraph, LangChain
- **LLMs**: AWS Bedrock (Claude), HuggingFace, Llama 3.1
- **OCR**: Tesseract, pytesseract
- **API**: FastAPI, Streamlit
- **Storage**: FAISS vector store

## Project Structure

```
agentic-doc-processor/
├── agents/          # Agent implementations
├── api/             # FastAPI endpoints
├── graph/           # LangGraph workflow
├── utils/           # LLM client, document loader
├── ocr/             # OCR processing
├── schemas/         # Data models
└── config.py        # Configuration
```

## License

MIT
