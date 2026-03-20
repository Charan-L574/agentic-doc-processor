# Agentic Document Processor — Complete Code Documentation

This document explains **every Python code file** in this repository, with:
- what the file does,
- key classes/functions,
- parameter meaning,
- syntax basics used in that file,
- how it connects to the end-to-end pipeline.

---

## 0) One-by-one learning TODO (execution order)

Use this exact order while learning:

1. `graph/state.py` (state contract)
2. `schemas/document_schemas.py` (data models)
3. `utils/config.py` (configuration loading)
4. `utils/logger.py` (logging setup)
5. `utils/llm_client.py` (provider + fallback engine)
6. `utils/document_loader.py` + `ocr/processor.py` (text extraction)
7. `prompts.py` (LLM instructions)
8. `agents/classifier_agent.py`
9. `agents/extractor_agent.py`
10. `agents/validator_agent.py`
11. `agents/self_repair_node.py`
12. `agents/redactor_agent.py`
13. `agents/reporter_agent.py`
14. `graph/workflow.py` (standard graph)
15. `agents/human_review_agent.py` + `agents/supervisor_agent.py` (HITL graph)
16. `utils/knowledge_lookup.py` + `utils/faiss_manager.py` (schema/profile retrieval)
17. `utils/observability.py` + `utils/graph_visualizer.py`
18. `api/main.py` (API orchestration)
19. `streamlit_app.py` (UI)
20. `tests/test_agents.py` (behavior verification)
21. package `__init__.py` files (exports and module boundaries)

---

## 1) End-to-end processing flow (how document processing works)

### 1.1 Standard flow

1. API receives `file_path` (`/process` in `api/main.py`).
2. `DocumentProcessingWorkflow.process_document()` in `graph/workflow.py` is invoked.
3. `utils.document_loader.document_loader.load_and_extract_text()` extracts text from PDF/image/text.
4. Initial `DocumentState` dictionary is created.
5. LangGraph executes nodes in order:
   - classify (`ClassifierAgent`)
   - extract (`ExtractorAgent`)
   - validate (`ValidatorAgent`)
   - conditional repair loop (`SelfRepairNode`) if needed
   - redact (`RedactorAgent`)
   - report (`ReporterAgent`)
6. API converts final state into `ProcessResponse` and returns JSON.

### 1.2 HITL flow

1. API `/process/start` starts supervisor graph.
2. Graph pauses at `interrupt()` in `HumanInLoopAgent.review_classification()`.
3. Human decision sent to `/thread/{id}/resume`.
4. Graph continues; may pause again at extraction review.
5. After final approval, redact/report complete and response is returned.

---

## 2) Python syntax quick primer used across project

- `import x` → load module.
- `from a import b` → import symbol `b` from module `a`.
- `class A:` → class definition.
- `def f(x: int) -> str:` → function with type hints.
- `self` → current object instance inside class methods.
- `@property` → method used like field access (`obj.value`).
- `try/except` → error handling.
- `Optional[T]` → can be `T` or `None`.
- `Dict[str, Any]` → dictionary with string keys and any values.
- `TypedDict` → typed dictionary contract.
- `Enum` → fixed set of allowed values.
- `BaseModel` (Pydantic) → validated data model.
- f-string: `f"hello {name}"`.
- slicing: `text[:300]`, `text[-300:]`.

---

## 3) File-by-file documentation

## A) Graph and state

### `graph/state.py`

**Purpose**
- Defines the shared pipeline state (`DocumentState`) and graph config (`GraphConfig`).

**Key constructs**
- `class DocumentState(TypedDict)`:
  - `file_path`, `raw_text`: raw inputs.
  - classification keys: `doc_type`, `classification_result`.
  - extraction keys: `extracted_fields`, `extraction_result`.
  - validation keys: `validation_result`, `needs_repair`, `current_accuracy`, `missing_schema_fields`.
  - redaction keys: `redacted_text`, `redaction_result`.
  - control keys: `errors`, `success`, `agent_timings`, `trace_log`.
  - HITL keys: `hitl_required`, `hitl_type`, `hitl_resolution`, `hitl_corrections`, `custom_doc_type`.
  - supervisor metadata keys for decisions and reasons.
- `class GraphConfig(TypedDict)`:
  - `max_repair_attempts`, `enable_responsible_ai_logging`, `visualize`.

**Why this is used**
- Every node reads/writes the same state structure. This prevents mismatch bugs.

---

### `graph/workflow.py`

**Purpose**
- Builds and runs the **standard** LangGraph workflow.
- Also contains `HITLWorkflow` compatibility shim that delegates to supervisor.

**Main class: `DocumentProcessingWorkflow`**

- `__init__(config: GraphConfig=None)`
  - sets defaults,
  - calls `_build_graph()`,
  - prepares `compiled_graph`.

- `_initialize_state(file_path, raw_text, ground_truth_pii=None)`
  - returns a fully initialized `DocumentState` with default values.

- Node wrappers:
  - `_classify_node`, `_extract_node`, `_validate_node`, `_repair_node`, `_redact_node`, `_report_node`.
  - each calls corresponding agent.

- `_should_repair(state) -> Literal["repair", "redact"]`
  - checks `needs_repair`, `current_accuracy`, and repair budget.

- `_build_graph()`
  - defines nodes and edges:
    - classify → extract → validate
    - validate → repair/redact (conditional)
    - repair → validate (loop)
    - redact → report → END

- `compile()`
  - compiles graph with `MemorySaver` checkpointer.

- `process_document(file_path, thread_id, ground_truth_pii=None)`
  - extracts text from file,
  - initializes state,
  - streams graph execution,
  - returns final state dict.

- `get_graph_mermaid()` / `_generate_simple_mermaid()`
  - returns clean Mermaid diagram for UI.

**Additional class: `HITLWorkflow`**
- Thin adaptor used for backward compatibility.
- `start_processing()` and `resume_processing()` delegate to `SupervisorAgent`.

---

### `graph/__init__.py`

**Purpose**
- Re-exports `DocumentState` and `GraphConfig` for easy imports.

**Syntax note**
- `__all__ = [...]` controls symbols exposed on `from graph import *`.

---

## B) Schemas

### `schemas/document_schemas.py`

**Purpose**
- Central data contract using Pydantic models and enums.

**Enums**
- `DocumentType`: financial_document, resume, job_offer, medical_record, id_document, academic, unknown.
- `ValidationStatus`: valid, invalid, needs_repair, repaired, valid_after_repair.
- `PIIType`: email, phone, ssn, credit_card, address, etc.

**Extraction field models**
- `FinancialDocumentFields`, `ResumeFields`, `JobOfferFields`, `MedicalRecordFields`, `IdDocumentFields`, `AcademicFields`, `GeneralDocumentFields`.

**Agent output models**
- `ClassificationResult`, `ExtractionResult`, `ValidationResult`, `PIIDetection`, `RedactionResult`, `MetricsReport`, `ResponsibleAILog`, `ProcessingResult`.

**Parameter/field behavior examples**
- `Field(default_factory=list)` creates a fresh list per instance.
- `Field(ge=0.0, le=1.0)` constrains numeric range.
- `timestamp: datetime = Field(default_factory=datetime.utcnow)` auto-generates runtime timestamp.

---

### `schemas/__init__.py`

**Purpose**
- Aggregates exports from `document_schemas.py` so callers import from one place.

---

## C) Agents

### `agents/classifier_agent.py`

**Purpose**
- Classifies document type from text excerpt.

**Key methods**
- `_prepare_text_excerpt(text, max_chars=600)`:
  - keeps beginning and end of text for stronger signal.
- `_parse_llm_response(content)`:
  - parses JSON directly or from markdown code block fences.
- `_normalize_doc_type(doc_type_str)`:
  - maps aliases (`cv`, `offer letter`, `aadhaar`) to enum values.
- `classify(state)`:
  - prepares prompt,
  - calls `llm_client.generate(...)`,
  - creates `ClassificationResult`,
  - updates timings + trace logs,
  - handles fallback-to-UNKNOWN on errors.

**Important parameters in LLM call**
- `temperature=0.0` for deterministic outputs.
- `max_tokens=150` because classification output is short.
- `groq_model="llama-3.1-8b-instant"` for speed.

---

### `agents/extractor_agent.py`

**Purpose**
- Extracts structured fields from document text.

**Core concepts**
- `SCHEMA_MAP`: doc type → Pydantic schema.
- `FEW_SHOT_EXAMPLES`: high-quality examples to improve extraction quality.

**Key methods**
- `_camel_to_snake(name)` and `_normalize_field_names(data)`:
  - normalizes LLM key naming style.
- `_normalize_extracted_values(data)`:
  - normalizes dates/names/strings for evaluation consistency.
- `_chunk_text(text, max_length=6000, overlap=500)`:
  - splits long text into overlapping chunks.
- `_extract_from_chunk(chunk, doc_type, force_provider=None, effective_label=None)`:
  - sends extraction prompt,
  - parses with `repair_json`.
- `_merge_extracted_fields(chunks_results)`:
  - combines chunk outputs.
- `extract(state)`:
  - derives effective doc label,
  - chunk-extracts,
  - normalizes,
  - applies aliases,
  - writes `ExtractionResult` and trace log.

**Why `repair_json` is used**
- LLM output can have trailing commas/invalid quoting; repair step makes parser robust.

---

### `agents/validator_agent.py`

**Purpose**
- Hybrid validation: rule checks + optional LLM semantic validation + schema accuracy gating.

**Important constants/maps**
- `FIELD_PRIORITIES`: high-value required fields per doc type.
- `FIELD_ALIASES`: alternate names → canonical schema keys.
- `ACCURACY_THRESHOLD = 0.80`.

**Key methods**
- `_normalize_extracted_fields(...)`: alias remap + scalar coercion.
- `_resolve_priority_fields(...)`: profile-aware required field resolution.
- `_validate_field_format(field_name, value)`: regex format checks.
- `_compute_accuracy(extracted_fields, doc_type, custom_doc_type=None)`:
  - computes schema completion,
  - can use knowledge profile from SQLite/FAISS.
- `_rule_based_validation(...)`: fast pre-checks.
- `_validate_against_json_schema(...)`: strict JSON Schema checks (if available).
- `_validate_with_llm(...)`: semantic validation prompt with knowledge context.
- `validate(state)`:
  - normalizes fields,
  - pre-checks,
  - fast path when fields are strong,
  - otherwise calls LLM,
  - enforces accuracy gate,
  - writes `ValidationResult`, `current_accuracy`, `missing_schema_fields`.

**Behavior details**
- High accuracy can demote some LLM errors to warnings.
- After max repair attempts, validator can accept with warnings.

---

### `agents/self_repair_node.py`

**Purpose**
- Attempts automatic correction if validation fails.

**Modes**
1. **Repair mode**: fix bad fields.
2. **Re-extraction mode**: re-extract when sparse/low-accuracy.

**Key methods**
- `_parse_llm_response(content)` uses `repair_json`.
- `_get_schema_fields(doc_type)` gets expected schema field names.
- `_should_re_extract(extracted_fields, state)` triggers when:
  - too few fields, or
  - `current_accuracy < 0.90`.
- `repair(state)`:
  - enforces max attempts,
  - picks prompt template,
  - calls LLM,
  - updates extraction + repair history + traces.

**Parameters worth noting**
- `self.max_attempts` from config.
- `self.llm_model = 'llama-3.3-70b-versatile'` for stronger correction quality.

---

### `agents/redactor_agent.py`

**Purpose**
- Detects and redacts PII using hybrid approach:
  - Presidio,
  - regex/custom detectors,
  - LLM augmentation.

**Major components**
- Presidio recognizer registry setup with custom Indian IDs:
  - Aadhaar, PAN, GSTIN, Passport, Voter ID, DL, UPI, IFSC.
- `PRESIDIO_TO_PII_TYPE` map converts provider entity labels to project enum.

**Key methods**
- `_detect_gender_patterns`, `_detect_custom_id_patterns`, `_detect_multiline_addresses`.
- `_detect_pii_with_presidio(text)`.
- `_validate_llm_pii(type, text, confidence)` to filter false positives.
- `_detect_pii_with_llm(text)` returns `(detections, llm_response)`.
- `_redact_text(text, detections)` applies replacements longest-first.
- `_compute_metrics(detected_pii, extracted_fields, ground_truth_pii)` computes precision/recall.
- `redact(state)` orchestrates full hybrid detection and updates state.

**Why longest-first replacement**
- Avoid partial replacement bugs (`John` replacing inside `John Smith` first).

---

### `agents/reporter_agent.py`

**Purpose**
- Produces final metrics and responsible AI reports.

**Key methods**
- `_compute_extraction_accuracy(extracted_fields, doc_type)`:
  - schema-aware field completion metric.
- `_compute_workflow_success(state)`:
  - verifies all key stages present.
- `_save_json_report(report_data, filename)` and `_save_csv_report(trace_logs, filename)`.
- `_create_metrics_summary(state)`:
  - extraction accuracy,
  - pii precision/recall,
  - workflow success,
  - latency by agent,
  - extra fields discovered beyond schema.
- `generate_report(state)`:
  - stores final reports in `reports/` and marks `state["success"]`.

---

### `agents/human_review_agent.py`

**Purpose**
- Implements LangGraph **interrupt-based** human review checkpoints.

**Key methods**
- `review_classification(state)`:
  - calls `interrupt({...})` with classification payload,
  - applies human override (`doc_type_override`) or reject.
- `review_extraction(state)`:
  - calls `interrupt({...})` with extracted fields and validation info,
  - applies human `corrections`,
  - can reject document,
  - sets validation as approved for continuation.

**Syntax focus**
- `human_input: Dict[str, Any] = interrupt({...})`:
  - function pauses graph here;
  - resumes with returned human payload.

---

### `agents/supervisor_agent.py`

**Purpose**
- Central orchestrator of HITL policy graph and routing decisions.

**Policy checkpoints**
- `_supervise_classification`: classification review is mandatory HITL.
- `_supervise_validation`: decides repair vs hitl_extract vs auto-approve.

**Routing methods**
- `_route_after_supervise_classify`
- `_route_after_hitl_classify`
- `_route_after_supervise_validate`
- `_route_after_hitl_extract`

**Graph build (`_build_graph`)**
- Adds specialist nodes + supervisor nodes + HITL nodes.
- Defines full flow and loops in one place.

**Public runtime API**
- `compile_workflows()`.
- `process_document(...)` standard flow delegate.
- `start_processing(file_path, thread_id)` for HITL start.
- `resume_processing(thread_id, human_input)` to continue from interrupt.

---

### `agents/__init__.py`

**Purpose**
- Instantiates/exports agent classes and singleton instances.

---

## D) Utilities

### `utils/config.py`

**Purpose**
- Loads `config.ini` and `.env`, exposes strongly-typed settings.

**Core API**
- `Env.get/getint/getfloat/getboolean` wrappers.
- Many `@property` methods for paths/providers/timeouts/thresholds.

**Important settings families**
- paths (`DATA_DIR`, `REPORTS_DIR`, ...)
- LLM providers (Groq, Bedrock, HF, local)
- retries/cache
- stack profile switches
- LangSmith observability
- API host/port/reload
- metrics thresholds

**Why `@property` everywhere**
- Gives IDE-friendly access (`settings.GROQ_MODEL`) with type conversion and fallback.

---

### `utils/logger.py`

**Purpose**
- Sets structured logging (console + file), flush-safe for uvicorn.

**Key parts**
- `_FlushingStreamHandler.emit()` ensures immediate flush.
- module guard `_LOGGING_CONFIGURED` prevents duplicate handlers.
- `setup_logging()` builds standard+structlog pipeline.
- global singleton `logger = setup_logging()`.

---

### `utils/retry_decorator.py`

**Purpose**
- Generic retry decorator using `tenacity` with exponential backoff.

**Main API**
- `with_retry(max_attempts=None, min_wait=None, max_wait=None, multiplier=None)`.

**Behavior**
- handles both async and sync functions.
- retries on selected network/provider exceptions.
- logs before sleep and after attempts.

---

### `utils/llm_client.py`

**Purpose**
- Unified LLM gateway with provider initialization, fallback routing, and local cache.

**Enum**
- `LLMProvider`: groq, bedrock_claude, bedrock_nova, huggingface, ollama, local_llama.

**Main responsibilities**
1. Initialize clients (`_initialize_groq/_bedrock/_huggingface/_ollama/_llama`).
2. Manage Groq multi-key rotation and cooldowns.
3. Cache responses in SQLite (`_cache_get/_cache_set`).
4. Invoke provider-specific methods:
   - `_invoke_groq`
   - `_invoke_bedrock_claude`
   - `_invoke_nova`
   - `_invoke_huggingface`
   - `_invoke_ollama`
   - `_invoke_llama` (GGUF/Transformers)
5. `generate(...)` orchestrates forced-provider or fallback sequence.

**Key `generate()` parameters**
- `prompt`, `system_prompt`
- `max_tokens`, `temperature`
- `force_provider` (optional)
- `groq_model`
- `groq_key` (selects preferred Groq key for round-robin plan)

**Fallback strategy**
- Groq keys (with cooldown-aware retries) → Bedrock Claude/Nova → HuggingFace → local Llama.

---

### `utils/document_loader.py`

**Purpose**
- Format-aware document text extraction using LangChain loaders.

**Key class: `LangChainDocumentLoader`**
- detects document type from extension,
- uses loader per file type (`PyPDFLoader`, `TextLoader`, etc.),
- OCR for images if needed,
- generic fallback for unknown types.

**Important methods**
- `detect_document_type(file_path)`
- `load_pdf/load_text/load_csv/load_image/load_docx/load_pptx/load_xlsx/load_html/load_generic`
- `load_file(file_path)` and `load_and_extract_text(file_path)`

---

### `utils/faiss_manager.py`

**Purpose**
- Manages FAISS semantic index + sentence-transformer embeddings.

**Core methods**
- `_load_or_create_index()`
- `add_documents(texts, metadata)`
- `search(query, k=3)`
- `save()` / `clear()`
- global getter `get_faiss_index()`.

---

### `utils/knowledge_lookup.py`

**Purpose**
- Maintains schema knowledge registry (SQLite + JSON custom schemas + FAISS retrieval).

**Data sources**
- Built-in Pydantic schema profiles.
- Optional custom profiles in `data/knowledge/custom_schemas.json`.

**Key methods**
- `_init_db`, `_upsert_entry`, `_bootstrap_knowledge`, `_rebuild_index`.
- `register_custom_schema(doc_type, required_fields, json_schema, notes)`.
- `list_profiles()`.
- `refresh()`.
- `get_validation_profile(doc_type, custom_doc_type=None)` (used by validator).

---

### `utils/observability.py`

**Purpose**
- Optional LangSmith wrapper with local no-op fallback.

**Main class: `ObservabilityClient`**
- `start_run(name, inputs, tags=None, metadata=None)`
- `end_run(run_id, outputs=None, error=None)`
- `is_active` property.

**Why this pattern**
- same API works whether LangSmith is configured or not.

---

### `utils/graph_visualizer.py`

**Purpose**
- Generates Mermaid diagrams for graph structure and execution trace.

**Key methods**
- `generate_mermaid_diagram(workflow)`.
- `extract_execution_path(trace_log)`.
- `generate_execution_path_diagram(trace_log)`.
- `visualize_execution_trace(trace_log)` (sequence diagram).

---

### `utils/__init__.py`

**Purpose**
- Re-export key utility symbols (`logger`, `with_retry`, `llm_client`, etc.).

---

## E) OCR package

### `ocr/processor.py`

**Purpose**
- OCR/text extraction support focused on images and PDFs.

**Key constructs**
- `FileType` enum.
- `OCRProcessor` class.

**Key methods**
- `_setup_tesseract()`.
- `detect_file_type(file_path)`.
- `extract_text_from_image(image_path, language=None)`.
- `extract_text_from_pdf(pdf_path, language=None, dpi=300)`:
  - tries `PyPDF2` first,
  - OCR fallback via `pdf2image` + tesseract.
- `extract_text_from_file(file_path)` universal route.
- `get_ocr_info(image_path)` confidence and word stats.

**Global instance**
- `ocr_processor = OCRProcessor()`.

---

### `ocr/__init__.py`

**Purpose**
- Re-exports `OCRProcessor`, `FileType`, and singleton `ocr_processor`.

---

## F) API and UI

### `api/main.py`

**Purpose**
- FastAPI app exposing processing, HITL, monitoring, knowledge, visualization endpoints.

**Startup/shutdown**
- compiles workflows,
- warms Presidio,
- warms knowledge lookup.

**Middleware**
- `monitoring_middleware` wraps each request with observability run.

**Major endpoints**
- `GET /health`
- `GET /monitoring/health`
- `POST /monitoring/langsmith/test`
- `POST /process` (standard non-interrupt flow)
- `POST /upload-and-process`
- `GET /visualize/graph`
- `POST /visualize/trace`
- `POST /visualize/execution-path`
- `GET /workflow/diagram`
- `GET /metrics/summary`
- `GET/POST /knowledge/schemas`
- `POST /knowledge/refresh`
- `POST /process/start` (HITL start)
- `POST /thread/{thread_id}/resume` (HITL resume)
- `POST /process/auto` (policy routing)

**Key helper functions**
- `_extract_supervisor_policy(state)`
- `_serialize_trace(trace_log)`
- `_build_process_response_from_state(final_state, start_time)`

**Pydantic request/response models**
- `ProcessRequest`, `ProcessResponse`, `HealthResponse`,
  `KnowledgeSchemaRequest`, `KnowledgeSchemaResponse`,
  `StartProcessingRequest`, `ResumeRequest`, `AutoProcessRequest`,
  `ProcessingStatusResponse`.

---

### `api/__init__.py`

**Purpose**
- Exposes FastAPI `app` object for server import.

---

### `streamlit_app.py`

**Purpose**
- Frontend dashboard for running and visualizing document processing.

**Main responsibilities**
- API connectivity checks (`check_api_health`, monitoring functions).
- Standard process calls (`process_document`).
- HITL process control (`start_document_processing`, `resume_document_processing`).
- UI renderers:
  - classification panel,
  - extraction table,
  - validation panel,
  - redaction details,
  - metrics charts,
  - knowledge lookup info,
  - Mermaid diagram rendering.

**Notable syntax patterns**
- `st.columns(...)` for layout.
- `st.metric(...)` for KPI cards.
- `st.expander(...)` for collapsible debug/info blocks.
- `requests.get/post(...)` for backend communication.

---

## G) Prompt definitions

### `prompts.py`

**Purpose**
- Stores all reusable prompt templates and system prompts.

**Sections**
- classifier prompts,
- extractor prompts,
- validator prompts,
- self-repair prompts,
- redactor prompts.

**Why centralized**
- Keeps agent logic separate from prompt content and allows easy iteration.

---

## H) Tests

### `tests/test_agents.py`

**Purpose**
- Unit tests for classifier/extractor/validator/repair/redactor/reporter and loader.

**Techniques used**
- `pytest` fixtures for sample states.
- `unittest.mock.patch` to mock LLM calls.
- assertions on state updates and output metrics.

**Why important**
- validates agent behavior independent of live model calls.

---

## I) Package entry files (`__init__.py`)

These files organize exports and make directories importable packages:
- `agents/__init__.py`
- `api/__init__.py`
- `graph/__init__.py`
- `ocr/__init__.py`
- `schemas/__init__.py`
- `utils/__init__.py`

They typically contain import re-exports and `__all__` declarations.

---

## 4) Quick mapping: which file owns which functionality

- **State contract** → `graph/state.py`
- **Graph execution (standard)** → `graph/workflow.py`
- **HITL orchestration policy** → `agents/supervisor_agent.py`
- **Human pause/resume nodes** → `agents/human_review_agent.py`
- **Classification** → `agents/classifier_agent.py`
- **Extraction** → `agents/extractor_agent.py`
- **Validation** → `agents/validator_agent.py`
- **Self-repair** → `agents/self_repair_node.py`
- **PII redaction** → `agents/redactor_agent.py`
- **Metrics/reporting** → `agents/reporter_agent.py`
- **LLM providers and fallback** → `utils/llm_client.py`
- **Document loading/OCR path** → `utils/document_loader.py`, `ocr/processor.py`
- **Knowledge profile retrieval** → `utils/knowledge_lookup.py`, `utils/faiss_manager.py`
- **Observability/tracing** → `utils/observability.py`
- **Graph visualization helpers** → `utils/graph_visualizer.py`
- **Backend API** → `api/main.py`
- **Frontend** → `streamlit_app.py`

---

## 5) New contributor checklist (practical)

1. Start from `graph/state.py` and `schemas/document_schemas.py`.
2. Read `utils/config.py` to understand runtime switches.
3. Follow one document in debugger through `graph/workflow.py`.
4. Then inspect each agent in the order: classify → extract → validate → repair → redact → report.
5. Finally read `api/main.py` and `streamlit_app.py` for end-user flow.
6. Run tests in `tests/test_agents.py` after any change.

---

## 6) Part 2 — Beginner classroom style (near line-by-line walkthrough)

This section slows down and explains key files like a classroom whiteboard.

### 6.1 `graph/state.py` line-by-line understanding

```python
from typing import TypedDict, Optional, Dict, List, Any
from datetime import datetime
```
- `typing` imports are used for type hints.
- `TypedDict` means: “dictionary with known keys and value types”.
- `datetime` is used to track timestamps in state and logs.

```python
class DocumentState(TypedDict):
```
- Declares the full state contract moving through LangGraph.
- Important: this is not a class with behavior; it is a typed dictionary schema.

```python
file_path: str
raw_text: str
```
- Minimum raw inputs.
- Every run starts with these keys populated.

```python
doc_type: Optional[DocumentType]
classification_result: Optional[ClassificationResult]
```
- Initially `None`.
- Filled by classifier node.

```python
extracted_fields: Optional[Dict[str, Any]]
extraction_result: Optional[ExtractionResult]
```
- `extracted_fields` is plain dictionary from LLM.
- `extraction_result` is strongly typed Pydantic object for structured metadata.

```python
needs_repair: bool
repair_attempts: int
current_accuracy: float
missing_schema_fields: List[str]
```
- These keys control the validation-repair loop.
- Router uses them to decide whether to run repair node.

```python
trace_log: List[ResponsibleAILog]
agent_timings: Dict[str, float]
errors: List[str]
success: bool
```
- `trace_log`: audit/observability record.
- `agent_timings`: latency by node.
- `errors`: human-readable error list.
- `success`: final completion flag.

```python
hitl_required: bool
hitl_type: Optional[str]
hitl_resolution: Optional[str]
```
- HITL metadata used in supervisor/human-review flows.

```python
class GraphConfig(TypedDict):
  max_repair_attempts: int
  enable_responsible_ai_logging: bool
  visualize: bool
```
- Graph-level constants and feature toggles.

**Why this file matters most**
- If this contract is wrong, every agent can break.
- Treat this file as the single source of truth for pipeline state.

---

### 6.2 `graph/workflow.py` line-by-line understanding

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
```
- `StateGraph`: graph builder for stateful nodes.
- `END`: special node marker to stop execution.
- `MemorySaver`: in-memory checkpoint store for resumable runs.

```python
class DocumentProcessingWorkflow:
```
- Standard non-HITL orchestration class.

```python
self.config = config or GraphConfig(...)
```
- Uses caller config if provided, otherwise defaults.

```python
self.graph = self._build_graph()
self.compiled_graph = None
```
- Build graph definition immediately.
- Compile lazily later (faster startup, flexible environment setup).

```python
def _initialize_state(...):
  return DocumentState(...)
```
- Creates fresh state with all keys present.
- Prevents key-missing errors inside agents.

Node wrappers:
```python
def _classify_node(self, state):
  return classifier_agent.classify(state)
```
- Each wrapper delegates to one agent.
- Wrapper exists so graph stays decoupled from direct imports in edge logic.

Routing decision:
```python
def _should_repair(self, state) -> Literal["repair", "redact"]:
```
- Reads `needs_repair`, `repair_attempts`, `current_accuracy`.
- Returns node name string; LangGraph uses this for conditional edge.

Graph construction:
```python
workflow = StateGraph(DocumentState)
workflow.add_node("classify", self._classify_node)
...
workflow.add_edge("classify", "extract")
```
- Adds all nodes first, then edges.
- Conditional edge from validate:
  - `repair` path loops back to validate.
  - `redact` path moves to completion.

Compile and execute:
```python
self.compiled_graph = self.graph.compile(checkpointer=memory)
```
- Checkpointer allows state snapshots and replay behavior.

```python
for state in self.compiled_graph.stream(initial_state, config):
  final_state = state
```
- `stream()` yields incremental node outputs.
- Last streamed state is used as final output.

Compatibility shim:
```python
class HITLWorkflow:
```
- Does not own graph logic anymore.
- Delegates start/resume to `SupervisorAgent` for policy orchestration.

---

### 6.3 `agents/classifier_agent.py` line-by-line understanding

Imports:
```python
import time, json
from prompts import CLASSIFIER_PROMPT, CLASSIFIER_SYSTEM_PROMPT
```
- `time` for latency measurement.
- `json` for parsing strict machine-readable output.
- prompt constants keep instructions centralized.

Excerpt function:
```python
def _prepare_text_excerpt(self, text, max_chars=600):
```
- Keeps prompt short and cheap.
- Uses start + end excerpt to preserve title/footer signals.

JSON parser:
```python
result = json.loads(content)
```
- First attempt direct parse.
- Then fallback extracts JSON from markdown fences.

Type normalization:
```python
return DocumentType(doc_type_clean)
```
- Strict enum conversion first.
- Alias table fallback maps human variants (`cv`, `marksheet`, `offer letter`).

Main classify method:
```python
response = llm_client.generate(...)
```
- Sends short prompt with deterministic settings.
- Uses fast Groq model for low latency.

```python
classification_result = ClassificationResult(...)
```
- Pydantic object captures doc_type/confidence/reasoning/timestamp.

```python
state["trace_log"].append(ResponsibleAILog(...))
```
- Stores full trace: provider, model, tokens, prompts, raw output, latency.

Error path:
```python
state["doc_type"] = DocumentType.UNKNOWN
```
- Fails safely so downstream nodes can still run.

---

### 6.4 `agents/extractor_agent.py` line-by-line understanding

Why this file is large:
- It has schema map, few-shot examples, normalization, chunking, merging, and robust parsing.

Schema selection:
```python
SCHEMA_MAP = {
  DocumentType.RESUME: ResumeFields,
  ...
}
```
- Chooses expected field structure by doc type.

Few-shot prompting:
```python
FEW_SHOT_EXAMPLES = {...}
```
- Teaches the model output shape via examples.
- Strongly increases extraction stability.

Name normalization:
```python
snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
```
- Converts `candidateName` → `candidate_name`.

Value normalization:
```python
parsed_date = datetime.strptime(data.strip(), fmt)
```
- Attempts many date formats.
- Converts to canonical `YYYY-MM-DD`.

Chunking:
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
```
- Splits long documents to avoid context overflow.
- Overlap preserves boundary context.

Single-chunk extract:
```python
response = llm_client.generate(..., groq_model="llama-3.3-70b-versatile")
repaired = repair_json(content, return_objects=True)
```
- Calls strong model for extraction.
- Repairs malformed JSON automatically.

Merging chunks:
```python
if isinstance(value, list): existing.extend(value)
```
- List fields aggregate across chunks.
- Scalar fields prefer latest non-null value.

State update:
```python
state["extracted_fields"] = normalized_fields
state["extraction_result"] = ExtractionResult(...)
```
- Keeps both raw dict and typed metadata object.

---

### 6.5 `agents/validator_agent.py` line-by-line understanding

This file mixes deterministic checks + semantic checks.

Regex setup:
```python
self.patterns = {"email": re.compile(...), ...}
```
- Fast lightweight format checks.

Alias remap:
```python
if alias_val not in (None, "", [], {}) and canon_val in (None, "", [], {}, None):
  normalized[canonical_key] = alias_val
```
- Prevents false misses when extractor uses alternate key names.

Required field resolution:
```python
priority = profile_required_fields or self.FIELD_PRIORITIES.get(...)
```
- Knowledge profile overrides static defaults.

Accuracy computation:
```python
accuracy = (total - len(missing)) / total
```
- Schema completeness score drives repair loop.

Rule validation:
```python
for field in resolved_priority_fields[:3]:
```
- Top-priority fields checked first (high signal).

LLM validation prompt:
```python
prompt = VALIDATOR_PROMPT.format(...)
```
- Injects schema fields, priority fields, extracted JSON, and knowledge context.

Main flow in `validate(state)`:
1. Normalize fields.
2. Run rule pre-check + schema validation.
3. Fast-path if enough quality already.
4. Else call LLM semantic validation.
5. Enforce accuracy gate.
6. Write `ValidationResult` + `needs_repair` + `current_accuracy`.

Design detail:
- For custom doc types, validator uses `UNKNOWN` fallback behavior to avoid wrong hard schema penalties.

---

### 6.6 `agents/self_repair_node.py` line-by-line understanding

Config read:
```python
self.max_attempts = int(self.env.get('workflow', 'max_repair_attempts', fallback=3))
```
- Pulls repair budget from config.

Decision logic:
```python
if non_null_fields < 3: return True
if current_accuracy < 0.90: return True
```
- Low information or low quality triggers re-extraction mode.

Prompt choice:
- `SELF_REPAIR_RE_EXTRACTION_PROMPT` for full re-extract.
- `SELF_REPAIR_PROMPT` for targeted fixes.

State effects:
```python
state["repair_attempts"] += 1
state["extracted_fields"] = repaired_fields
state["validation_status"] = ValidationStatus.REPAIRED.value
```
- Updates counters and fields for next validation pass.

---

### 6.7 `agents/redactor_agent.py` line-by-line understanding

Initialization:
- Builds Presidio registry.
- Loads custom Indian recognizers.

Entity mapping:
```python
PRESIDIO_TO_PII_TYPE = {"EMAIL_ADDRESS": PIIType.EMAIL, ...}
```
- Normalizes external entity labels to internal schema.

Detection layers:
1. Presidio detection.
2. Regex detectors for gender/custom IDs/multi-line addresses.
3. LLM-based contextual detection.

Deduping logic:
```python
existing_texts = {pii.original_text.lower() for pii in presidio_pii}
```
- Avoids duplicate replacements and substring conflicts.

Redaction application:
```python
sorted(..., key=lambda x: len(x.original_text), reverse=True)
```
- Replaces longer spans first.

Metrics:
- precision/recall computed against ground-truth PII if available, else schema-derived references.

---

### 6.8 `utils/llm_client.py` line-by-line understanding

This is the largest “infrastructure” file.

Provider enum:
```python
class LLMProvider(str, Enum): ...
```
- Central provider naming used across whole codebase.

Constructor responsibilities:
- initialize cache settings,
- initialize clients (Groq A/B/C, Bedrock, HF, Ollama, local model),
- setup round-robin key control and cooldown maps.

Cache table:
```sql
CREATE TABLE IF NOT EXISTS llm_cache (...)
```
- SQLite key-value cache for prompt results.

Groq resilience:
- `_groq_attempt_plan(...)` picks key order.
- `_set_groq_key_cooldown(...)` temporarily disables rate-limited key.
- `_invoke_groq_with_fallback(...)` iterates available keys.

Provider invocations:
- `_invoke_groq`, `_invoke_bedrock_claude`, `_invoke_nova`, `_invoke_huggingface`, `_invoke_ollama`, `_invoke_llama`.

Master method:
```python
def generate(...):
```
Flow:
1. Build cache key.
2. Return cached response if available.
3. Honor forced provider if requested.
4. Otherwise fallback chain execution.
5. Save successful response to cache.
6. Raise error if all providers fail.

Key parameters:
- `force_provider`: bypass normal fallback policy.
- `groq_model`: model override per call.
- `groq_key`: preferred Groq key route.

---

### 6.9 `api/main.py` line-by-line understanding

Core role:
- Converts HTTP requests into supervisor/workflow method calls and standardized JSON responses.

App setup:
```python
app = FastAPI(...)
app.add_middleware(CORSMiddleware, ...)
```
- Creates API app and enables browser cross-origin requests.

Monitoring middleware:
```python
@app.middleware("http")
```
- Wraps every request for latency + observability tracing.

Startup hook:
```python
@app.on_event("startup")
```
- Compiles workflows and warms heavy components.

Standard process endpoint:
```python
@app.post("/process", response_model=ProcessResponse)
```
Main steps:
1. Validate file path.
2. Execute `supervisor_agent.process_document(...)` in executor thread.
3. Extract result segments from final state.
4. Build normalized response payload.
5. End observability run.

HITL endpoints:
- `/process/start`: runs until first interrupt.
- `/thread/{thread_id}/resume`: resumes with human input.
- `/process/auto`: policy-based mode selection.

Knowledge endpoints:
- list/register/refresh schema profiles.

Visualization endpoints:
- graph mermaid, execution path, trace diagram.

Serialization helpers:
- `_serialize_trace(...)` converts Pydantic log objects into plain dicts.
- `_build_process_response_from_state(...)` centralizes response shape consistency.

---

### 6.10 `streamlit_app.py` line-by-line understanding

Purpose:
- Human-facing dashboard that calls API and renders outputs clearly.

Network helpers:
- `check_api_health`, `process_document`, HITL start/resume helpers.

Rendering helpers:
- `display_classification_results`
- `display_extraction_results`
- `display_validation_results`
- `display_redaction_results`
- `display_metrics`
- `display_knowledge_lookup_info`

Interesting logic:
- extraction accuracy label changes for custom doc types (no schema baseline).
- validation can demote errors to warnings when accuracy is strong.
- metrics chart rebuilt from trace log to include all stages.

---

## 7) How to use this Part 2 effectively

1. Open one file at a time in the order above.
2. For each method, identify:
   - inputs from `state`,
   - writes to `state`,
   - external dependencies (`llm_client`, schemas, prompt constants).
3. Run one sample document and watch logs to map runtime behavior to these notes.
4. After understanding, only then start cloud refactor (provider abstraction is easier once flow is clear).

---

## 8) Part 3 — State-transition tables (before/after each node)

Use these tables while debugging. At each node, check **what keys are expected before**, **what gets written after**, and **what controls routing**.

### 8.1 Standard workflow transitions (`graph/workflow.py`)

| Step | Node | Required Input Keys (from state) | Main Output Keys Written | Route Decision |
|---|---|---|---|---|
| 1 | `classify` | `raw_text`, `file_path` | `doc_type`, `classification_result`, `agent_timings[ClassifierAgent]`, `trace_log[]` | Always to `extract` |
| 2 | `extract` | `doc_type`, `raw_text` | `extracted_fields`, `extraction_result`, `agent_timings[ExtractorAgent]`, `trace_log[]` | Always to `validate` |
| 3 | `validate` | `doc_type`, `extracted_fields`, `raw_text` | `validation_status`, `validation_result`, `needs_repair`, `current_accuracy`, `missing_schema_fields`, `agent_timings[ValidatorAgent]`, `trace_log[]` | Conditional: `repair` or `redact` |
| 4A | `repair` | `extracted_fields`, `validation_result`, `doc_type`, `raw_text`, `repair_attempts` | `extracted_fields` (repaired), `repair_attempts`, `validation_status=repaired`, `needs_repair=False`, `trace_log[]`, `repair_history[]` | Back to `validate` |
| 4B | `redact` | `raw_text`, `extracted_fields`, optional `ground_truth_pii` | `redacted_text`, `redaction_result`, `agent_timings[RedactorAgent]`, `trace_log[]` | Always to `report` |
| 5 | `report` | all prior stage outputs | `metrics`, `success`, `agent_timings[ReporterAgent]`, report files in `reports/` | `END` |

---

### 8.2 Validation routing decision table

Routing function: `_should_repair(state)` in `graph/workflow.py`.

| Condition | Expression | Route |
|---|---|---|
| Validation says repair needed | `needs_repair == True` and attempts left | `repair` |
| Accuracy below threshold | `current_accuracy < 0.80` and attempts left | `repair` |
| Repair budget exhausted | `repair_attempts >= max_repair_attempts` | `redact` |
| Good quality | `needs_repair == False` and `current_accuracy >= 0.80` | `redact` |

---

### 8.3 HITL supervisor workflow transitions (`agents/supervisor_agent.py`)

| Step | Node | Required Input Keys | Main Output Keys Written | Route Decision |
|---|---|---|---|---|
| 1 | `classify` | `raw_text`, `file_path` | `doc_type`, `classification_result`, `trace_log[]` | to `supervise_classification` |
| 2 | `supervise_classification` | `classification_result`, `doc_type` | `supervisor_*classification*`, `hitl_required=True` | Always `human_review_classify` |
| 3A | `human_review_classify` (approved/corrected) | current classification payload | `hitl_type=classify`, `hitl_resolution`, optional `doc_type/custom_doc_type override` | to `extract` |
| 3B | `human_review_classify` (rejected) | same as above | `errors += rejection`, `success=False` | to `report` |
| 4 | `extract` | resolved `doc_type` or `custom_doc_type`, `raw_text` | `extracted_fields`, `extraction_result` | to `validate` |
| 5 | `validate` | `extracted_fields`, `doc_type`, `raw_text` | `validation_result`, `needs_repair`, `current_accuracy` | to `supervise_validation` |
| 6 | `supervise_validation` | `needs_repair`, `current_accuracy`, `repair_attempts`, `custom_doc_type` | `supervisor_*validation* decision+reason` | `repair` / `human_review_extract` / `redact` |
| 7A | `repair` | low-quality extraction state | repaired extraction fields, counters | back to `validate` |
| 7B | `human_review_extract` approved/corrected | extracted fields + validator output | `hitl_type=extract`, `hitl_resolution`, optional `hitl_corrections`, `validation_result=valid` | to `redact` |
| 7C | `human_review_extract` rejected | same | `errors += rejection`, `success=False` | to `report` |
| 8 | `redact` | `raw_text`, `extracted_fields` | `redacted_text`, `redaction_result` | to `report` |
| 9 | `report` | final assembled state | `metrics`, `success` | `END` |

---

### 8.4 Human interrupt payload contracts (HITL)

#### Classification checkpoint (`hitl_type = classify`)

**Interrupt payload sent to UI**
- `hitl_type`
- `doc_type`
- `confidence`
- `reasoning`
- `message`

**Resume payload expected by backend**
```json
{
  "resolution": "approved" | "corrected" | "rejected",
  "doc_type_override": "resume" | "financial_document" | "custom_type" (optional)
}
```

#### Extraction checkpoint (`hitl_type = extract`)

**Interrupt payload sent to UI**
- `hitl_type`
- `doc_type`
- `extracted_fields`
- `accuracy`
- `missing_fields`
- `validation_errors`
- `message`

**Resume payload expected by backend**
```json
{
  "resolution": "approved" | "corrected" | "rejected",
  "corrections": {"field": "value"} (optional)
}
```

---

### 8.5 Node-by-node invariant checklist (quick debug)

Use this as a strict checklist while debugging a failed run:

1. After `classify`:
  - `doc_type` must not be missing.
  - `classification_result` should exist even on fallback (UNKNOWN case).

2. After `extract`:
  - `extracted_fields` must be dict (possibly empty on failure).
  - `extraction_result` should exist.

3. After `validate`:
  - `validation_result` must exist.
  - `current_accuracy` should be a float in `[0,1]`.
  - `missing_schema_fields` should be list (possibly empty).

4. After `repair`:
  - `repair_attempts` must increment.
  - `extracted_fields` should be updated.

5. After `redact`:
  - `redaction_result` must exist.
  - `redacted_text` should exist (fallback: original text).

6. After `report`:
  - `metrics` must exist.
  - `success` must be boolean.

---

### 8.6 API-level state/response mapping (`api/main.py`)

| State key | API response field |
|---|---|
| `doc_type` / `custom_doc_type` | `doc_type` |
| `classification_result.confidence` | `confidence` |
| `classification_result.reasoning` | `reasoning` |
| `extracted_fields` | `extracted_fields` |
| `validation_result.*` | `validation` object |
| `redaction_result.*` | `redaction` object |
| `metrics` | `metrics` |
| `errors` | `errors` |
| `trace_log` | `trace_log` (serialized) |
| `supervisor_*` fields | `supervisor_policy` |

---

### 8.7 Common failure signatures and where to look

| Symptom | Most likely file | What to inspect first |
|---|---|---|
| Wrong doc type | `agents/classifier_agent.py`, `prompts.py` | alias map + classifier prompt rules |
| Empty extraction | `agents/extractor_agent.py`, `utils/document_loader.py` | chunking + parsing + OCR/text loader |
| Endless repair loop | `agents/validator_agent.py`, `graph/workflow.py` | `current_accuracy`, threshold, `max_repair_attempts` |
| PII not masked | `agents/redactor_agent.py` | detector merge, `_validate_llm_pii`, replacement order |
| HITL resume not progressing | `agents/human_review_agent.py`, `agents/supervisor_agent.py`, `api/main.py` | payload contract and route functions |
| API success=false unexpectedly | `agents/reporter_agent.py` | `_compute_workflow_success` criteria |

---

### 8.8 Safe extension strategy (for AWS cloud refactor)

When introducing cloud adapters, preserve these state contracts:
- Do **not** change `DocumentState` key names unless absolutely required.
- Keep agent write-keys unchanged so API serializers continue to work.
- Add provider-specific metadata under nested keys in `context_data`, not top-level state.

Recommended change order:
1. Add adapter interfaces.
2. Swap internals of loaders/clients.
3. Keep agent signatures untouched.
4. Re-run state-transition checklist after each step.

---

## 9) Part 4 — Concrete sample state snapshots (before/after each node)

This section gives realistic JSON examples you can compare with runtime logs.

> Note: Values are representative examples for learning/debugging. Actual token counts, confidence, and latency vary by model and input document quality.

---

### 9.1 Initial state (before graph starts)

```json
{
  "file_path": "data/samples/resume_01.pdf",
  "raw_text": "JOHN DOE\nEmail: john@email.com\nPhone: +1 555-123-4567\n...",
  "ground_truth_pii": null,
  "doc_type": null,
  "classification_result": null,
  "extracted_fields": null,
  "extraction_result": null,
  "validation_status": null,
  "validation_result": null,
  "needs_repair": false,
  "repair_attempts": 0,
  "current_accuracy": 0.0,
  "missing_schema_fields": [],
  "redacted_text": null,
  "redaction_result": null,
  "metrics": null,
  "trace_log": [],
  "agent_timings": {},
  "errors": [],
  "success": false,
  "hitl_required": false,
  "hitl_type": null,
  "hitl_resolution": null,
  "hitl_corrections": null,
  "custom_doc_type": null
}
```

---

### 9.2 After `classify` node

#### Before
```json
{
  "doc_type": null,
  "classification_result": null
}
```

#### After
```json
{
  "doc_type": "resume",
  "classification_result": {
    "doc_type": "resume",
    "confidence": 0.95,
    "reasoning": "Work experience and skills sections are present",
    "timestamp": "2026-03-20T10:12:21.100Z"
  },
  "agent_timings": {
    "ClassifierAgent": 0.83
  },
  "trace_log": [
    {
      "agent_name": "ClassifierAgent",
      "llm_provider": "groq",
      "llm_model_used": "llama-3.1-8b-instant",
      "tokens_input": 128,
      "tokens_output": 34,
      "error_occurred": false
    }
  ]
}
```

---

### 9.3 After `extract` node

#### Before
```json
{
  "doc_type": "resume",
  "extracted_fields": null,
  "extraction_result": null
}
```

#### After
```json
{
  "extracted_fields": {
    "candidate_name": "John Doe",
    "email": "john@email.com",
    "phone": "+1 555-123-4567",
    "address": "New York, NY",
    "summary": "Software engineer with 4 years of backend experience.",
    "skills": ["Python", "FastAPI", "AWS", "SQL"],
    "education": [
      {
        "degree": "B.Tech Computer Science",
        "institution": "XYZ University",
        "graduation_date": "2022-06-01",
        "gpa": 8.7
      }
    ],
    "work_experience": [
      {
        "job_title": "Backend Engineer",
        "employer": "Acme Tech",
        "start_date": "2022-07-01",
        "end_date": null,
        "responsibilities": ["Built APIs", "Optimized SQL queries"]
      }
    ]
  },
  "extraction_result": {
    "doc_type": "resume",
    "confidence": 0.85,
    "chunk_count": 1
  },
  "agent_timings": {
    "ClassifierAgent": 0.83,
    "ExtractorAgent": 2.41
  }
}
```

---

### 9.4 After `validate` node (valid path)

#### Before
```json
{
  "extracted_fields": {
    "candidate_name": "John Doe",
    "email": "john@email.com",
    "phone": "+1 555-123-4567"
  },
  "validation_result": null
}
```

#### After
```json
{
  "validation_status": "valid",
  "validation_result": {
    "status": "valid",
    "is_valid": true,
    "errors": [],
    "warnings": ["JSON Schema validation passed"]
  },
  "needs_repair": false,
  "current_accuracy": 0.91,
  "missing_schema_fields": ["linkedin_url", "certifications", "languages"],
  "agent_timings": {
    "ValidatorAgent": 0.72
  }
}
```

---

### 9.5 After `validate` node (needs repair path)

#### Before
```json
{
  "extracted_fields": {
    "candidate_name": "John Doe",
    "email": "john@email.com"
  },
  "repair_attempts": 0
}
```

#### After
```json
{
  "validation_status": "invalid",
  "validation_result": {
    "status": "invalid",
    "is_valid": false,
    "errors": [
      "Missing required field: phone",
      "Missing required field: work_experience"
    ],
    "warnings": ["Sparse extraction: only 2 fields populated"]
  },
  "needs_repair": true,
  "current_accuracy": 0.42,
  "missing_schema_fields": ["phone", "work_experience", "education", "skills"]
}
```

Routing result from workflow: `repair`.

---

### 9.6 After `repair` node

#### Before
```json
{
  "repair_attempts": 0,
  "needs_repair": true,
  "current_accuracy": 0.42,
  "missing_schema_fields": ["phone", "work_experience", "education", "skills"]
}
```

#### After
```json
{
  "repair_attempts": 1,
  "validation_status": "repaired",
  "needs_repair": false,
  "extracted_fields": {
    "candidate_name": "John Doe",
    "email": "john@email.com",
    "phone": "+1 555-123-4567",
    "skills": ["Python", "FastAPI"],
    "education": [
      {
        "degree": "B.Tech Computer Science",
        "institution": "XYZ University",
        "graduation_date": "2022-06-01",
        "gpa": 8.7
      }
    ],
    "work_experience": [
      {
        "job_title": "Backend Engineer",
        "employer": "Acme Tech",
        "start_date": "2022-07-01",
        "end_date": null,
        "responsibilities": ["Built APIs"]
      }
    ]
  },
  "repair_history": [
    {
      "attempt": 1,
      "mode": "re_extraction"
    }
  ]
}
```

Then control goes back to `validate` for re-check.

---

### 9.7 After `redact` node

#### Before
```json
{
  "raw_text": "John Doe\nEmail: john@email.com\nPhone: +1 555-123-4567\nAddress: 12 Main Street, New York, NY 10001",
  "redaction_result": null
}
```

#### After
```json
{
  "redacted_text": "[NAME_REDACTED]\nEmail: [EMAIL_ADDRESS_REDACTED]\nPhone: [PHONE_NUMBER_REDACTED]\nAddress: [ADDRESS_REDACTED]",
  "redaction_result": {
    "pii_count": 4,
    "precision": 0.93,
    "recall": 0.95,
    "pii_detections": [
      {"pii_type": "name", "original_text": "John Doe", "detection_source": "presidio"},
      {"pii_type": "email", "original_text": "john@email.com", "detection_source": "presidio"},
      {"pii_type": "phone", "original_text": "+1 555-123-4567", "detection_source": "presidio"},
      {"pii_type": "address", "original_text": "12 Main Street, New York, NY 10001", "detection_source": "regex"}
    ]
  }
}
```

---

### 9.8 After `report` node (final state)

#### Before
```json
{
  "metrics": null,
  "success": false
}
```

#### After
```json
{
  "metrics": {
    "extraction_accuracy": 0.92,
    "pii_recall": 0.95,
    "pii_precision": 0.93,
    "workflow_success": true,
    "total_processing_time": 6.44,
    "agent_latencies": {
      "ClassifierAgent": 0.83,
      "ExtractorAgent": 2.41,
      "ValidatorAgent": 0.72,
      "RedactorAgent": 1.36,
      "ReporterAgent": 0.22
    },
    "error_count": 0,
    "retry_count": 0
  },
  "success": true
}
```

Files written by reporter (example):
- `reports/responsible_ai_log_resume_01_20260320_101221.json`
- `reports/responsible_ai_log_resume_01_20260320_101221.csv`
- `reports/metrics_report_resume_01_20260320_101221.json`

---

### 9.9 HITL snapshot sequence (`/process/start` + `/resume`)

#### A) `/process/start` returns interrupted (classification)

```json
{
  "thread_id": "hitl_20260320_101500_123456",
  "status": "interrupted",
  "interrupt_data": {
    "hitl_type": "classify",
    "doc_type": "resume",
    "confidence": 0.81,
    "reasoning": "Found work experience section",
    "message": "Please confirm the document type detected by the AI classifier."
  }
}
```

#### B) Human sends correction

Request to `/thread/{thread_id}/resume`:
```json
{
  "resolution": "corrected",
  "doc_type_override": "job_offer"
}
```

#### C) Graph continues and may interrupt at extraction review

```json
{
  "thread_id": "hitl_20260320_101500_123456",
  "status": "interrupted",
  "interrupt_data": {
    "hitl_type": "extract",
    "doc_type": "job_offer",
    "accuracy": 0.67,
    "missing_fields": ["salary", "start_date"],
    "validation_errors": ["Missing required field: salary"],
    "message": "Extraction accuracy is 67% (below 80% threshold). Please review and correct the extracted fields."
  }
}
```

#### D) Human approves corrected fields

```json
{
  "resolution": "corrected",
  "corrections": {
    "candidate_name": "Jane Smith",
    "company_name": "Acme Inc",
    "position_title": "Software Engineer",
    "salary": "120000",
    "start_date": "2026-04-01"
  }
}
```

#### E) Final response

```json
{
  "thread_id": "hitl_20260320_101500_123456",
  "status": "complete",
  "result": {
    "success": true,
    "doc_type": "job_offer",
    "classification_path": "hitl_corrected_classification"
  }
}
```

---

### 9.10 Minimal JSON templates you can reuse for debugging

#### Template: standard `/process` response core
```json
{
  "success": true,
  "doc_type": "resume",
  "confidence": 0.95,
  "extracted_fields": {},
  "validation": {"status": "valid", "is_valid": true, "errors": [], "warnings": []},
  "redaction": {"pii_count": 0, "precision": 1.0, "recall": 1.0, "pii_detections": []},
  "metrics": {"extraction_accuracy": 0.9, "workflow_success": true},
  "errors": []
}
```

#### Template: interrupted HITL response core
```json
{
  "thread_id": "hitl_...",
  "status": "interrupted",
  "interrupt_data": {
    "hitl_type": "classify|extract",
    "message": "..."
  }
}
```

---

## 10) Part 5 — Error playbook (top 20 issues, root cause, exact fixes)

Use this section as your first-response runbook when something breaks.

### 10.1 Fast triage order (always follow)

1. Check API health: `/health`.
2. Inspect latest response `errors` + `trace_log` in API output.
3. Identify failing stage from `trace_log.agent_name`.
4. Open corresponding file/function from the table below.
5. Re-run one document and compare with Part 4 snapshots.

---

### 10.2 Top 20 errors and fixes

| # | Error / Symptom | Root cause | Exact file + function | Quick fix steps |
|---|---|---|---|---|
| 1 | `File not found` from `/process` | Relative/invalid path | `api/main.py` → `process_document` | Use absolute path or verify `settings.PROJECT_ROOT` join logic; ensure file exists before request |
| 2 | `No text extracted from document` | OCR/loader couldn’t extract text | `graph/workflow.py` → `process_document`; `utils/document_loader.py` | Test file with `document_loader.load_and_extract_text`; if scanned PDF, check OCR dependencies |
| 3 | Classifier always returns `unknown` | Prompt/alias mismatch or LLM parse failure | `agents/classifier_agent.py` → `_parse_llm_response`, `_normalize_doc_type` | Add alias mapping, verify prompt, log raw response, keep JSON strictness |
| 4 | JSON parse error in classifier | LLM returned non-JSON text | `agents/classifier_agent.py` → `_parse_llm_response` | Keep markdown-fence extraction branch; add stricter system prompt; reduce model temperature |
| 5 | Empty `extracted_fields` | Chunk prompt failed or parse failed | `agents/extractor_agent.py` → `_extract_from_chunk`, `extract` | Inspect `response["content"]`; validate `repair_json` result is dict; tune chunk size/overlap |
| 6 | Wrong field names (camelCase etc.) | Model emits mixed key conventions | `agents/extractor_agent.py` → `_normalize_field_names` | Keep camel→snake normalization and alias post-pass enabled |
| 7 | Important fields missing even after extract | Few-shot or schema targeting weak | `agents/extractor_agent.py` + `prompts.py` | Improve doc-type examples; include missing fields in prompt and validator priority list |
| 8 | Endless repair loop | `needs_repair` stays true, low accuracy never recovers | `graph/workflow.py` → `_should_repair`; `validator_agent.py` thresholds | Verify `max_repair_attempts`; confirm `current_accuracy` updates; lower threshold only with care |
| 9 | Repair runs but state unchanged | Repaired JSON parse failed or no overwrite | `agents/self_repair_node.py` → `repair` | Log raw repair output; ensure parsed dict assigned to `state["extracted_fields"]` |
| 10 | Validation too strict on custom docs | Applying wrong schema to custom type | `agents/validator_agent.py` → `validate` | Ensure `custom_doc_type` path uses UNKNOWN-compatible validation and profile-based required fields |
| 11 | Validation says invalid despite good extraction | Alias mismatch or schema mismatch | `agents/validator_agent.py` → `_normalize_extracted_fields`, `_compute_accuracy` | Add aliases for observed field variants; verify schema model keys |
| 12 | PII not redacted though visible | Detection filtered out or replacement mismatch | `agents/redactor_agent.py` → `_validate_llm_pii`, `_redact_text` | Relax filter for that entity type, check original text exactness, keep longest-first replacement |
| 13 | Too many false-positive redactions | Over-aggressive PII validation rules | `agents/redactor_agent.py` → `_validate_llm_pii`, `_validate_pii_detection` | Tighten regex/entity checks; increase confidence cutoffs for noisy types |
| 14 | `Presidio initialization failed` warning | Missing Presidio/spaCy resources | `agents/redactor_agent.py` → `__init__` | Install Presidio deps and language model; fallback to LLM-only still works but slower/less deterministic |
| 15 | HITL start works but resume does nothing | Wrong resume payload shape | `api/main.py` → `resume_document_processing`; `human_review_agent.py` | Send exact JSON contract (`resolution`, optional `doc_type_override`/`corrections`) |
| 16 | HITL thread cannot resume | Wrong `thread_id` or checkpoint missing | `agents/supervisor_agent.py` → `_run_until_pause`, `resume_processing` | Reuse exact thread_id from interrupted response; ensure graph compiled with checkpointer |
| 17 | API latency spikes unexpectedly | Cold starts or heavy model fallback | `api/main.py` startup + `utils/llm_client.py` | Keep warm-up on startup; verify Groq available; inspect fallback path in logs |
| 18 | LLM rate-limit bursts | Groq key cooldown/rotation misconfigured | `utils/llm_client.py` → `_invoke_groq_with_fallback` | Ensure keys A/B/C configured; verify cooldown timestamps and round-robin selection |
| 19 | Repeated identical requests still slow | Cache disabled or key mismatch | `utils/llm_client.py` → `_cache_get/_cache_set` | Enable cache in config; verify key inputs stable (prompt/system/max_tokens/temp) |
| 20 | Reports missing/empty | Reporter failed writing files | `agents/reporter_agent.py` → `_save_json_report`, `_save_csv_report`, `generate_report` | Ensure `reports/` exists and writable; catch serialization issues; inspect `state["errors"]` |

---

### 10.3 Error-to-log pattern mapping

| Log pattern | Stage | Meaning | Next action |
|---|---|---|---|
| `Classification failed` | classify | LLM parse/provider error | Inspect raw classifier output in trace log |
| `Could not parse JSON from response` | extract/repair | malformed model output | Check `repair_json` path and prompt format |
| `Accuracy ... below ... threshold` | validate | schema completion low | verify missing fields list and repair mode |
| `Routing to repair node` repeatedly | workflow router | loop condition still true | inspect `repair_attempts`, threshold, extracted fields delta |
| `Presidio ... failed` | redact | rule-based detector unavailable | install dependency or accept LLM-only fallback |
| `All configured LLM providers failed` | llm client | provider outage/misconfig | test each provider init path and credentials |

---

### 10.4 Exact debug commands/checks (manual checklist)

1. Verify API running:
  - call `/health` from browser or Streamlit helper.
2. Trigger one deterministic test input:
  - use same file repeatedly.
3. Compare outputs with Part 4 snapshots:
  - check keys, not only values.
4. Validate these state keys in order:
  - `doc_type` → `extracted_fields` → `validation_result` → `redaction_result` → `metrics`.
5. If mismatch appears:
  - go to matching file/function in table 10.2 and patch there first.

---

### 10.5 Safe fix workflow (to avoid regressions)

1. Reproduce on one file.
2. Fix the smallest relevant function.
3. Re-run same file and confirm transition snapshots.
4. Run `tests/test_agents.py` for nearby coverage.
5. Only then test broad datasets.

---

### 10.6 Severity guide

| Severity | Definition | Action |
|---|---|---|
| P0 | Crash/data loss/security issue | hotfix immediately, block releases |
| P1 | Core stage broken (classify/extract/validate/redact) | fix same day |
| P2 | Quality regression with workaround | fix in next sprint |
| P3 | Cosmetic/reporting inconsistency | backlog |

---

## 11) Part 6 — Cloud migration playbook (local → AWS)

This playbook is designed to migrate your current local stack safely without breaking existing behavior.

---

### 11.1 Target AWS architecture (practical)

1. **Client/UI**
  - Streamlit UI (can remain local first, later host on ECS/App Runner).
2. **API Layer**
  - API Gateway → Lambda (thin upload/status endpoints) or direct FastAPI on ECS.
3. **Workflow Workers**
  - ECS Fargate service runs LangGraph pipeline (`SupervisorAgent` + agents).
4. **Storage**
  - S3 for uploads, processed artifacts, and reports.
5. **OCR**
  - Textract for scanned docs (fallback: local OCR adapter).
6. **Cache**
  - ElastiCache Redis for response/job cache (fallback: SQLite/local cache).
7. **Vector/Knowledge Retrieval**
  - OpenSearch (or keep FAISS initially), with staged migration.
8. **Observability**
  - LangSmith + CloudWatch logs/metrics.
9. **Secrets**
  - AWS Secrets Manager for API keys (Groq/LangSmith/etc).

---

### 11.2 Migration principles (do not break local)

1. Keep local mode fully runnable.
2. Introduce adapters behind interfaces.
3. Add provider selection via config/env only.
4. Do not change `DocumentState` contract unless necessary.
5. Migrate one capability at a time and validate after each phase.

---

### 11.3 File-level refactor plan (exact split)

Create these folders/files:

1. `services/storage_service.py`
  - Interface: `put_file`, `get_file`, `exists`, `get_signed_url`.
2. `services/ocr_service.py`
  - Interface: `extract_text(file_ref)`.
3. `services/cache_service.py`
  - Interface: `get`, `set`, `delete`, `ttl`.
4. `services/vector_store_service.py`
  - Interface: `index`, `search`, `refresh`.
5. `adapters/local_storage.py`
6. `adapters/s3_storage.py`
7. `adapters/local_ocr.py`
8. `adapters/textract_ocr.py`
9. `adapters/local_cache.py`
10. `adapters/redis_cache.py`
11. `adapters/faiss_vector_store.py`
12. `adapters/opensearch_vector_store.py`
13. `workers/ecs_worker.py`
14. `lambda_handlers/start_job.py`
15. `lambda_handlers/get_status.py`
16. `lambda_handlers/submit_review.py`

Minimal changes in existing files:
- `utils/config.py`: add cloud provider switches and AWS resource names.
- `api/main.py`: route to job-based async mode when cloud profile enabled.
- `utils/knowledge_lookup.py`: optional backend switch (FAISS/OpenSearch).
- `ocr/processor.py` / `utils/document_loader.py`: wire to `OCRService` interface.

---

### 11.4 New config keys to add (`config.ini` + env)

Add under `[stack]` and new sections:

```ini
[stack]
profile = local
execution_mode = sync
storage_provider = local_fs
ocr_provider = tesseract
cache_provider = sqlite
vector_provider = faiss

[aws]
region = ap-south-1
s3_bucket = agentic-doc-processor-dev
ecs_cluster = agentic-doc-cluster
ecs_service = agentic-doc-worker
redis_endpoint =
opensearch_endpoint =
textract_enabled = true
```

Env variables (Secrets Manager or `.env` during dev):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `GROQ_API_KEY` (+ optional B/C)
- `LANGSMITH_API_KEY`

---

### 11.5 Migration phases with acceptance criteria

#### Phase 0 — Baseline freeze

Actions:
1. Create branch `cloud-stack`.
2. Save baseline metrics from local runs.

Acceptance:
- Existing local tests pass.
- Baseline metrics report saved.

#### Phase 1 — Interface extraction (no behavior change)

Actions:
1. Add service interfaces and local adapters.
2. Replace direct calls with interface calls.

Acceptance:
- No output diff for same test input.
- `DocumentState` transitions unchanged.

#### Phase 2 — S3 storage adapter

Actions:
1. Implement `S3StorageAdapter`.
2. Store uploads/reports in S3 when profile=cloud.

Acceptance:
- Upload/read/report roundtrip works with S3.
- Local mode still writes to local filesystem.

#### Phase 3 — Textract OCR adapter

Actions:
1. Implement `TextractOCRAdapter`.
2. Use Textract for PDFs/images in cloud profile.

Acceptance:
- OCR quality on scanned sample is at least baseline.
- Fallback to local OCR works if Textract fails.

#### Phase 4 — Redis cache adapter

Actions:
1. Implement `RedisCacheAdapter`.
2. Route LLM/job cache to Redis in cloud profile.

Acceptance:
- Cache hit path observed in logs.
- No cache-related crashes on reconnect/restart.

#### Phase 5 — Vector store migration (optional staged)

Actions:
1. Keep FAISS first in cloud for low risk.
2. Add OpenSearch adapter and toggle by config.

Acceptance:
- `knowledge_lookup` returns equivalent profile for key doc types.

#### Phase 6 — Async execution mode

Actions:
1. Introduce job queue semantics (start/status/resume).
2. Move heavy processing to ECS worker.

Acceptance:
- `/process/start` and `/resume` work end-to-end via worker.
- Thread checkpoint behavior unchanged.

#### Phase 7 — Production hardening

Actions:
1. CloudWatch alarms, DLQ strategy, retry policies.
2. IAM least-privilege policies.
3. Cost guardrails.

Acceptance:
- Error rate and latency within targets.
- Budget alarms functional.

---

### 11.6 API contract strategy during migration

Keep response shape stable:
- `ProcessResponse` fields unchanged.
- HITL payload contracts unchanged.
- Add only optional metadata fields when needed.

Why:
- Streamlit and any external client continue working without rewrite.

---

### 11.7 Suggested code diffs (high-level)

1. Replace direct OCR call sites with:
  - `ocr_service.extract_text(file_ref)`.
2. Replace direct file writes with:
  - `storage_service.put_file(path, bytes)`.
3. Replace local cache calls with:
  - `cache_service.get/set(...)`.
4. Keep business logic in agents unchanged.

This keeps migration focused on infra, not model logic.

---

### 11.8 Validation checklist after each phase

Run after every migration step:

1. One resume sample (standard flow).
2. One scanned PDF (OCR-heavy).
3. One ID document (PII-heavy redaction).
4. One HITL interrupted flow with correction.

Compare:
- `doc_type`
- `extracted_fields` completeness
- `validation_status`
- `pii_count` / `recall`
- `workflow_success`

Use Part 3 + Part 4 tables as expected-state reference.

---

### 11.9 Cost and reliability guardrails

1. Enforce max tokens per stage by doc type.
2. Keep retry budgets bounded.
3. Use queue + worker autoscaling for spikes.
4. Persist all report artifacts to S3.
5. Add fallback chain in cloud exactly as local logic.

---

### 11.10 Rollback strategy

If cloud phase fails:
1. Set `profile=local`.
2. Disable cloud provider toggles.
3. Restart API/workers.
4. Re-run baseline sanity tests.

Because local and cloud share same agent logic, rollback is config-only when adapters are cleanly separated.

---

## 12) Part 7 — Week-by-week execution schedule (with daily milestones)

This schedule assumes one primary developer and one review cycle per week.

---

### 12.1 Overall timeline (8 weeks)

- Week 1: Baseline freeze + branch strategy + interface scaffolding
- Week 2: Storage abstraction + S3 adapter
- Week 3: OCR abstraction + Textract adapter
- Week 4: Cache abstraction + Redis adapter
- Week 5: Vector/knowledge abstraction + OpenSearch optional path
- Week 6: Async worker path (ECS execution mode)
- Week 7: HITL cloud flow stabilization + API compatibility
- Week 8: Hardening, load tests, rollback drills, release prep

---

### 12.2 Week 1 — Baseline and scaffolding

**Goal**: lock current behavior and create migration-safe structure.

Day 1:
- Create `cloud-stack` branch.
- Capture 5 baseline sample runs and save reports.

Day 2:
- Add `services/` interface files (`storage_service`, `ocr_service`, `cache_service`, `vector_store_service`).

Day 3:
- Add local adapters (`adapters/local_storage.py`, `adapters/local_ocr.py`, `adapters/local_cache.py`, `adapters/faiss_vector_store.py`).

Day 4:
- Wire interfaces into existing code paths without changing logic.

Day 5:
- Run regression checks using Part 4 snapshots.
- Commit checkpoint: `phase-1-interface-scaffold`.

**Exit criteria**:
- Local functionality unchanged.
- API response shape unchanged.

---

### 12.3 Week 2 — Storage to S3

**Goal**: support cloud storage while preserving local mode.

Day 1:
- Implement `adapters/s3_storage.py` (`put`, `get`, `exists`, `signed_url`).

Day 2:
- Add config/env toggles for storage provider.

Day 3:
- Route upload/report artifacts through `StorageService` in `api/main.py` and reporter paths.

Day 4:
- Validate upload/download/report persistence in S3 dev bucket.

Day 5:
- Add failure fallback to local storage on provider errors (optional in dev).
- Commit checkpoint: `phase-2-s3-storage`.

**Exit criteria**:
- Cloud profile writes to S3.
- Local profile still uses filesystem.

---

### 12.4 Week 3 — OCR to Textract

**Goal**: switch OCR backend by profile.

Day 1:
- Implement `adapters/textract_ocr.py`.

Day 2:
- Add OCR service factory and profile switch.

Day 3:
- Connect `utils/document_loader.py` / `ocr/processor.py` entry points to OCR service abstraction.

Day 4:
- Test scanned PDF and image documents vs baseline extraction quality.

Day 5:
- Tune fallback behavior for Textract failures/timeouts.
- Commit checkpoint: `phase-3-textract-ocr`.

**Exit criteria**:
- OCR extraction works in cloud mode.
- Fallback path documented and tested.

---

### 12.5 Week 4 — Cache to Redis

**Goal**: externalize cache for distributed workers.

Day 1:
- Implement `adapters/redis_cache.py`.

Day 2:
- Add cache provider switch in config.

Day 3:
- Route LLM cache and transient job state to cache abstraction.

Day 4:
- Validate hit/miss behavior under repeated requests.

Day 5:
- Add reconnect/retry handling for Redis outages.
- Commit checkpoint: `phase-4-redis-cache`.

**Exit criteria**:
- Cache hit path visible in logs.
- Service remains stable during transient cache failure.

---

### 12.6 Week 5 — Knowledge/vector backend migration

**Goal**: keep FAISS path and optionally add OpenSearch.

Day 1:
- Add vector store abstraction adapters for FAISS/OpenSearch.

Day 2:
- Wire `utils/knowledge_lookup.py` to abstraction.

Day 3:
- Seed index in target backend and validate retrieval parity.

Day 4:
- Compare profile resolution outputs for core doc types.

Day 5:
- Keep OpenSearch optional behind feature toggle.
- Commit checkpoint: `phase-5-vector-store`.

**Exit criteria**:
- Same `get_validation_profile` behavior for key cases.

---

### 12.7 Week 6 — Async execution on ECS

**Goal**: move heavy processing off synchronous API path.

Day 1:
- Create `workers/ecs_worker.py` skeleton.

Day 2:
- Add job contract (`start`, `status`, `result` mapping).

Day 3:
- Implement queue polling and supervisor workflow execution in worker.

Day 4:
- Add status/result persistence model and timeout controls.

Day 5:
- End-to-end run: API start → worker execution → status complete.
- Commit checkpoint: `phase-6-ecs-worker`.

**Exit criteria**:
- Long documents process asynchronously with stable status reporting.

---

### 12.8 Week 7 — HITL cloud flow stabilization

**Goal**: preserve interrupt/resume semantics in distributed mode.

Day 1:
- Validate thread/checkpoint persistence strategy.

Day 2:
- Ensure `/process/start` and `/thread/{id}/resume` contracts unchanged.

Day 3:
- Test both classification and extraction interruptions.

Day 4:
- Validate corrected/rejected branches and final report generation.

Day 5:
- Resolve race conditions and retry edge cases.
- Commit checkpoint: `phase-7-hitl-cloud-stable`.

**Exit criteria**:
- HITL workflow parity with local behavior.

---

### 12.9 Week 8 — Hardening and release readiness

**Goal**: make production-safe.

Day 1:
- CloudWatch dashboards and alarm thresholds.

Day 2:
- IAM least-privilege review for all services.

Day 3:
- Load and soak test (multi-document batch).

Day 4:
- Rollback drill (`profile=local` fallback and cloud disable).

Day 5:
- Final regression pack + release notes.
- Commit checkpoint: `phase-8-release-ready`.

**Exit criteria**:
- Meets reliability and quality targets.
- Rollback tested and documented.

---

### 12.10 Commit checkpoint format (recommended)

Use predictable commit names:

1. `phase-1-interface-scaffold`
2. `phase-2-s3-storage`
3. `phase-3-textract-ocr`
4. `phase-4-redis-cache`
5. `phase-5-vector-store`
6. `phase-6-ecs-worker`
7. `phase-7-hitl-cloud-stable`
8. `phase-8-release-ready`

And for each checkpoint include:
- scope summary,
- changed files list,
- known risks,
- test evidence snapshot.

---

### 12.11 Daily done-template (copy/paste)

```text
Date:
Phase:
Tasks completed:
Files changed:
Tests run:
Observed issues:
Next day plan:
```

---

### 12.12 Risk watchlist during schedule

1. API contract drift (breaking Streamlit).
2. HITL checkpoint non-determinism across workers.
3. OCR quality drop moving from local to Textract.
4. Cache inconsistency under retries.
5. Cost spikes from unconstrained model usage.

Mitigation:
- keep strict response schema checks,
- enforce token limits,
- monitor fallback frequency,
- run weekly rollback rehearsal.

---

Part 7 gives execution planning from day one to release. If you want, I can add **Part 8** next: a **task board version** (Kanban-ready checklist with priorities, owners, and estimates).