"""
Configuration Management for Agentic Document Processor
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    SAMPLES_DIR: Path = DATA_DIR / "samples"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Groq API Configuration (PRIMARY - Check https://console.groq.com/docs/models for current list)
    GROQ_API_KEY: Optional[str] = None
    GROQ_API_KEY_B: Optional[str] = None  # Backup Groq key — used when primary hits rate-limit/request-error
    GROQ_MODEL: str = "llama-3.1-8b-instant"  # Most stable Groq model (fast: 0.5-1.5s)
    # IMPORTANT: Check https://console.groq.com/docs/models for currently supported models
    # Many models get decommissioned - always verify before deploying!
    # Common stable models (verify availability):
    #   - "llama-3.3-70b-versatile" (if available - best quality ~92%)
    #   - "llama-3.1-8b-instant" (most stable, fast 0.5-1.5s, ~82-85% acc)
    #   - "llama-3.2-3b-preview" (ultra-fast <1s, lower quality ~75%)
    GROQ_MAX_TOKENS: int = 800  # Reduced from 1024 for faster generation
    GROQ_TEMPERATURE: float = 0.0  # Greedy decoding (deterministic)
    GROQ_TIMEOUT: int = 8  # Fail fast — Groq responds in <3s when healthy; wait max 8s then fall back
    
    # AWS Bedrock Configuration (FALLBACK - Most Reliable)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    BEDROCK_MODEL_ID: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # Claude 3.5 Haiku (cross-region): faster + smarter than 3 Haiku
    BEDROCK_MAX_TOKENS: int = 400  # Default (overridden by agent-specific values)
    BEDROCK_TEMPERATURE: float = 0.0  # Greedy decoding
    BEDROCK_TIMEOUT: int = 30  # Reduced from 90s — avoids 90s hangs on Bedrock fallback
    
    # HuggingFace API Configuration (Faster fallback option)
    HF_API_KEY: Optional[str] = None
    HF_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"  # or mistralai/Mistral-7B-Instruct-v0.3
    HF_MAX_TOKENS: int = 512  # Reduced for speed
    HF_TEMPERATURE: float = 0.0  # Greedy decoding
    HF_TIMEOUT: int = 30  # Reduced timeout
    
    # Ollama Local Server (preferred local fallback — run: ollama serve)
    OLLAMA_BASE_URL: str = "http://localhost:11434"  # Default Ollama server URL
    OLLAMA_MODEL: str = "llama3.1"  # Ollama model name (run: ollama pull llama3.1)
    OLLAMA_TIMEOUT: int = 60  # Seconds to wait for Ollama response

    # Local Llama Fallback via Transformers (GPU/CPU) - used if Ollama unavailable
    LOCAL_MODEL_NAME: str = "meta-llama/Llama-3.2-3B-Instruct"  # Llama 3.2 3B (ungated, fast on CPU)
    LLAMA_CONTEXT_LENGTH: int = 512  # Reduced for <4s latency
    LLAMA_TEMPERATURE: float = 0.0  # Greedy decoding for speed
    LLAMA_MAX_TOKENS: int = 64  # Aggressive limit for speed
    LLAMA_BATCH_SIZE: int = 256  # Optimized batch size
    USE_GPU_INFERENCE: bool = True  # Enable GPU acceleration if available

    # GGUF model (optional — set path to a local .gguf file)
    LLAMA_MODEL_PATH: Optional[str] = None  # e.g. C:/models/llama-3.1-8b-q4.gguf
    LLAMA_N_GPU_LAYERS: int = 0  # GPU layers for GGUF (0 = CPU only)
    
    @property
    def llama_model_path_obj(self) -> Optional[Path]:
        """Convert LLAMA_MODEL_PATH string to Path object (deprecated)"""
        if self.LLAMA_MODEL_PATH:
            path = Path(self.LLAMA_MODEL_PATH)
            if not path.is_absolute():
                path = self.PROJECT_ROOT / path
            return path
        return None
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: int = 2  # seconds
    RETRY_MAX_WAIT: int = 10  # seconds
    RETRY_MULTIPLIER: int = 2
    
    # OCR Configuration
    TESSERACT_CMD: Optional[str] = r"C:\Users\charanl\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    OCR_LANGUAGES: str = "eng"  # Comma-separated language codes
    
    # Document Processing
    MAX_CHUNK_SIZE: int = 3000  # tokens
    CHUNK_OVERLAP: int = 200
    
    # PII Detection
    PII_FIELDS: list[str] = [
        "email", "phone", "ssn", "credit_card", 
        "address", "name", "date_of_birth"
    ]
    
    # Metrics Thresholds (Production Requirements)
    MIN_EXTRACTION_ACCURACY: float = 0.90
    MIN_PII_RECALL: float = 0.95
    MIN_PII_PRECISION: float = 0.90
    MIN_WORKFLOW_SUCCESS_RATE: float = 0.90
    MAX_P95_LATENCY_MS: float = 4000.0  # 4 seconds
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    
    # LangGraph Visualization
    LANGGRAPH_VISUALIZER_ENABLED: bool = True
    
    # Flowise Integration
    FLOWISE_ENABLED: bool = False
    FLOWISE_API_URL: Optional[str] = None
    FLOWISE_API_KEY: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
        self._setup_tesseract()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [
            self.DATA_DIR,
            self.SAMPLES_DIR,
            self.REPORTS_DIR,
            self.LOGS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_tesseract(self) -> None:
        """Setup Tesseract OCR path for Windows"""
        if self.TESSERACT_CMD is None:
            # Common Tesseract installation paths on Windows
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.TESSERACT_CMD = path
                    break


# Global settings instance
settings = Settings()
