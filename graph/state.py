"""
LangGraph State Definition
"""
from typing import TypedDict, Optional, Dict, List, Any
from datetime import datetime
from schemas.document_schemas import (
    DocumentType,
    ClassificationResult,
    ExtractionResult,
    ValidationResult,
    RedactionResult,
    ResponsibleAILog,
)


class DocumentState(TypedDict):
    """
    State object passed between LangGraph nodes
    
    This defines the complete state that flows through the workflow.
    Each agent reads from and writes to this state.
    """
    # Input
    file_path: str
    raw_text: str
    
    # Classification
    doc_type: Optional[DocumentType]
    classification_result: Optional[ClassificationResult]
    
    # Extraction
    extracted_fields: Optional[Dict[str, Any]]
    extraction_result: Optional[ExtractionResult]
    
    # Validation
    validation_status: Optional[str]
    validation_result: Optional[ValidationResult]
    needs_repair: bool
    repair_attempts: int
    current_accuracy: float          # Live accuracy score (0.0–1.0); drives accuracy-based repair
    missing_schema_fields: List[str] # Fields still missing after extraction; used by self-repair
    
    # Redaction
    redacted_text: Optional[str]
    redaction_result: Optional[RedactionResult]
    
    # Metrics & Reporting
    metrics: Optional[Dict[str, Any]]
    
    # Responsible AI Logging
    trace_log: List[ResponsibleAILog]
    
    # Timing & Metadata
    start_time: datetime
    agent_timings: Dict[str, float]
    
    # Error Handling
    errors: List[str]
    retry_count: int
    
    # Success flag
    success: bool


class GraphConfig(TypedDict):
    """Configuration for LangGraph execution"""
    max_repair_attempts: int
    enable_responsible_ai_logging: bool
    visualize: bool
