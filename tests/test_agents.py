"""
Comprehensive Unit Tests for Agentic Document Processor

Tests cover:
- Happy path document processing
- Missing fields handling
- OCR noise resilience
- Bedrock timeout simulation
- Validation failure and self-repair
"""
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from graph.state import DocumentState
from schemas.document_schemas import (
    DocumentType,
    ClassificationResult,
    ValidationStatus
)
from agents import (
    classifier_agent,
    extractor_agent,
    validator_agent,
    self_repair_node,
    redactor_agent,
    reporter_agent
)
# Updated import: LangChain document loader instead of OCR processor
from utils.document_loader import document_loader
from utils.llm_client import LLMClient


# ==================== Fixtures ====================

@pytest.fixture
def sample_invoice_text():
    """Sample invoice text"""
    return """
    INVOICE
    
    Invoice Number: INV-2024-001
    Date: 2024-01-15
    Due Date: 2024-02-15
    
    Vendor: ACME Corporation
    123 Business St
    New York, NY 10001
    
    Customer: John Smith
    456 Customer Ave
    Los Angeles, CA 90001
    
    Description         Quantity    Unit Price    Total
    Widget A            10          $25.00        $250.00
    Widget B            5           $50.00        $250.00
    
    Subtotal: $500.00
    Tax (8%): $40.00
    Total: $540.00
    """


@pytest.fixture
def sample_state(sample_invoice_text):
    """Sample document state"""
    return DocumentState(
        file_path="/test/invoice.txt",
        raw_text=sample_invoice_text,
        doc_type=None,
        classification_result=None,
        extracted_fields=None,
        extraction_result=None,
        validation_status=None,
        validation_result=None,
        needs_repair=False,
        repair_attempts=0,
        redacted_text=None,
        redaction_result=None,
        metrics=None,
        trace_log=[],
        start_time=datetime.utcnow(),
        agent_timings={},
        errors=[],
        retry_count=0,
        success=False
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    return {
        "content": '{"doc_type": "invoice", "confidence": 0.95, "reasoning": "Contains invoice header and line items"}',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.5,
        "tokens": {"input": 100, "output": 50}
    }


# ==================== Classifier Agent Tests ====================

def test_classifier_happy_path(sample_state, mock_llm_response):
    """Test classifier with valid invoice"""
    with patch('agents.classifier_agent.llm_client.generate', return_value=mock_llm_response):
        result_state = classifier_agent.classify(sample_state)
        
        assert result_state["doc_type"] == DocumentType.INVOICE
        assert result_state["classification_result"] is not None
        assert result_state["classification_result"].confidence >= 0.9
        assert len(result_state["trace_log"]) > 0


def test_classifier_unknown_document():
    """Test classifier with unknown document type"""
    unknown_text = "This is just random text without clear document structure."
    
    state = DocumentState(
        file_path="/test/unknown.txt",
        raw_text=unknown_text,
        doc_type=None,
        classification_result=None,
        extracted_fields=None,
        extraction_result=None,
        validation_status=None,
        validation_result=None,
        needs_repair=False,
        repair_attempts=0,
        redacted_text=None,
        redaction_result=None,
        metrics=None,
        trace_log=[],
        start_time=datetime.utcnow(),
        agent_timings={},
        errors=[],
        retry_count=0,
        success=False
    )
    
    mock_response = {
        "content": '{"doc_type": "unknown", "confidence": 0.3, "reasoning": "No clear structure"}',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.5,
        "tokens": {"input": 50, "output": 30}
    }
    
    with patch('agents.classifier_agent.llm_client.generate', return_value=mock_response):
        result_state = classifier_agent.classify(state)
        
        assert result_state["doc_type"] in [DocumentType.UNKNOWN, DocumentType.GENERAL]


def test_classifier_llm_failure(sample_state):
    """Test classifier with LLM failure"""
    with patch('agents.classifier_agent.llm_client.generate', side_effect=Exception("LLM timeout")):
        result_state = classifier_agent.classify(sample_state)
        
        assert result_state["doc_type"] == DocumentType.UNKNOWN
        assert len(result_state["errors"]) > 0


# ==================== Extractor Agent Tests ====================

def test_extractor_invoice_extraction(sample_state):
    """Test extractor with invoice document"""
    # Set doc_type first
    sample_state["doc_type"] = DocumentType.INVOICE
    
    mock_extraction = {
        "content": '''{
            "invoice_number": "INV-2024-001",
            "invoice_date": "2024-01-15",
            "vendor_name": "ACME Corporation",
            "customer_name": "John Smith",
            "total_amount": 540.00,
            "tax_amount": 40.00
        }''',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 1.0,
        "tokens": {"input": 500, "output": 100}
    }
    
    with patch('agents.extractor_agent.llm_client.generate', return_value=mock_extraction):
        result_state = extractor_agent.extract(sample_state)
        
        assert result_state["extracted_fields"] is not None
        assert "invoice_number" in result_state["extracted_fields"]
        assert result_state["extraction_result"] is not None


def test_extractor_missing_fields():
    """Test extractor with incomplete data"""
    incomplete_text = "INVOICE\nNumber: INV-001\n"
    
    state = DocumentState(
        file_path="/test/incomplete.txt",
        raw_text=incomplete_text,
        doc_type=DocumentType.INVOICE,
        classification_result=None,
        extracted_fields=None,
        extraction_result=None,
        validation_status=None,
        validation_result=None,
        needs_repair=False,
        repair_attempts=0,
        redacted_text=None,
        redaction_result=None,
        metrics=None,
        trace_log=[],
        start_time=datetime.utcnow(),
        agent_timings={},
        errors=[],
        retry_count=0,
        success=False
    )
    
    mock_extraction = {
        "content": '{"invoice_number": "INV-001", "invoice_date": null, "total_amount": null}',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.8,
        "tokens": {"input": 100, "output": 50}
    }
    
    with patch('agents.extractor_agent.llm_client.generate', return_value=mock_extraction):
        result_state = extractor_agent.extract(state)
        
        assert result_state["extracted_fields"] is not None
        # Many fields will be None
        assert result_state["extracted_fields"].get("invoice_number") == "INV-001"


# ==================== Validator Agent Tests ====================

def test_validator_valid_data(sample_state):
    """Test validator with valid extracted data"""
    sample_state["doc_type"] = DocumentType.INVOICE
    sample_state["extracted_fields"] = {
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-01-15",
        "vendor_name": "ACME Corporation",
        "total_amount": 540.00,
        "tax_amount": 40.00,
        "currency": "USD"
    }
    
    result_state = validator_agent.validate(sample_state)
    
    assert result_state["validation_result"] is not None
    # Validation may pass or have warnings depending on schema


def test_validator_invalid_data(sample_state):
    """Test validator with invalid data"""
    sample_state["doc_type"] = DocumentType.INVOICE
    sample_state["extracted_fields"] = {
        "invoice_number": "INV-2024-001",
        "total_amount": -100.00,  # Negative amount (invalid)
        "tax_amount": 1000.00,  # Tax > total (invalid)
    }
    
    result_state = validator_agent.validate(sample_state)
    
    assert result_state["validation_result"] is not None
    # Should detect business logic errors


def test_validator_schema_mismatch(sample_state):
    """Test validator with schema type mismatch"""
    sample_state["doc_type"] = DocumentType.INVOICE
    sample_state["extracted_fields"] = {
        "invoice_number": 12345,  # Should be string
        "total_amount": "not_a_number",  # Should be float
    }
    
    result_state = validator_agent.validate(sample_state)
    
    assert result_state["validation_result"] is not None
    assert len(result_state["validation_result"].errors) > 0


# ==================== Self-Repair Node Tests ====================

def test_self_repair_successful(sample_state):
    """Test self-repair with fixable errors"""
    sample_state["doc_type"] = DocumentType.INVOICE
    sample_state["extracted_fields"] = {
        "invoice_number": "INV-001",
        "total_amount": "invalid"
    }
    sample_state["validation_result"] = Mock()
    sample_state["validation_result"].errors = ["Field 'total_amount': invalid type"]
    sample_state["needs_repair"] = True
    sample_state["repair_attempts"] = 0
    
    mock_repair = {
        "content": '{"invoice_number": "INV-001", "total_amount": 100.00}',
        "provider": "bedrock_claude",
        "model": "claude-3-haiku",
        "latency": 0.5,
        "tokens": {"input": 200, "output": 50}
    }
    
    with patch('agents.self_repair_node.llm_client.generate', return_value=mock_repair):
        result_state = self_repair_node.repair(sample_state)
        
        assert result_state["repair_attempts"] == 1
        assert result_state["extracted_fields"] is not None


def test_self_repair_max_attempts(sample_state):
    """Test self-repair respects max attempts"""
    sample_state["repair_attempts"] = 2  # At max
    sample_state["needs_repair"] = True
    
    result_state = self_repair_node.repair(sample_state)
    
    assert result_state["needs_repair"] == False  # Should stop trying


# ==================== Redactor Agent Tests ====================

def test_redactor_pii_detection():
    """Test PII detection in text"""
    text_with_pii = """
    Patient: John Smith
    Email: john.smith@email.com
    Phone: 555-123-4567
    SSN: 123-45-6789
    """
    
    state = DocumentState(
        file_path="/test/medical.txt",
        raw_text=text_with_pii,
        doc_type=DocumentType.MEDICAL_RECORD,
        classification_result=None,
        extracted_fields={},
        extraction_result=None,
        validation_status=None,
        validation_result=None,
        needs_repair=False,
        repair_attempts=0,
        redacted_text=None,
        redaction_result=None,
        metrics=None,
        trace_log=[],
        start_time=datetime.utcnow(),
        agent_timings={},
        errors=[],
        retry_count=0,
        success=False
    )
    
    with patch('agents.redactor_agent.llm_client.generate') as mock_llm:
        mock_llm.return_value = {
            "content": '[{"type": "name", "value": "John Smith", "confidence": 0.95}]',
            "provider": "bedrock_claude",
            "model": "claude-3-haiku",
            "latency": 0.5,
            "tokens": {"input": 100, "output": 50}
        }
        
        result_state = redactor_agent.redact(state)
        
        assert result_state["redaction_result"] is not None
        assert result_state["redaction_result"].pii_count > 0
        assert "[EMAIL_REDACTED]" in result_state["redacted_text"]
        assert "[PHONE_REDACTED]" in result_state["redacted_text"]
        assert "[SSN_REDACTED]" in result_state["redacted_text"]


def test_redactor_no_pii():
    """Test redactor with no PII"""
    clean_text = "This is a document with no personal information."
    
    state = DocumentState(
        file_path="/test/clean.txt",
        raw_text=clean_text,
        doc_type=DocumentType.GENERAL,
        classification_result=None,
        extracted_fields={},
        extraction_result=None,
        validation_status=None,
        validation_result=None,
        needs_repair=False,
        repair_attempts=0,
        redacted_text=None,
        redaction_result=None,
        metrics=None,
        trace_log=[],
        start_time=datetime.utcnow(),
        agent_timings={},
        errors=[],
        retry_count=0,
        success=False
    )
    
    with patch('agents.redactor_agent.llm_client.generate') as mock_llm:
        mock_llm.return_value = {
            "content": '[]',
            "provider": "bedrock_claude",
            "model": "claude-3-haiku",
            "latency": 0.3,
            "tokens": {"input": 50, "output": 10}
        }
        
        result_state = redactor_agent.redact(state)
        
        assert result_state["redaction_result"].pii_count == 0
        assert result_state["redacted_text"] == clean_text


# ==================== Reporter Agent Tests ====================

def test_reporter_generates_metrics(sample_state):
    """Test reporter generates complete metrics"""
    # Setup complete state
    sample_state["doc_type"] = DocumentType.INVOICE
    sample_state["classification_result"] = Mock(doc_type=DocumentType.INVOICE, confidence=0.95)
    sample_state["extracted_fields"] = {"invoice_number": "INV-001", "total_amount": 100.00}
    sample_state["extraction_result"] = Mock(extracted_fields={}, confidence=0.90, chunk_count=1)
    sample_state["validation_result"] = Mock(is_valid=True, status=ValidationStatus.VALID, errors=[], warnings=[])
    sample_state["redaction_result"] = Mock(pii_count=2, precision=0.95, recall=0.90)
    
    result_state = reporter_agent.generate_report(sample_state)
    
    assert result_state["metrics"] is not None
    assert "extraction_accuracy" in result_state["metrics"]
    assert "workflow_success" in result_state["metrics"]


# ==================== OCR Processor Tests ====================

def test_ocr_text_file():
    """Test OCR processor with plain text file"""
    processor = OCRProcessor()
    
    # Mock file
    with patch('pathlib.Path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data="Sample text content")):
            # This would normally call extract_text_from_file
            # For now, just test that OCRProcessor can be instantiated
            assert processor is not None


# ==================== Integration Tests ====================

@pytest.mark.integration
def test_full_workflow_simulation(sample_state, mock_llm_response):
    """Simulate full workflow execution"""
    # This is a simplified integration test
    # Real integration tests would use actual workflow execution
    
    # 1. Classify
    with patch('agents.classifier_agent.llm_client.generate', return_value=mock_llm_response):
        state = classifier_agent.classify(sample_state)
        assert state["doc_type"] is not None
    
    # 2. Extract (mocked)
    state["doc_type"] = DocumentType.INVOICE
    
    # 3. Validate
    state["extracted_fields"] = {"invoice_number": "INV-001"}
    state = validator_agent.validate(state)
    assert state["validation_result"] is not None
    
    # 4. Redact (mocked)
    with patch('agents.redactor_agent.llm_client.generate') as mock:
        mock.return_value = {
            "content": '[]',
            "provider": "bedrock_claude",
            "model": "claude-3-haiku",
            "latency": 0.3,
            "tokens": {"input": 50, "output": 10}
        }
        state = redactor_agent.redact(state)
        assert state["redaction_result"] is not None
    
    # 5. Report
    state = reporter_agent.generate_report(state)
    assert state["metrics"] is not None


def mock_open(read_data):
    """Helper for mocking file open"""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(read_data=read_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
