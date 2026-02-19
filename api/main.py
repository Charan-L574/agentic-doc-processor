"""
FastAPI Server for Document Processing API
"""
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from graph.workflow import workflow
from utils.logger import logger
from utils.graph_visualizer import graph_visualizer
from schemas.document_schemas import ProcessingResult


# Request/Response models
class ProcessRequest(BaseModel):
    """Request model for document processing"""
    file_path: str = Field(..., description="Local file path to process")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for checkpointing")


class ProcessResponse(BaseModel):
    """Response model for document processing"""
    success: bool
    message: str
    doc_type: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    extracted_fields: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    redaction: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[list[str]] = None
    processing_time: float
    timestamp: str
    trace_log: Optional[list[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    llm_available: bool


# Create FastAPI app
app = FastAPI(
    title="Agentic Document Processor",
    description="Production-grade document processing with LangGraph + Amazon Bedrock",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("FastAPI server starting up")

    # Compile workflow
    try:
        workflow.compile()
        logger.info("Workflow compiled successfully")
    except Exception as e:
        logger.error(f"Failed to compile workflow: {e}")

    # Warm up Presidio NLP engine (spaCy loads lazily on first .analyze() call — 2-3s)
    # Running a dummy call here means zero extra latency on the first real document.
    try:
        from agents.redactor_agent import redactor_agent
        if redactor_agent.analyzer:
            redactor_agent.analyzer.analyze(text="warmup", language="en")
            logger.info("Presidio NLP engine warmed up successfully")
    except Exception as e:
        logger.warning(f"Presidio warm-up failed (non-fatal): {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("FastAPI server shutting down")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Agentic Document Processor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from utils.llm_client import llm_client
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        llm_available=llm_client.is_available()
    )


@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Process a document through the agentic pipeline
    
    Args:
        request: ProcessRequest with file_path
    
    Returns:
        ProcessResponse with results
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"API: Processing document: {request.file_path}")
        
        # Validate file exists - handle both absolute and relative paths
        file_path = Path(request.file_path)
        if not file_path.is_absolute():
            # Resolve relative paths against project root
            file_path = settings.PROJECT_ROOT / file_path
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        # Process document
        thread_id = request.thread_id or f"api_{datetime.utcnow().timestamp()}"
        
        # Run workflow in executor to avoid blocking
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            workflow.process_document,
            str(file_path),
            thread_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Extract detailed results from final_state
        classification_result = final_state.get("classification_result")
        extraction_result = final_state.get("extraction_result")
        validation_result = final_state.get("validation_result")
        redaction_result = final_state.get("redaction_result")
        
        # Build response
        response = ProcessResponse(
            success=final_state.get("success", False),
            message="Document processed successfully" if final_state.get("success") else "Processing completed with errors",
            doc_type=final_state.get("doc_type").value if final_state.get("doc_type") else None,
            confidence=classification_result.confidence if classification_result else None,
            reasoning=classification_result.reasoning if classification_result else None,
            extracted_fields=final_state.get("extracted_fields"),
            validation={
                "status": validation_result.status.value if validation_result else "unknown",
                "is_valid": validation_result.is_valid if validation_result else False,
                "errors": validation_result.errors if validation_result else [],
                "warnings": validation_result.warnings if validation_result else []
            } if validation_result else None,
            redaction={
                "pii_count": redaction_result.pii_count if redaction_result else 0,
                "precision": redaction_result.precision if redaction_result else 0,
                "recall": redaction_result.recall if redaction_result else 0,
                "pii_detections": [
                    {
                        "pii_type": det.pii_type.value if hasattr(det.pii_type, 'value') else str(det.pii_type),
                        "original_text": det.original_text,
                        "redacted_text": det.redacted_text,
                        "detection_source": det.detection_source if hasattr(det, 'detection_source') else "unknown"
                    }
                    for det in (redaction_result.pii_detections if hasattr(redaction_result, 'pii_detections') else redaction_result.detected_pii if hasattr(redaction_result, 'detected_pii') else [])
                ] if redaction_result else [],
                "redacted_text": redaction_result.redacted_text if redaction_result else None
            } if redaction_result else None,
            metrics=final_state.get("metrics"),
            errors=final_state.get("errors", []),
            processing_time=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            trace_log=[
                log.model_dump() if hasattr(log, 'model_dump') else log.dict() if hasattr(log, 'dict') else log
                for log in final_state.get("trace_log", [])
            ]
        )
        
        logger.info(
            f"API: Document processing complete",
            file=request.file_path,
            success=response.success,
            processing_time=processing_time
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a file and process it
    
    Args:
        file: Uploaded file
        background_tasks: Background tasks
    
    Returns:
        ProcessResponse
    """
    try:
        logger.info(f"API: Uploading file: {file.filename}")
        
        # Save uploaded file temporarily
        temp_dir = settings.DATA_DIR / "uploads"
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file_path = temp_dir / f"{timestamp}_{file.filename}"
        
        # Write file
        contents = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(contents)
        
        logger.info(f"API: File saved to {temp_file_path}")
        
        # Process document
        request = ProcessRequest(file_path=str(temp_file_path))
        response = await process_document(request)
        
        # Optionally clean up file in background
        # if background_tasks:
        #     background_tasks.add_task(temp_file_path.unlink, missing_ok=True)
        
        return response
    
    except Exception as e:
        logger.error(f"API: Upload and process failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize/graph")
async def visualize_graph(format: str = "mermaid"):
    """
    Get workflow graph visualization
    
    Args:
        format: Output format - "mermaid" or "json"
    
    Returns:
        Mermaid diagram or graph JSON
    """
    try:
        if format == "mermaid":
            mermaid_diagram = graph_visualizer.generate_mermaid_diagram(workflow)
            return {
                "format": "mermaid",
                "diagram": mermaid_diagram
            }
        elif format == "json":
            # Return graph structure as JSON
            if hasattr(workflow, 'compiled_graph') and workflow.compiled_graph:
                graph = workflow.compiled_graph.get_graph()
                return {
                    "format": "json",
                    "nodes": [node for node in graph.nodes],
                    "edges": [(edge[0], edge[1]) for edge in graph.edges]
                }
            else:
                return {"error": "Graph not compiled"}
        else:
            raise HTTPException(status_code=400, detail="Format must be 'mermaid' or 'json'")
    
    except Exception as e:
        logger.error(f"API: Graph visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/trace")
async def visualize_execution_trace(trace_log: List[Dict[str, Any]]):
    """
    Generate execution trace visualization from trace log (sequence diagram)
    
    Args:
        trace_log: List of ResponsibleAILog entries
    
    Returns:
        Mermaid sequence diagram showing actual execution flow
    """
    try:
        sequence_diagram = graph_visualizer.visualize_execution_trace(trace_log)
        
        # Extract execution path
        execution_path = graph_visualizer.extract_execution_path(trace_log)
        
        return {
            "format": "mermaid",
            "diagram": sequence_diagram,
            "execution_path": execution_path,
            "total_steps": len(trace_log)
        }
    
    except Exception as e:
        logger.error(f"API: Trace visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize/execution-path")
async def visualize_execution_path(trace_log: List[Dict[str, Any]]):
    """
    Generate execution path diagram showing the actual path taken through the workflow
    
    Args:
        trace_log: List of ResponsibleAILog entries from document processing
    
    Returns:
        Mermaid diagram with highlighted execution path (classify -> extract -> validate -> etc)
    """
    try:
        # Generate path diagram with highlighted nodes
        path_diagram = graph_visualizer.generate_execution_path_diagram(trace_log)
        
        # Extract execution path for reference
        execution_path = graph_visualizer.extract_execution_path(trace_log)
        
        # Count repair attempts
        repair_count = execution_path.count("repair")
        
        return {
            "format": "mermaid",
            "diagram": path_diagram,
            "execution_path": execution_path,
            "repair_attempts": repair_count,
            "total_steps": len(trace_log)
        }
    
    except Exception as e:
        logger.error(f"API: Execution path visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        if background_tasks:
            background_tasks.add_task(temp_file_path.unlink, missing_ok=True)
        
        return response
    
    except Exception as e:
        logger.error(f"API: Upload and process failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/diagram")
async def get_workflow_diagram():
    """
    Get Mermaid diagram of the workflow
    
    Returns:
        Mermaid diagram string
    """
    try:
        from utils.graph_visualizer import graph_visualizer
        
        mermaid_diagram = graph_visualizer.generate_mermaid_diagram(workflow)
        return {
            "diagram": mermaid_diagram,
            "format": "mermaid",
            "visualizer_url": "https://mermaid.live/edit"
        }
    except Exception as e:
        logger.error(f"Failed to get workflow diagram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get summary of recent processing metrics
    
    Returns:
        Metrics summary
    """
    try:
        # Read recent report files
        import json
        
        reports_dir = settings.REPORTS_DIR
        report_files = sorted(
            reports_dir.glob("metrics_report_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:10]  # Last 10 reports
        
        summaries = []
        for report_file in report_files:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
                summaries.append({
                    "document": report_data.get("document"),
                    "timestamp": report_data.get("timestamp"),
                    "doc_type": report_data.get("doc_type"),
                    "extraction_accuracy": report_data.get("metrics", {}).get("extraction_accuracy"),
                    "workflow_success": report_data.get("metrics", {}).get("workflow_success")
                })
        
        return {"recent_reports": summaries}
    
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting FastAPI server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
