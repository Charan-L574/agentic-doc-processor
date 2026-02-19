"""
LangGraph Workflow - Orchestrates all agents in a stateful graph
"""
from datetime import datetime
from typing import Dict, Any, Literal
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import DocumentState, GraphConfig
from agents import (
    classifier_agent,
    extractor_agent,
    validator_agent,
    self_repair_node,
    redactor_agent,
    reporter_agent,
)
# REVERTED: Using hybrid Presidio+LLM redactor (LLM-only had bugs)

from utils.document_loader import document_loader
from utils.logger import logger


class DocumentProcessingWorkflow:
    """
    LangGraph workflow for document processing
    
    Graph Flow:
    START → classify → extract → validate → [self_repair] → redact → report → END
    
    Conditional edges:
    - validate → self_repair (if needs_repair)
    - self_repair → validate (retry validation)
    """
    
    def __init__(self, config: GraphConfig = None):
        """
        Initialize workflow
        
        Args:
            config: Graph configuration
        """
        self.config = config or GraphConfig(
            max_repair_attempts=1,  # 1 repair max — extraction with 2000 tokens should succeed first try
            enable_responsible_ai_logging=True,
            visualize=False
        )
        
        self.graph = self._build_graph()
        self.compiled_graph = None
        
        logger.info("DocumentProcessingWorkflow initialized")
    
    def _initialize_state(self, file_path: str, raw_text: str) -> DocumentState:
        """
        Initialize document state
        
        Args:
            file_path: Path to document
            raw_text: Extracted text from document
        
        Returns:
            Initial DocumentState
        """
        return DocumentState(
            file_path=file_path,
            raw_text=raw_text,
            doc_type=None,
            classification_result=None,
            extracted_fields=None,
            extraction_result=None,
            validation_status=None,
            validation_result=None,
            needs_repair=False,
            repair_attempts=0,
            current_accuracy=0.0,
            missing_schema_fields=[],
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
    
    # Node functions
    def _classify_node(self, state: DocumentState) -> DocumentState:
        """Classifier node"""
        logger.info("Workflow: Executing classify node")
        return classifier_agent.classify(state)
    
    def _extract_node(self, state: DocumentState) -> DocumentState:
        """Extractor node"""
        logger.info("Workflow: Executing extract node")
        return extractor_agent.extract(state)
    
    def _validate_node(self, state: DocumentState) -> DocumentState:
        """Validator node"""
        logger.info("Workflow: Executing validate node")
        return validator_agent.validate(state)
    
    def _repair_node(self, state: DocumentState) -> DocumentState:
        """Self-repair node"""
        logger.info("Workflow: Executing repair node")
        return self_repair_node.repair(state)
    
    def _redact_node(self, state: DocumentState) -> DocumentState:
        """Redactor node - Hybrid Presidio+LLM for robust PII detection"""
        logger.info("Workflow: Executing redact node (Hybrid mode)")
        return redactor_agent.redact(state)
    
    def _report_node(self, state: DocumentState) -> DocumentState:
        """Reporter node"""
        logger.info("Workflow: Executing report node")
        return reporter_agent.generate_report(state)
    
    # Conditional edge function
    def _should_repair(self, state: DocumentState) -> Literal["repair", "redact"]:
        """
        Determine if repair is needed.

        Routes to repair when EITHER condition holds (and budget remains):
          • Validator set needs_repair=True  (missing required fields / invalid data)
          • Extraction accuracy is below the 90 % threshold
        """
        needs_repair = state.get("needs_repair", False)
        repair_attempts = state.get("repair_attempts", 0)
        max_attempts = self.config.get("max_repair_attempts", 3)
        current_accuracy = state.get("current_accuracy", 1.0)
        ACCURACY_THRESHOLD = 0.80  # Matches validator threshold — avoids redundant repair

        below_accuracy = current_accuracy < ACCURACY_THRESHOLD

        if (needs_repair or below_accuracy) and repair_attempts < max_attempts:
            reason = []
            if needs_repair:
                reason.append("validation errors")
            if below_accuracy:
                reason.append(f"accuracy {current_accuracy:.0%} < {ACCURACY_THRESHOLD:.0%}")
            logger.info(
                f"Routing to repair node (attempt {repair_attempts + 1}/{max_attempts}): "
                + ", ".join(reason)
            )
            return "repair"
        else:
            logger.info(
                f"Routing to redact node "
                f"(accuracy={current_accuracy:.0%}, needs_repair={needs_repair}, "
                f"attempts={repair_attempts}/{max_attempts})"
            )
            return "redact"
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph state graph
        
        Returns:
            StateGraph instance
        """
        # Create state graph
        workflow = StateGraph(DocumentState)
        
        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("repair", self._repair_node)
        workflow.add_node("redact", self._redact_node)
        workflow.add_node("report", self._report_node)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add edges
        workflow.add_edge("classify", "extract")
        workflow.add_edge("extract", "validate")  # Validation re-enabled
        
        # Conditional edge: validate → repair or redact
        workflow.add_conditional_edges(
            "validate",
            self._should_repair,
            {
                "repair": "repair",
                "redact": "redact"
            }
        )
        
        # After repair, go back to validate
        workflow.add_edge("repair", "validate")
        
        # Continue to reporter
        workflow.add_edge("redact", "report")
        
        # End
        workflow.add_edge("report", END)
        
        logger.info("LangGraph workflow built successfully")
        return workflow
    
    def compile(self) -> None:
        """Compile the graph for execution"""
        if self.compiled_graph is None:
            # Use memory saver for checkpointing
            memory = MemorySaver()
            self.compiled_graph = self.graph.compile(checkpointer=memory)
            logger.info("Workflow compiled with checkpointing")
    
    def visualize(self, output_path: str = None) -> None:
        """
        Visualize the workflow graph
        
        Args:
            output_path: Optional path to save visualization
        """
        try:
            from IPython.display import Image, display
            
            # Get graph visualization
            graph_image = self.compiled_graph.get_graph().draw_mermaid_png()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(graph_image)
                logger.info(f"Graph visualization saved to {output_path}")
            else:
                display(Image(graph_image))
        
        except ImportError:
            logger.warning("IPython not available, visualization skipped")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def process_document(
        self,
        file_path: str,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a single document through the workflow
        
        Args:
            file_path: Path to document file
            thread_id: Thread ID for checkpointing
        
        Returns:
            Final state dict
        """
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Ensure graph is compiled
            if self.compiled_graph is None:
                self.compile()
            
            # Convert to Path object
            file_path_obj = Path(file_path)
            
            # Extract text using LangChain document loaders
            logger.info(f"Loading document with LangChain loaders: {file_path_obj.name}")
            raw_text = document_loader.load_and_extract_text(file_path_obj)
            
            # Check if text was extracted
            if not raw_text or len(raw_text.strip()) == 0:
                logger.warning(f"No text extracted from document: {file_path_obj.name}")
                logger.info("This could mean: (1) OCR failed, (2) Document is truly empty, or (3) Unsupported format")
                
                # For images, provide more helpful error
                if file_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
                    raise ValueError(
                        f"No text extracted from image '{file_path_obj.name}'. "
                        f"Possible causes: (1) Image contains no text, (2) Tesseract OCR not working, "
                        f"(3) Image quality too low. Check logs for OCR details."
                    )
                else:
                    raise ValueError(f"No text extracted from document '{file_path_obj.name}'")
            
            logger.info(f"Extracted {len(raw_text)} characters")
            
            # Initialize state
            initial_state = self._initialize_state(str(file_path_obj), raw_text)
            
            # Execute graph
            logger.info("Starting workflow execution")
            
            config = {"configurable": {"thread_id": thread_id}}
            
            # Run the graph
            final_state = None
            for state in self.compiled_graph.stream(initial_state, config):
                # state is a dict with node names as keys
                final_state = state
                
                # Log progress
                for node_name, node_state in state.items():
                    logger.debug(f"Completed node: {node_name}")
            
            # Extract the final state from last node output
            if final_state:
                # Get the last value (final state)
                final_state_data = list(final_state.values())[-1]
            else:
                raise RuntimeError("Workflow did not produce final state")
            
            logger.info(
                "Workflow execution complete",
                success=final_state_data.get("success", False)
            )
            
            return final_state_data
        
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            raise
    
    def get_graph_mermaid(self) -> str:
        """
        Get Mermaid diagram representation of the graph
        
        Returns:
            Mermaid diagram string (clean, user-friendly version)
        """
        # Always use the clean, hand-crafted diagram for better user experience
        # LangGraph's auto-generated diagram is too complex and confusing
        return self._generate_simple_mermaid()
    
    def _generate_simple_mermaid(self) -> str:
        """Generate a clean, user-friendly Mermaid diagram showing the workflow"""
        return """graph LR
    START([🚀 START]) --> classify[📋 Classify<br/>Document Type]
    classify --> extract[📤 Extract<br/>Structured Fields]
    extract --> validate[✓ Validate<br/>Schema & Rules]
    validate --> repair{🔧 Valid?}
    repair -->|❌ No| self_repair[🔄 Repair<br/>Fix Errors]
    self_repair --> validate
    repair -->|✅ Yes| redact[🔒 Redact<br/>Mask PII]
    redact --> report[📊 Report<br/>Generate Metrics]
    report --> END([✔️ END])
    
    style START fill:#90EE90,stroke:#006400,stroke-width:3px
    style END fill:#90EE90,stroke:#006400,stroke-width:3px
    style classify fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style extract fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style validate fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style self_repair fill:#FFD700,stroke:#FF8C00,stroke-width:2px
    style redact fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style report fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style repair fill:#FFA500,stroke:#FF6347,stroke-width:2px
"""


# Global workflow instance
workflow = DocumentProcessingWorkflow()
