"""
LangGraph Visualization Utilities
"""
from pathlib import Path
from typing import Optional
import base64
from io import BytesIO

from utils.logger import logger


class GraphVisualizer:
    """
    Utility class for LangGraph visualization
    
    Provides:
    1. Graph structure visualization (Mermaid diagrams)
    2. Execution trace visualization
    3. PNG/SVG export capabilities
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory for saving visualization outputs
        """
        self.output_dir = output_dir or Path("./visualizations")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"GraphVisualizer initialized, output_dir: {self.output_dir}")
    
    def generate_mermaid_diagram(self, workflow) -> str:
        """
        Generate Mermaid diagram from LangGraph workflow
        
        Args:
            workflow: LangGraph workflow/graph instance (DocumentProcessingWorkflow)
        
        Returns:
            Mermaid diagram as string
        """
        try:
            # Check if it's our custom workflow class
            if hasattr(workflow, 'get_graph_mermaid'):
                mermaid = workflow.get_graph_mermaid()
                return mermaid
            
            # Try compiled graph (standard LangGraph)
            elif hasattr(workflow, 'compiled_graph') and workflow.compiled_graph:
                graph = workflow.compiled_graph.get_graph()
                mermaid = graph.draw_mermaid()
                return mermaid
            
            # Try graph attribute
            elif hasattr(workflow, 'graph'):
                # Compile if needed
                if not hasattr(workflow, 'compiled_graph') or workflow.compiled_graph is None:
                    if hasattr(workflow, 'compile'):
                        workflow.compile()
                
                if workflow.compiled_graph:
                    graph = workflow.compiled_graph.get_graph()
                    mermaid = graph.draw_mermaid()
                    return mermaid
            
            # Direct StateGraph
            elif hasattr(workflow, 'get_graph'):
                graph = workflow.get_graph()
                mermaid = graph.draw_mermaid()
                return mermaid
            
            # Fallback
            raise AttributeError("Cannot find graph visualization method")
            
        except Exception as e:
            logger.error(f"Failed to generate Mermaid diagram: {e}")
            return f"graph TD\n  Error[Error: {str(e)}]"
    
    def save_mermaid_to_file(self, mermaid_diagram: str, filename: str = "workflow.mmd") -> Path:
        """
        Save Mermaid diagram to file
        
        Args:
            mermaid_diagram: Mermaid diagram string
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_diagram)
        
        logger.info(f"Mermaid diagram saved to {output_path}")
        return output_path
    
    def export_as_png(self, workflow, filename: str = "workflow.png") -> Optional[Path]:
        """
        Export workflow as PNG image
        
        Args:
            workflow: LangGraph workflow/graph instance (DocumentProcessingWorkflow)
            filename: Output filename
        
        Returns:
            Path to saved file or None if failed
        """
        try:
            png_data = None
            
            # Try compiled graph (standard LangGraph)
            if hasattr(workflow, 'compiled_graph') and workflow.compiled_graph:
                graph = workflow.compiled_graph.get_graph()
                png_data = graph.draw_mermaid_png()
            
            # Try graph attribute
            elif hasattr(workflow, 'graph'):
                # Compile if needed
                if not hasattr(workflow, 'compiled_graph') or workflow.compiled_graph is None:
                    if hasattr(workflow, 'compile'):
                        workflow.compile()
                
                if workflow.compiled_graph:
                    graph = workflow.compiled_graph.get_graph()
                    png_data = graph.draw_mermaid_png()
            
            # Direct StateGraph
            elif hasattr(workflow, 'get_graph'):
                graph = workflow.get_graph()
                png_data = graph.draw_mermaid_png()
            
            if png_data:
                output_path = self.output_dir / filename
                with open(output_path, "wb") as f:
                    f.write(png_data)
                
                logger.info(f"PNG exported to {output_path}")
                return output_path
            else:
                logger.warning("Could not generate PNG data")
                return None
        
        except ImportError:
            logger.warning("PNG export requires: pip install pygraphviz")
            return None
        except Exception as e:
            logger.error(f"Failed to export PNG: {e}")
            return None
    
    def get_execution_trace(self, state_history: list) -> dict:
        """
        Generate execution trace from state history
        
        Args:
            state_history: List of state snapshots during execution
        
        Returns:
            Structured execution trace
        """
        trace = {
            "total_steps": len(state_history),
            "steps": []
        }
        
        for idx, state in enumerate(state_history):
            step = {
                "step_number": idx + 1,
                "agent_timings": state.get("agent_timings", {}),
                "doc_type": state.get("doc_type"),
                "validation_status": state.get("validation_status"),
                "repair_attempts": state.get("repair_attempts", 0),
                "errors": state.get("errors", [])
            }
            trace["steps"].append(step)
        
        return trace
    
    def extract_execution_path(self, trace_log: list) -> list:
        """
        Extract the actual execution path from trace log
        
        Args:
            trace_log: List of ResponsibleAILog entries from state
        
        Returns:
            List of node names in execution order
        """
        path = []
        for entry in trace_log:
            agent = entry.get("agent_name", "unknown")
            if agent and agent not in ["START", "END"]:
                # Normalize agent name: "ClassifierAgent" -> "classify", "ExtractorAgent" -> "extract"
                agent_normalized = agent.replace("Agent", "").lower()
                # Handle "SelfRepairNode" -> "repair"
                if "repair" in agent_normalized:
                    agent_normalized = "repair"
                path.append(agent_normalized)
        return path
    
    def generate_execution_path_diagram(self, trace_log: list) -> str:
        """
        Generate Mermaid diagram showing ONLY the actual execution path taken
        
        Args:
            trace_log: List of ResponsibleAILog entries from state
        
        Returns:
            Mermaid diagram with only executed nodes (no gray boxes)
        """
        path = self.extract_execution_path(trace_log)
        
        # Start diagram
        diagram = ["graph LR"]
        diagram.append("    classDef executed fill:#90EE90,stroke:#006400,stroke-width:3px,color:#000")
        diagram.append("")
        
        # Only show executed nodes (no gray boxes)
        executed_nodes = list(dict.fromkeys(path))  # Remove duplicates, preserve order
        
        # Add START
        diagram.append("    START([🚀 START]):::executed")
        
        # Add only executed nodes with emojis
        node_emojis = {
            "classify": "📋",
            "extract": "📤",
            "validate": "✓",
            "repair": "🔧",
            "redact": "🔒",
            "report": "📊"
        }
        
        for node in executed_nodes:
            emoji = node_emojis.get(node, "")
            node_label = f"{emoji} {node.capitalize()}"
            diagram.append(f"    {node}[{node_label}]:::executed")
        
        # Add END
        diagram.append("    END([✔️ END]):::executed")
        diagram.append("")
        
        # Add edges based on execution path
        if path:
            # Connect START to first node
            diagram.append(f"    START --> {path[0]}")
            
            # Add edges for actual path (skip first since we just connected START)
            for i in range(len(path) - 1):
                current = path[i]
                next_node = path[i + 1]
                
                # Style arrows based on path
                if current == "validate" and next_node == "repair":
                    diagram.append(f"    {current} -.->|needs repair| {next_node}")
                elif current == "repair" and next_node == "validate":
                    diagram.append(f"    {current} -.->|retry| {next_node}")
                else:
                    diagram.append(f"    {current} --> {next_node}")
            
            # Connect last node to END
            diagram.append(f"    {path[-1]} --> END")
        
        return "\n".join(diagram)
    
    def visualize_execution_trace(self, trace_log: list) -> str:
        """
        Generate Mermaid sequence diagram from actual execution trace
        
        Args:
            trace_log: List of ResponsibleAILog entries from state
        
        Returns:
            Mermaid sequence diagram showing actual execution flow
        """
        diagram = ["sequenceDiagram"]
        diagram.append("    participant User")
        diagram.append("    participant Workflow")
        
        # Extract unique agents from trace
        agents = []
        for entry in trace_log:
            agent = entry.get("agent_name", "")
            if agent and agent not in ["START", "END"] and agent not in agents:
                # Normalize agent name for display
                agent_display = agent.replace("Agent", "").replace("Node", "")
                agents.append(agent_display)
                diagram.append(f"    participant {agent_display}")
        
        diagram.append("    participant Reporter")
        diagram.append("")
        
        # Add execution flow
        diagram.append("    User->>Workflow: Submit Document")
        
        prev_agent = "Workflow"
        for entry in trace_log:
            agent = entry.get("agent_name", "")
            # Use output_data as action since we don't have a separate action field
            action_raw = entry.get("output_data", "")[:50]  # First 50 chars
            # Sanitize action text for Mermaid - remove special chars, newlines, quotes
            action = action_raw.replace('"', "'").replace('\n', ' ').replace('\r', '').replace(':', ' -').strip()
            if not action:
                action = "Processing"
            timestamp = entry.get("timestamp", "")
            
            if agent and agent not in ["START", "END"]:
                # Normalize agent name for display
                agent_cap = agent.replace("Agent", "").replace("Node", "")
                
                # Add action with timing
                if action:
                    diagram.append(f"    {prev_agent}->>+{agent_cap}: {action}")
                    
                    # Add note for timing
                    if "agent_timings" in entry:
                        timing = entry.get("agent_timings", {}).get(agent, 0)
                        if timing > 0:
                            diagram.append(f"    Note over {agent_cap}: {timing:.2f}s")
                    
                    diagram.append(f"    {agent_cap}-->>-{prev_agent}: Result")
                    prev_agent = agent_cap
        
        diagram.append(f"    {prev_agent}->>Reporter: Generate Report")
        diagram.append("    Reporter->>User: Final Result")
        
        return "\n".join(diagram)


# Global visualizer instance
graph_visualizer = GraphVisualizer()


def visualize_workflow(workflow, output_format: str = "mermaid") -> Optional[Path]:
    """
    Quick utility to visualize workflow
    
    Args:
        workflow: LangGraph workflow
        output_format: "mermaid", "png", or "both"
    
    Returns:
        Path to saved file(s)
    """
    if output_format in ["mermaid", "both"]:
        mermaid = graph_visualizer.generate_mermaid_diagram(workflow)
        mermaid_path = graph_visualizer.save_mermaid_to_file(mermaid)
        logger.info(f"Mermaid diagram: {mermaid_path}")
    
    if output_format in ["png", "both"]:
        png_path = graph_visualizer.export_as_png(workflow)
        if png_path:
            logger.info(f"PNG diagram: {png_path}")
    
    return graph_visualizer.output_dir
