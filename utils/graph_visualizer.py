"""
LangGraph Visualization Utilities
"""
from utils.logger import logger


class GraphVisualizer:
    """
    Utility class for LangGraph visualization
    
    Provides:
    1. Graph structure visualization (Mermaid diagrams)
    2. Execution trace visualization
    """
    
    def __init__(self):
        """Initialize visualizer"""
        logger.info("GraphVisualizer initialized")
    
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
        diagram = ["graph TD"]
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
