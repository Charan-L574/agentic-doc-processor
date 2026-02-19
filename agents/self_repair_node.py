"""
Self-Repair Node - Fixes invalid JSON using LLM
"""
import time
import json
from datetime import datetime
from typing import Dict, Any

from graph.state import DocumentState
from schemas.document_schemas import ValidationStatus, ResponsibleAILog
from utils.llm_client import llm_client
from utils.logger import logger


class SelfRepairNode:
    """
    Self-repair node that uses LLM to fix validation errors or re-extract fields
    
    Two modes:
    1. Repair Mode: Fix validation errors in existing extracted fields
    2. Re-extraction Mode: Extract fields when none/few were extracted initially
    """
    
    REPAIR_PROMPT_TEMPLATE = """Fix validation errors in extracted data fields.

**Extracted Fields:**
{extracted_fields}

**Errors to Fix:**
{errors}

**Reference Text:**
{text_excerpt}

**Task:** Correct the errors and return valid JSON with fixed fields.

**Output Format:** Return only the corrected JSON object."""
    
    RE_EXTRACTION_PROMPT_TEMPLATE = """Re-extract fields from this {doc_type} document. Current accuracy: {current_accuracy}% — target 90-100%.

SCHEMA FIELDS (extract all):
{schema_fields}

PREVIOUS EXTRACTION (keep non-null values, fill in the rest):
{extracted_fields}

MISSING FIELDS (focus here):
{missing_fields_list}

ISSUES:
{validation_errors}

DOCUMENT:
{document_text}

Return ONLY JSON with all schema fields (keep existing + fill missing). Missing=null. No markdown."""
    
    SYSTEM_PROMPT = "Extract and repair data fields. Return valid JSON only."
    
    def __init__(self):
        self.name = "SelfRepairNode"
        # Import schema map here to avoid circular imports
        from agents.extractor_agent import ExtractorAgent
        self.schema_map = ExtractorAgent.SCHEMA_MAP
        logger.info(f"{self.name} initialized")
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response with robust extraction
        
        Args:
            content: LLM response content
        
        Returns:
            Parsed JSON dict
        
        Raises:
            ValueError if parsing fails
        """
        original_content = content
        content = content.strip()
        
        # FIX: Remove double braces (Claude artifact)
        if content.startswith('{{') and content.endswith('}}'):
            content = content[1:-1].strip()
            logger.debug("Removed double braces from repair response")
        
        # Strategy 1: Direct JSON parsing
        try:
            result = json.loads(content)
            logger.debug(f"Parsed JSON using Strategy 1 (direct)")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Extract from markdown code blocks
        if "```json" in content:
            try:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    json_str = content[json_start:json_end].strip()
                    result = json.loads(json_str)
                    logger.debug(f"Parsed JSON using Strategy 2 (```json)")
                    return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Strategy 2a failed: {e}")
        
        if "```" in content:
            try:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end != -1:
                    json_str = content[json_start:json_end].strip()
                    result = json.loads(json_str)
                    logger.debug(f"Parsed JSON using Strategy 2b (```)")
                    return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Strategy 2b failed: {e}")
        
        # Strategy 3: Find JSON object by braces (handle nested objects + truncation)
        try:
            first_brace = content.find('{')
            if first_brace != -1:
                # Skip double brace if present
                if content[first_brace:first_brace+2] == '{{':
                    first_brace += 1
                
                brace_count = 0
                in_string = False
                escape_next = False
                last_valid_end = -1
                
                for i in range(first_brace, len(content)):
                    char = content[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = content[first_brace:i+1]
                                result = json.loads(json_str)
                                logger.debug(f"Parsed JSON using Strategy 3 (brace counting)")
                                return result
                            else:
                                # Track last complete sub-object
                                last_valid_end = i
                
                # If we got here, JSON is truncated. Try to salvage what we can.
                if last_valid_end > first_brace:
                    # Find last comma before truncation
                    json_portion = content[first_brace:last_valid_end+1]
                    last_comma = json_portion.rfind(',')
                    if last_comma > 0:
                        json_portion = json_portion[:last_comma]
                    
                    # Close remaining braces
                    open_braces = json_portion.count('{') - json_portion.count('}')
                    for _ in range(open_braces):
                        json_portion += '}'
                    
                    try:
                        result = json.loads(json_portion)
                        logger.warning(f"Recovered truncated JSON with {len(result)} fields")
                        return result
                    except:
                        pass
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 3 failed: {e}")
        except Exception as e:
            logger.debug(f"Strategy 3 exception: {e}")
        
        # All strategies failed - log full response for debugging
        logger.error(f"Failed to parse JSON after all strategies. Response length: {len(original_content)}")
        logger.error(f"First 1000 chars: {original_content[:1000]}")
        logger.error(f"Last 500 chars: {original_content[-500:] if len(original_content) > 500 else original_content}")
        raise ValueError(f"Could not parse JSON from response. Length: {len(original_content)}, First 500 chars: {original_content[:500]}")
    
    def _get_schema_fields(self, doc_type) -> str:
        """
        Get schema field names for document type
        
        Args:
            doc_type: Document type
        
        Returns:
            Formatted string of expected field names
        """
        from schemas.document_schemas import DocumentType
        
        schema_class = self.schema_map.get(doc_type)
        if not schema_class:
            return "No specific schema available"
        
        # Get field names from Pydantic model
        try:
            fields = schema_class.model_fields.keys()
            return "\n".join(f"  - {field}" for field in fields)
        except:
            return "Unable to extract schema fields"
    
    def _should_re_extract(self, extracted_fields: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Determine if fields are too sparse or below accuracy threshold — needs re-extraction.

        Triggers re-extraction when:
        - fewer than 3 non-null fields  (always), OR
        - validator stored accuracy < 90 % in state
        """
        if not extracted_fields:
            return True

        # Count non-null fields
        non_null_fields = sum(1 for v in extracted_fields.values() if v not in [None, "", [], {}])
        if non_null_fields < 3:
            logger.info(f"Only {non_null_fields} non-null fields extracted, triggering re-extraction")
            return True

        # Accuracy-based trigger
        current_accuracy = state.get("current_accuracy", 1.0)
        if current_accuracy < 0.90:
            logger.info(
                f"Accuracy {current_accuracy:.0%} below 90% threshold, triggering re-extraction"
            )
            return True

        return False
    
    def repair(self, state: DocumentState) -> DocumentState:
        """
        Attempt to repair invalid extracted fields
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with repaired fields
        """
        logger.info(f"{self.name}: Starting self-repair")
        start_time = time.time()
        
        # Check repair attempt limit
        max_attempts = 3  # Match workflow GraphConfig.max_repair_attempts
        current_attempts = state.get("repair_attempts", 0)
        
        if current_attempts >= max_attempts:
            logger.warning(f"{self.name}: Max repair attempts reached ({max_attempts})")
            # Mark as failed and stop
            state["validation_status"] = ValidationStatus.FAILED
            state["validation_errors"].append("Max repair attempts reached")
            return state
        
        state["repair_attempts"] = current_attempts + 1
        logger.info(f"{self.name}: Repair attempt {state['repair_attempts']} of {max_attempts}")
        
        try:
            extracted_fields = state["extracted_fields"] or {}
            validation_result = state.get("validation_result")
            errors = validation_result.errors if validation_result else []
            doc_type = state["doc_type"]
            
            # Determine if we need re-extraction or just repair
            needs_re_extraction = self._should_re_extract(extracted_fields, state)
            
            # Prepare text excerpt (longer for re-extraction)
            text_length = 1000 if needs_re_extraction else 500
            text_excerpt = " ".join(state.get("raw_text", state.get("text", "")).split()[:text_length])
            
            if needs_re_extraction:
                logger.info(f"{self.name}: Using RE-EXTRACTION mode (accuracy below 90% or sparse fields)")
                # Get schema fields for guidance
                schema_fields = self._get_schema_fields(doc_type)

                # Retrieve missing field list stored by validator (or compute fresh)
                missing_fields = state.get("missing_schema_fields", [])
                current_accuracy = state.get("current_accuracy", 0.0)
                accuracy_pct = int(round(current_accuracy * 100))

                if missing_fields:
                    missing_fields_list = "\n".join(f"  - {f}" for f in missing_fields)
                else:
                    missing_fields_list = "  (unknown — extract all schema fields)"

                prompt = self.RE_EXTRACTION_PROMPT_TEMPLATE.format(
                    doc_type=doc_type.value if hasattr(doc_type, 'value') else str(doc_type),
                    current_accuracy=accuracy_pct,
                    document_text=text_excerpt,
                    schema_fields=schema_fields,
                    extracted_fields=json.dumps(extracted_fields, indent=2),
                    missing_fields_list=missing_fields_list,
                    validation_errors="\n".join(f"- {error}" for error in errors) if errors else "No specific errors - fields are missing"
                )
            else:
                logger.info(f"{self.name}: Using REPAIR mode (fixing validation errors)")
                prompt = self.REPAIR_PROMPT_TEMPLATE.format(
                    extracted_fields=json.dumps(extracted_fields, indent=2),
                    errors="\n".join(f"- {error}" for error in errors),
                    text_excerpt=text_excerpt
                )
            
            # 70b on key-2 for quality repair without hitting key-1 rate limit
            response = llm_client.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1200,
                groq_model="llama-3.3-70b-versatile",
                groq_key=2  # Secondary key (heavy: ~1200 tok output)
            )
            
            latency = time.time() - start_time
            
            # Log response for debugging
            logger.debug(f"{self.name}: LLM response length: {len(response['content'])} chars")
            
            # Parse repaired fields
            try:
                repaired_fields = self._parse_llm_response(response["content"])
            except ValueError as parse_error:
                logger.error(f"{self.name}: JSON parsing failed: {parse_error}")
                logger.error(f"{self.name}: Full LLM response:\n{response['content']}")
                raise
            
            # Update state
            state["extracted_fields"] = repaired_fields
            state["repair_attempts"] = current_attempts + 1
            state["validation_status"] = ValidationStatus.REPAIRED.value
            state["agent_timings"][self.name] = latency
            
            # Add repair info to state
            if "repair_history" not in state:
                state["repair_history"] = []
            state["repair_history"].append({
                "attempt": current_attempts + 1,
                "errors_fixed": errors,
                "mode": "re_extraction" if needs_re_extraction else "repair"
            })
            
            # Responsible AI logging
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=json.dumps(extracted_fields),
                    output_data=json.dumps(repaired_fields),
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=response["model"],
                    tokens_used=response["tokens"]["input"] + response["tokens"]["output"],
                    error_occurred=False,
                    llm_provider=response["provider"],
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=response.get("user_prompt", ""),
                    context_data={
                        "repair_attempt": current_attempts + 1,
                        "max_attempts": 3,
                        "validation_errors": errors,
                        "field_count": len(extracted_fields),
                        "mode": "re_extraction" if needs_re_extraction else "repair",
                        "current_accuracy": state.get("current_accuracy", 0.0),
                        "missing_fields": state.get("missing_schema_fields", []),
                        "doc_type": doc_type.value if hasattr(doc_type, 'value') else str(doc_type)
                    },
                    raw_output=response["content"],
                    tokens_input=response["tokens"]["input"],
                    tokens_output=response["tokens"]["output"],
                    retry_attempt=current_attempts + 1
                )
            )
            
            logger.info(
                f"{self.name}: Repair complete",
                attempt=current_attempts + 1,
                latency_ms=latency * 1000
            )
            
            # Route back to validator
            state["needs_repair"] = False  # Will be re-evaluated by validator
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Repair failed", error=str(e))
            
            # Update state with error
            state["errors"].append(f"{self.name}: {str(e)}")
            state["repair_attempts"] = current_attempts + 1
            state["needs_repair"] = False  # Stop trying after error
            
            # Log error
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=str(state.get("extracted_fields", {})),
                    output_data="",
                    timestamp=datetime.utcnow(),
                    latency_ms=(time.time() - start_time) * 1000,
                    llm_model_used="unknown",
                    error_occurred=True,
                    error_message=str(e),
                    llm_provider="unknown",
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt="",
                    context_data={
                        "repair_attempt": current_attempts + 1,
                        "validation_errors": state.get("validation_result", {}).errors if hasattr(state.get("validation_result", {}), "errors") else []
                    },
                    raw_output="",
                    tokens_input=0,
                    tokens_output=0,
                    retry_attempt=current_attempts + 1
                )
            )
            
            return state


# Node instance
self_repair_node = SelfRepairNode()
