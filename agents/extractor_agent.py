"""
Extractor Agent - Extracts structured fields from documents
"""
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

from graph.state import DocumentState
from schemas.document_schemas import (
    DocumentType,
    ExtractionResult,
    ResponsibleAILog,
    FinancialDocumentFields,
    ContractFields,
    ResumeFields,
    JobOfferFields,
    MedicalRecordFields,
    IdDocumentFields,
    AcademicFields,
    GeneralDocumentFields,
)
from utils.llm_client import llm_client, LLMProvider
from utils.logger import logger


class ExtractorAgent:
    """
    Agent responsible for extracting structured fields from documents
    
    Uses Claude Haiku + optional FAISS semantic lookup for context.
    Supports chunking for long documents.
    """
    
    # Schema definitions for each document type (OPTIMIZED - 6 CORE TYPES)
    SCHEMA_MAP = {
        DocumentType.FINANCIAL_DOCUMENT: FinancialDocumentFields,
        DocumentType.RESUME: ResumeFields,
        DocumentType.JOB_OFFER: JobOfferFields,
        DocumentType.MEDICAL_RECORD: MedicalRecordFields,
        DocumentType.ID_DOCUMENT: IdDocumentFields,
        DocumentType.ACADEMIC: AcademicFields,
    }
    
    # Few-shot examples for better extraction accuracy — one full example per doc type
    # showing EVERY schema field so the LLM knows exactly what to return.
    FEW_SHOT_EXAMPLES = {
        DocumentType.FINANCIAL_DOCUMENT: """
EXAMPLE - Financial Document (Invoice):
Input: "INVOICE #INV-2024-001 | Date: March 15, 2024 | Due: April 15, 2024
FROM: CloudTech LLC, 500 Tech Park, San Jose CA 95110
TO: Acme Corp, 200 Business Ave, New York NY 10001
Items: Cloud Hosting x1 = $1,200.00, Support Plan x1 = $300.00
Subtotal: $1,500.00 | Tax (8%): $120.00 | Total: $1,620.00
Payment Method: Bank Transfer"
Output: {
  "document_number": "INV-2024-001",
  "document_date": "2024-03-15",
  "due_date": "2024-04-15",
  "issuer_name": "Cloudtech Llc",
  "issuer_address": "500 Tech Park, San Jose CA 95110",
  "recipient_name": "Acme Corp",
  "recipient_address": "200 Business Ave, New York NY 10001",
  "total_amount": 1620.00,
  "tax_amount": 120.00,
  "currency": "USD",
  "payment_method": "Bank Transfer",
  "line_items": [
    {"description": "Cloud Hosting", "quantity": 1, "unit_price": 1200.00, "total": 1200.00},
    {"description": "Support Plan", "quantity": 1, "unit_price": 300.00, "total": 300.00}
  ]
}
""",
        DocumentType.RESUME: """
EXAMPLE - Resume / CV:
Input: "JOHN DOE | john@email.com | (555) 123-4567 | New York, NY
LinkedIn: linkedin.com/in/johndoe
SUMMARY: Experienced data scientist specializing in NLP and predictive modeling.
SKILLS: Python, ML, AWS, SQL
EDUCATION: MS Computer Science, Stanford University, May 2019, GPA: 3.9
EXPERIENCE: Senior Data Scientist, TechCorp (Jan 2020 - Present)
  - Built ML pipelines, Led team of 4 engineers
CERTIFICATIONS: AWS Certified ML Specialist, Google Cloud Professional
LANGUAGES: English (Native), Spanish (Intermediate)"
Output: {
  "candidate_name": "John Doe",
  "email": "john@email.com",
  "phone": "(555) 123-4567",
  "address": "New York, NY",
  "linkedin_url": "linkedin.com/in/johndoe",
  "summary": "Experienced data scientist specializing in NLP and predictive modeling.",
  "education": [
    {"degree": "Master Of Science In Computer Science", "institution": "Stanford University", "graduation_date": "2019-05-01", "gpa": 3.9}
  ],
  "work_experience": [
    {"job_title": "Senior Data Scientist", "employer": "Techcorp", "start_date": "2020-01-01", "end_date": null, "responsibilities": ["Built ML pipelines", "Led team of 4 engineers"]}
  ],
  "skills": ["Python", "ML", "AWS", "SQL"],
  "certifications": ["AWS Certified ML Specialist", "Google Cloud Professional"],
  "languages": ["English (Native)", "Spanish (Intermediate)"]
}
""",
        DocumentType.MEDICAL_RECORD: """
EXAMPLE - Medical Record:
Input: "Patient: Sarah Williams | DOB: 03/22/1985 | Patient ID: MED-98765
Department: Endocrinology | Visit Date: December 5, 2023
Physician: Dr. James Anderson
Diagnosis: Type 2 Diabetes Mellitus (E11.9)
Prescribed: Metformin 500mg twice daily, Lisinopril 10mg daily
Lab: HbA1c 7.2%, Fasting Glucose 145 mg/dL
Follow-up: January 10, 2024
Notes: Patient advised lifestyle modifications and dietary changes."
Output: {
  "patient_name": "Sarah Williams",
  "patient_id": "MED-98765",
  "date_of_birth": "1985-03-22",
  "visit_date": "2023-12-05",
  "physician_name": "Dr. James Anderson",
  "department": "Endocrinology",
  "diagnosis": "Type 2 Diabetes Mellitus (E11.9)",
  "prescribed_medications": ["Metformin 500mg twice daily", "Lisinopril 10mg daily"],
  "lab_results": [
    {"test": "HbA1c", "result": "7.2%"},
    {"test": "Fasting Glucose", "result": "145 mg/dL"}
  ],
  "follow_up_date": "2024-01-10",
  "notes": "Patient advised lifestyle modifications and dietary changes."
}
""",
        DocumentType.JOB_OFFER: """
EXAMPLE - Job Offer:
Input: "TechCorp Inc. | OFFER LETTER
Candidate: Michael Chen | Position: Senior Software Engineer
Start Date: January 15, 2024 | Salary: $145,000/year
Location: San Francisco, CA | Reports to: VP Engineering
Benefits: Health, Dental, Vision, 401k (4% match), 20 days PTO
Offer valid until: December 31, 2023"
Output: {
  "company_name": "TechCorp Inc.",
  "candidate_name": "Michael Chen",
  "position_title": "Senior Software Engineer",
  "start_date": "2024-01-15",
  "salary": "145000",
  "employment_type": "full-time",
  "work_location": "San Francisco, CA",
  "reporting_to": "VP Engineering",
  "department": null,
  "offer_date": null,
  "deadline_to_accept": "2023-12-31",
  "benefits": ["Health", "Dental", "Vision", "401k (4% match)", "20 days PTO"],
  "conditions": []
}
""",
        DocumentType.ID_DOCUMENT: """
EXAMPLE - Identity Document (Passport):
Input: "PASSPORT | United States of America
Surname: JOHNSON  Given Names: EMILY ROSE
Date of Birth: 15 APR 1990  Sex: F
Nationality: USA  Place of Birth: New York, NY
Personal No: 987-65-4321
Passport No: A12345678  Issued: 10 JAN 2020  Expires: 09 JAN 2030
Issuing Authority: U.S. Department of State
Address: 123 Main Street, New York, NY 10001"
Output: {
  "document_type": "Passport",
  "document_number": "A12345678",
  "full_name": "Emily Rose Johnson",
  "date_of_birth": "1990-04-15",
  "gender": "Female",
  "nationality": "American",
  "place_of_birth": "New York, NY",
  "address": "123 Main Street, New York, NY 10001",
  "issue_date": "2020-01-10",
  "expiration_date": "2030-01-09",
  "issuing_authority": "U.S. Department of State"
}
""",
        DocumentType.ACADEMIC: """
EXAMPLE - Academic Document (Transcript):
Input: "OFFICIAL TRANSCRIPT | STANFORD UNIVERSITY
Student: Alex Rivera  |  ID: STU-20190456
Program: Bachelor of Science in Computer Science
Graduation: June 15, 2023  |  GPA: 3.87 / 4.0
Honors: Magna Cum Laude, Dean's List (4 semesters)
Courses: CS101 Intro to Programming A, CS201 Data Structures A-, MATH301 Linear Algebra B+"
Output: {
  "document_type": "Transcript",
  "student_name": "Alex Rivera",
  "student_id": "STU-20190456",
  "institution_name": "Stanford University",
  "degree_program": "Bachelor of Science in Computer Science",
  "graduation_date": "2023-06-15",
  "gpa": 3.87,
  "doi": null,
  "courses": [
    {"course": "CS101 Intro to Programming", "grade": "A"},
    {"course": "CS201 Data Structures", "grade": "A-"},
    {"course": "MATH301 Linear Algebra", "grade": "B+"}
  ],
  "honors": ["Magna Cum Laude", "Dean's List (4 semesters)"]
}
"""
    }

    # Schema field lists shown in the prompt so LLM knows EXACTLY what to return
    SCHEMA_FIELDS = {
        DocumentType.FINANCIAL_DOCUMENT: [
            "document_number", "document_date", "due_date", "issuer_name", "issuer_address",
            "recipient_name", "recipient_address", "total_amount", "tax_amount", "currency",
            "payment_method", "line_items"
        ],
        DocumentType.RESUME: [
            "candidate_name", "email", "phone", "address", "linkedin_url", "summary",
            "education", "work_experience", "skills", "certifications", "languages"
        ],
        DocumentType.JOB_OFFER: [
            "candidate_name", "company_name", "position_title", "offer_date", "start_date",
            "salary", "employment_type", "work_location", "department", "reporting_to",
            "benefits", "conditions", "deadline_to_accept"
        ],
        DocumentType.MEDICAL_RECORD: [
            "patient_name", "patient_id", "date_of_birth", "visit_date", "physician_name",
            "department", "diagnosis", "prescribed_medications", "lab_results",
            "follow_up_date", "notes"
        ],
        DocumentType.ID_DOCUMENT: [
            "document_type", "document_number", "full_name", "date_of_birth", "gender",
            "nationality", "place_of_birth", "address", "issue_date", "expiration_date",
            "issuing_authority"
        ],
        DocumentType.ACADEMIC: [
            "document_type", "student_name", "student_id", "institution_name",
            "degree_program", "graduation_date", "gpa", "doi", "courses", "honors"
        ],
    }

    EXTRACTION_PROMPT_TEMPLATE = """You are extracting structured data from a {doc_type} document.

STEP 1 — Read the example carefully to understand expected field names and format:
{few_shot_examples}

STEP 2 — Extract EVERY field listed below from the document. You MUST include ALL fields in your JSON output, even if the value is null or [].

REQUIRED FIELDS FOR {doc_type_upper}:
{field_checklist}

STEP 3 — Rules:
- Use EXACTLY the field names listed above — no aliases, no renamed keys
- dates → YYYY-MM-DD format
- money → float without $ or commas
- names → Title Case
- list fields (work_experience, education, skills, certifications, medications, courses, etc.) → ALWAYS return as array, use [] if none
- work_experience: each entry MUST have: job_title, employer, start_date, end_date, responsibilities
- education: each entry MUST have: degree, institution, graduation_date, gpa
- NEVER skip a field — use null for missing scalars, [] for missing lists

DOCUMENT:
{document_text}

Return ONLY valid JSON. No markdown, no explanation, no extra keys outside the list above."""
    
    SYSTEM_PROMPT = "Document extraction engine. Return JSON only with exact schema field names. Dates=YYYY-MM-DD, money=float, missing=null."
    
    def __init__(self):
        self.name = "ExtractorAgent"
        logger.info(f"{self.name} initialized")
    
    def _camel_to_snake(self, name: str) -> str:
        """
        Convert camelCase or PascalCase to snake_case
        
        Args:
            name: Field name in camelCase/PascalCase
        
        Returns:
            Field name in snake_case
        """
        import re
        # Insert underscore before uppercase letters (except at start)
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return snake
    
    def _normalize_field_names(self, data: Any) -> Any:
        """
        Recursively normalize all dictionary keys from camelCase to snake_case
        
        Args:
            data: Dictionary, list, or primitive value
        
        Returns:
            Data structure with normalized field names
        """
        if isinstance(data, dict):
            # Normalize keys and recursively process values
            return {self._camel_to_snake(key): self._normalize_field_names(value) 
                    for key, value in data.items()}
        elif isinstance(data, list):
            # Recursively process list items
            return [self._normalize_field_names(item) for item in data]
        else:
            # Primitive value - return as-is
            return data
    
    def _normalize_extracted_values(self, data: Any) -> Any:
        """
        Normalize extracted values for better evaluation matching
        Handles dates, amounts, names, etc.
        
        Args:
            data: Extracted data structure
        
        Returns:
            Normalized data structure
        """
        from datetime import datetime
        import re
        
        if isinstance(data, dict):
            # Recursively normalize dict values
            return {key: self._normalize_extracted_values(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            # Recursively normalize list items
            return [self._normalize_extracted_values(item) for item in data]
        
        elif isinstance(data, str):
            # Normalize string values
            
            # 1. Date normalization: Try to parse and format as YYYY-MM-DD
            date_formats = [
                '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
                '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y',
                '%d %B %Y', '%d %b %Y', '%Y-%m-%dT%H:%M:%S'
            ]
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(data.strip(), fmt)
                    # Return in standard YYYY-MM-DD format
                    return parsed_date.strftime('%Y-%m-%d')
                except (ValueError, AttributeError):
                    continue
            
            # 2. Name normalization: Title Case for names
            # Check if this looks like a name (contains only letters and spaces)
            if data.strip() and all(c.isalpha() or c.isspace() for c in data.strip()):
                # Convert to Title Case
                return data.strip().title()
            
            # 3. Return cleaned string (strip whitespace)
            return data.strip()
        
        else:
            # Return primitive values as-is (int, float, bool, None)
            return data
    
    def _chunk_text(self, text: str, max_length: int = 6000, overlap: int = 500) -> list[str]:
        """
        Split text into chunks if too long, with overlap to prevent data loss
        
        Args:
            text: Full text
            max_length: Maximum chunk length (6000 for Claude's large context window)
                       Increased from 3500 to handle most documents in single chunk
            overlap: Characters to overlap between chunks (prevents split field loss)
        
        Returns:
            List of text chunks with overlap
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        # Track overlap words for next chunk
        overlap_words = []
        overlap_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_length:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start next chunk with overlap from end of previous chunk
                current_chunk = overlap_words.copy() + [word]
                current_length = overlap_length + word_length
                overlap_words = []
                overlap_length = 0
            else:
                current_chunk.append(word)
                current_length += word_length
                
                # Track last N characters for overlap
                overlap_words.append(word)
                overlap_length += word_length
                
                # Keep only last overlap characters
                while overlap_length > overlap and len(overlap_words) > 1:
                    removed_word = overlap_words.pop(0)
                    overlap_length -= len(removed_word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.info(f"Text split into {len(chunks)} chunks with {overlap}-char overlap")
        return chunks
    
    def _extract_from_chunk(
        self,
        chunk: str,
        doc_type: DocumentType,
        force_provider=None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract fields from a single text chunk

        Args:
            chunk: Text chunk
            doc_type: Document type

        Returns:
            Tuple of (extracted fields dict, llm response dict)
        """
        # Get few-shot example for this document type
        few_shot_examples = self.FEW_SHOT_EXAMPLES.get(doc_type, "")
        schema_fields = self.SCHEMA_FIELDS.get(doc_type, [])
        field_checklist = "\n".join(f"  - {f}" for f in schema_fields)

        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
            doc_type=doc_type.value,
            doc_type_upper=doc_type.value.upper().replace("_", " "),
            few_shot_examples=few_shot_examples,
            field_checklist=field_checklist,
            document_text=chunk
        )
        
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1500,
            force_provider=force_provider,
            groq_model="llama-3.3-70b-versatile",  # 70b for best extraction quality
            groq_key=2  # Secondary key (heavy: ~1500 tok output)
        )
        
        # Parse JSON response with robust extraction
        content = response["content"].strip()
        
        # Log response details
        logger.debug(f"Extractor LLM response length: {len(content)} chars")
        
        # FIX 1: Remove double braces (Claude sometimes adds extra layer)
        if content.startswith('{{') and content.endswith('}}'):
            content = content[1:-1].strip()
            logger.debug("Removed double braces from JSON")
        
        # Strategy 1: Direct JSON parsing
        try:
            extracted = json.loads(content)
            logger.debug("Parsed JSON using Strategy 1 (direct)")
            return extracted, response
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 1 failed at position {e.pos}: {e.msg}")
            pass
        
        # Strategy 2: Remove text before JSON (handle LLM preambles)
        try:
            # Find first { and last }
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                # Remove double braces if present
                if json_str.startswith('{{') and json_str.endswith('}}'):
                    json_str = json_str[1:-1].strip()
                extracted = json.loads(json_str)
                logger.debug("Parsed JSON using Strategy 2 (extracted from text)")
                return extracted, response
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Strategy 2 failed: {e}")
            pass
        
        # Strategy 3: Extract from markdown code blocks
        if "```json" in content:
            try:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    json_str = content[json_start:json_end].strip()
                    extracted = json.loads(json_str)
                    logger.debug("Parsed JSON using Strategy 3 (```json)")
                    return extracted, response
            except json.JSONDecodeError as e:
                logger.debug(f"Strategy 2a failed: {e}")
        
        if "```" in content:
            try:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end != -1:
                    json_str = content[json_start:json_end].strip()
                    extracted = json.loads(json_str)
                    logger.debug("Parsed JSON using Strategy 2b (```)")
                    return extracted, response
            except json.JSONDecodeError as e:
                logger.debug(f"Strategy 2b failed: {e}")
        
        # Strategy 3: Find JSON object by braces (handle nested objects and strings)
        try:
            first_brace = content.find('{')
            if first_brace != -1:
                brace_count = 0
                in_string = False
                escape_next = False
                
                for i in range(first_brace, len(content)):
                    char = content[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\\\':
                        escape_next = True
                        continue
                    
                    if char == '\"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = content[first_brace:i+1]
                                extracted = json.loads(json_str)
                                logger.debug(f"Parsed JSON using Strategy 3 (brace counting)")
                                return extracted, response
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 3 failed: {e}")
        except Exception as e:
            logger.debug(f"Strategy 3 exception: {e}")
        
        # Strategy 4: Try to auto-complete truncated JSON
        def _count_open_structures(s: str):
            """Count unclosed braces and brackets, ignoring string contents."""
            open_b, open_br = 0, 0
            in_str, esc = False, False
            for ch in s:
                if esc:
                    esc = False; continue
                if ch == '\\':
                    esc = True; continue
                if ch == '"':
                    in_str = not in_str; continue
                if not in_str:
                    if ch == '{': open_b += 1
                    elif ch == '}': open_b -= 1
                    elif ch == '[': open_br += 1
                    elif ch == ']': open_br -= 1
            return open_b, open_br

        try:
            first_brace = content.find('{')
            if first_brace != -1:
                json_str = content[first_brace:]
                if json_str.startswith('{{'):
                    json_str = json_str[1:]

                open_braces, open_brackets = _count_open_structures(json_str)

                if open_braces > 0 or open_brackets > 0:
                    logger.warning(f"JSON truncated: {open_braces} unclosed braces, {open_brackets} unclosed brackets")
                    json_str = json_str.rstrip()

                    # Step 1: Cut back to the last fully-closed value
                    # Find the last position where all strings are closed (even quote count so far)
                    last_safe = 0
                    in_str2, esc2 = False, False
                    b2, br2 = 0, 0
                    for i, ch in enumerate(json_str):
                        if esc2:
                            esc2 = False; continue
                        if ch == '\\':
                            esc2 = True; continue
                        if ch == '"':
                            in_str2 = not in_str2; continue
                        if not in_str2:
                            if ch in ('{', '['): b2 += (ch == '{'); br2 += (ch == '[')
                            elif ch == '}': b2 -= 1
                            elif ch == ']': br2 -= 1
                            # Record a "safe" position after every complete value at top-level
                            if ch in (',', '}', ']') and not in_str2:
                                last_safe = i + 1

                    # Cut to last safe point and strip trailing comma
                    if last_safe > 0:
                        json_str = json_str[:last_safe].rstrip().rstrip(',')

                    # Step 2: Recount and close
                    open_braces, open_brackets = _count_open_structures(json_str)
                    for _ in range(open_brackets):
                        json_str += ']'
                    for _ in range(open_braces):
                        json_str += '}'

                    logger.debug(f"Auto-completing JSON: closing {open_brackets} brackets, {open_braces} braces")
                    extracted = json.loads(json_str)
                    logger.warning("Parsed truncated JSON using Strategy 4 (auto-complete)")
                    return extracted, response
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 4 failed: {e}")
        except Exception as e:
            logger.debug(f"Strategy 4 exception: {e}")
        
        # All strategies failed
        logger.error(f"Failed to parse extraction response. Length: {len(content)}")
        logger.error(f"First 1000 chars: {content[:1000]}")
        logger.error(f"Last 1000 chars: {content[-1000:] if len(content) > 1000 else content}")
        raise ValueError(f"Could not parse JSON from response. Length: {len(content)}, First 500 chars: {content[:500]}")
    
    def _merge_extracted_fields(
        self,
        chunks_results: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge extraction results from multiple chunks
        
        Args:
            chunks_results: List of extraction results from chunks
        
        Returns:
            Merged fields dict
        """
        if len(chunks_results) == 1:
            return chunks_results[0]
        
        # Simple merge strategy: take non-null values, prefer later chunks
        merged = {}
        for chunk_result in chunks_results:
            for key, value in chunk_result.items():
                if value is not None:
                    if isinstance(value, list):
                        # Extend lists
                        merged.setdefault(key, []).extend(value)
                    else:
                        # Overwrite with non-null values
                        merged[key] = value
        
        return merged
    
    def extract(self, state: DocumentState) -> DocumentState:
        """
        Extract structured fields from document
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with extraction result
        """
        logger.info(f"{self.name}: Starting extraction")
        start_time = time.time()
        
        try:
            doc_type = state["doc_type"]
            raw_text = state["raw_text"]
            
            text_excerpt = raw_text[:500]
            
            # Chunk text if needed
            chunks = self._chunk_text(raw_text)
            logger.info(f"Processing {len(chunks)} chunks")
            
            # Extract from each chunk
            chunk_results = []
            llm_responses = []
            total_tokens = 0
            
            # Get schema definition only needed for self-repair — not sent in initial extraction
            from utils.llm_client import LLMProvider
            for i, chunk in enumerate(chunks, 1):
                logger.debug(f"Extracting from chunk {i}/{len(chunks)}")
                extracted, response = self._extract_from_chunk(
                    chunk=chunk,
                    doc_type=doc_type
                )
                chunk_results.append(extracted)
                llm_responses.append(response)
                total_tokens += response["tokens"]["input"] + response["tokens"]["output"]
            
            # Merge results
            merged_fields = self._merge_extracted_fields(chunk_results)
            
            # Normalize field names: camelCase -> snake_case
            normalized_fields = self._normalize_field_names(merged_fields)
            
            # Normalize values: dates to YYYY-MM-DD, names to Title Case, etc.
            normalized_fields = self._normalize_extracted_values(normalized_fields)
            
            # Apply FIELD_ALIASES to catch any remaining field name variations
            # e.g. "company" → "company_name", "role" → "position_title"
            from agents.validator_agent import ValidatorAgent
            aliases = ValidatorAgent.FIELD_ALIASES
            aliased_fields = {}
            for key, value in normalized_fields.items():
                canonical = aliases.get(key.lower(), key)
                # Always store under the canonical name.
                # A non-null value may overwrite a previously stored null for the same canonical.
                if canonical not in aliased_fields or value not in (None, "", [], {}):
                    aliased_fields[canonical] = value
            normalized_fields = aliased_fields
            
            latency = time.time() - start_time
            
            # Use last response for logging (representative)
            last_response = llm_responses[-1] if llm_responses else {}
            
            # Create result
            extraction_result = ExtractionResult(
                doc_type=doc_type,
                extracted_fields=normalized_fields,
                confidence=0.85,  # TODO: compute actual confidence
                chunk_count=len(chunks),
                timestamp=datetime.utcnow()
            )
            
            # Update state
            state["extracted_fields"] = normalized_fields
            state["extraction_result"] = extraction_result
            state["agent_timings"][self.name] = latency
            
            # Responsible AI logging
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=text_excerpt,
                    output_data=json.dumps(merged_fields),
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=last_response.get("model", "claude-haiku"),
                    tokens_used=total_tokens,
                    error_occurred=False,
                    llm_provider=last_response.get("provider", "unknown"),
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=last_response.get("user_prompt", ""),
                    context_data={
                        "doc_type": doc_type.value,
                        "chunk_count": len(chunks),
                        "text_length": len(raw_text)
                    },
                    raw_output=last_response.get("content", ""),
                    tokens_input=sum(r["tokens"]["input"] for r in llm_responses),
                    tokens_output=sum(r["tokens"]["output"] for r in llm_responses),
                    retry_attempt=0
                )
            )
            
            logger.info(
                f"{self.name}: Extraction complete",
                doc_type=doc_type.value,
                fields_count=len(merged_fields),
                chunks=len(chunks),
                latency_ms=latency * 1000
            )
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Extraction failed", error=str(e))
            
            # Update state with error
            state["errors"].append(f"{self.name}: {str(e)}")
            state["extracted_fields"] = {}
            state["extraction_result"] = ExtractionResult(
                doc_type=state["doc_type"],
                extracted_fields={},
                confidence=0.0,
                chunk_count=0
            )
            
            # Log error
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=state["raw_text"][:500],
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
                        "doc_type": state.get("doc_type", DocumentType.FINANCIAL_DOCUMENT).value if state.get("doc_type") else "unknown",
                        "text_length": len(state.get("raw_text", ""))
                    },
                    raw_output="",
                    tokens_input=0,
                    tokens_output=0,
                    retry_attempt=0
                )
            )
            
            return state


# Agent instance
extractor_agent = ExtractorAgent()
