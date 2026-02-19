"""
Redactor Agent - Presidio + LLM PII detection and redaction
"""
import json
import re
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from graph.state import DocumentState
from schemas.document_schemas import PIIType, PIIDetection, RedactionResult, ResponsibleAILog
from utils.llm_client import llm_client
from utils.logger import logger


class RedactorAgent:
    """
    Hybrid agent responsible for detecting and redacting PII
    
    Uses Presidio + LLM hybrid approach:
    1. Presidio (rule-based + ML) for fast, accurate PII detection
    2. LLM for context-aware enhancement and complex patterns
    3. Merge results for comprehensive coverage
    
    NOTE: PII metrics (precision, recall) are calculated GLOBALLY across all documents
    in the evaluation script, not per-document. This provides overall model performance.
    """
    
    PII_DETECTION_PROMPT = """Identify ALL genuine personal identifiable information (PII) in the following text. Be THOROUGH but PRECISE.

**⚠️ CRITICAL: DO NOT detect domains/URLs ending with .com, .net, .org, .edu, .gov, .io, .ai, etc. These are NOT names!**
**⚠️ DO NOT detect text containing dots (.) unless it's a valid PII format (email with @, or proper date with separators)**
**⚠️ DO NOT detect concatenated dates like "march152024" or "april142024" - these are NOT real dates!**

**Text:**
{text}

**PII Types to Detect - Be THOROUGH but PRECISE:**

1. EMAIL: Complete email addresses
   - MUST have @ and domain
   - DO NOT detect domain parts alone ("acme.com") or username fragments ("john.doe")

2. PHONE: Phone numbers
   - MUST have 10+ digits
   - With formatting: "+1-555-1234", "(415) 555-1234"
   - Without formatting: "4155551234", "9876543210"
   - Various formatting: spaces, dashes, parentheses

3. SSN: National identification numbers
   - US SSN: 9 digits ("123-45-6789" or "123456789")
   - India Aadhaar: 12 digits ("1234 5678 9012")
   - Other formats: 9-12 digits with or without formatting
   - Detect partially redacted: "XXX-XX-1234", "XXXX XXXX 1234"

4. CREDIT_CARD: Credit/debit card numbers
   - 13-19 digits with optional spaces/dashes
   - Look for "Card ending in", "**** **** **** 1234"

5. BANK_ACCOUNT: Bank account numbers
   - 8-20 digits
   - DO NOT detect: Routing numbers, IFSC codes, SWIFT codes (bank identifiers)

6. TAX_ID: Tax identification numbers
   - India PAN: AAAAA9999A ("ABCDE1234F")
   - India GSTIN: 15 alphanumeric ("22AAAAA0000A1Z5")
   - Other tax IDs: Various formats

7. NAME: Full person names
   - MUST detect BOTH multi-word ("John Smith") AND single-word concatenated names ("davidpark", "robertthompson", "lisarodriguez")
   - Look for names even if concatenated without spaces: "sarahwilliams", "michaelchen"
   - Include titles if attached: "dr.sarahwilliams", "mr.johnsmith"
   - DO NOT detect: Company names, domains (.com, .net, .org), software/tech terms (github, linkedin), job titles

8. ADDRESS: COMPLETE street addresses
   - MUST include: Street/House number + Street name + City
   - Examples: "123 Main St, San Jose, CA 95110" or "Plot 45, MG Road, Bangalore 560001"
   - DO NOT detect: City names alone, postal codes alone

9. DATE_OF_BIRTH: Actual birthdates of people
   - MUST be in context of a person (near "DOB", "Birth", etc.)
   - DO NOT detect: Invoice dates, billing periods, random dates

10. MEDICAL_ID: Medical record numbers, health insurance numbers, patient IDs

**CRITICAL EXCLUSIONS (DO NOT detect):**
❌ Durations: "30 days", "6 years", "60 hrs", "monthly", "annually"
❌ City names alone (without full address)
❌ Month/year: "May 2019", "March 2024"
❌ Concatenated dates WITHOUT separators: "march152024", "april142024", "february282024"
❌ Domains/URLs: "acme.com", "company.net", "github.com", "linkedin.com", "gmail.com", "stanford.edu"
❌ Domain fragments: "acmecorp.com", "cloudtech.com", "pge.com", "mariosnyc.com"
❌ Email fragments: "john.doe" (without @)
❌ Tech terms: "Docker", "Python", "Kubernetes", "synaptic", "deeplearning"
❌ Latin phrases: "summa cum laude", "magna cum laude"
❌ Job titles: "Senior Engineer", "Manager"
❌ Company names
❌ Postal codes alone
❌ Bank identifiers: routing numbers, SWIFT codes, sort codes
❌ Invoice/document numbers: "INV-2024-001", "PO-12345"
❌ Alphanumeric codes: "wa1234567", "xx1234567", "r01", "i10"

**Response Format:**
{{
  "pii_detections": [
    {{
      "field_name": "email_address",
      "pii_type": "EMAIL",
      "original_text": "john@example.com",
      "redacted_text": "[EMAIL_REDACTED]",
      "confidence": 0.95
    }}
  ]
}}

CRITICAL: Return ONLY the JSON object. Start with {{ and end with }}. No explanations.
Be THOROUGH: Catch all real PII, even if formatting is imperfect. Better to over-detect (we'll filter) than miss actual PII."""
    
    # Presidio entity type mapping to our PIIType
    PRESIDIO_TO_PII_TYPE = {
        # ── Standard Presidio entities ──
        "EMAIL_ADDRESS": PIIType.EMAIL,
        "PHONE_NUMBER": PIIType.PHONE,
        "US_SSN": PIIType.SSN,
        "CREDIT_CARD": PIIType.CREDIT_CARD,
        "PERSON": PIIType.NAME,
        "LOCATION": PIIType.ADDRESS,
        "DATE_TIME": PIIType.DATE_OF_BIRTH,
        "MEDICAL_LICENSE": PIIType.MEDICAL_ID,
        "US_PASSPORT": PIIType.SSN,
        "US_DRIVER_LICENSE": PIIType.SSN,
        "US_BANK_NUMBER": PIIType.CREDIT_CARD,
        "US_ITIN": PIIType.SSN,
        "UK_NHS": PIIType.MEDICAL_ID,
        "IBAN_CODE": PIIType.CREDIT_CARD,
        "IP_ADDRESS": PIIType.NAME,
        "CRYPTO": PIIType.CREDIT_CARD,
        "NRP": PIIType.NAME,
        # ── Spanish ──
        "ES_NIF": PIIType.SSN,
        "ES_NIE": PIIType.SSN,
        # ── Italian ──
        "IT_DRIVER_LICENSE": PIIType.SSN,
        "IT_FISCAL_CODE": PIIType.SSN,
        "IT_VAT_CODE": PIIType.TAX_ID,
        "IT_IDENTITY_CARD": PIIType.SSN,
        "IT_PASSPORT": PIIType.SSN,
        # ── Polish ──
        "PL_PESEL": PIIType.SSN,
        # ── Indian custom entities ──
        "IN_AADHAAR": PIIType.SSN,
        "IN_PAN": PIIType.TAX_ID,
        "IN_GSTIN": PIIType.TAX_ID,
        "IN_PASSPORT": PIIType.SSN,
        "IN_VOTER_ID": PIIType.SSN,
        "IN_DRIVING_LICENSE": PIIType.SSN,
        "IN_UPI": PIIType.CREDIT_CARD,
        "IN_IFSC": PIIType.CREDIT_CARD,
    }
    
    def __init__(self):
        """Initialize RedactorAgent with Presidio + LLM (all languages + Indian recognizers)"""
        self.name = "RedactorAgent"

        try:
            from presidio_analyzer import PatternRecognizer, Pattern, RecognizerRegistry

            # ── Registry: English-only (content is always English) ──
            all_languages = ["en"]
            registry = RecognizerRegistry(supported_languages=all_languages)
            registry.load_predefined_recognizers(languages=all_languages)

            # ── Indian custom pattern recognizers (language="en" so they fire on en analysis) ──

            # Aadhaar: 12 digits, first digit 2-9 (XXXX XXXX XXXX)
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_AADHAAR",
                supported_language="en",
                patterns=[Pattern("AADHAAR", r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b", 0.85)],
                context=["aadhaar", "aadhar", "uid", "unique identification", "uidai"],
            ))

            # PAN Card: AAAAA9999A
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_PAN",
                supported_language="en",
                patterns=[Pattern("PAN", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", 0.90)],
                context=["pan", "permanent account number", "income tax"],
            ))

            # GSTIN: 15-char (22AAAAA0000A1Z5)
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_GSTIN",
                supported_language="en",
                patterns=[Pattern("GSTIN", r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", 0.92)],
                context=["gstin", "gst", "gstn", "goods and services tax"],
            ))

            # Indian Passport: letter + 7 digits
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_PASSPORT",
                supported_language="en",
                patterns=[Pattern("IN_PASSPORT", r"\b[A-PR-WY-Z][1-9]\d{6}[1-9]\b", 0.85)],
                context=["passport", "passport no", "passport number", "travel document"],
            ))

            # Indian Voter ID (EPIC): 3 letters + 7 digits
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_VOTER_ID",
                supported_language="en",
                patterns=[Pattern("VOTER_ID", r"\b[A-Z]{3}\d{7}\b", 0.80)],
                context=["voter id", "epic", "election card", "voter card"],
            ))

            # Indian Driving License: SS-RR-YYYY-NNNNNNN
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_DRIVING_LICENSE",
                supported_language="en",
                patterns=[Pattern("IN_DL", r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", 0.85)],
                context=["driving license", "driver license", "dl no", "driving licence"],
            ))

            # UPI ID: handle@bank
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_UPI",
                supported_language="en",
                patterns=[Pattern("UPI", r"\b[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}\b", 0.85)],
                context=["upi", "upi id", "upi payment", "vpa"],
            ))

            # Indian bank IFSC code: 4 letters + 0 + 6 alphanumeric
            registry.add_recognizer(PatternRecognizer(
                supported_entity="IN_IFSC",
                supported_language="en",
                patterns=[Pattern("IFSC", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", 0.88)],
                context=["ifsc", "ifsc code", "bank code", "neft", "rtgs"],
            ))

            self.analyzer = AnalyzerEngine(
                registry=registry,
                supported_languages=all_languages,
            )
            self.anonymizer = AnonymizerEngine()
            logger.info(
                f"{self.name} initialized (Presidio + LLM hybrid, "
                f"languages={all_languages}, 8 Indian recognizers loaded)"
            )
        except Exception as e:
            logger.warning(f"Presidio initialization failed: {e}. Falling back to LLM-only.")
            self.analyzer = None
            self.anonymizer = None
    
    def _detect_gender_patterns(self, text: str) -> List[PIIDetection]:
        """Detect gender using regex — catches 'Gender : Male', 'Sex: Female', standalone 'Male'/'Female'."""
        detections = []
        # Pattern 1: labelled gender fields  e.g. "Gender : Male", "Sex: F"
        labelled = re.finditer(
            r'(?i)\b(?:gender|sex)\s*[:\-]?\s*(male|female|m\b|f\b|other|non-binary|transgender)',
            text
        )
        for m in labelled:
            original = m.group(0).strip()
            gender_value = m.group(1).strip()
            detections.append(PIIDetection(
                field_name="gender",
                pii_type=PIIType.GENDER,
                original_text=original,
                redacted_text="[GENDER_REDACTED]",
                detection_source="regex",
                confidence=0.98
            ))
        # Pattern 2: standalone value right after newline/colon with label on same line
        # e.g. line "Gender : Male" not caught by pattern1
        standalone = re.finditer(
            r'(?i)(?<=\n)[ \t]*(male|female)[ \t]*(?:\r?\n|$)',
            text
        )
        seen = {d.original_text.lower() for d in detections}
        for m in standalone:
            original = m.group(0).strip()
            if original.lower() not in seen:
                detections.append(PIIDetection(
                    field_name="gender",
                    pii_type=PIIType.GENDER,
                    original_text=original,
                    redacted_text="[GENDER_REDACTED]",
                    detection_source="regex",
                    confidence=0.90
                ))
                seen.add(original.lower())
        return detections

    def _detect_pii_with_presidio(self, text: str) -> List[PIIDetection]:
        """        Detect PII using Presidio (fast, rule-based + ML)
        
        Args:
            text: Text to analyze
        
        Returns:
            List of detected PII
        """
        if not self.analyzer:
            return []
        
        try:
            # Analyze with Presidio
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=None  # Detect all supported entities
            )
            
            pii_detections = []
            for result in results:
                # Extract the actual text
                original_text = text[result.start:result.end]
                
                # ULTRA-LOW confidence for maximum recall (targeting 90%+ recall)
                MIN_CONFIDENCE = 0.20  # AGGRESSIVE: 0.25→0.20 for 90%+ recall
                if result.score < MIN_CONFIDENCE:
                    continue
                
                # Validate detection to reduce false positives (ULTRA-STRICT validation)
                is_valid = self._validate_pii_detection(result.entity_type, original_text, result.score)
                if not is_valid:
                    logger.debug(f"FILTERED Presidio: {result.entity_type}='{original_text}' (confidence={result.score:.2f})")
                    continue
                else:
                    logger.debug(f"ACCEPTED Presidio: {result.entity_type}='{original_text}' (confidence={result.score:.2f})")
                
                # Map Presidio entity type to our PIIType
                pii_type = self.PRESIDIO_TO_PII_TYPE.get(
                    result.entity_type,
                    PIIType.NAME  # Default fallback
                )
                
                # Create redacted version
                redacted_text = f"[{result.entity_type}_REDACTED]"
                
                detection = PIIDetection(
                    field_name=result.entity_type.lower(),
                    pii_type=pii_type,
                    original_text=original_text,
                    redacted_text=redacted_text,
                    detection_source="presidio",
                    confidence=result.score
                )
                pii_detections.append(detection)
            
            logger.info(f"Presidio detected {len(pii_detections)} PII instances (after confidence filtering)")
            return pii_detections
        
        except Exception as e:
            logger.error(f"Presidio PII detection failed: {e}")
            return []
    
    # Blacklist of common false positive patterns (COMPREHENSIVE)
    FALSE_POSITIVE_PATTERNS = {
        # Technology/software terms
        'docker', 'kubernetes', 'python', 'javascript', 'tensorflow', 'pytorch',
        'java', 'react', 'angular', 'vue', 'django', 'flask', 'aws', 'azure',
        # Common words often misdetected
        'customer', 'service', 'business', 'company', 'developer', 'engineer',
        'manager', 'director', 'senior', 'junior', 'lead', 'principal',
        # Food/restaurant terms
        'italian', 'chinese', 'mexican', 'french', 'japanese', 'thai',
        'salad', 'caesar', 'pasta', 'pizza', 'burger', 'sandwich', 'soup',
        'menu', 'restaurant', 'appetizer', 'entree', 'dessert',
        # Time-related (not dates)
        'annually', 'monthly', 'daily', 'weekly', 'yearly', 'quarterly',
        'hours', 'days', 'weeks', 'months', 'years', 'hrs', 'mins'
    }
    
    def _validate_pii_detection(self, entity_type: str, text: str, confidence: float) -> bool:
        """
        Validate PII detection to filter false positives (ULTRA-STRICT MODE)
        
        Args:
            entity_type: Presidio entity type (PERSON, DATE_TIME, PHONE_NUMBER, etc.)
            text: The detected text
            confidence: Confidence score
        
        Returns:
            True if valid, False if likely false positive
        """
        # Map Presidio entity_type to our PIIType for validation
        entity_to_pii_map = {
            "PERSON": "NAME",
            "DATE_TIME": "DATE_TIME",
            "PHONE_NUMBER": "PHONE",
            "US_SSN": "SSN",
            "EMAIL_ADDRESS": "EMAIL",
            "CREDIT_CARD": "CREDIT_CARD",
            "US_BANK_NUMBER": "BANK_ACCOUNT",
            "LOCATION": "ADDRESS",
            "MEDICAL_LICENSE": "MEDICAL_ID",
            "UK_NHS": "MEDICAL_ID",
            "US_PASSPORT": "SSN",
            "US_DRIVER_LICENSE": "SSN",
            "US_ITIN": "SSN",
            "ES_NIF": "SSN",
            "ES_NIE": "SSN",
            "IT_DRIVER_LICENSE": "SSN",
            "IT_FISCAL_CODE": "SSN",
            "IT_VAT_CODE": "SSN",
            "IT_IDENTITY_CARD": "SSN",
            "IT_PASSPORT": "SSN",
            "PL_PESEL": "SSN",
            "IN_AADHAAR": "SSN",
            "IN_PAN": "SSN",
            "IN_GSTIN": "SSN",
            "IN_PASSPORT": "SSN",
            "IN_VOTER_ID": "SSN",
            "IN_DRIVING_LICENSE": "SSN",
            "IN_UPI": "CREDIT_CARD",
            "IN_IFSC": "BANK_ACCOUNT",
        }
        
        # Convert entity_type to pii_type
        pii_type = entity_to_pii_map.get(entity_type, "NAME")  # Default to NAME for unknown types
        
        # Use the SAME ultra-strict validation as LLM detections
        return self._validate_llm_pii(pii_type, text, confidence)
    
    def _validate_llm_pii(self, pii_type: str, text: str, confidence: float) -> bool:
        """
        Validate LLM-detected PII to filter false positives (ULTRA-STRICT MODE)
        
        Args:
            pii_type: Type of PII (EMAIL, PHONE, NAME, etc.)
            text: The detected text
            confidence: Confidence score
        
        Returns:
            True if valid, False if likely false positive
        """
        text_lower = text.lower().strip()
        text_len = len(text)
        
        # ULTRA-CRITICAL #1 PRIORITY: Reject ALL domains/URLs IMMEDIATELY
        # This is the PRIMARY source of false NAME detections - MUST be first check!
        if pii_type == "NAME":
            # Reject if contains dot (ALL domain fragments: acmecorp.com, jennifer.ma, dr.ro)
            if '.' in text:
                logger.debug(f"🚫 REJECTED DOMAIN (LLM): pii_type={pii_type}, text='{text}'")
                return False
            
            # Reject if ends with domain extensions (even without dot)
            domain_extensions = ['com', 'net', 'org', 'edu', 'gov', 'io', 'ai', 'co', 'uk', 'de', 'fr', 'jp', 'cn', 'se', 'in']
            if any(text_lower.endswith(ext) for ext in domain_extensions):
                return False
            
            # Reject if contains www, http, or @
            if any(x in text_lower for x in ['www', 'http', '@', '//']):
                return False
        
        # ULTRA-EARLY REJECTION: Obviously invalid patterns (BEFORE any other checks)
        
        # Reject short alphanumeric codes that are clearly not PII
        if text_len <= 4 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
            return False  # Rejects: "r01", "i10", "wa12", etc.
        
        # Reject academic/technical abbreviations
        if text_lower in ['summa cum laude', 'magna cum laude', 'cum laude', 'phd', 'mba', 'bsc', 'msc', 'ba', 'ma']:
            return False
        
        # Reject Latin/academic terms that look like names
        latin_terms = ['synaptic', 'neural', 'neuroscience', 'biology', 'chemistry', 'physics']
        if text_lower in latin_terms:
            return False
        
        # Reject if text is mostly punctuation or special characters
        if sum(not c.isalnum() and not c.isspace() for c in text) > len(text) / 2:
            return False
        
        # Minimum lengths for each type (more lenient)
        min_lengths = {
            "EMAIL": 6,  # a@b.co
            "PHONE": 10,  # 1234567890
            "SSN": 9,  # 123456789 (without dashes)
            "CREDIT_CARD": 13,  # Minimum card length
            "BANK_ACCOUNT": 8,  # Minimum account length
            "NAME": 6,  # Full name minimum
            "ADDRESS": 20,  # Full address (strict - need all components)
            "DATE_OF_BIRTH": 8,  # MM/DD/YY or similar
            "MEDICAL_ID": 5,
        }
        
        if text_len < min_lengths.get(pii_type, 3):
            return False
        
        # ULTRA-AGGRESSIVE: Lower thresholds for 90%+ recall
        structured_pii = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "BANK_ACCOUNT"]
        if pii_type in structured_pii:
            if confidence < 0.30:  # AGGRESSIVE: 0.40→0.30 for maximum recall
                return False
        else:
            # For unstructured PII (NAME, ADDRESS), lower threshold
            if confidence < 0.40:  # AGGRESSIVE: 0.50→0.40 for maximum recall
                return False
        
        # Type-specific validation
        if pii_type == "EMAIL":
            # Must contain @ and domain
            if "@" not in text or "." not in text.split("@")[-1]:
                return False
        
        elif pii_type == "NAME":
            
            # CRITICAL: Reject pure numeric (06123456, 123456, 918273645) - student IDs, account numbers, etc.
            if text.replace('-', '').replace(' ', '').replace('.', '').isdigit():
                logger.debug(f"🚫 REJECTED NUMERIC ID as NAME: '{text}'")
                return False
            
            # CRITICAL: Reject if contains ANY digits (real names don't have digits)
            if any(c.isdigit() for c in text):
                logger.debug(f"🚫 REJECTED NAME with digits: '{text}'")
                return False
            
            # ULTRA-STRICT: Reject anything that looks like a domain (with or without dots)
            # Pattern: any text ending with common domain extensions (com, net, org, etc.)
            domain_pattern = r'(.*)(com|net|org|edu|gov|io|ai|co|in|uk|de|fr|jp|cn|se)($|[^a-z])'
            if re.search(domain_pattern, text_lower):
                return False  # Catches: acmecorp.com, cloudtech.com, deeplearning.ai, stanford.edu
            
            # Reject if contains domain/tech keywords (even concatenated)
            domain_keywords = ['www', 'http', 'https', 'github', 'gitlab', 'linkedin', 'facebook',
                             'twitter', 'instagram', 'gmail', 'yahoo', 'outlook', 'hotmail',
                             'email', 'domain', 'website', 'server', 'cloud', 'tech', 'mail']
            if any(keyword in text_lower for keyword in domain_keywords):
                return False  # Catches: linkedin.comindavidparkds, github.comdavidparkml, techmail.com
            
            # ALLOW: Names with single-letter initials + dots (l.rafaelreif, ananthap.chandrakasan, iana.waitz)
            # Pattern: single letter + dot + name OR name + single letter + dot + name
            has_valid_initial = bool(re.search(r'\b[a-z]\.[a-z]+', text_lower))  # l.reif, i.waitz
            
            # Reject dots UNLESS it's a valid initial pattern
            if '.' in text and not has_valid_initial:
                return False  # Reject: jennifer.ma, david.pa, acmecorp.com (but allow: l.rafaelreif)
            
            # Filter common non-names (EXTENSIVE LIST)
            non_names = [
                # Business
                'invoice', 'receipt', 'total', 'amount', 'customer', 'vendor', 
                'business', 'company', 'manager', 'director', 'engineer', 'developer',
                # Tech
                'docker', 'python', 'kubernetes', 'senior', 'junior', 'software',
                # Food/Restaurant
                'italian', 'chinese', 'mexican', 'french', 'japanese', 'thai',
                'salad', 'caesar', 'pasta', 'pizza', 'burger', 'sandwich', 
                'soup', 'dessert', 'appetizer', 'entree', 'menu', 'restaurant',
                # Common words  
                'authorization', 'payment', 'transaction', 'billing', 'direct', 'indirect',
                'service', 'product', 'order', 'delivery'
            ]
            # Check both with and without spaces
            text_normalized = text_lower.replace(' ', '')
            if text_lower in non_names or text_normalized in [n.replace(' ', '') for n in non_names]:
                return False
            
            # Check for food terms anywhere in text
            food_terms = ['salad', 'pasta', 'pizza', 'burger', 'sandwich', 'soup']
            if any(food in text_lower for food in food_terms):
                return False
            
            # ULTRA-STRICT: Blacklist for common generic single words
            generic_words = [
                'model', 'parameter', 'hyperparameter', 'data', 'user', 'admin', 'system',
                'marie', 'anderson', 'garcia', 'smith', 'johnson', 'williams', 'jones',  # Common last names alone
                'schengen', 'visa', 'passport', 'note', 'signature'  # Generic document terms
            ]
            if text_lower in generic_words:
                return False
            
            # Reject technical terms containing these substrings
            tech_substrings = ['param', 'hyper', 'algo', 'config', 'system', 'admin']
            if any(substr in text_lower for substr in tech_substrings):
                return False
            
            # SOFTENED: Reject only obvious degree names (not professional titles)
            # Removed 'doctor', 'engineer', 'scientist' to avoid rejecting real names
            academic_terms = ['bachelor', 'bachelorof', 'masterof', 'doctorateof', 'science', 'arts', 
                            'mathematics', 'physics', 'chemistry', 'biology', 'computerscience',
                            'philosophy', 'phd', 'mba', 'bsc', 'msc', 'degree', 'diploma', 'certificate',
                            'major', 'minor', 'concentration', 'specialization', 'cumlaudegpa', 'cumlaude',
                            'undergraduate', 'graduate', 'postdoctoral', 'fellowship', 'scholarship']
            if any(term in text_lower for term in academic_terms):
                logger.debug(f"🚫 REJECTED ACADEMIC TERM as NAME: '{text}'")
                return False
            
            # Accept names with OR without spaces
            if ' ' not in text:
                # Single-word names: Balance precision and recall
                # REJECT: lowercase concatenated (jennifermarie, andersonrobertjames, robertjamesanderson)
                if text.islower():
                    logger.debug(f"🚫 REJECTED LOWERCASE as NAME: '{text}'")
                    return False  # Reject all lowercase single words (not proper names)
                
                # REJECT: All uppercase concatenated long words (ROBERTJAMESANDERSON)
                if text.isupper() and text_len > 10:
                    logger.debug(f"🚫 REJECTED LONG UPPERCASE as NAME: '{text}'")
                    return False
                
                # Minimum length: 4 chars (catches: Lisa, John, Chen, but rejects: lee, kim)
                if text_len < 4:
                    return False
                    
                # Must be pure letters (no digits, no special chars already checked above)
                if not text.isalpha():
                    return False
            else:
                # Multi-word names: allow single uppercase initial (e.g. "CHARAN L", "John F. Kennedy")
                words = text.split()
                for w in words:
                    is_initial = len(w) == 1 and w.isupper()  # e.g. "L", "F"
                    if len(w) < 2 and not is_initial:
                        return False
                # Reject if all parts are lowercase
                if all(w.islower() for w in words):
                    return False
        
        elif pii_type == "SSN":
            # National ID validation (US SSN, India Aadhaar, etc.) - ULTRA-STRICT
            digits = ''.join(c for c in text if c.isdigit())
            letters = ''.join(c for c in text if c.isalpha())
            x_count = text.lower().count('x')
            
            # Reject short codes (r01, i10, wa12, bw1234567, etc.)
            if text_len <= 4:
                return False
            
            # UK NI format: EXACTLY AB123456C (2 letters + 6 digits + 1 letter = 9 chars)
            if len(letters) >= 2:
                clean_text = text.replace('-', '').replace(' ', '').upper()
                # STRICT: Must be EXACTLY 2 letters + 6 digits + 1 letter
                if re.match(r'^[A-Z]{2}\d{6}[A-Z]$', clean_text) and len(clean_text) == 9:
                    return True
                else:
                    # If it has letters but doesn't match UK NI format, reject it
                    # This catches: "bw1234567" (2 letters + 7 digits), "567891234a", etc.
                    return False
            
            # Numeric formats (US SSN, India Aadhaar) - NO LETTERS ALLOWED
            total = len(digits) + x_count
            
            # Valid lengths: 9 (US SSN), 10, 11, 12 (India Aadhaar)
            # Allow partially redacted (e.g., \"XXX-XX-1234\")
            if total not in [9, 10, 11, 12]:
                return False
            
            # CRITICAL: Reject bank routing numbers (exact range check)
            if total == 9 and x_count == 0 and len(digits) == 9:
                num = int(digits)
                if 10000001 <= num <= 129999999:
                    return False  # Bank routing number range
            
            # Accept valid SSN
            return True
        
        elif pii_type in ["DATE_TIME", "DATE_OF_BIRTH"]:
            # ULTRA-STRICT: Reject concatenated dates (april142024, march152024, july151985patient)
            # Real dates should be reasonable length (< 50 chars)
            if text_len > 50:
                return False
            
            # Real dates MUST have separators: spaces, dashes, slashes, or commas
            if not any(sep in text for sep in [' ', '-', '/', ',']):
                return False  # Reject ALL dates without separators
            
            # EXPLICIT: Reject month+day+year concatenated patterns
            # Patterns like: march152024, april142024, february282024, july151985patient
            if re.match(r'^(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{1,8}', text_lower):
                return False
            
            # CRITICAL: Reject if contains document-specific or medical terms AT ALL
            # Examples: "march102024electronicsignatureverified", "june221978address", "april102024visualacuity"
            # "march152024transcript", "february282024note"
            reject_terms = [
                'patient', 'provider', 'invoice', 'billing', 'payment', 'employer', 'employee',
                'date', 'effective', 'expiration', 'signature', 'verified', 'address', 'acuity',
                'visual', 'electronic', 'digital', 'scan', 'document', 'record',
                'transcript', 'note', 'report', 'letter', 'memo', 'certificate'
            ]
            if any(term in text_lower for term in reject_terms):
                return False
            
            # Reject ALL durations and time-related terms
            duration_keywords = ['hrs', 'hour', 'day', 'week', 'month', 'year', 'min', 'sec',
                                'annually', 'monthly', 'daily', 'weekly', 'yearly', 'quarterly',
                                'summer', 'winter', 'spring', 'fall', 'season', 'period', 'pm', 'am']
            if any(keyword in text_lower for keyword in duration_keywords):
                return False
            
            # Reject pure numbers
            if text.isdigit():
                return False
            
            # Must have proper date format with separators
            has_proper_format = bool(re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text))
            if not has_proper_format:
                # Also accept written dates like "March 15, 2024"
                has_written_format = bool(re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}', text_lower))
                if not has_written_format:
                    return False
        
        elif pii_type == "ADDRESS":
            # Street indicators — Western + Indian formats
            street_indicators = [
                # Western
                'street', 'st', 'avenue', 'ave', 'road', 'rd', 'blvd', 'boulevard',
                'drive', 'dr', 'lane', 'ln', 'way', 'court', 'ct', 'place', 'pl',
                # Indian address components
                'building', 'bldg', 'floor', 'nagar', 'colony', 'layout', 'main',
                'cross', 'block', 'sector', 'phase', 'post', 'plot', 'near', 'no.',
                'complex', 'society', 'apartment', 'apt', 'flat', 'house', 'enclave',
                'residency', 'residencies', 'towers', 'tower', 'park', 'garden',
            ]
            has_street_indicator = any(ind in text_lower for ind in street_indicators)
            has_number = any(c.isdigit() for c in text)
            has_space = ' ' in text

            # Concatenated address without spaces — reject
            if has_street_indicator and has_number and not has_space:
                return False

            # Require at least a location indicator + number/city
            if not (has_street_indicator and has_space):
                return False

            # Must have 3+ words (short single-word locations are not full addresses)
            if text.count(' ') < 2:
                return False
        
        elif pii_type == "PHONE":
            # Must have 10+ digits
            digits = ''.join(c for c in text if c.isdigit())
            if len(digits) < 10 or len(digits) > 15:
                return False
            
            # Reject pure letter strings (no digits)
            if not any(c.isdigit() for c in text):
                return False
            
            # ALLOW: Toll-free numbers (1800, 1866, 1877, 1888, 1855)
            # Pattern: 1-800-XXX-XXXX or 18005551234
            is_tollfree = digits.startswith(('1800', '1866', '1877', '1888', '1855', '1844'))
            
            # CRITICAL: Reject year-like patterns UNLESS it's toll-free
            if not is_tollfree:
                if re.match(r'^20\d{2}', digits):  # Starts with 2000-2099
                    return False
                if re.match(r'^19\d{2}', digits):  # Starts with 1900-1999
                    return False
            
            # CRITICAL: Reject text descriptions (cardendingin1234, 1800555bank2265)
            text_desc_words = ['card', 'ending', 'last', 'first', 'bank', 'account', 'number', 'mrn', 'txn', 'id']
            if any(word in text_lower for word in text_desc_words):
                return False
            
            # CRITICAL: Reject if it's a simple sequence (1234567890, 9876543210)
            if digits in ['1234567890', '0123456789', '9876543210', '0987654321']:
                return False
            
            # Require proper phone formatting (parentheses, dashes, spaces, or + prefix)
            has_formatting = any(sym in text for sym in ['(', ')', '-', '+', ' '])
            # OR it's just 10 consecutive digits with no alpha
            is_pure_10_digits = (len(digits) == 10 and text.replace(digits, '').strip() == '')
            if not (has_formatting or is_pure_10_digits):
                return False
        
        elif pii_type in ["MEDICAL_ID", "MEDICAL_LICENSE"]:
            # CRITICAL: Reject text descriptions (mrn445566, californiamedicallicensea123456, deanumberfr1234567)
            desc_words = ['mrn', 'medical', 'license', 'california', 'number', 'dea', 'txn', 'transaction', 'patient', 'provider', 'student']
            if any(word in text_lower for word in desc_words):
                return False
            
            # Reject student IDs (typically 8 digits, all numeric, starting with 0)
            # Pattern: 06123456, 01234567, etc.
            if text.isdigit() and len(text) == 8 and text.startswith('0'):
                return False  # Likely student ID, not medical ID
            
            # Reject if text is longer than 15 chars (too verbose)
            if len(text) > 15:
                return False
        
        elif pii_type in ["BANK_ACCOUNT", "US_BANK_NUMBER", "CREDIT_CARD"]:
            # Extract digits
            digits = ''.join(c for c in text if c.isdigit())
            
            # CRITICAL: Reject text descriptions (cardendingin1234, creditcard1234)
            desc_words = ['card', 'ending', 'last', 'first', 'account', 'bank', 'credit', 'debit', 'visa', 'mastercard', 'amex']
            if any(word in text_lower for word in desc_words):
                return False
            
            # CRITICAL: Reject alphanumeric codes (95d12345678901234 has letters mixed with digits)
            # Bank/credit card numbers should be mostly digits with only separators (-, space)
            clean_text = text.replace('-', '').replace(' ', '')
            if not clean_text.isdigit():
                return False  # Has letters or special chars - not a valid number
            
            # Must have at least 8 digits
            if len(digits) < 8:
                return False
            
            # Reject if contains spaces AND letters (descriptions)
            if ' ' in text and any(c.isalpha() for c in text):
                return False
        
        elif pii_type in ["SSN", "US_SSN"]:
            # Get pure digits
            digits = ''.join(c for c in text if c.isdigit())
            
            # CRITICAL: Reject routing numbers (111000025 - 9 digits starting with 0-12)
            if len(digits) == 9:
                num = int(digits)
                # Routing numbers: 000000001-129999999
                if num <= 129999999:
                    return False
            
            # Accept valid SSN format
            return len(digits) == 9 or len(digits) == 12  # US SSN or Aadhaar
        
        elif pii_type == "TAX_ID":
            # Tax ID validation (India PAN, GSTIN, etc.) - ULTRA-STRICT
            text_upper = text.upper().strip().replace('-', '').replace(' ', '')
            
            # Reject pure 9-digit numbers (likely SSN/routing, not tax ID)
            if text_upper.isdigit() and len(text_upper) == 9:
                num = int(text_upper)
                # Reject anything that looks like routing number or random number
                if num < 100000000 or num > 999999999:
                    return False  # Suspiciously low/high
                # Additional: reject if it looks like a routing number
                if 10000001 <= num <= 129999999:
                    return False
                # Additional: Most valid tax IDs aren't pure digits
                return False  # Be strict: pure 9-digit numbers are rarely tax IDs
            
            # India PAN: ABCDE1234F (5 letters + 4 digits + 1 letter)
            if len(text_upper) == 10 and re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', text_upper):
                return confidence >= 0.65
            
            # India GSTIN: 22AAAAA0000A1Z5 (15 alphanumeric)
            if len(text_upper) == 15 and re.match(r'^[0-9]{2}[A-Z0-9]{10}[A-Z][0-9][A-Z]$', text_upper):
                return confidence >= 0.65
            
            # Other tax IDs: 8-15 alphanumeric (MUST have letters)
            if 8 <= len(text_upper) <= 15 and text_upper.isalnum():
                # MUST contain at least one letter (tax IDs aren't pure numbers)
                if not any(c.isalpha() for c in text_upper):
                    return False  # Reject pure digit tax IDs
                return confidence >= 0.60
            
            return False
        
        # Legacy support for old PII type names
        elif pii_type in ["PAN", "GSTIN"]:
            # Redirect to TAX_ID
            return self._validate_llm_pii("TAX_ID", text, confidence)

        elif pii_type == "GENDER":
            # Accept only clear gender values
            allowed = {"male", "female", "m", "f", "other", "non-binary", "transgender"}
            return text_lower in allowed or any(g in text_lower for g in ["male", "female"])

        return True
    
    def _detect_pii_with_llm(self, text: str) -> tuple[List[PIIDetection], Dict[str, Any]]:
        """
        Detect PII using LLM
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (list of detected PII, llm response dict)
        """
        # Initialize to ensure it's always defined
        llm_response = {}
        response = ""
        
        try:
            # Truncate text if too long (keep first 3000 chars)
            text_sample = text[:3000] if len(text) > 3000 else text
            
            # Create prompt
            prompt = self.PII_DETECTION_PROMPT.format(text=text_sample)
            
            # System prompt for thorough but precise PII detection
            system_prompt = "You are a precise PII detection system. Detect ALL genuine personal information that could identify real individuals. Be thorough but precise - avoid detecting domains (.com, .org), URLs, company names, or non-PII text. Return ONLY valid JSON with no explanations."
            
            # 70b on key-2 — precise PII detection, key-1 reserved for classifier/validator
            llm_response = llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=600,
                temperature=0.0,
                groq_model="llama-3.1-8b-instant",  # 8b-instant: fast entity detection, frees key-2 budget
                groq_key=2  # Secondary key
            )
            response = llm_response.get("content", "")
            
            # Parse JSON response
            response_text = response.strip()
            
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Convert to PIIDetection objects with validation
            pii_detections = []
            for item in result.get("pii_detections", []):
                try:
                    # Map string type to PIIType enum
                    pii_type_str = item.get("pii_type", "NAME").upper()
                    pii_type = PIIType[pii_type_str] if hasattr(PIIType, pii_type_str) else PIIType.NAME
                    
                    original_text = item.get("original_text", "")
                    confidence = item.get("confidence", 0.9)
                    
                    # Validate LLM detection (apply same strict rules)
                    is_valid = self._validate_llm_pii(pii_type_str, original_text, confidence)
                    if not is_valid:
                        logger.debug(f"FILTERED LLM: {pii_type_str}='{original_text}' (confidence={confidence:.2f})")
                        continue
                    else:
                        logger.debug(f"ACCEPTED LLM: {pii_type_str}='{original_text}' (confidence={confidence:.2f})")
                    
                    detection = PIIDetection(
                        field_name=item.get("field_name", "unknown"),
                        pii_type=pii_type,
                        original_text=original_text,
                        redacted_text=item.get("redacted_text", "[REDACTED]"),
                        detection_source="llm",
                        confidence=confidence
                    )
                    pii_detections.append(detection)
                except Exception as e:
                    logger.warning(f"Failed to parse PII detection: {e}")
                    continue
            
            return pii_detections, llm_response
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM PII response: {e}")
            if response:
                logger.debug(f"Raw response: {response}")
            return [], llm_response if llm_response else {}
        except Exception as e:
            logger.error(f"LLM PII detection failed: {e}")
            return [], llm_response if llm_response else {}
    
    def _redact_text(self, text: str, pii_detections: List[PIIDetection]) -> str:
        """
        Apply redactions to text
        
        Args:
            text: Original text
            pii_detections: List of PII to redact
        
        Returns:
            Redacted text
        """
        redacted_text = text
        
        # Sort by position (longest first to handle overlaps)
        sorted_detections = sorted(
            pii_detections,
            key=lambda x: len(x.original_text),
            reverse=True
        )
        
        # Apply redactions
        for detection in sorted_detections:
            if detection.original_text in redacted_text:
                redacted_text = redacted_text.replace(
                    detection.original_text,
                    detection.redacted_text
                )
        
        return redacted_text
        return redacted_text
    
    def _compute_metrics(
        self,
        detected_pii: List[PIIDetection],
        extracted_fields: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Compute PII precision and recall
        
        Args:
            detected_pii: Detected PII list
            extracted_fields: Extracted fields (ground truth)
        
        Returns:
            Tuple of (precision, recall)
        """
        # Ground truth PII fields from extracted data
        ground_truth_pii = set()
        
        for field_name, value in extracted_fields.items():
            if value and isinstance(value, str):
                field_lower = field_name.lower()
                
                # Identify PII fields
                if any(term in field_lower for term in [
                    "email", "phone", "ssn", "social_security",
                    "credit_card", "card_number", "name", "address",
                    "patient_id", "medical_id", "dob", "date_of_birth"
                ]):
                    ground_truth_pii.add(value)
        
        # Detected PII values
        detected_values = set(pii.original_text for pii in detected_pii)
        
        # Compute metrics
        if len(detected_values) == 0:
            precision = 1.0 if len(ground_truth_pii) == 0 else 0.0
        else:
            true_positives = len(detected_values & ground_truth_pii)
            precision = true_positives / len(detected_values)
        
        if len(ground_truth_pii) == 0:
            recall = 1.0
        else:
            true_positives = len(detected_values & ground_truth_pii)
            recall = true_positives / len(ground_truth_pii)
        
        return precision, recall
    
    def redact(self, state: DocumentState) -> DocumentState:
        """
        Detect and redact PII from document using Presidio + LLM hybrid
        
        Args:
            state: Current document state
        
        Returns:
            Updated state with redaction result
        """
        logger.info(f"{self.name}: Starting Presidio + LLM PII detection and redaction")
        start_time = time.time()
        
        try:
            raw_text = state["raw_text"]
            extracted_fields = state.get("extracted_fields", {})
            
            # 1. Detect PII using Presidio (primary, fast)
            presidio_pii = self._detect_pii_with_presidio(raw_text)
            logger.info(f"Presidio detected {len(presidio_pii)} PII instances")

            # 1b. Rule-based gender detection (fast, no LLM)
            gender_pii = self._detect_gender_patterns(raw_text)
            if gender_pii:
                logger.info(f"Gender detection found {len(gender_pii)} instance(s)")
                presidio_pii.extend(gender_pii)

            # 2. Enhance with LLM for context-aware detection (HYBRID MODE RE-ENABLED)
            llm_pii, llm_response = self._detect_pii_with_llm(raw_text)
            logger.info(f"LLM detected {len(llm_pii)} additional PII instances")
            
            # 3. Merge results with smart deduplication
            detected_pii = presidio_pii.copy()
            existing_texts = {pii.original_text.lower() for pii in presidio_pii}
            
            for pii in llm_pii:
                pii_text_lower = pii.original_text.lower()
                
                # Skip if exact match already exists
                if pii_text_lower in existing_texts:
                    continue
                
                # Skip if this text is contained in a larger existing detection
                # (e.g., don't add "John" if "John Smith" already detected)
                is_substring = False
                for existing in existing_texts:
                    if pii_text_lower in existing or existing in pii_text_lower:
                        # Keep the longer one
                        if len(pii_text_lower) > len(existing):
                            # Remove shorter existing detection
                            detected_pii = [p for p in detected_pii if p.original_text.lower() != existing]
                            existing_texts.discard(existing)
                            break
                        else:
                            is_substring = True
                            break
                
                if not is_substring:
                    detected_pii.append(pii)
                    existing_texts.add(pii_text_lower)
            
            logger.info(f"Total unique PII instances: {len(detected_pii)} (Presidio: {len(presidio_pii)}, LLM: {len(llm_pii)})")
            
            # 2. Redact text
            redacted_text = self._redact_text(raw_text, detected_pii)
            
            # 3. Compute metrics
            precision, recall = self._compute_metrics(detected_pii, extracted_fields)
            
            latency = time.time() - start_time
            
            # Create result
            redaction_result = RedactionResult(
                redacted_text=redacted_text,
                pii_detections=detected_pii,
                detected_pii=detected_pii,  # For backward compatibility
                pii_count=len(detected_pii),
                precision=precision,
                recall=recall,
                timestamp=datetime.utcnow()
            )
            
            # Update state
            state["redacted_text"] = redacted_text
            state["redaction_result"] = redaction_result
            state["agent_timings"][self.name] = latency
            
            # Responsible AI logging
            state["trace_log"].append(
                ResponsibleAILog(
                    agent_name=self.name,
                    input_data=raw_text[:500],
                    output_data=f"PII Count: {len(detected_pii)}, Precision: {precision:.2f}, Recall: {recall:.2f}",
                    timestamp=datetime.utcnow(),
                    latency_ms=latency * 1000,
                    llm_model_used=llm_response.get("model", "unknown"),
                    tokens_used=llm_response.get("tokens", {}).get("input", 0) + llm_response.get("tokens", {}).get("output", 0),
                    error_occurred=False,
                    llm_provider=llm_response.get("provider", "unknown"),
                    system_prompt="",
                    user_prompt=llm_response.get("user_prompt", ""),
                    context_data={
                        "text_length": len(raw_text),
                        "pii_count": len(detected_pii),
                        "precision": precision,
                        "recall": recall
                    },
                    raw_output=llm_response.get("content", ""),
                    tokens_input=llm_response.get("tokens", {}).get("input", 0),
                    tokens_output=llm_response.get("tokens", {}).get("output", 0),
                    retry_attempt=0
                )
            )
            
            logger.info(
                f"{self.name}: Redaction complete",
                pii_count=len(detected_pii),
                precision=precision,
                recall=recall,
                latency_ms=latency * 1000
            )
            
            return state
        
        except Exception as e:
            logger.error(f"{self.name}: Redaction failed", error=str(e))
            
            # Update state with error
            state["errors"].append(f"{self.name}: {str(e)}")
            state["redacted_text"] = state["raw_text"]  # No redaction
            state["redaction_result"] = RedactionResult(
                redacted_text=state["raw_text"],
                pii_detections=[],
                detected_pii=[],
                pii_count=0,
                precision=0.0,
                recall=0.0
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
                    system_prompt=self.PII_DETECTION_PROMPT[:500],
                    user_prompt="",
                    context_data={
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
redactor_agent = RedactorAgent()
