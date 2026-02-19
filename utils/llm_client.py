"""
LLM Client with Bedrock Claude Haiku and Local Llama3.1 Fallback
"""
import json
import time
from typing import Optional, Dict, Any, List
from enum import Enum

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import settings
from utils.logger import logger
from utils.retry_decorator import with_retry


class LLMProvider(str, Enum):
    """LLM Provider types"""
    GROQ = "groq"
    BEDROCK_CLAUDE = "bedrock_claude"
    BEDROCK_NOVA = "bedrock_nova"   # Amazon Nova Lite: 4s, same quality as Claude Haiku
    HUGGINGFACE = "huggingface"
    LOCAL_LLAMA = "local_llama"


class LLMClient:
    """
    Unified LLM client with automatic fallback
    
    Primary: Groq (Llama 3.1 8B Instant - FASTEST, 300+ tokens/sec)
    Secondary: AWS Bedrock Claude 3 Haiku
    Tertiary: HuggingFace Inference API
    Fallback: Local Llama3.1 (CPU-based, slowest)
    """
    
    def __init__(self):
        self.groq_client = None
        self.groq_client_b = None   # Backup Groq client (GROQ_API_KEY_B)
        self.bedrock_client = None
        self.huggingface_client = None
        self.llama_client = None
        self.llama_tokenizer = None
        self.llama_device = None
        self.llama_model_name = None
        self._initialize_groq()
        self._initialize_groq_b()
        self._initialize_bedrock()
        self._initialize_huggingface()
        self._initialize_llama()
    
    def _initialize_groq(self) -> None:
        """Initialize Groq API client (primary key)"""
        if not settings.GROQ_API_KEY:
            logger.info("GROQ_API_KEY not configured, skipping Groq initialization")
            self.groq_client = None
            return
            
        try:
            from groq import Groq
            
            self.groq_client = Groq(
                api_key=settings.GROQ_API_KEY,
                timeout=settings.GROQ_TIMEOUT
            )
            
            logger.info(f"Groq API client (primary) initialized (model: {settings.GROQ_MODEL})")
        except ImportError:
            logger.warning("groq package not installed. Install with: pip install groq")
            self.groq_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")
            self.groq_client = None

    def _initialize_groq_b(self) -> None:
        """Initialize backup Groq API client (GROQ_API_KEY_B).

        Used as a second Groq attempt when the primary key hits a rate-limit
        or request error, keeping Groq-class latency before falling back to
        the much slower Bedrock (90 s timeout).
        """
        if not getattr(settings, 'GROQ_API_KEY_B', None):
            logger.info("GROQ_API_KEY_B not configured — backup Groq key unavailable")
            self.groq_client_b = None
            return

        try:
            from groq import Groq

            self.groq_client_b = Groq(
                api_key=settings.GROQ_API_KEY_B,
                timeout=settings.GROQ_TIMEOUT
            )
            logger.info(f"Groq API client (backup) initialized (model: {settings.GROQ_MODEL})")
        except Exception as e:
            logger.warning(f"Failed to initialize backup Groq client: {e}")
            self.groq_client_b = None
    
    def _initialize_bedrock(self) -> None:
        """Initialize AWS Bedrock client"""
        # Skip initialization if credentials not configured
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            logger.info("AWS credentials not configured, skipping Bedrock initialization")
            self.bedrock_client = None
            return
            
        try:
            boto_config = Config(
                region_name=settings.AWS_REGION,
                connect_timeout=settings.BEDROCK_TIMEOUT,
                read_timeout=settings.BEDROCK_TIMEOUT,
                retries={'max_attempts': settings.MAX_RETRIES}
            )
            
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                config=boto_config
            )
            
            logger.info("Bedrock Claude client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Bedrock client: {e}")
            self.bedrock_client = None
    
    def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace Inference API client"""
        try:
            if settings.HF_API_KEY:
                from huggingface_hub import InferenceClient
                
                self.huggingface_client = InferenceClient(
                    token=settings.HF_API_KEY,
                    timeout=settings.HF_TIMEOUT
                )
                logger.info("HuggingFace Inference API client initialized successfully")
            else:
                logger.warning("HF_API_KEY not configured, HuggingFace fallback unavailable")
                self.huggingface_client = None
        except ImportError:
            logger.warning("huggingface_hub not installed, HuggingFace fallback unavailable")
            self.huggingface_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace client: {type(e).__name__}: {e}", exc_info=True)
            self.huggingface_client = None
    
    def _initialize_llama(self) -> None:
        """Initialize local LLM client using GGUF (llama-cpp-python) or Transformers"""
        # Check if GGUF model path is configured
        if hasattr(settings, 'LLAMA_MODEL_PATH') and settings.LLAMA_MODEL_PATH:
            self._initialize_llama_gguf()
        else:
            self._initialize_llama_transformers()
    
    def _initialize_llama_gguf(self) -> None:
        """Initialize local LLM using llama-cpp-python with GGUF file"""
        try:
            from llama_cpp import Llama
            import os
            
            model_path = settings.LLAMA_MODEL_PATH
            if not os.path.exists(model_path):
                logger.warning(f"GGUF model file not found: {model_path}")
                self.llama_client = None
                return
            
            n_gpu_layers = getattr(settings, 'LLAMA_N_GPU_LAYERS', 0)
            n_ctx = settings.LLAMA_CONTEXT_LENGTH
            n_batch = getattr(settings, 'LLAMA_BATCH_SIZE', 256)  # Use configured batch size
            
            logger.info(f"Initializing Llama with GGUF file: {model_path} (GPU layers: {n_gpu_layers}, ctx: {n_ctx}, batch: {n_batch})...")
            
            self.llama_client = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                verbose=False
            )
            
            self.llama_model_name = f"local-llama-gguf"
            self.llama_tokenizer = None  # GGUF uses internal tokenizer
            logger.info(f"Local LLM (GGUF) initialized successfully with {n_gpu_layers} GPU layers, context: {n_ctx}")
            
        except ImportError:
            logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self.llama_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize GGUF LLM: {type(e).__name__}: {e}", exc_info=True)
            self.llama_client = None
    
    def _initialize_llama_transformers(self) -> None:
        """Initialize local LLM client using Transformers with GPU support"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_info = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
            
            # Use model from config
            model_name = settings.LOCAL_MODEL_NAME
            
            logger.info(f"Initializing local LLM '{model_name}' on {gpu_info}...")
            
            # Load tokenizer
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            
            # Configure for efficient inference
            if torch.cuda.is_available() and settings.USE_GPU_INFERENCE:
                try:
                    # Try 4-bit quantization for GPU (saves memory)
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.llama_client = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16
                    )
                    logger.info("Loaded model with 4-bit quantization")
                except Exception as e:
                    logger.warning(f"4-bit quantization failed, loading in FP16: {e}")
                    # Fallback to FP16 without quantization
                    self.llama_client = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16
                    )
            else:
                # CPU inference with FP32
                logger.info("Loading model for CPU inference...")
                self.llama_client = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.llama_device = device
            self.llama_model_name = model_name
            logger.info(f"Local LLM initialized successfully on {gpu_info}")
            
        except ImportError as e:
            logger.warning(f"Required packages not installed for local LLM: {e}")
            self.llama_client = None
            self.llama_tokenizer = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {type(e).__name__}: {e}", exc_info=True)
            self.llama_client = None
            self.llama_tokenizer = None
    
    @with_retry()
    def _invoke_groq(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        groq_model: Optional[str] = None,
        client=None   # Pass specific Groq client instance (key 1 or key 2)
    ) -> Dict[str, Any]:
        """
        Invoke Groq API with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            client: Override which Groq client (key) to use
        
        Returns:
            Response dict
        """
        groq_client = client or self.groq_client
        if not groq_client:
            raise RuntimeError("Groq client not initialized")
        
        max_tokens = max_tokens or settings.GROQ_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        model = groq_model or settings.GROQ_MODEL
        
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            content = response.choices[0].message.content
            latency = time.time() - start_time
            
            # Extract token counts (match structure with other providers)
            tokens_in = 0
            tokens_out = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_in = response.usage.prompt_tokens or 0
                tokens_out = response.usage.completion_tokens or 0
            
            return {
                "content": content,
                "provider": "groq",
                "model": model,
                "latency": latency,
                "tokens": {
                    "input": tokens_in,
                    "output": tokens_out
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    @with_retry()
    def _invoke_bedrock_claude(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke Bedrock Claude with retry logic
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            Exception if Bedrock invocation fails
        """
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")
        
        max_tokens = max_tokens or settings.BEDROCK_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.BEDROCK_TEMPERATURE
        
        # Construct Claude 3 message format
        messages = [{"role": "user", "content": prompt}]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        start_time = time.time()
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=settings.BEDROCK_MODEL_ID,  # Configurable via .env / config.py
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            latency = time.time() - start_time
            
            content = response_body['content'][0]['text']
            
            return {
                "content": content,
                "provider": LLMProvider.BEDROCK_CLAUDE,
                "model": settings.BEDROCK_MODEL_ID,
                "latency": latency,
                "tokens": {
                    "input": response_body.get('usage', {}).get('input_tokens', 0),
                    "output": response_body.get('usage', {}).get('output_tokens', 0)
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Bedrock API error: {error_code}", error=str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected Bedrock error: {e}")
            raise

    NOVA_LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"

    @with_retry()
    def _invoke_nova(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke Amazon Nova Lite via Bedrock — 2x faster than Claude Haiku, same quality.
        Uses Converse-style message format.
        """
        if not self.bedrock_client:
            raise RuntimeError("Bedrock client not initialized")

        max_tokens = max_tokens or settings.BEDROCK_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.BEDROCK_TEMPERATURE

        request_body: Dict[str, Any] = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}
        }
        if system_prompt:
            request_body["system"] = [{"text": system_prompt}]

        start_time = time.time()
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.NOVA_LITE_MODEL_ID,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response["body"].read())
            latency = time.time() - start_time

            content_list = response_body.get("output", {}).get("message", {}).get("content", [])
            content = content_list[0].get("text", "") if content_list else ""
            usage = response_body.get("usage", {})

            return {
                "content": content,
                "provider": LLMProvider.BEDROCK_NOVA,
                "model": self.NOVA_LITE_MODEL_ID,
                "latency": latency,
                "tokens": {
                    "input": usage.get("inputTokens", 0),
                    "output": usage.get("outputTokens", 0)
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        except Exception as e:
            logger.error(f"Nova Lite error: {e}")
            raise

    def _invoke_llama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke local LLM (routes to GGUF or Transformers based on model type)
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if LLM client not initialized
        """
        if not self.llama_client:
            raise RuntimeError("Local LLM client not initialized")
        
        # Check if using GGUF (llama-cpp-python) or Transformers
        if hasattr(self.llama_client, '__call__'):  # GGUF Llama object
            return self._invoke_llama_gguf(prompt, system_prompt, max_tokens, temperature)
        else:  # Transformers model
            return self._invoke_llama_transformers(prompt, system_prompt, max_tokens, temperature)
    
    def _invoke_llama_gguf(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """Invoke GGUF model using llama-cpp-python - SPEED OPTIMIZED"""
        max_tokens = max_tokens or getattr(settings, 'LLAMA_MAX_TOKENS', 64)
        temperature = temperature if temperature is not None else settings.LLAMA_TEMPERATURE
        
        # Balanced truncation for speed while preserving content
        prompt_max = 800  # Enough for document excerpt (was 200, too aggressive)
        if len(prompt) > prompt_max:
            # Keep beginning and end for context
            half = prompt_max // 2
            prompt = prompt[:half] + "\n...[truncated]...\n" + prompt[-half:]
        
        # Build Llama 3.1 prompt format
        if system_prompt:
            system_max = 100  # Increased from 50 for clarity
            if len(system_prompt) > system_max:
                system_prompt = system_prompt[:system_max]
            full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            full_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        start_time = time.time()
        
        try:
            response = self.llama_client(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False
            )
            
            latency = time.time() - start_time
            content = response['choices'][0]['text'].strip()
            
            return {
                "content": content,
                "provider": LLMProvider.LOCAL_LLAMA,
                "model": self.llama_model_name,
                "latency": latency,
                "tokens": {
                    "input": response['usage']['prompt_tokens'],
                    "output": response['usage']['completion_tokens']
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        except Exception as e:
            logger.error(f"GGUF LLM invocation error: {e}", exc_info=True)
            raise
    
    def _invoke_llama_transformers(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke local LLM using Transformers
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if LLM client not initialized
        """
        if not self.llama_client or not self.llama_tokenizer:
            raise RuntimeError("Local LLM client not initialized")
        
        import torch
        
        # Aggressively optimize for speed
        max_tokens = max_tokens or getattr(settings, 'LLAMA_MAX_TOKENS', 128)
        temperature = temperature if temperature is not None else settings.LLAMA_TEMPERATURE
        
        # Aggressively truncate for much faster processing
        prompt_max = 400  # Reduced from 800
        if len(prompt) > prompt_max:
            prompt = prompt[:prompt_max] + "..."
        
        # Minimal prompt format for speed
        if system_prompt:
            # Very short system prompt
            system_max = 100  # Reduced from 200
            if len(system_prompt) > system_max:
                system_prompt = system_prompt[:system_max]
            input_text = f"{system_prompt}\n\n{prompt}"
        else:
            input_text = prompt
        
        start_time = time.time()
        
        try:
            # Tokenize
            inputs = self.llama_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.LLAMA_CONTEXT_LENGTH
            )
            
            # Move to device
            device = next(self.llama_client.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with maximum speed optimizations
            with torch.no_grad():
                outputs = self.llama_client.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding
                    num_beams=1,  # No beam search
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id,
                    use_cache=True,  # KV cache
                    early_stopping=True,  # Stop as soon as possible
                    repetition_penalty=1.0  # No penalty computation
                )
            
            # Decode only the generated tokens (skip input)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            content = self.llama_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            latency = time.time() - start_time
            
            return {
                "content": content.strip(),
                "provider": LLMProvider.LOCAL_LLAMA,
                "model": getattr(self, 'llama_model_name', 'local-llm'),
                "latency": latency,
                "tokens": {
                    "input": inputs['input_ids'].shape[1],
                    "output": len(generated_tokens)
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        
        except Exception as e:
            logger.error(f"Local LLM invocation error: {e}", exc_info=True)
            raise
    
    def _invoke_huggingface(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Invoke HuggingFace Inference API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if HuggingFace client not initialized
        """
        if not self.huggingface_client:
            raise RuntimeError("HuggingFace client not initialized")
        
        max_tokens = max_tokens or settings.HF_MAX_TOKENS
        temperature = temperature if temperature is not None else settings.HF_TEMPERATURE
        
        # Format messages for HuggingFace chat completion
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            # Use chat completion API for instruction-tuned models
            response = self.huggingface_client.chat_completion(
                messages=messages,
                model=settings.HF_MODEL,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency = time.time() - start_time
            
            content = response.choices[0].message.content
            
            # Extract token counts if available
            tokens_in = 0
            tokens_out = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_in = response.usage.prompt_tokens or 0
                tokens_out = response.usage.completion_tokens or 0
            
            return {
                "content": content,
                "provider": LLMProvider.HUGGINGFACE,
                "model": settings.HF_MODEL,
                "latency": latency,
                "tokens": {
                    "input": tokens_in,
                    "output": tokens_out
                },
                "system_prompt": system_prompt,
                "user_prompt": prompt
            }
        
        except Exception as e:
            logger.error(f"HuggingFace invocation error: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        force_provider: Optional[LLMProvider] = None,
        groq_model: Optional[str] = None,
        groq_key: int = 1  # 1 = primary key (classifier/validator), 2 = secondary key (extractor/repair/redactor)
    ) -> Dict[str, Any]:
        """
        Generate completion with automatic fallback
        
        Fallback chain: Bedrock → HuggingFace → Llama (local)
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            force_provider: Force specific provider (skip fallback)
        
        Returns:
            Response dict with content and metadata
        
        Raises:
            RuntimeError if all providers fail
        """
        # Try forced provider if specified
        if force_provider == LLMProvider.GROQ:
            logger.info("Using forced provider: Groq")
            return self._invoke_groq(prompt, system_prompt, max_tokens, temperature)
        elif force_provider == LLMProvider.BEDROCK_NOVA:
            logger.info("Using forced provider: Nova Lite")
            return self._invoke_nova(prompt, system_prompt, max_tokens, temperature)
        elif force_provider == LLMProvider.BEDROCK_CLAUDE:
            logger.info("Using forced provider: Bedrock Claude")
            return self._invoke_bedrock_claude(prompt, system_prompt, max_tokens, temperature)
        elif force_provider == LLMProvider.HUGGINGFACE:
            logger.info("Using forced provider: HuggingFace")
            return self._invoke_huggingface(prompt, system_prompt, max_tokens, temperature)
        elif force_provider == LLMProvider.LOCAL_LLAMA:
            logger.info("Using forced provider: Local Llama")
            return self._invoke_llama(prompt, system_prompt, max_tokens, temperature)
        
        errors = []

        # ROUTING STRATEGY:
        # - groq_model specified → Groq first (fast agents: classifier, validator, redactor ~0.5s)
        # - groq_model not specified → Nova Lite first (quality agents: extractor, self-repair ~4s)
        #   Nova Lite: same accuracy as Claude Haiku but 2x faster, no rate limits
        if groq_model and (self.groq_client or self.groq_client_b):
            # Select key: use secondary (groq_client_b) when groq_key==2 and available, else primary
            selected_client = (
                self.groq_client_b if groq_key == 2 and self.groq_client_b
                else self.groq_client
            )
            key_label = "key-2" if (groq_key == 2 and self.groq_client_b) else "key-1"
            try:
                logger.info(f"Using Groq ({key_label}) model={groq_model}")
                return self._invoke_groq(prompt, system_prompt, max_tokens, temperature,
                                         groq_model=groq_model, client=selected_client)
            except Exception as groq_error:
                logger.warning(f"Groq ({key_label}) failed: {groq_error}")
                errors.append(f"Groq-{key_label}: {groq_error}")
            # Fall through to Nova below

        # Nova Lite (primary for quality agents, fallback for fast agents when Groq fails)
        if self.bedrock_client:
            try:
                logger.info("Using Nova Lite (4s, no rate limits)")
                return self._invoke_nova(prompt, system_prompt, max_tokens, temperature)
            except Exception as nova_error:
                logger.warning(f"Nova Lite failed: {nova_error}, falling back to Claude")
                errors.append(f"Nova: {nova_error}")
                # Fallback to Claude Haiku
                try:
                    logger.info("Falling back to Bedrock Claude Haiku")
                    return self._invoke_bedrock_claude(prompt, system_prompt, max_tokens, temperature)
                except Exception as bedrock_error:
                    logger.warning(f"Bedrock Claude failed: {bedrock_error}")
                    errors.append(f"Bedrock: {bedrock_error}")
        else:
            logger.debug("Bedrock client not available, skipping")
        
        # Fallback to HuggingFace
        if self.huggingface_client:
            try:
                logger.info("Falling back to HuggingFace Inference API")
                return self._invoke_huggingface(prompt, system_prompt, max_tokens, temperature)
            
            except Exception as hf_error:
                logger.warning(f"HuggingFace fallback failed: {hf_error}", exc_info=True)
                errors.append(f"HuggingFace: {hf_error}")
        else:
            logger.info("HuggingFace client not available, skipping")
        
        # Fallback to local Llama
        if self.llama_client:
            try:
                logger.info("Falling back to local Llama3.1")
                return self._invoke_llama(prompt, system_prompt, max_tokens, temperature)
            
            except Exception as llama_error:
                logger.error(f"Llama fallback also failed: {llama_error}")
                errors.append(f"Llama: {llama_error}")
                raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")
        else:
            logger.error("No fallback available")
            raise RuntimeError(f"All configured LLM providers failed: {'; '.join(errors)}")
    
    def is_available(self, provider: Optional[LLMProvider] = None) -> bool:
        """
        Check if LLM provider is available
        
        Args:
            provider: Specific provider to check, or None for any
        
        Returns:
            True if provider is available
        """
        if provider == LLMProvider.BEDROCK_CLAUDE:
            return self.bedrock_client is not None
        elif provider == LLMProvider.HUGGINGFACE:
            return self.huggingface_client is not None
        elif provider == LLMProvider.LOCAL_LLAMA:
            return self.llama_client is not None
        else:
            return (
                self.bedrock_client is not None 
                or self.huggingface_client is not None 
                or self.llama_client is not None
            )
    
    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Simplified invoke method that returns just the content string
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text content
        """
        response = self.generate(prompt, system_prompt, max_tokens, temperature)
        return response.get("content", "")
    
    def get_active_model_name(self) -> str:
        """
        Get the name of the currently active model
        
        Returns:
            Model name string
        """
        if self.bedrock_client:
            return "claude-3-haiku (Bedrock)"
        elif self.huggingface_client:
            return f"{settings.HF_MODEL.split('/')[-1]} (HuggingFace)"
        elif self.llama_client:
            return "llama-3.1-8b (Local)"
        else:
            return "unknown"


# Global LLM client instance
llm_client = LLMClient()
