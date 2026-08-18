"""
Criteria Generator - Direct API client supporting multiple LLM providers.
Supports: OpenAI, Azure OpenAI, Claude, Gemini, Grok, DeepSeek, vLLM.
All providers use structured output when available.
"""
import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .agents import run_pipeline, parse_json
from .prompts import get_prompt
from .schemas import AGENT_SCHEMAS


def normalize_questions(
    questions: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]
) -> tuple[bool, List[Dict[str, Any]]]:
    """Normalize generate() input without mutating caller-owned objects."""
    if isinstance(questions, str):
        return True, [{"id": "0", "question": questions}]

    if isinstance(questions, dict):
        item = questions.copy()
        item.setdefault("id", "0")
        return True, [item]

    items = []
    for i, q in enumerate(questions):
        if isinstance(q, str):
            items.append({"id": str(i), "question": q})
        elif isinstance(q, dict):
            item = q.copy()
            item.setdefault("id", str(i))
            items.append(item)
        else:
            raise TypeError("questions must be a string, dict, or list of strings/dicts")
    return False, items


class CriteriaGenerator:
    """
    Generate evaluation criteria for questions using LLMs.
    
    Supports:
    - OpenAI/Azure: gpt-4o, gpt-4, o1-mini, o3-mini, o4-mini
    - Claude: claude-3-opus, claude-3-sonnet, claude-3-haiku
    - Gemini: gemini-pro, gemini-1.5-pro, gemini-2.0-flash
    - Grok: grok-2, grok-3
    - DeepSeek: deepseek-chat, deepseek-reasoner
    - vLLM: any model (requires base_url)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.4,
        embedding_model: str = "text-embedding-3-small",
        n_scenario_expands: int = 3,
        n_perspective_expands: int = 4,
        n_criteria_expands: int = 3,
        dedup_threshold: float = 0.6,
        max_workers: int = 8,
        max_retries: int = 5,
        debug: bool = False,
    ):
        """
        Initialize the generator.
        
        Args:
            model: Model name (auto-detects provider)
            base_url: API base URL (required for vLLM, optional for others)
            api_key: API key (uses env vars if not provided)
            temperature: Generation temperature
            embedding_model: Model for embeddings (deduplication)
            n_scenario_expands: Scenario expansion iterations
            n_perspective_expands: Perspective expansion iterations
            n_criteria_expands: Criteria expansion iterations
            dedup_threshold: Cosine similarity threshold for deduplication (0-1)
            max_workers: Parallel workers for batch processing
            max_retries: Max retries on rate limit errors
            debug: Print raw LLM outputs for debugging
        """
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.n_scenario_expands = n_scenario_expands
        self.n_perspective_expands = n_perspective_expands
        self.n_criteria_expands = n_criteria_expands
        self.dedup_threshold = dedup_threshold
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.debug = debug
        self._verbose = False
        self._log_fn = None
        
        self.provider = self._detect_provider(model)
        self.client = self._init_client(model, base_url, api_key)
        self.embed_client = self._init_embed_client()
    
    def _detect_provider(self, model: str) -> str:
        m = model.lower()
        if 'claude' in m:
            return 'claude'
        elif 'gemini' in m:
            return 'gemini'
        elif 'grok' in m:
            return 'grok'
        elif 'deepseek' in m:
            return 'deepseek'
        elif any(x in m for x in ['gpt', 'o1', 'o3', 'o4']):
            return 'openai'
        else:
            return 'vllm'
    
    def _init_client(self, model: str, base_url: Optional[str], api_key: Optional[str]):
        if self.provider == 'claude':
            import anthropic
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY required")
            return anthropic.Anthropic(api_key=key)
        
        elif self.provider == 'gemini':
            from google import genai
            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY required")
            return genai.Client(api_key=key)
        
        elif self.provider == 'grok':
            from openai import OpenAI
            key = api_key or os.getenv("XAI_API_KEY")
            if not key:
                raise ValueError("XAI_API_KEY required")
            return OpenAI(base_url="https://api.x.ai/v1", api_key=key)
        
        elif self.provider == 'deepseek':
            from openai import OpenAI
            key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not key:
                raise ValueError("DEEPSEEK_API_KEY required")
            return OpenAI(base_url="https://api.deepseek.com/v1", api_key=key)
        
        elif self.provider == 'openai':
            azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY_GPT4O")
            if azure_key and not api_key and not base_url:
                from openai import AzureOpenAI
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com")
                return AzureOpenAI(azure_endpoint=endpoint, api_key=azure_key, api_version="2024-12-01-preview")
            else:
                from openai import OpenAI
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OPENAI_API_KEY or AZURE_OPENAI_API_KEY required")
                return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        
        else:  # vllm
            from openai import OpenAI
            url = base_url or os.getenv("VLLM_SERVER_URL")
            if not url:
                raise ValueError("base_url or VLLM_SERVER_URL required for vLLM")
            key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
            return OpenAI(base_url=url, api_key=key)
    
    def _init_embed_client(self):
        self._embed_provider = None

        # 1. OpenAI / Azure OpenAI
        azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY_GPT4O")
        if azure_key:
            from openai import AzureOpenAI
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com")
            self._embed_provider = 'openai'
            return AzureOpenAI(azure_endpoint=endpoint, api_key=azure_key, api_version="2024-02-01")

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from openai import OpenAI
            self._embed_provider = 'openai'
            return OpenAI(api_key=openai_key)

        # 2. Google (Gemini embedding)
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            from google import genai
            self._embed_provider = 'google'
            self._embed_model_google = "gemini-embedding-001"
            return genai.Client(api_key=google_key)

        # 3. Local sentence-transformers only when both OpenAI and Google keys absent
        try:
            print("Using local sentence-transformers for embeddings")
            from sentence_transformers import SentenceTransformer
            model_name = self.embedding_model if '/' in self.embedding_model else "all-MiniLM-L6-v2"
            self._local_embed_model = SentenceTransformer(model_name)
            self._embed_provider = 'local'
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Please install it with `pip install sentence-transformers` or set OPENAI_API_KEY / GOOGLE_API_KEY")
        return None
    
    def _parse_image(self, image: str):
        """Parse image string, return (base64_data, media_type)."""
        if image.startswith("data:"):
            header, data = image.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
            return data, media_type
        return image, "image/png"
    
    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)
        elif self._verbose:
            print(msg)
    
    def _call_llm(self, agent_name: str, args: Dict, image: Optional[str] = None) -> Any:
        """Call LLM with structured output and retry logic."""
        prompt = get_prompt(agent_name, **args)
        schema = AGENT_SCHEMAS.get(agent_name)
        self._last_raw_response = None
        
        self._log(f"    [LLM] Calling {agent_name} via {self.model}...")
        call_start = time.time()
        
        for attempt in range(self.max_retries):
            try:
                if self.provider == 'claude':
                    result = self._call_claude(prompt, schema, image)
                elif self.provider == 'gemini':
                    result = self._call_gemini(prompt, schema, image)
                else:
                    result = self._call_openai_compatible(prompt, schema, image)
                
                self._last_raw_response = result
                
                if self.debug:
                    self._log(f"[DEBUG {agent_name}] raw result: {result}")
                
                if result is None:
                    raise ValueError("LLM returned None")
                
                elapsed = time.time() - call_start
                self._log(f"    [LLM] {agent_name} done ({elapsed:.1f}s)")
                return result
                
            except Exception as e:
                if self._is_rate_limit(e):
                    wait = min(60, 5 * (attempt + 1))
                    self._log(f"    [LLM] Rate limit hit, waiting {wait}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                self._log(f"    [LLM] {agent_name} error (attempt {attempt + 1}): {e}")
                continue
        raise Exception(f"Max retries ({self.max_retries}) exceeded for {agent_name}")
    
    def _is_rate_limit(self, e: Exception) -> bool:
        err = str(e).lower()
        return any(x in err for x in ['429', 'rate', 'quota', 'limit', 'overloaded'])
    
    def _call_claude(self, prompt: str, schema, image: Optional[str] = None) -> Any:
        import httpx
        
        content = []
        if image:
            b64_data, media_type = self._parse_image(image)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        kwargs = {"model": self.model, "messages": messages, "max_tokens": 32768}
        kwargs["timeout"] = httpx.Timeout(600.0, connect=30.0)
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        
        if schema:
            response = self.client.messages.parse(**kwargs, output_format=schema)
            return response.parsed_output.model_dump()
        else:
            response = self.client.messages.create(**kwargs)
            return self._extract_json(response.content[0].text)
    
    def _call_gemini(self, prompt: str, schema, image: Optional[str] = None) -> Any:
        from google.genai import types
        import base64
        
        config = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        config["max_output_tokens"] = 32768
        
        if "3.0" in self.model:
            config["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        
        if schema:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = schema.model_json_schema()
        
        contents = []
        if image:
            b64_data, media_type = self._parse_image(image)
            contents.append(types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=media_type))
        contents.append(prompt)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        # Use Pydantic validation if schema provided, otherwise just parse JSON
        if schema:
            return schema.model_validate_json(response.text).model_dump()
        return self._extract_json(response.text)
    
    def _call_openai_compatible(self, prompt: str, schema, image: Optional[str] = None) -> Any:
        """For OpenAI, Grok, DeepSeek, vLLM - all support OpenAI API format."""
        if image:
            b64_data, media_type = self._parse_image(image)
            data_uri = f"data:{media_type};base64,{b64_data}"
            content = [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ]
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "messages": messages}
        
        is_reasoning = any(x in self.model.lower() for x in ['o1', 'o3', 'reasoner'])
        if self.temperature is not None and not is_reasoning:
            kwargs["temperature"] = self.temperature
        
        if schema and not is_reasoning:
            if self.provider in ['openai', 'grok']:
                # OpenAI/Grok: use beta.parse with Pydantic schema
                kwargs["response_format"] = schema
                response = self.client.beta.chat.completions.parse(**kwargs)
                if response.choices[0].message.parsed:
                    return response.choices[0].message.parsed.model_dump()
                return self._extract_json(response.choices[0].message.content)
            elif self.provider == 'vllm':
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema()
                    }
                }
            elif self.provider == 'deepseek':
                kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        return self._extract_json(text)
    
    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text response."""
        if not isinstance(text, str):
            return text
        # Direct parse
        try:
            return parse_json(text)
        except:
            pass
        
        # Find JSON in markdown
        for marker in ['```json', '```']:
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find('```', start)
                if end > start:
                    try:
                        return parse_json(text[start:end].strip())
                    except:
                        pass
        return text
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        provider = getattr(self, '_embed_provider', None)

        self._log(f"    [Embed] Computing embeddings for {len(texts)} texts (provider={provider})...")
        embed_start = time.time()

        if provider == 'local':
            result = self._local_embed_model.encode(texts, convert_to_numpy=True)
            self._log(f"    [Embed] Done ({time.time() - embed_start:.1f}s)")
            return result

        if provider == 'openai' and self.embed_client:
            batch_size = 2048
            all_embeddings = []
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start:batch_start + batch_size]
                if len(texts) > batch_size:
                    self._log(f"    [Embed] Batch {batch_start // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} texts)...")
                for attempt in range(self.max_retries):
                    try:
                        response = self.embed_client.embeddings.create(model=self.embedding_model, input=batch)
                        all_embeddings.extend([d.embedding for d in response.data])
                        break
                    except Exception as e:
                        if self._is_rate_limit(e):
                            wait = 5 * (attempt + 1)
                            self._log(f"    [Embed] Rate limit, waiting {wait}s...")
                            time.sleep(wait)
                            continue
                        raise
                else:
                    raise Exception("Embedding failed after max retries")
            self._log(f"    [Embed] Done ({time.time() - embed_start:.1f}s)")
            return np.array(all_embeddings)

        if provider == 'google' and self.embed_client:
            batch_size = 100
            all_vectors = []
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start:batch_start + batch_size]
                if len(texts) > batch_size:
                    self._log(f"    [Embed] Batch {batch_start // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} texts)...")
                for attempt in range(self.max_retries):
                    try:
                        result = self.embed_client.models.embed_content(
                            model=getattr(self, '_embed_model_google', 'text-embedding-004'),
                            contents=batch,
                        )
                        embs = result.embeddings
                        all_vectors.extend([e.values if hasattr(e, 'values') else e for e in embs])
                        break
                    except Exception as e:
                        if self._is_rate_limit(e):
                            wait = 5 * (attempt + 1)
                            self._log(f"    [Embed] Rate limit, waiting {wait}s...")
                            time.sleep(wait)
                            continue
                        raise
                else:
                    raise Exception("Embedding failed after max retries")
            self._log(f"    [Embed] Done ({time.time() - embed_start:.1f}s)")
            return np.array(all_vectors)

        self._log("    [Embed] Fallback to hash-based pseudo-embeddings")
        return np.array([[hash(t) % 1000 / 1000 for _ in range(384)] for t in texts])
    
    def _preprocess_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess item: append web_content to question if present."""
        item = item.copy()
        if "web_content" in item and item["web_content"]:
            web = item.pop("web_content")
            item["question"] = f"{item['question']}\n\n[Retrieved Web Context]\n{web}\n[End of Web Context]"
        if "prompt_id" not in item and "id" in item:
            item["prompt_id"] = item["id"]
        return item
    
    def generate(
        self,
        questions: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]],
        show_progress: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate evaluation criteria for question(s).
        
        Args:
            questions: Single question (str or dict) or list of questions.
                       Each dict should have at least "id" and "question".
                       Optional fields: "image" (base64), "web_content" (str).
                       Can include intermediate results for resumption.
            show_progress: Show progress bar for batch processing.
        
        Returns:
            Single result dict or list of result dicts.
            Each result contains: id, question, scenarios, perspectives,
            criteria, final_criteria, and any error if occurred.
        
        Examples:
            # Single question string
            result = gen.generate("What is AI?")
            
            # Single question dict
            result = gen.generate({"id": "q1", "question": "What is AI?"})
            
            # With image and web content
            result = gen.generate({
                "id": "q1",
                "question": "Analyze this chart",
                "image": "base64...",
                "web_content": "Retrieved context..."
            })
            
            # Batch processing
            results = gen.generate([
                {"id": "q1", "question": "What is AI?"},
                {"id": "q2", "question": "How does ML work?"},
            ])
        """
        single_input, items = normalize_questions(questions)
        
        # Process single item
        def process_one(item, use_parallel=False, verbose=False):
            item = self._preprocess_item(item)
            return run_pipeline(
                item=item,
                call_fn=self._call_llm,
                embed_fn=self._get_embeddings,
                n_scenario_expands=self.n_scenario_expands,
                n_perspective_expands=self.n_perspective_expands,
                n_criteria_expands=self.n_criteria_expands,
                dedup_threshold=self.dedup_threshold,
                use_parallel=use_parallel,
                verbose=verbose,
                log_fn=self._log_fn,
            )
        
        if len(items) == 1:
            self._verbose = True
            result = process_one(items[0], use_parallel=True, verbose=True)
            self._verbose = False
            if "prompt_id" in result:
                result["id"] = result.pop("prompt_id")
            return result if single_input else [result]
        
        self._verbose = False
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_one, item, False, False): item for item in items}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if "prompt_id" in result:
                    result["id"] = result.pop("prompt_id")
                results.append(result)
        
        return results
