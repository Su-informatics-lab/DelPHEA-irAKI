"""
LLM Client for DelPHEA-irAKI with Clinical Safety
===========================================================

Handles communication with vLLM server for expert reasoning with
comprehensive error handling that provides clinical context for failures.

Clinical Safety Principles:
- Fail loudly with clear clinical context
- Never silently default or correct errors
- Provide actionable error messages for operators
- Log full context for post-incident review
"""

import json
import logging
from typing import Any, Dict, Optional

import httpx

from config.core import RuntimeConfig


class StructuredOutputError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


class ClinicalSafetyError(Exception):
    """Base exception for clinical safety-critical failures."""


class ExpertReasoningError(ClinicalSafetyError):
    """Expert failed to generate valid clinical reasoning."""


class ConsensusComputationError(ClinicalSafetyError):
    """Failed to compute consensus for patient case."""


class VLLMClient:
    """Enhanced vLLM client with fail-loud error handling for clinical safety."""

    def __init__(
        self, runtime_config: RuntimeConfig, http_client: httpx.AsyncClient = None
    ):
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.VLLMClient")

        self.endpoint = runtime_config.get_vllm_endpoint()
        if not self.endpoint:
            raise ClinicalSafetyError(
                "CRITICAL: vLLM endpoint not configured. "
                "Set VLLM_ENDPOINT or use --vllm-endpoint."
            )

        if not runtime_config.model_name:
            raise ClinicalSafetyError(
                "CRITICAL: No model specified. Configure RuntimeConfig.model_name."
            )

        if http_client:
            self.client = http_client
            self._owns_client = False
        else:
            timeout = httpx.Timeout(
                connect=30.0, read=runtime_config.timeout, write=10.0, pool=5.0
            )
            self.client = httpx.AsyncClient(timeout=timeout)
            self._owns_client = True

        self.headers = {"Content-Type": "application/json"}
        if runtime_config.api_key:
            self.headers["Authorization"] = f"Bearer {runtime_config.api_key}"

        self.model_name = runtime_config.model_name
        self.logger.info(
            f"Initialized vLLM client for clinical assessment "
            f"[endpoint: {self.endpoint}, model: {self.model_name}]"
        )

    async def _post_and_read_text(self, payload: Dict[str, Any]) -> str:
        """
        Minimal, fail-loud POST to the server. Assumes a /generate style endpoint that
        returns either {'text': '...'} or {'choices': [{'text': '...'}]}.
        Adjust if your server is OpenAI-compatible (then point to /v1/completions or /v1/chat/completions).
        """
        url = f"{self.endpoint.rstrip('/')}/generate"
        resp = await self.client.post(
            url, headers=self.headers, content=json.dumps(payload)
        )
        if resp.status_code != 200:
            raise ClinicalSafetyError(
                f"LLM HTTP {resp.status_code} from {url}. Body: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise ClinicalSafetyError(
                f"Non-JSON response from LLM: {resp.text[:500]}"
            ) from e

        # YAGNI parse: prefer 'text', else first choice.text
        if isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
            text = data["text"]
        elif isinstance(data, dict) and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            text = choice.get("text") or choice.get("message", {}).get("content")
        else:
            raise ClinicalSafetyError(
                f"Unexpected LLM payload shape. Keys: {list(data)[:10]}"
            )

        self.logger.debug(f"LLM raw (first 200): {text[:200]!r}")
        return text

    async def _llm_call(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Replace with actual vLLM call format your server expects.
        This version is a simple /generate with model + prompt.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # If server supports JSON mode/guidance, pass it through
        if response_format and response_format.get("type") == "json_object":
            payload["response_format"] = response_format

        return await self._post_and_read_text(payload)

    async def generate_structured_response(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Force strict JSON with retries and loud failures."""
        required_keys = []
        if response_format and isinstance(response_format.get("schema"), dict):
            required_keys = list(response_format["schema"].keys())

        base_prompt = prompt
        for attempt in range(max_retries + 1):
            if attempt > 0:
                prompt = base_prompt + (
                    "\n\nIMPORTANT:\n"
                    "Return ONLY a single valid JSON object. No markdown, no prose, no code fences.\n"
                    f"Required top-level keys: {required_keys or '[see spec]'}\n"
                )

            text = await self._llm_call(
                prompt=prompt,
                response_format=response_format,
                temperature=0.2 if attempt else 0.5,
                max_tokens=2048,
            )

            # Try parse
            try:
                obj = json.loads(text)
            except Exception:
                # Try first {...} window
                try:
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        obj = json.loads(text[start : end + 1])
                    else:
                        raise
                except Exception:
                    if attempt < max_retries:
                        continue
                    raise StructuredOutputError(
                        f"Model did not return valid JSON after {max_retries+1} attempts."
                    )

            if required_keys:
                missing = [k for k in required_keys if k not in obj]
                if missing:
                    if attempt < max_retries:
                        base_prompt += (
                            f"\n\nThe previous output was missing keys: {missing}. "
                            "Return ONLY JSON with ALL required keys."
                        )
                        continue
                    raise StructuredOutputError(
                        f"Missing required keys after {max_retries+1} attempts: {missing}"
                    )

            return obj

        raise StructuredOutputError("Unknown structured output failure.")

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.endpoint}/health", headers=self.headers, timeout=10.0
            )
            if response.status_code == 200:
                self.logger.info(f"vLLM service healthy at {self.endpoint}")
                return True
            self.logger.warning(
                f"vLLM service unhealthy: HTTP {response.status_code} from {self.endpoint}"
            )
            return False
        except httpx.NetworkError as e:
            raise ClinicalSafetyError(
                f"CRITICAL: Cannot verify vLLM service health at {self.endpoint}. Network error: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return False

    async def close(self):
        if getattr(self, "_owns_client", False) and self.client:
            await self.client.aclose()
            self.logger.info("vLLM client connection closed")


class MockVLLMClient:
    """Mock vLLM client for testing without infrastructure."""

    def __init__(self, runtime_config: RuntimeConfig, http_client=None):
        """Initialize mock client for testing."""
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.MockVLLMClient")
        self.logger.warning(
            "USING MOCK CLIENT: Not suitable for clinical use. "
            "Only for testing DelPHEA-irAKI workflow."
        )

    async def generate_structured_response(
        self, prompt: str, response_format: Dict, expert_context: Optional[Dict] = None
    ) -> Dict:
        """Generate mock response for testing."""
        # return realistic mock data for testing
        import random

        expert_id = (
            expert_context.get("expert_id", "mock") if expert_context else "mock"
        )

        # generate mock scores
        scores = {f"Q{i}": random.randint(3, 8) for i in range(1, 17)}

        return {
            "expert_id": expert_id,
            "scores": scores,
            "p_iraki": random.uniform(0.3, 0.8),
            "ci_iraki": [random.uniform(0.2, 0.4), random.uniform(0.6, 0.9)],
            "confidence": random.uniform(0.6, 0.9),
            "reasoning": "Mock clinical reasoning for testing purposes.",
            "verdict": random.choice([True, False]),
            "differential": ["irAKI", "ATN", "Prerenal"],
            "evidence": ["Temporal relationship", "Urinalysis findings"],
            "next_steps": ["Consider biopsy", "Hold ICI", "Start steroids"],
        }

    async def health_check(self) -> bool:
        """Mock health check always returns True."""
        return True

    async def close(self):
        """Mock cleanup."""
