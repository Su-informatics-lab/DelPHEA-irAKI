"""
LLM client for DelPHEA-irAKI.

Handles communication with vLLM server for expert reasoning.
"""

import json
import logging
from typing import Dict

import httpx

from config.core import RuntimeConfig

logger = logging.getLogger(__name__)


class VLLMClient:
    """Enhanced vLLM client with fail-loud error handling."""

    def __init__(
        self, runtime_config: RuntimeConfig, http_client: httpx.AsyncClient = None
    ):
        """Initialize vLLM client with configuration.

        Args:
            runtime_config: Runtime configuration
            http_client: Optional shared HTTP client

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.VLLMClient")

        # validate endpoint
        self.endpoint = runtime_config.get_vllm_endpoint()
        if not self.endpoint:
            raise ValueError("vLLM endpoint cannot be empty")

        # use provided client or create new one
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

        self.logger.info(f"Initialized vLLM client for endpoint: {self.endpoint}")

    async def generate_structured_response(
        self, prompt: str, response_format: Dict
    ) -> Dict:
        """Generate structured JSON response with comprehensive error handling.

        Args:
            prompt: Input prompt
            response_format: Expected response format

        Returns:
            Dict: Parsed JSON response

        Raises:
            httpx.HTTPStatusError: If API returns error status
            ValueError: If response cannot be parsed
            TimeoutError: If request times out
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        if not response_format:
            raise ValueError("Response format must be specified")

        request_data = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "response_format": response_format,
        }

        try:
            self.logger.debug(f"Sending request to {self.endpoint}")
            response = await self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                headers=self.headers,
                json=request_data,
            )

            if response.status_code != 200:
                error_msg = f"vLLM API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise httpx.HTTPStatusError(
                    error_msg,
                    request=response.request,
                    response=response,
                )

            result = response.json()

            # validate response structure
            if "choices" not in result or not result["choices"]:
                raise ValueError(f"Invalid API response structure: {result}")

            content = result["choices"][0]["message"]["content"]

            # parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {content[:200]}...")
                raise ValueError(f"Invalid JSON in API response: {e}")

        except httpx.TimeoutException as e:
            self.logger.error(f"Request timeout after {self.config.timeout}s")
            raise TimeoutError(f"vLLM request timed out: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error in vLLM client: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if vLLM server is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        health_endpoints = ["/health", "/healthz", "/v1/models"]

        for endpoint in health_endpoints:
            try:
                response = await self.client.get(
                    f"{self.endpoint}{endpoint}", timeout=5.0
                )
                if response.status_code == 200:
                    self.logger.info(f"Health check passed at {endpoint}")
                    return True
            except Exception as e:
                self.logger.debug(f"Health check failed at {endpoint}: {e}")
                continue

        self.logger.warning(f"All health checks failed for {self.endpoint}")
        return False

    async def close(self):
        """Clean up HTTP client if owned."""
        if self._owns_client:
            await self.client.aclose()
            self.logger.debug("Closed HTTP client")
