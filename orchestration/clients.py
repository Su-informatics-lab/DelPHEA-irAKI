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
import time
from typing import Dict, Optional

import httpx

from config.core import RuntimeConfig

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
        """Initialize vLLM client with configuration.

        Args:
            runtime_config: Runtime configuration
            http_client: Optional shared HTTP client

        Raises:
            ValueError: If configuration is invalid
            ClinicalSafetyError: If critical configuration missing
        """
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.VLLMClient")

        # validate endpoint with clinical context
        self.endpoint = runtime_config.get_vllm_endpoint()
        if not self.endpoint:
            raise ClinicalSafetyError(
                "CRITICAL: vLLM endpoint not configured. "
                "Cannot proceed with irAKI assessment without LLM infrastructure. "
                "Set VLLM_ENDPOINT environment variable or use --vllm-endpoint flag."
            )

        # validate model for clinical use
        if not runtime_config.model_name:
            raise ClinicalSafetyError(
                "CRITICAL: No model specified for clinical reasoning. "
                "DelPHEA-irAKI requires a validated model for medical assessment. "
                "Configure with --model-name flag."
            )

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

        self.logger.info(
            f"Initialized vLLM client for clinical assessment "
            f"[endpoint: {self.endpoint}, model: {runtime_config.model_name}]"
        )

    async def generate_structured_response(
        self, prompt: str, response_format: Dict, expert_context: Optional[Dict] = None
    ) -> Dict:
        """Generate structured JSON response with comprehensive error handling.

        Args:
            prompt: The prompt for the LLM
            response_format: Expected response format specification
            expert_context: Optional context about the requesting expert

        Returns:
            Dict: Parsed JSON response from the model

        Raises:
            ExpertReasoningError: If expert reasoning generation fails
            ClinicalSafetyError: If response is clinically invalid
        """
        # extract clinical context for error messages
        expert_id = (
            expert_context.get("expert_id", "unknown") if expert_context else "unknown"
        )
        expert_specialty = (
            expert_context.get("specialty", "unknown") if expert_context else "unknown"
        )
        case_id = (
            expert_context.get("case_id", "unknown") if expert_context else "unknown"
        )
        assessment_round = (
            expert_context.get("round", "unknown") if expert_context else "unknown"
        )

        request_payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "response_format": response_format,
        }

        try:
            start_time = time.time()

            response = await self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                json=request_payload,
                headers=self.headers,
            )

            elapsed = time.time() - start_time

            # check HTTP status with clinical context
            if response.status_code != 200:
                error_detail = response.text[:500]  # first 500 chars of error

                if response.status_code == 503:
                    raise ClinicalSafetyError(
                        f"CRITICAL: vLLM service unavailable for {expert_specialty} expert "
                        f"assessment of case {case_id} (Round {assessment_round}). "
                        f"Cannot proceed with irAKI evaluation without expert input. "
                        f"Service endpoint: {self.endpoint}. Error: {error_detail}"
                    )
                elif response.status_code == 408:
                    raise ExpertReasoningError(
                        f"TIMEOUT: {expert_specialty} expert (ID: {expert_id}) "
                        f"failed to complete assessment for case {case_id} within "
                        f"{self.config.timeout}s. Clinical reasoning may be too complex. "
                        f"Consider increasing timeout or simplifying assessment."
                    )
                else:
                    raise ExpertReasoningError(
                        f"ERROR: {expert_specialty} expert reasoning failed for case {case_id}. "
                        f"HTTP {response.status_code}: {error_detail}. "
                        f"This prevents complete irAKI assessment - manual review required."
                    )

            response_json = response.json()

            # validate response structure
            if "choices" not in response_json or not response_json["choices"]:
                raise ExpertReasoningError(
                    f"INVALID RESPONSE: {expert_specialty} expert produced no assessment "
                    f"for case {case_id} (Round {assessment_round}). "
                    f"Model may have refused to provide medical assessment. "
                    f"Response: {json.dumps(response_json)[:500]}"
                )

            message_content = response_json["choices"][0]["message"]["content"]

            # parse JSON response
            try:
                parsed_response = json.loads(message_content)
            except json.JSONDecodeError as e:
                raise ExpertReasoningError(
                    f"MALFORMED RESPONSE: {expert_specialty} expert produced invalid JSON "
                    f"for case {case_id}. Cannot parse clinical assessment. "
                    f"Parse error: {e}. Raw content: {message_content[:500]}"
                )

            # validate clinical content based on round
            if assessment_round in ["1", "3", "Round1", "Round3"]:
                # validate assessment has required fields
                if "scores" not in parsed_response:
                    raise ClinicalSafetyError(
                        f"INCOMPLETE ASSESSMENT: {expert_specialty} expert failed to provide "
                        f"question scores for case {case_id} (Round {assessment_round}). "
                        f"irAKI probability cannot be computed without complete scoring."
                    )

                if "p_iraki" not in parsed_response:
                    raise ClinicalSafetyError(
                        f"MISSING PROBABILITY: {expert_specialty} expert did not provide "
                        f"P(irAKI) for case {case_id}. Consensus cannot be computed. "
                        f"Expert must provide probability estimate for all cases."
                    )

                # validate probability is in valid range
                p_iraki = parsed_response.get("p_iraki")
                if not isinstance(p_iraki, (int, float)) or not 0 <= p_iraki <= 1:
                    raise ClinicalSafetyError(
                        f"INVALID PROBABILITY: {expert_specialty} expert provided "
                        f"P(irAKI)={p_iraki} for case {case_id}, which is outside [0,1]. "
                        f"This violates probability axioms and prevents consensus computation."
                    )

            self.logger.info(
                f"Successfully generated {expert_specialty} assessment for case {case_id} "
                f"(Round {assessment_round}) in {elapsed:.2f}s"
            )

            return parsed_response

        except httpx.TimeoutException as e:
            raise ExpertReasoningError(
                f"NETWORK TIMEOUT: {expert_specialty} expert assessment for case {case_id} "
                f"timed out after {self.config.timeout}s. Network issues may prevent "
                f"complete irAKI evaluation. Consider retry or manual assessment. "
                f"Endpoint: {self.endpoint}"
            ) from e

        except httpx.NetworkError as e:
            raise ClinicalSafetyError(
                f"NETWORK FAILURE: Cannot reach vLLM service for {expert_specialty} "
                f"assessment of case {case_id}. irAKI evaluation cannot proceed. "
                f"Verify network connectivity to {self.endpoint}. Error: {e}"
            ) from e

        except Exception as e:
            # catch-all for unexpected errors
            self.logger.error(
                f"Unexpected error in {expert_specialty} assessment for case {case_id}: {e}",
                exc_info=True,
            )
            raise ClinicalSafetyError(
                f"UNEXPECTED ERROR: {expert_specialty} assessment failed for case {case_id}. "
                f"This is an unhandled error that requires immediate attention. "
                f"Error type: {type(e).__name__}. Details logged. "
                f"DO NOT proceed with partial assessment - patient safety risk."
            ) from e

    async def health_check(self) -> bool:
        """Check vLLM service health with clinical context.

        Returns:
            bool: True if service is healthy and ready for clinical use

        Raises:
            ClinicalSafetyError: If health check reveals critical issues
        """
        try:
            response = await self.client.get(
                f"{self.endpoint}/health", headers=self.headers, timeout=10.0
            )

            if response.status_code == 200:
                self.logger.info(f"vLLM service healthy at {self.endpoint}")
                return True
            else:
                self.logger.warning(
                    f"vLLM service unhealthy: HTTP {response.status_code} from {self.endpoint}"
                )
                return False

        except httpx.NetworkError as e:
            raise ClinicalSafetyError(
                f"CRITICAL: Cannot verify vLLM service health at {self.endpoint}. "
                f"DelPHEA-irAKI requires functioning LLM infrastructure for clinical assessment. "
                f"Network error: {e}"
            ) from e

        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return False

    async def close(self):
        """Clean up resources."""
        if self._owns_client and self.client:
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
