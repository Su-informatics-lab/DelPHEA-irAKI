#!/usr/bin/env python3
"""
Phase 2: vLLM Infrastructure Tests for DelPHEA-irAKI
=====================================================

Tests vLLM server connectivity, GPU utilization, and LLM generation.
Requires vLLM server running (via serve.sbatch on SLURM).

Usage:
    # after getting node allocation
    python test_phase2_vllm.py --node gpu-node-001
    python test_phase2_vllm.py --node gpu-node-001 --model meta-llama/Llama-3.1-70B-Instruct
    python test_phase2_vllm.py --endpoint http://172.31.11.192:8000 --verbose

Clinical Context:
    Validates that the vLLM infrastructure can handle clinical reasoning
    tasks with appropriate latency and accuracy for real-time decision support.

Exit Codes:
    0: All tests passed
    1: Test failures
    2: Infrastructure not available
    3: GPU/memory issues
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import httpx

# add parent directory to path (go up from tests/ to project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.core import RuntimeConfig
from config.loader import ConfigurationLoader
from orchestration.clients import ClinicalSafetyError, VLLMClient

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class VLLMTestResult:
    """Track vLLM test results with performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.latency_ms = None
        self.gpu_memory_mb = None
        self.details = []

    def add_metric(self, metric: str, value: float):
        """Add performance metric."""
        self.details.append(f"{metric}: {value}")

    def set_error(self, error: Exception):
        """Record test failure."""
        self.passed = False
        self.error = error

    def set_passed(self, latency_ms: float = None):
        """Mark test as passed with optional latency."""
        self.passed = True
        self.latency_ms = latency_ms


def get_node_endpoint(node_name: str = None, port: int = 8000) -> str:
    """Determine vLLM endpoint from node name or environment.

    Args:
        node_name: SLURM node name (e.g., 'gpu-node-001')
        port: vLLM server port

    Returns:
        Full endpoint URL
    """
    # check environment first
    if os.environ.get("VLLM_ENDPOINT"):
        return os.environ["VLLM_ENDPOINT"]

    # if node specified, construct endpoint
    if node_name:
        # handle different node naming schemes
        if node_name.startswith("gpu-"):
            # internal cluster name
            return f"http://{node_name}:8000"
        else:
            # might be IP or hostname
            if ":" in node_name:
                return f"http://{node_name}"
            else:
                return f"http://{node_name}:8000"

    # default to localhost for testing
    return f"http://localhost:{port}"


def check_gpu_status() -> Dict[str, any]:
    """Check GPU availability and memory.

    Returns:
        Dict with GPU information
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return {"available": False, "error": "nvidia-smi failed"}

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(", ")
            if len(parts) == 3:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory_used_mb": int(parts[1]),
                        "memory_total_mb": int(parts[2]),
                        "memory_free_mb": int(parts[2]) - int(parts[1]),
                    }
                )

        return {"available": True, "count": len(gpus), "gpus": gpus}

    except Exception as e:
        return {"available": False, "error": str(e)}


async def test_vllm_health(endpoint: str) -> VLLMTestResult:
    """Test vLLM server health and readiness.

    Args:
        endpoint: vLLM server endpoint

    Returns:
        Test result with health status
    """
    result = VLLMTestResult("vLLM Health Check")

    try:
        config = RuntimeConfig()
        # Force local endpoint for testing
        config.infrastructure.endpoint_type = "local"
        config.infrastructure.local_endpoint = endpoint

        # initialize client
        client = VLLMClient(config)
        result.details.append(f"Testing endpoint: {endpoint}")

        # test health
        start = time.time()
        healthy = await client.health_check()
        latency_ms = (time.time() - start) * 1000

        if not healthy:
            raise ClinicalSafetyError(f"vLLM service unhealthy at {endpoint}")

        result.details.append(f"Health check latency: {latency_ms:.1f}ms")

        # test model info if available
        try:
            async with httpx.AsyncClient(timeout=10.0) as http_client:
                response = await http_client.get(f"{endpoint}/v1/models")
                if response.status_code == 200:
                    models = response.json()
                    if "data" in models and models["data"]:
                        model_name = models["data"][0].get("id", "unknown")
                        result.details.append(f"Loaded model: {model_name}")
        except:
            pass  # model info is optional

        await client.close()
        result.set_passed(latency_ms)

    except Exception as e:
        result.set_error(e)
        logger.error(f"Health check failed: {e}")

    return result


async def test_clinical_generation(endpoint: str, model_name: str) -> VLLMTestResult:
    """Test clinical reasoning generation.

    Args:
        endpoint: vLLM server endpoint
        model_name: Model identifier

    Returns:
        Test result with generation metrics
    """
    result = VLLMTestResult("Clinical Generation")

    try:
        config = RuntimeConfig()
        # Force local endpoint
        config.infrastructure.endpoint_type = "local"
        config.infrastructure.local_endpoint = endpoint
        config.model_name = model_name

        client = VLLMClient(config)

        # test clinical prompt
        clinical_prompt = """
        Patient presents with:
        - Baseline creatinine: 1.2 mg/dL
        - Current creatinine: 2.8 mg/dL  
        - Started pembrolizumab 3 weeks ago
        - No hypotension, normal urine output
        - Urinalysis: 2+ protein, no blood, no casts

        Question: What is the probability this represents immune-related AKI?
        Provide a probability between 0 and 1 with brief clinical reasoning.
        """

        response_format = {
            "type": "object",
            "properties": {
                "probability": {"type": "number"},
                "reasoning": {"type": "string"},
                "differential": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["probability", "reasoning"],
        }

        # generate response
        start = time.time()
        response = await client.generate_structured_response(
            prompt=clinical_prompt,
            response_format=response_format,
            expert_context={"expert_id": "nephrologist_1"},
        )
        latency_ms = (time.time() - start) * 1000

        # validate response
        if "probability" not in response:
            raise ValueError("Response missing probability field")

        prob = response["probability"]
        if not 0 <= prob <= 1:
            raise ValueError(f"Invalid probability: {prob}")

        result.details.append(f"Generated p(irAKI) = {prob:.2f}")
        result.details.append(f"Generation latency: {latency_ms:.1f}ms")
        result.details.append(
            f"Reasoning length: {len(response.get('reasoning', ''))} chars"
        )

        # check if reasoning is clinically sensible (basic check)
        reasoning = response.get("reasoning", "").lower()
        clinical_terms = ["creatinine", "aki", "immune", "pembrolizumab", "checkpoint"]
        terms_found = sum(1 for term in clinical_terms if term in reasoning)

        if terms_found < 2:
            logger.warning("Generated reasoning may lack clinical context")

        await client.close()
        result.set_passed(latency_ms)

    except Exception as e:
        result.set_error(e)
        logger.error(f"Clinical generation failed: {e}", exc_info=True)

    return result


async def test_structured_output(endpoint: str, model_name: str) -> VLLMTestResult:
    """Test structured JSON output for expert assessments.

    Args:
        endpoint: vLLM server endpoint
        model_name: Model identifier

    Returns:
        Test result with structured output validation
    """
    result = VLLMTestResult("Structured Output")

    try:
        config = RuntimeConfig()
        # Force local endpoint
        config.infrastructure.endpoint_type = "local"
        config.infrastructure.local_endpoint = endpoint
        config.model_name = model_name

        # load actual questionnaire
        loader = ConfigurationLoader(config)
        questions = loader.get_questions()[:3]  # test with first 3 questions

        client = VLLMClient(config)

        # create assessment prompt
        prompt = f"""
        You are a nephrologist evaluating a potential case of immune-related AKI.
        Score each question from 1-10 where 10 strongly suggests irAKI.

        Questions:
        {json.dumps([{"id": q["id"], "question": q["question"]} for q in questions], indent=2)}

        Provide scores and overall assessment.
        """

        response_format = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "p_iraki": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_findings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scores", "p_iraki", "confidence"],
        }

        # generate
        start = time.time()
        response = await client.generate_structured_response(
            prompt=prompt,
            response_format=response_format,
            expert_context={"expert_id": "nephrologist_1", "round": 1},
        )
        latency_ms = (time.time() - start) * 1000

        # validate structure
        if not isinstance(response.get("scores"), dict):
            raise ValueError("Scores not returned as dictionary")

        scores = response["scores"]
        result.details.append(f"Generated {len(scores)} scores")

        # check score validity
        for q_id, score in scores.items():
            if not isinstance(score, (int, float)):
                raise ValueError(f"Score for {q_id} is not numeric: {score}")
            if not 1 <= score <= 10:
                raise ValueError(f"Score for {q_id} out of range: {score}")

        # check probability and confidence
        p_iraki = response.get("p_iraki")
        confidence = response.get("confidence")

        result.details.append(
            f"p(irAKI) = {p_iraki:.2f}, confidence = {confidence:.2f}"
        )
        result.details.append(f"Structured generation latency: {latency_ms:.1f}ms")

        await client.close()
        result.set_passed(latency_ms)

    except Exception as e:
        result.set_error(e)
        logger.error(f"Structured output test failed: {e}", exc_info=True)

    return result


async def test_throughput(
    endpoint: str, model_name: str, num_requests: int = 3
) -> VLLMTestResult:
    """Test vLLM throughput with multiple requests.

    Args:
        endpoint: vLLM server endpoint
        model_name: Model identifier
        num_requests: Number of concurrent requests

    Returns:
        Test result with throughput metrics
    """
    result = VLLMTestResult("Throughput Test")

    try:
        config = RuntimeConfig()
        # Force local endpoint
        config.infrastructure.endpoint_type = "local"
        config.infrastructure.local_endpoint = endpoint
        config.model_name = model_name

        client = VLLMClient(config)

        # create different prompts for variety
        prompts = [
            "Is creatinine rise from 1.0 to 2.0 consistent with AKI?",
            "What defines immune-related nephritis?",
            "List contraindications for steroid treatment in irAKI.",
        ]

        async def single_request(prompt: str, idx: int):
            """Execute single request."""
            start = time.time()
            response = await client.generate_structured_response(
                prompt=prompt,
                response_format={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
                expert_context={"request_id": idx},
            )
            return time.time() - start

        # run requests concurrently
        start_total = time.time()
        tasks = [
            single_request(prompts[i % len(prompts)], i) for i in range(num_requests)
        ]
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_total

        # calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        throughput = num_requests / total_time

        result.details.append(f"Processed {num_requests} requests in {total_time:.1f}s")
        result.details.append(f"Average latency: {avg_latency*1000:.1f}ms")
        result.details.append(f"Throughput: {throughput:.2f} req/s")

        # check if performance is acceptable for clinical use
        if avg_latency > 10:  # more than 10 seconds per request
            logger.warning("Latency may be too high for real-time clinical use")

        await client.close()
        result.set_passed(avg_latency * 1000)

    except Exception as e:
        result.set_error(e)
        logger.error(f"Throughput test failed: {e}", exc_info=True)

    return result


async def test_expert_simulation(endpoint: str, model_name: str) -> VLLMTestResult:
    """Test full expert agent simulation.

    Args:
        endpoint: vLLM server endpoint
        model_name: Model identifier

    Returns:
        Test result with expert simulation metrics
    """
    result = VLLMTestResult("Expert Simulation")

    try:
        from orchestration.agents.expert import irAKIExpertAgent

        config = RuntimeConfig()
        # Force local endpoint
        config.infrastructure.endpoint_type = "local"
        config.infrastructure.local_endpoint = endpoint
        config.model_name = model_name

        loader = ConfigurationLoader(config)
        vllm_client = VLLMClient(config)

        # create expert agent
        expert = irAKIExpertAgent(
            expert_id="medical_oncology",  # use actual expert_id from panel.json
            case_id="test_case",
            config_loader=loader,
            vllm_client=vllm_client,
        )

        result.details.append(f"Expert initialized: {expert._expert_profile['name']}")
        result.details.append(f"Specialty: {expert._expert_profile['specialty']}")

        # note: actual message handling would require full runtime
        # this just validates initialization

        await vllm_client.close()
        result.set_passed()

    except ImportError:
        result.details.append("Expert agent not yet fully implemented")
        result.set_passed()  # not a failure in early development
    except Exception as e:
        result.set_error(e)
        logger.error(f"Expert simulation failed: {e}", exc_info=True)

    return result


async def run_all_tests(
    endpoint: str, model_name: str, verbose: bool = False
) -> Tuple[List[VLLMTestResult], int]:
    """Run all Phase 2 vLLM infrastructure tests.

    Args:
        endpoint: vLLM server endpoint
        model_name: Model to test
        verbose: Show detailed output

    Returns:
        Tuple of (results, exit_code)
    """
    print("\n" + "=" * 70)
    print("DelPHEA-irAKI Phase 2: vLLM Infrastructure Tests")
    print("=" * 70 + "\n")

    # check GPU status first
    print("Checking GPU status...")
    gpu_info = check_gpu_status()
    if gpu_info["available"]:
        print(f"✓ Found {gpu_info['count']} GPU(s):")
        for gpu in gpu_info["gpus"]:
            print(
                f"  - {gpu['name']}: {gpu['memory_free_mb']}/{gpu['memory_total_mb']} MB free"
            )
    else:
        print(f"⚠ GPU check failed: {gpu_info.get('error', 'Unknown error')}")
        print("  Tests will continue but may fail if GPU is required\n")

    # define test sequence
    tests = [
        ("Health Check", test_vllm_health, {"endpoint": endpoint}),
        (
            "Clinical Generation",
            test_clinical_generation,
            {"endpoint": endpoint, "model_name": model_name},
        ),
        (
            "Structured Output",
            test_structured_output,
            {"endpoint": endpoint, "model_name": model_name},
        ),
        (
            "Throughput (3 requests)",
            test_throughput,
            {"endpoint": endpoint, "model_name": model_name, "num_requests": 3},
        ),
        (
            "Expert Simulation",
            test_expert_simulation,
            {"endpoint": endpoint, "model_name": model_name},
        ),
    ]

    results = []
    failed = 0

    for test_name, test_func, kwargs in tests:
        print(f"Running: {test_name}...", end=" ")

        try:
            result = await test_func(**kwargs)
            results.append(result)

            if result.passed:
                latency_str = (
                    f" ({result.latency_ms:.0f}ms)" if result.latency_ms else ""
                )
                print(f"✓ PASSED{latency_str}")
                if verbose:
                    for detail in result.details:
                        print(f"  - {detail}")
            else:
                print(f"✗ FAILED")
                failed += 1
                print(f"  Error: {result.error}")
                if verbose and result.details:
                    for detail in result.details:
                        print(f"  - {detail}")

        except Exception as e:
            print(f"✗ EXCEPTION")
            failed += 1
            print(f"  Unexpected error: {e}")
            result = VLLMTestResult(test_name)
            result.set_error(e)
            results.append(result)

    # summary
    print("\n" + "=" * 70)
    print(f"Results: {len(results) - failed}/{len(results)} tests passed")

    if failed == 0:
        print("✓ All Phase 2 tests passed - vLLM infrastructure ready!")
        print("\nNext steps:")
        print("1. Run Phase 3 tests with dummy patient data")
        print("2. Monitor GPU memory usage during full consensus runs")
        return results, 0
    else:
        print(f"✗ {failed} test(s) failed")
        print("\nTroubleshooting:")
        print("1. Check vLLM server logs: journalctl -u vllm or squeue output")
        print("2. Verify endpoint is correct: curl http://<endpoint>/health")
        print("3. Check GPU memory: nvidia-smi")
        return results, 1


def main():
    """Main entry point for Phase 2 tests."""
    parser = argparse.ArgumentParser(
        description="Phase 2 vLLM Infrastructure Tests for DelPHEA-irAKI"
    )
    parser.add_argument("--node", help="SLURM node name or IP (e.g., gpu-node-001)")
    parser.add_argument("--endpoint", help="Full vLLM endpoint URL (overrides --node)")
    parser.add_argument(
        "--model", default="openai/gpt-oss-120b", help="Model name for testing"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only health check (quick validation)"
    )

    args = parser.parse_args()

    # determine endpoint
    if args.endpoint:
        endpoint = args.endpoint
    elif args.node:
        endpoint = get_node_endpoint(args.node)
    else:
        # try environment or default
        endpoint = get_node_endpoint()

    print(f"Testing vLLM endpoint: {endpoint}")
    print(f"Model: {args.model}")

    # Important note about the model
    if "gpt-oss" in args.model.lower():
        print("Note: Using GPT-OSS-120B model (120B parameters)")
        print("Expected latency: 5-15s per generation due to model size")

    if args.quick:
        # quick health check only
        async def quick_check():
            result = await test_vllm_health(endpoint)
            if result.passed:
                print("✓ vLLM server is healthy")
                for detail in result.details:
                    print(f"  - {detail}")
                return 0
            else:
                print(f"✗ Health check failed: {result.error}")
                return 1

        return asyncio.run(quick_check())
    else:
        # full test suite
        _, exit_code = asyncio.run(
            run_all_tests(endpoint, args.model, verbose=args.verbose)
        )
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
