"""
DelPHEA-irAKI: Delphi Personalized Health Explainable Agents for immune-related AKI
====================================================================================

Main entry point for the DelPHEA-irAKI clinical decision support system.
This orchestrates the multi-agent Delphi consensus process for distinguishing
immune-related AKI from alternative causes in patients receiving ICIs.

Architecture:
------------
    Main (this file)
         │
    ┌────┴────┐
    │ Runtime │──> Agents (Moderator + Experts)
    └─────────┘         │
         │              ▼
    Configuration   Delphi Process
    Data Loading    Consensus

Usage:
------
    # Run irAKI assessment for a case
    python delphea_iraki.py --case-id iraki_case_001

    # Health check
    python delphea_iraki.py --health-check

    # Verbose mode
    python delphea_iraki.py --case-id iraki_case_002 --verbose

Clinical Context:
----------------
Processes patient cases through expert panel simulation to distinguish
true immune-related AKI from mimics, preventing unnecessary steroid
treatment and cancer therapy interruption.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict

import httpx
from autogen_core import AgentId, SingleThreadedAgentRuntime

sys.path.append(str(Path(__file__).parent))

from config.core import DelphiConfig, RuntimeConfig
from config.loader import ConfigurationLoader
from dataloader import DataLoader
from orchestration.agents.expert import irAKIExpertAgent
from orchestration.agents.moderator import irAKIModeratorAgent
from orchestration.clients import VLLMClient
from orchestration.messages import StartCase

# configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def run_health_check(runtime_config: RuntimeConfig) -> bool:
    """Run system health check.

    Validates:
    - Configuration files exist and are valid
    - Data loader can initialize
    - VLLM endpoint is reachable (if not local)

    Args:
        runtime_config: Runtime configuration

    Returns:
        bool: True if all checks pass
    """
    logger = logging.getLogger("health_check")
    logger.info("Starting health check...")

    try:
        # check configuration
        config_loader = ConfigurationLoader(runtime_config)
        logger.info(
            f"✓ Configuration loaded: {len(config_loader.expert_panel['expert_panel']['experts'])} experts"
        )

        # check data loader - it handles dummy vs real internally
        data_loader = DataLoader(
            data_dir=runtime_config.data_dir, use_dummy=not runtime_config.use_real_data
        )

        if data_loader.is_available():
            mode = "DUMMY" if data_loader.use_dummy else "REAL"
            logger.info(f"✓ DataLoader initialized in {mode} mode")
        else:
            logger.error("✗ DataLoader not available")
            return False

        # check VLLM endpoint if provided
        if runtime_config.vllm_endpoint and not runtime_config.local_model:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        f"{runtime_config.vllm_endpoint}/health"
                    )
                    if response.status_code == 200:
                        logger.info(
                            f"✓ VLLM endpoint reachable: {runtime_config.vllm_endpoint}"
                        )
                    else:
                        logger.warning(
                            f"⚠ VLLM endpoint returned status {response.status_code}"
                        )
                except Exception as e:
                    logger.warning(f"⚠ Could not reach VLLM endpoint: {e}")

        logger.info("✓ Health check passed")
        return True

    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False


async def run_iraki_assessment(case_id: str, runtime_config: RuntimeConfig) -> Dict:
    """Run irAKI assessment for a patient case.

    Args:
        case_id: Patient case identifier
        runtime_config: Runtime configuration

    Returns:
        Dict: Assessment results with consensus probability and recommendations
    """
    logger = logging.getLogger("assessment")
    logger.info(f"Starting irAKI assessment for case: {case_id}")

    # initialize components
    config_loader = ConfigurationLoader(runtime_config)
    delphi_config = DelphiConfig()

    # initialize data loader directly - no wrapper needed
    data_loader = DataLoader(
        data_dir=runtime_config.data_dir, use_dummy=not runtime_config.use_real_data
    )

    # load patient case
    try:
        patient_case = data_loader.load_patient_case(case_id)
        logger.info(f"Loaded case: {patient_case['case_id']}")
    except ValueError as e:
        logger.error(f"Failed to load case {case_id}: {e}")
        raise

    # create runtime for agents
    runtime = SingleThreadedAgentRuntime()

    # initialize VLLM client if using remote endpoint
    llm_client = None
    if runtime_config.vllm_endpoint and not runtime_config.local_model:
        llm_client = VLLMClient(
            base_url=runtime_config.vllm_endpoint, model=runtime_config.model_name
        )

    # register moderator agent
    moderator_id = AgentId("moderator", "iraki")
    await runtime.register_agent(
        "moderator",
        lambda: irAKIModeratorAgent(
            case_id=case_id,
            config_loader=config_loader,
            data_loader=data_loader,  # pass DataLoader directly
            delphi_config=delphi_config,
        ),
        agent_id=moderator_id,
    )

    # register expert agents based on configuration
    expert_configs = config_loader.expert_panel["expert_panel"]["experts"]

    # limit experts if specified
    if runtime_config.expert_count:
        expert_configs = expert_configs[: runtime_config.expert_count]

    for expert_config in expert_configs:
        expert_id = AgentId(expert_config["id"], "expert")
        await runtime.register_agent(
            expert_config["id"],
            lambda ec=expert_config: irAKIExpertAgent(
                expert_config=ec, llm_client=llm_client, config_loader=config_loader
            ),
            agent_id=expert_id,
        )

    logger.info(f"Registered {len(expert_configs)} expert agents")

    # start assessment by sending case to moderator
    await runtime.send_message(
        StartCase(
            case_id=case_id,
            patient_data=patient_case,
            expert_ids=[e["id"] for e in expert_configs],
        ),
        moderator_id,
    )

    # run until completion
    await runtime.stop()

    # return results (placeholder for now)
    return {
        "case_id": case_id,
        "status": "completed",
        "consensus": "Assessment complete - results would be here",
    }


def main():
    """Main entry point for DelPHEA-irAKI."""
    parser = argparse.ArgumentParser(
        description="DelPHEA-irAKI: Expert consensus system for immune-related AKI classification"
    )

    parser.add_argument(
        "--case-id",
        type=str,
        help="Patient case ID to assess (e.g., iraki_case_001 or dummy_001)",
    )

    parser.add_argument(
        "--health-check", action="store_true", help="Run system health check"
    )

    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default="http://localhost:8000",
        help="VLLM server endpoint (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--expert-count",
        type=int,
        default=None,
        help="Number of experts to use (default: all configured experts)",
    )

    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real patient data if available (default: use dummy data)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--local-model",
        action="store_true",
        help="Use local model instead of VLLM endpoint",
    )

    args = parser.parse_args()

    # set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # create infrastructure config with the endpoint
    from config.core import DelphiConfig, InfrastructureConfig

    infrastructure_config = InfrastructureConfig()

    # determine endpoint type and set accordingly
    if args.local_model:
        infrastructure_config.endpoint_type = InfrastructureConfig.ENDPOINT_LOCAL
        infrastructure_config.local_endpoint = args.vllm_endpoint
    else:
        # assume AWS endpoint by default (can be enhanced to detect)
        infrastructure_config.endpoint_type = InfrastructureConfig.ENDPOINT_AWS
        infrastructure_config.aws_endpoint = args.vllm_endpoint

    # create runtime configuration with proper parameters
    runtime_config = RuntimeConfig(
        infrastructure=infrastructure_config,
        use_real_data=args.use_real_data,
    )

    # create delphi config if expert count is specified
    delphi_config = None
    if args.expert_count:
        delphi_config = DelphiConfig(expert_count=args.expert_count)

    # run appropriate command
    if args.health_check:
        success = asyncio.run(run_health_check(runtime_config))
        sys.exit(0 if success else 1)

    elif args.case_id:
        try:
            # pass delphi_config if created
            if delphi_config:
                results = asyncio.run(
                    run_iraki_assessment(args.case_id, runtime_config, delphi_config)
                )
            else:
                results = asyncio.run(
                    run_iraki_assessment(args.case_id, runtime_config)
                )
            print(f"\nAssessment Results:\n{results}")
        except Exception as e:
            logging.error(f"Assessment failed: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)
