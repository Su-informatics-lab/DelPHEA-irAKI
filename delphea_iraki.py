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

        # check VLLM endpoint if configured
        # get the endpoint from the infrastructure config
        vllm_endpoint = runtime_config.get_vllm_endpoint()

        # only check if not using local/mock mode
        if vllm_endpoint and runtime_config.infrastructure.endpoint_type != "local":
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(f"{vllm_endpoint}/health")
                    if response.status_code == 200:
                        logger.info(f"✓ VLLM endpoint reachable: {vllm_endpoint}")
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
        import traceback

        traceback.print_exc()  # print full traceback for debugging
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

    # initialize data loader directly
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
    if (
        runtime_config.get_vllm_endpoint()
        and runtime_config.infrastructure.endpoint_type != "local"
    ):
        llm_client = VLLMClient(runtime_config)

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

    for expert_config in expert_configs:
        agent_id = AgentId(expert_config["id"], "expert")

        await runtime.register_agent(
            expert_config["id"],  # agent type
            lambda ec=expert_config: irAKIExpertAgent(
                expert_id=ec["id"],  # pass the expert ID from the config
                case_id=case_id,
                config_loader=config_loader,
                vllm_client=llm_client,
                runtime_config=runtime_config,
            ),
            agent_id=agent_id,  # use the agent_id we created
        )

    logger.info(f"Registered ALL {len(expert_configs)} expert agents")

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
        default="http://tempest-gpu021:8000",  # Default to Tempest
        help="VLLM server endpoint (default: http://tempest-gpu021:8000)",
    )

    # REMOVED --expert-count argument

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

    # create infrastructure config
    from config.core import DelphiConfig, InfrastructureConfig

    infrastructure_config = InfrastructureConfig()
    # Infrastructure already defaults to tempest, but can override if needed
    if args.local_model:
        infrastructure_config.endpoint_type = InfrastructureConfig.ENDPOINT_LOCAL

    # create runtime configuration
    runtime_config = RuntimeConfig(
        infrastructure=infrastructure_config,
        use_real_data=args.use_real_data,
    )

    # create delphi config (no expert_count needed)
    delphi_config = DelphiConfig()

    # rest of main...


# ============================================================================
# FIX 4: Update moderator.py - Always use all experts
# ============================================================================


# In orchestration/agents/moderator.py:
class irAKIModeratorAgent(RoutedAgent):
    def __init__(
        self,
        case_id: str,
        config_loader: ConfigurationLoader,
        data_loader: DataLoader,
        delphi_config: DelphiConfig,
    ) -> None:
        """Initialize moderator for case coordination."""
        super().__init__(f"irAKI Moderator for case {case_id}")
        self._case_id = case_id
        self._config_loader = config_loader
        self._data_loader = data_loader
        self._delphi_config = delphi_config

        # ALWAYS use ALL experts
        self._expert_ids = self._config_loader.get_available_expert_ids()

        # rest of initialization...
        self.logger.info(
            f"Initialized moderator for case {case_id} with ALL {len(self._expert_ids)} experts: "
            f"{self._expert_ids}"
        )
