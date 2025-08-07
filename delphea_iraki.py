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

import httpx
from autogen_core import AgentId, SingleThreadedAgentRuntime

# data layer imports
from data.loader import DataLoaderWrapper

# configuration imports
from config.core import DelphiConfig, RuntimeConfig
from config.loader import ConfigurationLoader

# orchestration imports
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

    Args:
        runtime_config: Runtime configuration

    Returns:
        bool: True if all checks pass
    """
    try:
        # validate configurations
        config_loader = ConfigurationLoader(runtime_config)

        # check vLLM endpoint
        vllm_client = VLLMClient(runtime_config)
        healthy = await vllm_client.health_check()
        await vllm_client.close()

        if healthy:
            print("✓ DelPHEA-irAKI system healthy")
            print(f"✓ Loaded {len(config_loader.get_available_expert_ids())} experts")
            print(f"✓ Loaded {len(config_loader.get_questions())} questions")
            print(f"✓ vLLM endpoint: {runtime_config.get_vllm_endpoint()}")
            print(f"✓ Model: {runtime_config.model_name}")
            return True
        else:
            print(
                f"✗ vLLM health check failed for {runtime_config.get_vllm_endpoint()}"
            )
            return False

    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


async def run_delphi_consensus(
    case_id: str,
    runtime_config: RuntimeConfig,
    delphi_config: DelphiConfig,
) -> int:
    """Run the Delphi consensus process for a case.

    Args:
        case_id: Case identifier
        runtime_config: Runtime configuration
        delphi_config: Delphi methodology configuration

    Returns:
        int: Exit code (0 for success)
    """
    logger = logging.getLogger("delphea_iraki.main")

    try:
        # initialize configuration loader
        logger.info("Loading configurations...")
        config_loader = ConfigurationLoader(runtime_config)

        # initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoaderWrapper(
            data_dir=runtime_config.data_dir, use_real_data=runtime_config.use_real_data
        )

        # create shared HTTP client for all vLLM requests
        logger.info("Setting up vLLM client...")
        shared_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0, read=runtime_config.timeout, write=10.0, pool=5.0
            )
        )

        # create shared vLLM client
        shared_vllm_client = VLLMClient(runtime_config, shared_http_client)

        # verify vLLM is available
        if not await shared_vllm_client.health_check():
            logger.error("vLLM server is not healthy")
            return 1

        # create runtime for agent orchestration
        logger.info("Initializing agent runtime...")
        runtime = SingleThreadedAgentRuntime()

        # register moderator agent
        logger.info("Registering moderator agent...")
        await runtime.register(
            "Moderator",
            lambda _: irAKIModeratorAgent(
                case_id=case_id,
                config_loader=config_loader,
                data_loader=data_loader,
                delphi_config=delphi_config,
            ),
        )

        # register expert agents
        expert_ids = config_loader.get_available_expert_ids()[
            : delphi_config.expert_count
        ]
        logger.info(f"Registering {len(expert_ids)} expert agents...")

        for expert_id in expert_ids:
            await runtime.register(
                f"Expert_{expert_id}",
                lambda _, eid=expert_id: irAKIExpertAgent(
                    expert_id=eid,
                    case_id=case_id,
                    config_loader=config_loader,
                    vllm_client=shared_vllm_client,  # share vLLM client
                ),
            )

        # start runtime
        logger.info("Starting agent runtime...")
        await runtime.start()

        # bootstrap the Delphi process
        logger.info(f"Starting Delphi process for case: {case_id}")
        await runtime.send_message(
            StartCase(case_id=case_id), AgentId("Moderator", case_id)
        )

        # wait for completion
        logger.info("Waiting for consensus completion...")
        await runtime.stop_when_idle()

        # cleanup
        logger.info("Cleaning up resources...")
        await shared_vllm_client.close()
        await shared_http_client.aclose()
        await runtime.stop()

        logger.info("✓ DelPHEA-irAKI completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in Delphi consensus: {e}", exc_info=True)
        return 1


async def main() -> int:
    """Main entry point for DelPHEA-irAKI system.

    Returns:
        int: Exit code
    """
    parser = argparse.ArgumentParser(
        description="DelPHEA-irAKI: Clinical Decision Support for immune-related AKI"
    )

    # core arguments
    parser.add_argument(
        "--case-id",
        default="iraki_case_001",
        help="Patient case identifier (default: iraki_case_001)",
    )

    # configuration files
    parser.add_argument(
        "--expert-panel-config",
        default="config/panel.json",
        help="Expert panel configuration file",
    )
    parser.add_argument(
        "--questionnaire-config",
        default="config/questionnaire.json",
        help="Assessment questionnaire configuration",
    )
    parser.add_argument(
        "--prompts-dir", default="prompts", help="Directory containing prompt templates"
    )

    # vLLM endpoint (can be overridden by infrastructure selection)
    parser.add_argument("--vllm-endpoint", help="Override vLLM server endpoint")
    parser.add_argument(
        "--ssh-tunnel",
        action="store_true",
        help="Use SSH tunnel for Tempest connection",
    )
    parser.add_argument(
        "--model-name", default="openai/gpt-oss-120b", help="Model name for inference"
    )

    # data configuration
    parser.add_argument(
        "--data-dir", default="irAKI_data", help="Directory containing patient data"
    )
    parser.add_argument(
        "--use-dummy-data", action="store_true", help="Use dummy data for testing"
    )

    # Delphi process configuration
    parser.add_argument(
        "--conflict-method",
        choices=["range", "std", "category"],
        default="category",
        help="Method for identifying conflicts: range, std, or category (default: category)",
    )

    # system options
    parser.add_argument(
        "--health-check", action="store_true", help="Run health check and exit"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # create runtime configuration
    runtime_config = RuntimeConfig(
        expert_panel_config=args.expert_panel_config,
        questionnaire_config=args.questionnaire_config,
        prompts_dir=args.prompts_dir,
        model_name=args.model_name,
        data_dir=args.data_dir,
        use_real_data=not args.use_dummy_data,
    )

    # configure infrastructure
    if args.infrastructure == "tempest":
        runtime_config.infrastructure.use_tempest = True
        if args.ssh_tunnel:
            print(
                f"SSH Tunnel command: {runtime_config.infrastructure.get_ssh_tunnel_command()}"
            )
            runtime_config.infrastructure.use_ssh_tunnel = True
    elif args.infrastructure == "aws":
        # AWS endpoint should be provided via environment or config
        runtime_config.infrastructure.use_aws = True

    # update vLLM endpoint based on infrastructure
    if args.vllm_endpoint:
        runtime_config.infrastructure.local_endpoint = args.vllm_endpoint

    # create Delphi configuration with intelligent conflict detection
    delphi_config = DelphiConfig(
        expert_count=len(config_loader.get_available_expert_ids()),  # use ALL experts
        conflict_threshold=3,  # will be replaced by category method
        export_full_transcripts=True,
    )

    # set conflict detection method
    delphi_config.conflict_method = args.conflict_method

    # run appropriate command
    if args.health_check:
        # health check mode
        healthy = await run_health_check(runtime_config)
        return 0 if healthy else 1
    else:
        # normal execution mode
        print("=" * 60)
        print("DelPHEA-irAKI: Starting irAKI Classification")
        print("-" * 60)
        print(f"Case ID: {args.case_id}")
        print(
            f"Expert Count: {len(config_loader.get_available_expert_ids())} (full panel)"
        )
        print(f"Conflict Detection: {delphi_config.conflict_method}")
        print(f"vLLM Endpoint: {runtime_config.get_vllm_endpoint()}")
        print(f"Model: {runtime_config.model_name}")
        print(f"Data Source: {'Real' if runtime_config.use_real_data else 'Dummy'}")
        print("=" * 60)

        return await run_delphi_consensus(
            case_id=args.case_id,
            runtime_config=runtime_config,
            delphi_config=delphi_config,
        )


if __name__ == "__main__":
    # run async main and exit with proper code
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
