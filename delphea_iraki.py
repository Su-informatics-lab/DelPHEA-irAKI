"""
DelPHEA-irAKI: Delphi Personalized Health Explainable Agents for immune-related AKI
====================================================================================

Main entry point for the DelPHEA-irAKI clinical decision support system.
This orchestrates the multi-agent Delphi consensus process for distinguishing
immune-related AKI from alternative causes in patients receiving ICIs.

Usage:
    python delphea_iraki.py --case-id iraki_case_001
    python delphea_iraki.py --health-check
    python delphea_iraki.py --case-id iraki_case_002 --verbose
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx
from autogen_core import AgentId, SingleThreadedAgentRuntime

# configuration imports
from config.core import DelphiConfig, RuntimeConfig
from config.loader import ConfigurationLoader

# data layer imports
from data.loader import DataLoaderWrapper

# orchestration imports
from orchestration.agents.expert import irAKIExpertAgent
from orchestration.agents.moderator import irAKIModeratorAgent
from orchestration.clients import VLLMClient
from orchestration.messages import StartCase

# configure logging
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
        # initialize configuration loader to validate configs
        config_loader = ConfigurationLoader(runtime_config)

        # check vLLM
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
) -> None:
    """Run the Delphi consensus process for a case.

    Args:
        case_id: Case identifier
        runtime_config: Runtime configuration
        delphi_config: Delphi methodology configuration
    """
    print(f"Starting DelPHEA-irAKI for case: {case_id}")
    print(f"vLLM endpoint: {runtime_config.get_vllm_endpoint()}")
    print(f"Model: {runtime_config.model_name}")

    # initialize configuration (this will fail fast if configs are invalid)
    config_loader = ConfigurationLoader(runtime_config)

    # initialize data loader
    data_loader = DataLoaderWrapper(runtime_config)

    print(f"✓ Loaded {len(config_loader.get_available_expert_ids())} experts")
    print(f"✓ Loaded {len(config_loader.get_questions())} questions")
    print("-" * 60)

    # create runtime
    runtime = SingleThreadedAgentRuntime()

    # create shared HTTP client for all vLLM requests
    shared_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=30.0, read=runtime_config.timeout, write=10.0, pool=5.0
        )
    )

    # create shared VLLMClient
    shared_vllm_client = VLLMClient(runtime_config, shared_http_client)

    # register moderator agent
    await runtime.register(
        "Moderator",
        lambda _: irAKIModeratorAgent(
            case_id, config_loader, data_loader, delphi_config
        ),
    )

    # register expert agents with shared VLLMClient
    expert_ids = config_loader.get_available_expert_ids()[: delphi_config.expert_count]
    for expert_id in expert_ids:
        await runtime.register(
            f"Expert_{expert_id}",
            lambda _, eid=expert_id: irAKIExpertAgent(
                eid, case_id, config_loader, shared_vllm_client
            ),
        )

    # start runtime
    await runtime.start()

    # bootstrap process
    await runtime.send_message(
        StartCase(case_id=case_id), AgentId("Moderator", case_id)
    )

    # wait for completion
    await runtime.stop_when_idle()

    # clean up shared resources
    await shared_vllm_client.close()
    await shared_http_client.aclose()
    await runtime.stop()

    print("\n✓ DelPHEA-irAKI completed successfully!")


async def main():
    """Main entry point for DelPHEA-irAKI system."""
    parser = argparse.ArgumentParser(
        description="DelPHEA-irAKI: Clinical irAKI Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Health check:
    python delphea_iraki.py --health-check
    
  Run classification:
    python delphea_iraki.py --case-id iraki_case_001
    
  Custom configuration:
    python delphea_iraki.py --case-id iraki_case_001 \\
        --expert-panel-config config/custom_panel.json \\
        --model-name meta-llama/Llama-3.1-70B-Instruct
        """,
    )

    # core configuration
    parser.add_argument(
        "--case-id", default="iraki_case_001", help="irAKI case identifier"
    )
    parser.add_argument(
        "--expert-panel-config",
        default="config/panel.json",
        help="Expert panel configuration file",
    )
    parser.add_argument(
        "--questionnaire-config",
        default="config/questionnaire.json",
        help="Questionnaire configuration file",
    )
    parser.add_argument("--prompts-dir", default="prompts", help="Prompts directory")

    # vLLM configuration
    parser.add_argument(
        "--vllm-endpoint",
        help="vLLM server endpoint (default: from config)",
    )
    parser.add_argument(
        "--model-name",
        default="openai/gpt-oss-120b",
        help="Model name for inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p for LLM sampling",
    )

    # delphi methodology parameters
    parser.add_argument(
        "--expert-count",
        type=int,
        default=8,
        help="Number of experts to use",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=int,
        default=3,
        help="Score difference threshold for triggering debate",
    )

    # data configuration
    parser.add_argument(
        "--data-dir",
        default="irAKI_data",
        help="Directory containing patient data",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy data instead of real patient data",
    )

    # system options
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # create runtime configuration
    runtime_config = RuntimeConfig(
        expert_panel_config=args.expert_panel_config,
        questionnaire_config=args.questionnaire_config,
        prompts_dir=args.prompts_dir,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        data_dir=args.data_dir,
        use_real_data=not args.use_dummy_data,
    )

    # override endpoint if provided
    if args.vllm_endpoint:
        runtime_config.infrastructure.aws_endpoint = args.vllm_endpoint

    # create delphi configuration
    delphi_config = DelphiConfig(
        expert_count=args.expert_count,
        conflict_threshold=args.conflict_threshold,
    )

    try:
        if args.health_check:
            # run health check
            success = await run_health_check(runtime_config)
            return 0 if success else 1
        else:
            # validate required files exist
            required_files = [
                Path(runtime_config.expert_panel_config),
                Path(runtime_config.questionnaire_config),
                Path(runtime_config.prompts_dir),
            ]

            missing = [f for f in required_files if not f.exists()]
            if missing:
                print("✗ Missing required files:")
                for f in missing:
                    print(f"  - {f}")
                return 1

            # run delphi consensus
            await run_delphi_consensus(args.case_id, runtime_config, delphi_config)
            return 0

    except FileNotFoundError as e:
        print(f"\n✗ Configuration file not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
