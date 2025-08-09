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
    # run irAKI assessment for a case
    python delphea_iraki.py --case-id iraki_case_001

    # health check
    python delphea_iraki.py --health-check

    # verbose mode
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
from typing import Dict, Optional

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

# -----------------------
# Logging configuration
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

for n in ("autogen_core", "autogen_core.events"):
    logging.getLogger(n).setLevel(logging.WARNING)


class _Trunc(logging.Filter):
    def __init__(self, limit=300):
        self.limit = limit

    def filter(self, record):
        if isinstance(record.msg, str) and len(record.msg) > self.limit:
            record.msg = record.msg[: self.limit] + "... [truncated]"
        return True


for h in logging.getLogger().handlers:
    h.addFilter(_Trunc(300))


# -----------------------
# Helpers (CLI summary & export loading)
# -----------------------
def print_cli_summary(case_id: str) -> None:
    import glob
    import json
    import os

    pattern = f"iraki_consensus_{case_id}_*.json"
    paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not paths:
        print("\nNo export file found for CLI summary.")
        print(f"Looked for: {pattern}")
        return

    path = paths[0]
    with open(path, "r") as f:
        data = json.load(f)

    fc = data.get("final_consensus", {})
    p = fc.get("p_iraki")
    ci = fc.get("ci_95") or [None, None]
    conf = fc.get("confidence")
    maj = data.get("majority_verdict", None)

    r3 = [r for r in data.get("expert_assessments", []) if r.get("round") == "round3"]
    r1_by_expert = {
        r["expert_id"]: r
        for r in data.get("expert_assessments", [])
        if r.get("round") == "round1"
    }

    # Prefer thresholded verdict if present; else fall back to llm verdict
    votes = sum(1 for r in r3 if r.get("verdict_from_prob", r.get("verdict_llm")))
    total = len(r3)
    label = "irAKI" if maj else "Other AKI"

    print("\n==================== irAKI CONSENSUS SUMMARY ====================")
    print(f"Case: {case_id}")
    if p is not None and ci[0] is not None and ci[1] is not None:
        print(f"Pooled P(irAKI): {p:.3f}   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    if conf is not None:
        print(f"Consensus confidence: {conf:.3f}")
    if total:
        print(f"Majority vote: {label} ({votes}/{total})")
    print("\nPer-expert Round 3:")
    for r in r3:
        ex = r["expert_id"]
        p3 = r.get("p_iraki")
        c3 = r.get("confidence")
        v = r.get("verdict_from_prob", r.get("verdict_llm"))
        # compute delta vs R1 if we have it
        d_p = None
        if ex in r1_by_expert and isinstance(p3, (int, float)):
            p1 = r1_by_expert[ex].get("p_iraki")
            if isinstance(p1, (int, float)):
                d_p = p3 - p1

        dp_str = ""
        if isinstance(d_p, (int, float)):
            sign = "+" if d_p >= 0 else ""
            dp_str = f" ({sign}{d_p:.2f} vs R1)"

        p_str = f"{p3:.2f}" if isinstance(p3, (int, float)) else "NA"
        c_str = f"{c3:.2f}" if isinstance(c3, (int, float)) else "NA"
        v_str = "irAKI" if v else "Other"

        print(f"  - {ex:<12} p={p_str}{dp_str}  conf={c_str}  verdict={v_str}")

    print("================================================================\n")
    print(f"Loaded export: {path}")


def load_latest_export(case_id: str) -> Optional[Dict]:
    import glob
    import json
    import os

    paths = sorted(
        glob.glob(f"iraki_consensus_{case_id}_*.json"),
        key=os.path.getmtime,
        reverse=True,
    )
    if not paths:
        return None
    with open(paths[0], "r") as f:
        return json.load(f)


# -----------------------
# Health check
# -----------------------
async def run_health_check(runtime_config: RuntimeConfig) -> bool:
    """Run system health check."""
    logger = logging.getLogger("health_check")
    logger.info("Starting health check...")

    checks_passed = True

    # check 1: configuration files
    try:
        _ = ConfigurationLoader(runtime_config)
        logger.info("✓ Configuration files loaded")
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        checks_passed = False

    # check 2: data loader
    try:
        data_loader = DataLoader(
            data_dir=runtime_config.data_dir, use_dummy=not runtime_config.use_real_data
        )
        available_patients = data_loader.get_available_patients(limit=1)
        if available_patients:
            logger.info(
                f"✓ Data loader initialized ({len(available_patients)} patients available)"
            )
        else:
            logger.warning("⚠ No patient data available")
    except Exception as e:
        logger.error(f"✗ Data loader initialization failed: {e}")
        checks_passed = False

    # check 3: vLLM endpoint (if not local)
    if runtime_config.infrastructure.endpoint_type != "local":
        endpoint = runtime_config.get_vllm_endpoint()
        if endpoint:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{endpoint}/health", timeout=5.0)
                    if response.status_code == 200:
                        logger.info(f"✓ vLLM endpoint healthy: {endpoint}")
                    else:
                        logger.error(
                            f"✗ vLLM endpoint unhealthy: {response.status_code}"
                        )
                        checks_passed = False
            except Exception as e:
                logger.error(f"✗ vLLM endpoint unreachable: {e}")
                checks_passed = False
        else:
            logger.warning("⚠ No vLLM endpoint configured")
    else:
        logger.info("✓ Local model mode - no endpoint check needed")

    return checks_passed


# -----------------------
# Main assessment runner
# -----------------------
async def run_iraki_assessment(case_id: str, runtime_config: RuntimeConfig) -> Dict:
    """Run irAKI assessment for a patient case."""
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
    patient_case = data_loader.load_patient_case(case_id)
    logger.info(
        f"Loaded case: {patient_case['case_id']}, "
        f"data preview: {str(patient_case)[:200]}..."
    )

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
    moderator_id = AgentId(type="moderator", key=case_id)
    await irAKIModeratorAgent.register(
        runtime,
        "moderator",
        lambda: irAKIModeratorAgent(
            case_id=case_id,
            config_loader=config_loader,
            data_loader=data_loader,
            delphi_config=delphi_config,
        ),
    )

    # register expert agents (factory lambda per expert)
    expert_configs = config_loader.expert_panel["expert_panel"]["experts"]
    for expert_config in expert_configs:
        agent_type = f"expert_{expert_config['id']}"  # unique per expert
        await irAKIExpertAgent.register(
            runtime,
            agent_type,
            lambda ec=expert_config: irAKIExpertAgent(
                expert_id=ec["id"],
                case_id=case_id,
                config_loader=config_loader,
                vllm_client=llm_client,
                runtime_config=runtime_config,
            ),
        )
        logger.debug("registered agent %s for case %s", agent_type, case_id)

    logger.info(
        "Registered %d expert agents: %s",
        len(expert_configs),
        [e["id"] for e in expert_configs],
    )

    # start the runtime
    runtime.start()
    logger.info("Agent runtime started")

    # start assessment by sending case to moderator
    await runtime.send_message(
        StartCase(
            case_id=case_id,
            patient_data=patient_case,
            expert_ids=[e["id"] for e in expert_configs],
        ),
        moderator_id,
    )
    logger.info("StartCase sent to moderator")

    # run until completion
    await runtime.stop_when_idle()
    logger.info("Runtime idle; assessment finished pipeline execution")

    # clean shutdown of all agents/resources
    await runtime.close()
    logger.info("Runtime closed; all agents cleaned up")

    # Print CLI summary (safe)
    try:
        print_cli_summary(case_id)
    except Exception as e:
        print(f"\n[CLI summary] Skipped due to error: {e}")

    # Return real results from the latest export (no more placeholder)
    export = load_latest_export(case_id)
    results: Dict = {"case_id": case_id, "status": "completed"}
    if export:
        results["final_consensus"] = export.get("final_consensus")
        results["majority_verdict"] = export.get("majority_verdict")
    else:
        results["note"] = "No export found; summary unavailable."
    return results


# -----------------------
# CLI
# -----------------------
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
        default="http://tempest-gpu021:8000",
        help="VLLM server endpoint (default: http://tempest-gpu021:8000)",
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

    # import here to avoid circular imports
    from config.core import InfrastructureConfig

    # create infrastructure config
    infrastructure_config = InfrastructureConfig()

    # handle endpoint configuration
    if args.local_model:
        infrastructure_config.endpoint_type = InfrastructureConfig.ENDPOINT_LOCAL
    elif args.vllm_endpoint:
        # heuristic: pick a type based on url; default to local if unknown
        if "tempest" in args.vllm_endpoint:
            infrastructure_config.endpoint_type = InfrastructureConfig.ENDPOINT_TEMPEST
        elif "172.31" in args.vllm_endpoint:
            infrastructure_config.endpoint_type = InfrastructureConfig.ENDPOINT_AWS
            infrastructure_config.aws_endpoint = args.vllm_endpoint
        else:
            infrastructure_config.endpoint_type = InfrastructureConfig.ENDPOINT_LOCAL
            infrastructure_config.local_endpoint = args.vllm_endpoint

    # create runtime configuration
    runtime_config = RuntimeConfig(
        infrastructure=infrastructure_config,
        use_real_data=args.use_real_data,
    )

    # run appropriate command
    if args.health_check:
        print("=" * 70)
        print("DelPHEA-irAKI System Health Check")
        print("=" * 70)

        success = asyncio.run(run_health_check(runtime_config))

        print("=" * 70)
        if success:
            print("✓ Health check PASSED")
            print("\nSystem is ready for:")
            print("1. Mock mode testing: --case-id <ID> --local-model")
            print("2. Real LLM testing: --case-id <ID> --vllm-endpoint <URL>")
            print("\nAvailable patient IDs (first 5):")

            # Show available patients
            try:
                loader = DataLoader(use_dummy=not args.use_real_data)
                patients = loader.get_available_patients(limit=5)
                for pid in patients:
                    print(f"  - {pid}")
            except Exception:
                print("  - iraki_case_001 (dummy)")

            sys.exit(0)
        else:
            print("✗ Health check FAILED")
            print("Check the errors above for details")
            sys.exit(1)

    elif args.case_id:
        try:
            print(f"Starting assessment for case: {args.case_id}")
            print(f"Mode: {'Local/Mock' if args.local_model else 'vLLM'}")
            print(f"Data: {'Real' if args.use_real_data else 'Dummy'}")

            results = asyncio.run(run_iraki_assessment(args.case_id, runtime_config))
            print(f"\nAssessment Results:\n{results}")
        except Exception as e:
            logging.error(f"Assessment failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python delphea_iraki.py --health-check")
        print("  python delphea_iraki.py --case-id iraki_case_001 --local-model")
        sys.exit(1)


if __name__ == "__main__":
    main()
