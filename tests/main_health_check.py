"""
Diagnostic script to test DelPHEA-irAKI components incrementally.
Run this to identify which components are working and which need fixes.
"""

import sys
from pathlib import Path

# add parent directory to path
sys.path.append(str(Path(__file__).parent))

print("=" * 70)
print("DelPHEA-irAKI DIAGNOSTIC TEST")
print("=" * 70)
print()

# Test 1: Basic imports
print("Test 1: Basic Imports")
print("-" * 30)
try:
    print("✓ pandas imported")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    print("✓ httpx imported")
except ImportError as e:
    print(f"✗ httpx import failed: {e}")

print()

# Test 2: Configuration classes
print("Test 2: Configuration Classes")
print("-" * 30)
try:
    from config.core import DelphiConfig, InfrastructureConfig, RuntimeConfig

    print("✓ Configuration classes imported")

    # try to create instances
    infra = InfrastructureConfig()
    print(f"✓ InfrastructureConfig created (endpoint_type: {infra.endpoint_type})")

    runtime = RuntimeConfig()
    print(f"✓ RuntimeConfig created")

    delphi = DelphiConfig()
    print(f"✓ DelphiConfig created (expert_count: {delphi.expert_count})")

except Exception as e:
    print(f"✗ Configuration error: {e}")
    import traceback

    traceback.print_exc()

print()

# Test 3: Configuration Loader
print("Test 3: Configuration Loader")
print("-" * 30)
try:
    from config.core import RuntimeConfig
    from config.loader import ConfigurationLoader

    runtime_config = RuntimeConfig()
    config_loader = ConfigurationLoader(runtime_config)
    print("✓ ConfigurationLoader created")

    # check if configs loaded
    if hasattr(config_loader, "expert_panel"):
        expert_count = len(
            config_loader.expert_panel.get("expert_panel", {}).get("experts", [])
        )
        print(f"✓ Expert panel loaded: {expert_count} experts")
    else:
        print("⚠ Expert panel not loaded")

    if hasattr(config_loader, "questionnaire"):
        question_count = len(config_loader.questionnaire.get("questions", []))
        print(f"✓ Questionnaire loaded: {question_count} questions")
    else:
        print("⚠ Questionnaire not loaded")

except Exception as e:
    print(f"✗ ConfigurationLoader error: {e}")
    import traceback

    traceback.print_exc()

print()

# Test 4: Data Loader
print("Test 4: Data Loader")
print("-" * 30)
try:
    from dataloader import DataLoader

    # try dummy mode first (safest)
    loader = DataLoader(use_dummy=True)
    print("✓ DataLoader created in dummy mode")

    if loader.is_available():
        print("✓ Dummy data is available")
        patients = loader.get_available_patients()
        print(f"✓ Available patients: {patients}")

        if patients:
            case = loader.load_patient_case(patients[0])
            print(f"✓ Loaded patient case: {case.get('case_id', 'unknown')}")
    else:
        print("✗ Data not available")

except Exception as e:
    print(f"✗ DataLoader error: {e}")
    import traceback

    traceback.print_exc()

print()

# Test 5: Check for required files
print("Test 5: Required Files")
print("-" * 30)
required_files = [
    "config/panel.json",
    "config/questionnaire.json",
    "orchestration/consensus.py",
    "orchestration/messages.py",
]

for filepath in required_files:
    if Path(filepath).exists():
        print(f"✓ Found: {filepath}")
    else:
        print(f"✗ Missing: {filepath}")

print()

# Test 6: Test health check function directly
print("Test 6: Health Check Function")
print("-" * 30)
try:
    import asyncio

    from config.core import RuntimeConfig
    from delphea_iraki import run_health_check

    runtime_config = RuntimeConfig()
    print("✓ RuntimeConfig created for health check")

    # run the health check
    success = asyncio.run(run_health_check(runtime_config))
    if success:
        print("✓ Health check passed")
    else:
        print("✗ Health check failed (returned False)")

except Exception as e:
    print(f"✗ Health check error: {e}")
    import traceback

    traceback.print_exc()

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print()
print("Next steps based on results above:")
print("1. Fix any import errors first")
print("2. Ensure configuration files exist in config/ directory")
print("3. Address any specific component failures")
print("4. Once all tests pass, try: python delphea_iraki.py --health-check --verbose")
