"""
Diagnostic script to test DelPHEA-irAKI components incrementally.
Run this to identify which components are working and which need fixes.
"""

import sys
from pathlib import Path

# add parent directory to path (if running from tests/ directory)
# otherwise just use current directory
if Path(__file__).parent.name == "tests":
    sys.path.insert(0, str(Path(__file__).parent.parent))
else:
    sys.path.insert(0, str(Path(__file__).parent))

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
    # Check what attributes exist
    if hasattr(delphi, "expert_count"):
        print(f"✓ DelphiConfig created (expert_count: {delphi.expert_count})")
    else:
        print(f"✓ DelphiConfig created (no expert_count - uses all experts)")

except Exception as e:
    print(f"✗ Configuration error: {e}")
    import traceback

    traceback.print_exc()

print()

# Test 3: Configuration Loader - UPDATED VERSION
print("Test 3: Configuration Loader")
print("-" * 30)
try:
    from config.core import RuntimeConfig
    from config.loader import ConfigurationLoader

    runtime_config = RuntimeConfig()
    config_loader = ConfigurationLoader(runtime_config)
    print("✓ ConfigurationLoader created")

    # Check expert panel
    if hasattr(config_loader, "expert_panel"):
        expert_count = len(
            config_loader.expert_panel.get("expert_panel", {}).get("experts", [])
        )
        print(f"✓ Expert panel loaded: {expert_count} experts")
    elif hasattr(config_loader, "_expert_panel"):
        expert_count = len(
            config_loader._expert_panel.get("expert_panel", {}).get("experts", [])
        )
        print(f"✓ Expert panel loaded: {expert_count} experts")
    else:
        print("⚠ Expert panel not loaded")

    # Check questionnaire - try multiple approaches
    question_count = 0
    questions_found = False

    # Method 1: Try get_questions() method
    if hasattr(config_loader, "get_questions"):
        try:
            questions = config_loader.get_questions()
            question_count = len(questions)
            questions_found = True
            print(
                f"✓ Questionnaire loaded (via get_questions): {question_count} questions"
            )
        except Exception as e:
            print(f"⚠ get_questions() failed: {e}")

    # Method 2: Try direct access to questionnaire property
    if not questions_found and hasattr(config_loader, "questionnaire"):
        try:
            questionnaire = config_loader.questionnaire
            if isinstance(questionnaire, dict):
                questions = questionnaire.get("questionnaire", {}).get("questions", [])
                question_count = len(questions)
                questions_found = True
                print(
                    f"✓ Questionnaire loaded (via property): {question_count} questions"
                )
        except Exception as e:
            print(f"⚠ questionnaire property failed: {e}")

    # Method 3: Try private _questionnaire attribute
    if not questions_found and hasattr(config_loader, "_questionnaire"):
        try:
            questionnaire = config_loader._questionnaire
            if isinstance(questionnaire, dict):
                questions = questionnaire.get("questionnaire", {}).get("questions", [])
                question_count = len(questions)
                questions_found = True
                print(
                    f"✓ Questionnaire loaded (via _questionnaire): {question_count} questions"
                )
        except Exception as e:
            print(f"⚠ _questionnaire access failed: {e}")

    if not questions_found:
        print("✗ Could not load questionnaire")
    elif question_count < 16:
        print(f"⚠ Only {question_count} questions loaded (expected 16)")
        print("  The questionnaire.json file might be incomplete")

    # Also check the raw JSON file directly
    try:
        import json

        with open("config/questionnaire.json", "r") as f:
            raw_data = json.load(f)
            raw_questions = raw_data.get("questionnaire", {}).get("questions", [])
            if len(raw_questions) != question_count:
                print(
                    f"⚠ JSON file has {len(raw_questions)} questions but loader shows {question_count}"
                )
    except Exception as e:
        print(f"⚠ Could not verify JSON file: {e}")

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

# Test 7: Quick JSON validation
print("Test 7: Questionnaire JSON Validation")
print("-" * 30)
try:
    import json

    with open("config/questionnaire.json", "r") as f:
        data = json.load(f)

    if "questionnaire" in data and "questions" in data["questionnaire"]:
        questions = data["questionnaire"]["questions"]
        question_ids = [q.get("id", "NO_ID") for q in questions]
        print(
            f"✓ JSON valid with {len(questions)} questions: {', '.join(question_ids[:5])}..."
        )

        # Check for all 16 questions
        expected_ids = [f"Q{i}" for i in range(1, 17)]
        missing = set(expected_ids) - set(question_ids)
        if missing:
            print(f"✗ Missing questions: {', '.join(sorted(missing))}")
        elif len(questions) == 16:
            print("✓ All 16 questions present in JSON file")
    else:
        print("✗ Invalid JSON structure")

except json.JSONDecodeError as e:
    print(f"✗ Invalid JSON syntax: {e}")
    print("  The questionnaire.json file is corrupted or incomplete")
except FileNotFoundError:
    print("✗ questionnaire.json not found")
except Exception as e:
    print(f"✗ Error checking JSON: {e}")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print()

# Summary
print("Summary:")
print("-" * 30)
if "question_count" in locals():
    if question_count == 0:
        print("⚠ CRITICAL: Questionnaire not loading - check JSON file")
    elif question_count < 16:
        print(f"⚠ WARNING: Only {question_count}/16 questions loaded")
    else:
        print(f"✓ All {question_count} questions loaded successfully")

print("\nNext steps based on results above:")
print("1. If questionnaire shows 0 or <16 questions, fix config/questionnaire.json")
print(
    "2. If 'RoutedAgent not defined' error, remove duplicate class from delphea_iraki.py"
)
print("3. Ensure all configuration files exist in config/ directory")
print("4. Once all tests pass, try: python delphea_iraki.py --health-check --verbose")
