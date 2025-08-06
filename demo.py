"""
Clean DelPHEA-irAKI Demo
===============================

Simple, professional demo script for the cleaned DelPHEA system.
"""

import os
import subprocess
import sys


def check_prerequisites():
    """Check if required files exist - fail fast"""
    required_files = [
        "delphea.py",
        "config/panel.json",
        "config/questionnaire.json",
        "prompts/iraki_assessment.json",
        "prompts/debate.json",
        "prompts/confidence_instructions.json",
    ]

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print("✗ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        return False

    return True


def run_example(example_type: str, case_id: str = "iraki_case_001"):
    """Run DelPHEA-irAKI example"""

    base_cmd = [sys.executable, "delphea.py", "--case-id", case_id, "--verbose"]

    if example_type == "health-check":
        print("🏥 Running system health check")
        cmd = base_cmd + ["--health-check"]

    elif example_type == "basic":
        print("🔬 Running basic irAKI classification")
        cmd = base_cmd

    elif example_type == "custom-config":
        print("⚙️ Running with custom configuration")
        cmd = base_cmd + [
            "--expert-panel-config",
            "config/panel.json",
            "--questionnaire-config",
            "config/questionnaire.json",
            "--prompts-dir",
            "prompts",
        ]

    else:
        print(f"❌ Unknown example: {example_type}")
        print("Available: health-check, basic, custom-config")
        return False

    print(f"🚀 Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Example completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Example failed with exit code {e.returncode}")
        return False

    except FileNotFoundError:
        print("❌ delphea.py not found in current directory")
        return False


def main():
    """Main demo runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean DelPHEA-irAKI Demo")
    parser.add_argument(
        "--example",
        choices=["health-check", "basic", "custom-config"],
        default="health-check",
        help="Example type to run",
    )
    parser.add_argument("--case-id", default="iraki_case_001", help="Case ID")

    args = parser.parse_args()

    print("🏥 Clean DelPHEA-irAKI Demo")
    print("=" * 50)

    if not check_prerequisites():
        print("\n💥 Prerequisites not met. Exiting.")
        return 1

    print("✅ All prerequisites met")
    print(f"Running example: {args.example}")
    print(f"Case ID: {args.case_id}")
    print()

    success = run_example(args.example, args.case_id)

    if success:
        print("\n🎉 Demo completed successfully!")
    else:
        print("\n💥 Demo failed.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
