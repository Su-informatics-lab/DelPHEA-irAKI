"""
Example runner for DelPHEA-irAKI system
=======================================

This script demonstrates how to run the irAKI classification system
with different configurations and options.

Usage:
    python run_iraki_example.py --example basic
    python run_iraki_example.py --example with-literature
    python run_iraki_example.py --example custom-panel
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

def run_example(example_type: str, case_id: str = "iraki_case_001"):
    """Run DelPHEA-irAKI with different example configurations"""

    base_cmd = [
        sys.executable, "delphea_iraki.py",
        "--case-id", case_id,
        "--verbose"
    ]

    if example_type == "basic":
        print("🔬 Running basic irAKI classification (no literature search)")
        cmd = base_cmd + [
            "--expert-panel-config", "config/expert_panel.json",
            "--questionnaire-config", "config/questionnaire_iraki.json",
            "--output-dir", "output/basic_example"
        ]

    elif example_type == "with-literature":
        print("📚 Running irAKI classification with literature search")
        cmd = base_cmd + [
            "--expert-panel-config", "config/expert_panel.json",
            "--questionnaire-config", "config/questionnaire_iraki.json",
            "--enable-literature-search",
            "--max-literature-results", "3",
            "--literature-recent-years", "3",
            "--output-dir", "output/literature_example"
        ]

    elif example_type == "custom-panel":
        print("👥 Running irAKI classification with custom expert panel")
        # Create custom panel configuration
        create_custom_panel_config()
        cmd = base_cmd + [
            "--expert-panel-config", "config/expert_panel_custom.json",
            "--questionnaire-config", "config/questionnaire_iraki.json",
            "--output-dir", "output/custom_panel_example"
        ]

    elif example_type == "health-check":
        print("🏥 Running health check")
        cmd = base_cmd + [
            "--health-check",
            "--expert-panel-config", "config/expert_panel.json",
            "--questionnaire-config", "config/questionnaire_iraki.json"
        ]

    else:
        print(f"❌ Unknown example type: {example_type}")
        print("Available examples: basic, with-literature, custom-panel, health-check")
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
        print("❌ DelPHEA-irAKI script not found. Make sure delphea_iraki.py is in the current directory.")
        return False

def create_custom_panel_config():
    """Create a custom expert panel configuration with fewer experts"""
    custom_config = {
        "expert_panel": {
            "description": "Custom focused expert panel for irAKI classification",
            "version": "1.0",
            "experts": [
                {
                    "id": "oncologist_lead",
                    "specialty": "medical_oncology",
                    "name": "Dr. Sarah Chen",
                    "experience_years": 15,
                    "credentials": "MD, PhD, FACP",
                    "institution": "Comprehensive Cancer Center",
                    "expertise": [
                        "immune checkpoint inhibitor toxicities",
                        "immune-related adverse events (irAEs)",
                        "cancer immunotherapy"
                    ],
                    "focus_areas": [
                        "temporal relationship between immunotherapy and AKI",
                        "other irAE assessment and correlation",
                        "immunotherapy dosing and scheduling effects"
                    ],
                    "clinical_experience": "Managed >2000 patients on immune checkpoint inhibitors",
                    "reasoning_style": "evidence-based, protocol-driven, considers long-term cancer outcomes"
                },
                {
                    "id": "nephrologist_lead",
                    "specialty": "nephrology",
                    "name": "Dr. Robert Martinez",
                    "experience_years": 18,
                    "credentials": "MD, FASN",
                    "institution": "Academic Medical Center",
                    "expertise": [
                        "acute kidney injury",
                        "drug-induced nephrotoxicity",
                        "immune-mediated kidney disease"
                    ],
                    "focus_areas": [
                        "AKI differential diagnosis",
                        "urinalysis interpretation",
                        "kidney biopsy indications"
                    ],
                    "clinical_experience": "Kidney biopsy interpretation, drug-induced AKI specialist",
                    "reasoning_style": "systematic exclusion of causes, biopsy-correlation focused"
                },
                {
                    "id": "pathologist_lead",
                    "specialty": "renal_pathology",
                    "name": "Dr. Maria Rodriguez",
                    "experience_years": 20,
                    "credentials": "MD, PhD",
                    "institution": "University Medical Center",
                    "expertise": [
                        "renal pathology",
                        "acute tubulo-interstitial nephritis",
                        "immune-mediated kidney pathology"
                    ],
                    "focus_areas": [
                        "histologic patterns of drug-induced AKI",
                        "immune infiltrate characterization",
                        "pathologic-clinical correlation"
                    ],
                    "clinical_experience": "Interpreted >5000 kidney biopsies, drug-induced pattern specialist",
                    "reasoning_style": "morphology-based, pattern recognition, tissue-clinical correlation"
                },
                {
                    "id": "pharmacist_lead",
                    "specialty": "clinical_pharmacy",
                    "name": "Dr. James Wilson",
                    "experience_years": 12,
                    "credentials": "PharmD, BCOP, BCCCP",
                    "institution": "Comprehensive Cancer Center",
                    "expertise": [
                        "oncology pharmacotherapy",
                        "drug interactions",
                        "adverse drug reactions"
                    ],
                    "focus_areas": [
                        "drug-drug interactions affecting nephrotoxicity",
                        "concomitant medication assessment",
                        "temporal drug exposure analysis"
                    ],
                    "clinical_experience": "Oncology pharmacy specialist, medication safety expert",
                    "reasoning_style": "pharmacologic mechanism-focused, drug timeline analysis"
                }
            ]
        },
        "panel_composition": {
            "primary_experts": [
                "oncologist_lead",
                "nephrologist_lead",
                "pathologist_lead"
            ],
            "supporting_experts": [
                "pharmacist_lead"
            ]
        }
    }

    # Create config directory if it doesn't exist
    Path("config").mkdir(exist_ok=True)

    # Write custom configuration
    import json
    with open("config/expert_panel_custom.json", "w") as f:
        json.dump(custom_config, f, indent=2)

    print("📝 Created custom expert panel configuration: config/expert_panel_custom.json")

def check_prerequisites():
    """Check if required files exist"""
    required_files = [
        "delphea_iraki.py",
        "config/expert_panel.json",
        "config/questionnaire_iraki.json"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all configuration files are present before running examples.")
        return False

    return True

def main():
    """Main example runner"""
    import argparse

    parser = argparse.ArgumentParser(description="DelPHEA-irAKI Example Runner")
    parser.add_argument("--example", choices=["basic", "with-literature", "custom-panel", "health-check"],
                        default="basic", help="Example type to run")
    parser.add_argument("--case-id", default="iraki_case_001", help="Case ID for analysis")
    parser.add_argument("--check-prereqs", action="store_true", help="Check prerequisites only")

    args = parser.parse_args()

    print("🏥 DelPHEA-irAKI Example Runner")
    print("=" * 50)

    if args.check_prereqs:
        if check_prerequisites():
            print("✅ All prerequisites met!")
        return

    if not check_prerequisites():
        return

    print(f"Running example: {args.example}")
    print(f"Case ID: {args.case_id}")
    print()

    success = run_example(args.example, args.case_id)

    if success:
        print()
        print("🎉 Example completed successfully!")
        print("📁 Check the output directory for results and transcripts")
        print("📋 Review the generated files for human expert validation")
    else:
        print()
        print("💥 Example failed. Check the error messages above.")

if __name__ == "__main__":
    main()
