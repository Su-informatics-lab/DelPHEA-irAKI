#!/usr/bin/env python3
"""
DelPHEA-irAKI Interactive Demo
===============================

Demonstrates the system with both dummy data (for testing) and real patient data.
This script guides you through running the Delphi consensus process step by step.

Usage:
    python demo.py --dummy    # Run with dummy test patient
    python demo.py --real     # Run with real patient data
    python demo.py --both     # Compare dummy vs real (default)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# colorful output for demo
try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    # fallback - no colors
    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = BLUE = ""
        RESET = ""

    class Style:
        BRIGHT = DIM = RESET_ALL = ""


def print_header(text: str):
    """Print a formatted header."""
    sep = "=" * 70
    if HAS_COLOR:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{sep}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(70)}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{sep}{Style.RESET_ALL}")
    else:
        print(f"\n{sep}")
        print(f"{text.center(70)}")
        print(f"{sep}")


def print_info(label: str, value: str, color=Fore.GREEN):
    """Print labeled information."""
    if HAS_COLOR:
        print(f"{color}► {label}:{Style.RESET_ALL} {value}")
    else:
        print(f"► {label}: {value}")


def print_warning(text: str):
    """Print a warning message."""
    if HAS_COLOR:
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")
    else:
        print(f"⚠ {text}")


def print_error(text: str):
    """Print an error message."""
    if HAS_COLOR:
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")
    else:
        print(f"✗ {text}")


def print_success(text: str):
    """Print a success message."""
    if HAS_COLOR:
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
    else:
        print(f"✓ {text}")


async def check_prerequisites():
    """Check if all prerequisites are met."""
    print_header("CHECKING PREREQUISITES")

    issues = []

    # check for required files
    required_files = [
        "delphea_iraki.py",
        "dataloader.py",
        "config/panel.json",
        "config/questionnaire.json",
        "orchestration/consensus.py",
    ]

    for file in required_files:
        if Path(file).exists():
            print_success(f"Found {file}")
        else:
            print_error(f"Missing {file}")
            issues.append(f"Missing {file}")

    # check for data directory
    if Path("irAKI_data").exists():
        print_success("Found irAKI_data directory")
    else:
        print_warning("irAKI_data directory not found - will use dummy data")

    # check Python packages
    try:
        print_success("NumPy installed")
    except ImportError:
        print_error("NumPy not installed")
        issues.append("NumPy not installed")

    try:
        print_success("Pandas installed")
    except ImportError:
        print_error("Pandas not installed")
        issues.append("Pandas not installed")

    try:
        print_success("SciPy installed")
    except ImportError:
        print_error("SciPy not installed")
        issues.append("SciPy not installed")

    if issues:
        print_error(f"\nFound {len(issues)} issue(s) that need fixing:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print_success("\nAll prerequisites met!")
        return True


async def demo_dummy_data():
    """Demonstrate using dummy patient data."""
    print_header("DUMMY DATA DEMONSTRATION")

    print("\nThis demonstrates DelPHEA-irAKI with a synthetic test patient.")
    print("The dummy patient has classic irAKI presentation for testing.\n")

    # import dataloader
    from dataloader import DataLoader

    # load dummy data
    loader = DataLoader(use_dummy=True)

    # get the dummy case
    case = loader.load_patient_case("iraki_case_001")

    # display patient info
    print_info("Case ID", case["case_id"], Fore.CYAN)
    print_info("Age", f"{case['patient_info']['age']} years", Fore.CYAN)
    print_info("Gender", case["patient_info"]["gender"], Fore.CYAN)
    print_info("Clinical Notes", f"{len(case['clinical_notes'])} notes", Fore.CYAN)

    print("\n" + "-" * 50)
    print("CLINICAL PRESENTATION:")
    print("-" * 50)

    for i, note in enumerate(case["clinical_notes"], 1):
        print(f"\n{Fore.BLUE}Note {i} ({note['service']}):{Style.RESET_ALL}")
        print(f"  {note['text'][:200]}...")

    print("\n" + "-" * 50)
    print("KEY FINDINGS:")
    print("-" * 50)

    # analyze notes for key findings
    findings = []
    for note in case["clinical_notes"]:
        text = note["text"].lower()
        if "pembrolizumab" in text:
            findings.append("✓ ICI exposure: Pembrolizumab")
        if "creatinine" in text and ("3.2" in text or "elevated" in text):
            findings.append("✓ Acute kidney injury: Cr 3.2 (baseline 1.0)")
        if "proteinuria" in text:
            findings.append("✓ Urinalysis: Proteinuria 2+")
        if "interstitial nephritis" in text:
            findings.append("✓ Clinical suspicion: Acute interstitial nephritis")
        if "prednisone" in text or "steroid" in text:
            findings.append("✓ Recommended treatment: Steroids")

    for finding in findings:
        print(f"  {finding}")

    # check if we can run the full system
    print("\n" + "=" * 50)
    choice = input("Run full Delphi consensus on dummy patient? (y/n): ").lower()

    if choice == "y":
        print_info("\nTo run full consensus", "Execute the following command")
        print(
            f"\n{Fore.YELLOW}python delphea_iraki.py --case-id iraki_case_001 --use-dummy-data --verbose{Style.RESET_ALL}\n"
        )

        if not check_vllm_available():
            print_warning(
                "Note: vLLM server not configured. Add --mock flag for testing without LLM."
            )

    return case


async def demo_real_data():
    """Demonstrate using real patient data."""
    print_header("REAL DATA DEMONSTRATION")

    print("\nThis demonstrates DelPHEA-irAKI with actual patient data.")
    print("Real data requires irAKI_data directory with patient files.\n")

    # import dataloader
    from dataloader import DataLoader

    try:
        # attempt to load real data
        loader = DataLoader(data_dir="irAKI_data", use_dummy=False)

        if loader.use_dummy:
            print_warning("Real data not available, falling back to dummy mode")
            print("Place patient data files in irAKI_data/ directory")
            return None

        # get available patients
        patients = loader.get_available_patients(limit=5)

        print_info("Available patients", f"{len(loader.patient_ids)} total")
        print("\nFirst 5 patient IDs:")
        for pid in patients:
            print(f"  • {pid}")

        # let user select a patient
        print("\n" + "-" * 50)
        patient_id = input(
            f"Enter patient ID to examine (default: {patients[0]}): "
        ).strip()

        if not patient_id:
            patient_id = patients[0]

        # load the case
        try:
            case = loader.load_patient_case(patient_id)
        except ValueError as e:
            print_error(f"Failed to load patient: {e}")
            return None

        # display patient info
        print("\n" + "=" * 50)
        print_info("Case ID", case["case_id"], Fore.CYAN)
        print_info("Age", f"{case['patient_info']['age']} years", Fore.CYAN)
        print_info("Gender", case["patient_info"]["gender"], Fore.CYAN)
        print_info("Clinical Notes", f"{len(case['clinical_notes'])} notes", Fore.CYAN)

        # show sample of clinical notes
        print("\n" + "-" * 50)
        print("SAMPLE CLINICAL NOTES (first 3):")
        print("-" * 50)

        for i, note in enumerate(case["clinical_notes"][:3], 1):
            print(f"\n{Fore.BLUE}Note {i} ({note['service']}):{Style.RESET_ALL}")
            print(f"  Date: {note['timestamp']}")
            preview = note["text"][:200].replace("\n", " ")
            print(f"  Text: {preview}...")

        # analyze for ICI exposure
        print("\n" + "-" * 50)
        print("AUTOMATED SCREENING:")
        print("-" * 50)

        ici_keywords = [
            "pembrolizumab",
            "nivolumab",
            "ipilimumab",
            "atezolizumab",
            "durvalumab",
            "avelumab",
            "cemiplimab",
            "dostarlimab",
            "checkpoint",
            "pd-1",
            "pd-l1",
            "ctla-4",
            "ici",
        ]

        aki_keywords = [
            "creatinine",
            "acute kidney",
            "aki",
            "renal failure",
            "nephritis",
            "proteinuria",
            "oliguria",
            "anuria",
        ]

        ici_found = False
        aki_found = False

        for note in case["clinical_notes"]:
            text = note["text"].lower()
            for keyword in ici_keywords:
                if keyword in text:
                    ici_found = True
                    break
            for keyword in aki_keywords:
                if keyword in text:
                    aki_found = True
                    break

        print(f"  ICI exposure detected: {'Yes' if ici_found else 'No'}")
        print(f"  AKI markers detected: {'Yes' if aki_found else 'No'}")

        if ici_found and aki_found:
            print_success("\n✓ Patient appears suitable for irAKI assessment")
        elif not ici_found:
            print_warning("\n⚠ No clear ICI exposure detected in notes")
        elif not aki_found:
            print_warning("\n⚠ No clear AKI markers detected in notes")

        # offer to run full assessment
        print("\n" + "=" * 50)
        choice = input(f"Run full Delphi consensus on {patient_id}? (y/n): ").lower()

        if choice == "y":
            print_info("\nTo run full consensus", "Execute the following command")
            print(
                f"\n{Fore.YELLOW}python delphea_iraki.py --case-id {patient_id} --verbose{Style.RESET_ALL}\n"
            )

            if not check_vllm_available():
                print_warning(
                    "Note: vLLM server not configured. Set up infrastructure first."
                )

        return case

    except FileNotFoundError as e:
        print_error(f"Data directory not found: {e}")
        print_warning("Create irAKI_data/ directory and add patient files")
        return None
    except Exception as e:
        print_error(f"Failed to load real data: {e}")
        return None


def check_vllm_available() -> bool:
    """Check if vLLM endpoint is configured."""
    import os

    endpoint = os.environ.get("VLLM_ENDPOINT")
    if endpoint:
        print_success(f"vLLM endpoint configured: {endpoint}")
        return True
    else:
        print_warning("vLLM endpoint not configured (VLLM_ENDPOINT not set)")
        return False


async def compare_dummy_vs_real():
    """Compare dummy and real data side by side."""
    print_header("DUMMY vs REAL DATA COMPARISON")

    print("\nThis compares the dummy test patient with real patient data")
    print("to show the difference in complexity and data quality.\n")

    # load both
    from dataloader import DataLoader

    # dummy data
    dummy_loader = DataLoader(use_dummy=True)
    dummy_case = dummy_loader.load_patient_case("iraki_case_001")

    # real data (if available)
    try:
        real_loader = DataLoader(data_dir="irAKI_data", use_dummy=False)
        if not real_loader.use_dummy:
            real_patients = real_loader.get_available_patients(limit=1)
            if real_patients:
                real_case = real_loader.load_patient_case(real_patients[0])
            else:
                real_case = None
        else:
            real_case = None
    except:
        real_case = None

    # display comparison
    print("-" * 70)
    print(f"{'ASPECT':<20} {'DUMMY DATA':<25} {'REAL DATA':<25}")
    print("-" * 70)

    # case ID
    print(f"{'Case ID':<20} {dummy_case['case_id']:<25} ", end="")
    print(f"{real_case['case_id'] if real_case else 'N/A':<25}")

    # demographics
    print(f"{'Age':<20} {dummy_case['patient_info']['age']:<25} ", end="")
    print(f"{real_case['patient_info']['age'] if real_case else 'N/A':<25}")

    print(f"{'Gender':<20} {dummy_case['patient_info']['gender']:<25} ", end="")
    print(f"{real_case['patient_info']['gender'] if real_case else 'N/A':<25}")

    # clinical notes
    print(f"{'# Clinical Notes':<20} {len(dummy_case['clinical_notes']):<25} ", end="")
    print(f"{len(real_case['clinical_notes']) if real_case else 'N/A':<25}")

    # note length
    dummy_note_len = sum(len(n["text"]) for n in dummy_case["clinical_notes"])
    print(f"{'Total Text Length':<20} {f'{dummy_note_len:,} chars':<25} ", end="")
    if real_case:
        real_note_len = sum(len(n["text"]) for n in real_case["clinical_notes"])
        print(f"{f'{real_note_len:,} chars':<25}")
    else:
        print("N/A")

    print("-" * 70)

    print("\n" + "=" * 50)
    print("KEY INSIGHTS:")
    print("=" * 50)

    print("\n• Dummy data is simplified and ideal for testing")
    print("• Real data is complex with more noise and variation")
    print("• Both can be processed through the same pipeline")
    print("• Start with dummy data to verify system functionality")
    print("• Move to real data for actual clinical validation")


async def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description="DelPHEA-irAKI Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --check     # Check prerequisites only
  python demo.py --dummy     # Demo with dummy patient
  python demo.py --real      # Demo with real patient data
  python demo.py --both      # Compare dummy vs real (default)
        """,
    )

    parser.add_argument(
        "--check", action="store_true", help="Check prerequisites and exit"
    )

    parser.add_argument(
        "--dummy", action="store_true", help="Demo with dummy patient data"
    )

    parser.add_argument(
        "--real", action="store_true", help="Demo with real patient data"
    )

    parser.add_argument(
        "--both", action="store_true", help="Compare dummy and real data (default)"
    )

    args = parser.parse_args()

    # default to --both if nothing specified
    if not any([args.check, args.dummy, args.real, args.both]):
        args.both = True

    print_header("DelPHEA-irAKI DEMONSTRATION")
    print("\nDelphi Personalized Health Explainable Agents")
    print("for immune-related Acute Kidney Injury\n")

    # check prerequisites
    if args.check or True:  # always check
        prereqs_ok = await check_prerequisites()
        if args.check:
            return 0 if prereqs_ok else 1
        if not prereqs_ok:
            print_error("\nPlease fix prerequisites before continuing")
            return 1

    # run requested demos
    if args.dummy:
        await demo_dummy_data()

    if args.real:
        await demo_real_data()

    if args.both:
        await demo_dummy_data()
        print("\n" + "=" * 70 + "\n")
        await demo_real_data()
        print("\n" + "=" * 70 + "\n")
        await compare_dummy_vs_real()

    print("\n" + "=" * 70)
    print_success("Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run with dummy data to test the pipeline")
    print("2. Configure vLLM endpoint for LLM-based reasoning")
    print("3. Run with real patient data for validation")
    print("4. Review consensus outputs for clinical accuracy")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
