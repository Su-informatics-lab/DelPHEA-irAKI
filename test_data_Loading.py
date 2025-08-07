#!/usr/bin/env python3
"""
Comprehensive test script for data loading functionality.
Tests both real and dummy data loading capabilities.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config.core import RuntimeConfig
from data.dummy_patient_generator import DummyPatientGenerator
from data.patient_loader_wrapper import PatientLoaderWrapper


def test_dummy_data_generation():
    """Test dummy data generation with different scenarios."""
    print("\n" + "=" * 60)
    print("TEST 1: Dummy Data Generation")
    print("=" * 60)

    scenarios = ["classic_iraki", "atn_case"]

    for scenario in scenarios:
        print(f"\n📝 Testing scenario: {scenario}")
        case = DummyPatientGenerator.generate_patient_case("test_001", scenario)

        print(f"  - Case ID: {case['case_id']}")
        print(f"  - Patient Age: {case['patient_info']['age']}")
        print(f"  - Summary: {case['patient_summary'][:80]}...")
        print(f"  - Total Notes: {case['total_notes']}")
        print(f"  - Medications: {list(case.get('medication_history', {}).keys())}")
        print(f"  - Lab values present: {bool(case.get('lab_values'))}")

        assert case["case_id"] == "test_001"
        assert case["total_notes"] > 0
        print(f"  ✅ Scenario '{scenario}' passed")

    return True


def test_real_data_loading():
    """Test real data loading if available."""
    print("\n" + "=" * 60)
    print("TEST 2: Real Data Loading")
    print("=" * 60)

    config = RuntimeConfig(data_dir="irAKI_data", use_real_data=True)

    loader = PatientLoaderWrapper(config)

    print(f"\n📊 Data source: {loader.get_data_source()}")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    # Get available patients
    patients = loader.get_available_patients(5)
    print(f"\n👥 Available patients (first 5): {patients}")

    # Load a patient if available
    if patients:
        case_id = patients[0]
        print(f"\n📋 Loading patient: {case_id}")
        case = loader.load_patient_case(case_id)

        print(f"  - Patient ID: {case['patient_info']['person_id']}")
        print(f"  - Age: {case['patient_info']['age']}")
        print(f"  - Gender: {case['patient_info']['gender']}")
        print(f"  - Total Notes: {case['total_notes']}")
        print(f"  - Data Completeness: {case.get('data_completeness', 'N/A')}")

        if case["clinical_notes"]:
            first_note = case["clinical_notes"][0]
            print(f"  - First Note Service: {first_note.get('service', 'Unknown')}")
            print(f"  - First Note Length: {len(first_note.get('text', ''))}")

    return True


def test_fallback_behavior():
    """Test fallback from real to dummy data."""
    print("\n" + "=" * 60)
    print("TEST 3: Fallback Behavior")
    print("=" * 60)

    # Test with non-existent data directory
    config = RuntimeConfig(data_dir="non_existent_directory", use_real_data=True)

    loader = PatientLoaderWrapper(config)

    print(f"\n🔄 Data source (should be DUMMY): {loader.get_data_source()}")
    assert loader.get_data_source() == "DUMMY", "Should fallback to DUMMY"

    # Should still be able to load data
    case = loader.load_patient_case("iraki_case_999")
    print(f"  - Loaded case: {case['case_id']}")
    print(f"  - Has patient info: {bool(case.get('patient_info'))}")
    print(f"  - Has clinical notes: {bool(case.get('clinical_notes'))}")

    assert case["case_id"] == "iraki_case_999"
    print("  ✅ Fallback behavior working correctly")

    return True


def test_wrapper_compatibility():
    """Test backward compatibility with DataLoaderWrapper."""
    print("\n" + "=" * 60)
    print("TEST 4: Backward Compatibility")
    print("=" * 60)

    # Import using old name
    from data.loader import DataLoaderWrapper

    config = RuntimeConfig(use_real_data=False)
    wrapper = DataLoaderWrapper(config)

    print(f"\n🔧 Old import working: {wrapper.__class__.__name__}")

    # Test basic functionality
    patients = wrapper.get_available_patients(3)
    print(f"  - Available patients: {patients}")

    case = wrapper.load_patient_case(patients[0])
    print(f"  - Loaded case: {case['case_id']}")

    assert len(patients) == 3
    assert case["case_id"] == patients[0]
    print("  ✅ Backward compatibility maintained")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("RUNNING COMPREHENSIVE DATA LOADING TESTS")
    print("🚀 " * 20)

    tests = [
        ("Dummy Data Generation", test_dummy_data_generation),
        ("Real Data Loading", test_real_data_loading),
        ("Fallback Behavior", test_fallback_behavior),
        ("Backward Compatibility", test_wrapper_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASSED"))
        except Exception as e:
            logger.error(f"Test '{test_name}' failed: {e}")
            results.append((test_name, f"❌ FAILED: {e}"))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        print(f"  {test_name}: {result}")

    # Overall result
    all_passed = all("PASSED" in r for _, r in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
