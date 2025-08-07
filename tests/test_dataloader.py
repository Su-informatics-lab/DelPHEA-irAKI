#!/usr/bin/env python3
"""
Test DataLoader functionality.
"""

import sys
from pathlib import Path

# add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataloader import DataLoader


def test_dummy_mode():
    """Test that dummy mode works without real data."""
    print("\n=== Testing Dummy Mode ===")

    loader = DataLoader(use_dummy=True)

    # should be available in dummy mode
    assert loader.is_available(), "Dummy mode should always be available"

    # should return one dummy patient
    patients = loader.get_available_patients()
    assert len(patients) == 1, "Should have exactly one dummy patient"
    assert patients[0] == "iraki_case_001", "Dummy patient ID should be iraki_case_001"

    # load dummy patient
    case = loader.load_patient_case("iraki_case_001")
    assert case["case_id"] == "iraki_case_001"
    assert case["patient_info"]["age"] == 68
    assert case["patient_info"]["gender"] == "Male"
    assert len(case["clinical_notes"]) == 2
    assert "pembrolizumab" in case["clinical_notes"][0]["text"].lower()
    assert "creatinine" in case["clinical_notes"][1]["text"].lower()

    print("✓ Dummy mode working correctly")


def test_real_data_loading():
    """Test loading real data if available."""
    print("\n=== Testing Real Data Loading ===")

    loader = DataLoader(data_dir="irAKI_data")

    if loader.use_dummy:
        print("⚠ Real data not available, fell back to dummy mode")
        return

    # check data is loaded
    assert loader.is_available(), "Data should be available"
    assert len(loader.patient_ids) > 0, "Should have loaded some patients"

    # get available patients
    patients = loader.get_available_patients(limit=5)
    assert len(patients) <= 5, "Should respect limit"
    assert all(
        p.startswith("iraki_case_") for p in patients
    ), "All IDs should have correct format"

    # load first patient
    if patients:
        case = loader.load_patient_case(patients[0])
        assert case["case_id"] == patients[0]
        assert "patient_info" in case
        assert "clinical_notes" in case
        assert isinstance(case["clinical_notes"], list)

        # check demographics
        info = case["patient_info"]
        assert "age" in info
        assert "gender" in info

        # check notes structure
        if case["clinical_notes"]:
            note = case["clinical_notes"][0]
            assert "timestamp" in note
            assert "service" in note
            assert "text" in note

    print(f"✓ Loaded {len(loader.patient_ids)} real patients successfully")


def test_fallback_behavior():
    """Test that loader fails loud when real data unavailable."""
    print("\n=== Testing Fail-Loud Behavior ===")

    # use non-existent directory - should raise exception
    try:
        loader = DataLoader(data_dir="non_existent_directory")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "Data directory does not exist" in str(e)
        print("✓ Correctly raised FileNotFoundError for missing data directory")

    # but dummy mode should still work
    loader = DataLoader(data_dir="non_existent_directory", use_dummy=True)
    assert loader.use_dummy, "Should be in dummy mode"
    assert loader.is_available(), "Dummy mode should be available"

    patients = loader.get_available_patients()
    assert len(patients) == 1, "Should have dummy patient"

    print("✓ Dummy mode works when explicitly requested")


def test_invalid_case_id():
    """Test handling of invalid case IDs."""
    print("\n=== Testing Invalid Case ID Handling ===")

    loader = DataLoader(use_dummy=True)

    try:
        loader.load_patient_case("invalid_format")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid case ID format" in str(e)
        print("✓ Correctly rejected invalid case ID format")

    try:
        loader.load_patient_case("iraki_case_abc")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid case ID" in str(e)
        print("✓ Correctly rejected non-numeric case ID")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("TESTING DATALOADER")
    print("=" * 50)

    tests = [
        test_dummy_mode,
        test_real_data_loading,
        test_fallback_behavior,
        test_invalid_case_id,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed.append(test.__name__)

    print("\n" + "=" * 50)
    if failed:
        print(f"FAILED: {len(failed)} test(s)")
        for name in failed:
            print(f"  - {name}")
        return 1
    else:
        print("SUCCESS: All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
