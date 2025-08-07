"""
Data loader wrapper for DelPHEA-irAKI.

Handles both real patient data and dummy data for testing.
"""

import logging
from typing import Dict, List

from config.core import RuntimeConfig

logger = logging.getLogger(__name__)


class DataLoaderWrapper:
    """Wrapper to handle both real and dummy data modes."""

    def __init__(self, runtime_config: RuntimeConfig):
        """Initialize data loader wrapper.

        Args:
            runtime_config: Runtime configuration with data paths
        """
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.DataLoaderWrapper")

        if runtime_config.use_real_data:
            # initialize real data loader
            self.logger.info("Initializing real patient data loader...")
            try:
                # import the production data loader
                from data_loader import irAKIDataLoader

                self.data_loader = irAKIDataLoader(
                    data_dir=runtime_config.data_dir,
                    cache_dir=runtime_config.cache_dir,
                    run_sanity_check=True,
                )
                self.logger.info(
                    f"Loaded {len(self.data_loader.patient_ids)} real patients"
                )
            except Exception as e:
                self.logger.error(f"Failed to load real data: {e}")
                self.logger.info("Falling back to dummy data mode")
                self.data_loader = None
                self.config.use_real_data = False
        else:
            self.data_loader = None
            self.logger.info("Using dummy data mode")

    def load_patient_case(self, case_id: str) -> Dict:
        """Load patient case data for irAKI assessment.

        Args:
            case_id: Case identifier

        Returns:
            Dict: Structured patient data for Delphi assessment
        """
        if self.config.use_real_data and self.data_loader:
            # use real data loader's load_patient_case method
            return self.data_loader.load_patient_case(case_id)
        else:
            # return dummy data for testing
            return self._get_dummy_patient_case(case_id)

    def get_available_patients(self, limit: int = 10) -> List[str]:
        """Get list of available patient IDs.

        Args:
            limit: Maximum number of IDs to return

        Returns:
            List[str]: Available patient/case IDs
        """
        if self.config.use_real_data and self.data_loader:
            # return real patient IDs formatted as case IDs
            patient_ids = self.data_loader.patient_ids[:limit]
            return [f"iraki_case_{pid}" for pid in patient_ids]
        else:
            # return dummy case IDs
            return [f"iraki_case_{str(i).zfill(3)}" for i in range(1, limit + 1)]

    def _get_dummy_patient_case(self, case_id: str) -> Dict:
        """Generate dummy patient case for testing.

        Args:
            case_id: Case identifier

        Returns:
            Dict: Dummy patient data with simple structure
        """
        # dummy clinical notes with timestamps
        dummy_notes = [
            {
                "timestamp": "2024-03-15T10:30:00",
                "service": "Oncology",
                "encounter_id": "ENC001",
                "text": (
                    "67-year-old female with metastatic melanoma on combination "
                    "immunotherapy (nivolumab + ipilimumab). Patient presenting with "
                    "elevated creatinine, increased from baseline 1.1 to 3.2 mg/dL over "
                    "2 weeks. No recent NSAID use or contrast exposure."
                ),
            },
            {
                "timestamp": "2024-03-14T14:20:00",
                "service": "Nephrology",
                "encounter_id": "ENC002",
                "text": (
                    "Consultation for AKI. Creatinine trending upward. Urinalysis shows "
                    "proteinuria 2+, RBC 10-20/hpf. Suspect possible immune-related "
                    "nephritis given recent ICI therapy. Recommend holding immunotherapy "
                    "and considering steroids if no improvement."
                ),
            },
            {
                "timestamp": "2024-03-13T09:00:00",
                "service": "Internal Medicine",
                "encounter_id": "ENC003",
                "text": (
                    "Admission for acute kidney injury. Past medical history includes "
                    "Stage II CKD, HTN, DM2. Current medications include metoprolol and "
                    "metformin. Volume status appears euvolemic. No signs of obstruction "
                    "on renal ultrasound."
                ),
            },
        ]

        return {
            "case_id": case_id,
            "patient_info": {
                "person_id": 99999,  # dummy ID
                "age": 67,
                "gender": "Female",
                "race": "White",
                "ethnicity": "Non-Hispanic",
            },
            "patient_summary": (
                "Patient 99999: 67-year-old Female with 3 clinical notes available "
                "for review."
            ),
            "clinical_notes": dummy_notes,
            "total_notes": len(dummy_notes),
            "data_completeness": 1.0,
            # empty placeholders for structured data
            "medication_history": {},
            "lab_values": {},
            "imaging_reports": [],
        }
