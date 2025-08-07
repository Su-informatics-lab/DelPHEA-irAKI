"""
Patient data loader wrapper for DelPHEA-irAKI.
Orchestrates between real and dummy data sources.
"""

import logging
from typing import Dict, List

from config.core import RuntimeConfig
from data.dummy_patient_generator import (
    DummyDataGenerator,  # FIXED: Using correct class name
)
from data.real_patient_loader import RealPatientLoader

logger = logging.getLogger(__name__)


class PatientLoaderWrapper:
    """Main patient data loader that handles both real and dummy data."""

    def __init__(self, runtime_config: RuntimeConfig):
        """Initialize the patient loader wrapper."""
        self.config = runtime_config
        self.real_loader = None
        self.using_real_data = False

        if runtime_config.use_real_data:
            try:
                logger.info("Attempting to load real patient data...")
                self.real_loader = RealPatientLoader(
                    data_dir=runtime_config.data_dir,
                    cache_dir=runtime_config.cache_dir,
                )

                if self.real_loader.is_available():
                    self.using_real_data = True
                    logger.info(
                        f"✅ Using REAL data: {len(self.real_loader.patient_ids)} patients available"
                    )
                else:
                    logger.warning(
                        "⚠️ No real patients found, falling back to dummy data"
                    )

            except Exception as e:
                logger.error(f"❌ Failed to initialize real data loader: {e}")

        if not self.using_real_data:
            logger.info("🎭 Using DUMMY data mode")

    def load_patient_case(self, case_id: str, scenario: str = "classic_iraki") -> Dict:
        """
        Load patient case (real or dummy).

        Args:
            case_id: Case identifier
            scenario: For dummy data, which scenario to use
        """
        if self.using_real_data and self.real_loader:
            logger.debug(f"Loading real patient: {case_id}")
            return self.real_loader.load_patient_case(case_id)
        else:
            logger.debug(f"Generating dummy patient: {case_id} (scenario: {scenario})")
            return DummyDataGenerator.generate_patient_case(
                case_id, scenario
            )  # FIXED: Using correct class name

    def get_available_patients(self, limit: int = 10) -> List[str]:
        """Get list of available patient IDs."""
        if self.using_real_data and self.real_loader:
            patient_ids = self.real_loader.patient_ids[:limit]
            return [f"iraki_case_{pid}" for pid in patient_ids]
        else:
            # Return dummy IDs
            return [f"iraki_case_{str(i).zfill(3)}" for i in range(1, limit + 1)]

    def get_data_source(self) -> str:
        """Get current data source type."""
        return "REAL" if self.using_real_data else "DUMMY"

    def get_statistics(self) -> Dict:
        """Get statistics about available data."""
        if self.using_real_data and self.real_loader:
            return {
                "data_source": "REAL",
                "total_patients": len(self.real_loader.patient_ids),
                "has_clinical_notes": not self.real_loader.clinical_notes_df.empty,
                "has_demographics": not self.real_loader.person_df.empty,
            }
        else:
            return {
                "data_source": "DUMMY",
                "total_patients": "unlimited",
                "scenarios": list(
                    DummyDataGenerator.CLINICAL_SCENARIOS.keys()
                ),  # FIXED: Using correct class name
            }
