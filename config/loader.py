"""
Configuration loader for DelPHEA-irAKI system.

Centralized configuration loading with fail-fast validation for expert panels,
questionnaires, and prompt templates.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.dummy import DummyDataGenerator

from config.core import RuntimeConfig

logger = logging.getLogger(__name__)


class DataLoaderWrapper:
    """Simple wrapper to handle real vs dummy data."""

    def __init__(self, runtime_config: RuntimeConfig):
        """Initialize wrapper."""
        self.config = runtime_config
        self.data_loader = None

        if runtime_config.use_real_data:
            try:
                from data_loader import irAKIDataLoader

                self.data_loader = irAKIDataLoader(
                    data_dir=runtime_config.data_dir,
                    cache_dir=runtime_config.cache_dir,
                )

                if not self.data_loader.patient_ids:
                    logger.warning("No real patients found, using dummy mode")
                    self.data_loader = None
                    self.config.use_real_data = False
                else:
                    logger.info(
                        f"Loaded {len(self.data_loader.patient_ids)} real patients"
                    )

            except Exception as e:
                logger.error(f"Failed to load real data: {e}")
                self.data_loader = None
                self.config.use_real_data = False

        if not self.config.use_real_data:
            logger.info("Using dummy data mode")

    def load_patient_case(self, case_id: str) -> Dict:
        """Load patient case (real or dummy)."""
        if self.data_loader:
            return self.data_loader.load_patient_case(case_id)
        else:
            # Use dummy data generator
            return DummyDataGenerator.generate_patient_case(case_id)

    def get_available_patients(self, limit: int = 10) -> List[str]:
        """Get available patient IDs."""
        if self.data_loader and self.data_loader.patient_ids:
            patient_ids = self.data_loader.patient_ids[:limit]
            return [f"iraki_case_{pid}" for pid in patient_ids]
        else:
            # Return dummy IDs
            return [f"iraki_case_{str(i).zfill(3)}" for i in range(1, limit + 1)]
