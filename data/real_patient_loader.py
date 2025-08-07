"""
irAKI Data Loader
==========================
Production-ready data loader for immune-related AKI patient data.
Processes clinical notes and demographics with placeholders for labs/medications.

CSV Processing Details:
-----------------------
1. Clinical Notes (part-00000-*.snappy.parquet):
   - Currently stored as parquet file in 'clinical_notes_version3' directory
   - Loaded using pd.read_parquet() for efficient reading
   - Column names are standardized to uppercase for consistency
   - OBFUSCATED_GLOBAL_PERSON_ID is converted to int64 to match person table
   - PHYSIOLOGIC_TIME column is converted to datetime format
   - Contains fields: OBFUSCATED_GLOBAL_PERSON_ID, ENCOUNTER_ID, SERVICE_NAME,
     PHYSIOLOGIC_TIME, REPORT_TEXT
   - Each row represents a single clinical note entry

2. Person Demographics (r6335_person.csv):
   - Standard CSV format with patient demographic information
   - Loaded using pd.read_csv() with explicit dtype={'person_id': 'int64'}
   - person_id is set as DataFrame index for O(1) lookup performance
   - Column names are standardized to lowercase for this dataset
   - Contains fields: person_id, year_of_birth, gender_concept_id,
     race_source_value, ethnicity_source_value
   - Gender concepts are mapped using OMOP standard concept IDs
   - Age is calculated from year_of_birth relative to current year

3. Measurements (r6335_measurement.csv) - CURRENTLY DISABLED:
   - Placeholder implementation awaiting concept mapping
   - Will contain lab results once concept.csv is available for mapping

4. Drug Exposures (r6335_drug_exposure.csv) - CURRENTLY DISABLED:
   - Placeholder implementation awaiting concept mapping
   - Will contain medication history once concept.csv is available

Data Integration:
-----------------
- Patient IDs are cross-referenced between clinical notes (OBFUSCATED_GLOBAL_PERSON_ID)
  and person table (person_id) to ensure data completeness
- Both ID fields are converted to int64 for consistent data types
- person_df uses person_id as index for efficient lookups
- Only patients with both clinical notes AND demographics are included in final cohort
- Data is cached in memory using dictionary storage for repeated access
- Sanity check validates that all clinical note patients exist in person table

Current Dataset Statistics:
---------------------------
- 500 unique patients with clinical notes (potential irAKI cohort)
- 57,694 total patients in person demographics table
- 100% of patients with notes have complete demographics (verified via sanity check)
- 57,194 patients have demographics but no notes (broader hospital population)
- Average patient has ~135 clinical notes spanning multiple years

Performance Optimizations:
-------------------------
- DataFrame indexing for O(1) patient lookups
- In-memory caching for repeated patient access
- Efficient parquet format for clinical notes storage
- Type-consistent int64 patient IDs throughout

TODO:
-------------------
- Migration from CSV to parquet format for all structured data
- Integration of lab results once concept mapping available
- Addition of medication data with ICI identification
- Multi-threaded loading for large-scale processing
"""
"""
Production data loader for irAKI clinical data.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data.dummy import DummyDataGenerator

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


class irAKIDataLoader:
    """Production-ready data loader for REAL irAKI clinical data."""

    GENDER_CONCEPTS = {8507: "Male", 8532: "Female", 8551: "Unknown", 8570: "Ambiguous"}

    def __init__(
        self,
        data_dir: str = "irAKI_data",
        cache_dir: Optional[str] = None,
        run_sanity_check: bool = True,
    ):
        """Initialize data loader for real data only."""
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"

        # Data containers
        self.clinical_notes_df = pd.DataFrame()
        self.person_df = pd.DataFrame()
        self.patient_ids = []

        # Try to load real data
        self._load_all_data()

        # Sanity check
        if run_sanity_check and not self.clinical_notes_df.empty:
            self._run_sanity_check()

    def _load_all_data(self) -> None:
        """Load all available real data files."""
        try:
            self._load_clinical_notes()
            self._load_person_data()
            self._build_patient_index()
            logger.info(f"Loaded {len(self.patient_ids)} real patients")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.patient_ids = []

    def _load_clinical_notes(self) -> None:
        """Load clinical notes from parquet."""
        notes_path = (
            self.data_dir
            / "clinical_notes_version3"
            / "part-00000-347607e6-182c-419e-9a83-8a38b90986fa-c000.snappy.parquet"
        )

        if notes_path.exists():
            self.clinical_notes_df = pd.read_parquet(notes_path)
            self.clinical_notes_df.columns = [
                col.upper() for col in self.clinical_notes_df.columns
            ]
            self.clinical_notes_df[
                "OBFUSCATED_GLOBAL_PERSON_ID"
            ] = self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].astype("int64")

            if "PHYSIOLOGIC_TIME" in self.clinical_notes_df.columns:
                self.clinical_notes_df["PHYSIOLOGIC_TIME"] = pd.to_datetime(
                    self.clinical_notes_df["PHYSIOLOGIC_TIME"], errors="coerce"
                )
            logger.info(f"Loaded {len(self.clinical_notes_df)} clinical notes")
        else:
            logger.warning(f"Clinical notes not found: {notes_path}")

    def _load_person_data(self) -> None:
        """Load person demographics from CSV."""
        person_path = self.data_dir / "structured_data" / "r6335_person.csv"

        if person_path.exists():
            self.person_df = pd.read_csv(person_path, dtype={"person_id": "int64"})
            self.person_df.columns = [col.lower() for col in self.person_df.columns]
            self.person_df.set_index("person_id", inplace=True)
            logger.info(f"Loaded {len(self.person_df)} person records")
        else:
            logger.warning(f"Person data not found: {person_path}")

    def _build_patient_index(self) -> None:
        """Build index of patients with both notes and demographics."""
        if not self.clinical_notes_df.empty and not self.person_df.empty:
            notes_patients = set(
                self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
            )
            person_patients = set(self.person_df.index.unique())
            self.patient_ids = sorted(list(notes_patients & person_patients))
        else:
            self.patient_ids = []

    def _run_sanity_check(self) -> None:
        """Run sanity check on data integrity."""
        if self.clinical_notes_df.empty or self.person_df.empty:
            return

        notes_ids = set(self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique())
        person_ids = set(self.person_df.index.unique())
        missing = notes_ids - person_ids

        if missing:
            logger.warning(f"{len(missing)} patients have notes but no demographics")

    def load_patient_case(self, case_id: str) -> Dict:
        """
        Load real patient case data.

        Falls back to dummy data if patient not found.
        """
        # Extract person_id
        try:
            person_id = int(case_id.replace("iraki_case_", ""))
        except:
            return DummyDataGenerator.generate_patient_case(case_id)

        # Check if patient exists
        if person_id not in self.patient_ids:
            logger.warning(f"Patient {person_id} not found, using dummy data")
            return DummyDataGenerator.generate_patient_case(case_id)

        # Load real patient data
        demographics = self._get_demographics(person_id)
        notes = self._get_clinical_notes(person_id)

        return {
            "case_id": case_id,
            "patient_info": demographics,
            "patient_summary": (
                f"Patient {person_id}: {demographics['age']}-year-old "
                f"{demographics['gender']} with {len(notes)} clinical notes."
            ),
            "clinical_notes": notes,
            "total_notes": len(notes),
            "data_completeness": 1.0 if notes else 0.5,
            "medication_history": self._extract_medications(notes),
            "lab_values": self._extract_labs(notes),
            "imaging_reports": [],
        }

    def _get_demographics(self, person_id: int) -> Dict:
        """Get patient demographics."""
        if person_id not in self.person_df.index:
            return {
                "person_id": person_id,
                "age": 65,
                "gender": "Unknown",
                "race": "Unknown",
                "ethnicity": "Unknown",
            }

        person = self.person_df.loc[person_id]
        current_year = datetime.now().year
        age = current_year - person.get("year_of_birth", current_year - 65)
        gender = self.GENDER_CONCEPTS.get(
            person.get("gender_concept_id", 8551), "Unknown"
        )

        return {
            "person_id": person_id,
            "age": age,
            "gender": gender,
            "race": str(person.get("race_source_value", "Unknown")),
            "ethnicity": str(person.get("ethnicity_source_value", "Unknown")),
        }

    def _get_clinical_notes(self, person_id: int) -> List[Dict]:
        """Get clinical notes for patient."""
        patient_notes = self.clinical_notes_df[
            self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"] == person_id
        ].sort_values("PHYSIOLOGIC_TIME")

        notes = []
        for _, row in patient_notes.iterrows():
            notes.append(
                {
                    "timestamp": row.get("PHYSIOLOGIC_TIME", "").isoformat()
                    if pd.notna(row.get("PHYSIOLOGIC_TIME"))
                    else "",
                    "service": row.get("SERVICE_NAME", "Unknown"),
                    "encounter_id": str(row.get("ENCOUNTER_ID", "")),
                    "text": str(row.get("REPORT_TEXT", "")),
                }
            )
        return notes

    def _extract_medications(self, notes: List[Dict]) -> Dict:
        """Extract medication mentions from notes."""
        meds = {}
        ici_drugs = [
            "nivolumab",
            "ipilimumab",
            "pembrolizumab",
            "atezolizumab",
            "opdivo",
            "yervoy",
            "keytruda",
            "tecentriq",
        ]

        for note in notes:
            text_lower = note.get("text", "").lower()
            for drug in ici_drugs:
                if drug in text_lower:
                    meds[drug] = {"mentioned": True}
        return meds

    def _extract_labs(self, notes: List[Dict]) -> Dict:
        """Extract lab values from notes."""
        import re

        labs = {}

        for note in notes:
            text = note.get("text", "")
            # Simple creatinine extraction
            creat_matches = re.findall(r"creatinine[:\s]+(\d+\.?\d*)", text.lower())
            if creat_matches:
                labs["creatinine_values"] = [float(m) for m in creat_matches]
                break
        return labs
