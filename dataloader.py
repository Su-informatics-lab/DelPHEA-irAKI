"""
irAKI Data Loader
==========================
Data loader for immune-related AKI patient data.
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

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for irAKI patient cohort."""

    # OMOP CDM standard concept IDs for gender
    GENDER_CONCEPTS = {8507: "Male", 8532: "Female", 8551: "Unknown", 8570: "Ambiguous"}

    def __init__(
        self,
        data_dir: str = "irAKI_data",
        cache_dir: Optional[str] = None,
        use_dummy: bool = False,
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the directory containing patient data files.
            cache_dir: Optional path for caching processed data.
            use_dummy: If True, use dummy data for testing.
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.use_dummy = use_dummy

        # initialize data containers
        self.clinical_notes_df = pd.DataFrame()
        self.person_df = pd.DataFrame()
        self.patient_ids: List[int] = []
        self._patient_cache: Dict[int, Dict] = {}

        # data availability flags
        self._has_clinical_notes = False
        self._has_demographics = False
        self._data_loaded = False

        # if explicitly using dummy mode, no need to load real data
        if use_dummy:
            return

        # for real data mode, fail loud if data directory doesn't exist
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}. "
                f"Please ensure the data directory exists or use use_dummy=True for testing."
            )

        # attempt to load real data
        try:
            self._load_all_data()
            self._data_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load patient data: {e}")

    def is_available(self) -> bool:
        """Check if patient data is available."""
        if self.use_dummy:
            return True

        return (
            self._data_loaded
            and len(self.patient_ids) > 0
            and self._has_clinical_notes
            and self._has_demographics
        )

    def get_available_patients(self, limit: int = 10) -> List[str]:
        """Get list of available patient IDs."""
        if self.use_dummy:
            return ["iraki_case_001"]

        if self.patient_ids:
            selected_ids = self.patient_ids[:limit]
            return [f"iraki_case_{pid}" for pid in selected_ids]

        return []

    def load_patient_case(self, case_id: str) -> Dict:
        """
        Load patient case data.

        Returns raw clinical notes and demographics for expert agents to analyze.
        """
        if self.use_dummy:
            if case_id != "iraki_case_001":
                raise ValueError(
                    f"Invalid case ID for dummy mode: {case_id}. "
                    f"Only 'iraki_case_001' is available in dummy mode."
                )
            return self._get_dummy_patient()

        # validate and extract person_id from case_id
        if not case_id.startswith("iraki_case_"):
            raise ValueError(f"Invalid case ID format: {case_id}")

        try:
            person_id = int(case_id.replace("iraki_case_", ""))
        except ValueError:
            raise ValueError(f"Invalid case ID: {case_id}")

        # check cache first
        if person_id in self._patient_cache:
            return self._patient_cache[person_id]

        # validate patient exists
        if person_id not in self.patient_ids:
            raise ValueError(
                f"Patient {person_id} not found in dataset. "
                f"Available patient IDs: {self.patient_ids[:5]}..."
            )

        # get demographics
        demographics = self._get_demographics(person_id)

        # get clinical notes
        clinical_notes = self._get_clinical_notes(person_id)

        # build patient case - raw data for agents to analyze
        patient_case = {
            "case_id": case_id,
            "patient_info": demographics,
            "patient_summary": (
                f"Patient {person_id}: {demographics.get('age', 'Unknown')}-year-old "
                f"{demographics.get('gender', 'Unknown')} with {len(clinical_notes)} clinical notes."
            ),
            "clinical_notes": clinical_notes,
            "total_notes": len(clinical_notes),
            # empty placeholders - agents will extract from notes
            "medication_history": {},
            "lab_values": {},
            "imaging_reports": [],
        }

        # cache the patient case
        self._patient_cache[person_id] = patient_case

        return patient_case

    def _load_all_data(self) -> None:
        """Load all available patient data files."""
        logger.info("Loading patient data...")

        self._load_clinical_notes()
        self._load_person_data()
        self._build_patient_index()

        if not self.patient_ids:
            raise RuntimeError("No valid patients found with complete data")

        logger.info(f"Loaded {len(self.patient_ids)} patients")

    def _load_clinical_notes(self) -> None:
        """Load clinical notes from parquet file."""
        notes_path = (
            self.data_dir
            / "clinical_notes_version3"
            / "part-00000-347607e6-182c-419e-9a83-8a38b90986fa-c000.snappy.parquet"
        )

        if not notes_path.exists():
            raise FileNotFoundError(f"Clinical notes file not found: {notes_path}")

        self.clinical_notes_df = pd.read_parquet(notes_path)

        # standardize column names to uppercase
        self.clinical_notes_df.columns = [
            col.upper() for col in self.clinical_notes_df.columns
        ]

        # convert person ID to int64
        self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"] = self.clinical_notes_df[
            "OBFUSCATED_GLOBAL_PERSON_ID"
        ].astype("int64")

        # convert timestamp
        self.clinical_notes_df["PHYSIOLOGIC_TIME"] = pd.to_datetime(
            self.clinical_notes_df["PHYSIOLOGIC_TIME"], errors="coerce"
        )

        self._has_clinical_notes = True
        logger.info(f"Loaded {len(self.clinical_notes_df)} clinical notes")

    def _load_person_data(self) -> None:
        """Load person demographics from CSV file."""
        person_path = self.data_dir / "structured_data" / "r6335_person.csv"

        if not person_path.exists():
            raise FileNotFoundError(
                f"Person demographics file not found: {person_path}"
            )

        self.person_df = pd.read_csv(person_path, dtype={"person_id": "int64"})

        # standardize column names to lowercase
        self.person_df.columns = [col.lower() for col in self.person_df.columns]

        # set index for O(1) lookups
        self.person_df.set_index("person_id", inplace=True)

        self._has_demographics = True
        logger.info(f"Loaded {len(self.person_df)} person records")

    def _build_patient_index(self) -> None:
        """Build index of patients with complete data."""
        if not self._has_clinical_notes or not self._has_demographics:
            self.patient_ids = []
            return

        # get unique patient IDs from each source
        notes_patients = set(
            self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
        )
        person_patients = set(self.person_df.index.unique())

        # find intersection
        complete_patients = notes_patients & person_patients

        if complete_patients:
            self.patient_ids = sorted(list(complete_patients))
            logger.info(
                f"Identified {len(self.patient_ids)} patients with complete data"
            )
        else:
            self.patient_ids = []

    def _get_demographics(self, person_id: int) -> Dict:
        """Extract patient demographic information."""
        if person_id not in self.person_df.index:
            return {
                "person_id": person_id,
                "age": None,
                "gender": "Unknown",
                "race": "Unknown",
                "ethnicity": "Unknown",
            }

        person = self.person_df.loc[person_id]
        current_year = datetime.now().year

        # calculate age
        year_of_birth = person.get("year_of_birth")
        age = current_year - int(year_of_birth) if pd.notna(year_of_birth) else None

        # map gender
        gender_concept = person.get("gender_concept_id")
        gender = self.GENDER_CONCEPTS.get(gender_concept, "Unknown")

        return {
            "person_id": person_id,
            "age": age,
            "gender": gender,
            "race": str(person.get("race_source_value", "Unknown")),
            "ethnicity": str(person.get("ethnicity_source_value", "Unknown")),
        }

    def _get_clinical_notes(self, person_id: int) -> List[Dict]:
        """
        Extract clinical notes for a patient.

        Returns raw notes with timestamps and metadata for agent analysis.
        """
        patient_notes = self.clinical_notes_df[
            self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"] == person_id
        ].sort_values("PHYSIOLOGIC_TIME")

        notes = []
        for _, row in patient_notes.iterrows():
            timestamp = row.get("PHYSIOLOGIC_TIME")
            timestamp_str = timestamp.isoformat() if pd.notna(timestamp) else None

            notes.append(
                {
                    "timestamp": timestamp_str,
                    "service": str(row.get("SERVICE_NAME", "Unknown")),
                    "encounter_id": str(row.get("ENCOUNTER_ID", "")),
                    "text": str(row.get("REPORT_TEXT", "")),
                }
            )

        return notes

    def _get_dummy_patient(self) -> Dict:
        """Return a fixed dummy patient for testing."""
        return {
            "case_id": "iraki_case_001",
            "patient_info": {
                "person_id": 1,
                "age": 68,
                "gender": "Male",
                "race": "White",
                "ethnicity": "Non-Hispanic",
            },
            "patient_summary": ("Patient 1: 68-year-old Male with 2 clinical notes."),
            "clinical_notes": [
                {
                    "timestamp": "2024-01-15T10:30:00",
                    "service": "Oncology",
                    "encounter_id": "ENC001",
                    "text": (
                        "Patient with metastatic melanoma, started on pembrolizumab "
                        "6 weeks ago. Baseline creatinine 1.0. Tolerating treatment well. "
                        "No signs of immune-related adverse events at this time."
                    ),
                },
                {
                    "timestamp": "2024-02-01T14:00:00",
                    "service": "Nephrology",
                    "encounter_id": "ENC002",
                    "text": (
                        "Referred for acute kidney injury. Creatinine 3.2, up from baseline 1.0 two weeks ago. "
                        "Urinalysis shows proteinuria 2+, no RBC casts. Patient denies NSAID use. "
                        "No recent contrast exposure. Blood pressure stable. "
                        "Clinical picture concerning for pembrolizumab-induced acute interstitial nephritis. "
                        "Recommend holding ICI and starting prednisone 1mg/kg."
                    ),
                },
            ],
            "total_notes": 2,
            "medication_history": {},
            "lab_values": {},
            "imaging_reports": [],
        }
