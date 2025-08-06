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

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# suppress pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataStatus(Enum):
    """Status indicators for data availability"""

    AVAILABLE = "available"
    PARTIAL = "partial"
    MISSING = "missing"
    ERROR = "error"


@dataclass
class ClinicalNote:
    """Structured clinical note"""

    note_id: str
    person_id: int
    encounter_id: str
    service_name: str
    note_datetime: datetime
    note_text: str
    text_length: int

    def to_dict(self) -> Dict:
        """Convert to dictionary with string dates"""
        data = asdict(self)
        data["note_datetime"] = (
            self.note_datetime.isoformat() if self.note_datetime else None
        )
        return data


@dataclass
class PatientDemographics:
    """Patient demographic information"""

    person_id: int
    age: int
    gender: str
    race: str
    ethnicity: str
    year_of_birth: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PatientData:
    """Complete patient data structure"""

    person_id: int
    demographics: PatientDemographics
    clinical_notes: List[ClinicalNote]
    data_status: Dict[str, DataStatus] = field(default_factory=dict)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for this patient"""
        return {
            "person_id": self.person_id,
            "age": self.demographics.age,
            "gender": self.demographics.gender,
            "total_notes": len(self.clinical_notes),
            "date_range": self._get_date_range(),
            "data_completeness": self._calculate_completeness(),
        }

    def _get_date_range(self) -> Dict:
        """Get the date range of available data"""
        if not self.clinical_notes:
            return {"start": None, "end": None}

        dates = [
            note.note_datetime for note in self.clinical_notes if note.note_datetime
        ]
        if not dates:
            return {"start": None, "end": None}

        return {
            "start": min(dates).isoformat(),
            "end": max(dates).isoformat(),
            "days_span": (max(dates) - min(dates)).days,
        }

    def _calculate_completeness(self) -> float:
        """Calculate data completeness score"""
        scores = {
            "demographics": (
                1.0
                if self.data_status.get("demographics") == DataStatus.AVAILABLE
                else 0.0
            ),
            "notes": 1.0 if len(self.clinical_notes) > 0 else 0.0,
        }
        return sum(scores.values()) / len(scores)


class irAKIDataLoader:
    """
    Production-ready data loader for irAKI clinical data.

    Handles loading and processing of:
    - Clinical notes (fully implemented)
    - Demographics (fully implemented)
    - Lab results (disabled - awaiting concept mapping)
    - Medications (disabled - awaiting concept mapping)
    """

    # standard OMOP concept IDs (for when concept.csv is available)
    GENDER_CONCEPTS = {8507: "Male", 8532: "Female", 8551: "Unknown", 8570: "Ambiguous"}

    def __init__(
        self,
        data_dir: str = "irAKI_data",
        cache_dir: Optional[str] = None,
        run_sanity_check: bool = True,
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory
            cache_dir: Optional directory for caching processed data
            run_sanity_check: Whether to run sanity check on initialization
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

        # data containers
        self.clinical_notes_df = None
        self.person_df = None

        # patient index
        self.patient_ids = []
        self.patient_data_cache = {}

        # load data
        self._load_all_data()

        # run sanity check if requested
        if run_sanity_check:
            sanity_results = self.sanity_check_patient_ids()
            if not sanity_results["is_notes_subset_of_person"]:
                logger.warning(
                    f"⚠️ Sanity check failed: {sanity_results['missing_from_person_count']} "
                    f"patient IDs from notes are not in person table"
                )

    def _load_all_data(self) -> None:
        """Load all available data files"""
        logger.info("Starting data loading process...")

        # load clinical notes (required)
        self._load_clinical_notes()

        # load person demographics (required)
        self._load_person_data()

        # medications and labs are disabled for now
        # self._load_measurements()
        # self._load_drug_exposures()

        # build patient index
        self._build_patient_index()

        logger.info(f"Data loading complete. Found {len(self.patient_ids)} patients.")

    def _load_clinical_notes(self) -> None:
        """Load clinical notes from parquet file"""
        notes_path = (
            self.data_dir
            / "clinical_notes_version3"
            / "part-00000-347607e6-182c-419e-9a83-8a38b90986fa-c000.snappy.parquet"
        )

        if not notes_path.exists():
            raise FileNotFoundError(f"Clinical notes file not found: {notes_path}")

        logger.info("Loading clinical notes...")
        self.clinical_notes_df = pd.read_parquet(notes_path)

        # standardize column names to uppercase
        self.clinical_notes_df.columns = [
            col.upper() for col in self.clinical_notes_df.columns
        ]

        # convert patient ID to int64 to match person table
        self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"] = self.clinical_notes_df[
            "OBFUSCATED_GLOBAL_PERSON_ID"
        ].astype("int64")

        # convert timestamps
        if "PHYSIOLOGIC_TIME" in self.clinical_notes_df.columns:
            self.clinical_notes_df["PHYSIOLOGIC_TIME"] = pd.to_datetime(
                self.clinical_notes_df["PHYSIOLOGIC_TIME"], errors="coerce"
            )

        logger.info(f"Loaded {len(self.clinical_notes_df)} clinical notes")

    def _load_person_data(self) -> None:
        """Load person demographics from CSV"""
        person_path = self.data_dir / "structured_data" / "r6335_person.csv"

        if not person_path.exists():
            raise FileNotFoundError(f"Person data file not found: {person_path}")

        logger.info("Loading person demographics...")
        # read with explicit dtype for person_id to avoid float64
        self.person_df = pd.read_csv(person_path, dtype={"person_id": "int64"})

        # standardize column names to lowercase for this dataset
        self.person_df.columns = [col.lower() for col in self.person_df.columns]

        # set person_id as index for efficient lookups
        self.person_df.set_index("person_id", inplace=True)

        logger.info(f"Loaded {len(self.person_df)} person records")

    # def _load_measurements(self) -> None:
    #     """Load measurements data if available - DISABLED pending concept mapping"""
    #     logger.info("Measurements loading disabled - awaiting concept mapping")
    #     self.measurements_df = pd.DataFrame()

    # def _load_drug_exposures(self) -> None:
    #     """Load drug exposure data if available - DISABLED pending concept mapping"""
    #     logger.info("Drug exposures loading disabled - awaiting concept mapping")
    #     self.drug_exposure_df = pd.DataFrame()

    def _build_patient_index(self) -> None:
        """Build index of all unique patient IDs"""
        logger.info("Building patient index...")

        # get unique patient IDs from clinical notes (now int64)
        notes_patients = set(
            self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
        )

        # get unique patient IDs from person table (from index)
        person_patients = set(self.person_df.index.unique())

        # use intersection to ensure we have both notes and demographics
        self.patient_ids = sorted(list(notes_patients & person_patients))

        logger.info(
            f"Found {len(self.patient_ids)} patients with both notes and demographics"
        )

        # analyze missing demographics in detail
        missing_demographics = notes_patients - person_patients
        if missing_demographics:
            logger.warning(
                f"{len(missing_demographics)} patients have notes but no demographics record. "
                f"These patients are completely missing from person table."
            )
            # optionally log first few missing IDs for investigation
            sample_missing = list(missing_demographics)[:5]
            logger.debug(f"Sample missing patient IDs: {sample_missing}")

        if len(person_patients - notes_patients) > 0:
            logger.warning(
                f"{len(person_patients - notes_patients)} patients have demographics but no notes"
            )

    def get_patient_demographics(self, person_id: int) -> Optional[PatientDemographics]:
        """
        Get demographics for a specific patient.

        Args:
            person_id: Patient identifier

        Returns:
            PatientDemographics object or None if not found
        """
        # use index-based lookup for efficiency
        if person_id not in self.person_df.index:
            logger.warning(f"No demographics found for patient {person_id}")
            return None

        person = self.person_df.loc[person_id]

        # calculate age
        current_year = datetime.now().year
        year_of_birth = person.get("year_of_birth", current_year - 65)
        age = current_year - year_of_birth

        # map gender using OMOP standard concepts
        gender_concept = person.get("gender_concept_id", 8551)
        gender = self.GENDER_CONCEPTS.get(gender_concept, "Unknown")

        # get race and ethnicity using source values
        race = person.get("race_source_value", "Unknown")
        ethnicity = person.get("ethnicity_source_value", "Unknown")

        return PatientDemographics(
            person_id=person_id,
            age=age,
            gender=gender,
            race=race if race and race != "nan" else "Unknown",
            ethnicity=ethnicity if ethnicity and ethnicity != "nan" else "Unknown",
            year_of_birth=year_of_birth,
        )

    def get_patient_notes(self, person_id: int) -> List[ClinicalNote]:
        """
        Get all clinical notes for a patient.

        Args:
            person_id: Patient identifier

        Returns:
            List of ClinicalNote objects
        """
        patient_notes = self.clinical_notes_df[
            self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"] == person_id
        ].copy()

        if patient_notes.empty:
            logger.warning(f"No clinical notes found for patient {person_id}")
            return []

        # sort by time
        patient_notes = patient_notes.sort_values("PHYSIOLOGIC_TIME")

        notes = []
        for idx, row in patient_notes.iterrows():
            note = ClinicalNote(
                note_id=f"{person_id}_{idx}",
                person_id=person_id,
                encounter_id=str(row.get("ENCOUNTER_ID", "")),
                service_name=row.get("SERVICE_NAME", "Unknown"),
                note_datetime=row.get("PHYSIOLOGIC_TIME"),
                note_text=row.get("REPORT_TEXT", ""),
                text_length=len(str(row.get("REPORT_TEXT", ""))),
            )
            notes.append(note)

        return notes

    def load_patient(
        self, person_id: int, use_cache: bool = True
    ) -> Optional[PatientData]:
        """
        Load complete data for a patient.

        Args:
            person_id: Patient identifier
            use_cache: Whether to use cached data if available

        Returns:
            PatientData object or None if patient not found
        """
        # check cache first
        if use_cache and person_id in self.patient_data_cache:
            logger.debug(f"Loading patient {person_id} from cache")
            return self.patient_data_cache[person_id]

        # check if patient exists
        if person_id not in self.patient_ids:
            logger.error(f"Patient {person_id} not found in database")
            return None

        logger.info(f"Loading data for patient {person_id}")

        # load demographics
        demographics = self.get_patient_demographics(person_id)
        if not demographics:
            logger.error(f"Failed to load demographics for patient {person_id}")
            return None

        # load clinical notes
        clinical_notes = self.get_patient_notes(person_id)

        # determine data status
        data_status = {
            "demographics": (
                DataStatus.AVAILABLE if demographics else DataStatus.MISSING
            ),
            "clinical_notes": (
                DataStatus.AVAILABLE if clinical_notes else DataStatus.MISSING
            ),
            "lab_results": DataStatus.MISSING,  # disabled for now
            "medications": DataStatus.MISSING,  # disabled for now
        }

        # create patient data object
        patient_data = PatientData(
            person_id=person_id,
            demographics=demographics,
            clinical_notes=clinical_notes,
            data_status=data_status,
        )

        # cache the data
        if use_cache:
            self.patient_data_cache[person_id] = patient_data

        return patient_data

    def get_all_patients_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all patients.

        Returns:
            DataFrame with patient summaries
        """
        summaries = []

        for i, patient_id in enumerate(self.patient_ids):
            if i % 50 == 0:
                logger.info(f"Processing patient {i+1}/{len(self.patient_ids)}")

            patient_data = self.load_patient(patient_id)
            if patient_data:
                summaries.append(patient_data.get_summary_stats())

        return pd.DataFrame(summaries)

    def export_patient_data(
        self, person_id: int, output_dir: str = "exported_data"
    ) -> str:
        """
        Export patient data to JSON file.

        Args:
            person_id: Patient identifier
            output_dir: Directory for exported files

        Returns:
            Path to exported file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        patient_data = self.load_patient(person_id)
        if not patient_data:
            raise ValueError(f"Patient {person_id} not found")

        # convert to exportable format
        export_data = {
            "person_id": patient_data.person_id,
            "demographics": patient_data.demographics.to_dict(),
            "clinical_notes": [note.to_dict() for note in patient_data.clinical_notes],
            "summary_stats": patient_data.get_summary_stats(),
            "data_status": {k: v.value for k, v in patient_data.data_status.items()},
        }

        # save to JSON
        output_file = output_path / f"patient_{person_id}.json"
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported patient {person_id} to {output_file}")
        return str(output_file)

    def get_sample_output(self) -> Dict:
        """
        Get sample output for demonstration.

        Returns:
            Dictionary with sample data from first patient
        """
        if not self.patient_ids:
            return {"error": "No patients found"}

        # load first patient
        first_patient_id = self.patient_ids[0]
        patient_data = self.load_patient(first_patient_id)

        if not patient_data:
            return {"error": f"Failed to load patient {first_patient_id}"}

        # create sample output
        sample = {
            "patient_summary": patient_data.get_summary_stats(),
            "demographics": patient_data.demographics.to_dict(),
            "first_note": (
                patient_data.clinical_notes[0].to_dict()
                if patient_data.clinical_notes
                else None
            ),
            "total_notes": len(patient_data.clinical_notes),
            "note_services": list(
                set(note.service_name for note in patient_data.clinical_notes)
            ),
            "data_availability": {
                k: v.value for k, v in patient_data.data_status.items()
            },
            "total_patients_available": len(self.patient_ids),
        }

        return sample

    def sanity_check_patient_ids(self) -> Dict:
        """
        Perform comprehensive sanity check on patient ID alignment.

        Returns:
            Dictionary with detailed analysis of ID relationships
        """
        logger.info("Running patient ID sanity check...")

        # get raw IDs before any processing
        notes_ids_raw = self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
        person_ids_raw = self.person_df.index.unique()

        # convert both to same type for comparison
        notes_ids_int = set()
        notes_ids_float = set()
        conversion_errors = []

        for nid in notes_ids_raw:
            try:
                # try to convert to int
                notes_ids_int.add(int(nid))
                # also store as float for comparison
                notes_ids_float.add(float(nid))
            except (ValueError, TypeError) as e:
                conversion_errors.append(
                    {"id": str(nid), "type": type(nid).__name__, "error": str(e)}
                )

        person_ids_int = set()
        for pid in person_ids_raw:
            try:
                person_ids_int.add(int(pid))
            except (ValueError, TypeError) as e:
                conversion_errors.append(
                    {"id": str(pid), "type": type(pid).__name__, "error": str(e)}
                )

        # check subset relationships
        is_subset_int = notes_ids_int.issubset(person_ids_int)
        is_subset_float = notes_ids_float.issubset(
            set(float(p) for p in person_ids_int)
        )

        # find differences
        missing_from_person = notes_ids_int - person_ids_int
        extra_in_person = person_ids_int - notes_ids_int

        # sample ID analysis
        sample_notes_ids = list(notes_ids_raw)[:5]
        sample_person_ids = list(person_ids_raw)[:5]

        sanity_results = {
            "notes_id_count": len(notes_ids_raw),
            "person_id_count": len(person_ids_raw),
            "notes_ids_unique_int": len(notes_ids_int),
            "person_ids_unique_int": len(person_ids_int),
            "is_notes_subset_of_person": is_subset_int,
            "is_notes_subset_of_person_float": is_subset_float,
            "intersection_count": len(notes_ids_int & person_ids_int),
            "missing_from_person_count": len(missing_from_person),
            "extra_in_person_count": len(extra_in_person),
            "conversion_errors": conversion_errors[:5] if conversion_errors else [],
            "sample_analysis": {
                "notes_ids_sample": [
                    {
                        "raw": str(nid),
                        "type": type(nid).__name__,
                        "as_int": int(nid) if not pd.isna(nid) else None,
                    }
                    for nid in sample_notes_ids
                ],
                "person_ids_sample": [
                    {"raw": str(pid), "type": type(pid).__name__, "as_int": int(pid)}
                    for pid in sample_person_ids
                ],
            },
            "missing_ids_sample": (
                list(missing_from_person)[:10] if missing_from_person else []
            ),
            "data_type_info": {
                "notes_id_dtype": str(
                    self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].dtype
                ),
                "person_id_dtype": str(self.person_df.index.dtype),
            },
        }

        # log key findings
        if is_subset_int:
            logger.info("✅ PASS: All clinical note patient IDs exist in person table")
        else:
            logger.warning(
                f"⚠️ FAIL: {len(missing_from_person)} patient IDs from notes are missing in person table"
            )
            logger.warning(f"Sample missing IDs: {list(missing_from_person)[:5]}")

        return sanity_results

    def investigate_missing_demographics(self) -> Dict:
        """
        Investigate patients with notes but missing demographics.

        Returns:
            Dictionary with analysis of missing demographic data
        """
        # get patient IDs from both sources
        notes_patients = set(
            self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
        )
        person_patients = set(self.person_df.index.unique())

        # find missing demographics
        missing_demographics = notes_patients - person_patients

        if not missing_demographics:
            return {"status": "No missing demographics found"}

        # analyze notes for patients with missing demographics
        missing_df = self.clinical_notes_df[
            self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].isin(
                missing_demographics
            )
        ]

        analysis = {
            "total_missing": len(missing_demographics),
            "missing_patient_ids": sorted(list(missing_demographics))[:10],  # first 10
            "notes_count_for_missing": len(missing_df),
            "services_for_missing": missing_df["SERVICE_NAME"].value_counts().to_dict(),
            "date_range_for_missing": {
                "earliest": missing_df["PHYSIOLOGIC_TIME"].min(),
                "latest": missing_df["PHYSIOLOGIC_TIME"].max(),
            },
        }

        return analysis

    def debug_id_mismatch(self, sample_size: int = 5) -> None:
        """
        Debug helper to understand ID mismatch issues.

        Args:
            sample_size: Number of sample IDs to display
        """
        print("\n" + "=" * 60)
        print("DEBUG: Patient ID Mismatch Analysis")
        print("=" * 60)

        # get raw IDs
        notes_ids = self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
        person_ids = self.person_df.index.unique()

        print(f"\nData types:")
        print(
            f"  Notes ID dtype: {self.clinical_notes_df['OBFUSCATED_GLOBAL_PERSON_ID'].dtype}"
        )
        print(f"  Person ID dtype: {self.person_df.index.dtype}")

        print(f"\nCounts:")
        print(f"  Unique notes patients: {len(notes_ids)}")
        print(f"  Unique person records: {len(person_ids)}")

        # sample IDs for comparison
        print(f"\nSample Notes IDs (first {sample_size}):")
        for i, nid in enumerate(notes_ids[:sample_size]):
            print(
                f"  [{i}] Value: {nid}, Type: {type(nid).__name__}, As int: {int(nid)}"
            )

        print(f"\nSample Person IDs (first {sample_size}):")
        for i, pid in enumerate(person_ids[:sample_size]):
            print(f"  [{i}] Value: {pid}, Type: {type(pid).__name__}")

        # check for overlap
        notes_set = set(int(n) for n in notes_ids)
        person_set = set(int(p) for p in person_ids)

        print(f"\nSet operations (after int conversion):")
        print(f"  Intersection size: {len(notes_set & person_set)}")
        print(f"  Notes - Person: {len(notes_set - person_set)}")
        print(f"  Person - Notes: {len(person_set - notes_set)}")

        # show missing examples
        missing = notes_set - person_set
        if missing:
            print(f"\nExample IDs in notes but not in person (first {sample_size}):")
            for mid in list(missing)[:sample_size]:
                print(f"  {mid}")

        print("=" * 60 + "\n")
        """
        Perform comprehensive sanity check on patient ID alignment.
        
        Returns:
            Dictionary with detailed analysis of ID relationships
        """
        logger.info("Running patient ID sanity check...")

        # get raw IDs before any processing
        notes_ids_raw = self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
        person_ids_raw = self.person_df.index.unique()

        # convert both to same type for comparison
        notes_ids_int = set()
        notes_ids_float = set()
        conversion_errors = []

        for nid in notes_ids_raw:
            try:
                # try to convert to int
                notes_ids_int.add(int(nid))
                # also store as float for comparison
                notes_ids_float.add(float(nid))
            except (ValueError, TypeError) as e:
                conversion_errors.append(
                    {"id": str(nid), "type": type(nid).__name__, "error": str(e)}
                )

        person_ids_int = set()
        for pid in person_ids_raw:
            try:
                person_ids_int.add(int(pid))
            except (ValueError, TypeError) as e:
                conversion_errors.append(
                    {"id": str(pid), "type": type(pid).__name__, "error": str(e)}
                )

        # check subset relationships
        is_subset_int = notes_ids_int.issubset(person_ids_int)
        is_subset_float = notes_ids_float.issubset(
            set(float(p) for p in person_ids_int)
        )

        # find differences
        missing_from_person = notes_ids_int - person_ids_int
        extra_in_person = person_ids_int - notes_ids_int

        # sample ID analysis
        sample_notes_ids = list(notes_ids_raw)[:5]
        sample_person_ids = list(person_ids_raw)[:5]

        sanity_results = {
            "notes_id_count": len(notes_ids_raw),
            "person_id_count": len(person_ids_raw),
            "notes_ids_unique_int": len(notes_ids_int),
            "person_ids_unique_int": len(person_ids_int),
            "is_notes_subset_of_person": is_subset_int,
            "is_notes_subset_of_person_float": is_subset_float,
            "intersection_count": len(notes_ids_int & person_ids_int),
            "missing_from_person_count": len(missing_from_person),
            "extra_in_person_count": len(extra_in_person),
            "conversion_errors": conversion_errors[:5] if conversion_errors else [],
            "sample_analysis": {
                "notes_ids_sample": [
                    {
                        "raw": str(nid),
                        "type": type(nid).__name__,
                        "as_int": int(nid) if not pd.isna(nid) else None,
                    }
                    for nid in sample_notes_ids
                ],
                "person_ids_sample": [
                    {"raw": str(pid), "type": type(pid).__name__, "as_int": int(pid)}
                    for pid in sample_person_ids
                ],
            },
            "missing_ids_sample": (
                list(missing_from_person)[:10] if missing_from_person else []
            ),
            "data_type_info": {
                "notes_id_dtype": str(
                    self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].dtype
                ),
                "person_id_dtype": str(self.person_df.index.dtype),
            },
        }

        # log key findings
        if is_subset_int:
            logger.info("✅ PASS: All clinical note patient IDs exist in person table")
        else:
            logger.warning(
                f"⚠️ FAIL: {len(missing_from_person)} patient IDs from notes are missing in person table"
            )
            logger.warning(f"Sample missing IDs: {list(missing_from_person)[:5]}")

        return sanity_results

    def debug_id_mismatch(self, sample_size: int = 5) -> None:
        """
        Debug helper to understand ID mismatch issues.

        Args:
            sample_size: Number of sample IDs to display
        """
        print("\n" + "=" * 60)
        print("DEBUG: Patient ID Mismatch Analysis")
        print("=" * 60)

        # get raw IDs
        notes_ids = self.clinical_notes_df["OBFUSCATED_GLOBAL_PERSON_ID"].unique()
        person_ids = self.person_df.index.unique()

        print(f"\nData types:")
        print(
            f"  Notes ID dtype: {self.clinical_notes_df['OBFUSCATED_GLOBAL_PERSON_ID'].dtype}"
        )
        print(f"  Person ID dtype: {self.person_df.index.dtype}")

        print(f"\nCounts:")
        print(f"  Unique notes patients: {len(notes_ids)}")
        print(f"  Unique person records: {len(person_ids)}")

        # sample IDs for comparison
        print(f"\nSample Notes IDs (first {sample_size}):")
        for i, nid in enumerate(notes_ids[:sample_size]):
            print(
                f"  [{i}] Value: {nid}, Type: {type(nid).__name__}, As int: {int(nid)}"
            )

        print(f"\nSample Person IDs (first {sample_size}):")
        for i, pid in enumerate(person_ids[:sample_size]):
            print(f"  [{i}] Value: {pid}, Type: {type(pid).__name__}")

        # check for overlap
        notes_set = set(int(n) for n in notes_ids)
        person_set = set(int(p) for p in person_ids)

        print(f"\nSet operations (after int conversion):")
        print(f"  Intersection size: {len(notes_set & person_set)}")
        print(f"  Notes - Person: {len(notes_set - person_set)}")
        print(f"  Person - Notes: {len(person_set - notes_set)}")

        # show missing examples
        missing = notes_set - person_set
        if missing:
            print(f"\nExample IDs in notes but not in person (first {sample_size}):")
            for mid in list(missing)[:sample_size]:
                print(f"  {mid}")

        print("=" * 60 + "\n")


def main():
    """Main execution function for standalone testing"""
    print("=" * 80)
    print("irAKI Clinical Data Loader - Production Test")
    print("=" * 80)

    try:
        # initialize loader (sanity check runs automatically)
        print("\n📁 Initializing data loader...")
        loader = irAKIDataLoader(data_dir="irAKI_data", run_sanity_check=True)

        # run detailed sanity check
        print("\n🔍 Running detailed patient ID sanity check...")
        sanity_results = loader.sanity_check_patient_ids()
        print(f"Notes patients: {sanity_results['notes_id_count']}")
        print(f"Person table patients: {sanity_results['person_id_count']}")
        print(
            f"Is notes subset of person? {sanity_results['is_notes_subset_of_person']}"
        )
        print(f"Intersection count: {sanity_results['intersection_count']}")
        print(
            f"Missing from person table: {sanity_results['missing_from_person_count']}"
        )

        if sanity_results["missing_from_person_count"] > 0:
            print(f"Sample missing IDs: {sanity_results['missing_ids_sample'][:5]}")

            # optionally run debug analysis
            print("\n🐛 Running debug analysis...")
            loader.debug_id_mismatch(sample_size=3)

        # get sample output
        print("\n📊 Loading sample patient data...")
        sample = loader.get_sample_output()

        # display results (abbreviated)
        print("\n✅ Sample Output Summary:")
        print("-" * 40)
        print(f"Patient ID: {sample['patient_summary']['person_id']}")
        print(f"Age: {sample['demographics']['age']}")
        print(f"Gender: {sample['demographics']['gender']}")
        print(f"Total Notes: {sample['total_notes']}")
        print(
            f"Date Range: {sample['patient_summary']['date_range']['days_span']} days"
        )
        print(f"Data Completeness: {sample['patient_summary']['data_completeness']}")

        # show available patients (now properly typed as int64)
        print(f"\n📈 Total patients available: {len(loader.patient_ids)}")
        print(f"First 5 patient IDs: {loader.patient_ids[:5]}")

        # export sanity check results
        sanity_file = Path("exported_data") / "sanity_check_results.json"
        sanity_file.parent.mkdir(exist_ok=True)
        with open(sanity_file, "w") as f:
            json.dump(sanity_results, f, indent=2, default=str)
        print(f"\n📄 Sanity check results exported to: {sanity_file}")

        print("\n✨ Data loader test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
