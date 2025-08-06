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
   - PHYSIOLOGIC_TIME column is converted to datetime format
   - Contains fields: OBFUSCATED_GLOBAL_PERSON_ID, ENCOUNTER_ID, SERVICE_NAME,
     PHYSIOLOGIC_TIME, REPORT_TEXT
   - Each row represents a single clinical note entry

2. Person Demographics (r6335_person.csv):
   - Standard CSV format with patient demographic information
   - Loaded using pd.read_csv() with automatic type inference
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
- Only patients with both clinical notes AND demographics are included in final cohort
- Data is cached in memory using dictionary storage for repeated access
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
import warnings
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import pickle

# suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        data['note_datetime'] = self.note_datetime.isoformat() if self.note_datetime else None
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
            'person_id': self.person_id,
            'age': self.demographics.age,
            'gender': self.demographics.gender,
            'total_notes': len(self.clinical_notes),
            'date_range': self._get_date_range(),
            'data_completeness': self._calculate_completeness()
        }

    def _get_date_range(self) -> Dict:
        """Get the date range of available data"""
        if not self.clinical_notes:
            return {'start': None, 'end': None}

        dates = [note.note_datetime for note in self.clinical_notes if note.note_datetime]
        if not dates:
            return {'start': None, 'end': None}

        return {
            'start': min(dates).isoformat(),
            'end': max(dates).isoformat(),
            'days_span': (max(dates) - min(dates)).days
        }

    def _calculate_completeness(self) -> float:
        """Calculate data completeness score"""
        scores = {
            'demographics': 1.0 if self.data_status.get('demographics') == DataStatus.AVAILABLE else 0.0,
            'notes': 1.0 if len(self.clinical_notes) > 0 else 0.0,
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
    GENDER_CONCEPTS = {
        8507: 'Male',
        8532: 'Female',
        8551: 'Unknown',
        8570: 'Ambiguous'
    }

    def __init__(self, data_dir: str = "irAKI_data", cache_dir: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory
            cache_dir: Optional directory for caching processed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / '.cache'
        self.cache_dir.mkdir(exist_ok=True)

        # data containers
        self.clinical_notes_df = None
        self.person_df = None

        # patient index
        self.patient_ids = []
        self.patient_data_cache = {}

        # load data
        self._load_all_data()

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
        notes_path = self.data_dir / 'clinical_notes_version3' / 'part-00000-347607e6-182c-419e-9a83-8a38b90986fa-c000.snappy.parquet'

        if not notes_path.exists():
            raise FileNotFoundError(f"Clinical notes file not found: {notes_path}")

        logger.info("Loading clinical notes...")
        self.clinical_notes_df = pd.read_parquet(notes_path)

        # standardize column names to uppercase
        self.clinical_notes_df.columns = [col.upper() for col in self.clinical_notes_df.columns]

        # convert timestamps
        if 'PHYSIOLOGIC_TIME' in self.clinical_notes_df.columns:
            self.clinical_notes_df['PHYSIOLOGIC_TIME'] = pd.to_datetime(
                self.clinical_notes_df['PHYSIOLOGIC_TIME'],
                errors='coerce'
            )

        logger.info(f"Loaded {len(self.clinical_notes_df)} clinical notes")

    def _load_person_data(self) -> None:
        """Load person demographics from CSV"""
        person_path = self.data_dir / 'structured_data' / 'r6335_person.csv'

        if not person_path.exists():
            raise FileNotFoundError(f"Person data file not found: {person_path}")

        logger.info("Loading person demographics...")
        self.person_df = pd.read_csv(person_path)

        # standardize column names to lowercase for this dataset
        self.person_df.columns = [col.lower() for col in self.person_df.columns]

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

        # get unique patient IDs from clinical notes
        notes_patients = set(self.clinical_notes_df['OBFUSCATED_GLOBAL_PERSON_ID'].unique())

        # get unique patient IDs from person table
        person_patients = set(self.person_df['person_id'].unique())

        # use intersection to ensure we have both notes and demographics
        self.patient_ids = sorted(list(notes_patients & person_patients))

        logger.info(f"Found {len(self.patient_ids)} patients with both notes and demographics")

        if len(notes_patients - person_patients) > 0:
            logger.warning(f"{len(notes_patients - person_patients)} patients have notes but no demographics")

        if len(person_patients - notes_patients) > 0:
            logger.warning(f"{len(person_patients - notes_patients)} patients have demographics but no notes")

    def get_patient_demographics(self, person_id: int) -> Optional[PatientDemographics]:
        """
        Get demographics for a specific patient.

        Args:
            person_id: Patient identifier

        Returns:
            PatientDemographics object or None if not found
        """
        person_row = self.person_df[self.person_df['person_id'] == person_id]

        if person_row.empty:
            logger.warning(f"No demographics found for patient {person_id}")
            return None

        person = person_row.iloc[0]

        # calculate age
        current_year = datetime.now().year
        year_of_birth = person.get('year_of_birth', current_year - 65)
        age = current_year - year_of_birth

        # map gender using OMOP standard concepts
        gender_concept = person.get('gender_concept_id', 8551)
        gender = self.GENDER_CONCEPTS.get(gender_concept, 'Unknown')

        # get race and ethnicity using source values
        race = person.get('race_source_value', 'Unknown')
        ethnicity = person.get('ethnicity_source_value', 'Unknown')

        return PatientDemographics(
            person_id=person_id,
            age=age,
            gender=gender,
            race=race if race and race != 'nan' else 'Unknown',
            ethnicity=ethnicity if ethnicity and ethnicity != 'nan' else 'Unknown',
            year_of_birth=year_of_birth
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
            self.clinical_notes_df['OBFUSCATED_GLOBAL_PERSON_ID'] == person_id
        ].copy()

        if patient_notes.empty:
            logger.warning(f"No clinical notes found for patient {person_id}")
            return []

        # sort by time
        patient_notes = patient_notes.sort_values('PHYSIOLOGIC_TIME')

        notes = []
        for idx, row in patient_notes.iterrows():
            note = ClinicalNote(
                note_id=f"{person_id}_{idx}",
                person_id=person_id,
                encounter_id=str(row.get('ENCOUNTER_ID', '')),
                service_name=row.get('SERVICE_NAME', 'Unknown'),
                note_datetime=row.get('PHYSIOLOGIC_TIME'),
                note_text=row.get('REPORT_TEXT', ''),
                text_length=len(str(row.get('REPORT_TEXT', '')))
            )
            notes.append(note)

        return notes

    def load_patient(self, person_id: int, use_cache: bool = True) -> Optional[PatientData]:
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
            'demographics': DataStatus.AVAILABLE if demographics else DataStatus.MISSING,
            'clinical_notes': DataStatus.AVAILABLE if clinical_notes else DataStatus.MISSING,
            'lab_results': DataStatus.MISSING,  # disabled for now
            'medications': DataStatus.MISSING   # disabled for now
        }

        # create patient data object
        patient_data = PatientData(
            person_id=person_id,
            demographics=demographics,
            clinical_notes=clinical_notes,
            data_status=data_status
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

    def export_patient_data(self, person_id: int, output_dir: str = "exported_data") -> str:
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
            'person_id': patient_data.person_id,
            'demographics': patient_data.demographics.to_dict(),
            'clinical_notes': [note.to_dict() for note in patient_data.clinical_notes],
            'summary_stats': patient_data.get_summary_stats(),
            'data_status': {k: v.value for k, v in patient_data.data_status.items()}
        }

        # save to JSON
        output_file = output_path / f"patient_{person_id}.json"
        with open(output_file, 'w') as f:
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
            "first_note": patient_data.clinical_notes[0].to_dict() if patient_data.clinical_notes else None,
            "total_notes": len(patient_data.clinical_notes),
            "note_services": list(set(note.service_name for note in patient_data.clinical_notes)),
            "data_availability": {k: v.value for k, v in patient_data.data_status.items()},
            "total_patients_available": len(self.patient_ids)
        }

        return sample


def main():
    """Main execution function for standalone testing"""
    print("=" * 80)
    print("irAKI Clinical Data Loader - Production Test")
    print("=" * 80)

    try:
        # initialize loader
        print("\n📁 Initializing data loader...")
        loader = irAKIDataLoader(data_dir="irAKI_data")

        # get sample output
        print("\n📊 Loading sample patient data...")
        sample = loader.get_sample_output()

        # display results
        print("\n✅ Sample Output:")
        print("-" * 40)
        print(json.dumps(sample, indent=2, default=str))

        # show available patients
        print(f"\n📈 Total patients available: {len(loader.patient_ids)}")
        print(f"First 5 patient IDs: {loader.patient_ids[:5]}")

        # export first patient (optional)
        if loader.patient_ids:
            print("\n💾 Exporting first patient data...")
            export_path = loader.export_patient_data(loader.patient_ids[0])
            print(f"Exported to: {export_path}")

        print("\n✨ Data loader test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())