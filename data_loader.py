"""
irAKI Data Loader
==========================
Production-ready data loader for immune-related AKI patient data.
Processes clinical notes and demographics with placeholders for labs/medications.
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

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Configure logging
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
class LabResult:
    """Structured lab result - placeholder for future implementation"""
    measurement_id: int
    person_id: int
    measurement_date: datetime
    measurement_concept_id: int
    value_as_number: Optional[float]
    unit_concept_id: Optional[int]
    measurement_source_value: str

    # TODO: Add concept name mapping when concept.csv is available
    measurement_name: str = "Unknown"
    unit_name: str = "Unknown"


@dataclass
class Medication:
    """Structured medication record - placeholder for future implementation"""
    drug_exposure_id: int
    person_id: int
    drug_concept_id: int
    start_date: datetime
    end_date: Optional[datetime]
    drug_source_value: str

    # TODO: Add concept name mapping when concept.csv is available
    drug_name: str = "Unknown"
    is_ici: bool = False
    is_nephrotoxic: bool = False


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
    lab_results: List[LabResult] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    data_status: Dict[str, DataStatus] = field(default_factory=dict)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for this patient"""
        return {
            'person_id': self.person_id,
            'age': self.demographics.age,
            'gender': self.demographics.gender,
            'total_notes': len(self.clinical_notes),
            'total_labs': len(self.lab_results),
            'total_medications': len(self.medications),
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
            'labs': 0.5 if len(self.lab_results) > 0 else 0.0,  # Weighted less since incomplete
            'medications': 0.5 if len(self.medications) > 0 else 0.0  # Weighted less since incomplete
        }
        return sum(scores.values()) / len(scores)


class irAKIDataLoader:
    """
    Production-ready data loader for irAKI clinical data.

    Handles loading and processing of:
    - Clinical notes (fully implemented)
    - Demographics (fully implemented)
    - Lab results (placeholder - awaiting concept mapping)
    - Medications (placeholder - awaiting concept mapping)
    """

    # Standard OMOP concept IDs (for when concept.csv is available)
    GENDER_CONCEPTS = {
        8507: 'Male',
        8532: 'Female',
        8551: 'Unknown',
        8570: 'Ambiguous'
    }

    # ICI drug keywords for identification
    ICI_KEYWORDS = [
        'nivolumab', 'opdivo',
        'pembrolizumab', 'keytruda',
        'ipilimumab', 'yervoy',
        'atezolizumab', 'tecentriq',
        'durvalumab', 'imfinzi',
        'avelumab', 'bavencio',
        'cemiplimab', 'libtayo'
    ]

    # Nephrotoxic drug keywords
    NEPHROTOXIC_KEYWORDS = [
        'nsaid', 'ibuprofen', 'naproxen', 'diclofenac',
        'ppi', 'omeprazole', 'pantoprazole', 'lansoprazole',
        'antibiotic', 'vancomycin', 'aminoglycoside', 'gentamicin'
    ]

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

        # Data containers
        self.clinical_notes_df = None
        self.person_df = None
        self.measurements_df = None
        self.drug_exposure_df = None

        # Patient index
        self.patient_ids = []
        self.patient_data_cache = {}

        # Load data
        self._load_all_data()

    def _load_all_data(self) -> None:
        """Load all available data files"""
        logger.info("Starting data loading process...")

        # Load clinical notes (required)
        self._load_clinical_notes()

        # Load person demographics (required)
        self._load_person_data()

        # Load measurements (optional)
        self._load_measurements()

        # Load drug exposures (optional)
        self._load_drug_exposures()

        # Build patient index
        self._build_patient_index()

        logger.info(f"Data loading complete. Found {len(self.patient_ids)} patients.")

    def _load_clinical_notes(self) -> None:
        """Load clinical notes from parquet file"""
        notes_path = self.data_dir / 'clinical_notes_version3' / 'part-00000-347607e6-182c-419e-9a83-8a38b90986fa-c000.snappy.parquet'

        if not notes_path.exists():
            raise FileNotFoundError(f"Clinical notes file not found: {notes_path}")

        logger.info("Loading clinical notes...")
        self.clinical_notes_df = pd.read_parquet(notes_path)

        # Standardize column names
        self.clinical_notes_df.columns = [col.upper() for col in self.clinical_notes_df.columns]

        # Convert timestamps
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

        # Standardize column names
        self.person_df.columns = [col.lower() for col in self.person_df.columns]

        logger.info(f"Loaded {len(self.person_df)} person records")

    def _load_measurements(self) -> None:
        """Load measurements data if available"""
        measurements_path = self.data_dir / 'structured_data' / 'r6335_measurement.csv'

        if not measurements_path.exists():
            logger.warning("Measurements file not found - lab results will not be available")
            self.measurements_df = pd.DataFrame()
            return

        logger.info("Loading measurements...")
        try:
            self.measurements_df = pd.read_csv(measurements_path)
            # Standardize column names
            self.measurements_df.columns = [col.upper() for col in self.measurements_df.columns]

            # Convert dates
            if 'MEASUREMENT_DATE' in self.measurements_df.columns:
                self.measurements_df['MEASUREMENT_DATE'] = pd.to_datetime(
                    self.measurements_df['MEASUREMENT_DATE'],
                    errors='coerce'
                )

            logger.info(f"Loaded {len(self.measurements_df)} measurement records")

            # TODO: When concept.csv is available, add concept mapping here
            logger.warning("Concept mapping not available - lab names will be concept IDs only")

        except Exception as e:
            logger.error(f"Error loading measurements: {e}")
            self.measurements_df = pd.DataFrame()

    def _load_drug_exposures(self) -> None:
        """Load drug exposure data if available"""
        drug_path = self.data_dir / 'structured_data' / 'r6335_drug_exposure.csv'

        if not drug_path.exists():
            logger.warning("Drug exposure file not found - medications will not be available")
            self.drug_exposure_df = pd.DataFrame()
            return

        logger.info("Loading drug exposures...")
        try:
            self.drug_exposure_df = pd.read_csv(drug_path)
            # Standardize column names
            self.drug_exposure_df.columns = [col.lower() for col in self.drug_exposure_df.columns]

            # Convert dates (handling the special format)
            for date_col in ['drug_exposure_start_date', 'drug_exposure_end_date']:
                if date_col in self.drug_exposure_df.columns:
                    self.drug_exposure_df[date_col] = pd.to_datetime(
                        self.drug_exposure_df[date_col],
                        format='%d%b%Y',
                        errors='coerce'
                    )

            logger.info(f"Loaded {len(self.drug_exposure_df)} drug exposure records")

            # TODO: When concept.csv is available, add concept mapping here
            logger.warning("Concept mapping not available - drug names will be from source values only")

        except Exception as e:
            logger.error(f"Error loading drug exposures: {e}")
            self.drug_exposure_df = pd.DataFrame()

    def _build_patient_index(self) -> None:
        """Build index of all unique patient IDs"""
        logger.info("Building patient index...")

        # Get unique patient IDs from clinical notes
        notes_patients = set(self.clinical_notes_df['OBFUSCATED_GLOBAL_PERSON_ID'].unique())

        # Get unique patient IDs from person table
        person_patients = set(self.person_df['person_id'].unique())

        # Use intersection to ensure we have both notes and demographics
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

        # Calculate age
        current_year = datetime.now().year
        year_of_birth = person.get('year_of_birth', current_year - 65)
        age = current_year - year_of_birth

        # Map gender (using OMOP standard concepts)
        gender_concept = person.get('gender_concept_id', 8551)
        gender = self.GENDER_CONCEPTS.get(gender_concept, 'Unknown')

        # Get race and ethnicity (using source values for now)
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

        # Sort by time
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

    def get_patient_labs(self, person_id: int) -> List[LabResult]:
        """
        Get lab results for a patient.

        TODO: Implement full lab extraction when concept mapping is available.
        Currently returns empty list or basic structure.

        Args:
            person_id: Patient identifier

        Returns:
            List of LabResult objects (currently empty/placeholder)
        """
        if self.measurements_df.empty:
            return []

        patient_labs = self.measurements_df[
            self.measurements_df['PERSON_ID'] == person_id
            ].copy()

        if patient_labs.empty:
            return []

        labs = []
        # TODO: Implement when concept mapping is available
        # For now, just structure the data
        for idx, row in patient_labs.head(10).iterrows():  # Limit to 10 for demo
            lab = LabResult(
                measurement_id=row.get('MEASUREMENT_ID', 0),
                person_id=person_id,
                measurement_date=row.get('MEASUREMENT_DATE'),
                measurement_concept_id=row.get('MEASUREMENT_CONCEPT_ID', 0),
                value_as_number=row.get('VALUE_AS_NUMBER'),
                unit_concept_id=row.get('UNIT_CONCEPT_ID'),
                measurement_source_value=str(row.get('MEASUREMENT_SOURCE_VALUE', ''))
            )
            labs.append(lab)

        logger.info(f"Loaded {len(labs)} lab results for patient {person_id} (placeholder implementation)")
        return labs

    def get_patient_medications(self, person_id: int) -> List[Medication]:
        """
        Get medications for a patient.

        TODO: Implement full medication extraction when concept mapping is available.
        Currently returns empty list or basic structure.

        Args:
            person_id: Patient identifier

        Returns:
            List of Medication objects (currently empty/placeholder)
        """
        if self.drug_exposure_df.empty:
            return []

        patient_meds = self.drug_exposure_df[
            self.drug_exposure_df['person_id'] == person_id
            ].copy()

        if patient_meds.empty:
            return []

        medications = []
        # TODO: Implement when concept mapping is available
        # For now, just structure the data
        for idx, row in patient_meds.head(10).iterrows():  # Limit to 10 for demo
            drug_source = str(row.get('drug_source_value', '')).lower()

            # Try to identify ICIs and nephrotoxic drugs from source value
            is_ici = any(keyword in drug_source for keyword in self.ICI_KEYWORDS)
            is_nephrotoxic = any(keyword in drug_source for keyword in self.NEPHROTOXIC_KEYWORDS)

            med = Medication(
                drug_exposure_id=row.get('drug_exposure_id', 0),
                person_id=person_id,
                drug_concept_id=row.get('drug_concept_id', 0),
                start_date=row.get('drug_exposure_start_date'),
                end_date=row.get('drug_exposure_end_date'),
                drug_source_value=drug_source,
                drug_name=drug_source,  # Using source value as name for now
                is_ici=is_ici,
                is_nephrotoxic=is_nephrotoxic
            )
            medications.append(med)

        logger.info(f"Loaded {len(medications)} medications for patient {person_id} (placeholder implementation)")
        return medications

    def load_patient(self, person_id: int, use_cache: bool = True) -> Optional[PatientData]:
        """
        Load complete data for a patient.

        Args:
            person_id: Patient identifier
            use_cache: Whether to use cached data if available

        Returns:
            PatientData object or None if patient not found
        """
        # Check cache first
        if use_cache and person_id in self.patient_data_cache:
            logger.debug(f"Loading patient {person_id} from cache")
            return self.patient_data_cache[person_id]

        # Check if patient exists
        if person_id not in self.patient_ids:
            logger.error(f"Patient {person_id} not found in database")
            return None

        logger.info(f"Loading data for patient {person_id}")

        # Load demographics
        demographics = self.get_patient_demographics(person_id)
        if not demographics:
            logger.error(f"Failed to load demographics for patient {person_id}")
            return None

        # Load clinical notes
        clinical_notes = self.get_patient_notes(person_id)

        # Load labs (placeholder)
        lab_results = self.get_patient_labs(person_id)

        # Load medications (placeholder)
        medications = self.get_patient_medications(person_id)

        # Determine data status
        data_status = {
            'demographics': DataStatus.AVAILABLE if demographics else DataStatus.MISSING,
            'clinical_notes': DataStatus.AVAILABLE if clinical_notes else DataStatus.MISSING,
            'lab_results': DataStatus.PARTIAL if lab_results else DataStatus.MISSING,
            'medications': DataStatus.PARTIAL if medications else DataStatus.MISSING
        }

        # Create patient data object
        patient_data = PatientData(
            person_id=person_id,
            demographics=demographics,
            clinical_notes=clinical_notes,
            lab_results=lab_results,
            medications=medications,
            data_status=data_status
        )

        # Cache the data
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

        # Convert to exportable format
        export_data = {
            'person_id': patient_data.person_id,
            'demographics': patient_data.demographics.to_dict(),
            'clinical_notes': [note.to_dict() for note in patient_data.clinical_notes],
            'summary_stats': patient_data.get_summary_stats(),
            'data_status': {k: v.value for k, v in patient_data.data_status.items()}
        }

        # Save to JSON
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

        # Load first patient
        first_patient_id = self.patient_ids[0]
        patient_data = self.load_patient(first_patient_id)

        if not patient_data:
            return {"error": f"Failed to load patient {first_patient_id}"}

        # Create sample output
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
        # Initialize loader
        print("\n📁 Initializing data loader...")
        loader = irAKIDataLoader(data_dir="irAKI_data")

        # Get sample output
        print("\n📊 Loading sample patient data...")
        sample = loader.get_sample_output()

        # Display results
        print("\n✅ Sample Output:")
        print("-" * 40)
        print(json.dumps(sample, indent=2, default=str))

        # Show available patients
        print(f"\n📈 Total patients available: {len(loader.patient_ids)}")
        print(f"First 5 patient IDs: {loader.patient_ids[:5]}")

        # Export first patient (optional)
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
