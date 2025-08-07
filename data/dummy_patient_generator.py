"""
Dummy data generator for DelPHEA-irAKI testing.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List


class DummyDataGenerator:
    """Generate realistic dummy patient data for testing."""

    # Template clinical scenarios
    CLINICAL_SCENARIOS = {
        "classic_iraki": {
            "summary": "Classic immune-related AKI case with clear temporal relationship to ICI",
            "medications": {
                "nivolumab": {"start_date": "2024-01-15", "ongoing": True},
                "ipilimumab": {"start_date": "2024-01-15", "ongoing": True},
            },
            "lab_pattern": {
                "creatinine_baseline": 1.1,
                "creatinine_peak": 3.2,
                "timeline": "2 weeks",
            },
            "notes": [
                {
                    "service": "Oncology",
                    "text": (
                        "67-year-old female with metastatic melanoma on combination "
                        "immunotherapy (nivolumab + ipilimumab). Patient presenting with "
                        "elevated creatinine, increased from baseline 1.1 to 3.2 mg/dL over "
                        "2 weeks. No recent NSAID use or contrast exposure."
                    ),
                },
                {
                    "service": "Nephrology",
                    "text": (
                        "Consultation for AKI. Creatinine trending upward. Urinalysis shows "
                        "proteinuria 2+, RBC 10-20/hpf. Suspect possible immune-related "
                        "nephritis given recent ICI therapy."
                    ),
                },
            ],
        },
        "atn_case": {
            "summary": "ATN from sepsis, not irAKI",
            "medications": {
                "pembrolizumab": {"start_date": "2023-10-01", "ongoing": True},
            },
            "lab_pattern": {
                "creatinine_baseline": 0.9,
                "creatinine_peak": 4.5,
                "timeline": "3 days",
            },
            "notes": [
                {
                    "service": "ICU",
                    "text": (
                        "58-year-old male with NSCLC on pembrolizumab admitted with septic shock "
                        "secondary to pneumonia. Hypotensive requiring pressors. Acute kidney injury "
                        "with creatinine rise from 0.9 to 4.5. Muddy brown casts on urinalysis."
                    ),
                },
            ],
        },
        "prerenal_case": {
            "summary": "Prerenal AKI from volume depletion",
            "medications": {
                "atezolizumab": {"start_date": "2024-02-01", "ongoing": True},
            },
            "lab_pattern": {
                "creatinine_baseline": 1.0,
                "creatinine_peak": 2.1,
                "timeline": "4 days",
            },
            "notes": [
                {
                    "service": "Emergency",
                    "text": (
                        "72-year-old with urothelial cancer on atezolizumab. Presents with "
                        "severe diarrhea x5 days from ICI-colitis. Volume depleted on exam. "
                        "Creatinine 2.1 from baseline 1.0. FeNa <1%. Improving with IVF."
                    ),
                },
            ],
        },
    }

    @classmethod
    def generate_patient_case(cls, case_id: str, scenario: str = None) -> Dict:
        """
        Generate a dummy patient case.

        Args:
            case_id: Case identifier
            scenario: Optional scenario type ('classic_iraki', 'atn_case', 'prerenal_case')

        Returns:
            Dict: Complete patient case data
        """
        # Extract patient ID from case_id
        try:
            if "iraki_case_" in case_id:
                patient_id = int(case_id.replace("iraki_case_", ""))
            else:
                patient_id = 99999
        except:
            patient_id = 99999

        # Select scenario
        if scenario and scenario in cls.CLINICAL_SCENARIOS:
            template = cls.CLINICAL_SCENARIOS[scenario]
        else:
            # Default to classic irAKI or random
            template = cls.CLINICAL_SCENARIOS.get(
                scenario, cls.CLINICAL_SCENARIOS["classic_iraki"]
            )

        # Generate demographics
        demographics = cls._generate_demographics(patient_id)

        # Generate clinical notes with timestamps
        clinical_notes = cls._generate_clinical_notes(template["notes"])

        # Build complete case
        return {
            "case_id": case_id,
            "patient_info": demographics,
            "patient_summary": (
                f"Patient {patient_id}: {demographics['age']}-year-old "
                f"{demographics['gender']}. {template['summary']}"
            ),
            "clinical_notes": clinical_notes,
            "total_notes": len(clinical_notes),
            "data_completeness": 1.0,
            "medication_history": template["medications"],
            "lab_values": cls._generate_lab_values(template["lab_pattern"]),
            "imaging_reports": cls._generate_imaging_reports(),
        }

    @staticmethod
    def _generate_demographics(patient_id: int) -> Dict:
        """Generate patient demographics."""
        random.seed(patient_id)  # Consistent demographics for same ID

        age = random.randint(45, 85)
        gender = random.choice(["Male", "Female"])
        race = random.choice(["White", "Black", "Asian", "Hispanic"])
        ethnicity = random.choice(["Hispanic", "Non-Hispanic"])

        return {
            "person_id": patient_id,
            "age": age,
            "gender": gender,
            "race": race,
            "ethnicity": ethnicity,
        }

    @staticmethod
    def _generate_clinical_notes(note_templates: List[Dict]) -> List[Dict]:
        """Generate clinical notes with proper timestamps."""
        notes = []
        base_time = datetime.now() - timedelta(days=7)

        for i, template in enumerate(note_templates):
            note_time = base_time + timedelta(days=i)
            notes.append(
                {
                    "timestamp": note_time.isoformat(),
                    "service": template["service"],
                    "encounter_id": f"ENC{str(i+1).zfill(3)}",
                    "text": template["text"],
                }
            )

        return notes

    @staticmethod
    def _generate_lab_values(lab_pattern: Dict) -> Dict:
        """Generate lab values based on pattern."""
        labs = {
            "creatinine_baseline": lab_pattern["creatinine_baseline"],
            "creatinine_peak": lab_pattern["creatinine_peak"],
            "egfr_baseline": int(60 / lab_pattern["creatinine_baseline"]),
            "egfr_current": int(60 / lab_pattern["creatinine_peak"]),
        }

        # Add urinalysis if irAKI pattern
        if lab_pattern["creatinine_peak"] / lab_pattern["creatinine_baseline"] > 2:
            labs["urinalysis"] = {
                "proteinuria": "2+",
                "RBC": "10-20/hpf",
                "WBC": "5-10/hpf",
                "eosinophils": "absent",
            }

        return labs

    @staticmethod
    def _generate_imaging_reports() -> List[str]:
        """Generate imaging reports."""
        return [
            "Renal ultrasound: No hydronephrosis or obstruction. "
            "Kidneys normal size with preserved corticomedullary differentiation."
        ]
