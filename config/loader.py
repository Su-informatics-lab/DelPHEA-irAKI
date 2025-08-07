"""
Configuration Loader Module for DelPHEA-irAKI
==============================================

Centralized configuration management with fail-fast validation for expert panels,
questionnaires, and prompt templates. Ensures configuration integrity at startup.

Module Architecture:
-------------------
    ┌───────────────────────────────┐
    │   Configuration Files         │
    ├───────────────────────────────┤
    │ • config/panel.json           │──┐
    │ • config/questionnaire.json   │──┼──> ConfigurationLoader
    │ • prompts/*.json              │──┘         │
    └───────────────────────────────┘           │
                                                 ▼
                                          ┌──────────────┐
                                          │  Validation  │
                                          │   & Cache    │
                                          └──────────────┘
                                                 │
                ┌────────────────────────────────┼────────────────────────────┐
                ▼                                ▼                            ▼
        Expert Profiles               Assessment Questions          Prompt Templates
        (11 specialties)               (16 irAKI factors)            (3+ templates)

Key Features:
------------
1. **Fail-Fast Validation**: All configs validated at startup
2. **Centralized Access**: Single source of truth for all configuration
3. **Type Safety**: Strict validation of required fields
4. **Caching**: Configurations loaded once and cached
5. **Expert Management**: Dynamic expert panel configuration

Note: This module also includes a simple DataLoaderWrapper that delegates to
the existing dataloader.py for patient data (YAGNI principle - no redundant code).

Clinical Context:
----------------
The configuration defines the virtual expert panel composition, clinical
assessment questions, and prompting strategies essential for accurate
irAKI classification consensus.

Nature Medicine Standards:
-------------------------
Configuration transparency ensures reproducible expert panel composition
and assessment criteria for publication requirements.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.core import RuntimeConfig

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """
    Centralized configuration loading with comprehensive validation.

    This class ensures all configuration files are present, valid, and
    contain required fields before the system starts processing cases.
    Follows fail-loud principles per Nature Medicine standards.
    """

    def __init__(self, runtime_config: RuntimeConfig):
        """
        Initialize configuration loader with validation.

        Args:
            runtime_config: Runtime configuration with file paths

        Raises:
            FileNotFoundError: If any required configuration file is missing
            ValueError: If configuration structure is invalid
            KeyError: If required fields are missing
        """
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.ConfigurationLoader")

        # load and validate all configurations at startup
        # fail fast if any configuration is invalid
        self.logger.info("Loading and validating configurations...")

        self._expert_panel = self._load_expert_panel()
        self._questionnaire = self._load_questionnaire()
        self._prompts = self._load_prompts()

        # create quick lookup indices
        self._expert_lookup = {
            expert["id"]: expert
            for expert in self._expert_panel["expert_panel"]["experts"]
        }
        self._question_lookup = {
            q["id"]: q for q in self._questionnaire["questionnaire"]["questions"]
        }

        self.logger.info(
            f"Configuration loaded successfully: "
            f"{len(self._expert_lookup)} experts, "
            f"{len(self._question_lookup)} questions, "
            f"{len(self._prompts)} prompt templates"
        )

    def _load_expert_panel(self) -> Dict:
        """
        Load and validate expert panel configuration.

        Returns:
            Dict: Validated expert panel configuration

        Raises:
            FileNotFoundError: If panel config file doesn't exist
            ValueError: If panel structure is invalid
        """
        panel_path = Path(self.config.expert_panel_config)
        if not panel_path.exists():
            raise FileNotFoundError(
                f"Expert panel config not found: {panel_path}. "
                f"Expected at: {panel_path.absolute()}"
            )

        try:
            with open(panel_path, "r", encoding="utf-8") as f:
                panel_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in panel config {panel_path}: {e}")

        # validate top-level structure
        if "expert_panel" not in panel_config:
            raise ValueError(
                "Expert panel config missing required 'expert_panel' key. "
                f"Found keys: {list(panel_config.keys())}"
            )

        if "experts" not in panel_config["expert_panel"]:
            raise ValueError(
                "Expert panel config missing 'experts' list. "
                f"Found keys: {list(panel_config['expert_panel'].keys())}"
            )

        experts = panel_config["expert_panel"]["experts"]
        if not experts:
            raise ValueError("Expert panel config has empty experts list")

        # validate each expert has required fields
        required_fields = ["id", "specialty", "name", "experience_years", "expertise"]
        for i, expert in enumerate(experts):
            missing_fields = [f for f in required_fields if f not in expert]
            if missing_fields:
                raise ValueError(
                    f"Expert {i} (id={expert.get('id', 'MISSING')}) missing "
                    f"required fields: {missing_fields}"
                )

            # validate field types
            if not isinstance(expert["experience_years"], (int, float)):
                raise ValueError(
                    f"Expert {expert['id']} has invalid experience_years: "
                    f"{expert['experience_years']} (must be numeric)"
                )

            if not isinstance(expert["expertise"], list):
                raise ValueError(
                    f"Expert {expert['id']} has invalid expertise: "
                    f"must be a list of strings"
                )

        # check for duplicate expert IDs
        expert_ids = [e["id"] for e in experts]
        if len(expert_ids) != len(set(expert_ids)):
            duplicates = [x for x in expert_ids if expert_ids.count(x) > 1]
            raise ValueError(f"Duplicate expert IDs found: {set(duplicates)}")

        self.logger.info(
            f"Loaded {len(experts)} experts from {panel_path}: "
            f"{', '.join(e['specialty'] for e in experts[:3])}..."
        )

        return panel_config

    def _load_questionnaire(self) -> Dict:
        """
        Load and validate questionnaire configuration.

        Returns:
            Dict: Validated questionnaire configuration

        Raises:
            FileNotFoundError: If questionnaire config doesn't exist
            ValueError: If questionnaire structure is invalid
        """
        questionnaire_path = Path(self.config.questionnaire_config)
        if not questionnaire_path.exists():
            raise FileNotFoundError(
                f"Questionnaire config not found: {questionnaire_path}. "
                f"Expected at: {questionnaire_path.absolute()}"
            )

        try:
            with open(questionnaire_path, "r", encoding="utf-8") as f:
                questionnaire_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in questionnaire config: {e}")

        # validate structure
        if "questionnaire" not in questionnaire_config:
            raise ValueError(
                "Questionnaire config missing 'questionnaire' key. "
                f"Found keys: {list(questionnaire_config.keys())}"
            )

        if "questions" not in questionnaire_config["questionnaire"]:
            raise ValueError("Questionnaire config missing 'questions' list")

        questions = questionnaire_config["questionnaire"]["questions"]
        if not questions:
            raise ValueError("Questionnaire has empty questions list")

        # validate each question
        required_fields = ["id", "question", "clinical_context"]
        for i, question in enumerate(questions):
            missing_fields = [f for f in required_fields if f not in question]
            if missing_fields:
                raise ValueError(
                    f"Question {i} (id={question.get('id', 'MISSING')}) "
                    f"missing required fields: {missing_fields}"
                )

            # validate clinical context structure
            context = question["clinical_context"]
            if not isinstance(context, dict):
                raise ValueError(
                    f"Question {question['id']} has invalid clinical_context: "
                    f"must be a dictionary"
                )

            # check for important clinical context fields
            if "importance" not in context:
                self.logger.warning(
                    f"Question {question['id']} missing 'importance' in clinical_context"
                )

        # check for duplicate question IDs
        question_ids = [q["id"] for q in questions]
        if len(question_ids) != len(set(question_ids)):
            duplicates = [x for x in question_ids if question_ids.count(x) > 1]
            raise ValueError(f"Duplicate question IDs found: {set(duplicates)}")

        self.logger.info(
            f"Loaded {len(questions)} assessment questions from {questionnaire_path}"
        )

        return questionnaire_config

    def _load_prompts(self) -> Dict[str, Dict]:
        """
        Load all prompt templates from prompts directory.

        Returns:
            Dict: Mapping of template names to prompt configurations

        Raises:
            FileNotFoundError: If prompts directory doesn't exist
            ValueError: If required prompts are missing
        """
        prompts_dir = Path(self.config.prompts_dir)
        if not prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found: {prompts_dir}. "
                f"Expected at: {prompts_dir.absolute()}"
            )

        if not prompts_dir.is_dir():
            raise ValueError(f"Prompts path is not a directory: {prompts_dir}")

        prompts = {}
        json_files = list(prompts_dir.glob("*.json"))

        if not json_files:
            raise ValueError(f"No JSON prompt files found in {prompts_dir}")

        for prompt_file in json_files:
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_content = json.load(f)

                # validate prompt has required structure
                if "base_prompt" not in prompt_content:
                    self.logger.warning(
                        f"Prompt file {prompt_file.name} missing 'base_prompt' field"
                    )

                prompts[prompt_file.stem] = prompt_content

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in prompt file {prompt_file}: {e}")
            except Exception as e:
                raise ValueError(f"Error loading prompt file {prompt_file}: {e}")

        # validate required prompt templates exist
        required_prompts = ["iraki_assessment", "debate", "confidence_instructions"]
        missing_prompts = [p for p in required_prompts if p not in prompts]

        if missing_prompts:
            raise ValueError(
                f"Required prompt templates missing: {missing_prompts}. "
                f"Available prompts: {list(prompts.keys())}"
            )

        self.logger.info(
            f"Loaded {len(prompts)} prompt templates from {prompts_dir}: "
            f"{', '.join(list(prompts.keys())[:3])}..."
        )

        return prompts

    # =========================================================================
    # PUBLIC ACCESS METHODS
    # =========================================================================

    def get_expert_profile(self, expert_id: str) -> Dict:
        """
        Get expert profile by ID.

        Args:
            expert_id: Expert identifier

        Returns:
            Dict: Expert profile with all fields

        Raises:
            KeyError: If expert_id not found
        """
        if expert_id not in self._expert_lookup:
            available = list(self._expert_lookup.keys())[:5]
            raise KeyError(
                f"Expert '{expert_id}' not found. " f"Available experts: {available}..."
            )
        return self._expert_lookup[expert_id]

    def get_available_expert_ids(self) -> List[str]:
        """
        Get list of all available expert IDs.

        Returns:
            List[str]: Expert identifiers in panel order
        """
        return [e["id"] for e in self._expert_panel["expert_panel"]["experts"]]

    def get_questions(self) -> List[Dict]:
        """
        Get all assessment questions with full context.

        Returns:
            List[Dict]: Questions with clinical context
        """
        return self._questionnaire["questionnaire"]["questions"].copy()

    def get_question_by_id(self, q_id: str) -> Dict:
        """
        Get specific question by ID.

        Args:
            q_id: Question identifier

        Returns:
            Dict: Question with all fields

        Raises:
            KeyError: If question not found
        """
        if q_id not in self._question_lookup:
            available = list(self._question_lookup.keys())[:5]
            raise KeyError(
                f"Question '{q_id}' not found. " f"Available questions: {available}..."
            )
        return self._question_lookup[q_id].copy()

    def get_prompt_template(self, template_name: str) -> Dict:
        """
        Get prompt template by name.

        Args:
            template_name: Name of prompt template (without .json)

        Returns:
            Dict: Prompt template configuration

        Raises:
            KeyError: If template not found
        """
        if template_name not in self._prompts:
            available = list(self._prompts.keys())
            raise KeyError(
                f"Prompt template '{template_name}' not found. "
                f"Available templates: {available}"
            )
        return self._prompts[template_name].copy()

    def get_panel_metadata(self) -> Dict:
        """
        Get expert panel metadata for reporting.

        Returns:
            Dict: Panel statistics and composition
        """
        experts = self._expert_panel["expert_panel"]["experts"]
        specialties = [e["specialty"] for e in experts]

        return {
            "total_experts": len(experts),
            "specialties": list(set(specialties)),
            "average_experience": sum(e["experience_years"] for e in experts)
            / len(experts),
            "panel_version": self._expert_panel["expert_panel"].get("version", "1.0.0"),
        }

    def validate_expert_subset(self, expert_ids: List[str]) -> bool:
        """
        Validate that all requested experts exist.

        Args:
            expert_ids: List of expert IDs to validate

        Returns:
            bool: True if all experts exist

        Raises:
            ValueError: If any expert ID is invalid
        """
        invalid_ids = [eid for eid in expert_ids if eid not in self._expert_lookup]

        if invalid_ids:
            raise ValueError(
                f"Invalid expert IDs: {invalid_ids}. "
                f"Valid IDs: {list(self._expert_lookup.keys())}"
            )

        return True


class DataLoaderWrapper:
    """
    Simple wrapper around existing DataLoader.

    Uses the existing dataloader.py which already handles both real
    and dummy data modes. No need to reinvent the wheel - YAGNI!
    """

    def __init__(self, runtime_config: RuntimeConfig):
        """
        Initialize wrapper using existing DataLoader.

        Args:
            runtime_config: Runtime configuration

        Raises:
            RuntimeError: If DataLoader initialization fails
        """
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.DataLoaderWrapper")

        try:
            # import the existing DataLoader
            from dataloader import DataLoader

            # DataLoader already handles real vs dummy internally
            # just pass the use_dummy flag based on runtime config
            self.data_loader = DataLoader(
                data_dir=runtime_config.data_dir,
                cache_dir=runtime_config.cache_dir,
                use_dummy=not runtime_config.use_real_data,
            )

            # check if initialization was successful
            if self.data_loader.is_available():
                mode = "DUMMY" if self.data_loader.use_dummy else "REAL"
                self.logger.info(f"DataLoader initialized in {mode} mode")

                if not self.data_loader.use_dummy:
                    self.logger.info(
                        f"Loaded {len(self.data_loader.patient_ids)} real patients"
                    )
            else:
                raise RuntimeError("DataLoader not available after initialization")

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import DataLoader: {e}. "
                f"Ensure dataloader.py is in the project root."
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize DataLoader: {e}")
            raise

    def load_patient_case(self, case_id: str) -> Dict:
        """
        Load patient case using existing DataLoader.

        Args:
            case_id: Case identifier (e.g., "iraki_case_001")

        Returns:
            Dict: Patient case data

        Raises:
            ValueError: If case_id is invalid
        """
        return self.data_loader.load_patient_case(case_id)

    def get_available_patients(self, limit: int = 10) -> List[str]:
        """
        Get available patient IDs from DataLoader.

        Args:
            limit: Maximum number of IDs to return

        Returns:
            List[str]: Available case identifiers
        """
        return self.data_loader.get_available_patients(limit=limit)

    def get_data_source(self) -> str:
        """
        Get description of current data source.

        Returns:
            str: "REAL" or "DUMMY" mode indicator
        """
        return "DUMMY" if self.data_loader.use_dummy else "REAL"
