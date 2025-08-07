"""
Configuration loader for DelPHEA-irAKI system.

Centralized configuration loading with fail-fast validation for expert panels,
questionnaires, and prompt templates.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from .core import RuntimeConfig

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """Centralized configuration loading with fail-fast validation."""

    def __init__(self, runtime_config: RuntimeConfig):
        """Initialize configuration loader with validation.

        Args:
            runtime_config: Runtime configuration containing paths

        Raises:
            FileNotFoundError: If required configuration files are missing
            ValueError: If configuration files have invalid structure
        """
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.ConfigurationLoader")

        # Load and validate all configurations at startup
        self._expert_panel = self._load_expert_panel()
        self._questionnaire = self._load_questionnaire()
        self._prompts = self._load_prompts()

    def _load_expert_panel(self) -> Dict:
        """Load and validate expert panel configuration.

        Returns:
            Dict: Expert panel configuration

        Raises:
            FileNotFoundError: If panel config file doesn't exist
            ValueError: If panel config structure is invalid
        """
        panel_path = Path(self.config.expert_panel_config)
        if not panel_path.exists():
            raise FileNotFoundError(f"Expert panel config not found: {panel_path}")

        with open(panel_path, "r") as f:
            panel_config = json.load(f)

        # validate structure
        if "expert_panel" not in panel_config:
            raise ValueError("Expert panel config missing 'expert_panel' key")

        if "experts" not in panel_config["expert_panel"]:
            raise ValueError("Expert panel config missing 'experts' list")

        experts = panel_config["expert_panel"]["experts"]
        if not experts:
            raise ValueError("Expert panel config has empty experts list")

        # validate each expert has required fields
        required_fields = ["id", "specialty", "name"]
        for expert in experts:
            for field in required_fields:
                if field not in expert:
                    raise ValueError(
                        f"Expert missing required field '{field}': {expert}"
                    )

        self.logger.info(f"Loaded {len(experts)} experts from {panel_path}")
        return panel_config

    def _load_questionnaire(self) -> Dict:
        """Load and validate questionnaire configuration.

        Returns:
            Dict: Questionnaire configuration

        Raises:
            FileNotFoundError: If questionnaire config file doesn't exist
            ValueError: If questionnaire config structure is invalid
        """
        questionnaire_path = Path(self.config.questionnaire_config)
        if not questionnaire_path.exists():
            raise FileNotFoundError(
                f"Questionnaire config not found: {questionnaire_path}"
            )

        with open(questionnaire_path, "r") as f:
            questionnaire_config = json.load(f)

        # validate structure
        if "questionnaire" not in questionnaire_config:
            raise ValueError("Questionnaire config missing 'questionnaire' key")

        if "questions" not in questionnaire_config["questionnaire"]:
            raise ValueError("Questionnaire config missing 'questions' list")

        questions = questionnaire_config["questionnaire"]["questions"]
        if not questions:
            raise ValueError("Questionnaire config has empty questions list")

        # validate each question has required fields
        required_fields = ["id", "question", "clinical_context"]
        for question in questions:
            for field in required_fields:
                if field not in question:
                    raise ValueError(
                        f"Question missing required field '{field}': {question}"
                    )

        self.logger.info(f"Loaded {len(questions)} questions from {questionnaire_path}")
        return questionnaire_config

    def _load_prompts(self) -> Dict[str, Dict]:
        """Load all prompt templates from prompts directory.

        Returns:
            Dict[str, Dict]: Mapping of prompt names to templates

        Raises:
            FileNotFoundError: If prompts directory doesn't exist
            ValueError: If required prompt files are missing
        """
        prompts_dir = Path(self.config.prompts_dir)
        if not prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        prompts = {}
        for prompt_file in prompts_dir.glob("*.json"):
            with open(prompt_file, "r") as f:
                prompts[prompt_file.stem] = json.load(f)

        if not prompts:
            raise ValueError(f"No prompt files found in {prompts_dir}")

        # validate required prompt files exist
        required_prompts = ["iraki_assessment", "debate", "confidence_instructions"]
        for required in required_prompts:
            if required not in prompts:
                raise ValueError(f"Required prompt file missing: {required}.json")

        self.logger.info(f"Loaded {len(prompts)} prompt templates from {prompts_dir}")
        return prompts

    def get_expert_profile(self, expert_id: str) -> Dict:
        """Get expert profile by ID.

        Args:
            expert_id: Expert identifier

        Returns:
            Dict: Expert profile

        Raises:
            ValueError: If expert not found
        """
        experts = self._expert_panel["expert_panel"]["experts"]
        for expert in experts:
            if expert["id"] == expert_id:
                return expert
        raise ValueError(f"Expert not found: {expert_id}")

    def get_available_expert_ids(self) -> List[str]:
        """Get list of all available expert IDs.

        Returns:
            List[str]: Expert identifiers
        """
        return [
            expert["id"] for expert in self._expert_panel["expert_panel"]["experts"]
        ]

    def get_questions(self) -> List[Dict]:
        """Get all questions with full context.

        Returns:
            List[Dict]: Question objects with clinical context
        """
        return self._questionnaire["questionnaire"]["questions"]

    def get_question_by_id(self, q_id: str) -> Dict:
        """Get specific question by ID.

        Args:
            q_id: Question identifier

        Returns:
            Dict: Question object

        Raises:
            ValueError: If question not found
        """
        questions = self.get_questions()
        for question in questions:
            if question["id"] == q_id:
                return question
        raise ValueError(f"Question not found: {q_id}")

    def get_prompt_template(self, template_name: str) -> Dict:
        """Get prompt template by name.

        Args:
            template_name: Name of prompt template

        Returns:
            Dict: Prompt template

        Raises:
            ValueError: If template not found
        """
        if template_name not in self._prompts:
            raise ValueError(f"Prompt template not found: {template_name}")
        return self._prompts[template_name]
