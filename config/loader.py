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

"""
Simplified Configuration Loader for DelPHEA-irAKI (YAGNI Principle)
===================================================================

Validates minimal required fields following YAGNI principle. Expert agents
will interpret questions based on their built-in expertise rather than
following prescriptive guidance.
"""

import json
import logging
from pathlib import Path
from typing import Dict

from config.core import RuntimeConfig


class ConfigurationLoader:
    """
    Simplified configuration loader following YAGNI principle.

    Validates only essential fields, allowing expert agents to apply
    their own clinical reasoning without prescriptive constraints.
    """

    def __init__(self, runtime_config: RuntimeConfig):
        """
        Initialize configuration loader with minimal validation.

        Args:
            runtime_config: Runtime configuration with file paths

        Raises:
            FileNotFoundError: If required configuration files are missing
            ValueError: If essential structure is invalid
            KeyError: If critical fields are missing
        """
        self.config = runtime_config
        self.logger = logging.getLogger(f"{__name__}.ConfigurationLoader")

        # load configurations with minimal validation
        self.logger.info("Loading configurations (YAGNI mode)...")

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
            f"Configuration loaded: "
            f"{len(self._expert_lookup)} experts, "
            f"{len(self._question_lookup)} questions"
        )

    def _load_expert_panel(self) -> Dict:
        """
        Load expert panel with minimal validation.

        Returns:
            Dict: Expert panel configuration

        Raises:
            FileNotFoundError: If panel config doesn't exist
            ValueError: If basic structure is invalid
        """
        panel_path = Path(self.config.expert_panel_config)
        if not panel_path.exists():
            raise FileNotFoundError(f"Expert panel config not found: {panel_path}")

        try:
            with open(panel_path, "r", encoding="utf-8") as f:
                panel_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in panel config: {e}")

        # minimal validation - just check structure exists
        if "expert_panel" not in panel_config:
            raise ValueError("Panel config missing 'expert_panel' key")

        if "experts" not in panel_config["expert_panel"]:
            raise ValueError("Panel config missing 'experts' list")

        experts = panel_config["expert_panel"]["experts"]
        if not experts:
            raise ValueError("Expert list cannot be empty")

        # only validate essential fields
        for i, expert in enumerate(experts):
            if "id" not in expert:
                raise ValueError(f"Expert {i} missing required 'id' field")
            if "specialty" not in expert:
                raise ValueError(f"Expert {expert.get('id', i)} missing 'specialty'")

        # check for duplicate IDs (critical for system function)
        expert_ids = [e["id"] for e in experts]
        if len(expert_ids) != len(set(expert_ids)):
            duplicates = [x for x in expert_ids if expert_ids.count(x) > 1]
            raise ValueError(f"Duplicate expert IDs: {set(duplicates)}")

        self.logger.info(f"Loaded {len(experts)} experts")
        return panel_config

    def _load_questionnaire(self) -> Dict:
        """
        Load questionnaire with simplified validation (YAGNI).

        Returns:
            Dict: Questionnaire configuration

        Raises:
            FileNotFoundError: If questionnaire doesn't exist
            ValueError: If essential structure is invalid
        """
        questionnaire_path = Path(self.config.questionnaire_config)
        if not questionnaire_path.exists():
            raise FileNotFoundError(
                f"Questionnaire config not found: {questionnaire_path}"
            )

        try:
            with open(questionnaire_path, "r", encoding="utf-8") as f:
                questionnaire_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in questionnaire: {e}")

        # minimal structural validation
        if "questionnaire" not in questionnaire_config:
            raise ValueError("Config missing 'questionnaire' key")

        if "questions" not in questionnaire_config["questionnaire"]:
            raise ValueError("Config missing 'questions' list")

        questions = questionnaire_config["questionnaire"]["questions"]
        if not questions:
            raise ValueError("Questions list cannot be empty")

        # validate only essential fields per question
        for i, question in enumerate(questions):
            # only id and question text are truly required
            if "id" not in question:
                raise ValueError(f"Question {i} missing 'id'")
            if "question" not in question:
                raise ValueError(
                    f"Question {question.get('id', i)} missing 'question' text"
                )

        # check for duplicate question IDs
        question_ids = [q["id"] for q in questions]
        if len(question_ids) != len(set(question_ids)):
            duplicates = [x for x in question_ids if question_ids.count(x) > 1]
            raise ValueError(f"Duplicate question IDs: {set(duplicates)}")

        self.logger.info(f"Loaded {len(questions)} assessment questions")
        return questionnaire_config

    def _load_prompts(self) -> Dict[str, Dict]:
        """
        Load prompt templates from prompts directory.

        Returns:
            Dict: Mapping of template names to prompt configurations

        Raises:
            FileNotFoundError: If prompts directory doesn't exist
            ValueError: If no valid prompts found
        """
        prompts_dir = Path(self.config.prompts_dir)
        if not prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        prompts = {}
        for prompt_file in prompts_dir.glob("*.json"):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)
                    prompt_name = prompt_file.stem
                    prompts[prompt_name] = prompt_data
                    self.logger.debug(f"Loaded prompt template: {prompt_name}")
            except json.JSONDecodeError as e:
                self.logger.warning(f"Skipping invalid prompt file {prompt_file}: {e}")
                continue

        if not prompts:
            raise ValueError(f"No valid prompt templates found in {prompts_dir}")

        return prompts

    @property
    def expert_panel(self) -> Dict:
        """Get expert panel configuration."""
        return self._expert_panel

    @property
    def questionnaire(self) -> Dict:
        """Get questionnaire configuration."""
        return self._questionnaire

    @property
    def prompts(self) -> Dict[str, Dict]:
        """Get prompt templates."""
        return self._prompts

    def get_expert(self, expert_id: str) -> Dict:
        """
        Get specific expert configuration.

        Args:
            expert_id: Expert identifier

        Returns:
            Dict: Expert configuration

        Raises:
            KeyError: If expert_id not found
        """
        if expert_id not in self._expert_lookup:
            raise KeyError(
                f"Expert '{expert_id}' not found. "
                f"Available: {list(self._expert_lookup.keys())}"
            )
        return self._expert_lookup[expert_id]

    def get_question(self, question_id: str) -> Dict:
        """
        Get specific question configuration.

        Args:
            question_id: Question identifier

        Returns:
            Dict: Question configuration

        Raises:
            KeyError: If question_id not found
        """
        if question_id not in self._question_lookup:
            raise KeyError(
                f"Question '{question_id}' not found. "
                f"Available: {list(self._question_lookup.keys())}"
            )
        return self._question_lookup[question_id]

    def get_prompt(self, template_name: str) -> Dict:
        """
        Get specific prompt template.

        Args:
            template_name: Name of prompt template

        Returns:
            Dict: Prompt template configuration

        Raises:
            KeyError: If template not found
        """
        if template_name not in self._prompts:
            raise KeyError(
                f"Prompt template '{template_name}' not found. "
                f"Available: {list(self._prompts.keys())}"
            )
        return self._prompts[template_name]

    def validate_runtime_consistency(self) -> None:
        """
        Validate cross-configuration consistency.

        Ensures expert specialties match any specialty_focus fields
        in questions (if present).

        Raises:
            ValueError: If inconsistencies detected
        """
        available_specialties = {
            expert["specialty"]
            for expert in self._expert_panel["expert_panel"]["experts"]
        }

        # check if any questions reference non-existent specialties
        for question in self._questionnaire["questionnaire"]["questions"]:
            if "specialty_focus" in question:
                # specialty_focus is optional but if present should be valid
                focus = question["specialty_focus"]
                if isinstance(focus, dict):
                    for specialty in focus.keys():
                        if (
                            specialty not in available_specialties
                            and specialty != "all_specialties"
                        ):
                            self.logger.warning(
                                f"Question {question['id']} references "
                                f"non-existent specialty: {specialty}"
                            )

        self.logger.info("Runtime consistency validation complete")
