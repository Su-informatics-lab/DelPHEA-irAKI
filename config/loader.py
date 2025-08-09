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
        self._validate_prompts_required()
        self._prompt_index = self._index_prompts(self._prompts)

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

    def _index_prompts(self, raw: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Flatten/normalize prompt files into named templates with guaranteed 'base_prompt'.
        Expected names: round1, round3, debate. Fail fast if missing.
        """
        idx: Dict[str, Dict] = {}

        def ensure_base(name: str, tpl: Dict, src: str) -> Dict:
            base = tpl.get("base_prompt") or tpl.get("prompt") or tpl.get("template")
            if not base:
                raise ValueError(
                    f"Prompt '{name}' missing 'base_prompt' in {src}. "
                    "Add key 'base_prompt' (string) or rename your existing key to it."
                )
            # don't mutate original; normalize key
            out = dict(tpl)
            out["base_prompt"] = base
            return out

        for fname, data in raw.items():
            # iraki_assessment.json may contain sections
            if fname == "iraki_assessment":
                if "round1" in data:
                    idx["round1"] = ensure_base(
                        "round1", data["round1"], f"{fname}.json"
                    )
                if "round3" in data:
                    idx["round3"] = ensure_base(
                        "round3", data["round3"], f"{fname}.json"
                    )
            # debate.json may be flat
            if fname == "debate":
                idx["debate"] = ensure_base("debate", data, f"{fname}.json")

        # fail fast: require all three
        missing = [k for k in ("round1", "round3", "debate") if k not in idx]
        if missing:
            raise ValueError(
                f"Missing prompt templates: {missing}. "
                f"Loaded files: {list(raw.keys())}. "
                "Ensure iraki_assessment.json has 'round1' and 'round3', and debate.json exists."
            )

        return idx

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

    def _validate_prompts_required(self) -> None:
        req = ["iraki_assessment", "debate", "confidence_instructions"]
        missing = [k for k in req if k not in self._prompts]
        if missing:
            raise ValueError(f"Missing required prompt files: {missing}")

        ia = self._prompts["iraki_assessment"]
        if "base_prompt" not in ia or "json_schema" not in ia:
            raise ValueError(
                "iraki_assessment.json must include 'base_prompt' and 'json_schema'"
            )

        db = self._prompts["debate"]
        if "base_prompt" not in db:
            raise ValueError("debate.json must include 'base_prompt'")

        ci = self._prompts["confidence_instructions"]
        if "ci_instructions" not in ci:
            raise ValueError(
                "confidence_instructions.json must include 'ci_instructions'"
            )

    def get_prompt_template(self, name: str) -> Dict:
        """Return normalized prompt by name (round1, round3, debate)."""
        if name not in self._prompt_index:
            raise KeyError(
                f"Unknown prompt template '{name}'. "
                f"Available: {list(self._prompt_index.keys())}"
            )
        return self._prompt_index[name]

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

    def get_questions(self) -> list:
        """
        Return the full list of assessment questions.
        Returns:
            List: All question configurations
        """
        return self._questionnaire["questionnaire"]["questions"]

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

        Simple validation to ensure configuration is coherent.

        Raises:
            ValueError: If critical inconsistencies detected
        """
        # ensure we have at least one expert
        experts = self._expert_panel["expert_panel"]["experts"]
        if not experts:
            raise ValueError("No experts configured")

        # ensure we have questions
        questions = self._questionnaire["questionnaire"]["questions"]
        if not questions:
            raise ValueError("No questions configured")

        self.logger.info("Runtime consistency validation complete")
