"""
DelPHEA-irAKI: Delphi Personalized Health Explainable Agents for immune-related AKI classification
=================================================================================================

High-level
----------
* Multi-agent Delphi workflow (Moderator + N LLM-backed experts) for irAKI vs. other AKI classification
* Three structured rounds → consensus & confidence-weighted probability of irAKI diagnosis
* Beta opinion pooling aggregates expert probabilities with 95 % credible interval
* Modular expert-panel and questionnaire configuration via JSON files
* Optional literature search integration (PubMed/bioRxiv) for evidence-based reasoning
* Confidence-weighted pooling (no ARN learning; ground truth unavailable)
* Runs against a vLLM server; designed for AWS H100/A100
* Human-expert chart-review validation in place of automated learning

────────────────────────────────  ASCII "plumbing"  ────────────────────────────────
Legend:  →  async message  |  ⤳  RPC  |  (topic) = pub-sub topic name
────────────────────────────────────────────────────────────────────────────────────

          ┌─────────────┐
          │  Moderator  │
          └─────┬───────┘
                │               ┌────────────── Round 1 ──────────────┐
                │ Questionnaire │ (topic: case/<case>)                │
                ▼               ▼                                     │
       ┌──────────────────────────────────────┐                       │
       │          Expert agents (LLM)         │◄──── Literature Search│
       │      [Configurable Specialties]      │      (Optional)       │
       └──────────────────────────────────────┘                       │
                ⤳ record_round1(…)                                    │
                │                                                     │
                │ if disagreement ≥ Δ                                 │
                │                                                     │
                │               ┌────────────── Round 2 ──────────────┐
                │ DebatePrompt  │ (topic: debate/<case>:Qi:Ej)        │
                └──────────────▶│                                     │
                                │  Experts post DebateComment         │
                                └─────────────────────────────────────┘
                │                                                     │
                │               ┌────────────── Round 3 ──────────────┐
                │ Questionnaire │ (topic: case/<case>)                │
                ▼               ▼                                     │
       ┌──────────────────────────────────────┐                       │
       │          Expert agents (LLM)         │                       │
       └──────────────────────────────────────┘                       │
                ⤳ record_round3(…)                                    │
                │                                                     │
                │ transcripts → **Beta Pooling**                      │
                ▼                                                     │
          ┌─────────────┐                                             │
          │  Consensus  │                                             │
          │ Calculation │                                             │
          └─────┬───────┘                                             │
                │                                                     │
                ▼                                                     │
          Human Expert Chart Review                                   │
────────────────────────────────────────────────────────────────────────────────────


Modular components
------------------
* config/panel.json – expert-panel configuration with specialties
* config/questionnaire.json – clinical assessment questions for irAKI classification
* search.py – PubMed/bioRxiv integration for evidence-based reasoning
* Flexible data loader for various input formats (todo: customize based on actual data)

Outputs
-------
* Beta-pooled consensus (probability + 95 % CI + binary verdict) written to logs/stdout
* Complete decision transcript for human-expert review
* Detailed reasoning chains for each question and expert
* Literature citations and evidence integration (if enabled)
"""

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import numpy as np
from autogen_core import (AgentId, MessageContext, RoutedAgent,
                          SingleThreadedAgentRuntime, TopicId, message_handler,
                          rpc, type_subscription)
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import beta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DelPHEAirAKIConfig:
    """Configuration for DelPHEA irAKI classification system"""
    # vLLM Server Configuration
    vllm_endpoint: str = "http://172.31.11.192:8000"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: Optional[str] = None

    # Inference Parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 3072
    timeout: float = 120.0

    # Delphi Process Configuration
    conflict_threshold: int = 3
    max_debate_rounds: int = 6
    max_debate_participants: int = 8

    # Configuration File Paths
    expert_panel_config: str = "config/panel.json"
    questionnaire_config: str = "config/questionnaire.json"
    prompts_dir: str = "prompts"

    # Consensus Configuration
    consensus_threshold: float = 0.8
    confidence_threshold: float = 0.7

    # Timeouts (seconds)
    round1_timeout: int = 900
    round3_timeout: int = 600
    debate_timeout: int = 240

    # Export Configuration
    export_full_transcripts: bool = True
    include_reasoning_chains: bool = True

config = DelPHEAirAKIConfig()

# =============================================================================
# MESSAGE SCHEMAS
# =============================================================================

class QuestionnaireMsg(BaseModel):
    model_config = ConfigDict(strict=True)
    case_id: str
    patient_info: dict
    icu_summary: str
    medication_history: dict
    lab_values: dict
    imaging_reports: str
    questions: List[dict]  # Full question objects with contexts
    round_phase: str = "round1"

class ExpertRound1Reply(BaseModel):
    model_config = ConfigDict(strict=True)
    case_id: str
    expert_id: str
    scores: Dict[str, int]
    evidence: Dict[str, str]
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: List[float]
    confidence: float = Field(ge=0.0, le=1.0)
    differential_diagnosis: List[str]

class ExpertRound3Reply(BaseModel):
    model_config = ConfigDict(strict=True)
    case_id: str
    expert_id: str
    scores: Dict[str, int]
    evidence: Dict[str, str]
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: List[float]
    confidence: float = Field(ge=0.0, le=1.0)
    changes_from_round1: Dict[str, str]
    verdict: bool
    final_diagnosis: str
    recommendations: List[str]

class DebatePrompt(BaseModel):
    model_config = ConfigDict(strict=True)
    case_id: str
    q_id: str
    minority_view: str
    round_no: int
    participating_experts: List[str]
    clinical_context: str

class DebateComment(BaseModel):
    q_id: str
    author: str
    text: str
    citations: List[str] = []
    satisfied: bool = False

class TerminateDebate(BaseModel):
    model_config = ConfigDict(strict=True)
    case_id: str
    q_id: str
    reason: str

class StartCase(BaseModel):
    case_id: str

class AckMsg(BaseModel):
    ok: bool
    message: Optional[str] = None

class HumanReviewExport(BaseModel):
    case_id: str
    final_consensus: Dict[str, Any]
    expert_assessments: List[Dict[str, Any]]
    debate_transcripts: List[Dict[str, Any]]
    reasoning_summary: str
    clinical_timeline: Dict[str, Any]

# =============================================================================
# BETA OPINION POOLING
# =============================================================================

def beta_pool_confidence(p_vec: np.ndarray,
                         ci_mat: np.ndarray,
                         weight_vec: np.ndarray = None) -> dict:
    """Beta opinion pooling with confidence estimation"""
    lo = ci_mat[:, 0]
    hi = ci_mat[:, 1]

    # Between-expert variance
    var_between = np.var(p_vec, ddof=1) if p_vec.size > 1 else 0.0
    between_score = 1.0 - (var_between / 0.25)
    between_score = np.clip(between_score, 0.0, 1.0)

    # Within-expert CI width
    half_widths = (hi - lo) / 2.0
    mean_half = half_widths.mean()
    within_score = 1.0 - (mean_half / 0.5)
    within_score = np.clip(within_score, 0.0, 1.0)

    # Harmonic-mean panel confidence
    if between_score == 0 or within_score == 0:
        consensus_conf = 0.0
    else:
        consensus_conf = 2.0 / (1.0 / between_score + 1.0 / within_score)

    # Beta opinion pool
    w = weight_vec if weight_vec is not None else np.ones_like(p_vec)
    a_post = 1.0 + np.sum(w * p_vec)
    b_post = 1.0 + np.sum(w * (1.0 - p_vec))

    post_mean = a_post / (a_post + b_post)
    post_ci_95 = beta.ppf([0.025, 0.975], a_post, b_post).tolist()

    return {
        "pooled_mean": post_mean,
        "pooled_ci": post_ci_95,
        "var_between": var_between,
        "mean_halfwidth": mean_half,
        "consensus_conf": consensus_conf,
        "between_score": between_score,
        "within_score": within_score,
    }

# =============================================================================
# vLLM CLIENT
# =============================================================================

class VLLMClient:
    """Production vLLM client with OpenAI-compatible API"""

    def __init__(self, config: DelPHEAirAKIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.VLLMClient")

        timeout = httpx.Timeout(config.timeout)
        self.client = httpx.AsyncClient(timeout=timeout)

        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

    async def generate_structured_response(self, prompt: str, response_format: Dict) -> Dict:
        """Generate structured JSON response using vLLM"""
        request_data = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "response_format": response_format,
        }

        try:
            response = await self.client.post(
                f"{self.config.vllm_endpoint}/v1/chat/completions",
                headers=self.headers,
                json=request_data
            )

            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"vLLM API error: {response.status_code}",
                    request=response.request,
                    response=response
                )

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from vLLM: {e}")
        except Exception as e:
            self.logger.error(f"Error in vLLM client: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if vLLM server is healthy"""
        try:
            for endpoint in ["/health", "/healthz"]:
                try:
                    response = await self.client.get(f"{self.config.vllm_endpoint}{endpoint}")
                    if response.status_code == 200:
                        return True
                except:
                    continue
            return False
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        await self.client.aclose()

# =============================================================================
# CONFIGURATION LOADERS
# =============================================================================

class ConfigurationLoader:
    """Centralized configuration loading with fail-fast validation"""

    def __init__(self, config: DelPHEAirAKIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ConfigurationLoader")

        # Load and validate all configurations at startup
        self._expert_panel = self._load_expert_panel()
        self._questionnaire = self._load_questionnaire()
        self._prompts = self._load_prompts()

    def _load_expert_panel(self) -> Dict:
        """Load and validate expert panel configuration"""
        panel_path = Path(self.config.expert_panel_config)
        if not panel_path.exists():
            raise FileNotFoundError(f"Expert panel config not found: {panel_path}")

        with open(panel_path, 'r') as f:
            panel_config = json.load(f)

        # Validate structure
        if "expert_panel" not in panel_config:
            raise ValueError("Expert panel config missing 'expert_panel' key")

        if "experts" not in panel_config["expert_panel"]:
            raise ValueError("Expert panel config missing 'experts' list")

        experts = panel_config["expert_panel"]["experts"]
        if not experts:
            raise ValueError("Expert panel config has empty experts list")

        # Validate each expert has required fields
        required_fields = ["id", "specialty", "name"]
        for expert in experts:
            for field in required_fields:
                if field not in expert:
                    raise ValueError(f"Expert missing required field '{field}': {expert}")

        self.logger.info(f"Loaded {len(experts)} experts from {panel_path}")
        return panel_config

    def _load_questionnaire(self) -> Dict:
        """Load and validate questionnaire configuration"""
        questionnaire_path = Path(self.config.questionnaire_config)
        if not questionnaire_path.exists():
            raise FileNotFoundError(f"Questionnaire config not found: {questionnaire_path}")

        with open(questionnaire_path, 'r') as f:
            questionnaire_config = json.load(f)

        # Validate structure
        if "questionnaire" not in questionnaire_config:
            raise ValueError("Questionnaire config missing 'questionnaire' key")

        if "questions" not in questionnaire_config["questionnaire"]:
            raise ValueError("Questionnaire config missing 'questions' list")

        questions = questionnaire_config["questionnaire"]["questions"]
        if not questions:
            raise ValueError("Questionnaire config has empty questions list")

        # Validate each question has required fields
        required_fields = ["id", "question", "clinical_context"]
        for question in questions:
            for field in required_fields:
                if field not in question:
                    raise ValueError(f"Question missing required field '{field}': {question}")

        self.logger.info(f"Loaded {len(questions)} questions from {questionnaire_path}")
        return questionnaire_config

    def _load_prompts(self) -> Dict[str, Dict]:
        """Load all prompt templates from prompts directory"""
        prompts_dir = Path(self.config.prompts_dir)
        if not prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        prompts = {}
        for prompt_file in prompts_dir.glob("*.json"):
            with open(prompt_file, 'r') as f:
                prompts[prompt_file.stem] = json.load(f)

        if not prompts:
            raise ValueError(f"No prompt files found in {prompts_dir}")

        # Validate required prompt files exist
        required_prompts = ["iraki_assessment", "debate", "confidence_instructions"]
        for required in required_prompts:
            if required not in prompts:
                raise ValueError(f"Required prompt file missing: {required}.json")

        self.logger.info(f"Loaded {len(prompts)} prompt templates from {prompts_dir}")
        return prompts

    def get_expert_profile(self, expert_id: str) -> Dict:
        """Get expert profile by ID"""
        experts = self._expert_panel["expert_panel"]["experts"]
        for expert in experts:
            if expert["id"] == expert_id:
                return expert
        raise ValueError(f"Expert not found: {expert_id}")

    def get_available_expert_ids(self) -> List[str]:
        """Get list of all available expert IDs"""
        return [expert["id"] for expert in self._expert_panel["expert_panel"]["experts"]]

    def get_questions(self) -> List[Dict]:
        """Get all questions with full context"""
        return self._questionnaire["questionnaire"]["questions"]

    def get_question_by_id(self, q_id: str) -> Dict:
        """Get specific question by ID"""
        questions = self.get_questions()
        for question in questions:
            if question["id"] == q_id:
                return question
        raise ValueError(f"Question not found: {q_id}")

    def get_prompt_template(self, template_name: str) -> Dict:
        """Get prompt template by name"""
        if template_name not in self._prompts:
            raise ValueError(f"Prompt template not found: {template_name}")
        return self._prompts[template_name]

# =============================================================================
# DATA LOADER
# =============================================================================

class irAKIDataLoader:
    """Data loader for irAKI cases"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.irAKIDataLoader")

    def load_patient_case(self, case_id: str) -> Dict:
        """Load patient case data for irAKI assessment"""
        return {
            "case_id": case_id,
            "patient_info": {
                "age": 67,
                "gender": "F",
                "primary_diagnosis": "metastatic melanoma",
                "comorbidities": ["Stage II CKD", "Hypertension", "Type 2 DM"],
                "admission_diagnosis": "AKI in setting of immunotherapy",
                "baseline_creatinine": 1.1,
                "weight": 70
            },
            "icu_summary": (
                f"ICU admission for case {case_id}: 67-year-old female with metastatic melanoma "
                "on combination immunotherapy (nivolumab + ipilimumab) presenting with AKI. "
                "Creatinine rose from baseline 1.1 to 3.2 mg/dL over 2 weeks. No obvious "
                "nephrotoxic exposures. Urine shows proteinuria and microscopic hematuria. "
                "Patient denied NSAIDs, contrast exposure in past month."
            ),
            "medication_history": {
                "immunotherapy": {
                    "agents": ["nivolumab", "ipilimumab"],
                    "start_date": "2024-01-15",
                    "last_dose": "2024-07-10",
                    "cycles_completed": 6,
                    "response": "partial response"
                },
                "concurrent_medications": [
                    "lisinopril 10mg daily", "metformin 1000mg bid",
                    "amlodipine 5mg daily", "atorvastatin 20mg daily"
                ],
                "recent_changes": "lisinopril held 5 days ago due to AKI"
            },
            "lab_values": {
                "creatinine_timeline": {
                    "baseline": 1.1,
                    "day_-14": 1.2,
                    "day_-7": 1.8,
                    "day_-3": 2.4,
                    "day_0": 3.2
                },
                "urinalysis": {
                    "protein": "2+",
                    "blood": "1+",
                    "rbc": "5-10/hpf",
                    "wbc": "2-5/hpf",
                    "casts": "rare granular"
                },
                "other_labs": {
                    "bun": 45,
                    "sodium": 138,
                    "potassium": 4.2,
                    "eosinophils": "12%",
                    "complement_c3": 85,
                    "complement_c4": 20
                }
            },
            "imaging_reports": (
                "Renal ultrasound: kidneys normal size, no hydronephrosis, "
                "increased echogenicity consistent with medical renal disease. "
                "No masses or stones identified."
            )
        }

# =============================================================================
# EXPERT AGENT
# =============================================================================

class irAKIExpertAgent(RoutedAgent):
    """Clinical expert agent for irAKI assessment"""

    def __init__(self, expert_id: str, case_id: str, config_loader: ConfigurationLoader) -> None:
        super().__init__(f"irAKI Expert {expert_id}")
        self._expert_id = expert_id
        self._case_id = case_id
        self._config_loader = config_loader
        self._vllm_client = VLLMClient(config)
        self.logger = logging.getLogger(f"expert.{expert_id}")

        # Load expert profile - fail fast if not found
        self._expert_profile = self._config_loader.get_expert_profile(expert_id)
        self.logger.info(f"Initialized expert: {self._expert_profile['name']} ({self._expert_profile['specialty']})")

    @message_handler
    async def handle_questionnaire(self, message: QuestionnaireMsg, ctx: MessageContext) -> None:
        """Handle Round 1 & 3 irAKI assessment"""
        # Get prompt template
        prompt_template = self._config_loader.get_prompt_template("iraki_assessment")

        # Build prompt with expert profile and case data
        prompt = self._build_assessment_prompt(message, prompt_template)

        try:
            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt,
                response_format={"type": "json_object"}
            )

            # Validate CI bounds
            if "ci_iraki" in llm_response:
                ci_lower, ci_upper = llm_response["ci_iraki"]
                ci_lower = max(0.0, min(ci_lower, llm_response["p_iraki"]))
                ci_upper = min(1.0, max(ci_upper, llm_response["p_iraki"]))
                llm_response["ci_iraki"] = [ci_lower, ci_upper]
            else:
                p = llm_response["p_iraki"]
                width = 0.1  # Default ±10% uncertainty
                llm_response["ci_iraki"] = [max(0.0, p - width), min(1.0, p + width)]

            # Create reply object
            if message.round_phase == "round1":
                required_fields = ["scores", "evidence", "p_iraki", "ci_iraki", "confidence", "differential_diagnosis"]
                reply_class = ExpertRound1Reply
            else:  # round3
                required_fields = ["scores", "evidence", "p_iraki", "ci_iraki", "confidence",
                                   "changes_from_round1", "verdict", "final_diagnosis", "recommendations"]
                reply_class = ExpertRound3Reply

            # Validate required fields
            missing = [f for f in required_fields if f not in llm_response]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            reply = reply_class(
                **llm_response,
                expert_id=self._expert_id,
                case_id=message.case_id
            )

            # Send via RPC to moderator
            ack_response = await self.send_message(reply, AgentId("Moderator", message.case_id))

            if not ack_response.ok:
                self.logger.error(f"Moderator rejected response: {ack_response.message}")
            else:
                self.logger.info(f"Successfully submitted {message.round_phase} assessment")

        except Exception as e:
            self.logger.error(f"Failed to generate {message.round_phase} response: {e}")
            raise

    @message_handler
    async def handle_debate_prompt(self, message: DebatePrompt, ctx: MessageContext) -> None:
        """Handle Round 2 debate"""
        prompt_template = self._config_loader.get_prompt_template("debate")
        prompt = self._build_debate_prompt(message, prompt_template)

        try:
            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt,
                response_format={"type": "json_object"}
            )

            comment = DebateComment(
                q_id=message.q_id,
                author=self._expert_id,
                text=llm_response.get("text", ""),
                citations=llm_response.get("citations", []),
                satisfied=llm_response.get("satisfied", False)
            )

            await self.publish_message(comment, ctx.topic_id)
            self.logger.info(f"Posted debate comment for {message.q_id}")

        except Exception as e:
            self.logger.error(f"Failed to generate debate response: {e}")
            raise

    @message_handler
    async def handle_terminate_debate(self, message: TerminateDebate, ctx: MessageContext) -> None:
        """Handle debate termination signal"""
        self.logger.info(f"Exiting debate for {message.case_id}:{message.q_id}")

    def _build_assessment_prompt(self, message: QuestionnaireMsg, template: Dict) -> str:
        """Build irAKI assessment prompt from template"""
        # Format questions with their contexts
        formatted_questions = []
        for i, question_obj in enumerate(message.questions):
            q_text = question_obj["question"]
            context = question_obj.get("clinical_context", {})

            q_formatted = f"{i+1}. {q_text}"
            if context:
                if "supportive_evidence" in context:
                    q_formatted += f"\n   Supporting evidence: {', '.join(context['supportive_evidence'][:3])}"
                if "contradictory_evidence" in context:
                    q_formatted += f"\n   Contradictory evidence: {', '.join(context['contradictory_evidence'][:3])}"

            formatted_questions.append(q_formatted)

        # Get confidence instructions
        confidence_template = self._config_loader.get_prompt_template("confidence_instructions")

        # Build prompt from template
        prompt = template["base_template"].format(
            expert_name=self._expert_profile["name"],
            expert_experience=f"{self._expert_profile['experience_years']} years {self._expert_profile['specialty']}",
            expert_focus=", ".join(self._expert_profile.get("expertise", [])),
            case_id=message.case_id,
            patient_info=json.dumps(message.patient_info, indent=2),
            icu_summary=message.icu_summary,
            medication_history=json.dumps(message.medication_history, indent=2),
            lab_values=json.dumps(message.lab_values, indent=2),
            imaging_reports=message.imaging_reports,
            questions="\n".join(formatted_questions),
            confidence_instructions=confidence_template["ci_instructions"],
            round_phase=message.round_phase,
            specialty=self._expert_profile["specialty"]
        )

        return prompt

    def _build_debate_prompt(self, message: DebatePrompt, template: Dict) -> str:
        """Build debate prompt from template"""
        return template["base_template"].format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            q_id=message.q_id,
            round_no=message.round_no,
            clinical_context=message.clinical_context,
            minority_view=message.minority_view
        )

# Global configuration loader
_config_loader = None

def get_config_loader() -> ConfigurationLoader:
    """Get global configuration loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader(config)
    return _config_loader

# =============================================================================
# MODERATOR AGENT (simplified, focusing on key fixes)
# =============================================================================

class irAKIModeratorAgent(RoutedAgent):
    """Master agent coordinating irAKI Delphi process"""

    def __init__(self, case_id: str) -> None:
        super().__init__(f"irAKI Moderator for case {case_id}")
        self._case_id = case_id
        self._config_loader = get_config_loader()
        self._data_loader = irAKIDataLoader()

        # Get available expert IDs from configuration
        self._expert_ids = self._config_loader.get_available_expert_ids()[:config.expert_count if hasattr(config, 'expert_count') else 8]

        # Round tracking
        self._round1_replies: List[ExpertRound1Reply] = []
        self._round3_replies: List[ExpertRound3Reply] = []
        self._chat_logs: Dict[str, List[Dict]] = defaultdict(list)

        # Debate management
        self._active_debates: Dict[str, Dict] = {}

        # Synchronization
        self._pending_round1: Set[str] = set()
        self._pending_round3: Set[str] = set()
        self._round1_done = asyncio.Event()
        self._round3_done = asyncio.Event()

        self.logger = logging.getLogger(f"iraki_moderator.{case_id}")
        self.logger.info(f"Initialized moderator with experts: {self._expert_ids}")

    @message_handler
    async def handle_start_case(self, message: StartCase, ctx: MessageContext) -> None:
        """Bootstrap irAKI Delphi process"""
        self.logger.info(f"Starting irAKI Delphi process for case {message.case_id}")
        patient_data = self._data_loader.load_patient_case(message.case_id)
        await self._run_round1(patient_data)

    async def _run_round1(self, patient_data: Dict) -> None:
        """Execute Round 1: individual irAKI assessments"""
        self.logger.info("=== ROUND 1: Individual irAKI Assessments ===")

        # Load questions with full context from configuration
        questions = self._config_loader.get_questions()

        # Initialize pending tracking with actual expert IDs
        self._pending_round1 = set(self._expert_ids)
        self._round1_done.clear()

        # Create questionnaire message
        questionnaire = QuestionnaireMsg(
            case_id=self._case_id,
            patient_info=patient_data["patient_info"],
            icu_summary=patient_data["icu_summary"],
            medication_history=patient_data["medication_history"],
            lab_values=patient_data["lab_values"],
            imaging_reports=patient_data["imaging_reports"],
            questions=questions,
            round_phase="round1"
        )

        await self.publish_message(questionnaire, TopicId("case", self._case_id))
        self.logger.info(f"Broadcast Round 1 questionnaire to {len(self._expert_ids)} experts")

        await self._wait_for_round_completion("round1")

    @rpc
    async def record_round1(self, message: ExpertRound1Reply, ctx: MessageContext) -> AckMsg:
        """Collect Round 1 expert replies"""
        # Validate expert ID
        if message.expert_id not in self._expert_ids:
            return AckMsg(ok=False, message=f"Unknown expert ID: {message.expert_id}")

        self._chat_logs[message.expert_id].append({
            "role": "expert_reply",
            "payload": message.model_dump(),
            "timestamp": time.time()
        })

        self._round1_replies.append(message)
        self._pending_round1.discard(message.expert_id)

        self.logger.debug(f"Round 1 reply from {message.expert_id} ({len(self._pending_round1)} pending)")

        if not self._pending_round1:
            self._round1_done.set()

        return AckMsg(ok=True, message="Round 1 reply recorded")

    @rpc
    async def record_round3(self, message: ExpertRound3Reply, ctx: MessageContext) -> AckMsg:
        """Collect Round 3 expert replies"""
        # Validate expert ID
        if message.expert_id not in self._expert_ids:
            return AckMsg(ok=False, message=f"Unknown expert ID: {message.expert_id}")

        self._chat_logs[message.expert_id].append({
            "role": "expert_reply",
            "payload": message.model_dump(),
            "timestamp": time.time()
        })

        self._round3_replies.append(message)
        self._pending_round3.discard(message.expert_id)

        self.logger.debug(f"Round 3 reply from {message.expert_id} ({len(self._pending_round3)} pending)")

        if not self._pending_round3:
            self._round3_done.set()

        return AckMsg(ok=True, message="Round 3 reply recorded")

    async def _wait_for_round_completion(self, round_phase: str) -> None:
        """Wait for round completion with timeout"""
        event = self._round1_done if round_phase == "round1" else self._round3_done
        timeout = getattr(config, f"{round_phase}_timeout", 300)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            self.logger.info(f"All {round_phase} replies received")

            if round_phase == "round1":
                await self._run_round2()
            else:
                await self._compute_final_consensus()

        except asyncio.TimeoutError:
            pending = self._pending_round1 if round_phase == "round1" else self._pending_round3
            self.logger.warning(f"Timeout in {round_phase}: {len(pending)} experts pending")

            if round_phase == "round1":
                await self._run_round2()
            else:
                await self._compute_final_consensus()

    async def _run_round2(self) -> None:
        """Execute Round 2: debate conflicts"""
        self.logger.info("=== ROUND 2: Conflict Resolution ===")

        conflicts = self._detect_conflicts()

        if not conflicts:
            self.logger.info("No conflicts detected, proceeding to Round 3")
            await self._run_round3()
            return

        self.logger.info(f"Detected conflicts in {len(conflicts)} questions")
        # Simplified debate implementation - could be expanded
        await self._run_round3()

    def _detect_conflicts(self) -> Dict[str, Dict]:
        """Detect questions with significant disagreement"""
        conflicts = {}
        questions = self._config_loader.get_questions()

        for i, question in enumerate(questions):
            q_id = question["id"]
            scores = [
                reply.scores.get(q_id, 5)
                for reply in self._round1_replies
                if q_id in reply.scores
            ]

            if len(scores) > 1 and max(scores) - min(scores) >= config.conflict_threshold:
                conflicts[q_id] = {
                    "question": question,
                    "score_range": f"{min(scores)}-{max(scores)}",
                    "clinical_context": question.get("clinical_context", {})
                }

        return conflicts

    async def _run_round3(self) -> None:
        """Execute Round 3: final consensus"""
        self.logger.info("=== ROUND 3: Final Consensus ===")

        patient_data = self._data_loader.load_patient_case(self._case_id)
        questions = self._config_loader.get_questions()

        self._pending_round3 = set(self._expert_ids)
        self._round3_done.clear()

        questionnaire = QuestionnaireMsg(
            case_id=self._case_id,
            patient_info=patient_data["patient_info"],
            icu_summary=patient_data["icu_summary"],
            medication_history=patient_data["medication_history"],
            lab_values=patient_data["lab_values"],
            imaging_reports=patient_data["imaging_reports"],
            questions=questions,
            round_phase="round3"
        )

        await self.publish_message(questionnaire, TopicId("case", self._case_id))
        self.logger.info("Broadcast Round 3 questionnaire")

        await self._wait_for_round_completion("round3")

    async def _compute_final_consensus(self) -> None:
        """Compute beta pooling consensus for irAKI classification"""
        self.logger.info("=== COMPUTING FINAL CONSENSUS ===")

        if not self._round3_replies:
            self.logger.error("No Round 3 replies received")
            return

        # Extract data for beta pooling
        p_vec = np.array([r.p_iraki for r in self._round3_replies])
        ci_mat = np.array([r.ci_iraki for r in self._round3_replies])
        w_vec = np.array([r.confidence for r in self._round3_replies])

        # Compute beta pooling with confidence estimation
        stats = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # Traditional majority vote for comparison
        consensus_verdict = sum(r.verdict for r in self._round3_replies) > len(self._round3_replies) / 2

        # Log results
        self.logger.info("=" * 80)
        self.logger.info(f"CASE {self._case_id} irAKI CONSENSUS RESULTS:")
        self.logger.info("-" * 80)
        self.logger.info(f"Beta Pooled P(irAKI):    {stats['pooled_mean']:.3f}")
        self.logger.info(f"95% Credible Interval:   [{stats['pooled_ci'][0]:.3f}, {stats['pooled_ci'][1]:.3f}]")
        self.logger.info(f"Consensus Confidence:    {stats['consensus_conf']:.3f}")
        self.logger.info(f"Majority Vote Verdict:   {'irAKI' if consensus_verdict else 'Other AKI'}")
        self.logger.info(f"Expert Count:            {len(self._round3_replies)}")
        self.logger.info("=" * 80)

        # Export for human review
        if config.export_full_transcripts:
            await self._export_for_human_review(stats, consensus_verdict)

    async def _export_for_human_review(self, stats: Dict, consensus_verdict: bool) -> None:
        """Export complete case for human expert review"""
        # Implementation similar to original but cleaner
        self.logger.info("Exporting case for human review...")
        # Could implement full export here

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for DelPHEA-irAKI system"""
    parser = argparse.ArgumentParser(description="DelPHEA-irAKI: Clinical irAKI Classification")

    # Core configuration
    parser.add_argument("--case-id", default="iraki_case_001", help="irAKI case identifier")
    parser.add_argument("--expert-panel-config", default="config/panel.json", help="Expert panel config")
    parser.add_argument("--questionnaire-config", default="config/questionnaire.json", help="Questionnaire config")
    parser.add_argument("--prompts-dir", default="prompts", help="Prompts directory")

    # System options
    parser.add_argument("--health-check", action="store_true", help="Health check and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Update configuration
    config.expert_panel_config = args.expert_panel_config
    config.questionnaire_config = args.questionnaire_config
    config.prompts_dir = args.prompts_dir

    try:
        if args.health_check:
            # Initialize configuration loader to validate configs
            config_loader = ConfigurationLoader(config)

            # Check vLLM
            vllm_client = VLLMClient(config)
            healthy = await vllm_client.health_check()
            await vllm_client.close()

            if healthy:
                print("✓ DelPHEA-irAKI system healthy")
                print(f"✓ Loaded {len(config_loader.get_available_expert_ids())} experts")
                print(f"✓ Loaded {len(config_loader.get_questions())} questions")
                return 0
            else:
                print("✗ vLLM health check failed")
                return 1
        else:
            print(f"Starting DelPHEA-irAKI for case: {args.case_id}")

            # Initialize configuration (this will fail fast if configs are invalid)
            config_loader = get_config_loader()

            print(f"✓ Loaded {len(config_loader.get_available_expert_ids())} experts")
            print(f"✓ Loaded {len(config_loader.get_questions())} questions")
            print("-" * 60)

            # Create runtime
            runtime = SingleThreadedAgentRuntime()

            # Register agents
            await runtime.register("Moderator", lambda key: irAKIModeratorAgent(key))

            # Register expert agents
            for expert_id in config_loader.get_available_expert_ids():
                await runtime.register(
                    f"Expert_{expert_id}",
                    lambda key, eid=expert_id: irAKIExpertAgent(eid, key, config_loader)
                )

            # Start runtime
            await runtime.start()

            # Bootstrap process
            await runtime.send_message(
                StartCase(case_id=args.case_id),
                AgentId("Moderator", args.case_id)
            )

            # Wait for completion
            await runtime.stop_when_idle()
            await runtime.stop()

            print("\n✓ DelPHEA-irAKI completed successfully!")
            return 0

    except FileNotFoundError as e:
        print(f"\n✗ Configuration file not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
