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
* config/panel.json – expert-panel configuration with specialties and personas
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

# Configure logging
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
    # vLLM Server Configuration (AWS fixed)
    vllm_endpoint: str = "http://172.31.11.192:8000"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_key: Optional[str] = None

    # Inference Parameters (optimized for clinical reasoning)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 3072
    timeout: float = 120.0

    # irAKI Classification Configuration
    questions_count: int = 12
    conflict_threshold: int = 3
    max_debate_rounds: int = 6
    max_debate_participants: int = 8

    # Expert Configuration
    expert_count: int = 8
    expert_specialties: List[str] = None
    expert_panel_config: str = "config/panel.json"
    questionnaire_config: str = "config/questionnaire.json"

    # Consensus Configuration with enhanced confidence
    use_equal_weighting: bool = True
    confidence_threshold: float = 0.7
    consensus_threshold: float = 0.8  # Minimum consensus confidence for strong agreement

    # Timeouts (seconds)
    round1_timeout: int = 900
    round3_timeout: int = 600
    debate_timeout: int = 240

    # Human Review Configuration
    export_full_transcripts: bool = True
    include_reasoning_chains: bool = True
    generate_summary_report: bool = True

    def __post_init__(self):
        if self.expert_specialties is None:
            self.expert_specialties = [
                "oncology", "nephrology", "rheumatology", "immunology",
                "critical_care", "clinical_pharmacology", "pathology", "informatics"
            ]

# Global configuration instance
config = DelPHEAirAKIConfig()

# =============================================================================
# MESSAGE SCHEMAS
# =============================================================================

class QuestionnaireMsg(BaseModel):
    """Round 1 & 3: irAKI assessment questionnaire"""
    model_config = ConfigDict(strict=True)

    case_id: str
    patient_info: dict
    icu_summary: str
    medication_history: dict
    lab_values: dict
    imaging_reports: str
    questions: List[str]
    round_phase: str = "round1"

class ExpertRound1Reply(BaseModel):
    """Round1: individual expert irAKI assessment with 95% CI"""
    model_config = ConfigDict(strict=True)

    case_id: str
    expert_id: str
    scores: Dict[str, int]              # question_id → 1‑9 Likert
    evidence: Dict[str, str]            # question_id → free‑text reasoning
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: List[float]               # [lower_95, upper_95] bounds
    confidence: float = Field(ge=0.0, le=1.0)  # self-reported confidence
    differential_diagnosis: List[str]

class ExpertRound3Reply(BaseModel):
    """Round3: final expert assessment with 95% CI"""
    model_config = ConfigDict(strict=True)

    case_id: str
    expert_id: str
    scores: Dict[str, int]
    evidence: Dict[str, str]
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: List[float]               # [lower_95, upper_95] bounds
    confidence: float = Field(ge=0.0, le=1.0)
    changes_from_round1: Dict[str, str]
    verdict: bool
    final_diagnosis: str
    recommendations: List[str]

class DebatePrompt(BaseModel):
    """Round 2: initiate debate for specific question"""
    model_config = ConfigDict(strict=True)

    case_id: str
    q_id: str
    minority_view: str
    round_no: int
    participating_experts: List[str]
    clinical_context: str

class DebateComment(BaseModel):
    """Expert contribution during Round-2 debate"""
    q_id: str
    author: str
    text: str
    citations: List[str] = []
    satisfied: bool = False

class TerminateDebate(BaseModel):
    """Signal end of debate for specific question"""
    model_config = ConfigDict(strict=True)

    case_id: str
    q_id: str
    reason: str

class StartCase(BaseModel):
    """Bootstrap Delphi process for irAKI classification"""
    case_id: str

class AckMsg(BaseModel):
    """Generic acknowledgment for RPC"""
    ok: bool
    message: Optional[str] = None

class HumanReviewExport(BaseModel):
    """Complete case export for human expert review"""
    case_id: str
    final_consensus: Dict[str, Any]
    expert_assessments: List[Dict[str, Any]]
    debate_transcripts: List[Dict[str, Any]]
    reasoning_summary: str
    clinical_timeline: Dict[str, Any]

@dataclass
class ExpertKey:
    """Structured expert identifier"""
    expert_id: str
    case_id: str
    specialty: str

    def __str__(self) -> str:
        return f"{self.expert_id}({self.specialty})@{self.case_id}"

# =============================================================================
# BETA OPINION POOLING WITH CONFIDENCE ESTIMATION
# =============================================================================

def beta_pool_confidence(p_vec: np.ndarray,
                         ci_mat: np.ndarray,
                         weight_vec: np.ndarray = None) -> dict:
    """
    Return pooled mean, 95% CI, and consensus confidence ∈ [0,1].

    This implements the mathematically grounded approach from the confidence
    aggregation specification.

    Args:
        p_vec: Expert point estimates, shape (k,)
        ci_mat: Expert 95% CIs, shape (k,2) [[lo, hi], ...]
        weight_vec: Optional confidence weights, shape (k,)

    Returns:
        Dict with pooled_mean, pooled_ci, consensus_conf, var_between, mean_halfwidth
    """

    lo = ci_mat[:, 0]
    hi = ci_mat[:, 1]

    # Between-expert variance (max possible is 0.25 for Bernoulli)
    var_between = np.var(p_vec, ddof=1) if p_vec.size > 1 else 0.0
    between_score = 1.0 - (var_between / 0.25)
    between_score = np.clip(between_score, 0.0, 1.0)

    # Within-expert CI width (max half-width is 0.5 for [0,1] interval)
    half_widths = (hi - lo) / 2.0
    mean_half = half_widths.mean()
    within_score = 1.0 - (mean_half / 0.5)
    within_score = np.clip(within_score, 0.0, 1.0)

    # Harmonic-mean panel confidence
    if between_score == 0 or within_score == 0:
        consensus_conf = 0.0
    else:
        consensus_conf = 2.0 / (1.0 / between_score + 1.0 / within_score)

    # Beta opinion pool with optional weighting
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
# vLLM CLIENT (unchanged)
# =============================================================================

class VLLMClient:
    """Production vLLM client with OpenAI-compatible API"""

    def __init__(self, config: DelPHEAirAKIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.VLLMClient")

        timeout = httpx.Timeout(config.timeout)
        self.client = httpx.AsyncClient(timeout=timeout)

        self.headers = {
            "Content-Type": "application/json",
        }
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
            parsed_content = json.loads(content)

            return parsed_content

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
        """Close the HTTP client"""
        await self.client.aclose()

# =============================================================================
# DATA COMPONENTS (unchanged)
# =============================================================================

class irAKIDataLoader:
    """Data loader for irAKI cases without ground truth"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.irAKIDataLoader")

    def load_patient_case(self, case_id: str) -> Dict:
        """Load patient case data focused on irAKI assessment"""
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
# ENHANCED EXPERT AGENT
# =============================================================================

class irAKIExpertAgent(RoutedAgent):
    """Clinical expert agent specialized in irAKI assessment with confidence intervals"""

    def __init__(self, expert_key: ExpertKey) -> None:
        super().__init__(f"irAKI Expert {expert_key}")
        self._expert_id = expert_key.expert_id
        self._case_id = expert_key.case_id
        self._specialty = expert_key.specialty
        self._vllm_client = VLLMClient(config)
        self.logger = logging.getLogger(f"expert.{expert_key.expert_id}")

        # load irAKI-focused clinical persona
        self._persona = self._load_expert_profile(expert_key.specialty)

    def _load_expert_profile(self, specialty: str) -> Dict:
        """Load expert persona from configuration file"""
        # Fail early if config doesn't exist
        config_path = Path(config.expert_panel_config)
        assert config_path.exists(), f"Expert panel config not found: {config_path}"

        with open(config_path, 'r') as f:
            panel_config = json.load(f)

        # Find expert by specialty
        experts = panel_config["expert_panel"]["experts"]
        for expert in experts:
            if expert["specialty"] == specialty:
                return {
                    "name": expert["name"],
                    "experience": f"{expert['experience_years']} years {specialty}",
                    "focus": ", ".join(expert["expertise"]),
                    "credentials": expert.get("credentials", ""),
                    "institution": expert.get("institution", ""),
                    "clinical_experience": expert.get("clinical_experience", ""),
                    "reasoning_style": expert.get("reasoning_style", "")
                }

        # If specialty not found, fail
        available_specialties = [e["specialty"] for e in experts]
        raise ValueError(f"Specialty '{specialty}' not found in panel config. Available: {available_specialties}")

    @message_handler
    async def handle_questionnaire(self, message: QuestionnaireMsg, ctx: MessageContext) -> None:
        """Handle Round 1 & 3 irAKI assessment with confidence intervals"""
        prompt = self._build_iraki_assessment_prompt(message)

        try:
            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt,
                response_format={"type": "json_object"}
            )

            # Validate and process confidence interval
            if "ci_iraki" in llm_response:
                ci_lower, ci_upper = llm_response["ci_iraki"]
                # Ensure CI bounds are valid
                ci_lower = max(0.0, min(ci_lower, llm_response["p_iraki"]))
                ci_upper = min(1.0, max(ci_upper, llm_response["p_iraki"]))
                llm_response["ci_iraki"] = [ci_lower, ci_upper]
            else:
                # Provide default CI if missing
                p = llm_response["p_iraki"]
                width = 0.1  # Default ±10% uncertainty
                llm_response["ci_iraki"] = [max(0.0, p - width), min(1.0, p + width)]

            # validate required fields based on round
            if message.round_phase == "round1":
                required_fields = ["scores", "evidence", "p_iraki", "ci_iraki", "confidence", "differential_diagnosis"]
                reply_class = ExpertRound1Reply
            else:  # round3
                required_fields = ["scores", "evidence", "p_iraki", "ci_iraki", "confidence",
                                   "changes_from_round1", "verdict", "final_diagnosis", "recommendations"]
                reply_class = ExpertRound3Reply

            missing = [f for f in required_fields if f not in llm_response]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            # create reply object
            reply = reply_class(
                **llm_response,
                expert_id=self._expert_id,
                case_id=message.case_id
            )

            # send via RPC to moderator
            ack_response = await self.send_message(reply, AgentId("Moderator", message.case_id))

            if not ack_response.ok:
                self.logger.error(f"Moderator rejected response: {ack_response.message}")
            else:
                self.logger.info(f"Successfully submitted {message.round_phase} irAKI assessment")

        except Exception as e:
            self.logger.error(f"Failed to generate {message.round_phase} response: {e}")

    @message_handler
    async def handle_debate_prompt(self, message: DebatePrompt, ctx: MessageContext) -> None:
        """Handle Round 2 debate with irAKI focus"""
        prompt = self._build_iraki_debate_prompt(message)

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
            self.logger.info(f"Posted irAKI debate comment for {message.q_id}")

        except Exception as e:
            self.logger.error(f"Failed to generate debate response: {e}")

    @message_handler
    async def handle_terminate_debate(self, message: TerminateDebate, ctx: MessageContext) -> None:
        """Handle debate termination signal"""
        self.logger.info(f"Exiting debate for {message.case_id}:{message.q_id}")

    def _build_iraki_assessment_prompt(self, message: QuestionnaireMsg) -> str:
        """Build irAKI-focused clinical assessment prompt with confidence interval requirement"""
        ci_instruction = """
IMPORTANT: You must provide a 95% confidence interval [lower_bound, upper_bound] for your irAKI probability estimate.
This should reflect your uncertainty in the estimate:
- Narrow intervals (e.g., [0.65, 0.75]) indicate high confidence in your assessment
- Wide intervals (e.g., [0.3, 0.8]) indicate substantial uncertainty
- The interval must contain your point estimate and be between 0.0 and 1.0
"""

        return f"""You are {self._persona['name']}, a {self._persona['experience']} specialist.
Clinical focus: {self._persona.get('focus', 'general clinical assessment')}

IMMUNE-RELATED AKI ASSESSMENT ({message.round_phase}):
Case ID: {message.case_id}

Patient Information:
{json.dumps(message.patient_info, indent=2)}

ICU Summary:
{message.icu_summary}

Immunotherapy Timeline:
{json.dumps(message.medication_history, indent=2)}

Laboratory Values & Renal Function:
{json.dumps(message.lab_values, indent=2)}

Imaging Reports:
{message.imaging_reports}

CLINICAL ASSESSMENT QUESTIONS (irAKI Focus):
Rate each question on a 9-point scale (1=strongly disagree, 9=strongly agree):

{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(message.questions))}

{ci_instruction}

INSTRUCTIONS:
- Focus specifically on immune-related vs. other AKI etiologies
- Consider temporal relationship to immunotherapy
- Evaluate alternative causes (prerenal, postrenal, other intrinsic)
- Base probability estimate on clinical evidence and {self._specialty} expertise
- Provide a realistic 95% confidence interval reflecting your uncertainty
{f"- Explain changes from Round 1 assessment" if message.round_phase == "round3" else ""}
{f"- Provide final binary verdict (irAKI: true/false)" if message.round_phase == "round3" else ""}
{f"- Suggest specific diagnosis and clinical recommendations" if message.round_phase == "round3" else ""}

Return JSON with:
- "scores": {{"Q1": <1-9>, "Q2": <1-9>, ...}}
- "evidence": {{"Q1": "<clinical reasoning>", "Q2": "<reasoning>", ...}}
- "p_iraki": <0.0-1.0 probability of immune-related AKI>
- "ci_iraki": [<lower_bound>, <upper_bound>] (95% confidence interval)
- "confidence": <0.0-1.0 in your assessment>
- "differential_diagnosis": ["<alternative etiology 1>", "<alternative 2>", ...]
{f'- "changes_from_round1": {{"Q1": "<explanation>", ...}}' if message.round_phase == "round3" else ""}
{f'- "verdict": <true/false for irAKI>' if message.round_phase == "round3" else ""}
{f'- "final_diagnosis": "<most likely specific diagnosis>"' if message.round_phase == "round3" else ""}
{f'- "recommendations": ["<clinical recommendation 1>", "<recommendation 2>", ...]' if message.round_phase == "round3" else ""}

Provide detailed clinical reasoning reflecting your {self._specialty} expertise in immune-related complications."""

    def _build_iraki_debate_prompt(self, message: DebatePrompt) -> str:
        """Build debate prompt for irAKI-focused Round 2"""
        return f"""You are {self._persona['name']}, participating in an irAKI classification expert debate.

DEBATE TOPIC: {message.q_id}
ROUND: {message.round_no}
CLINICAL CONTEXT: {message.clinical_context}

MINORITY POSITION TO CONSIDER:
{message.minority_view}

As a {self._specialty} specialist, provide your clinical perspective on this irAKI classification disagreement.

INSTRUCTIONS:
- Present evidence-based clinical reasoning from your specialty perspective
- Focus on immune-related vs. other AKI mechanisms
- Address the minority position constructively
- Include relevant medical literature if applicable
- If satisfied with the discussion, set "satisfied": true

Return JSON:
{{"text": "<your clinical argument and reasoning>", "citations": ["<reference1>", "<reference2>"], "satisfied": <true/false>}}"""

# =============================================================================
# ENHANCED MODERATOR AGENT with Beta Pooling
# =============================================================================

class irAKIModeratorAgent(RoutedAgent):
    """Master agent coordinating irAKI Delphi process with enhanced confidence estimation"""

    def __init__(self, case_id: str) -> None:
        super().__init__(f"irAKI Moderator for case {case_id}")
        self._case_id = case_id
        self._data_loader = irAKIDataLoader()

        # round tracking
        self._round1_replies: List[ExpertRound1Reply] = []
        self._round3_replies: List[ExpertRound3Reply] = []
        self._chat_logs: Dict[str, List[Dict]] = defaultdict(list)

        # debate management
        self._active_debates: Dict[str, Dict] = {}

        # synchronization
        self._pending_round1: Set[str] = set()
        self._pending_round3: Set[str] = set()
        self._round1_done = asyncio.Event()
        self._round3_done = asyncio.Event()

        self.logger = logging.getLogger(f"iraki_moderator.{case_id}")

    def _log_chat(self, author_id: str, role: str, msg: BaseModel) -> None:
        """Log message to expert's chat history for human review"""
        self._chat_logs[author_id].append({
            "role": role,
            "payload": msg.model_dump(),
            "timestamp": time.time()
        })

    @message_handler
    async def handle_start_case(self, message: StartCase, ctx: MessageContext) -> None:
        """Bootstrap irAKI Delphi process"""
        self.logger.info(f"Starting irAKI Delphi process for case {message.case_id}")
        patient_data = self._data_loader.load_patient_case(message.case_id)
        await self._run_round1(patient_data)

    async def _run_round1(self, patient_data: Dict) -> None:
        """Execute Round 1: individual irAKI assessments"""
        self.logger.info("=== ROUND 1: Individual irAKI Assessments ===")

        questions = self._load_assessment_questions()

        # initialize pending tracking
        self._pending_round1 = {
            self._generate_expert_key(i, self._case_id)
            for i in range(config.expert_count)
        }
        self._round1_done.clear()

        # create and broadcast questionnaire
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
        self.logger.info(f"Broadcast Round 1 irAKI questionnaire to {config.expert_count} experts")

        await self._wait_for_round_completion("round1")

    def _load_assessment_questions(self) -> List[str]:
        """Load irAKI assessment questions from configuration file"""
        # Fail early if questionnaire config doesn't exist
        questionnaire_path = Path(config.questionnaire_config)
        assert questionnaire_path.exists(), f"Questionnaire config not found: {questionnaire_path}"

        with open(questionnaire_path, 'r') as f:
            questionnaire_config = json.load(f)

        # Extract questions from config structure
        questions_data = questionnaire_config["questionnaire"]["questions"]
        questions = [q["question"] for q in questions_data]

        # Update questions count based on actual loaded questions
        config.questions_count = len(questions)

        self.logger.info(f"Loaded {len(questions)} assessment questions from {questionnaire_path}")
        return questions

    def _generate_expert_key(self, expert_idx: int, case_id: str) -> str:
        """Generate canonical expert key"""
        return f"exp{expert_idx}:{case_id}"

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
                await self._compute_final_iraki_consensus()

        except asyncio.TimeoutError:
            pending = self._pending_round1 if round_phase == "round1" else self._pending_round3
            self.logger.warning(f"Timeout in {round_phase}: {len(pending)} experts pending")

            if round_phase == "round1":
                await self._run_round2()
            else:
                await self._compute_final_iraki_consensus()

    @rpc
    async def record_round1(self, message: ExpertRound1Reply, ctx: MessageContext) -> AckMsg:
        """Collect Round 1 expert replies"""
        self._log_chat(message.expert_id, "expert_reply", message)
        self._round1_replies.append(message)

        expert_key = f"{message.expert_id}:{message.case_id}"
        self._pending_round1.discard(expert_key)

        self.logger.debug(f"Round 1 reply from {expert_key} ({len(self._pending_round1)} pending)")

        if not self._pending_round1:
            self._round1_done.set()

        return AckMsg(ok=True, message="Round 1 irAKI reply recorded")

    @rpc
    async def record_round3(self, message: ExpertRound3Reply, ctx: MessageContext) -> AckMsg:
        """Collect Round 3 expert replies"""
        self._log_chat(message.expert_id, "expert_reply", message)
        self._round3_replies.append(message)

        expert_key = f"{message.expert_id}:{message.case_id}"
        self._pending_round3.discard(expert_key)

        self.logger.debug(f"Round 3 reply from {expert_key} ({len(self._pending_round3)} pending)")

        if not self._pending_round3:
            self._round3_done.set()

        return AckMsg(ok=True, message="Round 3 irAKI reply recorded")

    async def _run_round2(self) -> None:
        """Execute Round 2: debate conflicts in irAKI assessment"""
        self.logger.info("=== ROUND 2: irAKI Classification Conflict Resolution ===")

        conflicts = self._detect_iraki_conflicts()

        if not conflicts:
            self.logger.info("No conflicts detected, proceeding to Round 3")
            await self._run_round3()
            return

        self.logger.info(f"Detected conflicts in {len(conflicts)} questions: {list(conflicts.keys())}")

        # initialize debates
        for q_id in conflicts:
            self._active_debates[q_id] = {
                "round": 1,
                "history": deque(maxlen=50),
                "satisfied": set(),
                "lock": asyncio.Lock()
            }

        # start all debates
        for q_id, conflict_data in conflicts.items():
            await self._start_debate(q_id, conflict_data)

    def _detect_iraki_conflicts(self) -> Dict[str, Dict]:
        """Detect questions with significant disagreement on irAKI classification"""
        conflicts = {}

        for i in range(config.questions_count):
            q_id = f"Q{i+1}"
            scores = [
                reply.scores.get(q_id, 5)
                for reply in self._round1_replies
                if q_id in reply.scores
            ]

            if len(scores) > 1 and max(scores) - min(scores) >= config.conflict_threshold:
                min_score, max_score = min(scores), max(scores)

                low_scorers = [
                    r.expert_id for r in self._round1_replies
                    if r.scores.get(q_id, 5) <= min_score + 1
                ]
                high_scorers = [
                    r.expert_id for r in self._round1_replies
                    if r.scores.get(q_id, 5) >= max_score - 1
                ]

                all_participants = list(set(low_scorers + high_scorers))[:config.max_debate_participants]

                conflicts[q_id] = {
                    "minority_view": self._compile_minority_view(q_id, low_scorers),
                    "participating_experts": all_participants,
                    "score_range": f"{min_score}-{max_score}",
                    "clinical_context": self._get_question_context(i)
                }

                self.logger.info(f"irAKI conflict in {q_id}: scores {min_score}-{max_score}, "
                                 f"{len(all_participants)} participants")

        return conflicts

    def _get_question_context(self, question_idx: int) -> str:
        """Get clinical context for specific question"""
        contexts = [
            "temporal relationship analysis",
            "prerenal etiology evaluation",
            "postrenal etiology evaluation",
            "other intrinsic causes assessment",
            "urinalysis interpretation",
            "immune activation markers",
            "immunotherapy response pattern",
            "complement and immune markers",
            "eosinophilia significance",
            "drug interaction assessment",
            "renal biopsy consideration",
            "overall clinical gestalt"
        ]
        return contexts[question_idx] if question_idx < len(contexts) else "clinical assessment"

    def _compile_minority_view(self, q_id: str, low_scorers: List[str]) -> str:
        """Compile minority position for irAKI debate"""
        minority_evidence = []
        for reply in self._round1_replies:
            if reply.expert_id in low_scorers and q_id in reply.evidence:
                minority_evidence.append(f"{reply.expert_id}: {reply.evidence[q_id]}")

        return "Minority view: " + " | ".join(minority_evidence[:2])

    async def _start_debate(self, q_id: str, conflict_data: Dict) -> None:
        """Start confidence-ordered debate for irAKI question"""
        participants_with_confidence = sorted(
            (
                (r.expert_id, r.confidence)
                for r in self._round1_replies
                if r.expert_id in conflict_data["participating_experts"]
            ),
            key=lambda x: x[1]
        )

        order = [expert_id for expert_id, _ in participants_with_confidence]

        self._active_debates[q_id].update({
            "order": order,
            "pointer": 0,
            "minority_view": conflict_data["minority_view"],
            "clinical_context": conflict_data["clinical_context"]
        })

        self.logger.info(f"Starting irAKI debate for {q_id} with order: {order}")
        await self._invite_next_speaker(q_id)

    async def _invite_next_speaker(self, q_id: str) -> None:
        """Invite next unsatisfied expert to speak in irAKI debate"""
        state = self._active_debates[q_id]
        order = state["order"]

        for _ in range(len(order)):
            expert_id = order[state["pointer"]]
            state["pointer"] = (state["pointer"] + 1) % len(order)

            if expert_id not in state["satisfied"]:
                prompt = DebatePrompt(
                    case_id=self._case_id,
                    q_id=q_id,
                    minority_view=state["minority_view"],
                    round_no=state["round"],
                    participating_experts=[expert_id],
                    clinical_context=state["clinical_context"]
                )

                topic = TopicId("debate", f"{self._case_id}:{q_id}:{expert_id}")
                await self.publish_message(prompt, topic)
                return

        await self._terminate_debate(q_id, "all_satisfied")

    @message_handler
    async def handle_debate_comment(self, message: DebateComment, ctx: MessageContext) -> None:
        """Handle expert debate comments for irAKI classification"""
        q_id = message.q_id
        if q_id not in self._active_debates:
            return

        state = self._active_debates[q_id]
        self._log_chat(message.author, "debate_comment", message)

        async with state["lock"]:
            state["history"].append(message)

            if message.satisfied:
                state["satisfied"].add(message.author)
                self.logger.info(f"{message.author} satisfied for {q_id}")

            if (state["round"] >= config.max_debate_rounds or
                    len(state["satisfied"]) == len(state["order"])):

                reason = ("all_satisfied" if len(state["satisfied"]) == len(state["order"])
                          else "hard_cap_reached")
                await self._terminate_debate(q_id, reason)
                return

            if state["pointer"] == 0:
                state["round"] += 1

            await self._invite_next_speaker(q_id)

    async def _terminate_debate(self, q_id: str, reason: str) -> None:
        """Terminate irAKI debate and notify participants"""
        term = TerminateDebate(case_id=self._case_id, q_id=q_id, reason=reason)

        for expert_id in self._active_debates[q_id]["order"]:
            topic = TopicId("debate", f"{self._case_id}:{q_id}:{expert_id}")
            await self.publish_message(term, topic)

        del self._active_debates[q_id]
        self.logger.info(f"Terminated irAKI debate {q_id}: {reason}")

        if not self._active_debates:
            await self._run_round3()

    async def _run_round3(self) -> None:
        """Execute Round 3: final irAKI consensus"""
        self.logger.info("=== ROUND 3: Final irAKI Consensus ===")

        patient_data = self._data_loader.load_patient_case(self._case_id)
        questions = self._load_assessment_questions()

        self._pending_round3 = {
            self._generate_expert_key(i, self._case_id)
            for i in range(config.expert_count)
        }
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
        self.logger.info("Broadcast Round 3 irAKI questionnaire")

        await self._wait_for_round_completion("round3")

    async def _compute_final_iraki_consensus(self) -> None:
        """Compute beta pooling consensus for irAKI classification with enhanced confidence"""
        self.logger.info("=== COMPUTING FINAL irAKI CONSENSUS WITH BETA POOLING ===")

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

        # Collect diagnoses and recommendations
        all_diagnoses = [r.final_diagnosis for r in self._round3_replies]
        all_recommendations = []
        for r in self._round3_replies:
            all_recommendations.extend(r.recommendations)

        # Confidence categorization
        consensus_conf = stats['consensus_conf']
        if consensus_conf >= config.consensus_threshold:
            confidence_level = "HIGH"
        elif consensus_conf >= 0.6:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"

        # Enhanced logging with beta pooling results
        self.logger.info("=" * 80)
        self.logger.info(f"CASE {self._case_id} irAKI CONSENSUS RESULTS (BETA POOLING):")
        self.logger.info("-" * 80)
        self.logger.info(f"Beta Pooled P(irAKI):    {stats['pooled_mean']:.3f}")
        self.logger.info(f"95% Credible Interval:   [{stats['pooled_ci'][0]:.3f}, {stats['pooled_ci'][1]:.3f}]")
        self.logger.info(f"Consensus Confidence:    {consensus_conf:.3f} ({confidence_level})")
        self.logger.info(f"Between-Expert Score:    {stats['between_score']:.3f}")
        self.logger.info(f"Within-Expert Score:     {stats['within_score']:.3f}")
        self.logger.info("-" * 80)
        self.logger.info(f"Between-Expert Variance: {stats['var_between']:.4f}")
        self.logger.info(f"Mean CI Half-Width:      {stats['mean_halfwidth']:.3f}")
        self.logger.info(f"Majority Vote Verdict:   {'irAKI' if consensus_verdict else 'Other AKI'}")
        self.logger.info("-" * 80)
        self.logger.info(f"Individual Probabilities: {[f'{p:.3f}' for p in p_vec]}")
        self.logger.info(f"Individual Confidences:   {[f'{c:.3f}' for c in w_vec]}")
        self.logger.info(f"Expert Verdicts:         {[r.verdict for r in self._round3_replies]}")
        self.logger.info("-" * 80)
        self.logger.info(f"Proposed Diagnoses:      {set(all_diagnoses)}")
        self.logger.info(f"Total Recommendations:   {len(set(all_recommendations))} unique")
        self.logger.info("=" * 80)

        # Clinical interpretation
        pooled_prob = stats['pooled_mean']
        if pooled_prob >= 0.7 and consensus_conf >= 0.7:
            clinical_recommendation = "STRONG evidence for irAKI - Recommend biopsy confirmation and steroid therapy"
        elif pooled_prob >= 0.5 and consensus_conf >= 0.6:
            clinical_recommendation = "MODERATE evidence for irAKI - Consider biopsy and empiric therapy"
        elif pooled_prob < 0.3 and consensus_conf >= 0.6:
            clinical_recommendation = "LOW probability irAKI - Focus on alternative AKI causes"
        else:
            clinical_recommendation = "UNCERTAIN - Additional evaluation needed, consider multidisciplinary review"

        self.logger.info(f"CLINICAL RECOMMENDATION: {clinical_recommendation}")
        self.logger.info("=" * 80)

        # Export for human review with enhanced statistics
        if config.export_full_transcripts:
            await self._export_for_human_review(stats, consensus_verdict)

    async def _export_for_human_review(self, stats: Dict, consensus_verdict: bool) -> None:
        """Export complete case with enhanced confidence statistics for human expert chart review"""

        # Compile expert assessments with confidence intervals
        expert_assessments = []
        for r1, r3 in zip(self._round1_replies, self._round3_replies):
            expert_assessments.append({
                "expert_id": r1.expert_id,
                "specialty": next((s for s in config.expert_specialties if s in r1.expert_id.lower()), "unknown"),
                "round1": {
                    "scores": r1.scores,
                    "probability": r1.p_iraki,
                    "confidence_interval": r1.ci_iraki,
                    "confidence": r1.confidence,
                    "differential": r1.differential_diagnosis
                },
                "round3": {
                    "scores": r3.scores,
                    "probability": r3.p_iraki,
                    "confidence_interval": r3.ci_iraki,
                    "confidence": r3.confidence,
                    "verdict": r3.verdict,
                    "diagnosis": r3.final_diagnosis,
                    "recommendations": r3.recommendations,
                    "changes": r3.changes_from_round1
                },
                "ci_width": r3.ci_iraki[1] - r3.ci_iraki[0] if len(r3.ci_iraki) >= 2 else 0.0
            })

        # Compile debate transcripts
        debate_transcripts = []
        for expert_id, chat_log in self._chat_logs.items():
            debate_comments = [msg for msg in chat_log if msg["role"] == "debate_comment"]
            if debate_comments:
                debate_transcripts.append({
                    "expert_id": expert_id,
                    "comments": debate_comments
                })

        # Generate enhanced reasoning summary
        reasoning_summary = self._generate_enhanced_reasoning_summary(stats)

        export_data = HumanReviewExport(
            case_id=self._case_id,
            final_consensus={
                "beta_pooled_probability": stats['pooled_mean'],
                "credible_interval_95": stats['pooled_ci'],
                "consensus_confidence": stats['consensus_conf'],
                "between_expert_score": stats['between_score'],
                "within_expert_score": stats['within_score'],
                "between_expert_variance": stats['var_between'],
                "mean_ci_halfwidth": stats['mean_halfwidth'],
                "majority_verdict": consensus_verdict,
                "confidence_level": ("HIGH" if stats['consensus_conf'] >= config.consensus_threshold
                                     else "MODERATE" if stats['consensus_conf'] >= 0.6 else "LOW"),
                "weighting_method": "beta_opinion_pooling_with_confidence_weighting",
                "expert_count": len(self._round3_replies)
            },
            expert_assessments=expert_assessments,
            debate_transcripts=debate_transcripts,
            reasoning_summary=reasoning_summary,
            clinical_timeline=self._extract_clinical_timeline()
        )

        # Save to file for human review
        export_filename = f"enhanced_human_review_{self._case_id}_{int(time.time())}.json"
        with open(export_filename, 'w') as f:
            json.dump(export_data.model_dump(), f, indent=2)

        self.logger.info(f"Exported enhanced case analysis for human review: {export_filename}")

    def _generate_enhanced_reasoning_summary(self, stats: Dict) -> str:
        """Generate enhanced executive summary with confidence metrics"""
        pro_iraki_reasons = []
        against_iraki_reasons = []
        confidence_analysis = []

        for reply in self._round3_replies:
            ci_width = reply.ci_iraki[1] - reply.ci_iraki[0] if len(reply.ci_iraki) >= 2 else 0.0
            conf_desc = "narrow" if ci_width < 0.2 else "moderate" if ci_width < 0.4 else "wide"

            expert_analysis = f"{reply.expert_id} (CI: {conf_desc}, conf: {reply.confidence:.2f}): {reply.final_diagnosis}"

            if reply.verdict:  # pro-irAKI
                pro_iraki_reasons.append(expert_analysis)
            else:  # against irAKI
                against_iraki_reasons.append(expert_analysis)

        summary = f"""
ENHANCED EXPERT REASONING SUMMARY WITH CONFIDENCE ANALYSIS:

Beta Pooling Results:
- Pooled Probability: {stats['pooled_mean']:.3f} [95% CI: {stats['pooled_ci'][0]:.3f}-{stats['pooled_ci'][1]:.3f}]
- Consensus Confidence: {stats['consensus_conf']:.3f}
- Between-Expert Agreement: {stats['between_score']:.3f}
- Within-Expert Precision: {stats['within_score']:.3f}

Pro-irAKI Arguments ({len(pro_iraki_reasons)} experts):
{chr(10).join('- ' + reason for reason in pro_iraki_reasons)}

Alternative Diagnosis Arguments ({len(against_iraki_reasons)} experts):
{chr(10).join('- ' + reason for reason in against_iraki_reasons)}

Confidence Assessment:
- Expert agreement (between-score): {'Strong' if stats['between_score'] > 0.8 else 'Moderate' if stats['between_score'] > 0.6 else 'Weak'}
- Individual precision (within-score): {'High' if stats['within_score'] > 0.8 else 'Moderate' if stats['within_score'] > 0.6 else 'Low'}
- Overall consensus confidence: {'High' if stats['consensus_conf'] > 0.8 else 'Moderate' if stats['consensus_conf'] > 0.6 else 'Low'}

Key Debates: {len([d for d in self._active_debates])} question(s) required debate resolution
Aggregation Method: Beta opinion pooling with confidence-weighted harmonic mean
Mathematical Framework: Bayesian belief aggregation with uncertainty quantification
"""
        return summary.strip()

    def _extract_clinical_timeline(self) -> Dict[str, Any]:
        """Extract key clinical timeline for human review"""
        return {
            "immunotherapy_start": "2024-01-15",
            "last_immunotherapy": "2024-07-10",
            "aki_onset": "approximately 2 weeks prior to admission",
            "creatinine_progression": "1.1 → 3.2 mg/dL over 2 weeks",
            "key_clinical_events": [
                "immunotherapy cycle 6 completed",
                "lisinopril held due to AKI",
                "no nephrotoxic exposures identified"
            ]
        }

# =============================================================================
# CLI INTERFACE (enhanced for confidence estimation)
# =============================================================================

async def main():
    """Main entry point for enhanced irAKI classification system"""
    parser = argparse.ArgumentParser(description="Enhanced DelPHEA-irAKI: Clinical Multi-Agent irAKI Classification with Beta Pooling")

    # vLLM Configuration (AWS fixed)
    parser.add_argument("--endpoint", default="http://172.31.11.192:8000",
                        help="vLLM server endpoint (AWS fixed)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name in HuggingFace format")
    parser.add_argument("--api-key", help="API key for vLLM server")

    # Inference Parameters
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for irAKI reasoning")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=3072,
                        help="Maximum tokens per response")

    # irAKI Case Configuration
    parser.add_argument("--case-id", default="iraki_case_001",
                        help="irAKI case identifier")
    parser.add_argument("--experts", type=int, default=8,
                        help="Number of expert agents")

    # Configuration File Paths
    parser.add_argument("--expert-panel-config", default="config/panel.json",
                        help="Expert panel configuration file")
    parser.add_argument("--questionnaire-config", default="config/questionnaire.json",
                        help="Clinical questionnaire configuration file")

    # Enhanced Confidence Configuration
    parser.add_argument("--consensus-threshold", type=float, default=0.8,
                        help="Minimum consensus confidence for high agreement")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Minimum individual confidence threshold")

    # Human Review Options
    parser.add_argument("--export-transcripts", action="store_true", default=True,
                        help="Export full transcripts for human review")
    parser.add_argument("--detailed-reasoning", action="store_true", default=True,
                        help="Include detailed reasoning chains")

    # System Options
    parser.add_argument("--health-check", action="store_true",
                        help="Perform health check and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Update global configuration
    config.vllm_endpoint = args.endpoint
    config.model_name = args.model
    config.api_key = args.api_key
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.max_tokens = args.max_tokens
    config.expert_count = args.experts
    config.expert_panel_config = args.expert_panel_config
    config.questionnaire_config = args.questionnaire_config
    config.consensus_threshold = args.consensus_threshold
    config.confidence_threshold = args.confidence_threshold
    config.export_full_transcripts = args.export_transcripts
    config.include_reasoning_chains = args.detailed_reasoning

    # Validate config files exist before starting
    expert_panel_path = Path(config.expert_panel_config)
    questionnaire_path = Path(config.questionnaire_config)

    assert expert_panel_path.exists(), f"Expert panel config not found: {expert_panel_path}"
    assert questionnaire_path.exists(), f"Questionnaire config not found: {questionnaire_path}"

    try:
        if args.health_check:
            vllm_client = VLLMClient(config)
            healthy = await vllm_client.health_check()
            await vllm_client.close()

            if healthy:
                print("✓ Enhanced DelPHEA-irAKI system healthy")
                print(f"✓ vLLM endpoint: {config.vllm_endpoint}")
                print(f"✓ Model: {config.model_name}")
                print("✓ Ready for irAKI classification with beta pooling")
                return 0
            else:
                print("✗ Enhanced DelPHEA-irAKI system health check failed")
                return 1
        else:
            print(f"Starting Enhanced DelPHEA-irAKI for case: {args.case_id}")
            print(f"vLLM endpoint: {config.vllm_endpoint}")
            print(f"Model: {config.model_name}")
            print(f"Expert panel config: {config.expert_panel_config}")
            print(f"Questionnaire config: {config.questionnaire_config}")
            print(f"Experts: {config.expert_count}")
            print(f"Focus: immune-related AKI vs other AKI classification")
            print(f"Aggregation: Beta opinion pooling with confidence intervals")
            print(f"Consensus threshold: {config.consensus_threshold}")
            print(f"Export for human review: {config.export_full_transcripts}")
            print("-" * 60)

            # Create runtime and moderator
            runtime = SingleThreadedAgentRuntime()

            # Register moderator
            await runtime.register("Moderator", lambda key: irAKIModeratorAgent(key))

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

            print("\n✓ Enhanced DelPHEA-irAKI completed successfully!")
            print("✓ Case exported with confidence analysis for human expert chart review")
            print("✓ Beta pooling consensus with mathematically grounded confidence estimation")
            return 0

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
