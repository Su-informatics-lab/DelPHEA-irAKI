"""
DelPHEA-irAKI: Delphi Personalized Health Explainable Agents for immune-related AKI Classification
================================================================================================

High-level
----------
* Multi-agent Delphi workflow (Moderator + N LLM-backed Experts) for irAKI vs other AKI classification
* 3 structured rounds → consensus & confidence-weighted probability of irAKI diagnosis
* Modular expert panel and questionnaire configuration via JSON files
* Optional literature search integration (PubMed/bioRxiv) for evidence-based reasoning
* Equal expert weighting (no ARN learning due to lack of ground truth)
* Runs against a vLLM server; designed for AWS H100/A100
* Human expert chart review validation instead of automated learning

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
       │          Expert agents (LLM)         │◄──── Literature Search │
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
                │ transcripts → Equal Weighting                       │
                ▼                                                     │
          ┌─────────────┐                                             │
          │  Consensus  │                                             │
          │ Calculation │                                             │
          └─────┬───────┘                                             │
                │                                                     │
                ▼                                                     │
          Human Expert Chart Review                                   │

Modular Components
------------------
* config/expert_panel.json: Expert panel configuration with specialties and personas
* config/questionnaire_iraki.json: Clinical assessment questions for irAKI classification
* modules/literature_search.py: PubMed/bioRxiv integration for evidence-based reasoning
* Flexible data loader for various input formats (to be customized based on actual data)

Outputs
-------
* Final consensus (probability + binary verdict for irAKI) written to logs/stdout
* Complete decision transcript for human expert review
* Detailed reasoning chains for each question and expert
* Literature citations and evidence integration (if enabled)

irAKI Classification Focus
-------------------------
* Target: Distinguish immune-related AKI from other AKI etiologies
* Expert specialties: Configurable via JSON (Oncology, Nephrology, Pathology, Pharmacy, etc.)
* Clinical questions: 12 irAKI-specific questions based on published literature
* Timeline analysis of immune therapy exposure and AKI onset
* Biomarker interpretation relevant to immune-mediated kidney injury
* Literature integration for evidence-based clinical reasoning

"""

import asyncio
import json
import logging
import re
import time
import argparse
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
from pydantic import BaseModel, Field, ConfigDict
import httpx
import numpy as np

from autogen_core import (
    SingleThreadedAgentRuntime,
    MessageContext,
    TopicId,
    AgentId,
    RoutedAgent,
    message_handler,
    type_subscription,
    rpc
)

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
    temperature: float = 0.7  # slightly higher for nuanced clinical reasoning
    top_p: float = 0.9
    max_tokens: int = 3072  # more tokens for detailed clinical reasoning
    timeout: float = 120.0

    # irAKI Classification Configuration
    questions_count: int = 12  # comprehensive irAKI assessment
    conflict_threshold: int = 3  # score difference to trigger debate (9-point scale)
    max_debate_rounds: int = 6  # allow more rounds for complex cases
    max_debate_participants: int = 8

    # Expert Configuration (irAKI-focused specialties)
    expert_count: int = 8
    expert_specialties: List[str] = None

    # Consensus Configuration (no ARN learning)
    use_equal_weighting: bool = True  # no ground truth available
    confidence_threshold: float = 0.7  # minimum confidence for consensus

    # Timeouts (seconds)
    round1_timeout: int = 900  # 15 minutes for initial assessment
    round3_timeout: int = 600  # 10 minutes for final assessment
    debate_timeout: int = 240  # 4 minutes per debate round

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
    medication_history: dict  # immune therapy timeline
    lab_values: dict  # renal function trajectory
    imaging_reports: str
    questions: List[str]
    round_phase: str = "round1"

class ExpertRound1Reply(BaseModel):
    """Round 1: individual expert irAKI assessment"""
    model_config = ConfigDict(strict=True)

    case_id: str
    expert_id: str
    scores: Dict[str, int]  # question_id → 1-9 (9-point Likert)
    evidence: Dict[str, str]  # question_id → clinical reasoning
    p_iraki: float = Field(ge=0.0, le=1.0)  # probability of irAKI
    confidence: float = Field(ge=0.0, le=1.0)
    differential_diagnosis: List[str]  # alternative AKI etiologies considered

class ExpertRound3Reply(BaseModel):
    """Round 3: final expert irAKI assessment"""
    model_config = ConfigDict(strict=True)

    case_id: str
    expert_id: str
    scores: Dict[str, int]
    evidence: Dict[str, str]
    p_iraki: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    changes_from_round1: Dict[str, str]
    verdict: bool  # final binary decision: irAKI vs other AKI
    final_diagnosis: str  # most likely specific etiology
    recommendations: List[str]  # clinical recommendations

class DebatePrompt(BaseModel):
    """Round 2: initiate debate for specific question"""
    model_config = ConfigDict(strict=True)

    case_id: str
    q_id: str
    minority_view: str
    round_no: int
    participating_experts: List[str]
    clinical_context: str  # specific clinical context for this question

class DebateComment(BaseModel):
    """Expert contribution during Round-2 debate"""
    q_id: str
    author: str
    text: str
    citations: List[str] = []  # medical references if any
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
# vLLM CLIENT (unchanged from original)
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
# DATA COMPONENTS (modified for irAKI)
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
# EXPERT AGENT (modified for irAKI focus)
# =============================================================================

class irAKIExpertAgent(RoutedAgent):
    """Clinical expert agent specialized in irAKI assessment"""

    def __init__(self, expert_key: ExpertKey) -> None:
        super().__init__(f"irAKI Expert {expert_key}")
        self._expert_id = expert_key.expert_id
        self._case_id = expert_key.case_id
        self._specialty = expert_key.specialty
        self._vllm_client = VLLMClient(config)
        self.logger = logging.getLogger(f"expert.{expert_key.expert_id}")

        # load irAKI-focused clinical persona
        self._persona = self._load_iraki_expert_persona(expert_key.specialty)

    def _load_iraki_expert_persona(self, specialty: str) -> Dict:
        """Load expert persona with irAKI-specific expertise"""
        personas = {
            "oncology": {
                "name": "Dr. Sarah Chen",
                "experience": "15 years medical oncology",
                "focus": "immune checkpoint inhibitor toxicities, irAE management, cancer immunotherapy"
            },
            "nephrology": {
                "name": "Dr. Robert Martinez",
                "experience": "18 years nephrology",
                "focus": "immune-mediated kidney disease, drug-induced AKI, glomerulonephritis"
            },
            "rheumatology": {
                "name": "Dr. Maria Rodriguez",
                "experience": "12 years rheumatology",
                "focus": "autoimmune kidney disease, lupus nephritis, immune complex diseases"
            },
            "immunology": {
                "name": "Dr. James Wilson",
                "experience": "14 years clinical immunology",
                "focus": "immune-mediated organ dysfunction, autoantibody-mediated disease"
            },
            "critical_care": {
                "name": "Dr. Lisa Zhang",
                "experience": "16 years critical care",
                "focus": "ICU-based AKI, immune-related complications in critically ill"
            },
            "clinical_pharmacology": {
                "name": "Dr. Michael Brown",
                "experience": "12 years clinical pharmacology",
                "focus": "drug-induced kidney injury, immunosuppressive drug interactions"
            },
            "pathology": {
                "name": "Dr. Jennifer Taylor",
                "experience": "20 years renal pathology",
                "focus": "immune-mediated kidney pathology, drug-induced nephritis"
            },
            "informatics": {
                "name": "Dr. David Wong",
                "experience": "10 years clinical informatics",
                "focus": "clinical decision support for immune-related adverse events"
            }
        }

        return personas.get(specialty, {
            "name": f"Dr. {self._expert_id.title()}",
            "experience": f"10+ years {specialty}",
            "focus": f"clinical expertise in {specialty} and immune-related complications"
        })

    @message_handler
    async def handle_questionnaire(self, message: QuestionnaireMsg, ctx: MessageContext) -> None:
        """Handle Round 1 & 3 irAKI assessment"""
        prompt = self._build_iraki_assessment_prompt(message)

        try:
            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt,
                response_format={"type": "json_object"}
            )

            # validate required fields based on round
            if message.round_phase == "round1":
                required_fields = ["scores", "evidence", "p_iraki", "confidence", "differential_diagnosis"]
                reply_class = ExpertRound1Reply
            else:  # round3
                required_fields = ["scores", "evidence", "p_iraki", "confidence",
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
        """Build irAKI-focused clinical assessment prompt"""
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

INSTRUCTIONS:
- Focus specifically on immune-related vs. other AKI etiologies
- Consider temporal relationship to immunotherapy
- Evaluate alternative causes (prerenal, postrenal, other intrinsic)
- Base probability estimate on clinical evidence and {self._specialty} expertise
{f"- Explain changes from Round 1 assessment" if message.round_phase == "round3" else ""}
{f"- Provide final binary verdict (irAKI: true/false)" if message.round_phase == "round3" else ""}
{f"- Suggest specific diagnosis and clinical recommendations" if message.round_phase == "round3" else ""}

Return JSON with:
- "scores": {{"Q1": <1-9>, "Q2": <1-9>, ...}}
- "evidence": {{"Q1": "<clinical reasoning>", "Q2": "<reasoning>", ...}}
- "p_iraki": <0.0-1.0 probability of immune-related AKI>
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
# MODERATOR AGENT (modified for irAKI and equal weighting)
# =============================================================================

class irAKIModeratorAgent(RoutedAgent):
    """Master agent coordinating irAKI Delphi process without ARN learning"""

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

        questions = self._generate_iraki_questions()

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

    def _generate_iraki_questions(self) -> List[str]:
        """Generate 12 irAKI-specific assessment questions"""
        return [
            "Temporal relationship supports irAKI (1=no temporal relationship, 9=clear temporal association)",
            "Exclusion of prerenal causes (1=likely prerenal, 9=prerenal causes well excluded)",
            "Exclusion of postrenal causes (1=likely postrenal, 9=postrenal causes well excluded)",
            "Exclusion of other intrinsic causes (1=other intrinsic likely, 9=other intrinsic well excluded)",
            "Urinalysis pattern consistent with irAKI (1=inconsistent, 9=highly consistent with immune injury)",
            "Systemic immune activation markers (1=absent, 9=strong evidence of immune activation)",
            "Response pattern to immunotherapy exposure (1=no clear pattern, 9=classic irAE presentation)",
            "Complement levels and immune markers (1=normal/uninformative, 9=clearly abnormal/supportive)",
            "Eosinophilia supporting immune mechanism (1=absent/uninformative, 9=clearly supportive)",
            "Exclusion of drug interactions and nephrotoxins (1=likely other drugs, 9=well excluded)",
            "Renal biopsy indication and expected findings (1=not indicated/non-immune, 9=indicated/immune pattern expected)",
            "Overall clinical gestalt for irAKI (1=very unlikely irAKI, 9=very likely irAKI)"
        ]

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
        questions = self._generate_iraki_questions()

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
        """Compute equal-weighted final consensus for irAKI classification"""
        self.logger.info("=== COMPUTING FINAL irAKI CONSENSUS ===")

        if not self._round3_replies:
            self.logger.error("No Round 3 replies received")
            return

        # compute equal-weighted consensus (no ARN learning)
        n_experts = len(self._round3_replies)
        mean_probability = sum(r.p_iraki for r in self._round3_replies) / n_experts
        consensus_verdict = sum(r.verdict for r in self._round3_replies) > n_experts / 2

        # collect all differential diagnoses and recommendations
        all_diagnoses = [r.final_diagnosis for r in self._round3_replies]
        all_recommendations = []
        for r in self._round3_replies:
            all_recommendations.extend(r.recommendations)

        # compute confidence metrics
        confidence_scores = [r.confidence for r in self._round3_replies]
        mean_confidence = sum(confidence_scores) / len(confidence_scores)
        confidence_agreement = len([c for c in confidence_scores if c >= config.confidence_threshold]) / len(confidence_scores)

        # log final results for human review
        self.logger.info("=" * 70)
        self.logger.info(f"CASE {self._case_id} irAKI CONSENSUS RESULTS:")
        self.logger.info(f"Mean irAKI Probability: {mean_probability:.3f}")
        self.logger.info(f"Consensus Verdict: {'irAKI' if consensus_verdict else 'Other AKI'}")
        self.logger.info(f"Expert Verdicts: {[r.verdict for r in self._round3_replies]}")
        self.logger.info(f"Expert Confidences: {[f'{c:.2f}' for c in confidence_scores]}")
        self.logger.info(f"Mean Confidence: {mean_confidence:.3f}")
        self.logger.info(f"High Confidence Rate: {confidence_agreement:.1%}")
        self.logger.info(f"Proposed Diagnoses: {set(all_diagnoses)}")
        self.logger.info(f"All Recommendations: {len(set(all_recommendations))} unique recommendations")
        self.logger.info(f"Equal Weighting Applied (No ARN Learning)")
        self.logger.info("=" * 70)

        # export for human review
        if config.export_full_transcripts:
            await self._export_for_human_review(mean_probability, consensus_verdict, mean_confidence)

    async def _export_for_human_review(self, final_prob: float, final_verdict: bool, confidence: float) -> None:
        """Export complete case for human expert chart review"""

        # compile expert assessments
        expert_assessments = []
        for r1, r3 in zip(self._round1_replies, self._round3_replies):
            expert_assessments.append({
                "expert_id": r1.expert_id,
                "specialty": next((s for s in config.expert_specialties if s in r1.expert_id.lower()), "unknown"),
                "round1": {
                    "scores": r1.scores,
                    "probability": r1.p_iraki,
                    "confidence": r1.confidence,
                    "differential": r1.differential_diagnosis
                },
                "round3": {
                    "scores": r3.scores,
                    "probability": r3.p_iraki,
                    "confidence": r3.confidence,
                    "verdict": r3.verdict,
                    "diagnosis": r3.final_diagnosis,
                    "recommendations": r3.recommendations,
                    "changes": r3.changes_from_round1
                }
            })

        # compile debate transcripts
        debate_transcripts = []
        for expert_id, chat_log in self._chat_logs.items():
            debate_comments = [msg for msg in chat_log if msg["role"] == "debate_comment"]
            if debate_comments:
                debate_transcripts.append({
                    "expert_id": expert_id,
                    "comments": debate_comments
                })

        # generate reasoning summary
        reasoning_summary = self._generate_reasoning_summary()

        export_data = HumanReviewExport(
            case_id=self._case_id,
            final_consensus={
                "iraki_probability": final_prob,
                "iraki_verdict": final_verdict,
                "consensus_confidence": confidence,
                "weighting_method": "equal_weighting",
                "expert_count": len(self._round3_replies)
            },
            expert_assessments=expert_assessments,
            debate_transcripts=debate_transcripts,
            reasoning_summary=reasoning_summary,
            clinical_timeline=self._extract_clinical_timeline()
        )

        # save to file for human review
        export_filename = f"human_review_{self._case_id}_{int(time.time())}.json"
        with open(export_filename, 'w') as f:
            json.dump(export_data.model_dump(), f, indent=2)

        self.logger.info(f"Exported case for human review: {export_filename}")

    def _generate_reasoning_summary(self) -> str:
        """Generate executive summary of expert reasoning"""
        pro_iraki_reasons = []
        against_iraki_reasons = []

        for reply in self._round3_replies:
            if reply.verdict:  # pro-irAKI
                pro_iraki_reasons.append(f"{reply.expert_id}: {reply.final_diagnosis}")
            else:  # against irAKI
                against_iraki_reasons.append(f"{reply.expert_id}: {reply.final_diagnosis}")

        summary = f"""
EXPERT REASONING SUMMARY:

Pro-irAKI Arguments ({len(pro_iraki_reasons)} experts):
{chr(10).join('- ' + reason for reason in pro_iraki_reasons)}

Alternative Diagnosis Arguments ({len(against_iraki_reasons)} experts):
{chr(10).join('- ' + reason for reason in against_iraki_reasons)}

Key Debates: {len(self._active_debates)} question(s) required debate resolution
Consensus Method: Equal expert weighting (no ground truth learning)
"""
        return summary.strip()

    def _extract_clinical_timeline(self) -> Dict[str, Any]:
        """Extract key clinical timeline for human review"""
        # this would extract the immunotherapy timeline, AKI progression, etc.
        # simplified for this example
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
# CLI INTERFACE (modified for irAKI)
# =============================================================================

async def main():
    """Main entry point for irAKI classification system"""
    parser = argparse.ArgumentParser(description="DelPHEA-irAKI: Clinical Multi-Agent irAKI Classification System")

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

    # update global configuration for irAKI
    config.vllm_endpoint = args.endpoint
    config.model_name = args.model
    config.api_key = args.api_key
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.max_tokens = args.max_tokens
    config.expert_count = args.experts
    config.export_full_transcripts = args.export_transcripts
    config.include_reasoning_chains = args.detailed_reasoning

    try:
        if args.health_check:
            vllm_client = VLLMClient(config)
            healthy = await vllm_client.health_check()
            await vllm_client.close()

            if healthy:
                print("✓ DelPHEA-irAKI system healthy")
                print(f"✓ vLLM endpoint: {config.vllm_endpoint}")
                print(f"✓ Model: {config.model_name}")
                print("✓ Ready for irAKI classification")
                return 0
            else:
                print("✗ DelPHEA-irAKI system health check failed")
                return 1
        else:
            print(f"Starting DelPHEA-irAKI for case: {args.case_id}")
            print(f"vLLM endpoint: {config.vllm_endpoint}")
            print(f"Model: {config.model_name}")
            print(f"Experts: {config.expert_count}")
            print(f"Focus: immune-related AKI vs other AKI classification")
            print(f"Weighting: Equal (no ground truth learning)")
            print(f"Export for human review: {config.export_full_transcripts}")
            print("-" * 60)

            # create runtime and moderator
            runtime = SingleThreadedAgentRuntime()

            # register moderator
            await runtime.register("Moderator", lambda key: irAKIModeratorAgent(key))

            # start runtime
            await runtime.start()

            # bootstrap process
            await runtime.send_message(
                StartCase(case_id=args.case_id),
                AgentId("Moderator", args.case_id)
            )

            # wait for completion
            await runtime.stop_when_idle()
            await runtime.stop()

            print("\n✓ DelPHEA-irAKI completed successfully!")
            print("✓ Case exported for human expert chart review")
            return 0

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
