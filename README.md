# DelPHEA‑irAKI (Immune‑related AKI)

DelPHEA simulates a small, diverse panel of virtual experts and runs a **modified Delphi** process over a fixed questionnaire to decide whether an AKI episode in an ICI‑treated patient is immune‑related. It ships with a **modular, testable pipeline** (Round‑1 → Debate → Round‑3 → Aggregate) and strict validation/repair logic.

> If you only read one thing, read the big ASCII below — it’s the whole procedure with every knob and exit condition annotated.

---

## What’s new (short)

* **Modular rounds** with a single `Moderator` orchestrating experts.
* **Deterministic debate loop** (minority → majority → handoff → minority follow‑up).
* **Validators + one‑shot retry with hint** (e.g., *importance must sum to 100*).
* **Autopatch fallback** (explicit placeholders) if retry still fails.
* **Exact QID echo** enforcement (scores/evidence keys == questionnaire QIDs).
* Simple **live backend smoke** path (OpenAI‑compatible / vLLM).

---

## Core contracts (data in/out)

**Round 1** (`AssessmentR1`)

* `scores: dict[qid, int]`
* `evidence: dict[qid, str]`
* `clinical_reasoning: str` (≥ 200 chars)
* `primary_diagnosis: str`
* `differential_diagnosis: list[str]` (≥ 2)
* `rationale / q_confidence / importance` **per‑QID**

  * `importance: dict[str, int]` and **sums to exactly 100**
* `p_iraki: float`, `ci_iraki: tuple[float, float]`, `confidence: float`

**Debate turn** (`DebateTurn`)

* `text: str`, `satisfied: bool`, optional `handoff_to: str | None`

**Round 3** (`AssessmentR3`)

* Same per‑QID fields + `changes_from_round1`, `final_diagnosis`, `recommendations`
* `verdict: bool`, `confidence_in_verdict: float`

**Consensus** (aggregator output)

* `iraki_probability: float`, `ci_iraki: tuple[float, float]`, `iraki_verdict: bool`, etc.

---

## The whole flow (annotated ASCII)

```
                                  ┌────────────────────────────────────────────────┐
                                  │                    Moderator                   │
                                  │  inputs: case, experts[], questionnaire, rules │
                                  └───────────────┬────────────────────────────────┘
                                                  │
                           (A) Round 1 fan-out    │  assess_round(1, case)
                                                  ▼
       ┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐
       │       Expert E1      │  ...    │       Expert Ek      │  ...    │       Expert EN      │
       │ assess_round1(case)  │         │ assess_round1(case)  │         │ assess_round1(case)  │
       └──────────┬───────────┘         └──────────┬───────────┘         └──────────┬───────────┘
                  │                                │                                │
                  │  (A1) Validate each payload:   │                                │
                  │  - exact QID echo (scores/evidence keys == questionnaire QIDs)  │
                  │  - schema + business rules (≥200 chars reasoning, etc.)         │
                  │  - importance is INT and sums to exactly 100                    │
                  │                                                                 │
                  └─────────────────────────────────────────────────────────────────┘
                                                  │
                                                  │ if any ValidationError:
                                                  │     → one-shot retry with repair_hint
                                                  │     → else autopatch & continue
                                                  ▼
                              (B) Router: detect QIDs needing debate
                               router.plan(r1, rules) → DebatePlan.by_qid = {qid: [minority_ids]}
                                                  │
                                                  ├───────────── if no_disagreement → debate_skipped=True
                                                  │
                                                  ▼
                                   (C) Debate orchestration per QID
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ INITIAL QUEUE (per QID):                                                                                             │
   │   1) minority_open from each minority expert (order: as listed by router)                                            │
   │   2) majority_rebuttal from all other experts                                                                        │
   │   3) minority_followup from the first minority (the “opener”)                                                        │
   │                                                                                                                      │
   │ LOOP (bounded): while queue not empty and turns < max_total_turns_per_qid                                           │
   │   • Pop next (role, eid). Skip if expert already satisfied or capped by max_turns_per_expert.                        │
   │   • Build clinical_context = { case, peer_turns=last max_history_turns, role }.                                      │
   │   • Call e.debate(qid, round_no=2, clinical_context, minority_view=text).                                            │
   │   • Record turn: {expert_id, qid, turn_index, speaker_role=minority/majority/participant}.                           │
   │   • If turn.satisfied==True → mark expert satisfied.                                                                 │
   │   • HONOR HANDOFF: if turn.handoff_to == valid target and target not satisfied and not capped:                       │
   │        - If target is already queued: pop it;                                                                        │
   │             * if current role is 'minority_open' AND target was next: keep its planned role;                         │
   │             * else reinsert at front as ('participant', target).                                                     │
   │        - If target not queued: insert at front as ('participant', target).                                           │
   │   • After any majority or participant, bubble opener’s 'minority_followup' to run next                               │
   │     (after any just-inserted handoff).                                                                               │
   │   • EXIT CONDITIONS (any):                                                                                           │
   │        - All minority+majority experts satisfied OR capped                                                           │
   │        - Queue exhausted                                                                                              │
   │        - Reached max_total_turns_per_qid                                                                             │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                  (D) Round 3 fan-out  assess_round(3, case, debate_ctx)
                                                  │
                                same validate → retry → autopatch discipline as Round 1
                                                  │
                                                  ▼
                                       (E) Aggregate to Consensus
                                       aggregator.aggregate(r3[]) → Consensus
                                                  │
                                                  ▼
                                     (F) Return full report (dict)
                                     {case_id, round1[], debate, round3[], consensus}
```

**Default knobs (tunable in `Moderator`):**

* `max_retries = 1` *(per expert, per round)*
* `debate_rounds = 3` *(kept for parity; actual loop bounded by turn caps)*
* `max_history_turns = 6` *(peer turns passed back to experts)*
* `max_turns_per_expert = 2`
* `max_total_turns_per_qid = 12`
* `quiet_turn_limit = 3` *(placeholder knob; quiet-turn detection TBD)*

---

## Debate specifics (who speaks, handoff, exit)

* **Kick‑start:** Every QID chosen by the router asks **all minority experts** to open.
* **Who speaks next:** All **majority rebuttals**, then bubble the **opener’s** minority follow‑up.
* **Handoff semantics:**

  * If a speaker sets `handoff_to = X` and X is eligible (not satisfied/capped):

    * If X is already queued **as next** and current role is `minority_open`, keep X’s planned role.
    * Otherwise, put X **at the front** as `('participant', X)`.
* **Exit conditions (any):**

  * Everyone (minority + majority sets) is **satisfied** or **capped**.
  * **Queue exhausted**.
  * **Max total turns per QID** reached.
* **Context to experts on each turn:**

  * `clinical_context = { case, peer_turns=[last N turns], role }`
  * `minority_view` is a compact string constructed from minority R1 scores+evidence.

---

## Validation, retry, autopatch

1. **Primary validation** (`validators.py`)

   * Schema + business rules for R1/R3.
   * **Importance** is `dict[str,int]` and **sums to 100**.
   * `ci_iraki` is a **tuple** `(low, high)`.
   * Round‑specific checks: R1 (reasoning length, PDx/DDx); R3 (changes/final dx/recs, verdict fields).

2. **On failure** the `Moderator` **retries once with a hint**

   * `_build_repair_hint(err, round_no)` composes a concise, model‑friendly message from `err.errors()`.
   * If message contains *“importance must sum to 100 (got X)”*, we append an explicit directive:

     * *“Ensure per‑QID 'importance' are integers that sum to exactly 100 (no rounding; adjust values, then regenerate).”*
   * The retry passes `repair_hint` **iff** the expert method signature accepts it (duck‑typed via `inspect.signature`).

3. **If retry still fails**, the `Moderator` performs a **minimal autopatch** so the pipeline can continue

   * Fill missing evidence with a visible placeholder.
   * Synthesize ≥200‑char `clinical_reasoning` if empty.
   * Fill missing PDx/DDx (R1) and R3 `changes_from_round1`/`recommendations` with explicit placeholders.
   * **Note:** Autopatch does **not** invent an “importance=100” distribution; correctness is expected to come from the retry.

4. **QID discipline** (after each R1/R3):

   * `set(scores.keys()) == set(evidence.keys()) == set(questionnaire_qids)` → else error.

---

## Quick usage

### Live OpenAI‑compatible backend (vLLM, etc.)

Set one of:

```bash
export ENDPOINT_URL="http://host:8000"
# or
export OPENAI_BASE_URL="http://host:8000"
# or
export LIVE_BACKEND_URL="http://host:8000"
```

Optional model name:

```bash
export MODEL_NAME="openai/gpt-oss-120b"
```

### Run tests

* Smoke (debate only, live backend):

```bash
pytest tests/test_round2_live_moderator_smoke.py -q
```

* Modular scripted rounds:

```bash
pytest tests/test_rounds_modular.py -q
```

* Importance retry behavior (sum must be 100):

```bash
pytest tests/test_importance_retry.py -q
```

---

## Configuration knobs (where they matter)

* **Moderator(...)**

  * `max_retries=1` *(per expert, per round)*
  * `debate_rounds=3` *(present; loop bounded by turn caps)*
  * `max_history_turns=6`, `max_turns_per_expert=2`, `max_total_turns_per_qid=12`
* **Router**: implements `plan(r1, rules) → DebatePlan(by_qid: dict[qid, list[minority_ids]])`.
* **Validators**: pydantic/business checks + helpful messages.
* **Aggregator**: reduces R3 payloads to a single `Consensus`.

---

## Developer notes

* `_build_repair_hint` is **duck‑typed**: any object with `.errors() -> list[{loc,msg}]` works (enables unit tests to inject fakes).
* `repair_hint` is only passed if the callee supports it (checked via `inspect.signature`).
* `ci_iraki` is a **tuple** throughout, not a list.
* `importance` is a **dict\[str,int]** that must sum to **exactly 100** (under‑ or over‑sum triggers retry; covered by tests).
* Debate honors directed `handoff_to` with clear precedence rules (keep planned role only for minority→next; otherwise insert as participant at front).

---

## Roadmap / open questions

* Quiet‑turn detection (use `quiet_turn_limit`) to end low‑value debates early.
* Structured handoff reasons and auto‑re‑routing when majority splinters.
* Richer aggregation (uncertainty propagation, expert weighting).
* Pluggable *explainers* to convert transcripts into clinician‑friendly synopses.
