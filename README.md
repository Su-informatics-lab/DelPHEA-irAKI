# DelPHEA-irAKI

**Del**phi **P**ersonalized **H**ealth **E**xplainable **A**gents for **i**mmune-**r**elated **AKI** Classification: 
A agent-simulated Delphi clinical decision support system for distinguishing immune-related AKI from other AKI etiologies 
using clinical notes and structured data.

## Overview

DelPHEA-irAKI implements a novel approach to immune-related AKI classification by simulating a virtual panel of medical 
experts conducting a modified Delphi consensus process. The system combines multiple LLM agents, each representing 
different medical specialties relevant to immune-mediated kidney injury, to collaboratively assess whether AKI cases 
are immune-related through iterative discussion and evidence-based reasoning.

## Key Features

### 🏥 **Modular Expert Panel Configuration**
- **JSON-based expert definitions** with detailed clinical personas
- **Configurable specialties**: Oncology, Nephrology, Pathology, Pharmacy, Informatics, Nursing, Critical Care, Rheumatology, Hospital Medicine, Emergency Medicine, Geriatric etc
- **Specialty-specific expertise** and reasoning styles
- **Flexible panel composition** for different clinical contexts

### 📚 **Literature Integration (Optional)**
- **PubMed API integration** for evidence-based reasoning
- **bioRxiv preprint search** for cutting-edge research
- **Specialty-specific literature queries** tailored to expert focus areas
- **Citation integration** in clinical reasoning and debates
- **Relevance scoring** and key sentence extraction

### 🔬 **Clinical Assessment Framework**
- **12 irAKI-specific questions** based on published literature (Sprangers et al. Nature Reviews Nephrology 2022)
- **9-point Likert scale** assessment with detailed clinical context
- **Configurable questionnaire** via JSON for easy updates
- **Evidence-based question design** with clinical decision support integration

### 🤖 **Multi-Agent Architecture**
- **vLLM Backend Integration**: High-performance inference with AWS H100/A100 deployment
- **3-Round Delphi Process**: Individual assessment → Conflict resolution → Final consensus
- **Equal Expert Weighting**: Transparent consensus without ground truth dependency
- **Automated Debate Facilitation**: Structured discussions for complex disagreements

### 📊 **Human Expert Validation**
- **Complete transcript export** for real physician validation
- **Detailed reasoning chains** from each expert specialty
- **Clinical timeline analysis** and evidence synthesis
- **Literature citation tracking** (when enabled)

## System Architecture

### Modular Components

```
DelPHEA-irAKI/
├── delphea.py                 # main
├── config/
│   ├── panel.json             # expert panel configuration
│   └── questionnaire.json     # clinical assessment questions
├── search.py                  # PubMed/bioRxiv integration
├── output/                    # results and transcripts
├── demo.py                    # examples
└── requirements.txt           # dependencies
```

### Expert Panel Configuration

The system uses JSON configuration files to define expert panels:

```json
{
  "expert_panel": {
    "experts": [
      {
        "id": "oncologist_001",
        "specialty": "medical_oncology",
        "name": "Dr. Sarah Chen",
        "experience_years": 15,
        "expertise": ["immune checkpoint inhibitor toxicities", "irAEs"],
        "focus_areas": ["temporal relationship analysis", "irAE correlation"],
        "reasoning_style": "evidence-based, protocol-driven"
      }
    ]
  }
}
```

### Clinical Questionnaire

The assessment questions are also configurable via JSON:

```json
{
  "questionnaire": {
    "questions": [
      {
        "id": "Q1", 
        "category": "temporal_relationship",
        "question": "The temporal relationship between ICI exposure and AKI onset strongly supports irAKI diagnosis",
        "clinical_context": {
          "supportive_evidence": ["AKI onset 2-5 months after ICI initiation"],
          "contradictory_evidence": ["AKI onset before ICI therapy"]
        }
      }
    ]
  }
}
```

## Quick Start

### 1. **Installation**
```bash
# clone repository and install dependencies
pip install -r requirements.txt

# ensure AWS vLLM infrastructure is configured
export VLLM_ENDPOINT="http://172.31.11.192:8000"
```

### 2. **Basic Usage**
```bash
# health check
python delphea.py --health-check

# basic irAKI classification
python delphea.py --case-id iraki_case_001 --verbose

# with literature search
python delphea.py --case-id iraki_case_001 --enable-literature-search --verbose
```

### 3. **Example Runner**
```bash
# run basic example
python demo.py --example basic

# run with literature search
python demo.py --example with-literature

# run with custom expert panel
python demo.py --example custom-panel

# check prerequisites
python demo.py --check-prereqs
```

### 4. **Custom Configuration**
```bash
# use custom expert panel and questionnaire
python delphea.py \
  --expert-panel-config my_experts.json \
  --questionnaire-config my_questions.json \
  --case-id my_case_001
```

## Clinical Assessment Questions

The irAKI classification system uses 12 evidence-based questions:

1. **Temporal Relationship**: ICI exposure timing vs. AKI onset
2. **Prerenal Exclusion**: Volume status, hypotension, medication effects  
3. **Postrenal Exclusion**: Obstruction, structural abnormalities
4. **Other Intrinsic Exclusion**: ATN, contrast nephropathy, other drugs
5. **Urinalysis Pattern**: Proteinuria, hematuria, cellular casts
6. **Immune Activation**: Systemic inflammatory markers
7. **Concomitant Medications**: PPI, NSAID, antibiotic assessment
8. **Other irAE Correlation**: Multi-organ immune manifestations
9. **Biopsy Indication**: Expected immune-mediated pathology
10. **Treatment Response**: Clinical course and steroid response
11. **Alternative Diagnosis Exclusion**: Systematic differential assessment  
12. **Clinical Gestalt**: Overall irAKI likelihood

## Expert Panel Specialties

### Primary Experts
- **Medical Oncology**: ICI treatment expertise, irAE management
- **Nephrology**: AKI differential diagnosis, kidney function assessment
- **Renal Pathology**: Histologic pattern recognition, biopsy interpretation

### Supporting Experts  
- **Clinical Pharmacy**: Drug interaction analysis, medication safety
- **Clinical Informatics**: Data pattern recognition, evidence synthesis
- **Oncology Nursing**: Patient monitoring, symptom assessment
- **Critical Care**: ICU-based AKI management, acute care

### Consulting Experts
- **Rheumatology**: Autoimmune mechanisms, immune complex disease
- **Hospital Medicine**: Comprehensive assessment, care coordination
- **Emergency Medicine**: Acute presentation patterns, rapid assessment

## Literature Search Integration

When enabled (`--enable-literature-search`), the system:

1. **Generates specialty-specific queries** for each expert
2. **Searches PubMed** for peer-reviewed literature (last 5 years)
3. **Searches bioRxiv** for recent preprints (last 2 years)  
4. **Ranks results by relevance** using irAKI-specific keywords
5. **Extracts key sentences** for clinical reasoning
6. **Integrates citations** into expert assessments

### Literature Search Example

```python
# Expert: Nephrology
Query: "immune checkpoint inhibitor acute kidney injury drug induced nephritis AKI"

# Expert: Oncology  
Query: "immune checkpoint inhibitor acute kidney injury immunotherapy toxicity irAE"

# Results integrated into clinical reasoning with citations
```

## Output for Human Review

### 1. **Consensus Results**
```json
{
  "final_consensus": {
    "iraki_probability": 0.73,
    "iraki_verdict": true,
    "consensus_confidence": 0.81,
    "expert_count": 8
  }
}
```

### 2. **Expert Assessments**
- Individual specialty perspectives with detailed reasoning
- Round 1 vs. Round 3 changes and explanations
- Literature citations (when enabled)
- Specialty-specific differential diagnoses

### 3. **Debate Transcripts**
- Question-specific disagreements and resolutions
- Expert arguments with supporting evidence
- Literature citations in debates
- Satisfaction indicators and consensus building

### 4. **Clinical Timeline**
- Immunotherapy exposure history
- AKI progression timeline
- Key clinical events and temporal relationships

## Configuration Options

### Expert Panel Customization

```bash
# full expert panel (10 experts)
--expert-panel-config config/expert_panel.json

# custom focused panel (4 experts)  
--expert-panel-config config/expert_panel_custom.json
```

### Literature Search Configuration

```bash
# enable with default settings
--enable-literature-search

# custom literature settings
--enable-literature-search \
--max-literature-results 3 \
--literature-recent-years 3 \
--literature-email "your@email.com"
```

### Assessment Configuration

```bash
# standard 12-question assessment
--questionnaire-config config/questionnaire_iraki.json

# custom question set
--questionnaire-config config/questionnaire_custom.json
```

## Development and Customization

### Adding New Expert Specialties

1. Update `config/expert_panel.json` with new expert definitions
2. Add specialty-specific keywords to `modules/literature_search.py` 
3. Update questionnaire focus areas if needed

### Modifying Assessment Questions

1. Edit `config/questionnaire_iraki.json`
2. Add clinical context and evidence criteria
3. Update scoring guidelines and decision support thresholds

### Extending Literature Search

1. Add new databases/APIs to `modules/literature_search.py`
2. Implement custom relevance scoring algorithms
3. Add specialty-specific query enhancement rules

## Clinical Research Applications

### 1. **Decision Support Development**
- Prototype for irAKI clinical decision tools
- Framework for other immune-related adverse events
- Educational tool for immunotherapy toxicity training

### 2. **Knowledge Discovery**  
- Analysis of expert reasoning patterns
- Identification of clinical decision-making processes
- Insights into specialty-specific perspectives on irAKI

### 3. **Quality Improvement**
- Standardization of irAKI assessment approaches
- Consensus development for challenging cases
- Training material for medical education

## Validation Approach

Since ground truth outcomes are not available for irAKI classification:

### 1. **Human Expert Chart Review**
- Real physicians review AI consensus decisions
- Assessment of clinical reasoning quality  
- Identification of missed considerations
- Comparison with human expert judgment

### 2. **Process Validation**
- Consistency of expert reasoning across cases
- Appropriate use of clinical evidence
- Specialty-specific contribution analysis
- Debate quality and resolution effectiveness

### 3. **Clinical Utility Assessment**
- Decision support value for clinicians
- Educational benefit for training programs
- Identification of knowledge gaps in irAKI assessment

## Future Directions

### 1. **Ground Truth Integration**
- Biopsy result correlation (when available)
- Response to immunosuppressive therapy outcomes
- Long-term renal function follow-up

### 2. **Expanded Clinical Scope**
- Other immune-related adverse events (hepatitis, pneumonitis)
- Multi-organ irAE assessment and correlation
- Treatment response prediction models

### 3. **Clinical Integration**
- Electronic health record integration
- Real-time clinical decision support
- Automated case identification and triage

### 4. **Enhanced Literature Integration**
- Real-time literature monitoring
- Guideline integration and updates
- Clinical trial result incorporation

## Contributing

This system is designed for clinical research in immune-related adverse events. 

### Research Collaboration 
- **Code/Technical**: Haining Wang (hw56@iu.edu)
- **General Questions**: Jing Su (su1@iu.edu)

## License

MIT for code. Clinical data requires appropriate institutional permissions.
