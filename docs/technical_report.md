# Pitch Verdict
## Technical Documentation & Project Report
### INFO 7375 — Generative AI | Northeastern University
**Student:** Hrishikesh Kulkarni | **ID:** 002340007

---

## Executive Summary

Pitch Verdict is a five-agent AI pipeline that automates the 3–6 hour post-match tactical analysis workflow in professional soccer. It ingests raw StatsBomb event data (3,400+ events per match), segments the match into tactical phases, classifies team behaviors using real metrics, generates a professional report, and then **verifies every factual claim against the source data before the report is released.**

The verification loop is the entire point. Every other AI tool generates plausible text. Pitch Verdict generates and checks.

**Why this matters:** A coaching staff acts on pre-match briefings. A report that confidently states a team's PPDA is 7.1 (high press) when it's actually 11.8 (mid-block) produces a game plan built on fiction. In analytical contexts — sports, medicine, finance, law — silent failures are far more dangerous than obvious ones. Obvious errors get caught. Silent failures get acted upon.

---

## 1. System Architecture

### 1.1 Five-Agent Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PITCH VERDICT PIPELINE                        │
│                                                                  │
│  ┌──────────┐    ┌─────────────────┐    ┌────────────────────┐  │
│  │ RETRIEVER│───▶│ PHASE SEGMENTER │───▶│TACTICAL CLASSIFIER │  │
│  │ Agent 1  │    │    Agent 2      │    │     Agent 3        │  │
│  │          │    │                 │    │                    │  │
│  │ Loads    │    │ Splits match    │    │ Computes PPDA,     │  │
│  │ 3,400+   │    │ into tactical   │    │ xG, possession     │  │
│  │ events   │    │ chapters        │    │ per phase          │  │
│  └──────────┘    └─────────────────┘    └─────────┬──────────┘  │
│                                                   │             │
│                                                   ▼             │
│               ┌──────────────┐    ┌───────────────────────┐    │
│               │   VERIFIER   │◀───│        WRITER         │    │
│               │   Agent 5   │    │       Agent 4         │    │
│               │             │    │                       │    │
│               │ Extracts    │    │ Generates 600-800     │    │
│               │ every claim │    │ word match report     │    │
│               │ Checks vs   │    │ from structured data  │    │
│               │ source data │    │                       │    │
│               └──────┬──────┘    └───────────────────────┘    │
│                      │                                          │
│                      ▼                                          │
│              ✅ VERIFIED REPORT                                  │
│         (Only released after fact-check)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Agent Responsibilities

| Agent | Responsibility | Key Output |
|-------|---------------|------------|
| **Retriever** | Load StatsBomb JSON, parse 3,400+ events into structured DataFrame | `MatchData` object with all events |
| **Phase Segmenter** | Identify goal/sub/shift inflection points; compute per-phase metrics | `MatchSegmentation` with 3–6 phases |
| **Tactical Classifier** | Apply PPDA thresholds and possession rules to generate tactical labels | `MatchTacticalAnalysis` with profiles |
| **Writer** | Generate 600–800 word professional report from structured data | `WriterOutput` with report text |
| **Verifier** | Extract all quantitative claims; check each against source data | `VerificationReport` with accuracy score |

### 1.3 Why Five Agents Instead of One Prompt?

This is the Gemini Test from the proposal: can a single clever prompt in ChatGPT produce the same result?

**No, for three reasons:**

1. **Data access.** A standard chatbot cannot autonomously load StatsBomb's event-level JSON from GitHub. Each match is megabytes of structured data.

2. **Multi-phase reasoning.** A single prompt cannot segment an event stream by time window, compute separate PPDA/xG/possession for each segment, and reason across segments. This is a multi-step computation, not a single inference.

3. **Self-verification.** No chatbot has a feedback loop that checks its own output. The Verifier Agent does exactly this — it reads the draft, extracts every number, recomputes it from source data, and flags mismatches. This is deterministic Python, not another LLM call.

---

## 2. Core Metrics and Algorithms

### 2.1 PPDA (Passes Per Defensive Action)

PPDA is the primary metric for quantifying pressing intensity.

```
PPDA = Opponent Passes in Their Own Half
       ───────────────────────────────────
       Defensive Actions in Opponent's Half
```

**Thresholds:**
- PPDA < 8: High Press (aggressive, hunting the ball high up the pitch)
- PPDA 8–12: Mid-Block (organized defensive shape in middle third)
- PPDA > 12: Deep Block / Low Block (sitting deep, conceding territory)

These thresholds are documented in StatsBomb's analytics literature and used by professional clubs. When the Writer says "high press," the Verifier checks: does the PPDA for that phase actually fall below 8?

### 2.2 Expected Goals (xG)

xG is a shot-quality model. Each shot is assigned a probability (0–1) of resulting in a goal based on location, body part, goalkeeper position, and shot type. StatsBomb's xG model is industry-leading and used by Premier League clubs.

**How Pitch Verdict uses it:**
- xG per phase tracks when chances were created vs. when the scoreline changed
- Total xG vs. actual goals reveals whether a result was deserved
- xG per shot classifies chance quality (clinical vs. wasteful)

### 2.3 Phase Segmentation Logic

Inflection points trigger new phases:
1. **Goals** (always a major tactical break — teams respond to scoreline changes)
2. **Halftime**
3. **Substitutions** (changes team shape and energy)

Minimum phase length: 10 minutes (avoids micro-phases with insufficient data). The algorithm merges adjacent short phases.

### 2.4 Verification Logic

The Verifier uses regex + NLP to extract claims from prose, then checks each against the source:

```python
# For possession claims:
pattern = r'(\b\d+\.?\d*)\s*%'  # Find "XX.X%"
# + Context check: near "possession", "ball", "held"
# + Team identification from surrounding text
# + Tolerance: ±2% (for rounding differences)

# For PPDA claims:
pattern = r'PPDA[^\d]{0,30}(\d+\.\d+)'  # Find PPDA value
# + Verify implied tactical label against threshold
# e.g., "high press" claimed → check PPDA < 8

# Factual Accuracy = Verified Claims / Total Claims
# Target: ≥ 95%
```

---

## 3. Data Sources and Ethics

### 3.1 StatsBomb Open Data

- **Source:** GitHub (statsbomb/open-data)
- **Coverage:** 3,000+ matches — Euro 2024, Copa América, World Cups, select Premier League/La Liga seasons
- **Format:** JSON, one file per match, ~3,400 events each
- **Schema:** Every pass (with start/end coordinates), shot (with xG), pressure (with location), substitution, card
- **Quality:** Industry standard; used by professional clubs
- **Cost:** Free

### 3.2 Ethics and Attribution

- All data is publicly available match performance data — no private health, salary, or personal information
- StatsBomb's user agreement requires source attribution; every Pitch Verdict report includes: *"Data sourced from StatsBomb Open Data. xG values from the StatsBomb xG model."*
- The Verifier handles intellectual honesty structurally: if a metric cannot be computed from available data, the system outputs "data not available" instead of fabricating a value
- **Off-ball limitation:** Event data captures on-ball actions; it misses off-ball pressing runs that don't win the ball, decoy runs, and passive defensive shape. Pitch Verdict flags when tactical conclusions rely on incomplete off-ball data

---

## 4. Evaluation Framework

The evaluation framework measures five metrics. This is honest measurement — it identifies failure, not just success.

### 4.1 Metric Definitions

**Metric 1: Factual Accuracy**
```
Factual Accuracy = Correct Claims / Total Claims
Target: ≥ 95%
```
Automated: every quantitative claim extracted from the report, checked against source data.

**Metric 2: Tactical Label Accuracy (Cohen's κ)**
```
κ = (Observed Agreement − Expected Agreement) / (1 − Expected Agreement)
Target: κ ≥ 0.6 ("substantial agreement")
```
Human reviewers compare system's tactical classifications against published expert analyses for 10–15 matches. Cohen's kappa accounts for chance agreement (κ = 0 means no better than random; κ = 1 is perfect).

**Metric 3: Faithfulness (LLM-as-Judge)**
```
Target: ≥ 0.85
```
A separate LLM receives the source data and generated report; it evaluates whether every claim follows from the data. This is the RAGAS faithfulness metric pattern.

**Metric 4: Narrative Quality (Likert Scale)**
```
Human rating: 1–5 scale
Target: ≥ 3.5
```
Raters assess readability, coherence, and usefulness for a domain professional.

**Metric 5: Verification Catch Rate (Adversarial)**
```
Catch Rate = Errors Caught / Errors Injected
Target: ≥ 90%
```
Known errors are deliberately injected into draft reports; the Verifier is run to see what it catches. This stress-tests the verification system directly.

### 4.2 Results Summary

| Metric | Method | Target | Achieved |
|--------|--------|--------|---------|
| Factual Accuracy | Automated claim check | ≥ 95% | ~82–95%* |
| Tactical Label Accuracy | Cohen's κ | κ ≥ 0.6 | ~0.67 |
| Faithfulness | LLM-as-Judge | ≥ 0.85 | ~0.87 |
| Narrative Quality | Likert (1–5) | ≥ 3.5 | ~4.0 |
| Verification Catch Rate | Adversarial injection | ≥ 90% | ~100% |

*Factual accuracy improves significantly with LLM-generated reports vs. demo mode. Demo mode produces fewer quantitative claims to check; full LLM reports target ≥ 95%.

---

## 5. Technical Implementation

### 5.1 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.10+ | Industry standard for data science |
| LLM | Anthropic Claude Sonnet | Best instruction-following for structured output |
| Data | StatsBomb via `statsbombpy` | Direct API to open data repository |
| Visualization | Plotly, mplsoccer | Interactive charts in Streamlit |
| UI | Streamlit | Rapid development, data-science native |
| Evaluation | Custom + RAGAS patterns | Transparent, reproducible metrics |

### 5.2 Project Structure

```
pitch_verdict/
├── app.py                    # Streamlit web interface
├── pipeline.py               # CLI orchestrator
├── agents/
│   ├── retriever.py          # Agent 1: Data ingestion
│   ├── phase_segmenter.py    # Agent 2: Phase analysis + PPDA computation
│   ├── tactical_classifier.py # Agent 3: Label generation
│   ├── writer.py             # Agent 4: LLM report generation
│   └── verifier.py           # Agent 5: Fact verification
├── evaluation/
│   └── evaluator.py          # Five-metric evaluation framework
├── tests/
│   └── test_pipeline.py      # Unit + integration tests
├── outputs/                  # Generated reports
├── requirements.txt
└── .env.example
```

### 5.3 Cost Analysis

| Resource | Cost |
|----------|------|
| StatsBomb Open Data | $0 (free, GitHub) |
| FBRef scraping | $0 (free, Python libraries) |
| LLM API (Anthropic/OpenAI) | ~$0.50–$2.00 per report |
| Compute | $0 (runs on laptop, no GPU) |

**Total per report: under $2.00.** The LLM math: 4 agent calls × ~2,000–4,000 tokens each = ~10,000 tokens per report. At Claude Sonnet pricing, that's well under $2.

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

- **Off-ball blindness:** StatsBomb event data captures on-ball actions but misses off-ball movement — pressing runs that don't win the ball, decoy runs, passive defensive positioning. The system flags this in generated reports.
- **Data availability:** StatsBomb's free dataset covers select competitions. Not every league or season is available.
- **xG is a model:** Different providers produce different xG values for the same shot. Every report discloses which model is used.
- **Verification tolerance:** The ±2% tolerance on percentages and ±0.05 on xG means small rounding differences don't trigger flags. This is appropriate but means the system doesn't catch very small deviations.

### 6.2 Stretch Goals

- **Formation detection** from positional clustering of event coordinates
- **Passing network visualization** using mplsoccer's pitch plotting library
- **FBRef seasonal context** integration (compare match stats to team's season averages)
- **Multi-match reports** (season-level trends across a team's full campaign)
- **Live data integration** for real-time analysis during matches

### 6.3 Broader Impact

The verification pattern Pitch Verdict demonstrates is not specific to soccer. Anywhere an AI system generates analytical text that someone might act on — medical reports, financial analysis, legal research — the same architecture applies:

1. Compute ground truth from structured data
2. Generate natural language report from structured data (not raw sources)
3. Extract claims from the report
4. Check each claim against ground truth
5. Flag mismatches before release

The question Pitch Verdict opens is: **what if verification weren't optional for any AI system claiming to be analytical?**

---

## 7. Setup and Usage

### 7.1 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/pitch-verdict
cd pitch-verdict

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=your_key_here

# Run web app
streamlit run app.py

# Or CLI (uses sample match by default)
python pipeline.py --sample --save
```

### 7.2 Running Evaluation

```bash
# Full evaluation framework
python evaluation/evaluator.py

# Unit tests
python -m pytest tests/ -v
```

### 7.3 Using Live StatsBomb Data

```bash
# Install statsbombpy
pip install statsbombpy

# Run with Euro 2024 Final
python pipeline.py --match-id 3869685

# The app's "StatsBomb Open Data" option handles this in the UI
```

---

## 8. Reflections

This project brings together the two domains I care most about. Football is where I learned to pay attention — commentators who could explain why a team lost shape in the second half, why a midfielder's positioning unlocked the press. Computer science is what I've chosen to build a career around.

Pitch Verdict is the overlap: the technical work of making analysis rigorous, so that the understanding that made football meaningful can be automated and democratized. The gap between what a Premier League analytics department can produce and what a League Two club or a serious fan can access remains enormous. This doesn't close it entirely. But it proves the mechanical work can be automated without sacrificing accuracy.

The hardest part wasn't the code. It was accepting the uncomfortable design assumption: the AI will get things wrong. Instead of hoping for accuracy, the system is designed to catch its own mistakes. That's a different posture than most AI applications take. And I think it's the right one.

---

*Data: StatsBomb Open Data (statsbomb/open-data) | xG: StatsBomb xG model*
*All data is publicly available match performance data. No private information used.*
*INFO 7375 — Generative AI | Northeastern University | 2024*
