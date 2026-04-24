# Pitch Verdict
### An Agentic Pipeline for Verified Tactical Match Reports in Soccer

> **The core insight:** Every AI can write a match report. Only Pitch Verdict checks if it's true.

## What This Is

Pitch Verdict is a five-agent AI pipeline that transforms raw StatsBomb event data into verified tactical match reports. The word *verified* is doing the heavy lifting — the system generates claims and then fact-checks every single one against the source data before you see the output.

## Why It Exists (The Problem)

After every professional soccer match, an analyst spends 3–6 hours:
1. Pulling raw event data from StatsBomb (3,400+ events per match)
2. Computing metrics for each tactical phase
3. Writing a coherent report
4. Manually checking the numbers

That's **200+ hours per season** on retrieval and synthesis — work that requires no human judgment. The judgment is in interpretation. Pitch Verdict handles the mechanical work.

**Why not just use ChatGPT?** Ask it for a tactical report and you'll get one — confident, well-written, plausible. You'll also have no idea if any of it is true. The model generates text, not verified facts. Pitch Verdict's Verifier Agent closes this gap.

## The Five-Agent Pipeline

```
Retriever → Phase Segmenter → Tactical Classifier → Writer → Verifier
```

1. **Retriever** — Loads StatsBomb JSON, parses 3,400+ events into structured format
2. **Phase Segmenter** — Splits match into chapters (goals, subs, tactical shifts), computes per-phase metrics
3. **Tactical Classifier** — Translates metrics into tactical labels using PPDA, xG, pass patterns
4. **Writer** — Produces a 600–800 word professional match report
5. **Verifier** — Extracts every factual claim, checks against source data, flags mismatches

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/pitch-verdict
cd pitch-verdict
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Run the web app
streamlit run app.py

# 4. Or run the pipeline directly
python pipeline.py --match-id 3869685 --competition "Euro 2024"
```

## Project Structure

```
pitch_verdict/
├── app.py                  # Streamlit web interface
├── pipeline.py             # CLI entry point
├── agents/
│   ├── retriever.py        # Agent 1: Data loading & parsing
│   ├── phase_segmenter.py  # Agent 2: Match phase analysis
│   ├── tactical_classifier.py  # Agent 3: Tactical labeling
│   ├── writer.py           # Agent 4: Report generation
│   └── verifier.py         # Agent 5: Fact verification
├── evaluation/
│   ├── evaluator.py        # Automated evaluation framework
│   └── adversarial.py      # Adversarial error injection tests
├── data/
│   └── sample/             # Sample StatsBomb JSON (included)
├── outputs/                # Generated reports
├── tests/
│   └── test_pipeline.py    # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## Evaluation Metrics

| Metric | Method | Target |
|--------|--------|--------|
| Factual Accuracy | Automated claim extraction + data lookup | ≥ 95% |
| Tactical Label Accuracy | Human audit (Cohen's κ) | κ ≥ 0.6 |
| Faithfulness | LLM-as-Judge | ≥ 0.85 |
| Narrative Quality | Human Likert rating (1–5) | ≥ 3.5 |
| Verification Catch Rate | Adversarial error injection | ≥ 90% |

## Data Sources

- **StatsBomb Open Data** — Free, GitHub-hosted, industry-standard event data for 3,000+ matches
- **Coverage** — Euro 2024, Copa América, World Cups, select Premier League / La Liga seasons
- **Cost** — $0 for data, ~$0.50–$2.00 per report in LLM API costs

## Ethics & Attribution

- All data is publicly available match performance data (no private information)
- Every report credits StatsBomb per their user agreement
- If a metric cannot be computed from available data, the system says "data not available" — it does not fabricate

## Technical Stack

- **Language** — Python 3.10+
- **LLM** — Anthropic Claude Sonnet (configurable)
- **Data** — StatsBomb Open Data via `statsbombpy`
- **Visualization** — `mplsoccer`, `matplotlib`
- **UI** — Streamlit
- **Evaluation** — Custom + RAGAS-inspired faithfulness scoring

---

*Built for INFO 7375 — Generative AI, Northeastern University*
*Data: StatsBomb Open Data | xG: StatsBomb xG model*
