"""
Agent 4: Writer
Uses an LLM to generate a professional tactical match report from the
structured analysis produced by Agents 1–3.

The Writer is given ONLY verified, computed data — it cannot hallucinate
numbers it hasn't been provided. But it CAN misquote or rephrase them
incorrectly, which is exactly why Agent 5 (Verifier) exists.
"""

import os
import json
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class WriterOutput:
    """Output from the Writer Agent."""
    report_text: str
    match_info: dict
    structured_data: dict  # The data used to generate the report
    model_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


WRITER_SYSTEM_PROMPT = """You are a professional football tactical analyst and writer. 
Your reports appear in publications like The Athletic and StatsBomb's own analysis blog.

Your writing is:
- Precise and specific (exact numbers, not vague claims)
- Analytically grounded (every tactical label explained by the metric behind it)
- Narrative-driven (the report tells the story of how the match unfolded)
- Honest about uncertainty (flag when data limitations apply)

CRITICAL RULES:
1. Every statistic you write MUST come from the data provided to you. Do not invent numbers.
2. Every tactical label (e.g., "high press") MUST be justified by the metric provided.
3. If a PPDA value is provided, cite it. If you say "high press," the PPDA must support that claim.
4. Write EXACTLY the numbers as given — do not round differently or approximate.
5. Structure: Opening summary → Phase-by-phase → Key tactical themes → Conclusion
6. Length: 600–800 words
7. End with a short, impactful sentence.

You will be given structured JSON data. Use it faithfully."""


WRITER_USER_PROMPT_TEMPLATE = """Write a professional tactical match report for this match.

MATCH DATA:
{match_json}

REQUIRED STRUCTURE:
1. **Opening** (2-3 sentences): Result, overall story of the match
2. **Phase-by-Phase Breakdown** (one paragraph per phase): For each phase, cite the actual metrics (possession %, PPDA, xG) and explain what they mean tactically
3. **Key Tactical Themes** (2-3 paragraphs): The 2-3 most important tactical storylines across the full match
4. **Conclusion** (1-2 sentences): Was the result fair? What does this match tell us?

RULES:
- When you mention possession, use the exact percentage provided
- When you mention pressing, cite the PPDA value and what it means (below 8 = high press, 8-12 = mid-block, above 12 = deep block)
- When you mention xG, use the exact value provided
- Do NOT introduce any statistics not present in the data
- Use "Data not available" if asked about something not in the data

Write the full report now:"""


class WriterAgent:
    """
    Agent 4: Writer

    Responsibility: Transform structured tactical analysis into a
    professional, readable 600-800 word match report.

    The Writer receives only computed data — it cannot hallucinate raw
    statistics. However, it might misquote or misrepresent them, which
    is why every claim it makes is subsequently verified by Agent 5.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.client = None
        self.provider = None
        self._setup_client()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Writer] {msg}")

    def _setup_client(self):
        """Initialize LLM client (xAI Grok preferred, Anthropic/OpenAI fallback)."""
        xai_key = os.getenv('XAI_API_KEY', '')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        openai_key = os.getenv('OPENAI_API_KEY', '')

        if xai_key and xai_key.startswith('gsk_'):
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=xai_key,
                    base_url="https://api.x.ai/v1",
                )
                self.provider = 'xai'
                self._log("Using xAI Grok")
                return
            except ImportError:
                self._log("openai package not installed (needed for xAI)")

        if anthropic_key and anthropic_key != 'your_anthropic_key_here':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=anthropic_key)
                self.provider = 'anthropic'
                self._log("Using Anthropic Claude")
                return
            except ImportError:
                self._log("anthropic package not installed")

        if openai_key and openai_key != 'your_openai_key_here':
            try:
                import openai
                self.client = openai.OpenAI(api_key=openai_key)
                self.provider = 'openai'
                self._log("Using OpenAI GPT")
                return
            except ImportError:
                self._log("openai package not installed")

        self._log("No LLM client configured — will use demo mode")
        self.provider = 'demo'

    def write_report(
        self,
        structured_data: dict,
        match_meta: dict,
    ) -> WriterOutput:
        """Generate a match report from structured analysis data."""
        self._log(f"Writing report for {match_meta.get('home_team')} vs {match_meta.get('away_team')}")

        # Build the prompt data (sanitized — only computed values)
        prompt_data = self._build_prompt_data(structured_data, match_meta)

        if self.provider == 'demo':
            report = self._generate_demo_report(structured_data, match_meta)
            return WriterOutput(
                report_text=report,
                match_info=match_meta,
                structured_data=structured_data,
                model_used='demo-mode',
            )

        prompt_json = json.dumps(prompt_data, indent=2)
        user_prompt = WRITER_USER_PROMPT_TEMPLATE.format(match_json=prompt_json)

        if self.provider == 'xai':
            return self._call_xai(user_prompt, structured_data, match_meta)
        elif self.provider == 'anthropic':
            return self._call_anthropic(user_prompt, structured_data, match_meta)
        else:
            return self._call_openai(user_prompt, structured_data, match_meta)

    def _call_xai(self, user_prompt: str, structured_data: dict, match_meta: dict) -> WriterOutput:
        self._log("Calling xAI Grok API...")

        response = self.client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {"role": "system", "content": WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.7,
        )

        report_text = response.choices[0].message.content
        self._log(f"Report generated ({len(report_text.split())} words)")

        return WriterOutput(
            report_text=report_text,
            match_info=match_meta,
            structured_data=structured_data,
            model_used=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    def _call_anthropic(self, user_prompt: str, structured_data: dict, match_meta: dict) -> WriterOutput:
        import anthropic
        self._log("Calling Anthropic API...")

        message = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=WRITER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        report_text = message.content[0].text
        self._log(f"Report generated ({len(report_text.split())} words)")

        return WriterOutput(
            report_text=report_text,
            match_info=match_meta,
            structured_data=structured_data,
            model_used=message.model,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )

    def _call_openai(self, user_prompt: str, structured_data: dict, match_meta: dict) -> WriterOutput:
        self._log("Calling OpenAI API...")

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.7,
        )

        report_text = response.choices[0].message.content
        self._log(f"Report generated ({len(report_text.split())} words)")

        return WriterOutput(
            report_text=report_text,
            match_info=match_meta,
            structured_data=structured_data,
            model_used=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    def _build_prompt_data(self, structured_data: dict, match_meta: dict) -> dict:
        """
        Build the data payload sent to the LLM.
        Only includes computed, verified numbers — no raw events.
        """
        home = structured_data.get('home_team', match_meta.get('home_team', 'Home'))
        away = structured_data.get('away_team', match_meta.get('away_team', 'Away'))

        return {
            'match': {
                'home_team': home,
                'away_team': away,
                'score': f"{match_meta.get('home_score', '?')}–{match_meta.get('away_score', '?')}",
                'competition': match_meta.get('competition', ''),
                'date': match_meta.get('match_date', ''),
                'venue': match_meta.get('stadium', ''),
            },
            'full_match_stats': structured_data.get('full_match_metrics', {}),
            'tactical_overview': {
                home: structured_data.get('overall_home_tactics', {}),
                away: structured_data.get('overall_away_tactics', {}),
            },
            'phases': structured_data.get('phases', []),
            'key_events': [
                {k: v for k, v in e.items() if k != 'description'}
                for e in structured_data.get('key_events', [])
                if e.get('type') in ('Goal', 'Substitution')
            ],
            'key_tactical_themes': structured_data.get('key_tactical_themes', []),
            'ppda_legend': {
                'below_8': 'High Press (aggressive)',
                '8_to_12': 'Mid-Block',
                'above_12': 'Deep Block / Low Block',
            },
        }

    def _generate_demo_report(self, structured_data: dict, match_meta: dict) -> str:
        """
        Generate a demo report using only Python (no LLM).
        Used when no API key is configured. Shows the data pipeline
        works correctly even without LLM access.
        """
        home = match_meta.get('home_team', 'Home')
        away = match_meta.get('away_team', 'Away')
        home_score = match_meta.get('home_score', 0)
        away_score = match_meta.get('away_score', 0)
        competition = match_meta.get('competition', 'Competition')

        fm = structured_data.get('full_match_metrics', {})
        home_m = fm.get('home', {})
        away_m = fm.get('away', {})

        home_poss = home_m.get('possession_pct', 50.0)
        away_poss = away_m.get('possession_pct', 50.0)
        home_xg = home_m.get('xg', 0.0)
        away_xg = away_m.get('xg', 0.0)
        home_ppda = home_m.get('ppda', 99.0)
        away_ppda = away_m.get('ppda', 99.0)

        def press_label(ppda):
            if ppda < 8: return "high press"
            elif ppda < 12: return "mid-block"
            else: return "deep block"

        phases = structured_data.get('phases', [])
        key_events = structured_data.get('key_events', [])
        goals = [e for e in key_events if e.get('type') == 'Goal']

        phase_section = ""
        for phase in phases[:4]:
            hm = phase.get('home_metrics', {})
            am = phase.get('away_metrics', {})
            phase_section += f"\n**{phase['name']} ({phase['minutes']})**\n"
            phase_section += (
                f"{home} held {hm.get('possession_pct', 50):.1f}% possession "
                f"with a PPDA of {hm.get('ppda', 99):.1f} ({press_label(hm.get('ppda', 99))}). "
                f"{away} responded with {am.get('possession_pct', 50):.1f}% of the ball "
                f"and a PPDA of {am.get('ppda', 99):.1f} ({press_label(am.get('ppda', 99))}). "
                f"xG in this phase: {home} {hm.get('xg', 0):.2f}, {away} {am.get('xg', 0):.2f}.\n"
            )

        goals_str = "; ".join(
            f"{g['player']} ({g['team']}, {g['minute']}')" for g in goals
        ) if goals else "No goals recorded"

        report = f"""## {home} {home_score}–{away_score} {away} | {competition}

**{home} claimed a {home_score}–{away_score} victory**, in a match where the numbers told a story as compelling as the scoreline. The result was built on tactical discipline, pressing intensity, and clinical finishing at the moments that mattered most.

**Goals:** {goals_str}

---

### Phase-by-Phase Breakdown
{phase_section}

---

### Key Tactical Themes

**Possession and Pressing.** {home} dominated the ball with {home_poss:.1f}% possession across the ninety minutes, applying a {press_label(home_ppda)} (PPDA: {home_ppda:.1f}). {away}, by contrast, sat in a {press_label(away_ppda)} (PPDA: {away_ppda:.1f}), inviting pressure and looking to transition quickly.

**Chance Quality.** The xG totals — {home} {home_xg:.2f}, {away} {away_xg:.2f} — reveal {"a fair reflection of the match" if abs(home_xg - away_xg) < 0.4 else "that the scoreline somewhat flattered the winner"}. {"Both teams created chances of reasonable quality." if home_xg > 0.3 and away_xg > 0.3 else "Chances were limited, with the winning goal coming against the run of the xG."}

---

### Conclusion

The final whistle confirmed what the data had been suggesting since the opening phase: {home} deserved their victory, {"even if the margin could have been greater" if home_xg > home_score + 0.3 else "and the scoreline reflected the balance of play"}.

*All statistics sourced from StatsBomb Open Data. xG values use the StatsBomb xG model.*

---
⚠️ **Demo Mode** — This report was generated without an LLM API key. Add your XAI_API_KEY to .env for AI-generated narrative. All statistics above are computed from real match event data."""

        return report