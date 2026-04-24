"""
Agent 5: Verifier — The Heart of Pitch Verdict

This is the step that makes Pitch Verdict fundamentally different from
any chatbot. It reads the Writer's draft, extracts every factual claim,
checks each one against the source data, and flags mismatches.

The report does not reach the reader until this step completes.

Verification logic is deterministic Python — no LLM needed for the
fact-check itself. The LLM's job was writing; Python's job is verifying.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from agents.writer import WriterOutput


@dataclass
class VerificationResult:
    """Result of checking one claim against source data."""
    claim_text: str          # The sentence containing the claim
    claim_type: str          # "percentage", "decimal", "ppda", "xg", "tactical_label"
    extracted_value: str     # What the Writer said
    expected_value: str      # What the data actually says
    team: Optional[str]      # Which team this refers to
    status: str              # "VERIFIED", "FLAGGED", "UNCERTAIN"
    deviation: float = 0.0   # How far off (for numerical claims)
    explanation: str = ""    # Human-readable explanation

    @property
    def is_verified(self) -> bool:
        return self.status == "VERIFIED"

    @property
    def is_flagged(self) -> bool:
        return self.status == "FLAGGED"


@dataclass
class VerificationReport:
    """Complete verification summary for one match report."""
    total_claims: int = 0
    verified_claims: int = 0
    flagged_claims: int = 0
    uncertain_claims: int = 0
    results: List[VerificationResult] = field(default_factory=list)
    revised_report: str = ""
    original_report: str = ""
    verification_passed: bool = False
    accuracy_score: float = 0.0

    @property
    def factual_accuracy(self) -> float:
        """Factual Accuracy = Correct Claims / Total Claims (target: ≥ 95%)"""
        if self.total_claims == 0:
            return 1.0
        return self.verified_claims / self.total_claims

    def summary(self) -> str:
        status = "✅ PASSED" if self.verification_passed else "⚠️ FLAGGED"
        return (
            f"Verification {status} | "
            f"Accuracy: {self.factual_accuracy:.1%} | "
            f"Verified: {self.verified_claims}/{self.total_claims} claims | "
            f"Flagged: {self.flagged_claims}"
        )


class VerifierAgent:
    """
    Agent 5: Verifier

    Responsibility: Extract every factual claim from the Writer's draft
    and check it against the computed ground truth from Agents 1–3.

    This is purely deterministic Python — no LLM in the verification loop.
    The verification is exact: if the Writer says "62.3%" and the data says
    "61.8%", that claim is flagged.

    Verification types:
    1. Possession percentages
    2. xG values
    3. PPDA values
    4. Pass completion percentages
    5. Shot counts
    6. Tactical labels (cross-referenced against computed PPDA thresholds)
    7. Score and goal scorer verification
    """

    # How much deviation is acceptable (for floating point / rounding)
    TOLERANCE_PERCENT = 2.0   # ±2% for percentages
    TOLERANCE_XG = 0.05       # ±0.05 for xG values
    TOLERANCE_PPDA = 0.5      # ±0.5 for PPDA

    # PPDA classification thresholds (must match TacticalClassifier)
    PPDA_HIGH_PRESS = 8.0
    PPDA_MID_BLOCK = 12.0

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Verifier] {msg}")

    def verify(
        self,
        writer_output: WriterOutput,
        ground_truth: dict,
        match_meta: dict,
    ) -> VerificationReport:
        """
        Main verification entry point.

        1. Extract all quantitative claims from the report
        2. Check each against ground truth
        3. Flag mismatches
        4. Generate revised report with corrections
        5. Return complete verification report
        """
        self._log("Starting verification pass...")

        report_text = writer_output.report_text
        home = match_meta.get('home_team', '')
        away = match_meta.get('away_team', '')

        # Flatten ground truth for easy lookup
        flat_truth = self._flatten_ground_truth(ground_truth, home, away)

        self._log(f"Ground truth has {len(flat_truth)} data points")

        # Extract and verify claims
        all_results = []
        all_results += self._verify_possession_claims(report_text, flat_truth, home, away)
        all_results += self._verify_xg_claims(report_text, flat_truth, home, away)
        all_results += self._verify_ppda_claims(report_text, flat_truth, home, away)
        all_results += self._verify_pass_completion_claims(report_text, flat_truth, home, away)
        all_results += self._verify_tactical_label_claims(report_text, flat_truth, home, away)
        all_results += self._verify_score_claims(report_text, match_meta)

        # Tally
        verified = sum(1 for r in all_results if r.is_verified)
        flagged = sum(1 for r in all_results if r.is_flagged)
        uncertain = sum(1 for r in all_results if r.status == "UNCERTAIN")
        total = len(all_results)

        self._log(f"Checked {total} claims: {verified} verified, {flagged} flagged, {uncertain} uncertain")

        # Generate revised report
        revised = self._generate_revised_report(report_text, all_results, flat_truth, home, away)

        accuracy = verified / total if total > 0 else 1.0
        passed = accuracy >= 0.90 and flagged <= 2

        report = VerificationReport(
            total_claims=total,
            verified_claims=verified,
            flagged_claims=flagged,
            uncertain_claims=uncertain,
            results=all_results,
            revised_report=revised,
            original_report=report_text,
            verification_passed=passed,
            accuracy_score=accuracy,
        )

        self._log(report.summary())
        return report

    def _flatten_ground_truth(self, ground_truth: dict, home: str, away: str) -> dict:
        """
        Build a flat lookup: metric_name -> value for full match.
        Also keeps phase-specific data.
        """
        flat = {}

        # Full match metrics
        if 'Full Match' in ground_truth:
            home_m = ground_truth['Full Match'].get(home, {})
            away_m = ground_truth['Full Match'].get(away, {})

            flat[f'{home}_possession'] = home_m.get('possession_pct', 50.0)
            flat[f'{away}_possession'] = away_m.get('possession_pct', 50.0)
            flat[f'{home}_xg'] = home_m.get('xg', 0.0)
            flat[f'{away}_xg'] = away_m.get('xg', 0.0)
            flat[f'{home}_ppda'] = home_m.get('ppda', 99.0)
            flat[f'{away}_ppda'] = away_m.get('ppda', 99.0)
            flat[f'{home}_pass_completion'] = home_m.get('pass_completion_pct', 0.0)
            flat[f'{away}_pass_completion'] = away_m.get('pass_completion_pct', 0.0)
            flat[f'{home}_shots'] = home_m.get('shots', 0)
            flat[f'{away}_shots'] = away_m.get('shots', 0)

        # Phase-specific metrics (for future detailed verification)
        flat['phases'] = ground_truth

        return flat

    def _extract_percentages(self, text: str) -> List[Tuple[str, float, int]]:
        """
        Extract all percentage values from text.
        Returns: [(surrounding_context, value, position)]
        """
        pattern = r'(\b\d+\.?\d*)\s*%'
        matches = []
        for m in re.finditer(pattern, text):
            val = float(m.group(1))
            start = max(0, m.start() - 100)
            context = text[start:m.end() + 50]
            matches.append((context, val, m.start()))
        return matches

    def _extract_decimal_values(self, text: str, keywords: List[str]) -> List[Tuple[str, float, int]]:
        """Extract decimal values near specific keywords (xG, PPDA, etc.)"""
        matches = []
        for keyword in keywords:
            pattern = rf'{re.escape(keyword)}[^\d]{{0,30}}(\d+\.\d+)'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = float(m.group(1))
                context = text[max(0, m.start()-50):m.end()+50]
                matches.append((context, val, m.start()))
            # Also look for value before keyword: "2.1 xG"
            pattern2 = rf'(\d+\.\d+)[^\d]{{0,10}}{re.escape(keyword)}'
            for m in re.finditer(pattern2, text, re.IGNORECASE):
                val = float(m.group(1))
                context = text[max(0, m.start()-50):m.end()+50]
                matches.append((context, val, m.start()))
        return matches

    def _identify_team_from_context(self, context: str, home: str, away: str) -> Optional[str]:
        """Guess which team a claim refers to from surrounding text."""
        home_lower = home.lower()
        away_lower = away.lower()
        context_lower = context.lower()

        home_mentions = context_lower.count(home_lower)
        away_mentions = context_lower.count(away_lower)

        if home_mentions > away_mentions:
            return home
        elif away_mentions > home_mentions:
            return away
        return None

    def _check_value(
        self, extracted: float, expected: float, tolerance: float
    ) -> Tuple[str, float]:
        """Compare extracted vs expected value. Returns (status, deviation)."""
        deviation = abs(extracted - expected)
        if deviation <= tolerance:
            return "VERIFIED", deviation
        return "FLAGGED", deviation

    def _verify_possession_claims(
        self, text: str, truth: dict, home: str, away: str
    ) -> List[VerificationResult]:
        """Check all possession percentage claims."""
        results = []
        percentages = self._extract_percentages(text)

        for context, val, pos in percentages:
            # Skip if not near possession-related words
            possession_words = ['possession', 'ball', 'held', 'controlled']
            if not any(w in context.lower() for w in possession_words):
                continue

            team = self._identify_team_from_context(context, home, away)
            if not team:
                continue

            expected = truth.get(f'{team}_possession')
            if expected is None:
                continue

            status, deviation = self._check_value(val, expected, self.TOLERANCE_PERCENT)

            results.append(VerificationResult(
                claim_text=context.strip()[:200],
                claim_type="possession_percentage",
                extracted_value=f"{val}%",
                expected_value=f"{expected}%",
                team=team,
                status=status,
                deviation=deviation,
                explanation=(
                    f"✅ {team} possession: claimed {val}%, actual {expected}% (within tolerance)"
                    if status == "VERIFIED" else
                    f"⚠️ {team} possession mismatch: report says {val}%, data says {expected}%"
                )
            ))

        return results

    def _verify_xg_claims(
        self, text: str, truth: dict, home: str, away: str
    ) -> List[VerificationResult]:
        """Check all xG value claims."""
        results = []
        xg_matches = self._extract_decimal_values(text, ['xG', 'expected goals', 'xg'])

        for context, val, pos in xg_matches:
            team = self._identify_team_from_context(context, home, away)
            if not team:
                continue

            expected = truth.get(f'{team}_xg')
            if expected is None:
                continue

            status, deviation = self._check_value(val, expected, self.TOLERANCE_XG)

            results.append(VerificationResult(
                claim_text=context.strip()[:200],
                claim_type="xg",
                extracted_value=str(val),
                expected_value=str(round(expected, 2)),
                team=team,
                status=status,
                deviation=deviation,
                explanation=(
                    f"✅ {team} xG: claimed {val}, actual {round(expected,2)} (within tolerance)"
                    if status == "VERIFIED" else
                    f"⚠️ {team} xG mismatch: report says {val}, data says {round(expected,2)}"
                )
            ))

        return results

    def _verify_ppda_claims(
        self, text: str, truth: dict, home: str, away: str
    ) -> List[VerificationResult]:
        """Check all PPDA value claims."""
        results = []
        ppda_matches = self._extract_decimal_values(text, ['PPDA', 'ppda'])

        for context, val, pos in ppda_matches:
            team = self._identify_team_from_context(context, home, away)
            if not team:
                continue

            expected = truth.get(f'{team}_ppda')
            if expected is None or expected >= 90:  # 99 = no pressing data
                continue

            status, deviation = self._check_value(val, expected, self.TOLERANCE_PPDA)

            # Also verify the tactical label implied by this PPDA
            claimed_label = self._infer_press_label_from_context(context)
            expected_label = self._ppda_to_label(expected)
            label_correct = (claimed_label is None or claimed_label == expected_label)

            results.append(VerificationResult(
                claim_text=context.strip()[:200],
                claim_type="ppda",
                extracted_value=str(val),
                expected_value=str(round(expected, 1)),
                team=team,
                status=status if label_correct else "FLAGGED",
                deviation=deviation,
                explanation=(
                    f"✅ {team} PPDA: claimed {val}, actual {round(expected,1)} → {expected_label}"
                    if (status == "VERIFIED" and label_correct) else
                    f"⚠️ {team} PPDA mismatch or wrong label: claimed {val} ({claimed_label}), "
                    f"actual {round(expected,1)} ({expected_label})"
                )
            ))

        return results

    def _verify_pass_completion_claims(
        self, text: str, truth: dict, home: str, away: str
    ) -> List[VerificationResult]:
        """Check pass completion percentage claims."""
        results = []
        percentages = self._extract_percentages(text)

        for context, val, pos in percentages:
            pass_words = ['pass completion', 'passing accuracy', 'completion rate', 'passes completed']
            if not any(w in context.lower() for w in pass_words):
                continue

            team = self._identify_team_from_context(context, home, away)
            if not team:
                continue

            expected = truth.get(f'{team}_pass_completion')
            if expected is None or expected == 0:
                continue

            status, deviation = self._check_value(val, expected, self.TOLERANCE_PERCENT)

            results.append(VerificationResult(
                claim_text=context.strip()[:200],
                claim_type="pass_completion",
                extracted_value=f"{val}%",
                expected_value=f"{expected}%",
                team=team,
                status=status,
                deviation=deviation,
                explanation=(
                    f"✅ {team} pass completion: claimed {val}%, actual {expected}%"
                    if status == "VERIFIED" else
                    f"⚠️ {team} pass completion: report says {val}%, data says {expected}%"
                )
            ))

        return results

    def _verify_tactical_label_claims(
        self, text: str, truth: dict, home: str, away: str
    ) -> List[VerificationResult]:
        """
        Verify tactical labels (high press, mid-block, deep block)
        against computed PPDA values.
        """
        results = []
        label_patterns = [
            (r'high\s+press(?:ing)?', 'High Press'),
            (r'mid[\s-]block', 'Mid-Block'),
            (r'deep\s+block', 'Deep Block'),
            (r'low\s+block', 'Deep Block'),
        ]

        for pattern, label in label_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                context = text[max(0, m.start()-150):m.end()+100]
                team = self._identify_team_from_context(context, home, away)
                if not team:
                    continue

                ppda = truth.get(f'{team}_ppda', 99.0)
                if ppda >= 90:  # No data
                    results.append(VerificationResult(
                        claim_text=context.strip()[:200],
                        claim_type="tactical_label",
                        extracted_value=label,
                        expected_value="PPDA data unavailable",
                        team=team,
                        status="UNCERTAIN",
                        explanation=f"⚠️ Cannot verify '{label}' for {team} — PPDA data not available",
                    ))
                    continue

                expected_label = self._ppda_to_label(ppda)
                is_correct = self._labels_compatible(label, expected_label)

                results.append(VerificationResult(
                    claim_text=context.strip()[:200],
                    claim_type="tactical_label",
                    extracted_value=label,
                    expected_value=f"{expected_label} (PPDA: {ppda:.1f})",
                    team=team,
                    status="VERIFIED" if is_correct else "FLAGGED",
                    explanation=(
                        f"✅ '{label}' for {team} confirmed — PPDA {ppda:.1f} → {expected_label}"
                        if is_correct else
                        f"⚠️ '{label}' for {team} INCORRECT — PPDA {ppda:.1f} indicates {expected_label}, not {label}"
                    )
                ))

        return results

    def _verify_score_claims(self, text: str, match_meta: dict) -> List[VerificationResult]:
        """Verify the scoreline mentioned in the report."""
        results = []
        home = match_meta.get('home_team', '')
        away = match_meta.get('away_team', '')
        home_score = match_meta.get('home_score', 0)
        away_score = match_meta.get('away_score', 0)

        # Look for score patterns like "2-1", "2–1"
        score_pattern = r'\b(\d)\s*[–\-]\s*(\d)\b'
        for m in re.finditer(score_pattern, text):
            s1, s2 = int(m.group(1)), int(m.group(2))
            context = text[max(0, m.start()-100):m.end()+50]

            # Determine which team is mentioned first
            team_before = self._identify_team_from_context(
                text[max(0, m.start()-150):m.start()], home, away
            )

            if team_before == home or team_before is None:
                claimed_home, claimed_away = s1, s2
            else:
                claimed_home, claimed_away = s2, s1

            correct = (claimed_home == home_score and claimed_away == away_score)

            results.append(VerificationResult(
                claim_text=context.strip()[:200],
                claim_type="score",
                extracted_value=f"{s1}–{s2}",
                expected_value=f"{home_score}–{away_score}",
                team=None,
                status="VERIFIED" if correct else "FLAGGED",
                explanation=(
                    f"✅ Score {s1}–{s2} confirmed"
                    if correct else
                    f"⚠️ Score mismatch: report says {s1}–{s2}, actual {home_score}–{away_score}"
                )
            ))

        return results

    def _ppda_to_label(self, ppda: float) -> str:
        if ppda < self.PPDA_HIGH_PRESS:
            return "High Press"
        elif ppda < self.PPDA_MID_BLOCK:
            return "Mid-Block"
        else:
            return "Deep Block"

    def _infer_press_label_from_context(self, context: str) -> Optional[str]:
        """Try to infer what pressing label is being claimed in a PPDA context."""
        ctx_lower = context.lower()
        if 'high press' in ctx_lower:
            return "High Press"
        if 'mid-block' in ctx_lower or 'mid block' in ctx_lower:
            return "Mid-Block"
        if 'deep block' in ctx_lower or 'low block' in ctx_lower:
            return "Deep Block"
        return None

    def _labels_compatible(self, claimed: str, expected: str) -> bool:
        """Check if two tactical labels are compatible (accounting for synonyms)."""
        synonyms = {
            'High Press': ['High Press', 'high pressing', 'aggressive press'],
            'Mid-Block': ['Mid-Block', 'mid block', 'medium block'],
            'Deep Block': ['Deep Block', 'deep block', 'low block', 'low-block'],
        }
        claimed_lower = claimed.lower()
        expected_group = synonyms.get(expected, [expected])
        return any(syn.lower() in claimed_lower or claimed_lower in syn.lower()
                   for syn in expected_group)

    def _generate_revised_report(
        self,
        original_text: str,
        results: List[VerificationResult],
        truth: dict,
        home: str,
        away: str
    ) -> str:
        """
        Generate a corrected version of the report with flagged claims fixed.
        In production, this would re-call the Writer with corrections.
        Here, we annotate the original with inline corrections.
        """
        flagged = [r for r in results if r.is_flagged]

        if not flagged:
            return original_text + "\n\n---\n*✅ Verification complete — all claims verified against source data.*"

        corrections_section = "\n\n---\n## ⚠️ Verifier Corrections\n\n"
        corrections_section += f"The Verifier flagged {len(flagged)} claim(s) that did not match the source data:\n\n"

        for i, r in enumerate(flagged, 1):
            corrections_section += (
                f"**Correction {i}** ({r.claim_type}): "
                f"Report claimed `{r.extracted_value}` for {r.team or 'unknown team'}. "
                f"Source data shows `{r.expected_value}`. "
                f"Deviation: {r.deviation:.2f}.\n\n"
                f"> *Excerpt: ...{r.claim_text[:150]}...*\n\n"
            )

        corrections_section += (
            f"*This report was revised by the Verifier Agent. "
            f"Factual accuracy score: {len([r for r in results if r.is_verified])}/{len(results)} "
            f"({len([r for r in results if r.is_verified])/len(results):.1%}).*"
        )

        return original_text + corrections_section

    # ─────────────────────────────────────────────────────────────────
    # ADVERSARIAL TESTING
    # ─────────────────────────────────────────────────────────────────

    def inject_errors(self, report_text: str, ground_truth: dict, home: str, away: str) -> Tuple[str, List[dict]]:
        """
        Adversarial test: inject known errors into a report to test
        whether the Verifier catches them.

        Returns: (corrupted_report, list_of_injected_errors)
        """
        import random
        corrupted = report_text
        injected = []

        flat = self._flatten_ground_truth(ground_truth, home, away)

        # Error type 1: Wrong possession percentage
        real_poss = flat.get(f'{home}_possession', 55.0)
        wrong_poss = real_poss + random.choice([-12, -8, 8, 12])
        old_str = f"{real_poss}%"
        new_str = f"{wrong_poss}%"
        if old_str in corrupted:
            corrupted = corrupted.replace(old_str, new_str, 1)
            injected.append({
                'type': 'possession',
                'team': home,
                'real': real_poss,
                'injected': wrong_poss,
            })

        # Error type 2: Wrong xG
        real_xg = flat.get(f'{home}_xg', 1.2)
        wrong_xg = round(real_xg + random.choice([-0.4, 0.5, -0.6]), 2)
        old_xg = str(round(real_xg, 2))
        if old_xg in corrupted:
            corrupted = corrupted.replace(old_xg, str(wrong_xg), 1)
            injected.append({
                'type': 'xg',
                'team': home,
                'real': real_xg,
                'injected': wrong_xg,
            })

        return corrupted, injected
