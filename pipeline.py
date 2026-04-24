"""
Pitch Verdict — Main Pipeline Orchestrator

Runs all five agents in sequence and returns the final verified report.

Usage:
    from pipeline import PitchVerdictPipeline
    pipeline = PitchVerdictPipeline()
    result = pipeline.run(use_sample=True)
    print(result['verified_report'])

Or via CLI:
    python pipeline.py --sample
    python pipeline.py --match-id 3869685
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from agents.retriever import RetrieverAgent
from agents.phase_segmenter import PhaseSegmenterAgent
from agents.tactical_classifier import TacticalClassifierAgent
from agents.writer import WriterAgent
from agents.verifier import VerifierAgent


class PitchVerdictPipeline:
    """
    Orchestrates the five-agent Pitch Verdict pipeline.

    Pipeline:
    Retriever → Phase Segmenter → Tactical Classifier → Writer → Verifier

    The final report is not released until the Verifier confirms
    factual accuracy ≥ 90%.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.retriever = RetrieverAgent(verbose=verbose)
        self.segmenter = PhaseSegmenterAgent(verbose=verbose)
        self.classifier = TacticalClassifierAgent(verbose=verbose)
        self.writer = WriterAgent(verbose=verbose)
        self.verifier = VerifierAgent(verbose=verbose)

    def _log(self, msg: str):
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Pipeline] {msg}")
            print(f"{'='*60}")

    def run(
        self,
        match_id: Optional[int] = None,
        local_path: Optional[str] = None,
        use_sample: bool = False,
    ) -> dict:
        """
        Run the complete pipeline and return results.

        Returns a dict with:
        - match_data: MatchData object
        - segmentation: MatchSegmentation
        - tactical_analysis: MatchTacticalAnalysis
        - writer_output: WriterOutput
        - verification: VerificationReport
        - verified_report: str (the final, verified report text)
        - pipeline_stats: dict (timing, token usage, etc.)
        """
        import time
        start_time = time.time()
        timings = {}

        # ─────────── AGENT 1: RETRIEVER ───────────
        self._log("Agent 1/5: Retriever — Loading match data")
        t0 = time.time()
        match_data = self.retriever.load_match(
            match_id=match_id,
            local_path=local_path,
            use_sample=use_sample,
        )
        match_summary = self.retriever.get_match_summary(match_data)
        timings['retriever'] = round(time.time() - t0, 2)
        print(f"  ✅ Loaded: {match_data.result_string} ({match_data.total_events} events)")

        # ─────────── AGENT 2: PHASE SEGMENTER ───────────
        self._log("Agent 2/5: Phase Segmenter — Breaking match into chapters")
        t0 = time.time()
        segmentation = self.segmenter.segment(match_data)
        timings['segmenter'] = round(time.time() - t0, 2)
        print(f"  ✅ {len(segmentation.phases)} phases identified")
        print(f"  Key events: {[e['description'] for e in segmentation.key_events]}")

        # ─────────── AGENT 3: TACTICAL CLASSIFIER ───────────
        self._log("Agent 3/5: Tactical Classifier — Labeling tactics")
        t0 = time.time()
        tactical_analysis = self.classifier.classify(segmentation)
        structured_data = self.classifier.get_structured_data_for_writer(
            tactical_analysis, segmentation
        )
        timings['classifier'] = round(time.time() - t0, 2)
        print(f"  ✅ Overall: {match_data.home_team} = {tactical_analysis.overall_home_profile.pressing_style} | "
              f"{match_data.away_team} = {tactical_analysis.overall_away_profile.pressing_style}")

        # ─────────── AGENT 4: WRITER ───────────
        self._log("Agent 4/5: Writer — Generating match report")
        t0 = time.time()
        match_meta = {
            'home_team': match_data.home_team,
            'away_team': match_data.away_team,
            'home_score': match_data.home_score,
            'away_score': match_data.away_score,
            'competition': match_data.competition,
            'season': match_data.season,
            'match_date': match_data.match_date,
            'stadium': match_data.stadium,
        }
        writer_output = self.writer.write_report(structured_data, match_meta)
        timings['writer'] = round(time.time() - t0, 2)
        word_count = len(writer_output.report_text.split())
        print(f"  ✅ Report generated ({word_count} words, model: {writer_output.model_used})")

        # ─────────── AGENT 5: VERIFIER ───────────
        self._log("Agent 5/5: Verifier — Fact-checking every claim")
        t0 = time.time()
        verification = self.verifier.verify(
            writer_output=writer_output,
            ground_truth=tactical_analysis.ground_truth,
            match_meta=match_meta,
        )
        timings['verifier'] = round(time.time() - t0, 2)
        print(f"  {verification.summary()}")

        # ─────────── FINAL OUTPUT ───────────
        total_time = round(time.time() - start_time, 2)

        pipeline_stats = {
            'total_time_seconds': total_time,
            'timings': timings,
            'match': match_data.result_string,
            'total_events_processed': match_data.total_events,
            'phases_identified': len(segmentation.phases),
            'key_events': len(segmentation.key_events),
            'model_used': writer_output.model_used,
            'prompt_tokens': writer_output.prompt_tokens,
            'completion_tokens': writer_output.completion_tokens,
            'total_claims_verified': verification.total_claims,
            'factual_accuracy': round(verification.factual_accuracy, 4),
            'verification_passed': verification.verification_passed,
            'flagged_claims': verification.flagged_claims,
        }

        self._log(f"Pipeline complete in {total_time}s | Accuracy: {verification.factual_accuracy:.1%}")

        return {
            'match_data': match_data,
            'segmentation': segmentation,
            'tactical_analysis': tactical_analysis,
            'structured_data': structured_data,
            'writer_output': writer_output,
            'verification': verification,
            'verified_report': verification.revised_report,
            'original_report': writer_output.report_text,
            'pipeline_stats': pipeline_stats,
            'match_meta': match_meta,
        }

    def save_output(self, result: dict, output_dir: str = "outputs"):
        """Save pipeline outputs to disk."""
        Path(output_dir).mkdir(exist_ok=True)

        home = result['match_data'].home_team.replace(' ', '_')
        away = result['match_data'].away_team.replace(' ', '_')
        date = result['match_data'].match_date
        base_name = f"{home}_vs_{away}_{date}"

        # Save verified report
        report_path = f"{output_dir}/{base_name}_report.md"
        with open(report_path, 'w') as f:
            f.write(result['verified_report'])
        print(f"  Report saved: {report_path}")

        # Save pipeline stats
        stats_path = f"{output_dir}/{base_name}_stats.json"
        with open(stats_path, 'w') as f:
            # Make stats JSON-serializable
            stats = result['pipeline_stats'].copy()
            json.dump(stats, f, indent=2)
        print(f"  Stats saved: {stats_path}")

        # Save verification details
        verif_path = f"{output_dir}/{base_name}_verification.json"
        verif_data = {
            'summary': result['verification'].summary(),
            'accuracy': result['verification'].factual_accuracy,
            'total_claims': result['verification'].total_claims,
            'verified': result['verification'].verified_claims,
            'flagged': result['verification'].flagged_claims,
            'results': [
                {
                    'type': r.claim_type,
                    'status': r.status,
                    'extracted': r.extracted_value,
                    'expected': r.expected_value,
                    'team': r.team,
                    'explanation': r.explanation,
                }
                for r in result['verification'].results
            ]
        }
        with open(verif_path, 'w') as f:
            json.dump(verif_data, f, indent=2)
        print(f"  Verification saved: {verif_path}")

        return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Pitch Verdict — Verified AI Tactical Match Reports'
    )
    parser.add_argument('--match-id', type=int, help='StatsBomb match ID')
    parser.add_argument('--local', type=str, help='Path to local StatsBomb JSON file')
    parser.add_argument('--sample', action='store_true', help='Use built-in sample match (Euro 2024 Final)')
    parser.add_argument('--save', action='store_true', help='Save outputs to disk')
    parser.add_argument('--quiet', action='store_true', help='Suppress pipeline logs')

    args = parser.parse_args()

    if not args.match_id and not args.local and not args.sample:
        print("No input specified — using sample match (Spain vs England, Euro 2024 Final)")
        print("Run with --help for options.\n")
        args.sample = True

    pipeline = PitchVerdictPipeline(verbose=not args.quiet)
    result = pipeline.run(
        match_id=args.match_id,
        local_path=args.local,
        use_sample=args.sample,
    )

    print("\n" + "="*60)
    print("VERIFIED MATCH REPORT")
    print("="*60)
    print(result['verified_report'])
    print("\n" + "="*60)
    print("VERIFICATION DETAILS")
    print("="*60)
    for r in result['verification'].results:
        print(f"  [{r.status}] {r.explanation}")

    if args.save:
        print("\n" + "="*60)
        print("SAVING OUTPUTS")
        print("="*60)
        pipeline.save_output(result)

    return result


if __name__ == '__main__':
    main()
