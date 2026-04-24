"""
Unit tests for Pitch Verdict pipeline.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd


class TestRetrieverAgent:
    """Tests for Agent 1: Retriever."""

    def setup_method(self):
        from agents.retriever import RetrieverAgent
        self.agent = RetrieverAgent(verbose=False)

    def test_sample_match_loads(self):
        match = self.agent.load_match(use_sample=True)
        assert match is not None
        assert match.home_team == "Spain"
        assert match.away_team == "England"

    def test_sample_match_has_events(self):
        match = self.agent.load_match(use_sample=True)
        assert len(match.events) > 100

    def test_sample_match_has_goals(self):
        match = self.agent.load_match(use_sample=True)
        goals = match.events[
            (match.events['event_type'] == 'Shot') &
            (match.events['shot_outcome'] == 'Goal')
        ]
        assert len(goals) == 3  # Nico Williams, Cole Palmer, Oyarzabal

    def test_sample_score_correct(self):
        match = self.agent.load_match(use_sample=True)
        assert match.home_score == 2
        assert match.away_score == 1

    def test_events_have_coordinates(self):
        match = self.agent.load_match(use_sample=True)
        events_with_coords = match.events.dropna(subset=['x', 'y'])
        assert len(events_with_coords) > 50

    def test_zone_classification(self):
        assert self.agent._classify_zone(20) == 'Defensive Third'
        assert self.agent._classify_zone(60) == 'Middle Third'
        assert self.agent._classify_zone(100) == 'Attacking Third'
        assert self.agent._classify_zone(np.nan) == 'Unknown'

    def test_match_summary(self):
        match = self.agent.load_match(use_sample=True)
        summary = self.agent.get_match_summary(match)
        assert 'Spain' in summary
        assert 'England' in summary
        assert 'possession_pct' in summary['Spain']
        assert 'xg' in summary['Spain']
        # Possession percentages should sum to ~100
        total = summary['Spain']['possession_pct'] + summary['England']['possession_pct']
        assert abs(total - 100) < 2.0


class TestPhaseSegmenter:
    """Tests for Agent 2: Phase Segmenter."""

    def setup_method(self):
        from agents.retriever import RetrieverAgent
        from agents.phase_segmenter import PhaseSegmenterAgent
        self.retriever = RetrieverAgent(verbose=False)
        self.segmenter = PhaseSegmenterAgent(verbose=False)
        self.match = self.retriever.load_match(use_sample=True)

    def test_segmentation_produces_phases(self):
        seg = self.segmenter.segment(self.match)
        assert len(seg.phases) >= 2

    def test_phases_cover_full_match(self):
        seg = self.segmenter.segment(self.match)
        assert seg.phases[0].start_minute == 0
        assert seg.phases[-1].end_minute >= 85

    def test_goals_are_key_events(self):
        seg = self.segmenter.segment(self.match)
        goal_events = [e for e in seg.key_events if e['type'] == 'Goal']
        assert len(goal_events) >= 2

    def test_phase_metrics_computed(self):
        seg = self.segmenter.segment(self.match)
        phase = seg.phases[0]
        assert phase.home_possession >= 0
        assert phase.away_possession >= 0
        assert abs(phase.home_possession + phase.away_possession - 100) < 2.0

    def test_ppda_values_reasonable(self):
        seg = self.segmenter.segment(self.match)
        for phase in seg.phases:
            # PPDA should be positive
            assert phase.home_ppda > 0
            assert phase.away_ppda > 0

    def test_ppda_classification(self):
        seg = self.segmenter.segment(self.match)
        assert seg.segmenter.classify_pressing(5.0) == "High Press"
        assert seg.segmenter.classify_pressing(10.0) == "Mid-Block"
        assert seg.segmenter.classify_pressing(15.0) == "Deep Block / Low Block"

    def test_full_match_metrics(self):
        seg = self.segmenter.segment(self.match)
        fm = seg.full_match_metrics
        assert 'home' in fm
        assert 'away' in fm
        assert fm['home']['possession_pct'] > 0


class TestTacticalClassifier:
    """Tests for Agent 3: Tactical Classifier."""

    def setup_method(self):
        from agents.retriever import RetrieverAgent
        from agents.phase_segmenter import PhaseSegmenterAgent
        from agents.tactical_classifier import TacticalClassifierAgent

        self.retriever = RetrieverAgent(verbose=False)
        self.segmenter = PhaseSegmenterAgent(verbose=False)
        self.classifier = TacticalClassifierAgent(verbose=False)

        match = self.retriever.load_match(use_sample=True)
        self.segmentation = self.segmenter.segment(match)

    def test_classification_produces_profiles(self):
        analysis = self.classifier.classify(self.segmentation)
        assert analysis is not None
        assert len(analysis.phase_profiles) > 0

    def test_pressing_labels_valid(self):
        analysis = self.classifier.classify(self.segmentation)
        valid_labels = {"High Press", "Mid-Block", "Deep Block"}
        for profiles in analysis.phase_profiles:
            for team, profile in profiles.items():
                assert profile.pressing_style in valid_labels, \
                    f"Unexpected label: {profile.pressing_style}"

    def test_ground_truth_populated(self):
        analysis = self.classifier.classify(self.segmentation)
        assert 'Full Match' in analysis.ground_truth

    def test_structured_data_has_required_keys(self):
        analysis = self.classifier.classify(self.segmentation)
        data = self.classifier.get_structured_data_for_writer(analysis, self.segmentation)
        required = ['home_team', 'away_team', 'full_match_metrics', 'phases',
                    'key_events', 'key_tactical_themes', 'ground_truth']
        for key in required:
            assert key in data, f"Missing key: {key}"


class TestVerifier:
    """Tests for Agent 5: Verifier."""

    def setup_method(self):
        from agents.retriever import RetrieverAgent
        from agents.phase_segmenter import PhaseSegmenterAgent
        from agents.tactical_classifier import TacticalClassifierAgent
        from agents.verifier import VerifierAgent
        from agents.writer import WriterOutput

        self.verifier = VerifierAgent(verbose=False)

        retriever = RetrieverAgent(verbose=False)
        segmenter = PhaseSegmenterAgent(verbose=False)
        classifier = TacticalClassifierAgent(verbose=False)

        match = retriever.load_match(use_sample=True)
        seg = segmenter.segment(match)
        analysis = classifier.classify(seg)
        structured = classifier.get_structured_data_for_writer(analysis, seg)

        self.ground_truth = analysis.ground_truth
        self.match_meta = {
            'home_team': match.home_team,
            'away_team': match.away_team,
            'home_score': match.home_score,
            'away_score': match.away_score,
        }

        # Create a test report with known values
        fm = structured.get('full_match_metrics', {})
        home_poss = fm.get('home', {}).get('possession_pct', 55.0)
        home_xg = fm.get('home', {}).get('xg', 1.2)
        away_xg = fm.get('away', {}).get('xg', 0.8)

        test_report = f"""
## Spain 2–1 England | UEFA Euro 2024

Spain claimed a 2–1 victory. Spain held {home_poss:.1f}% possession throughout.

The xG totals — Spain {home_xg:.2f}, England {away_xg:.2f} — reflect the balance of play.

Spain applied a high press with a PPDA of 7.5, consistently hunting the ball in England's half.
        """

        self.writer_output = WriterOutput(
            report_text=test_report,
            match_info=self.match_meta,
            structured_data=structured,
            model_used='test',
        )

    def test_verification_runs(self):
        result = self.verifier.verify(
            self.writer_output, self.ground_truth, self.match_meta
        )
        assert result is not None

    def test_score_verified(self):
        result = self.verifier.verify(
            self.writer_output, self.ground_truth, self.match_meta
        )
        score_results = [r for r in result.results if r.claim_type == 'score']
        assert any(r.is_verified for r in score_results)

    def test_adversarial_injection_works(self):
        original = "Spain held 65.0% possession. Spain xG was 1.20."
        gt = {
            'Full Match': {
                'Spain': {'possession_pct': 65.0, 'xg': 1.20, 'ppda': 7.5,
                          'pass_completion_pct': 85.0, 'shots': 12},
                'England': {'possession_pct': 35.0, 'xg': 0.80, 'ppda': 12.0,
                            'pass_completion_pct': 78.0, 'shots': 8},
            }
        }
        corrupted, errors = self.verifier.inject_errors(original, gt, 'Spain', 'England')
        if errors:
            assert corrupted != original

    def test_factual_accuracy_between_0_and_1(self):
        result = self.verifier.verify(
            self.writer_output, self.ground_truth, self.match_meta
        )
        assert 0.0 <= result.factual_accuracy <= 1.0


class TestFullPipeline:
    """Integration test: run the full pipeline end-to-end."""

    def test_pipeline_runs_with_sample(self):
        from pipeline import PitchVerdictPipeline
        pipeline = PitchVerdictPipeline(verbose=False)
        result = pipeline.run(use_sample=True)

        assert 'match_data' in result
        assert 'verification' in result
        assert 'verified_report' in result
        assert len(result['verified_report']) > 100

    def test_pipeline_stats_populated(self):
        from pipeline import PitchVerdictPipeline
        pipeline = PitchVerdictPipeline(verbose=False)
        result = pipeline.run(use_sample=True)

        stats = result['pipeline_stats']
        assert stats['total_events_processed'] > 100
        assert stats['phases_identified'] >= 2
        assert 0.0 <= stats['factual_accuracy'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
