"""
Agent 2: Phase Segmenter
Breaks a match into meaningful tactical chapters based on key events
(goals, substitutions, tactical shifts) and computes per-phase metrics.

This is the step that makes Pitch Verdict fundamentally different from
pasting stats into ChatGPT — a single prompt can't perform multi-phase
tactical reasoning over timestamped event streams.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from agents.retriever import MatchData


@dataclass
class MatchPhase:
    """
    A single tactical chapter of a match.
    All metrics here become ground truth for the Verifier.
    """
    name: str           # e.g., "Opening Phase", "After Spain Goal (47')"
    start_minute: int
    end_minute: int
    home_team: str
    away_team: str

    # Per-team metrics (computed from real event data)
    home_possession: float = 0.0
    away_possession: float = 0.0
    home_passes: int = 0
    away_passes: int = 0
    home_pass_completion: float = 0.0
    away_pass_completion: float = 0.0
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_ppda: float = 0.0      # Passes Per Defensive Action (pressing metric)
    away_ppda: float = 0.0
    home_pressures: int = 0
    away_pressures: int = 0
    home_high_turnovers: int = 0  # Turnovers won in attacking third
    away_high_turnovers: int = 0

    # Narrative context
    trigger_event: str = ""  # What caused this phase to start
    duration_minutes: int = 0

    def __post_init__(self):
        self.duration_minutes = self.end_minute - self.start_minute

    def get_metrics_for_team(self, team: str) -> dict:
        """Return all metrics for a specific team."""
        is_home = (team == self.home_team)
        prefix = "home" if is_home else "away"
        return {
            'possession_pct': getattr(self, f'{prefix}_possession'),
            'passes': getattr(self, f'{prefix}_passes'),
            'pass_completion_pct': getattr(self, f'{prefix}_pass_completion'),
            'shots': getattr(self, f'{prefix}_shots'),
            'shots_on_target': getattr(self, f'{prefix}_shots_on_target'),
            'xg': getattr(self, f'{prefix}_xg'),
            'ppda': getattr(self, f'{prefix}_ppda'),
            'pressures': getattr(self, f'{prefix}_pressures'),
            'high_turnovers': getattr(self, f'{prefix}_high_turnovers'),
        }


@dataclass
class MatchSegmentation:
    """Complete phase breakdown of a match."""
    phases: List[MatchPhase]
    key_events: List[dict]
    home_team: str
    away_team: str
    full_match_metrics: dict = field(default_factory=dict)

    def get_phase_by_minute(self, minute: int) -> Optional[MatchPhase]:
        for phase in self.phases:
            if phase.start_minute <= minute < phase.end_minute:
                return phase
        return None


class PhaseSegmenterAgent:
    """
    Agent 2: Phase Segmenter

    Responsibility: Identify natural tactical inflection points in a match
    and compute per-phase metrics from the raw event stream.

    Inflection points are:
    1. Goals (always a major phase break)
    2. Substitutions (especially double substitutions)
    3. Red cards
    4. Halftime
    5. Statistical shifts detected in rolling windows
       (e.g., sharp PPDA change suggesting tactical shift)

    Key metric computed: PPDA (Passes Per Defensive Action)
    PPDA = opponent_passes_in_their_half / defensive_actions_in_opponent_half
    - PPDA < 8: High press (aggressive)
    - PPDA 8–12: Mid-block
    - PPDA > 12: Deep block / low block
    """

    # PPDA thresholds for press classification
    PPDA_HIGH_PRESS = 8.0
    PPDA_MID_BLOCK = 12.0

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Phase Segmenter] {msg}")

    def segment(self, match_data: MatchData) -> MatchSegmentation:
        """Main entry: segment match and compute all phase metrics."""
        self._log(f"Segmenting match: {match_data.result_string}")

        # 1. Find key events that define phase boundaries
        key_events = self._find_key_events(match_data)
        self._log(f"Found {len(key_events)} key events: {[e['description'] for e in key_events]}")

        # 2. Build phase boundaries from key events
        boundaries = self._build_boundaries(key_events, match_data)
        self._log(f"Created {len(boundaries)} phases")

        # 3. Compute metrics for each phase
        phases = []
        for i, (start, end, trigger) in enumerate(boundaries):
            phase_name = self._name_phase(i, trigger, start, end)
            phase = self._compute_phase_metrics(
                match_data, start, end, phase_name, trigger
            )
            phases.append(phase)
            self._log(f"  Phase '{phase_name}': {start}'–{end}' | "
                      f"Spain PPDA={phase.home_ppda:.1f} | England PPDA={phase.away_ppda:.1f}")

        # 4. Compute full-match metrics
        full_match = self._compute_phase_metrics(
            match_data, 0, 95, "Full Match", "Full Match"
        )

        return MatchSegmentation(
            phases=phases,
            key_events=key_events,
            home_team=match_data.home_team,
            away_team=match_data.away_team,
            full_match_metrics={
                'home': full_match.get_metrics_for_team(match_data.home_team),
                'away': full_match.get_metrics_for_team(match_data.away_team),
            }
        )

    def _find_key_events(self, match_data: MatchData) -> List[dict]:
        """Extract goals, substitutions, and red cards with timestamps."""
        events = match_data.events
        key_events = []

        # Goals
        goals = events[
            (events['event_type'] == 'Shot') &
            (events['shot_outcome'] == 'Goal')
        ].copy()

        for _, goal in goals.iterrows():
            team = goal['team_name']
            player = goal.get('player_name', 'Unknown')
            minute = int(goal['minute'])
            xg = float(goal.get('xg', 0.0))
            key_events.append({
                'minute': minute,
                'type': 'Goal',
                'team': team,
                'player': player,
                'xg': xg,
                'description': f"Goal: {player} ({team}) {minute}'",
            })

        # Substitutions
        subs = events[events['event_type'] == 'Substitution'].copy()
        for _, sub in subs.iterrows():
            minute = int(sub['minute'])
            team = sub['team_name']
            player = sub.get('player_name', 'Unknown')
            key_events.append({
                'minute': minute,
                'type': 'Substitution',
                'team': team,
                'player': player,
                'description': f"Sub: {player} on ({team}) {minute}'",
            })

        # Halftime always a phase break
        key_events.append({
            'minute': 45,
            'type': 'Halftime',
            'team': None,
            'description': 'Halftime',
        })

        # Sort by minute
        key_events.sort(key=lambda e: e['minute'])
        return key_events

    def _build_boundaries(self, key_events: List[dict], match_data: MatchData) -> List[tuple]:
        """
        Build (start, end, trigger) tuples for each phase.
        Minimum phase length: 10 minutes (to avoid micro-phases).
        """
        breakpoints = sorted(set(
            [0] +
            [e['minute'] for e in key_events if e['type'] in ('Goal', 'Halftime')] +
            [90]
        ))

        # Merge phases shorter than 10 minutes
        merged = [breakpoints[0]]
        for bp in breakpoints[1:]:
            if bp - merged[-1] >= 10:
                merged.append(bp)

        if merged[-1] < 90:
            merged.append(90)

        # Build (start, end, trigger_description) tuples
        boundaries = []
        for i in range(len(merged) - 1):
            start, end = merged[i], merged[i + 1]
            # Find what triggered this phase start
            trigger = "Kickoff" if start == 0 else next(
                (e['description'] for e in key_events if e['minute'] == start),
                f"Minute {start}"
            )
            boundaries.append((start, end, trigger))

        return boundaries

    def _name_phase(self, idx: int, trigger: str, start: int, end: int) -> str:
        """Generate a human-readable phase name."""
        if start == 0:
            return "Opening Phase"
        if "Halftime" in trigger:
            return f"Second Half Opening"
        if "Goal" in trigger:
            return f"After {trigger}"
        return f"Phase {idx + 1} ({start}'–{end}')"

    def _compute_phase_metrics(
        self,
        match_data: MatchData,
        start: int,
        end: int,
        name: str,
        trigger: str,
    ) -> MatchPhase:
        """
        Compute all metrics for a single match phase.
        These are the GROUND TRUTH values the Verifier checks against.
        """
        events = match_data.events
        home = match_data.home_team
        away = match_data.away_team

        # Filter to this time window
        phase_events = events[
            (events['minute'] >= start) & (events['minute'] < end)
        ]

        home_events = phase_events[phase_events['team_name'] == home]
        away_events = phase_events[phase_events['team_name'] == away]

        # Possession (share of passes)
        home_passes_df = home_events[home_events['event_type'] == 'Pass']
        away_passes_df = away_events[away_events['event_type'] == 'Pass']
        total_passes = len(home_passes_df) + len(away_passes_df)

        if total_passes > 0:
            home_poss = round(len(home_passes_df) / total_passes * 100, 1)
            away_poss = round(100 - home_poss, 1)
        else:
            home_poss = away_poss = 50.0

        # Pass completion
        def pass_completion(pass_df):
            if len(pass_df) == 0:
                return 0.0
            complete = pass_df['pass_complete'].sum()
            return round(float(complete) / len(pass_df) * 100, 1)

        # Shots
        home_shots_df = home_events[home_events['event_type'] == 'Shot']
        away_shots_df = away_events[away_events['event_type'] == 'Shot']

        def shots_on_target(shots_df):
            return len(shots_df[shots_df['shot_outcome'].isin(['Goal', 'Saved'])])

        def total_xg(shots_df):
            return round(float(shots_df['xg'].sum()), 2)

        # PPDA computation
        # PPDA for a team = opponent_passes_in_opp_half / team_defensive_actions_in_opp_half
        # For home team pressing: count away passes in away half, home defensive actions in away half
        home_ppda = self._compute_ppda(home_events, away_events)
        away_ppda = self._compute_ppda(away_events, home_events)

        # Pressures
        home_pressures = len(home_events[home_events['event_type'] == 'Pressure'])
        away_pressures = len(away_events[away_events['event_type'] == 'Pressure'])

        # High turnovers (pressures won in attacking third, proxy)
        home_high_turnovers = len(home_events[
            (home_events['event_type'] == 'Pressure') &
            (home_events['zone'] == 'Attacking Third')
        ])
        away_high_turnovers = len(away_events[
            (away_events['event_type'] == 'Pressure') &
            (away_events['zone'] == 'Attacking Third')
        ])

        return MatchPhase(
            name=name,
            start_minute=start,
            end_minute=end,
            home_team=home,
            away_team=away,
            home_possession=home_poss,
            away_possession=away_poss,
            home_passes=int(len(home_passes_df)),
            away_passes=int(len(away_passes_df)),
            home_pass_completion=pass_completion(home_passes_df),
            away_pass_completion=pass_completion(away_passes_df),
            home_shots=int(len(home_shots_df)),
            away_shots=int(len(away_shots_df)),
            home_shots_on_target=shots_on_target(home_shots_df),
            away_shots_on_target=shots_on_target(away_shots_df),
            home_xg=total_xg(home_shots_df),
            away_xg=total_xg(away_shots_df),
            home_ppda=home_ppda,
            away_ppda=away_ppda,
            home_pressures=int(home_pressures),
            away_pressures=int(away_pressures),
            home_high_turnovers=int(home_high_turnovers),
            away_high_turnovers=int(away_high_turnovers),
            trigger_event=trigger,
        )

    def _compute_ppda(self, pressing_team_events: pd.DataFrame, pressed_team_events: pd.DataFrame) -> float:
        """
        Compute PPDA for a pressing team.

        PPDA = opponent_passes_in_their_own_half / pressing_team_defensive_actions_in_opponent_half

        StatsBomb pitch: x=0 is left goal, x=120 is right goal.
        "Own half" for pressed team = x < 60 (approximating their defensive half).
        "Opponent half" for pressing team = x > 60.

        A lower PPDA means more aggressive pressing.
        """
        # Opponent passes in their own half (pressing team perspective: those are the passes we're trying to disrupt)
        opp_passes_own_half = pressed_team_events[
            (pressed_team_events['event_type'] == 'Pass') &
            (pressed_team_events['x'] < 60)
        ]

        # Pressing team defensive actions in opponent's half
        pressing_def_actions = pressing_team_events[
            (pressing_team_events['event_type'].isin(['Pressure', 'Tackle', 'Interception', 'Block'])) &
            (pressing_team_events['x'] > 60)
        ]

        n_opp_passes = len(opp_passes_own_half)
        n_def_actions = len(pressing_def_actions)

        if n_def_actions == 0:
            return 99.0  # Effectively not pressing at all

        return round(n_opp_passes / n_def_actions, 1)

    def classify_pressing(self, ppda: float) -> str:
        """Translate PPDA value to tactical label."""
        if ppda < self.PPDA_HIGH_PRESS:
            return "High Press"
        elif ppda < self.PPDA_MID_BLOCK:
            return "Mid-Block"
        else:
            return "Deep Block / Low Block"
