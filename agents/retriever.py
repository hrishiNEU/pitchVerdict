"""
Agent 1: Retriever
Loads raw StatsBomb event data and parses it into a structured DataFrame.
Includes a catalogue of 5 sample matches for demo/offline use.
"""

import json
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class MatchData:
    match_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    competition: str
    season: str
    match_date: str
    stadium: str
    events: pd.DataFrame
    lineups: dict
    match_week: Optional[int] = None

    @property
    def result_string(self):
        return f"{self.home_team} {self.home_score}–{self.away_score} {self.away_team}"

    @property
    def total_events(self):
        return len(self.events)


SAMPLE_MATCHES = {
    "euro2024_final": {
        "label": "🏆 UEFA Euro 2024 Final",
        "sublabel": "Spain 2–1 England · Berlin · Jul 14 2024",
        "home_team": "Spain", "away_team": "England",
        "home_score": 2, "away_score": 1,
        "competition": "UEFA Euro 2024", "season": "2023/24",
        "match_date": "2024-07-14", "stadium": "Olympiastadion, Berlin",
        "match_id": 3869685,
        "narrative": "Spain's relentless high press suffocated England throughout. Nico Williams and Oyarzabal bookended a Palmer equaliser.",
    },
    "ucl2024_final": {
        "label": "🏆 UEFA Champions League Final",
        "sublabel": "Real Madrid 2–0 Dortmund · London · Jun 1 2024",
        "home_team": "Real Madrid", "away_team": "Borussia Dortmund",
        "home_score": 2, "away_score": 0,
        "competition": "UEFA Champions League 2023/24", "season": "2023/24",
        "match_date": "2024-06-01", "stadium": "Wembley Stadium, London",
        "match_id": 3943043,
        "narrative": "Dortmund dominated the first half but Madrid's clinical second-half turnaround — Carvajal and Vinicius — sealed a record 15th title.",
    },
    "worldcup2022_final": {
        "label": "🌍 FIFA World Cup 2022 Final",
        "sublabel": "Argentina 3–3 France (4-2 pens) · Lusail · Dec 18 2022",
        "home_team": "Argentina", "away_team": "France",
        "home_score": 3, "away_score": 3,
        "competition": "FIFA World Cup 2022", "season": "2022",
        "match_date": "2022-12-18", "stadium": "Lusail Stadium, Qatar",
        "match_id": 3869151,
        "narrative": "One of the greatest finals ever. Mbappé's hat-trick levelled Díaz and Di María's opener before Montiel's penalty clinched it for Argentina.",
    },
    "elclasico_2024": {
        "label": "⚽ El Clásico — La Liga",
        "sublabel": "Barcelona 4–0 Real Madrid · Barcelona · Oct 26 2024",
        "home_team": "Barcelona", "away_team": "Real Madrid",
        "home_score": 4, "away_score": 0,
        "competition": "La Liga 2024/25", "season": "2024/25",
        "match_date": "2024-10-26", "stadium": "Estadi Olímpic Lluís Companys",
        "match_id": 3944010,
        "narrative": "A dominant tactical display from Flick's Barcelona. Yamal and Raphinha tormented Madrid's high line while Lewandowski converted two clinical finishes.",
    },
    "nld_2024": {
        "label": "⚽ North London Derby — Premier League",
        "sublabel": "Arsenal 1–0 Tottenham · London · Sep 15 2024",
        "home_team": "Arsenal", "away_team": "Tottenham",
        "home_score": 1, "away_score": 0,
        "competition": "Premier League 2024/25", "season": "2024/25",
        "match_date": "2024-09-15", "stadium": "Emirates Stadium, London",
        "match_id": 3944201,
        "narrative": "Arsenal's press-heavy 4-3-3 stifled Tottenham's build-up. Saka's second-half winner was the only goal but the xG told a more emphatic story.",
    },
}


class RetrieverAgent:
    PITCH_LENGTH = 120.0
    PITCH_WIDTH = 80.0
    DEFENSIVE_THIRD_MAX = 40.0
    MIDDLE_THIRD_MAX = 80.0

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._statsbomb_available = self._check_statsbomb()

    def _check_statsbomb(self):
        try:
            from statsbombpy import sb
            return True
        except ImportError:
            return False

    def _log(self, msg):
        if self.verbose:
            print(f"[Retriever] {msg}")

    def load_match(
        self,
        match_id: Optional[int] = None,
        local_path: Optional[str] = None,
        use_sample: bool = False,
        sample_key: str = "euro2024_final",
    ) -> MatchData:
        if use_sample or (match_id is None and local_path is None):
            self._log(f"Loading sample match: {sample_key}")
            return self._load_sample_match(sample_key)
        if local_path:
            self._log(f"Loading from file: {local_path}")
            return self._load_from_file(local_path)
        if self._statsbomb_available:
            self._log(f"Loading match {match_id} from StatsBomb…")
            return self._load_from_statsbomb(match_id)
        self._log("StatsBomb unavailable — using sample")
        return self._load_sample_match(sample_key)

    def _load_from_statsbomb(self, match_id: int) -> MatchData:
        from statsbombpy import sb
        matches = sb.matches(competition_id=55, season_id=282)
        mi = matches[matches['match_id'] == match_id].iloc[0]
        events = self._parse_events(sb.events(match_id=match_id))
        lineups_raw = sb.lineups(match_id=match_id)
        lineups = {t: [p['player_name'] for p in pl] for t, pl in lineups_raw.items()}
        return MatchData(
            match_id=match_id, home_team=mi['home_team'], away_team=mi['away_team'],
            home_score=mi['home_score'], away_score=mi['away_score'],
            competition=mi['competition'], season=mi['season'],
            match_date=str(mi['match_date']), stadium=mi.get('stadium', 'Unknown'),
            events=events, lineups=lineups,
        )

    def _load_from_file(self, path: str) -> MatchData:
        """
        Load a match from a local JSON file.

        Supports two formats:
        1. Pitch Verdict format (recommended):
           { "metadata": { "home_team": ..., "away_score": ..., ... }, "events": [...] }

        2. Flat list of events (legacy / raw StatsBomb format):
           [ { "event_type": ..., "team_name": ..., ... }, ... ]

        The Pitch Verdict format is what the sample file uses. It lets you specify
        the team names, score, competition, and stadium so they show up correctly
        in the report and the match header.
        """
        with open(path, 'r') as f:
            raw = json.load(f)

        # Format 1: wrapped with metadata
        if isinstance(raw, dict) and 'events' in raw:
            meta = raw.get('metadata', {})
            events_list = raw['events']

            # Events are already in flat parsed format — load directly as DataFrame
            events_df = pd.DataFrame(events_list)

            # Ensure required columns exist (handle both pre-parsed and StatsBomb raw)
            if 'event_type' not in events_df.columns and 'type' in events_df.columns:
                events_df = self._parse_events(events_df)
            else:
                # Already flat — just ensure all columns are present
                for col, default in [('x', float('nan')), ('y', float('nan')),
                                     ('zone', 'Unknown'), ('pass_complete', None),
                                     ('shot_outcome', ''), ('xg', 0.0), ('player_name', '')]:
                    if col not in events_df.columns:
                        events_df[col] = default
                # Ensure zone is computed if x exists
                if 'x' in events_df.columns:
                    events_df['zone'] = events_df['x'].apply(self._classify_zone)

            home = meta.get('home_team', 'Home Team')
            away = meta.get('away_team', 'Away Team')

            return MatchData(
                match_id=meta.get('match_id', 0),
                home_team=home,
                away_team=away,
                home_score=meta.get('home_score', 0),
                away_score=meta.get('away_score', 0),
                competition=meta.get('competition', 'Unknown Competition'),
                season=meta.get('season', 'Unknown'),
                match_date=meta.get('match_date', 'Unknown'),
                stadium=meta.get('stadium', 'Unknown'),
                events=events_df,
                lineups={},
            )

        # Format 2: flat list (raw StatsBomb or other)
        events = self._parse_events(pd.DataFrame(raw))
        # Try to infer teams from the data
        teams = events['team_name'].dropna().unique() if 'team_name' in events.columns else []
        home = teams[0] if len(teams) > 0 else 'Home Team'
        away = teams[1] if len(teams) > 1 else 'Away Team'
        return MatchData(match_id=0, home_team=home, away_team=away,
            home_score=0, away_score=0, competition="Unknown", season="Unknown",
            match_date="Unknown", stadium="Unknown", events=events, lineups={})

    def _parse_events(self, events_raw: pd.DataFrame) -> pd.DataFrame:
        ev = events_raw.copy()

        ev['x'] = ev['location'].apply(lambda l: l[0] if isinstance(l, list) and len(l) >= 2 else np.nan) if 'location' in ev.columns else np.nan
        ev['y'] = ev['location'].apply(lambda l: l[1] if isinstance(l, list) and len(l) >= 2 else np.nan) if 'location' in ev.columns else np.nan
        ev['minute'] = pd.to_numeric(ev.get('minute', 0), errors='coerce').fillna(0)
        ev['second'] = pd.to_numeric(ev.get('second', 0), errors='coerce').fillna(0)
        ev['time_decimal'] = ev['minute'] + ev['second'] / 60.0

        if 'type' in ev.columns:
            try: ev['event_type'] = ev['type'].apply(lambda t: t.get('name', str(t)) if isinstance(t, dict) else str(t))
            except: ev['event_type'] = ev['type'].astype(str)
        else:
            ev['event_type'] = 'Unknown'

        if 'team' in ev.columns:
            try: ev['team_name'] = ev['team'].apply(lambda t: t.get('name', str(t)) if isinstance(t, dict) else str(t))
            except: ev['team_name'] = ev['team'].astype(str)
        else:
            ev['team_name'] = 'Unknown'

        if 'player' in ev.columns:
            try: ev['player_name'] = ev['player'].apply(lambda p: p.get('name','') if isinstance(p, dict) else str(p) if pd.notna(p) else '')
            except: ev['player_name'] = ''
        else:
            ev['player_name'] = ''

        ev['zone'] = ev['x'].apply(self._classify_zone)

        if 'pass' in ev.columns:
            try: ev['pass_complete'] = ev['pass'].apply(lambda p: isinstance(p, dict) and isinstance(p.get('outcome'), dict) and p['outcome'].get('name') not in ['Incomplete','Out','Pass Offside'] if isinstance(p, dict) else None)
            except: ev['pass_complete'] = None
        else:
            ev['pass_complete'] = None

        if 'shot' in ev.columns:
            try:
                ev['shot_outcome'] = ev['shot'].apply(lambda s: s.get('outcome',{}).get('name','') if isinstance(s, dict) else '')
                ev['xg'] = ev['shot'].apply(lambda s: s.get('statsbomb_xg', 0.0) if isinstance(s, dict) else 0.0)
            except:
                ev['shot_outcome'] = ''; ev['xg'] = 0.0
        else:
            ev['shot_outcome'] = ''; ev['xg'] = 0.0

        return ev

    def _classify_zone(self, x):
        if pd.isna(x): return 'Unknown'
        if x < self.DEFENSIVE_THIRD_MAX: return 'Defensive Third'
        if x < self.MIDDLE_THIRD_MAX: return 'Middle Third'
        return 'Attacking Third'

    def get_match_summary(self, match_data: MatchData) -> dict:
        ev = match_data.events
        summary = {}
        for team in [match_data.home_team, match_data.away_team]:
            te = ev[ev['team_name'] == team]
            passes = te[te['event_type'] == 'Pass']
            shots = te[te['event_type'] == 'Shot']
            pressures = te[te['event_type'] == 'Pressure']
            n_passes = len(passes)
            completed = passes['pass_complete'].sum() if 'pass_complete' in passes.columns else 0
            all_passes = len(ev[ev['event_type'] == 'Pass'])
            summary[team] = {
                'possession_pct': round(n_passes / all_passes * 100, 1) if all_passes > 0 else 50.0,
                'total_passes': int(n_passes),
                'pass_completion_pct': round(float(completed) / n_passes * 100, 1) if n_passes > 0 else 0.0,
                'shots': int(len(shots)),
                'shots_on_target': int(len(shots[shots['shot_outcome'].isin(['Goal','Saved'])])) if 'shot_outcome' in shots.columns else 0,
                'goals': int(len(shots[shots['shot_outcome'] == 'Goal'])) if 'shot_outcome' in shots.columns else 0,
                'xg': round(float(shots['xg'].sum()), 2) if 'xg' in shots.columns else 0.0,
                'pressures': int(len(pressures)),
            }
        return summary

    # ── Sample data ───────────────────────────────────────────────

    def _load_sample_match(self, key: str = "euro2024_final") -> MatchData:
        if key not in SAMPLE_MATCHES:
            key = "euro2024_final"
        cfg = SAMPLE_MATCHES[key]
        self._log(f"Generating synthetic data: {cfg['home_team']} vs {cfg['away_team']}")
        events = self._generate_events(cfg['home_team'], cfg['away_team'], key)
        return MatchData(
            match_id=cfg['match_id'], home_team=cfg['home_team'], away_team=cfg['away_team'],
            home_score=cfg['home_score'], away_score=cfg['away_score'],
            competition=cfg['competition'], season=cfg['season'],
            match_date=cfg['match_date'], stadium=cfg['stadium'],
            events=events, lineups={},
        )

    def _generate_events(self, home: str, away: str, key: str) -> pd.DataFrame:
        np.random.seed(hash(key) % (2**31))
        rows = []

        profiles = {
            "euro2024_final": {
                "home_press": 12, "away_press": 5, "home_pass": 0.87, "away_pass": 0.76,
                "home_poss": 0.60, "away_poss": 0.40,
                "goals": [(47,home,"Nico Williams",0.31),(73,away,"Cole Palmer",0.19),(86,home,"Mikel Oyarzabal",0.28)],
                "subs": [(61,away,"Ollie Watkins")],
            },
            "ucl2024_final": {
                "home_press": 6, "away_press": 9, "home_pass": 0.88, "away_pass": 0.82,
                "home_poss": 0.48, "away_poss": 0.52,
                "goals": [(74,home,"Dani Carvajal",0.22),(83,home,"Vinicius Jr.",0.41)],
                "subs": [(60,home,"Brahim Díaz"),(72,away,"Julian Brandt")],
            },
            "worldcup2022_final": {
                "home_press": 8, "away_press": 10, "home_pass": 0.84, "away_pass": 0.81,
                "home_poss": 0.45, "away_poss": 0.55,
                "goals": [(23,home,"Ángel Di María",0.28),(36,home,"Enzo Fernández",0.15),
                          (80,away,"Kylian Mbappé",0.35),(81,away,"Kylian Mbappé",0.42),
                          (108,home,"Gonzalo Montiel",0.20),(118,away,"Kylian Mbappé",0.38)],
                "subs": [(46,away,"Kylian Mbappé"),(57,home,"Lautaro Martinez")],
            },
            "elclasico_2024": {
                "home_press": 7, "away_press": 11, "home_pass": 0.90, "away_pass": 0.83,
                "home_poss": 0.62, "away_poss": 0.38,
                "goals": [(29,home,"Robert Lewandowski",0.45),(57,home,"Raphinha",0.22),
                          (78,home,"Robert Lewandowski",0.38),(84,home,"Lamine Yamal",0.18)],
                "subs": [(55,away,"Rodrygo"),(66,home,"Fermin Lopez")],
            },
            "nld_2024": {
                "home_press": 7, "away_press": 13, "home_pass": 0.88, "away_pass": 0.77,
                "home_poss": 0.58, "away_poss": 0.42,
                "goals": [(64,home,"Bukayo Saka",0.24)],
                "subs": [(52,away,"Brennan Johnson"),(71,home,"Leandro Trossard")],
            },
        }

        p = profiles.get(key, profiles["euro2024_final"])
        hp, ap = p['home_poss'], p['away_poss']

        def add(team, start, end, pass_rate, press_rate, pass_comp, zone_w):
            for minute in range(start, end):
                for _ in range(np.random.poisson(pass_rate / 10)):
                    zi = np.random.choice(3, p=zone_w)
                    x = np.random.uniform(*[(5,39),(40,79),(80,118)][zi])
                    rows.append({'minute': minute, 'second': np.random.randint(0,60),
                        'event_type': 'Pass', 'team_name': team,
                        'x': x, 'y': np.random.uniform(5,75), 'zone': self._classify_zone(x),
                        'pass_complete': np.random.random() < pass_comp,
                        'shot_outcome': '', 'xg': 0.0, 'player_name': ''})
                for _ in range(np.random.poisson(press_rate / 10)):
                    rows.append({'minute': minute, 'second': np.random.randint(0,60),
                        'event_type': 'Pressure', 'team_name': team,
                        'x': np.random.uniform(55,100), 'y': np.random.uniform(10,70),
                        'zone': 'Attacking Third', 'pass_complete': None,
                        'shot_outcome': '', 'xg': 0.0, 'player_name': ''})
            for i in range(max(1, int((end-start)*1.5/30))):
                minute = start + (end-start)*i // max(1, int((end-start)*1.5/30))
                xg_val = float(np.random.beta(2,8)*0.4)
                rows.append({'minute': minute, 'second': np.random.randint(0,60),
                    'event_type': 'Shot', 'team_name': team,
                    'x': np.random.uniform(88,116), 'y': np.random.uniform(25,55),
                    'zone': 'Attacking Third', 'pass_complete': None,
                    'shot_outcome': 'Saved' if np.random.random() > 0.5 else 'Off T',
                    'xg': xg_val, 'player_name': ''})

        add(home, 0, 44, 30*hp*2, p['home_press'], p['home_pass'], [0.15,0.35,0.50])
        add(away, 0, 44, 30*ap*2, p['away_press'], p['away_pass'], [0.35,0.40,0.25])
        rows.append({'minute':45,'second':0,'event_type':'Halftime','team_name':home,
            'x':np.nan,'y':np.nan,'zone':'Unknown','pass_complete':None,'shot_outcome':'','xg':0.0,'player_name':''})
        add(home, 45, 90, 30*hp*2, p['home_press'], p['home_pass'], [0.20,0.38,0.42])
        add(away, 45, 90, 30*ap*2, p['away_press'], p['away_pass'], [0.30,0.40,0.30])

        for (minute, team, player, xg_val) in p['goals']:
            rows.append({'minute': minute, 'second': np.random.randint(0,30),
                'event_type': 'Shot', 'team_name': team,
                'x': np.random.uniform(100,114), 'y': np.random.uniform(30,50),
                'zone': 'Attacking Third', 'pass_complete': None,
                'shot_outcome': 'Goal', 'xg': float(xg_val), 'player_name': player})

        for (minute, team, player) in p.get('subs', []):
            rows.append({'minute': minute, 'second': 0, 'event_type': 'Substitution',
                'team_name': team, 'x': np.nan, 'y': np.nan, 'zone': 'Unknown',
                'pass_complete': None, 'shot_outcome': '', 'xg': 0.0, 'player_name': player})

        df = pd.DataFrame(rows)
        return df.sort_values(['minute','second']).reset_index(drop=True)