"""Shared Cricsheet parsing and win-probability utilities for the ODI project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_BALLS = 300
MAX_WICKETS = 10


def project_data_dir() -> Path:
    """Resolve data directory whether notebooks run from repo root or project dir."""
    here = Path(__file__).parent / "data"
    if here.exists():
        return here
    alt = Path("projects/cricket-win-probability/data")
    return alt if alt.exists() else here


def load_cricsheet_json(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def match_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    info = doc.get("info", {})
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    if winner is None and outcome.get("result") == "tie":
        winner = "tie"
    elif winner is None and outcome.get("result") == "no result":
        winner = "no result"

    margin = outcome.get("by", {})
    dates = info.get("dates", [])
    return {
        "match_id": Path(info.get("match_type_number", "")).name if False else doc.get("meta", {}).get("revision", ""),
        "teams": teams,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "event": info.get("event"),
        "season": info.get("season"),
        "date": dates[0] if dates else None,
        "winner": winner,
        "margin": margin,
        "outcome": outcome,
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
    }


def _runs_from_delivery(delivery: dict[str, Any]) -> tuple[int, int, bool]:
    runs = delivery.get("runs", {})
    total = int(runs.get("total", 0))
    batter = int(runs.get("batter", 0))
    wicket = bool(delivery.get("wickets"))
    return total, batter, wicket


def balls_to_dataframe(doc: dict[str, Any]) -> pd.DataFrame:
    """Expand Cricsheet JSON to one row per legal delivery across all innings."""
    info = doc.get("info", {})
    teams = info.get("teams", ["Team A", "Team B"])
    rows: list[dict[str, Any]] = []
    innings_totals: list[int] = []

    for inn_idx, innings in enumerate(doc.get("innings", [])):
        batting_team = innings.get("team", teams[min(inn_idx, len(teams) - 1)])
        bowling_team = teams[1] if batting_team == teams[0] else teams[0]
        cumulative_runs = 0
        cumulative_wickets = 0
        ball_number = 0

        for over_block in innings.get("overs", []):
            over_num = over_block.get("over", 0)
            for delivery in over_block.get("deliveries", []):
                ball_number += 1
                runs_total, runs_batter, is_wicket = _runs_from_delivery(delivery)
                cumulative_runs += runs_total
                if is_wicket:
                    cumulative_wickets += 1

                rows.append(
                    {
                        "innings": inn_idx + 1,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over": over_num + 1,
                        "ball_in_over": len(
                            [d for d in over_block.get("deliveries", []) if d is delivery]
                        ),
                        "ball_number": ball_number,
                        "runs_total": runs_total,
                        "runs_batter": runs_batter,
                        "is_wicket": int(is_wicket),
                        "cumulative_runs": cumulative_runs,
                        "cumulative_wickets": cumulative_wickets,
                    }
                )

        innings_totals.append(cumulative_runs)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    first_innings_runs = innings_totals[0] if innings_totals else 0
    df["first_innings_runs"] = first_innings_runs
    df["target"] = first_innings_runs + 1
    df["match_id"] = doc.get("meta", {}).get("data_version", "unknown")
    meta = match_metadata(doc)
    for key, val in meta.items():
        if key not in df.columns:
            df[key] = val if not isinstance(val, (list, dict)) else str(val)
    return df


def chase_innings_states(
    chase_df: pd.DataFrame,
    target: int,
    batting_team_won: int,
    match_id: str = "unknown",
) -> pd.DataFrame:
    """Build one row per ball in the chase with modelling features."""
    records: list[dict[str, Any]] = []
    for _, row in chase_df.iterrows():
        balls_bowled = int(row["ball_number"])
        balls_remaining = max(MAX_BALLS - balls_bowled, 0)
        wickets_lost = int(row["cumulative_wickets"])
        wickets_remaining = MAX_WICKETS - wickets_lost
        runs_scored = int(row["cumulative_runs"])
        runs_required = max(target - runs_scored, 0)
        overs_completed = balls_bowled / 6.0
        rrr = runs_required / balls_remaining if balls_remaining > 0 else (0.0 if runs_required == 0 else 99.0)
        crr = runs_scored / balls_bowled * 6 if balls_bowled > 0 else 0.0
        over_num = int(row["over"])
        phase_powerplay = int(over_num <= 10)
        phase_death = int(over_num >= 40)
        phase_middle = int(not phase_powerplay and not phase_death)

        records.append(
            {
                "match_id": match_id,
                "ball_number": balls_bowled,
                "over": over_num,
                "runs_scored": runs_scored,
                "wickets_lost": wickets_lost,
                "balls_bowled": balls_bowled,
                "balls_remaining": balls_remaining,
                "wickets_remaining": wickets_remaining,
                "runs_required": runs_required,
                "runs_required_per_ball": rrr,
                "current_run_rate": crr,
                "target": target,
                "overs_completed": overs_completed,
                "phase_powerplay": phase_powerplay,
                "phase_middle": phase_middle,
                "phase_death": phase_death,
                "batting_team_won": batting_team_won,
            }
        )
    return pd.DataFrame(records)


def build_game_states(doc: dict[str, Any]) -> pd.DataFrame:
    """Return chase-innings ball states for a completed ODI."""
    df = balls_to_dataframe(doc)
    if df.empty or df["innings"].nunique() < 2:
        return pd.DataFrame()

    chase = df[df["innings"] == 2].copy()
    meta = match_metadata(doc)
    teams = meta["teams"]
    chase_team = chase["batting_team"].iloc[0]
    winner = meta.get("winner")
    if winner == "tie":
        batting_team_won = 0
    elif winner == chase_team:
        batting_team_won = 1
    else:
        batting_team_won = 0

    target = int(chase["target"].iloc[0])
    match_id = f"{meta.get('date', 'na')}_{teams[0]}_{teams[1]}".replace(" ", "_")
    return chase_innings_states(chase, target, batting_team_won, match_id)


# Simplified ODI resource table (educational approximation, not ICC-official).
# Rows: wickets lost (0-9), Cols: overs completed (0-50).
_RESOURCE_GRID = np.array(
    [
        [100, 99, 98, 96, 94, 92, 90, 88, 86, 84, 82],
        [94, 92, 90, 87, 84, 81, 78, 75, 72, 68, 65],
        [87, 84, 81, 77, 73, 69, 65, 61, 57, 53, 49],
        [79, 75, 71, 66, 61, 56, 51, 46, 41, 36, 31],
        [70, 65, 60, 54, 48, 42, 36, 30, 24, 18, 12],
        [60, 54, 48, 41, 34, 27, 20, 13, 6, 3, 1],
        [49, 42, 35, 27, 19, 11, 5, 2, 1, 0, 0],
        [37, 29, 21, 13, 6, 2, 1, 0, 0, 0, 0],
        [24, 16, 8, 3, 1, 0, 0, 0, 0, 0, 0],
        [10, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
_OVERS_BREAKS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])


def resource_remaining(overs_completed: float, wickets_lost: int) -> float:
    """Percentage of scoring resources remaining (0-100)."""
    wickets_lost = int(min(max(wickets_lost, 0), 9))
    overs_completed = float(min(max(overs_completed, 0), 50))
    col = int(np.searchsorted(_OVERS_BREAKS, overs_completed, side="right") - 1)
    col = int(np.clip(col, 0, len(_OVERS_BREAKS) - 1))
    return float(_RESOURCE_GRID[wickets_lost, col])


def dls_win_probability(states: pd.DataFrame, first_innings_runs: int) -> pd.Series:
    """
    Educational DLS-style win probability for the chasing team.

    Compares resource used by chaser vs resource used by defender at same stage.
    """
    target = first_innings_runs + 1
    probs = []
    for _, row in states.iterrows():
        overs = row["overs_completed"]
        wickets = int(row["wickets_lost"])
        runs = row["runs_scored"]
        rem = resource_remaining(overs, wickets)
        # Defender used (100 - rem) proportionally; par score from resources consumed.
        used = 100.0 - rem
        par = first_innings_runs * used / 100.0
        if rem <= 0.01:
            prob = 1.0 if runs >= target else 0.0
        else:
            # Logistic transition around par score with scale tied to remaining resource.
            scale = max(first_innings_runs * rem / 400.0, 1.0)
            prob = 1.0 / (1.0 + np.exp(-(runs - par) / scale))
        probs.append(prob)
    return pd.Series(probs, index=states.index).clip(0.0, 1.0)


def cumulative_runs_by_innings(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-format cumulative runs for plotting."""
    out = []
    for inn, grp in df.groupby("innings"):
        out.append(
            grp.assign(
                innings_label=grp["batting_team"],
                x=grp["ball_number"] / 6.0,
            )[["x", "cumulative_runs", "innings_label", "innings"]]
        )
    return pd.concat(out, ignore_index=True)
