"""Build bundled sample matches and training CSV for the cricket win-probability project."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from cricsheet_utils import build_game_states, dls_win_probability, project_data_dir

PROJ = Path(__file__).parent
DATA = PROJ / "data"
SAMPLE = DATA / "sample_matches"
RNG = random.Random(42)
NP = np.random.default_rng(42)
MAX_WICKETS = 10


def _delivery(runs: int, wicket: bool = False) -> dict:
    d = {
        "batter": "Batter",
        "bowler": "Bowler",
        "non_striker": "NonStriker",
        "runs": {"batter": runs, "extras": 0, "total": runs},
    }
    if wicket:
        d["wickets"] = [{"player_out": "Batter", "kind": "bowled"}]
    return d


def _innings(team: str, run_sequence: list[int], wicket_balls: set[int]) -> dict:
    overs = []
    ball = 0
    idx = 0
    for over in range(50):
        deliveries = []
        for _ in range(6):
            ball += 1
            runs = run_sequence[idx] if idx < len(run_sequence) else 0
            idx += 1
            deliveries.append(_delivery(runs, wicket=ball in wicket_balls))
        overs.append({"over": over, "deliveries": deliveries})
    return {"team": team, "overs": overs}


def _match(
    teams: tuple[str, str],
    date: str,
    event: str,
    winner: str,
    first_runs: list[int],
    second_runs: list[int],
    first_wickets: set[int],
    second_wickets: set[int],
    outcome_by: dict | None = None,
) -> dict:
    outcome = {"winner": winner}
    if outcome_by:
        outcome["by"] = outcome_by
    return {
        "meta": {"data_version": "1.1.0", "created": date, "revision": 1},
        "info": {
            "balls_per_over": 6,
            "dates": [date],
            "event": event,
            "gender": "male",
            "match_type": "ODI",
            "teams": list(teams),
            "venue": "Sample Ground",
            "city": "Sample City",
            "season": date[:4],
            "outcome": outcome,
        },
        "innings": [
            _innings(teams[0], first_runs, first_wickets),
            _innings(teams[1], second_runs, second_wickets),
        ],
    }


def _random_run_sequence(n: int, mean: float, rng: np.random.Generator) -> list[int]:
    probs = [0.55, 0.22, 0.12, 0.06, 0.03, 0.015, 0.005]
    values = [0, 1, 2, 3, 4, 6, 1]
    seq = []
    for _ in range(n):
        r = rng.choice(values, p=probs)
        if rng.random() < 0.02:
            r = 4
        seq.append(int(r))
    # Scale to approximate target mean run rate
    current = sum(seq) / max(n / 6, 1) * 6
    if current < mean * 0.85:
        for i in rng.choice(n, size=min(20, n), replace=False):
            seq[i] = min(seq[i] + 1, 6)
    return seq


def _simulate_odi(rng: np.random.Generator) -> dict:
    teams = ("Team A", "Team B")
    first_wickets = set(rng.choice(300, size=int(rng.integers(5, 9)), replace=False) + 1)
    first_runs = _random_run_sequence(300, mean=5.2, rng=rng)
    first_total = sum(first_runs[:300])  # noqa: F841 — used implicitly via sequence

    chase_success = rng.random() < 0.48
    target_margin = int(rng.integers(1, 40))
    if chase_success:
        second_total_target = first_total + 1 - target_margin
    else:
        second_total_target = first_total - target_margin

    second_runs = _random_run_sequence(300, mean=5.0, rng=rng)
    scale = second_total_target / max(sum(second_runs), 1)
    second_runs = [min(int(round(r * scale)), 6) for r in second_runs]
    second_wickets = set(rng.choice(280, size=int(rng.integers(4, 9)), replace=False) + 1)

    if chase_success:
        winner = teams[1]
        outcome_by = {"wickets": MAX_WICKETS - len(second_wickets)}
    else:
        winner = teams[0]
        outcome_by = {"runs": abs(second_total_target - first_total)}

    return _match(
        teams=teams,
        date=f"20{int(rng.integers(15, 25))}-{int(rng.integers(1,12)):02d}-{int(rng.integers(1,28)):02d}",
        event="Simulated ODI",
        winner=winner,
        first_runs=first_runs,
        second_runs=second_runs,
        first_wickets=first_wickets,
        second_wickets=second_wickets,
        outcome_by=outcome_by,
    )


def build_case_study_matches() -> None:
    SAMPLE.mkdir(parents=True, exist_ok=True)

    # Approximate 2019 World Cup Final chase profile (NZ 241, ENG chase tied)
    wc2019 = _match(
        teams=("New Zealand", "England"),
        date="2019-07-14",
        event="ICC Cricket World Cup Final",
        winner="England",
        first_runs=_random_run_sequence(300, 4.8, NP)[:240] + [1],
        second_runs=_random_run_sequence(300, 4.9, NP),
        first_wickets=set([45, 89, 134, 178, 210, 238]),
        second_wickets=set([52, 98, 145, 192, 236, 271, 285]),
        outcome_by={"wickets": 0},
    )
    # Force near-tie totals
    wc2019["innings"][0]["overs"] = wc2019["innings"][0]["overs"][:42]
    wc2019["innings"][1]["overs"] = wc2019["innings"][1]["overs"][:50]

    india_aus_qf = _match(
        teams=("Australia", "India"),
        date="2011-03-24",
        event="ICC Cricket World Cup Quarter-Final",
        winner="India",
        first_runs=_random_run_sequence(300, 5.4, NP),
        second_runs=_random_run_sequence(300, 5.6, NP),
        first_wickets=set([38, 95, 150, 205, 260]),
        second_wickets=set([60, 120, 180]),
        outcome_by={"wickets": 5},
    )

    upset = _match(
        teams=("England", "Ireland"),
        date="2011-03-02",
        event="ICC Cricket World Cup",
        winner="Ireland",
        first_runs=_random_run_sequence(300, 5.8, NP),
        second_runs=_random_run_sequence(300, 6.0, NP),
        first_wickets=set([40, 88, 130, 170, 210, 250]),
        second_wickets=set([55, 115, 175, 230]),
        outcome_by={"wickets": 3},
    )

    files = {
        "2019_wc_final_eng_nz.json": wc2019,
        "2011_wc_qf_ind_aus.json": india_aus_qf,
        "2011_wc_eng_ire_upset.json": upset,
    }
    for name, doc in files.items():
        (SAMPLE / name).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"Wrote {name}")


def build_training_sample(n_matches: int = 400) -> None:
    rows = []
    for i in range(n_matches):
        doc = _simulate_odi(NP)
        doc["meta"]["revision"] = i
        states = build_game_states(doc)
        if states.empty:
            continue
        first_runs = int(states["target"].iloc[0] - 1)
        states = states.copy()
        states["dls_prob"] = dls_win_probability(states, first_runs)
        states["match_idx"] = i
        rows.append(states)

    training = pd.concat(rows, ignore_index=True)
    out = DATA / "odi_training_sample.csv"
    training.to_csv(out, index=False)
    print(f"Wrote {out} ({len(training):,} rows from {training['match_idx'].nunique()} matches)")


def main() -> None:
    build_case_study_matches()
    build_training_sample()
    print("Done.")


if __name__ == "__main__":
    main()
