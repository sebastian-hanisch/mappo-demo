"""Exakter Referenzlöser (Google OR-Tools CP-SAT) - zeigt, was ein zentraler Planer
mit vollständiger Information über ALLE Aufträge im Voraus erreichen würde, im
Gegensatz zum dezentralen Contract Net Protocol, das Aufträge nur nacheinander
kennt und nie revidiert.

Modell bewusst viel einfacher als quaycrane-demo's quaycrane_cp_solver.py: hier
gibt es KEINE gemeinsame Schiene und keine Non-Crossing-Regel zwischen Agenten
(siehe cn_scenario.py) - nur Sequencing INNERHALB eines Agenten (ein Agent kann
nicht zwei Aufträge gleichzeitig bearbeiten, plus Fahrzeit zwischen ihnen).
Zwischen verschiedenen Agenten gibt es überhaupt keine Interaktions-Constraint.

Minimiert wird primär der Makespan; lexikografisches Tie-Breaking über die Summe
aller Endzeiten verhindert, dass der Solver unter mehreren gleich-optimalen
Lösungen eine mit unnötigem Leerlauf zurückgibt (derselbe Fund/dieselbe Formel
wie in quaycrane-demo, siehe feedback_cp_sat_lexicographic_tiebreak.md)."""

import math
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

SCALE = 10  # interne Zeitauflösung: 1 CP-SAT-Einheit = 0.1 Minuten


def _scaled(x):
    # Aufrunden statt runden: das Modell darf nie eine kürzere Zeit annehmen, als real
    # gebraucht wird (siehe quaycrane_cp_solver.scaled_fn für den konkreten Rundungsfund).
    return math.ceil(x * SCALE - 1e-6)


@dataclass
class ExactResult:
    feasible: bool
    optimal: bool
    assignment: dict  # job_index -> agent_id
    starts: dict  # job_index -> Startzeit (unskaliert)
    makespan: float
    wall_time_ms: float


def build_model(instance):
    """Baut das reine Constraint-Modell (ohne Zielfunktion/Solver-Aufruf) - separat
    aufrufbar für Tests, die gezielt Variablen fixieren wollen."""
    n, k = instance.n_jobs, instance.n_agents
    model = cp_model.CpModel()

    durations = [_scaled(job.duration) for job in instance.jobs]
    total_work = sum(durations)
    max_travel = _scaled(instance.travel_time_per_unit * 2 * (n + 1))
    horizon = total_work + n * max_travel + 1

    x = {(j, a): model.NewBoolVar(f"x_{j}_{a}") for j in range(n) for a in range(k)}
    for j in range(n):
        model.AddExactlyOne(x[j, a] for a in range(k))

    start = [model.NewIntVar(0, horizon, f"start_{j}") for j in range(n)]
    end = [model.NewIntVar(0, horizon, f"end_{j}") for j in range(n)]
    for j in range(n):
        model.Add(end[j] == start[j] + durations[j])
        for a in range(k):
            travel0 = _scaled(instance.travel_time(instance.agent_start_positions[a], instance.jobs[j].position))
            model.Add(start[j] >= travel0).OnlyEnforceIf(x[j, a])

    for i in range(n):
        for j in range(i + 1, n):
            same = model.NewBoolVar(f"same_{i}_{j}")
            both_a = [model.NewBoolVar(f"both_{i}_{j}_{a}") for a in range(k)]
            for a in range(k):
                model.AddBoolAnd([x[i, a], x[j, a]]).OnlyEnforceIf(both_a[a])
                model.AddBoolOr([x[i, a].Not(), x[j, a].Not()]).OnlyEnforceIf(both_a[a].Not())
            model.AddBoolOr(both_a).OnlyEnforceIf(same)
            model.AddBoolAnd([b.Not() for b in both_a]).OnlyEnforceIf(same.Not())

            i_before_j = model.NewBoolVar(f"ibj_{i}_{j}")
            j_before_i = model.NewBoolVar(f"jbi_{i}_{j}")
            travel_ij = _scaled(instance.travel_time(instance.jobs[i].position, instance.jobs[j].position))
            model.Add(end[i] + travel_ij <= start[j]).OnlyEnforceIf([same, i_before_j])
            model.Add(end[j] + travel_ij <= start[i]).OnlyEnforceIf([same, j_before_i])
            model.AddBoolOr([i_before_j, j_before_i]).OnlyEnforceIf(same)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, end)
    return model, x, start, end, makespan, horizon


def solve_with_ortools(instance, time_limit_seconds=10.0):
    t0 = time.perf_counter()
    n, k = instance.n_jobs, instance.n_agents
    model, x, start, end, makespan, horizon = build_model(instance)

    tie_break_weight = n * horizon + 1
    model.Minimize(makespan * tie_break_weight + sum(end))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1  # deterministisch, fairer Vergleich
    status = solver.Solve(model)
    wall_time_ms = (time.perf_counter() - t0) * 1000

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactResult(
            feasible=False, optimal=False, assignment={}, starts={}, makespan=0.0, wall_time_ms=wall_time_ms
        )

    assignment = {}
    starts = {}
    for j in range(n):
        a = next(a for a in range(k) if solver.Value(x[j, a]))
        assignment[j] = a
        starts[j] = solver.Value(start[j]) / SCALE

    return ExactResult(
        feasible=True,
        optimal=status == cp_model.OPTIMAL,
        assignment=assignment,
        starts=starts,
        makespan=solver.Value(makespan) / SCALE,
        wall_time_ms=wall_time_ms,
    )
