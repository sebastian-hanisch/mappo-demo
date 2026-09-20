"""Kennzahlen aus einem Protokoll-Lauf, plus der Vergleich gegen die zentrale
OR-Tools-CP-SAT-Referenz (was ein Planer mit vollständiger Information erreicht
hätte)."""

from cn_ortools_reference import solve_with_ortools


def stats_up_to_step(result, step):
    """Kennzahlen für die Schritt-Animation: Zustand nach den ersten `step + 1`
    Auftragsvergaben."""
    visited = result.steps[: step + 1]
    if not visited:
        return {"jobs_awarded": 0, "worst_agent_free_time": 0.0}
    last = visited[-1]
    return {
        "jobs_awarded": len(visited),
        "worst_agent_free_time": max(last.agent_free_times_after),
    }


def comparison(instance, protocol_result, time_limit_seconds=10.0):
    ortools_result = solve_with_ortools(instance, time_limit_seconds=time_limit_seconds)
    cnp_makespan = protocol_result.makespan
    ortools_makespan = ortools_result.makespan if ortools_result.feasible else None

    gap_pct = None
    if ortools_result.feasible and ortools_makespan > 0:
        gap_pct = (cnp_makespan - ortools_makespan) / ortools_makespan * 100.0

    return {
        "cnp_makespan": cnp_makespan,
        "ortools_makespan": ortools_makespan,
        "ortools_feasible": ortools_result.feasible,
        "ortools_optimal": ortools_result.optimal,
        "ortools_wall_time": ortools_result.wall_time_ms / 1000.0,
        "gap_pct": gap_pct,
    }
