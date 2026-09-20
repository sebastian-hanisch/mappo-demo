"""Exakte Referenzen zu dem, was JEDE Policy dieser Demo erreichen kann. Alle Policies halten die
Ankündigungs-Reihenfolge der Aufträge fest (Contract Nets Regel) und wählen nur, WELCHER Agent
welchen Auftrag bekommt - deshalb ist der beste Makespan über alle Zuteilungen in dieser
festen Reihenfolge (`fixed_order_ceiling`) eine harte Obergrenze für jede gelernte Policy:

    jede Policy >= fixed_order_ceiling >= CP-SAT-Optimum (- Rundung)

Außerdem: die vollständige Enumeration aller gemeinsamen Policies für eine winzige Instanz
(`enumerate_joint_policies`) - der exakte Beleg für Nicht-Stationarität (siehe README)."""

from itertools import product

import cn_constants as C
from cn_scenario import Instance, Job


def fixed_order_ceiling(instance):
    """Bester Makespan über alle Zuteilungen (Auftrag -> Agent), Aufträge je Agent in
    Ankündigungsreihenfolge. Tiefensuche mit Schranke."""
    k = instance.n_agents
    best = [float("inf")]

    def dfs(j, positions, free):
        current = max(free)
        if current >= best[0]:
            return
        if j == instance.n_jobs:
            best[0] = current
            return
        job = instance.jobs[j]
        for a in range(k):
            finish = free[a] + instance.travel_time(positions[a], job.position) + job.duration
            new_positions = list(positions)
            new_free = list(free)
            new_positions[a] = job.position
            new_free[a] = finish
            dfs(j + 1, new_positions, new_free)

    dfs(0, list(instance.agent_start_positions), [0.0] * k)
    return best[0]


def joint_dp_optimum(instance):
    """Dasselbe Optimum als memoisierte Rekursion über den Zustand (Auftrag, Positionen, Freizeiten)
    - unabhängig von `fixed_order_ceiling` implementiert, dient als Gegenprobe im Test."""
    k = instance.n_agents
    memo = {}

    def solve(j, positions, free):
        if j == instance.n_jobs:
            return max(free)
        key = (j, positions, free)
        if key in memo:
            return memo[key]
        job = instance.jobs[j]
        best = float("inf")
        for a in range(k):
            finish = free[a] + instance.travel_time(positions[a], job.position) + job.duration
            new_positions = positions[:a] + (job.position,) + positions[a + 1:]
            new_free = free[:a] + (finish,) + free[a + 1:]
            best = min(best, solve(j + 1, new_positions, new_free))
        memo[key] = best
        return best

    return solve(0, tuple(instance.agent_start_positions), (0.0,) * k)


def joint_policy_makespan(instance, policy_table, biases=C.BIAS_ACTIONS):
    """policy_table[agent][job_index] = Aktions-Index. Makespan dieser gemeinsamen Policy."""
    k = instance.n_agents
    positions = list(instance.agent_start_positions)
    free = [0.0] * k
    for job in instance.jobs:
        best_key, best_agent, best_finish = None, 0, 0.0
        for a in range(k):
            finish = free[a] + instance.travel_time(positions[a], job.position) + job.duration
            key = (finish + biases[policy_table[a][job.index]], a)
            if best_key is None or key < best_key:
                best_key, best_agent, best_finish = key, a, finish
        positions[best_agent] = job.position
        free[best_agent] = best_finish
    return max(free)


def enumerate_joint_policies(instance, biases=C.BIAS_ACTIONS):
    """ALLE gemeinsamen Policies, bei denen jeder Agent pro Auftrag (nicht pro Zustand!) eine
    feste Aktion wählt: len(biases) ** (n_agents * n_jobs) Stück. Gibt eine Liste
    (policy_table, makespan) zurück - nur für winzige Instanzen gedacht."""
    k, n = instance.n_agents, instance.n_jobs
    results = []
    for flat in product(range(len(biases)), repeat=k * n):
        table = tuple(tuple(flat[a * n:(a + 1) * n]) for a in range(k))
        results.append((table, joint_policy_makespan(instance, table, biases)))
    return results


def miniature_instance():
    """Die handrechenbare 2x2-Miniatur (2 Aufträge, 2 Agenten, Anfahrt 1.0/Einheit):
    Aufträge bei Position 5 (Dauer 5) und 0 (Dauer 10), Agenten starten bei 5 und 15.
    Contract Net = 20, Optimum = 15 (Agent 0 nimmt Auftrag 2, Agent 1 nimmt Auftrag 1)."""
    return Instance(
        n_jobs=2, n_agents=2,
        jobs=(Job(index=0, position=5.0, duration=5.0), Job(index=1, position=0.0, duration=10.0)),
        agent_start_positions=(5.0, 15.0), travel_time_per_unit=1.0,
    )
