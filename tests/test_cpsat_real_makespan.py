"""Regression: die CP-SAT-Referenz meldet den exakt nachgerechneten Makespan, nicht den Wert des aufgerundeten Rastermodells.

Das Modell rechnet auf einem 0,1-Minuten-Raster, bei dem jede Dauer und Anfahrt AUFGERUNDET wird. Sein Zielwert lag dadurch
bis über 1 % über dem, was dieselbe Zuteilung real braucht, und ein Contract Net, das real besser ist, wirkte als „schlägt das
zentrale Optimum“ (Seed 9 und 10 bei n = 6, k = 2, Variabilität 0)."""
import itertools
import math
import random

from cn_ortools_reference import solve_with_ortools
from cn_protocol import run_protocol
from cn_scenario import generate_instance


def _exact_makespan(inst):
    """Unabhängige exakte Referenz: Teilmengen-DP, Route je Teilmenge per Permutation (ungerundete Zeiten)."""
    n, k, tau = inst.n_jobs, inst.n_agents, inst.travel_time_per_unit
    cost = [[0.0] * (1 << n) for _ in range(k)]
    for a in range(k):
        for subset in range(1, 1 << n):
            jobs = [j for j in range(n) if subset >> j & 1]
            best = math.inf
            for perm in itertools.permutations(jobs):
                pos, t = inst.agent_start_positions[a], 0.0
                for j in perm:
                    t += abs(pos - inst.jobs[j].position) * tau + inst.jobs[j].duration
                    pos = inst.jobs[j].position
                best = min(best, t)
            cost[a][subset] = best
    full = (1 << n) - 1
    f = [0.0] + [math.inf] * full
    for a in range(k):
        g = [math.inf] * (full + 1)
        for t in range(full + 1):
            s = t
            while True:
                g[t] = min(g[t], max(f[t ^ s], cost[a][s]))
                if s == 0:
                    break
                s = (s - 1) & t
        f = g
    return f[full]


def test_contract_net_is_not_shown_better_than_central_reference():
    for seed in (9, 10):
        inst = generate_instance(6, 2, 0.0, 1.0, seed)
        res = solve_with_ortools(inst, time_limit_seconds=5.0)
        assert run_protocol(inst).makespan >= res.makespan - 1e-9, seed
        assert res.model_makespan >= res.makespan - 1e-9   # Rasterwert nie unter dem realen Makespan


def test_reported_makespan_is_never_below_and_close_to_exact_optimum():
    rng = random.Random(2)
    for _ in range(15):
        n, k = rng.randint(4, 5), rng.randint(2, 3)
        inst = generate_instance(n, k, rng.choice([0.0, 0.5]), rng.choice([0.5, 1.0, 2.0]), rng.randint(0, 999))
        exact = _exact_makespan(inst)
        res = solve_with_ortools(inst, time_limit_seconds=5.0)
        assert res.feasible and res.optimal
        assert res.makespan >= exact - 1e-9
        assert res.makespan <= exact * 1.01
        assert run_protocol(inst).makespan >= exact - 1e-9
