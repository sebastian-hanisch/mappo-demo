import random

import cn_constants as C
from cn_bruteforce import solve_bruteforce
from cn_ortools_reference import SCALE, solve_with_ortools
from cn_protocol import run_protocol
from cn_scenario import generate_instance
from marl_env import rollout, to_fast
from marl_exact import (
    enumerate_joint_policies, fixed_order_ceiling, joint_dp_optimum, joint_policy_makespan, miniature_instance,
)
from marl_iql import greedy_policy, new_q_tables
from marl_obs import make_observer


def test_miniature_hand_computed_values():
    instance = miniature_instance()
    assert run_protocol(instance).makespan == 20.0
    assert fixed_order_ceiling(instance) == 15.0
    assert joint_dp_optimum(instance) == 15.0


def test_miniature_joint_policy_enumeration():
    instance = miniature_instance()
    results = enumerate_joint_policies(instance)
    assert len(results) == 3 ** 4 == 81
    assert min(ms for _, ms in results) == 15.0
    assert sum(1 for _, ms in results if ms == 15.0) == 18


def test_miniature_non_stationarity_exact_fact():
    """Zwei optimale Konventionen A und B sind einzeln optimal, ihre Mischung nicht - der exakte
    Beleg für Nicht-Stationarität: die beste Antwort eines Agenten hängt von den Policies der anderen ab."""
    instance = miniature_instance()
    normal, up, down = 0, 1, 2
    # Policy A: Agent 0 normal/normal, Agent 1 greift Auftrag 1 (down), dann normal
    a = ((normal, normal), (down, normal))
    # Policy B: Agent 0 lehnt Auftrag 1 ab (up), dann normal; Agent 1 normal/normal
    b = ((up, normal), (normal, normal))
    assert joint_policy_makespan(instance, a) == 15.0
    assert joint_policy_makespan(instance, b) == 15.0
    mixed = (a[0], b[1])  # A's Agent 0 + B's Agent 1
    assert joint_policy_makespan(instance, mixed) == 20.0  # = Contract Net
    # "normal/normal" für Agent 0 ist keine beste Antwort gegen einen normal bietenden Agent 1 ...
    all_normal = ((normal, normal), (normal, normal))
    assert joint_policy_makespan(instance, all_normal) == 20.0
    best_vs_normal_partner = min(
        joint_policy_makespan(instance, ((x, y), (normal, normal))) for x in range(3) for y in range(3)
    )
    assert best_vs_normal_partner < 20.0
    # ... aber gegen A's Agent 1 (greift Auftrag 1) ist sie eine
    best_vs_a_partner = min(
        joint_policy_makespan(instance, ((x, y), a[1])) for x in range(3) for y in range(3)
    )
    assert joint_policy_makespan(instance, (a[0], a[1])) == best_vs_a_partner == 15.0


def test_ceiling_equals_dp_and_bruteforce_ordering():
    for seed in range(30):
        instance = generate_instance(2 + seed % 5, 2 + seed % 2, 0.4, 1.0, seed)
        ceiling = fixed_order_ceiling(instance)
        assert abs(ceiling - joint_dp_optimum(instance)) < 1e-9, f"seed={seed}"
        # Bruteforce darf zusätzlich die Reihenfolge innerhalb eines Agenten wählen => nie schlechter
        assert solve_bruteforce(instance)[0] <= ceiling + 1e-9, f"seed={seed}"


def test_dominance_any_policy_ge_ceiling_ge_cpsat_optimum():
    tolerance_per_job = 2.0 / SCALE
    for seed in range(300):
        n_jobs, n_agents = 2 + seed % 7, 2 + seed % 3
        instance = generate_instance(n_jobs, n_agents, 0.4, 1.0, seed)
        obs_fn, n_states = make_observer(n_jobs, n_agents)
        rng = random.Random(seed)
        q = new_q_tables(n_agents, n_states)
        for agent_table in q:
            for row in agent_table:
                for i in range(len(row)):
                    row[i] = rng.uniform(-3, 0)
        policy_ms = rollout(to_fast(instance), greedy_policy(q), obs_fn)
        ceiling = fixed_order_ceiling(instance)
        assert policy_ms >= ceiling - 1e-9, f"seed={seed}"
        if seed % 10 == 0:  # CP-SAT ist teurer - jede 10. Instanz reicht
            optimum = solve_with_ortools(instance, time_limit_seconds=5.0).makespan
            assert ceiling >= optimum - tolerance_per_job * n_jobs, f"seed={seed}"
