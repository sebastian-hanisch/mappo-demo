"""Unabhängige Orakel für den MARL-Kern: exakt rationale Simulation der Dispatch-Schleife, Vollenumeration aller
Zuteilungen (Decke/DP), Aufzählung der 81 gemeinsamen Policies der 2x2-Miniatur, eine eigene Neuimplementierung
des Q-Lernens mit denselben Zufallsströmen (Tabellen müssen exakt übereinstimmen) und Cross-Play über den Vehikel-Pfad."""
import itertools
import math
import random
import statistics
from fractions import Fraction as F

import cn_constants as C
import marl_env as me
import marl_evaluation as mev
import marl_exact as mx
import marl_iql as mi
import marl_obs as mo
from cn_scenario import generate_instance

BIAS = [F(b) for b in C.BIAS_ACTIONS]


def _sim(inst, policy_fn):
    k = inst.n_agents
    pos = [F(p) for p in inst.agent_start_positions]
    free = [F(0)] * k
    owner = []
    for j in inst.jobs:
        best = None
        for a in range(k):
            travel = abs(pos[a] - F(j.position)) * F(inst.travel_time_per_unit)
            act = policy_fn(a, j.index)
            fin = free[a] + travel + F(j.duration)
            key = (fin + BIAS[act], a)
            if best is None or key < best[0]:
                best = (key, a, fin)
        _, w, fin = best
        pos[w], free[w] = F(j.position), fin
        owner.append(w)
    return max(free), owner


def test_kernel_and_vehicle_path_match_rational_simulation():
    rng = random.Random(9)
    for _ in range(60):
        n, k = rng.randint(1, 7), rng.randint(1, 4)
        inst = generate_instance(n, k, rng.choice([0, 0.3, 1.0]), rng.choice([0.2, 1.0, 2.0]), rng.randrange(10 ** 5))
        table = {(a, j): rng.randrange(3) for a in range(k) for j in range(n)}
        ms_ref, owner = _sim(inst, lambda a, j: table[(a, j)])
        obs_fn, _ = mo.make_observer(n, k)
        per_job = mo.n_free_buckets() * mo.n_travel_buckets()

        def policy(a, s):
            return table[(a, s // per_job)]

        assert abs(me.rollout(me.to_fast(inst), policy, obs_fn) - float(ms_ref)) < 1e-9
        disp = me.run_dispatch(inst, policy, obs_fn)
        assert abs(disp.protocol_result.makespan - float(ms_ref)) < 1e-9
        assert [disp.protocol_result.assignment[j] for j in range(n)] == owner


def test_ceiling_and_dp_match_full_enumeration_of_assignments():
    rng = random.Random(5)
    for _ in range(40):
        n, k = rng.randint(1, 6), rng.randint(1, 3)
        inst = generate_instance(n, k, rng.choice([0, 0.3, 1.0]), rng.choice([0.2, 1.0, 2.0]), rng.randrange(10 ** 5))
        best = math.inf
        for owner in itertools.product(range(k), repeat=n):
            pos = [F(p) for p in inst.agent_start_positions]
            free = [F(0)] * k
            for j, a in zip(inst.jobs, owner):
                free[a] += abs(pos[a] - F(j.position)) * F(inst.travel_time_per_unit) + F(j.duration)
                pos[a] = F(j.position)
            best = min(best, float(max(free)))
        assert abs(mx.fixed_order_ceiling(inst) - best) < 1e-9
        assert abs(mx.joint_dp_optimum(inst) - best) < 1e-9


def test_miniature_81_policies_by_independent_enumeration():
    inst = mx.miniature_instance()
    res = {}
    for flat in itertools.product(range(3), repeat=4):
        tab = ((flat[0], flat[1]), (flat[2], flat[3]))
        res[tab] = _sim(inst, lambda a, j, tab=tab: tab[a][j])[0]
    assert len(res) == 81 and min(res.values()) == 15
    assert sum(1 for m in res.values() if m == 15) == 18
    assert _sim(inst, lambda a, j: 0)[0] == 20


def _train_reference(inst, sigma, episodes, seed):
    n, k = inst.n_jobs, inst.n_agents
    obs_fn, n_states = mo.make_observer(n, k)
    q = [[[0.0] * 3 for _ in range(n_states)] for _ in range(k)]
    env, expl = random.Random(f"iql-env-{seed}"), random.Random(f"iql-explore-{seed}")
    for ep in range(episodes):
        dur = [j.duration * math.exp(env.gauss(0.0, sigma)) if sigma > 0 else j.duration for j in inst.jobs]
        eps = max(C.EPSILON_END, 1.0 - (1.0 - C.EPSILON_END) * ep / (C.EPSILON_DECAY_FRACTION * episodes))
        pos, free, traj = list(inst.agent_start_positions), [0.0] * k, []
        for j in range(n):
            best, sts, acts = None, [], []
            for a in range(k):
                travel = abs(pos[a] - inst.jobs[j].position) * inst.travel_time_per_unit
                s = obs_fn(free[a], travel, j)
                if expl.random() < eps:
                    act = expl.randrange(3)
                else:
                    act = max(range(3), key=lambda i: (q[a][s][i], -i))
                fin = free[a] + travel + dur[j]
                key = (fin + C.BIAS_ACTIONS[act], a)
                if best is None or key < best[0]:
                    best = (key, a, fin)
                sts.append(s)
                acts.append(act)
            _, w, fin = best
            pos[w], free[w] = inst.jobs[j].position, fin
            traj.append((sts, acts))
        reward = -max(free) / C.REWARD_SCALE
        for t in range(n - 1, -1, -1):
            for a in range(k):
                s, ac = traj[t][0][a], traj[t][1][a]
                target = reward if t == n - 1 else max(q[a][traj[t + 1][0][a]])
                q[a][s][ac] += C.ALPHA * (target - q[a][s][ac])
    return q


def test_iql_training_matches_independent_reimplementation_exactly():
    for n, k, seed, sigma, episodes, train_seed in [(5, 2, 4, 0.3, 300, 1), (8, 3, 8, 0.3, 200, 3), (3, 3, 2, 0.0, 200, 5)]:
        inst = generate_instance(n, k, 0.3, 1.0, seed)
        ref = _train_reference(inst, sigma, episodes, train_seed)
        mine = mi.train_iql(inst, C.ENV_RECURRING, sigma, episodes, train_seed, 0.3).q
        diff = max(abs(x - y) for a in range(k) for ra, rb in zip(ref[a], mine[a]) for x, y in zip(ra, rb))
        assert diff < 1e-12


def test_crossplay_matrix_matches_vehicle_path_composition():
    n, k = 5, 3
    inst = generate_instance(n, k, 0.3, 1.0, 8)
    obs_fn, _ = mo.make_observer(n, k)
    qs = [mi.train_iql(inst, C.ENV_RECURRING, 0.3, 400, s, 0.3).q for s in (1, 2)]
    held = me.heldout_instances(inst, C.ENV_RECURRING, 0.3, 0.3, 8, n=6)
    matrix = mev.crossplay(qs, held, n, k)
    for a in range(2):
        for b in range(2):
            def policy(agent, s, a=a, b=b):
                return mi.greedy_action((qs[a] if agent == 0 else qs[b])[agent][s])
            ref = statistics.fmean(me.run_dispatch(i, policy, obs_fn).protocol_result.makespan for i in held)
            assert abs(ref - matrix[a][b]) < 1e-9
