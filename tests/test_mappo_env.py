import numpy as np

import cn_constants as C
from cn_protocol import run_protocol
from cn_scenario import generate_instance
from marl_env import heldout_instances, rollout, to_fast
from marl_iql import greedy_policy
from marl_obs import make_observer
from mappo_env import (
    Sampler, arrays_from_instances, cnp_makespans, geometry, global_bucket, local_bucket, n_global_buckets,
    n_local_buckets, replay_dispatch, simulate,
)
from mappo_ppo import greedy_act_fn, train_ppo, trace_policy


def test_vector_kernel_zero_bias_equals_run_protocol():
    for seed in range(300):
        instance = generate_instance(1 + seed % 10, 1 + seed % 4, 0.5, 0.2 + (seed % 5) * 0.4, seed)
        pos, dur = arrays_from_instances([instance])
        assert cnp_makespans(geometry(instance), pos, dur)[0] == run_protocol(instance).makespan, f"seed={seed}"


def test_vector_kernel_equals_marl_kernel_for_random_policies_bitwise():
    rng = np.random.default_rng(0)
    for seed in range(200):
        n_jobs, n_agents = 3 + seed % 7, 2 + seed % 3
        instance = generate_instance(n_jobs, n_agents, 0.4, 0.5 + (seed % 4) * 0.4, seed)
        geom = geometry(instance)
        obs_fn, n_states = make_observer(n_jobs, n_agents)
        q = rng.normal(size=(n_agents, n_states, 3)).tolist()
        pos, dur = arrays_from_instances([instance])
        z = np.array(q)

        def act_fn(j, free, apos, travel, jp, jd):
            return z[np.arange(n_agents)[None, :], local_bucket(geom, j, free, travel)].argmax(-1)

        vec = simulate(geom, pos, dur, act_fn)[0]
        scalar = rollout(to_fast(instance), greedy_policy(q), obs_fn)
        assert vec == scalar, f"seed={seed}"


def test_local_bucket_matches_observer():
    instance = generate_instance(8, 3, 0.3, 1.0, 1)
    geom = geometry(instance)
    obs_fn, n_states = make_observer(8, 3)
    assert n_local_buckets(geom) == n_states
    rng = np.random.default_rng(1)
    free = rng.uniform(0, 60, size=(50, 3))
    travel = rng.uniform(0, 20, size=(50, 3))
    for j in range(8):
        vec = local_bucket(geom, j, free, travel)
        for b in range(50):
            for a in range(3):
                assert vec[b, a] == obs_fn(free[b, a], travel[b, a], j)


def test_global_bucket_range_and_uniqueness():
    geom = geometry(generate_instance(4, 2, 0.3, 1.0, 1))
    assert n_global_buckets(geom) == 4 * 4 ** 2
    seen = set()
    edges = [0.0, geom.free_edges[0], geom.free_edges[1], geom.free_edges[2]]
    for j in range(4):
        for f0 in edges:
            for f1 in edges:
                state = global_bucket(geom, j, np.array([[f0 + 0.01, f1 + 0.01]]))[0]
                assert 0 <= state < n_global_buckets(geom)
                seen.add(state)
    assert len(seen) == 4 * 16


def test_sampler_random_mode_excludes_demo_and_heldout_and_recurring_uses_base():
    instance = generate_instance(6, 3, 0.3, 1.0, 9)
    sampler = Sampler(C.ENV_RANDOM, instance, 0.3, 0.3, 0)
    pos, _ = sampler.sample(200)
    drawn = {tuple(row) for row in pos}
    assert tuple(to_fast(instance).pos) not in drawn
    held = heldout_instances(instance, C.ENV_RANDOM, 0.3, 0.3, 9, n=5)
    assert not (drawn & {tuple(to_fast(i).pos) for i in held})

    rec = Sampler(C.ENV_RECURRING, instance, 0.3, 0.3, 0)
    pos, dur = rec.sample(10)
    assert np.allclose(pos, [j.position for j in instance.jobs])
    assert not np.allclose(dur[0], [j.duration for j in instance.jobs])
    assert np.allclose(Sampler(C.ENV_RECURRING, instance, 0.0, 0.3, 0).sample(3)[1], [j.duration for j in instance.jobs])


def test_replay_dispatch_reproduces_traced_makespan_via_vehicle_functions():
    instance = generate_instance(8, 3, 0.3, 1.0, 5)
    geom = geometry(instance)
    result = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 2000, 0, C.METHOD_MAPPO, C.ACTOR_NET)
    trace = trace_policy(geom, result, instance)
    dispatch = replay_dispatch(instance, trace["actions"])
    assert abs(dispatch.protocol_result.makespan - trace["makespan"]) < 1e-9
    pos, dur = arrays_from_instances([instance])
    assert simulate(geom, pos, dur, greedy_act_fn(geom, result))[0] == trace["makespan"]


def test_trace_policy_shapes_and_critic_prediction_kinds():
    instance = generate_instance(6, 3, 0.3, 1.0, 5)
    geom = geometry(instance)
    for actor_kind in (C.ACTOR_NET, C.ACTOR_TABLE):
        mappo = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 300, 0, C.METHOD_MAPPO, actor_kind)
        ippo = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 300, 0, C.METHOD_IPPO, actor_kind)
        t_m, t_i = trace_policy(geom, mappo, instance), trace_policy(geom, ippo, instance)
        assert t_m["probs"].shape == (6, 3, 3) and t_m["actions"].shape == (6, 3)
        assert t_m["critic_makespan"].shape == (6,)      # zentral: ein Wert je Auftrag
        assert t_i["critic_makespan"].shape == (6, 3)    # lokal: ein Wert je Agent
        assert np.allclose(t_m["probs"].sum(-1), 1.0)
