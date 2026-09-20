import random

import cn_constants as C
from cn_protocol import run_protocol
from cn_schedule import schedule_from_assignment
from cn_scenario import generate_instance
from marl_env import (
    cnp_makespan_fast, heldout_instances, make_episode_source, noisy_durations, noisy_instance,
    rollout, run_dispatch, to_fast,
)
from marl_iql import greedy_policy, new_q_tables
from marl_obs import describe_state, make_observer


def _random_q(n_jobs, n_agents, seed):
    rng = random.Random(seed)
    _, n_states = make_observer(n_jobs, n_agents)
    q = new_q_tables(n_agents, n_states)
    for agent_table in q:
        for row in agent_table:
            for a in range(len(row)):
                row[a] = rng.uniform(-5, 0)
    return q


def test_untrained_policy_equals_raw_contract_net():
    for seed in range(50):
        instance = generate_instance(2 + seed % 8, 2 + seed % 3, 0.4, 1.0, seed)
        obs_fn, n_states = make_observer(instance.n_jobs, instance.n_agents)
        q = new_q_tables(instance.n_agents, n_states)
        dispatch = run_dispatch(instance, greedy_policy(q), obs_fn)
        reference = run_protocol(instance)
        assert dispatch.protocol_result.makespan == reference.makespan, f"seed={seed}"
        assert dispatch.protocol_result.assignment == reference.assignment, f"seed={seed}"


def test_fast_kernel_zero_bias_equals_run_protocol():
    for seed in range(300):
        instance = generate_instance(1 + seed % 10, 1 + seed % 4, 0.5, 0.2 + (seed % 5) * 0.4, seed)
        assert cnp_makespan_fast(to_fast(instance)) == run_protocol(instance).makespan, f"seed={seed}"


def test_vehicle_path_equals_fast_kernel_for_random_q_tables():
    for seed in range(60):
        n_jobs, n_agents = 3 + seed % 6, 2 + seed % 3
        instance = generate_instance(n_jobs, n_agents, 0.4, 1.0, seed)
        q = _random_q(n_jobs, n_agents, seed)
        obs_fn, _ = make_observer(n_jobs, n_agents)
        policy = greedy_policy(q)
        fast_ms = rollout(to_fast(instance), policy, obs_fn)
        dispatch = run_dispatch(instance, policy, obs_fn)
        assert dispatch.protocol_result.makespan == fast_ms, f"seed={seed}"
        # Der Zeitplan aus der Zuteilung reproduziert denselben Makespan (echte, nicht modifizierte Zeiten)
        _, ms = schedule_from_assignment(instance, dispatch.protocol_result.schedules)
        assert abs(ms - fast_ms) < 1e-9, f"seed={seed}"


def test_dispatch_records_shifted_and_original_bids():
    instance = generate_instance(4, 2, 0.3, 1.0, 3)
    obs_fn, _ = make_observer(4, 2)
    # Agent 0 bietet immer "hoch" (+Delta), Agent 1 immer normal
    policy = lambda agent, state: 1 if agent == 0 else 0
    dispatch = run_dispatch(instance, policy, obs_fn)
    for step, originals, actions in zip(dispatch.protocol_result.steps, dispatch.original_bids, dispatch.actions):
        assert actions == (1, 0)
        assert abs(step.bids[0].finish_time - (originals[0] + C.BIAS_DELTA_MIN)) < 1e-12
        assert step.bids[1].finish_time == originals[1]


def test_single_agent_two_jobs_makespan_independent_of_q():
    instance = generate_instance(2, 1, 0.3, 1.0, 4)
    obs_fn, _ = make_observer(2, 1)
    _, expected = schedule_from_assignment(instance, {0: (0, 1)})
    for seed in range(5):
        q = _random_q(2, 1, seed)
        assert rollout(to_fast(instance), greedy_policy(q), obs_fn) == expected


def test_noisy_durations_determinism_and_sigma_zero():
    base = [5.0, 10.0, 7.5]
    assert noisy_durations(base, 0.0, random.Random("x")) == base
    a = noisy_durations(base, 0.3, random.Random("seed-1"))
    b = noisy_durations(base, 0.3, random.Random("seed-1"))
    assert a == b and a != base and all(x > 0 for x in a)


def test_noisy_instance_keeps_positions_changes_durations():
    instance = generate_instance(5, 2, 0.3, 1.0, 2)
    noisy = noisy_instance(instance, 0.4, random.Random("n"))
    assert [j.position for j in noisy.jobs] == [j.position for j in instance.jobs]
    assert [j.duration for j in noisy.jobs] != [j.duration for j in instance.jobs]


def test_heldout_set_independent_of_training_seed():
    instance = generate_instance(6, 3, 0.3, 1.0, 9)
    for mode in (C.ENV_RECURRING, C.ENV_RANDOM):
        a = heldout_instances(instance, mode, 0.3, 0.3, 9, n=5)
        b = heldout_instances(instance, mode, 0.3, 0.3, 9, n=5)
        assert [[j.duration for j in i.jobs] for i in a] == [[j.duration for j in i.jobs] for i in b]


def test_random_pool_excludes_heldout_and_demo_instance():
    instance = generate_instance(6, 3, 0.3, 1.0, 9)
    source = make_episode_source(C.ENV_RANDOM, instance, 0.3, 0.3, 0)
    drawn = {tuple(source().pos) for _ in range(50)}
    assert tuple(to_fast(instance).pos) not in drawn
    held = heldout_instances(instance, C.ENV_RANDOM, 0.3, 0.3, 9, n=5)
    assert not (drawn & {tuple(to_fast(i).pos) for i in held})


def test_observer_state_range_and_description():
    obs_fn, n_states = make_observer(8, 3)
    assert n_states == 8 * 12
    assert obs_fn(0.0, 0.0, 0) == 0
    assert obs_fn(1e9, 1e9, 7) == n_states - 1
    job, free_text, travel_text = describe_state(n_states - 1, 8, 3)
    assert job == 8 and "≥" in free_text and "≥" in travel_text
    # jeder Zustand ist eindeutig beschreibbar
    assert len({describe_state(s, 8, 3) for s in range(n_states)}) == n_states
