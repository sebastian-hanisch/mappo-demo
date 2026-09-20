from cn_protocol import run_protocol
from cn_scenario import Instance, Job, generate_instance
from cn_schedule import schedule_from_assignment


def _instance(jobs, n_agents=2, agent_start_positions=None, travel_time_per_unit=1.0):
    if agent_start_positions is None:
        agent_start_positions = tuple(0.0 for _ in range(n_agents))
    return Instance(
        n_jobs=len(jobs), n_agents=n_agents, jobs=jobs,
        agent_start_positions=agent_start_positions, travel_time_per_unit=travel_time_per_unit,
    )


def test_schedule_from_assignment_matches_protocol_result_across_random_instances():
    for seed in range(20):
        instance = generate_instance(
            n_jobs=8, n_agents=2, duration_variability=0.4, travel_time_per_unit=1.0, seed=seed
        )
        result = run_protocol(instance)
        agent_finish_times, makespan = schedule_from_assignment(instance, result.schedules)
        assert agent_finish_times == result.agent_finish_times, f"seed={seed}"
        assert makespan == result.makespan, f"seed={seed}"


def test_schedule_from_assignment_hand_computed_tiny_example():
    jobs = (Job(index=0, position=1.0, duration=10.0), Job(index=1, position=2.0, duration=1.0))
    instance = _instance(jobs, n_agents=2, agent_start_positions=(0.0, 100.0))
    schedules = {0: (0, 1), 1: ()}
    agent_finish_times, makespan = schedule_from_assignment(instance, schedules)
    # Agent 0: 0 -> Job0 (Anfahrt 1, Dauer 10) -> frei ab 11 -> Job1 (Anfahrt 1, Dauer 1) -> frei ab 13.
    assert agent_finish_times == (13.0, 0.0)
    assert makespan == 13.0


def test_schedule_from_assignment_empty_schedule_gives_zero():
    jobs = (Job(index=0, position=1.0, duration=10.0),)
    instance = _instance(jobs, n_agents=2)
    agent_finish_times, makespan = schedule_from_assignment(instance, {0: (), 1: ()})
    assert agent_finish_times == (0.0, 0.0)
    assert makespan == 0.0


def test_schedule_from_assignment_missing_agent_key_treated_as_empty():
    jobs = (Job(index=0, position=1.0, duration=10.0),)
    instance = _instance(jobs, n_agents=2)
    agent_finish_times, makespan = schedule_from_assignment(instance, {0: (0,)})
    assert agent_finish_times == (11.0, 0.0)
    assert makespan == 11.0
