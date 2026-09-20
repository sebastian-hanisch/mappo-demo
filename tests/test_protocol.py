from cn_ortools_reference import SCALE, solve_with_ortools
from cn_protocol import run_protocol
from cn_scenario import Instance, Job, generate_instance

# Siehe test_ortools_reference.py für die Begründung: CP-SAT rundet Zeiten bewusst AUF,
# was seinen berichteten Makespan geringfügig über den wahren stetigen Wert heben kann.
_ROUNDING_TOLERANCE_PER_JOB = 2.0 / SCALE


def _instance(jobs, n_agents=2, agent_start_positions=None, travel_time_per_unit=1.0):
    if agent_start_positions is None:
        agent_start_positions = tuple(0.0 for _ in range(n_agents))
    return Instance(
        n_jobs=len(jobs), n_agents=n_agents, jobs=jobs,
        agent_start_positions=agent_start_positions, travel_time_per_unit=travel_time_per_unit,
    )


def test_matches_hand_computed_tiny_instance_two_agents_two_jobs():
    # Agent 0 startet bei 0, Agent 1 bei 100 - Job 0 liegt bei 1 (trivial fuer Agent 0),
    # Job 1 liegt bei 2 (auch trivial fuer Agent 0, aber Agent 0 ist inzwischen beschaeftigt).
    jobs = (Job(index=0, position=1.0, duration=10.0), Job(index=1, position=2.0, duration=1.0))
    instance = _instance(jobs, n_agents=2, agent_start_positions=(0.0, 100.0))

    result = run_protocol(instance)

    # Job 0: Agent 0 bietet 0+1+10=11, Agent 1 bietet 100+99+10=209 -> Agent 0 gewinnt.
    assert result.assignment[0] == 0
    # Job 1: Agent 0 (jetzt bei Position 1, frei ab 11) bietet 11+1+1=13;
    # Agent 1 (bei 100, frei ab 0) bietet 0+98+1=99 -> Agent 0 gewinnt trotzdem hier.
    assert result.assignment[1] == 0
    assert result.agent_finish_times[0] == 13.0
    assert result.agent_finish_times[1] == 0.0
    assert result.makespan == 13.0


def test_every_job_assigned_exactly_once_across_random_instances():
    for seed in range(30):
        instance = generate_instance(
            n_jobs=10, n_agents=3, duration_variability=0.5, travel_time_per_unit=1.0, seed=seed
        )
        result = run_protocol(instance)
        assigned = sorted(result.assignment.keys())
        assert assigned == list(range(10)), f"seed={seed}"
        all_scheduled = sorted(j for jobs in result.schedules.values() for j in jobs)
        assert all_scheduled == list(range(10)), f"seed={seed}"


def test_once_awarded_a_job_never_changes_agent_across_later_steps():
    for seed in range(20):
        instance = generate_instance(
            n_jobs=8, n_agents=2, duration_variability=0.6, travel_time_per_unit=1.0, seed=seed
        )
        result = run_protocol(instance)
        fixed = {}
        for step in result.steps:
            if step.job_index in fixed:
                assert fixed[step.job_index] == step.winner_agent_id, f"seed={seed}"
            else:
                fixed[step.job_index] = step.winner_agent_id
        # Am Ende muss das mit der finalen Zuteilung uebereinstimmen (keine Nachbesserung).
        assert fixed == result.assignment, f"seed={seed}"


def test_protocol_makespan_never_beats_cp_sat_optimum_across_random_instances():
    for seed in range(10):
        instance = generate_instance(
            n_jobs=6, n_agents=2, duration_variability=0.4, travel_time_per_unit=1.0, seed=seed
        )
        result = run_protocol(instance)
        exact = solve_with_ortools(instance, time_limit_seconds=5.0)
        assert exact.feasible, f"seed={seed}"
        tolerance = _ROUNDING_TOLERANCE_PER_JOB * instance.n_jobs
        assert result.makespan >= exact.makespan - tolerance, f"seed={seed}"


def test_single_agent_cnp_still_serves_jobs_in_announcement_order():
    # Mit nur einem Agenten gibt es keine Zuteilungs-WAHL mehr (jedes Gebot gewinnt
    # trivial) - aber die REIHENFOLGE bleibt trotzdem an die Ankündigungsreihenfolge
    # gebunden, das Protokoll kann Aufträge nicht zur besseren Sequenzierung umsortieren.
    # Das ist der Grund, warum selbst der Ein-Agenten-Fall gegen CP-SAT (das die
    # guenstigste Reihenfolge frei waehlen darf) verlieren kann - siehe
    # test_single_agent_cnp_can_lose_to_cp_sat_on_ordering_alone unten.
    for seed in range(10):
        instance = generate_instance(
            n_jobs=5, n_agents=1, duration_variability=0.5, travel_time_per_unit=1.0, seed=seed
        )
        result = run_protocol(instance)
        assert result.schedules[0] == tuple(range(5)), f"seed={seed}"


def test_single_agent_cnp_can_lose_to_cp_sat_on_ordering_alone():
    # Gegenbeispiel, das beweist: die Lücke zu CP-SAT kommt nicht nur aus falscher
    # Agenten-WAHL, sondern schon allein aus der erzwungenen Ankündigungsreihenfolge.
    instance = generate_instance(n_jobs=5, n_agents=1, duration_variability=0.5, travel_time_per_unit=1.0, seed=0)
    result = run_protocol(instance)
    exact = solve_with_ortools(instance, time_limit_seconds=5.0)
    assert exact.feasible
    assert result.makespan > exact.makespan


def test_announcement_order_is_job_index_order():
    instance = generate_instance(n_jobs=7, n_agents=2, duration_variability=0.3, travel_time_per_unit=1.0, seed=3)
    result = run_protocol(instance)
    assert [step.job_index for step in result.steps] == list(range(7))


def test_step_trace_final_step_reconstructs_full_schedule():
    instance = generate_instance(n_jobs=6, n_agents=2, duration_variability=0.3, travel_time_per_unit=1.0, seed=4)
    result = run_protocol(instance)
    assert result.steps[-1].agent_free_times_after == result.agent_finish_times
