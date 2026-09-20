"""Erschöpfender Referenzlöser - probiert jede Zuordnung von Aufträgen zu Agenten
UND jede Reihenfolge innerhalb eines Agenten durch. Nur für sehr kleine n_jobs
praktikabel (Testzwecke: Cross-Check gegen cn_ortools_reference, nicht live in
der App gezeigt)."""

from itertools import permutations, product

from cn_schedule import schedule_from_assignment


def _schedule_makespan(instance, assignment_by_agent):
    """assignment_by_agent: dict agent_id -> Tupel von job_index in Bearbeitungsreihenfolge."""
    _, makespan = schedule_from_assignment(instance, assignment_by_agent)
    return makespan


def solve_bruteforce(instance):
    n, k = instance.n_jobs, instance.n_agents
    best_makespan = float("inf")
    best_assignment = None

    for agent_choice in product(range(k), repeat=n):
        by_agent = {a: [] for a in range(k)}
        for job_index, agent_id in enumerate(agent_choice):
            by_agent[agent_id].append(job_index)

        # Für jeden Agenten jede Reihenfolge seiner eigenen Aufträge durchprobieren.
        agent_orderings = [
            list(permutations(jobs)) if jobs else [()] for jobs in by_agent.values()
        ]
        for combo in product(*agent_orderings):
            assignment_by_agent = {a: combo[a] for a in range(k)}
            makespan = _schedule_makespan(instance, assignment_by_agent)
            if makespan < best_makespan:
                best_makespan = makespan
                best_assignment = assignment_by_agent

    return best_makespan, best_assignment
