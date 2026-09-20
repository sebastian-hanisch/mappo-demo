"""Ein einziger gemeinsamer Baustein: aus einer beliebigen Zuteilung (welcher
Agent bearbeitet welche Aufträge, in welcher Reihenfolge) die resultierenden
Fertigstellungszeiten und den Makespan neu berechnen. Wird sowohl von der
Brute-Force-Referenz als auch von der Task-Swap-Verhandlung (`cn_negotiation.py`)
und deren Visualisierung verwendet - dieselbe Akkumulationsformel wie in
`cn_bidding.compute_bid`/`cn_protocol.run_protocol`."""


def schedule_from_assignment(instance, schedules):
    """schedules: dict agent_id -> Tupel von job_index, in Bearbeitungsreihenfolge.
    Agenten ohne Eintrag gelten als leer (Fertigstellungszeit 0.0). Gibt
    (agent_finish_times, makespan) zurück, agent_finish_times in Agenten-ID-
    Reihenfolge."""
    agent_finish_times = []
    for agent_id in range(instance.n_agents):
        position = instance.agent_start_positions[agent_id]
        free_time = 0.0
        for job_index in schedules.get(agent_id, ()):
            job = instance.jobs[job_index]
            travel = instance.travel_time(position, job.position)
            free_time = free_time + travel + job.duration
            position = job.position
        agent_finish_times.append(free_time)
    agent_finish_times = tuple(agent_finish_times)
    makespan = max(agent_finish_times) if agent_finish_times else 0.0
    return agent_finish_times, makespan
