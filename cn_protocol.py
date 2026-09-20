"""Führt das vollständige Contract Net Protocol (Ankündigen -> Bieten -> Zuschlag)
über eine ganze Instanz aus - Aufträge werden strikt in Index-Reihenfolge
angekündigt (= Ankunftsreihenfolge; der Manager kennt zukünftige Aufträge nicht
im Voraus). Kein Backtracking ist strukturell möglich: `run_protocol` hängt
Zuschläge nur an, ändert nie einen bereits vergebenen an."""

from dataclasses import dataclass

from cn_bidding import AgentState, award, compute_bid


@dataclass(frozen=True)
class AwardStep:
    step: int
    job_index: int
    bids: tuple  # eine Bid pro Agent, in Agenten-ID-Reihenfolge
    winner_agent_id: int
    agent_positions_after: tuple
    agent_free_times_after: tuple


@dataclass(frozen=True)
class ProtocolResult:
    steps: tuple
    assignment: dict  # job_index -> agent_id
    schedules: dict  # agent_id -> Tupel von job_index, in Bearbeitungsreihenfolge
    agent_finish_times: tuple
    makespan: float


def run_protocol(instance):
    agents = [
        AgentState(agent_id=a, position=instance.agent_start_positions[a], free_time=0.0)
        for a in range(instance.n_agents)
    ]
    steps = []
    assignment = {}
    schedules = {a: [] for a in range(instance.n_agents)}

    for job in instance.jobs:
        bids = tuple(compute_bid(instance, agent, job) for agent in agents)
        winner_bid = award(bids)
        winner = agents[winner_bid.agent_id]

        agents[winner.agent_id] = AgentState(
            agent_id=winner.agent_id,
            position=job.position,
            free_time=winner_bid.finish_time,
            assigned_jobs=winner.assigned_jobs + (job.index,),
        )
        assignment[job.index] = winner.agent_id
        schedules[winner.agent_id].append(job.index)

        steps.append(
            AwardStep(
                step=job.index,
                job_index=job.index,
                bids=bids,
                winner_agent_id=winner.agent_id,
                agent_positions_after=tuple(a.position for a in agents),
                agent_free_times_after=tuple(a.free_time for a in agents),
            )
        )

    agent_finish_times = tuple(a.free_time for a in agents)
    makespan = max(agent_finish_times) if agent_finish_times else 0.0

    return ProtocolResult(
        steps=tuple(steps),
        assignment=assignment,
        schedules={a: tuple(jobs) for a, jobs in schedules.items()},
        agent_finish_times=agent_finish_times,
        makespan=makespan,
    )
