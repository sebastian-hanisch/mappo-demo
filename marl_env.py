"""Die Dispatch-Umgebung: dieselbe Ankündigen-Bieten-Zuschlag-Schleife wie Contract
Net, nur dass jeder Agent pro Auftrag eine Aktion wählt - einen Aufschlag auf sein
eigenes Gebot (`C.BIAS_ACTIONS`). Der Zuschlag bleibt Contract Nets Regel: niedrigstes
(modifiziertes) Gebot gewinnt, Gleichstand -> niedrigste Agenten-ID.

Zwei Pfade mit identischem Ergebnis (getestet):
- `rollout`: schneller Float-Kernel für das Training (Millionen Entscheidungen);
  die Vehikel-Dataclasses wären hier ~10x langsamer.
- `run_dispatch`: derselbe Ablauf über `cn_bidding.compute_bid`/`award` und
  `ProtocolResult` - für Anzeige und Schritt-Ansichten.

Rauschen und Trainings-Zufall nutzen Pythons `random.Random` (bei String-Seeds
versionsstabil), nicht numpy-Generatoren, deren Streams nicht über Versionen
garantiert sind."""

import math
import random
from dataclasses import dataclass
from functools import lru_cache

import cn_constants as C
from cn_bidding import AgentState, Bid, award, compute_bid
from cn_protocol import AwardStep, ProtocolResult
from cn_scenario import Instance, Job, generate_instance


class FastInstance:
    """Reine Float-Listen statt Dataclasses - nur für den Trainings-Kernel."""

    __slots__ = ("pos", "dur", "start", "tr", "n", "k")


def to_fast(instance):
    fi = FastInstance()
    fi.pos = [j.position for j in instance.jobs]
    fi.dur = [j.duration for j in instance.jobs]
    fi.start = list(instance.agent_start_positions)
    fi.tr = instance.travel_time_per_unit
    fi.n = instance.n_jobs
    fi.k = instance.n_agents
    return fi


def noisy_durations(durations, sigma, rng):
    """Jede Dauer wird mit exp(N(0, sigma)) multipliziert; sigma=0 lässt sie unverändert
    (und verbraucht dann keinen Zufall)."""
    if sigma <= 0:
        return list(durations)
    return [d * math.exp(rng.gauss(0.0, sigma)) for d in durations]


def noisy_fast(base_fast, sigma, rng):
    fi = FastInstance()
    fi.pos, fi.start, fi.tr, fi.n, fi.k = base_fast.pos, base_fast.start, base_fast.tr, base_fast.n, base_fast.k
    fi.dur = noisy_durations(base_fast.dur, sigma, rng)
    return fi


def noisy_instance(instance, sigma, rng):
    durations = noisy_durations([j.duration for j in instance.jobs], sigma, rng)
    jobs = tuple(Job(index=j.index, position=j.position, duration=d) for j, d in zip(instance.jobs, durations))
    return Instance(
        n_jobs=instance.n_jobs, n_agents=instance.n_agents, jobs=jobs,
        agent_start_positions=instance.agent_start_positions,
        travel_time_per_unit=instance.travel_time_per_unit,
    )


def rollout(fi, policy, obs_fn, biases=C.BIAS_ACTIONS, record=False):
    """Ein Durchlauf. policy(agent_id, state) -> Aktions-Index. Gibt den Makespan zurück, mit
    record=True zusätzlich die Trajektorie: pro Auftrag (Zustände aller Agenten, Aktionen
    aller Agenten)."""
    k = fi.k
    pos = list(fi.start)
    free = [0.0] * k
    traj = []
    for j in range(fi.n):
        job_pos, job_dur = fi.pos[j], fi.dur[j]
        best_key = None
        best_agent = 0
        best_finish = 0.0
        states = []
        actions = []
        for a in range(k):
            travel = abs(pos[a] - job_pos) * fi.tr
            state = obs_fn(free[a], travel, j)
            action = policy(a, state)
            finish = free[a] + travel + job_dur
            key = (finish + biases[action], a)
            if best_key is None or key < best_key:
                best_key, best_agent, best_finish = key, a, finish
            if record:
                states.append(state)
                actions.append(action)
        pos[best_agent] = job_pos
        free[best_agent] = best_finish
        if record:
            traj.append((states, actions))
    makespan = max(free)
    return (makespan, traj) if record else makespan


def cnp_makespan_fast(fi):
    """Rohes Contract Net über den Float-Kernel (alle Agenten bieten normal)."""
    return rollout(fi, lambda a, s: 0, lambda free, travel, j: 0)


@dataclass(frozen=True)
class DispatchResult:
    protocol_result: ProtocolResult  # Gebote in den Steps = MODIFIZIERTE Gebote
    states: tuple                    # pro Auftrag: Zustände aller Agenten
    actions: tuple                   # pro Auftrag: Aktionen aller Agenten
    original_bids: tuple             # pro Auftrag: unmodifizierte Gebote (finish_time) aller Agenten


def run_dispatch(instance, policy, obs_fn, biases=C.BIAS_ACTIONS):
    """Wie `cn_protocol.run_protocol`, aber jeder Agent wählt pro Auftrag einen Gebots-
    Aufschlag. Über die Vehikel-Funktionen `compute_bid` und `award`."""
    agents = [
        AgentState(agent_id=a, position=instance.agent_start_positions[a], free_time=0.0)
        for a in range(instance.n_agents)
    ]
    steps, all_states, all_actions, all_original = [], [], [], []
    assignment = {}
    schedules = {a: [] for a in range(instance.n_agents)}

    for job in instance.jobs:
        original = [compute_bid(instance, agent, job) for agent in agents]
        states = [obs_fn(agent.free_time, bid.travel_time, job.index) for agent, bid in zip(agents, original)]
        actions = [policy(a, s) for a, s in enumerate(states)]
        modified = tuple(
            Bid(agent_id=b.agent_id, travel_time=b.travel_time, finish_time=b.finish_time + biases[act])
            for b, act in zip(original, actions)
        )
        winner_bid = award(modified)
        winner_id = winner_bid.agent_id
        true_finish = original[winner_id].finish_time
        winner = agents[winner_id]
        agents[winner_id] = AgentState(
            agent_id=winner_id, position=job.position, free_time=true_finish,
            assigned_jobs=winner.assigned_jobs + (job.index,),
        )
        assignment[job.index] = winner_id
        schedules[winner_id].append(job.index)
        steps.append(AwardStep(
            step=job.index, job_index=job.index, bids=modified, winner_agent_id=winner_id,
            agent_positions_after=tuple(a.position for a in agents),
            agent_free_times_after=tuple(a.free_time for a in agents),
        ))
        all_states.append(tuple(states))
        all_actions.append(tuple(actions))
        all_original.append(tuple(b.finish_time for b in original))

    finish_times = tuple(a.free_time for a in agents)
    result = ProtocolResult(
        steps=tuple(steps), assignment=assignment,
        schedules={a: tuple(jobs) for a, jobs in schedules.items()},
        agent_finish_times=finish_times, makespan=max(finish_times) if finish_times else 0.0,
    )
    return DispatchResult(
        protocol_result=result, states=tuple(all_states), actions=tuple(all_actions),
        original_bids=tuple(all_original),
    )


# --- Trainings-Umgebungen ------------------------------------------------------

POOL_SEED_OFFSET = 10_000_000       # Trainingspool im Modus "zufällig"
HELDOUT_SEED_OFFSET = 3_000_000_000  # frische Held-out-/Generalisierungs-Instanzen


@lru_cache(maxsize=8)
def random_pool(n_jobs, n_agents, duration_variability, travel_time_per_unit):
    """Trainingspool für den Modus "zufällige Instanzen": RANDOM_POOL_SIZE Instanzen mit
    denselben (n, k, Streuung, Anfahrt), aber anderen Seeds als Demo-Instanz und Held-out."""
    return tuple(
        to_fast(generate_instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, POOL_SEED_OFFSET + i))
        for i in range(C.RANDOM_POOL_SIZE)
    )


def make_episode_source(env_mode, base_instance, sigma, duration_variability, train_seed):
    """Gibt next_episode() -> FastInstance zurück. Der Trainings-Zufall hängt NUR vom
    train_seed ab, nie vom Szenario-Seed (entkoppelt)."""
    env_rng = random.Random(f"iql-env-{train_seed}")
    if env_mode == C.ENV_RANDOM:
        pool = random_pool(
            base_instance.n_jobs, base_instance.n_agents, duration_variability,
            base_instance.travel_time_per_unit,
        )
        return lambda: pool[env_rng.randrange(len(pool))]
    base_fast = to_fast(base_instance)
    return lambda: noisy_fast(base_fast, sigma, env_rng)


def heldout_instances(base_instance, env_mode, sigma, duration_variability, scenario_seed, n=C.N_HELDOUT):
    """Held-out-Instanzen (als `Instance`, für CP-SAT und Anzeige). Hängen nur vom Szenario ab,
    nie vom Trainings-Seed. Wiederkehrend: Rauschen-Ziehungen desselben Szenarios; zufällig:
    frische Instanzen aus Seeds außerhalb des Trainingspools."""
    if env_mode == C.ENV_RANDOM:
        return tuple(
            generate_instance(
                base_instance.n_jobs, base_instance.n_agents, duration_variability,
                base_instance.travel_time_per_unit, HELDOUT_SEED_OFFSET + scenario_seed * 1000 + i,
            )
            for i in range(n)
        )
    rng = random.Random(f"heldout-{scenario_seed}")
    return tuple(noisy_instance(base_instance, sigma, rng) for _ in range(n))
