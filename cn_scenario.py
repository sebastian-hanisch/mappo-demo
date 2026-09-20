"""Zufällige Kran-zu-Container-Auftrag-Instanzen für die Contract-Net-Demo - eine
bewusst VEREINFACHTE Variante des Quay-Crane-Scheduling-Problems aus quaycrane-demo:
keine gemeinsame Schiene, keine Non-Crossing-Regel zwischen Kränen. Das ist Absicht,
nicht eine übersehene Komplexität - diese Demo zeigt einen Koordinations-MECHANISMUS
(Contract Net Protocol), nicht die physische Kran-Choreographie, die quaycrane-demo
bereits eigenständig behandelt. quaycrane-demo dient hier nur als Namensgeber/Vorbild
für das Vokabular (Kräne, Aufträge, Positionen), nicht als strukturelle Vorlage."""

from dataclasses import dataclass

import numpy as np

from cn_constants import DURATION_BASE_RANGE, POSITION_RANGE_MAX, SPIKE_MULTIPLIER, SPIKE_PROBABILITY_SCALE


@dataclass(frozen=True)
class Job:
    index: int
    position: float
    duration: float


@dataclass(frozen=True)
class Instance:
    n_jobs: int
    n_agents: int
    jobs: tuple  # Tupel von Job, in ANKÜNDIGUNGS-Reihenfolge (= Index-Reihenfolge)
    agent_start_positions: tuple  # Länge n_agents
    travel_time_per_unit: float

    def travel_time(self, pos_a, pos_b):
        return abs(pos_a - pos_b) * self.travel_time_per_unit


def generate_instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    rng = np.random.default_rng(seed)

    positions = rng.uniform(0, POSITION_RANGE_MAX, size=n_jobs)
    base_durations = rng.uniform(DURATION_BASE_RANGE[0], DURATION_BASE_RANGE[1], size=n_jobs)
    spike_mask = rng.random(n_jobs) < duration_variability * SPIKE_PROBABILITY_SCALE
    durations = np.where(spike_mask, base_durations * SPIKE_MULTIPLIER, base_durations)

    jobs = tuple(
        Job(index=i, position=float(positions[i]), duration=float(durations[i]))
        for i in range(n_jobs)
    )
    agent_start_positions = tuple(
        (c + 0.5) * POSITION_RANGE_MAX / n_agents for c in range(n_agents)
    )

    return Instance(
        n_jobs=n_jobs,
        n_agents=n_agents,
        jobs=jobs,
        agent_start_positions=agent_start_positions,
        travel_time_per_unit=travel_time_per_unit,
    )
