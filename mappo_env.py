"""Die vektorisierte Dispatch-Umgebung für PPO: derselbe Ablauf wie `marl_env.rollout` (Aufträge in
Index-Reihenfolge, jeder Agent wählt einen Gebots-Aufschlag, Zuschlag = niedrigstes abgegebenes Gebot,
Gleichstand -> niedrigste Agenten-ID), aber für ein ganzes BATCH von Episoden gleichzeitig in numpy.
Die Operationsreihenfolge ist identisch zum Float-Kernel - das Ergebnis stimmt bitweise überein (getestet).

Außerdem: die Merkmale für Akteur und Kritiker (stetig fürs Netz, Buckets für die Tabelle) und
`replay_dispatch`, das aufgezeichnete Aktionen durch die Vehikel-Funktionen `compute_bid`/`award` schickt
(für Gantt-Chart und Schritt-Ansicht)."""

from functools import lru_cache

import numpy as np

import cn_constants as C
from marl_env import random_pool, run_dispatch
from marl_obs import make_observer, n_free_buckets, n_travel_buckets

BIAS = np.array(C.BIAS_ACTIONS)
TRAVEL_EDGES = np.array(C.TRAVEL_EDGES)
N_ACTIONS = len(C.BIAS_ACTIONS)


class Geometry:
    """Alles an einer Instanz, was nicht vom einzelnen Auftrag abhängt: Größe, Startpositionen, Anfahrt."""

    def __init__(self, n_jobs, n_agents, start_positions, travel_time_per_unit):
        self.n = n_jobs
        self.k = n_agents
        self.start = np.array(start_positions, dtype=float)
        self.tr = travel_time_per_unit
        self.fair = n_jobs * C.MEAN_DURATION / n_agents  # "faire Last" pro Agent
        self.free_edges = np.array([f * self.fair for f in C.FREE_TIME_EDGE_FRACTIONS])


def geometry(instance):
    return Geometry(instance.n_jobs, instance.n_agents, instance.agent_start_positions, instance.travel_time_per_unit)


def arrays_from_instances(instances):
    pos = np.array([[j.position for j in i.jobs] for i in instances])
    dur = np.array([[j.duration for j in i.jobs] for i in instances])
    return pos, dur


@lru_cache(maxsize=8)
def _pool_arrays(n_jobs, n_agents, duration_variability, travel_time_per_unit):
    pool = random_pool(n_jobs, n_agents, duration_variability, travel_time_per_unit)
    return np.array([f.pos for f in pool]), np.array([f.dur for f in pool])


class Sampler:
    """Zieht Trainings-Batches. Der Zufall hängt NUR vom Trainings-Seed ab, nie vom Szenario-Seed:
    im Modus "zufällig" ist die Demo-Instanz nur Größenlieferant, im Modus "wiederkehrend" die Basis."""

    def __init__(self, env_mode, base_instance, sigma, duration_variability, train_seed):
        self.env_mode = env_mode
        self.sigma = sigma
        self.n = base_instance.n_jobs
        self.rng = np.random.default_rng([train_seed, 17])
        if env_mode == C.ENV_RANDOM:
            self.pool_pos, self.pool_dur = _pool_arrays(
                base_instance.n_jobs, base_instance.n_agents, duration_variability,
                base_instance.travel_time_per_unit,
            )
        else:
            self.base_pos = np.array([j.position for j in base_instance.jobs])
            self.base_dur = np.array([j.duration for j in base_instance.jobs])

    def sample(self, batch):
        if self.env_mode == C.ENV_RANDOM:
            idx = self.rng.integers(0, len(self.pool_pos), batch)
            return self.pool_pos[idx], self.pool_dur[idx]
        dur = self.base_dur * np.exp(self.sigma * self.rng.standard_normal((batch, self.n)))
        return np.broadcast_to(self.base_pos, (batch, self.n)), dur


def simulate(geom, pos, dur, act_fn):
    """Ein Batch von Episoden. act_fn(j, free, apos, travel, jp, jd) -> Aktionen [B, k]. Gibt die
    Makespans [B] zurück. Operationsreihenfolge wie `marl_env.rollout` (bitweise gleiche Ergebnisse)."""
    batch, n = pos.shape
    k = geom.k
    agent_pos = np.tile(geom.start, (batch, 1))
    free = np.zeros((batch, k))
    rows = np.arange(batch)
    for j in range(n):
        job_pos = pos[:, j]
        job_dur = dur[:, j]
        travel = np.abs(agent_pos - job_pos[:, None]) * geom.tr
        actions = act_fn(j, free, agent_pos, travel, job_pos, job_dur)
        finish = free + travel + job_dur[:, None]
        winner = (finish + BIAS[actions]).argmin(1)
        free[rows, winner] = finish[rows, winner]
        agent_pos[rows, winner] = job_pos
    return free.max(1)


def cnp_makespans(geom, pos, dur):
    """Rohes Contract Net (alle Agenten bieten normal) über den vektorisierten Kernel."""
    return simulate(geom, pos, dur, lambda j, free, apos, travel, jp, jd: np.zeros(free.shape, dtype=int))


# --- Merkmale --------------------------------------------------------------------

def local_features(geom, j, free, travel, job_pos, job_dur, agent_pos, use_job_index=True):
    """Stetige lokale Merkmale je Agent, Form [k, B, F]: eigene Freizeit (in fairer Last), Anfahrt/10,
    eigene Position/20, Auftragsdauer/10, Auftragsposition/20 und (optional) der öffentliche Auftrags-Zähler
    j/n. Nichts davon verrät etwas über die Zustände anderer Agenten."""
    batch, k = free.shape
    feats = [
        free / geom.fair, travel / 10.0, agent_pos / 20.0,
        np.repeat((job_dur / 10.0)[:, None], k, 1), np.repeat((job_pos / 20.0)[:, None], k, 1),
    ]
    if use_job_index:
        feats.append(np.full((batch, k), j / geom.n))
    return np.stack(feats, -1).transpose(1, 0, 2)


def add_agent_id(x):
    """Hängt die Agenten-ID als One-Hot an [k, B, F] an (für geteilte Parameter)."""
    k, batch, _ = x.shape
    one_hot = np.broadcast_to(np.eye(k)[:, None, :], (k, batch, k))
    return np.concatenate([x, one_hot], -1)


def central_features(geom, j, free, travel, job_pos, job_dur, agent_pos, all_pos, all_dur, scope):
    """Merkmale des ZENTRALEN Kritikers, Form [B, F]: Zustand aller Agenten + aktueller Auftrag.
    scope == "full" ergänzt die Positionen/Dauern der KÜNFTIGEN Aufträge - Wissen, das im Einsatz
    niemand hat (privilegiert, nur im Training)."""
    batch, _ = free.shape
    feats = [
        free / geom.fair, agent_pos / 20.0, travel / 10.0,
        (job_pos / 20.0)[:, None], (job_dur / 10.0)[:, None], np.full((batch, 1), j / geom.n),
    ]
    if scope == C.SCOPE_FULL:
        future = (np.arange(geom.n) > j)[None, :]
        feats.append(all_pos / 20.0 * future)
        feats.append(all_dur / 10.0 * future)
    return np.concatenate(feats, 1)


def local_bucket(geom, j, free, travel):
    """Zustands-Index wie `marl_obs.make_observer` (dieselben Buckets wie IQL), vektorisiert."""
    free_bucket = np.searchsorted(geom.free_edges, free, "right")
    travel_bucket = np.searchsorted(TRAVEL_EDGES, travel, "right")
    return (j * n_free_buckets() + free_bucket) * n_travel_buckets() + travel_bucket


def n_local_buckets(geom):
    return geom.n * n_free_buckets() * n_travel_buckets()


def global_bucket(geom, j, free):
    """Globaler Zustand für den Tabellen-Kritiker: Auftrags-Index + Freizeit-Buckets ALLER Agenten."""
    buckets = np.searchsorted(geom.free_edges, free, "right")
    state = np.full(free.shape[0], j)
    for a in range(geom.k):
        state = state * n_free_buckets() + buckets[:, a]
    return state


def n_global_buckets(geom):
    return geom.n * n_free_buckets() ** geom.k


def replay_dispatch(instance, actions):
    """actions[j][a] = Aktions-Index von Agent a bei Auftrag j. Läuft über die Vehikel-Funktionen und
    liefert ein `DispatchResult` (Gantt-fähig, mit Original- und abgegebenen Geboten)."""
    obs_fn, _ = make_observer(instance.n_jobs, instance.n_agents)
    per_job = n_free_buckets() * n_travel_buckets()
    return run_dispatch(instance, lambda agent, state: actions[state // per_job][agent], obs_fn)
