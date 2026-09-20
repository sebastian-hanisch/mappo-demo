"""Die Beobachtung eines Agenten - streng LOKAL: nur der öffentliche Ankündigungs-Zähler
(welcher Auftrag gerade dran ist), die eigene Freizeit und die eigene Anfahrt zum
angekündigten Auftrag. Keine Gebote und keine Zustände anderer Agenten.

Gebucketet, damit die Q-Tabelle klein und tabellarisch bleibt: 4 Freizeit-Buckets x 3
Anfahrt-Buckets pro Auftrag => S = n_jobs * 12 Zustände (höchstens 120).

Die Auftragsdauer ist bewusst NICHT Teil der Beobachtung: alle Bieter sehen dieselbe
Dauer, sie kürzt sich im Gebotsvergleich heraus (im Prototyp auch empirisch ohne Wirkung)."""

import bisect

import cn_constants as C


def _edges(n_jobs, n_agents):
    fair_share = n_jobs * C.MEAN_DURATION / n_agents
    return tuple(f * fair_share for f in C.FREE_TIME_EDGE_FRACTIONS)


def n_free_buckets():
    return len(C.FREE_TIME_EDGE_FRACTIONS) + 1


def n_travel_buckets():
    return len(C.TRAVEL_EDGES) + 1


def make_observer(n_jobs, n_agents):
    """Gibt (obs_fn, n_states) zurück. obs_fn(free_time, travel_time, job_index) -> Zustands-Index."""
    free_edges = _edges(n_jobs, n_agents)
    travel_edges = C.TRAVEL_EDGES
    n_travel = n_travel_buckets()

    def obs_fn(free_time, travel_time, job_index):
        return (job_index * n_free_buckets() + bisect.bisect(free_edges, free_time)) * n_travel \
            + bisect.bisect(travel_edges, travel_time)

    return obs_fn, n_jobs * n_free_buckets() * n_travel


def describe_state(state, n_jobs, n_agents):
    """Lesbare Beschreibung eines Zustands-Index: (Auftrag 1-basiert, Freizeit-Text, Anfahrt-Text)."""
    n_travel = n_travel_buckets()
    travel_bucket = state % n_travel
    rest = state // n_travel
    free_bucket = rest % n_free_buckets()
    job_index = rest // n_free_buckets()

    free_edges = _edges(n_jobs, n_agents)
    free_labels = (
        [f"frei < {free_edges[0]:.0f} min"]
        + [f"frei {free_edges[i]:.0f}-{free_edges[i + 1]:.0f} min" for i in range(len(free_edges) - 1)]
        + [f"frei ≥ {free_edges[-1]:.0f} min"]
    )
    travel_edges = C.TRAVEL_EDGES
    travel_labels = (
        [f"Anfahrt < {travel_edges[0]:.0f} min"]
        + [f"Anfahrt {travel_edges[i]:.0f}-{travel_edges[i + 1]:.0f} min" for i in range(len(travel_edges) - 1)]
        + [f"Anfahrt ≥ {travel_edges[-1]:.0f} min"]
    )
    return job_index + 1, free_labels[free_bucket], travel_labels[travel_bucket]
