"""Das Gebots-Grundprinzip des Contract Net Protocol (Smith, 1980) - der eine
gelehrte Mechanismus dieser Demo. Jeder Agent bietet auf einen angekündigten
Auftrag mit seiner eigenen, rein lokalen Fertigstellungszeit: "wenn ich diesen
Auftrag ans Ende meiner eigenen Warteschlange hänge, wann bin ich fertig?"

Das ist bewusst die einfachste mögliche Gebotsformel (Cheapest-Insertion-am-Ende,
kein Umsortieren). Genau das macht die Zuteilung strukturell UNUMKEHRBAR - ein
Agent kann ein einmal angehängtes Gebot nie wieder lösen, selbst wenn ein späterer
Auftrag zeigt, dass eine andere Reihenfolge oder Zuteilung besser gewesen wäre.
Diese Schwäche zu beheben ist die Aufgabe der SPÄTEREN Stücke dieser Linie
(Kombinatorische Auktionen, DCOP, MARL - siehe project_multiagent_coordination_
dag_scoping.md), nicht dieser Demo."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentState:
    agent_id: int
    position: float
    free_time: float
    assigned_jobs: tuple = ()


@dataclass(frozen=True)
class Bid:
    agent_id: int
    travel_time: float
    finish_time: float


def compute_bid(instance, agent_state, job):
    """Ein Agent bietet die Fertigstellungszeit, die er selbst erreichen würde,
    wenn er `job` als NÄCHSTEN (letzten) Auftrag seiner eigenen Warteschlange
    übernimmt - reine lokale Information, kein Wissen über andere Agenten
    oder zukünftige Aufträge."""
    travel = instance.travel_time(agent_state.position, job.position)
    finish = agent_state.free_time + travel + job.duration
    return Bid(agent_id=agent_state.agent_id, travel_time=travel, finish_time=finish)


def award(bids):
    """Zuschlag an das niedrigste Gebot (Fertigstellungszeit); Gleichstand wird
    deterministisch über die niedrigste Agenten-ID aufgelöst (testbar, reproduzierbar)."""
    return min(bids, key=lambda b: (b.finish_time, b.agent_id))
