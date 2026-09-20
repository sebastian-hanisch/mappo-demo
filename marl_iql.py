"""Independent Q-Learning (IQL): jeder Agent lernt seine EIGENE Q-Tabelle q[agent][zustand][aktion]
und behandelt die anderen Agenten als Teil der Umgebung - obwohl diese gleichzeitig ihre eigenen
Policies ändern (Nicht-Stationarität, siehe README). Alle Agenten teilen sich eine Belohnung:
-Makespan / REWARD_SCALE, erst am Ende einer Episode (dünn - echtes Credit-Assignment).

Tabellarisch und numpy-frei bis auf den Rest der Demo: reine Python-Listen sind für diese
Größenordnung (<= 120 Zustände, 3 Aktionen) schneller als numpy-Skalar-Zugriffe."""

import random
import time
from dataclasses import dataclass

import cn_constants as C
from marl_env import make_episode_source, rollout
from marl_obs import make_observer


def epsilon_at(episode, n_episodes):
    """ε fällt linear von 1.0 auf EPSILON_END über die ersten EPSILON_DECAY_FRACTION der Episoden."""
    decay_episodes = C.EPSILON_DECAY_FRACTION * n_episodes
    if decay_episodes <= 0:
        return C.EPSILON_END
    return max(C.EPSILON_END, 1.0 - (1.0 - C.EPSILON_END) * episode / decay_episodes)


def q_update(q_value, target, alpha=C.ALPHA):
    """Ein Q-Learning-Schritt: q += alpha * (target - q). target = Belohnung (terminal) bzw.
    max_a' Q(s', a') (γ=1, keine Zwischenbelohnung)."""
    return q_value + alpha * (target - q_value)


def new_q_tables(n_agents, n_states, n_actions=len(C.BIAS_ACTIONS)):
    return [[[0.0] * n_actions for _ in range(n_states)] for _ in range(n_agents)]


def greedy_action(q_row):
    """Argmax mit Tie-Break auf den niedrigsten Index - Aktion 0 ("normal") gewinnt Gleichstände,
    deshalb verhält sich eine untrainierte Policy exakt wie Contract Net."""
    best = 0
    best_value = q_row[0]
    for i in range(1, len(q_row)):
        if q_row[i] > best_value:
            best, best_value = i, q_row[i]
    return best


def greedy_policy(q):
    return lambda agent, state: greedy_action(q[agent][state])


def copy_q(q):
    return [[list(row) for row in agent_table] for agent_table in q]


@dataclass(frozen=True)
class TrainResult:
    q: list            # q[agent][zustand][aktion]
    snapshots: dict    # episoden-zahl -> q-Kopie (Zwischenstände)
    n_episodes: int
    n_decisions: int   # Agenten-Entscheidungen insgesamt (Episoden * Aufträge * Agenten)
    wall_time_s: float


def train_iql(
    base_instance, env_mode, sigma, n_episodes, train_seed, duration_variability=0.0,
    checkpoints=(), learners=None, biases=C.BIAS_ACTIONS,
):
    """Trainiert IQL. `learners`: Agenten-IDs, die lernen (Default: alle); die übrigen bieten
    immer normal. `checkpoints`: Episoden-Zahlen, deren Q-Zwischenstand in `snapshots` landet."""
    started = time.perf_counter()
    k, n = base_instance.n_agents, base_instance.n_jobs
    obs_fn, n_states = make_observer(n, k)
    n_actions = len(biases)
    q = new_q_tables(k, n_states, n_actions)
    learner_set = set(range(k)) if learners is None else set(learners)
    next_episode = make_episode_source(env_mode, base_instance, sigma, duration_variability, train_seed)
    explore_rng = random.Random(f"iql-explore-{train_seed}")
    checkpoint_set = set(checkpoints)
    snapshots = {}

    def policy_factory(epsilon):
        def policy(agent, state):
            if agent not in learner_set:
                return 0
            if explore_rng.random() < epsilon:
                return explore_rng.randrange(n_actions)
            return greedy_action(q[agent][state])
        return policy

    for episode in range(n_episodes):
        fi = next_episode()
        makespan, traj = rollout(fi, policy_factory(epsilon_at(episode, n_episodes)), obs_fn, biases, record=True)
        reward = -makespan / C.REWARD_SCALE
        last = len(traj) - 1
        for t in range(last, -1, -1):
            states, actions = traj[t]
            for agent in learner_set:
                s, a = states[agent], actions[agent]
                target = reward if t == last else max(q[agent][traj[t + 1][0][agent]])
                q[agent][s][a] = q_update(q[agent][s][a], target)
        if episode + 1 in checkpoint_set:
            snapshots[episode + 1] = copy_q(q)

    return TrainResult(
        q=q, snapshots=snapshots, n_episodes=n_episodes, n_decisions=n_episodes * n * k,
        wall_time_s=time.perf_counter() - started,
    )
