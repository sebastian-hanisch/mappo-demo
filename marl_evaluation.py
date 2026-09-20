"""Auswertung einer gelernten Policy - jede Zahl, die die App zeigt, kommt von hier:

- `comparison`: Contract Net / IQL / Zufalls-Bias / CP-SAT, nominal UND auf Held-out-Instanzen
- `learning_curve`: Policy-Güte an den Trainings-Checkpoints
- `generalisation_check`: dieselbe Policy auf FRISCHEN Szenarien (Memorieren vs. Lernen)
- `seed_lottery`/`crossplay`: Streuung über Trainings-Seeds und Cross-Play-Strafe
  (Nicht-Stationarität als gemessene Zahl)

Prozentangaben "vs. CNP": (IQL - CNP) / CNP * 100 - negativ heißt besser als Contract Net."""

import random
import statistics

import cn_constants as C
from cn_ortools_reference import solve_with_ortools
from cn_scenario import generate_instance
from marl_env import HELDOUT_SEED_OFFSET, cnp_makespan_fast, rollout, to_fast
from marl_iql import greedy_action, greedy_policy, train_iql
from marl_obs import make_observer

FRESH_SEED_OFFSET = HELDOUT_SEED_OFFSET + 500_000_000


def pct_vs(value, reference):
    return (value - reference) / reference * 100.0 if reference > 0 else 0.0


def cnp_makespans(instances):
    return [cnp_makespan_fast(to_fast(i)) for i in instances]


def heldout_optima(instances, time_limit_seconds=C.ORTOOLS_HELDOUT_TIME_LIMIT_SECONDS):
    """CP-SAT-Makespan je Instanz (None, wenn im Zeitlimit nichts gefunden)."""
    results = []
    for instance in instances:
        r = solve_with_ortools(instance, time_limit_seconds=time_limit_seconds)
        results.append(r.makespan if r.feasible else None)
    return results


def policy_makespans(q, instances, n_jobs, n_agents):
    obs_fn, _ = make_observer(n_jobs, n_agents)
    policy = greedy_policy(q)
    return [rollout(to_fast(i), policy, obs_fn) for i in instances]


def random_bias_makespans(instances, n_jobs, n_agents, seed_label):
    """Baseline: jeder Agent wählt seinen Gebots-Aufschlag ZUFÄLLIG - zeigt, dass der Lerngewinn
    nicht einfach von "irgendwelchen" Aufschlägen kommt."""
    obs_fn, _ = make_observer(n_jobs, n_agents)
    rng = random.Random(f"random-bias-{seed_label}")
    n_actions = len(C.BIAS_ACTIONS)
    policy = lambda agent, state: rng.randrange(n_actions)
    return [rollout(to_fast(i), policy, obs_fn) for i in instances]


def learning_curve(train_result, instance, heldout, cnp_heldout=None):
    """Punkte (episoden, nominal_makespan, heldout_mittel) für alle Zwischenstände + Endstand."""
    n, k = instance.n_jobs, instance.n_agents
    snapshots = dict(train_result.snapshots)
    snapshots[train_result.n_episodes] = train_result.q
    points = []
    for episodes in sorted(snapshots):
        q = snapshots[episodes]
        nominal = policy_makespans(q, [instance], n, k)[0]
        heldout_mean = statistics.fmean(policy_makespans(q, heldout, n, k))
        points.append((episodes, nominal, heldout_mean))
    return points


def comparison(instance, q, heldout, heldout_cnp, heldout_opt, ortools_time_limit=C.ORTOOLS_TIME_LIMIT_SECONDS):
    """Nominal (die gezeigte Instanz, unverrauscht) und Held-out. heldout_cnp/heldout_opt sind
    vorberechnet (unabhängig vom Training, deshalb separat gecacht)."""
    n, k = instance.n_jobs, instance.n_agents
    cnp_nominal = cnp_makespan_fast(to_fast(instance))
    iql_nominal = policy_makespans(q, [instance], n, k)[0]
    ortools = solve_with_ortools(instance, time_limit_seconds=ortools_time_limit)
    ortools_makespan = ortools.makespan if ortools.feasible else None

    # CP-SAT rundet Zeiten AUF (siehe cn_ortools_reference): sein Wert kann knapp über einem
    # tatsächlich erreichbaren Zeitplan liegen. Das echte Optimum ist <= jeder zulässigen
    # Lösung, deshalb wird die Referenz für Lücken nie größer als CNP oder IQL angesetzt.
    reference = None if ortools_makespan is None else min(ortools_makespan, cnp_nominal, iql_nominal)

    random_nominal = statistics.fmean(
        random_bias_makespans([instance] * C.N_RANDOM_BIAS_DRAWS, n, k, f"nominal-{n}-{k}")
    )

    iql_heldout = policy_makespans(q, heldout, n, k)
    ratios_pct = [pct_vs(i, c) for i, c in zip(iql_heldout, heldout_cnp)]
    heldout_cnp_mean = statistics.fmean(heldout_cnp)
    heldout_iql_mean = statistics.fmean(iql_heldout)
    # dieselbe Klemmung wie oben, je Instanz: Referenz = min(CP-SAT, CNP, IQL)
    if heldout_opt and all(o is not None for o in heldout_opt):
        heldout_opt_mean = statistics.fmean(min(o, c, i) for o, c, i in zip(heldout_opt, heldout_cnp, iql_heldout))
    else:
        heldout_opt_mean = None

    return {
        "cnp_nominal": cnp_nominal,
        "iql_nominal": iql_nominal,
        "random_bias_nominal": random_nominal,
        "ortools_nominal": ortools_makespan,
        "optimum_reference": reference,
        "ortools_feasible": ortools.feasible,
        "ortools_optimal": ortools.optimal,
        "ortools_wall_time": ortools.wall_time_ms / 1000.0,
        "iql_vs_cnp_nominal": iql_nominal - cnp_nominal,
        "iql_vs_cnp_nominal_pct": pct_vs(iql_nominal, cnp_nominal),
        "cnp_gap_pct": None if reference is None else pct_vs(cnp_nominal, reference),
        "iql_gap_pct": None if reference is None else pct_vs(iql_nominal, reference),
        "heldout_cnp_mean": heldout_cnp_mean,
        "heldout_iql_mean": heldout_iql_mean,
        "heldout_opt_mean": heldout_opt_mean,
        "heldout_iql_vs_cnp": heldout_iql_mean - heldout_cnp_mean,
        "heldout_iql_vs_cnp_pct": pct_vs(heldout_iql_mean, heldout_cnp_mean),
        "heldout_ratios_pct": ratios_pct,
        "heldout_beat_frac": sum(r < -1e-9 for r in ratios_pct) / len(ratios_pct),
        "heldout_lose_frac": sum(r > 1e-9 for r in ratios_pct) / len(ratios_pct),
        "heldout_worst_pct": max(ratios_pct),
        "heldout_best_pct": min(ratios_pct),
    }


def generalisation_check(q, instance, duration_variability, scenario_seed, n=C.N_FRESH_SCENARIOS):
    """Dieselbe Policy auf n FRISCHEN Szenarien gleicher Größe (neue Positionen/Dauern). Nur ein
    Vergleich gegen Contract Net - CP-SAT wird hier nicht gebraucht."""
    n_jobs, n_agents = instance.n_jobs, instance.n_agents
    fresh = [
        generate_instance(
            n_jobs, n_agents, duration_variability, instance.travel_time_per_unit,
            FRESH_SEED_OFFSET + scenario_seed * 1000 + i,
        )
        for i in range(n)
    ]
    cnp = cnp_makespans(fresh)
    iql = policy_makespans(q, fresh, n_jobs, n_agents)
    ratios = [pct_vs(i, c) for i, c in zip(iql, cnp)]
    return {
        "fresh_cnp_mean": statistics.fmean(cnp),
        "fresh_iql_mean": statistics.fmean(iql),
        "fresh_iql_vs_cnp_pct": pct_vs(statistics.fmean(iql), statistics.fmean(cnp)),
        "fresh_beat_frac": sum(r < -1e-9 for r in ratios) / len(ratios),
        "fresh_lose_frac": sum(r > 1e-9 for r in ratios) / len(ratios),
    }


def crossplay(q_list, instances, n_jobs, n_agents):
    """Matrix M[a][b]: mittlerer Makespan über `instances`, wenn Agent 0 die Policy aus Trainingslauf a
    spielt und alle anderen Agenten die aus Trainingslauf b. Die Diagonale ist das gemeinsam trainierte
    Team (Selbstspiel) - alle Policies sind gleich gut trainiert, der Vergleich ist konfundierungsfrei."""
    obs_fn, _ = make_observer(n_jobs, n_agents)
    fast = [to_fast(i) for i in instances]
    matrix = []
    for a, q_a in enumerate(q_list):
        row = []
        for b, q_b in enumerate(q_list):
            def policy(agent, state, q_a=q_a, q_b=q_b):
                return greedy_action((q_a if agent == 0 else q_b)[agent][state])
            row.append(statistics.fmean(rollout(fi, policy, obs_fn) for fi in fast))
        matrix.append(row)
    return matrix


def seed_lottery(
    instance, env_mode, sigma, episodes, first_seed, duration_variability, heldout, heldout_cnp,
    n_seeds=C.N_LOTTERY_SEEDS,
):
    """Trainiert n_seeds Läufe (Seeds first_seed, first_seed+1, ...) und misst pro Lauf nominal + Held-out,
    dazu Cross-Play. Der Trainingsaufwand ist gedeckelt (LOTTERY_EPISODE_CAP)."""
    n, k = instance.n_jobs, instance.n_agents
    episodes = min(episodes, C.LOTTERY_EPISODE_CAP)
    seeds = [first_seed + i for i in range(n_seeds)]
    q_list = [
        train_iql(instance, env_mode, sigma, episodes, s, duration_variability).q for s in seeds
    ]
    cnp_nominal = cnp_makespan_fast(to_fast(instance))
    nominal = [policy_makespans(q, [instance], n, k)[0] for q in q_list]
    heldout_means = [statistics.fmean(policy_makespans(q, heldout, n, k)) for q in q_list]
    cnp_heldout_mean = statistics.fmean(heldout_cnp)
    matrix = crossplay(q_list, heldout, n, k)

    diag = [matrix[i][i] for i in range(n_seeds)]
    off = [matrix[a][b] for a in range(n_seeds) for b in range(n_seeds) if a != b]
    heldout_pct = [pct_vs(m, cnp_heldout_mean) for m in heldout_means]
    return {
        "seeds": seeds,
        "episodes": episodes,
        "cnp_nominal": cnp_nominal,
        "nominal": nominal,
        "nominal_pct": [pct_vs(m, cnp_nominal) for m in nominal],
        "heldout_means": heldout_means,
        "heldout_pct": heldout_pct,
        "heldout_std_pct": statistics.pstdev(heldout_pct),
        "crossplay_matrix": matrix,
        "crossplay_penalty_pct": (statistics.fmean(off) - statistics.fmean(diag)) / cnp_heldout_mean * 100.0,
        "n_worse_than_cnp": sum(p > 0 for p in heldout_pct),
    }
