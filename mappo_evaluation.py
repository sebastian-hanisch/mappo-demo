"""Auswertung der PPO-Verfahren (IPPO/MAPPO) gegen Contract Net, den IQL-Referenzlauf aus marl-demo und
das zentrale CP-SAT-Optimum - jede Zahl der App kommt von hier:

- `comparison`: nominal (gezeigte Instanz) und Held-out für CNP, IQL, IPPO, MAPPO, CP-SAT
- `learning_curve`, `generalisation_check`
- `crossplay`, `seed_lottery`: Streuung über Trainings-Seeds und Cross-Play-Strafe je Verfahren
- `critic_lab`: was der zentrale Kritiker tatsächlich ändert - erklärte Varianz des Returns auf
  IDENTISCHEN Trajektorien für lokalen, zentralen und (privilegiert) zentralen Kritiker + Auftragsliste
- `miniature_results`: die 2x2-Miniatur mit allen Verfahren

Prozentangaben "vs. CNP": (Verfahren - CNP) / CNP * 100 - negativ heißt besser als Contract Net."""

import statistics

import numpy as np

import cn_constants as C
from cn_ortools_reference import solve_with_ortools
from cn_scenario import generate_instance
from marl_evaluation import FRESH_SEED_OFFSET, pct_vs
from marl_evaluation import crossplay as iql_crossplay
from marl_evaluation import generalisation_check as iql_generalisation_check
from marl_evaluation import policy_makespans as iql_policy_makespans
from marl_exact import miniature_instance
from marl_iql import train_iql
from mappo_env import Sampler, arrays_from_instances, central_features, cnp_makespans, geometry, local_features, simulate
from mappo_nets import MLP, Adam
from mappo_ppo import greedy_act_fn, stochastic_act_fn, train_ppo


def result_makespans(geom, result, pos, dur, actor_params=None):
    return simulate(geom, pos, dur, greedy_act_fn(geom, result, actor_params))


def _stats(nominal, heldout_ms, cnp_nominal, heldout_cnp):
    ratios = [pct_vs(m, c) for m, c in zip(heldout_ms, heldout_cnp)]
    heldout_mean = float(np.mean(heldout_ms))
    return {
        "nominal": float(nominal),
        "nominal_pct": pct_vs(nominal, cnp_nominal),
        "heldout_mean": heldout_mean,
        "heldout_pct": pct_vs(heldout_mean, float(np.mean(heldout_cnp))),
        "ratios_pct": ratios,
        "beat_frac": sum(r < -1e-9 for r in ratios) / len(ratios),
        "lose_frac": sum(r > 1e-9 for r in ratios) / len(ratios),
        "worst_pct": max(ratios),
    }


def comparison(instance, heldout, heldout_cnp, heldout_opt, iql_q, ppo_results, ortools_time_limit=C.ORTOOLS_TIME_LIMIT_SECONDS):
    """Alle Verfahren nominal und Held-out. `ppo_results`: dict name -> PPOResult (z.B. "ippo", "mappo")."""
    n, k = instance.n_jobs, instance.n_agents
    geom = geometry(instance)
    pos, dur = arrays_from_instances([instance])
    hpos, hdur = arrays_from_instances(heldout)
    cnp_nominal = float(cnp_makespans(geom, pos, dur)[0])
    heldout_cnp = list(heldout_cnp)

    learners = {}
    heldout_ms = {"iql": iql_policy_makespans(iql_q, heldout, n, k)}
    learners["iql"] = _stats(
        iql_policy_makespans(iql_q, [instance], n, k)[0], heldout_ms["iql"], cnp_nominal, heldout_cnp,
    )
    for name, result in ppo_results.items():
        heldout_ms[name] = result_makespans(geom, result, hpos, hdur)
        learners[name] = _stats(
            result_makespans(geom, result, pos, dur)[0], heldout_ms[name], cnp_nominal, heldout_cnp,
        )

    ortools = solve_with_ortools(instance, time_limit_seconds=ortools_time_limit)
    ortools_makespan = ortools.makespan if ortools.feasible else None
    # CP-SAT rundet Zeiten AUF: sein Wert kann knapp über einem erreichbaren Zeitplan liegen. Das echte
    # Optimum ist <= jeder zulässigen Lösung - die Referenz wird deshalb nie größer als CNP oder ein
    # Lernverfahren angesetzt (nominal wie je Held-out-Instanz).
    reference = None
    if ortools_makespan is not None:
        reference = min([ortools_makespan, cnp_nominal] + [s["nominal"] for s in learners.values()])
    if heldout_opt and all(o is not None for o in heldout_opt):
        heldout_opt_mean = float(np.mean([
            min([o, c] + [heldout_ms[name][i] for name in learners])
            for i, (o, c) in enumerate(zip(heldout_opt, heldout_cnp))
        ]))
    else:
        heldout_opt_mean = None

    return {
        "cnp_nominal": cnp_nominal,
        "ortools_nominal": ortools_makespan,
        "optimum_reference": reference,
        "ortools_feasible": ortools.feasible,
        "ortools_optimal": ortools.optimal,
        "ortools_wall_time": ortools.wall_time_ms / 1000.0,
        "cnp_gap_pct": None if reference is None else pct_vs(cnp_nominal, reference),
        "heldout_cnp_mean": float(np.mean(heldout_cnp)),
        "heldout_opt_mean": heldout_opt_mean,
        "learners": learners,
        "gap_pct": {
            name: (None if reference is None else pct_vs(s["nominal"], reference)) for name, s in learners.items()
        },
    }


def learning_curve(result, instance, heldout):
    """Punkte (episoden, nominal, heldout_mittel) für alle Zwischenstände + Endstand."""
    geom = geometry(instance)
    pos, dur = arrays_from_instances([instance])
    hpos, hdur = arrays_from_instances(heldout)
    snapshots = dict(result.snapshots)
    snapshots[result.n_episodes] = result.actor
    return [
        (ep, float(result_makespans(geom, result, pos, dur, params)[0]),
         float(result_makespans(geom, result, hpos, hdur, params).mean()))
        for ep, params in sorted(snapshots.items())
    ]


def generalisation_check(result, instance, duration_variability, scenario_seed, n=C.N_FRESH_SCENARIOS):
    """Dieselbe Policy auf n FRISCHEN Szenarien gleicher Größe (neue Positionen/Dauern) gegen Contract Net."""
    fresh = [
        generate_instance(
            instance.n_jobs, instance.n_agents, duration_variability, instance.travel_time_per_unit,
            FRESH_SEED_OFFSET + scenario_seed * 1000 + i,
        )
        for i in range(n)
    ]
    geom = geometry(instance)
    pos, dur = arrays_from_instances(fresh)
    cnp = cnp_makespans(geom, pos, dur)
    ms = result_makespans(geom, result, pos, dur)
    ratios = [pct_vs(m, c) for m, c in zip(ms, cnp)]
    return {
        "fresh_vs_cnp_pct": pct_vs(float(ms.mean()), float(cnp.mean())),
        "fresh_beat_frac": sum(r < -1e-9 for r in ratios) / len(ratios),
        "fresh_lose_frac": sum(r > 1e-9 for r in ratios) / len(ratios),
    }


def iql_generalisation(q, instance, duration_variability, scenario_seed, n=C.N_FRESH_SCENARIOS):
    """Dieselbe Rückgabe wie `generalisation_check`, für den IQL-Referenzlauf (marl_evaluation)."""
    raw = iql_generalisation_check(q, instance, duration_variability, scenario_seed, n)
    return {
        "fresh_vs_cnp_pct": raw["fresh_iql_vs_cnp_pct"],
        "fresh_beat_frac": raw["fresh_beat_frac"], "fresh_lose_frac": raw["fresh_lose_frac"],
    }


def crossplay(geom, act_fns, pos, dur):
    """Matrix M[a][b]: mittlerer Makespan, wenn Agent 0 die Policy aus Trainingslauf a spielt und alle
    anderen Agenten die aus Lauf b. Diagonale = gemeinsam trainiertes Team. Bei geteilten Parametern
    (Netz) heißt das: Netz a entscheidet für Agenten-ID 0, Netz b für alle übrigen."""
    size = len(act_fns)
    matrix = np.zeros((size, size))
    for a in range(size):
        for b in range(size):
            def act_fn(j, free, apos, travel, jp, jd, a=a, b=b):
                mixed = act_fns[b](j, free, apos, travel, jp, jd).copy()
                mixed[:, 0] = act_fns[a](j, free, apos, travel, jp, jd)[:, 0]
                return mixed
            matrix[a, b] = simulate(geom, pos, dur, act_fn).mean()
    return matrix.tolist()


def _lottery_entry(nominal_ms, heldout_means, matrix, cnp_nominal, cnp_heldout_mean):
    size = len(nominal_ms)
    diag = [matrix[i][i] for i in range(size)]
    off = [matrix[a][b] for a in range(size) for b in range(size) if a != b]
    heldout_pct = [pct_vs(m, cnp_heldout_mean) for m in heldout_means]
    return {
        "nominal_pct": [pct_vs(m, cnp_nominal) for m in nominal_ms],
        "heldout_pct": heldout_pct,
        "heldout_std_pct": statistics.pstdev(heldout_pct),
        "crossplay_matrix": matrix,
        "crossplay_penalty_pct": (statistics.fmean(off) - statistics.fmean(diag)) / cnp_heldout_mean * 100.0,
        "n_worse_than_cnp": sum(p > C.LOTTERY_WORSE_TOLERANCE_PCT for p in heldout_pct),
    }


def seed_lottery(
    instance, env_mode, sigma, duration_variability, episodes, first_seed, heldout, heldout_cnp,
    actor_kind, scope, clip, use_job_index, n_seeds=C.N_LOTTERY_SEEDS, methods=(C.METHOD_IPPO, C.METHOD_MAPPO),
    include_iql=True,
):
    """Trainiert IQL, IPPO und MAPPO je n_seeds Mal (Seeds first_seed, first_seed+1, ...) mit derselben
    (gedeckelten) Episodenzahl und misst nominal, Held-out und Cross-Play je Verfahren."""
    n, k = instance.n_jobs, instance.n_agents
    geom = geometry(instance)
    episodes = min(episodes, C.LOTTERY_EPISODE_CAP)
    seeds = [first_seed + i for i in range(n_seeds)]
    pos, dur = arrays_from_instances([instance])
    hpos, hdur = arrays_from_instances(heldout)
    cnp_nominal = float(cnp_makespans(geom, pos, dur)[0])
    cnp_heldout_mean = float(np.mean(heldout_cnp))
    out = {"seeds": seeds, "episodes": episodes, "cnp_nominal": cnp_nominal}

    if include_iql:
        q_list = [train_iql(instance, env_mode, sigma, episodes, s, duration_variability).q for s in seeds]
        out["iql"] = _lottery_entry(
            [iql_policy_makespans(q, [instance], n, k)[0] for q in q_list],
            [statistics.fmean(iql_policy_makespans(q, heldout, n, k)) for q in q_list],
            iql_crossplay(q_list, heldout, n, k), cnp_nominal, cnp_heldout_mean,
        )
    for method in methods:
        results = [
            train_ppo(
                instance, env_mode, sigma, duration_variability, episodes, s, method, actor_kind, scope, clip,
                use_job_index,
            )
            for s in seeds
        ]
        out[method] = _lottery_entry(
            [float(result_makespans(geom, r, pos, dur)[0]) for r in results],
            [float(result_makespans(geom, r, hpos, hdur).mean()) for r in results],
            crossplay(geom, [greedy_act_fn(geom, r) for r in results], hpos, hdur), cnp_nominal, cnp_heldout_mean,
        )
    return out


# --- Kritiker-Vermessung ---------------------------------------------------------------------

def _collect_trajectories(geom, result, sampler, episodes, rng, batch=500):
    """Sammelt Trajektorien der (stochastischen) Policy: lokale Merkmale je Agent, zentrale Merkmale
    (ohne/mit Auftragsliste) und die Belohnung. Für ALLE Kritiker dieselben Daten."""
    k, n = geom.k, geom.n
    local, joint, full, rewards = [], [], [], []
    for _ in range(episodes // batch):
        pos, dur = sampler.sample(batch)
        f_local = np.zeros((batch, n, k, 5 + (1 if result.use_job_index else 0)))
        f_joint = f_full = None
        base_act = stochastic_act_fn(geom, result, rng)
        store = {}

        def act_fn(j, free, apos, travel, jp, jd):
            f_local[:, j] = local_features(geom, j, free, travel, jp, jd, apos, result.use_job_index).transpose(1, 0, 2)
            cj = central_features(geom, j, free, travel, jp, jd, apos, pos, dur, C.SCOPE_JOINT)
            cf = central_features(geom, j, free, travel, jp, jd, apos, pos, dur, C.SCOPE_FULL)
            if "joint" not in store:
                store["joint"] = np.zeros((batch, n, cj.shape[-1]))
                store["full"] = np.zeros((batch, n, cf.shape[-1]))
            store["joint"][:, j] = cj
            store["full"][:, j] = cf
            return base_act(j, free, apos, travel, jp, jd)

        makespan = simulate(geom, pos, dur, act_fn)
        local.append(f_local)
        joint.append(store["joint"])
        full.append(store["full"])
        rewards.append(-makespan / C.REWARD_SCALE)
    return np.concatenate(local), np.concatenate(joint), np.concatenate(full), np.concatenate(rewards)


def _fit_and_score(x_train, r_train, x_test, r_test, groups, n, steps, hidden, seed):
    """Fittet einen Kritiker (groups=k: je Agent ein eigenes Netz; groups=1: ein Netz) auf den Return und
    gibt die erklärte Varianz auf den Testdaten zurück."""
    rng = np.random.default_rng(seed)
    features = x_train.shape[-1]
    net = MLP(groups, [features, hidden, hidden, 1], rng, last_zero=True)
    opt = Adam(net.params(), 3e-3)
    mu, sd = r_train.mean(), r_train.std()
    for _ in range(steps):
        ii = rng.integers(0, x_train.shape[0], 512)
        jj = rng.integers(0, n, 512)
        if groups == 1:
            x = x_train[ii, jj][None]
            y = ((r_train[ii] - mu) / sd)[None, :, None]
        else:
            x = x_train[ii, jj].transpose(1, 0, 2)
            y = np.broadcast_to(((r_train[ii] - mu) / sd)[None, :, None], (groups, 512, 1))
        hs = net.forward(x)
        opt.step(net.backward(hs, 2 * (hs[-1] - y) / hs[-1].size), 0.0)
    if groups == 1:
        pred = net.forward(x_test.reshape(1, -1, features))[-1].reshape(x_test.shape[0], n) * sd + mu
        err = ((r_test[:, None] - pred) ** 2).mean()
    else:
        pred = net.forward(x_test.transpose(2, 0, 1, 3).reshape(groups, -1, features))[-1].reshape(
            groups, x_test.shape[0], n) * sd + mu
        err = ((r_test[None, :, None] - pred) ** 2).mean()
    return float(1.0 - err / r_test.var())


def critic_lab(instance, env_mode, sigma, duration_variability, result, seed=0):
    """Erklärte Varianz des Returns für drei Kritiker auf IDENTISCHEN Trajektorien derselben Policy:
    lokal (je Agent), zentral (alle Agenten + aktueller Auftrag) und zentral + Auftragsliste (privilegiert).
    Zeigt, was der zentrale Kritiker ändert (die Varianz der Vorteilsschätzung) - und was nicht."""
    geom = geometry(instance)
    train = _collect_trajectories(
        geom, result, Sampler(env_mode, instance, sigma, duration_variability, 9001),
        C.CRITIC_LAB_TRAIN_EPISODES, np.random.default_rng([seed, 5]),
    )
    test = _collect_trajectories(
        geom, result, Sampler(env_mode, instance, sigma, duration_variability, 9002),
        C.CRITIC_LAB_TEST_EPISODES, np.random.default_rng([seed, 6]),
    )
    steps, hidden, n = C.CRITIC_LAB_STEPS, C.CRITIC_LAB_HIDDEN, geom.n
    return {
        "local": _fit_and_score(train[0], train[3], test[0], test[3], geom.k, n, steps, hidden, seed),
        "joint": _fit_and_score(train[1], train[3], test[1], test[3], 1, n, steps, hidden, seed),
        "full": _fit_and_score(train[2], train[3], test[2], test[3], 1, n, steps, hidden, seed),
    }


# --- 2x2-Miniatur ------------------------------------------------------------------------------

def miniature_results(n_seeds=C.MINIATURE_SEEDS, episodes=C.MINIATURE_EPISODES):
    """Wie viele Trainings-Seeds erreichen auf der 2x2-Miniatur (Optimum 15 min) den optimalen Zeitplan -
    für IQL und alle PPO-Varianten. Die Miniatur ist zu klein, um einen Kritiker-Effekt zu zeigen."""
    instance = miniature_instance()
    geom = geometry(instance)
    pos, dur = arrays_from_instances([instance])
    optimum = 15.0
    rows = {}
    hits = 0
    for s in range(n_seeds):
        q = train_iql(instance, C.ENV_RECURRING, 0.0, episodes, s, 0.0).q
        hits += abs(iql_policy_makespans(q, [instance], 2, 2)[0] - optimum) < 1e-9
    rows["IQL"] = hits
    for actor_kind in (C.ACTOR_NET, C.ACTOR_TABLE):
        for method in (C.METHOD_IPPO, C.METHOD_MAPPO):
            hits = 0
            for s in range(n_seeds):
                r = train_ppo(instance, C.ENV_RECURRING, 0.0, 0.0, episodes, s, method, actor_kind)
                hits += abs(float(result_makespans(geom, r, pos, dur)[0]) - optimum) < 1e-9
            rows[f"{method.upper()} ({C.ACTOR_LABELS[actor_kind].split(' ')[0]})"] = hits
    return {"n_seeds": n_seeds, "episodes": episodes, "optimum_hits": rows}
