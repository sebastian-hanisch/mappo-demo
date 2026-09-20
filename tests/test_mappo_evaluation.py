import statistics
from dataclasses import replace

import numpy as np
import pytest

import cn_constants as C
from cn_scenario import generate_instance
from marl_env import heldout_instances
from marl_evaluation import pct_vs
from marl_iql import new_q_tables
from marl_obs import make_observer
from mappo_env import arrays_from_instances, cnp_makespans, geometry
from mappo_evaluation import (
    comparison, critic_lab, crossplay, generalisation_check, learning_curve, miniature_results, result_makespans,
    seed_lottery,
)
from mappo_ppo import greedy_act_fn, train_ppo


def _untrained(instance, actor_kind, method):
    result = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 1, 0, method, actor_kind)
    if actor_kind == C.ACTOR_TABLE:
        return replace(result, actor=np.zeros_like(result.actor))
    actor = [w.copy() for w in result.actor]
    actor[len(actor) // 2 - 1][:] = 0.0
    actor[-1][:] = 0.0
    return replace(result, actor=actor)


def test_untrained_learners_are_exactly_contract_net_in_comparison():
    instance = generate_instance(6, 3, 0.3, 1.0, 4)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 4, n=6)
    _, n_states = make_observer(6, 3)
    q = new_q_tables(3, n_states)
    results = {m: _untrained(instance, C.ACTOR_NET, m) for m in (C.METHOD_IPPO, C.METHOD_MAPPO)}
    cmp = comparison(instance, heldout, cnp_makespans(geometry(instance), *arrays_from_instances(heldout)),
                     [None] * 6, q, results, ortools_time_limit=5.0)
    for name, stats in cmp["learners"].items():
        assert stats["nominal_pct"] == 0.0 and stats["heldout_pct"] == 0.0, name
        assert stats["beat_frac"] == 0.0 and stats["lose_frac"] == 0.0, name


def test_reference_never_above_any_feasible_schedule_and_gaps_nonnegative():
    from marl_evaluation import heldout_optima
    from marl_iql import train_iql

    instance = generate_instance(6, 3, 0.3, 1.0, 5)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 5, n=5)
    geom = geometry(instance)
    hc = cnp_makespans(geom, *arrays_from_instances(heldout))
    q = train_iql(instance, C.ENV_RECURRING, 0.3, 2000, 0, 0.3).q
    results = {
        m: train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 2000, 0, m, C.ACTOR_NET)
        for m in (C.METHOD_IPPO, C.METHOD_MAPPO)
    }
    cmp = comparison(instance, heldout, hc, heldout_optima(heldout, 5.0), q, results, ortools_time_limit=5.0)
    smallest = min([cmp["cnp_nominal"]] + [s["nominal"] for s in cmp["learners"].values()])
    assert cmp["optimum_reference"] <= smallest + 1e-9
    assert cmp["cnp_gap_pct"] >= -1e-9 and all(g >= -1e-9 for g in cmp["gap_pct"].values())
    assert cmp["heldout_opt_mean"] <= min(
        [cmp["heldout_cnp_mean"]] + [s["heldout_mean"] for s in cmp["learners"].values()]
    ) + 1e-9


def test_learning_curve_has_all_checkpoints_and_final():
    instance = generate_instance(5, 2, 0.3, 1.0, 1)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 1, n=4)
    r = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 600, 0, C.METHOD_MAPPO, C.ACTOR_NET, checkpoints=(100, 300))
    assert [p[0] for p in learning_curve(r, instance, heldout)] == [100, 300, 600]


def test_generalisation_check_is_deterministic_and_reports_fractions():
    instance = generate_instance(6, 3, 0.3, 1.0, 5)
    r = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 600, 0, C.METHOD_MAPPO, C.ACTOR_NET)
    a = generalisation_check(r, instance, 0.3, 5, n=6)
    assert a == generalisation_check(r, instance, 0.3, 5, n=6)
    assert 0.0 <= a["fresh_beat_frac"] <= 1.0 and 0.0 <= a["fresh_lose_frac"] <= 1.0


def test_crossplay_diagonal_is_selfplay_and_shape():
    instance = generate_instance(5, 3, 0.3, 1.0, 2)
    geom = geometry(instance)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 2, n=5)
    hpos, hdur = arrays_from_instances(heldout)
    results = [train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 300, s, C.METHOD_MAPPO, C.ACTOR_NET) for s in range(3)]
    matrix = crossplay(geom, [greedy_act_fn(geom, r) for r in results], hpos, hdur)
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    for i in range(3):
        assert abs(matrix[i][i] - statistics.fmean(result_makespans(geom, results[i], hpos, hdur))) < 1e-9


def test_lottery_structure_seeds_and_cap():
    instance = generate_instance(5, 2, 0.3, 1.0, 2)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 2, n=4)
    hc = cnp_makespans(geometry(instance), *arrays_from_instances(heldout))
    lot = seed_lottery(
        instance, C.ENV_RECURRING, 0.3, 0.3, 10 ** 6, 7, heldout, hc, C.ACTOR_TABLE, C.SCOPE_JOINT, C.PPO_CLIP, True,
        n_seeds=2,
    )
    assert lot["seeds"] == [7, 8] and lot["episodes"] == C.LOTTERY_EPISODE_CAP
    for method in ("iql", C.METHOD_IPPO, C.METHOD_MAPPO):
        assert len(lot[method]["heldout_pct"]) == 2 and len(lot[method]["crossplay_matrix"]) == 2


def test_critic_lab_ordering_local_below_joint_below_privileged():
    """Erklärte Varianz auf identischen Trajektorien: lokal < zentral < zentral + Auftragsliste."""
    instance = generate_instance(8, 3, 0.3, 1.0, 4)
    r = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 5000, 0, C.METHOD_MAPPO, C.ACTOR_NET)
    ev = critic_lab(instance, C.ENV_RECURRING, 0.3, 0.3, r)
    assert ev["local"] < ev["joint"] < ev["full"], ev
    assert 0.0 < ev["local"] < 1.0 and ev["full"] > 0.5


def test_miniature_all_methods_reach_the_optimum():
    """2x2-Miniatur: IPPO und MAPPO (Netz) finden das Optimum 15 in >= 18/20 Seeds, Tabellen in 20/20 -
    die Miniatur ist zu klein, um einen Kritiker-Effekt zu zeigen."""
    out = miniature_results(n_seeds=20)
    hits = out["optimum_hits"]
    assert hits["IPPO (Netz)"] >= 18 and hits["MAPPO (Netz)"] >= 18
    assert hits["IPPO (Tabelle)"] == 20 and hits["MAPPO (Tabelle)"] == 20
    assert hits["IQL"] >= 18


# --- Presets ------------------------------------------------------------------------------

def _preset_setup(params):
    instance = generate_instance(
        params["n_jobs"], params["n_agents"], params["duration_variability"], params["travel_time_per_unit"],
        params["seed"],
    )
    sigma = params["sigma"] if params["env_mode"] == C.ENV_RECURRING else 0.0
    heldout = heldout_instances(instance, params["env_mode"], sigma, params["duration_variability"], params["seed"])
    return instance, sigma, heldout


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_presets_produce_expected_bands(name):
    params = C.PRESETS[name]
    instance, sigma, heldout = _preset_setup(params)
    geom = geometry(instance)
    result = train_ppo(
        instance, params["env_mode"], sigma, params["duration_variability"], params["episodes"], params["train_seed"],
        params["method"], params["actor"], params["scope"], params["clip"], params["use_job_index"],
    )
    pos, dur = arrays_from_instances([instance])
    hpos, hdur = arrays_from_instances(heldout)
    nominal = pct_vs(result_makespans(geom, result, pos, dur)[0], cnp_makespans(geom, pos, dur)[0])
    heldout_pct = pct_vs(result_makespans(geom, result, hpos, hdur).mean(), cnp_makespans(geom, hpos, hdur).mean())
    band = C.PRESET_EXPECTED_BANDS[name]
    assert band["nominal_pct"][0] <= nominal <= band["nominal_pct"][1], f"{name}: nominal_pct={nominal:.1f}"
    assert band["heldout_pct"][0] <= heldout_pct <= band["heldout_pct"][1], f"{name}: heldout_pct={heldout_pct:.1f}"


def _lottery(name, **kwargs):
    params = C.PRESETS[name]
    instance, sigma, heldout = _preset_setup(params)
    hc = cnp_makespans(geometry(instance), *arrays_from_instances(heldout))
    return seed_lottery(
        instance, params["env_mode"], sigma, params["duration_variability"], params["episodes"], params["train_seed"],
        heldout, hc, params["actor"], params["scope"], params["clip"], params["use_job_index"], **kwargs,
    )


def test_lottery_preset_ppo_is_stable_and_iql_is_not():
    lot = _lottery("PPO stabilisiert die Lotterie")
    assert lot["iql"]["n_worse_than_cnp"] >= 3 and lot["iql"]["heldout_std_pct"] > 4.0
    for method in (C.METHOD_IPPO, C.METHOD_MAPPO):
        assert lot[method]["n_worse_than_cnp"] == 0 and lot[method]["heldout_std_pct"] < 1.5, method


def test_clipping_off_brings_instability_and_crossplay_penalty_back():
    on = _lottery("IPPO = MAPPO", include_iql=False, methods=(C.METHOD_MAPPO,))
    off = _lottery("Ohne Clipping", include_iql=False, methods=(C.METHOD_MAPPO,))
    assert on[C.METHOD_MAPPO]["heldout_std_pct"] < 1.0
    assert off[C.METHOD_MAPPO]["heldout_std_pct"] >= 3.0
    assert off[C.METHOD_MAPPO]["crossplay_penalty_pct"] >= 8.0
