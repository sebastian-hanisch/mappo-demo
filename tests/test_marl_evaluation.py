import statistics

import cn_constants as C
from cn_scenario import generate_instance
from marl_env import heldout_instances
from marl_evaluation import (
    cnp_makespans, comparison, crossplay, generalisation_check, heldout_optima, learning_curve, pct_vs,
    policy_makespans, random_bias_makespans, seed_lottery,
)
from marl_iql import new_q_tables, train_iql
from marl_obs import make_observer


def test_pct_vs():
    assert pct_vs(90.0, 100.0) == -10.0
    assert pct_vs(120.0, 100.0) == 20.0
    assert pct_vs(5.0, 0.0) == 0.0


def test_untrained_policy_is_exactly_cnp_in_comparison():
    instance = generate_instance(6, 3, 0.3, 1.0, 4)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 4, n=6)
    _, n_states = make_observer(6, 3)
    q = new_q_tables(3, n_states)
    cmp = comparison(instance, q, heldout, cnp_makespans(heldout), [None] * 6, ortools_time_limit=5.0)
    assert cmp["iql_nominal"] == cmp["cnp_nominal"]
    assert cmp["heldout_iql_mean"] == cmp["heldout_cnp_mean"]
    assert cmp["iql_vs_cnp_nominal_pct"] == 0.0
    assert cmp["heldout_beat_frac"] == 0.0 and cmp["heldout_lose_frac"] == 0.0


def test_reference_never_above_any_feasible_schedule():
    instance = generate_instance(6, 3, 0.3, 1.0, 5)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 5, n=5)
    q = train_iql(instance, C.ENV_RECURRING, 0.3, 2000, 0, 0.3).q
    hc = cnp_makespans(heldout)
    cmp = comparison(instance, q, heldout, hc, heldout_optima(heldout, 5.0), ortools_time_limit=5.0)
    assert cmp["optimum_reference"] <= min(cmp["cnp_nominal"], cmp["iql_nominal"]) + 1e-9
    assert cmp["cnp_gap_pct"] >= -1e-9 and cmp["iql_gap_pct"] >= -1e-9
    assert cmp["heldout_opt_mean"] <= min(cmp["heldout_cnp_mean"], cmp["heldout_iql_mean"]) + 1e-9


def test_random_bias_worse_than_cnp_on_average():
    instance = generate_instance(8, 3, 0.3, 1.0, 5)
    ms = random_bias_makespans([instance] * 30, 8, 3, "t")
    assert statistics.fmean(ms) > cnp_makespans([instance])[0]


def test_learning_curve_has_all_checkpoints_and_final():
    instance = generate_instance(5, 2, 0.3, 1.0, 1)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 1, n=4)
    r = train_iql(instance, C.ENV_RECURRING, 0.3, 600, 0, 0.3, checkpoints=(100, 300))
    points = learning_curve(r, instance, heldout)
    assert [p[0] for p in points] == [100, 300, 600]


def test_generalisation_check_and_heldout_are_deterministic():
    instance = generate_instance(6, 3, 0.3, 1.0, 5)
    q = train_iql(instance, C.ENV_RECURRING, 0.3, 500, 0, 0.3).q
    assert generalisation_check(q, instance, 0.3, 5, n=6) == generalisation_check(q, instance, 0.3, 5, n=6)


def test_crossplay_diagonal_is_selfplay_and_matrix_shape():
    instance = generate_instance(5, 3, 0.3, 1.0, 2)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 2, n=5)
    q_list = [train_iql(instance, C.ENV_RECURRING, 0.3, 500, s, 0.3).q for s in range(3)]
    matrix = crossplay(q_list, heldout, 5, 3)
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    for i in range(3):
        self_play = statistics.fmean(policy_makespans(q_list[i], heldout, 5, 3))
        assert abs(matrix[i][i] - self_play) < 1e-9


def test_lottery_starts_at_selected_seed_and_is_capped():
    instance = generate_instance(5, 2, 0.3, 1.0, 2)
    heldout = heldout_instances(instance, C.ENV_RECURRING, 0.3, 0.3, 2, n=4)
    lot = seed_lottery(instance, C.ENV_RECURRING, 0.3, 10 ** 6, 7, 0.3, heldout, cnp_makespans(heldout), n_seeds=2)
    assert lot["seeds"] == [7, 8]
    assert lot["episodes"] == C.LOTTERY_EPISODE_CAP or lot["episodes"] < 10 ** 6
