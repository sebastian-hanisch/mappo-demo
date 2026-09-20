import cn_constants as C
from cn_scenario import generate_instance
from marl_exact import miniature_instance
from marl_iql import epsilon_at, greedy_action, new_q_tables, q_update, train_iql
from marl_obs import make_observer


def test_q_update_terminal_hand_computed():
    # Q=0, alpha=0.5, terminale Belohnung -2 -> 0 + 0.5 * (-2 - 0) = -1
    assert q_update(0.0, -2.0, alpha=0.5) == -1.0


def test_q_update_nonterminal_hand_computed():
    # Q=-1, max Q(s') = -3, alpha=0.5 -> -1 + 0.5 * (-3 - (-1)) = -2
    assert q_update(-1.0, -3.0, alpha=0.5) == -2.0


def test_epsilon_schedule_anchor_points():
    n = 1000
    assert epsilon_at(0, n) == 1.0
    assert abs(epsilon_at(350, n) - 0.525) < 1e-12  # halber Weg über die ersten 70 %
    assert abs(epsilon_at(700, n) - C.EPSILON_END) < 1e-12
    assert epsilon_at(999, n) == C.EPSILON_END


def test_greedy_action_tie_breaks_to_action_zero():
    assert greedy_action([0.0, 0.0, 0.0]) == 0
    assert greedy_action([-1.0, -1.0, -0.5]) == 2
    assert greedy_action([-1.0, -0.5, -0.5]) == 1


def test_training_is_deterministic_and_seed_dependent():
    instance = generate_instance(6, 3, 0.3, 1.0, 5)
    a = train_iql(instance, C.ENV_RECURRING, 0.3, 500, 7, 0.3)
    b = train_iql(instance, C.ENV_RECURRING, 0.3, 500, 7, 0.3)
    c = train_iql(instance, C.ENV_RECURRING, 0.3, 500, 8, 0.3)
    assert a.q == b.q
    assert a.q != c.q


def test_training_seed_independent_of_scenario_seed_stream():
    # Der Trainings-Zufall hängt nur vom train_seed ab: gleiche Basisinstanz, gleicher train_seed
    # => identisches Q, egal welcher Szenario-Seed die Instanz erzeugt hat (hier: dieselbe Instanz).
    i1 = generate_instance(6, 2, 0.3, 1.0, 11)
    i2 = generate_instance(6, 2, 0.3, 1.0, 11)
    assert train_iql(i1, C.ENV_RECURRING, 0.3, 300, 3, 0.3).q == train_iql(i2, C.ENV_RECURRING, 0.3, 300, 3, 0.3).q


def test_snapshots_recorded_at_checkpoints():
    instance = generate_instance(5, 2, 0.3, 1.0, 1)
    r = train_iql(instance, C.ENV_RECURRING, 0.3, 400, 0, 0.3, checkpoints=(100, 300))
    assert set(r.snapshots) == {100, 300}
    assert r.snapshots[100] != r.q
    assert r.n_decisions == 400 * 5 * 2


def test_only_named_learners_learn():
    instance = generate_instance(5, 3, 0.3, 1.0, 1)
    r = train_iql(instance, C.ENV_RECURRING, 0.3, 300, 0, 0.3, learners=(0,))
    _, n_states = make_observer(5, 3)
    zero = new_q_tables(3, n_states)
    assert r.q[1] == zero[1] and r.q[2] == zero[2]
    assert r.q[0] != zero[0]


def test_iql_reaches_exact_optimum_on_2x2_miniature():
    instance = miniature_instance()
    for seed in range(10):
        r = train_iql(instance, C.ENV_RECURRING, 0.0, 1000, seed, 0.0)
        obs_fn, _ = make_observer(2, 2)
        from marl_env import rollout, to_fast
        from marl_iql import greedy_policy
        assert rollout(to_fast(instance), greedy_policy(r.q), obs_fn) == 15.0, f"train_seed={seed}"
