import numpy as np
import pytest

import cn_constants as C
from cn_scenario import generate_instance
from mappo_env import arrays_from_instances, cnp_makespans, geometry, simulate
from mappo_nets import MLP, Adam, softmax
from mappo_ppo import (
    gae, greedy_act_fn, normalize, ppo_logit_grad, ppo_loss, ppo_objective, table_value_update, train_ppo,
)


# --- GAE ------------------------------------------------------------------------------

def test_gae_hand_computed_lambda_half():
    values = np.array([[1.0, 2.0, 3.0]])
    adv = gae(values, np.array([10.0]), lam=0.5)
    # delta = [V1-V0, V2-V1, R-V2] = [1, 1, 7]; A2 = 7, A1 = 1 + .5*7 = 4.5, A0 = 1 + .5*4.5 = 3.25
    assert np.allclose(adv, [[3.25, 4.5, 7.0]])


def test_gae_lambda_one_is_monte_carlo_and_lambda_zero_is_td():
    values = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(gae(values, np.array([10.0]), lam=1.0), [[9.0, 8.0, 7.0]])  # R - V_t
    assert np.allclose(gae(values, np.array([10.0]), lam=0.0), [[1.0, 1.0, 7.0]])  # einzelne TD-Fehler


def test_gae_single_step_is_reward_minus_value():
    values = np.array([[4.0], [1.0]])
    reward = np.array([10.0, 3.0])
    adv = gae(values, reward, lam=0.9)
    assert np.allclose(adv[:, 0], reward - values[:, 0])
    assert np.allclose(adv[:, 0] + values[:, 0], reward)  # Kritiker-Ziel = Belohnung


def test_gae_handles_per_agent_values():
    values = np.zeros((2, 3, 4))
    values[:, :, 2] = 1.0
    adv = gae(values, np.array([5.0, 6.0]), lam=1.0)
    assert adv.shape == (2, 3, 4)
    assert np.allclose(adv[0, :, 0], 5.0) and np.allclose(adv[0, :, 2], 4.0)


# --- PPO-Ziel und Gradient ---------------------------------------------------------------

def test_ppo_objective_hand_cases():
    assert np.isclose(ppo_objective(1.5, 2.0, 0.2), 2.4)    # ueber 1.2 abgeschnitten
    assert np.isclose(ppo_objective(0.5, -1.0, 0.2), -0.8)  # unter 0.8 abgeschnitten
    assert np.isclose(ppo_objective(1.1, 1.0, 0.2), 1.1)    # im Bereich: ungeclippt


def test_logit_grad_hand_case_uniform_policy():
    logits = np.zeros((1, 3))
    acts = np.array([0])
    grad = ppo_logit_grad(logits, acts, np.array([1.0]), np.log(np.array([1 / 3])), 0.2, 0.0)
    # -A * r * (onehot - p) mit r = 1, p = 1/3
    assert np.allclose(grad[0], [-2 / 3, 1 / 3, 1 / 3])


def test_logit_grad_is_zero_in_clipped_region():
    logits = np.array([[0.0, 0.0, 0.0]])
    acts = np.array([0])
    # Ratio 1.5 bei positivem Vorteil -> geclippt
    old_logp = np.log(np.array([1 / 3])) - np.log(1.5)
    assert np.allclose(ppo_logit_grad(logits, acts, np.array([2.0]), old_logp, 0.2, 0.0), 0.0)
    # Ratio 0.5 bei negativem Vorteil -> geclippt
    old_logp = np.log(np.array([1 / 3])) - np.log(0.5)
    assert np.allclose(ppo_logit_grad(logits, acts, np.array([-1.0]), old_logp, 0.2, 0.0), 0.0)
    # dieselben Verhältnisse in der ANDEREN Richtung sind nicht geclippt
    old_logp = np.log(np.array([1 / 3])) - np.log(1.5)
    assert not np.allclose(ppo_logit_grad(logits, acts, np.array([-2.0]), old_logp, 0.2, 0.0), 0.0)


def test_logit_grad_matches_finite_differences():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(6, 3))
    acts = rng.integers(0, 3, 6)
    adv = rng.normal(size=6)
    old_logp = np.log(softmax(logits)[np.arange(6), acts]) + rng.normal(scale=0.05, size=6)
    for clip, entropy in ((0.2, 0.0), (0.2, 0.02), (1e9, 0.01)):
        analytic = ppo_logit_grad(logits, acts, adv, old_logp, clip, entropy)
        numeric = np.zeros_like(logits)
        eps = 1e-6
        for i in range(logits.shape[0]):
            for c in range(3):
                up, down = logits.copy(), logits.copy()
                up[i, c] += eps
                down[i, c] -= eps
                numeric[i, c] = (
                    ppo_loss(up, acts, adv, old_logp, clip, entropy) - ppo_loss(down, acts, adv, old_logp, clip, entropy)
                ) / (2 * eps)
        assert np.abs(analytic - numeric).max() < 1e-7, f"clip={clip} entropy={entropy}"


def test_softmax_score_function_identity():
    """Die Summe pi_a * grad log pi_a ueber alle Aktionen ist 0 (Baseline-Freiheit des Policy-Gradienten)."""
    logits = np.array([[0.3, -1.2, 0.8]])
    p = softmax(logits)[0]
    total = sum(p[a] * (np.eye(3)[a] - p) for a in range(3))
    assert np.allclose(total, 0.0)


def test_normalize_mean_zero_std_one():
    x = normalize(np.random.default_rng(0).normal(3.0, 2.0, size=(4, 5)))
    assert abs(x.mean()) < 1e-12 and abs(x.std() - 1.0) < 1e-6


# --- MLP / Optimierer ----------------------------------------------------------------------

def test_mlp_backprop_matches_finite_differences():
    rng = np.random.default_rng(1)
    mlp = MLP(2, [4, 5, 3], rng)
    x = rng.normal(size=(2, 7, 4))

    def loss():
        return 0.5 * (mlp.forward(x)[-1] ** 2).sum()

    hs = mlp.forward(x)
    grads = mlp.backward(hs, hs[-1])
    eps = 1e-6
    for p, g in zip(mlp.params(), grads):
        for idx in [(0, 0, 0), (1, 0, min(2, p.shape[2] - 1))]:
            old = p[idx]
            p[idx] = old + eps
            up = loss()
            p[idx] = old - eps
            down = loss()
            p[idx] = old
            assert abs((up - down) / (2 * eps) - g[idx]) < 1e-6


def test_last_zero_gives_exactly_zero_logits():
    mlp = MLP(1, [6, 16, 16, 3], np.random.default_rng(0), last_zero=True)
    assert np.all(mlp.forward(np.random.default_rng(1).normal(size=(1, 20, 6)))[-1] == 0.0)


def test_adam_moves_against_gradient_and_clips_norm():
    p = [np.zeros(3)]
    Adam(p, lr=0.1).step([np.array([1.0, 0.0, -1.0])], max_norm=0)
    assert p[0][0] < 0 < p[0][2] and p[0][1] == 0
    big = [np.zeros(1)]
    small = [np.zeros(1)]
    Adam(big, lr=0.1).step([np.array([100.0])], max_norm=0.5)
    Adam(small, lr=0.1).step([np.array([0.5])], max_norm=0.5)
    assert np.allclose(big[0], small[0])  # Adam ist skaleninvariant, das Clipping ändert nur die Norm


def test_table_value_update_lr_one_is_empirical_mean():
    values = np.zeros(3)
    # zwei Besuche von Zustand 1 mit Fehlern 2 und 4 -> Mittelwert 3; Zustand 0 nie besucht
    table_value_update(values, np.array([1, 1]), np.array([2.0, 4.0]), lr=1.0)
    assert np.allclose(values, [0.0, 3.0, 0.0])
    table_value_update(values, np.array([1]), np.array([1.0]), lr=0.5)
    assert np.allclose(values, [0.0, 3.5, 0.0])


# --- Training ------------------------------------------------------------------------------

def _fresh_result(instance, actor_kind, method=C.METHOD_MAPPO):
    """Ein Ergebnis mit dem UNTRAINIERTEN Akteur (Null-Logits) - ein Trainingslauf mit 0 Iterationen gibt es nicht."""
    result = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 1, 0, method, actor_kind)
    if actor_kind == C.ACTOR_TABLE:
        zero_actor = np.zeros_like(result.actor)
    else:
        zero_actor = [w.copy() for w in result.actor]
        half = len(zero_actor) // 2
        zero_actor[half - 1][:] = 0.0  # letzte Gewichtsschicht
        zero_actor[-1][:] = 0.0        # letzte Bias-Schicht
    from dataclasses import replace
    return replace(result, actor=zero_actor)


@pytest.mark.parametrize("actor_kind", [C.ACTOR_NET, C.ACTOR_TABLE])
def test_untrained_policy_equals_raw_contract_net(actor_kind):
    for seed in range(20):
        instance = generate_instance(4 + seed % 5, 2 + seed % 3, 0.4, 1.0, seed)
        geom = geometry(instance)
        pos, dur = arrays_from_instances([instance])
        result = _fresh_result(instance, actor_kind)
        ms = simulate(geom, pos, dur, greedy_act_fn(geom, result))
        assert ms[0] == cnp_makespans(geom, pos, dur)[0], f"seed={seed}"


@pytest.mark.parametrize("actor_kind", [C.ACTOR_NET, C.ACTOR_TABLE])
@pytest.mark.parametrize("method", [C.METHOD_IPPO, C.METHOD_MAPPO])
def test_training_is_deterministic_and_seed_dependent(actor_kind, method):
    instance = generate_instance(5, 3, 0.3, 1.0, 4)

    def run(seed):
        return train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 300, seed, method, actor_kind).actor

    a, b, c = run(7), run(7), run(8)
    if actor_kind == C.ACTOR_TABLE:
        assert np.array_equal(a, b) and not np.array_equal(a, c)
    else:
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert any(not np.array_equal(x, y) for x, y in zip(a, c))


@pytest.mark.parametrize("actor_kind", [C.ACTOR_NET, C.ACTOR_TABLE])
def test_random_mode_training_ignores_scenario_seed(actor_kind):
    """Im Modus "zufällig" ist die Demo-Instanz nur Größenlieferant: gleicher Trainings-Seed => gleicher Akteur."""
    i1 = generate_instance(5, 3, 0.3, 1.0, 1)
    i2 = generate_instance(5, 3, 0.3, 1.0, 99)
    a = train_ppo(i1, C.ENV_RANDOM, 0.3, 0.3, 300, 5, C.METHOD_MAPPO, actor_kind).actor
    b = train_ppo(i2, C.ENV_RANDOM, 0.3, 0.3, 300, 5, C.METHOD_MAPPO, actor_kind).actor
    if actor_kind == C.ACTOR_TABLE:
        assert np.array_equal(a, b)
    else:
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_snapshots_and_bookkeeping():
    instance = generate_instance(5, 2, 0.3, 1.0, 1)
    r = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 500, 0, C.METHOD_MAPPO, C.ACTOR_NET, checkpoints=(100, 300))
    assert set(r.snapshots) == {100, 300}
    assert any(not np.array_equal(x, y) for x, y in zip(r.snapshots[100], r.actor))
    assert r.n_decisions == 500 * 5 * 2  # 5 Iterationen x 100 Episoden x Aufträge x Agenten
    assert r.critic["kind"] == "net_central"
    assert train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 300, 0, C.METHOD_IPPO, C.ACTOR_NET).critic["kind"] == "net_local"


def test_shared_parameters_agents_differ_only_through_agent_id():
    from mappo_ppo import action_probs

    instance = generate_instance(5, 3, 0.3, 1.0, 2)
    geom = geometry(instance)
    result = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 600, 0, C.METHOD_MAPPO, C.ACTOR_NET)
    same = np.full((1, 3), 4.0)
    args = (geom, result, 2, same, same, same, np.array([7.0]), np.array([9.0]))
    probs = action_probs(*args)[0]
    assert not np.allclose(probs[0], probs[1])  # verschiedene IDs => (i.d.R.) verschiedene Aktionen
    weights = [w.copy() for w in result.actor]
    weights[0][:, -3:, :] = 0.0  # Agenten-ID-Eingänge der ersten Schicht abschalten
    probs = action_probs(*args, actor_params=weights)[0]
    assert np.allclose(probs[0], probs[1]) and np.allclose(probs[1], probs[2])


def test_scope_and_job_index_options_change_critic_and_actor_inputs():
    instance = generate_instance(5, 3, 0.3, 1.0, 2)
    joint = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 200, 0, C.METHOD_MAPPO, C.ACTOR_NET, C.SCOPE_JOINT)
    full = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 200, 0, C.METHOD_MAPPO, C.ACTOR_NET, C.SCOPE_FULL)
    assert full.critic["params"][0].shape[1] > joint.critic["params"][0].shape[1]  # Auftragsliste = mehr Eingänge
    with_idx = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 200, 0, C.METHOD_MAPPO, C.ACTOR_NET, use_job_index=True)
    no_idx = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 200, 0, C.METHOD_MAPPO, C.ACTOR_NET, use_job_index=False)
    assert with_idx.actor[0].shape[1] == no_idx.actor[0].shape[1] + 1
    # Für IPPO ist der Kritiker-Umfang egal; für die Tabelle sind Umfang und Index festgelegt
    ippo = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 200, 0, C.METHOD_IPPO, C.ACTOR_NET, C.SCOPE_FULL)
    assert ippo.scope == C.SCOPE_JOINT
    table = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 200, 0, C.METHOD_MAPPO, C.ACTOR_TABLE, C.SCOPE_FULL, use_job_index=False)
    assert table.scope == C.SCOPE_JOINT and table.use_job_index is True


def test_clip_off_changes_training_and_is_recorded():
    instance = generate_instance(5, 3, 0.3, 1.0, 4)
    on = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 400, 0, C.METHOD_MAPPO, C.ACTOR_NET, clip=0.2)
    off = train_ppo(instance, C.ENV_RECURRING, 0.3, 0.3, 400, 0, C.METHOD_MAPPO, C.ACTOR_NET, clip=None)
    assert off.clip is None
    assert any(not np.array_equal(x, y) for x, y in zip(on.actor, off.actor))
