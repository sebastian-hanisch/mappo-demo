"""Unabhängige Orakel für PPO/MAPPO: GAE per direkter Summenformel und von Hand, PPO-Ziel per Einzelwert-Schleife,
Gradient per finiter Differenz (inkl. Kette durch das MLP), vektorisierte Umgebung gegen exakt rationale Simulation,
Tabellen-Kritiker per Schleife - und eine Neuimplementierung der Trainingsschritte (Schleifen statt Tensoren) für
Tabelle und Netz, IPPO und MAPPO, deren Parameter bei identischen Zufallsströmen exakt übereinstimmen müssen."""
import math
from fractions import Fraction as F

import numpy as np

import cn_constants as C
import mappo_env as env
import mappo_ppo as ppo
from cn_scenario import generate_instance
from mappo_nets import MLP, Adam


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _gae_episode(vals, reward, lam):
    n = len(vals)
    out, run = [0.0] * n, 0.0
    for t in reversed(range(n)):
        delta = (reward - vals[t]) if t == n - 1 else (vals[t + 1] - vals[t])
        run = delta + lam * run
        out[t] = run
    return out


def test_gae_matches_direct_sum_and_hand_examples():
    rng = np.random.default_rng(0)
    for _ in range(40):
        batch, n = int(rng.integers(1, 4)), int(rng.integers(1, 8))
        v, r = rng.normal(size=(batch, n)), rng.normal(size=batch) * 5
        lam, gamma = float(rng.choice([0.0, 0.5, 0.9, 1.0])), float(rng.choice([1.0, 0.95]))
        ref = np.zeros_like(v)
        for b in range(batch):
            for t in range(n):
                tot = 0.0
                for step in range(n - t):
                    s = t + step
                    delta = (r[b] - v[b, s]) if s == n - 1 else (gamma * v[b, s + 1] - v[b, s])
                    tot += (gamma * lam) ** step * delta
                ref[b, t] = tot
        np.testing.assert_allclose(ppo.gae(v, r, lam, gamma), ref, atol=1e-9)
    hand = np.array([[1.0, 2.0, 3.0]])
    for lam, expected in [(0.5, [3.25, 4.5, 7]), (1.0, [9, 8, 7]), (0.0, [1, 1, 7])]:
        np.testing.assert_allclose(ppo.gae(hand, np.array([10.0]), lam), [expected], atol=1e-12)


def test_ppo_loss_and_gradient_against_brute_force_and_finite_differences():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(80):
        n = int(rng.integers(1, 7))
        logits, acts, adv = rng.normal(size=(n, 3)), rng.integers(0, 3, n), rng.normal(size=n)
        p = np.array([_softmax(z) for z in logits])
        old = np.log(p[np.arange(n), acts]) + rng.normal(size=n) * rng.choice([0.05, 0.3, 1.0])
        clip, ent = float(rng.choice([0.1, 0.2, 1e9])), float(rng.choice([0.0, 0.01, 0.03]))
        ratio = p[np.arange(n), acts] / np.exp(old)
        if clip < 1e8 and (np.abs(np.abs(ratio - 1) - clip) < 2e-4).any():
            continue
        obj = [min(ratio[i] * adv[i], min(max(ratio[i], 1 - clip), 1 + clip) * adv[i]) for i in range(n)]
        entropy = -(p * np.log(p + 1e-12)).sum(-1)
        loss = ppo.ppo_loss(logits, acts, adv, old, clip, ent)
        assert abs(loss - (-np.mean(obj) - ent * entropy.mean())) < 1e-9
        grad = ppo.ppo_logit_grad(logits, acts, adv, old, clip, ent)
        for i in range(n):
            for c in range(3):
                up, dn = logits.copy(), logits.copy()
                up[i, c] += 1e-6
                dn[i, c] -= 1e-6
                fd = (ppo.ppo_loss(up, acts, adv, old, clip, ent) - ppo.ppo_loss(dn, acts, adv, old, clip, ent)) / 2e-6
                assert abs(fd - grad[i, c]) < 1e-6
        checked += 1
    assert checked > 40


def test_mlp_backprop_chain_through_ppo_loss_matches_finite_differences():
    rng = np.random.default_rng(2)
    for groups in (1, 3):
        net = MLP(groups, [5, 7, 7, 3], np.random.default_rng(1))
        x = rng.normal(size=(groups, 6, 5))
        acts, adv = rng.integers(0, 3, (groups, 6)), rng.normal(size=(groups, 6))
        old = rng.normal(size=(groups, 6)) * 0.3 - 1.0

        def loss():
            return ppo.ppo_loss(net.forward(x)[-1], acts, adv, old, 0.2, 0.01)

        hs = net.forward(x)
        grads = net.backward(hs, ppo.ppo_logit_grad(hs[-1], acts, adv, old, 0.2, 0.01))
        for par, gr in zip(net.params(), grads):
            for _ in range(5):
                idx = tuple(int(rng.integers(0, s)) for s in par.shape)
                keep = par[idx]
                par[idx] = keep + 1e-6
                up = loss()
                par[idx] = keep - 1e-6
                dn = loss()
                par[idx] = keep
                assert abs((up - dn) / 2e-6 - gr[idx]) < 1e-6


def test_vectorised_environment_matches_exact_rational_simulation():
    rng = np.random.default_rng(3)
    for _ in range(40):
        n, k, batch = int(rng.integers(1, 7)), int(rng.integers(1, 5)), int(rng.integers(1, 4))
        inst = generate_instance(n, k, 0.3, float(rng.choice([0.2, 1.0, 2.0])), int(rng.integers(10 ** 5)))
        pos = np.array([[j.position for j in inst.jobs]] * batch)
        dur = np.array([[j.duration for j in inst.jobs]] * batch) * np.exp(0.3 * rng.standard_normal((batch, n)))
        table = rng.integers(0, 3, (batch, n, k))
        got = env.simulate(env.geometry(inst), pos, dur, lambda j, free, apos, travel, jp, jd: table[:, j, :])
        for b in range(batch):
            where, free = [F(x) for x in inst.agent_start_positions], [F(0)] * k
            for j in range(n):
                best = None
                for a in range(k):
                    trv = abs(where[a] - F(float(pos[b, j]))) * F(inst.travel_time_per_unit)
                    fin = free[a] + trv + F(float(dur[b, j]))
                    key = (fin + F(C.BIAS_ACTIONS[table[b, j, a]]), a)
                    if best is None or key < best[0]:
                        best = (key, a, fin)
                _, w, fin = best
                where[w], free[w] = F(float(pos[b, j])), fin
            assert abs(float(max(free)) - got[b]) < 1e-9


def test_table_value_update_matches_per_entry_mean():
    rng = np.random.default_rng(4)
    for _ in range(30):
        size, count, lr = int(rng.integers(1, 9)), int(rng.integers(1, 30)), float(rng.choice([0.3, 1.0]))
        vals, idx, err = rng.normal(size=size), rng.integers(0, size, count), rng.normal(size=count)
        ref = vals.copy()
        for s in range(size):
            if (idx == s).any():
                ref[s] += lr * err[idx == s].mean()
        got = vals.copy()
        ppo.table_value_update(got, idx, err, lr)
        np.testing.assert_allclose(got, ref, atol=1e-12)


def _grad_sample(z, a, adv, old_logp, clip, ent, count):
    p = _softmax(z)
    ratio = p[a] / math.exp(old_logp)
    active = (adv >= 0 and ratio < 1 + clip) or (adv < 0 and ratio > 1 - clip)
    g = np.zeros(3)
    if active:
        for c in range(3):
            g[c] += adv * ratio * ((1.0 if c == a else 0.0) - p[c])
    if ent > 0:
        h = -sum(p[c] * math.log(p[c] + 1e-12) for c in range(3))
        for c in range(3):
            g[c] += ent * (-p[c] * (math.log(p[c] + 1e-12) + h))
    return -g / count


def _norm(x):
    x = np.asarray(x)
    return (x - x.mean()) / (x.std() + 1e-8)


def _train_table_reference(inst, env_mode, episodes, seed, method):
    geom = env.geometry(inst)
    k, n = geom.k, geom.n
    states = env.n_local_buckets(geom)
    z = np.zeros((k, states, 3))
    opt = Adam([z], C.TABLE_LR_POLICY)
    vt = np.zeros(env.n_global_buckets(geom)) if method == C.METHOD_MAPPO else np.zeros((k, states))
    rng = np.random.default_rng([seed, 7])
    sampler = env.Sampler(env_mode, inst, 0.3, 0.3, seed)
    batch = C.TABLE_BATCH_EPISODES
    for _ in range(max(1, episodes // batch)):
        pos, dur = sampler.sample(batch)
        ls, gs = np.zeros((batch, n, k), int), np.zeros((batch, n), int)
        acts, olp = np.zeros((batch, n, k), int), np.zeros((batch, n, k))

        def act_fn(j, free, apos, travel, jp, jd):
            st = env.local_bucket(geom, j, free, travel)
            u = rng.random((batch, k, 1))
            out = np.zeros((batch, k), int)
            for b in range(batch):
                for a in range(k):
                    p = _softmax(z[a, st[b, a]])
                    ac = min(int((u[b, a, 0] > np.cumsum(p)).sum()), 2)
                    out[b, a], ls[b, j, a], acts[b, j, a], olp[b, j, a] = ac, st[b, a], ac, math.log(p[ac] + 1e-12)
                if method == C.METHOD_MAPPO:
                    gs[b, j] = env.global_bucket(geom, j, free[b:b + 1])[0]
            return out

        reward = -env.simulate(geom, pos, dur, act_fn) / C.REWARD_SCALE
        adv = np.zeros((batch, n, k))
        if method == C.METHOD_MAPPO:
            vals = np.array([[vt[gs[b, j]] for j in range(n)] for b in range(batch)])
            a_raw = np.array([_gae_episode(vals[b], reward[b], C.TABLE_GAE_LAMBDA) for b in range(batch)])
            adv = np.repeat(_norm(a_raw)[:, :, None], k, 2)
            errs = {}
            for b in range(batch):
                for j in range(n):
                    errs.setdefault(gs[b, j], []).append(a_raw[b, j])
            for s, e in errs.items():
                vt[s] += C.TABLE_LR_VALUE * np.mean(e)
        else:
            raw = []
            for a in range(k):
                vals = np.array([[vt[a, ls[b, j, a]] for j in range(n)] for b in range(batch)])
                a_raw = np.array([_gae_episode(vals[b], reward[b], C.TABLE_GAE_LAMBDA) for b in range(batch)])
                adv[:, :, a] = _norm(a_raw)
                raw.append(a_raw)
            for a in range(k):
                errs = {}
                for b in range(batch):
                    for j in range(n):
                        errs.setdefault(ls[b, j, a], []).append(raw[a][b, j])
                for s, e in errs.items():
                    vt[a, s] += C.TABLE_LR_VALUE * np.mean(e)
        for _e in range(C.TABLE_EPOCHS):
            grad = np.zeros_like(z)
            for b in range(batch):
                for j in range(n):
                    for a in range(k):
                        s = ls[b, j, a]
                        grad[a, s] += _grad_sample(
                            z[a, s], acts[b, j, a], adv[b, j, a], olp[b, j, a], C.PPO_CLIP, C.TABLE_ENTROPY, batch * n,
                        )
            opt.step([grad], 0.0)
    return z, vt


def test_table_training_matches_loop_reimplementation_for_ippo_and_mappo():
    inst = generate_instance(4, 2, 0.3, 1.0, 8)
    for env_mode in (C.ENV_RECURRING, C.ENV_RANDOM):
        for method in (C.METHOD_MAPPO, C.METHOD_IPPO):
            res = ppo.train_ppo(inst, env_mode, 0.3, 0.3, 128, 3, method, C.ACTOR_TABLE, C.SCOPE_JOINT, C.PPO_CLIP)
            z, vt = _train_table_reference(inst, env_mode, 128, 3, method)
            assert np.abs(z - res.actor).max() < 1e-9
            assert np.abs(vt - res.critic["values"]).max() < 1e-9


def _train_net_reference(inst, episodes, seed, method):
    geom = env.geometry(inst)
    k, n = geom.k, geom.n
    batch, hidden = C.NET_BATCH_EPISODES, C.NET_HIDDEN
    rng = np.random.default_rng([seed, 11])
    dummy = np.zeros((2, k))
    f_local = env.add_agent_id(env.local_features(geom, 0, dummy, dummy, np.zeros(2), np.zeros(2), dummy, True)).shape[-1]
    actor = MLP(1, [f_local, hidden, hidden, 3], rng, last_zero=True)
    if method == C.METHOD_MAPPO:
        f_c = env.central_features(
            geom, 0, dummy, dummy, np.zeros(2), np.zeros(2), dummy, np.zeros((2, n)), np.zeros((2, n)), C.SCOPE_JOINT,
        ).shape[-1]
        critic = MLP(1, [f_c, hidden, hidden, 1], rng, last_zero=True)
    else:
        critic = MLP(1, [f_local, hidden, hidden, 1], rng, last_zero=True)
    a_opt, c_opt = Adam(actor.params(), C.NET_LR), Adam(critic.params(), C.NET_LR)
    sampler = env.Sampler(C.ENV_RECURRING, inst, 0.3, 0.3, seed)
    mu = sd = None
    for _ in range(max(1, episodes // batch)):
        pos, dur = sampler.sample(batch)
        fl = np.zeros((k, batch, n, f_local))
        acts, olp, fc = np.zeros((k, batch, n), int), np.zeros((k, batch, n)), []

        def act_fn(j, free, apos, travel, jp, jd):
            x = env.add_agent_id(env.local_features(geom, j, free, travel, jp, jd, apos, True))
            logits = actor.forward(x.reshape(1, -1, x.shape[-1]))[-1].reshape(k, batch, 3)
            u = rng.random((k, batch, 1))
            out = np.zeros((batch, k), int)
            for a in range(k):
                for b in range(batch):
                    p = _softmax(logits[a, b])
                    ac = min(int((u[a, b, 0] > np.cumsum(p)).sum()), 2)
                    out[b, a], acts[a, b, j], olp[a, b, j] = ac, ac, math.log(p[ac] + 1e-12)
            fl[:, :, j] = x
            if method == C.METHOD_MAPPO:
                fc.append(env.central_features(geom, j, free, travel, jp, jd, apos, pos, dur, C.SCOPE_JOINT))
            return out

        reward = -env.simulate(geom, pos, dur, act_fn) / C.REWARD_SCALE
        if mu is None:
            mu, sd = float(reward.mean()), float(reward.std() + 1e-6)
        if method == C.METHOD_MAPPO:
            rows = np.stack(fc, 1).reshape(1, batch * n, -1)
            vals = critic.forward(rows)[-1].reshape(batch, n) * sd + mu
            a_raw = np.array([_gae_episode(vals[b], reward[b], C.NET_GAE_LAMBDA) for b in range(batch)])
            adv = np.repeat(_norm(a_raw)[:, :, None], k, 2)
            tgt_n = (a_raw + vals - mu) / sd
            for _e in range(C.NET_EPOCHS):
                hs = critic.forward(rows)
                out = hs[-1].reshape(batch, n)
                d = (2 * (out - tgt_n) / (batch * n)).reshape(1, batch * n, 1)
                c_opt.step(critic.backward(hs, d), C.NET_MAX_GRAD_NORM)
        else:
            rows = fl.reshape(1, k * batch * n, -1)
            vals = (critic.forward(rows)[-1].reshape(k, batch, n) * sd + mu).transpose(1, 2, 0)
            a_raw, adv = np.zeros((batch, n, k)), np.zeros((batch, n, k))
            for a in range(k):
                a_raw[:, :, a] = np.array([_gae_episode(vals[b, :, a], reward[b], C.NET_GAE_LAMBDA) for b in range(batch)])
                adv[:, :, a] = _norm(a_raw[:, :, a])
            tgt_n = ((a_raw + vals - mu) / sd).transpose(2, 0, 1)
            for _e in range(C.NET_EPOCHS):
                hs = critic.forward(rows)
                out = hs[-1].reshape(k, batch, n)
                d = (2 * (out - tgt_n) / (k * batch * n)).reshape(1, k * batch * n, 1)
                c_opt.step(critic.backward(hs, d), C.NET_MAX_GRAD_NORM)
        xf, af, of = fl.reshape(k, batch * n, -1), acts.reshape(k, batch * n), olp.reshape(k, batch * n)
        advf = adv.transpose(2, 0, 1).reshape(k, batch * n)
        for _e in range(C.NET_EPOCHS):
            for idx in np.array_split(rng.permutation(batch * n), C.NET_MINIBATCHES):
                m = len(idx)
                hs = actor.forward(xf[:, idx].reshape(1, k * m, -1))
                lg = hs[-1].reshape(k, m, 3)
                d = np.zeros((k, m, 3))
                for a in range(k):
                    for q in range(m):
                        d[a, q] = _grad_sample(lg[a, q], af[a, idx[q]], advf[a, idx[q]], of[a, idx[q]], C.PPO_CLIP, C.NET_ENTROPY, k * m)
                a_opt.step(actor.backward(hs, d.reshape(hs[-1].shape)), C.NET_MAX_GRAD_NORM)
    return [w.copy() for w in actor.params()]


def test_net_training_matches_loop_reimplementation_for_ippo_and_mappo():
    inst = generate_instance(4, 2, 0.3, 1.0, 8)
    for method in (C.METHOD_MAPPO, C.METHOD_IPPO):
        res = ppo.train_ppo(inst, C.ENV_RECURRING, 0.3, 0.3, 100, 3, method, C.ACTOR_NET, C.SCOPE_JOINT, C.PPO_CLIP, True)
        ref = _train_net_reference(inst, 100, 3, method)
        assert max(np.abs(a - b).max() for a, b in zip(ref, res.actor)) < 1e-9
