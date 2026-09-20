"""PPO für das kooperative Dispatch-Spiel: IPPO (jeder Agent hat einen eigenen, lokalen Kritiker) und
MAPPO (ein zentraler Kritiker sieht im Training den gemeinsamen Zustand aller Agenten). Beim Einsatz
entscheidet jeder Akteur nur aus seiner lokalen Beobachtung - das ist CTDE (Centralized Training,
Decentralized Execution).

Zwei Akteur-Arten:
- "net":   kleines MLP (2 x 16 tanh), stetige Merkmale, GETEILTE Parameter (Yu et al.), Agenten-ID als One-Hot.
- "table": eine Softmax-Tabelle je Agent über dieselben Buckets wie IQL.

Gemeinsam: Aktionen {normal, +25, -25} auf das eigene Gebot, Belohnung -Makespan/10 nur am Episodenende,
GAE, geclipptes PPO-Ziel (Clipping abschaltbar), Entropie-Bonus. Ein untrainierter Akteur (Null-Logits)
wählt per Tie-Break Aktion 0 und verhält sich exakt wie Contract Net."""

import time
from dataclasses import dataclass

import numpy as np

import cn_constants as C
from mappo_env import (
    N_ACTIONS, Sampler, add_agent_id, arrays_from_instances, central_features, geometry, global_bucket,
    local_bucket, local_features, n_global_buckets, n_local_buckets, simulate,
)
from mappo_nets import MLP, Adam, forward_params, softmax, unpack


# --- Bausteine (einzeln testbar) ----------------------------------------------------

def gae(values, reward, lam, gamma=1.0):
    """Generalized Advantage Estimation mit Belohnung nur am Ende. values: [B, n, ...] (Wert VOR jeder
    Entscheidung), reward: [B] (terminal). Gibt Vorteile derselben Form wie `values` zurück."""
    batch, n = values.shape[:2]
    adv = np.zeros_like(values)
    running = np.zeros_like(adv[:, 0])
    for t in range(n - 1, -1, -1):
        if t == n - 1:
            delta = reward.reshape((batch,) + (1,) * (values.ndim - 2)) - values[:, t]
        else:
            delta = gamma * values[:, t + 1] - values[:, t]
        running = delta + gamma * lam * running
        adv[:, t] = running
    return adv


def ppo_objective(ratio, adv, clip):
    """Das geclippte PPO-Ziel min(r*A, clip(r, 1-e, 1+e)*A) für einzelne Werte (zu maximieren)."""
    return np.minimum(ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv)


def ppo_loss(logits, acts, adv, old_logp, clip, entropy_coef):
    """Skalarer Verlust (zu minimieren) - passend zu `ppo_logit_grad`; wird im Test per finiten Differenzen
    gegen den geschlossenen Gradienten geprüft."""
    p = softmax(logits)
    chosen = np.take_along_axis(p, acts[..., None], -1)[..., 0]
    ratio = np.exp(np.log(chosen + 1e-12) - old_logp)
    entropy = -(p * np.log(p + 1e-12)).sum(-1)
    return float(-np.mean(ppo_objective(ratio, adv, clip)) - entropy_coef * np.mean(entropy))


def ppo_logit_grad(logits, acts, adv, old_logp, clip, entropy_coef):
    """Gradient des mittleren PPO-Verlusts nach den Logits (geschlossene Form): im ungeclippten Bereich
    -A * r * (onehot(a) - pi) / N, im geclippten Bereich 0; der Entropie-Term kommt dazu."""
    p = softmax(logits)
    count = acts.size
    idx = np.arange(count)
    chosen = p.reshape(-1, N_ACTIONS)[idx, acts.ravel()].reshape(acts.shape)
    ratio = np.exp(np.log(chosen + 1e-12) - old_logp)
    active = ((adv >= 0) & (ratio < 1 + clip)) | ((adv < 0) & (ratio > 1 - clip))
    coef = np.where(active, adv * ratio, 0.0)
    onehot = np.zeros_like(p)
    onehot.reshape(-1, N_ACTIONS)[idx, acts.ravel()] = 1.0
    grad = coef[..., None] * (onehot - p)
    if entropy_coef > 0:
        log_p = np.log(p + 1e-12)
        entropy = -(p * log_p).sum(-1, keepdims=True)
        grad = grad + entropy_coef * (-p * (log_p + entropy))
    return -grad / count


def normalize(adv):
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def sample_from_logits(logits, rng):
    p = softmax(logits)
    u = rng.random(logits.shape[:-1] + (1,))
    return (u > np.cumsum(p, -1)).sum(-1).clip(0, N_ACTIONS - 1), p


def table_value_update(values, indices, errors, lr):
    """Tabellen-Kritiker: jeder besuchte Eintrag bewegt sich um lr * (mittlerer Fehler seiner Besuche) -
    mit lr=1 wird er exakt zum Mittelwert der Ziele. `values` wird in place verändert."""
    total = np.bincount(indices, weights=errors, minlength=len(values))
    count = np.bincount(indices, minlength=len(values))
    values += lr * total / np.maximum(count, 1)


def effective_clip(clip):
    return C.PPO_CLIP_OFF if clip is None else clip


# --- Ergebnis ---------------------------------------------------------------------

@dataclass(frozen=True)
class PPOResult:
    method: str
    actor_kind: str
    scope: str
    clip: object            # None = aus
    use_job_index: bool
    n_jobs: int
    n_agents: int
    actor: object           # Netz: Liste [Gewichte..., Biases...]; Tabelle: Logits [k, S, 3]
    critic: dict            # Parameter des Kritikers (für die Anzeige)
    snapshots: dict         # episoden -> Akteur-Parameter (Zwischenstände)
    n_episodes: int
    n_decisions: int        # tatsächlich trainierte Agenten-Entscheidungen
    wall_time_s: float
    critic_ev: float        # erklärte Varianz des Kritikers im letzten Zehntel des Trainings
    adv_var: float          # Varianz der (unnormierten) Vorteile im letzten Zehntel


def _explained_variance(target, predicted):
    return float(1.0 - (target - predicted).var() / (target.var() + 1e-9))


# --- Training: Netz -----------------------------------------------------------------

def _train_net(sampler, geom, n_episodes, train_seed, method, scope, clip, use_job_index, checkpoints):
    k, n = geom.k, geom.n
    hidden = C.NET_HIDDEN
    batch = C.NET_BATCH_EPISODES
    rng = np.random.default_rng([train_seed, 11])
    dummy = np.zeros((2, k))
    f_local = add_agent_id(local_features(geom, 0, dummy, dummy, np.zeros(2), np.zeros(2), dummy, use_job_index)).shape[-1]
    actor = MLP(1, [f_local, hidden, hidden, N_ACTIONS], rng, last_zero=True)
    if method == C.METHOD_MAPPO:
        f_central = central_features(
            geom, 0, dummy, dummy, np.zeros(2), np.zeros(2), dummy, np.zeros((2, n)), np.zeros((2, n)), scope,
        ).shape[-1]
        critic = MLP(1, [f_central, hidden, hidden, 1], rng, last_zero=True)
    else:
        f_central = 0
        critic = MLP(1, [f_local, hidden, hidden, 1], rng, last_zero=True)
    actor_opt = Adam(actor.params(), C.NET_LR)
    critic_opt = Adam(critic.params(), C.NET_LR)
    clip_value = effective_clip(clip)

    def actor_logits(x):
        hs = actor.forward(x.reshape(1, -1, x.shape[-1]))
        return hs[-1].reshape(x.shape[0], x.shape[1], N_ACTIONS), hs

    n_iter = max(1, n_episodes // batch)
    done = 0
    snapshots = {}
    diag = []
    mu = sd = None

    for it in range(n_iter):
        pos, dur = sampler.sample(batch)
        feats_local = np.zeros((k, batch, n, f_local))
        acts = np.zeros((k, batch, n), dtype=int)
        old_logp = np.zeros((k, batch, n))
        feats_central = np.zeros((batch, n, f_central)) if method == C.METHOD_MAPPO else None

        def act_fn(j, free, agent_pos, travel, job_pos, job_dur):
            x = add_agent_id(local_features(geom, j, free, travel, job_pos, job_dur, agent_pos, use_job_index))
            logits, _ = actor_logits(x)
            a, p = sample_from_logits(logits, rng)
            feats_local[:, :, j] = x
            acts[:, :, j] = a
            old_logp[:, :, j] = np.log(np.take_along_axis(p, a[..., None], -1)[..., 0] + 1e-12)
            if method == C.METHOD_MAPPO:
                feats_central[:, j] = central_features(
                    geom, j, free, travel, job_pos, job_dur, agent_pos, pos, dur, scope,
                )
            return a.T

        makespan = simulate(geom, pos, dur, act_fn)
        reward = -makespan / C.REWARD_SCALE
        if mu is None:
            mu, sd = float(reward.mean()), float(reward.std() + 1e-6)

        x_flat = feats_local.reshape(k, batch * n, -1)
        if method == C.METHOD_MAPPO:
            hs = critic.forward(feats_central.reshape(1, batch * n, -1))
            values = hs[-1].reshape(batch, n) * sd + mu
        else:
            hs = critic.forward(x_flat.reshape(1, -1, x_flat.shape[-1]))
            values = (hs[-1].reshape(k, batch, n) * sd + mu).transpose(1, 2, 0)
        adv_raw = gae(values, reward, C.NET_GAE_LAMBDA)
        target = adv_raw + values
        diag.append((_explained_variance(target, values), float(adv_raw.var())))

        if method == C.METHOD_MAPPO:
            adv = np.repeat(normalize(adv_raw)[:, :, None], k, 2)
        else:
            adv = np.stack([normalize(adv_raw[:, :, a]) for a in range(k)], -1)
        adv = adv.transpose(2, 0, 1)
        target_n = (target - mu) / sd

        for _ in range(C.NET_EPOCHS):
            if method == C.METHOD_MAPPO:
                hs = critic.forward(feats_central.reshape(1, batch * n, -1))
                out = hs[-1].reshape(batch, n)
                d_out = (2 * (out - target_n) / out.size).reshape(1, batch * n, 1)
            else:
                hs = critic.forward(x_flat.reshape(1, -1, x_flat.shape[-1]))
                out = hs[-1].reshape(k, batch, n)
                d_out = (2 * (out - target_n.transpose(2, 0, 1)) / out.size).reshape(hs[-1].shape)
            critic_opt.step(critic.backward(hs, d_out), C.NET_MAX_GRAD_NORM)

        acts_flat = acts.reshape(k, batch * n)
        old_flat = old_logp.reshape(k, batch * n)
        adv_flat = adv.reshape(k, batch * n)
        samples = batch * n
        for _ in range(C.NET_EPOCHS):
            for idx in np.array_split(rng.permutation(samples), C.NET_MINIBATCHES):
                logits, hs_a = actor_logits(x_flat[:, idx])
                d_logits = ppo_logit_grad(
                    logits, acts_flat[:, idx], adv_flat[:, idx], old_flat[:, idx], clip_value, C.NET_ENTROPY,
                ).reshape(hs_a[-1].shape)
                actor_opt.step(actor.backward(hs_a, d_logits), C.NET_MAX_GRAD_NORM)

        done += batch
        for c in checkpoints:
            if done >= c and done - batch < c:
                snapshots[c] = [w.copy() for w in actor.params()]

    tail = diag[-max(1, len(diag) // 10):]
    critic_info = {
        "kind": "net_central" if method == C.METHOD_MAPPO else "net_local",
        "params": [w.copy() for w in critic.params()], "mu": mu, "sd": sd,
    }
    return (
        [w.copy() for w in actor.params()], critic_info, snapshots, done,
        float(np.mean([t[0] for t in tail])), float(np.mean([t[1] for t in tail])),
    )


# --- Training: Tabelle --------------------------------------------------------------

def _train_table(sampler, geom, n_episodes, train_seed, method, clip, checkpoints):
    k, n = geom.k, geom.n
    n_states = n_local_buckets(geom)
    z = np.zeros((k, n_states, N_ACTIONS))
    opt = Adam([z], C.TABLE_LR_POLICY)
    values_table = np.zeros(n_global_buckets(geom)) if method == C.METHOD_MAPPO else np.zeros((k, n_states))
    rng = np.random.default_rng([train_seed, 7])
    batch = C.TABLE_BATCH_EPISODES
    clip_value = effective_clip(clip)
    n_iter = max(1, n_episodes // batch)
    done = 0
    snapshots = {}
    diag = []
    agents = np.arange(k)

    for _ in range(n_iter):
        pos, dur = sampler.sample(batch)
        local_states = np.zeros((batch, n, k), dtype=int)
        global_states = np.zeros((batch, n), dtype=int)
        acts = np.zeros((batch, n, k), dtype=int)
        old_logp = np.zeros((batch, n, k))

        def act_fn(j, free, agent_pos, travel, job_pos, job_dur):
            states = local_bucket(geom, j, free, travel)
            p = softmax(z[agents[None, :], states])
            u = rng.random((batch, k, 1))
            a = (u > np.cumsum(p, -1)).sum(-1).clip(0, N_ACTIONS - 1)
            local_states[:, j] = states
            acts[:, j] = a
            old_logp[:, j] = np.log(np.take_along_axis(p, a[..., None], -1)[..., 0] + 1e-12)
            if method == C.METHOD_MAPPO:
                global_states[:, j] = global_bucket(geom, j, free)
            return a

        makespan = simulate(geom, pos, dur, act_fn)
        reward = -makespan / C.REWARD_SCALE
        if method == C.METHOD_MAPPO:
            values = values_table[global_states]
        else:
            values = values_table[agents[None, None, :], local_states]
        adv_raw = gae(values, reward, C.TABLE_GAE_LAMBDA)
        target = adv_raw + values
        diag.append((_explained_variance(target, values), float(adv_raw.var())))

        if method == C.METHOD_MAPPO:
            adv = np.repeat(normalize(adv_raw)[:, :, None], k, 2)
        else:
            adv = np.stack([normalize(adv_raw[:, :, a]) for a in range(k)], -1)

        if method == C.METHOD_MAPPO:
            table_value_update(values_table, global_states.ravel(), (target - values).ravel(), C.TABLE_LR_VALUE)
        else:
            for a in range(k):
                table_value_update(
                    values_table[a], local_states[:, :, a].ravel(), (target[:, :, a] - values[:, :, a]).ravel(),
                    C.TABLE_LR_VALUE,
                )

        states_flat = local_states.reshape(batch * n, k)
        acts_flat = acts.reshape(batch * n, k)
        adv_flat = adv.reshape(batch * n, k)
        old_flat = old_logp.reshape(batch * n, k)
        for _ in range(C.TABLE_EPOCHS):
            grad = np.zeros_like(z)
            for a in range(k):
                s = states_flat[:, a]
                g = ppo_logit_grad(z[a][s], acts_flat[:, a], adv_flat[:, a], old_flat[:, a], clip_value, C.TABLE_ENTROPY)
                for c in range(N_ACTIONS):
                    grad[a, :, c] += np.bincount(s, weights=g[:, c], minlength=n_states)
            opt.step([grad], 0.0)

        done += batch
        for c in checkpoints:
            if done >= c and done - batch < c:
                snapshots[c] = z.copy()

    tail = diag[-max(1, len(diag) // 10):]
    critic_info = {"kind": "table_global" if method == C.METHOD_MAPPO else "table_local", "values": values_table.copy()}
    return (
        z.copy(), critic_info, snapshots, done,
        float(np.mean([t[0] for t in tail])), float(np.mean([t[1] for t in tail])),
    )


def train_ppo(
    base_instance, env_mode, sigma, duration_variability, n_episodes, train_seed,
    method=C.DEFAULT_METHOD, actor_kind=C.DEFAULT_ACTOR, scope=C.DEFAULT_SCOPE, clip=C.DEFAULT_CLIP,
    use_job_index=True, checkpoints=(),
):
    """Trainiert IPPO oder MAPPO. `scope` (nur Netz + MAPPO): was der zentrale Kritiker sieht. `clip=None`
    schaltet das PPO-Clipping ab. `checkpoints`: Episodenzahlen, deren Akteur-Zwischenstand gespeichert wird.
    Der Trainings-Zufall hängt nur von `train_seed` ab."""
    started = time.perf_counter()
    geom = geometry(base_instance)
    sampler = Sampler(env_mode, base_instance, sigma, duration_variability, train_seed)
    if actor_kind == C.ACTOR_TABLE:
        scope, use_job_index = C.SCOPE_JOINT, True
        actor, critic, snapshots, done, critic_ev, adv_var = _train_table(
            sampler, geom, n_episodes, train_seed, method, clip, checkpoints,
        )
    else:
        if method == C.METHOD_IPPO:
            scope = C.SCOPE_JOINT
        actor, critic, snapshots, done, critic_ev, adv_var = _train_net(
            sampler, geom, n_episodes, train_seed, method, scope, clip, use_job_index, checkpoints,
        )
    return PPOResult(
        method=method, actor_kind=actor_kind, scope=scope, clip=clip, use_job_index=use_job_index,
        n_jobs=geom.n, n_agents=geom.k, actor=actor, critic=critic, snapshots=snapshots,
        n_episodes=n_episodes, n_decisions=done * geom.n * geom.k, wall_time_s=time.perf_counter() - started,
        critic_ev=critic_ev, adv_var=adv_var,
    )


# --- Einsatz (dezentral: nur lokale Beobachtung) --------------------------------------

def _net_logits(geom, result, actor_params, j, free, agent_pos, travel, job_pos, job_dur):
    x = add_agent_id(local_features(geom, j, free, travel, job_pos, job_dur, agent_pos, result.use_job_index))
    weights, biases = unpack(actor_params)
    out = forward_params(weights, biases, x.reshape(1, -1, x.shape[-1]))[-1]
    return out.reshape(x.shape[0], x.shape[1], N_ACTIONS)  # [k, B, 3]


def action_probs(geom, result, j, free, agent_pos, travel, job_pos, job_dur, actor_params=None):
    """Aktionswahrscheinlichkeiten [B, k, 3] aller Agenten - jeder aus seiner LOKALEN Beobachtung."""
    params = result.actor if actor_params is None else actor_params
    if result.actor_kind == C.ACTOR_TABLE:
        states = local_bucket(geom, j, free, travel)
        return softmax(params[np.arange(geom.k)[None, :], states])
    return softmax(_net_logits(geom, result, params, j, free, agent_pos, travel, job_pos, job_dur)).transpose(1, 0, 2)


def greedy_act_fn(geom, result, actor_params=None):
    """Gierige Ausführung (Argmax, Tie -> Aktion 0) für `simulate`."""
    params = result.actor if actor_params is None else actor_params

    def act_fn(j, free, agent_pos, travel, job_pos, job_dur):
        return action_probs(geom, result, j, free, agent_pos, travel, job_pos, job_dur, params).argmax(-1)

    return act_fn


def stochastic_act_fn(geom, result, rng, actor_params=None):
    """Zieht Aktionen aus der Policy (für Trajektorien-Sammlung, z.B. die Kritiker-Vermessung)."""

    def act_fn(j, free, agent_pos, travel, job_pos, job_dur):
        p = action_probs(geom, result, j, free, agent_pos, travel, job_pos, job_dur, actor_params)
        u = rng.random(p.shape[:-1] + (1,))
        return (u > np.cumsum(p, -1)).sum(-1).clip(0, N_ACTIONS - 1)

    return act_fn


def trace_policy(geom, result, instance):
    """Ein gieriger Durchlauf auf EINER Instanz mit allem, was die Schritt-Ansicht zeigt: Aktionswahrscheinlichkeiten
    und gewählte Aktionen je Auftrag und Agent, Zustand vor jeder Entscheidung sowie die Vorhersage des
    Kritikers (zentral: ein Wert je Auftrag; lokal: ein Wert je Agent)."""
    pos, dur = arrays_from_instances([instance])
    steps = []

    def act_fn(j, free, agent_pos, travel, job_pos, job_dur):
        probs = action_probs(geom, result, j, free, agent_pos, travel, job_pos, job_dur)
        actions = probs.argmax(-1)
        steps.append({
            "j": j, "free": free.copy(), "apos": agent_pos.copy(), "travel": travel.copy(),
            "jp": job_pos.copy(), "jd": job_dur.copy(), "probs": probs[0].copy(), "actions": actions[0].copy(),
        })
        return actions

    makespan = float(simulate(geom, pos, dur, act_fn)[0])
    return {
        "makespan": makespan,
        "probs": np.array([s["probs"] for s in steps]),        # [n, k, 3]
        "actions": np.array([s["actions"] for s in steps]),    # [n, k]
        "free": np.array([s["free"][0] for s in steps]),       # [n, k]
        "travel": np.array([s["travel"][0] for s in steps]),   # [n, k]
        "critic_makespan": critic_predictions(geom, result, steps, pos, dur),
    }


def critic_predictions(geom, result, steps, pos, dur):
    """Vom Kritiker vorhergesagter Makespan vor jedem Auftrag: [n] (zentral) oder [n, k] (lokal)."""
    critic = result.critic
    if critic["kind"].startswith("table"):
        values = critic["values"]
        if critic["kind"] == "table_global":
            out = [values[global_bucket(geom, s["j"], s["free"])[0]] for s in steps]
        else:
            out = [values[np.arange(geom.k), local_bucket(geom, s["j"], s["free"], s["travel"])[0]] for s in steps]
        return -np.array(out) * C.REWARD_SCALE
    weights, biases = unpack(critic["params"])
    out = []
    for s in steps:
        if critic["kind"] == "net_central":
            x = central_features(
                geom, s["j"], s["free"], s["travel"], s["jp"], s["jd"], s["apos"], pos, dur, result.scope,
            ).reshape(1, 1, -1)
            v = forward_params(weights, biases, x)[-1].reshape(())
        else:
            x = add_agent_id(local_features(
                geom, s["j"], s["free"], s["travel"], s["jp"], s["jd"], s["apos"], result.use_job_index,
            ))
            v = forward_params(weights, biases, x.reshape(1, -1, x.shape[-1]))[-1].reshape(geom.k)
        out.append(v * critic["sd"] + critic["mu"])
    return -np.array(out) * C.REWARD_SCALE
