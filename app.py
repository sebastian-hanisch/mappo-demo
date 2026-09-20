"""
MAPPO / CTDE an der Kran-Auftragsvergabe – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der "Konzepte"-Reihe, Multi-Agenten-Koordinations-Linie - Fortsetzung von marl-demo (unabhängiges
Q-Learning). CTDE (Centralized Training, Decentralized Execution): ein zentraler Kritiker sieht im Training den
gemeinsamen Zustand aller Agenten, im Einsatz entscheidet jeder Akteur nur aus seiner lokalen Beobachtung.
MAPPO (zentraler Kritiker) wird gegen IPPO (lokale Kritiker) als Ablation gestellt - und gegen IQL. Was der
zentrale Kritiker tatsächlich ändert, ist gemessen: nicht das, was man erwartet.
"""

import time
from dataclasses import replace

import streamlit as st

import cn_constants as C
from cn_evaluation import stats_up_to_step
from cn_presets import (
    CLIP_KEYS,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    parse_clip,
    randomize_seed,
    sync_query_params,
    training_key,
)
from cn_protocol import run_protocol
from cn_scenario import generate_instance
from cn_visualization import build_bid_chart, build_schedule_figure
from marl_env import heldout_instances
from marl_evaluation import heldout_optima, learning_curve as iql_learning_curve
from marl_exact import fixed_order_ceiling
from marl_iql import train_iql
from marl_visualization import build_crossplay_heatmap, build_policy_step_chart, build_ratio_histogram
from mappo_env import arrays_from_instances, cnp_makespans, geometry, replay_dispatch
from mappo_evaluation import (
    comparison,
    critic_lab,
    generalisation_check,
    iql_generalisation,
    learning_curve,
    miniature_results,
    seed_lottery,
)
from mappo_ppo import train_ppo, trace_policy
from mappo_visualization import (
    METHOD_NAMES,
    build_action_probability_chart,
    build_critic_prediction_chart,
    build_explained_variance_chart,
    build_learning_curves,
    build_lottery_comparison,
    build_policy_heatmap,
)

st.set_page_config(page_title="MAPPO (CTDE) – Sebastian Hanisch", layout="wide")

CHECKPOINT_CHOICES = C.EPISODES_CHOICES


def _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    return generate_instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)


def _unpack(tkey):
    (n_jobs, n_agents, var, travel, seed, env_mode, sigma, episodes, train_seed,
     actor_kind, scope, clip, use_job_index) = tkey
    return _instance(n_jobs, n_agents, var, travel, seed), var, seed, env_mode, sigma, episodes, train_seed, \
        actor_kind, scope, clip, use_job_index


@st.cache_data(show_spinner=False)
def _compute_cnp(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    instance = _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)
    return instance, run_protocol(instance)


@st.cache_data(show_spinner=False)
def _compute_heldout(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, env_mode, sigma):
    """Held-out-Instanzen samt CNP- und CP-SAT-Werten - hängt nur vom Szenario ab, nicht vom Training."""
    instance = _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)
    held = heldout_instances(instance, env_mode, sigma, duration_variability, seed)
    geom = geometry(instance)
    hpos, hdur = arrays_from_instances(held)
    return held, [float(x) for x in cnp_makespans(geom, hpos, hdur)], heldout_optima(held)


@st.cache_data(show_spinner=False)
def _compute_iql(tkey):
    instance, var, _, env_mode, sigma, episodes, train_seed, *_ = _unpack(tkey)
    checkpoints = [c for c in CHECKPOINT_CHOICES if c < episodes]
    return train_iql(instance, env_mode, sigma, episodes, train_seed, var, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _compute_ppo(tkey, method):
    instance, var, _, env_mode, sigma, episodes, train_seed, actor_kind, scope, clip, use_job_index = _unpack(tkey)
    checkpoints = [c for c in CHECKPOINT_CHOICES if c < episodes]
    return train_ppo(
        instance, env_mode, sigma, var, episodes, train_seed, method, actor_kind, scope, clip, use_job_index,
        checkpoints=checkpoints,
    )


@st.cache_data(show_spinner=False)
def _compute_evaluation(tkey):
    instance, var, seed, env_mode, sigma, *_ = _unpack(tkey)
    n_jobs, n_agents, _, travel = tkey[0], tkey[1], tkey[2], tkey[3]
    iql = _compute_iql(tkey)
    ppo = {m: _compute_ppo(tkey, m) for m in (C.METHOD_IPPO, C.METHOD_MAPPO)}
    held, held_cnp, held_opt = _compute_heldout(n_jobs, n_agents, var, travel, seed, env_mode, sigma)
    return {
        "cmp": comparison(instance, held, held_cnp, held_opt, iql.q, ppo, C.ORTOOLS_TIME_LIMIT_SECONDS),
        "curves": {
            "iql": iql_learning_curve(iql, instance, held),
            **{m: learning_curve(r, instance, held) for m, r in ppo.items()},
        },
        "gen": {
            "iql": iql_generalisation(iql.q, instance, var, seed),
            **{m: generalisation_check(r, instance, var, seed) for m, r in ppo.items()},
        },
        "ceiling": fixed_order_ceiling(instance),
        "lab": critic_lab(instance, env_mode, sigma, var, ppo[C.METHOD_MAPPO]),
    }


@st.cache_data(show_spinner=False)
def _compute_lottery(tkey):
    instance, var, seed, env_mode, sigma, episodes, train_seed, actor_kind, scope, clip, use_job_index = _unpack(tkey)
    held, held_cnp, _ = _compute_heldout(tkey[0], tkey[1], var, tkey[3], seed, env_mode, sigma)
    return seed_lottery(
        instance, env_mode, sigma, var, episodes, train_seed, held, held_cnp, actor_kind, scope, clip, use_job_index,
    )


@st.cache_data(show_spinner=False)
def _compute_miniature():
    return miniature_results()


def _fmt_int(n):
    """Tausendertrennung deutsch (Punkt)."""
    return f"{n:,}".replace(",", ".")


def _fmt_seconds(seconds):
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds:.2f} s"


def _start_lottery(key):
    st.session_state["lottery_owner"] = key


def _start_miniature():
    st.session_state["miniature_started"] = True


def _delta_metric(column, label, value, reference, reference_label, help_text=None):
    """Delta-Regel des Portfolios: Wert DIESER Karte minus Referenz, niedriger ist besser."""
    delta = value - reference
    if abs(delta) < 1e-6:
        column.metric(label, f"{value:.1f} min", delta="±0.0 min", delta_color="off", help=help_text)
    else:
        column.metric(
            label, f"{value:.1f} min", delta=f"{delta:+.1f} min ggü. {reference_label}",
            delta_color="inverse", help=help_text,
        )


def _episodes_to_reach(points, cnp_heldout_mean, threshold_pct):
    """Erster Checkpoint, dessen Held-out-Mittel mindestens threshold_pct besser als Contract Net ist."""
    for episodes, _, heldout_mean in points:
        if (heldout_mean - cnp_heldout_mean) / cnp_heldout_mean * 100.0 <= -threshold_pct:
            return episodes
    return None


st.title("🎛️ MAPPO / CTDE an der Kran-Auftragsvergabe")
st.markdown(
    """
Fortsetzung von **marl-demo** (unabhängiges Q-Learning). Dort war die Schwäche die **Nicht-Stationarität**:
jeder Agent lernt, während die anderen sich gleichzeitig ändern. **CTDE** - *Centralized Training,
Decentralized Execution* - setzt genau dort an: ein **zentraler Kritiker** sieht im Training den gemeinsamen
Zustand aller Agenten, im Einsatz entscheidet jeder **Akteur** nur aus seiner lokalen Beobachtung.
**MAPPO** (Yu et al. 2022, PPO mit zentralem Kritiker) ist das etablierte Verfahren; **IPPO** (jeder Agent mit
eigenem, lokalem Kritiker) ist die Ablation, die den Effekt des Kritikers isoliert.
"""
)
st.caption(
    "Was der zentrale Kritiker wirklich ändert, ist hier gemessen - und nicht das, was man erwartet: das PPO-Verfahren "
    "selbst (v.a. das Clipping) stabilisiert das Training, der Kritiker senkt vor allem die Varianz der Schätzung. "
    "MAPPO/CTDE ist nur innerhalb kooperativen Multi-Agenten-Lernens der Standard; das zentrale CP-SAT bleibt der "
    "praktische Industriestandard."
)

with st.expander("Wie funktioniert diese Demo?", expanded=True):
    st.markdown(
        r"""
**Dieselbe Umgebung wie marl-demo**: Aufträge werden nacheinander angekündigt, jeder Agent (Kran) wählt eine
Aktion - `normal bieten`, `hoch bieten` (+25 min, lehnt eher ab) oder `niedrig bieten` (−25 min, greift eher) -,
den Zuschlag vergibt weiter Contract Nets Regel. Belohnung für alle: `−Makespan / 10`, erst am Ende. Ein
untrainierter Akteur verhält sich **exakt wie Contract Net**.

**Akteur und Kritiker** (Actor-Critic): der **Akteur** ist die Policy - er wählt die Aktion aus der lokalen
Beobachtung (eigene Freizeit, Anfahrt, Auftrag). Der **Kritiker** schätzt, wie gut ein Zustand ist (den erwarteten
Makespan), und liefert damit den *Vorteil* (`Advantage`) jeder Entscheidung: *war die Aktion besser oder schlechter
als erwartet?* Nur der Akteur wird im Einsatz gebraucht - **der Kritiker existiert nur im Training**.

- **IPPO**: jeder Agent hat einen eigenen Kritiker, der nur seine lokale Beobachtung sieht.
- **MAPPO**: ein **zentraler** Kritiker sieht den Zustand *aller* Agenten. Optional zusätzlich die Liste der
  künftigen Aufträge (**privilegiert**: Wissen, das im Einsatz niemand hat).

**PPO in einem Satz**: Die Policy wird nur in kleinen Schritten verbessert - ein *Clipping* schneidet
Änderungen ab, die das Verhältnis neue/alte Wahrscheinlichkeit über `1 ± ε` treiben. Das ist der Baustein, der
Konventionen davor bewahrt, sich gegenseitig aufzuschaukeln (ausschalten im Bereich "Erweitert").

**Akteur-Arten**: *Netz* - ein kleines neuronales Netz (2 x 16 tanh, geteilte Parameter, Agenten-ID als Eingang)
auf stetigen Merkmalen; *Tabelle* - eine Softmax-Tabelle je Agent über dieselben Buckets wie IQL. Beide reine
numpy-Implementierungen.

**Trainingsumgebung**: *Wiederkehrend* (dieselben Aufträge, Dauern × `exp(N(0, σ))`) oder *zufällige Instanzen*.
Wie in marl-demo wird jede Policy zusätzlich auf 30 Instanzen getestet, die das Training nie gesehen hat
(**Held-out**), und mit Contract Net, IQL und dem zentralen CP-SAT-Optimum verglichen. IQL, IPPO und MAPPO
bekommen immer dieselbe Trainingsmenge.

**Was gemessen wird**: Kosten und Ergebnis, Stabilität über Trainings-Seeds (Seed-Lotterie, Cross-Play), die
IPPO-vs-MAPPO-Ablation und eine **Kritiker-Vermessung** - wie viel der Streuung des Ergebnisses ein lokaler, ein
zentraler und ein privilegierter Kritiker auf *identischen* Trajektorien erklären.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = C.PRESET_HELP
preset_names = list(C.PRESETS.keys())
for row_start in range(0, len(preset_names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, preset_names[row_start:row_start + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Anzahl Aufträge", *bounds("n_jobs_slider"), key="n_jobs_slider")
    n_agents = st.slider("Anzahl Agenten (Kräne)", *bounds("n_agents_slider"), key="n_agents_slider")
    duration_variability = st.slider(
        "Streuung der Auftragsdauer", *bounds("duration_variability_slider"), key="duration_variability_slider",
    )
    travel_time_per_unit = st.slider(
        "Anfahrtszeit pro Positionseinheit", *bounds("travel_time_per_unit_slider"),
        key="travel_time_per_unit_slider",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.button(
        "🎲 Neue Instanz generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für Auftragspositionen und -dauern.",
    )

    st.markdown("**Training**")
    env_mode = st.radio(
        "Trainingsumgebung", options=list(C.ENV_LABELS), format_func=lambda k: C.ENV_LABELS[k],
        key="env_mode_radio",
        help="Wiederkehrend: immer dieselben Aufträge, nur die Dauern schwanken. Zufällig: jede Episode "
        "eine neue Instanz.",
    )
    if env_mode == C.ENV_RECURRING:
        sigma = st.slider(
            "Rauschen σ der Dauern", C.SIGMA_MIN, C.SIGMA_MAX, key="sigma_slider", step=0.05,
            help="Jede Dauer wird pro Trainings-Episode mit exp(N(0, σ)) multipliziert. σ = 0: Training "
            "und Test identisch.",
        )
    else:
        # Ein nicht gerenderter Widget-Key verliert sonst seinen Wert (Permalink, Rückwechsel).
        st.session_state["sigma_slider"] = st.session_state["sigma_slider"]
        sigma = st.session_state["sigma_slider"]
    episodes = st.select_slider(
        "Trainings-Episoden", options=C.EPISODES_CHOICES, key="episodes_slider",
        format_func=lambda x: f"{x:,}".replace(",", "."),
        help="Mehr Episoden = mehr Erfahrung, aber auch längere Rechenzeit. IQL, IPPO und MAPPO bekommen dieselbe Menge.",
    )
    train_seed = st.number_input(
        "Trainings-Seed", *bounds("train_seed_input"), key="train_seed_input", step=1,
        help="Zufall des Lernens (Exploration, Initialisierung) - unabhängig vom Szenario-Seed.",
    )

    st.markdown("**Verfahren**")
    method = st.radio(
        "Angezeigtes Verfahren", options=list(C.METHOD_LABELS), format_func=lambda k: C.METHOD_LABELS[k],
        key="method_radio",
        help="IPPO und MAPPO werden immer beide trainiert und verglichen. Hier wählen Sie, welches die "
        "Detailansichten (Phase 2) und die Verdict-Karte zeigen.",
    )
    actor_kind = st.radio(
        "Akteur", options=list(C.ACTOR_LABELS), format_func=lambda k: C.ACTOR_LABELS[k], key="actor_radio",
        help="Netz: kleines neuronales Netz mit stetigen Merkmalen. Tabelle: Softmax-Tabelle über Buckets (wie IQL).",
    )
    if actor_kind == C.ACTOR_NET:
        scope = st.radio(
            "Kritiker-Sicht (MAPPO)", options=list(C.SCOPE_LABELS), format_func=lambda k: C.SCOPE_LABELS[k],
            key="scope_radio",
            help="Was der zentrale Kritiker im Training sieht. 'Auftragsliste' ist privilegiertes Wissen, das im "
            "Einsatz niemand hat.",
        )
    else:
        st.session_state["scope_radio"] = st.session_state["scope_radio"]
        scope = st.session_state["scope_radio"]
    with st.expander("Erweitert"):
        clip_choice = st.radio(
            "PPO-Clipping ε", options=list(CLIP_KEYS), format_func=lambda k: "aus" if k == "off" else k.replace(".", ","),
            key="clip_select", horizontal=True,
            help="Schneidet zu große Policy-Änderungen ab. 'aus' zeigt, was ohne Clipping passiert.",
        )
        if actor_kind == C.ACTOR_NET:
            use_job_index = st.checkbox(
                "Auftrags-Index als Beobachtung", key="jidx_toggle",
                help="Der öffentliche Zähler j/n. Ohne ihn kann ein Netz einen wiederkehrenden Plan schlechter "
                "auswendig lernen.",
            )
        else:
            st.session_state["jidx_toggle"] = st.session_state["jidx_toggle"]
            use_job_index = True
    clip = parse_clip(clip_choice)

sync_query_params(
    n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, env_mode, episodes, sigma, train_seed,
    method, actor_kind, scope, clip, use_job_index,
)

scenario_key = (int(n_jobs), int(n_agents), duration_variability, travel_time_per_unit, int(seed))
tkey = training_key(
    *scenario_key, env_mode, sigma, episodes, train_seed, actor_kind, scope, clip, use_job_index,
)

with st.spinner("Führe Contract Net Protocol aus..."):
    instance, cnp_result = _compute_cnp(*scenario_key)
with st.spinner("Trainiere IQL, IPPO und MAPPO und werte aus (bei vielen Episoden einige Sekunden)..."):
    iql = _compute_iql(tkey)
    ppo = {m: _compute_ppo(tkey, m) for m in (C.METHOD_IPPO, C.METHOD_MAPPO)}
    evaluation = _compute_evaluation(tkey)
featured = ppo[method]
featured_name = METHOD_NAMES[method]
cmp = evaluation["cmp"]
ortools_makespan = cmp["ortools_nominal"] if cmp["ortools_feasible"] else None
geom = geometry(instance)
trace = trace_policy(geom, featured, instance)
final_dispatch = replay_dispatch(instance, trace["actions"])

# --- Phase 1: Contract Net Rekapitulation -----------------------------------

st.markdown("## 🎯 Phase 1: Contract Net Protocol (Rekapitulation)")

if "cn_step" not in st.session_state or st.session_state.get("cn_step_owner") != scenario_key:
    st.session_state["cn_step"] = instance.n_jobs - 1
    st.session_state["cn_step_owner"] = scenario_key

max_step = instance.n_jobs - 1
step_col, play_col = st.columns([5, 1])
with step_col:
    if max_step == 0:
        step = 0
        st.caption("Nur ein Auftrag - kein Regler nötig.")
    else:
        step = st.slider("Schritt (Auftragsvergabe)", 0, max_step, key="cn_step")
with play_col:
    auto_play_cnp = st.button("▶️ Abspielen", width="stretch", key="cnp_play")

chart_col, bid_col = st.columns([3, 2])
schedule_slot = chart_col.empty()
bid_slot = bid_col.empty()


def _render_cnp(current_step):
    schedule_slot.plotly_chart(
        build_schedule_figure(instance, cnp_result, current_step, ortools_makespan),
        width="stretch", key=f"cnp_schedule_{current_step}",
    )
    bid_slot.plotly_chart(
        build_bid_chart(cnp_result.steps[current_step]),
        width="stretch", key=f"cnp_bids_{current_step}",
    )


if auto_play_cnp:
    for s in range(0, max_step + 1):
        _render_cnp(s)
        time.sleep(0.4)
    step = max_step
else:
    _render_cnp(step)

live = stats_up_to_step(cnp_result, step)
lm1, lm2 = st.columns(2)
lm1.metric("Aufträge bisher vergeben", f"{live['jobs_awarded']} / {instance.n_jobs}")
lm2.metric("Aktuell schlechteste freie Zeit", f"{live['worst_agent_free_time']:.1f} min")

st.markdown("---")

# --- Phase 2a: Lernkurve -----------------------------------------------------

st.markdown(f"## 📈 Phase 2a: Was beim Training passiert ({featured_name})")
st.caption(
    "Jeder Punkt ist die gierige Policy nach so vielen Episoden, gemessen auf den Held-out-Instanzen. IQL, IPPO und "
    "MAPPO bekommen dieselbe Trainingsmenge; das Gantt zeigt die Policy des angezeigten Verfahrens auf der "
    "gezeigten Instanz."
)

curves = evaluation["curves"]
curve = curves[method]
learn_max = len(curve) - 1
learn_owner = (tkey, method)
if "learn_step" not in st.session_state or st.session_state.get("learn_owner") != learn_owner:
    st.session_state["learn_step"] = learn_max
    st.session_state["learn_owner"] = learn_owner

lstep_col, lplay_col = st.columns([5, 1])
with lstep_col:
    if learn_max == 0:
        learn_step = 0
        st.caption("Nur ein Zwischenstand - kein Regler nötig.")
    else:
        learn_step = st.slider(
            "Trainingsstand", 0, learn_max, key="learn_step", format="%d",
            help="0 = frühester Zwischenstand, letzter Wert = fertig trainiert.",
        )
with lplay_col:
    auto_play_learn = st.button("▶️ Abspielen", width="stretch", key="learn_play")

learn_left, learn_right = st.columns([3, 2])
learn_curve_slot = learn_left.empty()
learn_gantt_slot = learn_right.empty()
learn_caption_slot = st.empty()


def _render_learn(idx):
    ep, nominal, heldout_mean = curve[idx]
    snapshot = replace(featured, actor=featured.snapshots.get(ep, featured.actor))
    snap_trace = trace_policy(geom, snapshot, instance)
    snap_dispatch = replay_dispatch(instance, snap_trace["actions"])
    learn_curve_slot.plotly_chart(
        build_learning_curves(curves, cmp["heldout_cnp_mean"], cmp["heldout_opt_mean"], marker_episodes=ep),
        width="stretch", key=f"learn_curve_{idx}",
    )
    learn_gantt_slot.plotly_chart(
        build_schedule_figure(instance, snap_dispatch.protocol_result, instance.n_jobs - 1, ortools_makespan),
        width="stretch", key=f"learn_gantt_{idx}",
    )
    pct = (heldout_mean - cmp["heldout_cnp_mean"]) / cmp["heldout_cnp_mean"] * 100.0
    learn_caption_slot.caption(
        f"Nach {_fmt_int(ep)} Episoden ({featured_name}): Held-out-Mittel **{heldout_mean:.1f} min** "
        f"({pct:+.1f} % gegenüber Contract Net {cmp['heldout_cnp_mean']:.1f} min), gezeigte Instanz "
        f"**{nominal:.1f} min**."
    )


if auto_play_learn:
    for i in range(learn_max + 1):
        _render_learn(i)
        time.sleep(0.7)
    learn_step = learn_max
else:
    _render_learn(learn_step)

st.markdown("---")

# --- Phase 2b: Akteur und Kritiker im Einsatz --------------------------------

st.markdown(f"## 🔍 Phase 2b: Akteur und Kritiker im Einsatz ({featured_name})")
st.caption(
    "Pro Auftrag: die Aktionswahrscheinlichkeiten jedes Akteurs - jeder entscheidet nur aus seiner lokalen "
    "Beobachtung - und wie sich sein Gebot dadurch verschiebt. Rechts die Vorhersage des Kritikers, der nur im "
    "Training existiert."
)

if "policy_step" not in st.session_state or st.session_state.get("policy_step_owner") != learn_owner:
    st.session_state["policy_step"] = instance.n_jobs - 1
    st.session_state["policy_step_owner"] = learn_owner

pstep_col, pplay_col = st.columns([5, 1])
with pstep_col:
    if max_step == 0:
        policy_step = 0
        st.caption("Nur ein Auftrag - kein Regler nötig.")
    else:
        policy_step = st.slider("Schritt (Auftragsvergabe)", 0, max_step, key="policy_step")
with pplay_col:
    auto_play_policy = st.button("▶️ Abspielen", width="stretch", key="policy_play")

prow1_left, prow1_right = st.columns([3, 2])
pgantt_slot = prow1_left.empty()
pbid_slot = prow1_right.empty()
prow2_left, prow2_right = st.columns([3, 2])
pprob_slot = prow2_left.empty()
pinfo_slot = prow2_right.empty()


def _render_policy(s):
    pgantt_slot.plotly_chart(
        build_schedule_figure(instance, final_dispatch.protocol_result, s, ortools_makespan),
        width="stretch", key=f"policy_gantt_{s}",
    )
    pbid_slot.plotly_chart(build_policy_step_chart(final_dispatch, s), width="stretch", key=f"policy_bids_{s}")
    pprob_slot.plotly_chart(
        build_action_probability_chart(trace, s), width="stretch", key=f"policy_probs_{s}",
    )
    lines = []
    for agent in range(instance.n_agents):
        probs = trace["probs"][s][agent]
        action = trace["actions"][s][agent]
        lines.append(
            f"- **Agent {agent + 1}**: eigene Freizeit {trace['free'][s][agent]:.0f} min, Anfahrt "
            f"{trace['travel'][s][agent]:.0f} min → *{C.ACTION_NAMES[action]}* (p = {probs[action]:.2f})"
        )
    pinfo_slot.markdown("\n".join(lines))


if auto_play_policy:
    for s in range(0, max_step + 1):
        _render_policy(s)
        time.sleep(0.6)
    policy_step = max_step
else:
    _render_policy(policy_step)

critic_left, critic_right = st.columns([3, 2])
with critic_left:
    st.plotly_chart(
        build_critic_prediction_chart(trace, method), width="stretch", key="critic_prediction",
    )
with critic_right:
    st.caption(
        f"**Was der Kritiker vorhersagt.** {'Ein zentraler Kritiker sieht den Zustand aller Agenten' if method == C.METHOD_MAPPO else 'Jeder Agent hat einen eigenen Kritiker, der nur seine lokale Beobachtung sieht'} "
        "und schätzt vor jedem Auftrag den erwarteten Makespan. Die Vorhersage bezieht sich auf die stochastische "
        "Trainings-Policy - die gierige Ausführung (gestrichelt) kann davon abweichen."
    )
    if method == C.METHOD_MAPPO and featured.scope == C.SCOPE_FULL:
        st.caption("Dieser Kritiker sieht zusätzlich die künftigen Aufträge (privilegiert) - er kennt die "
                   "Schwierigkeit der Instanz, die der Akteur nie sieht.")

with st.expander("🗺️ Policy-Landkarte: wann lehnt ein Agent ab, wann greift er zu?"):
    heatmap_fig, heatmap_job = build_policy_heatmap(geom, featured)
    st.plotly_chart(heatmap_fig, width="stretch", key="policy_heatmap")
    st.caption(
        f"Aktionswahrscheinlichkeiten von Agent 1 bei Auftrag {heatmap_job + 1} (mittlere Position und Dauer), je "
        f"nach eigener Freizeit und Anfahrt. {'Beim Tabellen-Akteur stufig, weil er nur Buckets kennt.' if featured.actor_kind == C.ACTOR_TABLE else 'Stetig, weil das Netz stetige Merkmale bekommt.'}"
    )

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------

st.subheader("📐 Der Preis des zentralen Kritikers")

learners = cmp["learners"]
feat = learners[method]
n_pct, h_pct = feat["nominal_pct"], feat["heldout_pct"]
gen = evaluation["gen"]

lottery_key = tkey
lottery_active = (
    st.session_state.get("lottery_owner") == lottery_key
    or st.session_state.get("lottery_preset_key") == lottery_key
)
lot = None
if lottery_active:
    with st.spinner("Trainiere mehrere Läufe für die Seed-Lotterie (IQL, IPPO, MAPPO)..."):
        lot = _compute_lottery(lottery_key)

st.markdown("**Die gezeigte Instanz (nominal, ohne Rauschen)**")
cols = st.columns(5)
cols[0].metric("Contract Net (roh)", f"{cmp['cnp_nominal']:.1f} min")
for col, name in zip(cols[1:4], ("iql", C.METHOD_IPPO, C.METHOD_MAPPO)):
    _delta_metric(col, METHOD_NAMES[name], learners[name]["nominal"], cmp["cnp_nominal"], "CNP")
if cmp["ortools_feasible"]:
    _delta_metric(
        cols[4], "CP-SAT (zentral)", cmp["optimum_reference"], feat["nominal"], featured_name,
        help_text=f"Echter industrieller Solver, {cmp['ortools_wall_time']:.2f}s - "
        + ("beweist Optimalität." if cmp["ortools_optimal"] else "Zeitlimit erreicht, beste gefundene Lösung.")
        + " CP-SAT rundet Zeiten auf; der Wert ist deshalb nie größer als eine zulässige Lösung angesetzt.",
    )
else:
    cols[4].metric("CP-SAT (zentral)", "kein Ergebnis im Zeitlimit")

st.markdown(f"**Held-out: Mittel über {C.N_HELDOUT} Instanzen, die das Training nie gesehen hat**")
hcols = st.columns(5)
hcols[0].metric("Contract Net (roh)", f"{cmp['heldout_cnp_mean']:.1f} min")
for col, name in zip(hcols[1:4], ("iql", C.METHOD_IPPO, C.METHOD_MAPPO)):
    _delta_metric(col, METHOD_NAMES[name], learners[name]["heldout_mean"], cmp["heldout_cnp_mean"], "CNP")
if cmp["heldout_opt_mean"] is not None:
    _delta_metric(hcols[4], "CP-SAT (zentral)", cmp["heldout_opt_mean"], feat["heldout_mean"], featured_name)
else:
    hcols[4].metric("CP-SAT (zentral)", "nicht für alle im Zeitlimit")

closed_text = ""
if cmp["heldout_opt_mean"] is not None and cmp["heldout_cnp_mean"] > cmp["heldout_opt_mean"]:
    closed = (cmp["heldout_cnp_mean"] - feat["heldout_mean"]) / (cmp["heldout_cnp_mean"] - cmp["heldout_opt_mean"]) * 100
    closed_text = f" {featured_name} schließt **{closed:.0f} %** der Held-out-Lücke von Contract Net zum zentralen Optimum."
st.caption(
    f"{featured_name} vs. Contract Net: nominal **{n_pct:+.1f} %**, Held-out **{h_pct:+.1f} %**.{closed_text}"
)

lot_feat = lot[method] if lot is not None else None
if n_pct > C.WORSE_THAN_CNP_THRESHOLD_PCT or h_pct > C.WORSE_THAN_CNP_THRESHOLD_PCT:
    reasons = []
    if env_mode == C.ENV_RANDOM and actor_kind == C.ACTOR_TABLE:
        reasons.append("Die Tabelle merkt sich, was wiederkehrt - auf immer neuen Instanzen gibt es nichts einzuprägen.")
    if episodes <= 300:
        reasons.append("Mit so wenig Episoden ist noch kaum etwas gelernt.")
    if clip is None:
        reasons.append("Ohne Clipping schaukeln sich Konventionen leichter auf.")
    if not reasons:
        reasons.append("Die Agenten haben eine Konvention gelernt, die auf dieser Instanz oder ihren Varianten nicht trägt.")
    st.warning(
        f"⚠️ **{featured_name} ist schlechter als Contract Net**: nominal {n_pct:+.1f} %, Held-out {h_pct:+.1f} %. "
        f"Gelernt heißt nicht besser - es gibt keine Garantie. " + " ".join(reasons)
    )
elif n_pct <= -C.CLEARLY_BETTER_THRESHOLD_PCT and h_pct <= -C.CLEARLY_BETTER_THRESHOLD_PCT:
    unstable = lot_feat is not None and (
        lot_feat["n_worse_than_cnp"] > 0 or lot_feat["heldout_std_pct"] >= C.LOTTERY_SPREAD_WARNING_PCT
    )
    if unstable:
        st.warning(
            f"⚠️ **Dieser Lauf schlägt Contract Net** (nominal {n_pct:+.1f} %, Held-out {h_pct:+.1f} %) - aber die "
            f"Seed-Lotterie unten zeigt, dass das Ergebnis vom Trainings-Seed abhängt: Streuung "
            f"{lot_feat['heldout_std_pct']:.1f} Punkte (Held-out von {min(lot_feat['heldout_pct']):+.1f} % bis "
            f"{max(lot_feat['heldout_pct']):+.1f} %), {lot_feat['n_worse_than_cnp']} von {len(lot['seeds'])} Läufen "
            f"schlechter als Contract Net."
        )
    elif feat["lose_frac"] > C.LOSE_FRACTION_WARNING:
        st.warning(
            f"⚠️ **Im Mittel besser, aber nicht überall**: nominal {n_pct:+.1f} %, Held-out {h_pct:+.1f} % - doch auf "
            f"**{feat['lose_frac'] * 100:.0f} %** der Held-out-Instanzen ist {featured_name} schlechter als Contract Net, "
            f"die schlechteste Instanz liegt bei **{feat['worst_pct']:+.0f} %**. Auf frischen Szenarien gleicher Größe: "
            f"{gen[method]['fresh_vs_cnp_pct']:+.1f} %."
        )
    else:
        stability = (
            "Wie stabil das über andere Trainings-Seeds ist, zeigt die Seed-Lotterie unten"
            if lot is None else "In der Seed-Lotterie unten liegen alle Trainingsläufe nah beieinander"
        )
        st.success(
            f"✅ **Lernen zahlt sich hier aus**: nominal {n_pct:+.1f} %, Held-out {h_pct:+.1f} % gegenüber Contract Net. "
            f"{stability}; auf frischen Szenarien gleicher Größe liegt dieselbe Policy bei "
            f"{gen[method]['fresh_vs_cnp_pct']:+.1f} %."
        )
else:
    st.info(
        f"Kaum Unterschied zu Contract Net (nominal {n_pct:+.1f} %, Held-out {h_pct:+.1f} %) - oder der Vorteil zeigt "
        f"sich nur auf einem der beiden."
    )

ippo_h, mappo_h = learners[C.METHOD_IPPO]["heldout_pct"], learners[C.METHOD_MAPPO]["heldout_pct"]
delta_pts = mappo_h - ippo_h
if abs(delta_pts) < C.ABLATION_NEGLIGIBLE_PTS:
    st.info(
        f"🔬 **Kritiker-Ablation**: IPPO {ippo_h:+.1f} % vs. MAPPO {mappo_h:+.1f} % (Held-out) - Unterschied "
        f"{delta_pts:+.1f} Punkte, **kein messbarer Unterschied**. Der zentrale Kritiker ist hier nicht die Ursache "
        f"des Gewinns. (Einzellauf; Rauschen etwa 1 Punkt.)"
    )
else:
    better = "MAPPO" if delta_pts < 0 else "IPPO"
    privileged = ppo[C.METHOD_MAPPO].scope == C.SCOPE_FULL
    if privileged:
        note = (
            "Der zentrale Kritiker kennt hier zusätzlich die Auftragsliste - Wissen, das im Einsatz niemand hat; der "
            "Vorteil kommt also zum Teil vom besseren Wissen über die Instanzschwierigkeit, nicht vom Verfahren."
        )
    else:
        note = (
            "Ein Einzellauf: das Rauschen liegt bei etwa 1 Punkt, und der Vorsprung des zentralen Kritikers schwankte in "
            "Voruntersuchungen je nach Hyperparametern zwischen 0,2 und 1,4 Punkten - vergleichen Sie mit der Seed-Lotterie."
        )
    st.info(
        f"🔬 **Kritiker-Ablation**: IPPO {ippo_h:+.1f} % vs. MAPPO {mappo_h:+.1f} % (Held-out) - {better} ist "
        f"{abs(delta_pts):.1f} Punkte besser. {note}"
    )

tab_cost, tab_guarantee, tab_stability, tab_critic = st.tabs(
    ["⏱️ Aufwand", "🎯 Keine Garantie", "🔀 Stabilität", "🔬 Kritiker-Vermessung"]
)

with tab_cost:
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Trainings-Episoden", _fmt_int(featured.n_episodes))
    t2.metric("IQL", _fmt_seconds(iql.wall_time_s), help="Trainingszeit auf diesem Rechner.")
    t3.metric("IPPO", _fmt_seconds(ppo[C.METHOD_IPPO].wall_time_s), help="Trainingszeit auf diesem Rechner.")
    t4.metric("MAPPO", _fmt_seconds(ppo[C.METHOD_MAPPO].wall_time_s), help="Trainingszeit auf diesem Rechner.")
    parts = []
    for name in ("iql", C.METHOD_IPPO, C.METHOD_MAPPO):
        reached = _episodes_to_reach(curves[name], cmp["heldout_cnp_mean"], 3.0)
        parts.append(
            f"**{METHOD_NAMES[name]}**: " + (f"ab {_fmt_int(reached)} Episoden" if reached is not None else "nicht in diesem Trainingsumfang")
        )
    st.markdown(
        "Erfahrung, bis das Held-out-Mittel mindestens 3 % besser als Contract Net ist: " + " · ".join(parts) + ". "
        f"CP-SAT löst diese Instanz in **{cmp['ortools_wall_time']:.2f} s** exakt und jedes neue Szenario einfach "
        "neu - Training lohnt sich nur, wenn dasselbe Szenario oft genug wiederkommt."
    )
    st.caption(
        f"{featured_name}: {_fmt_int(featured.n_decisions)} Agenten-Entscheidungen im Training. "
        "Kritiker-Vorteil (die Vorteilsschätzung wird ruhiger) kostet Rechenzeit im Training, nicht im Einsatz."
    )

with tab_guarantee:
    g1, g2, g3 = st.columns(3)
    g1.metric(
        "Held-out-Instanzen besser als Contract Net", f"{feat['beat_frac'] * 100:.0f} %",
        help=f"Anteil der {C.N_HELDOUT} Held-out-Instanzen, auf denen {featured_name} einen kleineren Makespan hat.",
    )
    g2.metric("Held-out-Instanzen schlechter als Contract Net", f"{feat['lose_frac'] * 100:.0f} %")
    g3.metric("Schlechteste Instanz", f"{feat['worst_pct']:+.0f} %", help="Größte Verschlechterung ggü. Contract Net.")
    st.plotly_chart(build_ratio_histogram(feat["ratios_pct"]), width="stretch", key="ratio_hist")
    f1, f2, f3 = st.columns(3)
    for col, name in zip((f1, f2, f3), ("iql", C.METHOD_IPPO, C.METHOD_MAPPO)):
        col.metric(
            f"Frische Szenarien: {METHOD_NAMES[name]}", f"{gen[name]['fresh_vs_cnp_pct']:+.1f} %",
            delta=f"besser in {gen[name]['fresh_beat_frac'] * 100:.0f} %, schlechter in {gen[name]['fresh_lose_frac'] * 100:.0f} %",
            delta_color="off",
            help=f"{C.N_FRESH_SCENARIOS} neue Instanzen gleicher Größe (neue Positionen und Dauern) gegenüber Contract Net.",
        )
    if env_mode == C.ENV_RECURRING:
        st.caption(
            "Im wiederkehrenden Szenario lernen alle Verfahren zu einem Teil den Plan auswendig - auf fremden "
            "Szenarien ist das schlechter als auf dem eigenen. Das Netz mit stetigen Merkmalen verliert dabei weniger "
            "als eine Tabelle."
        )
    st.caption(
        f"Obergrenze für jede Policy dieser Art: **{evaluation['ceiling']:.1f} min** (bestmögliche Zuteilung bei fester "
        f"Ankündigungsreihenfolge) - keine gelernte Policy kann sie unterbieten, und CP-SAT darf zusätzlich die "
        f"Reihenfolge innerhalb eines Agenten wählen."
    )

with tab_stability:
    st.markdown("**Gemessen: die Seed-Lotterie**")
    if lot is None:
        est = C.N_LOTTERY_SEEDS * (
            iql.wall_time_s + ppo[C.METHOD_IPPO].wall_time_s + ppo[C.METHOD_MAPPO].wall_time_s
        ) * min(1.0, C.LOTTERY_EPISODE_CAP / featured.n_episodes)
        st.button(
            f"🎰 Seed-Lotterie starten (IQL, IPPO, MAPPO x {C.N_LOTTERY_SEEDS} Trainingsläufe, ca. {est:.0f} s)",
            on_click=_start_lottery, args=(lottery_key,), key="lottery_start",
        )
        st.caption(
            "Trainiert jedes Verfahren mit denselben Einstellungen und anderen Trainings-Seeds und zeigt, wie stark das "
            "Ergebnis vom Zufall des Lernens abhängt - plus Cross-Play zwischen den Läufen."
        )
    else:
        st.plotly_chart(build_lottery_comparison(lot), width="stretch", key="lottery_chart")
        lc = st.columns(3)
        for col, name in zip(lc, ("iql", C.METHOD_IPPO, C.METHOD_MAPPO)):
            entry = lot[name]
            col.markdown(f"**{METHOD_NAMES[name]}**")
            col.metric("Läufe schlechter als Contract Net", f"{entry['n_worse_than_cnp']} von {len(lot['seeds'])}")
            col.metric("Streuung (Std, Held-out)", f"{entry['heldout_std_pct']:.1f} Punkte")
            col.metric("Cross-Play-Strafe", f"{entry['crossplay_penalty_pct']:+.1f} % von CNP")
        st.caption(
            "Cross-Play-Werte verschiedener Verfahren sind nur eingeschränkt vergleichbar: ein Verfahren, dessen Policies "
            "kaum vom Contract-Net-Verhalten abweichen oder stark streuen, zeigt kleine Strafen, ohne stabiler zu sein."
        )
        if lot["episodes"] < featured.n_episodes:
            st.caption(f"Aus Zeitgründen mit {_fmt_int(lot['episodes'])} statt {_fmt_int(featured.n_episodes)} Episoden trainiert (für alle Verfahren gleich).")
        iql_spread, ppo_spread = lot["iql"]["heldout_std_pct"], lot_feat["heldout_std_pct"]
        if lot_feat["heldout_std_pct"] >= C.LOTTERY_SPREAD_WARNING_PCT or lot_feat["n_worse_than_cnp"] > 0:
            st.warning(
                f"⚠️ **{featured_name} hängt vom Trainings-Seed ab**: Streuung {ppo_spread:.1f} Punkte, "
                f"{lot_feat['n_worse_than_cnp']} von {len(lot['seeds'])} Läufen schlechter als Contract Net. "
                + ("Ohne PPO-Clipping ist das zu erwarten - schalten Sie es unter *Erweitert* wieder ein. " if clip is None else "")
            )
        else:
            st.success(
                f"✅ **{featured_name} ist stabil über Trainings-Seeds**: Streuung {ppo_spread:.1f} Punkte gegenüber "
                f"{iql_spread:.1f} bei IQL. Der Gewinn stammt vom PPO-Verfahren - IPPO (lokale Kritiker) und MAPPO "
                f"unterscheiden sich kaum (Streuung {lot[C.METHOD_IPPO]['heldout_std_pct']:.1f} vs. "
                f"{lot[C.METHOD_MAPPO]['heldout_std_pct']:.1f} Punkte)."
            )
        cross = lot_feat["crossplay_penalty_pct"]
        if cross >= C.CROSSPLAY_PENALTY_WARNING_PCT:
            st.warning(
                f"⚠️ **Cross-Play**: setzt man Agent 1 aus einem Trainingslauf und die übrigen Agenten aus einem anderen ein, "
                f"wird es im Mittel {cross:.1f} % (von Contract Net) schlechter als das gemeinsam trainierte Team. "
                f"**Der zentrale Kritiker behebt die Konventionen nicht** - er existiert im Einsatz nicht mehr, und die "
                f"Akteure sind weiterhin nur aufeinander eingespielt."
            )
        else:
            st.info(f"Cross-Play: {cross:+.1f} % (von Contract Net) gegenüber dem gemeinsam trainierten Team - hier greifen die Konventionen kaum ineinander (Differenzen unter etwa 2 Punkten sind Rauschen).")
        st.plotly_chart(
            build_crossplay_heatmap(lot[method]["crossplay_matrix"], lot["seeds"], cmp["heldout_cnp_mean"]),
            width="stretch", key="crossplay_heatmap",
        )
        st.caption(
            f"Cross-Play für {featured_name}: Zeilen = Trainingslauf von Agent 1, Spalten = Trainingslauf der übrigen "
            "Agenten; Diagonale = gemeinsam trainiertes Team; Farbe = Held-out-Makespan relativ zu Contract Net."
        )

    st.markdown("**Exakt durchgerechnet: die 2x2-Miniatur**")
    st.markdown(
        "2 Aufträge, 2 Agenten (Optimum 15 min, Contract Net 20 min - siehe marl-demo). Hier zeigt sich kein "
        "Kritiker-Effekt: alle Verfahren finden das Optimum, wenn sie genug Episoden bekommen."
    )
    if not st.session_state.get("miniature_started"):
        st.button(
            f"▶️ Miniatur trainieren ({C.MINIATURE_SEEDS} Seeds x 5 Verfahren, wenige Sekunden)",
            on_click=_start_miniature, key="miniature_start",
        )
    else:
        with st.spinner("Trainiere die Miniatur..."):
            mini = _compute_miniature()
        mcols = st.columns(len(mini["optimum_hits"]))
        for col, (name, hits) in zip(mcols, mini["optimum_hits"].items()):
            col.metric(name, f"{hits} von {mini['n_seeds']}", help=f"Seeds, die nach {_fmt_int(mini['episodes'])} Episoden das Optimum erreichen.")

with tab_critic:
    lab = evaluation["lab"]
    st.markdown("**Was ändert der zentrale Kritiker? Erklärte Varianz auf identischen Trajektorien**")
    st.plotly_chart(build_explained_variance_chart(lab), width="stretch", key="explained_variance")
    st.markdown(
        f"Auf denselben {_fmt_int(C.CRITIC_LAB_TRAIN_EPISODES)} Trainings- und {_fmt_int(C.CRITIC_LAB_TEST_EPISODES)} "
        "Test-Episoden der MAPPO-Policy erklärt ein **lokaler** Kritiker nur "
        f"**{lab['local'] * 100:.0f} %** der Varianz des Returns, ein **zentraler** "
        f"**{lab['joint'] * 100:.0f} %**, ein zentraler mit **Auftragsliste** (privilegiert) "
        f"**{lab['full'] * 100:.0f} %**. Der zentrale Kritiker senkt also die Varianz der Vorteilsschätzung deutlich - "
        "das ist sein tatsächlicher Effekt."
    )
    st.warning(
        "⚠️ **Aber weniger Varianz ist nicht gleich bessere Policy.** Der Vorsprung des privilegierten Kritikers ist "
        "überwiegend Wissen über die *Schwierigkeit der Instanz* (welche Aufträge kommen noch?) - Wissen, das der Akteur "
        "im Einsatz nie hat und deshalb nicht in bessere Entscheidungen umsetzen kann. Ob der bessere Kritiker das "
        "Ergebnis verbessert, zeigt die Ablation oben."
    )
    e1, e2 = st.columns(2)
    e1.metric(
        "Kritiker im Training: IPPO (lokal)", f"{ppo[C.METHOD_IPPO].critic_ev * 100:.0f} %",
        help="Erklärte Varianz auf den Trainingsdaten im letzten Zehntel des Trainings (mit der jeweils aktuellen Policy).",
    )
    e2.metric(
        "Kritiker im Training: MAPPO (zentral)", f"{ppo[C.METHOD_MAPPO].critic_ev * 100:.0f} %",
        help="Erklärte Varianz auf den Trainingsdaten im letzten Zehntel des Trainings (mit der jeweils aktuellen Policy).",
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Actor-Critic.** Der Akteur $\pi_\theta(u \mid o)$ wählt die Aktion $u \in \{0, +\Delta, -\Delta\}$ ($\Delta = 25$ min)
aus der lokalen Beobachtung $o$; der Kritiker $V_\phi$ schätzt den erwarteten Return $R = -\text{Makespan}/10$.
IPPO: $V_\phi(o_a)$ je Agent. MAPPO: $V_\phi(s)$ mit dem globalen Zustand $s$ (alle Agenten + aktueller Auftrag;
optional die künftigen Aufträge).

**Vorteil (GAE, $\gamma = 1$, Belohnung nur am Ende).**
$\delta_t = V_{t+1} - V_t$ (letzter Auftrag: $R - V_t$), $\hat A_t = \delta_t + \lambda\,\hat A_{t+1}$, $\lambda = 0.9$.
Kritiker-Ziel: $\hat A_t + V_t$. Vorteile werden pro Batch normiert.

**Geclipptes PPO-Ziel.** Mit $r = \pi_\theta(u\mid o) / \pi_{\theta_\text{alt}}(u\mid o)$:

$$
L = -\mathbb{E}\Big[\min\big(r\,\hat A,\ \text{clip}(r,\ 1-\varepsilon,\ 1+\varepsilon)\,\hat A\big)\Big] - c_H\,\mathbb{E}[H(\pi)]
$$

Im ungeclippten Bereich ist der Gradient nach den Logits $-\hat A\, r\, (\text{onehot}(u) - \pi)/N$, im geclippten
Bereich 0. ($\varepsilon = 0.2$, $c_H = 0.01$; "ohne Clipping" setzt $\varepsilon \to \infty$.)

**Kritiker-Vermessung.** Erklärte Varianz $1 - \mathbb{E}[(R - V)^2] / \text{Var}(R)$ auf Testtrajektorien, für lokalen,
zentralen und privilegierten Kritiker auf *denselben* Daten und mit gleicher Netzgröße.

**Obergrenze für jede Policy.** Alle Policies halten die Ankündigungsreihenfolge fest und wählen nur, welcher Agent
welchen Auftrag bekommt, daher
$\text{Makespan(Policy)} \ge \text{Bestes bei fester Reihenfolge} \ge \text{CP-SAT-Optimum}$.

Implementiert in `mappo_env.py` (vektorisierte Umgebung, Merkmale), `mappo_nets.py` (MLP, Adam), `mappo_ppo.py`
(GAE, PPO, IPPO/MAPPO), `mappo_evaluation.py` (alle gezeigten Kennzahlen) und den unveränderten `marl_*`-Modulen
(IQL-Referenz, Held-out, exakte Referenzen).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
