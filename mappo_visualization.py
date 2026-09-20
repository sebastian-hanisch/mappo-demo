"""Visualisierungen der MAPPO-Demo. Cross-Play-Heatmap, Held-out-Histogramm und die Gebots-Schritt-Ansicht
kommen unverändert aus `marl_visualization`; der Gantt-Chart aus `cn_visualization`. Neu: Lernkurven-Vergleich
(IQL/IPPO/MAPPO), Aktionswahrscheinlichkeiten, Policy-Heatmap, Kritiker-Vorhersage, Erklärte-Varianz-Balken und
die Drei-Verfahren-Lotterie."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cn_constants as C
from cn_visualization import AGENT_COLORS
from mappo_ppo import action_probs

CNP_COLOR = "#7F7F7F"
OPT_COLOR = "#009E73"
METHOD_COLORS = {"iql": "#7F7F7F", C.METHOD_IPPO: "#E69F00", C.METHOD_MAPPO: "#0072B2"}
METHOD_NAMES = {"iql": "IQL", C.METHOD_IPPO: "IPPO", C.METHOD_MAPPO: "MAPPO"}
ACTION_COLORS = ["#B0B0B0", "#D55E00", "#009E73"]  # normal, hoch, niedrig


def build_learning_curves(curves, cnp_heldout_mean, ortools_heldout_mean, marker_episodes=None):
    """curves: name -> [(episoden, nominal, heldout_mittel), ...]. Gezeigt wird das Held-out-Mittel (ehrliches
    Maß) je Verfahren; Referenzlinien: Contract Net und CP-SAT (jeweils Held-out-Mittel)."""
    fig = go.Figure()
    all_episodes = []
    for name, points in curves.items():
        episodes = [p[0] for p in points]
        all_episodes += episodes
        fig.add_trace(go.Scatter(
            x=episodes, y=[p[2] for p in points], mode="lines+markers", name=METHOD_NAMES.get(name, name),
            line=dict(color=METHOD_COLORS.get(name), width=3),
        ))
    x_range = [min(all_episodes), max(all_episodes)]
    fig.add_trace(go.Scatter(
        x=x_range, y=[cnp_heldout_mean] * 2, mode="lines", name="Contract Net",
        line=dict(color=CNP_COLOR, width=2, dash="dash"),
    ))
    if ortools_heldout_mean is not None:
        fig.add_trace(go.Scatter(
            x=x_range, y=[ortools_heldout_mean] * 2, mode="lines", name="Zentrales Optimum (CP-SAT)",
            line=dict(color=OPT_COLOR, width=2, dash="dash"),
        ))
    if marker_episodes is not None:
        fig.add_vline(x=marker_episodes, line_width=1, line_color="#333333")
    ticks = sorted(set(all_episodes))
    fig.update_xaxes(type="log", title="Trainings-Episoden", tickvals=ticks, ticktext=[f"{t:,}".replace(",", ".") for t in ticks])
    fig.update_yaxes(title="Makespan, Held-out-Mittel (min)")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.35))
    return fig


def build_action_probability_chart(trace, step):
    """Aktionswahrscheinlichkeiten aller Agenten bei einem Auftrag - jeder aus seiner LOKALEN Beobachtung."""
    probs = trace["probs"][step]
    k = probs.shape[0]
    labels = [f"Agent {a + 1}" for a in range(k)]
    fig = go.Figure()
    for action, name in enumerate(C.ACTION_NAMES):
        fig.add_trace(go.Bar(
            x=labels, y=probs[:, action], name=name, marker_color=ACTION_COLORS[action],
            text=[f"{p:.2f}" for p in probs[:, action]], textposition="inside",
        ))
    chosen = trace["actions"][step]
    for a in range(k):
        fig.add_annotation(
            x=labels[a], y=1.02, text=f"→ {C.ACTION_NAMES[chosen[a]]}", showarrow=False, font=dict(size=10), yanchor="bottom",
        )
    fig.update_layout(
        barmode="stack", yaxis=dict(title="Wahrscheinlichkeit", range=[0, 1.15]), height=300,
        margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25),
    )
    return fig


def policy_heatmap_data(geom, result, agent=0, job_index=None, resolution=25):
    """Aktionswahrscheinlichkeiten eines Agenten über (eigene Freizeit x Anfahrt) bei mittlerer Position/Dauer:
    [3, ny, nx] plus die Achsenwerte."""
    j = geom.n // 2 if job_index is None else job_index
    free_axis = np.linspace(0, 2.0 * geom.fair, resolution)
    travel_axis = np.linspace(0, 20 * geom.tr, resolution)
    grid_free, grid_travel = np.meshgrid(free_axis, travel_axis)
    points = grid_free.size
    free = np.tile(grid_free.reshape(-1, 1), (1, geom.k))
    travel = np.tile(grid_travel.reshape(-1, 1), (1, geom.k))
    agent_pos = np.full((points, geom.k), 10.0)
    probs = action_probs(geom, result, j, free, agent_pos, travel, np.full(points, 10.0), np.full(points, C.MEAN_DURATION))
    out = probs[:, agent, :].reshape(resolution, resolution, 3).transpose(2, 0, 1)
    return out, free_axis, travel_axis, j


def build_policy_heatmap(geom, result, agent=0, job_index=None):
    """Zwei Heatmaps: wie wahrscheinlich 'hoch bieten' (ablehnen) bzw. 'niedrig bieten' (greifen) je nach
    eigener Freizeit und Anfahrt."""
    probs, free_axis, travel_axis, j = policy_heatmap_data(geom, result, agent, job_index)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("P(hoch bieten = ablehnen)", "P(niedrig bieten = greifen)"))
    for col, action in ((1, 1), (2, 2)):
        fig.add_trace(go.Heatmap(
            z=probs[action], x=free_axis, y=travel_axis, zmin=0, zmax=1, colorscale="Blues",
            colorbar=dict(title="P", len=0.9) if col == 2 else None, showscale=col == 2,
        ), row=1, col=col)
        fig.update_xaxes(title_text="eigene Freizeit (min)", row=1, col=col)
    fig.update_yaxes(title_text="Anfahrt (min)", row=1, col=1)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig, j


def build_critic_prediction_chart(trace, method):
    """Vorhergesagter Makespan des Kritikers vor jedem Auftrag vs. der tatsächliche (gierige) Makespan.
    Zentral: eine Linie; lokal (IPPO): je Agent eine."""
    steps = list(range(1, len(trace["actions"]) + 1))
    fig = go.Figure()
    predicted = trace["critic_makespan"]
    if predicted.ndim == 1:
        fig.add_trace(go.Scatter(
            x=steps, y=predicted, mode="lines+markers", name="Zentraler Kritiker (sieht alle Agenten)",
            line=dict(color=METHOD_COLORS[C.METHOD_MAPPO], width=3),
        ))
    else:
        for a in range(predicted.shape[1]):
            fig.add_trace(go.Scatter(
                x=steps, y=predicted[:, a], mode="lines+markers", name=f"Lokaler Kritiker Agent {a + 1}",
                line=dict(color=AGENT_COLORS[a % len(AGENT_COLORS)], width=2),
            ))
    fig.add_trace(go.Scatter(
        x=[steps[0], steps[-1]], y=[trace["makespan"]] * 2, mode="lines", name="Tatsächlicher Makespan (gierig)",
        line=dict(color=CNP_COLOR, width=2, dash="dash"),
    ))
    fig.update_xaxes(title="vor Auftrag", dtick=1)
    fig.update_yaxes(title="vorhergesagter Makespan (min)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.3))
    return fig


def build_explained_variance_chart(lab):
    """Erklärte Varianz des Returns auf identischen Trajektorien für drei Kritiker."""
    labels = ["Lokal (je Agent)", "Zentral (alle Agenten)", "Zentral + Auftragsliste (privilegiert)"]
    values = [lab["local"], lab["joint"], lab["full"]]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=["#E69F00", "#0072B2", "#56B4E9"],
        text=[f"{v:.2f}" for v in values], textposition="outside",
    ))
    fig.update_yaxes(title="erklärte Varianz des Returns", range=[0, 1.05])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig


def build_lottery_comparison(lottery):
    """Held-out-Ergebnis (% vs. Contract Net) je Trainings-Seed, nebeneinander für IQL, IPPO und MAPPO."""
    fig = go.Figure()
    for name in ("iql", C.METHOD_IPPO, C.METHOD_MAPPO):
        pct = lottery[name]["heldout_pct"]
        fig.add_trace(go.Scatter(
            x=[METHOD_NAMES[name]] * len(pct), y=pct, mode="markers", name=METHOD_NAMES[name],
            marker=dict(size=12, color=METHOD_COLORS[name], line=dict(width=1, color="#333333")),
            text=[f"Seed {s}" for s in lottery["seeds"]], hovertemplate="%{text}: %{y:+.1f} %<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=CNP_COLOR, annotation_text="Contract Net", annotation_position="top left")
    fig.update_yaxes(title="Held-out-Makespan vs. Contract Net (%)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig
