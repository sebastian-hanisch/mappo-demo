"""Visualisierungen der MARL-Demo: Lernkurve, Gebots-Schritt-Ansicht, Cross-Play-Heatmap,
Seed-Lotterie, Held-out-Histogramm. Der Gantt-Chart (`build_schedule_figure`) kommt unverändert
aus `cn_visualization`."""

import plotly.graph_objects as go

import cn_constants as C
from cn_visualization import AGENT_COLORS, lock_axes

CNP_COLOR = "#7F7F7F"
IQL_COLOR = "#0072B2"
OPT_COLOR = "#009E73"
WARN_COLOR = "#D55E00"


def build_learning_curve(points, cnp_nominal, ortools_nominal, cnp_heldout_mean, marker_episodes=None):
    """points: [(episoden, nominal, heldout_mittel), ...]. Referenzlinien: Contract Net (nominal und
    Held-out) und das CP-SAT-Optimum. marker_episodes: senkrechte Linie für den gerade gezeigten Checkpoint."""
    episodes = [p[0] for p in points]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=[p[1] for p in points], mode="lines+markers", name="Gelernt (nominal)",
        line=dict(color=IQL_COLOR, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=[p[2] for p in points], mode="lines+markers", name="Gelernt (Held-out-Mittel)",
        line=dict(color=IQL_COLOR, width=2, dash="dot"),
    ))
    x_range = [min(episodes), max(episodes)]
    fig.add_trace(go.Scatter(
        x=x_range, y=[cnp_nominal] * 2, mode="lines", name="Contract Net (nominal)",
        line=dict(color=CNP_COLOR, width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=[cnp_heldout_mean] * 2, mode="lines", name="Contract Net (Held-out-Mittel)",
        line=dict(color=CNP_COLOR, width=1.5, dash="dot"),
    ))
    if ortools_nominal is not None:
        fig.add_trace(go.Scatter(
            x=x_range, y=[ortools_nominal] * 2, mode="lines", name="Zentrales Optimum (CP-SAT, nominal)",
            line=dict(color=OPT_COLOR, width=2, dash="dash"),
        ))
    if marker_episodes is not None:
        fig.add_vline(x=marker_episodes, line_width=1, line_color="#333333")
    fig.update_xaxes(type="log", title="Trainings-Episoden")
    fig.update_yaxes(title="Makespan (min)")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_policy_step_chart(dispatch, step, action_names=C.ACTION_NAMES):
    """Pro Agent: Originalgebot vs. modifiziertes Gebot in diesem Schritt; Gewinner grün umrandet."""
    step_data = dispatch.protocol_result.steps[step]
    original = dispatch.original_bids[step]
    actions = dispatch.actions[step]
    labels = [f"Agent {b.agent_id + 1}<br>{action_names[a]}" for b, a in zip(step_data.bids, actions)]
    modified = [b.finish_time for b in step_data.bids]
    winner = step_data.winner_agent_id

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=list(original), name="Eigenes Gebot (Contract Net)", marker_color="#B0B0B0",
        text=[f"{v:.1f}" for v in original], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=modified, name="Abgegebenes Gebot (nach Aktion)",
        marker_color=[AGENT_COLORS[i % len(AGENT_COLORS)] for i in range(len(labels))],
        marker_line_color=["#2CA02C" if i == winner else "rgba(0,0,0,0)" for i in range(len(labels))],
        marker_line_width=[4 if i == winner else 0 for i in range(len(labels))],
        text=[f"{v:.1f}" for v in modified], textposition="outside",
    ))
    fig.update_layout(
        barmode="group", yaxis_title="Fertigstellungszeit als Gebot (min)", height=320,
        margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25),
    )
    return lock_axes(fig)


def build_crossplay_heatmap(matrix, seeds, cnp_heldout_mean):
    """Zeilen: Trainingslauf von Agent 0, Spalten: Trainingslauf der übrigen Agenten. Farbe/Text = Held-out-
    Makespan relativ zu Contract Net (%, negativ = besser). Diagonale = gemeinsam trainiertes Team."""
    pct = [[(v - cnp_heldout_mean) / cnp_heldout_mean * 100.0 for v in row] for row in matrix]
    labels = [f"Seed {s}" for s in seeds]
    fig = go.Figure(go.Heatmap(
        z=pct, x=labels, y=labels, colorscale="RdBu_r", zmid=0,
        text=[[f"{v:+.1f} %" for v in row] for row in pct], texttemplate="%{text}",
        colorbar=dict(title="vs. CNP (%)"),
    ))
    fig.update_xaxes(title="Trainingslauf der Agenten 2..n")
    fig.update_yaxes(title="Trainingslauf von Agent 1", autorange="reversed")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


def build_lottery_chart(lottery):
    """Ein Punkt pro Trainings-Seed: nominal und Held-out, jeweils in % gegenüber Contract Net."""
    labels = [f"Seed {s}" for s in lottery["seeds"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=lottery["nominal_pct"], mode="markers", name="nominal",
        marker=dict(size=12, color=IQL_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=lottery["heldout_pct"], mode="markers", name="Held-out-Mittel",
        marker=dict(size=12, color=WARN_COLOR, symbol="diamond"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=CNP_COLOR, annotation_text="Contract Net", annotation_position="top left")
    fig.update_yaxes(title="Makespan vs. Contract Net (%)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_ratio_histogram(ratios_pct):
    """Verteilung der Held-out-Ergebnisse: pro Instanz (IQL - CNP)/CNP in %."""
    fig = go.Figure(go.Histogram(
        x=ratios_pct, nbinsx=15, marker_color=IQL_COLOR, name="Held-out-Instanzen",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=CNP_COLOR)
    fig.update_xaxes(title="Makespan vs. Contract Net je Instanz (%; links besser, rechts schlechter)")
    fig.update_yaxes(title="Anzahl Instanzen")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return lock_axes(fig)
