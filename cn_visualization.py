"""Plotly-Visualisierungen: ein wachsender Gantt-Chart (welcher Agent bearbeitet
welchen Auftrag wann) und ein Balkendiagramm der Gebote für den aktuellen Schritt."""

import plotly.graph_objects as go

AGENT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe/Ito, farbenblind-sicher


def lock_axes(fig):
    """fixedrange auf beiden Achsen: verhindert Pinch-Zoom/Drag-Pan im Chart, damit auf Touch-Geräten stattdessen die Seite
    normal gescrollt wird (Hover-Tooltips bleiben davon unberührt)."""
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _job_intervals(instance, result, up_to_step):
    """Rekonstruiert für jeden Agenten Start-/Endzeit jedes ihm bis `up_to_step`
    zugeteilten Auftrags, in Vergabereihenfolge (== Bearbeitungsreihenfolge, da jedes
    Gebot die eigene Warteschlange nur am Ende verlängert)."""
    intervals = {a: [] for a in range(instance.n_agents)}
    awarded_jobs = {step.job_index: step.winner_agent_id for step in result.steps if step.step <= up_to_step}

    for agent_id in range(instance.n_agents):
        position = instance.agent_start_positions[agent_id]
        free_time = 0.0
        for job_index in result.schedules.get(agent_id, ()):
            if job_index not in awarded_jobs:
                continue
            job = instance.jobs[job_index]
            travel = instance.travel_time(position, job.position)
            start = free_time + travel
            end = start + job.duration
            intervals[agent_id].append((job_index, start, end))
            free_time = end
            position = job.position
    return intervals


def build_schedule_figure(instance, result, step, ortools_makespan=None):
    intervals = _job_intervals(instance, result, step)
    fig = go.Figure()

    for agent_id in range(instance.n_agents):
        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        for job_index, start, end in intervals[agent_id]:
            fig.add_trace(
                go.Bar(
                    x=[end - start], y=[f"Agent {agent_id + 1}"], base=start, orientation="h",
                    marker_color=color, showlegend=False, hovertext=f"Auftrag {job_index + 1}",
                    hoverinfo="text",
                    text=f"A{job_index + 1}", textposition="inside",
                )
            )

    if ortools_makespan is not None:
        fig.add_vline(
            x=ortools_makespan, line_dash="dash", line_color="gray",
            annotation_text="Zentrales Optimum", annotation_position="top",
        )

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Zeit (min)",
        yaxis_title=None,
        height=120 + 60 * instance.n_agents,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[f"Agent {a + 1}" for a in range(instance.n_agents)],
        autorange="reversed",
    )
    return lock_axes(fig)


def build_bid_chart(step_data):
    """step_data: eine cn_protocol.AwardStep."""
    agent_ids = [f"Agent {b.agent_id + 1}" for b in step_data.bids]
    finish_times = [b.finish_time for b in step_data.bids]
    colors = [
        "#2CA02C" if b.agent_id == step_data.winner_agent_id else "#B0B0B0"
        for b in step_data.bids
    ]

    fig = go.Figure(
        go.Bar(x=agent_ids, y=finish_times, marker_color=colors, text=[f"{ft:.1f}" for ft in finish_times],
               textposition="outside")
    )
    fig.update_layout(
        yaxis_title="Gebotene Fertigstellungszeit (min)",
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return lock_axes(fig)
