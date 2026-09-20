"""Touch-Scrolling-Konvention des Portfolios: jede Plotly-Figur sperrt ihre Achsen (fixedrange) über `lock_axes`."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BARE_RETURN = re.compile(r"return fig\b(?!,)")


def test_visualization_modules_never_return_an_unlocked_figure():
    for path in ROOT.glob("*_visualization.py"):
        source = path.read_text(encoding="utf-8")
        bare = len(BARE_RETURN.findall(source))
        expected = 1 if path.name == "cn_visualization.py" else 0     # 1 = das `return fig` von lock_axes selbst
        assert bare == expected, f"{path.name}: ungesperrte Figur"
        assert "return fig, " not in source
        assert "lock_axes" in source


def test_lock_axes_sets_fixedrange_on_all_axes():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from cn_visualization import lock_axes

    fig = lock_axes(make_subplots(rows=1, cols=2))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1)
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    assert axes and all(a.get("fixedrange") is True for a in axes)
