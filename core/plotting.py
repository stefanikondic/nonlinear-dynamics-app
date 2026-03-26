import numpy as np
import plotly.graph_objects as go
from plotly.figure_factory import create_quiver
import matplotlib.pyplot as plt


def compute_axis_limits(x_values, y_values, padding_ratio=0.08):
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    x_span = x_max - x_min
    y_span = y_max - y_min

    if x_span == 0:
        x_span = 1.0
    if y_span == 0:
        y_span = 1.0

    x_pad = padding_ratio * x_span
    y_pad = padding_ratio * y_span

    return (
        [x_min - x_pad, x_max + x_pad],
        [y_min - y_pad, y_max + y_pad],
    )


def apply_axis_limits(fig, x_values, y_values, padding_ratio=0.08):
    x_range, y_range = compute_axis_limits(x_values, y_values, padding_ratio)
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range, scaleanchor="x", scaleratio=1)
    return fig


def create_phase_figure(X, Y, U, V, stride=2):
    # ↓↓↓ OVDJE IDE STRIDE ↓↓↓
    X_plot = X[::stride, ::stride]
    Y_plot = Y[::stride, ::stride]
    U_plot = U[::stride, ::stride]
    V_plot = V[::stride, ::stride]

    fig = create_quiver(
        X_plot.flatten(),
        Y_plot.flatten(),
        U_plot.flatten(),
        V_plot.flatten(),
        scale=0.2,
        arrow_scale=0.4,
        name="Vector field",
    )

    fig.update_layout(
        title="Phase portrait",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def add_trajectory(fig, x_traj, y_traj, name="Trajectory"):
    fig.add_trace(
        go.Scatter(
            x=x_traj,
            y=y_traj,
            mode="lines",
            name=name,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x_traj[0]],
            y=[y_traj[0]],
            mode="markers",
            marker=dict(size=7),
            showlegend=False,
        )
    )

    return fig


def add_fixed_points(fig, fixed_points):
    if not fixed_points:
        return fig

    x_vals = [p[0] for p in fixed_points]
    y_vals = [p[1] for p in fixed_points]

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name="Fixed points",
            marker=dict(size=10, symbol="x"),
        )
    )

    return fig


def extract_zero_contours(X, Y, Z):
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X, Y, Z, levels=[0])

    contour_lines = []
    for level_segments in cs.allsegs:
        for seg in level_segments:
            if len(seg) >= 2:
                contour_lines.append(seg.copy())

    plt.close(fig_tmp)
    return contour_lines


def add_nullclines_from_contours(
    fig,
    X,
    Y,
    F,
    G,
    show_x_nullcline=True,
    show_y_nullcline=True,
):
    if show_x_nullcline:
        x_nullclines = extract_zero_contours(X, Y, F)
        for i, vertices in enumerate(x_nullclines):
            fig.add_trace(
                go.Scatter(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    mode="lines",
                    name="x-nullcline (dx/dt = 0)",
                    line=dict(width=2),
                    showlegend=(i == 0),
                )
            )

    if show_y_nullcline:
        y_nullclines = extract_zero_contours(X, Y, G)
        for i, vertices in enumerate(y_nullclines):
            fig.add_trace(
                go.Scatter(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    mode="lines",
                    name="y-nullcline (dy/dt = 0)",
                    line=dict(width=2, dash="dash"),
                    showlegend=(i == 0),
                )
            )

    return fig
