import numpy as np
import plotly.graph_objects as go
from plotly.figure_factory import create_quiver


def create_phase_figure(X, Y, U, V):
    fig = create_quiver(
        X.flatten(),
        Y.flatten(),
        U.flatten(),
        V.flatten(),
        scale=0.15,
        arrow_scale=0.3,
        name="Vector field",
    )

    x_range, y_range = compute_axis_limits(X, Y, padding_ratio=0.05)

    fig.update_layout(
        title="Phase portrait",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        showlegend=False,
    )

    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range, scaleanchor="x", scaleratio=1)

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
            name=f"{name} start",
            marker=dict(size=8),
            showlegend=False,
        )
    )

    return fig


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


def add_nullclines(fig, X, Y, F, G, show_x_nullcline=True, show_y_nullcline=True):
    if show_x_nullcline:
        fig.add_trace(
            go.Contour(
                x=X[0, :],
                y=Y[:, 0],
                z=F,
                contours=dict(
                    start=0,
                    end=0,
                    size=1,
                    coloring="none",
                ),
                line=dict(width=2),
                name="x-nullcline (dx/dt = 0)",
                showscale=False,
            )
        )

    if show_y_nullcline:
        fig.add_trace(
            go.Contour(
                x=X[0, :],
                y=Y[:, 0],
                z=G,
                contours=dict(
                    start=0,
                    end=0,
                    size=1,
                    coloring="none",
                ),
                line=dict(width=2, dash="dash"),
                name="y-nullcline (dy/dt = 0)",
                showscale=False,
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
