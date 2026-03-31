import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.figure_factory import create_quiver, create_streamline
from scipy.integrate import solve_ivp


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


def create_streamline_figure(X, Y, U, V, density=1.0, arrow_scale=0.09):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    if X.shape != Y.shape or X.shape != U.shape or X.shape != V.shape:
        raise ValueError("X, Y, U, and V must all have the same shape.")

    nrows, ncols = X.shape
    if nrows < 3 or ncols < 3:
        raise ValueError("Streamlines require at least a 3x3 grid.")

    X_inner = X[1:-1, 1:-1]
    Y_inner = Y[1:-1, 1:-1]
    U_inner = U[1:-1, 1:-1]
    V_inner = V[1:-1, 1:-1]

    if not np.isfinite(U_inner).all() or not np.isfinite(V_inner).all():
        raise ValueError(
            "Streamlines cannot be drawn because the vector field contains NaN or inf values."
        )

    speed = np.sqrt(U_inner**2 + V_inner**2)
    if np.all(speed < 1e-14):
        raise ValueError(
            "Streamlines cannot be drawn because the vector field is zero on the selected domain."
        )

    x = X_inner[0, :]
    y = Y_inner[:, 0]

    try:
        fig = create_streamline(
            x=x,
            y=y,
            u=U_inner,
            v=V_inner,
            density=density,
            arrow_scale=arrow_scale,
        )
    except Exception as e:
        raise ValueError(
            "Plotly could not generate streamlines on this domain. "
            "Try slightly changing the domain or use Arrows."
        ) from e

    fig.update_layout(
        title="Phase portrait",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def compute_streamline(rhs, x0, y0, t_max=10, max_step=0.05):
    sol_f = solve_ivp(rhs, (0, t_max), [x0, y0], max_step=max_step)
    sol_b = solve_ivp(rhs, (0, -t_max), [x0, y0], max_step=max_step)

    x = np.concatenate([sol_b.y[0][::-1], sol_f.y[0]])
    y = np.concatenate([sol_b.y[1][::-1], sol_f.y[1]])

    return x, y


def generate_seed_points(X, Y, U, V, n_seeds=20, min_speed=1e-3):
    seeds = []

    xs = X.flatten()
    ys = Y.flatten()
    speeds = np.sqrt(U.flatten() ** 2 + V.flatten() ** 2)

    mask = speeds > min_speed

    xs = xs[mask]
    ys = ys[mask]

    if len(xs) == 0:
        return []

    idx = np.linspace(0, len(xs) - 1, n_seeds).astype(int)

    for i in idx:
        seeds.append((xs[i], ys[i]))

    return seeds


def create_streamline_figure_custom(
    rhs,
    X,
    Y,
    U,
    V,
    n_seeds=30,
    t_max=10,
    max_step=0.05,
):
    fig = go.Figure()

    seeds = generate_seed_points(X, Y, U, V, n_seeds=n_seeds)

    for x0, y0 in seeds:
        try:
            x, y = compute_streamline(rhs, x0, y0, t_max, max_step)

            if len(x) < 2 or len(y) < 2:
                continue

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        except Exception:
            continue

    fig.update_layout(
        title="Phase portrait (custom streamlines)",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def add_separatrices(fig, branches):
    for typ, x, y in branches:
        if len(x) < 2:
            continue

        if typ == "stable":
            dash = "dash"
            width = 3
        else:
            dash = "solid"
            width = 3

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=width, dash=dash),
                showlegend=False,
            )
        )

    return fig
