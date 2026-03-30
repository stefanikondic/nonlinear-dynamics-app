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


def _pick_arrow_indices(n_points, n_arrows=1, margin=0.15):
    if n_points < 5 or n_arrows <= 0:
        return []

    start = int(np.floor(margin * (n_points - 1)))
    end = int(np.ceil((1.0 - margin) * (n_points - 1)))

    if end <= start:
        return []

    if n_arrows == 1:
        return [(start + end) // 2]

    return np.linspace(start, end, n_arrows).astype(int).tolist()


def _add_quiver_arrow_on_streamline(
    fig,
    x,
    y,
    n_arrows=1,
    arrow_scale=0.25,
    arrow_scale_ratio=0.35,
):
    """
    Dodaje male geometrijske strelice duž strujnice koristeći create_quiver.
    Time strelice ostaju u data koordinatama i lijepo prate zoom.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3 or len(y) < 3 or n_arrows <= 0:
        return fig

    arrow_indices = _pick_arrow_indices(len(x), n_arrows=n_arrows)
    if not arrow_indices:
        return fig

    xq = []
    yq = []
    uq = []
    vq = []

    for i in arrow_indices:
        if i <= 0 or i >= len(x) - 1:
            continue

        dx = x[i + 1] - x[i - 1]
        dy = y[i + 1] - y[i - 1]

        if not np.isfinite(dx) or not np.isfinite(dy):
            continue

        norm = np.hypot(dx, dy)
        if norm < 1e-14:
            continue

        # Mala strelica duž tangente
        dx_unit = dx / norm
        dy_unit = dy / norm

        xq.append(x[i])
        yq.append(y[i])
        uq.append(arrow_scale * dx_unit)
        vq.append(arrow_scale * dy_unit)

    if not xq:
        return fig

    arrow_fig = create_quiver(
        xq,
        yq,
        uq,
        vq,
        scale=1.0,
        arrow_scale=arrow_scale_ratio,
        name="",
    )

    for trace in arrow_fig.data:
        trace.showlegend = False
        trace.hoverinfo = "skip"
        fig.add_trace(trace)

    return fig


def create_streamline_figure_custom(
    rhs,
    X,
    Y,
    U,
    V,
    n_seeds=30,
    t_max=10,
    max_step=0.05,
    arrows_per_streamline=1,
    arrow_scale=0.25,
    arrow_scale_ratio=0.35,
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

            fig = _add_quiver_arrow_on_streamline(
                fig,
                x,
                y,
                n_arrows=arrows_per_streamline,
                arrow_scale=arrow_scale,
                arrow_scale_ratio=arrow_scale_ratio,
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
