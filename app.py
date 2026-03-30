import traceback

import numpy as np
import streamlit as st
import sympy as sp

from core.analysis import analyze_fixed_points, find_fixed_points_numeric
from core.parser import parse_system
from core.plotting import (
    add_fixed_points,
    add_nullclines_from_contours,
    add_trajectory,
    apply_axis_limits,
    create_phase_figure,
    create_streamline_figure,
    create_streamline_figure_custom,
)
from core.system import (
    compute_scalar_fields,
    compute_vector_field,
    create_mesh,
    integrate_trajectory,
    normalize_vector_field,
)
from ui.controls import get_parameter_value
from ui.styles import apply_app_styles
from utils.helpers import parse_initial_conditions, substitute_parameters, wrap_function

st.set_page_config(page_title="Nonlinear Dynamics App", layout="centered")
apply_app_styles()

st.title("Nonlinear Dynamics App")
st.caption(
    "Enter a system and click PLOT. Open Advanced settings only if you want finer control."
)

# ------------------------------------------------------------
# MAIN INPUTS
# ------------------------------------------------------------
top_left, top_right = st.columns([1, 1], gap="large")

with top_left:
    st.subheader("System")
    f_str = st.text_input("dx/dt =", "y")
    g_str = st.text_input("dy/dt =", "-x")

    parse_ok = False
    parse_error = None
    f_expr = g_expr = f_num = g_num = None
    params = []

    try:
        f_expr, g_expr, f_num, g_num, params = parse_system(f_str, g_str)
        parse_ok = True
    except Exception as e:
        parse_error = str(e)

    param_values = {}
    if parse_ok and params:
        st.subheader("Parameters")
        for p in params:
            name = str(p)
            param_values[name] = get_parameter_value(name, default="1.0")

with top_right:
    st.subheader("Initial conditions")
    ics_text = st.text_area(
        "Enter one initial condition per line as: x0, y0",
        value="1, 0",
        height=165,
    )

if parse_error:
    st.error(parse_error)

# ------------------------------------------------------------
# DEFAULT VALUES FOR ADVANCED SETTINGS
# ------------------------------------------------------------
xmin = -3.0
xmax = 3.0
ymin = -3.0
ymax = 3.0
n = 40
stride = 2

traj_t_max = 20.0
n_points = 1000
show_forward = True
show_backward = False

field_style = "Arrows"
normalize_vectors = True

n_seeds = 50
streamline_t_max = 10
streamline_density = 2.0
streamline_arrow_scale = 0.09

show_fixed_points = True
show_fixed_point_analysis = True
fp_grid_density = 9

show_x_nullcline = False
show_y_nullcline = False
nullcline_n = 150

# ------------------------------------------------------------
# ADVANCED SETTINGS
# ------------------------------------------------------------
with st.expander("Advanced settings", expanded=False):
    adv_left, adv_right = st.columns([1, 1], gap="large")

    with adv_left:
        st.subheader("Domain and grid")
        xmin = st.number_input("x_min", value=xmin)
        xmax = st.number_input("x_max", value=xmax)
        ymin = st.number_input("y_min", value=ymin)
        ymax = st.number_input("y_max", value=ymax)
        n = st.slider("Grid density", 10, 100, n)
        stride = st.slider("Vector field density (stride)", 1, 5, stride)

        st.subheader("Field style")
        field_style = st.radio(
            "Vector field style",
            ["Arrows", "Streamlines (classic)", "Streamlines (robust)"],
            index=0,
            horizontal=False,
        )
        normalize_vectors = st.checkbox(
            "Normalize vector field",
            value=normalize_vectors,
        )

        if field_style == "Streamlines (classic)":
            streamline_density = st.slider(
                "Streamline density",
                0.5,
                3.0,
                streamline_density,
                0.1,
            )
            streamline_arrow_scale = st.slider(
                "Streamline arrow scale",
                0.02,
                0.2,
                streamline_arrow_scale,
                0.01,
            )

        elif field_style == "Streamlines (robust)":
            n_seeds = st.slider("Number of streamlines", 10, 200, n_seeds)
            streamline_t_max = st.slider(
                "Streamline integration time",
                1,
                50,
                streamline_t_max,
            )

        if (
            field_style in ["Streamlines (classic)", "Streamlines (robust)"]
            and normalize_vectors
        ):
            st.warning(
                "Normalized vector fields may behave poorly near zeros of the field."
            )

    with adv_right:
        st.subheader("Trajectories")
        traj_t_max = st.number_input(
            "Trajectory t_max",
            value=traj_t_max,
            min_value=0.1,
        )
        n_points = st.slider("Integration points", 100, 5000, n_points, step=100)
        show_forward = st.checkbox("Show forward trajectories", value=show_forward)
        show_backward = st.checkbox("Show backward trajectories", value=show_backward)

        st.subheader("Fixed points")
        show_fixed_points = st.checkbox("Show fixed points", value=show_fixed_points)
        show_fixed_point_analysis = st.checkbox(
            "Show fixed-point analysis",
            value=show_fixed_point_analysis,
        )
        fp_grid_density = st.slider(
            "Fixed-point search density",
            5,
            25,
            fp_grid_density,
            step=2,
        )

        st.subheader("Nullclines")
        show_x_nullcline = st.checkbox(
            "Show x-nullcline (dx/dt = 0)",
            value=show_x_nullcline,
        )
        show_y_nullcline = st.checkbox(
            "Show y-nullcline (dy/dt = 0)",
            value=show_y_nullcline,
        )
        nullcline_n = st.slider(
            "Nullcline density",
            50,
            300,
            nullcline_n,
            step=10,
        )

st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
plot_clicked = st.button(label="PLOT", type="primary")

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
if plot_clicked:
    try:
        if not parse_ok:
            raise ValueError(parse_error or "System could not be parsed.")

        initial_conditions = parse_initial_conditions(ics_text)

        f_wrapped = wrap_function(f_num, params, param_values)
        g_wrapped = wrap_function(g_num, params, param_values)

        def rhs(t, z):
            x, y = z
            return [f_wrapped(x, y), g_wrapped(x, y)]

        f_expr_eval = substitute_parameters(f_expr, params, param_values)
        g_expr_eval = substitute_parameters(g_expr, params, param_values)

        X, Y = create_mesh(xmin, xmax, ymin, ymax, n, n)
        U_raw, V_raw = compute_vector_field(f_wrapped, g_wrapped, X, Y)

        if normalize_vectors:
            U_plot, V_plot = normalize_vector_field(U_raw, V_raw)
        else:
            U_plot, V_plot = U_raw, V_raw

        Xn, Yn = create_mesh(xmin, xmax, ymin, ymax, nullcline_n, nullcline_n)
        Fn, Gn = compute_scalar_fields(f_wrapped, g_wrapped, Xn, Yn)

        if not np.isfinite(U_raw).any() or not np.isfinite(V_raw).any():
            st.warning(
                "Vector field could not be evaluated on the chosen domain. "
                "The system may be outside its domain there, e.g. log(x) for x <= 0 or sqrt(x) for x < 0."
            )

        if not np.isfinite(Fn).any() or not np.isfinite(Gn).any():
            st.warning(
                "Some scalar-field values are undefined on this domain, so nullclines may be incomplete."
            )

        if field_style == "Arrows":
            fig = create_phase_figure(X, Y, U_plot, V_plot, stride=stride)

        elif field_style == "Streamlines (classic)":
            fig = create_streamline_figure(
                X,
                Y,
                U_plot,
                V_plot,
                density=streamline_density,
                arrow_scale=streamline_arrow_scale,
            )

        elif field_style == "Streamlines (robust)":
            fig = create_streamline_figure_custom(
                rhs,
                X,
                Y,
                U_raw,
                V_raw,
                n_seeds=n_seeds,
                t_max=streamline_t_max,
                max_step=0.05,
            )

        else:
            raise ValueError(f"Unknown field style: {field_style}")

        fixed_points = []
        if show_fixed_points:
            fixed_points = find_fixed_points_numeric(
                f_expr_eval,
                g_expr_eval,
                xmin,
                xmax,
                ymin,
                ymax,
                nx_guess=fp_grid_density,
                ny_guess=fp_grid_density,
            )
            fig = add_fixed_points(fig, fixed_points)

        jacobian_expr = None
        fixed_point_analysis = []

        if show_fixed_points and show_fixed_point_analysis and fixed_points:
            jacobian_expr, fixed_point_analysis = analyze_fixed_points(
                f_expr_eval,
                g_expr_eval,
                fixed_points,
            )

        fig = add_nullclines_from_contours(
            fig,
            Xn,
            Yn,
            Fn,
            Gn,
            show_x_nullcline=show_x_nullcline,
            show_y_nullcline=show_y_nullcline,
        )

        all_x = [X.flatten()]
        all_y = [Y.flatten()]

        for i, (x0, y0) in enumerate(initial_conditions, start=1):
            if show_forward:
                x_traj, y_traj = integrate_trajectory(
                    f_wrapped,
                    g_wrapped,
                    x0,
                    y0,
                    t_span=(0, traj_t_max),
                    n_points=n_points,
                )
                fig = add_trajectory(fig, x_traj, y_traj, name=f"Forward {i}")
                all_x.append(np.asarray(x_traj))
                all_y.append(np.asarray(y_traj))

            if show_backward:
                x_traj_b, y_traj_b = integrate_trajectory(
                    f_wrapped,
                    g_wrapped,
                    x0,
                    y0,
                    t_span=(0, -traj_t_max),
                    n_points=n_points,
                )
                fig = add_trajectory(fig, x_traj_b, y_traj_b, name=f"Backward {i}")
                all_x.append(np.asarray(x_traj_b))
                all_y.append(np.asarray(y_traj_b))

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        fig = apply_axis_limits(fig, X.flatten(), Y.flatten(), padding_ratio=0.03)

        st.markdown("---")
        st.plotly_chart(fig, width="stretch")

        if show_fixed_points:
            st.subheader("Fixed points")

            if fixed_points:
                for i, (xp, yp) in enumerate(fixed_points, start=1):
                    fx_val = f_wrapped(xp, yp)
                    gy_val = g_wrapped(xp, yp)
                    st.write(
                        f"{i}. ({xp:.10g}, {yp:.10g}) | "
                        f"f={fx_val:.3e}, g={gy_val:.3e}"
                    )
            else:
                st.write("No fixed points found in the selected domain.")

            if show_fixed_points and show_fixed_point_analysis and fixed_point_analysis:
                st.subheader("Jacobian and classification")

                st.write("Symbolic Jacobian:")
                st.latex("J(x,y) = " + sp.latex(jacobian_expr))

                for i, result in enumerate(fixed_point_analysis, start=1):
                    px, py = result["point"]
                    J = result["jacobian"]
                    eigs = result["eigenvalues"]
                    classification = result["classification"]

                    st.markdown(f"**Point {i}: ({px:.6g}, {py:.6g})**")
                    st.write("Jacobian:")
                    st.code(np.array2string(J, precision=6, suppress_small=True))

                    eig1, eig2 = eigs[0], eigs[1]
                    st.write(
                        "Eigenvalues:",
                        f"{eig1.real:.6g}{eig1.imag:+.6g}j, "
                        f"{eig2.real:.6g}{eig2.imag:+.6g}j",
                    )
                    st.write("Classification:", classification)

        info_left, info_right = st.columns([1, 1], gap="large")

        with info_left:
            st.subheader("Parsed system")
            st.latex(r"\dot{x} = " + sp.latex(f_expr))
            st.latex(r"\dot{y} = " + sp.latex(g_expr))

            if params:
                st.subheader("Parameter values used")
                for p in params:
                    name = str(p)
                    st.write(f"{name} = {param_values[name]:.10g}")

        with info_right:
            st.subheader("Evaluated system")
            st.latex(r"\dot{x} = " + sp.latex(f_expr_eval))
            st.latex(r"\dot{y} = " + sp.latex(g_expr_eval))

    except Exception as e:
        st.error(str(e))
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
