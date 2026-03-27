import streamlit as st
import numpy as np
import traceback
import sympy as sp

from core.parser import parse_system
from core.system import (
    compute_scalar_fields,
    create_mesh,
    compute_vector_field,
    integrate_trajectory,
    normalize_vector_field,
)
from core.plotting import (
    add_fixed_points,
    add_nullclines_from_contours,
    add_trajectory,
    apply_axis_limits,
    create_phase_figure,
    create_streamline_figure,
)
from core.analysis import find_fixed_points_numeric, analyze_fixed_points


st.title("Nonlinear Dynamics App")
st.caption(
    "Examples: sinx, cosx, sinhx, asinx, sqrtx, lnx, e^x, x^2, 2x, xy, pi. "
    "Any symbol other than x and y is treated as a parameter."
)

f_str = st.text_input("dx/dt =", "y")
g_str = st.text_input("dy/dt =", "-x")


def parse_initial_conditions(text):
    initial_conditions = []
    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        x_str, y_str = line.split(",")
        initial_conditions.append((float(x_str), float(y_str)))

    return initial_conditions


def wrap_function(func, params, param_values):
    def wrapped(x, y):
        param_list = [param_values[str(p)] for p in params]
        return func(x, y, *param_list)

    return wrapped


def substitute_parameters(expr, params, param_values):
    if not params:
        return expr

    subs_dict = {p: param_values[str(p)] for p in params}
    return expr.subs(subs_dict)


def parse_parameter_value(value_str, param_name):
    try:
        value = sp.sympify(value_str)
        if not value.is_real:
            raise ValueError
        return float(value.evalf())
    except Exception as e:
        raise ValueError(
            f"Invalid value for parameter '{param_name}': {value_str}"
        ) from e


def get_parameter_value(param_name, default="1.0"):
    value_str = st.text_input(
        f"{param_name}",
        value=default,
        key=f"{param_name}_text",
    )
    return parse_parameter_value(value_str, param_name)


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


st.subheader("Initial conditions")
ics_text = st.text_area(
    "Enter one initial condition per line as: x0, y0",
    value="1, 1",
)

st.subheader("Domain and grid")
xmin = st.number_input("x_min", value=-3.0)
xmax = st.number_input("x_max", value=3.0)
ymin = st.number_input("y_min", value=-3.0)
ymax = st.number_input("y_max", value=3.0)
n = st.slider("Grid density", 10, 100, 40)
stride = st.slider("Vector field density (stride)", 1, 5, 2)

st.subheader("Integration")
t_max = st.number_input("t_max", value=20.0, min_value=0.1)
n_points = st.slider("Integration points", 100, 5000, 1000, step=100)
show_forward = st.checkbox("Show forward trajectories", value=True)
show_backward = st.checkbox("Show backward trajectories", value=False)

st.subheader("Fixed points")
show_fixed_points = st.checkbox("Show fixed points", value=True)
show_fixed_point_analysis = st.checkbox("Show fixed-point analysis", value=True)
fp_grid_density = st.slider("Fixed-point search density", 5, 25, 9, step=2)

st.subheader("Nullclines")
show_x_nullcline = st.checkbox("Show x-nullcline (dx/dt = 0)", value=False)
show_y_nullcline = st.checkbox("Show y-nullcline (dy/dt = 0)", value=False)
nullcline_n = st.slider("Nullcline density", 50, 300, 150, step=10)

field_style = st.radio(
    "Vector field style",
    ["Arrows", "Streamlines"],
    index=0,
)

normalize_vectors = st.checkbox("Normalize vector field", value=True)

if field_style == "Streamlines":
    streamline_density = st.slider("Streamline density", 0.5, 3.0, 1.0, 0.1)
    streamline_arrow_scale = st.slider("Streamline arrow scale", 0.02, 0.2, 0.09, 0.01)

if parse_error:
    st.error(parse_error)

if st.button(label="PLOT", type="primary"):
    try:
        if not parse_ok:
            raise ValueError(parse_error or "System could not be parsed.")

        initial_conditions = parse_initial_conditions(ics_text)

        f_wrapped = wrap_function(f_num, params, param_values)
        g_wrapped = wrap_function(g_num, params, param_values)

        f_expr_eval = substitute_parameters(f_expr, params, param_values)
        g_expr_eval = substitute_parameters(g_expr, params, param_values)

        X, Y = create_mesh(xmin, xmax, ymin, ymax, n, n)
        U, V = compute_vector_field(f_wrapped, g_wrapped, X, Y)
        if normalize_vectors:
            U, V = normalize_vector_field(U, V)

        Xn, Yn = create_mesh(xmin, xmax, ymin, ymax, nullcline_n, nullcline_n)
        Fn, Gn = compute_scalar_fields(f_wrapped, g_wrapped, Xn, Yn)

        if not np.isfinite(U).any() or not np.isfinite(V).any():
            st.warning(
                "Vector field could not be evaluated on the chosen domain. "
                "The system may be outside its domain there, e.g. log(x) for x <= 0 or sqrt(x) for x < 0."
            )

        if not np.isfinite(Fn).any() or not np.isfinite(Gn).any():
            st.warning(
                "Some scalar-field values are undefined on this domain, so nullclines may be incomplete."
            )

        if field_style == "Arrows":
            fig = create_phase_figure(X, Y, U, V, stride=stride)
        else:
            fig = create_streamline_figure(
                X,
                Y,
                U,
                V,
                density=streamline_density,
                arrow_scale=streamline_arrow_scale,
            )

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
                    t_span=(0, t_max),
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
                    t_span=(0, -t_max),
                    n_points=n_points,
                )
                fig = add_trajectory(fig, x_traj_b, y_traj_b, name=f"Backward {i}")
                all_x.append(np.asarray(x_traj_b))
                all_y.append(np.asarray(y_traj_b))

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        # fig = apply_axis_limits(fig, all_x, all_y, padding_ratio=0.06)
        fig = apply_axis_limits(fig, X.flatten(), Y.flatten(), padding_ratio=0.03)

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

        st.subheader("Parsed system")
        st.latex(r"\dot{x} = " + sp.latex(f_expr))

        st.latex(r"\dot{y} = " + sp.latex(g_expr))

        if params:
            st.subheader("Parameter values used")
            for p in params:
                name = str(p)
                st.write(f"{name} = {param_values[name]:.10g}")

        st.subheader("Evaluated system")
        st.latex(r"\dot{x} = " + sp.latex(f_expr_eval))
        st.latex(r"\dot{y} = " + sp.latex(g_expr_eval))

    except Exception as e:
        st.error(str(e))
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
