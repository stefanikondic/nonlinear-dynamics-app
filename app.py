import traceback

import numpy as np
import streamlit as st
import sympy as sp

from core.analysis import (
    analyze_fixed_points,
    extract_saddles,
    find_fixed_points_numeric,
)
from core.plotting import (
    add_fixed_points,
    add_nullclines_from_contours,
    add_separatrices,
    add_trajectory,
    apply_axis_limits,
    create_phase_figure,
    create_streamline_figure,
    create_streamline_figure_custom,
)
from core.system import (
    compute_scalar_fields,
    compute_separatrices,
    compute_vector_field,
    create_mesh,
    integrate_trajectory,
    normalize_vector_field,
)
from ui.layout import render_advanced_settings, render_top_inputs
from ui.styles import apply_app_styles
from utils.helpers import parse_initial_conditions, substitute_parameters, wrap_function

st.set_page_config(page_title="Nonlinear Dynamics App", layout="centered")
apply_app_styles()

st.title("Nonlinear Dynamics App")
st.caption(
    "Enter a system and click PLOT. Open Advanced settings only if you want finer control."
)

top_inputs = render_top_inputs()

if top_inputs["parse_error"]:
    st.error(top_inputs["parse_error"])

settings = render_advanced_settings()

st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
plot_clicked = st.button(label="PLOT", type="primary")

if plot_clicked:
    try:
        if not top_inputs["parse_ok"]:
            raise ValueError(top_inputs["parse_error"] or "System could not be parsed.")

        initial_conditions = parse_initial_conditions(top_inputs["ics_text"])

        f_wrapped = wrap_function(
            top_inputs["f_num"],
            top_inputs["params"],
            top_inputs["param_values"],
        )
        g_wrapped = wrap_function(
            top_inputs["g_num"],
            top_inputs["params"],
            top_inputs["param_values"],
        )

        def rhs(t, z):
            x, y = z
            return [f_wrapped(x, y), g_wrapped(x, y)]

        f_expr_eval = substitute_parameters(
            top_inputs["f_expr"],
            top_inputs["params"],
            top_inputs["param_values"],
        )
        g_expr_eval = substitute_parameters(
            top_inputs["g_expr"],
            top_inputs["params"],
            top_inputs["param_values"],
        )

        X, Y = create_mesh(
            settings["xmin"],
            settings["xmax"],
            settings["ymin"],
            settings["ymax"],
            settings["n"],
            settings["n"],
        )
        U_raw, V_raw = compute_vector_field(f_wrapped, g_wrapped, X, Y)

        if settings["normalize_vectors"]:
            U_plot, V_plot = normalize_vector_field(U_raw, V_raw)
        else:
            U_plot, V_plot = U_raw, V_raw

        Xn, Yn = create_mesh(
            settings["xmin"],
            settings["xmax"],
            settings["ymin"],
            settings["ymax"],
            settings["nullcline_n"],
            settings["nullcline_n"],
        )
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

        if settings["field_style"] == "Arrows":
            fig = create_phase_figure(
                X,
                Y,
                U_plot,
                V_plot,
                stride=settings["stride"],
            )

        elif settings["field_style"] == "Streamlines (classic)":
            fig = create_streamline_figure(
                X,
                Y,
                U_plot,
                V_plot,
                density=settings["streamline_density"],
                arrow_scale=settings["streamline_arrow_scale"],
            )

        elif settings["field_style"] == "Streamlines (robust)":
            fig = create_streamline_figure_custom(
                rhs,
                X,
                Y,
                U_raw,
                V_raw,
                n_seeds=settings["n_seeds"],
                t_max=settings["streamline_t_max"],
                max_step=0.05,
            )

        else:
            raise ValueError(f"Unknown field style: {settings['field_style']}")

        fixed_points = []
        if settings["show_fixed_points"]:
            fixed_points = find_fixed_points_numeric(
                f_expr_eval,
                g_expr_eval,
                settings["xmin"],
                settings["xmax"],
                settings["ymin"],
                settings["ymax"],
                nx_guess=settings["fp_grid_density"],
                ny_guess=settings["fp_grid_density"],
            )
            fig = add_fixed_points(fig, fixed_points)

        jacobian_expr = None
        fixed_point_analysis = []

        if (
            settings["show_fixed_points"]
            and settings["show_fixed_point_analysis"]
            and fixed_points
        ):
            jacobian_expr, fixed_point_analysis = analyze_fixed_points(
                f_expr_eval,
                g_expr_eval,
                fixed_points,
            )

        if settings["show_separatrices"] and fixed_point_analysis:
            try:
                saddles = extract_saddles(fixed_point_analysis)

                branches = compute_separatrices(
                    rhs,
                    saddles,
                    eps=settings["sep_eps"],
                    t_max=settings["sep_tmax"],
                    n_points=settings["sep_n"],
                )

                fig = add_separatrices(fig, branches)

            except Exception as e:
                st.warning(f"Separatrices failed: {e}")

        fig = add_nullclines_from_contours(
            fig,
            Xn,
            Yn,
            Fn,
            Gn,
            show_x_nullcline=settings["show_x_nullcline"],
            show_y_nullcline=settings["show_y_nullcline"],
        )

        all_x = [X.flatten()]
        all_y = [Y.flatten()]

        for i, (x0, y0) in enumerate(initial_conditions, start=1):
            if settings["show_forward"]:
                x_traj, y_traj = integrate_trajectory(
                    f_wrapped,
                    g_wrapped,
                    x0,
                    y0,
                    t_span=(0, settings["traj_t_max"]),
                    n_points=settings["n_points"],
                )
                fig = add_trajectory(fig, x_traj, y_traj, name=f"Forward {i}")
                all_x.append(np.asarray(x_traj))
                all_y.append(np.asarray(y_traj))

            if settings["show_backward"]:
                x_traj_b, y_traj_b = integrate_trajectory(
                    f_wrapped,
                    g_wrapped,
                    x0,
                    y0,
                    t_span=(0, -settings["traj_t_max"]),
                    n_points=settings["n_points"],
                )
                fig = add_trajectory(fig, x_traj_b, y_traj_b, name=f"Backward {i}")
                all_x.append(np.asarray(x_traj_b))
                all_y.append(np.asarray(y_traj_b))

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        fig = apply_axis_limits(fig, X.flatten(), Y.flatten(), padding_ratio=0.03)

        st.markdown("---")
        st.plotly_chart(fig, width="stretch")

        if settings["show_fixed_points"]:
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

            if (
                settings["show_fixed_points"]
                and settings["show_fixed_point_analysis"]
                and fixed_point_analysis
            ):
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
            st.latex(r"\dot{x} = " + sp.latex(top_inputs["f_expr"]))
            st.latex(r"\dot{y} = " + sp.latex(top_inputs["g_expr"]))

            if top_inputs["params"]:
                st.subheader("Parameter values used")
                for p in top_inputs["params"]:
                    name = str(p)
                    st.write(f"{name} = {top_inputs['param_values'][name]:.10g}")

        with info_right:
            st.subheader("Evaluated system")
            st.latex(r"\dot{x} = " + sp.latex(f_expr_eval))
            st.latex(r"\dot{y} = " + sp.latex(g_expr_eval))

    except Exception as e:
        st.error(str(e))
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
