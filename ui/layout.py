import streamlit as st

from core.parser import parse_system
from ui.controls import get_parameter_value


def render_top_inputs():
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

    return {
        "f_str": f_str,
        "g_str": g_str,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "f_expr": f_expr,
        "g_expr": g_expr,
        "f_num": f_num,
        "g_num": g_num,
        "params": params,
        "param_values": param_values,
        "ics_text": ics_text,
    }


def render_advanced_settings():
    settings = {
        "xmin": -3.0,
        "xmax": 3.0,
        "ymin": -3.0,
        "ymax": 3.0,
        "n": 40,
        "stride": 2,
        "traj_t_max": 20.0,
        "n_points": 1000,
        "show_forward": True,
        "show_backward": False,
        "field_style": "Arrows",
        "normalize_vectors": True,
        "n_seeds": 50,
        "streamline_t_max": 10,
        "streamline_density": 2.0,
        "streamline_arrow_scale": 0.09,
        "show_fixed_points": True,
        "show_fixed_point_analysis": True,
        "fp_grid_density": 9,
        "show_x_nullcline": False,
        "show_y_nullcline": False,
        "nullcline_n": 150,
    }

    with st.expander("Advanced settings", expanded=False):
        adv_left, adv_right = st.columns([1, 1], gap="large")

        with adv_left:
            st.subheader("Domain and grid")
            settings["xmin"] = st.number_input("x_min", value=settings["xmin"])
            settings["xmax"] = st.number_input("x_max", value=settings["xmax"])
            settings["ymin"] = st.number_input("y_min", value=settings["ymin"])
            settings["ymax"] = st.number_input("y_max", value=settings["ymax"])
            settings["n"] = st.slider("Grid density", 10, 100, settings["n"])
            settings["stride"] = st.slider(
                "Vector field density (stride)", 1, 5, settings["stride"]
            )

            st.subheader("Field style")
            settings["field_style"] = st.radio(
                "Vector field style",
                ["Arrows", "Streamlines (classic)", "Streamlines (robust)"],
                index=0,
                horizontal=False,
            )
            settings["normalize_vectors"] = st.checkbox(
                "Normalize vector field",
                value=settings["normalize_vectors"],
            )

            if settings["field_style"] == "Streamlines (classic)":
                settings["streamline_density"] = st.slider(
                    "Streamline density",
                    0.5,
                    3.0,
                    settings["streamline_density"],
                    0.1,
                )
                settings["streamline_arrow_scale"] = st.slider(
                    "Streamline arrow scale",
                    0.02,
                    0.2,
                    settings["streamline_arrow_scale"],
                    0.01,
                )

            elif settings["field_style"] == "Streamlines (robust)":
                settings["n_seeds"] = st.slider(
                    "Number of streamlines", 10, 200, settings["n_seeds"]
                )
                settings["streamline_t_max"] = st.slider(
                    "Streamline integration time",
                    1,
                    50,
                    settings["streamline_t_max"],
                )

            if (
                settings["field_style"]
                in ["Streamlines (classic)", "Streamlines (robust)"]
                and settings["normalize_vectors"]
            ):
                st.warning(
                    "Normalized vector fields may behave poorly near zeros of the field."
                )

        with adv_right:
            st.subheader("Trajectories")
            settings["traj_t_max"] = st.number_input(
                "Trajectory t_max",
                value=settings["traj_t_max"],
                min_value=0.1,
            )
            settings["n_points"] = st.slider(
                "Integration points",
                100,
                5000,
                settings["n_points"],
                step=100,
            )
            settings["show_forward"] = st.checkbox(
                "Show forward trajectories",
                value=settings["show_forward"],
            )
            settings["show_backward"] = st.checkbox(
                "Show backward trajectories",
                value=settings["show_backward"],
            )

            st.subheader("Fixed points")
            settings["show_fixed_points"] = st.checkbox(
                "Show fixed points",
                value=settings["show_fixed_points"],
            )
            settings["show_fixed_point_analysis"] = st.checkbox(
                "Show fixed-point analysis",
                value=settings["show_fixed_point_analysis"],
            )
            settings["fp_grid_density"] = st.slider(
                "Fixed-point search density",
                5,
                25,
                settings["fp_grid_density"],
                step=2,
            )

            st.subheader("Nullclines")
            settings["show_x_nullcline"] = st.checkbox(
                "Show x-nullcline (dx/dt = 0)",
                value=settings["show_x_nullcline"],
            )
            settings["show_y_nullcline"] = st.checkbox(
                "Show y-nullcline (dy/dt = 0)",
                value=settings["show_y_nullcline"],
            )
            settings["nullcline_n"] = st.slider(
                "Nullcline density",
                50,
                300,
                settings["nullcline_n"],
                step=10,
            )
            settings["show_separatrices"] = st.checkbox(
                "Show separatrices",
                value=False,
            )
            settings["sep_eps"] = st.number_input(
                "Separatrix epsilon",
                value=1e-4,
                format="%.6f",
            )
            settings["sep_tmax"] = st.number_input(
                "Separatrix t_max",
                value=20.0,
            )
            settings["sep_n"] = st.slider(
                "Separatrix resolution",
                200,
                5000,
                1000,
                step=100,
            )

    return settings
