import streamlit as st

from core.parser import parse_system
from core.system import create_mesh, compute_vector_field, integrate_trajectory
from core.plotting import plot_vector_field, plot_trajectory

st.title("Nonlinear Dynamics App")

f_str = st.text_input("dx/dt =", "y - y**3")
g_str = st.text_input("dy/dt =", "-x - y**2")

st.subheader("Domain")
xmin = st.number_input("xmin", value=-5.0)
xmax = st.number_input("xmax", value=5.0)
ymin = st.number_input("ymin", value=-5.0)
ymax = st.number_input("ymax", value=5.0)

n = st.slider("Grid density", 10, 50, 20)

st.subheader("Integration")
t_max = st.number_input("t_max", value=20.0, min_value=0.1)
n_points = st.slider("Integration points", 100, 5000, 1000, step=100)

st.subheader("Initial conditions")
ics_text = st.text_area(
    "Enter one initial condition per line as: x0, y0", value="1, 0\n-1, 1\n2, -2"
)

show_forward = st.checkbox("Show forward trajectories", value=True)
show_backward = st.checkbox("Show backward trajectories", value=True)


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


if st.button("Plot"):
    try:
        f_expr, g_expr, f_num, g_num = parse_system(f_str, g_str)
        initial_conditions = parse_initial_conditions(ics_text)

        X, Y = create_mesh(xmin, xmax, ymin, ymax, n, n)
        U, V = compute_vector_field(f_num, g_num, X, Y)

        fig, ax = plot_vector_field(X, Y, U, V)

        for x0, y0 in initial_conditions:
            if show_forward:
                x_traj, y_traj = integrate_trajectory(
                    f_num, g_num, x0, y0, t_span=(0, t_max), n_points=n_points
                )
                plot_trajectory(ax, x_traj, y_traj)

            if show_backward:
                x_traj_b, y_traj_b = integrate_trajectory(
                    f_num, g_num, x0, y0, t_span=(0, -t_max), n_points=n_points
                )
                plot_trajectory(ax, x_traj_b, y_traj_b)

        st.pyplot(fig)

        st.subheader("Parsed system")
        st.latex(r"\dot{x} = " + str(f_expr))
        st.latex(r"\dot{y} = " + str(g_expr))

    except Exception as e:
        st.error(f"Error: {e}")
