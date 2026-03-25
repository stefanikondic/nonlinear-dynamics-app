import streamlit as st

from core.parser import parse_system
from core.system import create_mesh, compute_vector_field, integrate_trajectory
from core.plotting import plot_trajectory, plot_vector_field

st.title("Nonlinear Dynamics App")

# unos funkcija
f_str = st.text_input("dx/dt =", "y - y**3")
g_str = st.text_input("dy/dt =", "-x - y**2")

# domen
xmin = st.number_input("xmin", value=-5.0)
xmax = st.number_input("xmax", value=5.0)
ymin = st.number_input("ymin", value=-5.0)
ymax = st.number_input("ymax", value=5.0)

# rezolucija
n = st.slider("Grid density", 10, 50, 20)

if st.button("Plot"):
    f_expr, g_expr, f_num, g_num = parse_system(f_str, g_str)

    X, Y = create_mesh(xmin, xmax, ymin, ymax, n, n)
    U, V = compute_vector_field(f_num, g_num, X, Y)

    fig, ax = plot_vector_field(X, Y, U, V)

    initial_conditions = [(1, 0), (-1, 1), (2, -2)]
    for x0, y0 in initial_conditions:
        x_traj, y_traj = integrate_trajectory(f_num, g_num, x0, y0)
        plot_trajectory(ax, x_traj, y_traj)

    st.pyplot(fig)

st.subheader("Initial conditions")
