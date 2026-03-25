from scipy.integrate import solve_ivp
import numpy as np


def create_mesh(xmin=-5, xmax=5, ymin=-5, ymax=5, nx=20, ny=20):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def compute_vector_field(f, g, X, Y):
    U = f(X, Y)
    V = g(X, Y)
    return U, V


def compute_scalar_fields(f, g, X, Y):
    F = f(X, Y)
    G = g(X, Y)
    return F, G


def integrate_trajectory(f, g, x0, y0, t_span=(0, 20), n_points=1000):
    def rhs(t, z):
        x, y = z
        return [f(x, y), g(x, y)]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        rhs,
        t_span,
        [x0, y0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="DOP853",
    )

    return sol.y[0], sol.y[1]
