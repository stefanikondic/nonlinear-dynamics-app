import numpy as np
from scipy.integrate import solve_ivp


def create_mesh(xmin=-5, xmax=5, ymin=-5, ymax=5, nx=20, ny=20):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def safe_evaluate(func, X, Y):
    with np.errstate(all="ignore"):
        Z = func(X, Y)

    if np.isscalar(Z):
        Z = np.full_like(X, Z, dtype=float)
    else:
        Z = np.asarray(Z, dtype=float)

    return Z


def compute_vector_field(f, g, X, Y):
    U = safe_evaluate(f, X, Y)
    V = safe_evaluate(g, X, Y)
    return U, V


def compute_scalar_fields(f, g, X, Y):
    F = safe_evaluate(f, X, Y)
    G = safe_evaluate(g, X, Y)
    return F, G


def integrate_trajectory(f, g, x0, y0, t_span=(0, 20), n_points=1000):
    def rhs(t, z):
        x, y = z
        dx = f(x, y)
        dy = g(x, y)
        return [dx, dy]

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


def normalize_vector_field(U, V, eps=1e-12):
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude < eps] = 1.0  # izbjegni dijeljenje nulom

    U_norm = U / magnitude
    V_norm = V / magnitude

    return U_norm, V_norm


def _normalize(v, tol=1e-12):
    n = np.linalg.norm(v)
    if n < tol:
        return None
    return v / n


def integrate_separatrix(
    rhs,
    point,
    direction,
    eps=1e-4,
    t_max=20,
    n_points=1000,
    forward=True,
):
    v = _normalize(direction)
    if v is None:
        return None

    z0 = np.array(point) + eps * v

    t_span = (0, t_max) if forward else (0, -t_max)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        rhs,
        t_span,
        z0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        method="DOP853",
    )

    return sol.y[0], sol.y[1]


def compute_separatrices(rhs, saddle_data, eps=1e-4, t_max=20, n_points=1000):
    branches = []

    for saddle in saddle_data:
        point = saddle["point"]
        eigvals = saddle["eigenvalues"]
        eigvecs = saddle["eigenvectors"]

        for i in range(len(eigvals)):
            lam = np.real(eigvals[i])
            v = np.real(eigvecs[:, i])

            if np.linalg.norm(v) < 1e-12:
                continue

            if lam > 0:
                # unstable → forward
                for sign in (+1, -1):
                    res = integrate_separatrix(
                        rhs, point, sign * v, eps, t_max, n_points, forward=True
                    )
                    if res:
                        x, y = res
                        branches.append(("unstable", x, y))

            elif lam < 0:
                # stable → backward
                for sign in (+1, -1):
                    res = integrate_separatrix(
                        rhs, point, sign * v, eps, t_max, n_points, forward=False
                    )
                    if res:
                        x, y = res
                        branches.append(("stable", x, y))

    return branches
