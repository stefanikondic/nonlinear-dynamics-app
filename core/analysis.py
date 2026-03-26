import numpy as np
import sympy as sp


x, y = sp.symbols("x y")


def remove_duplicate_points(points, tol=1e-6):
    unique = []

    for px, py in points:
        is_new = True
        for qx, qy in unique:
            if np.hypot(px - qx, py - qy) < tol:
                is_new = False
                break
        if is_new:
            unique.append((px, py))

    return unique


def point_in_domain(px, py, xmin, xmax, ymin, ymax, tol=1e-8):
    return xmin - tol <= px <= xmax + tol and ymin - tol <= py <= ymax + tol


def find_fixed_points_numeric(
    f_expr,
    g_expr,
    xmin,
    xmax,
    ymin,
    ymax,
    nx_guess=9,
    ny_guess=9,
    tol=1e-8,
):
    guesses_x = np.linspace(xmin, xmax, nx_guess)
    guesses_y = np.linspace(ymin, ymax, ny_guess)

    found_points = []

    for x0 in guesses_x:
        for y0 in guesses_y:
            try:
                sol = sp.nsolve(
                    (f_expr, g_expr),
                    (x, y),
                    (float(x0), float(y0)),
                    tol=1e-14,
                    maxsteps=100,
                    prec=40,
                )

                px = complex(sol[0])
                py = complex(sol[1])

                if abs(px.imag) > 1e-8 or abs(py.imag) > 1e-8:
                    continue

                px = float(px.real)
                py = float(py.real)

                if not point_in_domain(px, py, xmin, xmax, ymin, ymax):
                    continue

                # provjera da je stvarno rješenje
                fx_val = complex(sp.N(f_expr.subs({x: px, y: py})))
                gy_val = complex(sp.N(g_expr.subs({x: px, y: py})))

                if abs(fx_val) > tol or abs(gy_val) > tol:
                    continue

                found_points.append((px, py))

            except Exception:
                continue

    found_points = remove_duplicate_points(found_points, tol=1e-5)
    found_points.sort(key=lambda p: (p[0], p[1]))

    return found_points
