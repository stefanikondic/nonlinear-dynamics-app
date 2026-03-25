import sympy as sp
import numpy as np


x, y = sp.symbols("x y")


def find_fixed_points_symbolic(f_expr, g_expr):
    solutions = sp.solve((f_expr, g_expr), (x, y), dict=True)
    fixed_points = []

    for sol in solutions:
        x_val = sol[x]
        y_val = sol[y]

        if x_val.is_real is False or y_val.is_real is False:
            continue

        x_num = complex(sp.N(x_val))
        y_num = complex(sp.N(y_val))

        if abs(x_num.imag) < 1e-10 and abs(y_num.imag) < 1e-10:
            fixed_points.append((float(x_num.real), float(y_num.real)))

    return remove_duplicate_points(fixed_points)


def remove_duplicate_points(points, tol=1e-8):
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
