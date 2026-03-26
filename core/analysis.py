import sympy as sp


def find_fixed_points_symbolic(f_expr, g_expr):
    x, y = sp.symbols("x y")

    solutions = sp.solve([f_expr, g_expr], (x, y), dict=True)

    fixed_points = []
    non_isolated = False

    for sol in solutions:
        if x not in sol or y not in sol:
            non_isolated = True
            continue

        x_val = sol[x]
        y_val = sol[y]

        if not x_val.is_number or not y_val.is_number:
            non_isolated = True
            continue

        if x_val.is_real is False or y_val.is_real is False:
            continue

        x_num = complex(sp.N(x_val))
        y_num = complex(sp.N(y_val))

        if abs(x_num.imag) < 1e-10 and abs(y_num.imag) < 1e-10:
            fixed_points.append((float(x_num.real), float(y_num.real)))

    return fixed_points, non_isolated
