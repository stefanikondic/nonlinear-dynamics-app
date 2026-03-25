import sympy as sp

x, y = sp.symbols("x y")


def parse_system(f_str, g_str):
    f_expr = sp.sympify(f_str)
    g_expr = sp.sympify(g_str)

    f_num = sp.lambdify((x, y), f_expr, "numpy")
    g_num = sp.lambdify((x, y), g_expr, "numpy")

    return f_expr, g_expr, f_num, g_num
