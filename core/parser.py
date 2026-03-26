import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

x, y = sp.symbols("x y")

transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def preprocess_expression(expr: str) -> str:
    expr = expr.strip().lower()
    expr = re.sub(r"\s+", "", expr)

    # ln -> log
    expr = re.sub(r"\bln\b", "log", expr)

    # alternative inverse trig names
    expr = re.sub(r"\barcsin\b", "asin", expr)
    expr = re.sub(r"\barccos\b", "acos", expr)
    expr = re.sub(r"\barctan\b", "atan", expr)

    # function shorthands like sinx, sqrtx, lnx, asinx, sinhx...
    simple_func_map = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "exp": "exp",
        "sqrt": "sqrt",
        "log": "log",
        "abs": "Abs",
    }

    # Longer names first so sinhx is not partially matched as sin(hx)
    for user_name, sympy_name in sorted(
        simple_func_map.items(), key=lambda kv: -len(kv[0])
    ):
        expr = re.sub(
            rf"\b{user_name}([xy])\b",
            rf"{sympy_name}(\1)",
            expr,
        )

    # e^x -> exp(x)
    # e^(x+y) -> exp(x+y)
    expr = re.sub(r"\be\^\(([^()]+)\)", r"exp(\1)", expr)
    expr = re.sub(r"\be\^([a-z0-9\.]+)", r"exp(\1)", expr)

    return expr


def parse_single_expression(expr_str: str) -> sp.Expr:
    original = expr_str
    expr_str = preprocess_expression(expr_str)

    local_dict = {
        "x": x,
        "y": y,
        "pi": sp.pi,
        "e": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
    }

    try:
        return parse_expr(
            expr_str,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )
    except Exception as e:
        raise ValueError(
            f"Could not parse expression '{original}'. "
            f"Try examples like sin(x), sinx, e^x, sqrt(x), lnx, x^2 + 2xy."
        ) from e


def parse_system(f_str: str, g_str: str):
    f_expr = parse_single_expression(f_str)
    g_expr = parse_single_expression(g_str)

    f_num = sp.lambdify((x, y), f_expr, modules="numpy")
    g_num = sp.lambdify((x, y), g_expr, modules="numpy")

    return f_expr, g_expr, f_num, g_num
