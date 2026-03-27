import sympy as sp


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


def wrap_function(func, params, param_values):
    def wrapped(x, y):
        param_list = [param_values[str(p)] for p in params]
        return func(x, y, *param_list)

    return wrapped


def substitute_parameters(expr, params, param_values):
    if not params:
        return expr

    subs_dict = {p: param_values[str(p)] for p in params}
    return expr.subs(subs_dict)


def parse_parameter_value(value_str, param_name):
    try:
        value = sp.sympify(value_str)
        if not value.is_real:
            raise ValueError
        return float(value.evalf())
    except Exception as e:
        raise ValueError(
            f"Invalid value for parameter '{param_name}': {value_str}"
        ) from e
