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


def compute_jacobian_symbolic(f_expr, g_expr):
    return sp.Matrix([f_expr, g_expr]).jacobian([x, y])


def evaluate_jacobian_at_point(jacobian_expr, px, py):
    J_eval = jacobian_expr.subs({x: px, y: py})
    J_eval = np.array(J_eval.evalf(), dtype=float)
    return J_eval


def classify_fixed_point(eigenvalues, tol=1e-8):
    eigs = np.array(eigenvalues, dtype=complex)

    real_parts = np.real(eigs)
    imag_parts = np.imag(eigs)

    if np.any(np.abs(real_parts) < tol) and np.any(np.abs(imag_parts) < tol):
        return "degenerate"

    if real_parts[0] * real_parts[1] < -tol:
        return "saddle"

    if np.all(np.abs(imag_parts) < tol):
        if np.all(real_parts < -tol):
            return "stable node"
        if np.all(real_parts > tol):
            return "unstable node"
        if np.any(np.abs(real_parts) < tol):
            return "degenerate"

    if np.any(np.abs(imag_parts) > tol):
        if np.all(real_parts < -tol):
            return "stable spiral"
        if np.all(real_parts > tol):
            return "unstable spiral"
        if np.all(np.abs(real_parts) < tol):
            return "center"
        return "degenerate"

    return "degenerate"


def analyze_fixed_points(f_expr, g_expr, fixed_points):
    jacobian_expr = compute_jacobian_symbolic(f_expr, g_expr)
    results = []

    for px, py in fixed_points:
        J = evaluate_jacobian_at_point(jacobian_expr, px, py)
        eigenvalues = np.linalg.eigvals(J)
        classification = classify_fixed_point(eigenvalues)

        results.append(
            {
                "point": (px, py),
                "jacobian": J,
                "eigenvalues": eigenvalues,
                "classification": classification,
            }
        )

    return jacobian_expr, results


def extract_saddles(fixed_point_analysis):
    saddles = []

    for result in fixed_point_analysis:
        if result["classification"] == "saddle":
            J = result["jacobian"]
            eigvals, eigvecs = np.linalg.eig(J)

            saddles.append(
                {
                    "point": result["point"],
                    "jacobian": J,
                    "eigenvalues": eigvals,
                    "eigenvectors": eigvecs,
                }
            )

    return saddles
