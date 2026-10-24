"""
Module 01: turn a PDE into a matrix equation with finite differences.

Problem: a steel rod of length L. Both ends are held at ambient temperature.
Heat is generated inside the rod, most strongly in the middle:

    -k T''(x) = q(x),    T(0) = 0,  T(L) = 0,    q(x) = q0 * sin(pi x / L)

T is the temperature rise above ambient (deg C). The exact solution is

    T(x) = q0 L^2 / (k pi^2) * sin(pi x / L)

Run:  python fd_1d_heat.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- problem data (SI units) ----------------------------------------------
L = 0.5      # m      rod length
k = 50.0     # W/m.K  conductivity of steel
q0 = 1e5     # W/m^3  peak heat generation

OUT = Path(__file__).parent / "out"


def q(x):
    return q0 * np.sin(np.pi * x / L)


def exact_T(x):
    return q0 * L**2 / (k * np.pi**2) * np.sin(np.pi * x / L)


def solve_fd(n):
    """Split the rod into n equal gaps (n + 1 grid points) and solve K T = F."""
    x = np.linspace(0.0, L, n + 1)
    h = L / n
    m = n - 1                      # unknowns: interior points only (ends are fixed)

    # Central difference: T'' at point i  ~  (T[i-1] - 2 T[i] + T[i+1]) / h^2
    # so  -k T''  becomes  (k / h^2) * (-T[i-1] + 2 T[i] - T[i+1])
    K = np.zeros((m, m))
    for i in range(m):
        K[i, i] = 2.0
        if i > 0:
            K[i, i - 1] = -1.0
        if i < m - 1:
            K[i, i + 1] = -1.0
    K *= k / h**2

    F = q(x[1:-1])                 # right-hand side at interior points

    T = np.zeros(n + 1)            # T[0] = T[n] = 0 already (Dirichlet ends)
    T[1:-1] = np.linalg.solve(K, F)
    return x, T, K


def main():
    OUT.mkdir(exist_ok=True)

    # 1) Look at the matrix for a tiny grid
    n_small = 5
    _, _, K = solve_fd(n_small)
    h = L / n_small
    print(f"Matrix K for n = {n_small} gaps ({n_small - 1} unknowns), scaled by h^2/k:")
    print(np.round(K * h**2 / k).astype(int))
    print("Each row only touches its neighbours -> the matrix is tridiagonal (sparse).\n")

    # 2) Refine the grid and watch the error
    print(f"{'n gaps':>7} {'h [mm]':>9} {'max error [degC]':>17} {'error ratio':>12}")
    prev = None
    for n in [4, 8, 16, 32, 64, 128]:
        x, T, _ = solve_fd(n)
        err = np.max(np.abs(T - exact_T(x)))
        ratio = "" if prev is None else f"{prev / err:12.2f}"
        print(f"{n:7d} {1000 * L / n:9.2f} {err:17.4e} {ratio}")
        prev = err
    print("Halving h cuts the error by ~4x  ->  error ~ h^2  (second-order accurate).\n")

    # 3) Plot a coarse solution against the exact one
    x, T, _ = solve_fd(8)
    xf = np.linspace(0, L, 400)
    plt.figure(figsize=(6, 4))
    plt.plot(xf * 1000, exact_T(xf), "k-", label="exact")
    plt.plot(x * 1000, T, "o--", label="finite difference, n = 8")
    plt.xlabel("x [mm]")
    plt.ylabel("temperature rise [°C]")
    plt.title("Rod with internal heat generation")
    plt.legend()
    plt.tight_layout()
    path = OUT / "fd_1d_heat.png"
    plt.savefig(path, dpi=150)
    print(f"Peak temperature rise (exact): {exact_T(L / 2):.2f} degC")
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
