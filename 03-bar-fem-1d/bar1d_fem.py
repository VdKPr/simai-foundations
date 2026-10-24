"""
Module 03: 1D bar FEM from scratch.

A steel bar of length L is fixed at x = 0. It carries a uniform axial load f
(N per metre of length, think self-weight of a hanging bar) and a point load P
at the free end x = L.

Strong form:   -(EA u')' = f,    u(0) = 0  (support),   EA u'(L) = P  (tip load)

Exact solution (for checking):
    u(x) = f/(EA) * (L x - x^2/2) + P x / EA
    N(x) = EA u'(x) = f (L - x) + P          (internal axial force)

Run:  python bar1d_fem.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- problem data (SI units) ----------------------------------------------
L = 1.0           # m     bar length
E = 200e9         # Pa    Young's modulus of steel
A = 1e-4          # m^2   cross-section (1 cm^2)
EA = E * A        # N     axial stiffness
f = 1e4           # N/m   uniform distributed axial load
P = 5e3           # N     point load at the tip

OUT = Path(__file__).parent / "out"


def exact_u(x):
    return f / EA * (L * x - x**2 / 2) + P * x / EA


def exact_force(x):
    return f * (L - x) + P


def solve_fem(n_el, apply_support=True):
    """Linear 2-node elements, all the same length h = L / n_el."""
    n_nodes = n_el + 1
    x = np.linspace(0.0, L, n_nodes)
    h = L / n_el

    K = np.zeros((n_nodes, n_nodes))   # global stiffness matrix
    F = np.zeros(n_nodes)              # global load vector

    # Element stiffness: a spring of stiffness EA/h between its two nodes
    k_e = (EA / h) * np.array([[1.0, -1.0],
                               [-1.0, 1.0]])
    # Element load: the distributed load on the element, split half to each node
    f_e = (f * h / 2.0) * np.array([1.0, 1.0])

    # Assembly: add each element's 2x2 block into the rows/columns of its nodes
    for e in range(n_el):
        nodes = [e, e + 1]
        K[np.ix_(nodes, nodes)] += k_e
        F[nodes] += f_e

    # Natural BC: the tip force just gets added to the last node's load
    F[-1] += P

    if not apply_support:
        # No support: the bar can slide as a rigid body, K is singular
        return x, np.linalg.solve(K, F), None, K

    # Essential BC u(0) = 0: remove node 0's row and column, solve for the rest
    free = np.arange(1, n_nodes)
    u = np.zeros(n_nodes)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    # Post-processing: strain is constant inside each element, so is the force
    force = EA * np.diff(u) / h
    return x, u, force, K


# 3-point Gauss rule on [-1, 1]: integrates polynomials up to degree 5 exactly
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(3)


def relative_errors(x, u, force):
    """Relative L2 errors of displacement and axial force over the whole bar."""
    err_u = ref_u = err_n = ref_n = 0.0
    for e in range(len(x) - 1):
        xa, xb = x[e], x[e + 1]
        h = xb - xa
        xq = xa + (GAUSS_X + 1.0) * h / 2.0      # Gauss points mapped into the element
        wq = GAUSS_W * h / 2.0
        u_h = u[e] + (u[e + 1] - u[e]) * (xq - xa) / h   # FE displacement: straight line
        err_u += np.sum(wq * (u_h - exact_u(xq)) ** 2)
        ref_u += np.sum(wq * exact_u(xq) ** 2)
        err_n += np.sum(wq * (force[e] - exact_force(xq)) ** 2)
        ref_n += np.sum(wq * exact_force(xq) ** 2)
    return np.sqrt(err_u / ref_u), np.sqrt(err_n / ref_n)


def main():
    OUT.mkdir(exist_ok=True)

    # 1) The assembled stiffness matrix for 4 elements
    n_el = 4
    x, u, force, K = solve_fem(n_el)
    h = L / n_el
    print(f"Global K for {n_el} elements, scaled by h/EA (before applying the support):")
    print(np.round(K * h / EA).astype(int))
    print("Same [-1, 2, -1] pattern as the finite-difference matrix in module 01.\n")

    # 2) Nodal displacements vs exact
    print(f"{'node':>4} {'x [m]':>6} {'u FEM [mm]':>12} {'u exact [mm]':>13} {'difference [mm]':>16}")
    for i, (xi, ui) in enumerate(zip(x, u)):
        print(f"{i:4d} {xi:6.2f} {1000 * ui:12.6f} {1000 * exact_u(xi):13.6f} {1000 * (ui - exact_u(xi)):16.2e}")
    print("Nodal values match the exact solution (special to this 1D problem).\n")

    # 3) Mesh convergence study
    print(f"{'elements':>8} {'h [m]':>8} {'disp. error':>12} {'rate':>5} {'force error':>12} {'rate':>5}")
    prev = None
    for n in [2, 4, 8, 16, 32, 64, 128]:
        xn, un, fn, _ = solve_fem(n)
        eu, en = relative_errors(xn, un, fn)
        if prev is None:
            print(f"{n:8d} {L / n:8.4f} {eu:12.3e} {'':>5} {en:12.3e} {'':>5}")
        else:
            ru = np.log2(prev[0] / eu)
            rn = np.log2(prev[1] / en)
            print(f"{n:8d} {L / n:8.4f} {eu:12.3e} {ru:5.2f} {en:12.3e} {rn:5.2f}")
        prev = (eu, en)
    print("Displacement error ~ h^2 (rate 2). Force error ~ h (rate 1): derived")
    print("quantities like stress converge one order slower than displacement.\n")

    # 4) What happens without the support?
    try:
        solve_fem(4, apply_support=False)
        print("Solved without a support (numerically meaningless result).")
    except np.linalg.LinAlgError as err:
        print(f"Without the support: numpy says '{err}'. The bar can slide freely")
        print("(rigid-body motion), so K has no unique solution.\n")

    # 5) Plots
    xf = np.linspace(0, L, 400)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].plot(xf, 1000 * exact_u(xf), "k-", label="exact")
    ax[0].plot(x, 1000 * u, "o--", label=f"FEM, {n_el} elements")
    ax[0].set(xlabel="x [m]", ylabel="u [mm]", title="Displacement")
    ax[0].legend()

    ax[1].plot(xf, exact_force(xf) / 1000, "k-", label="exact")
    ax[1].stairs(force / 1000, x, label=f"FEM, {n_el} elements", baseline=None)
    ax[1].set(xlabel="x [m]", ylabel="N [kN]", title="Axial force (one value per element)")
    ax[1].legend()

    ns = np.array([2, 4, 8, 16, 32, 64, 128])
    errs = np.array([relative_errors(*solve_fem(n)[:3]) for n in ns])
    ax[2].loglog(L / ns, errs[:, 0], "o-", label="displacement")
    ax[2].loglog(L / ns, errs[:, 1], "s-", label="axial force")
    ax[2].set(xlabel="element size h [m]", ylabel="relative L2 error", title="Mesh convergence")
    ax[2].legend()

    plt.tight_layout()
    path = OUT / "bar1d_fem.png"
    plt.savefig(path, dpi=150)
    print(f"Tip displacement: FEM {1000 * u[-1]:.4f} mm, exact {1000 * exact_u(L):.4f} mm")
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
