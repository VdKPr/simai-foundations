"""
Module 07: nonlinear problems and Newton-Raphson.

Part 1 - one unknown: a softening spring.
         Newton's quadratic convergence, and what failure looks like when the
         load is beyond what the spring can carry (a limit point).

Part 2 - many unknowns: the 1D Allen-Cahn equation (your thesis equation, in
         its simpler second-order form), stepped in time with implicit Euler.
         Every time step is a nonlinear system solved by Newton. We compare
         three initial guesses and count Newton iterations.

Run:  python newton.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OUT = Path(__file__).parent / "out"

# ---------------------------------------------------------------------------
# Part 1: softening spring, N(x) = k (x - x^3 / (3 a^2))
# It stiffens less and less as it stretches; the most it can carry is at x = a.
# ---------------------------------------------------------------------------
K_SPRING = 1000.0    # N/mm   initial stiffness
A_LIM = 1.0          # mm     stretch at which the spring carries its maximum force
N_MAX = 2 / 3 * K_SPRING * A_LIM


def spring_force(x):
    return K_SPRING * (x - x**3 / (3 * A_LIM**2))


def tangent_stiffness(x):            # dN/dx: the slope of the force-displacement curve
    return K_SPRING * (1 - x**2 / A_LIM**2)


def newton_spring(load, x0=0.0, tol=1e-10, max_iter=25):
    x = x0
    history = []
    for it in range(max_iter):
        residual = spring_force(x) - load        # out-of-balance force
        history.append((it, x, residual))
        if abs(residual) < tol:
            return x, history, True
        x = x - residual / tangent_stiffness(x)  # the Newton update
    return x, history, False


def part1():
    print("PART 1: softening spring, max force it can carry = "
          f"{N_MAX:.1f} N at x = {A_LIM} mm\n")
    for load in [300.0, 650.0, 700.0]:
        x, hist, ok = newton_spring(load)
        print(f"Load {load:.0f} N  ->  {'converged' if ok else 'FAILED'} after {len(hist) - 1} iterations")
        print(f"   {'iter':>4} {'x [mm]':>14} {'out-of-balance force [N]':>26}")
        for it, xi, r in hist[:8]:
            print(f"   {it:4d} {xi:14.10f} {r:26.3e}")
        if len(hist) > 8:
            print("   ...")
        print()
    print("300 N: residual 3e2 -> 9e0 -> 3e-2 -> 3e-7 -> 0. Correct digits roughly double each step.")
    print("700 N: more than the spring can carry. No solution exists, so Newton wanders.\n")


# ---------------------------------------------------------------------------
# Part 2: 1D Allen-Cahn, phi_t = eps^2 phi_xx - (phi^3 - phi), zero-flux ends
# Implicit Euler: R(phi) = (phi - phi_old)/dt - eps^2 Lap phi + phi^3 - phi = 0
# ---------------------------------------------------------------------------
EPS = 0.03
N_GRID = 200
DT = 0.05
N_STEPS = 60


def laplacian_neumann(n, h):
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    Lap = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    Lap[0, 1] = 2.0          # zero-flux end: mirror the neighbour
    Lap[-1, -2] = 2.0
    return (Lap / h**2).tocsr()


def newton_step(phi_old, guess, Lap, tol=1e-10, max_iter=50):
    """Solve one implicit Euler step. Returns the new field and the iteration count."""
    I = sp.identity(len(phi_old), format="csr")
    phi = guess.copy()
    for it in range(max_iter):
        R = (phi - phi_old) / DT - EPS**2 * (Lap @ phi) + phi**3 - phi
        if np.abs(R).max() < tol:
            return phi, it
        J = I / DT - EPS**2 * Lap + sp.diags(3 * phi**2 - 1)   # Jacobian = dR/dphi
        phi = phi + spla.spsolve(J.tocsc(), -R)
    raise RuntimeError("Newton did not converge")


def part2():
    x = np.linspace(0, 1, N_GRID)
    h = x[1] - x[0]
    Lap = laplacian_neumann(N_GRID, h)
    rng = np.random.default_rng(0)
    phi0 = 0.3 * np.sin(3 * np.pi * x) + 0.05 * rng.standard_normal(N_GRID)

    print(f"PART 2: 1D Allen-Cahn, {N_GRID} unknowns, {N_STEPS} implicit steps of dt = {DT}")
    print(f"{'initial guess for each step':>34} {'total Newton iters':>19} {'per step':>9}")
    results = {}
    for name in ["zero", "previous step", "linear extrapolation"]:
        phi_prev, phi = phi0.copy(), phi0.copy()
        snapshots, iters = [phi0.copy()], []
        for n in range(N_STEPS):
            if name == "zero":
                guess = np.zeros_like(phi)
            elif name == "previous step":
                guess = phi
            else:                                    # 2 phi_n - phi_(n-1)
                guess = 2 * phi - phi_prev if n > 0 else phi
            phi_new, it = newton_step(phi, guess, Lap)
            phi_prev, phi = phi, phi_new
            iters.append(it)
            snapshots.append(phi.copy())
        results[name] = (np.array(iters), snapshots)
        print(f"{name:>34} {sum(iters):19d} {np.mean(iters):9.2f}")

    same = all(np.allclose(results[k][1][-1], results["zero"][1][-1], atol=1e-8) for k in results)
    print(f"\nAll three guesses give the same final answer: {same}")
    print("The guess changes how many iterations Newton needs, not what it converges to.\n")
    return x, results


def main():
    OUT.mkdir(exist_ok=True)
    part1()
    x, results = part2()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xs = np.linspace(0, 1.5, 300)
    ax[0].plot(xs, spring_force(xs), "k-")
    ax[0].axhline(N_MAX, color="r", ls="--", lw=0.8, label=f"max force {N_MAX:.0f} N")
    ax[0].axhline(700, color="b", ls=":", lw=1, label="700 N load: no solution")
    ax[0].set(xlabel="stretch x [mm]", ylabel="spring force [N]", title="Softening spring")
    ax[0].legend()

    for load, style in [(300, "o-"), (650, "s-")]:
        _, hist, _ = newton_spring(load)
        r = np.array([abs(h[2]) for h in hist])
        ax[1].semilogy(range(len(r)), np.maximum(r, 1e-16), style, label=f"load {load} N")
    ax[1].set(xlabel="Newton iteration", ylabel="|out-of-balance force| [N]",
              title="Quadratic convergence (digits double)")
    ax[1].legend()

    _, snaps = results["previous step"]
    for k in [0, 5, 20, N_STEPS]:
        ax[2].plot(x, snaps[k], label=f"t = {k * DT:.2f}")
    ax[2].set(xlabel="x", ylabel="phase field φ", title="Allen–Cahn: phases separate into ±1")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "newton.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
