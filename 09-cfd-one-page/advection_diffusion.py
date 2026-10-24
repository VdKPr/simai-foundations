"""
Module 09: the core difficulty of CFD in one dimension.

Something (heat, dye, momentum) is carried along by a flow of speed u and
spreads by diffusion Gamma. Steady state, phi = 0 at the inlet, phi = 1 at
the outlet:

    u dphi/dx = Gamma d2phi/dx2,     Peclet number Pe = u L / Gamma

When the flow wins (large Pe), all the change squeezes into a thin layer at
the outlet, like a boundary layer at a wall. Exact solution:

    phi(x) = (exp(Pe x / L) - 1) / (exp(Pe) - 1)

We discretize with finite volumes and two choices for the value at a cell face:
  central : average of the two neighbouring cells      (accurate, can wiggle)
  upwind  : value from the cell the flow comes from    (never wiggles, smears)

Run:  python advection_diffusion.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
U = 1.0
PE = 50.0
GAMMA = U * L / PE
OUT = Path(__file__).parent / "out"


def exact(x):
    return np.expm1(PE * x / L) / np.expm1(PE)


def solve_fvm(n_cells, scheme):
    """Cell-centred finite volumes. Each row is a flux balance for one cell."""
    h = L / n_cells
    xc = (np.arange(n_cells) + 0.5) * h
    F = U                  # convective flux strength at every face
    D = GAMMA / h          # diffusive conductance between neighbouring cells
    A = np.zeros((n_cells, n_cells))
    b = np.zeros(n_cells)
    phi_in, phi_out = 0.0, 1.0

    for i in range(n_cells):
        # ---- east face: flux leaving the cell -------------------------------
        if i < n_cells - 1:
            A[i, i] += D
            A[i, i + 1] -= D
            if scheme == "central":
                A[i, i] += F / 2
                A[i, i + 1] += F / 2
            else:                              # upwind, flow goes left to right
                A[i, i] += F
        else:                                  # outlet boundary, half a cell away
            A[i, i] += 2 * D
            b[i] += 2 * D * phi_out
            if scheme == "central":
                b[i] -= F * phi_out            # face value = the boundary value
            else:
                A[i, i] += F                   # face value = the cell the flow leaves
        # ---- west face: flux entering the cell ------------------------------
        if i > 0:
            A[i, i] += D
            A[i, i - 1] -= D
            if scheme == "central":
                A[i, i] -= F / 2
                A[i, i - 1] -= F / 2
            else:
                A[i, i - 1] -= F
        else:                                  # inlet boundary
            A[i, i] += 2 * D
            b[i] += 2 * D * phi_in + F * phi_in
    return xc, np.linalg.solve(A, b)


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Peclet number Pe = {PE:.0f}: the layer at the outlet is about L / Pe = {L / PE:.2f} thick\n")
    print(f"{'cells':>6} {'cell Pe':>8} | {'central: max error':>19} {'min value':>10} |"
          f" {'upwind: max error':>18} {'min value':>10}")
    sols = {}
    for n in [10, 20, 40, 100, 400]:
        row = f"{n:6d} {U * (L / n) / GAMMA:8.2f} |"
        for scheme in ["central", "upwind"]:
            xc, phi = solve_fvm(n, scheme)
            sols[scheme, n] = (xc, phi)
            err = np.abs(phi - exact(xc)).max()
            row += f" {err:19.3e} {phi.min():10.3f} |" if scheme == "central" else f" {err:18.3e} {phi.min():10.3f}"
        print(row)
    print("\nCell Pe above 2: central wiggles (negative values that can't exist physically).")
    print("Upwind never wiggles, but smears the layer: it behaves as if diffusion were")
    print(f"larger by u*h/2. At 20 cells that adds {U * L / 20 / 2 / GAMMA:.2f}x the real diffusion.")
    print("Fix: refine where the gradients are (cell Pe < 2), or use higher-order bounded schemes.\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xf = np.linspace(0, L, 1000)
    for a, n in [(ax[0], 20), (ax[1], 100)]:
        a.plot(xf, exact(xf), "k-", lw=2, label="exact")
        for scheme, style in [("central", "o-"), ("upwind", "s--")]:
            xc, phi = sols[scheme, n]
            a.plot(xc, phi, style, ms=4, label=scheme)
        a.axhline(0, color="0.6", lw=0.5)
        a.set(xlabel="x", ylabel="φ", title=f"{n} cells, cell Péclet = {U * L / n / GAMMA:.1f}")
        a.legend()
    fig.tight_layout()
    path = OUT / "advection_diffusion.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
