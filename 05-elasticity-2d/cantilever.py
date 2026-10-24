"""
Module 05: 2D linear elasticity - a cantilever in plane stress (scikit-fem).

A steel cantilever, 1 m long and 0.1 m deep, clamped on the left edge.
A downward shear load P (per metre of depth) is spread over the right edge.

We compare against beam theory:
    tip deflection   d = P L^3 / (3 E I)  +  P L / (kappa G A)   (bending + shear)
    bending stress   sigma = M y / I,  with M = P (L - x)

Three experiments:
  1. Tip deflection vs mesh for linear (P1) and quadratic (P2) triangles.
  2. Bending stress at x = X_PROBE (a smooth region) vs mesh: converges.
  3. Peak von Mises stress at the clamped corner vs mesh: keeps rising.

Run:  python cantilever.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skfem import (Basis, ElementTriDG, ElementTriP1, ElementTriP2, ElementVector,
                   FacetBasis, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import dot, sym_grad
from skfem.models.elasticity import linear_elasticity, linear_stress, plane_stress

# ---- problem data ------------------------------------------------------------
L, H = 1.0, 0.1            # m   length and depth
E, NU = 200e9, 0.3         # Pa, -   steel
P = 1.0e5                  # N per metre of depth, total downward tip load
X_PROBE = L / 2            # m   where to read the top-fibre bending stress

LAM, MU = plane_stress(E, NU)
I = H**3 / 12              # second moment of area per metre depth
G = MU
KAPPA = 5 / 6              # shear correction factor for a rectangle

OUT = Path(__file__).parent / "out"


def beam_theory_deflection():
    bending = P * L**3 / (3 * E * I)
    shear = P * L / (KAPPA * G * H)
    return bending, shear


def beam_theory_stress(x, y):
    """Bending stress sigma_xx at (x, y); y measured from the bottom edge."""
    M = P * (L - x)                    # hogging moment: top fibre in tension
    return M * (y - H / 2) / I


@LinearForm
def tip_traction(v, w):
    return dot(np.array([0.0, -P / H]), v)      # uniform shear over the right edge


def solve_cantilever(ny, scalar_element):
    nx = 10 * ny                                 # keep elements roughly square
    mesh = MeshTri.init_tensor(np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1))
    element = ElementVector(scalar_element)      # two unknowns per node: (u_x, u_y)
    basis = Basis(mesh, element, intorder=4)

    K = asm(linear_elasticity(LAM, MU), basis)
    right = FacetBasis(mesh, element, facets=mesh.facets_satisfying(lambda x: np.isclose(x[0], L)))
    F = asm(tip_traction, right)

    clamped = basis.get_dofs(lambda x: np.isclose(x[0], 0.0))   # u_x = u_y = 0 on the left
    u = solve(*condense(K, F, D=clamped))

    # Tip deflection: vertical displacement at the middle of the right edge
    tip_v = (basis.probes(np.array([[L], [H / 2]])) @ u)[1]     # returns [u_x, u_y]

    # Stresses: computed from displacement gradients, then smoothed per element
    dg = basis.with_element(ElementTriDG(ElementTriP1()))
    sigma = linear_stress(LAM, MU)(sym_grad(basis.interpolate(u)))
    sxx = dg.project(sigma[0, 0])
    syy = dg.project(sigma[1, 1])
    sxy = dg.project(sigma[0, 1])
    vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)   # plane-stress von Mises

    mid_top = dg.probes(np.array([[X_PROBE], [H * (1 - 1e-9)]])) @ sxx
    return dict(mesh=mesh, basis=basis, u=u, tip=-tip_v, sxx_mid=float(mid_top[0]),
                vm=vm, vm_max=vm.max(), dg=dg, ndof=basis.N)


def main():
    OUT.mkdir(exist_ok=True)
    bend, shear = beam_theory_deflection()
    d_beam = bend + shear
    s_beam = beam_theory_stress(X_PROBE, H)
    print(f"Beam theory tip deflection: {1000 * d_beam:.4f} mm "
          f"(bending {1000 * bend:.4f} + shear {1000 * shear:.4f})")
    print(f"Beam theory bending stress at x = {X_PROBE:.2f} m, top fibre: {s_beam / 1e6:.2f} MPa\n")

    print(f"{'elem':>4} {'mesh':>8} {'unknowns':>9} {'tip [mm]':>9} {'% of beam':>9} "
          f"{'probe stress [MPa]':>22} {'peak von Mises [MPa]':>21}")
    results = {}
    for name, el in [("P1", ElementTriP1()), ("P2", ElementTriP2())]:
        for ny in [1, 2, 4, 8, 16]:
            r = solve_cantilever(ny, el)
            results[name, ny] = r
            print(f"{name:>4} {10 * ny:>4}x{ny:<3} {r['ndof']:9d} {1000 * r['tip']:9.4f} "
                  f"{100 * r['tip'] / d_beam:8.1f}% {r['sxx_mid'] / 1e6:22.2f} {r['vm_max'] / 1e6:21.1f}")
        print()
    print("P1 triangles are far too stiff on coarse meshes (locking); P2 is close at once.")
    print("Probe stress converges to beam theory. The peak at the clamped corner keeps")
    print("rising with refinement: a singularity, not a real stress.\n")

    # Plot the P2 16x160 solution: deformed shape coloured by von Mises stress
    r = results["P2", 16]
    mesh, basis, u = r["mesh"], r["basis"], r["u"]
    disp = basis.probes(mesh.p) @ u                 # [all u_x, then all u_y] at the nodes
    n = mesh.p.shape[1]
    ux, uy = disp[:n], disp[n:]
    vm_nodes = r["dg"].probes(mesh.p) @ r["vm"]
    scale = 0.1 * L / np.abs(uy).max()              # exaggerate so the bending is visible

    fig, ax = plt.subplots(figsize=(11, 3.2))
    tpc = ax.tripcolor(mesh.p[0] + scale * ux, mesh.p[1] + scale * uy, mesh.t.T,
                       vm_nodes / 1e6, shading="gouraud", cmap="inferno")
    ax.plot([0, L, L, 0, 0], [0, 0, H, H, 0], "k--", lw=0.8, label="undeformed")
    fig.colorbar(tpc, ax=ax, label="von Mises stress [MPa]")
    ax.set(aspect="equal", xlabel="x [m]", ylabel="y [m]",
           title=f"Cantilever, P2 triangles, displacement magnified {scale:.0f}x")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = OUT / "cantilever.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
