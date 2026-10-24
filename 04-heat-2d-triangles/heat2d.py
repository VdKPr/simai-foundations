"""
Module 04: 2D steady heat conduction with triangles (scikit-fem).

Part 1 - check the method: a square plate with a known exact answer.
         Compare linear (P1) and quadratic (P2) triangles under mesh refinement.

Part 2 - a real-looking problem: a 40 mm x 10 mm aluminium heat spreader.
         A hot component on the top edge pushes heat in, the bottom edge sits
         on a cold plate, the side edges are insulated. Solve it twice:
           (a) cold plate as a fixed temperature   (Dirichlet)
           (b) cold plate as convection, h(T - T_inf)  (Robin)
         and check the energy balance: heat in = heat out.

Run:  python heat2d.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skfem import (Basis, BilinearForm, ElementTriP1, ElementTriP2, FacetBasis,
                   Functional, LinearForm, MeshTri, condense, solve)
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot

OUT = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# The weak forms. Compare with module 02: integral of k grad(T) . grad(v) = loads
# ---------------------------------------------------------------------------
@BilinearForm
def conduction(u, v, w):
    return w.k * dot(grad(u), grad(v))


@BilinearForm
def robin_lhs(u, v, w):          # convection: the h*T*v part moves to the left side
    return w.h * u * v


@LinearForm
def robin_rhs(v, w):             # ... and the h*T_inf*v part stays on the right
    return w.h * w.T_inf * v


@LinearForm
def flux_in(v, w):               # applied heat flux (Neumann)
    return w.q * v


# ---------------------------------------------------------------------------
# Part 1: unit square, -lap(u) = 2 pi^2 sin(pi x) sin(pi y), u = 0 on edges
# exact u = sin(pi x) sin(pi y)
# ---------------------------------------------------------------------------
def exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


@LinearForm
def source(v, w):
    x, y = w.x
    return 2 * np.pi**2 * exact(x, y) * v


@Functional
def l2_err(w):
    x, y = w.x
    return (w.uh - exact(x, y)) ** 2


@Functional
def grad_err(w):                 # error in the gradient = error in heat flux
    x, y = w.x
    ex = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    ey = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return (w.uh.grad[0] - ex) ** 2 + (w.uh.grad[1] - ey) ** 2


def solve_square(n_refine, element):
    mesh = MeshTri().refined(n_refine)
    basis = Basis(mesh, element, intorder=6)
    K = conduction.assemble(basis, k=1.0)
    F = source.assemble(basis)
    u = solve(*condense(K, F, D=basis.get_dofs()))   # u = 0 on the whole boundary
    uh = basis.interpolate(u)
    e0 = np.sqrt(l2_err.assemble(basis, uh=uh))
    e1 = np.sqrt(grad_err.assemble(basis, uh=uh))
    h = 1.0 / 2**n_refine
    return h, basis.N, e0, e1


def part1():
    print("PART 1: square plate with a known answer")
    print(f"{'elem':>4} {'h':>8} {'unknowns':>9} {'temp error':>11} {'rate':>5} {'flux error':>11} {'rate':>5}")
    for name, el in [("P1", ElementTriP1()), ("P2", ElementTriP2())]:
        prev = None
        for n in range(1, 7):
            h, ndof, e0, e1 = solve_square(n, el)
            if prev is None:
                print(f"{name:>4} {h:8.4f} {ndof:9d} {e0:11.3e} {'':>5} {e1:11.3e} {'':>5}")
            else:
                r0 = np.log2(prev[0] / e0)
                r1 = np.log2(prev[1] / e1)
                print(f"{name:>4} {h:8.4f} {ndof:9d} {e0:11.3e} {r0:5.2f} {e1:11.3e} {r1:5.2f}")
            prev = (e0, e1)
    print("P1: temperature error ~ h^2, flux error ~ h.")
    print("P2: one order better in both, for more unknowns per element.\n")


# ---------------------------------------------------------------------------
# Part 2: heat spreader
# ---------------------------------------------------------------------------
W, H = 0.040, 0.010        # m    plate width and thickness (2D cross-section)
K_AL = 200.0               # W/m.K  aluminium
CHIP_W = 0.010             # m    width of the hot component, centred on the top edge
Q_CHIP = 1.0e5             # W/m^2  heat flux from the component
H_CONV = 2000.0            # W/m^2.K  cold-plate heat transfer coefficient
T_INF = 25.0               # degC coolant temperature


def spreader_mesh(nx=80, ny=20):
    mesh = MeshTri.init_tensor(np.linspace(0, W, nx + 1), np.linspace(0, H, ny + 1))
    return mesh.with_boundaries({
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "chip": lambda x: np.isclose(x[1], H) & (np.abs(x[0] - W / 2) < CHIP_W / 2),
    })


@Functional
def heat_out(w):
    return w.h * (w.T - w.T_inf)


def part2():
    print("PART 2: aluminium heat spreader with a hot component")
    mesh = spreader_mesh()
    basis = Basis(mesh, ElementTriP1())
    chip = FacetBasis(mesh, ElementTriP1(), facets=mesh.boundaries["chip"])
    bottom = FacetBasis(mesh, ElementTriP1(), facets=mesh.boundaries["bottom"])

    K = conduction.assemble(basis, k=K_AL)
    F = flux_in.assemble(chip, q=Q_CHIP)          # Neumann: just added to the load vector
    # Insulated side edges: zero flux, so nothing to add. That's the natural BC.

    # (a) Dirichlet: bottom edge held at exactly T_INF
    T_a = np.zeros(basis.N)
    D = basis.get_dofs("bottom")
    T_a[D] = T_INF
    T_a = solve(*condense(K, F, x=T_a, D=D))

    # (b) Robin: bottom edge loses heat by convection
    K_b = K + robin_lhs.assemble(bottom, h=H_CONV)
    F_b = F + robin_rhs.assemble(bottom, h=H_CONV, T_inf=T_INF)
    T_b = solve(K_b, F_b)

    q_in = Q_CHIP * CHIP_W
    q_out = heat_out.assemble(bottom, h=H_CONV, T=bottom.interpolate(T_b), T_inf=T_INF)
    print(f"  heat in from component : {q_in:8.1f} W per metre depth")
    print(f"  heat out to cold plate : {q_out:8.1f} W per metre depth   (Robin case)")
    print(f"  max temperature (a) bottom fixed at {T_INF:.0f} degC : {T_a.max():6.1f} degC")
    print(f"  max temperature (b) convection, h = {H_CONV:.0f}      : {T_b.max():6.1f} degC")
    print("  Same part, same load: the boundary condition you choose moves the answer a lot.\n")

    fig, ax = plt.subplots(2, 1, figsize=(9, 5))
    for a, T, title in [(ax[0], T_a, "(a) bottom edge fixed at 25 °C (Dirichlet)"),
                        (ax[1], T_b, f"(b) bottom edge convection, h = {H_CONV:.0f} W/m²K (Robin)")]:
        plot(mesh, T, ax=a, shading="gouraud", colorbar=True, vmin=T_INF, vmax=T_b.max(),
             cmap="inferno")
        a.set_title(title)
        a.set_aspect("equal")
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
    fig.suptitle("Temperature [°C]; heat enters at the top centre")
    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    part1()
    fig = part2()
    path = OUT / "heat2d.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
