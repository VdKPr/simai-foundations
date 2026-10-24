"""
Module 06: time stepping - explicit vs implicit, stability, stiffness.

A 100 mm steel rod. The middle 20 mm starts 100 degC hotter than the rest;
both ends are held at 0 degC (temperature rise above ambient). Heat diffuses:

    dT/dt = alpha * d2T/dx2

Space is discretized with finite differences (module 01), giving dT/dt = -A T.
We march in time three ways and compare with a reference that is exact in time
(so every error you see comes from the time stepping alone).

Run:  python transient_heat.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- problem data -------------------------------------------------------------
L = 0.1                       # m
ALPHA = 50.0 / (7850 * 490)   # m^2/s  thermal diffusivity of steel = k / (rho c)
T_END = 60.0                  # s
OUT = Path(__file__).parent / "out"


def setup(n):
    """n gaps -> n-1 interior unknowns. Returns grid, matrix A and initial field."""
    x = np.linspace(0, L, n + 1)[1:-1]
    h = L / n
    m = n - 1
    A = (np.diag(2.0 * np.ones(m)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1))
    A *= ALPHA / h**2
    T0 = np.where(np.abs(x - L / 2) <= 0.01, 100.0, 0.0)     # hot middle section
    return x, h, A, T0


def reference(A, T0, t):
    """Exact-in-time solution of dT/dt = -A T, via the eigenvectors of A."""
    lam, V = np.linalg.eigh(A)
    return V @ (np.exp(-lam * t) * (V.T @ T0))


def march(A, T0, dt, n_steps, method):
    """Advance n_steps with the chosen method. Returns the final field and max|T| history."""
    I = np.eye(len(T0))
    if method == "explicit":            # T_new = T + dt * (-A T)         uses only known values
        step = I - dt * A
    elif method == "implicit":          # T_new = T + dt * (-A T_new)     solve for T_new
        step = np.linalg.inv(I + dt * A)
    elif method == "crank-nicolson":    # average of the two
        step = np.linalg.solve(I + 0.5 * dt * A, I - 0.5 * dt * A)
    T = T0.copy()
    history = [np.abs(T).max()]
    for _ in range(n_steps):
        T = step @ T
        history.append(np.abs(T).max())
    return T, np.array(history)


def main():
    OUT.mkdir(exist_ok=True)
    n = 50
    x, h, A, T0 = setup(n)
    dt_crit = h**2 / (2 * ALPHA)
    T_ref = reference(A, T0, T_END)
    print(f"Rod: {n} gaps of {1000 * h:.1f} mm, diffusivity {ALPHA:.2e} m^2/s")
    print(f"Explicit stability limit: dt < h^2 / (2 alpha) = {dt_crit:.3f} s\n")

    cases = [("explicit", 0.9), ("explicit", 1.05),
             ("implicit", 0.9), ("implicit", 10), ("implicit", 100),
             ("crank-nicolson", 10), ("crank-nicolson", 100)]
    print(f"{'method':>15} {'dt / dt_crit':>12} {'dt [s]':>8} {'steps':>6} {'max error [degC]':>17}")
    runs = {}
    for method, factor in cases:
        n_steps = int(np.ceil(T_END / (factor * dt_crit)))
        dt = T_END / n_steps                         # land exactly on T_END
        T, hist = march(A, T0, dt, n_steps, method)
        err = np.abs(T - T_ref).max()
        runs[method, factor] = (T, hist, dt)
        label = f"{err:17.3e}" if err < 1e6 else f"{'BLEW UP':>17}"
        print(f"{method:>15} {dt / dt_crit:12.2f} {dt:8.3f} {n_steps:6d} {label}")
    print("\nExplicit: cheap steps, but above the limit the error explodes.")
    print("Implicit: stable at any dt; big steps lose accuracy, not stability.\n")

    print("Why fine meshes make explicit methods expensive (stiffness):")
    print(f"{'gaps':>6} {'h [mm]':>7} {'dt limit [s]':>13} {'explicit steps to 60 s':>23}")
    for nn in [50, 100, 200, 400]:
        hh = L / nn
        dtc = hh**2 / (2 * ALPHA)
        print(f"{nn:6d} {1000 * hh:7.2f} {dtc:13.4f} {int(np.ceil(T_END / dtc)):23d}")
    print("Halving h cuts the allowed dt by 4x, even though the physics is no faster.\n")

    # ---- plots ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xx = 1000 * x
    ax[0].plot(xx, T0, color="0.7", label="t = 0")
    ax[0].plot(xx, T_ref, "k-", lw=2, label="reference, t = 60 s")
    ax[0].plot(xx, runs["explicit", 0.9][0], "--", label="explicit, dt = 0.9 limit")
    ax[0].plot(xx, runs["implicit", 100][0], "o", ms=3, label="implicit, dt = 100x limit")
    ax[0].plot(xx, runs["crank-nicolson", 100][0], "-", lw=1, label="Crank-Nicolson, dt = 100x limit")
    ax[0].set(xlabel="x [mm]", ylabel="temperature rise [°C]", title="Profiles at t = 60 s")
    ax[0].legend(fontsize=8)

    for key, style in [(("explicit", 0.9), "-"), (("explicit", 1.05), "-")]:
        T, hist, dt = runs[key]
        ax[1].semilogy(np.arange(len(hist)) * dt, hist, style,
                       label=f"explicit, dt = {key[1]} x limit")
    ax[1].set(xlabel="time [s]", ylabel="max |T| [°C]  (log scale)",
              title="Just above the limit, explicit blows up")
    ax[1].legend()
    fig.tight_layout()
    path = OUT / "transient_heat.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
