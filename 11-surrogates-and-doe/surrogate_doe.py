"""
Module 11: a surrogate model for the heat spreader, and how to choose samples.

The "simulation" is the module 04 heat spreader, solved with scikit-fem.
Four design inputs:
    Q  component heat flux      5e4 - 2e5   W/m^2
    h  cold-plate coefficient   500 - 5000  W/m^2.K   (log scale)
    k  plate conductivity       20 - 250    W/m.K     (log scale: ceramic ... aluminium)
    H  plate thickness          2 - 15      mm
Output: peak temperature rise above the coolant [degC].

Design of experiments (DOE): random sampling vs Latin hypercube.
Surrogates: quadratic response surface (classic) vs Gaussian process (GP).
Each combination is trained at several budgets and scored on 300 test designs.

Run:  python surrogate_doe.py      (about 1-2 minutes: it runs ~1,000 FE solves)
"""

import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from skfem import (Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm, MeshTri, solve)
from skfem.helpers import dot, grad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")      # GP optimizer chatter
OUT = Path(__file__).parent / "out"

W, CHIP_W = 0.040, 0.010
LOW = np.array([5e4, np.log(500), np.log(20), 0.002])
HIGH = np.array([2e5, np.log(5000), np.log(250), 0.015])


# ---- the "expensive simulation" -------------------------------------------
@BilinearForm
def conduction(u, v, w):
    return w.k * dot(grad(u), grad(v))


@BilinearForm
def robin(u, v, w):
    return w.h * u * v


@LinearForm
def flux(v, w):
    return w.q * v


def simulate(Q, h, k, H, nx=60, ny=12):
    """Peak temperature rise of the heat spreader (coolant at 0)."""
    mesh = MeshTri.init_tensor(np.linspace(0, W, nx + 1), np.linspace(0, H, ny + 1))
    mesh = mesh.with_boundaries({
        "bottom": lambda x: np.isclose(x[1], 0.0),
        "chip": lambda x: np.isclose(x[1], H) & (np.abs(x[0] - W / 2) < CHIP_W / 2)})
    e = ElementTriP1()
    basis = Basis(mesh, e)
    K = conduction.assemble(basis, k=k) + robin.assemble(
        FacetBasis(mesh, e, facets=mesh.boundaries["bottom"]), h=h)
    F = flux.assemble(FacetBasis(mesh, e, facets=mesh.boundaries["chip"]), q=Q)
    return solve(K, F).max()


def to_physical(U):
    """Map points in the unit cube [0,1]^4 to physical inputs (log scale for h and k)."""
    Z = LOW + U * (HIGH - LOW)
    return np.column_stack([Z[:, 0], np.exp(Z[:, 1]), np.exp(Z[:, 2]), Z[:, 3]])


def run_sims(U):
    return np.array([simulate(*row) for row in to_physical(U)])


# ---- design of experiments ------------------------------------------------
def doe(name, n, seed):
    if name == "random":
        return np.random.default_rng(seed).random((n, 4))
    return qmc.LatinHypercube(d=4, seed=seed).random(n)


# ---- surrogates -------------------------------------------------------------
def fit_surrogate(name, U, y):
    target = np.log(y)                          # temperature rise spans a wide range
    if name == "quadratic RSM":
        model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(U, target)
    else:
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(4), nu=2.5)
        model = GaussianProcessRegressor(kernel, normalize_y=True, alpha=1e-8,
                                         n_restarts_optimizer=3, random_state=0).fit(U, target)
    return lambda Unew: np.exp(model.predict(Unew))


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    U_test = np.random.default_rng(999).random((300, 4))
    y_test = run_sims(U_test)
    t_sim = (time.perf_counter() - t0) / len(U_test)
    print(f"Test set: 300 designs, peak rise {y_test.min():.1f} to {y_test.max():.1f} degC")
    print(f"One FE solve: {1000 * t_sim:.1f} ms\n")

    budgets = [10, 20, 40, 80]
    seeds = range(5)
    results = {}
    print(f"{'surrogate':>14} {'DOE':>8} " + " ".join(f"{f'n = {b}':>10}" for b in budgets))
    print(f"{'':>14} {'':>8} " + " ".join(f"{'median':>10}" for b in budgets) + "   (relative error, mean of 5 repeats)")
    cache = {}
    for sname in ["quadratic RSM", "GP"]:
        for dname in ["random", "LHS"]:
            meds, p95s = [], []
            for b in budgets:
                m_list, p_list = [], []
                for s in seeds:
                    key = (dname, b, s)
                    if key not in cache:
                        U = doe(dname, b, s)
                        cache[key] = (U, run_sims(U))
                    U, y = cache[key]
                    pred = fit_surrogate(sname, U, y)(U_test)
                    rel = np.abs(pred - y_test) / y_test
                    m_list.append(np.median(rel))
                    p_list.append(np.percentile(rel, 95))
                meds.append(np.mean(m_list))
                p95s.append(np.mean(p_list))
            results[sname, dname] = (meds, p95s)
            print(f"{sname:>14} {dname:>8} " + " ".join(f"{100 * m:9.2f}%" for m in meds))

    print("\n95th-percentile relative error:")
    for (sname, dname), (meds, p95s) in results.items():
        print(f"{sname:>14} {dname:>8} " + " ".join(f"{100 * p:9.2f}%" for p in p95s))

    U, y = cache[("LHS", 80, 0)]
    gp = fit_surrogate("GP", U, y)
    t0 = time.perf_counter()
    gp(U_test)
    t_gp = (time.perf_counter() - t0) / len(U_test)
    print(f"\nGP prediction: {1e6 * t_gp:.0f} microseconds per design, "
          f"about {t_sim / t_gp:.0f}x faster than one FE solve")
    print("(and this FE model is tiny: for a 3D model taking hours, the ratio is enormous)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for (sname, dname), (meds, p95s) in results.items():
        style = "-" if sname == "GP" else "--"
        marker = "o" if dname == "LHS" else "s"
        ax[0].loglog(budgets, 100 * np.array(meds), style + marker, label=f"{sname}, {dname}")
        ax[1].loglog(budgets, 100 * np.array(p95s), style + marker, label=f"{sname}, {dname}")
    ax[0].set(xlabel="number of simulations", ylabel="median relative error [%]", title="Median error")
    ax[1].set(xlabel="number of simulations", ylabel="95th percentile error [%]", title="Bad-case error")
    ax[0].legend()
    fig.tight_layout()
    path = OUT / "surrogate_doe.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
