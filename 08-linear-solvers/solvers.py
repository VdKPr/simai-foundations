"""
Module 08: how big sparse systems K u = F are actually solved.

Test problem: steady heat conduction on a square (2D) or cube (3D) grid,
the same [-1, 2, -1]-style matrix as module 01, now in more dimensions.

  1. Direct solver (sparse LU): exact, but "fill-in" eats memory, worst in 3D.
  2. Conjugate gradient (CG): iterative, light on memory, but the iteration
     count grows as the mesh is refined.
  3. CG + algebraic multigrid (AMG) preconditioner: iteration count barely grows.
  4. Initial guesses: what makes a guess good for an iterative solver.

Run:  python solvers.py      (needs pyamg: pip install pyamg)
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import pyamg
except ImportError:
    pyamg = None

TOL = 1e-8


def poisson(n, dim):
    """Finite-difference Laplacian on an n^dim grid of interior points (scaled by h^2)."""
    T = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    I = sp.identity(n)
    if dim == 2:
        A = sp.kron(T, I) + sp.kron(I, T)
    else:
        A = sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(I, I), T)
    return A.tocsr()


def run_cg(A, b, M=None, x0=None):
    count = [0]

    def cb(_):
        count[0] += 1

    t = time.perf_counter()
    x, info = spla.cg(A, b, x0=x0, rtol=TOL, maxiter=20000, M=M, callback=cb)
    return x, count[0], time.perf_counter() - t, info


def direct(A, b):
    t = time.perf_counter()
    lu = spla.splu(A.tocsc())
    x = lu.solve(b)
    fill = (lu.L.nnz + lu.U.nnz) / A.nnz
    return x, time.perf_counter() - t, fill


def table(dim, sizes):
    print(f"\n{dim}D grids  (iterations to reach relative residual {TOL:g}; times in seconds)")
    head = (f"{'grid':>9} {'unknowns':>9} {'nonzero %':>9} {'cond. no.':>9} | {'LU time':>8} {'fill-in':>8} |"
            f" {'CG its':>6} {'time':>6}")
    if pyamg:
        head += f" | {'CG+AMG its':>10} {'time':>6}"
    print(head)
    for n in sizes:
        A = poisson(n, dim)
        N = A.shape[0]
        b = np.ones(N)
        x_lu, t_lu, fill = direct(A, b)
        _, it_cg, t_cg, _ = run_cg(A, b)
        # condition number = largest / smallest eigenvalue, known exactly for this matrix
        cond = (1 - np.cos(n * np.pi / (n + 1))) / (1 - np.cos(np.pi / (n + 1)))
        row = (f"{'x'.join([str(n)] * dim):>9} {N:9d} {100 * A.nnz / N**2:9.4f} {cond:9.0f} |"
               f" {t_lu:8.3f} {fill:7.1f}x | {it_cg:6d} {t_cg:6.3f}")
        if pyamg:
            t0 = time.perf_counter()
            ml = pyamg.smoothed_aggregation_solver(A)
            _, it_amg, t_amg, _ = run_cg(A, b, M=ml.aspreconditioner())
            t_amg += time.perf_counter() - t0          # include AMG setup
            row += f" | {it_amg:10d} {t_amg:6.3f}"
        print(row)


def warm_start_demo():
    print("\nInitial guess for CG: same 2D system (256x256), three starting points")
    n = 256
    A = poisson(n, 2)
    g = np.arange(1, n + 1) / (n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    x_true = (16 * X * (1 - X) * Y * (1 - Y) * np.exp(X)).ravel()   # a known smooth answer
    b = A @ x_true

    rng = np.random.default_rng(0)
    modes = np.sin(np.pi * np.outer(np.arange(1, 6), np.arange(1, n + 1)) / (n + 1))
    smooth = (modes.T @ rng.standard_normal((5, 5)) @ modes).ravel()   # gentle, large-scale error
    smooth /= np.abs(smooth).max()
    noise = rng.uniform(-1, 1, x_true.size)                             # point-to-point jitter
    scale = np.abs(x_true).max()

    print(f"{'initial guess':>32} {'pointwise error':>16} {'initial residual':>17} {'CG iterations':>14}")
    for name, x0 in [("zero", np.zeros_like(x_true)),
                     ("answer + 1% smooth error", x_true + 0.01 * scale * smooth),
                     ("answer + 1% random noise", x_true + 0.01 * scale * noise)]:
        err = np.abs(x0 - x_true).max() / scale
        res = np.linalg.norm(b - A @ x0) / np.linalg.norm(b)
        _, it, _, _ = run_cg(A, b, x0=x0)
        print(f"{name:>32} {100 * err:15.1f}% {res:17.2e} {it:14d}")


def main():
    if pyamg is None:
        print("(pyamg not installed: the multigrid column is skipped. pip install pyamg)")
    table(2, [32, 64, 128, 256])
    print("2D: CG iterations double each time the grid doubles. AMG stays nearly flat.")
    table(3, [16, 24, 32])
    print("3D: LU fill-in and time grow much faster than the number of unknowns.")
    warm_start_demo()
    print("The noisy guess is 100x closer to the answer than zero, yet needs MORE iterations:")
    print("the stiffness matrix amplifies jagged error into a huge residual. Judge a guess by")
    print("its residual, not by how close it looks.")


if __name__ == "__main__":
    main()
