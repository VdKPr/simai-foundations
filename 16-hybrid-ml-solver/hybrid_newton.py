"""
Module 16: hybrid ML + solver - the RANG pattern.

The solver: 1D Allen-Cahn with implicit Euler time steps (module 07). Each step is a
nonlinear system solved by Newton to a residual of 1e-10.

The ML part only proposes the STARTING GUESS for Newton. Newton always has the last
word, so the final answer is the same whatever the guess.

Candidate guesses for each step:
  previous      phi_n
  extrapolate   2 phi_n - phi_(n-1)
  ML            a small network that predicts the correction to the extrapolation
                from the local neighbourhood of each grid point
  safe select   compute the residual of all three, start from the smallest

Tests:
  A. new initial conditions, same settings as training   (in distribution)
  B. a time step twice as large as in training             (out of distribution)
  C. interfaces half as wide (EPS halved)                   (out of distribution: new physics)

Run:  python hybrid_newton.py      (about 1-2 minutes)
"""

import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

EPS = 0.03
N = 200
H = 1.0 / (N - 1)
TOL = 1e-10


def laplacian():
    off = np.ones(N - 1)
    L = sp.diags([off, -2 * np.ones(N), off], [-1, 0, 1], format="lil")
    L[0, 1] = 2.0
    L[-1, -2] = 2.0
    return (L / H**2).tocsr()


LAP = laplacian()
I = sp.identity(N, format="csr")


def residual(phi, phi_old, dt):
    return (phi - phi_old) / dt - EPS**2 * (LAP @ phi) + phi**3 - phi


def newton(phi_old, guess, dt, max_iter=50):
    phi = guess.copy()
    for it in range(max_iter):
        R = residual(phi, phi_old, dt)
        if np.abs(R).max() < TOL:
            return phi, it
        J = I / dt - EPS**2 * LAP + sp.diags(3 * phi**2 - 1)
        phi = phi + spla.spsolve(J.tocsc(), -R)
    raise RuntimeError("Newton did not converge")


def initial_condition(rng):
    x = np.linspace(0, 1, N)
    k = rng.integers(2, 6)
    return 0.3 * np.sin(k * np.pi * x + rng.uniform(0, np.pi)) + 0.05 * rng.standard_normal(N)


def trajectory(phi0, dt, n_steps):
    phis = [phi0, phi0]                       # first step: no history, so repeat phi0
    for _ in range(n_steps):
        phi, _ = newton(phis[-1], 2 * phis[-1] - phis[-2], dt)
        phis.append(phi)
    return np.array(phis[1:])


def stencil_features(phi_n, phi_prev, width=2):
    """For each grid point: phi_n and phi_(n-1) at the point and its 2 neighbours each side."""
    pad_n = np.pad(phi_n, width, mode="reflect")
    pad_p = np.pad(phi_prev, width, mode="reflect")
    cols = [pad_n[i:i + N] for i in range(2 * width + 1)] + [pad_p[i:i + N] for i in range(2 * width + 1)]
    return np.column_stack(cols)


def train_guesser(dt, n_traj=20, n_steps=60, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_traj):
        traj = trajectory(initial_condition(rng), dt, n_steps)
        for n in range(1, len(traj) - 1):
            extrap = 2 * traj[n] - traj[n - 1]
            X.append(stencil_features(traj[n], traj[n - 1]))
            y.append(traj[n + 1] - extrap)    # learn only the correction to extrapolation
    X, y = np.vstack(X), np.concatenate(y)
    pick = rng.choice(len(X), size=min(60000, len(X)), replace=False)
    model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=200, random_state=0, tol=1e-8)
    model.fit(X[pick], y[pick] * 1000)        # scale up: corrections are small numbers
    return lambda phi_n, phi_prev: 2 * phi_n - phi_prev + model.predict(stencil_features(phi_n, phi_prev)) / 1000


def evaluate(guesser, dt, n_traj, n_steps, seed):
    rng = np.random.default_rng(seed)
    names = ["previous", "extrapolate", "ML", "safe select"]
    iters = {k: 0 for k in names}
    seconds = {k: 0.0 for k in names}
    picked = {"previous": 0, "extrapolate": 0, "ML": 0}
    for _ in range(n_traj):
        phi0 = initial_condition(rng)
        prev, cur = phi0, phi0
        for _ in range(n_steps):
            t0 = time.perf_counter()
            cands = {"previous": cur, "extrapolate": 2 * cur - prev}
            t_cheap = time.perf_counter() - t0
            t0 = time.perf_counter()
            cands["ML"] = guesser(cur, prev)
            t_ml = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = {k: np.abs(residual(v, cur, dt)).max() for k, v in cands.items()}
            best = min(res, key=res.get)
            t_sel = time.perf_counter() - t0
            nxt = None
            for name in names:
                guess = cands[best] if name == "safe select" else cands[name]
                t0 = time.perf_counter()
                phi, it = newton(cur, guess, dt)
                seconds[name] += time.perf_counter() - t0
                iters[name] += it
                if name == "extrapolate":
                    nxt = phi
            seconds["previous"] += t_cheap
            seconds["extrapolate"] += t_cheap
            seconds["ML"] += t_ml
            seconds["safe select"] += t_cheap + t_ml + t_sel
            picked[best] += 1
            prev, cur = cur, nxt              # all guesses converge to the same answer
    total_steps = n_traj * n_steps
    return iters, seconds, picked, total_steps


def report(title, iters, seconds, picked, steps):
    print(title)
    print(f"{'starting guess':>14} {'Newton iters':>13} {'per step':>9} {'time incl. guess cost':>22}")
    for k in iters:
        print(f"{k:>14} {iters[k]:13d} {iters[k] / steps:9.2f} {seconds[k]:20.2f} s")
    share = ", ".join(f"{k} {100 * v / steps:.0f}%" for k, v in picked.items())
    print(f"   safe selector picked: {share}\n")


def main():
    global EPS
    dt = 0.05
    t0 = time.perf_counter()
    guesser = train_guesser(dt)
    print(f"ML guesser trained on 20 trajectories at dt = {dt} ({time.perf_counter() - t0:.0f} s)\n")

    report("A) New initial conditions, same time step as training (dt = 0.05)",
           *evaluate(guesser, dt, n_traj=5, n_steps=60, seed=100))
    report("B) Time step doubled (dt = 0.10): the ML model was never trained on this",
           *evaluate(guesser, 2 * dt, n_traj=5, n_steps=30, seed=100))
    EPS = 0.015
    report("C) Interfaces half as wide (EPS = 0.015): physics the ML model never saw",
           *evaluate(guesser, dt, n_traj=5, n_steps=60, seed=100))
    EPS = 0.03
    print("Every row reaches the same answer (residual < 1e-10): Newton is the final corrector.")
    print("Extrapolation already needs only ~2 Newton iterations per step, so there is almost")
    print("nothing left for ML to save, and evaluating the network costs more time than it saves.")
    print("The selector costs three residual evaluations per step and stops using the ML guess")
    print("when it stops helping (case C).")


if __name__ == "__main__":
    main()
