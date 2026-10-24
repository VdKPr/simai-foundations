"""
Module 13: physics-informed neural networks (PINNs), in plain PyTorch.

Problem: steady heat conduction in a rod, made dimensionless (length 1, k = 1):
    -T''(x) = q(x),   T(0) = T(1) = 0

Experiments:
  1. Forward problem, smooth answer:   T = sin(pi x)
  2. Forward problem with a wiggle:    T = sin(pi x) + 0.3 sin(8 pi x)
     (a) boundary-condition penalty weight 1     -> watch it fail
     (b) boundary-condition penalty weight 1000  -> works, but learns slowly
  3. Hard boundary conditions (built into the network) on problem 1
  4. Inverse problem: unknown conductivity k, found from 6 noisy temperature readings
  5. Cost: PINN training time vs a finite-difference solve of the same problem

Run:  python pinn.py      (CPU is fine; about 1-2 minutes)
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).parent / "out"
PI = np.pi


def make_net(width=32, depth=3, seed=0):
    torch.manual_seed(seed)
    layers, n_in = [], 1
    for _ in range(depth):
        layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
        n_in = width
    layers.append(torch.nn.Linear(n_in, 1))
    return torch.nn.Sequential(*layers)


def d_dx(y, x):
    """Derivative of y with respect to x by automatic differentiation."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def train_forward(q_fn, hard_bc=False, iters=4000, bc_weight=1.0, seed=0,
                  T_exact=None, checkpoints=()):
    net = make_net(seed=seed)
    x = torch.linspace(0, 1, 101).reshape(-1, 1).requires_grad_(True)   # collocation points
    xb = torch.tensor([[0.0], [1.0]])                                   # boundary points

    def T(xx):
        out = net(xx)
        return xx * (1 - xx) * out if hard_bc else out                   # hard BC: zero at both ends by construction

    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    q = q_fn(x.detach())
    xs = np.linspace(0, 1, 501)
    history = {}
    for it in range(1, iters + 1):
        opt.zero_grad()
        Tx = T(x)
        residual = -d_dx(d_dx(Tx, x), x) - q                             # how badly the PDE is violated
        loss = (residual**2).mean()
        if not hard_bc:
            loss = loss + bc_weight * (T(xb) ** 2).mean()                # soft BC: a penalty in the loss
        loss.backward()
        opt.step()
        if it in checkpoints:
            pred = T(torch.as_tensor(xs).reshape(-1, 1)).detach().numpy().ravel()
            history[it] = rel_l2(pred, T_exact(xs))
    predict = lambda xx: T(torch.as_tensor(xx).reshape(-1, 1)).detach().numpy().ravel()
    return predict, history


def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


def fd_solve(q_fn, n=101):
    """Module 01's finite-difference solve, for comparison."""
    x = np.linspace(0, 1, n)
    h = x[1] - x[0]
    m = n - 2
    A = (np.diag(2 * np.ones(m)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)) / h**2
    T = np.zeros(n)
    T[1:-1] = np.linalg.solve(A, q_fn(torch.tensor(x[1:-1])).numpy())
    return x, T


def main():
    OUT.mkdir(exist_ok=True)
    xs = np.linspace(0, 1, 501)

    smooth_T = lambda x: np.sin(PI * x)
    smooth_q = lambda x: PI**2 * torch.sin(PI * x)
    wiggle_T = lambda x: np.sin(PI * x) + 0.3 * np.sin(8 * PI * x)
    wiggle_q = lambda x: PI**2 * torch.sin(PI * x) + 0.3 * (8 * PI) ** 2 * torch.sin(8 * PI * x)

    # ---- 1 & 2: forward problems -------------------------------------------
    print("1-2) Forward problems, same network, 4000 Adam steps (relative error at steps 500/1000/4000)")
    cps = (500, 1000, 4000)
    runs = [("smooth, BC weight 1", smooth_q, smooth_T, 1.0),
            ("wiggle, BC weight 1", wiggle_q, wiggle_T, 1.0),
            ("wiggle, BC weight 1000", wiggle_q, wiggle_T, 1000.0)]
    results = {}
    for name, q_fn, T_fn, wgt in runs:
        t0 = time.perf_counter()
        pinn, hist = train_forward(q_fn, bc_weight=wgt, T_exact=T_fn, checkpoints=cps)
        t_train = time.perf_counter() - t0
        results[name] = (pinn, T_fn)
        errs = "  ".join(f"{100 * hist[c]:9.3f}%" for c in cps)
        ends = pinn(np.array([0.0, 1.0]))
        print(f"   {name:>24}: {errs}   T(0) = {ends[0]:+6.2f}, T(1) = {ends[1]:+6.2f}   ({t_train:4.1f} s)")
    print("   Wiggle with weight 1: the ends sit at +5 and -5 instead of 0. The wiggle's curvature")
    print("   makes the PDE part of the loss start near 2e4, so a boundary penalty of 25 barely")
    print("   registers. The network drifts by a straight line, which has zero curvature, so")
    print("   the PDE residual can't see it.")
    print("   Weight 1000 fixes it, but the wiggly answer takes far longer to learn than the")
    print("   smooth one: sharp and high-frequency features are slow for neural networks.\n")

    # ---- 3: hard boundary conditions ---------------------------------------
    print("3) Hard boundary conditions on the smooth problem: T = x (1 - x) * network(x)")
    pinn, _ = train_forward(smooth_q, hard_bc=True)
    print(f"   relative error {100 * rel_l2(pinn(xs), smooth_T(xs)):7.3f}%, "
          f"T(0) = {pinn(np.array([0.0]))[0]:+.1e} exactly by construction")
    print("   No penalty weight to tune; the loss has only the PDE left to satisfy.\n")

    # ---- 4: inverse problem ---------------------------------------------------
    print("4) Inverse problem: find the conductivity k from 6 noisy temperature readings")
    k_true = 2.5
    rng = np.random.default_rng(1)
    x_obs = np.array([0.1, 0.25, 0.4, 0.6, 0.75, 0.9])
    T_obs = smooth_T(x_obs) / k_true * (1 + 0.01 * rng.standard_normal(6))   # -k T'' = pi^2 sin(pi x)
    net = make_net(seed=0)
    log_k = torch.zeros(1, requires_grad=True)                                 # start from k = 1
    x = torch.linspace(0, 1, 101).reshape(-1, 1).requires_grad_(True)
    xo, To = torch.tensor(x_obs).reshape(-1, 1), torch.tensor(T_obs).reshape(-1, 1)
    opt = torch.optim.Adam(list(net.parameters()) + [log_k], lr=2e-3)
    T = lambda xx: xx * (1 - xx) * net(xx)
    for _ in range(6000):
        opt.zero_grad()
        k = torch.exp(log_k)                                                   # keeps k positive
        res = -k * d_dx(d_dx(T(x), x), x) - smooth_q(x.detach())
        loss = (res**2).mean() + 100 * ((T(xo) - To) ** 2).mean()              # physics + data
        loss.backward()
        opt.step()
    k_found = float(torch.exp(log_k).detach())
    print(f"   true k = {k_true}, found k = {k_found:.3f} ({100 * abs(k_found - k_true) / k_true:.2f}% off)")
    print("   This is where PINNs earn their keep: physics + sparse data -> an unknown parameter.\n")

    # ---- 5: cost comparison ---------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(100):
        x_fd, T_fd = fd_solve(smooth_q)
    t_fd = (time.perf_counter() - t0) / 100
    print("5) Cost of the forward problem")
    print(f"   finite differences, 101 points: {1000 * t_fd:.2f} ms, error "
          f"{100 * rel_l2(T_fd, smooth_T(x_fd)):.3f}%")
    print("   PINN: several seconds of training for a similar or worse error.")
    print("   For a plain forward solve with known inputs, classical solvers win easily.\n")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (name, (pinn, T_fn)) in zip(ax, results.items()):
        a.plot(xs, T_fn(xs), "k-", lw=2, label="exact")
        a.plot(xs, pinn(xs), "--", label="PINN, 4000 steps")
        a.set(xlabel="x", ylabel="T", title=name)
        a.legend()
    fig.tight_layout()
    path = OUT / "pinn.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
