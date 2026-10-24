"""
Module 14: a neural operator (Fourier Neural Operator, FNO) in plain PyTorch.

Operator to learn: stiffness profile -> displacement field, for a bar.
A bar of length 1 is fixed at x = 0 and carries a uniform axial load. Its
stiffness EA(x) varies randomly along its length (think of a cast part with
porosity, or a tapered, layered or graded beam). The exact displacement is

    u(x) = integral from 0 to x of (1 - s) / EA(s) ds

So each sample is a whole input FUNCTION EA(x) and a whole output function u(x).

We train on 64-point grids and compare:
  - an MLP: 64 numbers in, 64 numbers out (tied to that grid)
  - an FNO: learns in terms of Fourier modes, so it can be evaluated on any grid
Then both are tested on new bars at 64 points and at 256 points.

Run:  python fno.py      (about 1-3 minutes on CPU)
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
FINE = 1024                     # reference grid: data is computed here, then subsampled


# ---- data ------------------------------------------------------------------------
def make_bars(n_samples, seed, amplitude=0.6, n_modes=8):
    """Random smooth stiffness profiles EA(x) = exp(random cosine series), and exact u(x)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, FINE + 1)
    j = np.arange(1, n_modes + 1)
    a = rng.standard_normal((n_samples, n_modes)) * amplitude / j
    EA = np.exp(a @ np.cos(np.pi * np.outer(j, x)))
    integrand = (1 - x) / EA
    u = np.concatenate([np.zeros((n_samples, 1)),
                        np.cumsum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) / FINE, axis=1)], axis=1)
    return x, EA, u


def on_grid(x, EA, u, n):
    """Subsample the fine data to an n-point grid (n must divide FINE)."""
    idx = np.arange(0, FINE + 1, FINE // n)[:n]
    return [torch.tensor(arr[..., idx], dtype=torch.float32) for arr in (x, EA, u)]


# ---- models ------------------------------------------------------------------------
class SpectralConv1d(nn.Module):
    """The FNO's key layer: FFT -> keep the lowest `modes` frequencies -> multiply by
    learned complex weights -> inverse FFT. Works on any grid size."""

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (width * width)
        self.w = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, v):                       # v: (batch, width, n)
        v_hat = torch.fft.rfft(v)
        out = torch.zeros_like(v_hat)
        out[:, :, :self.modes] = torch.einsum("bin,ion->bon", v_hat[:, :, :self.modes], self.w)
        return torch.fft.irfft(out, n=v.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, width=32, modes=12, layers=4):
        super().__init__()
        self.lift = nn.Linear(2, width)         # input channels: EA(x) and x itself
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, EA, x):                   # EA, x: (batch, n)
        v = self.lift(torch.stack([torch.log(EA), x.expand_as(EA)], dim=-1)).permute(0, 2, 1)
        for i, (s, l) in enumerate(zip(self.spectral, self.local)):
            v = s(v) + l(v)                     # global (Fourier) + local (pointwise) paths
            if i < len(self.spectral) - 1:
                v = nn.functional.gelu(v)
        return self.project(v.permute(0, 2, 1)).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n))

    def forward(self, EA, x):
        return self.net(torch.log(EA))


def rel_l2(pred, true):
    return torch.linalg.norm(pred - true, dim=-1) / torch.linalg.norm(true, dim=-1)


def train(model, EA, x, u, epochs=150, batch=25):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    scale = u.abs().max()
    for _ in range(epochs):
        perm = torch.randperm(EA.shape[0])
        for i in range(0, EA.shape[0], batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = rel_l2(model(EA[b], x) * scale, u[b]).mean()
            loss.backward()
            opt.step()
        sched.step()
    return lambda E, xx: model(E, xx).detach() * scale


def resample(arr, n_new):
    """Linear interpolation of each row onto an n_new-point grid (same spacing rule)."""
    n_old = arr.shape[-1]
    x_old = np.arange(n_old) / n_old
    x_new = np.arange(n_new) / n_new
    return torch.tensor(np.array([np.interp(x_new, x_old, r) for r in arr.numpy()]), dtype=torch.float32)


def main():
    OUT.mkdir(exist_ok=True)
    x_f, EA_tr_f, u_tr_f = make_bars(1000, seed=0)
    _, EA_te_f, u_te_f = make_bars(200, seed=1)
    x64, EA_tr, u_tr = on_grid(x_f, EA_tr_f, u_tr_f, 64)
    _, EA_te64, u_te64 = on_grid(x_f, EA_te_f, u_te_f, 64)
    x256, EA_te256, u_te256 = on_grid(x_f, EA_te_f, u_te_f, 256)
    print(f"Training: 1000 bars on a 64-point grid. Test: 200 new bars, at 64 and at 256 points.")
    print(f"Stiffness varies up to {EA_tr_f.max() / EA_tr_f.min():.0f}x across the training set\n")

    models = {}
    for name, model in [("MLP", MLP()), ("FNO", FNO1d())]:
        n_par = sum(p.numel() for p in model.parameters())
        t0 = time.perf_counter()
        models[name] = train(model, EA_tr, x64, u_tr)
        print(f"{name}: {n_par:,} parameters, trained in {time.perf_counter() - t0:.0f} s")

    print(f"\n{'':>26} {'64-point test':>14} {'256-point test':>15}")
    mlp, fno = models["MLP"], models["FNO"]
    e_mlp64 = rel_l2(mlp(EA_te64, x64), u_te64)
    u_mlp256 = resample(mlp(resample(EA_te256, 64), x64), 256)   # MLP can only work on 64 points
    e_mlp256 = rel_l2(u_mlp256, u_te256)
    e_fno64 = rel_l2(fno(EA_te64, x64), u_te64)
    e_fno256 = rel_l2(fno(EA_te256, x256), u_te256)
    print(f"{'MLP (+ interpolation)':>26} {100 * e_mlp64.median():13.2f}% {100 * e_mlp256.median():14.2f}%"
          "   median relative L2 error")
    print(f"{'FNO (run directly)':>26} {100 * e_fno64.median():13.2f}% {100 * e_fno256.median():14.2f}%")
    print(f"{'MLP worst case':>26} {100 * e_mlp64.max():13.2f}% {100 * e_mlp256.max():14.2f}%")
    print(f"{'FNO worst case':>26} {100 * e_fno64.max():13.2f}% {100 * e_fno256.max():14.2f}%")
    print("\nAt the training resolution the FNO is about twice as accurate, with a third of the")
    print("parameters. The MLP is tied to its 64 inputs: to use it at 256 points we squeeze the")
    print("input down and stretch the output back up. The FNO runs on the finer grid directly,")
    print("but its error grows there: 'works on any grid' is approximate, so test it.\n")

    t0 = time.perf_counter()
    with torch.no_grad():
        fno(EA_te256, x256)
    print(f"FNO prediction for 200 bars at 256 points: {1000 * (time.perf_counter() - t0):.1f} ms in total")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, a in enumerate(ax[:2]):
        a.plot(x256, EA_te256[k], "k-")
        a2 = a.twinx()
        a2.plot(x256, 1000 * u_te256[k], "C0-", lw=2, label="exact u")
        a2.plot(x256, 1000 * fno(EA_te256[k:k + 1], x256)[0], "C1--", label="FNO at 256 pts (trained on 64)")
        a.set(xlabel="x", ylabel="stiffness EA(x) (black)", title=f"Test bar {k + 1}")
        a2.set_ylabel("displacement u(x) x 1000")
        a2.legend(loc="lower right", fontsize=8)
    ax[2].hist(100 * e_mlp256.numpy(), bins=30, alpha=0.6, label="MLP + interpolation")
    ax[2].hist(100 * e_fno256.numpy(), bins=30, alpha=0.6, label="FNO")
    ax[2].set(xlabel="relative L2 error [%] at 256 points", ylabel="number of test bars",
              title="Error spread on 200 new bars")
    ax[2].legend()
    fig.tight_layout()
    path = OUT / "fno.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
