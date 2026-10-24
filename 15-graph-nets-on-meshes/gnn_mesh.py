"""
Module 15: a graph neural network that learns heat diffusion on unstructured meshes.

Ground truth: transient heat conduction on random triangle meshes of the unit square
(edges held at 0), solved with scikit-fem and implicit Euler time steps.

The GNN learns ONE time step: given the temperature at every node, predict the change
over the next step. It only sees each node's neighbours through mesh edges, like an
FEM stencil. Then we "roll it out": feed its own prediction back in, 50 times, on a
mesh it has never seen (with a different number of nodes).

Two GNNs are trained: one on clean inputs, one with small random noise added to its
inputs during training (the MeshGraphNets trick for stable rollouts). Both are tested
on two unseen meshes: one as dense as the training meshes, one about twice as dense.

Run:  python gnn_mesh.py      (about 2-4 minutes on CPU)
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
from scipy.spatial import Delaunay
from skfem import Basis, BilinearForm, ElementTriP1, MeshTri
from skfem.helpers import dot, grad

OUT = Path(__file__).parent / "out"
DT = 0.0005
N_STEPS = 100


# ---- ground-truth simulations ------------------------------------------------------
@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


def random_mesh(n_interior, rng):
    """Random points inside the unit square plus evenly spaced boundary points."""
    k = 14
    s = np.linspace(0, 1, k + 1)[:-1]
    boundary = np.vstack([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    interior = 0.04 + 0.92 * rng.random((n_interior, 2))
    p = np.vstack([boundary, interior])
    tri = Delaunay(p)
    return MeshTri(p.T, tri.simplices.T)


def simulate(mesh, rng, n_steps=N_STEPS):
    """Implicit Euler heat diffusion from 1-3 random hot spots. Returns (n_steps+1, n_nodes)."""
    basis = Basis(mesh, ElementTriP1())
    M, K = mass.assemble(basis), stiffness.assemble(basis)
    bnd = basis.get_dofs().flatten()
    inner = np.setdiff1d(np.arange(basis.N), bnd)
    A = (M + DT * K)[inner][:, inner].tocsc()
    solve_step = spla.factorized(A)
    x, y = mesh.p
    T = np.zeros(basis.N)
    for _ in range(rng.integers(1, 4)):
        cx, cy = 0.2 + 0.6 * rng.random(2)
        T += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.08**2))
    T[bnd] = 0.0
    traj = [T.copy()]
    for _ in range(n_steps):
        rhs = (M @ T)[inner]
        T = np.zeros_like(T)
        T[inner] = solve_step(rhs)
        traj.append(T.copy())
    return np.array(traj), bnd


def graph_of(mesh, bnd):
    """Mesh edges in both directions, with relative position features."""
    e = mesh.facets                                   # each column: the two nodes of one edge
    senders = np.concatenate([e[0], e[1]])
    receivers = np.concatenate([e[1], e[0]])
    d = mesh.p[:, senders] - mesh.p[:, receivers]
    length = np.linalg.norm(d, axis=0)
    edge_feat = np.vstack([d / 0.07, length / 0.07]).T   # scaled by a typical edge length
    is_bnd = np.zeros(mesh.p.shape[1])
    is_bnd[bnd] = 1.0
    return dict(snd=torch.tensor(senders), rcv=torch.tensor(receivers),
                edge=torch.tensor(edge_feat, dtype=torch.float32),
                is_bnd=torch.tensor(is_bnd, dtype=torch.float32))


# ---- the graph network ---------------------------------------------------------------
def mlp(n_in, n_out, hidden=32):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


class MeshGNN(nn.Module):
    def __init__(self, hidden=32, steps=3):
        super().__init__()
        self.encode = mlp(2, hidden)
        self.edge_mlps = nn.ModuleList([mlp(2 * hidden + 4, hidden) for _ in range(steps)])
        self.node_mlps = nn.ModuleList([mlp(2 * hidden, hidden) for _ in range(steps)])
        self.decode = mlp(hidden, 1)

    def forward(self, T, g):
        h = self.encode(torch.stack([T, g["is_bnd"]], dim=1))
        dT = (T[g["snd"]] - T[g["rcv"]]).unsqueeze(1)                  # temperature difference along each edge
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            # message from each sender to its receiver: both ends, the edge geometry, and dT
            msg = edge_mlp(torch.cat([h[g["rcv"]], h[g["snd"]], g["edge"], dT], dim=1))
            agg = torch.zeros_like(h).index_add_(0, g["rcv"], msg)     # sum of incoming messages
            h = h + node_mlp(torch.cat([h, agg], dim=1))               # update every node
        return self.decode(h).squeeze(1)                               # predicted change in T


def merge(graphs):
    """Batch several meshes as one big graph with no edges between them."""
    out, offset = {k: [] for k in ["snd", "rcv", "edge", "is_bnd"]}, 0
    for g in graphs:
        out["snd"].append(g["snd"] + offset)
        out["rcv"].append(g["rcv"] + offset)
        out["edge"].append(g["edge"])
        out["is_bnd"].append(g["is_bnd"])
        offset += len(g["is_bnd"])
    return {k: torch.cat(v) for k, v in out.items()}


def train(samples, scale, noise, epochs=40, batch=16, seed=0):
    torch.manual_seed(seed)
    model = MeshGNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(samples))
        for i in range(0, len(order), batch):
            chunk = [samples[j] for j in order[i:i + batch]]
            g = merge([c[2] for c in chunk])
            T_in = torch.cat([c[0] for c in chunk])
            T_out = torch.cat([c[1] for c in chunk])
            if noise > 0:                                   # train on slightly wrong inputs
                T_in = T_in + noise * torch.randn_like(T_in) * (1 - g["is_bnd"])
            target = (T_out - T_in) / scale
            interior = g["is_bnd"] == 0
            loss = ((model(T_in, g) - target)[interior] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def rollout(model, T0, g, scale, n_steps):
    T, out = T0.clone(), [T0.clone()]
    with torch.no_grad():
        for _ in range(n_steps):
            T = T + scale * model(T, g)
            T = T * (1 - g["is_bnd"])                       # boundary stays at 0 exactly
            out.append(T.clone())
    return torch.stack(out)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    samples = []
    for _ in range(16):                                     # 16 training meshes, 150-250 interior nodes
        mesh = random_mesh(rng.integers(150, 250), rng)
        traj, bnd = simulate(mesh, rng, n_steps=40)
        g = graph_of(mesh, bnd)
        tr = torch.tensor(traj, dtype=torch.float32)
        samples += [(tr[k], tr[k + 1], g) for k in range(0, 40, 2)]
    scale = float(torch.cat([(s[1] - s[0]) for s in samples]).std())
    print(f"Training data: 16 random meshes x 20 single steps = {len(samples)} examples "
          f"({time.perf_counter() - t0:.1f} s to simulate)")

    models = {}
    for name, noise in [("clean inputs", 0.0), ("noisy inputs", 0.003)]:
        t0 = time.perf_counter()
        models[name] = train(samples, scale, noise)
        print(f"GNN trained on {name}: {time.perf_counter() - t0:.0f} s")

    def rel(a, b):
        return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))

    checkpoints = [1, 10, 25, 50, 100]
    tests = {}
    for label, n_int in [("same density", 200), ("2x denser", 400)]:
        test_rng = np.random.default_rng(123)
        test_mesh = random_mesh(n_int, test_rng)
        traj, bnd = simulate(test_mesh, test_rng, n_steps=N_STEPS)
        g = graph_of(test_mesh, bnd)
        truth = torch.tensor(traj, dtype=torch.float32)
        print(f"\nUnseen test mesh, {label}: {test_mesh.p.shape[1]} nodes. Rollout of {N_STEPS} steps "
              f"(training only saw single steps up to step 40)")
        print(f"{'relative error at':>24} " + " ".join(f"{f'step {c}':>9}" for c in checkpoints))
        rolls = {}
        for name, model in models.items():
            r = rollout(model, truth[0], g, scale, N_STEPS)
            rolls[name] = r
            print(f"{'GNN, ' + name:>24} " + " ".join(f"{100 * rel(r[c], truth[c]):8.1f}%" for c in checkpoints))
        frozen = truth[0].expand_as(truth)
        print(f"{'no change (baseline)':>24} " + " ".join(f"{100 * rel(frozen[c], truth[c]):8.1f}%" for c in checkpoints))
        tests[label] = (test_mesh, truth, rolls)
    print("\nErrors compound over the rollout: each step starts from the previous prediction.")
    print("Training on slightly noisy inputs teaches the model to correct its own drift.")
    print("A denser mesh than any in training is out of distribution: everything gets worse.")

    test_mesh, truth, rolls = tests["same density"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    p, t = test_mesh.p, test_mesh.t.T
    vmax = float(truth[25].max())
    for a, field, title in [(ax[0], truth[25], "FEM truth, step 25"),
                            (ax[1], rolls["noisy inputs"][25], "GNN rollout, step 25"),
                            (ax[2], rolls["noisy inputs"][25] - truth[25], "GNN - truth")]:
        kw = dict(vmin=0, vmax=vmax, cmap="inferno") if "GNN -" not in title else dict(cmap="RdBu_r")
        tpc = a.tripcolor(p[0], p[1], t, field.numpy(), shading="gouraud", **kw)
        a.triplot(p[0], p[1], t, lw=0.2, color="w" if "GNN -" not in title else "0.6")
        a.set(aspect="equal", title=title, xticks=[], yticks=[])
        fig.colorbar(tpc, ax=a, shrink=0.8)
    for name, r in rolls.items():
        ax[3].semilogy(range(1, N_STEPS + 1), [100 * rel(r[c], truth[c]) for c in range(1, N_STEPS + 1)],
                       label=f"GNN, {name}")
    ax[3].set(xlabel="rollout step", ylabel="relative error [%]",
              title="Error growth (same-density mesh)")
    ax[3].legend()
    fig.tight_layout()
    path = OUT / "gnn_mesh.png"
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
