"""
Module 10: ML on physical data - scaling, splits and error metrics.

Dataset: 2,000 steel/aluminium cantilever designs (per metre of width).
Inputs : length L [m], depth H [m], Young's modulus E [Pa], tip load P [N/m]
Output : tip deflection [mm], from Timoshenko beam theory (so we know the truth)

The same small neural network (scikit-learn MLP) is trained three ways:
  A. raw inputs and raw target
  B. inputs and target standardized (zero mean, unit spread)
  C. log of inputs and target, then standardized  (physics is a power law)

Then: random split vs extrapolation split, and which error metric to report.

Run:  python physical_data.py
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # the raw model never converges
RNG = np.random.default_rng(0)
NU = 0.3
KAPPA = 5 / 6


def tip_deflection_mm(L, H, E, P):
    I = H**3 / 12
    G = E / (2 * (1 + NU))
    return 1000 * (P * L**3 / (3 * E * I) + P * L / (KAPPA * G * H))


def make_dataset(n):
    L = RNG.uniform(0.5, 2.0, n)
    H = RNG.uniform(0.05, 0.20, n)
    E = RNG.uniform(70e9, 210e9, n)          # aluminium ... steel
    P = RNG.uniform(1e3, 5e4, n)
    X = np.column_stack([L, H, E, P])
    y = tip_deflection_mm(L, H, E, P)
    return X, y


def relative_errors(y_true, y_pred):
    return np.abs(y_pred - y_true) / np.abs(y_true)


def pct(v):
    return f"{100 * v:9.2f}%" if v < 10 else f"{'>1000%':>10}"


def report(name, y_true, y_pred):
    rel = relative_errors(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"{name:>38} {rmse:9.3f} {pct(np.median(rel))} {pct(np.percentile(rel, 95))} {pct(rel.max())}")
    return rel


def fit_predict(X_tr, y_tr, X_te, mode):
    """Train the same MLP with three different preprocessing choices."""
    net = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, tol=1e-7,
                       random_state=0, learning_rate_init=1e-3)
    if mode == "raw":
        net.fit(X_tr, y_tr)
        return net.predict(X_te)
    if mode == "standardized":
        xs, ys = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr[:, None])
        net.fit(xs.transform(X_tr), ys.transform(y_tr[:, None]).ravel())
        return ys.inverse_transform(net.predict(xs.transform(X_te))[:, None]).ravel()
    if mode == "log":
        xs = StandardScaler().fit(np.log(X_tr))
        ys = StandardScaler().fit(np.log(y_tr)[:, None])
        net.fit(xs.transform(np.log(X_tr)), ys.transform(np.log(y_tr)[:, None]).ravel())
        return np.exp(ys.inverse_transform(net.predict(xs.transform(np.log(X_te)))[:, None]).ravel())


def main():
    X, y = make_dataset(2000)
    print(f"Inputs span: L {X[:, 0].min():.2f}-{X[:, 0].max():.2f} m, "
          f"E {X[:, 2].min():.1e}-{X[:, 2].max():.1e} Pa, P {X[:, 3].min():.0f}-{X[:, 3].max():.0f} N/m")
    print(f"Target span: deflection {y.min():.4f} to {y.max():.1f} mm  "
          f"({y.max() / y.min():.0f}x range)\n")

    # ---- 1. preprocessing -------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    print("1) Same network, three preprocessing choices (random 75/25 split)")
    print(f"{'':>38} {'RMSE [mm]':>9} {'median':>10} {'95th pct':>10} {'worst':>10}")
    print(f"{'':>38} {'':>9} {'rel. error':>10} {'rel. error':>10} {'rel. error':>10}")
    for mode, label in [("raw", "A. raw numbers"), ("standardized", "B. standardized"),
                        ("log", "C. log + standardized")]:
        report(label, y_te, fit_predict(X_tr, y_tr, X_te, mode))
    print("Raw: E is ~1e11 and L is ~1, so the network can't learn. Standardizing fixes the")
    print("scales. Logs turn the power law d ~ P L^3 / (E H^3) into a straight line.\n")

    # ---- 2. random vs extrapolation split --------------------------------
    print("2) Interpolation vs extrapolation, scored on the SAME test beams (L > 1.7 m)")
    test_long = X_te[:, 0] > 1.7
    short_tr = X_tr[:, 0] <= 1.7
    print(f"{'':>38} {'RMSE [mm]':>9} {'median':>10} {'95th pct':>10} {'worst':>10}")
    for mode, label in [("standardized", "B"), ("log", "C")]:
        report(f"{label}, trained on all lengths", y_te[test_long],
               fit_predict(X_tr, y_tr, X_te[test_long], mode))
        report(f"{label}, trained on L <= 1.7 m only", y_te[test_long],
               fit_predict(X_tr[short_tr], y_tr[short_tr], X_te[test_long], mode))
    print("Same test beams; the only change is whether training covered them. RMSE, median")
    print("and 95th-percentile errors all grow when the model has to extrapolate. The log")
    print("model (C) suffers far less: a power law in log space is a straight line, and")
    print("straight lines extrapolate.\n")

    # ---- 3. metrics -------------------------------------------------------
    print("3) What the RMSE hides (pipeline B, random split)")
    pred = fit_predict(X_tr, y_tr, X_te, "standardized")
    rel = relative_errors(y_te, pred)
    small = y_te < np.percentile(y_te, 25)
    big = y_te > np.percentile(y_te, 75)
    print(f"   stiffest quarter of designs  (deflection < {np.percentile(y_te, 25):.3f} mm): "
          f"median relative error {100 * np.median(rel[small]):6.1f}%")
    print(f"   floppiest quarter of designs (deflection > {np.percentile(y_te, 75):.3f} mm): "
          f"median relative error {100 * np.median(rel[big]):6.1f}%")
    print("RMSE in mm is dominated by the big deflections. Report relative errors, and")
    print("percentiles (median, 95th, worst), not just one average.")


if __name__ == "__main__":
    main()
