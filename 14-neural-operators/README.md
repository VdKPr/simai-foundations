# 14 · Neural Operators: FNO and DeepONet

**Time:** two evenings · **Code:** `fno.py` (PyTorch, about 2 minutes on CPU) · **Unlocks:** Neural Operators; Must Know "PINN vs neural operator"

## Why you care

Neural operators are the main architecture behind modern "AI solvers" (NVIDIA PhysicsNeMo, many startup products, a lot of Ansys and Siemens research). The interview questions are always: what does "learning an operator" mean, how does an FNO work, how does it differ from a PINN, and is the "works on any mesh resolution" claim real? You'll be able to answer all four from your own experiment.

## Function in, function out

So far, models mapped a few numbers to a few numbers (module 11), or a few numbers to a field (module 12). A **neural operator** maps a **whole function to a whole function**:

| Input function | Output function | Physics |
|---|---|---|
| stiffness profile EA(x) | displacement u(x) | this module |
| permeability field | pressure field | Darcy flow in rock |
| initial condition | the field at a later time | weather, Navier–Stokes |
| geometry | surface pressure | car aerodynamics |

In maths, a map from functions to functions is called an **operator**. The solution operator of a PDE takes "the inputs that define the problem" to "the solution". A neural operator learns that map from examples, then solves new problems from the same family in milliseconds.

**PINN vs neural operator, finally:**

| | PINN (module 13) | Neural operator |
|---|---|---|
| Learns | one solution, for one set of inputs | the map from inputs to solutions, for a whole family |
| Trained with | the PDE residual (often no data) | input–output pairs from a solver (data) |
| New problem | retrain | one fast forward pass |
| Strength | inverse problems, sparse data | fast predictions across many designs |

## How the FNO works

The **Fourier Neural Operator** stacks layers that each do two things in parallel:

1. **Global path.** Take the FFT of the field (break it into sine and cosine waves), keep only the lowest k frequencies (here 12), multiply each by learned weights, and transform back. This mixes information across the *whole* domain in one step, which suits elliptic physics, where every point feels every other point (module 01).
2. **Local path.** A pointwise linear map at each location.

The two paths are added together and passed through a nonlinearity. Because the learned weights live on *frequencies*, not on grid points, the same trained FNO can be run on a different grid. That's the **discretization-invariance** claim.

**DeepONet**, the other well-known architecture, splits the work in two:

- A **branch network** reads the input function (sampled at fixed sensor points) and outputs coefficients.
- A **trunk network** takes a coordinate x and outputs basis functions.

The prediction is their dot product. It's the module 12 idea of "modes + coefficients", with both parts learned. DeepONet can be queried at any point x, which makes it convenient for scattered output locations.

## The experiment

A bar fixed at x = 0 with a uniform axial load, and a randomly varying stiffness EA(x) (up to 149× variation across the training set). Think of a cast part with porosity, or a graded material. The exact displacement is an integral (module 03 maths), so we have perfect labels.

- **Training:** 1,000 bars on a 64-point grid.
- **Test:** 200 new bars, at 64 points and at 256 points (4× finer than training).
- **Models:** a plain MLP (64 numbers in, 64 out) vs an FNO.

| Model | Parameters | Median error, 64 pts | Median error, 256 pts | Worst, 256 pts |
|---|---|---|---|---|
| MLP (+ interpolation at 256) | 164,672 | 0.31% | 0.31% | 5.2% |
| FNO (run directly) | **55,649** | **0.16%** | 1.19% | 5.9% |

(Relative L2 field errors. The MLP's 256-point result comes from squeezing the input down to 64 points and interpolating the output back up.)

What it shows, honestly:

1. **At the training resolution, the FNO is twice as accurate with a third of the parameters.** Its structure suits the problem: displacement is an integral of the whole stiffness profile, which is a global operation, and that's what the Fourier layer does.
2. **"Runs on any grid" is true, but not free.** The FNO ran on the 4× finer grid without retraining, but its error grew 7×. The plot shows most of it at the two ends of the bar. The FFT assumes the field is periodic (the right end wraps around to the left end), and a bar's displacement isn't. Common fixes are padding the domain before the FFT and training on mixed resolutions. **Never assume resolution transfer works; test it.**
3. **For smooth fields, simple interpolation is a strong baseline.** The MLP + interpolation didn't degrade at all. A fair comparison always includes the boring baseline.

**Speed:** the FNO predicts 200 full displacement fields in about 40 ms.

## Run it

```bat
python fno.py
```

About 2 minutes on CPU (most of it FNO training). With a CUDA build of PyTorch on your 4060, it would be much faster, but the code as written runs on the CPU.

### What to look for

1. The parameter counts and the 64-point column: FNO better with fewer parameters.
2. The 256-point column: the FNO's error growing.
3. **Plots:** the dashed FNO prediction peeling away from the exact curve at the ends of the bar; the error histogram with the FNO's spread at 256 points.

## Exercises

1. Change `modes=12` to `modes=4` in `FNO1d`. What happens to accuracy, and why?
2. Train on 256-point grids instead of 64 (change the `on_grid(..., 64)` calls for training). Does the FNO's 256-point error drop to around its 64-point level? What does that tell you about where its error came from?
3. A startup claims "our neural operator replaces CFD for any car shape". Name three things you'd check.

<details>
<summary>Answers</summary>

1. At 64 points the median error roughly doubles (0.16% → 0.30%): 4 modes can't represent the finer changes in stiffness, and the local path only partly makes up for it. Curiously, at 256 points it gets *better* (1.19% → 0.89%). With fewer modes, there's less room to learn grid-specific detail that doesn't transfer. The number of modes trades accuracy against robustness.
2. Trained at 256 points, the FNO scores 0.29% at 256 points, but 1.03% when moved down to 64. Each model is best at the resolution it was trained on, and transferring in either direction costs several times the error. So the earlier 1.19% came from the resolution change, not from the operator being hard to learn. If you need several resolutions, train on several.
3. Any three of: test on shapes *outside* the training distribution (new body types, not just variations of training cars); accuracy of integrated quantities engineers use (drag coefficient to within a count or two, not just a field error); behaviour in separated-flow regions and near walls (module 09); what CFD settings produced the training data; how many training cases were needed; and an honest timing comparison that includes meshing and data generation.

</details>

## Where this connects

- **Module 12 → 14:** POD + GP is a linear "modes + coefficients" model with a fixed mesh; DeepONet makes both parts neural; FNO works with Fourier modes and any regular grid.
- **Module 15:** FNO needs a regular grid for the FFT. Real FEA and CFD meshes are irregular. Graph networks and geometry-aware operators handle that.
- **Fairness in benchmarks:** the "1000× faster than CFD" question in the guide. Inference is fast; data generation and training are not. Count both.

## Check yourself

1. What does "learning an operator" mean?
2. What happens inside one Fourier layer?
3. Why might an FNO suit elliptic problems?
4. What did the 256-point test reveal about discretization invariance?
5. How does DeepONet split the problem?

<details>
<summary>Answers</summary>

1. Learning a map from whole input functions (fields) to whole output functions, so one trained model solves a whole family of problems.
2. FFT → keep the lowest k frequencies → multiply by learned weights → inverse FFT, plus a pointwise linear path, then a nonlinearity.
3. Elliptic solutions depend on the whole domain at once, and the Fourier path mixes information globally in one layer.
4. The model runs on a new grid without retraining, but accuracy can degrade (here 7×, mostly at the non-periodic ends). It's an approximate property that needs testing.
5. A branch network encodes the input function into coefficients; a trunk network turns output coordinates into basis functions; the output is their dot product.

</details>

## Words introduced here

operator · neural operator · solution operator · Fourier Neural Operator (FNO) · FFT · Fourier modes · discretization invariance · periodicity / padding · DeepONet · branch / trunk network · super-resolution

## Now read in the guide

- `answers/neural-operators.md`, including *PINN vs neural operator*
- The guide section *Neural Operators*: all questions
- `answers/ml-vs-numerical-solvers.md`

**Next:** [15 · Graph networks on meshes](../15-graph-nets-on-meshes/)
