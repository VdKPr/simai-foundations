# 13 · Physics-Informed Neural Networks (PINNs)

**Time:** two evenings · **Code:** `pinn.py` (needs PyTorch; CPU is fine, about 1–2 minutes) · **Unlocks:** Physics-Informed Neural Networks (PINNs); Must Know "PINN vs neural operator"

## Why you care

PINNs are the most famous idea in scientific ML and appear in nearly every AI-for-simulation interview. You need two things:

1. A clear picture of how they work.
2. An honest view of where they're useful and where they aren't. Interviewers at simulation companies are skeptical of hype, and showing that you know the limits is a strong signal.

You also used one in your thesis comparison (the vanilla MLP gave fewer Newton iterations than the PINN).

## The idea

A PINN is a neural network T(x) whose training loss is the **PDE residual**: how badly the network's output violates the equation, measured at many sample points called **collocation points**.

```math
\text{loss} = \underbrace{\frac{1}{N}\sum_i \big(-T''(x_i) - q(x_i)\big)^2}_{\text{physics}} \;+\; \lambda \underbrace{\big(T(0)^2 + T(1)^2\big)}_{\text{boundary conditions}} \;\Big[+\; \underbrace{\sum_j (T(x_j) - T_j^{\text{measured}})^2}_{\text{data, if any}}\Big]
```

- **T″ comes from automatic differentiation (autodiff):** PyTorch can differentiate the network's output with respect to its input exactly, as many times as needed. No mesh, no finite differences.
- **No training data is needed** for a forward problem. The equation itself is the supervision.
- **Boundary conditions** are either a **soft** penalty (the λ term) or **hard**, built into the network's form so they always hold, for example T(x) = x(1 − x)·network(x).

Compare with module 02: a PINN minimizes the **strong-form** residual at points. FEM satisfies the **weak form** within a set of shape functions.

## Results from the script

A dimensionless rod with −T″ = q and both ends at 0. Same network (3 layers of 32 neurons), 4,000 Adam steps.

| Case | Error at 500 steps | at 1,000 | at 4,000 | T(0), T(1) |
|---|---|---|---|---|
| Smooth (T = sin πx), BC weight 1 | 0.094% | 0.070% | **0.006%** | 0.00, 0.00 |
| Wiggle (+ 0.3 sin 8πx), BC weight 1 | 502% | 445% | **398%** | **+5.09, −5.09** |
| Wiggle, BC weight 1000 | 235% | 159% | **0.046%** | 0.00, 0.00 |

### Lesson 1: loss balance is the number-one practical problem

With the wiggle and BC weight 1, training "succeeds" (the loss goes down) but the answer is wrong by 400%: the ends sit at ±5 instead of 0.

Why: the wiggle's curvature makes the PDE part of the loss start around 2 × 10⁴, while a boundary error of 5 costs a penalty of only 25. The optimizer happily ignores the boundary. The network adds a straight line to the right answer, and since a straight line has zero curvature, the PDE residual can't see it.

Raising the BC weight to 1000 fixes it. In real problems, with several loss terms in different units, choosing these weights is the main headache. That's why the literature has adaptive weighting schemes, and why **nondimensionalizing** the equations (making every term "about 1") comes first.

### Lesson 2: sharp and high-frequency features are slow

Even with good weighting, at 1,000 steps the smooth case is at 0.07% while the wiggle case is still 159% off. Neural networks fit smooth, low-frequency structure first and fine detail later (spectral bias, module 01). Real fields with sharp layers, shocks or multiple scales are much harder for PINNs than smooth textbook cases.

### Lesson 3: hard boundary conditions help

T = x(1 − x)·network(x) gives 0.001% error on the smooth problem, with T(0) = 0 exactly and no weight to tune.

### Lesson 4: inverse problems are where PINNs earn their keep

**Inverse problem:** the conductivity k is unknown. We have six temperature readings with 1% noise. The PINN learns T(x) and k together, with loss = physics + data.

**Result: true k = 2.5, found k = 2.499 (0.03% off).**

A classical solver would need an outer optimization loop around many forward solves. The PINN does it in one training run, and it naturally mixes sparse, noisy measurements with the physics. Parameter identification, data assimilation and fields reconstructed from a few sensors are its real niche.

### Lesson 5: for plain forward problems, classical solvers win

| Method | Time | Error |
|---|---|---|
| Finite differences, 101 points (module 01) | 0.3 ms | 0.008% |
| PINN, 4,000 steps | ~12 s | 0.006% |

Roughly 40,000× slower for similar accuracy. This gap grows in 2D and 3D, and a trained PINN solves only *one* problem: change the load, retrain. If you want fast predictions for *many* problems, you need a surrogate or a neural operator (module 14).

## Run it

```bat
pip install torch        (or the CUDA command from pytorch.org)
python pinn.py
```

### What to look for

1. The middle row's T(0) and T(1) of ±5 while the loss is "small", and the middle plot showing a wiggly curve tilted by a straight line.
2. The 1,000-step column: smooth is nearly done, wiggle is nowhere near.
3. The inverse problem landing on k = 2.499.

## Exercises

1. In the inverse problem, use only 2 measurement points, `x_obs = np.array([0.3, 0.7])` (and 2 noise values). Is k still found? Why might it be?
2. Set the noise in the inverse problem to 10% (`0.10 * rng.standard_normal(6)`). How far off is k now?
3. Your manager read that "PINNs replace FEM". Write your two-sentence reply.

<details>
<summary>Answers</summary>

1. Yes: k = 2.485, 0.6% off. Here the physics fixes the *shape* of T completely (the source is known and the boundary values are 0), so even two readings pin down its *size*, which sets k. In real problems where a whole field is unknown, not one number, sparse data is much less forgiving.
2. Only 0.2% off (k = 2.495) in this run. The physics forces one fixed shape, so the six noisy readings effectively get averaged, and their errors partly cancel. That's the physics acting as a **regularizer**: it stops the model from chasing the noise. With a less constrained problem, 10% noise would hurt much more.
3. "PINNs are useful for inverse problems and for combining sparse measurements with physics, but for standard forward simulations they are much slower and less reliable than FEM, and they must be retrained for every new load or geometry. For fast predictions across many designs, surrogates and neural operators trained on FEM data are the relevant tools."

</details>

## Where this connects

- **PINN vs neural operator** (Must Know question): a PINN learns **one solution** of one problem, trained by the equation. A neural operator learns the **map from inputs to solutions** for a whole family of problems, trained on data. Module 14.
- **Hard vs soft constraints** apply to every physics-ML model. The guide has a full answer.
- **Your thesis:** the MLP warm start beat the PINN warm start on Newton iterations. Lesson 1 is one plausible reason: PINN training balances competing terms, while a plain regression fit to good solver data doesn't have to.

## Check yourself

1. What is in a PINN's loss function?
2. How does a PINN get second derivatives of its output?
3. Soft vs hard boundary conditions: what's the difference?
4. Why did the wiggle case with BC weight 1 fail, even though the loss went down?
5. Name the strongest and the weakest use case for PINNs.

<details>
<summary>Answers</summary>

1. The squared PDE residual at collocation points, plus boundary (and initial) condition terms, plus optionally a data term.
2. Automatic differentiation through the network with respect to its input.
3. Soft: a penalty term in the loss, so the BC holds only approximately and needs a weight. Hard: built into the network's form, so it holds exactly.
4. The PDE term was ~1000× larger than the BC term, so the optimizer reduced the PDE residual and ignored the boundaries. The extra straight line has zero curvature and is invisible to the PDE term.
5. Strongest: inverse problems and combining sparse data with physics. Weakest: plain forward solves, where classical solvers are far faster and more reliable.

</details>

## Words introduced here

PINN · collocation points · PDE residual · automatic differentiation · soft / hard boundary conditions · loss balancing / weighting · nondimensionalization · spectral bias · inverse problem · parameter identification · data assimilation

## Now read in the guide

- `answers/pinns.md`
- `answers/pinn-training-failures.md`: you've now seen one first-hand
- `answers/pinn-training-recipe.md`
- `answers/hard-vs-soft-constraints.md`
- `answers/automatic-differentiation.md`

**Next:** [14 · Neural operators](../14-neural-operators/)
