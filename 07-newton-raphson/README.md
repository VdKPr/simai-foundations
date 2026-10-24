# 07 · Nonlinear Problems and Newton–Raphson

**Time:** two evenings · **Code:** `newton.py` · **Unlocks:** Simulation Fundamentals (Newton–Raphson, "speed up a solver without changing its answer"); Structural Mechanics (sources of nonlinearity)

## Why you care

Plasticity, contact, large deformation, temperature-dependent materials and radiation are all nonlinear, and Newton–Raphson is how almost every FEA and CFD code solves them. "Solution did not converge" is the most common FEA headache, and understanding why is a real skill.

This is also your thesis. RANG is a way to give Newton a better starting guess. This module shows why that helps; module 16 adds the ML part.

## What makes a problem nonlinear

In modules 03–05, the stiffness K was fixed, so one solve of K u = F gave the answer. A problem is **nonlinear** when the stiffness depends on the solution itself:

| Source | Example | Why the stiffness changes |
|---|---|---|
| Material | plasticity, rubber | stress–strain curve isn't a straight line |
| Geometry | large deflection, buckling | the shape changes, so the load paths change |
| Contact | bolted joints, gears | parts touch or separate depending on the deformation |
| Thermal | k(T), radiation (εσT⁴) | properties or boundary heat loss depend on temperature |

## The residual and the Newton step

Write the problem as "out-of-balance force = 0":

```math
R(u) = F_{\text{internal}}(u) - F_{\text{external}} = 0
```

R is the **residual**. When it's zero, the structure is in equilibrium.

**Newton–Raphson** repeats two moves:

1. At the current guess, compute the **tangent stiffness** K_T = dR/du: the slope of the force–displacement curve right now. (For many unknowns, this matrix of derivatives is called the **Jacobian**.)
2. Pretend the problem is linear with that slope, and solve for the correction:

```math
K_T\,\Delta u = -R(u), \qquad u \leftarrow u + \Delta u
```

In one dimension: slide down the tangent line to where it crosses zero, then repeat from there.

## Part 1: one unknown, a softening spring

The spring gets softer as it stretches, and can carry at most **667 N** (at 1 mm stretch). Newton starts from x = 0.

**300 N, 4 iterations:**

| Iteration | Out-of-balance force [N] |
|---|---|
| 0 | 3 × 10² |
| 1 | 9 × 10⁰ |
| 2 | 3 × 10⁻² |
| 3 | 3 × 10⁻⁷ |
| 4 | 0 |

The exponent roughly doubles every step. This is **quadratic convergence**, Newton's signature. It only kicks in close to the answer.

**650 N, 6 iterations.** Close to the peak, the tangent stiffness is nearly zero, so each step overshoots and convergence is slower at first.

**700 N: failed.** The spring can't carry 700 N, so there is no solution. Newton bounces around, sometimes far away, and never settles.

In a real structure, that peak is a **limit point**: collapse, snap-through or buckling. **Sometimes "not converged" is the physics telling you the part fails.** Other times it's numerical: too big a load increment, chattering contact, or a bad guess.

**Remedies engineers use:**

- **Load stepping:** apply the load in increments (Ansys calls them substeps), using each converged result as the next guess.
- **Line search / damping:** take only part of the Newton step when the full step makes things worse.
- **Arc-length method:** follow the force–displacement curve past a limit point, controlling the path length instead of the load.

## Part 2: many unknowns, Allen–Cahn

The 1D **Allen–Cahn equation** is the second-order version of your thesis equation. A **phase field** φ describes two phases (φ = +1 and φ = −1) separated by thin interfaces:

```math
\frac{\partial \varphi}{\partial t} = \varepsilon^2\,\frac{\partial^2 \varphi}{\partial x^2} - (\varphi^3 - \varphi)
```

The φ³ term makes it nonlinear. With implicit Euler (module 06), **every time step is a nonlinear system**: 200 unknowns, a residual vector R and a sparse 200 × 200 Jacobian. Each Newton iteration is one sparse linear solve (module 08).

We start each step's Newton loop from three different guesses:

| Initial guess for each step | Total Newton iterations (60 steps) | Per step |
|---|---|---|
| zero | 224 | 3.73 |
| previous step's answer | 180 | 3.00 |
| linear extrapolation, 2φₙ − φₙ₋₁ | 122 | 2.03 |

All three give the same final answer. **The guess changes how much work Newton does, not what it converges to,** because Newton still drives the residual to 10⁻¹⁰ either way.

That is exactly the RANG idea: a cheap guesser proposes, and Newton is always the final corrector. In your thesis, simple extrapolation (the bottom row) was the baseline that was hard to beat. Module 16 adds an ML guess and the safe-selector logic.

## Run it

```bat
python newton.py
```

### What to look for

1. **Part 1 tables:** the residual exponent doubling at 300 N, the slower start at 650 N, the wandering x values at 700 N.
2. **Part 2 table:** fewer iterations with better guesses, identical answers.
3. **The plots:** the spring curve with the 700 N line above its peak; the residual falling faster and faster on a log scale; φ separating into +1 and −1 regions with sharp interfaces.

## Exercises

1. Solve the 300 N case starting from `x0 = 1.2` instead of 0: `newton_spring(300, x0=1.2)`. Where does it end up? Is that answer physically reachable?
2. Load stepping: reach 650 N in five equal increments (130, 260, …, 650 N), using each converged x as the next starting guess. Count the total iterations. Is it faster than the 6 iterations of the direct attempt? Then what is load stepping for?
3. A frame in Ansys stops converging at 80% of the applied load. Give one physical and one numerical explanation, and what you'd try for each.

<details>
<summary>Answers</summary>

1. It converges to x = 1.556 mm, a different solution, on the falling side of the curve, where the tangent stiffness is negative. Nonlinear problems can have more than one solution, and the starting guess picks which one Newton finds. That branch is unstable, so under a slowly applied load the real spring never gets there.
2. 21 iterations in total (3 + 4 + 4 + 4 + 6), so not faster here. Load stepping is for **robustness**: it keeps each Newton solve close to its answer, which saves cases that fail from a cold start. It's also required for **path-dependent** problems like plasticity, where the result depends on the load history, not just the final load.
3. Physical: the frame really buckles or collapses at that load (a limit point). Check with a linear buckling analysis, or follow the curve with the arc-length method. Numerical: too large a load increment, contact switching on and off, or a poorly conditioned model. Use smaller substeps, line search, or fix the contact settings.

</details>

## Where this connects to ML

- **"Speed up a solver without changing its answer"** is a guide question, and this module is the answer in miniature: ML proposes an initial guess, Newton corrects it, and the convergence tolerance guarantees the final answer.
- **Fair baselines matter.** An ML guess must beat the previous step and extrapolation, not just a zero guess. In the table above, extrapolation already removes 45% of the iterations compared with zero.
- **Iterations aren't the whole cost.** Each iteration costs one Jacobian assembly plus one sparse solve. An ML guess also costs time to evaluate. Measure wall-clock time, not just iteration counts. Your thesis taught you this.

## Check yourself

1. What is the residual, physically?
2. What is the tangent stiffness, and how is it used in a Newton step?
3. What does quadratic convergence look like in a residual table?
4. Name two reasons Newton fails to converge, one physical and one numerical.
5. Why can an ML initial guess never make the final answer worse?

<details>
<summary>Answers</summary>

1. The out-of-balance force: internal force minus applied load. Zero means equilibrium.
2. The derivative of the residual with respect to the unknowns (the slope of the force–displacement curve, or the Jacobian matrix). Newton solves K_T Δu = −R to get the correction.
3. The exponent of the residual roughly doubles each iteration: 10⁻², 10⁻⁴, 10⁻⁸.
4. Physical: no equilibrium exists (a limit point, collapse). Numerical: a bad initial guess, too large a load step, or contact chatter.
5. Because Newton keeps iterating until the residual is below the same tolerance. The guess affects how many iterations it takes, not where it stops. (A very bad guess could make it fail or land on another solution, which is why a safety check on the guess matters.)

</details>

## Words introduced here

nonlinear problem · residual · tangent stiffness · Jacobian · Newton–Raphson · quadratic convergence · limit point · load stepping / substeps · line search · arc-length method · path dependence · Allen–Cahn equation · phase field · warm start

## Now read in the guide

- `answers/newton-raphson-and-nonlinear-solvers.md`, including *Speeding up a solver with ML without changing its answer*
- `answers/nonlinear-structural-analysis.md`: *Three sources of nonlinearity*, *Plasticity and load-history dependence*
- `answers/buckling.md`

**Next:** [08 · Linear solvers](../08-linear-solvers/)
