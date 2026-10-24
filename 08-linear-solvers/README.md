# 08 · Linear Solvers: Direct, Iterative, Preconditioned

**Time:** one evening · **Code:** `solvers.py` (needs `pyamg`) · **Unlocks:** Simulation Fundamentals (sparse linear solvers)

## Why you care

Every module so far ended in "solve K u = F". For a real 3D model with millions of unknowns, that solve is most of the run time and most of the memory. It's also the step every Newton iteration and every implicit time step repeats.

The Ansys Mechanical solver settings expose this choice directly: the **sparse direct** solver vs the **PCG (preconditioned conjugate gradient)** iterative solver. Knowing when each wins is a standard question, and it tells you where ML acceleration can realistically help.

## Two families

### Direct solvers: exact, memory-hungry

A direct solver factors K into triangular pieces, K = L·U (**LU factorization**; for symmetric K, the Cholesky variant), then solves by substitution. It gives the answer to machine precision in one go, and once K is factored you can reuse the factors for many load cases cheaply.

The catch is **fill-in**. K is sparse, but its factors L and U are not: zeros inside K become non-zeros in L and U. The factors can need 10× to 100×+ the memory of K itself, and fill-in is much worse in 3D than in 2D.

### Iterative solvers: light, but count matters

An iterative solver starts from a guess and improves it step by step. The standard one for symmetric positive matrices (heat conduction, linear elasticity) is **conjugate gradient (CG)**. Each iteration needs only one multiplication K × vector, so memory stays at the size of K.

The catch is the iteration count. It depends on the **condition number**: the ratio of K's largest to smallest eigenvalue. Roughly, it's the ratio of the stiffest to the floppiest way the model can deform. For FEM meshes, the condition number grows like 1/h², so CG iterations grow like 1/h: **refine the mesh, and every solve gets slower per unknown.**

### Preconditioning: change the problem so it's easy

A **preconditioner** is a cheap approximate inverse of K. CG is applied to the preconditioned system, which has a much smaller condition number.

The best general-purpose preconditioner for these problems is **multigrid**. It removes smooth, large-scale error on coarse versions of the mesh, where that error is cheap to fix, and jagged, small-scale error on the fine mesh. **Algebraic multigrid (AMG)** builds the coarse levels automatically from the matrix itself. Its key property: **the iteration count barely depends on mesh size.**

## Results from the script

### 2D (heat conduction on a square grid)

| Grid | Unknowns | Non-zero % | Condition no. | LU time | Fill-in | CG iterations | CG + AMG iterations |
|---|---|---|---|---|---|---|---|
| 32 × 32 | 1,024 | 0.48% | 441 | 0.003 s | 7.5× | 59 | 7 |
| 64 × 64 | 4,096 | 0.12% | 1,712 | 0.015 s | 10.9× | 119 | 9 |
| 128 × 128 | 16,384 | 0.03% | 6,744 | 0.08 s | 15.0× | 239 | 9 |
| 256 × 256 | 65,536 | 0.008% | 26,768 | 0.5 s | 19.1× | 470 | 11 |

### 3D (the same equation on a cube)

| Grid | Unknowns | LU time | Fill-in | CG time | CG iterations | CG + AMG iterations |
|---|---|---|---|---|---|---|
| 16³ | 4,096 | 0.09 s | 45× | 0.002 s | 39 | 7 |
| 24³ | 13,824 | 1.3 s | 90× | 0.016 s | 59 | 8 |
| 32³ | 32,768 | 7.9 s | **144×** | 0.044 s | 79 | 9 |

Your timings will differ; the trends won't.

What the tables show:

1. **Sparsity:** at 65k unknowns, only 0.008% of the matrix is non-zero. Storing the zeros would need 34 GB; storing the non-zeros needs a few MB.
2. **Condition number × 4, CG iterations × 2** each time the grid doubles in 2D.
3. **AMG keeps CG iterations at 7–11** no matter the size. That's why multigrid is the go-to for large elliptic problems.
4. **3D is where direct solvers hurt.** Fill-in goes 45× → 144× and LU time goes up ~90× for 8× more unknowns. At 32³, plain CG is ~180× faster than LU. In 2D, LU is competitive.

**Rule of thumb:** direct solvers for small-to-medium models, 2D models, many load cases on one matrix, or badly conditioned problems (thin shells, nearly incompressible materials, contact). Iterative + good preconditioner for large 3D solid models, especially when memory is the limit.

### What makes a good initial guess for CG

Same 256 × 256 system, three starting points:

| Initial guess | Pointwise error | Initial residual (relative) | CG iterations |
|---|---|---|---|
| zero | 100% | 1.0 | 686 |
| answer + 1% smooth error | 1% | 0.074 | **401** |
| answer + 1% random noise | 1% | **135** | **724** |

(686 differs from 470 in the table above because this problem has a different right-hand side.)

The noisy guess is 100× closer to the answer than zero, yet needs **more** iterations. K amplifies jagged error enormously (jagged error means large strain energy), so a guess with small pointwise error can still have a huge residual. **Judge a guess by its residual, not by how close it looks.** That's the "residual-aware" part of RANG's name.

## Run it

```bat
pip install pyamg
python solvers.py
```

It takes 10–20 seconds; most of that is the 32³ LU factorization.

### What to look for

1. In both tables, the CG column doubling and the AMG column hardly moving.
2. The 3D LU time and fill-in exploding.
3. The last table: the "1% random noise" row beating zero on pointwise error and losing on iterations.

## Exercises

1. Add `48` to the 3D sizes in `main()`: `table(3, [16, 24, 32, 48])`. Before running, predict the CG iteration count and whether LU will finish in under a minute. (It might take a while, or run out of memory. That's the point.)
2. In `run_cg`, tighten `TOL` from `1e-8` to `1e-12`. How do the CG and AMG iteration counts change? What does that say about the cost of tight tolerances?
3. Which solver would you pick for (a) a 2D axisymmetric pressure vessel with 50 load cases, (b) a 5-million-node 3D engine block, linear static?

<details>
<summary>Answers</summary>

1. CG: 119 iterations in about 0.2 s (AMG: 10). LU at 48³ (110,592 unknowns), measured while writing this module: 138 s, 307× fill-in, about 5 GB of peak memory. Only 3.4× more unknowns than 32³, but 17× the time. That's exactly why big 3D models use iterative solvers.
2. Plain CG: 119 → 144 at 64 × 64 and 470 → 577 at 256 × 256, about 20–25% more. AMG: 9 → 12 and 11 → 15. Four extra orders of magnitude cost much less than the first eight did, but they're not free, and they're cheapest with a good preconditioner. Tolerance is a cost knob: don't ask for 10⁻¹² when the mesh's own discretization error is 10⁻³.
3. (a) Direct: small 2D model, factor once and reuse the factors for all 50 load cases. (b) Iterative with a strong preconditioner (PCG or AMG): a direct solver's fill-in would need too much memory.

</details>

## Where this connects to ML

- **ML can propose initial guesses or preconditioners**, but the guess must have a small *residual*, not just a small pointwise error. A surrogate's prediction often has small-scale jitter, so it may need smoothing first, and a residual check decides whether to use it at all.
- **Speed-up claims need fair baselines.** "1000× faster than the solver" against plain CG on a fine mesh is very different from 1000× faster than AMG. Ask which solver and preconditioner the baseline used (a guide question).
- **The solve is not always the bottleneck.** Meshing, assembly and post-processing also take time. Know which part a proposed ML method actually replaces.

## Check yourself

1. What is fill-in, and why does it matter more in 3D?
2. What does the condition number measure, and how does it change with mesh refinement?
3. What does a preconditioner do?
4. What is special about multigrid's iteration count?
5. Why can a guess with 1% pointwise error be worse than a zero guess?

<details>
<summary>Answers</summary>

1. Zeros in K that become non-zeros in the LU factors. In 3D each node has more neighbours and the matrix bandwidth is much larger, so far more fill-in happens and memory explodes.
2. The ratio of the largest to smallest eigenvalue (stiffest vs floppiest deformation mode). It grows like 1/h² under refinement, so iterative solvers need more iterations.
3. It acts as a cheap approximate inverse of K, turning the system into one with a small condition number that CG solves in few iterations.
4. It stays roughly constant as the mesh is refined.
5. The residual is K × error, and K amplifies jagged error. A noisy guess has a huge residual, and iterative solvers have to reduce the residual.

</details>

## Words introduced here

LU factorization · Cholesky · fill-in · iterative solver · conjugate gradient (CG) · condition number · preconditioner · PCG · multigrid · algebraic multigrid (AMG) · sparse direct solver · tolerance

## Now read in the guide

- `answers/sparse-linear-solvers.md`
- `answers/ml-vs-numerical-solvers.md`: *Questions to ask about a 1000x speed-up claim*

**Next:** [09 · CFD on one page](../09-cfd-one-page/)
