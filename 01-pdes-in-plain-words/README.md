# 01 · PDEs in Plain Words

**Time:** one evening · **Code:** `fd_1d_heat.py` · **Unlocks:** Simulation Fundamentals

## Why you care

Every simulation Ansys or Siemens sells is solving a PDE. Every ML-for-simulation method does one of three things with that solve: replaces it, speeds it up, or learns from its output. If "PDE" feels foggy, everything after it will feel foggy too.

## The idea

A **PDE (partial differential equation)** is a balance law written at every point of a body: "forces balance" or "heat in = heat out", applied to every tiny piece of material.

The unknown is not a single number. It is a **field**, meaning a value at every point: temperature T(x, y, z), displacement u(x, y, z), pressure p(x, y, z, t).

### Example: a bar under axial load

```math
-\frac{d}{dx}\left(EA\,\frac{du}{dx}\right) = f(x)
```

Read it from the inside out:

| Piece | Meaning |
|---|---|
| du/dx | strain |
| EA · du/dx | internal axial force N |
| d/dx of that | how fast the internal force changes along the bar |
| f | applied load per unit length (N/m) |

So the equation says: *the change in internal force along the bar balances the applied load.* It is the free-body diagram of a thin slice, turned into an equation.

**Same shape, different physics.** For steady heat conduction in a rod, replace EA with k (conductivity), u with T (temperature) and f with q (heat generated per unit volume). The equation is the same with different letters, which is why one solver can do both. The script in this module uses the heat version.

## Boundary conditions

A PDE on its own has infinitely many solutions. **Boundary conditions** (what is known at the edges of the body) pick out the one you want.

| Type | What you fix | Structural example | Thermal example |
|---|---|---|---|
| **Dirichlet** | the value itself | fixed support, u = 0 | wall held at 25 °C |
| **Neumann** | the flux (the derivative) | applied force or pressure | applied heat flux; insulated = zero flux |
| **Robin** | a mix of value and flux | spring support | convection, h(T − T∞) |

Time-dependent problems also need an **initial condition**: the whole field at t = 0.

## Three families of PDEs

| Family | Physics | Behaviour | Examples |
|---|---|---|---|
| **Elliptic** | equilibrium, steady state | every point feels every other point instantly | static stress, steady heat |
| **Parabolic** | diffusion | sharp features smooth out over time | transient heat |
| **Hyperbolic** | waves | disturbances travel at finite speed; sharp fronts survive | vibration, impact, shock waves |

**Why this matters for ML:** neural networks learn smooth functions easily and sharp ones slowly. This tendency is called **spectral bias**. Elliptic and parabolic problems give smooth fields, so surrogates do well on them. Hyperbolic problems, shocks, cracks and contact give sharp fronts, so surrogates struggle there. Interviewers like this point.

## Why we need numerical methods

Exact solutions exist only for simple shapes and loads. Real parts have fillets, holes and contact. So we **discretize**: replace the continuous field with values at a finite number of points. That turns the PDE into a large system of ordinary equations, **K u = F**.

There are three main ways to discretize:

- **FDM (finite difference method):** replace each derivative with differences between neighbouring grid points. Simple, but needs a regular grid. This module's script uses it.
- **FVM (finite volume method):** split the domain into small cells and enforce "flux in = flux out" for each one. Standard in CFD (Ansys Fluent, OpenFOAM).
- **FEM (finite element method):** build the field from simple pieces on elements of any shape, and enforce the balance in an averaged sense (module 02). Standard in structures (Ansys Mechanical, Abaqus).

All three end in the same place: a big, mostly empty matrix equation. The cost of a simulation is mostly the cost of solving that equation, sometimes with millions of unknowns and sometimes thousands of times over (every time step, every design iteration). **That cost is the whole reason ML-for-simulation exists.**

## Run it

```bat
python fd_1d_heat.py
```

The script models a 0.5 m steel rod with both ends at ambient temperature and heat generated inside, strongest in the middle. It solves for the temperature with finite differences and compares the result with the exact answer (a 50.66 °C peak rise).

### What to look for

1. **The printed matrix.** Each row has only three non-zero entries, [−1, 2, −1]: each point only "talks" to its two neighbours. A matrix that is mostly zeros is called **sparse**. Sparsity is what makes million-unknown simulations possible at all.
2. **The error table.** Every time the grid spacing h halves, the error drops by about 4×. Error proportional to h² is called **second-order accuracy**. You'll see the same pattern in FEM in module 03.
3. **The plot.** With only 8 gaps, the dots already sit almost on the exact curve. Smooth problems are easy.

## Common confusions

- **PDE vs ODE.** An ODE (ordinary differential equation) has one independent variable, only x or only t. A PDE has several, such as x, y and t. The 1D bar is technically an ODE; we use it because every idea carries over unchanged to 2D and 3D and it fits on one page.
- **Strong form.** The PDE exactly as written above, required to hold at every point. Module 02 introduces the weak form, which is what FEM actually uses.
- **Grid vs mesh.** A grid is regular rows and columns (FDM). A mesh is any arrangement of elements or cells, and can follow curved geometry (FEM, FVM).

## Check yourself

1. In −(EA u′)′ = f, which part is the internal axial force?
2. An insulated end of a rod: which type of boundary condition, and what value?
3. Transient heating of a plate: elliptic, parabolic or hyperbolic?
4. Why is a crash simulation harder for a neural-network surrogate than a steady thermal simulation?
5. What does a large simulation spend most of its time on?

<details>
<summary>Answers</summary>

1. EA u′ (axial stiffness × strain).
2. Neumann, with zero heat flux: k dT/dx = 0.
3. Parabolic: diffusion over time.
4. Crash is hyperbolic-like, with sharp fronts, contact and plasticity. Neural networks learn smooth fields easily and sharp features slowly (spectral bias), and crash also depends on load history.
5. Solving the large sparse system K u = F, often repeatedly.

</details>

## Words introduced here

PDE · field · boundary condition (Dirichlet, Neumann, Robin) · initial condition · elliptic / parabolic / hyperbolic · spectral bias · discretize · FDM / FVM / FEM · sparse matrix · second-order accuracy · strong form · grid vs mesh

## Now read in the guide

- `answers/solving-pdes-numerically.md`
- `answers/fem-vs-fvm-vs-fdm.md`
- `answers/boundary-conditions.md`, the section *Dirichlet, Neumann and Robin conditions* only

**Next:** [02 · The weak form](../02-weak-form/)
