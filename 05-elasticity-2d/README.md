# 05 · 2D Elasticity, Stress and Mesh Convergence

**Time:** two evenings · **Code:** `cantilever.py` · **Unlocks:** Structural Mechanics (FEA)

## Why you care

Structural FEA is the core of Ansys Mechanical, Abaqus and Simcenter. Three interview favourites come straight out of this module:

1. Why are linear triangles and tetrahedra "too stiff" in bending?
2. Why does the peak stress at a sharp corner keep rising as you refine the mesh?
3. What's the difference between a stress concentration and a stress singularity?

The third one matters a lot for ML: if you train a surrogate to predict "maximum stress" without understanding singularities, you're training it on a number that depends on the mesh, not the physics.

## Linear elasticity in plain words

Three ingredients, each one you already know:

| Ingredient | Equation in words | 1D version (module 03) |
|---|---|---|
| **Equilibrium** | forces on every tiny piece balance | −(EA u′)′ = f |
| **Strain from displacement** | strain = how much neighbouring points stretch apart (the symmetric part of the displacement gradient) | ε = u′ |
| **Hooke's law** | stress = stiffness × strain, using E and Poisson's ratio ν | σ = E ε |

**Poisson's ratio ν** is how much a material shrinks sideways when you stretch it (0.3 for steel). In 2D, every node now has **two unknowns**, u_x and u_y.

**Plane stress vs plane strain.** A 2D model is a slice of a 3D part. Two choices:

- **Plane stress:** a thin plate, free to thin out, so the out-of-plane stress σ_zz = 0. Used here.
- **Plane strain:** a long body (dam, pipe, tunnel) where the out-of-plane strain ε_zz = 0.

The weak form is virtual work again (module 02), now with stress and virtual strain:

```math
\int_\Omega \boldsymbol{\sigma}(\mathbf u) : \boldsymbol{\varepsilon}(\mathbf v)\;dA \;=\; \int_{\text{edge}} \mathbf t \cdot \mathbf v\;ds
```

Read it as "internal virtual work = external virtual work". **t** is the **traction**: force per unit area applied on the edge.

**Von Mises stress** combines all the stress components into one number you can compare with the yield strength of a ductile metal. For plane stress:

```math
\sigma_{vM} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\sigma_{yy} + \sigma_{yy}^2 + 3\,\tau_{xy}^2}
```

## The problem

A steel cantilever (E = 200 GPa, ν = 0.3), 1 m long and 0.1 m deep, clamped on the left edge. A downward load of 100 kN per metre depth is spread over the right edge.

Beam theory gives reference answers:

- **Tip deflection:** 2.016 mm. That's 2.000 mm from bending plus 0.016 mm from shear (**Timoshenko beam theory** adds the shear part to the classic **Euler–Bernoulli** bending part).
- **Bending stress** at mid-span, top fibre: σ = M·y / I = 30 MPa.

## Results from the script

| Element | Mesh | Unknowns | Tip deflection (% of beam theory) | Stress at mid-span [MPa] | Peak von Mises [MPa] |
|---|---|---|---|---|---|
| P1 | 10 × 1 | 44 | **22.9%** | 6.67 | 16.2 |
| P1 | 40 × 4 | 410 | 82.0% | 24.61 | 48.2 |
| P1 | 160 × 16 | 5,474 | 98.5% | 29.58 | 65.8 |
| P2 | 10 × 1 | 126 | **99.1%** | 28.77 | 58.1 |
| P2 | 40 × 4 | 1,458 | 99.8% | 29.92 | 64.9 |
| P2 | 160 × 16 | 21,186 | 99.8% | 29.99 | **82.9** |

Three lessons, one per column.

### 1. Linear triangles lock in bending

With one row of linear triangles, the beam deflects **less than a quarter** of what it should. A P1 triangle has straight edges and constant strain inside, so it can't take the curved shape of a bending beam. To fake the curvature it has to shear, and shearing is stiff. The element is "too stiff". This is called **shear locking**.

P2 triangles have curved deformation inside each element, so one row already gets 99%.

**In practice:** don't mesh thin, bending-dominated parts with linear tetrahedra. Use quadratic elements, or many linear elements through the thickness.

### 2. Stress in a smooth region converges

At mid-span, far from the clamp and the load, the top-fibre stress converges to the beam theory value of 30 MPa (29.99 MPa on the finest P2 mesh).

### 3. Stress at a sharp corner never converges

The peak von Mises stress sits at the top corner of the clamped edge. It goes 58 → 61 → 65 → 72 → 83 MPa with refinement and **keeps rising**. Beam theory says the root stress is 60 MPa.

This is a **stress singularity**. In the mathematical model, a perfectly rigid clamp meeting a sharp 90° corner gives infinite stress, so a finer mesh just finds more of that infinity. It is an artifact of the idealization, not something the real part feels.

| | Stress concentration | Stress singularity |
|---|---|---|
| Example | a hole in a plate, a fillet (real geometry) | a sharp re-entrant corner, a rigid clamp edge, a point load |
| Value | finite (e.g. about 3× the nominal stress at a hole) | infinite in the model |
| With mesh refinement | converges | keeps rising |
| What to do | trust it once converged | model the real fillet or support stiffness, or read stress a short distance away |

**Saint-Venant's principle:** how a load or support is applied only matters locally. About one beam depth away, the stress settles to the same answer regardless of the details (try exercise 3).

## Run it

```bat
python cantilever.py
```

### What to look for

1. The **% of beam** column for P1 creeping up slowly, and for P2 being right from the start.
2. The **probe stress** column converging to 30 MPa.
3. The **peak von Mises** column climbing without settling.
4. **The plot:** the deformed beam (magnified 50×), with stress highest at the top and bottom fibres near the clamp and zero along the middle (the **neutral axis**).

## Exercises

Change the constants at the top of `cantilever.py`.

1. Double `P`. What happens to the deflection, the probe stress and the peak stress?
2. Make the beam slender: `H = 0.02`. How does P1 on the 40 × 4 mesh do now, compared with 82% before? And P2?
3. Read the stress near the clamp: set `X_PROBE = 0.1` (one beam depth away), then `X_PROBE = 0.05`. Does the probe stress converge? Does it match beam theory?

<details>
<summary>Answers</summary>

1. Everything doubles. The model is linear, so it obeys superposition (module 03, exercise 2).
2. P1 drops to 34% of the beam-theory deflection on the 40 × 4 mesh; P2 still gets 99.9%. Thinner means more bending-dominated, so locking is worse. (The deflection is now 250 mm, far too big for linear theory to be valid. A real analysis would need geometric nonlinearity, module 07.)
3. Yes, both converge. At x = 0.1 m: 53.99 MPa vs 54.0 MPa from beam theory, a match. At x = 0.05 m: 56.68 MPa vs 57.0 MPa, still converged but 0.6% off, because the clamp's local effect hasn't died out yet. That's Saint-Venant's principle in numbers: beam theory is reliable about one depth away from the support.

</details>

## Where this connects to ML

- **Choose targets that converge.** "Max von Mises over the whole part" includes singular points, so its value depends on the mesh. A surrogate trained on it learns mesh noise. Better targets: stress at a defined location away from singularities, stress averaged over a small region, or deflection.
- **Label quality depends on mesh choices.** On the same 10 × 1 mesh, P1 gives 22.9% of the true deflection and P2 gives 99.1%. If half your training data used one element type and half the other, the labels would disagree wildly for identical physics.
- **Safety factor surrogates** (yield strength ÷ peak stress) inherit both problems. The guide has a full answer on this.

## Check yourself

1. What are the three ingredients of linear elasticity?
2. When do you choose plane stress and when plane strain?
3. Why do linear triangles lock in bending, and what are two fixes?
4. How do you tell a stress concentration from a singularity using only the mesh?
5. Why is "maximum von Mises stress" a risky target for a surrogate?

<details>
<summary>Answers</summary>

1. Equilibrium, strain–displacement (kinematics) and Hooke's law (constitutive law).
2. Plane stress for thin plates loaded in their plane (σ_zz = 0). Plane strain for long bodies where the length direction can't deform (ε_zz = 0).
3. They have constant strain and straight edges, so they can't represent curvature without spurious shear, which makes them too stiff. Fixes: quadratic elements, or more linear elements through the thickness.
4. Refine the mesh: a concentration converges to a finite value, a singularity keeps rising.
5. It often sits at a singularity, so its value depends on the mesh, not just the design. The model ends up learning numerical artifacts.

</details>

## Words introduced here

linear elasticity · Poisson's ratio · plane stress / plane strain · traction · von Mises stress · Euler–Bernoulli / Timoshenko beam theory · shear locking · stress concentration · stress singularity · Saint-Venant's principle · neutral axis

## Now read in the guide

- `answers/stress-strain-and-linear-elasticity.md`
- `answers/von-mises-stress-and-yield.md`
- `answers/stress-concentrations-and-singularities.md`
- `answers/elements-and-shape-functions.md`: *Why linear tetrahedra are too stiff in bending*
- `answers/verification-validation-and-mesh-convergence.md`: *Sanity checks before trusting an FEA result*

**Next:** [06 · Time stepping](../06-time-stepping/)
