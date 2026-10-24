# 04 · 2D Heat Conduction With Triangles

**Time:** two evenings · **Code:** `heat2d.py` (needs `scikit-fem`) · **Unlocks:** Heat Transfer and Multiphysics; Simulation Fundamentals

## Why you care

Module 03 did everything by hand in 1D. Real parts are 2D or 3D, with irregular shapes. Here you move to triangles and let a library do the bookkeeping, so you can concentrate on the physics and the boundary conditions. scikit-fem is the library you used in your thesis.

Thermal simulation is also one of the most common targets for ML surrogates (electronics cooling, battery packs, data centres), because the fields are smooth and the solves are repeated thousands of times in design studies.

## From 1D to 2D: what changes, what doesn't

**What stays the same:** weak form → shape functions → element matrices → assembly → boundary conditions → solve. The whole pipeline from module 03.

**What changes:**

| | 1D (module 03) | 2D (this module) |
|---|---|---|
| Element | line segment, 2 nodes | triangle, 3 nodes |
| Shape function | a "hat" | a "tent": 1 at one corner, 0 at the other two, a flat tilted plane in between |
| Element matrix | 2×2 | 3×3 |
| Derivative | u′ | gradient ∇T (a vector: slope in x and slope in y) |

The 2D weak form for steady heat conduction, with conductivity k and heat flux q applied on part of the edge:

```math
\int_\Omega k\,\nabla T \cdot \nabla v \;dA \;=\; \int_{\text{edge}} q\,v \;ds
```

It's the 1D version with u′v′ replaced by ∇T·∇v.

**Heat flux** is how much heat crosses a unit area per second (W/m²). **Fourier's law** says it flows downhill in temperature: **q = −k ∇T**. So the "flux error" in the script is the error in the gradient of T. It plays the same role as stress in module 03: a derived quantity, one order less accurate.

## How scikit-fem maps onto what you know

You write only the weak form; the library does mesh loops, element matrices and assembly.

| In the code | What it is |
|---|---|
| `@BilinearForm def conduction` | left side of the weak form, gives K |
| `@LinearForm def flux_in` | right side, gives F |
| `Basis(mesh, ElementTriP1())` | which shape functions on which mesh |
| `FacetBasis(..., facets=...)` | the same, but for integrals along an edge (boundary conditions) |
| `condense(K, F, D=...)` | impose known temperatures (essential BC): remove those rows and columns, as in module 03 |
| `solve(...)` | sparse solve of K T = F |

FEniCS, Firedrake and scikit-fem all work this way: you give the weak form, the library runs the pipeline.

## Part 1: linear vs quadratic triangles

A unit square with a known exact solution, solved on finer and finer meshes with two element types:

- **P1 (linear triangle):** 3 nodes at the corners. Temperature is a flat plane inside each triangle.
- **P2 (quadratic triangle):** 6 nodes: corners plus **midside nodes** on each edge. Temperature is a curved surface inside each triangle.

Output from the script:

| Element | Temperature error rate | Flux error rate | Error with 1,089 unknowns |
|---|---|---|---|
| P1 | 2 (error ∝ h²) | 1 (∝ h) | 1.35 × 10⁻³ |
| P2 | 3 (∝ h³) | 2 (∝ h²) | 6.87 × 10⁻⁵ |

For the **same number of unknowns**, P2 is about 20× more accurate here. For smooth problems, higher-order elements are worth their extra cost. That's why Ansys Mechanical uses quadratic elements (with midside nodes) for solid bodies by default. Module 05 shows the other reason: linear triangles in bending are badly "too stiff".

## Part 2: a heat spreader, and all three boundary conditions

A 40 mm × 10 mm aluminium plate (k = 200 W/m·K), as a 2D cross-section:

- **Top centre, 10 mm wide:** a hot component pushes in 10⁵ W/m². **Neumann**: just added to F.
- **Left and right edges:** insulated. **Neumann with zero flux**: nothing to add. The natural BC from module 02 does it for free.
- **Bottom edge:** sits on a cold plate at 25 °C, modelled two ways:
  - **(a) Dirichlet:** bottom fixed at exactly 25 °C (a perfect cold plate).
  - **(b) Robin:** convection, heat out = h(T − 25) with h = 2000 W/m²·K. In the weak form, the h·T·v part goes into K and the h·25·v part goes into F.

Results:

| Bottom-edge model | Max temperature |
|---|---|
| (a) fixed 25 °C | 28.1 °C |
| (b) convection, h = 2000 | 40.9 °C |

**Same part, same heat load, 13 °C apart.** The boundary condition is a modelling decision, and it can matter more than the mesh. A perfect cold plate is optimistic.

**Energy balance check.** The script also integrates the heat leaving through the bottom edge in case (b): 1000.0 W per metre depth, exactly the 1000 W going in. Checking a global balance like this is one of the standard sanity checks before trusting any result.

## Run it

```bat
pip install scikit-fem
python heat2d.py
```

### What to look for

1. **Part 1 rates.** They settle to 2 and 1 for P1, and to 3 and 2 for P2, as in the table.
2. **The energy balance line.** Heat in = heat out.
3. **The two plots,** drawn on the same colour scale. Case (a) is almost uniform blue; case (b) is much hotter, with the hot spot under the component spreading downward.

## Exercises

Predict first, then run. Change the constants at the top of Part 2.

1. Halve the cold-plate coefficient: `H_CONV = 1000`. What happens to the max temperature?
2. Swap aluminium for a ceramic: `K_AL = 25` (roughly alumina). Max temperature?
3. In `part2()`, replace `ElementTriP1()` with `ElementTriP2()` in all three places. Does the max temperature change? What does that tell you about this mesh?

<details>
<summary>Answers</summary>

1. 53.4 °C, up by 12.5 °C. The average bottom-edge temperature rise is total heat / (h × width) = 1000 / (h × 0.04): 12.5 °C at h = 2000, 25 °C at h = 1000. The conduction part inside the plate doesn't change.
2. 63.7 °C. Lower conductivity means a bigger temperature drop inside the plate itself: heat can't spread sideways as easily, and the path down to the cold plate has more resistance. The convection part (12.5 °C) is unchanged.
3. Almost no change: 40.85 °C either way, and a mesh twice as fine gives the same. The P1 mesh was already converged for the max temperature. Checking this (refine or raise the element order, see if the answer moves) is a mesh convergence check.

</details>

## Where this connects to ML

- **Boundary conditions must be inputs to a surrogate.** A surrogate trained only on case (a) data will be confidently wrong on case (b). If the BC can vary in use, the model has to see it as an input.
- **Smooth fields, cheap solves:** this is exactly the kind of problem where a surrogate works well (module 11). The guide's thermal design question walks through one.
- **Energy balance as a test.** An ML prediction can be checked the same way: does heat in match heat out? Physics checks like this are part of evaluating surrogates.

## Check yourself

1. In 2D, what replaces u′v′ in the weak form?
2. What is Fourier's law, and why is flux one order less accurate than temperature?
3. What are midside nodes, and what do they buy you?
4. How does a Robin (convection) condition enter K and F?
5. Why might an engineer's surrogate fail when the customer changes the cooling assumption?

<details>
<summary>Answers</summary>

1. ∇T·∇v, the dot product of the two gradients.
2. q = −k∇T. Flux comes from the gradient (a derivative) of the temperature field, and derivatives converge one order slower.
3. Extra nodes in the middle of each element edge. They make the field quadratic inside the element, giving one order higher accuracy and much better bending behaviour.
4. h·T·v goes into K (it depends on the unknown T); h·T∞·v goes into F (it's known).
5. The BC was fixed in all its training data, so the model never learned how the answer depends on it. That's extrapolation, not interpolation.

</details>

## Words introduced here

triangular element · gradient · heat flux · Fourier's law · bilinear form / linear form · P1 / P2 elements · midside nodes · insulated boundary · convection coefficient h · energy balance check

## Now read in the guide

- `answers/heat-transfer-basics.md`: *Conduction, convection and radiation*
- `answers/elements-and-shape-functions.md`: *Linear vs quadratic elements*, *Element families*
- `answers/boundary-conditions.md`: the whole page now

**Next:** [05 · 2D elasticity, stress and mesh convergence](../05-elasticity-2d/)
