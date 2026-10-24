# 03 · 1D Bar FEM From Scratch

**Time:** two evenings · **Code:** `bar1d_fem.py` · **Unlocks:** Simulation Fundamentals; Structural Mechanics (FEA)

## Why you care

Every FEA package, from Ansys Mechanical to Abaqus, runs the same pipeline: mesh → element matrices → assembly → boundary conditions → solve → post-process. Here you write all of it in about 40 lines, for a bar small enough to check by hand. After this module, "how is the global stiffness matrix assembled?" is something you can answer from memory, not just recite.

## The problem

A 1 m steel bar (E = 200 GPa, A = 1 cm², so EA = 2 × 10⁷ N):

- **Fixed** at x = 0
- **Uniform axial load** f = 10 kN/m along its length (like the self-weight of a hanging bar)
- **Tip load** P = 5 kN at x = L

```
  |▓
  |▓====================================●──→ P = 5 kN
  |▓  → → → → → → → → → → → → → → → → →
  |▓        f = 10 kN/m along the bar
 fixed                                 x = L
```

From module 02, we solve the weak form ∫ EA u′ v′ dx = ∫ f v dx + P v(L).

The exact answer is known, which is why we use this problem to check the code:

- Displacement: u(x) = f/(EA)·(Lx − x²/2) + Px/EA, giving a tip displacement of 0.5 mm
- Internal force: N(x) = f(L − x) + P, which is 15 kN at the support and 5 kN at the tip

## The pipeline, step by step

Open `bar1d_fem.py` next to this page. Each step below is a few lines in `solve_fem()`.

### 1. Mesh

Split the bar into `n_el` elements of length h = L / n_el. That gives n_el + 1 nodes. The unknowns are the displacements at the nodes.

### 2. Element stiffness: each element is a spring

For one linear element, the two shape functions are straight lines (module 02), and their slopes are ±1/h. Putting them into K_ij = ∫ EA N_i′ N_j′ dx gives

```math
k_e = \frac{EA}{h}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
```

This is exactly the stiffness matrix of a spring with stiffness EA/h. **A linear bar element is a spring.** That's a good thing to say in an interview.

### 3. Element load

The distributed load on one element is f·h. The integral ∫ f N_i dx splits it half to each node: f·h/2 each.

### 4. Assembly

Element e connects nodes e and e + 1. Add its 2×2 block into the global K at rows and columns (e, e + 1). Where two elements share a node, their contributions **add up**, like two springs in series pulling on the same joint.

In code, it is one line inside the loop:

```python
K[np.ix_(nodes, nodes)] += k_e
```

### 5. Boundary conditions

- **Tip load (natural):** add P to the last entry of F. That's all.
- **Support (essential):** u₀ = 0 is known, so remove node 0's row and column and solve for the rest.

### 6. Solve and post-process

Solve K u = F. Then compute the axial force in each element from its two nodal displacements: N = EA·(u_right − u_left)/h. **Strain and stress come from differences of displacements**, so they are one value per element here.

> Note: the script stores K as a full (dense) array because it's tiny. Real codes store only the non-zeros with `scipy.sparse`. That's module 08.

## Run it

```bat
python bar1d_fem.py
```

### What to look for

1. **The global matrix.** It has the same [−1, 2, −1] pattern as the finite-difference matrix in module 01. For this simple problem, FEM and FDM land on the same matrix. They split apart for irregular meshes and real geometry, where FDM can't go.
2. **Nodal displacements.** FEM matches the exact solution at every node, to the last digit. This is a special property of 1D problems like this one. **Don't expect it in 2D or 3D.**
3. **The force plot (middle).** The FE force is a staircase; the exact force is a straight line. Each step crosses the exact line at the element's centre. In this problem, the element-centre value is exactly right. In general, FE stresses are most accurate at points inside the element, which is why commercial codes compute stress at the integration points (**Gauss points**) and then extrapolate or average to the nodes.
4. **The convergence table.** Each time the element size h halves:

   | | Error drops by | Rate | Meaning |
   |---|---|---|---|
   | Displacement | 4× | 2.00 | error ∝ h² |
   | Axial force | 2× | 1.00 | error ∝ h |

   **Rule: derived quantities (strain, stress, heat flux) converge one order slower than the primary field (displacement, temperature).** That's why a mesh that is fine for deflection can still be too coarse for peak stress, and why mesh convergence studies watch stress, not displacement.
5. **No support.** The script tries to solve without the support and numpy reports a singular matrix. The bar can slide along freely (a **rigid-body motion**), so there's no unique answer. In Ansys, an under-constrained model gives the same kind of error.

## Exercises

Predict the answer first, then change the code and run it.

1. Set `n_el = 1` in `main()`. What is the tip displacement? The displacement at mid-length? The force at the support? (Read the last two from the plots, or add `print(u, force)`.)
2. Double `f` and `P`. What happens to every displacement?
3. Build K for 1000 elements and count its non-zero entries: `_, _, _, K = solve_fem(1000)`, then `print(np.count_nonzero(K), K.size)`. What fraction of K is non-zero?

<details>
<summary>Answers</summary>

1. Tip: still exactly 0.5 mm (the nodes are exact). Mid-length: FEM gives 0.25 mm (a straight line from 0 to 0.5), but the exact value is 0.3125 mm. Support force: FEM gives 10 kN, but the exact value is 15 kN, **33% too low**. A coarse mesh underestimates the peak stress, and the peak is exactly where failure starts.
2. Every displacement doubles. The problem is linear, so the response scales with the load (and loads can be added up: superposition). For linear problems, a few solves give you any load case, so an ML surrogate is rarely worth it. ML pays off when the problem is nonlinear or the geometry changes.
3. 3,001 non-zeros out of 1,002,001 entries, about 0.3%. Storing and solving only the non-zeros is what makes large models possible. In 3D the fraction is still tiny, because each node only connects to the nodes of its neighbouring elements.

</details>

## Where this connects to ML

- **A surrogate learns this function.** An FEM solve is a function: (E, A, f, P, L, geometry) → displacement field. A surrogate model learns that mapping from many solves (module 11).
- **Relative L2 error** is the metric the script computes, and it's the standard metric in ML-for-simulation papers (module 10).
- **Assembly looks like message passing.** Each node gathers contributions from its neighbouring elements. Graph neural networks on meshes do the same thing with learned functions (module 15).
- **Training labels have discretization error.** If your training data comes from a coarse mesh, the surrogate learns the coarse-mesh error too. The convergence table shows how big that error can be for stress.

## Check yourself

1. What does a linear bar element's stiffness matrix look like, and what simple mechanical object is it?
2. What happens at a node shared by two elements during assembly?
3. How is the tip load applied? How is the support applied?
4. If displacement error drops 4× when h halves, by how much does the stress error drop?
5. Why does numpy say "singular matrix" when the support is removed?

<details>
<summary>Answers</summary>

1. (EA/h)·[[1, −1], [−1, 1]]: a spring of stiffness EA/h.
2. Both elements' contributions are added into the same rows and columns of K.
3. Tip load: added to the load vector F (natural). Support: that node's row and column are removed, because its displacement is known (essential).
4. About 2×, because stress is a derivative of displacement and converges one order slower.
5. The bar can move as a rigid body with no strain energy, so K has a zero eigenvalue and no inverse.

</details>

## Words introduced here

element stiffness matrix · assembly · global stiffness matrix · Gauss points · mesh convergence · rate of convergence · derived quantities · rigid-body motion · singular matrix · superposition · relative L2 error

## Now read in the guide

- `answers/finite-element-method.md`: *The FEM pipeline*, *Element matrices*, *Assembly and sparsity*, *Applying boundary conditions*. Compare its 1D code with yours.
- `answers/elements-and-shape-functions.md`: *How the error falls with mesh size and element order*
- `answers/verification-validation-and-mesh-convergence.md`: *Mesh convergence and the grid convergence index*

**Next:** [04 · 2D heat conduction with triangles](../04-heat-2d-triangles/)
