# 12 · Reduced-Order Models: POD and SVD

**Time:** two evenings · **Code:** `pod_rom.py` · **Unlocks:** Surrogate Models, ROMs and DOE (the ROM questions)

## Why you care

Module 11 predicted one number (peak temperature). Engineers usually want the **whole field**: the full temperature or stress map. A field on a real mesh has millions of values, far too many to predict one by one.

The trick: the fields from a family of designs are not arbitrary. They are all built from a few characteristic shapes. Find those shapes once, and each new field is just a handful of coefficients. That is a **reduced-order model (ROM)**, and **POD (proper orthogonal decomposition)** is the standard way to find the shapes. Ansys Twin Builder's static and dynamic ROMs and many "digital twin" products are built on this idea.

You've met it before in mechanics: **modal analysis.** A vibrating beam's motion is a combination of a few mode shapes. POD finds "mode shapes" from data instead of from the equations.

## POD in plain words

1. **Collect snapshots.** Run the simulation for many designs (same mesh each time). Each result is a column of numbers, one per node. Stack them side by side into a **snapshot matrix** S (here: 1,701 nodes × 100 designs).
2. **Subtract the mean field.**
3. **Take the SVD** (singular value decomposition): S − mean = Φ Σ Vᵀ.
   - The columns of **Φ** are the **POD modes**: the characteristic shapes, ordered from most to least important.
   - The **singular values σ** in Σ say how much each mode contributes. Their squares are the "energy" each mode captures.
4. **Keep the first r modes.** Any field from the family is then approximately: mean + c₁·mode₁ + … + c_r·mode_r.

For a new design, you only need to predict r numbers (the coefficients c), not 1,701.

**Non-intrusive ROM:** learn the map *design inputs → coefficients* with a regression model (a GP here, as in module 11). "Non-intrusive" means you only use solver outputs, not the solver's equations or matrices. An **intrusive** ROM projects the solver's own equations onto the modes. It's more accurate and physics-consistent, but needs access to the solver's internals.

## Results from the script

Two families of heat-spreader designs, 100 training snapshots and 50 new test designs each:

- **Set A:** vary the heat flux Q, the cold-plate coefficient h and the conductivity k.
- **Set B:** vary only the component's position along the top edge (a moving heat source).

**Energy captured by r modes:**

| | r = 1 | r = 2 | r = 3 | r = 5 | r = 10 |
|---|---|---|---|---|---|
| Set A | 97.6% | 99.997% | 100.000% | 100% | 100% |
| Set B | 92.3% | 98.4% | 99.6% | 99.96% | 99.997% |

**Best-case error on new designs** (project the true field onto r modes):

| | r = 1 | r = 2 | r = 3 | r = 5 | r = 10 | r = 20 |
|---|---|---|---|---|---|---|
| Set A | 7.2% | 0.35% | **0.001%** | 0.000% | 0.000% | 0.000% |
| Set B | 2.9% | 1.4% | 0.73% | 0.20% | 0.060% | 0.014% |

What it shows:

1. **Set A needs only 3 modes.** There's a physics reason: the problem is linear in Q, so the field is (Q/k) × a shape that depends only on the ratio h/k, a Biot-like dimensionless number (module 10). Three inputs, but effectively one shape-changing parameter.
2. **Set B needs many more modes for the same accuracy.** A feature that *moves* is hard to build from a fixed set of shapes: the modes look like waves along the top edge (see the plot). This is the classic weakness of POD and linear ROMs. It's much worse for moving shocks, flames and contact fronts, where the field has sharp features.
3. **"97.6% energy" does not mean 97.6% accurate.** With r = 1, set A has 97.6% of the energy but a 7.2% error on new designs. Look at errors, not just energy.

**Non-intrusive ROM for set A** (5 modes + one GP per coefficient): **0.10% median field error, 1.5% worst**, on 50 new designs. The truncation floor at 5 modes is 0.000%, so all the error comes from the GPs predicting the coefficients. The ROM stores 5 modes and 5 small GPs instead of 100 full fields.

> **A lesson from writing this module.** The first version snapped the component to mesh edges, in 0.5 mm jumps. The set B ROM then had 2.5% error instead of 0.2%: the coefficients changed in steps as the component moved, and the GP couldn't fit steps. After making the position continuous, the ROM error dropped to the truncation floor. **Artifacts in how the simulation is set up become surrogate error.**

## Run it

```bat
python pod_rom.py
```

About 5 seconds.

### What to look for

1. **The singular value plot:** set A drops to 10⁻¹⁴ by mode 7 (machine precision); set B decays slowly.
2. **The mode shapes:** set A's modes are smooth global patterns; set B's look like waves along the top edge, needed to shift the hot spot around.
3. **The ROM line:** errors close to the truncation floor means the regression part is good.

## Exercises

1. Explain in one sentence why set A is so low-rank, using the idea of dimensionless groups.
2. Build the same non-intrusive ROM for set B (a GP from position → coefficients) with r = 5, 10 and 20. How does its error compare with the projection floor?
3. When would you choose POD + GP over a neural network that predicts the whole field directly?

<details>
<summary>Answers</summary>

1. The temperature field is (Q/k) × f(h/k), so the field's *shape* depends on one dimensionless number, and a few modes cover every shape.
2. Median errors of 0.20%, 0.060% and 0.018% for r = 5, 10, 20: essentially equal to the projection floor. With a continuous input, the GP fits the coefficients well. For set B, the limit is how many modes you keep, not the regression.
3. When data is limited (tens to hundreds of simulations), the design space is low-dimensional, and the mesh is fixed. POD + GP is data-efficient, fast and interpretable. A neural network needs more data, but handles varying meshes and geometry, high-dimensional inputs and nonlinear features better (modules 14–15).

</details>

## Where this connects to ML

- **POD is PCA** (principal component analysis) applied to simulation fields. Same maths, different name.
- **Autoencoders are nonlinear POD.** A neural network compresses each field to a few numbers and decompresses it back. It can handle moving features better than linear modes, at the cost of more data and training.
- **The "modes + coefficient regression" pattern** reappears in DeepONet (module 14): a "trunk" network learns basis functions, and a "branch" network learns their coefficients.

## Check yourself

1. What is a snapshot matrix?
2. What do the POD modes and singular values represent?
3. What's the difference between an intrusive and a non-intrusive ROM?
4. Why does POD struggle with moving features?
5. In a POD + regression ROM, what are the two sources of error?

<details>
<summary>Answers</summary>

1. Simulation results for many designs, each flattened to a column (one value per node), stacked side by side.
2. Modes: characteristic field shapes, ordered by importance. Singular values: how much each mode contributes (their squares are energy).
3. Intrusive projects the solver's equations onto the modes and needs solver access. Non-intrusive learns the coefficients from input–output data only.
4. A feature in a new position isn't a combination of a few fixed shapes, so you need many modes (slow singular value decay).
5. Truncation (too few modes) and regression (wrong predicted coefficients).

</details>

## Words introduced here

reduced-order model (ROM) · POD · snapshot matrix · SVD · POD modes · singular values · energy · truncation · non-intrusive / intrusive ROM · PCA · autoencoder · digital twin

## Now read in the guide

- The guide section *Surrogate Models, Reduced-Order Models and Design of Experiments*: the ROM and POD questions
- `answers/modal-analysis.md`: compare physical modes with POD modes

**Next:** [13 · Physics-informed neural networks](../13-pinns/)
