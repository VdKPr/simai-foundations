# 11 · Surrogate Models and Design of Experiments

**Time:** two evenings · **Code:** `surrogate_doe.py` (scikit-fem + scikit-learn, about 30 s) · **Unlocks:** Surrogate Models, ROMs and DOE; the Must Know question "what is a surrogate model and why do companies want one?"

## Why you care

A **surrogate model** is a cheap stand-in for an expensive simulation: it learns the mapping from design inputs to results from a set of simulation runs, then answers new "what if?" questions in microseconds. This is the single most common commercial use of ML in simulation, and a product category for Ansys (optiSLang, SimAI) and Siemens (HEEDS, Simcenter).

Engineers use surrogates for **design exploration** (thousands of variants), **optimization** (an optimizer calls the model millions of times), **uncertainty studies** (Monte Carlo over manufacturing tolerances) and **real-time tools** (a quick estimate inside a CAD or sales tool).

Two questions decide whether a surrogate works:

1. **Which simulations do you run to train it?** That's design of experiments (DOE).
2. **Which model do you fit?** That depends on how much data you have and how nonlinear the response is.

## Design of experiments

With a budget of n simulations in a 4-dimensional design space, where do you put them?

| Method | How it picks points | Pros and cons |
|---|---|---|
| **Full factorial grid** | every combination of k levels per input | k^d runs: 3 levels in 10 dimensions is 59,049 runs; hopeless beyond a few inputs |
| **Random (Monte Carlo)** | uniform random points | simple; but clumps and gaps by chance |
| **Latin hypercube (LHS)** | split each input's range into n slices; use each slice exactly once | every input's range is evenly covered with any n; the industry default |
| **Sobol / quasi-random** | a deterministic sequence that fills space evenly | good coverage, easy to extend later |

"Latin" comes from a Latin square: like Sudoku, one point per row and per column, for every pair of input axes.

**Scale matters here too** (module 10): h and k span a decade each, so the DOE samples them evenly in *log* space. Otherwise most points would land at the high end.

## Two surrogate models

**Quadratic response surface (RSM).** Fit a full quadratic polynomial in the inputs: in 4D that's 15 coefficients (constant, 4 linear, 4 squares, 6 cross terms). It's the classic approach: transparent and cheap. But it can't represent anything more curved than a quadratic, and it needs at least 15 runs.

**Gaussian process (GP).** A GP assumes that designs close together have similar outputs, and learns from the data how "close" works along each input direction (one **length scale** per input). It passes through the training points and, between them, blends their values smoothly. Two strengths:

- It's very data-efficient: the go-to model for tens to a few hundred expensive runs.
- It gives an **uncertainty** with every prediction (module 17).

Its cost grows like n³ with the number of training points, so it isn't used for very large datasets; neural networks take over there.

## Results from the script

The heat spreader from module 04 with four design inputs: component heat flux Q, cold-plate coefficient h, conductivity k (ceramic to aluminium), plate thickness H. Output: peak temperature rise (5–116 °C over the test set). Each number is averaged over 5 repeats with different random designs, scored on the same 300 test designs.

**Median relative error:**

| Surrogate | DOE | n = 10 | n = 20 | n = 40 | n = 80 |
|---|---|---|---|---|---|
| Quadratic RSM | random | 9.8% | 2.2% | 1.9% | 1.8% |
| Quadratic RSM | LHS | 8.3% | 2.5% | 1.9% | 1.9% |
| GP | random | 7.5% | 3.2% | 1.2% | 0.42% |
| GP | LHS | 6.9% | 3.6% | 1.1% | **0.33%** |

**95th-percentile relative error:**

| Surrogate | DOE | n = 10 | n = 20 | n = 40 | n = 80 |
|---|---|---|---|---|---|
| Quadratic RSM | random | 32.9% | 11.0% | 8.5% | 6.6% |
| Quadratic RSM | LHS | 32.7% | 10.0% | 8.4% | 6.9% |
| GP | random | 32.8% | 12.7% | 6.6% | 2.8% |
| GP | LHS | 26.5% | 11.5% | 6.3% | **2.0%** |

What it shows:

1. **The RSM stops improving at about 1.9%.** More data can't fix a model that is too simple for the physics. This is **model-form error**: the true response isn't quadratic. At n = 10 it has fewer runs than its 15 coefficients, so it's underdetermined.
2. **The GP keeps improving** with more data: 0.33% median at 80 runs. Flexible models win once there's enough data.
3. **LHS helps a little, not a lot**, mainly in the bad cases (95th percentile: 2.0% vs 2.8% at n = 80). It's a cheap, sensible default, not magic.
4. **The bad cases are 5–10× worse than the median.** Always report both (module 10).

**Speed:** one FE solve here takes about 9 ms; a GP prediction takes about 2 µs, **~5,000× faster**. For a 3D model that takes 2 hours to solve, the ratio is in the millions. That's the business case.

**And the cost:** the GP is only trustworthy inside the 4D box it was trained on, for this geometry family and these boundary conditions. Change the component shape, add a fin, and it knows nothing.

## Run it

```bat
python surrogate_doe.py
```

About 30 seconds: around 1,000 small FE solves, plus GP fitting.

### What to look for

1. The RSM rows flattening out while the GP rows keep falling.
2. The LHS rows slightly below random, mostly in the 95th-percentile table.
3. The speed-up line at the end.

## Exercises

1. Remove the log transform of the target in `fit_surrogate` (fit `y` instead of `np.log(y)`, and remove the `np.exp`). How do the GP errors change at n = 80?
2. A colleague proposes a 3-level full factorial over 7 design inputs. How many simulations is that? What would you suggest instead, and how many runs would you start with?
3. The GP is 0.33% accurate on the test set. A customer asks for a design with H = 25 mm. What do you tell them?

<details>
<summary>Answers</summary>

1. With LHS at n = 80, the median error goes from 0.33% to 0.72% and the 95th percentile from 2.0% to 4.0%: about twice as bad. Without the log, the GP has to model a quantity spanning 5–116 °C with a single smoothness setting, and the multiplicative physics (rise ∝ Q) isn't a simple shape any more. It's the module 10 lesson again.
2. 3⁷ = 2,187 runs. Use Latin hypercube or Sobol sampling. A common rule of thumb is to start with about 10 runs per input dimension (~70 here), fit a GP, check the error on a held-out set, and add runs where the GP is most uncertain (module 17).
3. That's outside the training range (2–15 mm), so the prediction is an extrapolation and its accuracy is unknown. Either run the FE model directly for that design, or extend the DOE to cover thicker plates and retrain.

</details>

## Where this connects to ML

- **Surrogates are only as good as their training data:** the DOE coverage, and the mesh quality of every label (modules 05, 09).
- **Small data → GP. Large data, high-dimensional inputs or field outputs → neural networks**, often combined with ROMs (module 12) or as neural operators (module 14).
- **Geometry as input is the hard frontier.** Here the geometry was one number (H). When the whole shape varies, you need geometric deep learning or graph networks (module 15 and the guide's 3D geometry section).

## Check yourself

1. What is a surrogate model, and name three things engineers use it for.
2. What does a Latin hypercube guarantee that random sampling doesn't?
3. Why did the quadratic RSM stop improving?
4. Why is a GP a good choice for 10–200 expensive simulations?
5. What determines where a surrogate can be trusted?

<details>
<summary>Answers</summary>

1. A cheap model trained on simulation runs that predicts outputs for new designs. Used for design exploration, optimization, uncertainty quantification and real-time estimates.
2. Each input's range is evenly covered: every one of the n slices is used exactly once.
3. Model-form error: the true response isn't quadratic, so more data can't fix it.
4. It's data-efficient, interpolates smoothly, learns how much each input matters, and gives an uncertainty estimate.
5. The range and coverage of the training designs, plus everything held fixed during training (geometry family, boundary conditions, physics).

</details>

## Words introduced here

surrogate model · design exploration · design of experiments (DOE) · full factorial · Monte Carlo sampling · Latin hypercube sampling (LHS) · Sobol sequence · response surface model (RSM) · Gaussian process (GP) · length scale · model-form error · underdetermined

## Now read in the guide

- `answers/surrogate-models.md`
- The guide section *Surrogate Models, Reduced-Order Models and Design of Experiments*: the DOE and surrogate questions (save the ROM ones for module 12)
- `answers/design-thermal-surrogate.md`: compare it with what you just built

**Next:** [12 · Reduced-order models: POD and SVD](../12-pod-rom/)
