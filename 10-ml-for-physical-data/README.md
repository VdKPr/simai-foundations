# 10 · ML on Physical Data: Scaling, Splits and Error Metrics

**Time:** one evening · **Code:** `physical_data.py` (needs `scikit-learn`) · **Unlocks:** ML Fundamentals for Physical Data; Evaluation and Testing

## Why you care

Part B starts here. Before any fancy architecture (PINNs, neural operators, graph nets), three plain decisions make or break an ML model on simulation data:

1. **How you scale** the inputs and outputs
2. **How you split** the data into training and test sets
3. **Which error you report**

Get these wrong and no architecture saves you. Get them right and a small network does surprisingly well. Interviewers probe exactly these, because they reveal whether someone has trained models on real engineering data or only on tutorial datasets.

## Quick ML vocabulary

You know most of this from ManuVision; here it is in this repo's language:

- **Training set:** the examples the model learns from. **Test set:** examples held back to measure how well it does on designs it hasn't seen.
- **MLP (multi-layer perceptron):** the basic neural network. Layers of weighted sums, each followed by a simple nonlinearity. It learns its weights by gradient descent on the training error.
- **Standardization:** subtract the mean and divide by the standard deviation for each input and output column, so every column is "about 1 in size".
- **Overfitting:** fitting the training data's noise instead of the underlying pattern. Training error is low, test error is high.

## The dataset

2,000 cantilever designs per metre width, with the exact deflection from Timoshenko beam theory (module 05), so we know the truth:

| Input | Range |
|---|---|
| Length L | 0.5–2.0 m |
| Depth H | 0.05–0.20 m |
| Young's modulus E | 70–210 GPa (aluminium to steel) |
| Tip load P | 1–50 kN/m |

The **target** is the tip deflection, which spans 0.0024 mm to 57.5 mm: a **23,600× range**. Physical data is like this. Values cover many orders of magnitude, and inputs come in wildly different units.

## 1. Scaling: the same network, three ways

| Preprocessing | RMSE [mm] | Median relative error | 95th percentile | Worst |
|---|---|---|---|---|
| A. raw numbers | 4,300 | >1000% | >1000% | >1000% |
| B. standardized | 0.29 | 10.5% | 199% | >1000% |
| C. log, then standardized | **0.094** | **1.3%** | **4.6%** | **9.4%** |

- **A fails completely.** E is about 10¹¹ and L is about 1. The network's weighted sums are swamped by E, and gradient descent can't make progress.
- **B learns, but not well.** The large deflections are fitted; the small ones are badly off in relative terms.
- **C is excellent.** The physics is a **power law**: deflection ∝ P·L³/(E·H³) (plus a small shear term). Taking logs turns a power law into a straight line: log δ = log P + 3 log L − log E − 3 log H + const. The network now only has to learn something nearly linear.

**General lesson:** use your physics knowledge to choose the representation. Logs for quantities spanning decades, dimensionless groups (Reynolds, Biot, slenderness L/H) where they exist, and per-channel standardization always. Compute the scaling statistics **on the training set only**; using test-set statistics leaks information about the test data into training.

## 2. Interpolation vs extrapolation

Scored on the **same** test beams (L > 1.7 m). The only difference is whether training data covered that range:

| Model | Training data | RMSE [mm] | Median | 95th percentile |
|---|---|---|---|---|
| B | all lengths | 0.50 | 4.1% | 28.7% |
| B | L ≤ 1.7 m only | 1.31 | 8.9% | 52.7% |
| C | all lengths | 0.165 | 1.1% | 3.5% |
| C | L ≤ 1.7 m only | 0.257 | 1.5% | 5.5% |

- **Interpolation:** predicting inside the range the model was trained on. **Extrapolation:** predicting outside it.
- Errors grow under extrapolation for both models. A **random split** only tests interpolation, so it flatters every model.
- The physics-aware representation (C) extrapolates much better, because a straight line in log space keeps going in the right direction.

**In practice:** customers always ask about the next design, which is often slightly outside the old ones. Test that way: split by design family, by geometry, or by parameter range, not randomly.

## 3. What a single number hides

For model B (random split):

| Designs | Median relative error |
|---|---|
| Stiffest quarter (deflection < 0.165 mm) | **74.5%** |
| Floppiest quarter (deflection > 2.1 mm) | 3.3% |

The overall RMSE of 0.29 mm looks fine because it's dominated by the big deflections. For the stiff designs, which are probably the ones being optimized, the model is useless.

**Report:**

- **Relative errors**, not only absolute ones
- **Percentiles** (median, 95th, worst case), not only the mean
- **Errors per region of the design space**, so a weak region can't hide behind an average

For fields (whole temperature or stress maps), the standard metric is the **relative L2 error**, ‖prediction − truth‖ / ‖truth‖: the same quantity you computed in module 03.

## Run it

```bat
pip install scikit-learn
python physical_data.py
```

It takes 10–20 seconds. Your numbers will differ slightly (different CPU maths libraries); the pattern won't.

## Exercises

1. In `make_dataset`, widen the load range to `P = RNG.uniform(1e2, 1e6, n)`. Which pipeline gets worse, and why?
2. Add the slenderness ratio L/H as a fifth input column for pipeline B. Does it help? Why might it?
3. Your manager says "the model has 0.3 mm RMSE, ship it". What three questions do you ask first?

<details>
<summary>Answers</summary>

1. Pipeline B gets worse (median 10.5% → 12.8%, 95th percentile 199% → 338%), because an even wider target range makes standardization alone inadequate. Pipeline C barely notices (1.2% median, 4.4% 95th), because a 10,000× range is just a wider straight line in log space.
2. It helps somewhat: RMSE 0.29 → 0.16 mm and median 10.5% → 8.7%, because deflection depends strongly on (L/H)³ and the network no longer has to discover that combination. But the 95th percentile stays near 200%. A good feature doesn't fix the scaling problem of a target spanning four decades; the log transform does.
3. Relative to what: what's the error on the smallest deflections we actually care about? Was it tested on designs outside the training range, or only on a random split? What's the worst case, not just the average?

</details>

## Where this connects to the rest of Part B

- Every later module uses these habits: scaling inputs, held-out test designs, relative and percentile errors.
- Module 11 uses DOE to choose which simulations to run in the first place.
- Module 17 makes the extrapolation problem explicit: how to *detect* that a new design is outside the training range.

## Check yourself

1. Why does training on raw physical units fail?
2. Why must the scaling statistics come only from the training set?
3. When would you log-transform the target?
4. What does a random train/test split actually test?
5. Name three things to report instead of a single RMSE.

<details>
<summary>Answers</summary>

1. Columns differ by many orders of magnitude, so the large ones dominate the weighted sums and the gradients. The network can't learn effectively.
2. Using test statistics leaks information about the test data into training, so the measured test error comes out optimistic.
3. When it's positive and spans orders of magnitude, or when the physics is multiplicative (a power law).
4. Interpolation: how well the model predicts designs similar to ones it trained on.
5. Relative error, percentiles including the worst case, and errors by region or design family (plus extrapolation tests).

</details>

## Words introduced here

training / test set · MLP · standardization · overfitting · power law · log transform · dimensionless groups · data leakage · interpolation / extrapolation · random vs grouped split · relative L2 error · percentile errors

## Now read in the guide

- The guide section *Machine Learning Fundamentals for Physical Data*: all questions
- `answers/out-of-distribution-and-generalization.md`: *Interpolation vs extrapolation*
- The guide section *Evaluation and Testing*: the questions on metrics and splits

**Next:** [11 · Surrogate models and design of experiments](../11-surrogates-and-doe/)
