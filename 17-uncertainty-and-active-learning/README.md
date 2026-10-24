# 17 · Uncertainty, Out-of-Distribution Detection and Active Learning

**Time:** one or two evenings · **Code:** `uq_active.py` (scikit-learn, about 30 s) · **Unlocks:** Uncertainty, Generalization and OOD; Must Know "How do you know whether a surrogate's prediction can be trusted for one specific new design?"

## Why you care

An engineer signing off a design can't use "the model has 1% average error". They need to know **whether this particular prediction, for this particular design, can be trusted**. That's the question every serious customer asks, and the one that separates a research demo from a product.

Three tools answer it:

1. **Uncertainty estimates:** the model says how sure it is about each prediction.
2. **Out-of-distribution (OOD) detection:** flag designs unlike anything in the training data.
3. **Active learning:** use the uncertainty to decide which simulation to run next.

## Two kinds of uncertainty

- **Aleatoric uncertainty:** randomness in the data itself, such as measurement noise or scatter in material properties. More data doesn't remove it.
- **Epistemic uncertainty:** the model's ignorance, because it hasn't seen enough data in that region. More data *does* remove it.

Simulation data is usually noise-free (same inputs, same answer), so for surrogates it's mostly **epistemic** uncertainty that matters, and it is largest exactly where the training data is missing: OOD.

## Two ways to get uncertainty

- **Gaussian process** (module 11): gives a standard deviation with every prediction, built into the maths. Far from training points, it falls back to "I don't know" (large std).
- **Deep ensemble:** train several neural networks the same way but with different random starting weights. Where the data pins the answer down, they agree. Where it doesn't, they disagree. Their spread is the uncertainty.

## The test problem

**Pin-fin efficiency**, a classic heat-transfer formula: η = tanh(mL)/(mL) with m = √(4h/(kD)), for an aluminium pin 3 mm in diameter. Inputs: convection coefficient h and fin length L. It's cheap, so we know the truth everywhere and can check every uncertainty claim.

- **Training box:** h = 10–300 W/m²·K, L = 5–40 mm.
- **In distribution:** 500 test designs inside the box.
- **Out of distribution:** 500 designs with h = 300–600 and L = 40–80 mm, entirely outside it.

## Results from the script

### 1. Gaussian process (15 training designs)

| Test designs | Mean error | Mean predicted std | Truth inside ±2 std |
|---|---|---|---|
| In distribution | 0.0025 | 0.0023 | 89% |
| Out of distribution | 0.035 (14× worse) | 0.137 (60× larger) | 100% |

- Outside the box, the error grows 14×, and the GP's own uncertainty grows even more. It "knows that it doesn't know", which is the most important property.
- **Calibration** asks whether the uncertainty is the right size. For a well-calibrated model, about 95% of the truths lie within ±2 std. In distribution it's 89% (slightly overconfident); out of distribution it's 100% (very cautious). Always check calibration; don't assume it.

### 2. Deep ensemble (5 networks, 40 training designs)

| Test designs | Mean error | Mean disagreement | Error ÷ disagreement |
|---|---|---|---|
| In distribution | 0.0014 | 0.0018 | 0.8× |
| Out of distribution | 0.076 | 0.026 | **3.0×** |

- **As an OOD detector it works:** with a threshold at the 95th percentile of in-distribution disagreement, 100% of OOD designs get flagged.
- **As an error estimate outside the box it's overconfident:** the real error is 3× larger than the disagreement. All five networks extrapolate in similar wrong ways, so they agree more than they should.

Use ensemble disagreement as an alarm, not as an error bar, when you're outside the data.

### 3. Active learning (Gaussian process, 5 starting designs)

Add one simulation at a time, either at a random point or where the GP is **most uncertain**:

| Worst error on 500 test designs | n = 5 | n = 10 | n = 15 | n = 20 | n = 30 |
|---|---|---|---|---|---|
| Random | 0.352 | 0.135 | 0.028 | 0.024 | 0.023 |
| Most uncertain | 0.352 | **0.024** | 0.008 | 0.008 | **0.002** |

With 30 simulations, choosing where the GP is least sure gives a **10× smaller worst-case error** than random sampling. It naturally fills the edges and corners of the design space, where random points rarely land. When each simulation costs hours, this is the difference between affordable and not.

## Simple OOD checks that always help

Before any fancy method:

- **Range check:** is every input inside the training range?
- **Distance check:** how far is the new design from its nearest training design, in scaled inputs?
- **Physics checks:** does the prediction conserve energy (module 04), respect symmetry, stay within physical bounds (an efficiency between 0 and 1, temperatures above the coolant)?
- **Fallback:** if any check fails, run the real solver. The surrogate's job is to make the common case fast, not to answer every question.

## Run it

```bat
python uq_active.py
```

### What to look for

1. **Section 1:** the predicted std jumping outside the box, and the calibration column.
2. **Section 2:** 100% of OOD designs flagged, but error/disagreement at 3× outside.
3. **Section 3:** the "most uncertain" row pulling ahead from n = 10.

## Exercises

1. In section 1, train the GP on 40 designs instead of 15 (`sample(BOX, 40, seed=0)`). Does in-distribution calibration move closer to 95%?
2. Design a deployment rule for this surrogate in a design tool, using range, distance and uncertainty checks. What happens when a check fails?
3. Why does uncertainty-based active learning help the *worst-case* error more than the median?

<details>
<summary>Answers</summary>

1. With 40 designs, the in-distribution error drops 10× (0.0025 → 0.00024) and coverage goes from 89% to 98%: now slightly *under*confident instead of over. Out of distribution, the error barely changes (0.035 → 0.046), because more data inside the box teaches nothing about outside it. Calibration moves around every time you retrain, so measure it every time.
2. One reasonable rule: (1) reject inputs outside the training ranges; (2) if the scaled distance to the nearest training design exceeds the largest distance seen in validation, warn; (3) if the predicted std exceeds a tolerance tied to the engineering margin (say 2% efficiency), warn. On any rejection or warning, run the real solver automatically, and add the result to the training set for the next retrain. That last step is active learning in production.
3. Uncertainty is largest where the data is sparsest, and that's also where the worst errors are. Filling those gaps removes the worst errors first. The median is dominated by well-covered regions that random sampling already handles.

</details>

## Where this connects

- **Module 10:** random splits hide OOD failure; this module is the defence.
- **Module 11:** DOE chooses the first batch of simulations; active learning chooses the next ones.
- **Module 16:** the residual-based safe selector is an OOD guard for a hybrid solver: when the ML guess is bad, a physics check catches it.

## Check yourself

1. Aleatoric vs epistemic uncertainty: which matters most for simulation surrogates, and why?
2. How does a deep ensemble produce an uncertainty estimate?
3. What does "calibrated" mean, and how do you check it?
4. Ensembles flagged 100% of OOD designs. Why shouldn't you trust their spread as an error bar there?
5. What is active learning, and why does it help when simulations are expensive?

<details>
<summary>Answers</summary>

1. Epistemic: simulations are deterministic, so the uncertainty comes from missing data, and it's largest outside the training data.
2. Several networks are trained from different random starting weights; the spread of their predictions is the uncertainty.
3. The predicted uncertainty matches the real error rate: for example, about 95% of the truths fall within ±2 std. Check it on held-out data.
4. The networks extrapolate in similar ways, so they agree more than they should. Here the real error was 3× the disagreement.
5. Choosing the next simulation where the model is most uncertain. Each costly simulation then reduces the error as much as possible, especially the worst-case error.

</details>

## Words introduced here

aleatoric / epistemic uncertainty · out-of-distribution (OOD) · calibration · coverage · deep ensemble · disagreement · active learning · range / distance / physics checks · fallback to the solver

## Now read in the guide

- `answers/out-of-distribution-and-generalization.md`: the whole page
- The guide section *Uncertainty, Generalization and Out-of-Distribution Detection*: all questions
- Then go back to the guide's *Must Know* section and answer every question out loud. That's the end of this repo.

**You've finished the roadmap.** Next: the guide's sections on 3D geometry, transformers, generative models, LLM agents and system design, which build on everything here.
