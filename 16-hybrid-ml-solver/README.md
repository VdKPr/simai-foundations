# 16 · Hybrid ML + Solver: the RANG Pattern

**Time:** one or two evenings · **Code:** `hybrid_newton.py` (scikit-learn, about 45 s) · **Unlocks:** Simulation Fundamentals: "Can ML speed up a nonlinear solver *without* changing its answer?"; Must Know "When should you *not* replace a numerical solver with ML?"

## Why you care

Modules 11–15 *replaced* the solver with a model. That's fast, but the answer is only as good as the model, and engineers who sign off designs are rightly nervous about that.

The alternative is to keep the solver and let ML **help** it: better initial guesses, better preconditioners, smarter time steps, coarse-to-fine corrections. The solver's convergence check stays in charge, so the answer is exactly as trustworthy as before. This is the pattern most likely to be accepted inside a commercial solver. It's also your thesis, and this module rebuilds its logic in miniature so you can explain it cleanly.

## The pattern

```
for each time step:
    candidates = { previous solution,
                   linear extrapolation 2·φₙ − φₙ₋₁,
                   ML prediction }
    pick the candidate with the smallest residual        ← the safe selector
    run Newton from it until the residual < 10⁻¹⁰         ← the solver decides the answer
```

Three ideas carry the whole design:

1. **ML only proposes.** Newton always corrects. However bad the guess, the final answer satisfies the same tolerance.
2. **Judge guesses by their residual** (module 08's lesson), and pick the best one each step. A residual evaluation costs about as much as one matrix–vector product, far cheaper than a Newton iteration.
3. **Compare against strong classical baselines.** Linear extrapolation is free and often excellent.

The ML model here is small and local. For each grid point, it looks at φ at that point and its two neighbours on each side, at the current and previous step, and predicts the **correction to linear extrapolation**. Learning a correction to a good baseline (rather than the whole answer) is a common trick for making ML help reliably.

## Results from the script

1D Allen–Cahn (module 07), 200 unknowns, 5 new trajectories. The ML model was trained on 20 trajectories at dt = 0.05 and interface width ε = 0.03.

**A. Same settings as training (in distribution)**

| Starting guess | Newton iterations | Per step | Wall time incl. guess cost |
|---|---|---|---|
| previous step | 859 | 2.86 | 0.49 s |
| extrapolation | 610 | 2.03 | **0.30 s** |
| ML | **605** | 2.02 | 0.72 s |
| safe selector | 605 | 2.02 | 0.75 s |

**B. Time step doubled (the ML model never saw dt = 0.10):** extrapolation 310 iterations, ML 310, selector 310.

**C. Interfaces half as wide (ε = 0.015, new physics):** extrapolation 611, ML **614** (slightly worse), selector 612. **The selector picked ML 0% of the time.**

Honest reading:

1. **Extrapolation is very hard to beat.** It already needs only about 2 Newton iterations per step. Since you always need at least about one iteration just to *check* convergence, there's at most one iteration per step left to save. ML saved 5 iterations out of 610 (under 1%).
2. **ML cost more wall time than it saved:** 0.72 s vs 0.30 s. Evaluating even a small network every step costs more than the iterations it removes. **Count wall time, not just iterations.**
3. **The safe selector does its job.** When the ML guess stopped helping (case C), the selector stopped choosing it, and the iteration count stayed at the extrapolation level. Without the selector, a worse ML model could cost iterations or, in a harder problem, push Newton towards divergence.
4. **Every row gives the same final answer.** That's the point of the pattern.

This is the same conclusion as your thesis: classical second-order extrapolation outperformed the neural warm start in most conditions, and the defensible contribution is the **residual-based safe-selector framework**, not "the network is better". That's a strong, honest interview story.

## When does an ML guess pay off?

The hybrid pattern is worth it when the classical guess is poor and each Newton iteration is expensive:

- **Large time steps or big parameter jumps** (design optimization, load stepping), where extrapolation from the last step is far off
- **Expensive iterations:** large 3D models where one Newton iteration means a big sparse factorization (module 08)
- **Hard nonlinearities** (contact, plasticity, phase change), where a bad guess means many iterations or divergence and a good one rescues convergence
- **Cheap models**, so the ML evaluation is a tiny fraction of an iteration's cost

## Run it

```bat
python hybrid_newton.py
```

About 45 seconds (most of it training the guesser).

### What to look for

1. Case A: ML and extrapolation nearly tied on iterations, ML clearly behind on time.
2. The "safe selector picked" lines: mostly extrapolation, with ML dropping to 0% in case C.
3. The closing lines: same answer every time.

## Exercises

1. Make the ML model useless: replace its prediction in `train_guesser` with `return lambda phi_n, phi_prev: phi_n + 0.5` (a bad guess). Which rows change, and which stay protected?
2. What's the lowest possible number of Newton iterations per step if the guess were perfect (exactly the answer)? And with any guess that isn't already within tolerance?
3. Write the two-sentence thesis summary you'd give an Ansys interviewer.

<details>
<summary>Answers</summary>

1. The ML row doubles to 1,200 iterations (4 per step). The safe-selector row stays at 615, about the extrapolation level, because the bad guess has a large residual and is picked 0% of the time. That's the protection the selector gives. (615 rather than 610 because "smallest residual" is a proxy: occasionally the previous step has a slightly smaller residual but needs one more iteration.)
2. A perfect guess needs 0 iterations: the first residual check already passes. Any other guess needs at least 1. With extrapolation at ~2 per step, the most any guess can save is about 1 iteration per step.
3. For example: "I built a hybrid solver for the Allen–Cahn phase-field equation where a neural network proposes Newton initial guesses and a residual-based selector picks the best candidate, so the solver's answer is never compromised. The honest finding was that second-order extrapolation was already near-optimal; the transferable contribution is the safe-selection framework and a fair benchmarking method that counts wall time, not just iterations."

</details>

## Where this connects

- **Module 07:** Newton's quadratic convergence is why a good guess saves only a couple of iterations.
- **Module 08:** residual, not pointwise error, is the right yardstick for a guess.
- **Module 17:** the selector is an out-of-distribution guard. When the ML guess is unlike anything seen before, its residual gives it away.
- **Guide:** "When should you *not* replace a numerical solver with ML?" This module is half the answer: often, help it instead.

## Check yourself

1. Why can't an ML initial guess change the final answer?
2. Why is linear extrapolation such a strong baseline?
3. Why judge candidate guesses by residual rather than by pointwise error?
4. In case A, ML saved iterations but lost on wall time. Why?
5. Name two conditions where an ML guess is more likely to pay off.

<details>
<summary>Answers</summary>

1. Newton keeps iterating until the residual is below the tolerance. The guess only changes how many iterations that takes.
2. Solutions change smoothly in time, so 2φₙ − φₙ₋₁ is already very close: second-order accurate in dt. Newton then needs only about 2 iterations.
3. The solver works on the residual, and a guess with small pointwise error can have a large residual if its error is jagged (module 08).
4. Evaluating the network each step took longer than the few iterations it saved.
5. Any two of: large steps or parameter jumps (poor classical guess), expensive iterations (big 3D models), hard nonlinearities where convergence is at risk, and very cheap ML models.

</details>

## Words introduced here

hybrid ML–solver · warm start · safe selector · residual-based selection · learning a correction (residual learning) · wall-clock benchmarking · solver-in-the-loop

## Now read in the guide

- `answers/newton-raphson-and-nonlinear-solvers.md`: *Speeding up a solver with ML without changing its answer*
- `answers/ml-vs-numerical-solvers.md`: *When not to replace a solver with ML*

**Next:** [17 · Uncertainty, OOD and active learning](../17-uncertainty-and-active-learning/)
