# 06 · Time Stepping: Explicit vs Implicit, Stability, Stiffness

**Time:** one or two evenings · **Code:** `transient_heat.py` · **Unlocks:** Simulation Fundamentals (CFL, explicit vs implicit, stiff equations); Time-Dependent Problems and Autoregressive Rollouts

## Why you care

Anything that changes in time (thermal cycling, crash, unsteady flow, vibration) is solved by marching forward in small time steps. Choosing the time step and the marching method is a daily decision for simulation engineers, and three standard interview questions live here: the CFL condition, explicit vs implicit, and stiff equations.

It matters for ML too. A neural network that predicts the next time step from the current one is a time-marching method, and it inherits the same stability problem. That's the "autoregressive rollout" section of the guide.

## From PDE to a system of ODEs

Discretize space first (finite differences here, FEM works the same way). The PDE ∂T/∂t = α ∂²T/∂x² becomes one ordinary equation per grid point:

```math
\frac{d\mathbf T}{dt} = -A\,\mathbf T
```

T is the vector of temperatures at the grid points. A is the tridiagonal matrix from module 01, scaled by **α (thermal diffusivity)** = k / (ρ c), which says how fast heat spreads (m²/s). This is called the **semi-discrete** form: space is discrete, time is still continuous.

Now we need to march this in time with a step dt.

## Three ways to take a step

| Method | Update | Cost per step | Stable for |
|---|---|---|---|
| **Explicit (forward Euler)** | T_new = T − dt·A·T | one matrix–vector product, no solve | dt < h² / (2α) only |
| **Implicit (backward Euler)** | (I + dt·A) T_new = T | a linear solve every step | any dt |
| **Crank–Nicolson** | average of the two | a linear solve every step | any dt (but see the results) |

**Explicit** uses only what you already know: today's temperatures. **Implicit** writes the equation in terms of tomorrow's unknown temperatures, so each step needs a solve.

**Why explicit can blow up, in plain words.** In one step, heat can only move between neighbouring grid points. If dt is too big, a hot point gives away more heat than it has. Next step it's colder than its neighbours, so it takes back too much, and the overshoot grows every step. The **stability limit** for diffusion is dt < h²/(2α).

## Results from the script

A 100 mm steel rod, middle 20 mm starting 100 °C hotter, ends at 0 °C, 60 s of cooling. The grid has 50 gaps of 2 mm, so the explicit limit is **0.154 s**. Errors are measured against a reference that is exact in time, so they come from the time stepping alone.

| Method | dt / limit | Steps | Max error at 60 s |
|---|---|---|---|
| Explicit | 0.9 | 434 | 0.014 °C |
| Explicit | 1.05 | 372 | **blew up** (10¹⁴ °C) |
| Implicit | 0.9 | 434 | 0.014 °C |
| Implicit | 10 | 39 | 0.16 °C |
| Implicit | 100 | 4 | 1.85 °C |
| Crank–Nicolson | 10 | 39 | **0.005 °C** |
| Crank–Nicolson | 100 | 4 | **34.6 °C**, with ringing |

What it shows:

1. **5% above the limit, explicit explodes.** The right-hand plot shows max |T| growing exponentially.
2. **Implicit never explodes.** Big steps cost accuracy (1.85 °C error with only 4 steps), not stability.
3. **Crank–Nicolson is second-order accurate**: at 10× the limit it's 35× more accurate than implicit Euler with the same steps. But with huge steps on a sharp initial profile it **rings**: the profile oscillates and dips to about −17 °C, colder than both the ends and the start. That's physically impossible. The fix is to take the first few steps with implicit Euler or start with smaller steps.

## Stiffness: why fine meshes punish explicit methods

| Gaps | h | Explicit dt limit | Explicit steps to reach 60 s |
|---|---|---|---|
| 50 | 2 mm | 0.154 s | 390 |
| 100 | 1 mm | 0.038 s | 1,560 |
| 200 | 0.5 mm | 0.0096 s | 6,240 |
| 400 | 0.25 mm | 0.0024 s | 24,958 |

Halving h cuts the allowed dt by 4×, even though the physics (cooling over about a minute) isn't any faster. The finest wiggles the grid can represent decay very fast, and the explicit method has to resolve them even though you don't care about them.

A system is **stiff** when it contains very fast and very slow behaviour at the same time, and you only care about the slow part. Stiff problems push you to implicit methods: take big steps for the slow physics and let the method damp the fast parts.

## The CFL condition (waves and flow)

For waves and flow (hyperbolic problems, module 01), the explicit limit is different:

```math
\Delta t < \frac{h}{c}
```

c is the wave or flow speed. In words: **in one time step, information must not travel further than one cell.** This is the **CFL (Courant–Friedrichs–Lewy) condition**. The number c·Δt/h is the **Courant number**, which must stay below about 1 for explicit schemes.

**Crash simulation example:** the speed of sound in steel is about 5000 m/s. With 5 mm elements, Δt < 0.005 / 5000 = 1 microsecond. A 100 ms crash event needs about 100,000 steps. Crash codes (LS-DYNA, Abaqus/Explicit) still use explicit methods, because the physics really is fast and each explicit step is cheap. One tiny element can shrink the time step for the whole model, which is why engineers use **mass scaling** (adding artificial mass to the smallest elements to slow their waves down).

## Run it

```bat
python transient_heat.py
```

### What to look for

1. The **BLEW UP** row, and the orange curve shooting to 10¹⁴ in the right plot.
2. In the left plot: the explicit and implicit profiles sitting on the reference, and the green Crank–Nicolson line ringing below zero.
3. The stiffness table: steps × 4 for every halving of h.

## Exercises

1. Refine to 100 gaps but keep the old time step of 0.138 s. In `main()`, set `n = 100` and change `("explicit", 0.9)` to `("explicit", 3.6)` (0.138 s is 3.6× the new limit). Stable or not? Why?
2. Replace the sharp initial profile with a smooth one: in `setup()`, use `T0 = 100 * np.sin(np.pi * x / L)`. Re-run. Does Crank–Nicolson at 100× still ring?
3. Which method would you choose for (a) a 2-hour thermal soak of an engine block, (b) a 50 ms drop test of a phone, and why?

<details>
<summary>Answers</summary>

1. It blows up (numpy overflows to NaN). The new limit is 0.038 s, so 0.138 s is 3.6× too big. Refining the mesh without shrinking dt breaks explicit methods.
2. No ringing. Crank–Nicolson's error drops to 0.11 °C, while implicit Euler's is 3.1 °C (on a 46 °C peak). Ringing comes from sharp features, which contain the fast-decaying high-frequency content that Crank–Nicolson doesn't damp.
3. (a) Implicit: slow diffusion over a long time, a stiff problem, so big steps are what you want. (b) Explicit: fast wave and contact physics needs tiny steps anyway, and explicit steps are cheap and handle contact well.

</details>

## Where this connects to ML

- **A learned time-stepper is an explicit method.** A network trained to map T_n → T_n+1 feeds its own output back in. Small errors can grow step by step until the rollout drifts or blows up, just like the orange curve. This is the **rollout stability** problem.
- **Common fixes from the literature:** add noise to the inputs during training so the model learns to correct its own errors (used in MeshGraphNets), or train on several unrolled steps at once instead of one.
- **Learned models can take big steps.** A network can learn the map over a large Δt directly, skipping the explicit limit. That's one source of ML speed-ups, but with no stability guarantee.

## Check yourself

1. What does "semi-discrete" mean?
2. What is the explicit stability limit for diffusion, and why does it depend on h²?
3. What is the CFL condition in one sentence?
4. What is a stiff system, and which kind of method does it push you towards?
5. Why do autoregressive ML surrogates have a stability problem?

<details>
<summary>Answers</summary>

1. Space has been discretized, time hasn't: the PDE has become a system of ODEs, dT/dt = −A T.
2. dt < h²/(2α). The fastest mode the grid can represent decays at a rate proportional to α/h², and an explicit method must take steps smaller than that time scale.
3. In one explicit step, information mustn't travel more than one cell: c·Δt/h ≲ 1.
4. A system with very fast and very slow behaviour where only the slow part matters. It pushes you to implicit methods.
5. Each prediction is the next input, so errors feed back and can grow over many steps, like an unstable explicit scheme.

</details>

## Words introduced here

thermal diffusivity · semi-discrete form · explicit / implicit time integration · forward Euler / backward Euler / Crank–Nicolson · stability limit · ringing · stiff system · CFL condition · Courant number · mass scaling · rollout stability

## Now read in the guide

- `answers/time-integration-and-stability.md`
- The section *Time-Dependent Problems and Autoregressive Rollouts* in the guide's main README: read the questions now, and the answers after module 15.

**Next:** [07 · Nonlinear problems and Newton–Raphson](../07-newton-raphson/)
