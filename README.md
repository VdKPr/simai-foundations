# simai-foundations (private)

My study path from FEM basics to ML-for-simulation, written in plain language.

The public interview guide (`ai-for-simulation-interview-questions`) is for **revision**: its answers are packed for recall the day before an interview. This repo is for **learning the ideas the first time**, so that the guide reads clearly afterwards.

## Rules this repo follows

1. **Every new term is bolded and explained in the same sentence where it first appears.** No term is used before it is explained.
2. **Mechanical picture first, math second**, and only the math needed to follow the idea.
3. **Every module except 02 has a script to run.** Reading alone does not stick.
4. **Small modules.** One module is one or two evenings.

## How to use a module

1. Read the module's `README.md` (15–30 minutes).
2. Run the script. Look for the things listed under *What to look for*.
3. Answer *Check yourself* before opening the answers.
4. If a term is still foggy, add one line to [`stuck-list.md`](stuck-list.md) and keep going.
5. Read the guide answers listed under *Now read in the guide*. They should make sense now. If one doesn't, that's a stuck-list line too.

## Roadmap

### Part A: FEM and numerics refresh

| # | Module | Guide section it unlocks | Status |
|---|---|---|---|
| 01 | [PDEs in plain words](01-pdes-in-plain-words/) | Simulation Fundamentals | ready |
| 02 | [The weak form: why FEM works](02-weak-form/) | Simulation Fundamentals | ready |
| 03 | [1D bar FEM from scratch](03-bar-fem-1d/) | Simulation Fundamentals; Structural Mechanics (FEA) | ready |
| 04 | [2D heat conduction with triangles](04-heat-2d-triangles/) | Heat Transfer and Multiphysics | ready |
| 05 | [2D elasticity, stress and mesh convergence](05-elasticity-2d/) | Structural Mechanics (FEA) | ready |
| 06 | [Time stepping: explicit vs implicit, stability](06-time-stepping/) | Simulation Fundamentals; Time-Dependent Problems | ready |
| 07 | [Nonlinear problems and Newton–Raphson](07-newton-raphson/) | Simulation Fundamentals; Structural Mechanics (FEA) | ready |
| 08 | [Linear solvers: direct, iterative, preconditioning](08-linear-solvers/) | Simulation Fundamentals | ready |
| 09 | [CFD on one page: Navier–Stokes, finite volumes, Reynolds number, turbulence](09-cfd-one-page/) | Fluid Dynamics (CFD) | ready |

### Part B: ML for simulation

| # | Module | Guide section it unlocks | Status |
|---|---|---|---|
| 10 | [ML for physical data: scaling, splits, error metrics](10-ml-for-physical-data/) | ML Fundamentals for Physical Data; Evaluation and Testing | ready |
| 11 | [Surrogate models and design of experiments](11-surrogates-and-doe/) | Surrogate Models, ROMs and DOE | ready |
| 12 | [Reduced-order models: POD and SVD](12-pod-rom/) | Surrogate Models, ROMs and DOE | ready |
| 13 | [Physics-informed neural networks (PINNs)](13-pinns/) | PINNs | ready |
| 14 | [Neural operators: FNO and DeepONet](14-neural-operators/) | Neural Operators | ready |
| 15 | [Graph networks on meshes](15-graph-nets-on-meshes/) | GNNs and Mesh-Based Learning; Time-Dependent Problems | ready |
| 16 | [Hybrid ML + solver (the RANG pattern)](16-hybrid-ml-solver/) | Simulation Fundamentals (ML-accelerated solvers) | ready |
| 17 | [Uncertainty, out-of-distribution and active learning](17-uncertainty-and-active-learning/) | Uncertainty, Generalization and OOD | ready |

**Skipped on purpose for now:** 3D geometric deep learning, transformers for physics, generative and foundation models, LLM agents, ML engineering and system design. They all build on Part B. Once Part B is done, read those sections straight from the guide.

**Pace:** about 4–5 evenings a week puts Part A at roughly 2–3 weeks and Part B at roughly 3–4 weeks.

## Setup (Windows, Python 3.13)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run any module from its folder, for example:

```bat
cd 03-bar-fem-1d
python bar1d_fem.py
```

Each script prints its results and saves plots in an `out/` folder next to it (ignored by git).

PyTorch is needed for modules 13–15. Install it separately, using the command from pytorch.org (the CUDA build for the RTX 4060). All scripts run on the CPU as written; the slowest (modules 14 and 15) take a few minutes.

## Repo layout

```
simai-foundations/
├── README.md              this file: rules, roadmap, setup
├── stuck-list.md          terms that didn't click yet
├── requirements.txt
├── 01-pdes-in-plain-words/
│   ├── README.md          notes + exercises + check yourself
│   └── fd_1d_heat.py      the script for that module
├── 02-weak-form/          (notes only)
├── 03-bar-fem-1d/
│   ...
└── 17-uncertainty-and-active-learning/
```

Every module folder has the same shape: a `README.md` with the notes, and one script. Scripts save their plots to an `out/` folder next to them.
