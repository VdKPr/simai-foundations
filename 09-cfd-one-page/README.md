# 09 · CFD on One Page

**Time:** one or two evenings · **Code:** `advection_diffusion.py` · **Unlocks:** Fluid Dynamics (CFD)

## Why you care

You come from structures, but a lot of AI-for-simulation work is CFD: external aerodynamics for cars, HVAC, electronics cooling, turbomachinery. Ansys Fluent and Siemens Simcenter STAR-CCM+ are huge businesses. You don't need to become a CFD engineer for these roles. You need the vocabulary and the few ideas that explain why CFD is hard, and why it's hard for ML too.

## The equations in words

For an **incompressible** fluid (constant density, which is fine for liquids and for air below roughly a third of the speed of sound), two equations:

**1. Mass conservation (continuity):** whatever flows into a small box flows out.

```math
\nabla \cdot \mathbf u = 0
```

**2. Momentum (Navier–Stokes):** F = ma for a small parcel of fluid.

```math
\underbrace{\rho\frac{\partial \mathbf u}{\partial t}}_{\text{change in time}} + \underbrace{\rho(\mathbf u\cdot\nabla)\mathbf u}_{\text{carried by the flow}} = \underbrace{-\nabla p}_{\text{pushed by pressure}} + \underbrace{\mu\nabla^2\mathbf u}_{\text{viscous friction}} + \underbrace{\mathbf f}_{\text{gravity etc.}}
```

Three of these terms you already know:

- The time term is module 06.
- The viscous term is diffusion (modules 01, 04), spreading momentum the way conduction spreads heat.
- The pressure term is a force, like the loads in FEA.

The new one is the **convection term** (u·∇)u: the flow carrying its own momentum along. It is **nonlinear** (velocity × velocity gradient, module 07), and it's the source of most of CFD's difficulty.

**Pressure–velocity coupling.** There's no separate equation for pressure. Pressure is whatever it needs to be so that continuity holds. Solvers handle this with iterative schemes such as **SIMPLE**, which alternate between guessing pressure and correcting velocity.

## The Reynolds number

```math
Re = \frac{\rho\,U\,L}{\mu} = \frac{\text{inertia (convection)}}{\text{viscous friction (diffusion)}}
```

U is a typical speed and L a typical size. Some typical values:

| Flow | Re |
|---|---|
| Honey dripping | ≪ 1 |
| Blood in a capillary | ~ 0.001 |
| Air over a car | ~ 10⁶–10⁷ |
| Air over an airliner wing | ~ 10⁷–10⁸ |

As Re grows:

1. Viscous effects squeeze into thin **boundary layers** next to walls. The layer thickness shrinks as Re grows.
2. The flow becomes **turbulent**: chaotic swirls (**eddies**) at a huge range of sizes, from the size of the car down to fractions of a millimetre.

The script shows the first effect in 1D. Its **Péclet number** is the Reynolds number's twin: carrying (convection) vs spreading (diffusion), for a scalar like heat or dye.

## What the script shows

A quantity φ is carried by a flow and diffuses a little. With Pe = 50, all the change squeezes into a layer about 0.02 thick at the outlet, like a boundary layer at a wall.

The **finite volume method** (module 01) balances fluxes for each cell. The flux leaving one cell is exactly the flux entering its neighbour, so mass, momentum and energy are **conserved exactly**. That's the reason CFD uses FVM.

The key choice is how to estimate φ on a cell face:

- **Central:** average of the two neighbouring cells. Second-order accurate.
- **Upwind:** take the value from the cell the flow comes from. First-order accurate.

| Cells | Cell Péclet | Central: max error | Central: min φ | Upwind: max error | Upwind: min φ |
|---|---|---|---|---|---|
| 10 | 5.0 | 1.58 | **−1.50** | 0.20 | 0.000 |
| 20 | 2.5 | 0.54 | **−0.25** | 0.16 | 0.000 |
| 40 | 1.25 | 0.16 | 0.000 | 0.12 | 0.000 |
| 100 | 0.5 | 0.029 | 0.000 | 0.069 | 0.000 |
| 400 | 0.125 | **0.0019** | 0.000 | 0.021 | 0.000 |

The exact φ is always between 0 and 1. Two lessons:

1. **Central wiggles when the cell Péclet number u·h/Γ is above 2.** It produces negative values, which are physically impossible (negative concentration, temperature below both boundaries). The cell is too coarse for the layer.
2. **Upwind never wiggles, but smears.** It acts as if diffusion were larger by u·h/2. At 20 cells, that's 1.25× the real diffusion: the computed layer is a different physics. This is called **numerical diffusion**. On fine meshes, upwind's first-order error is 10× worse than central's.

Real CFD codes use **bounded higher-order schemes** (second-order upwind, limiters) to get the accuracy of the first without the wiggles. And meshes are refined near walls so the cells actually resolve the layer.

## Turbulence: why CFD is expensive

Turbulent eddies span sizes from the whole body down to tiny dissipative scales, and the smallest scale shrinks as Re grows. Three ways to deal with it:

| Approach | What it resolves | Cost | Typical use |
|---|---|---|---|
| **DNS** (direct numerical simulation) | every eddy | cost grows ~Re³; impossible at car scale | research, low Re |
| **LES** (large eddy simulation) | large eddies; small ones modelled | 10–100× RANS | acoustics, unsteady wakes, combustion |
| **RANS** (Reynolds-averaged Navier–Stokes) | only the average flow; all turbulence modelled | cheapest | most industrial CFD |

A **turbulence model** (for RANS: Spalart–Allmaras, k-ε, k-ω SST) adds extra equations that estimate how much extra mixing the unresolved eddies cause. That mixing is called **eddy viscosity**. Most ML training data from industry is RANS, so it carries RANS model error.

**y+** is the distance from the wall to the first cell centre, in units of the viscous layer thickness. Two strategies:

- **Resolve the layer:** y+ ≈ 1 near the wall. Accurate, needs many thin cells.
- **Wall functions:** y+ ≈ 30–300, and a formula bridges the gap. Cheaper, less accurate in separated flow.

**Flow separation** is when the boundary layer detaches from the surface (behind a car, a stalled wing). It's sensitive, unsteady and hard to predict, for RANS models and for ML surrogates alike.

## Run it

```bat
python advection_diffusion.py
```

### What to look for

1. The **min φ** column going negative for central at 10 and 20 cells, and staying at 0 for upwind.
2. At 400 cells: central is 10× more accurate than upwind.
3. **The plot at 20 cells:** central's last point plunges below zero; upwind's curve rises too early (smeared).

## Exercises

1. Raise `PE` to 500. How many cells does central need for cell Péclet < 2? What does that say about meshing a real boundary layer at Re = 10⁶?
2. At 20 cells, upwind's layer is visibly too thick. Using the "extra diffusion u·h/2" idea, what Péclet number is upwind effectively solving for?
3. A colleague's ML surrogate for car drag was trained on RANS k-ω SST data. Name two sources of error in its predictions that have nothing to do with the neural network.

<details>
<summary>Answers</summary>

1. Cell Pe = u·h/Γ = Pe·h/L < 2, so h < 2L/500 = 0.004 L: more than 250 uniform cells for a 1D layer. At Re = 10⁶, boundary layers are far thinner. That's why CFD meshes use stretched, very thin cells near walls (inflation layers), and why grid size explodes in 3D.
2. Total diffusion = Γ + u·h/2 = 0.02 + 0.025 = 0.045, so the effective Pe = u·L/0.045 ≈ 22 instead of 50. Upwind on a coarse mesh is solving a different, more diffusive problem.
3. Any two of: turbulence-model error (RANS's own approximation of the real flow); mesh discretization error, especially near walls and in separated regions; numerical diffusion from the scheme; and steady RANS used for a flow that is really unsteady (wake shedding). The surrogate can't be more accurate than its labels.

</details>

## Where this connects to ML

- **Labels carry solver error.** CFD training data has turbulence-model, mesh and scheme errors baked in. A surrogate learns them as if they were physics.
- **Thin layers and separation are hard for ML**, for the same reason as hyperbolic fronts (module 01): sharp features and spectral bias. Surrogates are usually least accurate exactly where engineers care most: near walls and in separated wakes.
- **Conservation is a useful check and constraint.** FVM conserves exactly; a neural network doesn't, unless it's built to. Checking mass balance on a surrogate's predicted velocity field is a quick test. The guide covers how to enforce a divergence-free field exactly.

## Check yourself

1. What does each term of the incompressible Navier–Stokes equation represent?
2. What does the Reynolds number compare, and what happens physically as it grows?
3. Why does CFD use the finite volume method?
4. Central vs upwind: what goes wrong with each on a coarse mesh?
5. RANS vs LES vs DNS: what does each resolve?

<details>
<summary>Answers</summary>

1. Time change and convection (the fluid's acceleration) = pressure force + viscous friction + body force. Plus continuity: what flows in flows out.
2. Inertia vs viscous friction. As it grows: thinner boundary layers, then turbulence.
3. It conserves mass, momentum and energy exactly at the cell level: the flux out of one cell is the flux into the next.
4. Central oscillates (non-physical over- and undershoots) when cell Péclet > 2. Upwind adds numerical diffusion, smearing sharp layers.
5. DNS: all eddies. LES: large eddies, small ones modelled. RANS: only the mean flow, all turbulence modelled.

</details>

## Words introduced here

incompressible flow · continuity · Navier–Stokes · convection term · pressure–velocity coupling · SIMPLE · Reynolds number · Péclet number · boundary layer · turbulence · eddies · finite volume method · conservation · central / upwind schemes · cell Péclet number · numerical diffusion · bounded higher-order schemes · DNS / LES / RANS · turbulence model · eddy viscosity · y+ · wall functions · flow separation

## Now read in the guide

- `answers/cfd-essentials.md`
- `answers/turbulence-modelling.md`
- `answers/boundary-layers-and-wall-treatment.md`
- `answers/fem-vs-fvm-vs-fdm.md` again: it will read differently now.

**That's the end of Part A.** Next: [10 · ML for physical data](../10-ml-for-physical-data/)
