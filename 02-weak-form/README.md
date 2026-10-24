# 02 · The Weak Form: Why FEM Works

**Time:** one evening · **Code:** none here (module 03 turns this page into code) · **Unlocks:** Simulation Fundamentals

## Why you care

The weak form is the one idea that FEM is built on. It also sorts the ML methods: a standard PINN uses the strong form, while Deep Ritz and variational PINNs use the weak or energy form. "What's the difference, and why does it matter?" is a common interview question.

You already know this idea under another name: **the principle of virtual work**.

## The problem with the strong form

FEM builds the displacement field out of straight-line pieces, one per element. This is called **piecewise linear**.

Look at what a straight-line-pieces function does to derivatives:

| | Inside an element | At a node between two elements |
|---|---|---|
| u | straight line | continuous (pieces meet) |
| u′ (strain) | constant | jumps |
| u″ | zero | does not exist (infinite spike) |

The strong form −(EA u′)′ = f needs u″ at every point, so a piecewise-linear u can never satisfy it. We need a version of the equation that only asks for u′.

## The fix: virtual work

The principle of virtual work states that a body is in equilibrium if, for **every** small virtual displacement δu that respects the supports:

> internal virtual work = external virtual work

For the bar (fixed at x = 0, distributed load f, tip load P):

```math
\int_0^L EA\,u'\,\delta u'\,dx \;=\; \int_0^L f\,\delta u\,dx \;+\; P\,\delta u(L)
```

- **Left:** internal force (EA u′) × virtual strain (δu′), summed along the bar.
- **Right:** applied loads × the virtual displacement where they act.

**This is the weak form.** Mathematicians call δu the **test function** and write it as v, but it is the same thing. Only first derivatives appear, so piecewise-linear functions are now fine.

## Where it comes from (three steps)

If you want to see that it really is the same equation as the strong form:

1. **Multiply** the strong form −(EA u′)′ = f by a test function v(x) that is zero at the support, v(0) = 0. (A virtual displacement can't move a support.)
2. **Integrate** over the length of the bar.
3. **Integrate by parts.** This moves one derivative from u onto v, and leaves a boundary term [EA u′ v] evaluated at x = 0 and x = L. At x = 0, v = 0, so that end drops out. At x = L, EA u′(L) is the tip force P, so the term becomes P v(L).

The result is exactly the virtual-work equation above.

### Two kinds of boundary conditions fall out of this

- The tip force P **appeared by itself** through the boundary term. Force and flux conditions (Neumann) are called **natural boundary conditions** because the weak form takes care of them automatically.
- The support u(0) = 0 did **not** appear anywhere. It has to be forced onto the solution directly. Fixed-value conditions (Dirichlet) are called **essential boundary conditions**.

"Weak" does not mean less accurate. It means the equation must hold *on average against every test function* rather than exactly at every point. If it holds for every possible test function, the solution matches the strong form wherever the strong form makes sense.

## Galerkin: from weak form to K u = F

The weak form is still about infinitely many test functions. To get something a computer can solve:

**1. Build u out of shape functions.**

```math
u(x) \approx \sum_j u_j \, N_j(x)
```

Each **shape function** N_j is a "hat": 1 at node j, 0 at every other node, straight lines in between. u_j is the displacement at node j, the unknown we solve for.

**2. Use the same shape functions as test functions.** Write one virtual-work equation per node, with v = N_i. Using the same functions for the solution and the test is called **Galerkin's method**.

**3. Substitute.** Each equation i becomes

```math
\sum_j \underbrace{\left[\int_0^L EA\,N_i'\,N_j'\,dx\right]}_{K_{ij}} u_j \;=\; \underbrace{\int_0^L f\,N_i\,dx \;+\; P\,N_i(L)}_{F_i}
```

That is **K u = F**: K is the **stiffness matrix** and F is the **load vector**.

K_ij is non-zero only if hats i and j overlap, which means nodes i and j share an element. So every node is coupled only to its neighbours, and K is sparse. This is the same sparsity you saw in module 01.

## The energy view (same answer, another angle)

For linear elastic problems, the weak form is equivalent to **minimizing the total potential energy**:

```math
\Pi(u) = \underbrace{\tfrac{1}{2}\int_0^L EA\,(u')^2\,dx}_{\text{strain energy}} \;-\; \underbrace{\int_0^L f\,u\,dx \;-\; P\,u(L)}_{\text{work done by the loads}}
```

Equilibrium is the displacement field with the lowest Π. Setting the change in Π to zero for every small variation v gives back the weak form. "Weak form", "virtual work" and "minimum potential energy" are three names for the same physics.

## Why this matters for ML

| Method | What it makes small | Form used |
|---|---|---|
| FEM | solves the weak form exactly within the chosen shape functions | weak |
| **PINN** | squared strong-form residual at sample points | strong: needs u″ through automatic differentiation |
| **Deep Ritz** | the energy Π, with a neural network as u | energy |
| **Variational PINN (VPINN)** | the weak-form residual against a set of test functions | weak |

A **PINN (physics-informed neural network)** is a neural network trained so that its output satisfies the PDE. Module 13 covers it properly.

**One interview line:** "FEM and Deep Ritz use the weak or energy form, so they need fewer derivatives and get force boundary conditions for free. A standard PINN uses the strong form, so it needs higher derivatives through autodiff and has to add the boundary conditions as penalty terms in its loss."

**Your thesis link.** The EFK equation is fourth order. Even its weak form still needs second derivatives, which piecewise-linear elements don't have. That's why you used a **mixed formulation**: split one fourth-order equation into two second-order equations, each of which works with ordinary elements.

## Common confusions

- **"Test function" is not a test in the software-testing sense.** It's a probe: a virtual displacement used to check balance.
- **Galerkin vs FEM.** Galerkin is the general idea (use the same functions for the solution and the test). FEM is Galerkin with local hat-shaped functions on a mesh.
- **Weak ≠ approximate.** The weak form is exact. The approximation only enters when you restrict u to a finite set of shape functions.

## Check yourself

1. Why can't a piecewise-linear displacement field satisfy the strong form?
2. In the weak form of the bar, where does the tip load P come from?
3. Which boundary condition must be imposed by hand in FEM: a fixed support or an applied pressure? What is the name for that type?
4. Why is the stiffness matrix sparse?
5. Name one way a standard PINN differs from FEM in how it uses the PDE.

<details>
<summary>Answers</summary>

1. Its second derivative is zero inside each element and doesn't exist at the nodes, but the strong form needs u″ everywhere.
2. From the boundary term produced by integration by parts, evaluated at x = L.
3. The fixed support. It is an essential (Dirichlet) condition.
4. K_ij = ∫ EA N_i′ N_j′ dx is non-zero only when the hat functions overlap, which happens only for nodes that share an element.
5. Any one of: a PINN minimizes the strong-form residual, so it needs higher derivatives (via autodiff); its boundary conditions are penalty terms in the loss, not built into the solution; it samples points instead of integrating over elements.

</details>

## Words introduced here

piecewise linear · principle of virtual work · weak form · test function · natural / essential boundary conditions · shape function · Galerkin's method · stiffness matrix · load vector · minimum potential energy · PINN · Deep Ritz · VPINN · mixed formulation

## Now read in the guide

- `answers/weak-form-and-galerkin.md`. Skip the *Petrov-Galerkin and stabilization* section for now.
- `answers/elements-and-shape-functions.md`, the section *Shape functions* only

**Next:** [03 · 1D bar FEM from scratch](../03-bar-fem-1d/)
