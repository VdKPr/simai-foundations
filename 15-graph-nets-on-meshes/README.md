# 15 · Graph Networks on Meshes

**Time:** two evenings · **Code:** `gnn_mesh.py` (PyTorch + scikit-fem, about 2–4 minutes on CPU) · **Unlocks:** Graph Neural Networks and Mesh-Based Learning; Time-Dependent Problems and Autoregressive Rollouts; Must Know "Why are GNNs a natural fit for simulation meshes?"

## Why you care

Real FEA and CFD meshes are **unstructured**: triangles and tetrahedra of different sizes, refined where the physics is sharp. The FNO from module 14 needs a regular grid for its FFT, and an MLP needs a fixed number of inputs. A **graph neural network (GNN)** works directly on the mesh as it is: nodes are mesh nodes, edges are element edges.

DeepMind's **MeshGraphNets** (2021) made this mainstream, and today's large models for car aerodynamics and crash use the same ideas. "Why do GNNs fit meshes?" and "why do rollouts drift?" are core interview questions.

## Message passing in plain words

A GNN repeats one move, **message passing**, a few times:

1. **Messages:** for every edge, a small network computes a message from the two nodes at its ends and the edge's own features (the edge vector, its length, the temperature difference along it).
2. **Aggregate:** every node sums the messages arriving from its neighbours.
3. **Update:** every node updates its state from its old state and that sum.

After k rounds, each node has information from k edges away. The same small networks are used at every node and edge, so the model works on **any mesh, with any number of nodes**.

**You've seen this before.** In FEM assembly (module 03), each node gathers contributions from its neighbouring elements. An explicit time step (module 06) updates each node from its neighbours. A GNN is a *learned* version of that local stencil. That's the honest answer to "why do GNNs fit meshes": the physics is local, and so is the network.

## The experiment

**Ground truth:** transient heat conduction on random triangle meshes of the unit square (edges held at 0 °C), starting from 1–3 random hot spots, solved with scikit-fem and implicit Euler.

**The GNN learns one time step:** given the temperature at every node, predict the change over the next step. 3 rounds of message passing, 32-number node states.

- **Training data:** 16 random meshes × 20 single steps.
- **Two versions:** one trained on clean inputs; one with small random **noise added to its inputs** during training (the MeshGraphNets trick).
- **Test:** two meshes it has never seen. One as dense as the training meshes (256 nodes), one about twice as dense (456 nodes). The GNN is **rolled out** for 100 steps: each prediction becomes the next input. Training only ever saw single steps, up to step 40.

## Results from the script

**Unseen mesh, same density (256 nodes), relative field error:**

| | step 1 | step 10 | step 25 | step 50 | step 100 |
|---|---|---|---|---|---|
| GNN, clean inputs | 3.8% | 14.5% | 17.5% | 66.8% | **663%** |
| GNN, noisy inputs | 3.9% | 14.4% | 14.7% | 15.6% | **43%** |
| no change (baseline) | 11.5% | 74.5% | 130% | 203% | 384% |

**Unseen mesh, 2× denser (456 nodes):**

| | step 1 | step 10 | step 25 | step 50 | step 100 |
|---|---|---|---|---|---|
| GNN, clean inputs | 5.2% | 28.8% | 44.6% | 62.1% | 201% |
| GNN, noisy inputs | 5.2% | 28.1% | 41.9% | 58.9% | 129% |
| no change (baseline) | 11.2% | 84.5% | 170% | 302% | 622% |

(The errors are relative to the true field, which cools and shrinks over time, so late-step percentages look big. Compare rows, not absolute values.)

What it shows:

1. **It works on meshes it has never seen.** One-step error is 4–5%, on a different mesh with a different number of nodes. An MLP or FNO can't even be applied there.
2. **Errors compound in a rollout.** The clean model's error grows from 4% at step 1 to 663% at step 100 on the same-density mesh. Each step starts from the previous step's slightly wrong output, the same mechanism as the unstable explicit scheme in module 06. Look at the right-hand plot: the blue curve takes off after step 30.
3. **Noise training stabilizes the rollout.** Same architecture, same data, only noisy inputs during training: 43% instead of 663% at step 100, and almost flat between steps 10 and 50. The model learns to pull slightly wrong inputs back towards physical fields, which is exactly the situation it meets in a rollout.
4. **A denser mesh than any in training is out of distribution.** Edge lengths are shorter than anything seen in training, and everything gets worse; noise training helps less. Training on a range of mesh densities is the fix.

**Honest scale check.** This is a tiny model trained in about a minute on a CPU. Production mesh models use 10–15 message-passing rounds, much larger node states, far more data, and days of GPU training. The *behaviours* (generalizing across meshes, rollout drift, the noise trick) are the same ones you'd see at full scale.

## Run it

```bat
python gnn_mesh.py
```

About 2–4 minutes on CPU (training two GNNs).

### What to look for

1. The step-1 column: small errors on both unseen meshes.
2. The clean vs noisy rows diverging after step 25.
3. **Plots:** truth vs GNN at step 25 (similar shapes, blotchy error map), and the error-growth curves.

## Exercises

1. Reduce message passing to 1 round (`MeshGNN(steps=1)` inside `train`). What happens to the one-step error, and why might it?
2. Make the noise much bigger: `0.03` instead of `0.003`. Predict what happens to the one-step error and to the long rollout before you run it.
3. Your team wants to train a crash GNN on 300 car simulations and use it for 200 ms rollouts. Name three risks you'd raise.

<details>
<summary>Answers</summary>

1. On the same-density mesh, the first-step error rises from 3.8% to 6.0%, and the rollout error at step 10 roughly doubles (14.5% → 27.6%). With one round, each node only sees its direct neighbours, but the implicit time step spreads heat further than one edge (module 06: implicit steps are non-local). The sharp early fields suffer most; on later, smoother fields the difference is small.
2. One-step errors on later, smoother fields jump from about 1% to 6–7%: the model learns to "correct" inputs that were actually right. The 100-step rollout (98%) is still far better than the clean model (663%) but worse than with noise 0.003 (43%). The noise level is a tuning knob, best matched to the size of the model's own one-step errors.
3. Any three of: rollout error compounding over thousands of steps; sharp, history-dependent physics (contact, plasticity, fracture) that GNNs and smooth models handle poorly (modules 01, 05, 07); 300 cars is little data for a new car design that is out of distribution; mesh density and element types differing from training; and validating only on the integrated quantities that matter (intrusion, peak acceleration), not just field errors.

</details>

## Where this connects

- **Module 03 / 06:** assembly and explicit stencils are hand-written message passing.
- **Module 06:** rollout drift is the learned version of explicit instability.
- **Module 14:** FNO for regular grids, GNN for irregular meshes. Graph-based neural operators and transformers on point clouds blend the two (the guide's 3D geometry and transformer sections).
- **Module 17:** the denser-mesh test is an OOD test. Detect it before trusting the model.

## Check yourself

1. What happens in one round of message passing?
2. Why can one trained GNN be applied to meshes with different numbers of nodes?
3. Why do autoregressive rollouts drift even when one-step errors are small?
4. What does training with noisy inputs do, and why does it help?
5. Why was the 2× denser mesh harder?

<details>
<summary>Answers</summary>

1. Each edge computes a message from its two end nodes and its own features; each node sums its incoming messages; each node updates its state from its old state and that sum.
2. The same small networks are applied at every node and edge, so the number of nodes doesn't matter.
3. Each step's output is the next step's input, so errors feed back and can grow, just like an unstable explicit time integrator.
4. It shows the model slightly wrong inputs during training, so it learns to correct them instead of amplifying them. That's exactly what happens during a rollout.
5. Its edges are shorter than any in training, so the model must extrapolate in edge geometry (out of distribution). Training on a range of densities fixes that.

</details>

## Words introduced here

unstructured mesh · graph neural network (GNN) · nodes / edges / edge features · message passing · aggregation · MeshGraphNets · one-step model · autoregressive rollout · rollout drift · noise injection · out-of-distribution mesh

## Now read in the guide

- `answers/gnns-and-meshgraphnets.md`
- The guide section *Graph Neural Networks and Mesh-Based Learning*: all questions
- The guide section *Time-Dependent Problems and Autoregressive Rollouts*: now read the answers too

**Next:** [16 · Hybrid ML + solver](../16-hybrid-ml-solver/)
