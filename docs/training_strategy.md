# V27 Final Training Strategy and Network Configuration Report

## 1. Research Goal

The final v27 experiment validates a fine-grained MPU replacement mechanism for multi-task photonic neural networks. The hardware motivation is that a passive 2-to-2 meta processing unit (MPU) can replace a conventional reconfigurable MZI with much smaller footprint and lower power. The cost is that the MPU is fixed after fabrication.

In the experiment, a shared MPU is still trainable during offline design. After the sharing pattern and common 2-to-2 transfer matrix are learned, the corresponding MZI neuron is interpreted as a passive MPU in deployment.

The core question is:

```text
At the same shared MPU replacement ratio, can Neuron-level replacement preserve more multi-task accuracy than coarse layer/stage-level replacement?
```

## 2. Final Network Structure

The final model uses a single long Clements mesh rather than a four-layer nonlinear optical stack.

```text
EMNIST image 28x28
-> fixed PCA preprocessing, 784 -> 64
-> one coherent 64x64 complex Clements mesh
-> one square-law intensity readout |E|^2
-> shared LayerNorm
-> shared GELU
-> task-specific electrical heads
```

The three tasks are:

```text
Task 1: Digits, 10 classes
Task 2: Uppercase letters, 26 classes
Task 3: Lowercase letters, 26 classes
```

The task-specific heads are used to prevent output-space competition from masking the optical sharing mechanism. The claim is restricted to the photonic mesh replacement strategy.

## 3. Why Single Mesh

Earlier four-layer 64-to-64 photonic stacks were less suitable for the final mechanism study because they introduced extra variables:

- repeated intensity detection and nonlinearities,
- multiple normalization layers,
- a strong early-layer sharing prior,
- larger electrical components that could dominate the optical mechanism.

The single-mesh setting directly matches the physical device question: each MPU replaces one 2-to-2 MZI neuron in a Clements interferometer.

## 4. Photonic Parameterization

The optical core is one complex Clements MZI mesh with 64 modes. It uses alternating even and odd 2-to-2 stages.

In code, one `SharedClementsMZILayer` contains:

```text
theta_even, phi_even: 32 x 32
theta_odd,  phi_odd : 32 x 31
```

Total number of 2-to-2 units:

```text
32 * 32 + 32 * 31 = 2016
```

Each sharing mask entry controls a complete 2-to-2 unit. Therefore, theta and phi share one mask. This matches the physical interpretation that an MPU replaces a full 2-to-2 transfer matrix, not an isolated phase parameter.

## 5. Shared MPU and Task-Private MZI Semantics

Each MZI neuron has:

```text
common/shared phase parameters
task-specific phase deltas
```

If a neuron is assigned as a shared MPU:

```text
shared_mask = 1
task-specific deltas are disabled
all tasks use the same theta/phi
```

If a neuron is task-private:

```text
shared_mask = 0
task-specific deltas remain enabled
each task can adjust the local 2-to-2 transform
```

The final visualization convention is:

```text
grey   = shared MPU
purple = task-private MZI
```

## 6. Warmup Stage

The first stage is a 0% shared warmup.

```text
warmup shared ratio = 0%
all MZI neurons are task-private
task-specific deltas are enabled across the full mesh
```

Warmup trains a high-performing task-private photonic model and exposes which MZI neurons are common or task-specific across tasks.

Saved warmup data include:

```text
mean_hist
task_hist
train_loss_hist
train_mean_hist
train_task_hist
best checkpoint
mask matrices
confusion matrices
```

The full warmup curves are stored in:

```text
data/warmup_main.json
data/warmup_replicate.json
```

## 7. Sharing Pattern Search

The final Neuron-level strategy uses a stage-prior refinement scheme.

Layer-level sharing uses the strong physical prior:

```text
share early physical stages
keep late stages task-private
```

This is a meaningful baseline because early transformations are expected to contain more common representations, while later transformations are more task-specific.

Neuron-level sharing starts from the same front-stage prior and then refines it:

1. Train a 0% shared warmup model.
2. Estimate Neuron-level sharing suitability from the warmup model.
3. Identify front-stage neurons that are less suitable for sharing.
4. Swap them back to task-private MZI.
5. Replace more suitable neurons elsewhere with shared MPUs.
6. Keep the total shared MPU ratio fixed.

Thus Neuron-level sharing is not compared against an artificially weak baseline. It improves on a strong layer/stage prior while using the same replacement budget.

## 8. neuron Scoring

The final scoring uses a Fisher-style replacement damage criterion. The goal is to identify neurons whose task-specific phase offsets can be removed with minimal estimated loss increase.

A good shared MPU candidate should:

- have low task-specific replacement damage,
- show compatible behavior across tasks,
- preserve common optical transformations.

A good task-private MZI candidate should:

- be sensitive to task-specific phase changes,
- show task-dependent behavior,
- be costly to force into a common transfer matrix.

This converts the hardware placement question into a Neuron-level commonness selection problem.

### 8.1 Effective task-specific phase of each 2-to-2 unit

For a MZI neuron \(i\), the trainable common phase parameters are \(\theta_i\) and \(\phi_i\). During warmup, each task \(t\) also has task-specific phase offsets \(\Delta\theta_{t,i}\) and \(\Delta\phi_{t,i}\). The effective task-specific MZI setting is therefore

```math
\theta_{t,i}^{\mathrm{eff}} = \theta_i + \Delta\theta_{t,i},
\qquad
\phi_{t,i}^{\mathrm{eff}} = \phi_i + \Delta\phi_{t,i}.
```

If neuron \(i\) is replaced by a shared MPU, all tasks must use one common 2-to-2 transfer matrix. In phase form, this means replacing the task-specific settings \(\{\theta_{t,i}^{\mathrm{eff}},\phi_{t,i}^{\mathrm{eff}}\}_t\) by a common circular mean.

Because phases are periodic, the common phase is computed as a circular mean rather than an arithmetic mean:

```math
\bar{\theta}_i
= \mathrm{atan2}\left(
    \frac{1}{T}\sum_{t=1}^{T}\sin\theta_{t,i}^{\mathrm{eff}},
    \frac{1}{T}\sum_{t=1}^{T}\cos\theta_{t,i}^{\mathrm{eff}}
\right),
```

```math
\bar{\phi}_i
= \mathrm{atan2}\left(
    \frac{1}{T}\sum_{t=1}^{T}\sin\phi_{t,i}^{\mathrm{eff}},
    \frac{1}{T}\sum_{t=1}^{T}\cos\phi_{t,i}^{\mathrm{eff}}
\right),
```

where \(T=3\) in the final EMNIST setting.

### 8.2 Cross-task phase agreement

The first criterion measures whether the tasks already prefer similar phase settings at neuron \(i\). For each phase, the circular concentration is

```math
R_{\theta,i}
=
\left|
\frac{1}{T}\sum_{t=1}^{T} e^{j\theta_{t,i}^{\mathrm{eff}}}
\right|,
\qquad
R_{\phi,i}
=
\left|
\frac{1}{T}\sum_{t=1}^{T} e^{j\phi_{t,i}^{\mathrm{eff}}}
\right|.
```

Here \(R=1\) means all tasks prefer the same phase, while \(R\approx 0\) means their preferred phases are dispersed around the unit circle. We combine the two phase agreements as

```math
A_i = \frac{1}{2}(R_{\theta,i} + R_{\phi,i}),
```

and use a softened consensus weight

```math
w_i^{\mathrm{cons}}
= 0.10 + 0.90 A_i^2.
```

The small offset prevents the score from becoming exactly zero for moderately conflicting neurons, while still strongly favoring high-agreement neurons.

### 8.3 Fisher importance from warmup gradients

The second criterion measures whether neuron \(i\) matters for the tasks. For each task, gradients are accumulated over a small number of training batches from the warmup model. The Neuron-level Fisher proxy is

```math
F_{t,i}
=
\left(\frac{\partial \mathcal{L}_t}{\partial \theta_i}\right)^2
+
\left(\frac{\partial \mathcal{L}_t}{\partial \phi_i}\right)^2,
```

where \(\mathcal{L}_t\) is the cross-entropy loss of task \(t\). In implementation, this is computed separately for even and odd Clements-stage tensors and for every 2-to-2 neuron.

The average Fisher utility is

```math
U_i
=
\sqrt{
\frac{1}{T}\sum_{t=1}^{T}F_{t,i}
+\epsilon
}.
```

The square root keeps the dynamic range moderate. We normalize it by its mesh-wide mean:

```math
\tilde{U}_i = \frac{U_i}{\frac{1}{N}\sum_{k=1}^{N}U_k+\epsilon},
```

where \(N\) is the number of candidate MZI neurons in the mesh.

### 8.4 Estimated replacement damage

A neuron should not be selected as a shared MPU merely because it has large gradients. If different tasks need different phase settings at that neuron, replacing it with one common MPU could be damaging. Therefore, the method estimates the loss increase caused by collapsing all task-specific phase settings to the circular mean.

The phase mismatch for task \(t\) is computed with wrapped angular distance:

```math
d_{t,i}
=
\frac{1}{2}
\left[
\mathrm{wrap}\left(\theta_{t,i}^{\mathrm{eff}}-\bar{\theta}_i\right)^2
+
\mathrm{wrap}\left(\phi_{t,i}^{\mathrm{eff}}-\bar{\phi}_i\right)^2
\right],
```

where

```math
\mathrm{wrap}(x)=\mathrm{atan2}(\sin x,\cos x).
```

Using the diagonal Fisher approximation, the replacement damage is estimated as

```math
D_i
=
\frac{1}{T}
\sum_{t=1}^{T}
F_{t,i} d_{t,i}.
```

This is the key term: a neuron is considered risky to replace if it is both important and task-specific. The damage is normalized by the mean damage over all neurons:

```math
\tilde{D}_i
=
\frac{D_i}{\frac{1}{N}\sum_{k=1}^{N}D_k+\epsilon}.
```

### 8.5 Final commonness score

The final score used by the `fisher_damage` mode is

```math
S_i
=
w_i^{\mathrm{cons}}
\frac{0.25+\tilde{U}_i}{0.25+\tilde{D}_i}.
```

The numerator favors neurons that are useful to the learned optical representation. The denominator penalizes neurons whose task-specific phase offsets would cause high estimated loss if removed. The consensus term favors neurons whose learned phase settings are already aligned across tasks.

Thus, high-scoring neurons are not merely unimportant neurons. They are neurons that are useful, cross-task consistent, and cheap to collapse into a common 2-to-2 transfer matrix. These are the preferred shared MPU candidates.

### 8.6 Stage-prior refinement rule

For a target shared ratio \(r\), the layer-level baseline selects the first physical stages until approximately \(rN\) neurons are shared. This produces a binary prior mask \(M_i^{\mathrm{stage}}\).

The Neuron-level method starts from the same mask, then performs a score-based refinement:

1. Rank currently shared front-stage neurons by \(S_i\).
2. Rank currently private neurons outside the selected stage region by \(S_i\).
3. Swap out low-score shared neurons and swap in high-score private neurons when the outside neuron has a higher score.
4. Stop when no beneficial swap is available or the replacement budget is met.

The number of shared neurons is preserved:

```math
\sum_i M_i^{\mathrm{neuron}} = \sum_i M_i^{\mathrm{stage}} \approx rN.
```

Therefore, any accuracy gain is attributed to better placement of the same number of shared MPU units, not to a smaller replacement ratio or a larger number of task-private parameters.

## 9. Fine-Tuning After Pattern Imposition

After the sharing mask is fixed, the model is initialized from the warmup checkpoint and fine-tuned.

For a given shared MPU ratio:

```text
shared MPU neurons: task-specific deltas disabled
task-private MZI neurons: task-specific deltas enabled
```

The mask is fixed during fine-tuning. The trainable parameters are the allowed common/shared phases, the allowed task-specific deltas, the shared normalization parameters, and the task-specific heads.

In deployment, the learned common theta/phi values of the shared neurons are interpreted as the fixed MPU transfer matrices.

## 10. Boundary Conditions

The 0% and 100% shared cases define the natural bounds:

```text
0% shared:
all neurons are task-private MZIs
maximum task-specific optical flexibility
upper accuracy bound

100% shared:
all neurons are shared MPUs
no task-specific optical deltas
fully shared lower bound
```

At 0% and 100%, Neuron-level and layer-level strategies are equivalent because there is no pattern selection freedom.

## 11. Main Result

The final main curve uses the 10%-step main sweep:

```text
shared ratio: 0, 10, 20, ..., 100%
seed: 44
train samples: 40000
test samples: 12000
warmup epochs: 80
fine-tuning epochs: 80
input rank: 64
hidden dimension: 64
number of photonic layers: 1
```

At 90% shared MPU replacement:

```text
Neuron-level sharing:  87.64% average test accuracy
layer-level sharing: 80.37% average test accuracy
gain: +7.26 percentage points
```

Task-wise 90% Neuron-level accuracy:

```text
Digits:    93.5%
Uppercase: 85.7%
Lowercase: 83.6%
```

Task-wise 90% layer-level accuracy:

```text
Digits:    87.4%
Uppercase: 77.2%
Lowercase: 76.5%
```

## 12. Interpretation

The result supports the following mechanism:

```text
Neuron-level sharing does not use more parameters or a lower replacement ratio.
It uses the same shared MPU budget but places the remaining task-private MZIs more selectively.
```

At high replacement ratios, the few remaining task-private MZI neurons become especially valuable. Coarse layer-level sharing concentrates them at the mesh tail. Neuron-level sharing redistributes them to preserve task-specific optical transformations wherever they are most needed.

This explains why the Neuron-level advantage is small at low MPU ratios but becomes large at 80-90% replacement.

## 13. Reproducibility Files

Core training:

```text
run_main_freeze_ratio_experiment.py
```

Main server scripts:

```text
scripts/run_main_10pct_sweep_reuse_warmup.sh
scripts/run_replicate_20pct_sweep.sh
```

Plotting and data export:

```text
plot_v27_nature_figures.py
plot_v27_single_mesh_architecture.py
export_v27_nature_figure_data.py
```

Main result data:

```text
data/main_10pct_sweep.json
data/warmup_main.json
```

Figure data package:

```text
paper_data_v27_nature_figures/
```

