# TRACER: Persistent Regularization for Robust Multimodal Finetuning

Hesam Asadollahzadeh 1 Feng Liu 1 Christopher Leckie 1 Sarah M. Erfani 1

# Abstract

Mainstream strategies for finetuning pretrained multimodal models often degrade out-ofdistribution (OOD) robustness, a phenomenon known as catastrophic forgetting. In this paper, we develop a theoretical framework for multimodal contrastive finetuning, yielding closed-form solutions and a geometric decomposition for each strategy. This framework shows that self-distillation is more effective than other regularization approaches to retain the knowledge of the pretrained model. Our analysis reveals a largely overlooked limitation: standard Exponential Moving Average (EMA) teachers, widely used in robust finetuning, suffer from collapse. To solve this, we prove that a Weighted Moving Average (WMA) teacher maintains a persistent regularizing force over finite horizons and yields bias-free convergence in the task subspace while preserving orthogonal knowledge. These insights motivate TRACER (Trajectory-Robust Anchoring for Contrastive Encoder Regularization), which combines contrastive learning with WMA-guided multi-perspective distillation. Extensive experiments on CLIP finetuning demonstrate consistent OOD accuracy and calibration gains across three backbone architectures, and comprehensive ablations confirm that TRACER is both principled and robust to hyperparameter choices. Code is available at https: //github.com/HesamAsad/TRACER.

# 1. Introduction

Pretrained models such as CLIP (Radford et al., 2021) have revolutionized machine learning through their remarkable zero-shot transfer and adaptive capabilities. These models

1School of Computing and Information Systems (CIS), Faculty of Engineering and IT (FEIT), University of Melbourne, Australia. Correspondence to: Hesam Asadollahzadeh <h.asadollahzadeh@unimelb.edu.au>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

derive their robustness from large-scale multimodal pretraining (Fang et al., 2022; Xu et al., 2024b), enabling diverse applications from visual recognition (Shen et al., 2022b; Zhang et al., 2022b) to generative modeling (Betker et al., 2023; Pi et al., 2024) and serving as backbones for large multimodal models (Alayrac et al., 2022; Liu et al., 2023; Zhu et al., 2024).

Despite these successes, adapting these pretrained models to downstream tasks via finetuning presents a fundamental challenge: while finetuning improves in-distribution (ID) performance, it often degrades out-of-distribution (OOD) robustness (Radford et al., 2021). This trade-off manifests as catastrophic forgetting of pretrained knowledge (Wortsman et al., 2022b), where models sacrifice their general-purpose representations to optimize for task-specific patterns, potentially overfitting to spurious correlations in the finetuning data.

Several empirical strategies have emerged to mitigate this trade-off. For example, LP-FT (Kumar et al., 2022) addresses the problem of randomly initialized heads distorting pretrained features by first learning a linear probe on frozen features before full finetuning. FLYP (Goyal et al., 2023) extends this idea by reusing CLIP’s pretrained text encoder as the classification head, maintaining consistency with the pretraining objective. Post-hoc methods like WiSE-FT (Wortsman et al., 2022b) and Model Stock (Jang et al., 2024) perform weight averaging between pretrained and finetuned models to recover lost robustness. Regularizationbased approaches, including $\small { \cal L } _ { 2 } { \mathrm { - } } { \mathrm { S P } }$ (Li et al., 2018) and self-distillation with dynamic teachers (Oh et al., 2024), introduce constraints to preserve pretrained knowledge. However, most dynamic-teacher methods rely on an Exponential Moving Average (EMA), whose regularizing influence provably weakens as the teacher approaches the student. This limitation is often overlooked in the robust-finetuning literature on catastrophic forgetting, yet it is precisely the reason why the OOD robustness is most fragile. Crucially, robust finetuning depends on maintaining sufficient regularization strength throughout training; when that strength decays, OOD robustness erodes even as ID accuracy improves. This motivates trajectory regularization: keeping the teacher anchored to the optimization path so that it continues to exert a meaningful restoring force.

Moreover, despite the proliferation of these methods, a theoretical understanding of what changes during contrastive finetuning and where forgetting occurs remains elusive. We address this gap by developing a theoretical framework that reveals the geometric structure of how different finetuning strategies modify pretrained representations. We find that the linearized contrastive finetuning objective can be reformulated as a matrix least-squares problem through what we call the contrastive target matrix. This reformulation enables closed-form solutions for common finetuning strategies, exposing their fundamentally different geometric behaviors.

Our theoretical insights lead to the design of TRACER (Trajectory-Robust Anchoring for Contrastive Encoder Regularization), a practical finetuning method that implements our geometric principles. As shown in Figure 1, TRACER combines contrastive learning with dynamic selfdistillation, yielding strong results on ImageNet and its distribution shifts. Across multiple CLIP architectures, TRACER consistently improves the ID-OOD trade-off. We validate these findings through extensive ablation studies spanning distillation components, regularization strength, teacher update schedules, and kernel shape, demonstrating both the robustness of our method to hyperparameter choices and contributions of each design element.

In summary, our work makes the following main contributions: (i) We introduce the contrastive target matrix reformulation of linearized contrastive loss, turning the objective into a least-squares problem and enabling closed-form solutions for standard finetuning and regularization strategies; (ii) We derive a geometric decomposition that separates task– subspace mixing from orthogonal preservation, explaining when and where forgetting occurs and providing a principled basis for dynamic teachers; (iii) We identify a largely overlooked limitation of standard teachers in robust finetuning, the inherent collapse of the EMA teacher–student learning signal, and show how trajectory regularization with a weighted moving-average (WMA) teacher preserves a meaningful regularization signal over finite horizons, enabling bias-free task-subspace convergence; we instantiate these principles in TRACER with consistent OOD gains on CLIP finetuning, supported by comprehensive ablations across four axes (§B).

# 2. Related Work

Contrastive language–image pretraining (Radford et al., 2021; Jia et al., 2021; Ilharco et al., 2021; Zhai et al., 2023) enables strong zero-shot transfer but naive finetuning can harm OOD robustness (Taori et al., 2020; Wortsman et al., 2022b). Robust finetuning explores weight interpolation/averaging (Wortsman et al., 2022b;a; Jang et al., 2024), weight- or output-space regularization (Li et al., 2018; Li & Hoiem, 2018), and contrastive variants aligned to text prompts or energies (Goyal et al., 2023; Mao et al., 2024; Nam et al., 2024; Shu et al., 2023). CaRot (Oh et al., 2024) couples contrastive training with new regularizers to jointly improve OOD accuracy and calibration.

Self-distillation and dynamic teachers stabilize learning and preserve knowledge (Hinton et al., 2015; Zhang et al., 2019; Mobahi et al., 2020; Laine & Aila, 2017; Tarvainen & Valpola, 2017). Momentum/EMA teachers are effective yet can introduce persistent bias toward initialization, and their teacher–student gap collapses as training converges, reducing regularization exactly when OOD robustness is most vulnerable. This critical flaw is rarely made explicit in the robust finetuning literature. Our WMA teacher generalizes EMA by weighting the entire trajectory on normalized time, enabling endpoint-aware curricula (e.g., arcsine/Beta kernels) and trajectory regularization that preserves a meaningful teacher gap over finite horizons. As we prove, this yields bias-free task-subspace convergence. Our theory complements linearized analyses of supervised and contrastive learning (Ji et al., 2023; Tian, 2022; Nakada et al., 2023; Xue et al., 2024; Hao et al., 2025) and explains forgetting via an explicit geometric decomposition. An extended literature review appears in §A.

# 3. Theoretical Analysis

To address the ID-OOD trade-off, where finetuning improves in-distribution accuracy at the cost of out-ofdistribution robustness, we develop a theoretical framework that reveals the underlying dynamics of this phenomenon.

# 3.1. Problem Setting and Preliminaries

Finetuning task. We consider robust finetuning of a pretrained vision–language model on paired image–text data $\{ ( \mathbf { x } _ { I } ^ { i } , \mathbf { x } _ { T } ^ { i } ) \} _ { i = 1 } ^ { n }$ 1 drawn from a downstream task. The goal is to adapt the model so that in-distribution accuracy improves on this task while pretrained, broadly transferable representations are preserved for out-of-distribution generalization.

Linearized image/text encoders. Following linearized analyses widely used in the theory of contrastive learning (Ji et al., 2023; Tian, 2022; Nakada et al., 2023; Xue et al., 2024), we model the image and text encoders as linear projections, $g _ { I } ( \mathbf { x } ) = \mathbf { W } _ { I } \mathbf { x }$ and $g _ { T } ( \mathbf { x } ) = \mathbf { W } _ { T } \mathbf { x }$ . The image encoder $\mathbf { W } _ { I }$ is adapted from a pretrained state W0, while the text encoder $\mathbf { W } _ { T }$ is frozen at its pretrained state $\mathbf { W } _ { T } ^ { 0 }$ We collect the n image and text features of a batch into matrices $\mathbf { X } _ { I } \in \mathbb { R } ^ { d _ { I } \times n }$ and $\mathbf { X } _ { T } \in \mathbb { R } ^ { d _ { T } \times n }$ , where $d _ { I } , d _ { T }$ are the input feature dimensions and $p$ is the shared embedding dimension.

Original MMCL objective. The linearized multimodal contrastive learning (MMCL) loss is a standard analytic surrogate of the symmetric InfoNCE objective (the full deriva-

![](images/87b2d2501da86573dbd94e5f472b5adf4fda154b0ffa404fa6d19a32975fed1a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Student θ"] -->|ω₀ = α₀ = κ(τ₀)| B["Teacher ψ"]
    B -->|1 - ω₁| C["timestep t"]
    C -->|ωₜ = (αₜ=κ(τₜ)/∑_{j=0}ᵗ^t αⱼ)| D["hard labels"]
    D -->|L_MMCL| E["ε_Image"]
    E --> F["ε_Text"]
    F --> G["ε_Image"]
    G --> H["ε_Image"]
    H --> I["ε_Image"]
    I --> J["ε_Image"]
    J --> K["ε_Image"]
    K --> L["ε_Image"]
    L --> M["ε_Image"]
    M --> N["ε_Image"]
    N --> O["ε_Image"]
    O --> P["ε_Image"]
    P --> Q["ε_Image"]
    Q --> R["ε_Image"]
    R --> S["ε_Image"]
    S --> T["ε_Image"]
    T --> U["ε_Image"]
    U --> V["ε_Image"]
    V --> W["ε_Image"]
    W --> X["ε_Image"]
    X --> Y["ε_Image"]
    Y --> Z["ε_Image"]
    Z --> AA["ε_Image"]
    AA --> AB["ε_Image"]
    AB --> AC["ε_Image"]
    AC --> AD["ε_Image"]
    AD --> AE["ε_Image"]
    AE --> AF["ε_Image"]
    AF --> AG["ε_Image"]
    AG --> AH["ε_Image"]
    AH --> AI["ε_Image"]
    AI --> AJ["ε_Image"]
    AJ --> AK["ε_Image"]
    AK --> AL["ε_Image"]
    AL --> AM["ε_Image"]
    AM --> AN["ε_Image"]
    AN --> AO["ε_Image"]
    AO --> AP["ε_Image"]
    AP --> AQ["ε_Image"]
    AQ --> AR["ε_Image"]
    AR --> AS["ε_Image"]
    AS --> AT["ε_Image"]
    AT --> AU["ε_Image"]
    AU --> AV["ε_Image"]
    AV --> AW["ε Vaccine"]
    AW --> AX["θ^0_CLIP"]
    AX --> AY["ω₀ = α₀ = κ(τ₀)"]
    AY --> AZ["timestep t"]
```
</details>

Figure 1. Overview of TRACER. The base contrastive objective is combined with a dynamic self-distillation loss from a Weighted Moving Average (WMA) teacher to preserve orthogonal pretrained knowledge while adaptively mixing within the task subspace. $\mathsf { \Pi } \mathbf { \check { \theta } } _ { \mathrm { C L I P } } ^ { 0 }$ represents the initial pretrained CLIP model. $\theta ^ { t }$ denotes the student model at time $t ,$ with its image and text encoder $( \mathcal { E } _ { \mathrm { I m a g e } }$ and ${ \mathcal { E } } _ { \mathrm { T e x t } } )$ being trained. The student receives gradient updates from ${ \mathcal { L } } _ { \mathrm { M M C L } }$ . The WMA teacher model $\psi ^ { t }$ is updated from the student’s parameters. The teacher then provides a teaching signal LSD-WMA to regularize the student. This interplay allows TRACER to adapt to new tasks while preserving pretrained knowledge. The complete training procedure is detailed in Algorithm 1.

tion appears in §C.1):

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{MMCL}} \left(\mathbf {W} _ {I}, \mathbf {W} _ {T}\right) = \frac {1}{n (n - 1)} \left[ \sum_ {i \neq j} s _ {i j} - (n - 1) \sum_ {i} s _ {i i} \right] \\ + R (\mathbf {W} _ {I}, \mathbf {W} _ {T}), \tag {1} \\ \end{array}
$$

where $s _ { i j } = ( \mathbf { W } _ { I } \mathbf { x } _ { I } ^ { i } ) ^ { \top } ( \mathbf { W } _ { T } \mathbf { x } _ { T } ^ { j } )$ is the image–text similarity, and $R ( \cdot )$ is a cross-Frobenius term. This loss balances pulling matched pairs together against pushing unmatched pairs apart. As shown in §C.2, when ${ \bf W } _ { T } = { \bf W } _ { T } ^ { 0 }$ is frozen, optimizing equation 1 over $\mathbf { W } _ { I }$ is equivalent (up to constants and a data-dependent quadratic term that arises naturally in the optimization) to a matrix least-squares problem driven by the contrastive target matrix introduced below.

Notation. We use $\mathbf { I } _ { n } \in \mathbb { R } ^ { n \times n }$ for the identity matrix and $\mathbf { J } _ { n } = \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } \in \mathbb { R } ^ { n \times n }$ for the all-ones matrix, where ${ \bf 1 } _ { n }$ denotes the n-dimensional all-ones vector. The matrix $n \mathbf { I } _ { n } - \mathbf { J } _ { n }$ acts as a centered contrastive operator on text features: it preserves the matched (diagonal) directions while subtracting the batch-mean direction, yielding a per-column “attract-paired/repel-others” signal. We let $\mathcal { P } _ { I } : = \mathbf { X } _ { I } ( \mathbf { X } _ { I } ^ { \top } \mathbf { X } _ { I } ) ^ { + } \mathbf { X } _ { I } ^ { \top }$ denote the orthogonal projector onto range $\mathbf { X } _ { I } )$ , the task subspace spanned by the finetuning image features, and ${ \bf I } - { \mathcal P } _ { I }$ the projector onto its orthogonal complement. Throughout, $( \cdot ) ^ { + }$ is the Moore–Penrose pseudoinverse and $\lVert \cdot \rVert _ { \mathrm { F } }$ denotes the Frobenius norm.

# 3.2. Loss Reformulation via the Contrastive Target Matrix

We now rewrite the MMCL objective equation 1 in a form that exposes closed-form solutions via a single algebraic construct.

Definition 3.1 (Contrastive Target Matrix). Given the frozen text encoder $\mathbf { W } _ { T } ^ { 0 }$ and finetuning texts $\mathbf { X } _ { T }$ , we define the contrastive target matrix as

$$
\mathbf {Y} _ {\mathrm{FT}} := \mathbf {W} _ {T} ^ {0} \mathbf {X} _ {T} (n \mathbf {I} _ {n} - \mathbf {J} _ {n}) \in \mathbb {R} ^ {p \times n}.
$$

Each column $\mathbf { y } _ { i }$ is constructed to attract the image embedding $\mathbf { x } _ { I } ^ { i }$ towards its paired text $\mathbf { x } _ { T } ^ { i }$ and repel it from the remaining texts in the batch (detailed in §C.1).

What is $\mathbf { Y } _ { \mathbf { F T } } .$ , and why is it “fixed”? The matrix $\mathbf { Y } _ { \mathrm { F T } }$ depends only on the frozen text encoder $\mathbf { W } _ { T } ^ { 0 }$ and the finetuning text features $\mathbf { X } _ { T } ;$ it does not depend on the trainable image weights $\mathbf { W } _ { I } .$ . Consequently, $\mathbf { Y } _ { \mathrm { F T } }$ stays fixed throughout finetuning and plays exactly the role of a target in a supervised regression problem: it is the centered contrastive signal that $\mathbf { W } _ { I } \mathbf { X } _ { I }$ should match. In analogy with linear regression, $\displaystyle ( \mathbf { X } _ { I } , \mathbf { Y } _ { \mathrm { F T } } )$ form (inputs, targets) and the linearized MMCL loss reduces to a matrix least-squares problem with $\mathbf { Y } _ { \mathrm { F T } }$ as the regression target. The recurring factor $\left( n \mathbf { I } _ { n } - \mathbf { J } _ { n } \right)$ in $\mathbf { Y } _ { \mathrm { F T } }$ implements the contrastive centering with ${ \mathbf I } _ { n }$ and ${ \bf J } _ { n }$ as defined above.

Using $\mathbf { Y } _ { \mathrm { F T } }$ , the linearized MMCL objective equation 1 reduces (up to constants and a data-dependent quadratic term) to:

$$
\min _ {\mathbf {W} _ {I}} \frac {1}{2} \left\| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}} \right\| _ {\mathrm{F}} ^ {2}. \tag {2}
$$

This formulation is crucial because it enables closed-form solutions for various finetuning strategies under gradient descent, offering insights into their behavior (see §C.2 for the formal trace-to-least-squares equivalence and §C.3 for the full derivations and proofs). Using this reformulation, we analyze how different finetuning strategies mitigate forgetting by preserving or adapting pretrained knowledge, and reveal the geometric structure of updates.

# 3.3. Closed-Form Solutions

This subsection follows the setting of Yang et al. (2024b) and provides closed-form solutions for various finetuning strategies following our reformulation (Equation 2), revealing a geometric decomposition: finetuning involves (i) preserving pretrained knowledge in directions orthogonal to the finetuning data, and (ii) adapting or mixing knowledge within the task-relevant subspace. We first present the closed-form solutions below.

Theorem 3.2 (Unified Framework for Contrastive Finetuning Solutions). Let $\mathcal { P } _ { I } : = \mathbf { X } _ { I } ( \mathbf { X } _ { I } ^ { \top } \mathbf { X } _ { I } ) ^ { + } \mathbf { X } _ { I } ^ { \top }$ be the orthogonal projector onto range $\left( \mathbf { X } _ { I } \right)$ . Gradient descent initialized at $\mathbf { W } _ { I } ^ { 0 }$ on the objective $\begin{array} { r } { \mathcal { L } ( \mathbf { W } _ { I } ) = \frac { 1 } { 2 } \left\| \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { Y } _ { F T } \right\| _ { F } ^ { 2 } + } \end{array}$ $\mathcal { R } ( \mathbf { W } _ { I } )$ converges to the following solutions:

1. Direct Finetuning $( \mathcal { R } ( \mathbf { W } _ { I } ) = 0 )$ :

$$
\mathbf {W} _ {F T} = \mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I}) + \mathbf {Y} _ {F T} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+}
$$

2. $L _ { 2 }$ Regularization (L2-SP (Li et al., 2018)), with $\begin{array} { r } { { \mathcal { R } } ( \mathbf { W } _ { I } ) = \frac { \lambda } { 2 } \left\| \mathbf { W } _ { I } - \mathbf { W } _ { I } ^ { 0 } \right\| _ { F } ^ { 2 } . } \end{array}$

$$
\mathbf {W} _ {L _ {2}} = (\mathbf {Y} _ {F T} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0}) (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {I}) ^ {- 1}
$$

3. Static Self-Distillation (SD (Furlanello et al., 2018)), with $\begin{array} { r } { { \mathcal { R } } ( \mathbf { W } _ { I } ) = \frac { \lambda } { 2 } \left\| \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { W } _ { I } ^ { 0 } \mathbf { X } _ { I } \right\| _ { F } ^ { 2 } . } \end{array}$

$$
\mathbf {W} _ {S D} = \mathbf {W} _ {I} ^ {0} \left(\mathbf {I} - \frac {1}{1 + \lambda} \mathcal {P} _ {I}\right) + \frac {1}{1 + \lambda} \mathbf {Y} _ {F T} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+}
$$

Here, + denotes the Moore-Penrose pseudoinverse and $\lambda >$ 0 is the regularization parameter.

Geometric Interpretation: As visualized in Figure 2, Direct Finetuning discards pretrained knowledge within the finetuning data subspace, replacing it with the new task solution, while preserving orthogonal components. $L _ { 2 }$ regularization shrinks the entire solution towards the pretrained weights, leading to a complex, non-surgical blend.

Self-Distillation achieves a nuanced trade-off: it preserves pretrained knowledge in the subspace orthogonal to the finetuning data $( \mathbf { W } _ { I } ^ { 0 } ( \mathbf { I } - \mathcal { P } _ { I } ) )$ , and within the task-relevant subspace, it computes a convex combination of the projected pretrained weights and the optimal solution for the new task. This enables control over knowledge retention and adaptation (further details in §C.4).

# 3.4. Dynamic Self-Distillation with a WMA Teacher

Why static SD is biased. Restricted to the task subspace, the static-SD solution in Theorem 3.2 is the convex combination $\begin{array} { r } { \mathbf { W } _ { S D } \mathbf { \mathcal { P } } _ { I } = \frac { \lambda } { 1 + \lambda } \mathbf { W } _ { I } ^ { 0 } \mathbf { \mathcal { P } } _ { I } + \frac { 1 } { 1 + \lambda } \mathbf { W } _ { \mathrm { F T } } ^ { \star } , } \end{array}$ 1+λ where $\mathbf { W } _ { \mathrm { F T } } ^ { \star } = \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top } ( \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top } ) ^ { + }$ is the minimum-norm task solution. Because the anchor is fixed at $\mathbf { W } _ { I } ^ { 0 }$ , for any finite $\lambda > 0$ the solution stays offset from $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ in the task subspace by exactly $\begin{array} { r } { \frac { \lambda } { 1 + \lambda } ( \mathbf { \bar { W } } _ { I } ^ { 0 } \mathbf { \bar { \mathcal { P } } } _ { I } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } ) } \end{array}$ , an anchor bias that cannot be removed by tuning λ alone (smaller λ reduces the bias but weakens orthogonal preservation; larger λ does the reverse). Geometrically (Figure 2), static SD lies on the segment between $\mathbf { W } _ { I } ^ { 0 } \mathcal { P } _ { I }$ and $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ rather than reaching the task optimum.

Intuition: static SD vs. EMA vs. WMA. The teacher choice controls where the regularizer is anchored along the trajectory: Static SD fixes the anchor at the start $( \mathbf { W } _ { I } ^ { 0 } )$ and is biased toward initialization; an EMA teacher stays near the current student state, so its regularizing signal vanishes as training converges (precisely when OOD robustness is most fragile); a WMA teacher stays near a trajectory-weighted consensus of the optimization path: it remembers the start but adapts over time, and with a suitable kernel retains meaningful mass at both ends of training. WMA thus addresses two limitations at once: (i) the static-SD anchor bias and (ii) the EMA collapse of the teacher–student gap. We instantiate it as a dynamic teacher that evolves as a WMA of the student’s trajectory, illustrated at the system level in Figure 1, with the formal definition below.

Definition 3.3 (WMA Teacher). The WMA teacher $\mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t }$ averages student states $\mathbf { W } _ { I } ^ { k }$ over time $k = 0 , \ldots , t ,$ weighted by a kernel $\kappa ( \tau _ { k } )$ on normalized time $\tau _ { k } =$ k+c1T +c2 ∈ (0, 1). The offsets c1, c2 > 0 keep τk strictly $\frac { k + c _ { 1 } } { T + c _ { 2 } } ~ \in ~ ( 0 , 1 )$ T+ $c _ { 1 } , c _ { 2 } \ > \ 0$ $\tau _ { k }$ inside (0, 1) (avoiding $\tau _ { 0 } = 0$ and $\tau _ { T } = 1 )$ , which is required for kernels such as $\mathrm { B e t a } ( 0 . 5 , 0 . 5 )$ that diverge at the endpoints; we use $c _ { 1 } = 0 . 5 , ~ c _ { 2 } = 1$ in all experiments. The online recursion is:

$$
\omega_ {t} = \frac {\kappa (\tau_ {t})}{\sum_ {j = 0} ^ {t} \kappa (\tau_ {j})},
$$

$$
\mathbf {W} _ {\text {Teacher}} ^ {t} = (1 - \omega_ {t}) \mathbf {W} _ {\text {Teacher}} ^ {t - 1} + \omega_ {t} \mathbf {W} _ {I} ^ {t},
$$

$$
\mathbf {W} _ {\text { Teacher }} ^ {0} = \mathbf {W} _ {I} ^ {0}.
$$

Persistent Regularization and Bias-Free Convergence: Unlike an Exponential Moving Average (EMA) teacher, whose regularizing influence vanishes as it converges to the student, the WMA teacher (especially with a U-shaped kernel like Beta(0.5,0.5)) maintains a persistent regularizing force over finite training horizons (see §C.5.2). This force continuously pulls the student towards its robust pretrained initialization. We prove that this adaptive anchoring achieves bias-free convergence to the task-optimal solution within the finetuning subspace:

![](images/7e872d03de16cd424e5745194a298867c39e175cecb72ff77ce768cde6fed749.jpg)

<details>
<summary>text_image</summary>

w2
W0I
span(XI)
W0IPI
W0I(I - PI)
WFT
0
w1
DFT
WFT*
Dpretrain
WFT = W0I(I - PI) + WFT*
preserve
WFT* replace
</details>

(a) Direct FT: Preserves orthogonal, replaces parallel component.

![](images/a81bff2932ac30e138ea7b57d3d85f0427d06bbc88a3ce1da353cdb149db41e6.jpg)

<details>
<summary>text_image</summary>

w2
W0I
span(XI)
W0IPI
W1(I - PI)
WL2
WFI
0
w1
Dft
W*FT
Dpretrain
Interpolates between W0 and WFT ridge shrinkage affects all directions
</details>

(b) L2-SP: Blends all directions, no structure preservation.

![](images/daffbb6b751447ea0650d1d3ea298a2a41fece3c21171a6160d74288a9e40c80.jpg)

<details>
<summary>text_image</summary>

w2
W0I
span(XI)
WDI(I - PI)
WSD
W0IPI
0 1/2 -mix
w1
DFT
W*FT
DPretrain
Preserve ⊥ + convex mix in ||
λ = 1 ⇒ λ/(1+λ) = 1/(1+λ) = 1/2 -mix
</details>

(c) SD: Preserves orthogonal, mixes parallel components.   
Figure 2. Geometric interpretation of finetuning strategies in 2D weight space. The green line represents $\operatorname { s p a n } ( \mathbf { X } _ { I } ^ { \top } )$ , the subspace where finetuning data concentrates. Starting from pretrained weights $\breve { \mathbf { W } } _ { I } ^ { 0 }$ (blue), each method combines the orthogonal component $\mathbf { W } _ { I } ^ { 0 } ( \mathbf { I } - \mathcal { P } _ { I } )$ and the new task solution $\mathbf { W } _ { \mathrm { F T } } ^ { \star } = \mathbf { Y } _ { \mathrm { F T } } \dot { \mathbf { X } } _ { I } ^ { \top } ( \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top } ) ^ { + }$ (green) differently: (a) Direct FT preserves the orthogonal component and replaces the parallel component entirely; (b) L2-SP creates a global blend without clean structural decomposition; (c) Static Self-Distillation preserves the orthogonal component and forms a convex combination of the parallel components (shown with $\lambda = 1$ giving equal weighting).

Theorem 3.4 (Bias-Free Convergence in the Task Subspace). Let $\mathbf { W } _ { F T } ^ { \star } = \mathbf { Y } _ { F T } \mathbf { X } _ { I } ^ { \top } ( \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top } ) ^ { + }$ denote the minimum-norm task solution $( i . e . ,$ , the minimum-Frobenius-norm minimizer of equation 2). The WMA teacher’s projection onto the task subspace converges to $\mathbf { W } _ { F T } ^ { \star } \mathcal { P } _ { I } ,$ , and consequently, the student’s projection also converges to $\mathbf { W } _ { F T } ^ { \star } \mathcal { P } _ { I }$ .

We use the name minimum-norm task solution for $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ to distinguish it from the Direct FT solution in Theorem 3.2, which additionally retains the orthogonal component $\mathbf { W } _ { I } ^ { 0 } ( \mathbf { I } - \mathcal { P } _ { I } )$ : $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ is the task-subspace target a preservation-aware method should reach within range(XI). The theorem shows that dynamic teachers eliminate the static anchor bias while preserving orthogonal knowledge (formal proof in §C.5).

# 4. Methodology: TRACER

Guided by these geometric insights and the proven benefits of a WMA teacher in the above section, we propose TRACER, a novel finetuning method for multimodal models. TRACER combines the standard symmetric InfoNCE loss with dynamic self-distillation guided by a WMA teacher, as illustrated in Figure 1.

The total training objective for TRACER is:

$$
\mathcal {L} _ {\text { TRACER }} = \mathcal {L} _ {\text { MMCL }} + \lambda_ {\text { SD }} \mathcal {L} _ {\text { SD - WMA }}. \tag {3}
$$

Multi-Modal Contrastive Loss $( { \mathcal { L } } _ { \mathrm { M M C L } } ) { \mathrm { : } }$ This is the primary finetuning loss, typically a symmetric InfoNCE objective. In our implementation, we also include a cross-Frobenius regularizer to prevent embedding collapse (standard CLIP finetuning recipe). This component drives the student model to learn new task-specific alignments.

Dynamic Self-Distillation Loss (LSD-WMA): This is the core mechanism for robust knowledge preservation and adaptive mixing. It ensures the student retains generalizable features by learning from an evolving teacher model. As detailed in §C.6, LSD-WMA is a composite distillation loss that includes several perspectives: (i) Feature Distillation (FD): Directly aligns student and teacher embeddings. (ii) Contrastive Relational Distillation (CRD): Matches batch-wise similarity distributions between student and teacher. (iii) Interactive Contrastive Learning (ICL): Encourages student-teacher cross-modal alignment. (iv) Cross Knowledge Distillation (Cross-KD): Aligns cross-modal logits to transfer relational structure. This multi-perspective approach operationalizes the theoretical insight of preserving distinct aspects of pretrained knowledge.

Weighted Moving Average (WMA) Teacher: The teacher model is a central component of TRACER. Unlike an EMA teacher, which gradually collapses onto the student, our WMA teacher is a weighted average of the entire student trajectory up to time t, using a carefully chosen weighting kernel (e.g., a Beta kernel with $\beta _ { 1 } ~ = ~ \beta _ { 2 } ~ = ~ 0 . 5$ as shown in Figure 1 and detailed in §C.5.1). This ensures that early pretrained states retain a non-trivial contribution to the teacher over finite horizons, aligning with our trajectoryregularization motivation. This persistent regularization provides a continuous restoring force, preventing the student from over-specializing on spurious correlations in the finetuning data.

![](images/d7604d12477ee7fbfa16eec104fbb3bb41014857d19b563d4f2c8d018f155337.jpg)

<details>
<summary>bar</summary>

| Model | Pre-training (MNIST) (%) | Fine-tuning (C-MNIST) (%) |
|---|---|---|
| Pre-trained | 96.8 | 7.2 |
| Direct FT | 59.0 | 98.6 |
| L2 Reg | 83.2 | 98.5 |
| Static Distill | 95.9 | 98.0 |
| Dynamic Distill | 96.7 | 98.6 |
</details>

![](images/8979baf9bb27888899a26cdee139021779947a719d133dadbf6bfb0ecdf68b83.jpg)

<details>
<summary>bar</summary>

| Method | Forgetting Rate (%) |
| :--- | :--- |
| Direct FT | 37.9 |
| L2 Reg | 13.6 |
| Static Distill | 1.8 |
| Dynamic Distill | -0.1 |
(b) - (-37.72%)
</details>

![](images/c10c0cb40daacd817742f5ec88a84ae878ad8bd922c0e4a5533a9874f25782f2.jpg)

<details>
<summary>scatter</summary>

| Method           | Original Task Accuracy (%) | Spurious Task Accuracy (%) |
| ---------------- | -------------------------- | -------------------------- |
| Direct FT        | 60                         | 98.6                       |
| L2 Reg           | 83                         | 98.5                       |
| Dynamic Distill  | 94                         | 98.5                       |
| Static Distill   | 93                         | 98.1                       |
</details>

Figure 3. Toy Experiment. We compare a pretrained model against four finetuning methods on a finetuning task. (a) Performance on the original MNIST and new Colored MNIST task. All finetuning methods successfully learn the new task. Direct FT and L2 Reg suffer severe performance degradation (catastrophic forgetting). (b) Catastrophic forgetting rate, quantified as the percentage drop in accuracy on the original task. Self-distillation methods are more effective at preserving knowledge. (c) The performance trade-off between the original task (x-axis) and the spurious task (y-axis). Distillation methods achieve a much better trade-off, retaining high original task accuracy while mastering the new task.

# 5. Experiments

This section evaluates TRACER on ImageNet and natural distribution shifts, including a controlled toy study to validate theoretical predictions. We conduct comprehensive ablations across four axes (distillation components, strength, teacher update frequency, and Beta-kernel shape); extended protocols are in §B, loss definitions in §C.6, and teacher details in §C.5.1.

# 5.1. Synthetic Experiment

We design a controlled toy experiment with spurious correlations (Arjovsky et al., 2019) to validate our theory. The behaviors of Direct Finetuning, L2 Regularization, and Self-Distillation in a non-linear architecture align with our closedform predictions.

# 5.1.1. EXPERIMENTAL SETUP

Datasets. We use two variants of the MNIST dataset (Le-Cun et al., 1998; Deng, 2012). (i) Original Pretraining Task: We create a multimodal version of MNIST, where each grayscale digit image is paired with a simple text description (e.g., an image of a ‘7’ is paired with the text “the digit 7”). The model is pretrained on this dataset to learn robust, general-purpose representations for digit recognition. (ii) Finetuning Task: We create a dataset to introduce a spurious correlation. Images of digits 0-4 are colored red with 95% probability, while digits 5-9 are colored blue with 95% probability. This setup forces the model during finetuning to learn an easy-to-exploit but non-causal feature (color) to solve the new task, creating a direct conflict with the original digit recognition knowledge.

Model Architecture. We employ lightweight, non-linear models to show that our theory extends beyond the linear case. The architecture consists of a ‘LightViT’ (Dosovitskiy et al., 2021) image encoder and a ‘LightTextTransformer’ (Vaswani et al., 2017) text encoder. Both models project their inputs into a shared 128-dimensional embedding space, where a standard InfoNCE contrastive loss is applied during pretraining.

Pretraining. The model is pretrained on MNIST using a contrastive objective for 10 epochs, achieving high accuracy on digit recognition but poor performance on the color-based task.

Finetuning Strategies. We finetune the pretrained image encoder on the ColoredMNIST (Arjovsky et al., 2019; Zhang et al., 2022a) task for 10 epochs while keeping the text encoder frozen, mirroring our theoretical setup. We compare the following methods: (i) Pretrained: The baseline model without any finetuning. (ii) Direct Finetuning: The image encoder is finetuned on the new task, as analyzed in §C.3. (iii) $L _ { 2 }$ Regularization: We add a penalty term $\begin{array} { r } { { \frac { \lambda } { 2 } } \left\| \mathbf { W } _ { I } - \mathbf { W } _ { I } ^ { 0 } \right\| _ { \mathrm { F } } ^ { 2 } } \end{array}$ to the finetuning loss, corresponding to our analysis of $L _ { 2 }$ regularization. (iv) Static Distillation: We use the initial pretrained model W0 as a fixed teacher and add a distillation loss term to the finetuning objective, as analyzed for $\mathbf { W } _ { S D }$ . (v) Dynamic Distillation: We use a teacher model whose weights are a moving average of the student’s weights, corresponding to our analysis of the WMA teacher.

# 5.1.2. RESULTS AND DISCUSSION

Analysis of Forgetting. As predicted by our theory, Direct Finetuning exhibits severe catastrophic forgetting. It achieves near-perfect accuracy (98.5%) on the new colorbased task by overwriting its original knowledge, causing its performance on the original MNIST test set to degrade from 96.8% to 59.0%, a forgetting rate of 37.9%. L2 Regularization offers an improvement, but still forgets 13.6% of the original task’s performance. In contrast, both Static and Dynamic Distillation demonstrate resilience to forgetting. They also master the new task but retain a larger portion of the original knowledge, with forgetting rates of only 1.8% and 0.1%, respectively. This result empirically supports our geometric interpretation: by interpolating between old and new knowledge within the task-relevant subspace while preserving knowledge in the orthogonal subspace, self-distillation methods achieve a better balance.

The Performance Trade-off. The scatter plot in Figure 3(c) visualizes this trade-off: distillation methods achieve a better Pareto frontier, with Dynamic Distillation finding a slightly better solution than its static counterpart, aligning with our theoretical analysis.

# 5.2. Main ImageNet Results and Ablations

We report the main ImageNet results and ablations below; detailed per-backbone tables follow.

Table 1. ImageNet accuracy. We report accuracy (↑) on ImageNet and its distribution shift variants by finetuning CLIP ViT-B/16 with six methods. All values are averaged over three seeds with standard deviations shown as subscripts. In each column, the best value is bold and the second-best is underlined. 

<table><tr><td>Method</td><td>IN</td><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjNet</td><td>Avg.</td></tr><tr><td>ZS</td><td>68.33</td><td>61.93</td><td>77.71</td><td>49.95</td><td>48.26</td><td>54.17</td><td>58.39</td></tr><tr><td>LP-FT</td><td> $82.44_{\pm 0.08}$ </td><td> $72.74_{\pm 0.18}$ </td><td> $72.81_{\pm 0.22}$ </td><td> $49.28_{\pm 0.31}$ </td><td> $50.31_{\pm 0.15}$ </td><td> $54.42_{\pm 0.14}$ </td><td> $59.91_{\pm 0.18}$ </td></tr><tr><td>FLYP</td><td> $82.72_{\pm 0.09}$ </td><td> $72.76_{\pm 0.21}$ </td><td> $71.32_{\pm 0.25}$ </td><td> $48.49_{\pm 0.35}$ </td><td> $49.87_{\pm 0.18}$ </td><td> $54.83_{\pm 0.16}$ </td><td> $59.45_{\pm 0.20}$ </td></tr><tr><td>Lipsum-FT</td><td> $83.32_{\pm 0.05}$ </td><td> $73.57_{\pm 0.12}$ </td><td> $75.93_{\pm 0.14}$ </td><td> $49.87_{\pm 0.28}$ </td><td> $51.43_{\pm 0.12}$ </td><td> $54.35_{\pm 0.11}$ </td><td> $61.03_{\pm 0.14}$ </td></tr><tr><td>CaRot</td><td> $83.15_{\pm 0.06}$ </td><td> $74.08_{\pm 0.14}$ </td><td> $77.74_{\pm 0.16}$ </td><td> $51.57_{\pm 0.24}$ </td><td> $52.68_{\pm 0.13}$ </td><td> $56.63_{\pm 0.12}$ </td><td> $62.54_{\pm 0.14}$ </td></tr><tr><td>TRACER</td><td> $82.76_{\pm 0.07}$ </td><td> $74.14_{\pm 0.15}$ </td><td> $79.33_{\pm 0.18}$ </td><td> $54.92_{\pm 0.26}$ </td><td> $53.69_{\pm 0.14}$ </td><td> $58.26_{\pm 0.13}$ </td><td> $64.07_{\pm 0.15}$ </td></tr></table>

# 5.2.1. EXPERIMENTAL SETUP

Objective. Our experiments are designed to validate our theoretical claims and assess TRACER, as a practical implementation of our framework, on robust finetuning. We focus on evaluating both accuracy and calibration under distribution shifts.

Datasets and Evaluation. We use ImageNet-1K (IN) (Deng et al., 2009; Russakovsky et al., 2015) as our in-distribution (ID) downstream task. To measure OOD robustness, we evaluate all finetuned models on a standard suite of five distribution shift datasets: ImageNet-V2 (IN-V2) (Recht et al., 2019), ImageNet-Rendition (IN-R) (Hendrycks et al., 2021a), ImageNet-Adversarial (IN-A) (Hendrycks et al., 2021b), ImageNet-Sketch (IN-S) (Wang et al., 2019), and ObjectNet (Barbu et al., 2019). We report the average performance across these five datasets as “Avg. shifts” or “OOD”.

Baselines. We compare TRACER against a comprehensive set of baselines, including zero-shot (ZS (Radford et al., 2021)), linear probing then finetuning (LP-FT (Kumar et al., 2022)), finetune-like-you-pretrain (FLYP (Goyal et al., 2023)), Lipsum-FT (Nam et al., 2024), and the recent robust finetuning method CaRot (Oh et al., 2024).

Metrics. We report top-1 accuracy and Expected Calibration Error (ECE), which measures the gap between predicted confidence and empirical accuracy (lower is better). We average across the five OOD datasets to summarize robustness.

Implementation Details and Experimental Setup. We finetune CLIP variants on ImageNet-1K (IN) and evaluate on five OOD datasets: IN-V2, IN-R, IN-A, IN-S, and ObjectNet, following Taori et al. (2020). For all methods, we finetune for 10 epochs using the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a weight decay of 0.01. The batch size is set to 224 for ViT-L/14 and 512 for ViT-B/16 and ResNet50. All experiments are run over three random seeds and we report mean and standard deviation. For TRACER, the WMA teacher uses a Beta(0.5, 0.5) weighting kernel and combines symmetric InfoNCE with the composite SD loss (§C.6).

Table 2. ImageNet Accuracy. (except ObjectNet) with additional baselines. All values are averaged over three seeds. 

<table><tr><td>Method</td><td>IN↑</td><td>IN-V2↑</td><td>IN-R↑</td><td>IN-A↑</td><td>IN-S↑</td><td>Avg. shifts↑</td></tr><tr><td>ZS</td><td>68.33</td><td>61.93</td><td>77.71</td><td>49.95</td><td>48.26</td><td>59.46</td></tr><tr><td>Direct FT</td><td> $82.83 \pm 0.10$ </td><td> $72.57 \pm 0.28$ </td><td> $68.53 \pm 0.32$ </td><td> $39.23 \pm 0.35$ </td><td> $47.97 \pm 0.22$ </td><td> $57.08 \pm 0.24$ </td></tr><tr><td>L2-SP (Li et al., 2018)</td><td> $82.87 \pm 0.09$ </td><td> $72.63 \pm 0.22$ </td><td> $68.77 \pm 0.24$ </td><td> $39.73 \pm 0.28$ </td><td> $48.23 \pm 0.15$ </td><td> $57.34 \pm 0.18$ </td></tr><tr><td>Static SD (Hinton et al., 2015)</td><td> $82.07 \pm 0.08$ </td><td> $73.13 \pm 0.26$ </td><td> $72.87 \pm 0.18$ </td><td> $42.33 \pm 0.38$ </td><td> $49.87 \pm 0.21$ </td><td> $59.55 \pm 0.22$ </td></tr><tr><td>LP-FT (Kamar et al., 2022)</td><td> $82.14 \pm 0.08$ </td><td> $72.09 \pm 0.20$ </td><td> $70.44 \pm 0.22$ </td><td> $46.32 \pm 0.30$ </td><td> $48.65 \pm 0.16$ </td><td> $59.38 \pm 0.18$ </td></tr><tr><td>FLYP (Goyal et al., 2023)</td><td> $82.72 \pm 0.09$ </td><td> $72.76 \pm 0.24$ </td><td> $71.32 \pm 0.26$ </td><td> $48.49 \pm 0.34$ </td><td> $49.87 \pm 0.19$ </td><td> $60.61 \pm 0.21$ </td></tr><tr><td>CAR-FT (Mao et al., 2024)</td><td> $83.27 \pm 0.06$ </td><td> $74.03 \pm 0.18$ </td><td> $75.37 \pm 0.28$ </td><td> $49.53 \pm 0.24$ </td><td> $52.97 \pm 0.20$ </td><td> $62.98 \pm 0.18$ </td></tr><tr><td>Lipsum-FT (Nun et al., 2024)</td><td> $83.33 \pm 0.05$ </td><td> $73.57 \pm 0.12$ </td><td> $75.93 \pm 0.14$ </td><td> $49.87 \pm 0.28$ </td><td> $51.43 \pm 0.12$ </td><td> $62.70 \pm 0.14$ </td></tr><tr><td>Model Stock (Jang et al., 2024)</td><td> $\mathbf{84.07} \pm 0.07$ </td><td> $\mathbf{74.83} \pm 0.16$ </td><td> $71.77 \pm 0.20$ </td><td> $51.23 \pm 0.30$ </td><td> $51.77 \pm 0.17$ </td><td> $62.40 \pm 0.18$ </td></tr><tr><td>ARF (Han et al., 2024)</td><td> $82.73 \pm 0.08$ </td><td> $72.77 \pm 0.19$ </td><td> $75.63 \pm 0.22$ </td><td> $50.27 \pm 0.28$ </td><td> $51.83 \pm 0.16$ </td><td> $62.63 \pm 0.17$ </td></tr><tr><td>CaRot (Oh et al., 2024)</td><td> $83.16 \pm 0.06$ </td><td> $74.08 \pm 0.14$ </td><td> $77.74 \pm 0.16$ </td><td> $51.57 \pm 0.22$ </td><td> $52.74 \pm 0.13$ </td><td> $64.03 \pm 0.14$ </td></tr><tr><td>TRACER</td><td> $82.76 \pm 0.07$ </td><td> $74.12 \pm 0.15$ </td><td> $\mathbf{79.30} \pm 0.18$ </td><td> $\mathbf{54.72} \pm 0.24$ </td><td> $\mathbf{53.69} \pm 0.14$ </td><td> $\mathbf{65.46} \pm 0.15$ </td></tr></table>

# 5.2.2. RESULTS AND ANALYSIS

OOD accuracy and calibration on ViT-B/16. As shown in Table 1, TRACER achieves strong OOD accuracy on ViT-B/16, particularly on the most challenging shifts (ObjectNet and IN-A). In Appendix Table 5, TRACER achieves the lowest average OOD ECE, indicating probabilistic reliability under shift. With additional baselines (Table 2), TRACER remains among the top OOD performers while staying competitive on IN. Additionally, the cross-backbone experiments (Table 3) show similar trends for RN50 and ViT-L/14.

OOD degradation as the empirical signature of catastrophic forgetting. We treat OOD accuracy degradation as the primary measurable symptom of catastrophic forgetting under finetuning: when a model retains less pretrained, broadly transferable structure, the loss shows up most clearly on inputs that differ from the finetuning distribution. Three results in our experiments support this view: (i) Direct FT drops below the zero-shot baseline on average OOD accuracy (Table 2: 57.08% vs. 59.46% ZS); (ii) the toy experiment (§5.1) measures explicit forgetting rates of 37.9% for Direct FT vs. 0.1% for dynamic SD; and (iii) the CKA/SVCCA analysis (Figure 4) shows that the deeper layers of Direct FT drift away from the pretrained representation, while TRACER keeps similarity > 0.97 across all layers. Together, these justify reading the OOD column of Tables 1–3 as a quantitative measure of how much pretrained knowledge each method preserves.

Table 3. ImageNet accuracy and ECE across backbones. We provide summarized results on CLIP RN50, ViT-B/16, and ViT-L/14, averaged over three seeds. The best and the second-best in each column are bold and underlined, respectively. OOD columns are highlighted to emphasize robustness. (See Table 1 and Appendix Table 5 for ViT-B/16, and Table 6 and 7 for details.) 

<table><tr><td rowspan="2">Method</td><td colspan="4">RN50</td><td colspan="4">ViT-B/16</td><td colspan="4">ViT-L/14</td></tr><tr><td>ID Acc.↑</td><td>ID ECE↓</td><td>OOD Acc.↑</td><td>OOD ECE↓</td><td>ID Acc.↑</td><td>ID ECE↓</td><td>OOD Acc.↑</td><td>OOD ECE↓</td><td>ID Acc.↑</td><td>ID ECE↓</td><td>OOD Acc.↑</td><td>OOD ECE↓</td></tr><tr><td>LP-FT</td><td>76.25</td><td>0.1042</td><td>41.62</td><td>0.3274</td><td>82.44</td><td>0.051</td><td>59.91</td><td>0.147</td><td>84.74</td><td>0.1056</td><td>64.11</td><td>0.2521</td></tr><tr><td>FLYP</td><td>76.16</td><td>0.0516</td><td>42.70</td><td>0.2127</td><td>82.72</td><td>0.064</td><td>59.45</td><td>0.184</td><td>86.19</td><td>0.0729</td><td>71.44</td><td>0.1470</td></tr><tr><td>CaRot</td><td>76.12</td><td>0.0471</td><td>42.71</td><td>0.2109</td><td>83.15</td><td>0.047</td><td>62.54</td><td>0.079</td><td>86.95</td><td>0.0349</td><td>74.13</td><td>0.0737</td></tr><tr><td>TRACER</td><td>76.48</td><td>0.0470</td><td>42.73</td><td>0.1807</td><td>82.76</td><td>0.045</td><td>64.07</td><td>0.073</td><td>86.27</td><td>0.0507</td><td>75.32</td><td>0.0732</td></tr></table>

Static SD vs. TRACER. Table 2 isolates the contribution of the trajectory-regularized teacher on ViT-B/16. A static self-distillation anchor at $\mathbf { W } _ { I } ^ { 0 }$ already recovers a large portion of OOD accuracy over Direct FT (Avg. shifts 57.08% → 59.55%), confirming that distillation-based preservation is necessary. Static SD still leaves a substantial gap to TRACER (59.55% → 65.46%, +5.9), largest where the pretrained representation is most informative and the static-SD anchor bias is most punishing: IN-R 72.87% → 79.30% (+6.4), IN-A 42.33% → 54.72% (+12.4). These are exactly the renditions/adversarial shifts where Theorem 3.4 predicts the WMA teacher’s benefit: it removes the static-SD tasksubspace bias while preserving the orthogonal directions that matter for unseen styles and natural adversarial examples.

Ablation Studies. Beyond the primary results, we conducted comprehensive ablation studies across four axes (detailed in §B) to thoroughly validate TRACER’s design choices. (1) Multi-perspective distillation (Table 8): (a) FD and CRD are the strongest single components. (b) The four components are complementary: every pair including FD or CRD beats its singletons. (c) All four together is best overall: the full TRACER setting achieves the highest Avg. All; FD stabilizes features, CRD preserves relational structure, ICL enriches mutual information, and Cross-KD blends relational and interactive cues. (2) Distillation strength (Table 9): Sweeping λSD from 0.1 to 10.0 reveals that moderate values (≈1.0–2.0) achieve the best ID-OOD trade-off, while higher values improve calibration at the cost of ID accuracy. (3) Teacher update frequency (Table 10): TRACER maintains stable OOD accuracy (∼64.0–64.2%) across update frequencies from every step to every 2500 steps, eliminating the need for brittle scheduling required by CaRot (Oh et al., 2024). (4) Beta-kernel shape (Table 11): Arcsine-like weighting (Beta(0.5, 0.5)) proves most effective. These extensive ablations confirm that TRACER’s design is both principled and robust.

Empirical Validation of Geometric Preservation. To verify our theoretical claim that TRACER preserves knowledge in the orthogonal subspace, we conduct a layer-wise representational similarity analysis using Centered Kernel Alignment (CKA) (Kornblith et al., 2019) on the CLIP ViT-B/16 image encoder (Figure 4). We extract feature maps from every layer (Patch Embeddings, Transformer Blocks 0–11, and the Final Projection) on the ImageNet validation set and compare finetuned models against the pretrained model using CKA and SVCCA (Raghu et al., 2017). As shown in Figure 4, Direct FT exhibits a drop in similarity in the deeper layers (Blocks 6–11) relative to the pretrained model. This confirms that catastrophic forgetting manifests as a Feature Distortion of high-level semantic representations. In contrast, TRACER maintains near-perfect similarity (> 0.97) across all layers. This provides empirical evidence for our geometric interpretation: TRACER successfully anchors the optimization to the pretrained geometry, performing surgical updates that adapt to the task without overwriting robust feature extractors.

Computational Efficiency and Complexity. TRACER also offers efficiency advantages over CaRot (Oh et al., 2024), whose spectral regularization scales as $\mathcal { O } ( d ^ { 3 } )$ . TRACER’s distillation operates on batch similarity matrices $( \mathcal { O } ( B ^ { 2 } )$ ), $B \ll d )$ , and the WMA update matches standard EMA cost $( \mathcal { O } ( P ) )$ . As shown in Table 4, this reduces training time per epoch compared to CaRot.

Table 4. Computational Efficiency. Computational efficiency comparison per epoch on ImageNet-1K using CLIP ViT-B/16 on an NVIDIA H100 GPU. 

<table><tr><td>Method</td><td>Cost</td><td>Time / Epoch</td><td>Overhead</td><td>Avg. OOD Acc.</td></tr><tr><td>Direct FT</td><td> $\mathcal{O}(P)$ </td><td> $\sim 16 \text{ min}$ </td><td> $1.00\times$ </td><td> $57.08\%$ </td></tr><tr><td>CaRot (Oh et al., 2024)</td><td> $\mathcal{O}(B^{2} + d^{3})$ </td><td> $\sim 29 \text{ min}$ </td><td> $1.81\times$ </td><td> $62.54\%$ </td></tr><tr><td>TRACER (Ours)</td><td> $\mathcal{O}(B^{2})$ </td><td> $\sim 22 \text{ min}$ </td><td> $1.38\times$ </td><td> $64.07\%$ </td></tr></table>

Teacher Dynamics and Regularization Strength. Our theory posits that the WMA teacher in TRACER provides a more persistent regularizing signal than the EMA teacher used in methods like CaRot. To validate this, we track the KL divergence between teacher and student throughout training on ImageNet. As shown in Figure 5, for the EMA teacher, the KL decays steadily, indicating that the teacher is rapidly collapsing onto the student and its regularizing influence is diminishing. In contrast, the WMA teacher maintains a higher and more stable KL throughout the entire training process. This sustained divergence supports the view that the WMA teacher provides a persistent “restoring force,” as predicted by our analysis in §C.5.2. This helps prevent the student from converging to a narrow task-specific minimum and reflects the persistent regularization strength required for robust generalization.

Per-Layer CKA   
![](images/2f26e1577ad61a66c7e794dc2de74b2ce4817a60a1995fadcf9f7eeeb6d4e551.jpg)

<details>
<summary>line</summary>

| Layer Index | Pre vs Direct-FT | Pre vs TRACER | Direct-FT vs TRACER |
| ----------- | ---------------- | ------------- | ------------------- |
| patch_embed | 0.98             | 1.00          | 0.99                |
| block_0     | 0.94             | 0.99          | 0.97                |
| block_1     | 0.96             | 0.99          | 0.98                |
| block_2     | 0.95             | 0.99          | 0.97                |
| block_3     | 0.93             | 0.99          | 0.96                |
| block_4     | 0.91             | 0.99          | 0.94                |
| block_5     | 0.90             | 0.98          | 0.93                |
| block_6     | 0.82             | 0.97          | 0.85                |
| block_7     | 0.85             | 0.97          | 0.86                |
| block_8     | 0.88             | 0.97          | 0.88                |
| block_9     | 0.89             | 0.97          | 0.91                |
| block_10    | 0.87             | 0.97          | 0.89                |
| block_11    | 0.78             | 0.97          | 0.85                |
| In_post     | 0.81             | 0.97          | 0.87                |
| final_pkj   | 0.77             | 0.97          | 0.84                |
</details>

Per-Layer SVCCA   
![](images/748b9f90a37b2e2d0476b2d9e99cdf13ac0b57e44acdfcb03eec6c1a84ad5cb8.jpg)

<details>
<summary>line</summary>

| Layer Index       | Pre vs Direct-FT | Pre vs TRACER | Direct-FT vs TRACER |
| ----------------- | ---------------- | ------------- | ------------------- |
| patch_embed       | 1.00             | 1.00          | 1.00                |
| block_0           | 0.95             | 0.98          | 0.95                |
| block_1           | 0.93             | 0.97          | 0.94                |
| block_2           | 0.92             | 0.96          | 0.93                |
| block_3           | 0.90             | 0.95          | 0.92                |
| block_4           | 0.88             | 0.94          | 0.91                |
| block_5           | 0.85             | 0.93          | 0.88                |
| block_6           | 0.82             | 0.92          | 0.85                |
| block_7           | 0.78             | 0.91          | 0.82                |
| block_8           | 0.75             | 0.90          | 0.78                |
| block_9           | 0.72             | 0.89          | 0.75                |
| block_10          | 0.70             | 0.88          | 0.73                |
| block_11          | 0.68             | 0.87          | 0.72                |
| In_post            | 0.65             | 0.86          | 0.71                |
| final_proj        | 0.66             | 0.85          | 0.72                |
</details>

Figure 4. Layer-wise Representational Similarity. We compare the internal representations of the Pretrained model against Direct FT and TRACER using CKA (left) and SVCCA (right) across all layers of the CLIP ViT-B/16 image encoder. TRACER (gold) preserves the geometric structure of the pretrained knowledge significantly better than Direct FT (pink), particularly in deeper layers.

![](images/0ace2f9d0b3ef8083dc691b4ba5368de24e3385c1db10fa066c8ff39385574c6.jpg)

<details>
<summary>line</summary>

| Training Step | CAROT-EMA, freq=500 | TRACER-BMA, freq=500 |
| ------------- | ------------------- | -------------------- |
| 0             | 0.12                | 0.22                 |
| 500           | 0.03                | 0.18                 |
| 1000          | 0.02                | 0.15                 |
| 2000          | 0.015               | 0.12                 |
| 3000          | 0.01                | 0.10                 |
| 4000          | 0.008               | 0.08                 |
| 5000          | 0.005               | 0.06                 |
</details>

Teacher Avg Entropy BMA vs EMA (freq=500)   
![](images/e2621cf109ee748a4c5d9408fc728daed202d1f04997734c9f29f7eb338ea707.jpg)

<details>
<summary>line</summary>

| Training Step | CAROT-EMA, freq=500 | TRACER-BMA, freq=500 |
| ------------- | ------------------- | -------------------- |
| 0             | 1.1                 | 1.1                  |
| 1000          | 0.7                 | 1.0                  |
| 2000          | 0.6                 | 0.9                  |
| 3000          | 0.55                | 0.85                 |
| 4000          | 0.52                | 0.83                 |
| 5000          | 0.5                 | 0.82                 |
</details>

Teacher Avg Confidence BMA vs EMA (freq=500)   
![](images/cbc9e50d56675f8b2f54f00cf4e432e1c95ecfcfb302f627107b848b0be50828.jpg)

<details>
<summary>line</summary>

| Training Step | CAROT-EMA, freq=500 | TRACER-BMA, freq=500 |
| ------------- | ------------------- | -------------------- |
| 0             | 0.68                | 0.69                 |
| 1000          | 0.74                | 0.71                 |
| 2000          | 0.76                | 0.72                 |
| 3000          | 0.77                | 0.73                 |
| 4000          | 0.78                | 0.73                 |
| 5000          | 0.78                | 0.73                 |
</details>

Figure 5. Teacher–Student Knowledge Gap During Training. Compared to the EMA teacher (blue), which shows rapidly vanishing KL divergence and thus a weakening regularization signal (left), the WMA teacher (orange) sustains a higher and more stable KL gap. This stability is supported by higher teacher entropy (middle) and moderated confidence (right), preventing overfitting. Together, these trends confirm that WMA provides a stronger and more persistent self-distillation signal than EMA.

Algorithmic Simplicity. While EMA-based methods often require complex, sparse update schedules (e.g., updating only every 500 steps with linear warmup and careful momentum tuning) to prevent collapse, TRACER is robust to update frequency. As shown in Table 10, TRACER maintains consistent performance (∼ 64.0 − 64.2% OOD accuracy) whether the teacher is updated every step or every 2500 steps, reducing the need for brittle hyperparameter tuning. Figure 6 illustrates this failure mode in prior methods: without careful tuning, the EMA teacher in CaRot collapses immediately when updated at every step, whereas TRACER’s WMA teacher remains stable even under a dense update schedule.

# 6. Conclusion

We proved that TRACER’s trajectory-averaging WMA teacher, unlike its EMA counterpart, maintains a persistent regularizing force over finite training horizons. This force continuously anchors the model to its robust pretrained initialization, helping prevent overfitting and improving outof-distribution performance. Our extensive ablation studies confirm that TRACER’s design is both principled and robust to hyperparameter choices across distillation strength, update frequency, and kernel shape. Our work bridges the geometry of finetuning with the practical design of robust methods, and these principles motivate future extensions to parameter-efficient methods, continual learning, randomfeature analyses of the same trajectory-regularization idea, and larger vision–language backbones.

Limitations and Scope. Our empirical evaluation focuses on CLIP-style vision–language backbones and standard vision robustness benchmarks. We do not yet provide experiments on broader modalities or multimodal LLM settings, and our theoretical analysis is developed in the linearized image/text-encoder regime; we discuss concrete extensions to random-feature settings, to larger VLMs, and the relationship to prompt-based adaptation in §F.

# Impact Statement

This work develops methods for improving the robustness and calibration of finetuned multimodal models under distribution shift. Our primary motivation is enhancing the reliability of foundation models as they are deployed in realworld applications where inputs may differ from training data. Robust and calibrated models are particularly valuable in safety-critical domains such as medical imaging and autonomous systems, where overconfident predictions under shift can lead to harmful outcomes. Our method is purely defensive in nature: it mitigates catastrophic forgetting and improves generalization. We see no obvious pathways by which this work accelerates harmful capabilities.

# Acknowledgements

This research was conducted by the ARC Centre of Excellence for Automated Decision-Making and Society (CE200100005), and funded by the Australian Government through the Australian Research Council. This research was supported by The University of Melbourne’s Research Computing Services and the Petascale Campus Initiative. We would like to thank Navid Akhavan Attar, Aryan Yazdan Parast, and Hugo Lyons Keenan for valuable discussions and feedback.

# Conflict of Interest Disclosure

The authors declare no financial conflicts of interest.

# References

Alayrac, J., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J. L., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, A., and Simonyan, K. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.   
Aljundi, R., Lin, M., Goujaud, B., and Bengio, Y. Gradient based sample selection for online continual learning. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 11816–11825, 2019.   
Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-Paz, D. Invariant risk minimization. CoRR, abs/1907.02893, 2019. URL http://arxiv.org/ abs/1907.02893.   
Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum, J., and Katz, B. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 9448–9458, 2019.   
Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. Computer Sci-

ence. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3): 8, 2023.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in selfsupervised vision transformers. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pp. 9630– 9640. IEEE, 2021. doi: 10.1109/ICCV48922.2021.00951. URL https://doi.org/10.1109/ICCV48922. 2021.00951.

Deng, J., Dong, W., Socher, R., Li, L., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA, pp. 248–255. IEEE Computer Society, 2009. doi: 10.1109/CVPR.2009.5206848. URL https://doi.org/10.1109/CVPR.2009. 5206848.

Deng, L. The MNIST database of handwritten digit images for machine learning research [best of the web]. IEEE Signal Process. Mag., 29(6):141–142, 2012. doi: 10. 1109/MSP.2012.2211477. URL https://doi.org/ 10.1109/MSP.2012.2211477.

Desai, K. and Johnson, J. Virtex: Learning visual representations from textual annotations. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 11162–11173. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.01101. URL https: //openaccess.thecvf.com/content/ CVPR2021/html/Desai\_VirTex\_Learning\_ Visual\_Representations\_From\_Textual Annotations\_CVPR\_2021\_paper.html.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum? id=YicbFdNTTy.

Fang, A., Ilharco, G., Wortsman, M., Wan, Y., Shankar, V., Dave, A., and Schmidt, L. Data determines distributional robustness in contrastive language image pretraining (CLIP). In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 6216–6234. PMLR, 2022. URL https://proceedings.mlr.press/ v162/fang22a.html.

Fang, A., Jose, A. M., Jain, A., Schmidt, L., Toshev, A. T., and Shankar, V. Data filtering networks. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/ forum?id=KAk6ngZ09F.   
Fang, Y., Wang, W., Xie, B., Sun, Q., Wu, L., Wang, X., Huang, T., Wang, X., and Cao, Y. EVA: exploring the limits of masked visual representation learning at scale. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 19358–19369. IEEE, 2023. doi: 10. 1109/CVPR52729.2023.01855. URL https://doi. org/10.1109/CVPR52729.2023.01855.   
Fang, Z., Wang, J., Hu, X., Wang, L., Yang, Y., and Liu, Z. Compressing visual-linguistic model via knowledge distillation. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pp. 1408–1418. IEEE, 2021. doi: 10.1109/ICCV48922.2021.00146. URL https://doi. org/10.1109/ICCV48922.2021.00146.   
Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/ forum?id=rJl-b3RcF7.   
French, R. M. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999.   
Furlanello, T., Lipton, Z. C., Tschannen, M., Itti, L., and Anandkumar, A. Born-again neural networks. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 1602–1611. PMLR, 2018. URL http://proceedings.mlr.press/ v80/furlanello18a.html.   
Garrido, Q., Chen, Y., Bardes, A., Najman, L., and Le-Cun, Y. On the duality between contrastive and noncontrastive self-supervised learning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/ forum?id=kDEL91Dufpa.   
Goyal, S., Kumar, A., Garg, S., Kolter, Z., and Raghunathan, A. Finetune like you pretrain: Improved finetuning of zero-shot vision models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 19338–19347. IEEE, 2023. doi: 10.1109/CVPR52729.

2023.01853. URL https://doi.org/10.1109/ CVPR52729.2023.01853.   
Grill, J., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., Doersch, C., Pires, B. Á., Guo, Z., Azar, M. G., Piot, B., Kavukcuoglu, K., Munos, R., and Valko, M. Bootstrap your own latent - A new approach to selfsupervised learning. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
Han, J., Lin, Z., Sun, Z., Gao, Y., Yan, K., Ding, S., Gao, Y., and Xia, G. Anchor-based robust finetuning of visionlanguage models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 26909–26918. IEEE, 2024. doi: 10.1109/CVPR52733.2024.02542. URL https:// doi.org/10.1109/CVPR52733.2024.02542.   
Hao, Y., Pan, X., Zhang, H., Ye, C., Pan, R., and Zhang, T. Understanding overadaptation in supervised finetuning: The role of ensemble methods. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net, 2025. URL https://openreview.net/ forum?id=1xsW6tvMb3.   
HaoChen, J. Z. and Ma, T. A theoretical study of inductive biases in contrastive learning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum? id=AuEgNlEAmed.   
HaoChen, J. Z., Wei, C., Kumar, A., and Ma, T. Beyond separability: Analyzing the linear transferability of contrastive representations to related subpopulations. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.   
He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. B. Momentum contrast for unsupervised visual representation learning. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pp. 9726–9735. Computer Vision Foundation / IEEE, 2020. doi: 10.1109/CVPR42600. 2020.00975. URL https://doi.org/10.1109/ CVPR42600.2020.00975.   
Hendrycks, D. and Dietterich, T. G. Benchmarking neural network robustness to common corruptions and perturbations. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA,

May 6-9, 2019. OpenReview.net, 2019. URL https: //openreview.net/forum?id=HJz6tiCqYm.   
Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., Song, D., Steinhardt, J., and Gilmer, J. The many faces of robustness: A critical analysis of out-of-distribution generalization. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pp. 8320–8329. IEEE, 2021a. doi: 10.1109/ICCV48922.2021.00823. URL https://doi. org/10.1109/ICCV48922.2021.00823.   
Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., and Song, D. Natural adversarial examples. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 15262–15271. Computer Vision Foundation / IEEE, 2021b. doi: 10.1109/CVPR46437.2021.01501. URL https://openaccess.thecvf.com/ content/CVPR2021/html/Hendrycks\_ Natural\_Adversarial\_Examples\_CVPR\_ 2021\_paper.html.   
Hinton, G. E., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, 2015. URL http://arxiv.org/abs/1503.02531.   
Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., and Gelly, S. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 2790–2799. PMLR, 2019. URL http://proceedings.mlr.press/ v97/houlsby19a.html.   
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.   
Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., and Schmidt, L. Openclip, July 2021. URL https://doi.org/10. 5281/zenodo.5143773. If you use this software, please cite it as below.   
Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. P., and Wilson, A. G. Averaging weights leads to wider optima and better generalization. In Proceedings of the Thirty-Fourth Conference on Uncertainty

in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018, pp. 876–885. AUAI Press, 2018. URL http://auai.org/uai2018/ proceedings/papers/313.pdf.   
Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel: Convergence and generalization in neural networks. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips. cc/paper\_files/paper/2018/file/ 5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper. pdf.   
Jang, D., Yun, S., and Han, D. Model stock: All we need is just a few fine-tuned models. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLIV, volume 15102 of Lecture Notes in Computer Science, pp. 207–223. Springer, 2024. doi: 10.1007/978-3-031-72784-9\_12. URL https://doi. org/10.1007/978-3-031-72784-9\_12.   
Ji, W., Deng, Z., Nakada, R., Zou, J., and Zhang, L. The power of contrast for feature learning: A theoretical analysis. J. Mach. Learn. Res., 24:330:1– 330:78, 2023. URL http://jmlr.org/papers/ v24/21-1501.html.   
Jia, C., Yang, Y., Xia, Y., Chen, Y., Parekh, Z., Pham, H., Le, Q. V., Sung, Y., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 4904–4916. PMLR, 2021. URL http://proceedings.mlr.press/ v139/jia21b.html.   
Jia, M., Tang, L., Chen, B., Cardie, C., Belongie, S. J., Hariharan, B., and Lim, S. Visual prompt tuning. In Avidan, S., Brostow, G. J., Cissé, M., Farinella, G. M., and Hassner, T. (eds.), Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXIII, Lecture Notes in Computer Science, pp. 709–727. Springer, 2022. doi: 10.1007/978-3-031-19827-4\_41. URL https://doi. org/10.1007/978-3-031-19827-4\_41.   
Jung, S., Ahn, H., Cha, S., and Moon, T. Continual learning with node-importance based adaptive group sparse regularization. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.   
Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. E. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 3519–3529. PMLR, 2019. URL http://proceedings.mlr.press/ v97/kornblith19a.html.   
Kumar, A., Raghunathan, A., Jones, R. M., Ma, T., and Liang, P. Fine-tuning can distort pretrained features and underperform out-of-distribution. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id=UYneFzXSJWh.   
Laine, S. and Aila, T. Temporal ensembling for semisupervised learning. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. Open-Review.net, 2017. URL https://openreview. net/forum?id=BJ6oOfqge.   
LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proc. IEEE, 86(11):2278–2324, 1998. doi: 10.1109/5.726791. URL https://doi.org/10.1109/5.726791.   
Li, X., Grandvalet, Y., and Davoine, F. Explicit inductive bias for transfer learning with convolutional networks. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 2830– 2839. PMLR, 2018. URL http://proceedings. mlr.press/v80/li18a.html.   
Li, X., Fang, Y., Liu, M., Ling, Z., Tu, Z., and Su, H. Distilling large vision-language model with out-of-distribution generalizability. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 2492–2503. IEEE, 2023a. doi: 10.1109/ICCV51070.2023.00236. URL https://doi. org/10.1109/ICCV51070.2023.00236.   
Li, X., Wang, Z., and Xie, C. Clipa-v2: Scaling CLIP training with 81.1% zero-shot imagenet accuracy within a \$10, 000 budget; an extra \$4, 000 unlocks 81.8% accuracy. CoRR, abs/2306.15658, 2023b. doi: 10.48550/ARXIV.

2306.15658. URL https://doi.org/10.48550/ arXiv.2306.15658.

Li, X. L. and Liang, P. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pp. 4582–4597. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.ACL-LONG. 353. URL https://doi.org/10.18653/v1/ 2021.acl-long.353.

Li, Y., Fan, H., Hu, R., Feichtenhofer, C., and He, K. Scaling language-image pre-training via masking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 23390–23400. IEEE, 2023c. doi: 10.1109/CVPR52729.2023.02240. URL https:// doi.org/10.1109/CVPR52729.2023.02240.

Li, Z. and Hoiem, D. Learning without forgetting. IEEE Trans. Pattern Anal. Mach. Intell., 40(12):2935–2947, 2018. doi: 10.1109/TPAMI.2017.2773081. URL https: //doi.org/10.1109/TPAMI.2017.2773081.

Li, Z., Li, X., Fu, X., Zhang, X., Wang, W., Chen, S., and Yang, J. Promptkd: Unsupervised prompt distillation for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 26607–26616. IEEE, 2024. doi: 10.1109/CVPR52733.2024.02513. URL https://doi.org/10.1109/CVPR52733. 2024.02513.

Liang, C., Yu, J., Yang, M., Brown, M., Cui, Y., Zhao, T., Gong, B., and Zhou, T. Module-wise adaptive distillation for multimodality foundation models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Liu, H., HaoChen, J. Z., Gaidon, A., and Ma, T. Selfsupervised learning is more robust to dataset imbalance. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview. net/forum?id=4AZz9osqrar.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Mao, X., Chen, Y., Jia, X., Zhang, R., Xue, H., and Li, Z. Context-aware robust fine-tuning. Int. J. Comput. Vis., 132(5):1685–1700, 2024. doi: 10.1007/ S11263-023-01951-2. URL https://doi.org/10. 1007/s11263-023-01951-2.   
McCloskey, M. and Cohen, N. J. Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of learning and motivation, 24:109–165, 1989.   
Mobahi, H., Farajtabar, M., and Bartlett, P. L. Selfdistillation amplifies regularization in hilbert space. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
Nakada, R., Gulluk, H. I., Deng, Z., Ji, W., Zou, J., and Zhang, L. Understanding multimodal contrastive learning and incorporating unpaired data. In International Conference on Artificial Intelligence and Statistics, 25-27 April 2023, Palau de Congressos, Valencia, Spain, volume 206 of Proceedings of Machine Learning Research, pp. 4348– 4380. PMLR, 2023. URL https://proceedings. mlr.press/v206/nakada23a.html.   
Nam, G., Heo, B., and Lee, J. Lipsum-ft: Robust fine-tuning of zero-shot models using random text guidance. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview. net/forum?id=2JF8mJRJ7M.   
Oh, C., Lim, H., Kim, M., Han, D., Yun, S., Choo, J., Hauptmann, A., Cheng, Z., and Song, K. Towards calibrated robust fine-tuning of vision-language models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.   
Oh, C., Li, Y., Song, K., Yun, S., and Han, D. Dawin: Training-free dynamic weight interpolation for robust adaptation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https: //openreview.net/forum?id=L8e7tBf4pP.   
Pi, R., Yao, L., Han, J., Liang, X., Zhang, W., and Xu, H. Ins-detclip: Aligning detection model to follow human-language instruction. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum? id=M0MF4t3hE9.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 2021. URL http://proceedings.mlr.press/v139/ radford21a.html.   
Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J. SVCCA: singular vector canonical correlation analysis for deep learning dynamics and interpretability. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 6076–6085, 2017.   
Rebuffi, S., Kolesnikov, A., Sperl, G., and Lampert, C. H. icarl: Incremental classifier and representation learning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 5533–5542. IEEE Computer Society, 2017. doi: 10.1109/CVPR.2017.587. URL https://doi.org/10.1109/CVPR.2017.587.   
Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do imagenet classifiers generalize to imagenet? In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 5389–5400. PMLR, 2019. URL http://proceedings.mlr.press/ v97/recht19a.html.   
Robins, A. Catastrophic forgetting, rehearsal and pseudorehearsal. Connection Science, 7(2):123–146, 1995.   
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M. S., Berg, A. C., and Fei-Fei, L. Imagenet large scale visual recognition challenge. Int. J. Comput. Vis., 115(3):211–252, 2015. doi: 10.1007/ S11263-015-0816-Y. URL https://doi.org/10. 1007/s11263-015-0816-y.   
Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., and Hadsell, R. Progressive neural networks. CoRR, abs/1606.04671, 2016. URL http://arxiv.org/ abs/1606.04671.   
Sariyildiz, M. B., Perez, J., and Larlus, D. Learning visual representations with caption annotations. In Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings,

Part VIII, volume 12353 of Lecture Notes in Computer Science, pp. 153–170. Springer, 2020. doi: 10.1007/ 978-3-030-58598-3\_10. URL https://doi.org/ 10.1007/978-3-030-58598-3\_10.   
Saunshi, N., Plevrakis, O., Arora, S., Khodak, M., and Khandeparkar, H. A theoretical analysis of contrastive unsupervised representation learning. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pp. 5628–5637. PMLR, 2019. URL http://proceedings.mlr.press/ v97/saunshi19a.html.   
Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S., Crowson, K., Schmidt, L., Kaczmarczyk, R., and Jitsev, J. LAION-5B: an open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.   
Shen, K., Jones, R. M., Kumar, A., Xie, S. M., HaoChen, J. Z., Ma, T., and Liang, P. Connect, not collapse: Explaining contrastive learning for unsupervised domain adaptation. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 19847–19878. PMLR, 2022a. URL https://proceedings.mlr. press/v162/shen22d.html.   
Shen, S., Li, L. H., Tan, H., Bansal, M., Rohrbach, A., Chang, K., Yao, Z., and Keutzer, K. How much can CLIP benefit vision-and-language tasks? In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022b. URL https://openreview.net/forum? id=zf\_Ll3HZWgy.   
Shu, Y., Guo, X., Wu, J., Wang, X., Wang, J., and Long, M. Clipood: Generalizing CLIP to out-ofdistributions. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 31716–31731. PMLR, 2023. URL https://proceedings.mlr.press/ v202/shu23a.html.   
Sun, Q., Fang, Y., Wu, L., Wang, X., and Cao, Y. EVA-CLIP: improved training techniques for CLIP at scale. CoRR, abs/2303.15389, 2023. doi: 10.48550/ARXIV.

2303.15389. URL https://doi.org/10.48550/ arXiv.2303.15389.   
Taori, R., Dave, A., Shankar, V., Carlini, N., Recht, B., and Schmidt, L. Measuring robustness to natural distribution shifts in image classification. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
Tarvainen, A. and Valpola, H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 1195–1204, 2017.   
Tian, J., Dai, X., Ma, C., He, Z., Liu, Y., and Kira, Z. Trainable projected gradient method for robust finetuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 7836–7845. IEEE, 2023a. doi: 10.1109/CVPR52729.2023.00757. URL https:// doi.org/10.1109/CVPR52729.2023.00757.   
Tian, J., Liu, Y., Smith, J. S., and Kira, Z. Fast trainable projection for robust fine-tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023b.   
Tian, Y. Understanding deep contrastive learning via coordinate-wise optimization. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.   
Titsias, M. K., Schwarz, J., de G. Matthews, A. G., Pascanu, R., and Teh, Y. W. Functional regularisation for continual learning with gaussian processes. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum? id=HkxCzeHFDB.   
Tschannen, M., Gritsenko, A. A., Wang, X., Naeem, M. F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Hénaff, O. J., Harmsen, J., Steiner, A., and Zhai, X. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. CoRR, abs/2502.14786, 2025. doi: 10.48550/ARXIV.2502.14786. URL https: //doi.org/10.48550/arXiv.2502.14786.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 5998–6008, 2017.   
Wang, F., Zhou, D., Ye, H., and Zhan, D. FOSTER: feature boosting and compression for class-incremental learning. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXV, volume 13685 of Lecture Notes in Computer Science, pp. 398–414. Springer, 2022a. doi: 10.1007/978-3-031-19806-9\_23. URL https://doi. org/10.1007/978-3-031-19806-9\_23.   
Wang, H., Ge, S., Lipton, Z. C., and Xing, E. P. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 10506–10518, 2019.   
Wang, T. and Isola, P. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 9929–9939. PMLR, 2020. URL http://proceedings.mlr.press/ v119/wang20k.html.   
Wang, Z., Codella, N., Chen, Y., Zhou, L., Dai, X., Xiao, B., Yang, J., You, H., Chang, K., Chang, S., and Yuan, L. Multimodal adaptive distillation for leveraging unimodal encoders for vision-language tasks. CoRR, abs/2204.10496, 2022b. doi: 10.48550/ARXIV. 2204.10496. URL https://doi.org/10.48550/ arXiv.2204.10496.   
Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Lopes, R. G., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L. Model soups: averaging weights of multiple finetuned models improves accuracy without increasing inference time. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 23965–23998. PMLR, 2022a. URL https://proceedings.mlr. press/v162/wortsman22a.html.   
Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., Lopes, R. G., Hajishirzi, H., Farhadi, A., Namkoong, H., and Schmidt, L. Robust fine-tuning of zero-shot models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New

Orleans, LA, USA, June 18-24, 2022, pp. 7949–7961. IEEE, 2022b. doi: 10.1109/CVPR52688.2022.00780. URL https://doi.org/10.1109/CVPR52688. 2022.00780.   
Wu, K., Peng, H., Zhou, Z., Xiao, B., Liu, M., Yuan, L., Xuan, H., Valenzuela, M., Chen, X. S., Wang, X., Chao, H., and Hu, H. Tinyclip: CLIP distillation via affinity mimicking and weight inheritance. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 21913–21923. IEEE, 2023. doi: 10.1109/ICCV51070.2023.02008. URL https://doi.org/10.1109/ICCV51070. 2023.02008.   
Xu, H., Xie, S., Tan, X. E., Huang, P., Howes, R., Sharma, V., Li, S., Ghosh, G., Zettlemoyer, L., and Feichtenhofer, C. Demystifying CLIP data. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024a. URL https://openreview.net/forum? id=5BCFlnfE1g.   
Xu, H., Xie, S., Tan, X. E., Huang, P., Howes, R., Sharma, V., Li, S., Ghosh, G., Zettlemoyer, L., and Feichtenhofer, C. Demystifying CLIP data. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024b. URL https://openreview.net/forum? id=5BCFlnfE1g.   
Xue, Y., Joshi, S., Gan, E., Chen, P., and Mirzasoleiman, B. Which features are learnt by contrastive learning? on the role of simplicity bias in class collapse and feature suppression. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 38938–38970. PMLR, 2023. URL https://proceedings.mlr.press/ v202/xue23d.html.   
Xue, Y., Joshi, S., Nguyen, D., and Mirzasoleiman, B. Understanding the robustness of multi-modal contrastive learning to distribution shift. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum? id=rtl4XnJYBh.   
Yan, S., Xie, J., and He, X. DER: dynamically expandable representation for class incremental learning. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 3014–3023. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.00303. URL https: //openaccess.thecvf.com/content/

CVPR2021/html/Yan\_DER\_Dynamically\_ Expandable\_Representation\_for\_Class Incremental\_Learning\_CVPR\_2021\_paper. html.   
Yang, C., An, Z., Huang, L., Bi, J., Yu, X., Yang, H., Diao, B., and Xu, Y. CLIP-KD: an empirical study of CLIP model distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 15952–15962. IEEE, 2024a. doi: 10.1109/CVPR52733.2024.01510. URL https://doi.org/10.1109/CVPR52733. 2024.01510.   
Yang, Z., Zhang, A., Wiseman, S., Kong, X., Ye, K., and Yin, D. Memory retaining finetuning via distillation. In NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability, 2024b. URL https://openreview.net/forum? id=NEYvQUr6em.   
Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., and Wu, Y. Coca: Contrastive captioners are imagetext foundation models. Trans. Mach. Learn. Res., 2022, 2022. URL https://openreview.net/forum? id=Ee277P3AYC.   
Yuan, X., Lin, Z., Kuen, J., Zhang, J., Wang, Y., Maire, M., Kale, A., and Faieta, B. Multimodal contrastive training for visual representation learning. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pp. 6995–7004. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.00692. URL https://openaccess.thecvf. com/content/CVPR2021/html/Yuan\_ Multimodal\_Contrastive\_Training\_for\_ Visual\_Representation\_Learning\_CVPR\_ 2021\_paper.html.   
Zenke, F., Poole, B., and Ganguli, S. Continual learning through synaptic intelligence. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pp. 3987–3995. PMLR, 2017. URL http://proceedings.mlr.press/ v70/zenke17a.html.   
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 11941–11952. IEEE, 2023. doi: 10.1109/ICCV51070. 2023.01100. URL https://doi.org/10.1109/ ICCV51070.2023.01100.

Zhang, L., Song, J., Gao, A., Chen, J., Bao, C., and Ma, K. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pp. 3712–3721. IEEE, 2019. doi: 10. 1109/ICCV.2019.00381. URL https://doi.org/ 10.1109/ICCV.2019.00381.   
Zhang, M., Sohoni, N. S., Zhang, H. R., Finn, C., and Ré, C. Correct-n-contrast: a contrastive approach for improving robustness to spurious correlations. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 26484–26516. PMLR, 2022a. URL https://proceedings.mlr. press/v162/zhang22z.html.   
Zhang, R., Guo, Z., Zhang, W., Li, K., Miao, X., Cui, B., Qiao, Y., Gao, P., and Li, H. Pointclip: Point cloud understanding by CLIP. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 8542–8552. IEEE, 2022b. doi: 10.1109/CVPR52688.2022.00836. URL https://doi.org/10.1109/CVPR52688. 2022.00836.   
Zhang, Y., Jiang, H., Miura, Y., Manning, C. D., and Langlotz, C. P. Contrastive learning of medical visual representations from paired images and text. In Proceedings of the Machine Learning for Healthcare Conference, MLHC 2022, 5-6 August 2022, Durham, NC, USA, volume 182 of Proceedings of Machine Learning Research, pp. 2–25. PMLR, 2022c. URL https://proceedings.mlr. press/v182/zhang22a.html.   
Zhang, Z. and Sabuncu, M. R. Self-distillation as instancespecific label smoothing. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.   
Zhou, K., Yang, J., Loy, C. C., and Liu, Z. Learning to prompt for vision-language models. International Journal of Computer Vision (IJCV), 2022.   
Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/ forum?id=1tZbq88f27.

# Appendix

# A. Additional Related Work

# A.1. Contrastive Language-Image Pretraining

Initial advancements in contrastive learning between vision and language modalities were made by Virtex (Desai & Johnson, 2021), ICMLM (Sariyildiz et al., 2020), and ConVIRT (Zhang et al., 2022c). These early approaches laid the groundwork for later models like CLIP (Radford et al., 2021; Ilharco et al., 2021) and ALIGN (Jia et al., 2021), which scaled contrastive techniques to larger datasets and model architectures. Subsequent work explores improved cross-modal interaction and training recipes (Yuan et al., 2021; Yu et al., 2022; Fang et al., 2023). Following these, several open-weight contrastive models have been introduced to improve CLIP’s performance and robustness (Sun et al., 2023; Zhai et al., 2023; Li et al., 2023b; Fang et al., 2024; Xu et al., 2024a; Schuhmann et al., 2022). For example, SigLIP (Zhai et al., 2023; Tschannen et al., 2025) modifies the contrastive loss by using a sigmoid function instead of softmax, and FLIP (Li et al., 2023c) integrates masking strategies to accelerate training.

# A.2. Theory of Contrastive Learning

A rich theoretical literature analyzes contrastive learning from first principles, characterizing when and why contrastive objectives recover useful features and class structure (Saunshi et al., 2019). The alignment–uniformity lens formalizes how pulling positives together while spreading embeddings uniformly on the sphere drives representation quality (Wang & Isola, 2020). For tractability, many works study linearized or simplified contrastive losses that replace log-exp with linear functions and show that their gradients align with those of standard objectives up to reweighting, enabling closed-form analysis and geometric insight (Ji et al., 2023; Tian, 2022; Nakada et al., 2023; Xue et al., 2024). This linearized viewpoint has proven effective in theoretical analyses across self-supervised contrastive learning (CL) (Ji et al., 2023; HaoChen et al., 2022; HaoChen & Ma, 2023; Shen et al., 2022a), multimodal contrastive learning (MMCL) (Nakada et al., 2023), non-contrastive methods (Liu et al., 2022), and supervised CL (Xue et al., 2023). Complementing these results, large-scale empirical studies suggest that many design choices of popular losses (e.g., log-exp, cosine similarity) are not essential for effective representation learning (Garrido et al., 2023).

# A.3. Finetuning, Forgetting, and Regularization

Catastrophic forgetting, adapting to new data at the expense of prior knowledge, has long been recognized as a central challenge in sequential and transfer learning (McCloskey & Cohen, 1989; French, 1999). Mitigation strategies include: (i) regularization, which constrains parameter updates via importance penalties or output consistency (Kirkpatrick et al., 2017; Zenke et al., 2017; Li & Hoiem, 2018); (ii) replay, which mixes current data with stored or synthesized memories (Robins, 1995; Rebuffi et al., 2017; Aljundi et al., 2019); and (iii) architectural growth, which expands capacity and distills across modules (Rusu et al., 2016; Yan et al., 2021; Wang et al., 2022a). L2-SP (Li et al., 2018) tethers the solution to the pretrained initialization via weight-space regularization, while output-space regularizers distill prior behaviors during adaptation (Li & Hoiem, 2018). Additionally, parameter-efficient finetuning methods such as adapters (Houlsby et al., 2019) and prefix tuning (Li & Liang, 2021) enable task adaptation without full model updates, thus mitigating forgetting. Among these, Low-Rank Adaptation (LoRA) (Hu et al., 2022) has gained prominence for finetuning large language models by injecting trainable low-rank matrices into existing weights, achieving competitive performance with reduced parameter updates and minimal forgetting. Further work explores functional regularization (Titsias et al., 2020) and knowledge-preserving contrastive losses (Jung et al., 2020) to encourage feature stability. As model sizes grow, scalable and minimally invasive adaptation techniques, balancing plasticity and stability, remain critical to continual and transfer learning paradigms.

# A.4. Robust Finetuning of CLIP

Robustness evaluates how well models maintain performance under distribution shifts, which can include synthetic corruptions (Hendrycks & Dietterich, 2019) as well as real-world variations in viewpoint, style, and time (Barbu et al., 2019; Hendrycks et al., 2021a; Wang et al., 2019; Recht et al., 2019). A standard protocol for evaluating CLIP-like models, proposed by (Taori et al., 2020), involves finetuning on ImageNet and measuring transfer performance on a suite of realistic OOD sets (ImageNet-V2, -A, -R, -Sketch, and ObjectNet), which is now standard practice. This evaluation highlights a central challenge: naive finetuning methods like Linear Probing (LP), which only trains a classification head, or Direct Full finetuning, which updates all parameters, often create a trade-off between in-distribution (ID) performance and OOD robustness. To address this, a diverse array of robust finetuning techniques has been developed. A prominent line of work involves post-hoc averaging or interpolating model weights. For instance, WiSE-FT (Wortsman et al., 2022b) averages the weights of the zero-shot and a fully finetuned model, while Model Soup (Wortsman et al., 2022a) averages the weights of multiple models found through a hyperparameter search. This concept is extended by Model Stock (Jang et al., 2024), which efficiently builds and averages a diverse set of minimally adapted models. Other post-hoc methods include TPGM (Tian et al., 2023a) and its efficient successor Fast TPGM (Tian et al., 2023b), which project finetuned weights back towards the initial weights, and DaWin (Oh et al., 2025), which introduces a training-free, dynamic interpolation where the mixing coefficient is decided on a per-sample basis using predictive entropy.

Beyond post-hoc modifications, many methods introduce regularization during the finetuning process itself. These can constrain the model in weight-space, such as L2-SP (Li et al., 2018) which penalizes weight deviation, or by maintaining an EMA of model parameters to find smoother, more robust solutions. Others operate in the output-space, where Knowledge Distillation (KD) (Hinton et al., 2015) aligns the student’s predictions with the robust zero-shot teacher. A particularly relevant strategy for Vision-Language Models is using the text modality for guidance. This includes continuing contrastive learning with supervised image-text pairs as in Finetune-Like-You-Pretrain (FLYP) (Goyal et al., 2023), aligning with fixed context-specific prompts in CAR-FT (Mao et al., 2024), regularizing the model’s energy function using random texts to preserve broad semantic alignment in Lipsum-FT (Nam et al., 2024), or improving discrimination with both positive and negative prompts as in CLIPood (Shu et al., 2023). Alternative strategies modify the training pipeline, such as the two-stage LP-FT approach (Kumar et al., 2022) which first finds a good head via linear probing before full finetuning. More advanced methods like CaRot (Oh et al., 2024) aim to simultaneously improve OOD accuracy and confidence calibration through a principled combination of contrastive learning and novel regularization terms.

# A.5. Knowledge Distillation and Self-Distillation

Knowledge Distillation (KD) was initially introduced for compression, where a smaller student learns from a larger teacher’s outputs (Hinton et al., 2015). The same principle underpins continual and transfer learning, where a pretrained model guides finetuning to preserve capabilities, often termed Learning without Forgetting (LwF) (Li & Hoiem, 2018). Self-distillation (SD) is a special case where the model learns from its own initial state (Zhang et al., 2019; Mobahi et al., 2020). Beyond single-modality SD, multimodal KD aligns internal signals and outputs to preserve cross-modal structure (Fang et al., 2021; Wang et al., 2022b; Li et al., 2023a; Liang et al., 2023; Li et al., 2024), with recent work demonstrating effective CLIP distillation via affinity matching and weight inheritance (Wu et al., 2023; Yang et al., 2024a).

# A.6. Dynamic Teachers, Weight Averaging, and Mode Connectivity

Temporal ensembling and EMA teachers stabilize training and improve targets (Laine & Aila, 2017; Tarvainen & Valpola, 2017), and they underpin momentum-encoder methods in self-supervised learning (He et al., 2020; Grill et al., 2020; Caron et al., 2021). Separately, model averaging and linear mode connectivity suggest that interpolations and averages often lie in flat, low-loss regions and improve robustness (Izmailov et al., 2018; Frankle & Carbin, 2019). Wise-FT leverages interpolation between pretrained and finetuned weights to strengthen OOD performance (Wortsman et al., 2022b;a).

Table 5. ImageNet calibration (ECE) on ViT-B/16. We report ECE (↓) on ImageNet and its distribution shift variants. In each column, the best value is bold and the second-best is underlined. 

<table><tr><td>Method</td><td>IN</td><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjNet</td><td>Avg.</td></tr><tr><td>ZS</td><td>0.057</td><td>0.055</td><td>0.054</td><td>0.097</td><td>0.085</td><td>0.078</td><td>0.074</td></tr><tr><td>LP-FT</td><td>0.051</td><td>0.089</td><td>0.061</td><td>0.205</td><td>0.166</td><td>0.212</td><td>0.147</td></tr><tr><td>FLYP</td><td>0.064</td><td>0.117</td><td>0.097</td><td>0.244</td><td>0.220</td><td>0.238</td><td>0.184</td></tr><tr><td>Lipsum-FT</td><td>0.038</td><td>0.052</td><td>0.043</td><td>0.129</td><td>0.102</td><td>0.132</td><td>0.091</td></tr><tr><td>CaRot</td><td>0.047</td><td>0.037</td><td>0.058</td><td>0.124</td><td>0.070</td><td>0.108</td><td>0.079</td></tr><tr><td>TRACER (Ours)</td><td>0.045</td><td>0.039</td><td>0.041</td><td>0.104</td><td>0.078</td><td>0.103</td><td>0.073</td></tr></table>

![](images/81e53bb5f340e3eaaf98b86b40648ea0722b83b3433f4727bea53764710da47c.jpg)

<details>
<summary>line</summary>

| Training Step | CAROT-EMA, freq=1 | TRACER-BMA, freq=1 |
| ------------- | ----------------- | ------------------ |
| 0             | 0.00              | 0.10               |
| 500           | 0.00              | 0.09               |
| 1000          | 0.00              | 0.08               |
| 1500          | 0.00              | 0.07               |
| 2000          | 0.00              | 0.06               |
| 2500          | 0.00              | 0.05               |
| 3000          | 0.00              | 0.04               |
| 3500          | 0.00              | 0.035              |
| 4000          | 0.00              | 0.03               |
| 4500          | 0.00              | 0.025              |
| 5000          | 0.00              | 0.02               |
</details>

(a) Teacher–Student KL

![](images/d963acc960d944bbf8b70a15beb7ff6a4f220b8beabee860211141df1b086a22.jpg)

<details>
<summary>line</summary>

| Training Step | CAROT-EMA, freq=1 | TRACER-BMA, freq=1 |
| ------------- | ----------------- | ------------------ |
| 0             | 1.1               | 1.1                |
| 500           | 0.7               | 0.9                |
| 1000          | 0.6               | 0.85               |
| 1500          | 0.55              | 0.8                |
| 2000          | 0.5               | 0.78               |
| 2500          | 0.48              | 0.76               |
| 3000          | 0.46              | 0.75               |
| 3500          | 0.45              | 0.74               |
| 4000          | 0.44              | 0.73               |
| 4500          | 0.43              | 0.72               |
| 5000          | 0.42              | 0.71               |
</details>

(b) Teacher Entropy

![](images/cdd4f1d775e3dc3be256722c3872f26efd273d3c4216a249be582e125069d440.jpg)

<details>
<summary>line</summary>

| Training Step | CAROT-EMA, freq=1 | TRACER-BMA, freq=1 |
| ------------- | ----------------- | ------------------ |
| 0             | 0.70              | 0.70               |
| 1000          | 0.76              | 0.73               |
| 2000          | 0.77              | 0.74               |
| 3000          | 0.78              | 0.745              |
| 4000          | 0.785             | 0.748              |
| 5000          | 0.79              | 0.75               |
</details>

(c) Teacher Confidence   
Figure 6. Comparison of Teacher Dynamics (Update Frequency = 1). We track the evolution of the teacher model for CaRot (EMA) and TRACER (WMA) when updated at every step. The EMA teacher (blue) rapidly collapses onto the student $( \mathrm { K L } \to 0 )$ , losing its regularizing capability. The WMA teacher (orange) maintains a persistent, stable gap, providing continuous regularization without needing brittle update schedules.

# B. Additional Experiments and Ablations

Experimental Protocol. We use the same seeds, and hyperparameter configurations as in the main experiments, varying only the stated factor per ablation.

Additional Experimental Results. To further demonstrate the generalizability of our method, we present results using the CLIP RN50 and ViT-L/14 backbones. A summary of these experiments, including ViT-B/16 for comparison, is provided in Table 3, with detailed results reported below.

Table 6. ImageNet results on CLIP ResNet50 

<table><tr><td rowspan="2">Method</td><td colspan="7">Acc.↑</td></tr><tr><td>IN</td><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>Avg. shifts</td></tr><tr><td>ZS</td><td>59.83</td><td>52.90</td><td>60.72</td><td>23.25</td><td>35.45</td><td>40.27</td><td>42.52</td></tr><tr><td>FT</td><td>76.21</td><td>64.87</td><td>50.66</td><td>18.11</td><td>33.90</td><td>42.32</td><td>41.97</td></tr><tr><td>LP-FT</td><td>76.25</td><td>64.48</td><td>49.55</td><td>18.60</td><td>33.33</td><td>42.13</td><td>41.62</td></tr><tr><td>FLYP</td><td>76.16</td><td>65.10</td><td>51.55</td><td>20.08</td><td>34.24</td><td>42.53</td><td>42.70</td></tr><tr><td>CaRot</td><td>76.12</td><td>65.36</td><td>52.16</td><td>19.32</td><td>34.05</td><td>42.67</td><td>42.71</td></tr><tr><td>TRACER (Ours)</td><td>76.48</td><td>65.58</td><td>51.54</td><td>19.52</td><td>34.34</td><td>42.66</td><td>42.73</td></tr></table>

ECE↓ 

<table><tr><td>ZS</td><td>0.0624</td><td>0.0559</td><td>0.0530</td><td>0.2048</td><td>0.0740</td><td>0.0899</td><td>0.0955</td></tr><tr><td>FT</td><td>0.0983</td><td>0.1623</td><td>0.1860</td><td>0.4692</td><td>0.2824</td><td>0.3023</td><td>0.2804</td></tr><tr><td>LP-FT</td><td>0.1042</td><td>0.1759</td><td>0.2709</td><td>0.5184</td><td>0.3520</td><td>0.3197</td><td>0.3274</td></tr><tr><td>FLYP</td><td>0.0516</td><td>0.0872</td><td>0.1439</td><td>0.3872</td><td>0.2021</td><td>0.2432</td><td>0.2127</td></tr><tr><td>CaRot</td><td>0.0471</td><td>0.0601</td><td>0.0948</td><td>0.3435</td><td>0.3435</td><td>0.2127</td><td>0.2109</td></tr><tr><td>TRACER (Ours)</td><td>0.0470</td><td>0.0564</td><td>0.1176</td><td>0.3456</td><td>0.1741</td><td>0.2097</td><td>0.1807</td></tr></table>

Table 7. ImageNet results on CLIP ViT-L/14 

<table><tr><td rowspan="2">Method</td><td colspan="7">Acc.↑</td></tr><tr><td>IN</td><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>Avg. shifts</td></tr><tr><td>ZS</td><td>75.55</td><td>69.85</td><td>87.85</td><td>70.76</td><td>59.61</td><td>66.59</td><td>70.93</td></tr><tr><td>FT</td><td>84.74</td><td>75.32</td><td>75.36</td><td>55.65</td><td>54.44</td><td>59.76</td><td>64.11</td></tr><tr><td>LP-FT</td><td>85.26</td><td>76.76</td><td>80.21</td><td>55.95</td><td>56.84</td><td>60.12</td><td>65.98</td></tr><tr><td>FLYP</td><td>86.19</td><td>78.21</td><td>83.81</td><td>68.85</td><td>60.20</td><td>66.15</td><td>71.44</td></tr><tr><td>CaRot</td><td>86.95</td><td>79.28</td><td>87.96</td><td>72.68</td><td>62.66</td><td>68.05</td><td>74.13</td></tr><tr><td>TRACER (Ours)</td><td>86.27</td><td>78.54</td><td>89.70</td><td>74.87</td><td>63.71</td><td>69.76</td><td>75.32</td></tr><tr><td colspan="8">ECE↓</td></tr><tr><td>ZS</td><td>0.0590</td><td>0.0686</td><td>0.0339</td><td>0.0640</td><td>0.1037</td><td>0.0852</td><td>0.0711</td></tr><tr><td>FT</td><td>0.1056</td><td>0.1741</td><td>0.1613</td><td>0.3151</td><td>0.3234</td><td>0.2865</td><td>0.2521</td></tr><tr><td>LP-FT</td><td>0.0993</td><td>0.1531</td><td>0.0872</td><td>0.2593</td><td>0.2613</td><td>0.2572</td><td>0.2036</td></tr><tr><td>FLYP</td><td>0.0729</td><td>0.1219</td><td>0.0621</td><td>0.1443</td><td>0.2164</td><td>0.1903</td><td>0.1470</td></tr><tr><td>CaRot</td><td>0.0349</td><td>0.0634</td><td>0.0353</td><td>0.0732</td><td>0.0914</td><td>0.1051</td><td>0.0737</td></tr><tr><td>TRACER (Ours)</td><td>0.0507</td><td>0.0581</td><td>0.0442</td><td>0.0665</td><td>0.1052</td><td>0.0918</td><td>0.0732</td></tr></table>

Ablation 1: Multi-perspective distillation. The ablation study on multi-perspective distillation (Table 8) quantifies the contribution of each perspective to out-of-distribution (OOD) accuracy and calibration. Results show that CRD and FD emerge as the strongest individual components for OOD accuracy and ECE, respectively, while combining all four perspectives yields the best overall performance and remains among the top performers on OOD metrics. These findings highlight the complementary nature of the terms: FD stabilizes features, CRD preserves batch-level relational structure, ICL enriches mutual information in the teacher’s space, and CrossKD blends relational and interactive cues.

Table 8. Ablation of TRACER components across ImageNet (IN) and distribution shifts. For each setting (row), accuracy (Acc.↑) is reported in %, and expected calibration error (ECE↓) in [0, 1]. OOD Avg. is the mean over {IN-V2, IN-R, IN-A, IN-S, ObjectNet}. Method names encode the presence of losses $( \mathcal { L } _ { \mathrm { F D } } , \mathcal { L } _ { \mathrm { C r o s s K D } } , \mathcal { L } _ { \mathrm { I C L } } , \mathcal { L } _ { \mathrm { C R D } } )$ as ✓or −. Rows with only one loss term active are in gray. 

<table><tr><td colspan="12">Acc.↑</td></tr><tr><td> $\mathcal{L}_{\text{FD}}$ </td><td> $\mathcal{L}_{\text{CrossKD}}$ </td><td> $\mathcal{L}_{\text{ICL}}$ </td><td> $\mathcal{L}_{\text{CRD}}$ </td><td>IN</td><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>Avg. shifts</td><td>Avg. All</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>82.69</td><td>72.73</td><td>71.35</td><td>48.52</td><td>49.84</td><td>54.86</td><td>59.40</td><td>63.33</td></tr><tr><td>-</td><td>-</td><td>-</td><td>√</td><td>83.17</td><td>74.29</td><td>77.75</td><td>53.09</td><td>53.03</td><td>57.46</td><td>63.12</td><td>66.47</td></tr><tr><td>-</td><td>-</td><td>√</td><td>-</td><td>82.50</td><td>73.13</td><td>72.12</td><td>49.23</td><td>50.03</td><td>55.26</td><td>59.95</td><td>63.71</td></tr><tr><td>-</td><td>-</td><td>√</td><td>√</td><td>83.23</td><td>74.31</td><td>76.50</td><td>52.39</td><td>52.53</td><td>57.00</td><td>62.55</td><td>65.99</td></tr><tr><td>-</td><td>√</td><td>-</td><td>-</td><td>83.19</td><td>74.04</td><td>74.67</td><td>50.65</td><td>51.39</td><td>56.40</td><td>61.43</td><td>65.06</td></tr><tr><td>-</td><td>√</td><td>-</td><td>√</td><td>83.08</td><td>74.40</td><td>78.68</td><td>53.53</td><td>53.37</td><td>57.45</td><td>63.49</td><td>66.75</td></tr><tr><td>-</td><td>√</td><td>√</td><td>-</td><td>83.03</td><td>73.95</td><td>74.11</td><td>50.60</td><td>51.28</td><td>55.77</td><td>61.14</td><td>64.79</td></tr><tr><td>-</td><td>√</td><td>√</td><td>√</td><td>83.27</td><td>74.48</td><td>77.60</td><td>52.76</td><td>52.94</td><td>57.28</td><td>63.01</td><td>66.39</td></tr><tr><td>√</td><td>-</td><td>-</td><td>-</td><td>83.06</td><td>74.16</td><td>78.14</td><td>54.39</td><td>53.14</td><td>57.79</td><td>63.52</td><td>66.78</td></tr><tr><td>√</td><td>-</td><td>-</td><td>√</td><td>82.45</td><td>73.91</td><td>79.67</td><td>54.88</td><td>53.93</td><td>58.02</td><td>64.08</td><td>67.14</td></tr><tr><td>√</td><td>-</td><td>√</td><td>-</td><td>83.08</td><td>74.39</td><td>77.59</td><td>53.84</td><td>53.10</td><td>57.59</td><td>63.30</td><td>66.60</td></tr><tr><td>√</td><td>-</td><td>√</td><td>√</td><td>82.92</td><td>74.40</td><td>79.21</td><td>54.61</td><td>53.75</td><td>57.99</td><td>63.99</td><td>67.15</td></tr><tr><td>√</td><td>√</td><td>-</td><td>-</td><td>83.01</td><td>74.21</td><td>78.91</td><td>54.09</td><td>53.28</td><td>58.04</td><td>63.71</td><td>66.92</td></tr><tr><td>√</td><td>√</td><td>-</td><td>√</td><td>82.27</td><td>73.71</td><td>79.81</td><td>54.65</td><td>53.82</td><td>58.14</td><td>64.03</td><td>67.07</td></tr><tr><td>√</td><td>√</td><td>√</td><td>-</td><td>83.06</td><td>74.38</td><td>78.38</td><td>54.07</td><td>53.25</td><td>57.86</td><td>63.59</td><td>66.83</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>82.81</td><td>73.94</td><td>79.55</td><td>54.83</td><td>53.96</td><td>58.02</td><td>64.06</td><td>67.19</td></tr></table>

<table><tr><td> $\mathcal{L}_{\text{FD}}$ </td><td> $\mathcal{L}_{\text{CrossKD}}$ </td><td> $\mathcal{L}_{\text{ICL}}$ </td><td> $\mathcal{L}_{\text{CRD}}$ </td><td>IN</td><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>Avg. shifts</td><td>Avg. All</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>0.0635</td><td>0.1171</td><td>0.0967</td><td>0.2435</td><td>0.2200</td><td>0.2383</td><td>0.1836</td><td>0.1632</td></tr><tr><td>-</td><td>-</td><td>-</td><td>√</td><td>0.0415</td><td>0.0412</td><td>0.0413</td><td>0.1328</td><td>0.0860</td><td>0.1211</td><td>0.0845</td><td>0.0773</td></tr><tr><td>-</td><td>-</td><td>√</td><td>-</td><td>0.0585</td><td>0.1000</td><td>0.0817</td><td>0.2117</td><td>0.1974</td><td>0.2168</td><td>0.1615</td><td>0.1444</td></tr><tr><td>-</td><td>-</td><td>√</td><td>√</td><td>0.0393</td><td>0.0523</td><td>0.0429</td><td>0.1534</td><td>0.1111</td><td>0.1441</td><td>0.1008</td><td>0.0905</td></tr><tr><td>-</td><td>√</td><td>-</td><td>-</td><td>0.0483</td><td>0.0830</td><td>0.0662</td><td>0.2007</td><td>0.1660</td><td>0.1918</td><td>0.1415</td><td>0.1260</td></tr><tr><td>-</td><td>√</td><td>-</td><td>√</td><td>0.0453</td><td>0.0374</td><td>0.0434</td><td>0.1141</td><td>0.0753</td><td>0.1096</td><td>0.0760</td><td>0.0709</td></tr><tr><td>-</td><td>√</td><td>√</td><td>-</td><td>0.0507</td><td>0.0824</td><td>0.0691</td><td>0.2007</td><td>0.1684</td><td>0.1968</td><td>0.1435</td><td>0.1280</td></tr><tr><td>-</td><td>√</td><td>√</td><td>√</td><td>0.0401</td><td>0.0442</td><td>0.0392</td><td>0.1345</td><td>0.0897</td><td>0.1260</td><td>0.0867</td><td>0.0790</td></tr><tr><td>√</td><td>-</td><td>-</td><td>-</td><td>0.0430</td><td>0.0674</td><td>0.0479</td><td>0.1592</td><td>0.1383</td><td>0.1661</td><td>0.1158</td><td>0.1037</td></tr><tr><td>√</td><td>-</td><td>-</td><td>√</td><td>0.0474</td><td>0.0380</td><td>0.0455</td><td>0.1034</td><td>0.0720</td><td>0.0975</td><td>0.0713</td><td>0.0673</td></tr><tr><td>√</td><td>-</td><td>√</td><td>-</td><td>0.0436</td><td>0.0684</td><td>0.0482</td><td>0.1632</td><td>0.1384</td><td>0.1691</td><td>0.1175</td><td>0.1052</td></tr><tr><td>√</td><td>-</td><td>√</td><td>√</td><td>0.0419</td><td>0.0454</td><td>0.0397</td><td>0.1157</td><td>0.0853</td><td>0.1162</td><td>0.0805</td><td>0.0740</td></tr><tr><td>√</td><td>√</td><td>-</td><td>-</td><td>0.0399</td><td>0.0537</td><td>0.0398</td><td>0.1340</td><td>0.1082</td><td>0.1374</td><td>0.0946</td><td>0.0855</td></tr><tr><td>√</td><td>√</td><td>-</td><td>√</td><td>0.0531</td><td>0.0404</td><td>0.0500</td><td>0.0936</td><td>0.0703</td><td>0.0888</td><td>0.0686</td><td>0.0660</td></tr><tr><td>√</td><td>√</td><td>√</td><td>-</td><td>0.0402</td><td>0.0572</td><td>0.0418</td><td>0.1420</td><td>0.1176</td><td>0.1499</td><td>0.1017</td><td>0.0915</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>0.0446</td><td>0.0416</td><td>0.0430</td><td>0.1027</td><td>0.0757</td><td>0.1054</td><td>0.0737</td><td>0.0688</td></tr></table>

Ablation 2: Distillation strength λSD. The ablation on distillation strength $\lambda _ { \mathrm { S D } }$ (Table 9) examines the balance between teacher influence and task adaptation. We sweep $\lambda _ { \mathrm { S D } } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 7 , 1 . 0 , 1 . 2 , 1 . 5 , 2 . 0 , 3 . 0 , 4 . 0 , 5 . 0 , 1 0 . 0 \}$ . Results indicate that moderate values of $\lambda _ { \mathrm { S D } } ~ ( \approx 1 . 0 – 2 . 0 )$ achieve the best OOD accuracy, while larger values improve calibration by lowering ECE but slightly reduce in-distribution (ID) accuracy. This aligns with our theory that stronger distillation enhances calibration through teacher anchoring, whereas moderate strength provides the optimal trade-off between adaptation and preservation for OOD performance.

Table 9. Ablation of distillation coefficient λSD across ImageNet (IN) and distribution shifts. For each setting (row), accuracy (Acc.↑) is reported in %, and expected calibration error (ECE↓) in [0, 1]. OOD Avg. is the mean over {IN-V2, IN-R, IN-A, IN-S, ObjectNet}. 

<table><tr><td colspan="8">Acc.↑ (%)</td></tr><tr><td> $\lambda_{SD}$ </td><td>IN</td><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>OOD Avg.</td></tr><tr><td>10.0</td><td>80.52</td><td>72.08</td><td>79.50</td><td>54.07</td><td>53.27</td><td>57.45</td><td>63.27</td></tr><tr><td>5.0</td><td>81.08</td><td>72.54</td><td>79.79</td><td>54.37</td><td>53.64</td><td>57.65</td><td>63.60</td></tr><tr><td>4.0</td><td>81.25</td><td>72.67</td><td>79.78</td><td>54.75</td><td>53.74</td><td>57.66</td><td>63.72</td></tr><tr><td>3.0</td><td>81.58</td><td>73.12</td><td>79.90</td><td>54.83</td><td>53.84</td><td>57.75</td><td>63.89</td></tr><tr><td>2.0</td><td>81.94</td><td>73.49</td><td>80.03</td><td>54.64</td><td>54.02</td><td>57.88</td><td>64.01</td></tr><tr><td>1.5</td><td>82.27</td><td>73.52</td><td>79.72</td><td>55.28</td><td>53.99</td><td>58.08</td><td>64.12</td></tr><tr><td>1.2</td><td>82.50</td><td>73.74</td><td>79.55</td><td>55.20</td><td>53.94</td><td>58.14</td><td>64.11</td></tr><tr><td>1.0</td><td>82.70</td><td>74.07</td><td>79.64</td><td>54.87</td><td>53.85</td><td>58.08</td><td>64.10</td></tr><tr><td>0.7</td><td>82.90</td><td>74.41</td><td>79.21</td><td>54.72</td><td>53.73</td><td>58.03</td><td>64.02</td></tr><tr><td>0.5</td><td>83.16</td><td>74.31</td><td>78.76</td><td>54.13</td><td>53.52</td><td>57.76</td><td>63.70</td></tr><tr><td>0.4</td><td>83.26</td><td>74.48</td><td>78.19</td><td>53.64</td><td>53.27</td><td>57.82</td><td>63.48</td></tr><tr><td>0.3</td><td>83.28</td><td>74.52</td><td>77.53</td><td>53.55</td><td>52.91</td><td>57.44</td><td>63.19</td></tr><tr><td>0.2</td><td>83.29</td><td>74.30</td><td>76.80</td><td>52.61</td><td>52.58</td><td>57.04</td><td>62.67</td></tr><tr><td>0.1</td><td>83.25</td><td>74.08</td><td>75.34</td><td>51.52</td><td>51.89</td><td>56.49</td><td>61.86</td></tr></table>

<table><tr><td> $\lambda_{SD}$ </td><td>IN</td><td>IN-V2</td><td>IN-R</td><td>ECE↓IN-A</td><td>IN-S</td><td>ObjectNet</td><td>OOD Avg.</td></tr><tr><td>10.0</td><td>0.0637</td><td>0.0475</td><td>0.0621</td><td>0.0839</td><td>0.0725</td><td>0.0795</td><td>0.0691</td></tr><tr><td>5.0</td><td>0.0631</td><td>0.0467</td><td>0.0600</td><td>0.0812</td><td>0.0700</td><td>0.0772</td><td>0.0670</td></tr><tr><td>4.0</td><td>0.0606</td><td>0.0457</td><td>0.0583</td><td>0.0817</td><td>0.0705</td><td>0.0801</td><td>0.0673</td></tr><tr><td>3.0</td><td>0.0590</td><td>0.0466</td><td>0.0566</td><td>0.0866</td><td>0.0701</td><td>0.0814</td><td>0.0683</td></tr><tr><td>2.0</td><td>0.0547</td><td>0.0422</td><td>0.0528</td><td>0.0885</td><td>0.0682</td><td>0.0863</td><td>0.0676</td></tr><tr><td>1.5</td><td>0.0511</td><td>0.0396</td><td>0.0490</td><td>0.0911</td><td>0.0707</td><td>0.0897</td><td>0.0680</td></tr><tr><td>1.2</td><td>0.0484</td><td>0.0408</td><td>0.0455</td><td>0.0963</td><td>0.0713</td><td>0.0957</td><td>0.0699</td></tr><tr><td>1.0</td><td>0.0465</td><td>0.0410</td><td>0.0445</td><td>0.1001</td><td>0.0742</td><td>0.1010</td><td>0.0722</td></tr><tr><td>0.7</td><td>0.0419</td><td>0.0445</td><td>0.0416</td><td>0.1120</td><td>0.0841</td><td>0.1144</td><td>0.0793</td></tr><tr><td>0.5</td><td>0.0409</td><td>0.0466</td><td>0.0390</td><td>0.1259</td><td>0.0953</td><td>0.1295</td><td>0.0873</td></tr><tr><td>0.4</td><td>0.0394</td><td>0.0502</td><td>0.0415</td><td>0.1353</td><td>0.1030</td><td>0.1363</td><td>0.0933</td></tr><tr><td>0.3</td><td>0.0397</td><td>0.0554</td><td>0.0424</td><td>0.1459</td><td>0.1161</td><td>0.1502</td><td>0.1020</td></tr><tr><td>0.2</td><td>0.0421</td><td>0.0626</td><td>0.0463</td><td>0.1628</td><td>0.1341</td><td>0.1660</td><td>0.1144</td></tr><tr><td>0.1</td><td>0.0470</td><td>0.0798</td><td>0.0601</td><td>0.1890</td><td>0.1594</td><td>0.1895</td><td>0.1356</td></tr></table>

Ablation 3: Teacher update frequency. The ablation on teacher update frequency (Table 10) investigates the trade-off between stability and plasticity in the dynamic teacher. We vary the frequency from 1 to 2500 steps (≈ 1 epoch). The results show that updating every 50–100 steps yields the highest OOD accuracy, whereas slower update schedules lead to lower OOD ECE. These findings align with the dynamic-teacher analysis: slower updates preserve early robustness and calibration, while faster updates allow the teacher to better track the evolving task solution and enhance accuracy.

Table 10. Ablation of teacher update frequency in TRACER distillation across ImageNet (IN) and distribution shifts. For each setting (row), accuracy (Acc.↑) is reported in %, and expected calibration error (ECE↓) in [0, 1]. OOD Avg. is the mean over {IN-V2, IN-R, IN-A, IN-S, ObjectNet}. The update frequency denotes the number of training steps between each teacher model update from the student; lower frequencies (e.g., 2-10 steps) result in a teacher that closely follows the student’s trajectory providing fine-grained regularization, while higher frequencies (e.g., 500-2500 steps) maintain a more stable teacher that changes less frequently, providing stronger regularization from earlier checkpoints and the initial pretrained model.

<table><tr><td rowspan="2">Update Freq.</td><td rowspan="2">IN</td><td rowspan="2">IN-V2</td><td colspan="5">Acc.↑(%)</td></tr><tr><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>OOD Avg.</td></tr><tr><td>2500</td><td>81.38</td><td>72.97</td><td>79.83</td><td>55.05</td><td>53.88</td><td>58.17</td><td>63.98</td></tr><tr><td>1000</td><td>81.90</td><td>73.27</td><td>79.71</td><td>54.93</td><td>54.21</td><td>58.26</td><td>64.08</td></tr><tr><td>500</td><td>82.13</td><td>73.53</td><td>79.80</td><td>54.72</td><td>54.23</td><td>58.30</td><td>64.12</td></tr><tr><td>100</td><td>82.54</td><td>73.92</td><td>79.76</td><td>55.09</td><td>54.00</td><td>58.31</td><td>64.22</td></tr><tr><td>50</td><td>82.62</td><td>73.84</td><td>79.69</td><td>54.96</td><td>53.96</td><td>58.12</td><td>64.11</td></tr><tr><td>10</td><td>82.57</td><td>74.01</td><td>79.58</td><td>54.96</td><td>53.88</td><td>58.00</td><td>64.09</td></tr><tr><td>5</td><td>82.70</td><td>73.98</td><td>79.57</td><td>54.81</td><td>53.93</td><td>58.07</td><td>64.07</td></tr><tr><td>2</td><td>82.73</td><td>74.12</td><td>79.57</td><td>54.51</td><td>53.99</td><td>58.09</td><td>64.06</td></tr><tr><td>1</td><td>82.76</td><td>74.14</td><td>79.33</td><td>54.92</td><td>53.69</td><td>58.26</td><td>64.07</td></tr></table>

<table><tr><td rowspan="2">Update Freq.</td><td rowspan="2">IN</td><td colspan="6">ECE↓</td></tr><tr><td>IN-V2</td><td>IN-R</td><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>OOD Avg.</td></tr><tr><td>2500</td><td>0.0614</td><td>0.0426</td><td>0.0632</td><td>0.0839</td><td>0.0722</td><td>0.0764</td><td>0.0677</td></tr><tr><td>1000</td><td>0.0562</td><td>0.0434</td><td>0.0581</td><td>0.0908</td><td>0.0698</td><td>0.0825</td><td>0.0689</td></tr><tr><td>500</td><td>0.0524</td><td>0.0419</td><td>0.0548</td><td>0.0935</td><td>0.0677</td><td>0.0868</td><td>0.0689</td></tr><tr><td>100</td><td>0.0475</td><td>0.0410</td><td>0.0487</td><td>0.0985</td><td>0.0694</td><td>0.0920</td><td>0.0699</td></tr><tr><td>50</td><td>0.0481</td><td>0.0413</td><td>0.0471</td><td>0.0992</td><td>0.0713</td><td>0.0949</td><td>0.0708</td></tr><tr><td>10</td><td>0.0473</td><td>0.0408</td><td>0.0468</td><td>0.0983</td><td>0.0714</td><td>0.0978</td><td>0.0710</td></tr><tr><td>5</td><td>0.0480</td><td>0.0400</td><td>0.0451</td><td>0.1001</td><td>0.0738</td><td>0.0975</td><td>0.0713</td></tr><tr><td>2</td><td>0.0471</td><td>0.0409</td><td>0.0440</td><td>0.1034</td><td>0.0733</td><td>0.0990</td><td>0.0721</td></tr><tr><td>1</td><td>0.0446</td><td>0.0394</td><td>0.0412</td><td>0.1041</td><td>0.0784</td><td>0.1030</td><td>0.0732</td></tr></table>

Ablation 4: Beta kernel shape. The ablation on the Beta kernel shape (Table 11) evaluates the role of endpoint-aware curricula. We vary $\beta \in \{ 0 . 2 , 0 . 5 , 0 . 7 , 0 . 9 , 1 . 0 , 1 . 5 \}$ }. Results show that smaller $\beta$ values (0.2–0.5), which emphasize endpoints, enhance both OOD accuracy and ECE by reinforcing strong early anchoring and late solution emphasis. In contrast, larger $\beta$ values favor mid-trajectory weighting, yielding marginal ID improvements at the cost of reduced OOD gains. These findings suggest that arcsine-like weighting is particularly effective for robust finetuning. Additional kernel families are discussed in §C.5.1.

Table 11. Ablation of $\beta$ value in Beta(β, β) distribution for teacher weighting in TRACER distillation across ImageNet (IN) and distribution shifts. For each setting (row), accuracy (Acc.↑) is reported in %, and expected calibration error (ECE↓) in [0, 1]. OOD Avg. is the mean over {IN-V2, IN-R, IN-A, IN-S, ObjectNet}. The $\beta$ value controls the shape of the distribution used for sampling teacher ensemble weights. Lower $\beta$ values $( < 1 )$ assign higher weights to the pretrained model and early training steps, $\beta = 1$ corresponds to uniform weighting, while higher $\beta$ values $( > 1 )$ emphasize intermediate training steps and down-weight both the initial pretrained model and final training steps.

<table><tr><td rowspan="2"> $\beta$ </td><td rowspan="2">IN</td><td rowspan="2">IN-V2</td><td rowspan="2">IN-R</td><td colspan="4">Acc. $\uparrow$ (%)</td></tr><tr><td>IN-A</td><td>IN-S</td><td>ObjectNet</td><td>OOD Avg.</td></tr><tr><td>1.5</td><td>83.19</td><td>74.38</td><td>77.15</td><td>52.43</td><td>52.95</td><td>57.06</td><td>62.79</td></tr><tr><td>1.0</td><td>83.09</td><td>74.39</td><td>78.10</td><td>52.93</td><td>53.25</td><td>57.41</td><td>63.22</td></tr><tr><td>0.9</td><td>83.14</td><td>74.26</td><td>78.28</td><td>53.59</td><td>53.35</td><td>57.41</td><td>63.38</td></tr><tr><td>0.7</td><td>82.97</td><td>74.32</td><td>78.89</td><td>54.28</td><td>53.58</td><td>57.80</td><td>63.77</td></tr><tr><td>0.5</td><td>82.76</td><td>74.14</td><td>79.33</td><td>54.92</td><td>53.69</td><td>58.26</td><td>64.07</td></tr><tr><td>0.2</td><td>81.91</td><td>73.46</td><td>79.96</td><td>55.27</td><td>54.08</td><td>58.34</td><td>64.22</td></tr></table>

<table><tr><td> $\beta$ </td><td>IN</td><td>IN-V2</td><td>IN-R</td><td>ECE↓IN-A</td><td>IN-S</td><td>ObjectNet</td><td>OOD Avg.</td></tr><tr><td>1.5</td><td>0.0403</td><td>0.0485</td><td>0.0418</td><td>0.1433</td><td>0.1069</td><td>0.1418</td><td>0.0965</td></tr><tr><td>1.0</td><td>0.0407</td><td>0.0478</td><td>0.0393</td><td>0.1317</td><td>0.0957</td><td>0.1286</td><td>0.0886</td></tr><tr><td>0.9</td><td>0.0401</td><td>0.0477</td><td>0.0402</td><td>0.1280</td><td>0.0926</td><td>0.1262</td><td>0.0869</td></tr><tr><td>0.7</td><td>0.0416</td><td>0.0456</td><td>0.0388</td><td>0.1148</td><td>0.0856</td><td>0.1161</td><td>0.0802</td></tr><tr><td>0.5</td><td>0.0446</td><td>0.0394</td><td>0.0412</td><td>0.1041</td><td>0.0784</td><td>0.1030</td><td>0.0732</td></tr><tr><td>0.2</td><td>0.0548</td><td>0.0424</td><td>0.0542</td><td>0.0917</td><td>0.0670</td><td>0.0843</td><td>0.0679</td></tr></table>

# C. Additional Theoretical Details

This section provides the full derivations, proofs, and detailed geometric interpretations for the theoretical analysis presented in the main paper.

# C.1. Derivation of $\mathcal { L } _ { \mathbf { C I } }$

We start with the linearized multimodal contrastive learning (MMCL) loss function, which balances positive and negative pairs across a batch, as commonly used in theoretical analyses (Ji et al., 2023; Tian, 2022; Nakada et al., 2023; Xue et al., 2024). The original formulation is given by:

$$
\mathcal {L} _ {\mathrm{MMCL}} (\mathbf {W} _ {I}, \mathbf {W} _ {T}) = \frac {1}{2 n (n - 1)} \sum_ {i} \sum_ {j \neq i} (s _ {i j} - s _ {i i}) + \frac {1}{2 n (n - 1)} \sum_ {i} \sum_ {j \neq i} (s _ {j i} - s _ {i i}) + \frac {\rho}{2} \| \mathbf {W} _ {I} ^ {\top} \mathbf {W} _ {T} \| _ {F} ^ {2}
$$

where $s _ { i j } = ( \mathbf { W } _ { I } \mathbf { x } _ { I } ^ { i } ) ^ { \top } ( \mathbf { W } _ { T } \mathbf { x } _ { T } ^ { j } )$ represents the similarity score between image i and text j.

Expanding the first term:

$$
{\frac {1}{2 n (n - 1)}} \sum_ {i} \sum_ {j \neq i} (s _ {i j} - s _ {i i}) = {\frac {1}{2 n (n - 1)}} \left[ \sum_ {i} \sum_ {j \neq i} s _ {i j} - \sum_ {i} \sum_ {j \neq i} s _ {i i} \right]
$$

Since for each i, there are (n − 1) values of $j \neq i ,$ , the second sub-sum simplifies:

$$
= \frac {1}{2 n (n - 1)} \left[ \sum_ {i} \sum_ {j \neq i} s _ {i j} - (n - 1) \sum_ {i} s _ {i i} \right]
$$

Expanding the second term:

$$
\frac {1}{2 n (n - 1)} \sum_ {i} \sum_ {j \neq i} (s _ {j i} - s _ {i i}) = \frac {1}{2 n (n - 1)} \left[ \sum_ {i} \sum_ {j \neq i} s _ {j i} - \sum_ {i} \sum_ {j \neq i} s _ {i i} \right]
$$

Similarly, this becomes:

$$
= \frac {1}{2 n (n - 1)} \left[ \sum_ {i} \sum_ {j \neq i} s _ {j i} - (n - 1) \sum_ {i} s _ {i i} \right]
$$

Combining both terms: Adding the first and second terms yields:

$$
{\frac {1}{2 n (n - 1)}} \left[ \sum_ {i} \sum_ {j \neq i} s _ {i j} + \sum_ {i} \sum_ {j \neq i} s _ {j i} - 2 (n - 1) \sum_ {i} s _ {i i} \right]
$$

Note that $\textstyle \sum _ { i } \sum _ { j \neq i } s _ { j i }$ is simply a re-indexing of $\textstyle \sum _ { j } \sum _ { i \neq j } s _ { i j }$ , which is equivalent to $\textstyle \sum _ { i } \sum _ { j \neq i } s _ { i j }$ . Therefore:

$$
\sum_ {i} \sum_ {j \neq i} s _ {i j} + \sum_ {i} \sum_ {j \neq i} s _ {j i} = 2 \sum_ {i} \sum_ {j \neq i} s _ {i j}
$$

Substituting back into LMMCL:

$$
\mathcal {L} _ {\mathrm{MMCL}} = \frac {1}{2 n (n - 1)} \left[ 2 \sum_ {i} \sum_ {j \neq i} s _ {i j} - 2 (n - 1) \sum_ {i} s _ {i i} \right] + \frac {\rho}{2} \| \mathbf {W} _ {I} ^ {\top} \mathbf {W} _ {T} \| _ {F} ^ {2}
$$

$$
= \frac {1}{n (n - 1)} \left[ \sum_ {i} \sum_ {j \neq i} s _ {i j} - (n - 1) \sum_ {i} s _ {i i} \right] + \frac {\rho}{2} \| \mathbf {W} _ {I} ^ {\top} \mathbf {W} _ {T} \| _ {F} ^ {2}
$$

We define the core contrastive alignment term as:

$$
\mathcal {L} _ {\mathrm{CL}} = \sum_ {i = 1} ^ {n} \sum_ {j \neq i} s _ {i j} - (n - 1) \sum_ {i = 1} ^ {n} s _ {i i}. \tag {4}
$$

And the regularization term as $\begin{array} { r } { R ( \mathbf { W } _ { I } , \mathbf { W } _ { T } ) = \frac { \rho } { 2 } \| \mathbf { W } _ { I } ^ { \top } \mathbf { W } _ { T } \| _ { F } ^ { 2 } } \end{array}$ . Thus, the total MMCL loss can be written as:

$$
\mathcal {L} _ {\mathrm{MMCL}} = \frac {1}{n (n - 1)} \mathcal {L} _ {\mathrm{CL}} + R (\mathbf {W} _ {I}, \mathbf {W} _ {T})
$$

# C.2. Reformulation of the Least-Squares Objective

We demonstrate how the contrastive alignment term ${ \mathcal { L } } _ { \mathrm { C L } }$ can be re-expressed as a matrix least-squares problem, which is the foundation of our theoretical analysis. Recall $\begin{array} { r } { \mathcal { L } _ { \mathrm { C L } } ^ { \mathrm { ~ e ~ } } = \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } s _ { i j } - ( n \stackrel { \cdot } { - } 1 ) \sum _ { i = 1 } ^ { n } s _ { i i } } \end{array}$ . Let ${ \bf H } _ { I } \doteq { \bf W } _ { I } \mathbf { \hat { X } } _ { I }$ and ${ \bf H } _ { T } = { \bf W } _ { T } ^ { 0 } { \bf X } _ { T }$ . Then $s _ { i j } = ( { \bf H } _ { I } ) _ { i } ^ { \top } ( { \bf H } _ { T } ) _ { j }$ . Let $\mathbf { S } = \mathbf { H } _ { I } ^ { \top } \mathbf { H } _ { T }$ . The sum of all similarities is $\begin{array} { r } { \mathbf { 1 } ^ { \top } \mathbf { S 1 } = \sum _ { i , j } s _ { i j } } \end{array}$ . The sum of diagonal similarities is $\textstyle \mathrm { T r } ( \mathbf { S } ) = \sum _ { i } s _ { i i }$ . Then, $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \sum _ { j \neq i } s _ { i j } = \sum _ { i , j } s _ { i j } - \sum _ { i } s _ { i i } = \mathbf { 1 } ^ { \top } \mathbf { S } \mathbf { 1 } - \operatorname { T r } ( \mathbf { S } ) } \end{array}$ . Substituting this into ${ \mathcal { L } } _ { \mathrm { C L } } ;$ :

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{CL}} = \left(\mathbf {1} ^ {\top} \mathbf {S 1} - \operatorname{Tr} (\mathbf {S})\right) - (n - 1) \operatorname{Tr} (\mathbf {S}) \\ = \mathbf {1} ^ {\top} \mathbf {S 1} - n \operatorname{Tr} (\mathbf {S}) \\ = \operatorname{Tr} (\mathbf {1 1} ^ {\top} \mathbf {S}) - n \operatorname{Tr} (\mathbf {S}) \\ = \mathrm{Tr} ((\mathbf {J} _ {n} - n \mathbf {I} _ {n}) ^ {\top} \mathbf {S}) \\ = \operatorname{Tr} \left(\left(\mathbf {J} _ {n} - n \mathbf {I} _ {n}\right) ^ {\top} \mathbf {H} _ {I} ^ {\top} \mathbf {H} _ {T}\right) \\ = \operatorname{Tr} \left(\mathbf {H} _ {T} \left(\mathbf {J} _ {n} - n \mathbf {I} _ {n}\right) \mathbf {H} _ {I} ^ {\top}\right) \\ = \operatorname{Tr} (\mathbf {W} _ {T} ^ {0} \mathbf {X} _ {T} (\mathbf {J} _ {n} - n \mathbf {I} _ {n}) (\mathbf {W} _ {I} \mathbf {X} _ {I}) ^ {\top}) \\ = - \operatorname{Tr} \left(\left(\mathbf {W} _ {T} ^ {0} \mathbf {X} _ {T} (n \mathbf {I} _ {n} - \mathbf {J} _ {n})\right) ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I}\right). \\ \end{array}
$$

Let ${ \mathbf Y } _ { \mathrm { F T } } = { \mathbf W } _ { T } ^ { 0 } { \mathbf X } _ { T } ( n { \mathbf I } _ { n } - { \mathbf J } _ { n } )$ , as defined in Definition 3.1. Then

$$
\mathcal {L} _ {\mathrm{CL}} = - \operatorname{Tr} \left(\mathbf {Y} _ {\mathrm{FT}} ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I}\right).
$$

Expanding the Frobenius norm of the least-squares residual gives:

$$
\begin{array}{l} \frac {1}{2} \left\| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}} \right\| _ {\mathrm{F}} ^ {2} = \frac {1}{2} \operatorname{Tr} \left(\left(\mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}}\right) ^ {\top} \left(\mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}}\right)\right) \\ = \frac {1}{2} \operatorname{Tr} \left(\mathbf {X} _ {I} ^ {\top} \mathbf {W} _ {I} ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {X} _ {I} ^ {\top} \mathbf {W} _ {I} ^ {\top} \mathbf {Y} _ {\mathrm{FT}} \right. \\ \left. - \mathbf {Y} _ {\mathrm{FT}} ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I} + \mathbf {Y} _ {\mathrm{FT}} ^ {\top} \mathbf {Y} _ {\mathrm{FT}}\right) \\ = \underbrace {\frac {1}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} \| _ {\mathrm{F}} ^ {2}} _ {(\star)} \underbrace {- \operatorname{Tr} (\mathbf {Y} _ {\mathrm{FT}} ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I})} _ {= \mathcal {L} _ {\mathrm{CL}}} + \underbrace {\frac {1}{2} \| \mathbf {Y} _ {\mathrm{FT}} \| _ {\mathrm{F}} ^ {2}} _ {\text {constant in} \mathbf {W} _ {I}}. \\ \end{array}
$$

A clarification on the trace–least-squares relationship. As the expansion above shows, $\begin{array} { r } { { \frac { 1 } { 2 } } \left\| \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { Y } _ { \mathrm { F T } } \right\| _ { \mathrm { F } } ^ { 2 } } \end{array}$ is not equal to $- \mathrm { T r } ( \mathbf { Y } _ { \mathrm { F T } } ^ { \top } \mathbf { W } _ { I } \mathbf { X } _ { I } )$ up to a constant alone; the two differ by the data-dependent quadratic term $\begin{array} { r } { \mathbf { \eta } ( \star ) = \frac { 1 } { 2 } \| \mathbf { W } _ { I } \mathbf { X } _ { I } \| _ { \mathrm { F } } ^ { 2 } } \end{array}$ . Hence minimizing the pure trace functional $- \mathrm { T r } ( \mathbf { Y } _ { \mathrm { F T } } ^ { \top } \mathbf { W } _ { I } \mathbf { X } _ { I } )$ in isolation is unbounded below and is not equivalent to minimizing the least-squares objective.

What we use throughout the rest of the analysis is the least-squares reformulation

$$
\min _ {\mathbf {W} _ {I}} \frac {1}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}} \| _ {\mathrm{F}} ^ {2} = \min _ {\mathbf {W} _ {I}} \frac {1}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} \| _ {\mathrm{F}} ^ {2} - \operatorname{Tr} \left(\mathbf {Y} _ {\mathrm{FT}} ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I}\right) + \text { const }, \tag {5}
$$

which differs from the original linearized MMCL term by the additive data-dependent quadratic $( { \star } )$ . This extra term can be viewed as the natural data-dependent quadratic regularization that arises whenever a bilinear similarity is matched against a fixed target $\mathbf { Y } _ { \mathrm { F I } }$ under a Frobenius surrogate, and it is precisely what makes the resulting problem a well-posed matrix least-squares program with closed-form solutions. All subsequent closed-form solutions (Theorem C.2) and dynamic-teacher analyses (§C.5) are derived from this least-squares objective; the original $- \operatorname { T r } ( \cdot )$ form is recovered by dropping (⋆), but our results are unaffected because they follow from the least-squares program.

# C.3. Unified Framework for Contrastive Finetuning: Proofs and Details

Our proofs rely on the following lemma for gradient descent on a matrix quadratic program.

Lemma C.1 (Gradient Descent for Matrix Quadratic Programs). Let $\mathcal { Q } : \mathbb { R } ^ { p \times d }  \mathbb { R } ^ { p \times d }$ be a positive semi-definite (PSD) linear operator and $\mathbf { P } \in \mathbb { R } ^ { p \times d }$ . Consider the quadratic objective

$$
f (\mathbf {W}) = \frac {1}{2} \langle \mathbf {W}, \mathcal {Q} (\mathbf {W}) \rangle_ {F} - \langle \mathbf {P}, \mathbf {W} \rangle_ {F}, \tag {6}
$$

where $\langle \cdot , \cdot \rangle _ { F }$ denotes the Frobenius inner product. Let $\| \mathcal { Q } \| _ { \mathrm { o p } }$ denote the operator norm of $\mathcal { Q }$ induced by the Frobenius norm. If $\mathbf { P } \in \mathrm { R a n g e } ( \mathcal { Q } )$ , then gradient descent initialized at $\mathbf { W } _ { 0 }$ with step size $\gamma \in ( 0 , \bar { , } 2 / \Vert \mathscr { Q } \Vert _ { \mathrm { o p } } )$ converges to

$$
\mathbf {W} _ {\infty} = (\mathbf {I} - \Pi_ {\mathcal {Q}}) (\mathbf {W} _ {0}) + \mathcal {Q} ^ {+} (\mathbf {P}), \tag {7}
$$

where $\Pi _ { \mathfrak { Q } }$ is the orthogonal projector onto Range(Q) and $\mathcal { Q } ^ { + }$ is the Moore-Penrose pseudoinverse of $\mathcal { Q } .$ .

Proof. The gradient of f is given by $\nabla f ( \mathbf { W } ) = \mathcal { Q } ( \mathbf { W } ) - \mathbf { P }$ , yielding the gradient descent update

$$
\mathbf {W} _ {t + 1} = \mathbf {W} _ {t} - \gamma (\mathcal {Q} (\mathbf {W} _ {t}) - \mathbf {P}). \tag {8}
$$

Since Q is PSD, we have the orthogonal decomposition

$$
\mathbb {R} ^ {p \times d} = \operatorname{Range} (\mathcal {Q}) \oplus \operatorname{Null} (\mathcal {Q}). \tag {9}
$$

Let $\Pi _ { \mathfrak { Q } }$ and $\boldsymbol { \Pi } _ { \mathcal { Q } ^ { \perp } } = \mathbf { I } - \boldsymbol { \Pi } _ { \mathcal { Q } }$ denote the orthogonal projectors onto the range and null space of $\mathcal { Q } ,$ respectively.

Analysis of the null space component. Projecting the gradient descent update onto $\operatorname { N u l l } ( \mathcal { Q } )$ yields

$$
\begin{array}{l} \Pi_ {\mathcal {Q} ^ {\perp}} \left(\mathbf {W} _ {t + 1}\right) = \Pi_ {\mathcal {Q} ^ {\perp}} \left(\mathbf {W} _ {t}\right) - \gamma \Pi_ {\mathcal {Q} ^ {\perp}} \left(\mathcal {Q} \left(\mathbf {W} _ {t}\right)\right) + \gamma \Pi_ {\mathcal {Q} ^ {\perp}} (\mathbf {P}) \\ = \Pi_ {\mathcal {Q} ^ {\perp}} (\mathbf {W} _ {t}), \\ \end{array}
$$

where we used that ${ \mathcal { Q } } ( \mathbf { W } _ { t } ) \in \mathrm { R a n g e } ( { \mathcal { Q } } )$ implies $\Pi _ { \mathcal { Q } ^ { \perp } } ( \mathcal { Q } ( \mathbf { W } _ { t } ) ) = \mathbf { 0 }$ , and our assumption $\mathbf { P } \in \mathrm { R a n g e } ( \mathcal { Q } )$ implies $\Pi _ { \mathcal { Q } ^ { \perp } } ( \mathbf { P } ) = \mathbf { 0 }$ Thus, the null space component remains invariant throughout the optimization:

$$
\Pi_ {\mathcal {Q} ^ {\perp}} (\mathbf {W} _ {t}) = \Pi_ {\mathcal {Q} ^ {\perp}} (\mathbf {W} _ {0}) \quad \forall t \geq 0. \tag {10}
$$

Analysis of the range component. Let $\mathbf { W } _ { t } ^ { \prime } = \Pi _ { \mathcal { Q } } ( \mathbf { W } _ { t } )$ denote the projection onto Range(Q). The dynamics of this component follow

$$
\mathbf {W} _ {t + 1} ^ {\prime} = (\mathbf {I} - \gamma \mathcal {Q}) \mathbf {W} _ {t} ^ {\prime} + \gamma \mathbf {P}. \tag {11}
$$

The restriction of $\mathcal { Q }$ to its range, denoted $\mathcal { Q } _ { R } : \mathrm { R a n g e } ( \mathcal { Q } )  \mathrm { R a n g e } ( \mathcal { Q } )$ , is positive definite (since for any non-zero $x \in { \mathrm { R a n g e } } ( \mathcal { Q } )$ , we must have $\mathcal { Q } ( x ) \neq 0 .$ , otherwise x would be in $\operatorname { N u l l } ( \mathcal { Q } ) )$ ). For $\gamma \in ( 0 , 2 \bar { / } \| \mathscr { Q } \| _ { \mathrm { o p } } )$ , the operator $\mathbf { I } - \gamma \mathcal { Q } _ { R }$ has spectral radius less than 1, making it a contraction mapping. By the Banach fixed-point theorem, the sequence $\left\{ \mathbf { \hat { W } } _ { t } ^ { \prime } \right\}$ converges to the unique fixed point $\mathbf { W } _ { \infty } ^ { \prime } \in \mathrm { R a n g e } ( \mathcal { Q } )$ satisfying

$$
\mathbf {W} _ {\infty} ^ {\prime} = (\mathbf {I} - \gamma \mathcal {Q}) \mathbf {W} _ {\infty} ^ {\prime} + \gamma \mathbf {P}. \tag {12}
$$

Rearranging gives $\mathcal { Q } ( \mathbf { W } _ { \infty } ^ { \prime } ) = \mathbf { P }$ , which has the unique solution ${ \bf W } _ { \infty } ^ { \prime } = { \sf Q } ^ { + } ( { \bf P } )$ in ${ \mathrm { R a n g e } } ( { \mathcal { Q } } )$

Synthesis. Combining the analyses of both components, we obtain

$$
\begin{array}{l} \mathbf {W} _ {\infty} = \lim _ {t \to \infty} \left(\Pi_ {\mathcal {Q} ^ {\perp}} (\mathbf {W} _ {t}) + \mathbf {W} _ {t} ^ {\prime}\right) \\ = \Pi_ {\mathcal {Q} ^ {\perp}} (\mathbf {W} _ {0}) + \mathcal {Q} ^ {+} (\mathbf {P}) \\ = (\mathbf {I} - \Pi_ {\mathcal {Q}}) (\mathbf {W} _ {0}) + \mathcal {Q} ^ {+} (\mathbf {P}), \\ \end{array}
$$

completing the proof.

Theorem C.2 (Unified Framework for Contrastive Finetuning Solutions (Full Proof)). Let $\mathcal { P } _ { I } : = \mathbf { X } _ { I } ( \mathbf { X } _ { I } ^ { \top } \mathbf { X } _ { I } ) ^ { + } \mathbf { X } _ { I } ^ { \top }$ denote the orthogonal projection onto the subspace spanned by the finetuning data $\mathbf { X } _ { I }$ . Consider the general finetuning objective:

$$
\mathcal {L} (\mathbf {W} _ {I}) = \frac {1}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}} \| _ {\mathrm{F}} ^ {2} + \mathcal {R} (\mathbf {W} _ {I}) \tag {13}
$$

where $\mathcal { R } ( \mathbf { W } _ { I } )$ represents different regularization strategies. Gradient descent initialized at $\mathbf { W } _ { I } ^ { 0 }$ with sufficiently small learning rate converges to the following solutions:

<table><tr><td>Strategy</td><td> $\mathcal{R}(\mathbf{W}_I)$ </td><td>Solution</td></tr><tr><td>Direct Finetuning</td><td>0</td><td> $\mathbf{W}_{\text{FT}} = \mathbf{W}_I^0 (\mathbf{I} - \mathcal{P}_I) + \mathbf{Y}_{\text{FT}} \mathbf{X}_I^\top (\mathbf{X}_I \mathbf{X}_I^\top)^+$ </td></tr><tr><td> $L_2$  Regularization(L2-SP (Li et al., 2018))</td><td> $\frac{\lambda}{2} \left\| \mathbf{W}_I - \mathbf{W}_I^0 \right\|_{\text{F}}^2$ </td><td> $\mathbf{W}_{L_2} = (\mathbf{Y}_{\text{FT}} \mathbf{X}_I^\top + \lambda \mathbf{W}_I^0)(\mathbf{X}_I \mathbf{X}_I^\top + \lambda \mathbf{I})^{-1}$ </td></tr><tr><td>Self-Distillation(SD (Furlanello et al., 2018))</td><td> $\frac{\lambda}{2} \left\| \mathbf{W}_I \mathbf{X}_I - \mathbf{W}_I^0 \mathbf{X}_I \right\|_{\text{F}}^2$ </td><td> $\mathbf{W}_{SD} = \mathbf{W}_I^0 (\mathbf{I} - \frac{1}{1+\lambda} \mathcal{P}_I) + \frac{1}{1+\lambda} \mathbf{Y}_{\text{FT}} \mathbf{X}_I^\top (\mathbf{X}_I \mathbf{X}_I^\top)^+$ </td></tr></table>

Here, + denotes the Moore-Penrose pseudoinverse and $\lambda > 0$ is the regularization parameter.

Proof. Let $\mathbf { C } _ { I } = \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top }$ .

Direct Finetuning. The objective is $\begin{array} { r } { \mathcal { L } ( \mathbf { W } _ { I } ) = \frac { 1 } { 2 } \left. \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { Y } _ { \mathrm { F T } } \right. _ { \mathrm { F } } ^ { 2 } } \end{array}$ . We rewrite this in the quadratic form of Lemma C.1:

$$
\begin{array}{l} \mathcal {L} (\mathbf {W} _ {I}) = \frac {1}{2} \left\langle \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}}, \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}} \right\rangle_ {F} \\ = \frac {1}{2} \langle \mathbf {W} _ {I}, \mathbf {W} _ {I} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) \rangle_ {F} - \langle \mathbf {W} _ {I}, \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} \rangle_ {F} + \frac {1}{2} \| \mathbf {Y} _ {\mathrm{FT}} \| _ {\mathrm{F}} ^ {2} \\ = \frac {1}{2} \left\langle \mathbf {W} _ {I}, \mathbf {W} _ {I} \mathbf {C} _ {I} \right\rangle_ {F} - \left\langle \mathbf {W} _ {I}, \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} \right\rangle_ {F} + \text { const. } \\ \end{array}
$$

This matches the form $\begin{array} { r } { f ( \mathbf { W } ) = \frac { 1 } { 2 } \langle \mathbf { W } , \mathcal { Q } ( \mathbf { W } ) \rangle _ { F } - \langle \mathbf { P } , \mathbf { W } \rangle _ { F } \mathrm { ~ w i t h ~ } \mathcal { Q } ( \mathbf { W } ) = \mathbf { W } \mathbf { C } _ { I } \mathrm { ~ a n d ~ } \mathbf { P } = \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top } . } \end{array}$ .

The operator Q is linear and positive semi-definite, as $\langle \mathbf { W } _ { I } , \boldsymbol { \mathcal { Q } } ( \mathbf { W } _ { I } ) \rangle _ { F } = \| \mathbf { W } _ { I } \mathbf { X } _ { I } \| _ { \mathrm { F } } ^ { 2 } \geq 0$ . The condition $\mathbf { P } \in \mathrm { R a n g e } ( \mathcal { Q } )$ holds because the rows of $\mathbf { P } = \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top }$ are linear combinations of the rows of $\mathbf { X } _ { I } ^ { \top }$ , which form the row space of $\mathbf { C } _ { I }$ .

By Lemma C.1, gradient descent converges to $\mathbf { W } _ { \infty } = \Pi _ { \mathcal { O } ^ { \perp } } ( \mathbf { W } _ { I } ^ { 0 } ) + \mathcal { Q } ^ { + } ( \mathbf { P } )$ .

1. Null Space Component: The null space of Q consists of matrices A such that ${ \mathcal { Q } } ( \mathbf { A } ) = \mathbf { A } \mathbf { C } _ { I } = \mathbf { 0 }$ . This holds if and only if the rows of A are in the null space of $\mathbf { C } _ { I }$ . The orthogonal projector onto this component of the initial matrix $\mathbf { W } _ { I } ^ { 0 }$ is $\Pi _ { \dot { \mathcal { Q } } ^ { \perp } } ^ { \phantom { 0 } } ( \mathbf { W } _ { I } ^ { 0 } ) = \mathbf { W } _ { I } ^ { 0 } ( \mathbf { I } - \mathcal { P } _ { I } )$ , where $\bar { \mathcal { P } } _ { I } \overset { - } { = } \mathbf { C } _ { I } \mathbf { C } _ { I } ^ { + }$ is the projector onto the row space of $\mathbf { X } _ { I }$ . This component is preserved.   
2. Range Component: The pseudoinverse $\mathcal { Q } ^ { + }$ finds the minimum Frobenius norm solution to $\mathcal { Q } ( \mathbf { W } ) = \mathbf { P }$ that lies in Range(Q). This is the solution to $\mathbf { W } \mathbf { C } _ { I } = \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top }$ , which is $\begin{array} { r } { \mathcal { Q } ^ { + } ( \mathbf { P } ) = ( \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top } ) \mathbf { C } _ { I } ^ { + } } \end{array}$ .

Combining the components gives the final solution:

$$
\mathbf {W} _ {\mathrm{FT}} = \mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I}) + \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+}.
$$

$L _ { 2 }$ Regularization. The objective $\begin{array} { r } { \mathcal { L } ( \mathbf { W } _ { I } ) = \frac { 1 } { 2 } \left\| \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { Y } _ { \mathrm { F T } } \right\| _ { \mathrm { F } } ^ { 2 } + \frac { \lambda } { 2 } \left\| \mathbf { W } _ { I } - \mathbf { W } _ { I } ^ { 0 } \right\| _ { \mathrm { F } } ^ { 2 } } \end{array}$ . This objective is strongly convex for $\lambda > 0$ . The unique minimizer is found by setting the gradient to zero:

$$
\begin{array}{l} \nabla_ {\mathbf {W} _ {I}} \mathcal {L} = (\mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}}) \mathbf {X} _ {I} ^ {\top} + \lambda (\mathbf {W} _ {I} - \mathbf {W} _ {I} ^ {0}) = 0 \\ \rightarrow \mathbf {W} _ {I} \mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} = \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0} \\ \rightarrow \mathbf {W} _ {I} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {I}) = \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0} \\ \end{array}
$$

Since $\mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top }$ is PSD, the matrix $( \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top } + \lambda \mathbf { I } )$ is positive definite and thus invertible. The solution is:

$$
\mathbf {W} _ {L _ {2}} = \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0}\right) \left(\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {I}\right) ^ {- 1}.
$$

A more detailed analysis of the limit behavior of this solution as λ → 0 and λ → ∞ is provided in §C.4.

Self-Distillation. The objective is $\begin{array} { r } { \mathcal { L } ( \mathbf { W } _ { I } ) = \frac { 1 } { 2 } \left\| \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { Y } _ { \mathrm { F T } } \right\| _ { \mathrm { F } } ^ { 2 } + \frac { \lambda } { 2 } \left\| \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { W } _ { I } ^ { 0 } \mathbf { X } _ { I } \right\| _ { \mathrm { F } } ^ { 2 } } \end{array}$ . Expanding and grouping terms reveals the quadratic structure:

$$
\begin{array}{l} \mathcal {L} (\mathbf {W} _ {I}) = \frac {1}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} \| _ {\mathrm{F}} ^ {2} - \operatorname{Tr} (\mathbf {Y} _ {\mathrm{FT}} ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I}) + \frac {1}{2} \| \mathbf {Y} _ {\mathrm{FT}} \| _ {\mathrm{F}} ^ {2} \\ + \frac {\lambda}{2} \left\| \mathbf {W} _ {I} \mathbf {X} _ {I} \right\| _ {\mathrm{F}} ^ {2} - \lambda \operatorname{Tr} ((\mathbf {W} _ {I} ^ {0} \mathbf {X} _ {I}) ^ {\top} \mathbf {W} _ {I} \mathbf {X} _ {I}) + \frac {\lambda}{2} \left\| \mathbf {W} _ {I} ^ {0} \mathbf {X} _ {I} \right\| _ {\mathrm{F}} ^ {2} \\ = \frac {1 + \lambda}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} \| _ {\mathrm{F}} ^ {2} - \operatorname{Tr} \left(\left(\mathbf {Y} _ {\mathrm{FT}} ^ {\top} + \lambda \left(\mathbf {W} _ {I} ^ {0} \mathbf {X} _ {I}\right) ^ {\top}\right) \mathbf {W} _ {I} \mathbf {X} _ {I}\right) + \text { const. } \\ = \frac {1 + \lambda}{2} \left\langle \mathbf {W} _ {I}, \mathbf {W} _ {I} \mathbf {C} _ {I} \right\rangle_ {F} - \left\langle \mathbf {W} _ {I}, \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0} \mathbf {C} _ {I} \right\rangle_ {F} + \text { const. } \\ \end{array}
$$

This matches the form of Lemma C.1 with $\mathcal { Q } _ { S D } ( \mathbf { W } _ { I } ) = ( 1 + \lambda ) \mathbf { W } _ { I } \mathbf { C } _ { I } \mathrm { ~ a n d ~ } \mathbf { P } _ { S D } = \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top } + \lambda \mathbf { W } _ { I } ^ { 0 } \mathbf { C } _ { I } .$

The operator ${ \mathcal { Q } } _ { S D }$ is PSD. Its range and null space are identical to those of Q from the Direct Finetuning case. The terms $\mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top }$ an d $\lambda \mathbf { W } _ { I } ^ { 0 } \mathbf { \bar { C } } _ { I }$ are both in $\mathrm { R a n g e } ( \mathcal { Q } _ { S D } )$ (as shown before for $\mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top }$ , and $\mathbf { W } _ { I } ^ { 0 } \mathbf { C } _ { I }$ by definition). Thus, their sum $\mathbf { P } _ { S D }$ is also in the range.

We apply Lemma C.1 to find the limit $\mathbf { W } _ { \infty } = \Pi _ { \mathcal { Q } _ { S D } ^ { \perp } } ( \mathbf { W } _ { I } ^ { 0 } ) + \mathcal { Q } _ { S D } ^ { + } ( \mathbf { P } _ { S D } )$ .

1. Null Space Component: $\mathbf { N u l l } ( \mathcal { Q } _ { S D } ) = \mathbf { N u l l } ( \mathcal { Q } )$ , so the invariant component is again $\Pi _ { \mathcal { Q } _ { S D } ^ { \perp } } ( \mathbf { W } _ { I } ^ { 0 } ) = \mathbf { W } _ { I } ^ { 0 } ( \mathbf { I } - \mathcal { P } _ { I } )$

2. Range Component: The pseudoinverse is $\begin{array} { r } { \mathcal { Q } _ { S D } ^ { + } = \frac { 1 } { 1 + \lambda } \mathcal { Q } ^ { + } } \end{array}$ , where $\mathcal { Q } ^ { + }$ corresponds to the direct finetuning case. Applying it to ${ \bf P } _ { S D } \colon$

$$
\begin{array}{l} \mathcal {Q} _ {S D} ^ {+} (\mathbf {P} _ {S D}) = \frac {1}{1 + \lambda} \mathcal {Q} ^ {+} \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0} \mathbf {C} _ {I}\right) \\ = \frac {1}{1 + \lambda} \left((\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top}) \mathbf {C} _ {I} ^ {+} + \lambda \mathcal {Q} ^ {+} (\mathcal {Q} (\mathbf {W} _ {I} ^ {0}))\right) \\ = \frac {1}{1 + \lambda} \left((\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top}) \mathbf {C} _ {I} ^ {+} + \lambda \Pi_ {\mathcal {Q}} (\mathbf {W} _ {I} ^ {0})\right) \\ = \frac {1}{1 + \lambda} \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+} + \lambda \mathbf {W} _ {I} ^ {0} \mathcal {P} _ {I}\right). \\ \end{array}
$$

Combining the components for the final solution $\mathbf { W } _ { S D } \mathbf { : }$ :

$$
\begin{array}{l} \mathbf {W} _ {S D} = \mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I}) + \frac {\lambda}{1 + \lambda} \mathbf {W} _ {I} ^ {0} \mathcal {P} _ {I} + \frac {1}{1 + \lambda} \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+} \\ = \mathbf {W} _ {I} ^ {0} \left(\mathbf {I} - \mathcal {P} _ {I} + \frac {\lambda}{1 + \lambda} \mathcal {P} _ {I}\right) + \frac {1}{1 + \lambda} \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+} \\ = \mathbf {W} _ {I} ^ {0} \left(\mathbf {I} - \frac {1}{1 + \lambda} \mathcal {P} _ {I}\right) + \frac {1}{1 + \lambda} \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+}. \\ \end{array}
$$

This completes the proof.

![](images/8697a26cb8d0712a675d9ef2861d7c76ce337543fb327fc93330f216cbd3576b.jpg)

# C.4. Geometric Interpretation of Solutions

The closed-form solutions presented in Theorem C.2 provide a geometric understanding of how different finetuning strategies modify pretrained representations. We decompose the solution for $\mathbf { W } _ { I }$ into components acting on the subspace spanned by the finetuning data $\mathbf { X } _ { I }$ (parallel component) and its orthogonal complement (orthogonal component).

Direct Finetuning. The solution $\mathbf { W } _ { \mathrm { F T } }$ is a sum of two orthogonal parts: (1) $\mathbf { W } _ { I } ^ { 0 } ( \mathbf { I } - \mathcal { P } _ { I } )$ : This is the projection of the pretrained weights onto the orthogonal complement of the finetuning data subspace $( \mathrm { N u l l } ( \mathbf X _ { I } ^ { \top } ) )$ . This component preserves the action of $\mathbf { W } _ { I } ^ { 0 }$ on data vectors orthogonal to the finetuning examples. (2 $) { \bf Y } _ { \mathrm { F T } } { \bf X } _ { I } ^ { \top } ( { \bf \bar { X } } _ { I } { \bf X } _ { I } ^ { \top } ) ^ { + }$ : This is the minimum-norm solution that fits the new contrastive task within the finetuning data subspace. This component lies entirely within the range of $\mathbf { X } _ { I } ^ { \top }$ .

Interpretation: Direct finetuning completely replaces (forgets) any pretrained knowledge related to features present in the finetuning data, substituting it with the new task-specific solution. It only preserves knowledge in directions entirely unrelated to the finetuning examples.

$L _ { 2 }$ Regularization. The solution $\mathbf { W } _ { L _ { 2 } }$ is the standard matrix ridge regression solution. It creates a complex blend of the new task solution and the initial weights. There is no clean separation of orthogonal and parallel components as in direct finetuning or self-distillation. The key insight is that $L _ { 2 }$ regularization modifies the data covariance matrix $\mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top }$ by adding λI, which acts as a ridge that prevents overfitting by shrinking the solution along all eigendirections of the data. Unlike direct finetuning and self-distillation, which primarily modify weights in the subspace spanned by $\mathbf { X } _ { I } , L _ { 2 }$ regularization affects all directions in the weight space, blending the old and new across the entire parameter space.

Detailed Analysis of the $L _ { 2 }$ Regularization Solution. The solution for $L _ { 2 }$ regularization is given by:

$$
\mathbf {W} _ {L _ {2}} = \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0}\right) \left(\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {I}\right) ^ {- 1}.
$$

To analyze its behavior, we consider the eigendecomposition of the data covariance matrix $\mathbf { C } _ { I } : = \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top }$ . Since $\mathbf { C } _ { I }$ is a real, symmetric, positive semi-definite (PSD) matrix, it has an eigendecomposition $\mathbf { C } _ { I } = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top }$ , where U is an orthogonal matrix of eigenvectors and Λ is a diagonal matrix of non-negative eigenvalues. Using this decomposition, the inverse term in the solution becomes:

$$
\left(\mathbf {C} _ {I} + \lambda \mathbf {I}\right) ^ {- 1} = \left(\mathbf {U} \boldsymbol {\Lambda} \mathbf {U} ^ {\top} + \lambda \mathbf {U} \mathbf {U} ^ {\top}\right) ^ {- 1} = \left(\mathbf {U} (\boldsymbol {\Lambda} + \lambda \mathbf {I}) \mathbf {U} ^ {\top}\right) ^ {- 1} = \mathbf {U} (\boldsymbol {\Lambda} + \lambda \mathbf {I}) ^ {- 1} \mathbf {U} ^ {\top}.
$$

The matrix $( \pmb { \Lambda } + \lambda \pmb { \ I } )$ is diagonal with entries $\lambda _ { k } + \lambda ,$ , so its inverse has entries $1 / ( \lambda _ { k } + \lambda )$ .

Analysis of the Limit as $\lambda  0 .$ . Let $r = \mathrm { r a n k } ( \mathbf { C } _ { I } )$ . We partition the eigenvectors U and eigenvalues Λ into components corresponding to non-zero and zero eigenvalues. Let $\mathbf { U } _ { r } \in \mathbb { R } ^ { d _ { I } \times r }$ contain eigenvectors for r positive eigenvalues $( \pmb { \Lambda } _ { r } )$ , and ${ \bf U } _ { 0 } \in  { \bf \Psi }$ $\mathbb { R } ^ { d _ { I } \times ( \hat { d } _ { I } - r ) }$ for zero eigenvalues. The projectors onto the range and null space of $\mathbf { C } _ { I }$ are $\mathcal { P } _ { \mathrm { r a n g e } } = \mathbf { U } _ { r } \mathbf { U } _ { r } ^ { \top }$ and $\mathcal { P } _ { \mathrm { n u l l } } = \mathbf { U } _ { 0 } \mathbf { U } _ { 0 } ^ { \top }$ , respectively. Note that $\mathcal { P } _ { \mathrm { r a n g e } } = \mathcal { P } _ { I }$ and $\bar { \mathcal { P } } _ { \mathrm { n u l l } } = \mathbf { I } - \mathcal { P } _ { I }$ .

The inverse term can be split:

$$
\left(\mathbf {C} _ {I} + \lambda \mathbf {I}\right) ^ {- 1} = \mathbf {U} _ {r} \left(\boldsymbol {\Lambda} _ {r} + \lambda \mathbf {I} _ {r}\right) ^ {- 1} \mathbf {U} _ {r} ^ {\top} + \frac {1}{\lambda} \mathbf {U} _ {0} \mathbf {U} _ {0} ^ {\top}.
$$

Substituting this back into $\mathbf { W } _ { L _ { 2 } } \mathbf { : }$

$$
\begin{array}{l} \mathbf {W} _ {L _ {2}} = \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0}\right) \left[ \mathbf {U} _ {r} \left(\boldsymbol {\Lambda} _ {r} + \lambda \mathbf {I} _ {r}\right) ^ {- 1} \mathbf {U} _ {r} ^ {\top} + \frac {1}{\lambda} \mathbf {U} _ {0} \mathbf {U} _ {0} ^ {\top} \right] \\ = \underbrace {\left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0}\right) \mathbf {U} _ {r} (\boldsymbol {\Lambda} _ {r} + \lambda \mathbf {I} _ {r}) ^ {- 1} \mathbf {U} _ {r} ^ {\top}} _ {\text {Term 1}} \\ + \underbrace {\left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0}\right) \frac {1}{\lambda} \mathbf {U} _ {0} \mathbf {U} _ {0} ^ {\top}} _ {\text { Term   2 }}. \\ \end{array}
$$

For Term 2, since $\mathbf { X } _ { I } ^ { \top } \mathbf { U } _ { 0 } = \mathbf { 0 }$ (columns of $\mathbf { U } _ { 0 }$ are in the null space of $\mathbf { C } _ { I } ) .$ , it simplifies to:

$$
\text { Term   } 2 = \frac {1}{\lambda} \mathbf {Y} _ {\text { FT }} \underbrace {\mathbf {X} _ {I} ^ {\top} \mathbf {U} _ {0}} _ {\mathbf {0}} \mathbf {U} _ {0} ^ {\top} + \mathbf {W} _ {I} ^ {0} \mathbf {U} _ {0} \mathbf {U} _ {0} ^ {\top} = \mathbf {W} _ {I} ^ {0} \mathcal {P} _ {\text { null }} = \mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I}).
$$

As $\lambda  0 .$ Term 1 converges to:

$$
\lim _ {\lambda \rightarrow 0} \text { Term }   1 = \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top}\right) \mathbf {U} _ {r} \boldsymbol {\Lambda} _ {r} ^ {- 1} \mathbf {U} _ {r} ^ {\top} = \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} \mathbf {C} _ {I} ^ {+},
$$

where $\mathbf { C } _ { I } ^ { + } = \mathbf { U } _ { r } \mathbf { A } _ { r } ^ { - 1 } \mathbf { U } _ { r } ^ { \top }$ is the Moore-Penrose pseudoinverse of $\mathbf { C } _ { I } .$ . Combining the limits of both terms, we get:

$$
\lim _ {\lambda \rightarrow 0} \mathbf {W} _ {L _ {2}} = \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+} + \mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I}).
$$

This is precisely the direct finetuning solution, $\mathbf { W } _ { \mathrm { F T } }$

Analysis of the Limit as $\lambda \to \infty$ . For the limit as $\lambda \to \infty$ , we factor out $\lambda { : }$

$$
\begin{array}{l} \mathbf {W} _ {L _ {2}} = \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \lambda \mathbf {W} _ {I} ^ {0}\right) \frac {1}{\lambda} \left(\frac {1}{\lambda} \mathbf {C} _ {I} + \mathbf {I}\right) ^ {- 1} \\ = \left(\frac {1}{\lambda} \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} + \mathbf {W} _ {I} ^ {0}\right) \left(\frac {1}{\lambda} \mathbf {C} _ {I} + \mathbf {I}\right) ^ {- 1}. \\ \end{array}
$$

As $\lambda  \infty ,$ , the term $\begin{array} { r } { \frac { 1 } { \lambda }  0 } \end{array}$ . Therefore, the expression converges to:

$$
\lim _ {\lambda \rightarrow \infty} \mathbf {W} _ {L _ {2}} = \left(\mathbf {0} + \mathbf {W} _ {I} ^ {0}\right) (\mathbf {0} + \mathbf {I}) ^ {- 1} = \mathbf {W} _ {I} ^ {0}.
$$

Thus, the regularization parameter λ smoothly interpolates the solution between two meaningful extremes: pure task adaptation and pure preservation of pretrained weights.

Self-Distillation. The solution $\mathbf { W } _ { S D }$ provides the most sophisticated and effective compromise. We can rewrite it to reveal its structure:

$$
\begin{array}{l} \mathbf {W} _ {S D} = \mathbf {W} _ {I} ^ {0} - \frac {1}{1 + \lambda} \mathbf {W} _ {I} ^ {0} \mathcal {P} _ {I} + \frac {1}{1 + \lambda} \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+} \\ = \mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I}) + \mathbf {W} _ {I} ^ {0} \mathcal {P} _ {I} - \frac {1}{1 + \lambda} \mathbf {W} _ {I} ^ {0} \mathcal {P} _ {I} + \frac {1}{1 + \lambda} \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+}\right) \\ = \underbrace {\mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I})} _ {\text { Component   orthogonal   to   finetuning   data   (Preserved)}} + \underbrace {\frac {\lambda}{1 + \lambda} \left(\mathbf {W} _ {I} ^ {0} \mathcal {P} _ {I}\right) + \frac {1}{1 + \lambda} \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+}\right)} _ {\text { Component   within   finetuning   data   subspace   (Convex   Combination) }} \\ \end{array}
$$

Interpretation: Self-Distillation operates with surgical precision: 1. Outside the finetuning subspace, it acts as an identity function, preserving the components of the pretrained model that are irrelevant to the new task. 2. Inside the finetuning subspace, it does not discard the pretrained knowledge. Instead, it computes a convex combination of the projected pretrained weights and the optimal solution for the new contrastive task. The hyperparameter λ smoothly controls this trade-off. This demonstrates that Self-Distillation achieves a “best of both worlds” scenario: preserving general capabilities while adapting to new information where necessary.

# C.5. Dynamic Self-Distillation: WMA Details and Convergence

We extend the analysis of static self-distillation to a dynamic teacher, specifically a Weighted Moving Average (WMA) teacher, which adapts its regularization throughout training. This section provides the detailed definitions, dynamics, and convergence proofs.

Definition C.3 (SD–WMA Objective (Repeated from Main Text)). At step t, the student weights WtI solve

$$
\mathcal {L} _ {\mathrm{SD-WMA}} \left(\mathbf {W} _ {I}\right) = \frac {1}{2} \left\| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}} \right\| _ {\mathrm{F}} ^ {2} + \frac {\lambda}{2} \left\| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {W} _ {\text {Teacher}} ^ {t - 1} \mathbf {X} _ {I} \right\| _ {\mathrm{F}} ^ {2}, \quad \text {initialized from } \mathbf {W} _ {I} ^ {t - 1}. \tag {14}
$$

Definition C.4 (Weighted Moving Average (WMA) Teacher (Repeated from Main Text)). Let the normalized time grid be

$$
\tau_ {k} = \frac {k + c _ {1}}{T + c _ {2}} \in (0, 1), \quad c _ {1}, c _ {2} > 0.
$$

Choose any nonnegative weighting kernel $\kappa : [ 0 , 1 ] \to \mathbb { R } _ { \geq 0 }$ and define unnormalized weights $\alpha _ { k } ~ = ~ \kappa ( \tau _ { k } )$ The online normalization and teacher recursion are

$$
\omega_ {t} = \frac {\alpha_ {t}}{\sum_ {j = 0} ^ {t} \alpha_ {j}}, \quad \mathbf {W} _ {\text {Teacher}} ^ {t} = (1 - \omega_ {t}) \mathbf {W} _ {\text {Teacher}} ^ {t - 1} + \omega_ {t} \mathbf {W} _ {I} ^ {t}, \quad \mathbf {W} _ {\text {Teacher}} ^ {0} = \mathbf {W} _ {I} ^ {0}. \tag {15}
$$

Remark C.5 (Teacher as a normalized history average). Unrolling equation 15 yields a normalized convex average of the student’s history:

$$
\mathbf {W} _ {\text {Teacher}} ^ {t} = \sum_ {k = 0} ^ {t} \underbrace {\frac {\alpha_ {k}}{\sum_ {j = 0} ^ {t} \alpha_ {j}}} _ {\omega_ {k | t}} \mathbf {W} _ {I} ^ {k}, \qquad \omega_ {k | t} \geq 0, \quad \sum_ {k = 0} ^ {t} \omega_ {k | t} = 1.
$$

Thus the teacher is an expectation with respect to the discrete distribution Categorica $. ( \omega _ { 0 | t } , \hdots , \omega _ { t | t } ) \colon { \mathbf { W } } _ { \mathrm { T e a c h e r } } ^ { t } = \mathbb { E } _ { K \sim \omega _ { \cdot | t } } [ { \mathbf { W } } _ { I } ^ { K } ] ,$ .

# C.5.1. WMA VS. EMA TEACHERS

This section contrasts the proposed Weighted Moving Average (WMA) teacher with the standard Exponential Moving Average (EMA), which underlies mean-teacher approaches.

EMA (mean-teacher). EMA maintains an exponentially decaying average:

$$
\mathbf {W} _ {\mathrm{EMA}} ^ {t} = \rho \mathbf {W} _ {\mathrm{EMA}} ^ {t - 1} + (1 - \rho) \mathbf {W} _ {I} ^ {t}, \quad \rho \in (0, 1), \quad \mathbf {W} _ {\mathrm{EMA}} ^ {0} = \mathbf {W} _ {I} ^ {0}. \tag {16}
$$

Unrolling this recursion gives a geometric kernel over lag:

$$
\mathbf {W} _ {\mathrm{EMA}} ^ {t} = \rho^ {t} \mathbf {W} _ {I} ^ {0} + (1 - \rho) \sum_ {k = 1} ^ {t} \rho^ {t - k} \mathbf {W} _ {I} ^ {k} = \sum_ {k = 0} ^ {t} \underbrace {\omega_ {k | t} ^ {\mathrm{EMA}}} _ {\text {depends on t - k}} \mathbf {W} _ {I} ^ {k},
$$

$\omega _ { 0 | t } ^ { \mathtt { E M A } } = \rho ^ { t } , \omega _ { k | t } ^ { \mathtt { E M A } } = ( 1 - \rho ) \rho ^ { t - k } \mathrm { f o r } k \geq 1$ , and $\begin{array} { r } { \sum _ { k = 0 } ^ { t } \omega _ { k | t } ^ { \mathrm { E M A } } = 1 } \end{array}$ . The kernel is stationary in lag: weights depend only on recency $t - k .$

WMA (normalized-time kernel). In contrast, WMA assigns weights via a kernel over normalized time $\tau _ { k } = ( k + c _ { 1 } ) / ( T + c _ { 2 } ) \ d t$

$$
\mathbf {W} _ {\mathrm{WMA}} ^ {t} = \sum_ {k = 0} ^ {t} \underbrace {\omega_ {k | t} ^ {\mathrm{WMA}}} _ {\propto \kappa (\tau_ {k})} \mathbf {W} _ {I} ^ {k}, \qquad \omega_ {k | t} ^ {\mathrm{WMA}} = \frac {\alpha_ {k}}{\sum_ {j = 0} ^ {t} \alpha_ {j}}, \quad \alpha_ {k} = \kappa (\tau_ {k}).
$$

Here the kernel is position-aware in absolute (normalized) time, not just lag. The symmetric Beta kernel $( \beta _ { 1 } = \beta _ { 2 } )$ permits simultaneous emphasis of both endpoints (early stability and late convergence), a pattern that is not attainable with any single-parameter EMA.

# Key differences.

• Shape control. EMA imposes a monotone geometric decay from the present; WMA can be early-peaked, late-peaked, flat (uniform), bimodal (e.g., arcsine), etc.   
• Invariance to schedule granularity. WMA weights are defined on normalized time: if the training is retimed or step granularity changes while preserving the path over [0, 1], the kernel κ need not be retuned. EMA depends on the absolute decay $\rho$ and typically requires retuning when $\bar { T }$ or logging cadence changes.   
• Endpoint behavior. With $\beta _ { 1 } = \beta _ { 2 } = \textstyle { \frac { 1 } { 2 } }$ (arcsine), WMA places substantial weight near $k \approx 0$ and $k \approx t ,$ preserving early information and emphasizing late iterates; EMA cannot simultaneously upweight both ends.   
• Recovering classical averages. Choosing κ uniform (Beta(1, 1)) yields the simple running average (Polyak/Ruppert; SWA (Izmailov et al., 2018)). EMA cannot realize an exactly uniform window without time-varying $\rho _ { t }$ .   
• Online normalization. Both EMA and WMA are online and convex at each step; WMA’s $\omega _ { t } = \alpha _ { t } / \sum _ { j \leq t } \alpha _ { j }$ admits arbitrary nonnegative $\alpha _ { t }$ induced by κ.

Mean-teacher within the WMA recursion. In SD–WMA (Definition C.4), the step weight is $\omega _ { t } = \alpha _ { t } / \sum _ { j = 0 } ^ { t } \alpha _ { j }$ , which is generally time-varying. To recover EMA exactly with constant $\omega \equiv 1 - \rho ,$ choose any $\alpha _ { 0 } > 0$ and set, for $t \geq 1$ ,

$$
\alpha_ {t} = \frac {\omega}{(1 - \omega) ^ {t}} \alpha_ {0} \quad \Longleftrightarrow \quad \alpha_ {t} = \frac {1 - \rho}{\rho^ {t}} \alpha_ {0}, \tag {17}
$$

which yields $\omega _ { t } \equiv \omega$ and makes the WMA recursion identical to equation 16. If one insists on $\alpha _ { t } = \kappa ( \tau _ { t } )$ with $\tau _ { t } = ( t + c _ { 1 } ) / ( T + c _ { 2 } )$ , EMA corresponds to an exponential kernel over normalized time, $\kappa ( \tau ) = C ( 1 - \omega ) ^ { - ( T + c _ { 2 } ) \tau + c _ { 1 } ^ { \prime } }$ , for suitable constants $C , c _ { 1 } ^ { \prime }$ (fixed per run), which reproduces $\omega _ { t } \equiv \omega$ via equation 17.

# C.5.2. THE PERSISTENT REGULARIZER OF THE WMA TEACHER

A key advantage of the WMA teacher over the more common EMA teacher lies in the dynamics of the regularization it provides. The self-distillation loss, ${ \mathcal { L } } _ { \mathrm { S D } } .$ , induces a regularizing gradient field, $\mathbf { g } _ { R } ( \mathbf { W } _ { I } ^ { t } ) : = \nabla _ { \mathbf { W } _ { I } } \mathcal { L } _ { \mathrm { S D } } ( \mathbf { W } _ { T } ^ { t } , \mathbf { W } _ { I } ^ { t } )$ ), that pulls the student towards the teacher. The persistence of this field is critical for preventing the student from over-specializing on the finetuning task.

The Vanishing Regularizer of EMA. An EMA teacher is a low-pass filter of the student’s trajectory: $\mathbf { W } _ { \mathrm { E M A } } ^ { t } = \rho \mathbf { W } _ { \mathrm { E M A } } ^ { t - 1 } + ( 1 -$ $\mathbf { \Pi } _ { \rho ) } ^ { } \mathbf { W } _ { I } ^ { t } . \mathbf { A s }$ the student’s updates converge $( \lVert \mathbf { W } _ { I } ^ { t + 1 } - \mathbf { W } _ { I } ^ { t } \rVert _ { F } \to 0 )$ , the teacher necessarily converges to the student’s final parameters $( \mathrm { l i m } _ { t  \infty } \Vert \mathbf { W } _ { \mathrm { E M A } } ^ { t } - \mathbf { W } _ { I } ^ { t } \Vert _ { F } = 0 )$ . Consequently, any regularizer based on the teacher–student gap vanishes:

$$
\lim _ {t \to \infty} \| \mathbf {g} _ {R} (\mathbf {W} _ {I} ^ {t}; \mathbf {W} _ {\mathrm{EMA}} ^ {t}) \| _ {F} = 0,
$$

allowing the optimization to be dominated entirely by the task loss near the end of training.

The Finite-Horizon Persistence of WMA. The WMA teacher is a weighted average of the entire student history: $\mathbf { W _ { W M A } ^ { \mathit { t } } } =$ $\scriptstyle \sum _ { k = 0 } ^ { t } \omega _ { k \mid t } \mathbf { W } _ { I } ^ { k }$ . For any fixed finite run of length $T ,$ any kernel with $\alpha _ { 0 } > 0$ yields $\omega _ { 0 \mid T } > 0 ,$ , so W0I contributes nontrivially to $\mathbf { W _ { W M A } ^ { \mathrm { T } } }$ Thus, when the student has moved away from initialization $( \mathbf { W } _ { I } ^ { T } \neq \mathbf { W } _ { I } ^ { 0 } )$ , the teacher can remain separated from the final iterate, yielding a nontrivial regularizing gradient at step T . Note that under online normalization, the relative weight on any fixed k typically satisfies $\omega _ { k | t } \to 0 \mathrm { a s } t \to \infty ;$ therefore any “non-vanishing” claim must be understood as a finite-horizon / late-training statement. In particular, under infinite-horizon training with online normalization and a convergent student $( \mathbf { W } _ { I } ^ { t } \to \mathbf { W } _ { I } ^ { \infty } )$ , one typically has $\mathbf { W } _ { \mathrm { W M A } } ^ { t } \to \mathbf { W } _ { I } ^ { \infty }$ , so the teacher–student gap can vanish asymptotically.

Theorem C.6 (Finite-Horizon Persistence of the WMA Regularizer). Fix a training horizon T and let the WMA teacher at the end of training be

$$
\mathbf {W} _ {\mathrm{WMA}} ^ {T} = \sum_ {k = 0} ^ {T} \omega_ {k | T} \mathbf {W} _ {I} ^ {k}, \quad \omega_ {k | T} \geq 0, \sum_ {k = 0} ^ {T} \omega_ {k | T} = 1,
$$

with $\omega _ { 0 | T } > 0$ (true for any kernel with $\alpha _ { 0 } > 0 )$ . Assume the student trajectory is monotone along a single direction in parameter space: there exist a unit matrix U with $\| \mathbf { U } \| _ { F } = 1$ and scalars $0 = a _ { 0 } \leq a _ { 1 } \leq \cdot \cdot \cdot \leq a _ { T }$ such that

$$
\mathbf {W} _ {I} ^ {k} = \mathbf {W} _ {I} ^ {0} + a _ {k} \mathbf {U} \quad \forall k \in \{0, \dots , T \}.
$$

Then, if $\mathbf { W } _ { I } ^ { T } \neq \mathbf { W } _ { I } ^ { 0 }$ , the teacher remains strictly behind the final iterate and

$$
\left\| \mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {\mathrm{WMA}} ^ {T} \right\| _ {F} \geq \omega_ {0 | T} \left\| \mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {I} ^ {0} \right\| _ {F} > 0.
$$

Moreover, suppose the self-distillation loss is locally approximately quadratic in the teacher–student parameter difference $( \mathrm { e . g . }$ , secondorder KL expansion) so that for W near $\mathbf { W } _ { I } ^ { T }$ ,

$$
\mathbf {g} _ {R} (W; \mathbf {W} _ {\mathrm{WMA}} ^ {T}) = \nabla_ {W} \mathcal {L} _ {\mathrm{SD}} (\mathbf {W} _ {\mathrm{WMA}} ^ {T}, W) \approx \mathbf {F} _ {T} (W - \mathbf {W} _ {\mathrm{WMA}} ^ {T}),
$$

and define the terminal linearization residual

$$
\mathbf {r} _ {T} := \mathbf {g} _ {R} (\mathbf {W} _ {I} ^ {T}; \mathbf {W} _ {\mathrm{WMA}} ^ {T}) - \mathbf {F} _ {T} (\mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {\mathrm{WMA}} ^ {T}).
$$

If the curvature satisfies $\mathbf { F } _ { T } \succeq \mu \mathbf { I }$ on span{U} for some $\mu > 0 ,$ , then the regularizing gradient at the end of training admits the signal-minus-error lower bound

$$
\left\| \mathbf {g} _ {R} \left(\mathbf {W} _ {I} ^ {T}; \mathbf {W} _ {\mathrm{WMA}} ^ {T}\right) \right\| _ {F} \geq \mu \omega_ {0 | T} \left\| \mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {I} ^ {0} \right\| _ {F} - \left\| \mathbf {r} _ {T} \right\| _ {F}.
$$

In particular, if $\| \mathbf { r } _ { T } \| _ { F } < \mu \omega _ { 0 | T } \| \mathbf { W } _ { I } ^ { T } - \mathbf { W } _ { I } ^ { 0 } \| _ { F } ,$ , then ${ \bf g } _ { R } ( { \bf W } _ { I } ^ { T } ; { \bf W } _ { \mathrm { W M A } } ^ { T } ) \neq { \bf 0 }$ .

In contrast, for an EMA teacher with fixed $\rho \in ( 0 , 1 )$ , if $\mathbf { W } _ { I } ^ { t } \to \mathbf { W } _ { I } ^ { \infty }$ then $\lVert \mathbf { W } _ { \mathrm { E M A } } ^ { t } - \mathbf { W } _ { I } ^ { t } \rVert _ { F } \to 0$ , and thus the corresponding regularizing gradient vanishes asymptotically.

Proof Sketch. Step 1: Teacher–student gap under monotone 1D motion. Under the assumed form $\mathbf { W } _ { I } ^ { k } = \mathbf { W } _ { I } ^ { 0 } + a _ { k } \mathbf { U }$

$$
\mathbf {W} _ {\mathrm{WMA}} ^ {T} = \sum_ {k = 0} ^ {T} \omega_ {k | T} (\mathbf {W} _ {I} ^ {0} + a _ {k} \mathbf {U}) = \mathbf {W} _ {I} ^ {0} + \Bigl (\sum_ {k = 0} ^ {T} \omega_ {k | T} a _ {k} \Bigr) \mathbf {U}.
$$

Hence

$$
\mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {\mathrm{WMA}} ^ {T} = \left(a _ {T} - \sum_ {k = 0} ^ {T} \omega_ {k | T} a _ {k}\right) \mathbf {U} = \sum_ {k = 0} ^ {T} \omega_ {k | T} (a _ {T} - a _ {k}) \mathbf {U}.
$$

Since $a _ { T } \geq a _ { k }$ and $\omega _ { k | T } \geq 0$ , all terms are nonnegative multiples of the same direction, so there is no cancellation and

$$
\left\| \mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {\mathrm{WMA}} ^ {T} \right\| _ {F} = \sum_ {k = 0} ^ {T} \omega_ {k | T} (a _ {T} - a _ {k}) \geq \omega_ {0 | T} (a _ {T} - a _ {0}) = \omega_ {0 | T} \left\| \mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {I} ^ {0} \right\| _ {F}.
$$

Step 2: Gradient lower bound. Let $\boldsymbol { \Delta } _ { T } = \mathbf { W } _ { I } ^ { T } - \mathbf { W } _ { \mathrm { W M A } } ^ { T } \in \mathrm { s p a n } \{ \mathbf { U } \}$ . By definition, ${ \bf g } _ { R } ( { \bf W } _ { I } ^ { T } ; { \bf W } _ { \mathrm { W M A } } ^ { T } ) = { \bf F } _ { T } \Delta _ { T } + { \bf r } _ { T }$ , so by the triangle inequality,

$$
\| \mathbf {g} _ {R} (\mathbf {W} _ {I} ^ {T}; \mathbf {W} _ {\mathrm{WMA}} ^ {T}) \| _ {F} \geq \| \mathbf {F} _ {T} \Delta_ {T} \| _ {F} - \| \mathbf {r} _ {T} \| _ {F}.
$$

Using $\mathbf { F } _ { T } \succeq \mu I$ on span{U} and Cauchy–Schwarz, $\| \mathbf { F } _ { T } \Delta _ { T } \| _ { F } \geq \mu \| \Delta _ { T } \| _ { F }$ , hence

$$
\| \mathbf {g} _ {R} (\mathbf {W} _ {I} ^ {T}; \mathbf {W} _ {\mathrm{WMA}} ^ {T}) \| _ {F} \geq \mu \| \Delta_ {T} \| _ {F} - \| \mathbf {r} _ {T} \| _ {F} \geq \mu \omega_ {0 | T} \| \mathbf {W} _ {I} ^ {T} - \mathbf {W} _ {I} ^ {0} \| _ {F} - \| \mathbf {r} _ {T} \| _ {F}.
$$

Step 3: EMA vanishing. $\operatorname { I f } \mathbf { W } _ { I } ^ { t } \to \mathbf { W } _ { I } ^ { \infty }$ , then the EMA recursion is a stable linear filter of a convergent signal, implying $\mathbf { W } _ { \mathrm { E M A } } ^ { t } \to \mathbf { W } _ { I } ^ { \infty }$ and thus $\lVert \mathbf { W } _ { \mathrm { E M A } } ^ { t } - \mathbf { \check { W } } _ { I } ^ { t } \rVert _ { F } \stackrel { . } { \to } 0 .$ □

Conclusion. This theorem characterizes a finite-horizon effect: even when the student’s updates become small near the end of training, a WMA teacher can remain separated from the terminal iterate at step T because it retains positive mass on earlier states, yielding a regularizing gradient whose magnitude is lower bounded by a signal minus approximation error term. Over an infinite horizon with online normalization (where $\omega _ { 0 \mid t }  0 )$ and a convergent student, the teacher–student gap can vanish, consistent with bias-free convergence results proved later for SD–WMA in the task subspace.

# C.5.3. CONVERGENCE ANALYSIS

We first state the single-step solution and then derive global convergence in the task subspace.

From static to dynamic SD as a sequence of quadratic problems. Before stating the single-step solution, it is useful to make explicit what the SD–WMA scheme is doing as an optimization process. At each step t, the SD–WMA objective

$$
\mathcal {L} _ {\mathrm{SD-WMA}} (\mathbf {W} _ {I}) = \frac {1}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {Y} _ {\mathrm{FT}} \| _ {\mathrm{F}} ^ {2} + \frac {\lambda}{2} \| \mathbf {W} _ {I} \mathbf {X} _ {I} - \mathbf {W} _ {\text {Teacher}} ^ {t - 1} \mathbf {X} _ {I} \| _ {\mathrm{F}} ^ {2}
$$

is still a static quadratic problem in $\mathbf { W } _ { I } { \mathrm { : } }$ it has the exact same algebraic form as the static self-distillation objective analyzed in Theorem C.2, with two simple substitutions relative to the static-SD case:

• the anchor in the distillation term is replaced from the fixed pretrained weights $\mathbf { W } _ { I } ^ { 0 }$ to the current WMA teacher $\mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t - 1 }$   
• the initialization of gradient descent is replaced from $\mathbf { W } _ { I } ^ { 0 }$ to the previous student iterate $\mathbf { W } _ { I } ^ { t - 1 }$ .

These two substitutions touch different parts of the closed-form solution given by Lemma C.1:

• changing the anchor $\mathbf { W } _ { I } ^ { 0 }  \mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t - 1 }$ modifies the range component of the solution (the part lying in range(XI ), i.e., the task subspace), since this anchor enters the data term $\mathbf { P } _ { S D } = \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top } + \lambda \mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t - 1 } \mathbf { C } _ { I }$ ;   
• changing the initialization $\mathbf { W } _ { I } ^ { 0 } \ \to \ \mathbf { W } _ { I } ^ { t - 1 }$ modifies the null-space component $\Pi _ { \mathfrak { Q } ^ { \perp } } ( \cdot )$ which Lemma C.1 preserves from the initialization unchanged; this is what allows the orthogonal pretrained knowledge to be carried forward from one step to the next.

The dynamic-teacher scheme is therefore best understood as a sequence of standard static-SD-style quadratic minimizations, with both the anchor and the initialization updated between rounds in a way that affects geometrically separate subspaces. Proposition C.7 below makes this decomposition concrete in closed form.

Proposition C.7 (Single-Step Solution). Let $\mathbf { W } _ { \mathrm { F T } } ^ { \star } { = } \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top } ( \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top } ) ^ { + }$ be the minimum-norm solution for the direct finetuning task, and let $\mathcal { P } _ { I }$ be the orthogonal projector onto range(XI). The SD–WMA update at step t yields

$$
\mathbf {W} _ {I} ^ {t} = \mathbf {W} _ {I} ^ {t - 1} (\mathbf {I} - \mathcal {P} _ {I}) + \frac {\lambda}{1 + \lambda} \mathbf {W} _ {\text {Teacher}} ^ {t - 1} \mathcal {P} _ {I} + \frac {1}{1 + \lambda} \mathbf {W} _ {\mathrm{FT}} ^ {\star}. \tag {18}
$$

Proof. This proposition is immediate from applying Lemma C.1 to the objective in Definition C.3. The objective at step t has the same structure as static self-distillation (analyzed in Theorem C.2), but with the pretrained weights $\mathbf { W } _ { I } ^ { 0 }$ in the regularization term replaced by $\mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t - 1 } .$ $\begin{array} { r } { { \frac { \lambda } { 2 } } \left\| \mathbf { W } _ { I } \mathbf { X } _ { I } - \mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t - 1 } \mathbf { X } _ { I } \right\| _ { \mathrm { F } } ^ { 2 } } \end{array}$ , and the initialization for gradient descent being IThis corresponds to the self-distillation case in Theorem C.2, where $\mathbf { W } _ { I } ^ { t - 1 }$ . Specifically, we find the minimizer of: minWI $\mathbf { W } _ { I } ^ { 0 }$ 2  is effectively replaced by $\begin{array} { r } { { \bf \frac { 1 } { 2 } } \left\| { \bf W } _ { I } { \bf X } _ { I } - { \bf Y } _ { \mathrm { F T } } \right\| _ { \mathrm { F } } ^ { 2 } + } \end{array}$ $\mathbf { W } _ { \mathrm { T e a c h } } ^ { t - 1 }$ er 2forfor purposin the F fining the fixed regularization target at this step. The solution form is then directly obtained by substituting formula, which yields: $\mathbf { W } _ { \mathrm { T e a c h e t } } ^ { t - 1 }$ $\mathbf { W } _ { I } ^ { 0 }$ $\mathbf { W } _ { S D }$

$$
\mathbf {W} _ {I} ^ {t} = \mathbf {W} _ {I} ^ {t - 1} (\mathbf {I} - \mathcal {P} _ {I}) + \frac {1}{1 + \lambda} \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+} + \frac {\lambda}{1 + \lambda} \mathbf {W} _ {\text { Teacher }} ^ {t - 1} \mathcal {P} _ {I}.
$$

Recognizing $\mathbf { W } _ { \mathrm { F T } } ^ { \star } = \mathbf { Y } _ { \mathrm { F T } } \mathbf { X } _ { I } ^ { \top } ( \mathbf { X } _ { I } \mathbf { X } _ { I } ^ { \top } ) ^ { + }$ , we get the desired result.

The key advantage over static SD emerges from the teacher’s evolution.

Theorem C.8 (Bias-Free Convergence in the Task Subspace). Let $\begin{array} { r } { a = \frac { \lambda } { 1 + \lambda } } \end{array}$ and define the teacher error $\mathbf { E } ^ { t } = ( \mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } ) \mathcal { P } _ { I }$ Then for any online weights $\left\{ \omega _ { t } \right\}$ as in equation 15:

(i) Teacher contraction. $\begin{array} { r } { \mathbf { E } ^ { t } = \left( 1 - \frac { \omega _ { t } } { 1 + \lambda } \right) \mathbf { E } ^ { t - 1 } } \end{array}$ .

(ii) Student tracking. $( \mathbf { W } _ { I } ^ { t } - \mathbf { W } _ { \mathrm { F I } } ^ { \star } ) \mathcal { P } _ { I } = a \mathbf { E } ^ { t - 1 }$ .

(iii) Convergence. If $\textstyle \sum _ { t > 1 } \omega _ { t } = \infty$ , then $\mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t } \mathcal { P } _ { I }  \mathbf { W } _ { \mathrm { F T } } ^ { \star }$ and $\mathbf { W } _ { I } ^ { t } \mathcal { P } _ { I }  \mathbf { W } _ { \mathrm { F T } } ^ { \star }$ .

Proof. Let $\mathbf { W } _ { I , \parallel } ^ { t } = \mathbf { W } _ { I } ^ { t } \mathcal { P } _ { I }$ and $\mathbf { W } _ { \mathrm { T e a c h e r } , \parallel } ^ { t } = \mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t } \mathcal { P } _ { I }$ . From Proposition C.7, projecting onto the subspace range $\left( \mathbf { X } _ { I } \right)$ gives:

$$
\mathbf {W} _ {I, \parallel} ^ {t} = \mathbf {W} _ {I} ^ {t - 1} (\mathbf {I} - \mathcal {P} _ {I}) \mathcal {P} _ {I} + \frac {\lambda}{1 + \lambda} \mathbf {W} _ {\text { Teacher }} ^ {t - 1} \mathcal {P} _ {I} + \frac {1}{1 + \lambda} \mathbf {W} _ {\mathrm{FT}} ^ {\star} \mathcal {P} _ {I}.
$$

Since $( \mathbf { I } - \mathcal { P } _ { I } ) \mathcal { P } _ { I } = \mathbf { 0 }$ , and $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ is already in the parallel subspace (by definition), we have $\mathbf { W } _ { \mathrm { F T } } ^ { \star } \mathcal { P } _ { I } = \mathbf { W } _ { \mathrm { F T } } ^ { \star } . \mathbf { \mu } \mathbf { S } \mathbf { o }$ ,

$$
\mathbf {W} _ {I, \parallel} ^ {t} = a \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} + (1 - a) \mathbf {W} _ {\mathrm{FT}} ^ {\star}, \tag {19}
$$

where $\begin{array} { r } { a = \frac { \lambda } { 1 + \lambda } } \end{array}$ . (ii) Subtracting $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ from both sides of equation 19:

$$
\mathbf {W} _ {I, \parallel} ^ {t} - \mathbf {W} _ {\mathrm{FT}} ^ {\star} = a \mathbf {W} _ {\text {Teacher}, \parallel} ^ {t - 1} + (1 - a) \mathbf {W} _ {\mathrm{FT}} ^ {\star} - \mathbf {W} _ {\mathrm{FT}} ^ {\star} = a \left(\mathbf {W} _ {\text {Teacher}, \parallel} ^ {t - 1} - \mathbf {W} _ {\mathrm{FT}} ^ {\star}\right) = a \mathbf {E} ^ {t - 1}.
$$

This proves part (ii).

(i) Now consider the teacher recursion (Definition C.4) projected onto $\mathcal { P } _ { I } \mathbf { : }$

$$
\mathbf {W} _ {\text { Teacher }, \parallel} ^ {t} = (1 - \omega_ {t}) \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} + \omega_ {t} \mathbf {W} _ {I, \parallel} ^ {t}.
$$

Substitute equation 19 into this:

$$
\mathbf {W} _ {\text { Teacher }, \parallel} ^ {t} = (1 - \omega_ {t}) \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} + \omega_ {t} (a \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} + (1 - a) \mathbf {W} _ {\text { FT }} ^ {\star}).
$$

Rearranging terms to isolate $\mathbf { E } ^ { t } = \mathbf { W } _ { \mathrm { T e a c h e r } , \parallel } ^ { t } - \mathbf { W } _ { \mathrm { F T } } ^ { \star }$

$$
\begin{array}{l} \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t} - \mathbf {W} _ {\mathrm{FT}} ^ {\star} = (1 - \omega_ {t}) \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} + \omega_ {t} a \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} + \omega_ {t} (1 - a) \mathbf {W} _ {\mathrm{FT}} ^ {\star} - \mathbf {W} _ {\mathrm{FT}} ^ {\star} \\ = \left(1 - \omega_ {t} + \omega_ {t} a\right) \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} - \left(1 - \omega_ {t} (1 - a)\right) \mathbf {W} _ {\mathrm{FT}} ^ {\star} \\ = (1 - \omega_ {t} (1 - a)) (\mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} - \mathbf {W} _ {\text { FT }} ^ {\star}). \\ \end{array}
$$

Since $\textstyle 1 - a = 1 - { \frac { \lambda } { 1 + \lambda } } = { \frac { 1 } { 1 + \lambda } }$ 11+λ , we have:

$$
\mathbf {E} ^ {t} = \left(1 - \frac {\omega_ {t}}{1 + \lambda}\right) \mathbf {E} ^ {t - 1}.
$$

This proves part (i).

(iii) Iterating the recurrence relation from part (i):

$$
\| \mathbf {E} ^ {t} \| _ {F} = \left(\prod_ {k = 1} ^ {t} \left(1 - \frac {\omega_ {k}}{1 + \lambda}\right)\right) \| \mathbf {E} ^ {0} \| _ {F}.
$$

For $\mathbf { E } ^ { t }$ to converge to $0 ,$ , we need the product term to converge to 0. This occurs if and only if the sum ∞ $\scriptstyle \sum _ { k = 1 } ^ { \infty } { \frac { \omega _ { k } } { 1 + \lambda } }$ diverges to ∞. Since $\lambda > 0 , 1 + \lambda$ is a finite constant. Thus, the condition for convergence is $\textstyle \sum _ { k = 1 } ^ { \infty } \omega _ { k } = \infty$ . From Definition $\begin{array} { r } { \mathbb { C } . 4 , \omega _ { t } = \frac { \alpha _ { t } } { \sum _ { j = 0 } ^ { t } \alpha _ { j } } . \operatorname { I f } \kappa ( \tau _ { t } ) } \end{array}$ is a continuous function on [0, 1] that is non-zero on a set of positive measure, then $\sum _ { k } \alpha _ { k }$ will diverge as $T \to \infty$ (assuming t goes up to $T )$ , and thus $\sum _ { k } \omega _ { k }$ will diverge. For common kernels like Beta distributions $( \mathrm { e . g . }$ , arcsine kernel), this condition holds. Since $\mathbf { E } ^ { t } \to \mathbf { 0 }$ we have $\mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t } \mathcal { P } _ { I }  \mathbf { W } _ { \mathrm { F T } } ^ { \star }$ . From part (ii), as $\mathbf { E } ^ { t - 1 }  \mathbf { 0 } .$ , it follows that $( \mathbf { W } _ { I } ^ { t } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } ) \mathbf { \mathcal { P } } _ { I }  \mathbf { 0 }$ , meaning $\mathbf { W } _ { I } ^ { t } \mathcal { P } _ { I }  \mathbf { W } _ { \mathrm { F T } } ^ { \star }$ . □

Corollary C.9 (Linear rate under a bounded step weight). If $\omega _ { t } \geq \omega _ { \operatorname* { m i n } } > 0$ for all $t \leq T$ , then

$$
\left\| \left(\mathbf {W} _ {\text { Teacher }} ^ {t} - \mathbf {W} _ {\mathrm{FT}} ^ {\star}\right) \mathcal {P} _ {I} \right\| _ {F} \leq \left(1 - \frac {\omega_ {\min}}{1 + \lambda}\right) ^ {t} \left\| \left(\mathbf {W} _ {\text { Teacher }} ^ {0} - \mathbf {W} _ {\mathrm{FT}} ^ {\star}\right) \mathcal {P} _ {I} \right\| _ {F}.
$$

Hence the training loss in the task subspace decays at least geometrically to the minimum, whereas static SD converges to a biased point for any fixed $\lambda > 0$ .

Proof. This follows directly from Theorem C.8 part (i). $\operatorname { I f } \omega _ { t } \geq \omega _ { \operatorname* { m i n } }$ , then $\begin{array} { r } { 1 - \frac { \omega _ { t } } { 1 + \lambda } \le 1 - \frac { \omega _ { \operatorname* { m i n } } } { 1 + \lambda } } \end{array}$ . Since $0 < \omega _ { \mathrm { m i n } } \le 1$ and $\lambda > 0 ,$ we have $\begin{array} { r } { 0 < \frac { \omega _ { \mathrm { m i n } } } { 1 + \lambda } < 1 , \mathrm { s o } 0 < 1 - \frac { \omega _ { \mathrm { m i n } } } { 1 + \lambda } < 1 } \end{array}$ . Thus, the error contracts geometrically. Static SD, as derived in Theorem C.2, converges to a solution that is a convex combination of $\mathbf { W } _ { I } ^ { 0 } \mathcal { P } _ { I }$ and $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ . This is a biased point unless $\mathbf { W } _ { I } ^ { 0 } \mathcal { P } _ { I } = \mathbf { W } _ { \mathrm { F T } } ^ { \star }$ . □

Geometric Interpretation of Dynamic Self-Distillation. We decompose the dynamics into orthogonal and parallel components with respect to range(XI ).

Orthogonal Preservation. Applying $\left( \mathbf { I } - \mathcal { P } _ { I } \right)$ to Proposition C.7 and using the idempotency of projectors, $\mathcal { P } _ { I } ( { \bf I } - \mathcal { P } _ { I } ) = { \bf 0 } .$ , we get:

$$
\mathbf {W} _ {I} ^ {t} (\mathbf {I} - \mathcal {P} _ {I}) = \mathbf {W} _ {I} ^ {t - 1} (\mathbf {I} - \mathcal {P} _ {I}) = \dots = \mathbf {W} _ {I} ^ {0} (\mathbf {I} - \mathcal {P} _ {I}),
$$

This demonstrates that SD–WMA preserves pretrained knowledge orthogonal to the finetuning subspace, just like static self-distillation.

Adaptive Task-Space Evolution. Within the task subspace, the student update is given by:

$$
\mathbf {W} _ {I, \parallel} ^ {t} = \frac {\lambda}{1 + \lambda} \mathbf {W} _ {\text { Teacher }, \parallel} ^ {t - 1} + \frac {1}{1 + \lambda} \mathbf {W} _ {\mathrm{FT}} ^ {\star}.
$$

Early training (t small): The teacher because few updates have occurred). T $\mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t - 1 }$ is still close to s the teacher ac $\mathbf { W } _ { I } ^ { 0 }$ (as  a st $\omega _ { k }$ for small k is often high for U-shaped kernels, or simplyg anchor, mitigating catastrophic forgetting during volatile updates.

$t \to \infty .$ $\mathbf { W } _ { \mathrm { T e a c h e r } , \parallel } ^ { t - 1 }$ $\mathbf { W } _ { \mathrm { F T } } ^ { \star } .$

$$
\lim _ {t \to \infty} \mathbf {W} _ {I, \parallel} ^ {t} = \frac {\lambda}{1 + \lambda} \mathbf {W} _ {\mathrm{FT}} ^ {\star} + \frac {1}{1 + \lambda} \mathbf {W} _ {\mathrm{FT}} ^ {\star} = \mathbf {W} _ {\mathrm{FT}} ^ {\star}.
$$

Thus, the dynamic teacher adapts, reducing anchor bias and enabling exact convergence to $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ in range(XI ).

Proposition C.10 (Dominance over Static SD). $\mathrm { I f } \ \lVert \mathbf { W } _ { \mathrm { T e a c h e r } , \parallel } ^ { t - 1 } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } \rVert _ { F } \leq \lVert \mathbf { W } _ { I , \parallel } ^ { 0 } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } \rVert _ { F }$ , then for the same λ the SD–WMA update attains lower squared error than static SD in the task subspace.

Proof. Let $\mathbf { W } _ { \mathrm { s t a t i c } } ^ { \star } \mathbf { \hat { s } } \mathbf { D }$ be the solution for static SD (from Theorem C.2). The squared error from $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ in the task subspace for static SD is proportional to the teacher is clo $\begin{array} { r } { \big \| \frac { \lambda } { 1 + \lambda } \mathbf { W } _ { I } ^ { 0 } \mathcal { P } _ { I } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } \big \| _ { F } ^ { 2 } } \end{array}$ . For dynamic SD, the instantanl subspace than the initial model $\begin{array} { r } { \lVert \frac { \lambda } { 1 + \lambda } \mathbf { W } _ { \mathrm { T e a c h e r } } ^ { t - 1 } \mathbf { \hat { \mathcal { P } } } _ { I } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } \rVert _ { F } ^ { 2 } } \end{array}$ . Ifhen $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ $\mathbf { W } _ { I } ^ { 0 } , \mathrm { i . e . , } \| \mathbf { W } _ { \mathrm { T e a c h e r , \parallel } } ^ { t - 1 } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } \| _ { F } \leq \| \mathbf { W } _ { I , \parallel } ^ { 0 } - \mathbf { W } _ { \mathrm { F T } } ^ { \star } \| _ { F }$ the dynamic SD solution will be closer to $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ in that subspace, thus achieving lower error. The convergence result (Theorem C.8) guarantees that the teacher gets arbitrarily close to $\mathbf { W } _ { \mathrm { F T } } ^ { \star }$ , eventually satisfying this condition. □

# C.6. Distillation Loss Definitions in TRACER

TRACER employs a composite self-distillation loss LSD-WMA from the WMA teacher, which consists of several complementary terms to transfer different aspects of knowledge. Let T denote the teacher model and S denote the student model. $\mathbf { h } _ { I _ { i } } ^ { \mathbf { T } }$ and $\mathbf { h } _ { T _ { i } } ^ { \mathbf { T } ^ { \dagger } }$ are image and text embeddings from the teacher for the i-th example, and similarly for the student. τ denotes the temperature parameter.

Feature Distillation (FD). This loss directly minimizes the Mean Squared Error between the student’s and teacher’s embeddings for each corresponding image-text pair in a mini-batch of size N . It helps align the feature spaces.

$$
\mathcal {L} _ {\mathrm{FD}} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\left\| \mathbf {h} _ {I _ {i}} ^ {\mathbf {T}} - \mathbf {h} _ {I _ {i}} ^ {\mathbf {S}} \right\| _ {2} ^ {2} + \left\| \mathbf {h} _ {T _ {i}} ^ {\mathbf {T}} - \mathbf {h} _ {T _ {i}} ^ {\mathbf {S}} \right\| _ {2} ^ {2}\right) \tag {20}
$$

Contrastive Relational Distillation (CRD). CRD aligns the student’s contrastive similarity distribution with the teacher’s. We first compute the image-to-text $( p )$ and text-to-image (q) softmax distributions for both student and teacher across the mini-batch:

$$
p _ {i} ^ {\mathbf {T}} [ j ] = \frac {\exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {T} \top} \mathbf {h} _ {T _ {j}} ^ {\mathbf {T}} / \tau)}{\sum_ {b = 1} ^ {N} \exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {T} \top} \mathbf {h} _ {T _ {b}} ^ {\mathbf {T}} / \tau)}, \quad p _ {i} ^ {\mathbf {S}} [ j ] = \frac {\exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {T _ {j}} ^ {\mathbf {S}} / \tau)}{\sum_ {b = 1} ^ {N} \exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {T _ {b}} ^ {\mathbf {S}} / \tau)} \tag {21}
$$

$$
q _ {i} ^ {\mathbf {T}} [ j ] = \frac {\exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {T} \top} \mathbf {h} _ {I _ {j}} ^ {\mathbf {T}} / \tau)}{\sum_ {b = 1} ^ {N} \exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {T} \top} \mathbf {h} _ {I _ {b}} ^ {\mathbf {T}} / \tau)}, \quad q _ {i} ^ {\mathbf {S}} [ j ] = \frac {\exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {I _ {j}} ^ {\mathbf {S}} / \tau)}{\sum_ {b = 1} ^ {N} \exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {I _ {b}} ^ {\mathbf {S}} / \tau)} \tag {22}
$$

The distillation loss is the sum of the KL-divergences between these distributions, averaged over the batch.

$$
\mathcal {L} _ {\mathrm{CRD}} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(D _ {K L} (p _ {i} ^ {\mathbf {T}} \| p _ {i} ^ {\mathbf {S}}) + D _ {K L} (q _ {i} ^ {\mathbf {T}} \| q _ {i} ^ {\mathbf {S}})\right) \tag {23}
$$

Interactive Contrastive Learning (ICL). ICL forces the student to learn within the teacher’s embedding space by performing contrastive learning between the student’s anchor embeddings and the teacher’s key embeddings. The loss is a symmetric InfoNCE objective computed on these mixed-model pairs.

$$
\mathcal {L} _ {\mathrm{ICL}} = - \frac {1}{2 N} \sum_ {i = 1} ^ {N} \left(\log \frac {\exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {T _ {i}} ^ {\mathbf {T}} / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {T _ {j}} ^ {\mathbf {T}} / \tau)} + \log \frac {\exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {I _ {i}} ^ {\mathbf {T}} / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {I _ {j}} ^ {\mathbf {T}} / \tau)}\right) \tag {24}
$$

Cross Knowledge Distillation (Cross-KD). This method acts as a hybrid of CRD and ICL. It aligns the student-to-teacher cross-modal similarity distribution with the teacher’s self-modal distribution using KL-divergence. We define the student-to-teacher

cross-modal distributions $( p ^ { \mathbf { S } \to \mathbf { T } } , q ^ { \mathbf { S } \to \mathbf { T } } )$ as:

$$
p _ {i} ^ {\mathbf {S} \rightarrow \mathbf {T}} [ j ] = \frac {\exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {T _ {j}} ^ {\mathbf {T}} / \tau)}{\sum_ {b = 1} ^ {N} \exp (\mathbf {h} _ {I _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {T _ {b}} ^ {\mathbf {T}} / \tau)} \tag {25}
$$

$$
q _ {i} ^ {\mathbf {S} \rightarrow \mathbf {T}} [ j ] = \frac {\exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {I _ {j}} ^ {\mathbf {T}} / \tau)}{\sum_ {b = 1} ^ {N} \exp (\mathbf {h} _ {T _ {i}} ^ {\mathbf {S} \top} \mathbf {h} _ {I _ {b}} ^ {\mathbf {T}} / \tau)} \tag {26}
$$

The loss then minimizes the divergence from these distributions to the teacher’s own relational distributions, $p _ { i } ^ { \mathbf { T } }$ and $q _ { i } ^ { \mathbf { T } }$ .

$$
\mathcal {L} _ {\text { CrossKD }} = \frac {1}{2 N} \sum_ {i = 1} ^ {N} \left(D _ {K L} (p _ {i} ^ {\mathbf {T}} \| p _ {i} ^ {\mathbf {S} \to \mathbf {T}}) + D _ {K L} (q _ {i} ^ {\mathbf {T}} \| q _ {i} ^ {\mathbf {S} \to \mathbf {T}})\right) \tag {27}
$$

Geometric bridge to composite distillation. Our analysis decomposes learning into an orthogonal preservation term and an in-subspace mixing term (Equation 2). The composite distillation terms are chosen to preserve structure consistent with this geometry: (i) FD anchors pointwise embeddings, biasing updates toward the teacher component within range(XI) while damping drift in orthogonal directions; (ii) CRD aligns the teacher’s batch-wise similarity distributions, preserving inter-example geometry (a probabilistic surrogate for preserving S=H⊤I HT ); (iii) ICL performs contrastive learning in the teacher’s semantic space, encouraging the student to operate on the teacher’s subspace and thus to mix along task-relevant directions; and (iv) CrossKD aligns cross-modal logits to transmit cross-modal relational structure that vanilla InfoNCE may underweight. Together with the WMA teacher, these terms operationalize the geometric principle at feature-, relation-, and cross-modal levels.

# C.7. Connection to Robustness via Inter-Class Feature Sharing

The self-distillation approach, particularly with a dynamic WMA teacher, can be understood through the lens of recent theoretical work on multimodal contrastive learning’s robustness mechanisms. Xue et al. (2024) identify inter-class feature sharing as a key mechanism behind MMCL’s strong robustness to distribution shift, where models learn to leverage information about features appearing across different classes to dissociate spurious correlations.

Building on the insight that self-distillation acts as instance-specific label smoothing (Zhang & Sabuncu, 2020), we argue that the self-distillation method provides a similar robustness benefit by acting as an informed label smoothing mechanism that preserves inter-class similarities learned during pretraining. To see this connection, recall the self-distillation solution from Theorem C.2:

$$
\mathbf {W} _ {S D} = \mathbf {W} _ {I} ^ {0} \left(\mathbf {I} - \frac {1}{1 + \lambda} \mathcal {P} _ {I}\right) + \frac {1}{1 + \lambda} \mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} (\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}) ^ {+} \tag {28}
$$

This solution exhibits three key properties that enhance robustness:

Preservation of Cross-Class Knowledge. The term $\begin{array} { r } { \mathbf { W } _ { I } ^ { 0 } \left( \mathbf { I } - \frac { 1 } { 1 + \lambda } \mathcal { P } _ { I } \right) } \end{array}$ maintains the pretrained model’s understanding of feature relationships across classes. Unlike direct finetuning which completely overwrites representations in the finetuning subspace, selfdistillation retains a weighted contribution from the original cross-class feature covariances. This is analogous to how Xue et al. (2024) show that MMCL leverages features appearing in multiple contexts to learn their independence from class labels.

Informed Smoothing via Pretrained Similarities. By regularizing towards $\mathbf { W } _ { I } ^ { 0 } \mathbf { X } _ { I }$ rather than arbitrary targets, self-distillation performs label smoothing that is informed by the pretrained model’s learned inter-class similarities. This extends the instance-specific label smoothing interpretation of Zhang & Sabuncu (2020) to the finetuning setting, where the smoothing is guided by pretrained knowledge. This regularization preserves the cross-covariance structure that Xue et al. (2024) identify as crucial for robustness, specifically the covariance between features that appear independently across different classes.

Robustness Through Feature Independence. Within the finetuning subspace, self-distillation computes a convex combination:

$$
\frac {\lambda}{1 + \lambda} \left(\mathbf {W} _ {I} ^ {0} \mathcal {P} _ {I}\right) + \frac {1}{1 + \lambda} \left(\mathbf {Y} _ {\mathrm{FT}} \mathbf {X} _ {I} ^ {\top} \left(\mathbf {X} _ {I} \mathbf {X} _ {I} ^ {\top}\right) ^ {+}\right) \tag {29}
$$

This combination maintains the pretrained understanding of feature independence while adapting to the new task. As Xue et al. (2024) demonstrate in their Data Model 2, when features can occur independently across classes $( \mathrm { e . g . }$ , “trees without green leaves” appearing in non-tree classes), models that preserve these cross-class relationships achieve stronger robustness. The self-distillation mechanism explicitly preserves these relationships through the weighted contribution of $\mathbf { W } _ { I } ^ { 0 } \mathcal { P } _ { I }$ .

The hyperparameter λ controls the strength of this inter-class knowledge preservation: larger values of λ maintain more of the pretrained model’s understanding of how features vary independently across different contexts, potentially enhancing robustness to distribution shift. This suggests that self-distillation’s effectiveness stems not merely from preventing catastrophic forgetting, but from actively preserving the rich inter-class feature relationships that contribute to robustness, a mechanism that parallels the theoretical insights of Xue et al. (2024) on why MMCL achieves strong out-of-distribution generalization.

# D. TRACER Algorithm

Algorithm 1 TRACER (Trajectory-Robust Anchoring for Contrastive Encoder Regularization)   
Require: Pretrained CLIP model $\theta_{\mathrm{CLIP}}^0 = \{\mathcal{E}_{\mathrm{Image}},\mathcal{E}_{\mathrm{Text}}^0\}$ Require: Finetuning dataset $\mathcal{D}_{\mathrm{FT}} = \{(\mathbf{x}_I,\mathbf{x}_T)\}_{i=1}^N$ Require: Learning rate $\eta$ , Weight decay $\delta$ , Batch size $B$ , Number of epochs $E$ Require: Distillation coefficient $\lambda_{\mathrm{SD}}$ Require: WMA kernel $\kappa(\tau_k)$ (e.g., $\mathrm{Beta}(\beta_1,\beta_2)$ ) and total steps $T_{\mathrm{total}}$ Require: Temperature $\tau_{\mathrm{NCE}}$ for InfoNCE losses

1: Initialize Student Model: $\theta_S \leftarrow \theta_{\mathrm{CLIP}}^0$ (image encoder $\mathcal{E}_{\mathrm{Image},S}$ , text encoder $\mathcal{E}_{\mathrm{Text},S}$ )

2: Initialize Teacher Model: $\theta_T \leftarrow \mathrm{copy}(\theta_S)$ 3: Initialize Optimizer: Opt $\leftarrow \mathrm{AdamW}(\theta_S.\mathrm{parameters}(),\eta ,\delta)$ 4: Initialize WMA state: cumulative_alpha $\leftarrow 0$ 5: global_step $\leftarrow 0$ 6: for epoch = 1 to $E$ do

7:    for batch = $\{(\mathbf{x}_I,\mathbf{x}_T)\}_{i=1}^B$ in $\mathcal{D}_{\mathrm{FT}}$ do

8:    global_step $\leftarrow$ global_step + 1
    {— Student Forward Pass —}

9: $\mathbf{h}_{I,S} \leftarrow \mathcal{E}_{\mathrm{Image},S}(\mathbf{x}_I)$ 10: $\mathbf{h}_{T,S} \leftarrow \mathcal{E}_{\mathrm{Text},S}(\mathbf{x}_T)$ 11:    Normalize student embeddings: $\mathbf{h}_{I,S} \leftarrow$ normalize $(\mathbf{h}_{I,S})$ , $\mathbf{h}_{T,S} \leftarrow$ normalize $(\mathbf{h}_{T,S})$ {— Compute Multi-Modal Contrastive Loss ( $\mathcal{L}_{\mathrm{MMCL}}$ ) —}

12:    logits$_{I \leftrightarrow T}$ $\leftarrow$ $\mathbf{h}_{I,S} \cdot \mathbf{h}_{T,S}^\top/\tau_{\mathrm{NCE}}$ 13: $\mathcal{L}_{\mathrm{MMCL}} \leftarrow$ InfoNCE ( $\logits_{I \leftrightarrow T}$ ) + InfoNCE ( $\logits_{I \leftrightarrow T}^\top$ ) {Symmetric InfoNCE}
    {— Teacher Forward Pass (with no gradient updates) —}

14:    with torch.no_grad() :

15: $\mathbf{h}_{I,T} \leftarrow \mathcal{E}_{\mathrm{Image},T}(\mathbf{x}_I)$ 16: $\mathbf{h}_{T,T} \leftarrow \mathcal{E}_{\mathrm{Text},T}(\mathbf{x}_T)$ 17:    Normalize teacher embeddings: $\mathbf{h}_{I,T} \leftarrow$ normalize $(\mathbf{h}_{I,T})$ , $\mathbf{h}_{T,T} \leftarrow$ normalize $(\mathbf{h}_{T,T})$ {—Compute Dynamic Self-Distillation Loss ( $\mathcal{L}_{\mathrm{SD-WMA}}$ ) —}

18: $\mathcal{L}_{\mathrm{FD}} \leftarrow \frac{1}{B} \sum_{i=1}^{B} (\|\mathbf{h}_{I,T}[i] - \mathbf{h}_{I,S}[i]\|_2^2 + \| \mathbf{h}_{T,T}[i] - \mathbf{h}_{T,S}[i]\|_2^2)$ 19: $\mathcal{L}_{\mathrm{CRD}} \leftarrow KL(\text {softmax } (\mathbf{h}_{I,T}\mathbf{h}_{T,T}^\top/\tau_{\mathrm{NCE}})\|\text {softmax } (\mathbf{h}_{I,S}\mathbf{h}_{T,S}^\top/\tau_{\mathrm{NCE}}))$ {+ text-to-image}
20: $\mathcal{L}_{\mathrm{ICL}} \leftarrow$ InfoNCE ( $\mathbf{h}_{I,S}$ , $\mathbf{h}_{T,T}$ ) + InfoNCE ( $\mathbf{h}_{T,S}$ , $\mathbf{h}_{I,T}$ )
21: $\mathcal{L}_{\mathrm{CrossKD}} \leftarrow KL(\text {softmax } (\mathbf{h}_{I,T}\mathbf{h}_{T,T}^\top/\tau_{\mathrm{NCE}})\|\text {softmax } (\mathbf{h}_{I,S}\mathbf{h}_{T,T}^\top/\tau_{\mathrm{NCE}}))$ {+ text-to-image}
22: $\mathcal{L}_{\mathrm{SD-WMA}} \leftarrow \mathcal{L}_{\mathrm{FD}} + \mathcal{L}_{\mathrm{CRD}} + \mathcal{L}_{\mathrm{ICL}} + \mathcal{L}_{\mathrm{CrossKD}}$ {— Total Loss and Optimization —}

23: $\mathcal{L}_{\mathrm{Total}} \leftarrow \mathcal{L}_{\mathrm{MMCL}} + \lambda_{\mathrm{SD}} \cdot \mathcal{L}_{\mathrm{SD-WMA}}$ 24:    Opt.zero_grad()
25: $\mathcal{L}_{\mathrm{Total}}.\mathrm{backward}(){}$ 26:    Opt.step()
    {— Update WMA Teacher —}

27: $\tau_{\mathrm{current}} \leftarrow (\text {global\_step} + c_1)/(T_{\mathrm{total}} + c_2)$ {Normalized time}
28: $\alpha_{\mathrm{current}} \leftarrow \kappa (\tau_{\mathrm{current}})$ 29:    cumulative_alpha $\leftarrow$ cumulative_alpha + α $_{\mathrm{current}}$ 30:    ω $_{\mathrm{current}} \leftarrow α_{\mathrm{current}}/\text {cumulative\_alpha}$ 31:    for parameter $p_S$ in θ $_S$ and $p_T$ in θ $_T$ do
32: $p_T \leftarrow (1 - ω_{\mathrm{current}}) \cdot p_T + ω_{\mathrm{current}} \cdot p_S$ 33:    end for

34:    end for

35: end for

36: return θ $_S$

# E. Reproducibility Details

To ensure full reproducibility, we detail our experimental setup, key hyperparameters, and implementation. The full codebase, configuration files, and reproduction scripts are publicly available at https://github.com/HesamAsad/TRACER.

# E.1. Computational Environment

• Operating System: Linux kernel 5.14.0-427.42.1.el9\_4.x86\_64.   
• GPU Hardware: NVIDIA H100 80GB HBM3.   
• NVIDIA Driver Version: 550.144.03.   
• CUDA Version: 12.4.   
• Python Version: 3.10.4.   
• PyTorch Version: 2.0.1+ (with CUDA support).

# E.2. Implementation and Training Details

Our implementation extends the OpenAI CLIP framework.

• Model Architectures: We use pretrained CLIP models (ViT-B/16, ResNet50, ViT-L/14) from OpenAI’s official clip library.   
• Total Loss: $\mathcal { L } _ { \mathrm { T R A C E R } } = \mathcal { L } _ { \mathrm { M M C L } } + \lambda _ { \mathrm { S D } } \mathcal { L } _ { \mathrm { S D - W M A } } .$   
– LMMCL: Symmetric InfoNCE loss, directly leveraging OpenAI CLIP’s core loss implementation. Optional cross-Frobenius regularizer coefficient was set to 0.05.   
– LSD-WMA: A composite self-distillation loss. For TRACER, this comprises Feature Distillation (FD), Contrastive Relational Distillation (CRD), Interactive Contrastive Learning (ICL), and Cross Knowledge Distillation (Cross-KD).   
• WMA Teacher: A custom Weighted Moving Average (WMA) teacher implementation, whose weighting kernel is a Beta distribution with $\beta _ { 1 } = \beta _ { 2 } = 0 . 5$ .

# E.3. Key Hyperparameters

The following hyperparameters were used for TRACER finetuning on ImageNet-1K:

• Epochs: 10.   
• Optimizer: AdamW.   
• Learning Rate: $1 \times 1 0 ^ { - 5 }$ .   
• Weight Decay: 0.1.   
• Batch Size: 512 (ViT-B/16, RN50), 224 (ViT-L/14).   
• Warmup Length: 500 steps (cosine LR schedule).   
• Mixed Precision: Enabled using torch.amp.autocast with torch.bfloat16.   
• Distillation Coefficient λSD: 0.9.   
• WMA Beta Kernel Parameter: 0.5 (for Beta(0.5,0.5) kernel, i.e., arcsine distribution).   
• Teacher Update Frequency: 0 or 1 (update every step).

# E.4. Data Processing

Standard OpenAI CLIP image preprocessing was applied. Input images are sourced from ImageNet-1K, and finetuning captions from OpenAI class templates.

# E.5. Code Availability

The full codebase is publicly available at https://github.com/HesamAsad/TRACER to facilitate direct reproduction. The repository includes the TRACER training pipeline, evaluation scripts for all reported ImageNet and OOD benchmarks, configuration files for the three CLIP backbones (RN50, ViT-B/16, ViT-L/14), and the WMA teacher implementation.

# F. Extended Discussion

This appendix expands on three discussion points that we touch on briefly in the main text: how TRACER’s gain scales across backbones, how TRACER relates to parameter-efficient and prompt-based adaptation, and how the theory could be extended beyond the linearized setting.

# F.1. Backbone scaling: ViT-B/16 vs. ViT-L/14

A natural question is why TRACER’s OOD gain over CaRot is larger on ViT-B/16 (+1.53 Avg. shifts in Table 1) than on ViT-L/14 (+1.19 Avg. shifts in Table 7). Three factors plausibly contribute. First, ViT-L/14 is a substantially stronger zero-shot starting point (70.93% vs. 58.39% OOD), so there is less absolute headroom for any regularizer to recover; both methods sit closer to a ceiling, compressing differences. Second, even with a smaller absolute gap, TRACER still attains the best OOD accuracy among all compared methods on ViT-L/14 (75.32%, vs. 74.13% for CaRot), with consistent gains on the hardest shifts (IN-R +1.74, IN-A +2.19, ObjectNet +1.71), mirroring the ViT-B/16 pattern. Third, TRACER scales favorably in compute: its distillation cost is $\mathcal { O } ( B ^ { 2 } )$ in the batch and matches standard EMA cost for the teacher update (Table 4), so its overhead does not grow with the cubic-in-d spectral term that dominates CaRot at large d. We therefore expect the favorable cost–quality trade-off to persist or improve as backbones grow. We frame this as an empirical observation: we do not claim a demonstrated scaling law for TRACER across the full vision–language model size spectrum, and we view systematic studies on larger VLMs (e.g., SigLIP variants and CLIP-style backbones at ≥ViT-H/14) as natural follow-up work.

# F.2. Relation to PEFT and prompt-based adaptation

TRACER is a regularization mechanism that operates on the standard full-finetuning gradient flow: it modifies which solution the optimizer converges to (anchoring it to a trajectory-weighted teacher), but it does not restrict where parameters can move. Parameter-efficient finetuning (PEFT) methods such as adapters (Houlsby et al., 2019) and LoRA (Hu et al., 2022), as well as prompt-based adaptation (Zhou et al., 2022; Jia et al., 2022), take the orthogonal route: they restrict the hypothesis class so that only a small set of injected parameters or input tokens can change, leaving the pretrained weights untouched by construction. The two strategies address different failure modes of finetuning. PEFT bounds the capacity for drift away from the pretrained model; TRACER bounds the direction and magnitude of drift within whatever capacity the optimizer is given. Because TRACER’s WMA teacher and composite distillation losses act on the student’s parameters (or, equivalently, its embedding statistics) regardless of how many of those parameters are trainable, TRACER can in principle wrap a LoRA-finetuned or prompt-tuned model directly: the WMA teacher would then average a low-dimensional trajectory in adapter/prompt space rather than full-encoder space. We therefore see TRACER as complementary to PEFT and prompt-based adaptation rather than competing with them; a systematic empirical study of TRACER on top of LoRA, adapters, and prompt tuning is a natural follow-up.

# F.3. Future work: NTK and random-feature extensions of the theory

Our theoretical analysis (§3, §C) is developed in the linearized image/text-encoder setting that is standard in the contrastivelearning theory literature (Ji et al., 2023; Tian, 2022; Nakada et al., 2023; Xue et al., 2024). In that setting, the closed-form solutions, the contrastive target matrix, and the bias-free convergence of the SD–WMA teacher all admit clean derivations that we believe capture the essential mechanism underlying TRACER’s empirical robustness on nonlinear CLIP backbones (Figures 3–4). A natural next step is to lift these results to the neural tangent kernel (NTK) regime (Jacot et al., 2018) and to random-feature (RF) models, where the nonlinear encoders are approximated by linear maps in a feature space induced by their gradients or by random nonlinearities. In the NTK regime, the same matrix least-squares reformulation should apply with $\mathbf { X } _ { I }$ replaced by NTK features, allowing both the orthogonal/parallel decomposition and the WMA bias-free convergence theorem to be re-derived for finite-width networks under lazy training. The RF view, in turn, would let us study how the spectrum of the random feature map interacts with the WMA kernel κ(τ ) and the regularization strength λ. Establishing such extensions would tighten the connection between our linearized analysis and large nonlinear vision–language models, and we leave them as a concrete open direction.

# G. AI Usage

Large Language Models were used to improve the manuscript’s grammar and readability, and to assist with code formatting and refactoring during implementation. All research design, theoretical analysis, experimental protocols, and interpretation of results were conducted entirely by the authors.