# Fixed-Point Masked Generative Modeling

Andrea Miele∗ Yiming Qin LTS4, EPFL LTS4, EPFL

Alba Carballo-Castro LTS4, EPFL

Justin Deschenaux CLAIRE, EPFL

Pascal Frossard LTS4, EPFL

# Abstract

Masked Generative Models (MGMs) enable parallel decoding and achieve strong performance across modalities, but require full-sequence bidirectional transformers at every step, making training costly and degrading quality under low sampling budgets. Existing work improves efficiency via better samplers or cheaper fixeddepth denoisers, but they still allocate a fixed amount of denoiser computation to each refinement step. We introduce Fixed-Point Masked Generative Models (FP-MGMs), which replace part of the denoiser with a fixed-point solver over shared attention layers to enable adaptive depth with fewer parameters. To make it more effective for masked generation, we first introduce a cross-step consistency loss, which aligns hidden representations at neighboring denoising steps and, second, three-state reuse (3SR) which warm-starts the solver using the previous solution by treating differently unchanged, still-masked, and newly revealed tokens respectively. Together, these components define our complete training-to-inference framework for fixed-point masked generation, CoFRe. We also show that pretrained MGMs can be converted into FP-MGMs with short fine-tuning, avoiding full retraining. Across modalities, CoFRe improves the quality and cost tradeoff. On OpenWebText, CoFRe reduces parameters by 38.8%, training time by 11.5%, and VRAM by 16.9%, while improving generative perplexity from 830.8 to 101.8 at a budget of 96 transformer-block forward passes, compared to MDLM. In ImageNette, CoFRe reduces training time by 48.6% and VRAM by 50.7%, while improving FID in all sample budgets tested. Overall, CoFRe offers a practical framework for cheaper training and stronger low-budget masked generation.

![](images/eae30b73fde5b65f48d3241d85ef3aa393603658a792ed0a6ca61b57352782a5.jpg)

<details>
<summary>line</summary>

| Budget | FP-MDLM | CoFRe | MDLM | MDLM + SDTT |
| ------ | ------- | ----- | ---- | ----------- |
| 96     | 5.81    | 5.52  | 5.91 | 5.58        |
| 192    | 5.76    | 5.47  | 5.81 | 5.53        |
| 384    | 5.75    | 5.39  | 5.75 | 5.49        |
| 768    | 5.70    | 5.23  | 5.70 | 5.45        |
</details>

![](images/59d03709600f932a1608c2c1c8dcb3afc8d8dd6d69aed32dba4fe2e0b899a7ab.jpg)

<details>
<summary>bar</summary>

| Metric       | FP-MDLM Improvement |
| ------------ | ------------------- |
| #Params      | -38.8%              |
| Training time| -11.5%              |
| VRAM         | -16.9%              |
</details>

Figure 1: FP-MDLM and CoFRe improve the quality–cost trade-off on OWT. (Left) Generative perplexity across forward-pass budgets, with entropy in parentheses; CoFRe gives the best quality at all shown budgets. (Right) Relative to MDLM, FP-MDLM and CoFRe use fewer parameters, less training time, and less VRAM.

# 1 Introduction

Masked generative models (MGMs) generate sequences by iteratively denoising masked tokens, enabling parallel decoding and strong generation quality across modalities. Prominent examples include

<table><tr><td>Method family</td><td>Adaptive network depth</td><td>Cheaper training</td><td>Strong low-budget generation</td></tr><tr><td>Sampler improvements[5, 9, 26, 27, 51, 63]</td><td>✗</td><td>✗</td><td>√</td></tr><tr><td>Efficient fixed-depth architectures[16, 65]</td><td>✗</td><td>~</td><td>✗</td></tr><tr><td>Controllable-depth / looped / DEQ-style models[3, 12, 22, 35, 53, 68]</td><td>√</td><td>~</td><td>✗</td></tr><tr><td>Ours: FP-MGMs</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison of prior approaches and FP-MGMs. ∼ denotes partial coverage. We only focus here on discrete data.

MDLMs for language [49, 57, 61] and MaskGIT for images [7], with other related masked-generation approaches extending to video, audio, and multimodal generation [10, 47, 62, 67]. However, MGMs are computationally expensive [16, 60], since each refinement step runs a full bidirectional transformer pass over the entire sequence. Training therefore consumes large amounts of VRAM and is notably slow. Furthermore, sampling under a low compute budget, i.e., the total number of transformer-block forward passes, produces poor-quality samples [14]. Thus, improving MGMs requires controlling not only the number of denoising steps, but also the cost and effective depth of the denoiser passes at each denoising step.

Prior work addresses these issues and improves MGMs efficiency along three main axes. First, alternative samplers improve sample quality at fixed compute by changing which tokens are revealed or updated at each refinement step [5, 9, 26, 27, 51, 63]. Second, efficient but fixed-depth architectures reduce architectural waste, for example by avoiding computation on [MASK] tokens [16], or by designing more efficient masked generative backbones [65]. Finally, adaptive routing and looped transformers, and DEQ-style models provide controllable effective depth by repeatedly applying shared modules or dynamically allocating computation [3, 12, 22, 35, 53, 68].

However, these directions leave an important gap for masked generative modeling, as detailed in Tab. 1. Sampler improvements change the denoising trajectory but usually leave the training procedure and per-step denoiser unchanged, so the compute spent at each sampling step remains fixed independently of step difficulty. Efficient, fixed-depth masked architectures reduce compute, but still rely on a denoiser with fixed-capacity. This suggests that improving MGM efficiency not only requires changing the sampler, but also acting inside the denoiser: the model should reuse parameters to reduce backbone cost, while still allowing different refinement steps to use different amounts of computation. Fixed-point / DEQ-style layers provide a natural substrate for this goal: they replace an explicit stack of distinct layers with repeated applications of a shared block, whose equilibrium is used as the layer output [3]. Fixed-Point Diffusion Models further show that this idea can improve the quality–cost trade-off in continuous diffusion denoisers [4]. Yet masked generation introduces discrete, token-wise state changes, so continuous-diffusion techniques do not directly transfer and can be suboptimal. This motivates our FP-MGM framework, CoFRe, which combines fixed-point denoisers with additional mechanisms tailored to masked generation: cross-step consistency and token-aware three-state reuse.

# Contributions We organize our contributions around three claims.

(1) Fixed-point masked denoisers improve the quality-cost trade-off. Compared to using a fixed stack of distinct transformer layers, a fixed-point layer repeatedly applies a shared block and treats the resulting equilibrium as the layer output. Since the same block is reused across solver iterations, the model can increase or decrease its effective depth by changing the number of iterations, without adding parameters. We introduce Fixed-Point Masked Generative Models (FP-MGMs), which replace the middle layers of a masked denoiser with a weight-sharing fixed-point block. We apply this approach to two representative MGMs: MDLM [57] for text, yielding FP-MDLM, and MaskGIT [7] for images, yielding FP-MaskGIT. In both cases, the fixed-point denoiser reduces the number of parameters, training time, and memory, while allowing the effective denoiser depth to vary through the number of fixed-point iterations. This addresses limitations (1) and (2) (Fig. 1, Right).

(2) Three-state reuse makes FP-MDLM practical. Warm-starting from a fixed-point solution from the last denoising step is a standard way to make FP models efficient. However, in MGMs, the input sequence changes abruptly across refinement steps as tokens are revealed or replaced. As a result, the previous fixed-point solution is not equally reliable across positions. We therefore introduce a three-state reuse rule, referred to as 3SR, that treats unchanged visible tokens, still-masked tokens, and newly revealed tokens differently. By design, unchanged visible tokens fully reuse the previous fixed-point solution, still-masked tokens partially reuse it, and newly revealed tokens rely more on the current input injection.

(3) Cross-time regularization is key for strong low-budget generation. Architecture and reuse alone are not sufficient for strong low-budget sampling: abrupt changes in the input space also induce non-smooth changes in the representation space across denoising steps. We introduce ${ \mathcal { L } } _ { \mathrm { C O N S } }$ , a cross-step consistency loss that aligns the representations of a noisier student state and a cleaner teacher state. Empirically, this loss behaves like cross-time self-distillation, sharpening masked-token predictions and driving most of the low-budget generation gains. This addresses limitation (3) (Fig. 1, Left). Together, the fixed-point denoiser, cross-step consistency, and 3SR define CoFRe: a complete training-to-inference recipe. Finally, we also show that pretrained MDLM checkpoints can be converted into FP-MDLMs with a short distillation stage, avoiding full retraining from scratch. These components improve low-budget sampling and reduce the cost of obtaining strong FP-MDLM checkpoints, addressing limitations (2) and (3).

We evaluate FP-MDLM on OpenWebText (OWT) [23] and downstream tasks [20], and FP-MaskGIT on ImageNette [13, 32]. On OWT, FP-MDLM reduces parameters by 38.8%, training time by 11.5%, and VRAM by 16.9% relative to MDLM, while improving low-budget generative perplexity from 830.8 to 375.6 at budget 96. With cross-step consistency and 3SR, our model CoFRe further improves over MDLM+SDTT [15] in the low-budget regime, reducing generative perplexity from 193.1 to 101.8 at budget 96 and from 47.0 to 37.8 at budget 768. On ImageNette, CoFRe reduces training time by 48.6% and VRAM by 50.7% relative to MaskGIT-Large, while improving FID across all reported budgets. We also show that a pretrained MGM can be efficiently converted into a more compute-efficient FP architecture: improving over the 1M-step FP-MDLM baseline at every sampling budget, with only 4% of the original pretraining steps. Overall, CoFRe makes masked generative models cheaper to train, easier to adapt, and stronger under limited sampling budgets.

# 2 Background

In this paper, we use $\mathbf { x } = ( x ^ { 1 } , \ldots , x ^ { d } ) \in \mathcal { V } ^ { d }$ to denote a clean sequence of length d over vocabulary $\nu : = [ \bar { V } ]$ . We denote by $\mathbf { z } _ { t }$ a corrupted version of x at noise level or refinement state t, and by [MASK] the special mask token. A masked denoiser maps $( { \bf z } _ { t } , t )$ to token logits $\ell _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , t )$ , from which predictions or samples are obtained.

# 2.1 Discrete generative models and masked generative models

Discrete generative models learn distributions over sequences of categorical variables, such as text tokens or quantized image latents.

In this work, we focus on MGMs, which corrupt data by replacing tokens with a special mask token [MASK] and train a denoiser parameterized by θ to recover the clean sequence. The denoiser outputs logits $\ell _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , t )$ , and the training objective is

$$
\mathcal {L} _ {\mathrm{MGM}} := \mathbb {E} _ {\mathbf {x} \sim \mathcal {D}, t \sim \mathcal {U} [ 0, 1 ]} [ w (t) \mathrm{CE} _ {\mathcal {M} _ {t}} (\ell_ {\theta} (\mathbf {z} _ {t}, t), \mathbf {x}) ]. \tag {1}
$$

where $w : [ 0 , 1 ] \to \mathbb { R } ^ { + }$ is a weighting function, $\mathcal { M } _ { t } = \{ i : z _ { t } ^ { i } = [ \mathrm { M A S K } ] \}$ is the set of masked positions, and $\mathrm { \dot { C E } } _ { \mathcal { M } _ { t } } ( \ell _ { \theta } ( { \mathbf { z } } _ { t } , t ) , { \mathbf { x } } )$ denotes the cross-entropy between the predicted logits and the clean tokens, evaluated only on positions in $\mathcal { M } _ { t }$ . Sampling starts from a fully masked sequence and iteratively reveals subsets of tokens using repeated denoiser evaluations.

MDLM MDLMs [49, 57, 61] instantiate MGMs for language. MDLMs define an absorbing-state discrete diffusion process in which clean tokens are independently replaced by [MASK] tokens according to a time-dependent noise schedule. Training corresponds to the MGM objective in Eq. 1 with the MDLM weighting w(t) = α′t1−αt , $\begin{array} { r } { w ( t ) = \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } } \end{array}$ where $\alpha _ { t }$ is the noise schedule giving the probability that a token remains clean at time t, using the notation of Sahoo et al. [57]. To keep predictions coherent across noise levels during accelerated sampling, auxiliary temporal objectives such as consistency regularization or self-distillation through time (SDTT) [15] are commonly used.

MaskGIT MaskGIT [7] is a MGM for image tokens, typically operating in the latent space of a pretrained tokenizer. Unlike MDLM, which follows a diffusion-time masking process, MaskGIT relies on confidence-based iterative decoding: at each step, it predicts masked tokens, scores their confidence, and permanently reveals a subset of high-confidence positions. Following Besnier et al. [5], we use Halton low-discrepancy schedules [25] to obtain more uniform spatial coverage; further details are given in Appendix D.3.

# 2.2 Deep equilibrium models, and fixed-point diffusion models

A fixed point of a map $F _ { \theta }$ is a state $\mathbf { h } ^ { \star }$ that remains unchanged after applying the map: $\mathbf { h } ^ { \star } =$ $F _ { \theta } ( \mathbf { h } ^ { \star } ; \mathbf { u } )$ , where u is an external input. An $n _ { \mathrm { t h } }$ fixed-point layer uses this equilibrium state as its output, and approximates it by iterating a shared transformation, $\mathbf { h } ^ { n + 1 } = F _ { \theta } ^ { \mathbf { \bar { \alpha } } } ( \mathbf { h } ^ { n } ; \mathbf { u } )$ , or by using a numerical solver such as Broyden’s method or Anderson acceleration. This can be viewed as a weight-sharing network whose effective depth is controlled by the number of solver iterations.

Deep Equilibrium Models (DEQs) [3] use this principle to define hidden representations implicitly, with gradients computed by implicit differentiation or approximate Jacobian-free methods. Fixed-Point Diffusion Models (FPDMs) [4] have shown that fixed-point denoisers can improve the quality– cost trade-off in continuous diffusion by replacing part of the denoiser with an implicit weightsharing layer. At each diffusion timestep, the denoiser solves a fixed-point problem over hidden representations, and nearby timesteps often have similar solutions, enabling efficient warm starts from previous fixed-point states. This makes fixed-point layers a natural substrate for parameter-efficient and controllable-depth denoisers. Directly adapting FPDMs to masked generation is not sufficient, however. Masked refinement changes the input state discretely and token-wise: some positions remain visible, some remain masked, and others are newly revealed or replaced. Thus, previous fixed-point solutions are not uniformly reusable, and low-budget generation can still suffer from cross-step representation drift. We therefore introduce FP-MGMs together with CoFRe, a complete training-to-inference framework that adds cross-step consistency and token-aware three-state reuse to fixed-point masked denoisers.

# 3 Fixed-Point Denoising Networks for Masked Sequence Modeling

Having introduced masked generative models and the fixed-point perspective in the previous sections, we now describe how to combine them into controllable-depth denoisers designed specifically for discrete masked generation.

# 3.1 Fixed-point MGMs

A standard masked generative denoiser maps $\mathbf { \Psi } ( \mathbf { z } _ { t } , t )$ to token logits through a finite stack of transformer layers. We instead decompose the denoiser into four parts: an explicit preprocessing stack $P _ { \theta _ { P } }$ , an input-conditioning projection $G _ { \theta _ { G } }$ , an implicit fixed-point block $F _ { \theta _ { F } }$ , and an explicit postprocessing stack $H _ { \theta _ { H } }$ . These respectively produce the initial hidden state, transform it into a conditioning signal for the fixed-point layers, solve for the denoising representation, and map this representation to logits:

$$
\mathbf {h} _ {\text { pre }, t} = P _ {\theta_ {P}} \left(\mathbf {z} _ {t}, t\right), \tilde {\mathbf {h}} _ {t} = G _ {\theta_ {G}} \left(\mathbf {h} _ {\text { pre }, t}\right), \mathbf {h} _ {t} ^ {\star} = \operatorname{Fix} \left(F _ {\theta_ {F}} (\cdot ; \tilde {\mathbf {h}} _ {t}, t)\right), \ell_ {\theta} \left(\mathbf {z} _ {t}, t\right) = H _ {\theta_ {H}} \left(\mathbf {h} _ {t} ^ {\star}, t\right). \tag {2}
$$

where $\theta = \{ \theta _ { P } , \theta _ { G } , \theta _ { F } , \theta _ { H } \}$ and $\ell _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , t )$ are the output token logits. If no separate projection is used, $G _ { \theta _ { G } }$ is the identity. In practice, the fixed point is approximated by N iterations:

$$
\mathbf {h} _ {t} ^ {0} = \mathbf {h} _ {\text { pre }, t}, \quad \mathbf {h} _ {t} ^ {n + 1} = F _ {\theta_ {F}} (\mathbf {h} _ {t} ^ {n}; \tilde {\mathbf {h}} _ {t}, t), \quad n = 0, \dots , N - 1. \tag {3}
$$

Let $K _ { \mathrm { p r e } } , K _ { \mathrm { f p } }$ , and $K _ { \mathrm { p o s t } }$ denote the number of transformer blocks in the preprocessing stack, the fixed-point block, and the postprocessing stack, respectively. A refinement step then uses $K _ { \mathrm { p r e } } + N K _ { \mathrm { f p } } + K _ { \mathrm { p o s t } }$ transformer-block evaluations, while only parameterizing $K _ { \mathrm { p r e } } + K _ { \mathrm { f p } } + K _ { \mathrm { p o s t } }$ distinct layers. Thus, weight sharing reduces parameter count, while the number of solver iterations controls the effective denoiser depth. The original MGM objective and sampling rule are unchanged; only the architecture used to compute the logits is modified.

Model variants We apply FP-MGMs to two masked generative models with different transformer denoisers: MDLM, which uses a diffusion transformer, and MaskGIT, which uses a bidirectional masked-token transformer. We detail ablations on architecture parameters in Appendices D and F.

![](images/a903a4fa78dd029bb3a688156cbc382050acd9cbeaf8bdd2f0626f52d1f9d8d2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["z_{t_c}"] --> B["P_{θ_P}"]
    B --> C["F_{θ_F}"]
    C --> D["H_{θ_H}"]
    D --> E["L_CE"]
    D --> F["L_CONS"]
    G["z_{t_s}"] --> H["P_{θ_P}"]
    H --> I["F_{θ_F}"]
    I --> J["H_{θ_H}"]
    J --> K["L_CONS"]
    style A fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style G fill:#ccf,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    subgraph Ground Truth
        B
        C
        D
        E
        F
        G
        H
        J
    end
```
</details>

![](images/987501316f98fa2dfe2fbda094d4fc473dc110428e00da05bbe89a91e85d8392.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Sampling zT"] --> B["..."]
    B --> C["zt+1"]
    C --> D["..."]
    D --> E["zt"]
    E --> F["Previous FP solution"]
    E --> G["Current input solution"]
    F --> H["Initialized state h_t^0 = γ_t ⊙ h_{t+1}^* + (1 - γ_t) ⊙ h_pre,t"]
    G --> H
    H --> I["A → A\nγ_t : 1.0\nReuse previous"]
    H --> J["[M"] → [M]\nγ_t : increasing 0.75 to 0.9\nLargely reuse previous]
    H --> K["[M"] → D\nγ_t : 0.2\nLightly reuse previous]
    I --> L["Already visible token"]
    J --> M["Still-masked token"]
    K --> N["Newly revealed token"]
```
</details>

Figure 2: Training and sampling for fixed-point masked generative models. (Left) During training, FP-MGMs keep the masked modeling objective while replacing the middle transformer stack with an iterated shared fixed-point block. For cross-step consistency, correlated masks from the same clean sequence define a noisier student state and cleaner teacher state $( t _ { c } < t _ { s } ) ;$ the model is trained with the base cross-entropy loss plus ${ \mathcal { L } } _ { \mathrm { C O N S } }$ to align their hidden representations. (Right) During sampling, the fixed-point solver is warm-started from the previous denoising step using three-state reuse: visible tokens reuse fully, still-masked tokens partially reuse, and newly revealed tokens rely more on the current pre-layer representation.

# 3.2 Training a FP-MGM

Training uses the same task objective as the original masked generative model; our main modification is architectural.

Following Bai and Melas-Kyriazi [4], we train with Stochastic Jacobian-Free Backpropagation (SJFB), which avoids backpropagating through the full solver trajectory. At each training step, we sample $N _ { \mathrm { n g } } \sim \mathcal { U } \{ 0 , \dots , 4 \}$ no-gradient iterations and $N _ { \mathrm { g } } \sim \mathcal { U } \{ 3 , \ldots , 6 \}$ gradient-tracked iterations, where U denotes a discrete uniform distribution, and with $\dot { N } = \dot { N _ { \mathrm { n g } } } + N _ { \mathrm { g } }$ , with N defined in Section 3.1. The no-gradient iterations move the hidden state closer to the fixed point solution without storing activations, while the gradient-tracked iterations provide a tractable training signal for the fixed-point block. The resulting logits are passed to the original MGM loss. Full hyperparameter details are given in Appendix D.1 and more details about SJFB in Appendix C.1.

Cross-step consistency regularization The fixed-point architecture improves efficiency by reducing the cost of each denoising step, while strong low-budget generation requires more than cheaper updates. First, each update must predict the clean data accurately, since errors can quickly accumulate when only a few denoising steps are used. Second, fixed-point solutions should be reusable across adjacent denoising states, so the model does not need to resolve each state from scratch. Our lagged logit analysis highlights the first challenge: low-budget denoising can exhibit substantial cross-step logit drift, motivating an additional consistency signal to align student-step predictions with cleaner future states, improving prediction accuracy with noisier data (Figure 11). Moreover, in masked sequence generation, tokens may be revealed between steps, so the previous fixed-point solution is a useful but imperfect warm start for the next state, further motivating stabilizing predictions across adjacent steps. Specifically, we add a short post-training stage, aligning the representation of a noisier student state with that of a cleaner teacher state from the same trajectory, using correlated masks, i.e., nested masks where the teacher context is always at least as clean as the student’s. For a clean sequence x, we construct a noisier student input $( \mathbf { z } _ { t _ { s } } , t _ { s } )$ and a cleaner teacher input $\left( \mathbf { z } _ { t _ { c } } , t _ { c } \right)$ from the same underlying example, with $t _ { c } < t _ { s }$ . We then add a consistency term to the base MDLM objective, $\mathcal { L } = \mathcal { L } _ { \mathrm { M D L M } } + \lambda \mathcal { L } _ { \mathrm { C O N S } }$ . In our main experiments, the consistency term is an MSE loss on hidden states, $\mathcal { L } _ { \mathrm { C O N S } } = \left. \mathbf { h } _ { s } - \mathrm { s g } ( \mathbf { h } _ { c } ) \right. _ { 2 } ^ { 2 }$ , where hs and $\mathbf { h } _ { c }$ are the student and teacher final tokenwise pre-logit hidden states, after the FP and postprocessing blocks, and sg(·) denotes stop-gradient.

Although this loss is applied in representation space, it also helps improve the model output: empirically, it behaves like cross-time self-distillation, sharpening masked-token predictions and improving low-budget prediction quality. More details about the loss choice, training criterion, and correlated masks are in Appendices F.7, F.8, and C.3.3.

# 3.3 Sampling with three-state reuse

At inference time, sampling follows the standard masked denoising process over a fixed number of denoising steps. Since each step requires solving a fixed-point problem, the solver initialization directly affects how much useful denoising can be obtained under a limited forward-pass budget. Following the allocation ablation in Appendix F.10, we use a decreasing fixed-point iteration schedule, which allocates more solver steps early in the denoising trajectory.

At denoising step $t ,$ the fixed-point block is conditioned on the current input injection $\tilde { \mathbf { h } } _ { t } \ =$ $G _ { \theta _ { G } } ( \mathbf { h } _ { \mathrm { p r e } , t } )$ , where $\mathbf { h } _ { \mathrm { p r e } , t } = P _ { \theta _ { P } } ( \mathbf { z } _ { t } , t )$ . Without reuse, the solver is initialized from the current pre-layer output, $\mathbf { h } _ { t } ^ { 0 } = \mathbf { \bar { h } } _ { \mathrm { p r e } , t } . \mathbf { A }$ natural reuse strategy is to warm-start the solver from the fixed-point solution of the previous denoising step, $\mathbf h _ { t } ^ { 0 } = \mathbf h _ { t + 1 } ^ { \star }$ . The fixed-point problem is unchanged; only the solver initialization differs.

However, full reuse applies the same initialization rule to all positions, implicitly assuming that the previous fixed-point solution remains equally well aligned with the current fixed-point problem. In masked denoising, this assumption is violated in a token-dependent way: unchanged visible tokens preserve their local evidence, still-masked tokens keep the same local mask symbol but receive an updated context, and newly revealed tokens undergo a local conditioning shift. Thus, the initialization error induced by reuse is not uniform across positions. For each denoising transition under no reuse, we measure the tokenwise movement of the solved fixed-point state, $\| \mathbf h _ { t + 1 } ^ { \star } ( i ) - \mathbf h _ { t } ^ { \star } ( i ) \| _ { 2 }$ , and group positions by their discrete transition type (Figure 3). Newly revealed tokens undergo the largest representation shifts, while already visible tokens move the least. Still-masked tokens’

![](images/b2eb5fc5dfe7a433ee92b75698baa083b9931e7582137b6ed4c29c273a2b930f.jpg)

<details>
<summary>line</summary>

| Denoising step | Newly revealed | Still masked | Already visible |
| -------------- | --------------- | ------------ | --------------- |
| 0              | 1.0             | 1.0          | 1.0             |
| 1              | 1.0             | 0.3          | 0.3             |
| 2              | 1.0             | 0.2          | 0.2             |
| 3              | 1.0             | 0.15         | 0.15            |
| 4              | 1.0             | 0.1          | 0.1             |
| 5              | 1.0             | 0.08         | 0.08            |
| 6              | 1.0             | 0.05         | 0.05            |
| 7              | 1.0             | 0.03         | 0.03            |
</details>

Figure 3: Token transition type determines how reusable fixed-point states are. Newly revealed tokens move much more than stable tokens, motivating strong reuse for visible tokens, partial reuse for masked tokens, and weak reuse for newly revealed tokens.

movement significantly decreases as denoising progresses, reflecting that their conditional context changes less once more tokens are visible; this supports stronger reuse at lower noise levels.

Inspired by this, we introduce a three-state reuse rule, $\mathbf { h } _ { t } ^ { 0 } = \gamma _ { t } \odot \mathbf { h } _ { t + 1 } ^ { \star } + ( 1 - \gamma _ { t } ) \odot \mathbf { h } _ { \mathrm { p r e } , \ i }$ t where the token-wise modulation coefficient $\gamma _ { t } ^ { i }$ is broadcast over hidden dimensions and is defined as:

$$
\gamma_ {t} ^ {i} = \left\{ \begin{array}{l l} 1, & \text { if   position   i   is   an   unchanged   visible   token }, \\ \gamma_ {\text { mask }}, & \text { if   position   i   is   still   masked }, \\ 0. 2, & \text { if   position   i   is   newly   revealed }. \end{array} \right.
$$

Thus, stable visible positions inherit the previous fixed-point solution, still-masked positions use partial reuse, and changed positions move closer to the current pre-layer output. For still-masked tokens, we increase the partial-reuse coefficient as denoising progresses, linearly interpolating from γmask,min to γmask,max. We select the coefficients of $\gamma t$ , including γmask, min and γmask,max, by grid search analyzing the generation quality and diversity metrics. The full coefficient schedule, tuning procedure, and sampling algorithm are given in Appendix D.1.2 and Algorithm 1.

# 3.4 Pretrained model conversion and short adaptation

Finally, we show that FP-MDLMs need not be trained from scratch – given a pretrained MDLM, we are able to convert it into a more parameter-efficient FP-MDLM by mapping selected transformer layers to the preprocessing, fixed-point, and postprocessing blocks, with FP-specific projections initialized close to identity.

We then use a short teacher-student adaptation stage, similar in spirit to network distillation, to transfer the behavior of the original MDLM into the converted FP architecture. The original MDLM is kept frozen as the teacher. Using correlated masks from the same clean sequence, the converted FP-MDLM is trained with the base MDLM cross-entropy loss plus a temperature-scaled KL loss on student-masked positions $\begin{array} { r } { \mathcal { M } _ { s } \colon \mathcal { L } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda \tau ^ { 2 } \frac { 1 } { | \mathcal { M } _ { s } | } \overset { \cdot } { \sum } _ { i \in \mathcal { M } _ { s } } \operatorname { K L } ( p _ { c } ^ { \tau } ( i \ ) \| p _ { s } ^ { \tau } ( i ) ) } \end{array}$ . This validates that transformer denoisers can be distilled into fixed-point denoisers with only short adaptation; details are given in Appendix C.3.

<table><tr><td colspan="5">Language generation on OWT</td><td colspan="4">Image generation on ImageNette</td></tr><tr><td rowspan="2">Budget</td><td colspan="2">MDLM + SDTTTrain: ≈139h + SDTTVRAM: 112.4 GiB/GPU</td><td colspan="2">CoFReTrain: ≈123h + 30kVRAM: 93.44 GiB/GPU</td><td rowspan="2">Budget</td><td colspan="2">MaskGIT-LargeTrain: 17h46mVRAM: 72.45 GiB</td><td>CoFReTrain: 9h08mVRAM: 35.74 GiB</td></tr><tr><td>Gen. PPL ↓</td><td>Entropy ↑</td><td>Gen. PPL ↓</td><td>Entropy ↑</td><td>FID ↓</td><td>IS ↑</td><td>FID ↓</td></tr><tr><td>96</td><td>193.050</td><td>5.580</td><td>101.791</td><td>5.434</td><td>48</td><td>174.0856</td><td>9.2860</td><td>96.7331</td></tr><tr><td>192</td><td>89.170</td><td>5.530</td><td>65.182</td><td>5.380</td><td>96</td><td>117.6439</td><td>13.3696</td><td>51.0077</td></tr><tr><td>384</td><td>62.290</td><td>5.490</td><td>48.755</td><td>5.283</td><td>192</td><td>54.6172</td><td>16.0220</td><td>27.6242</td></tr><tr><td>768</td><td>47.040</td><td>5.450</td><td>37.846</td><td>5.142</td><td>384</td><td>30.0202</td><td>14.6473</td><td>22.8381</td></tr></table>

Table 2: Main quality–cost results for language (Left) and image generation (Right). Budgets count transformer-block forward passes. For language, the training/VRAM values indicate the main backbone cost; SDTT uses additional short distillation stages. Entropy is reported to contextualize diversity; these values correspond to the selected operating point for each budget, while Appendix F.9 provides the broader quality–diversity landscape obtained by varying the allocation between denoising steps and fixed-point iterations.

# 4 Experiments

We organize the experiments around three questions. First, does CoFRe improve the end-to-end quality–cost trade-off in language and image generation (Section 4.1)? Second, can a pretrained MDLM be converted into an effective FP-MDLM with a short adaptation stage? Third, which components are responsible for the gains? We answer the first question with the main OWT and ImageNette results in Table 2, the second with a 40k-step checkpoint-adaptation experiment (Section 4.2), and the third through ablations on three-state reuse, the consistency objective, and the adaptation initialization (Section 4.3). Additional base-model comparisons and extended sweeps are reported in Appendix F.

# 4.1 CoFRe improves the quality-cost trade-off

Experimental setup. For language modeling, we evaluate on OWT [23] with context length 1024, sentence packing, and the GPT-2 tokenizer. We follow the MDLM training setup of Sahoo et al. [57]; CoFRe uses the same data, tokenizer, and objective, but replaces the middle transformer stack with a fixed-point block. We report generative perplexity (Gen. PPL, via GPT-2 Large) as a measure of quality, and unigram entropy as a measure of diversity and uncertainty [57], across fixed transformerblock budgets; see Appendix E for details. Following the allocation ablation in Appendix F.10, we use a decreasing fixed-point iteration schedule.

For image generation, we evaluate on ImageNette [13, 32] at 256×256 resolution. Following Besnier et al. [5], images are tokenized into 16×16=256 latent tokens using the ImageFolder VQ-4096/XQGAN-4096 tokenizer [41, 42]. We compare MaskGIT-Large and CoFRe under the same setup, evaluating FID (realism and alignment with the data distribution) [28] and IS (quality and diversity) [59], as well as latency, training time, and VRAM; further details are in Appendix E.

Results. Table 2 reports the main quality–cost comparison for both modalities. We emphasize that the reported CoFRe numbers correspond to one point on a broader quality–diversity trade-off: Appendix F.9 sweeps the allocation between denoising steps and fixed-point iterations, showing that the strongest CoFRe configurations improve generative perplexity without relying on a degenerate entropy regime. For language, CoFRe improves generative perplexity over MDLM+SDTT at every reported budget, reducing from 193.1 to 101.8 at budget 96 and from 47.0 to 37.8 at budget 768. Beyond these gains, CoFRe also reduces backbone cost: CoFRe uses 93.44 GiB/GPU and approximately 123h of training, compared to 112.4 GiB/GPU and approximately 139h for MDLM before the additional SDTT stage (Figure 1).

The same pattern holds for image generation. CoFRe improves FID over MaskGIT-Large at every reported budget, for example from 174.1 to 96.7 at budget 48 and from 30.0 to 22.8 at budget 384. It also reduces training time from 17h46m to 9h08m and VRAM from 72.45 GiB to 35.74 GiB. Overall, Table 2 shows that fixed-point denoisers improve the quality–cost trade-off across modalities, and that CoFRe turns this efficiency gain into stronger low-budget generation. Further work and additional experiments in Appendix F.

# 4.2 Adapting a pretrained MDLM checkpoint into FP-MDLM

Experimental setup We use a pretrained MDLM teacher from Sahoo et al. [57] trained on OWT with sequence length 1024. The converted FP-MDLM is initialized from this checkpoint as described in Section 3.4, and then adapted on OWT with the same sequence length. During adaptation, we use the KL consistency loss on correlated masked inputs. Then during post-training, the KL coefficient is linearly warmed up from 0 to 0.1 over 5k global steps and then kept constant at 0.1 for the remainder of adaptation. We compare the generative perplexity and unigram entropy of base FP-MDLM and adapted FP-MDLM across different budgets.

Results Figure 4 and Table 9 shows that a pretrained MDLM checkpoint can be converted into a stronger FP-MDLM with only 40k adaptation steps. We first compare the models without reuse in order to isolate the effect of the adaptation on the vanilla fixed-point model itself. In this setting, the adapted checkpoint improves generative perplexity at every budget, from 375.6 to 296.8 at budget 96 and from 179.7 to 149.6 at budget 768, while keeping entropy close to the baseline.

The adaptation also improves the behaviour of reuse. For the baseline FP-MDLM, reuse is inconsistent and can hurt generation quality at larger budgets. After adaptation, however, reuse becomes beneficial in the medium- and high-budget regimes: both full reuse and three-state reuse improve over no reuse at budgets 192, 384, and 768. The strongest results are obtained with three-state reuse, which reaches generative perplexities of 192.2, 149.4, and 131.3 at these budgets. Overall, these results show that pretrained MDLM checkpoints can be turned into effective FP-MDLM generators with a short adaptation stage, and that this adaptation restores the benefit of reuse when the sampling budget is sufficiently large. More details and results in Table 9.

![](images/05d5df90cddffa9cf9a5ede1892e3e091d1b6f41fb090a9dfad50c7419edd1cd.jpg)

<details>
<summary>bar</summary>

| Sampling Budget | MDLM  | FP-MDLM No Reuse | Adapted FP-MDLM No Reuse |
| --------------- | ----- | ---------------- | ------------------------ |
| 96              | 820   | 380              | 300                      |
| 192             | 350   | 270              | 200                      |
| 384             | 200   | 220              | 180                      |
</details>

Figure 4: Short from-scratch adaptation improves FP-MDLM on OWT. Adapted FP-MDLM improves generation quality at every budget tested.

# 4.3 Ablations

We isolate two design choices that are not covered by the main end-to-end results: two main ingredients of CoFRe: three-state reuse and consistency loss, and the pretrained layer initialization used during checkpoint adaptation. Each ablation changes only the component under study while keeping the training or sampling protocol fixed.

# 4.3.1 Three-state reuse

Experimental setup We evaluate inference-time solution reuse by keeping the FP-MDLM checkpoint and the rest of the sampling setup fixed, and varying only the initialization of the fixed-point solver. We compare three settings: no reuse, full reuse, and three-state reuse, and report GPT-2 Large generative perplexity together with sample entropy. We also analyze how the initialization to solved distance compares across the three regimes.

![](images/1a09f70aeaa9c81178155123b161fbe59d1c483f62088ced11a5d7f93f8815ca.jpg)

<details>
<summary>line</summary>

| Budget | No reuse | Full reuse | 3-state reuse |
| ------ | -------- | ---------- | ------------- |
| 96     | 380      | 520        | 450           |
| 192    | 280      | 250        | 250           |
| 384    | 200      | 220        | 200           |
| 768    | 180      | 270        | 280           |
</details>

Figure 5: Effect of different warm-start of the fixed-point on FP-MDLM base (Left) and FP-MDL $\mathbf { \bar { M } } \mathbf { + } \mathcal { L } _ { \mathrm { C O N S } }$ (Right).

Results Figure 5 separates inference-time reuse from consistency post-training. On the base FP-MDLM (left), 3SR improves over full reuse mainly at low and medium budgets, with the gap reducing at larger budgets. Figure 3 explains this behavior: both reuse variants reduce the initialization-tosolution distance, but full reuse treats all tokens uniformly, including newly revealed tokens whose fixed-point states change the most. In contrast, 3SR weakens reuse for newly revealed tokens, partially reuses still-masked tokens, and strongly reuses stable visible tokens.

With LCONS (right), the same trend holds, but all methods achieve much lower generative perplexity. Thus, LCONS and 3SR are complementary: consistency improves prediction quality, while 3SR provides better token-aware solver initialization. Additional results in Appendices F.1, F.10, and C.2.

# 4.3.2 Adaptation initialization.

Experimental setup We compare two 40k-step FP-MDLM adaptation runs on OWT with sequence length 1024. The Initialized model is initialized from a pretrained MDLM checkpoint using the layer-mapping procedure described in Section 3.4, while the Not initialized model uses the same FP-MDLM architecture and adaptation objective but does not use this pretrained layer initialization. Both models are adapted with the same frozen teacher, correlated-mask KL objective, optimizer and sampling settings.

Results Tables 3 and 8 show that pretrained initialization improves short adaptation. Without reuse, generative perplexity drops from 296.8 to 276.0 at budget 96 and from 149.6 to 139.0 at budget 768. With 3SR, initialization is consistently better from budget 192 onward, reaching 126.9 at budget 768. The initialized run also trains faster (Figure 8), showing that the layer mapping provides a better starting point.

Table 3: Initializing from a pretrained MDLM improves FP-MDLM adaptation with 3SR. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="2">No init</td><td>Gen PPL ↓</td><td>298.643</td><td>192.227</td><td>149.427</td><td>131.291</td></tr><tr><td>Entropy ↑</td><td>5.722</td><td>5.658</td><td>5.611</td><td>5.597</td></tr><tr><td rowspan="2">Init</td><td>Gen PPL ↓</td><td>286.403</td><td>184.708</td><td>147.497</td><td>126.872</td></tr><tr><td>Entropy ↑</td><td>5.695</td><td>5.649</td><td>5.619</td><td>5.572</td></tr></table>

# 5 Related work

Efficient training of MGMs PUMA [37] aligns training with inference-time unmasking patterns. DiffuGPT/DiffuLLaMA [24] and Dream [64] adapt pretrained autoregressive models into bidirectional diffusion models, rather than training from scratch. We reduce training cost via a weight-shared fixed-point solver, and show that pre-trained MGMs can be adapted into FP-MGMs.

Efficient few-step generation with MGMs Unmasking schedules and token-order policies select which positions to update at each step [5, 30, 33, 36, 44, 51]. Discrete solvers and timestep schedules reduce the number of sampling steps [18, 46, 50, 56]. Distillation compresses many-step teachers into few-step students [15, 31, 40, 58, 70]. Self-speculative decoding produces non-factorized predictions over masked positions in a single forward pass by drafting and validating tokens [6]. PGM [16] removes explicit mask tokens to improve the throughput during sampling. CDLM [38] reduces the number of sampling steps via a consistency objective and uses block-wise causal attention to enable KV caching. We instead keep bidirectional attention, reducing the per-step cost via a weight-shared fixed-point solver, and add cross-step consistency and three-state reuse.

Implicit depth and looped models Universal [12] and looped Transformers [34, 53, 66] repeatedly apply shared blocks for adaptive depth. DEQ [3] solve for the fixed point of a shared layer via implicit differentiation, and the Generative Equilibrium Transformer [21] uses a DEQ for onestep diffusion distillation. In autoregressive LLMs, Mixture-of-Recursions and Relaxed Recursive Transformers [1, 2] use different depths for different tokens. In continuous diffusion, Fixed-Point Diffusion Models [4] combine an implicit solver with state reuse. The implicit-depth methods above apply to continuous diffusion or autoregressive language models. MGMs differ because the local conditioning of each position changes non-uniformly across denoising steps in one step, in an arbitrary order. We add two mechanisms to the fixed-point solver. (1) The three-state reuse rule handles clean, masked, and newly decoded tokens differently. (2) The cross-step consistency behaves like self-distillation and substantially improves in low-budget generative perplexity.

# 6 Conclusion

We introduce Fixed-Point Masked Generative Models (FP-MGMs), which replace part of the denoising transformer with an implicit weight-sharing block. When applied to MDLM and MaskGIT, FP-MGMs reduce parameters, training time, and memory while improving performance on lowbudget generation.The fixed-point architecture provides the efficiency gains, but effective low-budget generation also requires stabilizing training and reuse across denoising steps. Cross-step consistency drives low-budget generation quality, while three-state reuse enables token-aware warm starts; together, they make CoFRe a complete training-to-inference recipe. We also show that pretrained MDLM checkpoints can be converted into a fixed-point model with only short adaptation. Overall, CoFRe offers a practical path toward cheaper and more flexible masked generative models.

# Acknowledgments and Disclosure of Funding

Yiming Qin and Alba Carballo-Castro were supported by the Swiss National Science Foundation (SNSF grant 10001445). Justin Deschenaux has received funding from the Swiss State Secretariat for Education, Research and Innovation (SERI).

# References

[1] Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/ abs/2410.20672. 9   
[2] Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, and Se-Young Yun. Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/ 2507.10524. 9   
[3] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep Equilibrium Models. Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/ 1909.01377v2. 2, 4, 9   
[4] Xingjian Bai and Luke Melas-Kyriazi. Fixed Point Diffusion Models. Conference on Computer Vision and Pattern Recognition (CVPR), 2024. URL http://arxiv.org/abs/2401.08741. 2, 4, 5, 9, 19, 24   
[5] Victor Besnier, Mickael Chen, David Hurych, Eduardo Valle, and Matthieu Cord. Halton Scheduler For Masked Generative Image Transformer. International Conference on Learning Representations (ICLR), 2025. URL http://arxiv.org/abs/2503.17076. 2, 4, 7, 9, 25, 26   
[6] Andrew Campbell, Valentin De Bortoli, Jiaxin Shi, and Arnaud Doucet. Self-Speculative Masked Diffusions. International Conference on Learning Representations (ICLR), 2026. URL http://arxiv.org/abs/2510.03929. 9   
[7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. MaskGIT: Masked Generative Image Transformer. Conference on Computer Vision and Pattern Recognition (CVPR), 2022. URL http://arxiv.org/abs/2202.04200. 2, 4   
[8] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling. arXiv preprint arXiv:1312.3005, 2014. URL http://arxiv.org/abs/1312.3005. 25   
[9] Sitong Chen, Shen Nie, Jiacheng Sun, Zijin Feng, Zhenguo Li, Ji-Rong Wen, and Chongxuan Li. Masked Diffusion Models as Energy Minimization. Advances in Neural Information Processing Systems (NeurIPS), 2025. URL http://arxiv.org/abs/2509.13866. 2

[10] Marco Comunità, Zhi Zhong, Akira Takahashi, Shiqi Yang, Mengjie Zhao, Koichi Saito, Yukara Ikemiya, Takashi Shibuya, Shusuke Takahashi, and Yuki Mitsufuji. SpecMaskGIT: Masked Generative Modeling of Audio Spectrograms for Efficient Audio Synthesis and Beyond. arXiv preprint arXiv:2406.17672, 2024. URL http://arxiv.org/abs/2406.17672. 2   
[11] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision Transformers Need Registers. International Conference on Learning Representations (ICLR), 2024. URL http://arxiv.org/abs/2309.16588. 26   
[12] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal Transformers. International Conference on Learning Representations (ICLR), 2019. URL http://arxiv.org/abs/1807.03819. 2, 9   
[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. Conference on Computer Vision and Pattern Recognition (CVPR), 2009. URL https://ieeexplore.ieee.org/document/5206848. 3, 7   
[14] Justin Deschenaux and Caglar Gulcehre. Promises, outlooks and challenges of Diffusion Language Modeling. arXiv preprint arXiv:2406.11473, 2024. URL https://arxiv.org/ abs/2406.11473. 2, 25   
[15] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast LLMs via Self-Distillation Through Time. International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2410.21035. 3, 9, 26, 32, 38   
[16] Justin Deschenaux, Lan Tran, and Caglar Gulcehre. Partition Generative Modeling: Masked Modeling Without Masks. International Conference on Learning Representations (ICLR), 2026. URL http://arxiv.org/abs/2505.18883. 2, 9, 27, 31, 32   
[17] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Rémi Leblond, Will Grathwohl, and Jonas Adler. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022. URL https://arxiv.org/abs/2211. 15089. 27   
[18] Alberto Foresti, Mustapha Bounoua, Giulio Franzese, Luca Ambrogioni, and Pietro Michiardi. Improved Sampling Schedules for Discrete Diffusion Models. arXiv preprint arXiv:2602.06849, 2026. URL http://arxiv.org/abs/2602.06849. 9   
[19] Samy Wu Fung, Howard Heaton, Qiuwei Li, Daniel McKenzie, Stanley Osher, and Wotao Yin. JFB: Jacobian-Free Backpropagation for Implicit Networks. Association for the Advancement of Artificial Intelligence (AAAI), 2022. URL https://arxiv.org/abs/2103.12803v4. 19   
[20] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602. 3, 25   
[21] Zhengyang Geng, Ashwini Pokle, and J. Zico Kolter. One-Step Diffusion Distillation via Deep Equilibrium Models. Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2401.08639. 9   
[22] Angeliki Giannou, Shashank Rajput, Jy yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2301.13196. 2   
[23] Aaron Gokaslan and Vanya Cohen. OpenWebText corpus. http://Skylion007.github. io/OpenWebTextCorpus, 2019. 3, 7   
[24] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling Diffusion Language Models via Adaptation from Autoregressive Models. International Conference on Learning Representations (ICLR), 2025. URL http://arxiv.org/abs/2410.17891. 9

[25] J. H. Halton. Algorithm 247: Radical-inverse quasi-random point sequence. Communications of the ACM, 7(12):701–702, 1964. ISSN 0001-0782. doi: 10.1145/355588.365104. URL https://doi.org/10.1145/355588.365104. 4, 26   
[26] Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Demystifying MaskGIT Sampler and Beyond: Adaptive Order Selection in Masked Diffusion. Transactions on Machine Learning Research, 2026. URL http://arxiv.org/abs/2510. 04525. 2   
[27] Andre He, Sean Welleck, and Daniel Fried. Reasoning with Latent Tokens in Diffusion Language Models. arXiv preprint arXiv:2602.03769, 2026. URL http://arxiv.org/abs/ 2602.03769. 2   
[28] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/ 1706.08500. 7   
[29] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. International Conference on Learning Representations (ICLR), 2020. URL https://openreview.net/forum?id=rygGQyrFvH. 31   
[30] Chunsan Hong, Seonho An, Min-Soo Kim, and Jong Chul Ye. Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies. International Conference on Learning Representations (ICLR), 2026. URL http://arxiv.org/abs/2510.05725. 9   
[31] Emiel Hoogeboom, David Ruhe, Jonathan Heek, Thomas Mensink, and Tim Salimans. Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD. arXiv preprint arXiv:2603.20155, 2026. URL https://arxiv.org/abs/2603.20155. 9   
[32] Jeremy Howard. Imagenette: A smaller subset of 10 easily classified classes from ImageNet. https://github.com/fastai/imagenette, 2019. 3, 7   
[33] Metod Jazbec, Theo X. Olausson, Louis Béthune, Pierre Ablin, Michael Kirchhof, João Monteiro, Victor Turrisi, Jason Ramapuram, and Marco Cuturi. Learning Unmasking Policies for Diffusion Language Models. arXiv preprint arXiv:2512.09106, 2026. URL http://arxiv.org/abs/2512.09106. 9   
[34] Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation. International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2602.11451. 9   
[35] Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks. arXiv preprint arXiv:2510.04871, 2025. URL https://arxiv.org/abs/2510.04871. 2   
[36] Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen. Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions. International Conference on Machine Learning (ICML), 2025. URL http://arxiv.org/abs/2502.06768. 9   
[37] Jaeyeon Kim, Jonathan Geuter, David Alvarez-Melis, Sham Kakade, and Sitan Chen. Stop Training for the Worst: Progressive Unmasking Accelerates Masked Diffusion Training. arXiv preprint arXiv:2602.10314, 2026. URL http://arxiv.org/abs/2602.10314. 9   
[38] Minseo Kim, Chenfeng Xu, Coleman Hooper, Harman Singh, Ben Athiwaratkun, Ce Zhang, Kurt Keutzer, and Amir Gholami. CDLM: Consistency Diffusion Language Models for Faster Sampling. Conference on Machine Learning and Systems (MLSys), 2026. URL https: //arxiv.org/abs/2511.19269. 9   
[39] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. International Conference on Machine Learning (ICML), 2019. URL https://arxiv.org/abs/1905.00414. 22

[40] David Li, Nikita Gushchin, Dmitry Abulkhanov, Eric Moulines, Ivan Oseledets, Maxim Panov, and Alexander Korotin. IDLM: Inverse-distilled Diffusion Language Models. arXiv preprint arXiv:2602.19066, 2026. URL https://arxiv.org/abs/2602.19066. 9, 32   
[41] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024. URL https://arxiv.org/abs/2410.01756. 7, 25   
[42] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Jindong Wang, Zhe Lin, and Bhiksha Raj. XQ-GAN: An open-source image tokenization framework for autoregressive generation. arXiv preprint arXiv:2412.01762, 2024. URL https://arxiv.org/abs/2412.01762. 7, 25   
[43] Lang Liu, Krishna Pillutla, Sean Welleck, Sewoong Oh, Yejin Choi, and Zaid Harchaoui. Divergence frontiers for generative models: Sample complexity, quantization effects, and frontier integrals. Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://openreview.net/forum?id=Z\_J5bCb4Rra. 30   
[44] Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stärk, Yilun Xu, Tommi Jaakkola, and Rafael Gómez-Bombarelli. Think While You Generate: Discrete Diffusion with Planned Denoising. International Conference on Learning Representations (ICLR), 2025. URL https: //arxiv.org/abs/2410.06264. 9   
[45] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution. International Conference on Machine Learning (ICML), 2024. URL http://arxiv.org/abs/2310.16834. 26   
[46] Omer Luxembourg, Haim Permuter, and Eliya Nachmani. Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models. arXiv preprint arXiv:2506.19037, 2025. URL http://arxiv.org/abs/2506.19037. 9   
[47] David Mizrahi, Roman Bachmann, Oguzhan Fatih Kar, Teresa Yeo, Mingfei Gao, Afshin ˘ Dehghan, and Amir Zamir. 4M: Massively Multimodal Masked Modeling. Advances in Neural Information Processing Systems (NeurIPS), 2023. URL http://arxiv.org/abs/ 2312.06647. 2   
[48] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up Masked Diffusion Models on Text. International Conference on Learning Representations (ICLR), 2025. URL http://arxiv.org/abs/2410.18514. 25   
[49] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. International Conference on Learning Representations (ICLR), 2025. URL https: //arxiv.org/abs/2406.03736. 2, 3   
[50] Yong-Hyun Park, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, and Yuki Mitsufuji. “Jump Your Steps”: Optimizing Sampling Schedule of Discrete Diffusion Models. International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/ 2410.07761. 9   
[51] Fred Zhangzhi Peng, Zachary Bezemek, Sawan Patel, Jarrid Rector-Brooks, Sherwood Yao, Avishek Joey Bose, Alexander Tong, and Pranam Chatterjee. Path Planning for Masked Diffusion Model Sampling. arXiv preprint arXiv:2502.03540, 2025. URL http://arxiv. org/abs/2502.03540. 2, 9   
[52] Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://openreview.net/forum?id=Tqx7nJp7PR. 30, 31   
[53] Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, and Daniel Y. Fu. Parcae: Scaling laws for stable looped language models. arXiv preprint arXiv:2604.12946, 2026. URL https: //arxiv.org/abs/2604.12946. 2, 9, 25

[54] Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang. Generative frontiers: Why evaluation matters for diffusion language models. arXiv preprint arXiv:2604.02718, 2026. URL https://arxiv. org/abs/2604.02718. 44   
[55] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019. URL https://openai.com/ blog/better-language-models/. 26   
[56] Yinuo Ren, Haoxuan Chen, Yuchen Zhu, Wei Guo, Yongxin Chen, Grant M. Rotskoff, Molei Tao, and Lexing Ying. Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms. Advances in Neural Information Processing Systems (NeurIPS), 2025. URL http://arxiv.org/abs/2502.00234. 9   
[57] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and Effective Masked Diffusion Language Models. Advances in Neural Information Processing Systems (NeurIPS), 2024. URL http://arxiv.org/abs/2406.07524. 2, 3, 7, 8, 25, 26, 44   
[58] Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The Diffusion Duality. International Conference on Machine Learning (ICML), 2025. URL http://arxiv.org/abs/2506.10892. 9   
[59] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in Neural Information Processing Systems (NeurIPS), 2016. URL https://arxiv.org/abs/1606.03498. 7   
[60] Ivan Sedykh, Nikita Sorokin, and Valentin Malykh. Not All Denoising Steps Are Equal: Model Scheduling for Faster Masked Diffusion Language Models. arXiv preprint arXiv:2604.02340, 2026. URL http://arxiv.org/abs/2604.02340. 2   
[61] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and Generalized Masked Diffusion for Discrete Data. Advances in Neural Information Processing Systems (NeurIPS), 2024. URL http://arxiv.org/abs/2406.04329. 2, 3   
[62] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable Length Video Generation From Open Domain Textual Description. arXiv preprint arXiv:2210.02399, 2022. URL http://arxiv.org/abs/2210.02399. 2   
[63] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2503.00307. 2, 30, 31   
[64] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Diffusion Large Language Models. arXiv preprint arXiv:2508.15487, 2025. URL http://arxiv.org/abs/2508.15487. 9   
[65] Zebin You, Jingyang Ou, Xiaolu Zhang, Jun Hu, Jun Zhou, and Chongxuan Li. Effective and Efficient Masked Image Generation Models. International Conference on Machine Learning (ICML), 2025. URL http://arxiv.org/abs/2503.07197. 2   
[66] Chengting Yu, Xiaobo Shu, Yadao Wang, Yizhen Zhang, Haoyi Wu, You Wu, Rujiao Long, Ziheng Chen, Yuchi Xu, Wenbo Su, and Bo Zheng. SpiralFormer: Looped Transformers Can Learn Hierarchical Dependencies via Multi-Resolution Recursion. arXiv preprint arXiv:2602.11698, 2026. URL http://arxiv.org/abs/2602.11698. 9   
[67] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. MAGVIT: Masked Generative Video Transformer. Conference on Computer Vision and Pattern Recognition (CVPR), 2023. URL https://openaccess.thecvf.com/content/CVPR2023/papers/ Yu\_MAGVIT\_Masked\_Generative\_Video\_Transformer\_CVPR\_2023\_paper.pdf. 2

[68] Shuibai Zhang, Caspian Zhuang, Chihan Cui, Zhihan Yang, Fred Zhangzhi Peng, Yanxin Zhang, Haoyue Bai, Zack Jia, Yang Zhou, Guanhua Chen, and Ming Liu. Expert-choice routing enables adaptive computation in diffusion language models. arxiv preprint arXiv:2604.01622, 2026. URL https://arxiv.org/abs/2604.01622. 2   
[69] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling. International Conference on Learning Representations (ICLR), 2025. URL http://arxiv.org/abs/2409.02908. 26, 27, 31   
[70] Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, and Vicky Kalogeiton. Di\$\mathtt{[M]}\$O: Distilling Masked Diffusion Models into One-step Generator. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. URL http://arxiv.org/abs/ 2503.15457. 9

# Contents

1 Introduction 1   
2 Background 3

2.1 Discrete generative models and masked generative models 3   
2.2 Deep equilibrium models, and fixed-point diffusion models 4

3 Fixed-Point Denoising Networks for Masked Sequence Modeling 4

3.1 Fixed-point MGMs 4   
3.2 Training a FP-MGM 5   
3.3 Sampling with three-state reuse . . . 6   
3.4 Pretrained model conversion and short adaptation . . . 6

4 Experiments 7

4.1 CoFRe improves the quality-cost trade-off . . .   
4.2 Adapting a pretrained MDLM checkpoint into FP-MDLM 8   
4.3 Ablations 8

4.3.1 Three-state reuse 8   
4.3.2 Adaptation initialization. . . 9

5 Related work 9

6 Conclusion 10

A Limitations and future work 18   
B Ethics statement 18   
C Additional Method Details 18

C.1 Stochastic Jacobian-Free Backpropagation . . . 18   
C.2 Three-State Reuse Details . 19   
C.3 Pretrained MDLM Conversion Details 22

C.3.1 Analysis of the pretrained MDLM checkpoint . . . . 22   
C.3.2 Layer mapping and initialization . . . 23   
C.3.3 Correlated mask construction 23   
C.3.4 Effect on the training loss when using or not initialization 24

D Experimental Details 24

D.1 Hyperparameter Tuning Protocol 24   
D.1.1 Fixed-Point MGM hyperparameters . . 24   
D.1.2 Learning-rate and solver tuning 25   
D.2 Language Modeling Setup 25

D.2.1 OpenWebText . 25   
D.2.2 Downstream evaluation . . . 25

D.3 Image Modeling Setup 25   
D.4 Sampling Precision . . . 26   
D.5 Training Costs and Resources . . 26

# E Metrics details 26

E.1 Generative perplexity 26   
E.2 Unigram Entropy 26   
E.3 Fréchet Inception Distance and Inception Score 27   
E.4 Throughput and Latency 27   
E.5 Training time and VRAM used 27

# F Additional Results 27

F.1 Extended OWT Generation Results . . . 27

F.1.1 Samples . . . 28   
F.1.2 MDLM vs FP-MDLM 28   
F.1.3 Initialized vs not initialized FP-MDLM adaptation 29   
F.1.4 Generation quality when adding different components of CoFRe . . . . . . 30   
F.1.5 Mauve score evaluation . . . 30   
F.1.6 Downstream evaluation . . . 31   
F.1.7 Sampling with nucleus sampling . . . . 31

F.2 Using consistency loss on MDLM 32   
F.3 Comparison against other distilled or accelerated diffusion language models . . . . 32   
F.4 Extended Image Modeling Results 34

F.4.1 Samples . . . . 34   
F.4.2 MaskGIT-Large vs MaskGIT-12 vs FP-MaskGIT 34   
F.4.3 Generation quality when adding different components of CoFRe . . . . . . 35

F.5 Component Ablation for FP-MGM 35   
F.6 Cross-step consistency recovers low-budget generation quality 37   
F.7 Consistency Loss Training Dynamics 38   
F.8 Ablations on the Consistency Loss Type 39   
F.9 Tradeoff Between Denoising Steps and Fixed-Point Iterations . . . . . 41   
F.10 Different budget allocation strategies . 42   
F.11 Latency and generation quality for language modeling 42   
F.12 Fixed-point residual analysis. . . . 43   
F.13 Effect of Training Duration on Generation Quality . . . 44

# A Limitations and future work

While this work advances masked generative modeling by introducing fixed-point denoisers, crossstep consistency regularization, and three-state reuse, several limitations remain. We discuss these limitations below, both to clarify the scope of our current results and to highlight directions for future research.

Scale and scope. Our experiments are limited to OWT-scale language modeling and ImageNette image generation. These settings allow controlled comparisons, but do not yet show whether FP-MGMs scale to larger language models, larger image datasets, or multimodal generation. Evaluating FP-MGMs at larger model and data scales is an important direction for future work.

Additional tuning and adaptation. FP-MGMs introduce extra design choices, including where to place the fixed-point block, how many solver iterations to use, how to set reuse coefficients, and how to allocate the sampling budget across denoising steps. We ablate these choices, but the current recipe remains partly heuristic. In addition, the best FP-MDLM results require a short consistency post-training stage, whose duration must be chosen carefully to avoid over-sharpening and entropy collapse.

Generality and practical speedups. Three-state reuse is designed for monotonic masked decoding, where tokens are revealed and then remain fixed; samplers that remask or revise visible tokens may require different reuse rules. Moreover, our main compute metric is transformer-block forward passes, which is hardware-independent but does not always translate directly into wall-clock gains because fixed-point solvers add control-flow overhead. Future work should develop adaptive stopping rules, optimized implementations, and reuse strategies for more general masked-generation trajectories.

# B Ethics statement

The objective of this work is to improve the efficiency of masked generative models by introducing fixed-point denoisers, cross-step consistency regularization, and three-state reuse. Masked generative models are relevant to a broad range of applications, including language generation, image synthesis, video, audio, and multimodal modeling. Improvements in their training and sampling efficiency may therefore reduce the computational cost of developing and deploying generative models, making such models more accessible to researchers with limited compute resources.

At the same time, FP-MGMs inherit the broader risks of generative models. More efficient generation can lower the cost of producing synthetic text or images, which may amplify existing concerns around misinformation, spam, copyright misuse, or the generation of biased and harmful content. Our experiments are limited to moderate-scale language and image benchmarks, and the generated samples still exhibit failure modes such as repetition, factual inconsistency, and topic drift. As a result, we do not view the current models as directly suitable for high-stakes applications such as medical, legal, or policy decision-making.

Overall, this work primarily contributes an architectural and algorithmic efficiency improvement. We expect its main near-term impact to be methodological, by providing a route toward cheaper training and stronger low-budget masked generation. Future work should evaluate FP-MGMs at larger scales and study how efficiency gains interact with safety, bias, memorization, and misuse risks in practical deployments.

# C Additional Method Details

# C.1 Stochastic Jacobian-Free Backpropagation

Training an FP-MGM requires differentiating through the implicit fixed-point block, whose output at denoising state t is defined as the hidden-state solution

$$
\mathbf {h} _ {t} ^ {\star} = F _ {\theta_ {F}} (\mathbf {h} _ {t} ^ {\star}; \tilde {\mathbf {h}} _ {t}, t),
$$

where $\tilde { \mathbf { h } } _ { t } = G _ { \theta _ { G } } ( \mathbf { h } _ { \mathrm { p r e } , t } )$ is the input-conditioning signal produced by the preceding explicit layers. In principle, one can backpropagate through this equilibrium using implicit differentiation, which gives a gradient involving the inverse Jacobian term

$$
\left(I - \frac {\partial F _ {\theta_ {F}} (\mathbf {h} _ {t} ^ {\star} ; \tilde {\mathbf {h}} _ {t} , t)}{\partial \mathbf {h} _ {t} ^ {\star}}\right) ^ {- 1}.
$$

However, explicitly forming or solving this Jacobian system is computationally expensive and can be unstable at scale. Jacobian-Free Backpropagation (JFB) [19] avoids this cost by first computing an approximate fixed point without storing intermediate activations, and then applying one additional fixed-point iteration with gradients enabled; the backward pass is therefore performed only through this final step, giving an approximate gradient of the form:

$$
\frac {\partial \mathcal {L}}{\partial \theta_ {F}} \approx \frac {\partial \mathcal {L}}{\partial \mathbf {h} _ {t} ^ {\star}} \frac {\partial F _ {\theta_ {F}} (\mathbf {h} _ {t} ^ {\star} ; \tilde {\mathbf {h}} _ {t} , t)}{\partial \theta_ {F}}.
$$

Stochastic Jacobian-Free Backpropagation (S-JFB) [4] generalizes this idea by unrolling a random number of fixed-point iterations during training. At each training step, S-JFB samples two integers $n \sim \mathcal { U } \{ 0 , \dots , N \}$ and $m \sim \mathcal { U } \{ 1 , \ldots \cdot , \bar { M } \}$ . It first performs n fixed-point iterations under a stopgradient/no-gradient context, producing an approximate equilibrium while avoiding the memory cost of storing these intermediate states. It then performs m additional iterations with gradient tracking enabled, and the loss is backpropagated only through these last m unrolled iterations. The hyperparameters N and M therefore control the maximum number of fixed-point iterations used without and with gradients, respectively. Compared with standard one-step JFB, S-JFB is slightly more expensive because it backpropagates through multiple final iterations rather than only one, but it remains much cheaper than full implicit differentiation or fully unrolled explicit networks. Its stochasticity also exposes the model to different approximation depths during training, which makes the fixed-point layer more robust and empirically improves optimization compared with the deterministic one-step JFB baseline.

# C.2 Three-State Reuse Details

Three-state reuse is the inference-time mechanism we propose to warm-start the fixed-point solver across masked denoising steps. Unlike full reuse, which initializes every token from the previous fixed-point solution, 3SR accounts for the fact that token states evolve non-uniformly during sampling: some tokens remain visible and unchanged, some remain masked but receive updated context, and others are newly revealed. This appendix gives the exact token-wise interpolation rule used to initialize the solver, the visible-fraction-dependent coefficient schedule, and the complete sampling procedure.

Three-state reuse schedule. We detail here the schedule used by 3SR. We base this schedule on the results obtained when looking at the distance between $\mathbf { h } _ { t } ^ { \star }$ and $\mathbf { h } _ { t + 1 } ^ { \star }$ , as plotting on Figure 3. We analyze this distance as it shows, for each type of transition, how far the previous fixed-point solution is from the current fixed-point solution, when the initialization is made without reuse. We therefore use the following schedule: Unchanged visible tokens use full reuse, with $\gamma _ { t } = 1 . 0$ . Still-masked tokens use $\gamma _ { t } = \gamma _ { \mathrm { m a s k e d } } = \gamma _ { \mathrm { m a s k , m i n } } + \left( \gamma _ { \mathrm { m a s k , m a x } } - \gamma _ { \mathrm { m a s k , m i n } } \right) v _ { t } .$ , while newly revealed tokens use $\gamma _ { t } = \gamma _ { \mathrm { c h a n g e d } } = 0 . 2$ where $\begin{array} { r } { v _ { t } = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \mathbb { 1 } [ z _ { t } ^ { i } \neq \mathrm { [ M A S K ] } ] } \end{array}$ is the fraction of visible tokens at step t. In our base setting, we use $\gamma _ { \mathrm { m a s k , m i n } } = 0 . 7 5 , \gamma _ { \mathrm { m a s k , m a x } } = 0 . 9 0 , \gamma _ { \mathrm { c h a n g e d } } = 0 . 2$ . We tune these hyperparameters using a grid search.

To select the reuse coefficients, we run a sweep over the masked-token and newly revealed-token interpolation parameters. Tables 4 and 5 report these sweeps across sampling budgets. Table 4 varies the reuse range for still-masked tokens and the maximum reuse assigned to newly revealed tokens, while Table 5 fixes the masked-token range and varies the newly revealed-token reuse coefficient more finely. Across these sweeps, performance is relatively robust within a moderate range of coefficients, supporting the use of a simple hand-tuned 3SR schedule rather than a learned or highly budget-specific policy.

Algorithm 1 FP-MGM sampling with three-state reuse   
Require: Schedule $1 = \tau_{T} > \cdots > \tau_{0} = 0$ , initial state $z_{\tau_{T}} = [MASK]^{d}$ Require: Preprocessing stack $P_{\theta_{P}}$ , input-conditioning projection $G_{\theta_{G}}$ , fixed-point block $F_{\theta_{F}}$ , post-processing stack $H_{\theta_{H}}$ Require: Per-step solver iterations $N_{i}$ Require: Reuse parameters $\gamma_{mask,min}, \gamma_{mask,max}, \gamma_{changed}$ 1: $h_{prev}^{\star} \leftarrow \emptyset$ 2: $z_{prev} \leftarrow \emptyset$ 3: for $i = T, T - 1, \ldots, 1$ do

4: $h_{pre,\tau_{i}} \leftarrow P_{\theta_{P}}(z_{\tau_{i}}, \tau_{i})$ 5: $\tilde{h}_{\tau_{i}} \leftarrow G_{\theta_{G}}(h_{pre,\tau_{i}})$ 6: if $h_{prev}^{\star} = \emptyset$ then

7: $h_{\tau_{i}}^{0} \leftarrow h_{pre,\tau_{i}}$ 8: else

9: $v_{\tau_{i}} \leftarrow \frac{1}{d} \sum_{j=1}^{d} 1[z_{\tau_{i}}^{j} \neq [MASK]]$ 10: $\gamma_{mask} \leftarrow \gamma_{mask,min} + (\gamma_{mask,max} - \gamma_{mask,min})v_{\tau_{i}}$ 11: Define token-wise reuse coefficients $\gamma_{\tau_{i}}^{j}$ as $\gamma_{\tau_{i}}^{j} = \begin{cases} 1, & \text{if } z_{\tau_{i}}^{j} = z_{\text{prev}}^{j} \neq [\text{MASK}], \\ \gamma_{\text{mask}}, & \text{if } z_{\tau_{i}}^{j} = z_{\text{prev}}^{j} = [\text{MASK}], \\ \gamma_{\text{changed}}, & \text{otherwise}. \end{cases}$ 12: $h_{\tau_{i}}^{0} \leftarrow \gamma_{\tau_{i}} \odot h_{\text{prev}}^{\star} + (1 - \gamma_{\tau_{i}}) \odot h_{\text{pre},\tau_{i}}$ 13: end if

14: for $n = 0, \ldots, N_{i} - 1$ do

15: $h_{\tau_{i}}^{n+1} \leftarrow F_{\theta_{F}}(h_{\tau_{i}}^{n}; \tilde{h}_{\tau_{i}}, \tau_{i})$ 16: end for

17: $h_{\tau_{i}}^{\star} \leftarrow h_{\tau_{i}}^{N_{i}}$ 18: $\ell_{\theta}(z_{\tau_{i}}, \tau_{i}) \leftarrow H_{\theta_{H}}(h_{\tau_{i}}^{\star}, \tau_{i})$ 19: Sample $z_{\tau_{i-1}}$ using the original MGM transition rule and logits $\ell_{\theta}(z_{\tau_{i}}, \tau_{i})$ 20: $h_{prev}^{\star} \leftarrow h_{\tau_{i}}^{\star}$ 21: $z_{prev} \leftarrow z_{\tau_{i}}$ 22: end for

23: return $z_{\tau_{0}}$

Table 4: Results for $\mathtt { f p . }$ \_mdlm with fixed strategy across budgets. Constant settings for all runs: visible $\gamma = 1 . 0 $ , changed $\gamma _ { \mathrm { m i n } } = 0 . 0$ . Values are shown to four decimal places. 

<table><tr><td rowspan="2">Masked γ range</td><td rowspan="2">Changed γmax</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="6">[0.60, 0.90]</td><td rowspan="2">0.1</td><td>Gen PPL ↓</td><td>94.1776</td><td>83.0655</td><td>47.4160</td><td>37.0117</td></tr><tr><td>Entropy ↑</td><td>5.5104</td><td>5.4820</td><td>5.3554</td><td>5.2280</td></tr><tr><td rowspan="2">0.2</td><td>Gen PPL ↓</td><td>95.3269</td><td>81.8788</td><td>47.2272</td><td>36.9850</td></tr><tr><td>Entropy ↑</td><td>5.5110</td><td>5.4766</td><td>5.3562</td><td>5.2260</td></tr><tr><td rowspan="2">0.3</td><td>Gen PPL ↓</td><td>96.1154</td><td>81.8405</td><td>46.8567</td><td>36.9900</td></tr><tr><td>Entropy ↑</td><td>5.5181</td><td>5.4817</td><td>5.3548</td><td>5.2394</td></tr><tr><td rowspan="6">[0.60, 0.95]</td><td rowspan="2">0.1</td><td>Gen PPL ↓</td><td>94.7594</td><td>82.7298</td><td>47.1981</td><td>36.7854</td></tr><tr><td>Entropy ↑</td><td>5.5115</td><td>5.4771</td><td>5.3496</td><td>5.2242</td></tr><tr><td rowspan="2">0.2</td><td>Gen PPL ↓</td><td>95.0930</td><td>81.3275</td><td>47.4469</td><td>37.2802</td></tr><tr><td>Entropy ↑</td><td>5.5126</td><td>5.4770</td><td>5.3526</td><td>5.2504</td></tr><tr><td rowspan="2">0.3</td><td>Gen PPL ↓</td><td>95.4836</td><td>83.2949</td><td>47.0793</td><td>36.8499</td></tr><tr><td>Entropy ↑</td><td>5.5122</td><td>5.4893</td><td>5.3624</td><td>5.2257</td></tr><tr><td rowspan="6">[0.75, 0.90]</td><td rowspan="2">0.1</td><td>Gen PPL ↓</td><td>97.2015</td><td>81.5204</td><td>45.9489</td><td>36.8647</td></tr><tr><td>Entropy ↑</td><td>5.5201</td><td>5.4752</td><td>5.3500</td><td>5.2429</td></tr><tr><td rowspan="2">0.2</td><td>Gen PPL ↓</td><td>97.8571</td><td>80.2788</td><td>46.4174</td><td>36.0494</td></tr><tr><td>Entropy ↑</td><td>5.5267</td><td>5.4656</td><td>5.3382</td><td>5.2489</td></tr><tr><td rowspan="2">0.3</td><td>Gen PPL ↓</td><td>96.8045</td><td>80.8193</td><td>46.2536</td><td>36.4045</td></tr><tr><td>Entropy ↑</td><td>5.5187</td><td>5.4681</td><td>5.3523</td><td>5.2345</td></tr><tr><td rowspan="6">[0.75, 0.95]</td><td rowspan="2">0.1</td><td>Gen PPL ↓</td><td>98.9172</td><td>80.3999</td><td>46.2176</td><td>36.8800</td></tr><tr><td>Entropy ↑</td><td>5.5264</td><td>5.4639</td><td>5.3496</td><td>5.2355</td></tr><tr><td rowspan="2">0.2</td><td>Gen PPL ↓</td><td>98.4236</td><td>81.4000</td><td>46.7018</td><td>37.2678</td></tr><tr><td>Entropy ↑</td><td>5.5262</td><td>5.4768</td><td>5.3453</td><td>5.2457</td></tr><tr><td rowspan="2">0.3</td><td>Gen PPL ↓</td><td>97.5960</td><td>80.5485</td><td>46.8595</td><td>36.5579</td></tr><tr><td>Entropy ↑</td><td>5.5210</td><td>5.4687</td><td>5.3531</td><td>5.2423</td></tr><tr><td rowspan="6">[0.85, 0.90]</td><td rowspan="2">0.1</td><td>Gen PPL ↓</td><td>98.6561</td><td>80.5434</td><td>47.7492</td><td>36.8783</td></tr><tr><td>Entropy ↑</td><td>5.5192</td><td>5.4795</td><td>5.3718</td><td>5.2484</td></tr><tr><td rowspan="2">0.2</td><td>Gen PPL ↓</td><td>99.3588</td><td>79.7671</td><td>47.5099</td><td>36.3204</td></tr><tr><td>Entropy ↑</td><td>5.5297</td><td>5.4734</td><td>5.3559</td><td>5.2056</td></tr><tr><td rowspan="2">0.3</td><td>Gen PPL ↓</td><td>98.3399</td><td>79.6512</td><td>47.7224</td><td>37.3283</td></tr><tr><td>Entropy ↑</td><td>5.5244</td><td>5.4711</td><td>5.3672</td><td>5.2439</td></tr><tr><td rowspan="6">[0.85, 0.95]</td><td rowspan="2">0.1</td><td>Gen PPL ↓</td><td>98.4593</td><td>79.7479</td><td>47.3798</td><td>37.1501</td></tr><tr><td>Entropy ↑</td><td>5.5246</td><td>5.4711</td><td>5.3686</td><td>5.2551</td></tr><tr><td rowspan="2">0.2</td><td>Gen PPL ↓</td><td>99.1255</td><td>78.1578</td><td>46.5278</td><td>36.6537</td></tr><tr><td>Entropy ↑</td><td>5.5231</td><td>5.4590</td><td>5.3563</td><td>5.2321</td></tr><tr><td rowspan="2">0.3</td><td>Gen PPL ↓</td><td>98.2182</td><td>79.6038</td><td>47.8894</td><td>36.5796</td></tr><tr><td>Entropy ↑</td><td>5.5240</td><td>5.4698</td><td>5.3720</td><td>5.2358</td></tr></table>

Table 5: Results for $\mathtt { f p . }$ \_mdlm with fixed strategy for masked $\gamma$ range [0.75, 0.90]. Constant settings for all runs: visible $\gamma = 1 . 0$ . Values are shown to four decimal places. 

<table><tr><td rowspan="2">Masked γ range</td><td rowspan="2">γchanged,min</td><td rowspan="2">γchanged,max</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="16">[0.75, 0.90]</td><td rowspan="2">0.00</td><td rowspan="2">0.00</td><td>Gen PPL ↓</td><td>101.4907</td><td>62.5693</td><td>42.3873</td><td>37.8086</td></tr><tr><td>Entropy ↑</td><td>5.4348</td><td>5.3472</td><td>5.1515</td><td>5.1822</td></tr><tr><td rowspan="2">0.10</td><td rowspan="2">0.10</td><td>Gen PPL ↓</td><td>100.9282</td><td>62.0509</td><td>41.9018</td><td>37.2647</td></tr><tr><td>Entropy ↑</td><td>5.4300</td><td>5.3373</td><td>5.1356</td><td>5.1530</td></tr><tr><td rowspan="2">0.20</td><td rowspan="2">0.20</td><td>Gen PPL ↓</td><td>101.1447</td><td>60.5738</td><td>40.7417</td><td>37.1662</td></tr><tr><td>Entropy ↑</td><td>5.4351</td><td>5.3447</td><td>5.0733</td><td>5.1485</td></tr><tr><td rowspan="2">0.25</td><td rowspan="2">0.25</td><td>Gen PPL ↓</td><td>99.7740</td><td>61.9484</td><td>41.2334</td><td>37.7585</td></tr><tr><td>Entropy ↑</td><td>5.4340</td><td>5.3493</td><td>5.1347</td><td>5.1718</td></tr><tr><td rowspan="2">0.30</td><td rowspan="2">0.30</td><td>Gen PPL ↓</td><td>100.9554</td><td>61.3730</td><td>42.3543</td><td>36.9675</td></tr><tr><td>Entropy ↑</td><td>5.4388</td><td>5.3411</td><td>5.0945</td><td>5.1459</td></tr><tr><td rowspan="2">0.00</td><td rowspan="2">0.10</td><td>Gen PPL ↓</td><td>97.2015</td><td>81.5204</td><td>45.9489</td><td>36.8647</td></tr><tr><td>Entropy ↑</td><td>5.5201</td><td>5.4752</td><td>5.3500</td><td>5.2429</td></tr><tr><td rowspan="2">0.00</td><td rowspan="2">0.20</td><td>Gen PPL ↓</td><td>97.8571</td><td>80.2788</td><td>46.4174</td><td>36.0494</td></tr><tr><td>Entropy ↑</td><td>5.5267</td><td>5.4656</td><td>5.3382</td><td>5.2489</td></tr><tr><td rowspan="2">0.00</td><td rowspan="2">0.30</td><td>Gen PPL ↓</td><td>96.8045</td><td>80.8193</td><td>46.2536</td><td>36.4045</td></tr><tr><td>Entropy ↑</td><td>5.5187</td><td>5.4681</td><td>5.3523</td><td>5.2345</td></tr></table>

# C.3 Pretrained MDLM Conversion Details

In this part, we give further details and context on how we initialize the FP-MDLM model when we adapt it, as presented in Section 4.2, and why these design choices are motivated.

# C.3.1 Analysis of the pretrained MDLM checkpoint

Layer similarity analysis with CKA. To better understand how to convert a pretrained MDLM into a FP model, we analyze the similarity of hidden representations across transformer layers. We use Linear Centered Kernel Alignment (CKA) [39], a standard representation-similarity measure that compares whether two layers encode examples with similar geometry. For each timestep t, we collect the residual-stream activations of every transformer layer on held-out OWT batches. For layer l, we flatten batch and sequence dimensions to obtain a feature matrix

$$
X _ {l} ^ {(t)} \in \mathbb {R} ^ {N \times d},
$$

where each row corresponds to one token representation. Given two centered feature matrices X and Y , Linear CKA is defined as

$$
\operatorname{CKA} (X, Y) = \frac {\left\| X ^ {\top} Y \right\| _ {F} ^ {2}}{\left\| X ^ {\top} X \right\| _ {F} \left\| Y ^ {\top} Y \right\| _ {F}}.
$$

CKA is close to 1 when two layers induce very similar representation geometry over the same token samples, and close to 0 when their representations are largely unrelated. We therefore use CKA to identify redundant groups of layers and potential boundaries between qualitatively different representation regimes.

![](images/f9cf4f2867d17b76d2aeef5323f7383d8fdd6021e4827133fff54963a6f8f506.jpg)

<details>
<summary>heatmap</summary>

| Layer | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|---|---|---|---|---|---|---|---|---|---|-----|-----|-----|
| L1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| L2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| L3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| L4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| L5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| L6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| L7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| L8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| L9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| L10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| L11 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| L12 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
The chart displays a heatmap with a color scale ranging from -1 to +1, representing the Linear CKA values for each layer in the heatmap cells.
</details>

(a) Mean Linear CKA across timesteps.

![](images/64fcbb8c70afe3dcc46137b65aa86e684c08ac835c517ad31ccc970f6ff22c7e.jpg)

<details>
<summary>line</summary>

| Layer pair index / for (L₁, L₁₊₁) ×10¹ | CKA(L, L₁₊₁) |
| -------------------------------------- | ------------ |
| 0.1                                    | 0.8          |
| 0.2                                    | 0.9          |
| 0.3                                    | 0.92         |
| 0.4                                    | 0.93         |
| 0.5                                    | 0.15         |
| 0.6                                    | 1.0          |
| 0.7                                    | 1.0          |
| 0.8                                    | 1.0          |
| 0.9                                    | 1.0          |
| 1.0                                    | 1.0          |
| 1.1                                    | 1.0          |
</details>

(b) Consecutive-layer CKA.   
Figure 6: Representation similarity in a pretrained MDLM. We compute Linear CKA between residual-stream activations of all transformer layers at timesteps $t \in \{ \bar { 0 . 1 } , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ , then average the similarities across timesteps. The heatmap shows a clear two-stage structure: layers 1–5 form an early block, layers 6–12 form a highly self-similar late block, and cross-block similarity is low. The consecutive-layer plot shows a sharp drop between layers 5 and 6, followed by near-saturated similarity among the later layers.

Results and interpretation. Figure 6 shows that the pretrained MDLM has a pronounced block structure in depth. Averaged across timesteps, Linear CKA identifies the strongest and most stable boundary between layers 5 and 6; the same boundary is recovered at all analyzed timesteps. Layers 1–5 form a moderately coherent early stage, with average within-block similarity 0.808, while layers 6–12 form an extremely tight late stage, with average within-block similarity 0.998. In contrast, similarity across the two blocks is very low, 0.060, indicating that the later layers operate in a representation regime that is strongly separated from the earlier layers. The consecutive-layer profile makes this transition especially visible: similarity increases gradually through the early stack, drops sharply from 0.938 between layers 4 and 5 to 0.162 between layers 5 and 6, and then remains nearly saturated from layer 6 onward. This suggests that the deeper part of the pretrained MDLM is highly redundant and already behaves like repeated refinement in a shared representation space. This provides empirical support for replacing the later transformer stack with a shared fixed-point block.

# C.3.2 Layer mapping and initialization

Motivated by the CKA analysis above, we initialize the converted FP-MDLM by mapping representative layers from the pretrained MDLM into the fixed-point architecture. Since layers 6-12 form a highly coherent late-stage block, we use layer 6 to initialize the shared fixed-point block. We keep the boundary layers explicit by mapping layer 1 to the preprocessing block and layer 12 to the postprocessing block (see Figure 7, Left).

![](images/8b6932f1d35ad6138cf74be1670940b7ad24a2dfe73a3df330e1c26c692591d1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Pretrained MDLM"] --> B["1"]
    B --> C["2"]
    C --> D["..."]
    D --> E["11"]
    E --> F["12"]
    G["FP-MDLM"] --> H["1st Pre"]
    H --> I["6th FP Layer"]
    I --> J["12th Post"]
```
</details>

![](images/701bc00de367d538c57f2235fbca18340d835b176c1096256ae0b2cb5c5895ff.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Pretrained MDLM"] --> B["Teacher"]
    C["FP-MDLM"] --> D["Student"]
    E["x0 WORD WORD WORD WORD WORD WORD"] --> F["Sample ts ~ training time sampler\nSample gap ~ Unif(gmin, gmax)\ntT = max(ts - gap, ε)"]
    G["xs WORD WORD [MASK"] [MASK] [MASK] [MASK] xT_WORD_WORD_WORD["MASK"] WORD["MASK"] xT] --> H["Correlated masks"]
```
</details>

![](images/483435c2ed7669f53f73e64b2c15e6881fa5d98a62cb766e855b024c78fe8805.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher"] --> B["l_T"]
    A --> C["p_T^τ"]
    D["Student"] --> E["l_s"]
    D --> F["p_s^τ"]
    B --> G["Logits"]
    C --> H["Softmax and temperature"]
    F --> I["Logits"]
    G --> J["L_KL(i) = KL(p_T^τ(· | i) || p_s^τ(· | i))"]
    H --> J
    I --> J
```
</details>

Figure 7: Adapting an MDLM checkpoint into FP-MDLM. (Left) FP-MDLM is initialized by mapping layers from a pretrained MDLM checkpoint to the preprocessing, fixed-point, and postprocessing blocks (see Appendix C.3.1). (Right) we then run a short adaptation stage with a teacher–student KL loss on logits, using correlated masks at two nearby noise levels, where the teacher input is less noisy than the student input.

# C.3.3 Correlated mask construction

For both logits-KL adaptation against a teacher checkpoint and the consistency loss ${ \mathcal { L } } _ { \mathrm { C O N S } }$ , we construct the student and cleaner masks in a correlated rather than independent way. We first sample a student noise level $t _ { s } ,$ then obtain a cleaner level $t _ { c }$ by subtracting a random gap $\Delta \sim$ $\mathcal { U } [ \mathrm { g } \bar { \mathrm { a p } } _ { \mathrm { m i n } } , \mathrm { g } \mathrm { a p } _ { \mathrm { m a x } } ]$ . These two times are converted into keep probabilities $\alpha _ { s }$ and $\alpha _ { c } ,$ with $\alpha _ { c } \geq \alpha _ { s }$ . For each token position, we draw a single uniform random value $u _ { i }$ and reuse it for both branches:

$$
z _ {s} ^ {i} = \left\{ \begin{array}{l l} \left[ \text {MASK} \right], & u _ {i} <   1 - \alpha_ {s}, \\ x ^ {i}, & \text {otherwise}, \end{array} \right. \quad z _ {c} ^ {i} = \left\{ \begin{array}{l l} \left[ \text {MASK} \right], & u _ {i} <   1 - \alpha_ {c}, \\ x ^ {i}, & \text {otherwise}. \end{array} \right.
$$

Because the same random numbers are reused, the two masks are nested: the cleaner mask is a subset of the student mask. Thus, the cleaner branch always sees an equally clean or cleaner context. This avoids extra variance from unrelated masking patterns and makes the consistency comparison focus on a controlled change in corruption level. In the rare case where both masks would otherwise be identical, we unmask one cleaner position so that the cleaner branch is strictly less noisy. The consistency term is evaluated only on positions masked in the student input.

# C.3.4 Effect on the training loss when using or not initialization

![](images/5d461ebbafce579f0a3d8de6986118340510025e34356dab66e0ddc04c03f086.jpg)

<details>
<summary>line</summary>

| Global step (x10^4) | No Initialization | Initialization with Pretrained MDLM weights |
| ------------------- | ----------------- | ------------------------------------------ |
| 0.0                 | 1.0               | 0.6                                        |
| 0.5                 | 0.5               | 0.45                                       |
| 1.0                 | 0.48              | 0.43                                       |
| 1.5                 | 0.47              | 0.42                                       |
| 2.0                 | 0.46              | 0.41                                       |
| 2.5                 | 0.45              | 0.40                                       |
| 3.0                 | 0.44              | 0.39                                       |
| 3.5                 | 0.43              | 0.38                                       |
| 4.0                 | 0.42              | 0.37                                       |
</details>

Figure 8: Comparison of the training loss between no initialization and initialization with pretrained MDLM weights when distilling the MDLM model into a FP model.

Figure 8 shows that initializing the FP model from pretrained MDLM weights provides a better starting point for short adaptation. The initialized run starts with a substantially lower training loss and remains below the non-initialized run throughout training. Both runs eventually approach similar loss values, but pretrained initialization reaches the low-loss regime much faster, which supports using layer mapping before the KL adaptation stage.

# D Experimental Details

# D.1 Hyperparameter Tuning Protocol

We detail in this section how we tune the majority of the hyperparameters introduced by our methods.

# D.1.1 Fixed-Point MGM hyperparameters

In order to tune the hyperparameters of our fixed-point models, we use a simple staged protocol. For language modeling, all tuning is done on OWT for 100k training steps with sequence length 128, which provides a fast and inexpensive proxy before running the full setup. We first tune the architecture of the fixed-point backbone, namely the number of preprocessing and postprocessing layers, while keeping the solver settings at the default values from Bai and Melas-Kyriazi [4]. We then tune the solver iteration budget, and finally the learning rate. The full procedure is as follows:

1. Choice of tuning setup. We first select the target setting for hyperparameter tuning. For language modeling, we tune on OWT for 100k steps with sequence length 128 in order to obtain fast and cheap comparisons.   
2. Architecture tuning. We tune the number of preprocessing and postprocessing layers in the fixed-point backbone, while keeping the fixed-point solver hyperparameters at the default values of Bai and Melas-Kyriazi [4]. We evaluate a small grid of candidate architectures and retain the best-performing one.   
3. Solver tuning. With the architecture fixed, we tune the fixed-point solver budget, including the number of no-gradient and with-gradient iterations. This isolates the effect of the implicit solver from that of the backbone architecture.   
4. Learning-rate tuning. With both the architecture and solver settings fixed, we tune the learning rate over a logarithmic grid and select the value that gives the best validation performance.   
5. Boundary check. Whenever the best hyperparameter lies at the edge of the tested range, we extend the search range and repeat the evaluation until the selected value is not on the boundary.   
6. Final selection. Finally, we choose the configuration that performs best under this tuning protocol and use it for the full training runs.

# D.1.2 Learning-rate and solver tuning

We tune the main optimization and solver hyperparameters through small-scale experiments before running the full training jobs. We first test the base learning rate used for MDLM. However, with the original learning-rate setting, the FP-MDLM training does not converge, as seen in previous work [53]. We therefore progressively lower the learning rate until the training dynamics become stable and the validation loss shows consistent convergence. We use the same gradient clipping as MDLM, as detailed in [57].

Adaptation hyperparameters For adaptation, we keep the optimizer and model hyperparameters fixed to the base FP-MDLM setup and tune only the adaptation-specific hyperparameters. In particular, we tune the consistency weight λ, the distillation temperature τ , and the teacher-student noise-gap range $[ \mathrm { g a p } _ { \mathrm { m i n } } , \mathrm { g a p } _ { \mathrm { m a x } } ] .$ . In our base adaptation setting, we use logits-KL with $\lambda = 0 . 1 , \tau = 1 . 5$ , $\mathrm { g a p } _ { \mathrm { m i n } } = 0 . 0 5$ , and $\mathrm { g a p } _ { \mathrm { m a x } } = 0 . 3 0$ , apply the loss only on positions masked in the student input, and linearly warm up the consistency weight over the first 5k steps. We select these hyperparameters using a small validation sweep, choosing the setting with the best generative perplexity at matched training compute.

# D.2 Language Modeling Setup

For FP-MDLM, each denoising step applies the explicit preprocessing block, then solves the implicit fixed-point layer, and finally applies the explicit postprocessing block.

We evaluate sampling quality under different compute budgets by sweeping the total forward-pass budget and the number of denoising steps. Unless stated otherwise, we use the ancestral sampler with no additional noise-removal step for MDLM , float64 logits before sampling, and compute sample quality with GPT-2 Large generative perplexity and sample entropy for text. We report both sampling quality and generation speed in tokens per second.

We do not include LM1B [8] because our primary evaluation target is generation quality under limited sampling budgets, rather than likelihood modelling alone. LM1B is most commonly used to report validation/test perplexity, which would mainly evaluate likelihood estimation and would not directly exercise the core advantages of CoFRe: adaptive-depth iterative masked generation and reuse across denoising steps. We therefore focus on settings such as OWT generation, where sample quality and quality–cost trade-offs can be measured more directly.

# D.2.1 OpenWebText

For language modeling, we evaluate on OWT with context length 1024, sentence packing, and the GPT-2 tokenizer. We reserve the last 100k documents for validation. The MDLM baseline follows Sahoo et al. [57]: a 12-layer Diffusion Transformer with hidden size 768, RoPE, dropout 0.1, Adam with learning rate $3 \times \mathrm { 1 \dot { 0 } ^ { - 4 } }$ , global batch size 512, EMA decay 0.9999, and 1M training steps. FP-MDLM uses the same data, tokenizer, and objective, but replaces the middle transformer stack with a fixed-point block. We report generative perplexity using GPT-2 Large and unigram entropy across fixed transformer-block budgets. We use a decreasing order of number of sampling steps as we found that it performs best.

# D.2.2 Downstream evaluation

We evaluate downstream performance with lm-eval-harness [20], following the masked-model evaluation protocol of Deschenaux and Gulcehre [14] and Nie et al. [48]. Because the harness is designed for autoregressive models, we adapt its scoring rule to masked generative models: each answer choice is scored using the variational likelihood bound available for MDLM and FP-MDLM/CoFRe, and the highest-scoring choice is selected.

# D.3 Image Modeling Setup

For image generation, we evaluate on ImageNette at 256 × 256 resolution. Images are center-cropped, resized, and tokenized into a 16 × 16 grid of discrete latent codes, giving a sequence length of 256. We use the ImageFolder VQ-4096/XQGAN-4096 tokenizer [41, 42] and follow the MaskGIT/Halton setup of Besnier et al. [5]. We compare MaskGIT-Large and FP-MaskGIT under the same training and sampling protocol: AdamW optimizer, learning rate $5 \times 1 0 ^ { - 4 }$ , weight decay 0.03, cosine learning-rate schedule, 1500 warmup steps, batch size 128, classifier-free guidance, and Halton sampling. We use Transformer dropout 0.1 and class-label dropout 0.1 for classifier-free guidance. Following Besnier et al. [5], the MaskGIT baseline uses one register [11]. FP-MaskGIT keeps the same tokenizer, masked-token prediction objective, and decoding procedure as MaskGIT-Large, but replaces the middle transformer stack with a fixed-point block. We report FID, IS, training time, and peak VRAM usage.

Halton sampler While effective, Besnier et al. [5] identified a notable drawback: confidenceguided sampling typically leads to spatially clustered token decoding, as the denoiser is inherently more confident near already-populated regions. Because MGMs sample positions independently from the product of marginals $\hat { \Pi } _ { \ell \in \cal S } p _ { \boldsymbol \theta } ( \boldsymbol x ^ { \flat } \mid \mathbf { z } _ { \tau } )$ instead of the exact joint distribution, clusters of neighboring tokens are more likely to produce spatial inconsistencies. To address this, Besnier et al. [5] propose to use low-discrepancy sequences [25] to guarantee more uniform spatial coverage during decoding. This modification prevents extreme clustering and leads to improved FID and IS compared to standard confidence sampling.

# D.4 Sampling Precision

Zheng et al. [69] found that when Masked Diffusion Models sample with low-precision logits, some logits can underflow. This can reduce the variety of sampled tokens and make Generative Perplexity look better than it really is. Because of this, we cast all logits to FP64 before sampling.

# D.5 Training Costs and Resources

All sampling experiments, for both text and images, are run on NVIDIA A100 GPUs with either 40GB or 80GB of memory. FP-MaskGIT training is also performed on A100 GPUs. For the text experiments, both MDLM and FP-MDLM are trained on 8 NVIDIA H200 GPUs with 141GB of memory per GPU. Unless otherwise stated, all reported latency, throughput, training-time, and VRAM measurements use these hardware settings.

# E Metrics details

In this section, we detail the main metrics used to monitor the performance of the baseline and proposed models and methods of this paper.

# E.1 Generative perplexity

We evaluate generated text using generative perplexity, following prior work on discrete diffusion language models [15, 45, 57]. This metric measures how well a strong autoregressive reference model predicts samples generated by our model. Concretely, we generate $N _ { \mathrm { s a m p } }$ samples and score them with GPT-2 Large:

$$
\text { GenPPL } = \exp \left(- \frac {1}{N _ {\text { samp }}} \sum_ {n = 1} ^ {N _ {\text { samp }}} \frac {1}{L _ {n}} \sum_ {i = 1} ^ {L _ {n}} \log p _ {\text { GPT - 2   Large }} (y _ {i} ^ {(n)} \mid \mathbf {y} _ {<   i} ^ {(n)})\right), \tag {4}
$$

where $L _ { n }$ is the length of generated sample $\mathbf { y } ^ { ( n ) }$ , and $p _ { \mathrm { G P T - 2 L a r g e } } ( y _ { i } ^ { ( n ) } \mid \mathbf { y } _ { < i } ^ { ( n ) } )$ is the probability assigned by GPT-2 Large [55] to token $y _ { i } ^ { ( n ) }$ given its prefix. Lower values indicate that generated text is more predictable under the reference language model and are therefore better.

As noted in prior work, sampling precision can affect this metric substantially. In particular, lowprecision logits may artificially reduce token diversity and make generative perplexity appear better than it truly is. To avoid this issue, we cast logits to float64 before sampling in all generativeperplexity evaluations [69].

# E.2 Unigram Entropy

Generative perplexity alone can reward degenerate text, for example if a model produces repetitive or low-diversity samples. To detect such failures, we also report the unigram entropy of generated text, following prior work [16, 17]. For a generated sequence $\mathbf { y } ^ { ( n ) }$ of length $L _ { n } .$ let $c ( v , \mathbf { y } ^ { ( n ) } )$ denote the number of occurrences of token v. The unigram entropy is

$$
H _ {\text { uni }} = - \frac {1}{N _ {\text { samp }}} \sum_ {n = 1} ^ {N _ {\text { samp }}} \sum_ {v \in \mathcal {V}} \frac {c (v , \mathbf {y} ^ {(n)})}{L _ {n}} \log \frac {c (v , \mathbf {y} ^ {(n)})}{L _ {n}}. \tag {5}
$$

Higher values indicate more diverse token usage, while unusually low entropy can reveal collapse or repetitive generation. We therefore interpret unigram entropy jointly with generative perplexity rather than in isolation.

# E.3 Fréchet Inception Distance and Inception Score

FID embeds real and generated images using a pretrained Inception network, fits a Gaussian distribution to each set of features, and computes the Fréchet distance between the two Gaussians:

$$
\mathrm{FID} = \| \mu_ {r} - \mu_ {g} \| _ {2} ^ {2} + \mathrm{Tr} \left(\Sigma_ {r} + \Sigma_ {g} - 2 (\Sigma_ {r} \Sigma_ {g}) ^ {1 / 2}\right),
$$

where $\left( \mu _ { r } , \Sigma _ { r } \right)$ and $( \mu _ { g } , \Sigma _ { g } )$ are the empirical feature mean and covariance of real and generated images. Lower FID indicates that generated images better match the real data distribution in Inception feature space. IS instead evaluates the class predictions of the Inception network on generated images:

$$
\mathrm{IS} = \exp \left(\mathbb {E} _ {x} \mathrm{KL} (p (y \mid x) \| p (y))\right).
$$

It is high when individual generated images produce confident class predictions while the marginal class distribution remains diverse.

Following the protocol used in the PGM paper (except the number of generated images), we compute these metrics on 10,000 generated images for efficiency, rather than the more common 50,000 [16]. When comparing models, we keep the evaluation protocol fixed across all methods.

# E.4 Throughput and Latency

We measure inference efficiency using both latency and throughput. Latency is the wall-clock time required to generate a batch of samples under a fixed sampling budget. Throughput is the amount of generated output per second, reported as tokens/s for language and images/s for image generation. Lower latency and higher throughput are better. We report these metrics after a short warmup phase and average them over repeated runs on a single device. Unless stated otherwise, all models are evaluated with the same batch size, numerical precision, and hardware setup to ensure a fair comparison (which is A100 40 or 80GB and float64 for sampling text, following [69]). More details regarding these results in Appendix F.11

# E.5 Training time and VRAM used

We report training time as the wall-clock time required to complete the corresponding training run, measured from the first optimization step to the final checkpoint. Unless stated otherwise, this does not include offline preprocessing, dataset download, evaluation, or sampling. For post-training stages such as $\mathcal { L } _ { \mathrm { C O N S } }$ , SDTT, or checkpoint adaptation, we report the additional number of training steps separately when relevant, since these stages are short compared to full pretraining.

We report VRAM as the peak GPU memory used during training, measured per GPU and expressed in GiB/GPU. For language modeling, MDLM and FP-MDLM are trained on 8 NVIDIA H200 GPUs. For image modeling, FP-MaskGIT is trained on NVIDIA A100 GPUs. Sampling experiments for both text and images are run on NVIDIA A100 GPUs with either 40GB or 80GB of memory. All training-time and VRAM comparisons are therefore intended to compare models within the same experimental setting and hardware configuration.

# F Additional Results

# F.1 Extended OWT Generation Results

In this section we detail all the results for FP-MDLM, MDLM and CoFRe on OpenWebText.

# F.1.1 Samples

We provide short uncurated samples to complement Gen. PPL and entropy. They illustrate typical low-budget failure modes: local fluency can be reasonable, but generations may drift semantically, repeat entities or phrases, and lose long-range coherence.

# Sample 1: repetition and semantic drift

Researchers from the Medical Research Institute of Germany’s Centre for Dermatology and Research Center for Brain Research, in Berlin recently, unearthed evidence from more than 50 decades in Berlin, when ARTISTS ate meals on seven different occasions – dishes in toilet tissue drippings; scraps of 22-ounce paper bags and towels; glove boxes, and perfumed socks, wrapped around their bedsits.

Some say that while the press claims that Germans have forged a legend in the Hollywood musical Waters on the White Strip, a portrait of Rolling Nirvana frontman Billie got published in Times Magazine. Perhaps most astonishing is Jonathan Weber, a native German born to immigrant parents who grew up in Berlin, Germany, earning only \$185 for his meals.

The biscuit consists of a sparge of rare, handcrafted alloy of peppers, beef kidneys, pork, pork and oil. “I have got the biscuits from German kitchens,” he says. “I also got the food from German restaurants. I even got ramen noodles from my own supermarket.”

Failure mode: the sample remains locally grammatical, but drifts between unrelated topics and repeats food/Germany motifs with weak semantic consistency.

# Sample 2: fluent fragments with topic drift

<|endoftext|>’s website, wellness.com.com, provides a nurturing health and wellness platform for professionals in the business, health and wellness industries. In the last years we have expanded our wellness Topics Page, post wellness information page on our website and Facebook page and has created a dedicated 24 hour monthly wellness page.

<|endoftext|>Schedstein’s favorite musical savior

This Hammerstein’s was the most memorable year of the millennium. The celebration of Hammerstein’s included an “open mic” festival, performances, movie screenings, and a parade of reviews, all musical from around the world, from MTV to music festivals.

Hopefully this is a forever, this is a this. This is a red. Hopefully it will always be a red. Call it. I can call it that but that warm smile is on. The color of smile. That warm breath is the bottom of “cheeks.”

Failure mode: the generation contains fluent local fragments, but abruptly switches topics and degenerates into repetitive, low-information phrasing.

# F.1.2 MDLM vs FP-MDLM

Table 6 compares MDLM and FP-MDLM across fixed transformer-block budgets on OWT. FP-MDLM substantially improves the low-budget regime: at budget 96, generative perplexity drops from 830.82 to 375.63, and at budget 192 from 343.33 to 273.28, while maintaining similar unigram entropy. This shows that replacing part of the denoiser with a fixed-point block improves the quality–cost trade-off when sampling compute is limited. At larger budgets, however, the standard MDLM becomes stronger, suggesting that the base FP-MDLM architecture alone mainly benefits the low-budget regime and requires additional regularization for strong high-budget generation.

Table 7 reports the corresponding validation perplexity and efficiency metrics. FP-MDLM uses substantially fewer parameters than MDLM, reducing the model size from 170M to 104M parameters. It also lowers training time and VRAM usage, from approximately 139h and 112.4 GiB/GPU to approximately 123h and 93.44 GiB/GPU, while having a slightly lower latency (Appendix F.11). The validation perplexity is worse than MDLM, which is expected from the reduced parameter count and weight sharing, but the generation results in Table 6 show that this trade-off is favorable under low sampling budgets.

Table 6: Generation quality across compute budgets for MDLM and FP-MDLM on OWT. The budget counts the total number of transformer-block forward passes. Training time and training VRAM are reported in the column headers. For MDLM, the budget is obtained by multiplying the number of denoising steps by 12, corresponding to the 12-layer backbone. FP-MDLM results are reported without reuse. 

<table><tr><td rowspan="2">Budget</td><td colspan="2">MDLM12 transformer layers</td><td colspan="2">FP-MDLMno reuse</td></tr><tr><td>Gen. PPL ↓</td><td>Entropy ↑</td><td>Gen. PPL ↓</td><td>Entropy ↑</td></tr><tr><td>96</td><td>830.8200</td><td>5.9100</td><td>375.6314</td><td>5.8102</td></tr><tr><td>192</td><td>343.3300</td><td>5.8100</td><td>273.2752</td><td>5.7630</td></tr><tr><td>384</td><td>196.7900</td><td>5.7500</td><td>215.1965</td><td>5.7259</td></tr><tr><td>768</td><td>143.8800</td><td>5.7000</td><td>179.6546</td><td>5.7016</td></tr><tr><td>1536</td><td>120.7700</td><td>5.6700</td><td>158.5044</td><td>5.6859</td></tr><tr><td>3072</td><td>112.7000</td><td>5.6600</td><td>155.8161</td><td>5.6858</td></tr></table>

Table 7: Validation perplexity, training time and VRAM on OpenWebText. 

<table><tr><td>Model</td><td>#Params</td><td>Val. PPL ↓</td><td>Training time (h)</td><td>VRAM</td></tr><tr><td colspan="5">OWT (1024)</td></tr><tr><td>MDLM</td><td>170M</td><td>23.07</td><td>≈ 139</td><td>112.4 GiB/GPU</td></tr><tr><td>FP-MDLM</td><td>104M</td><td>27.45</td><td>≈ 123</td><td>93.44 GiB/GPU</td></tr></table>

# F.1.3 Initialized vs not initialized FP-MDLM adaptation

Table 8 compares 40k-step FP-MDLM adaptation with and without initialization from a pretrained MDLM checkpoint. Pretrained initialization improves the no-reuse setting at every budget, reducing generative perplexity from 296.76 to 276.00 at budget 96 and from 149.63 to 139.00 at budget 768. This indicates that the layer-mapping initialization provides a better starting point for adapting the fixed-point architecture.

The effect is also visible when reuse is enabled. For both initialized and non-initialized models, reuse becomes more beneficial at medium and high budgets, and three-state reuse gives the best results in most of these regimes. With initialization and 3SR, the adapted FP-MDLM reaches the best overall perplexities at budgets 192, 384, and 768. These results suggest that pretrained initialization not only improves the adapted checkpoint itself, but also makes the resulting fixed-point states more reusable across denoising steps.

Table 8: Pretrained initialization improves short FP-MDLM adaptation. We compare 40k-step FP-MDLM adaptation with and without initialization from a pretrained MDLM checkpoint. For each sampling budget, we report the best generative perplexity and entropy across denoising-step sweeps under no reuse (NR), full reuse (R), and three-state reuse (3SR). Initialization improves generation quality at every budget and gives the best results with 3SR at medium and high budgets. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Reuse</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="3">No initFP-MDLM@40k</td><td>NR</td><td>Gen PPL ↓Entropy ↑</td><td>296.7645.690</td><td>204.7875.661</td><td>170.8355.629</td><td>149.6265.603</td></tr><tr><td>R</td><td>Gen PPL ↓Entropy ↑</td><td>336.0575.705</td><td>201.5215.662</td><td>161.8455.637</td><td>131.8005.590</td></tr><tr><td>3SR</td><td>Gen PPL ↓Entropy ↑</td><td>298.6435.722</td><td>192.2275.658</td><td>149.4275.611</td><td>131.2915.597</td></tr><tr><td rowspan="3">InitFP-MDLM@40k</td><td>NR</td><td>Gen PPL ↓Entropy ↑</td><td>276.0035.696</td><td>194.3555.649</td><td>163.4245.622</td><td>139.0035.578</td></tr><tr><td>R</td><td>Gen PPL ↓Entropy ↑</td><td>312.9125.702</td><td>195.5895.656</td><td>149.9345.617</td><td>130.4435.588</td></tr><tr><td>3SR</td><td>Gen PPL ↓Entropy ↑</td><td>286.4035.695</td><td>184.7085.649</td><td>147.4975.619</td><td>126.8725.572</td></tr></table>

# F.1.4 Generation quality when adding different components of CoFRe

In this section, we ablate the components of CoFRe and analyze how each one affects OWT generation quality. Detailed results are reported in Table 9.

Table 9: Component ablation for FP-MDLM generation on OpenWebText. We report generative perplexity and entropy across sampling budgets. The table compares MDLM, the base FP-MDLM checkpoint trained for 1M steps, the same model with inference-time reuse or three-state reuse, the checkpoint after cross-step consistency regularization, and FP-MDLM checkpoints obtained by adapting a pretrained MDLM with a 40k-step teacher-student KL loss on logits. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="2">MDLM</td><td>Gen PPL ↓</td><td>830.8200</td><td>343.3300</td><td>196.7900</td><td>143.8800</td></tr><tr><td>Entropy ↑</td><td>5.9100</td><td>5.8100</td><td>5.7500</td><td>5.7000</td></tr><tr><td rowspan="2">FP-MDLM</td><td>Gen PPL ↓</td><td>375.6314</td><td>273.2752</td><td>215.1965</td><td>179.6546</td></tr><tr><td>Entropy ↑</td><td>5.8102</td><td>5.7630</td><td>5.7259</td><td>5.7016</td></tr><tr><td rowspan="2">FP-MDLM + reuse</td><td>Gen PPL ↓</td><td>516.677</td><td>253.210</td><td>229.409</td><td>269.007</td></tr><tr><td>Entropy ↑</td><td>5.815</td><td>5.729</td><td>5.728</td><td>2.209</td></tr><tr><td rowspan="2">FP-MDLM + 3SR</td><td>Gen PPL ↓</td><td>454.100</td><td>249.322</td><td>196.307</td><td>254.258</td></tr><tr><td>Entropy ↑</td><td>5.795</td><td>5.736</td><td>5.663</td><td>5.384</td></tr><tr><td rowspan="2">FP-MDLM +  $\mathcal{L}_{\text{CONS}}$ </td><td>Gen PPL ↓</td><td>104.153</td><td>70.275</td><td>54.927</td><td>41.673</td></tr><tr><td>Entropy ↑</td><td>5.447</td><td>5.388</td><td>5.343</td><td>5.156</td></tr><tr><td rowspan="2">FP-MDLM +  $\mathcal{L}_{\text{CONS}}+$  reuse</td><td>Gen PPL ↓</td><td>117.592</td><td>68.550</td><td>50.195</td><td>37.567</td></tr><tr><td>Entropy ↑</td><td>5.438</td><td>5.387</td><td>5.283</td><td>5.142</td></tr><tr><td rowspan="2">FP-MDLM +  $\mathcal{L}_{\text{CONS}}+$  3SR</td><td>Gen PPL ↓</td><td>101.791</td><td>65.182</td><td>48.755</td><td>37.846</td></tr><tr><td>Entropy ↑</td><td>5.434</td><td>5.380</td><td>5.283</td><td>5.142</td></tr><tr><td rowspan="2">Adapted FP-MDLM</td><td>Gen PPL ↓</td><td>296.764</td><td>204.787</td><td>170.835</td><td>149.626</td></tr><tr><td>Entropy ↑</td><td>5.690</td><td>5.661</td><td>5.629</td><td>5.603</td></tr><tr><td rowspan="2">Adapted FP-MDLM+ reuse</td><td>Gen PPL ↓</td><td>336.057</td><td>201.521</td><td>161.845</td><td>131.800</td></tr><tr><td>Entropy ↑</td><td>5.705</td><td>5.662</td><td>5.637</td><td>5.590</td></tr><tr><td rowspan="2">Adapted FP-MDLM+ 3SR</td><td>Gen PPL ↓</td><td>298.643</td><td>192.227</td><td>149.427</td><td>131.291</td></tr><tr><td>Entropy ↑</td><td>5.722</td><td>5.658</td><td>5.611</td><td>5.597</td></tr></table>

Table 9 decomposes the effect of the main CoFRe components. The base FP-MDLM substantially improves over MDLM in the very low-budget regime, reducing generative perplexity from 830.82 to 375.63 at budget 96. However, this advantage decreases at larger budgets, where the unregularized FP-MDLM remains worse than MDLM. This confirms that the fixed-point architecture alone improves the low-budget quality–cost trade-off, but is not sufficient for strong generation across all budgets.

Reuse alone is unstable on the base FP-MDLM checkpoint. Full reuse and 3SR improve some medium-budget results, but can hurt in other regimes; in particular, full reuse at budget 768 leads to severe degeneration, as reflected by both high generative perplexity and very low entropy. This supports the motivation for adding a regularization mechanism that makes representations smoother and more reusable across denoising steps.

Cross-step consistency is the largest contributor to generation quality. Adding LCONS reduces generative perplexity from 375.63 to 104.15 at budget 96 and from 179.65 to 41.67 at budget 768. Once the model is regularized with LCONS, reuse becomes much more effective: 3SR gives the best results at budgets 96, 192, and 384, while full reuse is slightly better at budget 768. Finally, the adapted FP-MDLM rows show a complementary path to obtaining useful fixed-point denoisers from pretrained MDLM checkpoints with only a short teacher-student adaptation stage.

# F.1.5 Mauve score evaluation

Following [52, 63], we evaluate generation quality with MAUVE, which measures the distributional gap between model-generated and human-written text using divergence frontiers [43, 52]. Like [63], for each model and sampling budget, we generate 5,000 samples and compare them against 5,000 OpenWebText reference samples. We compute also MAUVE beside generative perplexity as a generation metric because it accounts for both sample quality and diversity, whereas perplexity alone can be uninformative for corrector-based samplers that may trade diversity for lower perplexity [69]. Table 10 shows that CoFRe consistently outperforms MDLM across all budgets, with the strongest improvements at budgets 96 and 192, suggesting that cross-step consistency regularization and three-state reuse improve the quality–diversity trade-off of generated text. (More details in [52] about MAUVE).

Table 10: MAUVE scores for MDLM and CoFRe generation on OpenWebText. We report MAUVE scores computed with 5,000 generated samples and 5,000 reference samples. CoFRe denotes FP-MDLM with cross-step consistency regularization and three-state reuse. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td>MDLM</td><td>MAUVE ↑</td><td>0.010594</td><td>0.010176</td><td>0.010197</td><td>0.009641</td></tr><tr><td>CoFRe</td><td>MAUVE ↑</td><td>0.013759</td><td>0.014188</td><td>0.012080</td><td>0.010704</td></tr></table>

# F.1.6 Downstream evaluation

Table 11 shows that CoFRe is competitive with MDLM on downstream multiple-choice tasks. It improves on LAMBADA (39.45 vs. 38.52), ARC-easy (36.70 vs. 34.26), and BoolQ (60.21 vs. 49.42), while underperforming on ARC-challenge, OpenBookQA, PIQA, RACE, and SIQA. This mixed behaviour is expected: these tasks are evaluated through likelihood-based answer scoring, so CoFRe does not directly benefit from its main advantage, which appears during low-budget generation. Overall, the results suggest that CoFRe preserves reasonable likelihood-based downstream performance, while its strongest gains remain in the sampling regime.

Table 11: Results on downstream evaluation tasks. 

<table><tr><td></td><td>LAMBADA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>OBQA</td><td>PIQA</td><td>RACE</td><td>SIQA</td></tr><tr><td>MDLM</td><td>38.52</td><td>34.26</td><td>24.66</td><td>49.42</td><td>28.60</td><td>58.27</td><td>28.04</td><td>38.84</td></tr><tr><td>CoFRe</td><td>39.45</td><td>36.70</td><td>22.87</td><td>60.21</td><td>25.20</td><td>55.33</td><td>27.27</td><td>35.82</td></tr></table>

# F.1.7 Sampling with nucleus sampling

As suggested by Deschenaux et al. [16], Wang et al. [63], nucleus sampling [29] can highly impact the generation of high-quality text sequences. We therefore use this method (with top-p = 0.9) on both MDLM and CoFRe and compare these results in Table 12.

Table 12: The effect of using nucleus sampling on MDLM and CoFRe generation on Open-WebText. We report generative perplexity and entropy across sampling budgets, comparing MDLM against CoFRe. Sampling is performed with nucleus sampling with $p = 0 . 9 ,$ as in [16, 63]. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="2">MDLM</td><td>Gen PPL ↓</td><td>830.8200</td><td>343.3300</td><td>196.7900</td><td>143.8800</td></tr><tr><td>Entropy ↑</td><td>5.9100</td><td>5.8100</td><td>5.7500</td><td>5.7000</td></tr><tr><td rowspan="2">MDLM+nucleus</td><td>Gen PPL ↓</td><td>292.957</td><td>119.284</td><td>69.479</td><td>51.187</td></tr><tr><td>Entropy ↑</td><td>5.609</td><td>5.538</td><td>5.481</td><td>5.433</td></tr><tr><td rowspan="2">FP-MDLM+  $\mathcal{L}_{\text{CONS}} + 3SR$ </td><td>Gen PPL ↓</td><td>101.791</td><td>65.182</td><td>48.755</td><td>37.846</td></tr><tr><td>Entropy ↑</td><td>5.434</td><td>5.380</td><td>5.283</td><td>5.142</td></tr><tr><td rowspan="2">FP-MDLM+  $\mathcal{L}_{\text{CONS}} + 3SR + \text{nucleus}$ </td><td>Gen PPL ↓</td><td>39.518</td><td>30.113</td><td>29.199</td><td>28.111</td></tr><tr><td>Entropy ↑</td><td>5.108</td><td>5.057</td><td>5.069</td><td>5.055</td></tr></table>

Table 12 shows that nucleus sampling substantially improves generation quality for both MDLM and CoFRe, but also introduces a clear tradeoff. For MDLM, using top-p = 0.9 reduces Gen PPL from 830.82 to 292.96 at budget 96, and from 143.88 to 51.19 at budget 768. However, this improvement comes with lower unigram entropy, which decreases from 5.91 to 5.61 at budget 96 and from 5.70 to

5.43 at budget 768. The same pattern is observed for CoFRe: adding nucleus sampling to CoFRe with LCONS and 3SR further reduces Gen PPL across all budgets, reaching 28.11 at budget 768, while also lowering unigram entropy. Overall, nucleus sampling improves the Gen PPL-quality tradeoff, but does so by making generations less diverse according to unigram entropy.

# F.2 Using consistency loss on MDLM

We next compare CoFRe against several MDLM-based baselines to isolate the effect of each component. In particular, we evaluate the original MDLM, MDLM trained with the same cross-step consistency regularization, MDLM with [15], and our full CoFRe method. This comparison allows us to assess whether the gains come only from consistency regularization, from improved sampling, or from the full CoFRe formulation.

Table 13: Comparison of MDLM variants and CoFRe generation on OpenWebText. We report generative perplexity and unigram entropy across sampling budgets, comparing MDLM, MDLM with cross-step consistency regularization, MDLM with SDTT, and CoFRe, defined as FP-MDLM with cross-step consistency regularization and three-state reuse. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="2">MDLM</td><td>Gen PPL ↓</td><td>830.820</td><td>343.330</td><td>196.790</td><td>143.880</td></tr><tr><td>Entropy ↑</td><td>5.910</td><td>5.810</td><td>5.750</td><td>5.700</td></tr><tr><td rowspan="2">MDLM +  $\mathcal{L}_{\text{CONS}}$ </td><td>Gen PPL ↓</td><td>621.121</td><td>250.313</td><td>141.397</td><td>102.731</td></tr><tr><td>Entropy ↑</td><td>5.840</td><td>5.757</td><td>5.690</td><td>5.636</td></tr><tr><td rowspan="2">MDLM + SDTT</td><td>Gen PPL ↓</td><td>193.050</td><td>89.170</td><td>62.290</td><td>47.040</td></tr><tr><td>Entropy ↑</td><td>5.580</td><td>5.530</td><td>5.490</td><td>5.450</td></tr><tr><td rowspan="2">CoFRe (FP-MDLM +  $\mathcal{L}_{\text{CONS}}$  + 3SR)</td><td>Gen PPL ↓</td><td>101.791</td><td>65.182</td><td>48.755</td><td>37.846</td></tr><tr><td>Entropy ↑</td><td>5.434</td><td>5.380</td><td>5.283</td><td>5.142</td></tr></table>

Table 13 shows that each additional modelling or sampling component improves Gen PPL over the MDLM baseline. Cross-step consistency regularization alone reduces MDLM Gen PPL by approximately 25.2%, 27.1%, 28.1%, and 28.6% across budgets 96, 192, 384, and 768, respectively. SDTT yields substantially larger gains, reducing Gen PPL to 193.05 at budget 96 and 47.04 at budget 768. CoFRe performs best across all budgets, reaching 101.79, 65.18, 48.76, and 37.85 Gen PPL. Compared to MDLM, this corresponds to relative reductions of approximately 87.8%, 81.0%, 75.2%, and 73.7%. These gains come with a reduction in unigram entropy, from 5.91-5.70 for MDLM to 5.43-5.14 for CoFRe, highlighting the quality-diversity tradeoff induced by the proposed method.

# F.3 Comparison against other distilled or accelerated diffusion language models

We next compare CoFRe against other distilled or accelerated diffusion language models. In particular, we evaluate PGM 6/6, PGM 6/6 with SDTT [16], IDLM-MDLM [40], and our full CoFRe method. For PGM and IDLM-MDLM, we report the effective sampling budget as 12× the number of sampling steps. We emphasize that CoFRe is not primarily designed as a distillation or acceleration method: rather, its goal is to provide a general generation framework that improves sample quality while reducing the effective cost of training and the cost of generation, especially in low-budget regimes. This comparison allows us to assess how CoFRe compares not only to MDLM-based variants, but also to methods that explicitly target faster sampling or improved diffusion language model generation.

![](images/dcb5aba8446f6b0cd930d424050c9957c3378bc80a8e2dcc8169ff2db67ac9b3.jpg)

<details>
<summary>line</summary>

| Budget | CoFRe | PGM 6/6 | PGM 6/6 + SDTT | IDLM-MDLM |
| ------ | ----- | ------- | -------------- | --------- |
| 96     | 5.43  | 5.84    | 5.60           | 5.53      |
| 192    | 5.38  | 5.79    | 5.56           | 5.32      |
| 384    | 5.28  | 5.73    | 5.52           | 5.09      |
| 768    | 5.14  | 5.69    | 5.48           | 4.78      |
</details>

Figure 9: Generative perplexity as a function of sampling budget on OpenWebText. We compare CoFRe against PGM 6/6, PGM 6/6 with SDTT, and IDLM-MDLM.

Table 14: Comparison of CoFRe, PGM, and IDLM-MDLM on OpenWebText. We report generative perplexity and entropy across sampling budgets. For PGM and IDLM-MDLM, the budget is computed as 12× the number of sampling steps. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td>CoFRe</td><td>Gen PPL ↓</td><td>101.791</td><td>65.182</td><td>48.755</td><td>37.846</td></tr><tr><td>FP-MDLM +  $\mathcal{L}_{\text{CONS}}$  + 3SR</td><td>Entropy ↑</td><td>5.434</td><td>5.380</td><td>5.283</td><td>5.142</td></tr><tr><td rowspan="2">PGM 6/6</td><td>Gen PPL ↓</td><td>693.513</td><td>312.812</td><td>179.928</td><td>132.786</td></tr><tr><td>Entropy ↑</td><td>5.843</td><td>5.785</td><td>5.732</td><td>5.688</td></tr><tr><td rowspan="2">PGM 6/6 + SDTT</td><td>Gen PPL ↓</td><td>249.076</td><td>126.604</td><td>83.952</td><td>66.316</td></tr><tr><td>Entropy ↑</td><td>5.599</td><td>5.561</td><td>5.524</td><td>5.484</td></tr><tr><td rowspan="2">IDLM-MDLM</td><td>Gen PPL ↓</td><td>69.165</td><td>29.659</td><td>16.843</td><td>11.312</td></tr><tr><td>Entropy ↑</td><td>5.529</td><td>5.323</td><td>5.089</td><td>4.782</td></tr></table>

Table 14 and Figure 9 show that CoFRe substantially improves over the PGM baselines across all sampling budgets. Compared to PGM 6/6, CoFRe reduces Gen PPL across all budgets. Applying SDTT to PGM gives a strong improvement over the undistilled PGM baseline, reducing Gen PPL from 693.51 to 249.08 at budget 96 and from 132.79 to 66.32 at budget 768. Nevertheless, CoFRe remains better than PGM 6/6 + SDTT at every budget, with relative reductions of approximately 59.1%, 48.5%, 41.9%, and 42.9%.

IDLM-MDLM achieves the lowest Gen PPL across all budgets, reaching 69.17, 29.66, 16.84, and 11.31. Compared to CoFRe, this corresponds to additional reductions of approximately 32.1%, 54.5%, 65.5%, and 70.1%. However, these improvements are accompanied by a stronger reduction in unigram entropy at larger budgets: IDLM-MDLM entropy decreases from 5.53 at budget 96 to 4.78 at budget 768, whereas CoFRe remains between 5.43 and 5.14. Overall, CoFRe provides a strong improvement over accelerated PGM variants while preserving higher entropy than IDLM-MDLM at medium and large budgets, again highlighting the quality-diversity tradeoff among accelerated diffusion language model samplers.

# F.4 Extended Image Modeling Results

# F.4.1 Samples

![](images/c81c4d4f66cf33a9b4baa4939cab417d1dd8ea03b0e3940ba645f44a2d6eebf5.jpg)

![](images/cd4dc3f9aa606bc07ff4f7d1962b42f648ec75cda686b226a0d20a7a1d08467f.jpg)

![](images/4001e150b668a081bb2d2a5bd6942ebd7821ba6f3a16e5927e7151b6670ce20d.jpg)

![](images/f505c629edf4f07e2b6db1332e924bdc589be229c1ee13d552d550f2c02ddfb5.jpg)

![](images/d03dd4aaa788ac1ca5976b9c36e3d168dcbda1ed46ccc873f7a0728a5c929171.jpg)

![](images/1a9680ec33f643b640de7a635df7b4d49a1debdd95c2f5ff88f7d2e62a5fd265.jpg)  
Figure 10: Generated samples using CoFRe with a budget of 460.

# F.4.2 MaskGIT-Large vs MaskGIT-12 vs FP-MaskGIT

Table 15: Generation quality vs. compute budget, where the budget counts the total number of transformer-block forward passes. Training time and training VRAM are reported in the column headers; percentages indicate reductions relative to MaskGIT-Large. Models are trained on Imagenette (10-class subset of ImageNet) for 50k training steps. MaskGIT-Large uses 24 transformer layers, MaskGIT-12 uses 12 transformer layers, and FP-MaskGIT uses 4 pre- and 4 post-transformer layers. All models use a global batch size (GBS) of 128. For 24-layer MaskGIT-Large, a budget of 48 corresponds to 2 decoding steps $( 2 \times 2 4 )$ , while for 12-layer MaskGIT-12 it corresponds to 4 decoding steps $( 4 \times 1 2 )$ . 

<table><tr><td rowspan="3">Budget</td><td colspan="2">MaskGIT-Large</td><td rowspan="2" colspan="2">FP-MaskGIT4 pre / 4 post layersTrain: 9h 08m (-48.6%)VRAM: 35.74 GiB (-50.7%)</td><td rowspan="2" colspan="2">MaskGIT-1212 transformer layersTrain: 7h 40m (-56.8%)VRAM: 28.14 GiB (-61.2%)</td></tr><tr><td>24 transformer layersTrain: 17h 46m</td><td>VRAM: 72.45 GiB</td></tr><tr><td>FID ↓</td><td>IS ↑</td><td>FID ↓</td><td>IS ↑</td><td>FID ↓</td><td>IS ↑</td></tr><tr><td>48</td><td>174.0856</td><td>9.2860</td><td>100.0964</td><td>15.0196</td><td>128.9871</td><td>12.6302</td></tr><tr><td>96</td><td>117.6439</td><td>13.3696</td><td>57.5823</td><td>15.8495</td><td>77.0653</td><td>15.9587</td></tr><tr><td>192</td><td>54.6172</td><td>16.0220</td><td>32.0072</td><td>15.0822</td><td>50.9830</td><td>15.3803</td></tr><tr><td>240</td><td>44.4930</td><td>15.6458</td><td>27.6470</td><td>14.6582</td><td>45.6384</td><td>15.2925</td></tr><tr><td>384</td><td>30.0202</td><td>14.6473</td><td>24.1946</td><td>14.4567</td><td>38.0946</td><td>15.2823</td></tr><tr><td>480</td><td>27.3975</td><td>14.2319</td><td>22.1171</td><td>13.8718</td><td>36.0931</td><td>15.2505</td></tr></table>

# F.4.3 Generation quality when adding different components of CoFRe

Table 16: Component ablation for FP-MaskGIT generation on ImageNette. We report FID and Inception Score across sampling budgets. The table compares MaskGIT baselines, the FP-MaskGIT backbone with different reuse variants, and FP-MaskGIT with cross-step consistency regularization. CoFRe corresponds to the consistency-trained FP-MaskGIT model with reuse. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>48</td><td>96</td><td>192</td><td>384</td></tr><tr><td rowspan="2">MaskGIT-Large</td><td>FID ↓</td><td>174.0856</td><td>117.6439</td><td>54.6172</td><td>30.0202</td></tr><tr><td>IS ↑</td><td>9.2860</td><td>13.3696</td><td>16.0220</td><td>14.6473</td></tr><tr><td rowspan="2">MaskGIT-12</td><td>FID ↓</td><td>128.9871</td><td>77.0653</td><td>50.9830</td><td>38.0946</td></tr><tr><td>IS ↑</td><td>12.6302</td><td>15.9587</td><td>15.3803</td><td>15.2823</td></tr><tr><td rowspan="2">FP-MaskGIT no reuse</td><td>FID ↓</td><td>100.0964</td><td>57.5823</td><td>32.0072</td><td>24.1946</td></tr><tr><td>IS ↑</td><td>15.0196</td><td>15.8495</td><td>15.0822</td><td>14.4567</td></tr><tr><td rowspan="2">FP-MaskGIT + reuse</td><td>FID ↓</td><td>101.5056</td><td>55.2582</td><td>31.7141</td><td>24.0396</td></tr><tr><td>IS ↑</td><td>14.4204</td><td>15.5139</td><td>14.5876</td><td>13.9587</td></tr><tr><td rowspan="2">FP-MaskGIT + 3SR</td><td>FID ↓</td><td>102.3251</td><td>54.4212</td><td>31.2308</td><td>23.5458</td></tr><tr><td>IS ↑</td><td>14.432</td><td>15.4369</td><td>14.912</td><td>14.0228</td></tr><tr><td rowspan="2">FP-MaskGIT +  $\mathcal{L}_{\text{CONS}}+$  no reuse</td><td>FID ↓</td><td>107.5012</td><td>56.7336</td><td>31.3828</td><td>24.6839</td></tr><tr><td>IS ↑</td><td>13.3413</td><td>16.3229</td><td>15.0487</td><td>14.0886</td></tr><tr><td rowspan="2">FP-MaskGIT +  $\mathcal{L}_{\text{CONS}}+$  reuse</td><td>FID ↓</td><td>97.8401</td><td>54.5651</td><td>30.0072</td><td>23.4257</td></tr><tr><td>IS ↑</td><td>15.0866</td><td>16.2692</td><td>14.5894</td><td>14.2665</td></tr><tr><td rowspan="2">FP-MaskGIT +  $\mathcal{L}_{\text{CONS}}+$  reuse, CoFRe</td><td>FID ↓</td><td>96.7331</td><td>51.0077</td><td>27.6242</td><td>22.8381</td></tr><tr><td>IS ↑</td><td>14.4074</td><td>15.9572</td><td>15.0822</td><td>14.4567</td></tr></table>

# F.5 Component Ablation for FP-MGM

We use small-scale proxy runs to select the main FP-MGM design choices before running the full experiments. For language, we ablate the number of explicit preprocessing/postprocessing layers, the number of fixed-point iterations used with and without gradients, the learning rate, and gradient clipping. Unless stated otherwise, these ablations are run for 100k training steps and are intended to compare configurations under the same compute setting, not to match the final full-scale results.

Language-modeling ablations on LM1B. Table 17 reports FP-MDLM ablations on LM1B without sentence packing. The main trend is that very shallow stochastic fixed-point training is cheapest, while larger stochastic solver budgets improve perplexity at moderate cost. Using a deterministic large solver budget, U(12, 12) without gradients and U(12, 12) with gradients, gives the best validation perplexity among FP-MDLM variants, but it is substantially slower and more memory-intensive. The stochastic setting U(0, 4) without gradients and U(3, 6) with gradients gives the best FP-MDLM test perplexity, and provides a better cost–quality trade-off. We therefore use this stochastic solver setting as the default for the larger FP-MDLM runs.

Table 17: FP-MDLM solver and architecture ablation on LM1B. Models are trained for 100k steps without sentence packing. $\textstyle { \mathcal { U } } ( a , b )$ denotes a discrete uniform distribution over solver iterations. We report validation perplexity and test perplexity on 500 samples. 

<table><tr><td>Model</td><td>Pre/Post</td><td>No-grad</td><td>Grad</td><td>LR</td><td>Clip</td><td>Time</td><td>VRAM</td><td>Val. PPL ↓</td><td>Test PPL ↓</td></tr><tr><td>FP-MDLM</td><td>2/2</td><td> $\mathcal{U}(0,2)$ </td><td> $\mathcal{U}(1,3)$ </td><td> $1\times 10^{-4}$ </td><td>0.5</td><td>8h05</td><td>36.0GB</td><td>50.66</td><td>48.057</td></tr><tr><td>FP-MDLM</td><td>2/2</td><td> $\mathcal{U}(0,4)$ </td><td> $\mathcal{U}(3,6)$ </td><td> $1\times 10^{-4}$ </td><td>0.5</td><td>9h53</td><td>41.0GB</td><td>46.54</td><td>44.885</td></tr><tr><td>FP-MDLM</td><td>2/2</td><td> $\mathcal{U}(12,12)$ </td><td> $\mathcal{U}(12,12)$ </td><td> $1\times 10^{-4}$ </td><td>0.5</td><td>16h49</td><td>56.2GB</td><td>42.50</td><td>45.136</td></tr><tr><td>FP-MDLM</td><td>1/1</td><td> $\mathcal{U}(0,2)$ </td><td> $\mathcal{U}(1,3)$ </td><td> $1\times 10^{-4}$ </td><td>0.5</td><td>7h05</td><td>30.7GB</td><td>47.37</td><td>50.740</td></tr><tr><td>FP-MDLM</td><td>1/1</td><td> $\mathcal{U}(0,4)$ </td><td> $\mathcal{U}(3,6)$ </td><td> $1\times 10^{-4}$ </td><td>0.5</td><td>8h53</td><td>36.18GB</td><td>44.60</td><td>46.620</td></tr><tr><td>FP-MDLM</td><td>1/1</td><td> $\mathcal{U}(0,3)$ </td><td> $\mathcal{U}(2,4)$ </td><td> $1\times 10^{-4}$ </td><td>0.5</td><td>7h52</td><td>34.4GB</td><td>49.19</td><td>47.690</td></tr><tr><td>FP-MDLM</td><td>1/1</td><td> $\mathcal{U}(0,4)$ </td><td> $\mathcal{U}(3,6)$ </td><td> $2\times 10^{-4}$ </td><td>0.5</td><td>8h51</td><td>36.18GB</td><td>44.06</td><td>45.926</td></tr><tr><td>FP-MDLM</td><td>1/1</td><td> $\mathcal{U}(0,4)$ </td><td> $\mathcal{U}(3,6)$ </td><td> $1\times 10^{-4}$ </td><td>1.0</td><td>8h50</td><td>36.18GB</td><td>44.60</td><td>46.773</td></tr><tr><td>FP-MDLM</td><td>1/1</td><td> $\mathcal{U}(0,4)$ </td><td> $\mathcal{U}(3,6)$ </td><td> $2\times 10^{-4}$ </td><td>1.0</td><td>8h50</td><td>36.18GB</td><td>42.795</td><td>45.175</td></tr><tr><td>MDLM DiT</td><td>-</td><td>-</td><td>-</td><td> $3\times 10^{-4}$ </td><td>-</td><td>11h02</td><td>44.0GB</td><td>42.24</td><td>48.522</td></tr></table>

Language-modeling ablations on OWT. Table 18 shows the corresponding proxy experiment on OWT with sequence length 128. The FP-MDLM variants use substantially fewer parameters than the MDLM DiT baseline. Among the FP-MDLM variants, configuration B, with learning rate $2 \times 1 0 ^ { - 4 }$ and gradient clipping 1.0, gives the best validation and test perplexity. It also improves generative perplexity over configuration A across all reported sampling budgets. MDLM remains stronger in likelihood at this small scale, but FP-MDLM is competitive in low-budget generation and uses fewer parameters. For FP-MDLM, we use $K _ { \mathrm { p r e } } = 1$ , Kfp = 1, and $K _ { \mathrm { p o s t } } = 1$ .

Table 18: FP-MDLM proxy ablation on OWT with sequence length 128. Models are trained for 100k steps. Generation cells report generative perplexity / unigram entropy. 

<table><tr><td>Model</td><td>Pre/Post</td><td>LR</td><td>Clip</td><td>Params</td><td>Val. PPL ↓</td><td>Test PPL ↓</td><td>B128</td><td>B64</td><td>B32</td><td>B16</td><td>B8</td></tr><tr><td>FP-MDLM A</td><td> $1/1$ </td><td> $1 \times 10^{-4}$ </td><td>0.5</td><td>104M</td><td>58.98</td><td>55.390</td><td>185.42 / 4.32</td><td>216.24 / 4.33</td><td>263.59 / 4.34</td><td>400.29 / 4.38</td><td>744.83 / 4.41</td></tr><tr><td>FP-MDLM B</td><td> $1/1$ </td><td> $2 \times 10^{-4}$ </td><td>1.0</td><td>104M</td><td>53.278</td><td>50.287</td><td>170.79 / 4.31</td><td>201.23 / 4.32</td><td>230.00 / 4.34</td><td>380.00 / 4.38</td><td>637.00 / 4.41</td></tr><tr><td>MDLM DiT</td><td>–</td><td> $3 \times 10^{-4}$ </td><td>1.0</td><td>169M</td><td>47.10</td><td>45.211</td><td>133.15 / 4.29</td><td>150.62 / 4.31</td><td>193.946 / 4.33</td><td>285.50 / 4.36</td><td>552.71 / 4.38</td></tr></table>

Image-generation ablation. For image generation, we use the same fixed-point replacement idea inside MaskGIT and compare the resulting FP-MaskGIT to fixed-depth MaskGIT baselines. Table 19 reports the completed ImageNette runs. FP-MaskGIT with a 4/4 explicit pre/post architecture substantially reduces training cost relative to MaskGIT-Large, from 17h46m to 9h08m and from 72.45GiB to 35.74GiB. It also improves FID at all reported budgets. Compared with the smaller 12-layer MaskGIT baseline, FP-MaskGIT is more expensive to train but gives better FID across the reported budgets, supporting the choice of the 4/4 FP-MaskGIT configuration for the main image experiments. For FP-MaskGIT, we use $K _ { \mathrm { p r e } } = 4 , K _ { \mathrm { f p } } = 1$ , and $K _ { \mathrm { p o s t } } = 4$ .

Table 19: Architecture ablation for FP-MaskGIT on ImageNette. We compare MaskGIT baselines against FP-MaskGIT with different numbers of explicit preprocessing and postprocessing layers. Smaller FP-MaskGIT variants perform progressively worse than the 4/4 configuration. For each model, we report FID and IS across sampling budgets. 

<table><tr><td>Model</td><td>Metric</td><td>48</td><td>96</td><td>192</td><td>240</td><td>384</td><td>480</td></tr><tr><td>MaskGIT-Large</td><td>FID ↓IS ↑</td><td>174.08569.2860</td><td>117.643913.3696</td><td>54.617216.0220</td><td>44.493015.6458</td><td>30.020214.6473</td><td>27.397514.2319</td></tr><tr><td>MaskGIT-12</td><td>FID ↓IS ↑</td><td>128.987112.6302</td><td>77.065315.9587</td><td>50.983015.3803</td><td>45.638415.2925</td><td>38.094615.2823</td><td>32.873114.1865</td></tr><tr><td>FP-MaskGIT 1/1</td><td>FID ↓IS ↑</td><td>121.764412.7667</td><td>72.194513.4721</td><td>46.239012.8199</td><td>41.140512.4595</td><td>34.619612.2882</td><td>30.184111.7910</td></tr><tr><td>FP-MaskGIT 2/2</td><td>FID ↓IS ↑</td><td>114.541813.5176</td><td>67.323814.2646</td><td>41.495113.5740</td><td>36.642713.1924</td><td>31.144613.0110</td><td>27.495112.4846</td></tr><tr><td>FP-MaskGIT 3/3</td><td>FID ↓IS ↑</td><td>107.319114.2686</td><td>62.453015.0570</td><td>36.751114.3281</td><td>32.144813.9253</td><td>27.669613.7339</td><td>24.806113.1782</td></tr><tr><td>FP-MaskGIT 4/4</td><td>FID ↓IS ↑</td><td>100.096415.0196</td><td>57.582315.8495</td><td>32.007215.0822</td><td>27.647014.6582</td><td>24.194614.4567</td><td>22.117113.8718</td></tr></table>

Overall, these ablations motivate the default FP-MGM recipe used in the main experiments: a stochastic fixed-point solver rather than a fully deterministic deep solve, a compact explicit pre/post architecture for language, and a 4/4 explicit pre/post architecture for image generation. The proxy results also show the main trade-off that carries through the full experiments: fixed-point denoisers reduce parameter and memory cost, but require careful solver-budget and reuse choices to obtain strong generation quality.

# F.6 Cross-step consistency recovers low-budget generation quality

In masked generative models, the loss is typically applied only to final predictions at masked positions, so intermediate states are supervised only indirectly. A common extension is consistency regularization, which aligns predictions or hidden states across nearby denoising steps. This can be done in output space, for example with KL on logits, or in representation space, for example with MSE or cosine similarity on hidden states. Targets may come from the same model with stop-gradient or from a teacher such as an EMA copy. This adds auxiliary supervision between denoising states while leaving the original masked modeling objective unchanged.

A related idea is self-distillation through time (SDTT), which trains a student with fewer denoising steps to match a teacher with more steps. $\operatorname { I f } p _ { \theta } ^ { ( m ) }$ is generation with m steps and $p _ { \nu } ^ { ( k ) }$ the student model with $k < m$ , SDTT minimizes $\mathbb { E } _ { \mathbf { z } _ { 0 } \sim \mathcal { D } , \mathbf { z } _ { t } \sim q _ { t } ( \mathbf { z } _ { t } | \mathbf { z } _ { 0 } ) } \delta \bigl ( \mathbf { x } _ { \nu } ( \mathbf { z } _ { t } , t ) , \tilde { \mathbf { x } } _ { \theta } ^ { \mathrm { t e a c h e r } } ( \mathbf { z } _ { t } , t , m / k ) \bigr )$ , where δ is typically a divergence such as $\mathrm { K L , }$ , sg denotes stop-gradient, and $\widetilde { \mathbf { x } } _ { \theta } ^ { \mathrm { t e a c h e r } }$ aggregates teacher predictions over multiple steps. After training, one student step can approximate several teacher steps. Consistency regularization thus provides local temporal supervision, whereas SDTT directly compresses the denoising process.

Experimental Settings For the cross-step consistency regularization experiments, we compare two FP-MDLM models with the same 1M-step base checkpoint, followed by a 30k consistency post-training stage. The baseline is trained from scratch using only the MDLM loss. The consistency model starts from the same training setup, but adds a consistency objective late in training. We explore several consistency objectives and found the best results with $\mathcal { L } _ { \mathrm { M S E } }$ introduced Sec. 3.2; the comparison is reported in Appendix F.8. Specifically, the consistency weight is introduced at step 1M, increased linearly from 0 to 0.1 over 5k steps, and then kept at 0.1 until step 1030000. We evaluate both models using GPT-2 Large generative perplexity and sample entropy, and report results both with and without solution reuse.

Results Table 9 shows that cross-step consistency regularization is the main driver of generation quality. Without reuse, adding $\mathcal { L } _ { \mathrm { C O N S } }$ improves generative perplexity at every budget, from 375.6 to 104.2 at budget 96 and from 179.7 to 41.7 at budget 768. This is a large improvement over the baseline FP-MDLM, while maintaining non-degenerate entropy across all budgets.

Reuse becomes useful once the model has been regularized with ${ \mathcal { L } } _ { \mathrm { C O N S } }$ . In the baseline checkpoint, reuse and 3SR are inconsistent and can hurt generation quality, especially at the largest budget. After consistency regularization, however, reuse further improves the low-perplexity model at budgets 192, 384, and 768. The best results are obtained with 3SR at budgets 96, 192, and 384, reaching generative perplexities of 101.8, 65.2, and 48.8, while standard reuse is slightly better at budget 768 with 37.6. Overall, ${ \mathcal { L } } _ { \mathrm { C O N S } }$ turns FP-MDLM into a much stronger generator, and reuse provides additional gains once the fixed-point denoiser has been regularized.

Lagged logit analysis. To understand the improvement from ${ \mathcal { L } } _ { \mathrm { C O N S } } .$ , we compare masked-token logits at a student denoising step s with logits at cleaner future steps $s + \ell$ on shared contexts. The consistency-trained model has lower lagged logit KL than the baseline across all sampling-step, solver-budget, and lag settings, reducing the KL by 15.2% on average. The effect is strongest for nearby denoising states, with a 19.0% average reduction at lag 1, and remains positive through lag 4. This suggests that ${ \mathcal { L } } _ { \mathrm { C O N S } }$ acts as a cross-time self-distillation signal: it makes student-step logits more aligned with cleaner future predictions, which helps explain the large generative perplexity gains in Tables 9 and 16.

![](images/b8abdc54ad9140fcfca1f5d8e9be541a2ff816c493cdadecadb6ab6bb96799f7.jpg)

<details>
<summary>line</summary>

| Sampling steps | L_CONS | Baseline |
| -------------- | ------ | -------- |
| 0.8 ×10¹       | 3.0 ×10² | 3.2 ×10² |
| 1.6 ×10¹       | 1.8 ×10² | 1.9 ×10² |
| 3.2 ×10¹       | 1.1 ×10² | 1.1 ×10² |
</details>

![](images/f17fec7019c2c6e5e1549c2a7b6225d13d1234489c4e9ad8d39069e0b2463ac7.jpg)

<details>
<summary>line</summary>

| Lag ℓ: student step s vs cleaner step s + ℓ | 8 sampling steps | 16 sampling steps | 32 sampling steps |
| ------------------------------------------- | ---------------- | ----------------- | ----------------- |
| 1                                           | 1.0              | 2.0               | 2.5               |
| 2                                           | 0.7              | 1.7               | 2.2               |
| 3                                           | 0.5              | 1.5               | 2.0               |
| 4                                           | 0.3              | 1.3               | 1.8               |
</details>

Figure 11: Lagged logit analysis. (Left) output-head-projected hidden-state changes decrease as the number of sampling steps increases, for both the baseline and the $\mathcal { L } _ { \mathrm { C O N S } }$ model. (Right) relative reduction in lagged logit KL from ${ \mathcal { L } } _ { \mathrm { C O N S } }$ compared to the baseline, measured between a student denoising step s and a cleaner future step $s + { \ell } .$ The consistency-trained model reduces lagged logit KL across lags and sampling-step settings, with the strongest gains at smaller lags.

# F.7 Consistency Loss Training Dynamics

In practice, we find that extending the consistency stage for too long can degrade generation quality by over-sharpening the model. We therefore use validation perplexity as an early stopping signal: starting from the pre- $. \mathcal { L } _ { \mathrm { C O N S } }$ checkpoint, we select the first checkpoint whose validation perplexity exceeds the pre- $\bar { \mathcal { L } } _ { \mathrm { C O N S } }$ value by 15%. This rule gave the best empirical trade-off between generative perplexity and entropy. We linearly warm up the consistency term at the beginning of post-training to avoid an abrupt change in the optimization objective.

Figure 12 illustrates this behavior. Generative perplexity improves rapidly during early consistency training, but later checkpoints become increasingly over-sharpened, as shown by the sharp drop in sample entropy. The validation perplexity curve is not monotonic: after first increasing beyond the 15% threshold, it can later decrease again. However, these later decreases do not correspond to recovered sample diversity, so selecting a later checkpoint based only on validation perplexity would be misleading. We therefore use the first threshold crossing as a conservative stopping rule. This is qualitatively similar to the caution in SDTT [15], where repeated distillation rounds can accumulate approximation error because each student becomes the next teacher; in both cases, validation metrics and sample diversity should be monitored rather than relying only on the distillation loss.

![](images/e0c3cd31c58f83ce7751ba2e3bcdced094dd4d3a8e71c847897a669565175d23.jpg)

<details>
<summary>line</summary>

| Budget | 1,025,000 | 1,027,000 | 1,030,000 | 1,035,000 | 1,040,000 |
| ------ | --------- | --------- | --------- | --------- | --------- |
| 96     | 235       | 175       | 110       | 25        | 15        |
| 192    | 175       | 130       | 75        | 15        | 10        |
| 384    | 140       | 95        | 60        | 10        | 5         |
| 768    | 115       | 85        | 45        | 5         | 2         |
</details>

![](images/23cee05f419b1d16079c0d2874a06906dee8253a6b858d39992010d98eef7ff1.jpg)

<details>
<summary>line</summary>

| Checkpoint (+1000k) | Validation PPL |
| ------------------- | -------------- |
| 25                  | 27.9           |
| 30                  | 32.0           |
| 35                  | 54.3           |
| 42                  | 91.0           |
| 48                  | 65.2           |
| 53                  | 41.6           |
| 58                  | 37.0           |
| 63                  | 32.8           |
| 68                  | 30.2           |
| 73                  | 28.9           |
| 78                  | 28.7           |
| 80                  | 30.0           |
| 83                  | 35.3           |
| 86                  | 45.9           |
| 89                  | 98.1           |
| 93                  | 147.0          |
| 96                  | 106.8          |
| 99                  | 72.8           |
| 103                 | 47.9           |
| 106                 | 36.1           |
</details>

Figure 12: Checkpoint selection for the ${ \mathcal { L } } _ { \mathrm { C O N S } }$ post-training stage. (Left) Generative perplexity across budgets improves rapidly during early consistency training, but later checkpoints over-sharpen the model, as reflected by collapsing entropy values shown in parentheses. Sampling is done without warm-start (e.g. no reuse) (Right) Validation perplexity is not monotonic: it first rises above the 15% threshold, then can decrease again at later checkpoints. Since these later decreases do not recover sample diversity, we use the first checkpoint whose validation perplexity exceeds the pre-LCONS value by 15% as our stopping rule. This empirically balances generation quality and sample diversity.

# F.8 Ablations on the Consistency Loss Type

We ablate the form of the consistency objective for FP-MaskGIT on ImageNette. Starting from the same FP-MaskGIT checkpoint, we compare output-space consistency with KL on logits, representation-space consistency with MSE or cosine distance, and variants that use an EMA teacher. We also test whether adding this consistency loss at the beginning of the training improves the final model. For each loss, we evaluate both no-reuse and fixed-point reuse at the same sampling budgets.

Table 20 shows that representation-space losses are generally more effective than output-space KL for improving generation quality. In particular, hidden-state MSE with reuse gives the best FID at budgets 48 and 192, while hidden-state cosine with EMA gives the best FID at budget 96. These results support our choice of hidden-state consistency as the main regularizer: it provides a stronger training signal for the fixed-point representation than directly matching logits, and it interacts well with reuse during sampling.

After identifying the three best loss configurations on FP-MaskGIT, we transfer them to FP-MDLM, 120k training steps at 128 length, to test whether they also work in a different modality. We first run this ablation on FP-MaskGIT because it is cheaper to train and evaluate than FP-MDLM, and because its image-generation metrics are more direct than proxy metrics such as generative perplexity and unigram entropy. These configuration are: Hidden state MSE (no EMA), Hidden state cosine (with EMA), Hidden state MSE (with EMA) (Table 21). Once we found the best configuration on FP-MDLM, 120k training steps at 128 length, we test if it extends to longer sequences, i.e. 1024.

Table 20: Ablation on the consistency loss type for FP-MaskGIT. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reuse</td><td rowspan="2">Metric</td><td colspan="3">Budget</td></tr><tr><td>48</td><td>96</td><td>192</td></tr><tr><td rowspan="2">BaselineFP-MaskGIT + 10kno KL loss</td><td>No reuse</td><td>FID ↓IS ↑</td><td>102.304114.6703</td><td>56.908615.9080</td><td>32.349015.0480</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>101.505614.4204</td><td>55.258215.5139</td><td>31.714114.5876</td></tr><tr><td rowspan="2">Cross-step logit KLno EMA teacher</td><td>No reuse</td><td>FID ↓IS ↑</td><td>101.707814.5171</td><td>55.802416.0273</td><td>31.450014.8362</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>98.403214.5321</td><td>53.700216.0719</td><td>29.412815.0317</td></tr><tr><td rowspan="2">Cross-step logit KLEMA teacher</td><td>No reuse</td><td>FID ↓IS ↑</td><td>102.267814.0976</td><td>56.555416.0031</td><td>31.291014.8055</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>100.410514.2942</td><td>54.280715.9352</td><td>29.737714.5184</td></tr><tr><td rowspan="2">Latent proj cosineno EMA</td><td>No reuse</td><td>FID ↓IS ↑</td><td>112.226813.0748</td><td>55.346415.9799</td><td>30.590215.0260</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>102.439614.2156</td><td>54.382815.7187</td><td>30.494114.9456</td></tr><tr><td rowspan="2">Latent proj cosinewith EMA</td><td>No reuse</td><td>FID ↓IS ↑</td><td>110.721613.1563</td><td>54.810915.6277</td><td>32.329914.9067</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>102.560714.0288</td><td>53.855515.5334</td><td>30.691814.9674</td></tr><tr><td rowspan="2">Hidden state MSEno EMA</td><td>No reuse</td><td>FID ↓IS ↑</td><td>107.501213.3413</td><td>56.733616.3229</td><td>31.382815.0487</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>97.840115.0866</td><td>51.007716.2692</td><td>27.624214.5894</td></tr><tr><td rowspan="2">Hidden state MSEwith EMA</td><td>No reuse</td><td>FID ↓IS ↑</td><td>110.461113.4316</td><td>54.208015.5567</td><td>30.604414.9206</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>97.906614.6514</td><td>51.180816.1668</td><td>28.336714.5935</td></tr><tr><td rowspan="2">Hidden state cosineno EMA</td><td>No reuse</td><td>FID ↓IS ↑</td><td>110.458213.7846</td><td>56.608315.8706</td><td>31.120314.9106</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>98.708914.4814</td><td>54.567715.7316</td><td>29.512914.4452</td></tr><tr><td rowspan="2">Hidden state cosinewith EMA</td><td>No reuse</td><td>FID ↓IS ↑</td><td>112.579113.0173</td><td>54.775315.9610</td><td>29.742714.9954</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>98.309914.4018</td><td>50.251515.4198</td><td>28.788814.7025</td></tr><tr><td rowspan="2">Pretraining + hidden state MSEwithout EMA teacher</td><td>No reuse</td><td>FID ↓IS ↑</td><td>114.195314.0358</td><td>57.139816.1214</td><td>31.896015.4513</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>108.511014.4205</td><td>50.310116.0090</td><td>28.911314.8960</td></tr><tr><td rowspan="2">Pretraining + hidden state MSEwith EMA teacher</td><td>No reuse</td><td>FID ↓IS ↑</td><td>116.060713.3924</td><td>56.641916.1979</td><td>30.783015.0050</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>108.888814.3478</td><td>51.962715.9366</td><td>27.812914.7259</td></tr><tr><td rowspan="2">Pretraining + hidden state cosinewithout EMA teacher</td><td>No reuse</td><td>FID ↓IS ↑</td><td>112.385313.6743</td><td>55.599515.9413</td><td>32.659415.1818</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>104.708814.5433</td><td>53.066415.9216</td><td>31.066715.0648</td></tr><tr><td rowspan="2">Pretraining + hidden state cosinewith EMA teacher</td><td>No reuse</td><td>FID ↓IS ↑</td><td>118.523013.3390</td><td>57.934716.1273</td><td>32.661715.2384</td></tr><tr><td>With reuse</td><td>FID ↓IS ↑</td><td>110.308713.9843</td><td>53.650516.0756</td><td>30.941615.3464</td></tr></table>

Table 21: Generation quality across budgets on OWT, 120k training steps, 128 sequence length. Baseline is 120k training steps. Consistency post training uses 100k pretraining steps followed by 20k post-training adaptation steps. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Reuse</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="2">BaselineFP-MDLM @120k</td><td>No reuse</td><td>Gen PPL ↓Entropy ↑</td><td>421.58194.4004</td><td>337.10574.3862</td><td>262.69824.3688</td><td>245.19794.3571</td></tr><tr><td>With reuse</td><td>Gen PPL ↓Entropy ↑</td><td>425.29544.4033</td><td>310.07464.3931</td><td>253.14284.3651</td><td>225.96414.3455</td></tr><tr><td rowspan="2">Hidden state MSEw/o EMA teacher</td><td>No reuse</td><td>Gen PPL ↓Entropy ↑</td><td>433.91534.4047</td><td>338.95484.3851</td><td>274.90604.3782</td><td>227.52964.3658</td></tr><tr><td>With reuse</td><td>Gen PPL ↓Entropy ↑</td><td>385.64674.3992</td><td>266.60364.3763</td><td>226.65004.3668</td><td>215.44394.3705</td></tr><tr><td rowspan="2">Hidden state cosinew/ EMA teacher</td><td>No reuse</td><td>Gen PPL ↓Entropy ↑</td><td>435.80424.4051</td><td>334.92154.3832</td><td>277.84214.3791</td><td>243.61874.3618</td></tr><tr><td>With reuse</td><td>Gen PPL ↓Entropy ↑</td><td>418.23754.4014</td><td>306.88424.3841</td><td>249.71644.3681</td><td>225.90324.3673</td></tr><tr><td rowspan="2">Hidden state MSEw/ EMA teacher</td><td>No reuse</td><td>Gen PPL ↓Entropy ↑</td><td>436.92254.4063</td><td>332.73364.3819</td><td>279.96044.3804</td><td>251.27124.3579</td></tr><tr><td>With reuse</td><td>Gen PPL ↓Entropy ↑</td><td>426.54674.4022</td><td>317.16264.3878</td><td>254.62324.3692</td><td>228.47494.3661</td></tr></table>

# F.9 Tradeoff Between Denoising Steps and Fixed-Point Iterations

This sweep studies how generation quality changes when compute is allocated differently between denoising steps and fixed-point iterations. We measure this tradeoff on $\mathrm { F P - M D L M } + \mathcal { L } _ { \mathrm { C O N S } } + 3 \mathrm { S R }$ , which corresponds to our CoFRe model. The heatmap shows that these two sources of compute are not interchangeable: using very few denoising steps gives poor generative perplexity even when many FP iterations are used, while increasing the number of denoising steps substantially improves sample quality. Once enough denoising steps are used, additional FP iterations still help, but with smaller gains. The entropy heatmap shows that the best generative perplexity values are not obtained by collapsing sample diversity, since entropy remains in a similar non-degenerate range across the strongest configurations. Overall, the sweep indicates that the quality–diversity tradeoff depends on how inference compute is split between outer denoising and inner fixed-point solving, supporting the use of non-uniform depth allocation at sampling time.

![](images/1a743ca283bc3a84662fc8a810a2f3b996a9c435d191f7114048328297989288.jpg)

<details>
<summary>heatmap</summary>

| Number of FP Iterations K | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Number of Steps T | 54.6 | 61.0 | 37.8 | 39.4 | 65.7 | 65.2 | 101.8 | 135.0 | 117.2 | 144.3 | 259.3 |
| Gen PPL | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 |
| *8=388 | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 | 3.25 |
| *8=768 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 | 3.25 | 3.50 |
| *8=96 | 1.25 | 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 | 3.25 | 3.50 | 3.75 |
| *8=192 | 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 | 3.25 | 3.50 | 3.75 | 4.00 |
| *8=96 (approximate) | 144.3 | 136.1 | 106.7 | 72.9 | 102.8 | 135.5 | 135.8 | 135.8 | 135.8 | 135.8 | 135.8 |
| *8=96 (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate) (approximate).
</details>

![](images/6612ba481c56855641b18e043a7d58df748406d64bac5eb28abb82d5c84708a0.jpg)  
Figure 13: Tradeoff between denoising steps and fixed-point iterations. We sweep the number of denoising steps and FP iterations for FP-MDLM with consistency regularization and three-state reuse. The left heatmap reports generative perplexity, where lower is better, and the right heatmap reports sample entropy. Each annotated cell corresponds to one evaluated allocation; blank cells were not evaluated. Allocating compute to very few denoising steps leads to poor generative perplexity even with many FP iterations, whereas configurations with sufficiently many denoising steps and moderate FP depth achieve the best quality while maintaining non-collapsed entropy.

# F.10 Different budget allocation strategies

We also ablate how the fixed-point iteration budget should be distributed across denoising steps for FP-MDLM+LCONS. Keeping the checkpoint and total forward-pass budget fixed, we compare five allocation strategies: fixed, decreasing, increasing, cosine, and front-loaded schedules. We evaluate each schedule under the three initialization regimes used throughout the paper: no reuse, full reuse, and 3SR. Table 22 and Figure 14 show that decreasing schedules perform best overall, with the lowest mean generative perplexity, the best average rank, and the most wins across initialization-budget settings. Fixed allocation is a strong second, while increasing and cosine schedules consistently underperform. This suggests that FP-MDLM benefits from spending more fixed-point computation early in the denoising trajectory, when the sequence is most corrupted and the denoising problem is hardest.

Table 22: Overall comparison of denoising strategies. Lower Gen. PPL and lower average rank are better. Wins count the number of init-budget settings where the strategy obtains the best Gen. PPL. More than 12 wins are possible in total because ties count as wins for all tied strategies. 

<table><tr><td>Strategy</td><td>Mean Gen. PPL ↓</td><td>Avg. rank ↓</td><td>Wins</td></tr><tr><td>Decreasing</td><td>64.65</td><td>1.33</td><td>8 / 12</td></tr><tr><td>Fixed</td><td>65.56</td><td>2.00</td><td>5 / 12</td></tr><tr><td>Front loaded</td><td>65.69</td><td>2.42</td><td>2 / 12</td></tr><tr><td>Increasing</td><td>73.10</td><td>4.08</td><td>0 / 12</td></tr><tr><td>Cosine</td><td>73.84</td><td>4.58</td><td>0 / 12</td></tr></table>

![](images/4bc236058fe3d7578485dc3b2f427be869d6eb3ddfc8cd2994d0c9b113d53d4f.jpg)

<details>
<summary>line</summary>

| Budget | Fixed | Increasing | Decreasing | Cosine | Front loaded |
| ------ | ----- | ---------- | ---------- | ------ | ----------- |
| 96     | 5.46  | 5.48       | 5.43       | 5.43   | 5.45        |
| 192    | 5.39  | 5.38       | 5.43       | 5.43   | 5.39        |
| 384    | 5.34  | 5.37       | 5.37       | 5.37   | 5.34        |
| 768    | 5.16  | 5.17       | 5.18       | 5.18   | 5.16        |
</details>

(a) No Reuse

![](images/09ffae29fc3adf59aeeb8142dd77f9493217b7b7166c09d9d46a04328f606c90.jpg)

<details>
<summary>line</summary>

| Budget | Fixed | Increasing | Decreasing | Cosine | Front loaded |
| ------ | ----- | ---------- | ---------- | ------ | ----------- |
| 96     | 120   | 120        | 120        | 120    | 120         |
| 192    | 60    | 60         | 60         | 60     | 60          |
| 384    | 40    | 40         | 40         | 40     | 40          |
| 768    | 30    | 30         | 30         | 30     | 30          |
</details>

(b) Reuse

![](images/b569116d66010340b4884866f49cb875d30027cc6f986422087a77fb576ef538.jpg)

<details>
<summary>line</summary>

| Budget | Fixed | Increasing | Decreasing | Cosine | Front loaded |
| ------ | ----- | ---------- | ---------- | ------ | ------------ |
| 96     | 100   | 125        | 100        | 100    | 100          |
| 192    | 75    | 75         | 75         | 75     | 75           |
| 384    | 50    | 50         | 50         | 50     | 50           |
| 768    | 40    | 40         | 40         | 40     | 40           |
</details>

(c) 3SR   
Figure 14: Generative perplexity as a function of budget for different denoising strategies. Entropy values are shown in parentheses. We observe that decreasing schedules perform best overall.

# F.11 Latency and generation quality for language modeling

We measure generation-only sampling latency, defined as the wall-clock time from fully masked token IDs to final generated token IDs. The timed region includes all denoising-loop computation: model forward passes, fixed-point iterations, reuse/CoFRe (3SR) initialization, mask-confidence updates, categorical sampling, and loop logic. We synchronize CUDA before and after sampling, and exclude decoding, external Gen. PPL evaluation, file I/O, model loading, and warmup/compilation effects. All methods are evaluated under matched transformer-block budgets with the same ancestral-cache sampler, batch size, sequence length, precision, tokenizer, device, and number of samples. We use two warmup batches followed by 20 timed batches.

Table 23 and Figure 15 show that CoFRe is modestly slower than MDLM+SDTT at equal transformerblock budget, with slowdowns between 1.12× and 1.45×, but achieves substantially better generation quality. For example, at budget 96, CoFRe reduces Gen. PPL from 193.05 to 101.791 with latency increasing from 1.71s to 2.47s. At budget 384, it improves Gen. PPL from 62.29 to 48.755 with only a 1.15× slowdown. Overall, CoFRe is not faster than MDLM+SDTT at equal budget in the current implementation, but provides a better latency–quality trade-off, reaching Gen. PPL below 100 in 4.40-14.01s, while MDLM+SDTT does not reach this quality at any tested budget.

Table 23: Generation-only sampling latency and quality on OWT. Latency is measured from fully masked token IDs to final generated token IDs, excluding tokenizer decoding and external Gen. PPL evaluation. 

<table><tr><td rowspan="2">Budget</td><td colspan="2">Latency (s) ↓</td><td rowspan="2">SlowdownCoFRe / MDLM+SDTT</td><td colspan="2">Gen. PPL ↓</td></tr><tr><td>MDLM+SDTT</td><td>CoFRe</td><td>MDLM+SDTT</td><td>CoFRe</td></tr><tr><td>96</td><td>1.71</td><td>2.47</td><td>1.45×</td><td>193.05</td><td>101.791</td></tr><tr><td>192</td><td>3.29</td><td>4.40</td><td>1.34×</td><td>89.17</td><td>65.182</td></tr><tr><td>384</td><td>6.39</td><td>7.34</td><td>1.15×</td><td>62.29</td><td>48.755</td></tr><tr><td>768</td><td>12.51</td><td>14.01</td><td>1.12×</td><td>47.04</td><td>37.846</td></tr></table>

![](images/4bc3b170d105ba3a4a45492e0e8be2a1f1d93619787a4b91619a5f129fab2a9e.jpg)

<details>
<summary>line</summary>

| Generation latency (s) | MDLM+SDTT | CoFRe |
| ---------------------- | --------- | ----- |
| 2                      | 96        | 96    |
| 4                      | 192       | 192   |
| 6                      | 384       | 384   |
| 12                     | 768       | 768   |
</details>

Figure 15: Generation quality as a function of wall-clock sampling latency on OWT. We report generation-only latency, measured from fully masked token IDs to final generated token IDs, excluding decoding and external Gen. PPL evaluation. Points are annotated by their transformer-block budget. CoFRe is modestly slower than MDLM+SDTT at matched budget, but reaches substantially lower Gen. PPL at lower wall-clock latency in the low- and medium-budget regimes.

# F.12 Fixed-point residual analysis.

We measure the relative hidden residual

$$
r _ {t} ^ {(n)} = \frac {\| F _ {\theta} (h _ {t} ^ {(n)} ; \tilde {h} _ {t} , t) - h _ {t} ^ {(n)} \| _ {2}}{\| h _ {t} ^ {(n)} \| _ {2}}
$$

across FP iterations and denoising steps. As shown in Figure 16, residuals decrease steadily with iterations, confirming that the repeated block behaves as an iterative fixed-point solver. Without reuse, the mean residual drops from 75.97 to 0.0180 after four iterations; with full reuse, it starts much closer and drops from 0.317 to 0.00545. Excluding the first step, full reuse reduces the median initial residual from 76.68 to 0.0385, i.e. it starts about 1984× closer to the current fixed point. These results validate the solver and the benefit of warm starts. However, lower residual does not necessarily imply better generation: full reuse is numerically closest to equilibrium, but can over-reuse stale states, motivating the token-aware 3SR rule.

![](images/81b70178dd7a3a4c6fba484230559999caec1e328dd8dc7ae9115098f771b8ab.jpg)

<details>
<summary>line</summary>

| FP iteration | No reuse | Full reuse |
| ------------ | -------- | ---------- |
| 0            | 100.0    | 0.3        |
| 1            | 0.2      | 0.03       |
| 2            | 0.05     | 0.01       |
| 3            | 0.02     | 0.005      |
| 4            | 0.01     | 0.002      |
</details>

![](images/b4bc0d2e54c7138646f0db012fe79d920a9bd14996187b94b473afd5fc8a0000.jpg)

<details>
<summary>line</summary>

| Denoising step | No reuse | Full reuse | Initial | Final |
| -------------- | -------- | ---------- | ------- | ----- |
| 0              | 100      | 10         | 100     | 100   |
| 5              | 0.01     | 0.01       | 0.01    | 0.01  |
| 10             | 0.01     | 0.01       | 0.01    | 0.01  |
| 15             | 0.01     | 0.01       | 0.01    | 0.01  |
| 20             | 0.01     | 0.01       | 0.01    | 0.01  |
| 25             | 0.01     | 0.01       | 0.01    | 0.01  |
| 30             | 0.01     | 0.01       | 0.01    | 0.01  |
| 35             | 0.01     | 0.01       | 0.01    | 0.01  |
| 40             | 0.01     | 0.01       | 0.01    | 0.01  |
| 45             | 0.01     | 0.01       | 0.01    | 0.01  |
| 50             | 0.01     | 0.01       | 0.01    | 0.01  |
| 55             | 0.01     | 0.01       | 0.01    | 0.01  |
| 60             | 0.01     | 0.01       | 0.01    | 0.01  |
| 65             | 0.01     | 0.01       | 0.01    | 0.01  |
</details>

Figure 16: Fixed-point residual analysis. (Left) Mean relative residual decreases with FP iterations, showing that the repeated block approaches a fixed point. Full reuse starts much closer to the solution than no reuse. (Right) Across denoising steps, reuse strongly reduces the initial residual and yields lower final residuals under the same iteration budget. Residuals validate the solver and warm-start mechanism, while generation ablations are needed to compare full reuse and 3SR quality.

# F.13 Effect of Training Duration on Generation Quality

In this section, we compare the effect of the training duration on generation quality. We compare here MDLM trained at 100k steps, MDLM trained at 1M steps (checkpoint from [57]), FP-MDLM trained at 100k steps, FP-MDLM trained at 1M steps and FP-MDLM trained at 100k + LCONS +3SR (so basically CoFRe at 100k training steps). For all the models, we evaluate both the generative perplexity and the unigram entropy. We report the results in Table 24.

Table 24: Training-stage comparison for MDLM and FP-MDLM generation on OpenWebText. We report generative perplexity and entropy across sampling budgets. The table compares MDLM and FP-MDLM checkpoints at different training stages, together with the FP-MDLM checkpoint trained with cross-step consistency regularization and evaluated with three-state reuse. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">Budget</td></tr><tr><td>96</td><td>192</td><td>384</td><td>768</td></tr><tr><td rowspan="2">MDLM@100k</td><td>Gen PPL ↓</td><td>642.425</td><td>277.504</td><td>177.962</td><td>134.952</td></tr><tr><td>Entropy ↑</td><td>5.866</td><td>5.779</td><td>5.730</td><td>5.693</td></tr><tr><td rowspan="2">MDLM</td><td>Gen PPL ↓</td><td>830.820</td><td>343.330</td><td>196.790</td><td>143.880</td></tr><tr><td>Entropy ↑</td><td>5.910</td><td>5.810</td><td>5.750</td><td>5.700</td></tr><tr><td rowspan="2">FP-MDLM@100k</td><td>Gen PPL ↓</td><td>379.484</td><td>268.113</td><td>213.801</td><td>180.315</td></tr><tr><td>Entropy ↑</td><td>5.790</td><td>5.755</td><td>5.722</td><td>5.694</td></tr><tr><td rowspan="2">FP-MDLM</td><td>Gen PPL ↓</td><td>375.631</td><td>273.275</td><td>215.197</td><td>179.655</td></tr><tr><td>Entropy ↑</td><td>5.810</td><td>5.763</td><td>5.726</td><td>5.702</td></tr><tr><td rowspan="2">FP-MDLM @100k+  $\mathcal{L}_{\text{CONS}} + 3SR$ </td><td>Gen PPL ↓</td><td>96.802</td><td>61.891</td><td>49.669</td><td>37.717</td></tr><tr><td>Entropy ↑</td><td>5.519</td><td>5.411</td><td>5.409</td><td>5.246</td></tr></table>

We observe that the generative quality after 100k steps closely match those at 1M steps, suggesting that 100k steps are likely sufficient to test new algorithms or framework when pretraining a model. If a model fails by that time, it is unlikely to succeed at 1M steps. Further analysis is needed to determine whether this reflects limitations of the metrics (generative perplexity and entropy may not capture differences at that scale) or constraints imposed by model capacity. Pynadath et al. [54] observes similar results at 50k steps when analyzing the generative frontiers of these models.