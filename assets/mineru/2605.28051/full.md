# BEYOND SURROGATE GRADIENTS: FULLY DIFFERENTIABLETOKEN PRUNING FOR VISION-LANGUAGE MODELS

Landi He

Shenzhen University of Advanced Technology

Mingde Yao

CUHK MMLab, CPII under InnoHK

Shawn Young

Shenzhen University of Advanced Technology

Lijian Xu ∗

Shenzhen University of Advanced Technology

xulijian@suat-sz.edu.cn

# ABSTRACT

Visual token pruning reduces the computational cost of Vision-Language Models (VLMs) by removing redundant visual tokens. Existing methods typically rely on Gumbel-Softmax to approximate discrete selection during training. However, the optimization is driven by surrogate gradients rather than the true selection process, leading to unreliable learning of token importance. In this paper, we propose DiffPrune, which reformulates pruning as continuous control of token information instead of discrete selection learning. Specifically, we introduce an Information Throttler that modulates each token using variance-preserving noise conditioned on importance scores, where higher scores induce less information suppression during training. This design directly operates on token representations, naturally providing a fully differentiable optimization path for learning token importance. At inference, tokens are removed via hard thresholding on the learned scores. Across ten VLM benchmarks, DiffPrune retains 96.5% of full-model accuracy while accelerating LLM prefill by 2.85×, with only 0.69 ms of inference overhead.

# 1 Introduction

Modern vision-language models (VLMs) suffer substantial inference costs due to the large number of visual tokens introduced by high-resolution images and videos [1]. To improve efficiency, recent studies have explored various approaches, including token compression, adaptive computation, and visual-token pruning [2, 3, 4]. Among them, visual-token pruning offers a structurally direct form of efficiency improvement, as it reduces sequence length without introducing additional latent representations or auxiliary decoding stages[5]. However, effective pruning requires a reliable token importance scorer, which is non-trivial in practice.

Existing visual-token pruning methods can be broadly divided into training-free and training-based approaches depending on whether the token importance scorer is learned. Training-free methods use heuristic signals from pretrained backbones (e.g., attention or feature redundancy) to estimate token importance [7, 8, 9, 10], requiring no additional training and enabling direct inference-time deployment, but they are often misaligned with downstream tasks and lack adaptability across models [11, 12]. In contrast, training-based methods [13, 14, 15] learn a scorer to estimate token importance. To enable optimization, they rely on relaxations such as Gumbel-Softmax (GS) or straight-through estimators (STE), which replace the discrete selection process with a differentiable surrogate. However, this surrogate does not match the true discrete pruning operation, so the training signal is based on surrogate gradients rather than the real pruning objective.

As shown in Figure 1(a), existing pipelines convert continuous importance scores into binary keep-or-drop decisions under a fixed budget, making the pruning operation inherently non-differentiable. GS and STE are used to approximate gradients through this discrete process, but they optimize a relaxed continuous surrogate rather than the original discrete selection problem [16, 17]. This mismatch between the surrogate objective and the true pruning objective leads to unstable optimization dynamics [18]. As illustrated in Figure 1(c), the resulting gradients exhibit high variance across training iterations, since small changes in importance scores can induce abrupt changes in the selected token subset. This effect becomes more pronounced as model size increases, where more tokens lie near the selection boundary.

![](images/7ab6a597c5df780ecce8ca153828651941970decb0f1c4a33eda2fab61ef5109.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input ∈ {0,1}"] --> B["0 0 1 0 1 1"]
    B --> C["0.1 0.4 0.9"]
    C --> D["Score"]
    style A fill:#f9f,stroke:#333
    style D fill:#bbf,stroke:#333
```
</details>

![](images/11c58d122b78e73a6241b1a0255ec3e34ab0af713bed4ff9ea534329c9e4cf20.jpg)

<details>
<summary>line</summary>

| Input | Fully Gradient |
|-------|----------------|
| 0.1   | 0.1            |
| 0.4   | 0.4            |
| 0.9   | 0.9            |
| 0.1   | 0.1            |
| 0.8   | 0.8            |
| 0.7   | 0.7            |
</details>

![](images/cebc4688b80ddcd60f0c10356a396bcff7cb254546adb4ee0dd77cd70f16c0d4.jpg)

<details>
<summary>bar</summary>

| Model | Gumbel-Softmax | DiffPrune(Ours) |
| :--- | :--- | :--- |
| deit-tiny | 0.04 | 0.19 |
| deit-small | 0.015 | 0.42 |
| deit-base | 0.01 | 0.36 |
4.4x (diffPrune(Ours)) 25.5x (diffPrune(Ours)) 28.4x
</details>

Figure 1: (a)&(b) Gumbel-Softmax-style pruning vs. DiffPrune. (c) DiffPrune demonstrates more coherent direction on DeiT backbone [6], which enables fully differentiable optimization.

In contrast, our method, as illustrated in Figure 1 (b), avoids optimizing through discrete relaxation. Instead, it maintains a continuous importance formulation throughout training and directly modulates token information flow under a fixed budget constraint. This yields a smoother optimization landscape and significantly improves gradient consistency. Importantly, our formulation remains fully differentiable throughout training, leading to substantially higher gradient alignment (28.4× on DeiT-Base).

In this paper, we propose DiffPrune, a fully differentiable visual-token pruning framework for VLMs. During training, every operator on the path from scorer logits to language-model loss has a backward gradient given by the analytic derivative of the same continuous function used in the forward pass. The gradient path uses no surrogate gradients, argmax, hard selection, or straight-through estimators. During training, DiffPrune does not make binary keep-or-drop decisions, but instead changes the amount of information carried by each visual token in a continuous way. A Soft Top-K head polarises the scorer logits into continuous weights and constrains their total budget to match hard top-K. The Information Throttler has two components. First, a variance-preserving (VP) noise injector uses these weights to modulate each token. Second, a trainable denoiser maps the noised tokens back to the representation space expected by the downstream LLM. At inference, Soft Top-K is replaced by hard top-K, the denoiser is removed, and only the scorer is added to the inference path. The base VLM remains frozen throughout.

Contributions can be concluded as follows.

• We show that the limitation of surrogate-gradient pruning lies at the operator level rather than the scorer level, and propose DiffPrune as a fully differentiable framework for visual-token pruning.   
• We introduce VP-Noise, a diffusion-inspired Throttler that interpolates each token with Gaussian noise according to its retention weight, suppressing token content without relying on a discrete keep-or-drop operation.   
• In controlled DeiT probes, DiffPrune reaches up to 28.4× higher cross-batch gradient-direction coherence than Gumbel-Softmax; across LLaVA-1.5-7B, LLaVA-NEXT-7B, and Qwen2.5-VL-7B evaluations, it preserves high task performance under aggressive pruning while adding only 0.69 ms of inference overhead.

# 2 Related Work

# 2.1 Training-Free Visual Token Compression

Training-free compression keeps the VLM backbone fixed and score tokens using signals already present in the model. We organize this line by the source of the token score [19, 20]. Attention-based methods use saliency from the vision tower or the language decoder [7, 8, 21, 22, 23, 24, 25]. Redundancy-based methods remove or merge visually similar features [9, 26, 10, 27, 28]. Coverage-based methods keep tokens that are diverse or spatially well spread [11, 29, 30, 31, 32]. These methods are strong plug-in baselines because they do not train a new selector. While strong as plug-in baselines, these methods do not adapt their compression rules to downstream language-model loss. Learned pruning instread must convert continuous scores into discrete keep-or-drop decisions while enabling gradient flow.

# 2.2 Learning-based- Token Selection

Learned pruning must train the steps. A common solution is surrogate-gradient pruning. The forward pass uses a discrete or nearly discrete mask, while the backward pass differentiates a smooth relaxation. This pattern appears in semi-structured weight sparsity, KV-cache eviction, and text-side token pruning [33, 34, 35, 36]. These methods use different relaxations, including Gumbel-Softmax with STE, differentiable sorting, and hard-concrete gates, and compress different objects. The core issue remains: the score-to-mask function in the forward pass is not the one used for backpropagation.

Learned visual-token pruning follows the same pattern. DynamicViT [37] samples token survival with Gumbel-Softmax and passes gradients through the soft relaxation. ATP-LLaVA [38] uses a temperature-sharpened sigmoid-threshold mask. Dynamic-LLaVA [39] and LightVLA [40] use binary or argmax-like selections with Gumbel-Softmax-style backward paths. These methods differ in scorer architecture, pruning location, and budget control, but are similar in how continuous scores become selected tokens.

Recent variants point to the same bottleneck. Shiva-DiT [41] replaces the softmax surrogate with a residual-aware STE in diffusion transformers. TwigVLM++ [42] trains a pruning head through distillation and policy-gradient reinforcement learning. These routes either keep a discrete forward selection with a surrogate backward path, or avoid analytic differentiation through selection. DiffPrune instead uses a continuous forward operator, whose backward gradients are the analytic derivatives of the same function. Section 3 formalises this contrast at the operator and loss-landscape levels.

# 3 Pilot Studies

# 3.1 Limitations of Gumbel-Softmax Pruning

The main problem of surrogate-gradient pruning lies in the discrete selection operator itself, rather than only in the scorer design. Let the logits output by the scorer be $s \in \mathbb { R } ^ { N }$ and Gumbel perturbation be g. In the forward pass, the model generates a hard mask via top-K; in the backward pass, the gradient is usually passed directly from the mask m back to the logits s:

$$
\text {forward:} m = \operatorname{topK} (s + g) \in \{0, 1 \} ^ {N},
$$

$$
\text { backward }: \quad \frac {\partial L}{\partial s} \leftarrow \frac {\partial L}{\partial m}. \tag {1}
$$

The forward computation and the backward gradient here do not correspond to the same differentiable function. The top-K selection actually executed in the forward pass is piecewise constant with respect to s: as long as the ordering of the elements does not change, the retained token set remains unchanged, and therefore its derivative is almost everywhere zero [43, 44]. Only when a score crosses the K-th order statistic and triggers an index swap does the output mask change discontinuously. In contrast, the backward pass does not use the true Jacobian of this hard selection operator. Instead, it approximates that Jacobian with the Jacobian of the identity map, directly passing ∂L/∂m to s. Therefore, the gradient is not from the actual forward computation but a surrogate estimate, introducing a systematic gradient bias [45] rather than unbiased random noise.

# 3.2 A Gradient-Continuous Alternative

Based on the above analysis, we propose an improved scheme: route every token through one analytic function whose value is the forward and whose Jacobian is the backward. Rather than ask whether token i is kept, we ask how much of its information is admitted, encode the answer as $\alpha _ { i } \in [ 0 , 1 ]$ , and apply a gradient-continuous Throttler $T ( x _ { i } ; \alpha _ { i } )$

to every token, with α supplied by a Soft Top-K head under $\begin{array} { r } { \sum _ { i } \alpha _ { i } \approx K ; } \end{array}$ unlike the soft companion in Eq. (1), Soft Top-K is the operator the forward executes. The probe below instantiates T as a per-token scale composed with variance-preserving Gaussian mixing, with the design unpacked in Section 4.

We visualize the scorer loss landscape with the filter-normalized two-dimensional slice of Li et al. [46]. The probe inserts a two-layer MLP scorer with roughly 14M parameters between the patch embedding layer and the frozen DeiT-base encoder. The patch embedding layer outputs N = 196 patch tokens, and the pruning ratio removes 70% of them, i.e., K = 58. We compare Gumbel-softmax, which sends the hard top-K tokens to the encoder with an STE backward pass, against DiffPrune, which applies Soft Top-K scores through a continuous token throttler.

(a) DiffPrune (Ours)   
![](images/849097fa82f0891fa3f9b0e336c6066958158977a3e603ed935b6c6a42222022.jpg)

<details>
<summary>heatmap</summary>

| direction d1 | direction d2 | value  |
| ------------ | ------------ | ------ |
| (various)    | (various)    | 4.12   |
| (various)    | (various)    | 4.20   |
| (various)    | (various)    | 4.24   |
</details>

(b) Gumbel-Softmax   
![](images/db7e65521b545cf2291e9d10891351336244f02b1c173d53be34b8267abf877b.jpg)

<details>
<summary>heatmap</summary>

| direction d₁ | value  |
| ------------ | ------ |
| θ_init       | 3.30   |
</details>

(c) Local loss variation along d2   
![](images/64e1c9ac72983ffcd85873529ba371567cb4a06d79415ce0cdf4114506f18d49.jpg)

<details>
<summary>line</summary>

| direction d₂ | DiffPrune (Ours) | Gumbel-Softmax |
| ------------ | ---------------- | -------------- |
| -0.3         | ~10⁻³            | ~10⁻²          |
| -0.2         | ~10⁻³            | ~10⁻²          |
| -0.1         | ~10⁻³            | ~10⁻²          |
| 0.0          | ~10⁻³            | ~10⁻²          |
| 0.1          | ~10⁻³            | ~10⁻²          |
| 0.2          | ~10⁻³            | ~10⁻²          |
| 0.3          | ~10⁻³            | ~10⁻²          |
</details>

Figure 2: Gradient direction visualization. Our fully differentiable DiffPrune yields smoother and more coherent gradient directions than Gumbel-Softmax-based methods, facilitating more effective optimization.

Figure 2(a) shows that DiffPrune yields a smooth loss surface with gradual contour changes. In contrast, Figure 2(b) exhibits more irregular loss contours with sharper local variations. Figure 2(c) makes this difference more explicit by plotting the stepwise loss variation along the $d _ { 2 }$ axis, measured as $| L ( \bar { \beta _ { i + 1 } } ) - L ( \beta _ { i } ) |$ . The mean stepwise loss variation of Gumbel-softmax is roughly 24× that of DiffPrune. These results suggest that the discontinuity introduced by hard token selection and STE is not only a backward-pass mismatch, but also appears directly in the scorer optimization landscape. By replacing discrete selection with continuous throttling during training, DiffPrune provides a smoother objective for optimizing the pruning scorer.

# 4 Methodology

# 4.1 Overview

DiffPrune builds a fully differentiable framework for visual-token pruning in VLMs. Figure 3 gives the overview: DiffPrune trains a token scorer through a continuous score-to-loss path and deploys it as a hard top-K pruner. Given an image I, the frozen vision encoder produces visual tokens $\mathbf { X } ^ { v } = \mathcal { E } _ { v } ( \mathbf { I } ) \in \mathbb { R } ^ { N \times d _ { v } }$ . The framework consists of a Scorer $S _ { \theta } .$ , a budgeted Soft Top-K map $\Phi _ { K }$ , a token-wise Throttler T , and a train-only Denoiser $\mathcal { D } _ { \phi }$ . The base VLM is frozen; only θ and ϕ are optimized.

Let W denote the text tokens, pΘ the frozen language model, and Z the multimodal sequence consumed by it. Training is written as

$$
\min _ {\theta , \phi} - \sum_ {t} \log p _ {\Theta} (y _ {t} ^ {*} \mid y _ {<   t} ^ {*}, \mathbf {Z})  , \tag {2}
$$

$$
\mathbf {Z} = \left[ \mathbf {P} _ {v \to \ell} (\bar {\mathbf {X}} ^ {v}); \mathcal {E} _ {t} (\mathbf {W}) \right],
$$

where $\bar { \mathbf { X } } ^ { v } = \mathcal { D } _ { \phi } ( T ( \mathbf { X } ^ { v } ; \pmb { \alpha } ) )$ and $\pmb { \alpha } = \Phi _ { K } ( S _ { \theta } ( \mathbf { X } ^ { v } ) / \tau )$ with $\textstyle \sum _ { i } \alpha _ { i } = K$ . Here K is the token budget, i.e., the target number of visual tokens retained by the deployed pruner; during training, it is represented as the total retention mass $\sum _ { i } \alpha _ { i }$ . Hard token selection is absent from Eq. (2); the scorer is trained through the continuous operators that are actually executed in the forward pass.

![](images/ef91f703e457b21eb59900fb4617565a3985a3d20af7a77f4dcc78c31e79d8f5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input image"] --> B["Visual Encoder"]
    C["Question"] --> D["Tokenizer"]
    B --> E["Scorer"]
    D --> F["Tokenizer"]
    E --> G["X^v"]
    E --> H["Global-Attention"]
    E --> I["S"]
    E --> J["Soft Top-K"]
    F --> K["α"]
    F --> L["Noise Gate"]
    F --> M["Denoiser"]
    G --> N["0.7"]
    G --> O["0.6"]
    G --> P["1"]
    G --> Q["0.9"]
    G --> R["0.85"]
    H --> S["0.27"]
    H --> T["0.13"]
    H --> U["0.9"]
    H --> V["0.75"]
    H --> W["0.85"]
    I --> X["X^v"]
    I --> Y["Diagonal-Attention"]
    I --> Z["..."]
    I --> AA["..."]
    I --> AB["..."]
    I --> AC["..."]
    I --> AD["..."]
    I --> AE["..."]
    I --> AF["..."]
    I --> AG["..."]
    I --> AH["..."]
    I --> AI["LLM"]
    J --> E
    K --> G
    L --> H
    M --> I
    N --> G
    O --> H
    P --> I
    Q --> I
    R --> I
    S --> G
    T --> H
    U --> I
    V --> I
    W --> I
    X --> I
    Y --> I
    Z --> I
    AA --> I
    AB --> I
    AC --> I
    AD --> I
    AE --> I
    AF --> I
    AG --> I
    AH --> I
```
</details>

Figure 3: Overview of DiffPrune. Visual tokens are first assigned importance scores by a Scorer. Soft Top-K transforms these scores into continuous retention weights satisfying a fixed budget constraint. Rather than making discrete keep-or-drop decisions, DiffPrune uses these weights to continuously throttle token information through a Noise Gate, where less important tokens receive stronger perturbations. A train-only Denoiser reconstructs the resulting representations before they are processed by the frozen LLM. At inference, DiffPrune collapses to standard hard top-K token selection using the learned importance scores.

At inference, DiffPrune uses the trained scorer as a ranking function and gathers the original top-K visual tokens:

$$
\hat {\mathbf {X}} ^ {v} = \operatorname{TopK} (\mathbf {X} ^ {v}; \mathcal {S} _ {\theta} (\mathbf {X} ^ {v}), K) \in \mathbb {R} ^ {K \times d _ {v}}. \tag {3}
$$

Here TopK(X; s, K) denotes score-indexed token gathering. The retained tokens keep their original order and position indices. The Throttler and Denoiser are removed from the deployed graph, leaving one scorer pass plus this gather operation.

# 4.2 Scorer and Budgeted Soft Top-K

The Scorer $\scriptstyle { S _ { \theta } }$ is a two-layer Transformer encoder followed by a linear head that assigns one scalar logit $s _ { i }$ to each visual token $\mathbf { x } _ { i } ^ { v } .$ . It predicts only an ordering signal; budget control is delegated to $\Phi _ { K }$ . We then apply a budgeted Soft Top-K map, a differentiable top-K operator that returns continuous retention weights under a fixed cardinality budget [47]:

$$
\boldsymbol {\alpha} = \Phi_ {K} (\mathbf {s} / \tau) \in [ 0, 1 ] ^ {N}, \quad \sum_ {i = 1} ^ {N} \alpha_ {i} = K. \tag {4}
$$

The budget is enforced by $\Phi _ { K }$ , so no auxiliary budget loss is used. The temperature τ controls sharpness: cosine annealing lowers τ during training, making α increasingly close to a binary top-K mask. With no score tie at the K-th boundary, the low-temperature limit matches the hard top-K ranking used at inference in Eq. (3). During training, the downstream VLM consumes α itself, and the backward pass differentiates the same continuous map used in the forward pass. This differs from the surrogate-gradient operator in Eq. (1), where the forward executes a hard mask but the backward follows a different smooth path.

# 4.3 Information Throttling with VP-Noise

Soft Top-K produces continuous retention weights, but the downstream VLM must experience them as token-level information restriction. DiffPrune therefore uses a token-wise Throttler $T ( \mathbf { x } _ { i } ; \alpha _ { i } ) \mapsto \tilde { \mathbf { x } } _ { i }$ for each visual token $\mathbf { x } _ { i } \in \mathbb { R } ^ { d _ { v } }$ and weight $\alpha _ { i } \in [ 0 , 1 ]$ . A simple scale gate, $\tilde { \mathbf { x } } _ { i } = \alpha _ { i } \mathbf { x } _ { i } ,$ is continuous but weak: low-scoring tokens keep their feature direction, and normalization layers can reduce the effect of magnitude shrinkage.

We therefore instantiate T with a variance-preserving noising form inspired by the forward process of diffusion models [48],

$$
\tilde {\mathbf {x}} _ {i} = \sqrt {\alpha_ {i}} \mathbf {x} _ {i} + \sqrt {1 - \alpha_ {i}} \boldsymbol {\epsilon} _ {i}, \quad \boldsymbol {\epsilon} _ {i} \sim \mathcal {N} (\mathbf {0}, \mathbf {I}), \tag {5}
$$

which we refer to as VP-Noise. For high-scoring tokens, $\alpha _ { i } \approx 1$ preserves the original representation; for low-scoring tokens, $\alpha _ { i } \approx 0$ replaces most visual content with isotropic noise. For normalized visual tokens, the square-root coefficients keep feature scale approximately stable while changing how much token content is available. For fixed noise, this operation is differentiable in $\alpha _ { i } ,$ so the Scorer is trained through the same noised representation consumed in the forward pass. Figure 4 validates VP-Noise from complementary visual and quantitative views. Panel (a) compares hard top-K pruning and VP-Noise gating at 25%, 50%, and 75% keep ratios, showing that VP-Noise keeps the retained spatial layout while corrupting low-retention patches instead of masking them out. Panel (b) reports ImageNet-1K Top-1/Top-5 accuracy under random scores; the small gap to hard top-K indicates that the continuous throttling path imposes comparable restriction strength.

![](images/1e8c6136660e9110933a88b60f046ddd87c9bd2b8863d5cdb20be523248cb135.jpg)

<details>
<summary>text_image</summary>

(a)
25%
K = 12
50%
K = 24
75%
K = 37
Hard Top-K Pruning
VP-Noise Gating (Ours)
</details>

![](images/acde16d5b0f44e435be162a8b5907bee297e95acd5d58efc6d74825aa152e58d.jpg)

<details>
<summary>line</summary>

| Token Budget K | Hard Top-K Pruning (Top-1) | Hard Top-K Pruning (Top-5) | VP-Noise Gating, Ours (Top-1) | VP-Noise Gating, Ours (Top-5) |
| -------------- | -------------------------- | -------------------------- | ----------------------------- | ----------------------------- |
| 12 (6%)        | 40                         | 65                         | 38                            | 62                            |
| 25 (13%)       | 60                         | 83                         | 60                            | 82                            |
| 49 (25%)       | 74                         | 90                         | 74                            | 91                            |
| 98 (50%)       | 81                         | 94                         | 80                            | 95                            |
| 147 (75%)      | 83                         | 95                         | 83                            | 96                            |
| 196 (100%)     | 84                         | 96                         | 84                            | 97                            |
</details>

Figure 4: Our VP-Noise suppresses token information in a manner functionally equivalent to hard top-K pruning, yet remains fully differentiable. (a) Visual comparison of VP-Noise and hard masking. (b) Their accuracy remains nearly identical under different pruning ratios, demonstrating the equivalence.

# 4.4 Diagonal-Attention Denoiser

VP-Noise makes token removal differentiable, but low-retention tokens may deviate from the feature distribution expected by the frozen downstream stack. We therefore insert a train-only Denoiser $\mathcal { D } _ { \phi }$ after the Throttler to realign the throttled sequence while the scorer is being learned.

The Denoiser should not undo the information restriction imposed by VP-Noise. DiffPrune implements it as a single Transformer block with an identity attention mask,

$$
\mathcal {D} _ {\phi} (\tilde {\mathbf {X}} ^ {v}) = \mathrm{Block} \Bigl (\tilde {\mathbf {X}} ^ {v}; \mathrm{mask} = \mathbf {I} _ {N} \Bigr),
$$

so each token is transformed independently through projection, normalization, and feed-forward layers, without copying content from other positions. The Denoiser is removed at inference.

Table 1: Results on LLaVA-1.5-7B. We compare DiffPrune with recent visual-token pruning methods under three retained-token budgets. Loc. indicates where pruning is applied, and retention after normalizing each benchmark score by the unpruned upper bound. Dashed lines divide training-free (above) and training-based (below) methods within each block. Best and second-best values are highlighted. 

<table><tr><td>Method</td><td>Loc.</td><td>GQA</td><td>MMB</td><td> $MMBCN$ </td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{v2}$ </td><td> $VQA^{Text}$ </td><td>SEED</td><td>VizWiz</td><td>Avg.</td></tr><tr><td colspan="13">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>-</td><td>61.9</td><td>64.7</td><td>58.1</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.5</td><td>58.2</td><td>60.5</td><td>54.3</td><td>100%</td></tr><tr><td colspan="13">Retain 192 Tokens (66.7% pruning)</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>L2-L32</td><td>52.7</td><td>61.2</td><td>53.5</td><td>1612</td><td>64.8</td><td>67.3</td><td>67.1</td><td>52.5</td><td>57.1</td><td>50.8</td><td>89.4%</td></tr><tr><td>HoloV (NeurIPS&#x27;25)</td><td>Pre</td><td>59.0</td><td>65.4</td><td>58.0</td><td>1820</td><td>85.6</td><td>69.8</td><td>76.7</td><td>57.4</td><td>-</td><td>50.9</td><td>98.2%</td></tr><tr><td>PRUNESID (ICLR&#x27;26)</td><td>Pre</td><td>60.1</td><td>63.7</td><td>-</td><td>1791</td><td>86.9</td><td>68.5</td><td>76.8</td><td>56.7</td><td>59.0</td><td>55.4</td><td>98.5%</td></tr><tr><td>PyramidDrop (CVPR&#x27;25)</td><td>L8-L24</td><td>57.3</td><td>63.6</td><td>56.8</td><td>1797</td><td>82.3</td><td>69.2</td><td>75.1</td><td>56.5</td><td>54.7</td><td>-</td><td>96.0%</td></tr><tr><td>VoCo-LLaMA (CVPR&#x27;25)</td><td>Pre</td><td>61.4</td><td>56.3</td><td>-</td><td>1596</td><td>84.5</td><td>66.6</td><td>-</td><td>50.6</td><td>51.1</td><td>-</td><td>91.1%</td></tr><tr><td>VisionZip (CVPR&#x27;25)</td><td>Pre</td><td>60.1</td><td>63.4</td><td>-</td><td>1834</td><td>84.9</td><td>68.2</td><td>77.4</td><td>57.8</td><td>57.1</td><td>-</td><td>97.8%</td></tr><tr><td>DiffPrune (Ours)</td><td>Pre</td><td>57.8</td><td>63.4</td><td>57.5</td><td>1791</td><td>86.5</td><td>70.2</td><td>76.6</td><td>55.3</td><td>59.2</td><td>55.9</td><td>98.2%</td></tr><tr><td colspan="13">Retain 128 Tokens (77.8% pruning)</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>L2-L32</td><td>49.6</td><td>56.1</td><td>55.9</td><td>1490</td><td>59.6</td><td>60.2</td><td>61.8</td><td>50.6</td><td>55.9</td><td>51.3</td><td>85.2%</td></tr><tr><td>HoloV (NeurIPS&#x27;25)</td><td>Pre</td><td>57.7</td><td>63.9</td><td>56.5</td><td>1802</td><td>82.8</td><td>69.8</td><td>75.5</td><td>56.8</td><td>-</td><td>51.5</td><td>96.8%</td></tr><tr><td>PRUNESID (ICLR&#x27;26)</td><td>Pre</td><td>58.8</td><td>62.1</td><td>-</td><td>1749</td><td>86.5</td><td>68.3</td><td>75.3</td><td>54.7</td><td>57.8</td><td>55.8</td><td>96.9%</td></tr><tr><td>PDrop (CVPR&#x27;25)</td><td>L8-L24</td><td>57.1</td><td>61.6</td><td>56.6</td><td>1761</td><td>82.3</td><td>68.4</td><td>72.9</td><td>56.6</td><td>53.3</td><td>-</td><td>94.7%</td></tr><tr><td>VoCo-LLaMA (CVPR&#x27;25)</td><td>Pre</td><td>61.5</td><td>56.4</td><td>-</td><td>1640</td><td>84.5</td><td>66.6</td><td>-</td><td>51.7</td><td>50.5</td><td>-</td><td>91.5%</td></tr><tr><td>VisionZip (CVPR&#x27;25)</td><td>Pre</td><td>58.9</td><td>62.6</td><td>-</td><td>1823</td><td>83.7</td><td>68.3</td><td>76.6</td><td>56.8</td><td>55.8</td><td>-</td><td>96.6%</td></tr><tr><td>DiffPrune (Ours)</td><td>Pre</td><td>57.5</td><td>62.9</td><td>57.4</td><td>1765</td><td>85.7</td><td>70.2</td><td>76.1</td><td>54.9</td><td>58.4</td><td>56.1</td><td>97.6%</td></tr><tr><td colspan="13">Retain 64 Tokens (88.9% pruning)</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>L2-L32</td><td>46.1</td><td>48.0</td><td>52.7</td><td>1256</td><td>48.0</td><td>51.1</td><td>55.0</td><td>47.8</td><td>51.9</td><td>50.8</td><td>76.8%</td></tr><tr><td>HoloV (NeurIPS&#x27;25)</td><td>Pre</td><td>55.3</td><td>63.3</td><td>55.1</td><td>1715</td><td>80.3</td><td>69.5</td><td>72.8</td><td>55.4</td><td>-</td><td>52.8</td><td>94.8%</td></tr><tr><td>PRUNESID (ICLR&#x27;26)</td><td>Pre</td><td>57.1</td><td>58.8</td><td>-</td><td>1733</td><td>83.8</td><td>67.8</td><td>73.7</td><td>54.2</td><td>56.1</td><td>56.9</td><td>95.1%</td></tr><tr><td>PDrop (CVPR&#x27;25)</td><td>L8-L24</td><td>47.5</td><td>58.8</td><td>50.5</td><td>1561</td><td>55.9</td><td>69.0</td><td>69.2</td><td>50.6</td><td>40.0</td><td>-</td><td>82.7%</td></tr><tr><td>VoCo-LLaMA (CVPR&#x27;25)</td><td>Pre</td><td>60.2</td><td>57.7</td><td>-</td><td>1623</td><td>83.4</td><td>67.7</td><td>-</td><td>51.1</td><td>50.0</td><td>-</td><td>91.2%</td></tr><tr><td>VisionZip (CVPR&#x27;25)</td><td>Pre</td><td>57.0</td><td>61.5</td><td>-</td><td>1756</td><td>80.9</td><td>68.8</td><td>74.2</td><td>56.0</td><td>53.4</td><td>-</td><td>94.2%</td></tr><tr><td>DiffPrune (Ours)</td><td>Pre</td><td>56.8</td><td>62.6</td><td>56.6</td><td>1723</td><td>83.4</td><td>70.1</td><td>73.9</td><td>54.3</td><td>57.6</td><td>57.2</td><td>96.5%</td></tr></table>

# 5 Experiments

# 5.1 Experimental Setup

DiffPrune is trained on 10% of the ImageNet-1K training split [49] with captions from ImageNet-1K-VL-Enriched [50]. Only the Scorer and Denoiser are optimized, adding approximately 84M trainable parameters, while the base VLM remains frozen. Complete implementation, training, baseline, and profiling details are provided in Section A; benchmark definitions and reporting conventions are provided in Section B.

# 5.2 Comparative Results

Results on LLaVA-1.5-7B. LLaVA-1.5-7B processes 336×336 inputs that produce N =576 visual tokens, and Table 1 reports three compression regimes that retain $\bar { 1 9 2 } / 1 2 8 / 6 4$ tokens (66.7%/77.8%/88.9% pruning). DiffPrune attains the strongest average retention at $K { = } 1 2 8$ and K=64; against the closest prior method, PRUNESID, the gap moves from a 0.3% deficit at $K { = } 1 9 2$ to a 1.4% lead at K=64, a direction consistent with the prediction of Section 3.1 that tighter budgets amplify the gap between fully differentiable and surrogate-gradient routes. The $K { = } 1 9 2$ gap to PRUNESID should be read together with the efficiency analysis of Section 5.3, where the PRUNESID selection module is shown to cost roughly 60× as much as ours.

Results on LLaVA-NEXT-7B. LLaVA-NEXT-7B uses 672×672 inputs and produces N=2880 visual tokens, five times the token count of LLaVA-1.5-7B. This setting gives a stronger test of visual-token pruning because the languagemodel prefill cost grows with the visual prefix length. Table 2 reports results at $K { = } 3 2 0$ , where each method keeps only 11.1% of the visual tokens. DiffPrune obtains 96.1% average retention, the best result among the reported methods and 0.4% higher than HoloV. We use the same scorer and pruning recipe as in $\mathrm { L L a V A – 1 } . 5 – 7 \mathrm { B }$ , with no resolution-specific scheduling or architectural change.

Table 2: Results on LLaVA-NEXT-7B at K=320 (88.9% pruning). This high-resolution setting starts from 2880 visual tokens; Avg. reports normalized average retention with Best and second-best values highlighted. 

<table><tr><td>Method</td><td>Loc.</td><td>GQA</td><td>MMB</td><td> $MMB_{CN}$ </td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{v2}$ </td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td colspan="11">Upper Bound, 2880 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>-</td><td>64.2</td><td>67.4</td><td>60.6</td><td>1851</td><td>86.5</td><td>70.1</td><td>80.8</td><td>64.9</td><td>100%</td></tr><tr><td colspan="11">Retain 320 Tokens (88.9% pruning)</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>L2-L32</td><td>55.9</td><td>61.6</td><td>51.9</td><td>1661</td><td>71.7</td><td>62.8</td><td>71.9</td><td>55.7</td><td>87.6%</td></tr><tr><td>PDrop (CVPR&#x27;25)</td><td>L8-L24</td><td>56.4</td><td>63.4</td><td>56.2</td><td>1663</td><td>77.6</td><td>67.5</td><td>73.5</td><td>54.4</td><td>90.7%</td></tr><tr><td>DART (EMNLP&#x27;25)</td><td>L2</td><td>61.7</td><td>65.3</td><td>58.2</td><td>1710</td><td>84.1</td><td>68.4</td><td>79.1</td><td>58.7</td><td>95.6%</td></tr><tr><td>HiRED (AAAI&#x27;25)</td><td>Pre</td><td>59.3</td><td>64.2</td><td>55.9</td><td>1690</td><td>83.3</td><td>66.7</td><td>75.7</td><td>58.8</td><td>93.4%</td></tr><tr><td>HoloV (NeurIPS&#x27;25)</td><td>Pre</td><td>61.7</td><td>65.3</td><td>57.5</td><td>1738</td><td>83.9</td><td>68.9</td><td>79.5</td><td>58.7</td><td>95.7%</td></tr><tr><td>DiffPrune (Ours)</td><td>Pre</td><td>62.3</td><td>64.7</td><td>57.4</td><td>1723</td><td>85.9</td><td>72.7</td><td>78.6</td><td>56.7</td><td>96.1%</td></tr></table>

Table 3: Results on Qwen2.5-VL-7B across three pruning rates. Native-resolution inputs yield image-dependent token counts, so results are grouped by pruning rate; Avg. is normalized by the unpruned model. 

<table><tr><td>Method</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td colspan="7">Upper Bound (100%)</td></tr><tr><td>Vanilla</td><td>82.8</td><td>2304</td><td>86.1</td><td>84.7</td><td>84.8</td><td>100%</td></tr><tr><td colspan="7">Pruning Rate = 66.7%</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>75.7</td><td>2072</td><td>82.2</td><td>78.5</td><td>77.9</td><td>92.3%</td></tr><tr><td>HoloV (NeurIPS&#x27;25)</td><td>78.3</td><td>2093</td><td>85.0</td><td>79.8</td><td>78.9</td><td>94.3%</td></tr><tr><td>DiffPrune (Ours)</td><td>81.7</td><td>2279</td><td>84.9</td><td>86.4</td><td>79.0</td><td>98.3%</td></tr><tr><td colspan="7">Pruning Rate = 77.8%</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>74.9</td><td>2036</td><td>80.7</td><td>78.0</td><td>69.0</td><td>89.2%</td></tr><tr><td>HoloV (NeurIPS&#x27;25)</td><td>76.5</td><td>2043</td><td>82.3</td><td>79.8</td><td>70.3</td><td>90.8%</td></tr><tr><td>DiffPrune (Ours)</td><td>81.0</td><td>2218</td><td>82.8</td><td>83.7</td><td>76.7</td><td>95.9%</td></tr><tr><td colspan="7">Pruning Rate = 88.9%</td></tr><tr><td>FastV (ECCV&#x27;24)</td><td>69.2</td><td>1940</td><td>78.6</td><td>77.4</td><td>60.3</td><td>84.3%</td></tr><tr><td>HoloV (NeurIPS&#x27;25)</td><td>72.4</td><td>2006</td><td>80.7</td><td>79.5</td><td>61.8</td><td>87.0%</td></tr><tr><td>DiffPrune (Ours)</td><td>76.7</td><td>2113</td><td>79.8</td><td>82.9</td><td>76.7</td><td>93.1%</td></tr></table>

Results on Qwen2.5-VL-7B. Qwen2.5-VL-7B differs from the LLaVA family in vision encoder, projector and language backbone, and processes images at native resolution, so the visual-token count varies per image and results are reported by pruning rate rather than a fixed budget K. Under the three pruning rates listed in Table 3, DiffPrune leads every evaluated baseline across the five benchmarks, with the margin over HoloV widening from 4.0% at 66.7% to 6.1% at 88.9%. No architectural or hyperparameter adaptation is applied across backbones; the same Scorer, Soft Top-K, Throttler, Denoiser and schedule used on LLaVA-1.5-7B are reused without modification, supporting the claim that the framework is backbone-agnostic by construction. We further examine this behavior across Qwen2.5-VL model scales in Section C.

# 5.3 Efficiency Analysis

We profile all methods on LLaVA-1.5-7B with 336×336 inputs and N=576 visual tokens, using a single NVIDIA A6000 GPU, batch size one, and FP16 precision. Latencies are averaged over 30 forward passes after warm-up. Table 4 reports TTFT at K=64, decomposed into vision encoding, pruning-module decision, and LLM prefill.

Vision encoding takes about 30 ms for all methods, since pruning is applied after the encoder. The main differences come from the pruner and the LLM prefill. PyramidDrop prunes inside the language model, so early layers still process the full visual prefix. Its LLM prefill remains 70.08 ms with 4.89 T FLOPs. In contrast, Pre-LLM methods reduce the visual prefix before it enters the language model, bringing LLM prefill to about 40 ms and FLOPs to 2.13–2.15 T. Among these methods, the pruner itself becomes the main source of latency. PRUNESID spends 43.39 ms on token selection, over 60× the cost of DiffPrune, while HoloV spends 2.77 ms. DiffPrune needs only 0.69 ms for selection and reaches the lowest TTFT in the comparison (73.73 ms). This follows directly from its inference path in Eq. (3): the Throttler and Denoiser are removed, leaving only one Scorer pass and index gathering.

Table 4: Efficiency breakdown on LLaVA-1.5-7B (K=64). Latencies are in ms; FLOPs are in T. 

<table><tr><td>Method</td><td>Loc.</td><td>K</td><td>FLOPs (T)</td><td>Enc.</td><td>Pruner</td><td>Prefill</td><td>TTFT</td></tr><tr><td>Full (LLaVA)</td><td>-</td><td>576</td><td>8.89</td><td>29.71</td><td>0.00</td><td>118.66</td><td>149.51</td></tr><tr><td>PruneSID</td><td>Pre</td><td>64</td><td>2.13</td><td>31.80</td><td>43.39</td><td>39.74</td><td>115.84</td></tr><tr><td>PyramidDrop</td><td>L8–L24</td><td>64</td><td>4.89</td><td>29.54</td><td>5.02</td><td>70.08</td><td>105.90</td></tr><tr><td>HoloV</td><td>Pre</td><td>64</td><td>2.15</td><td>31.96</td><td>2.77</td><td>41.48</td><td>77.39</td></tr><tr><td>DiffPrune (Ours)</td><td>Pre</td><td>64</td><td>2.13</td><td>30.41</td><td>0.69</td><td>41.61</td><td>73.73</td></tr></table>

Table 5: Component ablation of DiffPrune on LLaVA-1.5-7B. Variants replace one training-time component; scores are average retention over GQA, MMBench, POPE, and MME. 

<table><tr><td>Variant</td><td>Replaced design</td><td>K=64</td><td>K=128</td><td>K=192</td></tr><tr><td>Full DiffPrune</td><td>-</td><td>95.5%</td><td>97.3%</td><td>98.3%</td></tr><tr><td>Gumbel-STE Throttler</td><td>gradient-continuous</td><td>87.4%</td><td>91.3%</td><td>94.0%</td></tr><tr><td>Scale-gate Throttler</td><td>VP-Noise form</td><td>93.2%</td><td>94.8%</td><td>96.5%</td></tr><tr><td>Global-attention Denoiser</td><td>diagonal</td><td>92.7%</td><td>95.0%</td><td>95.9%</td></tr></table>

# 5.4 Ablation Studies

We isolate three training-time design choices in DiffPrune: the gradient-continuous Throttler, the VP-Noise form, and the diagonal Denoiser. All variants keep the base VLM, Scorer architecture, training data, and schedule fixed. Table 5 reports average retention over GQA, MMBench, POPE, and MME on LLaVA-1.5-7B.

Replacing the Throttler with a Gumbel-Softmax + STE variant gives the largest drop. Retention decreases by 8.1, 6.0, and 4.3 points at $K { = } 6 4 / 1 2 8 / 1 9 2$ , respectively. Since the rest of the pipeline is unchanged, this result supports the claim from Section 3 that the gradient-continuous training path is important in the full VLM setting, not only in the DeiT probe.

The other two rows separate DiffPrune’s internal design choices. Replacing VP-Noise with the fully differentiable scale gate $\tilde { \mathbf { x } } _ { i } = \alpha _ { i } \mathbf { x } _ { i }$ reduces retention by 2.3, 2.5, and 1.8 points, showing that noise injection gives a stronger information restriction than simple rescaling. Replacing diagonal attention in the Denoiser with global self-attention reduces retention by 2.8, 2.3, and 2.4 points. This supports per-token denoising, since global attention can let corrupted low-score tokens recover information from high-score tokens.

# 5.5 Qualitative Analysis

Figure 5 visualizes the token scores learned by DiffPrune on an example from the VQAv2 benchmark used in our evaluation. For readability, we show the two extremes of the score ranking: the top-64 tokens cover visually salient regions such as the face, hands, and clothing, while the bottom-64 tokens mainly fall on visually uniform background areas. The accompanying 128×128 cosine-similarity matrix [top-64; |; bottom-64] provides a feature-space view of the same ranking: in this example, the retained-token block has lower internal similarity, whereas the lowest-scored block is more redundant. This qualitative evidence is consistent with the quantitative results, while remaining illustrative rather than conclusive: the learned Scorer appears to allocate its limited budget toward visually informative and less redundant tokens.

# 6 Conclusion

We identify the discrete selection operator as a central obstacle in learned visual-token pruning: continuous importance scores must eventually become hard keep-or-drop decisions, and common surrogate-gradient solutions train through a backward path that is not the function executed in the forward pass. DiffPrune addresses this mismatch by replacing training-time selection with continuous token-information control. Its Soft Top-K head and Information Throttler keep the scorer-to-loss path analytically differentiable during training, while inference collapses to a hard top-K scorer-only path. The resulting framework improves gradient-direction coherence by up to 28.4×, reduces local loss variation by roughly 24× in our diagnostic study, and adds only 0.69 ms of selection overhead on LLaVA-1.5-7B. These findings suggest that full differentiability is a useful design rule for learned pruning, not merely an implementation detail. Extending the same principle to other sparsity operators remains future work.

![](images/da95e391a8bc92d5f9147190f8121d4cbf0ef5f1d5889421be9ff91bc6098514.jpg)

<details>
<summary>text_image</summary>

Token Groups (n=64 each)
Top-64 (highest)
Bottom-64 (lowest)
</details>

![](images/f5dfe16f4c7d6998e75e417e870a534449a3a71c409777f98acea25abe6d1de8.jpg)

<details>
<summary>heatmap</summary>

| Top-64 (0-64) | Bottom-64 (512-576) | Cosine Similarity |
| ------------- | ------------------- | ------------------ |
| 0             | 0                   | 1.0                |
| 16            | 16                  | 0.8                |
| 32            | 32                  | 0.6                |
| 48            | 48                  | 0.4                |
| 64            | 64                  | 0.2                |
| 528           | 528                 | 0.0                |
| 544           | 544                 | 0.2                |
| 560           | 560                 | 0.4                |
| 576           | 576                 | 0.6                |
</details>

Figure 5: Learned token ranking on a VQAv2 example at K=64. Left: Tokens are ranked by learned importance scores, where red and blue indicate the top-K retained tokens and the lowest-scored pruned tokens, respectively. Right: Similarity matrix of retained and pruned tokens. Pruned tokens are more redundant.

# A Implementation Details and Training Hyperparameters

Implementation. All experiments are implemented in PyTorch and run on a single NVIDIA RTX A6000 GPU. The implementation is built on the LLaVA-v1.5 codebase [51]. For every evaluated backbone, the vision encoder, multimodal projector, and language model remain frozen; only the DiffPrune Scorer and Denoiser are optimized.

Architecture. The Scorer contains two Transformer encoder blocks with hidden dimension 1024 and 16 attention heads, followed by a linear projection that maps each visual token to a scalar importance logit. The Denoiser is a single Transformer block with the diagonal attention mask described in Section 4.4. Together, these two trainable modules add approximately 84M parameters, about 1.2% of LLaVA-1.5-7B. At inference time, both the Throttler and Denoiser are removed; the deployed graph contains only the Scorer and hard top-K gathering in Eq. (3).

Training. The Scorer and Denoiser are trained jointly in a single stage on a 10% subset of the ImageNet-1K training split [49], paired with captions from ImageNet-1K-VL-Enriched [50]. We use AdamW with learning rate $2 \times 1 0 ^ { - \tilde { 4 } }$ cosine decay, and batch size 32. The objective is caption generation under the next-token negative log-likelihood in Eq. (2); no labels from downstream evaluation benchmarks are used for training. The Soft Top-K temperature is annealed from $\tau _ { \mathrm { s t a r t } } { = } 2 . 0 \ \mathrm { t o } \ \tau _ { \mathrm { e n d } } { = } 0 .$ .1 over 2,000 steps, gradually sharpening the continuous retention weights toward the hard top-K ranking used at inference. Training runs for one epoch over the subset with early stopping based on validation loss and takes approximately ten GPU-hours.

Backbones and token budgets. LLaVA-1.5-7B uses 336×336 images and produces N=576 visual tokens. We evaluate K∈{192, 128, 64}, corresponding to 66.7%, 77.8%, and 88.9% pruning. LLaVA-NEXT-7B uses 672×672 inputs and produces N =2880 visual tokens; we report K=320. Qwen2.5-VL-7B processes images at native resolution, so the number of visual tokens varies by example; for this backbone, results are reported by pruning rate rather than by a fixed K. Unless otherwise stated, the same Scorer, Soft Top-K operator, Throttler, Denoiser, and training schedule are reused across backbones.

Baselines and reporting convention. Baselines include FastV [7], SparseVLM [8], DART [29], PyramidDrop [24], DivPrune [11], VisionZip [10], HoloV [30], PRUNESID [32], ATP-LLaVA [38], Dynamic-LLaVA [39], p-MoD [52], GlimpsePrune [53], and GumbelDiffPrune. GumbelDiffPrune is an in-house controlled variant that replaces the DiffPrune Throttler with Gumbel-Softmax and a Straight-Through Estimator while keeping the rest of the pipeline fixed. When applicable, Loc. records whether pruning happens before the language model (Pre) or inside it (La–Lb). Entries are marked “–” when a compatible result is unavailable.

Efficiency profiling. Latency is profiled on LLaVA-1.5-7B with 336×336 inputs, N=576 visual tokens, batch size one, and FP16 precision on the same NVIDIA A6000 GPU. We average 30 forward passes after warm-up. Time-to-firsttoken is decomposed into vision encoding, pruning-module decision time, and language-model prefill, so the reported pruning overhead isolates the cost added by each selector.

# B Benchmark Descriptions

We evaluate DiffPrune on ten standard VLM benchmarks spanning compositional reasoning, perception, hallucination, text-centric understanding, and real-world VQA.

GQA [54]. GQA is a compositional visual question answering benchmark built from Visual Genome scene graphs. It emphasizes multi-step reasoning over objects, attributes, spatial relations, and logical operations, making it useful for testing whether token pruning preserves structured visual evidence.

MMBench and MMBench-CN [55]. MMBench evaluates multimodal perception and reasoning through multiplechoice questions organized into fine-grained ability dimensions. We report both the English benchmark and its Chinese counterpart, MMBench-CN, which follows the same evaluation taxonomy and enables testing under a language distribution shift.

MME [56]. MME evaluates perception and cognition through manually designed yes/no questions. Its perception subset covers skills such as existence, count, position, and color recognition, while its cognition subset covers broader reasoning categories. Because each image is paired with controlled questions, MME is sensitive to both visual information loss and hallucinated content.

POPE [57]. POPE measures object hallucination by querying whether specific objects are present in an image. Questions are sampled under random, popular, and adversarial strategies, probing whether a model relies on visual evidence rather than dataset-level object co-occurrence priors.

ScienceQA-IMG [58]. ScienceQA-IMG contains image-grounded science questions that require both visual understanding and domain knowledge. We use the image-containing subset, following standard VLM evaluation practice.

VQAv2 [59]. VQAv2 is an open-ended visual question answering benchmark designed to reduce language priors through complementary image pairs. It tests whether the model’s answer changes with the visual evidence rather than being driven only by question statistics.

TextVQA [60]. TextVQA focuses on reading and reasoning over text appearing inside images, such as signs, labels, documents, and scene text. This benchmark stresses OCR-sensitive visual tokens and is therefore a stringent test for pruning methods that remove large fractions of the visual prefix.

SEED-Bench [61]. SEED-Bench evaluates image-centric multimodal comprehension across dimensions such as scene understanding, spatial relation, instance identity, and instance attribute recognition. We use it to assess whether pruning preserves broad semantic coverage beyond conventional VQA accuracy.

VizWiz [62]. VizWiz contains questions about images taken by blind users in everyday settings. The images often contain unusual framing, low quality, or unanswerable queries, making the benchmark a realistic stress test for robustness under noisy visual inputs.

For each benchmark, we report its standard score. To summarize performance across heterogeneous metrics, we additionally report average retention: each pruned score is normalized by the corresponding unpruned upper bound, and the normalized values are averaged across benchmarks.

![](images/a559f76d00634266cd5b1187d87ef37f8ae007a95d26e2395e5c7df1361e4f2d.jpg)

<details>
<summary>line</summary>

| Pruning Rate (%) | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-32B |
| ---------------- | ------------- | ------------- | -------------- |
| 30               | 2140          | 2300          | 2450           |
| 50               | 2130          | 2290          | 2410           |
| 70               | 2100          | 2280          | 2380           |
| 80               | 2070          | 2220          | 2380           |
| 90               | 1910          | 2110          | 2300           |
</details>

![](images/472eba3028bef91a8156fd4052381f2d7edb6125d6ee6aa3f0d93d6d3195d334.jpg)

<details>
<summary>line</summary>

| Pruning Rate (%) | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-32B |
| ---------------- | ------------- | ------------- | -------------- |
| 30               | 81.0          | 88.0          | 91.0           |
| 50               | 81.0          | 87.0          | 90.0           |
| 70               | 80.5          | 86.0          | 88.0           |
| 80               | 80.0          | 83.5          | 87.0           |
| 90               | 79.5          | 82.5          | 84.0           |
</details>

![](images/dfd519296b660fd5e99f8e70d37fe2507ad6abb92a301972390ca820d5d93c3d.jpg)

<details>
<summary>line</summary>

| Pruning Rate (%) | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-32B |
| ---------------- | ------------- | ------------- | -------------- |
| 30               | 77.0          | 82.5          | 85.0           |
| 50               | 76.5          | 82.5          | 84.0           |
| 70               | 76.0          | 81.5          | 83.0           |
| 80               | 75.5          | 81.0          | 82.0           |
| 90               | 72.0          | 76.5          | 79.5           |
</details>

Figure A1: Scalability across Qwen2.5-VL model scales. Performance on MME, ScienceQA, and MMBench as a function of the visual-token pruning rate for Qwen2.5-VL-3B, 7B, and 32B.

# C Scalability Across Model Scales

To further test whether DiffPrune depends on a particular model capacity, we evaluate the 3B, 7B, and 32B variants of Qwen2.5-VL [63] on MME, ScienceQA, and MMBench. We use five visual-token pruning rates: 25%, 50%, 66.7%, 77.8%, and 88.9%. The same pruning formulation and training recipe are used across model sizes.

Figure A1 shows that, across all three scales, performance decreases smoothly as the pruning rate increases, with no abrupt accuracy collapse even at 88.9% pruning. On MME, the 32B model remains above 2,400 points near the 77.8% pruning setting, while the 7B and 3B variants show steady rather than discontinuous degradation from their respective unpruned baselines. MMBench and ScienceQA show the same pattern: the 32B and 7B results remain close through high pruning rates, and all models preserve usable performance under aggressive compression. These results suggest that DiffPrune’s budgeted, score-driven formulation is not tied to a single Qwen2.5-VL model size.

# D Limitation and Future Work

DiffPrune relies on the continuity of visual-token embeddings. VP-Noise throttles a token by interpolating its feature vector with isotropic Gaussian noise under a variance-preserving schedule before the frozen language model consumes it. This assumption is natural for VLM visual tokens, but it does not transfer directly to pure LLM settings, where the objects being pruned may be discrete text tokens, token-indexed KV entries, or weight structures whose semantics are not preserved under the same feature-space perturbation. DiffPrune further introduces training-time cost through the Scorer and Denoiser, even though the Throttler and Denoiser are removed at inference. Finally, while our pilot studies show substantially higher gradient-direction coherence and smoother scorer loss landscapes than Gumbel-Softmax pruning, a complete theoretical account of how feature dimensionality, the geometry of the K-th order statistic, and surrogate-gradient bias interact remains open.

Despite these limitations, these properties also suggest promising directions for future work in high-resolution visual domains. In particular, medical imaging and related settings are characterized by substantial redundancy in visual tokens [64, 65, 66]. This phenomenon has been widely observed in medical image analysis [67, 68, 69, 70], multimodal medical foundation models [71, 72, 73, 74], and long-context visual reasoning tasks [75, 76], suggesting that DiffPrune may be especially effective in these scenarios.

# References

[1] Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, and Srimat Chakradhar. Vita: An efficient videoto-text algorithm using vlm for rag-based video analysis system. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2266–2274, 2024.   
[2] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, et al. Fastvlm: Efficient vision encoding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19769–19780, 2025.   
[3] Sixun Dong, Juhua Hu, Mian Zhang, Ming Yin, Yanjie Fu, and Qi Qian. Mmtok: Multimodal coverage maximization for efficient inference of vlms. ICLR, 2026.

[4] Yixu Feng, Zinan Zhao, Yanxiang Ma, Chenghao Xia, Chengbin Du, Yunke Wang, and Chang Xu. See what matters: Differentiable grid sample pruning for generalizable vision-language-action model. ICML, 2026.   
[5] Ahmadreza Jeddi, Negin Baghbanzadeh, Elham Dolatabadi, and Babak Taati. Similarity-aware token pruning: Your vlm but faster. arXiv preprint arXiv:2503.11549, 2025.   
[6] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pages 10347–10357. PMLR, 2021.   
[7] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. 2024.   
[8] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. ICML, 2025.   
[9] Daniel Bolya, Cheng-Yang Fu, Xiaodong Dai, Peize Zhang, and Judy Hoffman. Token merging: Your ViT but faster. In ICLR, 2023.   
[10] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In CVPR, pages 19792–19802, 2025.   
[11] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. 2025.   
[12] Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, and Shanghang Zhang. Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. Advances in Neural Information Processing Systems, 38:25438–25468, 2026.   
[13] Tiannan Wang, Wangchunshu Zhou, Yan Zeng, and Xinsong Zhang. Efficientvlm: Fast and accurate visionlanguage models via knowledge distillation and modal-adaptive pruning. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13899–13913, 2023.   
[14] Xijun Wang, Junyun Huang, Rayyan Abdalla, Chengyuan Zhang, Ruiqi Xian, and Dinesh Manocha. Bi-vlm: Pushing ultra-low precision post-training quantization boundaries in vision-language models. arXiv preprint arXiv:2509.18763, 2025.   
[15] Rinyoichi Takezoe, Yaqian Li, Zihao Bo, Anzhou Hou, Mo Guang, and Kaiwen Long. Learnpruner: Rethinking attention-based token pruning in vision language models. ICLR 2026, 2026.   
[16] Yumiao Zhao, Bo Jiang, Yuhe Ding, Xiao Wang, Jin Tang, and Bin Luo. Fine-grained vlm fine-tuning via latent hierarchical adapter learning. arXiv preprint arXiv:2508.11176, 2025.   
[17] Xiaoyu Liang, Chaofeng Guan, Jiaying Lu, Huiyao Chen, Huan Wang, and Haoji Hu. Dynamic token reduction during generation for vision language models. arXiv preprint arXiv:2501.14204, 2025.   
[18] Ningfeng Yang and Tor Aamodt. Improving the straight-through estimator with zeroth-order information. Advances in Neural Information Processing Systems, 38:167408–167440, 2026.   
[19] Kele Shao, TAO Keda, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, and Huan Wang. A survey of token compression for efficient multimodal large language models. Transactions on Machine Learning Research, 2025.   
[20] Linli Yao, Long Xing, Yang Shi, Sida Li, Yuanxin Liu, Yuhao Dong, Yi-Fan Zhang, Lei Li, Qingxiu Dong, Xiaoyi Dong, et al. Towards efficient multimodal large language models: A survey on token compression. TechRxiv preprint, 2026.   
[21] Haoran Zhang et al. [CLS] attention is all you need for training-free visual token pruning: Make VLM inference faster, 2024. arXiv preprint.   
[22] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In AAAI, volume 39, pages 5334–5342, 2025.   
[23] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1773–1781, 2025.   
[24] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. CVPR, 2025.

[25] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, and Bo Yuan. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In CVPR, pages 19803–19813, 2025.   
[26] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. 2024.   
[27] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. 2025.   
[28] Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token pruning for vision-language model acceleration. 2024.   
[29] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for “important tokens” in multimodal language models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9972–9991, 2025.   
[30] Xin Zou, Di Lu, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Xu Zheng, Linfeng Zhang, and Xuming Hu. Don’t just chase “highlighted tokens” in mllms: Revisiting visual holistic context retention. NeurIPS, 2025.   
[31] Kaiyuan Li, Xiaoyue Chen, Chen Gao, Yong Li, and Xinlei Chen. Balanced token pruning: Accelerating vision language models beyond local optimization. 2025.   
[32] Zhengyao Fang, Pengyuan Lyu, Chengquan Zhang, Guangming Lu, Jun Yu, and Wenjie Pei. Prune redundancy, preserve essence: Vision token compression in vlms via synergistic importance-diversity. In The Fourteenth International Conference on Learning Representations, 2025.   
[33] Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz, Pavlo Molchanov, and Xinchao Wang. Maskllm: Learnable semi-structured sparsity for large language models. Advances in Neural Information Processing Systems, 37:7736–7758, 2024.   
[34] Enshuai Zhou, Yifan Hao, Chao Wang, Rui Zhang, Di Huang, Jiaming Guo, Xing Hu, Zidong Du, Qi Guo, and Yunji Chen. Lkv: End-to-end learning of head-wise budgets and token selection for llm kv cache eviction. arXiv preprint arXiv:2605.06676, 2026.   
[35] Junyan Li, Li Lyna Zhang, Jiahang Xu, Yujing Wang, Shaoguang Yan, Yunqing Xia, Yuqing Yang, Ting Cao, Hao Sun, Weiwei Deng, et al. Constraint-aware and ranking-distilled token pruning for efficient transformer inference. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1280–1290, 2023.   
[36] Zhenkai Wu, Xiaowen Ma, Zhenliang Ni, Dengming Zhang, Han Shu, Xin Jiang, and Xinghao Chen. Vlm-pruner: Buffering for spatial sparsity in an efficient vlm centrifugal token pruning paradigm. CVPR 2026, 2025.   
[37] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937–13949, 2021.   
[38] Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. ATP-LLaVA: Adaptive token pruning for large vision language models. In CVPR, pages 24972–24982, 2025.   
[39] Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaosheng Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, and Shaohui Lin. Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification. In International Conference on Learning Representations, volume 2025, pages 69927–69955, 2025.   
[40] Titong Jiang, Xuefeng Jiang, Yuan Ma, Xin Wen, Bailin Li, Kun Zhan, Peng Jia, Yahui Liu, Sheng Sun, and Xianpeng Lang. The better you learn, the smarter you prune: Towards efficient vision-language-action models via differentiable token pruning. arXiv preprint arXiv:2509.12594, 2025.   
[41] Jiaji Zhang, Hailiang Zhao, Guoxuan Zhu, Ruichao Sun, Jiaju Wu, Xinkui Zhao, Hanlin Tang, Weiyi Lu, Kan Liu, Tao Lan, et al. Shiva-dit: Residual-based differentiable top-k selection for efficient diffusion transformers. arXiv preprint arXiv:2602.05605, 2026.   
[42] Zhenwei Shao, Mingyang Wang, Weijun Zhang, Zhou Yu, Wenwen Pan, Yan Yang, Tao Wei, Hongyuan Zhang, and Jun Yu. Growing a multi-head twig via distillation and reinforcement learning to accelerate large vision-language models, 2025.   
[43] Alexander Shekhovtsov, Viktor Yanush, and Boris Flach. Path sample-analytic gradient estimators for stochastic binary networks. Advances in neural information processing systems, 33:12884–12894, 2020.   
[44] Rushi Shah, Mingyuan Yan, Michael Curtis Mozer, and Dianbo Liu. Improving discrete optimisation via decoupled straight-through gumbel-softmax. arXiv preprint arXiv:2410.13331, 2024.

[45] Alexander Shekhovtsov. Bias-variance tradeoffs in single-sample binary gradient estimators. In DAGM German Conference on Pattern Recognition, pages 127–141. Springer, 2021.   
[46] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. Advances in neural information processing systems, 31, 2018.   
[47] Łukasz Struski, Michal B. Bednarczyk, Igor T. Podolak, and Jacek Tabor. LapSum - one method to differentiate them all: Ranking, sorting and top-k selection. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 56990–57007. PMLR, 13–19 Jul 2025.   
[48] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.   
[49] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255, 2009.   
[50] Visual Layer. Imagenet-1k-vl-enriched, 2023. HuggingFace dataset.   
[51] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
[52] Jun Zhang, Desen Meng, Zhengming Zhang, Zhenpeng Huang, Tao Wu, and Limin Wang. p-mod: Building mixture-of-depths mllms via progressive ratio decay. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3705–3715, 2025.   
[53] Quan-Sheng Zeng, Yunheng Li, Qilong Wang, Peng-Tao Jiang, Zuxuan Wu, Ming-Ming Cheng, and Qibin Hou. A glimpse to compress: Dynamic visual token pruning for large vision-language models. arXiv preprint arXiv:2508.01548, 2025.   
[54] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.   
[55] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.   
[56] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. NeurIPS, 2025.   
[57] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305, 2023.   
[58] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521, 2022.   
[59] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.   
[60] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.   
[61] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024.   
[62] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342, 2010.   
[63] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025.

[64] Xiaoyu Yang, Lijian Xu, Hongsheng Li, and Shaoting Zhang. One leaf reveals the season: Occlusion-based contrastive learning with semantic-aware views for efficient visual representation. In International Conference on Machine Learning, pages 71425–71440, 2025.   
[65] Shawn Young, Xingyu Zeng, and Lijian Xu. Fewer tokens, greater scaling: Self-adaptive visual bases for efficient and expansive representation learning. arXiv preprint arXiv:2511.19515, 2025.   
[66] Xiaoyu Yang, Lijian Xu, Xingyu Zeng, Xiaosong Wang, Hongsheng Li, and Shaoting Zhang. Scalar: Spatialconcept alignment for robust vision in harsh open world. Pattern Recognition, page 113203, 2026.   
[67] Xiaoyu Yang, Lijian Xu, Simon Yu, Qing Xia, Hongsheng Li, and Shaoting Zhang. Segmentation and vascular vectorization for coronary artery by geometry-based cascaded neural network. IEEE Transactions on Medical Imaging, 44(1):259–269, 2024.   
[68] Xiaoyu Yang, Lijian Xu, Simon Yu, Qing Xia, Hongsheng Li, and Shaoting Zhang. Geometry-based end-to-end segmentation of coronary artery in computed tomography angiography. In International Workshop on Trustworthy Machine Learning for Healthcare, pages 190–196. Springer, 2023.   
[69] Zhuo Chen, Shawn Young, and Lijian Xu. Tc-ssa: Token compression via semantic slot aggregation for gigapixel pathology reasoning. arXiv preprint arXiv:2603.01143, 2026.   
[70] Peihang Wu, Zehong Chen, and Lijian Xu. Multimodal model for computational pathology: Representation learning and image compression. arXiv preprint arXiv:2603.18660, 2026.   
[71] Lijian Xu, Ziyu Ni, Xinglong Liu, Xiaosong Wang, Hongsheng Li, and Shaoting Zhang. Learning a multi-task transformer via unified and customized instruction tuning for chest radiograph interpretation. arXiv preprint arXiv:2311.01092, 2023.   
[72] Lijian Xu, Hao Sun, Ziyu Ni, Hongsheng Li, and Shaoting Zhang. Medvilam: A multimodal large language model with advanced generalizability and explainability for medical data understanding and generation. arXiv preprint arXiv:2409.19684, 2024.   
[73] Lijian Xu, Ziyu Ni, Hao Sun, Hongsheng Li, and Shaoting Zhang. A foundation model for generalizable disease diagnosis in chest x-ray images. arXiv preprint arXiv:2410.08861, 2024.   
[74] Wangyu Feng, Shawn Young, and Lijian Xu. Efficient chest x-ray representation learning via semantic-partitioned contrastive learning. arXiv preprint arXiv:2603.07113, 2026.   
[75] Yonghan Gao, Zehong Chen, Lijian Xu, Jingzhi Chen, Jingwei Guan, and Xingyu Zeng. Zerosense: How vision matters in long context compression. arXiv preprint arXiv:2603.11846, 2026.   
[76] Landi He, Xiaoyu Yang, and Lijian Xu. The model knows which tokens matter:automatic token selection via noise gating. arXiv preprint arXiv:2603.07135, 2026.