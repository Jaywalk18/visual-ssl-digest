# YARD: Y-Architecture Register Decoding for Efficient Hallucination Mitigation in Large Vision-Language Models

Ting Chen1,\*

Guan Huang3

Geng Li2,\*

Mai Chen3

Guohao Chen2,\*

Langsheng Lei3

Yu Hu1,†

Jun Du3

1Guangdong University of Technology

2Nanyang Technological University

3Shenzhen TENCLASS Technology Co., Ltd.

\*Equal contribution. †Corresponding author.

# Abstract

Contrastive decoding (CD) seeks to mitigate hallucinations in Large Vision-Language Models (LVLMs) by contrasting the output distributions of a standard model and a visually degraded model. However, existing trainingfree CD methods suffer from sub-optimal degraded branches: completely dropping visual tokens is too extreme and induces language hallucinations, while corrupting input images offers coarse control over visual evidence and suffers from high inference latency due to requiring two full forward passes. To address these dilemmas, we propose YARD, a trainingfree Y-Architecture Register Decoding framework. Motivated by the observation that reliable text-to-vision grounding predominantly emerges in the middle decoder layers, YARD constructs the degraded branch internally by sharing shallow-layer computations and branching exactly at this critical stage. For the degraded branch, YARD replaces patch-level visual tokens with register tokens, which preserve global image semantics but lack fine-grained local evidence. This image-aware yet locally under-grounded design provides a faithful contrastive signal without extreme modality mismatch, while the Y-architecture strictly avoids a costly second forward pass. Extensive experiments on generative and discriminative hallucination benchmarks demonstrate that YARD consistently achieves state-of-the-art hallucination mitigation across multiple LVLMs, alongside a significant reduction in inference latency.

# 1 Introduction

Large vision-language models (LVLMs) (Liu et al., 2024b; Dai et al., 2023; Wang et al., 2024a; Zhu et al., 2024) connect visual encoders with large language models, enabling strong visual understand-

![](images/0d5f7c022d5770476d47f8cfab690631cdd046af5a7d21129a02c155ca58c1b3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Distorted Visual Inputs V"] --> B["Visual Encoder"]
    C["Original Visual Inputs V"] --> D["Visual Encoder"]
    B --> E["Large Language Model"]
    D --> F["Large Language Model"]
    E --> G["Degraded logits (y | x, v)"]
    F --> H["Clean logits (y | x, v)"]
    I["Textual Input X"] --> J["What is on the beach in the picture?"]
    K["Textual Input X"] --> L["What is on the beach in the picture?"]
    M["Textual Input X"] --> N["What is on the beach in the picture?"]
    O["Textual Input X"] --> P["What is on the beach in the picture?"]
    Q["Textual Input X"] --> R["What is on the beach in the picture?"]
    S["Textual Input X"] --> T["What is on the beach in the picture?"]
    U["Textual Input X"] --> V["What is on the beach in the picture?"]
```
</details>

(a)

![](images/d3673a35e48e1991a4d6853d1e2ae8d970acc02262cb83a101fa9c6478cf6b5e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Visual Inputs V"] --> B["Visual Encoder"]
    C["Textual Input X"] --> D["What is on the beach in the picture?"]
    B --> E["YARD"]
    D --> E
    E --> F["Clean Branch"]
    E --> G["Degraded Branch"]
    F --> H["Degraded logits (y | x, v)"]
    G --> I["Clean logits (y | x, v)"]
    H --> J["Visual Contrastive Decoding"]
    I --> J
```
</details>

(b)   
Figure 1: Comparison between (a) pixel-level degradation that corrupts the visual input and (b) YARD that constructs an image-aware but locally under-grounded degraded branch inside the LLM decoder.

ing and language generation capabilities across visual question answering, image captioning, and open-ended multimodal dialogue. However, despite their ability to produce fluent and contextually coherent responses, hallucination remains a central obstacle to their reliability(Li et al., 2023b). LVLMs may generate text that is semantically plausible but inconsistent with the visual content, such as non-existent objects(Rohrbach et al., 2018), incorrect attributes(Pham and Schott, 2024), or spatial relations unsupported by the image(Guan et al., 2024). Such failures suggest that generation is not always faithfully grounded in the visual evidence, making hallucination mitigation a key challenge for reliable LVLM deployment(Bai et al., 2024).

Among existing training-free approaches to hallucination mitigation, contrastive decoding (Chuang et al., 2024) can be viewed as a form of hallucination-component separation at the logit level. Specifically, the clean logits of an LVLM contain both predictions supported by genuine visual evidence and hallucination-related components induced by language priors or insufficient grounding. The role of the degraded branch is to construct a contrastive distribution that exposes these hallucination-prone components, so that they can be selectively suppressed through logit subtraction (Li et al., 2023a).

Existing methods typically instantiate the degraded branch in two ways. One line of work masks or removes visual information inside the LLM, reducing the degraded branch to an approximately text-only language-prior branch (Favero et al., 2024; Zhu et al., 2025; Wang et al., 2024b; Woo et al., 2025; Huo et al., 2024; Deng and Yang, 2025; Manevich and Tsarfaty, 2024; Li et al., 2025a). However, once visual conditioning is completely removed, the induced hallucinations often arise from generic language priors and may not align with the input image. Another line perturbs the original image in pixel space by adding noise, masking image regions, or corrupting visual details (Chen et al., 2024b; Yang et al., 2024; Zhao et al., 2025). However, pixel-level degradation cannot selectively disrupt visual signals: color, texture, object boundaries, spatial layout, and local semantics are tightly entangled in pixel space. As a result, such perturbations may fail to isolate the local evidence responsible for hallucination and instead contaminate the degraded logits with signals associated with genuine visual evidence.

These limitations motivate us to move degradation into a more structured feature-level space, where global image semantics can be preserved while fine-grained local visual evidence is selectively weakened. Table 1 further supports this choice, showing that feature-level degradation exposes a purer hallucination signal than text-only or pixel-level degradation. However, existing contrastive decoding methods mainly construct degraded branches from either the language side or the visual-input side, leaving feature-level degradation inside the LVLM decoder largely underexplored. This raises two key design questions: where should the degraded branch be constructed inside the LVLM, and how should visual evidence be degraded at the feature level?

To answer this question, we analyze the crossmodal information flow between visual and text tokens in the decoder(Zhang et al., 2025b; Jiang et al., 2025b). We observe that perturbing visual tokens near the output layers has only a limited impact on final predictions, whereas perturbing text tokens leads to severe generation collapse(Chen et al., 2024a; Fan et al., 2025). This suggests that, in deeper layers, task-relevant visual evidence has largely been integrated into text-side representations. Meanwhile, the layer-wise evolution of text-to-vision attention shows that reliable visual grounding mainly emerges in the middle layers. Therefore, as illustrated in Figure 2, the degraded branch should be constructed around the middle decoder layers. Furthermore, Table 1 shows that register-based degradation provides the most effective degraded branch. This leads to the second design question: how should we degrade the visual condition after branching? Table 1 shows that Global Info. achieves better hallucination mitigation than Local Info., which still relies on residual local visual evidence. This suggests that a more suitable degraded branch should preserve the global semantics of the input image, so that it remains image-relevant, while removing fine-grained local evidence to prevent reliable local grounding. Such a global-but-local-missing condition can more effectively expose hallucination-prone signals.

Based on these analyses, we propose YARD, a training-free Y-Architecture contrastive decoding framework. YARD realizes feature-level degradation from two perspectives: where to degrade and how to degrade. First, for where to degrade, YARD shares the shallow decoder layers between the clean and degraded branches and splits them at a middle layer. This allows the degraded branch to inherit early multimodal context while preventing it from fully integrating local visual evidence. Second, for how to degrade, YARD constructs a feature-level visual condition that preserves global information while removing local information. Specifically, prior studies (Darcet et al., 2024) have observed abnormally high-norm attention-sink tokens in vision encoders, which tend to absorb and carry coarse image-level information. YARD applies sink-shift to redirect these sink activations to newly introduced register tokens, enabling them to inherit the role of global-information carriers. These register representations are then used to replace fine-grained visual tokens in the degraded branch. In addition, YARD improves inference efficiency by avoiding a complete second forward pass. Empirically, YARD consistently reduces hallucinations in both generative and discriminative evaluations. These gains transfer across diverse LVLM architectures, demonstrating that YARD provides a general training-free contrastive signal rather than an architecture-specific decoding heuristic. Our contributions are summarized as follows:

![](images/ea83462c768da37cadd386811cfbf3a021b0b63ef191270c5276c8c888199297.jpg)

<details>
<summary>bar</summary>

| Category | Accuracy (%) | F1 (%) |
| :--- | :--- | :--- |
| Baseline | 82.0 | 80.4 |
| Late V=0 | 82.0 | 80.4 |
| Late T=0 | 0.0 | 0.0 |
| Early V=0 | 54.7 | 64.0 |
</details>

(a)

![](images/85b865f6a64ab59ea233afc74509b4c7a88e5c367a92a317410a7e254948e91f.jpg)

<details>
<summary>text_image</summary>

Is there a Knife?
</details>

![](images/0a555a87340069b1bb4d24d1763c90026b1abac1ead338692f56f09f5963b4f2.jpg)

<details>
<summary>natural_image</summary>

Wedding scene with bride and groom at a table, no visible text or symbols
</details>

![](images/84137a28b45189201c69d719a7d0265de449967a1085200f5089bce8d11296b9.jpg)

<details>
<summary>natural_image</summary>

Wedding scene with bride and groom at a table, illuminated by warm lights (no text or symbols visible)
</details>

(b)

![](images/9392821d04b0a064088eb6dbfa1c150a3972a050aa706f79bc7cf5841f974114.jpg)

<details>
<summary>natural_image</summary>

Two people in wedding attire preparing food at a table, with a cake and decorative lights visible (no text or symbols)
</details>

Figure 2: Motivating analysis of visual evidence flow. (a) Zero-out intervention shows that late-layer predictions mainly rely on text-side representations. (b) Text-to-vision attention becomes more grounded at the middle layer, e.g., layer 10, indicating the key window for local visual evidence extraction.

• We propose register-based feature-level degradation to construct an image-aware but locally under-grounded contrastive branch without corrupting the visual input.   
• We analyze degraded-branch purity and crossmodal information flow, motivating faithful degradation at middle decoder layers.   
• We present YARD, an efficient training-free Y-Architecture contrastive decoding framework that consistently mitigates hallucinations across benchmarks and LVLMs.

# 2 Related Work

# 2.1 Contrastive Decoding for Hallucination Mitigation

Training-free contrastive decoding mitigates hallucinations by contrasting a clean branch with a hallucination-prone degraded branch, and suppressing predictions overly favored by the degraded condition. Existing methods mainly differ in how this degraded branch is constructed.

One line of work applies degradation at the input level. VCD (Leng et al., 2024) perturbs images with Gaussian noise, while ICD (Wang et al., 2024b), DCD (Wu et al., 2025), VACoDe (Kim et al., 2024), and Octopus(Suo et al., 2025) introduce instruction, multi-source, or adaptive perturbations. These methods are effective but typically require largely separate clean and degraded computation paths, and their degradation is only indirectly related to the internal grounding process.

Another line of work obtains contrastive or corrective signals from internal states or decoding dynamics. DoLa (Chuang et al., 2024) contrasts logits across decoder layers, LayerCD (Tong et al., 2025) extends layer-wise contrast to visual representations, and OPERA (Huang et al., 2024) and ECD (Fieback et al., 2025) use decoding-time regularization or auxiliary scoring. While avoiding explicit input corruption, they do not explicitly construct a degraded visual branch at the stage where text-to-vision grounding is formed.

In contrast, YARD constructs the degraded branch inside the LLM decoder. By branching at the middle layers, it aligns degradation with the formation of visual grounding while sharing shallow-layer computation between clean and degraded branches.

# 2.2 Register Tokens in Vision Transformers

Register tokens were introduced to address highnorm outlier artifacts in Vision Transformers(Xiao et al., 2024; Sun et al., 2024). Darcet et al. (Darcet et al., 2024) showed that such outliers often emerge in spatially low-informative regions of ViTs such as DINOv2(Oquab et al., 2023) and CLIP(Radford et al., 2021), and proposed learnable registers to absorb them. Jiang et al. (Jiang et al., 2025a) further showed that register-like tokens can be constructed at test time without retraining by redirecting anomalous MLP activations to an appended token.

Different from prior work that uses registers to stabilize clean visual representations, YARD uses them as degraded visual conditions for contrastive decoding. Since register representations preserve coarse global semantics but lack stable patch-level correspondence, they naturally form an image-aware but locally unreliable branch for hallucination mitigation.

# 3 Preliminaries

Training-free register tokens. Given an image, a Vision Transformer produces patch tokens $\mathbf { Z } =$ $\{ z _ { 1 } , \dots , z _ { N } \}$ , where each $z _ { j } \in \mathbb { R } ^ { d }$ represents a local image patch. Prior work shows that ViTs can produce high-norm outlier tokens at spatially uninformative regions, and introduces register tokens to absorb such artifacts (Darcet et al., 2024). Jiang et al. (Jiang et al., 2025a) further show that register tokens can be constructed at test time without retraining, by appending an extra token and redirecting anomalous MLP activations to it.

In this work, we use register tokens not to improve the clean visual representation, but to construct degraded visual conditions for contrastive decoding. Compared with ordinary patch tokens, register tokens preserve coarse global visual semantics but lack stable patch-level correspondence and fine-grained local details. This makes them suitable for forming a branch that remains imageaware while being locally under-grounded. Details are provided in Appendix B.

# 4 Motivating Analysis

Observation 1: Middle Layers Are the Critical Window for Visual Evidence Transfer We first examine where a degraded branch should be constructed inside the LVLM decoder. Although vision tokens provide the initial visual source, final generation may increasingly rely on text-token representations, as text tokens progressively absorb task-relevant visual evidence through causal selfattention.

To verify this, we zero out either vision or text hidden states at decoder layer ℓ:

$$
\mathbf {H} _ {\backslash Z} ^ {(\ell)} = [ \mathbf {0}; \mathbf {T} ^ {(\ell)} ], \quad \mathbf {H} _ {\backslash T} ^ {(\ell)} = [ \mathbf {Z} ^ {(\ell)}; \mathbf {0} ], \tag {1}
$$

where $\mathbf { H } ^ { ( \ell ) } = [ \mathbf { Z } ^ { ( \ell ) } ; \mathbf { T } ^ { ( \ell ) } ]$ . The intervened states are then passed through the remaining decoder layers.

As shown in Figure 2(a), zeroing out vision tokens at the first layer leads to a clear performance drop, confirming that vision tokens are indispensable as the source of visual evidence. However, zeroing out vision tokens near the output layers has only a limited effect on final predictions. In contrast, zeroing out text tokens causes complete generation failure, with both accuracy and F1 dropping to 0. This suggests that, by the late decoder layers, task-relevant visual evidence has largely been transferred into text-side representations. Therefore, perturbing vision tokens after this transfer provides only a limited basis for constructing an effective degraded branch.

This finding further raises a layer-wise question: when does visual evidence become effectively transferred into text tokens? To answer this, we visualize text-to-vision attention across decoder layers by projecting it back to the image space. As shown in Figure 2(b), shallow-layer attention is diffuse and often distracted by irrelevant regions, whereas middle-layer attention becomes concentrated on task-relevant objects. This suggests that the middle layers form the key window where text tokens begin to acquire local visual evidence before it is fully internalized in deeper text-side representations. Therefore, the degraded branch should be constructed around the middle decoder layers: branching too early lacks sufficient cross-modal context, while branching too late leaves limited room for weakening visual evidence. This analysis determines where to degrade; we next examine how to construct the degraded branch to better expose hallucination-related signals.

Table 1: Ablation of degraded-branch construction. Pixel-level corrupts the input image; Text-only removes visual tokens; Local Info. randomly keeps a subset of vision tokens; Global Info. replaces fine-grained visual tokens with register representations. Parentheses report the hallucination increase after removing α. $\mathrm { C H } _ { s }$ and $\mathrm { C H } _ { i }$ are CHAIR-based metrics for evaluating hallucinations in image captions. 

<table><tr><td rowspan="2">Degradation</td><td colspan="2">w/ α</td><td colspan="2">w/o α</td></tr><tr><td>CHs↓</td><td>CHi↓</td><td>CHs↓</td><td>CHi↓</td></tr><tr><td>Pixel-level</td><td>23.0</td><td>8.3</td><td>28.7 (+5.7)</td><td>12.1 (+3.8)</td></tr><tr><td>Text-only</td><td>21.7</td><td>7.0</td><td>24.0 (+2.3)</td><td>7.9 (+0.9)</td></tr><tr><td colspan="5">Feature-level degraded conditions</td></tr><tr><td>Local Info.</td><td>25.0</td><td>7.9</td><td>26.4 (+1.4)</td><td>8.4 (+0.5)</td></tr><tr><td>Global Info.</td><td>19.7</td><td>6.5</td><td>21.0 (+1.3)</td><td>7.0 (+0.5)</td></tr></table>

Observation 2: Feature-level degradation exposes a purer hallucination signal. Table 1 compares different degraded-branch constructions from the perspective of signal purity. To more strictly examine whether the degraded branch precisely exposes hallucination-prone logits, we remove the conservative clean-logit anchoring controlled by α in standard contrastive decoding and adopt a direct subtraction form:

$$
\ell_ {i} ^ {\mathrm{w/o} \alpha} = \ell_ {i} ^ {c} - \ell_ {i} ^ {d}. \tag {2}
$$

This setting is not intended to achieve the best decoding performance, but instead serves as a stress test for the degraded branch. Ideally, the clean logits contain both correct predictions supported by genuine visual evidence and hallucination-prone predictions caused by insufficient grounding, while the degraded logits should be concentrated on the latter. In this case, even without the additional clean-branch protection introduced by α, direct subtraction should mainly suppress hallucinationrelated components without substantially damaging visually grounded predictions.

![](images/ebf10e6ce29ebdc6f28679f362cb2e54cc32435d3cd2beac9092377d9ce6f043.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Original Visual Inputs V"] --> B["Vision Encoder"]
    B --> C["LLM Decoder Layer 1"]
    C --> D["LLM Decoder Layer k"]
    D --> E["Clean Decoder"]
    E --> F["Degraded Decoder"]
    F --> G["Clean logits (y | x, v)"]
    
    H["What is on the beach?"] --> I["Textual Input X"]
    
    J["Image: Original Visual Inputs V"] --> B
    K["Image: Projector"] --> L["Projection"]
    
    M["Contrastive Decoding"] --> N["(1 + a) · Clean logits - a · Degraded logits = YARD logits"]
    N --> O["People"]
    N --> P["Umbrellas"]
    N --> Q["Surfboard"]
    
    R["hallucinated object &quot;Surfboard&quot; eliminated"] --> S["Degraded logits (y | x, v')"]
    S --> T["People"]
    S --> U["Umbrellas"]
    S --> V["Surfboard"]
```
</details>

Figure 3: Overview of Y-Architecture Register Decoding.

The results show that pixel-level degradation produces the largest increase in $\mathrm { C H } _ { s } / \mathrm { C H } _ { i }$ under the w/o α setting, indicating that its degraded branch is contaminated by residual visual semantics and fails to precisely isolate hallucination components. Text-only degradation leads to a smaller increase, but this mainly comes from its image-agnostic nature after completely removing visual conditioning. Since its degraded logits are weakly aligned with the clean prediction for the current image, they contain fewer image-specific correct object signals as well as fewer image-specific hallucination signals; therefore, direct subtraction is less likely to damage visually grounded predictions, but the resulting contrastive signal is also less aligned with the hallucination components in the clean branch.

In contrast, feature-level degradation exhibits the smallest drop after removing α, suggesting that its register-conditioned branch induces the purest hallucination signal. By preserving global image semantics while weakening local grounding, its degraded logits contain fewer correct object signals supported by genuine visual evidence and are more concentrated on the hallucination component targeted by contrastive subtraction. Among feature-level variants, Table 1 shows that Global Info. outperforms Local Info. as the degraded condition. Local Info., obtained by randomly retaining a subset of vision tokens, may leak residual patch-level evidence into the degraded branch. In contrast, Global Info. preserves coarse image-level semantics while removing fine-grained local evidence, forming a cleaner global-but-local-missing condition. This explains why it achieves the lowest $\mathrm { C H } _ { s } / \mathrm { C H } _ { i }$ and the smallest hallucination increase after removing α.

# 5 Method

The observations above motivate a simple principle: rather than degrading the visual source itself, the degraded branch should intervene in the process by which text tokens extract fine-grained visual evidence from vision tokens. To this end, we propose YARD, a training-free Y-Architecture contrastive decoding framework that constructs the degraded branch inside the LLM decoder. The clean and degraded branches share the shallow decoder layers and split at a middle layer, where text-to-vision grounding has begun to emerge but fine-grained local evidence has not yet been fully integrated. After branching, the clean branch retains the original patch-level visual tokens, while the degraded branch discards them and conditions generation on register representations together with text tokens. This produces a branch that remains globally image-aware but locally under-grounded.

Given an image, the vision encoder produces patch-level visual tokens $\begin{array} { l l l } { \mathbf { Z } } & { = } & { \left\{ z _ { 1 } , \dots , z _ { N } \right\} } \end{array}$ . We construct training-free register tokens R = $\{ r _ { 1 } , \hdots , r _ { M } \}$ following Section 3, and denote the text tokens as $\mathbf { T } = \{ t _ { 1 } , \ldots , t _ { L } \}$ . The input sequence to the LLM decoder is

$$
\mathbf {X} ^ {(0)} = [ \mathbf {Z}; \mathbf {R}; \mathbf {T} ]. \tag {3}
$$

Here, Z provides fine-grained local visual evidence, while R provides a coarse but spatially imprecise visual condition.

Let the LLM decoder contain D layers, and let K denote the branching layer. YARD first processes the full sequence through the shared prefix:

$$
\mathbf {H} ^ {(K)} = F _ {1: K} \left(\mathbf {X} ^ {(0)}\right) = \left[ \mathbf {Z} ^ {(K)}; \mathbf {R} ^ {(K)}; \mathbf {T} ^ {(K)} \right]. \tag {4}
$$

This shared computation allows both branches to inherit the same early multimodal context, while avoiding the full second forward pass required by input-level contrastive decoding.

At layer K, YARD splits the computation into a clean branch and a degraded branch:

$$
\begin{array}{l} \mathbf {H} _ {c} ^ {(K)} = \left[ \begin{array}{l} \mathbf {Z} ^ {(K)}; \mathbf {R} ^ {(K)}; \mathbf {T} ^ {(K)} \end{array} \right], \tag {5} \\ \mathbf {H} _ {d} ^ {(K)} = [ \mathbf {R} ^ {(K)}; \mathbf {T} ^ {(K)} ]. \\ \end{array}
$$

The clean branch keeps the original visual tokens and therefore preserves access to fine-grained patch-level evidence. The degraded branch removes these patch tokens and relies on register representations, which preserve coarse image-level semantics but lack stable local correspondence.

The two branches are then forwarded through the remaining decoder layers:

$$
\mathbf {H} _ {c} ^ {(D)} = F _ {K + 1: D} \left(\mathbf {H} _ {c} ^ {(K)}\right), \tag {6}
$$

$$
\mathbf {H} _ {d} ^ {(D)} = F _ {K + 1: D} (\mathbf {H} _ {d} ^ {(K)}).
$$

Let $\mathbf { h } _ { c , i } ^ { ( D ) }$ and $\mathbf { h } _ { d , i } ^ { ( D ) }$ denote the hidden states used to predict the next token at step i in the clean and degraded branches, respectively. The vocabularylevel logits are obtained through the language modeling head $g _ { \mathrm { l m } }$ :

$$
\boldsymbol {\ell} _ {i} ^ {c} = g _ {\mathrm{lm}} (\mathbf {h} _ {c, i} ^ {(D)}), \quad \boldsymbol {\ell} _ {i} ^ {d} = g _ {\mathrm{lm}} (\mathbf {h} _ {d, i} ^ {(D)}). \tag {7}
$$

YARD then applies contrastive decoding with the register-conditioned degraded logits:

$$
\boldsymbol {\ell} _ {i} ^ {\text { yard }} = (1 + \alpha) \boldsymbol {\ell} _ {i} ^ {c} - \alpha \boldsymbol {\ell} _ {i} ^ {d}, \tag {8}
$$

where α controls the contrastive strength. The final decoding distribution is obtained by applying softmax to ℓyard. $\bar { \ell } _ { i } ^ { \mathrm { y a r d } }$ Following standard contrastive decoding practice, we use a clean-branch plausibility constraint to avoid over-promoting tokens with very low clean probability.

This design creates a dual degradation effect. First, the degraded branch inherits the shared shallow-to-middle representations, where text-tovision grounding has begun but has not yet fully integrated fine-grained local evidence. Second, after branching, it is explicitly deprived of patch-level visual tokens and can only rely on register-level global semantics. As a result, the degraded branch remains semantically coherent with the image, but lacks the local visual evidence required for faithful grounding. Predictions that are plausible under this locally under-grounded branch, but insufficiently supported by the clean branch, are therefore suppressed during contrastive decoding.

# 6 Experiments

We evaluate YARD on both generative and discriminative hallucination benchmarks. For generative evaluation, we use AMBER(Wang et al., 2023), Object HalBench(Yu et al., 2024), and MME-Hallucination(Fu et al., 2025); for discriminative evaluation, we use AMBER discrimination(Wang et al., 2023) and POPE(Li et al., 2023b). Using LLaVA-1.5(Liu et al., 2024b) as the main backbone, we compare YARD with representative training-free methods, including ICD(Wang et al., 2024b), OPERA(Huang et al., 2024), VCD(Leng et al., 2024), M3ID(Favero et al., 2024), AVISC(Woo et al., 2025), EVAS (Zhang et al., 2025a), FuzzyCD(Kim et al., 2025), and TAME (Tang et al., 2025). We further evaluate transferability on LLaVA-NeXT(Liu et al., 2024c), Qwen2-VL(Wang et al., 2024a), Qwen3-VL(Bai et al., 2025), InstructBLIP(Dai et al., 2023), and Mini-Gemini(Li et al., 2025b).

Algorithm 1 YARD Contrastive Decoding 

<table><tr><td>Require: Image I, text query T, branching layer K, contrastive strength α</td></tr><tr><td>Ensure: Model response with register-conditioned contrastive decoding</td></tr><tr><td>1: Encode I into patch tokens Z and register tokens R</td></tr><tr><td>2: Run shared decoder prefix:  $\mathbf{H}^{(K)} = F_{1:K}([Z;R;T])$ </td></tr><tr><td>3: Split branches:  $\mathbf{H}_{c}^{(K)} = [\mathbf{Z}^{(K)};\mathbf{R}^{(K)};\mathbf{T}^{(K)}],\mathbf{H}_{d}^{(K)} = [R^{(K)};T^{(K)}]$ </td></tr><tr><td>4: Obtain  $\ell_{i}^{c},\ell_{i}^{d}$  from remaining layers and decode with  $\ell_{i}^{\text{yard}} = (1+\alpha)\ell_{i}^{c}-\alpha\ell_{i}^{d}$ </td></tr></table>

# 6.1 Main Results

Does YARD reduce hallucinations more effectively? Tables 2 and 3 compare YARD with existing hallucination mitigation methods. Overall, YARD achieves consistent improvements on both generative and discriminative hallucination evaluations, indicating that the proposed degraded branch provides an effective contrastive signal.

On the generative benchmarks in Table 2, YARD substantially reduces object-level hallucination. On Object HalBench, it achieves the best CHs and CHi, reducing them from 27.0/10.5 to 19.7/6.5 compared with the LLaVA-1.5-7B baseline. YARD also improves AMBER hallucination metrics, lowering CH and Hal. while achieving the best coverage and cognition scores. These results suggest that YARD suppresses hallucinated content without simply making the model overly conservative. On MME-Hallucination, YARD remains competitive with the strongest baselines and obtains the best color score, showing that it preserves general visual perception ability.

Table 3 shows that YARD also improves object-existence discrimination. On LLaVA-1.5- 7B, YARD improves AMBER accuracy/F1 from

Table 2: Generative hallucination evaluation on LLaVA-1.5-7B across AMBER, Object HalBench, and MME. Cov. measures object coverage, Hal measures the response-level hallucination rate, Cog reflects the tendency to generate plausible but image-unfaithful objects; Exist. and Pos. denote MME sub-scores for existence and position. 

<table><tr><td rowspan="2">Method</td><td colspan="4">AMBER</td><td colspan="2">Object HalBench</td><td colspan="5">MME-Hallucination</td></tr><tr><td>CH↓</td><td>Cov.↑</td><td>Hal↓</td><td>Cog↓</td><td>CHs↓</td><td>CHi↓</td><td>Exist.↑</td><td>Count↑</td><td>Pos.↑</td><td>Color↑</td><td>Total↑</td></tr><tr><td>LLaVA-1.5-7B</td><td>8.3</td><td>45.0</td><td>32.0</td><td>2.2</td><td>27.0</td><td>10.5</td><td>180.00</td><td>110.00</td><td>121.33</td><td>141.67</td><td>553.00</td></tr><tr><td>+ICD</td><td>6.3</td><td>46.3</td><td>25.8</td><td>2.2</td><td>22.3</td><td>7.4</td><td>180.00</td><td>126.67</td><td>113.33</td><td>143.33</td><td>563.33</td></tr><tr><td>+OPERA</td><td>4.6</td><td>45.6</td><td>18.9</td><td>1.6</td><td>28.3</td><td>12.1</td><td>190.00</td><td>133.33</td><td>121.67</td><td>155.00</td><td>600.00</td></tr><tr><td>+VCD</td><td>6.1</td><td>46.7</td><td>25.8</td><td>2.1</td><td>23.0</td><td>8.3</td><td>185.00</td><td>126.67</td><td>121.67</td><td>135.00</td><td>568.33</td></tr><tr><td>+M3ID</td><td>5.4</td><td>48.7</td><td>25.5</td><td>1.4</td><td>26.3</td><td>9.2</td><td>173.33</td><td>106.67</td><td>96.67</td><td>155.00</td><td>531.67</td></tr><tr><td>+AVISC</td><td>7.1</td><td>45.2</td><td>27.6</td><td>2.1</td><td>24.1</td><td>7.8</td><td>180.00</td><td>138.33</td><td>123.33</td><td>158.33</td><td>599.99</td></tr><tr><td>+EVAS</td><td>4.7</td><td>46.1</td><td>18.8</td><td>1.6</td><td>27.0</td><td>11.3</td><td>190.00</td><td>140.00</td><td>121.67</td><td>160.33</td><td>612.00</td></tr><tr><td>+FuzzyCD</td><td>6.2</td><td>48.1</td><td>27.3</td><td>1.8</td><td>20.3</td><td>7.3</td><td>185.00</td><td>148.33</td><td>123.33</td><td>153.33</td><td>609.99</td></tr><tr><td>+TAME</td><td>12.0</td><td>40.6</td><td>35.6</td><td>3.4</td><td>26.3</td><td>10.6</td><td>190.00</td><td>133.66</td><td>128.33</td><td>163.00</td><td>614.99</td></tr><tr><td>+YARD</td><td>4.6</td><td>48.9</td><td>21.6</td><td>1.2</td><td>19.7</td><td>6.5</td><td>185.00</td><td>140.00</td><td>125.00</td><td>163.33</td><td>613.33</td></tr></table>

Table 3: Discriminative hallucination evaluation on AMBER and POPE. Accuracy measures the proportion of correctly answered yes/no questions, while F1 balances precision and recall. 

<table><tr><td rowspan="3">Method</td><td colspan="2">AMBER</td><td colspan="8">POPE.MSCOCO</td></tr><tr><td colspan="2">Discrimination</td><td colspan="2">Random</td><td colspan="2">Popular</td><td colspan="2">Adversarial</td><td colspan="2">ALL</td></tr><tr><td>Accuracy</td><td>F1</td><td>Accuracy</td><td>F1</td><td>Accuracy</td><td>F1</td><td>Accuracy</td><td>F1</td><td>Accuracy</td><td>F1</td></tr><tr><td>LLaVA-1.5-7B</td><td>67.00</td><td>71.10</td><td>85.67</td><td>83.71</td><td>84.61</td><td>82.47</td><td>81.51</td><td>80.54</td><td>83.93</td><td>82.24</td></tr><tr><td>+ICD</td><td>75.92</td><td>81.58</td><td>85.20</td><td>84.02</td><td>83.70</td><td>82.64</td><td>81.20</td><td>80.45</td><td>83.37</td><td>82.37</td></tr><tr><td>+VCD</td><td>67.30</td><td>71.10</td><td>86.77</td><td>85.50</td><td>84.93</td><td>83.81</td><td>82.33</td><td>81.53</td><td>84.68</td><td>83.61</td></tr><tr><td>+M3ID</td><td>67.25</td><td>70.90</td><td>84.5</td><td>82.55</td><td>82.93</td><td>81.12</td><td>79.93</td><td>78.50</td><td>82.46</td><td>80.72</td></tr><tr><td>+AVISC</td><td>70.70</td><td>75.45</td><td>86.77</td><td>85.29</td><td>85.23</td><td>83.86</td><td>82.57</td><td>81.48</td><td>84.86</td><td>83.55</td></tr><tr><td>+EVAS</td><td>77.93</td><td>83.81</td><td>85.74</td><td>84.49</td><td>84.97</td><td>83.36</td><td>82.03</td><td>80.74</td><td>84.25</td><td>82.86</td></tr><tr><td>+YARD</td><td>78.70</td><td>84.10</td><td>89.03</td><td>88.41</td><td>86.67</td><td>86.25</td><td>82.73</td><td>82.90</td><td>86.14</td><td>85.85</td></tr><tr><td>InstructBLIP</td><td>68.20</td><td>74.60</td><td>82.61</td><td>82.47</td><td>79.51</td><td>79.59</td><td>78.98</td><td>79.43</td><td>80.37</td><td>80.50</td></tr><tr><td>+VCD</td><td>69.65</td><td>75.90</td><td>86.01</td><td>84.96</td><td>83.33</td><td>82.49</td><td>81.03</td><td>80.57</td><td>83.49</td><td>82.68</td></tr><tr><td>+AVISC</td><td>72.60</td><td>78.60</td><td>85.67</td><td>84.31</td><td>83.57</td><td>82.41</td><td>80.90</td><td>80.14</td><td>83.38</td><td>82.29</td></tr><tr><td>+YARD</td><td>73.62</td><td>79.07</td><td>90.77</td><td>90.05</td><td>89.07</td><td>88.43</td><td>87.33</td><td>86.87</td><td>89.06</td><td>88.45</td></tr></table>

67.00/71.10 to 78.70/84.10, and POPE accuracy/F1 from 83.93/82.24 to 86.14/85.85. The same trend holds on InstructBLIP, where YARD improves POPE overall accuracy/F1 from 80.37/80.50 to 89.06/88.45. This demonstrates that YARD is effective not only for reducing hallucinated descriptions, but also for discriminative hallucination evaluation.

Taken together, these results support our design hypothesis: an image-aware but locally undergrounded degraded branch provides a more targeted contrastive signal, suppressing predictions that are plausible under weak local grounding but unsupported by clean visual evidence.

# 6.2 Transfer Across LVLM Architectures

Does YARD generalize to different LVLM families? We further evaluate YARD as a plugand-play decoding method across diverse LVLM architectures, including Qwen3-VL, Qwen2-VL, LLaVA-NeXT, LLaVA-v1.5, InstructBLIP, and Mini-Gemini. As shown in Table 4, YARD consistently improves both Object HalBench and POPE across all evaluated models.

The gains are observed on both strong and hallucination-prone backbones. For example, on Qwen3-VL-4B, YARD improves POPE F1/Acc. from 89.2/89.8 to 92.7/93.6, while reducing CH /CH from 8.7/6.1 to 7.5/5.4. On LLaVA-NeXT, where object hallucination is more severe, YARD reduces $\mathrm { C H } _ { s } / \mathrm { C H } _ { i }$ from 20.5/9.3 to 11.0/5.2.

These consistent improvements suggest that YARD is not tied to a specific LLaVA-style architecture. Instead, middle-layer register-conditioned degradation captures a general property of LVLM decoding, enabling training-free transfer across different model families.

![](images/d93c01efba8e5a129b36df21c8518548bae3a0f2f33ddb94a5f28fd80c961352.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a bus stop with multiple vintage and red trains parked on grassy ground under clear blue sky (no signage or text visible)
</details>

Q: Please describe this image in detail.

Baseline: The image conveys the story of the two school buses, one red and blue, parked next to each other in the grass or dirt. There’s a boy near the blue bus on the right, possibly disembarking or about to board either bus. Additionally, a single car and a row of parked buses are visible in the open area.

YARD: The image shows buses parked next to each other, lined up on a grassy area near a parking lot. The buses appear to be in various states of maintenance and come in various colors. Large trees and a clear blue sky can be seen in the background, giving the scene an open outdoor setting.

Figure 4: Qualitative comparison between baseline and Yard. Hallucinated sentences are highlighted in red.

# 6.3 Qualitative Analysis

# Are YARD responses more visually grounded?

Figure 4 shows a representative open-ended generation example. The baseline response introduces unsupported objects, such as a boy near the bus and a single car, which are plausible in the scene but not grounded in the image. In contrast, YARD focuses on visually supported content, including the buses, grassy area, trees, and sky, without adding non-existent objects.

This example illustrates how YARD suppresses language-plausible but visually unsupported predictions. By contrasting the clean branch with an image-aware but locally under-grounded degraded branch, YARD encourages responses that remain more faithful to the visual evidence.

# 6.4 Ablation Study

Are middle-layer branching and registerconditioned degradation necessary? Table 5 ablates two key design choices of YARD: the branching layer K and the degraded visual condition.

First, the branching layer is critical. Branching too early (K = 1) provides insufficient multimodal context, leading to weaker POPE performance and higher CHAIR scores. Branching too late (K = 25) is also suboptimal, as visual evidence has been largely absorbed into text-side representations, leaving limited room for degradation. In contrast, middle-layer branching (K = 10) achieves the best overall performance, supporting our observation that this stage is the critical window for local visual evidence extraction.

Table 4: Cross-architecture evaluation of YARD on Object HalBench and POPE across diverse VLM backbones. 

<table><tr><td rowspan="2">Model</td><td colspan="2">Object HalBench</td><td colspan="2">POPE</td></tr><tr><td> $CH_s \downarrow$ </td><td> $CH_i \downarrow$ </td><td>F1↑</td><td>Acc.↑</td></tr><tr><td>Qwen3-VL-4B</td><td>8.7</td><td>6.1</td><td>89.2</td><td>89.8</td></tr><tr><td>+ YARD</td><td>7.5</td><td>5.4</td><td>92.7</td><td>93.6</td></tr><tr><td>Qwen3-VL-8B</td><td>7.9</td><td>5.4</td><td>88.4</td><td>88.9</td></tr><tr><td>+ YARD</td><td>7.3</td><td>5.0</td><td>91.2</td><td>91.7</td></tr><tr><td>Qwen2-VL-7B</td><td>12.7</td><td>7.8</td><td>87.1</td><td>87.9</td></tr><tr><td>+ YARD</td><td>10.8</td><td>6.3</td><td>88.6</td><td>89.2</td></tr><tr><td>LLaVA-NeXT</td><td>20.5</td><td>9.3</td><td>83.1</td><td>84.7</td></tr><tr><td>+ YARD</td><td>11.0</td><td>5.2</td><td>86.5</td><td>87.3</td></tr><tr><td>LLaVA-v1.5</td><td>27.0</td><td>10.5</td><td>82.2</td><td>83.9</td></tr><tr><td>+ YARD</td><td>19.7</td><td>6.5</td><td>85.9</td><td>86.1</td></tr><tr><td>InstructBLIP</td><td>15.3</td><td>5.5</td><td>80.5</td><td>80.4</td></tr><tr><td>+ YARD</td><td>12.5</td><td>4.2</td><td>88.5</td><td>89.1</td></tr><tr><td>Mini-Gemini</td><td>14.0</td><td>6.9</td><td>85.1</td><td>86.6</td></tr><tr><td>+ YARD</td><td>13.2</td><td>6.5</td><td>86.4</td><td>87.1</td></tr></table>

Second, the degraded condition also matters. Average degradation remains inferior to YARD, suggesting that coarse averaged features are not sufficiently informative. Text-only variants remove visual information more aggressively and provide a less image-specific contrastive signal; although they reduce CHAIR to some extent, they yield weaker POPE Acc./F1. Overall, YARD achieves the best POPE Acc./F1 and the lowest CHs/CHi, confirming that middle-layer branching and register-conditioned degradation are both necessary for constructing an image-aware but locally under-grounded degraded branch.

Table 5: Ablation study of degraded-branch construction. K denotes the branching layer; Average uses averaged visual features, and Text-Only removes visual tokens. Acc. and F1 are averaged over POPE subsets; lower CHs/CHi is better. 

<table><tr><td rowspan="2">Variant</td><td colspan="2">POPE</td><td colspan="2">CHAIR</td></tr><tr><td>Acc.↑</td><td>F1↑</td><td>CH $_{s}$ ↓</td><td>CH $_{i}$ ↓</td></tr><tr><td>YARD (K=1)</td><td>84.5</td><td>83.7</td><td>22.9</td><td>7.3</td></tr><tr><td>YARD (K=25)</td><td>83.2</td><td>81.6</td><td>25.0</td><td>7.9</td></tr><tr><td>Average. (K=10)</td><td>84.9</td><td>84.3</td><td>23.7</td><td>7.7</td></tr><tr><td>Text-Only. (K=1)</td><td>84.3</td><td>82.6</td><td>23.0</td><td>7.2</td></tr><tr><td>Text-Only. (K=10)</td><td>85.1</td><td>84.2</td><td>21.7</td><td>7.0</td></tr><tr><td>YARD (K=10)</td><td>86.1</td><td>85.9</td><td>19.7</td><td>6.5</td></tr></table>

# 7 Conclusion

We propose YARD, a training-free Y-Architecture contrastive decoding framework. YARD introduces register-based feature-level degradation at the middle decoder layers to construct an image-aware but locally under-grounded degraded branch, while sharing shallow-layer computation to avoid a complete second forward pass. Experiments show that YARD consistently reduces hallucinations across multiple LVLM architectures and benchmarks.

# Limitations

YARD is training-free at the decoding stage, but it relies on constructing effective register representations for the target LVLM. While this design is effective for the Transformer-based LVLMs evaluated in this work, architectures with different vision encoders, LLM backbones, or vision-language projection mechanisms may require additional adaptation. In addition, the performance of YARD may be affected by the choice of branching layer and the construction of register representations. Extending register-based degradation to broader multimodal architectures and input formats is a natural direction for future work.

# References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.   
Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930.   
Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024a. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.   
Zhaorun Chen, Zhuokai Zhao, Hongyin Luo, Huaxiu Yao, Bo Li, and Jiawei Zhou. 2024b. Halc: Object hallucination reduction via adaptive focal-contrast decoding. arXiv preprint arXiv:2403.00425.   
Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and Pengcheng He. 2024. Dola: Decoding by contrasting layers improves factuality in large language models. In International Conference on Learning Representations, volume 2024, pages 54158–54183.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267.   
Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. 2024. Vision transformers need registers. In International Conference on Learning Representations, volume 2024, pages 2632–2652.   
Jingyuan Deng and Yujiu Yang. 2025. Maskcd: Mitigating lvlm hallucinations by image head masked contrastive decoding. arXiv preprint arXiv:2510.02790.   
Yingqi Fan, Anhao Zhao, Jinlan Fu, Junlong Tong, Hui Su, Yijie Pan, Wei Zhang, and Xiaoyu Shen. 2025. Visipruner: Decoding discontinuous cross-modal dynamics for efficient multimodal llms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 18896–18913.   
Alessandro Favero, Luca Zancato, Matthew Trager, Siddharth Choudhary, Pramuditha Perera, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto. 2024. Multi-modal hallucination control by visual information grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14303–14312.   
Laura Fieback, Nishilkumar Balar, Jakob Spiegelberg, and Hanno Gottschalk. 2025. Efficient contrastive decoding with probabilistic hallucination detectionmitigating hallucinations in large vision language models. arXiv preprint arXiv:2504.12137.   
Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2025. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38.   
Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, and 1 others. 2024. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14375–14385.   
Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. 2024. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13418– 13427.   
Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Fushuo Huo, Wenchao Xu, Zhong Zhang, Haozhao Wang, Zhicheng Chen, and Peilin Zhao. 2024. Selfintrospective decoding: Alleviating hallucinations for large vision-language models. arXiv preprint arXiv:2408.02032.   
Nicholas Jiang, Amil Dravid, Alexei Efros, and Yossi Gandelsman. 2025a. Vision transformers don’t need trained registers. Advances in neural information processing systems, 38:56557–56595.   
Zhangqi Jiang, Junkai Chen, Beier Zhu, Tingjin Luo, Yankun Shen, and Xu Yang. 2025b. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25004–25014.   
Jieun Kim, Jinmyeong Kim, Yoonji Kim, and Sung-Bae Cho. 2025. Fuzzy contrastive decoding to alleviate object hallucination in large vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20572–20581.   
Sihyeon Kim, Boryeong Cho, Sangmin Bae, Sumyeong Ahn, and Se-Young Yun. 2024. Vacode: Visual augmented contrastive decoding. arXiv preprint arXiv:2408.05337.   
Sicong Leng, Yun Xing, Zesen Cheng, Yang Zhou, Hang Zhang, Xin Li, Deli Zhao, Shijian Lu, Chunyan Miao, and Lidong Bing. 2025. The curse of multimodalities: Evaluating hallucinations of large multimodal models across language, visual, and audio. Advances in Neural Information Processing Systems, 38.   
Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882.   
Jiaming Li, Jiacheng Zhang, Zequn Jie, Lin Ma, and Guanbin Li. 2025a. Mitigating hallucination for large vision language model by inter-modality correlation calibration decoding. arXiv preprint arXiv:2501.01926.   
Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023a. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pages 12286–12312.   
Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. 2025b. Mini-gemini: Mining the potential of multi-modality vision language models. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292– 305.   
Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024a. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.   
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024b. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306.   
Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024c. Llavanext: Improved reasoning, ocr, and world knowledge.   
Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024d. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.   
Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521.   
Avshalom Manevich and Reut Tsarfaty. 2024. Mitigating hallucinations in large vision-language models (lvlms) via language-contrastive decoding (lcd). In Findings of the Association for Computational Linguistics: ACL 2024, pages 6008–6022.   
Sean O’Brien and Mike Lewis. 2023. Contrastive decoding improves reasoning in large language models. arXiv preprint arXiv:2309.09117.   
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, and 1 others. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.   
Nhi Pham and Michael Schott. 2024. H-pope: Hierarchical polling-based probing evaluation of hallucinations in large vision-language models. arXiv preprint arXiv:2411.04077.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. 2018. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4035–4045.   
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.   
Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. 2024. Massive activations in large language models. arXiv preprint arXiv:2402.17762.   
Wei Suo, Lijun Zhang, Mengyang Sun, Lin Yuanbo Wu, Peng Wang, and Yanning Zhang. 2025. Octopus: Alleviating hallucination via dynamic contrastive decoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29904–29914.   
Barrett Tang, Zile Huang, Chengzhi Liu, Qiang Sun, Harry Yang, and Ser-Nam Lim. 2025. Intervening anchor token: Decoding strategy in alleviating hallucinations for mllms. In International Conference on Learning Representations, volume 2025, pages 27745–27776.   
Bingkui Tong, Jiaer Xia, and Kaiyang Zhou. 2025. Mitigating hallucination in multimodal llms with layer contrastive decoding. arXiv preprint arXiv:2509.25177.   
Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, and 1 others. 2023. Amber: An llmfree multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397.   
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024a. Qwen2- vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.   
Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. 2024b. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. In Findings of the Association for Computational Linguistics: ACL 2024, pages 15840–15853.   
Sangmin Woo, Donguk Kim, Jaehyuk Jang, Yubin Choi, and Changick Kim. 2025. Don’t miss the forest for the trees: Attentional vision calibration for large vision language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 1927–1951.   
Jiulong Wu, Yucheng Shen, Haixin Sun, and Min Cao. 2025. Mitigating hallucinations in large visionlanguage models via dual contrastive decoding. In Proceedings of the 7th ACM International Conference on Multimedia in Asia, pages 1–8.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, volume 2024, pages 21875–21895.   
Dingchen Yang, Bowen Cao, Guang Chen, and Changjun Jiang. 2024. Pensieve: Retrospect-thencompare mitigates visual hallucination. arXiv preprint arXiv:2403.14401.   
Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, and 1 others. 2024. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807– 13816.   
Xiaofeng Zhang, Yihao Quan, Chen Shen, Chaochen Gu, Xiaosong Yuan, Shaotian Yan, Jiawei Cao, Hao Cheng, Kaijie Wu, and Jieping Ye. 2025a. Shallow focus, deep fixes: Enhancing shallow layers vision attention sinks to alleviate hallucination in lvlms. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3512–3534.   
Yuhui Zhang, Alyssa Unell, Xiaohan Wang, Dhruba Ghosh, Yuchang Su, Ludwig Schmidt, and Serena Yeung-Levy. 2024. Why are visually-grounded language models bad at image classification? Advances in Neural Information Processing Systems, 37:51727– 51753.   
Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. 2025b. Cross-modal information flow in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19781–19791.   
Jianfei Zhao, Feng Zhang, Xin Sun, Lingxing Kong, Zhixing Tan, and Chong Feng. 2025. Cross-image contrastive decoding: Precise, lossless suppression of language priors in large vision-language models. arXiv preprint arXiv:2505.10634.   
Deyao Zhu, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny, and 1 others. 2024. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In International Conference on Learning Representations, volume 2024, pages 18378–18394.   
Lanyun Zhu, Deyi Ji, Tianrun Chen, Peng Xu, Jieping Ye, and Jun Liu. 2025. Ibd: Alleviating hallucinations in large vision-language models via imagebiased decoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1624–1633.

# A Appendix

Overview. This appendix provides additional analyses and implementation details for YARD. We first report the inference-time comparison A.1 to analyze the hallucination–efficiency trade-off, followed by the implementation details A.2 used across experiments. We then provide distributional analyses of clean and degraded branches, including both aggregate divergence metrics A.3 and a top-token case study A.8, to further explain why register-conditioned feature-level degradation forms a more suitable degraded branch than pixellevel or text-only alternatives. Next, we examine the sensitivity of YARD to the number of register tokens and the contrastive strength α A.4. We also evaluate YARD on general multimodal benchmarks A.5 to verify that hallucination mitigation is not achieved by sacrificing general visual understanding. Finally, we describe the evaluation benchmarks A.6, present a detailed formulation of YARD and contrastive decoding A.7, and provide additional qualitative examples.

A.1 Efficiency Analysis   
![](images/6ac7ac1fbd603d306a5ebae2eafe32c638e7ddecbd2859620c26d32b60ab92b2.jpg)

<details>
<summary>bar</summary>

| Method   | IT  | CHi |
| -------- | --- | --- |
| Baseline | 2.7 | 10.5 |
| VCD      | 5.2 | 4.1 |
| ICD      | 4.4 | 3.6 |
| Ours     | 3.3 | 6.4 |
</details>

Figure 5: Comparison of inference time and hallucination rate across decoding methods. YARD achieves lower CHi with only a modest inference overhead compared with the baseline, while prior contrastive decoding methods introduce substantially higher cost or hallucination.

We further analyze the inference cost of YARD compared with representative decoding-time hallucination mitigation methods. Figure 5 reports inference time (IT) and CHi on LLaVA-1.5-7B (Liu et al., 2024b). The vanilla baseline requires 2.6 seconds and obtains a CHi of 10.5. VCD (Leng et al., 2024) reduces $\mathrm { C H } _ { i }$ to 8.3, but requires an additional degraded forward pass, increasing inference time to 5.1 seconds. ICD (Wang et al., 2024b) further reduces CHi to 7.4, but still incurs a substantial inference overhead, requiring 4.3 seconds.

In contrast, YARD achieves the lowest CHi of 6.5 with only a modest increase in inference time from 2.6 to 3.2 seconds. This efficiency comes from the Y-Architecture design: the clean and degraded branches share the shallow decoder layers, and the degraded branch is only computed after the middle-layer split. Thus, YARD provides a stronger hallucination–efficiency trade-off than prior contrastive decoding methods that rely on largely separate degraded computation.

# A.2 Implementation Details

YARD is applied in a fully training-free manner without parameter updates, additional supervision, or external verifiers. Unless otherwise specified, we construct the degraded branch at approximately one-third of the LLM decoder depth, following our observation that text-to-vision grounding begins to emerge in the middle layers. For LLaVA-1.5- 7B(Liu et al., 2024b), this corresponds to branching after the 10-th decoder layer. For other backbones, we use the same one-third-depth rule to select the branching layer.

We use 1 register token to construct the degraded visual condition in all experiments. The contrastive strength is set to α = 1 by default. These hyperparameters are kept fixed across benchmarks unless explicitly stated otherwise. All evaluations are conducted with the original model weights and the default decoding settings of each backbone. All reported results are consistently reproduced across multiple independent runs, indicating that the observed improvements are stable rather than incidental.

Table 6: Distributional gap between clean and degraded branches. Top-1 Match measures whether the two branches predict the same top token. KL, symmetric KL, JS, and TV measure distributional divergence between clean and degraded output distributions. 

<table><tr><td>Method</td><td>Top-1 Match↓</td><td>KL↑</td><td>Sym. KL↑</td><td>JS↑</td><td>TV↑</td></tr><tr><td>Pixel-level</td><td>0.9233</td><td>0.0617</td><td>0.1246</td><td>0.0130</td><td>0.0682</td></tr><tr><td>Ours-Reg.</td><td>0.7121</td><td>0.7380</td><td>1.6726</td><td>0.1197</td><td>0.2740</td></tr><tr><td>Text-only</td><td>0.6675</td><td>0.8667</td><td>2.0323</td><td>0.1406</td><td>0.3067</td></tr></table>

# A.3 Distributional Analysis of Clean and Degraded Branches

To further understand the behavior of different degraded-branch constructions, we compare the output distributions of the clean and degraded branches in Table 6. Top-1 Match measures how often the two branches predict the same most likely token, while KL, symmetric KL, JS, and TV quantify the distributional discrepancy between the two output distributions. We use these metrics as diagnostic tools rather than direct performance measures: an effective degraded branch should be different enough from the clean branch to expose hallucination-prone predictions, but should not be so distant that it becomes unrelated to the current image.

Pixel-level degradation remains too close to the clean branch. It obtains a high Top-1 Match of 0.9233 and very small divergence values across ${ \mathrm { K L } } , { \mathrm { J S } }$ , and TV. This suggests that the pixel-level degraded branch still preserves substantial normal visual semantics and correct object evidence from the clean branch. Consequently, when the clean-logit anchoring term is removed in the w/o α stress test, direct subtraction may suppress not only hallucination-related logits but also visually grounded predictions, leading to a large increase in $\mathrm { C H } _ { s }$ and $\mathrm { C H } _ { i }$ in Table 1.

Text-only degradation exhibits the opposite behavior. It has the lowest Top-1 Match and the largest distributional divergences, indicating that its degraded logits are much farther from the clean branch. Although this strong shift can induce hallucination-prone predictions, the resulting distribution is largely image-agnostic and dominated by language priors. Therefore, it is less aligned with image-specific hallucination components in the clean branch and does not faithfully model the case where the model has seen the image but fails to use local visual evidence correctly.

Feature-level degradation with register tokens yields a more balanced distributional gap. Compared with pixel-level degradation, it produces a much larger discrepancy from the clean branch, suggesting that local grounding has been effectively weakened. Compared with text-only degradation, it remains closer to the clean branch, indicating that it still preserves image-level semantics rather than collapsing into a pure language prior. This intermediate behavior matches our intended degraded condition: image-aware but locally undergrounded. Together with the w/o α stress test, these results support that register-conditioned featurelevel degradation provides a more targeted contrastive signal for suppressing visually unsupported predictions.

Table 7: Sensitivity analysis on the number of register tokens. The shaded column denotes the default setting used in our experiments. Higher F1 is better, while lower $\mathrm { C H } _ { s }$ and $\mathrm { C H } _ { i }$ indicate fewer hallucinations. 

<table><tr><td># Reg. Tokens</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td>F1↑</td><td>85.9</td><td>86.0</td><td>85.9</td><td>86.1</td><td>86.0</td><td>85.7</td></tr><tr><td>CHs↓</td><td>19.7</td><td>20.3</td><td>21.7</td><td>20.7</td><td>19.3</td><td>21.3</td></tr><tr><td>CHi↓</td><td>6.5</td><td>6.5</td><td>7.0</td><td>7.1</td><td>6.2</td><td>6.7</td></tr></table>

# A.4 Parameter Sensitivity Analysis

Number of register tokens. Table 7 studies the effect of using different numbers of register tokens in the degraded branch. Overall, YARD is relatively robust to this hyperparameter: across 1 to 32 register tokens, POPE F1(Li et al., 2023b) remains within a narrow range from 85.7 to 86.1, while CHs and $\mathrm { C H } _ { i }$ also remain consistently low. Using too few register tokens may provide insufficient capacity to represent global image-level semantics, whereas using too many registers can introduce redundant degraded visual conditions and slightly weaken the contrastive signal. We choose 1 register token as the default setting.

Table 8: Sensitivity analysis on the contrastive strength $\alpha .$ The shaded column denotes the default setting used in our experiments. Higher F1 is better, while lower $\mathrm { C H } _ { s }$ and CHi indicate fewer hallucinations. 

<table><tr><td> $\alpha$ </td><td>w/o</td><td>0.25</td><td>0.5</td><td>0.75</td><td>1.0</td></tr><tr><td>F1↑</td><td>85.0</td><td>85.2</td><td>85.5</td><td>85.9</td><td>85.9</td></tr><tr><td>CH $_{s}$ ↓</td><td>21.0</td><td>20.3</td><td>17.7</td><td>18.0</td><td>19.7</td></tr><tr><td>CH $_{i}$ ↓</td><td>7.0</td><td>7.0</td><td>6.6</td><td>5.9</td><td>6.5</td></tr></table>

Contrastive strength $\alpha .$ Table 8 analyzes the sensitivity of YARD to the contrastive strength α. As α increases, the degraded-branch signal is more strongly subtracted, which generally improves both POPE F1(Li et al., 2023b) and hallucination metrics. For example, increasing α from 0 to 1.0 improves POPE F1(Li et al., 2023b) from 85.0 to 85.9 and reduces $\mathrm { C H } _ { i }$ from 7.0 to 6.5.

We use $\alpha = 1 . 0$ as the default setting, as it offers a stable trade-off between discriminative accuracy and hallucination reduction.

# A.5 General Multimodal Benchmark Results

Hallucination mitigation methods may reduce unsupported generations by making the model overly conservative or by weakening its general visual understanding ability. To examine whether YARD introduces such a trade-off, we additionally evaluate it on standard multimodal benchmarks that are not specifically designed for hallucination measurement. These benchmarks cover different aspects of general LVLM capability, including visual question answering, text-rich visual understanding, science question answering, and comprehensive multimodal perception.

Table 9: General multimodal benchmark results on LLaVA-1.5-7B. YARD preserves general visual understanding performance while improving hallucination robustness. Higher values indicate better performance. 

<table><tr><td>Benchmark</td><td>Baseline</td><td>YARD</td><td> $\Delta$ </td></tr><tr><td>GQA</td><td>61.9</td><td>62.1</td><td>+0.2</td></tr><tr><td>MMBench</td><td>64.7</td><td>64.9</td><td>+0.2</td></tr><tr><td>TextVQA</td><td>58.2</td><td>58.2</td><td>0.0</td></tr><tr><td> $SQA^I$ </td><td>69.5</td><td>69.6</td><td>+0.1</td></tr><tr><td>MME</td><td>1594</td><td>1741</td><td>+147</td></tr></table>

As shown in Table 9, YARD largely preserves the general multimodal performance of the base LLaVA-1.5-7B(Liu et al., 2024b) model. On GQA(Hudson and Manning, 2019), MMBench(Liu et al., 2024d), and SQAI, YARD achieves small but consistent improvements over the baseline. On TextVQA(Singh et al., 2019), where fine-grained text recognition and visual-linguistic alignment are important, YARD maintains the same performance as the baseline. These results suggest that the proposed register-conditioned degraded branch does not simply suppress visual information or make the model less responsive to image content.

Notably, YARD improves the MME(Fu et al., 2025) score from 1594 to 1741. Since MME evaluates a broad range of perception and cognition abilities, this improvement indicates that YARD can enhance hallucination robustness while preserving, and in some cases improving, general visual perception. Together with the hallucination-specific results in the main paper, these findings suggest that YARD provides a targeted contrastive signal: it suppresses visually unsupported predictions without broadly degrading the model’s multimodal understanding ability.

# A.6 Benchmark Details

AMBER(Wang et al., 2023). AMBER is a hallucination-oriented benchmark for evaluating both generative and discriminative hallucination in LVLMs. For generative evaluation, it reports hallucination-related metrics such as CH, Hal., and Cog., together with coverage, which reflects whether the model preserves sufficient visual content. For discriminative evaluation, it measures whether the model correctly judges visual content. We report both generative metrics and discrimination accuracy/F1.

Object HalBench / CHAIR(Yu et al., 2024). Object HalBench evaluates object hallucination in image captioning and open-ended generation. It measures whether generated object mentions are supported by the image. We report CHs and CHi, where lower values indicate fewer hallucinated objects at the sentence and instance levels. This benchmark directly reflects whether YARD suppresses visually unsupported object predictions.

MME-Hallucination(Fu et al., 2025). MME-Hallucination evaluates fine-grained hallucinationrelated perception abilities, including object existence, count, position, and color. These categories test whether the model can produce visually faithful responses under different grounding requirements. We report each subcategory score and the total score, where higher values indicate better performance.

POPE-MSCOCO(Li et al., 2023b). POPE-MSCOCO evaluates object hallucination through binary object-existence questions. It contains random, popular, and adversarial sampling settings, where the adversarial split includes objects more likely to trigger language-prior hallucination. We report accuracy and F1 for each split and the overall result.

GQA(Hudson and Manning, 2019). GQA is a visual question answering benchmark emphasizing real-world visual reasoning and compositional understanding. It covers objects, attributes, spatial relations, and multi-step reasoning. We use GQA to evaluate whether YARD preserves general visual reasoning beyond hallucination-specific settings.

MMBench(Liu et al., 2024d). MMBench evaluates general multimodal understanding across perception and reasoning categories. It contains multiple-choice questions covering diverse visual and linguistic abilities. We report overall accuracy to assess broad multimodal capability.

TextVQA(Singh et al., 2019). TextVQA evaluates visual question answering over text-rich images. It requires recognizing and reasoning over scene text, making it sensitive to fine-grained visual-text alignment. We use it to examine whether YARD preserves text-oriented visual understanding.

ScienceQA-IMG(Lu et al., 2022). ScienceQA-IMG is the image subset of ScienceQA. It covers science questions requiring both visual understanding and textual reasoning. We report accuracy on this subset to evaluate multimodal reasoning in scientific contexts.

MME(Fu et al., 2025). MME is a comprehensive benchmark for perceptual and cognitive abilities of multimodal large language models. It covers object recognition, counting, spatial reasoning, color perception, OCR, and commonsense reasoning. We report the total MME score to assess whether YARD preserves general multimodal perception while mitigating hallucinations.

# A.7 Detailed Formulation of YARD

This section provides a more detailed formulation of YARD. Given an input image I and a text query x, the vision encoder first produces a sequence of patch-level visual tokens

$$
\mathbf {Z} = \left[ \mathbf {z} _ {1}, \dots , \mathbf {z} _ {N} \right] \in \mathbb {R} ^ {N \times d _ {v}}, \tag {9}
$$

where each $\mathbf { z } _ { j }$ corresponds to a local image patch. The visual tokens are then mapped into the LLM hidden space through the multimodal projector $\phi ( \cdot )$ :

$$
\tilde {\mathbf {Z}} = \phi (\mathbf {Z}) \in \mathbb {R} ^ {N \times d}. \tag {10}
$$

For simplicity, we use Z to denote the projected visual tokens in the following derivation.

Training-free register construction. YARD constructs register tokens in a training-free manner. Following test-time register construction, we append M non-image register tokens to the visual sequence:

$$
\mathbf {R} ^ {(0)} = [ \mathbf {r} _ {1} ^ {(0)}, \dots , \mathbf {r} _ {M} ^ {(0)} ] \in \mathbb {R} ^ {M \times d}. \tag {11}
$$

These tokens are not associated with any spatial image patch. Instead, they are used to absorb global image-level information that would otherwise be stored in high-norm outlier patch tokens.

Let $\kappa _ { \mathrm { r e g } }$ denote the set of register neurons identified in the vision encoder. For a selected MLP layer, let $a _ { j , k }$ denote the activation of neuron k on patch token $j$ . For each register neuron $k \in \mathcal { K } _ { \mathrm { r e g } }$ , we redirect the maximum anomalous activation from patch tokens to the appended register token:

$$
a _ {r, k} \leftarrow \max _ {1 \leq j \leq N} a _ {j, k}, \tag {12}
$$

$$
a _ {j, k} \leftarrow 0, \quad \forall j \in \{1, \dots , N \}.
$$

This operation shifts high-norm activation away from image patch tokens and into the register tokens, yielding register representations that preserve coarse image-level semantics while lacking reliable patch-level correspondence. The final LLM input sequence is therefore

$$
\mathbf {X} ^ {(0)} = [ \mathbf {Z}; \mathbf {R}; \mathbf {T} ], \tag {13}
$$

where $\mathbf { T } = [ \mathbf { t } _ { 1 } , \dots , \mathbf { t } _ { L } ]$ denotes the text-token sequence.

Shared-prefix decoding. Let the LLM decoder contain D layers, denoted as

$$
F _ {1: D} = F _ {D} \circ F _ {D - 1} \circ \dots \circ F _ {1}. \tag {14}
$$

YARD chooses a branching layer K around the middle decoder layers, where text-to-vision grounding begins to emerge. Before branching, the clean and degraded branches share the same computation:

$$
\mathbf {H} ^ {(K)} = F _ {1: K} (\mathbf {X} ^ {(0)}) = [ \mathbf {Z} ^ {(K)}; \mathbf {R} ^ {(K)}; \mathbf {T} ^ {(K)} ]. \tag {15}
$$

This shared prefix serves two purposes. First, it allows both branches to inherit the same early multimodal context. Second, it avoids recomputing the shallow decoder layers, reducing the overhead compared with input-level contrastive decoding.

Y-Architecture branch construction. At layer K, YARD splits the computation into a clean branch and a degraded branch:

$$
\mathbf {H} _ {c} ^ {(K)} = [ \mathbf {Z} ^ {(K)}; \mathbf {R} ^ {(K)}; \mathbf {T} ^ {(K)} ], \tag {16}
$$

$$
\mathbf {H} _ {d} ^ {(K)} = [ \mathbf {R} ^ {(K)}; \mathbf {T} ^ {(K)} ]. \tag {17}
$$

The clean branch retains the original patch-level visual tokens $\mathbf { Z } ^ { ( K ) }$ and therefore preserves access to fine-grained local visual evidence. In contrast, the degraded branch removes $\mathbf { Z } ^ { ( K ) }$ ) and relies only on register tokens $\mathbf { R } ^ { ( K ) }$ as the visual condition. Since register tokens contain global image semantics but lack stable local correspondence, the degraded branch remains image-aware while being locally under-grounded.

The two branches are then forwarded through the remaining decoder layers:

$$
\mathbf {H} _ {c} ^ {(D)} = F _ {K + 1: D} \left(\mathbf {H} _ {c} ^ {(K)}\right), \tag {18}
$$

$$
\mathbf {H} _ {d} ^ {(D)} = F _ {K + 1: D} \Big (\mathbf {H} _ {d} ^ {(K)} \Big).
$$

Let $\mathbf { h } _ { c , i } ^ { ( D ) }$ and $\mathbf { h } _ { d , i } ^ { ( D ) }$ denote the hidden states used for next-token prediction at decoding step i in the clean and degraded branches. The corresponding vocabulary-level logits are

$$
\boldsymbol {\ell} _ {i} ^ {c} = g _ {\mathrm{LM}} (\mathbf {h} _ {c, i} ^ {(D)}), \quad \boldsymbol {\ell} _ {i} ^ {d} = g _ {\mathrm{LM}} (\mathbf {h} _ {d, i} ^ {(D)}), \tag {19}
$$

where gLM(·) is the language modeling head.

Register-conditioned contrastive decoding. YARD combines the clean and degraded logits using the standard contrastive decoding form:

$$
\boldsymbol {\ell} _ {i} ^ {\mathrm{yard}} = (1 + \alpha) \boldsymbol {\ell} _ {i} ^ {c} - \alpha \boldsymbol {\ell} _ {i} ^ {d}, \tag {20}
$$

where α controls the strength of contrastive subtraction. The next-token distribution is then computed as

$$
p _ {\text { yard }} (y _ {i} \mid I, x, y _ {<   i}) = \text { softmax } \left(\ell_ {i} ^ {\text { yard }}\right) _ {y _ {i}}. \tag {21}
$$

Following common contrastive decoding practice, we apply a clean-branch plausibility constraint to avoid promoting tokens that are unlikely under the clean branch. Let

$$
p _ {c} (y) = \mathrm{softmax} (\boldsymbol {\ell} _ {i} ^ {c}) _ {y} \tag {22}
$$

be the clean-branch probability for token y. We define the candidate set

$$
\mathcal {V} _ {i} = \left\{y \in \mathcal {V} \mid p _ {c} (y) \geq \tau \max _ {y ^ {\prime} \in \mathcal {V}} p _ {c} (y ^ {\prime}) \right\}, \tag {23}
$$

where τ is a plausibility threshold and V is the vocabulary. The final distribution is restricted to Vi:

$$
p _ {\text { yard }} (y _ {i}) = \frac {\exp (\ell_ {i , y _ {i}} ^ {\text { yard }}) \mathbb {I} [ y _ {i} \in \mathcal {V} _ {i} ]}{\sum_ {y \in \mathcal {V} _ {i}} \exp (\ell_ {i , y} ^ {\text { yard }})}. \tag {24}
$$

Dual degradation effect. YARD creates a dual degradation mechanism. The first degradation comes from the branching position. Since the degraded branch splits at layer K, it only inherits early-to-middle multimodal representations:

$$
\mathbf {H} ^ {(K)} = [ \mathbf {Z} ^ {(K)}; \mathbf {R} ^ {(K)}; \mathbf {T} ^ {(K)} ], \tag {25}
$$

where text-to-vision grounding has begun to emerge but fine-grained local evidence has not yet been fully integrated into text-side representations. The second degradation comes from the visual condition after branching:

$$
[ \mathbf {Z} ^ {(K)}; \mathbf {R} ^ {(K)} ] \quad \longrightarrow \quad [ \mathbf {R} ^ {(K)} ]. \tag {26}
$$

This removes patch-level local evidence while retaining register-level global semantics. As a result, the degraded logits tend to emphasize predictions that are plausible under coarse image semantics but insufficiently supported by local visual grounding. Contrastive subtraction then suppresses such predictions:

$$
\ell_ {i, y} ^ {\text { yard }} = \ell_ {i, y} ^ {c} + \alpha \left(\ell_ {i, y} ^ {c} - \ell_ {i, y} ^ {d}\right). \tag {27}
$$

If a token y is overly favored by the locally undergrounded degraded branch, i.e.,

$$
\ell_ {i, y} ^ {d} \gg \ell_ {i, y} ^ {c}, \tag {28}
$$

its final logit is reduced. This is the mechanism by which YARD suppresses language-plausible but visually unsupported hallucination-prone tokens.

Computational advantage. Compared with input-level contrastive decoding, which requires two largely independent forward passes,

$$
\mathrm{Cost} _ {\mathrm{input-CD}} \approx 2 \cdot \mathrm{Cost} (F _ {1: D}), \tag {29}
$$

YARD shares the prefix computation and only duplicates the remaining layers:

$$
\begin{array}{l} \mathrm{Cost} _ {\mathrm{YARD}} \approx \mathrm{Cost} (F _ {1: K}) \\ + \operatorname{Cost} (F _ {K + 1: D}; | \mathbf {Z} | + | \mathbf {R} | + | \mathbf {T} |) \\ + \operatorname{Cost} (F _ {K + 1: D}; | \mathbf {R} | + | \mathbf {T} |). \tag {30} \\ \end{array}
$$

Since the degraded branch removes the N patchlevel visual tokens after layer K, its remaining sequence length is shorter:

$$
| \mathbf {R} | + | \mathbf {T} | \ll | \mathbf {Z} | + | \mathbf {R} | + | \mathbf {T} |. \tag {31}
$$

Thus, YARD improves efficiency by both sharing shallow-layer computation and reducing the sequence length of the degraded branch.

Contrastive decoding. Contrastive decoding is an inference-time strategy for mitigating hallucinations by contrasting the model prediction under a clean visual condition with that under a degraded condition. Given an image v, a text query t, and previously generated tokens $y _ { < i }$ at decoding step i, an LVLM produces vocabulary-level logits conditioned on the original multimodal input:

$$
\ell_ {i} ^ {c} = f _ {\theta} (v, t, y _ {<   i}) \in \mathbb {R} ^ {| \mathcal {V} |}, \tag {32}
$$

where V denotes the vocabulary and the superscript c indicates the clean branch. The corresponding clean next-token distribution is

$$
p _ {i} ^ {c} (y) = \frac {\exp (\ell_ {i , y} ^ {c})}{\sum_ {y ^ {\prime} \in \mathcal {V}} \exp (\ell_ {i , y ^ {\prime}} ^ {c})}. \tag {33}
$$

To obtain a hallucination-prone reference, contrastive decoding further constructs a degraded condition $v _ { d }$ and computes the degraded logits:

$$
\ell_ {i} ^ {d} = f _ {\theta} (v _ {d}, t, y _ {<   i}) \in \mathbb {R} ^ {| \mathcal {V} |}, \tag {34}
$$

with the corresponding degraded distribution

$$
p _ {i} ^ {d} (y) = \frac {\exp (\boldsymbol {\ell} _ {i , y} ^ {d})}{\sum_ {y ^ {\prime} \in \mathcal {V}} \exp (\boldsymbol {\ell} _ {i , y ^ {\prime}} ^ {d})}. \tag {35}
$$

The degraded branch is expected to assign higher probability to tokens that are plausible under weakened visual grounding, and therefore serves as a reference for identifying hallucination-prone predictions.

The contrastive logits are then computed by subtracting the degraded logits from the clean logits:

$$
\boldsymbol {\ell} _ {i} ^ {c d} = (1 + \alpha) \boldsymbol {\ell} _ {i} ^ {c} - \alpha \boldsymbol {\ell} _ {i} ^ {d}, \tag {36}
$$

where $\alpha \geq 0$ controls the contrastive strength. Equivalently, for each candidate token $y \in \mathcal { V }$ , the contrastive score is

$$
s _ {i} (y) = (1 + \alpha) \ell_ {i, y} ^ {c} - \alpha \ell_ {i, y} ^ {d}. \tag {37}
$$

The next-token distribution is obtained by applying softmax to the contrastive logits:

$$
p _ {c d} \left(y _ {i} = y \mid v, v _ {d}, t, y _ {<   i}\right) = \frac {\exp \left(s _ {i} (y)\right)}{\sum_ {y ^ {\prime} \in \mathcal {V}} \exp \left(s _ {i} \left(y ^ {\prime}\right)\right)}. \tag {38}
$$

The next token is then selected as

$$
y _ {i} = \arg \max _ {y \in \mathcal {V}} p _ {c d} (y \mid v, v _ {d}, t, y _ {<   i}), \tag {39}
$$

or sampled from $p _ { c d }$ depending on the decoding strategy.

In practice, contrastive decoding is often combined with a clean-branch plausibility constraint to avoid over-promoting tokens that receive very low probability under the original visual input. Let $\beta \in [ 0 , 1 ]$ denote a plausibility threshold. The candidate set can be defined as

$$
\mathcal {V} _ {i} ^ {\text {head}} = \left\{y \in \mathcal {V} \mid p _ {i} ^ {c} (y) \geq \beta \max _ {y ^ {\prime} \in \mathcal {V}} p _ {i} ^ {c} (y ^ {\prime}) \right\}. \tag {40}
$$

The final decoding decision is then restricted to this candidate set:

$$
y _ {i} = \arg \max _ {y \in \mathcal {V} _ {i} ^ {\text { head }}} s _ {i} (y). \tag {41}
$$

This constraint ensures that the contrastive objective mainly suppresses tokens over-favored by the degraded branch, rather than introducing unlikely tokens that are not supported by the clean branch.

Intuitively, the clean logits contain both visually grounded predictions and hallucination-related components, while the degraded logits are intended to emphasize predictions that remain plausible when visual grounding is weakened. By subtracting the degraded logits, contrastive decoding suppresses tokens that rely more on priors or insufficient grounding than on reliable visual evidence. Conventional input-level methods construct $v _ { d }$ by perturbing the image or instruction, e.g., through image noise, masking, or instruction perturbation. Such methods typically require an additional degraded forward pass:

$$
\ell_ {i} ^ {c} = f _ {\theta} (v, t, y _ {<   i}), \quad \ell_ {i} ^ {d} = f _ {\theta} (v _ {d}, t, y _ {<   i}), \tag {42}
$$

which doubles a large portion of the inference computation. In contrast, our method constructs the degraded branch inside the decoder, allowing the clean and degraded branches to share early-layer computation while still producing contrastive logits for hallucination mitigation.

# A.8 Top-Token Case Study of Degraded Branches

To provide a more intuitive view of how different degraded branches behave during decoding, we present a next-token case study in Figure 6. This example compares the clean branch with three degraded branches at the same decoding step, where the baseline response eventually hallucinates the object chair. The prefix before prediction is “The image depicts a well-lit workstation with a black”, and we inspect the top-5 next-token predictions from each branch.

The pixel-level degraded branch produces top predictions that are highly similar to the clean

Input image   
![](images/78c6394ce7b0a14364719d0f201aed32bdbbb02631eda121d4a4cb5949399f37.jpg)

<details>
<summary>natural_image</summary>

Close-up of Sony headphones resting on a keyboard, no visible text or symbols on the devices or background
</details>

Hallucinated object: chair   
Prefix: “The image depicts a well-lit   
workstation with a black”   
Next baseline token: office

Top-5 next-token predictions 

<table><tr><td>Branch</td><td>Top-5 next tokens</td></tr><tr><td>Clean</td><td>Sony, computer, and, keyboard, pair</td></tr><tr><td>Pixel-level deg.</td><td>Sony, and, computer, pair, head</td></tr><tr><td>Ours-register deg.</td><td>computer, and, des, chair, keyboard</td></tr><tr><td>Text-only deg.</td><td>des, and, computer, background, chair</td></tr></table>

Distributional gap to the clean branch 

<table><tr><td>Branch vs. clean</td><td>Top-10 overlap</td><td>SymKL</td></tr><tr><td>Pixel-level deg.</td><td>0.80</td><td>0.22</td></tr><tr><td>Ours-register deg.</td><td>0.60</td><td>7.80</td></tr><tr><td>Text-only deg.</td><td>0.40</td><td>9.20</td></tr></table>

Pixel-level degradation largely copies the clean distribution. Text-only degradation drifts toward generic language-prior tokens such as background. In contrast, the register-based branch remains scene-aware by retaining tokens such as computer and keyboard, while exposing the hallucination-prone token chair.

Figure 6: Next-token case study comparing clean and degraded branches. Pixel-level degradation stays close to the clean visual-context distribution, while text-only degradation moves toward image-agnostic language priors. The register-based degraded branch remains related to the visual scene while surfacing the hallucination-prone token chair, illustrating its image-aware but locally under-grounded behavior.

branch. For example, both branches assign high ranks to scene-relevant tokens such as Sony, computer, and keyboard, and their Top-10 overlap remains high. This suggests that pixel-level degradation does not sufficiently separate the degraded branch from the clean visual-context distribution. As a result, its degraded logits may still contain substantial correct visual evidence, making contrastive subtraction less precise.

The text-only degraded branch shows the opposite behavior. It moves farther away from the clean branch and surfaces the hallucination-prone token chair, but it also drifts toward generic languageprior completions such as background. Since this branch is no longer conditioned on the image, its top predictions are less aligned with the current visual scene. Thus, although text-only degradation can expose hallucination-prone tokens, the resulting contrastive signal is overly image-agnostic.

In contrast, the register-based degraded branch provides a more balanced behavior. It still preserves image-related tokens such as computer and keyboard, showing that it remains semantically connected to the visual scene. Meanwhile, it also exposes the hallucination-prone token chair, indicating that local visual grounding has been weakened. This supports our motivation for feature-level degradation: the degraded branch should be imageaware but locally under-grounded, rather than either copying the clean distribution or collapsing into a text-only prior.

Potential Risks. YARD is designed to mitigate hallucinations in LVLMs, but it does not eliminate all visually unsupported generations. In safetycritical applications, such as medical, legal, or autonomous decision-making scenarios, its outputs should still be verified by humans or external tools. In addition, because YARD modifies decoding behavior without retraining, its effectiveness may vary across model architectures, prompts, and visual domains. A potential risk is over-reliance on reduced hallucination rates as a guarantee of factual correctness. We therefore recommend using YARD as a complementary inference-time mitigation method rather than as a standalone safety mechanism.

# B Implementation Details of Training-Free Register Construction

Register construction on LLaVA-v1.5. In our main LLaVA-v1.5 setting, we do not train new register tokens. Instead, we construct them on the fly inside the CLIP ViT visual tower using forward hooks. During the register discovery stage, we use register\_discovery\_max\_images=128 calibration images. For each image, we append num\_register\_tokens=M zero-valued register slots to the end of the original visual token sequence, and register forward hooks on the mlp.fc1 module of each ViT layer to record the intermediate activations.

To identify register-related dimensions, we score neurons by averaging the absolute fc1 activations at anomalous token positions across layers and calibration images. We then select the top register\_topk\_neurons=10 globally activated neurons and store them in a neuron cache. This cache is computed once and reused during inference.

At inference time, we load the cached neuron indices and again append zero register slots to the visual token sequence. In the corresponding fc1 hooks, we apply sink shift with register\_intervention\_scale=1.0: for the selected neuron dimensions, the strongest sink activations are redirected to the appended register slots, while ordinary patch tokens keep their original values under register\_normal\_values=same. The resulting register tokens are then passed through the multimodal projector to align them with the LLM hidden dimension.

For contrastive decoding, we use merge\_pool\_mode=register to construct the degraded visual condition from the projected register representations. The clean branch retains the full set of original visual tokens. The two branches share the shallow LLM decoder computation and split at split\_layer=10; after branching, the degraded branch is forwarded for num\_degraded\_layers=22 layers. This implementation constructs the degraded branch without retraining and avoids a complete second forward pass.

Input image   
![](images/b15dff3c73878077e3a6735c83445398e45a7100e5630e9d6b999f5a0b933b4d.jpg)

<details>
<summary>natural_image</summary>

Snowboarder in action against a green forest wall (no text or symbols visible)
</details>

# Baseline

The image features a skateboard next to a tree, with some leaves covering a portion of the snowboard. The skateboard is mostly located on top of a fence, and the tree branches reach towards it. The fence beneath the skateboard also has a camera placed at the bottom of the image.

# YARD

The image features a snowboard leaning against a fence with green leaves in the background. The snowboard is quite large and is resting on a wooden fence. It has a blue design and is placed on its side, making it stand out against the green foliage. The fence is visible in the background, partially covered by leaves, while the snowboard remains the main object in the scene.

Figure 7: Qualitative comparison between the baseline and YARD. The baseline repeatedly hallucinates skateboard and introduces an unsupported camera, while YARD correctly describes the visible snowboard and avoids unsupported objects. Hallucinated words are highlighted in red.   
Input image   
![](images/802acec2bfd03280590494fda797fc8fc28e3def809ddc81d6da2c54ecf4d4bd.jpg)

<details>
<summary>natural_image</summary>

Pastoral scene with sheep grazing on a green hillside, bordered by leafless trees and a fence (no text or symbols visible)
</details>

# Baseline

The scene depicts a large, peaceful, grassy field where a mixed herd of sheep and baby lambs is grazing. There are people observing the flock from a nearby bush or landscaped position. The sheep are scattered throughout the field, with some lambs huddled together while others graze farther across the hillside.

# YARD

The image depicts a peaceful scene of several sheep grazing on a lush green hillside. There are ten sheep in total, with some standing and others lying down on the grassy field. The sheep are scattered throughout the field, some closer to the foreground, while others are further in the background. The overall scene is calm and natural, with the green hillside providing a simple backdrop for the flock.

Figure 8: Qualitative comparison between the baseline and YARD. The baseline hallucinates unsupported people, while YARD focuses on visually grounded content and correctly describes the flock of sheep on the grassy hillside. Hallucinated words are highlighted in red.   
Input image   
![](images/0988c58bcec622f4fc91e2d7c794bb88688ec25616c74437584add2228902851.jpg)

<details>
<summary>natural_image</summary>

Windsur performing on water with wind turbines in background (no text or symbols visible)
</details>

# Baseline

The image features a man skillfully riding a board in ocean waves while holding onto a sail, demonstrating his expertise in kite surfing. He is wearing a wetsuit and is completely engaged in the thrilling watersport. The kite is visible in the wind’s motion, providing power for the rider as he moves quickly across the water.

# YARD

The image captures a man skillfully windsurfing on a body of water. He is wearing a black wet suit and riding a red and black wind surfboard. The windsurfer is leaning back as he enjoys the thrill of the sport, with the sail attached to the board clearly visible and helping propel him across the water.

Figure 9: Qualitative comparison between the baseline and YARD. The baseline hallucinates an unsupported kite, while YARD correctly focuses on the visually grounded windsurfing scene, including the person, the sail, and the surfboard. Hallucinated words are highlighted in red.

Input image   
![](images/e2a8b39ab2eadcb56ba1ad02cc47d075dfbb631ea0b4585f1093d2f8ffac4d79.jpg)

<details>
<summary>natural_image</summary>

Blue train traveling on railway tracks with green hills and ocean in background (no visible text or symbols)
</details>

# Baseline

The image features a bright blue passenger train traveling near a forest. It occupies almost the entire visible width of the scene, running parallel to the tracks near the road sign. There are six visible cars on the train, each with a clear blue color and easy to distinguish by their alignment. The train is moving steadily along the railway, with the surrounding greenery providing a natural backdrop.

# YARD

The image features a long blue passenger train traveling on the tracks. The train is stretched out across the scene, covering a significant portion of the image. The tracks appear to be made of metal, and the train is moving along them in a straight line. In addition to the train, there is surrounding greenery and a signpost visible near the railway, completing the outdoor scene.

Figure 10: Qualitative comparison between the baseline and YARD. The baseline hallucinates unsupported cars as separate objects, while YARD consistently focuses on the visually grounded train and its surrounding scene. Hallucinated words are highlighted in red.   
Input image   
![](images/a9f1c6025dac4ef6e8c0ad4c082c91f2fa2ab5e19ee6303e3c5cc0ca770ec87d.jpg)

<details>
<summary>natural_image</summary>

Blue bicycle with two dogs parked beside a metal rack, no visible text or symbols
</details>

# Baseline

The scene displays a small dog sitting inside a wire basket placed on a transport motorcycle. The dog appears to be comfortable in its enclosed space, offering a sense of security. Behind the motorcycle, there is a trash can with a cardboard box on it. Near the edge of the image, additional street-side objects can also be seen.

# YARD

The scene features a dog sitting inside a wire basket attached to the front of a blue bicycle. The dog appears to be a black and white terrier, comfortably occupying the basket. The bicycle is parked next to a garbage can, which is located on the right side of the image. A bench and a car can also be seen in the surrounding street scene.

Figure 11: Qualitative comparison between the baseline and YARD. The baseline hallucinates an unsupported motorcycle, while YARD correctly identifies the bicycle and preserves other visually grounded content such as the dog, the garbage can, and the surrounding street scene. Hallucinated words are highlighted in red.   
Input image   
![](images/d34930211d2a2f3251b51d783cf99c89b01820cdf3c985192c77700edfecba05.jpg)

<details>
<summary>natural_image</summary>

Interior view of a bathroom with tiled walls, white towels, and a window (no text or symbols visible)
</details>

# Baseline

The image shows a clean bathroom that features a white toilet under a towel rail. Toilet paper can be seen on the wall, and there is a toothbrush holder with toothbrushes. Above the toilet, a pull-down window blind helps control the light entering the room, while other bathroom items are arranged around the sink area.

# YARD

The image shows a bathroom with white towels hanging from a towel rack on the wall. The towels are folded neatly, and there are four of them, with two on each side. The toilet is located in the middle of the bathroom, with a window nearby and a sink area visible in the surrounding space. The scene appears clean and orderly, with simple bathroom fixtures arranged along the wall.

Figure 12: Qualitative comparison between the baseline and YARD. The baseline hallucinates unsupported toothbrushes, while YARD focuses on visually grounded bathroom elements such as the toilet, towels, window, and sink area. Hallucinated words are highlighted in red.