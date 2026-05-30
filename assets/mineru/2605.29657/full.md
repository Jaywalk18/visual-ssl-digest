# OccamToken: Efficient VLM Inference with Training-Free and Budget-Adaptive Token Pruning

Geng Li1,∗, Guohao Chen1,∗, Ting Chen1,∗, Shilin Shan1, Kuangji Zuo1, Bofan Lyu1, Tuo An1, Gen Li1,†, Jianfei Yang1,†

1Nanyang Technological University (NTU) ∗Equal contribution., †Corresponding authors.

Vision-language models (VLMs) rely on long visual token sequences for visual understanding, making the prefill stage expensive in both computation and memory. Most existing pruning methods follow an absolute-ranking paradigm, assigning importance scores to visual tokens and retaining a fixed top-K subset. In this work, we argue that this paradigm is fundamentally brittle: attention sinks distort token importance rankings, while image redundancy and query-dependent visual evidence make fixed token budgets unreliable across inputs. We propose OccamToken, a training-free framework that replaces absolute token ranking with register-anchored relative evidence testing. Instead of asking which tokens are globally important, OccamToken evaluates whether a visual token provides information beyond a register-based reference. Our key insight is that register tokens naturally absorb low-information attention patterns, making them a stable reference for identifying genuinely informative visual evidence. Based on this principle, OccamToken performs both image-adaptive redundancy pruning and query-adaptive relevance pruning through dynamic thresholds derived from register attention. Across LLaVA-NeXT, LLaVA-v1.5, and Qwen3-VL, OccamToken consistently improves the accuracy–efficiency trade-off without additional training. Notably, on LLaVA-NeXT, it reduces 2,880 visual tokens to approximately 40 while preserving over 93% of the original accuracy, enabling stable visual token compression even in the extreme 1.4% retention regime.

![](images/dce41184734590e2bfdf8fa12253e0cff499152a17280b653d66baf5b09042dc.jpg)

MARS Lab

# 1 Introduction

Vision-language models (VLMs) (Liu et al., 2024a; Bai et al., 2023; Li et al., 2025; Liu et al., 2024b) have achieved remarkable progress in multimodal understanding and reasoning, as evidenced by recent generalpurpose VLMs and multimodal evaluation benchmarks (Liu et al., 2024c; Yue et al., 2024; Li et al., 2023b). A standard VLM connects a vision encoder with a large language model by encoding images into patch-level visual tokens, projecting them into the language embedding space, and prepending them to textual instructions (Liu et al., 2023; Dai et al., 2023). As high-resolution, multi-image, and video-capable VLMs become increasingly common, each input may contain hundreds or even thousands of visual tokens (Liu et al., 2024b; Bai et al., 2025). These tokens are processed during the prefill stage and stored in the key-value cache for autoregressive decoding, making visual sequence length a major source of computation and memory overhead (Vaswani et al., 2017; Dao et al., 2022; Chen et al., 2024). This motivates visual token compression: reducing redundant visual tokens while preserving the evidence required for multimodal reasoning (Shang et al., 2025; Zhang et al., 2025c; Yang et al., 2025; Zhang et al., 2025b).

Existing compression methods typically follow a score-and-select paradigm: they assign importance or redundancy scores to visual tokens and retain a compact subset, often under a fixed top-K (Chen et al., 2024) budget. The scoring signals can be broadly categorized into attention-based cues, such as [CLS]-to-patch or [text]-to-vision attention (Chen et al., 2024; Zhang et al., 2025c,a; Yang et al., 2025), and feature-level cues, such as token similarity, diversity, duplication, or set coverage (Alvar et al., 2025; Wen et al., 2025; Deng et al., 2025). Among these directions, attention-based scoring remains a dominant and widely adopted choice because it directly reuses the interaction patterns already computed by the vision encoder or the language model. However, we observe that this attention-based pruning paradigm suffers from two structural

![](images/704d485f482291da4be0a4171ca6d480bfc2e1079a50590b3b9ad046028f6527.jpg)

<details>
<summary>line</summary>

| Attention score x | Before register | After register |
| ----------------- | --------------- | -------------- |
| 10^-4             | 1.0             | 1.0            |
| 10^-3             | 0.671           | 0.671          |
| 10^-2             | 0.0             | 0.0            |
</details>

(a) Attention sink absorption

![](images/2ccabbe85536b2a27a8e8dcea7a3a84ddda98936edd57d6e53e72f7d78963781.jpg)

<details>
<summary>line</summary>

| Rank | Sample A | Sample B | Sample C |
|------|----------|----------|----------|
| 0    | 1.0      | 1.0      | 1.0      |
| 20   | 0.01     | 0.01     | 0.01     |
| 40   | 0.005    | 0.005    | 0.005    |
| 60   | 0.002    | 0.002    | 0.002    |
| 80   | 0.001    | 0.001    | 0.001    |
| 100  | 0.0005   | 0.0005   | 0.0005   |
| 120  | 0.0002   | 0.0002   | 0.0002   |
| 140  | 0.0001   | 0.0001   | 0.0001   |
| 160  | 0.0001   | 0.0001   | 0.0001   |
</details>

(b) Static vs. relative cutoff

![](images/bedef6d393d8603dea15ef8e05effa67a44e9a2f573c8a0e91daeae067b8de22.jpg)

<details>
<summary>line</summary>

| Sample Index | OccamToken (λ₁) | OccamToken (λ₂) |
| ------------ | --------------- | --------------- |
| 0            | ~750            | ~150            |
| 500          | ~750            | ~150            |
| 1000         | ~750            | ~150            |
| 1500         | ~750            | ~150            |
| 2000         | ~750            | ~150            |
| 2500         | ~750            | ~150            |
</details>

(c) Adaptive token budget   
Figure 1 Motivation for relative comparison. (a) Register insertion suppresses attention sinks, yielding a less sink-dominated CLS→vision attention distribution and increasing $n _ { \mathrm { e f f } }$ from 42 to 281. (b) A fixed top-K cutoff corresponds to different evidence levels across samples, while relative cutoffs adapt to distributional variation. (c) OccamToken produces sample-adaptive token budgets through two-stage register-anchored pruning.

deficiencies.

First, low-information tokens can absorb excessive score mass and make visual-token importance less distinguishable. For visual token pruning, the score distribution should ideally separate informative visual evidence from redundant or background regions. However, a few visually uninformative tokens can receive unusually large attention scores, causing the remaining visual tokens to receive smaller and less distinguishable scores. Prior work (Barbero et al., 2025; Yu et al., 2024) describes this phenomenon as attention sinks, i.e., high-norm outlier tokens attract excessive attention. Consequently, truly informative tokens are squeezed into a narrower score range, reducing score gaps and obscuring the boundary between informative and redundant tokens. Second, the static top-K cutoff cannot adapt to the per-sample variation in importance-score distributions. In practice, attention-score distributions vary substantially across images and queries. A fixed K therefore cannot adapt its retention budget to each sample, often preserving redundant tokens in simple cases while discarding critical evidence in complex ones. Existing adaptive methods either rely on extra training (Ye et al., 2025b; Huang et al., 2025) or use training-free image-level statistics (Fang et al., 2026; Shang et al., 2025), which remain weakly responsive to query-conditioned evidence demands.

We find that these two deficiencies can be jointly addressed by using register tokens as reference anchors. Register tokens, introduced in recent transformer studies (Darcet et al., 2024; Jiang et al., 2025), can aggregate sink-like attention scattered across low-information image regions. Meanwhile, prior work shows that register tokens encode image-level global information (Jiang et al., 2025), and our analysis further suggests that they are less sensitive to fine-grained local patch evidence; see Section 3.3. Thus, a register token is not merely a sink absorber, but also provides a global yet non-discriminative summary of the visual input.

This motivates a reference-adaptive pruning paradigm: instead of asking whether a visual token ranks within a fixed top-K set, we ask whether it provides stronger evidence than the register token under the current attention distribution. Because the register token is scored in the same attention competition as ordinary visual tokens, it serves as a semantic reference that both reduces sink-induced score distortion and defines a sample-adaptive pruning boundary. Building on this principle, we propose OccamToken, a training-free two-stage framework for reference-adaptive visual token pruning during VLM inference. Stage I operates at the vision encoder output and removes image-level redundancy with [CLS]-to-register attention. Stage II operates inside the language model and further prunes query-irrelevant visual tokens with [text]-to-register attention. Our main contributions:

• We propose register-anchored dynamic thresholding, a training-free mechanism that links attention-sink mitigation to adaptive visual token pruning. By inserting a test-time register token into the same attention competition, the mechanism absorbs sink-like attention from low-information tokens and uses the register score as a distribution-adaptive pruning threshold.   
• We instantiate this mechanism as OccamToken, a training-free two-stage pruning framework for efficient VLM inference. It first removes image-level redundancy at the vision-encoder output with [CLS]-to-register attention, and then prunes query-irrelevant tokens inside the language model with [text]-to-register attention, yielding image- and query-adaptive token budgets.

• Extensive experiments on image-understanding benchmarks across LLaVA-v1.5 (Liu et al., 2024a), LLaVA-NeXT (Liu et al., 2024b), and Qwen3-VL (Bai et al., 2025) show that OccamToken outperforms the compared state-of-the-art visual token pruning baselines, including both training-free and training-based methods, under matched token budgets. Notably, OccamToken pushes training-free visual token compression on LLaVA-NeXT into the 1.4% retention regime while preserving over 93% of the full-token accuracy.

# 2 Related Work

# 2.1 Visual Token Compression in VLMs

The redundancy of visual tokens during VLM inference has motivated various compression strategies, including pruning, merging, and hybrid methods. Existing approaches typically estimate token importance from attention signals Zhang et al. (2025a); Shang et al. (2025); Yang et al. (2025); Chen et al. (2024); Ye et al. (2025a); Zhang et al. (2025c), or from feature-level criteria such as diversity, duplication, and coverage Alvar et al. (2025); Wen et al. (2025); Deng et al. (2025); Dhouib et al. (2025). Most of these methods still rely on fixed retained-token budgets, which cannot fully adapt to sample-dependent redundancy.

Recent adaptive methods attempt to relax this constraint by learning token-budget predictors or deriving budgets from training-free statistics. However, learned approaches require additional parameters, data, or model-specific adaptation Ye et al. (2025b); Huang et al. (2025); Takezoe et al. (2026); Shao et al. (2025), while training-free statistical methods depend on externally defined criteria such as sparsity, similarity, or global redundancy scores Shang et al. (2025); Fang et al. (2026). In contrast, OccamToken introduces a reference-based adaptive mechanism: the register token participates in the same softmax competition as visual tokens, making its score an internal semantic reference for dynamic pruning. This enables training-free, query-adaptive token allocation without learned predictors or externally defined statistical cutoffs.

# 2.2 Attention Sinks and Register Tokens

Attention sinks are a common Transformer phenomenon in which a small number of tokens attract disproportionately large attention weights (Sun et al., 2024; Gu et al., 2025; Cancedda, 2024). Xiao et al. Xiao et al. (2024) studied this behavior in language models, while Darcet et al. Darcet et al. (2024) observed high-norm outlier tokens in low-information regions of vision Transformers and introduced trained register tokens to absorb such artifacts. Jiang et al. Jiang et al. (2025) further showed that similar behavior can be induced at test time by redirecting sink-like activations to an appended register token, avoiding retraining.

Prior work primarily uses register tokens to stabilize Transformer representations or mitigate attention-sink artifacts. OccamToken instead repurposes the register token as a semantic reference anchor for visual token pruning. By using the register score as an adaptive threshold, our method connects attention-sink mitigation with dynamic token budgeting in a training-free and plug-and-play manner.

# 3 Method

# 3.1 Preliminaries

VLM inference pipeline. Given an image x and a text query, a vision encoder (Radford et al., 2021; Zhai et al., 2023) g first maps the image into a sequence of visual tokens $\mathcal { V } = g ( x ) = \dot { \{ v _ { i } \} } _ { i = 1 } ^ { N } \in \mathbb { R } ^ { N \times d _ { v } }$ , together with a [CLS] token $v _ { \mathrm { c l s } } .$ A visual projector (Dai et al., 2023) p then maps these visual representations into the language-model embedding space. Let $\mathcal { T } = \{ t _ { j } \} _ { i = 1 } ^ { M } \in \mathbb { R } ^ { \tilde { M } \times d }$ denote the embedded text tokens. The language model takes the concatenated sequence $[ p ( \gamma ) ; { \check { \tau } } ]$ as input and performs autoregressive decoding. Since the number of visual tokens is often much larger than the number of text tokens, i.e., $N \gg M$ , visual tokens dominate the prefill-stage computation and memory cost, making the compression of V critical for efficient VLM inference.

Attention mechanism. Both the vision encoder and the language model are built on Transformer blocks, whose core operation is multi-head attention. For clarity, we describe a single attention head. Given an input sequence $\mathbf { X } \in \mathbb { R } ^ { L \times d }$ , the attention weights and output are computed as

Original Image   
![](images/c3235e3039df1317fa8efff72205767cdb83a8f26df156b883bb320d5140a70c.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with white cabinetry, oven, and kitchenware (no visible text or symbols)
</details>

Retain tokens: 576

After Stage I   
![](images/9c3daa56310504a23b7625c51c2de36b95806cd50d31ad9d5c6c581626367908.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern living room with furniture and wall-mounted equipment (no visible text or symbols)
</details>

Retain tokens: 256

Q: Does the knife in the ife block have brown colo   
![](images/64855676ee0f15ec6f78244c3429f07045a88c01a8367e2848af4d2ce60bd98f.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with white cabinets, oven, and kitchenware (no visible text or symbols)
</details>

A: No Retain tokens: 9

Q: Are there refrigerators to the left of the stove?   
![](images/3febca7217e92cc957da7a6d56aea9b107599a850658c3df3ccd222840239a2f.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with white cabinets and kitchenware (no visible text or symbols)
</details>

A: Yes Retain tokens: 17   
Figure 2 Visualization of query-adaptive token budgets. For the same image, Stage I performs query-agnostic redundancy pruning, while Stage II selects query-relevant visual evidence. Different questions induce different retained token budgets: the knife-block question keeps 9 tokens, whereas the refrigerator-stove relation question keeps 17 tokens. This demonstrates that OccamToken adapts the final visual budget to query-specific evidence demand rather than enforcing a fixed top-K budget.

$$
\mathbf {A} = \operatorname{Softmax} \left(\frac {\mathbf {Q K} ^ {\top}}{\sqrt {d _ {h}}}\right), \quad \mathbf {O} = \mathbf {A U}, \tag {1}
$$

where $\mathbf { Q } = \mathbf { X } \mathbf { W } _ { q } , \mathbf { K } = \mathbf { X } \mathbf { W } _ { k }$ , and $\mathbf { U } = \mathbf { X } \mathbf { W } _ { u }$ denote the query, key, and value projections, respectively, and $d _ { h }$ is the head dimension. The entry $A _ { i j }$ measures how strongly token i attends to token j. Existing visual token pruning methods often use attention weights from specific evaluator tokens, such as [CLS] or [text] tokens, to visual tokens as importance scores.

Test-time register token. Jiang et al. Jiang et al. (2025) trace attention sinks to sparse register neurons in the MLPs and introduce a test-time register token r to absorb their high-norm activations. During inference, abnormal channel-wise activations from patch tokens are redirected to the corresponding dimensions of r, and r then participates in subsequent self-attention computations. Thus, the register token serves as a sink recipient while being scored in the same softmax distribution as ordinary visual tokens.

# 3.2 From Absolute Ranking to Relative Comparison

The goal of visual token pruning is to select a compact subset $s \subset \nu$ such that the model output conditioned on $s$ closely approximates that conditioned on the full visual token set V. Given an importance scoring function s, a widely adopted pruning rule is to rank visual tokens by their scores and retain the top-K tokens:

$$
\mathcal {S} = \operatorname{top-} K (\mathcal {V}, s). \tag {2}
$$

While simple and effective, this top-K ranking rule exposes two limitations.

Observation 1: Attention sinks reduce the separability of importance scores. Due to the sum-to-one constraint of softmax attention, attention allocation is inherently competitive. Attention sink tokens can absorb a disproportionate fraction of this fixed attention mass, leaving the remaining visual tokens with compressed scores and smaller score gaps. This makes informative tokens less distinguishable from redundant ones. We quantify this effect using the effective number of attended tokens, i.e., $n _ { \mathrm { e f f } } { = } \exp ( H ( \alpha ) )$ , where $H ( \alpha ) = -$ $\textstyle \sum _ { i } \alpha _ { i }$ i log $\alpha _ { i }$ is the entropy of the attention distribution. Here, $n _ { \mathrm { e f f } }$ corresponds to the support size of a uniform distribution with the same entropy. As shown in Figure 1 (a), before introducing a register token to absorb sink-like activations, $n _ { \mathrm { e f f } }$ is only 42. After sink absorption, it increases to 281, indicating a substantially less sink-dominated scoring distribution.

Observation 2: A fixed top-K cutoff is mismatched with varying score distributions. Top-K pruning imposes a rank-based decision rule: the token at the K-th rank defines the pruning boundary. However, even after sink artifacts are mitigated, the attention-score distribution can still vary substantially across images and queries, as shown in Figure 1 (b). As a result, the same value of K may correspond to very different evidence levels across samples. For information-dispersed samples, it may discard critical evidence; for information-concentrated samples, it may retain unnecessary redundancy. We refer to this issue as a cutoff-distribution mismatch: a static rank-based cutoff cannot adapt to sample-dependent score distributions.

![](images/3107a2474b2232d08b1002043e7dea680e9e685af5e724764b78c7b209f60ed1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Adaptive Redundancy Pruning
        V1["V1"] --> V1a["0.81"]
        V2["V2"] --> V2a["0.29"]
        V3["V3"] --> V3a["0.69"]
        V4["V4"] --> V4a["0.24"]
        V5["V5"] --> V5a["0.59"]
        reg["reg"] --> regp["0.47"]
    end

    subgraph Register-Anchored Relevance Pruning
        T1["T1"] --> T2["T2"]
        T2 --> T1p["RegRP"]
        T1p --> T1q["2"]
        T1q --> T1r["1"]
        T1r --> T1s["0.68"]
        T1s --> V1["V1"]
        T1s --> V3["V3"]
        T1s --> V5["V5"]
        reg["reg"] --> regp["0.41"]
    end

    subgraph Visual Feature Extraction
        VisionEncoder["Vision Encoder"] --> AdapRP["AdapRP"]
        AdapRP --> Projector["Projector"]
        Projector --> LLMDecoderLayer["i"]
        LLMDecoderLayer["i"] --> RegRP["RegRP"]
    end

    subgraph Output
        V1a --> V1b["0.81"]
        V2a --> V2b["0.29"]
        V3a --> V3b["0.69"]
        V4a --> V4b["0.24"]
        V5a --> V5b["0.59"]
        reg["reg"] --> regp["0.47"]

    subgraph Output
        T1p --> T1q["0.68"]
        T2p --> T2r["0.30"]
        T1q --> T1s["0.76"]
        T2p --> T2r
        reg["reg"] --> regp["0.41"]
    end
```
</details>

Figure 3 Overview of OccamToken. Given an input image and a text query, our framework performs two-stage adaptive visual token pruning. Stage I AdapRP (Adaptive Redundancy Pruning): At the vision encoder output, a test-time register token absorbs attention sinks. The [CLS] token scores all visual tokens and the register token jointly; tokens scoring below $\lambda _ { 1 } \cdot s _ { 1 } ( r )$ are pruned, yielding an image-adaptive token budget. Stage II RegRP (Register-Anchored Relevance Pruning): Within the LLM, text tokens score the surviving visual tokens via max attention, while the register token’s mean text attention serves as a dynamic threshold. Tokens scoring below $\lambda _ { 2 } \cdot s _ { 2 } ( r )$ are removed, producing a query-adaptive final budget.

These observations motivate moving beyond absolute ranking. Instead of asking whether a token ranks among the top K, pruning should ask whether its score exceeds a meaningful reference level under the current score distribution. We therefore propose a relative comparison paradigm, which replaces the static cutoff with an adaptive reference score produced within the attention competition:

$$
\mathcal {S} (x) = \left\{v \in \mathcal {V} (x) \mid s _ {x} (v) \geq \lambda \cdot s _ {x} (r (x)) \right\}, \tag {3}
$$

where x denotes the input sample, $r ( x )$ is a sample-dependent reference token, and λ controls the pruning intensity. Under this criterion, a visual token is retained not because of its global rank, but because it provides stronger evidence than the reference token. The number of retained tokens $\vert S ( x ) \vert$ therefore varies with the score distribution of each sample.

# 3.3 Register Token as a Dynamic Reference Anchor

Eq. 3 suggests that adaptive pruning requires a reference token whose score can serve as a sample-dependent threshold. However, selecting such a reference is non-trivial. An ideal reference should satisfy three properties: (i) it should be introduced without additional training; (ii) it should participate in the same softmax attention computation as visual tokens, so that its score is directly comparable to visual-token scores; and (iii) it should represent non-discriminative visual information, providing a meaningful cutoff between informative and uninformative tokens.

The first two properties are in fact implicit in Observation 1. The register token redirects high-norm sink activations at inference time, increasing $n _ { \mathrm { e f f } }$ from 42 to 281 without model retraining. Moreover, once included in the token sequence, it participates in the same softmax attention computation as ordinary visual tokens, yielding a sample-specific score directly comparable to visual-token scores.

Can the register token provide a meaningful non-discriminative reference?

We answer this question by examining what information the register token encodes. First, prior work shows that register tokens capture image-level global information. Jiang et al. Jiang et al. (2025) conduct linear probing experiments and find that the register token achieves classification accuracy comparable to the [CLS] token, suggesting that it encodes global visual semantics.

Second, we observe that the register token contains little fine-grained local patch evidence. To verify this, we rank image patches by their [CLS]-to-patch attention scores on LLaVA-v1.5 (Liu et al., 2024a), then mask either the top 50% or bottom 50% of patches. We compute the cosine similarity of the register representation before and after masking, obtaining 0.981 and 0.967, respectively. The register representation remains largely stable regardless of which half of the patches is removed, indicating that it is insensitive to specific local patch evidence.

Together, these findings suggest that the register token encodes a global summary rather than localized discriminative content. This gives it a useful semantic role: it acts as a global but non-discriminative reference anchor. Comparing a visual token with the register token therefore asks whether the local patch provides stronger evidence than this global reference. Accordingly, Eq. 3 retains tokens whose scores exceed the register score, enabling sample-adaptive pruning without a fixed top-K cutoff.

# 3.4 Stage I: Image-Adaptive Redundancy Pruning

Following the relative comparison paradigm introduced in Section 3.3, we evaluate each visual token relative to the register token rather than ranking tokens by an absolute top-K cutoff. We instantiate this principle in two stages. Stage I operates at the output of the vision encoder, where the [CLS] token serves as the evaluator to remove redundant visual tokens in an image-adaptive manner. Stage II operates inside the language model, where text tokens serve as query-conditioned evaluators to further prune visual tokens that are irrelevant to the current query.

[CLS] scoring and dynamic pruning. After constructing the register token as described in Section 3.3, we use the [CLS] token to jointly score all visual tokens and the register token:

$$
s _ {1} (v _ {i}) = A _ {\mathrm{cls} \rightarrow v _ {i}}, \quad s _ {1} (r) = A _ {\mathrm{cls} \rightarrow r}, \quad \forall v _ {i} \in \mathcal {V}. \tag {4}
$$

The register score then serves as an image-adaptive reference threshold. Specifically, the retained token set after Stage I is defined as

$$
\mathcal {S} _ {1} = \left\{v _ {i} \in \mathcal {V} \mid s _ {1} (v _ {i}) \geq \lambda_ {1} \cdot s _ {1} (r) \right\} \cup \{r \}, \tag {5}
$$

where $\lambda _ { 1 }$ controls the pruning intensity, and the register token is always retained for Stage II. Since the threshold is derived from the current image’s attention distribution, the number of retained tokens $| S _ { 1 } |$ varies with image content, enabling image-adaptive budget allocation.

Generalization to diverse architectures. Some VLMs (Bai et al., 2025, 2023) do not provide a [CLS] token in the vision encoder. To support such architectures, we introduce a mutual-scoring scheme that uses visual tokens themselves as evaluators. All visual tokens and the register token attend to each other, and the importance of each candidate token is measured by the average attention it receives:

$$
s _ {1} ^ {\text { mutual }} (v _ {i}) = \frac {1}{| \mathcal {V} | + 1} \sum_ {u \in \mathcal {V} \cup \{r \}} A _ {u \rightarrow v _ {i}}, \tag {6}
$$

$$
s _ {1} ^ {\text { mutual }} (r) = \frac {1}{| \mathcal {V} | + 1} \sum_ {u \in \mathcal {V} \cup \{r \}} A _ {u \rightarrow r}.
$$

The Stage I pruning rule is then applied analogously, using $\lambda _ { 1 } \cdot s _ { 1 } ^ { \mathrm { m u t u a l } } ( r )$ as the dynamic threshold. We validate this scheme on Qwen (Wang et al., 2024; Bai et al., 2025), showing that register-anchored thresholding remains effective even for vision encoder architectures without a [CLS] token; see Section 4.

# 3.5 Stage II: Register-Anchored Relevance Pruning

After Stage I, the retained set $S _ { 1 }$ has been filtered for image-level redundancy, but not all surviving tokens are necessarily relevant to the current query. Stage II therefore operates inside the language model, where cross-modal attention is used to further remove query-irrelevant visual tokens.

[Text]-to-vision scoring. The retained set $S _ { 1 }$ is concatenated with the text tokens $\tau$ and fed into the language model. At the l-th decoder layer, we extract the attention scores from text tokens to tokens in $S _ { 1 }$ . For the register token, which provides a global but non-discriminative reference, we use the mean text-to-register attention to obtain a stable reference score. For each visual token, we use the maximum attention it receives from text tokens, since a visual token may be relevant to only a few query words and averaging could dilute such localized correlations:

Algorithm 1 OccamToken: Two-Stage Adaptive Visual Token Pruning 

<table><tr><td colspan="2">Require: Image I, text query T, coefficients  $\lambda_1, \lambda_2$ , pruning layer l, register neuron set  $\mathcal{K}_{\text{reg}}$ </td></tr><tr><td colspan="2">Ensure: Model response with adaptively pruned visual tokens</td></tr><tr><td colspan="2">1: Encode I into visual tokens V and obtain MLP activations h</td></tr><tr><td>2: For k ∈  $\mathcal{K}_{\text{reg}}$ : hr,k← maxvi∈V hi,k, hi,k← 0, ∀vi∈V</td><td>▷ redirect sinks to r</td></tr><tr><td colspan="2">Stage I: Image-Adaptive Redundancy Pruning</td></tr><tr><td>3: Compute s1(vi),s1(r) via [CLS] attention; S1← {vi∈V | s1(vi) ≥ λ1s1(r)} ∪ {r}</td><td>▷ Eq. (5)</td></tr><tr><td colspan="2">Stage II: Register-Anchored Relevance Pruning</td></tr><tr><td>4: Run LLM to layer l with [S1; T]; compute s2(vi),s2(r) from text attention</td><td>▷ Eq. (7)</td></tr><tr><td>5: S2← {vi∈S1\{r\}|s2(vi)≥λ2s2(r)} ∪ {r}</td><td>▷ Eq. (8)</td></tr><tr><td colspan="2">6:return Continue autoregressive decoding with [S2; T]</td></tr></table>

$$
s _ {2} (r) = \frac {1}{M} \sum_ {j = 1} ^ {M} A _ {t _ {j} \rightarrow r} ^ {(l)}, \tag {7}
$$

$$
s _ {2} (v _ {i}) = \max _ {j = 1, \dots , M} A _ {t _ {j} \to v _ {i}} ^ {(l)}, \quad \forall   v _ {i} \in \mathcal {S} _ {1} \setminus \{r \}.
$$

Using $\lambda _ { 2 } \cdot s _ { 2 } ( r )$ as the query-adaptive threshold, the final retained set is defined as

$$
\mathcal {S} _ {2} = \{v _ {i} \in \mathcal {S} _ {1} \setminus \{r \} \mid s _ {2} (v _ {i}) \geq \lambda_ {2} \cdot s _ {2} (r) \} \cup \{r \}, \tag {8}
$$

where $\lambda _ { 2 }$ controls the pruning intensity. Because $s _ { 2 } ( r )$ is computed from the current query-conditioned attention distribution, the threshold varies across queries. As a result, the same image can yield different final token budgets for different questions, enabling query-adaptive budget allocation.

# 4 Experiments

We evaluate OccamToken on image-understanding tasks using LLaVA-v1.5 7B (Liu et al., 2024a), LLaVA-NeXT 7B (Liu et al., 2024b), and Qwen3-VL 8B (Bai et al., 2025), covering the standard 576-token setting, the high-resolution 2,880-token setting, and a non-LLaVA-family architecture. Following prior work (Takezoe et al., 2026; Alvar et al., 2025), we report results on GQA (Hudson and Manning, 2019), ScienceQA (Lu et al., 2022), POPE (Li et al., 2023b), MME (Fu et al., 2025), MMBench (Liu et al., 2024c), VizWiz (Gurari et al., 2018) and RealworldQA(RQA.) (xAI, 2024). The full-token model serves as the upper bound for each backbone. We report benchmark-specific scores and relative accuracy (RelAcc.), computed by normalizing performance to the corresponding full-token model. For adaptive methods, we report the average retained visual-token count and compare against fixed-budget baselines under matched average budgets. Additional results and analyses are provided in the appendix.

# 4.1 Main Results on Image Understanding

Tables 1 and 2 report the main image-understanding results on LLaVA-v1.5 7B and LLaVA-NeXT 7B. These two backbones represent different visual-sequence regimes: LLaVA-v1.5 uses the standard 576-token image setting, while LLaVA-NeXT processes high-resolution inputs with up to 2,880 visual tokens. For fair comparison with fixed-budget baselines, we match the average number of retained visual tokens for OccamToken and report both benchmark-specific scores and relative accuracy.

Results on LLaVA-v1.5. Table 1 compares OccamToken with learned, fixed-budget, and image-adaptive pruning methods on LLaVA-v1.5 7B (Liu et al., 2024a). At average budgets of 128, 64, and 32 visual tokens, OccamToken preserves 99.1%, 98.1%, and 97.2% relative accuracy, respectively. This indicates that the method remains close to the full-token model even when the visual sequence is substantially reduced. Compared with training-free fixed-budget baselines (Yang et al., 2025; Wen et al., 2025), OccamToken shows a more stable performance under aggressive compression. For example, at the 32-token budget, several fixed-budget methods exhibit larger drops in relative accuracy, while OccamToken maintains stronger aggregate performance. OccamToken is also competitive with learned pruning methods (Shao et al., 2025; Takezoe et al., 2026), despite requiring no trainable pruning module or additional training data.

<table><tr><td>Method</td><td>Type</td><td>GQA</td><td> $SQA^1$ </td><td> $VQA^T$ </td><td>POPE</td><td>MME</td><td> $VQA^{v2}$ </td><td>MMB</td><td> $MMB^{CN}$ </td><td>RelAcc.</td></tr><tr><td colspan="11">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>-</td><td>61.9</td><td>69.5</td><td>58.2</td><td>85.9</td><td>1862</td><td>78.5</td><td>64.7</td><td>58.3</td><td>100.0%</td></tr><tr><td colspan="11">Retain Averaged 128 Tokens (↓ 77.8%)</td></tr><tr><td>VisionZip ‡</td><td>LEARNED</td><td>58.9</td><td>68.3</td><td>57.0</td><td>83.7</td><td>1823</td><td>76.6</td><td>62.6</td><td>-</td><td>97.3%</td></tr><tr><td>TwigVLM</td><td>LEARNED</td><td>60.6</td><td>69.5</td><td>57.8</td><td>86.6</td><td>1818</td><td>77.9</td><td>63.5</td><td>-</td><td>99.0%</td></tr><tr><td>LearnPruner</td><td>LEARNED</td><td>60.3</td><td>68.5</td><td>57.3</td><td>86.7</td><td>1820</td><td>77.3</td><td>63.8</td><td>56.8</td><td>98.5%</td></tr><tr><td>FastV</td><td>FIXED</td><td>49.6</td><td>60.2</td><td>50.6</td><td>59.6</td><td>1490</td><td>61.8</td><td>56.1</td><td>51.4</td><td>82.1%</td></tr><tr><td>SparseVLM</td><td>FIXED</td><td>56.0</td><td>67.1</td><td>54.9</td><td>80.5</td><td>1696</td><td>73.8</td><td>60.0</td><td>51.1</td><td>92.6%</td></tr><tr><td>DivPrune</td><td>FIXED</td><td>59.3</td><td>69.0</td><td>56.1</td><td>86.7</td><td>1718</td><td>76.0</td><td>62.0</td><td>54.8</td><td>96.4%</td></tr><tr><td>DART</td><td>FIXED</td><td>57.9</td><td>69.1</td><td>56.3</td><td>80.4</td><td>1721</td><td>74.7</td><td>60.7</td><td>57.3</td><td>95.4%</td></tr><tr><td>VisPruner</td><td>FIXED</td><td>58.2</td><td>69.1</td><td>57.0</td><td>84.6</td><td>1794</td><td>75.8</td><td>62.7</td><td>57.3</td><td>97.2%</td></tr><tr><td>VisionZip</td><td>FIXED</td><td>57.6</td><td>68.9</td><td>56.8</td><td>83.2</td><td>1762</td><td>75.6</td><td>62.0</td><td>56.7</td><td>96.3%</td></tr><tr><td>PruMerge+</td><td>IMG.</td><td>58.2</td><td>69.1</td><td>54.0</td><td>83.1</td><td>-</td><td>75.0</td><td>61.8</td><td>55.8</td><td>95.7%</td></tr><tr><td>PruneSID</td><td>IMG.</td><td>58.8</td><td>68.5</td><td>54.7</td><td>86.5</td><td>1749</td><td>75.3</td><td>62.1</td><td>-</td><td>96.3%</td></tr><tr><td>OccamToken</td><td>IMG.+QRY.</td><td>60.9</td><td>69.1</td><td>58.0</td><td>86.3</td><td>1825</td><td>77.9</td><td>63.9</td><td>57.5</td><td>99.1%</td></tr><tr><td colspan="11">Retain Averaged 64 Tokens (↓ 88.9%)</td></tr><tr><td>VisionZip ‡</td><td>LEARNED</td><td>57.0</td><td>68.8</td><td>56.0</td><td>80.9</td><td>1756</td><td>74.2</td><td>61.5</td><td>-</td><td>95.1%</td></tr><tr><td>TwigVLM</td><td>LEARNED</td><td>58.8</td><td>70.0</td><td>55.8</td><td>82.7</td><td>1760</td><td>75.6</td><td>60.4</td><td>-</td><td>96.0%</td></tr><tr><td>LearnPruner</td><td>LEARNED</td><td>58.9</td><td>68.3</td><td>56.6</td><td>86.8</td><td>1750</td><td>76.0</td><td>62.6</td><td>55.7</td><td>96.9%</td></tr><tr><td>FastV</td><td>FIXED</td><td>46.1</td><td>51.1</td><td>47.8</td><td>48.0</td><td>1256</td><td>55.0</td><td>48.0</td><td>42.7</td><td>71.4%</td></tr><tr><td>SparseVLM</td><td>FIXED</td><td>52.7</td><td>62.2</td><td>51.8</td><td>75.1</td><td>1505</td><td>68.2</td><td>56.2</td><td>46.1</td><td>85.6%</td></tr><tr><td>DivPrune</td><td>FIXED</td><td>57.8</td><td>68.2</td><td>54.7</td><td>85.6</td><td>1674</td><td>74.1</td><td>59.3</td><td>52.3</td><td>93.9%</td></tr><tr><td>DART</td><td>FIXED</td><td>54.7</td><td>69.3</td><td>54.7</td><td>73.9</td><td>1705</td><td>71.3</td><td>59.5</td><td>54.0</td><td>91.9%</td></tr><tr><td>VisPruner</td><td>FIXED</td><td>55.4</td><td>69.1</td><td>55.8</td><td>80.4</td><td>1689</td><td>72.7</td><td>61.3</td><td>55.1</td><td>93.9%</td></tr><tr><td>VisionZip</td><td>FIXED</td><td>55.1</td><td>69.0</td><td>55.5</td><td>77.0</td><td>1690</td><td>72.4</td><td>60.1</td><td>55.4</td><td>93.0%</td></tr><tr><td>PruMerge+</td><td>IMG.</td><td>55.4</td><td>69.5</td><td>52.0</td><td>75.7</td><td>-</td><td>71.3</td><td>59.6</td><td>52.1</td><td>91.3%</td></tr><tr><td>PruneSID</td><td>IMG.</td><td>57.1</td><td>67.8</td><td>54.2</td><td>83.8</td><td>1733</td><td>73.7</td><td>58.8</td><td>-</td><td>94.0%</td></tr><tr><td>OccamToken</td><td>IMG.+QRY.</td><td>59.3</td><td>69.0</td><td>57.4</td><td>86.2</td><td>1801</td><td>77.2</td><td>63.2</td><td>56.9</td><td>98.1%</td></tr><tr><td colspan="11">Retain Averaged 32 Tokens (↓ 94.4%)</td></tr><tr><td>LearnPruner</td><td>LEARNED</td><td>57.2</td><td>68.2</td><td>56.1</td><td>84.5</td><td>1672</td><td>74.0</td><td>60.8</td><td>55.5</td><td>94.8%</td></tr><tr><td>FastV</td><td>FIXED</td><td>41.5</td><td>42.6</td><td>42.5</td><td>32.5</td><td>1090</td><td>43.4</td><td>37.8</td><td>33.2</td><td>58.5%</td></tr><tr><td>SparseVLM</td><td>FIXED</td><td>48.3</td><td>57.3</td><td>46.1</td><td>67.9</td><td>1290</td><td>58.6</td><td>51.4</td><td>40.6</td><td>76.5%</td></tr><tr><td>DivPrune</td><td>FIXED</td><td>54.9</td><td>68.6</td><td>52.9</td><td>81.5</td><td>1611</td><td>71.2</td><td>57.6</td><td>49.1</td><td>90.5%</td></tr><tr><td>DART</td><td>FIXED</td><td>52.9</td><td>69.3</td><td>52.2</td><td>69.1</td><td>1615</td><td>67.1</td><td>58.5</td><td>50.0</td><td>88.0%</td></tr><tr><td>VisPruner</td><td>FIXED</td><td>52.2</td><td>69.2</td><td>53.9</td><td>72.7</td><td>1567</td><td>67.7</td><td>58.4</td><td>52.7</td><td>89.0%</td></tr><tr><td>VisionZip</td><td>FIXED</td><td>51.8</td><td>68.8</td><td>53.1</td><td>68.7</td><td>1536</td><td>67.1</td><td>57.7</td><td>50.3</td><td>87.2%</td></tr><tr><td>PruMerge+</td><td>IMG.</td><td>52.9</td><td>67.9</td><td>49.2</td><td>66.7</td><td>-</td><td>65.6</td><td>55.1</td><td>45.9</td><td>84.7%</td></tr><tr><td>OccamToken</td><td>IMG.+QRY.</td><td>59.1</td><td>69.1</td><td>56.3</td><td>86.2</td><td>1780</td><td>75.5</td><td>62.9</td><td>56.2</td><td>97.2%</td></tr></table>

Table 1 Performance comparison on LLaVA-v1.5 7B. Learned: methods with trainable pruning modules; Fixed: methods using a fixed retained-token count or ratio; Img.: image-adaptive budget allocation; Img.+Qry.: joint image- and query-adaptive budget allocation.

Results on LLaVA-NeXT. Table 2 evaluates OccamToken on LLaVA-NeXT 7B (Liu et al., 2024b), where the original visual sequence is much longer. This setting is more demanding because both the prefill cost and the amount of image-dependent redundancy increase with the number of visual tokens. At average budgets of 320 and 160 tokens, OccamToken preserves 97.3% and 95.9% relative accuracy, respectively. Under the more aggressive 40-token setting, which retains only about 1.4% of the original visual tokens, OccamToken still maintains 93.8% relative accuracy. The compared fixed-budget methods (Alvar et al., 2025) degrade more noticeably in this low-budget regime, suggesting that adaptive token allocation is particularly useful when the retained visual evidence is highly constrained.

<table><tr><td>Method</td><td>Venue</td><td>Params.</td><td>GQA</td><td> $SQA^I$ </td><td> $VQA^T$ </td><td>MME</td><td> $VQA^{v2}$ </td><td>MMB</td><td>POPE</td><td>RelAcc.</td></tr><tr><td colspan="11">Upper Bound, 2,880 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>-</td><td>-</td><td>64.2</td><td>70.2</td><td>61.3</td><td>1842</td><td>81.2</td><td>67.9</td><td>86.8</td><td>100%</td></tr><tr><td colspan="11">Retain Averaged 320 Tokens (↓ 88.9%)</td></tr><tr><td>TwigVLM</td><td>ICCV-25</td><td>610M</td><td>62.2</td><td>68.7</td><td>57.4</td><td>1758</td><td>79.7</td><td>65.0</td><td>-</td><td>96.3%</td></tr><tr><td>DivPrune</td><td>CVPR-25</td><td>-</td><td>61.1</td><td>67.7</td><td>56.2</td><td>1721</td><td>77.2</td><td>65.1</td><td>84.7</td><td>95.0%</td></tr><tr><td>VisionZip</td><td>CVPR-25</td><td>-</td><td>59.3</td><td>67.3</td><td>58.9</td><td>1689</td><td>76.2</td><td>63.4</td><td>82.1</td><td>94.0%</td></tr><tr><td>DART</td><td>EMNLP-25</td><td>-</td><td>59.5</td><td>67.5</td><td>57.6</td><td>1710</td><td>75.7</td><td>64.2</td><td>81.0</td><td>93.8%</td></tr><tr><td>OccamToken</td><td>-</td><td>-</td><td>63.0</td><td>68.7</td><td>58.5</td><td>1771</td><td>80.0</td><td>65.5</td><td>85.7</td><td>97.3%</td></tr><tr><td colspan="11">Retain Averaged 160 Tokens (↓ 94.4%)</td></tr><tr><td>DivPrune</td><td>CVPR-25</td><td>-</td><td>59.3</td><td>67.2</td><td>54.1</td><td>1614</td><td>75.0</td><td>63.2</td><td>80.0</td><td>91.7%</td></tr><tr><td>VisionZip</td><td>CVPR-25</td><td>-</td><td>55.5</td><td>68.3</td><td>56.2</td><td>1607</td><td>71.4</td><td>60.4</td><td>74.8</td><td>89.4%</td></tr><tr><td>DART</td><td>EMNLP-25</td><td>-</td><td>56.8</td><td>67.8</td><td>54.9</td><td>1700</td><td>72.5</td><td>62.0</td><td>75.3</td><td>90.6%</td></tr><tr><td>OccamToken</td><td>-</td><td>-</td><td>61.3</td><td>68.5</td><td>57.4</td><td>1740</td><td>77.2</td><td>65.2</td><td>85.9</td><td>95.9%</td></tr><tr><td colspan="11">Retain Averaged 40 Tokens (↓ 98.6%)</td></tr><tr><td>VisionZip</td><td>CVPR-25</td><td>-</td><td>42.2</td><td>65.2</td><td>43.0</td><td>991</td><td>48.9</td><td>51.2</td><td>24.6</td><td>63.8%</td></tr><tr><td>DivPrune</td><td>CVPR-25</td><td>-</td><td>54.0</td><td>66.8</td><td>49.1</td><td>1257</td><td>67.8</td><td>57.7</td><td>76.4</td><td>83.4%</td></tr><tr><td>OccamToken</td><td>-</td><td>-</td><td>59.5</td><td>68.5</td><td>51.8</td><td>1744</td><td>75.6</td><td>64.9</td><td>85.7</td><td>93.8%</td></tr></table>

Table 2 Performance comparison on LLaVA-NeXT 7B under different token retention budgets. 

<table><tr><td>Method</td><td>MME</td><td>MMB</td><td>VizWiz</td><td>POPE</td><td>GQA</td><td>RQA</td><td> $VQA^T$ </td><td>SQA</td><td>RelAcc.</td></tr><tr><td colspan="10">Upper Bound (100%)</td></tr><tr><td>Baseline</td><td>2375</td><td>84.5</td><td>44.8</td><td>88.5</td><td>61.9</td><td>71.5</td><td>80.1</td><td>94.2</td><td>100%</td></tr><tr><td colspan="10">Retain 22.2% (↓77.8%)</td></tr><tr><td>DivPrune</td><td>2073</td><td>82.7</td><td>42.1</td><td>87.4</td><td>57.8</td><td>63.4</td><td>72.1</td><td>84.8</td><td>92.5%</td></tr><tr><td>Ours</td><td>2314</td><td>81.5</td><td>41.9</td><td>89.2</td><td>60.0</td><td>68.1</td><td>74.1</td><td>86.8</td><td>95.6%</td></tr><tr><td colspan="10">Retain 11.1% (↓88.9%)</td></tr><tr><td>DivPrune</td><td>1880</td><td>78.7</td><td>38.8</td><td>83.7</td><td>54.4</td><td>59.2</td><td>65.9</td><td>80.3</td><td>86.5%</td></tr><tr><td>Ours</td><td>2243</td><td>79.2</td><td>39.0</td><td>89.0</td><td>58.4</td><td>65.6</td><td>70.2</td><td>81.4</td><td>92.0%</td></tr></table>

Table 3 Performance comparison on Qwen3-VL 8B.

<table><tr><td>Method</td><td>Tok.</td><td>Pre.ms</td><td>Dec.ms</td><td>KVMB</td><td>Acc.</td></tr><tr><td>v1.5</td><td>576</td><td>59</td><td>13.0</td><td>313</td><td>85.9</td></tr><tr><td>VisionZip</td><td>128</td><td>28</td><td>12.3</td><td>89</td><td>83.2</td></tr><tr><td>Ours</td><td>128</td><td>30</td><td>12.4</td><td>92</td><td>86.3</td></tr><tr><td>VisionZip</td><td>32</td><td>23</td><td>12.1</td><td>41</td><td>68.7</td></tr><tr><td>Ours</td><td>32</td><td>26</td><td>12.3</td><td>47</td><td>86.2</td></tr><tr><td>NeXT</td><td>2880</td><td>164</td><td>16.2</td><td>1093</td><td>86.8</td></tr><tr><td>VisionZip</td><td>160</td><td>45</td><td>11.9</td><td>105</td><td>74.8</td></tr><tr><td>Ours</td><td>160</td><td>53</td><td>12.1</td><td>125</td><td>85.9</td></tr></table>

Table 4 Efficiency on LLaVA models.

Overall, these results show that OccamToken provides a consistent accuracy–compression trade-off across both standard and long visual-sequence settings. Rather than assigning the same retained-token count to every sample, the register-anchored threshold allows the retained budget to vary with the current attention distribution. This behavior is especially beneficial under stronger compression, where preserving a small amount of task-relevant visual evidence becomes more important.

Transfer to Another Backbone. To examine whether OccamToken is tied to a specific LLaVA-style architecture, we further evaluate it on Qwen3-VL 8B (Bai et al., 2025). As shown in Table 3, OccamToken improves over the compared training-free baseline under both retention ratios. At 22.2% token retention, it achieves 95.6% relative accuracy; at the stronger 11.1% retention setting, it preserves 92.0% relative accuracy. These results suggest that the register-anchored pruning rule is not tied to LLaVA-specific token interfaces and can be applied to VLMs beyond the LLaVA family.

![](images/718c3dad0f580df623d629b815572d609dda89c8f2837be8cfd2e376123f5d1e.jpg)

<details>
<summary>line</summary>

(a) GQA + TextVQA
| Visual Token Number | OccamToken (%) | w/o Stage-I (%) | GQA UB=61.9 (%) | TextVQA UB=58.2 (%) | left axis dataset (%) | right axis dataset (%) |
|---|---|---|---|---|---|---|
| 32 | 59 | 58.7 | 61.9 | 58.5 | 56.5 | 56.5 |
| 64 | 59.3 | 58.4 | 61.9 | 58.5 | 56.8 | 57.0 |
| 128 | 61.0 | 59.9 | 61.9 | 58.5 | 57.0 | 57.5 |
</details>

![](images/89070a3a5fb376ab0264a944fa14553a414271cdca31b61a64eb94795bac01f4.jpg)

<details>
<summary>line</summary>

| Visual Token Number | OccamToken | w/o Stage-I | MMBench UB=64.7 | POPE UB=85.9 | left axis dataset | right axis dataset |
| ------------------- | ---------- | ----------- | --------------- | ------------ | ----------------- | ------------------ |
| 32                  | 63.0       | 62.0        | 64.5            | 86.5         | 86.0              | 85.0               |
| 64                  | 64.5       | 61.5        | 64.5            | 86.5         | 86.0              | 85.0               |
| 128                 | 64.5       | 62.0        | 64.5            | 86.5         | 86.0              | 85.0               |
</details>

![](images/a9f94072d2541f050600794f5b6419f94a72e825142192888cb7aa5f8bfb4ee3.jpg)

<details>
<summary>bar</summary>

(c) Latency
| Method | Latency (ms) |
|---|---|
| Baseline | 73 |
| w/o λ₁ | 41 |
| λ₁ = 0.035 | 35 |
</details>

Figure 4 Ablation and efficiency analysis. Full OccamToken consistently improves over the variant without Stage I under matched token budgets, confirming the benefit of two-stage adaptive pruning. Stage-I also reduces latency, yielding a better accuracy–efficiency trade-off.

# 4.2 Ablation Study.

We ablate the key components of OccamToken to examine whether the two-stage design is necessary. The comparison between OccamToken and the variant without Stage I is particularly informative. Although both variants perform query-adaptive pruning in the language model, removing Stage I consistently reduces relative accuracy under matched average token budgets. This shows that Stage I is not merely an efficiency shortcut; it removes image-level redundancy before the visual tokens enter LLM computation and provides a cleaner candidate set for subsequent query-adaptive pruning.

# 4.3 Efficiency Analysis

We further evaluate whether OccamToken improves inference efficiency while preserving visual understanding. Since visual tokens dominate the multimodal prefix, reducing the retained visual sequence directly lowers the prefill cost and KV-cache memory. On LLaVA-v1.5 7B, OccamToken reduces the visual sequence from 576 tokens to 32 tokens, decreasing the KV cache from 313 MB to 47 MB while maintaining 86.2 POPE accuracy. On LLaVA-NeXT 7B, it reduces 2,880 visual tokens to 160 tokens, lowering the KV cache from 1093 MB to 125 MB while preserving 85.9 POPE accuracy. Compared with simpler fixed-budget methods such as VisionZip (Yang et al., 2025), OccamToken is not always the fastest method at the same final token count because it introduces lightweight two-stage adaptive scoring. However, this small overhead leads to substantially better accuracy preservation, particularly under aggressive compression. Therefore, OccamToken should be understood not as a latency-only method, but as a practical accuracy–efficiency trade-off: it significantly reduces prefill and memory costs while retaining reliable visual evidence.

# 5 Conclusion

This paper presents OccamToken, a training-free framework for efficient VLM inference through registeranchored visual token pruning. Instead of relying on fixed top-K selection, OccamToken uses a test-time register token as a dynamic reference to mitigate attention-sink distortion and adapt the pruning threshold to each input. Its two-stage design further combines image-level redundancy removal with query-conditioned relevance selection, enabling adaptive token allocation without additional training or trainable pruning modules. Experiments across LLaVA-v1.5, LLaVA-NeXT and Qwen3-VL show that OccamToken consistently improves the accuracy–efficiency trade-off. In particular, it pushes training-free visual token compression into the 1.4% retention regime on LLaVA-NeXT while preserving over 93% of the full-token accuracy. These results suggest that reference-based adaptive pruning is a simple and effective direction for scalable multimodal inference.

Limitations. OccamToken is training-free at the pruning stage, but it relies on constructing a reliable test-time register token for the target VLM. While this is effective for the Transformer-based backbones evaluated in this work, architectures with different feed-forward blocks, attention interfaces, or vision–language projection designs may require additional adaptation. Extending register construction and reference-based pruning to broader multimodal architectures is a natural direction for future work.

# References

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9392–9401, 2025.   
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.   
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.   
Federico Barbero, Alvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, and Razvan Pascanu. Why do llms attend to the first token? arXiv preprint arXiv:2504.02732, 2025.   
Nicola Cancedda. Spectral filters, dark signals, and attention sinks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4792–4808, 2024.   
Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer, 2024.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.   
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.   
Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In International Conference on Learning Representations (ICLR), 2024.   
Jinhong Deng, Wen Li, Joey Tianyi Zhou, and Yang He. Scope: Saliency-coverage oriented token pruning for efficient multimodel llms. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.   
Mohamed Dhouib, Davide Buscaldi, Sonia Vanier, and Aymen Shabou. Pact: Pruning and clustering-based token reduction for faster visual language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14582–14592, 2025.   
Zhengyao Fang, Pengyuan Lyu, Chengquan Zhang, Guangming Lu, Jun Yu, and Wenjie Pei. Prune redundancy, preserve essence: Vision token compression in vlms via synergistic importance-diversity. In The Fourteenth International Conference on Learning Representations, 2026.   
Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.   
Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.   
Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. In The Thirteenth International Conference on Learning Representations, 2025.   
Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018.   
Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaosheng Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, and Shaohui Lin. Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification. In The Thirteenth International Conference on Learning Representations, 2025.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.   
Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2758–2766, 2017.   
Nicholas Jiang, Amil Dravid, Alexei A Efros, and Yossi Gandelsman. Vision transformers don’t need trained registers. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.   
Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023a.   
Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.   
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305, 2023b.   
Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 5971–5984, 2024.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024a.   
Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024b.   
Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024c.   
Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521, 2022.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.   
Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22857–22867, 2025.   
Zhenwei Shao, Mingyang Wang, Zhou Yu, Wenwen Pan, Yan Yang, Tao Wei, Hongyuan Zhang, Ning Mao, Wei Chen, and Jun Yu. Growing a twig to accelerate large vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20064–20074, 2025.   
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.   
Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. In First Conference on Language Modeling, 2024.   
Rinyoichi Takezoe, Yaqian Li, Zi-Hao Bo, Anzhou Hou, Mo Guang, and Kaiwen Long. Learnpruner: Rethinking attention-based token pruning in vision language models. In The Fourteenth International Conference on Learning Representations, 2026.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.   
Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for “important tokens” in multimodal language models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9972–9991, 2025.   
xAI. RealworldQA: A benchmark for real-world spatial understanding. https://huggingface.co/datasets/xai-org/ RealworldQA, 2024. Accessed: 2025-04-26.   
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024.   
Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia, pages 1645–1653, 2017.   
Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19792–19802, 2025.   
Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 22128–22136, 2025a.   
Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. Atp-llava: Adaptive token pruning for large vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24972–24982, 2025b.   
Zhongzhi Yu, Zheng Wang, Yonggan Fu, Huihong Shi, Khalid Shaikh, and Yingyan Celine Lin. Unveiling and harnessing hidden attention sinks: Enhancing large language models without training through attention calibration. arXiv preprint arXiv:2406.15765, 2024.   
Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 9127–9134, 2019.   
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.   
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.   
Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20857–20867, 2025a.   
Qizhe Zhang, Mengzhen Liu, Lichen Li, Ming Lu, Yuan Zhang, Junwen Pan, Qi She, and Shanghang Zhang. Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b.   
Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis A Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In International Conference on Machine Learning, pages 74840–74857. PMLR, 2025c.

![](images/c44219179935181b966c3258a3ba327e7661d9df862644876c378578fc43d2a7.jpg)

<details>
<summary>line</summary>

| Pruning Layer | Accuracy (%) |
| ------------- | ------------ |
| 8             | 56.8         |
| 9             | 58.1         |
| 10            | 59.0         |
| 11            | 59.3         |
| 12            | 59.1         |
| 13            | 59.3         |
| 14            | 59.5         |
</details>

![](images/1886af40d044d777ceceb6d47f74ed53abdcd7e6ca5a7d211331a44987655f35.jpg)

<details>
<summary>line</summary>

| Pruning Layer | F1-Score (%) |
| ------------- | ------------ |
| 8             | 83.5         |
| 9             | 85.5         |
| 10            | 85.9         |
| 11            | 86.2         |
| 12            | 86.5         |
| 13            | 86.6         |
| 14            | 86.4         |
</details>

![](images/64228402769077836d6ec1f80bc529c41b99cd793c8d21e1a22182f4953adb6b.jpg)

<details>
<summary>line</summary>

| Pruning Layer | Accuracy (%) |
| ------------- | ------------ |
| 8             | 60.0         |
| 9             | 61.9         |
| 10            | 62.7         |
| 11            | 63.2         |
| 12            | 63.1         |
| 13            | 63.3         |
</details>

Figure 5 Effect of the Stage II pruning layer on LLaVA-v1.5. We evaluate OccamToken by applying query-adaptive pruning at different LLM decoder layers under the same token budget. Across GQA, POPE, and MMB, performance improves rapidly from shallow layers and becomes largely stable around layer 11. Considering both accuracy and efficiency, we select layer 11 as the default Stage-II pruning layer for LLaVA-v1.5. 

<table><tr><td>Setting</td><td> $\lambda_1$ </td><td> $\lambda_2$ </td><td> $N_1$ </td><td> $N_2$ </td><td>Speedup</td><td>POPE</td></tr><tr><td>w/o Stage I</td><td>0</td><td>1</td><td>2880</td><td>160.8</td><td>1.00×</td><td>85.7</td></tr><tr><td>Mild</td><td>0.01</td><td>1.08</td><td>1274</td><td>163.9</td><td>1.21×</td><td>86.3</td></tr><tr><td>Moderate</td><td>0.025</td><td>0.62</td><td>833</td><td>157.4</td><td>1.37×</td><td>86.0</td></tr><tr><td>Aggressive</td><td>0.05</td><td>1.3</td><td>470</td><td>161.3</td><td>1.54×</td><td>85.6</td></tr><tr><td>Aggressive</td><td>0.045</td><td>1.3</td><td>523</td><td>159.3</td><td>1.51×</td><td>85.9</td></tr></table>

Table 5 Stage I ablation and $\lambda _ { 1 }$ sensitivity on POPE with LLaVA-NeXT. We adjust $\lambda _ { 2 }$ to keep the final retained token budget comparable across settings. $N _ { 1 }$ and $N _ { 2 }$ denote the average number of visual tokens retained after Stage I and Stage II. Speedup is computed against the setting without Stage I pruning.

# Appendix

The appendix is organized as follows. Appendix A provides implementation details, including parameter settings, register construction, pruning-layer selection, architecture-specific instantiations, efficiency measurement, and component ablations. Appendix B describes the image and video benchmarks used in our evaluation, together with their evaluation protocols and metrics. Appendix C reports additional experimental results, including video understanding, Mini-Gemini, larger LLaVA backbones, and transfer experiments on non-LLaVA architectures. Appendix D discusses the broader impact of efficient VLM inference.

# A Implementation Details

Parameter settings. All experiments are conducted on a single NVIDIA RTX PRO 6000 GPU. We select the pruning layer and Stage-I coefficient according to the empirical accuracy–efficiency trade-off shown in Figure 5 and Table 5.

For Stage-II pruning, Figure 5 shows that the pruning layer has a clear effect on downstream performance. When query-adaptive pruning is applied at shallow decoder layers, the performance is consistently lower on GQA, POPE, and MMB. This suggests that early LLM layers have not yet formed sufficiently reliable text-to-vision alignment, making their attention signals less suitable for query-conditioned token selection. As the pruning layer moves deeper, performance improves rapidly and becomes largely stable around layer 11. After this point, applying pruning at later layers brings only marginal accuracy gains, but reduces the efficiency benefit because more decoder layers have already processed the full visual sequence. Therefore, we choose layer 11 as the default Stage II pruning layer for LLaVA-series models, which provides a practical balance between reliable query-aware attention and early enough token removal. For Qwen-series models, we use layer 10 following the same principle: pruning is applied at an intermediate decoder layer where cross-modal attention is sufficiently formed while substantial later computation can still benefit from the reduced visual prefix.

For Stage I pruning, we set $\lambda _ { 1 }$ according to the visual-token scale of each backbone. For LLaVA-v1.5, whose visual prefix contains 576 tokens, we use $\lambda _ { 1 }$ in the range of 0.01–0.02, which keeps about 200–300 candidate tokens before Stage II pruning. For LLaVA-NeXT, whose visual prefix contains 2,880 tokens, stronger Stage I pruning is necessary to remove the much larger amount of image-level redundancy, and we use $\lambda _ { 1 }$ in the range of 0.04–0.05.

Table 5 provides a detailed sensitivity analysis on LLaVA-NeXT. Without Stage I, Stage II must operate over all 2,880 visual tokens and obtains 85.7 POPE with baseline speed. A mild Stage I setting already reduces the candidate set to 1,274 tokens and improves the speed to 1.21×, while slightly increasing POPE to 86.3. A moderate setting with $\lambda _ { 1 } = 0 . 0 2 5$ further reduces the candidate set to 833 tokens and reaches 1.37× speedup with 86.0 POPE. Among the stronger settings, we choose $\lambda _ { 1 } = 0 . 0 4 5$ as the default for LLaVA-NeXT. This setting keeps 523 candidate tokens after Stage I, achieves a 1.51× speedup, and maintains 85.9 POPE. Compared with $\lambda _ { 1 } = 0 . 0 2 5$ , it preserves the same POPE score while improving speed from 1.37× to 1.51×. Compared with the more aggressive $\lambda _ { 1 } = 0 . 0 5$ , it retains nearly the same speedup, 1.51× versus 1.54×, but improves POPE from 85.6 to 85.9. Thus, $\lambda _ { 1 } = 0 . 0 4 5$ lies near the practical Pareto frontier: it captures almost the efficiency gain of aggressive pruning while avoiding the accuracy degradation caused by removing too many candidates too early.

This parameter choice also matches the intended role of Stage I. Stage I is query-agnostic, so it should remove broad image-level redundancy rather than make the final evidence-selection decision. If Stage-I is too weak, many redundant tokens remain and the speedup is limited. If Stage I is too aggressive, potentially query-relevant evidence may be removed before Stage II can evaluate it. The selected $\lambda _ { 1 } = 0 . 0 4 5$ removes over 80% of the original visual tokens before Stage II, keeps the final budget comparable, and still maintains accuracy above the no-Stage I baseline. This makes it a robust default for high-resolution LLaVA-NeXT experiments.

We construct one register token using the top-10 register-related sparse neurons. Since these neurons are stable across samples, we identify them from a small calibration subset and cache the neuron indices for reuse. OccamToken does not enforce a hard token budget. Instead, we use 50–100 calibration samples to estimate the relationship between pruning coefficients and the average retained-token count, and then report the average budget over the full evaluation set. In practice, the results are stable under small coefficient variations, making budget matching simple and reliable.

Register construction. For Stage I, we construct the test-time register token following Jiang et al. Jiang et al. (2025). Let $\mathcal { V } = \{ v _ { i } \} _ { i = 1 } ^ { N }$ 1 denote the visual patch tokens produced by the vision encoder, and let $h _ { i , k }$ denote the activation of token $v _ { i }$ on channel k in the corresponding MLP layer. Given the register-related neuron set $\kappa _ { \mathrm { r e g } } .$ , we redirect sink-like channel activations from ordinary visual tokens to the appended register token r:

$$
h _ {r, k} \leftarrow \max _ {v _ {i} \in \mathcal {V}} h _ {i, k}, \quad h _ {i, k} \leftarrow 0, \forall v _ {i} \in \mathcal {V}, \quad k \in \mathcal {K} _ {\text { reg }}. \tag {9}
$$

The register token is then inserted into the visual-token sequence, forming $\widetilde { \mathcal { V } } = \mathcal { V } \cup \{ r \}$ , and participates in the same attention computation as ordinary patch tokens. Unless otherwise specified, the register token is always preserved after pruning, since it serves as the reference anchor for both image-adaptive and query-adaptive pruning.

Stage-I pruning. Stage I is applied at the output of the vision encoder before visual tokens are projected into the language-model embedding space. For VLMs whose vision encoder provides a [CLS] token, such as LLaVA-v1.5 and LLaVA-NeXT, we use the [CLS] token as the image-level evaluator. Let $A ^ { \mathrm { v e } }$ denote the attention map in the vision encoder after register construction, averaged over attention heads when multi-head attention is used. We compute the Stage-I visual-token score and register reference score as

$$
s _ {1} (v _ {i}) = A _ {\mathrm{cls} \rightarrow v _ {i}} ^ {\mathrm{ve}}, \quad s _ {1} (r) = A _ {\mathrm{cls} \rightarrow r} ^ {\mathrm{ve}}. \tag {10}
$$

The Stage-I pruning threshold is defined by the scaled register score,

$$
\tau_ {1} = \lambda_ {1} s _ {1} (r), \tag {11}
$$

and the retained token set is

$$
\mathcal {S} _ {1} = \{v _ {i} \in \mathcal {V} \mid s _ {1} (v _ {i}) \geq \tau_ {1} \} \cup \{r \}. \tag {12}
$$

For architectures without an explicit [CLS] token, we use the mutual-scoring variant described in Section 3.4. In this case, token importance is estimated by the average attention received from visual and register tokens:

$$
s _ {1} ^ {\text { mut }} (v _ {i}) = \frac {1}{| \mathcal {V} | + 1} \sum_ {u \in \mathcal {V} \cup \{r \}} A _ {u \rightarrow v _ {i}} ^ {\text { ve }}, \tag {13}
$$

$$
s _ {1} ^ {\mathrm{mut}} (r) = \frac {1}{| \mathcal {V} | + 1} \sum_ {u \in \mathcal {V} \cup \{r \}} A _ {u \rightarrow r} ^ {\mathrm{ve}}.
$$

The same pruning rule in Eq. (12) is then applied by replacing $s _ { 1 } ( \cdot )$ with $s _ { 1 } ^ { \mathrm { m u t } } ( \cdot )$ .

Stage-II pruning. Stage II is applied inside the language model to further remove query-irrelevant visual tokens. After Stage I, the retained visual tokens and the register token are projected into the language embedding space:

$$
\mathbf {Z} _ {1} = p \left(\mathcal {S} _ {1}\right), \quad \mathbf {X} ^ {(0)} = \left[ \mathbf {Z} _ {1}; \mathbf {T} \right], \tag {14}
$$

where $p ( \cdot )$ denotes the visual projector and $\mathbf { T } = \{ t _ { j } \} _ { j = 1 } ^ { M }$ denotes the embedded text query. We run the language model until the pruning layer l and extract the text-to-visual/register attention scores. Following our main setting, Stage II is applied at the 11th decoder layer.

For each surviving visual token, we compute its query-conditioned score by the maximum attention received from all text tokens:

$$
s _ {2} (v _ {i}) = \max _ {j = 1, \dots , M} A _ {t _ {j} \rightarrow v _ {i}} ^ {(l)}, \quad v _ {i} \in \mathcal {S} _ {1} \setminus \{r \}. \tag {15}
$$

For the register token, we use the mean text-to-register attention as a stable query-conditioned reference:

$$
s _ {2} (r) = \frac {1}{M} \sum_ {j = 1} ^ {M} A _ {t _ {j} \rightarrow r} ^ {(l)}. \tag {16}
$$

The Stage II threshold and final retained set are therefore

$$
\begin{array}{l} \tau_ {2} = \lambda_ {2} s _ {2} (r), \\ \mathcal {S} _ {2} = \left\{v _ {i} \in \mathcal {S} _ {1} \setminus \{r \} \mid s _ {2} (v _ {i}) \geq \tau_ {2} \right\} \tag {17} \\ \cup \{r \}. \\ \end{array}
$$

The coefficient $\lambda _ { 2 }$ controls the overall compression strength and is adjusted only to match the target average token budget used for comparison with fixed-budget baselines.

Dynamic budgets and fair comparison. Unlike fixed top-K pruning, OccamToken does not enforce the same number of retained tokens for every input. For an evaluation set $\mathcal { D } ,$ , the reported retained token count is the dataset-level average

$$
\overline {{K}} = \frac {1}{| \mathcal {D} |} \sum_ {(I, T) \in \mathcal {D}} | \mathcal {S} _ {2} (I, T) |. \tag {18}
$$

When comparing against a fixed-budget baseline with target budget $K _ { \mathrm { t a r } } .$ , we select $\lambda _ { 2 }$ such that the average retained token count matches the target budget:

$$
\lambda_ {2} ^ {\star} = \arg \min _ {\lambda_ {2}} \left| \frac {1}{| \mathcal {D} |} \sum_ {(I, T) \in \mathcal {D}} | \mathcal {S} _ {2} (I, T; \lambda_ {2}) | - K _ {\text { tar }} \right|. \tag {19}
$$

This protocol ensures that performance differences are attributed to the pruning criterion rather than to retaining more visual tokens on average. For LLaVA-v1.5, we match the average budgets of 128/64/32 tokens. For LLaVA-NeXT, we match the average budgets of 640/320/160/40 tokens. Because OccamToken uses a dynamic pruning boundary, individual samples may retain more or fewer tokens than the target budget depending on their image-level redundancy and query-conditioned evidence demand.

Architecture-specific instantiations. For LLaVA-v1.5 and LLaVA-NeXT, Stage I uses [CLS]-to-register and [CLS]-to-visual attention in the vision encoder. For Qwen2-VL, whose visual encoder does not expose an explicit [CLS] token, we instantiate Stage I with the mutual-scoring scheme in Eq. (13). For Video-LLaVA, we treat visual tokens from all frames as a unified visual-token set:

$$
\mathcal {V} _ {\text { video }} = \bigcup_ {f = 1} ^ {F} \mathcal {V} ^ {(f)}, \tag {20}
$$

where V(f) denotes the visual tokens extracted from the f-th frame. We then apply the same two-stage pruning procedure without additional temporal modules. This keeps OccamToken training-free and plug-and-play across image and video VLMs.

Efficiency measurement. All experiments are conducted on a single NVIDIA Pro 6000 GPU. For efficiency evaluation, we reportprefill latency, decode time, and KV-cache memory. Let N denote the original number of visual tokens and K denote the average number of retained visual tokens after pruning. Since the languagemodel prefill cost grows with the multimodal sequence length, reducing the visual prefix changes the effective prefill length from N + M to approximately $\overline { { K } } + M$ , where M is the text length. The KV-cache memory also scales linearly with the retained sequence length:

$$
\mathrm{Mem} _ {\mathrm{KV}} \propto 2 L H d _ {h} \cdot (\overline {{K}} + M), \tag {21}
$$

where L is the number of decoder layers, H is the number of attention heads, and $d _ { h }$ is the head dimension. Prefill latency is averaged over the evaluation samples under the same decoding configuration. Since OccamToken only introduces register construction and lightweight threshold comparisons, the measured speedup mainly comes from reducing the number of visual tokens processed by the language model.

Compact component ablation. Table 6 summarizes a compact component ablation on Qwen3-VL 8B under matched average token retention ratios. OccamToken achieves the highest RelAcc. at both 22.2% and 11.1% retention, obtaining 95.6% and 92.0%, respectively. Compared with the fixed-budget variant, OccamToken improves RelAcc. by 2.5 and 3.3 points, indicating that adaptive token allocation is beneficial when the retained visual budget is constrained. Removing Stage-I leads to smaller drops of 0.4 and 0.2 RelAcc., suggesting that image-level redundancy pruning provides a modest but consistent complementary benefit before query-adaptive pruning. Replacing the register-anchored threshold with a mean-based threshold reduces RelAcc. by 0.7 and 1.3 points, showing that the register score is a more effective pruning reference than a generic mean statistic. Furthermore, removing register construction from the mean-threshold variant further decreases RelAcc. from 94.9 to 94.3 at 22.2% retention and from 90.7 to 90.2 at 11.1% retention, suggesting that test-time register construction also contributes to more stable token selection. Overall, these results support the combined roles of adaptive allocation, register construction, register-anchored thresholding, and Stage-I pruning in OccamToken.

# B Evaluation Benchmarks

We evaluate OccamToken on a diverse collection of image and video understanding benchmarks to assess its accuracy–efficiency trade-off under different visual reasoning scenarios. The image benchmarks cover general visual question answering, compositional reasoning, OCR-oriented understanding, object hallucination, multilingual evaluation, expert-level multimodal reasoning, and robustness to real-world low-quality images. The video benchmarks further test whether the proposed pruning strategy can handle temporally redundant visual tokens while preserving evidence for spatio-temporal reasoning. Overall, our evaluation includes 11 image understanding benchmarks and 4 video understanding benchmarks.

<table><tr><td rowspan="2">Variant</td><td colspan="4">Design Choice</td><td colspan="2">RelAcc. (%)</td></tr><tr><td>Fixed Budget</td><td>Build Reg.</td><td>Reg. Threshold</td><td>Stage-I</td><td>22.2%</td><td>11.1%</td></tr><tr><td>OccamToken</td><td>×</td><td>√</td><td>√</td><td>√</td><td>95.6</td><td>92.0</td></tr><tr><td>Fixed budget</td><td>√</td><td>√</td><td>×</td><td>√</td><td>93.1</td><td>88.7</td></tr><tr><td>w/o Stage-I</td><td>×</td><td>√</td><td>√</td><td>×</td><td>95.2</td><td>91.8</td></tr><tr><td>Mean threshold</td><td>×</td><td>√</td><td>×</td><td>√</td><td>94.9</td><td>90.7</td></tr><tr><td>Mean w/o reg.</td><td>×</td><td>×</td><td>×</td><td>√</td><td>94.3</td><td>90.2</td></tr></table>

Table 6 Compact component ablation on Qwen3-VL 8B. We report RelAcc. under matched average token retention ratios and ablate fixed budgeting, register construction, register-anchored thresholding, and Stage-I pruning.

Unless otherwise specified, we follow the standard evaluation protocol of each benchmark and report the commonly used metric in prior VLM evaluation. For multiple-choice and open-ended image benchmarks, we report accuracy or the official benchmark score. For MME, we report the total score following its original protocol. For video question-answering benchmarks, we follow the evaluation setting used by Video-LLaVA and related video VLM baselines. All results are computed using the same decoding configuration within each model family, so performance differences mainly reflect the effect of visual token pruning.

# B.1 Image Understanding Benchmarks

GQA Hudson and Manning (2019) is a large-scale benchmark for real-world visual reasoning and compositional question answering. It contains over 22 million questions generated from scene graphs, covering object attributes, spatial relationships, relational reasoning, and multi-step compositional understanding. Compared with conventional VQA benchmarks, GQA places stronger emphasis on structured reasoning and grounding, making it useful for evaluating whether token pruning preserves fine-grained relational evidence. We report accuracy on the test-dev balanced split.

MMBench (MMB) Liu et al. (2024c) is an objective benchmark for evaluating general multimodal capabilities of vision-language models. It organizes evaluation into a three-level ability hierarchy, including perception and reasoning categories, intermediate ability groups, and fine-grained ability dimensions. MMBench contains approximately 3,000 multiple-choice questions and uses CircularEval together with ChatGPT-based answer matching. MMBench-CN (MMBCN) is the Chinese-language variant of MMBench and follows the same evaluation protocol, allowing us to examine multilingual multimodal understanding.

MME (Fu et al., 2025) evaluates both perceptual and cognitive abilities of multimodal large language models. It covers 14 subtasks, including object existence, count, position, color, scene understanding, commonsense reasoning, numerical calculation, text translation, and code reasoning. MME uses manually constructed instruction-answer pairs with a concise “Yes/No” format to reduce ambiguity in answer matching. We report the total score, defined as the sum of accuracy and accuracy+ across all subtasks, with a maximum score of 2800.

POPE Li et al. (2023b) evaluates object hallucination by asking binary questions about whether a specific object exists in the image. It constructs questions under three sampling strategies: random, popular, and adversarial. This benchmark is particularly relevant for visual token pruning, since overly aggressive pruning may remove small or less salient objects and increase hallucination. We report accuracy averaged across the three sampling strategies.

ScienceQA-IMG (SQAI) Lu et al. (2022) is the image subset of the ScienceQA benchmark. It covers diverse science topics across natural science, language science, and social science, and often requires multi-step reasoning over both visual information and textual context. We use the image subset and report accuracy.

<table><tr><td>Method</td><td>GQA</td><td> $SQA^I$ </td><td> $VQA^T$ </td><td>MME</td><td>POPE</td><td>MMB</td><td>Avg</td></tr><tr><td colspan="8">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>62.4</td><td>70.7</td><td>65.2</td><td>1841</td><td>85.8</td><td>69.3</td><td>100%</td></tr><tr><td colspan="8">Retain Averaged 192 Tokens (↓ 66.7%)</td></tr><tr><td>VisionZip</td><td>60.3</td><td>70.7</td><td>63.4</td><td>1846</td><td>82.3</td><td>68.9</td><td>98.2%</td></tr><tr><td>VisionZip ‡</td><td>61.6</td><td>70.2</td><td>63.6</td><td>1804</td><td>85.5</td><td>67.2</td><td>98.4%</td></tr><tr><td>Mutual</td><td>61.4</td><td>71.0</td><td>64.7</td><td>1837</td><td>84.9</td><td>69.1</td><td>99.4%</td></tr><tr><td>CLS</td><td>61.4</td><td>70.8</td><td>64.5</td><td>1849</td><td>84.0</td><td>69.3</td><td>99.3%</td></tr><tr><td colspan="8">Retain Averaged 128 Tokens (↓ 77.8%)</td></tr><tr><td>VisionZip</td><td>58.7</td><td>70.0</td><td>61.3</td><td>1841</td><td>78.5</td><td>68.1</td><td>96.1%</td></tr><tr><td>VisionZip ‡</td><td>60.0</td><td>70.1</td><td>61.6</td><td>1810</td><td>83.2</td><td>67.0</td><td>97.0%</td></tr><tr><td>Mutual</td><td>60.7</td><td>70.6</td><td>64.3</td><td>1833</td><td>83.0</td><td>68.0</td><td>98.4%</td></tr><tr><td>CLS</td><td>60.8</td><td>70.3</td><td>64.1</td><td>1842</td><td>83.5</td><td>68.5</td><td>98.6%</td></tr><tr><td colspan="8">Retain Averaged 64 Tokens (↓ 88.9%)</td></tr><tr><td>VisionZip</td><td>55.8</td><td>70.7</td><td>59.1</td><td>1737</td><td>69.6</td><td>65.9</td><td>91.8%</td></tr><tr><td>VisionZip ‡</td><td>57.7</td><td>71.0</td><td>60.1</td><td>1779</td><td>80.0</td><td>66.3</td><td>95.1%</td></tr><tr><td>Mutual</td><td>59.8</td><td>70.6</td><td>62.1</td><td>1803</td><td>82.8</td><td>66.2</td><td>96.8%</td></tr><tr><td>CLS</td><td>59.6</td><td>70.5</td><td>61.5</td><td>1798</td><td>83.2</td><td>66.3</td><td>96.6%</td></tr></table>

Table 7 Performance comparison on Mini-Gemini under different average token budgets. We evaluate two Stage-I instantiations of OccamToken: Mutual, which uses mutual attention among visual/register tokens when no dedicated evaluator is assumed, and CLS, which uses [CLS]-based attention scoring. Both variants consistently outperform VisionZip and VisionZip‡ across compression levels, showing that register-anchored thresholding generalizes beyond LLaVA-style architectures and is not tied to a specific scoring mechanism.

TextVQA (VQAT) Singh et al. (2019) focuses on visual question answering that requires reading and reasoning about text in images. It contains 45,336 questions on 28,408 images from the OpenImages dataset. Since answers often depend on signs, labels, documents, and other text-rich regions, TextVQA is important for evaluating whether pruning preserves small but semantically crucial visual regions. We report accuracy on the validation split.

VQAv2 (VQAv2) Goyal et al. (2017) is a large-scale open-ended visual question answering benchmark. It contains 265,016 images paired with over 1.1 million questions, with 10 human-provided answers for each question. VQAv2 reduces language priors by pairing similar questions with different images that lead to different answers. We report accuracy on the test-dev split.

MMMU Yue et al. (2024) is a massive multi-discipline multimodal understanding benchmark containing 11,500 college-level questions across 30 subjects and six core disciplines. It includes visually complex inputs such as charts, diagrams, tables, maps, chemical structures, and scientific figures. MMMU is therefore a challenging testbed for evaluating whether token pruning preserves fine-grained, domain-specific evidence required for expert-level reasoning. We report accuracy on the validation split.

SEED-Bench (SEEDI) Li et al. (2023a) evaluates multimodal large language models across diverse visual understanding abilities. It covers scene understanding, instance identity, attributes, location, counting, spatial relation, visual reasoning, text understanding, and action recognition. The benchmark contains 19,000 multiple-choice questions with human annotations. We use the image subset, denoted as SEEDI, and report accuracy.

VizWiz Gurari et al. (2018) consists of over 31,000 visual questions collected from visually impaired users.

The images are captured in real-world conditions and often contain challenging artifacts such as blur, poor framing, occlusion, low resolution, or irrelevant visual content. Each question is paired with 10 crowdsourced answers, and the benchmark includes an “unanswerable” option when the image does not provide sufficient evidence. VizWiz is useful for evaluating the robustness of token pruning under naturally noisy and imperfect visual inputs. We report accuracy on the test split.

# B.2 Video Understanding Benchmarks

Video understanding introduces additional challenges for visual token pruning. Compared with static images, videos contain a much larger number of visual tokens due to the temporal dimension, and many adjacent frames may be highly redundant. At the same time, important evidence may appear only in a short temporal segment or depend on motion patterns across frames. Therefore, video benchmarks provide a natural testbed for evaluating whether OccamToken can remove temporal redundancy while preserving task-relevant spatio-temporal evidence.

TGIF-QA Jang et al. (2017) extends visual question answering to short animated GIF videos. It contains approximately 165,000 question-answer pairs and introduces several spatio-temporal reasoning tasks, including repetition counting, repeating action identification, and state transition detection, in addition to frame-level question answering. These tasks require models to capture temporal dynamics rather than relying only on static appearance. We report accuracy and GPT-evaluated scores following the Video-ChatGPT evaluation protocol.

MSVD-QA Xu et al. (2017) is built on the Microsoft Research Video Description dataset. It contains 1,970 video clips and approximately 50,500 open-ended question-answer pairs. The questions span common types such as what, who, how, when, and where, covering a wide range of visual entities, actions, and scene-level semantics. MSVD-QA is widely used for evaluating video question answering and provides a standard benchmark for measuring whether compressed video tokens preserve essential semantic information. We report accuracy and GPT-evaluated scores.

MSRVTT-QA Xu et al. (2017) is constructed from the MSR-VTT video dataset and contains 10,000 video clips with 243,000 question-answer pairs. Compared with MSVD-QA, MSRVTT-QA is larger and contains more diverse real-world video content. The questions also span what, who, how, when, and where types, requiring models to integrate visual appearance, temporal context, and language understanding. This benchmark is useful for evaluating the scalability of token pruning under diverse video inputs. We report accuracy and GPT-evaluated scores.

ActivityNet-QA Yu et al. (2019) contains 58,000 human-annotated question-answer pairs on 5,800 videos from the ActivityNet dataset. The videos usually depict complex human activities and often require longer-range temporal reasoning. The questions cover motion, spatial relationships, temporal relationships, and event-level understanding. ActivityNet-QA is particularly challenging because relevant evidence may be distributed across long video segments rather than concentrated in a single frame. We report accuracy and GPT-evaluated scores.

# C More Experiments Results

# C.1 Additional Results on Video Understanding

Video understanding. We additionally evaluate OccamToken on Video-LLaVA (Lin et al., 2024) to examine its applicability beyond image-only inputs. As shown in Table 8, OccamToken preserves 97.7% of the full-token Video-LLaVA performance, outperforming SparseVLM and VisionZip by 11.2 and 4.5 relative-accuracy points, respectively. Notably, OccamToken remains close to the full-token baseline on MSRVTT and slightly improves the score on ActivityNet. These results provide additional evidence that register-anchored pruning can remove redundant visual tokens while retaining the evidence needed for video understanding.

<table><tr><td>Method</td><td>TGIF</td><td>MSVD</td><td>MSRVTT</td><td>ActNet</td><td>Rel.</td></tr><tr><td>Video-LLaVA</td><td>47.1</td><td>69.8</td><td>56.7</td><td>43.1</td><td>100%</td></tr><tr><td>SparseVLM</td><td>44.7</td><td>68.2</td><td>31.0</td><td>42.6</td><td>86.5%</td></tr><tr><td>VisionZip</td><td>42.4</td><td>63.5</td><td>52.1</td><td>43.0</td><td>93.2%</td></tr><tr><td>Ours</td><td>44.9</td><td>66.7</td><td>56.1</td><td>43.5</td><td>97.7%</td></tr></table>

Table 8 Additional comparison on video understanding benchmarks. OccamToken preserves 97.7% relative accuracy on Video-LLaVA and outperforms prior pruning methods across the averaged relative score.

# C.2 Results on Mini-Gemini

Mini-Gemini. Table 7 reports additional results on Mini-Gemini, a VLM equipped with a dual vision encoder architecture and an additional high-resolution encoder for fine-grained visual understanding. This architecture differs substantially from the LLaVA family, and therefore provides a useful testbed for evaluating whether OccamToken is tied to a specific vision encoder design or a particular scoring token. On Mini-Gemini, we evaluate two Stage-I instantiations: the [CLS]-based scoring scheme and the mutual-scoring scheme described in Section 3.4.

The results show that both scoring schemes effectively support register-anchored dynamic thresholding. At the 192-token setting, the mutual-scoring variant achieves 99.4% relative accuracy, while the [CLS]-based variant achieves 99.3%, both preserving almost all of the full-token performance. At the 128-token setting, the [CLS] variant obtains the best average result with 98.6% relative accuracy, while the mutual-scoring variant remains close at 98.4%. Even under the more aggressive 64-token setting, both variants remain robust, achieving 96.8% and 96.6% relative accuracy for mutual scoring and [CLS] scoring, respectively. These results indicate that the effectiveness of OccamToken does not come from a specific choice of evaluator, but from the register-anchored relative comparison principle.

Compared with VisionZip and VisionZip‡, OccamToken variants consistently provide a stronger accuracy– efficiency trade-off, especially under tighter token budgets. For example, when retaining only 64 tokens on average, the mutual-scoring variant improves relative accuracy from 91.8% with VisionZip and 95.1% with VisionZip‡ to 96.8%. The gains are particularly visible on benchmarks requiring fine-grained or reliable visual evidence, such as GQA, VQAT , MME, and POPE. This suggests that register-anchored thresholding is better able to preserve task-relevant tokens when the compression ratio becomes high. Overall, the Mini-Gemini results further support the architecture generality of OccamToken: the same reference-adaptive pruning principle remains effective beyond LLaVA-style VLMs and can be instantiated with either [CLS]-based or mutual attention scoring.

# C.3 Results on Qwen-VL 7B

Qwen2-VL 7B. Table 9 answers the architecture-transfer question by evaluating OccamToken on Qwen2-VL, whose native dynamic-resolution visual encoder does not provide an explicit [CLS] token. This setting tests whether our method depends on a specific evaluator token or whether the register-anchored relative comparison is itself transferable. We instantiate Stage I with the mutual-scoring scheme in Section 3.4, while keeping the register token as the dynamic reference anchor. OccamToken achieves 96.5% relative accuracy at 22.2% retention and 93.8% at 11.1% retention, outperforming PACT by 3.7 and 8.0 points, respectively. The larger gap under tighter compression further supports our analysis: as the pruning boundary becomes more critical, comparing visual tokens against an internal register reference is more robust than relying on a static retention ratio. These results show that OccamToken is not tied to [CLS]-based scoring and can generalize to different visual encoder designs.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>MMMU</td><td> $SEED^I$ </td><td>VizWiz</td><td>RelAcc.</td></tr><tr><td colspan="10">Upper Bound (100%)</td></tr><tr><td>Baseline</td><td>62.3</td><td>79.1</td><td>2318</td><td>87.9</td><td>84.7</td><td>51.1</td><td>76.5</td><td>68.2</td><td>100%</td></tr><tr><td colspan="10">Token Retention Ratio 22.2% (↓ 77.8%)</td></tr><tr><td>PACT  $CVPR-25$ </td><td>55.8</td><td>72.2</td><td>2012</td><td>82.4</td><td>78.3</td><td>48.8</td><td>71.7</td><td>67.4</td><td>92.8%</td></tr><tr><td>OccamToken</td><td>61.3</td><td>77.1</td><td>2157</td><td>86.8</td><td>81.4</td><td>47.7</td><td>74.1</td><td>67.1</td><td>96.5%</td></tr><tr><td colspan="10">Token Retention Ratio 11.1% (↓ 88.9%)</td></tr><tr><td>PACT  $CVPR-25$ </td><td>50.1</td><td>63.1</td><td>1785</td><td>71.4</td><td>75.0</td><td>48.5</td><td>66.0</td><td>63.1</td><td>85.8%</td></tr><tr><td>OccamToken</td><td>60.2</td><td>76.7</td><td>2096</td><td>84.1</td><td>77.5</td><td>46.9</td><td>70.6</td><td>64.8</td><td>93.8%</td></tr></table>

Table 9 Performance comparison on Qwen2-VL 7B under different token retention ratios. 

<table><tr><td>Method</td><td>GQA</td><td> $SQA^I$ </td><td> $VQA^T$ </td><td>POPE</td><td>MME</td><td>MMB</td><td>RelAcc.</td></tr><tr><td colspan="8">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>63.2</td><td>72.8</td><td>61.3</td><td>85.9</td><td>1818</td><td>67.7</td><td>100%</td></tr><tr><td colspan="8">Retain 128 Tokens (↓ 77.8%)</td></tr><tr><td>VisionZip</td><td>57.9</td><td>74.0</td><td>58.7</td><td>85.2</td><td>1743</td><td>66.7</td><td>97.1%</td></tr><tr><td>OccamToken</td><td>61.2</td><td>74.1</td><td>60.1</td><td>85.5</td><td>1798</td><td>66.2</td><td>98.8%</td></tr><tr><td colspan="8">Retain 64 Tokens (↓ 88.9%)</td></tr><tr><td>VisionZip</td><td>56.2</td><td>74.4</td><td>57.4</td><td>76.0</td><td>1676</td><td>64.9</td><td>93.6%</td></tr><tr><td>OccamToken</td><td>59.3</td><td>73.9</td><td>59.6</td><td>85.4</td><td>1781</td><td>65.8</td><td>97.9%</td></tr></table>

Table 10 Performance comparison on LLaVA-v1.5 13B under different token retention budgets. OccamToken consistently outperforms VisionZip at the same average token budget, with especially large gains under the 64-token setting. These results show that register-anchored adaptive pruning remains effective for larger VLM backbones.

# C.4 Results on LLaVA-v1.5 13B

LLaVA-v1.5 13B. Table 10 reports additional results on LLaVA-v1.5 13B under two token retention budgets. This setting evaluates whether OccamToken remains effective when applied to a larger language backbone while using the standard 576-token visual prefix. Compared with the 7B setting, the 13B model provides stronger language reasoning capacity, making it a useful testbed for examining whether visual token pruning can preserve the benefits of model scaling. OccamToken consistently achieves a better accuracy–efficiency trade-off than VisionZip. At the 128-token setting, OccamToken preserves 98.8% relative accuracy, outperforming VisionZip by 1.7 percentage points while retaining the same average number of visual tokens. It improves over VisionZip on most benchmarks, including GQA, SQAI, VQAT , POPE, and MME, showing that register-anchored pruning preserves both general visual reasoning and fine-grained visual evidence. The advantage becomes more pronounced under stronger compression. When retaining only 64 tokens on average, OccamToken maintains 97.9% relative accuracy, whereas VisionZip drops to 93.6%. This corresponds to a 4.3-point gain in relative accuracy under the same token budget. The improvement is particularly large on POPE and MME: OccamToken improves POPE from 76.0 to 85.4 and MME from 1676 to 1781. These results suggest that fixed or similarity-based compression can discard visually critical evidence under tight budgets, while register-anchored relative comparison better preserves tokens that remain informative beyond the global reference. Overall, the LLaVA-v1.5 13B results further confirm that OccamToken is not restricted to 7B-scale VLMs. It remains effective for larger backbones and continues to provide robust performance under aggressive

visual token compression.

# D Broader Impact

OccamToken improves the inference efficiency of VLMs by reducing redundant visual tokens without additional training. This can lower computational cost, memory usage, and energy consumption, making multimodal models more practical for resource-constrained deployment. However, efficiency improvements may also accelerate the deployment of VLMs in sensitive scenarios. Moreover, token pruning may affect model reliability when critical visual evidence is removed. Therefore, OccamToken should be carefully evaluated before being used in safety-critical applications and should inherit the same safeguards required for the underlying VLMs.

![](images/de57be1f90406f0442724fbd1f49c3e4a33813a8b6bbd7266f425e071b755a8e.jpg)

<details>
<summary>natural_image</summary>

Silhouettes of people waving at an airplane against a cloudy sky (no text or symbols visible)
</details>

Q: Does the man that is to the left of the man seem to be standing?   
![](images/ed8789975f6cb1a6ef5e8520ae4ae4cfcbdbf80d8320d5f0e543646f9d69af74.jpg)

<details>
<summary>natural_image</summary>

People celebrating on a beach with an airplane flying overhead, no visible text or symbols
</details>

A: Yes Retain tokens: 17

Q: Does the sunny beach look bright?   
![](images/c40428339c517b5055c09923372ac49fa267665f0ce033c7f154dc334e817139.jpg)

<details>
<summary>natural_image</summary>

Silhouettes of three people on a beach with an airplane flying overhead, no visible text or symbols
</details>

A: Yes Retain tokens: 11

Q: What is the airplane flying above?   
![](images/b7e19a45987a95fb34658b79c2db747e4b379fe06c200e2769b5313e7384a131.jpg)

<details>
<summary>natural_image</summary>

Two silhouetted figures on a beach with a drone flying overhead and floating objects in the sky (no text or symbols visible)
</details>

A:Ocean Retain tokens:30

![](images/9a8a9331d848f3798b612e37ce3918cda9fbe4e0061fc827f314fec72b0fcb59.jpg)

<details>
<summary>natural_image</summary>

Child in red jacket and gray helmet standing next to a soccer ball on a stick, with green pixelated background (no text or symbols)
</details>

Q: Is the field soft and snowy?   
![](images/53f842d6d2f08477095e5d042c71546de6bca914d6847fa6a111b5b4d194600e.jpg)

<details>
<summary>natural_image</summary>

Child playing baseball on grass, wearing helmet and jacket (no visible text or symbols)
</details>

A: No Retain tokens: 12

Q: What makes up the jacket, cloth or leather?   
![](images/ab2e3605876d6193c86fabf3147ccbf62a03d7918da3b82afee94b6bc50cf653.jpg)

<details>
<summary>natural_image</summary>

Child playing a soccer ball on grass, wearing helmet and pink jacket with red squares (no text or symbols visible)
</details>

A: Cloth Retain tokens: 9

Q: Do the balls to the left of the other ball   
![](images/71de79b3409ff83dbd6da3a3c2bd1d056674ecb0c112889484c276bf2e94d28d.jpg)

<details>
<summary>natural_image</summary>

Child playing badminton on grass, wearing helmet and jacket (no visible text or symbols)
</details>

A: Yes Retain tokens: 14

![](images/1d22ffceb5b1e7485641e887525df6662d2131771810999860eb5f7494e698da.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person wearing a pixelated white and black outfit, with a sneaker partially visible (no text or symbols)
</details>

Q: What's in front of the cat?   
![](images/54d6fb715b26c5853031d5bf703ee08284593c2f166fe177bc99a4c29646a564.jpg)

<details>
<summary>natural_image</summary>

Close-up of a white cat interacting with a black sneaker, no visible text or symbols
</details>

A: Shoe Retain tokens: 20

Q: Do you see any cats?   
![](images/dd91b0590978f9e37595dff51c8f43e0cfdc3708e9aeb7cbe46362213cdb233d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a fluffy cat wearing a white sneaker, with blurred greenery in the background (no text or symbols visible)
</details>

A: Yes Retain tokens: 8

Q: What's in front of the tree?   
![](images/9f7b85a3a1cf9cda63450f78ace873ddaa5e7ea7c02e923661392f56413fd7a8.jpg)

<details>
<summary>natural_image</summary>

A gray fluffy cat wearing a white sneakers, with blurred greenery in the background (no text or symbols visible)
</details>

A: Man Retain tokens: 8   
Figure 6 Visualization of query-adaptive token budgets. Given the same image, OccamToken retains different visual tokens and different token counts for different queries. This shows that the final budget is determined by query-specific evidence demand rather than a fixed top-K rule.

![](images/b6ef800e440c81f1ec5e52a8d526246ea671e369d9b0aa9bb9769348afcfd7f8.jpg)

<details>
<summary>natural_image</summary>

Pixelated image of a white object on a pixelated background with green and gray tones (no text or symbols)
</details>

Q: Is that a small fence?   
![](images/5c01a34d5ea2a5095123cc7037d9ff5c0535df9af569aff74daa444413dff8e7.jpg)

<details>
<summary>natural_image</summary>

White table with white chairs and a small object on the table, set against a blurred outdoor background (no text or symbols visible)
</details>

A: Yes Retain tokens: 21

Q: Which kind of furniture is in front of the fence?   
![](images/657ddb58eff70a838d2a517f92272ffa851b941009aec876e86d23a6519019c6.jpg)

<details>
<summary>natural_image</summary>

Modern white table with curved legs and a small green object on top, set against a blurred background (no text or symbols visible)
</details>

A: Chair Retain tokens: 32

![](images/7ea08b04013f83d8b58f31dff4ea0e241cc73273667e80a95677e60992cf8973.jpg)

<details>
<summary>natural_image</summary>

Interior view of a room with patterned walls and furniture (no visible text or symbols)
</details>

Q: Does the knife in the knife block have black color?   
![](images/297d097e3836817aeae1d1d7c9a23fd1fb74429b92e5f579fd5dff8491e12012.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with white cabinets and appliances (no visible text or symbols)
</details>

A: No Retain tokens: 9

Q: Are there refrigerators to the left of the stove?   
![](images/948050796fdb33e898e6edf9f9957d43f2a158526a49a05f8f60b273aca191e0.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with white cabinets and kitchenware (no visible text or symbols)
</details>

A: Yes Retain tokens: 16

![](images/376a6c4cd980037d1f54d3c5183abd781a25647a29ef9e3baa837876d052b67c.jpg)

<details>
<summary>natural_image</summary>

Two people in white protective suits walking outdoors, one wearing a helmet and the other in a suit (no visible text or symbols)
</details>

Q: How is the animal that is in the backpack called?   
![](images/1a6ed10d8fd5c6cf58ac6ab67acaa4c778092bde2d891c0772798855d30c2e59.jpg)

<details>
<summary>natural_image</summary>

Street scene with pedestrians and bicycles, no visible text or symbols
</details>

A: Dog Retain tokens: 26

Q: Does the backpack appear to be clean and blue?   
![](images/a543cfc8e33b0f012fe6281e20ea2431e6220423b65e8f754bd28aba6bdcdade.jpg)

<details>
<summary>natural_image</summary>

Street scene with cyclists and pedestrians, no visible text or symbols
</details>

A: Yes Retain tokens: 16

![](images/d37c2aa74f6906699d98846edae9a8ba4fcfc7096b64a2e8ec11b29f26c6bfa9.jpg)

<details>
<summary>natural_image</summary>

Woman in a light blue dress posing outdoors with blurred background (no visible text or symbols)
</details>

Q: Is the building in front of the trees that are not short?   
![](images/ecf93a7338617e31cd01e5faccc9d6dd03c66f1f08f44bea51a2adfeee0d19e9.jpg)

<details>
<summary>natural_image</summary>

Person walking on a beach wearing casual clothing (no visible text or symbols)
</details>

A: Yes Retain tokens: 30

Q: What color is the shirt?   
![](images/80126a4d83fa55c17a460f11a7644d345ce911ceaf5fde58741b2e427b69d30e.jpg)

<details>
<summary>natural_image</summary>

Person standing on a sandy beach with blurred background of trees and a yurt (no visible text or symbols)
</details>

A: Green Retain tokens: 22   
Figure 7 Additional visualization of query-adaptive token budgets. Each row shows the same image with different textual queries and the visual tokens retained by OccamToken. Although the image content is shared, the retained regions and token counts vary across questions, reflecting different evidence demands for object existence, attributes, categories, and spatial relations. This shows that OccamToken does not impose a fixed top-K budget, but selects query-dependent visual evidence after image-level redundancy pruning. The examples qualitatively demonstrate that Stage-II pruning adapts the final token budget to the current image–query pair.

![](images/4cffcd02bafabcb1bd9ae76eabf7906d5802630bad25b66f99b79a7394615fa2.jpg)  
Figure 8 Visualization of Stage-I image-adaptive redundancy pruning. We visualize the visual tokens retained after the first pruning stage across diverse images. Stage-I removes image-level redundant tokens before the LLM while preserving tokens distributed over informative objects, foreground regions, and scene context. The retained token counts vary across images, showing that OccamToken adapts the intermediate visual budget to image-specific redundancy rather than enforcing a fixed top-K allocation.