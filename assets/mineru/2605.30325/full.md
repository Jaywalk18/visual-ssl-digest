# Veda: Scalable Video Diffusion via Distilled Sparse Attention

Shihao Han 1 2 \* Hao Yang 1 † Xiaofeng Mei 1 Xinting Hu 3 Yi Jiang 1 Xiaojuan Qi 2

Full Attention (FA3)  Sample Time: 19.4 minutes

![](images/0c172ca09a84051ae584c6e85c32b7cea3c897ab25335e2fda0ca2e9b05748ce.jpg)

Veda (Ours, 95% Sparsity)  Sample T s Sparsity)|Sample Ti ime: 3.8 minute

![](images/82dea7f0ff3640e8ef94326125f997d941349e863f846e43d185a62d02336af3.jpg)

![](images/e5ee8d2f583626215a2cb9ececf80ee279c5c3bb3559a66eb78893456fdff3dd.jpg)

![](images/67664e829ec8b04949b14e7509126c32c9d50698dd3a4a47af685fc8bcbd1707.jpg)

![](images/b2022eb289727cd71316509f919809620103fca3301fa384066f24d490e20045.jpg)

![](images/581203ce98ae2d9ea73f3f29e401ddc51236de6ff54a69554815c96ad60bd0f6.jpg)

![](images/d098c8bbfa439add4ddc242e92d2826efc6eaf9985031f4969dbdc13be74d357.jpg)

![](images/205fe14bcac58ea2e99ab2fd9a74b991c60ea7ad627e9306ce4a38e83cdb1332.jpg)

![](images/7945f834c0accc67b310dcbe1ea1f51ae70faa40c7a9c20ab0ff291d1a30a7a8.jpg)

![](images/7fabc837687461f2b331638a1787f7dab7e2ce5d9e80b4086ec746205a8046b9.jpg)

![](images/8ca4f4333fbba772999b1a06b242f908bba9391de6afeb2003c6591e67df3146.jpg)

<details>
<summary>line</summary>

| Sequence Length | FA3    | Veda (Ours) |
| --------------- | ------ | ----------- |
| 50K             | 65.6ms | 25.5ms      |
| 122K            | 315.3ms| 78.3ms      |
| 245K            | 1576.5ms| 309.1ms     |
</details>

Figure 1. Veda achieves a 5.1× end-to-end GPU speedup for high-resolution, long-video generation. Evaluated on Waver-T2V-12B (720P / 241 frames), Veda at 95% sparsity reduces the sample time from 19.4 to 3.8 minutes without compromising visual quality.

# Abstract

Scaling Diffusion Transformers to generate highresolution, long videos is constrained by the quadratic cost of self-attention, and existing sparse attention methods degrade under high sparsity. We show empirically that generation quality is determined not by the sparsity ratio itself, but by how well the sparse mask aligns with the tilewise geometry of full attention. Based on this insight, we propose Veda, a distilled sparse attention framework that formulates tile selection as an explicit reconstruction problem from full attention. Veda integrates statistics-aware tile scoring with head-aware tiling to reduce estimation error and structural mismatch, enabling aggressive sparsity. A hardware-efficient tile-skipping kernel converts theoretical sparsity into practical wall-clock speedups. Experiments on large video diffusion models, including Waver and Wan2.1, demonstrate substantial acceleration with no noticeable degradation in generation quality. To generate 720P 10-second videos on Waver-T2V-12B, Veda achieves a 5.1 end-to-end speedup and a 10.5 self-attention speedup, reducing attention overhead from 92% to 50%. Notably, the gains in-

1ByteDance Inc. 2The University of Hong Kong 3University of Science and Technology of China \*Work done during internship at ByteDance. †Project Lead. Correspondence to: Xinting Hu <xinting@ustc.edu.cn>, Xiaojuan Qi <xjqi@hku.hk>.

Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

crease with sequence length, indicating that Veda scales favorably with spatiotemporal resolution across models.

# 1. Introduction

Diffusion Transformers (DiTs) (Peebles & Xie, 2023) have emerged as the leading architecture for high-fidelity video synthesis (Lin et al., 2024; StepVideoTeam, 2025; Wan-Team, 2025; Team et al., 2025; Yang et al., 2025; Hunyuan-Team, 2025). Nevertheless, their scalability is bottlenecked by the quadratic computational and memory complexity of self-attention (Vaswani et al., 2017) with respect to the spatiotemporal token sequence length. Sparse attention offers a natural path toward alleviating this bottleneck by restricting attention computation to a subset of token pairs. In practice, however, sparse attention must be realized at the granularity of tiles rather than individual tokens to align with modern GPU hardware, which performs attention through blockwise matrix multiplications (Markidis et al., 2018; Zhang et al., 2025c). This reformulates sparse attention as a tile selection problem: tokens are grouped into tiles, and each query tile attends to only a small number of key tiles, producing a tile-wise mask that can be directly materialized by a tile-skipping kernel for efficient execution.

However, translating sparsity into wall-clock acceleration often comes at the cost of generation quality. To preserve spatiotemporal structure, existing approaches construct tile masks with two main paradigms. Methods such as SVG (Xi et al., 2025) and STA (Zhang et al., 2025c) leverage inductive biases of pretrained models to define candidate spatiotemporal masks and employ online or offline search strategies to select the most effective pre-defined patterns (see Fig. 2, SVG). While these methods are simple and hardwarefriendly, they lack adaptability to the highly dynamic and head-specific attention structures learned by DiTs.

![](images/77ba45a36db5aad7499eb6f465b7896ddc521ca1a959543507b55ae2508f2b84.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["QKV Projection"]
    B --> C["Router"]
    B --> D["Reorder"]
    C --> E["Sparse Attention"]
    D --> E
    E --> F["Reorder"]
    F --> G["Output"]
    style A fill:#f9f,stroke:#333
    style G fill:#ccf,stroke:#333
```
</details>

![](images/292d68b2ce68e4d04a2f659e5f9d878e017604630267ac8fde02b070dcab6aa0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Gate Projection"]
    A --> C["QKV Projection"]
    B --> D["Tile & Pad"]
    C --> D
    D --> E["×"]
    E --> F["Block Attention"]
    F --> G["Top-k Selection"]
    G --> H["Sparse Attention"]
    H --> I["⊕"]
    I --> J["Untile & Unpad"]
    J --> K["Output"]
    E --> L["Average Pooling"]
    L --> M["Block Attention"]
    M --> N["Top-k Selection"]
    N --> H
    style E fill:#f9f,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#cff,stroke:#333
    style J fill:#ffc,stroke:#333
```
</details>

![](images/5b1de8aac91fbfc280474a423528c904a783eb127eb07f31a0600a21e7419b80.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["QKV Projection"]
    B --> C["Tile & Pad"]
    C --> D["Attention"]
    C --> E["Triplet Pooling"]
    D --> F["Max Pooling"]
    F --> G["L_distill"]
    E --> H["QK Projection"]
    H --> I["Block Attention"]
    I --> J["Top-k Selection"]
    J --> K["Sparse Attention"]
    K --> L["Untile & Unpad"]
    L --> M["Output"]
    style A fill:#f9f,stroke:#333
    style M fill:#bbf,stroke:#333
    note1["Training-Time Only"] -.-> C
    note2["Offline Search"] -.-> C
    note3["Gradient Flow"] -.-> I
    note4["Full Sequence"] -.-> K
    note5["Pooled Sequence"] -.-> K
    note6["Pooled Sequence"] -.-> K
```
</details>

Figure 2. Architectural comparison of sparse attention methods. We compare the proposed Veda (ours) with representative methods that use (i) static sparse patterns (SVG (Xi et al., 2025)) and (ii) dynamically learned sparse masks (VSA (Zhang et al., 2025b)). Solid lines indicate the inference data flow, while dashed lines denote auxiliary components used only during training.

In contrast, VMOBA (Wu et al., 2025) and VSA (Zhang et al., 2025b) adopt dynamic tile selection by estimating tile importance from compressed representations, such as pooled features or low-rank approximations, and ranking tiles to synthesize dynamic tile-selection masks. Although more flexible (see Fig. 2, VSA), these approaches suffer from inaccurate estimation: tile importance is learned only implicitly through the diffusion objective, without explicit supervision to preserve the structural geometry of full attention. Moreover, their reliance on coarse statistics, such as mean pooling, fails to capture salient signal peaks within tiles, causing the estimated tile scores to misrepresent true query-key correlations. As a result, both paradigms exhibit pronounced quality degradation (Fig. 3) when sparsity is pushed beyond moderate levels, producing structured artifacts including spatial warping, water-ripple patterns, and temporal flickering.

To understand what limits sparse attention in video diffusion, we conduct an empirical study on the relationship between sparsity and generation quality (see Fig. 4). We construct an “Oracle” sparse mask by max-pooling the fullattention matrix into a tile-level score map and retaining the highest-response tiles, thereby inducing a reference perquery ranking of key tiles. Models using this Oracle mask preserve generation quality even under very high sparsity, demonstrating that sparsity itself is not the primary cause of degradation (Fig. 4). Instead, we find that mask quality, specifically, the degree to which the sparse mask aligns with the tile-wise structure of full attention, dominates performance.

Consistent with this observation, existing methods fail for two complementary reasons: methods with pre-defined sparse patterns suffer from structural mismatch with headspecific attention geometry, while dynamic methods incur estimation error due to implicit supervision and insufficient tile statistics. Furthermore, we observe substantial heterogeneity across attention heads and diffusion timesteps, indicating that a single, uniform tiling strategy is inherently suboptimal. This suggests that sparse attention must control two coupled factors: how tokens are grouped into tile, and which tiles are selected.

Guided by these findings, we propose Veda (Fig. 2, Veda), a distilled sparse attention framework that preserves the tile-wise structure and ranking of full attention even under extreme sparsity. Unlike prior dynamic methods in which tile selection emerges implicitly from the diffusion objective, Veda learns tile selection as an explicit target. Specifically, Veda employs a lightweight estimator to distill tile-level attention scores from a full-attention backbone. To reduce estimation error, we enrich tile representations with extrema-aware statistics that preserve salient peaks beyond simple mean pooling. To further mitigate structural mismatch, Veda introduces a Head-Aware Tiling strategy: under a fixed hardware tile budget, heads that primarily capture local spatial interactions and those that model long-range temporal dependencies are assigned distinct spatiotemporal factorizations of tiles. Together, these components enable aggressive pruning of attention tiles while maintaining alignment with the intrinsic structure of full attention. Finally, to translate theoretical FLOPs reduction into practical end-to-end acceleration, we implement a tile-skipping sparse attention kernel. This kernel materializes the predicted sparsity through efficient tile skipping and reaches approximately 80% of the MFU achieved by FlashAttention-3 (Shah et al., 2024) while executing sparse attention, ensuring that sparsity yields tangible latency gains rather than overhead-dominated speedups.

We conduct extensive experiments on the Waver and Wan2.1 video diffusion models and consistently observe substantial acceleration with no noticeable degradation in generation quality. Across both backbones, Veda maintains stable visual fidelity and temporal coherence even at high sparsity, indicating that the speed gains do not come at the cost of structural artifacts. On Waver-T2V-12B generating 720P 10-second videos, Veda achieves a 5.1× end-to-end speedup and a 10.5 self-attention speedup, reducing attention overhead from 92% to 50%. On Wan2.1-T2V-14B 720P 5- second, Veda further delivers a 2.63× end-to-end speedup with a 7.08 acceleration in self-attention. Notably, the speedup increases with sequence length, demonstrating that Veda scales favorably with spatiotemporal resolution across models (Fig. 8).

# 2. Related Work

Sparse Attention in Language Modeling Sparse attention was first developed and stress-tested at scale in language modeling (Touvron et al., 2023; QwenTeam, 2025; DeepSeekTeam, 2025a). Early works introduced static sparsity constraints like Sliding Window Attention (Hassani et al., 2023; Fu et al., 2024) and StreamingLLM (Xiao et al., 2023) that restrict attention to local neighborhoods or specific sinks to maintain efficiency. More recent approaches have evolved toward dynamic sparsity, utilizing either heuristic approximations (e.g., MMInference (Li et al., 2025), MoBA (Lu et al., 2025), NSA (Yuan et al., 2025)) or predictive mechanisms that estimate significant attention tiles (e.g., SeerAttention (Gao et al., 2025), DSA (DeepSeek-Team, 2025b)). Our approach is closely related to this predictive paradigm. However, direct transfer to video generation is non-trivial: tightly coupled spatiotemporal dependencies can amplify small sparsity errors into structured distortions and temporal flicker, motivating video-specific sparsity designs.

Sparsity Mechanisms for Video Synthesis Sparsifying the attention matrix in Video DiTs offers a pathway to efficient long-video generation. Works such as SVG (Xi et al., 2025), Sparse-vDiT (Chen et al., 2025), VORTA (Sun et al., 2025) and STA (Zhang et al., 2025c) rely on static sparsity patterns, which are either manually defined or derived from offline statistics. To produce content-adaptive sparsity at runtime, methods like VSA (Zhang et al., 2025b) and VMOBA (Wu et al., 2025) rely on the backbone network to implicitly learn the optimal sparse structure alongside the video generation task. Despite wall-clock gains, the learned selection can drift away from the tile structure implicitly relied upon by the pretrained full-attention model, leading to visible artifacts under high sparsity. This motivates our key stance: learning where to attend benefits from explicit objectives that preserve the tile-wise structure of full attention.

Efficiency for Video DiTs Beyond Sparse Attention Beyond sparse attention, research on accelerating Video DiTs generally follows complementary paradigms. Methods such as PAB (Zhao et al., 2025) and TeaCache (Fan et al., 2025) reuse intermediate activations across denoising steps, but the locality assumption can break in few-step regimes where large timestep jumps induce substantial feature disparity. In parallel, sampling distillation methods such as CausVid (Yin et al., 2025) and rCM (Zheng et al., 2025) reduce the number of model evaluations by compressing the diffusion trajectory, yet the per-step spatial cost remains quadratic $( O ( N ^ { 2 } ) )$ ). Our sparse attention framework is orthogonal to both directions and can be combined with them to maximize end-to-end acceleration.

# 3. What Matters for a Good Sparse Model?

# 3.1. Preliminary

Modern video diffusion DiTs (Peebles & Xie, 2023) model global dependencies over spatiotemporal tokens, yet attention quickly becomes the primary computation and memory bottleneck at high resolution and long duration. Specifically, a video latent of shape $( T , H , W )$ is flattened into a token sequence of length $N = T H W$ . For a single attention head, let Q, K, $\bar { \mathbf { V } } \in \mathbb { R } ^ { N \times d }$ denote the query, key, and value matrices. Attention (Vaswani et al., 2017) is computed as

$$
\mathbf {A} = \operatorname{Softmax} \left(\frac {\mathbf {Q K} ^ {\top}}{\sqrt {d}}\right), \quad \mathbf {O} = \mathbf {A V}, \tag {1}
$$

which incurs $O ( N ^ { 2 } )$ computation and memory complexity. To mitigate the quadratic cost, sparse attention introduces a mask to restrict computation to a subset of token pairs. Following VSA (Zhang et al., 2025b), this can be written as A = Softmax $( \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { d } + \mathbf { M } )$ , where full attention corresponds to $\mathbf M = \mathbf 0$ , and sparsity is induced by setting most entries of M to .

Modern hardware accelerators $( \mathrm { e . g . }$ , GPUs with Tensor Cores) are optimized for tile-level data movement and computation. Accordingly, we formulate sparse attention at the tile granularity. Concretely, N tokens are grouped into $N _ { T }$ tiles of size B, yielding tiled tensors $\widetilde { \mathbf { Q } } , \widetilde { \mathbf { K } } , \widetilde { \mathbf { V } } \in \mathbb { R } ^ { N _ { T } \times B \times d }$ $\widetilde { \mathbf { M } } _ { i j } \in \{ 0 , 1 \}$ denotes a binary tile-selection mask for the $( i , j )$ tile pair; selected tile pairs are evaluated by the sparse kernel, while unselected pairs are equivalently assigned additive mask value in the softmax. Under a fixed Top-k budget, each query tile $\widetilde { \mathbf { Q } } _ { i }$ attends to k key tiles and masks out the rest, yielding a tile mask $\widetilde { \mathbf { M } }$ that satisfies

$$
\left| \left\{j \mid \widetilde {\mathbf {M}} _ {i j} = 1 \right\} \right| = k, \quad \forall i \in \{1, \dots , N _ {T} \}.
$$

The sparse attention for query tile $\widetilde { \mathbf { Q } } _ { i }$ is computed as:

$$
\hat {\mathbf {K}} _ {i} = \mathrm{Concat} _ {j: \widetilde {\mathbf {M}} _ {i j} = 1} (\widetilde {\mathbf {K}} _ {j}), \quad \hat {\mathbf {V}} _ {i} = \mathrm{Concat} _ {j: \widetilde {\mathbf {M}} _ {i j} = 1} (\widetilde {\mathbf {V}} _ {j}),
$$

$$
\mathbf {O} _ {i} = \operatorname{Softmax} \left(\frac {\widetilde {\mathbf {Q}} _ {i} \hat {\mathbf {K}} _ {i} ^ {\top}}{\sqrt {d}}\right) \hat {\mathbf {V}} _ {i}. \tag {2}
$$

This enables a tile-skipping kernel to bypass the masked tiles for practical wall-clock speedups.

# 3.2. Empirical Study

We first consider two common sparse attention designs that follow the pipeline in 3.1: (i) Static-pattern methods that instantiate a tile-structured mask M from pre-defined spatiotemporal layouts; and (ii) Dynamic methods that construct tile representations via tile pooling and predict tile importance for per-query Top-k selection.

![](images/435678de82047b1d5ba4f95d1fdb7a0e27aba741c78ea7c9ee44c5d071a220e7.jpg)

<details>
<summary>text_image</summary>

Frame t
Frame t + 1
Frame t + 2
Frame t + 3
</details>

Figure 3. Sparsity-Induced Artifacts. At 90% sparsity, samples generated with average pooling exhibit “water-ripple” distortions and stochastic noise, which are distinct from standard hallucinations.1

Observation 1: structural artifacts emerge at high sparsity. At high sparsity $( \geq 9 0 \% )$ , both static and dynamic baselines exhibit a distinct class of structural artifacts that are different from semantic hallucinations of the base model. Typical failures include water-ripple patterns, local geometric distortions, and temporal flickering across frames (Fig. 3).

Observation 2: mask quality dominates sparsity ratio. To disentangle the effect of the sparsity ratio from mask

![](images/2328fc52897634ac9ed2aedb17928159fd404808d06076bafe3f3768a3411578.jpg)

<details>
<summary>text_image</summary>

Full Attention
Oracle Mask
Average Pooling
Veda (Ours)
</details>

Figure 4. Impact of Mask Precision. At a fixed sparsity level (90%), generation quality is governed by the accuracy of tile selection. The Oracle Mask, selected from full-attention Top-k scores, preserves high-fidelity structure and serves as an upper bound. In contrast, the Average Pooling mask used in VSA (Zhang et al., 2025b) misestimates tile importance and causes severe artifacts. Veda distills the oracle selection into an efficient sparse mask, recovering oracle-like quality under the same budget.

quality, we perform a controlled intervention at a fixed sparsity level (e.g., 90%). Using the “optimal” mask derived from full attention scores yields substantially better generations than a pooling-based mask under the same sparsity budget (Fig. 4). This gap indicates that the bottleneck is not the sparsity ratio itself, but whether the tile mask preserves the tile-wise structure of full attention.

![](images/c752c9eaf58a591756413eab4f1db8d7bb1930369b925161b7733f448ad4bbcf.jpg)

<details>
<summary>scatter</summary>

| Mask Type         | Recall   |
| ----------------- | -------- |
| Oracle Mask       | 34.2%    |
| Average Pooling Mask | 34.2%    |
| Veda Mask         | 66.4%    |
</details>

Figure 5. Tile recall analysis against full attention oracle. Visualization of tile-level selection agreement with the full attention oracle mask (Top-k tiles by aggregated attention scores). Orange indicates true positives (correctly selected tiles), red denotes false positives (incorrectly selected), and blue shows false negatives (missed tiles).

Metric: tile recall w.r.t. full attention. To quantify tilewise alignment, we define a simple recall metric against full attention. For each query tile i, let $\widetilde { \mathbf { M } } _ { i } ^ { \mathrm { f u } }$ be the mask of Topk key tiles under full attention (computed by aggregating token-level scores into tile scores), and $\widetilde { \mathbf { M } } _ { i } ^ { \mathrm { s p } }$ be the mask of selected Top-k tiles of a sparse method. Define the selected key-tile sets as $S _ { i } ^ { \mathrm { f u } } = \{ j | \widetilde { \textbf { M } } _ { i j } ^ { \mathrm { f u } } = 1 \}$ and $S _ { i } ^ { \mathrm { s p } } = \{ j ^ { } \ |$ $\widetilde { \mathbf { M } } _ { i j } ^ { \mathrm { s p } } = 1 \}$ . We report tile recall:

$$
\text { Recall } @ k = \frac {1}{N _ {T}} \sum_ {i = 1} ^ {N _ {T}} \frac {\left| \mathcal {S} _ {i} ^ {\mathrm{sp}} \cap \mathcal {S} _ {i} ^ {\mathrm{fu}} \right|}{k}. \tag {3}
$$

Across sparsity levels, higher tile recall correlates strongly with fewer structural artifacts, making it a reliable indicator of stability under extreme sparsity (Fig. 5).

![](images/efdc2b6052c3c6d56b6787dbb0ae42c846c5ad7a5e063462b4448a1a2c1159b5.jpg)

<details>
<summary>heatmap</summary>

| Layer | Model | Recall |
|-------|-------|--------|
| Before Padding | Key Tlines | 20.5% |
| After Padding - Tiling [2,4,16] | Key Tlines | 31.2% |
| After Padding - Tiling [4,4,8] | Key Tlines | 80.9% |
| After Padding - Tiling [8,16,1] | Key Tlines | 57.4% |
| After Padding - Tiling [2,4,16] | Key Tlines | 48.7% |
| After Padding - Tiling [4,4,8] | Key Tlines | 63.6% |
| After Padding - Tiling [8,16,1] | Key Tlines | 20.5% |
</details>

Figure 6. Diversity of Attention Patterns. Distinct attention heads exhibit varying spatial and temporal footprints that evolve across diffusion timesteps, indicating the need for head-specific adaptation.

Observation 3: head-wise heterogeneity challenges uniform tile designs. Finally, inspecting attention patterns reveals pronounced head-wise diversity across layers and diffusion timesteps (Fig. 6). A single, uniform tiling strategy can under-aggregate critical correlations for some heads while over-aggregating others, which degrades tile recall under high sparsity and motivates head-aware tile designs.

# 4. Veda

We introduce Veda, a sparse attention framework designed to distill the intrinsic structure and ranking of full attention. As a lightweight, architecturally modular component, Veda actively aligns tile selection with the full-attention backbone even under extreme sparsity. The overall architecture is shown in Fig. 2, and this section is organized as follows. In Sec. 4.1, Veda casts tile selection as an explicit alignment objective: a statistics-aware score estimator is supervised to recover full-attention query-key tile scores. Sec. 4.2 addresses head-wise heterogeneity with Head-Aware Tiling, assigning head-dependent spatiotemporal tile configurations to reduce structural mismatch. Finally, Sec. 4.3 presents a hardware-efficient tile-skipping kernel that materializes the predicted masks into practical end-to-end acceleration.

# 4.1. Distilled Tile Scoring for Sparse Attention

Veda optimizes a binary tile mask $\widetilde { \bf M }$ under a fixed Top-k budget to preserve the tile-wise structure and ranking induced by full attention. Rather than relying on the diffusion objective to shape sparsity implicitly, we cast tile selection as an explicit alignment problem with supervision derived from the full-attention backbone.

Target: Full-Attention Score Construction. To guide the sparse selection, we first derive a reference tile ranking from the full-attention backbone. Given the query and key features Q $\mathbf { \Psi } _ { : , \mathbf { K } } \in \mathbb { R } ^ { N \times d }$ , the full attention map is $\mathbf { A } ^ { * } =$ Softmax $( \mathbf { Q K } ^ { \top } / \sqrt { d } )$ . To map this token-level density to tile-level importance, we apply a max-pooling operation over the corresponding query-key tile regions:

$$
\mathbf {S} _ {i j} ^ {\mathrm{tgt}} = \max _ {(u, v) \in \operatorname{Tile} (i, j)} \mathbf {A} _ {u v} ^ {*}. \tag {4}
$$

We opt for max-pooling rather than averaging because attention distributions are typically sparse and peaky; averaging tends to dilute salient high-frequency signals with background noise, whereas max-pooling preserves the existence of critical dependencies.

Statistic-Aware Estimator. We employ a lightweight estimator $\mathcal { E } _ { \phi }$ to reconstruct the target scores using compressed tile representations. First, for each query/key tile, we construct a TripPool descriptor by concatenating Avg, Max, Min triplet statistics:

$$
\operatorname{TripPool} [ \cdot ] = \operatorname{Avg} [ \cdot ] \oplus \operatorname{Max} [ \cdot ] \oplus \operatorname{Min} [ \cdot ], \tag {5}
$$

where ⊕ denotes the concatenation. To capture heterogeneous dependency patterns, the estimator employs headspecific MLP projections $\phi _ { q }$ and $\phi _ { k }$ . For a given attention head, these projections map the statistics into a shared latent space. The predicted score Spredij ${ \bf S } _ { i j } ^ { \mathrm { p r e d } }$ between query tile $\widetilde { \mathbf { Q } } _ { i }$ and key tile $\widetilde { \mathbf { K } } _ { j }$ is then computed as:

$$
\mathbf {S} _ {i j} ^ {\text { pred }} = \frac {\phi_ {q} (\text { TripPool } [ \widetilde {\mathbf {Q}} _ {i} ]) \cdot \phi_ {k} (\text { TripPool } [ \widetilde {\mathbf {K}} _ {j} ]) ^ {\top}}{\sqrt {d ^ {\prime}}}, \tag {6}
$$

where $d ^ { \prime }$ is the estimator latent dimension. Note that while we omit the head index for brevity, distinct projection weights are learned for each head.

Optimization. We align the predicted tile-score distribution to the full-attention reference via a row-wise distillation objective. Concretely, we apply row-wise normalization on $\mathbf { S } ^ { \mathrm { t g t } }$ and Softmax on Spred over key tiles to obtain per-query attention weights $\mathbf { A } ^ { \mathrm { t g t } }$ and $\mathbf { A } ^ { \mathrm { p r e d } }$ , and minimize

$$
\mathcal {L} _ {\text { distill }} = \mathcal {D} _ {\mathrm{KL}} \left(\mathbf {A} ^ {\mathrm{tgt}} \| \mathbf {A} ^ {\mathrm{pred}}\right), \tag {7}
$$

averaged over query tiles and heads.

By applying per-query Top-k selection on Stgt and Spred, $\mathbf { S } ^ { \mathrm { t g t } }$ $\mathbf { S } ^ { \mathrm { p r e d } }$ we retain the k highest-scoring key tiles for each query tile and mask out the rest, constructing the oracle and predicted binary tile masks $\widetilde { \mathbf { M } } ^ { * }$ and $\widetilde { \mathbf { M } } .$ . The resulting mask is materialized by a tile-skipping sparse attention kernel to compute sparse attention outputs. The backbone is trained under this sparse execution with the standard diffusion denoising objective ${ \mathcal { L } } _ { \mathrm { d i f f } } .$ , while ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ provides explicit supervision to improve tile-score estimation and the induced Top-k selection.

Crucially, to ensure the estimator actively adapts to the backbone without perturbing the pre-trained generative manifold, we apply a stop-gradient (sg) operation on the backbone features feeding into the estimator. This decouples mask learning from feature learning, ensuring stable convergence. In our experiments, we observed that allowing the gradient to propagate back into the base model leads to a noticeable degradation in generation quality. This empirical finding indicates that implicitly coupling sparse mask learning with the diffusion objective is detrimental; forcing the generative backbone itself to learn how to perform sparse attention fundamentally disrupts its pre-trained representation capabilities.

# 4.2. Head-aware Tiling Search

Attention heads in Video DiTs exhibit pronounced heterogeneity in their spatiotemporal dependency patterns, so a single uniform tiling can be suboptimal under high sparsity. Veda therefore assigns a per-layer-per-head tiling configuration $\pi _ { l , h } = ( p _ { t } , p _ { h } , p _ { w } )$ : For each head h at layer l, $\pi _ { l , h }$ defines how the N tokens are grouped into $N _ { T }$ tiles over temporal, spatial-height, and spatial-width dimensions. To remain hardware-efficient, we restrict π to factorizations of the hardware tile size B (e.g., B=128), and the configuration set Ω is:

$$
\Omega = \{(p _ {t}, p _ {h}, p _ {w}) \in \mathbb {N} ^ {3} \mid p _ {t} p _ {h} p _ {w} = B \}, \tag {8}
$$

We select $\pi _ { l , h }$ offline on a calibration set by directly minimizing the approximation error of full-attention output ${ \bf O } ^ { \mathrm { f u } }$ after applying sparse attention. For each candidate $\pi \in \Omega$ and calibration sample $x \sim \mathcal { D } _ { \mathrm { c a l } }$ , we first tile the headspecific tokens according to π and derive a reference Top-k mask $\widetilde { \mathbf { M } } ^ { * } ( x ; \pi )$ from the full-attention map, following the score pooling and per-query Top-k selection in Sec. 4.1. We then apply $\widetilde { \mathbf { M } } ^ { * } ( x ; \pi )$ to execute tile-sparse attention and obtain the corresponding output $\mathbf { O } _ { l , h } ^ { \mathrm { s p } } ( x ; \pi )$ after mapping tokens back to the original order, while $\mathbf { O } _ { l , h } ^ { \mathrm { f u } } ( x )$ denotes the full-attention output of the same head. The optimal configuration is chosen as

$$
\pi_ {l, h} ^ {*} = \arg \min _ {\pi \in \Omega} \mathbb {E} _ {x \sim \mathcal {D} _ {\mathrm{cal}}} \left[ \left\| \mathbf {O} _ {l, h} ^ {\mathrm{fu}} (x) - \mathbf {O} _ {l, h} ^ {\mathrm{sp}} (x; \pi) \right\| _ {F} ^ {2} \right], \tag {9}
$$

which favors tilings that preserve the full-attention output most faithfully under the fixed Top-k budget. Algorithm 1 summarizes the procedure.

# 4.3. Hardware Implementation

Hardware-Efficient Tile-sparse Kernel. To translate the theoretical reduction in FLOPs into tangible wall-clock acceleration, we implement a highly optimized tile-skipping attention kernel leveraging the ThunderKittens DSL (Spector et al., 2024). We address the irregular memory access patterns introduced by sparse attention by exploiting the asynchronous Tensor Memory Access (TMA) and Warp Specialization features of the NVIDIA Hopper (Markidis et al., 2018). Our kernel decouples data movement from computation via a producer-consumer paradigm. The producer warps orchestrate TMA instructions to fetch only the selected non-contiguous key/value tiles from global memory into a circular shared memory buffer. Simultaneously, consumer warps execute tensor core operations (WGMMA) on the buffered data. This design effectively hides the latency of sparse memory gathering behind dense matrix math. Consequently, our kernel reaches approximately 80% of the MFU achieved by the highly optimized FlashAttention-3 (Shah et al., 2024) on 480P, 81-frame resolution sequences (sequence length $L \approx 3 4 \mathrm { K } )$ .

Efficient Ground-Truth Heatmap Generation. Training the sparse predictor requires a ground-truth tile-wise heatmap derived from the full-attention distribution A. Generating exact pooled scores without storing A is non-trivial because softmax normalization couples all keys within each query row. We address this with a TileLang (Wang et al., 2025) kernel that computes pooled tile scores in two passes. The first pass performs tile-wise $\mathbf { Q } \mathbf { K } ^ { \top }$ in SRAM and writes per-tile unnormalized maxima together with running row statistics to global memory. The second pass finalizes the row statistics and normalizes the cached maxima to recover exact tile-wise scores. This implementation reaches  0.9 FlashAttention-3 throughput while producing the pooled supervision, and its tile-wise independence further supports partial query processing by supervising only a random subset of query tiles. Such sparse supervision dramatically reduces the supervision overhead and accelerates the overall training process, all without substantially compromising the final model performance.

# 5. Experiments

Implementation Details. We evaluate Veda using the Waver-T2V (1B/12B) (Zhang et al., 2025d) and Wan2.1- T2V (1.3B/14B) (WanTeam, 2025) architectures. To assess scalability, we conduct experiments across varying resolutions and token counts: 480P (81 frames, 34,020 tokens) for the 1B/1.3B models; 720P (81 frames, 75,600 tokens) for Wan2.1-14B; and 720P (241 frames, 219,600 tokens) for Waver-T2V-12B. All models are trained on an internal dataset encompassing a broad spectrum of general scenes. For Veda, we employ a two-stage training protocol. In the first stage, the backbone is frozen while the projector is optimized for 1,000 steps using Xavier uniform initialization and a learning rate of $6 \times 1 0 ^ { - 4 }$ to align the sparse predictor. In the second stage, we unfreeze all parameters and perform sparse fine-tuning at the target sparsity level using a learning rate of $6 \times 1 0 ^ { - 5 }$ for backbone and $6 \times 1 0 ^ { - 4 }$ for sparse predictor. Unlike prior dynamic sparse methods that require a handcrafted, slow sparsity warm-up to prevent

![](images/ff751a0a171896baa95c10dd3bff0cbac5223721bb042d2f07e45a2d6b6f2caf.jpg)  
Veda Better Tie Baseline Better   
Percentage (%)

Figure 7. Human evaluation on Waver-bench 1.0 using Waver-T2V-1B. We compare Veda against Full Attention and VSA. Videos are generated at 480P, 81 frames (sequence length ≈ 34k). Veda achieves a parity win/tie rate against Full Attention at 90% sparsity. Notably, Veda at 95% sparsity yields superior results compared to VSA even when VSA operates at a significantly lower sparsity (87.5%), and demonstrates a substantial performance margin over VSA at equivalent (95%) sparsity levels.

training collapse, Veda’s Stage 1 explicitly decouples mask learning from feature learning via a stop-gradient operation. This provides a highly stable initialization, allowing Stage 2 to rapidly adapt to target sparsity levels without fragile scheduling, and we therefore maintain the sparsity ratio at a fixed value without warmup across both stages. For experiments at the 1B/1.3B scale, models are trained for 23k steps, and we report results using Exponential Moving Average (EMA) weights with a decay of 0.9999.

Evaluation Metrics. We assess generation performance through a combination of human evaluation and standardized benchmarks. For human evaluation, we utilize Waverbench 1.0 (Zhang et al., 2025d), a curated dataset of 304 prompts that cover a wide range of scenarios. Complementary to human ratings, we report quantitative metrics evaluated on the VBench (Huang et al., 2024) suite. To ensure fair comparison, all latency and speedup measurements are conducted on identical GPU hardware; dense baselines use FlashAttention-3, and sparse methods use the same tile-skipping sparse attention kernel.

Comparison with Existing Baselines. We benchmark Veda against Full Attention, Oracle mask and VSA (Zhang et al., 2025b). To ensure a fair comparison, all methods are trained on the identical dataset and number of steps as our approach, utilizing the specific warm-up settings proposed in their respective literature. Similarly, for all post-training sparse baselines, we strictly adhere to an equivalent training budget and learning rate schedule.

# 5.1. Generation Quality Comparison

We evaluate generation quality via a blinded Side-by-Side (SBS) study on Waver-bench 1.0 across four dimensions: Overall Quality (OQ), Motion Quality (MQ), Visual Quality (VQ), and Prompt Following (PF). Results are shown in Fig. 7.

Comparing Veda (90% sparsity) to the Full Attention baseline, we observe perceptual parity. Preference rates for VQ and PF are nearly identical (e.g., 50% vs. 49% for VQ), while a 63% tie rate in MQ confirms temporal consistency is maintained. Consequently, OQ remains indistinguishable from the full model, proving Veda preserves generative capabilities despite aggressive pruning.

![](images/9973c12fc48c84595f2668e7d0d6cb6852e09c7a49e9ecdfdcd58841493e3fb7.jpg)

<details>
<summary>bar_stacked</summary>

| Model | Layer Latency (ms) |
| :--- | :--- |
| Full Attn | 315.3 |
| Veda (Ours) | 78.3 |
| Left (Blue) | 41.2 |
| Right (Orange) | 274.1 |
| Bottom (Blue) | 39.5 |
</details>

Waver-T2V-12B, 720P/121 Frames (SP=4)

![](images/42096b8f09851c712a628957a7e92834c668f515421b0fb62ce340df3963f17a.jpg)

<details>
<summary>bar_stacked</summary>

| Category | MLP | Prepare Mask | Attention Forward |
| :--- | :--- | :--- | :--- |
| Full Attn | 17.0 | 48.6 | 65.6 |
| Veda (Ours) | 11.4 | 5.5 | 25.5 |
2.57x
</details>

Waver-T2V-1B, 480P/121 Frames

![](images/386280608440cd3f6643ae13f3956b121542cdcc93c9b5a18eb66a89d4c7800a.jpg)

<details>
<summary>bar_stacked</summary>

| Model | Layer Latency (ms) |
| :--- | :--- |
| Full Attn | 583.7 |
| Veda (Ours) | 220.7 |
| Total | 178.7 |
| Top Layer Latency (ms) | 405.0 |
| Bottom Layer Latency (ms) | 36.7 |
The label '2.64x' indicates a performance multiplier relative to the 'Full Attn' bar. The 'Full Attn' bar is labeled '583.7', while the 'Veda (Ours)' bar is labeled '220.7'.
</details>

Wan2.1-T2V-14B, 720P/81 Frames

![](images/b8ae3c1661002b5342fcf90ec5221c4cfaff321c130056e8167c1e93c5f3fb94.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Blue Segment | Orange Segment | Light Blue Segment |
| :--- | :--- | :--- | :--- |
| Full Attn | 14.6 | 22.6 | 37.2 |
| Veda (Ours) | 13.9 | 3.2 | 21.1 |
A red arrow indicates a 1.76x increase from Full Attn to Veda (Ours). The orange segment is labeled as 3.9.
</details>

Wan2.1-T2V-1.3B, 480P/81 Frames   
Figure 8. Wall-clock latency decomposition and speedup analysis. We measure the end-to-end per-layer inference latency of Veda against a FlashAttention-3 baseline on Waver-T2V and Wan2.1- T2V at two resolutions (720p and 480p). Stacked bars break down the runtime into MLP computation, sparse mask preparation (Prepare Mask), and attention kernel execution. Veda delivers substantial speedups, reaching up to 4.03× on Waver-T2V and 2.64× on Wan2.1-T2V at 720p.

![](images/9c98aa85fc56cee5f52c13ab4d1cef93c5573f09af7f6e6b8b209e69215f5bb4.jpg)  
Prompt: Wide shot of an underwater scene in the Great Barrier Reef, coral ures and colorful fish visible in clear blue water. Small schools of striped fish swim past viv corals, while a gre sea turtle moves its flippers and glides slowly above the reef. A pair of butterflyfish dart between c l branches, circling each other around waving anemones. The camera pans smoothly from this lively section to an area with pale, bleached coral, where fewer fish swim and a lone parrotfish pecks at the damaged structure. In a recovery zone, tiny coral polyps extend from new growth as small wrasse and cleaner shrimp crawl over the rocky surface.

Figure 9. Qualitative results of Waver-T2V-12B with 95% sparsity using Veda. Two representative samples are shown, each synthesized at 720P resolution with 241 frames. The text prompt for each sample is shown beneath the corresponding image. More qualitative comparisons are included in the supplementary material.   
Table 1. VBench evaluation of Veda against full attention and state-of-the-art sparsification methods on video diffusion models. We compare Veda against Full Attention and state-of-the-art sparsification techniques on Waver-T2V and Wan2.1-T2V. Sparsity indicates the reduction in computation. “Full Attn.” serves as the dense baseline. ↑ and ↓ denote better performance. 

<table><tr><td>Model</td><td>Config</td><td>Method</td><td>Subject Cons. ↑</td><td>Background Cons. ↑</td><td>Motion Smooth. ↑</td><td>Dynamic Deg. ↑</td><td>Aesthetic Quality ↑</td><td>Image Quality ↑</td><td>Sparsity ↑</td><td>FLOPs ↓</td><td>Wall Time ↓</td></tr><tr><td rowspan="5">Waver 1.0</td><td rowspan="5">1B480P81F</td><td>Full Attn.</td><td>0.938</td><td>0.955</td><td>0.979</td><td>0.969</td><td>0.655</td><td>0.693</td><td>0%</td><td> $4.83 \times 10^{16}$ </td><td>69.3s</td></tr><tr><td>Oracle Mask</td><td>0.907</td><td>0.950</td><td>0.974</td><td>0.993</td><td>0.639</td><td>0.685</td><td>90%</td><td>-</td><td>126.3s</td></tr><tr><td>VSA</td><td>0.933</td><td>0.949</td><td>0.978</td><td>0.983</td><td>0.648</td><td>0.692</td><td>87.5%</td><td> $1.56 \times 10^{16}$ </td><td>34.3s</td></tr><tr><td>Ours (S=90%)</td><td>0.940</td><td>0.954</td><td>0.980</td><td>0.963</td><td>0.650</td><td>0.699</td><td>90%</td><td> $1.47 \times 10^{16}$ </td><td>31.9s</td></tr><tr><td>Ours (S=95%)</td><td>0.934</td><td>0.951</td><td>0.978</td><td>0.979</td><td>0.649</td><td>0.698</td><td>95%</td><td> $1.28 \times 10^{16}$ </td><td>30.6s</td></tr><tr><td rowspan="5">Wan2.1</td><td rowspan="5">1.3B480P81F</td><td>Full Attn.</td><td>0.940</td><td>0.969</td><td>0.977</td><td>0.844</td><td>0.629</td><td>0.670</td><td>0%</td><td> $5.82 \times 10^{16}$ </td><td>58.5s</td></tr><tr><td>Oracle Mask</td><td>0.957</td><td>0.968</td><td>0.931</td><td>0.861</td><td>0.325</td><td>0.642</td><td>90%</td><td>-</td><td>94.6s</td></tr><tr><td>VSA</td><td>0.911</td><td>0.950</td><td>0.975</td><td>0.795</td><td>0.549</td><td>0.661</td><td>87.5%</td><td> $2.55 \times 10^{16}$ </td><td>42.5s</td></tr><tr><td>Ours (S=90%)</td><td>0.887</td><td>0.941</td><td>0.972</td><td>0.913</td><td>0.543</td><td>0.663</td><td>90%</td><td> $2.46 \times 10^{16}$ </td><td>37.6s</td></tr><tr><td>Ours (S=95%)</td><td>0.790</td><td>0.926</td><td>0.928</td><td>0.733</td><td>0.394</td><td>0.661</td><td>95%</td><td> $2.27 \times 10^{16}$ </td><td>35.8s</td></tr></table>

Against VSA (Zhang et al., 2025b), Veda excels even under stricter constraints. Veda (95% sparsity) surpasses VSA (87.5% sparsity) in VQ (54% vs. 46%). At equal 95% sparsity, Veda dominates VQ (76% vs. 24%) and OQ (39% vs. 16%). This validates that our distillation strategy retains information-dense regions better than prior heuristic pooling methods.

In addition to human evaluation, we provide a quantitative analysis using the VBench suite in Table 1. The results demonstrate that Veda effectively retains key generative capabilities despite the significant reduction in computation. On the Waver-T2V architecture, our method maintains subject consistency and motion smoothness scores that are comparable to the Full Attention baseline, even at 95% sparsity. Across differing model scales, Veda consistently exhibits a favorable efficiency-quality trade-off relative to the VSA baseline, delivering competitive metric performance while achieving lower FLOPs and reduced wall-clock latency.

# 5.2. Inference Efficiency Comparison

We analyze wall-clock latency on NVIDIA Hopper GPUs to quantify practical speedups over FlashAttention-3 (FA3) (Shah et al., 2024). We report the end-to-end latency of a single Transformer layer, including both the sparse mask predictor overhead and the sparse attention kernel execution.

In Fig. 8, for Waver-T2V-12B at 720P with 121 frames, a dense Transformer layer takes 315.3 ms, while Veda reduces it to 78.3 ms (4.03 ). The Veda breakdown is 39.5 ms for MLP computation, 1.9 ms for sparse mask preparation, and 36.9 ms for sparse attention execution, so the combined sparse-attention overhead remains well below dense attention. We observe a similar trend on Wan2.1-T2V-14B at 720P with 81 frames: layer latency decreases from 583.7 ms to 220.7 ms (2.64 ), demonstrating consistent gains across backbones with different architectures and sequence configurations.

We further evaluate scalability with sequence length. As shown in Fig. 1, at 50,220 tokens (480P, 121 frames), Veda achieves a 2.57× speedup over FA3 (25.5 ms vs. 65.6 ms). When scaling to 245,760 tokens (720P, 241 frames, SP=8), FA3 grows quadratically to 1576.5 ms, whereas Veda scales near-linearly and remains at 309.1 ms, yielding a 5.1 speedup. Overall, these results confirm that Veda mitigates the quadratic attention cost and delivers robust wall-clock improvements for high-resolution, long-sequence video generation.

# 5.3. Ablation Study

Ablation on Head-aware Tiling. We compare various static tile configurations against our Head-aware Tiling strategy. The static baselines include configurations emphasizing spatial granularity ([8, 8, 2]), temporal granularity ([4, 4, 8]), and alternative spatial-temporal trade-offs ([4, 8, 4]). Among these fixed configurations, [4, 4, 8] demonstrates the strongest performance and serves as the primary static baseline for comparison. We evaluate our Head-aware Tiling method against this optimal static configuration ([4, 4, 8]). As shown in Fig. 10, our approach achieves consistent improvements across all metrics, with notable gains of +7.2% in Motion Quality and +9.6% in Overall Quality. These results demonstrate the superiority of head-aware tiling over uniform static tiling strategies.

Table 2. Ablation on tile statistics for the score estimator. Results report the average training loss. Triplet pooling (mean, maximum, minimum) best reconstructs the oracle attention structure. 

<table><tr><td>Pooling Strategy</td><td>Training Loss (↓)</td></tr><tr><td>MaxMin + Projector</td><td>0.982</td></tr><tr><td>Avg + Projector</td><td>0.965</td></tr><tr><td>Triplet + Projector (Ours)</td><td>0.912</td></tr></table>

Ablation on Tile Score Estimator. We evaluate three pooling schemes for compressing token-wise features into tile-level representations: (1) Avg, (2) MaxMin (concatenating maximum and minimum), and (3) Triplet (concatenating mean, maximum, and minimum). We train the score estimator with the diffusion backbone frozen and report the mean squared error (MSE) to oracle full-attention tile scores. As shown in Table 2, Triplet achieves the lowest MSE (0.912), outperforming Avg and MaxMin.

# 5.4. Qualitative Results

Fig. 9 illustrates qualitative generation results of Waver-T2V-12B at 720p resolution for 241-frame sequences under 95% sparsity. Despite the extreme sparsity (95%), Veda maintains high visual quality and coherent motion, without noticeable sparse-induced structural artifacts (e.g., warping, water-ripple patterns, or temporal flickering). More qualitative examples are included in the Supplementary Material.

# 6. Conclusion and Future Work

We presented Veda, a hardware-aware sparse attention framework that addresses the scalability bottleneck of highresolution video Diffusion Transformers. Veda reformulates sparsity as an explicit tile selection problem and supervises tile scoring via full-attention distillation, rather than relying on the diffusion objective to implicitly shape sparse patterns. The induced Top-k masks preserve the tile-wise structure and ranking of full attention even under extreme pruning. By combining a statistics-aware tile score estimator and Head-Aware Tiling with a custom tile-skipping kernel, Veda translates algorithmic sparsity into practical wall-clock gains, achieving up to 5.1× end-to-end speedup on Waver and Wan2.1 with no noticeable degradation in generation quality, while exhibiting favorable scaling as spatiotemporal sequence length increases. Looking forward, further improvements may come from tighter kernel-level fusion to reduce mask preparation overhead, more advanced scheduling to sustain high MFU at sparsity levels beyond 95%, and richer distillation signals or adaptive sparsity strategies that allocate computation across timesteps and heads. Incorporating lightweight temporal caching of tile scores across adjacent diffusion steps could further improve stability.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Chen, P., Zeng, X., Zhao, M., Ye, P., Shen, M., Cheng, W., Yu, G., and Chen, T. Sparse-vdit: Unleashing the power of sparse attention to accelerate video diffusion transformers. arXiv preprint arXiv:2506.03065, 2025.   
DeepSeekTeam. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2025a.   
DeepSeekTeam. Deepseek-v3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025b.   
Fan, Z., Wang, Z., and Zhang, W. Taocache: Structuremaintained video generation acceleration. arXiv preprint arXiv:2508.08978, 2025.   
Fu, T., Huang, H., Ning, X., Zhang, G., Chen, B., Wu, T., Wang, H., Huang, Z., Li, S., Yan, S., Dai, G., Yang, H., and Wang, Y. Mixture of attention spans: Optimizing llm inference efficiency with heterogeneous sliding-window lengths. arXiv preprint arXiv:2406.14909, 2024.   
Gao, Y., Zeng, Z., Du, D., Cao, S., Zhou, P., Qi, J., Lai, J., So, H. K.-H., Cao, T., Yang, F., and Yang, M. Seerattention: Learning intrinsic sparse attention in your llms. arXiv preprint arXiv:2410.13276, 2025.

Hassani, A., Walton, S., Li, J., Li, S., and Shi, H. Neighborhood attention transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6185–6194, 2023.   
Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.   
HunyuanTeam. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.   
Li, Y., Jiang, H., Zhang, C., Wu, Q., Luo, X., Ahn, S., Abdi, A. H., Li, D., Gao, J., Yang, Y., et al. Mminference: Accelerating pre-filling for long-context vlms via modality-aware permutation sparse attention. arXiv preprint arXiv:2504.16083, 2025.   
Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He, X., Ye, Y., Yuan, S., Chen, L., Jia, T., Zhang, J., Tang, Z., Pang, Y., She, B., Yan, C., Hu, Z., Dong, X., Chen, L., Pan, Z., Zhou, X., Dong, S., Tian, Y., and Yuan, L. Opensora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.   
Litman, E. Scaled-dot-product attention as onesided entropic optimal transport. arXiv preprint arXiv:2508.08369, 2025.   
Lu, E., Jiang, Z., Liu, J., Du, Y., Jiang, T., Hong, C., Liu, S., He, W., Yuan, E., Wang, Y., Huang, Z., Yuan, H., Xu, S., Xu, X., Lai, G., Chen, Y., Zheng, H., Yan, J., Su, J., Wu, Y., Zhang, N. Y., Yang, Z., Zhou, X., Zhang, M., and Qiu, J. Moba: Mixture of block attention for long-context llms. arXiv preprint arXiv:2502.13189, 2025.   
Markidis, S., Chien, S. W. D., Laure, E., Peng, I. B., and Vetter, J. S. Nvidia tensor core programmability, performance and precision. In 2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), pp. 522–531. IEEE, May 2018. doi: 10.1109/ipdpsw.2018.00091.   
Peebles, W. and Xie, S. Scalable Diffusion Models with Transformers . In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182, Los Alamitos, CA, USA, October 2023. IEEE Computer Society. doi: 10.1109/ICCV51070.2023.00387.   
QwenTeam. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2025.   
Ramsauer, H., Schafl, B., Lehner, J., Seidl, P., Widrich,¨ M., Adler, T., Gruber, L., Holzleitner, M., Pavlovic, M., ´ Sandve, G. K., Greiff, V., Kreil, D., Kopp, M., Klambauer,

G., Brandstetter, J., and Hochreiter, S. Hopfield networks is all you need. arXiv preprint arXiv:2008.02217, 2021.   
Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. arXiv preprint arXiv:2407.08608, 2024.   
Spector, B. F., Arora, S., Singhal, A., Fu, D. Y., and Re,´ C. Thunderkittens: Simple, fast, and adorable ai kernels. arXiv preprint arXiv:2410.20399, 2024.   
StepVideoTeam. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.   
Sun, W., Tu, R.-C., Ding, Y., Jin, Z., Liao, J., Liu, S., and Tao, D. Vorta: Efficient video diffusion via routing sparse attention. arXiv preprint arXiv:2505.18809, 2025.   
Team, M. L., Cai, X., Huang, Q., Kang, Z., Li, H., Liang, S., Ma, L., Ren, S., Wei, X., Xie, R., and Zhang, T. Longcatvideo technical report. arXiv preprint arXiv:2510.22200, 2025.   
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., \` Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pp. 6000–6010, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.   
Wang, L., Cheng, Y., Shi, Y., Tang, Z., Mo, Z., Xie, W., Ma, L., Xia, Y., Xue, J., Yang, F., et al. Tilelang: A composable tiled programming model for ai systems. arXiv preprint arXiv:2504.17577, 2025.   
WanTeam. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
Wu, J., Hou, L., Yang, H., Tao, X., Tian, Y., Wan, P., Zhang, D., and Tong, Y. Vmoba: Mixture-of-block attention for video diffusion models. arXiv preprint arXiv:2506.23858, 2025.   
Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., Chen, J., Stoica, I., Keutzer, K., and Han, S. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.   
Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Zhang, Y., Wang, W., Cheng, Y., Bin, X., Gu, X., Dong, Y., and Tang, J. Cogvideox: Text-to-video diffusion models with an expert transformer. In Yue, Y., Garg, A., Peng, N., Sha, F., and Yu, R. (eds.), International Conference on Learning Representations, volume 2025, pp. 83048– 83077, 2025.   
Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. From slow bidirectional to fast autoregressive video diffusion models. arXiv preprint arXiv:2412.07772, 2025.   
Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., Xie, Z., Wei, Y. X., Wang, L., Xiao, Z., Wang, Y., Ruan, C., Zhang, M., Liang, W., and Zeng, W. Native sparse attention: Hardware-aligned and natively trainable sparse attention. arXiv preprint arXiv:2502.11089, 2025.   
Zhang, J., Xiang, C., Huang, H., Wei, J., Xi, H., Zhu, J., and Chen, J. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137, 2025a.   
Zhang, P., Chen, Y., Huang, H., Lin, W., Liu, Z., Stoica, I., Xing, E., and Zhang, H. VSA: Faster Video Diffusion with Trainable Sparse Attention. arXiv preprint arXiv:2505.13389, 2025b.   
Zhang, P., Chen, Y., Su, R., Ding, H., Stoica, I., Liu, Z., and Zhang, H. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025c.   
Zhang, Y., Yang, H., Zhang, Y., Hu, Y., Zhu, F., Lin, C., Mei, X., Jiang, Y., Peng, B., and Yuan, Z. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025d.   
Zhao, X., Jin, X., Wang, K., and You, Y. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2025.   
Zheng, K., Wang, Y., Ma, Q., Chen, H., Zhang, J., Balaji, Y., Chen, J., Liu, M.-Y., Zhu, J., and Zhang, Q. Large scale diffusion distillation via score-regularized continuoustime consistency. arXiv preprint arXiv:2510.08431, 2025.

# A. Theoretical Framework for Veda

In this section, we establish the theoretical foundations of our approach and provide deeper algorithmic and mathematical insights into the nature of sparse attention.

# A.1. The Necessity of Accurate Oracle Masks

The self-attention mechanism can be formally viewed as an Energy-Based Model (EBM)(Ramsauer et al., 2021; Litman, 2025). Let $\mathbf { u } \in \mathbb { R } ^ { N }$ denote the logits (pre-softmax scores), where $u _ { j } = \mathbf { q } ^ { T } \mathbf { k } _ { j }$ . The generating potential function of the attention distribution is the Log-Sum-Exp (LSE) function(Litman, 2025), Ψ(u):

$$
\Psi (\mathbf {u}) = \tau \log \sum_ {j = 1} ^ {N} \exp (u _ {j} / \tau), \tag {10}
$$

where τ is the temperature parameter. The attention probability distribution P is the gradient of this potential, $P = \nabla \Psi ( \mathbf { u } )$ . The LSE function is dominated by the maximum term, $u _ { m a x }$ .

Proposition A.1 (Distributional Shift via Mask Error). Let $\mathcal { M } ^ { * }$ be the set of indices corresponding to the true Top-k keys (Oracle), and let Mˆ be an estimated mask such that argmax(u) ∈/ Mˆ . The perturbed potential Ψ˜ excludes the dominant energy term. This exclusion forces a renormalization of probability mass over the retained keys Mˆ :

$$
\tilde {P} _ {j} = \frac {\exp (u _ {j} / \tau)}{\sum_ {r \in \hat {\mathcal {M}}} \exp (u _ {r} / \tau)} = \frac {Z}{\hat {Z}} P _ {j} > P _ {j}, \quad \forall j \in \hat {\mathcal {M}}, \tag {11}
$$

where $\begin{array} { r } { Z = \sum _ { r = 1 } ^ { N } \exp ( u _ { r } / \tau ) } \end{array}$ and $\begin{array} { r } { \hat { Z } = \sum _ { r \in \hat { \mathcal { M } } } \exp ( u _ { r } / \tau ) . } \end{array}$ .

This renormalization can assign inflated confidence to noise or irrelevant tokens, a phenomenon we term structural hallucination. Unlike linear operations, where errors are additive, the softmax nonlinearity can amplify mask errors through exponentiated logits and row-wise renormalization. Thus, accurate recovery of the Oracle Mask is a structural prerequisite for high-sparsity attention.

# A.2. Limitations of Linear Pooling Representations

Existing methods like VSA(Zhang et al., 2025b) or Sparge Attention(Zhang et al., 2025a) often employ Average Pooling to compress tiles. We argue that linear pooling is theoretically insufficient for approximating the bilinear attention operation. From a spectral perspective, Average Pooling is a low-pass filter equivalent to a projection matrix $W _ { a v g }$ . The approximated score becomes $\hat { u } _ { i j } = \mathbf { q } _ { i } ^ { T } W _ { a v g } ^ { T } W _ { a v g } \mathbf { k } _ { j }$ .

Lemma A.2 (Loss of Second-Order Statistics). The inner product $\langle \mathbf { q } , \mathbf { k } \rangle$ depends on both the angle and magnitude of the high-frequency components of the vectors. The projection $W _ { a v g }$ collapses the vector variance onto the mean, discarding the high-frequency harmonics required to distinguish orthogonal vectors. Mathematically, $\langle \bar { \bf q } , \bar { \bf k } \rangle \neq \overline { { \langle { \bf q } , { \bf k } \rangle } }$ . The loss of orthogonality leads to rank deficiency in the approximated attention matrix, making it impossible to distinguish between distinct but mean-similar tiles.

# A.3. Reconstructing Attention via Sufficient Statistics

To reconstruct the attention distribution accurately with minimal I/O, we propose a statistical estimator based on vector decomposition. We treat the elements of dimension-d vectors q and k as samples from a distribution.

Theorem A.3 (Dot Product Decomposition). The dot product of two vectors $\mathbf { q } , \mathbf { k } \in \mathbb { R } ^ { d }$ can be decomposed into a mean product term and a covariance term:

$$
\mathbf {q} \cdot \mathbf {k} = d \mu_ {q} \mu_ {k} + d \operatorname{Cov} (\mathbf {q}, \mathbf {k}), \tag {12}
$$

where $\mathrm { C o v } ( { \bf q } , { \bf k } ) = \rho \sigma _ { q } \sigma _ { k }$ .

While $\mu \left( \operatorname { A v g } \right)$ captures the centroid, reconstructing the covariance requires information about the spread (σ) and correlation $( \rho )$ . We employ the Bhatia-Davis inequality to bound the variance using only extremum statistics.

Lemma A.4 (Variance Bound via Extrema). For any bounded random variable X with support $[ m , M ]$ and mean $\mu ,$ the variance is bounded by:

$$
\sigma^ {2} \leq (M - \mu) (\mu - m). \tag {13}
$$

Assuming the vector elements follow a unimodal distribution (e.g., Gaussian or Laplacian, typical in LayerNorm-normalized embeddings), the standard deviation is proportional to the range: $\begin{array} { r } { \sigma \approx \frac { M - m } { \alpha } } \end{array}$ . By retaining the triplet $\{ \mu , m , M \}$ (Avg, Min, Max), we effectively construct a hyper-rectangle bounding box that constrains the manifold of the vectors. We derive our estimator $\mathcal { S } ( \mathbf { q } , \mathbf { k } )$ as:

$$
\mathcal {S} \approx d \cdot \mu_ {q} \mu_ {k} + \lambda \Delta_ {q} \Delta_ {k} \tag {14}
$$

where $\Delta _ { x } = \sqrt { ( M _ { x } - \mu _ { x } ) ( \mu _ { x } - m _ { x } ) }$ represents the dispersion bound for $x \in \{ q , k \}$ . Here, the first term anchors the centroid interaction, while the second term approximates the maximal energy potential of the covariance.

# A.4. The Necessity of Intra-Tile Similarity

We now justify why tokens within a tile should be similar for this approximation to hold. Consider tile-sparse attention where a single statistic represents a tile of size B. Let $\pmb { \mu _ { \ q } }$ and $\mu _ { \mathbf { k } }$ denote the tile means, and define the residual vectors $\mathbf { r } _ { i } ^ { q } = \mathbf { q } _ { i } - \mu _ { \mathbf { q } }$ and $\mathbf { r } _ { i } ^ { k } = \mathbf { k } _ { i } - \mu _ { \mathbf { k } }$ . The approximation error E from the neglected residual interactions can be written as:

$$
E = \left| \sum_ {i = 1} ^ {B} \langle \mathbf {r} _ {i} ^ {q}, \mathbf {r} _ {i} ^ {k} \rangle \right|. \tag {15}
$$

By the Cauchy-Schwarz inequality, this error is bounded by the product of the tile-wise standard deviations:

$$
E \leq \left(\sum_ {i = 1} ^ {B} \| \mathbf {r} _ {i} ^ {q} \| _ {2} ^ {2}\right) ^ {1 / 2} \left(\sum_ {i = 1} ^ {B} \| \mathbf {r} _ {i} ^ {k} \| _ {2} ^ {2}\right) ^ {1 / 2} = B \cdot \sigma_ {\mathbf {q}} \cdot \sigma_ {\mathbf {k}}, \tag {16}
$$

where $\begin{array} { r } { \sigma _ { \mathbf { q } } ^ { 2 } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \| \mathbf { r } _ { i } ^ { q } \| _ { 2 } ^ { 2 } } \end{array}$ and $\begin{array} { r } { \sigma _ { \mathbf { k } } ^ { 2 } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \| \mathbf { r } _ { i } ^ { k } \| _ { 2 } ^ { 2 } . } \end{array}$

Corollary A.5 (Clustering Condition). A sufficient condition for the sparse approximation to be robust $( i . e . , E  0 )$ is $\sigma _ { \mathbf { q } }  0 o r \sigma _ { \mathbf { k } }  0 ,$ , and reducing both variances tightens the bound. This implies that tokens within a tile should lie on a local low-dimensional manifold with minimal variance. If tiles contain diverse tokens (high σ), the uncertainty bound of the estimator grows, rendering the sparse mask unreliable.

# A.5. Token Clustering as Bandwidth Minimization

To satisfy the condition $\sigma  0 ,$ , we formulate token clustering as a graph labeling problem. Let $\mathcal { G } = ( \nu , \mathcal { E } )$ be the token graph where edge weights $W _ { i j } \propto \exp ( \mathbf { q } _ { i } ^ { T } \mathbf { k } _ { j } )$ . We seek a bijective mapping $\phi : \mathcal { V } \to \{ 1 , \dots , N \}$ that minimizes the matrix bandwidth.

Definition A.6 (Manifold Embedding under 3D RoPE). Let tokens $x _ { i }$ reside on a 3D spatio-temporal manifold M induced by Rotary Positional Embeddings (RoPE). The attention affinity decays with the geodesic distance on this manifold: $W _ { i j } \approx f ( \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { \Sigma } )$ . A locality-preserving mapping ϕ must satisfy the Holder continuity condition: ¨

$$
\left| \phi (u) - \phi (v) \right| \leq C \cdot \left\| \mathbf {x} _ {u} - \mathbf {x} _ {v} \right\| _ {2} ^ {\alpha}. \tag {17}
$$

Standard raster scanning $( \phi _ { \mathrm { f l a t } } )$ violates this condition for temporal neighbors, yielding a Lipschitz constant $C \propto H \times W$ , resulting in a dispersed, high-bandwidth attention matrix. We propose Head-aware 3D Tiling, which creates a mapping $\phi _ { \mathrm { t i l e } }$ where temporal neighbors satisfy $| \phi _ { \mathrm { t i l e } } ( u ) - \phi _ { \mathrm { t i l e } } ( v ) | \le p _ { h } p _ { w }$ . This effectively folds the 3D manifold into 1D memory while preserving local neighborhoods, implicitly performing graph clustering. Furthermore, we employ Anisotropic Metric Adaptation to search for optimal tile shapes $\left( p _ { t } , p _ { h } , p _ { w } \right)$ , aligning the tiling strategy with the specific induced metric $d _ { h } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } )$ of each attention head. This topological optimization minimizes attention bandwidth, effectively clustering highaffinity tokens to lower intra-tile variance σ. Consequently, the geometric regularization satisfies the spectral approximation conditions, validating the representational power of the compressed statistics $\{ \mu , m , M \}$ .

# A.6. Projector Optimization via Information Geometry

Finally, we address the non-differentiability of the Top-k selection required to train the MLP projector. We leverage the framework of Entropic Optimal Transport (EOT) (Litman, 2025). The attention mechanism operates on a statistical manifold equipped with the Fisher-Rao metric. The standard Euclidean gradient is suboptimal for optimization on the probability simplex.

Theorem A.7 (Local KL Geometry). Let $P _ { \mathrm { o r a c l e } }$ be the ground truth distribution derived from full attention, and $P _ { \mathrm { p r e d } }$ be the distribution predicted by the MLP. Around the oracle parameters $\theta _ { t } ,$ the Kullback-Leibler (KL) divergence admits the local second-order approximation:

$$
\mathcal {L} = D _ {\mathrm{KL}} (P _ {\text { oracle }} \| P _ {\text { pred }}) \approx \frac {1}{2} (\theta_ {t} - \theta_ {s}) ^ {T} I (\theta_ {t}) (\theta_ {t} - \theta_ {s}). \tag {18}
$$

Thus, KL supervision locally weights prediction errors by the Fisher Information Matrix $I ( \theta _ { t } )$ , matching the natural geometry of the probability simplex.

By utilizing KL divergence as the training objective, we train the score estimator before the discrete Top-k operation and therefore avoid differentiating through Top-k. Instead, we force the MLP to learn the geometry of the energy landscape (the LSE potential field). This encourages the predicted logits uˆ to maintain the same ranking and local geometry as the oracle logits u, leading to accurate mask recall during inference.

# B. Detailed Implementation Algorithms for Veda

We provide the complete algorithmic pseudocode for the core components of Veda, which are omitted from the main text due to space constraints below.

Algorithm 1 Head-Aware Tiling Search   
1: Input: Latent dims (T, H, W), Calibration Set $D_{cal}$ , Tile Size B = 128, Top-k Budget $k_{top}$ .
2: Output: Optimal Configuration Map $\pi^{*} \in \Omega^{L \times N_{h}}$ .
3: // Construct hardware-aligned search space
4: $\Omega \leftarrow \{(p_{t}, p_{h}, p_{w}) \in \mathbb{N}^{3} \mid p_{t} p_{h} p_{w} = B\}$ 5: Initialize cumulative error $E \in R^{L \times N_{h} \times |\Omega|}$ to 0.
6: for sample $x \in D_{cal}$ do
7: Perform inference to cache $\{Q, K, V\}$ 8: for layer $l \in \{1 \ldots L\}$ , head $h \in \{1 \ldots N_{h}\}$ do
9: // Get Full-Attention Map and Output
10: $A_{l,h}^{\mathrm{fu}} \leftarrow \mathrm{Softmax}(Q_{l,h} K_{l,h}^{\top} / \sqrt{d})$ 11: $O_{l,h}^{\mathrm{fu}} \leftarrow A_{l,h}^{\mathrm{fu}} V_{l,h}$ 12: for $\pi \in \Omega$ do
13: // 1. Tiling QKV and Full-Attention Map
14: $\tilde{Q}, \tilde{K}, \tilde{V} \leftarrow \text{Tiling}(\{Q_{l,h}, K_{l,h}, V_{l,h}\}, \pi)$ 15: $A^{\pi} \leftarrow \text{Softmax}(\tilde{Q}\tilde{K}^{\top} / \sqrt{d})$ 16: // 2. Generate Oracle Mask from Full Attention
17: $S^{tile} \leftarrow \text{RowNorm}(MaxPool(A^{\pi}))$ 18: $\widetilde{M} \leftarrow \text{Top-k}(S^{tile}, k_{\text{top}})$ 19: // 3. Compute Sparse Output and Error
20: $O_{l,h}^{sp} \leftarrow UnTile(SparseAttn(\tilde{Q}, \tilde{K}, \tilde{V}, \widetilde{M}), \pi)$ 21: $E_{l,h,\pi} \leftarrow E_{l,h,\pi} + \|O_{l,h}^{\mathrm{fu}} - O_{l,h}^{\mathrm{sp}}\|_{F}^{2}$ 22: end for
23: end for
24: end for
25: // Select configuration minimizing expected error
26: $\pi_{l,h}^{*} \leftarrow \arg\min_{\pi \in\Omega} E_{l,h,\pi} \quad \forall l, h$ 27: return $\pi^{*}$

Algorithm 2 Tile Score Estimation & Mask Generation   
1: Input: Query Q, Key $K \in R^{N_{h} \times N \times d}$ , Tile Size B, Top-k Budget $k_{top}$ , Latent Dim $d'$ 2: Output: Predicted distribution $A^{pred}$ and binary mask $\widetilde{M} \in \{0, 1\}^{N_{h} \times T_{q} \times T_{k}}$ 3: // Tiling
4: $T_{q} \leftarrow N/B$ , $T_{k} \leftarrow N/B$ 5: $\tilde{Q} \leftarrow \text{Tile}(Q, \{N_{h}, T_{q}, B, d\})$ 6: $\tilde{K} \leftarrow \text{Tile}(K, \{N_{h}, T_{k}, B, d\})$ 7: // Triplet Statistics Extraction (TripPool)
8: $Z_{q} \leftarrow \text{Avg}[\tilde{Q}] \oplus \text{Max}[\tilde{Q}] \oplus \text{Min}[\tilde{Q}]$ 9: $Z_{k} \leftarrow \text{Avg}[\tilde{K}] \oplus \text{Max}[\tilde{K}] \oplus \text{Min}[\tilde{K}]$ 10: // Projection & Score Estimation
11: for each head $h \in \{1, \ldots, N_{h}\}$ do
12: $\hat{\mathbf{q}}_{h} \leftarrow \phi_{q}^{(h)}(\mathbf{Z}_{q,h}), \quad \hat{\mathbf{k}}_{h} \leftarrow \phi_{k}^{(h)}(\mathbf{Z}_{k,h})$ 13: $S_{h}^{pred} \leftarrow \frac{\hat{\mathbf{q}}_{h} \hat{\mathbf{k}}_{h}^{\top}}{\sqrt{d'}}$ 14: $A_{h}^{pred} \leftarrow \text{Softmax}(S_{h}^{pred})$ // row-wise over key tiles
15: $\widetilde{M}_{h} \leftarrow I(S_{h}^{pred} \geq Top-k-Threshold(S_{h}^{pred}, k_{top}))$ 16: end for
17: return $A^{pred}, \widetilde{M}$

Algorithm 3 Tile-Wise Distillation for Tile Score Estimator   
1: Input: Query Q, Key $K \in R^{N_{h} \times N \times d}$ , Tile Size B, Top-k Budget $k_{top}$ , Latent Dim $d'$ 2: Output: Distillation Loss $L_{distill}$ 3: // Full-Attention Teacher
4: $A^{*} \leftarrow Softmax\left(\frac{QK^{\top}}{\sqrt{d}}\right)$ 5: // Target Construction (Score & Weight)
6: $S^{tgt} \leftarrow MaxPool(A^{*}, kernel = B, stride = B)$ 7: $A^{tgt} \leftarrow RowNormalize(S^{tgt})$ 8: // Student Prediction (Estimator with Stop-Gradient)
9: $A^{pred}, \widetilde{M} \leftarrow Algorithm 2(\text{sg}(Q), \text{sg}(K), B, k_{\text{top}}, d')$ 10: // Distribution Alignment
11: $L_{distill} \leftarrow Mean_{h,i}\left[\mathcal{D}_{\text{KL}}(A^{\text{tgt}}[h, i, :] || A^{\text{pred}}[h, i, :])\right]$ 12: backward $L_{distill}$

# C. Waver-T2V Implementation Details

This section provides specifications for both the Waver-T2V-1B and Waver-T2V-12B models (Zhang et al., 2025d). The tables below detail the architectural configurations and training hyperparameters. The 1B variant employs a Multimodal Diffusion Transformer (MM-DiT) (Yang et al., 2025) backbone with 8 Dual Flow blocks and 22 Single Flow blocks. The 12B variant scales to 16 Dual Flow layers and 40 Single Flow layers with expanded dimensionality. Both models utilize the Wan-VAE for latent representation and Flan-T5-XXL for text encoding.

Table 3. Model configuration comparison between Waver-T2V-1B and Waver-T2V-12B. Both architectures utilize a hybrid backbone comprising Dual Flow and Single Flow Multimodal Diffusion Transformer (MM-DiT) blocks.   
Waver-T2V-1B 

<table><tr><td>Model Config</td><td>Value</td></tr><tr><td>Model Size</td><td>1B</td></tr><tr><td>Backbone Architecture</td><td>MM-DiT</td></tr><tr><td>Dual Flow Layers</td><td>8</td></tr><tr><td>Single Flow Layers</td><td>22</td></tr><tr><td>Patch Size</td><td> $1 \times 2 \times 2$ </td></tr><tr><td>Hidden Dimension (Attention Head)</td><td>128</td></tr><tr><td>Num Heads</td><td>12</td></tr><tr><td>In Channels</td><td>16</td></tr><tr><td>Out Channels</td><td>16</td></tr><tr><td>Cross Attention Dim</td><td>1536</td></tr><tr><td>Text Embedding Dim</td><td>4096</td></tr><tr><td>VAE Model</td><td>Wan-VAE</td></tr><tr><td>Activation Function</td><td>GELU</td></tr><tr><td>Normalization</td><td>RMS Norm</td></tr><tr><td>Positional Embedding</td><td>RoPE</td></tr></table>

Waver-T2V-12B 

<table><tr><td>Model Config</td><td>Value</td></tr><tr><td>Model Size</td><td>12B</td></tr><tr><td>Backbone Architecture</td><td>MM-DiT</td></tr><tr><td>Dual Flow Layers</td><td>16</td></tr><tr><td>Single Flow Layers</td><td>40</td></tr><tr><td>Patch Size</td><td> $1 \times 2 \times 2$ </td></tr><tr><td>Hidden Dimension (Attention Head)</td><td>128</td></tr><tr><td>Num Heads</td><td>24</td></tr><tr><td>In Channels</td><td>36</td></tr><tr><td>Out Channels</td><td>16</td></tr><tr><td>Cross Attention Dim</td><td>3072</td></tr><tr><td>Text Embedding Dim</td><td>4096</td></tr><tr><td>VAE Model</td><td>Wan-VAE</td></tr><tr><td>Activation Function</td><td>GELU</td></tr><tr><td>Normalization</td><td>AdaLayerNormZero</td></tr><tr><td>Positional Embedding</td><td>RoPE</td></tr></table>

Table 4. Training hyperparameters comparison between Waver-T2V-1B and Waver-T2V-12B. Both models employ Rectified Flow matching objectives with distinct optimization configurations.   
Waver-T2V-1B 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td> $6.0 \times 10^{-5}$ </td></tr><tr><td>LR Scheduler</td><td>Constant</td></tr><tr><td>Batch Size</td><td>128</td></tr><tr><td>Resolution</td><td> $480 \times 864$  (161 frames)</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>AdamW  $\beta$ </td><td>(0.9, 0.999)</td></tr><tr><td>Weight Decay</td><td> $1.0 \times 10^{-2}$ </td></tr><tr><td>Gradient Clipping</td><td>1.0</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>Training Objective</td><td>Flow Matching</td></tr></table>

Waver-T2V-12B 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td> $5.0 \times 10^{-6}$ </td></tr><tr><td>LR Scheduler</td><td>Constant</td></tr><tr><td>Batch Size</td><td>64</td></tr><tr><td>Resolution</td><td> $720 \times 1280$  (241 frames)</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>AdamW  $\beta$ </td><td>(0.9, 0.999)</td></tr><tr><td>Weight Decay</td><td> $1.0 \times 10^{-2}$ </td></tr><tr><td>Gradient Clipping</td><td>1.0</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>Training Objective</td><td>Flow Matching</td></tr></table>

# D. Human Evaluation of Dynamic versus Static Tile Size Configurations

To empirically validate the effectiveness of our Head-aware dynamic tile size selection strategy, we conduct a human evaluation study comparing various static tile configurations against our proposed approach, where human annotators assess the perceptual quality and generation fidelity of the outputs.

![](images/c6520a43e044a2132ed9c07aaa84b89eb6a3c9f141e8ab0743c23bf9be1c2e51.jpg)

<details>
<summary>radar</summary>

| Category          | StaticTile Size [4,8,4] vs [4,4,8] | Head-aware Tile Size vs [4,4,8] | StaticTile Size [8,8,2] vs [4,4,8] |
| ----------------- | ---------------------------------- | ------------------------------- | --------------------------------- |
| Motion Quality    | -10%                               | 10%                             | 0%                                |
| Visual Quality    | 0%                                 | 10%                             | 0%                                |
| Prompt Following  | -10%                               | 10%                             | 0%                                |
| Overall Quality   | 0%                                 | 10%                             | 0%                                |
</details>

Figure 10. Human evaluation of dynamic versus static tile size configurations. We benchmark various static tile size settings alongside our proposed Head-aware dynamic selection method against the strongest static baseline ([4, 4, 8]). While the [4, 4, 8] configuration emerges as the optimal fixed setting, our dynamic approach consistently outperforms all static alternatives, achieving positive net win rates across evaluations and demonstrating superior generation quality through adaptive tile size allocation.