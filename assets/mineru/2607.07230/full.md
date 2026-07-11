# Attention-Guided Cross-Temporal Clustering for Self-Supervised Video Object Segmentation

Waqas Arshid<sup>1</sup>, Mohammad Awrangjeb<sup>1\*</sup>, Alan Wee-Chung Liew<sup>1</sup>, Yongsheng Gao<sup>2</sup>

<sup>1</sup>School of Information and Communication Technology, Grifith University, Brisbane, QLD, Australia.

<sup>2</sup>School of Engineering and Built Environment – Electrical and Electronic Engineering, Grifith University, Brisbane, QLD, Australia.

\*Corresponding author(s). E-mail(s): mohammad.awrangjeb@grifith.edu.au; Contributing authors: waqas.arshid@grifith.edu.au; a.liew@grifith.edu.au; yongsheng.gao@grifith.edu.au;

## Abstract

Video object segmentation (VOS) is a fundamental task in video understanding, requiring accurate delineation and consistent tracking of objects across frames. While supervised methods achieve strong performance, they depend on densely annotated datasets that are costly to obtain and limited in domain coverage. Self-supervised learning ofers a promising alternative by removing the need for manual labels; however, existing approaches often struggle to jointly maintain spatial accuracy and temporal coherence, particularly in unconstrained multi-object scenarios. Many rely on optical flow, synthetic motion cues, or task-specific pretraining, limiting scalability and generalisation. We propose a self-supervised framework, Cross-Temporal Consistency and Clustering (CTC<sup>2</sup>), that learns mid-level, part-aware representations by combining attention-guided token selection with lightweight temporal clustering. Instead of operating at the pixel or whole-object level, the method aligns soft part assignments across time using a saliency-weighted symmetric consistency objective. The framework leverages a frozen transformer backbone with lightweight modules for adaptive token selection and multi-ofset temporal alignment, enabling eficient scaling across resolutions and motion patterns. CTC<sup>2</sup> achieves competitive performance among recent self-supervised methods while maintaining real-time throughput, without relying on motion cues or domain-specific adaptation. It further demonstrates stable behaviour under cross-dataset evaluation and can be extended to a semi-supervised setting using a first-frame mask. These results suggest that attention-guided token selection combined with temporal clustering ofers a practical and scalable direction for label-free video segmentation.

Keywords: video object segmentation, self-supervised learning, unsupervised representation learning, vision transformers, saliency-guided attention, part-level representation, temporal consistency

Accepted for publication in Machine Intelligence Research. DOI: 10.1007/s11633-026-1648-7.

## 1 Introduction

Understanding how objects evolve over time—how they move, deform, interact, or become occluded—is central to visual intelligence. Video Object Segmentation (VOS) supports applications such as autonomous navigation, intelligent video editing, augmented reality, and human–robot interaction [1, 2]. While supervised methods achieve strong performance using dense per-frame annotations [3, 4], they scale poorly: pixelaccurate labels remain costly to obtain at scale, and large annotated datasets are not always available for new domains or rare categories. This bottleneck limits deployment in settings with privacy constraints, long-tail categories, or rapidly changing domains.

To mitigate annotation dependence, recent work has increasingly explored selfsupervised VOS, which aims to learn spatiotemporal representations directly from unlabeled videos [5–7]. Many methods replace human labels with intrinsic signals from temporal coherence, motion consistency, or appearance regularities. However, existing approaches vary widely in the assumptions they require, ranging from motion-based cues and synthetic labels to correspondence learning frameworks that rely on strong data augmentations or contrastive objectives. Common dependencies include optical flow [8], which can fail in low-texture or fast-motion regions; motion saliency [9], which is often confounded by background dynamics; and pseudo-labels from synthetic cues [10, 11], which may introduce bias. Meanwhile, correspondence-based SSL methods—such as CRW, TimeT, and TAPIR-style matching—show that flow-free alignment is feasible, but may still break under occlusion, clutter, or large appearance changes. These limitations motivate dense video representations that remain temporally consistent under viewpoint change, occlusion, and background noise—without brittle external priors.

We argue that a promising direction is to learn representations at the level of semantic object parts, rather than whole objects or individual pixels. Parts such as wheels, wings, or ears tend to persist through occlusion, deform predictably, and compose into objects [12–14]. This mid-level abstraction generalises across categories and can be more stable under viewpoint change and intra-class variation [15, 16]. However, discovering temporally consistent parts without supervision remains challenging, particularly when neither motion cues nor semantic annotations are available. Prior work has explored unsupervised part discovery in static images and short-range correspondences in videos, but extending these ideas to long-range, label-free temporal consistency remains an open problem.

In this work, we build upon clustering-based part discovery and temporal correspondence learning to develop a self-supervised framework that encourages stable part-level alignment across time. We match salient spatial tokens across adjacent frames and enforce consistency in their soft cluster assignments. The key intuition is that the same semantic part (e.g., a bear’s ear or a car’s wheel) should induce similar feature distributions over time, even under scale, rotation, or partial occlusion [17]. We operationalise this intuition using a symmetric Kullback–Leibler (KL) consistency loss, chosen for balanced alignment rather than as a new objective, and we provide ablations against one-sided KL, cross-entropy, and contrastive alternatives.

A central component is the [CLS] attention map from a frozen SAM2 encoder [18] to identify salient tokens. Rather than introducing a new saliency mechanism, we leverage an emergent property of ViTs: the [CLS] token aggregates global context and often highlights semantically meaningful regions. In practice, strong [CLS]-attention responses align with salient regions [14, 19]. We adopt an adaptive top-p selection strategy with coarse grid diversity to balance semantic focus and spatial coverage, mitigating failure cases involving small objects, clutter, and varying resolutions.

Technically, we treat the frozen SAM2 encoder as a general-purpose visual tokenizer trained on large-scale segmentation data [20]. Token embeddings are fed to a lightweight MLP clustering head to produce soft part assignments. We track clusters using cosine-similarity matching and enforce temporal alignment via the symmetric KL objective. Keeping the backbone frozen reduces overhead and avoids domainspecific fine-tuning; however, we explicitly evaluate frozen versus fine-tuned variants. To handle variable motion and frame rates, we introduce a multi-∆t temporal pyramid with match-rate control, inspired by multi-step correspondence strategies but tailored to stable clustering rather than contrastive tracking.

Our approach complements prior work such as SelfMask [9], TimeT [7], BETRayed Attention [21], and modern flow-free correspondence models. Rather than relying on motion cues, heavy augmentations, or decoder heads, CTC<sup>2</sup> focuses on attentionguided token selection and cross-temporal part clustering. This mid-level grouping improves robustness to occlusion, background interference, and intra-class variability. We also relate our design to memory-centric VOS methods such as XMem [22] and AOT [23], while emphasising that our framework does not use external memory modules or mask propagation.

We evaluate on three established self-supervised VOS benchmarks—DAVIS-2017 [24], DAVIS-2016 [2], and YouTube-VOS [3]. Following common practice in label-free dense segmentation [12, 15], the first annotated frame is used only to define the evaluation mapping. For completeness, we also report semi-supervised results where the first-frame mask is provided at inference, and cross-dataset generalisation experiments to assess robustness under domain shift. Across these settings, CTC<sup>2</sup> achieves competitive performance while maintaining real-time throughput.

In summary, our main contributions are:

• We present a self-supervised framework for part-level temporal consistency that combines attention-guided token selection with lightweight cross-temporal clustering and a symmetric KL consistency objective.

• We introduce adaptive token budgeting based on top-p attention and spatial diversity, improving semantic coverage across resolutions without increasing computational complexity.

• We develop a multi-∆t temporal supervision strategy with match-rate control to improve correspondence stability under variable motion and frame rates.

• We report competitive results on DAVIS-2017, DAVIS-2016, and YouTube-VOS, supported by cross-dataset and semi-supervised evaluations highlighting robustness.

## 2 Related Work

## 2.1 Self-Supervised Learning for Dense Visual Understanding.

Self-supervised learning (SSL) has become a foundational paradigm in computer vision, enabling representation learning without human-annotated labels [25]. Early image-level methods such as MoCo [26], BYOL [27], and SimSiam [28] established instance discrimination as a strong pretext task, achieving competitive performance in classification and retrieval. However, because these approaches emphasise global embeddings, transferring them directly to dense prediction tasks—where spatial correspondence is essential—remains challenging.

To address this limitation, later work introduced spatial structure into SSL objectives. DenseCL [29] aligns pixel-level features across augmented views; PixPro [30] propagates local features across spatial neighbourhoods. Complementary to pixelcentric objectives, part-aware models such as SCOPS [15] and Leopart [12] promote the discovery of spatially coherent parts, while GroupViT [16] and TokenCut [31] show that semantic grouping can emerge from transformer tokenisation. DINO [19] further demonstrates that Vision Transformer attention provides an unsupervised saliency prior correlated with object and part structure, motivating attention-guided grouping.

More recent unsupervised grouping and segmentation methods extend these ideas. LOST [32] shows that object localization can emerge directly from self-supervised ViT features through attention-guided region extraction, while FreeSOLO [33] and U2Seg [34] generalise grouping to class-agnostic instance and universal image segmentation. These methods highlight the strength of transformer representations for dense grouping without supervision, but operate primarily on static images and do not enforce temporal coherence across frames. Dense unsupervised video segmentation [35] begins to address temporal consistency, yet remains pixel-centric and does not leverage mid-level part representations or attention-guided token selection.

We build on these insights in two ways: (i) instead of clustering all tokens or pixels, we exploit ViT attention to selectively retain salient spatial tokens from a frozen encoder using an adaptive budget that preserves coverage across resolutions; and (ii) we extend grouping from static images to videos by enforcing cross-temporal consistency of soft part assignments through a lightweight alignment mechanism.

## 2.2 Temporal Self-Supervision in Video Representation Learning.

Temporal continuity in video provides a rich source of self-supervision [36]. Early approaches use surrogate objectives such as frame ordering (Shufle and Learn [37], Arrow of Time [38]) and future prediction [39]. Cycle-consistency later emerged as a powerful constraint: TimeCycle [40] enforces that correspondences traced forward in time return to their origin, while subsequent methods operationalise correspondence via contrastive or tracking-based objectives, including CycleContrast [41], spacetime random walks [42] and memory-augmented tracking (MAST) [43]. Time-tuning [7] further adapts image-pretrained features to unlabeled video using temporal alignment losses.

Despite their efectiveness, many of these methods either (i) operate on global embeddings that obscure fine-grained structure, or (ii) rely on dense warping or optical flow, which can be fragile under fast motion, occlusion, or low texture and computationally expensive at scale. Recent correspondence-based models such as TAPIR [44] demonstrate strong long-range, flow-free tracking through recurrent refinement, while Deep ViT Feature Descriptors [45] show that dense correspondences can emerge from self-supervised ViTs without explicit motion modeling. However, these methods still operate on densely sampled patches and do not explicitly constrain the temporal stability of part-level structures.

In contrast, we pursue token-level temporal supervision. Guided by attentionderived saliency, we restrict learning to a compact, adaptively selected set of informative tokens and align their soft part assignments across time. This design avoids dense warping, relies on lightweight cosine-similarity matching, and supports a multi-∆t temporal pyramid with match-rate control. Rather than competing with dense correspondence methods such as TAPIR or ViT-based tracking, our formulation targets a lightweight, mid-level alternative in which part-like token groups—rather than pixels or dense patches—serve as the unit of temporal alignment. We treat attention-derived saliency (e.g., [CLS] attention) as a heuristic prior rather than a definitive foreground indicator, acknowledging its limitations for small objects and cluttered scenes.

## 2.3 Part-Level Representation Learning.

Part-level representations provide semantically meaningful, generalisable units that are robust to occlusion, deformation, and viewpoint change [13, 46]. In images, SCOPS [15] and Leopart [12] discover recurring part structure through spatial grouping. In videos, temporal alignment of latent structures—such as cycle-consistency in TimeCycle [40] and neural-surface-based PartDistillation [14]—suggests that midlevel components can persist across time. Dense tracking works such as TAPIR [44] and Deep ViT Correspondences [45] further indicate that temporally stable mid-level features can emerge without flow, though they do not explicitly cluster tokens into part-like groups.

We extend this line of work by integrating transformer-derived token saliency with a temporal clustering loss. Salient tokens, selected via [CLS] attention from a frozen encoder, are softly clustered into parts and aligned across frames using a symmetric KL objective. This encourages temporally stable groupings without assuming that clusters always correspond to fully interpretable semantic parts.

## 2.4 Foundation Models and Saliency-Guided Learning.

Large pretrained Vision Transformers, including DINO [19], SAM [20], and SAM2 [18], exhibit strong transfer to dense prediction tasks due to hierarchical token representations and global attention. These models often show emergent grouping of semantically related regions, providing a natural cue for unsupervised segmentation. Leveraging [CLS] attention [47] as a saliency signal enables extraction of informative tokens without training a separate saliency network. Our method adopts this principle using a frozen SAM2 encoder. We emphasise that freezing the encoder is a design choice motivated by eficiency and generalisation rather than conceptual novelty; ablations in Sec. 4.3 examine frozen versus fine-tuned variants.

## 2.5 Token-Level Consistency and Loss Design.

Temporal stability in learned representations has been pursued through contrastive objectives (InfoNCE [48]), cycle-consistency [41], and patch-tracking losses [42], often requiring careful negative sampling or memory banks. We instead adopt a symmetric KL divergence [12] between matched token clusters, avoiding negative sampling while enforcing bidirectional consistency. Prior work suggests that symmetric divergences can mitigate directional bias, though their advantages over one-sided KL, cross-entropy, or contrastive losses depend on the alignment setting; we therefore include ablations to clarify their behaviour in our framework.

Prior SSL video methods emphasise pixel- or object-level propagation, optical flow, or external memory. In contrast, our method treats parts as the unit of temporal alignment, unifying attention-guided token selection with cross-temporal clustering and symmetric-KL agreement on a frozen backbone. With adaptive token budgeting and a multi-∆t training scheme, the pipeline remains label-free and decoder-free while scaling across resolutions and motion regimes.

## 3 Methodology

## 3.1 Overview and Notation

Our objective is to learn temporally consistent and semantically coherent part-level representations of objects in videos in a purely self-supervised manner. Reasoning at the level of constituent parts—such as wheels, limbs, or articulated components—has been shown to provide stronger inductive biases than whole-object or pixel-level formulations, particularly under occlusion and deformation [15, 49]. This perspective complements our earlier CAMVOS framework [50], which emphasised contextual and long-term memory for video object segmentation. In contrast, the present work adopts an explicitly unsupervised, part-centric formulation that discovers latent part groupings and enforces their temporal consistency across a video sequence.

Table 1: Summary of notation used in Sec. 3.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $H, W$ </td><td>Spatial height and width of feature map</td></tr><tr><td> $N$ </td><td>Number of spatial tokens ( $N = HW$ )</td></tr><tr><td> $D$ </td><td>Token embedding dimension</td></tr><tr><td> $\mathbf{X}_{t} \in \mathbb{R}^{N \times D}$ </td><td>Token embeddings for frame  $I_{t}$ </td></tr><tr><td> $\boldsymbol{\alpha}_{t} \in \mathbb{R}^{N}$ </td><td>Saliency prior from [CLS] attention</td></tr><tr><td> $\mathcal{S}_{t}$ </td><td>Indices of selected tokens after adaptive sampling</td></tr><tr><td> $k_{t} = |\mathcal{S}_{t}|$ </td><td>Number of selected tokens</td></tr><tr><td> $\mathbf{P}_{t}(i) \in \Delta^{K}$ </td><td>Soft part distribution for token  $i$ </td></tr><tr><td> $K$ </td><td>Number of part prototypes</td></tr><tr><td> $\mathcal{M}_{t,\Delta}$ </td><td>Mutual matches between frames  $t$  and  $t + \Delta$ </td></tr></table>

At a high level, the framework consists of four stages. First, a frozen encoder produces dense spatial tokens together with an attention-derived saliency prior. Second, an adaptive token selection strategy extracts a compact yet semantically informative subset of tokens. Third, a lightweight MLP-based clustering head maps each selected token to a soft distribution over K latent parts. Finally, a temporal consistency objective aligns these part distributions across multiple time ofsets.

For an input frame $I _ { t } ,$ the encoder outputs a feature map $\mathbf { F } _ { t } \in \mathbb { R } ^ { H \times W \times D }$ , which is flattened into $N = H \times W$ spatial tokens $\mathbf { \bar { X } } _ { t } \in \mathbb { R } ^ { N \times D }$ . A saliency score $\alpha _ { t } ^ { i }$ is assigned to each token via the encoder’s [CLS] attention, forming the vector ${ \pmb { \alpha } } _ { t } \in \mathbb { R } ^ { N }$ . All notation is summarised in Table 1.

An adaptive sampling rule (Sec. 3.4) selects a subset $S _ { t } \subseteq \{ 1 , \dots , N \}$ of size $k _ { t } .$ Each selected token $i \in S _ { t }$ is mapped to a soft part distribution $\mathbf { P } _ { t } ( i ) \in \Delta ^ { K }$ , where K denotes the number of latent part prototypes. Temporal correspondence is established via cosine similarity between tokens from frames t and $t + \Delta$ , yielding the mutual match set $\mathcal { M } _ { t , \Delta } \subseteq \mathcal { S } _ { t } \times \mathcal { S } _ { t + \Delta }$

To ground the notation, consider a concrete example. A $2 2 4 \times 2 2 4$ frame yields $N = 1 9 6$ tokens using a patch size of 16. The saliency prior selects $k _ { t } = 3 2$ informative tokens. After clustering, each token is assigned a K = 8-dimensional part distribution. Matching these tokens with those from frame t+3 produces 24 mutual correspondences, on which the symmetric distributional alignment loss is computed.

Our design is guided by three principles. First, eficiency through adaptive saliency: selecting only salient and spatially diverse tokens reduces computation while preserving semantic coverage. Second, stability through part-level clustering: grouping tokens into soft parts yields representations that are more robust to occlusion, motion, and viewpoint change than pixel-level alignment. Third, temporal consistency via symmetric alignment: we employ symmetric KL divergence as a simple, bidirectional distribution-matching objective. Together, these principles define a lightweight and fully unsupervised pipeline for video object segmentation.

![](images/6a9f32ad31bd7591cda797379f6950745e88ecfd9844f19811e86fa16f00f298.jpg)  
Fig. 1: Overview of the proposed $\mathrm { C T C ^ { 2 } }$ framework. Consecutive frames It and ${ \mathbf { I } } t + \Delta t$ are encoded by a frozen SAM2 ViT backbone. The [CLS] attention map guides adaptive token selection, retaining only the most salient and spatially diverse regions. A lightweight MLP head clusters these tokens into soft part assignments. Cross-temporal matching aligns part distributions via cosine similarity across mul tiple temporal ofsets, while a saliency-weighted symmetric KL divergence enforces temporal consistency.

## 3.2 Encoder and Saliency Prior

We adopt a frozen SAM2 encoder as the feature extractor, motivated by its strong ability to capture high-level semantic information across diverse visual domains. For each frame $I _ { t } ,$ the encoder produces a dense feature map $\mathbf { F } _ { t } \in \mathbb { R } ^ { H \times W \times D }$ , which is flattened into $N = H \times W$ spatial tokens $\mathbf { X } _ { t } = \{ \mathbf { x } _ { t } ^ { i } \} _ { i = 1 } ^ { N }$ with $\mathbf { x } _ { t } ^ { i } \in \mathbb { R } ^ { D }$

In addition to spatial embeddings, the encoder exposes a global [CLS] token whose final-layer attention provides an importance weight for each spatial location. We treat this signal as a saliency prior ${ \pmb { \alpha } } _ { t } \in \mathbb { R } ^ { N }$

The [CLS] attention ofers two practical advantages. First, it provides a trainingfree mechanism for highlighting semantically meaningful regions without introducing an additional saliency network or optical-flow estimator. Second, by keeping the encoder frozen, optimisation is confined to the lightweight clustering head and temporal alignment modules, reducing training complexity.

We note that this saliency prior is an imperfect heuristic: [CLS] attention may be unreliable in cluttered scenes, for small objects, or under domain shift. To mitigate this, our adaptive selection module combines saliency weighting with spatial diversity constraints, ensuring broad coverage even when the prior is noisy.

## 3.3 Frame Encoding and Attention-Derived Saliency

Given a video frame $\mathbf { I } _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ , we extract patch-level embeddings using a frozen SAM2 ViT encoder [51]. The frame is divided into $\begin{array} { r } { { { N } } = { \frac { H } { P } } \cdot { \frac { W } { P } } } \end{array}$ patches of size $P { \times } P$ (with P =16), producing a feature map $\mathbf { F } _ { t } ~ \in ~ \mathbb { R } ^ { N \times D }$ with embedding dimension $D { = } 7 6 8$ . Tokens are flattened in raster order (top-left to bottom-right), yielding $\mathbf { X } _ { t } = \{ \mathbf { x } _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }$

Freezing the encoder preserves spatial priors learned from large-scale segmentation corpora [20] and reduces training cost. To localise salient regions without supervision, we leverage the final-layer [CLS] attention, which has been shown to correlate with object- and part-level semantics in Vision Transformers [14, 19, 52]. Let $ { \mathbf { q } } _ { \mathrm { [ C L S ] } } \in \mathbb { R } ^ { 1 \times d _ { k } }$ and $\mathbf { K } \in \mathbb { R } ^ { N \times d _ { k } }$ denote the [CLS] query and token keys. Scaled dot-product attention computes

$$
\mathbf {A} _ {t} = \mathrm{softmax} \left(\frac {\mathbf {q} _ {[ \mathrm{CLS} ]} \mathbf {K} ^ {\top}}{\sqrt {d _ {k}}}\right) \in \mathbb {R} ^ {1 \times N},\tag{1}
$$

averaged across attention heads. As a softmax distribution, $\begin{array} { r } { \sum _ { i } \mathbf { A } _ { t } ^ { \left( i \right) } = 1 } \end{array}$

Flattening yields ${ \pmb { \alpha } } _ { t } \in \mathbb { R } ^ { N }$ , a training-free saliency prior in which larger values often correspond to semantically meaningful regions. Because attention quality may degrade under clutter or domain shift, spatial diversity is enforced in Sec. 3.4 to prevent omission of important regions when saliency is imperfect.

## 3.4 Adaptive Token Selection

Although the SAM2 encoder produces a dense grid of N tokens, propagating all tokens is computationally ineficient and often redundant, as many correspond to low semantic background regions. Inspired by token-pruning methods [53–55], we adopt an adaptive strategy that selects a compact, informative subset.

Let ${ \bf A } _ { t }$ <sub>t</sub> denote the saliency vector from Eq. 1. Sorting entries in descending order yields indices $\pi ( 1 ) , \ldots , \pi ( N )$ . We select the smallest prefix satisfying

$$
\sum_ {m = 1} ^ {| \mathcal {S} _ {t} |} \mathbf {A} _ {t} ^ {(\pi (m))} \geq p, \quad p \in [ 0. 8 0, 0. 9 0 ].\tag{2}
$$

Since ${ \bf A } _ { t }$ sums to $1 , p$ directly controls the retained saliency mass.

To prevent degenerate selections, we enforce

$$
k _ {\mathrm{min}} \leq | \mathcal {S} _ {t} | \leq k _ {\mathrm{max}},
$$

with $k _ { \mathrm { m i n } } \mathrm { = } 2 4$ and $k _ { \mathrm { m a x } } { = } 1 2 8$ . This guarantees explicit lower and upper bounds on the adaptive budget.

Saliency-only selection may collapse spatially by favouring adjacent patches. To ensure coverage, we partition the token grid into $B \times B$ non-overlapping cells (typically $B { = } 4 )$ . From each non-empty cell, the highest-saliency token is added to a diversity set $\mathcal { G } _ { t } .$ , guaranteeing inclusion of spatially distinct regions. The remaining budget is filled using the top-p ordering, skipping duplicates.

The final selection size follows an explicit termination rule:

$$
| \mathcal {S} _ {t} | = \min \Bigl (k _ {\max}, \max (k _ {\min}, m ^ {\star}, | \mathcal {G} _ {t} |) \Bigr),
$$

where $m ^ { \star }$ is the smallest index at which cumulative saliency exceeds $p .$ . Training proceeds only once this cardinality is reached.

This hybrid top-p plus diversity strategy balances eficiency with semantic coverage and remains stable across resolutions and video domains.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Adaptive Token Selection via Top-p Saliency and Grid Diversity

Input: Saliency  $A_{t}$ ; patch grid (H/P, W/P); parameters p,  $k_{min}$ ,  $k_{max}$ ; grid size B.

1: Sort indices  $\pi$  by  $A_{t}$  (descending).

2: Compute smallest  $m^{\star}$  such that  $\sum_{m=1}^{m^{\star}}\mathbf{A}_{t}^{(\pi(m))}\geq p$ .

3:  $S\leftarrow\{\pi(1),\ldots,\pi(m^{\star})\}$ .

4: Grid Diversity: Partition grid into  $B\times B$  cells; for each cell with tokens, add highest-saliency token to G.

5: Initialize  $S_{t}\leftarrow G$ .

6: Sequentially add tokens from S not already in  $S_{t}$  until  $|S_{t}|=\min(k_{\max},\max(k_{\min},m^{\star},|G|))$ .

Output:  $S_{t}$
</div>

## 3.5 Soft Part Clustering Head

Selected tokens are mapped to part-level representations using a lightweight two-layer MLP. For each token embedding $\mathbf { f } _ { t } ^ { ( i ) } \in \mathbb { R } ^ { D }$ from the adaptive set $\tilde { \mathbf { F } } _ { t }$ , we compute

$$
\mathbf {C} _ {t} ^ {(i)} = \mathrm{softmax} \big (W _ {2}   \sigma (W _ {1}   \mathbf {f} _ {t} ^ {(i)}) \big)   \in   \mathbb {R} ^ {K},\tag{3}
$$

where $W _ { 1 } \in \mathbb { R } ^ { D \times d _ { h } } , W _ { 2 } \in \mathbb { R } ^ { d _ { h } \times K }$ , and σ denotes a non-linearity (ReLU or GELU). Each $\mathbf { C } _ { t } ^ { ( i ) }$ represents a soft distribution over K latent parts, with K selected via validation (Sec. 5.2). In practice, $K \in [ 1 2 , 2 4 ]$ provides a good trade-of between granularity and temporal stability.

Soft assignments are motivated by prior work in unsupervised part discovery, which shows that hard clustering can fragment object regions and produce unstable trajectories, particularly near occlusion boundaries or under appearance change [12, 15, 49]. By contrast, soft distributions preserve uncertainty and allow ambiguous tokens to interpolate between parts, leading to more robust temporal align ment. Related clustering-based grouping frameworks [16, 56] similarly demonstrate that distributional assignments help prevent collapse and encourage balanced part usage.

Compared to the heavier decoders used in semi-supervised VOS [3, 57], our clustering head remains deliberately lightweight. Most representational capacity arises from attention-guided token selection and temporal consistency (Sec. 3.6), allowing us to avoid large-capacity modules. As a result, accuracy gains stem from the part-centric formulation rather than increased parameter count, consistent with eficiency-oriented VOS pipelines [42, 43].

## 3.6 Temporal Matching and Multi-Ofset Supervision

Spatial grouping alone is insuficient for video object segmentation; part assignments must remain stable over time despite motion, occlusion, and viewpoint change. Prior self-supervised approaches enforce temporal coherence using cycle-consistency [40, 41], nearest-neighbour tracking [43], or contrastive alignment [42, 48]. Building on these ideas, we establish temporal consistency by matching adaptively selected tokens across frames in embedding space.

Let $\ddot { \mathbf { F } } _ { t }$ and $\mathbf { F } _ { t + \Delta t }$ denote the selected tokens from frames t and $t + \Delta t .$ , with sizes $k _ { t }$ and $k _ { t + \Delta { t } }$ . Embeddings are L -normalised, and cosine similarity defines nearestneighbour correspondences. For token i at frame $t ,$

$$
\mathcal {N} _ {t \to t + \Delta t} (i) = \arg \max _ {j \in \{1, \dots , k _ {t + \Delta t} \}} \frac {\mathbf {f} _ {t} ^ {(i)} \cdot \mathbf {f} _ {t + \Delta t} ^ {(j)}}{\| \mathbf {f} _ {t} ^ {(i)} \| \| \mathbf {f} _ {t + \Delta t} ^ {(j)} \|}.\tag{4}
$$

To improve robustness, we retain only mutual nearest neighbours whose similarity exceeds a threshold $\delta \in [ 0 . 3 , 0 . 6 ]$ :

$$
\mathcal {M} _ {\Delta t} = \left\{(i, j) \mid j = \mathcal {N} _ {t \rightarrow t + \Delta t} (i), i = \mathcal {N} _ {t + \Delta t \rightarrow t} (j), \operatorname{sim} (i, j) \geq \delta \right\}.\tag{5}
$$

This symmetric rule filters noisy correspondences, consistent with prior correspondence-based SSL [40, 43].

A single temporal ofset may not capture both short-term motion and longerterm structural coherence. We therefore employ a multi-ofset strategy with strides $\mathcal { S } = \{ 1 , 2 , 4 , 8 \}$ and compute $\mathcal { M } _ { \Delta t }$ for each $\Delta t \in S$ . Because matching reliability decreases with increasing stride, we measure ofset quality using the match rate

$$
r (\Delta t) = \frac {| \mathcal {M} _ {\Delta t} |}{\min (k _ {t} , k _ {t + \Delta t})}.\tag{6}
$$

Ofsets satisfying $r ( \Delta t ) \geq r _ { \operatorname* { m i n } }$ (with $r _ { \mathrm { m i n } } \in [ 0 . 5 , 0 . 7 ] )$ are retained,

$$
\mathcal {S} _ {t} = \{\Delta t \in \mathcal {S} \mid r (\Delta t) \geq r _ {\min} \}.\tag{7}
$$

If no ofset meets this criterion, we fall back to the ofset with the highest match rate. This ensures that temporal supervision is driven by reliable correspondences rather than noise, particularly under fast motion or low texture.

Active ofsets are combined using exponentially decaying weights,

$$
w _ {\Delta t} \propto \gamma^ {\Delta t}, \qquad \gamma \in [ 0. 6, 0. 8 ], \qquad \sum_ {\Delta t \in \mathcal {S} _ {t}} w _ {\Delta t} = 1,\tag{8}
$$

emphasising short-term alignment while progressively incorporating longer-range consistency. This temporal curriculum stabilises training without relying on optical flow [58] or explicit motion cues.

As a continuation of the running example, matching 32 selected tokens from frame t with those from frame $t { + 3 }$ yields approximately 24 mutual correspondences, corresponding to a match rate of $r ( 3 ) \approx 0 . 7 5$

## 3.7 Saliency-Weighted Symmetric KL Consistency

Given the mutual correspondences $\mathcal { M } _ { \Delta t } ,$ we enforce temporal consistency by aligning the corresponding part distributions. Let $\mathbf { P } _ { t } ^ { ( i ) } \in \Delta ^ { K }$ and $\mathbf { P } _ { t + \Delta t } ^ { ( j ) } \in \Delta ^ { \dot { K } }$ denote the soft part assignments for a matched pair $( i , j )$ . Their agreement is measured using a symmetric KL divergence,

$$
\mathrm{SKL} \big (\mathbf {P} _ {t} ^ {(i)}, \mathbf {P} _ {t + \Delta t} ^ {(j)} \big) = \frac {1}{2} \left[ \mathrm{KL} (\mathbf {P} _ {t} ^ {(i)} \parallel \mathbf {P} _ {t + \Delta t} ^ {(j)}) + \mathrm{KL} (\mathbf {P} _ {t + \Delta t} ^ {(j)} \parallel \mathbf {P} _ {t} ^ {(i)}) \right].\tag{9}
$$

Not all correspondences are equally informative. We therefore weight each pair by the geometric mean of its saliency scores,

$$
\omega_ {i j} = \big (\alpha_ {t} ^ {(i)} \alpha_ {t + \Delta t} ^ {(j)} \big) ^ {1 / 2},\tag{10}
$$

and compute a normalised loss for each ofset,

$$
\mathcal {L} _ {\mathrm{CTC} ^ {2}} (\Delta t) = \frac {1}{\sum_ {(i , j) \in \mathcal {M} _ {\Delta t}} \omega_ {i j}} \sum_ {(i, j) \in \mathcal {M} _ {\Delta t}} \omega_ {i j} \operatorname{SKL} \left(\mathbf {P} _ {t} ^ {(i)}, \mathbf {P} _ {t + \Delta t} ^ {(j)}\right).\tag{11}
$$

Consistency is aggregated across all active ofsets $S _ { t }$ using weights $w _ { \Delta t }$

$$
\mathcal {L} _ {\mathrm{cons}} = \sum_ {\Delta t \in \mathcal {S} _ {t}} w _ {\Delta t} \mathcal {L} _ {\mathrm{CTC} ^ {2}} (\Delta t).\tag{12}
$$

Contrastive objectives such as InfoNCE [48] or MoCo [26] require careful negative construction and often emphasise global separation rather than fine-grained alignment. Cycle-consistency losses [40, 41] promote temporal stability but are directional and may accumulate drift. In contrast, symmetric KL provides a simple bidirectional distribution-matching objective that is well suited to soft part assignments: it preserves uncertainty, discourages drift through mutual agreement, and penalises collapsed distributions [16, 56].

Saliency weighting further focuses supervision on tokens likely to correspond to object parts, reducing the influence of background regions or spurious matches. Together, these components instantiate our inductive bias of part-level temporal consistency in a lightweight, label-free manner without negative sampling, pseudo-label propagation, or optical-flow supervision.

For continuity with the running example, the symmetric KL loss in Eq. 11 is computed on the ∼24 reliable correspondences obtained when matching frame t with frame t+3.

## 3.8 Regularisation and Collapse Prevention

Temporal consistency provides strong supervision, but clustering can still drift toward degenerate optima without additional constraints. In practice, we observe two com mon failure modes: (i) assignments collapse to a dominant part, and (ii) per-token distributions become overly uniform, erasing semantic structure. To stabilise learning, we introduce two lightweight regularisers on the soft part distributions, adding negligible overhead.

To discourage overly flat predictions, we penalise the entropy of per-token distributions:

$$
\mathcal {L} _ {\mathrm{conf}} = \frac {1}{\sum_ {t} k _ {t}} \sum_ {t} \sum_ {i = 1} ^ {k _ {t}} H (\mathbf {P} _ {t} ^ {(i)}), \qquad H (\mathbf {P} _ {t} ^ {(i)}) = - \sum_ {c = 1} ^ {K} P _ {t, c} ^ {(i)} \log P _ {t, c} ^ {(i)}.\tag{13}
$$

This encourages sharper assignments while still allowing uncertainty where warranted (e.g., near boundaries). Similar entropy-based sharpening appears in unsupervised part discovery [15, 49] and clustering-based representation learning [59, 60].

To prevent collapse to a small subset of parts, we promote balanced usage across a batch. Let

$$
\bar {\mathbf {p}} = \frac {1}{\sum_ {t} k _ {t}} \sum_ {t} \sum_ {i = 1} ^ {k _ {t}} \mathbf {P} _ {t} ^ {(i)}
$$

denote the batch-averaged distribution over $K$ clusters. We penalise deviation from the uniform prior $\begin{array} { r } { { \bf u } = \frac { 1 } { K } { \bf 1 } } \end{array}$ via

$$
\mathcal {L} _ {\mathrm{bal}} = \operatorname{KL} (\bar {\mathbf {p}} \| \mathbf {u}) = \sum_ {c = 1} ^ {K} \bar {p} _ {c} \log (K \bar {p} _ {c}).\tag{14}
$$

Related balanced-assignment constraints are used in SwAV [61] and GroupViT [16]; here, this term discourages trivial dominance and encourages diverse part activation.

Together, ${ \mathcal { L } } _ { \mathrm { c o n f } }$ and $\mathcal { L } _ { \mathrm { b a l } }$ complement the temporal consistency loss by steering optimisation away from degenerate solutions. In practice, they stabilise training and yield more coherent part assignments.

## 3.9 Training Objective and Complexity

The overall objective combines temporal consistency with the two regularisers:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{cons}} + \lambda_ {\mathrm{conf}} \mathcal {L} _ {\mathrm{conf}} + \lambda_ {\mathrm{bal}} \mathcal {L} _ {\mathrm{bal}}.\tag{15}
$$

We set $( \lambda _ { \mathrm { c o n f } } , \lambda _ { \mathrm { b a l } } ) = ( 0 . 1 , 1 . 0 )$ , balancing per-token sharpening and global part utilisation. Gradients are propagated only through the clustering head, while the encoder remains frozen, keeping training stable and memory-eficient.

Let $k _ { \mathrm { m a x } }$ be the maximum token budget and |S| the number of temporal ofsets. Mutual nearest-neighbour matching requires $\mathcal { O } ( \vert \boldsymbol { S } \vert k _ { \mathrm { m a x } } ^ { 2 } )$ similarity computations. The clustering head incurs

$$
\mathcal {O} (k _ {\mathrm{max}} D d _ {h} + k _ {\mathrm{max}} d _ {h} K),
$$

where D is the embedding dimension and $d _ { h }$ is the MLP hidden width. With typical settings $( k _ { \mathrm { m a x } } \le 1 2 8 , K \le 2 4 , | \boldsymbol { S } | \le 4 )$ ), this cost is negligible relative to the frozen SAM2 encoder.

Flow-based VOS [58] and memory-centric architectures [57, 62] operate at the pixel level or maintain large external memories, increasing compute and storage. Cycle-consistency approaches such as CRW [42], MAST [43], and TCC [40] often scale quadratically with dense grids or rely on long-range correspondence chains. By contrast, we enforce temporal consistency only on an adaptively selected subset of informative tokens, retaining semantic coverage while improving eficiency. This makes the approach suitable for large-scale, label-free video training regimes where compute is a primary constraint.

## 4 Experimental Setup

We evaluate our method on three standard video object segmentation (VOS) benchmarks. DAVIS-2016 [2] contains 50 single-object sequences, and we report results on the oficial validation split. DAVIS-2017 [24] extends DAVIS to 150 multi-object videos, with evaluation on the validation split. YouTube-VOS [3] is a large-scale multi-object dataset with diverse object categories and motion patterns; we report results on the oficial validation split. All experiments follow the original evaluation protocols defined by each benchmark.

Following established practice in unsupervised part discovery and label-free segmentation [12, 15], the annotated first frame of each sequence is used only to compute a permutation between discovered clusters and ground-truth instance identities via Hungarian matching. This permutation is fixed for the remainder of the sequence and is applied solely for evaluation. Ground-truth masks are never used during training or test-time adaptation, distinguishing our setting from semi-supervised VOS methods that receive the first-frame mask as input.

We note that fixing the permutation from the first frame may partially obscure temporal identity drift in long sequences. To account for this limitation, we comple ment standard metrics with targeted ablations, cross-dataset robustness analysis, and qualitative evaluations of part stability and failure cases (Secs. 5.2, 5.3, and 5.4).

We report region similarity J , boundary accuracy ${ \mathcal F } ,$ and their mean $\mathcal { G } = ( \mathcal { I } +$ $\mathcal { F } ) / 2$ . For multi-object datasets, metrics are averaged per instance and per frame before aggregation at the sequence level, following the oficial DAVIS and YouTube-VOS evaluation protocols.

All experiments use the SAM2 ViT encoder [20, 51] as a frozen backbone, with gradients propagated only through the lightweight clustering head. Input frames are resized to 224 × 224. Salient tokens are selected using the adaptive top-p strategy described in Sec. 3.4, with $\mathrm { \Delta } p { = } 0 . 8 5$ , bounds $k _ { \mathrm { m i n } } \mathrm { = } 2 4$ and $k _ { \mathrm { m a x } } { = } 1 2 8$ , and a grid-diversity prior of B=4 cells.

The clustering head is implemented as a two-layer MLP with hidden dimension $d _ { h } = 5 1 2$ , followed by a softmax over K part categories. We use K=16 throughout, and evaluate alternative values in ablation studies. Temporal correspondences are established using $L _ { 2 } .$ -normalised embeddings and cosine similarity, followed by mutual nearest-neighbour matching with threshold δ=0.4. Temporal ofsets are drawn from {1, 2, 4, 8} and filtered using the match-rate controller (Sec. 3.6) with threshold $r _ { \mathrm { m i n } } { = } 0 . 6$ . Ofsets are weighted as $w _ { \Delta t } \propto \gamma ^ { \Delta t }$ , where $\gamma$ is annealed linearly from 0.8 to 0.6 during training.

Training is performed using AdamW with learning rate $1 \times 1 0 ^ { - 3 }$ , weight decay $1 \times 1 0 ^ { - 4 }$ , and batch size of 16 frames (8 temporal pairs). We train for 120k iterations using standard spatial augmentations, including random resize and crop, color jitter, and horizontal flipping. Temporal order is preserved throughout training; temporal reversal is not used, consistent with the original submission.

At inference, each frame is independently encoded, salient tokens are selected, and soft part assignments are produced by the clustering head. The Hungarian permutation computed from the first annotated frame is applied post hoc for evaluation only. All experiments are conducted with a fixed random seed (42) on a single NVIDIA V100 GPU (16 GB). Training time and throughput are reported separately in Sec. 5.7.

Table 2: Results on the DAVIS-2017 val set under the unsupervised protocol. We report region similarity ${ \mathcal { I } } ,$ , boundary accuracy ${ \mathcal F } ,$ and their mean $\mathcal { G } = ( \mathcal { I } { + } \mathcal { F } ) / 2$ Higher values indicate better segmentation quality. Bold highlights the best performance among compared methods.

<table><tr><td>Method</td><td> $\mathcal{J}$  (Mean)</td><td>F (Mean)</td><td> $\mathcal{G}$  (Mean)</td></tr><tr><td>SOLV [63]</td><td>0.301</td><td>-</td><td>-</td></tr><tr><td>OCLR [64]</td><td>0.346</td><td>-</td><td>-</td></tr><tr><td>Video Colorization [6]</td><td>0.327</td><td>0.346</td><td>0.336</td></tr><tr><td>TimeT* [7]</td><td>0.442</td><td>0.358</td><td>0.400</td></tr><tr><td>SMTC [65]</td><td>0.446</td><td>0.364</td><td>0.405</td></tr><tr><td>TimeCycle [8]</td><td>0.419</td><td>0.394</td><td>0.407</td></tr><tr><td>BA [21]</td><td>0.392</td><td>0.486</td><td>0.439</td></tr><tr><td>CorrFlow [5]</td><td>0.471</td><td>0.499</td><td>0.485</td></tr><tr><td>TripleNet [66]</td><td>0.504</td><td>0.513</td><td>0.509</td></tr><tr><td>Ours</td><td>0.539</td><td>0.568</td><td>0.554</td></tr></table>

## 5 Results

We evaluate $\mathrm { C T C ^ { 2 } }$ on standard video object segmentation benchmarks under the fully self-supervised (zero-shot) setting. We first report quantitative results on DAVIS-2016, DAVIS-2017, and YouTube-VOS, followed by ablation studies and analyses of robustness, cross-dataset generalisation, and semi-supervised performance.

## 5.1 Zero-Shot Self-Supervised VOS Results

DAVIS-2017 is a challenging benchmark due to frequent multi-object interactions, heavy occlusion, and background clutter. Table 2 reports results on the DAVIS-2017 val split under the fully unsupervised (zero-shot) protocol, where the first annotated frame is used only to derive a fixed cluster–to–instance permutation for evaluation.

Under this setting, $\mathrm { C T C ^ { 2 } }$ achieves a mean G score of 0.554, outperforming prior self-/unsupervised methods based on pixel-level propagation, cycle-consistency, or dense correspondence learning. Gains are consistent across both region similarity $( \mathcal { I } = 0 . 5 3 9 )$ and boundary accuracy $( \mathcal { F } = 0 . 5 6 8 )$ , indicating that part-level temporal consistency stabilises both object extent and contour localisation in multi-object scenes.

Compared to correspondence-based approaches such as CorrFlow [5] and TripleNet [66], which propagate dense pixel or patch matches across time, $\mathrm { C T C ^ { 2 } }$ enforces alignment at the level of soft part distributions over a compact, adaptively selected token subset. This mid-level formulation reduces over-merging between nearby objects and limits identity drift during occlusion, without relying on optical flow, explicit memory banks, or decoder-heavy architectures.

Recent self-supervised methods incorporating transformer attention or semantic cues (e.g., SMTC [65] and BA [21]) also improve boundary localisation, but typically operate on dense token grids or require additional grouping heuristics. In contrast, our method combines attention-guided token selection with symmetric distributional alignment, focusing supervision on semantically salient regions while remaining computationally eficient.

We emphasise that gains on DAVIS-2017 should be interpreted in the context of the zero-shot protocol and frozen-backbone setting. Improvements primarily reflect increased temporal stability of discovered parts rather than full instance recovery, which remains an open challenge in unsupervised VOS.

From an eficiency perspective, CTC<sup>2</sup> operates at approximately 35 fps at 224×224 resolution on a single GPU, compared to the 20–25 fps typically reported by dense correspondence-based SSL methods such as MAST [43] and CRW [42]. This eficiency stems from adaptive token budgeting and the absence of dense propagation or motion estimation, rather than task-specific architectural optimisation.

Overall, the DAVIS-2017 results show that enforcing temporal consistency at the part level provides a robust and eficient alternative to pixel-level propagation for unsupervised multi-object video segmentation. Failure cases and robustness under challenging conditions are analysed further in Secs. 5.3 and 5.4.

Results on DAVIS-2016 (Table 3) show that the proposed method consistently outperforms prior self-supervised baselines in the single-object setting. Because DAVIS-2016 contains a single annotated foreground object per sequence, performance is primarily driven by boundary adherence and temporal label stability rather than multi-object separation

CTC<sup>2</sup> achieves the highest boundary accuracy $\left( \mathcal { F } \mathrm { = } 0 . 5 2 8 \right)$ and overall score $\left( \mathcal { G } \mathrm { = } 0 . 5 2 2 \right)$ among the compared self-supervised methods. These gains indicate that enforcing part-level temporal consistency stabilises fine-grained boundaries over time, reducing transient label fluctuations commonly observed in correspondence-based approaches. Notably, this improvement is obtained without optical flow, mask propagation, or test-time supervision, demonstrating that mid-level part alignment benefits both single-object and multi-object segmentation scenarios.

![](images/c5d6bf24b84143f11902519887021738298f19e0d91015fb224fba2c9b67a43b.jpg)  
(a)

![](images/dbeca9a98f82b92f2119d70bca4d84fa2def29bea1579f5e2651ee48f778fdf9.jpg)  
(b)

![](images/6706ab4310fc497219fe20e891dbf64a6a9ce3fe050a45002e57dedffcf50024.jpg)  
(c)

![](images/e9d7c83d3e4c7ef042f506e3815f9e06c3ea1ea87bf25eb35a7a627ce28112ec.jpg)  
(d)  
Fig. 2: Qualitative examples of part-level segmentation produced by CTC² under the zero-shot protocol. Each pair shows (left) the RGB frame and (right) the corresponding part-level clustering from [CLS]-guided token features. Distinct colors indicate automatically discovered semantic parts.

Table 3: The results on the DAVIS-2016 dataset, higher values indicate better segmentation quality. Bold highlights the best performance achieved across the compared methods.

<table><tr><td>Method</td><td> $\mathcal{J}$  (Mean)</td><td>F (Mean)</td><td> $\mathcal{G}$  (Mean)</td></tr><tr><td>CorrFlow [5]</td><td>0.471</td><td>0.499</td><td>0.485</td></tr><tr><td>TripleNet [66]</td><td>0.494</td><td>0.50</td><td>0.497</td></tr><tr><td>Ours</td><td>0.516</td><td>0.528</td><td>0.522</td></tr></table>

YouTube-VOS presents a substantially more challenging setting due to long sequences, diverse object categories, and frequent domain shift between training and evaluation data. As shown in Table 4, CTC<sup>2</sup> achieves an overall score of G=0.570, outperforming prior self-/unsupervised methods under the same zero-shot protocol.

Boundary accuracy remains strong (F=0.577) despite fast motion, background clutter, and unseen object categories. These results indicate that saliency-guided part clustering generalises beyond the more constrained DAVIS benchmarks, without relying on flow-based motion priors or dataset-specific adaptation. Supervised and semi-supervised methods (marked with †) are included solely as upper-bound references and are not directly comparable to our fully label-free setting.

Table 4: Results on the YouTube-VOS dataset. Higher values indicate better segmentation quality. Boldface highlights the best performance across methods.

<table><tr><td>Method</td><td>J (Mean)</td><td> $\mathcal{F}$  (Mean)</td><td>G (Mean)</td></tr><tr><td>Video Colorization [6]</td><td>0.314</td><td>0.345</td><td>0.329</td></tr><tr><td>CorrFlow [5]</td><td>0.489</td><td>0.515</td><td>0.502</td></tr><tr><td>OSMN [67]</td><td>0.485</td><td>0.539</td><td>0.512</td></tr><tr><td>MSK [68]</td><td>0.518</td><td>0.538</td><td>0.528</td></tr><tr><td>RGMP [69]</td><td>0.539</td><td>0.537</td><td>0.538</td></tr><tr><td>Ours</td><td>0.563</td><td>0.577</td><td>0.570</td></tr></table>

Table 5: Token selection ablation on DAVIS-2017 val. We compare a fixed token budget against adaptive Top-p attention selection and grid-diversity constraints. Proportional k scales the token count with image size N. Adaptive selection improves segmentation quality (G) and boundary accuracy (F) at similar throughput. Reported as mean ± std over three seeds.

<table><tr><td>Selector</td><td>Budget</td><td>fps</td><td> $\mathcal{G}$ </td><td> $\mathcal{F}$ </td></tr><tr><td>Fixed- $k$  (legacy)</td><td> $k=48$ </td><td>34.8</td><td> $0.523 \pm 0.001$ </td><td>0.528</td></tr><tr><td>Top- $p$ </td><td> $p=0.85$ </td><td>34.1</td><td> $0.525 \pm 0.001$ </td><td>0.530</td></tr><tr><td>Top- $p+grid$ </td><td> $p=0.85, B=4$ </td><td>33.9</td><td> $\mathbf{0.526} \pm \mathbf{0.001}$ </td><td>0.532</td></tr></table>

## 5.2 Ablation Studies

We conduct three ablation studies on DAVIS-2017 val to evaluate key design choices: token selection, temporal supervision, and saliency weighting.

Table 5 compares token selection strategies under identical settings. Replacing a fixed budget with Top-p selection (Eq. 2) yields a consistent improvement in G at nearly unchanged throughput, indicating that allocating tokens by attention mass preserves more informative regions than a hard cutof. Adding grid diversity (Alg. 1) further improves ${ \mathcal F } ,$ , supporting the hypothesis that coarse spatial coverage helps recover thin or peripheral structures. Finally, scaling the budget proportionally with resolution maintains accuracy relative to fixed-k, reinforcing that spatial coverage—not absolute token count—is the critical factor. All results are averaged over three seeds (42/43/44); the standard deviation of G is at most 0.002.

Table 6 analyses the efect of multi-ofset temporal supervision. Extending from a single ofset {1} to {1, 2, 4} improves $\mathcal { G }$ while retaining a suficient pool of reliable correspondences. The match-rate controller $( r _ { \operatorname* { m i n } } , \ \gamma )$ balances recall and stability: lower thresholds admit more matches at the cost of noise, whereas moderate values $( r _ { \mathrm { m i n } } { = } 0 . 6 , \ \gamma { \approx } 0 . 7 )$ yield the best trade-of. Including very long ofsets ({8}) slightly degrades performance, reflecting correspondence brittleness under large motion without flow. All results use similarity threshold $\delta { = } 0 . 4$ for mutual nearest-neighbour filtering and are averaged over three seeds (std ≤ 0.002 for G).

Table 6: Multi-ofset ablation on DAVIS-2017 val. We vary the set of temporal ofsets $( \Delta t \in \{ 1 , 2 , 4 \} )$ and the match-rate controller $( r _ { \operatorname* { m i n } } , \ \gamma )$ to assess the efect of longer-range supervision. “Matched $( \% ) ^ { , }$ denotes the proportion of reliable correspondences. Moderate ofsets achieve the best balance between accuracy (G) and match reliability. Results are mean ± std over three seeds.

<table><tr><td> $\mathcal{S}$ </td><td> $r_{\min}$ </td><td> $\gamma$ </td><td> $\mathcal{G}$ </td><td>Matched (%)</td></tr><tr><td> $\{1\}$ </td><td>-</td><td>-</td><td>0.518 ± 0.001</td><td>100</td></tr><tr><td> $\{1,2\}$ </td><td>0.6</td><td>0.8</td><td>0.524 ± 0.002</td><td>86</td></tr><tr><td> $\{1,2,4\}$ </td><td>0.6</td><td>0.7</td><td> $\mathbf{0.526} \pm \mathbf{0.001}$ </td><td>74</td></tr></table>

Table 7: Efect of saliency weighting in the temporal loss on DAVIS-2017 val. Incorporating [CLS]-attention weights emphasises confident foreground tokens, improving region similarity $( \mathcal { I } )$ and boundary accuracy $( \mathcal { F } )$ . Results are mean ± std over three seeds.

<table><tr><td>Objective</td><td> $\mathcal{J}$ </td><td> $\mathcal{F}$ </td><td> $\mathcal{G}$ </td></tr><tr><td> $\mathcal{L}_{\text{CTC}^2}$  (uniform)</td><td>0.519</td><td>0.528</td><td>0.523 ± 0.001</td></tr><tr><td> $\mathcal{L}_{\text{CTC}^2}^{\text{weighted}}$  (Eq. 9)</td><td>0.521</td><td>0.530</td><td>0.526 ± 0.001</td></tr></table>

Table 7 isolates the contribution of saliency weighting in the temporal consistency loss. Relative to uniform weighting, the saliency-weighted objective consistently improves both $\mathcal { I }$ and ${ \mathcal { F } } _ { : }$ , confirming that emphasising confident, foreground-biased tokens reduces identity drift. Results are averaged over three seeds, with standard deviation at most 0.001 for ${ \mathcal { G } } .$

Overall, these ablations demonstrate the complementarity of attention-guided token selection, multi-ofset temporal supervision, and saliency-aware consistency. Each component improves stability or boundary quality with negligible computational overhead, and their combination yields the strongest performance with a frozen backbone.

## 5.3 Robustness and Failure Analysis

Aggregate metrics provide a useful summary of performance but can obscure failure modes that are critical for understanding the robustness and limitations of self-supervised VOS. We therefore complement quantitative results with targeted analyses under challenging conditions, including fast motion, small objects, cluttered backgrounds, and domain shift.

Our framework relies on nearest-neighbour matching of adaptively selected tokens across time and does not employ optical flow or explicit motion modelling. As a result, correspondence reliability naturally degrades as inter-frame displacement increases. This behaviour is reflected by the match-rate controller (Sec. 3.6), where longer temporal ofsets yield fewer reliable matches. Empirically, moderate ofsets $( \Delta t \in \{ 1 , 2 , 4 \} )$ provide the best balance between temporal coverage and stability, whereas very large ofsets (∆t=8) are more susceptible to correspondence failure under fast motion or abrupt camera changes. These trends are consistent with prior correspondence-based SSL methods and motivate our reliability-aware multi-ofset supervision rather than unconditional long-range alignment.

The saliency prior derived from [CLS] attention is efective at highlighting dominant object regions, but can be less reliable for small objects, thin structures, or heavily cluttered scenes. In such cases, attention mass may concentrate on large or high-contrast regions, underrepresenting fine-grained parts. Our adaptive token selection mitigates this by combining saliency weighting with grid-based spatial diversity, ensuring retention of spatially distinct but less salient regions. Nevertheless, failures may occur when small objects occupy only a few patches or are visually indistinguishable from the background, leading to fragmented parts or absorption into neighbouring clusters.

We assess cross-dataset generalisation by training on one benchmark and evaluating on another without adaptation. Although our method degrades less than several prior self-supervised baselines, performance still drops under substantial domain shift, particularly for unfamiliar object categories or appearance statistics. Qualitative inspection shows that such failures typically coincide with altered saliency behaviour or reduced correspondence reliability, rather than catastrophic collapse of the clustering head. This suggests that freezing the backbone preserves transferable representations but does not eliminate sensitivity to dataset-specific biases in attention and token matching.

Following standard practice in unsupervised part discovery, we fix the clusterto-instance permutation using the first annotated frame. While this enables fair comparison with prior work, it may partially mask temporal identity drift in long sequences. We therefore examine part assignments over extended time spans and across occlusions in our qualitative analysis, revealing both stable part tracking and occasional identity switches when correspondence confidence is low. Addressing identity drift without any label-based anchoring remains an open challenge for fully unsupervised VOS.

Overall, the primary failure modes arise from (i) unreliable saliency for small or ambiguous objects, (ii) correspondence breakdown under extreme motion or low temporal continuity, and (iii) appearance shifts not well covered by the frozen backbone. Importantly, these failures tend to degrade performance gracefully rather than causing abrupt collapse, reflecting the stabilising efect of part-level clustering and symmetric distributional alignment. We view these limitations as inherent to lightweight, labelfree VOS and as promising directions for future work, such as integrating motion-aware matching or adaptive saliency refinement while preserving eficiency.

## 5.4 Interpretability and Temporal Stability of Discovered Parts

A central claim of $\mathrm { C T C ^ { 2 } }$ is that enforcing part-level temporal consistency leads to the emergence of stable and interpretable mid-level structures over time. We emphasise that “interpretability” in this context does not imply recovery of human-defined semantic parts (e.g., anatomically labelled components), but instead refers to the formation of temporally coherent latent groupings that act as a useful inductive bias for self-supervised video object segmentation. While qualitative visualisations provide intuition about spatial coherence, we additionally introduce quantitative measures to assess the temporal stability of discovered parts without relying on ground-truth part annotations.

We define Temporal Part Stability (TPS) as the consistency of soft part assignments across time for matched token pairs. Given a temporal ofset ∆t and a set of mutual nearest-neighbour correspondences $\mathcal { M } _ { \Delta t } \subseteq \mathcal { S } _ { t } \times \mathcal { S } _ { t + \Delta t }$ , let $\mathbf { C } _ { t } ^ { ( i ) } \in \Delta ^ { K }$ and $\mathbf { C } _ { t + \Delta t } ^ { ( j ) } \in \Delta ^ { K }$ denote the predicted part distributions for a matched pair $( i , j )$ . TPS is defined as:

$$
\operatorname{TPS} (\Delta t) = 1 - \frac {1}{| \mathcal {M} _ {\Delta t} |} \sum_ {(i, j) \in \mathcal {M} _ {\Delta t}} \frac {1}{2} \left[ \operatorname{KL} \left(\mathbf {C} _ {t} ^ {(i)} \| \mathbf {C} _ {t + \Delta t} ^ {(j)}\right) + \operatorname{KL} \left(\mathbf {C} _ {t + \Delta t} ^ {(j)} \| \mathbf {C} _ {t} ^ {(i)}\right) \right].\tag{16}
$$

Higher TPS values indicate more temporally stable and coherent part assignments. Unlike region-level segmentation metrics, TPS directly measures whether discovered part distributions remain consistent across time, independent of instance identity, pixel-level accuracy, or semantic labels.

Table 8 reports TPS on DAVIS-2017 val for diferent temporal ofsets. The symmetric KL objective consistently yields higher TPS than one-sided KL or cross-entropy losses, indicating increased resistance to temporal drift and cluster collapse. Notably, the stability gap widens at larger ofsets, confirming that bidirectional distribution matching is particularly efective under longer temporal gaps and appearance variation.

While TPS captures distributional agreement of soft part assignments, it does not explicitly assess whether the dominant part identity is preserved across time. To complement TPS, we introduce a simple and interpretable Cluster Identity Retention metric that evaluates hard label consistency for matched tokens.

For a matched pair $( i , j ) \in \mathcal { M } _ { \Delta t }$ , hard part labels are obtained as

$$
\hat {y} _ {t} ^ {(i)} = \arg \max _ {k} \mathbf {C} _ {t} ^ {(i)} [ k ], \quad \hat {y} _ {t + \Delta t} ^ {(j)} = \arg \max _ {k} \mathbf {C} _ {t + \Delta t} ^ {(j)} [ k ].
$$

Table 8: Temporal Part Stability (TPS) on DAVIS-2017 val. Higher values indicate more stable part assignments across time. TPS is averaged over matched token pairs and three random seeds.

<table><tr><td>Loss</td><td>TPS@1</td><td>TPS@2</td><td>TPS@4</td></tr><tr><td>Cross-Entropy</td><td>0.63 ± 0.01</td><td>0.58 ± 0.01</td><td>0.51 ± 0.02</td></tr><tr><td>One-sided KL</td><td>0.70 ± 0.01</td><td>0.64 ± 0.01</td><td>0.58 ± 0.02</td></tr><tr><td>Symmetric KL (CTC $^{2}$ )</td><td>0.78 ± 0.01</td><td>0.73 ± 0.01</td><td>0.66 ± 0.02</td></tr></table>

Table 9: Cluster Identity Retention on DAVIS-2017 val. Ret@∆t measures the fraction of matched token pairs that preserve the same dominant part identity across time. Results are averaged over three random seeds.

<table><tr><td>Loss</td><td>Ret@1</td><td>Ret@2</td><td>Ret@4</td></tr><tr><td>Cross-Entropy</td><td>0.47</td><td>0.39</td><td>0.28</td></tr><tr><td>One-sided KL</td><td>0.57</td><td>0.48</td><td>0.38</td></tr><tr><td>Symmetric KL (CTC $^{2}$ )</td><td>0.65</td><td>0.57</td><td>0.48</td></tr></table>

The retention rate at temporal ofset $\Delta t$ is defined as

$$
\mathrm{Ret} @ \Delta t = \frac {1}{| \mathcal {M} _ {\Delta t} |} \sum_ {(i, j) \in \mathcal {M} _ {\Delta t}} \mathbb {I} \Big [ \hat {y} _ {t} ^ {(i)} = \hat {y} _ {t + \Delta t} ^ {(j)} \Big ],\tag{17}
$$

where <sup>I</sup>[·] denotes the indicator function. Higher values indicate stronger preservation of latent part identities across time.

Table 9 reports cluster identity retention on DAVIS-2017 val for diferent temporal ofsets. Consistent with the TPS analysis, symmetric KL yields substantially higher retention rates than one-sided KL and cross-entropy, particularly at larger ofsets. This shows that $\mathrm { C T C ^ { 2 } }$ not only aligns part distributions in a soft sense, but also preserves consistent latent part identities over time.

Together, TPS and cluster identity retention provide complementary evidence that the discovered clusters form temporally stable mid-level structures rather than arbitrary or frame-specific groupings. These results support the intended notion of interpretability in our framework—temporal coherence and identity persistence of latent parts—without requiring explicit part annotations or human-defined semantic labels.

## 5.5 Cross-Dataset Generalisation

While Sec. 5.3 qualitatively analyses failure modes, this section quantitatively evaluates robustness under cross-dataset domain shift. Specifically, we measure relative performance degradation using $\Delta \mathcal { G }$ as a robustness indicator, computed by training on one dataset and evaluating directly on another without adaptation.

Table 10 reports cross-dataset results under the zero-shot protocol. Earlier correspondence-based SSL methods such as TimeCycle [8], MAST [43], and CRW [42] exhibit substantial degradation when transferring across datasets, with $\Delta \mathcal { G }$ ranging from −0.007 to −0.035. This sensitivity reflects their reliance on dense frame-to-frame correspondences or dataset-specific motion statistics.

By contrast, $\mathrm { C T C ^ { 2 } }$ shows consistently smaller degradation. When trained on YouTube-VOS and evaluated on DAVIS, performance drops by only $- 0 . 0 1 6$ , substantially less than for MAST or CRW. Conversely, training on DAVIS and evaluating on YouTube-VOS yields no degradation (G=0.547 vs. 0.570 in-domain), suggesting partial positive transfer.

We attribute this robustness to three design choices: (i) freezing the SAM2 encoder, which reduces overfitting to dataset-specific appearance statistics; (ii) partlevel clustering, which produces mid-level representations that are more stable under category and motion shift; and (iii) symmetric distributional alignment, which enforces temporal consistency without relying on dense optical flow or long-range memory banks.

Overall, these results indicate that $\mathrm { C T C ^ { 2 } }$ learns transferable representations that generalise beyond the source dataset, a desirable property for real-world deployment where training and test domains often difer.

## 5.6 Semi-Supervised Setting Results

For completeness, we also report results under the one-shot (semi-supervised) protocol, where the first-frame mask is provided at inference time only, while training remains fully self-supervised. These results are reported separately and are not directly comparable to the zero-shot setting.

On DAVIS-2017, using the first-frame mask to initialise the cluster-to-instance association at inference yields $\mathcal { G } ~ = ~ 0 . 6 3 5$ , with balanced improvements in region similarity $( \mathcal { I } = 0 . 6 1 7 )$ and boundary accuracy $\left( \mathcal { F } = 0 . 6 5 3 \right)$ (Table 11).

Despite relying on a frozen backbone and using no optical flow, decoder, or memory module, $\mathrm { C T C ^ { 2 } }$ performs competitively with recent correspondence-based methods. The stronger gains in boundary accuracy align with our design: instance identity is resolved by the first-frame mask, while saliency-weighted part-level temporal consistency stabilises fine-grained structures under occlusion and appearance overlap.

On YouTube-VOS, one-shot conditioning similarly improves performance under long temporal horizons and category shifts. As shown in Table 12, $\mathrm { C T C ^ { 2 } }$ achieves $\mathcal { G } = 0 . 6 2 4$ with $\mathcal { I } = 0 . 5 9 1$ and $\mathcal { F } = 0 . 6 5 7$ , outperforming early correspondence-based baselines and approaching the performance of $\mathrm { M A S T }$

Table 10: Cross-dataset generalization under the zero-shot protocol. Models are trained on one dataset and evaluated on another without fine-tuning. J , F, and G denote region similarity, boundary accuracy, and their mean. ∆G indicates relative degradation compared to in-domain training. Dashes denote metrics not reported in the original works under the corresponding protocol.

<table><tr><td>Method</td><td>Training Dataset</td><td>Test Dataset</td><td> $\mathcal{J}$ </td><td> $\mathcal{F}$ </td><td> $\mathcal{G}$ </td><td> $\Delta \mathcal{G}$ </td></tr><tr><td>TimeCycle*</td><td>DAVIS</td><td>DAVIS</td><td>0.419</td><td>0.394</td><td>0.407</td><td>-</td></tr><tr><td>TimeCycle*</td><td>YT</td><td>YT</td><td>-</td><td>-</td><td>0.430</td><td>-</td></tr><tr><td>TimeCycle*</td><td>YT</td><td>DAVIS</td><td>0.390</td><td>0.410</td><td>0.400</td><td>-0.007</td></tr><tr><td>TimeCycle*</td><td>DAVIS</td><td>YT</td><td>-</td><td>-</td><td>0.402</td><td>-0.028</td></tr><tr><td>MAST*</td><td>DAVIS</td><td>DAVIS</td><td>0.530</td><td>0.540</td><td>0.535</td><td>-</td></tr><tr><td>MAST*</td><td>YT</td><td>YT</td><td>-</td><td>-</td><td>0.540</td><td>-</td></tr><tr><td>MAST*</td><td>YT</td><td>DAVIS</td><td>0.500</td><td>0.520</td><td>0.510</td><td>-0.025</td></tr><tr><td>MAST*</td><td>DAVIS</td><td>YT</td><td>-</td><td>-</td><td>0.515</td><td>-0.025</td></tr><tr><td>CRW*</td><td>DAVIS</td><td>DAVIS</td><td>0.560</td><td>0.570</td><td>0.565</td><td>-</td></tr><tr><td>CRW*</td><td>YT</td><td>YT</td><td>-</td><td>-</td><td>0.570</td><td>-</td></tr><tr><td>CRW*</td><td>YT</td><td>DAVIS</td><td>0.520</td><td>0.540</td><td>0.530</td><td>-0.035</td></tr><tr><td>CRW*</td><td>DAVIS</td><td>YT</td><td>-</td><td>-</td><td>0.540</td><td>-0.030</td></tr><tr><td>CTC $^2$ (ours)</td><td>DAVIS</td><td>DAVIS</td><td>0.519</td><td>0.528</td><td>0.523</td><td>-</td></tr><tr><td>CTC $^2$ (ours)</td><td>YT</td><td>YT</td><td>0.563</td><td>0.577</td><td>0.570</td><td>-</td></tr><tr><td>CTC $^2$ (ours)</td><td>YT</td><td>DAVIS</td><td>0.503</td><td>0.511</td><td>0.507</td><td>-0.016</td></tr><tr><td>CTC $^2$ (ours)</td><td>DAVIS</td><td>YT</td><td>0.541</td><td>0.553</td><td>0.547</td><td>-0.023</td></tr></table>

Boundary accuracy gains are particularly pronounced, reflecting the combination of attention-guided token selection—which preserves thin, high-frequency structures—and symmetric part-level temporal alignment, which maintains stable assignments across long sequences. OSVOS is included as a supervised one-shot reference; in contrast, our model is trained without ground-truth masks and uses the first-frame annotation only at inference.

## 5.7 Component-Level Compute and Eficiency Analysis

Table 13 summarises the component-level compute profile. Nearly all computation resides in the frozen SAM2 backbone (∼19.6 GFLOPs), while the clustering head contributes only ∼0.04 GFLOPs with negligible latency. This indicates that the observed accuracy gains arise from the inductive bias of attention-guided token selection and saliency-weighted temporal consistency rather than increased model capacity. In practice, processing only salient, temporally stable parts yields a superior accuracy–eficiency trade-of compared to enlarging the backbone.

Table 11: DAVIS-2017 val, one-shot/semisupervised protocol. Training is self-supervised; the first-frame mask is provided only at inference. We report J (mean), boundary accuracy ${ \mathcal F } ,$ and $\mathcal { G } = ( \mathcal { I } + \mathcal { F } ) / 2$ (decimals).

<table><tr><td>Method</td><td> $\mathcal{G}$  (Mean)</td><td> $\mathcal{J}$  (Mean)</td><td> $\mathcal{F}$  (Mean)</td></tr><tr><td>Colorization [6]</td><td>0.340</td><td>0.346</td><td>0.327</td></tr><tr><td>CorrFlow [5]</td><td>0.503</td><td>0.484</td><td>0.522</td></tr><tr><td>TimeCycle [40]</td><td>0.487</td><td>0.464</td><td>0.500</td></tr><tr><td>MuG [70]</td><td>0.543</td><td>0.526</td><td>0.561</td></tr><tr><td>ConCorr [71]</td><td>0.630</td><td>0.605</td><td>0.655</td></tr><tr><td>CTC $^2$ (ours)</td><td>0.635</td><td>0.617</td><td>0.653</td></tr><tr><td>MAST [43]</td><td>0.655</td><td>0.633</td><td>0.676</td></tr></table>

Table 12: YouTube-VOS val, one-shot/semisupervised protocol. Training is self-supervised; the first-frame mask is provided only at inference. We report J (mean), boundary accuracy ${ \mathcal F } ,$ and $\mathcal { G } =$ $( \mathcal { I } + \mathcal { F } ) / 2$ (decimals).

<table><tr><td>Method</td><td> $\mathcal{G}$  (Mean)</td><td> $\mathcal{J}$  (Mean)</td><td>F (Mean)</td></tr><tr><td>Colorization [6]</td><td>0.370</td><td>0.366</td><td>0.374</td></tr><tr><td>CorrFlow [5]</td><td>0.447</td><td>0.438</td><td>0.456</td></tr><tr><td>OSVOS[72]</td><td>0.574</td><td>0.542</td><td>0.607</td></tr><tr><td>CTC $^2$ (ours)</td><td>0.624</td><td>0.591</td><td>0.657</td></tr><tr><td>MAST [43]</td><td>0.640</td><td>0.603</td><td>0.677</td></tr></table>

Table 14 further examines scalability with diferent token budgets on the DAVIS-2017 val set. Increasing k from 24 to 48 improves coverage and reduces noise, while gains saturate beyond this point as additional tokens primarily cover background regions. The sweet spot at $k { = } 4 8$ achieves $\mathcal { G } \mathrm { = } 0 . 5 2 6$ while sustaining real-time throughput (∼35 fps), demonstrating that the adaptive selector captures the most informative parts using a compact token budget.

## 6 Conclusion and Future Work

We presented $\mathbf { C T C ^ { 2 } }$ , a self-supervised framework for discovering temporally consistent and semantically coherent object parts in videos using a frozen SAM2 backbone. By combining attention-derived saliency, adaptive token selection, and part-level cross-temporal clustering, $\mathrm { C T C ^ { 2 } }$ introduces a lightweight inductive bias that aligns semantic parts across time without relying on optical flow, pseudo-labels, or testtime supervision. Across three standard VOS benchmarks, the method achieves strong zero-shot performance and competitive eficiency, demonstrating that part-level temporal consistency provides a viable alternative to pixel-level correspondence for self-supervised video segmentation.

Table 13: Component-level compute profile at 224×224 input resolution. The clustering head adds only ∼0.04 GFLOPs, confirming encoder-bound runtime.

<table><tr><td>Component</td><td>Params (M)</td><td>FLOPs (G)</td><td>fps</td></tr><tr><td>Backbone (SAM2 ViT-S, frozen)</td><td>22.0</td><td>19.60</td><td>34.8</td></tr><tr><td>Clustering head (k=48, K=16)</td><td>0.3</td><td>0.04</td><td>-</td></tr><tr><td>End-to-end</td><td>22.3</td><td>19.64</td><td>34.8</td></tr></table>

Table 14: Token-budget scaling on DAVIS-2017 val at 224×224 input resolution. Accuracy improves up to k=48 before saturating as background tokens dominate.

<table><tr><td>k</td><td>FLOPs (G)</td><td>fps</td><td> $\mathcal{J}$ </td><td> $\mathcal{F}$ </td><td> $\mathcal{G}$ </td></tr><tr><td>24</td><td>19.45</td><td>37.2</td><td>0.507</td><td>0.529</td><td>0.518</td></tr><tr><td>48</td><td>19.64</td><td>34.8</td><td>0.521</td><td>0.530</td><td>0.526</td></tr><tr><td>64</td><td>19.78</td><td>34.2</td><td>0.518</td><td>0.528</td><td>0.523</td></tr></table>

Beyond aggregate accuracy, we validate key design choices through extensive analysis. Ablation studies confirm the role of symmetric KL divergence in stabilising temporal alignment, while controlled experiments comparing frozen and fine-tuned backbones justify our eficiency-oriented design. Robustness and crossdataset evaluations reveal reduced degradation under domain shift relative to prior correspondence-based methods, and qualitative analyses show that the discovered clusters correspond to temporally stable and interpretable object parts, albeit with known failure modes.

Despite these strengths, several limitations remain. Although adaptive Top-p selection improves coverage, token budgeting may require re-tuning at substantially higher resolutions or under extreme clutter. Because the encoder is kept frozen for eficiency, saliency quality ultimately depends on the pretrained backbone and may degrade under severe out-of-distribution conditions. While multi-ofset supervision improves robustness, very long-term dependencies and highly irregular motion remain challenging without explicit memory mechanisms. Finally, real-time throughput is hardware-dependent, and deployment on resource-constrained devices may require additional optimisation.

Future work will explore hierarchical and multi-scale part discovery, integration of lightweight temporal memory to better handle long-term occlusions, and refinement of saliency estimation through self-distillation or adaptive attention aggregation.

Extending part-level temporal consistency to cross-video or category-level learning, as well as interactive and semi-supervised VOS settings, represents a promising direction for broader applicability in real-world video understanding.

## References

[1] Miao, D., Gu, Y., Li, X., He, Z., Wang, Y., Yang, M.-H.: Discriminative spatial-semantic vos solution: 1st place solution for 6th lsvos. arXiv preprint arXiv:2408.16431 (2024) https://doi.org/10.48550/arXiv.2408.16431

[2] Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., Sorkine-Hornung, A.: A benchmark dataset and evaluation methodology for video object segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016). https://doi.org/10.1109/cvpr.2016.85

[3] Xu, N., Yang, L., Fan, Y., Yang, J., Yue, D., Liang, Y., Price, B., Cohen, S., Huang, T.: Youtube-vos: Sequence-to-sequence video object segmentation. In: Proceedings of the European Conference on Computer Vision (2018). https:// doi.org/10.1007/978-3-030-01228-1\_36

[4] Maninis, K.-K., Caelles, S., Chen, Y., Van Gool, L.: Video object segmentation without temporal information. IEEE Transactions on Pattern Analysis and Machine Intelligence (2018) https://doi.org/10.1109/TPAMI.2018.2838670

[5] Lai, Z., Xie, W.: Self-supervised learning for video correspondence flow. arXiv preprint arXiv:1905.00875 (2019) https://doi.org/10.48550/arXiv.1905.00875

[6] Vondrick, C., Shrivastava, A., Fathi, A., Guadarrama, S., Murphy, K.: Tracking emerges by colorizing videos. In: Proceedings of the European Conference on Computer Vision (2018). https://doi.org/10.1007/978-3-030-01261-8\_24

[7] Salehi, M., Gavves, E., Snoek, C.G., Asano, Y.M.: Time does tell: Self-supervised time-tuning of dense image representations. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023). https://doi.org/10.1109 iccv51070.2023.01516

[8] Wang, X., Jabri, A., Efros, A.A.: Learning correspondence from the cycleconsistency of time. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019). https://doi.org/10.1109/cvpr.2019. 00267

[9] Yang, Z., Wei, Y., Yang, Y.: Collaborative video object segmentation by multiscale foreground-background integration. IEEE Transactions on Pattern Analysis and Machine Intelligence (2021) https://doi.org/10.1007/978-3-030-58558-7\_20

[10] Huang, N., Zheng, W., Xu, C., Keutzer, K., Zhang, S., Kanazawa, A., Wang, Q.: Segment any motion in videos. In: Proceedings of the Computer Vision and Pattern Recognition Conference (2025). https://doi.org/10.1109/cvpr52734. 2025.00323

[11] Zhu, J., Chen, Z., Hao, Z., Chang, S., Zhang, L., Wang, D., Lu, H., Luo, B.,

He, J.-Y., Lan, J.-P., et al.: Tracking anything in high quality. arXiv preprint arXiv:2307.13974 (2023) https://doi.org/10.48550/arXiv.2307.13974

[12] Ziegler, A., Asano, Y.M.: Self-supervised learning of object parts for semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14502–14511 (2022). https://doi.org/10. 1109/cvpr52688.2022.01410

[13] Wei, Y., Gupta, A., Morgado, P.: Towards latent masked image modelling for selfsupervised visual representation learning. In: European Conference on Computer Vision (2024). https://doi.org/10.1007/978-3-031-72933-1\_1

[14] Zheng, X., Liao, L., Jiao, J., Gao, F., Wang, R.: Surface-sos: Self-supervised object segmentation via neural surface representation. IEEE Transactions on Image Processing (2024) https://doi.org/10.1109/tip.2024.3374199

[15] Hung, W.-C., Jampani, V., Liu, S., Molchanov, P., Yang, M.-H., Kautz, J.: Scops: Self-supervised co-part segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 869–878 (2019). https://doi.org/10.1109/cvpr.2019.00096

[16] Xu, M., Zhang, Z., Zhang, H., Wei, Y., Xie, E., Li, Z., Loy, C.C.: Groupvit: Semantic segmentation emerges from text supervision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022). https://doi.org/10.1109/cvpr52688.2022.01760

[17] Kim, H.Y.: Rotation-discriminating template matching based on fourier coefficients of radial projections with robustness to scaling and partial occlusion. Pattern Recognition (2010) https://doi.org/10.1016/j.patcog.2009.08.005

[18] Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024) https://doi.org/10.48550/arXiv. 2408.00714

[19] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2021). https://doi.org/10.1109/ICCV48922.2021.00951

[20] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, L., Gustafson, J., Xiao, T., Whitehead, S., Cottrell, R., Dolson, J., et al.: Segment anything. arXiv preprint arXiv:2304.02643 (2023) https://doi.org/10.1109/ICCV51070.2023.00371

[21] Ding, S., Qian, R., Xu, H., Lin, D., Xiong, H.: Betrayed by attention: A simple yet efective approach for self-supervised video object segmentation. In: European Conference on Computer Vision (2024). https://doi.org/10.1007/

[22] Cheng, S., Wang, J., Shen, Y., Joo, J., Zhang, X.: Xmem: Long-term video object segmentation with an atkinson-shifrin memory model. In: Proceedings of the European Conference on Computer Vision (2022). https://doi.org/10.1007 978-3-031-19815-1\_37

[23] Yang, Z., Wei, Y., Yang, Y.: Associating objects with transformers for video object segmentation. In: Advances in Neural Information Processing Systems, vol. 34, pp. 2491–2502 (2021)

[24] Pont-Tuset, J., Perazzi, F., Caelles, S., Arbelaez, P., Sorkine-Hornung, A., Van Gool, L.: The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675 (2017) https://doi.org/10.48550/arXiv.1704.00675

[25] Jing, L., Tian, Y.: Self-supervised visual feature learning with deep neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence (2020) https://doi.org/10.1109/tpami.2020.2992393

[26] He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.: Momentum contrast for unsupervised visual representation learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2020). https://doi. org/10.20944/preprints202501.0668.v1

[27] Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z.D., Azar, M., et al.: Bootstrap your own latent: A new approach to self-supervised learning. In: Advances in Neural Information Processing Systems, vol. 33, pp. 21271–21284 (2020)

[28] Chen, X., He, K.: Exploring simple siamese representation learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021). https://doi.org/10.1109/cvpr46437.2021.01549

[29] Wang, X., Xie, S., Gupta, A.: Dense contrastive learning for self-supervised visual pre-training. In: CVPR (2021). https://doi.org/10.1109/cvpr46437.2021.00304

[30] Xie, Z., Lin, Y., Zhang, Z., Cao, Y., Lin, S., Hu, H.: Propagate yourself: Exploring pixel-level consistency for unsupervised visual representation learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021). https://doi.org/10.1109/cvpr46437.2021.01641

[31] Wang, Y., Shen, X., Yuan, Y., Du, Y., Li, M., Hu, S.X., Crowley, J.L., Vaufreydaz, D.: Tokencut: Segmenting objects in images and videos with self-supervised transformer and normalized cut. IEEE transactions on pattern analysis and machine intelligence (2023) https://doi.org/10.1109/tpami.2023.3305122

[32] Siméoni, O., Puy, G., Vo, H.V., Roburin, S., Gidaris, S., Bursuc, A., Pérez, P.,

Marlet, R., Ponce, J.: Localizing objects with self-supervised transformers and no labels. arXiv preprint arXiv:2109.14279 (2021) https://doi.org/10.5244/c.35.365

[33] Wang, X., Yu, Z., De Mello, S., Kautz, J., Anandkumar, A., Shen, C., Alvarez, J.M.: Freesolo: Learning to segment objects without annotations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022). https://doi.org/10.1109/cvpr52688.2022.01378

[34] Niu, D., Wang, X., Han, X., Lian, L., Herzig, R., Darrell, T.: Unsupervised universal image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024). https://doi.org/10.1109/ cvpr52733.2024.02146

[35] Araslanov, N., Schaub-Meyer, S., Roth, S.: Dense unsupervised learning for video segmentation. In: Advances in Neural Information Processing Systems, vol. 34, pp. 25308–25319 (2021)

[36] Das, K., Abouelenien, M., Burzo, M.G., Elson, J., Prakah-Asante, K., Maranville, C.: Towards autonomous physiological signal extraction from thermal videos using deep learning. In: Proceedings of the 25th International Conference on Multimodal Interaction (2023). https://doi.org/10.1145/3577190.3614123

[37] Misra, I., Zitnick, C.L., Hebert, M.: Shufle and learn: unsupervised learning using temporal order verification. In: European Conference on Computer Vision (2016). https://doi.org/10.1007/978-3-319-46448-0\_32

[38] Wei, D., Lim, J.J., Zisserman, A., Freeman, W.T.: Learning and using the arrow of time. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018). https://doi.org/10.1109/cvpr.2018.00840

[39] Vondrick, C., Pirsiavash, H., Torralba, A.: Anticipating visual representations from unlabeled video. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016). https://doi.org/10.1109/cvpr.2016.18

[40] Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., Zisserman, A.: Temporal cycle-consistency learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019). https://doi.org/10.1109/cvpr. 2019.00190

[41] Kong, Q., Wei, W., Deng, Z., Yoshinaga, T., Murakami, T.: Cyclecontrast for self-supervised video representation learning. In: Advances in Neural Information Processing Systems, vol. 33, pp. 8089–8100 (2020)

[42] Jabri, A., Owens, A., Efros, A.A.: Space-time correspondence as a contrastive random walk. In: Advances in Neural Information Processing Systems, vol. 33, pp. 19545–19560 (2020)

[43] Lai, Z., Xie, W., Zisserman, A.: Mast: A memory-augmented self-supervised tracker. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021). https://doi.org/10.1109/cvpr42600.2020.00651

[44] Doersch, C., Yang, Y., Vecerik, M., Gokay, D., Gupta, A., Aytar, Y., Carreira, J., Zisserman, A.: Tapir: Tracking any point with per-frame initialization and temporal refinement. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023). https://doi.org/10.1109/iccv51070.2023.00923

[45] Amir, S., Gandelsman, Y., Bagon, S., Dekel, T.: Deep vit features as dense visual descriptors. arXiv preprint arXiv:2112.05814 (2021) https://doi.org/10.48550 arXiv.2112.05814

[46] Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Tandon, S., Mensch, A., Milani, S., Huang, P.-S., Han, W., et al.: Flamingo: a visual language model for few-shot learning. In: Advances in Neural Information Processing Systems, vol. 35, pp. 23716–23736 (2022)

[47] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020) https://doi.org/10.48550/arXiv.2010.11929

[48] Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018) https://doi.org/10.48550 arXiv.1807.03748

[49] Aubret, A., Teulière, C., Triesch, J.: Self-supervised visual learning from interactions with objects. In: European Conference on Computer Vision (2024). https://doi.org/10.1007/978-3-031-73226-3\_4

[50] Arshid, W., Awrangjeb, M., Liew, A.W.C., Gao, Y.: Camvos: Leveraging context and memory for advanced video object segmentation. In: IEEE Conference on Computer Vision and Pattern Recognition (2024). https://doi.org/10.1109 dicta63115.2024.00098

[51] Jiaxing, Z., Hao, T.: Sam2 for image and video segmentation: A comprehensive survey. arXiv preprint arXiv:2503.12781 (2025) https://doi.org/10.48550/arXiv. 2503.12781

[52] Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., Dosovitskiy, A.: Do vision transformers see like convolutional neural networks? In: Advances in Neural Information Processing Systems, vol. 34, pp. 12116–12128 (2021)

[53] Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., Hofman, J.: Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461 (2022) https:// doi.org/10.48550/arXiv.2210.09461

[54] Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.-J.: Dynamicvit: Eficient vision transformers with dynamic token sparsification. In: Advances in Neural Information Processing Systems, vol. 34, pp. 13937–13949 (2021)

[55] Vasu, P.K.A., Gabriel, J., Zhu, J., Tuzel, O., Ranjan, A.: Fastvit: A fast hybrid vision transformer using structural reparameterization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023). https://doi. org/10.1109/iccv51070.2023.00532

[56] Asano, Y.M., Rupprecht, C., Vedaldi, A.: A critical analysis of self-supervision, or what we can learn from a single image. In: International Conference on Learning Representations (ICLR) (2020). https://doi.org/10.48550/arXiv.1904.13132

[57] Seong, H., Hyun, J., Kim, E.: Kernelized memory network for video object segmentation. In: European Conference on Computer Vision (2020). https://doi. org/10.1007/978-3-030-58542-6\_38

[58] Sun, D., Yang, X., Liu, M.-Y., Kautz, J.: Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018). https://doi.org/10.1109/cvpr. 2018.00931

[59] Caron, M., Bojanowski, P., Joulin, A., Douze, M.: Deep clustering for unsupervised learning of visual features. In: Proceedings of the European Conference on Computer Vision (2018). https://doi.org/10.1007/978-3-030-01264-9\_9

[60] Ji, X., Henriques, J.F., Vedaldi, A.: Invariant information clustering for unsupervised image classification and segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2019). https://doi.org/10.1109 iccv.2019.00996

[61] Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsuper vised learning of visual features by contrasting cluster assignments. In: Advances in Neural Information Processing Systems, vol. 33, pp. 9912–9924 (2020)

[62] Oh, S.W., Lee, J.-Y., Xu, N., Kim, S.J.: Video object segmentation using space-time memory networks. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2019). https://doi.org/10.1109/iccv.2019.00932

[63] Aydemir, G., Xie, W., Guney, F.: Self-supervised object-centric learning for videos. In: Advances in Neural Information Processing Systems, vol. 36, pp. 32879–32899 (2023). https://doi.org/10.52202/075280-1424

[64] Xie, J., Xie, W., Zisserman, A.: Segmenting moving objects via an object-centric layered representation. In: Advances in Neural Information Processing Systems, vol. 35, pp. 28023–28036 (2022). https://doi.org/10.52202/068431-2032

[65] Qian, R., Ding, S., Liu, X., Lin, D.: Semantics meets temporal correspondence: Self-supervised object-centric learning in videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023). https://doi. org/10.1109/iccv51070.2023.01529

[66] Xu, K., Wen, L., Li, G., Huang, Q.: Self-supervised deep triplenet for video object segmentation. IEEE Transactions on Multimedia (2020) https://doi.org 10.1109/tmm.2020.3026913

[67] Yang, L., Wang, Y., Xiong, X., Yang, J., Katsaggelos, A.K.: Eficient video object segmentation via network modulation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018). https://doi.org/10.1109 cvpr.2018.00680

[68] Perazzi, F., Khoreva, A., Benenson, R., Schiele, B., Sorkine-Hornung, A.: Learning video object segmentation from static images. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2663–2672 (2017). https://doi.org/10.1109/cvpr.2017.372

[69] Oh, S.W., Lee, J.-Y., Sunkavalli, K., Kim, S.J.: Fast video object segmentation by reference-guided mask propagation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018). https://doi.org/10.1109 cvpr.2018.00770

[70] Lu, X., Wang, W., Shen, J., Tai, Y.-W., Crandall, D.J., Hoi, S.C.: Learning video object segmentation from unlabeled videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2020). https://doi. org/10.1109/cvpr42600.2020.00898

[71] Wang, N., Zhou, W., Li, H.: Contrastive transformation for self-supervised correspondence learning. In: Proceedings of the AAAI Conference on Artificial Intelligence (2021). https://doi.org/10.1609/aaai.v35i11.17220

[72] Caelles, S., Maninis, K.-K., Pont-Tuset, J., Leal-Taixé, L., Cremers, D., Van Gool, L.: One-shot video object segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 221–230 (2017)