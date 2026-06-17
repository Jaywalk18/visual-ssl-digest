# Selective Synergistic Learning for Video Object-Centric Learning

WonJun Moon1† and Jae-Pil Heo2⋆

1 KAIST, South Korea

2 Sungkyunkwan University, South Korea

Abstract. Typical video object-centric learning (VOCL) approaches employ slot-based frameworks that rely on reconstruction-driven encoder–decoder architectures, where learning is mediated by two spatial maps: attention maps from the encoder and object maps from the decoder. As these two distinct maps exhibit different properties, a recent dense alignment strategy attempted to reconcile this discrepancy by enforcing agreement across all spatio-temporal patches via contrastive learning. However, this indiscriminate alignment inadvertently propagates the inherent weaknesses of each module, such as noisy encoder predictions and blurred decoder boundaries. Moreover, computing dense similarities across all pairs incurs a computational cost quadratic in the total number of spatio-temporal patches, severely limiting scalability. Motivated by this, we propose Selective Synergistic Learning (SSync). Instead of exhaustive patch-to-patch alignment, SSync prevents error propagation by selectively distilling only the most reliable cues: leveraging the encoder strictly for boundary refinement and the decoder for interior denoising. This is realized via a pseudo-labeling with linear complexity, eliminating the need for quadratic spatial comparisons. Also, to prevent the reinforcement of architectural biases like slot redundancy, we introduce a transitive pseudo-label merging that consolidates overlapping slots based on spatio-temporal activation consistency. Extensive studies demonstrate that SSync improves decomposition quality and serves as a versatile, plug-and-play module while also exhibiting exceptional robustness to slot configurations. Code is available at github.com/wjun0830/SSync.

Keywords: Video Object-Centric Learning · Object Discovery

## 1 Introduction

Object-centric learning is a fundamental paradigm for structured visual reasoning, enabling downstream applications such as object editing, generation, and scene understanding [1–7]. By decomposing scenes into object representations, objectcentric frameworks provide an interpretable and modular representation space that facilitates reasoning beyond pixel-level perception [8]. Among them, slot attention [9] has emerged as a dominant method, grouping patch features into latent slots and reconstructing the input through a decoder that implicitly produces object maps. It has recently been extended to videos, leveraging temporal cues to achieve consistent object discovery and tracking across frames [10–15].

![](images/cca723548b02fb4dcf747c429f049cdee9f5ed07f440ff22999a9087c235d361.jpg)  
Fig. 1: Overall flow and motivation. (a) Video frames are sequentially processed, where slots are recurrently updated based on the previous frame’s slots. (b) Slot attention map captures sharp boundaries but contains noise, while the decoder object map offers a consistent representation but suffers from blurry edges. To leverage their complementary strengths, SSync reciprocally distills the sharp boundary cues from the attention map and the consistent semantics of the decoder object map.

Despite their empirical success, slot architectures suffer from a fundamental structural mismatch. Learning is mediated by two spatial maps: the encoder’s attention maps and the decoder’s reconstructed object maps. Ideally, these maps should be logically consistent, as both branches are jointly optimized through a shared reconstruction objective. However, in practice, they exhibit distinct inductive biases. The encoder, often built upon high-capacity vision backbones [16], produces spatially sharp yet noisy assignments, exhibiting high-frequency sensitivity. In contrast, the decoder, typically implemented as a lightweight MLP, imposes a low-frequency spatial prior, yielding smooth and temporally coherent but blurry object boundaries [14]. This structural asymmetry induces persistent misalignment between patch assignments and rendering regions, resulting in unstable slot semantics and fragmented object decomposition.

A straightforward solution is to enforce consistency between these two spatial maps. Recently, SRL [14] attempts to bridge this gap through dense contrastive alignment across all spatio-temporal patches. While conceptually appealing, we argue that dense alignment is fundamentally misaligned with the heterogeneous inductive biases of the encoder and decoder since it treats every patch as an equally reliable teacher. Consequently, it inadvertently propagates the inherent failure modes of each branch. Also, computing similarities across all spatio-temporal patches incurs a quadratic memory cost, severely limiting its applicability to long sequences or high-resolution videos.

In this work, we introduce a selective alignment principle: mutual supervision should occur only where each branch provides reliable cues. A key insight is that, once reliable regions are correctly identified, the alignment mechanism itself can be remarkably simple. We introduce Selective Synergistic Learning (SSync), a selective mutual distillation framework that filters supervision at the patch level instead of enforcing global agreement. Concretely, we exploit the encoder’s boundary sensitivity to refine decoder masks at object boundaries, while leveraging the decoder’s spatial coherence to denoise encoder assignments within interior regions. Specifically, we exploit local consistency cues to discern object boundaries within the encoder and coherent interior regions within the decoder, and formulate an efficient supervisory signal through pseudo-labeling. By aligning only in these reliable regions, SSync mitigates error propagation inherent in dense objectives while substantially reducing memory complexity. Our contributions are: (1) Reliability-based selective distillation. We identify the spatial complementarity between encoder and decoder expertise and reformulate mutual learning as a selective cross-distillation framework, (2) Redundancy control via transitive merging. To stabilize on-the-fly pseudo-labeling during optimization, we introduce a transitive merging strategy, and (3) Strong performance & practical generalization. SSync achieves state-of-the-art results across VOCL benchmarks, improves memory efficiency relative to dense alignment, and remains robust to varying slot configurations.

## 2 Related Work

## 2.1 Object-Centric Representation Learning

Object-centric learning seeks to decompose scenes into discrete structural entities [17–19]. Slot Attention [9] established a foundational framework by iteratively grouping spatial features into latent slots and reconstructing the image via a shared decoder. This encoder–decoder formulation enables unsupervised object discovery while maintaining permutation invariance over slots.

Recent works have extended slot-based learning to dynamic visual scenes [20]. SAVi [12] and SAVi++ [13] leveraged motion cues such as optical flow and depth to promote temporal consistency. STEVE [11] incorporated transformerbased models to capture complex object interactions. Videosaur [10] introduced objectives that encourage grouping of motion-consistent regions, and SlotContrast [21] enhanced slot discriminability via temporal contrastive learning. While these significantly improve object discovery in videos, they predominantly rely on a reconstruction objective, which implicitly assumes a convergence between encoder attention and decoder object maps. In practice, however, these two representations exhibit distinct inductive biases, leading to spatial discrepancies.

To address this discrepancy, SRL [14] introduced a mutual refinement process via a dense alignment strategy, employing a contrastive objective across all spatio-temporal patches. While this reduces inconsistency, it overlooks the unique strengths and inherent limitations of each branch by uniformly enforcing agreement across all regions. This indiscriminate alignment inevitably conflates complementary strengths with architectural weaknesses, all while incurring prohibitive memory costs. In contrast, we explicitly identify the expertise of each branch and instantiate synergy through selective cross-distillation, thereby preventing error propagation and ensuring linear scalability.

## 2.2 Pseudo-Labeling

Pseudo-labeling has been widely adopted in semi-supervised and self-supervised learning to exploit high-confidence predictions as supervisory signals [22–27]. Selftraining approaches such as Noisy Student [28] and Meta Pseudo Labels [29] also demonstrate that iterative refinement of pseudo-labels can significantly improve representation quality.

However, directly applying pseudo-labeling to slot learning introduces unique challenges. Since pseudo-labels are generated on-the-fly from the model’s own predictions, they are inherently unstable; early training errors can easily propagate via pseudo-labeling, making the framework highly vulnerable to noise. To resolve this, we introduce a transitive pseudo-label merging that analyzes spatio-temporal activation overlap to detect redundancy and consolidate slot identities globally. This redundancy-aware refinement stabilizes selective supervision and prevents feedback loops caused by imperfect on-the-fly pseudo-labels.

## 3 Selective Synergistic Learning

## 3.1 Preliminaries

Video Slot Learning and Spatial Maps. We consider an input video represented as a grid of patch tokens. Let $z _ { t , p }$ denote the feature of patch $p \in \{ 1 , \ldots , P \}$ at time $t \in \{ 1 , \ldots , T \}$ , where $P = \bar { H } \times W$ . A slot attention encoder predicts $S$ slot prototypes. By computing the dot product between the projected patch queries $\mathbf { q } _ { t , p }$ (derived from $z _ { t , p } )$ and the slot keys $\mathbf { k } _ { t , s } ,$ the encoder generates an encoder attention map over slots for each patch:

$$
\mathbf {A} _ {t, p} = \operatorname{Softmax} \left(\mathbf {q} _ {t, p} ^ {\top} \mathbf {k} _ {t, 1: S}\right), \tag {1}
$$

where $\mathbf { A } _ { t , p , s }$ is the probability that patch $z _ { t , p }$ is explained by slot s. A decoder reconstructs the input and produces a decoder object map $\dot { \mathbf { D } } \in \mathbb { R } ^ { T \times P \times S }$ , which is normalized over the slot dimension such that $\begin{array} { r } { \sum _ { s = 1 } ^ { S } \bar { \mathbf { D } } _ { t , p , s } = 1 } \end{array}$ for all t and $p .$ Both maps provide patch-to-slot assignments but are generated by different modules and thus exhibit different error characteristics.

Bias Misalignment in Encoder-Decoder Maps. Ideally, the attention map A and the object map D should be consistent, since they are optimized jointly under reconstruction-driven learning. In practice, however, A and D are persistently misaligned since they inherit distinct spatial characteristics from different modules. The encoder typically produces spatially sharp attention maps but is vulnerable to noisy assignments, while the decoder tends to yield spatiotemporally coherent maps but with blurred object boundaries. This heterogeneity implies that reliability is region-dependent: boundary regions benefit from the encoder’s sharpness, whereas interior regions benefit from the decoder’s coherence.

Limitations of Dense Alignment. SRL [14] attempted to reduce discrepancy between A and D via dense alignment (e.g., contrastive objectives [30, 31] across the full spatio-temporal volume). While straightforward, it implicitly assumes that all patches provide equally reliable supervision; when this assumption fails, indiscriminate agreement propagates erroneous signals from unreliable artifacts. Consequently, SRL only partially resolves the discrepancy, as each module inadvertently reinforces the other’s inherent flaws. We provide quantitative and qualitative evidence of this failure in Fig. 2 and Tab. 1. Also, dense patch-wise

![](images/531b7bc727ba446a1dfa53f070beb48c245c52741b7f2c84a08f863227e46b0d.jpg)

<details>
<summary>text_image</summary>

Slot Attention Map (A)
Decoder Object Map (D)
Slot Attention Map (A)
Decoder Object Map (D)
Slot Contrast
SRL
Ours
</details>

Fig. 2: Visualization of the attention map A and object map D. Compared to SlotContrast [21], SRL [14] reduces noise and produces sharper slot assignments. However, dense alignment also propagates both noise and blur across the two branches. Consequently, noisy and spatially blurry representations are still observed in both maps, indicating incomplete resolution of encoder–decoder discrepancy.

objective requires pairwise spatio-temporal comparisons, resulting in quadratic complexity $\mathcal { O } ( ( T \cdot H \cdot W ) ^ { 2 } )$ ), which limits its scalability to long or high-resolution videos.

Motivation: Selective Alignment Principle. Above observations reveal that dense alignment requires simultaneous reliability of both maps, while encoder-decoder asymmetry

Table 1: Encoder-decoder consistency. We used symmetric Adjusted Rand Index (ARI) and asymmetric mean Best Overlap (mBO).

<table><tr><td>Method</td><td>ARI $\mathbf{A} \leftrightarrow \mathbf{D}$ </td><td>mBO $\mathbf{A} \rightarrow \mathbf{D}$ </td><td>mBO $\mathbf{D} \rightarrow \mathbf{A}$ </td></tr><tr><td>SlotContrast</td><td>73.0</td><td>63.6</td><td>62.9</td></tr><tr><td>SRL</td><td>77.6</td><td>65.4</td><td>60.8</td></tr><tr><td>Ours</td><td>85.7</td><td>72.2</td><td>68.6</td></tr></table>

during optimization often invalidates this assumption. Therefore, we propose a selective alignment principle: mutual supervision should be applied only where each branch is structurally reliable. Concretely, we leverage encoder-driven boundaries and decoder-driven interiors to align the counterpart branch, proving that minimal supervision is sufficient for robust alignment once reliable regions are identified. This prevents error propagation caused by indiscriminate agreement and eliminates the need for dense spatio-temporal pairwise comparisons.

## 3.2 Structuring Pseudo-Labels

To instantiate the selective alignment principle, we first construct structured pseudo-labels from the model’s internal spatial maps. These pseudo-labels serve as teacher signals for cross-module refinement. For each spatio-temporal patch at (t, p), we derive hard slot assignments from the probabilistic encoder attention map A and decoder object map D:

$$
\hat {s} _ {t, p} ^ {A} = \arg \max _ {s} \mathbf {A} _ {t, p, s}, \quad \hat {s} _ {t, p} ^ {D} = \arg \max _ {s} \mathbf {D} _ {t, p, s}. \tag {2}
$$

These provide candidate targets, whose reliability varies across regions.

Local Consistency Analysis. To identify structurally reliable regions, we measure local slot consistency separately on the encoder attention assignments and the decoder object-map assignments. Let $\mathcal { N } _ { \mathrm { s p } } ( p )$ denote the spatial

8-neighborhood of patch $p$ on the $H \times W$ patch grid. We define the spatiotemporal neighborhood of a patch $( t , p )$ by augmenting $\mathcal { N } _ { \mathrm { s p } } ( p )$ with temporally adjacent patches at the same spatial location to further detect motion edges:

$$
\mathcal {N} (t, p) = \{(t, q) \mid q \in \mathcal {N} _ {\mathrm{sp}} (p) \} \cup \{(t - 1, p), (t + 1, p) \}, \tag {3}
$$

where out-of-range temporal indices are omitted. For $\mathbf { X } \in \{ \mathbf { A } , \mathbf { D } \}$ , we quantify the local disagreement and agreement counts as:

$$
c _ {(t, p)} ^ {\neq , \mathbf {X}} = \sum_ {(t ^ {\prime}, q) \in \mathcal {N} (t, p)} \mathbb {I} \big [ \hat {s} _ {t ^ {\prime}, q} ^ {\mathbf {X}} \neq \hat {s} _ {t, p} ^ {\mathbf {X}} \big ], \qquad c _ {(t, p)} ^ {= , \mathbf {X}} = \sum_ {(t ^ {\prime}, q) \in \mathcal {N} (t, p)} \mathbb {I} \big [ \hat {s} _ {t ^ {\prime}, q} ^ {\mathbf {X}} = \hat {s} _ {t, p} ^ {\mathbf {X}} \big ]. \tag {4}
$$

Boundary Region Selection. Boundary regions are characterized by local disagreement while still maintaining semantic grounding. Since encoder attention assignments are typically sharper, we detect boundary candidates from A:

$$
\mathcal {P} _ {\mathrm{bd}} = \left\{(t, p) \mid c _ {(t, p)} ^ {\neq , \mathbf {A}} > n _ {\mathrm{bd}} \wedge c _ {(t, p)} ^ {= , \mathbf {A}} \geq 1 \right\}, \tag {5}
$$

where $n _ { \mathrm { b d } }$ is a predefined threshold that controls the sensitivity of boundary detection. This criterion captures meaningful object transitions while excluding isolated noisy assignments by requiring at least one additional agreeing neighbor.

Interior Region Selection. In contrast, interior (non-boundary) regions exhibit high local consistency, which is more reliably reflected in the decoder object map assignments. We thus define the set of interior patches $\mathcal { P } _ { \mathrm { n b d } }$ using D:

$$
\mathcal {P} _ {\mathrm{nbd}} = \left\{(t, p) \mid c _ {(t, p)} ^ {\neq , \mathbf {D}} <   n _ {\mathrm{nbd}} \right\}, \tag {6}
$$

where $n _ { \mathrm { n b d } }$ controls the strictness of interior selection; a patch is included in $\mathcal { P } _ { \mathrm { n b d } }$ only if the number of differing neighbor assignments is strictly smaller than $n _ { \mathrm { n b d } }$ . Consequently, these boundary and interior sets $( \mathcal { P } _ { \mathrm { b d } }$ and $\mathcal { P } _ { \mathrm { n b d } } )$ provide reliable regions for asymmetric cross-distillation: boundary patches supervise the decoder using encoder-derived pseudo-labels, while interior patches supervise the encoder using decoder-derived pseudo-labels.

From a broader perspective, our selection acts as a relaxed morphological erosion [32] parameterized by a $3 \times 3$ kernel of all ones. Unlike erosion, which strictly requires all neighboring pixels within a kernel to agree (making it highly susceptible to noise patches), our parameterized spatiotemporal formulation explicitly filters noise and overcomes standard erosion’s lack of temporal awareness.

## 3.3 Transitive Pseudo-Label Merging

Although selective alignment restricts supervision to structurally reliable regions of each module, we note that pseudo-labels are still imperfect as they are generated on-the-fly from intermediate model predictions. As a result, they may inherit systematic semantic errors such as over-fragmentation3, where a single object is split across multiple slot identities. When such fragmented assignments are directly used as supervision targets, the labels themselves become inherently flawed. This inadvertently reinforces over-fragmentation by forcing the model to associate a single semantic object with multiple, conflicting slot identities. To prevent this instability, we refine pseudo-labels before applying selective alignment losses. Specifically, we detect redundant slots based on spatio-temporal activation overlap and consolidate them using a transitive connectivity criterion.

For each slot s, we determine whether a spatio-temporal patch is active by thresholding its attention value against the slot-wise mean activation $\mu _ { s } \colon$

$$
\mathbf {M} _ {t, p, s} = \mathbb {I} \left[ \mathbf {A} _ {t, p, s} > \mu_ {s} \right], \quad \mu_ {s} = \frac {1}{T P} \sum_ {t, p} \mathbf {A} _ {t, p, s}. \tag {7}
$$

This yields a binary mask M that identifies regions where each slot exhibits above-average activation over space and time. Using the binary active-region masks M, we measure the overlap between slots $( s , s ^ { \prime } )$ via frame-averaged IoU:

$$
\mathrm{IoU} (s, s ^ {\prime}) = \frac {1}{T} \sum_ {t = 1} ^ {T} \frac {\sum_ {p} \mathbf {M} _ {t , p , s} \mathbf {M} _ {t , p , s ^ {\prime}}}{\sum_ {p} \mathbf {M} _ {t , p , s} + \sum_ {p} \mathbf {M} _ {t , p , s ^ {\prime}} - \sum_ {p} \mathbf {M} _ {t , p , s} \mathbf {M} _ {t , p , s ^ {\prime}}}. \tag {8}
$$

We construct a redundancy graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where each node $s \in \mathcal V$ corresponds to a slot identity. An undirected edge $( s , s ^ { \prime } ) \in \mathcal { E }$ is added if

$$
\mathrm{IoU} (s, s ^ {\prime}) > \tau_ {\text { merge }}, \tag {9}
$$

indicating that the two slots exhibit substantial spatio-temporal overlap and are therefore likely to represent the same object.

Since redundancy may occur transitively $( \mathrm { e . g . } , s _ { 1 }$ overlaps with $s _ { 2 } ,$ and $s _ { 2 }$ overlaps with $s _ { 3 } )$ , we group redundant slots by computing the connected components of $\mathcal { G } .$ . Each connected component $\mathcal { C } _ { k } \subset \mathcal { V }$ represents a cluster of mutually redundant slots. We denote the set of all disjoint clusters as $\mathbb { C } =$ $\{ \mathcal { C } _ { 1 } , \mathcal { C } _ { 2 } , \ldots , \mathcal { C } _ { m } \}$ . For each component $\mathcal { C } _ { k }$ , we select a dominant slot identity:

$$
s _ {k} ^ {\star} = \arg \max _ {s \in \mathcal {C} _ {k}} \sum_ {t, p} \mathbb {I} \left[ \hat {s} _ {t, p} ^ {A} = s \right], \tag {10}
$$

where the slot with the largest footprint is chosen as the representative identity.

We then define a relabeling function $\phi ( \cdot )$ and apply $\phi$ to unify pseudo-label assignments to provide coherent object identities for selective alignment:

$$
\phi (s) = \left\{ \begin{array}{l l} s _ {k} ^ {\star} & \text { if   } s \in \mathcal {C} _ {k}, \\ s & \text { otherwise. } \end{array} \right. \tag {11}
$$

$$
\hat {s} _ {t, p} ^ {A} \leftarrow \phi (\hat {s} _ {t, p} ^ {A}), \qquad \hat {s} _ {t, p} ^ {D} \leftarrow \phi (\hat {s} _ {t, p} ^ {D}). \tag {12}
$$

Using the relabeled pseudo-labels, we recompute the local consistency counts and re-derive the boundary/interior sets $\mathcal { P } _ { \mathrm { b d } }$ and $\mathcal { P } _ { \mathrm { n b d } }$ via Eq. $( 4 ) – ( 6 )$ . For simplicity, we reuse $\hat { s } _ { t , p } ^ { A }$ $\hat { s } _ { t , p } ^ { D }$ to denote the merged pseudo-labels after applying $\phi ( \cdot )$ .

## 3.4 Training Objective

Given the refined pseudo-labels obtained after transitive merging, we perform asymmetric cross-distillation between the encoder attention map A and the decoder object map D. For boundary patches at $( t , p ) \in \mathcal { P } _ { \mathrm { b d } }$ , the decoder is supervised using one-hot pseudo-labels derived from the encoder:

$$
\mathcal {L} _ {\mathrm{bd}} = \frac {1}{| \mathcal {P} _ {\mathrm{bd}} |} \sum_ {(t, p) \in \mathcal {P} _ {\mathrm{bd}}} \left\| \mathbf {D} _ {t, p} - \operatorname{onehot} \left(\hat {s} _ {t, p} ^ {A}\right) \right\| _ {2} ^ {2}, \tag {13}
$$

while the interior $( t , p ) \in \mathcal { P } _ { \mathrm { n b d } }$ of the encoder is supervised from the decoder as:

$$
\mathcal {L} _ {\mathrm{nbd}} = \frac {1}{| \mathcal {P} _ {\mathrm{nbd}} |} \sum_ {(t, p) \in \mathcal {P} _ {\mathrm{nbd}}} \left\| \mathbf {A} _ {t, p} - \text { onehot } (\hat {s} _ {t, p} ^ {D}) \right\| _ {2} ^ {2}. \tag {14}
$$

We adopt an MSE objective rather than cross-entropy, as its scale compatibility with the reconstruction loss enables stable joint optimization without loss reweighting, and its bounded gradients improve robustness to imperfect pseudo-labels during early training [34, 35].

Finally, the selective alignment losses are combined with the base objective $\mathcal { L } _ { \mathrm { b a s e } }$ , which includes reconstruction and temporal slot contrastive objective [21] following SRL [14]. Following a warm-up phase covering the first 30% of the total iterations $\eta _ { t } .$ the complete training objective is defined as:

$$
\mathcal {L} = \mathcal {L} _ {\text { base }} + \mathbb {I} [ \eta > 0. 3 \eta_ {\mathrm{t}} ] \cdot \lambda_ {\text { SSync }} \left(\mathcal {L} _ {\mathrm{bd}} + \mathcal {L} _ {\mathrm{nbd}}\right), \tag {15}
$$

where η denotes the current iteration and $\lambda _ { \mathrm { S S y n c } }$ is the coefficient for SSync.

## 4 Experiments

## 4.1 Evaluation Settings

Datasets and Metrics. Following prior works [10, 14, 21], we run experiments on three standard VOCL benchmarks: MOVi-C and MOVi-E [36], and YouTube-VIS (YTVIS) 2021 [37–39]. The primary challenge of MOVi-C is overfragmentation due to complex object interactions, whereas MOVi-E contains a large number of small-scale objects, making boundary precision critical. YTVIS further evaluates robustness under real-world video dynamics. To further assess generalizability, we also evaluate under the RandSF.Q protocol [20], benchmarking on lower-resolution variants of MOVi-C and MOVi-D, as well as the High-Quality YTVIS dataset. Lastly, to verify generalization beyond video domain, we report image-level object-centric performance on MOVi-E and COCO2017 [40].

For metrics, we use Foreground ARI (FG-ARI) [41,42] and mBO [17]. FG-ARI measures permutation-invariant clustering consistency between predicted slot assignments and ground-truth (GT) masks over foreground pixels. mBO evaluates object-level coverage by computing the maximum Intersection-over-Union (IoU) between each GT object and predicted slots. We also report ARI and mIoU when available in prior work.

Table 2: Experimental results. Results are averaged across 3 runs. The best and the second best results are denoted by red and orange.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">MOVi-C</td><td colspan="2">MOVi-E</td><td colspan="2">YouTube-VIS</td></tr><tr><td>FG-ARI↑</td><td>mBO↑</td><td>FG-ARI↑</td><td>mBO↑</td><td>FG-ARI↑</td><td>mBO↑</td></tr><tr><td>SAVi [12]</td><td>ICLR&#x27;22</td><td>22.2</td><td>13.6</td><td>42.8</td><td>16.0</td><td>-</td><td>-</td></tr><tr><td>STEVE [11]</td><td>NeurIPS&#x27;22</td><td>36.1</td><td>26.5</td><td>50.6</td><td>26.6</td><td>15.0</td><td>19.1</td></tr><tr><td>VideoSAUR [10]</td><td>NeurIPS&#x27;23</td><td>64.8</td><td>38.9</td><td>73.9</td><td>35.6</td><td>28.9</td><td>26.3</td></tr><tr><td>VideoSAURv2 [10]</td><td>NeurIPS&#x27;23</td><td>-</td><td>-</td><td>77.1</td><td>34.4</td><td>31.2</td><td>29.7</td></tr><tr><td>SlotContrast [21]</td><td>CVPR&#x27;25</td><td>69.3</td><td>32.7</td><td>82.9</td><td>29.2</td><td>38.0</td><td>33.7</td></tr><tr><td>SRL [14]</td><td>ICLR&#x27;26</td><td>74.3</td><td>34.5</td><td>81.9</td><td>29.3</td><td>42.9</td><td>35.6</td></tr><tr><td>SlotCurri [15]</td><td>CVPR&#x27;26</td><td> $77.6 \pm 0.9$ </td><td> $32.8 \pm 0.2$ </td><td> $83.7 \pm 0.2$ </td><td> $28.9 \pm 0.7$ </td><td> $44.8 \pm 1.2$ </td><td> $35.5 \pm 2.2$ </td></tr><tr><td>SSync (Ours)</td><td></td><td> $79.4 \pm 0.6$ </td><td> $39.5 \pm 0.1$ </td><td> $84.0 \pm 0.9$ </td><td> $34.8 \pm 1.9$ </td><td> $42.6 \pm 0.2$ </td><td> $38.7 \pm 0.6$ </td></tr></table>

Table 3: Results under the RandSF.Q protocol [20]. tsim denotes the time similarity loss from VideoSAUR [10], and SSC refers to the temporal slot contrastive loss from SlotContrast [21]. All baselines are reproduced using their official implementations.

<table><tr><td rowspan="2">Method</td><td colspan="4">MOVi-C</td><td colspan="4">MOVi-D</td><td colspan="4">HQ-YTVIS</td></tr><tr><td>ARI</td><td>FGARI</td><td>mBO</td><td>mIoU</td><td>ARI</td><td>FGARI</td><td>mBO</td><td>mIoU</td><td>ARI</td><td>FGARI</td><td>mBO</td><td>mIoU</td></tr><tr><td>RandSF. $Q_{tsim}$  [20]</td><td>70.7</td><td>63.3</td><td>31.1</td><td>28.1</td><td>39.3</td><td>70.5</td><td>25.6</td><td>24.3</td><td>39.2</td><td>56.3</td><td>37.2</td><td>37.0</td></tr><tr><td>RandSF. $Q_{tsim}$ +SSync</td><td>73.0</td><td>67.3</td><td>32.3</td><td>29.9</td><td>45.5</td><td>71.2</td><td>27.6</td><td>25.7</td><td>42.9</td><td>58.6</td><td>39.4</td><td>39.2</td></tr><tr><td>RandSF. $Q_{ssc}$  [20]</td><td>52.7</td><td>67.8</td><td>24.2</td><td>22.1</td><td>37.3</td><td>85.8</td><td>28.0</td><td>26.8</td><td>40.7</td><td>57.2</td><td>38.3</td><td>37.8</td></tr><tr><td>RandSF. $Q_{ssc}$ +SSync</td><td>55.5</td><td>71.4</td><td>25.1</td><td>23.1</td><td>39.4</td><td>86.5</td><td>27.9</td><td>26.8</td><td>48.6</td><td>57.5</td><td>42.1</td><td>41.9</td></tr></table>

Implementation Details. We adopt prior training protocols [14, 20, 21] to ensure fair comparison; all architectural components and base objectives follow the respective baselines. SSync additionally introduces nbd, nnbd, $\lambda _ { \mathrm { S S y n c } }$ and $\tau _ { \mathrm { m e r g e } } .$ Among them, we fix $n _ { \mathrm { b d } } = n _ { \mathrm { n b d } } = 1$ and $\lambda _ { \mathrm { S S y n c } } = 1 . 0$ across all datasets, which demonstrates SSync’s robustness and its minimal requirement for dataset-specific tuning. We only adjust the merging threshold $\tau _ { \mathrm { m e r g e } } ,$ setting it to 0.7 for MOVi-C, 0.65 for MOVi-E, and 0.6 for YTVIS. This accounts for varying object densities, reflecting theoretically established principles in clustering where overlap criteria must adapt to spatial density [44, 45]. Importantly, we

observe that performance remains consistent within a reasonable range around $^ { 2 / 3 , }$ indicating that SSync does not rely on exhaustive threshold tuning. Details are in the Appendix.

Table 4: Image objectcentric learning.  
(a) Results on MOVi-E.

<table><tr><td>Method</td><td>FG-ARI ↑</td></tr><tr><td>VideoSAUR [10]</td><td>78.4</td></tr><tr><td>SOLV [43]</td><td>80.8</td></tr><tr><td>SlotContrast [21]</td><td>84.8</td></tr><tr><td>SlotCurri</td><td>84.9</td></tr><tr><td>SSync</td><td>86.0</td></tr></table>

(b) Results on COCO.

<table><tr><td>Method</td><td colspan="2">FG-ARImBO</td></tr><tr><td>Baseline</td><td>40.5</td><td>28.8</td></tr><tr><td>SRL [14]</td><td>42.8</td><td>29.4</td></tr><tr><td>SlotCurri [15]</td><td>43.4</td><td>28.9</td></tr><tr><td>SSync</td><td>47.9</td><td>33.1</td></tr></table>

## 4.2 Comparison to the State-of-the-art (SOTA) Methods

Video Benchmarks. Comparison to SOTA VOCL methods is shown in Tab. 2, where SSync consistently achieves superior or highly competitive performance compared to prior approaches across benchmarks. On MOVi-C, where objects are frequently decomposed into multiple slots, SSync achieves the best results by consolidating fragmented identities through selective alignment. SSync also yields substantial gains on MOVi-E, where the primary challenge is to accurately capture the boundaries of numerous small objects. Lastly, on YTVIS dataset,

SSync achieves competitive FG-ARI while obtaining the highest mBO, indicating stronger object coverage. It is worth noting that YTVIS provides GT annotations only for a sparse set of salient foreground objects. Therefore, these metrics may not fully capture qualitative differences outside the evaluated objects. Accordingly, we provide qualitative comparisons in the Appendix; in the shown examples, SSync appears to distinguish secondary objects and background regions more consistently than competing methods. Overall, these results suggest that selective alignment more effectively resolves encoder–decoder discrepancy than the dense alignment strategy, despite its simplicity, requiring no additional architectural components used by SRL [14].

In addition, we integrate SSync into two variants of RandSF.Q [20] built upon VideoSAUR [10] and SlotContrast [21]. As reported in Tab. 3, SSync consistently improves performance, which indicates that SSync functions as a plug-and-play method, reinforcing its practical applicability.

Image Benchmarks. To evaluate generalization beyond video benchmarks, we assess SSync on image-level object-centric benchmarks. To illustrate, SSync achieves a SOTA FG-ARI of 86.0 on MOVi-E, and attains an ARI of 47.9 and mBO of 33.1 on real-world COCO2017, substantially outperforming SRL. This suggests that the proposed SSync improves spatial consistency independently of motion signals, further validating its robustness across diverse visual domains.

Memory Efficiency. We compare the memory footprint against SRL [14] using mixedprecision (FP16) on YTVIS (518×518 resolution) in Tab. 5 (maximum VRAM per GPU). SRL exhibits quadratic memory growth $( \mathcal { O } ( ( T \cdot H \cdot W ) ^ { 2 } ) )$ ) due to dense patch-to-patch comparisons, whereas SSync scales approximately linearly with the number of patches. As a result, at batch size 32 per GPU (T =4), SRL requires 70GB while SSync uses only 27GB, reducing memory usage by roughly 60% (and only a marginal 5% increase over Slot-Contrast [21]). Furthermore, SRL already encoun-

Table 5: Memory comparison (SRL/SSync). Values represent the maximum VRAM allocated per GPU in GB. OOM indicates Out-Of-Memory errors on an NVIDIA RTX PRO 6000 Blackwell GPU (97GB).

<table><tr><td rowspan="2">Frames (T)</td><td colspan="2">Batch Size per GPU</td></tr><tr><td>32</td><td>64</td></tr><tr><td>T=4</td><td>70 / 27</td><td>OOM / 59</td></tr><tr><td>T=6</td><td>OOM / 48</td><td>OOM / 89</td></tr><tr><td>T=8</td><td>OOM / 60</td><td>OOM / 93</td></tr></table>

ters OOM at larger temporal lengths or batch sizes on our GPU, whereas SSync remains feasible. Note that per GPU VRAM values correspond to the maximum reserved memory, thereby not strictly linear with increasing T due to the properties of CUDA and cuDNN.

## 4.3 Ablation Study

All studies are conducted on MOVi-C.

Component Ablation. We analyze the contribution of each component in Tab. 6. Baseline model achieves 69.0 FG-ARI and 30.6 mBO. Applying boundary supervision $( { \mathcal { L } } _ { \mathrm { b d } } )$ alone improves performance to 72.9 FG-ARI and 33.4 mBO. This confirms that selectively distilling sharp boundary cues from the encoder effectively refines decoder object maps. Similarly, interior supervision $( { \mathcal { L } } _ { \mathrm { { n b d } } } )$ alone yields 71.4 FG-ARI and 33.3 mBO, indicating that decoder-guided denoising stabilizes noisy encoder assignments, leading to stable training. When both selective alignment losses are combined, performance increases substantially to 77.1 FG-ARI and 38.0 mBO, demonstrating that boundary calibration and interior denoising are complementary. Finally, incorporating transitive merging further improves performance to 79.4 FG-ARI and 39.5 mBO. This additional gain highlights the importance of stabilizing pseudo-label identities before

Table 6: Component ablation study. ${ \mathcal { L } } _ { \mathrm { b d } } , ~ { \mathcal { L } } _ { \mathrm { n b d } } ,$ and T.M. represent calibrating the decoder boundary, denoising the attention map, and transitive pseudo-label merging, respectively.

<table><tr><td colspan="3">Selected Components</td><td colspan="2">MOVi-C</td></tr><tr><td> $\mathcal{L}_{\text{bd}}$ </td><td> $\mathcal{L}_{\text{nbd}}$ </td><td>T.M.</td><td>FG-ARI</td><td>mBO</td></tr><tr><td></td><td></td><td></td><td>69.0</td><td>30.6</td></tr><tr><td>√</td><td></td><td></td><td>72.9</td><td>33.4</td></tr><tr><td></td><td>√</td><td></td><td>71.4</td><td>33.3</td></tr><tr><td>√</td><td>√</td><td></td><td>77.1</td><td>38.0</td></tr><tr><td>√</td><td>√</td><td>√</td><td>79.4</td><td>39.5</td></tr></table>

cross-distillation. Overall, these results validate the selective alignment principle: region-aware supervision improves both boundary precision and slot consistency, while redundancy-aware merging ensures global semantic coherence.

Robustness to Number of Slots. A major challenge in video slot attention is over-fragmentation, which worsens when the number of slots (S) exceeds the actual objects in a scene. This forces prior works [14, 21] to carefully tune $S$ per dataset to prevent multiple slots from undesirably cooperating to reconstruct a single object. As shown in Tab. 7, when prior approaches are trained with varying slot counts $( S \in \{ 7 , 1 1 , 1 5 \} )$ , the performance deteriorates significantly at $S = 1 5$ , indicating severe sensitivity to slot over-parameterization. In contrast, SSync maintains consistent performance across all configurations. This stability is driven by our transitive pseudo-label merging, which leverages spatiotemporal consistency to consolidate overlapping, fragmented slots into unified object representations. By inherently counteracting over-fragmentation, SSync drastically reduces sensitivity to the hyperparameter S, alleviating the burden of dataset-specific tuning and offering a highly practical framework for VOCL.

Transitive Pseudo-Label Merging. A key design choice in transitive merging is the source of active-region extraction used to identify redundant slots. While we derive active regions from the encoder attention map A, we evaluate alternative formulations in Tab. 8 (a). Specifically, we compare: (a0) deriving active regions solely from the decoder object map D; (a1–a2) hybrid logical criteria, where redundancy is independently computed from both A and D via Eq. 7–9 and combined using union (merge if either suggests redundancy) or intersection (merge only if both agree); and (a3) averaging IoU scores from A and D (Eq. 8 applied to each map) prior to thresholding with $\tau _ { \mathrm { m e r g e } }$ in Eq. 9. Although using D as the merging source yields slightly lower performance, the overall results remain consistently strong across all configurations.

Table 7: Performance comparison with varying number of slots.

<table><tr><td rowspan="2">Method</td><td colspan="2">slot=7</td><td colspan="2">slot=11</td><td colspan="2">slot=15</td></tr><tr><td>FG-ARI↑</td><td>mBO↑</td><td>FG-ARI↑</td><td>mBO↑</td><td>FG-ARI↑</td><td>mBO↑</td></tr><tr><td>SlotContrast [21]</td><td>74.9</td><td>27.9</td><td>69.3</td><td>32.7</td><td>61.8</td><td>31.2</td></tr><tr><td>SRL [14]</td><td>76.5</td><td>31.6</td><td>74.3</td><td>34.5</td><td>72.8</td><td>31.1</td></tr><tr><td>Ours</td><td>76.9</td><td>39.8</td><td>79.4</td><td>39.5</td><td>78.8</td><td>41.0</td></tr></table>

This robustness suggests that redundancy detection reflects intrinsic spatio-temporal activation patterns, rather than dependence on a specific map representation.

We also compare transitive merging with a pairwise baseline (Tab. 8(b)). Whereas ours constructs a connectivity graph over redundant slots and merges all slots within a connected component simultaneously, pairwise merging combines only the most similar slot pair whose IoU exceeds $\tau _ { \mathrm { m e r g e } } ,$ . Results suggest that over-fragmented objects are often distributed across multiple slots, requiring global consolidation rather than pairwise agglomeration.

Table 8: Variants of transitive merging strategies. A and D denote attention map and decoder object map, respectively.

<table><tr><td></td><td>Strategy</td><td>FG-ARI↑</td><td>mBO↑</td></tr><tr><td></td><td>Ours</td><td>79.4</td><td>39.5</td></tr><tr><td></td><td colspan="3">Varying criteria for merging</td></tr><tr><td>(a0)</td><td>D</td><td>77.5</td><td>38.7</td></tr><tr><td>(a1)</td><td>A ∨ D</td><td>78.6</td><td>39.8</td></tr><tr><td>(a2)</td><td>A ∧ D</td><td>78.9</td><td>39.1</td></tr><tr><td>(a3)</td><td>Avg(A, D)</td><td>79.2</td><td>40.0</td></tr><tr><td></td><td colspan="3">Merging Range</td></tr><tr><td>(b)</td><td>Pairwise</td><td>78.2</td><td>39.3</td></tr><tr><td></td><td colspan="3">Comparison to Slot Reg. [14]</td></tr><tr><td>(c)</td><td>Slot Reg.</td><td>74.6</td><td>35.8</td></tr></table>

Finally, we compare our merging strategy with the slot regularization method used in SRL [14] (Tab. 8(c)). Slot regularization mitigates redundancy by penalizing overlapping slot pairs only during a predefined warm-up stage. In contrast, our method explicitly consolidates redundant slot identities based on spatio-temporal coverage similarity throughout the training. Empirical results demonstrate that transitive merging provides a more effective remedy for over-fragmentation, while eliminating the need for heuristic scheduling of regularization phases.

Impact of $\lambda _ { \mathrm { S S y n c } } . ~ \lambda _ { \mathrm { S S y n c } }$ controls the relative weight of the SSync loss against the base objective. As shown in Tab. 9a, SSync consistently improves performance over the baseline (69.0 FG-ARI, 30.6 mBO). Performance increases steadily as $\lambda _ { \mathrm { S S y n c } }$ grows until $\lambda _ { \mathrm { S S y n c } } ~ = ~ 1 . 0$ , where selective alignment and reconstruction are well balanced. Beyond this point, further increasing the weight yields diminishing returns, indicating that excessive alignment may over-constrain the representation. Nevertheless, performance remains stable for slightly larger values $( \mathrm { e . g . , \lambda _ { S S y n c } = 1 . 2 } )$ , demonstrating robustness to moderate over-weighting.

Impact of $n _ { \mathrm { b d } }$ and $n _ { \mathrm { n b d } } . \mathrm { ~ } n _ { \mathrm { b d } }$ and $n _ { \mathrm { n b d } }$ determine the sensitivity of boundary and interior region selection. Larger $n _ { \mathrm { b d } }$ imposes stricter boundary criteria, while larger $n _ { \mathrm { n b d } }$ relaxes interior selection. Since interior regions typically occupy the majority in most scenes, we fix $n _ { \mathrm { b d } } = n _ { \mathrm { n b d } } = 1$ to balance between the complementary supervision signals. Yet, our empirical results (Tab. 9b and 9c) confirm that performance remains stable across a reasonable range of values.

Impact of $\tau _ { \mathrm { m e r g e } }$ . The merging threshold $\tau _ { \mathrm { m e r g e } }$ controls the required spatiotemporal overlap for consolidating redundant slots. At the extremes, $\tau _ { \mathrm { m e r g e } } = 1 . 0$ disables merging, while $\tau _ { \mathrm { m e r g e } } = 0$ collapses all slots into a single identity. Tab. 9d shows that $\tau _ { \mathrm { m e r g e } } = 0 . 7$ performs best on MOVi-C. Yet, the performance remains consistently high for thresholds around $2 / 3 .$ This indicates that redundancy consolidation depends primarily on a clear overlap structure rather than precise threshold tuning, further validating the robustness of transitive merging.

Table 9: Hyperparameter analysis. The gray row denotes our default configuration.  
(a) Impact of $\lambda _ { \mathrm { S S y n c } }$ .

<table><tr><td> $\lambda_{SSync}$ </td><td>FG-ARI</td><td>mBO</td></tr><tr><td>0.1</td><td>74.1</td><td>37.8</td></tr><tr><td>0.5</td><td>79.1</td><td>39.0</td></tr><tr><td>1.0</td><td>79.4</td><td>39.5</td></tr><tr><td>1.2</td><td>78.9</td><td>40.2</td></tr></table>

(b) Impact of nbd.

<table><tr><td> $n_{bd}$ </td><td>FG-ARI</td><td>mBO</td></tr><tr><td>1</td><td>79.4</td><td>39.5</td></tr><tr><td>2</td><td>78.9</td><td>39.3</td></tr><tr><td>3</td><td>78.2</td><td>39.4</td></tr><tr><td>4</td><td>76.6</td><td>36.7</td></tr></table>

(c) Impact of $n _ { \mathrm { n b d } } .$

<table><tr><td> $n_{\text{nbd}}$ </td><td>FG-ARI</td><td>mBO</td></tr><tr><td>1</td><td>79.4</td><td>39.5</td></tr><tr><td>2</td><td>78.9</td><td>41.2</td></tr><tr><td>3</td><td>78.8</td><td>40.1</td></tr><tr><td>4</td><td>76.7</td><td>36.6</td></tr></table>

(d) Impact of τmerge.

<table><tr><td> $\tau_{\text{merge}}$ </td><td>FG-ARI</td><td>mBO</td></tr><tr><td>0.65</td><td>78.3</td><td>40.1</td></tr><tr><td>0.7</td><td>79.4</td><td>39.5</td></tr><tr><td>0.75</td><td>79.1</td><td>39.5</td></tr><tr><td>0.8</td><td>78.2</td><td>39.0</td></tr></table>

## 4.4 Analysis

Analysis of the impact of denoising and deblurring. To quantify the impact in reducing isolated noise and boundary spillover, we report two diagnostics on MOVi-C in Tab. 10. First, the frame-averaged connected components (FCC) measures fragmentation: for each frame, we count the number of connected components in each per-slot binary mask and sum them over slots, then average across frames. A lower FCC indicates a reduction in isolated noise and over-fragmentation. SSync significantly suppresses spurious predictions, approaching the GT reference of 6.27. Second, we measure boundary spillover to assess spatial precision. We first match GT objects and predicted slots and retain only matched pairs

Table 10: Effects of encoder denoising and decoder deblurring on MOVi-C. $\mathrm { F C C } _ { 8 }$ measures mask fragmentation via the average number of connected components per frame, where connectivity is defined using an 8-neighborhood. GT $\mathrm { F C C } _ { \mathrm { 8 } }$ is 6.27. Match90 reports the number of matched GT–slot pairs achieving at least 90% GT coverage. For these qualified pairs, we report the outside leakage (Leak) which indicates the total number of pixels assigned to the matched slot but lying outside the corresponding GT mask, summed over the spatio-temporal volume at the original video resolution $( \times 1 0 ^ { 3 }$ pixels).

<table><tr><td rowspan="2">Method</td><td>Encoder</td><td colspan="2">Decoder</td></tr><tr><td> $FCC_8 \downarrow$ </td><td> $Match_{90} \uparrow$ </td><td> $Leak \downarrow$ </td></tr><tr><td>SlotContrast [21]</td><td>33.20</td><td>639</td><td>98.95</td></tr><tr><td>SRL [14]</td><td>21.03</td><td>684</td><td>86.26</td></tr><tr><td>SSync (Ours)</td><td>8.79</td><td>702</td><td>72.02</td></tr></table>

whose GT coverage is at least 90% $\left( \mathrm { M A T C H _ { 9 0 } } \right)$ . For these pairs, we quantify outside leakage as the number of predicted pixels lying outside the GT mask (lower indicates deblurred boundary). SSync not only yields more qualified matches than SRL, but also reduces the mean outside leakage per video by 16.5%. Collectively, these demonstrate that SSync excels at denoising fragmented assignments and sharpening slot coverage at object boundaries. Full results are in the Appendix.

Boundary Analysis. While mBO measures region overlap, it may not fully capture boundary sharpness. Thus, we evaluate boundary F -score [46] in Tab. 11. As shown, SSync improves F -score, supporting our claim that selective alignment leverages the encoder’s boundarysensitive cues to refine object contours.

Table 11: Boundary F -score on MOVi-C.

<table><tr><td>Method</td><td>F-score ↑</td></tr><tr><td>SlotContrast [21]</td><td>0.184</td></tr><tr><td>SRL [14]</td><td>0.222</td></tr><tr><td>SSync (Ours)</td><td>0.255</td></tr></table>

Qualitative Results. In Fig. 3, we present qualitative comparisons on MOVi-C. SlotContrast [21] exhibits severe noisy mask predictions, accompanied by spatially inflated and blurred object boundaries. Although SRL [14] mitigates part of this instability, residual noisy assignments and boundary ambiguity remain evident. In contrast, SSync produces more coherent object decomposition; boundaries are sharper and spatially consistent, while interior regions exhibit stable semantic grouping. Notably, background regions (although not evaluated by foreground-specific metrics like FG-ARI used in Tab. 2) are consistently represented as unified and temporally coherent entities, while these regions are often fragmented across multiple slots in competing works. Also, we visualize the selected boundary and interior regions at the bottom of Fig. 3. Results illustrate that the model effectively separates spatio-temporal object transitions from interior regions across the video sequence. Additional qualitative comparisons and detailed region-selection visualizations are provided in the Appendix.

![](images/41b54993e074e0803224bd6b81a4d961f9e54800f680dc5b175ee07b1a0179ec.jpg)

<details>
<summary>text_image</summary>

Video
GT
SlotContrast
SRL
Ours
Non
Boundary
Boundary
</details>

Fig. 3: Qualitative comparison on MOVi-C. From top to bottom, we visualize the input, GT masks, predictions from SlotContrast, SRL, and SSync, followed by the non-boundary and boundary regions selected by SSync for interior and boundary supervision, respectively.

## 5 Conclusion

In this paper, we proposed Selective Synergistic Learning (SSync) for VOCL. SSync achieves impressive performance gains by both effectively and efficiently mitigating the discrepancy between the encoder’s slot attention maps and the decoder’s object maps. Specifically, SSync performs selective alignment between the attention map and object map only on regions where each map possesses its respective expertise, enabled by an efficient pseudo-labeling scheme. To further reduce the risk of pseudo-label corruption caused by object over-fragmentation, we introduced a transitive pseudo-label merging mechanism. By analyzing spatiotemporal slot activations and merging redundant slots via a connectivity-based criterion, we refine pseudo-label targets to be more semantically coherent and robust throughout training. Extensive experiments across VOCL benchmarks and additional evaluation protocols demonstrate the effectiveness of SSync, while remaining modular and easy to integrate into existing slot-based pipelines.

Table 12: Hyperparameters for SSync training on MOVi-C, MOVi-E, and YouTube-VIS 2021.

<table><tr><td>Hyperparameter</td><td>MOVi-C</td><td>MOVi-E</td><td>YouTube-VIS</td></tr><tr><td colspan="4">Training Configurations</td></tr><tr><td>Training Steps</td><td>100k</td><td>300k</td><td>100k</td></tr><tr><td>Batch Size</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning Rate</td><td>0.0008</td><td>0.0008</td><td>0.0008</td></tr><tr><td>ViT Architecture</td><td>DINOv2-Small</td><td>DINOv2-Base</td><td>DINOv2-Base</td></tr><tr><td>Image Size</td><td> $336 \times 336$ </td><td> $336 \times 336$ </td><td> $518 \times 518$ </td></tr><tr><td colspan="4">Slot Attention &amp; Architecture</td></tr><tr><td>Number of Slots</td><td>11</td><td>15</td><td>7</td></tr><tr><td>Slot Dimension ( $D_{\text{slots}}$ )</td><td>64</td><td>128</td><td>64</td></tr><tr><td>Iterations (first / other frames)</td><td>3 / 2</td><td>3 / 2</td><td>3 / 2</td></tr><tr><td>Decoder Type</td><td>MLP</td><td>MLP</td><td>MLP</td></tr><tr><td colspan="4">SSync Parameters (Ours)</td></tr><tr><td>Boundary Sensitivity ( $n_{bd}$ )</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Non-boundary Sensitivity ( $n_{nbd}$ )</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Loss Coefficient ( $\lambda_{SSync}$ )</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Merging Threshold ( $\tau_{merge}$ )</td><td>0.7</td><td>0.65</td><td>0.6</td></tr></table>

## 6 Training Details

Regarding the training configurations, we followed SlotContrast $[ 2 1 ] ^ { 4 }$ and $\mathrm { S R L } [ 1 4 ] ^ { 5 }$ to set up the experiments in Tab. 2 and Tab. 4. Across all datasets, we maintain a uniform batch size of 128 and a learning rate of 8e-4. For our hardware resources, all experiments were conducted using 2× NVIDIA RTX A6000 GPUs. The only exception is the YouTube-VIS dataset; due to its higher input resolution of 518×518, requiring greater VRAM, we performed those specific experiments on the NVIDIA RTX PRO 6000 Blackwell GPUs.

In addition, for Tab. 3, we followed RandSF.Q [20]. We use DINOv2-Small for all datasets to process 224 × 224 images, and trained for 50000 iterations for all experiments. Our hyperparameters $( n _ { \mathrm { b d } } , n _ { \mathrm { n b d } } , \lambda _ { \mathrm { S S y n c } } , \mathrm { a n d } \tau _ { \mathrm { m e r g e } } )$ are kept consistent across all benchmarks to demonstrate the robustness of our framework, while we utilize the same value for MOVi-D as established for MOVi-C. For more details of this setting, we refer to the scripts in the official repository6.

## 7 Detailed Comparison with SRL

To further demonstrate the practical advantages of SSync, we provide a detailed comparison with SRL [14], summarized in Table 13. Our design emphasizes simplifying mutual learning while improving scalability, efficiency, and robustness. Unlike SRL, which introduces two additional MLP projectors to map encoder and decoder features into the embedding space for contrastive learning, SSync operates directly on the native output space (i.e., slot attention maps). This projector-free design introduces no additional parameters, making SSync a fully plug-and-play module that can be integrated into existing slot-based architectures with minimal implementation overhead.

Table 13: Detailed Comparison with SRL. Compared to SRL, SSync eliminates auxiliary projectors and complex contrastive objectives, achieving linear complexity and higher parameter efficiency.

<table><tr><td>Aspect</td><td>SRL [14]</td><td>SSync (Ours)</td></tr><tr><td>Alignment Scope</td><td>Dense (all patches)</td><td>Selective (boundary/interior)</td></tr><tr><td>Alignment Method</td><td>Ternary contrastive loss</td><td>Pseudo-label MSE</td></tr><tr><td>Redundancy Handling Complexity</td><td>Warm-up Reg. (fixed) $\mathcal{O}((T \cdot H \cdot W)^2)$ </td><td>Transitive Merging (adaptive) $\mathcal{O}(T \cdot H \cdot W)$ </td></tr><tr><td>Extra Modules</td><td>2 Projectors</td><td>None</td></tr></table>

SSync also significantly reduces computational complexity. SRL relies on a ternary contrastive loss that computes dense similarities across all spatio-temporal patches, resulting in quadratic complexity, $\mathcal { O } ( ( T \cdot H \cdot W ) ^ { 2 } )$ . In contrast, SSync applies a selective MSE objective only to reliable regions, reducing the complexity to linear time, $\mathcal { O } ( T \cdot H \cdot W )$ . This enables scalable training on higher-resolution inputs and longer video sequences where SRL becomes memory-intensive.

Regarding redundancy mitigation, SRL relies on a slot regularization technique $\left( \lambda _ { \mathrm { r e g } } \right)$ paired with a complex warm-up schedule. This approach necessitates intricate hyperparameter tuning to synchronize multiple warm-up phases and strictly enforces a uniform distribution to keep exactly half of the slots (fixed) vacant during the initial stage. In contrast, SSync introduces transitive pseudo-label merging, an adaptive strategy that dynamically resolves redundancy throughout the entire training process by responding to the real-time evolution of the scene composition.

Finally, SRL’s ternary contrastive objective enforces dense alignment across all regions, implicitly assuming uniform reliability between encoder and decoder outputs. In contrast, SSync reformulates alignment as selective cross-distillation, allowing each branch to supervise only the regions where it is most reliable. This targeted supervision avoids propagating structural weaknesses such as encoder noise and decoder blur, leading to improved object decomposition quality.

## 8 Qualitative illustration of SSync Mechanisms.

To provide a comprehensive understanding of SSync, we illustrate the training mechanism of SSync in Fig. 4. Our synergistic objectives are activated following the warm-up covering 30% of the total training iteration. At the onset of activation, the attention map exhibits noisy patches, while the object map suffers from blurry object boundaries; also, both suffer from object over-fragmentation. To extract reliable supervision from these imperfect maps, we first apply transitive merging, which identifies and fuses redundant slots by analyzing their spatiotemporal overlap, yielding semantically coherent groupings even before global convergence. The bottom right panels then depict our selective alignment principle: we identify boundary regions $( \mathcal { P } _ { b d } )$ from the sharp attention maps and nonboundary (interior) regions $( \mathcal { P } _ { n b d } )$ from the consistent decoder maps. With this reliable mutual supervision, the model refines its prediction iteratively. Consequently, as training progresses to 40% of the total training iterations, SSync results in significantly cleaner object discovery, where each slot captures a unified object identity with precise boundaries and consistent interior semantics.

![](images/7a5678c6078064069f986cd2ee78d4ef5520abca225b4f15317485f060c62fa9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Video"] --> B["Slot Attention Map"]
  A --> C["Decoder Object Map"]
  B --> D["Transitive Pseudo-Label Merging"]
  C --> D
  D --> E["Boundary Detection"]
  D --> F["Non-Boundary Detection"]
  E --> G["30% iter"]
  F --> G
  G --> H["40% iter"]
```
</details>

Fig. 4: Visualization of the Evolution of SSync.

## 9 Boundary & Non-Boundary (Interior) Analysis

In this section, we further visualize the patches classified as boundary and interior regions by SSync. Fig. 5 presents representative frames from 10 example videos. The visualizations show that SSync effectively identifies object boundary regions and semantically consistent interior regions. Boundary patches are predominantly located along object transitions, while interior patches concentrate within stable object cores. This demonstrates that the local consistency criterion successfully captures high-frequency structural transitions without disrupting interior coherence.

![](images/4b15ff9fd57453010c9cd4293b4f35b7fd98553b9af472aa85086a1a2e091df2.jpg)

<details>
<summary>text_image</summary>

Grid of video and non Boundary detection panels with red pixelated footprints, showing object detection and spatial patterns
</details>

Fig. 5: Visualizations of detected boundary and non-boundary patches.

Table 14: Analysis of encoder denoising via spatial fragmentation on MOVi-C. We measure fragmentation using Frame-averaged Total Connected Components (FCC), where we count the number of connected components for each slot’s binary mask per frame and sum them across all slots. A lower FCC indicates a reduction in isolated noise and over-fragmentation. GT FCC is constant across methods.

<table><tr><td rowspan="2">Method</td><td colspan="2"> $FCC_4$ </td><td colspan="2"> $FCC_8$ </td></tr><tr><td>GT</td><td>Pred ↓</td><td>GT</td><td>Pred ↓</td></tr><tr><td>SlotContrast [21]</td><td>7.273</td><td>33.785</td><td>6.268</td><td>33.205</td></tr><tr><td>SRL [14]</td><td>7.273</td><td>21.338</td><td>6.268</td><td>21.030</td></tr><tr><td>SSync (Ours)</td><td>7.273</td><td>8.899</td><td>6.268</td><td>8.793</td></tr></table>

We note that, in some cases, patches inside an object are classified as boundary regions. This occurs when the semantic content at a spatial location changes across consecutive frames due to object motion. In such cases, the patch is treated as a temporal boundary, allowing the model to capture dynamic transitions along the temporal axis. This behavior indicates that the boundary selection mechanism is not limited to spatial edges but also adapts to spatio-temporal changes, facilitating improved modeling of object dynamics.

## 10 Impact of Denoising and Deblurring

We report the full analysis results corresponding to Tab. 10 to quantify the effects of denoising and deblurring. First, to validate the denoising effect of SSync, we measure the reduction in the number of fragmented components. Tab. 14 reports frame-averaged connected components (FCC), which measure spatial fragmentation of predicted slot masks. SlotContrast exhibits severe overfragmentation, with FCC values exceeding 33 under both 4- and 8-neighbor connectivity. SRL partially mitigates this issue, reducing FCC to approximately 21. In contrast, SSync substantially lowers fragmentation $\left( \mathrm { F C C } _ { 8 } \colon 8 . 7 9 3 \right)$ , approaching the ground-truth structure (6.268). This pronounced reduction indicates that selective interior denoising effectively suppresses noisy slot assignments and consolidates object identities.

To further validate the effectiveness of SSync in refining object boundaries, we evaluate outside leakage in Tab. 15. Given a GT-slot pair where the predicted slot mask covers over 75% or 90% of the GT mask, we quantify the area of the predicted slot that spills over into the background. As reported, SSync consistently achieves the lowest leakage across both coverage thresholds while simultaneously increasing the number of high-coverage matches. These results demonstrate the effectiveness of selectively refining the decoder boundary representation by leveraging the attention maps’ boundary sensitivity. Altogether, the fragmentation and leakage analyses confirm that SSync excels at denoising the representations in the encoder attention map and deblurring the spatially inflated boundaries in the decoder object map by selectively distilling the complementary strengths of each module.

Table 15: Analysis of decoder deblurring via measuring outside leakage on MOVi-C. For each ground-truth (GT) object, we match the best corresponding predicted slot and compute the GT coverage of that slot. We report the number of matched slots achieving at least 75% and 90% GT coverage, denoted as Match75 and Match90, respectively. For these matched slots, we measure outside leakage, defined as the number of pixels assigned to the slot but lying outside the corresponding GT mask, aggregated over the full spatio-temporal volume of the original video resolution (24 × 336 × 336).

<table><tr><td rowspan="2">Method</td><td colspan="2">GT Coverage ≥75%</td><td colspan="2">GT Coverage ≥90%</td></tr><tr><td>Leak (×103)↓</td><td>Match75↑</td><td>Leak (×103)↓</td><td>Match90↑</td></tr><tr><td>SlotContrast [21]</td><td>95.22</td><td>915</td><td>98.95</td><td>639</td></tr><tr><td>SRL [14]</td><td>82.06</td><td>952</td><td>86.26</td><td>684</td></tr><tr><td>SSync (Ours)</td><td>71.03</td><td>957</td><td>72.02</td><td>702</td></tr></table>

## 11 Evolution of Synergistic Refinement

To further investigate the mutual refinement process between the encoder and decoder, we analyze the evolution of three diagnostic metrics throughout the training: (1) outside leakage (for boundary blurring), (2) FCC (for spatial denoising), and (3) the overlap ratio between the encoder-derived boundary set $\mathcal { P } _ { \mathrm { b d } }$ and the decoder-derived non-boundary set $\mathcal { P } _ { \mathrm { n b d } }$ . We adopt these specific proxies rather than direct region-wise (boundary and non-boundary) evaluations because the disproportionate scale of interior regions tends to dilute the sensitivity of noise measurements, making direct interior-based metrics less informative. As shown in Tab. 16, the synergistic effect of SSync enables each module to iteratively overcome its inherent inductive biases by leveraging the other’s strengths.

To illustrate, the decoder object map (D) suffers from blurry boundaries (high outside leakage) due to the smoothing nature of the reconstruction objective. For instance, at 30% of training, the decoder’s outside leakage (GT Coverage ≥75%) is approximately $9 3 . 9 6 \times 1 0 ^ { 3 }$ patches. However, as the decoder receives sharp boundary guidance from the encoder’s attention maps through our selective alignment loss $( \mathcal { L } _ { b d } )$ , its leakage significantly drops to $\bar { 7 } 1 . 0 3 \times \bar { 1 } 0 ^ { 3 }$ by the end of training. This sharp reduction demonstrates that the encoder’s boundary sensitivity effectively deblurs the decoder’s spatial assignments.

Conversely, the encoder’s attention assignments are initially susceptible to noisy predictions, as reflected by a high FCC of 41.48 (connectivity measured with 4 spatial neighborhood patches) at 30% iterations. Yet, as the inherently less noisy decoder output (FCC = 13.15 at 30% of total iterations) is distilled to the interior regions of the encoder attention map via ${ \mathcal { L } } _ { n b d }$ , the encoder’s FCC sharply decreases to 8.90, effectively removing isolated noisy patches and leading to more coherent object discovery. Collectively, these metrics confirm that the selective synergistic mechanism transforms a potential mismatch between the two modules into mutual improvement.

Table 16: Evolution of deblurring and denoising metrics on MOVi-C. We measure boundary blurring via outside leakage $( \times 1 0 ^ { 3 }$ pixels; lower is better) and spatial noise via FCC (lower is better). The synergistic refinement allows the decoder to sharpen boundaries and the encoder to suppress noise over training iterations. The key indicators of deblurring and denoising are emphasized with yellow for better clarity.

<table><tr><td>Metric</td><td>Module</td><td>30% Iter</td><td>50% Iter</td><td>100% Iter</td></tr><tr><td colspan="5">Decoder Deblurring Effect</td></tr><tr><td rowspan="2">Outside Leakage (GT Coverage ≥75%) ↓</td><td>Decoder (D)</td><td>93.96</td><td>75.20</td><td>71.03</td></tr><tr><td>Encoder (A)</td><td>74.60</td><td>73.54</td><td>70.18</td></tr><tr><td rowspan="2">Outside Leakage (GT Coverage ≥90%) ↓</td><td>Decoder (D)</td><td>99.21</td><td>80.72</td><td>72.02</td></tr><tr><td>Encoder (A)</td><td>76.89</td><td>77.89</td><td>73.65</td></tr><tr><td colspan="5">Encoder Denoising Effect</td></tr><tr><td rowspan="2">FCC (4 spatial neighborhood) ↓</td><td>Decoder (D)</td><td>13.75</td><td>7.01</td><td>6.88</td></tr><tr><td>Encoder (A)</td><td>41.48</td><td>9.39</td><td>8.90</td></tr><tr><td rowspan="2">FCC (8 spatial neighborhood) ↓</td><td>Decoder (D)</td><td>13.15</td><td>6.89</td><td>6.76</td></tr><tr><td>Encoder (A)</td><td>40.75</td><td>9.29</td><td>8.79</td></tr></table>

Table 17: Evolution of the overlap ratio between encoder-derived boundary patches $\mathcal { P } _ { b d }$ and decoder-derived non-boundary patches $\mathcal { P } _ { n b d }$ .

<table><tr><td>Metric</td><td>30% Iter</td><td>50% Iter</td><td>100% Iter</td></tr><tr><td>Overlap ratio (IoU) between  $\mathcal{P}_{bd}$  and  $\mathcal{P}_{nbd}$ </td><td>0.277</td><td>0.064</td><td>0.059</td></tr></table>

Beyond the improvements in leakage and fragmentation, we track the spatial evolution of the two selective supervision sets during training. Specifically, we measure the overlap ratio between the encoder-derived boundary patches $\mathcal { P } _ { b d } \ ( \mathrm { E q . \ 5 } )$ and the decoder-derived non-boundary patches $\mathcal { P } _ { n b d }$ (Eq. 6). At the activation point (30% of iterations), the overlap (measured with IoU) is relatively high (0.277), reflecting that the boundary-interior separation remains imperfect immediately following the warm-up phase. As training proceeds, the overlap ratio rapidly decreases to 0.064 at 50% and stabilizes at 0.059 at convergence, indicating that the two supervision sets become increasingly disjoint. This trend suggests that the model effectively learns to separate $\mathcal { L } _ { b d }$ and ${ \mathcal { L } } _ { n b d }$ on distinct spatial regions.

## 12 Quality of Learned Slots: Object Dynamics Prediction

To validate the usefulness of our learned slots, we evaluate their transferability to a downstream task: object dynamics prediction. Following prior work, we employ the SlotFormer [47] framework as the dynamics prediction module on top of our frozen object-centric model to autoregressively predict future slots for F rollout steps, given B burn-in frames. Specifically, we set $( B , F )$ to (14, 10), (5, 10), and (10, 5) for the MOVi-C, MOVi-E, and YouTube-VIS datasets, respectively. The module is trained for 100,000 iterations with a batch size of 128. As shown in Tab. 18, SSync achieves consistent improvements across all datasets compared to previous methods. Notably, while prior models struggle with dynamics prediction on MOVi-E due to the prevalence of small objects, SSync excels in modeling fine-grained dynamics by capturing precise semantic boundaries. These results demonstrate that SSync not only enhances static object discovery but also produces robust representations that better capture object dynamics in realistic video settings.

Table 18: Results on object dynamics prediction. SlotFormer (SF) is employed to evaluate the learned slots from each method.

<table><tr><td rowspan="2">Method</td><td colspan="2">MOVi-C</td><td colspan="2">MOVi-E</td><td colspan="2">YouTube-VIS</td></tr><tr><td>FG-ARI↑</td><td>mBO↑</td><td>FG-ARI↑</td><td>mBO↑</td><td>FG-ARI↑</td><td>mBO↑</td></tr><tr><td>Reconstruction + SF</td><td>50.7</td><td>25.9</td><td>70.6</td><td>24.3</td><td>27.4</td><td>28.9</td></tr><tr><td>SlotContrast + SF</td><td>63.8</td><td>26.1</td><td>70.5</td><td>24.9</td><td>29.2</td><td>29.6</td></tr><tr><td>SRL + SF</td><td>68.9</td><td>27.4</td><td>70.4</td><td>24.9</td><td>32.2</td><td>30.0</td></tr><tr><td>SSync (Ours) + SF</td><td>69.1</td><td>29.0</td><td>72.1</td><td>27.1</td><td>32.1</td><td>30.9</td></tr></table>

Table 19: Performance comparison across varying warmup steps.

<table><tr><td>Ratio of Total Iterations</td><td>FG-ARI (↑)</td><td>mBO (↑)</td></tr><tr><td>5%</td><td>79.1</td><td>40.8</td></tr><tr><td>10%</td><td>78.6</td><td>38.3</td></tr><tr><td>15%</td><td>79.0</td><td>40.5</td></tr><tr><td>20%</td><td>78.6</td><td>40.2</td></tr><tr><td>30% (Default)</td><td>79.4</td><td>39.5</td></tr><tr><td>50%</td><td>79.1</td><td>39.8</td></tr></table>

## 13 Ablation on the SSync Warmup Ratio

In this section, we investigate the impact of the SSync activation schedule on the final decomposition performance. By default, SSync activates the selective alignment loss after the first 30% of training iterations. To validate the robustness against different warmup schedules, we evaluate the model’s performance by varying the warmup length from 5% to 50% of the total training steps. As summarized in Tab. 19, the results demonstrate that SSync is generally robust to the choice of warmup length, consistently maintaining competitive scores across different configurations.

## 14 Ablation on Activation Thresholding for Transitive Merging

In our default SSync framework, the transitive merging module determines whether a spatio-temporal patch is active by thresholding its attention value against the slot-wise mean activation. To further investigate the sensitivity and potential optimization of this activation criterion, we conduct an ablation study replacing the mean-based threshold with a quantile-based strategy. Specifically, instead of the mean-based threshold, we retain only the top k-th quantile (i.e., the top 10%, 20%, and 30%) of the highest activation values per slot to define the active regions for constructing the overlap graph.

Table 20: Ablation on transitive merging thresholds. We replace the default slot-wise mean activation threshold in Eq. (7) using a quantile-based thresholding strategy.

<table><tr><td>Threshold Strategy</td><td>FG-ARI (↑)</td><td>mBO (↑)</td></tr><tr><td>Ours</td><td>79.4</td><td>39.5</td></tr><tr><td>Quantile Top 10%</td><td>79.6</td><td>37.8</td></tr><tr><td>Quantile Top 20%</td><td>79.8</td><td>39.6</td></tr><tr><td>Quantile Top 30%</td><td>80.2</td><td>40.4</td></tr></table>

As shown in Tab. 20, varying the quantile threshold slightly enhances the performance. The results indicate that tuning the quantile hyperparameter may yield the best overall performance. However, we point out that employing a fixed quantile introduces a dataset-dependent hyperparameter; since the optimal quantile is inherently tied to the average scale and scale variance of objects within a specific dataset, a fixed threshold may not generalize well to other domains where object sizes significantly differ. In contrast, the slot-wise mean activation dynamically adapts to the spatial extent of each object on the fly, providing a parameter-free and highly generalizable criterion. Therefore, we adopt the mean activation as our robust default to ensure stable performance across diverse environments without the burden of dataset-specific tuning, while noting that quantile thresholding remains a viable option for domain-specific performance maximization.

## 15 Ablation on Patch Selection Variants

In this section, we compare our boundary and interior selection mechanism with alternative approaches, as detailed in Tab. 21. To isolate the benefits of the selection process, these comparisons are conducted without applying the transitive merging method.

As discussed in Sec. 3.2, our selection mechanism essentially operates as a relaxed morphological erosion, but with two key enhancements. Since standard erosion is susceptible to noise and ignores temporal dynamics, we incorporate a noise filtering term (Eq. 5) and extend the

spatial comparison to the spatiotemporal domain, enabling the detection of motion edges for temporal coherence. Consequently, while standard erosion yields reasonable performance (validating our core philosophy), our enhanced selection method achieves more significant gains.

Table 21: Variants of boundary and interior selection mechanisms.

<table><tr><td>Selection</td><td>FG-ARI↑</td><td>mBO↑</td></tr><tr><td>Ours</td><td>77.1</td><td>38.0</td></tr><tr><td> $\mathbf{X}$ </td><td>69.0</td><td>30.6</td></tr><tr><td>Erosion</td><td>76.4</td><td>36.9</td></tr><tr><td>Entropy (Avg)</td><td>72.1</td><td>37.6</td></tr><tr><td>Entropy (low 5%)</td><td>69.5</td><td>35.6</td></tr><tr><td>Entropy (low 10%)</td><td>71.9</td><td>34.0</td></tr><tr><td>Entropy (low 20%)</td><td>71.0</td><td>35.9</td></tr><tr><td>Entropy (low 30%)</td><td>69.4</td><td>37.8</td></tr><tr><td>Entropy (low 50%)</td><td>69.5</td><td>37.6</td></tr></table>

Additionally, we evaluate an entropy-based variant that utilizes prediction entropy as a reliability measure. In this setup, we treat low-entropy patches as reliable and selectively use them for distillation. As demonstrated, even when varying the entropy threshold from the average value to different percentiles, our straightforward design consistently outperforms these entropy-based alternatives.

## 16 Additional Ablation Studies on SSync Design

Finally, we compare our proposed SSync with alternative designs in Tab. 22. First, we implement two soft variants of SSync: one that replaces hard pseudo-labels with soft supervision targets based on the model’s internal confidence (Eq. (13)- (14)), and another that further relaxes the region selection by computing disagreement scores using the raw probability distributions of neighboring patches (Eq. (3)-(4)). Next, we evaluate a reverse SSync, which extracts interior regions from the encoder attention map and boundary regions from the decoder object map for synergistic learning. As shown, our default formulation outperforms all alternatives. This shows that

Table 22: Ablation on SSync design choices. We evaluate the impact of replacing hard pseudolabels with soft supervision (Soft Targets/Selection) and reversing the supervision targets between the encoder and decoder (Reversed SSync).

<table><tr><td>Selection</td><td>FG-ARI↑</td><td>mBO↑</td></tr><tr><td>Ours</td><td>77.1</td><td>38.0</td></tr><tr><td>Baseline</td><td>69.0</td><td>30.6</td></tr><tr><td>Soft Targets</td><td>72.9</td><td>34.0</td></tr><tr><td>+ Soft Selection</td><td>72.6</td><td>35.9</td></tr><tr><td>Reversed SSync</td><td>67.6</td><td>37.6</td></tr></table>

using soft distributions as targets inherently retains uncertainty, which can propagate structural ambiguity between modules, while the inferior performance of the reverse strategy explicitly validates our claim that the encoder excels at capturing sharp boundaries, whereas the decoder is better suited for maintaining consistent interiors.

## 17 Additional Qualitative Results.

We provide further qualitative comparisons on MOVi-C and the real-world YouTube-VIS 2021 benchmark, illustrating three representative sequences per dataset (Fig. 6 and Fig. 7). On MOVi-C (Fig. 6), SSync produces more temporally coherent decompositions with tighter object extents than both SlotContrast and SRL. In the first sequence (Fig. 6a), SSync assigns a semantically meaningful and stable slot to the background structure while preserving compact foreground masks, whereas SRL often misses background details and SlotContrast fragments both foreground and background regions. In the second sequence (Fig. 6b), SSync remains robust to repetitive background textures that frequently induce spurious slot activations in prior methods, avoiding the failure mode where textured patterns are mistakenly segmented as additional objects. In the third sequence (Fig. 6c), SSync yields object-level masks that are both less overfragmented and sharper at boundaries, consistent with our design that leverages encoder cues for boundary refinement and decoder coherence for interior denoising.

We observe the same trends on YouTube-VIS 2021 (Fig. 7), where real-world videos exhibit stronger appearance variation, non-rigid motion, and occlusions. In the first sequence (Fig. 7a), SlotContrast frequently splits a single person across multiple slots in early frames, and both SlotContrast and SRL struggle to separate two visually similar chairs. In contrast, SSync maintains stable person instances and consistently distinguishes the two separate chairs throughout the clip. In the second sequence (Fig. 7b), SlotContrast fails to recover the person in the lower part of the frame, while SRL detects the person but produces a less semantically consistent grouping of the tennis-court surface; SSync preserves the person and partitions the scene with clearer, more semantically aligned boundaries. In the final sequence (Fig. 7c), both SRL and SlotContrast struggle to segment the stepping stones. Furthermore, SRL tends to split a largely uniform background into multiple slots, and SlotContrast continues to over-fragment moving entities, whereas SSync keeps the background compact and foreground instances stable over time. Overall, SSync more effectively distinguishes unannotated secondary objects and background elements in the visualized scenes. These qualitative improvements align with our strong FG-ARI and mBO performances, while large-scale quantitative validation of these specific effects in real-world videos is left for future work.

## 18 Failure Case Analysis (Limitation).

In Fig. 8, we analyze representative failure cases on MOVi-C, MOVi-E, and YouTube-VIS to clarify the current limitations of SSync and to motivate future directions. Overall, we observe two failure modes: (i) early-frame identity underfragmentation and (ii) part-level over-fragmentation for large objects with strong intra-object variation.

First, on MOVi-C and MOVi-E datasets, we observe that under-fragmentation may occur in early frame predictions since motion cues in earlier frames may be insufficient to distinguish objects entering from similar spatial locations or overlapped, semantically similar objects. For instance, in Fig. 8a, two small objects (e.g., red and yellow) emerge from a similar direction and remain covered by one slot until their motion trajectories diverge; only after they separate spatially does the model consistently allocate distinct identities. Similarly, in Fig. 8b, a green object placed in front of a bag with a similar green pattern is not immediately recognized as a separate entity, but becomes distinguishable in later frames once relative motion and spatial separation increase. These cases suggest that incorporating bidirectional or offline temporal modeling (e.g., using future frames during refinement) could further mitigate early-frame errors.

On YouTube-VIS, a common failure mode arises for large-scale objects whose different parts exhibit markedly different visual characteristics. A representative example is a cargo truck (Fig. 8c), where the driver’s cabin and the container have substantially different textures. Despite transitive pseudo-label merging, SSync can still assign these semantically related parts to different slots, indicating that spatio-temporal overlap alone may be insufficient when intra-object appearance variance is high. We anticipate that addressing this limitation likely requires stronger part-to-whole grouping priors.

![](images/c9c16a9280f4d1bdce9c37fb6a233d250f9a7db98200113d8324ea8a5a0e3a9b.jpg)  
Fig. 6: Qualitative results on the MOVi-C dataset.

![](images/daa1a18ea58a59c0939c9e0e361cefc40f48dcb9cbf6b52b13867230d5b63770.jpg)

<details>
<summary>text_image</summary>

Video
GT
Ours
SRL
ShotContrast
</details>

(a) Video 1  
![](images/235e23d7e4cefdc3bd5c42f1378fd685662248b61f17cd7b2d2d275dbdb4f717.jpg)

<details>
<summary>text_image</summary>

Video
GT
Ours
SRL
SlotContrast
</details>

(b) Video 2

![](images/bfbf1c64637c879f51ca9a7b5f954f5bc5284f348408a0752e9a7bb149f8abdc.jpg)

<details>
<summary>text_image</summary>

Video
GT
Ours
SRL
SlotContrast
</details>

(c) Video 3  
Fig. 7: Qualitative results on the YouTube-VIS dataset.

![](images/5b070c4d2a1b45a214d812b38e6b5ee2dc5b0879e5360fb13393621751a7b737.jpg)  
Fig. 8: Failure Mode Analysis.

## References

1. Yanbo Wang, Letao Liu, and Justin Dauwels. Slot-vae: Object-centric scene generation with slot attention. In International Conference on Machine Learning, pages 36020–36035. PMLR, 2023.  
2. Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, and Animesh Garg. Slotdiffusion: Object-centric generative modeling with diffusion models. Advances in Neural Information Processing Systems, 36:50932–50958, 2023.  
3. Jindong Jiang, Fei Deng, Gautam Singh, and Sungjin Ahn. Object-centric slot diffusion. arXiv preprint arXiv:2303.10834, 2023.  
4. Jiaqi Xu, Cuiling Lan, Wenxuan Xie, Xuejin Chen, and Yan Lu. Slot-vlm: Objectevent slots for video-language modeling. Advances in Neural Information Processing Systems, 37:632–659, 2024.  
5. Yerim Jeon, Miso Lee, WonJun Moon, and Jae-Pil Heo. Masking matters: Unlocking the spatial reasoning capabilities of llms for 3d scene-language understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 38668–38677, 2026.  
6. Tatiana Zemskova and Dmitry Yudin. 3dgraphllm: Combining semantic graphs and large language models for 3d scene understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8885–8895, 2025.  
7. Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. Advances in Neural Information Processing Systems, 37:113991–114017, 2024.  
8. Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. In The Twelfth International Conference on Learning Representations, 2024.  
9. Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Objectcentric learning with slot attention. Advances in neural information processing systems, 33:11525–11538, 2020.  
10. Andrii Zadaianchuk, Maximilian Seitzer, and Georg Martius. Object-centric learning for real-world videos by predicting temporal feature similarities. Advances in Neural Information Processing Systems, 36:61514–61545, 2023.  
11. Gautam Singh, Yi-Fu Wu, and Sungjin Ahn. Simple unsupervised object-centric learning for complex and naturalistic videos. Advances in Neural Information Processing Systems, 35:18181–18196, 2022.  
12. Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy, and Klaus Greff. Conditional Object-Centric Learning from Video. In International Conference on Learning Representations (ICLR), 2022.  
13. Gamaleldin Elsayed, Aravindh Mahendran, Sjoerd Van Steenkiste, Klaus Greff, Michael C Mozer, and Thomas Kipf. Savi++: Towards end-to-end object-centric learning from real-world videos. Advances in Neural Information Processing Systems, 35:28940–28954, 2022.  
14. Hyun Seok Seong, WonJun Moon, and Jae-Pil Heo. From vicious to virtuous cycles: Synergistic representation learning for unsupervised video object-centric learning. In The Fourteenth International Conference on Learning Representations, 2026.  
15. WonJun Moon, Hyun Seok Seong, and Jae-Pil Heo. Reconstruction-guided slot curriculum: Addressing object over-fragmentation in video object-centric learning.  
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.  
16. Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. Featured Certification.  
17. Maximilian Seitzer, Max Horn, Andrii Zadaianchuk, Dominik Zietlow, Tianjun Xiao, Carl-Johann Simon-Gabriel, Tong He, Zheng Zhang, Bernhard Schölkopf, Thomas Brox, and Francesco Locatello. Bridging the gap to real-world object-centric learning. In The Eleventh International Conference on Learning Representations, 2023.  
18. Hongjia Liu, Rongzhen Zhao, Haohan Chen, and Joni Pajarinen. Metaslot: Break through the fixed number of slots in object-centric learning. Advances in neural information processing systems, 2025.  
19. Rongzhen Zhao, Vivienne Huiling Wang, Juho Kannala, and Joni Pajarinen. Multiscale fusion for object representation. In The Thirteenth International Conference on Learning Representations, 2025.  
20. Rongzhen Zhao, Jian Li, Juho Kannala, and Joni Pajarinen. Predicting video slot attention queries from random slot-feature pairs. arXiv preprint arXiv:2508.01345, 2025.  
21. Anna Manasyan, Maximilian Seitzer, Filip Radovic, Georg Martius, and Andrii Zadaianchuk. Temporally consistent object-centric learning by contrasting slots. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5401–5411, 2025.  
22. Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems, 33:596–608, 2020.  
23. Gilhan Park, WonJun Moon, SuBeen Lee, Tae-Young Kim, and Jae-Pil Heo. Mitigating background shift in class-incremental semantic segmentation. In European Conference on Computer Vision, pages 71–88. Springer, 2024.  
24. Tianheng Cheng, Xinggang Wang, Shaoyu Chen, Qian Zhang, and Wenyu Liu. Boxteacher: Exploring high-quality pseudo labels for weakly supervised instance segmentation. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 3145–3154, 2023.  
25. Dong-Hyun Lee et al. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In Workshop on challenges in representation learning, ICML, volume 3, page 896. Atlanta, 2013.  
26. Junnan Li, Richard Socher, and Steven CH Hoi. Dividemix: Learning with noisy labels as semi-supervised learning. arXiv preprint arXiv:2002.07394, 2020.  
27. Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and Takahiro Shinozaki. Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling. Advances in neural information processing systems, 34:18408–18419, 2021.  
28. Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10687–10698, 2020.  
29. Hieu Pham, Zihang Dai, Qizhe Xie, and Quoc V Le. Meta pseudo labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11557–11568, 2021.  
30. Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673, 2020.  
31. Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR, 2020.  
32. Robert M Haralick, Stanley R Sternberg, and Xinhua Zhuang. Image analysis using mathematical morphology. IEEE transactions on pattern analysis and machine intelligence, (4):532–550, 1987.  
33. Ke Fan, Zechen Bai, Tianjun Xiao, Tong He, Max Horn, Yanwei Fu, Francesco Locatello, and Zheng Zhang. Adaptive slot attention: Object discovery with dynamic slot number. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23062–23071, 2024.  
34. Aritra Ghosh, Himanshu Kumar, and P Shanti Sastry. Robust loss functions under label noise for deep neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017.  
35. Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. Advances in neural information processing systems, 31, 2018.  
36. Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3749–3761, 2022.  
37. Tao Wang, Ning Xu, Kean Chen, and Weiyao Lin. End-to-end video instance segmentation via spatial-temporal graph neural networks. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10797–10806, 2021.  
38. Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5188–5197, 2019.  
39. Linjie Yang, Yuchen Fan, Yang Fu, and Ning Xu. The 3rd large-scale video object segmentation challenge-video instance segmentation track. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.(CVPR) Workshop, 2021.  
40. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.  
41. William M Rand. Objective criteria for the evaluation of clustering methods. Journal of the American Statistical association, 66(336):846–850, 1971.  
42. Klaus Greff, Raphaël Lopez Kaufman, Rishabh Kabra, Nick Watters, Christopher Burgess, Daniel Zoran, Loic Matthey, Matthew Botvinick, and Alexander Lerchner. Multi-object representation learning with iterative variational inference. In International conference on machine learning, pages 2424–2433. PMLR, 2019.  
43. Görkay Aydemir, Weidi Xie, and Fatma Guney. Self-supervised object-centric learning for videos. Advances in Neural Information Processing Systems, 36:32879– 32899, 2023.  
44. Songtao Liu, Di Huang, and Yunhong Wang. Adaptive nms: Refining pedestrian detection in a crowd. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6459–6468, 2019.  
45. Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu, et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, volume 96, pages 226–231, 1996.  
46. Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732, 2016.  
47. Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, and Animesh Garg. Slotformer: Unsupervised visual dynamics simulation with object-centric models. In The Eleventh International Conference on Learning Representations, 2023.