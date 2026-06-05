# Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models

Shuwen Yu 1 Zhanxuan Hu 1 Yi Zhao 1 Yonghang Tai 1 Huafeng Li 2

# Abstract

Foundation models have driven rapid progress in computer vision, yet the two dominant paradigms, vision-language foundation models (VLMs) and vision-only foundation models (VFMs), remain only partially compatible. VLMs offer languagegrounded semantic alignment but are often visually coarse, while VFMs learn discriminative perceptual geometry but lack semantic grounding. We propose GPUA, a Geometry-Preserving Unsupervised Alignment framework that integrates the complementary strengths of VFMs and VLMs. Inspired by cross-lingual alignment, GPUA treats VFM features as a visual language and learns an orthogonal mapping that translates the VFM space into the VLM semantic space, preserving geometry and narrowing the modality gap without labels and model parameter updates. GPUA is task-agnostic and requires only feature-level access to pretrained models. Experiments across diverse benchmarks demonstrate improved cross-model compatibility and strong gains in downstream zero-shot recognition and segmentation with negligible overhead. Our code is available at: https://github.com/ Yuteam14/GPUA.

# 1. Introduction

Foundation models have become the cornerstone of modern computer vision, where two paradigms dominate: vision– language foundation models (VLMs) and vision-only foundation models (VFMs). VLMs, exemplified by CLIP (Radford et al., 2021), provide a powerful language-grounded semantic interface that enables open-vocabulary recognition

1Yunnan Normal University, Kunming, China 2Kunming University of Science and Technology, Kunming, China. Correspondence to: Zhanxuan Hu <zhanxuanhu@gmail.com>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

and strong cross-domain transfer. However, their visual representations are often semantically aligned yet perceptually coarse, making them less sensitive to fine-grained structures and local details. In contrast, VFMs such as DINO-style self-supervised models (Oquab et al., 2023; Simeoni et al. ´ , 2025) learn highly discriminative visual representations with strong locality and structural awareness, but lack explicit semantic grounding and struggle to support open-vocabulary reasoning. These complementary strengths and limitations naturally raise an important question:

Can we integrate heterogeneous foundation models to obtain representations that are both semantically grounded and perceptually discriminative?

Recent efforts on multi-foundation model fusion seek to exploit the complementary strengths of heterogeneous pretrained models, particularly in open-vocabulary semantic segmentation (Dong et al., 2023). A common paradigm uses CLIP as the source of open-vocabulary semantics, augments it with DINO-style VFMs to enhance discriminative patch-level cues (Wysoczanska et al. ´ , 2024; Barsellotti et al., 2025), or further leverages SAM-style promptable segmenters for high-quality mask generation (Yang & Gong, 2024; Sun et al., 2024; Zhang et al., 2025). Through patchlevel prediction and cross-model consistency, such pipelines partially compensate for CLIP’s limitations in localization and fine-grained boundary delineation, achieving strong performance on segmentation benchmarks.

Despite their success, existing fusion pipelines often suffer from two fundamental limitations. First, they typically assume non-trivial access to foundation models (e.g., intermediate feature extraction or dense mask queries), which may be infeasible for closed-source models, API-based services, or restricted deployment scenarios. Second, their designs are highly task- and structure-specific: fusion mechanisms are tightly coupled with pixel-level prediction, mask generation, and spatial post-processing, and therefore do not readily extend to more general image-level tasks such as zero-shot classification, where the output is a global semantic decision rather than dense correspondence. These limitations highlight the need for a more principled, taskagnostic mechanism that makes heterogeneous foundation models directly compatible at the representation level.

![](images/de56c4339ce141af31fe642a136d8640bbdaab33c5cbef5d16d23b62c712e1e8.jpg)

<details>
<summary>scatter</summary>

| dim1 | dim2 | Category        |
|------|------|-----------------|
| -60  | 60   | Visual Feature  |
| -40  | 40   | Visual Feature  |
| -20  | 20   | Visual Feature  |
| 0    | 0    | Visual Feature  |
| 20   | -20  | Visual Feature  |
| 40   | -40  | Visual Feature  |
| 60   | -60  | Visual Feature  |
| -60  | 60   | Text Prototype   |
| -40  | 40   | Text Prototype   |
| -20  | 20   | Text Prototype   |
| 0    | 0    | Text Prototype   |
| 20   | -20  | Text Prototype   |
| 40   | -40  | Text Prototype   |
| 60   | -60  | Text Prototype   |
</details>

![](images/040614e3f9e2c3e850abe1c25e7c9f39c7d28b50fbbbc4a983d1641c975b8a3c.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| -60  | 10   |
| -40  | 30   |
| -20  | 50   |
| 0    | 70   |
| 20   | 40   |
| 40   | 20   |
| 60   | 0    |
| 40   | -20  |
| 20   | -40  |
| 0    | -60  |
| -20  | -40  |
| -40  | -20  |
| -60  | 10   |
</details>

![](images/11c80cf403c8c1cf964e0cffe1a7adc4bd37e9149283fd38400e9e9e7865420b.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| -80  | 0    |
| -60  | 20   |
| -40  | 40   |
| -20  | 60   |
| 0    | 40   |
| 20   | 20   |
| 40   | 0    |
| 60   | -20  |
| 80   | -40  |
</details>

Figure 1. t-SNE visualization of foundation-model representations on the Pets. (a) The original CLIP space exhibits a pronounced modality gap between image-text embeddings. (b) VFM features yield more compact intra-class clusters, yet lack globally consistent alignment to semantic concepts. (c) GPUA (Ours) projects visual clusters onto their corresponding semantic anchors (⋆) while preserving intra-class structure, demonstrating effective geometry-preserving alignment and recovering accurate instance-to-prototype correspondences.

In this work, we introduce GPUA, a fundamentally different framework for unsupervised vision–language alignment. Inspired by unsupervised cross-lingual alignment (Lample et al., 2018; Ouali et al., 2023), we view visual representations as a distinct visual language and recast vision– language compatibility as a cross-modal translation problem: aligning the visual vocabulary induced by an image encoder with the semantic vocabulary defined in a language-aligned space. Concretely, GPUA learns an orthogonal transformation that translates VFM representations into the VLM semantic space. While several prior works also attempt to bridge VFMs and VLMs by learning feature-space transformations (Ouali et al., 2023; Barsellotti et al., 2025; Jose et al., 2025), they typically rely on task-specific supervision and end-to-end training. In contrast, GPUA performs fully unsupervised alignment and learns the mapping without updating any pretrained model parameters. As illustrated in Fig. 1, the orthogonality constraint preserves the intrinsic geometry of VFM features, leading to stable alignment and effectively narrowing the modality gap.

Importantly, GPUA does not require a perfectly calibrated initial vision–language embedding space. Instead, it treats the language-aligned semantic space provided by a VLM as a fixed reference and learns a lightweight translation from any given visual space into this reference. This design makes GPUA naturally extensible: it can go beyond aligning a single VFM to a VLM by incorporating multiple visual spaces induced by heterogeneous encoders, mapping them into a shared semantic coordinate system, and performing unified inference through simple fusion. By translating and aggregating complementary perceptual geometries in the same semantic space, GPUA injects fine-grained visual discrimination into language-grounded recognition, effectively building a practical bridge between VLMs and VFMs.

Contributions. Our main contributions are three-fold: (1) We advocate unsupervised vision–language alignment as a principled and practical route to improve compatibility between VLMs and VFMs, and demonstrate that aligning heterogeneous foundation models is a promising direction for open-vocabulary vision. (2) We propose GPUA, a simple yet effective alignment framework that learns an orthogonal transformation to map VFM features into the semantic space of VLMs, without requiring labels or model parameter updates. (3) GPUA achieves consistent improvements across downstream benchmarks with negligible additional computation, offering a favorable accuracy–efficiency tradeoff and plug-and-play deployment for zero-shot recognition and segmentation.

# 2. Related Work

# 2.1. Foundation Models

Recent years have witnessed the rise of foundation models trained on large-scale data, which provide general-purpose representations transferable across a wide range of tasks. In computer vision, two paradigms have become particularly influential: vision–language foundation models (VLMs) and vision-only foundation models (VFMs).

Vision–Language Foundation Models. VLMs such as CLIP (Radford et al., 2021) learn aligned visual and textual representations through contrastive pretraining on massive image–text corpora. A key advantage of this paradigm lies in the language-grounded semantic space, which enables training-free inference and zero-shot recognition by matching images against text prototypes. This capability has inspired research on vision–language alignment enhancement (Huang et al., 2025b), and has also been successfully extended to various downstream domains, including remote sensing (Wang et al., 2024b; Liu et al., 2024a) and medical imaging (Lu et al., 2024). Despite their strong semantic alignment, recent studies have reported that VLM visual representations are often perceptually coarse and insufficiently sensitive to fine-grained details, particularly under distribution shift or in inductive settings.

Vision-only Foundation Models. In contrast, VFMs such as DINO (Oquab et al., 2023; Simeoni et al. ´ , 2025) are trained via large-scale self-supervised learning and excel at capturing intrinsic visual structures, local correspondences, and instance-level discrimination. Although VFMs exhibit strong perceptual discrimination and robust geometric structure, they still suffer from inherent limitations. Most notably, VFMs lack explicit semantic grounding and therefore cannot support open-vocabulary reasoning or text-driven inference.

# 2.2. Fusion of Foundation Models

Motivated by the complementary strengths of VFMs and VLMs, a growing body of work explores combining multiple pretrained models for improved open-vocabulary perception. Representative pipelines typically adopt CLIP as a source of language-grounded semantics, enhance it with DINO-style VFMs for fine-grained visual discrimination, and further integrate SAM-like promptable segmenters for mask-level reasoning. Such approaches have demonstrated strong empirical performance in tasks such as open-vocabulary semantic segmentation (Wysoczanska ´ et al., 2024; Barsellotti et al., 2025; Hu et al., 2026) and vision-language grounding (Liu et al., 2024b).

# 2.3. Cross-lingual Alignment

Cross-lingual alignment is a fundamental problem in representation learning, aiming to establish compatibility between embedding spaces learned from different languages. Extensive studies in natural language processing have shown that independently learned linguistic embedding spaces can be effectively aligned by exploiting shared structural regularities, commonly instantiated through orthogonal mappings that preserve intrinsic geometric relations under the isomorphism hypothesis (Lample et al., 2018; Artetxe et al., 2018; Jawanpuria et al., 2019). These results suggest that large-scale representations typically exhibit similar geometric structures, supporting reliable alignment across domains. Inspired by this line of work, similar alignment principles have been extended beyond language to other modalities, including the alignment between visual representations and semantic concepts (Ouali et al., 2023; Schrodi et al., 2024). However, most existing approaches in the vision domain rely on labeled data, task-specific supervision, or extensive fine-tuning, which limits their applicability. Following this principle, we further extend it to unsupervised alignment on pretrained representations.

# 3. Method

# 3.1. Preliminaries

Our goal is to integrate the complementary strengths of Vision–Language Foundation Models (VLMs) and Visiononly Foundation Models (VFMs) in a fully unsupervised manner without requiring any optimization of model parameters. Specifically, we seek a lightweight mechanism that enables VFM representations, rich in perceptual geometry yet semantically ungrounded, to become directly compatible with the language-aligned semantic space of VLMs. We observe that this problem is conceptually analogous to cross-lingual alignment (Mikolov et al., 2013).

Cross-lingual alignment. In cross-lingual NLP, embeddings learned from different languages encode similar semantics but reside in heterogeneous vector spaces. Alignment is commonly achieved by learning a structurepreserving mapping that translates representations from one language into another, typically under an orthogonality constraint to maintain geometric consistency. Formally, let $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ and $\mathbf { Y } \in \mathbb { R } ^ { N \times d }$ denote embeddings learned from two different languages. Cross-lingual alignment seeks a mapping $\mathbf { W } \in \mathbb { R } ^ { d \times \bar { d } }$ by solving:

$$
\min _ {\mathbf {W}} \| \mathbf {X} \mathbf {W} - \mathbf {Y} \| _ {F} ^ {2}, \quad \text { s.t. } \mathbf {W} ^ {\top} \mathbf {W} = \mathbf {I}, \tag {1}
$$

which admits a closed-form Procrustes solution when wordlevel correspondences are available.

Unsupervised cross-lingual alignment. A key assumption in Eq. (1) is the availability of reliable cross-lingual correspondences. In practice, however, such correspondences are often unavailable. To address this issue, unsupervised cross-lingual alignment methods (Grave et al., 2019) introduce a latent correspondence matrix P and jointly estimate P and W:

$$
\min _ {\mathbf {P}, \mathbf {W}} \| \mathbf {X} \mathbf {W} - \mathbf {P} \mathbf {Y} \| _ {F} ^ {2}, \quad \text { s.t. } \mathbf {W} ^ {\top} \mathbf {W} = \mathbf {I}, \tag {2}
$$

where $\mathbf { P } \in \{ 0 , 1 \} ^ { N \times N }$ encodes hard alignments between the two embedding sets and each row of P contains exactly one non-zero entry, i.e.,

$$
\sum_ {j = 1} ^ {M} P _ {i j} = 1, \quad \forall i \in \{1, \dots , N \}. \tag {3}
$$

Such formulations enable alignment without supervision by alternating between correspondence estimation and geometric mapping.

Notably, recent work (e.g., LFA (Ouali et al., 2023)) has successfully adopted Eq. (2) to mitigate the modality gap within VLMs, demonstrating strong empirical gains. However, directly transferring these unsupervised cross-lingual techniques to VFM–VLM integration is often suboptimal. First, existing formulations typically rely on alternating optimization over the mapping W and the correspondence matrix P, making the solution highly sensitive to initialization and prone to poor local optima. Second, the correspondence estimation stage often ignores the intrinsic structure of the data, which may lead to unstable or semantically inconsistent assignments under domain shift. To address these limitations, we propose a new unsupervised alignment framework, termed Geometry-Preserving Unsupervised Alignment (GPUA).

![](images/e32711e732bcb158d3610cb079fa3e0ef49c4613ecad5d980050a867ea833e99.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["a photo of cat"] --> B["CLIP"]
    C["Visual Encoder"] --> D["Visual Space"]
    C --> E["Visual Logits"]
    C --> F["Semantic Logits"]
    D --> G["Soft Assignment"]
    E --> H["Soft Labels(P)"]
    F --> I["Correspondence Mining"]
    J["VFM"] --> K["Unsupervised Correspondence Mining (UCM)"]
    L["Textual Encoder"] --> M["Visual Center"]
    L --> N["Text Prototype"]
    L --> O["Weighted Sum"]
    L --> P["Linear Transformation"]
    Q["Orthogonal Transformation"] --> R["New Feature Space"]
    S["THS Loss"] --> T["Push"]
    S --> U["Pull"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style J fill:#ccf,stroke:#333
    style L fill:#ccf,stroke:#333
    style Q fill:#cfc,stroke:#333
    style R fill:#fcc,stroke:#333
    style S fill:#fcc,stroke:#333
    style T fill:#fcc,stroke:#333
    style U fill:#fcc,stroke:#333
    style V fill:#cff,stroke:#333
```
</details>

Figure 2. The pipeline of GPUA. Stage 1: An unsupervised correspondence estimation module infers soft assignments (P) by jointly enforcing structural consistency and semantic alignment between visual features and semantic prototypes. Stage 2: These correspondences are used to derive the optimal orthogonal transformation (W), which is further refined via the THS loss to yield a hubness-robust embedding space.

# 3.2. Geometry-Preserving Unsupervised Alignment

Overview. Following the cross-lingual alignment perspective, we view VFM-VLM integration as translating VFM embeddings into the language-aligned semantic space of a frozen VLM. Concretely, given VFM visual features $\mathbf { Z \in }$ $\mathbb { R } ^ { N \times d _ { \tau } }$ and VLM text prototypes $\mathbf { Y } \in \mathbb { R } ^ { K \times d _ { t } }$ , our goal is to obtain: (1) a correspondence matrix $\mathbf { P } \in \mathbb { R } ^ { N \times K }$ that captures (soft) instance-to-prototype associations, thereby bridging the two spaces in the absence of labels; and (2) a geometry-preserving mapping $\mathbf { W } \in \mathbb { R } ^ { d _ { v } \times d _ { t } }$ that translates VFM features into the VLM semantic space while maintaining the intrinsic geometry of VFM embeddings, enforced via an orthogonality constraint $\mathbf { W } ^ { \top } \mathbf { W } = \mathbf { I } .$ .

Unlike prior unsupervised alignment approaches that jointly estimate P and W through alternating optimization, we adopt a simple two-stage strategy (Fig. 2). First, Unsupervised Correspondence Mining (UCM) infers reliable correspondences P by explicitly leveraging both the semantic structure provided by the VLM space and the geometric structure inherent to VFM features. Then, Geometry-Preserving Alignment (GPA) computes an orthogonal mapping W based on the mined correspondences, yielding a closed-form and stable translation from the VFM space to the VLM semantic space. This decoupled design substantially reduces sensitivity to initialization and leads to a simple and stable alignment pipeline.

# 3.3. Unsupervised Correspondence Mining (UCM)

We begin by revisiting correspondence mining in a frozen VLM semantic space. Given VLM image features X ∈ $\mathbb { R } ^ { N \times d _ { t } }$ and text prototypes $\mathbf { Y } \in \mathbb { R } ^ { K \times d _ { t } }$ , assigning each image to a prototype can be written as the following leastsquares matching problem:

$$
\min _ {\mathbf {P}} \| \mathbf {X} - \mathbf {P Y} \| _ {F} ^ {2}, \quad \text {s.t.} \mathbf {P} \in \{0, 1 \} ^ {N \times K}, \mathbf {P 1} = \mathbf {1}, (4)
$$

where P is a hard assignment matrix whose i-th row contains exactly one non-zero entry, indicating the matched prototype for the i-th sample.

Connection to K-means. Interestingly, Eq. (4) is tightly connected to the standard matrix formulation of K-means. Let X denote data points, Y denote cluster centroids, and

P be the one-hot assignment matrix with $\mathbf { P 1 } = \mathbf { 1 }$ . The K-means objective can be written as:

$$
\min _ {\mathbf {P}, \mathbf {Y}} \| \mathbf {X} - \mathbf {P Y} \| _ {F} ^ {2}, \quad \text { s.t. } \mathbf {P} \in \{0, 1 \} ^ {N \times K}, \mathbf {P 1} = \mathbf {1}. \tag {5}
$$

Comparing Eq. (4) with Eq. (5), we observe that correspondence mining in the VLM space corresponds exactly to the assignment step of K-means when the “centroids” are fixed to be the text prototypes Y (see Appendix A). In other words, optimizing P in Eq. (4) amounts to updating cluster labels given centroids, the same operation performed in the K-means assignment step.

Formulation of UCM. The above K-means interpretation also clarifies a key limitation of VLM-only correspondence mining. Optimizing minP $\| \mathbf { X } - \mathbf { P Y } \| _ { F } ^ { 2 }$ updates assignments solely by matching samples to fixed text prototypes, and thus relies heavily on semantic scores in the VLM space. However, it does not explicitly enforce that samples assigned to the same prototype form compact and geometrically coherent groups under the underlying data distribution. This issue becomes more pronounced under domain shift, where VLM similarities may be noisy and lead to unstable correspondences.

Motivated by this observation, we ask a simple question: Can we inject the geometry-rich structure captured by VFMs into correspondence mining? To this end, we construct a Kmeans-style structural model in the VFM space. Specifically, given VFM features $\mathbf { Z } \in \mathbb { R } ^ { N \times d _ { v } }$ , we introduce learnable VFM centroids $\mathbf { C } \in \mathbb { R } ^ { K \times d _ { v } }$ and encourage each sample to be close to its assigned centroid. Crucially, we couple this structural view with the semantic view from the VLM by sharing the same assignment matrix P across both spaces. This naturally leads to the following unified objective:

$$
\min _ {\mathbf {P}, \mathbf {C}} (1 - \lambda) \| \mathbf {Z} - \mathbf {P C} \| _ {F} ^ {2} + \lambda \| \mathbf {X} - \mathbf {P Y} \| _ {F} ^ {2}, \tag {6}
$$

$\mathrm { ~ s . t . ~ } \ \mathbf { P } \in \{ 0 , 1 \} ^ { N \times K } , \quad \mathbf { P 1 } = \mathbf { 1 } .$

where the first term promotes geometric coherence in the VFM space, while the second term enforces semantic consistency with VLM text prototypes, and $\lambda \in [ 0 , 1 ]$ is trade-off parameter. By optimizing a shared P, UCM produces correspondences that are simultaneously language-grounded and geometry-aware.

In practice, however, restricting P to the set of permutation (one-hot) matrices yields a combinatorial optimization problem that is intractable in practice. To obtain an efficient and stable solver, we relax P to a scaled assignment matrix and add an entropic regularizer. Specifically, we allow P to take non-negative real values and enforce its row/column marginals:

$$
\mathbf {P} \in \Pi (\mathbf {r}, \mathbf {c}) = \left\{\mathbf {P} \in \mathbb {R} _ {+} ^ {N \times K} \mid \mathbf {P 1} = \mathbf {r}, \mathbf {P} ^ {\top} \mathbf {1} = \mathbf {c} \right\}, \tag {7}
$$

where $\mathbf { r } \in \mathbb { R } _ { + } ^ { N }$ and $\mathbf { c } \in \mathbb { R } _ { + } ^ { K }$ specify the desired row- and column-sums. A common choice is uniform marginals, i.e., $\begin{array} { r } { \mathbf { r } \ = \ \frac { 1 } { N } \mathbf { 1 } } \end{array}$ and $\begin{array} { r } { { \bf c } = \frac { 1 } { K } { \bf 1 } } \end{array}$ , which amounts to a scaled (approximately doubly-stochastic) correspondence matrix. With this relaxation, we solve the entropically regularized problem:

$$
\min _ {\mathbf {P} \in \Pi (\mathbf {r}, \mathbf {c}), \mathbf {C}} (1 - \lambda) \| \mathbf {Z} - \mathbf {P C} \| _ {F} ^ {2} + \lambda \| \mathbf {X} - \mathbf {P Y} \| _ {F} ^ {2} - \varepsilon \mathcal {H} (\mathbf {P}), \tag {8}
$$

where $\varepsilon > 0$ controls the strength of regularization and

$$
\mathcal {H} (\mathbf {P}) = - \sum_ {i = 1} ^ {N} \sum_ {k = 1} ^ {K} P _ {i k} \log P _ {i k} \tag {9}
$$

is the entropy of P. The entropic term promotes smooth assignments and enables efficient optimization via Sinkhornstyle matrix scaling.

Optimization of UCM. Since the proposed objective depends on two sets of variables, namely the assignment matrix P and the VFM centroids C, directly optimizing them jointly is non-trivial. We therefore adopt an alternating optimization strategy, where one variable block is updated while fixing the other, leading to a progressive reduction of the overall objective.

Updating P. With fixed centroids C, optimizing Eq. (6) with respect to P reduces to a linear objective over the relaxed assignment space. By expanding the Frobenius norms and discarding terms independent of P, the subproblem can be written as:

$$
\max _ {\mathbf {P} \in \Pi (\mathbf {r}, \mathbf {c})} \left\langle \mathbf {P}, (1 - \lambda) \mathbf {Z} \mathbf {C} ^ {\top} + \lambda \mathbf {X} \mathbf {Y} ^ {\top} \right\rangle + \varepsilon H (\mathbf {P}), \tag {10}
$$

Eq. (10) corresponds to an entropy-regularized optimal transport problem, which can be efficiently solved using the Sinkhorn–Knopp algorithm (Cuturi, 2013). The resulting correspondence matrix yields a geometry-aware soft assignment that jointly reflects structural and semantic affinities.

Updating C. Given the updated assignment matrix P, the objective for the VFM centroids becomes a structural least-squares problem:

$$
\min _ {\mathbf {C}} \| \mathbf {Z} - \mathbf {P C} \| _ {F} ^ {2}. \tag {11}
$$

This formulation admits a closed-form solution where each latent centroid $\mathbf { C } _ { k }$ is re-estimated as the weighted barycenter of the visual features:

$$
\mathbf {C} _ {k} = \frac {\sum_ {i = 1} ^ {N} P _ {i k} \mathbf {Z} _ {i}}{\sum_ {i = 1} ^ {N} P _ {i k}}. \tag {12}
$$

By iteratively performing these updates, UCM uncovers a coherent latent correspondence that faithfully aligns the geometric structure of visual features with the semantic topology induced by VLMs.

Algorithm 1 GPUA Optimization Algorithm   
1: Input: visual features Z, VLM visual features X, semantic prototypes Y, trade-off parameter $\lambda$ , iterations T
2: Initialization:
3: $\mathbf{P}^{(0)} \leftarrow \text{Softmax}(\mathbf{X}\mathbf{Y}^{\top}); \mathbf{C}^{(0)} \leftarrow \text{update}(\mathbf{Z}, \mathbf{P}^{(0)})$ 4: Stage 1: Unsupervised Correspondence Mining
5: for t = 0 to T - 1 do
6: $\mathbf{R}^{(t)} \leftarrow (1 - \lambda) \mathbf{Z}\mathbf{C}^{(t)\top} + \lambda \mathbf{X}\mathbf{Y}^{\top}$ 7: $\mathbf{P}^{(t+1)} \leftarrow \text{Sinkhorn}(\mathbf{R}^{(t)})$ 8: $\mathbf{C}^{(t+1)} \leftarrow \text{update}(\mathbf{Z}, \mathbf{P}^{(t+1)})$ 9: end for
10: # Pseudo-labels via argmax on $\mathbf{P}^{(T)}$ 11: Stage 2: Geometry-Preserving Alignment
12: $W \leftarrow UV^{\top}$ via SVD( $Z^{\top}P^{(T)}Y$ )
13: # Update W by Eq.(15)
14: $W \leftarrow W - \eta\nabla_{W}L_{THS}$ 15: Return refined mapping $W^{*}$

# 3.4. Geometry-Preserving Alignment (GPA)

Given the correspondence matrix P from Stage 1, GPA learns an orthogonal mapping W to translate visual features into the VLM semantic space, so that aligned features match the prototype mixture $\mathbf { P Y } \colon$ :

$$
\min _ {\mathbf {W}} \| \mathbf {Z} \mathbf {W} - \mathbf {P} \mathbf {Y} \| _ {F} ^ {2}, \quad \text { s.t. } \mathbf {W} ^ {\top} \mathbf {W} = \mathbf {I}. \tag {13}
$$

The orthogonality constraint enforces an (approximately) isometric translation, preventing degenerate scaling/shearing and preserving neighborhood geometry that is critical for stable nearest-prototype inference. In practice, Eq. (13) is an orthogonal Procrustes problem with a closed-form solution:

$$
\mathbf {W} _ {0} = \mathbf {U V} ^ {\top}, \quad \mathbf {U} \boldsymbol {\Sigma} \mathbf {V} ^ {\top} = \operatorname{SVD} \left(\mathbf {Z} ^ {\top} \mathbf {P Y}\right). \tag {14}
$$

Although $\mathbf { W } _ { 0 }$ enables alignment without requiring model parameter updates, the aligned space may still exhibit hubness (Lample et al., 2018), where a few prototypes become nearest neighbors of many samples and distort the local neighborhoods. To suppress hubs, we refine $\mathbf { W } _ { 0 }$ with a topology-aware ranking loss:

$$
\mathcal {L} _ {\mathrm{THS}} = \frac {1}{N K} \sum_ {i = 1} ^ {N} \sum_ {c \in \mathcal {N} _ {i} ^ {K}} \left[ d _ {i} ^ {+} + m _ {i, c} ^ {\text { base }} + h _ {c} - d _ {i, c} \right] _ {+}, \tag {15}
$$

where $d _ { i } ^ { + } \ = \ \| \mathbf { W } ^ { \top } \mathbf { z } _ { i } - \mathbf { y } _ { \ell _ { i } } \| _ { 2 } , \ d _ { i , c } \ = \ \| \mathbf { W } ^ { \top } \mathbf { z } _ { i } - \mathbf { y } _ { c } \| _ { 2 }$ , $\mathbf { y } _ { \ell _ { i } }$ denotes the semantic prototype corresponding to the pseudo-label $\ell _ { i }$ of sample i, and $\dot { \mathcal { N } } _ { i } ^ { K }$ denotes the K nearest competing prototypes. We use a semantic margin $m _ { i , c } ^ { \mathrm { b a s e } } =$ $( 1 - \mathbf { y } _ { \boldsymbol { \ell } _ { i } } ^ { \top } \mathbf { y } _ { c } ) / s$ and a hubness penalty

$$
h _ {c} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {I} \big (c \in \mathcal {N} _ {i} ^ {K} \big), \tag {16}
$$

which up-weights margins for overly-central prototypes, discouraging them from becoming hubs. Starting from W0, we apply a few gradient steps on ${ \mathcal { L } } _ { \mathrm { T H S } }$ to obtain the refined mapping W∗. The overall optimization procedure of GPUA is summarized in Algorithm 1.

# 4. Experiments

# 4.1. Experimental Settings

Zero-Shot Classification. We evaluate our method primarily on the zero-shot image classification task, which directly reflects the quality of cross-model alignment through image–text matching without task-specific training. Following the standard evaluation protocol of CLIP (Radford et al., 2021), we conduct experiments across a diverse set of public benchmarks covering different visual domains and levels of granularity. Detailed dataset descriptions and statistics are provided in Appendix C.

Open-Vocabulary Segmentation. We further assess the generality of our alignment framework on the openvocabulary semantic segmentation task, which involves heterogeneous foundation models and therefore serves as a challenging testbed for cross-model compatibility. Experiments are conducted on multiple representative benchmarks, and performance is evaluated using the standard mean Intersection-over-Union (mIoU) metric. The detailed dataset information and experimental configurations are reported in Appendix C.

# 4.2. Implementation details

General Setup. All experiments are conducted in a fully unsupervised setting. We only require feature extraction from frozen foundation models, without updating any model parameters or introducing task-specific fine-tuning. As the visual foundation model (VFM), we adopt DI-NOv3 (Simeoni et al. ´ , 2025) due to its strong capability in capturing fine-grained visual structures and patch-level geometry. GPUA operates purely at the feature level and learns a lightweight alignment transformation on top of these frozen representations.

Zero-Shot Classification Setting. For zero-shot classification, GPUA learns a single feature transformation using the training split of each dataset, while keeping both the visual and textual encoders frozen. In this setting, we align global image representations with textual semantics, and therefore use the CLS token of the visual encoder as the image-level feature. During inference, the learned transformation is directly applied to image features, which are then matched with fixed textual prototypes encoded by the text encoder via cosine similarity. No test-time adaptation, distribution estimation, or prompt optimization is performed.

Table 1. Quantitative comparison of zero-shot classification performance. GPUA (Ours) leverages the full training set, while GPUA\* represents the performance trained with only 16 samples per class. Best results are highlighted in bold. 

<table><tr><td>Method</td><td>Flowers</td><td>Pets</td><td>Caltech</td><td>FGVC</td><td>EuroSAT</td><td>UCF101</td><td>DTD</td><td>Food</td><td>Cars</td><td>SUN</td><td>ImageNet</td><td>Avg.</td></tr><tr><td>CLIP (Radford et al., 2021)</td><td>70.7</td><td>89.1</td><td>93.2</td><td>24.7</td><td>48.3</td><td>67.5</td><td>43.5</td><td>85.9</td><td>65.6</td><td>62.5</td><td>66.6</td><td>65.2</td></tr><tr><td>ZERO (Farina et al., 2024)</td><td>67.2</td><td>87.8</td><td>94.4</td><td>25.2</td><td>42.2</td><td>69.2</td><td>45.9</td><td>86.8</td><td>69.0</td><td>67.6</td><td>71.2</td><td>66.0</td></tr><tr><td>MTA (Zanella &amp; Ben Ayed, 2024)</td><td>68.1</td><td>88.2</td><td>94.2</td><td>25.2</td><td>45.4</td><td>68.7</td><td>45.9</td><td>85.0</td><td>68.5</td><td>66.7</td><td>70.1</td><td>66.0</td></tr><tr><td>TDA (Karmanov et al., 2024)</td><td>71.4</td><td>88.6</td><td>94.2</td><td>23.9</td><td>58.0</td><td>70.7</td><td>47.4</td><td>86.1</td><td>67.3</td><td>67.6</td><td>69.5</td><td>67.7</td></tr><tr><td>ZLaP (Kalantidis et al., 2024)</td><td>73.5</td><td>87.1</td><td>93.1</td><td>25.4</td><td>55.6</td><td>71.5</td><td>48.6</td><td>86.9</td><td>65.6</td><td>67.4</td><td>70.0</td><td>67.7</td></tr><tr><td>DPE (Zhang et al., 2024a)</td><td>75.1</td><td>91.1</td><td>94.8</td><td>29.0</td><td>55.8</td><td>70.4</td><td>54.2</td><td>86.2</td><td>67.3</td><td>70.1</td><td>71.9</td><td>69.6</td></tr><tr><td>DMN (Zhang et al., 2024b)</td><td>74.5</td><td>92.0</td><td>95.4</td><td>30.0</td><td>59.4</td><td>72.5</td><td>55.8</td><td>85.1</td><td>68.0</td><td>70.2</td><td>72.2</td><td>70.5</td></tr><tr><td>StatA (Zanella et al., 2025)</td><td>75.2</td><td>92.4</td><td>94.2</td><td>24.7</td><td>67.3</td><td>73.5</td><td>48.4</td><td>87.1</td><td>68.0</td><td>68.7</td><td>69.9</td><td>69.9</td></tr><tr><td>TIPPLE (Lu et al., 2025)</td><td>71.3</td><td>90.2</td><td>93.9</td><td>25.4</td><td>51.8</td><td>71.2</td><td>49.2</td><td>86.0</td><td>67.8</td><td>68.1</td><td>71.0</td><td>67.8</td></tr><tr><td>COSMIC (Huang et al., 2025a)</td><td>82.1</td><td>94.2</td><td>96.8</td><td>31.4</td><td>58.8</td><td>76.2</td><td>58.2</td><td>86.6</td><td>71.3</td><td>72.3</td><td>78.2</td><td>73.3</td></tr><tr><td>GPUA* (Ours)</td><td>86.6</td><td>94.5</td><td>98.1</td><td>34.7</td><td>80.3</td><td>78.4</td><td>56.7</td><td>87.9</td><td>77.4</td><td>72.6</td><td>74.3</td><td>76.5</td></tr><tr><td>Δ</td><td>+16.8</td><td>+4.3</td><td>+4.8</td><td>+5.7</td><td>+30.4</td><td>+11.9</td><td>+10.8</td><td>+0.3</td><td>+10.8</td><td>+10.4</td><td>+9.9</td><td>+10.6</td></tr><tr><td>GPUA (Ours)</td><td>83.8</td><td>95.0</td><td>95.3</td><td>33.8</td><td>88.2</td><td>80.4</td><td>58.5</td><td>89.5</td><td>77.7</td><td>74.2</td><td>75.4</td><td>77.4</td></tr><tr><td>Δ</td><td>+14.0</td><td>+6.0</td><td>+3.8</td><td>+5.5</td><td>+34.9</td><td>+13.2</td><td>+14.7</td><td>+3.0</td><td>+11.7</td><td>+11.7</td><td>+10.5</td><td>+11.8</td></tr></table>

Open-Vocabulary Semantic Segmentation Setting. For open-vocabulary semantic segmentation, we evaluate GPUA on top of multiple representative segmentation frameworks to assess its general applicability. In contrast to imagelevel recognition, dense prediction requires fine-grained spatial representations. Accordingly, we perform alignment at the patch level and leverage DINOv3 patch features to enhance visual–semantic correspondence. For the vision–language models (VLMs), we consider three representative CLIP-based segmentation frameworks, namely MaskCLIP (Dong et al., 2023), SCLIP (Wang et al., 2024a), and SC-CLIP (Bai et al., 2025). These methods share a common design principle: they exploit the attention mechanisms of CLIP to improve the quality of patch-level visual features for dense prediction. This makes them a natural testbed for evaluating whether geometry-preserving alignment with a strong VFM can further enhance patch-level visual–semantic consistency.

Within each framework, GPUA is integrated as a plug-in alignment module by aligning DINOv3 patch-level features to the corresponding VLM semantic space. Importantly, GPUA does not modify the segmentation head, loss functions, or task-specific training objectives. This design isolates the effect of the proposed alignment strategy, ensuring that the observed performance gains stem from improved visual–semantic correspondence rather than architectural or optimization changes. Additional implementation details are provided in the Appendix.

# 4.3. Main Results

Zero-shot Classification. We evaluate GPUA on zeroshot image classification and compare it with representative CLIP-based alignment methods across 11 benchmarks spanning diverse domains and category granularities (Table 1). Although some compared methods (e.g., the test-time adaptation approaches COSMIC (Huang et al., 2025a) and TIP-PLE (Lu et al., 2025)) rely on inference-time adaptation, we include them as strong baselines under the same evaluation protocol to comprehensively assess the effectiveness of GPUA across different alignment paradigms. Overall, GPUA achieves the best performance on most datasets and improves the average accuracy by a clear margin, with notably larger gains on datasets with strong domain shift (e.g., EuroSAT) and fine-grained categories (e.g., FGVC, Cars). These results indicate that geometry-preserving alignment with a strong VFM enhances cross-model compatibility and yields more reliable visual–semantic matching. A key advantage of GPUA is that it learns a single orthogonal mapping offline from frozen model features and then applies it as a fixed translation during inference. This further suggests that the observed gains mainly stem from a higher-quality aligned embedding space itself. Finally, we report a lowdata variant, GPUA\*, where the alignment is learned from only a small set of unlabeled samples per class. Despite the limited data, GPUA\* remains competitive and consistently outperforms prior unsupervised alignment baselines, highlighting the robustness and sample efficiency of the proposed framework.

Table 2. Generalizability of GPUA on zero-shot semantic segmentation benchmarks. 

<table><tr><td>Method</td><td>ADE</td><td>V20</td><td>C59</td></tr><tr><td>CLIP (Radford et al., 2021)</td><td>3.1</td><td>49.1</td><td>11.1</td></tr><tr><td>MaskCLIP (Dong et al., 2023)</td><td>11.9</td><td>54.2</td><td>22.2</td></tr><tr><td>+ GPUA (DINOv3)</td><td>15.9</td><td>65.7</td><td>27.9</td></tr><tr><td>SCLIP (Wang et al., 2024a)</td><td>16.1</td><td>81.5</td><td>34.2</td></tr><tr><td>+ GPUA (DINOv3)</td><td>19.1</td><td>87.8</td><td>36.3</td></tr><tr><td>SC-CLIP (Bai et al., 2025)</td><td>20.1</td><td>84.3</td><td>40.1</td></tr><tr><td>+ GPUA (DINOv3)</td><td>21.3</td><td>87.6</td><td>41.0</td></tr><tr><td>ProxyCLIP (Lan et al., 2024)</td><td>19.7</td><td>83.0</td><td>37.2</td></tr><tr><td>Talk2DINO (Barsellotti et al., 2025)</td><td>21.1</td><td>87.1</td><td>39.8</td></tr><tr><td>LPOSS+ (Stojnić et al., 2025)</td><td>22.7</td><td>82.5</td><td>39.3</td></tr></table>

![](images/12e1c2c3328f815d85931959b2eab8a416e9f4602d4402799510bae55e3385c0.jpg)  
(a) SC-CLIP

![](images/4ce336499f5b2c0667120c7ccd1878883dc89f10b8a2e88400fcecff61319a01.jpg)  
(b) GPUA

![](images/5b6ecd34eac4bba0b7ffe735b717a64969eb270a891c729b9f1b5ad08c59e058.jpg)  
(c) Ground Truth   
Figure 3. Qualitative comparison on open-vocabulary semantic segmentation. (a) Predictions produced by SC-CLIP; (b) Predictions after incorporating GPUA; (c) Ground-truth segmentation masks. By aligning geometry-aware DINOv3 patch features with the VLM semantic space, GPUA enhances patch-level visual–semantic correspondence without modifying the segmentation architecture.

Open-vocabulary Semantic Segmentation. We further evaluate GPUA on open-vocabulary semantic segmentation to assess its effectiveness for dense prediction. As reported in Table 2, GPUA yields consistent improvements across multiple benchmarks and different CLIP-based segmentation frameworks.This is achieved without introducing any segmentation-specific architectural modifications: GPUA is used solely as a lightweight plug-in alignment module that leverages geometry-aware VFM (DINOv3) patch features to improve patch-level visual–semantic correspondence.

Importantly, GPUA does not alter the segmentation heads, loss functions, or training/inference protocols of the underlying frameworks, nor does it rely on additional task-specific modules or complex feature interaction designs adopted in methods such as (Barsellotti et al., 2025) . Therefore, the performance gains can be attributed to better aligned patch representations rather than additional task-specific engineering.Despite its simplicity, GPUA attains performance comparable to or exceeding methods specifically designed for open-vocabulary segmentation based on VFM representations.Qualitative comparisons are provided in Figure 3.

# 4.4. Ablation Study

We conduct an ablation study to analyze the effects of different alignment strategies in our zero-shot classification framework, with results summarized in Table 3.

Role of VFM Choice. We first consider a variant that removes VFM optimization, where correspondence mining is performed purely in the CLIP space by aligning CLIP visual features to CLIP text prototypes. Although this CLIPonly alignment already leads to a modest improvement, the overall gain remains limited (e.g., 89.2% on Pets, 68.3% on UCF101, and 71.6% on ImageNet). A key observation is that performance improves substantially when CLIP and DINO features are concatenated. This improvement is not merely due to feature aggregation, but rather because the combined representation exhibits higher feature quality, which in turn enables more accurate estimation of the correspondence matrix P. More reliable correspondences directly benefit the subsequent geometry-preserving alignment, leading to a higher-quality mapping W.

Table 3. Ablation study of zero-shot classification variants. ✓ indicates the activation of a component. Only-S denotes mining correspondences using semantic priors (from CLIP) only, while DINO and CLIP denote the use of their visual features. 

<table><tr><td>Only-S</td><td>CLIP</td><td>DINO</td><td>Pets</td><td>UCF101</td><td>ImageNet</td></tr><tr><td>√</td><td>√</td><td>√</td><td>89.2</td><td>68.3</td><td>71.6</td></tr><tr><td></td><td>√</td><td></td><td>91.2</td><td>76.1</td><td>69.9</td></tr><tr><td></td><td></td><td>√</td><td>93.1</td><td>75.3</td><td>70.6</td></tr><tr><td></td><td>√</td><td>√</td><td>94.5</td><td>78.4</td><td>74.3</td></tr></table>

Table 4. Ablation study of different loss functions on representative benchmarks. 

<table><tr><td>Method</td><td>EuroSAT</td><td>DTD</td><td>UCF101</td><td>SUN</td><td>Food</td></tr><tr><td>CSLS</td><td>72.4</td><td>49.6</td><td>69.2</td><td>67.3</td><td>79.5</td></tr><tr><td>Contrastive</td><td>75.5</td><td>55.9</td><td>76.3</td><td>71.4</td><td>87.4</td></tr><tr><td>Adaptive</td><td>73.2</td><td>55.4</td><td>75.3</td><td>72.2</td><td>87.2</td></tr><tr><td>Triplet</td><td>75.3</td><td>55.4</td><td>76.2</td><td>72.5</td><td>87.5</td></tr><tr><td>THS (Ours)</td><td>80.3</td><td>56.7</td><td>78.4</td><td>72.6</td><td>87.9</td></tr></table>

This result also highlights an important property of GPUA: it naturally supports the fusion of multiple heterogeneous visual foundation models.

Effect of Loss Design. We evaluate the impact of various alignment objectives in Table 4. While discriminative losses like Contrastive and Triplet improve performance over the CSLS baseline, our THS loss consistently yields the best results across all five representative datasets. Specifically, THS achieves a significant margin over the second-best Triplet loss on challenging tasks such as EuroSAT (+5.0%) and UCF101 (+2.2%). Notably, THS also outperforms the hubness-aware baseline Adaptive (Ouali et al., 2023), surpassing it by 7.1% on EuroSAT and 1.3% on DTD. These results empirically validate that explicitly suppressing hub nodes leads to a more balanced and discriminative embedding space, confirming the effectiveness of our design in robust cross-modal alignment.

# 5. Conclusion

In this paper, we propose GPUA, a flexible framework for unsupervised vision–language alignment that bridges vision-only and vision–language foundation models through geometry-preserving feature translation. By leveraging frozen foundation models and learning a lightweight orthogonal mapping at the feature level, GPUA effectively improves cross-model compatibility and delivers consistent gains on zero-shot classification and open-vocabulary semantic segmentation without test-time adaptation or taskspecific tuning.

Despite its effectiveness, GPUA has several limitations. In particular, the correspondence estimation and alignment process does not explicitly account for data imbalance across categories, which may lead to suboptimal correspondences when class distributions are highly skewed. Addressing data imbalance and incorporating adaptive weighting or uncertainty-aware correspondence modeling constitute promising directions for future work.

# Acknowledgements

This work is supported by the Basic Research Project of Yunnan Province (Grant No. 202501CF070004), Xingdian Talent Support Program, and Intelligent Computing Center, Yunnan Normal University.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Artetxe, M., Labaka, G., and Agirre, E. A robust selflearning method for fully unsupervised cross-lingual mappings of word embeddings. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 789–798, 2018.   
Bai, S., Liu, Y., Han, Y., Zhang, H., Tang, Y., Zhou, J., and Lu, J. Self-calibrated clip for training-free openvocabulary segmentation. IEEE Transactions on Image Processing, 2025.   
Barsellotti, L., Bianchi, L., Messina, N., Carrara, F., Cornia, M., Baraldi, L., Falchi, F., and Cucchiara, R. Talking to dino: Bridging self-supervised vision backbones with language for open-vocabulary segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22025–22035, 2025.

Bossard, L., Guillaumin, M., and Van Gool, L. Food-101– mining discriminative components with random forests. In European conference on computer vision, pp. 446–461. Springer, 2014.

Caesar, H., Uijlings, J., and Ferrari, V. Coco-stuff: Thing and stuff classes in context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1209–1218, 2018.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3606–3613, 2014.

Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., and Schiele, B. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3213– 3223, 2016.

Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in Neural Information Processing Systems, 26, 2013.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 248–255. Ieee, 2009.

Dong, X., Bao, J., Zheng, Y., Zhang, T., Chen, D., Yang, H., Zeng, M., Zhang, W., Yuan, L., Chen, D., et al. Maskclip: Masked self-distillation advances contrastive languageimage pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10995–11005, 2023.

Everingham, M., Van Gool, L., Williams, C. K., Winn, J., and Zisserman, A. The pascal visual object classes challenge 2012 (voc2012) results (2012), 2011.

Farina, M., Franchi, G., Iacca, G., Mancini, M., and Ricci, E. Frustratingly easy test-time adaptation of vision-language models. Advances in Neural Information Processing Systems, 37:129062–129093, 2024.

Fei-Fei, L., Fergus, R., and Perona, P. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop, pp. 178–178. IEEE, 2004.

Grave, E., Joulin, A., and Berthet, Q. Unsupervised alignment of embeddings with wasserstein procrustes. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1880–1890. PMLR, 2019.

Helber, P., Bischke, B., Dengel, A., and Borth, D. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.   
Hu, Z., Xu, Q., Duan, Y., Tai, Y., and Li, H. Sota: Selfadaptive optimal transport for zero-shot classification with multiple foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.   
Huang, F., Jiang, J., Jiang, Q., Li, H., Khan, F. N., and Wang, Z. Cosmic: Clique-oriented semantic multi-space integration for robust clip test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9772–9781, 2025a.   
Huang, S., Zhang, H., and Li, X. Enhance vision-language alignment with noise. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 17449– 17457, 2025b.   
Jawanpuria, P., Balgovind, A., Kunchukuttan, A., and Mishra, B. Learning multilingual word embeddings in latent metric space: a geometric approach. Transactions of the Association for Computational Linguistics, 7:107– 120, 2019.   
Jose, C., Moutakanni, T., Kang, D., Baldassarre, F., Darcet, T., Xu, H., Li, D., Szafraniec, M., Ramamonjisoa, M., Oquab, M., et al. Dinov2 meets text: A unified framework for image-and pixel-level vision-language alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24905–24916, 2025.   
Kalantidis, Y., Tolias, G., et al. Label propagation for zeroshot classification with vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23209–23218, 2024.   
Karmanov, A., Guan, D., Lu, S., El Saddik, A., and Xing, E. Efficient test-time adaptation of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14162–14171, 2024.   
Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 554–561, 2013.   
Lample, G., Conneau, A., Ranzato, M., Denoyer, L., and Jegou, H. Word translation without parallel data. In ´ International Conference on Learning Representations, 2018.

Lan, M., Chen, C., Ke, Y., Wang, X., Feng, L., and Zhang, W. Proxyclip: Proxy attention improves clip for openvocabulary segmentation. In European Conference on Computer Vision, pp. 70–88. Springer, 2024.   
Liu, F., Chen, D., Guan, Z., Zhou, X., Zhu, J., Ye, Q., Fu, L., and Zhou, J. Remoteclip: A vision language foundation model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 62:1–16, 2024a.   
Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pp. 38–55. Springer, 2024b.   
Lu, M. Y., Chen, B., Williamson, D. F., Chen, R. J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L. P., Gerber, G., et al. A visual-language foundation model for computational pathology. Nature medicine, 30(3):863–874, 2024.   
Lu, Z., Bai, J., Li, X., Xiao, Z., and Wang, X. Task-toinstance prompt learning for vision-language models at test time. IEEE Transactions on Image Processing, 2025.   
Maji, S., Rahtu, E., Kannala, J., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.   
Mikolov, T., Le, Q. V., and Sutskever, I. Exploiting similarities among languages for machine translation. arXiv preprint arXiv:1309.4168, 2013.   
Mottaghi, R., Chen, X., Liu, X., Cho, N.-G., Lee, S.-W., Fidler, S., Urtasun, R., and Yuille, A. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 891–898, 2014.   
Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pp. 722–729. IEEE, 2008.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.   
Ouali, Y., Bulat, A., Matinez, B., and Tzimiropoulos, G. Black box few-shot adaptation for vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15534–15546, 2023.

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. Cats and dogs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3498–3505. IEEE, 2012.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, pp. 8748–8763. PmLR, 2021.   
Schrodi, S., Hoffmann, D. T., Argus, M., Fischer, V., and Brox, T. Two effects, one trigger: On the modality gap, object bias, and information imbalance in contrastive vision-language representation learning. arXiv preprint arXiv:2404.07983, 2024.   
Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, ´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.   
Soomro, K., Zamir, A. R., and Shah, M. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.   
Stojnic, V., Kalantidis, Y., Matas, J., and Tolias, G. Lposs: ´ Label propagation over patches and pixels for openvocabulary semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9794–9803, 2025.   
Sun, S., Li, R., Torr, P., Gu, X., and Li, S. Clip as rnn: Segment countless visual concepts without training endeavor. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13171–13182, 2024.   
Wang, F., Mei, J., and Yuille, A. Sclip: Rethinking selfattention for dense vision-language inference. In European Conference on Computer Vision, pp. 315–332. Springer, 2024a.   
Wang, Z., Prabha, R., Huang, T., Wu, J., and Rajagopal, R. Skyscript: A large and semantically diverse visionlanguage dataset for remote sensing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 5805–5813, 2024b.   
Wysoczanska, M., Sim ´ eoni, O., Ramamonjisoa, M., Bursuc, ´ A., Trzcinski, T., and P ´ erez, P. Clip-dinoiser: Teaching ´ clip a few dino tricks for open-vocabulary semantic segmentation. In European Conference on Computer Vision, pp. 320–337. Springer, 2024.

Xiao, J., Hays, J., Ehinger, K. A., Oliva, A., and Torralba, A. Sun database: Large-scale scene recognition from abbey to zoo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3485–3492. IEEE, 2010.   
Yang, X. and Gong, X. Tuning-free universally-supervised semantic segmentation. IEEE Access, 12:187329– 187342, 2024.   
Zanella, M. and Ben Ayed, I. On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23783–23793, 2024.   
Zanella, M., Fuchs, C., De Vleeschouwer, C., and Ben Ayed, I. Realistic test-time adaptation of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 25103–25112, 2025.   
Zhang, C., Stepputtis, S., Sycara, K., and Xie, Y. Dual prototype evolving for test-time generalization of visionlanguage models. Advances in Neural Information Processing Systems, 37:32111–32136, 2024a.   
Zhang, D., Liu, F., and Tang, Q. Corrclip: Reconstructing patch correlations in clip for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24677–24687, 2025.   
Zhang, Y., Zhu, W., Tang, H., Ma, Z., Zhou, K., and Zhang, L. Dual memory networks: A versatile adaptation approach for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28718–28728, 2024b.   
Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., and Torralba, A. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127(3):302–321, 2019.

# A. Matrix form of K-means.

Consider a dataset $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { R } ^ { d }$ and K clusters. Stack data points row-wise into

$$
\mathbf {X} \triangleq \left[ \begin{array}{c} \mathbf {x} _ {1} ^ {\top} \\ \vdots \\ \mathbf {x} _ {N} ^ {\top} \end{array} \right] \in \mathbb {R} ^ {N \times d}.
$$

Let cluster centroids be $\{ \mathbf { c } _ { k } \} _ { k = 1 } ^ { K } \subset \mathbb { R } ^ { d }$ , stacked as

$$
\mathbf {C} \triangleq \left[ \begin{array}{c} \mathbf {c c} _ {1} ^ {\top} \\ \vdots \\ \mathbf {c c} _ {K} ^ {\top} \end{array} \right] \in \mathbb {R} ^ {K \times d}.
$$

Define the (hard) assignment matrix $\mathbf { P } \in \{ 0 , 1 \} ^ { N \times K }$ by

$$
P _ {i k} = \left\{ \begin{array}{l l} 1, & \text { if   } \mathbf {x} _ {i} \text {   is   assigned   to   cluster   } k, \\ 0, & \text { otherwise }, \end{array} \right. \quad \text { s.t. } \quad \mathbf {P 1} _ {K} = \mathbf {1} _ {N},
$$

i.e., each row of P is one-hot.

The standard K-means objective.

$$
\min _ {\{\mathbf {c} _ {k} \}, \{\mathcal {C} _ {k} \}} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {C} _ {k}} \| \mathbf {x} _ {i} - \mathbf {c} _ {k} \| _ {2} ^ {2} \tag {17}
$$

is equivalent to the following matrix formulation:

$$
\min _ {\mathbf {P} \in \{0, 1 \} ^ {N \times K}, \mathbf {C} \in \mathbb {R} ^ {K \times d}} \| \mathbf {X} - \mathbf {P C} \| _ {F} ^ {2} \quad \text { s.t. } \quad \mathbf {P 1} _ {K} = \mathbf {1} _ {N}. \tag {18}
$$

Proof. We show that the matrix formulation in (18) induces the same alternating updates as the classical K-means algorithm.

Update of P. Fixing the centroids C, the optimization over P becomes

$$
\min _ {\mathbf {P} \in \{0, 1 \} ^ {N \times K}, \mathbf {P 1} = \mathbf {1}} \| \mathbf {X} - \mathbf {P C} \| _ {F} ^ {2} = \sum_ {i = 1} ^ {N} \min _ {k \in \{1, \dots , K \}} \| \mathbf {x} _ {i} - \mathbf {c} _ {k} \| _ {2} ^ {2}.
$$

Hence, the optimal assignment is obtained by

$$
P _ {i k} = 1 \quad \Longleftrightarrow \quad k = \arg \min _ {j \in \{1, \dots , K \}} \| \mathbf {x} _ {i} - \mathbf {c} _ {j} \| _ {2} ^ {2},
$$

which exactly coincides with the assignment step of standard K-means.

Update of C. Fixing the assignments P, the optimization over C becomes

$$
\min _ {\mathbf {C}} \left\| \mathbf {X} - \mathbf {P C} \right\| _ {F} ^ {2}.
$$

This is a least-squares problem with the closed-form solution

$$
\mathbf {C} = (\mathbf {P} ^ {\top} \mathbf {P}) ^ {- 1} \mathbf {P} ^ {\top} \mathbf {X},
$$

or equivalently, for each cluster k,

$$
\mathbf {c} _ {k} = \frac {\sum_ {i = 1} ^ {N} P _ {i k} \mathbf {x} _ {i}}{\sum_ {i = 1} ^ {N} P _ {i k}},
$$

which is exactly the centroid update in classical K-means.

Therefore, alternating minimization of (18) with respect to P and C recovers the standard K-means algorithm.

# B. Further Analyses

Parameter Sensitivity Analysis. We study the effect of the fusion coefficient λ, which controls the trade-off between geometric consistency in the VFM space and semantic alignment in the VLM space. As shown in Figure 4, classification accuracy remains relatively stable when λ is in the range [0.6, 0.9], reaching the best performance around λ = 0.9. As λ increases, the model places greater emphasis on semantic alignment, which generally improves performance. However, when λ becomes too large (close to 1.0), the optimization degenerates into relying almost entirely on VLM-based semantic matching, effectively weakening or removing the geometric constraints provided by VFMs. In this case, the model loses the discriminative structural information encoded in VFM features, leading to a noticeable performance drop. Similarly, when λ is too small $( \mathrm { e . g . , < 0 . 4 ) }$ , the model overemphasizes geometric consistency while underutilizing semantic guidance, which also degrades performance. These results highlight the importance of jointly leveraging both geometric and semantic information rather than relying on either one alone. Based on these observations, we set λ = 0.9 as the default value for a balanced and effective alignment.

Convergence Analysis. We analyze the convergence of our model by tracking classification accuracy and optimization loss over training iterations, as shown in Figure 5. Most datasets converge quickly, with accuracy stabilizing after a few iterations and loss steadily decreasing. While structured datasets such as Caltech101 and Pets reach high accuracies early, more challenging datasets like Eurosat and DTD require slightly more iterations to stabilize. Overall, the results confirm that our optimization framework converges reliably across diverse domains.

# C. Additional Implementation Details

We evaluate our method on a diverse set of benchmark datasets covering both image-level recognition and dense prediction tasks, and provide implementation details for each setting below.

Benchmark Datasets. Our experiments span two tasks: zero-shot image classification and open-vocabulary semantic segmentation.

Zero-Shot Classification. For image-level recognition, we evaluate on 11 widely used benchmarks with diverse visual domains and category granularities: ImageNet (Deng et al., 2009), SUN397 (Xiao et al., 2010), FGVCAircraft (Maji et al., 2013), EuroSAT (Helber et al., 2019), StanfordCars (Krause et al., 2013), Food101 (Bossard et al., 2014), Oxford-IIIT Pets (Parkhi et al., 2012), Oxford Flowers102 (Nilsback & Zisserman, 2008), Caltech101 (Fei-Fei et al., 2004), DTD (Cimpoi et al., 2014), and UCF101 (Soomro et al., 2012).

Open-Vocabulary Segmentation. For dense prediction, we evaluate on three representative open-vocabulary semantic segmentation benchmarks without a background class: ADE20K (ADE) (Zhou et al., 2019), PASCAL VOC20 (Everingham et al., 2011), and PASCAL Context59 (Mottaghi et al., 2014).

Model Configuration. We adopt a frozen CLIP ViT-B/16 as the base vision–language model (VLM) and DINOv3 ViT-B/16 as the vision foundation model (VFM) to provide structural priors. All model parameters are learned on an auxiliary training set and remain fixed during inference, without access to test data or any test-time optimization.

Task-Specific Settings. Zero-shot classification images are center-cropped to 224 × 224 at inference. Class names are embedded using a single fixed prompt, e.g., “a photo of a [CLASS]”, and predictions are obtained by computing the similarity between image features and class text embeddings.

Open-vocabulary segmentation training sets are constructed by extracting patch-level features using the VLM. Each semantic category leverages the semantic prior of the VLM to select the most relevant patches: specifically, the similarity between each patch feature and the corresponding text embedding is computed, and the top 1,024 patches per category with the highest similarity scores are retained to form the final training set for alignment. During inference, evaluation protocols of the corresponding baseline methods are followed, including sliding-window or multi-scale strategies when applicable.

Experimental Details. During training, the orthogonal matrix W is obtained using a Sinkhorn-based optimal transport solver with entropy regularization ϵ = 0.01. The fusion weight is set to λ = 0.9 to balance visual and semantic distributions. Visual features from VFM and VLM are combined using a simple concatenation approach. All experiments are conducted on a single NVIDIA RTX 4090 GPU.

Table 5. Effect of different VFM and VLM combinations on zero-shot classification using GPUA. The upper block reports results under the 16-shot per-class training setting, while the lower block uses the full training set. 

<table><tr><td>Method</td><td>Flowers</td><td>Pets</td><td>Caltech</td><td>FGVC</td><td>EuroSAT</td><td>UCF101</td><td>DTD</td><td>Food</td><td>Cars</td><td>SUN</td><td>ImageNet</td><td>Avg.</td></tr><tr><td colspan="13">low-data setting (16 samples per class)</td></tr><tr><td>Only DINOv2</td><td>86.52</td><td>91.61</td><td>97.65</td><td>28.32</td><td>67.49</td><td>77.24</td><td>53.01</td><td>81.51</td><td>71.67</td><td>68.05</td><td>73.23</td><td>72.39</td></tr><tr><td>DINOv2 + CLIP</td><td>87.5</td><td>93.4</td><td>98.0</td><td>30.4</td><td>78.7</td><td>79.4</td><td>54.3</td><td>86.2</td><td>76.4</td><td>72.9</td><td>76.5</td><td>75.8</td></tr><tr><td>Only DINOv3</td><td>86.6</td><td>93.1</td><td>97.3</td><td>33.7</td><td>83.9</td><td>75.3</td><td>55.9</td><td>85.5</td><td>76.4</td><td>67.4</td><td>70.6</td><td>75.1</td></tr><tr><td>DINOv3 + CLIP</td><td>84.1</td><td>94.5</td><td>98.0</td><td>33.4</td><td>75.7</td><td>78.4</td><td>55.4</td><td>88.0</td><td>77.4</td><td>72.2</td><td>74.3</td><td>75.6</td></tr><tr><td colspan="13">Full training set</td></tr><tr><td>DINOv2 + CLIP</td><td>84.7</td><td>94.7</td><td>97.0</td><td>30.5</td><td>83.2</td><td>80.9</td><td>58.2</td><td>88.5</td><td>77.3</td><td>74.2</td><td>77.1</td><td>76.9</td></tr><tr><td>DINOv3 + CLIP</td><td>83.8</td><td>95.0</td><td>95.3</td><td>33.8</td><td>88.2</td><td>80.4</td><td>58.5</td><td>89.5</td><td>77.7</td><td>74.2</td><td>75.4</td><td>77.4</td></tr></table>

Table 6. Comparison of different alignment losses on zero-shot classification benchmarks. 

<table><tr><td>Method</td><td>Flowers</td><td>Pets</td><td>Caltech</td><td>Aircraft</td><td>EuroSAT</td><td>UCF101</td><td>DTD</td><td>Food</td><td>Cars</td><td>SUN</td><td>ImageNet</td><td>Avg.</td></tr><tr><td>CSLS</td><td>83.1</td><td>83.1</td><td>92.2</td><td>31.5</td><td>72.4</td><td>69.2</td><td>49.6</td><td>79.5</td><td>69.7</td><td>67.3</td><td>66.7</td><td>69.5</td></tr><tr><td>Adaptive</td><td>85.1</td><td>94.0</td><td>98.1</td><td>34.0</td><td>73.2</td><td>75.3</td><td>55.4</td><td>87.2</td><td>76.6</td><td>72.2</td><td>73.9</td><td>75.0</td></tr><tr><td>Contrastive</td><td>85.6</td><td>94.4</td><td>97.9</td><td>33.3</td><td>75.5</td><td>76.3</td><td>55.9</td><td>87.4</td><td>76.2</td><td>71.4</td><td>72.7</td><td>75.1</td></tr><tr><td>Triplet</td><td>85.3</td><td>93.7</td><td>98.0</td><td>33.7</td><td>75.3</td><td>76.2</td><td>55.4</td><td>87.5</td><td>77.0</td><td>72.5</td><td>74.1</td><td>75.3</td></tr><tr><td>THS</td><td>86.6</td><td>94.5</td><td>98.1</td><td>34.7</td><td>80.3</td><td>78.4</td><td>56.7</td><td>87.9</td><td>77.4</td><td>72.6</td><td>74.3</td><td>76.5</td></tr></table>

# D. Additional GPUA Results

In this section, we present additional experiments, as summarized in Tables 6 to 8 and Figures 4 to 5.

Effect of backbone architectures. The COSMIC baseline reported in Table 1 adopts a DINOv2-based visual backbone following its original implementation. To ensure a fair comparison, we further evaluate the impact of different visual foundation model (VFM) and vision–language model (VLM) combinations in Table 5. In particular, when using the same backbone architecture as COSMIC (e.g., DINOv2), GPUA still consistently outperforms the baseline across datasets and training settings. These results indicate that the performance improvements of GPUA mainly originate from the proposed alignment strategy rather than simply benefiting from stronger backbone architectures such as DINOv3. Moreover, GPUA demonstrates stable performance across different backbone combinations, suggesting that the proposed method is largely model-agnostic and can generalize effectively across heterogeneous representation spaces.

Generalizability across domains. We further evaluate GPUA on multiple ImageNet-related benchmarks, including ImageNet, ImageNet-A, ImageNet-V2, ImageNet-R, and ImageNet-S. As shown in Table 7, GPUA consistently improves performance across all benchmarks, increasing the average accuracy from 59.1 to 67.3 compared with CLIP. In particular, GPUA achieves notable gains on challenging distribution-shift benchmarks such as ImageNet-A and ImageNet-S, demonstrating strong cross-dataset generalization. The generalization capability of GPUA mainly benefits from the strong visual–semantic representations provided by large-scale pretrained foundation models. Building upon these representations, GPUA further enhances cross-modal feature consistency while preserving the original semantic structure, leading to more reliable visual–semantic matching across different data distributions.

Extension to Dense Prediction Tasks. We further evaluate GPUA on additional zero-shot semantic segmentation benchmarks to assess its generalizability on dense prediction tasks. As shown in Table 8, we extend the evaluation to COCO-Stuff164K (Caesar et al., 2018) and Cityscapes (Cordts et al., 2016), in addition to ADE20K, V20, and C59. GPUA consistently achieves the best performance across all datasets, outperforming both SC-CLIP and Talk2DINO. Notably, simply incorporating stronger visual foundation model (VFM) features does not necessarily guarantee improved segmentation performance. For example, the prior VFM-based method Talk2DINO performs comparably to or even worse than SC-CLIP on several datasets. In contrast, GPUA consistently improves performance over both methods across all benchmarks, including further gains on Stuff164K and Cityscapes. These results suggest that the improvements mainly originate from more effective cross-modal feature alignment rather than solely from stronger backbone representations.

Table 7. Out-of-Domain Generalization: Comparison between CLIP and GPUA on ImageNet Benchmarks. 

<table><tr><td>Method</td><td>ImageNet</td><td>ImageNet-A</td><td>ImageNet-V2</td><td>ImageNet-R</td><td>ImageNet-S</td><td>Average</td></tr><tr><td>CLIP</td><td>66.7</td><td>47.9</td><td>60.9</td><td>74.0</td><td>46.1</td><td>59.1</td></tr><tr><td>GPUA</td><td>76.5</td><td>57.4</td><td>68.2</td><td>77.5</td><td>56.9</td><td>67.3</td></tr></table>

Table 8. Extension of GPUA to Zero-Shot Semantic Segmentation Benchmarks. 

<table><tr><td>Method</td><td>ADE20K</td><td>V20</td><td>C59</td><td>Cityscapes</td><td>Stuff164K</td></tr><tr><td>SC-CLIP</td><td>20.1</td><td>84.3</td><td>40.1</td><td>41.0</td><td>26.9</td></tr><tr><td>GPUA</td><td>21.3</td><td>87.6</td><td>41.0</td><td>42.0</td><td>29.0</td></tr><tr><td>Talk2DINO</td><td>21.1</td><td>87.1</td><td>39.8</td><td>36.6</td><td>28.1</td></tr></table>

![](images/5ade9026d27f9dd7e4d8af114635fa64732f046e7c42bef16a545b4ff7d03813.jpg)

<details>
<summary>line</summary>

| Lambda (weight for VLM) | ImageUCF101 | FGVCAircraft | Food101 | OxfordFlowers | OxfordPets | Caltech101 | StanfordCars | EuroSAT | DescribableTextures | SUN397 | ImageNet |
| ------------------------ | ----------- | ------------ | ------- | ------------- | ---------- | ---------- | ------------ | ------- | -------------------- | ------ | -------- |
| 0.0                      | 75          | 27           | 82      | 82            | 93         | 98         | 65           | 70      | 48                   | 70     | 75       |
| 0.1                      | 75          | 27           | 83      | 83            | 93         | 98         | 65           | 70      | 48                   | 70     | 75       |
| 0.2                      | 75          | 27           | 83      | 83            | 93         | 98         | 66           | 70      | 48                   | 70     | 75       |
| 0.3                      | 75          | 27           | 84      | 83            | 93         | 98         | 67           | 70      | 48                   | 70     | 75       |
| 0.4                      | 75          | 27           | 84      | 83            | 93         | 98         | 69           | 70      | 48                   | 70     | 75       |
| 0.5                      | 75          | 27           | 85      | 83            | 93         | 98         | 70           | 70      | 50                   | 70     | 75       |
| 0.6                      | 75          | 27           | 85      | 83            | 93         | 98         | 71           | 72      | 51                   | 71     | 75       |
| 0.7                      | 75          | 27           | 86      | 83            | 93         | 98         | 73           | 73      | 52                   | 72     | 76       |
| 0.8                      | 76          | 28           | 86      | 84            | 93         | 98         | 74           | 74      | 53                   | 73     | 76       |
| 0.9                      | 76          | 32           | 86      | 86            | 93         | 98         | 75           | 78      | 54                   | 73     | 76       |
| 1.0                      | 76          | 31           | 86      | 86            | 93         | 96         | -            | -       | -                    | -      | -        |
</details>

Figure 4. Sensitivity analysis of the fusion coefficient λ. Classification accuracy (%) versus λ across 11 datasets, showing optimal performance around λ = 0.9.

![](images/19762062a9084c1210534cd594b56f672b1724a319ad788cdd41f0b4194a624d.jpg)

<details>
<summary>line</summary>

| Iteration | ImageUCF101 | FGVCAircraft | Food101 | OxfordFlowers | OxfordPets | Caltech101 | StanfordCars | EuroSAT | DescribableTextures | SUN397 | ImageNet |
| --------- | ----------- | ------------ | ------- | ------------- | ---------- | ---------- | ------------ | ------- | ------------------- | ------ | -------- |
| 1         | 0.00328     | 0.00305      | 0.00312 | 0.00289       | 0.00306    | 0.00316    | 0.00304      | 0.00312 | 0.0035              | 0.00329 | 0.00322  |
| 2         | 0.00325     | 0.00299      | 0.0031  | 0.00287       | 0.00305    | 0.00313    | 0.00303      | 0.0031  | 0.00348            | 0.00328 | 0.00322  |
| 3         | 0.00324     | 0.00298      | 0.0031  | 0.00287       | 0.00305    | 0.00313    | 0.00303      | 0.0031  | 0.00347            | 0.00328 | 0.00322  |
| 4         | 0.00324     | 0.00298      | 0.0031  | 0.00287       | 0.00305    | 0.00313    | 0.00303      | 0.0031  | 0.00347            | 0.00328 | -        |
| 5         | 0.00324     | 0.00298      | 0.0031  | 0.00287       | 0.00305    | 0.00313    | 0.00303      | 0.0031  | 0.00347            | 0.00328 | —        |
| 6         | 0.00324     | 0.00298      | 0.0031  | 0.00287       | 0.00305    | 0.00313    | 0.00303      | 0.0031  | 0.00347            | 0.00328 | ~        |
| 7         | 0.00324     | 0.00298      | 0.0031  | 0.00287       | 0.00305    | 0.00313    | 0.00303      | 0.0031  | 0.00347            | 0.0O    | ~        |
| 8         | 0.0O        | —            | —       | —             | —          | —          | —            | —       | —                   | —      | ~        |
| 9         | —           | —            | —       | —             | —          | —          | —            | —       | —                   | —      | ~        |
| 10        | —           | —            | —       | —             | —          | —          | —            | —       | —                   | —      | ~        |
| Final     | —           | —            | —       | —             | —          | —          | —            | —       | —                   | ~      | ~        |
| Final     | —           | —            | —       | —             | —          | —          | —            | ~       | ~                   | ~      | ~        |
| Final     | —           | —            | —       | ~       | ~          | ~          | ~            | ~       | ~                   | ~      | ~        |
| Final     | —           | —            | ~       | ~       | ~          | ~          | ~            | ~       | ~                   | ~      | ~        |
| Final     | —           | —            | ~       | ~       | ~          | ~          | ~            | ~       | ~                   | ~      | ~        |
| Final     | —           | —            | ~       | ~       | ~          | ~          | ~            | ~       | ~                   | ~      | ~        |

| Final     | Sun397      | ImageNet     |
| --------- | ----------- | -------------- |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
|
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Final     | Sun397      | ImageNet     |
| Global    A: - Sun397 - Sun397 - ImageNet
| Sun397 - Sun397 - Sun397 - ImageNet
| Sun397 - Sun397 - Sun397 - ImageNet
| Sun397 - Sun397 - Sun397 - ImageNet
| Sun397 - Sun397 - Sun397 - ImageNet
| Sun397 - Sun397 - Sun397 - ImageNet
| Sun397 - Sun39<ecel><ecel><ecel><ecel><ecel><ecel><ecel><ecel><ecel><ecel><ecel><ecel><nl>
</details>

Figure 5. Convergence of the proposed method on 11 datasets. The method converges within a few iterations, indicating efficient and stable optimization.