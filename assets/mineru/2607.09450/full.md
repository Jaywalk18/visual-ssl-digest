# Robustifying Vision-Language Models via Test-Time Prompt Adaptation

Xingyu Zhu <sup>1</sup> <sup>2</sup> Huanshen Wu <sup>2</sup> Shuo Wang <sup>2</sup> <sup>\*</sup> Beier Zhu <sup>2</sup> Jiannan Ge <sup>2</sup> Jiaheng Zhang <sup>1</sup> Long Chen <sup>3</sup>

## Abstract

Pre-trained Vision-Language Models (VLMs) such as CLIP achieve strong zero-shot generalization, but their performance degrades sharply under adversarial perturbations. Existing test-time adaptation methods typically rely on sample-level confidence heuristics, overlooking the intrinsic distributional structure of the data. This samplecentric approach limits robustness, as it fails to distinguish confident adversarial mispredictions from true semantic consistency. In this work, we observe that adversarial distortion is structurally brittle: while holistic representations are corrupted, semantic integrity is often preserved in the distribution of augmented views. Motivated by this insight, we propose RITA, a Robust testtIme prompT Adaptation framework that shifts from sample-level estimates to distribution-level alignment. Specifically, RITA employs optimal transport to align the distribution of augmented visual features with textual prototypes, mitigating adversarial outliers and rectifying cross-modal semantic misalignment. Furthermore, we introduce a dynamic cache to progressively accumulate reliable cues from the test stream for online refinement. Extensive experiments demonstrate that RITA significantly improves adversarial robustness without compromising clean accuracy.

## 1. Introduction

Vision-Language Models (VLMs) (Li et al., 2022; Alayrac et al., 2022; Li et al., 2023; Zhu et al., 2024b; 2026c) like CLIP (Radford et al., 2021), pre-trained on massive imagetext pairs, have achieved remarkable zero-shot generalization. Despite this success, VLMs remain highly vulnerable to adversarial perturbations: imperceptible noise can cause severe performance degradation (Szegedy et al., 2014; Madry et al., 2018; Zhu et al., 2024a; 2025b), posing security risks in real word applications.

![](images/ab7cdbc897de5457e6e6ead39ccf81cc4b2a3c78ff28d5e13e8c811fef9c4ef4.jpg)  
(b)

![](images/e4a2642b8661f17f3099bb6c15f565cb8186f0f6998c4dc03786cf66237608fc.jpg)  
(d)

![](images/6412fd39af3cf4b21afba3cd9afaff61e3f458647b40429c50afa469ca518ac2.jpg)

![](images/e640cf607d5135cdaf0b686b651030005195a58a92ca2d11d8818dfa6ff70d79.jpg)  
Figure 1. Augmented views retain more semantic cues under adversarial perturbations, enabeling a cache for distribution alignment that improves adversarial performance. (a) Visualization of adversarially perturbed images, where each point represents an image and different colors denote ground-truth classes. (b) Visualization of multiple augmented views generated from the same adversarial image, colored by class label, where semantic structure partially re-emerges with improved class separability compared to (a). (c) Our method leverages the selected augmented views as a cache and aligns them with textual prompts. (d) Performance comparison across different VLM backbones, demonstrating improved robustness under adversarial attacks.

Existing efforts to enhance the robustness of VLMs generally fall into two categories. The first line of work utilizes adversarial training (Mao et al., 2023; Schlarmann et al., 2024; Wang et al., 2024; Zhang et al., 2024), which aims to immunize models by explicitly integrating adversarial examples into the optimization loop. While effective, these approaches typically incur prohibitive computational costs due to on-the-fly attack generation and require access to taskspecific labeled data, thereby undermining the scalability and zero-shot flexibility inherent to foundation models. The second line of work explores test-time prompt tuning (Yoon et al., 2024; Shu et al., 2022; Zhao et al., 2025), an efficient paradigm that adapts learnable prompt contexts or predictions during inference without modifying model parameters. However, most existing test-time methods (Wang et al., 2025; Sheng et al., 2025) primarily rely on samplelevel confidence (e.g., entropy) to filter augmented views. These methods treat augmentations as isolated data points, overlooking the intrinsic distributional structure and latent semantics. Consequently, they fail to distinguish between confident adversarial mispredictions and true semantic consistency, limiting their effectiveness under attack.

To address the above limitation, we revisit how adversarial perturbations interact with test-time augmentations in VLMs. A typical white-box attack is crafted on the original input image to maximally disrupt image-text matching based on its holistic representation. This can drastically shift the attacked image embedding in the feature space and make samples from different classes highly entangle as shown in Figure 1(a). Importantly, this adversarial effect is not equally persistent across augmentations. Geometric transformations such as random cropping and flipping change the spatial correspondence of pixels, so the perturbation pattern optimized for the original configuration becomes partially mismatched after transformation. Empirically, different views exhibit heterogeneous behaviors: while some views remain strongly influenced by the attack, others still produce confident and semantically consistent predictions (Figure 1(b)). This observation suggests that robustness should be built by exploiting the distribution of augmented views and the semantic relations it contains, rather than relying solely on sample-level confidence heuristics.

Motivated by the above observation, we propose RITA, a robust test-time framework that shifts from sample-level matching to distribution-level alignment. Instead of relying on a single image embedding, RITA models the visual input as a discrete distribution over augmented views. To bridge these visual features with textual prototypes, we formulate the alignment objective as an Optimal Transport (OT) (Cuturi, 2013) problem. As illustrated in Figure 1(c), this formulation enables us to evaluate the global geometric correspondence between the visual distribution and textual representations, mitigating the influence of adversarial outliers to rectify semantic misalignment. Moreover, test samples arrive as a continuous stream, providing additional information beyond a single image. To leverage this property, RITA incorporates a dynamic cache mechanism that progressively accumulates reliable semantic views, and uses them to further refine distribution alignment online. As demonstrated in Figure 1(d), this progressive adaptation significantly enhances zero-shot robustness and maintaining competitive performance on clean data.

Extensive experiments on multiple standard benchmarks under diverse adversarial attacks demonstrate that RITA significantly enhances zero-shot robustness while preserving competitive performance on clean data. Our contributions are summarized as follows:

• We propose RITA, a robust test-time prompt adaptation framework that leverages augmented views to rectify adversarial misalignment at the distribution level.

• We formulate cross-modal semantic alignment as an Optimal Transport problem, complemented by a dynamic cache for progressive refinement.

• We conduct comprehensive evaluations, showing that RITA consistently outperforms existing test-time adaptation methods.

## 2. Related Work

Adversarial Defense in VLMs. The vulnerability of VLMs (e.g., CLIP (Radford et al., 2021)) to adversarial perturbations remains a critical challenge (Dong et al., 2018; Madry et al., 2018; Zhao et al., 2023; Zhu et al., 2026d). Attacks have evolved from uni-modal perturbations (Carlini & Wagner, 2017a; Xie et al., 2019) to multi-modal strategies like Co-Attack (Zhang et al., 2022a) that disrupt cross-modal alignment. To enhance robustness, prior defenses utilize training-time strategies, primarily adversarial contrastive tuning (Mao et al., 2023; Schlarmann et al., 2024; Wang et al., 2024). However, these methods typically require expensive re-training and labeled data, limiting practicality. Consequently, inference-time robustness has emerged to secure models without weight updates, such as diffusion purification (Feng et al., 2023; 2025) and optimization-based methods (Wang et al., 2021; Zhang et al., 2022b). Notably, Test-Time Prompt Tuning approaches like TAPT (Wang et al., 2025) and R-TPT (Sheng et al., 2025) adapt prompts using unlabeled data via contrastive learning or entropy minimization. Nevertheless, these paradigms typically treat visual inputs as isolated points, failing to exploit the underlying distributional geometry of adversarial samples. In contrast, we propose RITA, a test-time prompt tuning framework that shifts from point-level alignment to distributionlevel modeling.

Optimal transport. Optimal Transport (OT) (Cuturi, 2013) provides a principled way to compare probability distributions by accounting for the geometry of the underlying feature space. With efficient solvers such as Sinkhorn (Altschuler et al., 2017; Mensch & Peyre´, 2020), OT has been widely used in generative modeling (Arjovsky et al., 2017), structural alignment (Xu et al., 2019), and domain adaptation (Courty et al., 2016). In vision–language learning, OT has also been applied to reduce semantic misalignment, including few-shot learning (Lazarou et al., 2021), distribution calibration (Guo et al., 2022; Damodaran et al., 2018; Zhu et al., 2025a;b), and prompt learning (Chen et al., 2023; Wang et al., 2023; Ren et al., 2025). For example, PLOT (Chen et al., 2023) aligns image features with multiple prompts via OT-based matching to capture diverse semantics, while ALIGN (Wang et al., 2023) further introduces hierarchical/token-level transportation for fine-grained cross-modal alignment (Zhu et al., 2026a;b). AWT (Zhu et al., 2024c) similarly formulates image-text distance as an OT problem to model semantic correlations in the joint space. However, these approaches primarily focus on representation enhancement in clean settings. They rely on undistorted visual manifolds and do not account for the severe structural perturbations caused by adversarial attacks. Differing from these approaches, RITA repurposes OT for adversarial defense, aiming to reconstruct the distributionlevel correspondence disrupted by attacks. This enables inference-time correction of structural misalignment without parameter updates, ensuring robust deployment.

## 3. Method

An overview of RITA is illustrated in Figure 2. We first introduce the preliminaries in Sec. 3.1, then detail our distributed feature modeling and dynamic distribution alignment in Sec. 3.2 and 3.3, respectively. Finally, we provide a theoretical justification in Sec. 3.4.

## 3.1. Preliminary

Test-time prompt tuning. Test-Time Prompt Tuning (TPT) (Shu et al., 2022) improves the zero-shot generalization of CLIP (Radford et al., 2021) by adapting textual prompts at inference time, without accessing labeled data or updating CLIP parameters. Given a test image $x _ { t }$ , TPT constructs a set of $N$ stochastic augmentations $\mathcal { X } _ { t } = \{ x _ { t } ^ { n } \} _ { n = 1 } ^ { N }$ For each augmented view $\boldsymbol { x } _ { t } ^ { n }$ , the visual representation is $\mathbf { x } _ { t } ^ { n } = \Phi _ { \mathrm { i m g } } ( x _ { t } ^ { n } )$ , where $\Phi _ { \mathrm { i m g } } ( \cdot )$ denotes the CLIP image encoder. On the text side, the prompt for class k is formulated as $z _ { k } = \{ \omega _ { 1 } , \omega _ { 2 } , \ldots , \omega _ { L } , c _ { k } \}$ , where $c _ { k }$ is the token embedding of the class name, and $\omega = \{ \omega _ { \ell } \} _ { \ell = 1 } ^ { L }$ are learnable context vectors shared across classes. The textual representation is $\mathbf { z } _ { k } = \Phi _ { \mathrm { t e x t } } ( z _ { k } )$ , with $\Phi _ { \mathrm { t e x t } } ( \cdot )$ being the CLIP text encoder. For each view $ { \boldsymbol { x } } _ { t } ^ { n }$ , the prediction probability is computed by feature matching:

$$
p (k \mid x _ {t} ^ {n}; \omega) = \frac {\exp \left(\cos (\mathbf {x} _ {t} ^ {n} , \mathbf {z} _ {k}) / \tau\right)}{\sum_ {j = 1} ^ {K} \exp \left(\cos (\mathbf {x} _ {t} ^ {n} , \mathbf {z} _ {j}) / \tau\right)},\tag{1}
$$

where τ is the temperature parameter and K is the number of categories. Specifically, TPT employs a confidence selection strategy to filter out unreliable augmentations. It selects a subset S of low-entropy views to compute the aggregated prediction $\begin{array} { r } { \bar { p } ( k \ \lvert \ x _ { t } ; \omega ) = \frac { 1 } { | S | } \sum _ { n \in S } p ( k \ \lvert \ x _ { t } ^ { n } ; \omega ) } \end{array}$ . The prompt ω is then optimized by minimizing the entropy of this aggregated distribution $\bar { p } \colon$

$$
\omega^ {*} = \underset {\omega} {\operatorname{argmin}} \Big (- \sum_ {k = 1} ^ {K} \bar {p} (k \mid x _ {t}; \omega) \log \bar {p} (k \mid x _ {t}; \omega) \Big).\tag{2}
$$

Optimal transport. Optimal Transport (OT) (Cuturi, 2013) provides a principled way to measure the discrepancy between two probability distributions. Consider two discrete distributions in the feature space, $\begin{array} { r } { \mathbb { P } = \sum _ { n = 1 } ^ { N } a ^ { n } \delta _ { \mathbf { x } ^ { n } } } \end{array}$ and $\begin{array} { r } { \mathbb { Q } = \sum _ { m = 1 } ^ { M } b ^ { m } \delta _ { \mathbf { z } ^ { m } } } \end{array}$ , where $\delta _ { \mathbf { v } }$ denotes the Dirac delta function at location $\mathbf { v } ,$ and $\mathbf { a } \in \Delta _ { N } , \mathbf { b } \in \Delta _ { M }$ are probability vectors. Given a cost matrix $\mathbf { C } \in \mathbb { R } ^ { N \times M }$ , where $\mathbf { C } _ { n m }$ measures the transport cost from $\mathbf { x } ^ { n }$ to $\mathbf { z } ^ { m }$ , the entropyregularized OT distance is defined as:

$$
d _ {\mathrm{OT}} (\mathbb {P}, \mathbb {Q}; \mathbf {C}) = \min _ {\mathbf {T} \in \Pi (\mathbf {a}, \mathbf {b})} \langle \mathbf {T}, \mathbf {C} \rangle - \lambda h (\mathbf {T}),\tag{3}
$$

where $\Pi ( \mathbf { a } , \mathbf { b } ) = \{ \mathbf { T } \in \mathbb { R } _ { + } ^ { N \times M } \mid \mathbf { T } \mathbb { 1 } _ { M } = \mathbf { a } , \mathbf { T } ^ { \top } \mathbb { 1 } _ { N } = \mathbf { b } \}$ is the transport polytope, $\begin{array} { r } { { h } ( { \bf T } ) = - \sum _ { n , m } T _ { n m } } \end{array}$ log $T _ { n m }$ is the entropic regularization term, and $\lambda \geq 0$ controls the regularization strength. This formulation enables efficient computation via the Sinkhorn algorithm.

## 3.2. Distributed Features Modeling

In this section, we present our core strategy for robust testtime inference. Moving beyond vulnerable holistic image embeddings, we model the adversarial image and textual prompts as discrete distributions and align them structurally using optimal transport.

Adversarial perturbations. Adversarial attacks aim to degrade model predictions by introducing small, carefully crafted perturbations. In a white-box setting, given a clean image x and its ground-truth label $y ,$ the adversarial example $x ^ { \prime }$ is generated by maximizing the cross-entropy loss within an $\ell _ { p }$ -norm constraint:

$$
x ^ {\prime} = \underset {\| x ^ {\prime} - x \| _ {p} \leq \epsilon_ {\mathrm{adv}}} {\operatorname{argmax}} \mathcal {L} _ {\mathrm{CE}} \Big (p (\cdot \mid x ^ {\prime}; z), y \Big),\tag{4}
$$

where $\epsilon _ { \mathrm { a d v } }$ denotes the perturbation budget, and $p ( \cdot \mid x ^ { \prime } ; z )$ represents the probability distribution over $K$ classes as defined in Eq. (1). Typically, we approximate this optimization using the iterative Projected Gradient Descent (PGD) (Madry et al., 2018).

Multi-prototype distribution alignment. While adversarial attacks distort alignment at the global representation level, relying on single-point embeddings is insufficient to capture the semantic variations. To alleviate this, we construct sets of diverse visual and textual representations and model their correspondence at the distribution level.

Specifically, given an adversarial image $\hat { x } _ { t }$ , we apply data augmentations to obtain N views $\{ \hat { x } _ { t } ^ { n } \} _ { n = 1 } ^ { N }$ , producing visual features $\{ \hat { \mathbf { x } } _ { t } ^ { n } \} _ { n = 1 } ^ { N }$ . Instead of a single text prototype, for each class $k ,$ we construct M prompts $\{ z _ { k } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ using learnable context vectors, yielding textual features $\{ \mathbf { z } _ { k } ^ { m } \bar  \} _ { m = 1 } ^ { M }$ . We model the adversarial image and the k-th class prototype as discrete distributions:

![](images/a462f9eb9c9ea25d79a1ed64b560fedb3e5f5637ee188620208e3fa54cda33f0.jpg)  
Figure 2. Overview of the proposed RITA framework. Given an adversarial test image, RITA extracts multi-view visual features and class-specific textual prototypes using a frozen CLIP encoder. Both modalities are modeled as discrete distributions and aligned via entropy-regularized optimal transport. Low-entropy views are used to update a dynamic cache of reliable semantics.

$$
\mathbb {P} _ {t} = \sum_ {n = 1} ^ {N} \frac {1}{N} \delta_ {\hat {\mathbf {x}} _ {t} ^ {n}}, \quad \mathbb {Q} _ {k} = \sum_ {m = 1} ^ {M} \frac {1}{M} \delta_ {\mathbf {z} _ {k} ^ {m}}.\tag{5}
$$

Here, we assign uniform weights to both modalities, $i . e .$ $\begin{array} { r } { { \bf a } _ { t } = \frac { 1 } { N } \mathbb { 1 } _ { N } } \end{array}$ and $\mathbf { b } _ { k } = \textstyle \frac { 1 } { M } \mathbb { 1 } _ { M }$ , such that the marginal constraints of OT are satisfied. We then measure the distributionlevel alignment between the visual distribution $\mathbb { P } _ { t }$ and the textual prototype distribution $\mathbb { Q } _ { k }$ via entropy-regularized optimal transport:

$$
d _ {\mathrm{OT}} \left(\mathbb {P} _ {t}, \mathbb {Q} _ {k}; \mathbf {C} _ {t, k}\right) = \min _ {\mathbf {T} _ {t, k} \in \Pi} \left(\left\langle \mathbf {T} _ {t, k}, \mathbf {C} _ {t, k} \right\rangle - \lambda h \left(\mathbf {T} _ {t, k}\right)\right),\tag{6}
$$

where $\mathbf { T } _ { t , k } \in \mathbb { R } _ { + } ^ { N \times M }$ is the transport plan, and the cost matrix is defined by the cosine distance, $\mathbf { C } _ { t , k } ( n , m ) = 1 -$ $\cos ( \hat { \mathbf { x } } _ { t } ^ { n } , \mathbf { z } _ { k } ^ { m } )$ . Intuitively, a smaller OT distance indicates better alignment between the test image and class k at the distribution level. Accordingly, we predict the label by selecting the class with the minimum transport cost:

$$
\hat {y} = \underset {k \in [ K ]} {\operatorname{argmin}} d _ {\mathrm{OT}} (\mathbb {P} _ {t}, \mathbb {Q} _ {k}; \mathbf {C} _ {t, k}).\tag{7}
$$

## 3.3. Dynamic Distribution Alignment

While distribution-level alignment in Sec. 3.2 effectively processes individual samples, it treats inference steps in isolation, neglecting the semantic consensus in the continuous test stream. To exploit this temporal information, we introduce a dynamic cache mechanism that accumulates reliable visual features to iteratively refine the alignment online.

Confidence-based cache update. To update the cache with reliable samples, we evaluate the confidence of each individual augmented view. Instead of solving the full global transport problem, we quantify the instance-level alignment by measuring the average affinity between the view $\hat { \mathbf { x } } _ { t } ^ { n }$ and the text distribution of class k:

$$
p _ {t} ^ {n} (k) = \frac {\exp \left(\frac {1}{M} \sum_ {m = 1} ^ {\tau M} \cos (\hat {\mathbf {x}} _ {t} ^ {n} , \mathbf {z} _ {k} ^ {m})\right)}{\sum_ {j = 1} ^ {K} \exp \left(\frac {1}{M} \sum_ {m = 1} ^ {\tau M} \cos (\hat {\mathbf {x}} _ {t} ^ {n} , \mathbf {z} _ {j} ^ {m})\right)}.\tag{8}
$$

We compute entropy $\begin{array} { r } { H ( p _ { t } ^ { n } ) = - \sum _ { k = 1 } ^ { K } p _ { t } ^ { n } ( k ) } \end{array}$ log p<sup>n</sup><sub>t</sub> (k) and retain confident views:

$$
\mathcal {B} _ {t} = \left\{(\hat {\mathbf {x}} _ {t} ^ {n}, \hat {y} _ {t} ^ {n})   \bigg |   H (p _ {t} ^ {n}) \leq \gamma ,   \hat {y} _ {t} ^ {n} = \underset {k} {\operatorname{argmax}} p _ {t} ^ {n} (k) \right\}.\tag{9}
$$

We maintain a class-wise cache $\{ \hat { \mathbf { X } } _ { k } \} _ { k = 1 } ^ { K }$ , where $\hat { \mathbf { X } } _ { k } \in \mathbb { }$ $\mathbb { R } ^ { N _ { k } \times d }$ stores cached visual features pseudo-labeled as class $k \ ( N _ { k }$ is the current cache size). The cache is updated online by prioritizing lower-entropy samples. To bridge the modality gap, we align cached visual features to the textual space of class k by solving an Orthogonal Procrustes (Ouali et al., 2023) problem:

$$
\mathbf {W} _ {k} ^ {*} = \underset {\mathbf {W} ^ {\top} \mathbf {W} = \mathbf {I}} {\text {argmin}} \left\| \hat {\mathbf {X}} _ {k} \mathbf {W} - \mathbf {1} _ {N _ {k}} \bar {\mathbf {z}} _ {k} ^ {\top} \right\| _ {F} ^ {2},\tag{10}
$$

where $\begin{array} { r } { \bar { \mathbf { z } } _ { k } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbf { z } _ { k } ^ { m } } \end{array}$ is the mean text embedding for class k. The aligned cached features are $\tilde { \mathbf { X } } _ { k } = \hat { \mathbf { X } } _ { k } \mathbf { W } _ { k } ^ { * }$

Cache-based distribution matching. For each class $k ,$ we instantiate a discrete probability distribution over the

aligned cached features:

$$
\tilde {\mathbb {Q}} _ {k} = \frac {1}{N _ {k}} \sum_ {j = 1} ^ {N _ {k}} \delta_ {\tilde {\mathbf {x}} _ {k} ^ {j}}, \quad \text { where } \tilde {\mathbf {x}} _ {k} ^ {j} \in \tilde {\mathbf {X}} _ {k}.\tag{11}
$$

We quantify the discrepancy via the same entropyregularized OT distance as in Sec. 3.2, but with a cachespecific cost matrix $\tilde { \mathbf { C } } _ { t , k } ( n , j ) = 1 - \cos ( \hat { \mathbf { x } } _ { t } ^ { n } , \tilde { \mathbf { x } } _ { k } ^ { j } )$ , and denote it as $d _ { \mathrm { c O T } } \big ( \mathbb { P } _ { t } , \tilde { \mathbb { Q } } _ { k } ; \tilde { \mathbf { C } } _ { t , k } \big )$

Final inference. We classify the test sample by identifying the category that minimizes the joint transport cost, which integrates both the global prompt alignment and the local cache consensus:

$$
\hat {y} = \underset {k \in [ K ]} {\operatorname{argmin}} \left(d _ {\mathrm{OT}} (\mathbb {P} _ {t}, \mathbb {Q} _ {k}; \mathbf {C} _ {t, k}) + \alpha   d _ {\mathrm{cOT}} (\mathbb {P} _ {t}, \tilde {\mathbb {Q}} _ {k}; \tilde {\mathbf {C}} _ {t, k})\right),\tag{12}
$$

where $\alpha \geq 0$ controls the contribution of the dynamic cache.

## 3.4. Theoretical Analysis

Standard methods like TPT (Shu et al., 2022) and R-TPT (Sheng et al., 2025) optimize centroid alignment via mean pooling and cosine similarity. For $\ell _ { 2 } \cdot$ -normalized CLIP features, this is equivalent to minimizing the squared Euclidean distance between centroids $( \| \mathbf { x } - \mathbf { z } \| ^ { 2 } = 2 ( 1 -$ $\cos ( \mathbf { x } , \mathbf { z } ) )$ . To reveal RITA’s geometric advantage, we analyze alignment using the 2-Wasserstein distance (W<sub>2</sub>). Let <sup>P</sup><sub>t</sub> $( \mu _ { x } , \Sigma _ { x } )$ and $\mathbb { Q } _ { k } \left( \mu _ { z } , \Sigma _ { z } \right)$ be the visual and textual distributions. We obtain the following decomposition:

Theorem 3.1 (Decomposition of Alignment Objective). The Optimal Transport objective $( \mathcal { L } _ { \mathrm { O T } } )$ imposes a stricter bound by decomposing into a centroid alignment term and a structural variance penalty:

$$
\underbrace {W _ {2} ^ {2} (\mathbb {P} _ {t} , \mathbb {Q} _ {k})} _ {\mathcal {L} _ {\mathrm{OT}} (R I T A)} \approx \underbrace {\| \boldsymbol {\mu} _ {x} - \boldsymbol {\mu} _ {z} \| ^ {2}} _ {\mathcal {L} _ {\text {mean}} (T P T \text {-equivalent})} + \underbrace {\mathfrak {B} ^ {2} (\boldsymbol {\Sigma} _ {x} , \boldsymbol {\Sigma} _ {z})} _ {\mathcal {R} _ {\text {var}} (\text {Structural Penalty})},\tag{13}
$$

where $\mathfrak { B } ^ { 2 } ( \mathbf { A } , \mathbf { B } ) = \mathrm { T r } ( \mathbf { A } + \mathbf { B } - 2 ( \mathbf { A } ^ { 1 / 2 } \mathbf { B } \mathbf { A } ^ { 1 / 2 } ) ^ { 1 / 2 } )$ is the Bures metric, quantifying the geometric mismatch between covariances.

Equation (13) holds exactly for Gaussian distributions and serves as a general lower bound, as derived in Appendix A. This inequality highlights a critical robustness gap: minimizing only $\mathcal { L } _ { \mathrm { m e a n } }$ leaves structural variance ${ \mathcal { R } } _ { \operatorname { v a r } }$ unconstrained, allowing attackers to distort distribution geometry.

## 4. Experiments

## 4.1. Setup

Datasets. We evaluate our method on eight image classification benchmarks spanning a wide range of visual domains, including generic object recognition (Caltech101 (Fei-Fei et al., 2004)), texture recognition (DTD (Cimpoi et al., 2014)), satellite imagery (EuroSAT (Helber et al., 2019)), human action recognition (UCF101 (Soomro et al., 2012)), as well as several fine-grained classification tasks, namely Pets (Parkhi et al., 2012), Cars (Krause et al., 2013), Flowers (Nilsback & Zisserman, 2008), and Aircraft (Maji et al., 2013). To further assess robustness under distribution shifts, we conduct additional evaluations on ImageNet (Deng et al., 2009) and four of its variants that share the same label space: ImageNetV2 (Recht et al., 2019), ImageNet-Sketch (Wang et al., 2019), ImageNet-A (Hendrycks et al., 2021b), and ImageNet-R (Hendrycks et al., 2021a). These benchmarks introduce significant variations in image sources, styles, and underlying visual statistics. Detailed analysis of these datasets is provided in Appendix C.1.

Implementation details. We build all experiments on the official pre-trained CLIP models with two backbones, CLIP-ViT-B/32 and CLIP-ViT-B/16. We generate adversarial examples using PGD (Madry et al., 2018) under an $L _ { \infty }$ constraint. We use ϵ = 4.0 with 7 steps for both backbones. At test time, we update only the prompt parameters while keeping the CLIP backbone freezed. The prompt number M is set to 4 and initialized with the template “a photo o $\because a ^ { \prime \prime }$ We use AdamW and fix the Test-Time Adaptation (TTA) step at 1 per test sample, with a learning rate of 0.005. We apply standard test-time augmentations for images, including random cropping, resizing, and horizontal flipping. For text, we use an Large Language Model (LLM) to generate class-specific descriptions(Zhu et al., 2024c). For each test image, we sample N = 64 augmented views (including the original). For cache construction, We set the entropy threshold to $\gamma = 0 . 8$ . More experiment details are provided in Appendix B.

Comparison methods. We compare RITA with CLIPbased test-time adaptation baselines, including TPT (Shu et al., 2022), R-TPT (Sheng et al., 2025), C-TPT (Yoon et al., 2024), and MTA (Zanella & Ayed, 2024), as well as the zero-shot CLIP baseline. We also report an Ensemble baseline that averages predictions over augmented views. To further evaluate compatibility with robust pre-trained weights, We incorporate three representative adversarially fine-tuned CLIP models, namely TeCoA (Mao et al., 2023), PMG (Wang et al., 2024), and FARE (Schlarmann et al., 2024). All methods follow the instance-level test-time adaptation protocol: each test sample is adapted and predicted independently, without access to other test samples.

## 4.2. Main Results

Results on fine-grained datasets. We evaluate our method on eight fine-grained benchmark datasets with ViT-B/32 and ViT-B/16 backbones, as presented in Table 1. The results show that RITA achieves the highest adversarial accuracy on nearly all datasets. Compared to vanilla CLIP, RITA improves average robustness by 45.0% and 50.9% with ViT-B/32 and ViT-B/16, respectively. This effectively mitigates the severe vulnerability of the baseline under adversarial attacks, which can cause near-zero accuracy on datasets such as Cars and Aircraft. Furthermore, compared to R-TPT, the state-of-the-art test-time adaptation method, RITA achieves consistent gains in average robust accuracy, outperforming it by 1.8% and 2.2% on the two architectures. Meanwhile, our method also achieves the highest average accuracy on clean samples.This demonstrates that RITA enhances adversarial robustness while preserving recognition performance in attack-free environments with negligible degradation. Notably, RITA performs better with ViT-B/16 than with ViT-B/32, aligning with the intuition that a fine-grained ViT backbone yields stronger recognition capabilities.

Table 1. Results (%) of adaptation methods on fine-grained classification datasets with ϵ set to 1.0. Bold and underlined entries indicate the best and second-best results, respectively. Acc. denotes accuracy on clean data, and Rob. denotes accuracy under adversarial perturbations.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">Caltech101</td><td colspan="2">Pets</td><td colspan="2">Cars</td><td colspan="2">Flower102</td><td colspan="2">Aircraft</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">UCF101</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td rowspan="7">ViT-B/32</td><td>CLIP</td><td>90.9</td><td>25.6</td><td>83.0</td><td>2.1</td><td>49.7</td><td>0.0</td><td>65.8</td><td>2.2</td><td>18.3</td><td>0.0</td><td>40.8</td><td>5.4</td><td>18.6</td><td>0.1</td><td>62.1</td><td>2.4</td><td>53.6</td><td>4.7</td></tr><tr><td>Ensemble</td><td>91.6</td><td>85.4</td><td>85.0</td><td>73.4</td><td>57.8</td><td>39.6</td><td>67.4</td><td>57.5</td><td>20.1</td><td>14.4</td><td>46.1</td><td>39.3</td><td>32.5</td><td>23.6</td><td>61.6</td><td>53.5</td><td>57.8</td><td>48.3</td></tr><tr><td>TPT</td><td>91.4</td><td>77.6</td><td>84.1</td><td>58.6</td><td>62.9</td><td>30.8</td><td>63.8</td><td>43.6</td><td>19.0</td><td>10.1</td><td>42.2</td><td>28.6</td><td>35.1</td><td>15.1</td><td>62.3</td><td>38.1</td><td>57.6</td><td>37.8</td></tr><tr><td>C-TPT</td><td>91.8</td><td>75.9</td><td>84.9</td><td>52.2</td><td>60.8</td><td>27.1</td><td>65.9</td><td>42.1</td><td>17.7</td><td>8.7</td><td>44.3</td><td>27.1</td><td>34.7</td><td>9.0</td><td>62.6</td><td>35.3</td><td>57.8</td><td>34.6</td></tr><tr><td>MTA</td><td>91.8</td><td>80.8</td><td>85.8</td><td>62.6</td><td>64.1</td><td>34.5</td><td>64.8</td><td>44.8</td><td>20.4</td><td>11.1</td><td>44.0</td><td>29.3</td><td>34.5</td><td>7.8</td><td>63.6</td><td>40.1</td><td>58.6</td><td>38.8</td></tr><tr><td>R-TPT</td><td>90.6</td><td>86.2</td><td>84.5</td><td>73.1</td><td>63.1</td><td>44.6</td><td>62.6</td><td>53.1</td><td>19.1</td><td>12.9</td><td>42.1</td><td>36.7</td><td>32.0</td><td>22.4</td><td>62.8</td><td>54.2</td><td>57.1</td><td>47.9</td></tr><tr><td>RITA</td><td>92.3</td><td>86.5</td><td>85.9</td><td>74.5</td><td>59.6</td><td>42.9</td><td>68.7</td><td>58.7</td><td>20.2</td><td>15.2</td><td>46.2</td><td>40.1</td><td>33.4</td><td>24.8</td><td>62.8</td><td>55.0</td><td>58.6</td><td>49.7</td></tr><tr><td rowspan="7">ViT-B/16</td><td>CLIP</td><td>85.9</td><td>10.8</td><td>83.5</td><td>0.5</td><td>55.7</td><td>0.0</td><td>61.7</td><td>0.1</td><td>15.7</td><td>0.0</td><td>40.4</td><td>2.4</td><td>23.7</td><td>0.0</td><td>58.9</td><td>0.5</td><td>53.2</td><td>1.8</td></tr><tr><td>Ensemble</td><td>92.1</td><td>87.4</td><td>88.7</td><td>77.2</td><td>63.2</td><td>46.7</td><td>70.8</td><td>59.9</td><td>25.9</td><td>17.9</td><td>50.9</td><td>43.2</td><td>32.9</td><td>26.7</td><td>64.6</td><td>54.3</td><td>61.1</td><td>51.6</td></tr><tr><td>TPT</td><td>94.1</td><td>79.6</td><td>87.4</td><td>62.8</td><td>66.5</td><td>35.5</td><td>66.1</td><td>48.3</td><td>23.4</td><td>12.3</td><td>45.9</td><td>29.1</td><td>42.6</td><td>7.4</td><td>67.9</td><td>39.7</td><td>61.7</td><td>39.3</td></tr><tr><td>C-TPT</td><td>93.9</td><td>76.5</td><td>88.2</td><td>55.8</td><td>65.8</td><td>30.5</td><td>69.6</td><td>45.5</td><td>23.9</td><td>9.8</td><td>45.9</td><td>26.6</td><td>42.3</td><td>7.1</td><td>65.6</td><td>34.7</td><td>61.9</td><td>35.8</td></tr><tr><td>MTA</td><td>94.3</td><td>81.9</td><td>88.0</td><td>64.5</td><td>67.7</td><td>38.2</td><td>65.0</td><td>46.9</td><td>24.0</td><td>12.6</td><td>46.5</td><td>28.7</td><td>42.5</td><td>13.7</td><td>67.5</td><td>40.8</td><td>61.9</td><td>40.9</td></tr><tr><td>R-TPT</td><td>93.7</td><td>87.8</td><td>87.2</td><td>74.7</td><td>67.0</td><td>46.9</td><td>68.7</td><td>55.7</td><td>23.9</td><td>17.3</td><td>46.4</td><td>39.7</td><td>34.7</td><td>26.8</td><td>67.2</td><td>55.4</td><td>61.1</td><td>50.5</td></tr><tr><td>RITA</td><td>93.8</td><td>88.5</td><td>89.8</td><td>77.3</td><td>64.2</td><td>47.1</td><td>71.6</td><td>61.3</td><td>26.2</td><td>19.2</td><td>51.5</td><td>44.7</td><td>33.4</td><td>27.6</td><td>65.5</td><td>55.8</td><td>62.0</td><td>52.7</td></tr></table>

Table 2. Classification accuracy (%) on 8 datasets using different adversarially finetuned CLIP models.

<table><tr><td rowspan="2">Method</td><td colspan="2">Caltech101</td><td colspan="2">Pets</td><td colspan="2">Cars</td><td colspan="2">Flower102</td><td colspan="2">Aircraft</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">UCF101</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td>TeCoA</td><td>77.6</td><td>64.3</td><td>59.8</td><td>39.9</td><td>20.6</td><td>9.3</td><td>37.1</td><td>23.9</td><td>5.7</td><td>2.8</td><td>24.1</td><td>14.4</td><td>15.9</td><td>12.4</td><td>40.5</td><td>23.4</td><td>35.2</td><td>23.8</td></tr><tr><td>+ Ensemble</td><td>76.0</td><td>68.3</td><td>61.3</td><td>54.8</td><td>19.4</td><td>14.5</td><td>38.4</td><td>33.2</td><td>6.9</td><td>4.9</td><td>27.5</td><td>25.2</td><td>11.5</td><td>12.0</td><td>39.4</td><td>34.5</td><td>35.1</td><td>30.9</td></tr><tr><td>+ MTA</td><td>79.5</td><td>56.5</td><td>61.6</td><td>28.8</td><td>20.9</td><td>10.4</td><td>37.2</td><td>24.0</td><td>6.6</td><td>3.8</td><td>25.8</td><td>19.7</td><td>12.2</td><td>11.2</td><td>42.0</td><td>17.3</td><td>35.7</td><td>21.4</td></tr><tr><td>+ R-TPT</td><td>77.1</td><td>70.3</td><td>60.9</td><td>54.0</td><td>23.1</td><td>17.8</td><td>35.4</td><td>30.8</td><td>7.0</td><td>4.7</td><td>26.7</td><td>24.7</td><td>12.0</td><td>11.8</td><td>40.1</td><td>35.0</td><td>35.3</td><td>31.1</td></tr><tr><td>+ RITA</td><td>78.1</td><td>70.4</td><td>62.3</td><td>54.9</td><td>21.1</td><td>15.7</td><td>39.2</td><td>34.0</td><td>7.1</td><td>5.5</td><td>27.8</td><td>25.5</td><td>12.2</td><td>12.1</td><td>40.9</td><td>35.8</td><td>36.1</td><td>31.7</td></tr><tr><td>PMG</td><td>82.3</td><td>70.8</td><td>61.8</td><td>41.4</td><td>24.7</td><td>12.5</td><td>36.2</td><td>25.3</td><td>5.2</td><td>2.9</td><td>22.8</td><td>16.0</td><td>17.1</td><td>12.8</td><td>42.6</td><td>27.8</td><td>36.5</td><td>26.1</td></tr><tr><td>+ Ensemble</td><td>78.9</td><td>71.7</td><td>60.7</td><td>54.1</td><td>16.6</td><td>11.6</td><td>37.8</td><td>32.7</td><td>7.2</td><td>5.1</td><td>26.6</td><td>25.2</td><td>14.0</td><td>13.9</td><td>42.0</td><td>37.7</td><td>35.5</td><td>31.5</td></tr><tr><td>+ MTA</td><td>79.5</td><td>65.4</td><td>61.8</td><td>31.3</td><td>17.9</td><td>12.9</td><td>36.7</td><td>21.6</td><td>5.8</td><td>3.3</td><td>22.9</td><td>18.7</td><td>13.8</td><td>12.5</td><td>43.1</td><td>22.6</td><td>35.1</td><td>23.5</td></tr><tr><td>+ R-TPT</td><td>79.3</td><td>73.2</td><td>62.0</td><td>55.1</td><td>18.3</td><td>14.9</td><td>35.3</td><td>30.4</td><td>5.4</td><td>4.1</td><td>25.4</td><td>23.0</td><td>13.2</td><td>12.9</td><td>42.3</td><td>38.2</td><td>35.2</td><td>31.4</td></tr><tr><td>+ RITA</td><td>79.9</td><td>73.7</td><td>62.5</td><td>55.4</td><td>18.9</td><td>13.9</td><td>38.7</td><td>33.9</td><td>7.5</td><td>5.4</td><td>27.4</td><td>25.4</td><td>14.4</td><td>14.1</td><td>43.5</td><td>38.6</td><td>36.6</td><td>32.6</td></tr><tr><td>FARE</td><td>86.6</td><td>62.9</td><td>77.7</td><td>38.1</td><td>40.4</td><td>9.7</td><td>48.7</td><td>22.5</td><td>10.2</td><td>2.3</td><td>32.4</td><td>18.1</td><td>22.4</td><td>11.0</td><td>52.9</td><td>22.2</td><td>46.4</td><td>23.3</td></tr><tr><td>+ Ensemble</td><td>86.1</td><td>80.2</td><td>77.9</td><td>70.1</td><td>38.8</td><td>29.5</td><td>48.9</td><td>42.3</td><td>10.5</td><td>7.8</td><td>36.9</td><td>32.4</td><td>13.5</td><td>11.6</td><td>52.8</td><td>45.6</td><td>45.6</td><td>39.9</td></tr><tr><td>+ MTA</td><td>87.7</td><td>70.0</td><td>78.4</td><td>45.0</td><td>40.6</td><td>24.8</td><td>49.2</td><td>30.5</td><td>11.0</td><td>6.6</td><td>32.8</td><td>25.3</td><td>13.5</td><td>11.8</td><td>53.9</td><td>29.8</td><td>45.8</td><td>30.4</td></tr><tr><td>+ R-TPT</td><td>86.5</td><td>81.1</td><td>77.4</td><td>70.1</td><td>43.0</td><td>33.0</td><td>46.2</td><td>40.2</td><td>10.0</td><td>7.4</td><td>33.6</td><td>30.0</td><td>12.9</td><td>12.1</td><td>53.8</td><td>46.8</td><td>45.4</td><td>40.0</td></tr><tr><td>+ RITA</td><td>86.8</td><td>81.9</td><td>78.7</td><td>70.5</td><td>41.2</td><td>31.2</td><td>50.1</td><td>43.1</td><td>11.8</td><td>8.6</td><td>37.5</td><td>32.9</td><td>14.2</td><td>12.7</td><td>54.3</td><td>47.0</td><td>46.8</td><td>41.0</td></tr></table>

Results with adversarially finetuned CLIP models. As an inherently plug-and-play framework, we integrated RITA with three representative adversarially fine-tuned models, with results summarized in Table 2. RITA exhibits significant synergy with these robust baselines, substantially enhancing robust accuracy across all datasets without sacrificing clean performance. Notably, when combined with FARE, RITA achieves a striking 81.9% robust accuracy on Caltech101, representing a 19.0% improvement over the baseline. These results demonstrate that RITA is a versatile adaptation framework that can seamlessly fortify existing adversarial defense models during inference.

Results under different attack types. Table 3 further assesses the generalizability of RITA against CW (Carlini & Wagner, 2017b) and DI (Xie et al., 2019) attacks. RITA consistently demonstrates superior defense across protocols, achieving a leading average accuracy of 54.0% under CW attacks. Under the more challenging DI attack, RITA maintains its advantage with 46.2% accuracy, surpassing R-TPT by 2.0%. These results suggest that RITA enhances the intrinsic robustness of VLMs against diverse adversarial threats via its effective test-time adaptation mechanism.

Table 3. Results (%) of adaptation methods on fine-grained classification datasets under different attacks using ViT-B/16 with ϵ = 1.0.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">Caltech101</td><td colspan="2">Pets</td><td colspan="2">Cars</td><td colspan="2">Flower102</td><td colspan="2">Aircraft</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">UCF101</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td rowspan="7">CW</td><td>CLIP</td><td>85.9</td><td>22.7</td><td>83.5</td><td>7.7</td><td>55.7</td><td>6.6</td><td>61.7</td><td>5.8</td><td>15.7</td><td>8.8</td><td>40.4</td><td>16.3</td><td>23.7</td><td>15.4</td><td>58.9</td><td>11.3</td><td>53.2</td><td>11.8</td></tr><tr><td>Ensemble</td><td>92.1</td><td>86.7</td><td>88.7</td><td>78.6</td><td>63.2</td><td>50.5</td><td>70.8</td><td>61.6</td><td>25.9</td><td>22.2</td><td>50.9</td><td>45.2</td><td>32.9</td><td>24.8</td><td>64.6</td><td>55.3</td><td>61.1</td><td>53.1</td></tr><tr><td>TPT</td><td>94.1</td><td>77.2</td><td>87.4</td><td>66.9</td><td>66.5</td><td>43.2</td><td>66.1</td><td>49.9</td><td>23.4</td><td>16.1</td><td>45.9</td><td>30.7</td><td>42.6</td><td>13.5</td><td>67.9</td><td>44.1</td><td>61.7</td><td>42.7</td></tr><tr><td>C-TPT</td><td>93.9</td><td>76.0</td><td>88.2</td><td>62.1</td><td>65.8</td><td>39.6</td><td>69.6</td><td>47.5</td><td>23.9</td><td>15.1</td><td>45.9</td><td>30.2</td><td>42.3</td><td>11.5</td><td>65.6</td><td>40.3</td><td>61.9</td><td>40.2</td></tr><tr><td>MTA</td><td>94.3</td><td>79.2</td><td>88.0</td><td>67.8</td><td>67.7</td><td>43.4</td><td>65.0</td><td>48.4</td><td>24.0</td><td>16.1</td><td>46.5</td><td>30.6</td><td>42.5</td><td>19.3</td><td>67.5</td><td>44.6</td><td>61.9</td><td>43.6</td></tr><tr><td>R-TPT</td><td>93.7</td><td>88.1</td><td>87.2</td><td>74.4</td><td>67.0</td><td>50.7</td><td>68.7</td><td>55.7</td><td>23.9</td><td>20.1</td><td>46.4</td><td>39.6</td><td>34.7</td><td>24.8</td><td>67.2</td><td>56.2</td><td>61.1</td><td>51.2</td></tr><tr><td>RITA</td><td>93.8</td><td>87.8</td><td>89.8</td><td>78.7</td><td>64.2</td><td>52.6</td><td>71.6</td><td>62.0</td><td>26.2</td><td>22.9</td><td>51.5</td><td>46.2</td><td>33.4</td><td>25.2</td><td>65.5</td><td>56.8</td><td>62.0</td><td>54.0</td></tr><tr><td rowspan="7">DI</td><td>CLIP</td><td>85.9</td><td>23.0</td><td>83.5</td><td>4.4</td><td>55.7</td><td>0.6</td><td>61.7</td><td>1.9</td><td>15.7</td><td>0.0</td><td>40.4</td><td>6.1</td><td>23.7</td><td>0.0</td><td>58.9</td><td>3.2</td><td>53.2</td><td>4.9</td></tr><tr><td>Ensemble</td><td>92.1</td><td>84.3</td><td>88.7</td><td>69.1</td><td>63.2</td><td>39.1</td><td>70.8</td><td>52.7</td><td>25.9</td><td>15.3</td><td>50.9</td><td>39.7</td><td>32.9</td><td>17.6</td><td>64.6</td><td>47.6</td><td>61.1</td><td>45.7</td></tr><tr><td>TPT</td><td>94.1</td><td>80.7</td><td>87.4</td><td>65.2</td><td>66.5</td><td>38.3</td><td>66.1</td><td>49.7</td><td>23.4</td><td>13.5</td><td>45.9</td><td>30.3</td><td>42.6</td><td>7.4</td><td>67.9</td><td>40.5</td><td>61.7</td><td>40.7</td></tr><tr><td>C-TPT</td><td>93.9</td><td>79.5</td><td>88.2</td><td>59.5</td><td>65.8</td><td>34.2</td><td>69.6</td><td>47.3</td><td>23.9</td><td>11.6</td><td>45.9</td><td>29.1</td><td>42.3</td><td>7.4</td><td>65.6</td><td>37.0</td><td>61.9</td><td>38.2</td></tr><tr><td>MTA</td><td>94.3</td><td>82.6</td><td>88.0</td><td>65.6</td><td>67.7</td><td>39.5</td><td>65.0</td><td>48.0</td><td>24.0</td><td>13.5</td><td>46.5</td><td>30.9</td><td>42.5</td><td>14.7</td><td>67.5</td><td>41.9</td><td>61.9</td><td>42.1</td></tr><tr><td>R-TPT</td><td>93.7</td><td>84.9</td><td>87.2</td><td>66.7</td><td>67.0</td><td>39.1</td><td>68.7</td><td>48.1</td><td>23.9</td><td>14.1</td><td>46.4</td><td>35.6</td><td>34.7</td><td>18.0</td><td>67.2</td><td>47.1</td><td>61.1</td><td>44.2</td></tr><tr><td>RITA</td><td>93.8</td><td>84.7</td><td>89.8</td><td>69.4</td><td>64.2</td><td>39.9</td><td>71.6</td><td>53.4</td><td>26.2</td><td>15.9</td><td>51.5</td><td>39.8</td><td>33.4</td><td>18.4</td><td>65.5</td><td>47.7</td><td>62.0</td><td>46.2</td></tr></table>

![](images/e71aada1442e12e63014f1a05ca8f07636f1fae4c468847b5809c6990dfb49fe.jpg)  
(a) Prompt Number

![](images/c54ab29eb990775da40bf1de69747858ad3b43be5bde5e163c45f7f0b4d508f4.jpg)  
(b) Augmentation View  
Figure 3. Ablation studies of key hyperparameters in DTD dataset using ViT-B/16 with ϵ = 1.0. (a) Performance variation with respect to the number of learnable prompts. (b) Impact of the number of augmentation views during inference.

## 4.3. Ablation Study

Number of learnable prompts and augmented views. As illustrated in Figure 3, we conduct sensitivity analyses on the DTD dataset regarding the number of learnable prompts and augmentation views. Figure 3(a) shows that increasing the number of learnable prompts from 4 to 32 leads to steady performance gains across both settings. Figure 3(b) assesses the sensitivity of augmentation view scales during inference, where adversarial accuracy peaks at 96 views. Notably, scaling to 128 views results in a slight performance decline, a trend consistently observed in noise-free environments as well. Consequently, selecting a moderate number of augmentation views enables a superior trade-off between efficiency and accuracy.

![](images/c816e8e2e15982c2aae7a769c4180a70e2fa9854adce99a1a714892ffe8f8b5f.jpg)  
(a) DTD

![](images/ef79dff2f7a9b546ce978435799fef1725422153a7f4b598bf7459fe53829370.jpg)  
(b) ImageNet  
Figure 4. Sensitivity analysis of the cache integration coefficient α. Classification accuracy (%) on (a) DTD and (b) ImageNet is evaluated across α values using ViT-B/16 with ϵ = 1.0.

Contribution coefficient of dynamic cache. We investigate the sensitivity of RITA to the integration coefficient α during inference on the DTD and ImageNet datasets, as shown in Figure 4. Experimental results confirm that the cache mechanism effectively boosts performance. On both datasets, increasing α from 0 to 0.1 yields a significant accuracy gain, followed by a slight upward trend that gradually stabilizes as α further scales. These results indicate that the visual priors stored in the cache not only play a crucial role in correcting predictions under adversarial settings, but also offer effective semantic complementarity during the inference stage. Ablation study of the dynamic cache is provided in Appendix D.1.

Different perturbation budgets. To assess the resilience of RITA under various attack intensities, we conduct ablation studies on all fine-grained datasets and ImageNet dataset, with the results illustrated in Figure 5. As the perturbation budget ϵ increases from 1 to 4, the accuracy declines as expected, indicating the generation of more potent perturbations. Simultaneously, we observe that increasing the number of TTA steps generally yields additional and consistent defensive gains across evaluations.

![](images/d70fa64eae0003bcb951f4219a682e4f2c080b497ab019e1be67ab5b234008bc.jpg)  
(a) Fine-grained datasets

![](images/52e95ec732fcdc84b4b5d201e621ac3c30bfa845d663eba605bd0e322b9e79fd.jpg)  
(b) ImageNet

Figure 5. Adversarial robustness (%) under varying perturbation budgets and TTA steps on (a) fine-grained datasets and (b) ImageNet. Robust accuracy is evaluated using ViT-B/16 under $\epsilon \in \lbrace 1 . 0 , 2 . 0 , 4 . 0 \rbrace$ and TTA steps $\in \{ 1 , 2 , 4 \}$ . +x.x indicates the increment relative to the case where TTA step = 1.  
![](images/f3d4fa607ee196a7152647c635ef2cf513f7702d8dff185f686cc9149ead900a.jpg)  
(a) Fine-grained datasets

![](images/adcd72e55cab078f6fbfcb814232f16856fb0e45cb3a337633e30b5f52004fd7.jpg)  
(b) ImageNet  
Figure 6. Evolution of adversarial robustness with respect to TTA steps across model architectures, where (a) presents the average robust accuracy over fine-grained datasets and (b) displays results on ImageNet. TTA step = 0 denotes the CLIP baseline.

Number of TTA steps. We report the average robust accuracy across three different backbones on all fine-grained datasets as shown in Figure 6(a) and the ImageNet dataset as illustrated in Figure 6(b). The experimental results indicate that all models reach their performance peak at step 2, while the accuracy slightly declines when the steps are increased to 4 across all evaluated datasets. This observation justifies our choice of a small number of iterations, e.g., TTA step = 1, which simultaneously ensures superior robustness and high computational efficiency for real-time inference.

Inference efficiency analysis. Table 4 presents the average comparison of running time, clean accuracy, and robust accuracy across various methods on fine-grained datasets. The empirical results indicate that RITA achieves superior classification performance while maintaining highly competitive inference efficiency compared with existing adaptation methods. Compared to MTA, which exhibits the fastest inference speed, RITA maintains a substantial lead of 11.8% in robust accuracy. These observations demonstrate that RITA strikes an excellent balance between computational cost and model robustness, delivering more resilient and precise inference results with minimal additional time latency.

Semantic alignment analysis. As shown in Figure 7, we visualize semantic alignment quality via KDE curves of the

Table 4. Comparison of running time on fine-grained datasets using ViT-B/16 with ϵ = 1.0.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Running Time</td><td colspan="2">Accuracy</td></tr><tr><td>Clean</td><td>Robust</td></tr><tr><td>TPT</td><td>1.52s/image</td><td>61.7</td><td>39.3</td></tr><tr><td>C-TPT</td><td>1.64s/image</td><td>61.9</td><td>35.8</td></tr><tr><td>MTA</td><td>1.20s/image</td><td>61.9</td><td>40.9</td></tr><tr><td>R-TPT</td><td>1.70s/image</td><td>61.1</td><td>50.5</td></tr><tr><td>RITA</td><td>1.76s/image</td><td>62.0</td><td>52.7</td></tr></table>

![](images/eaa5b291786a7610bb8dc5d3e373fce6db6f6e494f741cca98ab4630ed5725c8.jpg)

![](images/ed42efef51591924bb418ba574e13235dc1569a9b6c6bc1b4190edb4c6656da1.jpg)  
(a) DTD  
(b) Caltech101  
Figure 7. Comparison of KL divergence distributions for semantic alignment between original visual features and augmented features from the cache mechanism. Kernel Density Estimation (KDE) curves are presented for (a) DTD and (b) Caltech101 datasets. Lower KL values signify more deterministic vision-text alignment.

Kullback-Leibler (KL) divergence. Specifically, we measure the divergence between the class-conditional distributions over text prototypes and the ideal one-hot targets. On both DTD and Caltech101 datasets, the curves for augmented features exhibit a pronounced leftward shift and higher concentration in the low-value region compared to the original features. This demonstrates that our cache mechanism effectively rectifies adversarial semantic biases, establishing more deterministic vision-text associations. Extended analysis is provided in Appendix D.2.

## Conclusion & Limitations

In this work, we propose RITA, a robust test-time prompt adaptation framework that enhances the adversarial robustness of VLMs without requiring retraining or access to labeled data. By modeling augmented visual features and prompt-induced textual prototypes as distributions and aligning them via optimal transport, RITA corrects cross-modal semantic misalignment caused by adversarial perturbations. Furthermore, a dynamic cache mechanism progressively aggregates reliable semantic cues from the test stream to refine alignment online. Extensive experiments across diverse benchmarks, attack types, and model backbones demonstrate that RITA consistently improves adversarial robustness while preserving competitive performance on clean data. Our results suggest that distribution-level alignment is a principled and effective paradigm for robust inference in large pre-trained VLMs.

While RITA demonstrates strong image classification performance, its extension to generative tasks like image captioning remains for future exploration. We believe our distribution-level alignment provides a foundation for adaptation in these broader multimodal scenarios.

## Impact Statement

This work improves the reliability and adversarial robustness of pre-trained VLMs, which is important for their deployment in safety-sensitive applications. The proposed method operates entirely at test time, without modifying model parameters or requiring additional training data, making it lightweight and easy to integrate into existing systems. Nevertheless, it does not eliminate all security risks, and adversarial attacks may continue to evolve. Future work should further investigate robust inference-time defenses and evaluate potential failure modes and misuse risks.

## Acknowledgement

This research is supported by the National Natural Science Foundation of China (No. 62576330) and the National Natural Science Foundation of Anhui (No.2508085MF143).

## References

Alayrac, J., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J. L., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, A., and Simonyan, K. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.

Altschuler, J. M., Weed, J., and Rigollet, P. Near-linear time approximation algorithms for optimal transport via sinkhorn iteration. In NIPS, pp. 1964–1974, 2017.

Arjovsky, M., Chintala, S., and Bottou, L. Wasserstein generative adversarial networks. In ICML, 2017.

Carlini, N. and Wagner, D. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security and Privacy, 2017a.

Carlini, N. and Wagner, D. A. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security and Privacy, pp. 39–57. IEEE Computer Society, 2017b.

Chen, G., Yao, W., Song, X., Li, X., Rao, Y., and Zhang, K. PLOT: prompt learning with optimal transport for visionlanguage models. In ICLR. OpenReview.net, 2023.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In CVPR, 2014.

Courty, N., Flamary, R., Tuia, D., and Rakotomamonjy, A. Optimal transport for domain adaptation. TPAMI, 2016.

Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. In NIPS, pp. 2292–2300, 2013.

Damodaran, B. B., Kellenberger, B., Flamary, R., Tuia, D., and Courty, N. Deepjdot: Deep joint distribution optimal transport for unsupervised domain adaptation. In ECCV (4), volume 11208 of Lecture Notes in Computer Science, pp. 467–483. Springer, 2018.

Deng, J., Dong, W., Socher, R., Li, L., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., and Li, J. Boosting adversarial attacks with momentum. In CVPR, pp. 9185–9193. IEEE Computer Society, 2018.

Fei-Fei, L., Fergus, R., and Perona, P. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In CVPR Workshops, 2004.

Feng, C., Yu, K., Liu, Y., Khan, S., and Zuo, W. Diverse data augmentation with diffusions for effective test-time prompt tuning. In ICCV, pp. 2704–2714. IEEE, 2023.

Feng, C., He, Y., Zou, J., Khan, S. H., Xiong, H., Li, Z., Zuo, W., Goh, R. S. M., and Liu, Y. Diffusion-enhanced test-time adaptation with text and image augmentation. Int. J. Comput. Vis., 133(8):5083–5098, 2025.

Guo, D., Tian, L., Zhao, H., Zhou, M., and Zha, H. Adaptive distribution calibration for few-shot learning with hierarchical optimal transport. In NeurIPS, 2022.

Helber, P., Bischke, B., Dengel, A., and Borth, D. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE J. Sel. Top. Appl. Earth Obs. Remote. Sens., 12(7):2217–2226, 2019.

Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., Song, D., Steinhardt, J., and Gilmer, J. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, 2021a.

Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., and Song, D. Natural adversarial examples. In CVPR, 2021b.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In ICCV Workshops, 2013.

Lazarou, M., Stathaki, T., and Avrithis, Y. Iterative label cleaning for transductive and semi-supervised few-shot learning. In ICCV, 2021.

Li, J., Li, D., Xiong, C., and Hoi, S. C. H. BLIP: bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In ICML, 2022.

Li, J., Li, D., Savarese, S., and Hoi, S. C. H. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, volume 202 of Proceedings of Machine Learning Research, pp. 19730–19742. PMLR, 2023.

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. Towards deep learning models resistant to adversarial attacks. In ICLR. OpenReview.net, 2018.

Maji, S., Rahtu, E., Kannala, J., Blaschko, M. B., and Vedaldi, A. Fine-grained visual classification of aircraft. CoRR, abs/1306.5151, 2013.

Mao, C., Geng, S., Yang, J., Wang, X., and Vondrick, C. Understanding zero-shot adversarial robustness for largescale models. In ICLR. OpenReview.net, 2023.

Mensch, A. and Peyre, G. Online sinkhorn: Optimal trans-´ port distances from sample streams. In NeurIPS, 2020.

Nilsback, M. and Zisserman, A. Automated flower classification over a large number of classes. In ICVGIP, 2008.

Ouali, Y., Bulat, A., Mart´ınez, B., and Tzimiropoulos, G. Black box few-shot adaptation for vision-language models. In ICCV, pp. 15488–15500. IEEE, 2023.

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. Cats and dogs. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI, USA, June 16-21, 2012, pp. 3498–3505. IEEE Computer Society, 2012. doi: 10.1109/CVPR.2012.6248092. URL https: //doi.org/10.1109/CVPR.2012.6248092.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 2021.

Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do imagenet classifiers generalize to imagenet? In ICML, 2019.

Ren, H., Tang, F., Zheng, H., Zhao, H., Guo, D., and Chang, Y. Modality-consistent prompt tuning with optimal transport. IEEE Trans. Circuits Syst. Video Technol., 35(3): 2499–2512, 2025.

Schlarmann, C., Singh, N. D., Croce, F., and Hein, M. Robust CLIP: unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. In ICML. OpenReview.net, 2024.

Sheng, L., Liang, J., Wang, Z., and He, R. R-TPT: improving adversarial robustness of vision-language models through test-time prompt tuning. In CVPR, pp. 29958– 29967. Computer Vision Foundation / IEEE, 2025.

Shu, M., Nie, W., Huang, D., Yu, Z., Goldstein, T., Anandkumar, A., and Xiao, C. Test-time prompt tuning for zero-shot generalization in vision-language models. In NeurIPS, 2022.

Soomro, K., Zamir, A. R., and Shah, M. UCF101: A dataset of 101 human actions classes from videos in the wild. CoRR, abs/1212.0402, 2012.

Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I. J., and Fergus, R. Intriguing properties of neural networks. In ICLR (Poster), 2014.

Wang, D., Shelhamer, E., Liu, S., Olshausen, B. A., and Darrell, T. Tent: Fully test-time adaptation by entropy minimization. In ICLR. OpenReview.net, 2021.

Wang, D., Li, M., Liu, X., Xu, M., Chen, B., and Zhang, H. Tuning multi-mode token-level prompt alignment across modalities. In NeurIPS, 2023.

Wang, H., Ge, S., Lipton, Z., and Xing, E. P. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems, pp. 10506–10518, 2019.

Wang, S., Zhang, J., Yuan, Z., and Shan, S. Pre-trained model guided fine-tuning for zero-shot adversarial robustness. In CVPR, pp. 24502–24511. IEEE, 2024.

Wang, X., Chen, K., Zhang, J., Chen, J., and Ma, X. TAPT: test-time adversarial prompt tuning for robust inference in vision-language models. In CVPR, pp. 19910–19920. Computer Vision Foundation / IEEE, 2025.

Xie, C., Zhang, Z., Zhou, Y., Bai, S., Wang, J., Ren, Z., and Yuille, A. L. Improving transferability of adversarial examples with input diversity. In CVPR, pp. 2730–2739. Computer Vision Foundation / IEEE, 2019.

Xu, H., Luo, D., and Carin, L. Scalable gromov-wasserstein learning for graph partitioning and matching. NeurIPS, 2019.

Yoon, H. S., Yoon, E., Tee, J. T. J., Hasegawa-Johnson, M. A., Li, Y., and Yoo, C. D. C-TPT: calibrated test-time prompt tuning for vision-language models via text feature dispersion. In ICLR. OpenReview.net, 2024.

Zanella, M. and Ayed, I. B. On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? In CVPR, pp. 23783–23793. IEEE, 2024.

Zhang, J., Yi, Q., and Sang, J. Towards adversarial attack on vision-language pre-training models. In ACM Multimedia, pp. 5005–5013. ACM, 2022a.

Zhang, J., Ma, X., Wang, X., Qiu, L., Wang, J., Jiang, Y., and Sang, J. Adversarial prompt tuning for vision-language models. In ECCV (45), volume 15103 of Lecture Notes in Computer Science, pp. 56–72. Springer, 2024.

Zhang, M., Levine, S., and Finn, C. MEMO: test time robustness via adaptation and augmentation. In NeurIPS, 2022b.

Zhao, S., Zhu, Q., Xiong, S., Ruan, S., Fan, Y., Duan, R., Guo, Q., and Wei, X. Enhancing adversarial robustness of vision language models via adversarial mixture prompt tuning. CoRR, abs/2505.17509, 2025.

Zhao, Y., Pang, T., Du, C., Yang, X., Li, C., Cheung, N., and Lin, M. On evaluating adversarial robustness of large vision-language models. In NeurIPS, 2023.

Zhu, X., Zhu, B., Tan, Y., Wang, S., Hao, Y., and Zhang, H. Enhancing zero-shot vision models by label-free prompt distribution learning and bias correcting. In NeurIPS, 2024a.

Zhu, X., Zhu, B., Tan, Y., Wang, S., Hao, Y., and Zhang, H. Selective vision-language subspace projection for fewshot CLIP. In ACM Multimedia, pp. 3848–3857. ACM, 2024b.

Zhu, X., Wang, S., Zhu, B., Li, M., Li, Y., Fang, J., Wang, Z., Wang, D., and Zhang, H. Dynamic multimodal prototype learning in vision-language models. In ICCV, pp. 2501– 2511. IEEE, 2025a.

Zhu, X., Zhu, B., Wang, S., Zhao, K., and Zhang, H. Enhancing CLIP robustness via cross-modality alignment. CoRR, abs/2510.24038, 2025b.

Zhu, X., Zhao, K., Yi, L., Wang, S., Wang, Z., Zhu, B., and Zhang, H. Look carefully: Adaptive visual reinforcements in multimodal large language models for hallucination mitigation. CoRR, abs/2602.24041, 2026a.

Zhu, X., Zhu, B., Fang, J., Wang, S., Zhang, Y., Wang, X., and He, X. Guardalign: Test-time safety alignment in multimodal large language models. CoRR, abs/2602.24027, 2026b.

Zhu, X., Zhu, B., Li, Y., Fang, J., Wang, S., Zhao, K., and Zhang, H. Hierarchical semantic alignment for image clustering. In AAAI, pp. 29177–29185. AAAI Press, 2026c.

Zhu, X., Zhu, B., Wang, S., Fang, J., Zhao, K., Zhang, H., and He, X. Principled steering via null-space projection for jailbreak defense in vision-language models. CoRR, abs/2603.22094, 2026d.

Zhu, Y., Ji, Y., Zhao, Z., Wu, G., and Wang, L. AWT: transferring vision-language models via augmentation, weighting, and transportation. In NeurIPS, 2024c.

## A. Proof of Theorem 1

Theorem 3.1 (Decomposition of Alignment Objective). The Optimal Transport objective $( \mathcal { L } _ { \mathrm { O T } } )$ imposes a stricter bound by decomposing into a centroid alignment term and a structural variance penalty:

$$
\underbrace {W _ {2} ^ {2} (\mathbb {P} _ {t} , \mathbb {Q} _ {k})} _ {\mathcal {L} _ {\mathrm{OT}} (R I T A)} \approx \underbrace {\| \pmb {\mu} _ {x} - \pmb {\mu} _ {z} \| ^ {2}} _ {\mathcal {L} _ {\mathrm{mean}} (T P T - e q u i v a l e n t)} + \underbrace {\mathfrak {B} ^ {2} (\pmb {\Sigma} _ {x} , \pmb {\Sigma} _ {z})} _ {\mathcal {R} _ {\mathrm{var}} (S t r u c t u r a l P e n a l t y)},\tag{13}
$$

where $\mathfrak { B } ^ { 2 } ( \mathbf { A } , \mathbf { B } ) = \mathrm { T r } ( \mathbf { A } + \mathbf { B } - 2 ( \mathbf { A } ^ { 1 / 2 } \mathbf { B } \mathbf { A } ^ { 1 / 2 } ) ^ { 1 / 2 } )$ is the Bures metric, quantifying the geometric mismatch between covariances.

Proof. Let $X \sim \mu$ and $Z \sim \nu$ be random vectors in $\mathbb { R } ^ { d }$ with mean vectors $\pmb { \mu } _ { x } , \pmb { \mu } _ { z }$ and covariance matrices $\Sigma _ { x } , \Sigma _ { z }$ respectively. The 2-Wasserstein distance is defined as the minimum expected transport cost over all valid joint couplings $\pi \in \Pi ( \mu , \nu )$

$$
W _ {2} ^ {2} (\mu , \nu) = \inf _ {\pi \in \Pi (\mu , \nu)} \mathbb {E} _ {(X, Z) \sim \pi} \big [ \| X - Z \| ^ {2} \big ].\tag{14}
$$

We first expand the squared Euclidean cost by centering the variables around their respective means. Let $\tilde { X } = X - \mu _ { x }$ and $\tilde { Z } = Z - \mu _ { z }$ . The cost function can be rewritten as:

$$
\begin{array}{r l} & {\| X - Z \| ^ {2} = \| (\tilde {X} - \tilde {Z}) + (\pmb {\mu} _ {x} - \pmb {\mu} _ {z}) \| ^ {2}} \\ & {\qquad = \| \pmb {\mu} _ {x} - \pmb {\mu} _ {z} \| ^ {2} + \| \tilde {X} \| ^ {2} + \| \tilde {Z} \| ^ {2} + 2 (\pmb {\mu} _ {x} - \pmb {\mu} _ {z}) ^ {\top} (\tilde {X} - \tilde {Z}) - 2 \tilde {X} ^ {\top} \tilde {Z}.} \end{array}\tag{15}
$$

Taking the expectation $\mathbb { E } _ { \pi } [ \cdot ]$ , the linear terms vanish because the variables are centered $( \mathrm { i . e . , } \mathbb { E } [ \tilde { X } ] = \mathbb { E } [ \tilde { Z } ] = 0 )$ . Utilizing the identity $\mathbb { E } [ \| \tilde { X } \| ^ { 2 } ] = \mathrm { T r } ( \mathbf { \bar { \Sigma } } _ { x } )$ , the expected cost simplifies to:

$$
\mathbb {E} _ {\boldsymbol \pi} \big [ \| X - Z \| ^ {2} \big ] = \| \pmb {\mu} _ {x} - \pmb {\mu} _ {z} \| ^ {2} + \mathrm{Tr} (\pmb {\Sigma} _ {x}) + \mathrm{Tr} (\pmb {\Sigma} _ {z}) - 2 \mathbb {E} _ {\boldsymbol \pi} \big [ \tilde {X} ^ {\top} \tilde {Z} \big ].\tag{16}
$$

To find $W _ { 2 } ^ { 2 }$ , we minimize Eq. (16) over the coupling π. Since the mean difference and trace terms are constants independent of $\pi ,$ , the problem reduces to maximizing the correlation term $\mathbb { E } _ { \pi } [ \tilde { X } ^ { \top } \tilde { Z } ]$ . For the family of elliptical distributions (e.g., Gaussians), it is a known result (Gelbrich, 1990) that the optimal coupling yields:

$$
\sup _ {\pi \in \Pi} \mathbb {E} _ {\pi} \big [ \tilde {X} ^ {\top} \tilde {Z} \big ] = \mathrm{Tr} \left((\pmb {\Sigma} _ {x} ^ {1 / 2} \pmb {\Sigma} _ {z} \pmb {\Sigma} _ {x} ^ {1 / 2}) ^ {1 / 2}\right).\tag{17}
$$

Substituting this optimal correlation back into Eq. (16), we obtain:

$$
\begin{array}{r l r} & & W _ {2} ^ {2} (\mu , \nu) = \| \pmb {\mu} _ {x} - \pmb {\mu} _ {z} \| ^ {2} + \mathrm{Tr} (\pmb {\Sigma} _ {x}) + \mathrm{Tr} (\pmb {\Sigma} _ {z}) - 2 \mathrm{Tr} \left((\pmb {\Sigma} _ {x} ^ {1 / 2} \pmb {\Sigma} _ {z} \pmb {\Sigma} _ {x} ^ {1 / 2}) ^ {1 / 2}\right) \\ & & = \underbrace {\| \pmb {\mu} _ {x} - \pmb {\mu} _ {z} \| ^ {2}} _ {\mathcal {L} _ {\mathrm{mean}}} + \underbrace {\mathrm{Tr} \left(\pmb {\Sigma} _ {x} + \pmb {\Sigma} _ {z} - 2 (\pmb {\Sigma} _ {x} ^ {1 / 2} \pmb {\Sigma} _ {z} \pmb {\Sigma} _ {x} ^ {1 / 2}) ^ {1 / 2}\right)} _ {\mathfrak {B} ^ {2} (\pmb {\Sigma} _ {x}, \pmb {\Sigma} _ {z})}. \end{array}\tag{18}
$$

The first term corresponds to the centroid distance $( { \mathcal { L } } _ { \mathrm { m e a n } } )$ , and the second term is the squared Bures metric $( { \mathfrak { B } } ^ { 2 } )$ representing the structural variance cost. For general distributions, this expression serves as a tight lower bound, confirming that minimizing $W _ { 2 } ^ { 2 }$ inherently constrains both the first-order (mean) and second-order (variance) geometric moments. □ □

Table 5. Results (%) of various adaptation methods on ImageNet and ImageNet-OOD datasets with ϵ = 1.0. OOD Avg. refers to the average results among four ImageNet-OOD datasets.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">ImageNet</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-V2</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-S</td><td colspan="2">OOD Avg.</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td rowspan="7">ViT-B/32</td><td>CLIP</td><td>62.0</td><td>0.7</td><td>29.5</td><td>0.1</td><td>54.7</td><td>1.5</td><td>66.2</td><td>6.9</td><td>40.8</td><td>4.5</td><td>47.8</td><td>3.2</td></tr><tr><td>Ensemble</td><td>64.4</td><td>52.3</td><td>34.1</td><td>21.4</td><td>58.1</td><td>45.6</td><td>64.5</td><td>55.2</td><td>39.2</td><td>31.2</td><td>49.0</td><td>38.3</td></tr><tr><td>TPT</td><td>63.6</td><td>36.6</td><td>34.5</td><td>9.3</td><td>56.9</td><td>30.4</td><td>69.1</td><td>49.4</td><td>41.6</td><td>30.4</td><td>50.5</td><td>31.2</td></tr><tr><td>C-TPT</td><td>63.5</td><td>33.4</td><td>30.5</td><td>7.6</td><td>55.9</td><td>27.4</td><td>67.0</td><td>45.0</td><td>41.8</td><td>30.1</td><td>48.8</td><td>28.7</td></tr><tr><td>MTA</td><td>64.9</td><td>40.1</td><td>37.7</td><td>11.1</td><td>58.3</td><td>33.2</td><td>70.3</td><td>52.3</td><td>43.4</td><td>31.5</td><td>52.4</td><td>32.0</td></tr><tr><td>R-TPT</td><td>64.4</td><td>52.1</td><td>36.9</td><td>21.9</td><td>58.0</td><td>45.5</td><td>67.5</td><td>55.8</td><td>41.7</td><td>31.2</td><td>51.0</td><td>38.6</td></tr><tr><td>RITA</td><td>64.8</td><td>52.7</td><td>35.4</td><td>22.5</td><td>58.5</td><td>45.9</td><td>65.7</td><td>56.3</td><td>40.8</td><td>32.3</td><td>50.1</td><td>39.3</td></tr><tr><td rowspan="7">ViT-B/16</td><td>CLIP</td><td>66.7</td><td>0.6</td><td>47.7</td><td>0.1</td><td>60.8</td><td>0.2</td><td>73.9</td><td>3.5</td><td>46.1</td><td>2.2</td><td>57.1</td><td>1.5</td></tr><tr><td>Ensemble</td><td>68.8</td><td>54.4</td><td>55.8</td><td>33.6</td><td>62.8</td><td>47.4</td><td>72.9</td><td>62.7</td><td>46.5</td><td>35.1</td><td>59.5</td><td>44.7</td></tr><tr><td>TPT</td><td>68.9</td><td>42.4</td><td>54.7</td><td>14.9</td><td>63.6</td><td>35.7</td><td>77.1</td><td>57.3</td><td>47.9</td><td>35.6</td><td>60.8</td><td>37.1</td></tr><tr><td>C-TPT</td><td>68.1</td><td>38.0</td><td>49.7</td><td>11.3</td><td>61.9</td><td>31.4</td><td>74.8</td><td>51.9</td><td>47.2</td><td>34.4</td><td>58.4</td><td>3.4</td></tr><tr><td>MTA</td><td>69.0</td><td>44.4</td><td>57.3</td><td>17.5</td><td>63.4</td><td>37.2</td><td>76.9</td><td>58.9</td><td>48.4</td><td>35.8</td><td>61.5</td><td>38.7</td></tr><tr><td>R-TPT</td><td>69.1</td><td>54.4</td><td>57.2</td><td>34.7</td><td>63.5</td><td>48.0</td><td>75.5</td><td>63.7</td><td>47.7</td><td>36.5</td><td>60.9</td><td>45.7</td></tr><tr><td>RITA</td><td>69.1</td><td>55.1</td><td>55.8</td><td>35.0</td><td>63.2</td><td>48.4</td><td>74.0</td><td>64.1</td><td>47.1</td><td>37.2</td><td>60.0</td><td>46.2</td></tr></table>

## B. Implementation Details

For fair comparison, all approaches use the same pre-trained CLIP backbone and identical AugMix-based test-time augmentations, without external data, foundation models, or additional knowledge. We report average accuracy on clean samples and adversarial accuracy under PGD attacks with different perturbation budgets for default. Adversarial examples are generated on the original CLIP model, reflecting a realistic threat model.

For most experiments, we adopt a default setting that utilizes class descriptors and sets the subset size |S| = 64. However, to accommodate the distinct characteristics of specific benchmarks, we adjust these parameters for several datasets as follows: 1) For EuroSAT, we disable descriptors for the ViT-B/16 backbone and set |S| = 32 for both ViT-B/16 and ViT-B/32; 2) For ImageNet-A and ImageNet-R, the size of subset |S| is 32; 3) For ImageNet-Sketch, descriptors are disabled, and the subset size |S| is 32. For all other datasets not explicitly mentioned, the default configuration is maintained. These refinements are designed to better align domain-specific semantic features with the test-time adaptation process.

## C. Additional Results

## C.1. Results on ImageNet and ImageNet-OOD datasets.

Table 5 presents the performance comparison across ImageNet and its four Out-of-Distribution (OOD) variants. The results demonstrate that RITA maintains a significant robustness advantage even when addressing large-scale general visual tasks. On the standard ImageNet, RITA achieves state-of-the-art or competitive accuracies across both architectures under both settings; for instance, it reaches 55.1% robustness on ViT-B/16, a substantial leap from the vanilla CLIP’s 0.6%. RITA’s performance is equally compelling on the more challenging OOD variants, where its average OOD robust accuracy reaches 39.3% and 46.2% in two backbones, outperforming other methods. These findings validate that RITA not only excels in fine-grained tasks but also provides robust protection against diverse adversarial threats in large-scale general scenarios and under various distribution shifts.

## C.2. Evaluation under enhanced adversarial attacks.

Table 6 presents the performance of various test-time adaptation methods across eight fine-grained datasets under a more stringent adversarial constraint, where ϵ is set to 4.0. The experimental results indicate that as the attack intensity increases, the accuracy of the vanilla CLIP drops nearly to zero, whereas RITA demonstrates exceptional interference resistance. Specifically, our method achieves state-of-the-art results across both backbones, reaching 32.9% and 35.0% in average robust accuracy, respectively. These findings further confirm that RITA maintains high classification precision while exhibiting remarkable predictive stability, thereby validating its substantial practical value in mitigating complex and intense adversaria risks in real-world scenarios.

Table 6. Results (%) of adaptation methods on fine-grained classification datasets with ϵ = 4.0.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">Caltech101</td><td colspan="2">Pets</td><td colspan="2">Cars</td><td colspan="2">Flower102</td><td colspan="2">Aircraft</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">UCF101</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td rowspan="7">ViT-B/32</td><td>CLIP</td><td>90.9</td><td>2.5</td><td>83.0</td><td>0.0</td><td>49.7</td><td>0.0</td><td>65.8</td><td>0.0</td><td>18.3</td><td>0.0</td><td>40.8</td><td>0.2</td><td>18.6</td><td>0.0</td><td>62.1</td><td>0.0</td><td>53.6</td><td>0.3</td></tr><tr><td>Ensemble</td><td>91.6</td><td>73.7</td><td>85.0</td><td>47.0</td><td>57.8</td><td>18.5</td><td>67.4</td><td>34.8</td><td>20.1</td><td>7.2</td><td>46.1</td><td>29.2</td><td>32.5</td><td>7.4</td><td>61.6</td><td>37.8</td><td>57.8</td><td>32.0</td></tr><tr><td>TPT</td><td>91.4</td><td>60.0</td><td>84.1</td><td>30.3</td><td>62.9</td><td>16.4</td><td>63.8</td><td>28.5</td><td>19.0</td><td>3.9</td><td>42.2</td><td>19.9</td><td>35.1</td><td>6.5</td><td>62.3</td><td>21.8</td><td>57.6</td><td>23.4</td></tr><tr><td>C-TPT</td><td>91.8</td><td>53.7</td><td>84.9</td><td>21.5</td><td>60.8</td><td>9.2</td><td>65.9</td><td>22.1</td><td>17.7</td><td>2.4</td><td>44.3</td><td>16.4</td><td>34.7</td><td>5.7</td><td>62.6</td><td>17.8</td><td>57.8</td><td>18.6</td></tr><tr><td>MTA</td><td>91.8</td><td>73.9</td><td>85.8</td><td>45.8</td><td>64.1</td><td>19.2</td><td>64.8</td><td>34.2</td><td>20.4</td><td>6.1</td><td>44.0</td><td>22.0</td><td>34.5</td><td>5.3</td><td>63.6</td><td>33.1</td><td>58.6</td><td>29.9</td></tr><tr><td>R-TPT</td><td>90.6</td><td>74.6</td><td>84.5</td><td>44.5</td><td>63.1</td><td>20.5</td><td>62.6</td><td>34.1</td><td>19.1</td><td>6.6</td><td>42.1</td><td>27.3</td><td>32.0</td><td>7.1</td><td>62.8</td><td>37.1</td><td>57.1</td><td>31.4</td></tr><tr><td>RITA</td><td>92.3</td><td>74.8</td><td>85.9</td><td>47.4</td><td>59.6</td><td>20.1</td><td>68.7</td><td>35.6</td><td>20.2</td><td>7.7</td><td>46.2</td><td>30.2</td><td>33.4</td><td>8.5</td><td>62.8</td><td>39.1</td><td>58.6</td><td>32.9</td></tr><tr><td rowspan="7">ViT-B/16</td><td>CLIP</td><td>85.9</td><td>0.7</td><td>83.5</td><td>0.0</td><td>55.7</td><td>0.0</td><td>61.7</td><td>0.0</td><td>15.7</td><td>0.0</td><td>40.4</td><td>0.0</td><td>23.7</td><td>0.0</td><td>58.9</td><td>0.0</td><td>53.2</td><td>0.1</td></tr><tr><td>Ensemble</td><td>92.1</td><td>76.8</td><td>88.7</td><td>47.9</td><td>63.2</td><td>22.4</td><td>70.8</td><td>37.6</td><td>25.9</td><td>10.2</td><td>50.9</td><td>33.2</td><td>32.9</td><td>6.6</td><td>64.6</td><td>35.6</td><td>61.1</td><td>33.8</td></tr><tr><td>TPT</td><td>94.1</td><td>60.6</td><td>87.4</td><td>31.0</td><td>66.5</td><td>13.8</td><td>66.1</td><td>23.7</td><td>23.4</td><td>4.4</td><td>45.9</td><td>17.4</td><td>42.6</td><td>4.6</td><td>67.9</td><td>20.3</td><td>61.7</td><td>21.9</td></tr><tr><td>C-TPT</td><td>93.9</td><td>49.6</td><td>88.2</td><td>21.1</td><td>65.8</td><td>9.2</td><td>69.6</td><td>17.2</td><td>23.9</td><td>2.0</td><td>45.9</td><td>12.7</td><td>42.3</td><td>5.2</td><td>65.6</td><td>14.2</td><td>61.9</td><td>16.4</td></tr><tr><td>MTA</td><td>94.3</td><td>73.6</td><td>88.0</td><td>51.2</td><td>67.7</td><td>25.7</td><td>65.0</td><td>31.7</td><td>24.0</td><td>7.4</td><td>46.5</td><td>21.5</td><td>42.5</td><td>6.5</td><td>67.5</td><td>30.9</td><td>61.9</td><td>31.0</td></tr><tr><td>R-TPT</td><td>93.7</td><td>78.3</td><td>87.2</td><td>45.6</td><td>67.0</td><td>23.9</td><td>68.7</td><td>34.8</td><td>23.9</td><td>10.5</td><td>46.4</td><td>30.4</td><td>34.7</td><td>6.3</td><td>67.2</td><td>35.2</td><td>61.1</td><td>33.1</td></tr><tr><td>RITA</td><td>93.8</td><td>78.5</td><td>89.8</td><td>48.1</td><td>64.2</td><td>24.2</td><td>71.6</td><td>38.4</td><td>26.2</td><td>11.3</td><td>51.5</td><td>34.6</td><td>33.4</td><td>7.9</td><td>65.5</td><td>37.2</td><td>62.0</td><td>35.0</td></tr></table>

Table 7. Results (%) of adaptation methods on fine-grained classification datasets using RN50 with ϵ set to 1.0.

<table><tr><td rowspan="2">Method</td><td colspan="2">Caltech101</td><td colspan="2">Pets</td><td colspan="2">Cars</td><td colspan="2">Flower102</td><td colspan="2">Aircraft</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">UCF101</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td>CLIP</td><td>84.9</td><td>2.6</td><td>83.5</td><td>0.0</td><td>53.7</td><td>0.0</td><td>61.7</td><td>0.0</td><td>14.7</td><td>0.0</td><td>40.4</td><td>0.8</td><td>18.7</td><td>0.0</td><td>57.5</td><td>0.0</td><td>51.8</td><td>0.4</td></tr><tr><td>Ensemble</td><td>84.6</td><td> $\underline{78.2}$ </td><td>85.1</td><td> $\underline{75.4}$ </td><td>52.6</td><td>38.4</td><td> $\underline{65.4}$ </td><td> $\underline{56.2}$ </td><td>15.7</td><td> $\underline{11.8}$ </td><td> $\underline{43.1}$ </td><td> $\underline{38.2}$ </td><td>22.0</td><td>14.6</td><td>56.3</td><td> $\underline{49.5}$ </td><td>53.1</td><td> $\underline{45.2}$ </td></tr><tr><td>TPT</td><td>85.1</td><td>7.0</td><td>84.7</td><td>0.1</td><td>54.4</td><td>0.0</td><td>62.1</td><td>0.0</td><td>15.3</td><td>5.2</td><td>42.4</td><td>4.3</td><td> $\underline{22.4}$ </td><td>0.0</td><td> $\underline{60.2}$ </td><td>0.3</td><td>53.3</td><td>2.1</td></tr><tr><td>C-TPT</td><td>84.8</td><td>3.7</td><td>83.6</td><td>0.0</td><td>55.6</td><td>0.0</td><td>64.8</td><td>0.0</td><td> $\underline{16.7}$ </td><td>7.2</td><td>41.5</td><td>1.3</td><td>22.2</td><td>0.0</td><td>60.1</td><td>0.1</td><td> $\underline{53.6}$ </td><td>1.5</td></tr><tr><td>MTA</td><td>85.3</td><td>65.9</td><td>84.8</td><td>59.8</td><td> $\underline{55.7}$ </td><td>17.8</td><td>61.0</td><td>31.5</td><td>15.9</td><td>10.3</td><td>40.3</td><td>18.8</td><td> $\underline{22.5}$ </td><td>1.6</td><td> $\underline{60.6}$ </td><td>31.3</td><td>53.2</td><td>29.6</td></tr><tr><td>R-TPT</td><td> $\underline{86.7}$ </td><td> $\underline{78.2}$ </td><td>84.6</td><td>74.2</td><td> $\underline{56.1}$ </td><td> $\underline{38.6}$ </td><td>60.6</td><td>51.9</td><td>16.4</td><td> $\underline{11.8}$ </td><td>41.3</td><td>33.5</td><td>21.2</td><td> $\underline{15.1}$ </td><td>59.5</td><td>49.2</td><td>53.3</td><td>43.0</td></tr><tr><td>RITA</td><td> $\underline{85.7}$ </td><td> $\underline{79.3}$ </td><td>86.3</td><td>77.4</td><td>54.2</td><td>39.9</td><td>66.2</td><td>56.9</td><td>16.9</td><td> $\underline{12.4}$ </td><td>43.8</td><td>39.5</td><td>20.2</td><td> $\underline{15.9}$ </td><td>58.8</td><td> $\underline{50.0}$ </td><td>54.0</td><td>46.4</td></tr></table>

Table 8. Robust accuracy (%) on fine-grained classification datasets under different attacks using ViT-B/16 with ϵ = 1.0.

<table><tr><td></td><td>Method</td><td>Caltech101</td><td>Pets</td><td>Cars</td><td>Flower102</td><td>Aircraft</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td><td>Avg.</td></tr><tr><td rowspan="3">adaptive</td><td>CLIP</td><td>30.4</td><td>4.9</td><td>7.2</td><td>3.2</td><td>0.2</td><td>2.4</td><td>0.0</td><td>0.8</td><td>6.1</td></tr><tr><td>R-TPT</td><td>87.3</td><td>76.3</td><td>55.3</td><td>60.8</td><td>17.4</td><td>35.3</td><td>20.5</td><td>52.2</td><td>50.6</td></tr><tr><td>RITA</td><td>88.2</td><td>79.5</td><td>57.8</td><td>66.2</td><td>19.1</td><td>37.6</td><td>23.8</td><td>54.8</td><td>53.4</td></tr><tr><td rowspan="3">AA</td><td>CLIP</td><td>13.2</td><td>4.9</td><td>0.3</td><td>2.6</td><td>0.0</td><td>0.0</td><td>0.0</td><td>4.6</td><td>3.2</td></tr><tr><td>R-TPT</td><td>87.9</td><td>78.0</td><td>51.4</td><td>59.4</td><td>20.3</td><td>41.5</td><td>24.2</td><td>58.2</td><td>52.6</td></tr><tr><td>RITA</td><td>89.8</td><td>82.0</td><td>53.7</td><td>61.8</td><td>22.6</td><td>47.7</td><td>26.0</td><td>59.9</td><td>55.4</td></tr><tr><td rowspan="3">FGSM</td><td>CLIP</td><td>6.2</td><td>2.4</td><td>0.5</td><td>0.0</td><td>0.0</td><td>0.4</td><td>0.0</td><td>1.8</td><td>1.4</td></tr><tr><td>R-TPT</td><td>84.8</td><td>73.6</td><td>43.6</td><td>54.3</td><td>19.9</td><td>36.2</td><td>23.1</td><td>50.3</td><td>48.2</td></tr><tr><td>RITA</td><td>85.9</td><td>74.5</td><td>44.2</td><td>59.4</td><td>21.7</td><td>42.1</td><td>24.4</td><td>51.8</td><td>50.5</td></tr></table>

## C.3. Analysis on an alternative CLIP backbone.

In Table 7, we further evaluate the performance of RITA using RN50 as the vision backbone to verify its generalizability across different architectures. The experimental results demonstrate that RITA exhibits superior robustness across all eight fine-grained classification datasets. Notably, despite RN50 having a relatively weaker baseline representation capability compared to the ViT series, RITA consistently outperforms other adaptation methods, while maintaining high clean accuracy. These findings provide strong evidence that our method delivers cross-architecture robustness gains and effectively mitigates the vulnerability of diverse vision encoders to adversarial attacks.

## C.4. Robustness evaluation under other attacks.

We conduct experiments under various adversarial attack protocols, including adaptive attack that is aware of augmentation strategies, AutoAttack (AA), and FGSM. As reported in Table 8, RITA consistently achieves the highest robust accuracy across all eight datasets and all attack types compared to the CLIP and R-TPT baselines. Specifically, under the more rigorous AutoAttack, RITA maintains an average robust accuracy of 55.4%, outperforming R-TPT by 2.8%. Notably, RITA shows significant gains on challenging datasets like DTD and EuroSAT across all attack settings. These results demonstrate that the defensive capability of RITA is not tailored to a specific attack but generalizes well to diverse adversarial threats, confirming its potential for securing VLMs in various hostile environments.

Table 9. Classification accuracy (%) on 10 datasets using EVA-CLIP and OpenCLIP backbones.

<table><tr><td rowspan="2">Method</td><td colspan="2">Caltech101</td><td colspan="2">Pets</td><td colspan="2">Cars</td><td colspan="2">Flower102</td><td colspan="2">Aircraft</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">UCF101</td><td colspan="2">SUN397</td><td colspan="2">Food101</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td>OpenCLIP</td><td>91.3</td><td>12.3</td><td>89.2</td><td>1.2</td><td>75.7</td><td>2.9</td><td>66.9</td><td>0.2</td><td>17.7</td><td>0.0</td><td>51.3</td><td>3.1</td><td>50.1</td><td>0.4</td><td>67.3</td><td>0.1</td><td>69.6</td><td>1.9</td><td>85.9</td><td>1.4</td><td>66.5</td><td>2.4</td></tr><tr><td>+ RITA</td><td>92.4</td><td>89.9</td><td>91.7</td><td>78.2</td><td>78.4</td><td>50.2</td><td>73.2</td><td>62.4</td><td>25.1</td><td>19.3</td><td>55.9</td><td>46.3</td><td>51.5</td><td>29.9</td><td>69.6</td><td>53.5</td><td>75.2</td><td>54.0</td><td>88.3</td><td>60.7</td><td>70.1</td><td>54.4</td></tr><tr><td>EVA-CLIP</td><td>86.3</td><td>5.2</td><td>92.2</td><td>0.3</td><td>78.6</td><td>4.5</td><td>75.9</td><td>1.2</td><td>24.8</td><td>0.0</td><td>53.1</td><td>1.7</td><td>67.0</td><td>0.5</td><td>63.2</td><td>0.0</td><td>79.7</td><td>4.2</td><td>89.4</td><td>0.9</td><td>71.0</td><td>1.6</td></tr><tr><td>+ RITA</td><td>87.1</td><td>84.6</td><td>93.1</td><td>80.4</td><td>79.5</td><td>48.2</td><td>78.6</td><td>64.8</td><td>27.5</td><td>17.5</td><td>56.2</td><td>41.4</td><td>69.8</td><td>38.2</td><td>64.9</td><td>50.8</td><td>82.3</td><td>59.2</td><td>91.2</td><td>62.4</td><td>73.0</td><td>54.8</td></tr></table>

![](images/2f7f43cba4ccb515e765ba74cfc0ce889d0045c7d3ebea9f7a7c9131db8be4b8.jpg)  
(a) DTD

![](images/15c051bc49189c5e76777771891fc0dbdaa98d21dd08a70b7d947f9ea47a6084.jpg)  
(b) Caltech101  
Figure 8. KL divergence per class on (a) DTD and (b) Caltech101. Lower KL values signify superior vision-text alignment.

## C.5. Generalization to Other VLM Backbones.

To further validate the architectural agnosticity of RITA, we extend our evaluation to other VLMs, including OpenCLIP and EVA-CLIP. As shown in Table 9, we expanded RITA to 10 datasets by incorporating SUN397 and Food101. The results consistently demonstrate that RITA significantly boosts adversarial robustness across all architectures. For instance, when applied to OpenCLIP, RITA improves the average robust accuracy from 2.4% to 54.4%. On the newly added SUN397 and Food101 datasets, RITA achieves substantial gains, reaching robust accuracies of 54.0% and 60.7% for OpenCLIP, and 59.2% and 62.4% for EVA-CLIP, respectively. These findings underscore that the robustness gains of RITA are consistent across diverse VLM backbones and broader dataset distributions, reinforcing its effectiveness as a general test-time adaptation framework.

## D. Verification of the importance of the cache mechanism

## D.1. Ablation study of the dynamic cache.

In Table 10, we conduct an ablation study to specifically validate the significance of the cache mechanism $d _ { c O T }$ within the RITA framework. The results indicate that while utilizing the cache mechanism in isolation yields lower standard accuracy (39.7% on fine-grained and 20.2% on ImageNet) due to the absence of real-time alignment, it consistently outperforms the vanilla CLIP baseline in robustness metrics, achieving 27.1% and 29.6% respectively, compared to $\mathrm { C L I P } \mathrm { s }$ 4.7% and 3.2%. This evidence underscores that the historical priors preserved in the cache serve as an essential reference for stabilizing predictions under adversarial perturbations. Most importantly, the synergy between the cache mechanism and the optimal transport module $d _ { O T }$ leads to peak performance across all metrics, notably boosting fine-grained robustness from 49.4% to 55.0%. This further demonstrates that the cache provides critical semantic supplementation to the multimodal alignment, establishing a more robust and historically-aware inference framework.

Table 10. Main component analysis (%) on fine-grained datasets and ImageNet dataset using ViT-B/32 with $\epsilon = 1 . 0 .$

<table><tr><td rowspan="2"> $d_{\text{OT}}$ </td><td rowspan="2"> $d_{\text{cOT}}$ </td><td colspan="2">Fine-grained</td><td colspan="2">ImageNet</td></tr><tr><td>Acc.</td><td>Rob.</td><td>Acc.</td><td>Rob.</td></tr><tr><td>✗</td><td>✗</td><td>53.6</td><td>4.7</td><td>47.8</td><td>3.2</td></tr><tr><td>√</td><td>✗</td><td>57.0</td><td>49.4</td><td>63.9</td><td>52.3</td></tr><tr><td>✗</td><td>√</td><td>39.7</td><td>27.1</td><td>20.2</td><td>29.6</td></tr><tr><td>√</td><td>√</td><td>57.9</td><td>55.0</td><td>64.8</td><td>52.7</td></tr></table>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 RITA: Robust test-tIme promptT Adaptation
1: Input: Test stream $\{\hat{x}_t\}$, encoders $\Phi_{\mathrm{img}}$, $\Phi_{\mathrm{text}}$, text prompts $\{z_k^{(m)}\}_{m=1}^M$, entropy threshold $\gamma$, max cache size per class $N_k$, cache weight $\alpha$.
2: Initialize: Empty cache $\{\hat{X}_k\}_{k=1}^K \leftarrow \emptyset$; extract text features $\mathbf{z}_k^m = \Phi_{\mathrm{text}}(z_k^{(m)})$ and construct text distributions $\mathbb{Q}_k = \frac{1}{M} \sum_{m=1}^M \delta_{\mathbf{z}_k^m}$.
3: for each test image $\hat{x}_t$ do
4: Generate $N$ augmented views $\{\hat{x}_t^n\}_{n=1}^N$, extract visual features $\mathbf{x}_t^n = \Phi_{\mathrm{img}}(\hat{x}_t^n)$, and construct visual distribution $\mathbb{P}_t = \frac{1}{N} \sum_{n=1}^N \delta_{\mathbf{x}_t^n}$.
5: Cache Update: For each view $\hat{x}_t^n$ with entropy $H(p_t^n) &lt; \gamma$, get pseudo-label $\hat{k} = \underset{k}{\operatorname{argmax}} p_t^n(k)$. Add $\mathbf{x}_t^n$ to $\hat{X}_{\hat{k}}$ (if $|\hat{X}_{\hat{k}}| \geq N_k$ and $H(p_t^n)$ is lower, replace the max-entropy sample).
6: for class $k = 1$ to $K$ do
7: Compute global OT distance $d_{OT}(\mathbb{P}_t, \mathbb{Q}_k; C_{t,k})$ where $C_{t,k}(n,m) = 1 - \cos(\mathbf{x}_t^n, \mathbf{z}_k^m)$.
8: If $\hat{X}_k \neq \emptyset$, align features $\tilde{X}_k = \hat{X}_k W_k^*$ to build cache dist $\tilde{\mathbb{Q}}_k$, and calculate cache OT distance $d_{cOT}(\mathbb{P}_t, \tilde{\mathbb{Q}}_k; \tilde{C}_{t,k})$
9: Else $d_{cOT} = 0$.
10: end for
11: Output: Predicted label $\hat{y} = \underset{k \in [K]}{\operatorname{argmin}} (d_{OT}(\mathbb{P}_t, \mathbb{Q}_k; C_{t,k}) + \alpha d_{cOT}(\mathbb{P}_t, \tilde{\mathbb{Q}}_k; \tilde{C}_{t,k}))$.
12: end for
</div>

## D.2. Semantic alignment analysis of the cache mechanism.

To demonstrate the effectiveness of the cache mechanism, we extract the augmented view features from the DTD and Caltech101 datasets and compare them with the original unaugmented visual features from a ”vision-text” alignment perspective. Specifically, for each class, we first compute the class-conditional assignment distribution over all text prototypes, which reflects how the visual features of each class are semantically aligned with the textual features. Subsequently, we measure the KL divergence between this empirical distribution and an ideal one-hot target distribution that assigns all probability mass to the ground-truth class. A lower KL value indicates stronger vision-text alignment.

As illustrated in Figure 8, we present the scatter plots of KL divergence for all classes in DTD and Caltech101. Each point represents the vision-text alignment quality of a specific category, where red dots denote original adversarial views and green dots represent selected augmented views utilized by our cache mechanism. The KL divergence of original views exhibits high variance and remains at an elevated level, reflecting severe adversarial bias. Upon introducing the cache mechanism, the green dots show both a downward shift and reduced dispersion. This indicates that the augmented views effectively calibrate the corrupted feature representations by aggregating historical priors, pulling the model closer to the ideal one-hot distribution. Even for categories where the original KL divergence is particularly high, the augmented features still achieve significant alignment gains. This cross-category consistency provides a granular foundation for the superior robustness of the RITA framework when encountering diverse adversarial attacks.

## E. Algorithm for RITA

Algorithm 1 summarizes the RITA framework. RITA extracts features from augmented views, aligns visual-textual distributions via optimal transport, and maintains a dynamic cache for progressive refinement.