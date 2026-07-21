# MDND: Unsupervised Learning Guided by Non-Differentiable Refinement for Shape Correspondence

Qinsong Li<sup>1</sup>\*, Jing Meng<sup>2</sup>\*, Haibo Wang<sup>2</sup>, Shengjun Liu<sup>2†</sup>

<sup>1</sup>Big Data Institute, Central South University, Changsha, Hunan 410083, China <sup>2</sup>School of Mathematics and Statistics, Central South University, Changsha, Hunan 410083, China qinsli.cg@csu.edu.cn, wykqhmj1112@163.com, wang haibo2017@163.com, shjliu.cg@csu.edu.cn

## Abstract

Deep functional map frameworks (DFM) for shape correspondence are powerful, yet fundamentally limited by their reliance on end-to-end differentiability. This constraint prevents the integration of highly accurate, non-differentiable refinement techniques, capping their overall performance, especially on challenging non-isometric shapes. To overcome this, we introduce MDND, a novel DFM paradigm built on the principle of merging differentiable and non-differentiable components. Our framework facilitates unsupervised learning guided by an internal, non-differentiable refinement. Specifically, MDND employs a dual-branch architecture: a nondifferentiable refinement branch leverages a novel, multiscale iterative solver to produce highly robust correspondences, acting as a refined target. Concurrently, a fully differentiable branch learns to predict correspondences from features. The entire system is trained end-to-end without supervision by enforcing a consistency loss that compels the differentiable branch to learn from the superior, refined results of the non-differentiable branch. Extensive experiments show that MDND sets a new state-of-the-art, demonstrating remarkable robustness on shapes with non-isometric deformations and topological noise.

Code — https://github.com/AMAWDBAC/MDND

## Introduction

Establishing correspondences between non-rigid 3D shapes is a fundamental problem in computer vision and graphics, with broad applications in texture transfer (Dinh, Yezzi, and Turk 2005), shape interpolation (Ezuz, Solomon, and Ben-Chen 2019), statistical shape analysis (Bogo et al. 2014), and animation (Sumner and Popovic 2004). Despite decades of ´ research, the problem remains highly challenging, especially when shapes undergo severe non-isometric deformations or contain substantial topological noise.

The functional map framework has emerged as a dominant paradigm for non-rigid shape correspondence, revolutionizing the field by recasting the problem from matching points to matching functions between shapes (Ovsjanikov et al. 2012; Pai et al. 2021; Liu et al. 2022). This approach represents the correspondence compactly as a small matrix, the functional map, in a spectral basis typically composed of the eigenfunctions of the Laplace-Beltrami Operator (LBO). The advent of deep learning has given rise to deep functional map (DFM) pipelines (Litany et al. 2017; Donati, Sharma, and Ovsjanikov 2020; Donati, Corman, and Ovsjanikov 2022; Vigano, Ovsjanikov, and Melzi 2025), which\` learn optimized feature descriptors directly from data to improve matching accuracy. Over time, the field has converged on a powerful, yet highly structured, standard architecture. These pipelines typically employ a Siamese-style network to extract pointwise features, which are then fed into a differentiable layer to solve for the functional map matrix. A key innovation was the introduction of a second, parallel branch that promotes consistency between the functional map and a soft pointwise map derived from feature similarity (Cao, Roetzer, and Bernard 2023; Luo et al. 2025). This twobranch architecture, which enforces consistency between the functional and spatial domains, has become the de facto standard for achieving state-of-the-art results in both supervised and unsupervised settings.

![](images/1461468a96122302ab08a94073f02a95d54eaefbf9d27b63e06d9fcabee8222f.jpg)  
Figure 1: Qualitative comparison on challenging shapes. We visualize correspondences via texture transfer, comparing our method against state-of-the-art approaches like ULRSSM (Cao, Roetzer, and Bernard 2023) and HybridFMaps (Bastian et al. 2024). Our method produces noticeably more accurate and coherent maps in challenging scenarios involving non-isometric deformations (top rows) and significant topological noise (bottom row).

This convergence on a standard pipeline, however, exposes a fundamental and shared limitation: an implicit reliance on end-to-end differentiability. This constraint is a significant bottleneck, as it precludes the integration of powerful axiomatic refinement methods. These methods (Melzi et al. 2019; Eisenberger, Lahner, and Cremers 2020; Hu et al. 2021; Li et al. 2024), often formulated as iterative optimization procedures, can produce highly accurate correspondences from a reasonable initialization. However, they frequently incorporate complex, non-differentiable operations such as nearest-neighbor searches or discrete optimization steps (Ren et al. 2021), making them incompatible with gradient-based training. Recent efforts to create differentiable versions of these axiomatic algorithms have been met with significant trade-offs. Such approaches often require storing and differentiating through large, dense softcorrespondence matrices, leading to quadratic memory complexity that is infeasible for high-resolution meshes (Eisenberger et al. 2020; Li, Donati, and Ovsjanikov 2022; Hu et al. 2023). Furthermore, differentiating through the linear system solvers inherent to many DFM pipelines is known to be numerically unstable (Donati, Sharma, and Ovsjanikov 2020). This suggests a flawed premise in the current research trajectory: by forcing the axiomatic method to become differentiable, its original robustness and elegance are often compromised in favor of a fragile, inefficient, and incomplete approximation that fits the deep learning mold.

In this work, we challenge the necessity of end-to-end differentiability and propose a novel DFM paradigm, termed MDND (Merging Differentiable and Non-Differentiable components). Instead of forcing axiomatic methods into a differentiable framework, we leverage their full, uncompromised power by treating them as non-differentiable supervisory oracles. In our proposed framework, a deep network first predicts an initial correspondence. This map is then refined by a powerful, off-the-shelf axiomatic algorithm, which may contain non-differentiable steps. The resulting high-quality map from this oracle is then used as a pseudo-ground-truth target. A consistency loss between the network’s initial prediction and the oracle’s refined output is backpropagated through the feature-learning network. This process effectively teaches the network to generate features that produce better initializations—ones that the axiomatic method can readily refine to a high-accuracy solution, thereby dramatically improving learning efficiency and final matching precision. A powerful learning framework requires an equally powerful oracle. However, many efficient and popular axiomatic refiners, such as ZoomOut (Melzi et al. 2019) or MWP (Hu et al. 2021), are built upon the LBO eigenbasis. While the LBO basis is intrinsically defined and thus robust to isometries, it fundamentally struggles to characterize the high-frequency, extrinsic details like bending and creasing that define non-isometric deformations. This makes LBO-based refiners inherently ill-suited for the most challenging matching problems. Inspired by recent work showing that a hybrid basis (Bastian et al. 2024)—combining the intrinsic LBO basis with an extrinsic basis derived from an elastic thin-shell energy (ELA) (Hartwig et al. 2023)—is far more expressive for non-isometric shapes, we propose a new iterative refinement by generalizing the principles of state-of-the-art refiners MWP (Hu et al. 2021) to this more powerful hybrid basis. Our refiner is theoretically grounded, efficient, and highly robust to severe nonisometric deformations and topological artifacts. In summary, the main contributions of this work include:

• We propose MDND, the first approach to integrate nondifferentiable iterative refinement into the deep functional map framework, aiming to enhance feature learning and matching accuracy.

• We introduce an effective refinement with theoretical justification, and seamlessly incorporate it as a supervisory oracle within the MDND framework, significantly improving robustness in challenging scenarios.

• Extensive experiments across a wide range of challenging conditions demonstrate that our method sets a new state-of-the-art, particularly in cases involving substantial non-isometric deformations and topological noise.

## Related Work

Shape correspondence is a central topic in geometric processing, and we refer the reader to recent surveys for a comprehensive overview (Sahillioglu 2020; Deng et al. 2022).˘ Our work builds upon three key pillars of research: axiomatic functional maps, deep functional map methods, and the development of expressive spectral bases.

Axiomatic Functional Maps and Refinement. The functional map framework, introduced by Ovsjanikov et al. (Ovsjanikov et al. 2012), provides an elegant algebraic representation of correspondences. This foundational work sparked a wave of axiomatic (i.e., non-learning-based) methods aimed at improving map quality. These approaches typically focus on designing sophisticated energy functions to enforce desirable properties, such as orientation preservation (Ren et al. 2018; Donati et al. 2022), consistency with geometric wavelets (Hu et al. 2021; Liu et al. 2022), or multi-shape consistency (Huang et al. 2020; Gao, Zorah, and Bernard 2021). To optimize these energies, powerful iterative refinement strategies have become standard practice. Methods like ZoomOut (Melzi et al. 2019), MWP (Hu et al. 2021), and Smoothshells (Eisenberger, Lahner, and Cremers 2020) iteratively alternate between solving for a functional map and updating a pointwise correspondence. While these axiomatic techniques can achieve high accuracy, their performance is fundamentally limited by their reliance on hand-crafted feature descriptors (Aubry, Schlickewei, and Cremers 2011; Sun, Ovsjanikov, and Guibas 2009; Salti, Tombari, and Di Stefano 2014; Liu et al. 2024a) and spectral bases of LBO, which often fail in the presence of strong non-isometric deformations.

Deep Functional Maps. To mitigate the reliance on hand-crafted features, the field has shifted towards Deep Functional Maps (DFM). FMNet (Litany et al. 2017) was the first to learn feature descriptors for functional maps in a supervised manner. Unsupervised learning was subsequently introduced, using losses based on geodesic distances (Halimi et al. 2019) or structural properties of the functional map (Roufosse, Sharma, and Ovsjanikov 2019). The DFM pipeline has progressively matured with architectural innovations. GeomFmaps (Donati, Sharma, and Ovsjanikov 2020) introduced a differentiable regularized map solver, DUOFMNet (Donati, Corman, and Ovsjanikov 2022), which learned orientation-aware features using complex functional maps (Donati et al. 2022). AttentiveFMaps (Li, Donati, and Ovsjanikov 2022) employed spectral attention to handle varying resolutions. A significant breakthrough came with the introduction of dual-branch architectures (Cao, Roetzer, and Bernard 2023, 2024; Sun et al. 2023; Luo et al. 2025), which enforce consistency between the functional map domain and the pointwise spatial domain. However, a common thread unites these advanced pipelines: their complete reliance on end-to-end differentiability prevents them from incorporating the powerful, non-differentiable solvers developed in the axiomatic literature. Compared to the above works, which only optimize descriptors, recent research has attempted to use generative models to directly learn functional mappings simultaneously (Zhuravlev, Lahner, and Golyanik 2025; Emery et al.¨ 2025). However, this work requires training on large labeled datasets and is less effective than the former.

Intrinsic-Extrinsic Bases in Functional Maps. The functional map framework is built upon spectral bases. The eigenvectors of LBO are the conventional choice due to their intrinsic nature, which provides robustness to isometric deformations. However, this very property is a limitation in non-isometric settings, where crucial extrinsic information (e.g., bending and creases) is lost. To address this, recent work has explored more expressive bases. Hartwig et al. (Hartwig et al. 2023) introduced an extrinsic basis derived from the Hessian of a thin-shell elastic energy (the ELAbasis), which is highly sensitive to such fine-grained details. Building on this, HybridFMaps (Bastian et al. 2024) demonstrated that a hybrid spectral space combining the intrinsic LBO basis and the extrinsic ELA basis is significantly more expressive for non-isometric shapes. Our work is motivated by these advancements, and we leverage a hybrid basis to construct a refinement oracle that is robust to the challenging deformations where purely intrinsic methods fail.

## Background

In this section, we provide an overview of the background knowledge related to several key modules integral to our approach.

## Functional Map

Let $T : \mathcal { M } \to \mathcal { N }$ be a pointwise map from shape M to shape ${ \mathcal { N } } .$ The induced functional map $T _ { F } : \mathcal { L } ^ { \hat { 2 } } ( { \mathcal { N } } ) \to$ $\mathscr { L } ^ { 2 } ( \mathcal { M } )$ transforms square-integrable real-valued functions from $\dot { \mathcal { N } }$ to M. Specifically, for any function $f _ { \mathcal { N } } \in \mathcal { L } ^ { 2 } ( \mathcal { N } )$ the corresponding function $f _ { \mathcal { M } } \ \in \ \mathcal { L } ^ { 2 } ( \mathcal { M } )$ is defined by the composition $\mathsf { \bar { f } } _ { M } = \mathop { T _ { F } } ( f _ { N } ) = \mathop { f _ { N } } \circ T$ . Assuming that $\mathcal { L } ^ { 2 } ( \dot { \mathcal { M } } )$ and $\dot { \mathcal { L } } ^ { 2 } ( \mathcal { N } )$ are equipped with basis functions $\{ \phi _ { i } ^ { \mathcal { M } } \} _ { i \geq 1 }$ and $\{ \phi _ { j } ^ { N } \} _ { j \ge 1 }$ , respectively, the functional map can be represented as a matrix $\mathbf { C _ { \lambda } } = \mathbf { \lambda } \left( c _ { i j } \right)$ , where $c _ { i j } =$ $\langle T _ { F } ( \phi _ { j } ^ { N } ) , \phi _ { i } ^ { N } \rangle$ . Each element of C captures the relationship between the two sets of basis functions.

In the discrete setting, shapes M and $\mathcal { N }$ are typically represented as triangular meshes with m and n vertices, respectively. The pointwise map T is denoted by $\Pi _ { \mathcal { M N } } \in \mathbb { R } ^ { \bar { m } \times n }$ where $\Pi _ { \mathcal { M N } } ( i , j ) = 1 \mathrm { i } \hat { \mathrm { f } } T ( i ) = j .$ , and 0 otherwise. Here, i and j represent vertex indices on shape $\mathcal { M }$ and ${ \mathcal { N } } ,$ respectively. And we regard it as a proper pointwise map. Let $\dot { \Phi } _ { \mathcal { M } } ^ { \mathrm { L B O } } \in \mathbb { R } ^ { m \times k }$ and $\mathbf { \bar { \Phi } } _ { \mathcal { N } } ^ { \mathrm { L B O } } \in \mathbf { \bar { \mathbb { R } } } ^ { n \times k }$ denote the matrices containing the first k discretized Laplacian eigenfunctions for each shape. The functional map $\mathbf { \bar { C } } _ { \mathcal { N M } } ^ { \mathrm { L B O } }$ is given by the projection of $\Pi _ { \mathcal { M N } }$ onto the corresponding functional basis:

$$
\mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} = (\Phi_ {\mathcal {M}} ^ {\mathrm{LBO}}) ^ {\dagger} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} ^ {\mathrm{LBO}},\tag{1}
$$

where $\dagger$ denotes the Moore-Penrose pseudo-inverse. Since the functional map $\mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { { L B O } } }$ in $\operatorname { E q . } ( 1 )$ arises from a pointwise correspondece, we call it proper functional map (Ren et al. 2021).

When the pointwise map (and, by extension, the functional map) is unknown, the functional map can be computed by solving the following optimization problem:

$$
\mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} = \underset {\mathbf {C}} {\arg \min} E _ {\mathrm{desc}} (\mathbf {C}) + \alpha E _ {\mathrm{reg}} (\mathbf {C}),\tag{2}
$$

where $E _ { \mathrm { d e s c } } \left( \mathbf { C } \right) = { \left\| { \mathbf { C } } _ { \mathcal { N M } } ^ { \mathrm { L B O } } ( \Phi _ { \mathcal { N } } ^ { \mathrm { L B O } } ) ^ { \dagger } \mathbf { F } _ { \mathcal { N } } - ( \Phi _ { \mathcal { M } } ^ { \mathrm { L B O } } ) ^ { \dagger } \mathbf { F } _ { \mathcal { M } } \right\| } _ { \mathrm { F } } ^ { 2 }$ enforces descriptor preservation. Here, $\mathbf { F } _ { \mathcal { M } } \in \mathbb { R } ^ { m \times d }$ and $\mathbf { F } _ { \mathcal { N } } ~ \in ~ \mathbb { R } ^ { n \times d }$ are d-dimensional feature matrices for M and ${ \mathcal { N } } ,$ , respectively. The term $E _ { \mathrm { r e g } } \left( \mathbf { C } \right)$ represents a regularization function that promotes structural consistency in $\mathbf { \bar { C } } ,$ , and α is a regularization parameter. Once $\mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { { L B O } } }$ is obtained, the pointwise map is derived through a nearest neighbor search in the spectral embedding spaces $( \Phi _ { \mathcal { N } } ^ { \mathrm { L B O } }$ and $\Phi _ { \mathcal { M } } ^ { \mathrm { L B O } } \mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { L B O } } )$ . However, the quality of the resulting map is often suboptimal. To improve this, various refinement techniques have been proposed (Melzi et al. 2019; Magnet et al. 2022; Donati et al. 2022; Ren et al. 2018; Hu et al. 2021), which iteratively alternate between optimizing the functional and pointwise maps to enhance accuracy.

## Deep Functional Map

Deep functional map (DFM) methods have become stateof-the-art for non-rigid shape correspondence. The standard DFM pipeline, illustrated in Figure 2, consists of three main stages: feature extraction, differentiable map estimation, and unsupervised loss computation.

Feature Extraction. Given two input shapes, M and ${ \mathcal { N } } ,$ a trainable Siamese network $\mathcal { F } _ { \theta }$ is employed to extract pointwise feature descriptors, $\mathbf { F } _ { \mathcal { M } }$ and $\mathbf { F } _ { \mathcal { N } }$ , respectively. Here, θ represents the learnable network parameters. DiffusionNet (Sharp et al. 2022) has emerged as the de facto standard backbone for this task, as it excels at learning robust features that are invariant to discretization and aware of orientation.

Differentiable Map Estimation. The learned features are then passed to a differentiable solver, which computes a functional map, $\mathbf { C } ,$ by minimizing an energy function such as the one specified in Equation (2). The differentiability of this step is crucial as it allows gradients to flow back to the feature extraction network during training.

![](images/f9ebed810aac4bd12a4bc7194d20ca586eb971da6780d1a41ffc0550c43def35.jpg)  
Figure 2: A comparison of deep functional map architectures. (a) The conventional dual-branch framework which is fully differentiable and thus limited to differentiable solvers. (b) Our proposed MDND framework, which introduces a paradigm shift by incorporating a non-differentiable iterative refinement oracle. This oracle generates a high-quality target map, while gradients for training the feature extractor propagate exclusively through the parallel, differentiable branch. This design decouples refinement from learning and avoids potential gradient conflicts.

Unsupervised Losses. To train the feature extractor $\mathcal { F } _ { \theta }$ without ground-truth correspondences, several structural regularizers are imposed on the computed functional maps. Foundational losses include an orthogonality regularizer, which encourages the map to be area-preserving, and a bijectivity regularizer, which enforces cycle consistency $( \mathrm { i . e . }$ mapping from M to $\mathcal { N }$ and back should approximate the identity). These are formulated as:

$$
\mathcal {L} _ {\mathrm{orth}} = \left\| \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{T}} \mathbf {C} _ {\mathcal {N M}} - \mathbf {I} \right\| _ {\mathrm{F}} ^ {2} + \left\| \mathbf {C} _ {\mathcal {M N}} ^ {\mathrm{T}} \mathbf {C} _ {\mathcal {M N}} - \mathbf {I} \right\| _ {\mathrm{F}} ^ {2},\tag{3}
$$

$$
\mathcal {L} _ {\mathrm{bij}} = \left\| \mathbf {C} _ {\mathcal {N M}} \mathbf {C} _ {\mathcal {M N}} - \mathbf {I} \right\| _ {\mathrm{F}} ^ {2} + \left\| \mathbf {C} _ {\mathcal {M N}} \mathbf {C} _ {\mathcal {N M}} - \mathbf {I} \right\| _ {\mathrm{F}} ^ {2}\tag{4}
$$

More recently, (Cao, Roetzer, and Bernard 2023) introduced a coupling loss used to enforce consistency between the map computed by the differentiable solver and the one derived from the soft pointwise map $\Pi _ { \mathcal { M N } } ^ { \mathrm { s o f t } }$ , which is derived from the feature similarity using a Softmax operation.:

$$
\mathcal {L} _ {\mathrm{couple}} = \left\| \mathbf {C} _ {\mathcal {N M}} - \Phi_ {\mathcal {M}} ^ {\dagger} \Pi_ {\mathcal {M N}} ^ {\mathrm{soft}} \Phi_ {\mathcal {N}} \right\| _ {\mathrm{F}} ^ {2}\tag{5}
$$

This encourages the learned functional map to correspond to a valid pointwise map, significantly improving matching accuracy.

Our MDND framework fundamentally departs from this standard pipeline. Instead of relying on a differentiable solver to compute an initial map, we leverage a powerful, non-differentiable iterative refinement algorithm. This algorithm directly optimizes the functional map to a much higher quality, and its output is then used as a supervisory signal to guide the learning of the feature network. This key difference allows us to break free from the constraints of end-toend differentiability and integrate the strengths of axiomatic refinement into the deep learning process.

## Method

Traditional iterative optimization methods are crucial in functional map computations; however, their direct integration into deep learning frameworks has been hindered by their non-differentiable nature. To overcome this limitation, we introduce a novel paradigm called MDND (Merging Differentiable and Non-Differentiable Branches), which facilitates information exchange between differentiable and non-differentiable components within deep functional maps. This approach challenges the prevailing assumption that only differentiable operations are suitable for such frameworks. Figure 2 offers a concise overview of our method. Specifically, the first branch generates hard correspondence derived from learned features and serves as the input to produce a well-structured functional map using a novel iterative optimization solver (see Algorithm 1). This map subsequently serves as a supervisory signal to guide the backpropagation of the second Differentiable branch, which operates on soft correspondences. A detailed explanation of our method is provided in the following.

## Feature Extractor

The first core component of our network is the Deep Feature Module, implemented as a Siamese network with shared weights. This module extracts features from the source and target shapes, which consist of m and n vertices, respectively. We employ DiffusionNet (Sharp et al. 2022), a stateof-the-art surface feature extractor that utilizes diffusion across the surface to generate features resilient to discretization variations. Furthermore, DiffusionNet incorporates a spatial gradient operation to effectively address bilateral symmetry. The extracted features for the source and target shapes are denoted by $\mathbf { F } _ { \mathcal { M } } \in \mathbb { R } ^ { m \times d }$ and $\mathbf { F } _ { \mathcal { N } } \in \mathbb { R } ^ { n \times d } ,$ respectively, where d signifies the dimensionality of the learned features.

## Non-Differentiable Iterative Refinement

Inspired by the efficiency of spectral filtering techniques like MWP (Hu et al. 2021), we propose a novel and efficient iterative optimization method, which we term Hybrid Wavelet

Filtering (HWF). The core idea is that a high-quality correspondence can be recovered through a remarkably simple iterative loop: (1) converting a pointwise map to its functional map representation, (2) refining this functional map via spectral filtering, and (3) converting the refined map back to an updated pointwise map. Starting with an initial correspondence, iterating these steps rapidly converges to an accurate solution at a very low computational cost.

Our primary contribution, which distinguishes HWF from MWP, is the generalization of this filtering process to hybrid spectral bases. Instead of operating solely on the standard LBO basis, our method leverages a combined basis of both LBO and ELA eigenfunctions. This is crucial for improving robustness in challenging scenarios involving nonisometric deformations and topological noise, where the LBO basis alone is insufficient. While inspired by MWP, we present a completely different theoretical derivation, which is provided in detail in the appendix.

The complete algorithmic workflow is detailed in Algorithm 1. In the algorithm, $\Phi _ { \mathcal { M } } ^ { \mathrm { L B O } }$ and $\Phi _ { \mathcal { M } } ^ { \mathrm { E L A } }$ are matrices composed of the LBO and ELA eigenfunctions, while $\Lambda$ LBO and $\mathrm { \Delta \mathrm { \Omega } ^ { \bullet } }$ are the diagonal matrices of their corresponding eigenvalues. $\{ g ( s _ { l } \lambda ) \} _ { l = 1 } ^ { L }$ represents a family of spectral manifold wavelet filters, $\Pi _ { \mathcal { M N } } ^ { \mathrm { h a r d } }$ is a vertex index sequence, and $( \mathbf { C } _ { N \mathcal { M } } ^ { \wedge } ) ^ { * }$ denotes to the adjoint operator of $\mathbf { C } _ { \mathcal { N } \mathcal { M } } ^ { \wedge }$

In the following section, we will detail how we embed this powerful, non-differentiable HWF algorithm into our deep functional map framework to serve as a supervisory oracle, guiding the learning of robust feature descriptors.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: HWF for Correspondence

Input: Initialize pointwise map $\Pi_{\mathcal{MN}}^{\text{hard}}$

Output: Refined $\Pi_{\mathcal{MN}}^{\text{hard}}$, $(\mathbf{C}_{\mathcal{NM}}^{\text{ELA}})^{\wedge}$, $(\mathbf{C}_{\mathcal{NM}}^{\text{LBO}})^{\wedge}$

Iterative updates between $\mathbf{C}_{\mathcal{NM}}^{\text{LBO}}$, $\mathbf{C}_{\mathcal{NM}}^{\text{ELA}}$ and $\Pi_{\mathcal{MN}}^{\text{hard}}$

For $i = 1$ to maxIter do

$\mathbf{C}_{\mathcal{NM}}^{\text{ELA}} = (\Phi_{\mathcal{M}}^{\text{ELA}})^{\dagger} \Pi_{\mathcal{MN}}^{\text{hard}} \Phi_{\mathcal{N}}^{\text{ELA}}$ $(\mathbf{C}_{\mathcal{NM}}^{\text{ELA}})^{\wedge} = \sum_{l=1}^{L} g(s_l \boldsymbol{\Lambda}_{\mathcal{M}}^{\text{ELA}}) \mathbf{C}_{\mathcal{NM}}^{\text{ELA}} g(s_l \boldsymbol{\Lambda}_{\mathcal{N}}^{\text{ELA}})$ $\mathbf{C}_{\mathcal{NM}}^{\text{LBO}} = (\Phi_{\mathcal{M}}^{\text{LBO}})^{\dagger} \Pi_{\mathcal{MN}}^{\text{hard}} \Phi_{\mathcal{N}}^{\text{LBO}}$ $(\mathbf{C}_{\mathcal{NM}}^{\text{LBO}})^{\wedge} = \sum_{l=1}^{L} g(s_l \boldsymbol{\Lambda}_{\mathcal{M}}^{\text{LBO}}) \mathbf{C}_{\mathcal{NM}}^{\text{LBO}} g(s_l \boldsymbol{\Lambda}_{\mathcal{N}}^{\text{LBO}})$ $\Pi_{\mathcal{MN}}^{\text{hard}} = \text{NNsearch}\begin{pmatrix} \Phi_{\mathcal{N}}^{\text{ELA}}((\mathbf{C}_{\mathcal{NM}}^{\text{ELA}})^{*}) &amp; \Phi_{\mathcal{M}}^{\text{ELA}} \\ \Phi_{\mathcal{N}}^{\text{LBO}}((\mathbf{C}_{\mathcal{NM}}^{\text{LBO}})^{*}) &amp; \Phi_{\mathcal{M}}^{\text{LBO}} \end{pmatrix}$

end
</div>

## The MDND Dual-Branch Architecture

Our framework is built on a dual-branch architecture designed to leverage the strengths of both differentiable learning and non-differentiable optimization. One branch operates in a fully differentiable manner to enable gradient-based training, while the other, non-differentiable branch acts as a powerful refinement oracle to provide high-quality supervision.

Non-Differentiable Refinement Branch. The purpose of this branch is to generate a highly accurate target correspondence. It begins by computing an initial hard pointwise map, $\Pi _ { \mathcal { M N } } ^ { \mathrm { h a r d } }$ , via a simple nearest-neighbor search on the learned features $\mathbf { F } _ { \mathcal { M } }$ and $\mathbf { F } _ { \mathcal { N } } { : }$

$$
\Pi_ {\mathcal {M N}} ^ {\mathrm{hard}} = \mathrm{NNsearch} (\mathbf {F} _ {\mathcal {N}}, \mathbf {F} _ {\mathcal {M}})\tag{6}
$$

This hard map, which can be memory-efficiently represented as a single vector of indices in implementation, serves as the input to our Hybrid Wavelet Filtering (HWF) algorithm (Algorithm 1). The HWF iteratively refines this initial guess, producing high-quality functional maps based on hybrid spectral bases, $( \dot { \mathbf { C } } _ { \mathcal { N M } } ^ { \mathrm { E L A } } ) ^ { \wedge }$ and $( \mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { L B O } } ) ^ { \wedge }$ . Since the entire HWF process is parameter-free and non-differentiable, we do not track gradients through this branch, treating its output solely as a supervisory signal.

Differentiable Learning Branch. This branch is responsible for learning the feature extractor $\mathcal { F } _ { \theta }$ . To maintain a differentiable path for backpropagation, we compute a soft pointwise map, $\Pi _ { \mathcal { M N } } ^ { \mathrm { s o f t } }$ , from the learned features. This is achieved by calculating a feature similarity matrix and applying a temperature-scaled Softmax operator:

$$
\Pi_ {\mathcal {M N}} ^ {\mathrm{soft}} = \mathrm{Softmax} (\mathbf {F} _ {\mathcal {M}} \mathbf {F} _ {\mathcal {N}} ^ {\mathrm{T}} / \tau),\tag{7}
$$

where $\tau$ is the temperature parameter that controls the softness of the resulting map. The output of this branch, $\Pi _ { \mathcal { M N } } ^ { \mathrm { s o f t } } ,$ is a soft correspondence that can be directly used in our loss function to update the network weights.

## Unsupervised Loss Function

Traditional DFM frameworks often rely on multiple structural regularizers, such as orthogonality and bijectivity losses. Balancing the weights of these competing terms can be challenging and can complicate the optimization landscape. To avoid this, we adopt a single, streamlined unsupervised loss function inspired by recent work (Hu et al. 2023). Our loss directly enforces consistency between the output of our two branches: the soft pointwise map from the differentiable branch and the high-quality functional map from the non-differentiable oracle. The alignment loss constructed upon hybrid spectral bases is defined as:

$$
\mathcal {L} _ {\mathrm{align}} = \left\| \Phi_ {\mathcal {M}} - \Pi_ {\mathcal {M N}} ^ {\mathrm{soft}} \Phi_ {\mathcal {N}} (\mathbf {C} _ {\mathcal {N M}} ^ {\wedge}) ^ {*} \right\| _ {\mathrm{F}} ^ {2}.\tag{8}
$$

During backpropagation, $\mathbf { C } _ { \mathcal { N M } } ^ { \wedge }$ obtained by the HWF refiner is treated as a fixed, detached constant, ensuring that gradients only flow through the differentiable branch to update the feature extractor. This elegant formulation allows the network to learn meaningful correspondences by chasing a high-quality, iteratively refined target, without the need for multiple, hand-weighted loss terms.

## Experiments

In this section, we conduct extensive experiments to evaluate our method. We compare MDND against a diverse set of previous approaches across a broad range of challenging scenarios, from near-isometric matching to settings with sig nificant non-isometric deformations and topological noise.

<table><tr><td>Method / Dataset</td><td>F_r/F_r</td><td>S_r/S_r</td><td>F_r/S_r</td><td>S_r/F_r</td><td>SMAL</td><td>DT4D-H inter</td><td>DT4D-H intra</td><td>TOPKIDS</td></tr><tr><td>BCICP</td><td>6.1</td><td>11.0</td><td>-</td><td>-</td><td>28.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ZoomOut</td><td>6.1</td><td>7.5</td><td>-</td><td>-</td><td>38.4</td><td>29.0</td><td>4.0</td><td>33.7</td></tr><tr><td>SmoothShells</td><td>2.5</td><td>4.7</td><td>-</td><td>-</td><td>36.1</td><td>6.4</td><td>1.2</td><td>10.8</td></tr><tr><td>DiscreteOp</td><td>5.6</td><td>13.1</td><td>-</td><td>-</td><td>38.1</td><td>27.6</td><td>3.6</td><td>35.5</td></tr><tr><td>MWP</td><td>3.1</td><td>4.1</td><td>-</td><td>-</td><td>20.9</td><td>25.4</td><td>25.4</td><td>5.7</td></tr><tr><td>FMNet</td><td>11.0</td><td>17.0</td><td>30.0</td><td>33.0</td><td>42.0</td><td>38.0</td><td>9.6</td><td>-</td></tr><tr><td>GeomFmaps</td><td>3.5</td><td>4.3</td><td>4.8</td><td>4.0</td><td>8.4</td><td>4.2</td><td>1.9</td><td>-</td></tr><tr><td>DeepShells</td><td>1.9</td><td>4.5</td><td>6.8</td><td>5.5</td><td>28.7</td><td>31.1</td><td>3.4</td><td>13.7</td></tr><tr><td>DUOFMNet</td><td>2.5</td><td>2.6</td><td>4.2</td><td>2.7</td><td>6.7</td><td>15.8</td><td>2.6</td><td>-</td></tr><tr><td>AttentiveFMaps</td><td>1.9</td><td>2.1</td><td>2.6</td><td>1.9</td><td>4.4</td><td>11.6</td><td>1.7</td><td>23.4</td></tr><tr><td>ConsistentFMaps</td><td>2.3</td><td>2.4</td><td>2.6</td><td>2.5</td><td>5.2</td><td>6.1</td><td>1.2</td><td>-</td></tr><tr><td>RFMNet</td><td>1.6</td><td>4.5</td><td>5.3</td><td>2.1</td><td>4.4</td><td>5.4</td><td>1.8</td><td>4.9</td></tr><tr><td>ULRSSM</td><td>1.6</td><td>1.8</td><td>6.4</td><td>4.5</td><td>4.4</td><td>4.1</td><td>0.9</td><td>9.2</td></tr><tr><td>MSSFMaps</td><td>1.9</td><td>2.6</td><td>4.6</td><td>2.8</td><td>4.3</td><td>4.1</td><td>1.8</td><td>-</td></tr><tr><td>HybridFMaps</td><td>1.5</td><td>1.8</td><td>8.2</td><td>1.8</td><td>3.3</td><td>3.5</td><td>1.0</td><td>5.0</td></tr><tr><td>MSRFMNet</td><td>1.7</td><td>2.1</td><td>2.6</td><td>2.0</td><td>4.5</td><td>36.2</td><td>1.5</td><td>33.2</td></tr><tr><td>DFAFM</td><td>1.6</td><td>1.9</td><td>2.7</td><td>1.9</td><td>3.9</td><td>4.2</td><td>0.9</td><td>6.3</td></tr><tr><td>Ours</td><td>1.6</td><td>1.9</td><td>2.1</td><td>1.6</td><td>3.1</td><td>4.4</td><td>1.0</td><td>3.5</td></tr></table>

Table 1: Quantitative comparison with state-of-the-art methods. We report the mean geodesic error (×100) across datasets representing near-isometric, non-isometric, and topologically noisy scenarios. Methods are grouped by category. The best and second-best results are highlighted in bold and underlined, respectively.

## Implementation Details

Our method was implemented in PyTorch, and all experiments were run on a single NVIDIA RTX 4090 GPU. Following standard evaluation protocols, we report the mean geodesic error, normalized by the square root of the source shape’s area. For fair comparison, we use 128-dimensional HKS descriptors (Sun, Ovsjanikov, and Guibas 2009) as input features for all methods and datasets. We set the LBO basis size to 128 and the ELA basis size to 200 for all datasets, except for SMAL, where it was reduced to 100 due to the nature of the shapes. No post-processing or test-time adaptation was applied to our results. For readability, all reported geodesic errors in our tables are multiplied by 100.

## Comparison with State-of-the-Art

Baselines. We compare our method against a comprehensive set of recent and influential works, which can be categorized as follows:

• Axiomatic methods: BCICP (Ren et al. 2018), ZoomOut (Melzi et al. 2019), Smooth-Shells (Eisenberger, Lahner, and Cremers 2020), DiscreteOp (Ren et al. 2021), and MWP (Hu et al. 2021).

• Supervised methods: FMNet (Litany et al. 2017) and GeomFmap (Donati, Sharma, and Ovsjanikov 2020).

• Unsupervised methods: A wide range of recent approaches including DeepShells (Eisenberger et al. 2020), DUOFMNet (Donati, Corman, and Ovsjanikov 2022), AttentiveFMaps (Li, Donati, and Ovsjanikov 2022), ConsistentFMaps (Sun et al. 2023), RFMNet (Hu et al. 2023), ULRSSM (Cao, Roetzer, and Bernard

2023), MSSFMaps (Magnet and Ovsjanikov 2024), HybridFMaps (Bastian et al. 2024), MSRFMNet (Liu et al. 2024b) and DFAFM (Luo et al. 2025).

![](images/5dfafe57d1a680002a2d4ab48828f59e27a61adb1923449a5330865a8e3af12f.jpg)  
Figure 3: Cross-dataset generalization results. Correspondence quality is visualized using texture transfer when training and testing on different datasets.

Near-isometric Shape Matching. We first evaluate MDND on two standard near-isometric benchmarks: FAUST (Bogo et al. 2014) and SCAPE (Anguelov et al. 2005). As shown in Table 1, our method achieves performance comparable to the state-of-the-art on the standard remeshed FAUST and SCAPE test sets. Notably, our method excels in the generalization tests (F→S and S→F), indicating its ability to learn robust features that transfer well across different shape collections.

Non-isometric Shape Matching. To assess performance on more challenging non-isometric shapes, we use the

<table><tr><td>Method/Dataset</td><td>SCAPE</td><td>SMAL</td><td>TOPKIDS</td></tr><tr><td>Differentiable Solver</td><td>2.0</td><td>4.2</td><td>9.4</td></tr><tr><td>Non-Differentiable Iterative Refinement</td><td>1.9</td><td>3.1</td><td>3.5</td></tr></table>

Table 2: Impact of the non-differentiable refinement branch. To measure the effectiveness of our core contribution, we replaced our non-differentiable oracle with a standard differentiable solver. The results show a clear performance drop for the fully differentiable version, confirming that our architecture is crucial for achieving state-of-the-art accuracy.

SMAL (Zuffi et al. 2017) and DT4D-H (Magnet et al. 2022) datasets. On SMAL, which contains various tetrapod species, Table 1 shows that our method significantly outperforms approaches that rely solely on the LBO basis, underscoring the importance of extrinsic information for non-isometric correspondence. On the DT4D-H dataset, our method demonstrates superiority over most competitors in both intra-class and inter-class matching scenarios.

Matching with Topological Noise. Finally, we test the robustness of our method on the TOPKIDS dataset (Lahner¨ et al. 2016), which features near-isometric deformations corrupted by significant topological artifacts. The results in Table 1 are striking: our method achieves the best performance, improving upon the next-best approach by a remarkable 30%. This highlights the exceptional robustness of our refinement oracle and learning framework.

## Ablation Studies

Our primary contributions are twofold: the integration of a non-differentiable iterative refinement method into the deep functional map framework, and the generalization of the MWP algorithm to hybrid bases (HWF). To validate the effectiveness of each component, we conduct the following targeted ablation studies.

(1) Differentiable Solver vs. Non-Differentiable Iterative Refinement. To verify the benefit of embedding a nondifferentiable optimizer within our learning framework, we compare our full MDND model against a baseline that adheres to a more traditional, fully differentiable pipeline. In this baseline, we replace our non-differentiable iterative refinement branch with a standard differentiable solver that computes the functional map directly from the features via Equation (2). We perform experiments on representative datasets covering near-isometric (SCAPE), non-isometric (SMAL), and topologically noisy (TOPKIDS) shapes.

The results, presented in Table 2, demonstrate that our proposed MDND framework with non-differentiable refinement consistently achieves superior matching accuracy across all three categories of datasets. This confirms that using a powerful, non-differentiable oracle to generate a highquality supervisory signal is more effective than relying on a purely differentiable solver.

(2) Analysis of the Refinement Oracle: MWP vs. HWF. To address the limitations of the LBO basis in non-isometric scenarios, we proposed HWF, which generalizes MWP to a hybrid basis. To isolate and validate the effectiveness of this contribution, we configure the MDND framework with three different refinement oracles:

<table><tr><td>Method/Dataset</td><td>SCAPE</td><td>SMAL</td><td>TOPKIDS</td></tr><tr><td>MWP (LBO only)</td><td>2.2</td><td>4.9</td><td>14.7</td></tr><tr><td>MWP (ELA only)</td><td>2.3</td><td>4.1</td><td>5.5</td></tr><tr><td>HWF (LBO + ELA)</td><td>1.9</td><td>3.1</td><td>3.5</td></tr></table>

Table 3: Effectiveness of the Hybrid-Basis Refiner (HWF). To isolate the contribution of our proposed HWF algorithm, we compare its performance against single-basis alternatives. The results confirm that combining both the LBO and ELA bases within our HWF refiner is crucial for achieving the best performance.

• MWP (LBO only): The standard MWP algorithm using only the LBO basis.

• MWP (ELA only): MWP adapted to use only the ELA basis.

• HWF (LBO + ELA): Our proposed method using the hybrid basis.

All other experimental settings remain constant. The results, shown in Table 3, clearly indicate that our HWF (LBO + ELA) achieves the best matching performance across all datasets. This validates our hypothesis that generalizing the refinement to a hybrid basis provides a significant advantage, leading to a more robust oracle capable of handling diverse and challenging geometric settings.

## Conclusions

In this paper, we introduced MDND, a novel deep functional map framework that merges the power of deep learning with the robustness of traditional axiomatic optimization. Our core contribution was a dual-branch architecture that leverages a powerful, non-differentiable iterative refinement oracle to provide high-quality supervision for a feature-learning network. This was enabled by our new HWF algorithm, which operates on expressive hybrid (LBO+ELA) bases. Our approach simplifies the training process and achieves state-of-the-art accuracy, particularly on challenging nonisometric and topologically noisy shapes.

Despite these strong results, we acknowledge the inherent limitations of a purely spectral approach. A promising direction for future work is to integrate our framework with explicit spatial deformation models, potentially bridging the gap between the spectral and spatial domains for even greater robustness.

## Acknowledgments

This work was supported by the Natural Science Foundation of China (No. 62172447, 62302530), the Hunan Provincial Natural Science Foundation of China (No. 2023JJ40769), and the Postgraduate Research and Innovation Project of

Hunan Province (No. CX20250157). This work was supported in part by the High Performance Computing Center of Central South University.

## References

Anguelov, D.; Srinivasan, P.; Koller, D.; Thrun, S.; Rodgers, J.; and Davis, J. 2005. SCAPE: Shape completion and animation of people. ACM Transactions on Graphics, 24: 408– 416.

Aubry, M.; Schlickewei, U.; and Cremers, D. 2011. The wave kernel signature: A quantum mechanical approach to shape analysis. In Proceedings of IEEE/CVF International Conference on Computer Vision Workshops, 1626–1633.

Bastian, L.; Xie, Y.; Navab, N.; and Lahner, Z. 2024. Hybrid¨ Functional Maps for Crease-Aware Non-Isometric Shape Matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3313–3323.

Bogo, F.; Romero, J.; Loper, M.; and Black, M. J. 2014. FAUST: Dataset and evaluation for 3D mesh registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3794–3801.

Cao, D.; Roetzer, P.; and Bernard, F. 2023. Unsupervised Learning of Robust Spectral Shape Matching. ACM Transactions on Graphics, 42: 1–15.

Cao, D.; Roetzer, P.; and Bernard, F. 2024. Revisiting map relations for unsupervised non-rigid shape matching. In 2024 International Conference on 3D Vision (3DV), 1371– 1381.

Deng, B.; Yao, Y.; Dyke, R. M.; and Zhang, J. 2022. A Survey of Non-Rigid 3D Registration. Computer Graphics Forum, 41: 559–589.

Dinh, H. Q.; Yezzi, A.; and Turk, G. 2005. Texture transfer during shape transformation. ACM Transactions on Graphics, 24: 289–310.

Donati, N.; Corman, E.; Melzi, S.; and Ovsjanikov, M. 2022. Complex functional maps: A conformal link between tangent bundles. Computer Graphics Forum, 41: 317–334.

Donati, N.; Corman, E.; and Ovsjanikov, M. 2022. Deep orientation-aware functional maps: Tackling symmetry issues in shape matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 742–751.

Donati, N.; Sharma, A.; and Ovsjanikov, M. 2020. Deep geometric functional maps: robust feature learning for shape correspondence. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8592–8601.

Eisenberger, M.; Lahner, Z.; and Cremers, D. 2020. Smooth shells: Multi-scale shape registration with functional maps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12265–12274.

Eisenberger, M.; Toker, A.; Leal-Taixe, L.; and Cremers, D.´ 2020. Deep shells: Unsupervised shape correspondence with optimal transport. In Proceedings of Advances in Neural Information Processing Systems, 10491–10502.

Emery, P.; Lei, L.; Angela, D.; and Maks, O. 2025. DiffuMatch: Category-Agnostic Spectral Diffusion Priors for

Robust Non-rigid Shape Matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1–10.

Ezuz, D.; Solomon, J.; and Ben-Chen, M. 2019. Reversible harmonic maps between discrete surfaces. ACM Transactions on Graphics, 38: 1–12.

Gao, M.; Zorah, L.; and Bernard, F. 2021. Isometric multishape matching. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14183–14193.

Halimi, O.; Litany, O.; Rodola, E.; Bronstein, A. M.; and Kimmel, R. 2019. Unsupervised learning of dense shape correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4370– 4379.

Hammond, D. K.; Vandergheynst, P.; and Gribonval, R. 2011. Wavelets on graphs via spectral graph theory. Applied and Computational Harmonic Analysis, 30: 129–150.

Hartwig, F.; Sassen, J.; Azencot, O.; Rumpf, M.; and Ben-Chen, M. 2023. An Elastic Basis for Spectral Shape Correspondencedonati2022deep. In ACM SIGGRAPH Conference Proceedings, 1–11.

Hu, L.; Li, Q.; Liu, S.; and Liu, X. 2019. Spectral graph wavelet descriptor for three-dimensional shape matching. Journal of Zhejiang University (Engineering Science), 53: 761–769.

Hu, L.; Li, Q.; Liu, S.; and Liu, X. 2021. Efficient deformable shape correspondence via multiscale spectral manifold wavelets preservation. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14536–14545.

Hu, L.; Li, Q.; Liu, S.; Yan, D.-M.; Xu, H.; and Liu, X. 2023. RFMNet: Robust deep functional maps for unsupervised non-rigid shape correspondence. Graphical Models, 129: 1–11.

Huang, R.; Ren, J.; Wonka, P.; and Ovsjanikov, M. 2020. Consistent ZoomOut: Efficient Spectral Map Synchronization. Computer Graphics Forum, 39: 265–278.

Lahner, Z.; Rodol¨ a, E.; Bronstein, M. M.; Cremers, D.;\` Burghard, O.; Cosmo, L.; Dieckmann, A.; Klein, R.; Sahillioglu, Y.; et al. 2016. SHREC’16: Matching of de-ˇ formable shapes with topological noise. In Eurographics Workshop on 3D Object Retrieval, EG 3DOR, 55–60.

Li, C.; and Ben Hamza, A. 2013. A multiresolution descriptor for deformable 3D shape retrieval. The Visual Computer, 29: 513–524.

Li, L.; Donati, N.; and Ovsjanikov, M. 2022. Learning Multi-resolution Functional Maps with Spectral Attention for Robust Shape Matching. In Proceedings of Advances in Neural Information Processing Systems, 1–17.

Li, Q.; Guo, Y.; Liu, X.; Hu, L.; Luo, F.; and Liu, S. 2024. Deformable shape matching with multiple complex spectral filter operator preservation. The Visual Computer, 40: 4885– 4898.

Litany, O.; Remez, T.; Rodola, E.; Bronstein, A.; and Bronstein, M. 2017. Deep functional maps: Structured prediction for dense shape correspondence. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5659–5667.

Liu, S.; Luo, F.; Li, Q.; Liu, X.; and Hu, L. 2024a. AWEDD: a descriptor simultaneously encoding multiscale extrinsic and intrinsic shape features. The Visual Computer, 40: 2537–2554.

Liu, S.; Meng, J.; Hu, L.; Guo, Y.; Liu, X.; Yang, X.; Wang, H.; and Li, Q. 2024b. Multiscale spectral manifold wavelet regularizer for unsupervised deep functional maps. Computer Graphics Forum, 43(7): 1–12.

Liu, S.; Wang, H.; Hu, L.; Li, Q.; and Liu, X. 2022. Incremental functional maps for accurate and smooth shape correspondence. The Visual Computer, 38: 3313–3325.

Luo, F.; Li, Q.; Hu, L.; Wang, H.; Xu, H.; Liu, X.; Liu, S.; and Chen, H. 2025. Deep Frequency Awareness Functional Maps for Robust Shape Matching. IEEE Transactions on Visualization and Computer Graphics, 31: 7781–7794.

Magnet, R.; and Ovsjanikov, M. 2024. Memory-Scalable and Simplified Functional Map Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4041–4050.

Magnet, R.; Ren, J.; Sorkine-Hornung, O.; and Ovsjanikov, M. 2022. Smooth non-rigid shape matching via effective dirichlet energy optimization. In 2022 International Conference on 3D Vision (3DV), 495–504.

Melzi, S.; Ren, J.; Rodola, E.; Sharma, A.; Wonka, P.; and\` Ovsjanikov, M. 2019. ZoomOut: Spectral upsampling for efficient shape correspondence. ACM Transactions on Graphics, 38: 1–20.

Ovsjanikov, M.; Ben-Chen, M.; Solomon, J.; Butscher, A.; and Guibas, L. 2012. Functional maps: A flexible representation of maps between shapes. ACM Transactions on Graphics, 31: 1–11.

Pai, G.; Ren, J.; Melzi, S.; Wonka, P.; and Ovsjanikov, M. 2021. Fast Sinkhorn filters : Using matrix scaling for nonrigid Shape Correspondence with functional maps. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 384–393.

Ren, J.; Melzi, S.; Wonka, P.; and Ovsjanikov, M. 2021. Discrete optimization for shape matching. Computer Graphics Forum, 40: 81–96.

Ren, J.; Poulenard, A.; Wonka, P.; and Ovsjanikov, M. 2018. Continuous and orientation-preserving correspondences via functional maps. ACM Transactions on Graphics, 37: 1–16.

Roufosse, J.-M.; Sharma, A.; and Ovsjanikov, M. 2019. Unsupervised deep learning for structured shape matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1617–1627.

Sahillioglu, Y. 2020. Recent advances in shape correspon-˘ dence. The Visual Computer, 36: 1705–1721.

Salti, S.; Tombari, F.; and Di Stefano, L. 2014. SHOT: Unique signatures of histograms for surface and texture description. Computer Vision and Image Understanding, 125: 251–264.

Sharp, N.; Attaiki, S.; Crane, K.; and Ovsjanikov, M. 2022. DiffusionNet: Discretization Agnostic Learning on Surfaces. ACM Transactions on Graphics, 41: 1–16.

Sumner, R. W.; and Popovic, J. 2004. Deformation transfer´ for triangle meshes. ACM Transactions on Graphics, 23: 399–405.

Sun, J.; Ovsjanikov, M.; and Guibas, L. 2009. A concise and provably informative multi-scale signature based on heat diffusion. Computer Graphics Forum, 28: 1383–1392.

Sun, M.; Mao, S.; Jiang, P.; Ovsjanikov, M.; and Huang, R. 2023. Spatially and Spectrally Consistent Deep Functional Maps. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14497–14507.

Vigano, G.; Ovsjanikov, M.; and Melzi, S. 2025. NAM:\` Neural Adjoint Maps for refining shape correspondences. ACM Transactions on Graphics, 44: 1–15.

Zhuravlev, A.; Lahner, Z.; and Golyanik, V. 2025. Denois-¨ ing functional maps: Diffusion models for shape correspon dence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26899–26909.

Zuffi, S.; Kanazawa, A.; Jacobs, D. W.; and Black, M. J. 2017. 3D menagerie: Modeling the 3D shape and pose of animals. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6365–6373.

# MDND: Unsupervised Learning Guided by Non-Differentiable Refinement for Shape Correspondence

Supplementary Materials

In these supplementary materials, we provide a detailed theoretical derivation of our proposed Hybrid Wavelet Filtering (HWF) algorithm and include additional experimental results that could not be accommodated in the main paper due to space constraints.

## Derivation of the HWF Algorithm

In this section, we provide the theoretical justification for our iterative refinement algorithm, HWF. We begin by reviewing the necessary preliminaries on spectral manifold wavelets and their relationship with functional maps. We then extend the existing commutativity constraint to a hybrid basis and formulate a corresponding optimization objective. Finally, we detail the iterative procedure for solving this objective.

## Background: Functional Maps and Spectral Wavelet Commutativity

Wavelet analysis is a cornerstone of signal processing, valued for its multi-scale analysis capabilities. To generalize this concept to non-Euclidean domains, Hammond et al. (Hammond, Vandergheynst, and Gribonval 2011) introduced spectral graph wavelets, which are defined in the spectral domain analogous to classical Fourier-based wavelet operations. This formulation preserves the desirable localization properties of wavelets while being computationally efficient. This idea was subsequently extended to Riemannian manifolds by replacing the graph Laplacian with the Laplace-Beltrami Operator (LBO) (Li and Ben Hamza 2013; Hu et al. 2019).

Definition 1 (Spectral Manifold Wavelet Operator). Let $g ( \cdot )$ be a non-negative, continuous function acting as a kernel, such as a low-pass or band-pass filter. The spectral manifold wavelet operator at scale s is defined as:

$$
\mathcal {W} _ {s} = g (s \Delta),\tag{9}
$$

where $\Delta$ denotes the Laplace–Beltrami operator $( L B O ) .$ Given the eigendecomposition of the LBO, ∆ $\Phi ^ { \mathrm { L B O } } \Lambda ^ { \mathrm { L B O } } ( \Phi ^ { \mathrm { L B O } } ) ^ { \dagger }$ , the matrix representation of this operator in the LBO eigenbasis is given by $\begin{array} { r l } { \mathbf { W } _ { s } } & { { } = } \end{array}$ $\mathrm { \Gamma } \dot { \Phi } ^ { \mathrm { L B O } } g ( s \mathbf { A } ^ { \mathrm { L B O } } ) ( \Phi ^ { \mathrm { L B O } } ) ^ { \mathrm { \dagger } }$

A key property for shape correspondence is that, for nearisometric mappings, the functional map operator commutes with the spectral manifold wavelet operator. This has been established in prior work (Liu et al. 2024b) and is formalized below.

Remark .1 (Commutativity Property). Let M and $\mathcal { N }$ be two Riemannian manifolds, and let $\mathbf { \dot { \boldsymbol { T } } } : \mathbf { \dot { \boldsymbol { M } } } \to \mathcal { N }$ be an isometric mapping. Let ${ \bf \dot { \cal T } } _ { F } : { \mathcal L } ^ { 2 } ( { \mathcal N } ) \to { \mathcal L } ^ { 2 } ( { \mathcal M } )$ be the induced functional map operator and let ${ \mathcal { W } } _ { s } ^ { { \mathcal { M } } }$ and $\mathcal { W } _ { s } ^ { \mathcal { N } }$ be the spectral manifold wavelet operators. Then, $T _ { F }$ commutes with the wavelet operators:

$$
T _ {F} \mathcal {W} _ {s} ^ {\mathcal {N}} = \mathcal {W} _ {s} ^ {\mathcal {M}} T _ {F}.\tag{10}
$$

In the discrete setting, using the LBO eigenbasis, this commutativity is expressed as a constraint on the functional map matrix $\dot { \bf C } _ { \mathcal { N M } } ^ { \mathrm { L B O } } .$

$$
\mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} g (s \boldsymbol {\Lambda} _ {\mathcal {N}} ^ {\mathrm{LBO}}) = g (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{LBO}}) \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}}.\tag{11}
$$

Proof. The proof follows directly from the definitions. The matrix representation of the operators are: $\begin{array} { r l } { T _ { F } } & { { } = } \end{array}$ $\Phi _ { \mathcal { M } } ^ { \mathrm { L B O } } \mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { L B O } } ( \Phi _ { \mathcal { N } } ^ { \mathrm { L B O } } ) ^ { \dagger } , \mathbf { W } _ { s } ^ { \mathcal { M } } \mathbf { \Phi } = \Phi _ { \mathcal { M } } ^ { \mathrm { L B O } } g ( s \mathbf { A } _ { \mathcal { M } } ^ { \mathrm { L B O } } ) ( \Phi _ { \mathcal { M } } ^ { \mathrm { L B O } } ) ^ { \dagger }$ and $\mathbf { W } _ { s } ^ { \tilde { \mathcal { N } } } ~ = ~ \bar { \Phi } _ { \mathcal { N } } ^ { \mathrm { L B O } } g ( s \mathbf { \Lambda } _ { \mathcal { N } } ^ { \mathrm { L B O } } ) ( \Phi _ { \mathcal { N } } ^ { \mathrm { L B O } } ) ^ { \dagger }$ . Substituting these into Equation (10) yields:

$$
\begin{array}{l} \Phi_ {\mathcal {M}} ^ {\mathrm{LBO}} \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} (\Phi_ {\mathcal {N}} ^ {\mathrm{LBO}}) ^ {\dagger} \Phi_ {\mathcal {N}} ^ {\mathrm{LBO}} g (s \boldsymbol {\Lambda} _ {\mathcal {N}} ^ {\mathrm{LBO}}) (\Phi_ {\mathcal {N}} ^ {\mathrm{LBO}}) ^ {\dagger} = \\ \Phi_ {\mathcal {M}} ^ {\mathrm{LBO}} g (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{LBO}}) (\Phi_ {\mathcal {M}} ^ {\mathrm{LBO}}) ^ {\dagger} \Phi_ {\mathcal {M}} ^ {\mathrm{LBO}} \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} (\Phi_ {\mathcal {N}} ^ {\mathrm{LBO}}) ^ {\dagger} \end{array}
$$

Since $\Phi ^ { \dagger } \Phi \ = \ \mathbf { I }$ , the above formula can be simplified to Equation (11). □

## Motivation for a Hybrid-Basis Approach

The commutativity property in Remark .1 provides a powerful constraint for near-isometric matching, as it is based on the intrinsic LBO basis. However, this very intrinsicality is a limitation. The LBO basis is, by definition, insensitive to extrinsic deformations, making it struggle to capture highfrequency details like sharp creases and bends. This inadequacy becomes a significant bottleneck in challenging nonisometric correspondence scenarios.

Inspired by recent work on the extrinsic elastic basis (ELA basis) (Hartwig et al. 2023), which excels at representing such fine-grained geometric features, we propose a method that integrates both intrinsic and extrinsic information. Our HWF algorithm is built upon a generalized commutativity constraint that operates on a hybrid basis composed of both LBO and ELA eigenfunctions.

In the subsequent sections, we will first formulate this hybrid-basis constraint as an energy function. Then, we will derive an efficient iterative algorithm to solve the corresponding optimization problem, demonstrating how to synergistically leverage both intrinsic and extrinsic spectral information for robust shape correspondence.

## Optimization Problem

As previously discussed, only using the LBO basis has limitations within the functional map framework. Therefore, we consider employing both the LBO basis and the ELA basis simultaneously. First, we present the hybrid basis function spaces $\Phi _ { \mathcal { M } } ~ \in ~ \mathbb { R } ^ { m \times k }$ and $\Phi _ { \mathcal { N } } ~ \in ~ \mathbf { \bar { R } } ^ { n \times k }$ for manifolds M and $\mathcal { N }$ respectively, where $\Phi _ { \mathcal { M } } = [ \Phi _ { \mathcal { M } } ^ { \mathrm { L B O } } \Phi _ { \mathcal { M } } ^ { \mathrm { E L A } } ]$ $\Phi _ { \mathcal { N } } = [ \Phi _ { \mathcal { N } } ^ { \mathrm { L B O } } \Phi _ { \mathcal { N } } ^ { \mathrm { E L A } } ]$ and k is the number of hybrid bases. From this, it can be observed that the hybrid bases matrix is formed by concatenating the two sets of bases column-wise. Since our optimization objective is based on Remark .1, we give the matrix form of the spectral manifold wavelet operator under the hybrid bases: $\mathbf { W } _ { s } ^ { \mathcal { M } } = \Phi _ { \mathcal { M } } g ( s \mathbf { \Lambda } _ { \mathcal { M } } ) \Phi _ { \mathcal { M } } ^ { \dagger }$ and $\mathbf { W } _ { s } ^ { N } = \Phi _ { N } g ( s \mathbf { \Lambda } _ { N } ) \Phi _ { N } ^ { \dagger } .$ The aforementioned eigenvalue matrices are diagonal matrices composed of the eigenvalues derived from the LBO operator and those generated by the shell energy decomposition. Similarly, we present the matrix representation of the functional map $T _ { F }$ under the hybrid bases as follows: $T _ { F } = \Phi _ { \mathcal { M } } \mathbf { C } _ { \mathcal { N M } } \Phi _ { \mathcal { N } } ^ { \dagger } , \Phi _ { \mathcal { M } }$ and $\Phi _ { \mathcal { M } }$ are hybrid bases as mentioned before, $\mathbf { C } _ { \mathcal { N M } } \ \in \ \mathbb { R } ^ { k \times k }$ is the matrix form of the functional map under hybrid bases. Accordingly, the operator commutativity constraints may be recast under a hybrid bases framework as:

$$
\mathbf {C} _ {\mathcal {N M}} g (s \boldsymbol {\Lambda} _ {\mathcal {N}}) = g (s \boldsymbol {\Lambda} _ {\mathcal {M}}) \mathbf {C} _ {\mathcal {N M}}.\tag{12}
$$

Proof. First, the functional map matrix is block-partitioned in accordance with the basis function structure, obtaining T<sub>F</sub> =

$$
\left[ \Phi_ {\mathcal {M}} ^ {\mathrm{LBO}} \Phi_ {\mathcal {M}} ^ {\mathrm{ELA}} \right] \left[ \begin{array}{c c} \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} & \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{Hybrid1}} \\ \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{Hybrid2}} & \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{ELA}} \end{array} \right] \left[ \begin{array}{c} (\Phi_ {\mathcal {N}} ^ {\mathrm{LBO}}) ^ {\dagger} \\ (\Phi_ {\mathcal {N}} ^ {\mathrm{ELA}}) ^ {\dagger} \end{array} \right].   \mathbf {W} _ {s} ^ {\mathcal {M}} =
$$

$$
\left[ \Phi_ {\mathcal {M}} ^ {\mathrm{LBO}} \Phi_ {\mathcal {M}} ^ {\mathrm{ELA}} \right] \left[ \begin{array}{c c} g (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{LBO}}) & \\ & (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{ELA}}) \end{array} \right] \left[ \begin{array}{c} (\Phi_ {\mathcal {M}} ^ {\mathrm{LBO}}) ^ {\dagger} \\ (\Phi_ {\mathcal {M}} ^ {\mathrm{ELA}}) ^ {\dagger} \end{array} \right]\tag{and}
$$

$$
\mathbf {W} _ {s} ^ {\mathcal {N}} = [ \Phi_ {\mathcal {N}} ^ {\mathrm{LBO}} \Phi_ {\mathcal {N}} ^ {\mathrm{ELA}} ] \left[ \begin{array}{c c} g (s \mathbf {\Lambda} _ {\mathcal {N}} ^ {\mathrm{LBO}}) & \\ & g (s \mathbf {\Lambda} _ {\mathcal {N}} ^ {\mathrm{ELA}}) \end{array} \right] \left[ \begin{array}{c} (\Phi_ {\mathcal {N}} ^ {\mathrm{LBO}}) ^ {\dagger} \\ (\Phi_ {\mathcal {N}} ^ {\mathrm{ELA}}) ^ {\dagger} \end{array} \right].
$$

Consequently, the matrix representation of operator commutation under the hybrid bases is given by:

$$
\begin{array}{l} \left[ \begin{array}{c c} \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} & \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{Hybrid1}} \\ \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{Hybrid2}} & \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{ELA}} \end{array} \right] \left[ \begin{array}{c c} g (s \boldsymbol {\Lambda} _ {\mathcal {N}} ^ {\mathrm{LBO}}) & \\ & g (s \boldsymbol {\Lambda} _ {\mathcal {N}} ^ {\mathrm{ELA}}) \end{array} \right] \\ \left[ \begin{array}{c c} g (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{LBO}}) & \\ & g (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{ELA}}) \end{array} \right] \left[ \begin{array}{c c} \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} & \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{Hybrid1}} \\ \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{Hybrid2}} & \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{ELA.}} \end{array} \right]. \end{array}\tag{Due}
$$

to the inconsistent spectral characteristics exhibited by the LBO operator and shell energy, we conclude that the functional map matrix blocks $\mathbf { \dot { C } } _ { \mathcal { N M } } ^ { \mathrm { H y b r i d 1 } }$ and $\mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { H y b r i d 2 } }$ admit only the trivial solution of zero matrices. Therefore, the final optimization objective under hybrid bases becomes: $\mathbf { C } _ { \mathcal { N M } } g ( s \mathbf { A } _ { \mathcal { N } } ) = g ( s \mathbf { A } _ { \mathcal { M } } ) \mathbf { C } _ { \mathcal { N M } }$ , where ${ \bf C } _ { \mathcal { N M } } =$

$$
\begin{array}{l} \left[ \begin{array}{c c} \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{LBO}} & \\ & \mathbf {C} _ {\mathcal {N M}} ^ {\mathrm{ELA}} \end{array} \right],   g (s \boldsymbol {\Lambda} _ {\mathcal {N}}) = \left[ \begin{array}{c c} g (s \boldsymbol {\Lambda} _ {\mathcal {N}} ^ {\mathrm{LBO}}) & \\ & g (s \boldsymbol {\Lambda} _ {\mathcal {N}} ^ {\mathrm{ELA}}) \end{array} \right] \\ \text {and} g (s \boldsymbol {\Lambda} _ {\mathcal {M}}) = \left[ \begin{array}{c c} g (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{LBO}}) & \\ & (s \boldsymbol {\Lambda} _ {\mathcal {M}} ^ {\mathrm{ELA}}) \end{array} \right]. \end{array}
$$

In Hilbert space, the Hilbert–Schmidt norm effectively captures geometric structural information across the domain and range of the mapping operator, including anisotropic metrics, outperforming the Frobenius norm in this regard. Furthermore, the utilization of the Hilbert–Schmidt norm plays a crucial role in the success of work (Hartwig et al. 2023). Thus, given L discrete scales $\left\{ s _ { l } \right\} _ { l = 1 } ^ { L }$ , we reformulate the discrete version of Equation (12) as an optimization objective using the Hilbert–Schmidt norm to measure the magnitude of linear operators:

$$
\begin{array}{l} \underset {\mathbf {C} _ {\mathcal {N M}}} {\min} E (\mathbf {C} _ {\mathcal {N M}}), \\ E (\mathbf {C} _ {\mathcal {N M}}) = \sum_ {l = 1} ^ {L} \| \mathbf {C} _ {\mathcal {N M}} g (s _ {l} \boldsymbol {\Lambda} _ {\mathcal {N}}) - g (s _ {l} \boldsymbol {\Lambda} _ {\mathcal {M}}) \mathbf {C} _ {\mathcal {N M}} \| _ {\mathrm{HS}} ^ {2} \end{array}\tag{13}
$$

However, on the one hand, directly solving the optimization problem tends to yield a trivial solution. On the other hand, although a pointwise map can naturally induce a functional map, the converse does not always hold. To address this issue, we constrain the functional map derived from a pointwise map, as specified in Equation (15). Consequently, the functional map optimization problem can be equivalently transformed into the task of recovering the corresponding pointwise map, which can be reformulated as follows:

$$
\min _ {\Pi_ {\mathcal {M N}}} \sum_ {l = 1} ^ {L} \| \mathbf {C} _ {\mathcal {N M}} g (s _ {l} \pmb {\Lambda} _ {\mathcal {N}}) - g (s _ {l} \pmb {\Lambda} _ {\mathcal {M}}) \mathbf {C} _ {\mathcal {N M}} \| _ {\mathrm{HS}} ^ {2}\tag{14}
$$

$$
\text { where } \mathbf {C} _ {\mathcal {N M}} = \Phi_ {\mathcal {M}} ^ {\dagger} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}}\tag{15}
$$

To recover the pointwise map from Equation (14), a natural approach is to substitute the functional map using its pointwise representation as given in Equation (15). Unfortunately, it will lead to a trivial solution. So we adopt the half-quadratic splitting strategy, replacing only one of the functional maps instead of both, as done in (Melzi et al. 2019). This results in two decoupled subproblems that can be solved more effectively.

$$
\min _ {\Pi_ {\mathcal {M N}}} \sum_ {l = 1} ^ {L} \left\| \Phi_ {\mathcal {M}} ^ {\dagger} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} g (s _ {l} \pmb {\Lambda} _ {\mathcal {N}}) - g (s _ {l} \pmb {\Lambda} _ {\mathcal {M}}) \mathbf {C} _ {\mathcal {N M}} \right\| _ {\mathrm{HS}} ^ {2}\tag{16}
$$

$$
\min _ {\mathbf {C} _ {\mathcal {N M}}} \left\| \mathbf {C} _ {\mathcal {N M}} - \Phi_ {\mathcal {M}} ^ {\dagger} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} \right\| _ {\mathrm{HS}} ^ {2}\tag{17}
$$

The solution to Equation (17) is straightforward, ${ \bf C } _ { \mathcal { N M } } =$ $\Phi _ { \mathcal { M } } ^ { \dag } \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } }$ . In other words, given the pointwise map, the corresponding functional map can be derived. Obtaining the pointwise map from Equation (16) needs more rigorous logical formulation, which is elaborated carefully in the next section.

## Solution of the Pointwise Map

Owing to the multi-scale analysis of wavelets, Equation (16) has stronger constraints on the pointwise map, and we can get the following equation at each scale $s _ { l }$

$$
\Phi_ {\mathcal {M}} ^ {\dagger} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} g (s _ {l} \pmb {\Lambda} _ {\mathcal {N}}) = g (s _ {l} \pmb {\Lambda} _ {\mathcal {M}}) \mathbf {C} _ {\mathcal {N M}}\tag{18}
$$

Using the tight wavelet frame (Hu et al. 2021), which satisfies $\begin{array} { r } { \sum _ { l = 1 } ^ { L } g ( s _ { l } \lambda ) ^ { 2 } \equiv 1 } \end{array}$ , where λ represents the eigenvalue of the LBO or ELA, we can reformulate the optimization problem for the pointwise map as follows:

$$
\min _ {\Pi_ {\mathcal {M N}}} \left\| \Phi_ {\mathcal {M}} ^ {\dagger} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} - \sum_ {l = 1} ^ {L} g (s _ {l} \boldsymbol {\Lambda} _ {\mathcal {M}}) \mathbf {C} _ {\mathcal {N M}} g (s _ {l} \boldsymbol {\Lambda} _ {\mathcal {N}}) \right\| _ {\mathrm{HS}} ^ {2}.\tag{19}
$$

Proof. Given the multi-scale superimposed constraints based on Equation (18): $\begin{array} { r } { \sum _ { l = 1 } ^ { L } \Phi _ { \mathcal { M } } ^ { \dag } \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } } g ( s _ { l } \pmb { \Lambda } _ { \mathcal { N } } ) \ = } \end{array}$ $\begin{array} { r } { \sum _ { l = 1 } ^ { L } g \big ( s _ { l } \pmb { \Lambda } _ { \mathcal { M } } \big ) \mathbf { C } _ { \mathcal { N M } } , } \end{array}$ then multiply g(s<sub>l</sub>Λ<sub>N</sub> ) on both sides, $\begin{array} { r } { \sum _ { l = 1 } ^ { L } \Phi _ { \mathcal { M } } ^ { \dag } \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } } g ^ { 2 } ( s _ { l } \mathbf { { A } } _ { \mathcal { N } } ) } \end{array}$ $\begin{array} { r } { \sum _ { l = 1 } ^ { L } g ( s _ { l } \mathbf { A } _ { \mathcal { M } } ) \mathbf { C } _ { \mathcal { N M } } g ( s _ { l } \mathbf { A } _ { \mathcal { N } } ) } \end{array}$ . Using $\begin{array} { r l } { \sum _ { l = 1 } ^ { L } g ( s _ { l } \lambda ) ^ { 2 } } & { { } \equiv } \end{array}$ $^ { 1 , }$ the constraint on the pointwise map satisfies the following equation: $\begin{array} { r l r } { \Phi _ { \mathcal { M } } ^ { \dagger } \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } } } & { { } } & { = } \end{array}$ $\begin{array} { r } { \sum _ { l = 1 } ^ { L } g ( s _ { l } \mathbf { A } _ { \mathcal { M } } ) \mathbf { C } _ { \mathcal { N M } } g ( s _ { l } \mathbf { A } _ { \mathcal { N } } ) } \end{array}$ □

For notational simplicity, we define $\begin{array} { r l } { \mathbf { C } _ { \mathcal { N M } } ^ { \diamond } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { l = 1 } ^ { L } g ( s _ { l } \mathbf { { A } } _ { \mathcal { M } } ) \mathbf { C } _ { \mathcal { N M } } g ( s _ { l } \mathbf { { A } } _ { \mathcal { N } } ) } \end{array}$ . Notably, Equation (19) only constrains the image of the pointwise map Π<sub>MN</sub> within the basis space $\Phi _ { \mathcal { M } }$ . However, in a complete space, this constraint alone is insufficient. To address this, we introduce an additional regularization term to restrain the pointwise map based on the foundational lemmas proposed in (Hartwig et al. 2023; Bastian et al. 2024). Before proceeding, we first present the following foundational lemmas (Hartwig et al. 2023; Bastian et al. 2024).

Lemma .1. Let $\mathbf { F } \in \mathbb { R } ^ { n , m }$ with $n , m > 0$ be a linear operator between two finite-dimensional Hilbert spaces, and the corresponding Hilbert–Schmidt norm, then :

(a) for all injective $\Phi _ { k } \in \mathbb { R } ^ { n , k } , k > 0$

$$
\| \mathbf {F} \| _ {\mathrm{HS}} ^ {2} = \left\| \Phi_ {k} \Phi_ {k} ^ {\dagger} \mathbf {F} \right\| _ {\mathrm{HS}} ^ {2} + \left\| (\mathbf {I} - \Phi_ {k} \Phi_ {k} ^ {\dagger}) \mathbf {F} \right\| _ {\mathrm{HS}} ^ {2}\tag{20}
$$

(b) and for all injective $\Phi _ { k } \in \mathbb { R } ^ { m , k } , k > 0$

$$
\| \mathbf {F} \| _ {\mathrm{HS}} ^ {2} = \left\| \Phi_ {k} \Phi_ {k} ^ {\dagger} \mathbf {F} \right\| _ {\mathrm{HS}} ^ {2} + \left\| (\mathbf {I} - \Phi_ {k} \Phi_ {k} ^ {\dagger}) \mathbf {F} \right\| _ {\mathrm{HS}} ^ {2}\tag{21}
$$

Leveraging the above lemma, we convert the constraint problem on a pointwise map into the following form:

$$
\min _ {\Pi_ {\mathcal {M N}}} \left\| \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} - \Phi_ {\mathcal {M}} \mathbf {C} _ {\mathcal {N M}} ^ {\wedge} \right\| _ {\mathrm{HS}} ^ {2}\tag{22}
$$

Proof. It is clear that min<sub>Π</sub> $\mathbf { \Lambda } _ { [ \mathcal { M N } } \left\| \Phi _ { \mathcal { M } } ^ { \dagger } \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } } - \mathbf { C } _ { \mathcal { N } \mathcal { M } } ^ { \wedge } \right\| _ { \mathrm { H S } } ^ { 2 }$ is equivalent to min $\begin{array} { r l } & { \operatorname { \mathrm { 1 } } _ { \boldsymbol { \mathcal { M N } } } \left\| \Phi _ { \mathcal { M } } \Phi _ { \mathcal { M } } ^ { \dagger } \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } } - \Phi _ { \mathcal { M } } \mathbf { C } _ { \mathcal { N } \mathcal { M } } ^ { \wedge } \right\| _ { \mathrm { H S } } ^ { 2 } . } \end{array}$ Using the $\Phi _ { \mathcal { M } } ^ { \dagger } \Phi _ { \mathcal { M } } = \mathbf { I }$ , a new optimization objective is obtained by incorporating another regularization term into the original one, $i . e . , \left\| \Phi _ { \mathcal { M } } \Phi _ { \mathcal { M } } ^ { \dagger } ( \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } } - \Phi _ { \mathcal { M } } \mathbf { C } _ { \mathcal { N } \mathcal { M } } ^ { \wedge } ) \right\| _ { \mathrm { H S } } ^ { 2 } +$ $\left\| ( \mathbf { I } - \Phi _ { \mathcal { M } } \Phi _ { \mathcal { M } } ^ { \dagger } ) ( \Pi _ { \mathcal { M N } } \Phi _ { \mathcal { N } } - \Phi _ { \mathcal { M } } \mathbf { C } _ { \mathcal { N } \mathcal { M } } ^ { \wedge } ) \right\| _ { \mathrm { H S } } ^ { 2 }$ $\begin{array} { r } { \big \| \boldsymbol { \Pi } _ { \mathcal { M N } } \Phi _ { \mathcal { N } } - \Phi _ { \mathcal { M } } \mathbf { C } _ { \mathcal { N M } } ^ { \wedge } \big \| _ { \mathrm { H S } } ^ { 2 } . } \end{array}$ . Therefore, Equation (22) yields. □

We additionally introduce the property of the shape difference operator (Hartwig et al. 2023) to further strengthen the constraint in Equation (22). The shape difference operator is defined as the product of the functional map and its adjoint operator $( \bar { \bf C } _ { N M } ^ { \mathrm { ~ \bar { ~ } } } { \bf C } _ { N M } ^ { \ast } )$ and the deviation of the shape difference operator from the identity reflects disparities in area distortion, where $\mathbf { C } _ { \mathcal { N M } } ^ { * }$ is the adjoint operator of the functional map $\mathbf { C } _ { \mathcal { N M } }$ . The definition of the adjoint operator relies on the following lemma.2 and lemma.3.

Lemma .2. For functions $f , g$ in $\langle \Phi _ { k } \rangle$ , where $x , y \in \mathbb { R } ^ { k }$ are the corresponding basis coefficients, namely $f = \Phi _ { k } x$ and $\begin{array} { r } { { \mathrm {  ~ \nabla ~ } } g = \Phi _ { k } y , } \end{array}$ , we obtain $\langle f , g \rangle _ { \mathbf { M } } = x ^ { \mathrm { T } } \mathbf { M } _ { k } y$ . Here, ${ { \bf { M } } _ { k } } =$ $\Phi _ { k } ^ { \mathrm { T } } \mathbf { M } \Phi _ { k } \in \mathbb { R } ^ { k , k }$ is the mass matrix with respect to the reduced basis $\Phi _ { k } .$ . where M is the area matrix on the manifold, and $\Phi _ { k }$ denotes the truncated basis matrix.

Lemma .3. Given the functional map $\mathbf { C } _ { \mathcal { N M } }$ . The corresponding adjoint operator can be represented by: $\mathbf { C } _ { \mathcal { N M } } ^ { * } =$ $\mathbf { A } _ { \mathcal { M } } ^ { - 1 } \mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { T } } \mathbf { A } _ { \mathcal { N } }$ , where $\mathbf { A } _ { \mathcal { N } }$ and $\mathbf { A } _ { \mathcal { M } }$ are the mass matrices with respect to the reduced basis $\Phi _ { k }$ based on the shape M and $\mathcal { N }$ respectively.

Proof. After obtaining the mass matrix under the reduced basis, we can derive the adjoint operator of the functional map according to the method in (Hartwig et al. 2023). The following equations are defined in (Hartwig et al. 2023).

$$
\langle \mathbf {C} _ {\mathcal {N M}} x, y \rangle_ {\mathbf {A} _ {\mathcal {N}}} = \langle x, \mathbf {C} _ {\mathcal {N M}} ^ {*} y \rangle_ {\mathbf {A} _ {\mathcal {M}}}\tag{23}
$$

Where $f$ and $g$ are descriptor functions defined on manifolds $\mathcal { M }$ and ${ \bar { \mathcal { N } } } ,$ , and $x$ and $y$ are the corresponding basis coefficients under the respective basis function spaces. $\langle \mathbf { C } _ { \mathcal { N M } } x , y \rangle _ { \mathbf { A } _ { N } } = ( \mathbf { C } _ { \mathcal { N M } } x ) ^ { \mathrm { { T } } } \mathbf { A } _ { \mathcal { N } } y = x ^ { \mathrm { { T } } } \mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { { T } } } \mathbf { A } _ { \mathcal { N } } y$ $\langle x , \mathbf { C } _ { \mathcal { N M } } ^ { * } y \rangle _ { \mathbf { A } _ { \mathcal { M } } } = x ^ { \mathrm { T } } \mathbf { A } _ { \mathcal { M } } \mathbf { C } _ { \mathcal { N M } } ^ { * } y$ $\mathbf { C } _ { \mathcal { N M } } ^ { \ast } = \mathbf { A } _ { M } ^ { - 1 } \mathbf { C } _ { \mathcal { N M } } ^ { \mathrm { T } } \mathbf { A } _ { N } .$ □

The functional map in Equation (22) can be regarded as a filtered functional map for eigenvalues. To fully leverage the advantages of the aforementioned shape difference operator, we reformulate the optimization problem as shown in Equation (24) to be consistent with Equation (22).

$$
\min _ {\Pi_ {\mathcal {M N}}} \left\| \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} (\mathbf {C} _ {\mathcal {N M}} ^ {\wedge}) ^ {*} - \Phi_ {\mathcal {M}} \right\| _ {\mathrm{HS}} ^ {2}\tag{24}
$$

Proof.

$$
\begin{array}{r l} & {\| \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} - \Phi_ {\mathcal {M}} (\mathbf {C} _ {\mathcal {N M}} ^ {\wedge}) \| _ {\mathrm{HS}} ^ {2}} \\ & {= \mathrm{tr} (\Phi_ {\mathcal {N}} ^ {*} \Pi_ {\mathcal {M N}} ^ {*} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} - \Phi_ {\mathcal {N}} ^ {*} \Pi_ {\mathcal {M N}} ^ {*} \Phi_ {\mathcal {M}} \mathbf {C} _ {\mathcal {N M}} ^ {\wedge}} \\ & {- (\mathbf {C} _ {\mathcal {N M}} ^ {\wedge}) ^ {*} \Phi_ {\mathcal {M}} ^ {*} \Pi_ {\mathcal {M N}} \Phi_ {\mathcal {N}} + (\mathbf {C} _ {\mathcal {N M}} ^ {\wedge}) ^ {*} \Phi_ {\mathcal {M}} ^ {*} \Phi_ {\mathcal {M}} \mathbf {C} _ {\mathcal {N M}} ^ {\wedge})} \end{array}
$$

By applying $\operatorname { t r } ( \mathbf { A } + \mathbf { B } ) \ = \ \operatorname { t r } ( \mathbf { A } ) + \operatorname { t r } ( \mathbf { B } ) , \operatorname { t r } ( \mathbf { A } \mathbf { B } ) \ =$ $\operatorname { t r } ( \mathbf { B } \mathbf { A } )$ and $( \mathbf { C } _ { \mathcal { N M } } ^ { \wedge } ) ^ { * } \mathbf { C } _ { \mathcal { N M } } ^ { \wedge } = \mathbf { I } ,$ it is therefore clear that Equation(22) is equal to Equation(24). □

This final form is advantageous because it can be efficiently solved for the hard correspondence matrix $\Pi _ { \mathcal { M N } }$ using a simple nearest-neighbor search.

## The Complete HWF Algorithm

Based on the analysis above, our HWF algorithm is an iterative process that alternates between solving the two decoupled subproblems:

• Update Functional Map: Given the current pointwise map $\Pi _ { \mathcal { M N } } ^ { ( i ) }$ , compute the updated functional map $\mathbf { C } _ { \mathcal { N M } } ^ { \left( i \right) }$ using its projection onto the hybrid basis (solving Eq. (17)). That is, the functional map matrix is obtained as the product of the basis function matrix defined on manifold M and the basis function matrix defined on manifold ${ \mathcal { N } } .$ , where the latter is constructed by performing row-index search based on a hard pointwise map.

• Update Pointwise Map: Given the updated functional map $\mathbf { C } _ { \mathcal { N M } } ^ { \left( i \right) }$ , compute the filtered map $\mathbf { C } _ { \mathcal { N M } } ^ { \wedge }$ and sove for the new pointwise map $\Pi _ { \mathcal { M N } } ^ { ( i + 1 ) }$ via a nearest-neighbor search on Equation (24). For manifold $\mathcal { M }$ , after concatenating the two sets of basis functions column-wise, each row can be regarded as a feature of a vertex. Similarly, for the basis function space of the manifold $\mathcal { N }$ constructed by the functional map, one can perform column-wise concatenation and then apply nearest-neighbor search on each row to obtain the pointwise map.

This process, summarized in Algorithm 1, iterates until convergence, synergistically leveraging both intrinsic and extrinsic information to find a robust correspondence.

## Additional Results

## Matching on Anisotropically Remeshed Shapes

To evaluate the robustness of our method to different mesh discretizations, we conduct tests on anisotropically remeshed versions of the FAUST (F a) and SCAPE (S a) datasets. For these tests, we use the models trained on their isotropically remeshed counterparts (F r and S r, respectively). The results, summarized in Table 4, show that our method achieves competitive performance, securing either the best or second-best results in most generalization scenarios and demonstrating strong resilience to changes in mesh structure.

<table><tr><td>Method / Dataset</td><td>F_r/F_a</td><td>F_r/S_a</td><td>S_r/F_a</td><td>S_r/S_a</td></tr><tr><td>BCICP</td><td>14.0</td><td>8.5</td><td>14.0</td><td>8.5</td></tr><tr><td>ZoomOut</td><td>15.0</td><td>8.7</td><td>15.0</td><td>8.7</td></tr><tr><td>SmoothShells</td><td>5.0</td><td>5.4</td><td>5.0</td><td>5.4</td></tr><tr><td>DiscreteOp</td><td>14.6</td><td>6.2</td><td>14.6</td><td>6.2</td></tr><tr><td>MWP</td><td>8.7</td><td>8.2</td><td>8.7</td><td>8.2</td></tr><tr><td>DeepShells</td><td>12.0</td><td>16.0</td><td>15.0</td><td>10.0</td></tr><tr><td>DUOFMNet</td><td>3.0</td><td>4.4</td><td>3.1</td><td>2.7</td></tr><tr><td>AttentiveFMaps</td><td>2.4</td><td>2.8</td><td>2.5</td><td>2.3</td></tr><tr><td>RFMNet</td><td>3.6</td><td>2.6</td><td>3.6</td><td>3.9</td></tr><tr><td>ULRSSM</td><td>2.5</td><td>8.9</td><td>7.0</td><td>1.9</td></tr><tr><td>HybridFMaps</td><td>2.0</td><td>8.8</td><td>2.6</td><td>1.8</td></tr><tr><td>DFAFM</td><td>2.0</td><td>2.9</td><td>2.6</td><td>1.9</td></tr><tr><td>Ours</td><td>2.1</td><td>2.0</td><td>2.2</td><td>1.9</td></tr></table>

Table 4: Comparative results on anisotropically remeshed shapes. This table evaluates generalization performance when models trained on isotropically remeshed data are tested on anisotropically remeshed versions of FAUST (F a) and SCAPE (S a).

## Parameter Analysis

In this section, we analyze the sensitivity of our method to two key hyperparameters: the number of elastic basis functions used in our hybrid refiner and the number of iterations performed by the HWF algorithm.

Number of Elastic Bases. Our method leverages both intrinsic (LBO) and extrinsic (ELA) bases. While we follow related work (Hu et al. 2023) for the number of LBO bases, the optimal number of ELA bases requires investigation. We experimented on the SCAPE dataset by varying the number of ELA basis functions. As shown in Figure 4, the results indicate that using 200 ELA basis functions provides the best trade-off. Fewer bases lead to degraded performance, while a larger number offers diminishing returns at the cost of increased computational overhead.

Number of Refinement Iterations. We analyzed the convergence of our HWF iterative algorithm by tracking the geodesic error on the SMAL dataset with an increasing number of iterations. The results are presented in Figure 5. To balance matching accuracy with computational efficiency, we selected three refinement iterations for all experiments reported in this paper.

![](images/be3442be837226bc8e600b79421eec7b1de5c683e922397328757a148314030e.jpg)  
Figure 4: Impact of the number of ELA basis functions. This plot shows the geodesic error on the SCAPE dataset as the number of ELA bases used in our refiner HWF is varied.

![](images/027c4bfbd8089e157880d043622a9e0ca08675493e94d6b0a127ce5aab512303.jpg)  
Figure 5: Convergence of the HWF refinement algorithm. This plot shows the geodesic error on the SMAL dataset as a function of the number of HWF iterations. The performance saturates after a few iterations.

Number of Wavelet Filters. The use of wavelet filters effectively performs frequency-domain filtering on functional maps, which is crucial for high-quality pointwise map recovery. We performed an ablation experiment on the SMAL dataset to analyze the effect of the number of wavelet filters, and the results in Figure 6 indicate that employing six filters yields the best performance. Therefore, to balance both accuracy and efficiency, we set the number of filters to six.

![](images/4be12894b7aa5f23de801bb5ab6d077b891204999959dfd823de2b78cd131b00.jpg)

Figure 6: Ablation analysis of the number of wavelet filters. The figure illustrates the trend of average geodesic error versus the number of wavelet filters on the SMAL dataset, demonstrating the effectiveness of multi-scale filtering in the HWF refinement algorithm.  
![](images/1c5eacd5fff7c93ca29629ba3e93028fc2813484d82ed65d67ddc91b14c8765c.jpg)  
Figure 7: Comparison of inference time on the SMAL dataset. The plot shows the inference time versus the number of vertices. The blue line represents ULRSSM, which includes a test-time adaptation (TTA) post-processing step. The red line represents our method, which requires no postprocessing and is significantly faster.

highly accurate but also significantly more efficient at test time.

## Inference Time Analysis

Our method is designed to be efficient at test time, as it does not require any costly post-processing modules. To analyze this efficiency, we compare the inference time of our method against ULRSSM (Cao, Roetzer, and Bernard 2023), a state-of-the-art approach that employs a costly test-time adaptation (TTA) post-processing step. In our framework, the final pointwise map is obtained directly from the nondifferentiable branch during a single forward pass. Figure 7 plots the inference time as a function of the number of vertices. The results clearly show that our method is not only

![](images/8e531c4c0da1ebcc1ad7be6417aec4ba312b4696a41469342863c11593630916.jpg)

![](images/26544526c7fe15b28a510fa41f1b5e02d223ab0e378f96a50c7432d232ff449c.jpg)  
Figure 8: Qualitative results on the non-isometric SMAL benchmark. The quality of the texture transfer highlights the accuracy of the correspondence maps generated by our method on these challenging animal shapes.

![](images/1141fc5f4eb8aab28eccc523b38403807a74a64c8e22627901ef3d6677acaeaf.jpg)

![](images/c4f6f4efd9022e35fe538976db95856b157e5842726356bf27fb12330cbe27f6.jpg)  
Figure 9: Matching results on the DT4D-H dataset. The quality of the texture transfer demonstrates our method’s ability to find meaningful correspondences between different classes of humanoid shapes.

![](images/0c0524fec1e41e62ac1be1649010c93c50c6be8be872859895d11bb710c4a2c9.jpg)  
Figure 10: Robustness to significant topological noise on the TOPKIDS dataset. The figure visualizes correspondence quality via texture transfer. The leftmost shape (kid00) serves as the source, mapped onto the various target shapes. Our method successfully handles the severe topological artifacts present in the data.