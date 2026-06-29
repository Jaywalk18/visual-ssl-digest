# Bridging Vision and Language Concepts through Optimal Transport Semantic Flow

Chenyang Zhang<sup>1∗</sup>, Anqi Dong<sup>2∗</sup>, Guangming Zhu<sup>1</sup>, Nuoye Xiong<sup>1</sup>, Siyuan Wang<sup>1</sup>, Lin Mei<sup>1</sup>, and Liang Zhang<sup>1†</sup>

<sup>1</sup> School of Computer Science and Technology,

Xidian University, Xi’an, China

<sup>2</sup> KTH Royal Institute of Technology,

Stockholm, Sweden

chenyang.zhang@stu.xidian.edu.cn, anqid@kth.se, liangzhang@xidian.edu.cn

Abstract. Concept Bottleneck Models (CBMs) promise transparent reasoning by predicting through human-interpretable concepts, yet their efectiveness fundamentally depends on how well visual and textual representations are aligned or matched. Existing vision–language CBMs often rely on pre-aligned encoders or global cosine similarity, which obscures fine-grained concept localization and fails to reflect true semantic geometry. In this work, we rethink concept alignment as dynamic cross-modal transport process instead of static projection and propose Optimal Transport Flow Concept Bottleneck Model (OTF-CBM). It first learns a data-driven semantic cost via Inverse Optimal Transport to measure cross-modal distances, and then performs unbalanced optimaltransport-based flow matching to model semantic transitions between visual patches and textual concepts. With velocity-based concept activation, OTF-CBM captures interpretable geometric relations without ODE integration. Experiments further show that OTF-CBM achieves great classification accuracy and concept faithfulness, ofering a new geometric and dynamical perspective for interpretable cross-modal reasoning. Our code will be released at https://github.com/ChenyangZhang00/OTF-CBM.

Keywords: Concept Bottleneck Model, optimal transport, inverse optimal transport, flow matching

## 1 Introduction

Concept bottleneck models expose an intermediate layer of human concepts — a model first predicts concept activations and then uses them to make final decisions [20, 39, 47]. This structure enables inspection, diagnosis, and targeted intervention. In vision–language settings, however, three limitations persist: (i) Fixed embedding dependence: reusing a shared CLIP space ties the visual encoder to a linguistic geometry that need not match concept granularity [31,36,45], (ii) Global–to–concept inference: predicting all concept scores from a single global feature blurs spatial evidence and weakens localization [27, 36, 45], and (iii) Inadequate similarity: cosine similarity aligns category labels trained by large contrastive objectives but is a poor proxy for fine region–to–concept relations [31].

![](images/8fac5aeb1270f657dafcc37c666f37eb2a67bc273c95951d151ef9614f415302.jpg)  
Fig. 1: Cross-modal concept visualization with OTF-CBM. The model localizes fine-grained parts (head, wings, legs) and aligns them with textual concepts. Compared with prior CBMs, it yields more coherent, spatially grounded components and smooth semantic flow from visual features to concept embeddings.

Efective CBM models should move beyond static projection and be able to construct a flexible, learnable cross-modal geometry that explicitly connects visual regions with textual concepts. Such a formulation requires establishing region-to-concept correspondences that are both semantically accurate and geometrically consistent across modalities. This introduces two key challenges: (i) discovering fine-grained correspondences between image patches and textual concepts in the absence of explicit component-level annotations, and (ii) modeling how these correspondences evolve into interpretable conceptual relations within a unified visual–textual space. A model capable of capturing both the spatial grounding of concepts and the dynamics of their semantic transitions would enable a more faithful and transparent conceptual reasoning process.

To this end, we propose OTF-CBM, a variant of CBM that explicitly models cross-modal geometry (Fig. 1). We first cast region–to–concept alignment as an optimal transport (OT) problem [29, 41] between unaligned visual and textual encoders, determining how visual evidence should be assigned to semantic concepts. Next, OTF-CBM learns a data-driven cross-modal cost via Inverse Optimal Transport (IoT) [4, 22, 26] from annotated region–concept associations, rather than relying on hand-crafted distances such as cosine or Euclidean metrics (Fig. 2). This yields a cost landscape under which the optimal transport plan reproduces true semantic correspondences. Since visual–textual mappings are inherently unbalanced (e.g., multiple patches may correspond to a single concept, and many background regions to none), we adopt an Unbalanced OT (UOT) formulation [5, 10, 32], allowing partial and many-to-one matching to emerge naturally. As shown in Fig. 3, the resulting transport geometry thus provides a faithful semantic metric between modalities and forms the basis of our concept reasoning.

![](images/03effc4849a13da52b2b5df2f3d14a55386418e4b371924eb5c50ce1ec5f4497.jpg)  
Balanced OT w.r.t Wasserstein distance  
Fig. 2: Failure of standard OT for cross-modal concept matching. With fixed ground cost, standard OT often misaligns visual patches and concepts in heterogeneous spaces.

Building upon on this learned geometry, we train a conditional velocity field through OT-based Flow Matching (FM) [23]. Instead of enforcing static similarity in a shared embedding space, the velocity field models the dynamic semantic flow from visual prototypes to textual concepts.<sup>⋆</sup> <sup>⋆</sup> <sup>⋆</sup> At inference time, OTF-CBM does not integrate an ODE to reconstruct a final state. Instead, it introduces a velocity-based concept activation mechanism that compares the predicted velocity at an intermediate time step with the ideal velocity toward each concept. This single-step, geometry-aware activation replaces conventional cosine similarity, capturing how visual evidence moves toward conceptual meaning and yielding concept predictions that are both more accurate and more interpretable than prior CBM variants [20, 31, 39, 47]. The contributions are as follows:

i) Cross-modal optimal transport for semantic correspondence. We introduce an Optimal Transport framework that establishes fine-grained semantic correspondences between visual and textual modalities, integrating Inverse OT for adaptive cost learning and Unbalanced OT for non-uniform mass transfer.

<sup>⋆</sup> <sup>⋆</sup> <sup>⋆</sup> Throughout, we use visual, region prototypes, and clusters interchangeably ii) Flow-based perspective on concept alignment. We reformulate concept reasoning as a dynamic transformation process driven by a learned velocity field, with geometry-aware semantic flow that infers concept activation from directional consistency.

![](images/c52e27b638a4d1d7bd69facc021dadfb7195068942dedba179630a3031aaa619.jpg)  
Fig. 3: Cross-modal concept matching with our solution. (i) We address this by learning a data-driven cost $\mathbf { c } _ { \theta }$ from component annotations, yielding semantically calibrated metrics. (ii) Combining learned cost with unbalanced OT enables flexible many-to-one and partial matchings suited to vision–language alignment.

iii) Great empirical performance. Experiments show that OTF-CBM outperforms state-of-the-art CBMs in accuracy, interpretability, and generality.

## 2 Related work

Concept-based interpretability aims to make predictions intervenable by routing decisions through human-understandable variables. Classical CBMs [20] factor prediction into concept extraction and label prediction so that downstream decisions depend on concepts. Early CBMs rely on expert-annotated attributes, which limit scalability and transfer. With large vision–language models such as CLIP [31] and GPT-style LLMs, recent works automate concept construction and supervision. Post-hoc CBM (P-CBM) [47] projects visual embeddings onto a CLIP-derived concept bank. Label-Free CBM [27] uses CLIP pseudo labels to train concept predictors without human annotation. LaBo [45] employs LLMs to propose diverse candidate concepts and applies submodular selection to refine the bank. PCBM-h [34] adds a residual head to refine concept predictions and improves accuracy at the expense of interpretability. These approaches improve scalability and performance but still impose global features on concept mappings that mix localized evidence and hinder fine-grained grounding. DOT-CBM casts CBM learning as optimal transport between image patches and concept embeddings, yielding spatially grounded concepts [44]. Our approach difers by learning cross-modal cost and using unbalanced OT, which accommodates heterogeneous semantics and many-to-one or background correspondences.

Optimal Transport theory [3, 29, 40] matches source distribution $\mu ( x )$ and target distribution $\nu ( y )$ by minimizing transport cost $\mathbf { c } ( x , y )$ over the set of couplings $\pi ( \mu , \nu ) ~ = ~ \left\{ \pi \geq 0 ~ | \pi { \bf 1 } = \mu , ~ \bar { { \bf 1 } } ^ { \top } \pi = \nu \right\}$ . The Kantorovich formulation [16, 30] reads

$$
\min _ {\pi \in \Pi (\mu , \nu)} \langle \mathbf {c}, \pi \rangle \equiv \iint_ {\mathcal {X} \times \mathcal {Y}} \mathbf {c} (x, y) \mathrm{d} \pi (x, y),
$$

and its entropic regularization [9, 12, 14, 15] with regularizer επ log π for computational eficiency, solving scalable Sinkhorn iterations and widely used in vision and multi-modal learning [8]. When masses are imbalanced, unbalanced OT (UOT) [6,7,18,32] relaxes the marginal constraints by penalizing deviations with divergences, enabling robust alignment in heterogeneous settings. Inverse Optimal Transport (IOT) [19, 22] seeks to learn a cost function that explains observed matching. Given an empirical coupling πˆ (from paired data) and a parameterized cost c<sub>θ</sub>, a practical formulation fits $\mathbf { c } _ { \theta }$ by matching the entropic OT plan $\pi _ { \theta }$ induced by $\mathbf { c } _ { \theta }$ to $\hat { \pi }$ that reads

$$
\min _ {\theta} \mathcal {D} (\hat {\pi}, \pi_ {\theta}) + \lambda   \mathcal {R} (\theta) \quad \text {s.t.} \quad \pi_ {\theta} \in \underset {\pi \in \Pi (\mu , \nu)} {\operatorname{argmin}} \left\langle \mathbf {c} _ {\theta}, \pi \right\rangle + \varepsilon   \pi \log \pi ,
$$

where $\mathcal { D }$ is a discrepancy $( \mathrm { e . g . , K L \ o r \ \ell _ { 2 } }$ between couplings), and R regularizes the cost class for stability and identifiability up to admissible gauges. Extensions include contrastive metric learning and multi-modal representation learning that learn semantically structured costs.

Flow Matching (FM) [1, 13, 23] learns a time-dependent velocity field that transports a sample $x _ { 0 }$ from a source distribution toward a target sample $x _ { 1 }$ without stochastic simulation. A common choice uses straight-line paths $\boldsymbol { x } _ { t } =$ $\left( 1 - t \right) x _ { 0 } + t x _ { 1 }$ for $t \in [ 0 , 1 ]$ and the target velocity $u _ { t } = x _ { 1 } - x _ { 0 }$ . A parametric field $v _ { \phi } ( x , t , \mathrm { c o n d } )$ is trained by the regression loss

$$
\mathcal {L} _ {\mathrm{FM}} = \mathbb {E} _ {(x _ {0}, x _ {1}) \sim \Pi , t \sim \mathcal {U} [ 0, 1 ]} \Big [ \| v _ {\phi} (x _ {t}, t, \mathrm{cond}) - u _ {t} \| ^ {2} \Big ],
$$

where $\pi$ specifies how training pairs are drawn and cond denotes optional conditioning. OT-consistent variants choose Π via minibatch OT or UOT so the supervised trajectories reflect transport geometry. OT-CFM [37, 46] integrates minibatch OT couplings to accelerate convergence and promote OT-consistent dynamics, while CrossFlow [24] extends FM to cross-modal generation without explicit noise injection. We also note that most existing OT and FM pipelines assume fixed costs or balanced marginals.

## 3 Method

We start from a standard concept bottleneck model (CBM) and replace its similarity-based concept inference with a geometry-aware matching mechanism.

In Sec. 3.1, we set up the CBM pipeline and highlight the limitation of pointwise similarity for concept prediction. The goal is simple: given image features and concept embeddings, we want a reliable way to (i) align the two modalities, and (ii) turn that alignment into stable concept activations.

Section 3.2 introduces Visual–Language Optimal Transport (VLOT), which builds an explicit cross-modal coupling between visual and concept embeddings. VLOT uses an unbalanced transport formulation and a background-aware penalty so the coupling can ignore irrelevant image content and handle missing or extra mass across modalities.

Section 3.3 makes the coupling meaningful by learning the transport cost from data. We do this via inverse optimal transport, so the resulting transport plans reflect the geometry that is actually useful for semantic matching, rather than relying on a hand-designed distance.

Finally, Section 3.4 turns the discrete transport solution into a continuous concept-inference rule. We train a straight-line velocity field using transport displacements as supervision, and we compute concept activations from how well the predicted instantaneous velocity agrees with the transport direction. This produces smooth, robust concept scores that plug directly.

## 3.1 Problem setting and notation

Classical visual classifiers learn a direct mapping $f : \mathcal { X }  \mathcal { Y }$ to approximate $P ( \boldsymbol { y } | \boldsymbol { x } )$ from data $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , but provide limited interpretability. Concept Bottleneck Models (CBMs) introduce an intermediate concept representation. Given a predefined concept set $\boldsymbol { \mathcal { C } } = \{ c _ { j } \} _ { j = 1 } ^ { M }$ , CBMs learn a two-stage mapping $f _ { X \to C } : \mathcal { X } \to \mathbb { R } ^ { M }$ and $f _ { C \to Y } : \mathbb { R } ^ { M } \to \mathcal { Y }$ . An input image is first mapped to a concept activation vector $\mathbf { a } \in \mathbb { R } ^ { M }$ , which is then used for classification. In vision–

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal{X}, \mathcal{C}$ </td><td>feature, concept space</td></tr><tr><td> $\mu, \nu$ </td><td>empirical distributions on  $\mathcal{X}$  and  $\mathcal{C}$ </td></tr><tr><td> $f_{\psi}$ </td><td>modality adapters to  $\mathbb{R}^{d_p}$ </td></tr><tr><td> $\mathbf{c}_{\theta}(x,c)$ </td><td>learned cross–modal transport cost</td></tr><tr><td> $\pi_{\theta}$ </td><td>OT or UOT plan induced by  $\mathbf{c}_{\theta}$ </td></tr><tr><td> $v_{\phi}(x,t \mid \boldsymbol{h})$ </td><td>conditional velocity field in FM</td></tr><tr><td> $\boldsymbol{h}$ </td><td>conditioning input in  $\mathbb{R}^{d_p}$ </td></tr><tr><td> $\boldsymbol{a}$ </td><td>concept activation vector</td></tr><tr><td> $f_{\text{cls}}, \hat{y}$ </td><td>classifier on concepts, predicted label</td></tr></table>

Table 1: Notation overview.

language CBMs, concept activations are typically computed via similarity in a shared embedding space, which may fail to capture structured cross-modal relations under complex distributions. Our method retains this bottleneck structure but replaces similarity-based concept inference with a geometry-aware transport and flow mechanism. The notations are summarized in Table 1.

![](images/9f4bbeae184cb15e0021f7eccc77c1d1e304369484c2e8431393845dbc367610.jpg)  
Fig. 4: Forward pipeline. Patch tokens are clustered into foreground and background. The learned cost $\mathbf { c } _ { \theta ^ { \ast } }$ forms a cost matrix to fixed concept embeddings with background penalties. Unbalanced OT yields a plan π. Samples from π to train a conditional velocity field. At inference, concept activations come from midpoint velocity alignment, then a concept classifier produces labels.

## 3.2 Geometric Coupling via Vision–Language OT

To connect arbitrary visual and textual encoders, we cast cross-modal alignment as an optimal transport (OT) problem. Unlike cosine similarity, OT returns an explicit coupling that describes how visual evidence is distributed over semantic concepts under a learned geometry. Given an image, let $x _ { 1 : N } = \{ x _ { i } \} _ { i = 1 } ^ { N } \subset \mathcal { X }$ denote visual patch embeddings and let $c _ { 1 : M } = \{ c _ { j } \} _ { j = 1 } ^ { M } \subset \mathcal { C }$ denote textual concept embeddings. A transport plan $\pi \in \mathbb { R } _ { + } ^ { N \times M }$ assigns nonnegative mass $\pi _ { i j }$ from patch $x _ { i }$ to concept $c _ { j }$

Region aggregation. Patch-level transport can be noisy because many neighboring patches describe the same region. We therefore cluster patch embeddings within each image using K-means and form K group tokens by averaging features inside each cluster. Let $\tilde { x } _ { 1 : K } = \{ \tilde { x } _ { k } \} _ { k = 1 } ^ { K }$ denote these prototype embeddings. Transport is then computed between prototypes and concepts, which reduces redundancy and encourages region-level rather than patch-level alignment.

Vision–Language OT (unbalanced). A direct application of balanced OT enforces strict marginal conservation as in $\pi ( \mu , \nu ) ~ = ~ \left\{ \pi \geq 0 ~ | \pi { \bf 1 } = \mu , ~ { \bf 1 } ^ { \top } \pi = \right.$ $\nu \}$ , which forces all visual mass to match some concept and forces all concepts to receive mass. This assumption is routinely violated in vision–language alignment: multiple regions may correspond to a single concept (many-to-one), background regions often have no semantic counterpart, and some concepts may be absent from the image. Under strict conservation, background prototypes are pushed into arbitrary matches, and absent concepts must absorb visual mass, leading to hallucinated correspondences.

We therefore adopt unbalanced optimal transport (UOT), which relaxes these marginal constraints via KL penalties. Let the prototype–concept cost matrix be $( \mathbf { c } _ { \theta } ) _ { k j } = \mathbf { c } _ { \theta } \big ( f _ { \psi } ( \tilde { x } _ { k } ) , c _ { j } \big )$ . We compute the coupling as

$$
\pi_ {\theta} = \arg \min _ {\pi \geq 0} \Big \{\langle \mathbf {c} _ {\theta}, \pi \rangle + \varepsilon \operatorname{KL} \big (\pi \| \mu \otimes \nu \big) + \tau_ {1} \operatorname{KL} \big (\pi \mathbf {1} \| \mu \big) + \tau_ {2} \operatorname{KL} \big (\mathbf {1} ^ {\top} \pi \| \nu \big) \Big \},
$$

where $\boldsymbol { \mu } \in \mathbb { R } ^ { K }$ and $\boldsymbol { \nu } \in \mathbb { R } ^ { M }$ are reference marginals. The relaxed column marginal permits multiple prototypes to concentrate on the same concept, naturally modeling many-to-one alignment. The relaxed row marginal allows surplus visual mass to shrink at finite cost, so unmatched background regions need not be forced into incorrect matches.

Foreground-aware geometric suppression. Although UOT permits mass variation, background-dominant prototypes can still introduce weak, noisy associations. We use CLS attention as a simple foreground prior: prototypes with low attention are treated as background. Let $B \subseteq \{ 1 , \dots , K \}$ denote the resulting index set. We then increase the cost of transporting background prototypes by $\mathbf { c } _ { k j } ^ { \prime } = \mathbf { c } _ { k j } + \lambda _ { \mathrm { b g } } \mathbb { 1 } ( k \in \mathcal { B } )$ , where $\lambda _ { \mathrm { b g } }$ controls suppression strength. This makes background transport uniformly more expensive, and under the unbalanced formulation the optimizer prefers to shrink background mass rather than match it to arbitrary concepts. Using $\mathbf { c } ^ { \prime } { \mathrm { . } }$ , we compute the final plan $\pi _ { \mathrm { V L O T } } \in \mathbb { R } ^ { K \times M }$ , which defines the geometric coupling between visual prototypes and textual concepts.

The resulting coupling simultaneously (i) aggregates spatially coherent visual evidence, (ii) allows many-to-one semantic assignment, and (iii) suppresses unmatched background mass. This forms a geometry-aware cross-modal alignment tailored to vision–language concept modeling.

## 3.3 Learning Cross-Modal Geometry via Inverse OT

The geometric coupling defined in the previous section depends critically on the choice of cross–modal cost function $\mathbf { c } _ { \theta } ( x , c )$ . This function determines how visual evidence is measured against textual concepts and therefore shapes the induced transport plan. A central challenge is that visual and textual embeddings are produced by heterogeneous encoders and inhabit spaces with diferent geometric structures. Applying a fixed metric, such as cosine similarity or squared Euclidean distance, implicitly assumes that both modalities share a compatible embedding geometry. In practice, this assumption rarely holds, and mis-specified distances lead to distorted transport plans even under unbalanced formulations.

To this end, we learn the cross–modal geometry from data instead of prescribing it. Specifically, we adopt Inverse Optimal Transport (IoT) to infer a cost function whose induced transport plan agrees with empirical semantic correspondences. Unlike classical optimal transport, which fixes a cost and solves for an optimal plan, IoT reverses the direction: given observed couplings, it learns the cost under which those couplings become optimal.

For a given image, let $x _ { 1 : N } \subset \mathcal { X }$ denote visual patches and $c _ { 1 : M } \subset { \mathcal { C } }$ denote textual concepts. Suppose we have an empirical coupling $\hat { \pi } \in \mathbb { R } ^ { N \times M }$ derived from supervision. Given a parameterized cost matrix $( { \bf c } _ { \theta } ) _ { i j } \ = \ { \bf c } _ { \theta } ( x _ { i } , c _ { j } )$ , we compute the predicted plan $\pi _ { \theta }$ by solving the unbalanced OT problem defined in Sec. 3.2. IoT then learns $\mathbf { c } _ { \theta }$ so that π<sub>θ</sub> approximates ${ \hat { \pi } } ,$ , thereby inducing a geometry consistent with observed cross–modal structure.

![](images/87073d3d43314041c4d4e6bc334ee520f7c6051bd6058ca1ddd137d17f1ac3cd.jpg)  
Fig. 5: Training IoT cost functional. With object–component annotations datasets, we build ground-truth transport plans. A learnable multi-basis cost $\mathbf { c } _ { \theta }$ produces cost matrices, and unbalanced Sinkhorn plans are fitted to these labels to reflect true crossmodal distances between visual patches and text embeddings.

In implementation, visual features $\boldsymbol { x } \in \mathbb { R } ^ { d _ { v } }$ and textual features $c \in \mathbb { R } ^ { d _ { t } }$ are first mapped into a shared $d _ { p }$ -dimensional space. A learnable visual adapter $f _ { \psi }$ projects visual tokens. The cost function is expressed as a linear combination of K kernel bases $\varPhi = \{ \phi _ { k } \} _ { k = 1 } ^ { K }$ so that

$$
\mathbf {c} _ {\theta} (x, c) = \left\langle \boldsymbol {\theta}, \Phi \big (f _ {\psi} (x), c \big) \right\rangle ,
$$

where $\pmb \theta \in \mathbb { R } ^ { K }$ are learnable coeficients. The basis set includes squared Euclidean distance, Euclidean distance, cosine similarity, 1 − cos, dot product, magnitude diference, and multi-scale RBF kernels (see Supplementary Material). This parameterization allows the model to represent a rich family of cross–modal $\mathrm { g e - }$ ometries while retaining interpretability through weighted primitive distances.

To learn the cost, we minimize the discrepancy between the predicted plan $\pi _ { \theta }$ and the empirical coupling $\hat { \pi }$ . Direct Fenchel–Young alignment $\langle \mathbf { c } _ { \theta } , \pi _ { \theta } - \hat { \pi } \rangle$ can be unstable when correspondences are sparse or noisy. We therefore adopt an absolute-weighted formulation:

$$
\mathcal {L} _ {\mathrm{IoT}} = \left\| (\pi_ {\theta} - \hat {\pi}) \odot \mathbf {c} _ {\theta} \right\| _ {1, 1} + \lambda_ {1} \| \pmb {\theta} \| _ {1},
$$

where ⊙ denotes elementwise multiplication and $\lVert \cdot \rVert _ { 1 , 1 }$ is the entrywise $\ell _ { 1 }$ norm. When annotation masks M are available, we compute $\| M \odot \left( \pi _ { \theta } - { \hat { \pi } } \right) \odot \mathbf { c } _ { \theta } \| _ { 1 , 1 }$ . In practice, optimization is stabilized by detaching $\pi _ { \theta }$ during early epochs before enabling full gradient propagation through the UOT solver. The learned cost $\mathbf { c } _ { \theta ^ { \ast } }$ (together with $f _ { \psi ^ { * } } )$ defines a frozen cross–modal geometry. This geometry is subsequently used by geometric coupling module and forward semantic flow stage to produce consistent and structure-aware concept alignment.

## 3.4 From Discrete Coupling to Velocity-Based Concept Activation

The transport plan π<sub>VLOT</sub> provides a discrete correspondence between visual prototypes and textual concepts under the learned cross–modal geometry. However, this coupling remains a static assignment matrix. While it identifies which regions align with which concepts, it does not describe how semantic evidence evolves within the embedding space, nor does it provide a continuous mechanism for measuring concept compatibility beyond the solved transport instance. In other words, π<sub>VLOT</sub> captures where mass moves, but not how semantic structure is organized dynamically.

To obtain a geometry-consistent and generalizable representation, we lift discrete couplings into a continuous semantic flow. For prototype–concept pairs sampled according to π<sub>VLOT</sub>, define

$$
x _ {0} = f _ {\psi} (\tilde {x} _ {k}) \in \mathbb {R} ^ {d _ {p}}, \qquad x _ {1} = c _ {j} \in \mathbb {R} ^ {d _ {p}},
$$

and let $u = x _ { 1 } - x _ { 0 }$ denote the semantic displacement implied by optimal transport. Rather than treating alignment as a combinatorial mapping, we interpret these displacements as samples from an underlying continuous vector field.

We therefore learn a conditional velocity field $v _ { \phi } ( x , t ,$ cond) that approximates these transport-induced directions in continuous time. For $t \sim \mathcal { U } [ 0 , 1 ]$ ， define $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ and $u _ { t } = x _ { 1 } - x _ { 0 }$ , and the velocity field is trained using flow matching, i.e.,

$$
\mathcal {L} _ {\mathrm{FM}} = \mathbb {E} _ {(x _ {0}, x _ {1}) \sim \pi_ {\mathrm{VLOT}}, t} \big [ \| v _ {\phi} (x _ {t}, t, \mathrm{cond}) - u _ {t} \| ^ {2} \big ].\tag{1}
$$

This objective promotes a smooth vector field whose local directions reproduce the semantic transport dynamics defined by the learned geometry.

In classical flow-based generative models, the learned velocity field is integrated via $\dot { x } = v _ { \phi } ( x _ { t } , t )$ to recover the terminal state $x _ { 1 }$ , requiring numerical ODE solvers at inference. For Concept Bottleneck Models, such reconstruction is unnecessary. Our objective is to estimate concept activations, not to generate textual embeddings. From the theory of ordinary diferential equations, under standard Lipschitz regularity, a velocity field uniquely determines trajectories; therefore, local velocity agreement already implies path consistency without explicit integration. Measuring instantaneous motion thus sufices to evaluate semantic alignment. π<sub>VLOT</sub> is used only during training to define semantic displacements; inference depends solely on the learned velocity field and projected embeddings.

Further theoretical support arises from Schrödinger bridge formulations, the stochastic analogue of optimal transport flows. For Brownian motion conditioned on endpoints $X _ { 0 } = x _ { 0 }$ and $X _ { 1 } = x _ { 1 }$ , the interpolating process admits the decomposition $X _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } + B _ { t }$ , where $B _ { t }$ is a Brownian bridge with conditional variance $\sigma _ { t } ^ { 2 } = t ( 1 - t ) \ [ 2 , 3 8 ]$ . The following lemma formalizes its midpoint property.

Lemma 1. Let $X _ { t }$ be the Schrödinger bridge on $[ 0 , 1 ]$ for Brownian motion with endpoints $X _ { 0 } = x _ { 0 }$ and $X _ { 1 } = x _ { 1 }$ . Then $X _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } + B _ { t }$ , where $B _ { t }$ is a Brownian bridge with $\sigma _ { t } ^ { 2 } ( B _ { t } ) = t ( 1 - t )$ . Thus, the variance is uniquely maximized at $\begin{array} { r } { t = \frac { 1 } { 2 } . } \end{array}$ , so the midpoint is the most uncertain (and therefore most informative) stage for evaluating alignment.

The lemma implies that the midpoint concentrates maximal conditional uncertainty along the transport trajectory. Evaluating velocity consistency at this stage captures the richest semantic signal while regularizing trajectories toward smooth, geometry-consistent flows.

We therefore derive concept activations directly from instantaneous velocity agreement at $t = 0 . 5$ . For a projected visual prototype $x _ { 0 } = f _ { \psi ^ { * } } ( \tilde { x } _ { k } )$ and each concept embedding $x _ { 1 , j }$ in the concept set, define the reference displacement $u ^ { ( k , j ) } = x _ { 1 , j } - x _ { 0 }$ with $\mathbf { x } _ { 1 / 2 } ^ { ( k , j ) } = \textstyle { \frac { 1 } { 2 } } ( x _ { 0 } + x _ { 1 , j } )$ . The predicted midpoint velocity is

$$
v _ {\text { pred }} ^ {(k, j)} = v _ {\phi} \left(x _ {1 / 2} ^ {(k, j)}, 0. 5, \text { cond }\right).
$$

We measure alignment via $S _ { k , j } = - { \left\| { v _ { \mathrm { p r e d } } ^ { \left( k , j \right) } - u ^ { \left( k , j \right) } } \right\| } ^ { 2 }$ , which quantifies how closely the predicted semantic motion matches the ideal displacement toward concept $j$ under the learned transport geometry. Unlike classical CBMs that rely on static feature similarity, this formulation evaluates velocity similarity: activation increases when predicted motion aligns with the concept direction.

Concept activations are obtained by aggregating the strongest prototype responses $\begin{array} { r } { a _ { j } = \frac { 1 } { K } \sum _ { k \in \mathrm { T o p K } ( S _ { \cdot , j } ) } S _ { k , j } } \end{array}$ . This readout requires neither ODE integration nor solving an optimal transport problem at inference. Instead, it derives concept evidence from local motion consistency within the learned cross–modal geometry, providing a dynamic yet computationally eficient alternative to static similarity-based bottlenecks. Finally, the normalized activation vector is then passed through a linear classifier to predict class logits

$$
\hat {y} = f _ {\mathrm{cls}} (\text { LayerNorm } (\mathbf {a})).\tag{2}
$$

## 4 Experimental results

To assess the efectiveness of OTF-CBM, we evaluated from three perspectives:

i) Predictive accuracy and concept faithfulness on standard image classification benchmarks.

ii) Quality of cross–modal semantic flows, assessing how learned velocities transport visual evidence toward concept embeddings under the learned geometry.

iii) Robustness and object centricity under cross-modal semantic flows shifts, using controlled background replacement and distribution changes to test that predictions rely on object evidence rather than context.

## 4.1 Datasets and Metrics

Evaluating CBM Performance. We evaluate on five image classification datasets spanning diferent granularities and domain complexity. CUB-200-2011 [42] (11,788 bird images, 200 subcategories) and AwA2 [43] (37,322 animal images, 50 classes) are standard for interpretable concept learning. ImageNet-1K [11] and CIFAR-100 [21] serve as large-scale benchmarks with diverse concepts and richer category structure. The scene-centric Places365 [48] probes robustness under contextual bias, since discriminative cues often lie in background rather than object appearance. Across all datasets, we report classification accuracy as the primary metric, assessing downstream performance and the efectiveness of the learned concept bottleneck.

Evaluating Cross-Modal Semantic Flows. We report four metrics to assess the learned geometry and dynamics in our approach. Specifically, Transport Reconstruction Error (TRE) measures how well the predicted transport plan recovers ground truth patch–concept correspondences, reflecting the fidelity of the learned cost geometry. Velocity Mean Squared Error (VMSE) quantifies the discrepancy between predicted and theoretical semantic velocities along visual–to–textual trajectories. Mean Cosine Ratio (MCR) evaluates directional consistency of the predicted flow field. Negative Pair Error (NPE) captures the proportion of mismatched or reversed flow directions. Together, these metrics assess dynamic cross–modal flows beyond static embedding similarity. We present the definition of metrics for cross-modal transport and flow learning evaluation.

1. TRE [22, 26]. Measures transport fidelity by comparing predicted and ground truth couplings by TRE $\vdots = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \left\| \pi _ { \theta } ^ { ( b ) } - \hat { \pi } ^ { ( b ) } \right\| _ { 1 }$

2. VMSE [23]. Measures the closeness between predicted instantaneous motion and ideal semantic displacement by

$$
\text { VMSE } := \frac {1}{N} \sum_ {i = 1} ^ {N} \left\| v _ {\phi} (x _ {\tau_ {i}} ^ {(i)}, \tau_ {i} \mid \boldsymbol {h} ^ {(i)}) - u ^ {(i)} \right\| _ {2} ^ {2}.\tag{3}
$$

3. MCR [25]. Quantifies directional consistency between predicted flow and target semantic direction.

$$
\mathrm{MCR} := \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {\left\langle v _ {\phi} \big (x _ {\tau_ {i}} ^ {(i)} , \tau_ {i} \mid \boldsymbol {h} ^ {(i)} \big) , u ^ {(i)} \right\rangle}{\left\| v _ {\phi} \big (x _ {\tau_ {i}} ^ {(i)} , \tau_ {i} \mid \boldsymbol {h} ^ {(i)} \big) \right\| _ {2} \| u ^ {(i)} \| _ {2}}.\tag{4}
$$

4. NPE [37]. Reports the fraction of pairs with reversed alignment.

$$
\mathrm{NPE} := \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \Big [ \left\langle v _ {\phi} \big (x _ {\tau_ {i}} ^ {(i)}, \tau_ {i} \mid \boldsymbol {h} ^ {(i)} \big), u ^ {(i)} \right\rangle <   0 \Big ].\tag{5}
$$

## 4.2 Implementation Details

We use pre-trained DINOv2 $V i T – L / / 4 \ [ 2 8 ]$ as the visual encoder $E _ { v }$ with $d _ { v } =$ 1024 and the pre-trained CLIP text encoder $E _ { t }$ with $d _ { t } = 7 6 8 \ [ 3 1 ]$ . All images are resized to $2 2 4 \times 2 2 4$ . A visual adapter $f _ { \psi }$ (three-layer MLP with ReLU) projects features to the shared space with $d _ { p } = 7 6 8$ . Optimization uses AdamW with learning rate $1 \times 1 0 ^ { - 4 }$ (0.1 decayeach epoch), weight decay $1 \times 1 0 ^ { - 5 }$ , batch size 64, and 20 training epochs. All experiments are implemented in PyTorch and run on two RTX 3090 GPUs. Additional hyperparameter details are in Supplementary Material.

## 4.3 Cost Learning Results

Learned Cross-Modal Cost Functions. To characterize the inverse optimal transport (IoT) module, we examine the basis weights of the learned cost $\mathbf { c } _ { \theta } ( x , c )$ . The cost is a linear combination of 18 candidate bases, with the complete list given in the Supplementary Material. IoT experiment result reports the five largest contributors: squared angular distance, dot product similarity, and three hybrid terms (Dot–RBF, inverse quadratic, and root exponential). Together, these components capture local correlation and global semantic separation. The training objective is nonconvex, so diferent initializations can lead to local minimizer. In practice, runs that achieve similar validation loss induce nearly identical cross-modal geometries and alignments, indicating stable and interpretable learned metric.

<table><tr><td>Cost Function</td><td>TRE ↓</td><td>Remarks</td></tr><tr><td> $\mathcal{W}_{2}^{2}$ </td><td>2.371</td><td>Standard metric</td></tr><tr><td>Cosine distance</td><td>1.826</td><td>Alignment baseline</td></tr><tr><td>IoT (ours)</td><td>0.824</td><td>Data-driven geometry</td></tr></table>

Table 2: TRE across diferent costs.

Comparison with Conventional Metrics. We compare the learned IoT cost with two baselines: the Wasserstein distance and cosine similarity, both computed in the shared space. TRE measures how well the induced plan recovers annotated patch to concept couplings. As shown in Table 2, IoT attains the lowest TRE and most closely matches the visual–textual correspondences, and the geometry strengthens downstream flow matching and concept inference.

<table><tr><td rowspan="2">Method</td><td colspan="5">Classification Accuracy (↑)</td></tr><tr><td>ImageNet</td><td>CUB</td><td>CIFAR100</td><td>AWA2</td><td>Places365</td></tr><tr><td>Vanilla-CBM</td><td>79.17</td><td>78.32</td><td>80.04</td><td>93.15</td><td>44.80</td></tr><tr><td>CEM</td><td>81.29</td><td>80.47</td><td>81.23</td><td>95.92</td><td>45.01</td></tr><tr><td>LaBo</td><td>82.93</td><td>81.30</td><td>84.10</td><td>96.92</td><td>45.43</td></tr><tr><td>SparseCBM</td><td>82.85</td><td>82.07</td><td>84.75</td><td>95.56</td><td>46.24</td></tr><tr><td>CoopCBM</td><td>82.73</td><td>82.10</td><td>84.66</td><td>97.08</td><td>48.21</td></tr><tr><td>DOT-CBM</td><td>83.84</td><td>85.39</td><td>85.83</td><td>96.83</td><td>50.65</td></tr><tr><td>OURS</td><td>85.62(+1.78)</td><td>89.92(+4.53)</td><td>90.21(+4.38)</td><td>98.88(+2.05)</td><td>55.13(+4.48)</td></tr></table>

Table 3: Classification performance across five datasets.

## 4.4 CBM + OT-FM Performance

We compare our proposed OTF-CBM with classical and recent concept bottleneck models (CBMs). Specifically, we include Vanilla-CBM [20], which introduced the explicit concept layer and three training paradigms (independent, sequential, joint). Our two-stage optimization corresponds to the independent strategy reported as most interpretable in [20]. We also evaluate CEM [17], LaBo [45], SparseCBM [33], CoopCBM [35], and DOT-CBM [44]. To eliminate confounding factors, all baselines are re-implemented with the same pre-trained ViT backbone and concept embeddings from the Label-Free CBM pipeline, ensuring consistent visual and conceptual representations. Results in Table 3 show that our model attains the best performance across five benchmarks, with average top-1 accuracy gains of +1.78% (ImageNet), +4.53% (CUB), +4.38% (CIFAR-100), +2.05% (AwA2), and +4.48% (Places365).

Evaluating Cross-Modal Flow Modeling. We further assess the efectiveness of our cross-modal semantic flow components. Table 4 compares four configurations: Static OT Alignment (no flow learning), OT Flow Matching (linear flow), IoT + OT Flow Matching (learned flow under balanced transport), and our full IoT + UOT Flow Matching (learned many-to-one flow). Our method attains the lowest velocity mean squared error (VMSE ↓), highest mean cosine ratio (MCR ↑), and smallest negative pair error (NPE ↓), indicating that the learned velocity field under the unbalanced transport geometry captures more faithful and directionally consistent cross-modal dynamics. The gains VMSE: 1.782 → 1.000, MCR: 0.884 → 0.999, and NPE: 0.197 → 0.021 show that modeling many-to-one semantic flows enhances geometric fidelity and stabilizes the transport trajectory across modalities.

## 4.5 Ablations

Table 5 reports ablation results on five benchmarks. Starting from a vanilla CBM baseline, replacing similarity-based concept inference with classical OT consistently improves performance, indicating that explicit cross-modal coupling is beneficial. Our proposed Vision–Language OT (VLOT) further strengthens alignment by accounting for modality-specific structure.

<table><tr><td>Method</td><td>Semantic Flow Modeling</td><td>VMSE ↓</td><td>MCR ↑</td><td>NPE ↓</td></tr><tr><td>Static OT Alignment</td><td>static matching</td><td>2.467</td><td>0.739</td><td>0.434</td></tr><tr><td>OT Flow Matching</td><td>Linear flow</td><td>2.001</td><td>0.803</td><td>0.299</td></tr><tr><td>IoT + OT Flow Matching</td><td>Learned flow</td><td>1.782</td><td>0.884</td><td>0.197</td></tr><tr><td>IoT + UOT Flow Matching (ours)</td><td>Learned flow (many-to-one)</td><td>1.000</td><td>0.999</td><td>0.021</td></tr></table>

Table 4: Evaluation of semantic flow modeling. The proposed IoT+UOT formulation yields the most accurate and directionally consistent visual-to-concept flows.

<table><tr><td rowspan="2">Method</td><td colspan="5">Classification Accuracy</td></tr><tr><td>ImageNet</td><td>CUB</td><td>CIFAR100</td><td>AWA2</td><td>PLACE365</td></tr><tr><td>Vanilla-CBM</td><td>79.17</td><td>78.32</td><td>80.04</td><td>93.15</td><td>44.80</td></tr><tr><td>with classical OT</td><td>80.42</td><td>80.31</td><td>85.23</td><td>94.38</td><td>47.7</td></tr><tr><td>with Vision-Language OT</td><td>82.68</td><td>84.31</td><td>87.44</td><td>95.61</td><td>50.59</td></tr><tr><td>+Inverse OT cost</td><td>84.87</td><td>87.44</td><td>89.09</td><td>96.56</td><td>52.99</td></tr><tr><td>+Semantic Flow</td><td>85.62</td><td>89.92</td><td>90.21</td><td>98.88</td><td>55.13</td></tr></table>

Table 5: Ablation studies on key components. Each design improves classification performance, interpretability, or eficiency, and their combination yields the best overall performance.

Learning the transport cost via inverse OT brings additional gains, highlighting the importance of accurately modeling cross-modal geometry. Finally, introducing the semantic flow mechanism yields the best overall performance, demonstrating that converting discrete transport plans into a continuous formulation improves robustness of concept inference. Overall, each component contributes cumulatively to the final results. Module hyperparameter choice ablations are provided in the Supplementary Material. We observe that the proposed framework is stable across a wide range of reasonable settings, with classification performance varying by less than ±1% in normal regimes.

## 4.6 Robustness and Generalization

Traditional CBMs often tie concepts to background or context, creating spurious correlations and weak OOD generalization. To probe this, we run a backgroundperturbation test: objects are segmented (e.g., birds on CUB), the background is replaced with randomly recolored noise, and the foreground is left intact. This creates an OOD shift that removes contextual bias while preserving semantic content; a concept-grounded model should retain accuracy.

Table 6 shows that OTF-CBM remains stable under this shift. On CUB, OOD accuracy improves by +14.5% over the strongest baseline (DOT-CBM); on Places365, the gain is +8.4%. The ID–OOD gap shrinks substantially, indicating that cross-modal flow modeling anchors reasoning on object evidence rather than background shortcuts, yielding robust generalization to unseen environments.

<table><tr><td rowspan="2">Method</td><td colspan="2">CUB</td><td colspan="2">PLACES365</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>Vanilla-CBM</td><td>78.3</td><td>30.1</td><td>44.8</td><td>21.3</td></tr><tr><td>CEM</td><td>80.4</td><td>40.2</td><td>45.0</td><td>22.9</td></tr><tr><td>LaBo</td><td>81.3</td><td>45.3</td><td>45.4</td><td>27.4</td></tr><tr><td>SparseCBM</td><td>82.0</td><td>44.8</td><td>46.2</td><td>27.3</td></tr><tr><td>CoopCBM</td><td>82.1</td><td>48.3</td><td>48.2</td><td>31.6</td></tr><tr><td>DOT-CBM</td><td>85.3</td><td>67.5</td><td>50.6</td><td>42.1</td></tr><tr><td>OURS</td><td>89.9(+4.6)</td><td>82.0(+14.5)</td><td>55.1(+4.5)</td><td>50.5(+8.4)</td></tr></table>

Table 6: Robustness evaluation under SAM-based background perturbations. Our method achieves the highest robustness across both in-domain (ID) and out-of-domain (OOD) conditions.

## 5 Conclusion

We introduced OTF-CBM for cross-modal concept matching. A data-driven cost learned by inverse optimal transport corrects metric mismatch, and unbalanced OT handles many-to-one correspondences and background mass. On this geometry, a trained velocity field replaces static cosine similarity with midpoint velocity alignment, yielding eficient inference and spatially grounded activations. Experiments show consistent gains in classification accuracy, interpretability, and out-of-distribution generalization over prior CBMs. More importantly, our goal is not merely to improve classification accuracy in CBMs. Instead, we hope this work provides evidence that a potential performance bottleneck in vision-language models may stem from the joint optimization of heterogeneous modalities. Learning well-structured representations for each modality in its own embedding space before establishing accurate visual-text correspondences may ofer a more efective solution to this bottleneck. We hope this perspective will inspire future research to further explore the combination of optimal transport and flow matching for more principled cross-modal representation learning.

## Acknowledgement

This work was supported by grants from the Natural Science Foundation of Shanxi Province (2024JCJCQN-66).

## References

1. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571 (2022) 5

2. Chen, Y., Georgiou, T.T.: Stochastic bridges of linear systems. IEEE Transactions on Automatic Control 61(2), 526–531 (2015) 11

3. Chen, Y., Georgiou, T.T., Pavon, M.: Stochastic control liaisons: [r]ichard [s]inkhorn meets [g]aspard [m]onge on a [s]chrödinger bridge. Siam Review 63(2), 249–313 (2021) 5

4. Chiu, W.T., Wang, P., Shafto, P.: Discrete probabilistic inverse optimal transport. In: International Conference on Machine Learning. pp. 3925–3946. PMLR (2022) 2

5. Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.X.: Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816 (2016) 3

6. Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.X.: Scaling algorithms for unbalanced optimal transport problems. Mathematics of computation 87(314), 2563– 2609 (2018) 5

7. Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.X.: Unbalanced optimal transport: Dynamic and Kantorovich formulations. Journal of Functional Analysis 274(11), 3090–3123 (2018) 5

8. Courty, N., Flamary, R., Habrard, A., Rakotomamonjy, A.: Joint distribution optimal transport for domain adaptation. In: Proceedings of the European Conference on Computer Vision, Munich, Germany. pp. 8–14 (2018) 5

9. Cuturi, M.: Sinkhorn distances: Lightspeed computation of optimal transport. Advances in Neural Information Processing Systems 26 (2013) 5

10. De Plaen, H., De Plaen, P.F., Suykens, J.A., Proesmans, M., Tuytelaars, T., Van Gool, L.: Unbalanced optimal transport: A unified framework for object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3198–3207 (2023) 3

11. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition. pp. 248–255. Ieee (2009) 12

12. Dong, A., Chen, C., Georgiou, T.T.: Network learning with directional sign patterns. In: 2024 IEEE 63rd Conference on Decision and Control (CDC). pp. 3924– 3929. IEEE (2024) 5

13. Dong, A., Chen, Y., Johansson, K.H., Karlsson, J.: Meanflow meets control: Scaling sampled-data control for swarms. arXiv e-prints pp. arXiv–2603 (2026) 5

14. Dong, A., Georgiou, T.T., Tannenbaum, A.: Data Assimilation for Sign-indefinite Priors: A generalization of Sinkhorn’s algorithm. Automatica 177, 112283 (2025) 5

15. Dong, A., Georgiou, T.T., Tannenbaum, A.: Negative probabilities and the Sinkhorn Algorithm: Promotion/Inhibition interactions in networks. EDITORIAL COMMITTEE p. 61 (2025) 5

16. Dong, A., Stephanovitch, A., Georgiou, T.T.: Monge–Kantorovich optimal transport through constrictions and flow-rate constraints. Automatica 160, 111448 (2024) 5

17. Espinosa Zarlenga, M., Barbiero, P., Ciravegna, G., Marra, G., Giannini, F., Diligenti, M., Shams, Z., Precioso, F., Melacci, S., Weller, A., et al.: Concept embedding models: Beyond the accuracy-explainability trade-of. Advances in Neural Information Processing Systems 35, 21400–21413 (2022) 14

18. Fatras, K., Séjourné, T., Flamary, R., Courty, N.: Unbalanced minibatch optimal transport; applications to domain adaptation. In: International conference on machine learning. pp. 3186–3197. PMLR (2021) 5

19. Galichon, A.: Optimal transport methods in economics. Princeton University Press (2016) 5

20. Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P.: Concept bottleneck models. In: International Conference on Machine Learning. pp. 5338–5348. PMLR (2020) 1, 3, 4, 14

21. Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features from tiny images (2009) 12

22. Li, R., Ye, X., Zhou, H., Zha, H.: Learning to match via inverse optimal transport. Journal of Machine Learning Research 20(80), 1–37 (2019) 2, 5, 12

23. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022) 3, 5, 12

24. Liu, Q., Yin, X., Yuille, A., Brown, A., Singh, M.: Flowing from words to pixels: A noise-free framework for cross-modality evolution. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2755–2765 (2025) 5

25. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022) 12

26. Ma, S., Sun, H., Ye, X., Zha, H., Zhou, H.: Learning cost functions for optimal transport. arXiv preprint arXiv:2002.09650 (2020) 2, 12

27. Oikarinen, T., Das, S., Nguyen, L.M., Weng, T.W.: Label-free concept bottleneck models. arXiv preprint arXiv:2304.06129 (2023) 2, 4

28. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023) 13

29. Peyré, G., Cuturi, M.: Computational optimal transport: With applications to data science. Now Foundations and Trends (2019) 2, 5

30. Rachev, S.T., Rüschendorf, L.: Mass Transportation Problems: Volume I: Theory. Springer (1998) 5

31. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763. PmLR (2021) 1, 2, 3, 4, 13

32. Séjourné, T., Peyré, G., Vialard, F.X.: Unbalanced optimal transport, from theory to numerics. Handbook of Numerical Analysis 24, 407–471 (2023) 3, 5

33. Semenov, A., Ivanov, V., Beznosikov, A., Gasnikov, A.: Sparse concept bottleneck models: Gumbel tricks in contrastive learning. arXiv preprint arXiv:2404.03323 (2024) 14

34. Shang, C., Zhou, S., Zhang, H., Ni, X., Yang, Y., Wang, Y.: Incremental residual concept bottleneck models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11030–11040 (2024) 4

35. Sheth, I., Ebrahimi Kahou, S.: Auxiliary losses for learning generalizable conceptbased models. Advances in Neural Information Processing Systems 36, 26966– 26990 (2023) 14

36. Srivastava, D., Yan, G., Weng, L.: Vlg-cbm: Training concept bottleneck models with vision-language guidance. Advances in Neural Information Processing Systems 37, 79057–79094 (2024) 1, 2

37. Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y.: Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482 (2023) 5, 13

38. Tong, A., Malkin, N., Fatras, K., Atanackovic, L., Zhang, Y., Huguet, G., Wolf, G., Bengio, Y.: Simulation-free schr\" odinger bridges via score and flow matching. arXiv preprint arXiv:2307.03672 (2023) 11

39. Vandenhirtz, M., Laguna, S., Marcinkevičs, R., Vogt, J.: Stochastic concept bottleneck models. Advances in Neural Information Processing Systems 37, 51787–51810 (2024) 1, 3

40. Villani, C.: Topics in optimal transportation, vol. 58. American Mathematical Soc. (2021) 5

41. Villani, C., et al.: Optimal transport: Old and new, vol. 338. Springer (2009) 2

42. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S., et al.: The caltech-ucsd birds-200-2011 dataset. Tech. rep., Technical Report CNS-TR-2011-001, California Institute of Technology (2011) 12

43. Xian, Y., Lampert, C.H., Schiele, B., Akata, Z.: Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly. IEEE Transactions on Pattern Analysis and Machine Intelligence 41(9), 2251–2265 (2018) 12

44. Xie, Y., Zeng, Z., Zhang, H., Ding, Y., Wang, Y., Wang, Z., Chen, B., Liu, H.: Discovering fine-grained visual-concept relations by disentangled optimal transport concept bottleneck models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 30199–30209 (2025) 4, 14

45. Yang, Y., Panagopoulou, A., Zhou, S., Jin, D., Callison-Burch, C., Yatskar, M.: Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19187–19197 (2023) 1, 2, 4, 14

46. Yue, A., Dong, A., Xu, H.: OAT-FM: Optimal Acceleration Transport for Improved Flow Matching. arXiv preprint arXiv:2509.24936 (2025) 5

47. Yuksekgonul, M., Wang, M., Zou, J.: Post-hoc concept bottleneck models. arXiv preprint arXiv:2205.15480 (2022) 1, 3, 4

48. Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., Torralba, A.: Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence 40(6), 1452–1464 (2017) 12