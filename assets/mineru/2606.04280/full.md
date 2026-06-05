# The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning

Justinas Zaliaduonis 1 2 Patrick Putzky 3 Till Richter 1 4 Sergios Gatidis 5

# Abstract

Contrastive learning has become a leading paradigm for self-supervised representation learning, yet the conditions under which it recovers meaningful latent geometry remain incompletely understood. We develop a measure-theoretic framework formalizing the diversity condition, a support requirement on positive-pair sampling that is necessary for isometric latent recovery. We show that the standard full-support von Mises-Fisher setting implies the satisfaction of the diversity condition and as a consequence global contrastive loss minimizers recover latent geometry up to orthogonal transformation, while restricted conditionals can make non-orthogonal maps attain strictly lower asymptotic contrastive loss. We introduce a support-corrected Information Noise Contrastive Estimation (InfoNCE) variant as a theoretical fix: this correction makes orthogonal latent space recovery achievable but does not uniquely select it. Experiments on synthetic benchmarks validate the identifiability predictions, and CIFAR-10 experiments are consistent with the qualitative prediction that architectural inductive bias becomes more important when sampling diversity is limited. Together, our results clarify how sampling mechanisms and encoder inductive bias interact in contrastive representation learning.

\*Equal contribution 1Technical University of Munich, Munich, Germany 2Stanford University, Stanford, USA 3Merantix Momentum GmbH, Berlin, Germany 4Helmholtz Munich, Munich, Germany 5Department of Radiology, Stanford University School of Medicine, Stanford, USA. Correspondence to: Justinas Zaliaduonis <justinas.zaliaduonis@gmail.com>, Patrick Putzky <patrick.putzky@merantix-momentum.com>, Till Richter <till.richter@helmholtz-munich.de>, Sergios Gatidis <sgatidis@stanford.edu>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

# 1. Introduction

The machine learning community has long envisioned methods that turn vast amounts of unlabeled data into dense, robust, and reusable representations useful for many different downstream tasks such as classification, regression, and search. Contrastive learning (CL) has emerged as a successful technique for achieving this goal, which in recent years has led to advances in language (Jaiswal et al., 2021), vision (Chen et al., 2020), video (Zhao et al., 2024), and multimodal (Radford et al., 2021) domains.

The vast range of applications in scientific fields like Biology (Richter et al., 2025; Bahrami et al., 2025), Physics (Cy et al., 2023; Wilkinson et al., 2025), and Climate Science (Ballard, 2022; Liu et al., 2026) has made CL one of the most widely adopted unsupervised learning methods (Uelwer et al., 2023). However, despite its empirical success, the precise mechanisms driving contrastive learning remain only partially understood. This gap in theoretical understanding results in heuristic-driven development, inefficient use of computational resources, and design choices that may not fully exploit the method’s potential. In this work, we seek to move beyond intuitive understanding of CL and provide a rigorous framework to reason about its regimes of success and failure.

One approach to explain the learning mechanisms posits that CL induces data representations invariant to nuisance factors (Dangovski et al., 2022; Liu et al., 2025; Poudel et al., 2022). However, this framework does not address which factors in the data are nuisance, nor how the choice of data augmentations implicitly determine this partition. Moreover, the choice of nuisance factors, often referred to as the style-content decomposition (von Kugelgen et al. ¨ , 2021), can be detrimental to downstream tasks: depending on the intended use of the learned representations, factors deemed “nuisance” by the contrastive objective may carry discriminative information necessary for a specific downstream task. We refer to this approach as the Invariance Explanation.

An alternative direction reasons that CL recovers the “true” generating factors of the data (Ji et al., 2023; Kirchhof et al., 2023; Sandilya et al., 2025). This approach assumes that data lies on a high-dimensional manifold but possesses a low-dimensional latent structure, and that a generative process maps this low-dimensional representation to the observed high-dimensional data. In this direction, CL recovers the latent structure. Figure 1 illustrates this setup. We refer to this approach as the Recovery Explanation.

![](images/61cfcefd20a983d6c46b2f6966904f1f83d6e521f17868c9094aec4ff2cc12ab.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Latent Space (Z)"] -->|g| B["Data Space (X)"]
    B -->|f1| C["Recovered Latent Space (Z')"]
    D["Latent Space (Z)"] -->|g| E["Data Space (X)"]
    E -->|f1| F["Recovered Latent Space (Z')"]
    G["Latent Space (Z)"] -->|g| H["Data Space (X)"]
    H -->|f2| I["Recovered Latent Space (Z')"]
```
</details>

Figure 1. Overview of contrastive learning and the role of sampling diversity and inductive bias. The generative process $g$ maps latent variables to observations, and the encoder $f$ learns to recover the latent structure. Here $f _ { 1 }$ denotes a low inductive bias encoder (e.g., MLP) and $f _ { 2 }$ a high inductive bias encoder (e.g., a model of the inverse process). Orange dot indicates the anchor point; green dots are co-occurring (positive) samples. Border colors on images match their latent positions. (a) Diversity holds: $f _ { 1 }$ recovers geometry. (b) Diversity violated (blue band): $f _ { 1 }$ fails. (c) Diversity violated: $f _ { 2 }$ recovers the latent structure despite restricted sampling diversity.

In this work, we argue that the Recovery Explanation provides a more epistemologically complete account of the mechanisms underlying contrastive learning. We introduce a constraint on the conditional law of the latent space, $P _ { \tilde { Z } | z } ,$ which we term the diversity condition, and show that it is necessary for recovering the latent space up to an isometric (distance-preserving) transformation. Building on the notion of latent space recovery introduced by (Zimmermann et al., 2021), we separate the classical full-support vMF setting from the practically relevant setting where sampling is restricted. Our proof of the full-support case uses a probabilistic Mazur-Ulam argument (Zaliaduonis & Gatidis, 2026) and does not require differentiability of the recovery map. Our main result shows that violated diversity can make geometry-distorting solutions preferable, and that correcting the support mismatch restores, but does not uniquely select, geometry-preserving minimizers.

The real-world sampling mechanisms often violate the diversity condition and the latent structure is typically unknown a priori. We propose a generalized InfoNCE objective to address this gap and prove that this adjustment allows geometry-preserving latent spaces to be among the optimal solutions of the objective.

We empirically study the qualitative implications of the theory on CIFAR-10 (Krizhevsky, 2009). Here, we examine how architectural inductive bias affects representation quality under different augmentation regimes. In summary, our contributions are:

• Formalize a measure-theoretic diversity condition on latent space sampling, distinguish it from the fullsupport vMF assumption used in prior identifiability results, and analyze what fails when this condition is violated.   
• Prove that violated diversity can make non-orthogonal recovery maps attain strictly lower asymptotic contrastive loss than any orthogonal map, and show that a support-corrected InfoNCE objective makes isometric embeddings achievable.   
• Empirically validate theoretical predictions on synthetic datasets and examine their qualitative implications on CIFAR-10, demonstrating how sampling strategies and encoder inductive bias jointly determine representation quality.   
• Distill design guidance for choosing augmentation

pipelines and encoder architectures from the theoryexperiment alignment.

# 2. Related Work

Identifiability in Contrastive Learning. The question of identifiability in unsupervised learning has a long history, beginning with classical results in Independent Component Analysis (ICA). Early work established that linear mixtures of non-Gaussian independent sources can be identified up to permutation and scaling, laying the theoretical foundation for blind source separation (Comon, 1994). Subsequent developments provided practical algorithms and characterized the fundamental limits of linear ICA (Hyvarinen & ¨ Oja, 2000). The broader goal of learning disentangled representations that capture meaningful factors of variation was later articulated (Bengio et al., 2013), though it was subsequently proved that unsupervised disentanglement is impossible without inductive biases on both the model and data (Locatello et al., 2019).

In the context of contrastive learning, a formal definition of latent space identifiability was introduced, with proofs that InfoNCE recovers ground-truth factors up to orthogonal transformation (Zimmermann et al., 2021). This analysis employs von Mises-Fisher conditionals and cross-entropy asymptotics on spherical manifolds. Connections between contrastive objectives and nonlinear ICA have also been established, demonstrating identifiability through temporal structure (Hyvarinen & Morioka ¨ , 2016).

Augmentations and Invariance. A complementary line of work examines how data augmentations shape learned representations by defining which factors should be preserved versus discarded. A content-style framework formalizes this intuition, proving that augmentation-based contrastive learning achieves block-identifiability of the invariant content partition under a latent variable model with nontrivial statistical and causal dependencies (von Kugelgen ¨ et al., 2021). The InfoMin principle proposes that optimal views for contrastive learning should share minimal mutual information while retaining task-relevant information, thereby discarding nuisance factors (Tian et al., 2020). A causal interpretation shows that data augmentations can be viewed as interventions on style variables, motivating an explicit invariance regularizer that enforces invariant prediction of proxy targets across augmentations (Mitrovic et al., 2021). Wen & Li (2021) study a complementary featurelearning regime: they analyze gradient-based learning in finite ReLU networks and show how suitable augmentations can decouple desired sparse features from nuisance dense features. Our analysis asks a different question, namely whether the asymptotic contrastive objective recovers latent geometry up to isometry, and how architectural inductive bias compensates when sampling diversity is insufficient.

Architectural Bias and Extensions. The role of encoder architecture in contrastive learning has received increasing attention. Recent work analyzes how architectural constraints influence the geometry of learned representations (HaoChen & Ma, 2023), while theoretical frameworks have been extended to multimodal settings (Tschannen et al., 2023). Other contributions address gaps between theoretical assumptions and practical implementations (Rusak et al., 2025).

Empirical Methods. Our theoretical analysis builds upon empirically successful methods including SimCLR (Chen et al., 2020), Contrastive Predictive Coding (CPC) (van den Oord et al., 2019), and VICReg (Bardes et al., 2022). These frameworks demonstrate the practical efficacy of contrastive objectives across vision, language, and multimodal domains.

# 3. Preliminaries

# 3.1. Contrastive Learning Framework

Following (Zimmermann et al., 2021), we analyze contrastive learning using tools from Nonlinear Independent Component Analysis (ICA) (Hyvarinen et al. ¨ , 2019). We consider an encoder $f : \mathcal { X } \to \mathcal { Z }$ mapping observations from the data space $\mathcal { X } \subset \mathbb { R } ^ { n }$ to a latent representation space $\mathcal { Z } \subset \mathbb { R } ^ { k }$ , where $k < n .$ We model the data as generated by an injective mapping $g : { \mathcal { Z } }  { \mathcal { X } }$ from a lower-dimensional latent manifold whose coordinates represent statistically independent factors (Figure 1).

Contrastive learning aims to learn encoder parameters θ by discriminating between co-occurring signals $x , \tilde { x } \in$ X (Chen et al., 2020). These signals may arise from natural mechanisms (e.g., different modalities of the same scene) or synthetic transformations (e.g., image augmentations). We define a view-generating function $\tau : \mathcal { X }  \mathcal { X }$ that produces related views $\tilde { x } = \tau ( x )$ , and aim to learn an encoder such that the underlying latent factors are identified.

# 3.2. Sampling Mechanism

Since the data is generated via the injective mapping g, observations satisfy $x = g ( z )$ and $\tilde { x } = \tau ( g ( z ) ) = g ( \tilde { z } )$ . As view generation is typically non-deterministic, we treat it stochastically through the conditional law $P _ { \tilde { X } \mid x }$ . This formulation connects to the latent dynamics $P _ { \tilde { Z } | z }$ via the pushforward measure:

$$
P _ {\tilde {X} | x} = g _ {*} P _ {\tilde {Z} | z} \tag {1}
$$

where $g _ { * }$ is the pushforward operator. Our analysis focuses on how $P _ { \tilde { Z } | z }$ affects the recovery map $h : = f \circ g : \mathcal { Z } \to \mathcal { Z }$ . The sampling mechanism need not be an image augmentation; it may also arise from temporal proximity, spatial crops, multimodal co-occurrence, or any other rule for drawing related observations. Independent views mean conditionally independent draws from $P _ { \tilde { Z } | z }$ for a fixed anchor $z ,$ not semantic independence of the resulting observations. Notation: uppercase $P , Q$ denotes probability laws, while lowercase $p , q$ denotes probability measure densities with respect to the stated reference measure.

# 3.3. Diversity Condition

We formalize the diversity condition and argue for its necessity in distance-preserving latent space reconstruction. The condition is motivated by comparing the theoretical ideal of full-support positive-pair sampling with practical sampling mechanisms. In the ideal vMF setting used by (Zimmermann et al., 2021), positive pairs can in principle cover the entire latent space around an anchor, which provides enough information to recover pairwise geometry. Practical mechanisms often restrict this support by keeping some latent coordinates fixed while changing others. The diversity condition identifies the weakest support requirement needed for recovery: every latent region that has nonzero marginal probability must also be reachable by the conditional positive-pair distribution.

Definition 3.1 (Diversity Condition). For a latent measurable space $\mathcal { Z }$ with marginal probability measure $P _ { Z }$ and conditional probability measure $P _ { \tilde { Z } | z }$ , the diversity condition holds ${ \mathrm { i f } } ,$ for $P _ { Z }$ -almost every $z \in { \mathcal { Z } } .$ ,

$$
P _ {Z} \ll P _ {\tilde {Z} | z}. \tag {2}
$$

Equivalently, for every measurable set $A \subseteq { \mathcal { Z } } , P _ { { \tilde { Z } } | z } ( A ) =$ 0 implies $P _ { Z } ( A ) = 0$ for $P _ { Z }$ -almost every anchor z.

Intuitively, the diversity condition requires that $P _ { \tilde { Z } | z }$ has sufficiently large support to cover any region where $P _ { Z }$ assigns nonzero probability. This implies that the viewgenerating process must perturb all latent features. If some generative features remain constant, the encoder cannot distinguish them along fixed dimensions, failing to invert $g$ for those components.

Finally, we relate the condition to practical sampling mechanisms. The diversity condition is stated at the level of the induced latent sampling mechanism $P _ { \tilde { Z } | z } ,$ ˜|z, rather than in terms of application-specific categories of transformations. Descriptions such as appearance, structure, or semantic content are therefore only informal indicators of which latent directions a sampling mechanism may vary. In practice, an augmentation or view-generation rule is useful for recovery only to the extent that it gives the conditional distribution support along the latent factors present under $P _ { Z }$ . Mechanisms that vary only a restricted subset of these factors may violate the condition, even if they produce visually distinct observations.

# 3.4. InfoNCE as Cross-Entropy Minimization

Information Noise Contrastive Estimation (InfoNCE) (van den Oord et al., 2019) is the standard contrastive objective used to train representations from one positive view and a set of negative samples. For each anchor, it increases similarity to the positive sample while decreasing similarity to negatives, making it a probabilistic discrimination loss over co-occurring and non-co-occurring samples. The InfoNCE objective quantifies encoder performance on this discrimination task.

Definition 3.2 (InfoNCE Loss (van den Oord et al., 2019)). Given a recovery map $h : \mathcal { Z } \to \mathcal { Z }$ , positive pairs $( z , \tilde { z } ) \sim$ $P _ { \mathrm { p o s } }$ , and negative samples $\{ z _ { i } ^ { - } \} _ { i = 1 } ^ { M } \stackrel { \mathrm { i . i . d . } } { \sim } P ^ { - }$ 1 i.i.d. ∼ P −:

$$
\mathcal{L}_{\mathrm{CL}}(h;\tau ,M) = \underset { \begin{array}{c}(z,\tilde{z})\sim P_{\mathrm{pos}}\\ \{z_{i}^{-}\}_{i = 1}^{M}\stackrel {\text{i.i.d.}}{\sim}P^{-} \end{array} }{\mathbb{E}}\left[-\log \frac{e^{h(z)^{\top}h(\tilde{z}) / \tau}}{D(z,\tilde{z})}\right]
$$

$$
D (z, \tilde {z}) = e ^ {h (z) ^ {\top} h (\tilde {z}) / \tau} + \sum_ {i = 1} ^ {M} e ^ {h (z) ^ {\top} h (z _ {i} ^ {-}) / \tau}
$$

where $\tau > 0$ is a temperature parameter and M is the number of negative samples.

Following (Zimmermann et al., 2021), we take the latent space to be a hypersphere $\mathbb { S } ^ { k - 1 }$ , motivated by the practical convention of $L ^ { \hat { 2 } } .$ -normalizing contrastive representations (Chen et al., 2020; Haas et al., 2024). We assume the conditional follows a von Mises-Fisher (vMF) distribution with concentration $\kappa > 0$ , where sampling frequency is inversely proportional to latent distance.

Theorem 3.3 (Asymptotics of ${ \mathcal { L } } _ { \mathrm { C I } }$ (Zimmermann et al., 2021)). Given a spherical latent space $\mathcal { Z } = \mathbb { S } ^ { k - 1 }$ , a uniform marginal law $P _ { Z } = U ( { \mathcal { Z } } )$ , a von Mises-Fisher conditional measure $P _ { \tilde { Z } | z }$ with density

$$
p (\tilde {z} | z) = \frac {e ^ {\kappa \tilde {z} ^ {\top} z}}{\int_ {\mathcal {Z}} e ^ {\kappa z ^ {\prime \top} z} d \sigma (z ^ {\prime})} \tag {3}
$$

where σ denotes spherical surface measure on $\mathcal { Z }$ and $\kappa > 0$ is the concentration parameter,

and a model conditional measure $Q _ { h , z }$ with density

$$
q _ {h} (\tilde {z} | z) = \frac {e ^ {h (\tilde {z}) ^ {\top} h (z) / \tau}}{\int_ {\mathcal {Z}} e ^ {h (z ^ {\prime}) ^ {\top} h (z) / \tau} d \sigma (z ^ {\prime})} \tag {4}
$$

For fixed $\tau > 0 ,$ , as the number of negative samples $M \to \infty$ the (normalized) contrastive loss converges to

$$
\begin{array}{l} \lim _ {M \to \infty} \mathcal {L} _ {C L} (h; \tau , M) - \log M + \log | \mathcal {Z} | = \\ \mathbb {E} _ {z \sim P _ {Z}} [ H (P _ {\tilde {Z} | z}, Q _ {h, z}) ] \tag {5} \\ \end{array}
$$

where $H ( P _ { \tilde { Z } | z } , Q _ { h , z } )$ denotes cross-entropy, using densities with respect to σ in the vMF setting.

This interpretation allows us to analyze contrastive learning through the lens of distribution matching between the sampling mechanism $P _ { \tilde { Z } | z }$ and the model measure $Q _ { h , z }$ .

# 3.5. Inductive Bias

Inductive bias refers to the structural assumptions that constrain a learning algorithm’s hypothesis space, enabling generalization (Vapnik, 1999). In contrastive learning, these assumptions arise through model architecture (e.g., translation equivariance in CNNs (LeCun et al., 1998), attention in Transformers (Dosovitskiy et al., 2021)) and geometric constraints on the embedding space. In our framework, an encoder is geometry-preserving when the recovery map $h = f \circ g$ recovers the latent structure up to an orthogonal transformation, i.e., $h ( z ) \approx A z$ for some $A \in O ( k )$ . Equivalently, the encoder approximates $f \approx A \circ g ^ { - 1 }$ on the data manifold. An effective inductive bias therefore makes approximate inverses of the data-generating process easier to represent and optimize, while restricting arbitrary non-geometry-preserving maps that can also satisfy the contrastive loss.

When the diversity condition is violated, the contrastive objective alone cannot uniquely determine the latent geometry. Inductive bias then acts as a compensatory mechanism, restricting admissible solutions to those consistent with architectural priors. In Section 4.3, we show that such biases are necessary for linearly identifiable reconstruction when diversity is violated. Crucially, we demonstrate experimentally (Section 5) that this necessity persists even asymptotically: the contrastive objective fails to recover latent geometry regardless of data quantity unless structural constraints are imposed.

# 4. Theoretical Results

We present the core theoretical results under the common assumption of $L ^ { 2 } .$ -normalized representations (Grill et al., 2020; Haas et al., 2023). Unless stated otherwise, the results use the following standing assumptions:

(A1) $\mathcal Z = \mathbb S ^ { k - 1 }$ is a unit hypersphere;   
(A2) the marginal $P _ { Z }$ is uniform on Z;   
(A3) the conditional $P _ { \tilde { Z } | z }$ is vMF with concentration $\kappa > 0$ before diversity violation is introduced;   
(A4) the encoder has sufficient capacity to realize any measurable recovery map considered below;   
(A5) representations are $L ^ { 2 } { \mathrm { - n o r m a l i z e d } } .$ , so $h : \mathcal { Z } \to \mathcal { Z }$ .

The full-support result below is closest to prior hypersphere identifiability results, but the proof uses a probabilistic Mazur-Ulam theorem and requires no differentiability of h. The subsequent results are the main extension: they characterize the asymptotic global optima when sampling diversity is violated. We introduce the following definitions to formalize our analysis.

Definition 4.1 (Isometry Almost Everywhere). Let $( \mathcal { Z } , \delta )$ be a metric space with measure µ. A measurable mapping $h : \mathcal { Z } \to \mathcal { Z }$ is an isometry almost everywhere if there exists an isometry $e : \mathcal { Z } \to \mathcal { Z }$ such that $h ( z ) = e ( z )$ for µ-almost all $z \in { \mathcal { Z } }$ .

Definition 4.2 (Equivalent Recovery Maps). Two mappings $h _ { 1 } , h _ { 2 } : { \mathcal { Z } } \to { \mathcal { Z } }$ are equivalent with respect to the contrastive loss if $\mathcal { L } _ { \mathrm { C L } } ( h _ { 1 } ; \tau , M ) = \mathcal { L } _ { \mathrm { C L } } ( h _ { 2 } ; \tau , M )$ for fixed τ and M .

# 4.1. Reconstruction Under Full-Support Sampling

Lemma 4.3 (Full-Support vMF Implies Diversity). Let $P _ { Z }$ be uniform on $\mathcal { Z } = \bar { \mathbb { S } } ^ { k - 1 }$ , and let $P _ { \tilde { Z } | z }$ have vMF density proportional to $\exp ( \kappa z ^ { \top } \tilde { z } )$ with $\kappa ~ > ~ 0$ with respect to spherical surface measure. Then the diversity condition holds.

Proof. The vMF density is strictly positive on the whole sphere. Therefore, any measurable set with zero $P _ { \tilde { Z } | z }$ measure also has zero spherical surface measure, and hence zero uniform marginal measure. □

Thus in the vMF setting the diversity condition is a consequence of full support, not an additional independent hypothesis. Contrastive learning then recovers the latent space up to orthogonal transformation.

Theorem 4.4 (Linear Identifiability Under Full Diversity). Under a uniform marginal on $\mathcal { Z } = \mathbb { S } ^ { k - 1 }$ and full-support vMF conditional, any recovery map $h : \mathcal { Z } \to \mathcal { Z }$ that globally minimizes the asymptotic contrastive objective is an isometry almost everywhere:

$$
h (z) = A z \quad \text {   for   } \mu \text {-almost   all   } z \in \mathcal {Z},
$$

where $A \in O ( k )$ is an orthogonal matrix.

Proof. See Appendix A.1.

Proof sketch. The asymptotic InfoNCE objective reduces to conditional cross-entropy. Under the full-support vMF conditional, any minimizer must match conditionals and therefore preserve inner products almost surely. On the sphere, this gives distance preservation, and the probabilistic Mazur-Ulam theorem extends this almost-sure isometry to a global orthogonal map.

This represents an ideal scenario: when the sampling mechanism perturbs all latent factors, the learned representations preserve pairwise distances, yielding a separable latent space suitable for downstream tasks. However, real-world augmentation pipelines rarely have full support, motivating the analysis in the following section.

# 4.2. Reconstruction Under Violated Diversity

In practice, the diversity condition is rarely satisfied. Most augmentation pipelines preserve certain semantic features while perturbing others. We model this by decomposing the latent vector $z = ( u , v )$ into an invariant component $u \in \mathbb { R } ^ { m }$ and a varying component $v \in \mathbb { R } ^ { \ell }$ , with $m + \ell = k$ and $m , \ell > 0$ . Let

$$
K (z) := \{\tilde {z} = (\tilde {u}, \tilde {v}) \in \mathbb {S} ^ {k - 1}: \tilde {u} = u \},
$$

$$
K (z) \cong \mathbb {S} _ {r (z)} ^ {\ell - 1}, \qquad r (z) := \sqrt {1 - \| u \| ^ {2}}.
$$

For $r ( z ) > 0$ , let $\sigma _ { K \left( z \right) }$ denote the intrinsic surface measure on $K ( z )$ . The constrained positive-pair conditional is the probability measure $P _ { \tilde { Z } | z } ^ { K }$ supported on $K ( z )$ with Radon-Nikodym density

$$
\frac {d P _ {\tilde {Z} | z} ^ {K}}{d \sigma_ {K (z)}} (\tilde {z}) = \frac {e ^ {\kappa z ^ {\top} \tilde {z}}}{\int_ {K (z)} e ^ {\kappa z ^ {\top} z ^ {\prime}} d \sigma_ {K (z)} (z ^ {\prime})}. \tag {6}
$$

This measure is singular with respect to ambient spherical measure on $\mathbb { S } ^ { k - 1 }$ , but it is a well-defined probability law on the lower-dimensional submanifold $K ( z )$ .

This collapses the sampling support from $\mathbb { S } ^ { k - 1 }$ onto a lowerdimensional submanifold $K ( z )$ , mirroring the content-style framework of (von Kugelgen et al. ¨ , 2021).

Theorem 4.5 (Loss of Identifiability Under Violated Diversity). For the asymptotic contrastive objective obtained as $M \ \to \ \infty$ under the constrained conditional (Equation 6) and standard full-sphere negatives, orthogonal recovery is not globally optimal. There exists a recovery map $h : \mathcal { Z } \to \mathcal { Z }$ that is not induced by any orthogonal transformation such that

$$
\mathcal {L} _ {\mathrm{CL}} (h) <   \mathcal {L} _ {\mathrm{CL}} (\tilde {h}), \quad \forall \tilde {h} \in O (k).
$$

This is a statement about global values of the limiting objective, not about finite-sample optimization dynamics.

Proof. See Appendix A.2.

![](images/2b12b6aed2da4e99fe890a7204b7c98dbcf628f900e72be4071b635b613afd1e.jpg)

Proof sketch. The constrained conditional fixes u and only varies v, while the standard model conditional still normalizes over the full sphere. We construct an explicit nonorthogonal map $h _ { \lambda } ( u , v ) = ( u , \lambda v ) / \| ( u , \lambda v ) \|$ ∥ with $\lambda < 1$ close to one. Because positive pairs share u, this map improves the alignment term to first order, while the uniformity term is stationary at the identity, yielding lower asymptotic loss than any orthogonal map.

The consequence is concrete: arbitrarily expressive encoders are actively disincentivized from learning orthogonal recovery maps when the diversity condition is violated. The contrastive objective rewards geometry-distorting solutions, leading to poorly structured latent spaces where semantic relationships are not preserved. Consequently, downstream tasks that rely on meaningful distance relationships in the representation space face a fundamental bottleneck that cannot be overcome by increasing encoder capacity alone.

# 4.3. Correcting the Model

The support mismatch between $P _ { \tilde { Z } | z } ^ { K }$ z and the standard fullsphere model conditional prevents the InfoNCE objective from favoring isometric solutions. We address this by constraining the model conditional to the same submanifold $K ( z )$ :

$$
q _ {h, z} ^ {K} (\tilde {z}) = \frac {e ^ {h (z) ^ {\top} h (\tilde {z}) / \tau}}{\int_ {K (z)} e ^ {h (z) ^ {\top} h (z ^ {\prime}) / \tau} d \sigma_ {K (z)} \left(z ^ {\prime}\right)}, \quad \tilde {z} \in K (z). \tag {7}
$$

Equivalently, for a positive pair $( z , \tilde { z } )$ and negatives $z _ { 1 } ^ { - } , \ldots , z _ { M } ^ { - } \stackrel { \mathrm { i . i . d . } } { \sim } U ( K ( z ) )$ , the per-anchor adapted loss is

$$
\ell_ {\mathrm{adapt}} = - \log \frac {\exp (h (z) ^ {\top} h (\tilde {z}) / \tau)}{D _ {a} (z , \tilde {z})},
$$

$$
D _ {a} (z, \tilde {z}) = \exp (h (z) ^ {\top} h (\tilde {z}) / \tau) \tag {8}
$$

$$
+ \sum_ {j = 1} ^ {M} \exp (h (z) ^ {\top} h (z _ {j} ^ {-}) / \tau).
$$

This corresponds to drawing negative samples from $K ( z )$ rather than the full latent space. Exact sampling requires access to the invariant component u, which is generally unavailable in real data. In practice, same-anchor augmentations provide a proxy: independent transformations of the same anchor preserve the components fixed by the augmentation mechanism while varying the remaining components. Appendix C gives the resulting training step, but the adapted loss should be read primarily as a theoretical diagnostic rather than a fully practical replacement for standard InfoNCE.

Theorem 4.6 (Orthogonal Mappings as Minimizers). Under the corrected model (Equation 7), any orthogonal transformation $h \in O ( k )$ minimizes the asymptotic contrastive loss.

Proof. See Appendix A.3.

Proof sketch. Once the model conditional is restricted to the same support $K ( z )$ as the true conditional, any orthogonal map preserves the inner products that define the vMF density on that support. Thus the model conditional matches the true conditional when $\kappa = 1 / \tau$ , minimizing cross-entropy. This proves achievability of orthogonal solutions, but not uniqueness.

Although the corrected objective admits orthogonal solutions, it does not guarantee that all minimizers are orthogonal. Thus, the adapted objective should not be interpreted as a fully general solution: it removes the support mismatch and makes geometry-preserving recovery achievable, but it does not make such recovery unique. The sampling mechanism determines which solutions are achievable, while inductive bias influences which solution is selected during optimization. This non-uniqueness highlights the necessity of inductive bias for selecting geometry-preserving solutions.

# 5. Experimental Results

# 5.1. Synthetic Dataset Experiments

We validate the theoretical predictions from Section 4 using a controlled synthetic setup with a spherical latent space $\mathcal { Z } = \mathbb { S } ^ { 2 } . ^ { 1 }$ Following the assumptions of our theoretical framework, the marginal law is uniform over the sphere, and positive pairs are sampled according to a von Mises-Fisher (vMF) conditional law $P _ { \tilde { Z } | z }$ with density $p ( \tilde { z } | z ) = \mathrm { v M F } ( z , \kappa )$ and concentration parameter $\kappa = 1 / \tau$ (see Equation 17), where τ is the temperature in the InfoNCE loss. The generative processes used in our experiments are illustrated in Figure 2. Details of the sampling procedures are provided in Appendix C.

Experimental Setup. We evaluate five generative processes $g : { \mathcal { Z } }  { \mathcal { X } }$ of varying complexity: Identity, injective Linear map $( \mathbb { S } ^ { 2 }  \bar { \mathbb { R } ^ { 7 } } )$ , Spiral rotation, Patches, and invertible MLP (Hyvarinen & Morioka¨ , 2016). Detailed definitions are provided in Appendix B.1. For each generative process, we compare two encoder architectures: an MLP encoder with hidden dimensions [128, 256, 256, 256, 128] representing low inductive bias, and an inverse encoder designed to invert the corresponding generative process, representing high inductive bias. The invertible MLP is an exception, as it lacks a strict analytic inverse; we therefore evaluate only the MLP encoder for this generative process. All experiments use InfoNCE loss with temperature $\tau = 0 . 3$ , Adam optimizer (Kingma & Ba, 2015) $( \mathrm { l r } = 1 0 ^ { - 3 } )$ , batch size 2000, and 5000 iterations. The generative process g remains frozen throughout training, and each configuration is evaluated over 5 independent runs.

Evaluation Metrics. We assess reconstruction quality using linear identifiability $( R ^ { 2 } )$ , which measures recovery up 1Code is available at https://github.com/ BosonicJustin/CLTheory.

to affine transformation as in (Hyvarinen & Morioka¨ , 2016). Additional metrics (Mean Correlation Coefficient and Angular Preservation Error) are provided in Appendix D.

Results. We summarize the experimental results in Table 1 and Figure 5.

Diversity Condition Holds. When the diversity condition is satisfied, the MLP encoder achieves near-perfect linear identifiability $( R ^ { 2 } \ge 0 . 9 9 )$ across all five generative processes, confirming the theoretical prediction of Theorem 4.4. This demonstrates that a sufficiently expressive encoder can recover a representation linearly identifiable with the ground-truth latent space when the diversity condition holds, across varying degrees of injective generative processes.

Diversity Condition Violated. When the diversity condition is violated by fixing the first latent dimension during positive pair sampling, MLP performance collapses dramatically. Linear identifiability drops to $R ^ { 2 } \in [ 0 . \mathrm { \bar { 0 5 } } , 0 . 1 3 ]$ for Identity, MLP, and Linear generative processes, and $R ^ { 2 } = 0$ .25 for Patches. The Spiral process is an exception $( R ^ { 2 } = 0 . 7 2 )$ , maintaining poor but not catastrophic performance with high variance across runs. This validates Theorem 4.5: the contrastive objective alone no longer incentivizes geometry-preserving solutions when the conditional sampling support is restricted.

Inductive Bias Compensation. Under violated diversity, incorporating inductive bias through inverse encoders restores near-perfect recovery $( R ^ { 2 } \ \bar { \geq } \ 0 . 8 8 )$ , with Identity, Linear, and Spiral processes achieving $R ^ { 2 } \ge 0 . 9 9$ . The inverse encoders, designed to mirror the structure of each generative process, successfully recover the latent geometry even when the sampling mechanism provides insufficient information. This demonstrates that architectural constraints can compensate for deficiencies in the sampling regime, highlighting the complementary roles of data augmentation and model design in contrastive learning.

Adapted InfoNCE Loss. The adapted InfoNCE loss partially recovers MLP performance under violated diversity $( R ^ { 2 } ~ \approx ~ 0 . 6 0 – 0 . 6 5 )$ , representing a substantial improvement over standard InfoNCE $( \bar { R ^ { 2 } } \approx 0 . 0 5  – 0 . 2 5$ for most processes). However, this falls short of the performance achieved with appropriate inductive bias $( R ^ { 2 } \ge 0 . 8 8 )$ . This confirms Theorem 4.6: correcting the model conditional makes isometric solutions achievable but does not guarantee them. From a practical standpoint, these results suggest that while loss modifications can mitigate the effects of violated diversity, incorporating architectural priors remains the more effective strategy, aligning with the empirical success of high inductive bias architectures such as CNNs and

(a) Identity   
![](images/205cba5d936e9104f1d4f8efc018a5750d7ac8392558bc425f1bd6cc22be088d.jpg)

(b) Linear   
![](images/d757c440e69bae457ac8c758b3d76db6679cfd77ff1d9e674a3fa0e87629f3cc.jpg)

(c) Spiral   
![](images/91e52c012b038c56e4d72f70c60c1f0bd0ea830cc405319f3de80f64d3b30307.jpg)

(d) Patches   
![](images/32ced65c109a0e085891f7c2c2de3629139b5af9eefede484d395684da7b9b69.jpg)

(e) Invertible MLP   
![](images/3fe01473de16af41585acb26414640816f5c6efdecc902661cb71225d061e93d.jpg)  
Figure 2. Generative processes mapping the unit sphere to observation space. Colors encode input coordinates (RGB = xyz), illustrating how each transformation warps the latent space: (a) identity preserves the sphere, (b) linear maps to an ellipsoid, (c) spiral twists points around the vertical axis, (d) patches applies piecewise rotations creating discontinuities, and (e) invertible MLP produces smooth nonlinear deformations.

Vision Transformers in contrastive learning pipelines.

# 5.2. CIFAR-10 Experiments

We test the qualitative implications of our theory on CIFAR-10 using SimCLR (Chen et al., 2020) with three encoder architectures of comparable size (∼11M parameters each). We rate inductive bias by how closely each architecture’s structural priors align with the spatial generative structure of natural images. Equivalently, this measures how strongly the architecture favors approximate inverses of image formation. ResNet-18 (He et al., 2016) has high inductive bias because convolutional layers encode spatial locality and translation equivariance. Vision Transformer (ViT) (Dosovitskiy et al., 2021) with 4×4 patches has medium inductive bias because self-attention can learn global spatial structure, but does not hardcode locality to the same extent. The MLP has low inductive bias because it treats each image as a flat vector and imposes little domain-specific restriction on the hypothesis space. All encoders project to 512-dimensional L2-normalized embeddings and are initialized with random weights. Because CIFAR-10 does not provide ground-truth latent coordinates, linear probe accuracy is not a direct test of identifiability. We use it as a real-data sanity check for the qualitative predictions of the theory: richer sampling mechanisms and better aligned architectural inductive bias should improve downstream representation quality.

Augmentation Regimes. We design three augmentation regimes to vary the degree to which the diversity condition is approximated (Figure 4): (1) All: color jitter, random crop, horizontal flip, grayscale, blur, and cutout (DeVries & Taylor, 2017), perturbing as many features as possible; (2) Crop Only: Perturbing all features slightly, but with a lesser degree than All augmentations; (3) All w/o crop: all augmentations except crop, varying the features in a more fixed manner. Training uses InfoNCE with $\tau = 0 . 5$ , Adam optimizer (Kingma & Ba, 2015) $( \mathrm { l r } = 3 \times 1 0 ^ { - 4 } )$ , batch size 2000, for 200 epochs. We evaluate via linear probing over 5 runs per configuration.

Results. The results (Figure 3) are consistent with the qualitative predictions of the theory. First, the “All” augmentation regime yields the highest accuracy across all architectures, consistent with the expectation that broader sampling support improves representation quality. Second, higher inductive bias consistently improves performance: ResNet-18 outperforms ViT, which outperforms MLP, across all augmentation regimes. Third, and most importantly, the performance gap between architectures widens as the diversity condition is increasingly violated. Under the “All” regime, the gap between ResNet-18 and MLP is moderate; under “All w/o crop”, this gap increases substantially. This interaction effect supports the compensatory role of inductive bias: when sampling diversity is insufficient, architectural priors become critical for recovering useful representations. Notably, for MLP the “All” and “Crop” regimes yield nearly identical performance, suggesting that without appropriate inductive bias, the encoder cannot exploit the additional augmentations, since cropping provides the majority of the meaningful signal.

![](images/77aa6af5c8ab7508bc90e2cdf194cdd744bbafeafd4c8d7ce48bcd933328b7ce.jpg)

<details>
<summary>scatter</summary>

| Model       | Condition  | Individual runs | Mean |
|-------------|------------|-----------------|------|
| ResNet-18   | All        | 57.0            | 56.0 |
| ResNet-18   | Crop       | 38.0            | 37.0 |
| ResNet-18   | w/o Crop   | 34.0            | 33.0 |
| ViT-Tiny    | All        | 55.0            | 54.0 |
| ViT-Tiny    | Crop       | 42.0            | 41.0 |
| ViT-Tiny    | w/o Crop   | 33.0            | 32.0 |
| MLP         | All        | 43.0            | 42.0 |
| MLP         | Crop       | 44.0            | 43.0 |
| MLP         | w/o Crop   | 32.0            | 31.0 |
</details>

Figure 3. Linear probe accuracy on CIFAR-10 by architecture and augmentation regime. Individual runs shown as points; bars indicate mean ±1 std. The “All” regime best approximates the diversity condition and yields highest accuracy across all architectures.

# 6. Conclusion

We have presented a theoretical framework for understanding when contrastive learning recovers meaningful latent representations. Our central contribution is the diversity condition (Definition 3.1), a requirement on $P _ { \tilde { Z } | } { } _ { i }$ z that is necessary for isometric latent recovery. When it holds, sufficiently expressive encoders recover the latent space up to an orthogonal transformation; when it is violated, the contrastive objective can actively disincentivize geometry-preserving solutions. The adapted InfoNCE objective makes isometric solutions achievable under violated diversity, but does not guarantee their selection. Our synthetic and CIFAR-10 experiments show that sampling diversity and architectural inductive bias jointly determine representation quality.

Table 1. Linear identifiability $( R ^ { 2 } )$ across generative processes under different experimental conditions. Results reported as mean ± std across 5 random seeds. The Invertible MLP process lacks a closed-form inverse, so no inverse encoder is available (indicated by N/A). 

<table><tr><td><img src="images/ade5e400ebf0cad49fd03018a44dd0cde39570c13f6bfd473e8d260f7b710b5b.jpg"/></td><td><img src="images/dc5a8bed7eea55695bf019bd8a2b717cab640e24b6ed508a035178f7ec80bc9d.jpg"/></td><td><img src="images/fc0e38708d3d6f191eb4672d51ce993f00f52d54a16852cec621163834466624.jpg"/></td><td><img src="images/4c3b5504577f85772ad19800cdda1f8cbe51e3463ab5215d3e46c5e8a2b9fb7f.jpg"/></td><td><img src="images/8019b422a0ef2a8b6ac2112dae6e67dfa9e652ceb90e21cbf4194106d732a2d8.jpg"/></td></tr><tr><td rowspan="2">Generative Process</td><td colspan="2">Diversity Holds</td><td colspan="2">Diversity Violated</td></tr><tr><td>InfoNCE</td><td>InfoNCE</td><td>InfoNCE Adapted</td><td>InfoNCE + Ind. Bias</td></tr><tr><td>Identity</td><td>1.00 ± 0.00</td><td>0.06 ± 0.03</td><td>0.63 ± 0.00</td><td>0.99 ± 0.00</td></tr><tr><td>Invertible MLP</td><td>1.00 ± 0.00</td><td>0.13 ± 0.12</td><td>0.60 ± 0.01</td><td>N/A</td></tr><tr><td>Linear</td><td>1.00 ± 0.00</td><td>0.05 ± 0.06</td><td>0.65 ± 0.03</td><td>0.99 ± 0.00</td></tr><tr><td>Patches</td><td>0.99 ± 0.00</td><td>0.25 ± 0.03</td><td>0.63 ± 0.02</td><td>0.88 ± 0.01</td></tr><tr><td>Spiral</td><td>1.00 ± 0.00</td><td>0.72 ± 0.33</td><td>0.62 ± 0.01</td><td>1.00 ± 0.00</td></tr></table>

![](images/26879799a8ce512cbbee8ff02e0920d69d979965b578b675394f403a218fe015.jpg)

<details>
<summary>natural_image</summary>

Two potted flowers on a wooden deck, one with purple flowers and green leaves, the other with yellow flowers (no text or symbols visible)
</details>

(a) Original

![](images/40dc3a6480d3faa0add1be176dec14314688649bbfc98981857951a186d4d67c.jpg)

<details>
<summary>natural_image</summary>

Potted purple flowers with green leaves on a wooden surface (no text or symbols visible)
</details>

(b) Crop

![](images/675b52e961d74389217c66e463612928cb1b17a2392970133aa2f872cb55d328.jpg)

<details>
<summary>natural_image</summary>

Two potted flowers on a wooden deck, one with purple flowers and green leaves (no text or symbols visible)
</details>

(c) All w/o crop

![](images/f1f6c91777f8529b140553d7144a62a0fdb4a23d22377c8c7cdcd311c641edb0.jpg)

<details>
<summary>natural_image</summary>

Potted orange pot with purple flowers and green leaves on a wooden surface (no text or symbols visible)
</details>

(d) All   
Figure 4. CIFAR-10 augmentation regimes. (a) Original image. (b) Crop Only: random resized crop altering spatial extent. (c) All without crop: color jitter, horizontal flip, rotation, and blur. (d) All augmentations combined. Cropping changes visible spatial extent and local statistics, while color and blur transformations mainly alter appearance in this example.

Limitations and future work. Motivated by practical $L ^ { 2 }$ normalization, our analysis uses a spherical latent space and vMF conditionals; future work should relax these assumptions and estimate diversity violation from data. Adapted InfoNCE is a theoretical diagnostic rather than a scalable objective: anchor-specific negatives require O(N(M + 1)) memory, and support correction restores achievability but not selection.

# Impact Statement

This work advances theoretical understanding of contrastive learning, a foundational technique for self-supervised representation learning. Our contributions are primarily theoretical, providing formal conditions (the diversity condition) under which contrastive methods succeed or fail at recovering meaningful latent structure.

The practical implications are indirect but potentially significant. By clarifying the interplay between sampling mechanisms and architectural inductive bias, our framework may guide more principled design of augmentation pipelines and encoder architectures, potentially reducing computational waste from trial-and-error experimentation. This could lower the environmental cost of training large-scale representation learning systems.

Contrastive learning underlies many deployed systems in vision, language, and multimodal domains. Improved theoretical understanding may help practitioners anticipate failure modes before deployment, particularly in high-stakes applications such as medical imaging or scientific discovery where representation quality directly affects downstream reliability.

We do not foresee direct negative societal consequences from this theoretical work. However, as with any advance in representation learning, improved methods could enhance both beneficial applications (e.g., drug discovery, climate modeling) and potentially harmful ones (e.g., surveillance). We encourage practitioners to consider the ethical implications of specific downstream applications enabled by better representation learning.

# References

Bahrami, M., Tejada-Lapuerta, A., Becker, S., Hashemi G., F. S., and Theis, F. J. scConcept: Contrastive pretraining for technology-agnostic single-cell representations beyond reconstruction. bioRxiv, 2025. doi: 10.1101/2025.10.14.682419. URL https://www.biorxiv.org/content/ early/2025/10/15/2025.10.14.682419.   
Ballard, T. Contrastive learning for climate model bias correction and super-resolution. In AAAI 2022 Fall Symposium: The Role of AI in Responding to Climate Challenges, 2022. URL https://www.climatechange.ai/papers/ aaaifss2022/10.   
Bardes, A., Ponce, J., and LeCun, Y. VICReg: Varianceinvariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022.   
Bengio, Y., Courville, A., and Vincent, P. Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35 (8):1798–1828, 2013.   
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 1597–1607. PMLR, 2020.   
Comon, P. Independent component analysis, a new concept? Signal Processing, 36(3):287–314, 1994.   
Cy, A., Chemparathy, A., Han, M., Dangovski, R., Lu, P. Y., and Soljacic, M. Studying phase transitions in contrastive learning With physics-inspired datasets. In ICLR 2023 Workshop Physics4ML, 2023. URL https://openreview.net/forum? id=djssHWljSA. Physics4ML poster.   
Dangovski, R., Jing, L., Loh, C., Han, S., Srivastava, A., Cheung, B., Agrawal, P., and Soljaciˇ c, M. Equivari-´ ant contrastive learning. In International Conference on Learning Representations, 2022.   
DeVries, T. and Taylor, G. W. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017. URL https:// arxiv.org/abs/1708.04552.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, ´ P. H., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z. D., Gheshlaghi Azar, M., Piot, B., Kavukcuoglu, K., Munos, R., and Valko, M. Bootstrap your own latent: A new approach to self-supervised learning. In Advances in Neural Information Processing Systems, volume 33, pp. 21271–21284, 2020.

Haas, J., Yolland, W., and Rabus, B. Linking neural collapse and L2 normalization with improved out-ofdistribution detection in deep neural networks. Transactions on Machine Learning Research, 2023. ISSN 2835- 8856. URL https://openreview.net/forum? id=fjkN5Ur2d6.

Haas, J., Yolland, W., and Rabus, B. Exploring simple, high quality out-of-distribution detection with L2 normalization. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https:// openreview.net/forum?id=daX2UkLMS0.

HaoChen, J. Z. and Ma, T. A theoretical study of inductive biases in contrastive learning. In International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=AuEgNlEAmed.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

Hyvarinen, A. and Morioka, H. Unsupervised feature ex- ¨ traction by time-contrastive learning and nonlinear ICA. In Advances in Neural Information Processing Systems, 2016.

Hyvarinen, A. and Oja, E. Independent component analysis: ¨ algorithms and applications. Neural Networks, 13(4-5): 411–430, 2000.

Hyvarinen, A., Sasaki, H., and Turner, R. E. Nonlinear ¨ ICA using auxiliary variables and generalized contrastive learning. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pp. 859–868. PMLR, 2019.

Jaiswal, A., Babu, A. R., Zadeh, M. Z., Banerjee, D., and Makedon, F. A survey on contrastive self-supervised learning. Technologies, 9(1):2, 2021. doi: 10.3390/ technologies9010002.

Ji, W., Deng, Z., Nakada, R., Zou, J., and Zhang, L. The power of contrast for feature learning: A theoretical analysis. Journal of Machine Learning Research, 24(330): 1–78, 2023. URL https://jmlr.org/papers/ v24/21-1501.html.   
Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. URL https://arxiv.org/ abs/1412.6980.   
Kirchhof, M., Kasneci, E., and Oh, S. J. Probabilistic contrastive learning recovers the correct aleatoric uncertainty of ambiguous inputs. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 17085– 17104. PMLR, 2023.   
Krizhevsky, A. Learning multiple layers of features from tiny images. Technical Report TR-2009, University of Toronto, 2009.   
LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi: 10.1109/5.726791.   
Liu, S., Zhao, L., and Chen, D. Clicv2: Image complexity representation via content invariance contrastive learning, 2025. URL https://arxiv.org/abs/2503. 06641.   
Liu, Z., Hu, K., Zhang, J., Ren, X., and Wang, X. Longterm air quality data filling based on contrastive learning. Information, 17(2), 2026. ISSN 2078-2489. doi: 10.3390/info17020121. URL https://www.mdpi. com/2078-2489/17/2/121.   
Locatello, F., Bauer, S., Lucic, M., Raetsch, G., Gelly, S., Scholkopf, B., and Bachem, O. Challenging common ¨ assumptions in the unsupervised learning of disentangled representations. In International Conference on Machine Learning, pp. 4114–4124, 2019.   
Mitrovic, J., McWilliams, B., Walker, J., Buesing, L., and Blundell, C. Representation learning via invariant causal mechanisms. In International Conference on Learning Representations, 2021.   
Poudel, R. P. K., Pandya, H., and Cipolla, R. Contrastive unsupervised learning of world model with invariant causal features, 2022. URL https://arxiv.org/abs/ 2209.14932.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In

Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 2021.   
Richter, T., Bahrami, M., Xia, Y., Fischer, D. S., and Theis, F. J. Delineating the effective use of self-supervised learning in single-cell genomics. Nature Machine Intelligence, 7(1):68–78, 2025.   
Rusak, E., Reizinger, P., Juhos, A., Bringmann, O., Zimmermann, R. S., and Brendel, W. InfoNCE: Identifying the gap between theory and practice. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pp. 4159–4167. PMLR, 2025.   
Sandilya, R., Perez, S., Lynch, C., Victoria, L., Zebley, B., Buchanan, D. M., Bhati, M. T., Williams, N., Spellman, T. J., Gunning, F. M., Liston, C., and Grosenick, L. Contrastive diffusion alignment: Learning structured latents for controllable generation, 2025. URL https://arxiv.org/abs/2510.14190.   
Tian, Y., Sun, C., Poole, B., Krishnan, D., Schmid, C., and Isola, P. What makes for good views for contrastive learning? In Advances in Neural Information Processing Systems, 2020.   
Tschannen, M., Mustafa, B., and Houlsby, N. CLIPPO: Image-and-language understanding from pixels only. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11006–11017, 2023.   
Uelwer, T., Robine, J., Wagner, S. S., Hoftmann, M., Up- ¨ schulte, E., Konietzny, S., Behrendt, M., and Harmeling, S. A survey on self-supervised representation learning, 2023. URL https://arxiv.org/abs/2308. 11455.   
van den Oord, A., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding, 2019. URL https://arxiv.org/abs/1807.03748.   
Vapnik, V. The Nature of Statistical Learning Theory. Springer: New York, 1999.   
von Kugelgen, J., Sharma, Y., Gresele, L., Brendel, W.,¨ Scholkopf, B., Besserve, M., and Locatello, F. Self-¨ supervised learning with data augmentations provably isolates content from style. In Advances in Neural Information Processing Systems, volume 34, pp. 16451–16467, 2021.   
Wang, T. and Isola, P. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 9929–9939. PMLR, 2020.

Wen, Z. and Li, Y. Toward understanding the feature learning process of self-supervised contrastive learning. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 11112–11122. PMLR, 2021. URL https://proceedings.mlr.press/ v139/wen21c.html.   
Wilkinson, A., Radev, R., and Alonso-Monsalve, S. Contrastive learning for robust representations of neutrino data. Physical Review D, 111(9), May 2025. ISSN 2470-0029. doi: 10.1103/physrevd.111. 092011. URL http://dx.doi.org/10.1103/ PhysRevD.111.092011.   
Wood, A. T. A. Simulation of the von Mises–Fisher distribution. Communications in Statistics-Simulation and Computation, 23(1):157–164, 1994. doi: 10.1080/ 03610919408813161.   
Zaliaduonis, J. and Gatidis, S. A probabilistic generalization of the Mazur-Ulam theorem, 2026. URL https:// arxiv.org/abs/2601.03900.   
Zhao, L., Gundavarapu, N. B., Yuan, L., Zhou, H., Yan, S., Sun, J. J., Friedman, L., Qian, R., Weyand, T., Zhao, Y., Hornung, R., Schroff, F., Yang, M.-H., Ross, D. A., Wang, H., Adam, H., Sirotenko, M., Liu, T., and Gong, B. VideoPrism: A foundational visual encoder for video understanding. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 60785–60811. PMLR, 2024.   
Zimmermann, R. S., Sharma, Y., Schneider, S., Bethge, M., and Brendel, W. Contrastive learning inverts the data generating process. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 12979–12990. PMLR, 2021.

# A. Theoretical Results: Proofs

This appendix provides complete proofs for the theoretical results presented in Section 4. We organize the material into three subsections corresponding to the three scenarios analyzed: full diversity (Section A.1), violated diversity (Section A.2), and the corrected model (Section A.3).

# A.1. Proofs for Reconstruction Under Full Diversity

We first establish that cross-entropy minimizers preserve inner products almost surely, which forms the foundation for proving linear identifiability.

Theorem A.1 (Cross-Entropy Minimizers Preserve Inner Products). Let $\mathcal { Z } = \mathbb { S } ^ { k - 1 } , \kappa > 0$ , and let $P _ { \tilde { Z } | z }$ have density

$$
p (\tilde {z} | z) = C _ {p} ^ {- 1} \exp (\kappa \tilde {z} ^ {\top} z),
$$

where $C _ { p }$ is the normalizing constant with respect to σ. Let $Q _ { h , z }$ have density $q _ { h } ( \cdot | z )$ with respect to σ. Let $h : \mathcal { Z } \to \mathcal { Z }$ be a recovery map. If h minimizes the cross-entropy

$$
\mathcal {L} _ {h} := \mathbb {E} _ {z \sim P _ {Z}} [ H (P _ {\tilde {Z} | z}, Q _ {h, z}) ],
$$

$t h e n p ( \tilde { z } | z ) = q _ { h } ( \tilde { z } | z ) P _ { \tilde { Z } | z } { - a . s . , a n d z ^ { \top } \tilde { z } = h ( z ) ^ { \top } h ( \tilde { z } ) } P _ { Z , \tilde { Z } } { - a . s . }$

Proof. Since h minimizes $\mathcal { L } _ { h }$ , it minimizes $H ( P _ { \tilde { Z } | z } , Q _ { h , z } )$ almost surely with respect to $P _ { Z }$ . The cross-entropy H is minimized when

$$
p (\cdot | z) = q _ {h} (\cdot | z) \quad P _ {\tilde {Z} | z} \text {-a.s.}
$$

For a fixed z, this implies:

$$
\frac {e ^ {\kappa \tilde {z} ^ {\top} z}}{C _ {p}} = \frac {e ^ {h (\tilde {z}) ^ {\top} h (z) / \tau}}{D (z)} \quad P _ {\tilde {Z} | z} \text {-a.s.} \tag {9}
$$

where $D ( z )$ and $C _ { p }$ are the normalizing constants for the respective von Mises-Fisher distributions.

Taking the logarithm yields:

$$
\log \left(\frac {D (z)}{C _ {p}}\right) \tau + \tau \kappa \tilde {z} ^ {\top} z = h (\tilde {z}) ^ {\top} h (z) \quad P _ {\tilde {Z} | z} \text {-a.s.} \tag {10}
$$

Since h maps onto the unit sphere,

$$
- 1 \leq h (\tilde {z}) ^ {\top} h (z) \leq 1 \quad P _ {\tilde {Z} | z} \text {-a.s.} \tag {11}
$$

Rearranging gives:

$$
- 1 \leq \log \left(\frac {D (z)}{C _ {p}}\right) \tau + \tau \kappa \tilde {z} ^ {\top} z \leq 1 \quad P _ {\tilde {Z} | z} \text {-a.s.} \tag {12}
$$

This simplifies to:

$$
\frac {- 1 - \log \left(\frac {D (z)}{C _ {p}}\right) \tau}{\tau \kappa} \leq \tilde {z} ^ {\top} z \leq \frac {1 - \log \left(\frac {D (z)}{C _ {p}}\right) \tau}{\tau \kappa} \tag {13}
$$

Since $P _ { \tilde { Z } | } ,$ z is a von Mises-Fisher distribution, the support of $\tilde { z } ^ { \top } z$ under this measure includes regions arbitrarily close to both 1 and −1. Specifically, for any $c \in ( - 1 , 1 )$ :

$$
\mu \left(\{\tilde {z} \in \mathcal {Z}: \tilde {z} ^ {\top} z \geq c \}\right) = \beta \int_ {0} ^ {\cos^ {- 1} (c)} e ^ {\kappa \cos \theta} \sin^ {k - 2} \theta d \theta > 0 \tag {14}
$$

where $\beta > 0$ is the spherical-measure normalization constant.

To satisfy the inequality almost surely with respect to $P _ { \tilde { Z } | z }$ , the bounds must equal the extremes:

$$
\frac {1 - \log \left(\frac {D (z)}{C _ {p}}\right) \tau}{\tau \kappa} = 1 \quad \text { and } \quad \frac {- 1 - \log \left(\frac {D (z)}{C _ {p}}\right) \tau}{\tau \kappa} = - 1 \tag {15}
$$

Both equations yield:

$$
\log \left(\frac {D (z)}{C _ {p}}\right) = 0 \implies D (z) = C _ {p} \tag {16}
$$

and

$$
\kappa \tau = 1 \tag {17}
$$

Substituting back:

$$
\tilde {z} ^ {\top} z = h (\tilde {z}) ^ {\top} h (z) \quad P _ {\tilde {Z} | z} \text {-a.s.} \tag {18}
$$

Since h minimizes $\mathcal { L } _ { h }$ , this holds for almost all z with respect to $P _ { Z }$ , implying:

$$
\tilde {z} ^ {\top} z = h (\tilde {z}) ^ {\top} h (z) \quad P _ {Z, \tilde {Z}} \text {-a.s.} \tag {19}
$$

![](images/7a5bb777f625e8479675e7ad7ebeba81af5bb6101f6d2d5e81b03727df19f7c6.jpg)

The following corollary translates inner product preservation to distance preservation, which is the key geometric property needed for identifiability.

Corollary A.2 (Cross-Entropy Minimizers Preserve Distances Almost Everywhere). If h minimizes the expected crossentropy loss $\mathbb { E } _ { z \sim P _ { Z } } [ H ( P _ { \tilde { Z } | z } , Q _ { h , z } ) ]$ ], then

$$
\| h (z) - h (\tilde {z}) \| ^ {2} = \| z - \tilde {z} \| ^ {2} P _ {Z, \tilde {Z}} \text {-a.s.}
$$

Proof. Since both $z , \tilde { z } \in \mathcal { Z } = \mathbb { S } ^ { k - 1 }$ and $h ( z ) , h ( \tilde { z } ) \in \mathcal { Z } ^ { \prime } = \mathbb { S } ^ { k - 1 }$ , we have $\| z \| ^ { 2 } = \| \tilde { z } \| ^ { 2 } = \| h ( z ) \| ^ { 2 } = \| h ( \tilde { z } ) \| ^ { 2 } = 1$ .

Expanding the squared Euclidean distance:

$$
\left\| z - \tilde {z} \right\| ^ {2} = \left\| z \right\| ^ {2} - 2 z ^ {\top} \tilde {z} + \left\| \tilde {z} \right\| ^ {2} = 2 - 2 z ^ {\top} \tilde {z} \tag {20}
$$

Similarly, for the mapped points:

$$
\left\| h (z) - h (\tilde {z}) \right\| ^ {2} = \left\| h (z) \right\| ^ {2} - 2 h (z) ^ {\top} h (\tilde {z}) + \left\| h (\tilde {z}) \right\| ^ {2} = 2 - 2 h (z) ^ {\top} h (\tilde {z}) \tag {21}
$$

By Theorem A.1, we have $z ^ { \top } \tilde { z } = h ( z ) ^ { \top } h ( \tilde { z } ) ~ P _ { Z , \tilde { Z } ^ { - \mathbf { a } . \mathbf { S } } }$ . Therefore:

$$
\left\| h (z) - h (\tilde {z}) \right\| ^ {2} = 2 - 2 h (z) ^ {\top} h (\tilde {z}) = 2 - 2 z ^ {\top} \tilde {z} = \left\| z - \tilde {z} \right\| ^ {2} \tag {22}
$$

holds $P _ { Z , \tilde { Z } ^ { - \mathrm { { a l m o s t } \ s u r e l y } } }$ .

The distance preservation property established above holds only almost everywhere with respect to the joint distribution. To conclude that the optimal recovery map is a global orthogonal transformation, we apply the probabilistic generalization of the Mazur-Ulam theorem (Zaliaduonis & Gatidis, 2026), which shows that isometries holding almost everywhere on probability spaces can be extended to global isometries on the entire space.

Proof of Theorem 4.4. By Corollary A.2, the optimal recovery map $h : \mathcal { Z } \to \mathcal { Z }$ preserves distances almost everywhere with respect to the joint distribution $P _ { Z , \tilde { Z } }$ . Specifically, there exists a set $N \subset \mathcal { Z } \times \mathcal { Z }$ with $P _ { Z , \tilde { Z } } ( N ) = 0$ such that

$$
\| h (z) - h (\tilde {z}) \| = \| z - \tilde {z} \| \quad \text {   for   all   } (z, \tilde {z}) \in (\mathcal {Z} \times \mathcal {Z}) \setminus N.
$$

Since the marginal law $P _ { Z }$ is uniform on the sphere and has full support, h preserves pairwise distances on a set of full $P _ { Z } \mathrm { - m e a s u r e }$ .

Applying the probabilistic Mazur-Ulam theorem (Zaliaduonis & Gatidis, 2026), which extends almost-everywhere isometries on probability spaces to global isometries, there exists $H : \mathbb { R } ^ { k }  \mathbb { R } ^ { k }$ of the form $H ( x ) = A x + b$ with $A \in O ( k )$ and $b \in \mathbb { R } ^ { k }$ , such that $h ( z ) = H ( z )$ for PZ -almost all $z \in { \mathcal { Z } }$ .

Since both $\mathcal { Z }$ and $\mathcal { Z } ^ { \prime }$ are unit spheres centered at the origin, and H agrees almost everywhere with $h : \mathcal { Z } \to \mathcal { Z } ^ { \prime }$ , the translation component must vanish $( b = 0 )$ and the linear part must preserve the unit sphere. This implies that A is an orthogonal matrix, $\mathrm { i . e . , } A ^ { \top } A = I ,$ .

Therefore, the optimal recovery map h coincides with an orthogonal transformation almost everywhere:

$$
h (z) = A z \quad \text {   for   } \mu \text {-almost all   } z \in \mathcal {Z},
$$

where $A \in O ( k )$ is an orthogonal matrix.

![](images/8be31600009eca23f7d5aa70b244403aa9104bf175939c5bf6725dabd61a0ecd.jpg)

# A.2. Proofs for Reconstruction Under Violated Diversity

When the diversity condition is violated, the conditional law is constrained to a lower-dimensional submanifold. We decompose each latent vector $z \in \mathcal { Z } = \mathbb { S } ^ { k - 1 }$ into two components:

$$
z = (u, v) ^ {\top} \in \mathbb {R} ^ {m} \times \mathbb {R} ^ {\ell}, \quad \text { where } m + \ell = k \text { and } m, \ell > 0.
$$

Here, $u \in \mathbb { R } ^ { m }$ represents latent dimensions that remain fixed under conditional sampling, while $v \in \mathbb { R } ^ { \ell }$ represents dimensions that can vary. Let $K ( z ) = \{ \tilde { z } \in \mathbb { S } ^ { k - 1 } : \tilde { u } = u \} \cong \mathbb { S } _ { r ( z ) } ^ { \ell - 1 }$ , where $r ( z ) = \sqrt { 1 - \| u \| ^ { 2 } }$ . Let $\sigma _ { K \left( z \right) }$ denote intrinsic surface measure on $K ( z )$ . The constrained conditional is the probability measure $P _ { \tilde { Z } | z } ^ { K }$ with density Z˜|z

$$
\frac {d P _ {\tilde {Z} | z} ^ {K}}{d \sigma_ {K (z)}} (\tilde {z}) = \frac {e ^ {\kappa \tilde {z} ^ {\top} z}}{\int_ {K (z)} e ^ {\kappa z ^ {\top} z ^ {\prime}} d \sigma_ {K (z)} \left(z ^ {\prime}\right)}. \tag {23}
$$

It is singular with respect to ambient spherical measure on $\mathbb { S } ^ { k - 1 }$ , but absolutely continuous with respect to the intrinsic measure on $K ( z )$ .

Despite this modification, the asymptotic relationship between the contrastive loss and cross-entropy minimization remains valid.

Theorem A.3 (Asymptotic Equivalence Under Violated Diversity). Given the constrained conditional probability measure defined above, the marginal law $P _ { Z }$ uniform on $\mathcal Z = \mathbb S ^ { k - 1 }$ , temperature $\tau > 0 ,$ , and number of negative samples $M > 0 ,$ , as $M \to \infty ;$

$$
\lim _ {M \to \infty} \mathcal {L} _ {C L} (h; \tau , M) - \log M + \log | \mathcal {Z} | = \mathbb {E} _ {z \sim P _ {Z}} [ H (P _ {\tilde {Z} | z} ^ {K}, Q _ {h, z}) ]
$$

Proof. The proof follows identically to Theorem 3.3, as the asymptotic analysis relies only on the law of large numbers and properties of the logarithm, not on the specific structure of the ground-truth conditional law. See (Wang & Isola, 2020) for the full argument. □

The critical distinction arises when comparing global values of the limiting objective. Under violated diversity, orthogonal recovery maps are not global minimizers.

Proof of Theorem 4.5. Let σ denote the normalized spherical measure on $\mathcal { Z } = \mathbb { S } ^ { k - 1 }$ . By Theorem $\mathbf { A . } 3 ,$ it suffices to compare the asymptotic contrastive loss up to constants independent of h:

$$
\mathcal {L} (h) = - \frac {1}{\tau} \mathbb {E} _ {(z, \tilde {z}) \sim P _ {\mathrm{pos}}} \left[ h (z) ^ {\top} h (\tilde {z}) \right] + \mathbb {E} _ {z \sim \sigma} \log \int_ {\mathbb {S} ^ {k - 1}} \exp \left(\frac {h (z) ^ {\top} h (z ^ {\prime})}{\tau}\right) d \sigma (z ^ {\prime}).
$$

For $\lambda > 0 .$ , define

$$
h _ {\lambda} (u, v) := \frac {(u , \lambda v)}{\sqrt {\| u \| ^ {2} + \lambda^ {2} \| v \| ^ {2}}}.
$$

$\mathrm { A t } \lambda = 1 , h _ { \lambda }$ is the identity. For $\lambda \neq 1 , h _ { \lambda }$ is not orthogonal because it changes inner products between points with different relative invariant and variant components.

Write the alignment and uniformity terms as

$$
\begin{array}{l} A (\lambda) := \mathbb {E} _ {(z, \tilde {z}) \sim P _ {\mathrm{pos}}} \left[ h _ {\lambda} (z) ^ {\top} h _ {\lambda} (\tilde {z}) \right], \\ U (\lambda) := \mathbb {E} _ {z \sim \sigma} \log \int_ {\mathbb {S} ^ {k - 1}} \exp \left(\frac {h _ {\lambda} (z) ^ {\top} h _ {\lambda} (z ^ {\prime})}{\tau}\right) d \sigma (z ^ {\prime}). \\ \end{array}
$$

Then $\mathcal { L } ( h _ { \lambda } ) = - \tau ^ { - 1 } A ( \lambda ) + U ( \lambda )$ .

We first show that shrinking the variant component improves alignment to first order. Under the constrained positive-pair distribution, $\tilde { u } = u .$ . Since $z , \tilde { z } \in \mathbb { S } ^ { k - 1 }$ , this implies $\| v \| = \| \tilde { v } \|$ . Therefore

$$
h _ {\lambda} (z) ^ {\top} h _ {\lambda} (\tilde {z}) = \frac {\| u \| ^ {2} + \lambda^ {2} v ^ {\top} \tilde {v}}{\| u \| ^ {2} + \lambda^ {2} \| v \| ^ {2}}.
$$

With $t = \lambda ^ { 2 }$ and

$$
f (t) = \frac {\| u \| ^ {2} + t v ^ {\top} \tilde {v}}{\| u \| ^ {2} + t \| v \| ^ {2}},
$$

we have

$$
f ^ {\prime} (t) = \frac {\| u \| ^ {2} (v ^ {\top} \tilde {v} - \| v \| ^ {2})}{(\| u \| ^ {2} + t \| v \| ^ {2}) ^ {2}}.
$$

By Cauchy-Schwarz, $v ^ { \top } \tilde { v } \leq \| v \| ^ { 2 }$ , with strict inequality whenever $v \neq \tilde { v }$ . The constrained vMF conditional is nondegenerate for $\kappa > 0 .$ , so $\boldsymbol { v } \neq \tilde { \boldsymbol { v } }$ on a set of positive $P _ { \mathrm { p o s } ^ { - } \mathrm { m e a s u r e } . }$ and $\| u \| > 0$ for σ-almost every z when $m > 0$ . Hence $A ^ { \prime } ( 1 ) < 0 .$ , and for some $c > 0$ ,

$$
A (\lambda) - A (1) = c (1 - \lambda) + o (1 - \lambda) \quad \text { as } \lambda \uparrow 1.
$$

We next show that the uniformity term has zero first derivative at the identity. Let

$$
g (z) := \left. \frac {d}{d \lambda} h _ {\lambda} (z) \right| _ {\lambda = 1} = (- \| v \| ^ {2} u, \| u \| ^ {2} v),
$$

so $z ^ { \top } g ( z ) = 0 . \operatorname { A t } \lambda = 1$ , the inner integral

$$
C := \int_ {\mathbb {S} ^ {k - 1}} \exp (z ^ {\top} z ^ {\prime} / \tau) d \sigma (z ^ {\prime})
$$

is independent of z by rotational symmetry. Differentiating under the integral, which is justified by smoothness and compactness of the sphere,

$$
U ^ {\prime} (1) = \frac {1}{\tau C} \iint \exp (z ^ {\top} z ^ {\prime} / \tau) \left(g (z) ^ {\top} z ^ {\prime} + z ^ {\top} g (z ^ {\prime})\right) d \sigma (z ^ {\prime}) d \sigma (z).
$$

For any fixed $a \in \mathbb { S } ^ { k - 1 }$ , rotational symmetry gives

$$
\int_ {\mathbb {S} ^ {k - 1}} z \exp (a ^ {\top} z / \tau) d \sigma (z) = \alpha a
$$

for some scalar α. Applying this identity to each of the two terms above and using $z ^ { \top } g ( z ) = 0$ yields $U ^ { \prime } ( 1 ) = 0$ . Thus

$$
U (\lambda) - U (1) = o (1 - \lambda) \quad \text {   as   } \lambda \uparrow 1.
$$

Combining the two expansions,

$$
\mathcal {L} (h _ {\lambda}) - \mathcal {L} (I) = - \frac {1}{\tau} (A (\lambda) - A (1)) + (U (\lambda) - U (1)) = - \frac {c}{\tau} (1 - \lambda) + o (1 - \lambda) <   0
$$

for all $\lambda < 1$ sufficiently close to 1.

Finally, every orthogonal map ${ \tilde { h } } \in O ( k )$ preserves inner products and the uniform spherical measure, so $\mathcal { L } ( \tilde { h } ) = \mathcal { L } ( I )$ . Taking $h = h _ { \lambda }$ for any $\lambda < 1$ sufficiently close to 1 gives

$$
\mathcal {L} _ {\mathrm{CL}} (h) <   \mathcal {L} _ {\mathrm{CL}} (\tilde {h}), \quad \forall \tilde {h} \in O (k),
$$

in the asymptotic regime $M \to \infty$

# A.3. Proofs for the Corrected Model

To address the support mismatch, we modify the model conditional to incorporate the same constraint structure:

$$
q _ {h, z} ^ {K} (\tilde {z}) = \frac {e ^ {h (z) ^ {\top} h (\tilde {z}) / \tau}}{\int_ {K (z)} e ^ {h (z) ^ {\top} h (z ^ {\prime}) / \tau} d \sigma_ {K (z)} \left(z ^ {\prime}\right)}, \quad \tilde {z} \in K (z). \tag {24}
$$

This modification ensures that supp $( Q _ { h , z } ^ { K } ) = \mathrm { s u p p } ( P _ { \tilde { Z } | z } ^ { K } )$ , eliminating the support mismatch.

Theorem A.4 (Asymptotic Form of Modified Contrastive Loss). Under the corrected model conditional, where negative samples are drawn uniformly from the constrained manifold $K ( z )$ rather than from the full sphere Z, the asymptotic contrastive loss takes the form:

$$
\lim _ {M \to \infty} \mathcal {L} (h, \tau , M) - \log (M) = \mathbb {E} _ {z \sim P _ {Z}} [ H (P _ {\tilde {Z} | z} ^ {K}, Q _ {h, z} ^ {K}) ] - \mathbb {E} _ {z \sim P _ {Z}} [ \log (| K (z) |) ]
$$

The proof follows the same steps as in (Zimmermann et al., 2021), but with conditionals defined on $K ( z )$ .

Proof. Step 1: Cross-entropy decomposition. The cross-entropy between the true and model conditionals, both defined on $K ( z )$ , is:

$$
H (P _ {\tilde {Z} | z} ^ {K}, Q _ {h, z} ^ {K}) = - \mathbb {E} _ {\tilde {z} \sim P _ {\tilde {Z} | z} ^ {K}} [ \log q _ {h, z} ^ {K} (\tilde {z}) ] \tag {25}
$$

$$
= - \mathbb {E} _ {\tilde {z} \sim P _ {\tilde {Z} | z} ^ {K}} \left[ \log \left(\frac {e ^ {h (z) ^ {\top} h (\tilde {z}) / \tau}}{\int_ {K (z)} e ^ {h (z) ^ {\top} h (z ^ {\prime}) / \tau} d \sigma_ {K (z)} \left(z ^ {\prime}\right)}\right) \right]. \tag {26}
$$

$$
H (P _ {\tilde {Z} | z} ^ {K}, Q _ {h, z} ^ {K}) = - \mathbb {E} _ {\tilde {z} \sim P _ {\tilde {Z} | z} ^ {K}} \left[ \frac {1}{\tau} h (z) ^ {\top} h (\tilde {z}) - \log C _ {h} (z) \right] \tag {27}
$$

$$
= - \frac {1}{\tau} \mathbb {E} _ {\tilde {z} \sim P _ {\tilde {Z} | z} ^ {K}} [ h (z) ^ {\top} h (\tilde {z}) ] + \log C _ {h} (z) \tag {28}
$$

where $\begin{array} { r } { C _ { h } ( z ) = \int _ { K ( z ) } e ^ { h ( z ) ^ { \top } h ( z ^ { \prime } ) / \tau } d \sigma _ { K ( z ) } ( z ^ { \prime } ) } \end{array}$ is the normalizing constant over the constrained manifold.

Step 2: Normalizing constant estimation. Using the fact that the uniform distribution on $K ( z )$ has density $1 / | K ( z ) |$ :

$$
C _ {h} (z) = \int_ {K (z)} e ^ {h (z) ^ {\top} h (z ^ {\prime}) / \tau} d \sigma_ {K (z)} (z ^ {\prime}) = | K (z) | \cdot \mathbb {E} _ {z ^ {\prime} \sim \mathrm{U} (K (z))} \left[ e ^ {h (z) ^ {\top} h (z ^ {\prime}) / \tau} \right] \tag {29}
$$

Step 3: Final form. Substituting the estimate of $C _ { h } ( z )$ back into the cross-entropy expression and splitting the logarithm:

$$
H (P _ {\tilde {Z} | z} ^ {K}, Q _ {h, z} ^ {K}) = - \frac {1}{\tau} \mathbb {E} _ {\tilde {z} \sim P _ {\tilde {Z} | z} ^ {K}} [ h (z) ^ {\top} h (\tilde {z}) ] \tag {30}
$$

$$
+ \log \mathbb {E} _ {z ^ {\prime} \sim \mathrm{U} (K (z))} \left[ e ^ {h (z) ^ {\top} h (z ^ {\prime}) / \tau} \right] + \log | K (z) | \tag {31}
$$

Taking expectations over $z \sim P _ { Z }$ yields the stated result.

Proof of Theorem 4.6. With the corrected model conditional, the supports of $P _ { \tilde { Z } | z } ^ { K }$ and $Q _ { h , z } ^ { K }$ now match: both are restricted to the submanifold $K ( z )$ . Within this constrained setting, the analysis proceeds analogously to Theorem A.1.

For any $h \in O ( k )$ , we have $h ( z ) ^ { \top } h ( \tilde { z } ) = z ^ { \top } \tilde { z }$ for all $z , \tilde { z } \in \mathcal { Z }$ . This means the model conditional $Q _ { h , . } ^ { K }$ z exactly matches the true conditional $P _ { \tilde { Z } | z } ^ { K }$ on the constrained manifold $K ( z )$ :

$$
q _ {h, z} ^ {K} (\tilde {z}) = \frac {e ^ {z ^ {\top} \tilde {z} / \tau}}{\int_ {K (z)} e ^ {z ^ {\top} z ^ {\prime} / \tau} d \sigma_ {K (z)} (z ^ {\prime})} = \frac {d P _ {\tilde {Z} | z} ^ {K}}{d \sigma_ {K (z)}} (\tilde {z})
$$

when $\kappa = 1 / \tau$ .

Since matching distributions achieves zero KL divergence and hence minimal cross-entropy, any orthogonal transformation minimizes the asymptotic contrastive loss. □

Although the corrected objective admits orthogonal solutions, it does not guarantee uniqueness. The following theorem shows that multiple equivalent solutions exist.

Theorem A.5 (Equivalence of Feature Extractors Under Constrained Sampling). Given a data-generating process $g : { \mathcal { Z } } $ X , uniform marginal law $P _ { Z } ,$ , ground-truth conditional law $P _ { \tilde { Z } | } { } _ { i }$ with density $p ( \tilde { z } | z )$ , and model conditional law $Q _ { h , z }$ with density $q _ { h } ( \tilde { z } | z )$ that define the expected cross-entropy loss

$$
\mathcal {L} _ {h} = \mathbb {E} _ {z \sim P _ {Z}} [ H (P _ {\tilde {Z} | z}, Q _ {h, z}) ],
$$

let $m : \mathcal { Z }  \mathcal { Z }$ be an invertible mapping that preserves the marginal law and the ground-truth conditional density:

$$
m _ {\#} P _ {Z} = P _ {Z}, \qquad p (m (\tilde {z}) | m (z)) = p (\tilde {z} | z) \quad \forall z, \tilde {z} \in \mathcal {Z}.
$$

Then any two mappings $h _ { 1 } : = f _ { 1 } \circ g$ and h2 := f2 ◦ g with $h _ { 2 } ( z ) : = h _ { 1 } ( m ( z ) )$ are equivalent, i.e., $\mathcal { L } _ { h _ { 1 } } = \mathcal { L } _ { h _ { 2 } }$

Proof. Starting with the cross-entropy loss for $h _ { 2 } { \mathrm { : } }$

$$
\mathcal {L} _ {h _ {2}} = \mathbb {E} _ {z \sim P _ {Z}} \left[ \mathbb {E} _ {\tilde {z} \sim P _ {\tilde {Z} | z}} [ - \log q _ {h _ {2}} (\tilde {z} | z) ] \right] \tag {32}
$$

Using the definition of h2:

$$
= \mathbb {E} _ {z \sim P _ {Z}} \left[ \mathbb {E} _ {\tilde {z} \sim P _ {\tilde {Z} | z}} [ - \log q _ {h _ {1}} (m (\tilde {z}) | m (z)) ] \right] \tag {33}
$$

Since m preserves the conditional density:

$$
= \mathbb {E} _ {z \sim P _ {\bar {Z}}} \left[ \mathbb {E} _ {m (\tilde {z}) \sim P _ {\bar {Z} | m (z)}} [ - \log q _ {h _ {1}} (m (\tilde {z}) | m (z)) ] \right] \tag {34}
$$

Since $m _ { \# } P _ { Z } = P _ { Z }$ , we can change variables in the outer expectation:

$$
= \mathbb {E} _ {m (z) \sim P _ {Z}} \left[ \mathbb {E} _ {m (\tilde {z}) \sim P _ {\tilde {Z} | m (z)}} [ - \log q _ {h _ {1}} (m (\tilde {z}) | m (z)) ] \right] \tag {35}
$$

Finally, because m is invertible:

$$
= \mathbb {E} _ {z \sim P _ {\bar {Z}}} \left[ \mathbb {E} _ {\tilde {z} \sim P _ {\bar {Z} | z}} [ - \log q _ {h _ {1}} (\tilde {z} | z) ] \right] = \mathcal {L} _ {h _ {1}} \tag {36}
$$

This theorem reveals a fundamental non-uniqueness in the solution space of the modified contrastive learning objective. Any conditional-preserving transformation of the latent space yields identical loss values, meaning that without additional constraints through inductive bias, the objective cannot distinguish between semantically meaningful representations and arbitrary rearrangements that preserve only local structure within constrained manifolds.

# B. Generative Processes and Encoder Architectures

This section describes the generative processes and encoder architectures used in our synthetic experiments.

# B.1. Generative Processes

We employ five generative processes $g : \mathbb { S } ^ { d - 1 }  \mathbb { R } ^ { D }$ of varying complexity to test our theoretical predictions across different data-generating mechanisms.

Algorithm 1 Identity Process   
Require: $z\in \mathbb{S}^{d - 1}$ 1: $x\gets z$ 2: return $x\in \mathbb{R}^d$

Algorithm 2 Linear Process   
Require: $z \in S^{d-1}$ , weight matrix $W \in R^{D \times d}$ with $\text{rank}(W) = d$ 1: $x \leftarrow Wz$ 2: return $x \in R^{D}$

Algorithm 3 Spiral Rotation Process   
Require: $z = (z_{1}, z_{2}, z_{3}) \in \mathbb{S}^{2}$ , period n
1: Compute rotation angle: $\theta \leftarrow n\pi z_{3}$ 2: Apply 2D rotation to first two coordinates:
3: $x_{1} \leftarrow \cos(\theta)z_{1} - \sin(\theta)z_{2}$ 4: $x_{2} \leftarrow \sin(\theta)z_{1} + \cos(\theta)z_{2}$ 5: $x_{3} \leftarrow z_{3}$ 6: return $x = (x_{1}, x_{2}, x_{3}) \in \mathbb{R}^{3}$

Algorithm 4 Patches Process   
Require: $z = (z_{1}, z_{2}, z_{3}) \in \mathbb{S}^{2}$ , number of slices K

1: Step 1: Apply piecewise rotation based on $z_{3}$ 2: Determine bucket $k \leftarrow \lfloor(z_{3} + 1) \cdot K/2 \rfloor$ 3: Compute angle $\theta_{k} \leftarrow -\pi / \max(1, K - k)$ 4: $z' \leftarrow R_{xy}(\theta_{k}) \cdot z$ {Rotate in $(x, y)$ plane}

5: Step 2: Apply 3D rotation (pitch = $\pi/2$ )

6: $z'' \leftarrow R_{y}(\pi/2) \cdot z'$ 7: Step 3: Apply second piecewise rotation

8: Determine new bucket, apply rotation as in Step 1

9: return $x \in R^{3}$

Algorithm 5 Invertible MLP Process (Hyvarinen & Morioka ¨ , 2016)   
Require: $z \in S^{d-1}$ , MLP layers $\{W_{i}, b_{i}\}_{i=1}^{L}$ with conditioning
1: $h_{0} \leftarrow z$ 2: for i = 1 to L do
3: $h_{i} \leftarrow \sigma(W_{i}h_{i-1} + b_{i}) \left\{\sigma = \text{LeakyReLU}\right\}$ 4: end for
5: return $x = h_{L} \in R^{D}$

# B.2. Encoder Architectures

We compare two classes of encoders: a generic MLP encoder representing low inductive bias, and inverse encoders designed to mirror the structure of each generative process, representing high inductive bias.

Algorithm 6 MLP Encoder (Low Inductive Bias)   
Require: $x \in R^{D}$ , hidden dims [128, 256, 256, 256, 128]
1: $h_{0} \leftarrow x$ 2: for i = 1 to L - 1 do
3: $h_{i} \leftarrow \text{ReLU}(\text{BatchNorm}(W_{i}h_{i-1} + b_{i}))$ 4: end for
5: $z' \leftarrow W_{L}h_{L-1} + b_{L}$ 6: $z \leftarrow z' / \|z'\|_{2}$ {Project to sphere}
7: return $z \in S^{d-1}$

Algorithm 7 Inverse Linear Encoder (High Inductive Bias)   
Require: $x \in R^{D}$ , learnable $W \in R^{d \times D}$ , $b \in R^{d}$ 1: $z' \leftarrow Wx + b$ 2: $z \leftarrow z' / \|z'\|_{2}$ 3: return $z \in S^{d-1}$

Algorithm 8 Inverse Spiral Encoder (High Inductive Bias)   
Require: $x \in R^{3}$ , period n
1: Predict rotation control: $c \leftarrow \text{MLP}_{\text{rot}}(x)$ {3-layer MLP}
2: Extract spatial components: $(x_{1}, x_{2}) \leftarrow x_{1:2}$ 3: Compute inverse rotation: $\theta \leftarrow -n\pi c$ 4: Apply inverse rotation:
5: $z_{1} \leftarrow \cos(\theta)x_{1} - \sin(\theta)x_{2}$ 6: $z_{2} \leftarrow \sin(\theta)x_{1} + \cos(\theta)x_{2}$ 7: $z' \leftarrow (z_{1}, z_{2}, c)$ 8: $z \leftarrow z' / \|z' \|_{2}$ 9: return $z \in S^{2}$

Algorithm 9 Inverse Patches Encoder (High Inductive Bias)   
Require: $x \in R^{3}$ , number of slices K

1: Predict original z-coordinate: $z_{\text{pred}} \leftarrow \tanh(\text{MLP}_{z}(x))$ 2: Predict bucket probabilities: $w \leftarrow \text{softmax}(\text{MLP}_{\text{bucket}}(x))$ 3: Step 1: Inverse second piecewise rotation

4: $x_{1} \leftarrow \sum_{k=0}^{K-1} w_{k} \cdot R_{xy}(-\theta_{k}) \cdot x$ {Soft inverse}

5: Step 2: Inverse 3D rotation

6: $x_{2} \leftarrow R_{y}(-\pi/2) \cdot x_{1}$ 7: Step 3: Inverse first piecewise rotation

8: $x_{3} \leftarrow \sum_{k=0}^{K-1} w_{k} \cdot R_{xy}(-\theta_{k}) \cdot x_{2}$ 9: Replace z-coordinate: $x_{\text{rec}} \leftarrow (x_{3,1}, x_{3,2}, z_{\text{pred}})$ 10: $z \leftarrow x_{\text{rec}} / \|x_{\text{rec}}\|_{2}$ 11: return $z \in S^{2}$

# C. Sampling Procedures

We use standard techniques for sampling from the uniform distribution on the sphere and the von Mises-Fisher distribution.

Algorithm 10 Uniform Sampling from $\mathbb { S } ^ { d - 1 }$   
Require: Dimension $d \in N$ 1: Draw $z_{i} \sim \mathcal{N}(0,1)$ independently for $i = 1, \ldots, d$ 2: $z \leftarrow (z_{1}, \ldots, z_{d})^{\top}$ 3: $v \leftarrow z / \|z\|_{2}$ 4: return $v \in S^{d-1}$

Algorithm 11 von Mises-Fisher Sampling (Wood, 1994)   
Require: Mean direction $\mu \in \mathbb{S}^{d - 1}$ , concentration $\kappa > 0$ 1: $p \leftarrow d - 1$ 2: $b \leftarrow p / (\sqrt{4\kappa^2 + p^2} + 2\kappa)$ 3: $x \leftarrow (1 - b) / (1 + b)$ 4: $c \leftarrow \kappa x + p\log (1 - x^2)$ 5: repeat

6: Sample $t \sim \mathrm{Beta}(p/2, p/2)$ 7: $w \leftarrow (1 - (1 + b)t) / (1 - (1 - b)t)$ 8: Sample $u \sim \mathrm{Uniform}(0, 1)$ 9: until $\log(u) \leq \kappa w + p\log(1 - xw) - c$ 10: Sample $g \sim \mathcal{N}(0, I_d)$ 11: $v \leftarrow g - (g^\top \mu)\mu$ {Project out $\mu$ component}

12: $v \leftarrow v / \|v\|_2$ 13: $s \leftarrow \sqrt{1 - w^2} \cdot v + w \cdot \mu$ 14: return $s \sim \mathrm{vMF}(\mu, \kappa)$

Algorithm 12 Conditional Sampling: Diversity Holds   
Require: Anchor $z \in \mathbb{S}^{d-1}$ , concentration $\kappa$ 1: $\tilde{z} \leftarrow \text{vMF}(z, \kappa)$ {Sample positive using Algorithm 11}

2: return $\tilde{z} \in \mathbb{S}^{d-1}$

Algorithm 13 Conditional Sampling: Diversity Violated   
Require: Anchor $z = (u, v) \in \mathbb{S}^{d-1}$ , fixed dimensions $d_{fixed}$ , concentration $\kappa$ 1: $u \leftarrow z_{1:d_{\text{fixed}}}$ {Fixed component}

2: $v \leftarrow z_{d_{\text{fixed}}+1:d}$ {Varying component}

3: $r \leftarrow \|v\|_{2}$ {Radius of sub-sphere}

4: if r > 0 then

5: $\hat{v} \leftarrow v/r$ {Normalize to sub-sphere}

6: $\tilde{v} \leftarrow \text{vMF}(\hat{v}, \kappa)$ {Sample on sub-sphere}

7: $\tilde{v} \leftarrow r \cdot \tilde{v}$ {Scale back}

8: else

9: $\tilde{v} \leftarrow v$ 10: end if

11: $\tilde{z} \leftarrow (u, \tilde{v})$ {Concatenate fixed and sampled}

12: return $\tilde{z} \in S^{d-1}$

Algorithm 14 Adapted InfoNCE with Same-anchor Negatives   
Require: Encoder h, batch $\{x_{1},\ldots,x_{N}\}$ , stochastic augmentation T, temperature $\tau$ , negatives per anchor M
1: for each anchor $x_{i}$ do
2: Draw positive view $\tilde{x}_{i} \leftarrow \mathcal{T}(x_{i})$ 3: for j = 1 to M do
4: Draw same-anchor negative $x_{i,j}^{-} \leftarrow \mathcal{T}(x_{i})$ independently
5: end for
6: Compute $\mathcal{L}_{i} = -\log \frac{\exp(h(x_{i})^{\top} h(\tilde{x}_{i}) / \tau)}{\exp(h(x_{i})^{\top} h(\tilde{x}_{i}) / \tau) + \sum_{j=1}^{M} \exp(h(x_{i})^{\top} h(x_{i,j}^{-}) / \tau)}.$ 7: end for
8: return $L = \frac{1}{N} \sum_{i=1}^{N} L_{i}$

Algorithm 14 approximates sampling negatives from $K ( z )$ in observation space. Each call to $\tau$ uses independent randomness conditional on the same anchor $x _ { i }$ , which is the observation-space analogue of drawing conditionally independent samples from $P _ { \tilde { Z } | z _ { i } }$ . The same sampling family is applied repeatedly to the same anchor, so all views preserve the invariant component encoded by the mechanism while varying the remaining components. This changes the negative-sampling task relative to standard InfoNCE, which samples negatives from other instances. In our framework this is consistent with the goal of recovering latent distance structure rather than discriminating instance identity, but it can trade crossinstance discrimination signal for better support matching. The approximation also samples from the augmentation-induced distribution on $K ( z )$ rather than uniformly from $K ( z )$ .

# D. Evaluation Metrics

We evaluate latent space reconstruction quality using three complementary metrics, following standard practice in the identifiability literature (Hyvarinen & Morioka ¨ , 2016; Zimmermann et al., 2021).

Algorithm 15 Linear Identifiability $( R ^ { 2 } )$   
Require: Ground-truth latents $Z = \{z_i\}_{i=1}^n$ , recovered latents $\hat{Z} = \{\hat{z}_i\}_{i=1}^n$ 1: Fit linear regression: $\hat{z} = Az + b$ minimizing $\sum_{i} \|z_i - (A\hat{z}_i + b)\|^2$ 2: Compute predictions: $\tilde{z}_i \leftarrow A\hat{z}_i + b$ 3: Compute total variance: $SS_{tot} \leftarrow \sum_{i=1}^n \|z_i - \bar{z}\|^2$ 4: Compute residual variance: $SS_{res} \leftarrow \sum_{i=1}^n \|z_i - \tilde{z}_i\|^2$ 5: $R^2 \leftarrow 1 - SS_{res}/SS_{tot}$ 6: return $R^2 \in (-\infty, 1]$ $\{1.0 = perfect linear recovery\}$

Algorithm 16 Mean Correlation Coefficient (MCC)   
Require: Ground-truth latents $Z \in R^{n \times d}$ , recovered latents $\hat{Z} \in R^{n \times d}$ 1: Compute correlation matrix $C \in R^{d \times d}$ :

2: $C_{ij} \leftarrow |\text{corr}(Z_{:,i}, \hat{Z}_{:,j})|$ 3: Find optimal assignment via Munkres algorithm:

4: $\pi^* \leftarrow \arg\max_\pi \sum_{i=1}^d C_{i,\pi(i)}$ 5: MCC $\leftarrow \frac{1}{d} \sum_{i=1}^d C_{i,\pi^*(i)}$ 6: return MCC $\in [0,1]$ $\{1.0 = perfect factor alignment\}$

Algorithm 17 Angular Preservation Error (APE)   
Require: Ground-truth latents $Z = \{z_i\}_{i=1}^n$ , recovered latents $\hat{Z} = \{\hat{z}_i\}_{i=1}^n$ 1: Initialize APE $\leftarrow 0$ 2: for i = 1 to n do

3: for j = 1 to n, $j \neq i$ do

4: APE $\leftarrow$ APE + $|z_i^\top z_j - \hat{z}_i^\top \hat{z}_j|$ 5: end for

6: end for

7: APE $\leftarrow$ APE/(n(n-1))

8: return APE $\in [0, 2]$ {0.0 = perfect isometry}

<table><tr><td>Metric</td><td>Range</td><td>Measures</td><td>Optimal</td></tr><tr><td> $R^{2}$ </td><td> $(-\infty, 1]$ </td><td>Linear predictability</td><td>1.0</td></tr><tr><td>MCC</td><td> $[0, 1]$ </td><td>Factor alignment (up to permutation)</td><td>1.0</td></tr><tr><td>APE</td><td> $[0, 2]$ </td><td>Angular/geometric preservation</td><td>0.0</td></tr></table>

Table 2. Summary of evaluation metrics for latent space reconstruction.

# E. Additional Experimental Results

# E.1. Diversity Condition Holds

Table 3 reports MCC, APE, and final loss when the diversity condition is satisfied.

Table 3. Evaluation metrics when diversity condition holds. Results reported as mean ± std across 5 random seeds using MLP encoder. Lower APE is better. 

<table><tr><td>Generative Process</td><td>MCC</td><td>APE</td><td>Final Loss</td></tr><tr><td>Identity</td><td>0.775 ± 0.060</td><td>0.007 ± 0.001</td><td>7.388 ± 0.007</td></tr><tr><td>Linear</td><td>0.854 ± 0.083</td><td>0.007 ± 0.001</td><td>7.386 ± 0.006</td></tr><tr><td>Invertible MLP</td><td>0.822 ± 0.038</td><td>0.010 ± 0.002</td><td>7.383 ± 0.012</td></tr><tr><td>Patches</td><td>0.836 ± 0.060</td><td>0.031 ± 0.001</td><td>7.429 ± 0.019</td></tr><tr><td>Spiral</td><td>0.858 ± 0.041</td><td>0.012 ± 0.001</td><td>7.390 ± 0.011</td></tr></table>

# E.2. Diversity Condition Violated

Tables 4-6 compare three approaches when diversity is violated: standard InfoNCE, adapted InfoNCE (Section 4.3), and InfoNCE with inductive bias.

Table 4. Mean Correlation Coefficient (MCC) when diversity condition is violated. Results reported as mean ± std across 5 random seeds. 

<table><tr><td rowspan="2">Generative Process</td><td colspan="3">Diversity Condition Violated</td></tr><tr><td>InfoNCE</td><td>InfoNCE Adapted</td><td>InfoNCE + Ind. Bias</td></tr><tr><td>Identity</td><td>0.143 ± 0.016</td><td>0.607 ± 0.039</td><td>0.801 ± 0.102</td></tr><tr><td>Linear</td><td>0.111 ± 0.075</td><td>0.592 ± 0.085</td><td>0.841 ± 0.094</td></tr><tr><td>Invertible MLP</td><td>0.178 ± 0.116</td><td>0.569 ± 0.048</td><td>N/A</td></tr><tr><td>Patches</td><td>0.265 ± 0.018</td><td>0.624 ± 0.035</td><td>0.687 ± 0.014</td></tr><tr><td>Spiral</td><td>0.629 ± 0.294</td><td>0.602 ± 0.044</td><td>0.999 ± 0.001</td></tr></table>

Table 5. Angular Preservation Error (APE) when diversity condition is violated. Results reported as mean ± std across 5 random seeds. Lower is better. 

<table><tr><td rowspan="2">Generative Process</td><td colspan="3">Diversity Condition Violated</td></tr><tr><td>InfoNCE</td><td>InfoNCE Adapted</td><td>InfoNCE + Ind. Bias</td></tr><tr><td>Identity</td><td>0.322 ± 0.007</td><td>0.152 ± 0.002</td><td>0.047 ± 0.000</td></tr><tr><td>Linear</td><td>0.325 ± 0.012</td><td>0.155 ± 0.002</td><td>0.046 ± 0.000</td></tr><tr><td>Invertible MLP</td><td>0.305 ± 0.034</td><td>0.168 ± 0.003</td><td>N/A</td></tr><tr><td>Patches</td><td>0.283 ± 0.008</td><td>0.154 ± 0.002</td><td>0.109 ± 0.003</td></tr><tr><td>Spiral</td><td>0.136 ± 0.115</td><td>0.156 ± 0.002</td><td>0.016 ± 0.003</td></tr></table>

Table 6. Final InfoNCE loss when diversity condition is violated. Results reported as mean ± std across 5 random seeds. 

<table><tr><td rowspan="2">Generative Process</td><td colspan="3">Diversity Condition Violated</td></tr><tr><td>InfoNCE</td><td>InfoNCE Adapted</td><td>InfoNCE + Ind. Bias</td></tr><tr><td>Identity</td><td>5.709 ± 0.000</td><td>6.689 ± 0.007</td><td>6.058 ± 0.009</td></tr><tr><td>Linear</td><td>5.708 ± 0.001</td><td>6.697 ± 0.005</td><td>6.043 ± 0.009</td></tr><tr><td>Invertible MLP</td><td>5.715 ± 0.002</td><td>6.009 ± 0.006</td><td>N/A</td></tr><tr><td>Patches</td><td>5.850 ± 0.021</td><td>6.028 ± 0.009</td><td>6.222 ± 0.007</td></tr><tr><td>Spiral</td><td>5.949 ± 0.140</td><td>6.011 ± 0.012</td><td>6.075 ± 0.014</td></tr></table>

# E.3. Progressive Diversity Violation

Table 7 and Figure 5c examine how performance degrades as the diversity condition is progressively violated in a 10- dimensional latent space. The parameter $d _ { \mathrm { f i x e d } }$ denotes the number of dimensions held constant during positive pair sampling.

Table 7. Linear identifiability $( R ^ { 2 } )$ as a function of diversity violation severity for 10D latent space. $d _ { \mathrm { f i x e d } }$ denotes the number of dimensions held constant during positive pair sampling.

<table><tr><td> $d_{\text{fixed}}$ </td><td>Violation Ratio</td><td>Linear</td><td>Monomial</td></tr><tr><td>0</td><td>0.0</td><td>0.969 ± 0.047</td><td>0.990 ± 0.000</td></tr><tr><td>1</td><td>0.1</td><td>0.023 ± 0.011</td><td>0.015 ± 0.005</td></tr><tr><td>2</td><td>0.2</td><td>0.014 ± 0.003</td><td>0.015 ± 0.003</td></tr><tr><td>3</td><td>0.3</td><td>0.027 ± 0.009</td><td>0.034 ± 0.008</td></tr><tr><td>4</td><td>0.4</td><td>0.061 ± 0.021</td><td>0.042 ± 0.004</td></tr><tr><td>5</td><td>0.5</td><td>0.099 ± 0.019</td><td>0.097 ± 0.015</td></tr><tr><td>6</td><td>0.6</td><td>0.168 ± 0.017</td><td>0.185 ± 0.021</td></tr><tr><td>7</td><td>0.7</td><td>0.213 ± 0.035</td><td>0.284 ± 0.022</td></tr><tr><td>8</td><td>0.8</td><td>0.238 ± 0.049</td><td>0.348 ± 0.022</td></tr><tr><td>9</td><td>0.9</td><td>0.299 ± 0.047</td><td>0.425 ± 0.056</td></tr><tr><td>10</td><td>1.0</td><td>0.032 ± 0.004</td><td>0.086 ± 0.032</td></tr></table>

# E.4. Constraint Ratio Experiments

Tables 8-11 show identifiability metrics as a function of the constraint ratio $\rho ,$ which interpolates between standard InfoNCE $( \rho = 0 )$ and the fully corrected objective $( \rho = 1 )$ . The parameter $\rho$ controls the fraction of negative samples drawn from the constrained manifold $K ( z )$ versus the full sphere Z.

Table 8. Linear identifiability $( R ^ { 2 } )$ across all generative processes and encoder types as a function of constraint ratio $\rho .$ Results reported as mean across 5 seeds. 

<table><tr><td rowspan="2">ρ</td><td colspan="2">Identity</td><td colspan="2">Linear</td><td>InvMLP</td><td colspan="2">Patches</td><td colspan="2">Spiral</td></tr><tr><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td><td>MLP</td><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td></tr><tr><td>0.0</td><td>.053</td><td>.994</td><td>.112</td><td>.995</td><td>.286</td><td>.265</td><td>.871</td><td>.331</td><td>.996</td></tr><tr><td>0.1</td><td>.228</td><td>.997</td><td>.293</td><td>.997</td><td>.317</td><td>.651</td><td>.892</td><td>.731</td><td>.781</td></tr><tr><td>0.2</td><td>.306</td><td>.999</td><td>.299</td><td>.999</td><td>.350</td><td>.763</td><td>.878</td><td>.622</td><td>.963</td></tr><tr><td>0.3</td><td>.226</td><td>1.00</td><td>.201</td><td>1.00</td><td>.554</td><td>.820</td><td>.870</td><td>.797</td><td>.999</td></tr><tr><td>0.4</td><td>.502</td><td>.999</td><td>.452</td><td>.999</td><td>.666</td><td>.745</td><td>.859</td><td>.813</td><td>.842</td></tr><tr><td>0.5</td><td>.575</td><td>.997</td><td>.487</td><td>.997</td><td>.544</td><td>.737</td><td>.819</td><td>.822</td><td>.991</td></tr><tr><td>0.6</td><td>.525</td><td>.991</td><td>.461</td><td>.991</td><td>.564</td><td>.719</td><td>.645</td><td>.803</td><td>.831</td></tr><tr><td>0.7</td><td>.469</td><td>.979</td><td>.333</td><td>.979</td><td>.524</td><td>.731</td><td>.674</td><td>.661</td><td>.846</td></tr><tr><td>0.8</td><td>.402</td><td>.958</td><td>.350</td><td>.958</td><td>.485</td><td>.681</td><td>.680</td><td>.620</td><td>.826</td></tr><tr><td>0.9</td><td>.440</td><td>.905</td><td>.457</td><td>.904</td><td>.487</td><td>.675</td><td>.623</td><td>.637</td><td>.811</td></tr><tr><td>1.0</td><td>.626</td><td>.709</td><td>.650</td><td>.680</td><td>.597</td><td>.633</td><td>.656</td><td>.624</td><td>.572</td></tr></table>

Table 9. Mean Correlation Coefficient (MCC) across all generative processes and encoder types as a function of constraint ratio $\rho .$ Results reported as mean across 5 seeds. 

<table><tr><td rowspan="2">ρ</td><td colspan="2">Identity</td><td colspan="2">Linear</td><td>InvMLP</td><td colspan="2">Patches</td><td colspan="2">Spiral</td></tr><tr><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td><td>MLP</td><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td></tr><tr><td>0.0</td><td>.102</td><td>.769</td><td>.125</td><td>.859</td><td>.247</td><td>.275</td><td>.660</td><td>.315</td><td>.997</td></tr><tr><td>0.1</td><td>.342</td><td>.819</td><td>.364</td><td>.836</td><td>.350</td><td>.636</td><td>.648</td><td>.648</td><td>.777</td></tr><tr><td>0.2</td><td>.381</td><td>.815</td><td>.362</td><td>.838</td><td>.453</td><td>.670</td><td>.648</td><td>.632</td><td>.978</td></tr><tr><td>0.3</td><td>.321</td><td>.841</td><td>.336</td><td>.827</td><td>.590</td><td>.711</td><td>.631</td><td>.685</td><td>.999</td></tr><tr><td>0.4</td><td>.538</td><td>.807</td><td>.542</td><td>.811</td><td>.617</td><td>.664</td><td>.654</td><td>.801</td><td>.876</td></tr><tr><td>0.5</td><td>.594</td><td>.776</td><td>.537</td><td>.834</td><td>.533</td><td>.699</td><td>.628</td><td>.808</td><td>.995</td></tr><tr><td>0.6</td><td>.522</td><td>.863</td><td>.496</td><td>.818</td><td>.600</td><td>.660</td><td>.540</td><td>.796</td><td>.876</td></tr><tr><td>0.7</td><td>.499</td><td>.778</td><td>.394</td><td>.758</td><td>.561</td><td>.657</td><td>.584</td><td>.628</td><td>.868</td></tr><tr><td>0.8</td><td>.448</td><td>.704</td><td>.408</td><td>.775</td><td>.516</td><td>.649</td><td>.612</td><td>.542</td><td>.861</td></tr><tr><td>0.9</td><td>.463</td><td>.698</td><td>.483</td><td>.703</td><td>.467</td><td>.663</td><td>.560</td><td>.650</td><td>.841</td></tr><tr><td>1.0</td><td>.607</td><td>.593</td><td>.592</td><td>.590</td><td>.569</td><td>.624</td><td>.628</td><td>.602</td><td>.611</td></tr></table>

Table 10. Angular Preservation Error (APE) across all generative processes and encoder types as a function of constraint ratio $\rho .$ Results reported as mean across 5 seeds. Lower is better. 

<table><tr><td rowspan="2">ρ</td><td colspan="2">Identity</td><td colspan="2">Linear</td><td>InvMLP</td><td colspan="2">Patches</td><td colspan="2">Spiral</td></tr><tr><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td><td>MLP</td><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td></tr><tr><td>0.0</td><td>.325</td><td>.047</td><td>.310</td><td>.046</td><td>.265</td><td>.280</td><td>.112</td><td>.257</td><td>.027</td></tr><tr><td>0.1</td><td>.288</td><td>.035</td><td>.269</td><td>.036</td><td>.265</td><td>.187</td><td>.101</td><td>.134</td><td>.114</td></tr><tr><td>0.2</td><td>.267</td><td>.020</td><td>.268</td><td>.020</td><td>.257</td><td>.146</td><td>.105</td><td>.195</td><td>.038</td></tr><tr><td>0.3</td><td>.286</td><td>.004</td><td>.293</td><td>.005</td><td>.210</td><td>.126</td><td>.107</td><td>.136</td><td>.011</td></tr><tr><td>0.4</td><td>.223</td><td>.014</td><td>.235</td><td>.014</td><td>.176</td><td>.148</td><td>.111</td><td>.126</td><td>.067</td></tr><tr><td>0.5</td><td>.202</td><td>.034</td><td>.223</td><td>.034</td><td>.204</td><td>.149</td><td>.122</td><td>.120</td><td>.027</td></tr><tr><td>0.6</td><td>.212</td><td>.054</td><td>.228</td><td>.055</td><td>.197</td><td>.151</td><td>.164</td><td>.123</td><td>.089</td></tr><tr><td>0.7</td><td>.225</td><td>.079</td><td>.255</td><td>.079</td><td>.206</td><td>.145</td><td>.158</td><td>.161</td><td>.088</td></tr><tr><td>0.8</td><td>.238</td><td>.105</td><td>.249</td><td>.106</td><td>.216</td><td>.157</td><td>.152</td><td>.171</td><td>.101</td></tr><tr><td>0.9</td><td>.225</td><td>.138</td><td>.221</td><td>.138</td><td>.212</td><td>.152</td><td>.169</td><td>.165</td><td>.114</td></tr><tr><td>1.0</td><td>.152</td><td>.155</td><td>.155</td><td>.158</td><td>.168</td><td>.154</td><td>.160</td><td>.156</td><td>.190</td></tr></table>

Table 11. Final InfoNCE loss across all generative processes and encoder types as a function of constraint ratio $\rho .$ Results reported as mean across $^ { 5 }$ seeds. 

<table><tr><td rowspan="2">ρ</td><td colspan="2">Identity</td><td colspan="2">Linear</td><td>InvMLP</td><td colspan="2">Patches</td><td colspan="2">Spiral</td></tr><tr><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td><td>MLP</td><td>MLP</td><td>Inv</td><td>MLP</td><td>Inv</td></tr><tr><td>0.0</td><td>5.71</td><td>6.05</td><td>5.71</td><td>6.05</td><td>5.03</td><td>5.17</td><td>6.19</td><td>5.07</td><td>6.07</td></tr><tr><td>0.1</td><td>6.07</td><td>6.18</td><td>6.06</td><td>6.18</td><td>5.38</td><td>5.50</td><td>6.31</td><td>5.45</td><td>6.44</td></tr><tr><td>0.2</td><td>6.21</td><td>6.30</td><td>6.22</td><td>6.30</td><td>5.53</td><td>5.62</td><td>6.40</td><td>5.55</td><td>6.37</td></tr><tr><td>0.3</td><td>6.32</td><td>6.38</td><td>6.32</td><td>6.38</td><td>5.63</td><td>5.69</td><td>6.47</td><td>5.65</td><td>6.38</td></tr><tr><td>0.4</td><td>6.40</td><td>6.46</td><td>6.40</td><td>6.46</td><td>5.71</td><td>5.76</td><td>6.53</td><td>5.72</td><td>6.53</td></tr><tr><td>0.5</td><td>6.46</td><td>6.53</td><td>6.47</td><td>6.53</td><td>5.78</td><td>5.83</td><td>6.60</td><td>5.79</td><td>6.52</td></tr><tr><td>0.6</td><td>6.52</td><td>6.59</td><td>6.53</td><td>6.59</td><td>5.84</td><td>5.88</td><td>6.68</td><td>5.85</td><td>6.62</td></tr><tr><td>0.7</td><td>6.57</td><td>6.62</td><td>6.57</td><td>6.63</td><td>5.89</td><td>5.93</td><td>6.71</td><td>5.89</td><td>6.72</td></tr><tr><td>0.8</td><td>6.62</td><td>6.67</td><td>6.62</td><td>6.67</td><td>5.94</td><td>5.97</td><td>6.74</td><td>5.94</td><td>6.75</td></tr><tr><td>0.9</td><td>6.65</td><td>6.68</td><td>6.66</td><td>6.68</td><td>5.98</td><td>6.01</td><td>6.78</td><td>5.97</td><td>6.81</td></tr><tr><td>1.0</td><td>6.69</td><td>6.69</td><td>6.70</td><td>6.69</td><td>6.01</td><td>6.03</td><td>6.79</td><td>6.01</td><td>6.94</td></tr></table>

# E.5. Synthetic Validation Figures

Figure 5 provides visual summaries comparing MLP and inverse encoder performance across conditions.

![](images/295bc74228f9e63646e2fedc8e10463ae207fdf991a2e6622c1d918c51b09f44.jpg)

<details>
<summary>bar</summary>

| Category          | MLP   | Inverse |
| ----------------- | ----- | ------- |
| Diversity Holds   | 1.00  | 0.79    |
| Diversity Violated| 0.25  | 0.85    |
</details>

(a)

![](images/363d19a881ebf9c5d6049c8039c0c57795e37a1dc44cdc0b587182c24e61f814.jpg)

<details>
<summary>line</summary>

| Constraint Ratio (ρ) | MLP  | Inverse |
| -------------------- | ---- | ------- |
| 0.0                  | 0.30 | 0.06    |
| 0.2                  | 0.23 | 0.05    |
| 0.4                  | 0.21 | 0.04    |
| 0.6                  | 0.18 | 0.09    |
| 0.8                  | 0.20 | 0.12    |
| 1.0                  | 0.16 | 0.16    |
</details>

(b)

![](images/8bf9e00866967043cadb923af1583dae6be55f75efc81f2147275a5b0d8797d7.jpg)

<details>
<summary>line</summary>

| Number of Fixed Dimensions (dfixed) | Linear | Monomial |
| ---------------------------------- | ------ | -------- |
| 0                                  | 1.0    | 1.0      |
| 1                                  | 0.0    | 0.0      |
| 2                                  | 0.0    | 0.0      |
| 3                                  | 0.0    | 0.0      |
| 4                                  | 0.05   | 0.05     |
| 5                                  | 0.1    | 0.1      |
| 6                                  | 0.15   | 0.2      |
| 7                                  | 0.2    | 0.3      |
| 8                                  | 0.25   | 0.4      |
| 9                                  | 0.3    | 0.45     |
| 10                                 | 0.05   | 0.1      |
</details>

(c)   
Figure 5. Synthetic validation of theoretical predictions. (a) Linear identifiability for MLP and inverse encoders. When the diversity condition holds, MLP achieves $R ^ { 2 } = 1 . 0 0 ;$ ; when violated, it collapses to $R ^ { 2 } = 0 . \dot { 2 } 5$ while inverse encoders remain robust $( R ^ { 2 } = 0 . 8 3 )$ . (b) Angular preservation error vs. constraint ratio $\rho . \ \mathrm { A t } \ \rho = 1$ (corrected InfoNCE), MLP converges to inverse encoder performance. (c) Linear identifiability on ${ \mathbb S } ^ { 9 }$ vs. incremental violation of diversity condition; even $d _ { \mathrm { f i x e d } } = 1$ causes catastrophic failure. Results averaged across 5 generative processes; shaded regions indicate ±1 std.