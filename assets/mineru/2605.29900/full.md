# OVA-IB: One vs All Information Bottleneck for Multi-Modal Alignment

Tianchao Li1

Zhaolong Wei6

Shujian Yu2,3

Jeremy Gummeson6

Xinrui Zu2

Jack C.P. Cheng1

Robert Jenssen3,4,5

1Hong Kong University of Science and Technology, Hong Kong

2Vrije Universiteit Amsterdam, Netherlands

3UiT – The Arctic University of Norway, Norway

4University of Copenhagen, Denmark

5Norwegian Computing Center, Norway

6University of Massachusetts Amherst, USA

# Abstract

Contrastive learning is effective for aligning paired views or modalities, but alignment beyond two modalities remains non-trivial and comparatively underexplored. Pairwise CLIP-style losses decompose multi-modal alignment into independent twoway comparisons and therefore do not explicitly model higher-order dependencies among multiple modalities. Recent beyond-pairwise objectives approach this problem from statistical or geometric perspectives, but arbitrary-modality alignment still lacks a principled criterion for defining what each modality should preserve and compress relative to the others. We revisit arbitrary-modality alignment through the Information Bottleneck principle. In multi-modal learning, sufficiency should preserve information predictable from the remaining modalities, while minimality should compress modality-specific information not supported by them. This naturally leads to a One-vs-All view, where each modality is characterized with respect to the remaining modalities. We propose OVA-IB, an Information Bottleneck framework for arbitrary-modality alignment. OVA-IB optimizes a tractable One-vs-All contrastive lower bound for sufficiency connected to a Dual Total Correlationstyle objective, uses a parameter-free geometry-aware projection score, and derives a tractable upper-bound regularizer for minimality by bounding each representation’s dependence on its own input with representation distributions induced by the remaining modalities. Experiments on classification, regression, modality-agnostic evaluation, and cross-modal retrieval benchmarks demonstrate strong and robust performance.

# 1 Introduction

Contrastive Learning (CL) has become a central paradigm for self-supervised representation learning, as it enables models to learn transferable representations without relying on extensive human annotations. In the single-modality setting, different views of the same samples are typically generated through data augmentation, and the model is trained to bring positive views closer while separating representations from different samples. As a widely used objective for this purpose, InfoNCE [31, 7] maximizes agreement between paired views and has been interpreted from both information-theoretic and geometric perspectives. From the information-theoretic perspective, InfoNCE maximizes a lower bound on mutual information between positive views [37]; from the geometric perspective, contrastive learning simultaneously promotes alignment between positive pairs and uniformity of representations on the hypersphere [32]. These two perspectives provide the foundation for understanding contrastive learning as both an information-preserving and geometry-shaping representation learning framework.

For two-modality representation learning, paired modalities provide natural positive views for contrastive alignment. A prominent example is CLIP [25], which applies a symmetric InfoNCE objective to image–text pairs and learns a shared embedding space with strong transfer performance. However, this formulation is intrinsically pairwise. For three or more modalities, a common extension is to sum CLIP-style losses over all modality pairs. While simple and scalable, this strategy decomposes multi-modal alignment into independent two-modality comparisons and therefore fails to explicitly capture higherorder dependencies jointly supported by multiple modalities. This motivates alignment objectives that move beyond pairwise agreement. Recent beyond-pairwise methods address this limitation from complementary directions. Symile [27] models higher-order statistical dependence among an arbitrary number of modalities, while Gram [10] and TRIANGLE [9] introduce geometry-aware objectives in the shared representation space. These works provide important statistical and geometric formulations of beyond-pairwise alignment. However, alignment beyond two modalities remains comparatively underexplored, and it is still unclear how to define what each modality representation should preserve and compress when multiple remaining modalities are available.

To instantiate this principle, we propose OVA-IB, a One-vs-All Information Bottleneck framework for multi-modal alignment with an arbitrary number of modalities. The key idea is to define both sufficiency and minimality in a modality-wise manner: for each modality, the remaining modalities serve as the reference for determining what information should be preserved and what modality-specific variation should be compressed. Specifically, the sufficiency term encourages each modality representation to retain information predictable from the remaining modalities. We derive a tractable One-vs-All contrastive lower bound for this objective and show that it is connected to a Dual Total Correlation [34] (DTC)-style dependence measure. To make the alignment geometry-aware, we score each modality by its projection onto the subspace spanned by the remaining modality embeddings. For minimality, we derive a tractable upperbound regularizer that controls each representation’s dependence on its own input using representation distributions induced by the remaining modalities.

Our contributions are summarized as follows:

• We propose OVA-IB, a One-vs-All Information Bottleneck framework for multimodal alignment with an arbitrary number of modalities.   
• We derive a DTC-style sufficiency objective that aligns each modality with the remaining modalities as complementary evidence, and introduce a geometry-aware projection score that aligns each embedding with the subspace spanned by the remaining modalities.

• We derive a tractable minimality regularizer that suppresses modality-specific nuisance information by bounding each representation’s dependence on its own input using distributions induced by the remaining modalities.   
• Experiments on classification, regression, modality-agnostic evaluation, and crossmodal retrieval benchmarks demonstrate that OVA-IB achieves strong and robust performance.

# 2 Related Work

# 2.1 Contrastive Learning for Multi-Modal Representation Learning

Contrastive learning (CL) has been widely adopted for self-supervised representation learning by aligning positive views and separating negative samples [31, 7, 37, 26, 32, 35, 38, 23]. In multi-modal learning, different modalities of the same sample naturally define positive pairs, enabling contrastive objectives to learn shared information across modalities. CLIP [25] applies a symmetric InfoNCE to image-text pairs and has become a standard paradigm for two-modality alignment:

$$
\mathcal {L} _ {\text {InfoNCE}} ^ {(1 \rightarrow 2)} (\theta) = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log \frac {\exp \left(\operatorname{sim} \left(\mathbf {z} _ {i} ^ {(1)} , \mathbf {z} _ {i} ^ {(2)}\right) / \tau\right)}{\sum_ {k = 1 , k \neq i} ^ {N} \exp \left(\operatorname{sim} \left(\mathbf {z} _ {i} ^ {(1)} , \mathbf {z} _ {k} ^ {(2)}\right) / \tau\right)}, \tag {1}
$$

$$
\mathcal {L} _ {\mathrm{CLIP}} ^ {(1, 2)} (\theta) = \frac {1}{2} (\mathcal {L} _ {\mathrm{InfoNCE}} ^ {(1 \rightarrow 2)} (\theta) + \mathcal {L} _ {\mathrm{InfoNCE}} ^ {(2 \rightarrow 1)} (\theta)), \tag {2}
$$

where z(mi $\mathbf { z } _ { i } ^ { ( m ) }$ represents the representation of the m-th modality of the i-th sample, τ is a temperature parameter, and θ represents all the learnable parameters. For more than two modalities, a common extension is to apply CLIP-style objectives over all modality pairs [29, 1, 13, 6, 2, 5, 22, 33, 8]. In the simplest case, for three modalities, the pairwise CLIP loss corresponds to

$$
\mathcal {L} _ {\mathrm{CLIP}} ^ {(1, 2, 3)} (\theta) = \mathcal {L} _ {\mathrm{CLIP}} ^ {(1, 2)} (\theta) + \mathcal {L} _ {\mathrm{CLIP}} ^ {(1, 3)} (\theta) + \mathcal {L} _ {\mathrm{CLIP}} ^ {(2, 3)} (\theta). \tag {3}
$$

It reduces multi-modal alignment to a collection of independent pairwise comparisons and therefore does not explicitly capture higher-order dependencies among multiple modalities.

Compared with the extensive literature on pairwise multi-modal contrastive learning, alignment objectives for more than two modalities remain comparatively underexplored, with only a small number of recent works explicitly addressing this setting. Symile [27] extends contrastive learning by modeling higher-order statistical dependence among arbitrary modalities, while Gram [10] and TRIANGLE [9] introduce geometryaware objectives based on the volume or area formed by the same-sample modality embeddings. These works provide important initial steps toward beyond-pairwise alignment from statistical and geometric perspectives. Complementary to them, we revisit arbitrary-modality alignment from the Information Bottleneck [30] perspective and ask how sufficiency and minimality should be defined for each modality when multiple remaining modalities are available.

# 2.2 Information Bottleneck for Multi-Modal Representation Learning

Information Bottleneck (IB) [30] provides an information-theoretic framework for learning representations that are sufficient for a target variable while remaining minimal with respect to the input. Given an input x, a target y, and a representation t, the standard IB objective is formulated as

$$
\max _ {\mathbf {t}} I (\mathbf {y}; \mathbf {t}) - \beta I (\mathbf {x}; \mathbf {t}), \tag {4}
$$

where $I ( \cdot ; \cdot )$ denotes mutual information and $\beta > 0$ controls the strength of compression. The first term encourages t to preserve information relevant to y, while the second term discourages t from retaining unnecessary information from x. This sufficiency– minimality trade-off provides a natural principle for multi-modal representation learning, where the goal is to preserve shared semantic information while suppressing modalityspecific nuisance factors [36, 24, 12].

The general idea of IB has recently been applied for two-modality alignment [3]. Given two modalities $( \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { ( 2 ) } )$ with their representations $( \mathbf { z } ^ { ( 1 ) } , \mathbf { z } ^ { ( 2 ) } )$ , the sufficiency for $\mathbf { x } ^ { ( 1 ) }$ can be approximated through cross-modal dependence, while the minimality can be upperbounded through the dependence between $\mathbf { z } ^ { ( 1 ) }$ and its own input $\mathbf { x } ^ { ( 1 ) }$ . This yields the IB-style objective:

$$
\max _ {\mathbf {z} ^ {(1)}, \mathbf {z} ^ {(2)}} I (\mathbf {z} ^ {(1)}; \mathbf {z} ^ {(2)}) - \frac {\beta}{2} (I (\mathbf {z} ^ {(1)}; \mathbf {x} ^ {(1)}) + I (\mathbf {z} ^ {(2)}; \mathbf {x} ^ {(2)})). \tag {5}
$$

This formulation is elegant. However, extending Eq. (5) for an arbitrary number of modalities is a non-trivial task. This is because each modality is compared against multiple remaining modalities, and there is no unique paired counterpart as in the twomodality case. Similarly, minimality specifies which information should be discarded relative to the evidence provided by the remaining modalities.

# 3 One vs All Information Bottleneck for Multi-Modal Alignment

# 3.1 Problem Setup and Multi-Modal IB Principle

Let $\mathcal { X } ^ { ( 1 ) } , \dots , \mathcal { X } ^ { ( M ) }$ denote M distinct modalities. We consider modality-specific encoders $f ^ { ( m ) } : \mathcal { X } ^ { ( m ) }  \mathbb { R } ^ { d }$ and projection heads $g ^ { ( m ) } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ for $m = 1 , \ldots , M$ . Given an input $\mathbf { x } ^ { ( m ) } \in \mathcal { X } ^ { ( m ) }$ , the encoder produces a representation, which is mapped to the embedding $\mathbf { z } ^ { ( m ) } = g ^ { ( m ) } ( f ^ { ( m ) } ( \mathbf { x } ^ { ( m ) } ) )$ in the shared representation space. The primary objective of multi-modal representation learning is to optimize the encoders $f ^ { ( m ) }$ by aligning the projected embeddings $\{ \mathbf { z } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ at the sample level. For simplicity, we denote by $[ M ] \backslash ( m )$ the set of all indices except m, $\mathbf { z } ^ { [ M ] } \overset { \cdot } { = } ( \mathbf { z } ^ { ( 1 ) } , \ldots , \mathbf { z } ^ { ( M ) } )$ represents representations of all M modalities, and ${ \bf z } ^ { [ M ] \setminus ( m ) } = ( { \bf z } ^ { ( 1 ) } , \ldots , { \bf z } ^ { ( m - 1 ) } , { \bf z } ^ { ( m + \bar { 1 } ) } , \ldots , { \bf z } ^ { ( M ) } )$ represents representations of all M modalities without z(m). $\mathbf { z } ^ { ( m ) }$

Our goal is to derive a multi-modal alignment objective from the Information Bottleneck principle. Following [3], we assume that each modality $\mathbf { x } ^ { ( m ) }$ consists of an essence $\mathbf { y } ^ { ( m ) }$ and nuisance $\mathbf { n } ^ { ( m ) }$ , where $\mathbf { y } ^ { ( m ) }$ contains all the relevant information in $\mathbf { x } ^ { ( m ) }$ , while $\mathbf { n } ^ { ( m ) }$ represents the modality-specific noise in $\mathbf { x } ^ { ( m ) }$ . Conceptually, a representation $\mathbf { z } ^ { ( m ) }$ is sufficient if it preserves information about $\mathbf { y } ^ { ( m ) }$ , and minimal if it discards information about $\mathbf { n } ^ { ( m ) }$ . This leads to the IB objective:

$$
\max _ {\mathbf {z} ^ {(1)}, \dots , \mathbf {z} ^ {(M)}} \frac {1}{M} \sum_ {m = 1} ^ {M} \left(I (\mathbf {z} ^ {(m)}; \mathbf {y} ^ {(m)}) - \beta I (\mathbf {z} ^ {(m)}; \mathbf {n} ^ {(m)})\right). \tag {6}
$$

The superscript m only indicates that the essence and nuisance components may differ across modalities; it does not imply that Eq. (6) defines separate single-modality IB objectives.

# 3.2 Multi-Modal Sufficiency

We first derive the sufficiency objective for the $m { \mathrm { - t h } }$ modality. From the IB objective in Eq. (6), sufficiency requires maximizing $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { y } ^ { ( m ) } )$ . In the two-modality setting, the essence of one modality can be characterized through its paired modality. In the arbitrary-modality setting, however, each modality is accompanied by multiple remaining modalities, which together provide complementary context for determining what should be preserved in $\mathbf { x } ^ { ( m ) }$ . Therefore, for each modality, we characterize $\mathbf { y } ^ { ( m ) }$ with respect to the remaining modalities $\mathbf { x } ^ { [ M ] \setminus ( \dot { m } ) }$ and define the essence in Definition 3.1. Intuitively, $\mathbf { x } ^ { [ M ] \setminus ( m ) }$ provides the cross-modal context that determines which information in $\mathbf { x } ^ { ( m ) }$ should be preserved by a sufficient representation.

Definition 3.1. For each $\begin{array} { r } { m o d a l i t y \mathbf { x } ^ { ( m ) } } \end{array}$ , the essence $\mathbf { y } ^ { ( m ) }$ is characterized with respect to the remaining modalities $\mathbf { x } ^ { [ M ] \setminus ( m ) }$ through the following Markov chains:

$$
\mathbf {y} ^ {(m)} \leftrightarrow \mathbf {x} ^ {[ M ] \setminus (m)} \leftrightarrow \mathbf {x} ^ {(m)}, \qquad \mathbf {x} ^ {[ M ] \setminus (m)} \leftrightarrow \mathbf {y} ^ {(m)} \leftrightarrow \mathbf {x} ^ {(m)}.
$$

The first relation states that, conditioned on the remaining modalities, the modality does not provide additional information about its cross-modal essence. The second relation states that the essence $\mathbf { y } ^ { ( m ) }$ contains the information in $\mathbf { x } ^ { ( m ) }$ that is supported by the remaining modalities.

Theorem 3.1. Under Definition 3.1, suppose the representation $\mathbf { z } ^ { ( m ) }$ is generated from the modality $\mathbf { x } ^ { ( m ) }$ , i.e., the encoder does not directly access $\mathbf { y } ^ { ( m ) }$ or $\mathbf { x } ^ { [ M ] \setminus ( m ) }$ . Then the sufficiency term satisfies:

$$
I (\mathbf {z} ^ {(m)}; \mathbf {y} ^ {(m)}) = I (\mathbf {z} ^ {(m)}; \mathbf {x} ^ {[ M ] \setminus (m)}). \tag {7}
$$

Proof. All proofs can be found in Appendix A.

Theorem 3.1 shows that maximizing the sufficiency term $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { y } ^ { ( m ) } )$ can be reduced to maximizing the dependence between $\mathbf { z } ^ { ( m ) }$ and $\mathbf { x } ^ { [ M ] { \check { ( } } m ) }$ . Since $\mathbf { z } ^ { [ M ] \setminus ( m ) }$ is obtained from $\mathbf { x } ^ { [ M ] \setminus ( m ) }$ through the encoders and projection heads of the remaining modalities, Data Processing Inequality gives $I ( \mathbf { z } ^ { ( m ) } ; \bar { \mathbf { x } } ^ { [ M ] \bar { \backslash } ( m ) } ) \ge I \big ( \mathbf { z } ^ { ( m ) } ; \mathbf { z } ^ { [ M ] \backslash ( m ) } \big )$ . Therefore, we optimize the observable sufficiency objective

$$
\max _ {\mathbf {z} ^ {(1)}, \dots , \mathbf {z} ^ {(M)}} \frac {1}{M} \sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \backslash (m)}). \tag {8}
$$

This objective aligns one modality with all remaining modalities, revealing a One-vs-All paradigm from the sufficiency term. Computing exactly $I ( { \mathbf { z } } ^ { ( m ) } ; { \mathbf { z } } ^ { [ M ] \setminus ( \bar { m } ) } )$ requires integrating over the representation spaces, which is generally intractable. To address this issue, we introduce a projection $\bar { t } : ( \mathbb { R } ^ { d } ) ^ { ( M - 1 ) } \to \bar { \mathbb { R } } ^ { d }$ into InfoNCE such that we can maximize the lower bound of such a multivariate mutual information term. Accordingly, we define the InfoNCE objective for the m-th modality with N samples as:

$$
\mathcal {L} _ {\text {InfoNCE}} ^ {(m)} (\theta) = - \frac {1}{N} \sum_ {n = 1} ^ {N} \log \frac {\exp (s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n} ^ {[ M ] \setminus (m)})) / \tau)}{\sum_ {n ^ {\prime} = 1 , n ^ {\prime} \neq n} ^ {N} \exp (s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n ^ {\prime}} ^ {[ M ] \setminus (m)})) / \tau)}, \tag {9}
$$

where the score function s outputs the similarity between one modality $\mathbf { z } ^ { ( m ) }$ and the output of t with $M - 1$ modalities $\mathbf { z } _ { n } ^ { [ M ] \setminus ( m ) }$ , and cosine similarity is a commonly used choice for s. As Theorem 3.2 demonstrates, we can minimize the proposed InfoNCE variant from Eq. (9) to achieve the sufficiency of the m-th modality by maximizing the lower bound of Eq. (8).

Theorem 3.2. Minimizing the loss function in $E q$ . (9) corresponds to maximizing the lower bound of $I ( { \mathbf { z } } ^ { ( m ) } ; { \mathbf { z } } ^ { [ M ] \setminus ( m ) } )$ .

A straightforward choice for the projection t is to implement an MLP to map the concatenated $\mathbf { z } ^ { [ M ] \setminus ( m ) }$ from $\mathbb { R } ^ { ( M - 1 ) d } \mathrm { t o } \mathbb { R } ^ { d }$ , and then compute the cosine similarity. First, the MLP introduces many learnable parameters. Even if the parameter update is not counted, its computational complexity is $\mathcal { O } ( d ^ { 2 } M )$ . The MLP also neglects the geometric relation among modalities in the shared representation space and performs dimensionality

reduction for cosine similarity.

If the remaining modalities jointly provide sufficient relevant information for the same sample, the cross-modal information of $\mathbf { z } ^ { ( m ) }$ should be well supported by the subspace, ${ \boldsymbol { S } } =$ span $\left( \mathbf { z } ^ { [ M ] \setminus ( m ) } \right)$ , spanned by $\mathbf { \dot { z } } ^ { [ M ] \setminus ( m ) }$ . Therefore, instead of forcing all modality embeddings to become identical, we align each modality embedding with its projection $\bar { \mathbf { z } } ^ { ( m ) }$ onto S. As Figure 1 shows, $\bar { \mathbf { z } } ^ { ( m ) }$ , the projection of $\mathbf { z } ^ { ( m ) }$ onto S, should be close to $\mathbf { z } ^ { ( m ) }$ . For the n-th sample, we can use $\mathbf { z } _ { n } ^ { [ M ] \setminus ( m ) }$ z[M]\(m)n to form a matrix:

$$
A _ {n} = [ \mathbf {z} _ {n} ^ {(1)} \dots \mathbf {z} _ {n} ^ {(m - 1)} \mathbf {z} _ {n} ^ {(m + 1)} \dots \mathbf {z} _ {n} ^ {(M)} ], (1 0)
$$

where $A _ { n } \in \mathbb { R } ^ { d \times ( M - 1 ) }$ . In this setting, we can compute the projection $\bar { \mathbf { z } } ^ { ( m ) }$ in the closed-form expression:

![](images/ccd075f652c0bf28db006e6aa100910ef2378803f0c4cbdf86e18f6248f56c25.jpg)

<details>
<summary>text_image</summary>

z^{(1)}
z^{(2)}
...
z^{(m-1)}
z^{(m+1)}
\bar{z}^{(m)}
...
z^{(M)}
S = \text{span}(z^{[M]\setminus(m)})
</details>

Figure 1: We align $\mathbf { z } ^ { ( m ) }$ with its projection $\bar { \mathbf { z } } ^ { ( m ) }$ onto $s ,$ encouraging each modality to preserve information predictable from the remaining modalities.

$$
\bar {\mathbf {z}} ^ {(m)} = A _ {n} (A _ {n} ^ {\top} A _ {n} + \lambda I) ^ {- 1} A _ {n} ^ {\top} \mathbf {z} ^ {(m)} \in \mathbb {R} ^ {d}, \tag {11}
$$

where $\lambda$ is a small numerical constant to guarantee the inverse. This closed-form projection incurs $\mathcal { O } ( d M ^ { 2 } )$ complexity per sample. Given typical settings where $d \gg M$ , it is substantially more efficient than an MLP-based projector $( \mathcal { O } ( d ^ { 2 } M ) )$ and introduces no learnable parameters. Using this method, we obtain t to project the remaining-modality embeddings into $\mathbb { R } ^ { d }$ , then we can compute the cosine similarity between $\mathbf { z } ^ { ( m ) }$ and $\bar { \mathbf { z } } ^ { ( m ) }$ for alignment.

The overall sufficiency objective is defined as:

$$
\mathcal {L} _ {S} (\theta) = \frac {1}{M} \sum_ {m = 1} ^ {M} \mathcal {L} _ {\text {InfoNCE}} ^ {(m)} (\theta). \tag {12}
$$

By Theorem 3.3, minimizing $\mathcal { L } _ { S }$ is theoretically maximizing the dual total correlation (DTC) [14].

Theorem 3.3. Optimizing Eq. (8) approximately maximizes the dual total correlation (DTC) among $\mathbf { z } ^ { ( 1 ) } , \ldots , \mathbf { z } ^ { ( \bar { M } ) }$ , which is defined as

$$
\mathrm{DTC} (\mathbf {z} ^ {[ M ]}) = H (\mathbf {z} ^ {[ M ]}) - \sum_ {m = 1} ^ {M} H (\mathbf {z} ^ {(m)} | \mathbf {z} ^ {[ M ] \setminus (m)}). \tag {13}
$$

The theoretical basis is the following sandwich bound, which shows that DTC is bounded from both sides by scaled forms of our One-vs-All sufficiency term in Eq. (8):

$$
\frac {1}{M} \sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}) \leq \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) \leq \frac {M - 1}{M} \sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}). \tag {14}
$$

Remark 3.1. Our sufficiency objective is closely related to recent beyond-pairwise alignment methods such as Symile, but differs in both its information-theoretic form and its interpretation. Symile maximizes total correlation [14] (TC), defined as:

$$
\mathrm{TC} (\mathbf {z} ^ {[ M ]}) = \sum_ {m = 1} ^ {M} H (\mathbf {z} ^ {(m)}) - H (\mathbf {z} ^ {[ M ]}). \tag {15}
$$

Our objective introduces two immediate advantages over Symile. First, for the sufficiency term, TC measures dependence among all modality representations as a whole, while DTC explicitly specifies what information each modality should preserve with respect to the remaining modalities, which directly matches the sufficiency definition. Figure 2 intuitively shows the distinction between DTC and TC. Second, beyond sufficiency, our additional minimality objective provided in the following section compresses modality-specific nuisance information.

# 3.3 Multi-Modal Minimality

For a representation $\mathbf { z } ^ { ( m ) }$ to be minimal, it must satisfy $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { n } ^ { ( m ) } ) = 0$ , which is equivalent to minimizing $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { n } ^ { ( m ) } )$ . Direct computation is intractable due to the unobserved $\mathbf { n } ^ { ( m ) }$ . To obtain a tractable surrogate, we leverage the structural dependency $\mathbf { n } ^ { ( m ) } \gets$ $\mathbf { x } ^ { ( m ) } \to \mathbf { z } ^ { ( m ) }$ . By Data Processing Inequality, this implies $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { n } ^ { ( m ) } ) \leq I ( \mathbf { z } ^ { ( m ) } ; \mathbf { x } ^ { ( m ) } )$ , providing a valid upper bound. Consequently, we minimize $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { x } ^ { ( m ) } )$ as a tractable upper-bound surrogate for the minimality. Integrating over the representation and input spaces to compute $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { x } ^ { ( m ) } )$ is generally intractable. However, all projection heads map modality-specific representations into the same shared representation space $\mathcal { Z } \subseteq \mathbb { R } ^ { d }$ . Therefore, for $\mathbf { x } ^ { ( m ) } , p _ { \theta ^ { ( m ) } } ( \mathbf { z } | \mathbf { x } ^ { ( m ) } )$ denotes the conditional density of the projected embedding of $\mathbf { x } ^ { ( m ) }$ , evaluated at the same shared-space coordinate $\mathbf { z } \in { \mathcal { Z } }$ .

![](images/a73500fe3fc51b1528830778f8bd09ccc367c82f523d5ac20d7b6efbc550689e.jpg)

<details>
<summary>text_image</summary>

z^{(1)}
z^{(2)}
z^{(3)}
</details>

![](images/35f503eaaac5694ccc64d24c74c242e0484182f0d859388de5bfd6b8284a0e2a.jpg)

<details>
<summary>text_image</summary>

z^{(1)}
z^{(2)}
z^{(3)}
</details>

![](images/152ca15a556c6c28859895ad2342fa2fdda1694f98f76a2abfc898de8e729a94.jpg)

<details>
<summary>text_image</summary>

z^{(1)}
z^{(2)}
z^{(3)}
</details>

Figure 2: Intuitive illustration of multivariate dependence among three modality representations ${ \bf z } ^ { ( 1 ) } , { \bf z } ^ { ( 2 ) } , { \bf z } ^ { ( 3 ) }$ . Left: common agreement among all modalities. Middle: TCstyle dependence compares the joint distribution with the product of marginals. Right: DTC-style dependence measures how each modality is predictable from the remaining modalities. The diagram is intended to provide intuition rather than a formal information decomposition.

Theorem 3.4. For each modality $\mathbf { x } ^ { ( m ) }$ , the minimality term admits the following upper bound:

$$
I \left(\mathbf {z} ^ {(m)}; \mathbf {x} ^ {(m)}\right) \leq \mathbb {E} _ {p \left(\mathbf {x} ^ {[ M ]}\right)} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} \left(\mathbf {z} \mid \mathbf {x} ^ {(m)}\right) \| p _ {\theta^ {[ M ] \backslash (m)}} \left(\mathbf {z} \mid \mathbf {x} ^ {[ M ] \backslash (m)}\right)\right) \right], \tag {16}
$$

where $\begin{array} { r } { p _ { \theta ^ { [ M ] \setminus ( m ) } } ( \mathbf { z } | \mathbf { x } ^ { [ M ] \setminus ( m ) } ) = \frac { \prod _ { i \neq m } p _ { \theta ^ { ( i ) } } ( \mathbf { z } | \mathbf { x } ^ { ( i ) } ) } { \int \prod _ { i \neq m } p _ { \theta } ^ { ( i ) } ( \mathbf { u } | \mathbf { x } ^ { ( i ) } ) d \mathbf { u } } } \end{array}$ R Qi̸=m p(i)θ (u|x(i)) du Qi̸=m pθ(i) (z|x(i)) is the normalized product of distributions induced by the remaining modalities, and $D _ { K L } ( \cdot | | \cdot )$ is Kullback–Leibler (KL) divergence [18].

This result provides a tractable surrogate for minimizing $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { x } ^ { ( m ) } )$ : by minimizing the upper bound in Eq. (16), the representation distribution of $\mathbf { x } ^ { ( m ) }$ is encouraged to match the normalized product of the representation distributions induced by the remaining modalities. Thus, the minimality objective also follows a One-vs-All paradigm. The upper bound in Theorem 3.4 does not have a closed form in general. Following recent theoretical insights that InfoNCE objectives induce Gaussian-like representations in high dimensions [4], we assume isotropic Gaussian distributions in the shared embedding space to derive a closed-form upper bound.

Proposition 3.1. Assume that each modality-induced representation distribution in the shared representation space is an isotropic Gaussian, $p _ { \theta ^ { ( m ) } } ( \mathbf { z } | \mathbf { x } ^ { ( m ) } ) = \mathcal { N } \left( \mathbf { z } ; \mu _ { \theta ^ { ( m ) } } ( \mathbf { x } ^ { ( m ) } ) , \sigma ^ { 2 } I \right)$ . For a modality $\mathbf { x } ^ { ( m ) }$ , the $\mu _ { \boldsymbol { \theta } } ^ { ( m ) } ( \mathbf { x } ^ { ( m ) } )$ -dependent part of the upper-bound term in Theorem 3.4 satisfies

$$
\mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \big [ D _ {\mathrm{KL}} \big (p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \left\| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} \mid \mathbf {x} ^ {[ M ] \setminus (m)})\right) \big ] \propto \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \big [ \left\| \mu_ {\theta^ {(m)}} (\mathbf {x} ^ {(m)}) - \bar {\mu} \right\| _ {2} ^ {2} \big ], (1 7)
$$

$\begin{array} { r } { \bar { \mu } = \frac { 1 } { M - 1 } \sum _ { i = 1 , i \neq m } ^ { M } \mu _ { \theta ^ { ( i ) } } ( \mathbf { x } ^ { ( i ) } ) } \end{array}$

Proposition 3.1 provides the tractable minimality objective for $\mathbf { x } ^ { ( m ) }$ :

$$
\mathcal {L} _ {M} ^ {(m)} (\theta) = \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \left[ | | \mu_ {\theta^ {(m)}} (\mathbf {x} ^ {(m)}) - \bar {\mu} | | _ {2} ^ {2} \right], \tag {18}
$$

where the expectation is empirically approximated by the sample average over the sample size N . For the overall minimality objective across all modalities, we average the modality-wise losses:

$$
\mathcal {L} _ {M} (\theta) = \frac {1}{M} \sum_ {m = 1} ^ {M} \mathcal {L} _ {M} ^ {(m)}, \tag {19}
$$

where minimizing $\mathcal { L } _ { M }$ provides a tractable surrogate for controlling $\begin{array} { r } { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } I ( \mathbf { z } ^ { ( m ) } ; \mathbf { x } ^ { ( m ) } ) } \end{array}$

# 3.4 Unified Multi-Modal Information Bottleneck

Combining the sufficiency and minimality principles gives the following multi-modal IB:

$$
\max _ {\mathbf {z} ^ {[ M ]}} \frac {1}{M} \sum_ {m = 1} ^ {M} \left(I (\mathbf {z} ^ {(m)}; \mathbf {x} ^ {[ M ] \setminus (m)}) - \beta I (\mathbf {z} ^ {(m)}; \mathbf {x} ^ {(m)})\right), \tag {20}
$$

where the first term encourages the representation of each modality to preserve information related to the remaining modalities, while the second term controls excessive dependence on modality-specific nuisance information. Using the tractable objectives above, we obtain the final training objective:

$$
\min _ {\theta} \mathcal {L} (\theta) = \mathcal {L} _ {S} (\theta) + \beta \mathcal {L} _ {M} (\theta). \tag {21}
$$

# 4 Experiments

We comprehensively evaluate OVA-IB across all-modality downstream tasks, modalityagnostic robustness using different modality combinations on downstream tasks, crossmodal retrieval, and component-wise ablations of the minimality regularizer and projection strategy.

# 4.1 Experimental Setup and Benchmarks

We evaluate OVA-IB on four multi-modal benchmarks: Vision & Touch, MuJoCo Push, CMU-MOSEI, and WESAD. These datasets cover robotic perception, physical interaction, audio-visual-language understanding, and wearable physiological sensing. We further construct four- and five-modality variants from Vision & Touch to evaluate scalability to higher-modality settings. For Vision & Touch, MuJoCo Push, and CMU-MOSEI, we follow the standardized MultiBench protocol with fixed input representations and evaluation splits across methods. Since WESAD is not part of MultiBench, we apply identical synchronized-window preprocessing and subject-wise splits to all methods for fair comparison. The dataset details are provided in Appendix B.1.

For classification and regression, we first pre-train modality encoders using each alignment objective. We then concatenate the learned modality representations and train a 4-layer MLP predictor with cross-entropy for classification and mean squared error for regression. We set $\beta \ : = \ : 1$ for all benchmarks except WESAD $( \beta ~ = ~ 0 . 5 )$ . The downstream tasks include trajectory-pair classification and orientation regression on Vision–Force–Depth (VFD), contact classification on Vision–Force–Proprioception (VFP), higher-modality trajectory classification on VFPD and VFPDO, object-position regression on MuJoCo Push, binary sentiment classification on CMU-MOSEI, and three-class affective-state classification on WESAD. For classification and regression, we evaluate OVA-IB across multiple datasets and task types, providing broad evidence of stability on supervised downstream tasks. For cross-modal retrieval, we select a task with clear practical relevance rather than arbitrary modality matching. WESAD provides such a setting, where wrist-side BVP is physiologically related to chest-side respiration and ECG through cardiopulmonary and pulse dynamics. Since suitable retrieval settings with clear practical relevance are limited, we repeat the Resp + ECG → BVP retrieval experiment three times with different random seeds and report mean ± standard deviation of mAP (mean average precision). Implementation details are provided in the Appendix B.2.

# 4.2 All-Modality Downstream Evaluation

We first evaluate all methods under the all-modality setting, where all modalities are used during downstream evaluation. This setting measures whether the learned representations provide useful task-relevant information from all multi-modal observations. Table 1 reports the classification and regression results. OVA-IB achieves the best performance on most benchmarks, with particularly large gains as the number of modalities increases. On VFPDO, OVA-IB outperforms Symile and Pairwise CLIP by 12.95% and 2.96%, respectively. It also obtains the lowest MSE on both regression tasks. On MOSEI, where standardized pre-extracted features leave limited representation-level headroom, OVA-IB remains competitive but is not the top method. Overall, all-modality results show that OVA-IB learns compact and informative representations across both classification and regression.

Table 1: All-modality downstream performance on classification and regression tasks. Accuracy (%) is reported for classification, and MSE is reported for regression. 

<table><tr><td rowspan="2">Method</td><td colspan="6">Classification (Acc %)</td><td colspan="2">Regression (MSE)</td></tr><tr><td>VFP</td><td>VFD</td><td>VFPD</td><td>VFPDO</td><td>MOSEI</td><td>WESAD</td><td>VFD</td><td>MuJoCo</td></tr><tr><td>Ours</td><td>97.12</td><td>95.08</td><td>94.73</td><td>95.91</td><td>76.02</td><td>76.18</td><td>0.0003</td><td>0.0340</td></tr><tr><td>Symile</td><td>96.44</td><td>92.57</td><td>84.97</td><td>82.96</td><td>77.00</td><td>73.42</td><td>0.0015</td><td>0.1228</td></tr><tr><td>Gram</td><td>96.85</td><td>89.91</td><td>90.81</td><td>90.59</td><td>72.77</td><td>66.22</td><td>0.0018</td><td>0.0425</td></tr><tr><td>TRIANGLE</td><td>95.91</td><td>93.27</td><td>-</td><td>-</td><td>76.13</td><td>56.87</td><td>0.0019</td><td>0.0656</td></tr><tr><td>Pairwise CLIP</td><td>96.55</td><td>93.77</td><td>93.25</td><td>92.95</td><td>75.05</td><td>70.83</td><td>0.0013</td><td>0.0810</td></tr><tr><td>Pairwise CLIP + IB</td><td>97.10</td><td>94.70</td><td>94.45</td><td>94.68</td><td>76.25</td><td>67.91</td><td>0.0009</td><td>0.3385</td></tr></table>

# 4.3 Modality-Agnostic Evaluation

We next evaluate modality-agnostic robustness, where models are pre-trained with all modalities but tested under different modality combinations. This setting measures whether the learned representation space remains effective when the availability of modality changes at test time. Table 2 reports a representative VFD trajectory-classification result, while Appendix B.3 provides the full modality-combination results on all classification and regression tasks. OVA-IB achieves the highest or near-highest performance across most modality combinations. Pairwise CLIP + IB occasionally leads on singleor dual-modality inputs, which is expected because pairwise objectives directly optimize specific modality pairs. In contrast, OVA-IB shows stronger gains when multiple modalities are available, indicating that aligning each modality with the remaining modalities yields representations that are less tied to a fixed pairwise configuration.

The same trend holds across the full results in Appendix B.3. On regression tasks such as MuJoCo Push and VFD orientation prediction, OVA-IB maintains the lowest MSE across modality combinations. On classification tasks, its advantage becomes more pronounced as the number of modalities increases. MOSEI is comparatively saturated due to its standardized pre-extracted features, but OVA-IB remains competitive on multimodal combinations. Overall, these results show that OVA-IB also improves robustness under varying modality combinations.

Table 2: Modality-combination evaluation on VFD trajectory classification. 

<table><tr><td>Method</td><td>V</td><td>F</td><td>D</td><td>V+F</td><td>V+D</td><td>F+D</td><td>V+F+D</td></tr><tr><td>Ours</td><td>94.06</td><td>93.31</td><td>91.13</td><td>95.55</td><td>94.20</td><td>94.14</td><td>95.08</td></tr><tr><td>Symile</td><td>87.65</td><td>92.68</td><td>87.23</td><td>89.76</td><td>87.44</td><td>88.62</td><td>92.57</td></tr><tr><td>Gram</td><td>92.16</td><td>92.25</td><td>86.27</td><td>93.12</td><td>92.35</td><td>88.16</td><td>89.91</td></tr><tr><td>TRIANGLE</td><td>92.40</td><td>92.21</td><td>86.95</td><td>92.66</td><td>92.75</td><td>88.75</td><td>93.27</td></tr><tr><td>Pairwise CLIP</td><td>93.05</td><td>92.27</td><td>90.91</td><td>93.50</td><td>93.44</td><td>91.46</td><td>93.77</td></tr><tr><td>Pairwise CLIP + IB</td><td>94.36</td><td>92.37</td><td>91.36</td><td>95.08</td><td>94.29</td><td>93.09</td><td>94.70</td></tr></table>

# 4.4 Cross-Modal Retrieval

We further evaluate OVA-IB on zero-shot cross-modal retrieval using WESAD. Unlike downstream classification and regression, this task directly tests whether the learned representation space aligns physiological signals across devices. Specifically, chest respiration and ECG are used as the query, and the goal is to retrieve the synchronized wrist BVP signal from a candidate pool. This setting is physiologically grounded: ECG reflects cardiac electrical activity, respiration provides cardiopulmonary context, and BVP captures peripheral pulse dynamics at the wrist.

Table 3: Cross-modal retrieval performance on WESAD. The task is Resp+ECG → BVP retrieval with mAP as the metric. 

<table><tr><td>Method</td><td>Resp+ECG → BVP</td></tr><tr><td>Ours</td><td>0.3004 ± 0.0067</td></tr><tr><td>Symile</td><td>0.2977 ± 0.0125</td></tr><tr><td>Gram</td><td>0.2974 ± 0.0019</td></tr><tr><td>TRIANGLE</td><td>0.2846 ± 0.0134</td></tr><tr><td>Pairwise CLIP</td><td>0.2918 ± 0.0057</td></tr><tr><td>Pairwise CLIP + IB</td><td>0.2938 ± 0.0187</td></tr></table>

Each method is evaluated using the scorer induced by its own pre-training objective: MIP for Symile, Gramian volume for Gram, triangle area for TRIANGLE, averaged cosine similarity for Pairwise CLIP, and the projection score for OVA-IB. This avoids mismatched scoring functions and ensures that each method is evaluated under its intended similarity metric. OVA-IB obtains the highest mAP on Resp + ECG → BVP retrieval. Although the margin is modest, the task is cross-device and zero-shot, making the improvement complementary to the supervised downstream evaluation. This indicates that the learned representation space is useful for direct cross-modal matching.

# 4.5 Ablation Study on $\beta$ and Projection Strategy

Finally, we ablate the two main components of OVA-IB: the One-vs-All minimality regularizer and the projection strategy. Table 4 reports the effect of removing the minimality term by setting $\beta = 0$ , as well as replacing the closed-form geometric projection with a learnable MLP projector. Removing the minimality term leads to consistent degradation across 7 of 8 benchmarks. The drop becomes larger as the number of modalities increases, with the largest decline on the five-modality VFPDO task. In regression, removing the minimality term substantially increases prediction error on both VFD and MuJoCo Push. These results confirm that the One-vs-All minimality term helps suppress modality-specific nuisance information and improves the transferability of the learned representations.

The MLP projector remains competitive on several classification tasks, suggesting that the One-vs-All objective itself provides a strong alignment signal. However, it becomes less stable on regression benchmarks, especially MuJoCo Push, where the error increases substantially. This indicates that our geometry-aware projection provides a more efficient alignment mechanism than the MLP. CMU-MOSEI shows minimal sensitivity to either component, because pre-extracted standardized features consistently limit the available representational headroom. Conversely, on the less standardized WESAD dataset, the minimality term has a more pronounced effect on the downstream performance. Overall, the ablation results validate that the minimality term and the geometry-aware projection together provide a scalable and efficient alignment objective.

Table 4: Ablation study on $\beta$ and projection strategy. Accuracy (%) and MSE are reported. $\Delta _ { \beta } = \beta ( \pmb { \Vdash } ) - \beta ( \pmb { \mathsf { X } } ) , \Delta _ { \mathrm { p r o j } } = \mathrm { O u r s } - \mathrm { M L P }$ . For MSE, negative $\Delta$ indicates improvement. 

<table><tr><td rowspan="2">Task</td><td rowspan="2">Dataset</td><td colspan="2"> $\beta$ </td><td colspan="2">Projector</td><td colspan="2">Improvement</td></tr><tr><td>✘</td><td>✓</td><td>MLP</td><td>Ours</td><td> $\Delta_{\beta}$ </td><td> $\Delta_{proj}$ </td></tr><tr><td rowspan="6">Classification (Acc)</td><td>VFP</td><td>96.22</td><td>97.12</td><td>96.93</td><td>97.12</td><td>+0.90</td><td>+0.19</td></tr><tr><td>VFD</td><td>91.30</td><td>95.08</td><td>95.08</td><td>95.08</td><td>+3.78</td><td>+0.00</td></tr><tr><td>VFPD</td><td>91.98</td><td>94.73</td><td>94.50</td><td>94.73</td><td>+2.75</td><td>+0.23</td></tr><tr><td>VFPDO</td><td>89.49</td><td>95.91</td><td>94.24</td><td>95.91</td><td>+6.42</td><td>+1.67</td></tr><tr><td>MOSEI</td><td>76.10</td><td>76.02</td><td>76.36</td><td>76.02</td><td>-0.08</td><td>-0.32</td></tr><tr><td>WESAD</td><td>65.09</td><td>76.18</td><td>70.66</td><td>76.18</td><td>+11.09</td><td>+5.52</td></tr><tr><td rowspan="2">Regression (MSE)</td><td>VFD</td><td>0.0022</td><td>0.0003</td><td>0.0013</td><td>0.0003</td><td>-0.0019</td><td>-0.0010</td></tr><tr><td>MuJoCo</td><td>0.0817</td><td>0.0340</td><td>0.3750</td><td>0.0340</td><td>-0.0477</td><td>-0.3410</td></tr></table>

# 5 Limitations

Similar to competing approaches such as Symile and Gram, our evaluation pre-trains modality-specific encoders from scratch, which constrains experiments to moderatelysized datasets. Extending OVA-IB to leverage pre-trained foundation models is a promising direction for future work. Additionally, OVA-IB assumes complete modality availability during pre-training, as each modality is aligned with the complementary context provided by the remaining modalities. Although we demonstrate modality-agnostic robustness at test time across arbitrary modality combinations, adapting the framework to handle missing modalities during pre-training remains an important future direction.

# 6 Conclusion

We introduced OVA-IB, a One-vs-All Information Bottleneck framework for arbitrarymodality alignment. By defining sufficiency and minimality with respect to the remaining modalities, OVA-IB derives a DTC-style contrastive sufficiency objective and a tractable minimality regularizer for suppressing modality-specific nuisance information. Together with a geometry-aware projection, the resulting objective provides an efficient objective for multi-modal alignment. Experiments on classification, regression, modality-agnostic evaluation, and cross-modal retrieval show that OVA-IB learns compact and robust representations, especially as the number of modalities increases.

# References

[1] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: transformers for multimodal self-supervised learning from raw video, audio and text. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc.   
[2] Jean-Baptiste Alayrac, Adri\`a Recasens, Rosalia Schneider, Relja Arandjelovi´c, Jason Ramapuram, Jeffrey De Fauw, Lucas Smaira, Sander Dieleman, and Andrew Zisserman. Self-supervised multimodal versatile networks. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc.   
[3] Antonio Almud´evar, Jos´e Miguel Hern´andez-Lobato, Sameer Khurana, Ricard Marxer, and Alfonso Ortega. Aligning multimodal representations through an information bottleneck. In Forty-second International Conference on Machine Learning, 2025.   
[4] Roy Betser, Eyal Gofer, Meir Yossef Levi, and Guy Gilboa. InfoNCE induces gaussian distribution. In The Fourteenth International Conference on Learning Representations, 2026.   
[5] Brian Chen, Andrew Rouditchenko, Kevin Duarte, Hilde Kuehne, Samuel Thomas, Angie Boggust, Rameswar Panda, Brian Kingsbury, Rogerio Feris, David Harwath, James Glass, Michael Picheny, and Shih-Fu Chang. Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos . In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 7992–8001, Los Alamitos, CA, USA, October 2021. IEEE Computer Society.   
[6] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. VAST: A vision-audio-subtitle-text omni-modality foundation model and dataset. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Hal Daum´e III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1597–1607. PMLR, 13–18 Jul 2020.   
[8] Sangyeon Cho, Jangyeong Jeon, Mingi Kim, and Junyeong Kim. Synergy-clip: Extending clip with multi-modal integration for robust representation learning. IEEE Access, 13:65630–65642, 2025.   
[9] Giordano Cicchetti, Eleonora Grassucci, and Danilo Comminiello. A TRIANGLE enables multimodal alignment beyond cosine similarity. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.   
[10] Giordano Cicchetti, Eleonora Grassucci, Luigi Sigillo, and Danilo Comminiello. Gramian multimodal representation learning and alignment. In The Thirteenth International Conference on Learning Representations, 2025.

[11] Thomas M. Cover and Joy A. Thomas. Elements of information theory. 2005.   
[12] Liang Dong. A unified information bottleneck framework for multimodal biomedical machine learning. Entropy, 28(4), 2026.   
[13] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind one embedding space to bind them all. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180–15190, 2023.   
[14] Te Sun Han. Linear dependence structure of the entropy space. Information and Control, 29(4):337–368, 1975.   
[15] Te Sun Han. Nonnegative entropy measures of multivariate symmetric correlations. Information and Control, 36(2):133–156, 1978.   
[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.   
[17] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997.   
[18] Solomon Kullback and R. A. Leibler. On information and sufficiency. Annals of Mathematical Statistics, 22:79–86, 1951.   
[19] Michelle A. Lee, Brent Yi, Roberto Mart´ın-Mart´ın, Silvio Savarese, and Jeannette Bohg. Multimodal sensor fusion with differentiable filters. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), page 10444–10451. IEEE Press, 2020.   
[20] Michelle A Lee, Yuke Zhu, Krishnan Srinivasan, Parth Shah, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Jeannette Bohg. Making sense of vision and touch: Selfsupervised learning of multimodal representations for contact-rich tasks. In 2019 International conference on robotics and automation (ICRA), pages 8943–8950. IEEE, 2019.   
[21] Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A Lee, Yuke Zhu, et al. Multibench: Multiscale benchmarks for multimodal representation learning. Advances in neural information processing systems, 2021(DB1):1, 2021.   
[22] Jing Liu, Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, and Jinhui Tang. Valor: Vision-audio-language omni-perception pretraining model and dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(2):708–724, 2025.   
[23] Achleshwar Luthra, Tianbao Yang, and Tomer Galanti. Self-supervised contrastive learning is approximately supervised contrastive learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.   
[24] Sijie Mai, Ying Zeng, and Haifeng Hu. Multimodal information bottleneck: Learning minimal sufficient unimodal and multimodal representations. Trans. Multi., 25:4121–4134, January 2023.

[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 18–24 Jul 2021.   
[26] Evgenia Rusak, Patrik Reizinger, Attila Juhos, Oliver Bringmann, Roland S Zimmermann, and Wieland Brendel. Infonce: Identifying the gap between theory and practice. In International Conference on Artificial Intelligence and Statistics, pages 4159–4167. PMLR, 2025.   
[27] Adriel Saporta, Aahlad Manas Puli, Mark Goldstein, and Rajesh Ranganath. Contrasting with symile: Simple model-agnostic representation learning for unlimited modalities. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.   
[28] Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. Introducing wesad, a multimodal dataset for wearable stress and affect detection. In Proceedings of the 20th ACM International Conference on Multimodal Interaction, ICMI ’18, page 400–408, New York, NY, USA, 2018. Association for Computing Machinery.   
[29] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding, 2020.   
[30] Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method, 2000.   
[31] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019.   
[32] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Hal Daum´e III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 9929–9939. PMLR, 13–18 Jul 2020.   
[33] Zehan Wang, Ziang Zhang, Minjie Hong, Hang Zhang, Luping Liu, Rongjie Huang, Xize Cheng, Shengpeng Ji, Tao Jin, Hengshuang Zhao, and Zhou Zhao. Omnibind: Large-scale omni multimodal representation via binding spaces. In The Thirteenth International Conference on Learning Representations, 2025.   
[34] Satosi Watanabe. Information theoretical analysis of multivariate correlation. IBM Journal of Research and Development, 4(1):66–82, 1960.   
[35] Mingqi Wu, Qiang Sun, and Archer Y. Yang. PCA++: How uniformity induces robustness to background noise in contrastive learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.   
[36] Qilong Wu, Yiyang Shao, Jun Wang, and Xiaobo Sun. Learning optimal multimodal information bottleneck representations. In Forty-second International Conference on Machine Learning, 2025.

[37] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph contrastive learning with augmentations. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc.   
[38] Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, and Wieland Brendel. Contrastive learning inverts the data generating process. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12979–12990. PMLR, 18–24 Jul 2021.

# A Proofs of Section 3

# A.1 Proof of Theorem 3.1

Proof. Under Definition 3.1, we have the Markov chains:

$$
\mathbf {y} ^ {(m)} \leftrightarrow \mathbf {x} ^ {[ M ] \backslash (m)} \leftrightarrow \mathbf {x} ^ {(m)}, \tag {22}
$$

$$
\mathbf {x} ^ {[ M ] \backslash (m)} \leftrightarrow \mathbf {y} ^ {(m)} \leftrightarrow \mathbf {x} ^ {(m)}. \tag {23}
$$

Since $\mathbf { z } ^ { ( m ) }$ is encoded from $\mathbf { x } ^ { ( m ) }$ via $f ^ { ( m ) }$ and $g ^ { ( m ) }$ and it does not directly access $\mathbf { y } ^ { ( m ) }$ or $\mathbf { x } ^ { [ M ] \setminus ( m ) }$ , combining this encoder relation with the Markov chains in Definition 3.1 yields:

$$
\mathbf {y} ^ {(m)} \rightarrow \mathbf {x} ^ {[ M ] \backslash (m)} \rightarrow \mathbf {x} ^ {(m)} \rightarrow \mathbf {z} ^ {(m)}. \tag {24}
$$

By the Data Processing Inequality (DPI) [11]:

$$
I (\mathbf {z} ^ {(m)}; \mathbf {y} ^ {(m)}) \leq I (\mathbf {z} ^ {(m)}; \mathbf {x} ^ {[ M ] \setminus (m)}), \tag {25}
$$

where $I ( \cdot ; \cdot )$ denotes the mutual information [11]. Since $\mathbf { x } ^ { [ M ] \setminus ( m ) }  \mathbf { y } ^ { ( m ) }  \mathbf { x } ^ { ( m ) }$ , and $\mathbf { z } ^ { ( m ) }$ is stochastically determined by $\mathbf { x } ^ { ( m ) }$ , we can obtain another Markov chain:

$$
\mathbf {x} ^ {[ M ] \backslash (m)} \rightarrow \mathbf {y} ^ {(m)} \rightarrow \mathbf {x} ^ {(m)} \rightarrow \mathbf {z} ^ {(m)}. \tag {26}
$$

Applying DPI to this chain yields:

$$
I (\mathbf {z} ^ {(m)}; \mathbf {y} ^ {(m)}) \geq I (\mathbf {z} ^ {(m)}; \mathbf {x} ^ {[ M ] \setminus (m)}). \tag {27}
$$

Hence, we can conclude that

$$
I (\mathbf {z} ^ {(m)}; \mathbf {y} ^ {(m)}) = I (\mathbf {z} ^ {(m)}; \mathbf {x} ^ {[ M ] \setminus (m)}). \tag {28}
$$

![](images/984f65ae4e867cdab048fdb0d9c2e9414a2dc43e473f57acfafe4141d3543cb3.jpg)

# A.2 Donsker–Varadhan representation for multivariate mutual information

Theorem A.1 (Donsker–Varadhan representation for multivariate mutual information). Let $P ( \mathbf { x } , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } )$ be the joint distribution of the random variables $\left( \mathbf { x } , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } \right)$ . Define the product of the marginals $Q = P ( \mathbf x ) P ( \mathbf z _ { 1 } , \dots , \mathbf z _ { k } )$ , where $P ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } )$ is the joint marginal of the $\mathbf { z } _ { i } \mathbf { \zeta } _ { s }$ . The multivariate mutual information is exactly the KL divergence $( I ( \mathbf { x } ; \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } ) : = D _ { \mathrm { K L } } ( P ( \mathbf { x } , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } ) | | Q ) )$ . Applying the classical Donsker–Varadhan representation directly to this KL divergence yields the following dual variational form:

$$
I (\mathbf {x}; \mathbf {z} _ {1}, \dots , \mathbf {z} _ {k}) = \sup _ {T} \mathbb {E} _ {P _ {X, Z _ {1}, \dots , Z _ {k}}} [ T ] - \log \mathbb {E} _ {Q} [ e ^ {T} ], \tag {29}
$$

where the supremum is taken over all (measurable) functions $T : \mathcal { X } \times \mathcal { Z } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { Z } _ { k } \to$ R such that the two expectations exist and are finite.

Proof. Define $P : = P ( \mathbf { x } , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } )$ and $Q : = P ( \mathbf { x } ) P ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } )$ . Then, by the definition of mutual information between x and the tuple $\left( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k } \right)$ ,

$$
I (\mathbf {x}; \mathbf {z} _ {1}, \dots , \mathbf {z} _ {k}) = D _ {K L} (P | | Q). \tag {30}
$$

So it is enough to prove that

$$
D _ {K L} (P | | Q) = \sup _ {T} \{\mathbb {E} _ {P} [ T ] - \log \mathbb {E} _ {Q} [ e ^ {T} ] \}, \tag {31}
$$

where $T : \mathcal { X } \times \mathcal { Z } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { Z } _ { k } \to \mathbb { R }$ is any measurable function, such that both E $_ P [ T ]$ and $\mathbb { E } _ { Q } [ e ^ { T } ]$ are finite. Define $Z : = \mathbb { E } _ { Q } [ e ^ { T } ]$ . Since $e ^ { T } > 0$ , we have $Z > 0$ . Now, we define a Gibbs distribution G:

$$
d G = \frac {1}{Z} e ^ {T} d Q, \tag {32}
$$

where $\begin{array} { r } { \int d G = \int \frac { 1 } { Z } e ^ { T } d Q = \frac { 1 } { Z } \mathbb { E } _ { Q } [ e ^ { T } ] = 1 } \end{array}$ . Equivalently,

$$
{\frac {d G}{d Q}} = {\frac {e ^ {T}}{Z}}. \tag {33}
$$

Taking logarithms on both sides:

$$
\log \frac {d G}{d Q} = \log \frac {e ^ {T}}{Z} = T - \log Z. \tag {34}
$$

Now take expectation under P on both sides:

$$
\mathbb {E} _ {P} \left[ \log \frac {d G}{d Q} \right] = \mathbb {E} _ {P} [ T - \log Z ]. \tag {35}
$$

Because log Z is a constant with respect to the random variable,

$$
\mathbb {E} _ {P} [ T - \log Z ] = \mathbb {E} _ {P} [ T ] - \log Z. \tag {36}
$$

Therefore,

$$
\mathbb {E} _ {P} \left[ \log \frac {d G}{d Q} \right] = \mathbb {E} _ {P} [ T ] - \log Z = \mathbb {E} _ {P} [ T ] - \log \mathbb {E} _ {Q} [ e ^ {T} ]. \tag {37}
$$

Define the gap $\Delta : = D _ { K L } ( P | | Q ) - \left( \mathbb { E } _ { P } [ T ] - \log \mathbb { E } _ { Q } [ e ^ { T } ] \right)$ . Using the definition of KL divergence,

$$
\Delta = \mathbb {E} _ {P} \left[ \log \frac {d P}{d Q} \right] - \left(\mathbb {E} _ {P} [ T ] - \log \mathbb {E} _ {Q} [ e ^ {T} ]\right) = \mathbb {E} _ {P} \left[ \log \frac {d P}{d Q} \right] - \mathbb {E} _ {P} \left[ \log \frac {d G}{d Q} \right]. (3 8)
$$

Since expectation is linear,

$$
\Delta = \mathbb {E} _ {P} \left[ \log \frac {d P}{d Q} - \log \frac {d G}{d Q} \right] = \mathbb {E} _ {P} \left[ \log \frac {\frac {d P}{d Q}}{\frac {d G}{d Q}} \right] = \mathbb {E} _ {P} \left[ \log \frac {d P}{d G} \right]. \tag {39}
$$

By the definition of KL divergence,

$$
\Delta = \mathbb {E} _ {P} \left[ \log \frac {d P}{d G} \right] = D _ {K L} (P | | G) \geq 0. \tag {40}
$$

Since this holds for every admissible T , we can take the supremum over all such T :

$$
D _ {K L} (P | | Q) \geq \sup _ {T} \left\{\mathbb {E} _ {P} [ T ] - \log \mathbb {E} _ {Q} \left[ e ^ {T} \right] \right\}. \tag {41}
$$

To show that this lower bound is tight, choose

$$
T ^ {*} = \log \frac {d P}{d Q} + C, \tag {42}
$$

where C ∈ R is any constant. Then

$$
e ^ {T ^ {*}} = e ^ {\log \frac {d P}{d Q} + C} = e ^ {C} \frac {d P}{d Q}. \tag {43}
$$

So

$$
\mathbb {E} _ {Q} [ e ^ {T ^ {*}} ] = \int e ^ {T ^ {*}} d Q = \int e ^ {C} \frac {d P}{d Q} d Q = e ^ {C} \int d P = e ^ {C}, \tag {44}
$$

$$
\mathbb {E} _ {P} [ T ^ {*} ] = \mathbb {E} _ {P} \left[ \log \frac {d P}{d Q} + C \right] = D _ {K L} (P | | Q) + C. \tag {45}
$$

Therefore,

$$
\mathbb {E} _ {P} [ T ^ {*} ] - \log \mathbb {E} _ {Q} [ e ^ {T ^ {*}} ] = (D _ {K L} (P | | Q) + C) - \log e ^ {C} = D _ {K L} (P | | Q). \tag {46}
$$

The equality holds, and Theorem A.1 is proved.

![](images/d52fefb2abf3204c6a01c3cdecf295ffbddadef783da9b8f8a280632d0563cb0.jpg)

# A.3 Proof of Theorem 3.2

Proof. For simplicity, we denote $\mathbf { r } ^ { ( m ) } = f ^ { ( m ) } ( \mathbf { x } ^ { ( m ) } )$ . First, we can rewrite the InfoNCE loss of the m-th modality as:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{InfoNCE}} ^ {(m)} = - \frac {1}{N} \sum_ {n = 1} ^ {N} \log \frac {\exp \Big (s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n} ^ {[ M ] \setminus (m)})) / \tau \Big)}{\sum_ {n ^ {\prime} = 1 , n ^ {\prime} \neq n} ^ {N} \exp \Big (s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n ^ {\prime}} ^ {[ M ] \setminus (m)})) / \tau \Big)} \\ = - \frac {1}{N} \sum_ {n = 1} ^ {N} \left[ \frac {s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n} ^ {[ M ] \setminus (m)}))}{\tau} \right. \\ - \log \sum_ {n ^ {\prime} = 1, n ^ {\prime} \neq n} ^ {N} \exp \left(\frac {s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n ^ {\prime}} ^ {[ M ] \setminus (m)}))}{\tau}\right) \Biggr ] \\ = - \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n} ^ {[ M ] \setminus (m)}))}{\tau} \\ + \frac {1}{N} \sum_ {n = 1} ^ {N} \log \sum_ {n ^ {\prime} = 1, n ^ {\prime} \neq n} ^ {N} \exp \left(\frac {s (\mathbf {z} _ {n} ^ {(m)} , t (\mathbf {z} _ {n ^ {\prime}} ^ {[ M ] \setminus (m)}))}{\tau}\right) \\ = - \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {1}{\tau} s \Big (g ^ {(m)} (\mathbf {r} ^ {(m)}), t \Big (g ^ {[ M ] \setminus (m)} (\mathbf {r} _ {n} ^ {[ M ] \setminus (m)}) \Big) \Big) \\ \end{array}
$$

$$
\left. \right. + \frac {1}{N} \sum_ {n = 1} ^ {N} \log \sum_ {n ^ {\prime} = 1, n ^ {\prime} \neq n} ^ {N} \exp \left(\frac {1}{\tau} s \left(g ^ {(m)} \left(\mathbf {r} _ {n} ^ {(m)}\right), t \left(g ^ {[ M ] \backslash (m)} \left(\mathbf {r} _ {n ^ {\prime}} ^ {[ M ] \backslash (m)}\right)\right)\right)\right), \tag {47}
$$

where $g ^ { [ M ] \setminus ( m ) } ( \mathbf { r } _ { n } ^ { [ M ] \setminus ( m ) } ) = \underbrace { g ^ { ( 1 ) } ( \mathbf { r } _ { n } ^ { ( 1 ) } ) , \dots , g ^ { ( M ) } ( \mathbf { r } _ { n } ^ { ( M ) } ) } _ { \mathrm { e x c l u d i n g ~ } m } .$ . We can rewrite the above as the {zexcluding m expectation form and therefore remove the subscript n:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{InfoNCE}} ^ {(m)} = - \mathbb {E} _ {P (\mathbf {r} ^ {[ M ]})} \bigg [ \frac {1}{\tau} s \big (g ^ {(m)} (\mathbf {r} _ {n} ^ {(m)}), g ^ {[ M ] \setminus (m)} (\mathbf {r} _ {n} ^ {[ M ] \setminus (m)}) \big) \bigg ] \\ + \mathbb {E} _ {P (\mathbf {r} ^ {(m)})} \left[ \log \mathbb {E} _ {P (\mathbf {r} ^ {[ M ] \setminus (m)})} \right] \exp \left(\frac {1}{\tau} s \big (g ^ {(m)} (\mathbf {r} _ {n} ^ {(m)}), \right. \\ \left. \left. g ^ {[ M ] \backslash (m)} (\mathbf {r} _ {n} ^ {[ M ] \backslash (m)})\right) \right] \Bigg ] - \log N \tag {48} \\ = \mathbb {E} _ {P (\mathbf {r} ^ {(m)})} \left[ - \mathbb {E} _ {P (\mathbf {r} ^ {[ M ] \setminus (m)} | \mathbf {r} ^ {(m)})} \left[ \frac {1}{\tau} s \big (g ^ {(m)} (\mathbf {r} _ {n} ^ {(m)}), \right. \right. \\ \left. \left. g ^ {[ M ] \setminus (m)} (\mathbf {r} _ {n} ^ {[ M ] \setminus (m)})\right) \right] \\ + \log \mathbb {E} _ {P (\mathbf {r} ^ {[ M ] \setminus (m)})} \left[ \exp \left(\frac {1}{\tau} s \big (g ^ {(m)} (\mathbf {r} _ {n} ^ {(m)}), \right. \right. \\ \end{array}
$$

$$
\left. \left. g ^ {[ M ] \backslash (m)} (\mathbf {r} _ {n} ^ {[ M ] \backslash (m)})\right) \right] \Biggr ] - \log N \tag {49}
$$

$$
= \mathbb {E} _ {P (\mathbf {r} ^ {(m)})} \Bigg [ - \mathbb {E} _ {P (\mathbf {r} ^ {[ M ] \setminus (m)} | \mathbf {r} ^ {(m)})} \Big [ T (\mathbf {r} ^ {[ M ]}) \Big ]
$$

$$
\left. + \log \mathbb {E} _ {P (\mathbf {r} ^ {[ M ] \setminus (m)})} \left[ e ^ {T (\mathbf {r} ^ {[ M ]})} \right] \right] - \log N \tag {50}
$$

$$
= - \left(\mathbb {E} _ {P (\mathbf {r} ^ {(m)})} \left[ \mathbb {E} _ {P (\mathbf {r} ^ {[ M ] \setminus (m)} | \mathbf {r} ^ {(m)})} \left[ T (\mathbf {r} ^ {[ M ]}) \right] \right. \right.
$$

$$
\left. \left. - \log \mathbb {E} _ {P (\mathbf {r} ^ {[ M ] \backslash (m)})} \left[ e ^ {T (\mathbf {r} ^ {[ M ]})} \right] \right]\right) - \log N \tag {51}
$$

$$
\geq - I (\mathbf {r} ^ {(m)}; \mathbf {r} ^ {[ M ] \setminus (m)}) - \log N. \tag {52}
$$

where $\mathbf { r } ^ { [ M ] } = \mathbf { r } ^ { ( 1 ) } , \ldots , \mathbf { r } ^ { ( M ) } , P ( \mathbf { r } ^ { [ M ] } )$ and $P ( { \bf r } ^ { [ M ] \backslash ( m ) } )$ are the joint distributions, and $P ( { \bf r } ^ { [ M ] \backslash ( m ) } | { \bf r } ^ { ( m ) } ) , P ( { \bf r } ^ { ( m ) } )$ are respectively conditional and marginal distributions, and $T$ is a mapping that we parameterize with temperature factor τ , the over-parameterized modality-specific projection head $g ^ { ( m ) }$ and the score function s. Based on Theorem $\mathrm { A . 1 }$ , minimizing $\mathcal { L } _ { \mathrm { { I n f o N C E } } }$ is equivalent to maximizing a lower bound of $I ( { \mathbf { r } ^ { ( m ) } } ; { \mathbf { r } ^ { [ M ] \backslash ( m ) } } )$ . Since z(m) is stochastically determined by r(m), it also holds that minimizing L(m)InfoNCE maximizes $\mathbf { z } ^ { ( m ) }$ $\mathbf { r } ^ { ( m ) }$ $\mathcal { L } _ { \mathrm { I n f o N C E } } ^ { ( m ) }$ the lower bound of $I ( \mathbf { z } ^ { ( m ) } ; \mathbf { z } ^ { [ M ] \backslash ( m ) } )$ .

![](images/c2668164e7eb4bb0aa655f89c5ee30bdccc76f453f92adaa6ac5769bb3ca19c9.jpg)

# A.4 Proof of Theorem 3.3

Proof. For brevity, denote

$$
\mathrm{TC} (\mathbf {z} ^ {[ M ]}) = \sum_ {m = 1} ^ {M} H (\mathbf {z} ^ {(m)}) - H (\mathbf {z} ^ {[ M ]}), \tag {53}
$$

and

$$
\mathrm{DTC} (\mathbf {z} ^ {[ M ]}) = H (\mathbf {z} ^ {[ M ]}) - \sum_ {m = 1} ^ {M} H (\mathbf {z} ^ {(m)} | \mathbf {z} ^ {[ M ] \setminus (m)}). \tag {54}
$$

The sum of One-vs-All mutual information terms can be rewritten as

$$
\sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}) = \sum_ {m = 1} ^ {M} \left[ H (\mathbf {z} ^ {(m)}) - H (\mathbf {z} ^ {(m)} \mid \mathbf {z} ^ {[ M ] \setminus (m)}) \right]. \tag {55}
$$

Adding and subtracting $H ( \mathbf { z } ^ { [ M ] } )$ , we obtain

$$
\sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}) = \sum_ {m = 1} ^ {M} \left[ H (\mathbf {z} ^ {(m)}) - H (\mathbf {z} ^ {[ M ]}) + H (\mathbf {z} ^ {[ M ]}) - H (\mathbf {z} ^ {(m)} \mid \mathbf {z} ^ {[ M ] \setminus (m)}) \right] \tag {56}
$$

$$
= \mathrm{TC} (\mathbf {z} ^ {[ M ]}) + \mathrm{DTC} (\mathbf {z} ^ {[ M ]}). \tag {57}
$$

By Han’s entropy inequalities [15]:

$$
H (\mathbf {z} ^ {[ M ]}) \leq \frac {1}{M - 1} \sum_ {m = 1} ^ {M} H (\mathbf {z} ^ {[ M ] \setminus (m)}), \tag {58}
$$

total correlation and dual total correlation satisfy

$$
\mathrm{TC} (\mathbf {z} ^ {[ M ]}) \leq (M - 1) \mathrm{DTC} (\mathbf {z} ^ {[ M ]}), \tag {59}
$$

and

$$
\mathrm{DTC} (\mathbf {z} ^ {[ M ]}) \leq (M - 1) \mathrm{TC} (\mathbf {z} ^ {[ M ]}). \tag {60}
$$

Using Inequality (59),

$$
\frac {1}{M} \sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}) = \frac {1}{M} \left[ \mathrm{TC} (\mathbf {z} ^ {[ M ]}) + \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) \right] \leq \mathrm{DTC} (\mathbf {z} ^ {[ M ]}). \tag {61}
$$

Using Inequality (60),

$$
\begin{array}{l} \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) = \frac {1}{M} \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) + \frac {M - 1}{M} \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) \\ \leq \frac {M - 1}{M} \mathrm{TC} (\mathbf {z} ^ {[ M ]}) + \frac {M - 1}{M} \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) \\ = \frac {M - 1}{M} \left[ \mathrm{TC} (\mathbf {z} ^ {[ M ]}) + \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) \right] \\ = \frac {M - 1}{M} \sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}). \tag {62} \\ \end{array}
$$

Combining Inequality (61) and Inequality (62) gives

$$
\frac {1}{M} \sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}) \leq \mathrm{DTC} (\mathbf {z} ^ {[ M ]}) \leq \frac {M - 1}{M} \sum_ {m = 1} ^ {M} I (\mathbf {z} ^ {(m)}; \mathbf {z} ^ {[ M ] \setminus (m)}), \tag {63}
$$

which completes the proof.

![](images/f6c1650fd36fc99ff3da95518f6825d165bad02bac9899357db043394e61f511.jpg)

# A.5 Proof of Theorem 3.4

Proof.

$$
\begin{array}{l} I \left(\mathbf {z} ^ {(m)}; \mathbf {x} ^ {(m)}\right) = \iint p _ {\theta^ {(m)}} \left(\mathbf {z}, \mathbf {x} ^ {(m)}\right) \log \frac {p _ {\theta^ {(m)}} \left(\mathbf {z} \mid \mathbf {x} ^ {(m)}\right)}{p _ {\theta^ {(m)}} (\mathbf {z})} d \mathbf {z} d \mathbf {x} ^ {(m)} (64) \\ = \int \underbrace {\int \cdots \int} _ {M} p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) p (\mathbf {x} ^ {(m)}, \mathbf {x} ^ {[ M ] \setminus (m)}) \\ \times \log \frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{p _ {\theta^ {(m)}} (\mathbf {z})} d \mathbf {z} d \mathbf {x} ^ {[ M ])} (65) \\ = \underbrace {\int \cdots \int} _ {M} \int p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) p (\mathbf {x} ^ {(m)}, \mathbf {x} ^ {[ M ] \setminus (m)}) \\ \times \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)})} \cdot \frac {\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)})}{p _ {\theta^ {(m)}} (\mathbf {z})}\right) d \mathbf {x} ^ {[ M ]} d \mathbf {z}, (66) \\ \end{array}
$$

where $d { \bf x } ^ { [ M ] } = d { \bf x } ^ { ( 1 ) } \ldots d { \bf x } ^ { ( M ) }$ for brevity, and $\begin{array} { r } { p _ { \theta ^ { ( m ) } } = \int p _ { \theta ^ { ( m ) } } \big ( \mathbf { z } \big | \mathbf { x } ^ { ( m ) } \big ) p \big ( \mathbf { x } ^ { ( m ) } \big ) d \mathbf { x } ^ { ( m ) } } \end{array}$ is the marginal representation distribution induced by the m-th encoder, and $\begin{array} { r } { \int \prod _ { i \neq m } p _ { \theta ^ { ( i ) } } ( \mathbf { z } | \mathbf { x } ^ { ( i ) } ) d \mathbf { z } \neq } \end{array}$ 1, so $\begin{array} { r } { \prod _ { i \neq m } p _ { \theta ^ { ( i ) } } ( \mathbf { z } | \mathbf { x } ^ { ( i ) } ) } \end{array}$ cannot be regarded as a probability. Under this condition, we introduce the normalization constant:

$$
C = \int \prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {u} | \mathbf {x} ^ {(i)}) d \mathbf {u}, \tag {67}
$$

and we denote by pθ[M]\(m) (z|x[M ]\(m)) = Qi̸=m pθ(i) (z|x(i)) . $\begin{array} { r } { p _ { \theta ^ { [ M ] \setminus ( m ) } } ( \mathbf { z } | \mathbf { x } ^ { [ M ] \setminus ( m ) } ) = \frac { \prod _ { i \neq m } p _ { \theta ^ { ( i ) } } ( \mathbf { z } | \mathbf { x } ^ { ( i ) } ) } { C } } \end{array}$ C For the logarithm, we can have:

$$
\log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)})} \cdot \frac {\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)})}{p _ {\theta^ {(m)}} (\mathbf {z})}\right) \tag {68}
$$

$$
= \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)})} \cdot \frac {\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)})}{p _ {\theta^ {(m)}} (\mathbf {z})} \cdot \frac {C}{C}\right) \tag {69}
$$

$$
= \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)}) / C} \cdot \frac {\prod_ {i \neq m} p _ {\theta^ {(i)}} (\mathbf {z} | \mathbf {x} ^ {(i)}) / C}{p _ {\theta^ {(m)}} (\mathbf {z})}\right) \tag {70}
$$

$$
= \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})} \cdot \frac {p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}{p _ {\theta^ {(m)}} (\mathbf {z})}\right) \tag {71}
$$

$$
= \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}\right) - \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}\right). \tag {72}
$$

Therefore, the mutual information becomes:

$$
I \left(\mathbf {z} ^ {(m)}; \mathbf {x} ^ {(m)}\right) \tag {73}
$$

$$
= \underbrace {\int \cdots \int} _ {M} p (\mathbf {x} ^ {[ M ]}) \int p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}\right) d \mathbf {z} d \mathbf {x} ^ {[ M ]}
$$

$$
- \underbrace {\int \cdots \int} _ {M} p (\mathbf {x} ^ {[ M ]}) \int p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}\right) d \mathbf {z} d \mathbf {x} ^ {[ M ]} \tag {74}
$$

$$
= \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \right. \left\| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})\right) \right]
$$

$$
- \underbrace {\int \cdots \int} _ {M} \int p (\mathbf {x} ^ {[ M ]}) p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}\right) d \mathbf {z} d \mathbf {x} ^ {[ M ]}, \tag {75}
$$

where $\begin{array} { r } { D _ { \mathrm { K L } } ( p ( x ) | | q ( x ) ) = \int p ( x ) \log \frac { p ( x ) } { q ( x ) } d x \geq 0 . \ p ( \mathbf { x } ^ { [ M ] } ) p _ { \theta ^ { ( m ) } } ( \mathbf { z } | \mathbf { x } ^ { ( m ) } ) } \end{array}$ defines a joint $( \mathbf { x } ^ { [ M ] } , \mathbf { z } )$

$$
\int p (\mathbf {x} ^ {[ M ]}) p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) d \mathbf {x} ^ {[ M ]} = \int p (\mathbf {x} ^ {(m)}) p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) = p _ {\theta^ {(m)}} (\mathbf {z}). \tag {76}
$$

Therefore, by Bayes’s rule, the induced conditional distribution over $\mathbf { x } ^ { [ M ] }$ given z is

$$
p _ {\theta^ {(m)}} (\mathbf {x} ^ {[ M ]} | \mathbf {z}) = \frac {p (\mathbf {x} ^ {[ M ]}) p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)})}{p _ {\theta^ {(m)}} (\mathbf {z})}. \tag {77}
$$

Using this factorization, the mutual information can be rewritten as

$$
I \left(\mathbf {z} ^ {(m)}; \mathbf {x} ^ {(m)}\right) \tag {78}
$$

$$
= \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \right. \left\| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})\right) \right]
$$

$$
- \underbrace {\int \cdots \int} _ {M} \int p _ {\theta^ {(m)}} (\mathbf {x} ^ {[ M ]} | \mathbf {z}) p _ {\theta^ {(m)}} (\mathbf {z}) \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}\right) d \mathbf {z} d \mathbf {x} ^ {[ M ]} \tag {79}
$$

$$
= \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \right. \right\lVert p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)}) \Bigg) \Bigg ]
$$

$$
- \underbrace {\int \cdots \int} _ {M} p _ {\theta^ {(m)}} (\mathbf {x} ^ {[ M ]} | \mathbf {z}) \int p _ {\theta^ {(m)}} (\mathbf {z}) \log \left(\frac {p _ {\theta^ {(m)}} (\mathbf {z})}{p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})}\right) d \mathbf {z} d \mathbf {x} ^ {[ M ]} \tag {80}
$$

$$
\begin{array}{l} = \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \right. \left\| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})\right) \right] \\ - \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]} | \mathbf {z})} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z}) \left\| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})\right) \right] \right. (81) \\ \leq \mathbb {E} _ {p (\mathbf {x} ^ {[ M ]})} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \right. \left\| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})\right) \right]. (82) \\ \end{array}
$$

![](images/c0bcad8f7e22ede85137c81fd5432127f01d54770a538dd5a4d9017abc2dc20e.jpg)

# A.6 Proof of Proposition 3.1

Proof of Proposition 3.1. Let $p _ { \theta ^ { ( m ) } } ( \mathbf { z } | \mathbf { x } ^ { ( m ) } ) = \mathcal { N } ( \mathbf { z } ; \mu ^ { ( m ) } , \sigma ^ { 2 } I )$ with $\boldsymbol \mu ^ { ( m ) } = \mu _ { \boldsymbol \theta ^ { ( m ) } } ( \mathbf { x } ^ { ( m ) } )$ , so $p _ { \theta ^ { [ M ] \setminus ( m ) } } \big ( \mathbf { z } \big | \mathbf { x } ^ { [ \bar { M } ] \setminus ( m ) } \big )$ also follows a Gaussian distribution, namely:

$$
p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)}) \sim \mathcal {N} (\mathbf {z}; \bar {\mu}, \frac {\sigma^ {2}}{M - 1} I), \tag {83}
$$

where the mean ¯µ = 1M−1 $\begin{array} { r } { \bar { \mu } = \frac { 1 } { M - 1 } \sum _ { i = 1 , i \neq m } ^ { M } \mu ^ { ( i ) } } \end{array}$ and the covariance σ2M−1 I. Using the closed-form ${ \frac { \sigma ^ { 2 } } { M - 1 } } I .$ expression of the KL divergence between Gaussian distributions, we obtain:

$$
D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \Big \| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})\right) = \frac {M - 1}{2 \sigma^ {2}} | | \mu^ {(m)} - \bar {\mu} | | _ {2} ^ {2} + C _ {M}, \tag {84}
$$

where $C _ { M }$ is a constant independent of the mean parameters. Therefore, up to constants, minimizing this KL divergence is equivalent to minimizing the squared distance between $\mu ^ { ( m ) }$ and the average of the means of the rest of the modalities $\bar { \mu } ,$ , namely:

$$
D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \Big \| p _ {\theta^ {[ M ] \setminus (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \setminus (m)})\right) \propto | | \mu_ {\theta^ {(m)}} (\mathbf {x} ^ {(m)}) - \bar {\mu} | | _ {2} ^ {2}. \tag {85}
$$

Then

$$
\mathbb {E} _ {p \left(\mathbf {x} ^ {(1)}, \dots , \mathbf {x} ^ {(M)}\right)} \left[ D _ {\mathrm{KL}} \left(p _ {\theta^ {(m)}} (\mathbf {z} | \mathbf {x} ^ {(m)}) \right. \| p _ {\theta^ {[ M ] \backslash (m)}} (\mathbf {z} | \mathbf {x} ^ {[ M ] \backslash (m)})\right) \Bigg ] \tag {86}
$$

$$
\propto \mathbb {E} _ {p (\mathbf {x} ^ {(1)}, \dots , \mathbf {x} ^ {(M)})} \left[ | | \mu_ {\theta^ {(m)}} (\mathbf {x} ^ {(m)}) - \bar {\mu} | | _ {2} ^ {2} \right]. \tag {87}
$$

![](images/48bb4a255c7961cabdd63ff80c8e368e07f158e6278f3c8cb1c9a3bb1a507bb7.jpg)

# B Experimental Details

# B.1 Dataset

Vision&Touch. We use the Vision&Touch dataset [20], which provides synchronized modalities from multiple sensors, including third-person RGB images, force–torque histories, end-effector pose, depth, and optical flow. Using this dataset, we construct the following downstream tasks:

Vision–Force–Depth ( VFD). This setting uses RGB images $[ 3 \times 1 2 8 \times 1 2 8 ]$ , force– torque histories $[ 3 2 \times 6 ]$ , and depth observations $[ 1 \times 1 2 8 \times 1 2 8 ]$ . We evaluate two tasks: next-step end-effector orientation prediction, formulated as a 4-dimensional regression problem, and trajectory-pair classification, where the model predicts whether two modality triplets are sampled from the same trajectory.

Vision–Force–Proprioception ( VFP). This setting uses RGB images, force–torque histories, and end-effector pose vectors [7]. The downstream task is binary contact prediction, where the model classifies whether the end-effector is in contact with the object.

Higher-modality trajectory classification. To evaluate OVA-IB beyond the standard three-modality setting, we construct four- and five-modality trajectory classification tasks from Vision & Touch. The four-modality (VFPD) setting uses RGB, force–torque, proprioception, and depth, while the five-modality (VFPDO) setting additionally includes optical flow $[ 2 \times 1 2 8 \times 1 2 8 ]$ . In both settings, the model predicts whether two modality tuples are sampled from the same trajectory.

MuJoCo Push. We use the MuJoCo Push dataset [19], a planar pushing benchmark in which a Franka Emika Panda arm interacts with a puck. The modalities include grayscale image histories $[ 3 2 \times 3 2 \times 3 2 ]$ , force–torque histories $[ 3 2 \times 6 ]$ , and end-effector pose histories $[ 3 2 \times 7 ]$ , where the leading dimension corresponds to temporal history. The task is to predict the next-step 2D position of the object on the table.

CMU-MOSEI. We use the CMU-MOSEI benchmark preprocessed by MultiBench [21], which contains visual $[ 3 5 \times 7 1 3 ]$ ], acoustic $[ 3 2 \times 7 4 ]$ , and textual [32 × 300] modalities. Following the standard binary sentiment setting, the task is to classify whether the sentiment label is positive.

WESAD. We use WESAD [28] to evaluate both cross-device physiological retrieval and downstream affective state classification. The dataset provides synchronized recordings from chest- and wrist-worn wearable devices. For retrieval, we construct the task Resp+ECG → BVP, where chest respiration $\left( \left[ 1 \times 3 5 0 0 \right] \right)$ ) and ECG ([1×3500]) retrieve the synchronized wrist BVP signal $\left( \left[ 1 \times 3 2 0 \right] \right)$ . This task is physiologically grounded: ECG reflects cardiac electrical activity, respiration provides cardiopulmonary context, and BVP captures peripheral pulse dynamics at the wrist. It thus evaluates whether learned representations align central chest-side signals with peripheral wrist-side responses from the same time window. Additionally, we evaluate the learned representations on a downstream three-class affective classification task (baseline, stress, amusement).

# B.2 Experimental Implementation

We implement all methods using modality-specific encoders followed by projection heads. To ensure a fair comparison, OVA-IB and all baselines use the same backbone architectures, projection heads, downstream predictors, optimizer, batch size, temperature, and random seed whenever applicable. The only difference across methods is the pretraining objective.

Modality encoders. For image-like modalities, we use ResNet-18 backbones [16]. RGB inputs are processed by a standard ResNet-18 visual encoder. For depth observations, which are single-channel inputs, we duplicate the depth map along the channel dimension to form a 3-channel input and process it with a standard ResNet-18. For optical flow, we adapt the first convolutional layer of ResNet-18 to accept 2 input channels. For MuJoCo Push, the image history is treated as a 32-channel image input, and we adapt the first convolutional layer of ResNet-18 to accept 32 input channels while keeping the remaining architecture unchanged.

For force–torque histories, we use a one-dimensional convolutional encoder. The force history sequence is passed through 1D convolutional layers with ReLU activations, followed by adaptive average pooling and a final linear layer. This extracts temporal sensor patterns while producing a fixed-dimensional representation independent of the input sequence length. For low-dimensional proprioceptive inputs such as end-effector pose vectors, we use a lightweight MLP encoder with LeakyReLU activation. For CMU-MOSEI, the visual, acoustic, and textual modalities are provided as pre-extracted temporal feature sequences, and we use LSTM [17] encoders to aggregate temporal information for each modality. For WESAD, we use a 3-layer 1D-CNN as the encoder.

All modality encoders output a 256-dimensional representation. Each representation is then passed through a modality-specific projection head using a 3-layer MLP, which maps it to a 256-dimensional embedding used for contrastive alignment and the OVA-IB objective.

Hyperparameters. All models are optimized with Adam. We use a batch size of 64, temperature $\tau = 0 . 0 1$ , representation dimension 256, and embedding dimension 256 for all experiments. Learning rates are selected by dataset and downstream task: 10−3 for MuJoCo Push regression, 10−5 for CMU-MOSEI sentiment classification, 10−4 for VFD trajectory classification, $1 0 ^ { - 3 }$ for VFD orientation regression, and $1 0 ^ { - 4 }$ for the four- and five-modality VFPD/VFPDO trajectory classification tasks. For all tasks, we use random seed 42, and we also use 0 and 1 for the cross-modal retrieval. For our OVA-IB loss, we set $\lambda = 1 0 ^ { - 8 }$ .

Downstream evaluation. After pretraining, we use the learned modality encoders to extract modality representations and concatenate the available modality embeddings for downstream prediction. The downstream predictor is a 4-layer MLP. For classification tasks, the predictor is trained with cross-entropy loss; for regression tasks, it is trained with mean squared error. For modality-agnostic evaluation, we reuse the encoders pre-trained by all modalities and evaluate them under different available modality combinations at test time.

Experimental Computer Resources Our experiments were run on an NVIDIA RTX 4090 GPU with 24 GB memory and an Intel(R) Xeon(R) Platinum 8470Q CPU.

# B.3 Modality-Agnostic Evaluation on Additional Tasks

We report the full modality-combination evaluation results on the remaining benchmarks. Following Section 4.3, models are pretrained with all modalities and evaluated using different available modality combinations at test time. These tables (5, 6, 7, 8, 9, 10) complement the main VFD classification results in Table 2 and show that OVA-IB remains robust across both single-modality and multi-modality evaluation settings.

Table 5: Modality-combination evaluation on MuJoCo Push regression. 

<table><tr><td>Method</td><td>V</td><td>F</td><td>P</td><td>V+F</td><td>V+P</td><td>F+P</td><td>V+F+P</td></tr><tr><td>Ours</td><td>0.0393</td><td>0.7107</td><td>0.3803</td><td>0.0707</td><td>0.0348</td><td>0.3753</td><td>0.0340</td></tr><tr><td>Symile</td><td>0.1584</td><td>0.6562</td><td>0.3808</td><td>0.2599</td><td>0.0910</td><td>0.3818</td><td>0.1228</td></tr><tr><td>Gram</td><td>0.0314</td><td>0.6677</td><td>0.3793</td><td>0.0722</td><td>0.0296</td><td>0.3859</td><td>0.0425</td></tr><tr><td>TRIANGLE</td><td>0.0459</td><td>0.7351</td><td>0.3795</td><td>0.0692</td><td>0.0379</td><td>0.3794</td><td>0.0656</td></tr><tr><td>Pairwise CLIP</td><td>0.0410</td><td>0.7084</td><td>0.3818</td><td>0.0686</td><td>0.0393</td><td>0.3744</td><td>0.0810</td></tr><tr><td>Pairwise CLIP + IB</td><td>0.1290</td><td>0.7031</td><td>0.3803</td><td>0.1839</td><td>0.1775</td><td>0.3763</td><td>0.3385</td></tr></table>

Table 6: Modality-combination evaluation on CMU-MOSEI sentiment classification. 

<table><tr><td>Method</td><td>V</td><td>A</td><td>T</td><td>V+A</td><td>V+T</td><td>A+T</td><td>V+A+T</td></tr><tr><td>Ours</td><td>71.30</td><td>71.04</td><td>75.76</td><td>71.36</td><td>75.65</td><td>75.76</td><td>76.02</td></tr><tr><td>Symile</td><td>71.04</td><td>71.04</td><td>72.99</td><td>71.04</td><td>71.04</td><td>71.04</td><td>77.00</td></tr><tr><td>Gram</td><td>71.60</td><td>71.04</td><td>76.28</td><td>71.56</td><td>76.64</td><td>75.85</td><td>72.77</td></tr><tr><td>TRIANGLE</td><td>71.04</td><td>71.04</td><td>77.01</td><td>71.04</td><td>75.78</td><td>76.56</td><td>76.13</td></tr><tr><td>Pairwise CLIP</td><td>71.23</td><td>71.08</td><td>76.10</td><td>71.73</td><td>76.08</td><td>76.04</td><td>75.05</td></tr><tr><td>Pairwise CLIP + IB</td><td>71.04</td><td>71.04</td><td>76.19</td><td>71.04</td><td>76.92</td><td>76.06</td><td>76.25</td></tr></table>

Table 7: Modality-combination evaluation on VFP contact classification. 

<table><tr><td>Method</td><td>V</td><td>F</td><td>P</td><td>V+F</td><td>V+P</td><td>F+P</td><td>V+F+P</td></tr><tr><td>Ours</td><td>82.77</td><td>96.74</td><td>83.97</td><td>96.71</td><td>83.56</td><td>96.99</td><td>97.12</td></tr><tr><td>Symile</td><td>83.32</td><td>97.42</td><td>82.90</td><td>95.71</td><td>83.27</td><td>97.33</td><td>96.44</td></tr><tr><td>Gram</td><td>83.23</td><td>97.10</td><td>83.22</td><td>95.66</td><td>83.22</td><td>97.40</td><td>96.85</td></tr><tr><td>TRIANGLE</td><td>82.31</td><td>97.16</td><td>83.08</td><td>95.81</td><td>82.19</td><td>97.06</td><td>95.91</td></tr><tr><td>Pairwise CLIP</td><td>83.01</td><td>97.24</td><td>83.22</td><td>95.68</td><td>82.98</td><td>97.19</td><td>96.55</td></tr><tr><td>Pairwise CLIP + IB</td><td>82.57</td><td>97.28</td><td>83.19</td><td>96.76</td><td>82.80</td><td>97.53</td><td>97.10</td></tr></table>

Table 8: Modality-combination evaluation on VFD orientation regression. 

<table><tr><td>Method</td><td>V</td><td>F</td><td>D</td><td>V+F</td><td>V+D</td><td>F+D</td><td>V+F+D</td></tr><tr><td>Ours</td><td>0.0002</td><td>0.0022</td><td>0.0003</td><td>0.0005</td><td>0.0002</td><td>0.0006</td><td>0.0003</td></tr><tr><td>Symile</td><td>0.0017</td><td>0.0022</td><td>0.0017</td><td>0.0017</td><td>0.0016</td><td>0.0017</td><td>0.0015</td></tr><tr><td>Gram</td><td>0.0020</td><td>0.0022</td><td>0.0023</td><td>0.0019</td><td>0.0018</td><td>0.0022</td><td>0.0018</td></tr><tr><td>TRIANGLE</td><td>0.0019</td><td>0.0022</td><td>0.0021</td><td>0.0020</td><td>0.0018</td><td>0.0021</td><td>0.0019</td></tr><tr><td>Pairwise CLIP</td><td>0.0017</td><td>0.0020</td><td>0.0017</td><td>0.0018</td><td>0.0014</td><td>0.0017</td><td>0.0013</td></tr><tr><td>Pairwise CLIP + IB</td><td>0.0010</td><td>0.0022</td><td>0.0015</td><td>0.0012</td><td>0.0008</td><td>0.0016</td><td>0.0009</td></tr></table>

Table 9: Modality-combination evaluation on four-modality VFPD trajectory classification. 

<table><tr><td>Method</td><td>V</td><td>F</td><td>P</td><td>D</td><td>V+F+P+D</td></tr><tr><td>Ours</td><td>91.14</td><td>94.56</td><td>89.38</td><td>90.20</td><td>94.73</td></tr><tr><td>Symile</td><td>83.29</td><td>93.01</td><td>88.72</td><td>82.91</td><td>84.97</td></tr><tr><td>Gram</td><td>90.14</td><td>89.93</td><td>87.67</td><td>89.70</td><td>90.81</td></tr><tr><td>Pairwise CLIP</td><td>92.94</td><td>91.54</td><td>88.55</td><td>91.13</td><td>93.25</td></tr><tr><td>Pairwise CLIP + IB</td><td>93.19</td><td>91.71</td><td>88.66</td><td>91.67</td><td>94.45</td></tr></table>

Table 10: Modality-combination evaluation on five-modality VFPDO trajectory classification. 

<table><tr><td>Method</td><td>V</td><td>F</td><td>P</td><td>D</td><td>O</td><td>V+F+P+D+O</td></tr><tr><td>Ours</td><td>90.79</td><td>95.37</td><td>90.81</td><td>90.02</td><td>87.14</td><td>95.91</td></tr><tr><td>Symile</td><td>82.18</td><td>92.70</td><td>87.88</td><td>82.55</td><td>82.42</td><td>82.96</td></tr><tr><td>Gram</td><td>90.40</td><td>89.11</td><td>87.50</td><td>89.99</td><td>81.62</td><td>90.59</td></tr><tr><td>Pairwise CLIP</td><td>92.54</td><td>92.06</td><td>87.89</td><td>89.27</td><td>86.28</td><td>92.95</td></tr><tr><td>Pairwise CLIP + IB</td><td>93.27</td><td>92.02</td><td>89.06</td><td>91.33</td><td>88.26</td><td>94.68</td></tr></table>

Table 11: Modality-combination evaluation on the WESAD dataset. 

<table><tr><td>Method</td><td>ECG</td><td>Resp</td><td>BVP</td><td>E+R</td><td>E+B</td><td>R+B</td><td>ERB</td></tr><tr><td>Ours</td><td>75.06</td><td>67.29</td><td>61.49</td><td>76.52</td><td>75.96</td><td>69.43</td><td>76.18</td></tr><tr><td>Symile</td><td>64.58</td><td>66.50</td><td>64.75</td><td>72.80</td><td>72.80</td><td>70.33</td><td>73.42</td></tr><tr><td>Gram</td><td>53.21</td><td>68.24</td><td>65.15</td><td>67.96</td><td>62.61</td><td>61.99</td><td>66.22</td></tr><tr><td>TRIANGLE</td><td>65.37</td><td>68.02</td><td>65.43</td><td>57.26</td><td>60.42</td><td>69.54</td><td>56.87</td></tr><tr><td>Pairwise CLIP</td><td>61.37</td><td>65.65</td><td>65.37</td><td>73.48</td><td>72.75</td><td>69.54</td><td>70.83</td></tr><tr><td>Pairwise CLIP + IB</td><td>75.39</td><td>66.39</td><td>67.29</td><td>77.14</td><td>76.97</td><td>70.78</td><td>73.03</td></tr></table>