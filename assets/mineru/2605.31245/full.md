# Toward Identifiable Sparse Autoencoders

Walter Nelson 1 Theofanis Karaletsos 2 Francesco Locatello 1

# Abstract

Recently, sparse autoencoders (SAEs) have emerged as an attractive tool for interpreting and interacting with representations in practical neural networks. While it is common empirical folklore, we also show theoretically that SAEs are highly unstable: different training runs are likely to produce different concept dictionaries and sparse codes. We characterize the model properties that hinder the stability of real-world SAEs, and address each of these problems through minimal changes to the architecture and training procedure. Together, these changes yield two versions of an identifiable SAE (iSAE), a variant of the standard TopK SAE with lower reconstruction error and improved stability. We explain this improvement theoretically by connecting SAEs with traditional dictionary learning approaches, and show that the dictionaries learned in practice satisfy an approximate restricted isometry condition, rendering the corresponding sparse codes in those models near-identifiable.

§ Code

# 1. Introduction

Sparse autoencoders (SAEs) decompose high-dimensional representations into a sparse linear combination of concepts from a large, learned dictionary. Due to their simplicity, flexibility, and well-characterized engineering (Gao et al., 2024), they have seen widespread adoption across vision, language, and other modalities.

In these settings, SAEs are used as a practical interface for analyzing and intervening on the internal representations. For example, individual dictionary atoms are often

1Institute of Science and Technology Austria 2Pyramidal Inc. and Achira Inc., USA. Correspondence to: Walter Nelson <walter.nelson@ista.ac.at>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

![](images/4562eb51a8327b69ab5dacbbb5078ad2159ed44e898f87279f024483dbcb3226.jpg)

<details>
<summary>natural_image</summary>

Abstract 3D wavy surface with grid lines, rendered in blue tones (no text or symbols)
</details>

Figure 1. Sparse autoencoders approximate nonlinear manifolds (dark blue, mostly occluded) with linear patches (light blue). We show that identifiability hinges on four key ingredients: (a) the approximation being good enough (low reconstruction error), (b) the manifold being sampled densely enough, (c) co-occurring concepts being distinct enough (an approximate restricted isometry property), and (d) sufficiently diverse concept co-occurrence patterns. When these hold, individual patches are identifiable, rendering the whole model statistically identifiable.

interpreted as human-meaningful concepts (e.g., topics, syntactic patterns, or visual components), enabling post-hoc interpretability. They have also been used for steering, where selectively activating or suppressing specific atoms alters model behavior in a targeted way. This line of work frequently relies, implicitly or explicitly, on the assumption that the sparse linear decompositions correspond to stable (or even semantically meaningful) structure in the underlying representation. In this work, we avoid this assumption by treating SAEs not as a priori interpretable objects, but as statistical estimators for the classical dictionary learning problem. Our goal is to investigate under what conditions their learned atoms and codes are in fact statistically identifiable.

We are motivated by prior work showing that modern SAEs suffer from various forms of non-identifiability (Song et al., 2025). In particular, prior work has shown that two SAEs trained on the same data are not guaranteed to yield the same concept dictionary (Song et al., 2025) or sparse codes (Paulo & Belrose, 2025), a phenomenon similar to the non-identifiability seen in disentangled representation learning (Locatello et al., 2019). This inconsistency potentially muddies the interpretation of the learned features, defeating the purpose of SAEs in the first place. SAE identifiability is also important when using them to interpret data and models in scientific applications, where identifiability ensures that statistical power is stable (Mencattini et al., 2026;

Donhauser et al., 2025).

The classical approaches to the compressed sensing and dictionary learning problems faced similar non-identifiability issues, which are mostly due to the fact that the dictionaries learned are highly overcomplete. We draw inspiration from these earlier works to improve the stability of the SAEs widely used in practice today. Concretely:

• We show theoretically (Theorem 3.2) and empirically that consistently learning the dictionary is not sufficient for consistently learning the codes, motivating our proposed code consistency metrics.   
• We show theoretically that when the learned dictionary satisfies an approximate restricted isometry property (aRIP), the sparse codes of an SAE are provably nearidentifiable (Theorem 3.6).   
• Empirically, we show that two variants of our model identifiable SAE (iSAE) exhibit improved stability and reconstruction over standard TopK baselines.

# 2. Related Work

A growing body of recent work applies SAEs as a tool for post-hoc interpretability, including mechanistic interpretability, particularly in large language models (Elhage et al., 2022; Olah et al., 2020; Bricken et al., 2023). In these settings, SAEs are trained on intermediate activations (e.g., residual streams or MLP layers), and individual dictionary elements are interpreted as corresponding to humanmeaningful concepts such as semantic topics, syntactic patterns, or behavioral circuits. This approach has enabled analyses of feature superposition (Elhage et al., 2022), circuit structure (Olah et al., 2020), and the localization of model behaviors, as well as interventions in which specific features are ablated or amplified to steer model outputs (Bricken et al., 2023; Turner et al., 2024). These applications rely on the empirical observation that SAE features are often sparse, localized, and partially interpretable, even in highly overcomplete regimes, provided that reconstruction performance is good enough.

Accordingly, most work on SAEs has focused on improving reconstruction error (Gao et al., 2024; Bussmann et al., 2024) at the frontier of the reconstruction-sparsity tradeoff (Fel et al., 2025). However, the evidence is unclear as to whether reconstruction performance is sufficient to give stability of the atoms and sparse codes. Song et al. (2025) point out the importance of the stability of the learned dictionary, arguing that TopK SAEs are stable according to this measure under re-trainings using a standard cosine similarity measure on the concept dictionary. Fel et al. (2025) propose to constrain the concepts in the dictionary to the convex hull of the data, finding that this improves stability according to the same cosine similarity measure. Building on prior work which shows that two SAEs with the same dictionary can produce vastly different sparse codes for the same data (Paulo & Belrose, 2025), we show that code identifiability is not implied by dictionary identifiability. Accordingly, we instead analyze the stability of both the dictionary and the sparse codes.

In most works on mechanistic interpretability, identifiability is assessed subjectively, by checking the alignment with human-interpretable concepts (Karvonen et al., 2025). In contrast, the historical view in sparse coding and dictionary learning is to assess when signals admit a stable sparse linear decomposition, without assigning a particular interpretation to it. Our work aims to bridge this gap by explicitly analyzing the conditions under which the representations learned by SAEs are stable and statistically identifiable, and by proposing metrics that capture this stability at the level of both dictionaries and codes.

Our theoretical characterization is based on the restricted isometry property (RIP) from compressed sensing. Candes et al. (2005) show that when a known dictionary satisfies the RIP, sparse codes are identifiable. Unfortunately, the classical RIP condition is combinatorially difficult to verify, a situation that does not become any easier when the dictionary is unknown (Spielman et al., 2012). To address this, we propose a data-driven relaxation of the RIP condition we refer to as approximate RIP (aRIP), which enforces the property only where data is actually observed. We show theoretically and empirically that this is sufficient for nearidentifiability of both the dictionary and the codes for a given data distribution. Appealingly, given a trained SAE, the aRIP condition is easily measured.

# 3. Unifying SAEs and Dictionary Learning

In this section, we contrast sparse autoencoders and the classical approaches to dictionary learning, with the goal of unifying them in theory and practice.

Sparse autoencoding. Consider a distribution $P ( \mathbf { x } )$ over representations $\mathbf { x } \in \mathbb { R } ^ { N }$ . A typical sparse autoencoder (SAE) takes the form:

$$
\mathbf {z} = \mathbf {f} (\mathbf {x}) \quad \hat {\mathbf {x}} = D \mathbf {z} + \mathbf {b} ^ {\prime} \tag {1}
$$

where $\mathbf { f } ~ : ~ \mathbb { R } ^ { N } ~ \to ~ \mathbb { R } ^ { K }$ is some encoder, $\mathbf { b } ^ { \prime } \in \mathbb { R } ^ { N }$ is a learned bias term, and $D \in \mathbb { R } ^ { N \times K }$ is the learned dictionary of N-dimensional atoms (or concepts, in SAE parlance). Typically, the encoder is near-linear of the form

$$
\mathbf {f} (\mathbf {x}) = \sigma (W (\mathbf {x} - \mathbf {b})) \tag {2}
$$

where $W \in \mathbb { R } ^ { K \times N }$ is the learned encoder matrix, σ is some sparsifier, and $\mathbf { b } \in \mathbb { R } ^ { N }$ is a bias term. The parameters are optimized by stochastic gradient descent to minimize $\| \mathbf { x } - { \hat { \mathbf { x } } } \| ^ { 2 }$ . Generally, the ambient dimension of the code space $K \gg N$ , but codes themselves z are k-sparse for some fixed k ≪ K. The current most popular choice for σ is the TopK function (Makhzani & Frey, 2013), which sparsifies to the top k largest coefficients. We refer to this architecture throughout as a TopK SAE, which also sets b = b′ (Gao et al., 2024).

Sparse coding and dictionary learning. In sparse coding, the dictionary D is fixed, and the goal in practice is to solve the optimization problem for a given input x:

$$
\min _ {\mathbf {z}} \| \mathbf {x} - D \mathbf {z} \| _ {2} \text {   s.t.   } \| \mathbf {z} \| _ {0} \leq k \tag {3}
$$

Traditional approaches include basis pursuit (Chen & Donoho, 1994) and orthogonal matching pursuit (OMP) (Chen et al., 1989). Optimization is generally intractable due to the highly non-convex nature of the $\ell _ { 0 }$ norm constraint and the resulting NP-hardness of the problem. Furthermore, even when it can be relaxed to the $\ell _ { 1 }$ form, without conditions on $D ,$ , the solution to the relaxation of (3) is known to be non-identifiable for general x. This means that multiple sparse codes $\mathbf { z } , \mathbf { z } ^ { \prime }$ are equally good at representing the same observation x, even when the dictionary is fixed. When the dictionary is learned, the situation is even more complicated. Most algorithms, such as K-SVD (Aharon et al., 2006), alternately solve a relaxation of (3) for numerous samples and update the dictionary.

# 3.1. Toward a unified approach

Sparse autoencoders have several appealing properties. They are easily implemented in modern deep learning frameworks and trained on modern hardware using stochastic gradient descent (Gao et al., 2024; Karvonen et al., 2025; Fel et al., 2025). For large language models in particular, their lightweight architecture means that a large cache of input activations can be inferred while the SAE is trained “online”, allowing training to scale to billions of tokens for the largest models (Karvonen et al., 2025; Gao et al., 2024). On the other hand, the solutions that SAEs learn are unstable in practice (Paulo & Belrose, 2025). For example, even if in some settings the dictionaries appear to be relatively stable in practice (Song et al., 2025), the amortized sparse codes themselves often are not (Paulo & Belrose, 2025). This means that the same SAE model trained twice on the same dataset and model might yield different concepts and sparse codes that affect downstream applications of the model.

Dictionary learning has the opposite characteristics. In general, it requires specialized approaches to optimization (Mairal et al., 2009) that fail to scale to the million- or billion-token regime required for SAEs to be useful for interpreting modern models. However, the solutions found by dictionary learning algorithms are often identifiable in theory and practice, meaning that the recovered dictionary and codes are learned consistently (Spielman et al., 2012). Given the high-stakes nature of the many common applications of SAEs, such as steering (O’Brien et al., 2025), interpretability (Cunningham et al., 2023) and oversight (Li et al., 2026), identifiability seems like a “bare minimum” requirement, highlighting the potential for bridging the gap between dictionary learning and SAEs.

In mechanistic interpretability applications, identifiability is often framed as recovering the “true” data-generating concepts under the linear representation hypothesis, a form of structural identifiability (Nelson et al., 2026). However, without access to the true concepts, this form of identifiability is impossible to evaluate. However, the run-to-run stability of the learned concepts (or atoms, in the language of dictionary learning) and sparse codes is easily assessed, including on real-world data. As such, we adopt the following definition of statistical identifiability for SAEs.

Definition 3.1 (SAE Identifiability). Let $P ( \mathbf { x } )$ denote a data distribution supported on X . Let $\mathbf { f } , \mathbf { f } ^ { \prime } : \mathcal { X }  \mathcal { Z } \subseteq \mathbb { R } ^ { K }$ denote the encoders of two k-sparse SAEs of the form in (1) trained independently on data from $P ( \mathbf { x } )$ , and let $D , D ^ { \prime } \in$ $\mathbb { R } ^ { N \times K }$ denote their decoders. Then, the SAE model is nearly identifiable in the limit of infinite data from $P ( \mathbf { x } )$ if there exist $\epsilon _ { z } , \epsilon _ { D } \geq 0$ such that $\| \mathbf { f } ( \mathbf { x } ) - \Pi \mathbf { f } ^ { \prime } ( \mathbf { x } ) \| _ { 2 } \leq \varepsilon _ { z }$ almost everywhere and $\| D - D ^ { \prime } \Pi \| \leq \epsilon _ { D }$ for some signed permutation matrix Π ∈ {−1, 0, 1}K×K. $\Pi \in \{ - 1 , 0 , 1 \} ^ { \dot { K } \times K }$

Intuitively, this definition says that an SAE is identifiable if independent trainings of the model (on infinite data) are guaranteed to yield approximately the same dictionary and sparse codes, up to some trivial equivariances in the model, namely the ordering and sign of the atoms in the dictionary (and therefore the sparse codes). Clearly, this definition is a “pre-requisite” for identifiability of the kind often pursued in mechanistic interpretability (Karvonen et al., 2025): if run-to-run identifiability is poor, identifiability of some ground-truth is necessarily poor as well. The benefit to this definition is that we don’t need to assume a particular data-generating process, and identifiability can therefore be assessed empirically with minimal assumptions.

Leveraging recent work on SAEs and a long history of dictionary learning results, we outline the following “dark triad” of SAE characteristics that hinder the understanding of identifiability in both theory and practice:

1. Bidirectional features. Most activation functions for SAEs, such as ReLU and TopK, restrict the non-zero coefficients of the sparse code z to be positive. As a result, Zhu et al. (2025) found that the dictionaries of trained SAEs often contain opposing concepts: for example, both a given atom d and an atom approximating its opposite d˜ ≈ −d will be present in the dictionary. In practice, this cuts the representation capacity of the dictionary in half, and renders the applicability of dictionary learning theory unclear at best (note that equation (3) does not specify ${ \bf z } \geq 0 )$ , because d and d˜ are coherent (have near-maximal absolute similarity), a known barrier to identifiability (Candes et al., 2005).

2. Dictionary conditioning. In general, solutions to equation (3) are unidentified without conditions on the concept dictionary D. For example, D must have low mutual coherence (maximum pairwise absolute concept similarity), or satisfy a condition like the restricted isometry property (Candes et al., 2005) in order for the sparse codes z to be uniquely recoverable from noisy observations. It is poorly understood whether SAEs learn dictionaries for which the corresponding sparse coding problem is identifiable.   
3. Encoder expressiveness. Even in the simplest case where the dictionary D is fixed and known, it is known that amortized inference of sparse codes is a difficult problem. For example, Gregor & LeCun (2010) shows that a specialized recurrent architecture is required to learn a good mapping $\textbf { x } \mapsto \textbf { z }$ from observations to sparse codes, even when paired observations $( \mathbf { x } , \mathbf { z } )$ are available from an oracle solver of equation (3). This suggests that the simple near-linear encoder typically employed in SAEs described in equation (2) is not expressive enough to learn the usual solutions to the sparse coding problem (3), which are the only solutions for which practical identifiability theory is available.

In the next sections, we describe the changes we make to the usual SAE model (1) that improve the performance of the model and advance our understanding of identifiability in this setting. The goal is to design a model with the “best of both worlds”: the well-characterized theory and guarantees of dictionary learning, with the ease of training and scalable engineering of sparse autoencoders.

# 3.2. Bidirectional features

Zhu et al. (2025) propose to change the TopK activation function in SAEs to an absolute value form motivated by the proximal gradient approach to solving equation (3). Specifically, they define:

$$
(\text { AbsTopK } _ {k} (u)) _ {i} = \left\{ \begin{array}{l l} u _ {i}, & i \in \mathcal {H} _ {k} (u), \\ 0, & i \notin \mathcal {H} _ {k} (u). \end{array} \right. \tag {4}
$$

where $\mathcal { H } _ { k } ( u )$ gives the indices of the k largest coefficients in absolute value. This addresses the first component of the “dark triad”, improving the expressivity of the model at the same number of parameters, reducing reconstruction error, and aligning more closely with existing approaches to dictionary learning, which allow for negative coefficients as in the $\ell _ { 0 }$ and $\ell _ { 1 }$ solutions to equation (3).

# 3.3. Dictionary conditioning

In statistics, the problem of estimating the dictionary D in this setting has a long history under the name “sparse overcomplete dictionary learning”, and has known estimation challenges. Indeed, even when D is known, the codes z might be non-identifiable (Candes et al., 2005), meaning multiple distinct sparse decompositions might exist. The issue becomes even worse under imperfect reconstruction, which is observed in practice in the SAE setting. To illustrate this failure mode, we construct a generic dictionary which can fail to identify codes, even if it approximates a reasonable observational distribution well in reconstruction. This is formalized in the following impossibility theorem.

Theorem 3.2 (Identifiability Impossibility). There exists a normalized dictionary $\bar { D ^ { \mathrm { ~ \in ~ } } } \bar { \mathbb { R } ^ { N \times K } }$ with the following property. Let $\mathbf { f } : \mathcal { X } \to \mathbb { R } ^ { K }$ be any k-sparse continuous encoder defining an SAE $( f , D )$ with reconstruction error tolerance $\epsilon \geq 0$ such that $\| D \mathbf { f } ( x ) - x \| \leq \epsilon f o t$ r almost all x drawn from a distribution $P ( x )$ . Assume further that every concept in D is activated with positive probability under $P ( x )$ . Then, there exists another continuous encoder f ′ : $\mathcal { X } \to \mathbb { R } ^ { K }$ that achieves the same reconstruction accuracy, but differs from f on a set of inputs with positive probability. Moreover, the sparsity patterns of f (x) and $\mathbf { f } ^ { \prime } ( x )$ differ with positive probability.

The proof is given in Appendix A. Theorem 3.2 shows that even if an SAE learns a perfectly stable dictionary, it may be the case that the corresponding sparse coding problem is non-identifiable. Indeed, the dictionary learning literature shows that if the dictionary is not fixed in advance but instead learned, the potential sources of non-identifiability are even more numerous (Spielman et al., 2012).

We now ask what conditions we can place on the SAE model in (1) to render both the dictionary and the codes identifiable, or nearly so. (Candes et al., 2005) show that a restricted isometry property (RIP) condition on the dictionary D is sufficient to render the codes unique, up to permutations. The condition takes the form

$$
(1 - \delta) \| \mathbf {z} \| ^ {2} \leq \| D \mathbf {z} \| ^ {2} \leq (1 + \delta) \| \mathbf {z} \| ^ {2} \tag {5}
$$

for some $\delta \geq 0 ,$ , and must hold for any k-sparse z. Intuitively, it says that the reconstructions must have approximately the same norms as the corresponding sparse codes. If D were square, any nearly orthogonal matrix would satisfy the condition. But, when D is overcomplete and is learned, the RIP condition is combinatorially difficult to enforce, because every dictionary has K choose k possible combinations of concepts where the condition must be checked. Intuitively, for overcomplete dictionaries, any combination of k atoms must be “nearly orthogonal”.

To address this difficulty of combinatorial complexity, we propose to enforce (5) only on the sparse codes z actually observed in the latent space of the sparse autoencoder. We refer to this condition as an approximate restricted isometry property, because it of course does not imply the usual RIP condition. However, we will show that it is sufficient to prove useful identifiability results.

Definition 3.3 (Approximate RIP). Let $P ( \mathbf { z } )$ be a distribution over sparse codes, and let $\mathbf { z } _ { 1 } , \mathbf { z } _ { 2 }$ be independent samples from $P ( \mathbf { z } )$ . Define $S$ to be a random variable, representing the union of the indices where $\mathbf { z } _ { 1 }$ and $\mathbf { z } _ { 2 }$ are non-zero. Then, D satisfies the approximate restricted isometry property (aRIP) with respect to $P ( \mathbf { z } )$ at level δ if $( 1 - \delta ) \lvert | \overline { { \mathbf { z } _ { S } ^ { \prime } } } \rvert | ^ { 2 } \leq \lvert | D _ { S } \mathbf { z } _ { S } ^ { \prime } \rvert | ^ { 2 } \leq ( 1 + \bar { \delta } ) \lvert | \mathbf { z } _ { S } ^ { \prime } \rvert | ^ { 2 }$ for any vector $\mathbf { z } _ { S } ^ { \prime } \in \mathbb { R } ^ { K }$ such that its nonzero coefficients have indices $S ,$ almost everywhere with respect to $P ( S )$ .

Intuitively, this means that any union of two observed sparsity patterns in the sparse code distribution must satisfy the usual restricted isometry property (RIP). This is in contrast to more general definitions which demand the condition hold for the union of any two (or three) sparsity patterns to achieve identifiability. Such definitions can be used to prove identifiability results that hold for all possible observable signals $\mathbf { x } \in \dot { \mathbb { R } } ^ { N }$ (Candes et al., 2005). In contrast, ours will only hold for a particular distribution of signals $P ( \mathbf { x } )$ . For these results to hold, we will need the following assumptions on the distribution $P ( \mathbf { z } )$ , which in the case of a sparse autoencoder of the form in (1) is the pushforward of the data distribution $P ( \mathbf { x } )$ by the encoder $f .$

Assumption 3.4 (Sufficient Richness). Let z and $\mathbf { z } ^ { \prime }$ be independent sparse codes from $P ( \mathbf { z } )$ , with nonzero index sets $S$ and $S ^ { \prime }$ respectively. Then $P ( \mathbf { z } )$ satisfies sufficient richness of supports if $P ( S \cap S ^ { \prime } = \{ i \} ) > 0$ and $P ( i \in$ $S , j \in S ^ { \prime } , S \cap S ^ { \prime } = \varnothing ) > 0$ for any pair of atoms $i \neq j$ . Intuitively, these imply that no atom can occur in a single support S and no two atoms can always co-occur.

Assumption 3.5 (Sufficient Diversity). Let z and S be as defined in Assumption 3.4. Then, $P ( \mathbf { z } )$ satisfies sufficient diversity of observations if for each observed support $S$ of size k, there exist k independent samples from $\mathbf { z } _ { S } \mid S$ such that the $k \times k$ matrix formed by stacking these samples has smallest singular value at least $\zeta > 0$ .

Measuring aRIP Because we don’t need to sample every possible combination of sparsity patterns to quantify aRIP (Definition 3.3), we might reasonably hope to measure how well the condition holds for a given distribution of sparse codes $P ( \mathbf { z } )$ using samples from the distribution. The key algebraic relationship to notice is that if $\lambda _ { 1 } , \ldots , \lambda _ { u }$ are the eigenvalues of the Gram submatrix $G _ { S } = D _ { S } ^ { \boldsymbol { \mathsf { T } } } D _ { S }$ in ascending order, the optimal aRIP constant δ satisfies $1 - \delta \leq \lambda _ { 1 } \leq \lambda _ { s } \leq 1 + \delta$ . When D is normalized, the mean eigenvalue $\bar { \lambda } = 1$ , and we have the following relationship:

$$
\mathcal {R} _ {\mathrm{aRIP}} (S) := \frac {1}{| S |} \sum_ {i = 1} ^ {| S |} (\lambda_ {i} - 1) ^ {2} = \frac {\operatorname{tr} \left(G _ {S} ^ {2}\right)}{| S |} - 1 \tag {6}
$$

Intuitively, this equation measures the average deviation of each eigenvalue from 1, whereas the aRIP condition witnesses only the largest deviations of the eigenvalues from 1. However, the eigenvalue bound does imply that at the minimum of $\mathcal { R } _ { \mathrm { a R I P } } ( S )$ over the space of normalized subdictionaries $D _ { S }$ , we have $\delta = 0$ .

We draw inspiration from work which regularizes the inputoutput Jacobian of neural networks for improving representation learning (Lee et al., 2022), allowing us to straightforwardly estimate (6). Let S be a set of atom indices. With a slight abuse of notation due to the random dimensionality of $S ,$ let $\mathbf { n } _ { S } \sim \mathcal { N } ( 0 , \mathbf { I } _ { | S | } )$ , so we have (up to an additive constant):

$$
\mathbb {E} _ {S} \left[ \mathcal {R} _ {\mathrm{aRIP}} (S) \right] = \mathbb {E} _ {S} \left[ \mathbb {E} _ {\mathbf {n} _ {S}} \left[ \frac {\| D _ {S} ^ {\intercal} D _ {S} \mathbf {n} _ {S} \| ^ {2}}{| S |} \right] \right] \tag {7}
$$

Ideally, we would compute (7) for all possible input pairs $\mathbf { x } _ { 1 } , \mathbf { x } _ { 2 }$ by setting $S = S _ { 1 } \cup S _ { 2 }$ where $S _ { i }$ is the set of nonzero indices for the sparse code $\mathbf { z } _ { i } = \mathbf { f } ( \mathbf { x } _ { i } )$ . However, this would be prohibitively expensive, so we make some approximations for computational tractability. Intuitively, for index sets of identical size, if their spans $D _ { S _ { 1 } }$ and $D _ { S _ { 2 } }$ are orthogonal to one another, then $\mathcal { R } _ { \mathrm { a R I P } } ( S )$ is equal to the sum of the individual terms $\mathcal { R } _ { \mathrm { a R I P } } ( S _ { 1 } )$ and $\mathcal { R } _ { \mathrm { a R I P } } ( S _ { 2 } )$ . On the other hand, if the spans are oblique to one another, $\mathcal { R } _ { \mathrm { a R I P } } ( S )$ depends on the interaction between the two sparse supports, and can be much larger than the sum of the two.

Motivated by this, we leverage the amortized encoder f in (1) to determine the sets S to quantify. In particular, given the unsparsified code $\tilde { \mathbf { z } } _ { i } ~ \in ~ \mathbb { R } ^ { K }$ for a given input $\mathbf { x } _ { i } ,$ w e define $\bar { S }$ to be the indices of the largest 2k coefficients of z˜ in magnitude, where k is the sparsity level. Because the encoder (2) is near-linear, this tends to have the effect of “picking” a sparse index set $T$ of size k such that its span is maximally parallel to that of $S _ { i }$ , and setting $S = S _ { i } \cup T$ . This ensures we select “maximally interactive” pairs of supports, which are most likely to be problematic.

Given the relatively low sparsity levels k we’re interested in, a single sample is enough to estimate the expectation (7). Indeed, given the simplicity of equation (7), we will see that we can even use ${ \mathcal { R } } _ { \mathrm { a R I P } }$ as a regularizer to obtain dictionaries that better satisfy the approximate RIP condition. The following theorem shows that this leaves us with a form of near-identifiability of the dictionary and sparse codes of the SAE, under mild additional assumptions on the sparse code distribution, whenever reconstruction error is low.

Theorem 3.6. Consider two SAEs of the form (1), optimized with $\mathcal { L } _ { \boldsymbol { \theta } } ( \mathbf { x } ) = \| \mathbf { x } - \hat { \mathbf { x } } \| ^ { 2 }$ such that they achieve reconstruction error $\mathcal { L } _ { \boldsymbol { \theta } } ( \mathbf { x } ) \leq \epsilon$ almost everywhere on X , satisfying aRIP (Definition 3.3) at level δ with respect to the pushforward of the data distribution by the encoder, and satisfying Assumptions 3.4 and 3.5. Then, if the dictionaries of the two SAEs are D and D′ respectively with codes z and ${ \mathbf z } ^ { \prime } ,$ we have

$$
\left\| D - D ^ {\prime} \Pi \right\| \leq \epsilon_ {D} \tag {8}
$$

$$
\left\| \mathbf {z} - \Pi \mathbf {z} ^ {\prime} \right\| \leq \epsilon_ {z} \tag {9}
$$

for some signed permutation matrix Π $\in \{ - 1 , 0 , 1 \} ^ { K \times K }$ and error terms ϵD and $\epsilon _ { z }$ which are functions of ϵ, ρ, k and properties of the data distribution $P ( \mathbf { x } )$ , where the second inequality holds almost surely. Furthermore, $\epsilon _ { D } , \epsilon _ { z } \xrightarrow { \epsilon , \rho  0 }$ 0.

Our theorem shows that if an SAE yields dictionaries which satisfy the aRIP condition on unions of sparse supports from independent pairs of samples, its sparse codes are nearly identifiable up to permutations, meaning the same sparse codes (and therefore the same dictionaries) have been recovered. The level of nearness is governed by the reconstruction error and the level of aRIP regularization achieved by the solutions. The proof is given in Appendix A.

# 3.4. Encoder expressiveness

The identifiability theory in the previous section assumes an encoder that can yield sparse codes with low reconstruction error for the given dictionary. Importantly, this means the encoder must be sufficiently expressive to capture the mapping x 7→ z of observations to sparse codes. However, existing architectures for SAEs use a near-linear encoder, with the only source of non-linearity being a sparsifier. Early work on neural sparse coding (Gregor & LeCun, 2010) found such encoders to be insufficiently expressive, even when the dictionary is fixed, an observation also noted with SAEs (Donhauser et al., 2025). Gregor & LeCun (2010) propose a multistep encoder architecture motivated by the iterative soft thresholding algorithm to resolve this. In our implementation, each step of the encoder takes the form:

$$
\mathbf {z} ^ {(i)} = \operatorname{AbsTopK} _ {k} (S \mathbf {z} ^ {(i - 1)} + W (\mathbf {x} - \mathbf {b})) \tag {10}
$$

where $\begin{array} { r } { \mathbf { z } ^ { ( 0 ) } = \mathbf { 0 } , } \end{array}$ , so the only additional parameter is the (learned) step size matrix S. We replace the near-linear encoder from equation (2) with $\mathbf { f } ( \mathbf { x } ) { \overline { { \mathbf { \alpha } } } } = \mathbf { z } ^ { ( T ) }$ , where T is the number of iterations. In practice, we use $T = 5$ in all experiments, although the results don’t seem particularly sensitive to choices between 3 and 10.

# 4. Experiments

In our experiments, we evaluate the role of bidirectional features (section 3.2), dictionary conditioning (section 3.3) Table 1. Performance and identifiability metrics on synthetic data. Model variants shown are: TopK; AbsTopK (bidirectional features); iSAE (bidirectional features + aRIP regularization); and iSAE-ME (bidirectional features + aRIP regularization + multistep encoding). MSE = mean squared reconstruction error, $\mathcal { R } _ { S } = \mathrm { a R } \hat { \mathrm { I P } }$ measurement, DCS = dictionary cosine similarity, IoU = intersection over union, $\ell _ { 2 } = \ell _ { 2 }$ error.

<table><tr><td rowspan="3">Model</td><td rowspan="3">MSE</td><td rowspan="3"> $\mathcal{R}_{S}$ </td><td colspan="5">Pairwise Identifiability</td></tr><tr><td colspan="2">SAE</td><td colspan="2">Oracle</td><td rowspan="2">DCS</td></tr><tr><td>IoU</td><td> $\ell_2$ </td><td>IoU</td><td> $\ell_2$ </td></tr><tr><td colspan="8">i.i.d.</td></tr><tr><td>TopK</td><td>0.447</td><td>0.098</td><td>0.113</td><td>0.882</td><td>0.172</td><td>0.786</td><td>0.606</td></tr><tr><td>AbsTopK</td><td>0.318</td><td>0.106</td><td>0.487</td><td>0.557</td><td>0.914</td><td>0.176</td><td>0.935</td></tr><tr><td>iSAE</td><td>0.318</td><td>0.101</td><td>0.487</td><td>0.558</td><td>0.913</td><td>0.177</td><td>0.933</td></tr><tr><td>iSAE-ME</td><td>0.001</td><td>0.102</td><td>0.997</td><td>0.020</td><td>0.997</td><td>0.012</td><td>0.999</td></tr><tr><td colspan="8">mixture</td></tr><tr><td>TopK</td><td>0.357</td><td>0.120</td><td>0.149</td><td>0.840</td><td>0.250</td><td>0.758</td><td>0.633</td></tr><tr><td>AbsTopK</td><td>0.303</td><td>0.229</td><td>0.393</td><td>0.637</td><td>0.675</td><td>0.427</td><td>0.804</td></tr><tr><td>iSAE</td><td>0.319</td><td>0.095</td><td>0.497</td><td>0.526</td><td>0.706</td><td>0.389</td><td>0.843</td></tr><tr><td>iSAE-ME</td><td>0.040</td><td>0.230</td><td>0.873</td><td>0.219</td><td>0.877</td><td>0.215</td><td>0.932</td></tr></table>

and encoder expressiveness (section 3.4) in the performance of the SAE. The main goal is to understand the impact of dictionary conditioning and design decisions on reconstruction performance and the level of empirical identifiability. To assess empirical identifiability, we fit multiple SAEs with identical hyperparameters to the same data but with different seeds for initialization and training.

Metrics We measure performance by the usual mean squared error of the reconstructions $\| \mathbf { x } - { \hat { \mathbf { x } } } \| ^ { 2 }$ . For each pair of SAEs, identifiability is measured along two axes: dictionary identifiability and code identifiability. We match the concepts in the pair of learned dictionaries according to their cosine similarity via the Hungarian algorithm (Song et al., 2025). The mean absolute dictionary cosine similarity (DCS) after matching is reported. The same matching is used to align the sparse codes z (taking care to align signs as well), and the intersection-over-union (IoU) in the matched sparsity patterns, along with the $\ell _ { 2 }$ error (normalized by the mean norm of the codes z) is reported.

Sparse coding oracle As we are also interested in studying the relationship between dictionary conditioning, dictionary identifiability, and sparse code identifiability, we also explore the use of an oracle solver to obtain sparse codes from the learned dictionary. Specifically, we apply orthogonal matching pursuit (OMP; Pati et al. (1993)) to the learned dictionary from each of the SAEs, and assess whether it recovers the same sparse codes by employing the same identifiability metrics described above (IoU and $\ell _ { 2 }$ error).

Settings We consider three settings. The first two are synthetic. We generate 768-dimensional data noiselessly from a “true” dictionary consisting of 4096 concepts, which are unit vectors generated uniformly either at random (the i.i.d. case) or from a mixture of Gaussians projected onto the unit sphere (the mixture case). The codes are simulated as i.i.d. unit Gaussian random variables, sparsified to sparsity level k by zeroing out all but the k largest elements in magnitude. To assess the performance of our models in the real world, we evaluate on activations from layer 12 of Pythia-160M (Biderman et al., 2023) on The Pile (Gao et al., 2020). These activations are 768-dimensional, and we train SAEs with a dictionary size of 4096 concepts and a sparsity level of k = 40, the same size as our synthetic experiments. We also assess the models on patch tokens from DINOv2-Base (Oquab et al., 2024) using ImageNet-1k (Deng et al., 2009). These tokens and the SAE have the same dimensionality (768-dimensional activations, 4096 concepts, k = 40).

Training recipe During the course of our experiments, we found that both the performance and stability of SAEs are highly dependent on the training recipe. We use the standard training recipe from Gao et al. (2024) for all of our experiments, detailed here:

• Normalization. The inputs and dictionary atoms (concepts) are normalized to have unit norm.   
• Tied intercept. The pre- and post-bias in equation (1) are tied to be the same parameter, i.e. b = b′.   
• Initialization. The intercept is initialized to the geometric median of the data. The encoder is initialized to the approximate left inverse of the decoder.   
• Auxiliary loss. We employ an auxiliary loss to prevent dead concepts.

In all experiments, these components of the model and training procedure are exactly the same for all models compared, including the TopK SAE baseline. Furthermore, all models are always trained for the same number of training steps (512M total tokens for the LLM model in batches of 2048, 40K steps for the synthetic model with a batch size of 2048). The synthetic, LLM and vision training schemes are all “online”, in the sense that no tokens are repeated during training. For LLM and vision training in particular, a buffer of 500K tokens is inferred, and batches are drawn at random from this buffer, which is replenished when it’s half-empty.

# 4.1. Bidirectional features

As described in section 3.2, we implement the AbsTopK activation function as an alternative to the more standard TopK activation function. This allows negative concept loadings in the sparse codes, and effectively doubles the capacity of the dictionary at the same number of parameters. Results: synthetic In both the i.i.d. and mixture synthetic cases (Table 1), transitioning TopK → AbsTopK substantially improves reconstruction error, by around 30-40%. With this improvement in reconstruction comes a substantial improvement in the identifiability of the dictionary, from essentially unidentified (a DCS of 0.6 corresponds to an average angle of over 45 degrees between concept loadings, which is significant in 768-dimensional space) to well but not perfectly identified. Importantly, these gains in dictionary identifiability can be realized by a good sparse coding algorithm: the OMP oracle achieves excellent identifiability using this dictionary, even in the mixture case which is clearly more challenging due to the coherence in the data-generating dictionary. However, the usual encoder seemingly cannot realize these gains, because the SAE has much worse code identifiability than the oracle.

Table 2. Performance and identifiability metrics on LLM & vision activations. Model variants shown are: TopK; AbsTopK (bidirectional features); iSAE (bidirectional features + aRIP regularization); and iSAE-ME (bidirectional features + aRIP regularization + multistep encoding). MSE = mean squared reconstruction error, $\mathcal { R } _ { S } = \mathrm { a R I P }$ measurement, DCS = dictionary cosine similarity, IoU = intersection over union, $\ell _ { 2 } = \ell _ { 2 }$ error. 

<table><tr><td rowspan="3">Model</td><td rowspan="3">MSE</td><td rowspan="3"> $\mathcal{R}_{S}$ </td><td colspan="5">Pairwise Identifiability</td></tr><tr><td colspan="2">SAE</td><td colspan="2">Oracle</td><td rowspan="2">DCS</td></tr><tr><td>IoU</td><td> $\ell_2$ </td><td>IoU</td><td> $\ell_2$ </td></tr><tr><td colspan="8">Pythia-160M</td></tr><tr><td>TopK</td><td>0.166</td><td>0.183</td><td>0.398</td><td>0.492</td><td>0.350</td><td>0.530</td><td>0.813</td></tr><tr><td>AbsTopK</td><td>0.177</td><td>0.183</td><td>0.631</td><td>0.212</td><td>0.398</td><td>0.292</td><td>0.873</td></tr><tr><td>iSAE</td><td>0.179</td><td>0.114</td><td>0.526</td><td>0.256</td><td>0.472</td><td>0.273</td><td>0.858</td></tr><tr><td>iSAE-ME</td><td>0.148</td><td>0.132</td><td>0.375</td><td>0.297</td><td>0.343</td><td>0.316</td><td>0.797</td></tr><tr><td colspan="8">DINOv2</td></tr><tr><td>TopK</td><td>0.217</td><td>0.163</td><td>0.519</td><td>0.442</td><td>0.320</td><td>0.576</td><td>0.836</td></tr><tr><td>AbsTopK</td><td>0.235</td><td>0.159</td><td>0.524</td><td>0.424</td><td>0.374</td><td>0.462</td><td>0.862</td></tr><tr><td>iSAE</td><td>0.237</td><td>0.130</td><td>0.585</td><td>0.372</td><td>0.386</td><td>0.449</td><td>0.846</td></tr><tr><td>iSAE-ME</td><td>0.191</td><td>0.137</td><td>0.356</td><td>0.451</td><td>0.305</td><td>0.494</td><td>0.807</td></tr></table>

Results: LLM & vision The results on Pythia-160M and DINOv2 activations are similar to one another, but different from the results in synthetic data. AbsTopK makes for worse reconstruction error than TopK in both models, a result which is not consistent with prior work on AbsTopK, reflecting potential differences in the training procedure (Zhu et al., 2025). However, in both settings AbsTopK models do have the highest dictionary stability and correspondingly the oracle identifiability metrics are improved relative to TopK. Unlike in the synthetic regime, however, the SAE is better able to exploit dictionary stability for improved code stability, with higher IoU and better $\ell _ { 2 }$ error than the oracle. This suggests that dictionary stability as measured by DCS is at least partly orthogonal to dictionary conditioning, because the oracle should always be able to exploit a (globally) well-conditioned dictionary at least as well as the SAE.

Table 3. Downstream performance of the resulting SAEs trained on activations from Pythia-160M, as evaluated by SAEBench. CE Loss measures how well the SAE reproduces activations such that the LLM’s loss is not excessively inflated (lower is better). Sparse Probing accuracy measures how well the SAE recovers pre-specified concepts (higher better). Spurious correlation removal (SCR) tests whether spurious correlations can be removed from a downstream supervised probe by zeroing the confounding latent. Targeted probe perturbation (TPP) assesses selectivity in the concepts by zeroing a latent and assessing the impact on other supervised probes. 

<table><tr><td>Model</td><td>CE Loss</td><td>Sparse Probing</td><td>SCR</td><td>TPP</td></tr><tr><td>TopK</td><td>3.974</td><td>0.909</td><td>0.381</td><td>0.043</td></tr><tr><td>AbsTopK</td><td>4.004</td><td>0.906</td><td>0.330</td><td>0.031</td></tr><tr><td>iSAE</td><td>4.003</td><td>0.907</td><td>0.348</td><td>0.043</td></tr><tr><td>iSAE-ME</td><td>3.906</td><td>0.908</td><td>0.293</td><td>0.044</td></tr></table>

# 4.2. Dictionary conditioning

Next, we explore the role of dictionary conditioning, by considering the addition of the regularization term (7) to the model (AbsTopK → iSAE). For all experiments using the regularization term (including in the subsequent section), we fix the weight of the term to 10−1.

Results: synthetic In synthetic data, dictionaries learned using the standard training recipe with the AbsTopK activation function as in the previous section satisfy the aRIP condition fairly well (Table 1). Furthermore, the dictionaries are stable, and the corresponding oracle identifiability is good, particularly in the i.i.d. case. In the mixture case, the addition of aRIP regularization is effective in improving the estimated aRIP constant of the dictionary, and confers a marginal improvement in the code identifiability of the oracle but not the SAE.

Results: LLM & vision When trained on LLM activations, the results are much clearer. Identifiability of the oracle is improved substantially, with substantial gains in code IoU and improvements in $\ell _ { 2 }$ error. This occurs despite the slight drop in dictionary stability in both settings, suggesting that dictionary conditioning is the mechanism by which this improved stability occurs. However, these gains are only realized by the linear encoder in DINOv2, and the identifiability gains of the SAE are marginal.

# 4.3. Encoder expressiveness

Finally, we explore the multi-step encoding scheme proposed by Gregor & LeCun (2010) described in section 3.4 (iSAE → iSAE-ME). This introduces a step size parameter S as shown in equation (10), but otherwise does not modify the model.

Results: synthetic In the synthetic regime, adding the multistep encoder is crucial to realizing the identifiability gains from the improved dictionary conditioning in the SAE. In particular, even when bidirectional features and dictionary conditioning (previous sections) are enough to improve the quality of the dictionary and therefore the oracle solver, the default near-linear encoder cannot properly amortize codes near the true solution. On the other hand, iSAE-ME learns a nearly perfectly stable model in the synthetic regime as a result of its improved expressivity.

Results: LLM & vision On the other hand, in Pythia-160M and DINOv2 activations, the expressive encoder actually worsens identifiability of both the codes and the dictionary. This is in spite of the fact that it attains the lowest reconstruction error on LLM activations we report in this paper, with a substantial improvement over all models including baselines. This combination of facts suggests that the primary remaining barrier to identifiability for iSAE-ME is one of optimization, and is not a fundamental limitation of the expressivity of the encoder architecture.

# 4.4. SAEBench: Pythia-160M

When SAEs are used to interact with LLM activations, reconstruction performance and statistical identifiability are only proxies for desirable behaviours such as correctly identifying human-interpretable concepts (Karvonen et al., 2025). We evaluate the two forms of identifiable SAEs (iSAE and iSAE-ME) and compare them to TopK and AbsTopK baselines in terms of how well they minimize impact on the LLM loss, sparse probing performance, and targeted ablation performance (Table 3). iSAE-ME has the best performance in terms of the cross-entropy loss, consistent with its superior reconstruction performance (Table 2). On the other hand, the TopK baseline has the best sparse probing performance, although all models perform reasonably well and the difference appears marginal. Spurious correlation removal (SCR) ablates concepts by setting them to zero, which perhaps biases this task in favour of TopK models where zero has a clear interpretation as “absence” of a concept. Indeed, the TopK baseline performs well on this task, although iSAE-ME outperforms on target probe perturbation (TPP), which checks whether this ablation hampers the performance of other supervised probes. iSAE consistently outperforms AbsTopK, despite its marginally worse reconstruction performance and identifiability in Pythia-160M, suggesting adherence to the aRIP condition is perhaps related to downstream task performance.

# 5. Discussion

In this paper, we have defined a theoretical notion of statistical identifiability that applies to sparse autoencoders.

Specifically, a sparse autoencoder is near-identifiable if its training procedure is “stable”: trained on the same distribution, it should yield (nearly) the same concept dictionary and amortized sparse codes, every time. Further, we have assessed whether the commonly used TopK model is identifiable according to this definition, finding that it is not, and showing that improvements to the model in the form of architecture and regularization can greatly improve the near-identifiability and performance of the model. As a result, we present two variants of an SAE that are more identifiable on synthetic and some real-world activations, including one variant which achieves a massive reduction in reconstruction error due to the improved expressivity of its encoder.

SAEs are increasingly being used to interpret and interact with the large-scale neural networks used in practice, such as large language models. In this setting, identifiability of the sparse autoencoder is a “bare minimum” requirement for certain applications. For example, mechanistic interpretability (MI) aims to uncover the “true” hidden workings of the model, assuming that there is a “true mechanism”. So, if two trained SAEs uncover two different candidate mechanisms in the form of concepts, they surely cannot both be correct. Furthermore, many approaches for model oversight (Li et al., 2026) and model steering (O’Brien et al., 2025) rely on concept identification, rendering SAE identifiability paramount (Cywinski & Deja ´ , 2025).

Our experiments highlight that quantifying identifiability empirically in this setting is not trivial. In particular, we are the first to directly measure the stability of the sparse codes, proposing to do so using the intersection-over-union metrics on the sparsity patterns and the ℓ2 error after aligning using the dictionary. We show that dictionary identifiability, as measured by cosine similarity (Song et al., 2025), is not sufficient to characterize identifiability of the sparse codes. We attribute this to the conditioning of the dictionary: even if the dictionary can be learned stably, this does not render the corresponding sparse coding problem identifiable, as made clear in our impossibility result Theorem 3.2.

Our theory then aims to measure and optimize for dictionary conditioning, in the hopes of improving the identifiability of the sparse codes. To do this, we relax the restricted isometry property (RIP) condition from sparse coding and dictionary learning (Candes et al., 2005) to an approximate form, rendering the condition measurable. Our identifiability result Theorem 3.6 leverages this approximate RIP condition to prove near-identifiability of the dictionary and sparse codes. We emphasize that this of course does not break the NPhardness of checking the RIP condition on the dictionary, and therefore we cannot claim that SAEs satisfying our aRIP condition correctly recover a particular ground-truth datagenerating process, nor that their identifiability properties will generalize outside of the distribution they’re trained on.

Experiments show marked benefit to improving the dictionary conditioning. We employed orthogonal matching pursuit (OMP) as an oracle solver just to show that sometimes, the amortized encoder is the bottleneck to learning good sparse codes, even when the dictionary is well-conditioned and learned fairly stably. In the synthetic regime, using a more expressive encoder allows the amortized encoder of the SAE to “catch up” to the oracle. On the other hand, in real-world activations, the oracle is uniformly weaker than the SAE encoder, suggesting that optimization dynamics when training SAEs on real activations are substantially more complex. Notably, the more expressive encoder does allow for massive gains in reconstruction performance, and so we find it interesting to report here.

Importantly, all the modifications that constitute our model scale extremely well. We successfully trained both iSAE and iSAE-ME to learn concept dictionaries of size 4K (4096) on real-world LLM activations (dimension 768). Training times are reasonable: the proposed aRIP regularization term adds negligible overhead, while the multistep encoder increases training times by about 10-15% over a standard TopK SAE. It takes about 5 hours to train iSAE-ME on 512M activations from Pythia-160M, including inference time of the LLM, compared to about 4 hours for the TopK baseline.

Limitations Although we are not the first to propose evaluating SAEs and their stability in the synthetic regime (Song et al., 2025), our experiments highlight that there remain gaps between synthetic data-generating processes and realworld activations from practical neural networks. It is of interest to develop more realistic synthetic data-generating processes that facilitate faster model development and the study of failure modes of these models. Furthermore, we experimented only with TopK SAEs. Given that the multistep encoding scheme we adapted from LISTA (Gregor & LeCun, 2010) is built for ReLU-style activation functions, it would be interesting to see how it impacts identifiability and performance in ReLU SAEs as well. This is of particular interest given the conflicting results of the performance of the AbsTopK activation function we present in Section 3.2.

Finally, although our experiments show a massive improvement in code stability, it is still far from perfect in the real settings we consider here. We hypothesize that the remaining gap is largely due to optimization, although we emphasize that despite the large size of the models we train in this paper, it is also possible that they lack the capacity to adequately represent the observation distribution. As noted in Song et al. (2025), this can be a barrier to identifiability, and therefore a larger-scale study of identifiability across model sizes might be warranted.

# Acknowledgements

This work was supported by the Chan Zuckerberg Initiative (CZI) through the AI Residency Program. We thank CZI for the opportunity to participate in this program and the CZI AI Infrastructure Team for support with the GPU cluster used to train our models.

# Impact Statement

The primary application of our improvements to sparse autoencoders is to render highly “black-box” models more interpretable, generally regarded as a good property for responsible use of artificial intelligence. However, interpretability is remarkably difficult to evaluate in practice, and even when achieved does not automatically lead to more responsible use. Accordingly, care must be taken to ensure that users of SAEs do not overstate their reliability or epistemic capability.

# References

Aharon, M., Elad, M., and Bruckstein, A. K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation. IEEE Transactions on Signal Processing, 54(11):4311–4322, January 2006. doi: 10. 1109/TSP.2006.881199.   
Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., and van der Wal, O. Pythia: A suite for analyzing large language models across training and scaling, 2023. URL https://arxiv.org/abs/2304.01373.   
Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.   
Bussmann, B., Leask, P., and Nanda, N. Batchtopk sparse autoencoders, 2024. URL https://arxiv.org/ abs/2412.06410.   
Candes, E., Romberg, J., and Tao, T. Stable signal recovery from incomplete and inaccurate measurements, 2005. URL https://arxiv.org/abs/math/ 0503066.   
Chen, S. and Donoho, D. Basis pursuit. In Proceedings of 1994 28th Asilomar Conference on Signals, Systems

and Computers, volume 1, pp. 41–44 vol.1, 1994. doi: 10.1109/ACSSC.1994.471413.

Chen, S., Billings, S. A., and Luo, W. Orthogonal least squares methods and their application to non-linear system identification. International Journal of control, 50 (5):1873–1896, 1989.

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models, 2023. URL https:// arxiv.org/abs/2309.08600.

Cywinski, B. and Deja, K. SAeUron: Interpretable con- ´ cept unlearning in diffusion models with sparse autoencoders. In Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., and Zhu, J. (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 11738–11775. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr. press/v267/cywinski25a.html.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Donhauser, K., Ulicna, K., Moran, G. E., Ravuri, A., Kenyon-Dean, K., Eastwood, C., and Hartford, J. Towards scientific discovery with dictionary learning: Extracting biological concepts from microscopy foundation models. In Forty-second International Conference on Machine Learning, 2025.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., and Olah, C. Toy models of superposition, 2022. URL https://arxiv.org/ abs/2209.10652.

Fel, T., Lubana, E. S., Prince, J. S., Kowal, M., Boutin, V., Papadimitriou, I., Wang, B., Wattenberg, M., Ba, D., and Konkle, T. Archetypal sae: Adaptive and stable dictionary learning for concept extraction in large vision models, 2025. URL https://arxiv.org/abs/2502. 12892.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. The pile: An 800gb dataset of diverse text for language modeling, 2020. URL https: //arxiv.org/abs/2101.00027.

Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders, 2024. URL https: //arxiv.org/abs/2406.04093.   
Gregor, K. and LeCun, Y. Learning fast approximations of sparse coding. In Proceedings of the 27th International Conference on International Conference on Machine Learning, ICML’10, pp. 399–406, Madison, WI, USA, 2010. Omnipress. ISBN 9781605589077.   
Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., Chanin, D., Lau, Y.-T., Farrell, E., McDougall, C., Ayonrinde, K., Till, D., Wearden, M., Conmy, A., Marks, S., and Nanda, N. Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability, 2025. URL https://arxiv.org/abs/2503. 09532.   
Lee, Y., Yoon, S., Son, M., and Park, F. C. Regularized autoencoders for isometric representation learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=mQxt8l7JL04.   
Li, A. J., Srinivas, S., Bhalla, U., and Lakkaraju, H. Evaluating adversarial robustness of concept representations in sparse autoencoders, 2026. URL https://arxiv. org/abs/2505.16004.   
Locatello, F., Bauer, S., Lucic, M., Raetsch, G., Gelly, S., Scholkopf, B., and Bachem, O. Challenging common ¨ assumptions in the unsupervised learning of disentangled representations. In international conference on machine learning, pp. 4114–4124. PMLR, 2019.   
Mairal, J., Bach, F., Ponce, J., and Sapiro, G. Online dictionary learning for sparse coding. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML ’09, pp. 689–696, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605585161. doi: 10.1145/ 1553374.1553463. URL https://doi.org/10. 1145/1553374.1553463.   
Makhzani, A. and Frey, B. J. k-sparse autoencoders. CoRR, abs/1312.5663, 2013. URL https://api. semanticscholar.org/CorpusID:14850799.   
Mencattini, T., Cadei, R., and Locatello, F. Exploratory causal inference in saence. International Conference on Learning Representations, 2026.   
Nelson, W., Fumero, M., Karaletsos, T., and Locatello, F. Statistical and structural identifiability in representation learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https:// openreview.net/forum?id=Wa3cfE3Iay.

O’Brien, K., Majercak, D., Fernandes, X., Edgar, R., Bullwinkel, B., Chen, J., Nori, H., Carignan, D., Horvitz, E., and Poursabzi-Sangdeh, F. Steering language model refusal with sparse autoencoders, 2025. URL https: //arxiv.org/abs/2411.11296.   
Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S. Zoom in: An introduction to circuits. Distill, 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https:// openreview.net/forum?id=a68SUt6zFt. Featured Certification.   
Pati, Y., Rezaiifar, R., and Krishnaprasad, P. Orthogonal matching pursuit: recursive function approximation with applications to wavelet decomposition. In Proceedings of 27th Asilomar Conference on Signals, Systems and Computers, pp. 40–44 vol.1, 1993. doi: 10.1109/ACSSC. 1993.342465.   
Paulo, G. and Belrose, N. Sparse autoencoders trained on the same data learn different features, 2025. URL https://arxiv.org/abs/2501.16615.   
Song, X., Muhamed, A., Zheng, Y., Kong, L., Tang, Z., Diab, M. T., Smith, V., and Zhang, K. Position: Mechanistic interpretability should prioritize feature consistency in saes, 2025. URL https://arxiv.org/abs/2505. 20254.   
Spielman, D. A., Wang, H., and Wright, J. Exact recovery of sparsely-used dictionaries. In Mannor, S., Srebro, N., and Williamson, R. C. (eds.), Proceedings of the 25th Annual Conference on Learning Theory, volume 23 of Proceedings of Machine Learning Research, pp. 37.1– 37.18, Edinburgh, Scotland, 25–27 Jun 2012. PMLR. URL https://proceedings.mlr.press/v23/ spielman12.html.   
Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. Steering language models with activation engineering, 2024. URL https: //arxiv.org/abs/2308.10248.   
Zhu, X., Khalili, M. M., and Zhu, Z. Abstopk: Rethinking sparse autoencoders for bidirectional features, 2025. URL https://arxiv.org/abs/2510.00404.

# A. Proofs

Notation We define an SAE as an encoder-decoder pair $( f , D )$ where $D \in \mathbb { R } ^ { N \times K }$ is a dictionary with unit-norm columns. For subspaces $\mathcal { U } , \mathcal { V } \subset \mathbb { R } ^ { N }$ , we denote the minimum and maximum principal angles between them $\theta _ { \operatorname* { m i n } } ( \mathcal { U } , \mathcal { V } )$ and $\theta _ { \mathrm { m a x } } ( \mathcal { U } , \mathcal { V } )$ respectively. For the sparse support of a code $\mathbf { z } , S \subset [ n ]$ , we denote the local RIP constant $\delta _ { S } = \| D _ { S } ^ { T } D _ { S } - I \| _ { \mathrm { o p } } ,$ , also a random variable.

We begin by proving our impossibility result, Theorem 3.2. The key idea is that there exist dictionaries which conceivably could approximate a particular distribution $P ( \mathbf { x } )$ well, but do not uniquely identify the corresponding sparse codes. We emphasize that this is a well-known result in sparse coding which in fact motivates much of that literature (Candes et al., 2005). For ease of reading, the formal statement of the theorem wordlessly converts the “every concept is activated with positive probability” assumption to an assumption that two explicitly constructed problematic concepts are activated with positive probability.

Theorem (3.2, formal). Fix integers N, K and a sparsity level k such that $2 \leq k \leq \operatorname* { m i n } \{ N , K - 2 \}$ . There exists a normalized dictionary $D \in \mathbb { R } ^ { N \times \mathbf { \bar { K } } }$ and two supports $S , S ^ { \prime } \subset [ K ]$ with $| S | = | S ^ { \prime } | = k$ and $| S \cap S ^ { \prime } | = k - 2$ such that the following holds.

Let X be the support of an observation distribution $P ( \mathbf { x } )$ . Let $f : \mathcal { X } \to \mathbb { R } ^ { K }$ be a continuous encoder such that $f ( \mathbf { x } )$ has at most k nonzeros for every $\mathbf { x } \in \mathcal { X }$ and

$$
\left\| D f (\mathbf {x}) - \mathbf {x} \right\| \leq \epsilon
$$

for every $\mathbf { x } \in \mathcal { X }$ . Then there exists another sparse autoencoder $( f ^ { \prime } , D )$ with a continuous encoder $f ^ { \prime } : \mathcal { X } \to \mathbb { R } ^ { K }$ such that $f ^ { \prime } ( \mathbf { x } )$ has at most k nonzeros for every $\mathbf { x } \in \mathcal { X }$ and

$$
\left\| D f ^ {\prime} (\mathbf {x}) - \mathbf {x} \right\| \leq \epsilon
$$

for every $\mathbf { x } \in \mathcal { X }$ . Further, if there exists $\mathbf { x } _ { 0 } \in \mathcal { X }$ such that $D f ( \mathbf { x } _ { 0 } )$ has a nonzero component along span $\{ e _ { k - 1 } , e _ { k } \}$ , then there exists $\mathbf { x } _ { 1 } \in { \mathcal { X } }$ such that $f ( \mathbf { x } _ { 1 } ) \neq f ^ { \prime } ( \mathbf { x } _ { 1 } )$ and the sparse supports of $f ( \mathbf { x } _ { \mathrm { 1 } } )$ ) and $f ^ { \prime } ( \mathbf { x } _ { \mathrm { 1 } } )$ are different.

Proof. Fix $\varepsilon > 0$ . Let $e _ { 1 } , \ldots , e _ { N }$ denote the standard basis.

Define the dictionary D by specifying its columns. For $1 \leq j \leq k$ set $D _ { j } = e _ { j }$ . Define two additional unit vectors in the plane spanned by $e _ { k - 1 }$ and $\textstyle e _ { k } .$

$$
D _ {k + 1} = \frac {e _ {k - 1} + \varepsilon e _ {k}}{\sqrt {1 + \varepsilon^ {2}}}
$$

and

$$
D _ {k + 2} = \frac {\varepsilon e _ {k - 1} + e _ {k}}{\sqrt {1 + \varepsilon^ {2}}}.
$$

For $j \in \{ k + 3 , \ldots , K \}$ choose any unit vectors $D _ { j }$ in $\operatorname { s p a n } \{ e _ { 1 } , \dots , e _ { k } \}$ . Every column has unit norm, so D is normalized. Define the two supports

$$
S = \{1, 2, \dots , k \}
$$

and

$$
S ^ {\prime} = \{1, 2, \dots , k - 2, k + 1, k + 2 \}.
$$

Then $| S | = | S ^ { \prime } | = k$ and $| S \cap S ^ { \prime } | = k - 2$ .

Denote the k-dimensional subspace by ${ \mathcal { U } } = \operatorname { s p a n } \{ e _ { 1 } , \dots , e _ { k } \}$ . Since every column of D lies in U, we have $D \mathbf { z } \in \mathcal { U }$ for every z ∈ RK . $\mathbf { z } \in \mathbb { R } ^ { K }$

Claim. There exist continuous maps $g _ { S } : \mathcal { U } \to \mathbb { R } ^ { K }$ and $g _ { S ^ { \prime } } : \mathcal { U } \to \mathbb { R } ^ { K }$ such that for every $\mathbf { u } \in \mathcal { U } .$ , the vector $g _ { S } ( \mathbf { u } )$ is supported on S, the vector $g _ { S ^ { \prime } } ( \mathbf { u } )$ is supported on $S ^ { \prime }$ , and

$$
D g _ {S} (\mathbf {u}) = \mathbf {u}
$$

and

$$
D g _ {S ^ {\prime}} (\mathbf {u}) = \mathbf {u}.
$$

Proof of claim. Write any $\mathbf { u } \in \mathcal { U }$ uniquely as $\textstyle \mathbf { u } = \sum _ { j = 1 } ^ { k } a _ { j } e _ { j }$ . Define $g _ { S } ( \mathbf { u } ) \in \mathbb { R } ^ { K }$ by setting $( g _ { S } ( \mathbf { u } ) ) _ { j } = a _ { j }$ for $j \in S$ and $( g _ { S } ( \mathbf { u } ) ) _ { j } = 0$ for $j \not \in S$ . Then $g _ { S } ( \mathbf { u } )$ is supported on S and $\begin{array} { r } { D g _ { S } ( \mathbf { u } ) = \sum _ { j = 1 } ^ { k } a _ { j } D _ { j } = \sum _ { j = 1 } ^ { k } a _ { j } e _ { j } = \mathbf { u } } \end{array}$ . Continuity is immediate.

Next, define $g _ { S ^ { \prime } } ( \mathbf { u } ) \in \mathbb { R } ^ { K }$ as follows. For $j \in \{ 1 , \dots , k - 2 \}$ set $( g _ { S ^ { \prime } } ( \mathbf { u } ) ) _ { j } = a _ { j }$ . For $j \notin S ^ { \prime }$ set $( g _ { S ^ { \prime } } ( \mathbf { u } ) ) _ { j } = 0$ . It remains to set the two coordinates $( g _ { S ^ { \prime } } ( \mathbf { u } ) ) _ { k + 1 }$ and $( g _ { S ^ { \prime } } ( \mathbf { u } ) ) _ { k + 2 }$ .

Let $\mathbf { b } = ( a _ { k - 1 } , a _ { k } ) ^ { T } \in \mathbb { R } ^ { 2 }$ . Let $M \in \mathbb { R } ^ { 2 \times 2 }$ denote the matrix whose columns are the coordinates of $D _ { k + 1 }$ and $D _ { k + 2 }$ in the basis $( e _ { k - 1 } , e _ { k } )$ . Then

$$
M = \frac {1}{\sqrt {1 + \varepsilon^ {2}}} \left[ \begin{array}{c c} 1 & \varepsilon \\ \varepsilon & 1 \end{array} \right].
$$

This matrix is invertible since its determinant equals $( 1 - \varepsilon ^ { 2 } ) / ( 1 + \varepsilon ^ { 2 } )$ , which is nonzero for $\varepsilon \neq 1$ . Define

$$
\left[ \begin{array}{c} (g _ {S ^ {\prime}} (\mathbf {u})) _ {k + 1} \\ (g _ {S ^ {\prime}} (\mathbf {u})) _ {k + 2} \end{array} \right] = M ^ {- 1} \mathbf {b}.
$$

Then by construction we have

$$
(g _ {S ^ {\prime}} (\mathbf {u})) _ {k + 1} D _ {k + 1} + (g _ {S ^ {\prime}} (\mathbf {u})) _ {k + 2} D _ {k + 2} = a _ {k - 1} e _ {k - 1} + a _ {k} e _ {k}.
$$

Therefore $D g _ { S ^ { \prime } } ( \mathbf { u } ) = \mathbf { u }$ . Since $g _ { S ^ { \prime } }$ is linear in u, it is continuous. This proves the claim.

Now return to the assumed encoder f. Define the reconstruction map

$$
\mathbf {u} (\mathbf {x}) = D f (\mathbf {x}).
$$

By the remark above, $\mathbf { u } ( \mathbf { x } ) \in \mathcal { U }$ for all $\mathbf { x } \in \mathcal { X }$ . Define two candidate encoders on $\mathcal { X }$ by

$$
f _ {S} (\mathbf {x}) = g _ {S} (\mathbf {u} (\mathbf {x}))
$$

and

$$
f _ {S ^ {\prime}} (\mathbf {x}) = g _ {S ^ {\prime}} (\mathbf {u} (\mathbf {x})).
$$

Each map is continuous as a composition of continuous maps. Also, $f _ { S } ( \mathbf { x } )$ is supported on $S$ and $f _ { S ^ { \prime } } ( \mathbf { x } )$ is supported on $S ^ { \prime }$ for every $\mathbf { x } \in \mathcal { X } .$ , so both satisfy the k-sparsity constraint.

Further, for every $\mathbf { x } \in \mathcal { X }$ we have

$$
D f _ {S} (\mathbf {x}) = D g _ {S} (\mathbf {u} (\mathbf {x})) = \mathbf {u} (\mathbf {x}) = D f (\mathbf {x})
$$

and similarly

$$
D f _ {S ^ {\prime}} (\mathbf {x}) = D g _ {S ^ {\prime}} (\mathbf {u} (\mathbf {x})) = \mathbf {u} (\mathbf {x}) = D f (\mathbf {x}).
$$

Therefore both satisfy the same reconstruction error bound as $f .$ For example,

$$
\left\| D f _ {S} (\mathbf {x}) - \mathbf {x} \right\| = \left\| D f (\mathbf {x}) - \mathbf {x} \right\| \leq \epsilon ,
$$

and the same holds for $f _ { S ^ { \prime } }$ . Set $f ^ { \prime }$ to be either $f _ { S }$ or $f _ { S ^ { \prime } }$ , which proves the first part of the lemma.

It remains to show the disagreement and different supports under the additional assumption. Suppose there exists $\mathbf { x } _ { 0 } \in \mathcal { X }$ such that ${ \bf u } ( { \bf x } _ { 0 } )$ has a nonzero component along span $\{ e _ { k - 1 } , e _ { k } \}$ . Writing $\textstyle \mathbf { u } ( \mathbf { x } _ { 0 } ) = \sum _ { j = 1 } ^ { k } a _ { j } e _ { j }$ , this means $( a _ { k - 1 } , a _ { k } ) \neq$ $( 0 , 0 )$ . Then $g _ { S } ( \mathbf { u } ( \mathbf { x } _ { 0 } ) )$ has a nonzero entry on at least one of the indices $k - 1$ or $k ,$ while $g _ { S ^ { \prime } } ( \mathbf { u } ( \mathbf { x } _ { 0 } ) )$ has a nonzero entry on at least one of the indices $k + 1$ or $k + 2$ . In particular, $g _ { S } ( \mathbf { u } ( \mathbf { x } _ { 0 } ) ) \neq g _ { S ^ { \prime } } ( \mathbf { u } ( \mathbf { x } _ { 0 } ) )$ and their supports are different.

If $f$ differs from $f _ { S }$ at some point, choose $f ^ { \prime } = f _ { S }$ and set $\mathbf { x } _ { 1 }$ to be any point where they differ. Otherwise $f$ equals $f _ { S }$ everywhere, and then choosing $f ^ { \prime } = f _ { S ^ { \prime } }$ and ${ \bf x } _ { 1 } = { \bf x } _ { 0 }$ gives $f ( \mathbf { x } _ { 1 } ) \neq f ^ { \prime } ( \mathbf { x } _ { 1 } )$ with different supports. This completes the proof. □

Now, we prove our identifiability result. We enumerate all assumptions used in our construction.

Assumptions Below, we enumerate all assumptions used in the lemmas and proofs. We assume both SAEs in the pair follow these assumptions.

(A1) Approximate RIP. For observed supports S and $S ^ { \prime } , \delta _ { S \cup S ^ { \prime } } \leq \delta$ for some $\delta < 1$ . Intuitively, this means that RIP holds on the observed union of supports for any two inputs.   
(A2) Sufficient richness of supports. For any $i \in [ K ]$ , we observe supports $S$ and $S ^ { \prime }$ such that $S \cap S ^ { \prime } = \{ i \}$ . Furthermore, for any pair of concepts i and $j ,$ there exist disjoint supports $S$ and $S ^ { \prime }$ such that $i \in S$ and $j \in S ^ { \prime }$ . Intuitively, this means that no two concepts always co-occur.   
(A3) Bounded reconstruction error. Observations can be decomposed as $\mathbf { x } = D \mathbf { z } + \boldsymbol { \epsilon } = D \mathbf { z } ^ { \prime } + \boldsymbol { \epsilon } ^ { \prime }$ , and $\| \epsilon \| , \| \epsilon ^ { \prime } \| \le \epsilon <$ < ζ 1−δ α(δ)4√k . Intuitively, this means the reconstruction error of each SAE is strictly bounded everywhere by a constant $\frac { \zeta \sqrt { 1 - \delta } \alpha ( \delta ) } { 4 \sqrt { k } }$ which depends on how well the aRIP constraint holds, how well the sufficient diversity constraint holds, and the sparsity level.   
(A4) Sufficient diversity of observations. For any observed support $S$ of size $k ,$ we have $\mathbf { C o v } [ \mathbf { Z } _ { S } \mid S ]$ is positive definite, and furthermore there exists a batch of k samples from $\mathbf { Z } _ { S } \mid S$ such that the $k \times k$ matrix of samples has smallest singular value larger than $\zeta .$   
(A5) Bounded data distribution. The observation distribution X has bounded support, that is, there exists $B > 0$ such that $\| \mathbf { X } \| \leq B$ everywhere.   
(A6) Least-squares optimality. For each observation $\begin{array} { r } { \mathbf { x } = D \mathbf { z } + \epsilon , } \end{array}$ , we assume the codes z are least-squares optimal on the support set S.

Our first lemma shows that when $2 k { \mathrm { - a R } } \mathbf { I } \mathbf { P }$ holds on any observed combination of supports, this forcibly separates the subspaces spanned by those supports. In the language of Figure 1, the span of any two distinct patches must be decomposable into its intersection and two approximately “unique” subspaces which are well-separated. On the other hand, if these two patches share no concepts, the patches themselves must be well-separated.

Lemma A.1. Let $S$ and $S ^ { \prime }$ be sparse supports of size k such that $| S \cap S ^ { \prime } | < k .$ Suppose $\delta _ { S \cup S ^ { \prime } } \leq \delta < 1$ . Then, there exists $\alpha ( \delta ) > 0$ such that sin $\theta _ { m a x } ( \mathcal { U } _ { S } , \mathcal { U } _ { S ^ { \prime } } ) \geq \alpha ( \delta )$ where $\mathcal { U } _ { S } = s p a n ( D _ { S } )$ and $\mathcal { U } _ { S ^ { \prime } } = s p a n ( D _ { S ^ { \prime } } )$ . Further, $i f \vert S \cap S ^ { \prime } \vert = 0 ,$ , we have sin $\theta _ { m i n } ( \mathcal { U } _ { S } , \mathcal { U } _ { S ^ { \prime } } ) \geq \alpha ( \delta )$ .

Proof. Denote the intersection by $I = S \cap S ^ { \prime }$ . Denote the isolated components of each support by $A = S \backslash I$ and $B = { \cal { S } } ^ { \prime } \backslash I .$

Claim. $\mathcal { U } _ { S } \cap \mathcal { U } _ { S ^ { \prime } } = \mathcal { U } _ { I }$

Proof of claim. First, note that RIP implies that $D _ { S } , D _ { S ^ { \prime } }$ and $D _ { S \cup S ^ { \prime } }$ ′ all have full column rank. Let $\mathbf { x } \in \mathcal { U } _ { S } \cap \mathcal { U } _ { S ^ { \prime } }$ , noting that then there exist decompositions $\mathbf { x } = D _ { S } \mathbf { z } = D _ { S ^ { \prime } } \mathbf { z } ^ { \prime }$ . As a result, we have:

$$
\mathbf {0} = D _ {S} \mathbf {z} - D _ {S ^ {\prime}} \mathbf {z} ^ {\prime} = D _ {I} (\mathbf {z} _ {I} - \mathbf {z} _ {I} ^ {\prime}) + D _ {A} \mathbf {z} _ {A} - D _ {B} \mathbf {z} _ {B} = D _ {S \cup S ^ {\prime}} \mathbf {w}
$$

where w stacks the individual components, and must be zero by the fact that $D _ { S \cup S ^ { \prime } }$ has full column rank. Thus ${ \bf z } _ { I } = { \bf z } _ { I } ^ { \prime }$ and ${ \bf x } = D _ { I } { \bf z } _ { I }$ , proving the claim.

Denote $\mathcal { U } _ { 1 } = \mathcal { U } _ { S } \cap \mathcal { U } _ { I } ^ { \perp }$ and $\mathcal { U } _ { 1 } ^ { \prime } = \mathcal { U } _ { S ^ { \prime } } \cap \mathcal { U } _ { I } ^ { \perp }$ .

Claim. $\theta _ { \operatorname* { m a x } } ( \mathcal { U } _ { S } , \mathcal { U } _ { S ^ { \prime } } ) \geq \theta _ { \operatorname* { m i n } } ( \mathcal { U } _ { 1 } , \mathcal { U } _ { 1 } ^ { \prime } )$ .

Proof of claim. We have $\mathcal { U } _ { S } = \mathcal { U } _ { 1 } \oplus \mathcal { U } _ { I } ^ { \perp }$ and $\mathcal { U } _ { S ^ { \prime } } = \mathcal { U } _ { 1 } ^ { \prime } \oplus \mathcal { U } _ { I } ^ { \bot }$ with dim $( \mathcal { U } _ { 1 } ) = \dim ( \mathcal { U } _ { 1 } ^ { \prime } ) = k - | I |$ . This means the nonzero principal angles between $\mathcal { U } _ { S }$ and $\mathcal { U } _ { S ^ { \prime } }$ are exactly the principal angles between $\mathcal { U } _ { 1 }$ and $\mathcal { U } _ { 1 } ^ { \prime }$ . In particular, we have the claim.

Denote by $P _ { I }$ the orthogonal projector onto $\mathcal { U } _ { I }$ .

Claim. The projected dictionary $( I - P _ { I } ) D _ { A \cup B }$ satisfies RIP at the same level δ.

Proof of claim. Let $\mathbf { z } _ { A \cup B } \ \in \ \mathbb { R } ^ { | A \cup B | }$ denote an arbitrary vector and let $\mathbf { z } _ { I }$ denote the least-squares minimizer of $\| D _ { A \cup B } \mathbf { z } _ { A \cup B } - D _ { I } \mathbf { z } _ { I } \| _ { 2 }$ . Let $\mathbf { r } \ = \ D _ { A \cup B } \mathbf { z } _ { A \cup B } - D _ { I } \mathbf { z } _ { I } \ = \ ( I - P _ { I } ) D _ { A \cup B } \mathbf { z } _ { A \cup B } \ \in \ \mathcal { U } _ { I } ^ { \perp }$ be the residual. Stacking $\mathbf { z } _ { A \cup B }$ and −zI into $\mathbf { w } \in \mathbb { R } ^ { | S \cup S ^ { \prime } | }$ , we have $D _ { S \cup S ^ { \prime } } \mathbf { w } = D _ { A \cup B } \mathbf { z } _ { A \cup B } - D _ { I } \mathbf { z } _ { I } = \mathbf { r }$ . Now, we have:

$$
\| (I - P _ {I}) D _ {A \cup B} \mathbf {z} _ {A \cup B} \| ^ {2} = \| \mathbf {r} \| ^ {2} = \| D _ {S \cup S ^ {\prime}} \mathbf {w} \| ^ {2} \geq (1 - \delta) \| \mathbf {w} \| ^ {2} \geq (1 - \delta) \| \mathbf {z} _ {A \cup B} \| ^ {2}
$$

and also by the fact that $\delta _ { A \cup B } \leq \delta _ { S \cup S ^ { \prime } } \colon$

$$
\| (I - P _ {I}) D _ {A \cup B} \mathbf {z} _ {A \cup B} \| ^ {2} \leq \| D _ {A \cup B} \mathbf {z} _ {A \cup B} \| ^ {2} \leq (1 + \delta) \| \mathbf {z} _ {A \cup B} \| ^ {2}
$$

which yields the claim by the arbitrariness of $\mathbf { z } _ { A \cup B }$ .

Claim. sin $\theta _ { \mathrm { m i n } } ( \mathcal { U } _ { 1 } , \mathcal { U } _ { 1 } ^ { \prime } ) \geq \alpha ( \delta )$ for $\begin{array} { r } { \alpha ( \delta ) = \sqrt { 1 - \left( \frac { 2 \delta } { 1 + \delta } \right) ^ { 2 } } . } \end{array}$

Proof of claim. Let $\mathbf { u } _ { S }$ and $\mathbf { u } _ { S ^ { \prime } }$ be unit vectors realizing the minimum principal angle $\langle { \bf u } _ { S } , { \bf u } _ { S ^ { \prime } } \rangle = \cos \theta _ { \mathrm { m i n } } ( \mathcal { U } _ { S } , \mathcal { U } _ { S ^ { \prime } } )$ . Then ${ \bf u } _ { S } = ( I - P _ { I } ) D _ { S } { \bf z } _ { S }$ and ${ \bf u } _ { S ^ { \prime } } = ( I - P _ { I } ) D _ { S ^ { \prime } } { \bf z } _ { S ^ { \prime } }$ . Then,

$$
\| \mathbf {u} _ {S} - \mathbf {u} _ {S ^ {\prime}} \| ^ {2} \geq (1 - \delta) (\| \mathbf {z} _ {S} \| ^ {2} + \| \mathbf {z} _ {S ^ {\prime}} \| ^ {2}) \geq 2 (1 - \delta) / (1 + \delta)
$$

where the first inequality follows by hypothesis and the second from the previous claim. Rearranging the usual definition of cosine similarity for unit vectors, we have cos ${ \theta } _ { \mathrm { { m i n } } } ( { \mathcal { U } } _ { S } , { \mathcal { U } } _ { S ^ { \prime } } ) \leq 2 \delta / ( 1 + \delta )$ ). By the Pythagorean identity, we have the claim.

Using the second claim, the first statement of the lemma follows. If $S \cap S ^ { \prime } = \emptyset .$ , then $\ d \mathcal { U } _ { 1 } = \ d \mathcal { U } _ { S }$ and $\mathcal { U } _ { 1 } ^ { \prime } = \mathcal { U } _ { S ^ { \prime } }$ so the latter statement of the lemma follows. □

Our second lemma shows that a given support in the first SAE must in some sense map uniquely to a single support in the second ${ \mathrm { S A E } } ,$ where uniqueness is defined in the same way as in the separation argument in the previous lemma. In the language of Figure 1, this lemma shows that if patches are well-separated in a pair of SAEs, low reconstruction error is enough to “adhere” a particular linear patch to a region of the observation manifold, and this linear patch can’t look too different from the linear patch approximating that region of the manifold in the second SAE.

Lemma A.2. Let $\mathbf { x } ^ { ( i ) } = D \mathbf { z } ^ { ( i ) } + \epsilon ^ { ( i ) }$ denote k distinct k-sparse decompositions with sparse support S under the first SAE satisfying (A4), with smallest singular value bounded below by ζ. Further, assume that these observations also admit a decomposition under the second SAE, $\mathbf { x } ^ { ( i ) } = D ^ { \prime } \mathbf { z } ^ { \prime ( i ) } + \epsilon ^ { \prime ( i ) }$ , with sparse support $T ^ { ( i ) }$ with $| T ^ { ( i ) } | = k$ . Let $\tau _ { S }$ be the set of supports that occur in the second SAE, and suppose

$$
\sin \theta_ {m a x} (\mathcal {V} _ {T}, \mathcal {V} _ {T ^ {\prime}}) \geq \beta > 0
$$

for all distinct $\begin{array} { r } { T , T ^ { \prime } \in \mathcal { T } . \mathrm { ~ } I f \epsilon < \frac { \sqrt { 1 - \delta } \zeta \beta } { 4 \sqrt { k } } } \end{array}$ , then all supports coincide: $T ^ { ( 1 ) } = \cdot \cdot \cdot = T ^ { ( k ) } = T$ for some T . Furthermore,

$$
\sin \theta_ {m a x} (\mathcal {U} _ {S}, \mathcal {V} _ {T}) \leq \frac {2 \sqrt {k}}{\sqrt {1 - \delta} \zeta} \epsilon
$$

and for every $T ^ { \prime } \neq T$ in $\tau _ { S }$ ,

$$
\sin \theta_ {m a x} (\mathcal {U} _ {S}, \mathcal {V} _ {T ^ {\prime}}) \geq \beta / 2
$$

Proof. Denote the reconstructions under the first $\mathtt { S A E }$ as $\hat { \mathbf { x } } ^ { ( i ) } = D _ { S } \mathbf { z } _ { S } ^ { ( i ) }$ . Our first claim is that if all $T ^ { ( i ) }$ are actually a single support $T ,$ we have that $\mathcal { U } _ { S }$ is near $\nu _ { T }$ .

Claim. Suppose $T ^ { ( 1 ) } = \cdot \cdot \cdot = T ^ { ( k ) } = T$ for some support T . Then

$$
\sin \theta_ {\mathrm{max}} (\mathcal {U} _ {S}, \mathcal {V} _ {T}) \leq \frac {2 \sqrt {k}}{\zeta \sqrt {1 - \delta}} \epsilon
$$

Proof of claim. Applying the triangle inequality, we have the tube constraint that the reconstruction under the first $\mathrm { S A E }$ cannot be far from $\mathcal { V } _ { T } \colon \| ( I - P _ { T } ) \hat { \mathbf { x } } ^ { ( i ) } \| \le 2 \epsilon$ . Stacking the reconstructions into a matrix ${ \hat { X } } .$ , we have $\| ( I - P _ { T } ) \hat { X } \| _ { \mathrm { o p } } \leq$ $\| ( I - P _ { T } ) \hat { X } \| _ { F } \leq 2 \epsilon \sqrt { k }$ . Let Q be an orthonormal basis matrix for US . Write ${ \hat { X } } = Q R$ , where R is invertible by the fact that $D _ { S }$ is full rank and (A4), and furthermore $\sigma _ { \operatorname* { m i n } } ( \hat { X } ) \geq \sigma _ { \operatorname* { m i n } } ( D _ { S } Z ) \geq \sigma _ { \operatorname* { m i n } } ( D _ { S } ) \sigma _ { \operatorname* { m i n } } ( Z ) \geq \zeta \sqrt { 1 - \delta }$ , thus

$$
\sin \theta_ {\mathrm{max}} (\mathcal {U} _ {S}, \mathcal {V} _ {T}) = \| (I - P _ {T}) Q \| _ {\mathrm{op}} \leq \| (I - P _ {T}) \hat {X} \| _ {\mathrm{op}} \| R ^ {- 1} \| _ {\mathrm{op}} \leq \frac {2 \epsilon \sqrt {k}}{\zeta \sqrt {1 - \delta}}
$$

where the denominator in the final inequality follows from the fact that $\sigma _ { \mathrm { m i n } } ( \hat { X } ) = 1 / \sigma _ { \mathrm { m i n } } ( R ^ { - 1 } )$ .

Next, we consider the case where the supports in the second SAE do not coincide.

Claim. $T ^ { ( 1 ) } = \cdot \cdot \cdot = T ^ { ( k ) } = T$ for some support T .

Proof of claim. Consider toward a contradiction that $T ^ { ( i ) } \neq T ^ { ( j ) }$ for some i and $j .$ . Noting that the argument in the previous claim only hinges on the projection operator $P _ { T ^ { ( i ) } }$ ) , we can apply the same argument to $T ^ { ( i ) }$ and $T ^ { ( j ) }$ independently, combining the results to obtain

$$
\sin \theta_ {\max} (\mathcal {V} _ {T ^ {(i)}}, \mathcal {V} _ {T ^ {(j)}}) \leq \sin \theta_ {\max} (\mathcal {U} _ {S}, \mathcal {V} _ {T ^ {(i)}}) + \sin \theta_ {\max} (\mathcal {U} _ {S}, \mathcal {V} _ {T ^ {(j)}}) \leq \frac {4 \epsilon \sqrt {k}}{\zeta \sqrt {1 - \delta}}
$$

which contradicts the hypotheses sin $\theta _ { \operatorname* { m a x } } \bigl ( \mathcal { V } _ { T ^ { ( i ) } } , \mathcal { V } _ { T ^ { ( j ) } } \bigr ) \geq \beta$ and $\begin{array} { r } { \epsilon < \frac { \sqrt { 1 - \delta } \zeta \beta } { 4 \sqrt { k } } } \end{array}$ .

Together, these claims give the main conclusion. The second conclusion follows from the fact that for any other $T ^ { \prime }$ , we have

$$
\sin \theta_ {\mathrm{max}} (\mathcal {U} _ {S}, \mathcal {V} _ {T ^ {\prime}}) \geq \sin \theta_ {\mathrm{max}} (\mathcal {U} _ {S}, \mathcal {V} _ {T}) - \sin \theta_ {\mathrm{max}} (\mathcal {V} _ {T}, \mathcal {V} _ {T ^ {\prime}}) \geq \beta - \frac {2 \epsilon \sqrt {k}}{\zeta \sqrt {1 - \delta}} \geq \beta / 2
$$

where the final bound follows from the hypothesis on ϵ.

With these two lemmas in hand, we can prove our identifiability theorem.

Theorem (3.6, formal). Let X denote a random observation, and consider a pair of trained sparse autoencoders $( f , D )$ and $( f ^ { \prime } , D ^ { \prime } )$ . Denote the sparse codes $\mathbf { Z } = f ( \mathbf { X } )$ and $\mathbf { Z } ^ { \prime } = f ^ { \prime } ( \mathbf { X } )$ , and suppose we have $\mathbf { X } = D \mathbf { Z } + \pmb { \epsilon } = D ^ { \prime } \mathbf { Z } ^ { \prime } + \pmb { \epsilon } ^ { \prime }$ where $\| \epsilon \| , \| \epsilon ^ { \prime } \| \le \epsilon$ for ϵ. Furthermore, suppose that the approximate RIP assumption (A1) is satisfied for both models, that the observed support distributions in both models are sufficiently rich to isolate concepts (A2), that the reconstruction error in both models is bounded (A3), and the observation distribution is bounded (A5) sufficiently diverse to witness all dimensions of each latent support (A4). Finally, assume that the models are sufficiently trained such that the sparse codes satisfy the least-squares solution on their supports (A6). Then, there exists a signed permutation π (or Π, in matrix form) such that we have:

1. Dictionary near-identifiability. $\| \mathbf { d } _ { i } - \mathbf { d } _ { \pi ( i ) } ^ { \prime } \| \leq 2 C _ { 1 } ( \delta ) \eta \mathrm { f o r } C _ { 1 } ( \delta ) = 1 + 2 / \alpha ( \delta )$   
2. Code near-identifiability. $\begin{array} { r } { \| \mathbf { Z } - \Pi \mathbf { Z } ^ { \prime } \| \leq \frac { 2 \epsilon } { \sqrt { 1 - \delta } } + \frac { ( 2 \sqrt { k } C _ { 1 } ( \delta ) \eta ) ( B + \epsilon ) } { 1 - \delta } } \end{array}$

where $\begin{array} { r } { \eta = \frac { 2 \sqrt { k } } { \zeta \sqrt { 1 - \delta } } \epsilon \ \mathrm { a n d } \alpha ( \delta ) = \sqrt { 1 - \Big ( \frac { 2 \delta } { 1 + \delta } \Big ) ^ { 2 } } . } \end{array}$

Proof. We begin by showing that for a given support in the first SAE, there is only one sparse support in second SAE that can accurately represent the observations from that support.

Claim. For a sparse support $S$ in the first SAE, we have a well-defined support map $T = \Phi ( S )$ where T is a sparse support in the second SAE. Furthermore, we have sin θmax(US, VT ) ≤ 2 kζ√1−δϵ $\begin{array} { r } { \theta _ { \operatorname* { m a x } } ( \mathcal { U } _ { S } , \mathcal { V } _ { T } ) \le \frac { 2 \sqrt { k } } { \zeta \sqrt { 1 - \delta } \epsilon } = : \eta . } \end{array}$ .

Proof of claim. Using (A4), select a batch of k observations $\mathbf { x } ^ { ( i ) }$ with the support S such that their coefficient matrix Z satisfies $\sigma _ { \operatorname* { m i n } } ( Z ) \geq \zeta$ . Note that each observation also admits a k-sparse decomposition with appropriately bounded error in the second SAE, supported on $T ^ { ( i ) }$ . By Lemma A.1, we have that for $T \neq T ^ { \prime } \in \mathcal { T } _ { S }$ , the set of these supports, sin $\begin{array} { r } { \theta _ { \operatorname* { m a x } } ( \mathcal { V } _ { T } , \mathcal { V } _ { T ^ { \prime } } ) \geq \sqrt { 1 - \left( \frac { 2 \delta } { 1 + \delta } \right) ^ { 2 } } = : \alpha ( \delta ) } \end{array}$ . By the fact that $\begin{array} { r } { \epsilon < \frac { \zeta \sqrt { 1 - \delta } \alpha ( \delta ) } { 4 \sqrt { k } } } \end{array}$ , Lemma A.2 implies that all the supports coincide: $T : = T ^ { ( 1 ) } = \cdot \cdot \cdot T ^ { ( k ) }$ , along with the bound. From here on, we set $T = \Phi ( S )$ .

Next, we use this support-by-support mapping to show the existence of a concept-by-concept mapping between the two SAEs. Pick a concept i in the first SAE. By the sufficient richness assumption (A2), we can select two supports $S _ { 1 }$ and $S _ { 2 }$ such that $S _ { 1 } \cap S _ { 2 } = \{ i \}$ . Let $T _ { 1 } = \Phi ( S _ { 1 } )$ and $T _ { 2 } = \Phi ( S _ { 2 } )$ be the corresponding supports in the second SAE as given in the previous claim, satisfying sin $\theta _ { \operatorname* { m a x } } ( \mathcal { U } _ { S _ { \ell } } , \mathcal { V } _ { T _ { \ell } } ) \leq \eta .$

Now, we need a technical claim which bounds the distance between the two supports in the same SAE in terms of the distance from the intersection.

Claim. For every $\hat { \mathbf { x } } \in \mathcal { U } _ { S _ { 1 } }$ , we have $\begin{array} { r } { \| ( I - P _ { S _ { 2 } } ) \hat { \mathbf { x } } \| \geq \alpha \| ( I - P _ { I } ) \hat { \mathbf { x } } \| , } \end{array}$

Proof of claim. Denote ${ \mathcal { U } } _ { I } = { \mathrm { s p a n } } ( \mathbf { d } _ { i } )$ , and define the residual spaces $\mathcal { U } _ { 1 } = \mathcal { U } _ { S _ { 1 } } \cap \mathcal { U } _ { I } ^ { \perp }$ and $\mathcal { U } _ { 2 } = \mathcal { U } _ { S _ { 2 } } \cap \mathcal { U } _ { I } ^ { \perp }$ . By the second claim in Lemma A.1, we have sin $\theta _ { \mathrm { m i n } } ( \mathcal { U } _ { 1 } , \mathcal { U } _ { 2 } ) \geq \alpha ( \delta )$ . For any $\hat { \mathbf { x } } \in \mathcal { U } _ { S _ { 1 } }$ ,

$$
\| (I - P _ {S _ {2}}) \hat {\mathbf {x}} \| = \| (I - P _ {\mathcal {U} _ {2}}) (I - P _ {I}) \hat {\mathbf {x}} \| \geq \alpha (\delta) \| (I - P _ {I}) \hat {\mathbf {x}} \|
$$

Now, we study the other SAE. Our first goal is to show that like in the first SAE, these supports must share at least one concept.

Claim. $T _ { 1 } \cap T _ { 2 } \neq \emptyset$

Proof of claim. Assume toward a contradiction that $T _ { 1 } \cap T _ { 2 } = \emptyset$ . Note that Lemma A.1 yields sin $\theta _ { \operatorname* { m i n } } ( \mathcal { V } _ { T _ { 1 } } , \mathcal { V } _ { T _ { 2 } } ) \geq \alpha ( \delta )$ . Pick v in $\gamma _ { T _ { 1 } }$ such that it’s closest to $\mathbf { d } _ { i }$ , then by the triangle inequality we have

$$
\left\| \left(I - P _ {T _ {2}}\right) \mathbf {v} \right\| \leq \left\| \mathbf {v} - \mathbf {d} _ {i} \right\| + \left\| \left(I - P _ {T _ {2}}\right) \mathbf {d} _ {i} \right\| \leq 2 \eta
$$

which contradicts the minimum principal angle bound.

Now, we show that the corresponding supports in the other SAE share exactly one concept.

Claim. $| T _ { 1 } \cap T _ { 2 } | = 1$

Proof of claim. Assume toward a contradiction that $J = T _ { 1 } \cap T _ { 2 }$ satisfies $\left| J \right| \geq 2 .$ . Now, given $\mathcal { V } _ { \mathcal { I } } = \mathcal { V } _ { T _ { 1 } } \cap \mathcal { V } _ { T _ { 2 } }$ , pick a unit vector v in $\nu _ { \mathcal { I } }$ such that $\textbf { v } \perp \textbf { d } _ { i }$ . By the previous claim, we can pick ${ \mathbf { u } } _ { 1 } \in { \mathcal { U } } _ { S }$ 1 η-close to v and $\mathbf { u } _ { 2 } \in \mathcal { U } _ { S _ { 2 } }$ , so $\| \mathbf { u } _ { 1 } - \mathbf { u } _ { 2 } \| \leq \| \mathbf { u } _ { 1 } - \mathbf { v } \| + \| \mathbf { u } _ { 2 } - \mathbf { v } \| \leq 2 \eta$ by the triangle inequality.

Now, we have $\| ( I - P _ { I } ) \mathbf { v } \| = 1$ by the perpendicularity, and therefore $\| ( I - P _ { I } ) \mathbf { u } _ { 1 } \| \geq 1 - \eta$ . Thus, by our previous technical claim we have $\| ( I - P _ { S _ { 2 } } ) \mathbf { u } _ { 1 } \| \geq \alpha ( \delta ) ( 1 - \eta )$ . Combining, we have $2 \eta \ge \| \mathbf { u } _ { 1 } - \mathbf { u } _ { 2 } \| \ge \| ( I - P _ { S _ { 2 } } ) \mathbf { u } _ { 1 } \| \ge \alpha ( \delta ) ( 1 - \eta )$ which contradicts the ϵ bound. Therefore $| J | = 1$ .

Because in the first $\mathrm { { S A E } } \ S _ { 1 }$ and $S _ { 2 }$ share a single concept, and $T _ { 1 }$ and $T _ { 2 }$ also share a single concept, we posit a mapping between the two. However, we have to show that this mapping is well-defined, in the sense that a different pair of supports in the first SAE sharing the same concept maps to the same concept in the second SAE.

Claim. Defining $\pi ( i )$ as the unique element of $T _ { 1 } \cap T _ { 2 }$ gives a well-defined mapping $\pi : [ K ] \to [ K ]$ , independent of the particular choices of $S _ { 1 }$ and $S _ { 2 }$ .

Proof of claim. Take any alternative choice $\tilde { S } _ { 1 }$ and ${ \tilde { S } } _ { 2 } .$ , and apply the previous two claims to obtain $\tilde { T } _ { 1 } = \Phi ( \tilde { S } _ { 1 } )$ and $\tilde { T } _ { 2 } = \Phi ( \tilde { S } _ { 2 } )$ with $\tilde { T } _ { 1 } \cap \tilde { T } _ { 2 } = \{ \tilde { j } \}$ . Take $\mathbf { v } \in \gamma _ { T _ { 1 } }$ such that $\lVert \mathbf { v } - \mathbf { d } _ { i } \rVert \leq \eta$ , by Lemma $\mathrm { A } . 2$ applied to $\mathcal { U } _ { S _ { 1 } }$ and $\nu _ { T _ { 1 } }$ . Then, we have

$$
\left\| \left(I - P _ {T _ {2}}\right) \mathbf {v} \right\| \leq \left\| \left(I - P _ {T _ {2}}\right) \mathbf {d} _ {i} \right\| + \left\| \mathbf {v} - \mathbf {d} _ {i} \right\| \leq 2 \eta
$$

yielding $\| ( I - P _ { j } ) \mathbf { v } \| \leq 2 \eta / \alpha ( \delta )$ via Lemma A.1. Thus, we have

$$
\left\| \left(I - P _ {j}\right) \mathbf {d} _ {i} \right\| \leq \left\| \mathbf {d} _ {i} - \mathbf {v} \right\| + \left\| \left(I - P _ {j}\right) \mathbf {v} \right\| \leq (1 + 2 / \alpha (\delta)) \eta
$$

and a similar argument yields $\begin{array} { r } { \| ( I - P _ { \tilde { i } } ) \mathbf { d } _ { i } \| \le ( 1 + 2 / \alpha ( \delta ) ) \eta : = \rho . } \end{array}$ . Suppose now toward a contradiction that $j \neq \widetilde { j }$ Choosing signs s and $s ^ { \prime }$ appropriately, we have $\| \mathbf { d } _ { i } - s \mathbf { d } _ { j } ^ { \prime } \| \leq 2 \rho$ and $\| \mathbf { d } _ { i } - s \mathbf { d } _ { \widetilde { i } } ^ { \prime } \| \leq 2 \rho$ . Applying the triangle inequality gives $\langle \mathbf { d } _ { j } ^ { \prime } , \mathbf { d } _ { \tilde { i } } ^ { \prime } \rangle \geq 1 - 8 \rho ^ { 2 } , \mathrm { i . e . } \rho \geq \sqrt { ( 1 - \delta ) / 8 }$ by restricting the RIP condition (A1). This yields a contradiction given our ϵ bound.

As a result, we can define $\pi ( i ) = j$ . It remains to show that it’s a permutation, which covers the same edge case as in the previous claim but in reverse.

Claim. π is a permutation of $[ K ]$ .

Proof of claim. Suppose toward a contradiction π is not injective, with $\begin{array} { r } { \pi ( i ) = \pi ( i ^ { \prime } ) = j } \end{array}$ with $i \neq i ^ { \prime }$ for $i \in S$ and $i ^ { \prime } \in S ^ { \prime }$ . For $T = \Phi ( S )$ and $T ^ { \prime } = \Phi ( S ^ { \prime } )$ , we have sin $\theta _ { \mathrm { m i n } } ( \mathcal { V } _ { T } , \mathcal { V } _ { T ^ { \prime } } ) = 0$ by the fact that span $( \mathbf { d } _ { i } ^ { \prime } ) \subset \mathcal { V } _ { T } \cap \mathcal { V } _ { T }$ ′ . On the other hand, Lemma A.1 gives sin $\theta _ { \operatorname* { m i n } } ( \mathcal { U } _ { S } , \mathcal { U } _ { S ^ { \prime } } ) \geq \alpha ( \delta )$ . By the triangle inequality and the fact that sin $\theta _ { \operatorname* { m a x } } ( \mathcal { U } _ { S } , \mathcal { V } _ { T } ) \le \eta$ and sin $\theta _ { \mathrm { m a x } } ( \mathcal { U } _ { S ^ { \prime } } , \mathcal { V } _ { T ^ { \prime } } ) \leq \eta$ , we have sin $\theta _ { \operatorname* { m i n } } ( \mathcal { V } _ { T } , \mathcal { V } _ { T ^ { \prime } } ) \ : \geq \ : \alpha ( \delta ) - 2 \eta$ . Given our ϵ bound, this is a contradiction. Thus π is injective, and given that the domain and codomain are finite and of the same size, a permutation.

The hard part is done. Now, using the mapping π we constructed, we can show individual concepts in the dictionary are near-identifiable up to signs, according to the permutation π.

Claim. For either $s _ { i } = + 1$ or $s _ { i } = - 1$ , we have $\| \mathbf { d } _ { i } - s _ { i } \mathbf { d } _ { \pi ( i ) } ^ { \prime } \| \leq 2 C _ { 1 } ( \delta ) \eta \operatorname { f o r } C _ { 1 } ( \delta ) = 1 + 2 / \alpha ( \delta )$ .

Proof of claim. Fix i and pick $S _ { 1 }$ and $S _ { 2 }$ with $S _ { 1 } \cap S _ { 2 } = \{ i \}$ . Set $T _ { 1 } = \Phi ( S _ { 1 } )$ ) and $T _ { 2 } = \Phi ( S _ { 2 } )$ ), by the previous claims we have $T _ { 1 } \cap T _ { 2 } = \{ \pi ( i ) \}$ . For any $\mathbf { v } \in \gamma _ { T _ { 1 } }$ , we have

$$
\| (I - P _ {T _ {2}}) \mathbf {v} \| \geq \alpha (\delta) \| (I - P _ {\pi (i)}) \mathbf {v} \|
$$

where $P _ { \pi ( i ) }$ is the projection matrix onto $\mathscr { V } _ { \pi ( i ) } = \mathrm { s p a n } ( \mathbf { d } _ { \pi ( i ) } ^ { \prime } )$ . Select $\mathbf { w } \in \mathcal { V } _ { T _ { 1 } }$ such that $\| \mathbf { d } _ { i } - \mathbf { w } \| \leq \eta$ . By the triangle inequality, we have $\left\| ( I - P _ { T _ { 2 } } ) \mathbf { v } \right\| \leq \left\| ( I - P _ { T _ { 2 } } ) \mathbf { d } _ { i } \right\| + \left\| \mathbf { d } _ { i } - \mathbf { v } \right\| \leq 2 \eta$ . As a result, we have $\| ( I - P _ { \pi ( i ) } ) \mathbf { v } \| \leq 2 \eta / \alpha ( \delta )$ . Thus,

$$
\| (I - P _ {\pi (i)}) \mathbf {d} _ {i} \| \leq \| \mathbf {d} _ {i} - \mathbf {v} \| + \| (I - P _ {\pi (i)}) \mathbf {v} \| \leq (1 + 2 / \alpha) \eta
$$

with the final bound following by the definition of point-set distance in terms of sines.

Finally, support stability and dictionary stability yields code stability.

Claim. For any observation ${ \bf x } = D { \bf z } + \epsilon = D ^ { \prime } { \bf z } ^ { \prime } + \epsilon ^ { \prime } .$ , we have $\begin{array} { r } { \| \mathbf { z } - \Pi \mathbf { z } ^ { \prime } \| \leq \| \mathbf { z } - \Pi \mathbf { z } ^ { \prime } \| \leq \frac { 2 \epsilon } { \sqrt { 1 - \delta } } + \frac { ( 2 \sqrt { k } C _ { 1 } ( \delta ) \eta ) ( B + \epsilon ) } { 1 - \delta } = \mathcal { O } ( \epsilon ) } \end{array}$ (2√k C1(δ) η)(B+ϵ) = O(ϵ).

Proof of claim. Let S denote the support of $\mathbf { z } ,$ and T denote the support $\mathbf { z } ^ { \prime } .$ . We have,

$$
\begin{array}{l} \| \mathbf {z} - \Pi \mathbf {z} ^ {\prime} \| = \| \mathbf {z} _ {S} - \Pi_ {T} \mathbf {z} _ {T} ^ {\prime} \| \\ \leq \| \mathbf {z} _ {S} - D _ {S} ^ {\dagger} \hat {\mathbf {x}} _ {T} \| + \| D _ {S} ^ {\dagger} \hat {\mathbf {x}} _ {T} - \Pi_ {T} \mathbf {z} _ {T} ^ {\prime} \| \\ \end{array}
$$

where $D _ { S } ^ { \dagger }$ is the left inverse of $D _ { S }$ , well-conditioned by (A1). Denote by $\bar { D } _ { S } = D ^ { \prime } \Pi _ { S }$ be the sign- and permutation-matched dictionary from the second SAE. Then, we have $D _ { S } ^ { \dagger } \hat { \mathbf { x } } _ { T } - \Pi _ { T } \mathbf { z } _ { T } ^ { \prime } = D _ { S } ^ { \dagger } ( \bar { D } _ { S } - D _ { S } ) ( \Pi _ { S } \mathbf { z } _ { T } ^ { \prime } )$ . As a result,

$$
\begin{array}{l} \| \mathbf {z} - \Pi \mathbf {z} ^ {\prime} \| \leq \| \mathbf {z} _ {S} - D _ {S} ^ {\dagger} \hat {\mathbf {x}} _ {T} \| + \| D _ {S} ^ {\dagger} \hat {\mathbf {x}} _ {T} - \Pi_ {T} \mathbf {z} _ {T} ^ {\prime} \| \\ \leq \frac {2 \epsilon}{\sqrt {1 - \delta}} + \| D _ {S} ^ {\dagger} \| _ {\mathrm{op}} \| \bar {D} _ {S} - D _ {S} \| _ {\mathrm{op}} \| \Pi_ {S} \mathbf {z} _ {T} ^ {\prime} \| \\ \leq \frac {2 \epsilon}{\sqrt {1 - \delta}} + \frac {1}{\sqrt {1 - \delta}} (2 \sqrt {k} C _ {1} (\delta) \eta) \frac {B + \epsilon}{\sqrt {1 - \delta}} \\ = \frac {2 \epsilon}{\sqrt {1 - \delta}} + \frac {(2 \sqrt {k} C _ {1} (\delta) \eta) (B + \epsilon)}{1 - \delta} \\ \end{array}
$$

which is $\mathcal { O } ( \epsilon )$ , the same rate we would hope for if we knew the dictionaries and supports perfectly in advance. Furthermore, the remainder of the constants could be made tighter by more careful treatment of the second term. □