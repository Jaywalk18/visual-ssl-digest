# Similarity search generalisation in contrastive learning with InfoNCE loss

Nick Whiteley School of Mathematics, University of Bristol, U.K.

July 13, 2026

## Abstract

Similarity search is a primary application of embedding models trained by contrastive learning. For one of the most popular contrastive learning loss functions, InfoNCE, we show that the population risk with k negative samples is O(1/k) close to an expected cross-entropy which quantifies deviation between i) a softmax similarity search over unseen data using the learned embedding function, and ii) an idealised softmax search over the same data but using similarity implicitly represented in th positive sample generator. This complements existing interpretations of InfoNCE in the k → ∞ limit which are phrased in terms of mutual information, and alignment versus uniformity in embeddings. To quantify generalisation performance, we introduce a new continuity bound for the InfoNCE loss, obtained via Gˆateaux diferentiation. The bound preserves the structure of averaging over negative samples present in the loss function and features an “inverse temperature” parameter which can be tuned to account for the algorithmic temperature. For embedding functions which are Lipschitz in a parameter, this yields a simple demonstration that the averaging efect of k negative samples in the InfoNCE loss carries over to stabilisation of the generalisation error as k grows.

## 1 Introduction

The InfoNCE loss [van den Oord et al., 2018] is one of the most popular loss functions in contrastive learning and is a fundamental ingredient in hugely impactful systems such as SimCLR [Chen et al., 2020], MoCo [He et al., 2020] and CLIP [Radford et al., 2021]. Similarity search is a primary application of these technologies and other embedding models trained using contrastive learning; the learned embedding is used to calculate cosine similarities and hence evaluate closeness among unseen data. The goal of the present work is to add clarity to our theoretical understanding of InfoNCE, similarity search and generalisation.

To date, theoretical generalisation analysis of contrastive learning has largely focused on downstream classification. In that context, “generalisation” has the conventional meaning of a model’s ability to make accurate predictions on unseen data: a pioneering step forward was made by Saunshi et al. [2019], who showed that the InfoNCE population risk can contribute to bounding the population risk of a downstream linear classifier, thus quantifying classification accuracy when an unseen input data point, such as an image or text document, is represented by its embedding vector. Subsequent refinements and extensions, discussed in more detail later, have been made by [Lei et al., 2023, Ghanooni et al., 2024, Hieu and Ledent, 2025].

In the present work we also analyse InfoNCE risk but from a diferent perspective, which we call similarity search generalisation: the performance of similarity search on unseen data using the learned embedding, compared to an idealised search which we show is implicitly defined by the ingredients of contrastive learning. We draw inspiration not only from learning-theoretic analyses of e.g., [Saunsh et al., 2019, Lei et al., 2023, Ghanooni et al., 2024, Hieu and Ledent, 2025], but also from widely refer enced interpretations of contrastive learning in terms of mutual information [van den Oord et al., 2018], “alignment” versus “uniformity” in embeddings [Wang and Isola, 2020a], and cross-entropy between conditional probability densities [Zimmermann et al., 2021].

## 1.1 Interpretations of InfoNCE

Each training tuple or mini-batch in contrastive learning comprises “anchor”, “positive” and “negative” samples<sup>1</sup>. Pairs of anchor and positive samples are generated in order to convey some notion of semantic similarity which is specific to input data modality, such as text, images, etc. Negative samples are usually additional data points which are independent of the anchor and positive samples.

Conventional theoretical interpretations of contrastive learning with InfoNCE loss address the regime where the number of negative samples grows. The InfoNCE loss function was introduced by van den Oord et al. [2018] as the categorical cross-entropy associated with correctly classifying a positive sample versus negative samples. They showed that minimising InfoNCE risk maximises a lower bound on mutua information, and argued that bound becomes tight as the number of negative samples grows. Wang and Isola [2020a] uncovered another interpretation of InfoNCE associated with taking the number of i.i.d. negative samples $k  \infty$ They showed that, in this limit, the logarithmically normalised InfoNCE population risk converges to a limiting risk function which separates into a sum of two terms, arguing that minimising InfoNCE loss promotes a balance between “alignment” and “uniformity” of embeddings. The same $k \to \infty$ limit was given another interpretation by Zimmermann et al. [2021], as cross-entropy between conditional probability density functions, in a setting where the input samples are assumed to be drawn from a distribution defined by pushing the uniform density on a hypersphere or convex subset of Euclidean space through an invertible transformation. They argued that contrastive learning can invert that data generating model.

In all of these interpretations, the convergence of the logarithmically normalised InfoNCE population risk as $k  \infty$ can be viewed as a consequence of a law of large numbers for the negative samples. The risk converges because averaging over these negative samples occurs within the loss function. In the present work we explore yet another interpretation of the k → ∞ averaging, relating InfoNCE population risk to similarity search on unseen data.

## 1.2 Generalisation analysis of contrastive learning

In order to quantify downstream classification performance, Saunshi et al. [2019] and in turn Lei et al. [2023] assumed a specific data-generating model involving discrete classes and class labels, with positive samples generated by drawing from the same class-conditional distribution as the anchor sample. How ever, their complexity analysis of InfoNCE risk (rather than classification risk) depends very little on this model. To apply a standard generalisation bound [Mohri et al., 2018, Thm. 3.3] they only really require that training tuples are i.i.d., and their empirical complexity bounds hold for any realisation of the data hence do not require any distributional assumptions at all. We would like to exploit this fact in order to apply their results outside of the context of classification.

Considering the interest in the regime $k \to \infty$ described above, this naturally leads us to the question of how InfoNCE generalisation error behaves as k grows. As discussed by Lei et al. [2023], the complexity bounds of Saunshi et al. [2019] feature an explicit factor which grows with k (e.g., see the factor of $\sqrt { k }$ in [Saunshi et al., 2019][Supp. material, eq. (30)]), seemingly at odds with studies indicating that a large number of negative samples is necessary for good generalisation performance in practice [Chen et al., 2020, He et al., 2020, Henaf, 2020, Khosla et al., 2020, Tian et al., 2020].

Lei et al. [2023] revisited this generalisation analysis and used alternative mathematical techniques to obtain refined complexity bounds, improving over [Saunshi et al., 2019] by eliminating the explicit factor of $\sqrt { k }$ for $\ell _ { 2 } \cdot$ -Lipschitz loss functions and reducing by a further factor of $\sqrt { k }$ for $\ell _ { \infty } - \mathrm { I }$ Lipschitz losses (up to factors logarithmic in k and the number of training tuples n). Lei et al. [2023] also obtained data dependent bounds in the case of self-bounding loss functions. Nevertheless, the complexity estimates obtained by Lei et al. [2023] for $\ell _ { 2 }$ or $\ell _ { \infty }$ -Lipschitz loss functions still depend on k and may grow with k in general; they involve summing and/or maximising over an index set which grows with k (see the quantities denoted A and C in [Lei et al., 2023][eqs. 4.2 and 4.5]). The resulting bounds for linear and nonlinear features [Lei et al., 2023][sec. 5, quantity $B _ { x } ]$ involve the maximum of the norms of the input data vectors, which will grow with k in general when the input domain is unbounded. One of the aims of the present work is to clarify whether such dependence on k is avoidable.

In more recent developments, the assumption of i.i.d. data tuples made by Saunshi et al. [2019] and Lei et al. [2023] was loosened by Hieu and Ledent [2025], bringing the analysis closer to practica dependency between samples using U-statistics. Ghanooni et al. [2024] developed generalisation analysis for adversarial contrastive learning.

## 1.3 Outline and contributions

• Section 2 presents the basic ingredients of contrastive learning with the InfoNCE loss. Our setup is purposefully general in some ways: our measure-theoretic notation is chosen to help interpret the InfoNCE population loss in terms of Markov kernels and softmax similarity search. However, our presentation is purposefully narrow in other ways: our priority is to give the reader, in just a few pages, an end-to-end account from interpretation of the InfoNCE loss, to easily interpretable generalisation bounds which exhibit the role of k and other quantities. We thus introduce contrastive learning as obtaining an embedding function $\phi$ by empirical risk minimisation (although strict minimisation is not required for our generalisation results to apply), and do not enter into details of how specific neural network architectures, gradient algorithms, mini-batches etc., are used in practice.

• In section 3.1 we introduce a Markov kernel $Q _ { \tau } ^ { \phi }$ on the input space ${ \mathcal { Z } } ,$ depending on the embedding function $\phi$ and temperature parameter τ, and in proposition 1 in section 3.2 show that as $k  \infty ,$ the (logarithmically normalised) InfoNCE population risk converges to cross-entropy between $Q _ { \tau } ^ { \phi }$ and the Markov kernel M which generates positive samples: $Z ^ { + } \sim M ( Z , \cdot )$ where $Z \sim \pi _ { \mathrm { d a t a } }$ and π<sub>data</sub> is a probability measure on $\mathcal { Z }$ . This presentation is partly inspired by [Zimmermann et al., 2021, Thm 1.] but does not require the specifics of their data-generating model. It is already known that the population risk converges in this limit, Wang and Isola [2020b, Thm 1.] report a rate $O ( 1 / \sqrt { k } )$ ; we clarify in proposition 1 the rate is $O ( 1 / k )$ . We also highlight the regularising role of the temperature parameter: the higher $\tau \mathrm { i s }$ , the more $Q _ { \tau } ^ { \phi } ( z , \cdot )$ is constrained to be close to $\pi _ { \mathrm { d a t a } } .$ , uniformly in z and ϕ.

• In section 3.3 we present an interpretation of the InfoNCE population risk which, to the knowledge of the author, is new. We introduce an empirical softmax similarity Markov kernel $\widehat { Q } _ { \tau , k } ^ { \phi }$ defined in terms of k unseen $( { \mathrm { i . e . } }$ , independent of training data) draws from $\pi _ { \mathrm { d a t a } } .$ . In proposition 2 we show that when $M ( z , \cdot )$ is dominated by $\pi _ { \mathrm { d a t a } }$ for all $z \in { \mathcal { Z } }$ , the (logarithmically normalised) InfoNCE population risk is $O ( 1 / k )$ close to the expected cross-entropy between $\widehat { Q } _ { \tau , k } ^ { \phi }$ and an idealised empirical Markov $\widehat { M _ { k } }$ kernel built from the same k unseen data points importance weighted according to the density of $M ( z , \cdot )$ with respect to $\pi _ { \mathrm { d a t a } }$ , where z is the search query point. In this sense $\hat { M } _ { k }$ conveys whatever notion of similarity is implicitly represented in M, and sampling from $\widehat { M _ { k } }$ has the interpretation of an idealised softmax similarity search.

• Motivated by these considerations of InfoNCE when k is large, we turn to generalisation analysis in section 4. The key mathematical contribution in section 4.1 is a new continuity bound for the InfoNCE loss, presented in proposition 4. This bound is obtained by via Gˆateaux diferentiation, exploiting the specific structure of the InfoNCE loss, whereas Lei et al. [2023]’s analysis applies more generally to $\ell _ { 2 } / \ell _ { \infty }$ -Lipschitz or self-bounding loss functions. The continuity bound is applied to bounding InfoNCE Rademacher complexity in section 4.2 when the embedding function is chosen from a class of functions Lipschitz in a parameter. This demonstrates the structure of averaging over k negative samples in the InfoNCE loss carries over to stabilisation of generalisation error as k grows.

• Possible extensions are discussed in section 5. All proofs are in the appendix.

## 1.4 Notation

Throughout this work, $\mathcal { Z }$ is a set and $\mathcal { F } _ { \mathcal { Z } }$ is a σ-algebra of subsets of $\mathcal { Z }$ . The set of probability measures on the measurable space $( \mathcal { Z } , \mathcal { F } _ { \mathcal { Z } } )$ is denoted $\mathcal { P } ( \mathcal { Z } )$ . The delta-Dirac measure located at $z \in { \mathcal { Z } }$ is denoted $\delta _ { z }$ . For $d \ge 2 , p \ge 1$ and $v = ( v _ { 1 } , \ldots , v _ { d } ) \in \mathbb { R } ^ { d }$ we write the norm $\begin{array} { r } { \| v \| _ { p } : = ( \sum _ { i = 1 } ^ { d } | v _ { j } | ^ { p } ) ^ { 1 / p } } \end{array}$ , and denote by $\langle \cdot , \cdot \rangle$ the Euclidean inner product. We write $\mathbb { S } ^ { d - 1 }$ for the set all of $v \in \mathbb { R } ^ { d }$ such that $\| v \| _ { 2 } = 1$ . For functions $f : \mathcal { Z } \to \mathbb { R } ^ { d }$ we define the norm $\begin{array} { r } { \| f \| _ { 2 , \infty } : = \operatorname* { s u p } _ { z \in \mathcal { Z } } \| f ( z ) \| _ { 2 } } \end{array}$ and denote by $B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ the Banach space of all measurable $f : \mathcal { Z } \to \mathbb { R } ^ { d }$ such that $\| f \| _ { 2 , \infty } < \infty$ . We denote by $B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } ) \setminus B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ the set of those functions $f$ such that $\| f ( z ) \| _ { 2 } = 1$ for all $z \in Z$ . For scalars $a , b ,$ the maximum and minimum are denoted $a \lor b$ and $a \wedge b .$ , respectively.

## 2 Contrastive learning with the InfoNCE loss

## 2.1 Preliminaries

The model we assume for contrastive learning training data has three ingredients:

• a set Z;

• a probability measure $\pi _ { \mathrm { d a t a } } \in \mathcal { P } ( \mathcal { Z } )$ ;

• a Markov kernel $M : \mathcal { Z } \times \mathcal { F } _ { \mathcal { Z } }  [ 0 , 1 ]$ , i.e., for each $z \in \mathcal { Z } , M ( z , \cdot ) \in \mathcal { P } ( \mathcal { Z } )$ , and for each $A \in { \mathcal { F } } _ { \mathcal { Z } }$ 2 $z \mapsto M ( z , A )$ is a measurable function.

The training data comprises $n \geq 1$ tuples, $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n } .$ , where, for $k \geq 1$ , each tuple $\mathbf Z _ { i } \in \mathcal Z ^ { 2 + k }$ consists of an anchor sample, $Z _ { i } ^ { a } ;$ ; a positive sample, $Z _ { i } ^ { + }$ ; and negative samples, $Z _ { i 1 } ^ { - } , \ldots , Z _ { i k } ^ { - }$ . The following is taken as a standing assumption throughout the entirety of this work.

(A) $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n }$ are i.i.d. and each tuple $\mathbf { Z } _ { i }$ is distributed: $Z _ { i } ^ { a } \sim \pi _ { \mathrm { d a t a } } , Z _ { i } ^ { + } \sim M ( Z _ { i } , \cdot ) , Z _ { i 1 } ^ { - } , \ldots , Z _ { i k } ^ { - }$ <sup>iid</sup>∼ π<sub>data</sub>.

In practice, positive samples are generated by applying various transformations to the anchor samples, sometimes called “views”. Examples include, for images: cropping, resizing, rotation, colour adjustment, jitter and blurring; and for text: synonym replacement, back-translation, and swapping, insertion and deletion of characters, tokens, or words. All of these transformations are usually subject to some degree of randomisation, for example randomly choosing the area to be cropped, the rotation angle, etc., and the choice of transformation, or the order in which to compose transformations, may also be randomised. We assume that any and all such randomisation is encapsulated in the Markov kernel M. We do not assume any particular functional form or algorithmic description of M is known; for our purposes it can be thought of as a “black box” sample generator (later subject to assumption (B), which is applied solely within section 3). Similarly, we do assume any particular functional form for $\pi _ { \mathrm { d a t a } }$ , in practice this distribution is unknown.

For our purposes, we may think of contrastive learning as choosing an embedding function $\widehat { \phi }$ from some class $\bar { \Phi } \subset B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ by minimising the empirical InfoNCE risk associated with $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n }$ . The procedure is outlined in algorithm 1. Later on, in section 4, we will consider specific choices of Φ.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 InfoNCE Empirical Risk Minimization

inputs: anchor samples  $Z_{1}^{a},\ldots,Z_{n}^{a}\stackrel{\mathrm{iid}}{\sim}\pi_{data}$ ; integer  $k\geq1$ , set of functions  $\Phi$ ; temperature param.  $\tau&gt;0$ .

for  $i=1,\ldots,n$ ,

– draw one positive sample  $Z_{i}^{+}\sim M(Z_{i}^{a},\cdot)$ 

– draw k negative samples  $Z_{i1}^{-},\ldots,Z_{ik}^{-}\stackrel{\mathrm{iid}}{\sim}\pi_{data}$ 

end for

do minimisation:

 $\widehat{\phi}=\operatorname{argmin}_{\phi\in\Phi}\sum_{i=1}^{n}\log\left(1+\sum_{j=1}^{k}\exp\left[\langle\phi(Z_{i}^{a}),\phi(Z_{ij}^{-})-\phi(Z_{i}^{+})\rangle/\tau\right]\right)$ .
</div>

The empirical and population InfoNCE risk functionals, $\widehat { R } _ { n } ( \cdot ; k , \tau )$ and $R ( \cdot ; k , \tau )$ , are defined as follows. For $\phi \in \mathcal { B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } ) , k \geq 1$ and $\tau > 0$

$$
\widehat {R} _ {n} (\phi ; k, \tau) := \frac {1}{n} \sum_ {i = 1} ^ {n} \log \left(1 + \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z _ {i} ^ {a}, \phi (Z _ {i j} ^ {-}) - \phi (Z _ {i} ^ {+}) \rangle / \tau}\right) - \log k,\tag{1}
$$

$$
R (\phi ; k, \tau) := \mathbb {E} \left[ \log \left(1 + \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}), \phi (Z _ {j} ^ {-}) - \phi (Z ^ {+}) \rangle / \tau}\right) \right] - \log k,\tag{2}
$$

where in (2), expectation is over $Z ^ { a } \sim \pi _ { \mathrm { d a t a } } , Z ^ { + } \sim M ( Z ^ { a } , \cdot ) , Z _ { 1 } ^ { - } , \ldots , Z _ { k } ^ { - } \stackrel { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } } ,$ so (A) implies $R ( \phi ; k , \tau ) = \mathbb { E } [ \widehat { R } _ { n } ( \phi ; k , \tau ) ]$ . The empirical risk $\widehat { R } _ { n } ( \phi ; k , \tau )$ difers from the quantity in Algorithm 1 only by the multiplicative $1 / n$ and logarithmic − log k normalisation factors, which the argmin operation is invariant to.

## 2.2 Discussion of the setup

• The assumption within (A) that the tuples $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n }$ are i.i.d. was made in [Saunshi et al., 2019, Lei et al., 2023]. Similarly to those works, we use the assumption of i.i.d. tuples to apply a standard generalisation bound [Mohri et al., 2018, Thm. 3.3] in the proof of proposition 3, section 4. The assumption that, within each tuple, negative samples are i.i.d. from the same distribution as the anchor samples is quite common in the literature, e.g., [Wang and Isola, 2020a, Zimmermann et al., 2021]. This assumption of i.i.d. negative samples is used in the proofs of propositions 1 and 2 in section 3, concerning interpretation of the InfoNCE population risk. However, this part of (A) is not needed for any of the results in section 4 concerning generalisation. In fact, propositions 4 and lemma 1 in section $^ { 4 , }$ used to bound empirical Rademacher complexity, hold for any realised values $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n }$ of $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n } $ , and hence do not require assumption (A) at all.

• In some presentations of contrastive learning and the InfoNCE loss, it is assumed that the joint distribution of anchor and positive pairs has a symmetric density and/or that the marginal distributions of the positive and negative samples are the same, e.g., [van den Oord et $\mathrm { a l . }$ , 2018, Wang and Isola, 2020a]. We make no such assumptions. Indeed the author considers it unrealistic to assume that if $Z \sim \pi _ { \mathrm { d a t a } }$ and $Z ^ { + } \sim M ( Z , \cdot )$ then the marginal distribution of $Z ^ { + }$ is exactly $\pi _ { \mathrm { d a t a } }$ . This would amount to saying that M admits $\pi _ { \mathrm { d a t a } }$ as an invariant distribution; in practice the distribution $\pi _ { \mathrm { d a t a } }$ is unknown and nothing is built in to achieve such invariance.

• A temperature parameter was not presented in the InfoNCE loss by [van den Oord et al., 2018] but was included in, $\mathrm { e . g . }$ , Wu et al. [2018], Chen et al. [2020], where the loss is sometimes called the normalized temperature-scaled cross entropy loss (NT-Xent). In their generalisation analyses, Saunshi et al. [2019],Lei et al. [2023] did not explicitly consider a temperature parameter, but considered embedding functions ϕ satisfying $\| \phi \| _ { 2 , \infty } \leq R$ for some finite $R > 0$ . In the present work our embedding functions are always members of $B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ , i.e., $\| \phi ( z ) \| _ { 2 } = 1$ for all $z \in { \mathcal { Z } }$ . This is not essential for many of our results, but we make this assumption to conform with common practice and presentation of contrastive learning; for $\phi \in { \cal B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } ) , \langle \phi ( z ) , \phi ( z ^ { \prime } ) \rangle$ ⟩ is the cosine similarity between $\phi ( z )$ and $\phi \left( z ^ { \prime } \right)$

• For the purposes of our analysis it will not be important that $\widehat { \phi }$ is an exact minimiser as in algorithm 1, we write it as such just for sake of illustration; our various bounds presented in sections 3 and 4 hold uniformly over embedding functions belonging $B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ or some some $\Phi \subset B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ .

## 3 Similarity search interpretation of the InfoNCE loss

## 3.1 Basics of similarity search

Suppose that we are given an embedding function $\phi \in { \cal B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ , for example $\widehat { \phi }$ from algorithm 1, and we use it for similarity search on unseen data, as follows: for some $m \geq 1$ , let $\xi _ { 1 } , \ldots , \xi _ { m }$ be points in $\mathcal { Z }$ (we use the notation $\xi _ { i }$ to distinguish these samples from any of the constituents of the training tuples $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n } )$ and let $z \in { \mathcal { Z } }$ be a query point. Similarity search is the task:

$$
\text { find } \quad \xi_ {\star} \in \{\xi_ {1}, \dots , \xi_ {m} \} \quad \text { which   maximises } \quad \langle \phi (z), \phi (\xi_ {\star}) \rangle .\tag{3}
$$

A “softmax”, randomised relaxation of this similarity search with the same query point z is to instead sample $\xi _ { \star }$ from the set $\{ \xi _ { 1 } , \ldots , \xi _ { m } \}$ as follows:

$$
\mathrm{set} \quad \xi_ {\star} = \xi_ {i} \quad \mathrm{withprobability} \quad \frac {W _ {i} ^ {1 / \tau} (z)}{\sum_ {j = 1} ^ {m} W _ {j} ^ {1 / \tau} (z)}, \quad \mathrm{where} \quad W _ {i} (z) := e ^ {\langle \phi (z), \phi (\xi_ {i}) \rangle}.\tag{4}
$$

When $\tau  0 ,$ , (4) reduces to (3). Our next objective is to explain the connection between (4) and the InfoNCE population risk $R ( \phi ; k , \tau )$ . We need some further definitions. For any $\phi \in \cal B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ and

$\tau > 0 .$ , define the Markov kernel<sup>2</sup>:

$$
Q _ {\tau} ^ {\phi} (z, A) := \frac { \int_ {A} e ^ {\langle \phi (z) , \phi (z ^ {\prime}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {\prime})}{ \int_ {\mathcal {Z}} e ^ {\langle \phi (z) , \phi (z ^ {\prime}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {\prime})}, \qquad z \in \mathcal {Z}, A \in \mathcal {F} _ {\mathcal {Z}}.\tag{5}
$$

We also introduce the following empirical counterpart of $Q _ { \tau } ^ { \phi }$ built from the unseen data $\xi _ { 1 } , \ldots , \xi _ { m }$ 2

$$
\widehat {Q} _ {\tau , m} ^ {\phi} (z, A) := \frac {\sum_ {i = 1} ^ {m} e ^ {\langle \phi (z) , \phi (\xi_ {i}) \rangle / \tau} \delta_ {\xi_ {i}} (A)}{\sum_ {i = 1} ^ {m} e ^ {\langle \phi (z) , \phi (\xi_ {i}) \rangle / \tau}}, \quad z \in \mathcal {Z},   A \in \mathcal {F} _ {\mathcal {Z}}.\tag{6}
$$

These Markov kernels have the following interpretations: sampling $\xi _ { \star } \sim \widehat { Q } _ { \tau , m } ^ { \phi } ( z , \cdot )$ is equivalent to $( 4 )$ In that sense, $\widehat { Q } _ { \tau , m } ^ { \phi }$ just encapsulates the probabilities in the softmax similarity search. If the unseen data are drawn from $\pi _ { \mathrm { d a t a } } , ~ \mathrm { i . e . , ~ } \xi _ { 1 } , \ldots , \xi _ { m } \stackrel { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$ , then by the strong law of large numbers, for any $z \in { \mathcal { Z } }$ and $A \in { \mathcal { F } } _ { { \mathcal { Z } } } , { \widehat { Q } } _ { \tau , m } ^ { \phi } ( z , A )$ converges almost surely to $Q _ { \tau } ^ { \phi } ( z , A )$ as $m  \infty$ . Thus $Q _ { \tau } ^ { \phi }$ captures the behaviour of the softmax similarity search (4) in the limit of a large amount of unseen data.

## 3.2 Relating population risk to integrated cross-entropy

We define the cross-entropy between $M ( z , \cdot )$ and $Q _ { \tau } ^ { \phi } ( z , \cdot )$ (with $\pi _ { \mathrm { d a t a } }$ taken as a dominating measure for the latter) as:

$$
\begin{array}{r l} & {\mathrm{CrossEnt} [ M (z, \cdot) \| Q _ {\tau} ^ {\phi} (z, \cdot) ] := - \int_ {\mathcal {Z}} \log \left[ \frac {\mathrm{d} Q _ {\tau} ^ {\phi} (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (z ^ {\prime}) \right] M (z, \mathrm{d} z ^ {\prime})} \\ & {\qquad = - \frac {1}{\tau} \int_ {\mathcal {Z}} \langle \phi (z), \phi (z ^ {\prime}) \rangle M (z, \mathrm{d} z ^ {\prime}) + \log \int_ {\mathcal {Z}} e ^ {\langle \phi (z), \phi (z ^ {\prime}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {\prime}),} \end{array}\tag{7}
$$

where $\frac { \mathrm { d } . } { \mathrm { d } . }$ denotes Radon-Nikodym derivative $^ 3 ,$ , and the second equality follows from the definition (5). The following proposition relates this cross-entropy, integrated with respect to $\pi _ { \mathrm { d a t a } }$ , to the population risk $R ( \phi ; k , \tau )$

Proposition 1. For any $\phi \in \mathcal { B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } ) , k \geq 1$ and $\tau > 0$

$$
- \log \left(1 + \frac {e ^ {2 / \tau}}{k}\right) \leq \int_ {\mathcal {Z}} \operatorname{CrossEnt} [ M (z, \cdot) \| Q _ {\tau} ^ {\phi} (z, \cdot) ] \pi_ {\text { data }} (\mathrm{d} z) - R (\phi ; k, \tau) \leq \frac {1}{8 k} (e ^ {2 / \tau} - 1) ^ {2}.
$$

Proposition 1 implies that, as $k \to \infty , R ( \phi ; k , \tau )$ converges to the $\pi _ { \mathrm { d a t a } } \mathrm { - i n t e g r a t e d }$ cross-entropy between $M ( z , \cdot )$ and $Q _ { \tau } ^ { \phi } ( z , \cdot )$ . Modulo the technical details of our measure-theoretic setup, it is already well known that $R ( \phi ; k , \tau )$ approaches a limit as $k  \infty , \mathrm { e . g . }$ , [van den Oord et al., 2018, Wang and Isola, 2020a, Zimmermann et al., 2021]. However, noting that for $x \geq 0 , \log ( 1 + x ) \leq x$ , proposition 1 implies a rate $O ( 1 / k )$ uniformly over $\displaystyle \dot { \phi } \in { \cal B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ , whereas a rate $O ( 1 / \sqrt { k } )$ was reported by [Wang and Isola, 2020b].

What more does proposition 1 tell us? The integrated cross-entropy considered in proposition 1 quantifies discrepancy between the Markov kernels M and $Q _ { \tau } ^ { \phi }$ . Thus proposition 1 indicates that, if we were to hypothetically choose ϕ by minimising $R ( \phi ; k , \tau )$ when k is large, this would amount to minimising discrepancy between M and $Q _ { \tau } ^ { \phi }$ . We might therefore take the view that choosing $\phi$ by minimising InfoNCE risk amounts, in efect, to learning the Markov kernel M. Consideration of $Q _ { \tau } ^ { \phi }$ sheds some light on the role of the temperature parameter τ here. Since $\| \phi ( z ) \| _ { 2 } = 1$ for all z, the following two inequalities follow from the definition of $Q _ { \tau } ^ { \phi }$ in (5) and hold for all $z \in { \mathcal { Z } }$ and $A \in { \mathcal { F } } _ { \mathcal { Z } }$

$$
e ^ {- 2 / \tau} \pi_ {\mathrm{data}} (A) \leq Q _ {\tau} ^ {\phi} (z, A) \leq e ^ {2 / \tau} \pi_ {\mathrm{data}} (A).\tag{8}
$$

Thus we see that when $\tau \to \infty , Q _ { \tau } ^ { \phi } ( z , \cdot )$ is constrained to be closer and closer to $\pi _ { \mathrm { d a t a } } .$ for all $z \in { \mathcal { Z } }$ , no matter what the choice of $\phi \in \cal B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ ). In this sense, a large value of $\tau$ limits the ability of $Q _ { \tau } ^ { \phi }$ to closely approximate an arbitrary M.

## 3.3 Relating population risk to expected empirical cross-entropy

Even if we are given $\phi ,$ the Markov kernel $Q _ { \tau } ^ { \phi }$ is not available in practice because the data-generating distribution $\pi _ { \mathrm { d a t a } }$ is unknown. Let us now look more closely at the empirical Markov kernel $\widehat { Q } _ { \tau , k } ^ { \phi } ,$ defined in (6) which we can access in practice for any given $\phi$ and $\xi _ { 1 } , \ldots , \xi _ { m }$ . Consider the domination assumption:

## (B) for all z ∈ Z, M (z, ·) ≪ π<sub>data</sub>,

From a practical point of view, (B) has a simple interpretation: it says that if the data-generating distribution π<sub>data</sub> assigns zero probability to any set $A \in { \mathcal { F } } _ { \mathcal { Z } }$ , then there must be zero probability of generating a positive sample in $A , { \mathrm { i . e . , ~ } } M ( z , A ) = 0$ , for any point $z \in { \mathcal { Z } }$

When (B) holds, $M ( z , \cdot )$ admits a density with respect to $\pi _ { \mathrm { d a t a } }$ , denoted $\frac { \mathrm { d } M ( z , \cdot ) } { \mathrm { d } \pi _ { \mathrm { d a t a } } } ( \cdot )$ and we may define the empirical Markov kernel and probability measure,

$$
\widehat {M} _ {m} (z, A) := \frac {\sum_ {i = 1} ^ {m} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {i}) \delta_ {\xi_ {i}} (A)}{\sum_ {i = 1} ^ {m} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {i})}, \qquad \widehat {\pi} _ {\mathrm{data}, m} (A) := \frac {1}{m} \sum_ {i = 1} ^ {m} \delta_ {\xi_ {i}} (A), \qquad z \in \mathcal {Z},   A \in \mathcal {F} _ {\mathcal {Z}}.
$$

Here $\xi _ { 1 } , \ldots , \xi _ { m }$ are the unseen data as in the similarity search $( 3 ) \AA - \thinspace ( 4 )$ . The Markov kernel $\widehat { M } _ { m }$ has the interpretation of importance-weighting each of the points $\xi _ { 1 } , \ldots , \xi _ { m }$ to account for how likely they are under $M ( z , \cdot )$ versus $\pi _ { \mathrm { d a t a } }$ . We may interpret such weighting as an abstract measure of similarity to the query point $z ,$ implicitly conveyed by whatever randomised transformations, or “views”, constitute the Markov kernel $M$

Similarly to (7), we define the cross-entropy between $\widehat { M } _ { m }$ and $\widehat { Q } _ { \tau , m } ^ { \phi }$

$$
\begin{array}{r l} & {\mathrm{CrossEnt} \left[ \widehat {M} _ {m} (z, \cdot) \| \widehat {Q} _ {\tau , m} ^ {\phi} (z, \cdot) \right] := - \int_ {\mathcal {Z}} \log \left[ \frac {\mathrm{d} \widehat {Q} _ {\tau , m} ^ {\phi} (z , \cdot)}{\mathrm{d} \widehat {\pi} _ {\mathrm{data} , m}} (z ^ {\prime}) \right] \widehat {M} _ {m} (z, \mathrm{d} z ^ {\prime})} \\ & {\qquad = - \frac {1}{\tau} \frac {\sum_ {i = 1} ^ {m} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {i}) \langle \phi (z) , \phi (\xi_ {i}) \rangle}{\sum_ {i = 1} ^ {m} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {i})} + \log \left(\frac {1}{m} \sum_ {i = 1} ^ {m} e ^ {\langle \phi (z), \phi (\xi_ {i}) \rangle / \tau}\right).} \end{array}\tag{9}
$$

(10)

Proposition 2. If (B) holds and $\xi _ { 1 } , \xi _ { 2 } , \ldots \overset { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$ , then for any $\phi \in { \cal B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } ) , k \geq 1$ and $\tau > 0$

$$
\begin{array}{r l r} & & {\left| \int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M} _ {k} (z, \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z, \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z) - R (\phi ; k, \tau) \right|} \\ & & {\leq \frac {4}{\tau k} \mathbb {E} \left[ \left| \frac {\mathrm{d} M (Z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (Z ^ {\prime}) \right| ^ {2} \right] + \log \left(1 + \frac {e ^ {2 / \tau}}{k}\right),} \end{array}
$$

where on the $r . h . s . , Z , Z ^ { \prime } \overset { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$

Proposition 2 tells us that when the unseen data points $\xi _ { 1 } , \xi _ { 2 } , \ldots$ are drawn from $\pi _ { \mathrm { d a t a } } ,$ and are equal in number to the number of negative samples per training tuple, $k ,$ then $R ( \phi ; k , \tau )$ is $O ( 1 / k )$ close to the expected value of (9), integrated with respect to $\pi _ { \mathrm { d a t a } }$ . Here the expectation, $\mathbb { E } ( \cdot )$ in proposition 2, integrates out $\xi _ { 1 } , \dots , \xi _ { k } \overset { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$ . In this sense, hypothetically minimising $R ( \phi ; k , \tau )$ with respect to $\phi ,$ when k is large, pushes $\widehat { Q } _ { \tau , k } ^ { \phi }$ towards $\widehat { M _ { k } }$ . In turn, this can be interpreted as meaning that the softmax similarity search (4), with $m = k$ , approximates sampling $\xi _ { \star } \sim \widehat { M } _ { k } ( z , \cdot )$

In practice the number of unseen data m will generally not be equal to the number of negative samples k per training tuple. Rather, we present the case $m = k$ because it is relatively simple to analyse mathematically. One could bound the diference between $R ( \phi ; k , \tau )$ and the expected crossentropy with some m $\neq k$ using similar arguments to those in the proofs of propositions 1 and 2, but for brevity we do not pursue such a bound here. In any case, propositions 1 and 2 together imply that $\begin{array} { r } { \int _ { \mathcal { Z } } \mathbb { E } \left( \mathrm { C r o s s E n t } \left[ \widehat { M } _ { k } ( z , \cdot ) \lVert \widehat { Q } _ { \tau , k } ^ { \phi } ( z , \cdot ) \right] \right) \pi _ { \mathrm { d a t a } } ( \mathrm { d } z ) } \end{array}$ converges to $\begin{array} { r } { \int _ { \mathcal { Z } } \mathrm { C r o s s E n t } [ M ( \boldsymbol { z } , \cdot ) \lVert Q _ { \tau } ^ { \phi } ( \boldsymbol { z } , \cdot ) ] \pi _ { \mathrm { d a t a } } ( \mathrm { d } \boldsymbol { z } ) } \end{array}$ as $k \to \infty$

## 3.4 Simplified bounds for the DCL loss function

Propositions 1 and 2 relate $R ( \phi ; k , \tau )$ to two diferent quantities. Can we say anything about whether, for finite $k , R ( \phi ; k , \tau )$ is closer to one or the other? The picture becomes clearer if we consider a slight change to the InfoNCE loss. Suppose the term $^ { 6 6 } 1 + ^ { 5 9 }$ is omitted from the loss in algorithm 1. This was called the Decoupled Contrastive Learning (DCL) loss by Yeh et al. [2022], who discussed its properties and gave evidence of superior performance in practice. The population risk becomes:

$$
\widetilde {R} (\phi ; k, \tau) := \mathbb {E} \left[ \log \left(\sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}), \phi (Z _ {j} ^ {-}) - \phi (Z ^ {+}) \rangle / \tau}\right) \right] - \log k,
$$

instead of (2).

It can be checked that with fairly minor modifications to the proofs of propositions 1 and 2 (see appendix A.1), under all the same conditions as in those propositions, the DCL population risk $\widetilde { R } ( \phi ; k , \tau )$ satisfies:

$$
0 \leq \int_ {\mathcal {Z}} \mathrm{CrossEnt} [ M (z, \cdot) \| Q _ {\tau} ^ {\phi} (z, \cdot) ] \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau) \leq \frac {1}{8 k} (e ^ {2 / \tau} - 1) ^ {2},\tag{11}
$$

i.e., compared to the proposition 1 the lower bound is zero, and

$$
\left| \int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M _ {k}} (z, \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z, \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau) \right| \leq \frac {4}{\tau k} \mathbb {E} \left[ \left| \frac {\mathrm{d} M (Z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (Z ^ {\prime}) \right| ^ {2} \right],\tag{12}
$$

i.e., the additive term $\log ( 1 + e ^ { 2 / \tau } / k )$ in proposition 2 vanishes.

We can see that if $\tau  0 ,$ k must grow exponentially fast in $1 / \tau$ to control the bound in (11), but only as fast as $1 / \tau$ to control the bound in (12). In practice, τ is often chosen somewhere in the range $0 . 0 7 - 0 . 5$ [Chen et al., 2020, He et al., 2020], so $e ^ { \bar { 1 } / \tau }$ could be quite a large number. This prompts the question: is the bound in (11) tight? The answer is that for small values of k it generally is not tight (lemma 4 shows that an alternative bound $2 / \tau$ holds for any $k \geq 1$ and $\tau > 0 )$ ), but as k grows, exponential dependence on $e ^ { 1 / \tau }$ cannot be avoided: it is shown in appendix A.1 that if $e ^ { 1 / \tau } > 2 .$ then an example can be constructed such that

$$
\begin{array}{r l} & {\underset {k \to \infty} {\limsup} \frac {\left| \int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M _ {k}} (z , \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z , \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k , \tau) \right|}{\int_ {\mathcal {Z}} \mathrm{CrossEnt} [ M (z , \cdot) \| Q _ {\tau} ^ {\phi} (z , \cdot) ] \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k , \tau)}} \\ & {\qquad \leq \frac {6 5}{\tau (e ^ {1 / \tau} - 2)} \mathbb {E} \left[ \left| \frac {\mathrm{d} M (Z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (Z ^ {\prime}) \right| ^ {2} \right].} \end{array}\tag{13}
$$

The numerical constant 65 represents some particular choices made in the construction of this example and so may be improved. Therefore, in this example and for large k, the numerator on the left hand side of (13) will be much smaller than the denominator if

$$
\tau e ^ {1 / \tau} \gg \mathbb {E} \left[ \left| \frac {\mathrm{d} M (Z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (Z ^ {\prime}) \right| ^ {2} \right].
$$

## 4 Generalisation analysis

The results of section 3 explain the behaviour of the population risk $R ( \phi ; k , \tau )$ when k is large, with the bounds in propositions 1 and 2 holding uniformly over any embedding function $\phi \in \cal B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ . As such, those bounds apply to $R ( \widehat \phi , k , \tau )$ where the embedding function $\widehat { \phi }$ is obtained by empirical risk minimisation as in algorithm 1, or by any other approximate minimisation scheme which outputs some member of $B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ ).

In order to quantify generalisation performance we would like to upper bound $R ( \widehat \phi , k , \tau )$ in terms of $\widehat { R } _ { n } ( \widehat { \phi } , k , \tau )$ . This would tell us how the quality of training, i.e., achieving a small value of $\widehat { R } _ { n } ( \widehat { \phi } , k , \tau )$ transfers to similarity search on unseen data, as discussed in section 3. Following the usual workflow of statistical learning theory, instead of analysing $\widehat { R } _ { n } ( \widehat { \phi } , k , \tau )$ directly, we seek to upper bound $R ( \cdot , k , \tau )$ in terms of $\widehat { R } _ { n } ( \cdot , k , \tau )$ uniformly over some class of embedding functions $\Phi \subset B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ to which ϕ is supposed to belong. Section 4.1 sets out tools for doing so applicable to a general function class Φ. These tools are applied in section 4.2 to a specific class of embedding functions which are Lipschitz in a parameter.

## 4.1 Generalisation and continuity bounds

$$
\mathbf {z} _ {i} = (z _ {i} ^ {a}, z _ {i} ^ {+}, z _ {i 1} ^ {-}, \dots , z _ {i k} ^ {-}) \in \mathcal {Z} ^ {2 + k}
$$

$$
\ell (\phi , \mathbf {z} _ {i}, k, \tau) := \log \left(\frac {1}{k} + \frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (z _ {i} ^ {a}), \phi (z _ {i j} ^ {-}) - \phi (z _ {i} ^ {+}) \rangle / \tau}\right)\tag{14}
$$

so that substituting the random tuple $\mathbf { Z } _ { i }$ in place of $\mathbf { z } _ { i }$ we have:

$$
\widehat {R} _ {n} (\phi ; k, \tau) = \frac {1}{n} \sum_ {i = 1} ^ {n} \ell (\phi , \mathbf {Z} _ {i}, k, \tau).
$$

The following proposition is an application to the InfoNCE risk of a well-known empirical Rademacher complexity generalisation bound for additive loss functions [Mohri et al., 2018, Thm. 3.3].

Proposition 3. Let Φ be a subset of $B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ ). For any $n \geq 1 , k \geq 1 , \tau > 0$ and $\delta \in ( 0 , 1 )$ , it holds with probability at least $1 - \delta$ that for all $\phi \in \Phi$ 2

$$
R (\phi ; k, \tau) \leq \widehat {R} _ {n} (\phi ; k, \tau) + 2 \mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {Z} _ {i}, k, \tau)   \bigg |   \mathbf {Z} _ {1}, \ldots , \mathbf {Z} _ {n} \right] + \frac {1 2}{\tau} \sqrt {\frac {\log \frac {2}{\delta}}{2 n}},
$$

where $\sigma _ { 1 } , \ldots , \sigma _ { n }$ are i.i.d. Rademacher variables, independent of $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n }$

The conditional expectation term in proposition 3 is the empirical Rademacher complexity associated with the InfoNCE loss functional (14) evaluated over $\Phi$ , with respect to the training sample of tuples $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n }$ . This empirical Rademacher complexity quantifies generalisation error, i.e., the capacity of Φ, via (14), to overfit the training data.

To obtain insight into this generalisation error, we need to bound the empirical Rademacher complex ity in such a way that the roles of Φ, k, τ, n, etc., become clear. A crucial step towards such a bound is to obtain some kind of quantitative continuity estimate for the loss function with respect to the embedding function $\phi .$ Such an estimate opens the door to bounds on Rademacher complexity using well-known techniques such as contraction [Maurer, 2016], and/or covering numbers, chaining and Dudley’s integral lemma, e.g., [Wainwright, 2019, Ch. 5]. The following proposition is the main technical contribution of section 4. Here $\beta \geq 1$ is a mathematical parameter, introduced in order to be able to tune the bound in proposition 4 to account for the algorithmic temperature parameter $\tau$ (this tuning is demonstrated in section 4.2). As such, we call $\beta$ the “inverse temperature” parameter.

Proposition 4. For any $k \geq 1 , \mathbf { z } = ( z ^ { a } , z ^ { + } , z _ { 1 } ^ { - } , \ldots , z _ { k } ^ { - } ) \in \mathcal { Z } ^ { 2 + k } , \tau > 0 , \beta \geq 1$ , and $\phi , \phi ^ { \prime } \in B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$

$$
\begin{array}{l} | \ell (\phi , \mathbf {z}, k, \tau) - \ell (\phi^ {\prime}, \mathbf {z}, k, \tau) | \\ \leq \frac {4}{\tau} \| \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}) \| _ {2} + \frac {1}{\tau} \| \phi (z ^ {+}) - \phi^ {\prime} (z ^ {+}) \| _ {2} + \frac {3}{\tau} e ^ {2 / \beta \tau} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| \phi (z _ {j} ^ {-}) - \phi^ {\prime} (z _ {j} ^ {-}) \| _ {2} ^ {\beta}\right) ^ {1 / \beta}. \end{array}
$$

A key feature of the bound in proposition 4 is that it preserves the structure of averaging over the negative samples present in the InfoNCE loss (14); specifically the negative samples $z _ { 1 } ^ { - } , \ldots , z _ { k } ^ { - }$ enter into the bound only through a “power-mean” parameterised by the inverse temperature $\beta .$ . We shall see in section 4.2 how this averaging over negative samples transfers to bounds on the Rademacher complexity.

Let us compare the bound in proposition 4 to an alternative estimate available in the literature, derived from Lipschitz continuity of the logistic loss:

$$
\ell^ {\log} (v) := \log \left(1 + \sum_ {j = 1} ^ {k} \exp (- v _ {j})\right), \qquad v = (v _ {1}, \dots , v _ {k}) \in \mathbb {R} ^ {k},
$$

as discussed in, e.g., [Lei et al., 2023]. Note that when:

$$
v _ {j} = \langle \phi (z ^ {a}), \phi (z ^ {+}) - \phi (z _ {j} ^ {-}) \rangle / \tau ,\tag{15}
$$

we have $\ell ^ { \log } ( v ) - \log k = \ell ( \phi , \mathbf { z } , k , \tau )$

It is known that $\ell ^ { \mathrm { l o g } }$ is 1-Lipschitz with respect to the $\| \cdot \| _ { \infty }$ norm on $\mathbb { R } ^ { k }$ [Lei et al., 2019] (and hence also 1-Lipschitz with respect to the $\| \cdot \| _ { p }$ norm for any $p \geq 1 \}$ ), that is:

$$
| \ell^ {\log} (v) - \ell^ {\log} (v ^ {\prime}) | \leq \| v - v ^ {\prime} \| _ {\infty},\tag{16}
$$

for all $v , v ^ { \prime } \in \mathbb { R } ^ { k }$ . In the case (15) note that $\ell ^ { \log } ( v )$ and $\ell ( \phi , { \bf z } , k , \tau )$ difer only by an additive factor of log k. Therefore, the 1-Lipschitz property of $\ell ^ { \log } ( \cdot )$ transfers to $\ell ( \cdot , { \bf z } , k , \tau )$ as follows: for any ϕ, $\phi ^ { \prime } \in B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ let (15) hold and let $v ^ { \prime } = \bar { ( } v _ { 1 } ^ { \prime } , \ldots , \bar { v } _ { k } ^ { \prime } )$ be defined by replacing ϕ in (15) with $\phi ^ { \prime }$ . Then as a consequence of (16) we have:

$$
\begin{array}{r l} & {| \ell (\phi , \mathbf {z}, k, \tau) - \ell (\phi^ {\prime}, \mathbf {z}, k, \tau) |} \\ & {\leq \frac {1}{\tau} \max _ {1 \leq j \leq k} \left| \langle \phi (z ^ {a}), \phi (z ^ {+}) - \phi (z _ {j} ^ {-}) \rangle - \langle \phi^ {\prime} (z ^ {a}), \phi^ {\prime} (z ^ {+}) - \phi^ {\prime} (z _ {j} ^ {-}) \rangle \right|} \\ & {\leq \frac {\sqrt {6}}{\tau} \left(\| \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}) \| _ {2} ^ {2} + \| \phi (z ^ {+}) - \phi^ {\prime} (z ^ {+}) \| _ {2} ^ {2} + \max _ {1 \leq j \leq k} \| \phi (z _ {j} ^ {-}) - \phi^ {\prime} (z _ {j} ^ {-}) \| _ {2} ^ {2}\right) ^ {1 / 2},} \end{array}\tag{17}
$$

(18)

where the second inequality uses the fact established in [Lei et al., 2023][Proof of Lemma 4.3] that for any $u ^ { a } , u ^ { + } , u ^ { - } \in \mathbb { S } ^ { d - 1 }$ , the mapping $( u ^ { a } , u ^ { + } , u ^ { - } ) \mapsto \langle u ^ { a } , u ^ { + } - u ^ { - } \rangle$ is 6-Lipschitz with respect to the $\| \cdot \| _ { 2 }$ norm on $\mathbb { R } ^ { 3 d }$

Crucially in (17) the structure of averaging over negative samples in (14) has been lost. Comparing to proposition 4, we may view (17) as resorting to maximisation rather than averaging. Noting that in proposition 4 we are free to choose the inverse temperature parameter $\beta \geq 1$ , we can take $\beta $ ∞ there, in which case $e ^ { 2 / \beta \tau }  1$ and the power-mean $( k ^ { - 1 } \dot { \sum _ { j = 1 } ^ { k } } \| \phi ( z _ { j } ^ { - } ) - \phi ^ { \prime } ( z _ { j } ^ { - } ) \| _ { 2 } ^ { \beta } ) ^ { 1 / \beta }$ tends to max $1 { \le } j { \le } k \| \phi ( z _ { j } ^ { - } ) - \phi ^ { \prime } ( z _ { j } ^ { - } ) \| _ { 2 }$ . In that limit it can be seen that the bounding quantity in proposition (4) and the r.h.s. of (18) are equivalent up to a fairly modest numerical scaling due to the elementary inequalities for scalars: $\begin{array} { r } { 3 ^ { - 1 / 2 } \sum _ { i = 1 } ^ { 3 } | a _ { i } | \leq ( \sum _ { i = 1 } ^ { 3 } | a _ { i } | ^ { 2 } ) ^ { 1 / 2 } \leq \sum _ { i = 1 } ^ { 3 } | a _ { i } | } \end{array}$ . The knock-on efect of (18) for bounding Rademacher complexity is illustrated in section 4.2.

## 4.2 Application to Lipschitz embedding functions

To demonstrate how the bound in proposition 4 can be applied, let us a consider a situation in which the class of functions Φ is of the form $\{ \phi _ { \theta } ; \theta \in \Theta \} \subset B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ for some parameter set Θ, and $\phi _ { \theta }$ is Lipschitz with respect to θ in the sense of the following assumption.

(C) The input space $\mathcal { Z } \mathrm { ~ } i s \mathbb { R } ^ { d _ { \mathrm { i n } } }$ for some $d _ { \mathrm { i n } } \geq 1$ , and the parameter space is:

$$
\Theta = \{\theta \in \mathbb {R} ^ {d _ {\Theta}}: \| \theta - \theta_ {0} \| _ {2} \leq R \},\tag{19}
$$

for some $d _ { \Theta } \geq 1 , \theta _ { 0 } \in \mathbb { R } ^ { d _ { \Theta } }$ and $R > 0$ . There is some finite constant $C _ { \Phi }$ such that for all $\theta , \theta ^ { \prime } \in \Theta$ and $z \in { \mathcal { Z } }$ ，

$$
\| \phi_ {\theta} (z) - \phi_ {\theta^ {\prime}} (z) \| _ {2} \leq C _ {\Phi} \| z \| _ {2} \| \theta - \theta^ {\prime} \| _ {2}.
$$

The setting of assumption (C) could easily be generalised in a number of ways. We consider a finite dimensional Euclidean parameter θ and dependence on $\left. { z } \right. _ { 2 }$ to simplify the exposition which follows. There, our priority is to give a swift and easily interpretable demonstration of how $k , \tau , n ,$ etc. impact the empirical Rademacher complexity appearing in proposition 3.

We shall use the following pseudo-metric on Φ, associated with a realisation of tuples $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } \in$ $\mathcal { Z } ^ { 2 + k }$ , to bound Rademacher complexity.

$$
\rho_ {n} ^ {\text { info }} (\phi , \phi^ {\prime}) := \left(\frac {1}{n} \sum_ {i = 1} ^ {n} | \ell (\phi , \mathbf {z} _ {i}, k, \tau) - \ell (\phi^ {\prime}, \mathbf {z} _ {i}, k, \tau) | ^ {2}\right) ^ {1 / 2}
$$

The following lemma illustrates how this pseudo-metric can be bounded in the setting of assumption (C). The proof of lemma 1 uses proposition 4 with a particular choice of inverse temperature parameter, $\beta = 2 / \tau$ . As noted earlier, in practice τ is usually chosen somewhere in the range $0 . 0 7 - 0 . 5$ and choosing $\beta = 2 / \tau$ conveniently reduces the factor $e ^ { 2 / \beta \tau }$ in proposition 4 to e.

Lemma 1. $I f ( \mathrm { C } )$ holds, then for any n $\iota \geq 1 , k \geq 1 , \tau > 0 , \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } \in \mathcal { Z } ^ { 2 + k }$ where $\mathbf z _ { i } = ( z _ { i } ^ { a } , z _ { i } ^ { + } , z _ { i 1 } ^ { - } , \dots , z _ { i k } ^ { - } )$ and $\theta , \theta ^ { \prime } \in \Theta$

$$
\rho_ {n} ^ {\mathrm{info}} (\phi_ {\theta}, \phi_ {\theta^ {\prime}}) \leq \frac {4 e}{\tau} B _ {\tau} (\mathbf {z _ {1}}, \ldots , \mathbf {z} _ {n}) C _ {\Phi} \| \theta - \theta^ {\prime} \| _ {2},
$$

where $C _ { \Phi }$ is as in assumption (C) and

$$
B _ {\tau} (\mathbf {z} _ {1}, \dots , \mathbf {z} _ {n}) := \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \| z _ {i} ^ {a} \| _ {2} ^ {2}\right) ^ {1 / 2} + \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \| z _ {i} ^ {+} \| _ {2} ^ {2}\right) ^ {1 / 2} + \left(\frac {1}{n k} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {2 / (1 \wedge \tau)}\right) ^ {(1 \wedge \tau) / 2}.
$$

Lemma 1 is put to use along with Dudley’s entropy integral in the proof of the following proposition, which bounds the empirical Rademacher complexity.

Proposition 5. Assume (C) holds and let $C _ { \Phi } , \ d _ { \Theta }$ and R be as therein. For any $n \geq 1 , \ k \geq 1$ ${ \bf z } _ { 1 } , \ldots , { \bf z } _ { n } \in \mathcal { Z } ^ { 2 + k }$ where $\mathbf z _ { i } = ( z _ { i } ^ { a } , z _ { i } ^ { + } , z _ { i 1 } ^ { - } , \dots , z _ { i k } ^ { - } )$ and $\tau > 0$ , let $B _ { \tau } ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } )$ be as in assumption (C). Then,

$$
\mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {z} _ {i}, k, \tau) \right] \leq \frac {9 6}{\tau} \sqrt {\frac {d _ {\Theta}}{n}} \min \left\{a \sqrt {\log (3) + 1}, \sqrt {\log (1 + 2 a) + 1} \right\},
$$

where $a : = e R b _ { \tau } ( { \bf z } _ { 1 } , \ldots , { \bf z } _ { n } ) C _ { \Phi }$ and $\sigma _ { 1 } , \ldots , \sigma _ { n }$ are i.i.d. Rademacher variables.

We make the following observations on the bound in proposition 5

• On the right of the inequality, the only place that k appears is in the averaging term within $B _ { \tau } ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } )$ . If the random sample $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n }$ was substituted in place of $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n }$ , then under the i.i.d. property of $Z _ { i j } ^ { - }$ in assumption (A) the law of large numbers would be applicable to this averaging term. In that sense we see the Rademacher complexity stabilises as $k  \infty$ . It can be checked that using (18) instead of proposition 4 results in a maximisation, rather than average, over the negative sample norms.

• Other than the leading factor of $\tau ^ { - 1 }$ , the influence of the temperature parameter has been transferred all the way through to the inverse exponent of a power-mean of the norms of negative samples $\lVert z _ { i j } ^ { - } \rVert _ { 2 }$ , within $B _ { \tau } ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } )$ . Thus as the temperature tends to zero $\tau  0$ , the bound becomes increasingly sensitive to large values of the negative sample norms $\| z _ { i j } ^ { - } \| _ { 2 }$ , suggesting that, roughly speaking, generalisation error is bigger at lower temperatures.

• The embedding dimension, d in $\mathbb { S } ^ { d - 1 }$ , makes no appearance in the bound of proposition 5. This lack of dimension dependence can be understood as an advantage of embedding onto the sphere $\mathbb { S } ^ { d - 1 }$ , rather than, e.g., embedding in $\mathbb { R } ^ { d }$ in an unconstrained manner.

• To interpret the minimum term, note that a $: \leq 1 \Leftrightarrow a \sqrt { \log ( 3 ) + 1 } \leq \sqrt { \log ( 1 + 2 a ) + 1 }$ . Therefore as $a \to 0$ , the bound is $O ( a )$ , whilst as a → ∞ it is $\dot { O } ( \sqrt { \log a } )$ . In turn, if either the parameter radius $R ,$ data-dependent term $B _ { \tau } ( \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } )$ or Lipschitz constant $C _ { \Phi }$ were to tend to zero (with all other quantities held constant), then so does $a ,$ , and hence the Rademacher complexity shrinks to zero.

## 5 Discussion

A combined similarity search generalisation bound. To summarise some of the results of this work we can combine, for example, propositions 2, 3 and 5 in the following theorem.

Theorem 1. Assume (A), (B) and (C) with $\Phi = \{ \phi _ { \theta } ; \theta \in \Theta \}$ as therein and $\xi _ { 1 } , \xi _ { 2 } , \ldots \overset { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$ . For any

$n \geq 1 , k \geq 1 , \tau > 0$ and $\delta \in ( 0 , 1 )$ it holds with probability at least $1 - \delta$ that for all $\phi \in \Phi$

$$
\begin{array}{l} \underbrace {\int_ {\mathcal {Z}} \mathbb {E} \left(\text {CrossEnt} \left[ \widehat {M} _ {k} (z , \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z , \cdot) \right]\right) \pi_ {\text {data}} (\mathrm{d} z)} _ {\text {similarity search expected cross - entropy}} \leq \underbrace {\widehat {R} _ {n} (\phi ; k , \tau)} _ {\text {empirical risk}} \\ + \underbrace {\frac {\text {const.}}{\tau} \sqrt {\frac {d _ {\Theta}}{n}} \min \left\{A \sqrt {\log (3) + 1} , \sqrt {\log (1 + 2 A) + 1} \right\}} _ {\text {complexity penalty}} \\ + \underbrace {\frac {\text {const.}}{\tau} \sqrt {\frac {\log \frac {2}{\delta}}{2 n}}} _ {\text {sample variability}} \\ + \underbrace {\frac {4}{\tau k} \mathbb {E} \left[ \left| \frac {\mathrm{d} M (Z , \cdot)}{\mathrm{d} \pi_ {\text {data}}} (Z ^ {\prime}) \right| ^ {2} \right] + \log \left(1 + \frac {e ^ {2 / \tau}}{k}\right)} _ {\text {finite k bias}}, \end{array}
$$

where $A : = e R b _ { \tau } ( { \bf { Z } } _ { 1 } , \ldots , { \bf { Z } } _ { n } ) C _ { \Phi } ; d _ { \Theta }$ , R and $C _ { \Phi }$ are as in $\mathrm { ( C ) } ;$ and $B _ { \tau } ( \cdot \cdot \cdot )$ is as in lemma 1.

This theorem illustrates the three contributions to the diference between similarity search expected cross entropy and the empirical risk $\widehat { R } _ { n } ( \phi , k , \tau )$ . The sample variability and complexity penalty terms arise from proposition 3 combined with the Rademacher complexity bound in proposition 5. The fi nite k bias term comes from proposition 2. The sample variability term does not depend on k and is $O ( n ^ { - 1 / 2 } )$ , the complexity penalty term stabilises as $k  \infty$ because of the averaging structure within $B _ { \tau } ( \mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n } )$ and hence is $O ( \bar { n } ^ { - 1 / 2 } )$ , and the bias term tends to zero as $k  \infty$ and is $O ( 1 / k )$ . As discussed in section 3.4, if using the DCL loss instead of InfoNCE, the $\log ( 1 + e ^ { 2 / \tau } / k )$ term disappears from the finite k bias. We leave it as an exercise to check that counterparts of proposition 3 and proposition 5 hold for the DCL risk, yielding complexity penalty and sample variability terms equal to those in theorem 1 up to numerical constants.

Regularisation from early stopping of optimisation algorithms. Our analysis has been algorithm free, in the sense that we have not considered any particular method for approximately minimising the InfoNCE risk. Never-the-less, if $\theta _ { 0 }$ in (C) is regarded as an initial point for a recursive optimisation algorithm (which optimises $\theta$ in order to approximately minimise the empirical risk associated with $\phi _ { \theta } )$ then R can be interpreted as a bound on the distance from $\theta _ { 0 }$ such an algorithm can move in some finite number of steps. We can see in proposition 5 that $R  0$ forces the Rademacher complexity to zero. In this sense, our results accommodate the idea that early stopping of an optimisation algorithm has a regularising efect.

Neural networks. In our analysis we have prioritised obtaining a clear and easily interpretable demon stration of the impact of $k , \tau ,$ etc. on generalisation. Assumption (C) serves this purpose. In $\operatorname { f a c t } ,$ assumption (C) is satisfied if ϕ is a fully connected multilayer perceptron with: Euclidean input domain; a 1-Lipschitz activation function such as ReLU; zero biases; weights constrained such that the product of spectral norms of the weight matrices is less than a constant; and θ comprises all entries of all the weight matrices of the network. In this situation $d _ { \Theta } \sim \mathrm { w i d t h ^ { 2 } \times d e p t h }$ . The resulting factor of $d _ { \Theta }$ in the bounds of proposition 5 could be ameliorated to some extent by scaling down the weight matrices (and hence $C _ { \Phi } )$ with network size, as is common practice, $\mathrm { e . g . }$ , [Yang et al., 2021]. However, for wide and/or deep networks, the explicit dependence on $d _ { \Theta }$ could make the bound in proposition 5 vacuous unless n is very large. Various techniques have been devised to obtain complexity bounds for neural networks which do not have explicit dependence on width and/or depth, $\mathrm { e . g . }$ , [Neyshabur et al., 2015, Bartlett et al., 2017, Golowich et al., 2018, Lei et al., 2023]. A potential topic for future research is to explore whether such techniques can be made to work together with the continuity bound in proposition 4. Another exciting direction is the notion of a path metric of a neural network, $\mathrm { e . g . , }$ [Gonon et al., 2025], which can help to bound complexity for a broad class of modern neural network architectures beyond the basic format of a multilayer perceptron and achieve scaling invariance.

Downstream classification. Whilst we have emphasised similarity search generalisation, the results of section 4 do not rely on any distributional assumptions other than the tuples $\mathbf { Z } _ { 1 } , \ldots , \mathbf { Z } _ { n }$ being i.i.d. As such, they are transferable to the classification setting of Saunshi et al. [2019], Lei et al. [2023].

## A Proofs and supporting results for section 2

Lemma 2. For any $k \geq 1 , b , c > 0$ and i.i.d. random variables X, $X _ { 1 } , \ldots , X _ { k }$ , each valued in the interval $[ - c , c ]$

$$
- \log \left(1 + \frac {e ^ {c}}{b k}\right) \leq \log (b \mathbb {E} [ e ^ {X} ]) - \mathbb {E} \left[ \log \left(\frac {1}{k} + \frac {b}{k} \sum_ {j = 1} ^ {k} e ^ {X _ {j}}\right) \right] \leq \frac {1}{8 k} (e ^ {2 c} - 1) ^ {2}.
$$

Proof. Denote $\begin{array} { r } { \widehat { \mu } : = k ^ { - 1 } \sum _ { j = 1 } ^ { k } e ^ { X _ { j } } } \end{array}$ and $\mu : = \mathbb { E } [ e ^ { X } ]$ . Note $\displaystyle \widehat \mu , \mu \in [ e ^ { - c } , e ^ { c } ]$ . With $f ( y ) : = \log ( 1 / k + y )$ we have:

$$
\log \left(b \mathbb {E} \left[ e ^ {X} \right]\right) - \mathbb {E} \left[ \log \left(\frac {1}{k} + \frac {b}{k} \sum_ {j = 1} ^ {k} e ^ {X _ {j}}\right) \right] = \log (b \mu) - f (b \mu) + f (b \mu) - \mathbb {E} [ f (b \widehat {\mu}) ].\tag{20}
$$

For the first diference on the r.h.s. of (20),

$$
0 \geq \log (b \mu) - f (b \mu) = \log (b \mu) - \log \left[ b \mu \left(1 + \frac {1}{k b \mu}\right) \right] \geq - \log \left(1 + \frac {e ^ {c}}{b k}\right).\tag{21}
$$

For the second diference on the r.h.s. of (20), by Taylor expansion of $y \mapsto f ( y )$ about $b \mu$ , there exists $\xi \in [ b e ^ { - c } , b e ^ { c } ]$ such that:

$$
\begin{array}{r} \log \left(\frac {1}{k} + b \widehat {\mu}\right) = \log \left(\frac {1}{k} + b \mu\right) + \frac {b (\widehat {\mu} - \mu)}{\frac {1}{k} + b \mu} - \frac {b ^ {2}}{2 (\frac {1}{k} + \xi) ^ {2}} (\widehat {\mu} - \mu) ^ {2} \\ \geq \log \left(\frac {1}{k} + b \mu\right) + \frac {b (\widehat {\mu} - \mu)}{\frac {1}{k} + b \mu} - \frac {1}{2 e ^ {- 2 c}} (\widehat {\mu} - \mu) ^ {2}. \end{array}
$$

Therefore, using $\mathbb { E } [ \widehat { \mu } ] = \mu$

$$
\mathbb {E} \left[ f (b \widehat {\mu}) \right] \geq f (b \mu) - \frac {1}{2 e ^ {- 2 c}} \mathbb {E} \left[ (\widehat {\mu} - \mu) ^ {2} \right].
$$

Using Popoviciu’s inequality on variances,

$$
\mathbb {E} \left[ (\widehat {\mu} - \mu) ^ {2} \right] = \frac {1}{k} \mathrm{Var} \left[ e ^ {X} \right] \leq \frac {1}{4 k} \left(e ^ {c} - e ^ {- c}\right) ^ {2}.
$$

Concavity of f and Jensen’s inequality implies $\mathbb { E } [ f ( b { \widehat { \mu } } ) ] \leq f ( b \mu )$ , so we have:

$$
0 \leq f (b \mu) - \mathbb {E} \left[ f (b \widehat {\mu}) \right] \leq \frac {1}{8 k} \left(e ^ {2 c} - 1\right) ^ {2}.\tag{22}
$$

The proof is completed by combining (21) and (22) with (20).

Proof of proposition 1. Fix any $z ^ { a } , z ^ { + } \in { \mathcal { Z } }$ . Let $Z _ { 1 } ^ { - } , \ldots , Z _ { k } ^ { - } \overset { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$ . Applying lemma 2 with $b =$ $e ^ { - \langle \phi ( z ^ { a } ) , \phi ( z ^ { + } ) \rangle / \tau } , c = 1 / \tau , X _ { j } = e ^ { \langle \phi ( z ^ { a } ) , \phi ( Z _ { j } ^ { - } ) \rangle / \tau }$ , and using $b \geq e ^ { - 1 / \tau }$

$$
\begin{array}{l} - \log \left(1 + \frac {e ^ {2 / \tau}}{k}\right) \\ \leq \log \left(\int_ {\mathcal {Z}} e ^ {\langle \phi (z ^ {a}), \phi (z ^ {-}) - \phi (z ^ {+}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {-})\right) - \mathbb {E} \left[ \log \left(\frac {1}{k} + \frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (z ^ {a}), \phi (Z _ {j} ^ {-}) - \phi (z ^ {+}) \rangle / \tau}\right) \right] \end{array}
$$

Now let $Z ^ { a } ~ \sim ~ \pi _ { \mathrm { d a t a } }$ and $Z ^ { + } \ \sim \ M ( Z ^ { a } , \cdot )$ , independent of $Z _ { 1 } ^ { - } , \ldots , Z _ { k } ^ { - }$ . The proof is completed by substituting $Z ^ { a }$ and $Z ^ { + }$ in place of $z ^ { a }$ and $z ^ { + }$ respectively, and taking expectation. □

Lemma 3. For any two probability measures $\nu , \pi \in \mathcal { P } ( \mathcal { Z } )$ such that $\nu \ll \pi$ , any measurable function $f : \mathcal { Z } \to [ - 1 , 1 ]$ ], any $m \geq 1$ and $\xi _ { 1 } , \ldots , \xi _ { m } \stackrel { \mathrm { i i d } } { \sim } \pi$

$$
\left| \mathbb {E} \left[ \frac {\sum_ {j = 1} ^ {m} f (\xi_ {j}) \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi_ {j})}{\sum_ {j = 1} ^ {m} \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi_ {j})} \right] - \mathbb {E} _ {\nu} [ f (\xi) ] \right| \leq \frac {4}{m} \mathbb {E} _ {\pi} \left[ \left| \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi) \right| ^ {2} \right],
$$

where $\mathbb { E } _ { \nu }$ and $\mathbb { E } _ { \pi }$ denote expectation with respectively $\xi \sim \nu$ and $\xi \sim \pi$

Proof. Define $A : = m ^ { - 1 } \textstyle \sum _ { j = 1 } ^ { m }$ <sup>dν</sup><sub>dπ</sub> $\left( \xi _ { j } \right) \left( f ( \xi _ { j } ) - \mathbb { E } _ { \nu } [ f ( \xi ) ] \right)$ and $B : = m ^ { - 1 } \textstyle \sum _ { j = 1 } ^ { m }$ <sup>dν</sup><sub>dπ</sub> $( \xi _ { j } )$ . Then

$$
\frac {\sum_ {j = 1} ^ {m} f (\xi_ {j}) \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi_ {j})}{\sum_ {j = 1} ^ {m} \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi_ {j})} - \mathbb {E} _ {\nu} [ f (\xi) ] = \frac {A}{B},
$$

and the quantity we seek to bound is $\textstyle | \mathbb { E } [ A / B ] |$ .

We have:

$$
{\frac {1}{B}} = 2 - B + {\frac {(B - 1) ^ {2}}{B}},
$$

and using the fact that $\mathbb { E } [ A ] = 0$

$$
\mathbb {E} \left[ \frac {A}{B} \right] = - \mathbb {E} [ A B ] + \mathbb {E} \left[ \frac {A}{B} (B - 1) ^ {2} \right].\tag{23}
$$

Using the facts that $\xi _ { 1 } , . . . \xi _ { m } \stackrel { \mathrm { i i d } } { \sim } \pi$ and $\begin{array} { r } { \mathbb { E } \left[ \frac { \mathrm { d } \nu } { \mathrm { d } \pi } ( \xi _ { j } ) f ( \xi _ { j } ) \right] = \mathbb { E } _ { \nu } [ f ( \xi ) ] } \end{array}$ , the expectation of every cross term in the product-of-sums AB is zero, so we have:

$$
\mathbb {E} [ A B ] = \mathbb {E} \left[ \frac {1}{m ^ {2}} \sum_ {j = 1} ^ {m} \left| \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi_ {j}) \right| ^ {2} (f (\xi_ {j}) - \mathbb {E} _ {\nu} [ f (\xi) ]) \right],
$$

and since $| f ( z ) | \leq 1$ for all $z \in { \mathcal { Z } }$ , we obtain:

$$
| \mathbb {E} [ A B ] | \leq \frac {2}{m} \mathbb {E} _ {\pi} \left[ \left| \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi) \right| ^ {2} \right].\tag{24}
$$

Again using $| f ( z ) | \leq 1$ we have $| A | \le 2 B$ , hence:

$$
\left| \mathbb {E} \left[ \frac {A}{B} (B - 1) ^ {2} \right] \right| \leq 2 \mathbb {E} [ (B - 1) ^ {2} ] = \frac {2}{m} \mathbb {E} _ {\pi} \left[ \left| \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi) - 1 \right| ^ {2} \right] \leq \frac {2}{m} \mathbb {E} _ {\pi} \left[ \left| \frac {\mathrm{d} \nu}{\mathrm{d} \pi} (\xi) \right| ^ {2} \right].\tag{25}
$$

The proof is completed by applying the triangle inequality to (23), then applying the bounds (24) and (25). □

Proof of proposition 2. Define:

$$
\widetilde {R} (\phi ; k, \tau) := \mathbb {E} \left[ \log \left(\sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}), \phi (Z _ {j} ^ {-}) - \phi (Z ^ {+}) \rangle / \tau}\right) \right] - \log k,\tag{26}
$$

where $Z ^ { a } \sim \pi _ { \mathrm { d a t a } } , Z ^ { + } \sim M ( Z ^ { a } , \cdot )$ and $Z _ { 1 } ^ { - } , \ldots , Z _ { k } ^ { - } \overset { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$ , and decompose:

$$
\begin{array}{r l} & {\int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M} _ {k} (z, \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z, \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z) - R (\phi ; k, \tau)} \\ & {= \int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M} _ {k} (z, \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z, \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau)} \\ & {+ \widetilde {R} (\phi ; k, \tau) - R (\phi ; k, \tau).} \end{array}\tag{27}
$$

(28)

In order to bound the diference in (27), let us first write out an expression for $\widetilde { R } ( \phi ; k , \tau )$ , as follows:

$$
\begin{array}{l} \widetilde {R} (\phi ; k, \tau) \\ = - \frac {1}{\tau} \mathbb {E} [ \langle \phi (Z ^ {a}), \phi (Z ^ {+}) \rangle ] + \mathbb {E} \left[ \log \left(\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}), \phi (Z _ {j} ^ {-}) \rangle / \tau}\right) \right] \\ = - \frac {1}{\tau} \int_ {\mathcal {Z}} \int_ {\mathcal {Z}} \langle \phi (z), \phi (z ^ {+}) \rangle M (z, \mathrm{d} z ^ {+}) \pi_ {\text {data}} (\mathrm{d} z) + \mathbb {E} \left[ \log \left(\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}), \phi (Z _ {j} ^ {-}) \rangle / \tau}\right) \right], \end{array}\tag{29}
$$

where the first equality is just rearrangement of (26), the second equality holds since $Z ^ { a } \sim \pi _ { \mathrm { d a t a } }$ and $Z ^ { + } \sim M ( Z ^ { a } , \cdot )$

Recalling the definitions:

$$
\widehat {Q} _ {\tau , k} ^ {\phi} (z, \mathrm{d} z ^ {\prime}) := \frac {\sum_ {j = 1} ^ {k} e ^ {\langle \phi (z) , \phi (\xi_ {j}) \rangle / \tau} \delta_ {\xi_ {j}} (\mathrm{d} z ^ {\prime})}{\sum_ {j = 1} ^ {k} e ^ {\langle \phi (z) , \phi (\xi_ {j}) \rangle / \tau}}, \qquad \widehat {M} _ {k} (z, \mathrm{d} z ^ {\prime}) := \frac {\sum_ {j = 1} ^ {k} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {j}) \delta_ {\xi_ {j}} (\mathrm{d} z ^ {\prime})}{\sum_ {j = 1} ^ {k} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {j})}.
$$

and $\begin{array} { r } { \widehat { \pi } _ { \mathrm { d a t a } , k } : = k ^ { - 1 } \sum _ { j = 1 } ^ { k } \delta _ { \xi _ { j } } } \end{array}$ , where $\xi _ { 1 } , \dots , \xi _ { k } \overset { \mathrm { i i d } } { \sim } \pi _ { \mathrm { d a t a } }$ , we have:

$$
\frac {\mathrm{d} \widehat {Q} _ {\tau , k} ^ {\phi} (z , \cdot)}{\mathrm{d} \widehat {\pi} _ {\mathrm{data} , k}} (z ^ {\prime}) = \frac {e ^ {\langle \phi (z) , \phi (z ^ {\prime}) \rangle / \tau}}{\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (z) , \phi (\xi_ {j}) \rangle / \tau}} \quad \text {for} \quad z ^ {\prime} \in \{\xi_ {1}, \ldots , \xi_ {k} \},\tag{30}
$$

noting $\{ \xi _ { 1 } , \ldots , \xi _ { k } \}$ is a subset of, or equal to, the support of $\widehat { M _ { k } } ( z , \cdot )$ . Therefore

$$
\begin{array}{r l} & {\int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M} _ {k} (z, \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z, \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z)} \\ & {= \int_ {\mathcal {Z}} \mathbb {E} \left[ - \int_ {\mathcal {Z}} \log \frac {\mathrm{d} \widehat {Q} _ {\tau , k} ^ {\phi} (z , \cdot)}{\mathrm{d} \widehat {\pi} _ {\mathrm{data} , k}} (z ^ {\prime}) \widehat {M} _ {k} (z, \mathrm{d} z ^ {\prime}) \right] \pi_ {\mathrm{data}} (\mathrm{d} z)} \\ & {= - \frac {1}{\tau} \int_ {\mathcal {Z}} \mathbb {E} \left[ \int_ {\mathcal {Z}} \langle \phi (z), \phi (z ^ {\prime}) \rangle \widehat {M} _ {k} (z, \mathrm{d} z ^ {\prime}) \right] \pi_ {\mathrm{data}} (\mathrm{d} z) + \mathbb {E} \left[ \log \left(\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}), \phi (\xi_ {j}) \rangle / \tau}\right) \right]} \\ & {= - \frac {1}{\tau} \int_ {\mathcal {Z}} \mathbb {E} \left[ \frac {\sum_ {j = 1} ^ {k} \langle \phi (z) , \phi (\xi_ {j}) \rangle}{\sum_ {j = 1} ^ {k} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {j})} \right] \pi_ {\mathrm{data}} (\mathrm{d} z) + \mathbb {E} \left[ \log \left(\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}), \phi (Z _ {j} ^ {-}) \rangle / \tau}\right) \right],} \end{array}
$$

where the first equality holds by definition of CrossEnt $\left[ \widehat { M } _ { k } ( z , \cdot ) \lVert \widehat { Q } _ { \tau , k } ^ { \phi } ( z , \cdot ) \right]$ ; the second equality is obtained by substituting in (30); the third equality holds by substituting in the definition of $\widehat { M _ { k } }$ and using the fact that $\xi _ { 1 } , \ldots , \xi _ { k }$ and $Z _ { 1 } ^ { - } , \ldots , Z _ { k } ^ { - }$ are identically distributed. Comparing with (29), we find:

$$
\begin{array}{l} \left| \int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M} _ {k} (z, \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z, \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau) \right| \\ \leq \frac {1}{\tau} \int_ {Z} \left| \mathbb {E} \left[ \frac {\sum_ {j = 1} ^ {k} \langle \phi (z) , \phi (\xi_ {j}) \rangle \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {j})}{\sum_ {j = 1} ^ {k} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {j})} \right] - \int_ {\mathcal {Z}} \langle \phi (z), \phi (z ^ {+}) \rangle M (z, \mathrm{d} z ^ {+}) \right| \pi_ {\mathrm{data}} (\mathrm{d} z). \end{array}
$$

In order to control the integrand for any fixed z, we apply lemma 3 with $\nu : = M ( z , \cdot ) , \pi : = \pi _ { \mathrm { d a t a } }$ and $f ( \xi ) : = \langle \phi ( z ) , \phi ( \xi ) \rangle$ , which satisfies $| f ( \xi ) | \le 1$ as required since $\phi \in \cal B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ , yielding:

$$
\left| \mathbb {E} \left[ \frac {\sum_ {j = 1} ^ {k} \langle \phi (z) , \phi (\xi_ {j}) \rangle \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {j})}{\sum_ {j = 1} ^ {k} \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (\xi_ {j})} \right] - \int_ {\mathcal {Z}} \langle \phi (z), \phi (z ^ {+}) \rangle M (z, \mathrm{d} z ^ {+}) \right| \leq \frac {4}{k} \int_ {\mathcal {Z}} \left| \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (z ^ {\prime}) \right| ^ {2} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {\prime}),
$$

and in turn:

$$
\begin{array}{r l} & {\left| \int_ {\mathcal {Z}} \mathbb {E} \left(\mathrm{CrossEnt} \left[ \widehat {M} _ {k} (z, \cdot) \| \widehat {Q} _ {\tau , k} ^ {\phi} (z, \cdot) \right]\right) \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau) \right|} \\ & {\qquad \leq \frac {4}{k \tau} \int_ {\mathcal {Z}} \int_ {\mathcal {Z}} \left| \frac {\mathrm{d} M (z , \cdot)}{\mathrm{d} \pi_ {\mathrm{data}}} (z ^ {\prime}) \right| ^ {2} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {\prime}) \pi_ {\mathrm{data}} (\mathrm{d} z).} \end{array}
$$

This completes our treatment of the diference in (27). For the diference in (28), recalling the definitions of $R ( \phi ; k , \tau )$ and $\widetilde { R } ( \phi ; k , \tau )$ in (2) and (26), we have:

$$
\begin{array}{l} 0 \leq R (\phi ; k, \tau) - \widetilde {R} (\phi ; k, \tau) \\ = \mathbb {E} \left[ \log \left(1 + \frac {1}{\sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z ^ {a}) , \phi (Z _ {j} ^ {-}) - \phi (Z ^ {+}) \rangle / \tau}}\right) \right] \\ \leq \log \left(1 + \frac {e ^ {2 / \tau}}{k}\right), \end{array}
$$

where the first inequality uses monotonicity of log, the equality uses log $( 1 / k + c ) - \log ( c ) = \log ( 1 + 1 / k c )$ 2 and the second inequality uses $\phi \in { \cal B } ( \mathcal { Z } , \mathbb { S } ^ { \hat { d } - 1 } )$

Having thus obtained bounds on the absolute values of the diferences in (27)-(28), the proof is completed by applying the triangle inequality there.

Lemma 4. For any $\phi \in \cal B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ , $k \geq 1$ and $\tau > 0$

$$
\int_ {\mathcal {Z}} \mathrm{CrossEnt} [ M (z, \cdot) \| Q _ {\tau} ^ {\phi} (z, \cdot) ] \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau) \leq \frac {2}{\tau}.
$$

Proof.

$$
\begin{array}{r l} & {\int_ {\mathcal {Z}} \mathrm{CrossEnt} [ M (z, \cdot) \| Q _ {\tau} ^ {\phi} (z, \cdot) ] \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau)} \\ & {= \mathbb {E} \left[ \log \int_ {\mathcal {Z}} e ^ {\langle \phi (Z), \phi (z ^ {-}) - \phi (Z ^ {+}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {-}) \right] - \mathbb {E} \left[ \log \frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z), \phi (Z _ {j} ^ {-}) - \phi (Z ^ {+}) \rangle / \tau} \right]} \\ & {= \mathbb {E} \left[ \log \int_ {\mathcal {Z}} e ^ {\langle \phi (Z), \phi (z ^ {-}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {-}) \right] - \mathbb {E} \left[ \log \frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z), \phi (Z _ {j} ^ {-}) \rangle / \tau} \right]} \\ & {= \mathbb {E} \left[ \log \frac {\int_ {\mathcal {Z}} e ^ {\langle \phi (Z) , \phi (z ^ {-}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {-})}{\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z) , \phi (Z _ {j} ^ {-}) \rangle / \tau}} \right] \leq \log \frac {e ^ {1 / \tau}}{e ^ {- 1 / \tau}} = \frac {2}{\tau},} \end{array}
$$

where the inequality holds since $\| \phi ( z ) \| _ { 2 } = 1$ for all $z .$

## A.1 Calculations for section 3.4

The bound in (12) is obtained as part of the proof of proposition 2, see (27) in particular. To check the bound in (11), write out the diference:

$$
\begin{array}{r l} & {\int_ {\mathcal {Z}} \mathrm{CrossEnt} [ M (z, \cdot) \| Q _ {\tau} ^ {\phi} (z, \cdot) ] \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau)} \\ & {= - \frac {1}{\tau} \int_ {\mathcal {Z}} \langle \phi (z), \phi (z ^ {+}) \rangle \pi_ {\mathrm{data}} (\mathrm{d} z) M (z, \mathrm{d} z ^ {+}) + \int_ {\mathcal {Z}} \log \left[ \int_ {\mathcal {Z}} e ^ {\langle \phi (z), \phi (z ^ {-}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {-}) \right] \pi_ {\mathrm{data}} (\mathrm{d} z)} \\ & {\quad + \frac {1}{\tau} \int_ {\mathcal {Z}} \langle \phi (z), \phi (z ^ {+}) \rangle \pi_ {\mathrm{data}} (\mathrm{d} z) M (z, \mathrm{d} z ^ {+}) - \mathbb {E} \left[ \log \left(\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (Z), \phi (Z _ {j} ^ {-}) \rangle / \tau}\right) \right]} \\ & {= \int_ {Z} \log \left[ \int_ {\mathcal {Z}} e ^ {\langle \phi (z), \phi (z ^ {-}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {-}) \right] - \mathbb {E} \left[ \log \left(\frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (z), \phi (Z _ {j} ^ {-}) \rangle / \tau}\right) \right] \pi_ {\mathrm{data}} (\mathrm{d} z)} \\ & {= \int_ {Z} \log \mu (z) - \mathbb {E} [ \log \widehat {\mu} (z) ] \pi_ {\mathrm{data}} (\mathrm{d} z),} \end{array}\tag{31}
$$

where the final equality holds with the shorthand notation:

$$
\mu (z) := \int_ {\mathcal {Z}} e ^ {\langle \phi (z), \phi (z ^ {-}) \rangle / \tau} \pi_ {\mathrm{data}} (\mathrm{d} z ^ {-}), \quad \widehat {\mu} (z) := \frac {1}{k} \sum_ {j = 1} ^ {k} e ^ {\langle \phi (z), \phi (Z _ {j} ^ {-}) \rangle / \tau}.
$$

By Jensen’s inequality, log $\mu ( z ) \geq \mathbb { E } [ \log \widehat { \mu } ( z ) ]$ . The upper-bound in (11) is derived by making a Taylor expansion of $y \mapsto \log ( y )$ similarly to as in the proof of lemma 2. The details are omitted, but in order to perform calculations for a specific example, consider the Taylor expansion up to third order (with z fixed):

$$
\log \widehat {\mu} (z) = \log \mu (z) + \frac {\widehat {\mu} (z) - \mu (z)}{\mu (z)} - \frac {(\widehat {\mu} (z) - \mu (z)) ^ {2}}{2 \mu (z) ^ {2}} + \frac {(\widehat {\mu} (z) - \mu (z)) ^ {3}}{3 \xi^ {3}},
$$

for some $\xi$ on the line segment between $\mu ( z )$ and $\widehat { \mu } ( z )$

Using $\mu ( z ) \vee \widehat { \mu } ( z ) \le e ^ { 1 / \tau }$ , we thus have:

$$
\log \mu (z) - \mathbb {E} [ \log \widehat {\mu} (z) ] \geq \frac {\mathbb {E} [ (\widehat {\mu} (z) - \mu (z)) ^ {2} ]}{2 \mu (z) ^ {2}} - \frac {\mathbb {E} [ | \widehat {\mu} (z) - \mu (z) | ^ {3} ]}{3 e ^ {3 / \tau}}.
$$

Since $Z _ { 1 } ^ { - } , \ldots , Z _ { k } ^ { - }$ are i.i.d., the Marcinkiewicz–Zygmund inequality gives $\mathbb { E } [ | \widehat { \mu } ( z ) - \mu ( z ) | ^ { 3 } ] = O ( k ^ { - 3 / 2 } )$ uniformly in z, and by direct calculation,

$$
\mathbb {E} [ (\widehat {\mu} (z) - \mu (z)) ^ {2} ] = \frac {1}{n} \mathrm{Var} [ e ^ {\langle \phi (z), \phi (Z _ {1} ^ {-}) \rangle / \tau} ],
$$

where $Z _ { 1 } ^ { - } \sim \pi _ { \mathrm { d a t a } }$

This shows that

$$
k \log \mu (z) - k \mathbb {E} [ \log \widehat {\mu} (z) ] \geq \frac {1}{2} \frac {\operatorname{Var} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ]}{\mathbb {E} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ] ^ {2}} - O (k ^ {- 1 / 2}),\tag{32}
$$

as $k \to \infty$

## A.2 An example

Now let us construct an example for which we shall lower bound $\mathrm { V a r } [ e ^ { \langle \phi ( z ) , \phi ( Z _ { 1 } ^ { - } ) \rangle / \tau } ] / \mathbb { E } [ e ^ { \langle \phi ( z ) , \phi ( Z _ { 1 } ^ { - } ) \rangle / \tau } ] ^ { 2 }$ The idea of the construction is to make ${ | \langle \phi ( z ) , \phi ( Z _ { 1 } ^ { - } ) \rangle } |$ close to zero with probability close to 1, and otherwise $\langle \phi ( z ) , \phi ( Z _ { 1 } ^ { - } ) \rangle = 1$

Assume $e ^ { 1 / \tau } > 2$ and define

$$
\epsilon := \frac {1}{\lceil e ^ {1 / \tau} - 1 \rceil},\tag{33}
$$

so that $\epsilon \in ( 0 , 1 )$ . Let $\delta \in ( 0 , 1 )$ and let $d ( \delta )$ be large enough that there exist $1 / \epsilon$ vectors in $\mathbb { S } ^ { d - 1 }$ which are δ-orthogonal, $\mathrm { i . e . }$ , vectors $u _ { 1 } , \ldots , u _ { 1 / \epsilon }$ such that for $i \neq j , | \langle u _ { i } , u _ { j } \rangle | \leq \delta$ . Over the course of the following construction we shall consider taking $\delta  0$ and when we do it will be automatically assumed that d grows suitably fast as δ shrinks that $1 / \epsilon \delta \cdot$ -orthogonal vectors exist $( d = 1 / \epsilon$ is suficient, since in that case the existence of $1 / \epsilon$ orthogonal vectors in $\mathbb { S } ^ { d - 1 }$ is trivial, but we allow for δ-orthogonality rather than strict orthogonality to emphasise that $d = 1 / \epsilon$ is not necessary for the construction).

Suppose that $\pi _ { \mathrm { d a t a } }$ and ϕ are such that for $Z \sim \pi _ { \mathrm { d a t a } } ,$ the random vector $\phi ( Z )$ is uniformly distributed on the set $\{ u _ { 1 } , \ldots , u _ { 1 / \epsilon } \}$ . It follows that for any $z \in { \mathcal { Z } }$ and $Z _ { j } ^ { - } \ \sim \ \pi _ { \mathrm { d a t a } } , \ \langle \phi ( z ) , \phi ( Z _ { j } ^ { - } ) \rangle = 1$ with probability ϵ, and with probability $1 - \epsilon , \langle \phi ( z ) , \phi ( Z _ { i } ^ { - } ) \rangle = X _ { 0 }$ for some random variable $X _ { 0 }$ such that $| X _ { 0 } | \leq \delta .$

We have:

$$
\frac {\mathrm{Var} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ]}{\mathbb {E} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ] ^ {2}} = \frac {\epsilon (1 - \epsilon) \left(e ^ {1 / \tau} - \mathbb {E} [ e ^ {X _ {0} / \tau} ]\right) ^ {2} + (1 - \epsilon) \mathrm{Var} [ e ^ {X _ {0} / \tau} ]}{\left(\epsilon e ^ {1 / \tau} + (1 - \epsilon) \mathbb {E} [ e ^ {X _ {0} / \tau} ]\right) ^ {2}}.
$$

As $\delta  0$ (with d increasing as necessary for the required number of δ-orthogonal vectors to exist), the r.h.s. of the above tends to:

$$
\frac {\epsilon (1 - \epsilon) \left(e ^ {1 / \tau} - 1\right) ^ {2}}{\left(1 + \epsilon (e ^ {1 / \tau} - 1)\right) ^ {2}}.
$$

Therefore for any $a \in ( 0 , 1 )$ we can choose $\delta > 0$ small enough that:

$$
\frac {\mathrm{Var} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ]}{\mathbb {E} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ] ^ {2}} \geq (1 - a) \frac {\epsilon (1 - \epsilon) \left(e ^ {1 / \tau} - 1\right) ^ {2}}{\left(1 + \epsilon (e ^ {1 / \tau} - 1)\right) ^ {2}}.
$$

Using $e ^ { 1 / \tau } > 2$ and (33) we have:

$$
\frac {1}{2} \leq \frac {e ^ {1 / \tau} - 1}{e ^ {1 / \tau}} \leq \epsilon (e ^ {1 / \tau} - 1) \leq 1,
$$

and so:

$$
\begin{array}{r l} & {\frac {\mathrm{Var} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ]}{\mathbb {E} [ e ^ {\langle \phi (z) , \phi (Z _ {1} ^ {-}) \rangle / \tau} ] ^ {2}} \geq (1 - a) \frac {(1 - \epsilon) (e ^ {1 / \tau} - 1) ^ {2}}{4 e ^ {1 / \tau}}} \\ & {\qquad \geq \frac {(1 - a)}{8} (e ^ {1 / \tau} - 1 - \frac {e ^ {1 / \tau} - 1}{\lceil e ^ {1 / \tau} - 1 \rceil})} \\ & {\qquad \geq \frac {(1 - a)}{8} (e ^ {1 / \tau} - 2).} \end{array}
$$

Combining with (31) and (32) we find that for this example:

$$
\liminf _ {k \to \infty} \left(\int_ {\mathcal {Z}} \mathrm{CrossEnt} [ M (z, \cdot) \| Q _ {\tau} ^ {\phi} (z, \cdot) ] \pi_ {\mathrm{data}} (\mathrm{d} z) - \widetilde {R} (\phi ; k, \tau)\right) \geq \frac {(1 - a)}{1 6} (e ^ {1 / \tau} - 2),
$$

from which (13) follows by combining with (12).

## B Proofs and supporting results for section 4

Lemma 5. For any $k \geq 1 , \tau > 0 , \mathbf { z } \in \mathcal { Z } ^ { 2 + k }$ , and $\phi , \phi ^ { \prime } \in B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$

$$
| \ell (\phi , \mathbf {z}, k, \tau) - \ell (\phi^ {\prime}, \mathbf {z}, k, \tau) | \leq \frac {4}{\tau}.
$$

Proof. Since $\| \phi \| _ { 2 , \infty } = 1$ , we have:

$$
\log \left(\frac {1}{k} + e ^ {- 2 / \tau}\right) \leq \ell (\phi , \mathbf {z}, k, \tau) \leq \log \left(\frac {1}{k} + e ^ {2 / \tau}\right),
$$

with the same inequality holding with ϕ replaced by $\phi ^ { \prime }$ . The claim of the lemma then follows by bounding:

$$
\begin{array}{l} | \ell (\phi , \mathbf {z}, k, \tau) - \ell (\phi^ {\prime}, \mathbf {z}, k, \tau) | \leq \log \left(\frac {1 + k e ^ {2 / \tau}}{1 + k e ^ {- 2 / \tau}}\right) \\ \qquad = \log \left(e ^ {2 / \tau} \left[ \frac {1 + k e ^ {2 / \tau}}{e ^ {2 / \tau} + k} \right]\right) \\ \qquad \leq \log \left(e ^ {2 / \tau} \left[ \frac {e ^ {2 / \tau} + k e ^ {2 / \tau}}{1 + k} \right]\right) = \frac {4}{\tau}, \end{array}
$$

where the inequality uses $e ^ { 2 / \tau } \geq 1$

Proof of proposition 3. Noting that $R ( \phi ; k , \tau ) = \mathbb { E } \left[ \widehat { R } _ { n } ( \phi ; k , \tau ) \right]$ , proposition 3 is almost a direct appli cation of [Mohri et al., 2018, Thm. 3.3]. The latter theorem applies to sample averages of functions with co-domain [0, 1], whereas in the present setting we have lower and upper bounds:

$$
a := \log \left(\frac {1}{k} + e ^ {- 2 / \tau}\right) \leq \ell (\phi , \mathbf {z} _ {i}, k, \tau) \leq \log \left(\frac {1}{k} + e ^ {2 / \tau}\right) =: b,
$$

which hold for any $\phi \in B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ and any $\mathbf { z } _ { i } \in \mathcal { Z } ^ { 2 + k }$

To obtain a function with co-domain [0, 1] denote:

$$
g (\phi , \mathbf {z} _ {i}) := \frac {\ell (\phi , \mathbf {z} _ {i} , k , \tau) - a}{b - a},
$$

where the dependence on $\tau , k$ is hidden for ease of presentation. An application of [Mohri et al., 2018, Thm. 3.3] to $\scriptstyle n ^ { - 1 } \sum _ { i = 1 } ^ { n } g ( \phi , \mathbf { Z } _ { i } )$ gives that with probability at least $1 - \delta$ , for all $\phi \in \Phi$

$$
\mathbb {E} [ g (\phi , \mathbf {Z} _ {1}) ] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} g (\phi , \mathbf {Z} _ {i}) + 2 \mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} g (\phi , \mathbf {Z} _ {i}) \bigg |   \mathbf {Z} _ {1}, \ldots , \mathbf {Z} _ {n} \right] + 3 \sqrt {\frac {\log \frac {2}{\delta}}{2 n}}.\tag{34}
$$

Since Rademacher complexity of a class of functions is invariant to adding a constant scalar to every member of the class, and factorising out $b - a$ , we have:

$$
\mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} g (\phi , \mathbf {Z} _ {i})   \bigg |   \mathbf {Z} _ {1}, \ldots , \mathbf {Z} _ {n} \right] = \frac {1}{b - a} \mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {Z} _ {i}, k, \tau)   \bigg |   \mathbf {Z} _ {1}, \ldots , \mathbf {Z} _ {n} \right].
$$

Substituting into (34), multiplying both sides of the inequality by $b - a$ , adding a to both sides and recalling the definitions (2) and (1) gives:

$$
R (\phi ; k, \tau) \leq \widehat {R} _ {n} (\phi ; k, \tau) + 2 \mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {Z} _ {i}, k, \tau)   \bigg |   \mathbf {Z} _ {1}, \ldots , \mathbf {Z} _ {n} \right] + 3 (b - a) \sqrt {\frac {\log \frac {2}{\delta}}{2 n}}.
$$

The proof is completed by upper-bounding $b - a$ as in the proof of lemma 5.

## B.1 Supporting results and proof of proposition 4

A Gˆateaux derivative can be thought of as a generalisation of the directional derivative in Euclidean space. We shall consider Gˆateaux derivatives of functionals mapping $B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ into $\mathbb { R }$ . For such a functional, say $H : B ( \mathcal { Z } , \mathbb { R } ^ { d } ) \to \mathbb { R }$ , the Gˆateaux derivative of H at $f \in B ( \mathcal { \vec { Z } } , \mathbb { R } ^ { d } )$ in direction $\eta \in B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ is the limit (if it exists):

$$
\delta H (f) [ \eta ] := \lim _ {\epsilon \to 0} \frac {H (f + \epsilon \eta) - H (f)}{\epsilon}.
$$

To prepare for the proof of proposition $^ { 4 , }$ we need the following definitions. For any $\tau > 0 , x , y \in \mathcal { Z }$ and $\mu \in \mathcal { P } ( \mathcal { Z } )$ , define the functionals $H _ { \tau } ( \cdot , x , y , \mu ) : \mathcal { B } ( \mathcal { Z } , \mathbb { R } ^ { d } ) \to $ <sup>R</sup> and $G _ { \tau } ( \cdot , \cdot , x , y , \mu ) : \mathcal { B } ( \mathcal { Z } , \mathbb { R } ^ { d } ) \times$ $B ( \mathcal { Z } , \mathbb { R } ^ { d } )  \bar { \mathbb { R } ^ { d } }$ •

$$
H _ {\tau} (f, x, y, \mu) := \log \int_ {\mathcal {Z}} e ^ {\langle f (x), f (z) - f (y) \rangle / \tau} \mu (\mathrm{d} z).\tag{35}
$$

$$
G _ {\tau} (f, \eta , \mu , x) := \frac {\int_ {\mathcal {Z}} \eta (z) e ^ {\langle f (x) , f (z) \rangle / \tau} \mu (\mathrm{d} z)}{\int_ {\mathcal {Z}} e ^ {\langle f (x) , f (z) \rangle / \tau} \mu (\mathrm{d} z)},\tag{36}
$$

where in the numerator of (36) the vector-valued function η is integrated elementwise.

Lemma 6. For any τ > 0, x, y ∈ Z and $\mu \in \mathcal P ( \mathcal Z )$ , the Gˆateaux derivative of the functional $H _ { \tau } ( \cdot , x , y , \mu )$ at a point $f \in \mathcal { B } ( \mathcal { Z } , \mathbb { R } ^ { d } )$ , in direction $\eta \in B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ , is:

$$
\delta H _ {\tau} (f, x, y, \mu) [ \eta ] := \frac {1}{\tau} \left[ \langle \eta (x), G _ {\tau} (f, f, x, \mu) - f (y) \rangle + \langle f (x), G _ {\tau} (f, \eta , \mu , x) - \eta (y) \rangle \right].
$$

Proof. For brevity throughout the proof we write “derivative” instead of “Gˆateaux derivative”. From (35) we have:

$$
H _ {\tau} (f, x, y, \mu) = - \frac {1}{\tau} \left\langle f (x), f (y) \right\rangle + \log \int_ {\mathcal {Z}} e ^ {\left\langle f (x), f (z) \right\rangle / \tau} \mu (\mathrm{d} z).\tag{37}
$$

Considering the first term on the l.h.s. of (37), for any $\epsilon > 0$

$$
\frac {1}{\tau} \left\langle f (x) + \epsilon \eta (x), f (y) + \epsilon \eta (y) \right\rangle - \frac {1}{\tau} \left\langle f (x), f (y) \right\rangle = \frac {\epsilon}{\tau} \left[ \left\langle f (x), \eta (y) \right\rangle + \left\langle \eta (x), f (y) \right\rangle \right] + \frac {\epsilon^ {2}}{\tau} \left\langle \eta (x), \eta (y) \right\rangle ,
$$

then using Cauchy-Schwartz and $\| \eta \| _ { 2 , \infty } < \infty$ , the derivative of $\begin{array} { r } { f \mapsto \frac { 1 } { \tau } \left. f ( x ) , f ( y ) \right. } \end{array}$ ⟩ in direction η is:

$$
\lim _ {\epsilon \to 0} \frac {\frac {1}{\tau} \left\langle f (x) + \epsilon \eta (x) , f (y) + \epsilon \eta (y) \right\rangle - \frac {1}{\tau} \left\langle f (x) , f (y) \right\rangle}{\epsilon} = \frac {1}{\tau} \left[ \left\langle f (x), \eta (y) \right\rangle + \left\langle \eta (x), f (y) \right\rangle \right].\tag{38}
$$

To find the derivative of the second term on the r.h.s. of (37) we use the chain rule. So first consider the functional $f \mapsto \Gamma ( f ) : = e ^ { \langle f ( x ) , f ( z ) \rangle / \tau }$ (where dependence on τ, x and z is suppressed from the notation). Using (38) with y replaced by z, the derivative of Γ at $f$ in direction η is:

$$
\delta \Gamma (f) [ \eta ] := \frac {1}{\tau} \left[ \langle f (x), \eta (z) \rangle + \langle \eta (x), f (z) \rangle \right] e ^ {\langle f (x), f (z) \rangle / \tau}.\tag{39}
$$

Our next objective is to show that the derivative of $\begin{array} { r } { f \mapsto \int _ { \mathcal { Z } } e ^ { \langle f ( x ) , f ( z ) \rangle / \tau } \mu ( \mathrm { d } z ) } \end{array}$ is given by (39) with z integrated out under $\mu .$ For this purpose we seek to apply the dominated convergence theorem, as follows. Fix any $\epsilon \in ( 0 , 1 ] , f \in B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ and $\eta \in B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ . By the mean value theorem there exists $c \in [ 0 , 1 ]$ such that by evaluating $\Gamma ( \cdot ) [ \epsilon \eta ]$ at the point: $( 1 - c ) f + c ( f + \epsilon \eta ) = f + \epsilon c \eta ,$

$$
\begin{array}{l} \frac {1}{\epsilon} \left| e ^ {\langle f (x) + \epsilon \eta (x), f (z) + \epsilon \eta (z) \rangle / \tau} - e ^ {\langle f (x), f (z) \rangle / \tau} \right| \\ = \frac {1}{\epsilon} | \delta \Gamma (f + \epsilon c \eta) [ \epsilon \eta ] | \\ = \frac {1}{\tau \epsilon} | \langle f (x) + \epsilon c \eta (x), \epsilon \eta (z) \rangle + \langle \epsilon \eta (x), f (z) + \epsilon c \eta (z) \rangle | e ^ {\langle f (x) + \epsilon c \eta (x), f (z) + \epsilon c \eta (z) \rangle / \tau} \\ = \frac {1}{\tau} | \langle f (x) + \epsilon c \eta (x), \eta (z) \rangle + \langle \eta (x), f (z) + \epsilon c \eta (z) \rangle | e ^ {\langle f (x) + \epsilon c \eta (x), f (z) + \epsilon c \eta (z) \rangle / \tau} \\ \leq \frac {1}{\tau} [ \| f + \epsilon c \eta \| _ {2, \infty} \| \eta \| _ {2, \infty} + \| \eta \| _ {2, \infty} \| f + \epsilon c \eta \| _ {2, \infty} ] e ^ {(\| f \| _ {2, \infty} + \| \eta \| _ {2, \infty}) ^ {2} / \tau} \\ \leq \frac {2}{\tau} (\| f \| _ {2, \infty} + \| \eta \| _ {2, \infty}) \| \eta \| _ {2, \infty} e ^ {(\| f \| _ {2, \infty} + \| \eta \| _ {2, \infty}) ^ {2} / \tau} <   \infty . \end{array}
$$

Since ϵ was any value in (0, 1], the dominated convergence theorem allows interchange of integration and diferentiation such that from (39) the derivative of $\begin{array} { r } { \dot { f } \mapsto \int _ { \mathcal { Z } } e ^ { \langle f ( x ) , f ( z ) \rangle / \tau } \mu ( \mathrm { d } z ) } \end{array}$ at $f$ in direction η is:

$$
\int_ {\mathcal {Z}} \frac {1}{\tau} \left[ \langle f (x), \eta (z) \rangle + \langle \eta (x), f (z) \rangle \right] e ^ {\langle f (x), f (z) \rangle / \tau} \mu (\mathrm{d} z).
$$

By one further application of the chain rule, the derivative of f 7→ log $\textstyle \int _ { \mathcal { Z } } e ^ { \langle f ( x ) , f ( z ) \rangle / \tau } \mu ( \mathrm { d } z )$ in direction η is:

$$
\begin{array}{c} \frac {\int_ {\mathcal {Z}} \frac {1}{\tau} \left[ \langle f (x) , \eta (z) \rangle + \langle \eta (x) , f (z) \rangle \right] e ^ {\langle f (x) , f (z) \rangle / \tau} \mu (\mathrm{d} z)}{\int_ {\mathcal {Z}} e ^ {\langle f (x) , f (z) \rangle / \tau} \mu (\mathrm{d} z)} \\ = \frac {1}{\tau} \left[ \langle f (x), G _ {\tau} (f, \eta , \mu , x) \rangle + \langle \eta (x), G _ {\tau} (f, f, \mu , x) \rangle \right], \end{array}\tag{40}
$$

where the definition (36) has been used. Recalling (37), the proof is completed by subtracting (38) from (40). □

Lemma 7. For any $k \geq 1 , \tau > 0 , \beta \geq 1 , x _ { 1 } , \ldots , x _ { k } \in [ - 1 , 1 ]$ and $a _ { 1 } , \ldots , a _ { k } \geq 0$

$$
\frac {\sum_ {j = 1} ^ {k} e ^ {x _ {j} / \tau} a _ {j}}{\sum_ {j = 1} ^ {k} e ^ {x _ {j} / \tau}} \leq e ^ {2 / (\beta \tau)} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} a _ {j} ^ {\beta}\right) ^ {1 / \beta}.
$$

Proof. Denote $p _ { j } : = e ^ { x _ { j } / \tau } / \sum _ { i = 1 } ^ { k } e ^ { x _ { i } / \tau }$ . Since $\beta \geq 1$ , Jensen’s inequality gives:

$$
\left(\sum_ {j = 1} ^ {k} p _ {j} a _ {j}\right) ^ {\beta} \leq \sum_ {j = 1} ^ {k} p _ {j} a _ {j} ^ {\beta}.
$$

Combining this inequality with the fact that $p _ { j } \le e ^ { 2 / \tau } / k$ gives:

$$
\sum_ {j = 1} ^ {k} p _ {j} a _ {j} \leq \left(\sum_ {j = 1} ^ {k} p _ {j} a _ {j} ^ {\beta}\right) ^ {1 / \beta} \leq e ^ {2 / (\beta \tau)} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} a _ {j} ^ {\beta}\right) ^ {1 / \beta}.
$$

Lemma 8. For any $k \geq 1 , \tau > 0 , ( z ^ { a } , z _ { 1 } ^ { - } , \ldots , z _ { k } ^ { - } ) \in \mathcal { Z } ^ { 1 + k } , \phi , \phi ^ { \prime } \in \mathcal { B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } ) , f \in \mathcal { B } ( \mathcal { Z } , \mathbb { R } ^ { d } )$ such that $\| f \| _ { 2 , \infty } \leq 1$ , and $\beta \geq 1$

$$
\left\| G _ {\tau} \left(f, \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}\right) \right\| _ {2} \leq e ^ {2 / \beta \tau} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \left\| \phi \left(z _ {j} ^ {-}\right) - \phi^ {\prime} \left(z _ {j} ^ {-}\right) \right\| _ {2} ^ {\beta}\right) ^ {1 / \beta},
$$

where $\begin{array} { r } { \widehat { \pi } : = k ^ { - 1 } \sum _ { j = 1 } ^ { k } \delta _ { z _ { i } ^ { - } } } \end{array}$

Proof. As in the statement, fix any $k \geq 1 , \tau > 0 , ( z ^ { a } , z _ { 1 } ^ { - } , \ldots , z _ { k } ^ { - } ) \in \mathcal { Z } ^ { 1 + k } , \phi , \phi ^ { \prime } \in \mathcal { B } ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ and $f \in \mathcal { B } ( \mathcal { Z } , \mathbb { R } ^ { d } )$ such that $\| f \| _ { 2 , \infty } \leq 1$ . Define the shorthand:

$$
p _ {j} := \frac {e ^ {\langle f (z ^ {a}) , f (z _ {j} ^ {-}) \rangle / \tau}}{\sum_ {l = 1} ^ {k} e ^ {\langle f (z ^ {a}) , f (z _ {\ell} ^ {-}) \rangle / \tau}}, \qquad j = 1, \ldots , k.
$$

From the definition of $G _ { \tau }$ in (36),

$$
G _ {\tau} (f, \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}) = \sum_ {j = 1} ^ {k} p _ {j} \left[ \phi (z _ {j} ^ {-}) - \phi^ {\prime} (z _ {j} ^ {-}) \right].
$$

Now choose any $\beta \geq 1$ . By an application of the triangle inequality for the $\| \cdot \| _ { 2 }$ norm and lemma 7 with there $x _ { j } = \langle f ( z ^ { a } ) , f ( z _ { j } ^ { - } ) \rangle$ (so that $x _ { j } \in [ - 1 , 1 ]$ as required since by assumption of the present lemma $\| f \| _ { 2 , \infty } \leq 1 )$ ,

$$
\left\| G _ {\tau} \left(f, \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}\right) \right\| _ {2} \leq \sum_ {j = 1} ^ {k} p _ {j} \left\| \phi \left(z _ {j} ^ {-}\right) - \phi^ {\prime} \left(z _ {j} ^ {-}\right) \right\| _ {2} \leq e ^ {2 / (\beta \tau)} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \left\| \phi \left(z _ {j} ^ {-}\right) - \phi^ {\prime} \left(z _ {j} ^ {-}\right) \right\| _ {2} ^ {\beta}\right) ^ {1 / \beta}.\tag{41}
$$

Proof of proposition $\it 4 .$ Fix any $k , ~ \tau$ and $\mathbf { z } = ( z ^ { a } , z ^ { + } , z _ { 1 } ^ { - } , \ldots , z _ { k } ^ { - } ) \in { \mathcal { Z } } ^ { 2 + k }$ as in the statement of the theorem. To lighten notation in the proof, for any $f \in \mathcal { B } ( \mathcal { Z } , \mathbb { R } ^ { d } )$ define the shorthand $\ell ( f ) \equiv \ell ( f , \mathbf { z } , k , \tau )$ and $H ( f ) \equiv H _ { \tau } ( f , z ^ { a } , z ^ { + } , \widehat { \pi } ^ { - } )$ ), with $\begin{array} { r } { \widehat { \pi } ^ { - } : = k ^ { - 1 } \sum _ { j = 1 } ^ { k } \delta _ { z _ { j } ^ { + } } } \end{array}$ and where $H _ { \tau }$ is defined in (35). Observe then:

$$
\ell (f) = \log \left(\frac {1}{k} + e ^ {H (f)}\right).
$$

Denoting by $\delta \ell ( f ) [ \eta ] , \delta e ^ { H ( f ) } [ \eta ]$ and $\delta H ( f ) [ \eta ]$ the Gˆateaux derivatives of respectively $f \mapsto \ell ( f ) , f \mapsto e ^ { H ( f ) }$ and $f \mapsto H ( f )$ at a point $f \in \mathcal { B } ( \mathcal { Z } , \mathbb { R } ^ { d } )$ and in direction $\eta \in B ( \mathcal { Z } , \mathbb { R } ^ { d } )$ , the chain rule gives:

$$
\delta \ell (f) [ \eta ] = \frac {0 + \delta e ^ {H (f)} [ \eta ]}{1 / k + e ^ {H (f)}} = \frac {e ^ {H (f)}}{1 / k + e ^ {H (f)}} \delta H (f) [ \eta ].\tag{42}
$$

Now fix any $\phi , \phi ^ { \prime } \in B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ . By the mean value theorem and (42), there exists $c \in [ 0 , 1 ]$ , such that, with $\xi : = ( 1 - c ) \phi + c \phi ^ { \prime } \in B ( \mathcal { Z } , \mathbb { R } ^ { d } )$

$$
\ell (\phi) - \ell (\phi^ {\prime}) = \delta \ell (\xi) [ \phi - \phi^ {\prime} ] = \frac {e ^ {H (\xi)}}{1 / k + e ^ {H (\xi)}} \delta H (\xi) [ \phi - \phi^ {\prime} ],\tag{43}
$$

and by applying lemma 6 with there $f = \xi , \eta = \phi - \phi ^ { \prime } , x = z ^ { a } , y = z ^ { + } , \mu = { \widehat { \pi } } ^ { - }$

$$
= \left[ \left\langle \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}), G _ {\tau} (\xi , \xi , z ^ {a}, \widehat {\pi}) - \xi (z ^ {+}) \right\rangle + \left\langle \xi (z ^ {a}), G _ {\tau} (\xi , \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}) - \phi (z ^ {+}) + \phi^ {\prime} (z ^ {+}) \right\rangle \right]
$$

$$
= \left\langle \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}), - \xi (z ^ {+}) \right\rangle + \left\langle \phi (z ^ {+}) - \phi^ {\prime} (z ^ {+}), - \xi (z ^ {a}) \right\rangle
$$

$$
+ \langle \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}), G _ {\tau} (\xi , \xi , \widehat {\pi}, z ^ {a}) \rangle + \langle \xi (z ^ {a}), G _ {\tau} (\xi , \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}) \rangle
$$

$$
= \left\langle \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}), - \xi (z ^ {+}) \right\rangle + \left\langle \phi (z ^ {+}) - \phi^ {\prime} (z ^ {+}), - \xi (z ^ {a}) \right\rangle\tag{44}
$$

$$
+ \langle \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}), G _ {\tau} (\xi , \xi , \widehat {\pi}, z ^ {a}) - G _ {\tau} (\xi , \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}) \rangle\tag{45}
$$

$$
+ \left\langle \xi (z ^ {a}) + \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}), G _ {\tau} (\xi , \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}) \right\rangle .\tag{46}
$$

We shall apply the Cauchy-Schwartz inequality to each of the inner-products in (44)-(46). In preparation, observe that since $\| \phi \| _ { 2 , \infty } = \| \phi ^ { \prime } \| _ { 2 , \infty } = 1$ , we have:

$$
\| \xi \| _ {2, \infty} \leq 1 \quad \text { and } \quad \| \xi - \phi + \phi^ {\prime} \| _ {2, \infty} \vee \| \xi + \phi - \phi^ {\prime} \| _ {2, \infty} \leq 3.\tag{47}
$$

Recalling the definition of $G _ { \tau }$ in (36), note that $G _ { \tau }$ is linear in its second argument. Combined with Jensen’s inequality, this gives:

$$
\| G _ {\tau} (\xi , \xi , \widehat {\pi}, z ^ {a}) - G _ {\tau} (\xi , \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}) \| _ {2} = \| G _ {\tau} (\xi , \xi - \phi + \phi^ {\prime}, \widehat {\pi}, z ^ {a}) \| _ {2} \leq \| \xi - \phi + \phi^ {\prime} \| _ {2, \infty} \leq 3.
$$

(48)

Since $\| \xi \| _ { 2 , \infty } \le 1$ we may apply lemma 8 with there $f = \xi$ to give, for any $\beta \geq 1$

$$
\| G _ {\tau} (\xi , \phi - \phi^ {\prime}, \widehat {\pi}, z ^ {a}) \| _ {2} \leq e ^ {2 / \beta \tau} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| \phi (z _ {j} ^ {-}) - \phi^ {\prime} (z _ {j} ^ {-}) \| _ {2} ^ {\beta}\right) ^ {1 / \beta}.\tag{49}
$$

Combining (43); the fact $e ^ { H ( \xi ) } / ( 1 / k + e ^ { H ( \xi ) } ) \leq 1 ;$ application of the Cauchy-Schwartz inequality to each of the inner-products in $( 4 4 ) ‐ ( 4 6 ) ;$ ; and the bounds (47), (48) and (49) gives:

$$
\begin{array}{l} | \ell (\phi) - \ell (\phi^ {\prime}) | \\ \qquad \leq \frac {4}{\tau} \| \phi (z ^ {a}) - \phi^ {\prime} (z ^ {a}) \| _ {2} + \frac {1}{\tau} \| \phi (z ^ {+}) - \phi^ {\prime} (z ^ {+}) \| _ {2} + \frac {3}{\tau} e ^ {2 / \beta \tau} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| \phi (z _ {j} ^ {-}) - \phi^ {\prime} (z _ {j} ^ {-}) \| _ {2} ^ {\beta}\right) ^ {1 / \beta}. \end{array}
$$

□

## B.2 Other proofs for section 4

Proof of lemma 1. It follows from proposition 4 that:

$$
\begin{array}{l} | \ell (\phi , \mathbf {z} _ {i}, k, \tau) - \ell (\phi^ {\prime}, \mathbf {z} _ {i}, k, \tau) | \\ \leq \frac {4}{\tau} \left(\| \phi (z _ {i} ^ {a}) - \phi^ {\prime} (z _ {i} ^ {a}) \| _ {2} + \| \phi (z _ {i} ^ {+}) - \phi^ {\prime} (z _ {i} ^ {+}) \| _ {2} + e ^ {2 / \beta \tau} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| \phi (z _ {i j} ^ {-}) - \phi^ {\prime} (z _ {i j} ^ {-}) \| _ {2} ^ {\beta}\right) ^ {1 / \beta}\right). \end{array}
$$

By applying the Lipschitz condition in assumption (C),

$$
\leq \frac {4}{\tau} C _ {\Phi} \| \theta - \theta^ {\prime} \| _ {2} \left(\| z _ {i} ^ {a} \| _ {2} + \| z _ {i} ^ {+} \| _ {2} + e ^ {2 / \beta \tau} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {\beta}\right) ^ {1 / \beta}\right).\tag{50}
$$

Applying the triangle inequality for the $\| \cdot \| _ { 2 }$ norm in $\mathbb { R } ^ { n }$

$$
\begin{array}{r l} & {\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \left[ \| z _ {i} ^ {a} \| _ {2} + \| z _ {i} ^ {+} \| _ {2} + e ^ {2 / \beta \tau} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {\beta}\right) ^ {1 / \beta} \right] ^ {2}\right) ^ {1 / 2}} \\ & {\qquad \leq \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \| z _ {i} ^ {a} \| _ {2} ^ {2}\right) ^ {1 / 2} + \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \| z _ {i} ^ {+} \| _ {2} ^ {2}\right) ^ {1 / 2} + e ^ {2 / \beta \tau} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {\beta}\right) ^ {2 / \beta} \right] ^ {1 / 2}.} \end{array}\tag{51}
$$

Now set $\beta = 1 \vee 2 / \tau$ . When $\tau < 1$ we have $\tau = 2 / \beta < 1$ and $\beta \tau = \tau \vee 2 = 2$ . In this case, by Jensen’s inequality,

$$
e ^ {2 / \beta \tau} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {\beta}\right) ^ {2 / \beta} \right] ^ {1 / 2} \leq e \left(\frac {1}{n k} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {\beta}\right) ^ {1 / \beta} = e \left(\frac {1}{n k} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {2 / \tau}\right) ^ {\tau / 2}.
$$

On the other hand, when $\tau \geq 1$ , we have $\beta \in [ 1 , 2 ] , 2 / \beta \geq 1$ and $\beta \tau = \tau \vee 2 \geq 2$ . In this case, by Jensen’s inequality,

$$
e ^ {2 / \beta \tau} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {\beta}\right) ^ {2 / \beta} \right] ^ {1 / 2} \leq e \left[ \frac {1}{n k} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {2} \right] ^ {1 / 2}.
$$

Combining the two cases: $\tau < 1$ and $\tau \geq 1$ 2

$$
e ^ {2 / \beta \tau} \left[ \frac {1}{n} \sum_ {i = 1} ^ {n} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {\beta}\right) ^ {2 / \beta} \right] ^ {1 / 2} \leq e \left[ \frac {1}{n k} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {k} \| z _ {i j} ^ {-} \| _ {2} ^ {2 / (1 \wedge \tau)} \right] ^ {(1 \wedge \tau) / 2}.\tag{52}
$$

On both sides of (50), take the square, then the arithmetic average over the index $i = 1 , \ldots , n$ , then take the square root. Combined with (51) and (52) this gives:

$$
\rho_ {n} ^ {\mathrm{info}} (\phi_ {\theta}, \phi_ {\theta^ {\prime}}) \leq \frac {4 e}{\tau} B _ {\tau} (\mathbf {z} _ {1}, \ldots , \mathbf {z} _ {n}) C _ {\Phi} \| \theta - \theta^ {\prime} \| _ {2}.
$$

Proof of proposition 5. Let $\Phi = \{ \phi _ { \theta } ; \theta \in \Theta \}$ as per assumption (C). As shorthand notation, let us absorb various quantities in the statement of lemma 1 into a constant L such that for all $\theta , \theta ^ { \prime } \in \Theta$

$$
\rho^ {\mathrm{info}} (\phi_ {\theta}, \phi_ {\theta^ {\prime}}) \leq L \| \theta - \theta^ {\prime} \| _ {2}.\tag{53}
$$

If for some $N \geq 1$ and $\epsilon > 0 , \{ \theta ^ { 1 } , \dots , \theta ^ { N } \}$ is an ϵ/L-cover of Θ with respect to the $\| \cdot - \cdot \| _ { 2 }$ distance, then it follows from (53) that $\left\{ \phi _ { \theta ^ { 1 } } , \dots , \phi _ { \theta ^ { N } } \right\}$ is an ϵ-cover of Φ with respect to $\rho ^ { \mathrm { i n f o } }$ . In turn, the associated covering numbers of Φ and Θ obey:

$$
\mathcal {N} (\epsilon , \Phi , \rho^ {\mathrm{info}}) \leq \mathcal {N} (\epsilon / L, \Theta , \| \cdot - \cdot \| _ {2}).
$$

Combining this inequality with Dudley’s entropy integral, $\mathrm { e . g . }$ , [Wainwright, 2019, eq. 5.48],

$$
\begin{array}{r l} & {\mathbb {E} \left[ \underset {\phi \in \Phi} {\sup} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {z} _ {i}, k, \tau) \right]} \\ & {\leq \frac {2 4}{\sqrt {n}} \int_ {0} ^ {\frac {4}{\tau} \wedge L R} \sqrt {\log \mathcal {N} (\epsilon , \Phi , \rho^ {\mathrm{info}})} \mathrm{d} \epsilon} \\ & {\leq \frac {2 4}{\sqrt {n}} \int_ {0} ^ {\frac {4}{\tau} \wedge L R} \sqrt {\log \mathcal {N} (\epsilon / L , \Theta , \| \cdot - \cdot \| _ {2})} \mathrm{d} \epsilon ,} \end{array}\tag{54}
$$

where $\sigma _ { 1 } , \ldots , \sigma _ { n }$ are i.i.d. Rademacher variables; the integral upper-limit term $4 / \tau$ in the first inequality holds because by lemma $5 , \rho ^ { \mathrm { i n f o } } ( \phi , \phi ^ { \prime } ) \leq 4 / \tau$ for all $\phi , \phi ^ { \prime } \in \Phi \subset B ( \mathcal { Z } , \mathbb { S } ^ { d - 1 } )$ ; the integral upper-limit LR appears since by combining (53) with the definition of Θ to we have $\rho ^ { \mathrm { i n f o } } ( \phi _ { \theta } , \phi _ { \theta ^ { \prime } } ) \leq L R .$

We have the standard Euclidean volumetric estimate: $\mathcal { N } ( \epsilon , \Theta , \| \cdot - \cdot \| _ { 2 } ) \le ( 1 + 2 R / \epsilon ) ^ { d _ { \Theta } }$ , see, e.g., [Wainwright, 2019, eq. 5.9]. Writing $U : = 4 / \tau \wedge L R$ , Cauchy-Schwartz gives:

$$
\begin{array}{l} \int_ {0} ^ {U} \sqrt {\log \mathcal {N} (\epsilon / L , \Theta , \| \cdot - \cdot \| _ {2})}   \mathrm{d} \epsilon \leq \left(\int_ {0} ^ {U} 1 ^ {2}   \mathrm{d} \epsilon\right) ^ {1 / 2} \left(\int_ {0} ^ {U} \log \mathcal {N} (\epsilon / L, \Theta , \| \cdot - \cdot \| _ {2})\right) ^ {1 / 2}   \mathrm{d} \epsilon \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qend{array}
$$

where the equality uses the fact that the anti-derivative in question is $\mathrm { \Phi } \mathrm { l o g } ( 1 + 2 R L / \epsilon ) + 2 R L \log ( 1 +$ $\epsilon / 2 R L )$ , and the final inequality uses log $\left( 1 + x \right) \leq x$ for $x \geq 0$

Returning to (54) in the case $4 / \tau \le R L$ , we have $U = 4 / \tau$ and:

$$
\begin{array}{r l} & {\mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {z} _ {i}, k, \tau) \right] \leq \frac {2 4 \sqrt {d _ {\Theta}}}{\sqrt {n}} \frac {4}{\tau} \left[ \log \left(1 + \frac {2 R L}{4 / \tau}\right) + 1 \right] ^ {1 / 2}} \\ & {\qquad = \frac {2 4 \sqrt {d _ {\Theta}}}{\sqrt {n}} \frac {4}{\tau} [ \log (1 + 2 a) + 1 ] ^ {1 / 2},} \end{array}
$$

where $L = 4 e B _ { \tau } ( { \bf z } _ { 1 } , \ldots , { \bf z } _ { n } ) C _ { \Phi } / \tau$ (from lemma 1) has been used and $a : = e R b _ { \tau } ( { \bf z } _ { 1 } , \ldots , { \bf z } _ { n } ) C _ { \Phi }$

On the other hand, if $4 / \tau > R L$ , we have $U = R L$ and

$$
\begin{array}{r l r} & & {\mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {z} _ {i}, k, \tau) \right] \leq \frac {2 4 \sqrt {d _ {\Theta}}}{\sqrt {n}} R L [ \log (3) + 1 ] ^ {1 / 2}} \\ & & {\leq \frac {2 4 \sqrt {d _ {\Theta}}}{\sqrt {n}} \frac {4}{\tau} a [ \log (3) + 1 ] ^ {1 / 2}.} \end{array}
$$

Since $4 / \tau > R L \Leftrightarrow 1 > e R B _ { \tau } ( \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { n } ) C _ { \Phi } \Leftrightarrow 1 > a$ , we obtain

$$
\mathbb {E} \left[ \sup _ {\phi \in \Phi} \frac {1}{n} \sum_ {i = 1} ^ {n} \sigma_ {i} \ell (\phi , \mathbf {z} _ {i}, k, \tau) \right] \leq \frac {9 6}{\tau} \sqrt {\frac {d _ {\Theta}}{n}} \min \left\{a \sqrt {\log (3) + 1}, \sqrt {\log (1 + 2 a) + 1} \right\}.
$$

## References

Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. Advances in Neural Information Processing Systems, 30, 2017.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1597–1607. PMLR, 2020.

Naghmeh Ghanooni, Waleed Mustafa, Yunwen Lei, Anthony Widjaja Lin, and Marius Kloft. Generalization bounds with logarithmic negative-sample dependence for adversarial contrastive learning. Transactions on Machine Learning Research, 2024.

Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. In Proceedings of the 31st Conference On Learning Theory, pages 297–299. PMLR, 2018.

Antoine Gonon, Nicolas Brisebarre, Elisa Riccietti, and R´emi Gribonval. A rescaling-invariant lipschitz bound based on path-metrics for modern relu network parameterizations. In International Conference on Machine Learning, pages 20047–20074. PMLR, 2025.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9729–9738. IEEE, 2020. doi: 10.1109/cvpr42600.2020.00975.

Olivier Henaf. Data-eficient image recognition with contrastive predictive coding. In International Conference on Machine Learning, pages 4182–4192. PMLR, 2020.

Nong Minh Hieu and Antoine Ledent. Generalization analysis for supervised contrastive representation learning under non-iid settings. In International Conference on Machine Learning, pages 23179–23218. PMLR, 2025.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in Neural Information Processing Systems, 33:18661–18673, 2020.

Yunwen Lei, Ur¨un Dogan, Ding-Xuan Zhou, and Marius Kloft. Data-dependent generalization bounds<sup>¨</sup> for multi-class classification. IEEE Transactions on Information Theory, 65(5):2995–3021, 2019.

Yunwen Lei, Tianbao Yang, Yiming Ying, and Ding-Xuan Zhou. Generalization analysis for contrastive representation learning. In International Conference on Machine Learning, pages 19200–19227. PMLR, 2023.

Andreas Maurer. A vector-contraction inequality for rademacher complexities. In International Conference on Algorithmic Learning Theory, pages 3–17. Springer, 2016.

Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of Machine Learning. MIT press, 2018.

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural net works. In Conference on Learning Theory, pages 1376–1401. PMLR, 2015.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021. doi: 10.48550/arxiv.2103.00020.

Nikunj Saunshi, Orestis Plevrakis, Sanjeev Arora, Mikhail Khodak, and Hrishikesh Khandeparkar. A theoretical analysis of contrastive unsupervised representation learning. In International Conference on Machine Learning, pages 5628–5637. PMLR, 2019.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In European conference on computer vision, pages 776–794. Springer, 2020.

A¨aron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proceedings of the 37th International Conference on Machine Learning. PMLR, 2020a.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere, 2020b. URL https://arxiv.org/abs/2005.10242v10.

Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via nonparametric instance discrimination. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3733–3742, 2018.

Ge Yang, Edward Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning large neural networks via zero-shot hyperparameter transfer. Advances in Neural Information Processing Systems, 34:17084–17097, 2021.

Chun-Hsiao Yeh, Cheng-Yao Hong, Yen-Chi Hsu, Tyng-Luh Liu, Yubei Chen, and Yann LeCun. Decoupled contrastive learning. In European conference on computer vision, pages 668–684. Springer, 2022.

Roland S. Zimmermann, Yash Sharma, Stefen Schneider, Matthias Bethge, and Wieland Brendel. Contrastive learning inverts the data generating process. In International Conference on Machine Learning, pages 12979–12990. PMLR, 2021.