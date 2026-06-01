# Self-Supervised Learning as Discrete Communication

Kawtar Zaher \* 1 2 Ilyass Moummad \* 1 Olivier Buisson 2 Alexis Joly 1

# Abstract

Most self-supervised learning (SSL) methods learn continuous visual representations by aligning different views of the same input, offering limited control over how information is structured across representation dimensions. In this work, we frame visual self-supervised learning as a discrete communication process between a teacher and a student network, where semantic information is transmitted through a fixed-capacity binary channel. Rather than aligning continuous features, the student predicts multi-label binary messages produced by the teacher. Discrete agreement is enforced through an element-wise binary crossentropy objective, while a coding-rate regularization term encourages effective utilization of the constrained channel, promoting structured representations. We further show that periodically reinitializing the projection head strengthens this effect by encouraging embeddings that remain predictive across multiple discrete encodings. Extensive experiments demonstrate consistent improvements over continuous agreement baselines on image classification, retrieval, and dense visual prediction tasks, as well as under domain shift through self-supervised adaptation. Beyond backbone representations, we analyze the learned binary codes and show that they form a compact and informative discrete language, capturing semantic factors reusable across classes.

# 1. Introduction

Self-supervised learning (SSL) has emerged as the predominant paradigm for learning visual representations without human annotations. Contemporary SSL methods can be broadly categorized into representation alignment approaches, which enforce consistency between dif-1INRIA, LIRMM, Universite de Montpellier, Mont- ´ pellier, France 2Institut National de l’Audiovisuel, Paris, France. Correspondence to: Kawtar Zaher <kzaher@ina.fr>, Ilyass Moummad <ilyass.moummad@inria.fr>, Alexis Joly <alexis.joly@inria.fr>.

Preprint. February 11, 2026.

ferent views of the same image through contrastive or selfdistillation objectives (Chen et al., 2020; He et al., 2020; Caron et al., 2021; Bardes et al., 2021; Zhou et al., 2021; Oquab et al., 2023; Wu et al., 2025), and generative or predictive approaches, which learn representations by reconstructing or predicting missing content in pixel space (He et al., 2022; Yang et al., 2025), token space (El-Nouby et al., 2024), or embedding space (Assran et al., 2023; Xu et al., 2025). Despite their methodological differences, most of these frameworks rely on continuous-valued targets and similarity- or regression-based objectives, such as crossentropy over prototypes or cosine distance minimization. While highly effective, continuous alignment enforces semantic consistency through global similarity in the embedding space, offering limited explicit control over representational structure. This may result in entangled features with multiple semantic factors mixed across dimensions.

Discrete representations provide a compelling alternative to address these limitations. Encoding information through binary variables allows each dimension to capture the presence or absence of a semantic attribute, naturally encouraging a decomposition of information across dimensions and enabling multi-label semantics. Such representations are well suited to capture the compositional structure of visual scenes and, in principle, can support up to $2 ^ { d }$ distinct configurations in d dimensions. However, in self-supervised learning, discrete representations have primarily been explored in reconstruction-based settings, using fixed codebooks for masked image modeling (Bao et al., 2021; Dong et al., 2023). These approaches are often outperformed by representation alignment methods on real-world data (Van Assel et al., 2025), and several recent works report that continuous targets remain more effective than discrete token-based formulations in predictive SSL objectives (He et al., 2022; El-Nouby et al., 2024; Xu et al., 2025). In supervised settings, discrete multi-label representations can be defined using ground-truth annotations (Lee et al., 2022), but without labels it remains unclear how to define informative discrete targets or ensure that each dimension carries meaningful distinct information.

In this work, we show that discrete representations can be effectively leveraged for self-supervised representation alignment. We propose to formulate SSL as a discrete communication problem between a teacher and a student, where information is transmitted through a fixed number of binary channels. The student predicts multi-label binary signals produced by the teacher using an element-wise binary cross-entropy objective, while a coding-rate regularization term encourages effective and balanced utilization of the discrete bottleneck. Unlike prototype-based SSL methods (YM. et al., 2020; Caron et al., 2020; 2021; Assran et al., 2022; Oquab et al., 2023; Simeoni et al. ´ , 2025), which enforce agreement around a single dominant prototype per image, our framework allows multiple semantic factors to be active simultaneously, yielding structured, factorized, and reusable representations. The discrete communication task serves solely as a self-supervised pretext objective: the backbone continues to learn continuous visual representations.

We evaluate the proposed discrete communication framework through visual representation learning experiments across a range of downstream tasks. Our approach matches or improves ImageNet-1K classification performance and yields consistent gains in image retrieval on in-domain and lightly out-of-distribution benchmarks, while also improving performance on unsupervised object detection, instance segmentation, and semi-supervised video object segmentation. Under severe domain shift, our model also achieves higher linear probing performance, which can be further improved through continued self-supervised fine-tuning on the target domain. Finally, we analyze both the learned continuous embeddings and the binary codes, showing that discrete agreement leads to more balanced and less correlated continuous representations, and yields compact and informative binary codes that capture reusable semantic factors shared across classes.

# 2. Related Work

Self-Supervised Learning and Representation Alignment. Most successful self-supervised learning (SSL) approaches rely on representation alignment across augmented views, using contrastive objectives (Chen et al., 2020; Tian et al., 2020; He et al., 2020) or teacher–student self-distillation (Grill et al., 2020; Caron et al., 2021; Zhou et al., 2021; Wu et al., 2025) to enforce invariance. In teacher–student frameworks, the student predicts momentum-updated teacher representations using similarity or regression losses in continuous embedding spaces, without requiring explicit negative samples (Grill et al., 2020; Caron et al., 2021; Wu et al., 2025). In parallel, reconstruction-based methods learn representations by recovering masked pixels or tokens (Bao et al., 2021; Dong et al., 2023; He et al., 2022; El-Nouby et al., 2024; Xu et al., 2025), but generally underperform alignment-based objectives for learning general-purpose representations on real-world data (Van Assel et al., 2025). While approaches such as BEiT (Bao et al., 2021) introduce discrete prediction targets through a fixed tokenizer, the discretization applies to the reconstruction objective rather than to the agreement mechanism or the structure of the learned representation.

Our work builds on alignment-based self-supervised learning, but departs from prior approaches by replacing continuous alignment with explicit multi-label discrete agreement, enabling direct control over the structure and capacity of the learned representations. While alignment-based approaches such as SimCLR (Chen et al., 2020), BYOL (Grill et al., 2020), VICReg (Bardes et al., 2021), and SimDINO (Wu et al., 2025) rely on continuous similarity or regression objectives, and methods such as DINO (Caron et al., 2021) or iBOT (Zhou et al., 2021) employ soft assignment to prototypes, the supervision signals they induce remain continuous and predominantly mono-modal. In contrast, our approach introduces an explicit discrete agreement mechanism based on multi-label binary signals, while still learning continuous representations in the backbone. This distinction allows us to impose structured constraints on how information is allocated across dimensions of the agreement space, without discretizing the learned embeddings themselves.

Information-Theoretic Perspectives on Self-Supervised Learning. A growing body of work has analyzed selfsupervised learning through an information-theoretic lens, viewing representation learning as the problem of maximizing informative content while avoiding collapse. Early formulations connect SSL objectives to mutual information maximization between augmented views (Ozsoy et al., 2022; Hjelm et al., 2018), typically approximated through contrastive or similarity-based losses. More recent approaches emphasize coding-rate and rate–distortion principles to characterize representation diversity and compression (Li et al., 2022b; Tong et al., 2023; Wu et al., 2025). In particular, the work of Yu et al. (Yu et al., 2020b) introduces the Maximal Coding Rate Reduction (MCR2) principle, motivating objectives that balance alignment with expressiveness (Yu et al., 2020a; Tong et al., 2023; Wu et al., 2025). SimDINO (Wu et al., 2025) further connects self-distillation methods in the DINO family to coding-rate maximization as an effective anti-collapse mechanism. Our work is closely related to these perspectives, but differs in how coding-rate principles are operationalized. Rather than using coding rate implicitly to regularize continuous embeddings, we introduce an explicit multi-label discrete bottleneck and encourage its effective utilization through coding-rate regularization. This combination constrains the total information capacity while promoting its balanced allocation across dimensions, which in turn encourages factorized representations. As a result, the learned representations can support semantic attributes that are shared across images and classes, rather than being tied to mutually exclusive categories.

Hashing. Deep hashing methods aim to learn compact binary codes for efficient similarity search, typically framing code learning as a compression problem and distilling knowledge from pretrained encoders to preserve neighborhood structure in Hamming space (Luo et al., 2023). Recent self-supervised hashing approaches incorporate contrastive, reconstruction-based, or regularization objectives to mitigate code collapse and improve retrieval performance (Cao et al., 2023; Ma et al., 2024; Gong et al., 2022; Shen et al., 2024; Li & van Gemert, 2021; Moummad et al., 2025). In contrast to these works, our objective is not to learn binary codes as final representations. Instead, we train teacher– student models from scratch using a hashing-like objective as a self-supervised pretext task, where binary codes serve as discrete prediction targets. Importantly, while binary crossentropy losses are commonly used in hashing to optimize similarity-preserving codes, they are typically embedded in pairwise or neighborhood-based formulations, whereas our loss enforces multi-label discrete agreement between teacher and student and does not directly optimize the codes themselves for retrieval.

# 3. Problem Setup

We consider two random variables $X _ { 1 } , X _ { 2 }$ representing two stochastic augmentations of the same image $I \in { \mathcal { X } } ;$ :

$$
X _ {1} = A _ {1} (I), \quad X _ {2} = A _ {2} (I), \tag {1}
$$

where $A _ { 1 } , A _ { 2 }$ are random augmentation operators (crop, color jitter, blur, etc.), and X denotes the space of images.

We aim to learn a representation mapping $f _ { \theta } : \mathcal { X }  \mathbb { R } ^ { d }$ and a binary communication head $\tilde { g } _ { \phi }$ that parameterizes a discrete representation space of fixed capacity. Specifically, $\tilde { g } _ { \phi }$ maps continuous features to a factorized Bernoulli distribution over B binary variables:

$$
\tilde {g} _ {\phi}: \mathbb {R} ^ {d} \rightarrow (0, 1) ^ {B}. \tag {2}
$$

Given an input x (either $X _ { 1 }$ or $X _ { 2 } )$ , the encoder produces a feature $h = f _ { \theta } ( x ) \in  { \mathbb { R } } ^ { d }$ , and the head outputs Bernoulli parameters $\begin{array} { r } { p = \tilde { g } _ { \phi } ( h ) } \end{array}$ . A binary code $z \in \{ 0 , 1 \} ^ { B }$ is then drawn as:

$$
z \sim P _ {\phi} (Z \mid X) = \prod_ {b = 1} ^ {B} \text { Bernoulli } (p _ {b}). \tag {3}
$$

In the following, we use this probabilistic formulation to define our objective, while practical optimization relies on tractable surrogates introduced in the next section.

We seek binary codes $Z _ { 1 }$ and $Z _ { 2 }$ associated with $X _ { 1 }$ and $X _ { 2 }$ that are:

(i) invariant across augmentations,   
(ii) spread out across samples and approximately factorized across bits.

From a communication perspective, $Z _ { 1 }$ and $Z _ { 2 }$ can be viewed as two messages transmitted through the same discrete channel under different noise realizations induced by stochastic data augmentations. From this perspective, our learning objective can be naturally formulated in information-theoretic terms:

$$
\max _ {\theta , \phi} \mathbb {E} \left[ I (Z _ {1}; Z _ {2}) \right] \tag {4}
$$

$\mathrm { w h e r e } \qquad I ( Z _ { 1 } ; Z _ { 2 } ) = H ( Z _ { 1 } ) - H ( Z _ { 1 } | Z _ { 2 } ) .$

Thus, maximizing the mutual information encourages invariance through the conditional entropy minimization, while the entropy maximization enforces maximal utilization of the B-bit discrete communication channel, leading to diverse and approximately independent bits.

In practice, we use $\tilde { g } _ { \phi } = \sigma \circ g _ { \phi }$ where $g _ { \phi } : \mathbb { R } ^ { d }  \mathbb { R } ^ { B }$ is a continuous projection head, and σ denotes the element-wise sigmoid function.

In the next section, we show how each component of the objective in Eq. 4 is approximated and optimized in practice.

# 4. Method

# 4.1. Architecture

We build on the simplified DINO framework proposed in SimDINO (Wu et al., 2025), which removes prototype assignments and contrastive normalization, resulting in a cleaner alignment objective that better isolates the effects of our discrete agreement mechanism. Similar to their work, our model consists of:

• a backbone $f _ { \theta } : \mathcal { X } \xrightarrow { } \mathbb { R } ^ { d } ( \mathrm { e . g . , a }$ Vision Transformer),   
• a projection head $g _ { \phi } : \mathbb { R } ^ { d }  \mathbb { R } ^ { B }$ , typically a small MLP.

For each training step, we sample two augmentations $x _ { 1 } \sim$ $X _ { 1 }$ and $x _ { 2 } \sim X _ { 2 }$ of the same image $I ,$ and we compute the embeddings $h _ { i }$ , apply the projection head to obtain the logits $a _ { i }$ and the probabilities $p _ { i } ,$ where $i \in \{ 1 , 2 \}$ :

$$
h _ {i} = f _ {\theta} (x _ {i}), \quad a _ {i} = g _ {\phi} (h _ {i}), \quad p _ {i} = \sigma (a _ {i}). \tag {5}
$$

In our discrete communication perspective, the projection head $g _ { \phi }$ parameterizes the discrete communication channel, while the backbone $f _ { \theta }$ learns representations that are robust to noise induced by data augmentations.

# 4.2. Binary Agreement Objective

The invariance term (i) of Eq. 4 corresponds to minimizing the conditional entropy $H ( Z _ { 1 } | Z _ { 2 } )$ , which enforces consistency between the binary codes associated with different augmentations of the same image.

As the true conditional distribution $P ( Z _ { 1 } | Z _ { 2 } )$ is unknown and intractable, we introduce a variational surrogate distribution $Q _ { \phi } ( Z _ { 1 } | Z _ { 2 } )$ , which yields the standard variational upper bound:

$$
H (Z _ {1} | Z _ {2}) \leq \mathbb {E} \big [ - l o g (Q _ {\phi} (Z _ {1} | Z _ {2}) \big ],
$$

thus: max $I ( Z _ { 1 } ; Z _ { 2 } ) \longleftrightarrow$ min $\mathbb { E } \big [ - l o g ( Q _ { \phi } ( Z _ { 1 } | Z _ { 2 } ) \big ]$ .

The framework consists of two branches: a teacher branch $( f _ { \theta } ^ { t } , g _ { \phi } ^ { t } )$ , and a student branch $( f _ { \theta } ^ { s } , g _ { \phi } ^ { s } )$ . Since $Z _ { 2 }$ is induced from $X _ { 2 }$ , we model $Q _ { \phi } ( Z _ { 1 } | Z _ { 2 } )$ as the factorized Bernoulli distribution whose parameters are predicted by the student network applied on $X _ { 2 } { \mathrm { : } }$

$$
Q _ {\phi} (Z _ {1} \mid Z _ {2}) = \prod_ {b = 1} ^ {B} \text { Bernoulli } (p _ {2, b} ^ {s}). \tag {6}
$$

We define the binary targets by applying a stop-gradient threshold to the probabilities obtained via the teacher branch:

$$
\hat {z} _ {1} ^ {t} = \mathbf {1} [ p _ {1} ^ {t} > 1 / 2 ], \qquad \hat {z} _ {2} ^ {t} = \mathbf {1} [ p _ {2} ^ {t} > 1 / 2 ]. \tag {7}
$$

We use hard thresholded targets to provide an explicit and unambiguous discrete supervision signal, while gradients are propagated only through the student branch.

Assuming conditional independence across bits, the negative log-likelihood of the teacher code $\hat { z } _ { 1 } ^ { t }$ under $Q _ { \phi } ( Z _ { 1 } | Z _ { 2 } )$ reduces to an element-wise binary cross-entropy. We use a symmetric cross-view formulation:

$$
\begin{array}{l} \mathcal {L} _ {B C E} = - \sum_ {b = 1} ^ {B} \left(\hat {z} _ {1, b} ^ {t} \log p _ {2, b} ^ {s} + (1 - \hat {z} _ {1, b} ^ {t}) \log (1 - p _ {2, b} ^ {s}) \right. \\ + \hat {z} _ {2, b} ^ {t} \log p _ {1, b} ^ {s} + (1 - \hat {z} _ {2, b} ^ {t}) \log (1 - p _ {1, b} ^ {s}) \big). \tag {8} \\ \end{array}
$$

This loss encourages the student to accurately recover the discrete message communicated by the teacher, implementing a bit-wise approximation of the conditional entropy minimization in Eq. 4.

# 4.3. Coding-Rate Regularization

The binary agreement objective introduced above provides a practical approximation of the conditional entropy term $H ( Z _ { 1 } \mid Z _ { 2 } )$ in Eq. 4, enforcing invariance across augmentations under a fixed-capacity discrete bottleneck. However, maximizing mutual information also requires maximizing the marginal entropy $H ( Z )$ , which controls how effectively the available capacity of the discrete channel is utilized.

From an information-theoretic perspective, maximizing $H ( Z )$ for a fixed-length binary code amounts to promoting high marginal entropy and low redundancy across bits. In practice, this corresponds to encouraging representations that are diverse across samples and approximately uncorrelated across dimensions, so that each bit contributes complementary information. While these objectives are naturally defined over the discrete variables, directly operating on hard-thresholded codes is challenging both due to nondifferentiability and because estimating marginal or joint entropies over high-dimensional binary codes is inherently combinatorial.

For this reason, we regularize the continuous prebinarization logits rather than the discrete codes themselves. While SimDINO applies a coding-rate regularization to L2- normalized continuous embeddings, we apply the same principle to the pre-binarization logits that parameterize the discrete communication channel. Concretely, we promote effective utilization of the discrete bottleneck by optimizing a coding-rate objective on the L2-normalized logits a, encouraging them to be well spread on the unit hypersphere:

$$
\mathcal {L} _ {\text { rate }} = - R _ {\epsilon} (a) = - \frac {1}{2} \log \det \left(I + \frac {d}{\epsilon^ {2}} A\right), \tag {9}
$$

$$
\text { s.t. } \quad A = \operatorname{Cov} [ a / \| a \| _ {2} ]  .
$$

# 4.4. Final Objective

The full final training objective is:

$$
\mathcal {L} = \mathcal {L} _ {B C E} + \beta \times \mathcal {L} _ {\text { rate }}, \tag {10}
$$

where $\beta$ is a hyperparameter. Both branches are initialized similarly, the student branch is updated through gradient backpropagation of the loss ${ \mathcal { L } } ,$ and the teacher branch through exponentially moving average (EMA). Together, $\mathcal { L } _ { B C E }$ and $\mathcal { L } _ { r a t e }$ provide a practical optimization of the two complementary terms of the mutual information objective in Eq. 4: invariance and capacity utilization.

# 4.5. Periodically Randomized Heads

While discrete agreement and coding-rate regularization encourage informative and factorized representations, keeping the projection head fixed throughout training may bias the backbone toward a specific parameterization of the discrete code space. To reduce this dependency and further promote robustness to the choice of coding scheme, we periodically reinitialize the projection head $g _ { \phi }$ every n epochs by resampling its parameters from a fixed initialization distribution $\mathcal { D }$ .

Each reset induces a new random mapping from the backbone to the binary code space, encouraging the backbone to learn features that remain predictive across multiple discrete encodings rather than adapting to a single fixed mapping. From a communication viewpoint, each projection head can be interpreted as a distinct parameterization of the discrete channel, and exposing the backbone to multiple such parameterizations promotes representations that make more effective use of the fixed-capacity discrete bottleneck.

# 5. Experiments

We follow the SimDINO setup, and by extension DINO, as closely as possible to ensure a fair comparison. Experiments are conducted using a Vision Transformer backbone (ViT-B/16) trained from random initialization, together with the same three-layer MLP projection head as in DINO. The projection head outputs B = 256 units, which corresponds to the default dimensionality used in DINO and SimDINO; this value is not tuned for our method and is kept identical across all models. In our framework, these outputs parameterize a 256-bit binary message in the discrete communication channel. Standard self-supervised data augmentations are used to generate multiple views, with local views fed to the student branch.

All models are trained using the official SimDINO implementation as a common optimization framework, ensuring identical architectures, optimization settings, and data augmentations across methods. Models are trained for 100 epochs on ImageNet-1K with a batch size of 256. For our method, the projection head parameters ϕ are periodically re-initialized during training. We reset the projection head every n = 10 epochs, a moderate frequency that we found to provide a good trade-off between stability and diversity; a detailed ablation on n is reported in Appendix A.

We do not include comparisons with DINOv2 (or v3), as it introduces a patch-level training objective that is not directly compatible with our binary agreement framework. In particular, applying a multi-label loss at the patch level is not necessarily desirable, as local patches do not naturally admit a multi-label semantic interpretation. While it would be possible to restrict the binary agreement objective to the global views only, this would result in a hybrid setup that complicates both evaluation and interpretation, and would blur the core contribution of this work.

In the remainder of the paper, we refer to our discrete communication framework as BITS (Binary Information Transmission for Self-Supervision).

# 5.1. ImageNet-1k Classification and Retrieval

We evaluate the learned continuous backbone representations on ImageNet-1K using two complementary tasks: classification and retrieval. Classification assesses global invariance and semantic alignment, while retrieval is particularly sensitive to the structure of the embedding space, as it relies on fine-grained similarity rather than class-level separation. This makes retrieval a natural benchmark for evaluating the benefits of factorized and multi-label representations. We report k-NN classification, linear probing accuracy, and mAP@ALL on the retrieval split of FFPQ (Liang et al., 2023).

We compare DINO (softmax-based agreement with a fixed continuous head), SimDINO (cosine-based agreement with a fixed continuous head), BITS-fixed (binary agreement with a fixed head), and BITS-reset (binary agreement with periodic head reinitialization every n = 10 epochs). Results are reported in Tab. 1.

Table 1. Classification and retrieval performance comparison on ImageNet-1k. 

<table><tr><td>Method</td><td>mAP</td><td>k-NN</td><td>linear probing</td></tr><tr><td>DINO</td><td>35.68</td><td>72.39</td><td>76.3</td></tr><tr><td>SimDINO</td><td>38.62</td><td>69.52</td><td>75.3</td></tr><tr><td>BITS-fixed</td><td>43.44</td><td>73.32</td><td>76.7</td></tr><tr><td>BITS-reset</td><td>50.64</td><td>73.5</td><td>77.8</td></tr></table>

Overall, both BITS variants outperform DINO and SimDINO across all metrics, with BITS-reset achieving the strongest performance. The largest relative gains are observed for retrieval, highlighting the benefit of multilabel agreement for tasks that rely on fine-grained similarity. These results indicate that discretizing the agreement mechanism, rather than the representation itself, induces more structured embeddings and yields consistent improvements over SimDINO, despite both approaches relying on codingrate regularization. The additional gains of BITS-reset over BITS-fixed further show that periodic head reinitialization helps fully exploit the backbone capacity under a discrete agreement objective.

# 5.2. Representation Geometry and Expressivity

To better understand the gains observed for retrieval, we next analyze the geometry and expressivity of the learned representations.We first study how variance is distributed across backbone dimensions following (Kim et al., 2025; Li et al., 2022a). We extract continuous backbone features from the ImageNet-1K validation set and compute the empirical covariance matrix, whose eigenvalues $\lambda _ { 1 } \geq \cdots \geq \lambda _ { d }$ are used to estimate the cumulative explained variance, computed for the first i dimensions as: $\begin{array} { r } { c v _ { i } = \sum _ { j = 1 } ^ { i } \lambda _ { j } \Big / \sum _ { j = 1 } ^ { d } \lambda _ { j } } \end{array}$ .

Fig. 1 reports the cumulative explained variance as a function of the number of backbone dimensions for all compared methods. Despite both relying on coding–rate regularization, SimDINO’s cosine-based agreement concentrates variance more strongly along a small number of dominant dimensions than BITS-fixed. In contrast, the discrete multi-label agreement used in BITS-fixed promotes a more balanced allocation of variance across dimensions.

DINO exhibits a variance distribution comparable to BITSfixed, indicating that enforcing a mono-label agreement already encourages a degree of factorization. However, this factorization arises from prototype selection and remains inherently limited to a single active semantic mode per sample. Finally, BITS-reset further improves variance uniformity, suggesting that periodically reinitializing the projection head encourages the backbone to distribute information more evenly by preventing adaptation to a single fixed coding scheme.

![](images/6a277021badac5a7e3c0480abde852d00215d0c15908f2f8352d0cf6b1366257.jpg)

<details>
<summary>line</summary>

| eigenvalue index | DINO  | SimDINO | BITS-fixed | BITS-reset |
| ---------------- | ----- | ------- | ---------- | ---------- |
| 0                | 0.0   | 0.0     | 0.0        | 0.0        |
| 200              | 0.7   | 0.8     | 0.75       | 0.6        |
| 400              | 0.9   | 0.95    | 0.9        | 0.85       |
| 600              | 0.95  | 0.98    | 0.95       | 0.9        |
| 800              | 0.98  | 0.99    | 0.98       | 0.95       |
| 1000             | 1.0   | 1.0     | 1.0        | 1.0        |
</details>

Figure 1. Cumulative explained variance of the dimensions of the representations of ImageNet-1k validation set.

While variance distribution characterizes how information is allocated across dimensions, it does not capture how much effective representational capacity is utilized. To quantify expressivity, we compute the effective representation dimension $d _ { e f f } = \left( \textstyle \sum _ { j } \lambda _ { j } \right) ^ { 2 } / \sum _ { j } \lambda _ { j } ^ { 2 }$ and the effective representation rank $r _ { e f f } ~ = ~ \exp \Big ( ~ -$ $\sum _ { j } \tilde { \lambda _ { j } } \log \tilde { \lambda _ { j } } )$ where $\begin{array} { r l r } { \tilde { \lambda _ { j } } } & { { } = } & { \lambda _ { j } / \sum _ { k } \lambda _ { k } } \end{array}$ in Tab. 2. While DINO and BITS-

fixed achieve comparable levels of factorization, BITS-fixed consistently attains higher effective dimensionality, reflecting the increased expressivity enabled by multi-label representations. While mono-label

Table 2. Dimensional collapse comparison between models. 

<table><tr><td>Method</td><td> $d_{eff}$ </td><td> $r_{eff}$ </td></tr><tr><td>DINO</td><td>233</td><td>400</td></tr><tr><td>SimDINO</td><td>156</td><td>324</td></tr><tr><td>BITS-fixed</td><td> $\underline{250}$ </td><td> $\underline{404}$ </td></tr><tr><td>BITS-reset</td><td> $\underline{358}$ </td><td> $\underline{480}$ </td></tr></table>

agreement promotes separation across samples, multi-label discrete agreement allows multiple semantic factors to be simultaneously active, increasing the number of distinguishable configurations.

BITS-reset further increases both $d _ { e f f }$ and $r _ { e f f }$ , indicating a more effective utilization of the fixed-capacity discrete channel. This suggests that periodic head reinitialization not only promotes factorization, but also encourages the backbone to exploit the full combinatorial expressivity of the binary representation space.

# 5.3. Ablations Studies

In this subsection, we perform multiple ablations to test different components of our framework.

Binary cross-entropy vs. cosine agreement on logits. We compare two agreement formulations within the same discrete binary channel: a per-bit binary crossentropy loss and a cosine similarity applied to the student logits and thresholded teacher outputs (with targets in $\{ - 1 , + 1 \} ,$ . Both variants use identical architectures and channel dimensionality. Tab. 3 shows that enforcing agreement independently for each bit leads to substantially better retrieval performance than using a global cosine similarity, despite identical discretization and capacity. This indicates that discretization alone is not sufficient: effectively exploiting a multi-label binary channel requires an agreement objective that operates at the

Table 3. Effect of agreement formulation within a discrete binary channel. 

<table><tr><td>Agreement</td><td>BCE</td><td>Cosine</td></tr><tr><td>mAP</td><td>43.44</td><td>36.34</td></tr></table>

level of individual dimensions. In contrast, cosine similarity treats the binary message as a single vector and allows compensations across bits, resulting in weaker utilization of the discrete representation.

Hard vs. soft binary targets. We compare soft continuous targets with hard binary targets obtained via thresholding in the discrete communication framework. Without temperature tuning, soft targets obtained through sigmoid activations lead to unstable optimization and fail to reliably converge. When the temperature is carefully tuned, the best-performing soft-target regime $( \mathbf { e . g . } , \tau = 0 . 1 )$ yields performance close to hard binary targets by producing nearbinary labels. However, this regime is narrow and highly sensitive to the chosen value (see Appendix A). In contrast, hard binary targets directly provide strictly binary labels in a deterministic and robust manner, yielding a stable supervision signal without additional tuning.

Coding-rate control. We analyze the effect of the codingrate coefficient β, which controls the utilization of the discrete binary channel. We report retrieval performance after 10 training epochs for different fixed values of $\beta$ in Tab. 4. Extremely small or

large values hinder effective channel usage, while $\beta \quad = \quad 0 . 1$ yields stable optimization and is used in all subsequent experiments.

Table 4. Effect of the coding rate regularization coefficient β after 10 epochs of training. 

<table><tr><td>β</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.5</td></tr><tr><td>mAP</td><td>0.22</td><td>10.46</td><td>7.21</td><td>1.1</td></tr></table>

Periodically randomized heads on cosine vs. on BCE. Applying periodic head resets to SimDINO (cosine-based SSL) leads to unstable training and divergence, whereas the same procedure remains stable under binary cross-entropy agreement. This highlights a key difference between global cosine alignment and per-bit binary supervision: cosinebased objectives are sensitive to abrupt changes in teacher representations, while the bounded per-dimension gradients induced by BCE make discrete agreement more robust to head reinitialization.

Overall, these ablations indicate that discrete agreement is driven by per-bit binary supervision with hard thresholding, with coding-rate regularization controlling channel utilization; head reinitialization is an optional optimization enabled by discretization.

# 5.4. Downstream Transfer under Domain Shift

We assess the transferability of BITS pre-trained backbone to downstream tasks under varying degrees of domain shift (using frozen representations unless stated otherwise).

From In-Domain to Light Domain Shift Retrieval. We first evaluate retrieval performance under in-domain and light domain shift using ImageNetV2, ImageNet100, PascalVOC2012 (Everingham et al.), and COCO2014 (Lin et al., 2014). These datasets progressively depart from the ImageNet-1k training distribution while preserving similar object-level semantics. As reported in Tab. 5, both BITS variants consistently outperform the continuous baselines across all settings, indicating that discrete binary agreement preserves transferable semantic structure beyond the source domain.

Table 5. Retrieval performance using mAP on in-domain and light out-of-distribution datasets. 

<table><tr><td>Method</td><td>INv2</td><td>IN100</td><td>Pascal</td><td>Coco</td></tr><tr><td>DINO</td><td>38.77</td><td>76.11</td><td>75.84</td><td>83.59</td></tr><tr><td>SimDINO</td><td>41.48</td><td>72.86</td><td>75.49</td><td>82.9</td></tr><tr><td>BITS-fixed</td><td>46.15</td><td>79.44</td><td>78.34</td><td>85.4</td></tr><tr><td>BITS-reset</td><td>52.9</td><td>82.29</td><td>79</td><td>85.31</td></tr></table>

Unsupervised Object Detection and Segmentation. We further assess transferability on object-centric tasks using MaskCut (Wang et al., 2023) for unsupervised object detection and instance segmentation on COCO2017 (Lin et al., 2014). As shown in Tab. 6, both BITS variants achieve stronger performance than DINO and SimDINO, confirming that the learned representations capture complementary object-level semantics, despite not being explicitly trained for dense prediction.

Video Object Segmentation. We evaluate local patch representations on DAVIS2017 (Pont-Tuset et al., 2017) using nearest-neighbor propagation between consecutive frames, following the DINO evaluation protocol. Results in Tab. 7 show that discrete agreement preserves spatially coherent patch features, despite not being explicitly optimized for dense prediction.

Table 6. Unsupervised object detection and segmentation via Mask-Cut evaluated on COCO val2017. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Detection</td><td colspan="3">Segmentation</td></tr><tr><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td>AP</td></tr><tr><td>DINO</td><td>3.9</td><td>1.5</td><td>1.8</td><td>3.1</td><td>1</td><td>1.4</td></tr><tr><td>SimDINO</td><td>5</td><td>1.6</td><td>2.2</td><td>4</td><td>1.4</td><td>1.6</td></tr><tr><td>BITS-fixed</td><td>6</td><td>2.5</td><td>2.8</td><td>5</td><td>1.8</td><td>2.2</td></tr><tr><td>BITS-reset</td><td>6.4</td><td>2.2</td><td>2.8</td><td>5.3</td><td>1.6</td><td>2.2</td></tr></table>

Table 7. Semi-supervised object segmentation on Davis 2017 Video dataset. 

<table><tr><td>Method</td><td> $(\mathcal{J}\&\mathcal{F})_m$ </td><td> $\mathcal{J}_m$ </td><td> $\mathcal{F}_m$ </td></tr><tr><td>DINO</td><td>61.83</td><td>60.44</td><td>63.21</td></tr><tr><td>SimDINO</td><td>61.98</td><td>60.06</td><td>63.9</td></tr><tr><td>BITS-fixed</td><td>62.78</td><td>60.99</td><td>64.56</td></tr><tr><td>BITS-reset</td><td>62.51</td><td>60.81</td><td>64.21</td></tr></table>

Severe Domain Shift: Fine-grained Recognition. We evaluate transfer under severe out-of-distribution shift using Birds525 (Piosenka, 2023), Food101 (Bossard et al., 2014), iNat2019 (Horn et al., 2019) and PlantNet300k (Garcin et al., 2021), four fine-grained and highly domain-specific datasets relying on localized visual cues. Given the severity of the domain shift and the fine-grained nature of the tasks, we focus on linear probing accuracy, which provides a standard protocol for evaluating semantic transfer in this setting; additional metrics are reported in Appendix C.

Table 8. Linear probing accuracy on fine-grained out-ofdistribution datasets. 

<table><tr><td>Method</td><td>Birds525</td><td>Food101</td><td>iNat2019</td><td>PlantNet</td></tr><tr><td colspan="5">Frozen ImageNet-1k pre-trained weights</td></tr><tr><td>DINO</td><td>87.05</td><td>75.59</td><td>36.63</td><td>67.66</td></tr><tr><td>SimDINO</td><td>92</td><td>75.79</td><td>36.2</td><td>67.18</td></tr><tr><td>BITS-fixed</td><td>95.66</td><td>81.52</td><td>53.2</td><td>72.16</td></tr><tr><td>BITS-reset</td><td>84.15</td><td>74.78</td><td>29.74</td><td>63.45</td></tr><tr><td colspan="5">Self-supervised fine-tuned on target domain</td></tr><tr><td>DINO</td><td>77.52</td><td>70.9</td><td>24.82</td><td>71.79</td></tr><tr><td>SimDINO</td><td>91.09</td><td>66.19</td><td>29.08</td><td>70.04</td></tr><tr><td>BITS-fixed</td><td>96.72</td><td>82.88</td><td>54.39</td><td>80.04</td></tr><tr><td>BITS-reset</td><td>95.35</td><td>82.69</td><td>49.01</td><td>79.1</td></tr></table>

As shown in Tab. 8, clear differences emerge between agreement mechanisms under severe domain shift. Among frozen models, BITS-fixed achieves the strongest linear probing performance, indicating that discrete binary agreement with a fixed coding scheme yields robust and linearly separable semantic factors that generalize across fine-grained domains. In contrast, BITS-reset underperforms in the frozen setting, revealing a robustness–factorization trade-off: periodic head resets encourages highly factorized but more sourcespecific representations. When self-supervised fine-tuning on the target domain is allowed, this gap is largely closed, with BITS-reset, reinitialized only once at the start of the 10-epoch adaptation, reaching performance comparable to BITS-fixed. By contrast, continued self-supervised training of continuous baselines does not consistently improve performance and can even degrade accuracy.

# 5.5. Analysis of the Learned Binary Language

Our framework introduces discrete communication as a selfsupervised pretext task, inducing explicit binary codes at the output of the projection head. Although these codes are not used directly at inference time, analyzing their structure provides insight into how discrete agreement organizes semantic information.

Fig. 2 visualizes the learned binary codes on INet10 (Howard, 2019) using t-SNE with Hamming distance, embedding the codes in three dimensions and visualizing different two-dimensional projections. Across projections, the embeddings display a recurring ring-like structure, with samples distributed at comparable Hamming distances, while avoiding collapse or domination by a small subset of bits. This geometry is consistent with a balanced use of the fixed-capacity binary channel induced by the discrete communication objective.

![](images/61d330f5839872d87c9b0aa329b5079ecd22c81b6957d4ba6e8be6b064dff48b.jpg)

<details>
<summary>bubble</summary>

| t-SNE | Value |
|-------|-------|
| 0     | 1     |
| 1     | 2     |
| 2     | 0     |
</details>

Figure 2. Visualization of t-SNE embeddings of the hash codes.

To quantify the information content of the codes, we measure their entropy. The average marginal entropy per bit is approximately 0.9, indicating that most bits are active and well balanced. More strikingly, the entropy computed over contiguous 8-bit blocks reaches approximately 7.1 bits out of a maximum of 8. This result goes beyond marginal balance and indicates that the discrete communication objective promotes high joint entropy over groups of bits, rather than concentrating information independently or redundantly across dimensions. We further evaluate the usefulness of the learned binary codes for retrieval. Using the full 256- bit representation yields an mAP of 47.59, compared to 50.64 when using the continuous backbone embeddings. Subsampling the code to 128, 64, and 32 bits results in a gradual performance decrease (45.97, 43.73, and 40.35, respectively). The relatively smooth degradation indicates that retrieval performance does not rely on a small subset of bits, but instead draws on information distributed across the binary channel, consistent with the high joint entropy observed in the code analysis.

![](images/9e06af399e5590835edbc27a655f6cbdddc16c7777c26274c32d4b08ab82d2f4.jpg)

<details>
<summary>text_image</summary>

class "tench"
class "french horn"
bit = 0
bit = 1
</details>

Figure 3. Visualization of images from two ImageNet classes conditioned on the value of bit 0 in each row. Top row (bit=0) contains humans. Bottom row (bit=1) consistently shows no humans.

To qualitatively analyze the semantic structure of the learned binary codes, we visualize images conditioned on the value of bit number 0 in Fig. 3 (additional examples are reported in Appendix D). For this bit, we observe a consistent separation between images depicting objects in isolation and images showing objects embedded in a scene with human presence or interaction. Importantly, this behavior is shared across semantically distinct classes, such as tench and french horn, indicating that the bit does not encode class-specific information but rather captures a higher-level contextual attribute. This suggests that individual bits can represent reusable semantic cues related to scene context or object usage, which are shared across classes.

# 6. Conclusion

We introduced a discrete communication perspective on self-supervised learning, where semantic consistency is enforced through agreement over a fixed-capacity binary channel rather than continuous similarity. By aligning teacher and student representations through binary messages, our approach induces structured, multi-label representations that are more factorized than those learned with standard continuous agreement objectives, while remaining competitive or superior across a wide range of downstream tasks. Beyond the specific instantiation studied in this work, our results suggest that discrete communication offers a general and flexible principle for representation learning. Future work may explore richer discrete languages beyond binary codes, such as learned vocabularies or symbolic tokens, as well as extensions to other self-supervised settings, including multimodal alignment and temporal modeling.

# Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Assran, M., Caron, M., Misra, I., Bojanowski, P., Bordes, F., Vincent, P., Joulin, A., Rabbat, M., and Ballas, N. Masked Siamese Networks for Label-Efficient Learning. In European conference on computer vision, pp. 456–473. Springer, 2022.   
Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619–15629, 2023.   
Bao, H., Dong, L., Piao, S., and Wei, F. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.   
Bardes, A., Ponce, J., and LeCun, Y. Vicreg: Varianceinvariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.   
Bossard, L., Guillaumin, M., and Van Gool, L. Food-101– mining discriminative components with random forests. In European conference on computer vision, pp. 446–461. Springer, 2014.   
Cao, H., Huang, L., Nie, J., and Wei, Z. Unsupervised Deep Hashing With Fine-Grained Similarity-Preserving Contrastive Learning for Image Retrieval. IEEE Transactions on Circuits and Systems for Video Technology, 34 (5):4095–4108, 2023.   
Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., and Joulin, A. Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. Advances in neural information processing systems, 33:9912–9924, 2020.   
Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., ´ Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.   
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PmLR, 2020.

Dong, X., Bao, J., Zhang, T., Chen, D., Zhang, W., Yuan, L., Chen, D., Wen, F., Yu, N., and Guo, B. Peco: Perceptual codebook for bert pre-training of vision transformers. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 552–560, 2023.

El-Nouby, A., Klein, M., Zhai, S., Bautista, M. A., Toshev, A., Shankar, V., Susskind, J. M., and Joulin, A. Scalable pre-training of large autoregressive image models. arXiv preprint arXiv:2401.08541, 2024.

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., and Zisserman, A. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results. http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html.

Garcin, C., Bonnet, P., Affouard, A., Lombardo, J.-C., Chouet, M., Servajean, M., Lorieul, T., Salmon, J., et al. Pl@ ntnet-300k: a plant image dataset with high label ambiguity and a long-tailed distribution. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Gong, Q., Wang, L., Lai, H., Pan, Y., and Yin, J. ViT2Hash: Unsupervised Information-Preserving Hashing. arXiv preprint arXiv:2201.05541, 2022.

Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P., ´ Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al. Bootstrap Your Own Latent A New Approach to Self-Supervised Learning. Advances in neural information processing systems, 33:21271–21284, 2020.

He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.

Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., and Bengio, Y. Learning Deep Representations by Mutual Information Estimation and Maximization. arXiv preprint arXiv:1808.06670, 2018.

Horn, G. V., macaodha, Maggie, and Kan, W. inaturalist 2019 at fgvc6. https://kaggle.com/ competitions/inaturalist-2019-fgvc6, 2019. Kaggle.

Howard, J. Imagenette: A smaller subset of 10 easily classified classes from imagenet, March 2019. URL https://github.com/fastai/imagenette.

Kim, D., Sohn, C.-B., Kim, D.-Y., and Kim, D.-Y. A taxonomy and theoretical analysis of collapse phenomena in unsupervised representation learning. Mathematics, 13(18), 2025. ISSN 2227-7390. doi: 10.3390/math13182986. URL https://www.mdpi.com/2227-7390/13/ 18/2986.   
Lee, Y., Kim, W., Park, W., and Choi, S. Discrete Infomax Codes for Supervised Representation Learning. Entropy, 24(4):501, 2022.   
Li, A. C., Efros, A. A., and Pathak, D. Understanding collapse in non-contrastive siamese representation learning. In European conference on computer vision, pp. 490–505. Springer, 2022a.   
Li, Y. and van Gemert, J. Deep Unsupervised Image Hashing by Maximizing Bit Entropy. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 2002–2010, 2021.   
Li, Z., Chen, Y., LeCun, Y., and Sommer, F. T. Neural Manifold Clustering and Embedding. arXiv preprint arXiv:2201.10000, 2022b.   
Liang, Y., Zhang, S., Li, L. K., and Wang, X. Unleashing the full potential of product quantization for large-scale image retrieval. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 61712–61724. Curran Associates, Inc., 2023.   
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: ´ Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pp. 740– 755. Springer, 2014.   
Luo, X., Wang, H., Wu, D., Chen, C., Deng, M., Huang, J., and Hua, X.-S. A Survey on Deep Hashing Methods. ACM Transactions on Knowledge Discovery from Data, 17(1):1–50, 2023.   
Ma, Z., Wang, S., Luo, X., Gu, Z., Chen, C., Li, J., Hua, X.- S., and Lu, G. HARR: Learning Discriminative and High-Quality Hash Codes for Image Retrieval. ACM Transactions on Multimedia Computing, Communications and Applications, 20(5):1–23, 2024.   
Moummad, I., Zaher, K., Goeau, H., and Joly, A. Image ¨ Hashing via Cross-View Code Alignment in the Age of Foundation Models. arXiv preprint arXiv:2510.27584, 2025.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features

without supervision. arXiv preprint arXiv:2304.07193, 2023.   
Ozsoy, S., Hamdan, S., Arik, S., Yuret, D., and Erdogan, A. Self-Supervised Learning with an Information Maximization Criterion. Advances in Neural Information Processing Systems, 35:35240–35253, 2022.   
Piosenka, G. Birds 525 species - image classification. https://www.kaggle.com/datasets/ gpiosenka/100-bird-species, May 2023. Kaggle dataset.   
Pont-Tuset, J., Perazzi, F., Caelles, S., Arbelaez, P., Sorkine- ´ Hornung, A., and Van Gool, L. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.   
Shen, X., Cai, H., Gong, X., and Zheng, Y. Contrastive Transformer Masked Image Hashing for Degraded Image Retrieval. In Proceedings of the Thirty-ThirdInternational Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence, 2024.   
Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, ´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.   
Tian, Y., Krishnan, D., and Isola, P. Contrastive Multiview Coding. In European conference on computer vision, pp. 776–794. Springer, 2020.   
Tong, S., Chen, Y., Ma, Y., and Lecun, Y. EMP-SSL: Towards Self-Supervised Learning in One Training Epoch. arXiv preprint arXiv:2304.03977, 2023.   
Van Assel, H., Ibrahim, M., Biancalani, T., Regev, A., and Balestriero, R. Joint embedding vs reconstruction: Provable benefits of latent space prediction for self supervised learning. arXiv preprint arXiv:2505.12477, 2025.   
Wang, X., Girdhar, R., Yu, S. X., and Misra, I. Cut and learn for unsupervised object detection and instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3124– 3134, 2023.   
Wu, Z., Zhang, J., Pai, D., Wang, X., Singh, C., Yang, J., Gao, J., and Ma, Y. Simplifying dino via coding rate regularization. arXiv preprint arXiv:2502.10385, 2025.   
Xu, S., Ma, Z., Chai, W., Chen, X., Jin, W., Chai, J., Xie, S., and Yu, S. X. Next-embedding prediction makes strong vision learners. arXiv preprint arXiv:2512.16922, 2025.   
Yang, L., Li, S.-W., Li, Y., Lei, X., Wang, D., Mohamed, A., Zhao, H., and Xu, H. In Pursuit of Pixel Supervision

for Visual Pre-training. arXiv preprint arXiv:2512.15715, 2025.   
YM., A., C., R., and A., V. Self-labelling via simultaneous clustering and representation learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum? id=Hyx-jyBFPr.   
Yu, Y., Chan, K. H. R., You, C., Song, C., and Ma, Y. Learning diverse and discriminative representations via the principle of maximal coding rate reduction, 2020a. URL https://arxiv.org/abs/2006.08558.   
Yu, Y., Chan, K. H. R., You, C., Song, C., and Ma, Y. Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction. Advances in neural information processing systems, 33:9422–9434, 2020b.   
Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., and Kong, T. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021.

# A. Supplementary Ablations.

Appendix A presents complementary ablation studies that further justify the design choices of our method and analyze their impact beyond the settings reported in the main text. Unless stated otherwise, ablations follow the same training protocol as the main experiments (100 epochs), ensuring that observed differences reflect relative trends rather than changes in training budget.

Head reset frequency. This ablation further characterizes the optional head reinitialization mechanism introduced in the main paper. In particular, We study the effect of the projection head reset frequency n on retrieval performance. As shown in Tab. 9, periodically resetting the head significantly improves the model compared to keeping a fixed head $( n = \infty )$ , which shows that periodically changing the parameterization of the discrete channel allows the backbone to further adapt its representations to the source domain, beyond what is achieved with a fixed coding scheme. Resetting too frequently degrades performance, likely because the model does not have sufficient time to adapt to each coding scheme.

Table 9. Effect of head reset frequency n. 

<table><tr><td>n</td><td>1</td><td>5</td><td>10</td><td>20</td><td>∞</td></tr><tr><td>mAP</td><td>45.21</td><td>49.68</td><td>50.64</td><td>50.47</td><td>43.44</td></tr></table>

Hard vs. soft binary targets. These experiments further justify the target construction choice adopted in the main experiments, namely hard thresholding at 0.5. This choice naturally follows from the Bernoulli parameterization and corresponds to maximum a posteriori decoding under a symmetric prior. We compare this strategy with soft continuous targets obtained via sigmoid activation. As shown in Tab. 10, soft targets lead to unstable optimization at high temperatures, with the binary cross-entropy loss failing to converge. We attribute this behavior to the ambiguity and instability of the targets across training iterations. When using lower temperatures, the model operates closer to the thresholding regime and can converge; however, performance remains sensitive to temperature tuning and consistently slightly inferior. In contrast, deterministic hard thresholding yields stable optimization and the best performance, providing a stronger and more reliable supervision signal under the discrete agreement objective.

Table 10. Effect of hard vs. soft targets with different temperatures τ . 

<table><tr><td rowspan="2">target</td><td colspan="5">soft</td><td rowspan="2">hard</td></tr><tr><td> $\tau = 1$ </td><td> $\tau = 0.2$ </td><td> $\tau = 0.1$ </td><td> $\tau = 0.05$ </td><td> $\tau = 0.01$ </td></tr><tr><td>mAP</td><td>collapse</td><td>43.06</td><td>43.28</td><td>43.06</td><td>43.19</td><td>43.44</td></tr></table>

Number of bits B. We examine how the dimensionality of the the projection head affects our model. As reported in Tab. 11, using too few bits limits the representation capacity of our model. In contrast, a larger number of bits decreases performance, which we hypothesize is due to over-parameterization and optimization difficulty, especially of the regularization objective. $B = 2 5 6$ provides a good compromise between capacity and trainability. Notably, $B = 2 5 6$ corresponds to the default projection dimensionality used in DINO-style frameworks, suggesting that this choice reflects a broadly effective operating point rather than dataset-specific or agreement-specific tuning.

Table 11. Effect of the number of bits B. 

<table><tr><td>B</td><td>128</td><td>256</td><td>512</td></tr><tr><td>mAP</td><td>38.46</td><td>43.05</td><td>39.32</td></tr></table>

Gradient clipping. Tab. 12 shows the effect of varying the clipping threshold cg (maximum gradient norm) on the model performance. Performance remains relatively stable across a broad range of clipping value with cg = 1 providing the best performance.

Table 12. Effect of gradient clipping value cg. 

<table><tr><td>cg</td><td>0.3</td><td>0.5</td><td>1</td><td>2</td><td>3</td></tr><tr><td>mAP</td><td>42.52</td><td>42.69</td><td>43.05</td><td>42.78</td><td>42.68</td></tr></table>

End-of-training learning rate. We find that the choice of the minimum learning rate achieved at the end of training affects the model performance, as shown in Tab. 13. As training progresses, BCE saturates rapidly and gradients vanish near convergence, making updates extremely small. This behavior reflects a general property of binary cross-entropy objectives near convergence, rather than a limitation specific to our discrete agreement formulation. Setting a slightly higher min lr allows the model to continue making meaningful weight updates.

Table 13. Effect of end-of-training learning rate min lr. 

<table><tr><td>min_lr</td><td> $10^{-6}$ </td><td> $5 \times 10^{-5}$ </td></tr><tr><td>mAP</td><td>43.05</td><td>43.44</td></tr></table>

# B. Self-supervised Training Curves.

Fig. 4 reports the evolution of validation retrieval performance during SSL training for continuous agreement baselines and our discrete agreement variants. The figure shows that our discrete binary agreement models exhibit faster convergence and consistently higher retrieval performance throughout training. While DINO exhibits a strong start, its convergence gradually slows down with epochs. In contrast, SimDINO starts lower but rapidly accelerates, gaining important convergence speed. On the other hand, our models both benefit from a substantial head start and maintain a consistently strong convergence rate. In particular, BITS-reset shows a consistent improvement, which highlights the benefit of periodic head reset for in-domain specialization. This behavior further showcases that discrete binary agreement provides a stronger supervisory signal early in training and leads to more structured representations as training progresses.

![](images/08e9643258cfdd9b0322298e980550aace69b1325f54dd701f9631f1bb91165e.jpg)

<details>
<summary>line</summary>

| training epoch | DINO  | SimDINO | BITS-fixed | BITS-reset |
| -------------- | ----- | ------- | ---------- | ---------- |
| 0              | 10.0  | 5.0     | 10.0       | 12.0       |
| 20             | 20.0  | 15.0    | 25.0       | 30.0       |
| 40             | 28.0  | 25.0    | 32.0       | 40.0       |
| 60             | 32.0  | 30.0    | 38.0       | 45.0       |
| 80             | 35.0  | 37.0    | 42.0       | 48.0       |
| 100            | 36.0  | 39.0    | 43.0       | 50.0       |
</details>

Figure 4. Pre-training on ImageNet-1k: mAP score evolution across epochs.

# C. More on Adaptation to Severe Domain Shift.

This appendix complements the Severe Domain Shift analysis of the main paper by reporting additional evaluation metrics, including mAP and k-NN classification alongside linear probing, and by extending the study to an additional fine-grained dataset.

We report in Tab. 14 results on Birds525 (Piosenka, 2023), a dataset that still shares partial semantic overlap with ImageNet, Food101 (Bossard et al., 2014), iNat2019 (Horn et al., 2019), and PlantNet300k (Garcin et al., 2021), that are highly specialized.

Table 16 reports retrieval (mAP), k-NN classification, and linear probing results on four fine-grained datasets under severe domain shift. Overall, frozen mAP and k-NN performance remain relatively low and unstable across methods, with substantial variability across datasets. This behavior is expected in fine-grained recognition settings, where classes are separated by subtle visual cues and exhibit strong inter-class ambiguity. In such regimes, global semantic structure and invariances learned during pre-training on other domains are often insufficient to support reliable retrieval or nearest-neighbor classification without adaptation. While some methods, such as SimDINO, achieve higher frozen mAP or k-NN scores on some datasets (e.g., Birds525), these trends do not translate into improved linear probing performance. Linear probing provides a relevant evaluation protocol under severe domain shift, as it explicitly allows the model to reweight and recombine learned features to resolve fine-grained distinctions.

Table 14. Retrieval and classification performance on different out-of-distribution datasets. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Birds525</td><td colspan="3">Food101</td><td colspan="3">iNat2019</td><td colspan="3">PlantNet300k</td></tr><tr><td>mAP</td><td>k-NN</td><td>LP</td><td>mAP</td><td>k-NN</td><td>LP</td><td>mAP</td><td>k-NN</td><td>LP</td><td>mAP</td><td>k-NN</td><td>LP</td></tr><tr><td colspan="13">Pre-trained out-of-domain ImageNet-1k weights</td></tr><tr><td>DINO</td><td>38.86</td><td>86.32</td><td>87.05</td><td>14.6</td><td>65.16</td><td>75.59</td><td>12.29</td><td>35.25</td><td>36.63</td><td>12.11</td><td>58.79</td><td>67.66</td></tr><tr><td>SimDINO</td><td>57.07</td><td>93.52</td><td>92</td><td>17.5</td><td>66.2</td><td>75.79</td><td>12.87</td><td>37.1</td><td>36.2</td><td>12.13</td><td>57.88</td><td>67.18</td></tr><tr><td>BITS-fixed</td><td>46.45</td><td>87.81</td><td>95.66</td><td>19.82</td><td>65.49</td><td>81.52</td><td>12.27</td><td>36.47</td><td>53.2</td><td>11.44</td><td>56.94</td><td>72.16</td></tr><tr><td>BITS-reset</td><td>31.46</td><td>73.71</td><td>84.15</td><td>16.05</td><td>53.53</td><td>74.78</td><td>8.76</td><td>23.83</td><td>29.74</td><td>5.93</td><td>43.05</td><td>63.45</td></tr><tr><td colspan="13">Self-supervised finetuned in-domain weights</td></tr><tr><td>DINO</td><td>34.94</td><td>81.25</td><td>77.52</td><td>22.1</td><td>65.89</td><td>70.9</td><td>14.24</td><td>35.38</td><td>24.82</td><td>32.03</td><td>70.44</td><td>71.79</td></tr><tr><td>SimDINO</td><td>57.18</td><td>92.61</td><td>91.09</td><td>13.73</td><td>57.67</td><td>66.19</td><td>14.57</td><td>35.97</td><td>29.08</td><td>19.48</td><td>63.08</td><td>70.04</td></tr><tr><td>BITS-fixed</td><td>59.86</td><td>91.62</td><td>96.72</td><td>33.17</td><td>75.67</td><td>82.88</td><td>19</td><td>40.59</td><td>54.39</td><td>34.8</td><td>72.12</td><td>80.04</td></tr><tr><td>BITS-reset</td><td>70.52</td><td>93.9</td><td>95.35</td><td>40.02</td><td>76.61</td><td>82.69</td><td>21.18</td><td>40.89</td><td>49.01</td><td>38.2</td><td>72.47</td><td>79.1</td></tr></table>

Under continued self-supervised fine-tuning on the target domain, BITS variants exhibit clear and consistent performance improvements across evaluation metrics. As discussed in the main paper, both BITS-fixed and BITS-reset achieve stronger linear probing accuracy than continuous baselines, confirming that discrete agreement provides a more effective inductive bias for adapting representations under severe domain shift. In addition to these gains in linear probing, continued selfsupervised adaptation also improves frozen retrieval and k-NN classification performance. In this setting, BITS-reset achieves the strongest results for mAP and k-NN classification, while BITS-fixed remains superior in linear probing. This complementary behavior reflects a trade-off between specialization and linear accessibility: periodic head reinitialization favors adaptation of the representation geometry for retrieval-based metrics, whereas a fixed discrete coding scheme preserves more linearly separable features for fine-grained classification.

# C.1. Probing Curves.

Figures 5 and 6 report respectively the linear probing and MLP probing performance on Birds525 and PlantNet300k as a function of the number of training iterations. The curves show that BITS-fixed exhibits a clear and consistent advantage from the very first iterations, indicating that the learned representations are immediately more linearly accessible under severe domain shift. This early performance gap persists throughout training and is observed for both linear and shallow nonlinear probes, further supporting the robustness of the semantic structure induced by discrete agreement.

![](images/f6be2ac74c6c1522f3cd497d0f6010b443813c6789ef51a7c17e320dfbdc2e5d.jpg)

<details>
<summary>line</summary>

| probing epoch | DINO  | SimDINO | BITS-fixed | BITS-reset |
| ------------- | ----- | ------- | ---------- | ---------- |
| 0             | 45.0  | 56.0    | 86.0       | 47.0       |
| 2             | 78.0  | 86.0    | 93.0       | 71.0       |
| 4             | 84.0  | 90.0    | 94.0       | 80.0       |
| 6             | 86.0  | 91.0    | 95.0       | 83.0       |
| 8             | 87.0  | 92.0    | 95.0       | 84.0       |
</details>

![](images/d5774ca37a6e83602eb5668e649e346920cbd2a89d92ae2c4abdae5eafc0905e.jpg)

<details>
<summary>line</summary>

| probing epoch | DINO  | SimDINO | BITS-fixed | BITS-reset |
| ------------- | ----- | ------- | ---------- | ---------- |
| 0             | 56.0  | 55.5    | 64.5       | 49.0       |
| 2             | 64.0  | 63.0    | 68.0       | 58.0       |
| 4             | 66.5  | 65.5    | 70.5       | 61.5       |
| 6             | 67.5  | 66.5    | 71.5       | 62.5       |
| 8             | 68.0  | 67.0    | 72.0       | 63.0       |
</details>

Figure 5. Linear probing results evolution on Birds525 (left) and PlantNet300k (right).

![](images/dae079e56aa68fdfa54d0a14b728140aadf6e94a24af78b13a04257581be2fff.jpg)

<details>
<summary>line</summary>

| probing epoch | DINO  | SimDINO | BITS-fixed | BITS-reset |
| ------------- | ----- | ------- | ---------- | ---------- |
| 0             | 15    | 20      | 70         | 25         |
| 2             | 55    | 70      | 90         | 50         |
| 4             | 70    | 85      | 95         | 65         |
| 6             | 80    | 90      | 97         | 70         |
| 8             | 82    | 92      | 98         | 75         |
</details>

![](images/d9e0f905a9b45ca7cc0caeff32c9a7843747e8ddcb6571eb32c54e6f7f9b593a.jpg)

<details>
<summary>line</summary>

| probing epoch | DINO  | SimDINO | BITS-fixed | BITS-reset |
| ------------- | ----- | ------- | ---------- | ---------- |
| 0             | 49.0  | 49.0    | 62.0       | 43.0       |
| 2             | 61.0  | 61.0    | 68.0       | 55.0       |
| 4             | 64.0  | 64.0    | 71.0       | 59.0       |
| 6             | 65.0  | 65.0    | 72.0       | 61.0       |
| 8             | 66.0  | 66.0    | 73.0       | 62.0       |
</details>

Figure 6. MLP probing results evolution on Birds525 (left) and PlantNet300k (right).

# C.2. Continued SSL Finetuning Curves.

Figure 7 reports the evolution of downstream retrieval performance during continued self-supervised fine-tuning on Birds525 and PlantNet300k. The curves show that BITS-reset achieves a markedly faster performance increase during adaptation, highlighting the benefit of head reinitialization for rapid specialization under severe domain shift.

![](images/9dbd3d1696786ca13c20814e8715bd3df5705a227197703933168ce592312055.jpg)

<details>
<summary>line</summary>

| finetuning epoch | DINO  | SimDINO | BITS-fixed | BITS-reset |
| ---------------- | ----- | ------- | ---------- | ---------- |
| 0                | 37.0  | 56.0    | 54.0       | 56.0       |
| 2                | 33.0  | 56.5    | 57.0       | 65.0       |
| 4                | 33.5  | 57.0    | 59.0       | 69.0       |
| 6                | 34.0  | 57.0    | 60.0       | 70.0       |
| 8                | 34.5  | 57.0    | 60.5       | 70.5       |
</details>

![](images/c241650484bbb6b54ca57fcc95c3761df3f9324661bdc0c7ca12eb2988ac5b90.jpg)

<details>
<summary>line</summary>

| finetuning epoch | DINO  | SimDINO | BITS-fixed | BITS-reset |
| ---------------- | ----- | ------- | ---------- | ---------- |
| 0                | 19.0  | 14.5    | 20.0       | 23.0       |
| 2                | 27.0  | 16.5    | 28.0       | 33.0       |
| 4                | 30.0  | 18.5    | 32.0       | 36.0       |
| 6                | 31.5  | 19.5    | 34.0       | 37.5       |
| 8                | 32.0  | 19.8    | 34.5       | 38.0       |
</details>

Figure 7. Full self-supervised finetuning results evolution on Birds525 (left) and PlantNet300k (right).

# D. More Qualitative Results.

We present additional qualitative visualization in Figs. 8, 9, 10. For three different bits, we visualize ImageNet-1k samples from different classes, conditioned on the bit value. Across all cases, we observe that individual bits activate for specific semantic attributes that are shared across classes, rather than encoding class-specific information. Samples activating a given bit exhibit coherent visual properties—such as contextual, structural, or object-level cues—that remain stable across different categories. These visualizations illustrate that the learned binary codes capture reusable semantic attributes that generalize beyond individual classes, supporting a compositional representation of visual concepts. This trans-class consistency highlights the structured and interpretable nature of the discrete language induced by the proposed discrete communication objective.

![](images/5c69b8aaf318bfa9b0f08ca632d9d3d3643e0c4aaf11782bac72700e89219811.jpg)  
Figure 8. Visualization of images from three ImageNet classes (warplane, black stork, kite) conditioned on the value of bit 17.

![](images/704a087af6b817acaaf38b464002bb062daf78d8e292bfcfa7805a32f2285693.jpg)  
Figure 9. Visualization of images from three ImageNet classes (bassinet, bib, nipple) conditioned on the value of bit 2.

![](images/1c753d28788068767ff2b6670555d4019cf9f824dfc1ed90e3e87a36e469cf93.jpg)  
Figure 10. Visualization of images from three ImageNet classes (English springer, malinois, church) conditioned on the value of bit 121.