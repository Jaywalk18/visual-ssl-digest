# HyFL-CLIP: Hyperbolic Fine-Tuning of CLIP for Robust Long-Context Understanding

Ji Ha Jang<sup>1,∗</sup> , Hayeon Kim<sup>1,∗</sup> , Chulwon Lee<sup>2</sup> , Junghun James Kim<sup>2</sup> , Se Young Chun<sup>1,2,3,†</sup>

<sup>1</sup>Dept. of Electrical and Computer Engineering, <sup>2</sup>IPAI, <sup>3</sup>INMC & AIIS, Seoul National University, Republic of Korea {jeeit17, khy5630, chul0e, jonghean12, sychun}@snu.ac.kr

Abstract. CLIP (Contrastive Language-Image Pre-training) has become a de facto paradigm for image-text alignment, but it struggles with long-context descriptions (> 77 tokens) due to absolute positional encoding and pretraining on short captions. In long contexts, sentences are often reordered, summarized, or partially omitted. Although prior works extend CLIP with longer positional encodings, they often sufer from degraded image-text alignment under such text perturbations. We attribute this limitation to the Euclidean contrastive objective, which enforces strict one-to-one matching and lacks explicit mechanisms for modeling hierarchical relationships between global context and its constituent elements. To address this issue, we propose HyFL-CLIP, a hyperbolic fine-tuning framework that distills the well-established textimage alignment learned in Euclidean CLIP into hyperbolic space via cross-manifold similarity distillation, leveraging its geometry to capture hierarchical and entailment relations. Our method models hierarchical semantics by linking summarized token-wise features, long-context descriptions, constituent short textual components, and images, capturing part–whole relationships via hyperbolic entailment with Einstein midpoint aggregation. Experiments on diverse benchmarks, including longcontext cross-modal retrieval, cross-modal retrieval with caption perturbations, intra-modality retrieval, and short-text cross-modal retrieval, show that HyFL-CLIP achieves more robust long-context understanding. In particular, it yields up to 19.5% improvement in long-text crossmodal retrieval under textual perturbations over the best prior method. We also show HyFL-CLIP can be seamlessly integrated into other model frameworks by applying it to Stable Difusion XL (SDXL). The project page is available at https://janeyeon.github.io/hyflclip.

Keywords: hyperbolic representation learning · long-context vision-language alignment · cross-manifold distillation

## 1 Introduction

Vision-language contrastive pre-training has laid the foundation for a wide range of vision-language learning tasks. Models such as CLIP [59], ALIGN [29] have achieved remarkable performance in text-image alignment by projecting image and text representations into a shared embedding space via contrastive learning. As a result, such models have been widely adopted not only for downstream tasks such as zero-shot classification and retrieval, but also for broader applications including image generation [57, 61] and large multi-modal systems [3, 39, 43, 84].

![](images/efd5d492ea69a31b826b9a027577862ddd8c14b944cc598c746950d323911ad2.jpg)  
Fig. 1: Robust long-context retrieval under caption perturbation. In long contexts, sentence order or structure may change preserving overall semantics, making robust text-image alignment essential. However, prior works on long-context understanding CLIP often exhibit degraded alignment under such perturbations. For example, even when only the sentence order is shufled (Order shufling), they fail to retrieve the correct image, missing key semantic elements such as the red pedestrian trafic signal, the black car, and the pedestrian wearing a white and red outfit.

However, long-text inputs (> 77 tokens) remain challenging for CLIP, as it is primarily trained on datasets composed of short captions and inherently supports a maximum input length of 77 tokens [79]. Existing approaches typically address this by extending positional encoding and aligning coarse and fine grained features in CLIP [4, 52, 75, 79], and designing training strategies to better learn long-text representations [16, 52, 72]. Although prior works on CLIP finetuning for long-context understanding [4, 16, 52, 72, 75, 79] have significantly improved long text-image retrieval performance, we observe that they remain highly sensitive to (i) changes in the number or ordering of sentences that constitute the long context and (ii) the deletion or substitution of only a few words (See Fig. 1). This limitation may partly stem from CLIP’s training paradigm, which primarily relies on one-to-one text-image matching and does not explicitly model entailment or part-whole relationships among semantic components. As a result, the global meaning of a long description can become concentrated on a few salient tokens, making the model sensitive to the removal or reordering of tokens or sentences and leading to degraded image–text alignment. Modeling long descriptions therefore requires capturing hierarchical relationships between the global context and its constituent semantic components.

Recently, hyperbolic space has been actively explored in vision–language contrastive learning to better model hierarchical relationships [13,54,60]. Unlike Euclidean space used by CLIP, hyperbolic space has negative curvature and naturally represents tree-like structures. However, most hyperbolic Vision-Language Models (VLMs) [13, 22, 54, 60, 67, 78] or Large Language Models (LLMs) [21] are trained from scratch, limiting their ability to leverage strong pretrained Euclidean models such as CLIP [59]. While recent studies have begun exploring fine-tuning approaches, they typically focus on single-modality adaptation or specific tasks [40, 55, 73, 78, 80, 82].

We propose HyFL-CLIP (Hyperbolic Fine-tuning for Long-context CLIP), which enhances robust long-context understanding by modeling semantic partwhole relationships. HyFL-CLIP transfers the well-established text-image alignment learned by Euclidean CLIP into hyperbolic space. To the best of our knowledge, this is the first work to leverage Euclidean vision–language relationships for generalized tasks in hyperbolic space. We summarize token-wise features using Einstein midpoint aggregation to connect long-context descriptions and their constituent elements with the visual modality. Part-whole relationships are modeled through hierarchical entailment, which relaxes strict one-to-one matching imposed by contrastive learning and aligns semantically related elements to better capture hierarchical structures. We demonstrate that HyFL-CLIP outperforms prior arts [4,16,52,72,75,79] across diverse long-context benchmarks. By effectively leveraging hyperbolic space, our model remains robust even when parts of the input text are missing or perturbed, whereas Euclidean baselines exhibit larger performance drops. Furthermore, HyFL-CLIP achieves up to 19.5% performance improvement on long-context cross-modal retrieval under caption perturbations, highlighting its ability to more robustly capture hierarchical semantic relationships in long contexts. We further adapt HyFL-CLIP to SDXL [57], demonstrating that it can be seamlessly integrated into existing frameworks. Our contributions are as follows:

• We introduce HyFL-CLIP, a hyperbolic fine-tuning framework that transfers the well-established image–text alignment learned by Euclidean CLIP into hyperbolic space, enabling well-pretrained Euclidean Vision–Language Models to operate efectively in hyperbolic representations.

• We bridge Euclidean and hyperbolic representations via cross-manifold similarity distillation, while aligning image and text using a hyperbolic geodesic contrastive loss. A hierarchical entailment loss further relaxes strict one-toone contrastive matching, enabling the model to better capture relationships between global context and its semantic components.

• HyFL-CLIP achieves strong performance across diverse long-context understanding benchmarks, consistently outperforming Euclidean baselines. In particular, it shows improved robustness under caption perturbations, maintaining stable retrieval performance even when the input text is reordered, partially removed, or corrupted.

## 2 Related Works

## 2.1 Vision-language foundation models

Vision–Language Models (VLMs) learn a shared embedding space between images and text, becoming key components for cross-modal understanding [18, 30, 58,63,65]. CLIP [59] aligns image and text using a contrastive objective, achieving strong zero-shot performance in tasks such as image–text retrieval and classification. Building on this paradigm, many pretrained VLMs [23, 26, 31, 35, 47] have been developed and widely used for vision–language tasks.

Although contrastive loss of CLIP is powerful, it also has several limitations. First, contrastive learning treats all non-paired samples as negatives in binary [7], even though some may share semantic overlap and act as false negatives in practice [11, 27]. In addition, CLIP-style training often relies heavily on the CLS token for global alignment, which can limit the model’s ability to capture hierarchical [19,48] or fine-grained structural associations [4,83]. To address these limitations, various approaches in Euclidean space have been explored [4, 11, 19, 27, 83] to improve the contrastive learning framework. In this work, we instead transfer the well-trained similarity geometry of CLIP into hyperbolic space and further refine it using hyperbolic entailment. This allows the model to more robustly retrieve correct image-text pairs even when long-context descriptions are perturbed.

## 2.2 Long-context understanding with CLIP

CLIP [59] can face challenges when handling text sequences longer than 77 tokens, as it is trained primarily on short captions and uses absolute positional encoding [79]. A common strategy is to extend CLIP to longer sequences using interpolated positional encoding and coarse to fine cross-modal alignment strategies, as explored in Long-CLIP [79] and HiMo-CLIP [75]. Subsequent works further enhance token-wise fine-grained visual–textual correspondence [4]. Other approaches introduce architectural modifications, such as relative positional encoding [52], dual-branch training for jointly handling short and long contexts [72], and dual-teacher distillation frameworks for long-context learning [16].

However, these approaches still rely on Euclidean embeddings with one-toone image–text alignment and do not explicitly model hierarchical inclusion relationships, which can lead to performance degradation under semantically preserving text modifications. In contrast, we leverage hyperbolic space to explicitly model hierarchical structures that connect summarized token-wise features, long-context descriptions, and the short textual components that compose them. By linking global semantic summaries with their constituent textual elements, the model can robustly preserve semantic understanding even when parts of the text are removed, reordered, or perturbed.

## 2.3 Hyperbolic representation learning in Vision-language models

Hyperbolic geometry provides an embedding space well suited for modeling finegrained and hierarchical relationships [17,64]. Due to its inherent tree-like structure, hyperbolic space naturally captures hierarchical data. As a result, hyperbolic embeddings have been widely applied in various domains, including graph, image understanding, and text [5, 9, 14, 33, 38, 44, 66, 68, 77].

Recently, several works have explored the use of hyperbolic geometry in VLMs [13, 22, 54, 60, 67, 78]. These models align image and text embeddings using geodesic contrastive losses, while further model hierarchical structure within the same modality [13, 60] or across modalities via part–to–whole relations [54]. Also, some Large Language Models (LLMs) [21] adopt hyperbolic manifold to better capture the hierarchical structure inherent in language.

Although prior works show promising results, they are generally trained from scratch and remain disconnected from well-pretrained Euclidean models, limiting their ability to leverage strong representations from models such as CLIP. While recent attempts address this gap by fine-tuning pretrained CLIP, they are often restricted to single-modality settings [73,78,80] or specific tasks and experimental setups [40,55,82], which may limit their applicability in more general settings. In contrast, we perform hyperbolic fine-tuning starting from a pretrained CLIP model. By guiding the model with an entailment objective, our approach preserves $\mathrm { C L I P } \mathrm { { ^ { \circ } s } }$ strong representations while enabling the text embedding space to capture relationships across diferent levels of semantic abstraction, leading to improved performance in long-context understanding.

## 3 Preliminaries

## 3.1 Hyperbolic geometry in the Lorentz model

Hyperbolic space is a Riemannian manifold with constant negative curvature $- \kappa .$ , where $\kappa \in \mathbb R ^ { + }$ . In this work, we employ the Lorentz (Minkowski) model as the underlying geometric space for hyperbolic fine-tuning, by distilling representations from a pre-trained Open-CLIP [12] model into the Lorentz manifold.

We consider the $( n + 1 )$ -dimensional Minkowski space $\mathbb { R } ^ { n + 1 }$ equipped with the Lorentzian inner product defined as below:

$$
\langle \mathbf {p}, \mathbf {q} \rangle_ {\mathbb {L}} = - p _ {\mathrm{time}} q _ {\mathrm{time}} + \langle \mathbf {p} _ {\mathrm{space}}, \mathbf {q} _ {\mathrm{space}} \rangle ,\tag{1}
$$

where $\langle \cdot , \cdot \rangle$ denotes the standard Euclidean inner product.

Under this metric, the n-dimensional Lorentz manifold $\mathbb { L } ^ { n }$ is defined as the upper sheet of the two-sheeted hyperboloid given by:

$$
\mathbb {L} ^ {n} = \left\{\mathbf {p} \in \mathbb {R} ^ {n + 1} \middle | \langle \mathbf {p}, \mathbf {p} \rangle_ {\mathbb {L}} = - \frac {1}{\kappa}, p _ {\text {time}} > 0 \right\}.\tag{2}
$$

Each point $\mathbf { p } \in \mathbb { L } ^ { n }$ is represented as below:

$$
\mathbf {p} = \left[ p _ {\text {time}}, \mathbf {p} _ {\text {space}} \right], \qquad p _ {\text {time}} = \sqrt {\frac {1}{\kappa} + \| \mathbf {p} _ {\text {space}} \| ^ {2}}.\tag{3}
$$

where $\mathbf { p } _ { \mathrm { s p a c e } } \in \mathbb { R } ^ { n }$ denotes the spatial component, $| | \cdot | |$ denotes the Euclidean norm, and the expression for $p _ { \mathrm { t i m e } }$ follows from the Lorentz manifold constraint.

The geodesic distance between two points $\mathbf { p } , \mathbf { q } \in \mathbb { L } ^ { n }$ is given by:

$$
d _ {\mathbb {L}} (\mathbf {p}, \mathbf {q}) = \sqrt {1 / \kappa} \cosh^ {- 1} (- \kappa \langle \mathbf {p}, \mathbf {q} \rangle_ {\mathbb {L}}).\tag{4}
$$

The hyperbolic radius of an embedding p corresponds to its hyperbolic distance from the hyperboloid origin o, measured by $d _ { \mathbb { L } } ( \mathbf { p } , \mathbf { o } )$

## 3.2 Tangent spaces

For each point $\mathbf { z } \in \mathbb { L } ^ { n }$ , the tangent space at z is defined as:

$$
T _ {\mathbf {z}} \mathbb {L} ^ {n} = \left\{\mathbf {v} \in \mathbb {R} ^ {n + 1}: \langle \mathbf {z}, \mathbf {v} \rangle_ {\mathbb {L}} = 0 \right\},\tag{5}
$$

which forms an n-dimensional Euclidean vector space consisting of all vectors orthogonal to z under the Lorentzian inner product. A tangent vector $\mathbf { v } \in T _ { \mathbf { z } } \mathbb { L } ^ { n }$ can be mapped back to the hyperboloid via the exponential map,

$$
\exp_ {\mathbf {z}} ^ {\kappa} (\mathbf {v}) = \cosh (\sqrt {\kappa} \| \mathbf {v} \| _ {\mathbb {L}}) \mathbf {z} + \frac {\sinh (\sqrt {\kappa} \| \mathbf {v} \| _ {\mathbb {L}})}{\sqrt {\kappa} \| \mathbf {v} \| _ {\mathbb {L}}} \mathbf {v}.\tag{6}
$$

In contrast, the logarithmic map transports $\mathbf { p } \in \mathbb { L } ^ { n }$ to the tangent space at z as:

$$
\log_ {\mathbf {z}} ^ {\kappa} (\mathbf {p}) = \frac {\cosh^ {- 1} (- \kappa \langle \mathbf {z} , \mathbf {p} \rangle_ {\mathbb {L}})}{\sqrt {(\kappa \langle \mathbf {z} , \mathbf {p} \rangle_ {\mathbb {L}}) ^ {2} - 1}} \operatorname{proj} _ {\mathbf {z}} (\mathbf {p}),\tag{7}
$$

where $\mathrm { p r o j } _ { \mathbf { z } } ( \mathbf { p } ) = \mathbf { p } + \kappa \langle \mathbf { z } , \mathbf { p } \rangle _ { \mathbb { L } } \mathbf { z }$ denotes the projection of p onto the tangent space $T _ { \mathbf { z } } \mathbb { L } ^ { n }$ . In our formulation, the reference point z is set to the hyperboloid origin $\mathbf { o } = [ \sqrt { 1 / \kappa } , \mathbf { 0 } ]$ ]. At this point, the tangent space $T _ { \mathbf { o } } \mathbb { L } ^ { n }$ reduces to an ndimensional Euclidean space, as tangent vectors have zero time components and are fully parameterized by their spatial coordinates, consistent with standard practice in hyperbolic representation learning [13, 54, 60].

## 3.3 Einstein midpoint

In hyperbolic space, the centroid of a set of points is computed via the Einstein midpoint. We first project hyperbolic points from the hyperboloid model $\mathbb { L } ^ { n }$ to the Klein model $\mathbb { K } ^ { n }$ , where a weighted average admits a closed-form expression. The Lorentz-Klein projection and its inverse are given by:

$$
\Pi_ {\mathbb {L} \to \mathbb {K}} (\mathbf {p}) = \frac {\mathbf {p} _ {\mathrm{space}}}{p _ {\mathrm{time}}} = \frac {\mathbf {p} _ {\mathrm{space}}}{\sqrt {\frac {1}{\kappa} + \| \mathbf {p} _ {\mathrm{space}} \| ^ {2}}}, \qquad \Pi_ {\mathbb {K} \to \mathbb {L}} (\mathbf {k}) = \frac {(1 , \mathbf {k})}{\sqrt {\kappa (1 - \| \mathbf {k} \| ^ {2})}},\tag{8}
$$

where $\Pi _ { \mathbb { L } \to \mathbb { K } }$ maps a point $\mathbf { p } \in \mathbb { L } ^ { n }$ to its Klein coordinate representation, and $\Pi _ { \mathbb { K }  \mathbb { L } }$ denotes the inverse projection that lifts a Klein point k back onto the Lorentz hyperboloid.

Given a set of points $\{ \mathbf { x } _ { j } \} _ { j = 1 } ^ { N } \subset \mathbb { L } ^ { n }$ , their Einstein midpoint x¯ is obtained by computing a Lorentz-factor-weighted mean in Klein coordinates as:

$$
\bar {\mathbf {x}} = \Pi_ {\mathbb {K} \to \mathbb {L}} \left(\frac {\sum_ {j = 1} ^ {N} \gamma_ {j}   \Pi_ {\mathbb {L} \to \mathbb {K}} (\mathbf {x} _ {j})}{\sum_ {j = 1} ^ {N} \gamma_ {j}}\right), \qquad \gamma_ {j} = \frac {1}{\sqrt {1 - \kappa   \| \Pi_ {\mathbb {L} \to \mathbb {K}} (\mathbf {x} _ {j}) \| ^ {2}}}.\tag{9}
$$

## 4 Method

In this section, we introduce our method HyFL-CLIP. We first describe the problem setting, and then present the key training objectives of our approach: short-text guided cross-manifold similarity distillation, hyperbolic geodesic contrastive learning, hierarchical entailment with Einstein midpoint aggregation, and an entropy regularizer for stabilizing hyperbolic embeddings.

## 4.1 Problem formulation

HyFL-CLIP fine-tunes a pre-trained CLIP model, which uses separate encoders for images and texts to produce aligned representations in a shared embedding space. We denote the model as $f ,$ consisting of image encoder $f _ { v }$ and a text encoder $f _ { t }$ . For a text-image pair $( I , T )$ , the visual encoder produces a set of embeddings $f _ { v } ( I ) \in \mathbb { R } ^ { ( K + 1 ) \setminus n }$ , consisting of a class token $\tilde { \mathbf { v } }$ that captures global image representation and a set of token-wise embeddings $\{ \tilde { \mathbf { v } } _ { k } \} _ { k = 1 } ^ { K }$ that encode local visual information. Similarly, the text encoder produces a set of embeddings $f _ { t } ( T ) \in \mathbb { R } ^ { ( L + 1 ) \times n }$ , consisting of a sentence-level token $\tilde { \mathbf { t } }$ that represents the overall semantic meaning of the text and token-wise embeddings $\{ \tilde { \mathbf { t } } _ { l } \} _ { l = 1 } ^ { L }$ that encode token-level linguistic information.

In the original CLIP model, the input text length is limited to $7 7$ tokens due to the use of learned absolute positional embedding. Before being fed into $f _ { t } ,$ text tokens are truncated to the first $7 7$ tokens. To enable CLIP models to handle longer contexts, prior works [4, 16, 52, 72, 75, 79] extend the positional embedding via interpolation to support longer input sequences $( e . g .$ , up to 248 tokens).

Our goal is to improve long-context understanding by explicitly modeling hierarchical semantic relationships within long descriptions in hyperbolic space, while preserving the well-established short text-image alignment learned by CLIP.

## 4.2 Short-text guided cross-manifold similarity distillation

In pre-trained CLIP, the similarity between short text–image pairs is well learned. Inspired by prior similarity-based distillation methods that transfer relational structure through pairwise similarity distributions [70, 74], we use this wellestablished geometry as a reference when transferring the learned embedding space of CLIP into hyperbolic space. Unlike prior methods that typically distill similarities within the same geometric space, we perform cross-manifold similarity distillation, which aligns the similarity distributions of the Euclidean teacher and the hyperbolic student. Formally, let $\tilde { \mathbf { v } } _ { i } \in \mathbb { R } ^ { n }$ denote the image embedding, and let $\tilde { \bf t } _ { i } ^ { s } , \tilde { \bf t } _ { i } ^ { l } \in \mathbb { R } ^ { n }$ denote the embeddings of the short and long texts corresponding to the same image, obtained from a pre-trained CLIP model that serves as the Euclidean teacher. Since CLIP is originally trained on short text–image pairs, we perform the distillation using the short-text embeddings to preserve the well-learned similarity geometry. We project these Euclidean embeddings into hyperbolic space via the exponential map at the origin: $\begin{array} { r } { \mathbf { v } _ { i } = \exp _ { \mathbf { o } } ^ { \kappa } ( \tilde { \mathbf { v } } _ { i } ) , \mathbf { t } _ { i } ^ { s } = } \end{array}$ $\begin{array} { r } { \exp _ { \mathbf { o } } ^ { \kappa } ( \tilde { \mathbf { t } } _ { i } ^ { s } ) , \mathbf { t } _ { i } ^ { l } = \exp _ { \mathbf { o } } ^ { \kappa } ( \tilde { \mathbf { t } } _ { i } ^ { l } ) } \end{array}$ ), where $\mathbf { v } _ { i } , \mathbf { t } _ { i } ^ { s } , \mathbf { t } _ { i } ^ { l } \in \mathbb { L } ^ { n }$

We define the Euclidean teacher similarity $S ^ { \mathrm { E } }$ as the cosine similarity between the short-text and image embeddings. In hyperbolic space, the student similarity $S ^ { \mathrm { H } }$ is defined as the negative geodesic distance in the Lorentz model $\left( \mathrm { E q . ~ ( 4 ) } \right)$ Formally,

$$
S ^ {\mathrm{E}} (\tilde {\mathbf {t}} _ {i} ^ {s}, \tilde {\mathbf {v}} _ {j}) = \frac {\langle \tilde {\mathbf {t}} _ {i} ^ {s} , \tilde {\mathbf {v}} _ {j} \rangle}{\| \tilde {\mathbf {t}} _ {i} ^ {s} \| \| \tilde {\mathbf {v}} _ {j} \|}, \qquad S ^ {\mathrm{H}} (\mathbf {t} _ {i} ^ {s}, \mathbf {v} _ {j}) = - d _ {\mathbb {L}} (\mathbf {t} _ {i} ^ {s}, \mathbf {v} _ {j}).\tag{10}
$$

We construct probability distributions over image candidates induced by the similarity scores. Given a similarity matrix $S$ and temperature τ , the distribution is defined as:

$$
P (S, \tau) _ {i j} = \frac {\exp (S _ {i j} / \tau)}{\sum_ {k} \exp (S _ {i k} / \tau)}.\tag{11}
$$

The Euclidean teacher distribution and hyperbolic student distribution are then obtained as $P ^ { \mathrm { E } } = P ( S ^ { \mathrm { E } } ( \tilde { \mathbf { t } } ^ { s } , \tilde { \mathbf { v } } ) , \tau _ { E } ) , P ^ { \mathrm { H } } = \mathbf { \widetilde { P } } ( S ^ { \mathrm { H } } ( \mathbf { t } ^ { s } , \mathbf { v } ) , \tau _ { H } )$ . The cross-manifold distillation loss is defined as the Kullback-Leibler divergence between the teacher and student distributions, given by:

$$
\mathcal {L} _ {\text { distill }} = \frac {1}{B} \sum_ {i = 1} ^ {B} \mathrm{KL} \left(P _ {i.} ^ {\mathrm{E}} \| P _ {i.} ^ {\mathrm{H}}\right),\tag{12}
$$

where $B$ denotes the batch size, i indexes the query samples in the mini-batch, and $\operatorname { K L } ( \cdot \| \cdot )$ denotes the Kullback-Leibler divergence.

## 4.3 Hyperbolic geodesic contrastive loss

After transferring Euclidean geometric relations through short text–image pairs, we further optimize the model using long text–image pairs through a geodesic contrastive objective on the Lorentz manifold $\mathbb { L } ^ { n }$ . Following the hyperbolic InfoNCE formulation of [13], we define the image-text contrastive loss ${ \mathcal { L } } _ { \mathrm { i t c } }$ in Lorentz space as follows:

$$
L _ {\mathrm{info}} (\mathbf {v}, \mathbf {t}; \tau_ {c}) = - \sum_ {i} \log \frac {\exp \left(- d _ {\mathbb {L}} (\mathbf {v} _ {i} , \mathbf {t} _ {i}) / \tau_ {c}\right)}{\sum_ {k \neq i} \exp \left(- d _ {\mathbb {L}} (\mathbf {v} _ {i} , \mathbf {t} _ {k}) / \tau_ {c}\right)},\tag{13}
$$

where $\left( \mathbf { v } _ { i } , \mathbf { t } _ { k } \right)$ denotes the matched text-image pair in the batch, while the remaining text embeddings $\{ \mathbf { t } _ { k } \} _ { k \neq i }$ serve as negative samples. The temperature parameter $\tau _ { c }$ controls the sharpness of the softmax distribution.

We employ a bidirectional contrastive objective for both short text–image and long text–image pairs, defined as:

$$
\mathcal {L} _ {v \leftrightarrow t} ^ {s} = \frac {1}{2} \left(\mathcal {L} _ {\mathrm{info}} (\mathbf {v}, \mathbf {t} ^ {s}; \tau_ {c}) + \mathcal {L} _ {\mathrm{info}} (\mathbf {t} ^ {s}, \mathbf {v}; \tau_ {c})\right),\tag{14}
$$

$$
\mathcal {L} _ {v \leftrightarrow t} ^ {\ell} = \frac {1}{2} \left(\mathcal {L} _ {\text { info }} (\mathbf {v}, \mathbf {t} ^ {\ell}; \tau_ {c}) + \mathcal {L} _ {\text { info }} (\mathbf {t} ^ {\ell}, \mathbf {v}; \tau_ {c})\right).\tag{15}
$$

![](images/95c299230e24558c562eb61ffc36a84b3224bfadcb16a70bb019cbf178854c75.jpg)  
Fig. 2: Overview of our HyFL-CLIP framework. Our $_ \mathrm { H y F L - C L I P }$ transfers Euclidean text–image alignment into hyperbolic space via a short-text guided crossmanifold similarity distillation. Hierarchical entailment with Einstein midpoint aggregation abstracts token-wise information within each modality and aligns it with a global representation. Hyperbolic geodesic contrastive loss aligns both long texts and their semantic components with the corresponding image, while an entropy regularizer stabilizes the embedding distribution.

The final hyperbolic geodesic contrastive objective is given by:

$$
\mathcal {L} _ {\mathrm{itc}} = \mathcal {L} _ {v \leftrightarrow t} ^ {\ell} + \lambda_ {1} \mathcal {L} _ {v \leftrightarrow t} ^ {s},\tag{16}
$$

where $\lambda _ { 1 }$ is a hyperparameter.

## 4.4 Hierarchical entailment via Einstein midpoint aggregation

Long-context understanding requires capturing a global representation as well as modeling the individual components and their semantic inclusion within the same modality. To capture such hierarchical structure, we construct a parent representation by aggregating token-wise features via a similarity-weighted Einstein midpoint. The weights are determined by their semantic similarity to the global representation from the opposite modality. We then enforce a hyperbolic entailment constraint between the aggregated representation and the corresponding global embedding.

For clarity, we first describe the formulation for the long-text $\mathbf { t } ^ { \ell } ;$ the same procedure is applied to the image modality v . Let $\{ \mathbf { t } _ { i , k } ^ { \ell } \} _ { k = 1 } ^ { K } \subset \mathbb { L } ^ { n }$ denote the long-text token embeddings of the i-th sample. To measure the semantic importance of each token-level feature, we compute attention weights $\alpha _ { i , k }$ as follows:

$$
\alpha_ {i, k} = \frac {\exp \left(- d _ {\mathbb {L}} (\mathbf {t} _ {i , k} ^ {\ell} , \mathbf {v} _ {i}) / \tau_ {\mathrm{ent}}\right)}{\sum_ {m} \exp \left(- d _ {\mathbb {L}} (\mathbf {t} _ {i , m} ^ {\ell} , \mathbf {v} _ {i}) / \tau_ {\mathrm{ent}}\right)}, \qquad \sum_ {k} \alpha_ {i, k} = 1.\tag{17}
$$

Then we abstract token-wise feature by calculating weighted Einstein midpoint, multiplying $\alpha _ { i , k }$ to the Lorentz factor in Eq. (9). The weighted Einstein midpoint

is given as:

$$
\bar {\mathbf {t}} _ {i} ^ {\ell} = \Pi_ {\mathbb {K} \to \mathbb {L}} (\frac {\sum_ {k} \alpha_ {i , k} \gamma_ {i , k} \Pi_ {\mathbb {L} \to \mathbb {K}} (\mathbf {t} _ {i , k} ^ {\ell})}{\sum_ {k} \alpha_ {i , k} \gamma_ {i , k}}), \qquad \bar {\mathbf {t}} _ {i} ^ {\ell} \in \mathbb {L} ^ {n}.\tag{18}
$$

We interpret the aggregated representation $\bar { \bf t } _ { i } ^ { \ell }$ as a more general semantic concept and enforce that the corresponding global embedding $\mathbf { t } _ { i } ^ { \ell } \in \mathbb { L } ^ { n }$ lies within its entailment cone. Following the hyperbolic entailment formulation in $[ 1 7 , 3 8 ]$ , the halfaperture of the cone centered at $\bar { \bf t } _ { i } ^ { \ell }$ is defined as:

$$
\omega (\bar {\mathbf {t}} _ {i} ^ {\ell}) = \arcsin \left(\frac {2 K}{\sqrt {\kappa} \| \bar {\mathbf {t}} _ {i} ^ {\ell} \| _ {\mathbb {L}}}\right),\tag{19}
$$

![](images/d928a4b7a047f6b5a8a9242f9f6e5fea2160accd70dea6bdf68f91078a7602f0.jpg)

where $K$ is a constant controlling stability near the origin. We enforce the entailment constraint by penalizing violations of the cone boundary using the following loss (see Fig. 3):

Entailment loss

$$
\mathcal {L} _ {\text { ent }} ^ {\mathbf {t} ^ {\ell}} = \frac {1}{B} \sum_ {i = 1} ^ {B} \max \left(0, \phi (\bar {\mathbf {t}} _ {i} ^ {\ell}, \mathbf {t} _ {i} ^ {\ell}) - \eta   \omega (\bar {\mathbf {t}} _ {i} ^ {\ell})\right),\tag{20}
$$

where $\phi ( \cdot , \cdot )$ denotes the hyperbolic angle between two embeddings in the Lorentz model. The above formulation is symmetrically applied to the image embeddings $\mathbf { v } ,$ yielding $\mathcal { L } _ { \mathrm { e n t } } ^ { \mathbf { v } }$ . The final hierarchical entailment loss is defined as:

Fig. 3: Entailment loss in hyperbolic space. Adapted from the MERU [13], this figure illustrates the entailment loss in hyperbolic space. The $\phi ( \mathbf { x } , \mathbf { y } )$ measures the geodesic angle between the two embeddings, and $\omega ( \mathbf { x } )$ denotes the aperture of the entailment cone centered at $\mathbf { x } .$ Geodesic angle between the two embeddings is used to determine if y lies within the entailment region of x.

$$
\mathcal {L} _ {\mathrm{ent}} = \mathcal {L} _ {\mathrm{ent}} ^ {\mathbf {t}} + \mathcal {L} _ {\mathrm{ent}} ^ {\mathbf {v}}.\tag{21}
$$

Thus, our final loss is given as:

$$
\mathcal {L} = \lambda_ {2} \mathcal {L} _ {\mathrm{distill}} + \mathcal {L} _ {\mathrm{itc}} + \lambda_ {3} \mathcal {L} _ {\mathrm{ent}} + \lambda_ {4} \mathcal {L} _ {\mathrm{reg}}.\tag{22}
$$

Inspired by entropy-based regularization strategies used in prior works [20, 34], we introduce a radius-entropy regularization term:

$$
\mathcal {L} _ {\mathrm{reg}} = - H (\mathbf {p}), \qquad p _ {i} = \frac {\exp \big (d _ {\mathbb {L}} (\mathbf {t} _ {i} ^ {\ell} , \mathbf {o}) \big)}{\sum_ {j} \exp \big (d _ {\mathbb {L}} (\mathbf {t} _ {j} ^ {\ell} , \mathbf {o}) \big)}.\tag{23}
$$

Here, $H ( \mathbf { p } ) = - \textstyle \sum _ { i } p _ { i }$ log p<sub>i</sub> denotes the entropy of the hyperbolic radius distribution. This regularization promotes balanced utilization of hyperbolic radii. More details on the hyperparameter choices are in the supplementary material.

Intuitively, the entailment cone provides a geometric margin of tolerance. The global text embedding is trained to lie within the cone of the aggregated midpoint, which summarizes token-level features. When a perturbation removes or reorders tokens, the midpoint shifts, but the cone’s nonzero aperture (Eq. 19) allows $\mathbf { t } _ { i } ^ { \ell }$ to remain within the entailment region, preserving alignment. By contrast, Euclidean contrastive objectives enforce point-to-point matching with no such margin, making them sensitive to any shift in the aggregated representation.

## 5 Experiments

## 5.1 Experimental setup

Training details. Following Long-CLIP [79], we train HyFL-CLIP on dataset ShareGPT4V [10], which contains 1.2 million image–caption pairs with multisentence annotations and long captions averaging 143.6 words. The batch size is 1024 and we train our models for 2 epochs. We set the learning rate to $1 \times 1 0 ^ { - 5 }$ and the weight decay to $2 . 5 \times 1 0 ^ { - 2 }$ , and optimize the model using AdamW. Our model builds upon Open-CLIP, and we experiment with two CLIP architectures, CLIP-ViT-B/16 and CLIP-ViT-L/14.

Evaluation setup. We evaluate our model against other baselines [4,16,52,72, 75, 76, 79] on four diferent tasks:

(1) Zero-shot long/short caption cross modality retrieval. We evaluate HyFL-CLIP on zero-shot long-caption text-image retrieval using DOCCI [53], DCI [71], Long-DCI [52], and Urban-1k [79], whose captions average 131.4 to 174.2 tokens. Performance was measured using Top-1 retrieval accuracy. To examine whether HyFL-CLIP successfully distills the text–image similarity geometry learned from short captions while adapting to long captions, we further evaluate short-text retrieval on COCO [42] and Flickr30K [56].

(2) Zero-shot long-caption cross-modal retrieval under caption perturbation. Using all datasets from Task (1), we perturb the captions to test whether models can robustly retrieve the correct images under semantic-preserving modifications. The perturbations include random word dropping $( p = 0 . 5 )$ , random subsampling of $n = 2 { \mathrm { ~ o r ~ } } n = 3$ sentences, sentence order shufling, and removal of the first sentence. Each model is evaluated five times per perturbation, and Top-1 retrieval accuracy is averaged across all datasets.

(3) Text-to-text intra modality retrieval. We perform text-to-text retrieval following the evaluation protocol of [51]. For COCO [42], Flickr30K [56], and nocaps [2] datasets, we ignore the images and use the first caption of each image as the query, while the remaining captions of the same image are treated as positives. We use the Karpathy split [32] for COCO and Flickr30K, and the validation split for nocaps. We further evaluate on purely textual datasets, including 20 Newsgroups [37] and IMDB Reviews [49].

(4) Text-to-image generation. To qualitatively evaluate how HyFL-CLIP can be integrated into diferent model frameworks, we apply it to Stable Difusion

XL [57] (SDXL). Since SDXL is originally trained with Euclidean text prompts, we encode text prompts from Long-DCI [52], DrawBench [62] and project them to the Euclidean tangent space using Eq. (7) before image generation. We compare the results with Long-CLIP [79] integrated with SDXL.

## 5.2 Experimental results

Zero-shot long-caption cross-modal retrieval. Tab. 1 presents long caption cross-modal retrieval results. HyFL-CLIP (Ours) consistently outperforms Euclidean baselines across datasets and architectures. Comparisons with stateof-the-art hyperbolic VLMs [13, 34, 54] are in the supplementary material.

Table 1: Comparison of zero-shot long-caption cross-modal retrieval. HyFL-CLIP (Ours) consistently outperforms existing long-context CLIP baselines across all datasets and model architectures. The best and second-best results are highlighted in bold and underline, respectively. We report numbers from the original papers when available; otherwise, we evaluate using our own implementation (marked with ∗). Results marked with † are obtained using the checkpoints provided by the original authors.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">DOCCI [53]</td><td colspan="2">DCI [71]</td><td colspan="2">Long-DCI [52]</td><td colspan="2">Urban-1k [79]</td></tr><tr><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td></tr><tr><td rowspan="8">ViT-B-16</td><td>Long-CLIP $^{\dagger}$  [79]</td><td>63.10</td><td>71.49</td><td>59.88</td><td>61.28</td><td>42.21</td><td>48.38</td><td>79.40</td><td>79.60</td></tr><tr><td>TULIP [52]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>50.20</td><td>50.60</td><td>88.10</td><td>86.60</td></tr><tr><td>HiMo-CLIP* [75]</td><td>77.37</td><td>79.35</td><td>71.09</td><td>69.93</td><td>58.59</td><td>57.00</td><td>89.20</td><td>89.20</td></tr><tr><td>FineLIP* [4]</td><td>77.16</td><td>79.14</td><td>69.38</td><td>68.03</td><td>57.18</td><td>55.22</td><td>89.30</td><td>86.90</td></tr><tr><td>LongD-CLIP [16]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>87.20</td><td>87.30</td></tr><tr><td>SmartCLIP [76]</td><td>77.40</td><td>78.00</td><td>64.90</td><td>64.00</td><td>53.40</td><td>52.80</td><td>90.00</td><td>87.40</td></tr><tr><td>Fix-CLIP [72]</td><td>-</td><td>-</td><td>59.70</td><td>63.00</td><td>-</td><td>-</td><td>80.90</td><td>81.10</td></tr><tr><td>HyFL-CLIP(Ours)</td><td>78.41</td><td>81.12</td><td>71.54</td><td>71.79</td><td>59.00</td><td>58.75</td><td>91.80</td><td>91.10</td></tr><tr><td rowspan="8">ViT-L-14</td><td>Long-CLIP $^{\dagger}$  [79]</td><td>66.78</td><td>78.61</td><td>64.13</td><td>67.83</td><td>46.55</td><td>54.25</td><td>82.40</td><td>86.20</td></tr><tr><td>TULIP [52]</td><td>77.90</td><td>79.10</td><td>-</td><td>-</td><td>55.70</td><td>56.40</td><td>90.10</td><td>91.10</td></tr><tr><td>HiMo-CLIP $^{\dagger}$  [75]</td><td>82.35</td><td>84.59</td><td>74.59</td><td>74.54</td><td>62.06</td><td>61.94</td><td>93.00</td><td>93.20</td></tr><tr><td>FineLIP [4]</td><td>82.20</td><td>83.10</td><td>-</td><td>-</td><td>60.80</td><td>60.70</td><td>93.20</td><td>93.00</td></tr><tr><td>LongD-CLIP [16]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>91.90</td><td>90.80</td></tr><tr><td>SmartCLIP [76]</td><td>81.60</td><td>82.50</td><td>68.20</td><td>69.80</td><td>57.60</td><td>58.50</td><td>93.30</td><td>90.10</td></tr><tr><td>Fix-CLIP [72]</td><td>-</td><td>-</td><td>65.10</td><td>66.70</td><td>-</td><td>-</td><td>86.80</td><td>87.70</td></tr><tr><td>HyFL-CLIP(Ours)</td><td>82.12</td><td>85.39</td><td>74.74</td><td>76.19</td><td>61.92</td><td>63.93</td><td>94.60</td><td>94.30</td></tr></table>

Zero-shot long-caption cross-modal retrieval under caption perturbation. Tab. 2 presents zero-shot long-caption cross-modal retrieval results under caption perturbations. The table reports the relative performance change (in %). HyFL-CLIP (Ours) consistently outperforms other baselines across all datasets and exhibits the smallest performance drop. This indicates that the hierarchical entailment loss enables the model to better capture part–whole semantic relationships and remain robust to perturbations that preserve the underlying semantic meaning. Full results for all datasets are in the supplementary material.

Table 2: Comparison of zero-shot long-caption cross-modal retrieval under caption perturbation. Under caption perturbations, other models exhibit large performance drops, while our method maintains strong performance across all types and degrees of perturbations. The small numbers shown beside each score indicate the percentage performance diferences (%) relative to the original score. Notation follows Tab. 1 (∗: our implementation; †: author-provided checkpoints).

<table><tr><td rowspan="2">Model</td><td>Word Dropout</td><td>Sent. Removal</td><td>Order Shuffling</td><td colspan="2">Random Subsampling</td></tr><tr><td>p=0.5</td><td>first</td><td>random</td><td>n=2</td><td>n=3</td></tr><tr><td>Long-CLIP†[79]</td><td>48.88 ↓35.81</td><td>55.41 ↓19.45</td><td>61.79 ↓3.46</td><td>26.50 ↓91.89</td><td>31.29 ↓79.89</td></tr><tr><td>HiMo-CLIP* [75]</td><td>58.75 ↓27.82</td><td>64.44 ↓17.42</td><td>72.94 ↓1.88</td><td>28.24 ↓83.58</td><td>34.84 ↓71.52</td></tr><tr><td>FineLIP* [4]</td><td>58.70 ↓26.59</td><td>64.18 ↓16.25</td><td>73.41 ↑1.17</td><td>27.20 ↓86.05</td><td>33.41 ↓74.32</td></tr><tr><td>HyFL-CLIP (Ours)</td><td>70.20 ↓9.21</td><td>68.22 ↓12.69</td><td>76.16 ↑1.27</td><td>32.55 ↓75.36</td><td>39.05 ↓63.94</td></tr></table>

Zero-shot short caption crossmodal retrieval. Tab. 3 presents the results of zero-shot shortcaption cross-modal retrieval on COCO and Flickr30k [42, 56] datasets. These results indicate that HyFL-CLIP successfully distills the pre-trained text-image similarity geometry while being optimized in hyperbolic space, without sacrificing its base long-caption retrieval performance.

Table 3: Comparison of zero-shot short-caption cross-modal retrieval. HyFL-CLIP achieves comparable or better short-caption retrieval performance than Euclidean baselines. Notation follows Tab. 1 (∗: our implementation; †: authorprovided checkpoints).

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">COCO [42]</td><td colspan="2">Flickr30k [56]</td></tr><tr><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td></tr><tr><td rowspan="5">ViT-B/16</td><td>Long-CLIP $^{\dagger}$  [79]</td><td>57.26</td><td>40.38</td><td>47.19</td><td>33.16</td></tr><tr><td>TULIP [52]</td><td>56.80</td><td>40.70</td><td>46.10</td><td>35.20</td></tr><tr><td>HiMo-CLIP* [75]</td><td>60.84</td><td>40.71</td><td>50.27</td><td>34.02</td></tr><tr><td>FineLIP* [4]</td><td>58.32</td><td>40.05</td><td>52.23</td><td>33.85</td></tr><tr><td>HyFL-CLIP (Ours)</td><td>58.20</td><td>41.50</td><td>50.60</td><td>35.20</td></tr></table>

Table 4: Comparison of text-to-text intra modality retrieval. Intra-modal retrieval performance across five datasets (Values in %). Bold indicates the best, and underlined indicates the second-best. Notation follows Tab. 1 (∗: our implementation; †: author-provided checkpoints).

<table><tr><td rowspan="2">Model</td><td colspan="2">Flickr30K [56]</td><td colspan="2">COCO [42]</td><td colspan="2">nocaps [2]</td><td colspan="2">IMDB [49]</td><td colspan="2">20News [37]</td></tr><tr><td>mAP</td><td>Pr@R</td><td>mAP</td><td>Pr@R</td><td>mAP</td><td>Pr@R</td><td>mAP</td><td>Pr@R</td><td>mAP</td><td>Pr@R</td></tr><tr><td>Long-CLIP $^{\dagger}$  [79]</td><td>52.31</td><td>47.56</td><td>27.51</td><td>24.58</td><td>37.16</td><td>35.47</td><td>51.88</td><td>50.41</td><td>26.78</td><td>30.40</td></tr><tr><td>HiMo-CLIP* [75]</td><td>58.02</td><td>52.66</td><td>29.36</td><td>26.03</td><td>40.24</td><td>38.08</td><td>52.65</td><td>50.59</td><td>35.49</td><td>37.61</td></tr><tr><td>FineLIP* [4]</td><td>52.55</td><td>47.78</td><td>24.87</td><td>22.26</td><td>34.10</td><td>33.15</td><td>51.78</td><td>50.40</td><td>17.65</td><td>21.24</td></tr><tr><td>HyFL-CLIP(Ours)</td><td>60.63</td><td>54.98</td><td>30.69</td><td>27.12</td><td>41.16</td><td>38.56</td><td>52.60</td><td>50.59</td><td>36.69</td><td>38.32</td></tr></table>

![](images/28086b69b46d62d7e493623e7e905d1ad87a2eb473d24795ca57f8f1d43b482e.jpg)  
Fig. 4: Embeddings and token weight visualization. We visualize the embedding distribution of text summary token (Einstein midpoint), text, and image using HoroPCA [8]. We also compare text token contribution weights based on their similarity to the image.

![](images/dcff89bc26a2c9346987a662197c6292bdc403de981a659a5fbba438e5529fc8.jpg)  
Fig. 5: SDXL text-to-image generation results. We replace the original text encoder with ours. Hyperbolic embeddings are mapped to the Euclidean space using the logarithmic map transport. Our model captures finer details compared to the baseline.

Text-to-text intra modality retrieval. Tab. 4 presents the results of text-totext intra-modality retrieval across five datasets. HyFL-CLIP (Ours) consistently achieves superior performance, demonstrating its strong capability in capturing semantic relationships within textual representations. By leveraging hierarchical entailment within the same modality, HyFL-CLIP better captures semantic abstraction between local textual components and global representations, leading to improved retrieval performance across diverse datasets.

Text-to-image generation. Fig. 5 presents qualitative text-to-image generation results using SDXL [57] conditioned on both short captions from Draw-Bench [62] and long captions from Long-DCI [52]. As shown in Fig. 5, our model captures finer-grained details, such as the number of clocks and subtle attributes of the panda, compared to the baseline model. When long prompts are used, Long-CLIP [79] often ignores or corrupts parts of the description, whereas our model better preserves the full semantic context and generates more faithful results. These results indicate that our framework successfully distills the Euclidean similarity structure to produce more expressive hyperbolic embeddings.

## 5.3 Ablation studies

We conduct an ablation study in ViT-L/14 to evaluate the contribution of each component in our framework. First, removing ${ \mathcal { L } } _ { \mathrm { e n t } }$ reduces retrieval accuracy from 69.8% to 69.3%, performance averaged across all benchmarks [42,52,53,56, 71,79]. Next, we further remove ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ , which results in 68.3% performance. Full results are in the supplementary material. Fig. 4 shows embedding visualizations of our final model and a comparison of token contribution weights. HyFL-CLIP (Ours) assigns higher weights to semantically meaningful tokens such as street, bicycles, and city, which directly correspond to the key visual elements in the image.

## 6 Conclusion

We propose HyFL-CLIP, a framework that enables robust long-context understanding in CLIP by distilling the well-established text–image similarity geometry of a pre-trained Euclidean CLIP model into hyperbolic space. By leveraging hyperbolic geometry, HyFL-CLIP addresses the limitations of Euclidean contrastive learning by modeling hierarchical and part–whole relationships between global captions and their constituent elements. This design encourages the model to capture semantic relationships between whole captions and their constituent parts, leading to more robust representations under caption perturbations. Extensive experiments on long- and short-caption cross-modal retrieval as well as text-to-text intra-modality retrieval demonstrate state-of-the-art performance, highlighting the efectiveness of hyperbolic hierarchical representations for longcontext understanding.

## Acknowledgements

This work was supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grants funded by the Korea government(MSIT) [No.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University) / No.RS-2025-02314125, Efective Human-Machine Teaming With Multimodal Hazy Oracle Models], the National Research Foundation of Korea(NRF) grants funded by the Korea government(MSIT) (Nos. RS-2022-NR067592, RS-2025-02263628), the AI Computing Infrastructure Enhancement (GPU Rental Support) User Support Program funded by the Ministry of Science and ICT (MSIT), Republic of Korea (No. RQT-25-120066), the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University, AI-Bio Research Grant through Seoul National University and the AI Seoul Tech Research Support Program of the Seoul Future Foundation.

# Supplementary Material for HyFL-CLIP: Hyperbolic Fine-Tuning of CLIP for Robust Long-Context Understanding

Ji Ha Jang<sup>1,∗</sup> , Hayeon Kim<sup>1,∗</sup> , Chulwon Lee<sup>2</sup> , Junghun James Kim<sup>2</sup> , Se Young Chun<sup>1,2,3,†</sup>

<sup>1</sup>Dept. of Electrical and Computer Engineering, <sup>2</sup>IPAI, <sup>3</sup>INMC & AIIS, Seoul National University, Republic of Korea {jeeit17, khy5630, chul0e, jonghean12, sychun}@snu.ac.kr

## S1 Additional Experimental Details

## S1.1 Zero-shot long/short caption cross modality retrieval

Datasets. We use four datasets for zero-shot long-caption retrieval following prior works [4, 16, 52, 72, 75, 79]. Descriptions of Connected and Contrasting Images (DOCCI) [53] contains approximately 15k images paired with long, human-written English descriptions, with captions averaging around 141.5 tokens. The annotations emphasize fine-grained visual details, including spatial relationships between objects, counting, and text appearing in the scene. Urban-1k, introduced in [79], extends the earlier Urban-200 dataset proposed in the same work. The dataset contains 1k images with long captions averaging 131.4 tokens. It is constructed by selecting visually similar images from the Visual Genome dataset [36], after which long descriptive captions are generated using GPT-4V [1] to provide detailed scene descriptions. Densely Captioned Images (DCI) [71] consists of 7,805 natural images annotated with dense, human-written descriptions aligned with segmentation masks. The captions are highly detailed, averaging approximately 174.2 tokens per image, and are designed to support fine-grained vision–language understanding. Long-DCI [52] is derived from DCI and contains about 7k images paired with long human-annotated captions, with an average length of 200 tokens.

For short-caption retrieval, we follow prior work [4,16,52,72,75,79] and evaluate on the COCO2017 5k validation split [42] and the full Flickr30k dataset [56], where the average caption lengths are approximately 13.5 and 15.8 tokens, respectively.

Evaluation protocol. We evaluate cross-modal retrieval using top-1 accuracy. All methods are evaluated using the ViT-B/16 backbone for fair comparison. Given a dataset of paired images and captions, we first encode all captions and images using the text and image encoders of the model to obtain their corresponding embeddings. Text inputs longer than 248 tokens are truncated during tokenization. The resulting features are then mapped to the hyperbolic space, where similarity between image and text embeddings is computed using the Lorentzian inner product with the learned curvature parameter. Unlike CLIP [59], we do not apply feature normalization, and similarities are computed directly in the hyperbolic space. For both image-to-text and text-to-image retrieval, each query is compared with all candidates from the other modality, and the item with the highest similarity score is retrieved. A prediction is counted as correct if the retrieved item matches the ground-truth pair.

For CLIP-based baselines [4, 16, 52, 72, 75, 79], image and text embeddings are $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ and similarities are computed using cosine similarity instead of the Lorentzian inner product.

## S1.2 Zero-shot long-caption cross-modal retrieval under caption perturbation

Caption perturbation. To evaluate robustness to caption perturbations, we generate modified captions using four types of perturbations. Word dropout randomly removes a fraction of words from the caption. In our experiments, we use a dropout probability of $p = 0 . 5$ , meaning that approximately half of the words are randomly removed. Sentence removal deletes the first sentence of the caption, which often contains a high-level summary of the scene, to simulate the absence of this summary-level information. Sentence order shufling randomly permutes the order of sentences within the caption while keeping the sentence contents unchanged. Random subsampling constructs a shortened caption by randomly selecting a subset of sentences from the original caption. We use $n = 2$ and $n = 3$ sentences in our experiments.

Evaluation protocol. The retrieval evaluation follows the protocol described in Sec S1.1. For each perturbation setting, the experiments are repeated five times for all datasets to account for randomness in the perturbation process, and the reported results are averaged across runs. The drop rates reported in the main paper indicate the relative performance decrease with respect to each model’s original retrieval performance without perturbations.

## S1.3 Text-to-text intra modality retrieval

Evaluation protocol. To evaluate whether the learned text embeddings capture semantic consistency between full captions and their constituent parts, we additionally perform text-to-text intra-modality retrieval following [51]. In this setting, captions are divided into a query set and a gallery set, where the gallery serves as the retrieval database. All captions are encoded using the text encoder to obtain embeddings. For our model, similarities between embeddings are computed using the Lorentzian inner product, while Euclidean baselines [4, 75, 79] compute similarities using cosine similarity. For each query caption, similarities are computed against all captions in the gallery and the captions are ranked according to their similarity scores.

Retrieval performance is evaluated using mean Average Precision (mAP) and Precision@R (Pr@R). The mAP measures the average precision over the ranked retrieval results, reflecting the overall ranking quality. Precision@R computes the precision at rank R, where R denotes the number of relevant captions for each query and therefore varies across queries. Both metrics are computed per query and averaged across all queries. All methods are evaluated using the ViT-B/16 backbone.

## S1.4 Text-to-image generation

Our Stable Difusion XL (SDXL) [57] implementation preserves the original twostream text-conditioning interface, where token-level embeddings are formed by concatenating an OpenCLIP-L/14 branch and an OpenCLIP-bigG branch, with pooled embeddings retained from the OpenCLIP-bigG branch. Token-level text features from HyFL-CLIP (Ours) and LongCLIP [79] are injected only into the compatible channels of the OpenCLIP-L/14 branch via α-interpolation, while the remaining channels and the second branch are kept unchanged. This branch-local design isolates the efect of the injected long-context text representations while keeping the rest of the generation pipeline fixed, following prior controlled studies on text representations in text-to-image generation [41]. For fair comparison, all methods are evaluated under the same protocol with a ViT-B/16 backbone. Implementation results with a ViT- $\cdot \mathrm { L } /$ 14 backbone are also reported in Sec. S5.3.

## S2 Additional Pipeline Details

## S2.1 Model architecture

We initialize our model from the pretrained weights of OpenCLIP [28]. The text encoder follows the CLIP [59] architecture and consists of 12-layer Transformer [?] with a hidden dimension of 512. The maximum input length is set to 248 tokens. Following prior works [4,16,52,72,75,79], we preserve the positional embeddings of the first 20 tokens and apply interpolation only to the remaining positions, allowing longer input sequences while maintaining the well-trained positional structure of CLIP.

$$
P E ^ {*} (p o s) = \left\{ \begin{array}{l l} P E (p o s), & p o s \leq 2 0 \\ (1 - \alpha)   P E \big (\lfloor \frac {p o s}{r} \rfloor \big) + \alpha   P E \big (\lceil \frac {p o s}{r} \rceil \big), & \text {otherwise} \end{array} \right.\tag{1}
$$

where $P E ( p o s )$ denotes the original positional embedding at position pos, and

$$
\alpha = \frac {p o s \bmod r}{r}.\tag{2}
$$

Here, $\alpha \in [ 0 , 1 ]$ controls the interpolation weight between the two neighboring positions, and r = 4 in our case. This results in an extended positional embedding length of 248 tokens from the original 77 tokens.

For images, we adopt a Vision Transformer [15] and experiment with two capacity configurations, ViT-B/16 and $\mathrm { V i T - L / 1 4 }$ , using a patch size of 16 and 14, respectively.

## S2.2 Model initialization and hyperparameter setting

We parameterize the curvature in the Lorentz space and treat it as a learnable parameter. The curvature is initialized with κ = 1.0 and converges to 0.9994 after training. Following prior works [13, 54, 60], we apply learnable scaling factors to image and text vectors, setting $\begin{array} { r } { c _ { \mathrm { i m g } } = c _ { \mathrm { t x t } } = \frac { 1 } { \sqrt { 5 1 2 } } } \end{array}$ for numerical stability.

For short-text guided cross-manifold similarity distillation, we set $\tau _ { E } = \tau _ { H } =$ 0.005 in Eq. (12). In the hyperbolic geodesic contrastive loss (Eq. (14) and Eq. (15)), the temperature parameter $\tau _ { c }$ is set to 0.07. For hierarchical entailment via Einstein midpoint aggregation, we follow the standard hyperbolic entailment cone formulation and set $K = 1$ in Eq. (19). The scaling parameter $\eta$ is set to 1.2 in Eq. (20). Finally, we set $\lambda _ { 1 } = 0 . 1$ in Eq. (16), and $\lambda _ { 2 } = 0 . 0 5 , \lambda _ { 3 } = 0 . 1$ $\lambda _ { 4 } = 0 . 1$ in Eq. (22). A sensitivity analysis of the hyperparameters is provided in Sec. S3.2.

## S2.3 Training details

Optimizer. Our model is trained for 2 epochs using four A100 GPUs with a global batch size of 1024. We employ the AdamW optimizer [46], setting $\beta _ { 1 } =$ $0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and a weight decay of $2 . 5 \times 1 0 ^ { - 2 }$ . We adopt a cosine learningrate scheduler [45] with a learning rate of $1 0 ^ { - 5 }$ , with a 200-step linear warm-up period.

Computational overhead. We compare the computational overhead of $\mathrm { H y F L - }$ CLIP (Ours) with several baselines [4,52,75,79]. All comparisons are conducted using the ViT-B/16 backbone. Tab. S1 reports the total computational cost in terms of FLOPs across the full training schedule. Overall, our model requires less computational overhead than the compared methods.

Tulip [52] employs a distillation-based training strategy. Specifically, it performs distillation from a teacher OpenCLIP model for 20 epochs followed by 1 epoch of fine-tuning, which we denote as $^ { 6 } 2 0 { + } 1 ^ { \prime }$ epochs in the table. Following the convention used in prior work, the computational cost of the teacher model during distillation is not included in the reported FLOPs.

Table S1: Comparison of total computational cost (FLOPs) across diferent methods. We report the total computational cost of HyFL-CLIP (Ours) and several baseline methods. Overall, our model requires less computational overhead than the compared methods.

<table><tr><td>Model</td><td>Epochs</td><td>Total compute (FLOPs)</td></tr><tr><td>Long-CLIP [79]</td><td>1</td><td> $2.66 \times 10^{17}$ </td></tr><tr><td>HiMo-CLIP [75]</td><td>10</td><td> $1.96 \times 10^{18}$ </td></tr><tr><td>FineLIP [4]</td><td>6</td><td> $1.19 \times 10^{18}$ </td></tr><tr><td>Tulip [52]</td><td>20 (+1)</td><td> $1.58 \times 10^{18}$ </td></tr><tr><td>HyFL-CLIP (Ours)</td><td>2</td><td> $5.34 \times 10^{17}$ </td></tr></table>

![](images/8d73fb2040b3f8bb475ac48b62018e5bf3935c720074cfbe42c266854f4bbe93.jpg)  
Full

![](images/f67a182de32362e546181e5169528b37964a6bb3f7367f9bb68dfa8afe1dbc56.jpg)  
Full - ?<sub>!"#</sub>

![](images/3d1d9668f93b93a6fa549e1b0b54b2e6a5656086937be2d07c7b4311a6519ee8.jpg)  
Full - ?<sub>!"#</sub>- ?<sub>\$%&#%''</sub>  
Fig. S1: Qualitative results of the full ablation study. We visualize the embedding distributions of images (Image), short texts (Short Text), corresponding long texts (Long Text), and Einstein midpoints of long texts (Summary Token) using HoroPCA [8] for the full model and ablated variants without $\mathcal { L } _ { e n t }$ and $\mathcal { L } _ { d i s t i l l }$ in ShareGPT4V dataset [10].

Table S2: Quantitative results of the full ablation study. We analyze the contribution of each objective by removing $\mathcal { L } _ { e n t }$ and $\mathcal { L } _ { d i s t i l l }$ from the full model. All ablations are conducted using the $\mathrm { V i T - L } / 1 4$ backbone. The experiments are performed across multiple datasets.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Full -  $\mathcal{L}_{ent}$  -  $\mathcal{L}_{distill}$ </td><td colspan="2">Full -  $\mathcal{L}_{ent}$ </td><td colspan="2">Full</td></tr><tr><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td></tr><tr><td>Urban-1k [79]</td><td>93.80</td><td>93.70</td><td>95.00</td><td>94.80</td><td>94.60</td><td>94.30</td></tr><tr><td>DOCCI [53]</td><td>82.35</td><td>82.96</td><td>82.18</td><td>85.41</td><td>82.12</td><td>85.39</td></tr><tr><td>DCI [71]</td><td>74.64</td><td>73.99</td><td>73.69</td><td>75.89</td><td>74.74</td><td>76.19</td></tr><tr><td>Long-DCI [52]</td><td>62.00</td><td>61.42</td><td>61.64</td><td>63.55</td><td>61.92</td><td>63.93</td></tr><tr><td>COCO [42]</td><td>42.77</td><td>61.78</td><td>41.56</td><td>61.06</td><td>45.56</td><td>61.56</td></tr><tr><td>Flickr30k [56]</td><td>35.78</td><td>54.92</td><td>40.80</td><td>55.70</td><td>41.18</td><td>56.32</td></tr></table>

## S3 Additional Ablation Results

## S3.1 Full ablation results

Tab. S2 and Fig. S1 present the full ablation results, highlighting the complementary efects of the two objectives, $\mathcal { L } _ { e n t }$ and $\mathcal { L } _ { d i s t i l l }$

Removing $\mathcal { L } _ { e n t }$ slightly degrades retrieval performance on long-context datasets. As shown in Fig. S1, the alignment between long-text embeddings and their corresponding summary tokens becomes weaker, resulting in a more dispersed distribution of long-text representations. This suggests that $\mathcal { L } _ { e n t }$ may help align long texts with their summary tokens and improve the stability of the longcontext representation space.

In contrast, removing $\mathcal { L } _ { d i s t i l l }$ results in a noticeable drop in retrieval performance, particularly on datasets with short captions. The corresponding visualization shows weakened alignment between image and text embeddings, where the image cluster drifts away from the text clusters. This suggests that $\mathcal { L } _ { d i s t i l l }$ helps preserve the cross-modal semantic structure inherited from the pretrained CLIP representation.

Table S3: Hyperparameter sensitivity analysis. We vary each hyperparameter $\left( \lambda _ { 1 }  – \lambda _ { 4 } \right)$ over a range of values while keeping the others fixed and report the resulting performance. The results show that the performance remains stable across a broad range of hyperparameter values, indicating that our method is not sensitive to precise hyperparameter tuning.

<table><tr><td rowspan="2"> $\lambda_1$ </td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td></tr><tr><td>74.58/75.35</td><td>74.63/75.95</td><td>75.18/75.70</td><td>75.15/75.93</td><td>75.35/75.75</td></tr><tr><td rowspan="2"> $\lambda_2$ </td><td>0.01</td><td>0.03</td><td>0.05</td><td>0.07</td><td>0.09</td></tr><tr><td>75.13/75.35</td><td>75.08/75.83</td><td>75.18/75.70</td><td>75.10/75.95</td><td>75.03/76.05</td></tr><tr><td rowspan="2"> $\lambda_3$ </td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td></tr><tr><td>75.05/75.83</td><td>75.33/76.03</td><td>75.18/75.70</td><td>75.13/75.65</td><td>75.18/75.88</td></tr><tr><td rowspan="2"> $\lambda_4$ </td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td></tr><tr><td>75.10/75.88</td><td>75.15/75.85</td><td>75.18/75.70</td><td>75.23/75.65</td><td>75.03/75.85</td></tr></table>

## S3.2 Hyperparameter sensitivity test

We analyze the sensitivity of the hyperparameters $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ , and $\lambda _ { 4 }$ . For each experiment, one hyperparameter is varied while the remaining ones are fixed to their default values. Specifically, when evaluating $\lambda _ { 1 }$ , we fix $\lambda _ { 2 } = 0 . 0 5 , \lambda _ { 3 } = 0 . 1$ ， and $\lambda _ { 4 } = 0 . 1$ . Similarly, $\lambda _ { 2 } , \lambda _ { 3 }$ , and $\lambda _ { 4 }$ are analyzed by varying each parameter individually while keeping the others fixed. Each entry of Tab. S3 reports the corresponding I2T / T2I retrieval performance. The results show that the performance remains stable across diferent hyperparameter settings, indicating that our method is not sensitive to precise hyperparameter tuning.

## S4 Analysis on robustness to caption perturbations

## S4.1 Embedding analysis under caption perturbations in Hyperbolic and Euclidean spaces

Embedding visualization with perturbed captions. We visualize the embedding distributions to examine the relationship between the original text–image pairs and their perturbed texts in both Euclidean and hyperbolic spaces. Specifically, we employ t-SNE [50] for HiMo-CLIP [75], which operates in Euclidean space, and HoroPCA [8] for HyFL-CLIP (Ours), whose embeddings lie in hyperbolic space.

As shown in Fig. S2, in Euclidean space the embeddings of perturbed texts (red stars) are separated from the original target text (yellow star), indicating that even subtle variations in the caption can lead to large shifts in the embedding space under the Euclidean distance metric. In contrast, in hyperbolic space the embeddings of perturbed texts remain close to the original target text embedding. This behavior highlights the property of the hyperbolic distance metric, which better preserves semantic proximity under caption perturbations and leads to more stable alignment between images and their corresponding textual descriptions.

Rank distribution of image-text similarity under caption perturbations. We conduct an analysis to evaluate how accurately the original target image can be retrieved given various perturbed captions. Specifically, each perturbed caption is mixed with all other captions, and the captions are then ranked according to their similarity to the original target image. In Euclidean space, embeddings are typically normalized to unit length, allowing the Euclidean distance to be measured through cosine similarity. In contrast, hyperbolic space enables the use of both the entailment relationship between embeddings and the hyperbolic distance. With HiMo-CLIP [75], image–text similarity is ranked using Euclidean distance, as the model operates in Euclidean space. In contrast, our model ranks image–text similarity using both hyperbolic distance and the entailment relationship.

Embedding visualization with perturbed captions

![](images/8c60682648088b76df4134f865edb823b6f225284ac5660ac45b932c88182efb.jpg)  
Fig. S2: Embedding visualization with perturbed captions. We visualize the target image and text embeddings together with captions generated by perturbing the target text embedding. Under Euclidean similarity (HiMo-CLIP [75]), perturbed caption embeddings are widely scattered in the embedding space. In contrast, our hyperbolic representation keeps perturbed captions tightly clustered around the target text embedding, indicating more stable semantic alignment with the target image.

## Rank distribution of image-text similarity with perturbed caption

![](images/a9f61f8a4d30d401c4681b709ece4af0aecc28777ea528b2e6ae31a5efc3bf23.jpg)  
Fig. S3: Rank distribution of image–text similarity under caption perturbations. Captions are ranked by similarity to the target image. Green bars show similarities with all captions, while red markers indicate captions perturbed from the reference caption. Euclidean similarity (HiMo-CLIP [75]) scatters perturbed captions across the ranking, whereas hyperbolic entailment clusters them near the top ranks.

Table S4: Comparison of ranking statistics. We report the mean rank and rank variance of perturbed captions in the caption similarity ranking. Lower mean rank values indicate that perturbed captions are retrieved closer to the top positions, while lower rank variance indicates more stable and consistent retrieval behavior. Compared to Euclidean distance, hyperbolic distance improves both the accuracy and stability of retrieving semantically similar perturbed captions, and hyperbolic entailment exposes this relationship more clearly.

<table><tr><td>Model</td><td>Mean Rank</td><td>Rank Variance</td></tr><tr><td>HiMo-CLIP [75] (Euclidean distance)</td><td>184.94</td><td>42250.26</td></tr><tr><td>Ours (Hyperbolic distance)</td><td>156.99</td><td>31007.93</td></tr><tr><td>Ours (Hyperbolic entailment)</td><td>78.50</td><td>12666.22</td></tr></table>

Fig. S3 and Tab. S4 present the ranking results. Consistent with the observations in Fig. S2, ranking based on Euclidean distance scatters perturbed captions widely across the rank distribution, even though they retain similar semantics to the original captions. In contrast, our hyperbolic distance produces a distribution in which perturbed captions are concentrated near the top ranks.

This trend is also reflected in the ranking statistics in Tab. S4. Here, the mean rank and rank variance are computed over the ranks of perturbed captions with respect to the target image. Compared to Euclidean distance, our hyperbolic distance yields a lower mean rank and lower rank variance, indicating that semantically similar perturbed captions are retrieved more consistently and appear closer to the top of the ranking.

This tendency becomes even more pronounced when using the hyperbolic entailment relationship. As shown in both Fig. S3 and Tab. S4, when ranking is based on the entailment relationship between the original caption and its perturbed captions, the perturbed captions are concentrated almost entirely at the top of the ranking. Correspondingly, Tab. S4 shows that the mean rank and rank variance of the perturbed captions are substantially reduced.

These two experiments demonstrate that hyperbolic space is more robust than Euclidean distance in preserving semantic relationships under perturbations, preventing semantically related representations from becoming discretely separated or drifting far from the original representation. Moreover, this tendency becomes more pronounced when using the entailment relationship, which is naturally defined in hyperbolic space.

Table S5: Performance degradation (%) under caption perturbations, measured relative to each model’s original (non-perturbed) performance. This comparison evaluates the influence of each training objective by analyzing how the robustness changes when individual losses are removed.

<table><tr><td>Dataset</td><td>w/o  $\mathcal{L}_{\text{distill}}$ </td><td>w/o  $\mathcal{L}_{\text{ent}}$ </td><td>Ours full</td></tr><tr><td>Urban-1k [79]</td><td>32.99</td><td>36.85</td><td>33.83</td></tr><tr><td>DOCCI [53]</td><td>29.57</td><td>33.85</td><td>30.48</td></tr><tr><td>DCI [71]</td><td>27.26</td><td>32.82</td><td>29.89</td></tr><tr><td>DCI-Long [52]</td><td>39.31</td><td>39.04</td><td>35.31</td></tr><tr><td>Average</td><td>32.28</td><td>35.64</td><td>32.38</td></tr></table>

## S4.2 Impact of training objectives on caption perturbation robustness

To further investigate the observations from Sec. S4.1 and identify which loss term contributes to robustness against caption perturbations, we conduct an ablation study by removing each loss term $( \mathcal { L } _ { e n t }$ and $\mathcal { L } _ { d i s t i l l } )$ individually. We then measure the performance degradation under caption perturbations. The types and number of perturbations follow the same evaluation protocol used in the main experiments (Sec. S1.2). For each model variant, the performance degradation is measured relative to its own original performance rather than the full model, and the results are averaged over all perturbation settings and datasets.

Tab. S5 shows that removing $\mathcal { L } _ { e n t }$ results in a substantially larger performance drop under caption perturbations compared to removing $\mathcal { L } _ { d i s t i l l }$ . This indicates that $\mathcal { L } _ { e n t }$ plays a more important role in improving robustness to caption perturbations. This observation is consistent with the findings in Sec. S4.1, where the entailment relationship in hyperbolic space relaxes the strict one-toone correspondence between image–text pairs and allows semantically related concepts to remain close in the embedding space, thereby improving robustness to caption perturbations.

Fig. S5, Fig. S6, and Fig. S7 show the visualization of text-token contribution weights when the model is trained with and without $\mathcal { L } _ { e n t }$ . When trained with $\mathcal { L } _ { e n t : }$ , the entailment relationship encourages the Einstein midpoint of token embeddings to align with the global token representation of the long text. Consequently, tokens corresponding to diferent visual regions or attributes of the image are more clearly highlighted.

Table S6: BLIP-VQA robustness under caption perturbations. We measure image–text consistency using BLIP-VQA on images retrieved from perturbed captions. Compared with HiMo-CLIP, our method maintains higher consistency with the captions across multiple perturbation strategies.

<table><tr><td rowspan="2">Model</td><td>Word Dropout</td><td colspan="2">Random Subsampling</td><td>Order Shuffling</td><td>Sent. Removal</td></tr><tr><td>p=0.5</td><td>n=2</td><td>n=3</td><td>random</td><td>first</td></tr><tr><td>GT</td><td>0.4725</td><td>0.4869</td><td>0.4809</td><td>0.4581</td><td>0.4395</td></tr><tr><td>HyFL-CLIP (Ours)</td><td>0.4112</td><td>0.4283</td><td>0.4160</td><td>0.4372</td><td>0.3997</td></tr><tr><td>HiMo-CLIP [75]</td><td>0.3864</td><td>0.3982</td><td>0.3917</td><td>0.4321</td><td>0.3795</td></tr></table>

## S4.3 VQA-based validation of retrieval results under caption perturbations

To evaluate whether perturbed captions remain semantically valid with respect to their paired images, we measure image–text consistency using an LLM-based VQA framework (Tab. S6). After applying caption perturbations, we use the perturbed text as input to the BLIP-VQA [25] to generate five questions derived from the caption. These questions are then used to evaluate the validity of three candidate images: the ground-truth (GT) image paired with the original caption, the top-1 retrieved image from our method, and the top-1 retrieved image from HiMo-CLIP. For each candidate image, we measure the True/False correctness of the answers produced by BLIP-VQA. We conduct the evaluation on the Urban-1k dataset [79]. For each type of caption perturbation, the experiment is repeated five times and the results are averaged. The quantitative results are summarized in Tab. S6. The results show that the original GT image maintains the highest alignment with the perturbed captions. Importantly, the images retrieved by our method also remain highly consistent with the perturbed captions, demonstrating that our model retrieves images that are still semantically valid even under caption perturbations.

## S4.4 Robustness under LLM-generated hard negative perturbations

Experimental setup. Beyond the perturbations considered in the main paper, we further conduct robustness experiments with hard-negative perturbations to examine whether the model can correctly distinguish positive captions from negative ones even under extremely subtle changes in long-context captions. We generate hard-negative captions for the Urban-1k dataset [79] by replacing a single word in the original captions using LLaMA-30B [69], following FG-OVD benchmark [6]. Content words (nouns, verbs, adjectives, and adverbs) are identified using part-of-speech tagging, and up to five of them are randomly selected per caption. Each selected word is replaced by the language model with a grammatically valid word of diferent meaning, generating up to five hardnegative captions per sample. An example of an LLM-generated hard-negative sample is shown in Fig. S11.

Evaluation protocol. For each model, we compute the similarity between the image and the positive caption, as well as the similarity between the image and the corresponding hard-negative caption. The accuracy is defined as the percentage of cases where the model assigns a higher similarity score to the positive caption than to the hard-negative caption. This evaluation protocol is consistent with existing benchmarks [6, 24, 81].

Table S7: Robustness to LLM-generated hard-negative perturbations. Hardnegative captions are generated for the Urban-1k dataset [79] by replacing a single word in the original captions using LLaMA-30B [69]. The metric measures the percentage of cases where the similarity between the image and the positive caption exceeds that between the image and the corresponding hard-negative caption. Our model achieves the highest score, indicating a stronger ability to distinguish subtle semantic diferences in captions.

<table><tr><td>Model</td><td>Accuracy (%)</td></tr><tr><td>Long-CLIP [79]</td><td>54.01</td></tr><tr><td>HiMo-CLIP [75]</td><td>52.94</td></tr><tr><td>FineLIP [4]</td><td>53.54</td></tr><tr><td>HyFL-CLIP (Ours)</td><td>62.07</td></tr></table>

Experimental results. Experimental results in Tab. S7 show that HyFL-CLIP (Ours) achieves strong performance in distinguishing hard-negative captions from positive ones. While other models perform close to near-random under this challenging setting, our model maintains a clear margin. This improvement can be attributed to our hierarchical entailment mechanism via Einstein midpoint aggregation, which promotes stronger alignment between token-level representations in long contexts and the global token.

## S5 Additional Results

## S5.1 Comparison with state-of-the-art hyperbolic VLMs

Experimental setup. We compare HyFL-CLIP with state-of-the-art hyperbolic VLMs, including MERU [13], HyCoCLIP [54], and UNCHA [34]. For each hyperbolic VLM, we further apply our hyperbolic fine-tuning framework to evaluate whether the proposed long-context adaptation is also beneficial for models already trained in hyperbolic space. Since these models already operate in hyperbolic space, we omit the Euclidean-to-hyperbolic distillation loss when finetuning them. We also include OpenCLIP [12] and HyFL-CLIP initialized from OpenCLIP for comparison. All experiments are conducted using ViT-B models.

Table S8: Comparison with state-of-the-art hyperbolic VLMs on zero-shot long-caption cross-modal retrieval. Each entry reports T2I / I2T Recall@1 (For hyperbolic VLMs, “+ Ours” denotes applying our hyperbolic fine-tuning framework without the Euclidean-to-hyperbolic distillation loss.

<table><tr><td>Model</td><td>DOCCI [53]</td><td>DCI [71]</td><td>Long-DCI [52]</td><td>Urban-1k [79]</td></tr><tr><td colspan="5">State-of-the-art hyperbolic VLMs</td></tr><tr><td>MERU [13]</td><td>49.80 / 53.16</td><td>46.52 / 49.77</td><td>32.11 / 34.00</td><td>45.70 / 51.90</td></tr><tr><td>MERU + Ours</td><td>58.10 / 61.71</td><td>55.03 / 58.93</td><td>41.08 / 44.75</td><td>74.20 / 74.30</td></tr><tr><td>HyCoCLIP [54]</td><td>52.04 / 53.78</td><td>47.87 / 49.02</td><td>32.60 / 33.39</td><td>47.70 / 56.60</td></tr><tr><td>HyCoCLIP + Ours</td><td>61.16 / 63.73</td><td>58.88 / 60.68</td><td>42.74 / 45.30</td><td>75.30 / 76.70</td></tr><tr><td>UNCHA [34]</td><td>46.76 / 46.73</td><td>45.12 / 44.82</td><td>29.16 / 29.27</td><td>39.90 / 43.80</td></tr><tr><td>UNCHA + Ours</td><td>63.57 / 66.92</td><td>60.58 / 62.48</td><td>44.23 / 47.03</td><td>77.10 / 76.80</td></tr><tr><td colspan="5">Euclidean VLM with our hyperbolic fine-tuning</td></tr><tr><td>OpenCLIP [12]</td><td>57.22 / 60.84</td><td>46.97 / 50.78</td><td>33.73 / 36.83</td><td>53.40 / 67.50</td></tr><tr><td>HyFL-CLIP (Ours)</td><td>81.12 / 78.41</td><td>71.79 / 71.54</td><td>58.75 / 59.00</td><td>91.10 / 91.80</td></tr></table>

Evaluation protocol. We evaluate zero-shot long-caption cross-modal retrieval on DOCCI [53], DCI [71], Long-DCI [52], and Urban-1k [79]. Each entry reports T2I / I2T Recall@1 (%). The hyperbolic VLMs are trained on substantially smaller-scale pretraining data than OpenCLIP, using approximately 20.5M image-text pairs compared with 2.3B pairs for OpenCLIP.

Experimental results. Tab. S8 shows that applying our fine-tuning framework consistently improves existing hyperbolic VLMs across all datasets. These results indicate that our method is efective not only for transferring Euclidean CLIP representations into hyperbolic space, but also for adapting existing hyperbolic VLMs to long-context scenarios. Nevertheless, the fine-tuned hyperbolic VLMs still underperform HyFL-CLIP initialized from OpenCLIP, suggesting that leveraging a strong Euclidean VLM through eficient fine-tuning is both efective and computationally practical.

## S5.2 Full results of zero-shot long-caption cross-modal retrieval under caption perturbations.

Experimental results. Table S9 shows the full zero-shot retrieval results under caption perturbations. HyFL-CLIP (Ours) achieves the best average performance across all perturbation settings, including word dropout, sentence removal, order shufling, and random subsampling. These results demonstrate that our method is more robust to noisy, incomplete, and reordered long captions compared to existing long-caption VLMs.

Table S9: Full robustness comparison under caption perturbations. Each entry reports the average retrieval performance under a specific caption perturbation setting, including word dropout, sentence removal, order shufling, and random subsampling. The average rows summarize performance across all evaluated datasets. Results marked with ∗ are evaluated using our own implementation, while results marked with † are obtained using checkpoints provided by the original authors.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="2">Word Dropout</td><td>Sent. Removal</td><td>Order Shuffling</td><td colspan="2">Random Subsampling</td></tr><tr><td>p=0.3</td><td>p=0.5</td><td>first</td><td>random</td><td>n=2</td><td>n=3</td></tr><tr><td rowspan="5">Long-CLIP†[79]</td><td>ShareGPT4V [10]</td><td>86.91</td><td>75.79</td><td>88.60</td><td>88.80</td><td>45.34</td><td>51.19</td></tr><tr><td>Urban1k [79]</td><td>61.73</td><td>44.78</td><td>59.40</td><td>68.82</td><td>22.51</td><td>27.98</td></tr><tr><td>DOCCI [53]</td><td>59.56</td><td>59.66</td><td>51.03</td><td>59.72</td><td>25.41</td><td>30.17</td></tr><tr><td>DCI [71]</td><td>49.48</td><td>38.74</td><td>45.77</td><td>53.09</td><td>23.58</td><td>28.70</td></tr><tr><td>DCI-Long [52]</td><td>34.84</td><td>25.45</td><td>32.23</td><td>38.53</td><td>15.65</td><td>18.43</td></tr><tr><td></td><td>Average</td><td>58.50</td><td>48.88</td><td>55.41</td><td>61.79</td><td>26.50</td><td>31.29</td></tr><tr><td rowspan="5">HiMo-CLIP* [75]</td><td>ShareGPT4V [10]</td><td>94.80</td><td>86.27</td><td>95.60</td><td>96.93</td><td>48.14</td><td>57.08</td></tr><tr><td>Urban1k [79]</td><td>73.47</td><td>54.98</td><td>68.95</td><td>81.16</td><td>24.77</td><td>32.74</td></tr><tr><td>DOCCI [53]</td><td>71.50</td><td>71.46</td><td>60.90</td><td>71.65</td><td>26.84</td><td>33.81</td></tr><tr><td>DCI [71]</td><td>59.16</td><td>47.05</td><td>54.93</td><td>64.15</td><td>25.12</td><td>30.04</td></tr><tr><td>DCI-Long [52]</td><td>45.70</td><td>33.97</td><td>41.81</td><td>50.80</td><td>16.35</td><td>20.55</td></tr><tr><td></td><td>Average</td><td>68.93</td><td>58.75</td><td>64.44</td><td>72.94</td><td>28.24</td><td>34.84</td></tr><tr><td rowspan="5">FineLIP* [4]</td><td>ShareGPT4V [10]</td><td>93.74</td><td>85.48</td><td>94.80</td><td>96.23</td><td>45.34</td><td>53.42</td></tr><tr><td>Urban1k [79]</td><td>73.45</td><td>54.19</td><td>68.20</td><td>82.57</td><td>24.70</td><td>31.96</td></tr><tr><td>DOCCI [53]</td><td>72.92</td><td>72.84</td><td>61.54</td><td>72.93</td><td>26.35</td><td>33.24</td></tr><tr><td>DCI [71]</td><td>58.27</td><td>47.19</td><td>54.18</td><td>64.33</td><td>23.79</td><td>28.78</td></tr><tr><td>DCI-Long [52]</td><td>45.41</td><td>33.80</td><td>42.17</td><td>51.00</td><td>15.84</td><td>19.65</td></tr><tr><td></td><td>Average</td><td>68.76</td><td>58.70</td><td>64.18</td><td>73.41</td><td>27.20</td><td>33.41</td></tr><tr><td rowspan="5">HyFL-CLIP (Ours)</td><td>ShareGPT4V [10]</td><td>95.40</td><td>95.44</td><td>96.45</td><td>97.37</td><td>56.52</td><td>63.59</td></tr><tr><td>Urban1k [79]</td><td>77.60</td><td>77.14</td><td>75.45</td><td>86.08</td><td>27.64</td><td>36.25</td></tr><tr><td>DOCCI [53]</td><td>68.18</td><td>68.48</td><td>65.13</td><td>75.53</td><td>30.93</td><td>37.20</td></tr><tr><td>DCI [71]</td><td>61.43</td><td>61.74</td><td>58.51</td><td>67.69</td><td>28.73</td><td>34.56</td></tr><tr><td>DCI-Long [52]</td><td>48.32</td><td>48.18</td><td>45.56</td><td>54.14</td><td>18.93</td><td>23.63</td></tr><tr><td></td><td>Average</td><td>70.19</td><td>70.20</td><td>68.22</td><td>76.16</td><td>32.55</td><td>39.05</td></tr></table>

## S5.3 SDXL experiments with ViT-L

Experimental setup. Following prior works [4, 52, 79], we apply our HyFL-CLIP model to Stable Difusion XL (SDXL) [57] in a plug-and-play manner. In the original SDXL architecture, the text encoder consists of a CLIP-L encoder and an OpenCLIP bigG encoder. The prompts are encoded by both encoders, and the resulting text embeddings are concatenated to condition the image generation process.

We replace the CLIP-L text encoder with our HyFL-CLIP-L. Since our model produces hyperbolic embeddings, these embeddings are mapped to the Euclidean space using the logarithmic map transport before concatenation. To extend the context length of the OpenCLIP bigG text encoder without retraining, we expand the positional embeddings via linear interpolation. The first 20 positional embeddings are preserved to maintain the well-trained short context structure. For the remaining positions, additional embeddings are inserted between consecutive positions through linear interpolation, since retraining this large encoder would be computationally expensive.

Table S10: Frechet Inception Distance (FID) and CLIP [59] similarity scores for text-to-image generation. We randomly sample 500 prompts from COCO [42], DOCCI [53], and Long-DCI [52], and evaluate on the full DrawBench [62] benchmark. Our model achieves competitive or best performance in both image quality and text–image alignment. The downward arrow $( \downarrow )$ indicates that lower values are better, while the upward arrow (↑) indicates that higher values are better.

<table><tr><td rowspan="2">Model</td><td colspan="3">FID Score ↓</td><td colspan="4">CLIP Similarity Score ↑</td></tr><tr><td>COCO</td><td>DOCCI</td><td>Long-DCI</td><td>COCO</td><td>DOCCI</td><td>Long-DCI</td><td>DrawBench</td></tr><tr><td>Long-CLIP [79]</td><td>26.64</td><td>26.03</td><td>30.62</td><td>0.438</td><td>0.446</td><td>0.435</td><td>0.427</td></tr><tr><td>HiMo-CLIP [75]</td><td>27.07</td><td>25.92</td><td>30.42</td><td>0.431</td><td>0.445</td><td>0.437</td><td>0.419</td></tr><tr><td>HyFL-CLIP (Ours)</td><td>27.23</td><td>24.52</td><td>28.28</td><td>0.444</td><td>0.451</td><td>0.443</td><td>0.427</td></tr></table>

Evaluation protocol. We randomly sample 500 prompts from COCO [42], DOCCI [53], and Long-DCI [52], and evaluate on the full DrawBench [62] benchmark. DOCCI [53] and Long-DCI [52] are used to evaluate the model’s ability to generate images from long captions. Using the corresponding captions, we generate images with SDXL integrated with HyFL-CLIP. Following FineLIP [4], we evaluate image quality using the Fréchet Inception Distance (FID) and measure text–image alignment using CLIP similarity. Other baselines [75, 79] are implemented using their respective ViT-L/14 backbone versions.

Experimental results. Fig. S8, Fig. S9, Fig. S10, and Tab. S10 present qualitative and quantitative comparisons between HyFL-CLIP (Ours) and other methods [75, 79] when integrated into SDXL [57]. The results demonstrate that our method integrates seamlessly with existing generation pipelines while producing high-quality text embeddings that remain well aligned with long captions.

## S5.4 Additional examples of text-token contribution weight visualization

We visualize the contribution of each text token to its paired target image to analyze how individual tokens influence image–text alignment. Specifically, given a long caption, we compute the similarity between each caption token and the paired image and use these similarities to weight the token embeddings when aggregating them into a single representation during training. The resulting weights quantify the contribution of each token to the aggregated representation, and we visualize these weights to analyze the contribution of individual tokens.

We visualize these contribution weights for each token in Fig. S5, Fig. S6, and Fig. S7. In each figure, the image and the full caption are displayed at the top, while the bottom-left and bottom-right panels correspond to the models trained without and with the entailment loss $( \mathcal { L } _ { e n t } )$ , respectively. Darker red indicates a larger contribution weight. Without $\mathcal { L } _ { e n t } .$ the contribution weights tend to be more uniformly distributed across tokens, whereas the model trained with $\mathcal { L } _ { e n t }$ assigns larger weights to semantically informative tokens that are strongly related to the visual content.

This behavior arises because our hierarchical entailment formulation aggregates token representations with diferent weights depending on how strongly each token contributes to the image–text alignment. As a result, the model emphasizes semantically relevant tokens when forming the final caption representation, leading to better alignment with the visual content. We further visualize token activations from HiMo-CLIP [75] using the same procedure (see Fig. S12, Fig. S13, and Fig. S14), and observe that our model activates a richer set of contextually relevant tokens compared to HiMo-CLIP [75], indicating that our formulation captures more informative semantic cues from long captions.

## S5.5 More examples on long-text retrieval under caption perturbation

We provide additional qualitative examples corresponding to Fig. 1 to further illustrate the robustness of our model under caption perturbations. For each long text query paired with a ground-truth (GT) image, we show the full captions after applying three perturbation strategies: random subsampling, order shuffling, and word dropout. The modified captions are presented at the top of the figure to highlight how the textual input changes under each perturbation. At the bottom, we display the top-1 retrieved images produced by HiMo-CLIP [75] and our method using the perturbed captions as queries.

In Fig. S4(a), the perturbed queries produced by subsampling and shufling still clearly describe a statue of a man scene. However, HiMo-CLIP [75] retrieves unrelated asphalt images instead of the correct relief sculpture scene. Even in the word-dropout case, the retrieved result fails to reflect the pose of the man described in the caption. A similar pattern appears in Fig. S4(b). Although the key object “shopping cart” remains present in all perturbed captions, HiMo-CLIP [75] consistently retrieves irrelevant images as the top result. In contrast, our method continues to retrieve the correct scene across all perturbation types. These examples demonstrate that our model is significantly more robust to caption perturbations than existing methods. For completeness, the full captions used in the experiments corresponding to Fig. 1 are provided in Tab. S11.

![](images/2902695be09f4473098ecbda2d2a444bcbd6944cabacc6a476a1ab7176e9809c.jpg)  
Darker gray stains are visible on the relief an the stairs below, The relief is of a man facing the left with his body facing the front and kneeling on his left leg with his right hand at the top of a depiction of the state of Texas.

![](images/256f4fa6dbf29ae7c351e68df597355074131ba37aae7e83c7d490b6715a9159.jpg)

![](images/ea92b1d33efd715969a3c2587dbaa5bca25f447756cf48c7adc77c4a8c4ef2e3.jpg)

## Robust Long-Context Retrieval under Caption Perturbation

![](images/26c8b8e1fd8086d233d0e1c9f35b36b4fb7c55b09a04a67a0cb75d4471f15993.jpg)

![](images/9bab45f1f1f92ffd120ad3bf70741e6353273a6eb66dcfaa1c818e4ba232c3fc.jpg)

![](images/6f8a0505ee6a44867e444e8b1643a129fcb6e5725e7330868a611d0bdab00c15.jpg)

![](images/dc9b8131de72cdb9459269494c732d420c118b5487d90d4694b3b01488e02734.jpg)  
(a) Example with a roadside cart scene.

![](images/10620858d6cdf9f4d4f2a395bfac5c6b03bcd4fc260b03466aa405f1ad4ac3d3.jpg)  
Robust Long-Context Retrieval under Caption Perturbation  
(b) Example with a relief sculpture scene.

Fig. S4: Robust long-context retrieval under caption perturbations on Long-DCI. We compare retrieval results obtained with HiMo-CLIP and our HyFL-CLIP when captions are perturbed in diferent ways, including random subsampling, order shufling, and word dropout. For each query, we show the ground-truth (GT) image and the top-1 retrieval result from each method. While HiMo-CLIP often retrieves incorrect images under caption perturbations, our method consistently retrieves the correct image, demonstrating improved robustness to long-context caption variations.

Text token contribution weight visualization  
Image  
![](images/856e25ceb706bf394ea7d8375087112cb403c543e5b2e078a3394c1f3adc480c.jpg)

$$
\mathcal {L} _ {\mathrm{ent}}
$$

## Caption

This is a very busy shopping district. there are many people in the street and the sidewalk. in the foreground there is a glass paneled red telephone booth with multiple reflections in the glass. close to the phone booth, there are two women, one dressed in mostly black leather and another in a blue and red shirt and jeans, carrying an orange bag - walking in the opposite direction from each other. there is a long sidewalk on the right side of the street made of various sizes of tan bricks. there is a concrete post and a light pole on the edge of the sidewalk. near the cobble stone street near the sidewalk there are some businesses such as vape life and the burger shed. above the burger shed they have decorated with flowers on the balcony railing and with a large wooden box on the sidewalk in front of the restaurant. along both sides of the road there are signs for businesses hanging from all the buildings of various heights. In the road there are two men wearing yellow vests as well as a multitude of other patrons. at the far end of the road is a large clock tower which is prominent amongst the various businesses and buildings. It is a mostly cloudy day but there are a few breaks in the clouds.

from

$$
\mathcal {L} _ {\mathrm{ent}}
$$

carrying walking from

Fig. S5: Visualization of text token contribution weights when trained with or without $\mathcal { L } _ { e n t }$ (hierarchical entailment). We visualize the contribution of individual text tokens to the merging process by measuring their similarity with the corresponding image representation. As shown in the figure, the model trained with $\mathcal { L } _ { e n t }$ highlights semantically important tokens more clearly, indicating that hierarchical entailment encourages the model to focus on informative words in the caption.

Text token contribution weight visualization without $\mathcal { L } _ { \mathrm { e n t } }$ (Hierarchical entailment) with $\mathcal { L } _ { \mathrm { e n t } }$ (Hierarchical entailment)  
Image  
![](images/728431e0b87fb8107f2f4349c2a0ec5de0f1d9d6e03c9405bca160d1f957e304.jpg)

## Caption

A well-lit underground train station. in the center of the platform, are retracting doors with a pair of escalators that either enter or exit the train platform. the platform grounded comprises of a border on the left and right side of white, and throughout the middle is a color blocked diagonal design. each color block comprises of four rows of either grey or white. there are two panels of information near the entrance / exit of the platform that contains schedules, useful transit information and a large map of the routes throughout the city. on the right side above the platform hangs directional signs to navigate where you need to go. on each ends are yellow panels that tell you where you need to go for each terminal. towards the middle is where to go prohibiting carts with bags, and which transit is on each side. also hanging above the platform is the name of the train station you are at, along with an orange no smoking sign. on the left side of the platform hangs multiple signs. one that advises the danger of falling onto the tracks. two blank navy panels and a cube-like illuminated green sign that points to the right. there is an orange-red train on each side of the platform.

<table><tr><td><img src="images/559a4ff08c1a7defe355f97b4968be500b9bed36d95913941e36f6c3c5e89bd3.jpg"/></td><td>With &quot;Cent (Intricateal entainment)</td></tr></table>

Fig. S6: Visualization of text token contribution weights when trained with or without $\mathcal { L } _ { e n t }$ (hierarchical entailment). We visualize the contribution of individual text tokens to the merging process by measuring their similarity with the corresponding image representation. As shown in the figure, the model trained with $\mathcal { L } _ { e n t }$ highlights semantically important tokens more clearly, indicating that hierarchical entailment encourages the model to focus on informative words in the caption.

![](images/f86d6393e3faff9c9755ad919f5654a04ac1894e9b08041abb8ef822a062c043.jpg)

$$
\mathcal {L} _ {\mathrm{ent}}
$$

## Caption

if is a photo of a placid looking ocean on a warm looking summer day. the sky is covered with numerous fluffy white clouds and the ocean water looks calm with only a few small waves. there is a small rocky looking island covered in trees a bit off the shore, as well as a number of vessels floating across the water. most of the ships are small and could only hold a couple of people, but there are also a number of larger ships further out far past the island that look like they hold cargo or other industrial purposes. there are some green leafy tree branches hanging down into the photo, obscuring the shot a bit, but not overly so. it looks like there is land far off in the distance across the water along the horizon, as well as a second island even further out near the center of the picture.

![](images/315fcb53a1d8aef9d470b1c5dd83ea179974746c6e4d0251331d874cacb50d16.jpg)  
Fig. S7: Visualization of text token contribution weights when trained with or without $\mathcal { L } _ { e n t }$ (hierarchical entailment). We visualize the contribution of individual text tokens to the merging process by measuring their similarity with the corresponding image representation. As shown in the figure, the model trained with $\mathcal { L } _ { e n t }$ highlights semantically important tokens more clearly, indicating that hierarchical entailment encourages the model to focus on informative words in the caption.

## Dataset: COCO [42]

Prompt : A corner of a rest room with a cookie and coffee.

![](images/e56a02303a5155262216c8c1b61028d7710219fee05d674568da35a53f20efa6.jpg)  
Long-CLIP [79]

![](images/1cf363ef579673f92d58fe9bf312f594b0390f7f0d3a5e4f3befa3f84cf52fe9.jpg)  
HiMo-CLIP [75]

![](images/8b1e3233a6ed7f1768cc24a4a5700c4d1c752aa878cdd1b9751c88b58ccdfdc6.jpg)  
HyFL-CLIP (Ours)

Prompt : A train is sitting along tracks under and overpass.  
![](images/9256d6e10c3cdd6822c17ad900fbed864b0972d3d59c654c6ce0d3c33e821f83.jpg)  
Long-CLIP [79]

![](images/be9766c32398f7f46f1a823d24f7008807c1cf7c98cd9b31ba3cd4b525501569.jpg)  
HiMo-CLIP [75]

![](images/94e97c77a5467a378d245024461e54c76e9fc8e89b2339ea20fffcd160efe785.jpg)  
HyFL-CLIP (Ours)

## Dataset: DrawBench [62]

Prompt : A bicycle on top of a boat.

![](images/abe666767aa0c9369f51cf7039325be9eeda863c7c1ffc8bc56e7980a2f32395.jpg)  
Long-CLIP [79]

![](images/3cea1c592de5e45bb0fcee7330e010c385e306740aaa31897aa23c32cff1a48a.jpg)  
HiMo-CLIP [75]

![](images/414a01e1d732068dc11797d669570ec35d18dfed1b2220acafb153679f1fde77.jpg)  
HyFL-CLIP (Ours)

Prompt : Four dogs on the street.  
![](images/10c8ae47708691ef714cb2bd075d0782b05483d4bd2dadbd50827f5cddd5e8dd.jpg)  
Long-CLIP [79]

![](images/8ff8abb51c434fef51b02c15e84a532b9634c33ad03f17e1ff5afa2b8f671f87.jpg)  
HiMo-CLIP [75]

![](images/f2a3e08e0f675cdd350d1a0836d01392a3517979ae571d6dd558cabfecb13ab1.jpg)  
HyFL-CLIP (Ours)

Fig. S8: Comparison of images generated from COCO [42] captions using HyFL-CLIP (Ours) integrated SDXL and baselines. Images generated with our model preserve finer visual details and exhibit higher fidelity to the given captions compared to the baseline methods. This demonstrates that HyFL-CLIP provides more precise semantic guidance for text-to-image generation.

## Dataset: DOCCI [53]

Prompt : A front view of two black leather chairs in the corner of a room. In between them is a small, round white table with metal legs. It is in the corner and has a book lying on top of it. A chair is on each side of it. The chairs are sitting on a light wood floor. Both chairs have shadows under them on the floor. The chair on the right has a shadow to the right of it and on the wall behind it. The wall on the left has light shining on it. The walls are white with a black trim and white molding on the bottom.

![](images/12b9aaf55e8230ef324646d8cc5337d2b045627cad88539cbd7c851b2b8564c3.jpg)  
Long-CLIP [79]

![](images/1586134fa0f8ab46215b86265a1a8df9f6d2b409a6323b8ed0ce078a4b57803c.jpg)  
HiMo-CLIP [75]

![](images/e59af939dca9609bb38a0aebdeed4a40b7248a21c781bdc2a6283c9b98d644d6.jpg)  
HyFL-CLIP (Ours)

Prompt : Indoor, front view of a taxidermied deer in a museum display. The deer is a light brown color with its body facing forward, and its head turned slightly to the left, it has a dark nose and medium-sized pointy antlers on the top of its head. The deer's back hooves are on a rock and the front hooves are on top of a log that stretches from the bottom right to the center left of the frame. A tree is to the left of the deer and placed vertically, and to the right of the deer is a concrete pillar that has a rectangular shape and goes to the top edge of the frame. The deer has light casting onto it, with a warm yellow glow on its left, and a white artificial light to its left.

![](images/1518ad55f4b7fca31c531319c9de08fd32cf3dc178909e8a4ae6b439b9553b8c.jpg)  
Long-CLIP [79]

![](images/45a1e0dcb799b46158dcd977516183a50acf891faa3a398f7f95514627eb519d.jpg)  
HiMo-CLIP [75]

![](images/4cb0e6f741c160eba185ab7d836ff26fabb31de7d38897574b98714747770f21.jpg)  
HyFL-CLIP (Ours)  
Fig. S9: Comparison of images generated from DOCCI [53] captions using HyFL-CLIP (Ours) integrated SDXL and baselines. Images generated with our model preserve finer visual details and exhibit higher fidelity to the given captions compared to the baseline methods. This demonstrates that HyFL-CLIP provides more precise semantic guidance for text-to-image generation.

## Dataset: Long-DCI [52]

Prompt : A large passenger airplane. The airplane is white, yellow, orange and red in color. It has two engines, one on each wing. The landing gear is down, showing three sets of wheels. There are two wheels on each set. There are three visible doors on the side. One at the front, middle and back of the airplane. "VietjetAir.com," is written on the side of the airplane, on the winglet, and on the vertical stabilizer in white writing. "Enjoy $\mathsf { f l y i n g l } ! ^ { \mathsf { \tiny { n } } }$ is also written on the side of the airplane in white writing. The number "VN-A626," can be seen on the back, top portion of the airplane. There is a bright yellow star on the engine and a bright yellow star in a red colored square on the front of the airplane. There is a tiny white light on the very back portion of the airplane. Forty four small windows can be seen on the side of the airplane, with two larger windows in the front. The plane is seen against a blue sky with clouds. The clouds are all white. Some of the clouds are thick in appearance. A few of the clouds appear to be very faint and wispy in appearance.

![](images/8bb25e3179e25f155fc1bca14d32eb343dae65c297eed6b565a3e62fbe22ff6f.jpg)  
Long-CLIP [79]

![](images/cff86d749f05fc4079ef00c88e67892ed2f72fd0ade56cf44c5011d99063cfe8.jpg)  
HiMo-CLIP [75]

![](images/4647d1fadac48c9b3745ca64f715db0ea208bcd9a9583c7b93ad8daba160d2fc.jpg)  
HyFL-CLIP (Ours)

Prompt : A forest type of area on the left there is a wide dark gray path. The path leads along the area up and then bends right. To the right of the path there is a drop in elevation and a small river is below. At the midway point of the path there is a stone dark gray and white bridge that leads to the right. That bridge has the bottom of it on the left carved out and there is water running under it. On the left and right of the area there is a lot of green there. The left is bright green and a mix of trees and plants. Further down the trail to the right there are some larger trees but the trunks are still mostly thin for the most part. If you go down the bridge to the right there is a heavily wooded area bright green trees that have a lot of yellow leaves or types of fruit hanging from the bright yellow. The path is covered in orange and brown leaves as is the river area on the right.  
![](images/949bdcc6c268d3b4d0c6309e359b3abd9d7575b99fa65640a6f608d9b8269817.jpg)  
Long-CLIP [79]

![](images/064516a79e991208bc21ec73e730d21265028b1caf50dcb56a9bcc611bc3fa3c.jpg)  
HiMo-CLIP [75]

![](images/7184f933d8a628cf0ad9b56681ca21ee0103e7ea87478369949a05572c270a9b.jpg)  
HyFL-CLIP (Ours)  
Fig. S10: Comparison of images generated from Long-DCI [52] captions using HyFL-CLIP (Ours) integrated SDXL and baselines. Images generated with our model preserve finer visual details and exhibit higher fidelity to the given captions compared to the baseline methods. This demonstrates that HyFL-CLIP provides more precise semantic guidance for text-to-image generation.

Original Image  
![](images/61f20b0dd0b4ae4007c8b5c247b522d3de3e2fb40b8eb5bce16fb59b36be385d.jpg)

Original caption: "An urban street scene at twilight captures vehicles and pedestrians. The sky is a gradient of blue, suggestive of early evening. Streetlights show a yellow traffic light signaling caution. On the left, a blur of a motorcycle moves past, implying motion. Various cars are scattered across the intersection, some facing the camera, others positioned sideways, in shades of white, silver, and darker colors. On the street's corner, a two-story building with a red roof is visible, while assorted commercial signages are partially visible in the background. In the foreground, pedestrians are crossing the street, with one person in motion, rendered as a blurred figure due to the camera's exposure. Electric cables crisscross overhead, and a green circular logo adorns a building to the right."

LLM-generated hard negative caption: "An urban street scene at twilight captures vehicles and pedestrians. The sky is a gradient of blue, suggestive of early evening. Streetlights show a green traffic light signaling caution. On the left, a blur of a motorcycle moves past, implying motion. Various cars are scattered across the intersection, some facing the camera, others positioned sideways, in shades of white, silver, and darker colors. On the street's corner, a two-story building with a red roof is visible, while assorted commercia signages are partially visible in the background. In the foreground, pedestrians are crossing the street, with one person in motion, rendered as a blurred figure due to the camera's exposure. Electric cables crisscross overhead, and a green circular logo adorns a building to the right."

Fig. S11: Example of an LLM-generated hard-negative caption from Urban-1k [79]. A single-word perturbation (yellow → green) produces a highly subtle yet semantically incorrect caption, illustrating the dificulty of distinguishing such hard negatives.

![](images/74c1ef1725d3edb1d95f6320d6198e9320ae22f6fa047d3ed89c1ca13f27adb2.jpg)

it looks like a group of men have come together in front of some very ornate gold buildings to watch something happening in the sky. there is a piece of equipment that looks like it has two wheels on it. it is on a tall wooden stand. it looks like that equipment may be shooting something gold colored into the sky, and it is making objects in the sky, a cloud in the sky, and yellow all over the clothes of the people watching. some people are standing. some are sitting. one man is holding up a phone, as if he is taking a picture.

Text token contribution weight visualization  
![](images/b73f9f884023ea15a71449b9b41d5a22608c122c0b764dfc945f49a15925cf77.jpg)  
Fig. S12: Token contribution comparison between HiMo-CLIP [75] and HyFL-CLIP (Ours). For the same image-caption pair, HyFL-CLIP assigns higher weights to semantically meaningful tokens that correspond to visual elements in the image.

![](images/a8f0f7301007e66761324ddf86906fa061369b7d9e7afcaa34ced248516902a6.jpg)

## Caption

The focus of the image is on the la floridita restaurant, a local institution, favored by ernest hemingway. the restaurant is painted in pink with a white marquee printed in green as well as a large neon sign that hangs over its entrance. there are plenty of windows in the restaurant to see the street outside. there appears to be a large crowd, all dressed in varying styles, outside the restaurant with several taxis parked nearby. there are several other building in the image. one appears to be an abandoned building directly behind the restaurant, with a lot of decay and broken windows. to the left of the restaurant there is a small two story hotel, painted in yellow but also appears to have some damage as windows are boarded up inside. to the right of the restaurant, is a large building that is colored a tan hue and features ornate trim and architectural flourishes which make it look to be well kept. cars line the street and there is a splash of green with a large tree growing.  
Text token contribution weight visualization  
![](images/52842c1adc1840982becf687bb891ac98a3bd24c78d5a9e488d9979fe73b8054.jpg)  
Fig. S13: Token contribution comparison between HiMo-CLIP [75] and HyFL-CLIP (Ours). For the same image-caption pair, HyFL-CLIP assigns higher weights to semantically meaningful tokens that correspond to visual elements in the image.

![](images/78903d1ada204d69fa495bb27537d1eb0619383470ea924573496317fd57e87e.jpg)

this is a photo of a stone fountain with statues around it. a large building is on its left and a more ornate building is on its right. the stone fountain has a crafted base with a symbol in its middle and black benches surrounding it. there are people standing and walking around in the background between the buildings. smaller sections of the building can be seen on either end. the sky has a lot of thin clouds and there is a mountain or hill range in the background. there are also little buildings on each side with roof coverings. people perhaps sit under the shade to take a break. there are multiple street lights as well behind the stone fountain.

Text token contribution weight visualization  
![](images/ba9ffe2553b952dcd9541938086d20c1b336081c05b4a9f0a64fde55ca904eff.jpg)  
Fig. S14: Token contribution comparison between HiMo-CLIP [75] and HyFL-CLIP (Ours). For the same image-caption pair, HyFL-CLIP assigns higher weights to semantically meaningful tokens that correspond to visual elements in the image.

Table S11: Examples of caption perturbations applied to a long text query. We show the full caption in Fig.1 and its perturbed variants generated by random subsampling, order shufling, and word dropout. These perturbations preserve partial semantic information while altering the structure or completeness of the caption.

<table><tr><td>Type</td><td>Caption</td></tr><tr><td>Long text query</td><td>The image shows an urban street intersection with vehicular and pedestrian activity. In the foreground, there is a pedestrian crossing the street marked with white zebra lines and a black car in the middle of the crosswalk. The traffic light for pedestrians is visible, displaying a red hand signal indicating a “Do Not Walk” command. A man wearing a white jacket and dark pants is walking away from the camera while carrying a black bag in his left hand. To the left, another pedestrian wearing a white and red outfit is crossing the street. Buildings with varying facades line the street, and clear blue skies with scattered clouds appear above.</td></tr><tr><td>Random subsampling</td><td>To the left, another pedestrian wearing a white and red outfit is crossing the street. The traffic light for pedestrians is visible, displaying a red hand signal indicating a “Do Not Walk” command. The image shows an urban street intersection with vehicular and pedestrian activity.</td></tr><tr><td>Order shuffling</td><td>Buildings with varying facades line the street, and clear blue skies with scattered clouds appear above. The traffic light for pedestrians is visible, displaying a red hand signal indicating a “Do Not Walk” command. The image shows an urban street intersection with vehicular and pedestrian activity. To the left, another pedestrian wearing a white and red outfit is crossing the street. In the foreground, there is a pedestrian crossing the street marked with white zebra lines and a black car in the middle of the crosswalk. A man wearing a white jacket and dark pants is walking away from the camera while carrying a black bag in his left hand.</td></tr><tr><td>Word dropout</td><td>The image shows an urban street intersection with vehicular and pedestrian activity. In the foreground, a crossing the marked with white zebra lines and a black car middle the. The traffic light for is visible, displaying a red hand signal indicating a “Do Not Walk” command. A man dark is walking the camera, black bag in his hand. Another pedestrian appears wearing a white and red the street. Buildings facades line the street, clear blue skies with scattered clouds.</td></tr></table>

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

2. Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson, M., Batra, D., Parikh, D., Lee, S., Anderson, P.: Nocaps: Novel object captioning at scale. In: CVPR (2019)

3. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. NeurIPS (2022)

4. Asokan, M., Wu, K., Albreiki, F.: Finelip: Extending clip’s reach via fine-grained alignment with longer text inputs. In: CVPR (2025)

5. Atigh, M.G., Schoep, J., Acar, E., van Noord, N., Mettes, P.: Hyperbolic image segmentation. In: CVPR (2022)

6. Bianchi, L., Carrara, F., Messina, N., Gennaro, C., Falchi, F.: The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding. In: CVPR (2024)

7. Byun, J., Kim, D., Moon, T.: Mafa: Managing false negatives for vision-language pre-training. In: CVPR (2024)

8. Chami, I., Gu, A., Nguyen, D.P., Ré, C.: Horopca: Hyperbolic dimensionality reduction via horospherical projections. In: ICML (2021)

9. Chami, I., Ying, Z., Ré, C., Leskovec, J.: Hyperbolic graph convolutional neural networks. NeurIPS 32 (2019)

10. Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., Lin, D.: Sharegpt4v: Improving large multi-modal models with better captions. In: ECCV (2024)

11. Chen, T.S., Hung, W.C., Tseng, H.Y., Chien, S.Y., Yang, M.H.: Incremental false negative detection for contrastive learning. ICLR (2022)

12. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: CVPR (2023)

13. Desai, K., Nickel, M., Rajpurohit, T., Johnson, J., Vedantam, S.R.: Hyperbolic image-text representations. In: ICML (2023)

14. Dhingra, B., Shallue, C., Norouzi, M., Dai, A., Dahl, G.: Embedding text in hyperbolic spaces. In: TextGraphs-12 (2018)

15. Dosovitskiy, A.: An image is worth 16x16 words: Transformers for image recognition at scale. ICLR (2021)

16. Feng, Y., Wen, C., Peng, Z., Zhu, S., et al.: Retaining knowledge and enhancing long-text representations in clip through dual-teacher distillation. In: CVPR (2025)

17. Ganea, O., Bécigneul, G., Hofmann, T.: Hyperbolic entailment cones for learning hierarchical embeddings. In: ICML. PMLR (2018)

18. Ge, Y., Ren, J., Gallagher, A., Wang, Y., Yang, M.H., Adam, H., Itti, L., Lakshminarayanan, B., Zhao, J.: Improving zero-shot generalization and robustness of multi-modal models. In: CVPR (2023)

19. Geng, S., Yuan, J., Tian, Y., Chen, Y., Zhang, Y.: Hiclip: Contrastive languageimage pretraining with hierarchy-aware attention. ICLR (2023)

20. Grandvalet, Y., Bengio, Y.: Semi-supervised learning by entropy minimization. NeurIPS (2004)

21. He, N., Anand, R., Madhu, H., Maatouk, A., Krishnaswamy, S., Tassiulas, L., Yang, M., Ying, R.: Helm: Hyperbolic large language models via mixture-of-curvature experts. NeurIPS (2025)

22. He, N., Yang, M., Ying, R.: Hypercore: The core framework for building hyperbolic foundation models with comprehensive modules. arXiv preprint arXiv:2504.08912 (2025)

23. He, X., Peng, Y.: Fine-grained image classification via combining vision and language. In: CVPR (2017)

24. Hsieh, C.Y., Zhang, J., Ma, Z., Kembhavi, A., Krishna, R.: Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. NeurIPS (2023)

25. Huang, K., Duan, C., Sun, K., Xie, E., Li, Z., Liu, X.: T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. TPAMI 47(5), 3563–3579 (2025)

26. Huang, Z., Zeng, Z., Liu, B., Fu, D., Fu, J.: Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849 (2020)

27. Huynh, T., Kornblith, S., Walter, M.R., Maire, M., Khademi, M.: Boosting contrastive self-supervised learning with false negative cancellation. In: CVPR (2022)

28. Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., Schmidt, L.: Openclip (Jul 2021). https://doi.org/10.5281/zenodo.5143773, https://doi. org/10.5281/zenodo.5143773

29. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: ICML (2021)

30. Jin, L., Luo, G., Zhou, Y., Sun, X., Jiang, G., Shu, A., Ji, R.: Refclip: A universal teacher for weakly supervised referring expression comprehension. In: CVPR (2023)

31. Kang, D.U., Kim, H., Chun, S.Y.: Cdam: Class distribution-induced attention map for open-vocabulary semantic segmentations. In: The Thirteenth International Conference on Learning Representations (ICLR) (2025), iCLR

32. Karpathy, A., Fei-Fei, L.: Deep visual-semantic alignments for generating image descriptions. In: CVPR (2015)

33. Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., Lempitsky, V.: Hyperbolic image embeddings. In: CVPR (2020)

34. Kim, H., Jang, J.H., Kim, J.J., Chun, S.Y.: Uncertainty-guided compositional alignment with part-to-whole semantic representativeness in hyperbolic visionlanguage models. In: CVPR (2026)

35. Kiros, R., Salakhutdinov, R., Zemel, R.S.: Unifying visual-semantic embeddings with multimodal neural language models. arXiv preprint arXiv:1411.2539 (2014)

36. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV (2017)

37. Lang, K.: Newsweeder: Learning to filter netnews. In: Machine learning proceedings 1995 (1995)

38. Le, M., Roller, S., Papaxanthos, L., Kiela, D., Nickel, M.: Inferring concept hierarchies from text corpora via hyperbolic embeddings. In: ACL (2019)

39. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: ICML (2023)

40. Li, L., Li, J., Lei, J., Xiao, J., Shao, F., Chen, L.: Learning hierarchical hyperbolic embeddings for compositional zero-shot learning. arXiv preprint arXiv:2512.20029 (2025)

41. Li, Y., Liu, X., Kag, A., Hu, J., Idelbayev, Y., Sagar, D., Wang, Y., Tulyakov, S., Ren, J.: Textcraftor: Your text encoder can be image quality controller. In: CVPR. pp. 7985–7995 (2024)

42. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014)

43. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. NeurIPS (2023)

44. Liu, Q., Nickel, M., Kiela, D.: Hyperbolic graph neural networks. NeurIPS 32 (2019)

45. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts. ICLR (2017)

46. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. ICLR (2019)

47. Lu, J., Batra, D., Parikh, D., Lee, S.: Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. NeurIPS 32 (2019)

48. Lv, X., Zhao, Y., Yin, H., Chen, Y., Liu, J.: Msg-clip: Enhancing clip’s ability to learn fine-grained structural associations through multi-modal scene graph alignment. Pattern Recognition

49. Maas, A., Daly, R.E., Pham, P.T., Huang, D., Ng, A.Y., Potts, C.: Learning word vectors for sentiment analysis. In: Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies (2011)

50. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of machine learning research (2008)

51. Mistretta, M., Baldrati, A., Agnolucci, L., Bertini, M., Bagdanov, A.D.: Cross the gap: Exposing the intra-modal misalignment in clip via modality inversion. ICLR (2025)

52. Najdenkoska, I., Derakhshani, M.M., Asano, Y.M., Van Noord, N., Worring, M., Snoek, C.G.: Tulip: Token-length upgraded clip. ICLR (2025)

53. Onoe, Y., Rane, S., Berger, Z., Bitton, Y., Cho, J., Garg, R., Ku, A., Parekh, Z., Pont-Tuset, J., Tanzer, G., et al.: Docci: Descriptions of connected and contrasting images. In: ECCV (2024)

54. Pal, A., van Spengler, M., di Melendugno, G.M.D., Flaborea, A., Galasso, F., Mettes, P.: Compositional entailment learning for hyperbolic vision-language models. ICLR (2024)

55. Peng, Z., Xu, Z., Zeng, Z., Wen, C., Huang, Y., Yang, M., Tang, F., Shen, W.: Understanding fine-tuning clip for open-vocabulary semantic segmentation in hyperbolic space. In: CVPR (2025)

56. Plummer, B.A., Wang, L., Cervantes, C.M., Caicedo, J.C., Hockenmaier, J., Lazebnik, S.: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In: CVPR (2015)

57. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)

58. Pratt, S., Covert, I., Liu, R., Farhadi, A.: What does a platypus look like? generating customized prompts for zero-shot image classification. In: CVPR (2023)

59. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML (2021)

60. Ramasinghe, S., Shevchenko, V., Avraham, G., Thalaiyasingam, A.: Accept the modality gap: An exploration in the hyperbolic space. In: CVPR (2024)

61. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR (2022)

62. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image difusion models with deep language understanding. NeuriPS (2022)

63. Sain, A., Bhunia, A.K., Chowdhury, P.N., Koley, S., Xiang, T., Song, Y.Z.: Clip for all things zero-shot sketch-based image retrieval, fine-grained or not. In: CVPR (2023)

64. Sala, F., De Sa, C., Gu, A., Ré, C.: Representation tradeofs for hyperbolic embeddings. In: ICML (2018)

65. Sarkar, S.D., Miksik, O., Pollefeys, M., Barath, D., Armeni, I.: Crossover: 3d scene cross-modal alignment. In: CVPR (2025)

66. Sinha, A., Zeng, S., Yamada, M., Zhao, H.: Learning structured representations with hyperbolic embeddings. NeurIPS (2024)

67. Srivastava, S., Wu, K.: Hypervlm: Hyperbolic space guided vision language modeling for hierarchical multi-modal understanding. In: ICCV (2025)

68. Tifrea, A., Bécigneul, G., Ganea, O.E.: Poincar\’e glove: Hyperbolic word embeddings. arXiv preprint arXiv:1810.06546 (2018)

69. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)

70. Tung, F., Mori, G.: Similarity-preserving knowledge distillation. In: ICCV (2019)

71. Urbanek, J., Bordes, F., Astolfi, P., Williamson, M., Sharma, V., Romero-Soriano, A.: A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions. In: CVPR (2024)

72. Wang, B., Ning, Z., Ding, J., Gao, X., Li, Y., Jiang, D., Yang, J., Liu, W.: Fixclip: Dual-branch hierarchical contrastive learning via synthetic captions for better understanding of long text. In: CVPR (2025)

73. Wang, Z., Ramasinghe, S., Xu, C., Monteil, J., Bazzani, L., Ajanthan, T.: Learning visual hierarchies in hyperbolic space for image retrieval. In: CVPR (2025)

74. Wu, K., Peng, H., Zhou, Z., Xiao, B., Liu, M., Yuan, L., Xuan, H., Valenzuela, M., Chen, X.S., Wang, X., et al.: Tinyclip: Clip distillation via afinity mimicking and weight inheritance. In: ICCV (2023)

75. Wu, R., Chen, P., Shen, F., Zhao, S., Hui, Q., Gao, H., Lu, T., Liu, Z., Zhao, F., Wang, K., et al.: Himo-clip: Modeling semantic hierarchy and monotonicity in vision-language alignment. AAAI (2025)

76. Xie, S., Lingjing, L., Zheng, Y., Yao, Y., Tang, Z., Xing, E.P., Chen, G., Zhang, K.: Smartclip: Modular vision-language alignment with identification guarantees. In: CVPR (2025)

77. Yan, S., Liu, Z., Xu, L.: Hyp-uml: Hyperbolic image retrieval with uncertaintyaware metric learning. arXiv preprint arXiv:2310.08390 (2023)

78. Yang, M., Feng, A., Xiong, B., Liu, J., King, I., Ying, R.: Hyperbolic fine-tuning for large language models. NeurIPS (2025)

79. Zhang, B., Zhang, P., Dong, X., Zang, Y., Wang, J.: Long-clip: Unlocking the long-text capability of clip. In: ECCV (2024)

80. Zhang, B., Tao, J., Zeng, Z., He, N., Maatouk, A., Yang, M., Ying, R.: Parameter-eficient fine-tuning of llms with mixture of space experts. arXiv preprint arXiv:2602.14490 (2026)

81. Zhao, T., Zhang, T., Zhu, M., Shen, H., Lee, K., Lu, X., Yin, J.: Vl-checklist: Evaluating pre-trained vision-language models with objects, attributes and relations. arXiv preprint arXiv:2207.00221 (2022)

82. Zhao, Y., Jiang, B., Ding, Y., Wang, X., Tang, J., Luo, B.: Fine-grained vlm finetuning via latent hierarchical adapter learning. arXiv preprint arXiv:2508.11176 (2025)

83. Zhong, Y., Yang, J., Zhang, P., Li, C., Codella, N., Li, L.H., Zhou, L., Dai, X., Yuan, L., Li, Y., et al.: Regionclip: Region-based language-image pretraining. In: CVPR (2022)

84. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)