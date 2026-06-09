# Evaluating the Representation Space of Diffusion Models via Self-Supervised Principles

Xiao Li†, Yixuan Jia†, Zekai Zhang, Xiang Li, Lianghe Shi, Jinxin Zhou1, Zhihui Zhu1, Liyue Shen, Qing Qu‡

University of Michigan · 1Ohio State University

† Joint first author ‡ Corresponding author

## Abstract

Diffusion models have demonstrated remarkable generative capabilities and have also emerged as powerful self-supervised representation learners, yet the connection between these two abilities remains less explored. Drawing inspiration from self-supervised learning (SSL), we introduce a framework for jointly evaluating the representation and generation capabilities of diffusion models. Specifically, we decompose features into invariant and residual components and derive the Invariant Contamination Ratio (ICR), a Fisher-based metric that quantifies how residual variation contaminates invariant signal in feature space. We use this framework to analyze both discriminative and generative behavior of diffusion models. On the representation side, we find that invariance peaks at intermediate noise levels, which also yield the best downstream classification performance. On the generative side, we study how training transitions from genuine generalization to memorization in data-limited regimes, and show that ICR serves as a sensitive training-time indicator of early learning: increasing residual energy along Fisher directions marks the onset of memorization, detectable from training features alone without external evaluators or held-out test sets. Overall, our results show that diffusion models can be monitored from a self-supervised perspective through the geometry of their learned representations.

Keywords: Diffusion model, Representation learning, Self-supervised learning

Date: June 9, 2026

Correspondence: xlxiao@umich.edu

Resources: Project Website

![](images/4a7b62537bfb3d43cf547aa24b6161f7e6bd456a4b3a09aedfbcbcaacda916d8.jpg)

DeepThink Lab

https://deepthink-umich.github.io

![](images/bb21bee69c4a2e8f7e9b759ebdeead6c8ebd2344b77cb242505b55fccdbf9831.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Training data sample x₀~p_data"] --> B["Image: Diffusion feature extraction h(a(x₀)) = s(x₀) + ξ(a,x₀)"]
  C --> D["Output: Σₛ = Covₓ₀(s), Σξ := Covₓ₀,a(ξ)"]
  D --> E["Fisher SNR eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_d"]
  E --> F["ICR Score: 1/(1 + ½dΣᵢλᵢ)"]
  F --> G["Output: Σₛvᵢ = λᵢΣξvᵢ."]
```
</details>

![](images/7ba641ad97d8bff57fca3d0aba0e8d5903e1ea8585058c7731d4efb94bb01aa2.jpg)

<details>
<summary>text_image</summary>

Low ICR
Views cluster tightly,
means spread far
High ICR
Views scatter widely,
means bunch together
(b) Invariant mean features s Augmented view features ξ
</details>

![](images/a810656e707d646c727024374921bdd023ac49cba349adfcabeb82a7d7a1bbfb.jpg)

<details>
<summary>line chart</summary>

| Time step (σt) | ICR   | Classification Acc. |
| -------------- | ----- | ------------------- |
| 0.0            | 0.8   | 85                  |
| 0.02           | 0.85  | 83                  |
| 0.14           | 0.85  | 80                  |
| 0.59           | 0.7   | 65                  |
| 1.92           | 0.5   | 40                  |
</details>

![](images/9a2e8a4bdd539f7e880b8941a7c40df2906a8a67fa627f69da83ba3a6c5e81ba.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | ICR   | FID Score |
| --------------------- | ----- | --------- |
| 2.5m                  | 0.40  | 30        |
| 7.5m                  | 0.32  | 15        |
| 15m                   | 0.28  | 10        |
| 25m                   | 0.26  | 5         |
| 50m                   | 0.26  | 5         |
| 100m                  | 0.26  | 5         |
</details>

![](images/d87daaa30f3bd5213c4f58c1023ce60234ffbacec3e8e31fbb9c91044deb5738.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | ICR   | Memorization Ratio (0-1) |
| --------------------- | ----- | ------------------------ |
| 2.5m                  | 0.36  | 0.00                     |
| 5m                    | 0.34  | 0.00                     |
| 7.5m                  | 0.30  | 0.00                     |
| 10m                   | 0.29  | 0.00                     |
| 20m                   | 0.32  | 0.05                     |
| 30m                   | 0.34  | 0.10                     |
| 75m                   | 0.36  | 0.25                     |
| 200m                  | 0.38  | 0.35                     |
</details>

Figure 1: Overview of the ICR Framework. Each training image is augmented and passed through the diffusion feature extractor, decomposing representations into an invariant component ?? and a residual ??; their covariances define ICR (a–b). ICR serves as a unified diagnostic: it identifies the optimal noise level for classification tasks, tracks generative quality without sampling, and anticipates memorization onset during training (c).

## Contents

1 Introduction 3  
2 Problem Setup 5  
3 Representation-Level Evaluation Based on Self-Supervised Principles 5

3.1 What Makes a Good Representation in SSL? . 6  
3.2 A Geometric Framework for Representation Evaluation 6

4 ICR across Noise Levels: a Semantic Window for Representation Learning 8  
5 Invariance and Expansion: From Generalization to Memorization 9

5.1 ICR Tracks FID in the Data-Rich Regime . 10  
5.2 Early Learning in the Representation Space Under Limited Data . 11  
5.3 How Feature Expansion Differs in Data-Rich and Data-Limited Regimes 12

6 Conclusion 13

A Related Work 19  
B Additional Discussions & Experiments 20

B.1 Component Dynamics Across the Noise Schedule 20  
B.2 ICR and Classification Accuracy in the Data-limited Regime . . 21  
B.3 Discussion on the Alignment and Uniformity Metrics [45] 21  
B.4 Discussion on the Class Separation and Silhouette Score Metrics 22  
B.5 Connection to Neural Collapse 24  
B.6 Technical Details on Calculating ICR 25

C Experimental Details 27  
D Alignment between Optimal Test Loss and ICR 29

## 1 Introduction

In recent years, diffusion models [1, 2, 3] have achieved remarkable success in generative modeling, serving as the backbone for inverse problems [4, 5, 6, 7] and many prominent large-scale generative systems, such as Stable Diffusion, Flux, and Veo [2, 8, 9, 10, 11]. Beyond their generative capabilities, recent studies [12, 13, 14, 15, 16] have demonstrated the superior unsupervised representation learning abilities of diffusion models, where the diffusion representation is extracted from the bottleneck layer at certain timesteps of the learned denoiser. These works leverage diffusion representations for various downstream tasks, including classification, segmentation, and image correspondence, often achieving performance comparable to or even exceeding established self-supervised learning (SSL) methods. In parallel, regularizing diffusion representations using powerful self-supervised learners, such as DINOv2 [17] and MAE [18], can significantly improve the training efficiency and generation quality of diffusion models [19, 20]. This highlights the strong interplay between representation learning and generative modeling within these paradigms.

Despite these advances, the representation learning paradigms of diffusion models and traditional SSL remain quite different. Diffusion models are trained with a denoising objective, recovering clean signals from Gaussian corrupted inputs, whereas most SSL methods [21, 22, 23, 24, 17] are explicitly designed to enforce invariance to data augmentations while preserving a rich, diverse embedding space. The distinction between training objectives raises natural questions about diffusion representation spaces: to what extent do they implicitly capture the beneficial characteristics directly optimized in SSL, and how do these properties evolve across varying noise levels and throughout the learning dynamics?

Bridging diffusion representations with those studied in standard SSL can help clarify how diffusion training shapes features and how these features might be further improved to benefit diffusion models. Moreover, adopting a representation-centered viewpoint of diffusion models offers an intrinsic way to understand these systems: if diffusion models are indeed powerful representation learners, the properties of their learned representations should encode clear signatures of whether the model is capturing low-dimensional structure in image manifolds [25, 26] rather than merely overfitting to the idiosyncratic details of the training data.

Moreover, this perspective is particularly helpful for understanding and monitoring the generalizability of diffusion models during training. Evaluating how well a model goes beyond memorizing the training set is practically difficult, especially in data-limited regimes where recent work has documented a distinct “early learning” phase [27, 28, 29, 30, 31]: the model first learns to generalize and then gradually starts to memorize individual samples. In this regard, prior work [32] showed that standard generation metrics such as Fréchet distance (FID) are not reliable memorization detectors, while exhaustive nearest neighbor tests [33, 34] rely on large numbers of generated samples and are often expensive to run. By importing insights from SSL and evaluating diffusion models through the geometry of their learned representations, we obtain intrinsic, training time signals that track the quality of the representation space. These signals offer a practical way to gauge generalized generation quality as training progresses and to identify good stopping points before overfitting dominates.

Summary of contributions. In this work we revisit diffusion models from a self-supervised representation learning perspective. Guided by classic SSL principles, we focus on two properties of internal features: representation invariance, reflecting the stability of shared content across random perturbations, and representation expansion, reflecting how well the features spread out in the available embedding space. We capture these two properties with a new intrinsic metric, the Invariant Contamination Ratio (ICR), which measures how much augmentation and noise sensitive variation contaminates the stable part of the representation space. To construct ICR, we introduce an invariance–residual decomposition of diffusion features that separates perturbation invariant structure from residual variation. Because ICR is label-free and can be computed entirely from training features, it can be monitored throughout training and across noise levels. Empirically, we find that ICR provides a reliable proxy for noise level dependent downstream representation performance and cleanly separates generalization from memorization during training in data-limited regimes. In summary, our main contributions are as follows:

![](images/2c7c82fd10f4d0a6e78a57492db6daf53874b68c9a62eda3a46e09704c8b6931.jpg)

<details>
<summary>text_image</summary>

t-SNE 1
Far2
Far1
NN1
NN2
Ref
Nearest neighbors among base images (using s)
Ref. Image
Nearest
Farthest
view=1
view=1
view=1
view=1
view=1
view=9
Nearest neighbors across all views (using ξ)
</details>

Figure 2: Nearest neighbors of invariant and residual components on ImageNet 64 × 64. We use a pretrained EDM diffusion model and, for each ImageNet training image, sample 9 augmented views and extract bottleneck representations. Left: t-SNE visualization of invariant representations ??; views of the same base image form tight clusters, and the marked nearest and farthest examples reflect their relative positions in this space. Top right: For a reference image (left), nearest and farthest neighbors among the base images are selected using cosine similarity on ??. Bottom right: For a reference augmented view (left), nearest and farthest neighbors across all augmented views are selected using cosine similarity on ??. Neighbors based on ?? tend to be semantically similar to the query, whereas neighbors based on ?? often appear semantically unrelated.

• A representation-based evaluation metric for diffusion models. We introduce an invariance–residual decomposition of diffusion representations and from it define the Invariant Contamination Ratio (ICR), a single label-free scalar that measures how much of the representation space is occupied by augmentation and noise-sensitive variation rather than stable structure. ICR can be computed solely from training features without labels or external networks.  
• Finding the optimal representation across noise levels. On standard image benchmarks, we show that the diffusion noise schedule admits an intermediate semantic window where ICR is minimized and linear classification accuracy is maximized. This gives a simple SSL-based rule to select noise scales that yield the strongest diffusion features for downstream tasks.  
• Tracking generalization and memorization in training dynamics. By following ICR over training, we observe distinct learning phases: in data-rich regimes it decreases steadily with improving generative quality, while in data-limited regimes it exhibits a U-shaped early learning pattern that precedes the rise of memorization. Thus ICR provides a practical, label-free early stopping signal, and its decomposition reveals that feature expansion during training is driven by invariant structure when data are abundant and by residual variation when data are scarce.

## 2 Problem Setup

Preliminaries on diffusion models. Diffusion models define a forward process that gradually perturbs data $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ toward a Gaussian distribution via the stochastic differential equation

$$
\mathrm{d} \boldsymbol {x} _ {t} = f (t) \boldsymbol {x} _ {t} \mathrm{d} t + g (t) \mathrm{d} \boldsymbol {w} _ {t}, \quad t \in [ 0, 1 ],
$$

where $f$ and $g$ are scalar functions and $\{ w _ { t } \}$ is a standard Wiener process. Let $p _ { t }$ denote the ??density of $x _ { t }$ ??and note that $p _ { 0 } = p _ { \mathrm { d a t a } }$ ?? ????. For simplicity, we consider the variance preserving setting ${ \boldsymbol { x } } _ { t } = { \boldsymbol { x } } _ { 0 } + { \boldsymbol { \sigma } } _ { t } { \boldsymbol { \epsilon } }$ with $\epsilon \sim \mathcal { N } ( 0 , I )$ ??. The reverse time process that maps noise back to data uses the score $\nabla \log { p _ { t } ( { \boldsymbol { x } } _ { t } ) }$ , and is given by the reverse SDE [35]

$$
\mathrm{d} \pmb {x} _ {t} = \big (f (t) \pmb {x} _ {t} - g ^ {2} (t) \nabla \log p _ {t} (\pmb {x} _ {t}) \big) \mathrm{d} t + g (t) \mathrm{d} \bar {\pmb {w}} _ {t},
$$

where $\{ \bar { \boldsymbol { w } } _ { t } \}$ is an independent Wiener process. This enables diffusion models to generate new ??samples from the underlying data distribution $p _ { \mathrm { d a t a } }$ by initializing from pure Gaussian noise and iteratively denoising via the score function.

Training loss of diffusion models. Modern diffusion models are typically trained to approximate the score function ∇ log $p _ { t } ( \pmb { x } _ { t } )$ . By Tweedie’s formula [36],

$$
\mathbb {E} \left[ \boldsymbol {x} _ {0} \mid \boldsymbol {x} _ {t} \right] = \boldsymbol {x} _ {t} + \sigma_ {t} ^ {2} \nabla \log p _ {t} (\boldsymbol {x} _ {t}), \tag {1}
$$

this is equivalent to learning the posterior mean $\mathbb { E } [ x _ { 0 } \mid x _ { t } ]$ via a denoising autoencoder $x _ { \theta } ( x _ { t } , t )$ ??[15, 13, 37]. Concretely, we minimize the weighted denoising loss

$$
\min _ {\boldsymbol {\theta}} \sum_ {i = 1} ^ {N} \int_ {0} ^ {1} \lambda_ {t} \mathbb {E} _ {\epsilon} \left[ \left\| x _ {\boldsymbol {\theta}} (x _ {t} ^ {(i)}, t) - x _ {0} ^ {(i)} \right\| ^ {2} \right] \mathrm{d} t, \tag {2}
$$

$\boldsymbol { x } _ { 0 } ^ { ( i ) } \stackrel { i . i . d . } { \sim } p _ { \mathrm { d a t a } }$ for $i = 1 , \ldots , N$ and $\lambda _ { t }$ weights different noise levels.

Layer selection for extracting representations. We freeze the diffusion backbone and extract representations from the layer that gives the strongest downstream performance according to [13]. In practice, this corresponds to a layer near the bottleneck of the U-Net architecture [38, 39] and the middle transformer block of SiT [40].

## 3 Representation-Level Evaluation Based on Self-Supervised Principles

As discussed in the introduction, recent work shows that diffusion models can act as strong self supervised representation learners, supporting competitive performance across various downstream tasks [12, 13, 15, 41]. However, their training paradigm differs markedly from standard self-supervised learning (SSL): while most SSL methods rely on explicit contrastive or predictive objectives to shape the embedding space [22, 42, 43, 17], diffusion models are trained with a denoising objective that reconstructs clean signals from Gaussian corrupted inputs. This difference raises a fundamental question:

To what extent does the denoising objective naturally satisfy the geometric properties of "good" representations sought in the SSL literature?

In diffusion models, the internal representations are high-dimensional and evolve across different noise scales, making it non-trivial to separate stable information from idiosyncratic variation. To evaluate this, we propose an evaluation metric rooted in the principles of modern image-based SSL.

## 3.1 What Makes a Good Representation in SSL?

Modern SSL methods [44, 45, 22, 21, 24, 17] are often built around two complementary principles:

• Representation invariance: Representations extracted from different stochastic perturbations of a sample are encouraged to remain stable in the embedding space.  
• Representation expansion: Representations should maintain a rich, spread-out structure across different images. The embedding distribution should avoid dimensional collapse and utilize many directions in the representation space to preserve unique image identities.

In the rest of this work, we use these principles to track how diffusion representations evolve across noise levels and training. We introduce a simple decomposition that splits each representation into a perturbation invariant component, stable across noisy and augmented views, and a residual component that captures variation induced by these perturbations.

## 3.2 A Geometric Framework for Representation Evaluation

Guided by the SSL principles in Section 3.1, we seek a representation-level metric that can be monitored across noise levels and training, and that reflects how much of the active representation space is devoted to perturbation invariant structure. Informally, we want this diagnostic to (i) measure relative invariance rather than an absolute distance scale, (ii) be robust to overall representation expansion during training, and (iii) remain label-free and efficiently computable from representations. A natural starting point is the Alignment and Uniformity criteria [45], which have been highly successful in characterizing contrastive SSL encoders. However, Alignment is an absolute squared distance between two augmented views of the same image and grows when representations take more directions, so it can increase even when the representation becomes more semantically stable, while Uniformity only measures how spread out representations are and does not distinguish invariant structure from augmentation-sensitive noise. (see Appendix B.3 for details).

These limitations motivate a different construction that explicitly separates invariant information from view-specific variation. Instead of working directly with distances between raw representations, we first decompose the representation space into stable and varying components, then leverage the spectral properties of these components to derive a summary metric for representations.

Invariant and residual decomposition in representation space. For each training image $x _ { 0 } \sim$ $p _ { \mathrm { d a t a } } ,$ let $a \sim { \mathcal { A } }$ denote a random perturbation encompassing both standard semantics-preserving ?? ??transformations [22, 42] and the additive Gaussian noise $\bar { \epsilon } \sim { \cal N } ( 0 , \sigma _ { t } ^ { 2 } I )$ injected by the diffusion objective1. Let $\pmb { h } ( \cdot ) \in \mathbb { R } ^ { d }$ , ??be the representation extracted from a fixed layer of the diffusion model, and consider the random representation $h ( a ( \boldsymbol { x } _ { 0 } ) )$ induced by the stochasticity of the perturbation . ??We decompose this representation into its conditional mean and a residual:

$$
\begin{array}{l} \boldsymbol {s} \left(\boldsymbol {x} _ {0}\right) := \mathbb {E} _ {a} \left[ \boldsymbol {h} \left(a \left(\boldsymbol {x} _ {0}\right)\right) \mid \boldsymbol {x} _ {0} \right], \\ \xi (a, x _ {0}) := \mathbf {h} (a (x _ {0})) - s (x _ {0}), \tag {3} \\ \end{array}
$$

![](images/ff2477952288d603f5e88f9feddd4e9b2a538233c8fd21cc2de00dd6ee4aa0da.jpg)

<details>
<summary>line chart</summary>

| Time step (σt) | ICR   | Classification Acc. (%) |
| -------------- | ----- | ---------------------- |
| 0.0            | 0.78  | 40                     |
| 0.02           | 0.82  | 45                     |
| 0.14           | 0.81  | 50                     |
| 0.59           | 0.65  | 65                     |
| 1.92           | 0.85  | 90                     |
</details>

(a) CIFAR10

![](images/56ebd0a7026a2030812a80ad93c21be5e30b70959705ddf824eeccd5d6c05242.jpg)

<details>
<summary>line chart</summary>

| Time step (σt) | ICR   | Classification Acc. (%) |
| -------------- | ----- | ----------------------- |
| 0.0            | 0.7   | 20                      |
| 0.02           | 0.8   | 25                      |
| 0.14           | 0.8   | 30                      |
| 0.59           | 0.6   | 40                      |
| 1.92           | 0.4   | 60                      |
</details>

(b) CIFAR100

![](images/73e767f3d42a44fe54b0512223b537a60af877ba7d09a2079be0ced15cc400b3.jpg)

<details>
<summary>line chart</summary>

| Time step (σt) | ICR   | Classification Acc. (%) |
| -------------- | ----- | ---------------------- |
| 0.06           | 0.4   | 4                      |
| 0.15           | 0.35  | 6                      |
| 0.25           | 0.35  | 8                      |
| 0.35           | 0.35  | 10                     |
| 0.5            | 0.4   | 12                     |
| 0.7            | 10    | 4                      |
| 0.9            | 12    | 0                      |
</details>

(c) ImageNet  
Figure 3: Correspondence between ICR and classification accuracy across noise levels. For each pretrained backbone (EDM [39] on CIFAR10 and CIFAR100, SiT-XL/2 [40] on ImageNet), we extract bottleneck representations at multiple noise levels $\sigma _ { t }$ . At each $\sigma _ { t } ,$ we estimate ICR (blue) using a ?? ??subset of training representations and train a classifier on the full training representations, reporting accuracy on the test set (slate). Across datasets, the noise levels that minimize ICR coincide with those that maximize classification accuracy.

yielding the additive form $h ( a ( \boldsymbol { x } _ { 0 } ) ) = s ( \boldsymbol { x } _ { 0 } ) + \xi ( a , \boldsymbol { x } _ { 0 } )$ . In this decomposition, $\pmb { s } ( \pmb { x } _ { 0 } )$ is the invariant ?? ??,component, which filters out transient variations to capture attributes resilient to corruption. Conversely, $\xi ( { a } , { x } _ { 0 } )$ is the residual component capturing the specific idiosyncratic variations of a single noisy view.

Figure 2 provides empirical support for this interpretation: multiple perturbed views of a single image form a cluster in the representation space centered at $s ( x _ { 0 } )$ . Notably, nearest-neighbor searches based on $\pmb { s } ( \pmb { x } _ { 0 } )$ retrieve semantically related images, whereas neighbors based solely on the residual $\xi ( { a } , { x } _ { 0 } )$ appear visually unrelated and lack shared category structure. Since $\mathbb { E } [ \xi \mid x _ { 0 } ] = \mathbf { 0 } ,$ ??,the law of total covariance implies that the total representation covariance $\Sigma _ { h }$ decomposes into two components:

$$
\pmb {\Sigma} _ {h} = \pmb {\Sigma} _ {s} + \pmb {\Sigma} _ {\xi},
$$

where $\Sigma _ { s } : = \operatorname { C o v } _ { x _ { 0 } } ( s ) , \quad \Sigma _ { \xi } : = \operatorname { C o v } _ { x _ { 0 } , a } ( \xi ) .$

This decomposition allows us to translate core SSL principles into geometric properties of the representation space:

• Representation expansion refers to the growth of the total covariance $\pmb { \Sigma } _ { h } = \pmb { \Sigma } _ { s } + \pmb { \Sigma } _ { \xi } ,$ , which characℎ ??terizes how features spread in the representation space across data samples. More specifically, it describes the extent to which the feature covariance occupies multiple directions, i.e., whether the representation is low-rank or broadly distributed in the embedding space. In practice, we track its trace, $\operatorname { T r } ( \Sigma _ { h } )$ , which measures the total variance (energy) of the representation and reflects ℎhow much of the high-dimensional feature space the model utilizes for distinguishing between different images and their perturbed views.  
• Representation invariance is measured by the relative dominance of $\Sigma _ { s }$ over $\Sigma _ { \xi }$ . Unlike traditional ??SSL, diffusion models require a noise-dependent balance: at low noise levels, a larger residual $\Sigma _ { \xi }$ is necessary for reconstructing fine-grained pixel details, whereas at intermediate noise levels, a higher degree of invariance is desired to capture stable semantic structures.

Fisher directions and invariant signal-to-noise ratios. Given the invariant and residual covariances $\left( \Sigma _ { s } , \Sigma _ { \xi } \right)$ , we consider the generalized eigenproblem [46]:

$$
\pmb {\Sigma} _ {s} \pmb {v} _ {i} = \lambda_ {i} \pmb {\Sigma} _ {\xi} \pmb {v} _ {i}, \qquad \lambda_ {1} \geq \lambda_ {2} \geq \dots \geq \lambda_ {d} \geq 0,
$$

with eigenvectors $v _ { i }$ orthogonal in the $\Sigma _ { \xi }$ inner product. In practice, these matrices are estimated ??empirically across the training dataset, where $\Sigma _ { \xi }$ aggregates residual variations within each sample and $\Sigma _ { s }$ captures the spread of invariant identities across all samples. Each $v _ { i }$ represents a direc-?? ??tion in representation space that is optimized to maximize the ratio of invariant signal energy to residual variation. Specifically, the corresponding eigenvalue $\lambda _ { i }$ admits the Rayleigh quotient [47] representation2:

$$
\lambda_ {i} = \max _ {\boldsymbol {v} \neq 0, \boldsymbol {v} \perp \boldsymbol {\Sigma} _ {\xi} \left\{\boldsymbol {v} _ {1}, \dots , \boldsymbol {v} _ {i - 1} \right\}} \frac {\boldsymbol {v} ^ {\top} \boldsymbol {\Sigma} _ {s} \boldsymbol {v}}{\boldsymbol {v} ^ {\top} \boldsymbol {\Sigma} _ {\xi} \boldsymbol {v}}. \tag {4}
$$

In this framework, a generalized eigenvalue $\lambda _ { i }$ measures the invariant signal-to-noise ratio along the corresponding Fisher direction [48] $v _ { i } \colon$ ?? it compares the variance of the stable invariant component ${ \boldsymbol { v } } ^ { \top } { \boldsymbol { \Sigma } } _ { s } { \boldsymbol { v } }$ to the residual variance ${ \pmb v } ^ { \top } { \pmb \Sigma } _ { \xi } { \pmb v }$ in that specific direction. This follows the same generalized eigenstructure as classical Fisher Linear Discriminant Analysis [48], where $\Sigma _ { s }$ and $\Sigma _ { \xi }$ play roles ??analogous to between-class and within-class covariances, respectively, with each individual image effectively acting as its own class.

Invariant Contamination Ratio (ICR). The collection of generalized eigenvalues $\{ \lambda _ { 1 } , \ldots , \lambda _ { d } \}$ , . . . , ??provides a directional profile of how strongly invariant structures dominate residual variations. To summarize this into a single, trackable scalar that quantifies the health of the representation, we define the Invariant Contamination Ratio (ICR):

$$
\mathrm{ICR} := \frac {1}{1 + \frac {1}{d} \sum_ {i = 1} ^ {d} \lambda_ {i}}. \tag {5}
$$

The term $\textstyle { \frac { 1 } { d } } \sum \lambda _ { i }$ represents the average invariant signal-to-noise ratio across all Fisher directions. ?? ??When the invariant component ?? dominates the residual $\xi$ across the majority of directions, this average is large, resulting in a low ICR. Conversely, as residual variation (or “contamination”) increases and begins to occupy a substantial portion of the representation space, the ICR approaches $1 . ^ { 3 }$ We note that in practice, we estimate $\Sigma _ { s }$ and $\Sigma _ { \xi }$ from as few as two augmentations per image ??and a subset of training representations; implementation details are given in Section B.6.

## 4 ICR across Noise Levels: a Semantic Window for Representation Learning

The denoising objective of diffusion models is inherently multiscale: for each clean image $x _ { 0 }$ the model sees a family of corrupted inputs $\{ x _ { t } \}$ indexed by the noise level $\sigma _ { t } ,$ and thus induces a family of representations $h _ { t } ( a ( \boldsymbol { x } _ { 0 } ) )$ ?? ??). In this section we use ICR to answer two questions:

• How does relative invariance vary across the diffusion noise schedule?

• Can this internal measure predict which noise levels yield the best downstream representations?

![](images/6aec7454360adc805d91ea06871e9d6d1f38d4321a2077fb2fad94df380678c0.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | ICR   | FID Score |
| --------------------- | ----- | --------- |
| 2.5m                  | 0.40  | 30        |
| 7.5m                  | 0.36  | 15        |
| 15m                   | 0.32  | 8         |
| 25m                   | 0.29  | 5         |
| 50m                   | 0.27  | 3         |
| 100m                  | 0.25  | 2         |
</details>

(a) CIFAR10, EDM,  = 50K

![](images/c077e998de7daf7044739d5708910f300562bea59a5d06135de53c9989ca13e3.jpg)

<details>
<summary>line chart</summary>

| Training Iteration (batches) | ICR    | FID Score |
| ---------------------------- | ------ | --------- |
| 25k                          | 0.47   | 180       |
| 75k                          | 0.46   | 100       |
| 125k                         | 0.44   | 60        |
| 175k                         | 0.43   | 40        |
| 225k                         | 0.425  | 35        |
| 275k                         | 0.42   | 30        |
</details>

(b) ImageNet-256, SiT-B/2,  = 1 28M  
Figure 4: ICR and FID dynamics in data rich diffusion training. We monitor generative performance (via FID) and ICR for EDM and SiT-B/2 based diffusion models trained on the full CIFAR10 and ImageNet datasets as training progresses. Both ICR (blue) and FID (brown) exhibit a monotonically decreasing trend, indicating improving internal representation invariance and sample quality over the course of training.

ICR predicts classification performance and reveals a semantic window. We start from pretrained diffusion backbones on CIFAR datasets [49] and ImageNet [50]. For each noise level $\sigma _ { t } ,$ we estimate $\operatorname { I C R } ( \sigma _ { t } )$ ?? using a subset of training representations extracted from inputs corrupted at noise level $\sigma _ { t } ,$ ?? ??and train a linear classifier on the full training representations at the same ?? . Figure 3 plots ICR and test accuracy as functions of $\sigma _ { t }$ .

??Across all datasets we observe a striking alignment: the ICR curve is U-shaped and attains a clear minimum at an intermediate noise level, while classification accuracy peaks in exactly the same range. We refer to this range as a semantic window: at very low noise, representations are too tied to fine-grained, augmentation-specific details; at very high noise, representations collapse toward noise; in between, relative invariance is strongest and the model uses its representation space in a way that is most useful for downstream tasks. Importantly, ICR is computed in a label-free way from training representations alone, yet it accurately predicts which noise scales will deliver the best linear probe performance. We further verify the robustness of this observation in the data-limited setting in Section B.2 and observe a consistent trend.

These observations are consistent with prior work [51, 26] showing that diffusion sampling exhibits a coarse-to-fine transition, with different noise levels capturing structure at different granularities. They also suggest that our augmentation-based evaluation developed from self-supervised principles extend naturally to diffusion models, providing a simple bridge between these two frameworks.

Motivated by this semantic window, in the next section we fix a representative intermediate noise level $\sigma ^ { \star }$ near the ICR minimum and study how the representation space evolves over the course of training at this scale, relating the dynamics of invariance to generative quality and memorization.

## 5 Invariance and Expansion: From Generalization to Memorization

In the previous section we examined how ICR varies across the diffusion noise schedule and identified an intermediate semantic window where downstream representations are strongest. We now fix a representative noise scale $\sigma ^ { \star }$ in this window and track how the representation space evolves over the course of training.

Our analysis proceeds in three steps. First, in the data-rich regime, we show that ICR closely tracks FID during training, linking improvements in generative quality to changes in the internal feature geometry. Second, in the data-limited regime, we observe an early learning phenomenon in the representation space: ICR exhibits a clear U-shaped trajectory and its minimum aligns with the onset of memorization. Finally, we examine trace-level statistics of the invariant and residual covariances to understand how feature expansion is allocated between invariant and residual components in the two regimes.

![](images/ef17cd85839a4484e8ba9322dd4ca492dc364a9188404d4fe128b14da0d1b68c.jpg)

<details>
<summary>line chart</summary>

| Training Iteration | ICR   | Memorization Ratio |
| ------------------ | ----- | ------------------ |
| 2.5m               | 0.36  | 0.00               |
| 5m                 | 0.34  | 0.00               |
| 7.5m               | 0.30  | 0.00               |
| 10m                | 0.29  | 0.00               |
| 20m                | 0.32  | 0.05               |
| 30m                | 0.34  | 0.10               |
| 75m                | 0.36  | 0.25               |
| 200m               | 0.37  | 0.37               |
</details>

Figure 5: ICR as an early signal of memorization in data limited diffusion training (CIFAR10). We evaluate an EDM-based diffusion model trained on a subset of CIFAR10 (4096 images). Left: ICR (blue) follows a clear U shaped trajectory as training progresses, while the memorization ratio (red) remains near zero early on and begins to rise only after the ICR minimum. Right: Qualitative inspection at 2.5M, 8.5M, and 200M training images seen. Generated samples (top) and their nearest training neighbors (bottom) show that visual quality initially improves, but eventually the model begins to memorize individual training images, in line with the ICR curve.

## 5.1 ICR Tracks FID in the Data-Rich Regime

We first consider the data-rich regime, where the diffusion model is trained on the full training set. At each checkpoint we compute ICR from training features and evaluate the Fréchet Inception Distance (FID) between generated samples and the real data distribution. Figure 4 illustrates the resulting trajectories as functions of training progress.

In this setting, ICR and FID exhibit a strong positive correlation: both demonstrate a monotonically decreasing trend. This alignment reflects an intuitive connection between generative quality and representation geometry. FID [52] (and related Fréchet-based metrics such as $\mathrm { F D } _ { \mathrm { D I N O v } 2 } )$ measures the distance between generated and real distributions in an external semantic feature space [53, 17], quantifying how well the model captures the data manifold. In contrast, ICR quantifies the “purity" of the internal diffusion representation by measuring the ratio of augmentation-sensitive residual energy to stable invariant signal.

The simultaneous improvement in FID and ICR suggests that the improvement in generative ability is directly reflected in the refinement of the representation space. As the model better approximates $p _ { \mathrm { d a t a } } ,$ , its internal features shift from capturing transient, view-specific noise toward a ??stable, low-dimensional image structure. In our framework, this manifests as a higher signal-to-noise ratio in the Fisher directions, where the invariant component $\Sigma _ { s }$ increasingly dominates the residual variation $\Sigma _ { \xi }$ .

![](images/922705ff7cf365a15e53d78c9a3b7a84107bf8a5547600c2e5bf9ef9658753d3.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | ICR    | Memorization Ratio |
| --------------------- | ------ | ------------------ |
| 0.6m                  | 0.55   | 0.0                |
| 6m                    | 0.45   | 0.0                |
| 9m                    | 0.44   | 0.0                |
| 15m                   | 0.48   | 0.4                |
| 24m                   | 0.49   | 0.5                |
| 30m                   | 0.50   | 0.6                |
| 36m                   | 0.51   | 0.65               |
| 42m                   | 0.51   | 0.7                |
</details>

(a) ImageNet-64, EDM,  = 10K

![](images/d44608c9b910d711ba511c6a941f4a2c13ba07b9a5b726558933fab63816ea2a.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (batches) | ICR    | Memorization Ratio |
| ------------------------ | ------ | ------------------ |
| 25k                      | 0.472  | 0.0                |
| 75k                      | 0.440  | 0.0                |
| 125k                     | 0.460  | 0.6                |
| 175k                     | 0.465  | 0.8                |
| 225k                     | 0.468  | 0.8                |
| 275k                     | 0.469  | 0.8                |
</details>

(b) ImageNet-256, SiT-B/2,  = 20K

Figure 6: ICR dynamics consistently anticipate memorization across large-scale datasets. We repeat the analysis of ICR (blue) and memorization ratio (red) on ImageNet in data limited settings. (a) EDM trained on a 10K image subset of ImageNet 64 × 64. (b) SiT-B/2 diffusion model trained on a 20K image subset of ImageNet 256 × 256. In both cases, ICR dips and then rises before the memorization ratio increases, mirroring the behavior on CIFAR10 experiments (Figure 5).  
![](images/1fd92e774c433689d73da37ef112eb2ed99aae47ca225d416360669c3738d384.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Training Iter = 0.6M imgs, ICR = 0.546"] --> B["Nearest neighbors in s (Example 1)"]
  C["Training Iter = 7M imgs, ICR = 0.438"] --> D["Nearest neighbors in s (Example 1)"]
  E["Training Iter = 50M imgs, ICR = 0.504"] --> F["Nearest neighbors in s (Example 1)"]
  G["Ref. Image"] --> H["Farthest"]
  I["Nearest"] --> J["Farthest"]
  K["Ref. Image"] --> L["Farthest"]
  M["Nearest"] --> N["Farthest"]
  O["Nearest"] --> P["Farthest"]
```
</details>

Figure 7: Nearest neighbors of invariant components throughout limited data training. We visualize nearest neighbors in ?? as in Figure 2. The neighbors qualitatively track ICR and the model’s generalization: in the first row (early training, near initialization), ICR is large and neighbors are not semantically meaningful; in the second row (intermediate training), ICR is smallest and neighbors are reasonable; in the third row (severe overfitting), ICR increases again and neighbor quality degrades.

## 5.2 Early Learning in the Representation Space Under Limited Data

In this subsection, we investigate representation dynamics in the limited data regime. In this setting, recent studies [27, 28, 30, 31] have demonstrated an early learning phenomenon: image generation quality first improves and the model generalizes well in an initial phase of training, before eventually deteriorating as the model begins to memorize.

To examine whether the representation space undergoes a similar early learning trajectory in data-limited settings, we track ICR and a memorization ratio [33, 34] across training. Concretely, we train an EDM [39] model on a subset of CIFAR10 with  = 4096 images, an EDM on a 10K ??subset of ImageNet 64 × 64, and a SiT-B/2 [40] model on a 20K subset of ImageNet 256 × 256, and for each setting report ICR computed on training features together with the memorization ratio.4

In these experiments, as reported in Figures 5 and 6, ICR follows a clear U-shaped curve, in sharp contrast to the data-rich case in Figure 4. This indicates that feature invariance in the limited data regime also exhibits an early learning pattern: it improves during the initial phase of training and then degrades as training continues.

![](images/a67a3eb829746c3ee40ab4c67fbca422a1804c97dba18d936b732591251eec1f.jpg)

<details>
<summary>stacked bar chart</summary>

| Training Iter (imgs) | Tr(Σ_S) | Tr(Σ_ξ) |
|---|---|---|
| 2.5m | 0.42 | 0.13 |
| 7.5m | 0.58 | 0.11 |
| 15m | 0.69 | 0.17 |
| 25m | 0.71 | 0.18 |
| 50m | 0.73 | 0.19 |
| 100m | 0.73 | 0.20 |
| 150m | 0.73 | 0.21 |
</details>

(a) CIFAR10, EDM,  = 4096

![](images/10b61fee828a7dd4c739da5309a03f88c42dd4ab6b8ea081f05f96648ca2b12c.jpg)

<details>
<summary>stacked bar chart</summary>

| Training Iter (imgs) | Tr(Σ_S) | Tr(Σ_ξ) |
|---|---|---|
| 2.5m | 0.39 | 0.11 |
| 4.5m | 0.43 | 0.12 |
| 7.5m | 0.47 | 0.13 |
| 10.5m | 0.53 | 0.14 |
| 15m | 0.61 | 0.15 |
| 20m | 0.64 | 0.16 |
| 25m | 0.67 | 0.17 |
| 50m | 0.72 | 0.18 |
| 100m | 0.75 | 0.19 |
| 150m | 0.77 | 0.20 |
</details>

(b) CIFAR10, EDM,  = 50K  
Figure 8: How feature expansion differs in data-limited and data-rich diffusion training. We train two EDM-based diffusion models on CIFAR10 with different training set sizes and track the traces of the invariant and residual covariances over training (as labeled).

We moreover observe that the memorization ratio remains essentially zero around the ICR minimum and begins to increase only afterward. This suggests that the onset of memorization is reflected directly in the representation space: beyond the point where ICR is minimized, the model increasingly fits sample-specific idiosyncrasies [54] rather than shared, stable semantic structure.

This transition in the representation is also visible in Figure 7, where we use ?? to find nearest neighbors across the training trajectory in the data-limited case. During the middle phase of training, when ICR is small, the nearest neighbors are semantically close to the query, whereas in the early under-trained phase and the late heavily memorizing phase, where ICR is larger, the retrieved neighbors are noticeably less meaningful.

Finally, we note that the ICR values reported here are computed purely from training features and do not require generation or an external evaluation network. Taken together with the previous subsection, this suggests that ICR can serve as a reliable “generalized” generation metric that is monitorable during training without sampling. In the data-rich regime, it closely tracks improvements in generative quality in a way that parallels FID. In the data limited regime, where prior work has pointed out that FID is not entirely trustworthy for detecting memorization [32], ICR can act as a coarse, easy-to-monitor early stopping signal, complementary to standard generation-based metrics such as the memorization ratio [33].

## 5.3 How Feature Expansion Differs in Data-Rich and Data-Limited Regimes

In the previous subsections, we highlighted that the relative feature invariance, as measured by ICR, follows very different trajectories in the data-rich and data-limited regimes. However, ICR is a relative quantity: it summarizes how much residual variance contaminates invariant directions, but does not reveal how the total representation energy evolves. In particular, when ICR increases in the late stage of training under limited data, it is not clear whether this reflects an overall shrinkage of the representation space, a reallocation of variance from invariant to residual components, or some combination of both. To disentangle these possibilities, we examine the traces of the invariant and residual covariances over training.

In Figure 8, we train two EDM-based diffusion models with different training set sizes to compare a data-limited setting (4096 images) with a data-abundant setting (50K images). Across both settings, the total feature energy, measured by $\mathrm { T r } ( \pmb { \Sigma } _ { s } ) + \mathrm { T r } ( \pmb { \Sigma } _ { \xi } )$ , increases steadily over training. The ??regimes differ, however, in how this growth is allocated between invariant and residual components.

In the data-abundant case, $\mathrm { T r } ( \Sigma _ { s } )$ continues to increase throughout training, while $\operatorname { T r } ( \pmb { \Sigma } _ { \xi } )$ grows only ??mildly, indicating that additional feature capacity is predominantly devoted to invariant structure. In the data limited case, $\mathrm { T r } ( \Sigma _ { s } )$ rises initially but then saturates, whereas $\operatorname { T r } ( \pmb { \Sigma } _ { \xi } )$ continues to grow. ??This suggests that, once the limited semantic structure in the dataset has been largely extracted, further feature expansion is dominated by augmentation-sensitive residual variability. Consistent with this picture, ICR decreases with training in the data-abundant regime but exhibits a U-shaped trajectory under limited data, reflecting the late stage shift from invariant to residual energy.

## 6 Conclusion

In this work we revisit diffusion models from a self-supervised representation learning perspective. We introduce an invariance–residual decomposition of diffusion representations and the Invariant Contamination Ratio (ICR), a label-free metric that measures how much augmentation and noisesensitive variation contaminates stable structure in the feature space. Using this framework, we demonstrate that diffusion noise levels admit a semantic window where ICR is minimized and downstream linear classification performance is maximized, providing a simple SSL-based way to identify the most informative denoising scales. Tracking ICR over training further reveals distinct learning phases: in data-rich regimes it decreases in tandem with improvements in generative quality, while in data-limited regimes, it exhibits an early learning pattern that anticipates the onset of memorization. These findings suggest that diffusion models can be monitored and evaluated through their own representation space, providing intrinsic training-time signals that complement conventional generation-based metrics.

## Acknowledgment

XL, YJ, XL, LS, ZZ (UM) and QQ acknowledge support from NSF CAREER CCF-2143904, NSF CCF-2212066, NSF CCF-2212326, NSF IIS 2402950, ONR N000142512339, and Google Research Scholar and Google TPU Award. LS and QQ acknowledge support from DARPA HR00112520042. JZ and ZZ (OSU) acknowledge funding support from NSF IIS 2312840 and IIS 2402952. LS acknowledges funding support from NSF IIS 2435746. We also thank all the anonymous reviewers for their valuable suggestions and fruitful discussions.

## References

[1] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International conference on machine learning, pp. 2256–2265, pmlr, 2015.  
[2] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.  
[3] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021.  
[4] I. Alkhouri, S. Liang, R. Wang, Q. Qu, and S. Ravishankar, “Diffusion-based adversarial purification for robust deep mri reconstruction,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 12841–12845, IEEE, 2024.  
[5] Y. Jia, S. Chen, Y. Pan, X. Li, L. Shi, C. Jung, H. Yuan, I. Alkhouri, Y. C. Wu, S. Ravishankar, et al., “Forcingdas: Unified and robust data assimilation via diffusion forcing,” arXiv preprint arXiv:2605.14285, 2026.  
[6] B. Song, S. M. Kwon, Z. Zhang, X. Hu, Q. Qu, and L. Shen, “Solving inverse problems with latent diffusion models via hard data consistency,” in International Conference on Learning Representations, vol. 2024, pp. 7624–7654, 2024.  
[7] X. Li, S. M. Kwon, S. Liang, I. R. Alkhouri, S. Ravishankar, and Q. Qu, “Decoupled data consistency with diffusion purification for image restoration,” arXiv preprint arXiv:2403.06054, 2024.  
[8] J. L. Watson, D. Juergens, N. R. Bennett, B. L. Trippe, J. Yim, H. E. Eisenach, W. Ahern, A. J. Borst, R. J. Ragotte, L. F. Milles, et al., “De novo design of protein structure and function with rfdiffusion,” Nature, 2023.  
[9] A. Lou, C. Meng, and S. Ermon, “Discrete diffusion modeling by estimating the ratios of the data distribution,” in International Conference on Machine Learning, pp. 32819–32848, PMLR, 2024.  
[10] B. F. Labs, S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, S. Kulal, K. Lacey, Y. Levi, C. Li, D. Lorenz, J. Müller, D. Podell, R. Rombach, H. Saini, A. Sauer, and L. Smith, “Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,” arXiv preprint, 2025.  
[11] Google, “Veo 3: Google’s most capable video generation model,” tech. rep., Google, 2025.  
[12] D. Baranchuk, A. Voynov, I. Rubachev, V. Khrulkov, and A. Babenko, “Label-efficient semantic segmentation with diffusion models,” in International Conference on Learning Representations, 2022.  
[13] W. Xiang, H. Yang, D. Huang, and Y. Wang, “Denoising diffusion autoencoders are unified selfsupervised learners,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15802– 15812, 2023.  
[14] S. Mukhopadhyay, M. Gwilliam, V. Agarwal, N. Padmanabhan, A. Swaminathan, S. Hegde, T. Zhou, and A. Shrivastava, “Diffusion models beat gans on image classification,” arXiv preprint arXiv:2307.08702, 2023.  
[15] X. Chen, Z. Liu, S. Xie, and K. He, “Deconstructing denoising diffusion models for self-supervised learning,” in International Conference on Learning Representations, vol. 2025, pp. 55458–55472, 2025.  
[16] L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariharan, “Emergent correspondence from image diffusion,” Advances in Neural Information Processing Systems, vol. 36, pp. 1363–1389, 2023.  
[17] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.  
[18] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.  
[19] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, “Representation alignment for generation: Training diffusion transformers is easier than you think,” in International Conference on Learning Representations, 2025.  
[20] J. Singh, X. Leng, Z. Wu, L. Zheng, R. Zhang, E. Shechtman, and S. Xie, “What matters for representation alignment: Global information or spatial structure?,” in International Conference on Learning Representations, 2026.  
[21] A. Bardes, J. Ponce, and Y. LeCun, “Vicreg: Variance-invariance-covariance regularization for selfsupervised learning,” in International Conference on Learning Representations, 2022.  
[22] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning, pp. 1597–1607, PMLR, 2020.  
[23] S. Chen, H. Zhang, M. Guo, Y. Lu, P. Wang, and Q. Qu, “Exploring low-dimensional subspace in diffusion models for controllable image editing,” Advances in neural information processing systems, vol. 37, pp. 27340–27371, 2024.  
[24] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, “Barlow twins: Self-supervised learning via redundancy reduction,” in International conference on machine learning, pp. 12310–12320, PMLR, 2021.  
[25] P. Wang, H. Zhang, Z. Zhang, S. Chen, Y. Ma, and Q. Qu, “Diffusion models learn low-dimensional distributions via subspace clustering,” arXiv preprint, 2024.  
[26] X. Li, Z. Zhang, X. Li, S. Chen, Z. Zhu, P. Wang, and Q. Qu, “Understanding representation dynamics of diffusion models via low-dimensional modeling,” Advances in Neural Information Processing Systems, vol. 38, pp. 107365–107404, 2026.  
[27] P. Li, Z. Li, H. Zhang, and J. Bian, “On the generalization properties of diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 2097–2127, 2023.  
[28] X. Li, Y. Dai, and Q. Qu, “Understanding generalizability of diffusion models requires rethinking the hidden gaussian structure,” Advances in neural information processing systems, vol. 37, pp. 57499–57538, 2024.  
[29] H. Zhang, Z. Huang, S. Chen, J. Zhou, Z. Zhang, P. Wang, and Q. Qu, “Understanding generalization in diffusion models via probability flow distance,” arXiv preprint arXiv:2505.20123, 2025.  
[30] R. Baptista, A. Dasgupta, N. B. Kovachki, A. Oberai, and A. M. Stuart, “Memorization and regularization in generative diffusion models,” arXiv preprint arXiv:2501.15785, 2025.  
[31] T. Bonnaire, R. Urfin, G. Biroli, and M. Mézard, “Why diffusion models don’t memorize: The role of implicit dynamical regularization in training,” Advances in Neural Information Processing Systems, vol. 38, pp. 141266–141286, 2026.  
[32] G. Stein, J. Cresswell, R. Hosseinzadeh, Y. Sui, B. Ross, V. Villecroze, Z. Liu, A. L. Caterini, E. Taylor, and G. Loaiza-Ganem, “Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 3732–3784, 2023.  
[33] E. Pizzi, S. D. Roy, S. N. Ravindra, P. Goyal, and M. Douze, “A self-supervised descriptor for image copy detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14532–14542, 2022.  
[34] H. Zhang, J. Zhou, Y. Lu, M. Guo, P. Wang, L. Shen, and Q. Qu, “The emergence of reproducibility and consistency in diffusion models,” in International Conference on Machine Learning, pp. 60558–60590, PMLR, 2024.  
[35] B. D. Anderson, “Reverse-time diffusion equation models,” Stochastic Processes and their Applications, vol. 12, no. 3, pp. 313–326, 1982.  
[36] B. Efron, “Tweedie’s formula and selection bias,” Journal of the American Statistical Association, 2011.  
[37] Z. Kadkhodaie, F. Guth, E. Simoncelli, and S. Mallat, “Generalization in diffusion models arises from geometry-adaptive harmonic representations,” in International Conference on Learning Representations, vol. 2024, pp. 46543–46567, 2024.  
[38] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention, pp. 234–241, Springer, 2015.  
[39] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” Advances in neural information processing systems, vol. 35, pp. 26565–26577, 2022.  
[40] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, and S. Xie, “Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers,” in European Conference on Computer Vision, pp. 23–40, Springer, 2024.  
[41] M. Fuest, P. Ma, M. Gui, J. Schusterbauer, V. T. Hu, and B. Ommer, “Diffusion models and representation learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.  
[42] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.  
[43] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al., “Bootstrap your own latent-a new approach to self-supervised learning,” Advances in neural information processing systems, vol. 33, pp. 21271–21284, 2020.  
[44] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.  
[45] T. Wang and P. Isola, “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” in International conference on machine learning, pp. 9929–9939, PMLR, 2020.  
[46] K. Fukunaga, Introduction to Statistical Pattern Recognition. Academic Press, 2 ed., 1990. Second edition.  
[47] R. A. Horn and C. R. Johnson, Matrix analysis. Cambridge university press, 2012.  
[48] R. A. Fisher, “The use of multiple measurements in taxonomic problems,” Annals of Eugenics, vol. 7, no. 2, pp. 179–188, 1936.  
[49] A. Krizhevsky, “Learning multiple layers of features from tiny images,” tech. rep., University of Toronto, 2009.  
[50] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255, Ieee, 2009.  
[51] B. Wang and J. J. Vastola, “Diffusion models generate images like painters: an analytical theory of outline first, details later,” arXiv preprint arXiv:2303.02490, 2023.  
[52] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.  
[53] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1–9, 2015.  
[54] Z. Zhang, X. Li, X. Li, L. Shi, M. Wu, M. Tao, and Q. Qu, “Generalization of diffusion models arises with a balanced representation space,” in International Conference on Learning Representations, 2026.  
[55] K. Deja, T. Trzciński, and J. M. Tomczak, “Learning data representations with joint diffusion models,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 543–559, Springer, 2023.  
[56] J. Zhang, C. Herrmann, J. Hur, L. Polania Cabrera, V. Jampani, D. Sun, and M.-H. Yang, “A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence,” Advances in Neural Information Processing Systems, vol. 36, pp. 45533–45547, 2023.  
[57] Y. Shi, C. Xue, J. H. Liew, J. Pan, H. Yan, W. Zhang, V. Y. Tan, and S. Bai, “Dragdiffusion: Harnessing diffusion models for interactive point-based image editing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8839–8849, 2024.  
[58] C. S. Sastry, S. H. Dumpala, and S. Oore, “Diffaug: A diffuse-and-denoise augmentation for training robust classifiers,” Advances in Neural Information Processing Systems, 2024.  
[59] X. Yang and X. Wang, “Diffusion model as representation learner,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18938–18949, 2023.  
[60] D. Li, H. Ling, A. Kar, D. Acuna, S. W. Kim, K. Kreis, A. Torralba, and S. Fidler, “Dreamteacher: Pretraining image backbones with deep generative models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16698–16708, 2023.  
[61] N. Stracke, S. A. Baumann, K. Bauer, F. Fundel, and B. Ommer, “Cleandift: Diffusion features without noise,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 117–127, 2025.  
[62] G. Luo, L. Dunlap, D. H. Park, A. Holynski, and T. Darrell, “Diffusion hyperfeatures: Searching through time and space for semantic correspondence,” Advances in Neural Information Processing Systems, vol. 36, pp. 47500–47510, 2023.  
[63] S. Mittal, K. Abstreiter, S. Bauer, B. Schölkopf, and A. Mehrjou, “Diffusion based representation learning,” in International conference on machine learning, pp. 24963–24982, PMLR, 2023.  
[64] Y. Wang, Y. Schiff, A. Gokaslan, W. Pan, F. Wang, C. De Sa, and V. Kuleshov, “Infodiffusion: Representation learning using information maximizing diffusion models,” in International Conference on Machine Learning, pp. 36336–36354, PMLR, 2023.  
[65] D. A. Hudson, D. Zoran, M. Malinowski, A. K. Lampinen, A. Jaegle, J. L. McClelland, L. Matthey, F. Hill, and A. Lerchner, “Soda: Bottleneck diffusion models for representation learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23115–23127, 2024.  
[66] K. Preechakul, N. Chatthee, S. Wizadwongsa, and S. Suwajanakorn, “Diffusion autoencoders: Toward a meaningful and decodable representation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10619–10629, 2022.  
[67] Y. Han, A. Han, W. Huang, C. Lu, and D. Zou, “Can diffusion models learn hidden inter-feature rules behind images?,” in International Conference on Machine Learning, pp. 21704–21732, PMLR, 2025.  
[68] Y. Wang, P. Wang, H. Jiang, Z. Yang, Q. Huang, and Z. Wang, “Revisiting spectral representations in generative diffusion models,” 2026.  
[69] K. K. Agrawal, A. K. Mondal, A. Ghosh, and B. Richards, “??-req : Assessing representation quality in self-supervised learning by measuring eigenspectrum decay,” Advances in Neural Information Processing Systems, vol. 35, pp. 17626–17638, 2022.  
[70] Q. Garrido, R. Balestriero, L. Najman, and Y. Lecun, “Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank,” in International conference on machine learning, pp. 10929–10974, PMLR, 2023.  
[71] V. Thilak, C. Huang, O. Saremi, L. Dinh, H. Goh, P. Nakkiran, J. Susskind, and E. Littwin, “Lidar: Sensing linear probing performance in joint embedding ssl architectures,” in International Conference on Learning Representations, vol. 2024, pp. 56726–56765, 2024.  
[72] M. Kamb and S. Ganguli, “An analytic theory of creativity in convolutional diffusion models,” in International Conference on Machine Learning, pp. 28795–28831, PMLR, 2025.  
[73] L. Shi, M. Wu, H. Zhang, Z. Zhang, M. Tao, and Q. Qu, “A closer look at model collapse: From a generalization-to-memorization perspective,” Advances in Neural Information Processing Systems, vol. 38, pp. 40658–40691, 2026.  
[74] B. Achilli, E. Ventura, G. Silvestri, B. Pham, G. Raya, D. Krotov, C. Lucibello, and L. Ambrogioni, “Losing dimensions: Geometric memorization in generative diffusion,” arXiv preprint arXiv:2410.08727, 2024.  
[75] S. Buchanan, D. Pai, Y. Ma, and V. De Bortoli, “On the edge of memorization in diffusion models,” Advances in Neural Information Processing Systems, vol. 38, pp. 96113–96157, 2026.  
[76] B. Wang, E. L. B. Finn, and B. Liu, “The two clocks and the innovation window: When and how generative models learn rules,” arXiv preprint arXiv:2605.10019, 2026.  
[77] B. Wang and C. Pehlevan, “An analytical theory of spectral bias in the learning dynamics of diffusion models,” Advances in Neural Information Processing Systems, vol. 38, pp. 95865–95963, 2026.  
[78] A. Favero, A. Sclocchi, and M. Wyart, “Bigger isn’t always memorizing: Early stopping overparameterized diffusion models,” arXiv preprint arXiv:2505.16959, 2025.  
[79] M. Niedoba, B. Zwartsenberg, K. P. Murphy, and F. Wood, “Towards a mechanistic explanation of diffusion model generalization,” in Forty-second International Conference on Machine Learning, 2025.  
[80] A. Lukoianov, C. Yuan, J. Solomon, and V. Sitzmann, “Locality in image diffusion models emerges from data statistics,” Advances in Neural Information Processing Systems, vol. 38, pp. 95121–95157, 2025.  
[81] G. Biroli, T. Bonnaire, V. De Bortoli, and M. Mézard, “Dynamical regimes of diffusion models,” Nature Communications, 2024.  
[82] A. Sclocchi, A. Favero, and M. Wyart, “A phase transition in diffusion models reveals the hierarchical nature of data,” Proceedings of the National Academy of Sciences, 2025.  
[83] L. Ambrogioni, “In search of dispersed memories: Generative diffusion models are associative memory networks,” Entropy, 2024.  
[84] B. Pham, G. Raya, M. Negri, M. J. Zaki, L. Ambrogioni, and D. Krotov, “Memorization to generalization: Emergence of diffusion models from associative memory,” arXiv preprint arXiv:2505.21777, 2025.  
[85] S. Kornblith, T. Chen, H. Lee, and M. Norouzi, “Why do better loss functions lead to less transferable features?,” Advances in Neural Information Processing Systems, vol. 34, pp. 28648–28662, 2021.  
[86] X. Li, Y. Jia, X. Li, J. A. Fessler, R. Wang, and Q. Qu, “Mclr: Improving conditional modeling via interclass likelihood-ratio maximization and unifying classifier-free guidance with alignment objectives,” arXiv preprint arXiv:2603.22364, 2026.  
[87] V. Papyan, X. Han, and D. L. Donoho, “Prevalence of neural collapse during the terminal phase of deep learning training,” Proceedings of the National Academy of Sciences, vol. 117, no. 40, pp. 24652–24663, 2020.  
[88] Z. Zhu, T. Ding, J. Zhou, X. Li, C. You, J. Sulam, and Q. Qu, “A geometric analysis of neural collapse with unconstrained features,” Advances in Neural Information Processing Systems, vol. 34, pp. 29820–29834, 2021.  
[89] T. Galanti, A. György, and M. Hutter, “On the role of neural collapse in transfer learning,” in International Conference on Learning Representations, 2022.  
[90] Z. Wang, Y. Luo, L. Zheng, Z. Huang, and M. Baktashmotlagh, “How far pre-trained models are from neural collapse on the target dataset informs their transferability,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5549–5558, 2023.  
[91] X. Li, S. Liu, J. Zhou, X. Lu, C. Fernandez-Granda, Z. Zhu, and Q. Qu, “Understanding and improving transfer learning of deep models via neural collapse,” Transactions on machine learning research, 2024.  
[92] P. Wang, X. Li, C. Yaras, Z. Zhu, L. Balzano, W. Hu, and Q. Qu, “Understanding deep representation learning via layerwise feature compression and discrimination,” Journal of Machine Learning Research, vol. 26, no. 220, pp. 1–61, 2025.  
[93] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2017.

## A Related Work

Diffusion-based representation learning. Many works treat a trained diffusion denoiser as a feature extractor and test its features on downstream tasks. These features work well for image classification [13, 14, 55], segmentation [12], correspondence [56, 16], and image editing [57], and recent surveys summarize this line of work [41]. Some articles also use diffusion models to generate augmentations and improve robustness under covariate shift [58]. Since representation quality often depends on the noise level used for feature extraction, several distillation and compression methods aim to reduce the need for expensive timestep search and to improve transfer [59, 60, 61, 62]. Other work changes the training objective or the network to better combine generation and representation learning, for example, by adding new information-based losses or by building an explicit autoencoding structure [63, 64, 65, 66]. [67] further studies whether diffusion models learn hidden dependencies among image features. In contrast to methods that mainly aim to improve downstream transfer or generation, we study how diffusion features change across noise levels and across training, and we link this behavior to self-supervised principles.

Representation dynamics and links to self-supervised learning. Prior work [26] study why diffusion representations often peak at an intermediate noise level and explain this unimodal behavior through a low-dimensional data model. They further show that this unimodal pattern disappears when diffusion models transition from generalization to memorization [26]. More recently, Wang et al. [68] connect diffusion models and self-supervised learning through a shared perturbation-kernel perspective and propose a spectral alignment objective that improves diffusion training. This SSL perspective is also reflected in recent diffusion training methods such as REPA, which align diffusion representations with embeddings from pretrained self-supervised encoders (e.g., DINOv2) to improve generation quality and training efficiency [17, 19, 20].

Our work differs from these lines in its goal. Rather than modifying the training objective or introducing an external teacher encoder, we study diffusion representations through principles inspired by self-supervised learning. In particular, our work is closely related to a line of research that develops label-free metrics for predicting downstream representation quality and guiding model selection, including metrics based on covariance spectrum decay, effective rank, and Fisher-style covariance decompositions [69, 70, 71]. Similar to these works, we seek to evaluate representation quality directly from feature statistics without training downstream classifiers. However, unlike prior metrics designed primarily for representation selection in SSL, we leverage the invariant and residual decomposition induced by augmentations and diffusion noise to study diffusionspecific phenomena, including semantic windows, generation quality, memorization, and training dynamics.

![](images/b7a40f28a2eec37d84f1285e86227d8ca838d02ee6fd8b8b6e5f3eb946553892.jpg)

<details>
<summary>stacked bar chart</summary>

| Time step (σt) | Tr(ΣS) | Tr(Σξ) |
|---|---|---|
| 0.0 | 0.025 | 0.015 |
| 0.01 | 0.035 | 0.025 |
| 0.02 | 0.065 | 0.045 |
| 0.06 | 0.12 | 0.055 |
| 0.14 | 0.195 | 0.115 |
| 0.3 | 0.255 | 0.175 |
| 0.59 | 0.275 | 0.215 |
| 1.09 | 0.245 | 0.285 |
| 1.92 | 0.195 | 0.335 |
| 3.26 | 0.135 | 0.345 |
</details>

(a) CIFAR10

![](images/0974c1227d355c5c81c78e56398983da17052e339b1a5795525b78134c9f28cb.jpg)

<details>
<summary>stacked bar chart</summary>

| Time step (σt) | Tr(ΣS) | Tr(Σξ) |
| :--- | :--- | :--- |
| 0.0 | 0.11 | 0.03 |
| 0.01 | 0.21 | 0.06 |
| 0.02 | 0.32 | 0.11 |
| 0.06 | 0.36 | 0.14 |
| 0.14 | 0.33 | 0.15 |
| 0.3 | 0.26 | 0.17 |
| 0.59 | 0.21 | 0.18 |
| 1.09 | 0.16 | 0.21 |
| 1.92 | 0.12 | 0.23 |
| 3.26 | 0.10 | 0.25 |
</details>

(b) CIFAR100  
Figure 9: Invariant and residual energy across diffusion noise levels. For pretrained EDM models on CIFAR10 and CIFAR100 in the data-rich regime, we plot the traces of the invariant and residual covariances, $\operatorname { T r } ( \Sigma _ { s } ( \sigma _ { t } ) )$ and $\operatorname { T r } ( \Sigma _ { \xi } ( \sigma _ { t } ) )$ , as functions of the noise level $\sigma _ { t }$ . Invariant energy $\mathrm { T r } ( \Sigma _ { s } )$ increases from low noise, peaks at an intermediate scale, and then decreases in the high noise regime, whereas residual energy $\operatorname { T r } ( \pmb { \Sigma } _ { \xi } )$ grows monotonically with $\sigma _ { t }$ .

Memorization and generalization in diffusion models. Our work contributes to the broad line of research aiming to understand memorization and generalization behaviors in diffusion models [37, 34, 72, 73, 29]. A large body of work has studied memorization in diffusion models from the perspectives of model complexity and data quantity [25, 74, 31, 75, 76], and has shown that memorization typically emerges after an initial generalization phase when diffusion models are trained with limited data [28, 77, 31, 78]. Other works seek to explain why diffusion models are able to recover the underlying score function from discrete empirical samples [79, 80]. The generation and generalization behaviors across the reverse diffusion process have also been studied in [81, 82]. In parallel, Ambrogioni et al. [83] established an asymptotic equivalence between generative diffusion models and modern Hopfield networks (associative memory networks), and a subsequent work [84] leveraged this associative memory perspective to identify spurious samples that emerge as diffusion models transition from generalization to memorization.

## B Additional Discussions & Experiments

## B.1 Component Dynamics Across the Noise Schedule

In Section 4, we show that there exists a strong correspondence between ICR and classification performance across noise levels. In this subsection, we also track the energy progression of the total variance of the invariant signal, $\mathrm { T r } ( \Sigma _ { s } )$ , and the residual variation, $\operatorname { T r } ( \pmb { \Sigma } _ { \xi } )$ .

As shown in Figure 9, $\operatorname { T r } ( \pmb { \Sigma } _ { \xi } )$ ?? grows monotonically with the noise level $\sigma _ { t } ,$ reflecting the increased reconstructive uncertainty inherent in high-noise denoising. Conversely, $\mathrm { T r } ( \Sigma _ { s } )$ exhibits a unimodal ??trajectory, peaking at intermediate scales. This suggests a “semantic window” where the model’s invariant subspace is most expansive. Notably, peak classification accuracy (as shown in Figure 3) does not always correspond with the maximum of $\mathrm { T r } ( \pmb { \Sigma } _ { s } )$ , but rather with the minimum of the ICR ratio. This indicates that representation utility is governed not by the absolute magnitude of the invariant signal, but by its strength relative to the contaminating residual variation.

![](images/a1f5b0e20b68b5801b9ec984008fbe1c1906fbe8220782817299e3d56cef9943.jpg)

<details>
<summary>line chart</summary>

| Time step (σt) | ICR   | Classification Acc. (%) |
| -------------- | ----- | ------------------------ |
| 0.0            | 0.33  | 60                       |
| 0.02           | 0.34  | 58                       |
| 0.14           | 0.35  | 55                       |
| 0.59           | 0.40  | 50                       |
| 1.92           | 0.65  | 40                       |
| 5.32           | 0.90  | 25                       |
</details>

Figure 10: Correspondence between ICR and classification accuracy across noise levels in the data-limited setting. We study the behavior of ICR across noise levels in the data-limited setting under prolonged training. In this regime, classification accuracy no longer exhibits the unimodal trend observed in the generalization phase, and instead decreases monotonically as noise increases. In contrast, ICR increases monotonically and maintains a clear negative correlation with classification accuracy. This demonstrates that ICR continues to track representation quality even when classification accuracy no longer follows the typical generalization pattern.

## B.2 ICR and Classification Accuracy in the Data-limited Regime

In Figure 3, we show that ICR exhibits a strong negative correlation with classification accuracy across noise levels when diffusion models are trained using the full training set. Interestingly, the classification accuracy follows a unimodal trend as the noise level increases, a phenomenon previously observed and analyzed in [13, 26]. Moreover, Li et al. [26] showed that this unimodal behavior is closely associated with the diffusion model correctly learning the underlying lowdimensional data distribution, and that it disappears once the model begins to memorize the training data, at which point the accuracy instead decreases monotonically with noise level.

To further test the robustness of the correlation between ICR and classification accuracy, we train an EDM model on a subset of 4,096 CIFAR10 images for a prolonged period until the model substantially overfits and memorizes the training data. We then evaluate both ICR and classification accuracy across noise levels at this checkpoint, and report the results in Figure 10.

As shown in the figure, classification accuracy no longer exhibits the unimodal trend observed during the generalization phase, and instead decreases monotonically as the noise level increases. Correspondingly, ICR exhibits a monotonic increasing trend across noise levels, further strengthening the correlation between ICR and representation quality even when the standard semantic window disappears due to memorization.

## B.3 Discussion on the Alignment and Uniformity Metrics [45]

Wang et al. [45] introduced two feature-level criteria for contrastive SSL encoders. In our notation, let $\pmb { h } ( a ( \pmb { x } ) ) \in \mathbb { R } ^ { d }$ denote the feature of an augmented image, and let $( x , x ^ { \prime } )$ be a positive pair obtained ??from two augmentations of the same image. The alignment loss is

$$
\mathcal {L} _ {\mathrm{align}} (\boldsymbol {h}) := \mathbb {E} _ {(x, x ^ {\prime})} \left[ \| \boldsymbol {h} (a _ {1} (\boldsymbol {x})) - \boldsymbol {h} (a _ {2} (\boldsymbol {x})) \| _ {2} ^ {\alpha} \right], \tag {6}
$$

![](images/919f8b0eaaa5ff2b59a3034943cace863863fb4d422115d9868ec097adbc3831.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | ICR    | FID Score |
| --------------------- | ------ | --------- |
| 2.5m                  | 0.40   | 30        |
| 7.5m                  | 0.36   | 15        |
| 15m                   | 0.32   | 10        |
| 25m                   | 0.30   | 5         |
| 50m                   | 0.28   | 3         |
| 100m                  | 0.26   | 2         |
</details>

(a) ICR and FID

![](images/5023bb182dd481d4ea06fe616458c8ec686aea8e7d16dbabcd1b4de95f707ac3.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | Alignment Loss | FID Score |
| --------------------- | -------------- | --------- |
| 2.5m                  | 0.150          | 30.0      |
| 7.5m                  | 0.190          | 10.0      |
| 15m                   | 0.260          | 5.0       |
| 25m                   | 0.300          | 4.0       |
| 50m                   | 0.330          | 3.0       |
| 100m                  | 0.340          | 2.5       |
</details>

(b) $\mathcal { L } _ { \mathrm { a l i g n } } ( h )$ and FID  
Figure 11: Alignment versus ICR in data-rich diffusion training (CIFAR10, EDM). We track FID together with ICR and the alignment loss $\mathcal { L } _ { \mathrm { a l i g n } }$ over training on full CIFAR10. Both ICR (blue) and FID (brown) decrease monotonically, indicating improving representation invariance and generative quality, while $\mathcal { L } _ { \mathrm { a l i g n } } \ : ( \mathrm { g r e e n } )$ increases despite being a lower is better metric.

the expected squared distance between two views of the same sample. The uniformity loss is

$$
\mathcal {L} _ {\text { uniform }} (\boldsymbol {h}; t) := \log \mathbb {E} _ {x, y} \left[ \exp \left(- t \| \boldsymbol {h} (a (\boldsymbol {x})) - \boldsymbol {h} (a ^ {\prime} (\boldsymbol {y})) \| _ {2} ^ {2}\right) \right], \quad t > 0, \tag {7}
$$

which encourages features to be spread out on the unit hypersphere. In the contrastive setting, encoders that achieve low alignment together with good uniformity tend to have strong downstream classification accuracy.

Our focus is slightly different. We are interested in how the representation space evolves across diffusion noise levels and along the training trajectory, and in particular in a relative notion of invariance that compares invariant structure to view specific variation while remaining stable under overall feature expansion. In this regime, the alignment loss becomes less informative. As shown in Figure 11, in the data-rich case the FID and ICR both decrease monotonically as training progresses, while $\mathcal { L } _ { \mathrm { a l i g n } }$ continues to increase, suggesting worse alignment. This apparent disagreement is largely due to the fact that $\mathcal { L } _ { \mathrm { a l i g n } }$ is an absolute squared distance: during diffusion training the overall feature variance grows (see Figure 8), so alignment can increase even when the relative invariant structure is improving.

## B.4 Discussion on the Class Separation and Silhouette Score Metrics

Besides the alignment and uniformity metrics discussed in [45], we additionally discuss two other representation metrics here: the class separation metric and the Silhouette score.

Class separation is a metric proposed in [85], which measures the within-class variation of representations relative to the overall variation.5 Formally, based on cosine distances between normalized features, let $\boldsymbol { h } _ { k , m }$ denote the representation of sample  from class $k ,$ where  is the number of classes and $N _ { k }$ ,??is the number of samples in class $k .$ ?? ?? ?? The class separation score $R ^ { 2 }$ is defined as

![](images/d3f273ac2c676c1ebabd983a02fd3ebb872dc182c78d8a4b80d62a994f2649c9.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | ICR    | Memorization Ratio |
| --------------------- | ------ | ------------------ |
| 2.5m                  | 0.36   | 0.00               |
| 5m                    | 0.32   | 0.00               |
| 7.5m                  | 0.29   | 0.00               |
| 10m                   | 0.28   | 0.00               |
| 20m                   | 0.30   | 0.05               |
| 30m                   | 0.32   | 0.10               |
| 75m                   | 0.34   | 0.25               |
| 200m                  | 0.36   | 0.35               |
</details>

(a) ICR

![](images/e47e115cc53f41e1f04d0efcf81eb7300391de0c74e85f698b43dc9a4f36ff06.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | Silhouette Score | Memorization Ratio (0-1) |
| --------------------- | ---------------- | ------------------------ |
| 2.5m                  | 0.17             | 0.00                     |
| 5m                    | 0.16             | 0.00                     |
| 7.5m                  | 0.15             | 0.00                     |
| 10m                   | 0.14             | 0.00                     |
| 20m                   | 0.12             | 0.05                     |
| 30m                   | 0.11             | 0.15                     |
| 75m                   | 0.11             | 0.25                     |
| 200m                  | 0.11             | 0.35                     |
</details>

(b) Silhouette score

![](images/4f8f8591c20a06570af8c8c69ac9144e6e8d1797b2ab8521cef274ab068445d7.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | Class Separation | Memorization Ratio (0-1) |
| --------------------- | ---------------- | ------------------------ |
| 2.5m                  | 0.36             | 0.00                     |
| 5m                    | 0.38             | 0.00                     |
| 7.5m                  | 0.36             | 0.00                     |
| 10m                   | 0.34             | 0.00                     |
| 20m                   | 0.30             | 0.05                     |
| 30m                   | 0.29             | 0.15                     |
| 75m                   | 0.28             | 0.25                     |
| 200m                  | 0.28             | 0.36                     |
</details>

(c) Class separation  
Figure 12: ICR, Silhouette score, and class separation in data-limited diffusion training (CI-FAR10, EDM). We revisit the experiment in Figure 4 (Training EDM on CIFAR10, with $_ { \mathrm { N = 4 , 0 9 6 } }$ images) by incorporating two additional representation metrics: (i) the Silhouette Score, a partially unsupervised metric that relies on pseudo-labels (e.g., from k-means), and (ii) Class Separation, a supervised metric that depends on ground-truth labels. As shown in the figure, ICR exhibits a stronger and more consistent alignment with the memorization ratio, whereas the other two metrics saturate too early and fail to capture this trend as reliably.

$$
R ^ {2} = 1 - \frac {\bar {d} _ {\mathrm{within}}}{\bar {d} _ {\mathrm{total}}},
$$

where

$$
\bar {d} _ {\mathrm{within}} = \sum_ {k = 1} ^ {K} \sum_ {m = 1} ^ {N _ {k}} \sum_ {n = 1} ^ {N _ {k}} \frac {1 - \mathrm{sim} (\pmb {h} _ {k , m} , \pmb {h} _ {k , n})}{K N _ {k} ^ {2}},
$$

$$
\bar {d} _ {\mathrm{total}} = \sum_ {j = 1} ^ {K} \sum_ {k = 1} ^ {K} \sum_ {m = 1} ^ {N _ {j}} \sum_ {n = 1} ^ {N _ {k}} \frac {1 - \mathrm{sim} (\pmb {h} _ {j , m} , \pmb {h} _ {k , n})}{K ^ {2} N _ {j} N _ {k}}.
$$

The metric was shown in [85] to correlate with transferability of ImageNet pre-trained models on downstream classification tasks. However, as the name suggests, the metric requires ground-truth class labels and is therefore not suitable for unsupervised settings. In addition, being a global statistic, it does not explicitly capture directional structure in the representation space as ICR does.

Silhouette score is a clustering-based metric that evaluates how well a representation separates data into groups (or classes when supervised labels are available). Given a feature representation $h _ { i } ,$ , define

• $a ( i )$ : the average distance between $h _ { i }$ and samples within the same cluster,  
• ( ): the minimum average distance between $h _ { i }$ and samples in other clusters.

The Silhouette score is then defined as

$$
s (i) = \frac {b (i) - a (i)}{\max (a (i) , b (i))},
$$

and the overall score is obtained by averaging $s ( i )$ over all samples.

?? ??The Silhouette score measures the balance between intra-cluster compactness and inter-cluster separation, where larger values indicate better separation. However, this metric is not fully unsupervised, since it still requires cluster assignments, typically obtained through algorithms such as k-means, making it sensitive to hyperparameters such as the number of clusters. In addition, it relies on pairwise distances, which can become unstable in high-dimensional feature spaces, and similarly does not explicitly capture directional structure.

To compare these two metrics with ICR, we redo the experiments in Figure 5 and report the results in Figure 12. We observe that both the class separation score and the Silhouette score exhibit a unimodal trend during training, suggesting that they can partially capture the transition from learning the underlying data distribution to memorizing training samples. However, as shown in the figure, ICR exhibits a substantially stronger and more consistent alignment with the memorization ratio, whereas the other two metrics saturate much earlier and fail to reliably track the later-stage memorization behavior. We conjecture that this limitation is partly due to the global nature of these metrics.

## B.5 Connection to Neural Collapse

In the main text we showed that the generalized eigenvalues admit a direct SNR interpretation: for each Fisher direction $v _ { i } ,$ the generalized Rayleigh quotient equals $\lambda _ { i } ,$ and

$$
\mathrm{ICR} = \frac {d}{d + \sum_ {i = 1} ^ {d} \lambda_ {i}}, \qquad \sum_ {i = 1} ^ {d} \lambda_ {i} = \mathrm{Tr} \big (\boldsymbol {\Sigma} _ {\xi} ^ {- 1} \boldsymbol {\Sigma} _ {s} \big).
$$

$\widetilde { \pmb { \Sigma } } _ { s } : = \pmb { \Sigma } _ { \xi } ^ { - 1 / 2 } \pmb { \Sigma } _ { s } \pmb { \Sigma } _ { \xi } ^ { - 1 / 2 }$ $\boldsymbol { u } _ { i } = \boldsymbol { \Sigma } _ { \xi } ^ { 1 / 2 } \boldsymbol { v } _ { i } ,$ so that $\pmb { u } _ { i }$ are the eigenvectors of $\widetilde { \Sigma } _ { s }$ with eigenvalues $\lambda _ { i }$ and $\begin{array} { r } { \mathrm { T r } ( \Sigma _ { z } ^ { - 1 } \Sigma _ { s } ) = \mathrm { T r } ( \widetilde { \Sigma } _ { s } ) = \sum _ { i } \lambda _ { i } } \end{array}$ .

The trace form $\mathrm { T r } ( \pmb { \Sigma } _ { \xi } ^ { - 1 } \pmb { \Sigma } _ { s } )$ ?? could be linked to a familiar quantity in the Neural Collapse (?? ??) ??literature [87, 88]. Neural Collapse refers to a phenomenon observed near the terminal phase of training in classification networks: penultimate features for each class collapse to a single mean vector, and these class means become maximally separated. A standard metric for quantifying this behavior is the $N C _ { 1 }$ score, defined for a -class classifier as

$$
\mathcal {N C} _ {1} = \frac {1}{K} \operatorname{Tr} \bigl (\boldsymbol {\Sigma} _ {B} ^ {\dagger} \boldsymbol {\Sigma} _ {W} \bigr),
$$

where

$$
\begin{array}{l} \pmb {h} _ {G} = \frac {1}{n K} \sum_ {k = 1} ^ {K} \sum_ {i = 1} ^ {n} \pmb {h} _ {k, i}, \qquad \bar {\pmb {h}} _ {k} = \frac {1}{n} \sum_ {i = 1} ^ {n} \pmb {h} _ {k, i}, (1 \leq k \leq K), \\ \pmb {\Sigma} _ {W} := \frac {1}{n K} \sum_ {k = 1} ^ {K} \sum_ {i = 1} ^ {n} \left(\pmb {h} _ {k, i} - \bar {\pmb {h}} _ {k}\right) \left(\pmb {h} _ {k, i} - \bar {\pmb {h}} _ {k}\right) ^ {\top}, \qquad \pmb {\Sigma} _ {B} := \frac {1}{K} \sum_ {k = 1} ^ {K} \left(\bar {\pmb {h}} _ {k} - \pmb {h} _ {G}\right) \left(\bar {\pmb {h}} _ {k} - \pmb {h} _ {G}\right) ^ {\top}. \\ \end{array}
$$

Here $\pmb { \Sigma } _ { W }$ and $\Sigma _ { B }$ denote the within-class and between-class covariance matrices of the penultimate ??features.

The design principle behind $N C _ { 1 }$ is closely related to ours: both metrics compare two covariance structures via a trace of a generalized eigenvalue type object, effectively measuring a signal-to-noise ratio in feature space. The key difference is that $N C _ { 1 }$ is label-based, contrasting between class and within-class variation, whereas ICR is built from a self-supervised principle that contrasts perturbation-invariant and perturbation-sensitive components without using labels.

Interestingly, several works have used $N C _ { 1 }$ and related quantities as metrics for assessing the transferability of pretrained discriminative models to downstream tasks [89, 90, 91, 92]. Our results suggest that a similar trace-based viewpoint extends naturally to diffusion models, and that a unified representation-based evaluation principle may be possible that covers both discriminative and generative settings through appropriate choices of “signal” and “noise” covariances.

## B.6 Technical Details on Calculating ICR

In Section 3.2, we introduced the formal definition of the metric ICR and argue that it can be efficiently estimated using only two augmented views and a subset of training features6. In this subsection, we briefly discuss alternative formulations of the metric, dive into more detail on the estimation of it and the robustness of the estimation regards different number of samples used.

As noted in the main manuscript, the computation of ICR involves solving a generalized eigenvalue problem, one may argue that a simpler alternative is a trace-based statistic such as $\mathrm { T r } ( \pmb { \Sigma } _ { \xi } ) / \mathrm { T r } ( \pmb { \Sigma } _ { s } ^ { - } )$ , which aggregates all directions into a single global statistic. However, this aggre-??gation loses directional information. For example, consider a representation where only a small number of directions carry strong invariant signal while the remaining directions are dominated by residual variation. In this case, the trace ratio averages over all directions and fails to reflect the presence of these highly informative directions. In contrast, the generalized eigenvalue formulation explicitly captures the signal-to-noise ratio along each direction and is therefore sensitive to such anisotropic structures.

Two-view approximation and empirical estimation. The conditional expectation over all augmentations and noise realizations is not available in practice. Following augmentation based self supervised learning, we approximate it using two independent views per image. For each $x ,$ sample $a _ { 1 } , a _ { 2 } \sim \mathcal { A }$ independently and set

$$
\boldsymbol {h} _ {1} = \boldsymbol {h} (a _ {1} (\boldsymbol {x} _ {0})), \qquad \boldsymbol {h} _ {2} = \boldsymbol {h} (a _ {2} (\boldsymbol {x} _ {0})).
$$

Under the decomposition above, this can be written as

$$
\boldsymbol {h} _ {1} = \boldsymbol {s} (\boldsymbol {x} _ {0}) + \boldsymbol {\xi} _ {1}, \qquad \boldsymbol {h} _ {2} = \boldsymbol {s} (\boldsymbol {x} _ {0}) + \boldsymbol {\xi} _ {2},
$$

where $\xi _ { v } = \xi ( a _ { v } , x _ { 0 } )$ are zero mean residuals that are conditionally uncorrelated across views given ?? ???? ,?? and share the same covariance $\Sigma _ { \xi }$ .

We construct effective semantic and nuisance covariances from the sum and difference of the two views. Define

$$
\boldsymbol {t} := \frac {1}{2} (\boldsymbol {h} _ {1} + \boldsymbol {h} _ {2}), \quad \boldsymbol {d} := \boldsymbol {h} _ {1} - \boldsymbol {h} _ {2}.
$$

A direct covariance calculation under the assumptions above yields

$$
\operatorname{Cov} (\pmb {d}) = 2 \pmb {\Sigma} _ {\xi}, \qquad \operatorname{Cov} (\pmb {t}) = \pmb {\Sigma} _ {s} + \frac {1}{2} \pmb {\Sigma} _ {\xi},
$$

![](images/9f79c45402dc0c6d1eb4c068d4619936e66f6f1105410ac76142c891a508360a.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | ICR (all 50,000 samples) | ICR (N=64) | ICR (N=128) | ICR (N=256) | ICR (N=512) | ICR (N=1024) | ICR (N=2048) | ICR (N=4096) | ICR (N=8192) | ICR (N=16384) | ICR (N=32768) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5m | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| 5m | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 |
| 7.5m | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| 10m | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 | 0.41 |
| 15m | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 |
| 20m | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| 25m | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 |
| 30m | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 | 0.47 |
| 50m | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 |
| 75m | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 |
| 100m | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| 200m | - | - | - | - | - | - | - | - | - | - | - |
| N = 64 | - | - | - | - | - | - | - | - | - | - | - |
| N = 128 | - | - | - | - | - | - | - | - | - | - | - |
| N = 256 | - | - | - | - | - | - | - | - | - | - | - |
| N = 512 | - | - | - | - | - | - | - | - | - | - | - |
| N = 1024 | - | - | - | - | - | - | - | - | - | - | - |
| N = 2768 | - | - | - | - | - | - | - | - | - | - | - |
| N = 8192 | - | - | - | - | - | - | - | - | - | - | - |
| N = 16384 | - | - | - | - | - | - | - | - | - | - | - |
| N = 32768 | - | - | - | - | - | - | - | - | - | - | - |
</details>

Figure 13: Stability of ICR estimates under subsampling. We evaluate ICR on CIFAR10 using a pretrained EDM model (4095 training samples) and a fixed noise level, varying the number of training samples used to estimate the covariances from $N = 1 6$ up to the full 50K images. As ??increases, the estimated ICR quickly showcase the similar trend close to the full data estimate.

so that $\Sigma _ { s }$ and $\Sigma _ { \xi }$ can be recovered from the second moments of ?? and ??:

$$
\boldsymbol {\Sigma} _ {\xi} = \frac {1}{2} \operatorname{Cov} (\boldsymbol {d}), \quad \boldsymbol {\Sigma} _ {s} = \operatorname{Cov} (\boldsymbol {t}) - \frac {1}{4} \operatorname{Cov} (\boldsymbol {d}).
$$

In our experiments we estimate Cov(??) and $\operatorname { C o v } ( d )$ from paired augmentations within each dataset and obtain empirical covariances $\widehat { \Sigma } _ { s }$ and $\widehat { \Sigma } _ { \xi }$ via the same formulas.

Estimating ICR from a subset of samples. In practice, ICR is computed from empirical covariances $\widehat { \Sigma } _ { s }$ and $\widehat { \Sigma } _ { \xi }$ estimated from a finite set of features, so the estimate may deviate from its population ??value depending on the number of samples used. Under standard covariance concentration results for subgaussian features, the estimation error of $\widehat { \Sigma } _ { s }$ and $\widehat { \Sigma } _ { \xi }$ scales on the order of $\sqrt { d / N }$ in operator ?? ?? ??norm, where  is the feature dimension and  is the number of samples. Since ICR depends on ??these covariances only through the trace term $\mathrm { T r } ( \pmb { \Sigma } _ { \xi } ^ { - 1 } \pmb { \Sigma } _ { s } )$ , we expect its finite sample estimate to be relatively stable once  is moderately larger than $d .$

?? ??To verify this empirically, we perform a sample complexity study on CIFAR10. Fixing a pretrained EDM model and a representative noise level, we compute ICR using random subsets of training images with sizes

$$
N \in \{1 6, 3 2, 6 4, 1 2 8, 2 5 6, 5 1 2, 1 0 2 4, 2 0 4 8, 4 0 9 6, 8 1 9 2, 1 6 3 8 4, 3 2 7 6 8, 5 0 0 0 0 \},
$$

where 50 000 is the full training set size. As shown in Figure 13, the estimated ICR stabilizes quickly: ,even with a few hundred to a few thousand samples, the ICR trends are already very close to the full dataset estimate. This robustness justifies our use of relatively small subsets of training features to monitor ICR throughout the experiments in the main text.

Sensitivity of ICR to augmentation design. Since the computation of ICR relies on collecting representations under random augmentations, we further investigate its robustness to different augmentation strengths. Specifically, we consider five augmentation settings: Level 1 uses Random-Crop alone; level 2 adds horizontal flipping; level 3 further adds Color Jitter (the default setting used throughout the manuscript); level 4 additionally adds random rotation; and level 5 further adds CenterCrop. We denote these augmentation settings as Aug.1 through Aug.5.

![](images/37fa2b1689530a0dab6c4a0ce0322fa8c48b4c236fde190909434974340603b0.jpg)

<details>
<summary>line chart</summary>

| Time step (σₜ) | Aug. 1 | Aug. 2 | Aug. 3 (Default) | Aug. 4 | Aug. 5 |
| -------------- | ------ | ------ | ---------------- | ------ | ------ |
| 0.0            | 0.1    | 0.15   | 0.2              | 0.35   | 0.5    |
| 0.02           | 0.08   | 0.12   | 0.18             | 0.3    | 0.45   |
| 0.14           | 0.1    | 0.15   | 0.2              | 0.3    | 0.5    |
| 0.59           | 0.2    | 0.3    | 0.4              | 0.5    | 0.6    |
| 1.92           | 0.7    | 0.75   | 0.8              | 0.85   | 0.85   |
</details>

(a) CIFAR10, EDM, N = 50K

![](images/56d6fb8a9eac5b6f2c52a6b061e645be7a7e77089c039819fa830fe915443525.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | Aug. 1 | Aug. 2 | Aug. 3 (Default) | Aug. 4 | Aug. 5 |
| --------------------- | ------ | ------ | ---------------- | ------ | ------ |
| 2.5m                  | 1.00   | 1.00   | 1.00             | 1.00   | 1.00   |
| 7.5m                  | 0.86   | 0.86   | 0.80             | 0.90   | 0.97   |
| 15m                   | 0.77   | 0.76   | 0.87             | 0.93   | 1.00   |
| 25m                   | 0.76   | 0.74   | 0.90             | 0.95   | 1.01   |
| 50m                   | 0.77   | 0.73   | 0.92             | 0.97   | 1.02   |
| 100m                  | 0.81   | 0.74   | 0.98             | 0.99   | 1.03   |
</details>

(b) CIFAR10, EDM, N = 4,096  
Figure 14: Sensitivity of ICR to augmentation design. . We study the sensitivity of ICR to augmentation design by varying the strength of augmentations. We consider five augmentation levels and compute ICR under each setting. (a) In the data-abundant setting (50K training samples), the ICR curves across noise levels remain highly consistent across different augmentation strengths. (b) In the data-limited setting (4,096 training samples), the temporal evolution of ICR during training exhibits the same qualitative trend once sufficiently strong augmentations are applied. For clarity, we normalize each curve by its initial value to focus on the relative trend.

Using these augmentation pipelines, we redo the experiments of ICR across noise levels in the data-rich setting and across training dynamics in the data-limited setting, and report the results in Figure 14. As shown in the figure, once a reasonably rich augmentation pipeline is used (e.g., random crop + flip + color jitter and stronger variants), the resulting ICR trends remain highly consistent, indicating that our findings are robust to the specific choice of augmentations.

In contrast, when overly weak augmentations are used (e.g., random crop alone), the behavior becomes noticeably less stable. We conjecture that this is because the invariant-residual decomposition requires sufficiently diverse perturbations in order to meaningfully separate invariant and residual components.

Sensitivity of ICR to the choice of feature extraction layer. We evaluate the sensitivity of ICR to layer selection. Specifically, we extract features from multiple layers around the middle of the diffusion model and compute ICR for each choice. The results in Figure 15 show that ICR exhibits consistent behavior across layers in both data-abundant and data-limited settings. In particular, the overall trends and the location of the semantic window both remain stable regardless of layer choice.

## C Experimental Details

Unless stated otherwise, we apply mean pooling to obtain a single feature vector per sample. For EDM models, we average over spatial dimensions, mapping tensors of shape $N \times C \times H \times W$ to

![](images/4ac408ef0b3086f8f5bbb96781dfa7f03efa65e4b97e286631300774627c8a5b.jpg)

<details>
<summary>line chart</summary>

| Time step (σt) | 8x8_in0 | 8x8_block0 | 8x8_block1 | 16x16_block0 | 16x16_block1 |
| -------------- | ------- | ---------- | ---------- | ------------ | ------------ |
| 0.0            | 0.45    | 0.48       | 0.45       | 0.25         | 0.25         |
| 0.02           | 0.40    | 0.42       | 0.38       | 0.18         | 0.18         |
| 0.14           | 0.42    | 0.45       | 0.39       | 0.17         | 0.17         |
| 0.59           | 0.52    | 0.55       | 0.55       | 0.35         | 0.35         |
| 1.92           | 0.75    | 0.75       | 0.75       | 0.75         | 0.75         |
| >1.92          | 0.85    | 0.85       | 0.85       | 0.85         | 0.85         |
</details>

(a) CIFAR10, EDM, N = 50K

![](images/7e72d96cb9014afc5e0a3afada5542d798b935fc2def4ca8412f3534367d48df.jpg)

<details>
<summary>line chart</summary>

| Training Iter. (imgs) | 8x8_in0 | 8x8_block0 | 8x8_block1 | 16x16_block0 | 16x16_block1 |
| --------------------- | ------- | ---------- | ---------- | ------------ | ------------ |
| 2.5m                  | 0.435   | 0.445      | 0.435      | 0.375        | 0.365        |
| 7.5m                  | 0.385   | 0.390      | 0.370      | 0.295        | 0.290        |
| 15m                   | 0.405   | 0.415      | 0.400      | 0.320        | 0.315        |
| 25m                   | 0.445   | 0.450      | 0.420      | 0.335        | 0.330        |
| 50m                   | 0.455   | 0.465      | 0.435      | 0.350        | 0.345        |
| 100m                  | 0.475   | 0.480      | 0.455      | 0.375        | 0.385        |
</details>

(b) CIFAR10, EDM, N = 4,096  
Figure 15: Sensitivity of ICR to the choice of feature extraction layer. We select multiple layers around the middle of the diffusion model and compute ICR using features from each layer. (a) In the data-abundant setting (50K training samples), the trend of the ICR curves across noise levels remain highly consistent across different layer choices, with the location of the semantic window largely unchanged. (b) In the data-limited setting (4,096 training samples), the temporal evolution of ICR during training exhibits the same qualitative trend across layers, including the U-shaped behavior associated with memorization.

$N \times C$ . For transformer-based models, we average over the token dimension, mapping $N \times T \times D$ ??to $N \times D$ .

?? ??In all experiments in Section 5, we extract features at a fixed intermediate noise scale, using $\sigma _ { t } = 0 . 2 9$ for EDM based models [39] and $t = 0 . 2$ for SiT based models [40].

Figure 2. We take a 2,000-image subset of ImageNet64 and extract representations from the dec.16x16\_block1 layer of a publicly available pretrained EDM model. For each image, this gives a tensor of shape (576 16 16); we also extract representations for 14 augmented views ( hflip , , ,shift±4x , shift±4y , cropc56 , croptl5 , cropbr54 , bright±0.08 , contrast0.85 , sat0.85 , cutout22 , blur3 ), for a total of 15 views. We flatten each tensor into a 147 $4 5 6 = 5 7 6 \times 1 6 \times 1 6 -$ ,dimensional vector. For each image, we take the mean across the 15 view representations to obtain ?? (shared by the image and its augmented views), and subtract ?? from each view representation to obtain the nuisance component ?? for that view.

Finally, we pick a reference image (could be augmented) and retrieve its nearest neighbors under cosine similarity, in terms of ?? or ??. We then plot the retrieved neighbors (could be augmented) in the first and second rows, respectively.

Figure 3. We evaluate ICR and linear probing accuracy using publicly available EDM models on CIFAR10 and a SiT-XL/2 model on ImageNet $2 5 6 \times 2 5 6$ , together with a CIFAR100 EDM model that we train ourselves. For EDM backbones we extract features from the dec.16x16\_block1 layer near the bottleneck; for SiT-XL/2 we use the output of transformer block 14, the midpoint of the 28-layer network. For EDM models we train a logistic regression classifier with scikit-learn on the full set of training features and report accuracy on test features. For SiT-XL/2, due to the larger feature set, we subsample 200K training features (200 images per class) and train a linear classifier with AdamW [93] for 100 epochs (batch size 8192, learning rate $1 0 ^ { - 2 } ,$ , weight decay 10−4), reporting test accuracy at the final epoch. These hyperparameters are fixed across all noise levels and are not tuned, since our goal is to capture trends rather than optimize absolute performance. For computing ICR we use a random subset of $N = 4 0 9 6$ training features for EDM models and $N = 2 0 \mathrm { K }$ for SiT-XL/2 at ?? ??each noise level, with the subset held fixed across noise levels in each experiment.

Figure 4. We train an EDM model on CIFAR10 and a SiT-B/2 model on ImageNet $2 5 6 \times 2 5 6$ using the full training datasets and report FID together with ICR. FID is computed from 50K generated images for each experiment. As above, ICR is estimated from a subset of training features, using $N = 4 0 9 6$ samples for the EDM model and  = 20K for the SiT-B/2 model.

Figure 5. We train an EDM model on CIFAR10 using 4096 training images and report ICR together with the memorization ratio over the course of training. For the nearest neighbor visualizations on the right, we take generated samples and find their nearest neighbors among the training images directly in pixel space. For the snapshot at Training $i t e r = 2 0 0 M ,$ , we slightly cherry pick a few generated samples that are clearly memorized to highlight this effect; all other visualizations use fixed random seeds aligned with this snapshot to keep the plots consistent across training iterations.

Figure 6. We train an EDM model on ImageNet 64 × 64 using 10K training images (10 per class) and a SiT-B/2 model on ImageNet 256 × 256 using 20K training images (20 per class). Both models are trained with standard class conditional setups. When extracting features for ICR computation, to keep the procedure label-free, we encode all samples using the null class.

Figure 7. We follow the same pipeline as in Figure 2, but use checkpoints from an EDM model trained on only 10K ImageNet 64 × 64 samples. We visualize three typical phases of limited-data training: (i) early learning (first row), after 0.6M images, where the model is still improving; (ii) the onset of overfitting (second row), after 7M images, where ICR is smallest and the nearest neighbors remain meaningful and share structure with the reference; and (iii) severe overfitting (last row), after 50M images, where ICR increases and nearest neighbors are no longer semantic.

Figure 8. We reuse the EDM models trained on CIFAR10 with 4096 images and with the full 50K images to report the traces of $\Sigma _ { s }$ and $\Sigma _ { \xi }$ over training. To ensure that our notion of feature expansion ??is not conflated with simple growth in representation norms, we $\ell _ { 2 }$ normalize each representation before computing the covariances.

## D Alignment between Optimal Test Loss and ICR

In the main part of the paper, we discussed that ICR can be used as an early-stopping indicator when training diffusion models with limited data. In this section, we provide some preliminary theoretical insights for the underlying cause of such functionality. Specifically, through a simple Gaussian toy model, we show that ICR moves in the same direction as the Bayes optimal linear denoising loss.

Toy two-layer linear model. Fix a noise level $\sigma _ { t }$ and an encoder $\pmb { U } \in \mathbb { R } ^ { d \times D }$ . Let $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { D }$ denote the noisy input and

$$
\textbf {h} = \textbf {U} x _ {t} \in \mathbb {R} ^ {d}
$$

be the feature. As in Section 3, we consider the invariance-variance decomposition of the feature $\textbf { \textit { h } } = \textbf { \textit { s } } + \boldsymbol { \xi }$ where ?? is the invariant component and ?? is the variant (residual) component induced by data augmentations and additive Gaussian noise. We denote their covariances by

$$
\boldsymbol {\Sigma} _ {s} (\boldsymbol {U}) = \operatorname{Cov} (s), \quad \boldsymbol {\Sigma} _ {\xi} (\boldsymbol {U}) = \operatorname{Cov} (\xi),
$$

We reconstruct the clean image from the feature using a linear decoder $W \in \mathbb { R } ^ { D \times d } .$ ,

$$
\widehat {x} _ {0} = W h,
$$

and define the population linear denoising loss $\mathcal { L } _ { \mathrm { l i n } } ( W ; U ) : = \mathbb { E } \big \| x _ { 0 } - W h \big \| ^ { 2 }$ and thus

$$
\mathcal {L} _ {\text { lin }} ^ {\star} (U) := \min _ {W} \mathcal {L} _ {\text { lin }} (W; U) \tag {8}
$$

for the optimal linear denoising loss. For analytical clarity, we assume that $( x _ { 0 } , h )$ are jointly Gaussian.

Proposition 1 (Monotonicity of Bayes optimal loss and ICR). Let $u _ { 1 } , u _ { 2 }$ be two encoders at the same noise level $\sigma _ { t } ,$ , and denote their invariance and variance covariances by

$$
\left(\boldsymbol {\Sigma} _ {s} (\boldsymbol {U} _ {k}), \boldsymbol {\Sigma} _ {\xi} (\boldsymbol {U} _ {k})\right), \qquad k \in \{1, 2 \}.
$$

Assume the Gaussian model above holds for each encoder.

(More variant energy hurts). If the invariant covariance is the same $\Sigma _ { s } ( U _ { 1 } ) ~ = ~ \Sigma _ { s } ( U _ { 2 } )$ , and the variant covariance of $u _ { 2 }$ dominates that of $u _ { 1 }$ in PSD order,

$$
\boldsymbol {\Sigma} _ {\xi} (\boldsymbol {U} _ {1}) \preceq \boldsymbol {\Sigma} _ {\xi} (\boldsymbol {U} _ {2}),
$$

then the optimal linear denoising loss and ICR both increase:

$$
\mathcal {L} _ {\mathrm{lin}} ^ {\star} \left(\boldsymbol {U} _ {1}\right) \leq \mathcal {L} _ {\mathrm{lin}} ^ {\star} \left(\boldsymbol {U} _ {2}\right), \quad \operatorname{ICR} \left(\boldsymbol {U} _ {1}\right) \leq \operatorname{ICR} \left(\boldsymbol {U} _ {2}\right). \tag {9}
$$

(More invariant energy helps). If instead the variant covariance is the same $\Sigma _ { \xi } ( U _ { 1 } ) = \Sigma _ { \xi } ( U _ { 2 } )$ , and the invariant covariance of $u _ { 2 }$ dominates that of $u _ { 1 }$ ,

$$
\boldsymbol {\Sigma} _ {s} (\boldsymbol {U} _ {1}) \preceq \boldsymbol {\Sigma} _ {s} (\boldsymbol {U} _ {2}),
$$

then the optimal linear denoising loss and ICR both decrease:

$$
\mathcal {L} _ {\text { lin }} ^ {\star} (\boldsymbol {U} _ {1}) \geq \mathcal {L} _ {\text { lin }} ^ {\star} (\boldsymbol {U} _ {2}), \quad \text { ICR } (\boldsymbol {U} _ {1}) \geq \text { ICR } (\boldsymbol {U} _ {2}). \tag {10}
$$

Proof. We first derive the Bayes optimal loss in the original feature basis. Recall that for any measurable $g : \mathbb { R } ^ { d }  \mathbb { R } ^ { D }$ ,

$$
\mathcal {L} (g; \boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi}) = \mathbb {E} \left\| \boldsymbol {x} _ {0} - g (\boldsymbol {h}) \right\| ^ {2}.
$$

Let $h = s + \xi$ with $\pmb { s } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { s } )$ and $\xi \sim { \cal N } ( 0 , \Sigma _ { \xi } )$ independent, and $\pmb { x } _ { 0 } = A \pmb { s } + C \pmb { \xi }$ as in the statement.

We now derive the covariance of $x _ { 0 }$ and ??:

$$
\boldsymbol {\Sigma} _ {h h} := \operatorname{Cov} (\boldsymbol {h}) = \operatorname{Cov} (\boldsymbol {s}) + \operatorname{Cov} (\boldsymbol {\xi}) = \boldsymbol {\Sigma} _ {s} + \boldsymbol {\Sigma} _ {\xi},
$$

$$
\boldsymbol {\Sigma} _ {x x} := \operatorname{Cov} (\boldsymbol {x} _ {0}) = \operatorname{Cov} (\boldsymbol {A} \boldsymbol {s} + \boldsymbol {C} \boldsymbol {\xi}) = \boldsymbol {A} \boldsymbol {\Sigma} _ {s} \boldsymbol {A} ^ {\top} + \boldsymbol {C} \boldsymbol {\Sigma} _ {\xi} \boldsymbol {C} ^ {\top},
$$

$$
\boldsymbol {\Sigma} _ {x h} := \operatorname{Cov} (\boldsymbol {x} _ {0}, \boldsymbol {h}) = \operatorname{Cov} (A \boldsymbol {s} + C \boldsymbol {\xi}, \boldsymbol {s} + \boldsymbol {\xi}) = A \boldsymbol {\Sigma} _ {s} + C \boldsymbol {\Sigma} _ {\boldsymbol {\xi}}.
$$

Now consider

$$
x _ {0} - g (\pmb {h}) = \underbrace {(x _ {0} - \mathbb {E} [ x _ {0} \mid \pmb {h} ])} _ {c _ {1}} + \underbrace {\left(\mathbb {E} [ x _ {0} \mid \pmb {h} ] - g (\pmb {h})\right)} _ {c _ {2}}.
$$

With this notation, we have

$$
\begin{array}{l} \mathbb {E} \left\| \boldsymbol {x} _ {0} - g (\boldsymbol {h}) \right\| ^ {2} = \mathbb {E} \left[ \left(\boldsymbol {c} _ {1} + \boldsymbol {c} _ {2}\right) ^ {\top} \left(\boldsymbol {c} _ {1} + \boldsymbol {c} _ {2}\right) \right] \\ = \mathbb {E} \left\| x _ {0} - \mathbb {E} [ x _ {0} \mid h ] \right\| ^ {2} + \mathbb {E} \left\| \mathbb {E} [ x _ {0} \mid h ] - g (h) \right\| ^ {2} \\ + 2 \mathbb {E} \left[ (x _ {0} - \mathbb {E} [ x _ {0} \mid h ]) ^ {\top} (\mathbb {E} [ x _ {0} \mid h ] - g (h)) \right]. \\ \end{array}
$$

Then we can show the cross term vanishes:

$$
\begin{array}{l} \mathbb {E} [ \boldsymbol {c} _ {1} ^ {\top} \boldsymbol {c} _ {2} ] = \mathbb {E} \left[ \mathbb {E} [ \boldsymbol {c} _ {1} ^ {\top} \boldsymbol {c} _ {2} | \boldsymbol {h} ] \right] = \mathbb {E} \left[ \mathbb {E} \left[ (\boldsymbol {x} _ {0} - \mathbb {E} [ \boldsymbol {x} _ {0} | \boldsymbol {h} ]) ^ {\top} \boldsymbol {c} _ {2} (\boldsymbol {h}) | \boldsymbol {h} \right] \right] \\ = \mathbb {E} \left[ \mathbb {E} [ x _ {0} - \mathbb {E} [ x _ {0} \mid h ] \mid h ] ^ {\top} c _ {2} (h) \right] \\ = \mathbb {E} \left[ (\mathbb {E} [ \boldsymbol {x} _ {0} \mid \boldsymbol {h} ] - \mathbb {E} [ \boldsymbol {x} _ {0} \mid \boldsymbol {h} ]) ^ {\top} \boldsymbol {c} _ {2} (\boldsymbol {h}) \right] = 0, \\ \end{array}
$$

where in the first equality we use the law of total expectation and in the third equality we use the fact that $c _ { 2 }$ only depends on ?? and is thus constant inside the conditional expectation. Hence

$$
\mathbb {E} \left\| \boldsymbol {x} _ {0} - g (\boldsymbol {h}) \right\| ^ {2} = \mathbb {E} \left\| \boldsymbol {x} _ {0} - \mathbb {E} [ \boldsymbol {x} _ {0} \mid \boldsymbol {h} ] \right\| ^ {2} + \mathbb {E} \left\| \mathbb {E} [ \boldsymbol {x} _ {0} \mid \boldsymbol {h} ] - g (\boldsymbol {h}) \right\| ^ {2}.
$$

Therefore $\mathbb { E } \| x _ { 0 } - g ( h ) \| ^ { 2 }$ is minimized when $g ( h ) = \mathbb { E } \left[ x _ { 0 } \mid h \right]$ with minimum value E $\| \boldsymbol { x } _ { 0 } - \mathbb { E } [ \boldsymbol { x } _ { 0 } \mid h ] \| ^ { 2 }$

Now with $x _ { 0 } , h$ jointly Gaussian, and $\Sigma _ { h h }$ in $\mathrm { v e r t i b l e } ^ { 7 }$ , we have

$$
\mathbb {E} \left[ \boldsymbol {x} _ {0} \mid \boldsymbol {h} \right] = \Sigma_ {x h} \Sigma_ {h h} ^ {- 1} \boldsymbol {h}.
$$

Write $\pmb { W } = \pmb { \Sigma } _ { x h } \pmb { \Sigma } _ { h h } ^ { - 1 }$ , then

$$
\begin{array}{l} \mathcal {L} ^ {\star} \left(\boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi}\right) = \mathbb {E} \left\| \boldsymbol {x} _ {0} - \mathbb {E} [ \boldsymbol {x} _ {0} \mid \boldsymbol {h} ] \right\| ^ {2} = \mathbb {E} \left\| \boldsymbol {x} _ {0} - \boldsymbol {W h} \right\| ^ {2} \\ = \mathbb {E} \left[ \left(x _ {0} - W h\right) ^ {\top} \left(x _ {0} - W h\right) \right] \\ = \operatorname{Tr} \left(\boldsymbol {\Sigma} _ {x x}\right) - \operatorname{Tr} \left(\boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x}\right). \\ \end{array}
$$

Now what’s left to show are the two co-monotonic facts. Consider a directional perturbation $\Delta \Sigma _ { \xi } \succeq 0$ . Then

$$
\delta \boldsymbol {\Sigma} _ {x x} = \boldsymbol {C} \Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {C} ^ {\top}, \quad \delta \boldsymbol {\Sigma} _ {x h} = \boldsymbol {C} \Delta \boldsymbol {\Sigma} _ {\xi}, \quad \delta \boldsymbol {\Sigma} _ {h h} = \Delta \boldsymbol {\Sigma} _ {\xi}.
$$

Using $\delta ( \Sigma _ { h h } ^ { - 1 } ) = - \Sigma _ { h h } ^ { - 1 } ( \delta \Sigma _ { h h } ) \Sigma _ { h h } ^ { - 1 }$ and differentiating

$$
\mathcal {L} ^ {\star} = \mathrm{Tr} (\pmb {\Sigma} _ {x x}) - \mathrm{Tr} (\pmb {\Sigma} _ {x h} \pmb {\Sigma} _ {h h} ^ {- 1} \pmb {\Sigma} _ {h x}),
$$

We can then calculate

$$
\begin{array}{l} \delta \mathcal {L} ^ {\star} = \mathrm{Tr} \left(\delta \pmb {\Sigma} _ {x x}\right) - \mathrm{Tr} \left(\delta \pmb {\Sigma} _ {x h} \pmb {\Sigma} _ {h h} ^ {- 1} \pmb {\Sigma} _ {h x}\right) - \mathrm{Tr} \left(\pmb {\Sigma} _ {x h} \delta \left(\pmb {\Sigma} _ {h h} ^ {- 1}\right) \pmb {\Sigma} _ {h x}\right) - \mathrm{Tr} \left(\pmb {\Sigma} _ {x h} \pmb {\Sigma} _ {h h} ^ {- 1} \delta \pmb {\Sigma} _ {h x}\right) \\ = \mathrm{Tr} \left(\boldsymbol {C} \Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {C} ^ {\top}\right) - \mathrm{Tr} \left(\boldsymbol {C} \Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x}\right) \\ - \operatorname{Tr} \left(\boldsymbol {\Sigma} _ {x h} \left(- \boldsymbol {\Sigma} _ {h h} ^ {- 1} \Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {\Sigma} _ {h h} ^ {- 1}\right) \boldsymbol {\Sigma} _ {h x}\right) - \operatorname{Tr} \left(\boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1} \Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {C} ^ {\top}\right) \\ = \operatorname{Tr} \left(\boldsymbol {C} \Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {C} ^ {\top}\right) - \operatorname{Tr} \left(\boldsymbol {C} \Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x}\right) \\ + \mathrm{Tr} \left(\pmb {\Sigma} _ {x h} \pmb {\Sigma} _ {h h} ^ {- 1} \Delta \pmb {\Sigma} _ {\xi} \pmb {\Sigma} _ {h h} ^ {- 1} \pmb {\Sigma} _ {h x}\right) - \mathrm{Tr} \left(\pmb {\Sigma} _ {x h} \pmb {\Sigma} _ {h h} ^ {- 1} \Delta \pmb {\Sigma} _ {\xi} \pmb {C} ^ {\top}\right). \\ \end{array}
$$

Using cyclicity of the trace to move $\Delta \pmb { \Sigma } _ { \xi }$ to the left in each term,

$$
\begin{array}{l} \delta \mathcal {L} ^ {\star} = \operatorname{Tr} \left(\Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {C} ^ {\top} \boldsymbol {C}\right) - \operatorname{Tr} \left(\Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x} \boldsymbol {C}\right) \\ + \operatorname{Tr} \left(\Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x} \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1}\right) - \operatorname{Tr} \left(\Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {C} ^ {\top} \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1}\right) \\ = \operatorname{Tr} \left(\Delta \boldsymbol {\Sigma} _ {\xi} \left[ \boldsymbol {C} ^ {\top} \boldsymbol {C} - \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x} \boldsymbol {C} + \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x} \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1} - \boldsymbol {C} ^ {\top} \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1} \right]\right). \\ \end{array}
$$

Now define

$$
\boldsymbol {M} := \boldsymbol {C} - \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1}.
$$

Then

$$
\begin{array}{l} \boldsymbol {M} ^ {\top} \boldsymbol {M} = \left(\boldsymbol {C} - \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1}\right) ^ {\top} \left(\boldsymbol {C} - \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1}\right) \\ = \left(\boldsymbol {C} ^ {\top} - \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x}\right) \left(\boldsymbol {C} - \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1}\right) \\ = \boldsymbol {C} ^ {\top} \boldsymbol {C} - \boldsymbol {C} ^ {\top} \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1} - \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x} \boldsymbol {C} + \boldsymbol {\Sigma} _ {h h} ^ {- 1} \boldsymbol {\Sigma} _ {h x} \boldsymbol {\Sigma} _ {x h} \boldsymbol {\Sigma} _ {h h} ^ {- 1}, \\ \end{array}
$$

which matches the bracket above. Hence

$$
\delta \mathcal {L} ^ {\star} \left(\boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi}\right) = \operatorname{Tr} \left(\Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {M} ^ {\top} \boldsymbol {M}\right).
$$

Since $\Delta \Sigma _ { \xi } \succeq 0$ and $M ^ { \top } M \succeq 0$ , the product inside the trace is positive semidefinite and hence

$$
\delta \mathcal {L} ^ {\star} \left(\boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi}\right) = \operatorname{Tr} \left(\Delta \boldsymbol {\Sigma} _ {\xi} \boldsymbol {M} ^ {\top} \boldsymbol {M}\right) \geq 0.
$$

Therefore, consider $\pmb { \Sigma } _ { \xi , 1 } , \pmb { \Sigma } _ { \xi , 2 } \geq \mathbf { 0 }$ with $\pmb { \Sigma } _ { \xi , 1 } \leq \pmb { \Sigma } _ { \xi , 2 }$ , and define

$$
\boldsymbol {\Sigma} _ {\xi} (t) := \boldsymbol {\Sigma} _ {\xi , 1} + t \left(\boldsymbol {\Sigma} _ {\xi , 2} - \boldsymbol {\Sigma} _ {\xi , 1}\right), \quad t \in [ 0, 1 ],
$$

and $\phi ( t ) = \mathcal { L } ^ { \star } \left( \Sigma _ { s } , \Sigma _ { \xi } ( t ) \right)$ . Then

$$
\phi^ {\prime} (t) = \operatorname{Tr} \left(\left(\boldsymbol {\Sigma} _ {\xi , 2} - \boldsymbol {\Sigma} _ {\xi , 1}\right) \boldsymbol {M} ^ {\top} \boldsymbol {M}\right) \geq 0,
$$

i.e., ??( ) is non-decreasing in  and therefore

$$
\mathcal {L} ^ {\star} (\Sigma_ {s}, \Sigma_ {\xi , 1}) \leq \mathcal {L} ^ {\star} (\Sigma_ {s}, \Sigma_ {\xi , 2}).
$$

Now we consider ICR. Fix ?? as in the statement and define

$$
\widetilde {\boldsymbol {\Sigma}} _ {s} := \boldsymbol {V} ^ {\top} \boldsymbol {\Sigma} _ {s} \boldsymbol {V}, \quad \widetilde {\boldsymbol {\Sigma}} _ {\xi} := \boldsymbol {V} ^ {\top} \boldsymbol {\Sigma} _ {\xi} \boldsymbol {V}.
$$

By cyclicity of the trace,

$$
\operatorname{ICR} \left(\boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi}\right) = \frac {\operatorname{Tr} \left(\boldsymbol {V} ^ {\top} \boldsymbol {\Sigma} _ {\xi} \boldsymbol {V}\right)}{\operatorname{Tr} \left(\boldsymbol {V} ^ {\top} \left(\boldsymbol {\Sigma} _ {s} + \boldsymbol {\Sigma} _ {\xi}\right) \boldsymbol {V}\right)} = \frac {\operatorname{Tr} \left(\widetilde {\boldsymbol {\Sigma}} _ {\xi}\right)}{\operatorname{Tr} \left(\widetilde {\boldsymbol {\Sigma}} _ {s} + \widetilde {\boldsymbol {\Sigma}} _ {\xi}\right)}.
$$

Fix $\Sigma _ { s }$ and consider the same path $\Sigma _ { \xi } ( t ) = \Sigma _ { \xi , 1 } + t ( \Sigma _ { \xi , 2 } - \Sigma _ { \xi , 1 } )$ , and set

$$
\widetilde {\boldsymbol {\Sigma}} _ {\xi} (t) := \boldsymbol {V} ^ {\top} \boldsymbol {\Sigma} _ {\xi} (t) \boldsymbol {V} = \boldsymbol {V} ^ {\top} \boldsymbol {\Sigma} _ {\xi , 1} \boldsymbol {V} + t \boldsymbol {V} ^ {\top} (\boldsymbol {\Sigma} _ {\xi , 2} - \boldsymbol {\Sigma} _ {\xi , 1}) \boldsymbol {V},
$$

$$
\alpha (t) := \operatorname{Tr} (\widetilde {\boldsymbol {\Sigma}} _ {\xi} (t)), \quad \beta := \operatorname{Tr} (\widetilde {\boldsymbol {\Sigma}} _ {s}) > 0.
$$

Then

$$
\operatorname{ICR} (t) := \operatorname{ICR} (\boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi} (t)) = \frac {\alpha (t)}{\alpha (t) + \beta}.
$$

Differentiating,

$$
\alpha^ {\prime} (t) = \operatorname{Tr} \left(\boldsymbol {V} ^ {\top} \left(\boldsymbol {\Sigma} _ {\xi , 2} - \boldsymbol {\Sigma} _ {\xi , 1}\right) \boldsymbol {V}\right) = \operatorname{Tr} \left(\left(\boldsymbol {\Sigma} _ {\xi , 2} - \boldsymbol {\Sigma} _ {\xi , 1}\right) \boldsymbol {V} \boldsymbol {V} ^ {\top}\right) \geq 0,
$$

because $\pmb { \Sigma } _ { \xi , 2 } - \pmb { \Sigma } _ { \xi , 1 } \succeq \mathbf { 0 }$ and $V V ^ { \top } \succ 0$ imply

$$
\operatorname{Tr} \left((\boldsymbol {\Sigma} _ {\xi , 2} - \boldsymbol {\Sigma} _ {\xi , 1}) \boldsymbol {V V} ^ {\top}\right) = \operatorname{Tr} \left((\boldsymbol {V V} ^ {\top}) ^ {1 / 2} (\boldsymbol {\Sigma} _ {\xi , 2} - \boldsymbol {\Sigma} _ {\xi , 1}) (\boldsymbol {V V} ^ {\top}) ^ {1 / 2}\right) \geq 0.
$$

Hence

$$
\mathrm{ICR} ^ {\prime} (t) = \frac {\beta \alpha^ {\prime} (t)}{(\alpha (t) + \beta) ^ {2}} \geq 0,
$$

so ICR( ) is non-decreasing in  and we obtain

$$
\operatorname{ICR} \left(\boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi , 1}\right) \leq \operatorname{ICR} \left(\boldsymbol {\Sigma} _ {s}, \boldsymbol {\Sigma} _ {\xi , 2}\right).
$$

which completes the proof.