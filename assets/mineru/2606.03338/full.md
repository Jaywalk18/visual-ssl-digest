# IDES T: Assessing Self-Supervised Learning Representations via Intrinsic Dimension

Julie Mordacq 1 2 Vicky Kalogeiton 2 Steve Oudot 1 2

# Abstract

Self-supervised learning (SSL) has emerged as a powerful paradigm for learning meaningful representations from unlabeled data. However, the standard protocol for evaluating these representations, linear probing, is computationally expensive, sensitive to hyperparameters, and provides limited insight into the geometric structure of the representation space. In this work, motivated by connections between neural network generalization and intrinsic dimension (ID) we propose IDES T, a method for estimating the ID of SSL representations via the Minimum Spanning Tree dimension estimator (dimMST). Across diverse datasets, architectures, and SSL pretraining objectives, we show that IDES T strongly correlates with downstream linear probe performances. Furthermore, we demonstrate that IDES T enables efficient hyperparameter selection, significantly reducing the computational cost compared to supervised alternatives. Our results highlight intrinsic dimensionality as a principled geometric proxy for assessing SSL representations, complementing standard supervised probing protocols.

# 1. Introduction

Can the geometry of learned representations provide reliable, label-free insights into their quality for downstream tasks? Self-supervised learning (SSL) offers a natural setting to ask this question, as its objectives are explicitly designed to structure representations without access to labels (Chen et al., 2020; Bardes et al., 2022; Assran et al., 2023; Venkataramanan et al., 2025; Simeoni et al. ´ , 2025; Mordacq et al., 2025). Beyond their strong performance, SSL representations are valued for their ability to transfer to a wide range of downstream tasks with minimal super-

1Inria Saclay 2LIX, CNRS, Ecole Polytechnique, IP Paris. Cor- ´ respondence to: Julie Mordacq <julie.mordacq@inria.fr>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

vision (Dufour et al., 2025; Couairon et al., 2025; Maruani et al., 2025; Degeorge et al., 2025). Rather than optimize task-specific decision boundaries, SSL methods shape the organization of data in the representation space through geometric constraints such as alignment between views, feature uniformity, and variance control. As a result, intrinsic properties of the resulting manifold, e.g., curvature and spectral structure, have emerged as potential indicators of representation quality, prompting recent efforts to investigate such geometric proxies (Ansuini et al., 2019; Garrido et al., 2023).

Among them, intrinsic dimension (ID), originally introduced by Bennett (1969), has emerged as a particularly informative quantity: it characterizes the effective number of degrees of freedom required to represent data in an embedding space. Recent studies have revealed a monotonic relationship between a model’s generalization error and the intrinsic dimension of its representations (Ansuini et al., 2019; Konz & Mazurowski, 2024), with lower intrinsic dimension often correlating with improved downstream accuracy. These findings, grounded in the manifold hypothesis (Goodfellow et al., 2016), suggest that representation quality is governed not only by separability, but also by how efficiently information is compressed into low-dimensional geometric structure. Yet, existing evidence is largely confined to supervised convolutional networks, leaving open how intrinsic dimension behaves across modern self-supervised methods and whether it indicates downstream performance in this setting.

In practice, estimating ID is far from trivial. Intrinsic dimension admits multiple mathematical formalizations, topological, fractal, and information-theoretic, each capturing different facets of data geometry, and since ID cannot be observed directly but must be estimated from finite samples, the choice of estimator matters. Most notably, nearest-neighborbased methods such as TwoNN (Facco et al., 2017) and maximum likelihood estimators (Levina & Bickel, 2004) have been widely adopted, but suffer from well-known limitations: They rely on strong locality and isotropy assumptions, are sensitive to noise and finite-sample effects, and become unstable in high-dimensional or highly structured representation spaces. These limitations are especially pronounced in SSL, which requires operating far from standard conditions. First, it operates in a non-asymptotic regime where n ≈ d rather than $n  \infty$ with d fixed, where n is the number of samples and d is the ambient dimension. Second, SSL objectives introduce dependencies between data points, violating standard independence assumptions. For instance, as shown in Figure 2, TwoNN and MLE are sensitive to the latter; they are unable to reliably detect the intrinsic dimension, with TwoNN even diverging.

<table><tr><td>BarlowTwins</td><td>DINO</td><td>DINOv3</td><td>EVA-CLIP</td><td>I-JEPA</td><td>SigLIP</td><td>VICReg</td></tr><tr><td>CLIP</td><td>DINOv2</td><td>EVA-02-CLIP</td><td>Franca</td><td>PE</td><td>SigLIP2</td><td>iBOT</td></tr></table>

![](images/435e0114d901310de97b7867da8d1fe2d90e24fb842a92d9d97dfaf61af87605.jpg)

<details>
<summary>bubble</summary>

| IdEst ImageNet | Top-1 Accuracy (%) |
| -------------- | ------------------ |
| 12             | 87                 |
| 14             | 86                 |
| 16             | 85                 |
| 18             | 84                 |
| 20             | 83                 |
| 22             | 82                 |
| 24             | 81                 |
</details>

![](images/7ea52ccc54724d6c82acabd1df7f419301192c48701e0d0fdd426089a9208553.jpg)

<details>
<summary>scatter</summary>

| IdEst iNat-18 | Top-1 Accuracy (%) |
| ------------- | ------------------ |
| 12            | 75                 |
| 15            | 80                 |
| 17            | 65                 |
| 20            | 60                 |
| 22            | 55                 |
| 25            | 50                 |
| 27            | 45                 |
</details>

![](images/2b5740a3b5169ee1abaf81bb825271cb7ad3773b12fe54afae19f4e68961d507.jpg)

<details>
<summary>scatter</summary>

| IdEst iNat-21 | Top-1 Accuracy (%) |
| ------------- | ------------------ |
| 12            | 90                 |
| 15            | 85                 |
| 17            | 80                 |
| 20            | 75                 |
| 22            | 70                 |
| 25            | 65                 |
| 27            | 60                 |
| 30            | 55                 |
</details>

![](images/c7bbabe2f4fed77fbb419a4402119331cef3370188ec1a6a975ef5338327adbc.jpg)

<details>
<summary>bubble</summary>

| IdEst SUN397 | Top-1 Accuracy (%) |
| ------------ | ------------------ |
| 14           | 82                 |
| 16           | 83                 |
| 18           | 85                 |
| 20           | 84                 |
| 22           | 82                 |
| 24           | 80                 |
| 26           | 76                 |
| 28           | 72                 |
</details>

Figure 1. Foundation Models and IDES T: Intra-Dataset Correlation. Linear probe accuracy of pretrained SSL models on ImageNet (left), iNat-18 (middle left), iNat-21 (middle right), SUN397 (right) versus IDES T on each respective dataset. Each point corresponds to a model checkpoint; point size reflects the number of parameters. We report Kendall’s $\tau \stackrel { \bar { \in } } { \in } [ - 1 , 1 ]$ and Spearman’s $\rho \in [ \bar { - } 1 , 1 ]$ . Correlations across all four benchmarks demonstrate IDES T’s ability to provide insights into models’ representation quality.

In this paper, we consider a complementary ID estimator grounded in the scaling behavior of minimum spanning trees (Costa & Hero, 2006). Asymptotically tied to the intrinsic Renyi entropy, this estimator balances local and global in-´ formation, enjoying robustness to noise, to sampling-density variations, and to high ambient dimensionality. Furthermore, its reliability under sampling sparsity, a behavior consistent with known theoretical results on MST length in sparse regimes (e.g., Theorem 1 in (Mordacq et al., 2025)), makes it particularly well-suited to the n ≈ d regime inherent in vision SSL.

Building on this perspective, we propose IDES T (pronounced as id est, Latin for ‘that is’), standing for ID Estimation for SSL using minimum spanning Tree. IDES T is an unsupervised criterion for evaluating self-supervised representations based on intrinsic dimension estimation via minimum spanning trees. We show that IDES T strongly correlates with downstream performance across a wide range of self-supervised objectives, including joint-embedding, joint-predictive and vision-language alignment (Figures 1, 3 and 4). Moreover, IDES T provides an efficient proxy to supervised linear probing, enabling practical hyperparameter selection without labels at a fraction of the computational cost of supervised probing.

Our main contributions can be summarized as follows:

i) We propose IDES T, an unsupervised criterion based on the intrinsic dimension of the representation (Section 3).   
ii) We demonstrate that IDES T serves as an efficient proxy to assess the quality of SSL representations (Section 4.1).   
iii) We show that IDES T enables unsupervised hyperparameter selection across diverse SSL objectives and architectures (Sections 4.2 and 4.3).

Beyond providing empirical insights, our work underscores the promise of geometric descriptors as a complement to traditional supervised evaluation methods.

# 2. Related Work

Our work builds on recent efforts to analyze geometric properties of deep neural networks (DNNs) in a label-free setting, and to understand their connection to generalization. Existing literature investigates these properties through two main lenses: spectral properties and intrinsic dimensionality.

# 2.1. Representation Spectrum

α-ReQ (Agrawal et al., 2022) and RankMe (Garrido et al., 2023) characterize the eigenspectrum of representations within SSL frameworks, by measuring the decay rate of empirical eigenvalues or calculating the effective rank of the representations, respectively. These studies showed that both metrics often strongly correlate with downstream performance and highlighted their utility for hyperparameter selection.

However, the power-law assumptions underlying α-ReQ are violated in the presence of representation collapse (He & Ozay, 2022) (where networks output identical or noninformative vectors regardless of the input, resulting in rank-deficient representations), leading to unreliable performance estimates when the embedding space becomes rank-deficient (Garrido et al., 2023). Furthermore, RankMe is limited to the study of Joint-Embedding Architectures (JEAs) where two networks are trained to produce similar embeddings for different views of the same image. Notably, RankMe is tailored to identify a pivotal challenge of JEAs: representation collapse (Jing et al., 2021). This narrow focus potentially limits the applicability of such metrics to other SSL paradigms less prone to this specific type of collapse. For instance, as shown in Table 1, RankMe is less effective on I-JEPA, a joint-predictive method.

More recently, Thilak et al. (2024) proposed LiDAR, which quantifies the rank of the Linear Discriminant Analysis matrix associated with the surrogate SSL task, a measure that intuitively captures how much information the representation retains for solving that task. Crucially, LiDAR leverages SSL pretraining information (i.e., augmentations), whereas our work targets the harder setting, where only frozen representations are accessible, without any access to training data or pairing structure, aiming to gain a deeper understanding of SSL models beyond hyperparameter selection.

# 2.2. Intrinsic dimension

Intrinsic dimension (ID) has been linked to the generalization of DNNs through two primary lenses: the optimization trajectory, analyzing the sequence of model states and parameter updates during training (Simsekli et al., 2020; Birdal et al., 2021; Dupuis et al., 2023; Tan et al., 2024), and the learned representations (Ansuini et al., 2019; Konz & Mazurowski, 2024; Ruppik et al., 2025). In this work, we focus on the latter. Analyzing optimization trajectories is often computationally expensive, as it requires processing all network parameters (which can reach hundred of millions in current vision models, e.g., ViT-L (Dosovitskiy et al., 2021)) and necessitates access to intermediate training checkpoints that are rarely available for large-scale, pre-trained models.

Regarding representational analysis, several studies have investigated the ID of Large Language Model (LLM) representations (Aghajanyan et al., 2021; Cai et al., 2021; Tulchinskii et al., 2023; Valeriani et al., 2023; Viswanathan et al., 2025; Lee et al., 2025; Ruppik et al., 2025) demonstrating that ID provides critical insights into training dynamics and generalization. However, research on computer vision models has largely remained restricted to supervised convolutional neural networks (CNNs). For instance, Ansuini et al. (2019) showed that the ID of supervised CNNs correlates with performance, while Konz & Mazurowski (2024) proposed a generalization scaling law based on representational ID, though their validation was limited to supervised CNN architectures.

Despite these advances, the use of intrinsic dimension estimations to characterize representation quality remains largely unexplored in self-supervised learning.

# 3. Generalization and Dimension Estimation

This section first introduces the theoretical connection between intrinsic dimension and generalization (Section 3.1). Second, it presents standard intrinsic dimension estimators used in prior studies of deep neural network representations, along with their limitations, and motivates the use of an alternative: dimMST (Section 3.2).

# 3.1. Theoretical connection to generalization

Consider a classification dataset D, consisting of $N _ { D }$ points $x \in \mathbb { R } ^ { n }$ with target labels $y = \mathcal { F } ( x )$ defined by an unknown function $\mathcal { F } : \mathbb { R } ^ { n }  \mathbb { R } ^ { C }$ , where C is the number of classes. The dataset is split into a training set $\mathcal { D } _ { \mathrm { t r a i n } }$ and a test set $\mathcal { D } _ { \mathrm { t e s t } }$ . We analyze ‘well-trained’ models $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { C }$ that interpolate the training data, such that $f ( x ) = { \mathcal { F } } ( x )$ for all $x \in \mathcal { D } _ { \operatorname { t r a i n } } .$ . Let $\mathcal { L }$ be a non-negative loss function (e.g., cross-entropy) satisfying $\mathcal { L } ( f ( x ) , \mathcal { F } ( x ) ) = 0$ if $f ( x ) = { \mathcal { F } } ( x )$ . The generalization error is expressed as the expected loss over $\mathcal { D } _ { \mathrm { t e s t } }$ . The model can be decomposed as $f = h \circ g .$ , where $g$ is an encoder (e.g., a pre-trained SSL backbone) that produces latent representations living on some d-dimensional manifold, and where h is a classification head.

Theorem 3.1. Konz & Mazurowski (2024) Let $K _ { L }$ the Lipschitz constant of the loss function. Then:

$$
\mathcal {L} \simeq \mathcal {O} \left(K _ {L} N _ {D} ^ {- 1 / d}\right) \tag {1}
$$

This relation suggests that, for a fixed dataset size $N _ { D }$ , the error is dominated by the $- \frac { 1 } { d }$ exponent. Consequently, a lower d implies a more efficient representation. We therefore use the estimated ID of the representations as an unsupervised criterion to assess the quality of the downstream task.

# 3.2. Intrinsic dimension estimators

The intrinsic dimension (ID) estimation problem assumes the data points are sampled on (or close to) some unknown d-submanifold of the ambient space. The goal is to estimate d from the data.

# 3.2.1. PARAMETRIC ESTIMATORS

While many estimators exist (Johnsson et al., 2014; Tempczyk et al., 2022; Binnie et al., 2025), two main estimators have been adopted in prior studies of deep neural network representations (Ansuini et al., 2019; Pope et al., 2021; Konz & Mazurowski, 2024; Ruppik et al., 2025):

![](images/f2589a3934abb34e9e91694e4c2f8a7f874cb62742314554f399e0b1fd759d78.jpg)

<details>
<summary>line</summary>

| Number of samples | 1-d Helical Manifold | Ground Truth | TwoNN | dim_MST | MLE |
| ----------------- | --------------------- | ------------ | ----- | ------- | --- |
| 40                | ~1.5                  | 1.0          | ~1.3  | ~1.4    | ~1.7 |
| 80                | ~1.2                  | 1.0          | ~1.4  | ~1.0    | ~1.6 |
| 120               | ~1.0                  | 1.0          | ~1.7  | ~1.0    | ~1.3 |
| 160               | ~1.0                  | 1.0          | ~2.5  | ~1.0    | ~1.2 |
| 200               | ~1.0                  | 1.0          | ~3.0  | ~1.0    | ~1.2 |
</details>

Figure 2. Impact of Sampling Distribution on Estimators. (Left) 200 points sampled evenly along a 1-dimensional helix. (Right) Estimated ID as a function of sample size. While dimMST converges accurately to the ground truth $d = 1$ as the sample size increases, MLE and TwoNN do not, with TwoNN even diverging to infinity.

Maximum Likelihood Estimation (MLE) (Levina & Bickel, 2004) and TwoNN (Facco et al., 2017). Both are parametric estimators, founded on the assumption that the data points are sampled i.i.d. from a probability distribution supported on the submanifold, with locally constant density, and both treat the data points locally as a homogeneous Poisson process. Under these conditions, the number of points within a small ε-ball, and the ratio between the distances to the second and first nearest neighbors, follow specific parametric distributions whose parameters depend directly on d and can be inferred, either via maximum likelihood estimation for MLE or via regression for TwoNN.

These methods are inherently tied to specific data distributions, and they become unstable as the input data deviate from those, even in simple cases. Consider, for instance, the sample of Figure 2, composed of up to 200 points evenly spaced along a 1-dimensional helix embedded in $\mathbb { R } ^ { 3 }$ . Despite the high regularity of both the submanifold and the sample, both methods fail to recover the correct intrinsic dimension, with TwoNN even diverging to infinity in the asymptotic regime. This behavior is consistent across standard choices of hyperparameters of the methods. In practice, this instability translates into a degradation of the performances in the presence of noise (Tulchinskii et al., 2023; Binnie et al., 2025), or on heavy-tailed distributions (Birdal et al., 2021).

# 3.2.2. METRIC INVARIANTS BASED ESTIMATORS

The limitations of the parametric methods TwoNN and MLE motivate a shift toward estimators based on the theory of Euclidean functionals (Yukich, 2006). In particular, the asymptotic growth rate of the length of Minimum Spanning Tree (MST) are related to the Renyi entropy of the ´ underlying distribution. This connection has enabled the derivation of several dimension estimators with proven consistency under relatively weak assumptions: compactness of the manifold and boundedness of the Lebesgue sampling density supported on the manifold (Costa & Hero, 2006). For instance, the Minimum Spanning Tree dimension estimator, or $\dim _ { \operatorname { \mathrm { M S T } } }$ for short, successfully recovers the ground truth dimension $d = 1$ in the example of Figure 2.

Given a point cloud $Z \mathrm { i n } \mathbb { R } ^ { D }$ , the Minimum Spanning Tree (MST) is the acyclic connected graph $G = ( V , E )$ , with vertex set $V = Z _ { \mathrm { { \scriptsize { ~ 2 ~ } } } }$ that minimizes the total edge length:

$$
L (G) = \sum_ {(z, z ^ {\prime}) \in E} \| z - z ^ {\prime} \| _ {2}. \tag {2}
$$

Costa & Hero (2006) studied the growth rate of the length of the minimum spanning tree for random point clouds in Riemannian manifolds. Given an i.i.d. n-sample $X _ { n }$ drawn from a fixed probability measure $P _ { X }$ supported on a compact Riemannian d-manifold M with density $f _ { X }$ w.r.t. the Hausdorff measure, there exists a constant $C ^ { \prime }$ independent of $f _ { X }$ and of M such that, almost surely:

$$
n ^ {- \frac {d - 1}{d}} \cdot L \left(\mathrm{MST} (X _ {n})\right) \xrightarrow [ n \to \infty ]{} C ^ {\prime} \int f _ {X} ^ {\frac {d - 1}{d}} \mathrm{d} \mathcal {H}, \tag {3}
$$

where H denotes the Hausdorff measure on M.

This result motivates the definition of dimMST given by Costa & Hero (2006):

Definition 3.2. Given a bounded metric space M , the MST dimension of M , denoted by dimMST(M ), is the infimal exponent d ∈ N such that $L \left( \operatorname { M S T } ( X ) \right) / \left| X \right| ^ { \frac { d - 1 } { d } }$ is uniformly bounded for all finite subsets $X \subseteq M !$ :

$$
\dim_ {\mathrm{MST}} (M) := \inf \left\{d: \exists C \text {   s.t.   } \frac {L (\operatorname{MST} (X))}{| X | ^ {\frac {d - 1}{d}}} \leq C \right.
$$

for every finite subset X of M o.

In practice, the ID is estimated via log-log linear regression (Birdal et al., 2021; Binnie et al., 2025). Specifically, given subsamples $X _ { n _ { i } }$ with increasing sizes $n _ { i } ,$ we fit:

$$
\log (L (\mathrm{MST} (X _ {n _ {i}}))) \approx \frac {d - 1}{d} \log (n _ {i}) + \log (C).
$$

The resulting slope m yields the intrinsic dimension via the relation $d = 1 / ( 1 - m )$ . The complete algorithm for computing dimMST is given in Algorithm 1.

Persistent Homology Dimension. In Topological Data Analysis (TDA), the MST relates to the so-called total degree-0 persistence of the Rips filtration (Oudot, 2015). This connection allows for the definition of the 0-dimensional Persistent Homology (PH) dimension, dim0PH, 0 which is equivalent to the dimMST (Adams et al., 2020). The PH dimension has been used in several dimensionestimation applications (Birdal et al., 2021; Dupuis et al., 2023). This connection provides further theoretical grounding for the robustness of dimMST. In particular, TDA-based measures such as the total persistence of the Rips filtration are provably stable under pertubations of the underlying distribution (Chazal et al., 2014). This was further observed empirically by Tulchinskii et al. (2023) in the context of LLM latent spaces.

# 4. IDES T: dimMST for Unsupervised Assessment of Self-Supervised methods

In this section, we apply dimMST to estimate the intrinsic dimension of self-supervised representations. This yields IDES T, which stands for Intrinsic Dimension Estimation for SSL using minimum spanning Trees. Our goal is to determine whether IDES T can yield unsupervised insights into downstream performances. Specifically, to empirically validate IDES T, we compare it against linear probing, the standard evaluation protocol for self-supervised learning (SSL).

Subsequently, we address the following research questions:

Q1. To what extent does IDES T reflect representation quality across pretrained SSL models?   
Q2. Can IDES T yield insight along self-supervised pretraining?   
Q3. Can IDES T be leveraged as a principled proxy for hyperparameter selection without requiring labeled data?

# Implementation of IDES T

To satisfy the assumptions of Theorem 3.1, IDES T operates on the frozen representation passed to the classifier head, following each method’s standard evaluation protocol (i.e., consistent with how linear probes are trained). For models without a class token (e.g., I-JEPA), we average-pool the patch tokens; for models with a [CLS] token (e.g., DINO, DINOv2), we use this token directly. Additional implementation details are provided in Appendix A.

# 4.1. IDES T and accuracy of foundation models

We begin by evaluating whether our proposed IDES T reflects representation quality across a diverse set of pretrained SSL models. Specifically, we compute IDES T on frozen representations and compare it against standard linear probing accuracy. We compute two rank correlation statistics: Spearman’s rank correlation coefficient ρ (Spearman, 1961) and Kendall’s rank correlation coefficient τ (Kendall, 1938). Both capture monotonic relationships between rankings.

Setup. We evaluate 14 SSL methods spanning four paradigms: pure joint-embedding (e.g., VICReg (Bardes et al., 2022), DINO (Caron et al., 2021)), joint-predictive (e.g., I-JEPA (Assran et al., 2023)), combined objectives (e.g., iBOT (Zhou et al., 2022), DINOv2 (Oquab et al., 2023)), and vision-language alignment (e.g., CLIP (Radford et al., 2021), EVA-CLIP (Sun et al., 2023)).

![](images/44623d4a956d02bc681d708bc1fd5ce787ae5b7fdd444cc3ca91ebda234ee98d.jpg)  
Figure 3. Foundation Models and IDES T: Inter-Dataset Correlation. Linear probe accuracy of pretrained SSL models on iNat-18 (top right), iNat-21 (top left), CIFAR-10 (bottom right), CIFAR-100 (bottom left) versus IDES T computed on ImageNet. Strong correlations demonstrate that IDES T computed on a single reference dataset: ImageNet, is indicative of model quality across datasets.

For each method, we evaluate two main architectures: ResNet (He et al., 2016) and ViT (Dosovitskiy et al., 2021), and we include various model scales (e.g., ViT-S, and ViT-G). This results in 33 different models. Table 3 (in supplementary) provides the complete list of models and architectures studied.

Results. We first examine Intra-Dataset Correlation in Figure 1, where IDES T is estimated on each dataset separately and compared against the corresponding linear probe accuracy. Figure 1 reports results on ImageNet (Deng et al., 2009) and three additional fine-grained datasets: iNat-18 (Van Horn et al., 2018), iNat-21 (Van Horn et al., 2021), and SUN397 (Xiao et al., 2010). Each point represents a pretrained SSL model, spanning diverse architectures, training objectives, and scales.

![](images/1d0c72d021259f465e1d5b9813b72bcccaf40c997f763f132089bc3c715f9c8e.jpg)

<details>
<summary>scatter</summary>

| Model       | Value |
|-------------|-------|
| BarlowTwins | ●     |
| CLIP        | ●     |
| DINO        | ●     |
| DINOv2      | ●     |
| EVA-02-CLIP | ●     |
| EVA-CLIP    | ●     |
| Franca      | ●     |
| I-JEPA      | ●     |
| PE          | ●     |
| SigLIP      | ●     |
| VICReg      | ●     |
| iBOT        | ●     |
</details>

![](images/3606ea45c289406437f6bea5514b73fcda8357d4404095d74f488af642bc1a2e.jpg)

<details>
<summary>scatter</summary>

| IdEst ImageNet | Top-1 Accuracy (%) |
| -------------- | ------------------ |
| 12             | 84                 |
| 14             | 83                 |
| 16             | 82                 |
| 18             | 81                 |
| 20             | 79                 |
| 22             | 77                 |
| 24             | 75                 |
</details>

![](images/3d2409c51b526200513fbae16c47dd6733aecdc6a24ed8ab2fa2414a9f5b653d.jpg)

<details>
<summary>bubble</summary>

| IdEst ImageNet | Top-1 Accuracy (%) |
| -------------- | ------------------ |
| 12             | 78                 |
| 13             | 80                 |
| 14             | 75                 |
| 15             | 76                 |
| 16             | 74                 |
| 17             | 73                 |
| 18             | 72                 |
| 19             | 71                 |
| 20             | 68                 |
| 21             | 67                 |
| 22             | 66                 |
| 23             | 65                 |
| 24             | 52                 |
</details>

Figure 4. Foundation Models and IDES T: Alternative Evaluation Protocol. Accuracy under alternative evaluation settings versus IDES T. Strong correlations demonstrate that IDES T computed on a single reference dataset: ImageNet, is indicative of model quality across evaluation protocols.

Across all datasets, Figure 1 reveals a consistent negative correlation between IDES T and downstream linear probing accuracy: models with lower intrinsic dimension tend to achieve higher performance. This trend holds on ImageNet (left) as well as on the fine-grained benchmarks iNat-18 (middle left), iNat-21 (middle right), and SUN397 (right). Notably, the relationship is consistent across a wide range of SSL paradigms, joint-embedding methods, joint-predictive methods, and vision-language pretraining, suggesting that intrinsic dimension acts as a unifying geometric descriptor of representation quality, independent of the training objective.

This is further supported by the correlation metrics reported atop each plot. Both Kendall’s τ ≈ −0.6 and Spearman’s ρ ≈ −0.8 confirm that IDES T reliably preserves the relative ordering of models across all four datasets, in strong agreement with linear probing rankings.

While our analysis has first focused on Intra-Dataset Correlation, a natural question arises: do these findings persist when IDES T and accuracy are measured on different datasets, or under alternative evaluation protocols?

In Figure 3, we study Inter-Dataset Correlation, where IDES T is computed on ImageNet and accuracy is evaluated on four target datasets: the large-scale fine-grained benchmarks iNat-18 and iNat-21, and the smaller-scale datasets CIFAR-10 and CIFAR-100. The correlation remains strong across both target datasets, suggesting that IDES T captures intrinsic properties of the learned representations rather than dataset-specific characteristics.

In Figure 4, we examine IDES T under Alternative Evaluation Protocols. IDES T remains indicative of performance on ImageNet under kNN evaluation, and on the complementary ImageNet-v2 validation set.

Overall, these results reveal that low intrinsic dimension is a consistent feature of well-performing representations, and suggest that IDES T serves as a simple, label-free proxy for downstream evaluation (Figures 1, 3 and 4), offering insight into the quality of self-supervised representations without annotated data.

Finding 1. IDES T negatively correlates with linear probing accuracy: low intrinsic dimension is a consistent geometric signature of strong representations across intra- and inter-dataset settings and alternative evaluation protocols.

# 4.2. Training dynamics: offline and online probing

Here, we study whether IDES T tracks representation quality during unsupervised training. To this end, we consider two complementary evaluation protocols: (i) offline linear probing and (ii) online probing. While both aim to assess downstream performance over the course of self-supervised pretraining, they differ in when and how the classifier is trained.

Offline linear probing. Offline linear probing is the standard evaluation protocol in self-supervised learning. The representation model is first trained without labels, after which a linear classifier is trained on frozen features. To analyze training dynamics, we extract multiple checkpoints during pretraining and perform linear probing independently for each checkpoint.

Figure 5 (top) reports results for VICReg (Bardes et al., 2022) (5a), DINO (Caron et al., 2021) (5b) and I-JEPA (Assran et al., 2023) (5c), respectively on ImageNet. We show linear probing top-1 accuracy (y axis on the left, in orange) as a function of training epochs, together with IDES T (y axis on the right, in blue). Across both models, IDES T closely follows the evolution of downstream accuracy throughout training. As representations improve and linear probing accuracy increases, the intrinsic dimension consistently decreases. This strong temporal correlation indicates that IDES T captures meaningful geometric changes in the representation space as training progresses, without requiring labels.

Online linear probing. We consider online probing, where a linear classification head is attached to the representation and trained jointly during self-supervised pretraining. Importantly, gradients from the classifier do not backpropagate into the representation encoder, ensuring that the learned features remain purely self-supervised. This setting allows us to monitor downstream performance continuously during training.

![](images/560625ba72c5da558e08204158b83cb1402696869c3ae2a66f68867f01f8b861.jpg)

<details>
<summary>line</summary>

| Epochs | ImageNet Top-1 Accuracy (%) | IdEst |
| ------ | --------------------------- | ----- |
| 25     | 65.0                        | 18.0  |
| 50     | 63.0                        | 21.0  |
| 75     | 61.0                        | 22.5  |
| 100    | 58.0                        | 23.0  |
</details>

(a) VICReg

![](images/7bbb526c09d3c03327245bad60b13dd49dfbb78bcd5d717df92e63763a95449a.jpg)

<details>
<summary>line</summary>

| Epochs | ImageNet Top-1 Accuracy (%) | IdEst |
| ------ | --------------------------- | ----- |
| 25     | 66                          | 20    |
| 50     | 63                          | 21    |
| 75     | 57                          | 22    |
| 100    | 51                          | 22    |
</details>

(b) DINO

![](images/a0d5acae5319ff913fb845ba49e01c38ad2a7d2281b010d13790e1f64d3a26a0.jpg)

<details>
<summary>line</summary>

| Epochs | ImageNet Top-1 Accuracy (%) | IdEst |
| ------ | --------------------------- | ----- |
| 80     | 63                          | 16    |
| 160    | 60                          | 17    |
| 240    | 59                          | 20    |
</details>

(c) I-JEPA   
Offline probing dynamics.

![](images/f2abcb84953f437023bc7ad2adb9ba65aa911d058fa35d2725828decb1695029.jpg)

<details>
<summary>line</summary>

| Epochs | DINO Loss (Blue) | DINO Loss (Yellow) |
| ------ | ---------------- | ------------------ |
| 0      | 10.0             | 5.0                |
| 25     | 4.0              | 2.0                |
| 50     | 3.0              | 1.5                |
| 75     | 2.5              | 1.2                |
| 100    | 2.0              | 1.0                |
</details>

![](images/fa9af4a6e21d74e10efcc4b363d8a679f1fb448e3631d210b37184edb3072541.jpg)

<details>
<summary>line</summary>

| Epochs | Online Classification Loss |
| ------ | -------------------------- |
| 0      | 10                         |
| 10     | 48                         |
| 20     | 30                         |
| 30     | 35                         |
| 40     | 32                         |
| 50     | 30                         |
| 60     | 20                         |
| 70     | 10                         |
| 80     | 5                          |
| 90     | 3                          |
| 100    | 2                          |
</details>

![](images/a0cf56ea50610a44762591de5ecf796180e79b462992085a98b4ed8667413c25.jpg)

<details>
<summary>line</summary>

| Epochs | ImageNet Top-1 Accuracy (%) - Line 1 | ImageNet Top-1 Accuracy (%) - Line 2 |
| ------ | ------------------------------------ | ------------------------------------ |
| 0      | 0                                    | 0                                    |
| 25     | 30                                   | 25                                   |
| 50     | 45                                   | 40                                   |
| 75     | 55                                   | 50                                   |
| 100    | 60                                   | 55                                   |
</details>

![](images/896f54fb76cc50cae86e2d6363ada6848577bc6619b8cc61bccba13c1a656e00.jpg)

<details>
<summary>line</summary>

| Epochs | t-temp = 0.01 | t-temp = 0.04 |
| ------ | ------------- | ------------- |
| 0      | 11.0          | 11.0          |
| 25     | 22.5          | 24.0          |
| 50     | 21.5          | 23.5          |
| 75     | 20.0          | 23.0          |
| 100    | 19.0          | 22.5          |
</details>

Online probing dynamics during DINO pretraining.   
Figure 5. Tracking Training Dynamics. (Top) Offline probing dynamics: Evolution of ImageNet-1k linear probing top-1 accuracy (left y-axis, in orange) and IDES T (right y-axis, in blue) over self-supervised pretraining epochs for (a) VICReg (ResNet-50), (b) DINO (ViT-S), and (c) I-JEPA (ViT-B). As representations improve and linear probing accuracy increases, IDES T consistently decreases, demonstrating its ability to capture meaningful geometric changes during the evolution of the representation. (Bottom) Online probing dynamics during self-supervised training: Evolution of the self-supervised loss, classification loss, ImageNet-1k online classification top-1 accuracy, and IDES T over training epochs for DINO. While early-stage representations are highly constrained, IDES T progressively tracks improvements in downstream accuracy as training proceeds.

Figure 5 (bottom) illustrates the training dynamics for DINO using a ViT-S, reporting the self-supervised loss (left), online classification loss and ImageNet top-1 accuracy (middle), and IDES T (right). We observe that during the early stages of training, particularly within the first 10 epochs, representations are highly constrained, rendering IDES T less informative. However, as training progresses, the selfsupervised loss decreases and classification accuracy improves; concurrently, IDES T systematically drops, closely tracking the evolution of the other metrics. This demonstrates that IDES T faithfully reflects representation quality even in an online setting, capturing improvements in downstream performance as they emerge during the pretraining.

Overall, we observe that across both offline and online protocols, and across diverse SSL objectives, IDES T mirrors the evolution of downstream classification performance. These results further support its utility as a label-free indicator of representation quality, effective not only at convergence but throughout the pretraining process, once the initial training stages are surpassed. This leads us to our second finding:

Finding 2. IDES T can serve as a label-free proxy for SSL pretraining effectiveness: it decreases as downstream task accuracy improves over the course of unsupervised pre-training.

# 4.3. Label-free metric for Model Selection

As stated in Theorem 3.1 and highlighted by previous experiments, performance on downstream tasks is bounded by the intrinsic dimension of the representations. Consequently, we evaluate IDES T as an hyperparameter selection criterion that may bypass the need for linear probing.

Selecting hyperparameters with IDES T. Given a set of candidate models $\mathcal { F } = \{ f _ { 1 } , \ldots , f _ { n } \}$ , each trained with varying hyperparameters, and their corresponding dimension estimates $\Delta = \{ d _ { 1 } , \ldots , d _ { n } \}$ , the selected model, f ∗, is defined as:

$$
f ^ {*} = f _ {\arg \min _ {i} d _ {i}} \tag {4}
$$

IDES T 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Unsup.</td><td colspan="4">VICReg (RN-50)</td><td colspan="4">DINO (ViT-S)</td><td colspan="4">I-JEPA (ViT-B)</td></tr><tr><td>lr</td><td>wd</td><td>var.</td><td>all</td><td>lr</td><td>s-temp.</td><td>t-temp.</td><td>all</td><td>lr</td><td>target-size</td><td>context-size</td><td>all</td></tr><tr><td rowspan="5">ImageNet</td><td>ACC-1 Bounds</td><td></td><td>[62.2, 66.5]</td><td>[37.5, 69.1]</td><td>[65.5, 67.2]</td><td>[37.5, 69.1]</td><td>[63.6, 69.1]</td><td>[48.4,67.8]</td><td>[61.2, 67.5]</td><td>[48.4, 69.1]</td><td>[61.9, 66.4]</td><td>[49.0, 66.4]</td><td>[61.3, 66.5]</td><td>[49.0,66.5]</td></tr><tr><td>α-ReQ</td><td>√</td><td>65.0</td><td>53.5</td><td>66.9</td><td>53.5</td><td>63.6</td><td>58.6</td><td>61.2</td><td>58.6</td><td>61.9</td><td>49.0</td><td>61.3</td><td>49.0</td></tr><tr><td>RankMe</td><td>√</td><td>66.5</td><td>69.1</td><td>65.5</td><td>69.1</td><td>63.6</td><td>67.8</td><td>67.5</td><td>63.6</td><td>61.9</td><td>55.9</td><td>61.3</td><td>61.9</td></tr><tr><td>LiDAR</td><td>×</td><td>65.0</td><td>69.1</td><td>66.8</td><td>65.0</td><td>68.2</td><td>65.6</td><td>65.0</td><td>65.5</td><td>63.4</td><td>66.4</td><td>66.0</td><td>63.4</td></tr><tr><td>IDEST</td><td>√</td><td>62.3</td><td>67.1</td><td>65.5</td><td>65.5</td><td>64.7</td><td>65.6</td><td>67.5</td><td>65.5</td><td>66.4</td><td>66.4</td><td>66.0</td><td>66.4</td></tr><tr><td rowspan="5">Fine-Grained</td><td>ImageNet Oracle</td><td>×</td><td>67.2</td><td>62.2</td><td>68.6</td><td>62.9</td><td>65.5</td><td>64.8</td><td>55.2</td><td>65.5</td><td>60.0</td><td>60.7</td><td>58.3</td><td>60.0</td></tr><tr><td>α-ReQ</td><td>√</td><td>66.7</td><td>47.8</td><td>68.5</td><td>66.5</td><td>63.8</td><td>59.4</td><td>60.6</td><td>59.4</td><td>59.5</td><td>41.4</td><td>57.9</td><td>41.4</td></tr><tr><td>RankMe</td><td>√</td><td>67.2</td><td>62.2</td><td>64.9</td><td>62.9</td><td>63.8</td><td>64.8</td><td>64.5</td><td>63.8</td><td>59.5</td><td>56.9</td><td>57.9</td><td>59.5</td></tr><tr><td>LiDAR</td><td>×</td><td>66.7</td><td>62.2</td><td>66.6</td><td>66.6</td><td>68.3</td><td>62.4</td><td>64.3</td><td>62.4</td><td>58.3</td><td>60.7</td><td>58.2</td><td>58.3</td></tr><tr><td>IDEST</td><td>√</td><td>61.8</td><td>58.3</td><td>64.9</td><td>65.0</td><td>63.9</td><td>62.4</td><td>64.5</td><td>62.4</td><td>60.0</td><td>60.0</td><td>58.3</td><td>60.0</td></tr></table>

Table 1. Unsupervised model selection with IDES T. We evaluate IDES T for hyperparameter selection against a supervised linear probe on ImageNet-1k, two unsupervised baselines: α-ReQ (Agrawal et al., 2022) and RankMe (Garrido et al., 2023), and a weakly-supervised one: LiDAR (Thilak et al., 2024). For each SSL objective (and architecture), hyperparameters are jointly selected across according to Equation (4). ‘Fine-grained’ denotes the average performance across all furhter datasets (i.e. iNat-21,SUN, Aircraft, CUB, CIFAR-10) excluding ImageNet-1k. Bold values indicate the top-performing model selected by the criteria. IDES T is competitive with unsupervised and weakly-supervised baselines across SSL methods, including VICReg, despite RankMe directly mirroring its objective.

Set-Up. We apply Equation (4) to identify the optimal model for a given hyperparameter configuration across several SSL frameworks including Joint-Embedding methods (VICReg (Bardes et al., 2022), DINO (Caron et al., 2021)) and the Joint-Predictive architecture (I-JEPA (Assran et al., 2023)). Training details for each model are provided in Appendix B.3.

We focus on various hyperparameters:

1. Optimization: learning rate (lr) and weight decay (wd);   
2. Loss-specific coefficients: variance coefficients in VI-CReg (var.); the teacher and student temperatures in DINO (t-temp., s-temp.);   
3. Size of masking for I-JEPA: target block size (target-size) and context block size (context-size).

We evaluate performance on ImageNet and the average accuracy across several fine-grained classification datasets, comparing models selected by IDES T against those selected by supervised validation accuracy, and include the all setting, where selection is performed over the full hyperparameter pool, the more challenging configuration.

Results. As shown in Table 1, IDES T can recover most ImageNet oracle performance, achieving results near the upper bound of validation accuracy across various architectures (ResNets and ViTs) and pre-training objectives. Notably, IDES T outperforms α-ReQ in most settings without suffering from significant performance drops in the worst-case scenarios. For instance, α-ReQ selects the lower-bound accuracy for I-JEPA.

Additionally, while RankMe was originally designed for JE-SSL methods (e.g., VICReg, DINO), and in fact mirrors the VICReg objective by directly rewarding high effective rank, IDES T maintains competitive performance on these

<table><tr><td>Model</td><td>Architecture</td><td>D</td><td>Param (M)</td><td>Linear Probe (min)</td><td>IDEST (min)</td></tr><tr><td rowspan="2">VICReg</td><td>ResNet-50</td><td>2048</td><td>24</td><td>64.5</td><td>2.1</td></tr><tr><td>ViT-S</td><td>384</td><td>22</td><td>65.8</td><td>1.8</td></tr><tr><td rowspan="3">DINOv2</td><td>ViT-B</td><td>768</td><td>86</td><td>64.5</td><td>2.1</td></tr><tr><td>ViT-L</td><td>1024</td><td>303</td><td>113.3</td><td>4.9</td></tr><tr><td>ViT-G</td><td>1408</td><td>1000</td><td>322.7</td><td>12.8</td></tr></table>

Table 2. Computational cost. Wall-clock time (min) on ImageNet-1k to evaluate frozen representations using either linear probing (10 epochs, batch size 1024) or IDES T (single feature-extraction pass followed by dimMST computation). Rows vary the backbone architecture, primarily changing the output dimension D and the number of parameters. Results are averages over 3 runs.

models while also generalising to settings where RankMe struggles, such as I-JEPA. IDES T demonstrates versatility to other SSL paradigms. As I-JEPA is not a standard Joint-Embedding Method (making dimensional collapse less of a primary concern), other metrics struggle, whereas IDES T successfully generalizes to this and other SSL paradigms.

Furthermore, LiDAR (Thilak et al., 2024) is a strong baseline when pretraining augmentations are accessible. Nonetheless, across all four models in the all setting, IDES T achieves comparable model selection than LiDAR in a strictly unsupervised setting. Additional results on DINO (ResNet-50) are available in Table 4.

Finding 3. IDES T can serve as an efficient label-free criterion for hyperparameter selection: it performs comparably to supervised baselines across diverse architectures and SSL objectives.

# 4.4. Computational Cost

We evaluate the computational cost of IDES T and compare it to linear probing. Compared to training a linear probe, which requires multiple epochs over the training dataset features to obtain reasonable estimates of downstream accuracy, computing IDES T only requires feature extraction and a dimMST computation. In Table 2, we compare the wall-clock compute time for IDES T and standard linear probe training on ImageNet (10 epochs across 2 GPUs), averaged over 3 runs. Computations were performed on H100 GPUs and an Intel Sapphire Rapids 8468 CPU. Across all architectures, IDES T is substantially faster than a 10-epoch linear probe (B = 1024, on 2 GPUs). Notably, for larger models, the cost is dominated by feature extraction, while the overhead of computing dimMST remains minimal across various output dimensions.

# 4.5. Limitations

While versatile, our framework has several limitations.

The formula in Theorem 3.1, provides only a bound on the convergence rate. Therefore, even if two models have similar intrinsic dimensions, their actual convergence rates may potentially differ in practice. Thus, IDES T should be viewed more as an indicator of accuracy than as a perfect predictor. This is indeed reflected in Figure 1 for instance, where $\rho \approx 0 . 8 , \tau \approx 0 . 6$ confirms strong global ranking ability, while acknowledging that the intrinsic dimension might not explain all the variance. In particular, ranking is inherently harder when model accuracies are tightly clustered, as in ImageNet or SUN397, than when they are more spread out, as in iNaturalist. Furthermore, one family deviates most visibly: vision-language models (e.g., CLIP, EVA-CLIP), where the cone effect introduces geometric misalignment between encoders that the contrastive objective preserves rather than resolves, constraining the representations (Liang et al., 2022).

Additionally, a more subtle limitation stems from the earlytraining regime of ViTs: IDES T is less informative during the first 10 training epochs, before representations develop stable geometric structure (Section 4.2).

# 5. Conclusion

In this work, we introduced IDES T, an unsupervised criterion for evaluating self-supervised representations based on the estimation of the intrinsic dimension via minimum spanning trees (dimMST). Building on the theoretical connection between intrinsic dimension and generalization, we demonstrated that IDES T serves as a robust and an efficient geometric proxy for downstream performance in SSL.

Our empirical evaluation across diverse datasets, architectures, and SSL objectives shows that IDES T correlates with supervised linear probing accuracy. Furthermore, IDES T provides a principled label-free metric for hyperparameter selection, performing on par with supervised oracles and generalizing across heterogeneous SSL paradigms.

By offering a unified framework for assessing SSL representations without requiring annotated data, our work highlights the potential of intrinsic geometric descriptors to complement standard evaluation protocols.

Future Work. Future work could explore several directions. First, the relationship between intrinsic and effective dimensions (e.g., RankMe (Garrido et al., 2023)) warrants closer study. As further discussed in Section F, a large gap between the two may signal curvature in the representation manifold, opening the door to differential-geometric tools for complementary analysis. Second, integrating intrinsic dimension estimates directly into SSL training objectives could enable geometry-aware learning, guiding models toward representations that are simultaneously compact and well-spread. Finally, extending IDES T to dense tasks, e.g., segmentation or generation, is a natural next step, as the underlying MST estimator is task-agnostic, though such settings do not fall within the theoretical framing leveraged here.

# Acknowledgments

This work is supported by Hi! PARIS, ANR/France 2030 program (ANR-23-IACL-0005) and Inria Action Exploratoire PREMEDIT (Precision Medicine using Topology). We were granted access to the HPC resources of IDRIS under the allocations 2025-A0190616899 made by GENCI. We would like to thank David Loiseaux and Eleftherios Tsonis for their useful feedback.

# Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Adams, H., Aminian, M., Farnell, E., Kirby, M., Mirth, J., Neville, R., Peterson, C., and Shonkwiler, C. A Fractal Dimension for Measures via Persistent Homology. In Topological Data Analysis, 2020.

Aghajanyan, A., Gupta, S., and Zettlemoyer, L. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In ACL-IJCNLP, 2021.

Agrawal, K. K., Mondal, A. K., Ghosh, A., and Richards, B. α-ReQ: Assessing representation quality in selfsupervised learning by measuring eigenspectrum decay. Adv. Neural Inform. Process. Syst., 2022.

Ansuini, A., Laio, A., Macke, J. H., and Zoccolan, D. Intrinsic dimension of data representations in deep neural networks. Adv. Neural Inform. Process. Syst., 2019.   
Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, 2023.   
Bardes, A., Ponce, J., and Lecun, Y. Vicreg: Varianceinvariance-covariance regularization for self-supervised learning. In ICLR, 2022.   
Bauer, U. Ripser: efficient computation of vietoris–rips persistence barcodes. Journal of Applied and Computational Topology, 2021.   
Bennett, R. The intrinsic dimensionality of signal collections. IEEE Transactions on Information Theory, 1969.   
Binnie, J. A., Dłotko, P., Harvey, J., Malinowski, J., and Yim, K. M. A survey of dimension estimation methods. arXiv, 2025.   
Birdal, T., Lou, A., Guibas, L. J., and Simsekli, U. Intrinsic dimension, persistent homology and generalization in neural networks. Adv. Neural Inform. Process. Syst., 2021.   
Bolya, D., Huang, P.-Y., Sun, P., Cho, J. H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv, 2025.   
Cai, X., Huang, J., Bian, Y., and Church, K. Isotropy in the contextual embedding space: Clusters and manifolds. In ICLR, 2021.   
Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., ´ Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In CVPR, 2021.   
Chazal, F., De Silva, V., and Oudot, S. Persistence stability for geometric complexes. Geometriae Dedicata, 2014.   
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In ICML, 2020.   
Costa, J. A. and Hero, A. O. Determining Intrinsic Dimension and Entropy of High-Dimensional Shape Spaces. In Krim, H. and Yezzi, A. (eds.), Statistics and Analysis of Shapes. 2006.   
Couairon, P., Chambon, L., Serrano, L., Haugeard, J.-E., Cord, M., and Thome, N. Jafar: Jack up any feature at any resolution. arXiv preprint arXiv:2506.11136, 2025.

Degeorge, L., Ghosh, A., Dufour, N., Picard, D., and Kalogeiton, V. How far can we go with imagenet for text-toimage generation? arXiv, 2025.   
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.   
Dufour, N., Picard, D., Kalogeiton, V., and Landrieu, L. Around the world in 80 timesteps: A generative approach to global visual geolocation. CVPR, 2025.   
Dupuis, B., Deligiannidis, G., and Simsekli, U. Generalization bounds using data-dependent fractal dimensions. In ICLR, 2023.   
Facco, E., d’Errico, M., Rodriguez, A., and Laio, A. Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific reports, 2017.   
Fang, X., Li, J., Sun, Q., and Wang, B. Rethinking the uniformity metric in self-supervised learning. In ICLR, 2024.   
Garrido, Q., Balestriero, R., Najman, L., and Lecun, Y. Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank. In ICML, 2023.   
Goodfellow, I., Bengio, Y., and Courville, A. Deep learning. MIT press Cambridge, 2016.   
He, B. and Ozay, M. Exploring the gap between collapsed & whitened features in self-supervised learning. In ICML, 2022.   
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In CVPR, 2016.   
Jing, L., Vincent, P., LeCun, Y., and Tian, Y. Understanding dimensional collapse in contrastive self-supervised learning. arXiv preprint arXiv:2110.09348, 2021.   
Johnsson, K., Soneson, C., and Fontes, M. Low bias local intrinsic dimension estimation from expected simplex skewness. IEEE TPAMI, 2014.   
Kendall, M. G. A new measure of rank correlation. Biometrika, 1938.   
Konz, N. and Mazurowski, M. A. The effect of intrinsic dataset properties on generalization: Unraveling learning differences between natural and medical images. In ICLR, 2024.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.   
Lee, A., Weber, M., Viegas, F., and Wattenberg, M. Shared ´ global and local geometry of language model embeddings. arXiv, 2025.   
Levina, E. and Bickel, P. Maximum likelihood estimation of intrinsic dimension. Advances in neural information processing systems, 17, 2004.   
Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., and Zou, J. Y. Mind the gap: Understanding the modality gap in multimodal contrastive representation learning. Adv. Neural Inform. Process. Syst., 2022.   
Maji, S., Rahtu, E., Kannala, J., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. arXiv, 2013.   
Maruani, N., Zhang, P., Chaudhuri, S., Fisher, M., Zhao, N., Kim, V. G., Alliez, P., Desbrun, M., and Yifan, W. Illustrator’s depth: Monocular layer index prediction for image decomposition. arXiv, 2025.   
Mordacq, J., Loiseaux, D., Kalogeiton, V., and Oudot, S. T-REGS: Minimum spanning tree regularization for selfsupervised learning. In Adv. Neural Inform. Process. Syst., 2025.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv, 2023.   
Oudot, S. Y. Persistence theory: from quiver representations to data analysis. American Mathematical Soc., 2015.   
Pope, P., Zhu, C., Abdelkader, A., Goldblum, M., and Goldstein, T. The intrinsic dimension of images and its impact on learning. In ICLR, 2021.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, 2021.   
Roy, O. and Vetterli, M. The effective rank: A measure of effective dimensionality. In European signal processing conference, 2007.   
Ruppik, B. M., von Rohrscheidt, J., van Niekerk, C., Heck, M., Vukovic, R., Feng, S., Lin, H.-c., Lubis, N., Rieck, B., Zibrowius, M., et al. Less is more: Local intrinsic dimensions of contextual language models. Adv. Neural Inform. Process. Syst., 2025.

Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, ´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv, 2025.   
Simsekli, U., Sener, O., Deligiannidis, G., and Erdogdu, M. A. Hausdorff dimension, heavy tails, and generalization in neural networks. Adv. Neural Inform. Process. Syst., 2020.   
Spearman, C. The proof and measurement of association between two things. The American Journal of Psychology, 1961.   
Sun, Q., Fang, Y., Wu, L., Wang, X., and Cao, Y. Eva-clip: Improved training techniques for clip at scale. arXiv, 2023.   
Tan, C. B., Garc´ıa-Redondo, I., Wang, Q., Bronstein, M. M., and Monod, A. On the limitations of fractal dimension as a measure of generalization. Adv. Neural Inform. Process. Syst., 2024.   
Tempczyk, P., Michaluk, R., Garncarek, L., Spurek, P., Tabor, J., and Golinski, A. Lidl: Local intrinsic dimension estimation using approximate likelihood. In ICML, 2022.   
Thilak, V., Huang, C., Saremi, O., Dinh, L., Goh, H., Nakkiran, P., Susskind, J. M., and Littwin, E. Lidar: Sensing linear probing performance in joint embedding ssl architectures. In ICLR, 2024.   
Tralie, C., Saul, N., and Bar-On, R. Ripser. py: A lean persistent homology library for python. JOSS, 2018.   
Tulchinskii, E., Kuznetsov, K., Kushnareva, L., Cherniavskii, D., Nikolenko, S., Burnaev, E., Barannikov, S., and Piontkovskaya, I. Intrinsic dimension estimation for robust detection of ai-generated texts. Adv. Neural Inform. Process. Syst., 2023.   
Valeriani, L., Doimo, D., Cuturello, F., Laio, A., Ansuini, A., and Cazzaniga, A. The geometry of hidden representations of large transformer models. Adv. Neural Inform. Process. Syst., 2023.   
Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Perona, P., and Belongie, S. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2018.   
Van Horn, G., Cole, E., Beery, S., Wilber, K., Belongie, S., and Mac Aodha, O. Benchmarking representation learning for natural world image collections. In CVPR, 2021.   
Venkataramanan, S., Pariza, V., Salehi, M., Knobel, L., Gidaris, S., Ramzi, E., Bursuc, A., and Asano, Y. M.

Franca: Nested matryoshka clustering for scalable visual representation learning. arXiv, 2025.   
Viswanathan, K., Gardinazzi, Y., Panerai, G., Cazzaniga, A., and Biagetti, M. The geometry of tokens in internal representations of large language models. arXiv, 2025.   
Wang, T. and Isola, P. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In ICML, 2020.   
Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F., Belongie, S., and Perona, P. Caltech-ucsd birds 200. 2010.   
Xiao, J., Hays, J., Ehinger, K. A., Oliva, A., and Torralba, A. Sun database: Large-scale scene recognition from abbey to zoo. In CVPR, 2010.   
Yukich, J. E. Probability theory of classical Euclidean optimization problems. Springer, 2006.   
Zbontar, J., Jing, L., Misra, I., LeCun, Y., and Deny, S. Barlow twins: Self-supervised learning via redundancy reduction. In ICML, 2021.   
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In ICCV, 2023.   
Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., and Kong, T. ibot: Image bert pre-training with online tokenizer. ICLR, 2022.

# Appendix to

# IDES T: Assessing Self-Supervised Learning Representations via Intrinsic Dimension

# Contents

A IDES T’s Implementation Details 13   
B Implementation Details for SSL models studied 14

B.1 Image Classification Training Details . 14   
B.2 Models studied in Section 4.1 . 15   
B.3 Models trained in Section 4.3 15

C Additional hyperparameter selection results. 16   
D Compute cost 17   
E Further study of Foundation Models 17

E.1 Additional properties studied in Joint-Embedding Self-Supervised Learning 17   
E.2 Foundation Models and further metrics. 19

F Effective and Intrinsic Dimensions 19

# A. IDES T’s Implementation Details

Minimum Spanning Tree. We complement the definition of the Minimum Spanning Tree (MST) given in Section 3.2 with a visual overview in Figure 6 and with formal definitions below.

![](images/91edc0ff4e9b093f18cb494a11aef83a90500afbfd1d2cd406b646aaeeb12495.jpg)

<details>
<summary>scatter</summary>

| Point cloud X | MST edge | Non-MST edge |
| ------------- | -------- | ------------ |
| Point 1       | 0        | 0            |
| Point 2       | 0        | 0            |
| Point 3       | 0        | 0            |
| Point 4       | 0        | 0            |
| Point 5       | 0        | 0            |
| Point 6       | 0        | 0            |
| Point 7       | 0        | 0            |
| Point 8       | 0        | 0            |
| Point 9       | 0        | 0            |
</details>

Figure 6. Overview of a Minimum Spanning Tree (MST). (Left) A point cloud $X \subset \mathbb { R } ^ { 2 } .$ . (Right) The $\operatorname { M S T } ( X )$ connects all points without forming any cycle, while minimizing the total edge length. Grey edges indicate pairwise connections not retained in the MST: including them would either create a cycle or increase the total length.

Definition A.1. (Spanning Tree). A spanning tree of X is an undirected graph $G = ( V , E )$ with vertex set $V = X$ and edge set $E \subseteq V \times V$ such that G is connected and acyclic.

Definition A.2. (Minimum Spanning Tree). A minimum spanning tree (MST) of X is a spanning tree $G = ( V , E )$ of minimum total edge weight:

$$
L \left(\operatorname{MST} (X)\right) := \sum_ {(u, v) \in E ^ {*}} \| u - v \| _ {2}.
$$

dimMST. As described in Section 3.2, IDES T leverages dimMST to estimate the intrinsic dimension of the representation space. In practice, the ID is estimated by log-log linear regression over subsamples of increasing size: given subsamples $Z _ { n }$ i with sizes $n _ { i } ,$ we fit $\begin{array} { r } { \log ( L \left( \mathrm { M S T } ( Z _ { n _ { i } } ) \right) ) \approx \frac { d - 1 } { d } \log ( n _ { i } ) + \log ( C ) } \end{array}$ , and recover $d = 1 / ( 1 - m )$ from the fitted slope m. The complete algorithm is given in Algorithm 1.

Algorithm 1 Computation of dimMST   
Input: The set of representations $Z = \{z_{1}, \ldots, z_{N}\}$ , minimum sample size $n_{min}$ , skip size $\Delta$ Output: The estimated dimension: $\dim_{\mathrm{MST}}(Z)$ Initialize $n \leftarrow n_{\min}$ , $E \leftarrow []$ while n < N do $Z_{n} \leftarrow \text{sample}(Z, n)$ # random sampling of n points $E[i] \leftarrow L(\text{MST}(Z_{n}))$ # computation of the minimum spanning tree $n \leftarrow n + \Delta$ end while $m, b \leftarrow \text{linear regression}(\log(n_{\min} : \Delta : N), \log(E))$ # linear regression of the log-log plot $\dim_{\mathrm{MST}}(Z) \leftarrow 1/(1 - m)$

We follow the implementation of Adams et al. (2020); Dupuis et al. (2023) for dimMST. For each dataset and model, we compute the dimension estimator. Preprocessing (distance matrix computation and sorting) dominates in practice and is optimized via Ripser (Tralie et al., 2018; Bauer, 2021), which implements an efficient sparse distance pipeline. To keep computation tractable, IDES T operates on a subsample of size $N \ll N _ { D }$ , where $N _ { D }$ denotes the full dataset size; we set $N = 5 0 { , } 0 0 0$ throughout.

# B. Implementation Details for SSL models studied

# B.1. Image Classification Training Details

Datasets. We evaluate the global quality of the SSL models using the widely adopted linear probing evaluation. We consider:

1. ImageNet dataset (Deng et al., 2009)   
2. Large-scale fine-grained datasets: iNat-18 (Van Horn et al., 2018), iNat-21 (Van Horn et al., 2021), SUN397 (Xiao et al., 2010)   
3. Small-scale fine-grained datasets: CIFAR-10 and CIFAR-100 (Krizhevsky et al., 2009), Aircraft (Maji et al., 2013), CUB200 (Welinder et al., 2010)

Evaluation protocol. For each baseline, we follow the protocol of (Simeoni et al. ´ , 2025) and train a linear layer on the final frozen representation. To obtain the frozen representation, we follow each method’s standard evaluation protocol, e.g., the CLS token after the layer norm, or avgpool if there is no CLS token. Specifically, we use SGD with a momentum of 0.9, and train for 10 epochs with a batch size of 1024, using random-resized-crop data. We perform the following grid search:

• Learning Rate in {0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1}

For the fine-grained dataset (i.e., Aircraft), following (Oquab et al., 2023), we use a lighter weight evaluation using scikit-learn’s LogisticRegression implementation with the L-BFGS solver.

IDES T 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Architecture</td><td rowspan="2">Repo</td><td rowspan="2">ImageNet</td><td colspan="4">Top-1 Accuracy (%)</td></tr><tr><td>v2</td><td>SUN397</td><td>iNat-18</td><td>iNat-21</td></tr><tr><td>BarlowTwins (Zbontar et al., 2021)</td><td>ResNet-50</td><td>github</td><td>72.9</td><td>60.3</td><td>71.6</td><td>38.1</td><td>53.4</td></tr><tr><td rowspan="4">CLIP (Radford et al., 2021)</td><td>ResNet-50</td><td>timm</td><td>69.6</td><td>52.8</td><td>76.5</td><td>13.7</td><td>46.4</td></tr><tr><td>ViT-B/16</td><td>timm</td><td>80.1</td><td>68.8</td><td>82.2</td><td>51.1</td><td>63.4</td></tr><tr><td>ViT-B/32</td><td>timm</td><td>76.1</td><td>63.5</td><td>80.6</td><td>41.9</td><td>54.6</td></tr><tr><td>ViT-L/14</td><td>timm</td><td>84.3</td><td>75.1</td><td>84.8</td><td>59.8</td><td>72.5</td></tr><tr><td rowspan="2">DINO (Caron et al., 2021)</td><td>ViT-S/16</td><td>github</td><td>76.0</td><td>64.4</td><td>72.5</td><td>48.4</td><td>60.9</td></tr><tr><td>ViT-B/16</td><td>github</td><td>77.8</td><td>66.7</td><td>74.8</td><td>49.9</td><td>63.4</td></tr><tr><td rowspan="6">DINOv2 (Oquab et al., 2023)</td><td>ViT-S/14</td><td>github</td><td>80.4</td><td>70.4</td><td>79.5</td><td>61.5</td><td>73.5</td></tr><tr><td>ViT-B/14</td><td>github</td><td>83.8</td><td>74.9</td><td>81.7</td><td>68.3</td><td>80.3</td></tr><tr><td>ViT-B/14-reg</td><td>github</td><td>83.9</td><td>75.1</td><td>81.9</td><td>67.4</td><td>79.4</td></tr><tr><td>ViT-L/14</td><td>github</td><td>85.7</td><td>77.7</td><td>82.8</td><td>71.3</td><td>83.1</td></tr><tr><td>ViT-L/14-reg</td><td>github</td><td>86.1</td><td>77.9</td><td>83.3</td><td>71.0</td><td>83.3</td></tr><tr><td>ViT-G/14</td><td>github</td><td>86.2</td><td>78.1</td><td>82.9</td><td>74.0</td><td>85.3</td></tr><tr><td rowspan="4">DINOv3 (Siméoni et al., 2025)</td><td>ViT-S/16</td><td>github</td><td>79.3</td><td>68.7</td><td>79.1</td><td>61.8</td><td>69.4</td></tr><tr><td>ViT-B/14</td><td>github</td><td>84.5</td><td>75.0</td><td>82.9</td><td>70.6</td><td>79.4</td></tr><tr><td>ViT-L/16</td><td>github</td><td>87.2</td><td>79.5</td><td>84.6</td><td>76.1</td><td>84.2</td></tr><tr><td>ViT-7B</td><td>github</td><td>88.4</td><td>81.4</td><td>85.4</td><td>80.2</td><td>88.7</td></tr><tr><td rowspan="3">EVA (Sun et al., 2023)</td><td>EVA01-g-14</td><td>timm</td><td>86.1</td><td>71.5</td><td>85.5</td><td>64.9</td><td>79.5</td></tr><tr><td>EVA02-b-16</td><td>timm</td><td>83.6</td><td>73.3</td><td>83.1</td><td>55.7</td><td>70.5</td></tr><tr><td>EVA02-E-14-plus</td><td>timm</td><td>87.6</td><td>75.6</td><td>86.5</td><td>70.6</td><td>82.7</td></tr><tr><td>Franca (Venkataramanan et al., 2025)</td><td>ViT-L/14 (In-21k)</td><td>github</td><td>84.2</td><td>75.4</td><td>80.7</td><td>73.5</td><td>70.6</td></tr><tr><td rowspan="3">iBoT (Zhou et al., 2022)</td><td>ViT-S/16</td><td>github</td><td>76.9</td><td>65.6</td><td>73.6</td><td>48.1</td><td>60.6</td></tr><tr><td>ViT-B/16</td><td>github</td><td>79.2</td><td>68.1</td><td>75.2</td><td>48.6</td><td>63.4</td></tr><tr><td>ViT-L/16</td><td>github</td><td>80.6</td><td>69.9</td><td>76.4</td><td>51.5</td><td>66.2</td></tr><tr><td>I-JEPA (Assran et al., 2023)</td><td>ViT-G/16</td><td>github</td><td>75.8</td><td>63.8</td><td>74.9</td><td>45.9</td><td>58.7</td></tr><tr><td rowspan="3">PE-Core (Bolya et al., 2025)</td><td>S16-336</td><td>timm</td><td>70.9</td><td>51.2</td><td>75.6</td><td>38.6</td><td>54.6</td></tr><tr><td>B14-224</td><td>timm</td><td>83.2</td><td>72.7</td><td>83.8</td><td>60.8</td><td>74.9</td></tr><tr><td>L14-336</td><td>timm</td><td>87.8</td><td>78.7</td><td>87.0</td><td>74.1</td><td>84.1</td></tr><tr><td rowspan="2">SigLIP (Zhai et al., 2023)</td><td>ViT-B-16-SigLIP</td><td>timm</td><td>82.5</td><td>68.9</td><td>82.7</td><td>53.3</td><td>69.6</td></tr><tr><td>ViT-L-16-SigLIP-256</td><td>timm</td><td>86.0</td><td>73.8</td><td>85.1</td><td>59.4</td><td>70.8</td></tr><tr><td>VICReg (Bardes et al., 2022)</td><td>ResNet50</td><td>github</td><td>73.0</td><td>59.9</td><td>71.7</td><td>40.2</td><td>53.8</td></tr></table>

Table 3. Foundation Models Studied. Overview of the pretrained models evaluated in this work, spanning diverse SSL and visionlanguage objectives and architectures (ResNet, ViT). Top-1 accuracy is reported on ImageNet, ImageNet-v2, and additional fine-grained datasets SUN397, iNat-18, iNat-21. Model weights are loaded from the official repositories or timm.

# B.2. Models studied in Section 4.1

The checkpoints used were either downloaded from the original gihub or timm. The complete list of pretraiend models is reported in Table 3.

# B.3. Models trained in Section 4.3

The complete list of models pretrained can be found below.

VICReg (Bardes et al., 2022). VICReg maximizes the informational content of embeddings by regularizing their empirical covariance matrix.

VICReg’s loss is defined with three components: (i) a term to encourage the variance (diagonal of the covariance matrix) inside the current batch to be equal to 1, preventing collapse with all the inputs mapped on the same vector; (ii) and a correlation regularization, encouraging the off-diagonal coefficients of the empirical covariance matrix to be close to 0, decorrelating the different dimensions of the embeddings. (iii) an invariance loss that matches positive pairs

We pre-trained ResNet-50 for 100 epochs using LARS, the projector used is an MLP with intermediate dimensions (8192, 8192, 2048), with a batch size of 2048, following the protocol of (Garrido et al., 2023)

$1 . \ \mathrm { { 1 x } } { \mathrm { { : } } } \ \mathrm { { w d } } = 1 e - 6 , \mathrm { { l r } } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \} , \operatorname { i n v } : 2 5 , \mathrm { { c o v } } : 5 , \mathrm { { v a r } } : 2 5 $

2. wd: lr = 0.3, wd ∈ {1e − 7, 1e − 6, 1e − 5, 1e − 4, 1e − 3}, inv : 25, cov : 5, var : 25   
3. cov.: lr = 0.3, wd = 1e − 6, inv : 25, cov : 5, var : 25 cov ∈ {0.4, 0.6, 0.8, 1, 4, 16}, var : 25

DINO (Caron et al., 2021). DINO uses a student-teacher framework. Two versions of the same network (the student and the teacher) are fed different views of the same image. The student is trained to match the teacher’s output probability distribution. To prevent the model from collapsing (i.e., giving the same output for every image), DINO uses a unique centering and sharpening operation on the teacher’s outputs. The teacher’s weights are updated as an exponential moving average of the student’s weights.

We pre-trained ViT-S for 100 epochs using Adam-W. The projector used is an MLP with intermediate dimensions (8192, 8192, 32768), with a batch size of 2048:

1. lr: lr ∈ {1.25e − 4, 2.25e − 4, 0.0025, 0.002, 0.0075}, t-temp. = 0.04, s-temp = 0.07   
2. s-temp: lr = 0.002, t-temp. = 0.04, s-temp = {0.07, 0.1, 0.2, 0.3, 0.4},   
3. t-temp: lr = 0.002, t-temp. = {0.01, 0.02, 0.03, 0.04, 0.05}, s-temp = 0.07,

I-JEPA (Assran et al., 2023) I-JEPA is a Joint-Predictive Architecture. It uses a context block to predict the representations of several target blocks from the same image. The context encoder is a Vision Transformer (ViT), which only processes the visible context patches. The predictor is a smaller ViT which takes the context encoder output and, conditioned on positional tokens, predicts the representations of a target block at a specific location. The weights of the target encoder are updated at each iteration via an exponential moving average of the context encoder weights.

We pre-trained ViT-B for 300 epochs with the same protocol as in the original papers with a batch size of 4096:

1. lr: wd sch = [0.04−0.4], lr ∈ {4e−5, 8e−5, 9e−5, 1e−4, 1.25e−4, 2e−4, 3e−4}, target block size = {0.15, 0.2}, context block size = {0.85, 1.0}   
2. Target Size Block (the size of the target block): wd sch = [0.04 − 0.4], lr = 1.25e − 5, target block size ∈ {{0.075, 0.2}, {0.1, 0.2}, {0.125, 0.2}, {0.2, 0.25}, {0.2, 0.25}}}, context block size = {0.85, 1.0}   
3. Context Size (the size of the context block): wd sch = [0.04−0.4], lr = 1.25e−5, target block size = {0.15, 0.2}, context block size ∈ {{0.4, 1.0}, {0.5, 1.0}, {0.65, 1.0}, {0.75, 1.0}, {0.90, 1.0}}

# C. Additional hyperparameter selection results.

To further validate IDES T’s performance on ResNet, we conducted additional experiments on DINO with a ResNet-50 varying s-temp (the student temperature) and the t-temp (teacher temperature), and with the all column in which methods must select from the full pool of hyperparameter configurations. The results are reported in Table 4

<table><tr><td rowspan="2">Method</td><td colspan="4">DINO (ResNet-50)</td></tr><tr><td>Unsup.</td><td>s-temp.</td><td>t-temp.</td><td>all</td></tr><tr><td>ACC-1 Bounds</td><td></td><td>[57.9, 67.5]</td><td>[63.0, 68.4]</td><td>[57.9, 68.4]</td></tr><tr><td>α-ReQ</td><td>√</td><td>61.9</td><td>63.0</td><td>63.0</td></tr><tr><td>RankMe</td><td>√</td><td>61.9</td><td>67.3</td><td>67.3</td></tr><tr><td>LiDAR</td><td>×</td><td>65.5</td><td>67.3</td><td>67.3</td></tr><tr><td>IDEST</td><td>√</td><td>67.5</td><td>67.6</td><td>67.6</td></tr></table>

Table 4. Unsupervised model selection with IDES T for DINO with a ResNet-50 backbone. We evaluate IDES T for hyperparameter selection against a supervised linear probe on ImageNet-1k, two unsupervised baselines: α-ReQ (Agrawal et al., 2022) and RankMe (Garrido et al., 2023), and a weakly-supervised one: LiDAR (Thilak et al., 2024). Hyperparameters are jointly selected across all hyperparameter axes according to Equation (4). Bold values indicate the top-performing model selected by the criteria.

# D. Compute cost

All trainings were performed on H100 GPUs. The total computational cost of the project, including training baselines, experiments, and ablation studies, amounts to approximately 12,000 GPU-hours.

# E. Further study of Foundation Models

# E.1. Additional properties studied in Joint-Embedding Self-Supervised Learning

![](images/fe6e136b74fc2a3f2f3bb7b38a4c896ee82840d62660d4f0f511b8727df9d2bb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Images"] -->|t~T| B["Transformed images"]
    B --> C["X"]
    C --> D["f"]
    D --> E["Y"]
    E --> F["h"]
    F --> G["Z"]
    G --> H["Embeddings"]
    I["I"] -->|t'~T| J["X'"]
    J --> K["f"]
    K --> L["Y'"]
    L --> M["h"]
    M --> N["Z'"]
    N --> O["Maximize agreement"]
```
</details>

Figure 7. Overview of Joint-Embedding Architectures. Two separate data augmentation operators are sampled from the same distribution $( t , t ^ { \prime } \sim \tau )$ and applied to each image in a batch I to obtain two views, X and X′. An encoder network $f$ and a projection head g are trained to maximize agreement between the resulting embeddings. After training, the encoder $f$ and its representations Y are retained for downstream tasks.

A dominant approach in SSL is joint embedding self-supervised learning (JE-SSL) (Chen et al., 2020; Bardes et al., 2022), where two networks are trained to produce similar embeddings for different views of the same image. An overview is presented in Figure 7. Several metrics have been proposed to assess the quality of the learned representations in these methods.

Uniformity metrics. Uniformity is a desirable properties of Joint-Embedding Self-Supervised methods (Wang & Isola, 2020; Fang et al., 2024; Mordacq et al., 2025), with the intuition that vectors should be roughly uniformly distributed on the unit hypersphere $S ^ { m - 1 }$ , preserving as much information of the data as possible. Two main metrics have been proposed:

• $\mathcal { L } _ { u }$ (Wang & Isola, 2020), based on the gaussian pairwise kernel:

$$
\mathcal {L} _ {u} = \log \underset {x, y \sim p _ {\text { data }}} {\mathbb {E}} \exp^ {- t | | f (x) - f (y) | | _ {2} ^ {2}}, t > 0. \tag {5}
$$

• $\mathcal { W } _ { 2 }$ (Fang et al., 2024), the quadractic Wasserstein distance between the distribution of the learned representation and $\mathcal { N } ( 0 , I _ { M } / m )$ :

$$
\mathcal {W} _ {2} := \sqrt {\left| \left| \hat {\mu} \right| \right| _ {2} ^ {2} + 1 + \operatorname{tr} \left(\hat {\Sigma}\right) - \frac {2}{m} \operatorname{tr} \left(\hat {\Sigma} ^ {\frac {1}{2}}\right)} \tag {6}
$$

where $\hat { \mu } , \hat { \Sigma }$ are the sample mean and covariance mean.

Though neither was designed with model comparison or hyperparameter selection in mind, we nonetheless investigate their potential across SSL paradigms.

RankMe (Garrido et al., 2023). RankMe is formally the smooth rank measure, originally introduced by Roy & Vetterli (2007):

$$
\operatorname{RankMe} (Z) = \exp (- \sum_ {k = 1} ^ {\min (N, K)} p _ {k} \log p _ {k}), \text { with } p _ {k} = \frac {\sigma_ {k} (Z)}{| | \sigma (Z) | | _ {1}} + \epsilon \tag {7}
$$

where $Z$ is the representations obtained.

![](images/078c80c181861cc2604e95f88d5ed24cfac4180214cf899adb71f9a07647a1f9.jpg)  
Figure 8. Alternative metrics and Foundation Models. Linear probing accuracy on ImageNet versus (Top) RankMe and (Bottom) uniformity metrics $( \mathcal { L } _ { u }$ and W2) for a diverse set of pretrained SSL models.

1-d Helical Manifold   
![](images/b0587ea518db22b58960de8813068a385059028d360ae6207d9c992a88757faf.jpg)

<details>
<summary>surface_3d</summary>

| x       | y       |
| ------- | ------- |
| -0.02   | -1.00   |
| -0.50   | -0.50   |
</details>

Figure 9. Effective vs. intrinsic dimensionality. A helix is embedded in a 3-dimensional ambient space (high effective dimension), yet is intrinsically a 1-dimensional manifold: point A requires three coordinates to specify its position in the ambient space, but a single arc-length coordinate suffices to locate it along the curve. The two notions thus capture complementary aspects of geometry—how spread out a representation is versus how many degrees of freedom it truly contains.

# E.2. Foundation Models and further metrics.

In Figure 8 we study whether metrics initially proposed for JE-SSL can reflects representation quality across a diverse set of pretrained SSL models. Specifically, we compute RankMe, $\mathcal { L } _ { u } , \mathcal { W } _ { 2 }$ on frozen representations and compare it against standard linear probing accuracy on ImageNet.

For RankMe (Garrido et al., 2023), the original study noted that it is not suited for comparisons across architectures. A key reason is that RankMe is bounded by the output dimension of the model: two models of different architectures that both span their full representation space will yield different values, making cross-architecture comparisons unreliable (as highlighted in Figure 8a, where DINOv3 ViT-7B is a clear outlier compared to DINOv3 ViT-L, despite similar ImageNet accuracies).

Concerning uniformity metrics (see Figure 8b), neither $\mathcal { L } _ { u }$ nor W2 correlates consistently with linear probing accuracy across models.

Overall, all three metrics retain some predictive signal, as reflected by non-trivial Kendall’s τ and Spearman’s ρ values, yet their limited correlation with accuracy motivates the search of alternative geometric proxies.

# F. Effective and Intrinsic Dimensions

RankMe (Garrido et al., 2023) and IDES T capture complementary notions of dimensionality. The former measures effective rank, the entropy of the singular-value distribution of the feature matrix, reflecting how uniformly the representation spreads across linear dimensions (Roy & Vetterli, 2007). The latter estimates intrinsic dimension: the minimum number of degrees of freedom required to describe the data on its underlying manifold. Figure 9 illustrates this distinction with a helix, whose points require three ambient coordinates yet lie on a curve parameterised by a single value.

RankMe correlates less strongly with downstream accuracy across foundation models (Figure 8; $\rho = 0 . 4 9 , \tau = 0 . 3 9 )$ ) than IDES T (Figure 1, $\rho = - 0 . 7 4 , \tau = - 0 . 5 5 )$ , confirming that intrinsic dimension captures information about representation quality beyond what linear spread alone can reveal. Yet RankMe’s substantial correlation suggests that the effective rank of representations remains a useful proxy for downstream performance.

Figure 10 tracks the training dynamics of DINO (ViT-S) and reveals a consistent trend: IDES T decreases while RankMe increases throughout training. This inverse relationship admits a natural information-bottleneck interpretation: high-quality representations compress the input onto a compact, low-dimensional manifold (low intrinsic dimension) while spreading that information uniformly across ambient dimensions (high effective rank) to avoid collapse.

![](images/94e3fe7ec0577690446e0c8cb6c4af7c11d16e030fa330ed38ba2caddb665c23.jpg)  
Figure 10. Tracking Training Dynamics: RankMe and IDES T. Evolution of the self-supervised loss, ImageNet-1k online classification top-1 accuracy, IDES T and RankMe. IDES T decreases while RankMe increases throughout training.

As hypothesized in (Ansuini et al., 2019), the gap between effective and intrinsic dimension relates to the curvature of the representation manifold: a flat manifold embedded in $\mathbb { R } ^ { d }$ has matching intrinsic and effective dimensions, whereas a highly curved one can occupy many ambient dimensions while remaining intrinsically low-dimensional, as illustrated by the helix in Figure 9. Differential geometry offers a principled framework to formalise this gap; curvature-aware metrics are a natural direction for future work to further disentangle the structure of learned representations.