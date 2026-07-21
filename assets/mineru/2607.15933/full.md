# Distributional Matching for Vector Quantization: A Unified Theoretical and Empirical Framework

Xianghong Fang $^{1*}$ Litao Guo $^{2}$ Hengchao Chen $^{1}$ Yuxuan Zhang $^{1}$ XiaofanXia $^{1}$ Dingjie Song $^{4}$ Yexin Liu $^{2}$ Hao Wang $^{5}$ Harry Yang $^{2}$ Qiang Sun $^{1}$ Yuan Yuan $^{3*}$

$^{1}$ University of Toronto $^{2}$ The Hong Kong University of Science and Technology $^{3}$ Boston College $^{4}$ Lehigh University $^{5}$ Southern University of Science and Technology
Website Code & Models

## Abstract

The effectiveness of modern visual representation learning and autoregressive models critically depends on vector quantization (VQ), which discretizes continuous feature representations using a learnable codebook. Despite its widespread use, existing VQ methods often suffer from training instability and codebook collapse, arising from gradient mismatch induced by the straight-through estimator and the under-utilization of code vectors. In this work, we show that both issues can be traced to a fundamental mismatch between the distributions of feature vectors and code vectors, leading to inefficient representation and information loss. Building on this observation, we propose a distributional matching framework for vector quantization. We introduce principled criteria for desirable VQ behavior and demonstrate through theoretical analysis and empirical evaluation that aligning feature and code vector distributions provides a unifying mechanism for mitigating training instability and codebook collapse. We instantiate this framework using a Wasserstein-based objective with an efficient closed-form under a mild Gaussian approximation, and further show that a nonparametric alternative based on maximum mean discrepancy yields comparable performance. Extensive experiments on visual tokenization benchmarks support the effectiveness and robustness of the proposed approach.

## 1 Introduction

Vector quantization (VQ) $[49]$ is a fundamental building block in a wide range of modern representation learning and visual tokenization frameworks, including autoregressive and hybrid generative models $[39, 10, 7, 28, 15, 47, 29]$ . By discretizing continuous feature representations using a learnable codebook, VQ enables compact and structured latent representations that are well suited for downstream modeling. Despite its broad adoption, VQ remains notoriously difficult to optimize in practice, often exhibiting unstable training dynamics and severe codebook collapse.

The first challenge stems from the non-differentiability of the quantization operation, which prevents gradients from being directly propagated from quantized features to their continuous counterparts. To address this issue, prior work introduces the straight-through estimator (STE) $[2, 49]$ , which approximates gradients by copying them from the quantized features to encoder outputs. However, the effectiveness of this approximation critically depends on the magnitude of the quantization error. When the discrepancy between continuous features and their assigned code vectors becomes large, the resulting gradient mismatch can lead to unstable optimization and degraded training behavior $[28]$ .

A second, closely related challenge is codebook collapse, where only a small subset of code vectors receives assignments while the majority remain unused. From a geometric perspective, this corresponds to a degenerate Voronoi partition $^{1}$ in which most cells are never activated $[59]$ . Although extensive research has sought to alleviate this problem, low code vector utilization often remains in practice, particularly with large codebooks $[9, 46, 53, 28, 59]$ . This limitation is further exacerbated as increasing the codebook size expands the number of Voronoi cells, making it substantially harder to ensure that all cells are sufficiently populated.

In this work, we examine these challenges from a distributional perspective. Rather than treating training instability and codebook collapse as separate phenomena, we observe that both issues are closely tied to a fundamental mismatch between the distributions of feature vectors produced by the encoder and the distributions of the learnable code vectors. Figure 1 illustrates this intuition using two representative scenarios. When the two distributions are poorly aligned, feature vectors concentrate around a small subset of code vectors, resulting in In contrast, when the distributions are well utilized approaches its maximum.

![](images/822441d8abc0efcf37e97174b0e7fb332b49b1bc1c39289b99123ac72fc9c689.jpg)

![](images/9354685db1a04af04d46964a964e6c42ac7bde7e767096611fa1a33ddbc0356b.jpg)  
Distributional Mismatch  
Figure 1: The symbols $\cdot$ and $\times$ represent the feature and code vectors, respectively. The left figure illustrates the distributional mismatch between the feature and code vectors, while the right figure visualizes their distributional match.  
Distributional Match

a small subset of code vectors, resulting in large quantization errors and low codebook utilization. In contrast, when the distributions are well matched, quantization errors are reduced and codebook utilization approaches its maximum.

Building on this observation, we introduce three principled criteria that characterize the desirable behavior in vector quantization. Guided by this criterion triple, we show through both theoretical analysis and empirical evaluation that distributional alignment between feature vectors and code vectors provides a unifying mechanism for mitigating training instability and codebook collapse. To operationalize this idea, we adopt a distribution matching objective based on the quadratic Wasserstein distance. Under a mild Gaussian approximation, this objective admits a closed-form expression that can be efficiently optimized during training. Importantly, we show that even when this approximation is relaxed, a nonparametric alternative based on maximum mean discrepancy (MMD) $[14, 44, 12]$ yields comparable performance, suggesting that distributional matching effectively captures the essential structure required for effective vector quantization. These insights translate into consistent improvements in reconstruction fidelity across visual tokenization benchmarks.

## 2 A Distribution Matching Perspective on Vector Quantization

This section introduces a novel distributional perspective for understanding vector quantization. By defining three principled criteria for VQ evaluation, we provide empirical and theoretical evidence that aligning feature and code vector distributions yields a near-optimal VQ solution.

## 2.1 An Overview of Vector Quantization

As a core component of visual tokenizers $[49, 28, 47]$ , VQ acts as a compressor that discretizes continuous latent features into discrete visual tokens by mapping them to the nearest code vectors within a learnable codebook.

Figure 2 illustrates the classic VQ process [49], which consists of an encoder $E(\cdot)$ , a decoder $D(\cdot)$ , and an updatable codebook $\{\mathbf{e}_k\}_{k=1}^K \in \mathbb{R}^d$ containing a finite set of code vectors. Here, $K$ denotes the codebook size and $d$ the code vector dimension. Given an image $\boldsymbol{x} \in \mathbb{R}^{H \times W \times 3}$ , the goal is to derive a spatial collection of codeword IDs $r \in \mathbb{N}^{h \times w}$ as image tokens. This is achieved by encoding the image to obtain

![](images/e48c6b82b9686fa69d3f083e0a2b2bd0e3ab38d86188c8b16cd32ca1657822eb.jpg)  
Figure 2: The illustration of VQ.

$z_{e}=E(\boldsymbol{x})\in\mathbb{R}^{h\times w\times d},$ followed by a spatial quantizer $\mathcal{Q}(\cdot)$ that maps each spatial feature $z_{e}^{ij}$ to its

nearest code vector $e_k$ :

$$
r ^ {i j} = \underset {k} {\arg \min} \| \pmb {z} _ {e} ^ {i j} - \pmb {e} _ {k} \| _ {2} ^ {2}.\tag{1}
$$

The resulting tokens retrieve the corresponding codebook entries $z_{q}^{ij} = \mathcal{Q}(z_{e}^{ij}) = e_{r^{ij}}$ , which are then passed through the decoder to reconstruct the image as $\widehat{\boldsymbol{x}} = D(\boldsymbol{z}_{q})$ . Despite its widespread adoption in visual tokenization, representation learning, and high-fidelity image synthesis [10], VQ still faces two key challenges: training instability and codebook collapse.

Training Instability This issue arises because during backpropagation, the gradient of $z_{q}$ cannot flow directly to $z_{e}$ due to the non-differentiable function $\mathcal{Q}$ . To optimize the encoder's parameters, VQ-VAE [49] employs the straight-through estimator (STE) [3], which copies gradients from $z_{q}$ to $z_{e}$ . However, this approach carries significant risks, especially when $z_{q}$ and $z_{e}$ are far apart. In such cases, the gradient gap can grow substantially, destabilizing training $^{2}$ . In this work, we tackle the training instability challenge from a distributional viewpoint. This perspective highlights that training instability is not merely an implementation artifact, but a consequence of systematic mismatch between feature and code distributions.

Codebook Collapse Codebook collapse occurs when only a small subset of code vectors receives gradients, while most remain unrepresentative and unupdated $[9, 53, 28, 59]$ . Researchers have proposed solutions such as improved codebook initialization $[60]$ , reinitialization strategies $[9, 51]$ , and classical clustering algorithms like k-means $[5]$ and k-means++ $[1]$ for codebook optimization $[39, 59]$ . Beyond deterministic approaches that select the best-matching token, stochastic quantization strategies have also been explored $[56, 38, 46]$ . However, these methods still exhibit low code vector utilization, particularly with large codebook sizes K $[59, 33]$ . In this paper, we address this issue via distributional matching between feature and code vectors.

## 2.2 Evaluation Criteria

While these quantities have appeared individually in prior work, we emphasize that considering them jointly provides a unified lens for analyzing both optimization stability and codebook collapse.

Given feature vectors $\{z_{i}\}_{i=1}^{N}$ from feature distribution $P_{A}$ and code vectors $\{e_{k}\}_{k=1}^{K}$ sampled from codebook distribution $P_{B}$ , vector quantization involves finding the nearest, and thus most representative, code vector for each feature:

$$
\boldsymbol {z} _ {i} ^ {\prime} = \underset {\boldsymbol {e} \in \{\boldsymbol {e} _ {k} \}} {\arg \min} \| \boldsymbol {z} _ {i} - \boldsymbol {e} \|.\tag{2}
$$

Each original feature vector $z_{i}$ is then quantized to $z_{i}^{\prime}$ . We denote the index of the selected code vector for the i-th feature as $r_{i}$ , such that $z_{i}^{\prime} = e_{r_{i}}$ . We next introduce three key criteria to evaluate this process.

Criterion 1 (Quantization Error). The quantization error measures the average distortion introduced and is defined as:

$$
\mathcal {E} (\{e _ {k} \}; \{\boldsymbol {z} _ {i} \}) = \frac {1}{N} \sum_ {i} \| \boldsymbol {z} _ {i} - \boldsymbol {z} _ {i} ^ {\prime} \| ^ {2}.\tag{3}
$$

A smaller E signifies a more accurate quantization of the original feature vectors. Beyond this geometric interpretation, E is also directly linked to training stability. As formally derived in Appendix C, the gradient discrepancy $G_{i}$ for a feature $z_{i}$ introduced by the straight-through estimator (STE) is theoretically bounded by its distance to the code vector:

$$
\mathcal {G} _ {i} \leq \| \mathbf {H} \| _ {2} \cdot \| \boldsymbol {z} _ {i} - \boldsymbol {z} _ {i} ^ {\prime} \|,\tag{4}
$$

where $H = \left. \frac{\partial^{2} L}{\partial x^{2}} \right|_{x = z_{i}^{\prime}}$ denotes the Hessian of the overall training loss L with respect to the latent input, evaluated at the quantized code $z_{i}^{\prime}$ , characterizing the local curvature of the loss landscape. Since E is the mean squared magnitude of the term $\|z_{i} - z_{i}^{\prime}\|$ , minimizing E explicitly tightens this bound. This ensures that the approximated gradients remain faithful to the true gradients, effectively mitigating the gradient mismatch that causes training instability.

Criterion 2 (Codebook Utilization Rate). The codebook utilization rate is the fraction of code vectors used in VQ

$$
\mathcal {U} (\{\boldsymbol {e} _ {k} \}; \{\boldsymbol {z} _ {i} \}) = \frac {1}{K} \sum_ {k = 1} ^ {K} \mathbf {1} \left(k \in \{r _ {i} \} _ {i = 1} ^ {N}\right),\tag{5}
$$

where the indicator evaluates to 1 if code vector $e_{k}$ is selected by at least one feature.

A higher U reduces the risk of codebook collapse. Ideally, U should reach 100%, meaning all code vectors are utilized. As discussed in Appendix D, U measures only the completeness of codebook utilization and cannot evaluate the degree of collapse. This motivates introducing the codebook perplexity criterion.

Criterion 3 (Codebook Perplexity). The codebook perplexity measures the uniformity of codebook utilization in VQ

$$
\mathcal {C} (\{\boldsymbol {e} _ {k} \}; \{\boldsymbol {z} _ {i} \}) = \exp (- \sum_ {k = 1} ^ {K} p _ {k} \log p _ {k}),\tag{6}
$$

where $p_{k} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(z_{i}' = e_{k})$ . A higher C indicates more uniform selection of code vectors in VQ. Ideally, C reaches its maximum at $C_{0} = \exp(-\sum_{k=1}^{K} \frac{1}{K} \log \frac{1}{K}) = K$ when code vectors are completely uniformly utilized. Thus, as a complement to Criterion 2, U combined with C effectively evaluates codebook collapse.

We refer to $(\mathcal{E},\mathcal{U},\mathcal{C})$ as the criterion triple. When comparing extreme cases of distributional match and mismatch shown in Figure 1, we find that distributional matching significantly outperforms mismatching across all three criteria. Using this triple, we can analyze the benefits of distribution matching more systematically.

Quantization Error vs. Codebook Utilization Rate Quantization error ( $E$ ) is a more fundamental objective than codebook utilization (U) in VQ. We theoretically show in [11] that minimizing E necessarily induces full codebook utilization, whereas the converse does not hold. This hierarchical relationship suggests treating E as the primary optimization target, with U serving as a complementary metric to monitor coverage.

Remark Notably, E is sensitive to the variance of the latent feature distribution (Appendix E). Consequently, direct comparisons of raw E values across latent spaces with different variances can be misleading. To fairly evaluate quantization methods via criterion triple, all comparisons should be made under identical latent distributions.

## 2.3 The Effects of Distribution Matching

We conduct a deliberately simple synthetic experiment to isolate the geometric effects of distribution mismatch (see details in Appendix J.1). Specifically, we assume that $P_{A}$ and $P_{B}$ are uniform distributions within two distinct disks, as shown in Figure 3. We sample feature vectors $\{z_{i}\}_{i=1}^{N}$ from the red disk and code vectors $\{e_{k}\}_{k=1}^{K}$ from the green circle. The criterion triple $(\mathcal{E},\mathcal{U},\mathcal{C})$ is calculated by Criteria 1 to 3.

We examine two cases. The first involves disks with identical radii but different centers. As shown in Figures 3a to 3d, when the centers move closer, the criterion triple improves toward optimal values. Specifically, E decreases from 1.19 to 0.05, U rises from 2% to 100%, and C increases from 3.8 to 344.9. The second case shows distributions with identical centers but different radii. When the codebook support lies within the feature support (Figures 3e and 3f), E is larger, U slightly lower, and C smaller compared to the aligned distributions in Figure 3d. When the codebook support extends beyond the feature support, E increases modestly while U and C drop significantly (Figures 3g and 3h). Detailed explanations are provided in Appendix F.

Overall, VQ achieves the optimal criterion triple when feature and codebook distributions are identical. This is further supported by quantitative analyses in Appendix G.

![](images/f223e2439736863327c061712c3331b64c7f7fab423cfaf255e05abe9dcfec35.jpg)  
(e) (0.36, 93.3%, 63.2) (f) (0.10, 99.8%, 250.5) (g) (0.07, 61.3%, 199.7) (h) (0.08, 45.3%, 151.5)  
Figure 3: Qualitative analyses of the criterion triple $(\mathcal{E},\mathcal{U},\mathcal{C})$ : The red and green disks represent the uniform distributions of feature and code vectors, respectively.

## 2.4 Theoretical Analyses

In this section, we provide theoretical evidence to support our empirical observations. Let the code vectors $\{e_{k}\}_{k=1}^{K}$ and feature vectors $\{z_{i}\}_{i=1}^{N}$ be independently and identically drawn from $P_{B}$ and $P_{A}$ , respectively. We say a codebook $\{e_{k}\}_{k=1}^{K}$ attains full utilization asymptotically with respect to $\{z_{i}\}_{i=1}^{N}$ if the codebook utilization rate $\mathcal{U}(\{e_{k}\}_{k=1}^{K};\{z_{i}\}_{i=1}^{N})$ tends to 1 in probability as N approaches infinity:

$$
\mathcal {U} (\{\boldsymbol {e} _ {k} \} _ {k = 1} ^ {K}; \{\boldsymbol {z} _ {k} \} _ {i = 1} ^ {N}) \xrightarrow {p} 1, \quad \mathrm{as} N \to \infty .\tag{7}
$$

For the codebook distribution $P_{B}$ , we say it attains full utilization asymptotically with respect to $P_{A}$ if, with probability 1, the randomly generated codebook $\{e_{k}\}_{k=1}^{K}$ achieves full utilization asymptotically.

Additionally, a codebook distribution $P_{B}$ is said to have vanishing quantization error asymptotically with respect to a domain $\Omega \subseteq R^{d}$ if the quantization error over all data of size N tends to zero in probability as K approaches infinity:

$$
\sup _ {\{\boldsymbol {z} _ {i} \} \subseteq \Omega} \mathcal {E} (\{\boldsymbol {e} _ {k} \} _ {k = 1} ^ {K}; \{\boldsymbol {z} _ {i} \} _ {i = 1} ^ {N}) \xrightarrow {p} 0, \quad \text {as} K \to \infty .\tag{8}
$$

Our first theorem shows that $\overline{\operatorname{supp}(\mathcal{P}_{A})} = \overline{\operatorname{supp}(\mathcal{P}_{B})}$ is sufficient and necessary for the codebook distribution $P_{B}$ to attain both full utilization and vanishing quantization error asymptotically. For simplicity, $P_{A}$ is assumed to have a density function $f_{A}$ with bounded support $\Omega \subseteq R^{d}$ .

Theorem 1. Assume $\Omega = \mathrm{supp}(\mathcal{P}_A)$ is a bounded open area. The codebook distribution $\mathcal{P}_B$ attains full utilization and vanishing quantization error asymptotically if and only if $\overline{\mathrm{supp}(\mathcal{P}_B)} = \overline{\mathrm{supp}(\mathcal{P}_A)}$ , where $\overline{\mathcal{S}}$ denotes the closure of the set $\mathcal{S}$ .

Theorem 1 establishes the optimal support of the codebook distribution. The boundedness of $\Omega$ is required as we consider the worst case quantization error in equation 8. In real applications, when $P_{A}$ follows an absolutely continuous distribution over an unbounded domain, then $\{z_{i}\}_{i=1}^{N}$ generated from $P_{A}$ will be bounded with high probability. Thus, Theorem 1 also provides theoretical insights for a target distribution $P_{A}$ with an unbounded domain.

Besides the optimal support, we also determine the optimal density of the codebook distribution by invoking existing results characterizing asymptotic optimal quantizers [13]. Specifically, we consider the case where $N$ approaches to infinity and define the expected quantization error of a codebook $\{e_k\}$ with respect to $\mathcal{P}_{\mathcal{A}}$ as

$$
\mathcal {E} (\{e _ {k} \} _ {k = 1} ^ {K}; \mathcal {P} _ {A}) = \mathbb {E} _ {\boldsymbol {z} \sim \mathcal {P} _ {A}} \min _ {\boldsymbol {e} \in \{\boldsymbol {e} _ {k} \}} \| \boldsymbol {z} - \boldsymbol {e} \| ^ {2}.\tag{9}
$$

A codebook $\{e_k^* \}_{k=1}^K$ is called the set of optimal centers for $\mathcal{P}_A$ if it achieves the minimal quantization error:

$$
\mathcal {E} (\{\boldsymbol {e} _ {k} ^ {*} \} _ {k = 1} ^ {K}; \mathcal {P} _ {A}) = \min _ {\{\boldsymbol {e} _ {k} \} _ {k = 1} ^ {K}} \mathcal {E} (\{\boldsymbol {e} _ {k} \} _ {k = 1} ^ {K}; \mathcal {P} _ {A}).\tag{10}
$$

Intuitively, Theorem 1 shows that mismatched supports inevitably lead to either unused code vectors or large quantization error, whereas matching supports is both necessary and sufficient to avoid these failure modes.

Theorem 2 demonstrates that, under weak regularity conditions, the empirical measure of the optimal centers for $P_{A}$ converges in distribution to a fixed distribution determined by $P_{A}$ . Notably, we do not assume a bounded domain in the following theorem.

Theorem 2 (Theorem 7.5, [13]). Suppose $Z \sim \mathcal{P}_A$ is absolutely continuous w.r.t. the Lesbegue measure in $\mathbb{R}^d$ and $\mathbb{E}\| Z\|^{2 + \delta} < \infty$ for some $\delta > 0$ . Then the empirical measure of the optimal centers for $\mathcal{P}_A$ ,

$$
\frac {1}{K} \sum_ {k = 1} ^ {K} \delta_ {\boldsymbol {e} _ {k} ^ {*}},\tag{11}
$$

converges weakly to a fixed distribution $\mathcal{P}_A^*$ , whose density function $f_A^*$ is proportional to $f_A^{(d + 2) / d}$ .

High-dimensional Implication. Theorem 2 characterizes the asymptotically optimal codebook distribution $P_{A}^{*}$ , whose density satisfies $f_{A}^{*}(z) \propto f_{A}(z)^{(d+2)/d}$ . A key implication arises in high-dimensional settings typical of visual tokenization: as the feature dimension d increases, the exponent converges to unity,

$$
\lim _ {d \rightarrow \infty} \frac {d + 2}{d} = \lim _ {d \rightarrow \infty} \left(1 + \frac {2}{d}\right) = 1.\tag{12}
$$

As a result, the optimal codebook density $f_{A}^{*}$ becomes increasingly close to the feature density $f_{A}$ in the high-dimensional regime. This observation provides a rigorous theoretical justification for our approach: explicitly aligning the codebook distribution with the feature distribution (i.e., $P_{B} \approx P_{A}$ ) serves as a principled and effective surrogate for the asymptotically optimal design in high-dimensional latent spaces.

## 3 Methodology

Based on the theoretical insights in Section 2, which show that matching the feature and codebook distributions is both necessary and sufficient for asymptotically optimal quantization, we propose a general framework for enhancing vector quantization via distributional matching. This framework can be instantiated using either parametric objectives (e.g., the Wasserstein distance for efficiency) or nonparametric alternatives (e.g., Maximum Mean Discrepancy, which offers robustness and adaptability to general distributions). In the main text, we focus primarily on the Wasserstein instantiation due to its computational efficiency arising from a closed-form solution. Finally, we integrate this objective into two representative VQ frameworks, namely VQ-VAE [49] and VQGAN [10].

## 3.1 A General Distributional Matching Framework

To resolve the fundamental mismatch between the feature distribution $P_{A}$ and the codebook distribution $P_{B}$ , we introduce an auxiliary alignment loss $L_{match}$ to the standard VQ objective. The goal is to minimize the divergence D between the two distributions:

$$
\mathcal {L} _ {\text { match }} = \mathcal {D} (\mathcal {P} _ {A}, \mathcal {P} _ {B}),\tag{13}
$$

where $P_{A}$ is empirically defined by the encoder outputs $\{z_{e}\}$ and $P_{B}$ by code vectors $\{e_{k}\}$ . Crucially, gradients from $L_{match}$ are only back-propagated to the codebook, preserving encoder feature expressiveness.

## 3.2 Wasserstein-Based Distribution Matching

We consider a parametric instantiation of the proposed framework using a mild Gaussian approximation to enable computationally efficient distributional matching. Specifically, we approximate the feature and codebook distributions as $\mathcal{P}_{A} \approx \mathcal{N}(\boldsymbol{\mu}_{1}, \boldsymbol{\Sigma}_{1})$ and $\mathcal{P}_{B} \approx \mathcal{N}(\boldsymbol{\mu}_{2}, \boldsymbol{\Sigma}_{2})$ . Under this approximation, we employ the quadratic Wasserstein distance, as defined in Appendix H, which admits a computationally efficient closed-form solution. Although other statistical distances, such as the

![](images/ed4557843265f9e77904d278f53b4b01fff9f598ea141f53c16e54aba0f07c0e.jpg)  
Figure 4: Illustration of the Wasserstein VQ. The architecture integrates an encoder-decoder network with a VQ module. In the VQ module, we augment the vanilla VQ framework [49] by incorporating our proposed Wasserstein loss $L_{W}$ to achieve distributional matching between features $z_{e}\left(z_{e}^{ij} \sim \mathcal{P}_{A}\right)$ and the codebook $e_{k}\left(e_{k} \sim \mathcal{P}_{B}\right)$ .

Kullback-Leibler divergence $[26, 17]$ , are viable alternatives, they typically lack simple closed-form representations, rendering them computationally prohibitive in high-dimensional settings. Consequently, we adopt the quadratic Wasserstein distance, whose closed-form representation for Gaussian distributions is provided by the following lemma:

Lemma 3 ([36]). The quadratic Wasserstein distance between $\mathcal{N}(\boldsymbol{\mu}_{1},\boldsymbol{\Sigma}_{1})$ and $\mathcal{N}(\boldsymbol{\mu}_{2},\boldsymbol{\Sigma}_{2})$

$$
\sqrt {\| \boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {2} \| _ {2} ^ {2} + \mathrm{tr} (\boldsymbol {\Sigma} _ {1} + \boldsymbol {\Sigma} _ {2} - 2 (\Sigma_ {1} ^ {\frac {1}{2}} \boldsymbol {\Sigma} _ {2} \boldsymbol {\Sigma} _ {1} ^ {\frac {1}{2}}) ^ {\frac {1}{2}})}.\tag{14}
$$

The lemma shows that the quadratic Wasserstein distance admits a closed-form expression in terms of the means and covariance matrices of the two distributions. In practice, we estimate these population quantities, $\mu_{1}$ , $\mu_{2}$ , $\Sigma_{1}$ , and $\Sigma_{2}$ , with their sample counterparts: $\widehat{\mu}_{1}$ , $\widehat{\mu}_{2}$ , $\widehat{\Sigma}_{1}$ , and $\widehat{\Sigma}_{2}$ . The empirical quadratic Wasserstein distance is then used as the optimization objective to align the feature and codebook distributions:

$$
\mathcal {L} _ {\mathcal {W}} = \sqrt {\| \widehat {\boldsymbol {\mu}} _ {1} - \widehat {\boldsymbol {\mu}} _ {2} \| _ {2} ^ {2} + \mathrm{tr} (\widehat {\boldsymbol {\Sigma}} _ {1} + \widehat {\boldsymbol {\Sigma}} _ {2} - 2 (\widehat {\boldsymbol {\Sigma}} _ {1} ^ {\frac {1}{2}} \widehat {\boldsymbol {\Sigma}} _ {2} \widehat {\boldsymbol {\Sigma}} _ {1} ^ {\frac {1}{2}}) ^ {\frac {1}{2}})}.
$$

A smaller value of $L_{W}$ indicates stronger alignment between the feature distribution $P_{A}$ and the codebook distribution $P_{B}$ . We refer to the VQ algorithm that employs $L_{W}$ as Wasserstein VQ.

Remark on the Gaussian Approximation The Gaussian approximation is introduced solely to obtain a closed-form and computationally efficient instantiation of the proposed distributional matching objective, simplifying the complex form in Definition 4 in Appendix H into Lemma 3. Importantly, this approximation does not constrain the learned feature representations: the Wasserstein loss $L_{W}$ is applied only to update the codebook parameters, while gradients to the encoder are detached. As a result, the encoder remains free to learn arbitrarily complex feature distributions.

In practice, latent representations in visual tokenization often exhibit approximately Gaussian statistics, a behavior commonly observed in deep representations (see Section 5 and Appendix I for empirical validation). One possible explanation is that multivariate normal distributions are information-theoretically advantageous, as they maximize entropy for a given covariance, thereby enabling efficient information transmission and robust signal reconstruction. As a result, multivariate normal distributions naturally emerge as both an ideal and empirically observed form for latent representations in visual tokenization and related signal recovery tasks.

## 3.3 MMD-Based Distribution Matching

To address scenarios where the Gaussian assumption may not hold (e.g., highly multi-modal distributions), we provide a non-parametric instantiation using Maximum Mean Discrepancy (MMD) $[14, 44]$ . MMD measures the distance between distributions in a Reproducing Kernel Hilbert Space (RKHS) without assuming any specific parametric form:

$$
\mathcal {L} _ {\mathrm{MMD}} = \frac {1}{N ^ {2}} \sum_ {i, j} k (\boldsymbol {z} _ {i}, \boldsymbol {z} _ {j}) - \frac {2}{N K} \sum_ {i, k} k (\boldsymbol {z} _ {i}, \boldsymbol {e} _ {k}) + \frac {1}{K ^ {2}} \sum_ {k, l} k (\boldsymbol {e} _ {k}, \boldsymbol {e} _ {l}),\tag{15}
$$

where $e_{k}$ and $z_{i}$ denote code vectors and encoder output spatial feature vectors, respectively, and $k(\cdot,\cdot)$ is a kernel function (e.g., a Gaussian RBF kernel). We refer to this approach as MMD VQ [12]. Similar to Wasserstein VQ, the MMD loss $L_{MMD}$ is applied exclusively to update the codebook parameters, with gradients to the encoder detached.

Remark While $L_{MMD}$ offers greater theoretical robustness by relaxing the Gaussian hypothesis, it typically incurs higher computational cost ( $\mathcal{O}((N+K)^{2})$ ) compared to the closed-form Wasserstein distance. In our experiments, we observe that Wasserstein VQ achieves performance comparable to MMD VQ (see Section 6), indicating that the Gaussian approximation sufficiently captures the essential structure for effective tokenization while remaining computationally efficient. Therefore, In the following experiments, we focus on Wasserstein instantiation for exploring the distributional matching framework.

## 3.4 Integration into VQ Architectures

Our distributional matching framework is agnostic to the underlying VQ architecture. We integrate it into two representative frameworks: VQ-VAE and VQGAN.

## 3.4.1 VQ-VAE Integration

We first examine Wasserstein VQ within the VQ-VAE framework [49]. As shown in Figure 4, the VQ-VAE model has three key components: an encoder $E(\cdot)$ , a decoder $D(\cdot)$ , and a quantizer $\mathcal{Q}(\cdot)$ with a learnable codebook $\{\mathbf{e}_k\}_{k=1}^K$ . As described earlier in Section 2.1, for an input image $\boldsymbol{x}$ , the encoder produces a spatial feature $z_e = E(\boldsymbol{x}) \in \mathbb{R}^{h \times w \times d}$ . The quantizer maps $z_e$ to a quantized feature $z_q$ , which the decoder uses to reconstruct the image as $\widehat{\boldsymbol{x}} = D(z_q)$ . Incorporating our Wasserstein loss $\mathcal{L}_{\mathcal{W}}$ into the VQ-VAE framework, the overall loss is formulated as follows:

$$
\mathcal {L} _ {\mathrm{VQ-VAE}} = \| \widehat {\pmb {x}} - \pmb {x} \| _ {2} ^ {2} + \beta \| \mathrm{sg} (\pmb {z} _ {q}) - \pmb {z} _ {e} \| _ {2} ^ {2} + \| \mathrm{sg} (\pmb {z} _ {e}) - \pmb {z} _ {q} \| _ {2} ^ {2} + \gamma \mathcal {L} _ {\mathcal {W}}.\tag{16}
$$

where sg denotes the stop-gradient. $\beta$ and $\gamma$ are hyper-parameters. We set $\gamma = 0.5$ for all VQ-VAE experiments.

By incorporating $L_{W}$ , the codebook is encouraged to globally match the first- and second-order statistics of the feature distribution, complementing the local assignment enforced by the standard VQ loss.

Remark Sections 2.3 and 2.4 provide empirical and theoretical evidence that minimizing quantization error is closely linked to aligning the codebook with the feature distribution. By encouraging the codebook to reflect the feature space structure and density, the alignment loss $L_{W}$ positions code vectors as cluster centers, minimizing the average squared distance between feature vectors and their assigned codes, $\|z_{e}-z_{q}\|_{2}^{2}$ . In essence, distribution alignment arranges prototypes to cover the feature manifold and fill gaps, reducing overall quantization error.

## 3.4.2 VQGAN Integration

To ensure high perceptual quality in the reconstructed images, we further investigate Wasserstein VQ within the VQGAN framework $[10]$ . VQGAN extends the VQ-VAE framework by integrating a VGG network $[43]$ and a patch-based discriminator $[10, 22]$ . The overall training objective of VQGAN can be written as follows:

$$
\mathcal {L} _ {\mathrm{VQGAN}} = \mathcal {L} _ {\mathrm{VQ-VAE}} + \mathcal {L} _ {\mathrm{Per}} + \lambda \mathcal {L} _ {\mathrm{GAN}}.\tag{17}
$$

Where $L_{Per}$ and $L_{GAN}$ denote the VGG-based perceptual loss [57] and the GAN loss [21, 30], respectively. Achieving state-of-the-art reconstruction fidelity via adversarial training typically incurs substantial computational cost. To reduce training overhead, we adopt the VQ-Transplant framework, which attains competitive reconstruction fidelity at a fraction of the cost [12]. We adopt the VQ-Transplant framework solely as a training-efficient backbone; the proposed Wasserstein VQ is orthogonal to this choice and can be integrated into standard VQGAN training as well. Specifically, it initializes the encoder and decoder from a state-of-the-art pretrained tokenizer (e.g., VAR [47]) rather than training from scratch, significantly lowering training cost while preserving reconstruction quality.

![](images/55bd6951671830febbe6cd46df8de6cca02090e73eaf31d88e255a6087e6dad2.jpg)  
(a) $\mathcal{E}$ w.r.t. $\mu$

![](images/98676789e71d4df7fcdc8e89a6c1e571934af271ae131f84f55c56ed3af78176.jpg)  
(b) $\mathcal{U}$ w.r.t. $\mu$

![](images/aaa9afed330f21b5050496ab6fd4f45c383642a1ef91609cf82366b0054f9f16.jpg)  
(c) C w.r.t. μ

![](images/d5bf8d90413fdfb56e3ee5e3604380f600263e72fbb2bf7f07e457e9860459b3.jpg)  
(d) $\mathcal{L}_W$ w.r.t. $\mu$

![](images/96e3bc23b0bb164e41c91311d721f6e12b93788bfa12965cdd337e69e485bd56.jpg)  
(e) E w.r.t. ν

![](images/b16be82abf0dca8b5e85c9f371dc6438bdbcb800e5d11cc2e99f0f9d8af467d8.jpg)  
(f) $\mathcal{U}$ w.r.t. $\nu$

![](images/ffba6f04930be3501293fcc05f5b2ceb295daa700f347b6ab1618ba494cbad0b.jpg)  
(g) C w.r.t. ν

![](images/680b531313375fdd94df1bd9c0745bff32ef131cf1dcd6ad822007e4e5e98eb6.jpg)  
(h) $\mathcal{L}_W$ w.r.t. $\nu$  
Figure 5: The performance metrics $(\mathcal{E},\mathcal{U},\mathcal{C})$ for various VQ approaches. For panels (a) to (d), the codebook distribution is initialized as a Gaussian distribution, while for panels (e) to (h), the codebook distribution is initialized as a uniform distribution.

## 4 An Atomic Setting for Evaluating the Criterion Triple

As discussed in Section 2.2, the variance of the latent distribution has a substantial impact on $\mathcal{E}$ . Therefore, a fair evaluation of the criterion triple should be conducted under identical latent distributions. However, in practical scenarios, either in VQ-VAE [49] or VQGAN [10], the encoder outputs exhibit significantly different distributions across quantization methods due to intrinsic differences in their algorithmic mechanisms. Moreover, these distributional discrepancies are not static but evolve dynamically throughout VQ training. As a result, directly comparing quantization errors fails to accurately reflect the intrinsic effectiveness of different VQ algorithms and may even lead to misleading conclusions. To address this, we introduce a controlled simulated experimental setting that enables a principled evaluation of the intrinsic behavior of VQ algorithms.

Specifically, we assume that the encoder output feature distributions of all VQ methods follow the same Gaussian distribution, i.e., $z_{i} \sim \mathcal{N}_{d}(\mu \cdot \mathbf{1}, I_{d})$ . Although real-world feature distributions are often more complex, this simplification isolates the effect of the quantization mechanism itself and allows for controlled and comparable comparisons across methods. For the codebook distribution, we initialize all VQ methods with the same standard Gaussian distribution (by sampling $e_{k} \sim \mathcal{N}_{d}(\mathbf{0}, I_{d})$ ), ensuring that the distribution variance is identical across all methods.

Our baselines include Vanilla VQ $[49]$ , EMA VQ $[39]$ , Online VQ $[59]$ , and Linear VQ, which employs a linear projection layer with frozen code vectors $[60, 61]$ . For all VQ algorithms except Linear VQ, the sampled code vectors are treated as trainable parameters and optimized according to their respective update rules. In the case of Linear VQ, we focus on training only the linear projection layer. Detailed experimental specifications are provided in Appendix J.4.

As visualized in Figures 5a to 5c, Wasserstein VQ outperforms all baselines in terms of the criterion triplet $(\mathcal{E},\mathcal{U},\mathcal{C})$ , particularly when the feature distribution and the initialized codebook distribution exhibit large deviations. While existing VQ methods perform well under the idealized setting of $\mu=0$ , this scenario is unrealistic: in practice, feature distributions are diverse and continuously evolving, making it infeasible to perfectly match the codebook distribution by codebook initialization. When a large initial distribution gap exists (e.g., $\mu=5$ ), existing methods fail to achieve effective distribution alignment and perform poorly, as depicted in Figure 5d, highlighting their strong dependence on codebook initialization. In contrast, Wasserstein VQ overcomes this limitation through explicit distributional matching regularization, thereby achieving superior performance across all criterion metrics.

We observe the same behavior when the feature and codebook distributions are uniform, with feature vectors sampled from $\operatorname{Unif}_{d}(\nu - 1, \nu + 1)$ and code vectors initialized from $\operatorname{Unif}_{d}(-1, 1)$ . As shown in Figures 5e to 5h, Wasserstein VQ again performs the best, suggesting that its effectiveness extends beyond Gaussian assumptions and exhibits distribution-agnostic behavior.

Table 1: Comparison of VQ-VAEs trained on FFHQ dataset following [49].

<table><tr><td>Approaches</td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>Rec. Loss(↓)</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>16384</td><td>3.8%</td><td>527.2</td><td>27.83</td><td>73.8</td><td>0.0119</td></tr><tr><td>STE++</td><td>256</td><td>16384</td><td>3.4%</td><td>476.7</td><td>27.54</td><td>72.3</td><td>0.0129</td></tr><tr><td>EMA VQ</td><td>256</td><td>16384</td><td>14.0%</td><td>1795.7</td><td>28.39</td><td>74.8</td><td>0.0106</td></tr><tr><td>Online VQ</td><td>256</td><td>16384</td><td>11.7%</td><td>1115.3</td><td>27.68</td><td>72.6</td><td>0.0125</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>16384</td><td>100%</td><td>15713.3</td><td>29.03</td><td>76.6</td><td>0.0093</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>50000</td><td>1.2%</td><td>516.8</td><td>27.83</td><td>73.6</td><td>0.0120</td></tr><tr><td>STE++</td><td>256</td><td>50000</td><td>1.0%</td><td>447.2</td><td>27.49</td><td>72.4</td><td>0.0131</td></tr><tr><td>EMA VQ</td><td>256</td><td>50000</td><td>10.3%</td><td>4075.7</td><td>28.61</td><td>75.3</td><td>0.0101</td></tr><tr><td>Online VQ</td><td>256</td><td>50000</td><td>6.0%</td><td>1642.9</td><td>28.37</td><td>74.6</td><td>0.0107</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>50000</td><td>100%</td><td>47496.4</td><td>29.24</td><td>77.0</td><td>0.0089</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>100000</td><td>0.6%</td><td>481.0</td><td>27.86</td><td>74.2</td><td>0.0118</td></tr><tr><td>STE++</td><td>256</td><td>100000</td><td>0.5%</td><td>450.7</td><td>27.52</td><td>72.4</td><td>0.0130</td></tr><tr><td>EMA VQ</td><td>256</td><td>100000</td><td>2.7%</td><td>2087.5</td><td>28.43</td><td>74.8</td><td>0.0105</td></tr><tr><td>Online VQ</td><td>256</td><td>100000</td><td>3.6%</td><td>1556.8</td><td>27.12</td><td>71.1</td><td>0.0142</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>100000</td><td>100%</td><td>93152.7</td><td>29.53</td><td>78.0</td><td>0.0083</td></tr></table>

Table 2: Comparison of VQ-VAEs trained on ImageNet dataset following [49].

<table><tr><td>Methods</td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>Rec. Loss (↓)</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>16,384</td><td>2.5%</td><td>360.7</td><td>24.44</td><td>57.5</td><td>0.0294</td></tr><tr><td>STE++</td><td>256</td><td>16,384</td><td>6.5%</td><td>889.9</td><td>24.88</td><td>58.9</td><td>0.0270</td></tr><tr><td>EMA VQ</td><td>256</td><td>16,384</td><td>14.5%</td><td>1861.5</td><td>24.98</td><td>59.2</td><td>0.0267</td></tr><tr><td>Online VQ</td><td>256</td><td>16,384</td><td>22.2%</td><td>1465.6</td><td>24.88</td><td>58.6</td><td>0.0273</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>16,384</td><td>100.0%</td><td>15539.1</td><td>25.47</td><td>61.2</td><td>0.0242</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>50,000</td><td>0.9%</td><td>378.7</td><td>24.40</td><td>57.7</td><td>0.0295</td></tr><tr><td>STE++</td><td>256</td><td>50,000</td><td>2.0%</td><td>851.7</td><td>24.89</td><td>59.0</td><td>0.0270</td></tr><tr><td>EMA VQ</td><td>256</td><td>50,000</td><td>16.8%</td><td>6139.3</td><td>25.37</td><td>60.9</td><td>0.0246</td></tr><tr><td>Online VQ</td><td>256</td><td>50,000</td><td>9.9%</td><td>2241.7</td><td>25.09</td><td>59.7</td><td>0.0260</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>50,000</td><td>100.0%</td><td>46133.2</td><td>25.72</td><td>62.3</td><td>0.0230</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>100,000</td><td>0.4%</td><td>337.0</td><td>24.43</td><td>57.4</td><td>0.0295</td></tr><tr><td>STE++</td><td>256</td><td>100,000</td><td>0.9%</td><td>730.6</td><td>24.86</td><td>59.1</td><td>0.0269</td></tr><tr><td>EMA VQ</td><td>256</td><td>100,000</td><td>3.0%</td><td>2170.0</td><td>25.13</td><td>60.1</td><td>0.0257</td></tr><tr><td>Online VQ</td><td>256</td><td>100,000</td><td>4.1%</td><td>1709.9</td><td>24.95</td><td>59.1</td><td>0.0267</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>100,000</td><td>100.0%</td><td>93264.7</td><td>25.88</td><td>63.0</td><td>0.0223</td></tr></table>

## 5 Experiments

In this section, we empirically demonstrate the effectiveness of the proposed distributional matching framework in visual tokenization tasks. Experiments are conducted within the VQ-VAE $[49]$ and VQGAN $[10]$ frameworks.

## 5.1 Evaluation on VQ-VAE Framework

Datasets and Baselines Experiments are conducted on four benchmark datasets: two low-resolution datasets, i.e., CIFAR-10 $[27]$ and SVHN $[34]$ , and two high-resolution datasets FFHQ $[23]$ and ImageNet $[8]$ . We evaluate our approach against representative VQ methods: Vanilla VQ $[49]$ , EMA VQ $[39]$ , which uses exponential moving average updates and is also referred to as k-means, Online VQ, which employs k-means++ in CVQ-VAE $[59]$ , and enhanced straight-through estimator STE++ $[20]$ . For experimental settings, see Appendix J.5.

Metrics We employ multiple evaluation metrics, including the codebook utilization rate (U), codebook perplexity (C), peak signal-to-noise ratio (PSNR), patch-level structural similarity index (SSIM), and pixel-level reconstruction loss (Rec. Loss). We exclude the raw quantization error (E) from the main VQ-VAE experiments, as E is highly sensitive to feature variance, which is not controlled under different training dynamics. Consequently, a direct comparison of E does not faithfully reflect the intrinsic effectiveness of VQ algorithms, as discussed in Section 2.2 and Section 4.

Main Results As shown in Tables 1, 2 in the main text, and Tables 11, 12 in Appendix K.1, our proposed Wasserstein VQ consistently outperforms all baselines across datasets, achieving superior performance on nearly all metrics under a wide range of settings. The improvements are particularly pronounced in codebook utilization (U) and codebook perplexity (C), where Wasserstein VQ attains near-complete codebook utilization and substantially higher codebook perplexity. This indicates that the learned codebooks more faithfully reflect the structure and density of the feature space. As vector quantization fundamentally acts as a compression mechanism from continuous latent representations to a discrete codebook, improved alignment between feature and code distributions directly translates into reduced information loss. Consistent with this interpretation, Wasserstein VQ achieves the lowest reconstruction loss and improved PSNR/SSIM across datasets.

Table 3: Analysis of generalization ability from ImageNet to FFHQ dataset.

<table><tr><td>Approaches</td><td>Tokens</td><td>Codebook Dim</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>Rec. Loss (↓)</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>16384</td><td>1.7%</td><td>332.9</td><td>27.18</td><td>71.6</td><td>0.0138</td></tr><tr><td>EMA VQ</td><td>256</td><td>16384</td><td>12.5%</td><td>1292.1</td><td>27.69</td><td>72.3</td><td>0.0126</td></tr><tr><td>Online VQ</td><td>256</td><td>16384</td><td>33.2%</td><td>2794.9</td><td>27.91</td><td>72.7</td><td>0.0120</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>16384</td><td>99.2%</td><td>13975.4</td><td>28.62</td><td>75.0</td><td>0.0103</td></tr></table>

Table 4: The computational overhead of various VQ approaches.

<table><tr><td>Approaches</td><td>Codebook Size</td><td>Times (second)</td><td>Codebook Size</td><td>Times (second)</td><td>Codebook Size</td><td>Times (second)</td></tr><tr><td>Vanilla VQ</td><td>16384</td><td>0.184</td><td>50000</td><td>0.408</td><td>100000</td><td>0.655</td></tr><tr><td>EMA VQ</td><td>16384</td><td>0.265</td><td>50000</td><td>0.807</td><td>100000</td><td>1.502</td></tr><tr><td>Online VQ</td><td>16384</td><td>1.810</td><td>50000</td><td>5.438</td><td>100000</td><td>10.64</td></tr><tr><td>Wasserstein VQ</td><td>16384</td><td>0.294</td><td>50000</td><td>0.525</td><td>100000</td><td>0.774</td></tr></table>

Gaussian Approximation Validation To empirically assess the Gaussian approximation underlying our Wasserstein objective, we analyze the distributional properties of the learned latent features. Using the FFHQ dataset, we extract latent vectors from a subset of 1,600 images (d = 8), resulting in a total of N = 409,600 samples. Figure 6 presents Quantile–Quantile (Q–Q) plots for the feature dimensions exhibiting the highest $(z_{6})$ and lowest $(z_{7})$ agreement with a Gaussian reference. Under an ideal Gaussian model, the samples (blue points) align with the theoretical reference line $(y = x)$ . As shown, the best-fitting dimension $(z_{6})$ demonstrates near-perfect alignment, with a coefficient of determination of $R^{2} = 0.9988$ . Importantly, even for the least-fitting dimension $(z_{7})$ , the Q–Q plot evaluated over the distribution achieves a high goodness-of-fit score $(R^{2} > 0.979)$ . While mild deviations appear in the extreme tails, the

![](images/7bc24ce776f5ef0606d55d04aaa431a3d0b42930807017dfdcb9a8b7f2701f83.jpg)

![](images/8165b5731d9df1ed0745ef9a4006eacdb5282d14e7a4aa044864c69248b05dc2.jpg)  
Figure 6: Gaussianity Validation via Q-Q Plots. We visualize the feature dimensions with the highest $z_{6}$ (Left) and lowest $(z_{7}, \text{Right})$ conformance to the Gaussian hypothesis. Note: For visual clarity, plots display a random subset of 5,000 points, while the reported statistics $(R^{2}, W)$ are computed on the full validation set $(N = 409, 600)$ . The strong alignment with the red identity line $(y = x)$ confirms that the learned features are predominantly Gaussian, with deviations confined to sparse heavy tails.

central mass of the distribution, which dominates the probability mass and governs the Wasserstein objective, is well approximated by a Gaussian. These results support the use of a Gaussian-based Wasserstein formulation as an effective approximation for distributional matching. A more comprehensive univariate and multivariate analysis, including Mahalanobis distance based normality assessment across all dimensions, is provided in Appendix I.

Representation Visualization To examine the learned distribution of feature and code vectors across different VQ methods trained on the FFHQ dataset (with a fixed codebook size of K = 8192), we randomly sample 3,000 feature vectors and 1,000 code vectors and visualize them via scatter plots. As shown in Figures 7a and 7b, in both Vanilla VQ and EMA VQ, most code vectors are concentrated near the origin, rendering them largely inactive. While Online VQ mitigates this central clustering, its code vectors are pushed toward the extremes of the feature space, as illustrated in Figure 7c. This distributional mismatch leads to increased information loss and reduced codebook utilization. By contrast, Wasserstein VQ achieves substantially better alignment between feature and code vector distributions, significantly reducing information loss and improving codebook utilization.

Cross-Dataset Generalization To evaluate the model's generalization capability, we conducted a cross-dataset transfer experiment. Specifically, we used the pre-trained checkpoint from Table 2 (trained on ImageNet) and evaluated its performance on the FFHQ dataset, which is out-of-distribution relative to ImageNet. The results are summarized in Table 3.

![](images/5f5c717653b23880ad630d2e85efd1a2b38d9296b012927d2528ff0f2c4e69a9.jpg)  
(a) Vanilla VQ

![](images/b5cfaf2678ab2773c7139e58ab5cc3fe93f09ece07f484ef62c3299b90c5a226.jpg)  
(b) EMA VQ

![](images/a93d88c832929ec257cdf4d4e823150a60cd8908821c81d957f52c4609e63e7a.jpg)  
(c) Online VQ

![](images/c0c23ab3c9abe6717bc2d600177cc948dc2ed85a5ae536da496993e7538e8401.jpg)  
(d) Wasserstein VQ  
Figure 7: Visualization of feature and codebook distributions. Blue · and red × represent the feature and code vectors, respectively.

Among all methods, Wasserstein VQ achieves the highest performance in this transfer setting, indicating that the proposed distribution alignment objective does not impair generalization. Moreover, Wasserstein VQ's strong performance on the out-of-distribution FFHQ dataset suggests that enforcing alignment may contribute to improved robustness under distribution shifts. Overall, these findings provide evidence that models trained with the proposed regularization generalize effectively to out-of-distribution data.

Computational Overhead Analyses We evaluate the runtime efficiency by measuring the forward and backward pass times of the VQ module. As shown in Table 4 in Appendix K.4, the runtime of Wasserstein VQ is comparable to Vanilla VQ, with only negligible overhead arising from the covariance computation. This confirms that the closed-form Wasserstein objective offers a highly efficient pathway to optimal codebook utilization without incurring a significant time overhead. More implementation details see Appendix K.4.

Sensitivity Analysis on $\gamma$ We conduct a sensitivity analysis with respect to $\gamma$ on the FFHQ dataset by varying $\gamma \in \{0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1\}$ . As reported in Table 15 in Appendix K.3, when $\gamma = 0$ , Wasserstein VQ yields the worst performance, as it degenerates into the vanilla VQ formulation without distributional matching. As $\gamma$ increases from $10^{-5}$ to $10^{-3}$ , the performance of Wasserstein VQ consistently improves. When $\gamma$ reaches $10^{-2}$ , Wasserstein VQ achieves full (100%) codebook utilization along with competitive quantitative results. Moreover, within the range $\gamma \in [10^{-2}, 1]$ , the performance of Wasserstein VQ remains stable, indicating that the method is not sensitive to the precise choice of $\gamma$ once it exceeds a moderate threshold.

Effect of Codebook Size on FFHQ To investigate the impact of codebook size on VQ performance, we vary the codebook size K across a wide range: $K \in \{1024, 2048, 4096, 8192, 16384, 50000, 100000\}$ . As shown in Table 1 in the main text and Table 13 in Appendix K.2, the vanilla VQ model suffers from severe codebook collapse even at relatively small sizes, such as K = 1024. In contrast, improved variants like EMA VQ and Online VQ handle smaller codebooks effectively, but still exhibit codebook collapse when K becomes very large (e.g., $K \geq 50000$ ). In contrast, Wasserstein VQ consistently maintains 100% codebook utilization across all codebook sizes, highlighting the effectiveness of enforcing distributional alignment via the quadratic Wasserstein distance in mitigating codebook collapse.

Effect of Codebook Dimensionality We investigate the impact of codebook dimensionality on VQ performance by conducting experiments on CIFAR-10 with d ranging from 2 to 32. As reported in Table 14 (Appendix K.2), our proposed Wasserstein VQ consistently outperforms all baselines across all dimensionalities. We also observe a manifestation of the curse of dimensionality: performance generally degrades as d increases. Vanilla VQ experiences the most severe decline, followed by EMA VQ and Online VQ, whereas Wasserstein VQ exhibits only minimal reduction in codebook utilization.

## 5.2 Evaluation on the VQGAN Framework

Datasets and Baselines We evaluate our proposed method against a comprehensive set of baselines on the ImageNet and FFHQ datasets. For ImageNet, the compared methods include DQVAE $[18]$ , DF-VQGAN $[35]$ , DiVAE $[42]$ , RQVAE $[28]$ , VQGAN $[10]$ , VQGAN-FC $[53]$ , VQGAN-EMA $[39]$ , VQGAN-LC $[60]$ , Llama GEN $[45]$ , and VAR $[47]$ . For FFHQ, we compare against RQVAE $[28]$ ,

Table 5: Reconstruction performance on the ImageNet-1K dataset. The suffixes "-a," "-b," and "-c" correspond to decoder adaptation for 5, 10, and 15 epochs, respectively. The best-performing result is highlighted in bold. Results marked with $\dagger$ are cited from VQGAN-LC [60], those with $\star$ from Llama GEN [45], and those with $\ddagger$ from VQ-Transplant [12].

<table><tr><td>Methods</td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{U} (\uparrow)$ </td><td>r-FID (↓)</td><td>r-IS (↑)</td><td>LPIPS (↓)</td><td>PSNR (↑)</td><td>SSIM (↑)</td></tr><tr><td> $DQVAE^†$ </td><td>256</td><td>1,024</td><td>-</td><td>4.08</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $DiVAE^†$ </td><td>256</td><td>16,384</td><td>-</td><td>4.07</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $RQVAE^†$ </td><td>256</td><td>16,384</td><td>-</td><td>3.20</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $RQVAE^†$ </td><td>512</td><td>16,384</td><td>-</td><td>2.69</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $RQVAE^†$ </td><td>1,024</td><td>16,384</td><td>-</td><td>1.83</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $DF-VQGAN^†$ </td><td>256</td><td>12,288</td><td>-</td><td>5.16</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $DF-VQGAN^†$ </td><td>1,024</td><td>8,192</td><td>-</td><td>1.38</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $Llama GEN^*$ </td><td>256</td><td>16,384</td><td>97.0%</td><td>2.19</td><td>-</td><td>-</td><td>20.79</td><td>67.5</td></tr><tr><td rowspan="3"> $VQGAN^†$ </td><td>256</td><td>16,384</td><td>3.4%</td><td>5.96</td><td>-</td><td>0.17</td><td>23.3</td><td>52.4</td></tr><tr><td>256</td><td>50,000</td><td>1.1%</td><td>5.44</td><td>-</td><td>0.17</td><td>22.5</td><td>52.5</td></tr><tr><td>256</td><td>100,000</td><td>0.5%</td><td>5.44</td><td>-</td><td>0.17</td><td>22.3</td><td>52.5</td></tr><tr><td rowspan="3"> $VQGAN-FC^†$ </td><td>256</td><td>16,384</td><td>11.2%</td><td>4.29</td><td>-</td><td>0.17</td><td>22.8</td><td>54.5</td></tr><tr><td>256</td><td>50,000</td><td>3.6%</td><td>4.96</td><td>-</td><td>0.15</td><td>23.1</td><td>54.7</td></tr><tr><td>256</td><td>100,000</td><td>1.9%</td><td>4.65</td><td>-</td><td>0.15</td><td>22.9</td><td>55.1</td></tr><tr><td rowspan="3"> $VQGAN-EMA^†$ </td><td>256</td><td>16,384</td><td>83.2%</td><td>3.41</td><td>-</td><td>0.14</td><td>23.5</td><td>56.6</td></tr><tr><td>256</td><td>50,000</td><td>40.2%</td><td>3.88</td><td>-</td><td>0.14</td><td>23.2</td><td>55.9</td></tr><tr><td>256</td><td>100,000</td><td>24.2%</td><td>3.46</td><td>-</td><td>0.13</td><td>23.4</td><td>56.2</td></tr><tr><td rowspan="4"> $VQGAN-LC^†$ </td><td>256</td><td>16,384</td><td>99.9%</td><td>3.01</td><td>-</td><td>0.13</td><td>23.2</td><td>56.4</td></tr><tr><td>256</td><td>50,000</td><td>99.9%</td><td>2.75</td><td>-</td><td>0.13</td><td>23.8</td><td>58.4</td></tr><tr><td>256</td><td>100,000</td><td>99.9%</td><td>2.62</td><td>-</td><td>0.12</td><td>23.8</td><td>58.9</td></tr><tr><td>1,024</td><td>100,000</td><td>99.5%</td><td>1.29</td><td>-</td><td>0.07</td><td>27.0</td><td>71.6</td></tr><tr><td rowspan="3">Wasserstein  $VQ-a^‡$ </td><td>512</td><td>16,384</td><td>99.8%</td><td>1.04</td><td>191.3</td><td>0.114</td><td>24.36</td><td>64.0</td></tr><tr><td>512</td><td>32,768</td><td>99.7%</td><td>0.98</td><td>193.9</td><td>0.111</td><td>24.37</td><td>64.3</td></tr><tr><td>512</td><td>65,536</td><td>99.6%</td><td>0.92</td><td>195.5</td><td>0.106</td><td>24.68</td><td>65.4</td></tr><tr><td rowspan="3">Wasserstein VQ-b</td><td>512</td><td>16,384</td><td>99.8%</td><td>0.98</td><td>192.9</td><td>0.114</td><td>24.34</td><td>63.7</td></tr><tr><td>512</td><td>32,768</td><td>99.8%</td><td>0.88</td><td>196.2</td><td>0.109</td><td>24.60</td><td>64.7</td></tr><tr><td>512</td><td>65,536</td><td>99.6%</td><td>0.81</td><td>198.7</td><td>0.105</td><td>24.77</td><td>65.5</td></tr><tr><td rowspan="3">Wasserstein VQ-c</td><td>512</td><td>16,384</td><td>99.8%</td><td>0.90</td><td>194.1</td><td>0.114</td><td>24.27</td><td>63.4</td></tr><tr><td>512</td><td>32,768</td><td>99.8%</td><td>0.85</td><td>196.4</td><td>0.109</td><td>24.43</td><td>64.1</td></tr><tr><td>512</td><td>65,536</td><td>99.6%</td><td>0.79</td><td>198.5</td><td>0.104</td><td>24.73</td><td>65.2</td></tr></table>

Table 6: Reconstruction performance on the ImageNet-1K dataset for multi-scale quantization algorithms. The suffixes "-a," "-b," and "-c" correspond to decoder adaptation for 5, 10, and 15 epochs, respectively. For each codebook size, the best-performing result is highlighted in bold. $^{\ddagger}$ : results are cited from VQ-Transplant [12].

<table><tr><td>Methods</td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{E} (\downarrow)$ </td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>r-FID (↓)</td></tr><tr><td> $VAR^{\ddagger}$ </td><td>680</td><td>4,096</td><td>0.283</td><td>100%</td><td>2,981.4</td><td>0.92</td></tr><tr><td rowspan="2">Vanilla  $VAR-a^{\ddagger}$ </td><td>680</td><td>4,096</td><td>0.305</td><td>38.2%</td><td>1,300.4</td><td>1.25</td></tr><tr><td>680</td><td>8,192</td><td>0.309</td><td>22.9%</td><td>1,422.6</td><td>1.30</td></tr><tr><td rowspan="2">EMA  $VAR-a^{\ddagger}$ </td><td>680</td><td>4,096</td><td>0.321</td><td>99.9%</td><td>1,806.6</td><td>1.69</td></tr><tr><td>680</td><td>8,192</td><td>0.312</td><td>99.8%</td><td>2,498.8</td><td>1.15</td></tr><tr><td rowspan="2">Online  $VAR-a^{\ddagger}$ </td><td>680</td><td>4,096</td><td>0.276</td><td>99.0%</td><td>1,950.9</td><td>1.05</td></tr><tr><td>680</td><td>8,192</td><td>0.269</td><td>73.9%</td><td>3,588.6</td><td>1.00</td></tr><tr><td rowspan="2">Wasserstein  $VAR-a^{\ddagger}$ </td><td>680</td><td>4,096</td><td>0.255</td><td>100%</td><td>3,286.2</td><td>0.93</td></tr><tr><td>680</td><td>8,192</td><td>0.240</td><td>100%</td><td>6,518.2</td><td>0.83</td></tr><tr><td rowspan="2">Wasserstein VAR-b</td><td>680</td><td>4,096</td><td>0.255</td><td>100%</td><td>3,286.2</td><td>0.88</td></tr><tr><td>680</td><td>8,192</td><td>0.240</td><td>100%</td><td>6,518.2</td><td>0.79</td></tr><tr><td rowspan="2">Wasserstein VAR-c</td><td>680</td><td>4,096</td><td>0.255</td><td>100%</td><td>3,286.2</td><td>0.81</td></tr><tr><td>680</td><td>8,192</td><td>0.240</td><td>100%</td><td>6,518.2</td><td>0.73</td></tr></table>

VQGAN [10], VQGAN-FC [53], VQGAN-EMA [39], VQ-WAE [50], MQVAE [19], and VQGAN-LC [60]. Implementation details see Appendix J.5.

Table 7: Reconstruction performance on the FFHQ dataset. Results marked with $\dagger$ are cited from VQGAN-LC [60], and those with $\ddagger$ from VQ-Transplant [12].

<table><tr><td>VQs</td><td>Tokens</td><td>Codebook Size K</td><td> $\mathcal{E}(\downarrow)$ </td><td> $\mathcal{U} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td></tr><tr><td>RQVAE $^{\dagger}$ </td><td>256</td><td>2048</td><td>-</td><td>-</td><td>22.9</td><td>67.0</td><td>0.13</td><td>7.04</td></tr><tr><td>VQ-WAE $^{\dagger}$ </td><td>256</td><td>1024</td><td>-</td><td>-</td><td>22.5</td><td>66.5</td><td>0.12</td><td>4.20</td></tr><tr><td>MQVAE $^{\dagger}$ </td><td>256</td><td>1024</td><td>-</td><td>78.2%</td><td>-</td><td>-</td><td>-</td><td>4.55</td></tr><tr><td>VQGAN $^{\dagger}$ </td><td>256</td><td>16384</td><td>-</td><td>2.3%</td><td>24.4</td><td>63.3</td><td>0.12</td><td>5.25</td></tr><tr><td>VQGAN-FC $^{\dagger}$ </td><td>256</td><td>16384</td><td>-</td><td>10.9%</td><td>24.8</td><td>64.6</td><td>0.11</td><td>4.86</td></tr><tr><td>VQGAN-EMA $^{\dagger}$ </td><td>256</td><td>16384</td><td>-</td><td>68.2%</td><td>25.4</td><td>66.1</td><td>0.10</td><td>4.79</td></tr><tr><td>VQGAN-LC $^{\dagger}$ </td><td>256</td><td>100000</td><td>-</td><td>99.5%</td><td>26.1</td><td>69.4</td><td>0.08</td><td>3.81</td></tr><tr><td>Wasserstein VQ $^{\ddagger}$ </td><td>512</td><td>16384</td><td>0.153</td><td>99.7%</td><td>27.25</td><td>75.4</td><td>0.075</td><td>1.81</td></tr><tr><td>Wasserstein VQ $^{\ddagger}$ </td><td>512</td><td>32768</td><td>0.142</td><td>99.7%</td><td>27.33</td><td>75.7</td><td>0.072</td><td>1.21</td></tr></table>

![](images/cef82439c6f10f9e09ede3e933e895534081c3fc72233a69416bddb0e6fd9e0b.jpg)  
Figure 8: Visualization of reconstructed ImageNet Images. The top row displays the original input images with a resolution of $256 \times 256$ pixels, while the bottom row shows the reconstructed images from the Wasserstein VAR.

![](images/24be0921e6ea4e4ba2faeb37329158a65f0e549928d4586465e5fa5ece46da22.jpg)  
Figure 9: Visualization of reconstructed FFHQ Images. The top row displays the original input images with a resolution of $256 \times 256$ pixels, while the bottom row shows the reconstructed images from the Wasserstein VQ.

Metrics We report standard reconstruction and perceptual quality metrics, including Fréchet Inception Distance (r-FID) [16], Reconstruction Inception Score (r-IS) [41], Learned Perceptual Image Patch Similarity (LPIPS) [57], Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index Measure (SSIM).

Main Results As shown in Tables 5 and 7, our proposed Wasserstein VQ achieves the lowest r-FID within the VQGAN framework. We further evaluate multi-scale quantization algorithms in Table 6. All experiments in this table are conducted under fully controlled conditions: the encoder–decoder architecture is fixed to VAR [47], and, importantly, the latent features are kept identical across all methods. This ensures that measurements of codebook utilization, perplexity, and quantization error are not influenced by variations in the latent features, as discussed in Section 2.2.

Under these controlled conditions, Wasserstein VAR consistently demonstrates the strongest overall performance, with particularly notable improvements in quantization error. Reconstruction quality further improves as the number of decoder adaptation epochs increases, reflecting stronger adversarial refinement. For instance, when K = 8192, the r-FID decreases from 0.83 (Wasserstein VAR-a) to 0.73 (Wasserstein VAR-c).

Reconstruction Visualization To further demonstrate the reconstruction performance, we visualize the reconstructed FFHQ and ImageNet images in Figure 9 and 8. As shown, the images reconstructed by our method are nearly identical to the original inputs. In particular, on the FFHQ dataset, only very minor differences can be observed in fine details.

Table 8: Reconstruction performance on the ImageNet-1K dataset for multi-scale quantization algorithms. The suffixes “-a,” “-b,” and “-c” correspond to decoder adaptation for 5, 10, and 15 epochs, respectively. For each codebook size, the best-performing result is highlighted in bold.

<table><tr><td>Methods</td><td>Tokens</td><td>Codebook Size K</td><td> $\mathcal{E} (\downarrow)$ </td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>r-FID (↓)</td></tr><tr><td rowspan="2">MMD VAR-a</td><td>680</td><td>4096</td><td>0.255</td><td>100%</td><td>3757.3</td><td>0.91</td></tr><tr><td>680</td><td>8192</td><td>0.234</td><td>100%</td><td>7539.4</td><td>0.81</td></tr><tr><td rowspan="2">Wasserstein VAR-a</td><td>680</td><td>4096</td><td>0.255</td><td>100%</td><td>3286.2</td><td>0.93</td></tr><tr><td>680</td><td>8192</td><td>0.240</td><td>100%</td><td>6518.2</td><td>0.83</td></tr><tr><td rowspan="2">MMD VAR-b</td><td>680</td><td>4096</td><td>0.255</td><td>100%</td><td>3757.3</td><td>0.87</td></tr><tr><td>680</td><td>8192</td><td>0.234</td><td>100%</td><td>7539.4</td><td>0.78</td></tr><tr><td rowspan="2">Wasserstein VAR-b</td><td>680</td><td>4096</td><td>0.255</td><td>100%</td><td>3286.2</td><td>0.88</td></tr><tr><td>680</td><td>8192</td><td>0.240</td><td>100%</td><td>6518.2</td><td>0.79</td></tr><tr><td rowspan="2">MMD VAR-c</td><td>680</td><td>4096</td><td>0.255</td><td>100%</td><td>3757.3</td><td>0.82</td></tr><tr><td>680</td><td>8192</td><td>0.234</td><td>100%</td><td>7539.4</td><td>0.75</td></tr><tr><td rowspan="2">Wasserstein VAR-c</td><td>680</td><td>4096</td><td>0.255</td><td>100%</td><td>3286.2</td><td>0.81</td></tr><tr><td>680</td><td>8192</td><td>0.240</td><td>100%</td><td>6518.2</td><td>0.73</td></tr></table>

![](images/3a449e7c5d00ff9546d46ed3cac22e2990e4667de3e08b0a344833250c9bfdb8.jpg)  
(a) Codebook Size

![](images/94dc3851ad59e6c1c2790bea9b5516af68889509414ffe7edf6c3767cfd1adba.jpg)  
(b) Data Sample Size  
Figure 10: Computational overhead (seconds) comparison between Wasserstein VQ and MMD VQ.

## 6 Discussion on Two Distributional Matching Approaches: Wasserstein VQ vs. MMD VQ

We compare Wasserstein VQ and MMD VQ from three perspectives: visual tokenization performance, computational efficiency, and robustness to non-Gaussian distributions. First, we assess visual tokenization performance under the VQGAN framework. As shown in Table 8, we evaluate performance with decoder adaptation conducted for 5, 10, and 15 epochs. Overall, the two methods achieve very similar performance, with Wasserstein VQ slightly outperforming MMD VQ when the decoder adaptation is extended to 15 epochs.

Second, we analyze the computational cost of the two approaches. Following the setup in Appendix K.4, we measure runtime efficiency by recording forward and backward pass times over 100 iterations. We consider two scenarios: (i) fixing the number of data samples at N = 8192 while varying the codebook size K from 128 to 32,768, and (ii) fixing the codebook size at K = 8192 while varying the number of data samples N from 128 to 32,768, providing complementary comparisons. As illustrated in Figures 10a and 10b, MMD VQ exhibits a rapidly increasing computational cost, reflecting its inefficiency. This difference arises because Wasserstein VQ performs distribution matching by aligning first- and second-order statistics, whereas MMD VQ relies on pairwise distances between all elements. Consequently, the computational complexity of MMD VQ grows substantially, especially when both the number of data samples and the codebook size are large.

Third, we evaluate the robustness of Wasserstein VQ and MMD VQ under non-Gaussian latent distributions. While Section 5.1 shows that the latent distributions in visual tokenization tasks are approximately Gaussian, this experiment allows us to explicitly assess robustness beyond the

![](images/c0a4e48060f75ea465622256d7a226a84a66691f91bbc3cff5795277a43f49dd.jpg)  
(a) Quantization error

![](images/e76285d23bfdb6eeaa09e4ce45dbc4f5da634f85e61114086c4ab41327ccf54c.jpg)  
(b) Codebook utilization  
Figure 11: comparison between Wasserstein VQ and MMD VQ.

Gaussian regime. To simulate a non-Gaussian distribution, we follow the setting in Section 4 and assume the encoder output features follow a bimodal mixture:

$$
z _ {i} \sim 0. 5 \cdot \mathcal {N} (\zeta \mathbf {1}, I) + 0. 5 \cdot \mathcal {N} (- \zeta \mathbf {1}, I),
$$

where the mode separation parameter $\zeta\in\{0.0,0.5,\ldots,4.0\}$ controls deviation from Gaussianity, with larger $\zeta$ indicating stronger non-Gaussianity. The codebook is initialized with a standard Gaussian distribution, and code vectors are treated as trainable parameters. After 10,000 training steps, we evaluate Wasserstein VQ and MMD VQ in terms of quantization error and codebook utilization rate. As shown in Figures 11a and 11b, for small $\zeta$ (nearly Gaussian), the two methods perform comparably. As $\zeta$ increases, MMD VQ consistently achieves lower quantization error and higher codebook utilization, whereas Wasserstein VQ degrades due to its limited moment-matching assumption. These synthetic experiments empirically demonstrate that MMD VQ is more robust to non-Gaussian feature distributions.

Summary. Wasserstein VQ and MMD VQ exhibit complementary strengths and limitations. Wasserstein VQ is computationally efficient and performs stably under nearly Gaussian latent distributions, benefiting from its first- and second-order moment-matching formulation, which scales well with large datasets and codebooks. However, it is less robust when feature distributions deviate from Gaussianity. In contrast, MMD VQ excels under non-Gaussian distributions, consistently achieving lower quantization error and higher codebook utilization, but at the cost of substantially higher computational overhead. In practice, Wasserstein VQ is preferred for efficiency and standard Gaussian-like scenarios, while MMD VQ provides an advantage when robustness to non-Gaussian features is critical.

## 7 Conclusion

We identify a fundamental distributional mismatch between feature and code vectors as a common cause of training instability and codebook collapse in vector quantization. To address this issue, we propose a general distributional matching framework that formalizes desirable VQ behavior through principled criteria. Within this framework, we show that aligning feature and code distributions provides a unified mechanism for improving training stability, codebook utilization, and representation efficiency. We instantiate the framework using a Wasserstein-based objective with an efficient closed-form solution under a mild Gaussian approximation, and further demonstrate that a nonparametric alternative based on maximum mean discrepancy achieves comparable performance. More broadly, this work offers a unified perspective on vector quantization and opens new avenues for designing stable and efficient discrete representation learning methods.

## 8 Limitation and Discussion

One limitation of this work is that we do not directly validate the effectiveness of the proposed distributional matching framework through downstream generative results, due to limited computational resources. Recent studies have shown that improved reconstruction performance does not necessarily translate into better generative performance. For instance, increasing the codebook size or token length in discrete visual tokenizers $[54]$ , or increasing the latent resolution or dimensionality in continuous tokenizers, can significantly improve reconstruction quality while simultaneously degrading generative performance $[52]$ .

In these cases, larger codebooks or longer token sequences in discrete tokenizers substantially increase the complexity of training autoregressive models, due to longer sequences and larger vocabularies. Similarly, higher latent resolution or dimensionality in continuous tokenizers makes the training of diffusion models more challenging, leading to reduced generative quality.

In contrast, distributional matching improves tokenizer quality without introducing additional burdens to downstream generative models. For example, compared with VAR $[47]$ , Wasserstein VAR achieves improved reconstruction without increasing the codebook size or token length. As a result, the improved reconstruction performance induced by distributional matching is more likely to translate into improved generative performance.

## References

[1] David Arthur and Sergei Vassilvitskii. k-means++: the advantages of careful seeding. In ACM-SIAM Symposium on Discrete Algorithms, 2007.

[2] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. ArXiv, 2013.

[3] Yoshua Bengio, Nicholas Léonard, and Aaron C. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. ArXiv, 2013.

[4] A. Bhattacharyya. On a measure of divergence between two statistical populations defined by their probability distributions. Bulletin of the Calcutta Mathematical Society, 1943.

[5] Paul S. Bradley and Usama M. Fayyad. Refining initial points for k-means clustering. In ICML, 1998.

[6] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv'e J'egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

[7] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022.

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, K. Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

[9] Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. ArXiv, 2020.

[10] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, 2021.

[11] Xianghong Fang, Wenlong Mou, Yuan Yuan, Dehan Kong, and Tim G. J. Rudner. A unifying view of vector, product, and scalar quantization: An information-theoretic perspective. In ICLR submission, 2026.

[12] Xianghong Fang, Yuan Yuan, Dehan Kong, and Tim G. J. Rudner. VQ-transplant: Efficient plug-and-play vq-module integration for pre-trained visual tokenizers. In ICLR, 2026.

[13] Siegfried Graf and Harald Luschgy. Foundations of quantization for probability distributions. Springer Science & Business Media, 2000.

[14] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Scholkopf, and Alex Smola. A kernel two-sample test. JMLR, 2012.

[15] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022.

[16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017.

[17] Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

[18] Mengqi Huang, Zhendong Mao, Zhuowei Chen, and Yongdong Zhang. Towards accurate image coding: Improved autoregressive image generation with dynamic vector quantization. In CVPR, 2023.

[19] Mengqi Huang, Zhendong Mao, Quang Wang, and Yongdong Zhang. Not all image regions matter: Masked vector quantization for autoregressive image generation. In CVPR, 2023.

[20] Minyoung Huh, Brian Cheung, Pulkit Agrawal, and Phillip Isola. Straightening out the straight-through estimator: Overcoming optimization challenges in vector quantized networks. In ICML, 2023.

[21] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017.

[22] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016.

[23] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019.

[24] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019.

[25] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020.

[26] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.

[27] Alex Krizhevsky. Learning multiple layers of features from tiny images. ArXiv, 2009.

[28] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, 2022.

[29] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Jindong Wang, Zhe Lin, and Bhiksha Raj. Xq-gan: An open-source image tokenization framework for autoregressive generation. ArXiv, 2024.

[30] Jae Hyun Lim and J. C. Ye. Geometric gan. ArXiv, 2017.

[31] David Lindley and Solomon Kullback. Information theory and statistics. Journal of the American Statistical Association, 1959.

[32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

[33] Fabian Mentzer, David C. Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. In ICLR, 2024.

[34] Yuval Netzer, Tao Wang, Adam Coates, A. Bissacco, Bo Wu, and A. Ng. Reading digits in natural images with unsupervised feature learning. ArXiv, 2011.

[35] Minheng Ni, Chenfei Wu, Haoyang Huang, Daxin Jiang, Wangmeng Zuo, and Nan Duan. Nüwa-lip: Language-guided image inpainting with defect-free vqgan. In CVPR, 2022.

[36] Ingram Olkin and Friedrich Pukelsheim. The distance between two random vectors with given dispersion matrices. Linear Algebra and its Applications, 1982.

[37] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Q. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russ Howes, Po-Yao (Bernie) Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Huijiao Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. TMLR, 2024.

[38] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.

[39] Ali Razavi, Aäron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. In NeurIPS, 2019.

[40] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015.

[41] Tim Salimans, Ian J. Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, 2016.

[42] Jie Shi, Chenfei Wu, Jian Liang, Xiang Liu, and Nan Duan. Divae: Photorealistic images synthesis with denoising diffusion decoder. ArXiv, 2022.

[43] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[44] Bharath K. Sriperumbudur, Arthur Gretton, Kenji Fukumizu, Bernhard Scholkopf, and Gert R. G. Lanckriet. Hilbert space embeddings and metrics on probability measures. JMLR, 2009.

[45] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. ArXiv, 2024.

[46] Yuhta Takida, Takashi Shibuya, Wei-Hsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka, Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: Variational bayes on discrete representation with self-annealed stochastic quantization. In ICML, 2022.

[47] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024.

[48] Hung-Yu Tseng, Lu Jiang, Ce Liu, Ming-Hsuan Yang, and Weilong Yang. Regularizing generative adversarial networks under limited data. In CVPR, 2021.

[49] Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017.

[50] Tung-Long Vuong, Trung-Nghia Le, He Zhao, Chuanxia Zheng, Mehrtash Harandi, Jianfei Cai, and Dinh Q. Phung. Vector quantized wasserstein auto-encoder. In ICML, 2023.

[51] Will Williams, Sam Ringer, Tom Ash, John Hughes, David Macleod, and Jamie Dougherty. Hierarchical quantized autoencoders. In NeurIPS, 2020.

[52] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In CVPR, 2025.

[53] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. In ICLR, 2022.

[54] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In ICLR, 2024.

[55] Han Zhang, Zizhao Zhang, Augustus Odena, and Honglak Lee. Consistency regularization for generative adversarial networks. ArXiv, 2019.

[56] Jiahui Zhang, Fangneng Zhan, Christian Theobalt, and Shijian Lu. Regularized vector quantization for tokenized image synthesis. In CVPR, 2023.

[57] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

[58] Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Differentiable augmentation for data-efficient gan training. In NeurIPS, 2020.

[59] Chuanxia Zheng and Andrea Vedaldi. Online clustered codebook. In ICCV, 2023.

[60] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. ArXiv, 2024.

[61] Yongxin Zhu, Bocheng Li, Yifei Xin, and Linli Xu. Addressing representation collapse in vector quantized models with one linear layer. ArXiv, 2024.

## A Optimal Support of The Codebook Distribution

Proof of Theorem 1. First, we assume $\overline{\mathrm{supp}(\mathcal{P}_B)} = \overline{\mathrm{supp}(\mathcal{P}_A)}$ . Then for any $z \in \mathrm{supp}(\mathcal{P}_A)$ , there exist a sequence of points in $\mathrm{supp}(\mathcal{P}_B)$ that converge to $z$ . Let $\{e_k\}_{k=1}^K$ be $K$ code vectors independently generated from $\mathcal{P}_B$ . Then the empirical distribution of $\{e_k\}_{k=1}^K$ tends to $\mathcal{P}_B$ as the size $K$ tends to infinity. Since $\Omega = \mathrm{supp}(\mathcal{P}_A)$ is a bounded region, we have the following:

$$
\sup _ {\boldsymbol {z} \in \overline {{\operatorname{supp} (\mathcal {P} _ {A})}}} \min _ {k} \| \boldsymbol {z} - \boldsymbol {e} _ {k} \| ^ {2} = \sup _ {\boldsymbol {z} \in \overline {{\operatorname{supp} (\mathcal {P} _ {B})}}} \min _ {k} \| \boldsymbol {z} - \boldsymbol {e} _ {k} \| ^ {2} \xrightarrow {p} 0, \quad \text {as} K \to \infty .\tag{18}
$$

This quantity is an upper bound on the quantization error $\mathcal{E}(\{z_{i}\};\{e_{k}\})$ . Thus,

$$
\sup _ {\{\boldsymbol {z} _ {i} \} \subseteq \Omega} \mathcal {E} \left(\{\boldsymbol {z} _ {i} \} _ {i = 1} ^ {N}; \{\boldsymbol {e} _ {k} \} _ {k = 1} ^ {K}\right) \leq \sup _ {\boldsymbol {z} \in \overline {{\Omega}}} \min _ {k} \| \boldsymbol {z} - \boldsymbol {e} _ {k} \| ^ {2} \xrightarrow {p} 0, \quad \text { as } K \to \infty .\tag{19}
$$

This demonstrates that $P_{B}$ has vanishing quantization error asymptotically. Furthermore, for any K code vectors $\{e_{k}\}_{k=1}^{K}$ independently drawn from $P_{B}$ , we have $\{e_{k}\}_{k=1}^{K} \subseteq \overline{\Omega}$ . Since the empirical distribution of $\{z_{i}\}_{i=1}^{N}$ tends to $P_{A}$ as the feature sample size N tends to infinity, we can easily show that for any fixed $\{e_{k}\}_{k=1}^{K} \subseteq \overline{\Omega}$ , the codebook utility rate satisfies

$$
\mathcal {U} \left(\{\boldsymbol {z} _ {i} \} _ {i = 1} ^ {N}, \{\boldsymbol {e} _ {k} \} _ {k = 1} ^ {K}\right) \xrightarrow {p} 1, \quad \text { as } N \to \infty .\tag{20}
$$

This shows that $\{e_k\}_{k=1}^K$ attains full utilization asymptotically, and thus $\mathcal{P}_B$ attains full utilization asymptotically.

On the other hand, we assume $P_{B}$ attains full utilization and vanishing quantization error asymptotically. Then we first claim that $\overline{\operatorname{supp}(\mathcal{P}_{A})} \subseteq \overline{\operatorname{supp}(\mathcal{P}_{B})}$ . Since $P_{B}$ has vanishing quantization error asymptotically, then for any $z \in \operatorname{supp}(\mathcal{P}_{A})$ , there exist a sequence of points in $\operatorname{supp}(\mathcal{P}_{B})$ that converge to z. This implies that $\operatorname{supp}(\mathcal{P}_{A}) \subseteq \overline{\operatorname{supp}(\mathcal{P}_{B})}$ and thus $\overline{\operatorname{supp}(\mathcal{P}_{A})} \subseteq \overline{\operatorname{supp}(\mathcal{P}_{B})}$ .

To show $\overline{\mathrm{supp}(\mathcal{P}_B)} = \overline{\mathrm{supp}(\mathcal{P}_A)}$ , it remains to show $\mathrm{supp}(\mathcal{P}_B) \subseteq \overline{\mathrm{supp}(\mathcal{P}_A)}$ . In fact, if $\mathrm{supp}(\mathcal{P}_B) \subseteq \overline{\mathrm{supp}(\mathcal{P}_A)}$ does not hold, then there exists an open region $\mathcal{R} \subseteq \mathrm{supp}(\mathcal{P}_B) - \overline{\mathrm{supp}(\mathcal{P}_A)}$ such that $\mathcal{P}_B(\mathcal{R}) > 0$ and

$$
\min _ {\boldsymbol {z} \in \mathrm{supp} (\mathcal {P} _ {A}), \boldsymbol {z} ^ {\prime} \in \mathcal {R}} \| \boldsymbol {z} - \boldsymbol {z} ^ {\prime} \| \geq \epsilon_ {0}\tag{21}
$$

for some $\epsilon_0 > 0$ . Since $\mathrm{supp}(\mathcal{P}_A) \subseteq \overline{\mathrm{supp}(\mathcal{P}_B)}$ , then there exists a sufficiently large $K_0$ such that the event

$$
\left\{\text { Generating } \{e _ {k} \} _ {k = 1} ^ {K _ {0}} \text {   i.i.d.   from   } \mathcal {P} _ {B} \text {   s.t.   } \{e _ {k} \} \subseteq \text { supp } (\mathcal {P} _ {A}), \sup _ {\boldsymbol {z} \in \text { supp } (\mathcal {P} _ {A})} \min _ {k} \| \boldsymbol {z} - \boldsymbol {e} _ {k} \| <   \epsilon_ {0} \right\}\tag{22}
$$

has some positive probability C > 0. Then with a positive probability of at least $C \cdot \mathcal{P}_{B}(\mathcal{R})$ , we can pick the first $K_{0}$ code vectors from Equation (22) and the $(K_{0} + 1)$ th code vector from R. For any such codebook of size $K_{0} + 1$ , we know the $(K_{0} + 1)$ th code vector will never be used regardless of the choice of the feature set $\{z_{i}\}$ . Therefore, the codebook utilization

$$
\sup _ {\{\boldsymbol {z} _ {i} \}} \mathcal {U} \left(\{\boldsymbol {e} _ {k} \} _ {k = 1} ^ {K _ {0} + 1}; \{\boldsymbol {z} _ {i} \}\right) \leq \frac {K _ {0}}{K _ {0} + 1} <   1.\tag{23}
$$

This contradicts the property that $\mathcal{P}_B$ attains full utilization asymptotically. Thus, $\mathrm{supp}(\mathcal{P}_B) \subseteq \overline{\mathrm{supp}(\mathcal{P}_A)}$ must hold. This concludes the proof.

## B Understanding Codebook Collapse Through the Lens of Voronoi Partition

## B.1 Voronoi Partition: Definition and Connection to Codebook Collapse

Let X be a metric space with distance function $d(\cdot,\cdot)$ , and let $\{e_{k}\}_{k=1}^{K}$ denote a set of code vectors. The Voronoi cell (or Voronoi region) $R_{k}$ associated with code vector $e_{k}$ is the set of all points in X whose distance to $e_{k}$ is no greater than their distance to any other code vector $e_{j}$ ( $j\neq k$ ):

$$
\mathcal {R} _ {k} = \{x \in \mathcal {X} \mid d (x, \boldsymbol {e} _ {k}) \leq d (x, \boldsymbol {e} _ {j}), \forall j \neq k \}.\tag{24}
$$

![](images/82b11f53fd65f020e9582e6cc34995ef7a005bb2e2aa2093b0012d5c37495750.jpg)  
(a)

![](images/e33b88ec8c1be4f64c65f220347876e7c513b63c0dd97383f05a278d1ab260f5.jpg)  
(b)

![](images/65300bb68247902b2e5122981bab99b84fdbc43a224cce88189aeb4e5cd4a60b.jpg)  
(c)

![](images/137b7c3d01bf5d41c8afc95431198e0b0286d2cdbf753562b054f541dc093c3b.jpg)  
(d)  
Figure 12: Visualization of the Voronoi partition. The symbols $\cdot$ and $\times$ represent the feature and code vectors, respectively.

The Voronoi diagram is the collection of all cells $\{R_{k}\}_{k=1}^{K}$ . Figure 12 illustrates a Voronoi diagram for 12 code vectors, partitioning the metric space into 12 regions according to $R_{k}$ . When d is the $\ell_{2}$ distance, vector quantization (VQ) can equivalently be expressed in terms of Voronoi regions:

$$
\forall \boldsymbol {z} _ {i} \in \mathcal {R} _ {j}, \quad \boldsymbol {z} _ {i} ^ {\prime} = \underset {\boldsymbol {e} \in \{\boldsymbol {e} _ {k} \}} {\arg \min} \| \boldsymbol {z} _ {i} - \boldsymbol {e} \| = \boldsymbol {e} _ {j},\tag{25}
$$

where $z_{i}$ is an arbitrary feature vector. Equation 25 provides an alternative viewpoint for nearest neighbor search: identify the region $\mathcal{R}_j$ containing $z_{i}$ and select the corresponding code vector $e_j$ .

Relation to Codebook Collapse Codebook collapse occurs in its most severe form when all feature vectors fall within the same Voronoi region. As illustrated in Figure 12a, all features occupy a single region, leading to the utilization of only one code vector. To prevent collapse, feature vectors should be distributed across all regions as uniformly as possible (Figure 12d).

## B.2 Limitations of Existing Vector Quantization Methods

We analyze why standard VQ methods often fail to prevent codebook collapse, using Vanilla VQ $[49]$ and k-means-based VQ $[39]$ as examples. Both share a similar assignment step but differ in their update mechanisms.

Assignment Step Given feature vectors $\{z_{i}\}_{i=1}^{N}$ and code vectors $\{e_{k}\}_{k=1}^{K}$ , at iteration t, both methods partition the feature space into Voronoi cells and assign features to the nearest code vectors:

$$
\mathcal {R} _ {k} ^ {(t - 1)} = \{x \in \mathcal {X} \mid \left\| x - \boldsymbol {e} _ {k} ^ {(t - 1)} \right\| _ {2} ^ {2} \leq \left\| x - \boldsymbol {e} _ {j} ^ {(t - 1)} \right\| _ {2} ^ {2}, \forall j \neq k \}, \quad \mathcal {S} _ {k} ^ {(t)} = \{\boldsymbol {z} _ {i} \mid \boldsymbol {z} _ {i} \in \mathcal {R} _ {k} ^ {(t - 1)} \}.
$$

Update Step in Vanilla VQ Code vectors are updated via gradient descent on the reconstruction loss:

$$
\mathcal {L} = \frac {1}{N} \sum_ {k = 1} ^ {K} \sum_ {\boldsymbol {z} _ {m} \in \mathcal {S} _ {k} ^ {(t)}} \left\| \boldsymbol {z} _ {m} - \boldsymbol {e} _ {k} ^ {(t - 1)} \right\| _ {2} ^ {2}.\tag{26}
$$

Update Step in k-means-based VQ Code vectors are updated using an exponential moving average:

$$
\boldsymbol {e} _ {k} ^ {(t)} = \alpha \boldsymbol {e} _ {k} ^ {(t - 1)} + (1 - \alpha) \frac {1}{| \mathcal {S} _ {k} ^ {(t)} |} \sum_ {\boldsymbol {z} _ {m} \in \mathcal {S} _ {k} ^ {(t)}} \boldsymbol {z} _ {m}.\tag{27}
$$

Codebook Collapse in Vanilla and k-means VQ Despite different update strategies, both methods can suffer from codebook collapse because the assignment step does not guarantee that all Voronoi cells receive feature vectors (Figures 12a–12c). Larger codebooks exacerbate the issue, leaving some cells unassigned and their corresponding code vectors underutilized.

Connection to Distribution Matching and Mitigation As demonstrated in Section 4, both VQ methods are sensitive to codebook initialization. Collapse is mitigated only if the codebook distribution approximates the feature distribution. However, in practice, feature distributions are typically unknown and dynamically evolving. To address this, we introduce a distributional matching constraint that aligns the codebook distribution with the feature distribution, ensuring complete codebook utilization.

## C The Relationship Between Gradient Gap and Quantization Error

In this section, we provide a formal derivation of the relationship between the Quantization Error (Criterion 1) and the Gradient Gap caused by the Straight-Through Estimator (STE). This analysis theoretically supports our claim that minimizing quantization error via distribution matching inherently mitigates training instability.

Problem Setup: Let L denote the global loss function. In the VQ process, the encoder outputs a continuous latent vector $z_{e}$ , which is discretized to a code $z_{q}$ . During backpropagation, the non-differentiable quantization operation is bypassed using the STE, which approximates the gradient of the encoder output $\frac{\partial L}{\partial z_{e}}$ using the gradient of the quantized code $\frac{\partial L}{\partial z_{q}}$ :

$$
\text { STE   Approximation: } \quad \frac {\partial \mathcal {L}}{\partial z _ {e}} \leftarrow \frac {\partial \mathcal {L}}{\partial z _ {q}}\tag{28}
$$

We define the Gradient Gap (G) as the magnitude of the discrepancy between the true gradient and the STE approximated gradient:

$$
\mathcal {G} := \left\| \frac {\partial \mathcal {L}}{\partial z _ {e}} - \frac {\partial \mathcal {L}}{\partial z _ {q}} \right\|\tag{29}
$$

Taylor Expansion Analysis: To analyze this gap, we perform a Taylor expansion of the gradient function around the quantized point $z_{q}$ . Let us consider the gradient function with respect to the latent variable x. Expanding $\frac{\partial L}{\partial z_{e}}$ at the point $x = z_{q}$ , we obtain:

$$
\frac {\partial \mathcal {L}}{\partial z _ {e}} = \left. \frac {\partial \mathcal {L}}{\partial x} \right| _ {x = z _ {q}} + \left. \frac {\partial^ {2} \mathcal {L}}{\partial x ^ {2}} \right| _ {x = z _ {q}} (z _ {e} - z _ {q}) + \mathcal {O} (\| z _ {e} - z _ {q} \| ^ {2})\tag{30}
$$

where $\left.\frac{\partial\mathcal{L}}{\partial x}\right|_{x = z_q}$ is exactly the gradient backpropagated from the quantized code, i.e., $\frac{\partial\mathcal{L}}{\partial z_q}$ , and $\mathbf{H} = \left.\frac{\partial^2\mathcal{L}}{\partial x^2}\right|_{x = z_q}$ is the Hessian matrix, representing the local curvature of the loss function at $z_q$ .

First-Order Approximation and Upper Bound: By retaining only the first-order term (as is common in gradient gap analysis) and ignoring the higher-order terms $\mathcal{O}(\|z_{e}-z_{q}\|^{2})$ , the relationship can be approximated as:

$$
\frac {\partial \mathcal {L}}{\partial z _ {e}} \approx \frac {\partial \mathcal {L}}{\partial z _ {q}} + \mathbf {H} (z _ {e} - z _ {q})\tag{31}
$$

Substituting this back into the definition of the Gradient Gap G, we get:

$$
\mathcal {G} = \left\| \frac {\partial \mathcal {L}}{\partial z _ {e}} - \frac {\partial \mathcal {L}}{\partial z _ {q}} \right\| \approx \| \mathbf {H} (z _ {e} - z _ {q}) \|\tag{32}
$$

By applying the definition of the consistent matrix norm (specifically, the spectral norm for the symmetric Hessian), we derive the upper bound:

$$
\mathcal {G} \leq \| \mathbf {H} \| _ {2} \cdot \| z _ {e} - z _ {q} \|\tag{33}
$$

where $\|H\|_{2}$ represents the spectral norm of the Hessian (indicative of the maximum curvature) and $\|z_{e}-z_{q}\|$ corresponds to the Quantization Error.

The derivation in Eq. 33 explicitly demonstrates that the Gradient Gap G is linearly bounded by the Quantization Error. In methods like Vanilla VQ, a large Quantization Error implies a looser bound on G, leading to inaccurate gradient signals passed to the encoder. This large discrepancy causes the “Training Instability” observed in previous works. By minimizing the Wasserstein distance, our method explicitly minimizes the Quantization Error (Criterion 1). This tightens the upper bound on G, ensuring that the gradient passed to the encoder $\left(\frac{\partial\mathcal{L}}{\partial z_{q}}\right)$ is a faithful approximation of the true gradient $\left(\frac{\partial\mathcal{L}}{\partial z_{e}}\right)$ .

![](images/feb69afc8f16603bb9d0f75bc127f188fdac6d2bcc51711447de7310823d08cc.jpg)  
(a) (50%, 4.92)

![](images/896c8fe495b7ed20c70b2895d528bdb0dafe733b9cc49d6ab1fadf547ff9a172.jpg)  
(b) (100%, 10.00)

![](images/7acacc64aafde1ff6bad32352c6f165407338e8881c4a0d6eae1eef904a22198.jpg)  
(c) (100%, 1.02)

![](images/6f35d29568364d19f1cd25d182ab07197a5cde4d49c7514ccdaf07a2337cd479.jpg)  
Figure 13: Visualization of the evaluation criteria $(\mathcal{U},\mathcal{C})$ .  
(d) (100%, 4.92)

## D Complementary Roles of Criterion 2 and 3 in Assessing Codebook Collapse

To clarify the complementary roles of Criterion 2 and Criterion 3, which are introduced in Section 2.2, we provide visual illustrations to facilitate intuitive understanding. The metric U is designed to quantify the completeness of codebook utilization. As shown in Figures 13a and 13b, the corresponding values of U are 50% and 100%, respectively $^{3}$ .

However, U alone is insufficient to characterize the severity of codebook collapse, since it does not account for imbalance among utilized code vectors. This limitation is illustrated in Figure 13c. Although all code vectors are utilized, resulting in $U = 100\%$ , the code vector $e_{3}$ overwhelmingly dominates the assignment distribution. Such extreme imbalance constitutes a degenerate form of codebook collapse, even though U attains its maximum value. This observation motivates the introduction of Criterion 3, which explicitly measures the uniformity, or conversely the imbalance, of codebook utilization.

A comparison between Figures 13b and 13c further highlights the discriminative capability of Criterion 3. While both cases share the same utilization completeness, namely U = 100%, their corresponding values of C differ substantially, being 10.00 and 1.02, respectively. This contrast demonstrates that Criterion 3 effectively captures differences in the distribution of $p_{k}$ that are not reflected by U. In particular, the near minimal value of C in Figure 13c correctly identifies this scenario as a collapsed codebook, which aligns with our intended interpretation.

Nevertheless, Criterion 3 alone is also insufficient for a comprehensive evaluation of codebook collapse. As illustrated by Figures 13a and 13d, identical values of C can correspond to markedly different levels of utilization completeness, as indicated by their substantially different values of U. This discrepancy shows that C by itself cannot quantify the proportion of actively utilized code vectors.

Consequently, in this work, we adopt the joint use of Criterion 2 and Criterion 3 to quantitatively assess codebook collapse. Effective mitigation is achieved only when both metrics attain sufficiently large values, indicating that the codebook is both fully utilized and balanced in its assignment distribution.

## E The Significant Impact of Distribution Variance on Quantization Error

As discussed in Section 2.3 and 2.4, the optimal criterion triple is achieved when the distributions $\mathcal{P}_A$ and $\mathcal{P}_B$ are identical. In this section, we further analyze the criterion triple by the lens of distribution variance under the condition that both distributions are identical. Specifically, we first sample a set of feature vectors $\{\pmb{z}_i\}_{i=1}^N$ along with a set of code vectors $\{\pmb{e}_k\}_{k=1}^K$ from the distribution $\mathcal{N}_d(\mathbf{0},\sigma^2\mathbf{I})$ or the distribution $\mathrm{Unif}_d(-\zeta,\zeta)$ . We then calculate the evaluation criteria according to their definitions in Section 2.2. As demonstrated in Table 9, $\sigma$ and $\zeta$ have a substantial impact on $\mathcal{E}$ , while $\mathcal{U}$ and $\mathcal{C}$ remains largely unaffected.

Table 9: The criterion triple influence by the distribution variance.

<table><tr><td rowspan="2">Evaluation Criteria</td><td colspan="5"> $\sigma$ </td><td colspan="5"> $\zeta$ </td></tr><tr><td>0.0001</td><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td><td>0.0001</td><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td> $\mathcal{E}$ </td><td>1.25e-8</td><td>1.25e-6</td><td>1.25e-4</td><td>1.24e-2</td><td>1.25</td><td>3.27e-9</td><td>3.27e-7</td><td>3.27e-5</td><td>3.27e-3</td><td>0.327</td></tr><tr><td> $\mathcal{U}$ </td><td>0.9934</td><td>0.9938</td><td>0.9940</td><td>0.9934</td><td>0.9941</td><td>0.9993</td><td>0.9986</td><td>0.9990</td><td>0.9992</td><td>0.9989</td></tr><tr><td> $\mathcal{C}$ </td><td>7265.3</td><td>7260.3</td><td>7267.7</td><td>7255.0</td><td>7275.8</td><td>7380.2</td><td>7372.2</td><td>7387.9</td><td>7397.5</td><td>7391.6</td></tr></table>

This experimental finding suggests that when the distribution variance of the feature vectors is uncontrollable or unknown, reporting a comparison of quantization error among various VQ methods is unreasonable. This is because the improvement in quantization error is predominantly attributed to the reduction in distribution variance rather than the effectiveness of the VQ methods. To evaluate various VQ methods in terms of the criterion triple, we establish an atomic and fair experimental setting in Section 4, where the feature distributions for all VQ methods are identical.

![](images/954ac9baa66e1ada2f813172fbaeb3cda657020de364b4678af91a07626445d4.jpg)

![](images/73cfa57ff661cc902be96844de08428f58e9cbe49439ca645c14ace01acec444.jpg)  
(a) E w.r.t. K  
(b) $\mathcal{E}$ w.r.t. $d$

![](images/02363982b9f1b4cef3a4cdcfcbe105c51e7656c3bf76ba75edeee865ed8537ce.jpg)

![](images/392a0d119ee919a842f301b3d55ca7216955c7d83cf7d018e55e9cd4e3fa4313.jpg)

![](images/ab26c1d2cbc4c8c20bc91d7b5d01569bc06c93a43e5bb75a943835ca06b773d7.jpg)

(c) $\mathcal{E}$ w.r.t. $N$  
![](images/b5bd7c0b00dfd7a3a12d2e7b673a67d9bd9b12e51d322c2230553a61dca05c34.jpg)

![](images/8d7323b31d1ca20c80c20a5a18a109a98e695b8b4d039a8123e2c78113111abf.jpg)

![](images/0205499b81abde0930b8acbcdba1679c0b3190e05a9cc2c8e426ab0d36f1f0c5.jpg)

(d) $\mathcal{E}$ w.r.t. $K$  
![](images/6add9ec1d012f7fc6c2441d063328833f39eaed93813b4023ac1e9e6ad748816.jpg)  
(e) $\mathcal{E}$ w.r.t. $d$

![](images/be69b3b6b5a35ca68fcd990524af25ffc0b56641fb775f4667a26a7166b5bec3.jpg)

![](images/beb9a8df2dd408cc7f811b9f1b51cb27f9ae7fc5b1f59b291d195f66b81d6e5d.jpg)

(f) $\mathcal{E}$ w.r.t. $N$  
![](images/2da3c63111b632f72170d5691a4a6d4209942dbe5aa5ceaf552f8020cd0ade0b.jpg)  
(j) $\mathcal{U}$ w.r.t. $K$  
(k) U w.r.t. d

(g) $\mathcal{U}$ w.r.t. $K$  
(h) $\mathcal{U}$ w.r.t. $d$  
(i) U w.r.t. N  
![](images/8e0707a1976c2a930191aa4a24afab4f39382cb48e74dac56b88e85b27171305.jpg)

(1) $\mathcal{U}$ w.r.t. $N$  
![](images/44c5c7f4187fc9eef21bfde2821dda2a493773ff8427f1ffbc5d325a80af75b6.jpg)  
(m) C w.r.t. K  
(n) $\mathcal{C}$ w.r.t. $d$

![](images/33b819732cd1c38325192696ae12d468d041280d9cbf506a3d4fc0ca0f13b9f0.jpg)  
(o) C w.r.t. N

![](images/41128063ae45eef50592badfa37ec35cb9cd0dc15226a8c16af37040dda2ffae.jpg)  
(p) C w.r.t. K

![](images/e89c3962bd476e986ad3c2509e44e357da7e40a8e66b2bc3e792d895cd6413a9.jpg)

![](images/8ef7ad67e393167155a39ef80e701661f010d3654d6b299e267ff423c0d2e890.jpg)  
(q) C w.r.t. d  
(r) C w.r.t. N  
Figure 14: Quantitative analyses of the criterion triple when $P_{A}$ and $P_{B}$ are Gaussian distributions.

## F Interpretation of Qualitative Distributional Matching Results

This section interprets the experimental results presented in Figure 3. The VQ process relies on nearest neighbor search for code vector selection. As evident from Figure 3a to 3d, actively selected code vectors are predominantly those located in close proximity to or within the feature distribution, while distant ones remain unselected. This leads to highly uneven code vector utilization $p_{k}$ , with those closer to the feature distribution being excessively used. This elucidates the significantly low U and C observed in Figure 3a. Furthermore, a notable quantization error, e.g., E = 1.19 in Figure 3a, arises when the codebook and feature distributions are mismatched, forcing feature vectors outside the codebook to settle for distant code vectors. Conversely, as the disk centers align, leading to a closer match between the two distributions, an increased number of code vectors become actively engaged. Additionally, code vectors are utilized more uniformly, and feature vectors can select nearer counterparts. This accounts for the improvement of criterion triple values towards optimality as the distributions align.

Analogously, we can employ nearest neighbor search to interpret the second case. When code vectors are distributed within the range of feature vectors, as illustrated in Figure 3e and Figure 3f, the majority of code vectors would be actively utilized, ensuring high U. However, the utilization of these code vectors is not uniform; code vectors on the periphery of the codebook distribution are more frequently used, leading to relatively low C. Feature vectors on the periphery will have larger distances to their nearest code vectors, resulting in higher E. Conversely, when feature vectors fall within the range of code vectors, as depicted in Figure 3g and Figure 3h, outer code vectors remain largely unused, leading to a lower U and C. Since only inner code vectors are active, each feature vector can find a nearby counterpart, maintaining low E.

## G Supplementary Quantitative Analyses on Distribution Matching: Further Supporting the Main Findings in Section 2.3

To further elucidate the effects of the distributional matching, we conduct more quantitative analyses centered around the criterion triple $(\mathcal{E},\mathcal{U},\mathcal{C})$ .

## G.1 Codebook Distribution and Feature Distribution are Gaussian Distributions

We begin by assuming that the distributions $P_{A}$ and $P_{B}$ are Gaussian. We generate a set of feature vectors $\{z_{i}\}_{i=1}^{N}$ from $\mathcal{N}_{d}(\mathbf{0},\mathbf{I})$ and a set of code vectors $\{e_{k}\}_{k=1}^{K}$ from $\mathcal{N}_{d}(\mu\cdot\mathbf{1},\mathbf{I})$ , with $\mu$ varying within $\{0.0, 0.5, 1.0, 1.5, 2.0, 2.5\}$ . The criterion triple results are presented in Figures 14a to 14c, Figures 14g to 14i, and Figures 14m to 14o. Across all tested configurations of K, d, N, we consistently observe that when $\mu = 0$ — indicating identical distributions between $P_{A}$ and $P_{B}$ — the criterion triple achieves the lowest E, highest U, and largest C. This empirical evidence reinforces the effectiveness of aligning feature and codebook distributions in VQ.

![](images/238e520ede840c9c04d58dee397b2ac4c02b8b5cdb1bfe2b12e24335c10550d6.jpg)

![](images/57f55131c73947eace2a96937839bbec9c0afee93470fe2aea160482705f220f.jpg)  
(a) $\mathcal{E}$ w.r.t. $K$

![](images/f88a9c7f790a685ef7dadcab4af6704119c2469590fb6ff78079ef0c5885abc8.jpg)  
(b) $\mathcal{E}$ w.r.t. $d$

![](images/e1232b433941cd9394aa68e7d8a83f26d01aabe6e3371f5d0333aa27bc77d030.jpg)

![](images/a7dabb8332ed99d25de9f5166fe151e68e04d92e2dd68e7d4a91c3a90f1da5a4.jpg)  
(c) E w.r.t. N

![](images/8e3f16e65635b62e988bba0d45fa368dce22b05fa214cb31b66e607bf1d33066.jpg)

![](images/6f83be81469eb31a090a821504c8f98b7fe58b9d28dbb4eba28412150cf798bd.jpg)

(d) $\mathcal{E}$ w.r.t. $K$  
![](images/d468618a0a170eaba1681d136489240494053fe7393a33bc2cb11a9a2f0bd99b.jpg)

![](images/ca1e10710aab1f5532fe5c799e90fa752eca91cae80ca511476cac3885bb0184.jpg)

(e) $\mathcal{E}$ w.r.t. $d$  
![](images/2477b9c530a7acf255feaa0cea86374027c91e2420d602f7faf4bb40f4941908.jpg)  
(h) $\mathcal{U}$ w.r.t. $d$  
(g) $\mathcal{U}$ w.r.t. $K$

(f) E w.r.t. N  
![](images/2a03dd51eca0022d314b1347497b1ee41044d6122fa8bbec88db6ba5ef320fa1.jpg)  
(i) $\mathcal{U}$ w.r.t. $N$  
(j) $\mathcal{U}$ w.r.t. $K$

![](images/4efb2a1a25646e01123d7f7ddcea206afeb9d922b5e9883c35498215b97b72f1.jpg)

![](images/c2d3ea2ada419535385214ba3aef781e5416a0b14b7f20f6d20b2c050e0cabf9.jpg)

(k) U w.r.t. d  
![](images/3b524c24ce1e6275e918ad3364b687d60ba8ada5f6d35e532056c184af8e5802.jpg)  
(m) C w.r.t. K  
(n) $\mathcal{C}$ w.r.t. $d$

(1) $\mathcal{U}$ w.r.t. $N$  
![](images/752d4937dc0bd7bb753dbe0eb291e6fb496d32c0c493647dc046aa3dfd2adece.jpg)  
(o) C w.r.t. N

![](images/eb55de4987b07728fdb4a77a69262141e52b48e6f7339bf0c8e84338c507a642.jpg)  
(p) $\mathcal{C}$ w.r.t. $K$

![](images/22df906456d4190e0ff47f8dca0367ad97787062c47aff612002411d93f1c2b0.jpg)  
(q) C w.r.t. d

![](images/4b423c0210210fec591f9022f42f5be8ee7a734c9d99cc8b240a859c4b88d1ea.jpg)  
(r) C w.r.t. N  
Figure 15: Quantitative analyses of the criterion triple when $P_{A}$ and $P_{B}$ are uniform distributions.

Additionally, we further analyze the criterion triple by varying the covariance matrix. We sample a set of feature vectors $\{z_{i}\}_{i=1}^{N}$ from the distribution $\mathcal{N}_{d}(\mathbf{0},\mathbf{I})$ and a set of code vectors $\{e_{k}\}_{k=1}^{K}$ from $\mathcal{N}_{d}(\mathbf{0},\sigma^{2}\mathbf{I})$ , where $\sigma$ is selected from $\{1,2,3,4,5,6\}$ . The results for the criterion triple are shown in Figures 14d to 14f, Figures 14j to 14l, and Figures 14p to 14r. When $\sigma=1$ , indicating identical distributions between $P_{A}$ and $P_{B}$ , all three evaluation criteria reach their optimal values: the lowest E, highest U, and largest C across all tested values of K,d,N. This result corroborates our earlier findings.

## G.2 Codebook Distribution and Feature Distribution are Unifrom Distributions

The above conclusion holds when $P_{A}$ and $P_{B}$ are other types of distributions, such as the uniform distribution. As shown in Figure 15, we sample a set of feature vectors $\{z_{i}\}_{i=1}^{N}$ from the distribution $\operatorname{Unif}_{d}(-1,1)$ and a set of code vectors $\{e_{k}\}_{k=1}^{K}$ from $\operatorname{Unif}_{d}(\nu-1,\nu+1)$ , where $\nu$ is selected from the set $\{0.0,0.5,1.0,1.5,2.0,2.5\}$ or from $\operatorname{Unif}_{d}(-\zeta,\zeta)$ , with $\zeta$ drawn from the set $\{1,2,3,4,5,6\}$ . We observe that when $\mu=0$ or $\zeta=1$ —indicating that $P_{A}$ and $P_{B}$ have identical distributions—the performance in terms of the criterion triple is optimal, achieving the lowerest E, the highest U, and the largest C across all tested values of K, d, N. Therefore, we conclude that our quantitative analyses are distribution-agnostic and can be generalized to other distributions.

## H Statistical Distances over Gaussian Distributions

We first introduce the definition of Wasserstein distance.

Definition 4. The Wasserstein distance or earth-mover distance with p norm is defined as below:

$$
W _ {p} (\mathbb {P} _ {r}, \mathbb {P} _ {g}) = (\inf _ {\gamma \in \Pi (\mathbb {P} _ {r}, \mathbb {P} _ {g})} \mathbb {E} _ {(x, y) \sim \gamma} \big [ \| x - y \| ^ {p} \big ]) ^ {1 / p}.\tag{34}
$$

where $\Pi(\mathcal{P}_{r},\mathcal{P}_{g})$ denotes the set of all joint distributions $\gamma(x,y)$ whose marginals are $P_{r}$ and $P_{g}$ respectively. Intuitively, when viewing each distribution as a unit amount of earth/soil, the Wasserstein distance (also known as earth-mover distance) represents the minimum cost of transporting “mass” from x to y to transform distribution $P_{r}$ into distribution $P_{g}$ . When p=2, this is called the quadratic Wasserstein distance.

![](images/a13ee63268d777672eda69d71809f74be6b283691a0864567fc37c91619a5476.jpg)

![](images/e2bbbd10274913b639ac71ab640c0a4094d941fc062f809e16c4305a95f65ddb.jpg)

![](images/6685663cbc78ad35b9ab0b3bb17c1ad24d9df139aedee52be29a06ced1f6ddc9.jpg)

![](images/19a04ffb72efcebf1cfc7351f4a7259cea8bc385ce537ebde53ab43c40b8ab16.jpg)

![](images/06af28308d18ba2088845dd309a3ad5d41450ae5bce3ef6a9ed9d615f6f497c8.jpg)

![](images/e90f836ddab3d47053e6099c5376e481d08c93a533a4075e98d8b02f6be0400e.jpg)

![](images/1b23439c9bb7b49097ca5e7679e96bd4698c274bbb41a1c5ef264424acb279a2.jpg)  
Figure 16: Comprehensive Q–Q plots for all latent dimensions. Subplots are ordered from the best-fitting dimension $z_{6}$ (top-left) to the least-fitting dimension $z_{7}$ (bottom-right) according to the $R^{2}$ score. Visualization: For visual clarity, blue points show a random subset of 5,000 samples. Statistics: The annotated $R^{2}$ and Shapiro–Wilk W statistics (green/red boxes) are computed on the full validation set (N = 409,600). Across all dimensions, the Q–Q plots indicate a strong Gaussian fit for the central mass of the distributions, with only mild deviations in the extreme tails.

In this paper, we achieve distributional matching using the quadratic Wasserstein distance under Gaussian distribution assumptions. We also examine other statistical distribution distances as potential loss functions for distributional matching and compare them with the Wasserstein distance. Specifically, we provide the Kullback-Leibler divergence and the Bhattacharyya distance over Gaussian distributions in Lemma 5 and Lemma 6. It can be observed that the KL divergence for two Gaussian distributions involves calculating the determinant of covariance matrices, which is computationally expensive in moderate and high dimensions. Moreover, the calculation of the determinant is sensitive to perturbations and it requires full rank (In the case of not full rank, the determinant is zero, rendering the logarithm of zero undefined), which can be impractical in many cases. Other statistical distances like Bhattacharyya Distance suffer from the same issue. In contrast, quadratic Wasserstein distance does not require the calculation of the determinant and full-rank covariance matrices.

Lemma 5 (Kullback-Leibler divergence [31]). Suppose two random variables $\mathbf{Z}_{1} \sim \mathcal{N}(\boldsymbol{\mu}_{1}, \boldsymbol{\Sigma}_{1})$ and $\mathbf{Z}_{2} \sim \mathcal{N}(\boldsymbol{\mu}_{2}, \boldsymbol{\Sigma}_{2})$ obey multivariate normal distributions, then Kullback-Leibler divergence between Z1 and $Z_{2}$ is:

$$
D _ {\mathrm{KL}} (\mathbf {Z} _ {1}, \mathbf {Z} _ {2}) = \frac {1}{2} ((\pmb {\mu} _ {1} - \pmb {\mu} _ {2}) ^ {T} \pmb {\Sigma} _ {2} ^ {- 1} (\pmb {\mu} _ {1} - \pmb {\mu} _ {2}) + \mathrm{tr} (\pmb {\Sigma} _ {2} ^ {- 1} \pmb {\Sigma} _ {1} - \mathbf {I}) + \ln \frac {\det \pmb {\Sigma} _ {2}}{\det \pmb {\Sigma} _ {1}}).
$$

Lemma 6 (Bhattacharyya Distance [4]). Suppose two random variables $\mathbf{Z}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $\mathbf{Z}_2 \sim \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ obey multivariate normal distributions, $\boldsymbol{\Sigma} = \frac{1}{2} (\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2)$ , then bhattacharyya distance between $\mathbf{Z}_1$ and $\mathbf{Z}_2$ is:

$$
\mathcal {D} _ {B} (\mathbf {Z} _ {1}, \mathbf {Z} _ {2}) = \frac {1}{8} (\boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {2}) ^ {T} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {2}) + \frac {1}{2} \ln \frac {\det \boldsymbol {\Sigma}}{\sqrt {\det \boldsymbol {\Sigma} _ {1} \det \boldsymbol {\Sigma} _ {2}}}.
$$

## I Statistical Assessment of Gaussianity

To further validate the Gaussian assumption discussed in Section 5.1, we conduct a comprehensive set of univariate and multivariate normality tests on the validation set. Specifically, we analyze N = 409,600 latent feature vectors (d = 8), extracted from 1,600 images in the FFHQ dataset.

Univariate Analysis. Figure 16 shows the Q–Q plots for all eight dimensions of the latent feature vector, sorted by their $R^{2}$ goodness-of-fit scores. Consistent with the observations in the main paper, the majority of dimensions $(z_{6}, z_{3}, z_{8}, z_{5}, z_{4})$ demonstrate an excellent fit to the Gaussian distribution, with $R^{2} > 0.99$ . Although the remaining dimensions exhibit mild deviations in the extreme tails, their

![](images/46d3fd715087ec6410d00a319026d8e46d138058f3e9e7e17dc43486a71a9392.jpg)

![](images/80ff171174ba9fbaed9fa45a1745de6ffed56d12bb33d3482a926a75994afeb9.jpg)  
Figure 17: Multivariate normality assessment using Mahalanobis distance. Left (Q–Q plot): The squared Mahalanobis distances of the latent features are compared against the theoretical $\chi_{8}^{2}$ quantiles. Deviations are primarily observed in the extreme tails, while the bulk of the samples closely follows the reference line. Right (CDF): The empirical cumulative distribution function (blue solid) closely matches the theoretical $\chi_{8}^{2}$ CDF (red dashed) over the primary probability mass, indicating strong agreement in the high-density region.

Shapiro–Wilk statistics remain high (W > 0.97), indicating that the central mass of the distributions is well approximated by a Gaussian.

Multivariate Analysis. To assess joint normality, we compute the squared Mahalanobis distance for each validation sample,

$$
d _ {i} ^ {2} = (z _ {i} - \mu) ^ {\top} \Sigma^ {- 1} (z _ {i} - \mu).
$$

Under the multivariate Gaussian hypothesis, $d_{i}^{2}$ should follow a Chi-squared distribution with d = 8 degrees of freedom ( $\chi_{8}^{2}$ ). The results are shown in Figure 17. The Q–Q plot (Left) indicates strong agreement between the empirical and theoretical quantiles for the majority of samples, particularly in the high-density region near the origin, while heavier tails emerge only in the extreme quantiles. To further quantify this alignment, Figure 17 (Right) compares the cumulative distribution functions (CDFs). The empirical CDF (blue solid line) closely matches the theoretical $\chi_{8}^{2}$ CDF (red dashed line) across the primary probability mass.

Overall, while deep feature representations naturally exhibit heavy-tailed behavior due to outliers, both univariate and multivariate tests consistently indicate that the bulk of the distribution is well modeled by a Gaussian. This provides strong empirical justification for using the Wasserstein distance, which is derived under Gaussian assumptions, as an efficient and effective optimization surrogate. Moreover, the comparable performance of MMD VQ, which does not impose any parametric distributional assumptions, further suggests that the Gaussian approximation captures the essential statistical structure required for high-fidelity tokenization.

Table 10: Hyperparameters for the experiments in Table 1, 2, 5, 11, 12, 7.

<table><tr><td>Frameworks</td><td>VQ-VAE</td><td>VQ-VAE</td><td>VQGAN</td><td>VQGAN</td></tr><tr><td>Dataset</td><td>CIFAR-10/SVHN</td><td>FFHQ/ImageNet</td><td>FFHQ</td><td>ImageNet</td></tr><tr><td>Input size</td><td> $32 \times 32 \times 3$ </td><td> $256 \times 256 \times 3$ </td><td> $256 \times 256 \times 3$ </td><td> $256 \times 256 \times 3$ </td></tr><tr><td>Latent size</td><td> $8 \times 8 \times 8$ </td><td> $16 \times 16 \times 8$ </td><td> $16 \times 16 \times 32$ </td><td> $16 \times 16 \times 32$ </td></tr><tr><td>encoder/decoder channels</td><td>64</td><td>64</td><td>160</td><td>160</td></tr><tr><td>encoder/decoder channel mult.</td><td>[1, 1, 2]</td><td>[1, 1, 2, 2, 4]</td><td>[1, 1, 2, 2, 4]</td><td>[1, 1, 2, 2, 4]</td></tr><tr><td>Batch size</td><td>128</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Initial Learning rate  $lr$ </td><td> $5 \times 10^{-5}$ </td><td> $5 \times 10^{-5}$ </td><td> $1 \times 10^{-5}$ </td><td> $1 \times 10^{-5}$ </td></tr><tr><td>Perceptual loss Coefficient</td><td>0</td><td>0</td><td>1.0</td><td>1.0</td></tr><tr><td>Adversarial loss Coefficient</td><td>0</td><td>0</td><td>0.4</td><td>0.4</td></tr><tr><td>Codebook dimensions</td><td>8</td><td>8</td><td>32</td><td>32</td></tr><tr><td>Training Epochs</td><td>50</td><td>30/4</td><td>30</td><td>5/10/15</td></tr><tr><td>GPU Resources</td><td>1 V100 16GB</td><td>1 A100 40GB</td><td>2 H100 80GB</td><td>2 H100 80GB</td></tr></table>

## J The Experimental Details

## J.1 Synthetic Experimental Details in Section 2.3

As depicted in Figure 3 in Section 2.3, we conduct a qualitative analyses of the criterion triple. Specifically, we sample a set of feature vectors $\{z_i\}_{i=1}^N$ from within the red circle, and a collection of code vectors $\{e_k\}_{k=1}^K$ from within the green circle, with parameters set to $K = 400$ , $N = 10000$ and $d = 2$ for the calculation of the criterion triple $(\mathcal{E}, \mathcal{U}, \mathcal{C})$ . For the visualization, we select $10\%$ of the feature vectors and $90\%$ of the code vectors for plotting.

## J.2 Synthetic Experimental Details in Appendix G

As illustrate in Figure 14 in Appendix G.1, we undertake comprehensive quantitative analyses centered around the criterion triple $(\mathcal{E},\mathcal{U},\mathcal{C})$ . In these analyses, we assume that $\mathcal{P}_A$ and $\mathcal{P}_B$ are Gaussian distributions, from which we sample a set of feature vectors $\{z_i\}_{i=1}^N$ and a collection of code vectors $\{e_k\}_{k=1}^K$ . The default parameters are set to $N = 200,000$ , $K = 1024$ , and $d = 32$ for all figures unless otherwise specified. For instance, in Figure 14a, $N$ and $d$ are taken at their default values, while the $K$ is varied within the set $\{128,256,512,1024,2048,4096,8192,16284\}$ . Additionally, each synthetic experiment is repeated five times, and the average results are reported, along with the calculation of $95\%$ confidence intervals. In all figures, mean results are represented by points, while the confidence intervals are shown as shaded areas. Identical parameter settings are employed when $\mathcal{P}_A$ and $\mathcal{P}_B$ are uniform distributions, as illustrated in Figure 15 in Appendix G.2.

## J.3 Synthetic Experimental Details in Appendix E

We set K = 8192, d = 8, N = 100000 when calculating the criterion triple $(\mathcal{E}, \mathcal{U}, \mathcal{C})$ in Appendix E. Each synthetic experiment is repeated five times, and the average results are reported in Table 9.

## J.4 Synthetic Experimental Details in Section 4

We provide experimental details of Figure 5 in Section 4. In our experimental setup, we evaluate five distinct VQ algorithms using the criterion triple $(\mathcal{E},\mathcal{U},\mathcal{C})$ . All experiments run on a single NVIDIA A100 GPU, with a codebook size K of 16,384 and dimensionality d of 8 across all algorithms. Each algorithm trains for 2,000 steps, with 50,000 feature vectors sampled from the specified Gaussian distribution at each step. For Wasserstein VQ, Vanilla VQ, and VQ + MLP, we use the SGD optimizer for training. For VQ EMA and Online Clustering, we use classical clustering algorithms—k-means [5] and k-means++[1]—to update code vectors.

## J.5 Experimental Details in Section 5

Data Augmentation For FFHQ and ImageNet-1k datasets, we follow LLama Gen [45] and apply iterative box downsampling to resize images to 256×256 resolution. For CIFAR-10 and SVHN, the images are kept at their original resolution.

Encoder-Decoder Architecture In VQ-VAE For the ImageNet and FFHQ datasets, our proposed Wasserstein VQ and all baseline methods adopt identical encoder-decoder architectures and parameter configurations. Across all baselines in these frameworks, the encoder—a U-Net [40]—downscales the input image by a factor of 16. For CIFAR-10 and SVHN datasets, the encoder reduces the input resolution by a factor of 4. Detailed hyperparameter configurations are summarized in Table 10.

Encoder-Decoder Architecture In VQGAN To accelerate adversarial training in VQGAN, our proposed Wasserstein VQ-a/b/c and Wasserstein VAR-a/b/c models on the ImageNet dataset, as well as the Wasserstein VQ model on the FFHQ dataset, are built upon the VQ-Transplant framework. Within this framework, we deploy the pretrained VAR tokenizer $[47]$ for initialization. Consequently, the encoder–decoder architecture is identical to that used in the VAR tokenizer.

Training Details in VQ-VAE All experiments employ identical training settings: we use the AdamW optimizer [32] with $\beta_{1} = 0.9$ and $\beta_{1} = 0.95$ , an initial learning rate $lr$ , and apply a halfcycle cosine decay schedule following a linear warm-up phase. For specific details on training epochs and batch sizes, refer to Table 10.

Training Details in VQ-GAN Our proposed Wasserstein VQ and Wasserstein VAR were trained on two NVIDIA H100 GPUs using the AdamW optimizer $[32]$ with $\beta_{1}=0.9$ and $\beta_{1}=0.95$ . During VQ module substitution, we used an initial learning rate of $10^{-4}$ with linear decay to $10^{-5}$ . For decoder adaptation, the learning rate remained constant at $lr=10^{-5}$ . For specific details on training epochs and batch sizes, refer to Table 10. For adversarial training, we follow the VQ-Transplant framework $[12]$ , which employs a frozen DINO-S $[6,37]$ discriminator with an architecture reminiscent of StyleGAN $[25,24]$ . Consistent with VQ-Transplant $[12]$ , we further incorporate DiffAug $[58]$ , consistency regularization $[55]$ , and LeCAM regularization $[48]$ to stabilize discriminator training.

Loss Weight in VQ-VAE For all three baselines, $\beta$ is typically set to a value within the range [0.25, 2]. In our experiments, $\beta$ is set to a fixed value of 1.0. For our proposed Wasserstein VQ model, we set $\beta$ to a much smaller value, e.g., $\beta = 0.1$ . The smaller $\beta$ values enable the Wasserstein distance to dominate the loss function, thereby more effectively narrowing the gap between the distributions.

Loss Weight in VQGAN For our proposed Wasserstein VQ and Wasserstein VAR models, the perceptual loss weight $\lambda_{P}$ is fixed at 1. In multi-scale quantization experiments, we set $\lambda_{G}=0.5$ , whereas in fixed-scale quantization experiments, $\lambda_{G}=0.4$ . The coefficient $\gamma$ is set to 0.2 for all configurations incorporating the Wasserstein distance (i.e., Wasserstein VQ and Wasserstein VAR).

## K Supplementary Results and Analyses in VQ-VAE Framework

## K.1 VQ-VAE Performance on CIFAR-10 and SVHN datasets

Due to space limitations in the main text, we have relocated the VQ-VAE evaluation on CIFAR-10 and SVHN datasets to the appendix. As demonstrated in Table 11, 12, our Wasserstein VQ consistently outperforms all baselines across both datasets, achieving superior results on nearly all evaluation metrics regardless of codebook size. Notably, we observe that Wasserstein VQ fails to reach $100\%$ codebook utilization on SVHN, which may be attributed to the dataset's limited diversity.

Table 11: Comparison of VQ-VAEs trained on CIFAR-10 dataset following [49].

<table><tr><td>Approaches</td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>Rec. Loss (↓)</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>8192</td><td>2.7%</td><td>186.9</td><td>27.15</td><td>0.83</td><td>0.0147</td></tr><tr><td>EMA VQ</td><td>64</td><td>8192</td><td>99.7%</td><td>6416.1</td><td>29.43</td><td>0.88</td><td>0.0095</td></tr><tr><td>Online VQ</td><td>64</td><td>8192</td><td>22.1%</td><td>995.4</td><td>28.20</td><td>0.85</td><td>0.0123</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>8192</td><td>100.0%</td><td>7781.8</td><td>29.88</td><td>0.90</td><td>0.0085</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>16384</td><td>1.6%</td><td>220.3</td><td>27.36</td><td>0.84</td><td>0.0141</td></tr><tr><td>EMA VQ</td><td>64</td><td>16384</td><td>80.8%</td><td>10557.3</td><td>29.43</td><td>0.88</td><td>0.0093</td></tr><tr><td>Online VQ</td><td>64</td><td>16384</td><td>13.4%</td><td>798.5</td><td>27.54</td><td>0.82</td><td>0.0141</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>16384</td><td>100.0%</td><td>15583.7</td><td>30.19</td><td>0.90</td><td>0.0080</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>32768</td><td>0.5%</td><td>154.8</td><td>27.10</td><td>0.83</td><td>0.0150</td></tr><tr><td>EMA VQ</td><td>64</td><td>32768</td><td>54.4%</td><td>14427.0</td><td>29.57</td><td>0.88</td><td>0.0091</td></tr><tr><td>Online VQ</td><td>64</td><td>32768</td><td>7.2%</td><td>1556.0</td><td>28.84</td><td>0.87</td><td>0.0106</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>32768</td><td>99.0%</td><td>29845.1</td><td>30.63</td><td>0.91</td><td>0.0071</td></tr></table>

## K.2 Analyses on Codebook Size and Dimensionality

Analyses of Codebook Size We investigate the impact of the codebook size K on the performance of VQ by varying across a wide range: $K \in [1024, 2048, 4096, 8192, 16384, 50000, 100000]$ . As shown in Table 1, 13, the vanilla VQ model suffers from severe codebook collapse even with a relatively small K, such as K = 1024. In contrast, improved algorithms like EMA VQ and Online VQ can handle smaller codebook sizes effectively, but they still experience codebook collapse when K is very large, e.g., $K \geq 50000$ . Notably, the Wasserstein VQ model consistently maintains 100% codebook utilization, irrespective of the codebook size. This underscores the effectiveness of distributional matching via the quadratic Wasserstein distance in mitigating the issue of codebook collapse.

Table 12: Comparison of VQ-VAEs trained on SVHN dataset following [49].

<table><tr><td>Approaches</td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>Rec. Loss (↓)</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>8192</td><td>8.1%</td><td>533.1</td><td>37.81</td><td>0.97</td><td>0.0018</td></tr><tr><td>EMA VQ</td><td>64</td><td>8192</td><td>56.8%</td><td>3363.0</td><td>40.38</td><td>0.98</td><td>0.0010</td></tr><tr><td>Online VQ</td><td>64</td><td>8192</td><td>27.8%</td><td>1325.1</td><td>39.04</td><td>0.97</td><td>0.0016</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>8192</td><td>88.2%</td><td>6154.5</td><td>41.04</td><td>0.98</td><td>0.0009</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>16384</td><td>3.4%</td><td>446.0</td><td>37.87</td><td>0.97</td><td>0.0017</td></tr><tr><td>EMA VQ</td><td>64</td><td>16384</td><td>22.2%</td><td>2593.8</td><td>40.19</td><td>0.98</td><td>0.0011</td></tr><tr><td>Online VQ</td><td>64</td><td>16384</td><td>13.5%</td><td>1090.5</td><td>39.12</td><td>0.97</td><td>0.0014</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>16384</td><td>87.5%</td><td>11967.2</td><td>41.49</td><td>0.98</td><td>0.0008</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>32768</td><td>1.8%</td><td>467.5</td><td>37.87</td><td>0.97</td><td>0.0017</td></tr><tr><td>EMA VQ</td><td>64</td><td>32768</td><td>35.8%</td><td>7662.9</td><td>40.25</td><td>0.98</td><td>0.0010</td></tr><tr><td>Online VQ</td><td>64</td><td>32768</td><td>7.0%</td><td>1334.8</td><td>39.26</td><td>0.97</td><td>0.0014</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>32768</td><td>88.7%</td><td>24376.3</td><td>41.84</td><td>0.98</td><td>0.0008</td></tr></table>

Table 13: Supplementary comparison of VQ-VAEs trained on FFHQ dataset following [49] w.r.t codebook size K.

<table><tr><td>Approaches</td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>Rec. Loss (↓)</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>1024</td><td>51.7%</td><td>446.2</td><td>27.64</td><td>73.0</td><td>0.0125</td></tr><tr><td>EMA VQ</td><td>256</td><td>1024</td><td>74.1%</td><td>618.9</td><td>27.66</td><td>72.7</td><td>0.0125</td></tr><tr><td>Online VQ</td><td>256</td><td>1024</td><td>100.0%</td><td>759.3</td><td>28.08</td><td>74.0</td><td>0.0114</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>1024</td><td>100.0%</td><td>977.4</td><td>28.11</td><td>74.4</td><td>0.0112</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>2048</td><td>27.6%</td><td>453.0</td><td>27.78</td><td>73.8</td><td>0.0121</td></tr><tr><td>EMA VQ</td><td>256</td><td>2048</td><td>100%</td><td>1608.0</td><td>28.39</td><td>74.9</td><td>0.0107</td></tr><tr><td>Online VQ</td><td>256</td><td>2048</td><td>100%</td><td>1462.6</td><td>28.34</td><td>74.6</td><td>0.0108</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>2048</td><td>100%</td><td>1840.5</td><td>28.32</td><td>75.3</td><td>0.0107</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>4096</td><td>12.5%</td><td>435.0</td><td>27.84</td><td>73.7</td><td>0.0119</td></tr><tr><td>EMA VQ</td><td>256</td><td>4096</td><td>76.7%</td><td>2443.1</td><td>28.49</td><td>75.0</td><td>0.0104</td></tr><tr><td>Online VQ</td><td>256</td><td>4096</td><td>70.7%</td><td>1600.0</td><td>28.25</td><td>74.1</td><td>0.0110</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>4096</td><td>100%</td><td>3895.4</td><td>28.54</td><td>75.1</td><td>0.0102</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>8192</td><td>5.6%</td><td>398.1</td><td>27.69</td><td>73.5</td><td>0.0122</td></tr><tr><td>EMA VQ</td><td>256</td><td>8192</td><td>28.9%</td><td>1839.2</td><td>28.39</td><td>74.8</td><td>0.0106</td></tr><tr><td>Online VQ</td><td>256</td><td>8192</td><td>34.9%</td><td>1474.4</td><td>28.15</td><td>73.9</td><td>0.0113</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>8192</td><td>100%</td><td>7731.5</td><td>28.81</td><td>76.2</td><td>0.0099</td></tr></table>

Table 14: Analysis On codebook dimension by the comparison of VQ-VAEs trained on CIFAR-10 dataset following [49]. (The codebook size K is fixed to 16384)

<table><tr><td>Approaches</td><td>Tokens</td><td>Codebook Dim</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>Rec. Loss (↓)</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>2</td><td>3.8%</td><td>532.2</td><td>27.00</td><td>0.80</td><td>0.0162</td></tr><tr><td>EMA VQ</td><td>256</td><td>2</td><td>97.6%</td><td>14460.3</td><td>27.25</td><td>0.80</td><td>0.0155</td></tr><tr><td>Online VQ</td><td>256</td><td>2</td><td>9.0%</td><td>611.8</td><td>26.62</td><td>0.79</td><td>0.0178</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>2</td><td>99.3%</td><td>12278.9</td><td>27.30</td><td>0.81</td><td>0.0155</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>4</td><td>1.3%</td><td>176.7</td><td>27.15</td><td>0.83</td><td>0.0149</td></tr><tr><td>EMA VQ</td><td>256</td><td>4</td><td>99.8%</td><td>13153.9</td><td>29.57</td><td>0.89</td><td>0.0092</td></tr><tr><td>Online VQ</td><td>256</td><td>4</td><td>11.1%</td><td>877.7</td><td>26.69</td><td>0.79</td><td>0.0173</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>4</td><td>100.0%</td><td>15724.7</td><td>29.93</td><td>0.89</td><td>0.0087</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>8</td><td>1.6%</td><td>220.3</td><td>27.36</td><td>0.84</td><td>0.0141</td></tr><tr><td>EMA VQ</td><td>256</td><td>8</td><td>80.8%</td><td>10557.3</td><td>29.43</td><td>0.88</td><td>0.0009</td></tr><tr><td>Online VQ</td><td>256</td><td>8</td><td>13.4%</td><td>798.5</td><td>27.54</td><td>0.82</td><td>0.0141</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>8</td><td>100.0%</td><td>15583.7</td><td>30.19</td><td>0.90</td><td>0.0080</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>16</td><td>1.1%</td><td>150.8</td><td>27.05</td><td>0.83</td><td>0.0152</td></tr><tr><td>EMA VQ</td><td>256</td><td>16</td><td>32.5%</td><td>4169.2</td><td>29.31</td><td>0.88</td><td>0.0099</td></tr><tr><td>Online VQ</td><td>256</td><td>16</td><td>18.2%</td><td>2051.0</td><td>28.29</td><td>0.85</td><td>0.0122</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>16</td><td>99.2%</td><td>14832.2</td><td>30.27</td><td>0.91</td><td>0.0078</td></tr><tr><td>Vanilla VQ</td><td>256</td><td>32</td><td>0.7%</td><td>94.37</td><td>26.67</td><td>0.81</td><td>0.0165</td></tr><tr><td>EMA VQ</td><td>256</td><td>32</td><td>7.0%</td><td>942.7</td><td>28.24</td><td>0.85</td><td>0.0122</td></tr><tr><td>Online VQ</td><td>256</td><td>32</td><td>18.8%</td><td>2278.0</td><td>28.92</td><td>0.87</td><td>0.0104</td></tr><tr><td>Wasserstein VQ</td><td>256</td><td>32</td><td>96.4%</td><td>14056.9</td><td>30.39</td><td>0.91</td><td>0.0076</td></tr></table>

Analyses of Codebook Dimensionality We further investigate the impact of codebook dimensionality d on VQ performance. Conducting experiments on CIFAR-10 with dimensionality d ranging from 2 to 32, our proposed Wasserstein VQ consistently outperforms all baselines regardless of dimensionality, as shown in Table 14. Notably, we observe the curse of dimensionality phe nomenon—performance degrades as dimensionality increases. Vanilla VQ exhibits the most severe degradation, followed by EMA VQ and Online VQ, while our Wasserstein VQ shows only minimal codebook utilization reduction.

Table 15: Sensitivity analysis of $\gamma$ trained on FFHQ dataset following [49].

<table><tr><td>Approaches</td><td> $\gamma$ </td><td>Tokens</td><td>Codebook Size</td><td> $\mathcal{U} (\uparrow)$ </td><td> $\mathcal{C} (\uparrow)$ </td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>Rec. Loss (↓)</td></tr><tr><td>Wasserstein VQ</td><td>0.0</td><td>256</td><td>16384</td><td>3.8%</td><td>527.2</td><td>27.83</td><td>73.8</td><td>0.0119</td></tr><tr><td>Wasserstein VQ</td><td>0.00001</td><td>256</td><td>16384</td><td>24.8%</td><td>903.0</td><td>27.97</td><td>74.2</td><td>0.0114</td></tr><tr><td>Wasserstein VQ</td><td>0.0001</td><td>256</td><td>16384</td><td>52.3%</td><td>6764.3</td><td>28.70</td><td>75.5</td><td>0.0100</td></tr><tr><td>Wasserstein VQ</td><td>0.001</td><td>256</td><td>16384</td><td>90.6%</td><td>9988.0</td><td>28.93</td><td>76.7</td><td>0.0094</td></tr><tr><td>Wasserstein VQ</td><td>0.01</td><td>256</td><td>16384</td><td>100%</td><td>14952.5</td><td>29.06</td><td>76.7</td><td>0.0092</td></tr><tr><td>Wasserstein VQ</td><td>0.1</td><td>256</td><td>16384</td><td>100%</td><td>15943.5</td><td>29.07</td><td>76.7</td><td>0.0092</td></tr><tr><td>Wasserstein VQ</td><td>0.5</td><td>256</td><td>16384</td><td>100%</td><td>15713.3</td><td>29.03</td><td>76.6</td><td>0.0093</td></tr><tr><td>Wasserstein VQ</td><td>1.0</td><td>256</td><td>16384</td><td>100%</td><td>14712.4</td><td>29.02</td><td>76.9</td><td>0.0093</td></tr></table>

## K.3 Sensitivity Analysis on $\gamma$ .

We conduct a sensitivity analysis with respect to $\gamma$ on the FFHQ dataset by varying $\gamma\in\{0,10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},1\}$ . As reported in Table 15, when $\gamma=0$ , Wasserstein VQ yields the worst performance, as it degenerates into the vanilla VQ formulation without distributional matching. As $\gamma$ increases from $10^{-5}$ to $10^{-3}$ , the performance of Wasserstein VQ consistently improves. When $\gamma$ reaches $10^{-2}$ , Wasserstein VQ achieves full (100%) codebook utilization along with competitive quantitative results. Moreover, within the range $\gamma\in[10^{-2},1]$ , the performance of Wasserstein VQ remains stable, indicating that the method is not sensitive to the precise choice of $\gamma$ once it exceeds a moderate threshold.

## K.4 Computational Overhead Comparison among Various VQ Approaches

To evaluate the runtime efficiency of different VQ approaches, we measure the forward and backward pass times of the VQ module over 100 iterations across three different codebook sizes, with the feature dimension set to d = 8 and the number of data samples N = 8192. As shown in Table 4, even at large codebook sizes (specifically, $K \geq 50,000$ ), the runtime of Wasserstein VQ is only slightly longer than that of Vanilla VQ. This demonstrates that Wasserstein VQ maintains high computational efficiency, and incorporating the quadratic Wasserstein distance does not introduce significant time overhead. Notably, while Online VQ exhibits substantial runtime increases at large codebook sizes, Wasserstein VQ remains considerably more efficient, further underscoring its scalability.

## K.5 Discussion with VQ-WAE [50]

VQ-WAE [50] introduces an alternative approach to distributional matching by employing Optimal Transport to optimize codebook vectors. Compared with our proposed distributional matching method, there are three key differences.

First, regarding theoretical contributions: VQ-WAE $[50]$ claims that achieving optimal transport (OT) between code vectors and feature vectors yields the best reconstruction performance. Their notion of optimality encompasses both the VQ process and the encoder-decoder reconstruction pipeline. While we contend that incorporating complex encoder-decoder functions renders rigorous theoretical analysis intractable, VQ-WAE nevertheless asserts this conclusion. In contrast, our work deliberately excludes encoder-decoder components, focusing solely on the VQ process, which admits rigorous mathematical modeling. Through our proposed criterion triple, we theoretically prove that distributional matching guarantees optimal performance.

Second, regarding distribution modeling: VQ-WAE [50] assumes both code vectors and feature vectors follow uniform discrete distributions, whereas our method models them as continuous distributions. Specifically, VQ-WAE [50] represents the distributions of feature vectors $\{z_{i}\}_{i=1}^{N}$ and code vectors $\{e_{k}\}_{k=1}^{K}$ as empirical measures:

$$
\mathcal {P} _ {A} = \frac {1}{N} \sum_ {i = 1} ^ {N} \delta_ {\boldsymbol {z} _ {i}}, \quad \mathcal {P} _ {B} = \frac {1}{N} \sum_ {k = 1} ^ {K} \delta_ {\boldsymbol {e} _ {k}}\tag{35}
$$

Table 16: Reconstruction performance ( $\downarrow$ : the lower the better and $\uparrow$ : the higher the better). $\dagger$ : Results cited from VQ-WAE [50]. Codebook size K is fixed to 512.

<table><tr><td>Dataset</td><td>Model</td><td>Tokens</td><td>SSIM (↑)</td><td>PSNR (↑)</td><td>LPIPS (↓)</td><td>Rec. Loss (↓)</td><td>Perplexity (↑)</td></tr><tr><td rowspan="8">CIFAR10</td><td>VQ-VAE $^{\dagger}$ </td><td>64</td><td>70</td><td>23.14</td><td>0.35</td><td></td><td>69.8</td></tr><tr><td>SQ-VAE $^{\dagger}$ </td><td>64</td><td>80</td><td>26.11</td><td>0.23</td><td></td><td>434.8</td></tr><tr><td>VQ-WAE $^{\dagger}$ </td><td>64</td><td>80</td><td>25.93</td><td>0.23</td><td></td><td>497.3</td></tr><tr><td>VQ-WAE (Our run)</td><td>64</td><td>13</td><td>14.60</td><td>0.41</td><td>0.247</td><td>1.0</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>83</td><td>27.19</td><td>0.03</td><td>0.015</td><td>192.5</td></tr><tr><td>EMA VQ</td><td>64</td><td>84</td><td>27.97</td><td>0.04</td><td>0.013</td><td>436.1</td></tr><tr><td>Online VQ</td><td>64</td><td>84</td><td>27.87</td><td>0.04</td><td>0.013</td><td>451.4</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>86</td><td>28.26</td><td>0.03</td><td>0.012</td><td>481.7</td></tr><tr><td rowspan="8">SVHN</td><td>VQ-VAE $^{\dagger}$ </td><td>64</td><td>88</td><td>26.94</td><td>0.17</td><td></td><td>114.6</td></tr><tr><td>SQ-VAE $^{\dagger}$ </td><td>64</td><td>96</td><td>35.37</td><td>0.06</td><td></td><td>389.8</td></tr><tr><td>VQ-WAE $^{\dagger}$ </td><td>64</td><td>96</td><td>34.62</td><td>0.07</td><td></td><td>485.1</td></tr><tr><td>VQ-WAE (Our run)</td><td>64</td><td>25</td><td>15.87</td><td>0.26</td><td>0.2026</td><td>1.0</td></tr><tr><td>Vanilla VQ</td><td>64</td><td>97</td><td>38.18</td><td>0.01</td><td>0.0016</td><td>407.1</td></tr><tr><td>EMA VQ</td><td>64</td><td>97</td><td>38.35</td><td>0.01</td><td>0.0017</td><td>408.9</td></tr><tr><td>Online VQ</td><td>64</td><td>97</td><td>38.54</td><td>0.01</td><td>0.0017</td><td>421.5</td></tr><tr><td>Wasserstein VQ</td><td>64</td><td>97</td><td>38.25</td><td>0.01</td><td>0.0016</td><td>423.5</td></tr></table>

where $\delta_{z_{i}}$ and $\delta_{e_{k}}$ denote Dirac delta functions centered at $z_{i}$ and $e_{k}$ , respectively. To align $P_{A}$ and $P_{B}$ , VQ-WAE formulates the OT problem as:

$$
\begin{array}{l} \min _ {\mathbf {P} \in \Pi (\mathcal {P} _ {A}, \mathcal {P} _ {B})} \sum_ {i = 1} ^ {N} \sum_ {k = 1} ^ {K} P _ {i k} \| \boldsymbol {z} _ {i} - \boldsymbol {e} _ {k} \| ^ {2}, \\ \text {s.t.} \quad \mathbf {P 1} _ {K} = \frac {1}{N} \mathbf {1} _ {N}, \quad \mathbf {P} ^ {\top} \mathbf {1} _ {N} = \frac {1}{K} \mathbf {1} _ {K}, \quad P _ {i k} \geq 0 \quad \forall i, k, \end{array}\tag{36}
$$

where P is the transport plan, and the feasible set is:

$$
\Pi (\mathcal {P} _ {A}, \mathcal {P} _ {B}) = \left\{\mathbf {P} \in \mathbb {R} _ {+} ^ {N \times K}   \middle |   \mathbf {P 1} _ {K} = \frac {1}{N} \mathbf {1} _ {N},   \mathbf {P} ^ {\top} \mathbf {1} _ {N} = \frac {1}{K} \mathbf {1} _ {K} \right\}\tag{37}
$$

In contrast, we simplify the distributional assumption by modeling $P_{A}$ and $P_{B}$ as Gaussian distributions.

Third, regarding computational efficiency, The OT problem in VQ-WAE is prohibitively complex, whereas our quadratic Wasserstein distance incurs minimal overhead. To mitigate complexity, VQ-WAE employs a Kantorovich potential network. However, upon reproducing their code (no official implementation was released; we derived it from their ICLR 2023 supplementary material, $^{4}$ ), we observed severe non-convergence—the method degenerated to using a single code vector, failing to achieve distributional matching. Notably, VQ-WAE underperformed all other VQ baselines (Table 16).

In comparison, our quadratic Wasserstein distance (Equation 15) requires only low-dimensional matrix operations (e.g., d = 8), achieving superior performance and effective matching (Figure 7).