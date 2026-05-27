# Uncertainty-DTW for Sequences and Visual Tokens

Lei Wang, Syuan-Hao Li, Yongsheng Gao, Piotr Koniusz

Abstract—Aligning structured data is a fundamental problem in computer vision and machine learning, underlying tasks such as time series analysis, human action recognition, and visual representation learning. Existing alignment methods, including Dynamic Time Warping (DTW) and its differentiable variants, rely on deterministic similarity measures and are therefore sensitive to heterogeneous and noisy features. In this work, we introduce uncertainty-aware alignment, a probabilistic framework that models pairwise correspondences with heteroscedastic uncertainty and performs structured matching along alignment paths. Our formulation, uncertainty-DTW (uDTW), assigns each correspondence a Normal distribution and parametrizes each alignment path by a Maximum Likelihood Estimate objective consisting of (i) a precision-weighted matching term that suppresses unreliable features, and (ii) a log-variance regularization that prevents degenerate solutions. This yields a probabilistic alignment mechanism that is robust to noise and interpretable, as uncertainty directly reflects the reliability of matches. We further generalize this framework from temporal sequences to tokenized visual representations, enabling structured matching over sets of visual tokens. The learned uncertainty can be interpreted as a reverse-attention: semantically relevant regions exhibit low uncertainty and dominate the alignment, while ambiguous/noisy regions have high uncertainty. This provides a connection between alignment, attention, and uncertainty modeling. We evaluate the proposed framework across diverse domains, including time series forecasting, Frechet mean estimation, few- ´ shot action recognition, and few-shot image classification on generic, fine-grained, and ultra-fine-grained benchmarks. The results demonstrate consistent improvements over state-of-theart methods and show that learned uncertainty correlates with semantic importance. These findings establish uncertainty-aware alignment as a general, robust, and interpretable framework for learning from structured data.

Index Terms—Probabilistic modeling, few-shot learning, time series analysis, visual tokens

# I. INTRODUCTION

A LIGNING structured data is a fundamental problem incomputer vision and machine learning, underpinning a computer vision and machine learning,underpinning a wide range of tasks such as time series forecasting, action recognition, and visual representation learning [2]–[6]. A central challenge in these problems is to establish reliable correspondences between elements of structured inputs, whether they are temporal samples, human body joints, or visual tokens, despite variability in dynamics, appearance, and noise. Among existing approaches, Dynamic Time Warping (DTW) [2] and its differentiable variants [3], [7] have been widely

This work is supported in part by the Australian Research Council (ARC) under Industrial Transformation Research Hub Grant IH180100002 (Corresponding authors: Yongsheng Gao and Piotr Koniusz).

Lei Wang, Syuan-Hao Li, and Yongsheng Gao are with the School of Engineering and Built Environment, Electrical and Electronic Engineering, Griffith University (email: l.wang4@griffith.edu.au, syuan-hao.li@griffithuni.edu.au, yongsheng.gao@griffith.edu.au).

Piotr Koniusz is with School of Computer Science and Engineering, University of New South Wales (email: piotr.koniusz@unsw.edu.au).

![](images/3685b043a143a5d4a8afd2fc7321f613c9153d7e317e07b3c4e240d88cb27c52.jpg)

![](images/5078b3076161311dd35cd448e579ff9fe1c19a2be097f4746c5f11d231e0a691.jpg)

![](images/f37499ded69796523bb6d667bde0d267d9385ce1cfaf7e2c174bdbccea380df4.jpg)

![](images/511d5303b6996da253033833889899040aa34016e8dc6054b0ff3440349fa231.jpg)  
(a) Matched pair

![](images/5a82b5cedf61f86314b96faa639e140afe89a71561ceecb57528a12bad1c7140.jpg)

![](images/5d39a9927629d94a9033a23186e5f36f340c49b568cbe08a8626b3b20ad4f33a.jpg)

![](images/acab9b5467556afd86f5b9e5d78a95cd15bdcd2cb6fdd6ef8112332d84bbb3e9.jpg)

![](images/20caaeaa72edd7bca0b126442ad8972ceae77be2c9d203c1379b8f83ade84e7b.jpg)  
(b) Unmatched pair   
Fig. 1. A connection between alignment, attention, and uncertainty: (a) matched pair; (b) unmatched pair. In each group, the query (top right) and support (bottom left) images are overlaid with learned uncertainty (jet colormap: blue - low uncertainty, red - high uncertainty). The top left shows the DAAM attention map [1] (brighter = higher importance), and the bottom right shows uDTW alignment paths. Uncertainty exhibits a reverse-attention effect: low uncertainty regions align with high-attention areas and dominate the alignment, while high-uncertainty regions are suppressed. Selected patches (red/blue boxes) with zoomed-in alignment spaces further illustrate this behavior: matched pairs show actived alignment paths, especially in low uncertainty regions, whereas unmatched pairs exhibit weaker and less activated paths, indicating the absence of meaningful correspondence.

adopted due to their ability to compute optimal alignments under flexible structural constraints. However, these methods rely on deterministic similarity measures and select correspondences based solely on minimum distance, making them sensitive to heterogeneous feature quality and prone to being dominated by noisy or unreliable observations.

Our prior work [5] introduced uncertainty-DTW (uDTW), a probabilistic formulation of differentiable DTW that incorporates heteroscedastic aleatoric uncertainty into sequence alignment. By modeling each pairwise correspondence with a Gaussian distribution, uDTW reweights alignment costs according to feature reliability while regularizing uncertainty through a Maximum Likelihood Estimation (MLE) formulation over alignment paths. This leads to more robust alignments and improved performance on time series and skeletonbased sequence tasks. Nevertheless, this formulation was originally developed in the context of temporal sequences, and its broader implications for visual data alignment and representation learning remain underexplored.

In this work, we revisit uDTW from a broader perspective and generalize it into an uncertainty-aware alignment framework. We show that uDTW can be interpreted as a form of probabilistic structured matching, where alignment is governed by a log MLE function that jointly accounts for similarity and uncertainty. This perspective shows that uncertainty plays a central role in modulating correspondences: unreliable features are downweighted under high variance, while overconfident matches are penalized by the variance regularization term. As a result, the framework achieves robustness to heteroscedastic noise and provides interpretable measures of correspondence reliability. Furthermore, this formulation establishes conceptual connections between alignment, attention mechanisms, and optimal transport, positioning uDTW as a principled alternative for structured matching under uncertainty.

![](images/726140b8bea305dd524f3d7c65d508ad0ce92fe130d405d8181067861c47f333.jpg)

<details>
<summary>natural_image</summary>

Abstract white line on black background with no text or symbols
</details>

(a) sDTWγ=0.01

![](images/08ac65af4fac7ea57de978a5cc2030d64c3be9906a8827294799cdbdccc5a5a0.jpg)

<details>
<summary>natural_image</summary>

Abstract white curved line on black background, no text or symbols present
</details>

(b) sDTWγ=0.1

![](images/b760007fa0ae925c7bbc191ab5fec514ad31472a2ee25e9fb59d6cd3cf497e64.jpg)

<details>
<summary>natural_image</summary>

Abstract white line drawing on black background with no text or symbols
</details>

(c) uDTWγ=0.01

![](images/a4a60f3f76e639259b8af91ab970d4493bac15d1dc9afb17b9a28665fcc81264.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale gradient with flowing white streaks on black background (no text or symbols)
</details>

(d) uDTWγ=0.1

![](images/f6967058a6cc35a56276d309d8cb475ddb56163d2b0993ae02add826e0de4285.jpg)

<details>
<summary>bar</summary>

| Range | Counts |
|---|---|
| 0-2 | 10 |
| 2-4 | 8 |
| 4-6 | 7 |
| 6-8 | 6 |
| 8-10 | 5 |
| 10-12 | 4 |
| 12-14 | 3 |
| 14-16 | 2 |
</details>

(e) uDTW uncertainty   
Fig. 2. (a)-(d) Alignment paths of sDTW and uDTW (white) for a pair of sequences. Pixel intensities are power-normalized (exponent 0.1) to enhance low-visibility paths. Increasing the softness parameter γ ((b),(d)) activates multiple paths, producing a smoother, distributed alignment. Compared to sDTW (a), uDTW (c) exhibits multiple plausible routes due to uncertainty-aware weighting. (e) Visualization of the uncertainty matrix Σ. The binarized path from (c) is overlaid with Σ to reveal uncertainty along the alignment (white indicates high uncertainty). Elevated uncertainty in the central region explains the emergence and merging of alternative paths. The histogram summarizes the distribution of Σ values.

Building on this foundation, we extend uncertainty-aware alignment beyond temporal sequences to tokenized visual representations, such as patch tokens extracted from Vision Transformers (ViTs). Visual tokens form high-dimensional sets with complex semantic structure, where identifying meaningful correspondences is challenging and critical. We show that uDTW enables matching over such ordered token sets, and that the learned uncertainty exhibits a striking and consistent behavior: tokens corresponding to semantically relevant regions tend to have low uncertainty and dominate the alignment, while ambiguous or noisy regions are suppressed. This gives rise to a reverse-attention effect, in which uncertainty acts as a reliability-aware gating mechanism that emphasizes informative correspondences without requiring explicit attention modeling. This property provides a new lens for understanding alignment in modern visual representations. Fig. 1 shows how uDTW-based token-to-token alignment and learned uncertainty relate (inversely) to attention.

We validate the proposed framework across a diverse set of domains, including time series analysis, few-shot action recognition, and few-shot image classification. By unifying these tasks under a single alignment perspective, we demonstrate that uncertainty-aware alignment consistently improves performance while offering interpretable insights into the structure of the data. These results highlight the generality and effectiveness of modeling uncertainty in matching problems. This paper extends our earlier conference work [5] from a sequence-specific method to a general framework for ordered data alignment. The main contributions are:

i. We generalize uDTW into an uncertainty-aware alignment formulation that performs matching via alignment paths with heteroscedastic uncertainty, providing robustness to noise and an interpretable likelihood-based perspective.   
ii. We establish links between uncertainty-aware alignment, attention mechanisms, and optimal transport, and show that uncertainty is a reliability-aware matching that differs from deterministic similarity-based approaches.   
iii. We extend the framework from temporal sequences to tokenized visual representations, enabling alignment over ViT features, revealing a reverse-attention effect driven by

uncertainty learning.

iv. We conduct extensive experiments on time series, skeleton sequences, and image classification (generic, fine-grained, and ultra-fine-grained, under supervised and unsupervised few-shot settings), demonstrating consistent improvements over state-of-the-art methods and providing new insights into uncertainty and semantic importance.

# II. RELATED WORK

Below, we review the most closely related work and clearly highlight how our approach differs from prior methods.

DTW and differentiable alignment. DTW [2] is a classical method for aligning sequences under temporal distortions via dynamic programming. Its ability to compute optimal alignment between two sequences has made it a fundamental tool in time series analysis, speech processing, and action recognition. However, standard DTW relies on deterministic pairwise distances and selects a single minimum-cost path, making it sensitive to noise and local mismatches. To enable integration with modern deep learning frameworks, differentiable variant of DTW, called Soft-DTW (sDTW) [3] replaces the “hard minimum” calculations with a smooth approximation, enabling gradient-based optimization. Subsequent works [4], [6]–[9] have explored differentiable alignment layers via implicit differentiation and declarative formulations, as well as efficient approximations for large-scale applications. Despite these advances, existing DTW-based models use deterministic similarity calculations as they treat features at each timestep as being equally reliable, and lack mechanism to model uncertainty in pairwise correspondences. In contrast, our work introduces a probabilistic formulation of alignment, where each correspondence is associated with uncertainty, leading to uncertainty-aware matching objective that adaptively downweights unreliable features to form robust alignment.

Attention- and learning-based correspondence. Attention mechanisms [10]–[12] are the dominant paradigm for learning correspondences in sequences and visual representations. By computing similarity-based weights (e.g., via softmax), attention enables flexible, dense matching between elements and has been widely adopted in transformers and related architectures. Variants have also been explored for sequence alignment, integrating attention into warping or matching processes [13]–[15]. However, attention mechanisms are inherently similarity-centric: they assign weights based on relative similarity scores without explicitly modeling the reliability of features. As a result, attention may amplify spurious correspondences when similarity estimates are noisy or ambiguous.

Our approach differs fundamentally in both structure and mechanism. Firstly, instead of dense matching, we perform structured alignment via paths, preserving matching order constraints. Secondly, uncertainty is explicitly modeled to directly model matching costs. This results in a distinct behavior: unreliable correspondences are suppressed rather than amplified, giving rise to a reverse-attention effect, where semantically relevant regions dominate the alignment.

Optimal transport and structured matching. Optimal Transport (OT) provides a powerful framework for matching distributions by computing a global transport plan between two distributions [16]. Regularized OT, $e . g .$ , Sinkhorn [17] enable scalable and differentiable matching in a variety of vision and learning problems. Recent works [18], [19] have explored connections between OT and sequence alignment, including hybrid formulations that combine OT with temporal constraints. A key distinction lies in the structure of the matching. OT computes global correspondences without enforcing strict ordering constraints, whereas DTW enforces that matching indexes between two sequences always progress forward (or sometimes stay still) in each matching step, albeit at a variable rate. Thus, generic OT ignores this structural constraint inherent to sequentially ordered data. In contrast, uDTW performs structured matching under monotonic constraints on indexes while incorporating heteroscedastic uncertainty.

Robust alignment and uncertainty modeling. Robust alignment under noise and variability has been widely studied. Classical approaches introduce constraints such as warping windows or global penalties to regularize alignments [20]– [22]. More recent methods [13], [15], [23] explore parametric warping functions, diffeomorphic transformations, or learned similarity metrics to improve robustness. Despite these efforts, robustness is typically achieved implicitly through model design rather than by explicitly modeling uncertainty. Consequently, these methods do not provide a principled measure of matching reliability and remain sensitive to heteroscedastic noise. In contrast, our approach explicitly models aleatoric uncertainty at the level of pairwise correspondences. This results in a maximum-likelihood formulation with a precisionweighted matching term and a log-variance regularization term, yielding both robustness and interpretability. Uncertainty directly reflects the reliability of features and plays a central role in shaping the alignment.

Alignment for visual tokens. With the emergence of ViTs [10], [11], visual data is increasingly represented as sets of tokens. Most existing approaches compare such representations using global pooling, similarity aggregation, or attentionbased matching [24]–[27], often without explicit structural constraints. We extend alignment from temporal sequences to tokenized visual representations, treating tokens as structured entities amenable to alignment. Our framework enables path-based matching over ordered token sets while using uncertainty to improve correspondences. Notably, the learned uncertainty exhibits a consistent and interpretable pattern: semantically relevant regions are assigned low uncertainty and dominate the alignment, while ambiguous regions are suppressed, enabling a new perspective linking alignment, uncertainty, and semantic importance.

# III. METHOD: UNCERTAINTY-AWARE ALIGNMENT

# A. Problem Formulation

Notations. We consider the problem of aligning ordered sets of observations. Let $\pmb { X } = [ \pmb { x } _ { 1 } , \pmb { x } _ { 2 } , \dots , \pmb { x } _ { \tau } ] \in \mathbb { R } ^ { d \times \tau }$ and $\pmb { X } ^ { \prime } =$ $[ { \pmb x } _ { 1 } ^ { \prime } , { \pmb x } _ { 2 } ^ { \prime } , \ldots , { \pmb x } _ { \tau ^ { \prime } } ^ { \prime } ] \in \mathbb { R } ^ { d \times \tau ^ { \prime } }$ be two collections of lengths τ and $\tau ^ { \prime } ,$ , where $\pmb { x } _ { i } , \overset { \cdot } { \pmb { x } _ { j } ^ { \prime } } \in \mathbb { R } ^ { d }$ are feature vectors or tokens extracted from time series, skeleton joints, or visual patch embeddings.

A path Π is defined as a sequence of index pairs $( m , n )$ satisfying structural constraints, and $\boldsymbol { A } _ { \tau , \tau ^ { \prime } }$ denotes the set of all admissibnumber $\begin{array} { r } { D ( \tau - 1 , \tau ^ { \prime } - 1 ) = \sum _ { i = 0 } ^ { \operatorname* { m i n } ( \tau - 1 , \tau ^ { \prime } - 1 ) } \binom { \tau - 1 } { i } \binom { \tau ^ { \prime } - 1 } { i } 2 ^ { i } } \end{array}$ Its corresponding matrix form is denoted by $\dot { \bf \Pi } \dot { \bf \Pi } \in \dot { [ 0 , 1 ] } ^ { \tau \times \tau ^ { \prime } }$ , where $\Pi _ { m n }$ indicates the contribution of aligning ${ \pmb x } _ { m }$ with $ { \boldsymbol { { x } } } _ { n } ^ { \prime }$ . In the classical (hard) setting, Π is a sparse binary matrix encoding a valid path, whereas in the soft-relaxed formulation it is denser. The alignment task seeks an optimal alignment plan that minimizes a pairwise distance measure. Let $\pmb { D } \equiv$ $[ d _ { m n } ^ { 2 } ] _ { ( m , n ) \in \mathcal { T } _ { \tau } \times \mathcal { T } _ { \tau ^ { \prime } } } \in \mathbb { R } _ { + } ^ { \tau \times \tau ^ { \prime } }$ be the pairwise distance matrix with entries $\stackrel { \cdot } { d } _ { m n } ^ { 2 } = \lVert \pmb { x } _ { m } - \pmb { x } _ { n } ^ { \prime } \rVert _ { 2 } ^ { 2 }$ . A classical DTW then seeks:

$$
\boldsymbol {\Pi} ^ {*} = \underset {\boldsymbol {\Pi} \in \mathcal {A} _ {\tau , \tau^ {\prime}}} {\arg \min} \langle \boldsymbol {\Pi}, \boldsymbol {D} \rangle , \tag {1}
$$

where $\langle \cdot , \cdot \rangle$ denotes the matrix inner product, defined as $\langle \Pi , D \rangle \equiv \langle \mathrm { v e c } ( \Pi ) , \mathrm { v e c } ( D ) \rangle$ .

Motivation for probabilistic alignment. In practical applications, feature observations are often corrupted by heteroscedastic noise, missing data, or ambiguous correspondences (e.g., occluded joints, variable-speed actions, or visual patch ambiguities). Deterministic alignment in Eq. (1) cannot capture such uncertainty, often leading to suboptimal matches. To address this limitation, we introduce a probabilistic alignment framework, where each pairwise correspondence $( m , n )$ is associated with a variance $\sigma _ { m n } ^ { 2 } > 0$ quantifying its reliability. Collecting these variances defines an uncertainty matrix $\mathbf { \deltaSigma } ^ { \mathbf { \mathbf { \bullet } } }$ $[ \sigma _ { m n } ^ { 2 } ] _ { ( m , n ) \in \mathcal { T } _ { \tau } \times \mathcal { T } _ { \tau ^ { \prime } } } \in \mathbb { R } _ { + } ^ { \tau \times \tau ^ { \prime } }$ . We further define its elementτ  wise inverse (precision) as $\Sigma ^ { \dagger } \equiv \bigl [ \sigma _ { m n } ^ { - 2 } \bigr ] _ { ( m , n ) \in \mathcal { T } _ { \tau } \times \mathcal { T } _ { \tau ^ { \prime } } } .$ 1，

Under this formulation, correspondences with higher uncertainty contribute less to the alignment cost, reducing the impact of noise. Formally, the weighted structured alignment problem becomes:

$$
\boldsymbol {\Pi} ^ {*} = \underset {\boldsymbol {\Sigma} ^ {\dagger}, \boldsymbol {\Pi} \in \mathcal {A} _ {\tau , \tau^ {\prime}}} {\arg \min} \left\langle \boldsymbol {\Pi}, \boldsymbol {D} \odot \boldsymbol {\Sigma} ^ {\dagger} \right\rangle + \beta \left\langle \boldsymbol {\Pi}, \log \boldsymbol {\Sigma} \right\rangle , \tag {2}
$$

where ⊙ denotes element-wise multiplication, and $\beta ~ \geq ~ 0$ controls the strength of the uncertainty regularization.

The first term represents a precision-weighted matching cost, down-weighting unreliable correspondences, while the second term penalizes large uncertainty estimates. In the following, we derive a principled uncertainty-aware alignment operator (uDTW) from Eq. (2), along with its probabilistic interpretation and properties.

# B. Probabilistic Alignment with Heteroscedastic Uncertainty

Maximum likelihood path estimation. Building on the general alignment formulation in Eq. (2), we model each correspondence $( m , n )$ as a random variable with Gaussian uncertainty: $p ( { \pmb x } _ { m } \mid { \pmb x } _ { n } ^ { \prime } , \sigma _ { m n } ^ { 2 } ) = \mathcal { N } ( { \pmb x } _ { m } ; { \pmb x } _ { n } ^ { \prime } , \sigma _ { m n } ^ { 2 } )$ , where $\sigma _ { m n } ^ { 2 } > 0$ captures the heteroscedastic noise associated with matching ${ \pmb x } _ { m }$ to $ { \boldsymbol { { x } } } _ { n } ^ { \prime }$ . Consider an alignment path $\Pi _ { i }$ . Under the assumption of independence, the likelihood of the path is

$$
\begin{array}{l} \mathcal {L} (\boldsymbol {\Pi} _ {i}) = \prod_ {(m, n) \in \boldsymbol {\Pi} _ {i}} p (\boldsymbol {x} _ {m} \mid \boldsymbol {x} _ {n} ^ {\prime}, \sigma_ {m n} ^ {2}) \\ = \prod_ {(m, n) \in \boldsymbol {\Pi} _ {i}} \frac {1}{(2 \pi \sigma_ {m n} ^ {2}) ^ {d / 2}} \exp \left(- \frac {\| \boldsymbol {x} _ {m} - \boldsymbol {x} _ {n} ^ {\prime} \| _ {2} ^ {2}}{2 \sigma_ {m n} ^ {2}}\right). \tag {3} \\ \end{array}
$$

Maximizing the above likelihood is equivalent to minimizing log-likelihood:

$$
\boldsymbol {\Sigma} _ {i} ^ {*} = \min _ {\left\{\sigma_ {m n} \right\} (m, n) \in \boldsymbol {\Pi} _ {i}} \sum_ {(m, n) \in \boldsymbol {\Pi} _ {i}} \frac {\| \boldsymbol {x} _ {m} - \boldsymbol {x} _ {n} ^ {\prime} \| _ {2} ^ {2}}{\sigma_ {m n} ^ {2}} + d \log \sigma_ {m n} ^ {2}. \tag {4}
$$

where the first term penalizes large distances scaled by precision, and the second term regularizes against overestimating uncertainty. $\boldsymbol { \Sigma } _ { i } ^ { * }$ contains variances for path $\Pi _ { i }$ . Eq. (4) generalizes classical DTW by incorporating heteroscedastic uncertainty. If all $\sigma _ { m n } ^ { 2 } > 0$ mn  vary across correspondences, the alignment becomes $\sigma _ { m n } ^ { 2 } = 1$ , it reduces to standard DTW. When

Relation to uDTW. To connect the matrix-based formulation with individual alignment path $\Pi _ { i }$ with its path-specific cost directly derived from the objective in Eq. (2):

$$
\left\{ \begin{array}{l} d _ {\boldsymbol {\Pi} _ {i}} ^ {2} \equiv \left\langle \boldsymbol {\Pi} _ {i}, \boldsymbol {D} \odot \boldsymbol {\Sigma} ^ {\dagger} \right\rangle \\ \Omega_ {\boldsymbol {\Pi} _ {i}} \equiv \left\langle \boldsymbol {\Pi} _ {i}, \log \boldsymbol {\Sigma} \right\rangle . \end{array} \right. \tag {5}
$$

Here, $d _ { \Pi _ { i } } ^ { 2 }$ represents the precision-weighted matching cost along the path, and $\Omega _ { \Pi _ { i } }$ is the regularization term penalizing large uncertainty estimates on the path. The alignment path $\Pi _ { i }$ under this probabilistic framework has variance $\boldsymbol { \Sigma } _ { i } ^ { * }$ :

$$
\boldsymbol {\Sigma} _ {i} ^ {*} = \min _ {\boldsymbol {\Sigma}} d _ {\boldsymbol {\Pi} _ {i}} ^ {2} + \beta \Omega_ {\boldsymbol {\Pi} _ {i}}, \quad \beta \geq 0, \tag {6}
$$

which defines the uDTW objective for $\Pi _ { i }$ . This formulation explicitly shows connection with Eq. (3).

Parameterizing uncertainty. Directly optimizing the full uncertainty matrix Σ is impractical for variable-length sequences. Instead, we predict uncertainties using a small neural uncertainty network $\sigma ( \cdot ; \mathcal { P } _ { \sigma } )$ $\sigma _ { m n } ^ { 2 }$ can be computed by either (i) additive per- , called SigmaNet. In practice, each pairwise mtoken variances mndicted pairwise variance $\stackrel { \cdot } { \sigma } _ { m n } ^ { 2 } { = } 0 . 5 [ \sigma ^ { 2 } ( \stackrel { \cdot } { \pmb { x } } _ { m } ) + \sigma ^ { 2 } ( \stackrel { \cdot } { \pmb { x } } _ { n } ^ { \prime } ) ]$ $\sigma _ { m n } ^ { 2 } { = } \sigma ^ { 2 } ( { \pmb x } _ { m } , { \pmb x } _ { n } ^ { \prime } )$ ] or (ii) a jointly pre- . These predicted Eq. (4) for end-to-end learning of both feature encoder and uncertainty predictor.

Soft relaxation. To facilitate differentiable training, soft uDTW aggregates over all admissible alignment paths:

$$
\left\{ \begin{array}{l} d _ {\mathrm{uDTW}} ^ {2} = \operatorname{SoftMin} _ {\gamma} \left(\underbrace {\left[ \langle \boldsymbol {\Pi} , \boldsymbol {D} \odot \boldsymbol {\Sigma} ^ {\dagger} \rangle \right] _ {\boldsymbol {\Pi} \in \mathcal {A} _ {\tau , \tau^ {\prime}}}} _ {\boldsymbol {w}}\right), \\ \Omega = \operatorname{SoftMinSel} _ {\gamma} \left(\boldsymbol {w}, [ \langle \boldsymbol {\Pi}, \log \boldsymbol {\Sigma} \rangle ] _ {\boldsymbol {\Pi} \in \mathcal {A} _ {\tau , \tau^ {\prime}}}\right), \end{array} \right. \tag {7}
$$

$$
\text { where } \left\{ \begin{array}{l} \operatorname{SoftMin} _ {\gamma} (\mathbf {w}) = \sum_ {i} w _ {i} \frac {\exp (- (w _ {i} - \mu) / \gamma)}{\sum_ {j} \exp (- (w _ {j} - \mu) / \gamma)}, \\ \operatorname{SoftMinSel} _ {\gamma} (\mathbf {w}, \boldsymbol {u}) = \sum_ {i} u _ {i} \frac {\exp (- (w _ {i} - \mu) / \gamma)}{\sum_ {j} \exp (- (w _ {j} - \mu) / \gamma)}, \end{array} \right. \tag {8}
$$

with $\mu$ the mean of w for numerical stability and $\gamma > 0$ controlling the smoothness. The operator $\mathrm { S o f t M i n S e l } _ { \gamma }$ acts as a soft selector, returning the value associated with the minimal path when $\gamma \to 0$ . Both $d _ { \mathrm { u D T W } } ^ { 2 } ( D , \Sigma ^ { \dagger } )$ and Ω(Σ) depend on the input sequences, making them fully differentiable w.r.t. the token embeddings. The vector w contains the path-aggregated precision-weighted distances for all admissible paths, and Ω is the aggregation of log-variance penalties along these paths.

This formulation decouples the precision-weighted path distance $d _ { \mathrm { u D T W } } ^ { 2 }$ from the uncertainty regularization $\Omega ,$ , yielding a fully differentiable, uncertainty-aware alignment. During training, Ω penalizes the model for extreme uncertainty, while $d _ { \mathrm { u D T W } } ^ { 2 }$ captures the smallest matching distance between an two sets aligned. When $\gamma  0 ,$ , Ω reduces to the log-variance along the minimal-cost path.

# C. uDTW Properties and Theoretical Insights

The probabilistic alignment formulation of uDTW is not only practical but also admits several desirable properties. Fig. 2 shows how uncertainty affects uDTW.

Robustness to noise. By modeling heteroscedastic uncertainty σ mn , $\sigma _ { m n } ^ { 2 }$ uDTW down-weights contributions from noisy or unreliable tokens.

Proposition 1 (Alignment under additive noise). Let $[ { \pmb x } _ { 1 } +$ $\epsilon _ { 1 } , \ldots , x _ { \tau } + \epsilon _ { \tau } ]$ and $[ { \pmb x } _ { 1 } ^ { \prime } + \epsilon _ { 1 } ^ { \prime } , \ldots , { \pmb x } _ { \tau ^ { \prime } } ^ { \prime } + \epsilon _ { \tau ^ { \prime } } ^ { \prime } ]$ be sequences with corresponding additive i.i.d. Gaussian noises $\epsilon _ { m } \sim \mathcal { N } ( 0 , \gamma _ { m } ^ { 2 } )$ and $\epsilon _ { n } ^ { \prime } \sim \mathcal { N } ( 0 , \gamma _ { n } ^ { \prime 2 } )$ . Then, for a given alignment path $\mathbf { I I } _ { i } .$ σ2mn The uncertainty-weighted distance can reduce noises Eϵ,ϵ′ [d2Πi ] = P(m,n)∈Πi $\begin{array} { r } { \mathbb { E } _ { \epsilon , \epsilon ^ { \prime } } [ d _ { \Pi _ { i } } ^ { 2 } ] = \sum _ { ( m , n ) \in \Pi _ { i } } \frac { \| \pmb { x } _ { m } - \pmb { x } _ { n } ^ { \prime } \| _ { 2 } ^ { 2 } } { \sigma _ { m n } ^ { 2 } } + d \sum _ { ( m , n ) \in \Pi _ { i } } \frac { \gamma _ { m } ^ { 2 } + \gamma _ { n } ^ { \prime } ^ { \textrm { \tiny 2 } } } { \sigma _ { m n } ^ { 2 } } . } \end{array}$ + d P(m,n)∈Πi γ 2m +γ ′n 2 . $\gamma _ { m } ^ { 2 } + { \gamma _ { n } ^ { \prime } } ^ { 2 }$ and the noisy samples in the numerator by dividing them by $\sigma _ { m n } ^ { 2 } , e . g .$ ., additively factorized $\sigma _ { m n } ^ { 2 } { = } 0 . 5 \bigl [ \dot { \sigma } ^ { 2 } ( { \pmb x } _ { m } ) \dot { + } \sigma ^ { 2 } ( { \pmb x } _ { n } ^ { \prime } ) \bigr ]$ .

Proof. Expanding the squared norm and taking expectation: $\begin{array} { r } { \mathbb { E } _ { \epsilon _ { m } , \epsilon _ { n } ^ { \prime } } \left[ \| \bar { \boldsymbol { x } } _ { m } + \epsilon _ { m } - ( \boldsymbol { x } _ { n } ^ { \prime } + \epsilon _ { n } ^ { \prime } ) \| _ { 2 } ^ { 2 } \right] = \mathbb { E } _ { \epsilon _ { m n } ^ { * } } \left[ \| \mathbf { y } _ { m n } + \epsilon _ { m n } ^ { * } \| _ { 2 } ^ { 2 } \right] = } \end{array}$ $\lVert \mathbf { y } _ { m n } \rVert _ { 2 } ^ { \tilde { 2 } } + \mathbb { E } _ { \epsilon _ { m n } ^ { * } } [ \lVert \epsilon _ { m n } ^ { * } \rVert _ { 2 } ^ { 2 } ]$ , wh er e $\mathbf { y } _ { m n } = \pmb { x } _ { m } - \pmb { x } _ { n } ^ { \prime } , \pmb { \epsilon } _ { m n } ^ { * } =$ $\epsilon _ { m } { - } \epsilon _ { n } ^ { \prime }$ and $\epsilon _ { m n } ^ { * } \sim \mathcal { N } ( 0 , \gamma _ { m } ^ { 2 } + \gamma _ { n } ^ { \prime 2 } )$ . Summing over $( m , n ) \in \pi _ { i }$ and applying the uncertainty weighting $\sigma _ { m n } ^ { 2 }$ gives the expected uDTW cost. □

Remark 1. uDTW generalizes sDTW by explicitly accounting for the observation noise, providing a principled mechanism for robust sequence alignment.

Relation to attention & path-constrained transport. Let $\begin{array} { r l } { A _ { \tau , \tau ^ { \prime } } { = } \{ ( m _ { 1 } , n _ { 1 } ) , \ldots , ( m _ { L } , n _ { L } ) \colon ( m _ { 1 } , n _ { 1 } ) { = } ( 1 , 1 ) , ( m _ { L } , n _ { L } ) { = } 1 , } \end{array}$ $( \tau , \tau ^ { \prime } ) , \ \bar { ( } m _ { \ell + 1 } - m _ { \ell } , \ n _ { \ell + 1 } - n _ { \ell } ) \in \{ ( 1 , 0 ) , ( 0 , 1 ) , ( 1 , 1 ) \} , \ L \geq$ $1 \}$ be the set of all admissible alignment paths of lengths L. Each path $\Pi _ { i } \in { \mathcal A } _ { \tau , \tau ^ { \prime } }$ can also be represented as a binary matrix $\mathbf { I I } _ { i } \in \{ 0 , 1 \} ^ { \bar { \tau } \times \tau ^ { \prime } }$ satisfying monotonicity constraints $( m _ { \ell + 1 } { - } m _ { \ell } , n _ { \ell + 1 } { - } n _ { \ell } ) \in \left\{ ( 1 , 0 ) , ( 0 , 1 ) , ( 1 , 1 ) \right\} \mathrm { f o r } l = 1 , \ldots , L .$

For each path $\Pi _ { i : }$ define the precision-weighted path cost $\ d w _ { i } \equiv \langle { \bf { I I } } _ { i } , D \odot \Sigma ^ { \dagger } \rangle$ . The soft uDTW distance is given by ${ d _ { \mathrm { u D T W } } ^ { 2 } = \mathrm { S o f t M i n } _ { \gamma } ( \mathbf { w } ) = \sum _ { i } w _ { i } \cdot \pi ( \mathbf { \boldsymbol { \Pi } } _ { i } ) }$ ), where exp  −(wi−µ)/γ π(Πi) = $\begin{array} { r } { \pi ( \Pi _ { i } ) \ = \ \frac { \exp { \left\{ - ( w _ { i } - \mu ) / \gamma \right\} } } { \sum _ { j : \Pi _ { i } \in \mathcal { A } _ { \tau , \tau ^ { \prime } } } \exp { \left( - ( w _ { j } - \mu ) / \gamma \right) } } } \end{array}$ defines a Probability

Mass Function (PMF) over all admissible alignment paths, i.e., i=1 $i . e . , \sum _ { i = 1 } ^ { D ( \tau - 1 , \tau ^ { \prime } - 1 ) } \pi ( \mathbf { \Pi } \Pi _ { i } ) = 1$ PD(τ−1,τ′−1) π(Π ) = 1.

The same weighting π(Π) is used in $\mathrm { S o f t M i n S e l } _ { \gamma }$ in Eq. (8), ensuring that both the alignment cost and the uncertainty regularization are aggregated under the PMF of all admissible paths. This weighting induces a pairwise softcoupling matrix $\Pi ^ { \star } \ = \ [ \pi _ { m n } ^ { \star } ] _ { ( m , n ) \in \mathcal { T } _ { \tau } \times \mathcal { T } _ { \tau ^ { \prime } } }$ , defined as the probability that elements ${ \pmb x } _ { m }$ and $ { \boldsymbol { { x } } } _ { n } ^ { \prime }$ are aligned: $\pi _ { m n } ^ { \star } =$ $\begin{array} { r } { \sum _ { \Pi \in \mathcal { A } _ { \tau , \tau ^ { \prime } } : [ \Pi _ { i } ] _ { m n } = 1 } \pi ( \Pi ) } \end{array}$ .

Theorem 1 (Equivalence of path aggregation and pairwise coupling aggregation). Under the above definitions, the soft uDTW distance admits the equivalent representation $\begin{array} { r } { d _ { u D T W } ^ { 2 } ( \mathbf { X } , \mathbf { X } ^ { \prime } ) = \sum _ { m = 1 } ^ { \tau } \sum _ { n = 1 } ^ { \tau ^ { \prime } } \pi _ { m n } ^ { \star } \frac { \| \hat { x _ { m } } - x _ { n } ^ { \prime } \| _ { 2 } ^ { 2 } } { \sigma _ { m n } ^ { 2 } } . } \end{array}$ τ ′ .

Substituting  and exchang $\begin{array} { r } { w _ { i } = \sum _ { m , n } [ \mathbf { \boldsymbol { \Pi } } _ { i } ^ { \star } ] _ { m n } \frac { \| \mathbf { \boldsymbol { x } } _ { m } - \mathbf { \boldsymbol { x } } _ { n } ^ { \prime } \| _ { 2 } ^ { 2 } } { \sigma _ { m n } ^ { 2 } } } \end{array}$ $\sum _ { i } w _ { i }$ $\pi ( \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } _ { i } )$ $d _ { \mathrm { u D T W } } ^ { 2 } =$ Pm,n $\sum _ { m , n } \bigg ( \sum _ { i } \pi ( \boldsymbol { \Pi } _ { i } ^ { \setminus } ) [ \boldsymbol { \Pi } _ { i } ^ { \star } ] _ { m n } \bigg ) \frac { \| \boldsymbol { x } _ { m } - \boldsymbol { x } _ { n } ^ { \prime } \| _ { 2 } ^ { 2 } } { \sigma _ { m n } ^ { 2 } }$ where the inner summation equals $\begin{array} { c } { = \pi _ { { m n } } ^ { \star } } \\ { \pi _ { { m n } } ^ { \star } \mathrm { \Delta } \Psi } \end{array}$ {z=π⋆mn definition, completing the proof.

The coupling $\pi _ { m n } ^ { * }$ can be interpreted as a structured attention map. Unlike conventional attention, which normalizes local similarity scores, each $\pi _ { m n } ^ { * }$ aggregates contributions over all admissible alignment paths, reserving matching order constraint. Furthermore, the precision weighting $\pmb { \Sigma } ^ { \dagger }$ modulates contributions across all paths, ensuring that uncertain correspondences are systematically down-weighted.

The same coupling induces a transport-like objective in which $\pi _ { m n } ^ { * }$ acts as a soft assignment between the two sequences. In contrast to classical optimal transport, the coupling here is restricted to those induced by distributions over admissible alignment paths, thereby encoding structural constraints such as monotonicity and continuity.

Connection to probabilistic alignment. The soft uDTW formulation admits a probabilistic interpretation in which alignment is modeled as a latent variable distributed over the space of admissible paths. The soft uDTW distance corresponds to the expected alignment cost $d _ { \mathrm { u D T W } } ^ { 2 }$ . Unlike deterministic alignment methods that select a single optimal path, our formulation maintains a distribution over all feasible paths, allowing the model to represent ambiguity in the alignment. A key property of this formulation is that the same path distribution $\pi ( \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } _ { i } )$ is used consistently across both the alignment objective and the uncertainty regularization. In particular, the uncertainty term $\begin{array} { r } { \Omega = \mathrm { S o f t M i n S e l } _ { \gamma } ( \mathbf { w } , \mathbf { u } ) = \sum _ { i } u _ { i } \cdot \pi ( \mathbf { \Pi } \mathbf { \Pi } _ { i } ) } \end{array}$ , aggregates pathwise log-variance quantities $u _ { i } = \langle \mathbf { I } \mathbf { I } _ { i } , \log \Sigma \rangle$ under the same distribution. This shared weighting ensures that the model jointly evaluates alignment quality and uncertainty under a common probabilistic structure, enabling coherent propagation of uncertainty through the alignment process. The temperature parameter $\gamma \mathrm { ~ > ~ } 0$ controls the concentration of the path distribution. This behavior is formalized below.

Theorem 2 (Deterministic limit of uDTW). Let $\pi ( \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } _ { i } )$ be defined as above, and let $\Pi ^ { * } = \arg \operatorname* { m i n } _ { \Pi _ { i } \in A _ { \tau , \tau ^ { \prime } } } w _ { i }$ denote an optimal alignment path. Then, as $\gamma  0 ,$ the distribution $\pi ( \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } \mathbf { \boldsymbol { \Pi } } ^ { } )$ converges to a Dirac distribution concentrated on Π∗, and $d _ { u D T W } ^ { 2 } \to w ^ { * } , \Omega \to u ^ { * }$ , where $w ^ { \ast } = w ( \mathbf { I } ^ { * } )$ and $u ^ { * } = u ( \mathbf { I I ^ { * } } )$ . Proof. For any $\mathbf { I I } _ { i } \ \neq \ \mathbf { I I ^ { * } }$ , we have $w _ { i } \mathrm { ~  ~ { ~ > ~ } ~ } w ^ { * }$ . Hence $\begin{array} { r } { \frac { \pi ( \dot { \Pi _ { i } } ) } { \pi ( \dot { \Pi ^ { * } } ) } = \exp \Bigl ( - \frac { w _ { i } - w ^ { * } } { \gamma } \Bigr ) \to 0 } \end{array}$ $\gamma \to 0$ . Thus nverge $\pi ( \mathbf { I I ^ { * } } ) \to 1$ $\pi ( \mathbf { { H } } _ { i } )  \dot { 0 }$ $\mathbf { \widetilde { I I } } _ { i } \neq \mathbf { I I } ^ { * }$ $d _ { \mathrm { u D T W } } ^ { 2 }$ and Ω follows directly from their definitions as expectations under π.

This result shows that uDTW provides a continuous relaxation of classical alignment, interpolating between deterministic path selection and distributed alignment over multiple paths. For finite γ, the model captures uncertainty in the alignment itself, while for $\gamma  0$ , it recovers the classical uDTW formulation. Crucially, the incorporation of heteroscedastic precision $\pmb { \Sigma } ^ { \dagger }$ ensures that this uncertainty is data-dependent, allowing unreliable correspondences to be systematically attenuated across all plausible alignments.

This probabilistic alignment view provides a formal latentvariable interpretation of the path distribution underlying the structured coupling presented previously.

# D. Extension to Tokenized Visual Representations

Although uDTW was originally formulated for temporal sequences, its probabilistic, uncertainty-aware framework extends naturally to tokenized visual representations, such as patch embeddings extracted from ViTs. Each token represents a local image patch (e.g., 16×16 pixels) and may include positional encoding. Visual tokens form a structured 2D grid, but weak ordering does not impede uDTW’s structured alignment. Unlike conventional attention mechanisms, uDTW incorporates uncertainty directly into the alignment cost, producing a reliability-aware mechanism that emphasizes informative tokens without requiring an explicit attention module.

Uncertainty-driven reverse attention. Let $[ { \pmb x } _ { m } ] _ { m \in \pmb { \mathbb { Z } } _ { T } }$ and $[ { \pmb x } _ { n } ^ { \prime } ] _ { n \in \mathbb { Z } _ { T } }$ denote the query and support token sets, arranged√ in $\ddot { \textbf { a } } \sqrt { T } \times \sqrt { T }$ token grid. The element-wise uncertainty modulates the contribution of each token pair in the alignment, reflecting its reliability. Using the soft uDTW formulation, the alignment probability of tokens m and n is given by $\pi _ { m n } ^ { * }$ The resulting matrix d, uncertainty-aware $[ \pi _ { m n } ^ { * } ] _ { ( m , n ) \in \mathcal { T } _ { T } \times \mathcal { T } _ { T } }$ defines a struc-en pairs, where low-uncertainty correspondences are automatically assigned higher alignment probability, producing a reliability-weighted, attention-like mapping across the token grid.

The critical insight is that the uncertainty directly modulates have higher precision the effective attention over token pairs. Low-uncertainty pairs $\sigma _ { m n } ^ { - 2 }$ , resulting in larger contributions to $d _ { \mathbf { I } _ { 1 } } ^ { 2 }$ i and, consequently, larger probabilities $\pi _ { m n } .$ . Conversely, high-uncertainty pairs contribute less to the alignment and receive lower $\pi _ { m n }$ . In other words, the learned uncertainty acts as a reliability-aware gating signal, producing a reverseattention effect: tokens with low uncertainty dominate the alignment, whereas ambiguous or unreliable tokens are systematically suppressed. Unlike standard attentions, this effect emerges directly from the probabilistic alignment formulation, without any additional attention module. Through this extension, uDTW provides a principled, unified framework for structured alignment over tokenized visual representations.

Semantic structure emergence in visual tokens. To better understand the effect of uncertainty-aware alignment on visual representations, we visualize token embeddings before and after the learned uDTW projection. Specifically, we compare raw DINOv3 patch-token embeddings with embeddings transformed by the projection layer trained using the uDTW objective. Token embeddings are visualized by mapping the first three PCA components of the feature space into RGB channels. Fig. 3 shows that the original DINOv3 embeddings capture coarse semantic information but often exhibit diffuse activations and weak structural consistency. In contrast, embeddings learned with uDTW become more compact and object-centric, with clearer foreground regions and suppressed background responses. This effect is particularly pronounced on ultra-fine-grained leaf datasets, where the projected embeddings preserve subtle structural characteristics such as contours, defects, and even vein patterns. These observations suggest that uDTW does not merely reweight pairwise similarities, but instead induces a reliability-aware token geometry that enhances structurally consistent and semantically meaningful correspondences. Moreover, the projected embeddings exhibit stronger spatial continuity across neighboring regions. Object silhouettes and elongated structures become noticeably more coherent after the uDTW projection. This behavior is consistent with the probabilistic alignment formulation, where lowuncertainty correspondences contribute more strongly across admissible alignment paths, while ambiguous or unreliable matches are systematically attenuated.

![](images/c63fc61b27dff290a620dfdf442ec6782019a1f64bfe5b6024a94f21ab81f606.jpg)

Fig. 3. Visualization of visual-token embeddings before and after the learned uDTW projection. Token embeddings are visualized by mapping the first three PCA components into RGB channels. Compared to the original DINOv3 embeddings, the uDTW-projected embeddings become more compact and object-centric, with clearer foreground-background separation and stronger structural continuity. On ultra-fine-grained leaf images, the projected embeddings preserve subtle structures such as contours, defects, and vein patterns, suggesting that uDTW induces a reliability-aware semantic token geometry.   
![](images/fe0d62d5ee41129fd773c8b0eec7da1e4ebdb7efe4765335dd0531a009631e0c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["MLP"] --> B["PG → Tanh → FO"]
    B --> C["SigmaNet"]
    C --> D["PG → Scaled Sigmoid"]
    D --> E["uDTW"]
    E --> F["ψ is ψ similar to test x?"]
    G["ψ' x'"] --> H["Forecasting the evolution of time series + uDTW"]
    H --> I["↑"]
    I --> J["↑"]
    J --> K["→"]
    K --> L["ψ"]
```
</details>

(a) Time series forecasting.

![](images/2dc0d59977da331e04bee83560dd7dd66b7b0cc406b96030cb08f65a058c92ba.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["query X₁ ... Xᵣ"] --> B["MLP"]
    C["support X′ ... Xᵣ"] --> B
    D["FC"] --> E["GNN"]
    E --> F["Transformer (optional)"]
    F --> G["FC"]
    H["FC"] --> I["ReLU"]
    I --> J["FC"]
    K["ReLU"] --> L["ReLU"]
    M["SigmaNet"] --> N["Sigma"]
    O["FC"] --> P["Scaled Sigmoid"]
    Q["Supervised Comparator with uDTW"] --> R["d_uDTW(D, Σ)"]
    R --> S["σ(Σ)"]
    S --> T["similarity"]
    S --> U["classification i.e."]
    S --> V["Σₙ ℓ(dᵤDTW,n, yₙ) + βΩₙ,"]
```
</details>

(b) Supervised FSAR.

![](images/e557e69dfd893533ac367efe7aafd0af121be076448e483d4db4d6395941652f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Unsupervised Comparator with DL+uDTW"] --> B["SigmaNet"]
    B --> C["LoSA"]
    C --> D["DL"]
    D --> E["Ω(Σ)"]
    E --> F["LeSA"]
    F --> G["NN"]
    H["Ψ↑"] --> I["σ"]
    I --> J["Γ"]
    J --> K["PI"]
    K --> L["Scaled Gaussian"]
    L --> M["Γ"]
    M --> N["Σ"]
    N --> O["Γ"]
    O --> P["Γ"]
    P --> Q["Γ"]
    Q --> R["Γ"]
    R --> S["Γ"]
    S --> T["Γ"]
    T --> U["Γ"]
    U --> V["Γ"]
    V --> W["Γ"]
    W --> X["Γ"]
    X --> Y["Γ"]
    Y --> Z["Γ"]
    Z --> AA["Γ"]
    AA --> AB["Γ"]
    AB --> AC["Γ"]
    AC --> AD["Γ"]
    AD --> AE["Γ"]
    AE --> AF["Γ"]
    AF --> AG["Γ"]
    AG --> AH["Γ"]
    AH --> AI["Γ"]
    AI --> AJ["Γ"]
    AJ --> AK["Γ"]
    AK --> AL["Γ"]
    AL --> AM["Γ"]
    AM --> AN["Γ"]
    AN --> AO["Γ"]
    AO --> AP["Γ"]
    AP --> AQ["Γ"]
    AQ --> AR["Γ"]
    AR --> AS["Γ"]
    AS --> AT["Γ"]
    AT --> AU["Γ"]
    AU --> AV["Γ"]
    AV --> AW["Γ"]
    AW --> AX["Γ"]
    AX --> AY["Γ"]
    AY --> AZ["Γ"]
    AZ --> BA["Γ"]
    BA --> BB["Γ"]
    BB --> BC["Γ"]
    BC --> BD["Γ"]
    BD --> BE["Γ"]
    BE --> BF["Γ"]
    BF --> BG["Γ"]
    BG --> BH["Γ"]
    BH --> BI["Γ"]
    BI --> BJ["Γ"]
    BJ --> BK["Γ"]
    BK --> BL["Γ"]
    BL --> BM["Γ"]
    BM --> BN["Γ"]
    BN --> BO["Γ"]
    BO --> BP["Γ"]
    BP --> BQ["Γ"]
    BQ --> BR["Γ"]
    BR --> BS["Γ"]
    BS --> BT["Γ"]
    BT --> BU["Γ"]
    BU --> BV["Γ"]
    BV --> BW["Γ"]
    BW --> BX["Γ"]
    BX --> BY["Γ"]
    BY --> BZ["Γ"]
    BZ --> CA["Γ"]
    CA --> CB["Γ"]
    CB --> CC["Γ"]
    CC --> CD["Γ"]
    CD --> CE["Γ"]
    CE --> CF["Γ"]
    CF --> CG["Γ"]
    CG --> CH["Γ"]
    CH --> CI["Γ"]
    CI --> CJ["Γ"]
    CJ --> CK["Γ"]
    CK --> CL["Γ"]
    CL --> CM["Γ"]
    CM --> CN["Γ"]
    CN --> CO["Γ"]
    CO --> CP["Γ"]
    CP --> CQ["Γ"]
    CQ --> CR["Γ"]
    CR --> CS["Γ"]
    CS --> CT["Γ"]
    CT --> CU["Γ"]
    CU --> CV["Γ"]
    CV --> CW["Γ"]
    CW --> CX["Γ"]
    CX --> CY["Γ"]
    CY --> CZ["Γ"]
```
</details>

(c) Unsupervised FSAR.   
Fig. 4. Uncertainty-aware alignment across tasks. (a) Time series forecasting: predicted sequences are aligned with ground truth using uDTW with learned uncertainty. (b) Supervised few-shot action recognition: support and query sequences are encoded into temporal embeddings and matched via uncertainty-aware alignment. (c) Unsupervised setting: the same encoder is retained, while alignment is optimized with a label-free objective.

# IV. LEARNING FRAMEWORK FOR STRUCTURED INPUTS

Below, we present three representative pipeline formulations where uDTW acts as a core component for computing uncertainty-aware alignment distances over warped paths.

# A. Time Series Forecasting and Classification

Encoder. Given an observed prefix $\mathbf { x } \in \mathbb { R } ^ { t }$ of a time series, the encoder $f ( \cdot ; \mathcal { P } )$ (following [3]) maps it to a predicted future ψ $\in \mathbb { R } ^ { \tau - t }$ using an MLP, where $\mathcal { P } \equiv [ \mathcal { P } _ { \mathrm { M L P } } , \mathcal { P } _ { \mathrm { S N } } ]$ denotes the parameters of the encoder and SigmaNet. Here, $\pmb { x } ^ { \prime } \in \mathbb { R } ^ { \tau - t }$ and $\psi \in \mathbb { R } ^ { \tau - t }$ are vectors representing the ground-truth future and predicted future of the time series, respectively. SigmaNet predicts per-timestep positive uncertainties for both ψ and $\mathbf { { x } ^ { \prime } } .$ Forecasting. The model is trained to minimize the uDTWbased forecasting loss across N training series (see Fig. 4a):

$$
\mathcal {L} _ {\text { forecast }} = \sum_ {n \in \mathcal {I} _ {N}} d _ {\mathrm{uDTW}} ^ {2} (\boldsymbol {\psi} _ {n}, \boldsymbol {x} _ {n} ^ {\prime}) + \beta \Omega (\boldsymbol {\psi} _ {n}, \boldsymbol {x} _ {n} ^ {\prime}), \tag {9}
$$

where $d _ { \mathrm { u D T W } } ^ { 2 }$ measures temporal alignment and Ω incorporates predictive uncertainty.

Classification. Following [3], we adopt the standard setup for this classical task. For each class c, the class prototype $\pmb { \mu _ { c } } \in$ $\mathbb { R } ^ { \tau ^ { \prime } }$ is estimated as the Frechet mean of all training sequences ´ in that class:

$$
\boldsymbol {\mu} _ {c} = \underset {\boldsymbol {\mu}} {\arg \min} \sum_ {n \in \mathcal {I} _ {N _ {c}}} d _ {\mathrm{uDTW}} ^ {2} (\boldsymbol {x} _ {n}, \boldsymbol {\mu}) + \beta \Omega (\boldsymbol {x} _ {n}, \boldsymbol {\mu}), \tag {10}
$$

where $\tau ^ { \prime }$ is set to the average sequence length across all classes. SigmaNet is trained jointly to predict per-timestep uncertainties for each prototype. In addition to the nearest centroid, we also consider the use of nearest neighbors. Each test sequence x is directly compared to all training sequences using uDTW, assigning the label of the closest sequence.

Inference. Inference follows the task-specific setting. (i) Forecasting: For a new prefix x, the prediction is $\psi ~ =$ $f ( \pmb { x } ; \mathcal { P } )$ . (ii) Classification: Let C denote the set of classes and $\{ \pmb { x } _ { n } \} _ { n = 1 } ^ { N }$ 1V the training set. For nearest centroid, $\hat { y } \ =$ arg m $\begin{array} { r } { \mathrm { i n } _ { c \in \mathcal { C } } d _ { \mathrm { u D T W } } ^ { 2 } ( \pmb { x } , \pmb { \mu } _ { c } ) + \beta \Omega ( \pmb { x } , \pmb { \mu } _ { c } ) } \end{array}$ , and for nearest neighbor, $\begin{array} { r } { \hat { y } = \arg \operatorname* { m i n } _ { n \in \{ 1 , . . . , N \} } d _ { \mathrm { u D T W } } ^ { 2 } ( \pmb { x } , \pmb { x } _ { n } ) + \beta \Omega ( \pmb { x } , \pmb { x } _ { n } ) } \end{array}$ , with uncertainties incorporated via SigmaNet.

# B. Few-shot Skeleton-based Action Recognition

Encoder. Each skeleton sequence with J joints per frame is partitioned into τ temporal blocks of length M with stride S. Each block $\pmb { X } \in \mathbb { R } ^ { 3 \times J \times M }$ contains 3D joint coordinates. Each block is encoded by a 3-layer MLP (FC, ReLU, FC, ReLU, Dropout, FC) producing a $d { \times } J$ feature map, followed by an $\mathrm { S ^ { 2 } G C }$ [28] for spatial modeling and a Transformer encoder for temporal interactions. A final FC produces the block embedding (see Fig. 4b). Formally, for query and support sequences: $\begin{array} { r } { \Psi = f ( \boldsymbol { X } ; \bar { \mathcal { P } } ) \in \mathbb { R } ^ { d ^ { \prime } \times \tau } , \ \bar { \Psi ^ { \prime } } = f ( \boldsymbol { X } ^ { \prime } ; \mathcal { P } ) \in \mathbb { R } ^ { \bar { d ^ { \prime } } \times \tau ^ { \prime } } } \end{array}$ , where the Supervised Comparator $\mathcal { P } = [ \mathcal { P } _ { \mathrm { M L P } } , \mathcal { P } _ { \mathrm { S ^ { 2 } G C } } , \mathcal { P } _ { \mathrm { T r a n s f } } , \mathcal { P } _ { \mathrm { F C } } , \mathcal { P } _ { \mathrm { S N } } ]$ includes the parameters of MLP, $\mathrm { { S ^ { 2 } G C } }$ , Transformer, FC, and SigmaNet. SigmaNet predicts per-block scalar uncertainties.

Supervised. For N-way Z-shot episodes with B episodes per batch, each query embedding is randomly sampled from one of the N classes, and support embeddings contain Z sequences per class. Similarity labels are set as $\delta _ { 1 } = 0$ for matching classes and $\delta _ { n \neq 1 } = 1$ for mismatches. The supervised uDTW loss is

$$
\mathcal {L} _ {\sup} = \sum_ {b \in \mathcal {I} _ {B}} \sum_ {n \in \mathcal {I} _ {N}} \sum_ {z \in \mathcal {I} _ {Z}} \left(d _ {\mathrm{uDTW}} ^ {2} (\boldsymbol {\Psi} _ {b}, \boldsymbol {\Psi} _ {b, n, z} ^ {\prime}) - \delta_ {n}\right) ^ {2} + \beta \Omega (\boldsymbol {\Psi} _ {b}, \boldsymbol {\Psi} _ {b, n, z} ^ {\prime}). \tag {11}
$$

Unsupervised. In the unsupervised setting, class labels are ignored. Query and support sequences in each episode are concatenateda dictionary $\{ M _ { k } \} _ { k = 1 } ^ { K }$ $\{ \Psi _ { b , n } ^ { \ddagger } \} _ { n = 1 } ^ { N \cdot Z + 1 }$ ‡b,n}N·Z+1n=1 . Sequences are projected onto CSA [29], with coefficients

$$
\alpha_ {k, b, n} = \left\{ \begin{array}{l l} \frac {\exp \left(- \frac {1}{\gamma^ {\prime}} d _ {\mathrm{uDTW}} ^ {2} \left(\boldsymbol {\Psi} _ {b , n} ^ {\dagger} , M _ {k}\right)\right)}{\sum_ {l \in \mathcal {M} \left(\boldsymbol {\Psi} _ {b , n} ^ {\dagger} ; K ^ {\prime}\right)} \exp \left(- \frac {1}{\gamma^ {\prime}} d _ {\mathrm{uDTW}} ^ {2} \left(\boldsymbol {\Psi} _ {b , n} ^ {\dagger} , M _ {l}\right)\right)} & \text { if } M _ {k} \in \mathcal {M} \left(\boldsymbol {\Psi} _ {b, n} ^ {\dagger}; K ^ {\prime}\right), \\ 0 & \text { otherwise }, \end{array} \right. \tag {12}
$$

where $K ^ { \prime } \leq K$ is the number of nearest dictionary atoms, $\mathcal { M } ( \cdot )$ retrieves the nearest atoms using uDTW, $\tau ^ { \prime }$ is set to the mean block count across the training set, and $\gamma ^ { \prime } = 0 . 7 $ . Dictionary atoms are updated via

$$
\boldsymbol {M} _ {k} \leftarrow \boldsymbol {M} _ {k} - \lambda_ {\mathrm{DL}} \sum_ {n = 1} ^ {N \cdot Z + 1} \nabla_ {\boldsymbol {M} _ {k}} d _ {\mathrm{uDTW}} ^ {2} \left(\boldsymbol {\Psi} _ {b, n} ^ {\ddagger}, \sum_ {l = 1} ^ {K} \alpha_ {l, b, n} \boldsymbol {M} _ {l}\right), \tag {13}
$$

with dict\_iter = 10 and $\lambda _ { \mathrm { D L } } ~ = ~ 0 . 0 0 1$ . Unsupervised Comparator (see Fig. 4c) is updated correspondingly:

$$
\mathcal {P} \leftarrow \mathcal {P} - \lambda_ {\mathrm{EN}} \sum_ {n = 1} ^ {N Z + 1} \nabla_ {\mathcal {P}} d _ {\mathrm{uDTW}} ^ {2} \left(\boldsymbol {\Psi} _ {b, n} ^ {\ddagger}, \boldsymbol {M} ^ {\prime}\right) + \beta \Omega \left(\boldsymbol {\Psi} _ {b, n} ^ {\ddagger}, \boldsymbol {M} ^ {\prime}\right), \tag {14}
$$

where $\begin{array} { r } { M ^ { \prime } { = } \sum _ { l { = } 1 } ^ { K } \alpha _ { l , b , n } M _ { l } } \end{array}$ , with $\lambda _ { \mathrm { E N } } = 0 . 0 0 1$

Inference. Query embeddings are matched to support embeddings using uDTW in the supervised setting. In the unsupervised setting, both query and support sequences are encoded into LCSA coefficients and compared using histogram intersection kernel, which measures similarity between their soft assignment distributions. The predicted label is assigned to the support sequence with the highest similarity score.

![](images/2e7e23edea6465a07db420af31180769ef60bef3ad5be247e0fdd37a142a4be2.jpg)

<details>
<summary>line</summary>

| x  | Eucl. | sdTW | β=10.0 | β=1.0 | β=0.1 |
|----|-------|------|--------|-------|-------|
| 0  | -1.5  | -1.5 | -1.5   | -1.5  | -1.5  |
| 20 | -0.5  | -0.5 | -0.5   | -0.5  | -0.5  |
| 40 | 0.5   | 0.5  | 0.5    | 0.5   | 0.5   |
| 60 | 1.5   | 1.5  | 1.5    | 1.5   | 1.5   |
| 80 | 2.0   | 2.0  | 2.0    | 2.0   | 2.0   |
| 100| 1.0   | 1.0  | 1.0    | 1.0   | 1.0   |
| 120| 0.5   | 0.5  | 0.5    | 0.5   | 0.5   |
| 140| -1.0  | -1.0 | -1.0   | -1.0  | -1.0  |
</details>

(a) β (where λ = 1)

![](images/3e11e63d7d95427eac7e049ecd206ad1869b61deb46e53a2bb5ef202c15966fa.jpg)

<details>
<summary>line</summary>

| x    | Eucl. | sDTW | λ=10.0 | λ=1.0 | λ=0.1 |
| ---- | ----- | ---- | ------ | ----- | ----- |
| 0    | -1.5  | -1.5 | -1.5   | -1.5  | -1.5  |
| 20   | -1.0  | -1.0 | -1.0   | -1.0  | -1.0  |
| 40   | 0.5   | 0.5  | 0.5    | 0.5   | 0.5   |
| 60   | 1.5   | 1.5  | 1.5    | 1.5   | 1.5   |
| 80   | 2.0   | 2.0  | 2.0    | 2.0   | 2.0   |
| 100  | 1.0   | 1.0  | 1.0    | 1.0   | 1.0   |
| 120  | 0.5   | 0.5  | 0.5    | 0.5   | 0.5   |
| 140  | -1.0  | -1.0 | -1.0   | -1.0  | -1.0  |
</details>

(b) λ (where β = 10)   
Fig. 5. Interpolation between two time series (grey and black dashed) on the Gun Point dataset. The barycenter is obtained by minimizing an uncertaintyaware uDTW objective, where β penalizes high matching uncertainty and λ regularizes it toward unit variance. Compared to Euclidean (green) and sDTW (blue) barycenters, uDTW produces distinct interpolations by adaptively weighting alignments according to uncertainty.

# C. Few-shot Image Classification

Encoder. We adopt a pretrained vision backbone (e.g., DI-NOv3 with a ViT encoder), as few-shot learning benefits from transferable representations learned from large-scale datasets. Given an input image, we extract the last-layer patch tokens and discard prefix tokens $( e . g .$ , class and register tokens). The resulting token sequence is projected into a $d ^ { \prime } .$ -dimensional embedding space using a single bias-free linear layer without activation. To model reliability, we use a SigmaNet that predicts a positive scalar uncertainty for each token. The set of learnable parameters is $\mathcal { P } \equiv [ \mathcal { P } _ { \mathrm { p r o j } } , \mathcal { P } _ { \mathrm { S N } } ]$ , where $\mathcal { P } _ { \mathrm { p r o j } }$ denotes the projection layer and $\mathcal { P } _ { \mathrm { S N } }$ the uncertainty predictor.

Supervised. Let $\{ \Phi _ { b } \} _ { b \in \mathcal { T } _ { B } }$ denote the query feature maps and $\{ \Phi _ { b , n , z } ^ { \prime } \} _ { b \in \mathcal { T } _ { B } , n \in \mathcal { T } _ { N } , z \in \mathcal { T } _ { Z } }$ the corresponding support feature maps. For each episode b, the query shares the same label as the supports from class $n = 1 , i . e . , y ( \Phi _ { b } ) = y ( \Phi _ { b , 1 , z } ^ { \prime } )$ , $\forall z \ \in \ \mathcal { T } _ { Z }$ , while supports from $n \ne 1$ belong to different classes. Accordingly, we define similarity labels $\delta _ { 1 } = 0$ for matching pairs and $\delta _ { n \neq 1 } ~ = ~ 1$ otherwise. The supervised objective minimizes a regression loss over pairwise distances:

$$
\mathcal {L} _ {\sup} = \sum_ {b \in \mathcal {I} _ {B}} \sum_ {n \in \mathcal {I} _ {N}} \sum_ {z \in \mathcal {I} _ {Z}} \left(d _ {\mathrm{uDTW}} ^ {2} (\Phi_ {b}, \Phi_ {b, n, z} ^ {\prime}) - \delta_ {n}\right) ^ {2} + \beta \Omega (\Phi_ {b}, \Phi_ {b, n, z} ^ {\prime}). \tag {15}
$$

Unsupervised. The Unsupervised Comparator is trained in a label-free (unsupervised) setting. Given a mini-batch of B images, we generate two stochastic augmented views per image, yielding paired token embeddings $\{ ( \Phi _ { i } , \Phi _ { i } ^ { \prime } ) \} _ { i = 1 } ^ { B }$ 1. The model is optimized using an uncertainty-aware objective:

$$
\mathcal {L} _ {\text { unsup }} = \frac {1}{B} \sum_ {i \in \mathcal {I} _ {B}} \left[ - \log \frac {\exp \left(- d _ {\mathrm{uDTW}} ^ {2} (\boldsymbol {\Phi} _ {i} , \boldsymbol {\Phi} _ {i} ^ {\prime}) / \tau\right)}{\sum_ {j = 1} ^ {B} \exp \left(- d _ {\mathrm{uDTW}} ^ {2} (\boldsymbol {\Phi} _ {i} , \boldsymbol {\Phi} _ {j} ^ {\prime}) / \tau\right)} + \beta \Omega (\boldsymbol {\Phi} _ {i}, \boldsymbol {\Phi} _ {i} ^ {\prime}) \right], \tag {16}
$$

where τ is the temperature. Positive pairs, corresponding to two views of the same image, are pulled together in the embedding space, while the off-diagonal terms in the denominator serve as in-batch negatives and are pushed apart.

![](images/729316199d229cbc10c62f74157c360cec84d66fd41353fdecfaf1b91a4fb817.jpg)

<details>
<summary>line</summary>

| Step | uDTW | sDTW |
|------|------|------|
| 1    | -1.5 | -1.2 |
| 2    | -0.8 | -0.5 |
| 3    | 0.2  | 0.1  |
| 4    | 0.9  | 0.7  |
| 5    | -0.3 | -0.1 |
| 6    | -0.6 | -0.4 |
| 7    | 0.4  | 0.2  |
| 8    | -0.7 | -0.3 |
| 9    | -0.9 | -0.6 |
| 10   | 0.1  | 0.0  |
| 11   | -0.2 | -0.3 |
| 12   | 0.5  | 0.4  |
| 13   | -0.4 | -0.2 |
| 14   | 0.8  | 0.6  |
| 15   | -0.6 | -0.1 |
| 16   | 0.3  | 0.1  |
| 17   | -0.5 | -0.3 |
| 18   | 0.7  | 0.5  |
| 19   | -0.8 | -0.4 |
| 20   | 0.2  | 0.1  |
| 21   | -0.3 | -0.2 |
| 22   | 0.6  | 0.4  |
| 23   | -0.7 | -0.5 |
| 24   | 0.9  | 0.7  |
| 25   | -0.4 | -0.1 |
| 26   | 0.5  | 0.3  |
| 27   | -0.6 | -0.4 |
| 28   | 0.8  | 0.6  |
| 29   | -0.9 | -0.2 |
| 30   | 0.3  | -0.5 |
| 31   | -0.2 | -0.3 |
| 32   | 0.7  | 0.4  |
| 33   | -0.5 | -0.1 |
| 34   | 0.9  | 0.6  |
| 35   | -0.8 | -0.3 |
| 36   | 0.4  | -0.2 |
| 37   | -0.6 | -0.4 |
| 38   | 0.8  | 0.5  |
| 39   | -0.7 | -0.1 |
| 40   | 0.5  | -0.3 |
| 41   | -0.4 | -0.2 |
| 42   | 0.6  | 0.4  |
| 43   | -0.9 | -0.5 |
| 44   | 0.2  | -0.1 |
| 45   | -0.3 | -0.4 |
| 46   | 0.7  | 0.6  |
| 47   | -0.6 | -0.3 |
| 48   | 0.9  | 0.7  |
| 49   | -0.8 | -0.2 |
| 50   | 0.3  | -0.5 |
| 51   | -0.2 | -0.1 |
| 52   | 0.6  | 0.4  |
| 53   | -0.5 | -0.3 |
| 54   | 0.8  | 0.5  |
| 55   | -0.7 | -0.1 |
| 56   | 0.4  | -0.4 |
| 57   | -0.3 | -0.2 |
| 58   | 0.7  | 0.6  |
| 59   | -0.9 | -0.5 |
| 60   | 0.2  | -0.1 |
| 61   | -0.2 | -0.4 |
| 62   | 0.6  | 0.4  |
| 63   | -0.5 | -0.3 |
| 64   | 0.9  | 0.7  |
| 65   | -0.8 | -0.2 |
| 66   | 0.3  | -0.5 |
| 67   | -0.1 | -0.1 |
| 68   | 0.7  | 0.6  |
| 69   | -0.6 | -0.4 |
| 70   | 0.4  | -0.3 |
| 71   | -0.4 | -0.2 |
| 72   | 0.6  | 0.5  |
| 73   | -0.9 | -0.1 |
| 74   | 0.2  | -0.5 |
| 75   | -0.3 | -0.1 |
| 76   | 0.7  | 0.6  |
| 77   | -0.7 | -0.3 |
| 78   | 0.5  | -0.2 |
| 79   | -0.2 | -0.4 |
| 80   | 0.8  | 0.7  |
| 81   | -0.8 | -0.1 |
| 82   | 0.3  | -0.4 |
| 83   | -0.1 | -0.2 |
| 84   | 0.6  | 0.5  |
| 85   | -0.5 | -0.3 |
| 86   | 0.9  | 0.6  |
| 87   | -0.7 | -0.2 |
| 88   | 0.4  | -0.5 |
| 89   | -0.3 | -0.1 |
| 90   | 0.7  | 0.6  |
| 91   | -0.9 | -0.4 |
| 92   | 0.2  | -0.1 |
| 93   | -0.2 | -0.4 |
| 94   | 0.6  | 0.5  |
| 95   | -0.6 | -0.3 |
| 96   | 0.8  | 0.7
 |
| 97   | -0.8 | -0.2 |
| 98   | 0.3    | -0.5 |
| 99   | -0.1    | -0.1 |
|1     | -1    | -1    |
</details>

![](images/68028ef22f445e277b0b25ee0b7b013106eebf7e537a65568b733e561f8fb27a.jpg)

<details>
<summary>line</summary>

| x    | uDTW  | sDTW  |
| ---- | ----- | ----- |
| 0.0  | -1.0  | -1.0  |
| 0.5  | 0.8   | 0.8   |
| 1.0  | 0.8   | 0.8   |
| 1.5  | -0.8  | -0.8  |
| 2.0  | -1.0  | -1.0  |
</details>

![](images/d72e7f09faf17b348206c9e27184f597e1e92d54ca33364f44de017a263ac16c.jpg)

<details>
<summary>line</summary>

| Step | uDTW | sDTW |
|------|------|------|
| 0    | -1.0 | -1.0 |
| 1    | 0.5  | 0.5  |
| 2    | 0.8  | 0.8  |
| 3    | 0.6  | 0.6  |
| 4    | 0.3  | 0.3  |
| 5    | 0.1  | 0.1  |
| 6    | -0.1 | -0.1 |
| 7    | -0.3 | -0.3 |
| 8    | -0.5 | -0.5 |
| 9    | -0.7 | -0.7 |
| 10   | -0.9 | -0.9 |
| 11   | -1.1 | -1.1 |
| 12   | -1.3 | -1.3 |
| 13   | -1.5 | -1.5 |
| 14   | -1.7 | -1.7 |
| 15   | -1.9 | -1.9 |
| 16   | -2.1 | -2.1 |
| 17   | -2.3 | -2.3 |
| 18   | -2.5 | -2.5 |
| 19   | -2.7 | -2.7 |
| 20   | -2.9 | -2.9 |
| 21   | -3.1 | -3.1 |
| 22   | -3.3 | -3.3 |
| 23   | -3.5 | -3.5 |
| 24   | -3.7 | -3.7 |
| 25   | -3.9 | -3.9 |
| 26   | -4.1 | -4.1 |
| 27   | -4.3 | -4.3 |
| 28   | -4.5 | -4.5 |
| 29   | -4.7 | -4.7 |
| 30   | -4.9 | -4.9 |
| 31   | -5.1 | -5.1 |
| 32   | -5.3 | -5.3 |
| 33   | -5.5 | -5.5 |
| 34   | -5.7 | -5.7 |
| 35   | -5.9 | -5.9 |
| 36   | -6.1 | -6.1 |
| 37   | -6.3 | -6.3 |
| 38   | -6.5 | -6.5 |
| 39   | -6.7 | -6.7 |
| 40   | -6.9 | -6.9 |
| 41   | -7.1 | -7.1 |
| 42   | -7.3 | -7.3 |
| 43   | -7.5 | -7.5 |
| 44   | -7.7 | -7.7 |
| 45   | -7.9 | -7.9 |
| 46   | -8.1 | -8.1 |
| 47   | -8.3 | -8.3 |
| 48   | -8.5 | -8.5 |
| 49   | -8.7 | -8.7 |
| 50   | -8.9 | -8.9 |
| 51   | -9.1 | -9.1 |
| 52   | -9.3 | -9.3 |
| 53   | -9.5 | -9.5 |
| 54   | -9.7 | -9.7 |
| 55   | -9.9 | -9.9 |
| 56   | -10.1| -10.1|
| 57   | -10.3| -10.3|
| 58   | -10.5| -10.5|
| 59   | -10.7| -10.7|
| 60   | -10.9| -10.9|
| 61   | -11.1| -11.1|
| 62   | -11.3| -11.3|
| 63   | -11.5| -11.5|
| 64   | -11.7| -11.7|
| 65   | -11.9| -11.9|
| 66   | -12.1| -12.1|
| 67   | -12.3| -12.3|
| 68   | -12.5| -12.5|
| 69   | -12.7| -12.7|
| 70   | -12.9| -12.9|
| 71   | -13.1| -13.1|
| 72   | -13.3| -13.3|
| 73   | -13.5| -13.5|
| 74   | -13.7| -13.7|
| 75   | -13.9| -13.9|
| 76   | -14.1| -14.1|
| 77   | -14.3| -14.3|
| 78   | -14.5| -14.5|
| 79   | -14.7| -14.7|
| 80   | -14.9| -14.9|
| 81   | -15.1| -15.1|
| 82   | -15.3| -15.3|
| 83   | -15.5| -15.5|
| 84   | -15.7| -15.7|
| 85   | -15.9| -15.9|
| 86   | -16.1| -16.1|
| 87   | -16.3| -16.3|
| 88   | -16.5| -16.5|
| 89   | -16.7| -16.7|
| 90   | -16.9| -16.9|
| 91   | -17.1| -17.1|
| 92   | -17.3| -17.3|
| 93   | -17.5| -17.5|
| 94   | -17.7| -17.7|
| 95   | -17.9| -17.9|
| 96   | -18.1| -18.1|
| 97   | -18.3| -18.3|
| 98   | -18.5| -18.5|
| 99   | -18.7| -18.7|
| 100  | -18.9| -18.9|
</details>

(a) CBF

![](images/0fc19956ec58d3b05eed566355d44739aae1507746c23c316ea5e9c798c1c11f.jpg)

<details>
<summary>line</summary>

| x    | uDTW  | sDTW  |
| ---- | ----- | ----- |
| 0    | -1.5  | -1.8  |
| 1    | -0.8  | -0.5  |
| 2    | 0.2   | 0.3   |
| 3    | 0.7   | 0.9   |
| 4    | 1.1   | 1.2   |
| 5    | 0.9   | 0.8   |
| 6    | 0.6   | 0.4   |
| 7    | 0.3   | 0.1   |
| 8    | 0.0   | -0.2  |
| 9    | -0.3  | -0.6  |
| 10   | -0.7  | -1.0  |
| 11   | -1.0  | -1.3  |
| 12   | -0.8  | -0.9  |
| 13   | -0.5  | -0.6  |
| 14   | -0.2  | -0.3  |
| 15   | 0.1   | 0.0   |
| 16   | 0.4   | 0.3   |
| 17   | 0.7   | 0.6   |
| 18   | 1.0   | 0.9   |
| 19   | 1.3   | 1.2   |
| 20   | 1.6   | 1.5   |
| 21   | 1.9   | 1.8   |
| 22   | 2.2   | 2.1   |
| 23   | 2.5   | 2.4   |
| 24   | 2.8   | 2.7   |
| 25   | 3.1   | 3.0   |
| 26   | 3.4   | 3.3   |
| 27   | 3.7   | 3.6   |
| 28   | 4.0   | 3.9   |
| 29   | 4.3   | 4.2   |
| 30   | 4.6   | 4.5   |
| 31   | 4.9   | 4.8   |
| 32   | 5.2   | 5.1   |
| 33   | 5.5   | 5.4   |
| 34   | 5.8   | 5.7   |
| 35   | 6.1   | 6.0   |
| 36   | 6.4   | 6.3   |
| 37   | 6.7   | 6.6   |
| 38   | 7.0   | 6.9   |
| 39   | 7.3   | 7.2   |
| 40   | 7.6   | 7.5   |
| 41   | 7.9   | 7.8   |
| 42   | 8.2   | 8.1   |
| 43   | 8.5   | 8.4   |
| 44   | 8.8   | 8.7   |
| 45   | 9.1   | 9.0   |
| 46   | 9.4   | 9.3   |
| 47   | 9.7   | 9.6   |
| 48   | 10.0  | 9.9   |
| 49   | 10.3  | 10.2  |
| 50+  | -10.0 | -9.5  |
</details>

![](images/fa2aca3f903e760570b47c5ec9a5f6bfb0f3cf05b9ab9a5fbe8ea1b5f50afc31.jpg)

<details>
<summary>line</summary>

| Step | uDTW | sDTW |
|------|------|------|
| 0    | -1.0 | -1.0 |
| 1    | -0.5 | -0.5 |
| 2    | 0.0  | 0.0  |
| 3    | 0.5  | 0.5  |
| 4    | 1.0  | 1.0  |
| 5    | 1.5  | 1.5  |
| 6    | 2.0  | 2.0  |
</details>

![](images/b3b5ed6cf18dbf059271a6d5bd9394ced278cb3d3bda9f81a6d9322893a01865.jpg)

<details>
<summary>line</summary>

| x    | uDTW  | sDTW  |
| ---- | ----- | ----- |
| 0    | -1.0  | -1.0  |
| 1    | -0.5  | -0.5  |
| 2    | 0.0   | 0.0   |
| 3    | 0.5   | 0.5   |
| 4    | 1.0   | 1.0   |
| 5    | 1.5   | 1.5   |
| 6    | 2.0   | 2.0   |
</details>

0  0 20 40 60 (b) Synthetic Control   
Fig. 6. Comparison of barycenters computed with sDTW and uDTW on CBF and Synthetic Control. Uncertainty for uDTW is shown in red. Even with high γ (e.g., γ = 10.0), uDTW produces reasonable barycenters; higher γ smooths the barycenter but increases uncertainty.

This InfoNCE-style loss provides a principled framework for self-supervised learning of token embeddings with structured alignment via uDTW. After unsupervised pretraining, the learned projection and SigmaNet are frozen and reused in standard few-shot prototype evaluation.

Inference. During evaluation, class prototypes are computed as the mean of the token embeddings in the support set for each class. In the supervised setting, query features can be matched either to individual support features or to these class prototypes using uDTW. In the unsupervised setting, both support and query samples are encoded using the frozen projector. Classification is then performed by computing the uDTW distance between each query and the class prototypes, and assigning the query to the class with the minimum distance.

# V. EXPERIMENT

# A. Datasets, Setups, and Evaluation Protocols

Time series analysis. We use the UCR archive [30], a standard benchmark which contains sequences from diverse domains such as astronomy, geology, and medical imaging, with variable lengths and complexity. For classification, we follow standard train/validation/test splits and evaluate nearestneighbor and nearest-centroid classifiers. Forecasting is evaluated by using the first portion (60%) of each sequence as input and the remainder (40%) as output, with metrics including MSE and alignment-based distances (e.g., sDTW). Barycenter computation is performed following sDTW [3].

Few-shot action recognition (sequences). Skeleton-based action recognition is evaluated on (i) NTU-60 [31], which contains 56,880 RGB+D sequences over 60 action classes with high intra-class variability, (ii) NTU-120 [32], an extension to 120 classes and 114,480 sequences captured from 106 subjects across 155 camera viewpoints, and (iii) Kinetics-skeleton [33], a large-scale human action dataset. Evaluation follows fewshot protocols [6], [32]. Baselines include Matching Networks [34], Prototypical Networks [35], and TAP [9], chosen for their strong performance in few-shot action recognition.

TABLE I CLASSIFICATION ACCURACY (MEAN±STD) ON THE UCR ARCHIVE. COLUMNS INDICATE THE DISTANCE USED FOR CLASS PROTOTYPES; K DENOTES THE NUMBER OF NEAREST NEIGHBORS. 

<table><tr><td rowspan="2"></td><td colspan="3">Nearest neighbor</td><td rowspan="2">Nearest centroid</td></tr><tr><td>K=1</td><td>K=3</td><td>K=5</td></tr><tr><td>Euclidean</td><td>71.2±17.5</td><td>72.3±18.1</td><td>73.0±16.7</td><td>61.3±20.1</td></tr><tr><td>DTW [2]</td><td>74.2±16.6</td><td>75.0±17.0</td><td>75.4±15.8</td><td>65.9±18.8</td></tr><tr><td>sDTW [3]</td><td>76.2±16.6</td><td>77.2±15.9</td><td>78.0±16.5</td><td>70.5±17.6</td></tr><tr><td>sDTW div. [7]</td><td>78.6±16.2</td><td>79.5±16.7</td><td>80.1±16.5</td><td>70.9±17.8</td></tr><tr><td>uDTW</td><td>80.0±15.0</td><td>81.2±17.8</td><td>83.3±16.2</td><td>72.2±16.0</td></tr></table>

TABLE II TIME SERIES FORECASTING ON ECG5000 (100 RUNS, MEAN±STD), WITH COLUMN-WISE DISTANCES USED FOR TRAINING AND ROW-WISE DISTANCES FOR TEST-TIME COMPARISON WITH GROUND TRUTH (LOWER IS BETTER). BEST METHOD(S) IN BOLD (STUDENT’S t-TEST). 

<table><tr><td></td><td>MSE</td><td>DTW</td><td>sDTW div.</td><td>uDTW</td></tr><tr><td>Euclidean</td><td>32.1±1.62</td><td>20.0±0.18</td><td>15.3±0.16</td><td>14.4±0.18</td></tr><tr><td>sDTW [3]</td><td>38.6±6.30</td><td>17.2±0.80</td><td>22.6±3.59</td><td>32.1±2.25</td></tr><tr><td>sDTW div. [7]</td><td>24.6±1.37</td><td>38.9±5.33</td><td>20.0±2.44</td><td>15.4±1.62</td></tr><tr><td>uDTW</td><td>23.0±1.22</td><td>16.7±0.08</td><td>16.8±1.62</td><td>8.27±0.79</td></tr></table>

Few-shot image classification. For generic few-shot classification, we evaluate on CIFAR-FS [36], miniImageNet [34], and tieredImageNet [37], which contain diverse object categories with moderate inter-class variation. Fine-grained datasets include CUB-200-2011 [38], Stanford Dogs [39], Stanford Cars [40], Aircraft [41], and NABirds [42], which require discriminative localized features to distinguish subtle differences within a category. Ultra-fine-grained (UFGVC) datasets [43], such as Cotton80 and SoyLocal, feature minimal visual variation and very few samples per class. Standard evaluation uses 5-way 1-/5-shot episodes for generic and finegrained datasets, and 5-way 1-/2-shot for UFGVC datasets due to the limited number of samples per class. 1 In the unsupervised setting, evaluations include Omniglot [44] and miniImageNet for generic few-shot tasks, as well as crossdomain tests on CUB-200-2011, Stanford Cars, Cotton80 and SoyLocal, where embeddings are learned on miniImageNet in an unsupervised manner, frozen, and then applied to targetdomain few-shot episodes. Baselines comprise supervised methods such as LA-PID [45], CPEA [46], ASCM [47], as well as unsupervised approaches including CACTUs [48], UMTRA [49], LASIUM [50], PsCo [51], Meta-GMVAE [52], and Meta-SVEBM [53]. We adopt DINOv3 (ViT-S/16), pretrained on LVD-1689M [54], as a frozen feature backbone. On top, we train a projection layer (single bias-free FC, 384→256) for each objective, and an additional SigmaNet for uDTW (a single FC, 256→256, followed by feature-wise average pooling and a scaled sigmoid, as in [5]). We evaluate three objectives: squared Euclidean distance, sDTW, and uDTW. Euclidean distance operates on pooled class prototypes, whereas sDTW and uDTW perform token-level alignment over patch embeddings, with uDTW further using per-token uncertainty.

![](images/8b1cacd8d9aa1fa4512bb49889fbd5e45b0eb5856e18a20195d48fed12073f71.jpg)

<details>
<summary>line</summary>

| x  | Eucl. | sDTW | uDTW | GT   |
|----|-------|------|------|------|
| 0  | 0.0   | 0.0  | 0.0  | 0.0  |
| 20 | 1.0   | 1.0  | 1.0  | -1.0 |
| 40 | 1.5   | 1.5  | 1.5  | -0.5 |
| 60 | 0.5   | 0.5  | 0.5  | 0.5  |
| 80 | 0.0   | 0.0  | 0.0  | -1.0 |
| 100| -1.0  | -1.0 | -1.0 | -2.0 |
</details>

Eucl. (a) ECG200

![](images/cbf70bc57004b2235108083a8c05bd4676d2928494c98b2951ef384b6529b48f.jpg)

<details>
<summary>line</summary>

| x    | Eucl. | sDTW | uDTW | GT   |
| ---- | ----- | ---- | ---- | ---- |
| 0    |       |      |      | -4.0 |
| 50   |       |      |      | 0.0  |
| 100  | 1.0   | 1.0  | 1.0  | 1.0  |
| 150  | -3.0  | -3.0 | -3.0 | -3.0 |
</details>

(b) ECG5000   
2  uDTW  2  uDTW Fig. 7. Prediction of time series continuation using MLPs with Euclidean, sDTW, or uDTW distances. Given the first part of a series, three MLPs predict the remaining values on ECG200 and ECG5000 (UCR archive). Red shading indicates uDTW uncertainty. Predictions and ground truth (GT) for a sample are shown. uDTW often captures sudden changes more accurately.

# 0 20 40 60 80 100  0 20 B. Analysis and Prediction of Time Series

1) Frechet Mean of Time Series:´ We visualize the Frechet´ mean under Euclidean distance, sDTW, and our uDTW.

Setup. Following [3], for each dataset in UCR, we randomly select a class and pick 10 time series to compute its barycenter. Our uDTW barycenter objective is minimized using L-BFGS [55] with a maximum of 100 iterations.

Qualitative results. First, we average two time series (Fig. 5). Averaging under uDTW produces substantially different results compared to Euclidean and sDTW. Fig. 6 shows barycenters from sDTW and uDTW. Our uDTW yields more reasonable barycenters, even for large $\gamma ( e . g . , \gamma = 1 0 $ , right column), where change points in the red curve appear sharper. With low smoothing $( \gamma = 0 . 1 )$ , both uDTW and sDTW can get stuck in poor local minima, but uDTW produces fewer sharp peaks due to the uncertainty measure. Higher γ values smooth the barycenter but increase uncertainty (see comparison of $\gamma = 0 . 1$ vs. γ = 10.0). $\mathbf { A } \mathbf { t } \ \gamma = 1$ , barycenters from sDTW and uDTW align well with the original time series.

2) Classification of Time Series: We evaluate uDTW on UCR time series classification, using a K-nearest neighbor classifier with softmax for prediction.

Setup. Each dataset is split 50/25/25 (train/val/test). The classifier is evaluated for $K \in \{ 1 , 2 , 3 \}$ .

Quantitative results. Table I compares uDTW with Euclidean, DTW, sDTW, and sDTW div. Using uDTW to compute barycenters improves the nearest centroid classifier by approximately 2% over sDTW div. For the K-nearest neighbor classifier, uDTW increases accuracy by 1.4%, 1.7%, and 3.2% for K = 1, 2, 3, respectively.

3) Forecasting the Evolution of Time Series: We evaluate uDTW for forecasting future values of time series, aiming to capture both smooth trends and abrupt changes.

Qualitative results. Fig. 7 visualizes the predictions. sDTW and uDTW predictions can differ, but uDTW better matches the ground truth and detects sharp changes more accurately.

Quantitative results. We validate uDTW on ECG5000 from the UCR archive, which contains 5000 electrocardiograms (500 for training and 4500 for testing), each of length 140. To evaluate predictions, we use two metrics: (i) MSE for pertime-step errors, and (ii) DTW, sDTW div., and uDTW to compare the overall shape of the time series. Shape-based metrics are important because varying series lengths can bias

TABLE III EVALUATIONS ON NTU-60 (LEFT) AND NTU-120 (RIGHT). 

<table><tr><td rowspan="2">#classes</td><td colspan="5">NTU-60</td><td colspan="5">NTU-120</td></tr><tr><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>20</td><td>40</td><td>60</td><td>80</td><td>100</td></tr><tr><td colspan="11">Supervised</td></tr><tr><td>MatchNets [34]</td><td>46.1</td><td>48.6</td><td>53.3</td><td>56.3</td><td>58.8</td><td>20.5</td><td>23.4</td><td>25.1</td><td>28.7</td><td>30.0</td></tr><tr><td>ProtoNet [35]</td><td>47.2</td><td>51.1</td><td>54.3</td><td>58.9</td><td>63.0</td><td>21.7</td><td>24.0</td><td>25.9</td><td>29.2</td><td>32.1</td></tr><tr><td>TAP [9]</td><td>54.2</td><td>57.3</td><td>61.7</td><td>64.7</td><td>68.3</td><td>31.2</td><td>37.7</td><td>40.9</td><td>44.5</td><td>47.3</td></tr><tr><td>Euclidean</td><td>38.5</td><td>42.2</td><td>45.1</td><td>48.3</td><td>50.9</td><td>18.7</td><td>21.3</td><td>24.9</td><td>27.5</td><td>30.0</td></tr><tr><td>sDTW [3]</td><td>53.7</td><td>56.2</td><td>60.0</td><td>63.9</td><td>67.8</td><td>30.3</td><td>37.2</td><td>39.7</td><td>44.0</td><td>46.8</td></tr><tr><td>sDTW div. [7]</td><td>54.0</td><td>57.3</td><td>62.1</td><td>65.7</td><td>69.0</td><td>30.8</td><td>38.1</td><td>40.0</td><td>44.7</td><td>47.3</td></tr><tr><td>uDTW</td><td>56.9</td><td>61.2</td><td>64.8</td><td>68.3</td><td>72.4</td><td>32.2</td><td>39.0</td><td>41.2</td><td>45.3</td><td>49.0</td></tr><tr><td colspan="11">Unsupervised</td></tr><tr><td>Euclidean</td><td>20.9</td><td>23.7</td><td>26.3</td><td>30.0</td><td>33.1</td><td>13.5</td><td>16.3</td><td>20.0</td><td>24.9</td><td>26.2</td></tr><tr><td>sDTW [3]</td><td>35.6</td><td>45.2</td><td>53.3</td><td>56.7</td><td>61.7</td><td>20.1</td><td>25.3</td><td>32.0</td><td>36.9</td><td>40.9</td></tr><tr><td>sDTW div. [7]</td><td>36.0</td><td>46.1</td><td>54.0</td><td>57.2</td><td>62.0</td><td>20.8</td><td>26.0</td><td>33.2</td><td>37.5</td><td>42.3</td></tr><tr><td>uDTW</td><td>37.0</td><td>48.3</td><td>55.3</td><td>58.0</td><td>63.3</td><td>22.7</td><td>28.3</td><td>35.9</td><td>39.4</td><td>44.0</td></tr></table>

TABLE IV EVALUATIONS ON 2D AND 3D KINETICS-SKELETON. 

<table><tr><td rowspan="2"></td><td colspan="2">Supervised</td><td colspan="2">Unsupervised</td></tr><tr><td>2D</td><td>3D</td><td>2D</td><td>3D</td></tr><tr><td>Euclidean</td><td>21.2</td><td>23.1</td><td>12.7</td><td>13.3</td></tr><tr><td>TAP [9]</td><td>32.9</td><td>36.0</td><td>-</td><td>-</td></tr><tr><td>sDTW [3]</td><td>34.7</td><td>39.6</td><td>23.3</td><td>28.3</td></tr><tr><td>sDTW div. [7]</td><td>35.0</td><td>40.1</td><td>24.0</td><td>28.9</td></tr><tr><td>uDTW</td><td>35.5</td><td>42.0</td><td>25.9</td><td>32.7</td></tr></table>

MSE, ignoring trends in the signal. We apply a Student’s ttest (significance level 0.05) over 100 runs to highlight the best performance. Table II shows that uDTW achieves nearbest results on both MSE and shape metrics.

# C. Few-shot Action Recognition

We evaluate uDTW for few-shot action recognition, classifying actions from a few labeled support examples.

Setup. For NTU-120, we follow the standard one-shot protocol [32]. Based on this, we construct a similar one-shot protocol for NTU-60, using 50 action classes for training and 10 for testing. We also evaluate on both 2D and 3D Kinetics-skeleton, splitting 200 actions for training and using the remainder for testing. Each video block is reshaped and resized to a 224 × 224 color image, and fed into MatchNets and ProtoNet to learn feature representations.

Quantitative results. Tables III and IV show that uDTW consistently outperforms sDTW and sDTW div. in both supervised and unsupervised settings. On Kinetics-skeleton, uDTW improves performance on 3D skeletons by 2.4% (supervised) and 4.4% (unsupervised). In the supervised setting, uDTW outperforms TAP by approximately 4% on NTU-60 and 2% on NTU-120. In the unsupervised setting, it surpasses sDTW by about 2% and 3% on NTU-60 and NTU-120, respectively.

# D. Few-shot Image Classification

1) Quantitative Results: We evaluate uDTW across generic, fine-, and ultra-fine-grained few-shot image classification.

Supervised few-shot classification. On generic benchmarks (Table V), uDTW consistently achieves the highest accuracy for both 1-shot and 5-shot tasks, improving over the secondbest methods (typically sDTW) by 0.5–2%. This demonstrates its strength in capturing intra-class structural variations and aligning features across limited support samples. On finegrained datasets (Table VI), uDTW generally outperforms prior methods, particularly in 1-shot scenarios. However, it occasionally ranks second, for example on Stanford Cars (compared to BTG-Net) and CUB-200-2011 (compared to AFS-FR), likely because these methods use highly discriminative pre-trained embeddings that already separate classes effectively, leaving limited room for additional gains from alignment. On UFGVC benchmarks (Table VII), uDTW consistently attains the best results, though margins over sDTW are modest, reflecting the challenge of extremely low-data, ultra-fine-grained categories where subtle inter-class differences constrain further improvements. These patterns indicate that uDTW provides the largest benefit when intra-class variability is high, the number of support examples is small, or

![](images/38fd57383e3ba96d8af0fd8a54ea59e04671b85617b830ef7e50a6ab068b597f.jpg)

Fig. 8. Comparison between DAAM attention and the learned uncertainty across MiniImageNet, Stanford Cars, CUB-200-2011, UFGVC, and Aircraft. Each triplet shows the input image, DAAM attention (brighter = higher importance), and uDTW uncertainty (blue = low, red = high). Low-uncertainty regions consistently coincide with high-attention responses, evidencing the reverse-attention effect induced by our uncertainty-aware alignment, where reliable (semantically informative) regions are emphasized while ambiguous ones are suppressed.   
TABLE V FEW-SHOT ACCURACY (%) ON GENERIC IMAGE CLASSIFICATION BENCHMARKS. BEST IN BOLD, SECOND-BEST UNDERLINED. 

<table><tr><td>Dataset</td><td>Method</td><td>1-shot</td><td>5-shot</td></tr><tr><td rowspan="8">CIFAR-FS</td><td>CPEA [46]</td><td> $77.82 \pm 0.66$ </td><td> $88.98 \pm 0.45$ </td></tr><tr><td>ASCM [47]</td><td> $77.25 \pm 0.75$ </td><td> $88.83 \pm 0.58$ </td></tr><tr><td>L2TT [56]</td><td> $73.60 \pm 0.10$ </td><td> $85.80 \pm 0.10$ </td></tr><tr><td>LA-PID [45]</td><td> $73.20 \pm 0.40$ </td><td> $85.80 \pm 0.30$ </td></tr><tr><td>SSL-ProtoNet [57]</td><td> $60.41 \pm 0.52$ </td><td> $76.52 \pm 0.38$ </td></tr><tr><td>Euclidean</td><td> $74.01 \pm 0.86$ </td><td> $85.45 \pm 0.57$ </td></tr><tr><td>sDTW</td><td> $79.96 \pm 0.83$ </td><td> $90.24 \pm 0.50$ </td></tr><tr><td>uDTW</td><td> $80.32 \pm 0.81$ </td><td> $91.15 \pm 0.50$ </td></tr><tr><td rowspan="14">miniImageNet</td><td>SemFew-Trans [58]</td><td> $78.94 \pm 0.66$ </td><td> $86.49 \pm 0.50$ </td></tr><tr><td>LA-PID [45]</td><td> $72.20 \pm 0.50$ </td><td> $87.30 \pm 0.50$ </td></tr><tr><td>CPEA [46]</td><td> $71.97 \pm 0.65$ </td><td> $87.06 \pm 0.38$ </td></tr><tr><td>CORE [59]</td><td> $70.50 \pm 0.44$ </td><td> $86.28 \pm 0.29$ </td></tr><tr><td>ANROT-HELANet [60]</td><td> $69.40 \pm 0.30$ </td><td> $88.10 \pm 0.40$ </td></tr><tr><td>ASCM [47]</td><td> $69.35 \pm 0.65$ </td><td> $84.87 \pm 0.58$ </td></tr><tr><td>FA-adapter [61]</td><td> $68.36 \pm 0.45$ </td><td> $83.44 \pm 0.28$ </td></tr><tr><td>TNPNet [62]</td><td> $68.22 \pm 0.87$ </td><td> $82.50 \pm 0.82$ </td></tr><tr><td>HELA-VFA [63]</td><td> $68.20 \pm 0.30$ </td><td> $86.70 \pm 0.70$ </td></tr><tr><td>MetaDiff [64]</td><td> $64.99 \pm 0.77$ </td><td> $81.21 \pm 0.56$ </td></tr><tr><td>SSL-ProtoNet [57]</td><td> $52.58 \pm 0.45$ </td><td> $70.87 \pm 0.36$ </td></tr><tr><td>Euclidean</td><td> $81.66 \pm 0.81$ </td><td> $92.56 \pm 0.42$ </td></tr><tr><td>sDTW</td><td> $85.82 \pm 0.69$ </td><td> $95.00 \pm 0.28$ </td></tr><tr><td>uDTW</td><td> $88.05 \pm 0.63$ </td><td> $95.59 \pm 0.28$ </td></tr><tr><td rowspan="11">tieredImageNet</td><td>SSL-ProtoNet [57]</td><td> $55.14 \pm 0.49$ </td><td> $74.23 \pm 0.40$ </td></tr><tr><td>ASCM [47]</td><td> $69.35 \pm 0.65$ </td><td> $84.87 \pm 0.58$ </td></tr><tr><td>CPEA [46]</td><td> $76.93 \pm 0.70$ </td><td> $90.12 \pm 0.45$ </td></tr><tr><td>LA-PID [45]</td><td> $68.10 \pm 0.50$ </td><td> $90.60 \pm 0.50$ </td></tr><tr><td>HELA-VFA [63]</td><td> $72.50 \pm 0.50$ </td><td> $87.60 \pm 0.10$ </td></tr><tr><td>MetaDiff [64]</td><td> $64.99 \pm 0.77$ </td><td> $81.21 \pm 0.56$ </td></tr><tr><td>FA-adapter [61]</td><td> $68.36 \pm 0.45$ </td><td> $86.49 \pm 0.50$ </td></tr><tr><td>SemFew-Trans [58]</td><td> $78.94 \pm 0.66$ </td><td> $86.49 \pm 0.50$ </td></tr><tr><td>Euclidean</td><td> $77.71 \pm 0.89$ </td><td> $88.49 \pm 0.57$ </td></tr><tr><td>sDTW</td><td> $81.54 \pm 0.81$ </td><td> $92.37 \pm 0.47$ </td></tr><tr><td>uDTW</td><td> $83.44 \pm 0.78$ </td><td> $93.07 \pm 0.46$ </td></tr></table>

TABLE VI FEW-SHOT CLASSIFICATION ACCURACY (%) ON FINE-GRAINED DATASETS. 

<table><tr><td>Dataset</td><td>Method</td><td>1-shot</td><td>5-shot</td></tr><tr><td rowspan="12">CUB-200-2011</td><td>AFS-FR [65]</td><td> $94.40 \pm 0.13$ </td><td> $97.96 \pm 0.05$ </td></tr><tr><td>CORE [59]</td><td> $82.36 \pm 0.41$ </td><td> $91.89 \pm 0.30$ </td></tr><tr><td>BSFA [66]</td><td> $82.27 \pm 0.46$ </td><td> $90.76 \pm 0.26$ </td></tr><tr><td>MLCN [67]</td><td> $77.96 \pm 0.44$ </td><td> $91.20 \pm 0.24$ </td></tr><tr><td>Sun et al. [68]</td><td>75.14</td><td>88.87</td></tr><tr><td>S2M2 [69]</td><td> $72.92 \pm 0.83$ </td><td> $86.55 \pm 0.51$ </td></tr><tr><td>RSaD [70]</td><td>71.15</td><td>84.03</td></tr><tr><td>T2L [71]</td><td>71.04</td><td>83.44</td></tr><tr><td>TripletMAML [72]</td><td> $70.46 \pm 0.17$ </td><td> $81.43 \pm 0.86$ </td></tr><tr><td>Euclidean</td><td> $72.00 \pm 0.93$ </td><td> $85.61 \pm 0.59$ </td></tr><tr><td>sDTW</td><td> $84.17 \pm 0.77$ </td><td> $93.74 \pm 0.44$ </td></tr><tr><td>uDTW</td><td> $87.77 \pm 0.68$ </td><td> $95.08 \pm 0.38$ </td></tr><tr><td rowspan="6">Stanford Dogs</td><td>T2L [71]</td><td>52.12</td><td>70.83</td></tr><tr><td>FOT [73]</td><td> $49.32 \pm 0.74$ </td><td> $68.18 \pm 0.69$ </td></tr><tr><td>RSaD [70]</td><td> $73.75 \pm 0.93$ </td><td> $86.65 \pm 0.54$ </td></tr><tr><td>Euclidean</td><td> $71.48 \pm 0.95$ </td><td> $86.31 \pm 0.54$ </td></tr><tr><td>sDTW</td><td> $75.32 \pm 0.80$ </td><td> $89.97 \pm 0.45$ </td></tr><tr><td>uDTW</td><td> $83.38 \pm 0.73$ </td><td> $93.58 \pm 0.37$ </td></tr><tr><td rowspan="7">Stanford Cars</td><td>T2L [71]</td><td>56.80</td><td>74.10</td></tr><tr><td>BTG-Net [74]</td><td> $90.28 \pm 0.34$ </td><td> $96.78 \pm 0.15$ </td></tr><tr><td>BSFA [66]</td><td> $88.93 \pm 0.38$ </td><td> $95.20 \pm 0.20$ </td></tr><tr><td>RSaD [70]</td><td> $87.27 \pm 0.70$ </td><td> $95.01 \pm 0.49$ </td></tr><tr><td>Euclidean</td><td> $68.34 \pm 0.92$ </td><td> $83.42 \pm 0.65$ </td></tr><tr><td>sDTW</td><td> $82.58 \pm 0.83$ </td><td> $94.82 \pm 0.38$ </td></tr><tr><td>uDTW</td><td> $90.45 \pm 0.63$ </td><td> $96.64 \pm 0.31$ </td></tr><tr><td rowspan="7">Aircraft</td><td>BSNet [75]</td><td> $64.83 \pm 1.00$ </td><td> $80.25 \pm 0.67$ </td></tr><tr><td>FRN [76]</td><td>70.17</td><td>83.81</td></tr><tr><td>CTX [77]</td><td>65.60</td><td>80.20</td></tr><tr><td>LE-ProtoPNet [78]</td><td> $82.82$ </td><td> $89.13$ </td></tr><tr><td>Euclidean</td><td> $62.58 \pm 1.00$ </td><td> $76.13 \pm 0.79$ </td></tr><tr><td>sDTW</td><td> $74.02 \pm 1.02$ </td><td> $88.03 \pm 0.67$ </td></tr><tr><td>uDTW</td><td> $83.89 \pm 0.88$ </td><td> $91.68 \pm 0.56$ </td></tr><tr><td rowspan="6">NABirds</td><td>CovaMNet [79]</td><td> $66.29 \pm 0.82$ </td><td> $82.54 \pm 0.87$ </td></tr><tr><td>LRPABN [80]</td><td> $67.73 \pm 0.81$ </td><td> $81.62 \pm 0.58$ </td></tr><tr><td>PACNet [81]</td><td> $75.30 \pm 0.90$ </td><td> $88.20 \pm 0.60$ </td></tr><tr><td>Euclidean</td><td> $71.75 \pm 1.32$ </td><td> $76.17 \pm 1.52$ </td></tr><tr><td>sDTW</td><td> $81.19 \pm 0.79$ </td><td> $92.85 \pm 0.42$ </td></tr><tr><td>uDTW</td><td> $85.72 \pm 0.72$ </td><td> $94.49 \pm 0.37$ </td></tr></table>

embeddings alone are insufficient for class separation.

Unsupervised few-shot classification. Across generic benchmarks (Table VIII), uDTW consistently attains the best or second-best accuracy across all settings, outperforming prior methods such as UMTRA and LASIUM. The improvements are particularly notable for higher-way and higher-shot tasks on miniImageNet, reflecting uDTW’s ability to align feature sequences effectively even without labels. On Omniglot, UM-TRA slightly surpasses uDTW in some high-shot settings, likely because it generates synthetic tasks tailored to the dataset’s character-level structure, reducing the incremental benefit of alignment. In cross-domain scenarios (Table IX), uDTW achieves the highest accuracy in all evaluated tasks, with the largest gains observed on Stanford Cars, where domain shift is more severe. This suggests that uDTW’s sequence alignment is especially effective for mitigating distribution mismatch, though on CUB-200-2011 the margin over sDTW is smaller, possibly because the domain shift is less extreme and embeddings already provide a strong prior.

TABLE VII FEW-SHOT ACCURACY (%) ON UFGVC BENCHMARKS. 

<table><tr><td></td><td>Dataset</td><td>Method</td><td>1-shot</td><td>2-shot</td></tr><tr><td rowspan="6">Supervised</td><td rowspan="3">Cotton80</td><td>Euclidean</td><td>60.65 ± 1.40</td><td>71.07 ± 1.60</td></tr><tr><td>sDTW</td><td>64.02 ± 1.38</td><td>71.40 ± 1.61</td></tr><tr><td>uDTW</td><td>64.72 ± 1.35</td><td>75.47 ± 1.54</td></tr><tr><td rowspan="3">SoyLocal</td><td>Euclidean</td><td>59.40 ± 1.36</td><td>67.63 ± 1.67</td></tr><tr><td>sDTW</td><td>58.85 ± 1.34</td><td>71.43 ± 1.62</td></tr><tr><td>uDTW</td><td>61.25 ± 1.35</td><td>72.67 ± 1.57</td></tr><tr><td rowspan="6">Unsupervised</td><td rowspan="3">Cotton80</td><td>Euclidean</td><td>51.77 ± 1.29</td><td>60.73 ± 1.65</td></tr><tr><td>sDTW</td><td>58.48 ± 1.25</td><td>69.27 ± 1.59</td></tr><tr><td>uDTW</td><td>60.12 ± 1.27</td><td>72.30 ± 1.57</td></tr><tr><td rowspan="3">SoyLocal</td><td>Euclidean</td><td>50.53 ± 1.45</td><td>58.60 ± 1.77</td></tr><tr><td>sDTW</td><td>54.13 ± 1.42</td><td>64.30 ± 1.67</td></tr><tr><td>uDTW</td><td>56.78 ± 1.40</td><td>64.83 ± 1.69</td></tr></table>

TABLE VIII UNSUPERVISED FEW-SHOT CLASSIFICATION ACCURACY (%) ON OMNIGLOT AND MINIIMAGENET. 

<table><tr><td rowspan="2">Method</td><td colspan="4">Omniglot</td><td colspan="4">miniImageNet</td></tr><tr><td>(5,1)</td><td>(5,5)</td><td>(20,1)</td><td>(20,5)</td><td>(5,1)</td><td>(5,5)</td><td>(5,20)</td><td>(5,50)</td></tr><tr><td>CACTUs-MAML [48]</td><td>68.84</td><td>87.78</td><td>48.09</td><td>73.36</td><td>39.90</td><td>53.97</td><td>63.84</td><td>69.64</td></tr><tr><td>CACTUs-ProtoNets [48]</td><td>68.12</td><td>83.58</td><td>47.75</td><td>66.27</td><td>39.18</td><td>53.36</td><td>61.54</td><td>63.55</td></tr><tr><td>UMTRA [49]</td><td> $\underline{83.80}$ </td><td> $\underline{95.43}$ </td><td> $\underline{74.25}$ </td><td> $\underline{92.12}$ </td><td>39.93</td><td>50.73</td><td>61.11</td><td>67.15</td></tr><tr><td>LASIUM-MAML [50]</td><td>83.26</td><td>95.29</td><td>-</td><td>-</td><td>40.19</td><td>54.56</td><td>65.17</td><td>69.13</td></tr><tr><td>LASIUM-ProtoNets [50]</td><td>80.15</td><td>91.10</td><td>-</td><td>-</td><td>40.05</td><td>52.53</td><td>61.09</td><td>64.89</td></tr><tr><td>Euclidean</td><td>82.20</td><td>93.09</td><td>62.61</td><td>81.32</td><td>76.92</td><td>91.21</td><td>93.93</td><td>94.47</td></tr><tr><td>sDTW</td><td>83.13</td><td>94.55</td><td>67.69</td><td>87.30</td><td>79.29</td><td>92.96</td><td>96.36</td><td>96.96</td></tr><tr><td>uDTW</td><td> $\underline{83.84}$ </td><td>94.88</td><td> $\underline{68.21}$ </td><td> $\underline{87.48}$ </td><td> $\underline{80.18}$ </td><td> $\underline{93.16}$ </td><td> $\underline{96.55}$ </td><td> $\underline{97.13}$ </td></tr></table>

2) Qualitative Results: Fig. 8 highlights a consistent correspondence between token-level uncertainty and attention across datasets. Regions with low uncertainty align closely with high-response areas in DAAM attention maps, while less informative or ambiguous regions exhibit higher uncertainty. This shows a clear reverse-attention effect induced by our uncertainty-aware alignment, where semantically meaningful regions are automatically emphasized through low uncertainty, without relying on an explicit attention mechanism. Compared to attention, the learned uncertainty provides a more direct measure of correspondence reliability, leading to cleaner and more interpretable token importance. These observations support our formulation, demonstrating that uDTW captures semantic relevance via uncertainty and offers a principled

TABLE IX CROSS-DOMAIN FEW-SHOT CLASSIFICATION ACCURACY (%) ON CUB-200-2011, STANFORD CARS, COTTON80, AND SOYLOCAL. 

<table><tr><td rowspan="2">Method</td><td colspan="2">CUB-200-2011</td><td colspan="2">Stanford Cars</td><td colspan="2">Cotton80</td><td colspan="2">SoyLocal</td></tr><tr><td>(5,5)</td><td>(5,20)</td><td>(5,5)</td><td>(5,20)</td><td>(5,1)</td><td>(5,2)</td><td>(5,1)</td><td>(5,2)</td></tr><tr><td>Meta-GMVAE [52]</td><td>47.48</td><td>54.08</td><td>31.39</td><td>38.36</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Meta-SVEBM [53]</td><td>45.50</td><td>54.61</td><td>34.27</td><td>46.23</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PsCo [51]</td><td>57.38</td><td>68.58</td><td>44.01</td><td>57.50</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Euclidean</td><td>69.20</td><td>76.94</td><td>57.55</td><td>68.70</td><td>58.18</td><td>67.43</td><td>55.08</td><td>62.30</td></tr><tr><td>sDTW</td><td>78.73</td><td>87.25</td><td>65.31</td><td>79.79</td><td>58.17</td><td>67.57</td><td>55.05</td><td>62.10</td></tr><tr><td>uDTW</td><td>79.20</td><td>89.12</td><td>69.56</td><td>79.96</td><td>58.27</td><td>68.47</td><td>55.70</td><td>62.37</td></tr></table>

alternative to similarity-based attention.

# VI. CONCLUSION

We introduced uncertainty-aware alignment, a probabilistic generalization of DTW that models heteroscedastic uncertainty in pairwise correspondences. Derived from a maximumlikelihood formulation, the resulting objective combines precision-weighted matching with log-variance regularization, enabling robust and interpretable alignment under noise and ambiguity. The soft relaxation induces a Gibbs like distribution over alignment paths, forming structured coupling. This coupling can be viewed as a path-constrained transport plan and a globally consistent attention mechanism, where uncertainty acts as a reliability-aware modulation that suppresses ambiguous matches. We further extended the framework to tokenized visual representations, where learned uncertainty induces a reverse-attention effect: informative regions exhibit low uncertainty and dominate the alignment, while noisy regions are attenuated. Experiments across diverse domains demonstrate consistent improvements and reveal that uncertainty correlates with semantic importance.

# REFERENCES

[1] Y. Liao, Y. Gao, and W. Zhang, “Dynamic accumulated attention map for interpreting evolution of decision-making in vision transformer,” Pattern Recognition, vol. 165, p. 111607, 2025.   
[2] M. Cuturi, “Fast global alignment kernels,” in International Conference on Machine Learning (ICML), 2011, pp. 929–936.   
[3] M. Cuturi and M. Blondel, “Soft-dtw: a differentiable loss function for time-series,” in International Conference on Machine Learning (ICML). PMLR, 2017, pp. 894–903.   
[4] L. Wang and P. Koniusz, “Temporal-viewpoint transportation plan for skeletal few-shot action recognition,” in Asian conference on computer vision (ACCV), 2022, pp. 4176–4193.   
[5] —, “Uncertainty-dtw for time series and sequences,” in European Conference on Computer Vision (ECCV), 2022, pp. 176–195.   
[6] L. Wang, J. Liu, L. Zheng, T. Gedeon, and P. Koniusz, “Meet jeanie: a similarity measure for 3d skeleton sequences via temporal-viewpoint alignment,” International Journal of Computer Vision, vol. 132, no. 9, pp. 4091–4122, 2024.   
[7] M. Blondel, A. Mensch, and J.-P. Vert, “Differentiable divergences between time series,” in International Conference on Artificial Intelligence and Statistics (AISTATS), 2021, pp. 3853–3861.   
[8] M. Dvornik, I. Hadji, K. G. Derpanis, A. Garg, and A. Jepson, “Dropdtw: Aligning common signal between sequences while dropping outliers,” Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 13 782–13 793, 2021.   
[9] B. Su and J.-R. Wen, “Temporal alignment prediction for supervised representation learning and few-shot sequence classification,” in International Conference on Learning Representations (ICLR), 2022.   
[10] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations (ICLR), 2020.   
[11] Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C.-J. Hsieh, “Dynamicvit: Efficient vision transformers with dynamic token sparsification,” Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 13 937–13 949, 2021.   
[12] Q. Chen, L. Wang, P. Koniusz, and T. Gedeon, “Motion meets attention: Video motion prompts,” in Asian Conference on Machine Learning (ACML), 2024, pp. 591–606.   
[13] X. Cai, T. Xu, J. Yi, J. Huang, and S. Rajasekaran, “Dtwnet: A dynamic time warping network,” Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.   
[14] B. K. Iwana, V. Frinken, and S. Uchida, “Dtw-nn: A novel neural network for time series recognition using dynamic alignment between inputs and weights,” Knowledge-Based Systems, vol. 188, p. 104971, 2020.

[15] S. Matsuo, X. Wu, G. Atarsaikhan, A. Kimura, K. Kashino, B. K. Iwana, and S. Uchida, “Deep attentive time warping,” Pattern Recognition, vol. 136, p. 109201, 2023.   
[16] E. F. Montesuma, F. M. N. Mboula, and A. Souloumiac, “Recent advances in optimal transport for machine learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 2, pp. 1161– 1180, 2024.   
[17] M. Cuturi, “Sinkhorn distances: Lightspeed computation of optimal transport,” Advances in Neural Information Processing Systems (NeurIPS), vol. 26, 2013.   
[18] J. Lee, M. Dabagia, E. Dyer, and C. Rozell, “Hierarchical optimal transport for multimodal distribution alignment,” Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.   
[19] H. Janati, M. Cuturi, and A. Gramfort, “Spatio-temporal alignments: Optimal transport through space and time,” in International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR, 2020, pp. 1695–1704.   
[20] S. Salvador and P. Chan, “Toward accurate dynamic time warping in linear time and space,” Intelligent Data Analysis, vol. 11, no. 5, pp. 561–580, 2007.   
[21] P. Senin, “Dynamic time warping algorithm review,” Information and Computer Science Department University of Hawaii at Manoa Honolulu, USA, vol. 855, no. 1-23, p. 40, 2008.   
[22] P.-F. Marteau, “Time warp edit distance with stiffness adjustment for time series matching,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 2, pp. 306–318, 2008.   
[23] E. Hoffer and N. Ailon, “Deep metric learning using triplet network,” in International Workshop on Similarity-Based Pattern Recognition. Springer, 2015, pp. 84–92.   
[24] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superglue: Learning feature matching with graph neural networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4938–4947.   
[25] H. Xiao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu, “Token-label alignment for vision transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 5495–5504.   
[26] A. Jung, S. Hong, and Y. Hyun, “Scale-aware token-matching for transformer-based object detector,” Pattern Recognition Letters, vol. 185, pp. 197–202, 2024.   
[27] J. Cao, P. Ye, S. Li, C. Yu, Y. Tang, J. Lu, and T. Chen, “Madtp: Multimodal alignment-guided dynamic token pruning for accelerating visionlanguage transformer,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 15 710–15 719.   
[28] H. Zhu and P. Koniusz, “Simple spectral graph convolution,” in International Conference on Learning Representations (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=CYO5T-YjWZV   
[29] L. Liu, L. Wang, and X. Liu, “In defense of soft-assignment coding,” in The IEEE International Conference on Computer Vision (ICCV), 2011, pp. 2486–2493.   
[30] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, “The ucr time series archive,” IEEE/CAA Journal of Automatica Sinica, vol. 6, no. 6, pp. 1293–1305, 2019.   
[31] A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, “Ntu rgb+ d: A large scale dataset for 3d human activity analysis,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 1010– 1019.   
[32] J. Liu, A. Shahroudy, M. Perez, G. Wang, L.-Y. Duan, and A. C. Kot, “Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 10, pp. 2684–2701, 2019.   
[33] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev et al., “The kinetics human action video dataset,” arXiv preprint arXiv:1705.06950, 2017.   
[34] O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al., “Matching networks for one shot learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 29, 2016.   
[35] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for fewshot learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.   
[36] L. Bertinetto, J. F. Henriques, P. Torr, and A. Vedaldi, “Meta-learning with differentiable closed-form solvers,” in International Conference on Learning Representations (ICLR), 2019. [Online]. Available: https://openreview.net/forum?id=HyxnZh0ct7   
[37] M. Ren, S. Ravi, E. Triantafillou, J. Snell, K. Swersky, J. B. Tenenbaum, H. Larochelle, and R. S. Zemel, “Meta-learning for semi-supervised few-shot classification,” in International Conference

on Learning Representations (ICLR), 2018. [Online]. Available: https://openreview.net/forum?id=HJcSzz-CZ   
[38] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” 2011.   
[39] A. Khosla, N. Jayadevaprakash, B. Yao, and F.-F. Li, “Novel dataset for fine-grained image categorization: Stanford dogs,” in The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), vol. 2, no. 1, 2011.   
[40] J. Krause, M. Stark, J. Deng, and L. Fei-Fei, “3d object representations for fine-grained categorization,” in The IEEE International Conference on Computer Vision Workshops (ICCVW), 2013, pp. 554–561.   
[41] S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi, “Finegrained visual classification of aircraft,” arXiv preprint arXiv:1306.5151, 2013.   
[42] G. Van Horn, S. Branson, R. Farrell, S. Haber, J. Barry, P. Ipeirotis, P. Perona, and S. Belongie, “Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 595–604.   
[43] X. Yu, Y. Zhao, Y. Gao, X. Yuan, and S. Xiong, “Benchmark platform for ultra-fine-grained visual categorization beyond human performance,” in The IEEE International Conference on Computer Vision (ICCV), 2021, pp. 10 285–10 295.   
[44] B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum, “Human-level concept learning through probabilistic program induction,” Science, vol. 350, no. 6266, pp. 1332–1338, 2015.   
[45] P. Zhang, X. Li, L. Yu, Z. Zhang, F. Dunkin, H. Liu, and Z. Li, “Boosting learning efficiency in few-shot tasks with layer-adaptive pid control,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.   
[46] F. Hao, F. He, L. Liu, F. Wu, D. Tao, and J. Cheng, “Class-aware patch embedding adaptation for few-shot image classification,” in The IEEE International Conference on Computer Vision (ICCV), 2023, pp. 18 905– 18 915.   
[47] P. Li, J. Chen, L. Shang, and C. Ping, “Adaptive saliency based contextual metric learning for few-shot open-set recognition,” Pattern Recognition, p. 113096, 2026.   
[48] K. Hsu, S. Levine, and C. Finn, “Unsupervised learning via metalearning,” in International Conference on Learning Representations (ICLR), 2019. [Online]. Available: https://openreview.net/forum?id= r1My6sR9tX   
[49] S. Khodadadeh, L. Boloni, and M. Shah, “Unsupervised meta-learning for few-shot image classification,” Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.   
[50] S. Khodadadeh, S. Zehtabian, S. Vahidian, W. Wang, B. Lin, and L. Boloni, “Unsupervised meta-learning through latent-space interpolation in generative models,” in International Conference on Learning Representations (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=XOjv2HxIF6i   
[51] H. Jang, H. Lee, and J. Shin, “Unsupervised meta-learning via few-shot pseudo-supervised contrastive learning,” in International Conference on Learning Representations (ICLR), 2023. [Online]. Available: https://openreview.net/forum?id=TdTGGj7fYYJ   
[52] D. B. Lee, D. Min, S. Lee, and S. J. Hwang, “Meta-GMVAE: Mixture of gaussian VAE for unsupervised meta-learning,” in International Conference on Learning Representations (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=wS0UFjsNYjn   
[53] D. Kong, B. Pang, and Y. N. Wu, “Unsupervised meta-learning via latent space energy-based model of symbol vector coupling,” in Fifth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems, 2021. [Online]. Available: https: //openreview.net/forum?id=-pLftu7EpXz   
[54] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, ´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.   
[55] D. C. Liu and J. Nocedal, “On the limited memory bfgs method for large scale optimization,” Mathematical Programming, vol. 45, p. 503–528, 1989.   
[56] G. Zheng, Q. Suo, M. Huai, and A. Zhang, “Learning to learn task transformations for improved few-shot classification,” in Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). SIAM, 2023, pp. 784–792.   
[57] J. Y. Lim, K. M. Lim, C. P. Lee, and Y. X. Tan, “Ssl-protonet: Selfsupervised learning prototypical networks for few-shot learning,” Expert Systems with Applications, vol. 238, p. 122173, 2024.   
[58] H. Zhang, J. Xu, S. Jiang, and Z. He, “Simple semantic-aided few-shot learning,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 28 588–28 597.

[59] J. Xu, X. Pan, J. Wang, W. Pei, Q. Liao, and Z. Xu, “Core: Correlationguided feature enhancement for few-shot image classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 2, pp. 3098–3110, 2024.   
[60] G. Y. Lee, T. Dam, M. M. Ferdaus, D. P. Poenar, and V. N. Duong, “Anrot-helanet: adverserially and naturally robust attention-based aggregation network via the hellinger distance for few-shot classification,” International Journal of Multimedia Information Retrieval, vol. 15, no. 1, p. 8, 2026.   
[61] J. Sun and J. Li, “Few-shot classification with fork attention adapter,” Pattern Recognition, vol. 156, p. 110805, 2024.   
[62] S. Wu, H. Luo, and X. Lin, “Tnpnet: An approach to fewshot open-set recognition via contextual transductive learning,” Neurocomputing, vol. 621, p. 129276, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231224020472   
[63] G. Y. Lee, T. Dam, D. P. Poenar, V. N. Duong, and M. M. Ferdaus, “Hela-vfa: A hellinger distance-attention-based feature aggregation network for few-shot classification,” in The IEEE Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 2173–2183.   
[64] B. Zhang, C. Luo, D. Yu, X. Li, H. Lin, Y. Ye, and B. Zhang, “Metadiff: Meta-learning with conditional diffusion for few-shot learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 2024, pp. 16 687–16 695.   
[65] J. Ren, Y. An, T. Lei, J. Yang, W. Zhang, Z. Pan, Y. Liao, Y. Gao, C. Sun, and W. Zhang, “Adaptive feature selection-based feature reconstruction network for few-shot learning,” Pattern Recognition, p. 112289, 2025.   
[66] Z. Zha, H. Tang, Y. Sun, and J. Tang, “Boosting few-shot fine-grained recognition with background suppression and foreground alignment,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 8, pp. 3947–3961, 2023.   
[67] Y. Dang, M. Sun, M. Zhang, Z. Chen, X. Zhang, Z. Wang, and D. Wang, “Multi-level correlation network for few-shot image classification,” in 2023 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2023, pp. 2909–2914.   
[68] J. Sun, K. Huang, D. Yang, and H. Liu, “Efficient group attentive learning for few-shot image classification,” Expert Systems with Applications, p. 131245, 2026.   
[69] P. Mangla, N. Kumari, A. Sinha, M. Singh, B. Krishnamurthy, and V. N. Balasubramanian, “Charting the right manifold: Manifold mixup for few-shot learning,” in The IEEE Winter Conference on Applications of Computer Vision (WACV), 2020, pp. 2218–2227.   
[70] H. Liu, C. P. Chen, X. Gong, and T. Zhang, “Robust saliency-aware distillation for few-shot fine-grained visual recognition,” IEEE Transactions on Multimedia, vol. 26, pp. 7529–7542, 2024.   
[71] N. Sun and P. Yang, “T2l: Trans-transfer learning for few-shot finegrained visual categorization with extended adaptation,” Knowledge-Based Systems, vol. 264, p. 110329, 2023.   
[72] A. Gulc ¨ u, Z. Kus¸, ¨ ˙I. T. S. Ozkan, and O. F. Karakus¸, “Tripletmaml: ¨ A metric-based model-agnostic meta-learning algorithm for few-shot classification,” Progress in Artificial Intelligence, pp. 1–15, 2026.   
[73] C. Wang, S. Song, Q. Yang, X. Li, and G. Huang, “Fine-grained few shot learning with foreground object transformation,” Neurocomputing, vol. 466, pp. 16–26, 2021.   
[74] Z.-X. Ma, Z.-D. Chen, L.-J. Zhao, Z.-C. Zhang, T. Zheng, X. Luo, and X.-S. Xu, “Bi-directional task-guided network for few-shot fine-grained image classification,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 8277–8286.   
[75] X. Li, J. Wu, Z. Sun, Z. Ma, J. Cao, and J.-H. Xue, “Bsnet: Bisimilarity network for few-shot fine-grained image classification,” IEEE Transactions on Image Processing, vol. 30, pp. 1318–1331, 2020.   
[76] D. Wertheimer, L. Tang, and B. Hariharan, “Few-shot classification with feature map reconstruction networks,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 8012– 8021.   
[77] C. Doersch, A. Gupta, and A. Zisserman, “Crosstransformers: spatiallyaware few-shot transfer,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 21 981–21 993, 2020.   
[78] Z. Ji, R. Wei, J. Liu, Y. Pang, and J. Han, “Interpretable few-shot image classification via prototypical concept-guided mixture of lora experts,” IEEE Transactions on Image Processing, 2026.   
[79] W. Li, J. Xu, J. Huo, L. Wang, Y. Gao, and J. Luo, “Distribution consistency based covariance metric networks for few-shot learning,” in Proceedings of the AAAI conference on Artificial Intelligence, vol. 33, 2019, pp. 8642–8649.   
[80] H. Huang, J. Zhang, J. Zhang, J. Xu, and Q. Wu, “Low-rank pairwise alignment bilinear network for few-shot fine-grained image classification,” IEEE Transactions on Multimedia, vol. 23, pp. 1666–1680, 2020.

[81] R. Zhang, J. Tan, Z. Cao, L. Xu, Y. Liu, L. Si, and F. Sun, “Partaware correlation networks for few-shot learning,” IEEE Transactions on Multimedia, vol. 26, pp. 9527–9538, 2024.

![](images/7e05cbaa20f6a3cfaa4a700157a91f5e5deac6ac556a15c4de9d67b8b753a381.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling young man wearing glasses (no text or symbols visible)
</details>

Lei Wang received his M.E. in Software Engineering from the University of Western Australia (UWA) in 2018 and his Ph.D. in Engineering and Computer Science from the Australian National University (ANU) in 2023. He is a Research Fellow in the School of Electrical and Electronic Engineering at Griffith University and a Visiting Scientist with Data61/CSIRO. He leads the Temporal Intelligence and Motion Extraction (TIME) Lab at Griffith University. He previously held research positions at ANU, UWA, and Data61/CSIRO. His research

focuses on motion-, data-, and model-centric approaches to action recognition and anomaly detection. He has authored numerous first-author papers in top-tier venues, including CVPR, ICCV, ECCV, ACM Multimedia, NeurIPS, ICLR, ICML, AAAI, TPAMI, IJCV, and TIP, and received the Sang Uk Lee Best Student Paper Award at ACCV 2022. He serves as an Area Chair for NeurIPS 2026, ACM Multimedia 2024-2026, ICASSP 2025, and ICPR 2024, and was recognized as an Outstanding Area Chair at ACM Multimedia 2024.

![](images/9b3184cf3d68a8e22a666c6991868bb7775256f96fd145cd84a68b62ba5253d1.jpg)

<details>
<summary>natural_image</summary>

Portrait of a person wearing glasses and a gray shirt (no text or symbols visible)
</details>

visual understanding.

Syuan-Hao Li received the B.S. degree in Computer Science, National Taitung University (NTTU), Taiwan, in 2025. He is currently a Ph.D. pathway student at Griffith University and a research intern at the Temporal Intelligence and Motion Extraction (TIME) Lab. He serves as a workshop coordinator for TIME 2026: the 2nd International Workshop on Transformative Insights in Multi-faceted Evaluation, hosted at the Web Conference (WWW 2026). His research interests include temporal modeling, multimodal intelligence, and fine- and ultra-fine-grained

![](images/4650c389de7c40712fba97d93da939f00e9553ccdb12d4d8439530b74f3dcb5f.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man wearing glasses and a purple shirt (no text or symbols visible)
</details>

Yongsheng Gao received the BSc and MSc degrees in Electronic Engineering from Zhejiang University, China, in 1985 and 1988, respectively, and the PhD degree in Computer Engineering from Nanyang Technological University, Singapore. He is currently a Professor with the School of Engineering and Built Environment, Griffith University, and Director of the ARC Research Hub for Driving Farming Productivity and Disease Prevention, Australia. He was previously the Leader of the Biosecurity Group at the Queensland Research Laboratory, National ICT

Australia (ARC Centre of Excellence), a consultant at Panasonic Singapore Laboratories, and an Assistant Professor at Nanyang Technological University. His research interests include smart farming, machine vision for agriculture, biosecurity, face recognition, biometrics, image retrieval, computer vision, pattern recognition, environmental informatics, and medical imaging. He is a recipient of the 2025 ARC Industry Laureate Fellow.

![](images/065637e2592d8cedccfc00c44690d5f913db6688dff39b194299f260230e5be3.jpg)

<details>
<summary>natural_image</summary>

Portrait of a bald man wearing a blue sweater over a checkered shirt, with no visible text or symbols.
</details>

Piotr Koniusz received the BSc degree in Telecommunications and Software Engineering from Warsaw University of Technology, Poland, in 2004, and the PhD degree in Computer Vision from CVSSP, University of Surrey, U.K., in 2013. He is an Associate Professor in Theoretical ML at the University of New South Wales (UNSW) and a Principal (now Visiting) Researcher with the Machine Learning Research Group, Data61/CSIRO. He was previously a postdoctoral researcher with the LEAR team at INRIA, France. His research interests include rep-

resentation learning (contrastive and self-supervised learning, unlearning), vision-language models, MLLMs, and deep and graph neural networks, as well as Machine Learning Safety. He has received awards including the Sang Uk Lee Best Student Paper Award (ACCV 2022), Runner-up APRS/IAPR Best Student Paper Award (DICTA 2022), and Outstanding Area Chair recognition (ICLR 2021–2023). He served as a Program Chair for NeurIPS 2025 and serves as a Senior Workshop Program Chair for NeurIPS 2026, and a Journal Track Chair for ACML 2026.