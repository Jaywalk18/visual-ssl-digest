# Von Mises-Fisher Mixture Model with Dynamic Shrinkage for Realistic Test-Time Transduction

Jiazhen Huang <sup>1</sup> Zhiming Liu <sup>1</sup> Changhu Wang <sup>2</sup> Wei Ju <sup>3</sup> Ziyue Qiao <sup>4</sup> Xiao Luo <sup>5</sup>

## Abstract

A range of methods aim to enhance the performance of vision-language models (VLMs) at test time. Among them, transduction has emerged as a promising paradigm due to its strong compatibility and efficiency. However, realistic evaluations often involve highly imbalanced class distributions, which cause performance degradation or even collapse. In this work, we systematically revisit transduction from the perspective of penalized likelihood estimation (PLE), showing that PLE with a KL-divergence anchor term naturally yields an adaptive shrinkage behavior between prior anchors and empirical estimates. From this viewpoint, the brittleness of transductive methods can be attributed to the absence of anchoring mechanism and static modeling of the shrinkage strength. Therefore, we propose Mixture of Von Mises-Fisher Models with Dynamic Shrinkage (MOON). MOON is built upon a mixture of von Mises-Fisher distributions to model feature representations on the unit hypersphere. To handle imbalance, MOON dynamically adjusts the shrinkage strength using zero-shot priors at both instance and class levels. Thus, it suppresses unreliable assignments and prevents harmful updates from outlier classes, thereby mitigating negative transfer. MOON is model-agnostic, training-free, and requires no task-specific hyperparameter tuning. Extensive experiments further validate the advantage of MOON in both performance and efficiency.

## 1. Introduction

Vision-language models (VLMs) have achieved remarkable success in the computer vision community. By pre-training on massive image-text datasets, these models have learned strong multimodal representation capabilities that align visual and textual concepts in a shared latent space. Therefore, VLMs generalize effectively to a wide range of downstream tasks and domains, from low-level image classification (Lu et al., 2022) to high-level visual question answering (Liu et al., 2023). This process requires few or even no training data. For example, CLIP (Radford et al., 2021) trains dual encoders through large-scale image-text contrastive learning. During inference, zero-shot predictions can be simply achieved by computing image-text embedding similarities.

Motivated by this, many methods aim to enhance the zeroshot predictive performance of VLMs at test time, where only unlabeled test samples are given. Among them, the most popular online test-time adaptation (TTA) (Dobler¨ et al., 2024; Huang et al., 2026; Chen et al., 2026) methods adjust model behavior on-the-fly using current incoming data streams. In parallel, a closely related line of research, often referred to as transduction or transductive learning (Liu et al., 2020), focuses on exploiting the structure of available unlabeled data to perform joint inference over all test samples within a task<sup>1</sup>. Specifically, transductive methods are typically achieved by performing soft probabilistic clustering at the embedding or logit level. Due to its “black-box” nature, which requires neither access to model internals nor expensive gradient backpropagation, transduction exhibits strong model-agnostic compatibility and high efficiency. Therefore, it has emerged as a particularly promising paradigm for realistic deployment.

However, in realistic test-time scenarios, only small batches of unlabeled samples are available, where the underlying class distributions are often highly imbalanced. For instance, data streams may exhibit strong temporal correlations, or only a few classes may appear within a mini-batch. Nevertheless, most existing methods and benchmarks assume that class marginals are fixed and uniform. As discussed in prior studies (Veilleux et al., 2021), this often leads to degenerate predictions and severely limits their applicability. Our evaluations in Fig. 1 further demonstrate this brittleness that under realistic settings, many transductive and online TTA methods suffer from performance degradation or even collapse. Statistically, the failures of transductive methods may arise as they implicitly treat prediction as a latent-variable estimation process. Under class imbalance, majority classes dominate the statistics, while estimates derived from limited, biased observations of minority classes become unreliable and increasingly deviate over iterations<sup>2</sup>. This ultimately leads to negative transfer (Wang et al., 2019).

![](images/9ba1179b7329e3c830d080fe49cc10768801e97935bd68f47bb24e1c0a5d796c.jpg)  
(a) Batch adaptation with a limited number of effective classes.

![](images/9d5b269f7a994710a2bd12d1029d9abb4a3f2cf05a7ee4e9865869f93d8e4798.jpg)  
(b) Online adaptation under non-i.i.d. data streams.  
Figure 1. Performance comparison on two realistic settings. Existing transductive or online TTA methods suffer from performance degradation or even collapse, while our proposed MOON consistently enhances VLM prediction and outperforms state-of-the-art baselines.

To alleviate this issue, recent studies introduce explicit regularizations to penalize excessive deviation of empirical estimates, among which KL-divergence-based anchor terms have proven effective (Zanella et al., 2025). Based on these observations, transduction can be systematically revisited under a unified penalized likelihood estimation (PLE) formulation, where prior knowledge is incorporated as a statistical anchor. Specifically, we theoretically demonstrate that KL-anchored estimator naturally yields an adaptive shrinkage behavior, where class statistics are updated as a convex combination between prior anchors and empirical estimates. Hence, we attribute the brittleness of existing transductive methods to the following limitations: First, methods without anchoring mechanisms tend to overfit local statistics, which rapidly amplifies noisy assignments and leads to catastrophic collapse; Second, even anchor-based methods typically rely on a static modeling of shrinkage strength that implicitly assumes the reliability of statistics remains constant across samples and classes. As illustrated in Fig. 2(a), this design remains far from optimal in both accuracy and robustness, and inevitably requires task-specific hyperparameter tuning, which is impractical in practice.

Therefore, we propose MOON, a simple yet effective method for realistic test-time transduction. MOON follows the KLanchored PLE objective and models feature representations using a mixture of von Mises-Fisher (vMF) distributions on the unit hypersphere. To robustly handle class imbalance, MOON dynamically adjusts the shrinkage strength using zeroshot priors at both instance and class levels. At the instance level, unreliable assignments are suppressed based on entropy, promoting a more robust label coverage. At the class level, harmful updates from outlier classes are identified and prevented, thus mitigating negative transfer. As a result, MOON enables a fine-grained, fully data-driven shrinkage, and can be seamlessly plugged into existing VLMs for enhancement. Our contribution can be concluded as follows:

![](images/a603491aef5af70bddf3eb3f8265ed2868883c64c1c87e1e7c1e222fc002231b.jpg)  
(a) Anchor term weight ?

![](images/f011ead62741250bc86f9090aa8507b817fb29510d3f3077cee6bde1d83f167e.jpg)  
(b) Accuracy-Runtime Trade-off  
Figure 2. (a) Controlling shrinkage strength with anchor weight α of state-of-the-art method StatA. Such static modeling is suboptimal in accuracy and robustness. (b) Accuracy-Runtime Tradeoff. MOON enables effective and efficient adaptation.

<sup>❶</sup> New Perspective with Theoretical Support. We systematically analyze the brittleness of existing methods under realistic class imbalance. By revisiting transduction from the perspective of penalized likelihood estimation (PLE), we theoretically prove that such estimators inherently exhibit adaptive shrinkage, which allows us to identify two limitations: absence of anchoring mechanism and static shrinkage strength modeling.

<sup>❷</sup> Novel Methodology. We propose MOON, which is based on a mixture of von Mises-Fisher (vMF) distributions. It dynamically adjusts the shrinkage strength using zeroshot priors, which effectively suppresses unreliable assignments and prevents harmful updates from outlier classes,

mitigating negative transfer.

<sup>❸</sup> Empirical Validation. We conduct extensive experiments across 11 datasets under two realistic settings, demonstrating that MOON is: (i) Effective: improving zero-shot CLIP by 13.2% across 10 scenarios on ImageNet, outperforming the strongest baseline by 8.8%; (ii) Efficient: Being training-free, processing thousands of samples within tens of milliseconds, which is merely 3.3% of CLIP’s inference latency; (iii) Practical: Operating under a model-agnostic black-box assumption and requiring no task-specific hyperparameter tuning, ensuring seamless deployment.

## 2. Related Work

Enhancing VLMs at Test-Time. A growing body of work focuses on adapting VLMs at test time to enhance their performance on downstream tasks, where only unlabeled target samples are available. Among them, the most popular testtime adaptation (TTA) methods can be broadly divided into two categories. The first focuses on updating model parameters online via lightweight fine-tuning, such as prompt tuning (Shu et al., 2022; Feng et al., 2023) or adapters (Abdul Samadh et al., 2023). This often requires expensive data augmentation and gradient backpropagation. The second avoids training and instead directly adjusts model outputs by maintaining caches (Zhang et al., 2024a), memories (Zhang et al., 2024b), or distribution modeling (Han et al., 2024) over historical data streams. Our work primarily relates to this category and refers to these methods as online TTA. In parallel, the concept of transduction was originally explored in few-shot learning (Martin et al., 2022), where it aims to exploit the structure of available data to perform joint inference over test samples. When extended to zero-shot learning, this paradigm can be viewed as a subclass within a broader TTA framework. Recent works like TransCLIP (Zanella et al., 2024) and ZLaP (Kalantidis et al., 2024) have investigated transduction for VLMs, while StatA (Zanella et al., 2025) further discusses it under the test-time class imbalance. Despite making progress, several limitations still bottleneck their performance. This motivates our MOON.

Imbalanced Learning in Realistic Scenarios. Most existing TTA methods and benchmarks assume perfectly classbalanced tasks at inference, i.e., the marginal class probabilities are treated as uniform. In contrast, realistic deployment scenarios often exhibit highly imbalanced class distributions, such as sparse, long-tailed, or non-i.i.d. batches (Ochal et al., 2023). In such contexts, the inductive biases of standard TTA methods can be harmful. First, limited and biased statistics might lead models to overfit to locally dominant distributions, resulting in negative transfer. Second, commonly adopted tricks like marginal entropy minimization (Wang et al., 2020) also become counter-productive as they force the model to align with a mismatched uniform prior.

Consequently, both transductive and online TTA methods may suffer from performance degradation or even collapse, as also empirically demonstrated in prior works (Zhao et al., 2023; Veilleux et al., 2021). Recent solutions either mitigate sampling bias with memories (Gong et al., 2022) or introduce statistical regularization to stabilize estimates (Zanella et al., 2025). Our MOON aligns with the latter by adopting a KL-anchored PLE framework with dynamic shrinkage.

## 3. Revisiting Test-Time Transduction

Problem definition. Consider a batch of N test samples $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { N }$ , with the label space consisting of K candidate classes. Let $\theta _ { v } ( \cdot )$ and $\theta _ { t } ( \cdot )$ denote the visual and textual encoders of a pre-trained VLM, respectively. For each class k, we obtain its textual embedding $\mathbf { t } _ { k } = \theta _ { t } ( \mathbf { c } _ { k } ) \in \mathbb { R } ^ { d }$ with a prompt $\mathbf { c } _ { k } \ ( \mathrm { e . g . }$ , “a photo of a [classname]”). Similarly, the visual feature embedding is extracted as $\mathbf { f } _ { i } = \theta _ { v } ( \mathbf { x } _ { i } ) \in \mathbb { R } ^ { d }$ After $\ell _ { 2 } \cdot$ -normalized onto the unit hypersphere $\mathbb { S } ^ { d - 1 }$ , zeroshot predictions are computed via cosine similarity:

$$
\hat {\mathbf {y}} _ {i} = \{\hat {\mathbf {y}} _ {i, k} \} _ {k = 1} ^ {K} \in \Delta_ {K}, \quad \hat {\mathbf {y}} _ {i, k} = \frac {\exp (\mathbf {f} _ {i} ^ {\top} \mathbf {t} _ {k} / \tau)}{\sum_ {j} \exp (\mathbf {f} _ {i} ^ {\top} \mathbf {t} _ {j} / \tau)},\tag{1}
$$

where $\Delta$ denotes the probability simplex, and $\tau$ is a fixed temperature coefficient from pre-training.

Realistic imbalanced settings. Following StatA (Zanella et al., 2025), we consider two realistic settings: (i) Batch adaptation: each batch contains a limited number of effective classes $K _ { \mathrm { e f f } } \left( 1 \le K _ { \mathrm { e f f } } \le \mathrm { m i n } \{ N , K \} \right)$ , where batches are processed independently. (ii) Online adaptation: test samples arrive as a non-i.i.d. data stream, whose temporal correlation is controlled by a Dirichlet parameter ξ. Here, historical batch information is accessible. For details on the sampling for the imbalanced data, please refer to App. C.4.

Penalized likelihood estimation. Recent transductive methods can be broadly conceptualized as a family of soft probabilistic clustering algorithms, which aim to infer latent class assignments and class-conditional distributions over the unlabeled test set. We revisit this process within a penalized likelihood estimation (PLE) formulation that jointly estimates the following variables: (i) Assignment vectors $\mathbf { z } _ { i } ~ = ~ \{ z _ { i , k } \} _ { k = 1 } ^ { K } \in \Delta _ { K }$ , representing the latent posterior class probability within the probability simplex (initialized from $\hat { \mathbf { y } } _ { i } )$ . (ii) Mixture models $\mathbf { M } = \{ \mathbf { M } _ { k } \} _ { k = 1 } ^ { K } .$ where each component ${ { \bf { M } } _ { k } }$ models the feature distribution of class k with a set of statistical parameters (e.g., mean and covariance). The general optimization objective is given by:

$$
\arg \min _ {\mathbf {z}, \mathbf {M}} \mathcal {L} _ {\mathrm{PLE}} = \arg \min _ {\mathbf {z}, \mathbf {M}} \left(- \sum_ {i = 1} ^ {N} \mathbf {z} _ {i} ^ {\top} \log \mathbf {p} _ {i} + \mathcal {R} (\mathbf {z})\right) + \alpha \mathcal {R} (\mathbf {M}).\tag{2}
$$

Here, the first term represents the standard negative loglikelihood (NLL), with $\mathbf { p } _ { i }$ denoting class-conditional likelihoods under M. The terms $\mathcal { R } ( \mathbf { z } )$ and $\mathcal { R } ( \mathbf { M } )$ serve as penalization regularizers for assignments and distribution parameters, respectively. Specifically, $\mathcal { R } ( \mathbf { z } )$ is typically introduced to mitigate the inherent biases of unsupervised clustering, $\mathrm { e . g . }$ , by encouraging smoothness or consistency with priors. $\mathcal { R } ( \mathbf { M } )$ is the key to prevent the model from overfitting to local statistics in realistic class-imbalanced scenarios, where α is a hyperparameter. This is achieved by penalizing the deviation between empirical estimate and the zero-shot prior anchor $\mathbf { M } ^ { \prime }$ via a KL divergence: $\mathcal { R } ( \mathbf { M } ) = \mathrm { K L } ( \mathbf { M } ^ { \prime } | | \mathbf { M } )$ ).

Adaptive anchor shrinkage. The effectiveness of the PLE formulation in realistic class imbalance largely hinges on the KL-based distribution anchor term $\mathcal { R } ( \mathbf { M } )$ . We theoretically demonstrate that KL-anchored PLE naturally yields an adaptive shrinkage behavior, where statistics are encouraged to update between empirical estimates and prior anchors. While recent works are implemented under a standard Gaussian assumption by default, this behavior naturally generalizes to the entire exponential family of distributions.

Formally, consider a K-class latent-variable mixture model with soft assignments $\{ z _ { i , k } \} _ { i = 1 } ^ { N }$ and class-conditional densities from a regular minimal exponential family, i.e.,

$$
p (x \mid \eta) = h (x) \exp \bigl (\eta^ {\top} T (x) - A (\eta) \bigr),\tag{3}
$$

where $\eta$ is the natural parameter, $T ( x )$ represents the sufficient statistic, $A ( \eta )$ is the log-partition function<sup>3</sup>, and $h ( x )$ denotes the base measure. Let $\begin{array} { r } { n _ { k } \ = \ \sum _ { i } z _ { i , k } } \end{array}$ and $\begin{array} { r } { S _ { k } = \sum _ { i } z _ { i , k } T ( x _ { i } ) } \end{array}$ denote the class-wise soft count and sufficient-statistic sum. Given a fixed anchor distribution $q _ { k } ( x ) = p ( x \mid \eta _ { k } ^ { \prime } )$ with mean parameter $\mu _ { k } ^ { \prime } = \nabla A ( \eta _ { k } ^ { \prime } )$ , the KL-anchored PLE optimizes during parameter update:

$$
\min _ {\{\eta_ {k} \}} - \sum_ {i = 1} ^ {N} \sum_ {k = 1} ^ {K} z _ {i, k} \log p (x _ {i} \mid \eta_ {k}) + \alpha \sum_ {k = 1} ^ {K} \mathrm{KL} \bigl (q _ {k} \| p (\cdot \mid \eta_ {k}) \bigr), \alpha > 0.\tag{4}
$$

Then the unique minimizer $\eta _ { k } ^ { \star }$ satisfies the closed-form shrinkage in the mean-parameter space:

$$
\nabla A (\eta_ {k} ^ {\star}) = \frac {S _ {k} + \alpha \mu_ {k} ^ {\prime}}{n _ {k} + \alpha} = \beta_ {k} \hat {\mu} _ {k} + (1 - \beta_ {k}) \mu_ {k} ^ {\prime},\tag{5}
$$

where $\begin{array} { r } { \hat { \mu } _ { k } = \frac { S _ { k } } { n _ { k } } } \end{array}$ is the empirical estimate when $n _ { k } > 0 .$ $\begin{array} { r } { \beta _ { k } = \frac { n _ { k } } { n _ { k } + \alpha } \in [ 0 , 1 ] } \end{array}$ denotes the shrinkage strength. Eq. (5) reveals that the update is a data-driven convex combination between the empirical estimate and the prior anchor, with $\beta _ { k } .$ controlled by soft count $n _ { k }$ and anchor weight $\alpha ,$ , adapting automatically to the amount of evidence available for each class: classes with abundant support approach standard maximum likelihood estimation (MLE), while rare classes remain strongly regularized by the anchor.

Limitation 1: no anchor. Notably, in the absence of the anchor term $( \alpha = 0 )$ , we have $\beta _ { k } \equiv 1$ , and Eq. (5) degenerates to standard MLE. Under realistic class imbalance, this causes estimation overfitting to locally biased statistics dominated by majority classes, which can rapidly amplify errors over iterations and lead to catastrophic collapse.

Limitation 2: bounded but static shrinkage. When $\alpha > 0$ Eq. (5) further implies $\| \nabla A ( \eta _ { k } ^ { \star } ) - \mu _ { k } ^ { \prime } \| = \beta _ { k } \| \hat { \mu } _ { k } - \mu _ { k } ^ { \prime } \|$ linearly bounding the deviation by $\beta _ { k }$ . In particular, if $n _ { k } =$ 0 (outlier classes), the update stays exactly at its anchor, i.e., $\nabla A ( \eta _ { k } ^ { \star } ) = \mu _ { k } ^ { \prime }$ , preventing any harmful deviation. However, we could find that these types of anchor-based methods still employ a static anchor strength $\alpha ,$ , implicitly assuming equal reliability of statistics across instances and classes. Therefore, their performance and robustness may remain suboptimal (as shown in Fig. 2(a)), which inspires our MOON. Detailed proofs are provided in App. D.

## 4. Our Proposed MOON

Since Eq. (5) applies to the entire exponential family, we propose to adopt a mixture of von Mises-Fisher (vMF) (Gopal & Yang, 2014; Hasnat et al., 2017; Govindarajan et al., 2024) distributions for modeling as normalized VLM embeddings are intrinsically constrained to the unit hypersphere. This distribution is commonly regarded as the natural generalization of Gaussian distribution onto the sphere (Martin et al., 2024). Formally, for a d-dimensional unit vector $\mathbf { f } _ { i } \in \mathbb { S } ^ { d - 1 }$ , the probability density function of a vMF component $\nu _ { k } ( \pmb { \mu } _ { k } , \kappa _ { k } )$ is defined as:

$$
p _ {i, k} ^ {\mathrm{vMF}} = p (\mathbf {f} _ {i}; \boldsymbol {\mu} _ {k}, \kappa_ {k}) \propto \mathcal {C} _ {d} (\kappa_ {k}) \exp (\kappa_ {k} \boldsymbol {\mu} _ {k} ^ {\top} \mathbf {f} _ {i}),\tag{6}
$$

where $\pmb { \mu } _ { k } \in \mathbb { S } ^ { d - 1 }$ denotes the mean direction vector, and $\kappa _ { k } \geq 0$ is a scalar concentration parameter measuring the isotropic precision. $\begin{array} { r } { \mathcal { C } _ { d } \big ( \kappa _ { k } \big ) = \frac { \kappa _ { k } ^ { d / 2 - 1 } } { ( 2 \pi ) ^ { d / 2 } I _ { d / 2 - 1 } \big ( \kappa _ { k } \big ) } } \end{array}$ is the normalization constant, derived from the order-ν modified Bessel function of the first kind $I _ { \nu } ( \cdot )$ On the basis of this, the KL divergence between $\nu _ { k }$ and anchor $\mathcal { V } _ { k } ^ { \prime }$ takes: KL $\begin{array} { r } { ( \mathcal { V } _ { k } ^ { \prime } \| \mathcal { V } _ { k } ) = \log \frac { \mathcal { C } _ { d } ( \kappa _ { k } ^ { \prime } ) } { \mathcal { C } _ { d } ( \kappa _ { k } ) } + \kappa _ { k } ^ { \prime } \mathcal { A } _ { d } ( \kappa _ { k } ^ { \prime } ) - \kappa _ { k } \mathcal { A } _ { d } ( \kappa _ { k } ^ { \prime } ) \mu _ { k } ^ { \top } \mu _ { k } ^ { \prime } } \end{array}$ (details in App. E.2), where $\begin{array} { r } { A _ { d } ( \kappa ) = \frac { I _ { d / 2 } ( \kappa ) } { I _ { d / 2 - 1 } ( \kappa ) } } \end{array}$ represents the Bessel function ratio. We initialize the anchor distribution $\mathcal { V } _ { k } ^ { \prime }$ with zero-shot priors leveraged from text:

$$
\boldsymbol {\mu} _ {k} ^ {\prime} = \mathbf {t} _ {k}, \quad \mathcal {A} _ {d} (\boldsymbol {\kappa} _ {k} ^ {\prime}) = 1 - \frac {\sum_ {i} \mathbf {z} _ {i , k} \| \mathbf {f} _ {i} - \boldsymbol {\mu} _ {k} ^ {\prime} \| ^ {2}}{2 \sum_ {i} \mathbf {z} _ {i , k}}.\tag{7}
$$

The derivation of $\boldsymbol { \mathcal { A } } _ { d } ( \boldsymbol { \kappa } _ { k } ^ { \prime } )$ is provided in App. F. Building upon the above formulation, we arrive at the final objective:

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{PLE}} (\mathbf {z}; \boldsymbol {\mu}, \boldsymbol {\kappa}) = \boldsymbol {\gamma} \left(- \sum_ {i = 1} ^ {N} \mathbf {z} _ {i} ^ {\top} \log (\mathbf {p} _ {i} ^ {\mathrm{vMF}}) + \mathcal {R} (\mathbf {z})\right) \\ + \boldsymbol {\alpha} \sum_ {k = 1} ^ {K} \mathrm{KL} \left(\mathcal {V} _ {k} ^ {\prime} \| \mathcal {V} _ {k}\right), \end{array}\tag{8}
$$

$$
\text { where } \quad \mathcal {R} (\mathbf {z}) = - \sum_ {i, j} \omega_ {i j} \mathbf {z} _ {i} ^ {\top} \mathbf {z} _ {j} + \sum_ {i = 1} ^ {N} \mathrm{KL} (\mathbf {z} _ {i} \| \hat {\mathbf {y}} _ {i}).
$$

For $\mathcal { R } ( \mathbf { z } )$ , we choose a widely-adopted combination of a Laplacian regularizer and a text-supervision term (Zanella et al., 2024; 2025), where $\omega _ { i j } = \mathbf { f } _ { i } ^ { \top } \mathbf { f } _ { j }$ denotes feature affinity. The former performs label propagation among nearby samples to encourage smooth assignments, while the latter penalizes deviations from zero-shot predictions. Moreover, we introduce two weights at instance and class level, γ and $_ { \alpha , \beta }$ enabling dynamic adjustment of shrinkage strength. Different from Eq. (2), both weights are driven by priors.

## 4.1. Dynamic Shrinkage for Realistic Class Imbalance

Instance-level adjustment. The first two terms in Eq. (8) actually form a standard MLE objective, which typically treats all test samples equally. However, in realistic scenarios, certain zero-shot predictions may be inherently noisy; estimation biases may also accumulate and propagate over iterations. A natural idea is to employ predictive entropy as a metric for reliability, as it is widely adopted in TTA for model optimization or memory updates (Wang et al., 2020; Karmanov et al., 2024). Therefore, we introduce an entropy-based weight $\gamma _ { i } \in [ 0 , 1 ]$ to dynamically re-weight the contribution of each sample to the MLE objective:

$$
\gamma_ {i} = 1 - \frac {H (\hat {\mathbf {y}} _ {i})}{\log K},\tag{9}
$$

where $\begin{array} { r } { H ( \hat { \mathbf { y } } _ { i } ) = - \sum _ { k = 1 } ^ { K } \hat { y } _ { i , k } } \end{array}$ log $\hat { y } _ { i , k }$ denotes entropy, and log $K$ serves as the normalization factor. Through this mechanism, certain predictions are encouraged, while uncertain or ambiguous ones are suppressed. As explicitly shown in Eq. (14), it serves as a coefficient for assignments $\mathbf { z } _ { i }$ and filters out unreliable samples during the parameter estimation of $\pmb { \mu } _ { k }$ and $\kappa _ { k }$ . In implementation, we update γ<sub>i</sub> with current assignments $\mathbf { z } _ { i }$ for stability, i.e., $\begin{array} { r } { \gamma _ { i } = \dot { 1 } - \frac { H ( \mathbf { z } _ { i } ) } { \log K } } \end{array}$

Class-level adjustment. Class imbalance inherently induces class sparsity, such as $K _ { \mathrm { e f f } } \ll K \mathrm { o r } \xi  0 .$ . More importantly, it’s impossible to identify which classes are effective (i.e., present) within batch, as labels are unavailable at test time. This makes transduction particularly vulnerable to negative transfer from outlier (i.e., absent) classes. Although the distribution anchor in Eq. (8) alleviates this issue by inducing an adaptive shrinkage behavior, it treats all classes equally and lacks dynamic, fine-grained shrinkage strength modeling. Moreover, it operates with a single scalar hyperparameter α, which requires task-specific tuning.

To address this limitation, Partial Domain Adaptation (PDA) (Cao et al., 2018) has provided successful experiences that we can learn from. PDA studies settings in which the target label space forms a subset of the source label space. This also works for our test-time settings, as the effective class set of a given test batch is also a subset of VLM’s predefined candidate class set. Consequently, the absence of classes can be interpreted as a form of source-target label space mismatch, which should be suppressed during adaptation. In PDA, such mismatch is typically quantified with zero-shot prediction confidence, since classes with higher confidence are more likely and frequent to be present in the target domain. Leveraging this insight, we replace the fixed scalar α with class-level dynamic weights $\pmb { \alpha } = \{ \alpha _ { k } \} _ { k = 1 } ^ { K }$

Intuitively, highly confident classes should encourage the parameters $\pmb { \mu } _ { k }$ and $\kappa _ { k }$ to align more closely to empirical estimates, pushing the strength $\begin{array} { r } { \beta _ { k } ~ = ~ \frac { n _ { k } } { n _ { k } + \alpha _ { k } } } \end{array}$ towards 1. Therefore, $\alpha _ { k }$ should be negatively correlated with class confidence. We define $\alpha _ { k }$ to be inversely related to confidence, as this form is widely used in statistical learning to impose regularization on less reliable signals (Zou, 2006):

$$
\alpha_ {k} = \frac {1}{\lambda_ {k}},\tag{10}
$$

where $\lambda _ { k }$ denotes the k-th class confidence derived from zero-shot priors. PDA methods often directly estimate $\lambda _ { k }$ from average confidence. In test-time settings, however, such a design is insufficient, as the effective label set varies across batches. On the one hand, those rare but effective classes, occurring infrequently yet consistently in the data streams, might be confused with truly outlier classes and instead suppress positive transfer. On the other hand, we don’t want to lose the generality under a distribution closer to uniform. For balance, $\lambda _ { k }$ is defined as the geometric mean of the average and the maximum confidence:

$$
\lambda_ {k} = \sqrt {\frac {1}{N} \sum_ {i = 1} ^ {N} \hat {\mathbf {y}} _ {i , k} \odot \max _ {i} \hat {\mathbf {y}} _ {i , k}}.\tag{11}
$$

This mildly sacrifices accuracy under severe class imbalance, but yields a more general solution across broader scenarios.

## 4.2. Optimization Algorithm

Since the proposed PLE objective jointly involves the assignments z and mixture parameters $\{ \mathcal { V } _ { k } ( \pmb { \mu } _ { k } , \kappa _ { k } ) \} _ { k = 1 } ^ { K }$ , we adopt an efficient optimization algorithm following recent works (Zanella et al., 2024). The algorithm follows the Block Successive Minimization (BSUM) framework (Razaviyayn et al., 2013), which alternately updates two blocks of variables via iterative block-coordinate descent on surrogate objectives. This algorithm is also theoretically guaranteed to converge, as shown in App. A. Given that both γ and α are non-negative, all terms in Eq. (8) except the Laplacian regularizer are convex with respect to each block.

Linear approximation w.r.t assignments z. Due to the presence of concave Laplacian regularizer $\begin{array} { r } { \sum _ { i , j } \omega _ { i j } \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } . } \end{array}$ , a closed-form update for z cannot be obtained directly. Therefore, we construct a linear upper bound by replacing this term with its first-order Taylor expansion at current iteration, $\begin{array} { r } { \mathrm { i . e . , } - \sum _ { i } \mathbf { z } _ { i } ^ { \top } \left( \sum _ { j } \omega _ { i j } \mathbf { z } _ { j } ^ { ( t ) } \right) } \end{array}$ , where $\mathbf { z } _ { j } ^ { ( t ) }$ denotes the assignment obtained at iteration t. By minimizing the constructed approximate surrogate objective, we have:

$$
\mathbf {z} _ {i} ^ {(t + 1)} = \frac {\hat {\mathbf {y}} _ {i} \odot \exp (\log \mathbf {p} _ {i} ^ {\mathrm{vMF}} + \sum_ {j} \omega_ {i j} \mathbf {z} _ {j} ^ {(t)})}{(\hat {\mathbf {y}} _ {i} \odot \exp (\log \mathbf {p} _ {i} ^ {\mathrm{vMF}} + \sum_ {j} \omega_ {i j} \mathbf {z} _ {j} ^ {(t)})) ^ {\top} \mathbb {1} _ {K}}.\tag{12}
$$

Detailed derivations are provided in App. G.1. In implementation, we omit the inner-loop optimization required in previous works and perform a single pass per iteration for efficiency. Note that $\gamma$ does not appear in Eq. (12) as it could be canceled out during the derivation of $\mathbf { z } _ { i }$

Closed-form update w.r.t parameters $\pmb { \mu }$ and κ. When fixing z, Eq. (8) becomes strictly convex with respect to the mixture parameters $\pmb { \mu }$ and $\kappa .$ . Therefore, we can derive closed-form updates by setting partial derivatives to zero:

$$
\begin{array}{r} \pmb {\mu} _ {k} = \frac {\sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} \mathbf {f} _ {i} + \alpha_ {k} \mathcal {A} _ {d} (\pmb {\kappa} _ {k} ^ {\prime}) \pmb {\mu} _ {k} ^ {\prime}}{\| \sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} \mathbf {f} _ {i} + \alpha_ {k} \mathcal {A} _ {d} (\pmb {\kappa} _ {k} ^ {\prime}) \pmb {\mu} _ {k} ^ {\prime} \|}, \\ \mathcal {A} _ {d} (\kappa_ {k}) = \frac {\| \sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} \mathbf {f} _ {i} + \alpha_ {k} \mathcal {A} _ {d} (\pmb {\kappa} _ {k} ^ {\prime}) \pmb {\mu} _ {k} ^ {\prime} \|}{\sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} + \alpha_ {k}}. \end{array}\tag{13}
$$

As shown in Sec. 3, we can rewrite the above updates in a more intuitive form. Under the mild assumption $\boldsymbol { \mathcal { A } } _ { d } ( \kappa _ { k } ^ { \prime } )$ ≈ 1, Eq. (13) is equivalent to (proof in App. H):

$$
\boldsymbol {\mu} _ {k} = \frac {\beta_ {k} \boldsymbol {v} _ {k} + (1 - \beta_ {k}) \boldsymbol {\mu} _ {k} ^ {\prime}}{\| \beta_ {k} \boldsymbol {v} _ {k} + (1 - \beta_ {k}) \boldsymbol {\mu} _ {k} ^ {\prime} \|}, \mathcal {A} _ {d} (\kappa_ {k}) = \| \beta_ {k} \boldsymbol {v} _ {k} + (1 - \beta_ {k}) \boldsymbol {\mu} _ {k} ^ {\prime} \|, \tag {14}\tag{14}
$$

where $\begin{array} { r } { { \pmb v } _ { k } \ = \ \frac { \sum _ { i = 1 } ^ { N } \gamma _ { i , k } { \bf z } _ { i , k } { \bf f } _ { i } } { \sum _ { i = 1 } ^ { N } \gamma _ { i , k } { \bf z } _ { i , k } } } \end{array}$ and $\begin{array} { r } { \beta _ { k } ~ = ~ \frac { \sum _ { i = 1 } ^ { N } \gamma _ { i , k } \mathbf { z } _ { i , k } } { \sum _ { i = 1 } ^ { N } \gamma _ { i , k } \mathbf { z } _ { i , k } + \alpha _ { k } } } \end{array}$ This offers an intuitive interpretation of the anchor shrinkage as in Eq. (5). Here, ${ \pmb v } _ { k }$ represents empirical estimates from standard MLE, and $\pmb { \mu } _ { k } ^ { \prime }$ serves as prior anchor. Our proposed adjustments are seamlessly integrated here: $\begin{array} { r } { n _ { k } = \sum _ { i } \gamma _ { i } \mathbf { z } _ { i , k } } \end{array}$ replaces soft count $\textstyle \sum _ { i } \mathbf { z } _ { i , k }$ , ensuring that noisy predictions are suppressed. $\alpha _ { k }$ further penalizes harmful deviations: for effective classes, $\alpha _ { k }  1$ while $n _ { k }$ increases, allowing the model to learn more from data; for outlier classes, $\alpha _ { k }$ dominates $\beta _ { k }$ , forcing the updates to shrink towards the anchor, thereby mitigating negative transfer.

Overall procedure. The overall procedure of MOON is summarized in App. A. The initializations and updates mentioned above directly yield the Bessel function ratio $\boldsymbol { \mathcal { A } } _ { d } ( \kappa _ { k } )$ , which corresponds to the mean resultant length of the vMF distribution $\bar { r } _ { k }$ . We then employ the well-known approximation (Banerjee et al., 2005) to estimate $\kappa _ { k }$ :

$$
\kappa_ {k} \approx \frac {d \bar {r} _ {k} - \bar {r} _ {k} ^ {3}}{1 - \bar {r} _ {k} ^ {2}}, \qquad \bar {r} _ {k} \triangleq \mathcal {A} _ {d} (\kappa_ {k}).\tag{15}
$$

Note that the parameter estimation of vMF mixtures is simpler and more computationally efficient than GMMs, as it uses fewer parameters and avoids the expensive quadratic forms and inversions of $\mathcal { R } ^ { d \times d }$ covariance matrices.

![](images/26635d05d1b8864e1b49ef449efd4e6265768fccddb3d034a655e034b05a2ba8.jpg)

![](images/7c9da1edd5dd59789518826a46d92b5ef54de2ff11d6e738ba6ec7d49080d356.jpg)  
Figure 3. Convergence analysis on ImageNet and DTD. We demonstrate performance curves over iterations for each method.

## 5. Experiments

We evaluate our method in several scenarios under two realistic settings, as defined in Sec. 3. We report the Top-1 accuracy across 11 public fine-grained classification datasets, and adopt CLIP ViT-B/16 as our default VLM backbone. Please see App. C for details on datasets, baselines, prompt templates, and other experimental specifics.

## 5.1. Main Results

Batch adaptation. We first report the results under batch adaptation in Tab. 1(a) and (b), with batch sizes of 64 and 1,000, respectively. The results show that existing transductive methods generally suffer from severe performance degradation under realistic class-imbalance, and most of them even collapse and underperform zero-shot CLIP. While StatA mitigates this issue by introducing anchor term R(M), its performance remains suboptimal. In contrast, our MOON consistently achieves the best average performance across all scenarios, effectively enhancing VLM predictions. Notably, the performance gains of MOON become more pronounced as class imbalance becomes more severe $( \frac { K _ { \mathrm { e f f } } } { \operatorname* { m i n } ( N , K ) }$ decreases). Moreover, MOON delivers the most significant gains on challenging large-scale datasets such as ImageNet, highlighting its superiority in practical applications.

Online adaptation. We further evaluate methods under online adaptation. As shown in Tab. 2, most online TTA methods remain relatively stable across different correlation strengths, without exhibiting performance degradation. Nevertheless, MOON still achieves state-of-the-art performance generally. We observe a slight drop only in the Low scenario, where the class distribution becomes closer to uniform. This can be attributed to the inherent bias of our $\alpha ,$ , as it is designed to favor sparse effective class sets. Similarly, MOON brings better improvements in scenarios with stronger correlations. For example, MOON outperforms StatA by 10.7% on ImageNet in the Separate scenario.

## 5.2. Efficiency Analysis

Runtime. Tab. 3 reports the runtime per batch on the ImageNet dataset. We observe that the CLIP inference, in-

Table 1. Main results for batch adaptation, averaged over 1,000 runs. The best and second-best results are marked in bold and underlined, respectively. We report three scenarios for each batch size 64 and 1,000, with varying range of effective classes $K _ { \mathrm { e f f } }$ . Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.  
(a) Setting where the batch size is 64: Very Low $( 1 { - } 4 K _ { \mathrm { e f f } } ) ,$ , Low (2–10), and Medium (5–25).

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCats</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td><img src="images/08af36a25af99ffd8ff5209eed3b8716873d255bd11c745be79d91cc947d1451.jpg"/></td><td><img src="images/2bc30897dca78cc045c841304ab35bad06fbc2107c835b5f709a2dfed06b0b9f.jpg"/></td><td>UCF101</td><td>Avg.</td></tr><tr><td rowspan="2"></td><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>MTA</td><td> $69.3_{+2.7}$ </td><td> $64.8_{+2.3}$ </td><td> $27.4_{+2.7}$ </td><td> $46.9_{-1.4}$ </td><td> $68.0_{+2.4}$ </td><td> $87.2_{+1.3}$ </td><td> $89.4_{+0.3}$ </td><td> $71.7_{+1.0}$ </td><td> $94.0_{+0.8}$ </td><td> $44.4_{+0.9}$ </td><td> $69.0_{+1.5}$ </td><td> $66.6_{+1.3}$ </td></tr><tr><td rowspan="7">Very Low(1–4)</td><td>Dirichlet</td><td> $79.2_{+12.6}$ </td><td> $75.7_{+13.2}$ </td><td> $28.2_{+3.5}$ </td><td> $47.2_{-1.1}$ </td><td> $68.2_{+2.6}$ </td><td> $88.1_{+2.2}$ </td><td> $87.5_{-1.6}$ </td><td> $71.2_{+0.5}$ </td><td> $88.8_{-4.4}$ </td><td> $50.3_{+6.8}$ </td><td> $69.0_{+1.5}$ </td><td> $68.5_{+3.3}$ </td></tr><tr><td>ZLaP</td><td> $14.5_{-52.1}$ </td><td> $13.0_{-49.5}$ </td><td> $8.4_{-16.3}$ </td><td> $36.6_{-11.7}$ </td><td> $23.7_{-41.9}$ </td><td> $31.9_{-54.0}$ </td><td> $57.0_{-32.1}$ </td><td> $22.4_{-48.3}$ </td><td> $52.4_{-40.8}$ </td><td> $13.0_{-30.5}$ </td><td> $29.2_{-38.3}$ </td><td> $27.5_{-37.8}$ </td></tr><tr><td>GDA-CLIP</td><td> $20.5_{-46.1}$ </td><td> $19.8_{-42.7}$ </td><td> $10.3_{-14.4}$ </td><td> $39.9_{-8.4}$ </td><td> $28.2_{-37.4}$ </td><td> $51.4_{-34.5}$ </td><td> $60.4_{-28.7}$ </td><td> $29.6_{-41.1}$ </td><td> $52.9_{-40.3}$ </td><td> $39.9_{-3.6}$ </td><td> $35.2_{-32.3}$ </td><td> $35.3_{-30.0}$ </td></tr><tr><td>TransCLIP</td><td> $21.6_{-45.0}$ </td><td> $21.1_{-41.4}$ </td><td> $11.6_{-13.1}$ </td><td> $45.1_{-3.2}$ </td><td> $34.7_{-30.9}$ </td><td> $59.2_{-26.7}$ </td><td> $72.4_{-16.7}$ </td><td> $36.4_{-34.3}$ </td><td> $62.3_{-30.9}$ </td><td> $26.1_{-17.4}$ </td><td> $37.7_{-29.8}$ </td><td> $38.9_{-26.3}$ </td></tr><tr><td>ADAPT</td><td> $60.8_{-5.8}$ </td><td> $56.0_{-6.5}$ </td><td> $21.4_{-3.3}$ </td><td> $45.9_{-2.4}$ </td><td> $59.7_{-5.9}$ </td><td> $81.4_{-4.5}$ </td><td> $84.7_{-4.4}$ </td><td> $66.8_{-3.9}$ </td><td> $90.4_{-2.8}$ </td><td> $39.2_{-4.3}$ </td><td> $61.4_{-6.1}$ </td><td> $60.7_{-4.5}$ </td></tr><tr><td>StatA</td><td> $72.9_{+6.3}$ </td><td> $66.0_{+3.5}$ </td><td> $29.3_{+4.6}$ </td><td> $56.8_{+8.5}$ </td><td> $76.2_{+10.6}$ </td><td> $90.3_{+4.4}$ </td><td> $95.5_{+6.4}$ </td><td> $77.6_{+6.9}$ </td><td> $93.0_{-0.2}$ </td><td> $46.1_{+2.6}$ </td><td> $70.2_{+2.7}$ </td><td> $70.4_{+5.1}$ </td></tr><tr><td>MOON</td><td> $82.8_{+16.2}$ </td><td> $77.2_{+14.7}$ </td><td> $32.0_{+7.3}$ </td><td> $53.7_{+5.4}$ </td><td> $78.6_{+13.0}$ </td><td> $96.4_{+10.5}$ </td><td> $96.0_{+6.9}$ </td><td> $77.7_{+7.0}$ </td><td> $94.9_{+1.7}$ </td><td> $55.6_{+12.1}$ </td><td> $75.7_{+8.2}$ </td><td> $74.6_{+9.4}$ </td></tr><tr><td rowspan="7">Low(2–10)</td><td>Dirichlet</td><td> $80.1_{+13.5}$ </td><td> $78.0_{+15.5}$ </td><td> $28.1_{+3.4}$ </td><td> $43.5_{-4.8}$ </td><td> $71.5_{+5.9}$ </td><td> $92.3_{+6.4}$ </td><td> $92.7_{+3.6}$ </td><td> $74.7_{+4.0}$ </td><td> $93.0_{-0.2}$ </td><td> $48.9_{+5.4}$ </td><td> $70.9_{+3.4}$ </td><td> $70.3_{+5.1}$ </td></tr><tr><td>ZLaP</td><td> $19.1_{-47.5}$ </td><td> $19.0_{-43.5}$ </td><td> $12.0_{-12.7}$ </td><td> $46.4_{-1.9}$ </td><td> $27.9_{-37.7}$ </td><td> $43.5_{-42.4}$ </td><td> $66.6_{-22.5}$ </td><td> $31.3_{-39.4}$ </td><td> $60.8_{-32.4}$ </td><td> $22.4_{-21.1}$ </td><td> $38.7_{-28.8}$ </td><td> $35.2_{-30.0}$ </td></tr><tr><td>GDA-CLIP</td><td> $18.6_{-48.0}$ </td><td> $19.4_{-43.1}$ </td><td> $12.6_{-12.1}$ </td><td> $50.3_{+2.0}$ </td><td> $25.2_{-40.4}$ </td><td> $49.7_{-36.2}$ </td><td> $60.4_{-28.7}$ </td><td> $33.1_{-37.6}$ </td><td> $55.2_{-38.0}$ </td><td> $28.4_{-15.1}$ </td><td> $37.2_{-30.3}$ </td><td> $35.5_{-29.8}$ </td></tr><tr><td>TransCLIP</td><td> $20.3_{-46.3}$ </td><td> $22.4_{-40.1}$ </td><td> $14.3_{-10.4}$ </td><td> $53.9_{+5.6}$ </td><td> $30.8_{-34.8}$ </td><td> $55.6_{-30.3}$ </td><td> $69.4_{-19.7}$ </td><td> $40.9_{-29.8}$ </td><td> $64.6_{-28.6}$ </td><td> $31.6_{-11.9}$ </td><td> $40.9_{-26.6}$ </td><td> $40.4_{-24.8}$ </td></tr><tr><td>ADAPT</td><td> $65.3_{-1.3}$ </td><td> $60.7_{-1.8}$ </td><td> $23.5_{-1.2}$ </td><td> $51.5_{+3.2}$ </td><td> $62.3_{-3.3}$ </td><td> $84.5_{-1.4}$ </td><td> $87.5_{-1.6}$ </td><td> $68.8_{-1.9}$ </td><td> $91.0_{-2.2}$ </td><td> $41.8_{-1.7}$ </td><td> $63.8_{-3.7}$ </td><td> $63.7_{-1.5}$ </td></tr><tr><td>StatA</td><td> $72.8_{+6.2}$ </td><td> $66.9_{+4.4}$ </td><td> $27.7_{+3.0}$ </td><td> $51.3_{+3.0}$ </td><td> $73.5_{+7.9}$ </td><td> $89.5_{+3.6}$ </td><td> $93.7_{+4.6}$ </td><td> $76.6_{+5.9}$ </td><td> $93.6_{+0.4}$ </td><td> $46.9_{+3.4}$ </td><td> $69.6_{+2.1}$ </td><td> $69.3_{+4.1}$ </td></tr><tr><td>MOON</td><td> $83.8_{+17.2}$ </td><td> $78.0_{+15.5}$ </td><td> $29.8_{+5.1}$ </td><td> $48.3_{\pm 0.0}$ </td><td> $76.7_{+11.1}$ </td><td> $95.5_{+9.6}$ </td><td> $94.6_{+5.5}$ </td><td> $77.3_{+6.6}$ </td><td> $95.3_{+2.1}$ </td><td> $50.9_{+7.4}$ </td><td> $74.2_{+6.7}$ </td><td> $73.1_{+7.9}$ </td></tr><tr><td rowspan="7">Medium(5–25)</td><td>Dirichlet</td><td> $77.7_{+11.1}$ </td><td> $72.9_{+10.4}$ </td><td> $26.1_{+1.4}$ </td><td> $38.6_{-9.7}$ </td><td> $71.6_{+6.0}$ </td><td> $90.8_{+4.9}$ </td><td> $88.4_{-0.7}$ </td><td> $71.5_{+0.8}$ </td><td> $93.7_{+0.5}$ </td><td> $42.9_{-0.6}$ </td><td> $67.8_{+0.3}$ </td><td> $67.5_{+2.2}$ </td></tr><tr><td>ZLaP</td><td> $29.0_{-37.6}$ </td><td> $27.9_{-34.6}$ </td><td> $16.5_{-8.2}$ </td><td> $49.0_{+0.7}$ </td><td> $36.0_{-29.6}$ </td><td> $59.1_{-26.8}$ </td><td> $76.4_{-12.7}$ </td><td> $42.9_{-27.8}$ </td><td> $72.0_{-21.2}$ </td><td> $32.0_{-11.5}$ </td><td> $50.3_{-17.2}$ </td><td> $44.7_{-20.6}$ </td></tr><tr><td>GDA-CLIP</td><td> $19.2_{-47.4}$ </td><td> $21.4_{-41.1}$ </td><td> $15.8_{-8.9}$ </td><td> $56.2_{+7.9}$ </td><td> $27.4_{-38.2}$ </td><td> $52.9_{-33.0}$ </td><td> $68.7_{-20.4}$ </td><td> $40.1_{-30.6}$ </td><td> $59.2_{-34.0}$ </td><td> $35.2_{-8.3}$ </td><td> $43.7_{-23.8}$ </td><td> $40.0_{-25.3}$ </td></tr><tr><td>TransCLIP</td><td> $15.5_{-51.1}$ </td><td> $22.8_{-39.7}$ </td><td> $17.0_{-7.7}$ </td><td> $58.2_{+9.9}$ </td><td> $32.9_{-32.7}$ </td><td> $56.3_{-29.6}$ </td><td> $72.6_{-16.5}$ </td><td> $45.0_{-25.7}$ </td><td> $65.6_{-27.6}$ </td><td> $37.5_{-6.0}$ </td><td> $46.5_{-21.0}$ </td><td> $42.7_{-22.5}$ </td></tr><tr><td>ADAPT</td><td> $66.8_{+0.2}$ </td><td> $61.7_{-0.8}$ </td><td> $25.0_{+0.3}$ </td><td> $52.8_{+4.5}$ </td><td> $65.4_{-0.2}$ </td><td> $85.9_{\pm 0.0}$ </td><td> $88.7_{-0.4}$ </td><td> $69.7_{-1.0}$ </td><td> $92.2_{-1.0}$ </td><td> $43.5_{\pm 0.0}$ </td><td> $66.7_{-0.8}$ </td><td> $65.3_{+0.1}$ </td></tr><tr><td>StatA</td><td> $70.7_{+4.1}$ </td><td> $65.3_{+2.8}$ </td><td> $26.0_{+1.3}$ </td><td> $45.0_{-3.3}$ </td><td> $71.1_{+5.5}$ </td><td> $88.2_{+2.3}$ </td><td> $90.8_{+1.7}$ </td><td> $73.7_{+3.0}$ </td><td> $93.9_{+0.7}$ </td><td> $47.5_{+4.0}$ </td><td> $69.1_{+1.6}$ </td><td> $67.4_{+2.2}$ </td></tr><tr><td>MOON</td><td> $79.5_{+12.9}$ </td><td> $72.6_{+10.1}$ </td><td> $26.0_{+1.3}$ </td><td> $42.9_{-5.4}$ </td><td> $73.6_{+8.0}$ </td><td> $92.5_{+6.6}$ </td><td> $90.7_{+1.6}$ </td><td> $74.4_{+3.7}$ </td><td> $94.6_{+1.4}$ </td><td> $44.7_{+1.2}$ </td><td> $71.5_{+4.0}$ </td><td> $69.4_{+4.1}$ </td></tr></table>

(b) Setting where the batch size is 1,000: Medium (5–25 $K _ { \mathrm { e f f } } ) ,$ High (25–50), and Very High (50-100).

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td rowspan="2"></td><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>MTA</td><td> $69.3_{+2.7}$ </td><td> $64.8_{+2.3}$ </td><td> $27.4_{+2.7}$ </td><td> $46.9_{-1.4}$ </td><td> $68.0_{+2.4}$ </td><td> $87.2_{+1.3}$ </td><td> $89.4_{+0.3}$ </td><td> $71.7_{+1.0}$ </td><td> $94.0_{+0.8}$ </td><td> $44.4_{+0.9}$ </td><td> $69.0_{+1.5}$ </td><td> $66.6_{+1.3}$ </td></tr><tr><td rowspan="7">Medium(5–25)</td><td>Dirichlet</td><td> $60.9_{-5.7}$ </td><td> $75.4_{+12.9}$ </td><td> $26.7_{+2.0}$ </td><td> $38.8_{-9.5}$ </td><td> $74.1_{+8.5}$ </td><td> $76.2_{-9.7}$ </td><td> $91.0_{+1.9}$ </td><td> $71.6_{+0.9}$ </td><td> $92.4_{-0.8}$ </td><td> $36.2_{-7.3}$ </td><td> $65.4_{-2.1}$ </td><td> $64.4_{-0.8}$ </td></tr><tr><td>ZLaP</td><td> $16.6_{-50.0}$ </td><td> $20.1_{-42.4}$ </td><td> $16.4_{-8.3}$ </td><td> $49.0_{+0.7}$ </td><td> $32.2_{-33.4}$ </td><td> $55.5_{-30.4}$ </td><td> $76.4_{-12.7}$ </td><td> $40.6_{-30.1}$ </td><td> $67.7_{-25.5}$ </td><td> $34.2_{-9.3}$ </td><td> $48.1_{-19.4}$ </td><td> $41.5_{-23.7}$ </td></tr><tr><td>GDA-CLIP</td><td> $32.3_{-34.3}$ </td><td> $32.5_{-30.0}$ </td><td> $19.0_{-5.7}$ </td><td> $61.0_{+12.7}$ </td><td> $39.5_{-26.1}$ </td><td> $68.1_{-17.8}$ </td><td> $79.6_{-9.5}$ </td><td> $50.3_{-20.4}$ </td><td> $69.1_{-24.1}$ </td><td> $39.2_{-4.3}$ </td><td> $49.5_{-18.0}$ </td><td> $49.1_{-16.1}$ </td></tr><tr><td>TransCLIP</td><td> $39.9_{-26.7}$ </td><td> $42.7_{-19.8}$ </td><td> $22.0_{-2.7}$ </td><td> $63.1_{+14.8}$ </td><td> $49.9_{-15.7}$ </td><td> $80.6_{-5.3}$ </td><td> $87.9_{-1.2}$ </td><td> $58.7_{-12.0}$ </td><td> $79.1_{-14.1}$ </td><td> $42.9_{-0.6}$ </td><td> $55.0_{-12.5}$ </td><td> $56.5_{-8.7}$ </td></tr><tr><td>ADAPT</td><td> $55.8_{-10.8}$ </td><td> $52.8_{-9.7}$ </td><td> $23.3_{-1.4}$ </td><td> $63.5_{+15.2}$ </td><td> $55.9_{-9.7}$ </td><td> $77.4_{-8.5}$ </td><td> $85.2_{-3.9}$ </td><td> $67.1_{-3.6}$ </td><td> $89.0_{-4.2}$ </td><td> $43.0_{-0.5}$ </td><td> $62.3_{-5.2}$ </td><td> $61.4_{-3.9}$ </td></tr><tr><td>StatA</td><td> $70.8_{+4.2}$ </td><td> $64.5_{+2.0}$ </td><td> $28.4_{+3.7}$ </td><td> $60.4_{+12.1}$ </td><td> $74.0_{+8.4}$ </td><td> $87.5_{+1.6}$ </td><td> $93.1_{+4.0}$ </td><td> $77.5_{+6.8}$ </td><td> $92.8_{-0.4}$ </td><td> $47.1_{+3.6}$ </td><td> $70.2_{+2.7}$ </td><td> $69.7_{+4.4}$ </td></tr><tr><td>MOON</td><td> $81.6_{+15.0}$ </td><td> $75.5_{+13.0}$ </td><td> $29.6_{+4.9}$ </td><td> $58.9_{+10.6}$ </td><td> $76.1_{+10.5}$ </td><td> $93.1_{+7.2}$ </td><td> $92.4_{+3.3}$ </td><td> $77.3_{+6.6}$ </td><td> $94.9_{+1.7}$ </td><td> $49.0_{+5.5}$ </td><td> $73.6_{+6.1}$ </td><td> $72.9_{+7.7}$ </td></tr><tr><td rowspan="7">High(25–50)</td><td>Dirichlet</td><td> $17.3_{-49.3}$ </td><td> $37.3_{-25.2}$ </td><td> $21.0_{-3.7}$ </td><td> $37.9_{-10.4}$ </td><td> $65.4_{-0.2}$ </td><td> $46.3_{-39.6}$ </td><td> $81.3_{-7.8}$ </td><td> $46.3_{-24.4}$ </td><td> $80.5_{-12.7}$ </td><td> $21.1_{-22.4}$ </td><td> $43.6_{-23.9}$ </td><td> $45.3_{-20.0}$ </td></tr><tr><td>ZLaP</td><td> $23.8_{-42.8}$ </td><td> $32.2_{-30.3}$ </td><td> $22.2_{-2.5}$ </td><td> $49.3_{+1.0}$ </td><td> $45.4_{-20.2}$ </td><td> $74.9_{-11.0}$ </td><td> $86.5_{-2.6}$ </td><td> $56.2_{-14.5}$ </td><td> $79.7_{-13.5}$ </td><td> $43.6_{+0.1}$ </td><td> $60.8_{-6.7}$ </td><td> $52.2_{-13.0}$ </td></tr><tr><td>GDA-CLIP</td><td> $37.8_{-28.8}$ </td><td> $41.5_{-21.0}$ </td><td> $23.8_{-0.9}$ </td><td> $62.4_{+14.1}$ </td><td> $49.2_{-16.4}$ </td><td> $76.1_{-9.8}$ </td><td> $90.9_{+1.8}$ </td><td> $64.7_{-6.0}$ </td><td> $77.9_{-15.3}$ </td><td> $47.4_{+3.9}$ </td><td> $61.7_{-5.8}$ </td><td> $57.6_{-7.6}$ </td></tr><tr><td>TransCLIP</td><td> $43.9_{-22.7}$ </td><td> $49.6_{-12.9}$ </td><td> $24.8_{+0.1}$ </td><td> $64.0_{+15.7}$ </td><td> $57.3_{-8.3}$ </td><td> $83.0_{-2.9}$ </td><td> $91.4_{+2.3}$ </td><td> $69.1_{-1.6}$ </td><td> $85.5_{-7.7}$ </td><td> $47.5_{+4.0}$ </td><td> $65.4_{-2.1}$ </td><td> $62.0_{-3.3}$ </td></tr><tr><td>ADAPT</td><td> $60.4_{-6.2}$ </td><td> $57.9_{-4.6}$ </td><td> $24.9_{+0.2}$ </td><td> $64.0_{+15.7}$ </td><td> $60.7_{-4.9}$ </td><td> $82.1_{-3.8}$ </td><td> $91.8_{+2.7}$ </td><td> $70.2_{-0.5}$ </td><td> $90.2_{-3.0}$ </td><td> $46.9_{+3.4}$ </td><td> $67.0_{-0.5}$ </td><td> $65.1_{-0.1}$ </td></tr><tr><td>StatA</td><td> $71.9_{+5.3}$ </td><td> $66.4_{+3.9}$ </td><td> $25.9_{+1.2}$ </td><td> $60.7_{+12.4}$ </td><td> $73.6_{+8.0}$ </td><td> $88.0_{+2.1}$ </td><td> $91.4_{+2.3}$ </td><td> $76.7_{+6.0}$ </td><td> $93.2_{+0.0}$ </td><td> $47.9_{+4.4}$ </td><td> $71.5_{+4.0}$ </td><td> $69.8_{+4.5}$ </td></tr><tr><td>MOON</td><td> $81.8_{+15.2}$ </td><td> $74.9_{+12.4}$ </td><td> $25.5_{+0.8}$ </td><td> $59.3_{+11.0}$ </td><td> $74.7_{+9.1}$ </td><td> $91.0_{+5.1}$ </td><td> $89.7_{+0.6}$ </td><td> $75.2_{+4.5}$ </td><td> $94.8_{+1.6}$ </td><td> $45.1_{+1.6}$ </td><td> $71.8_{+4.3}$ </td><td> $71.3_{+6.0}$ </td></tr><tr><td rowspan="7">Very High(50–100)</td><td>Dirichlet</td><td> $10.8_{-55.8}$ </td><td> $15.7_{-46.8}$ </td><td> $17.5_{-7.2}$ </td><td> $37.8_{-10.5}$ </td><td> $51.2_{-14.4}$ </td><td> $29.1_{-56.8}$ </td><td> $79.3_{-9.8}$ </td><td> $24.3_{-46.4}$ </td><td> $59.1_{-34.1}$ </td><td> $19.0_{-24.5}$ </td><td> $26.1_{-41.4}$ </td><td> $33.6_{-31.6}$ </td></tr><tr><td>ZLaP</td><td> $32.7_{-33.9}$ </td><td> $44.0_{-18.5}$ </td><td> $25.4_{+0.7}$ </td><td> $49.3_{+1.0}$ </td><td> $55.2_{-10.4}$ </td><td> $83.3_{-2.6}$ </td><td> $87.3_{-1.8}$ </td><td> $64.8_{-5.9}$ </td><td> $87.9_{-5.3}$ </td><td> $45.2_{+1.7}$ </td><td> $67.8_{+0.3}$ </td><td> $58.4_{-6.8}$ </td></tr><tr><td>GDA-CLIP</td><td> $41.7_{-24.9}$ </td><td> $48.5_{-14.0}$ </td><td> $26.3_{+1.6}$ </td><td> $62.4_{+14.1}$ </td><td> $56.5_{-9.1}$ </td><td> $82.8_{-3.1}$ </td><td> $92.3_{-3.2}$ </td><td> $73.1_{+2.4}$ </td><td> $86.5_{-6.7}$ </td><td> $49.2_{+5.7}$ </td><td> $70.5_{+3.0}$ </td><td> $62.7_{-2.5}$ </td></tr><tr><td>TransCLIP</td><td> $44.5_{-22.1}$ </td><td> $53.0_{-9.5}$ </td><td> $25.6_{+0.9}$ </td><td> $64.1_{+15.8}$ </td><td> $60.9_{-4.7}$ </td><td> $85.2_{-0.7}$ </td><td> $91.9_{+2.8}$ </td><td> $74.3_{+3.6}$ </td><td> $90.5_{-2.7}$ </td><td> $48.1_{+4.6}$ </td><td> $70.7_{+3.2}$ </td><td> $64.4_{-0.8}$ </td></tr><tr><td>ADAPT</td><td> $64.7_{-1.9}$ </td><td> $61.9_{-0.6}$ </td><td> $25.8_{+1.1}$ </td><td> $64.0_{+15.7}$ </td><td> $64.9_{-0.7}$ </td><td> $85.5_{-0.4}$ </td><td> $92.6_{+3.5}$ </td><td> $73.2_{+2.5}$ </td><td> $92.5_{-0.7}$ </td><td> $48.0_{+4.5}$ </td><td> $70.3_{+2.8}$ </td><td> $67.6_{+2.3}$ </td></tr><tr><td>StatA</td><td> $71.8_{+5.2}$ </td><td> $67.1_{+4.6}$ </td><td> $23.9_{-0.8}$ </td><td> $60.7_{+12.4}$ </td><td> $70.2_{+4.6}$ </td><td> $87.1_{+1.2}$ </td><td> $91.1_{+2.0}$ </td><td> $74.3_{+3.6}$ </td><td> $93.7_{+0.5}$ </td><td> $48.0_{+4.5}$ </td><td> $70.7_{+3.2}$ </td><td> $69.0_{+3.7}$ </td></tr><tr><td>MOON</td><td> $80.4_{+13.8}$ </td><td> $71.9_{+9.4}$ </td><td> $23.4_{-1.3}$ </td><td> $59.3_{+11.0}$ </td><td> $70.7_{+5.1}$ </td><td> $87.8_{+1.9}$ </td><td> $89.3_{+0.2}$ </td><td> $72.3_{+1.6}$ </td><td> $93.3_{+0.1}$ </td><td> $44.3_{+0.8}$ </td><td> $68.9_{+1.4}$ </td><td> $69.2_{+4.0}$ </td></tr></table>

a single iteration could yield competitive performance.

## 5.3. Ablation Studies

cluding both visual and textual encoding, dominates the total computational cost. Considering the net runtime of methods, our MOON highlights its exceptional efficiency. Despite requiring iterative optimization, MOON is still twice as fast as the single-pass ADAPT. Moreover, this efficiency advantage becomes increasingly pronounced as the batch size grows. This empirically validates a significant advantage of vMF mixtures that has been largely overlooked in prior works. We further provide a complexity analysis in App. A.

Components. We study the effect of key components in Tab. 4. Obviously, iteratively updating parameters $( \mu , \kappa )$ improves performance, as it allows the class-conditional distributions to progressively adapt to the underlying latent structure of data. By dynamically adjusting shrinkage strength, our α robustly mitigates negative transfer from outlier classes, and its benefit scales up as the scenario becomes more imbalanced. Surprisingly, the effect of γ is marginal when batch size is small. A possible explanation is that the estimation error is dominated by high statistical variance and class sparsity in small batches; whereas in large batches, sufficient dense samples provide enough stable statistics for γ to filter out noise. Overall, simply incorporating zero-shot

Convergence. Fig. 3 illustrates the convergence curves of four transductive methods on ImageNet and DTD. We find that our MOON converges rapidly within just a few iterations, maintaining stable and consistent improvements. In contrast, methods without anchoring mechanism (e.g., TransCLIP) tend to deviate from global, reliable estimates as the iteration proceeds, leading to collapse. Notably, even

Table 2. Main results for online adaptation, averaged over 100 runs. The best and second-best results are marked in bold and underlined, respectively. We report four scenarios for batch size 128, with different Dirichlet parameter ξ. Separate denotes sequential classes. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.

<table><tr><td>Scenario</td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCats</td><td>Food101</td><td>Pets</td><td><img src="images/54de295fcd2be8679c497d81870737d6baa76307486c1745ad4aa1613bf2e0b2.jpg"/></td><td><img src="images/2548f5701a6785740a782a7d667b3c9a123ae1efb70e44d7ed78b88c5e581b07.jpg"/></td><td>DTD</td><td>UCFI01</td><td>Avg.</td></tr><tr><td rowspan="2"></td><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>MTA</td><td> $69.3_{+2.7}$ </td><td> $64.8_{+2.3}$ </td><td> $27.4_{+2.7}$ </td><td> $46.9_{-1.4}$ </td><td> $68.0_{+2.4}$ </td><td> $87.2_{+1.3}$ </td><td> $89.4_{+0.3}$ </td><td> $71.7_{+1.0}$ </td><td> $94.0_{+0.8}$ </td><td> $44.4_{+0.9}$ </td><td> $69.0_{+1.5}$ </td><td> $66.6_{+1.3}$ </td></tr><tr><td rowspan="7">Low( $\xi = 0.1$ )</td><td>TENT</td><td> $66.6_{\pm 0.0}$ </td><td> $64.5_{+2.0}$ </td><td> $24.6_{-0.1}$ </td><td> $51.8_{+3.5}$ </td><td> $65.7_{+0.1}$ </td><td>85.9</td><td> $89.3_{+0.2}$ </td><td> $70.6_{-0.1}$ </td><td> $93.4_{+0.2}$ </td><td> $44.0_{+0.5}$ </td><td> $67.8_{+0.3}$ </td><td> $65.8_{+0.6}$ </td></tr><tr><td>TDA</td><td> $68.3_{+1.7}$ </td><td> $66.0_{+3.5}$ </td><td> $25.4_{+0.7}$ </td><td> $60.6_{+12.3}$ </td><td> $66.9_{+1.3}$ </td><td> $86.1_{+0.2}$ </td><td> $89.6_{+0.5}$ </td><td> $72.5_{+1.8}$ </td><td> $93.4_{+0.2}$ </td><td> $45.5_{+2.0}$ </td><td> $71.0_{+3.5}$ </td><td> $67.7_{+2.5}$ </td></tr><tr><td>DMN</td><td> $68.0_{+1.4}$ </td><td> $64.8_{+2.3}$ </td><td> $24.9_{+0.2}$ </td><td> $59.8_{+11.5}$ </td><td> $67.0_{+1.4}$ </td><td> $84.2_{-1.7}$ </td><td> $89.9_{+0.8}$ </td><td> $73.3_{+2.6}$ </td><td> $92.2_{-1.0}$ </td><td> $44.8_{+1.3}$ </td><td> $70.3_{+2.8}$ </td><td> $67.2_{+2.0}$ </td></tr><tr><td>OGA</td><td> $68.8_{+2.2}$ </td><td> $66.1_{+3.6}$ </td><td> $24.9_{+0.2}$ </td><td> $61.8_{+13.5}$ </td><td> $66.2_{+0.6}$ </td><td> $86.2_{+0.3}$ </td><td> $90.2_{+1.1}$ </td><td> $72.0_{+1.3}$ </td><td> $93.4_{+0.2}$ </td><td> $44.9_{+1.4}$ </td><td> $69.4_{+1.9}$ </td><td> $67.6_{+2.4}$ </td></tr><tr><td>ADAPT</td><td> $69.2_{+2.6}$ </td><td> $65.4_{+2.9}$ </td><td> $24.4_{-0.3}$ </td><td> $51.9_{+3.6}$ </td><td> $67.6_{+2.0}$ </td><td> $77.4_{-8.5}$ </td><td> $88.5_{-0.6}$ </td><td> $72.9_{+2.2}$ </td><td> $92.4_{-0.8}$ </td><td> $45.2_{+1.7}$ </td><td> $69.9_{+2.4}$ </td><td> $65.9_{+0.7}$ </td></tr><tr><td>StatA</td><td> $66.2_{-0.4}$ </td><td> $63.6_{+1.1}$ </td><td> $24.3_{-0.4}$ </td><td> $52.3_{+4.0}$ </td><td> $67.4_{+1.8}$ </td><td> $88.0_{+2.1}$ </td><td> $92.5_{+3.4}$ </td><td> $72.7_{+2.0}$ </td><td> $94.2_{+1.0}$ </td><td> $46.8_{+3.3}$ </td><td> $68.8_{+1.3}$ </td><td> $67.0_{+1.7}$ </td></tr><tr><td>MOON</td><td> $67.0_{+0.4}$ </td><td> $63.5_{+1.0}$ </td><td> $23.2_{-1.5}$ </td><td> $50.1_{+1.8}$ </td><td> $66.5_{+0.9}$ </td><td> $91.2_{+5.3}$ </td><td> $91.6_{+2.5}$ </td><td> $71.4_{+0.7}$ </td><td> $94.0_{+0.8}$ </td><td> $45.3_{+1.8}$ </td><td> $68.2_{+0.7}$ </td><td> $66.5_{+1.3}$ </td></tr><tr><td rowspan="7">Medium( $\xi = 0.01$ )</td><td>TENT</td><td> $66.7_{+0.1}$ </td><td> $64.3_{+1.8}$ </td><td> $24.6_{-0.1}$ </td><td> $47.9_{-0.4}$ </td><td> $65.6_{\pm 0.0}$ </td><td> $85.9_{\pm 0.0}$ </td><td> $89.4_{+0.3}$ </td><td> $70.6_{-0.1}$ </td><td> $93.3_{+0.1}$ </td><td> $44.0_{+0.5}$ </td><td> $67.8_{+0.3}$ </td><td> $65.5_{+0.2}$ </td></tr><tr><td>TDA</td><td> $68.2_{+1.6}$ </td><td> $65.6_{+3.1}$ </td><td> $25.2_{+0.5}$ </td><td> $56.5_{+8.2}$ </td><td> $66.5_{+0.9}$ </td><td> $85.8_{-0.1}$ </td><td> $89.3_{+0.2}$ </td><td> $72.6_{+1.9}$ </td><td> $93.5_{+0.3}$ </td><td> $45.2_{+1.7}$ </td><td> $70.1_{+2.6}$ </td><td> $67.1_{+1.9}$ </td></tr><tr><td>DMN</td><td> $68.0_{+1.4}$ </td><td> $64.8_{+2.3}$ </td><td> $24.9_{+0.2}$ </td><td> $56.2_{+7.9}$ </td><td> $66.8_{+1.2}$ </td><td> $81.9_{-4.0}$ </td><td> $89.0_{-0.1}$ </td><td> $73.0_{+2.3}$ </td><td> $92.1_{-1.1}$ </td><td> $44.9_{+1.4}$ </td><td> $69.6_{+2.1}$ </td><td> $66.5_{+1.2}$ </td></tr><tr><td>OGA</td><td> $68.6_{+2.0}$ </td><td> $65.4_{+2.9}$ </td><td> $24.9_{+0.2}$ </td><td> $58.6_{+10.3}$ </td><td> $66.2_{+0.6}$ </td><td> $85.9_{\pm 0.0}$ </td><td> $89.8_{+0.7}$ </td><td> $72.2_{+1.5}$ </td><td> $93.4_{+0.2}$ </td><td> $44.8_{+1.3}$ </td><td> $69.1_{+1.6}$ </td><td> $67.2_{+1.9}$ </td></tr><tr><td>ADAPT</td><td> $69.3_{+2.7}$ </td><td> $67.2_{+4.7}$ </td><td> $24.6_{-0.1}$ </td><td> $58.5_{+10.2}$ </td><td> $68.6_{+3.0}$ </td><td> $83.9_{-2.0}$ </td><td> $90.2_{+1.1}$ </td><td> $73.2_{+2.5}$ </td><td> $92.5_{-0.7}$ </td><td> $45.3_{+1.8}$ </td><td> $71.3_{+3.8}$ </td><td> $67.7_{+2.5}$ </td></tr><tr><td>StatA</td><td> $69.6_{+3.0}$ </td><td> $65.9_{+3.4}$ </td><td> $27.3_{+2.6}$ </td><td> $52.3_{+4.0}$ </td><td> $73.2_{+7.6}$ </td><td> $89.1_{+3.2}$ </td><td> $94.6_{+5.5}$ </td><td> $75.6_{+4.9}$ </td><td> $94.3_{+1.1}$ </td><td> $46.8_{+3.3}$ </td><td> $69.7_{+2.2}$ </td><td> $68.9_{+3.7}$ </td></tr><tr><td>MOON</td><td> $76.5_{+9.9}$ </td><td> $73.9_{+11.4}$ </td><td> $28.3_{+3.6}$ </td><td> $51.5_{+3.2}$ </td><td> $75.1_{+9.5}$ </td><td> $95.3_{+9.4}$ </td><td> $94.8_{+5.7}$ </td><td> $75.7_{+5.0}$ </td><td> $95.3_{+2.1}$ </td><td> $50.0_{+6.5}$ </td><td> $73.4_{+5.9}$ </td><td> $71.8_{+6.6}$ </td></tr><tr><td rowspan="7">High( $\xi = 0.001$ )</td><td>TENT</td><td> $66.8_{+0.2}$ </td><td> $64.3_{+1.8}$ </td><td> $24.8_{+0.1}$ </td><td> $45.6_{-2.7}$ </td><td> $65.6_{\pm 0.0}$ </td><td> $86.1_{+0.2}$ </td><td> $89.4_{+0.3}$ </td><td> $70.5_{-0.2}$ </td><td> $93.4_{+0.2}$ </td><td> $44.0_{+0.5}$ </td><td> $67.9_{+0.4}$ </td><td> $65.3_{+0.1}$ </td></tr><tr><td>TDA</td><td> $67.9_{+1.3}$ </td><td> $65.1_{+2.6}$ </td><td> $25.1_{+0.4}$ </td><td> $55.3_{+7.0}$ </td><td> $66.3_{+0.7}$ </td><td> $85.5_{-0.4}$ </td><td> $89.0_{-0.1}$ </td><td> $72.5_{+1.8}$ </td><td> $93.6_{+0.4}$ </td><td> $45.1_{+1.6}$ </td><td> $69.7_{+2.2}$ </td><td> $66.8_{+1.6}$ </td></tr><tr><td>DMN</td><td> $67.9_{+1.3}$ </td><td> $64.8_{+2.3}$ </td><td> $24.9_{+0.2}$ </td><td> $56.3_{+8.0}$ </td><td> $66.8_{+1.2}$ </td><td> $79.9_{-6.0}$ </td><td> $88.9_{-0.2}$ </td><td> $72.9_{+2.2}$ </td><td> $92.1_{-1.1}$ </td><td> $44.8_{+1.3}$ </td><td> $69.4_{+1.9}$ </td><td> $66.3_{+1.0}$ </td></tr><tr><td>OGA</td><td> $67.9_{+1.3}$ </td><td> $64.6_{+2.1}$ </td><td> $24.9_{+0.2}$ </td><td> $59.0_{+10.7}$ </td><td> $66.2_{+0.6}$ </td><td> $85.6_{-0.3}$ </td><td> $89.7_{+0.6}$ </td><td> $72.2_{+1.5}$ </td><td> $93.5_{+0.3}$ </td><td> $44.7_{+1.2}$ </td><td> $68.9_{+1.4}$ </td><td> $67.0_{+1.8}$ </td></tr><tr><td>ADAPT</td><td> $68.6_{+2.0}$ </td><td> $66.5_{+4.0}$ </td><td> $24.6_{-0.1}$ </td><td> $55.4_{+7.1}$ </td><td> $68.0_{+2.4}$ </td><td> $81.3_{-4.6}$ </td><td> $89.1_{\pm 0.0}$ </td><td> $73.0_{+2.3}$ </td><td> $92.5_{-0.7}$ </td><td> $45.3_{+1.8}$ </td><td> $70.3_{+2.8}$ </td><td> $66.8_{+1.5}$ </td></tr><tr><td>StatA</td><td> $71.9_{+5.3}$ </td><td> $66.0_{+3.5}$ </td><td> $27.9_{+3.2}$ </td><td> $51.8_{+3.5}$ </td><td> $74.7_{+9.1}$ </td><td> $89.3_{+3.4}$ </td><td> $94.8_{+5.7}$ </td><td> $76.4_{+5.7}$ </td><td> $94.4_{+1.2}$ </td><td> $47.0_{+3.5}$ </td><td> $69.8_{+2.3}$ </td><td> $69.5_{+4.2}$ </td></tr><tr><td>MOON</td><td> $82.2_{+15.6}$ </td><td> $77.2_{+14.7}$ </td><td> $30.0_{+5.3}$ </td><td> $51.6_{+3.3}$ </td><td> $77.4_{+11.8}$ </td><td> $95.8_{+9.9}$ </td><td> $95.3_{+6.2}$ </td><td> $77.0_{+6.3}$ </td><td> $95.7_{+2.5}$ </td><td> $51.2_{+7.7}$ </td><td> $74.6_{+7.1}$ </td><td> $73.4_{+8.2}$ </td></tr><tr><td rowspan="7">Separate</td><td>TENT</td><td> $66.7_{+0.1}$ </td><td> $64.2_{+1.7}$ </td><td> $24.7_{\pm 0.0}$ </td><td> $37.0_{-11.3}$ </td><td> $65.6_{\pm 0.0}$ </td><td> $86.1_{+0.2}$ </td><td> $89.3_{+0.2}$ </td><td> $70.8_{+0.1}$ </td><td> $93.4_{+0.2}$ </td><td> $43.9_{+0.4}$ </td><td> $67.9_{+0.4}$ </td><td> $64.5_{-0.7}$ </td></tr><tr><td>TDA</td><td> $67.4_{+0.8}$ </td><td> $64.6_{+2.1}$ </td><td> $24.9_{+0.2}$ </td><td> $55.3_{+7.0}$ </td><td> $65.9_{+0.3}$ </td><td> $85.2_{-0.7}$ </td><td> $88.9_{-0.2}$ </td><td> $72.3_{+1.6}$ </td><td> $93.6_{+0.4}$ </td><td> $45.0_{+1.5}$ </td><td> $69.6_{+2.1}$ </td><td> $66.6_{+1.4}$ </td></tr><tr><td>DMN</td><td> $67.7_{+1.1}$ </td><td> $64.7_{+2.2}$ </td><td> $24.9_{+0.2}$ </td><td> $55.1_{+6.8}$ </td><td> $66.7_{+1.1}$ </td><td> $78.5_{-7.4}$ </td><td> $88.0_{-1.1}$ </td><td> $72.8_{+2.1}$ </td><td> $91.9_{-1.3}$ </td><td> $44.8_{+1.3}$ </td><td> $69.0_{+1.5}$ </td><td> $65.8_{+0.6}$ </td></tr><tr><td>OGA</td><td> $67.2_{+0.6}$ </td><td> $64.1_{+1.6}$ </td><td> $24.9_{+0.2}$ </td><td> $58.0_{+9.7}$ </td><td> $66.1_{+0.5}$ </td><td> $85.4_{-0.5}$ </td><td> $89.5_{+0.4}$ </td><td> $72.1_{+1.4}$ </td><td> $93.4_{+0.2}$ </td><td> $44.6_{+1.1}$ </td><td> $68.7_{+1.2}$ </td><td> $66.7_{+1.5}$ </td></tr><tr><td>ADAPT</td><td> $68.2_{+1.6}$ </td><td> $65.7_{+3.2}$ </td><td> $25.0_{+0.3}$ </td><td> $53.7_{+5.4}$ </td><td> $67.4_{+1.8}$ </td><td> $78.2_{-7.7}$ </td><td> $88.2_{-0.9}$ </td><td> $72.4_{+1.7}$ </td><td> $92.5_{-0.7}$ </td><td> $44.8_{+1.3}$ </td><td> $69.5_{+2.0}$ </td><td> $66.0_{+0.7}$ </td></tr><tr><td>StatA</td><td> $71.7_{+5.1}$ </td><td> $64.9_{+2.4}$ </td><td> $28.9_{+4.2}$ </td><td> $48.2_{-0.1}$ </td><td> $75.2_{+9.6}$ </td><td> $88.9_{+3.0}$ </td><td> $95.2_{+6.1}$ </td><td> $77.6_{+6.9}$ </td><td> $94.3_{+1.1}$ </td><td> $45.8_{+2.3}$ </td><td> $69.0_{+1.5}$ </td><td> $69.1_{+3.8}$ </td></tr><tr><td>MOON</td><td> $82.4_{+15.8}$ </td><td> $76.7_{+14.2}$ </td><td> $32.1_{+7.4}$ </td><td> $51.2_{+2.9}$ </td><td> $77.8_{+12.2}$ </td><td> $95.5_{+9.6}$ </td><td> $95.1_{+6.0}$ </td><td> $77.9_{+7.2}$ </td><td> $95.9_{+2.7}$ </td><td> $53.6_{+10.1}$ </td><td> $74.6_{+7.1}$ </td><td> $73.9_{+8.6}$ </td></tr></table>

Table 3. Runtime (seconds) on ImageNet. The second row indicates the number of iterations. “CLIP forward” denotes the inference latency, while others report the net algorithm time.

<table><tr><td rowspan="2">Batch size</td><td rowspan="2">CLIP forward</td><td>TransCLIP</td><td>StatA</td><td>ADAPT</td><td>MOON</td></tr><tr><td>10</td><td>10</td><td>1</td><td>10</td></tr><tr><td>128</td><td>8.47</td><td>0.08</td><td>0.11</td><td>0.06</td><td>0.03</td></tr><tr><td>1,000</td><td>8.99</td><td>0.24</td><td>0.31</td><td>0.13</td><td>0.03</td></tr><tr><td>50,000</td><td>36.90</td><td>0.47</td><td>0.59</td><td>0.21</td><td>0.05</td></tr></table>

Table 4. Ablation study on components. Each reported performance is averaged over all datasets and scenarios.

Table 5. Design choices of $\alpha _ { k } .$ Results are reported on batch adaptation, Medium scenario, batch size of 1,000. ”Average” denotes the mean performance over all datasets.

<table><tr><td> $\alpha_{k}$ </td><td> $1/\lambda_{k}$  (ours)</td><td> $\lambda_{k}$ </td><td> $\exp(-\lambda_{k})$ </td></tr><tr><td>ImageNet</td><td>81.6</td><td>68.5</td><td>71.1</td></tr><tr><td>Average</td><td>72.9</td><td>67.6</td><td>68.8</td></tr></table>

<table><tr><td> $\alpha$ </td><td> $\gamma$ </td><td>Update  $\mu$ </td><td>Update  $\kappa$ </td><td>Bs=64</td><td>Bs=1,000</td></tr><tr><td>X</td><td>√</td><td>√</td><td>√</td><td>69.0</td><td>68.9</td></tr><tr><td>√</td><td>X</td><td>√</td><td>√</td><td>72.3</td><td>70.4</td></tr><tr><td>√</td><td>√</td><td>X</td><td>√</td><td>66.8</td><td>66.5</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>70.8</td><td>68.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>72.4</td><td>71.1</td></tr></table>

update remain noticeably influenced by empirical estimates. These results further support our inverse design $\alpha _ { k } = 1 / \lambda _ { k }$  
priors for adjustments enables more fine-grained shrinkage in realistic scenarios, with negligible additional cost.

We then examine the design of class confidence $\lambda _ { k }$ in Tab. 6, using three representative tasks from both extreme and mild imbalance scenarios. Average-only confidence may suppress rare-but-valid classes, while max-only confidence may overestimate classes with occasional high-confidence predictions. Various types of means mitigate this issue similarly. We choose geometric mean as it yields a more generalizable solution across broader scenarios, especially those mild, near-uniform cases where MOON performs below average.

## 5.4. Further Analysis

Design choices of α and λ. We first analyze the design of the anchor weight α<sub>k</sub> in Eq. (10). As shown in Tab. 5, directly setting $\alpha _ { k } = \lambda _ { k }$ performs poorly, since it assigns stronger shrinkage to more confident classes. We also compare an alternative form $\alpha _ { k } = \exp ( - \lambda _ { k } )$ , but find it less effective. A possible reason is that it bounds $\alpha _ { k }$ in (0, 1], and in turn bounds $\beta _ { k }$ within $[ n _ { k } / ( n _ { k } + 1 ) , 1 )$ , making the

Comparisons with variants. Although vMF distribution is topologically native to the hypersphere of normalized VLM embeddings, Tab. 7 shows that replacing the GMM in StatA with a vMF mixture unexpectedly underperforms standard StatA. We explain this with the linear representation hypothesis (Park et al., 2023), which suggests that VLM representations reside in low-dimensional linear subspaces and exhibit strong anisotropy. Consequently, constrained by isotropic scalar $\kappa ,$ vMF struggles to capture such manifold geometry, while Gaussian covariance implicitly approximates it in Euclidean space. However, in MOON, this limitation is largely alleviated with our dynamic shrinkage strength modeling: MOON achieves comparable performance to its Gaussian variant with much higher efficiency.

Table 6. Design choices of $\lambda _ { k } ,$ across extreme and mild imbalance scenarios. Each reported performance is averaged over all datasets.

<table><tr><td> $\lambda_k$ </td><td>Task 1</td><td>Task 2</td><td>Task 3</td><td>Avg.</td></tr><tr><td colspan="5">Extreme imbalance</td></tr><tr><td>Geo. mean (ours)</td><td>73.1</td><td>72.9</td><td>73.4</td><td>73.1</td></tr><tr><td>Ari. mean</td><td>73.5</td><td>74.2</td><td>74.1</td><td>73.9</td></tr><tr><td>Harm. mean</td><td>73.4</td><td>73.5</td><td>73.8</td><td>73.6</td></tr><tr><td>Average only</td><td>72.4</td><td>73.3</td><td>73.9</td><td>73.2</td></tr><tr><td>Max only</td><td>71.7</td><td>70.8</td><td>71.6</td><td>71.4</td></tr><tr><td colspan="5">Mild imbalance</td></tr><tr><td>Geo. mean (ours)</td><td>69.4</td><td>69.2</td><td>66.5</td><td>68.4</td></tr><tr><td>Ari. mean</td><td>69.0</td><td>68.6</td><td>65.7</td><td>67.8</td></tr><tr><td>Harm. mean</td><td>69.3</td><td>69.0</td><td>66.1</td><td>68.1</td></tr><tr><td>Average only</td><td>68.2</td><td>67.1</td><td>66.7</td><td>67.3</td></tr><tr><td>Max only</td><td>69.0</td><td>68.9</td><td>66.9</td><td>68.3</td></tr></table>

Table 7. Comparisons with variants. Each reported performance is averaged over all datasets and scenarios.

<table><tr><td rowspan="2">Method</td><td colspan="2">Batch</td><td>Online</td></tr><tr><td>Bs=64</td><td>Bs=1,000</td><td>Bs=128</td></tr><tr><td>StatA</td><td>69.01</td><td>69.46</td><td>68.61</td></tr><tr><td>StatA_vMF</td><td>68.69</td><td>68.15</td><td>68.31</td></tr><tr><td>MOON_Gaussian</td><td>71.49</td><td>71.07</td><td>71.23</td></tr><tr><td>MOON</td><td>72.36</td><td>71.14</td><td>71.42</td></tr></table>

Extend MOON to sample-wise. While online TTA methods usually assume access to only a single image at each step<sup>4</sup>, vanilla MOON does not support sample-wise mode, as transductive learning inherently requires a batch of samples to perform probabilistic soft clustering. However, MOON can be easily extended to a sample-wise online mode by introducing a memory bank to collect historical samples, as what recent work (Zhang et al., 2025) has done.

Formally, assume a memory bank $B = ( \mathbf { f } _ { j } , \bar { \mathbf { z } } _ { j } )$ , where $\mathbf { f } _ { j } ~ \in ~ \hat { S ^ { d - 1 } }$ is cached historical feature, and $\bar { \mathbf { z } } _ { j } ~ \in ~ \Delta ^ { K }$ is the corresponding soft label. Consider a newly arriving sample with normalized feature $f _ { \star }$ and zero-shot prediction $\hat { \mathbf { y } } _ { \star }$ . Assume that current mixture parameters $\mathbf { \bar { \boldsymbol { M } } } \boldsymbol { B } = \{ ( \pmb { \mu } _ { k } ^ { B } , \kappa _ { k } ^ { B } ) \} _ { k = 1 } ^ { K }$ have already been estimated from available data in B. Therefore, the PLE objective becomes<sup>5</sup>:

Table 8. Extend MOON to sample-wise. Results are reported on online adaptation, Medium scenario, batch size of 128.

<table><tr><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Food101</td><td>DTD</td></tr><tr><td>CLIP</td><td>66.6</td><td>62.5</td><td>85.9</td><td>43.5</td></tr><tr><td>StatA</td><td>69.6</td><td>65.9</td><td>89.1</td><td>46.8</td></tr><tr><td>MOON</td><td>76.5</td><td>73.9</td><td>95.3</td><td>50.0</td></tr><tr><td> $MOON^†$ </td><td>73.0</td><td>69.5</td><td>92.4</td><td>44.1</td></tr></table>

Table 9. Scaling to other CLIP backbones and VLMs.

<table><tr><td rowspan="2"></td><td rowspan="2">#Params</td><td colspan="2">ImageNet</td><td colspan="2">Average</td></tr><tr><td>Zero-shot</td><td>Ours</td><td>Zero-shot</td><td>Ours</td></tr><tr><td>CLIP RN50</td><td>102M</td><td>58.2</td><td>76.8 +18.6</td><td>58.7</td><td>65.7 +7.0</td></tr><tr><td>CLIP RN101</td><td>120M</td><td>61.3</td><td>77.1 +15.8</td><td>59.5</td><td>64.8 +5.3</td></tr><tr><td>CLIP ViT-B/32</td><td>151M</td><td>62.0</td><td>78.5 +16.5</td><td>61.9</td><td>68.0 +6.1</td></tr><tr><td>CLIP ViT-L/14</td><td>428M</td><td>73.5</td><td>84.9 +11.4</td><td>72.6</td><td>77.4 +4.8</td></tr><tr><td>OpenCLIP</td><td>150M</td><td>73.0</td><td>83.4 +10.4</td><td>72.5</td><td>76.7 +4.2</td></tr><tr><td>SigLIP</td><td>878M</td><td>82.3</td><td>91.3 +9.0</td><td>81.8</td><td>85.1 +3.2</td></tr><tr><td>EVA-CLIP</td><td>1.14B</td><td>78.0</td><td>81.2 +3.2</td><td>76.5</td><td>78.4 +1.9</td></tr></table>

$$
\min _ {\mathbf {z} _ {\star} \in \Delta^ {K}} - \mathbf {z} _ {\star} ^ {\top} \log \mathbf {p} _ {\star} ^ {\mathcal {B}} - \sum_ {j \in \mathcal {B}} \omega_ {\star_ {j}}, \mathbf {z} _ {\star} ^ {\top} \bar {\mathbf {z}} _ {j} + \mathrm{KL} (\mathbf {z} _ {\star} \| \hat {\mathbf {y}} _ {\star}).\tag{16}
$$

Solving the problem yields the closed-form predictor:

$$
\mathbf {z} _ {\star} = \frac {\hat {\mathbf {y}} _ {\star} \odot \exp (\log \mathbf {p} _ {\star} ^ {\mathcal {B}} + \sum_ {j \in \mathcal {B}} \omega_ {\star j} \bar {\mathbf {z}} _ {j})}{(\hat {\mathbf {y}} _ {\star} \odot \exp (\log \mathbf {p} _ {\star} ^ {\mathcal {B}} + \sum_ {j \in \mathcal {B}} \omega_ {\star j} \bar {\mathbf {z}} _ {j})) ^ {\top} \mathbb {1} _ {K}}.\tag{17}
$$

Hence, MOON can make an immediate prediction for a newly arriving sample without rerunning full-batch transduction.

At the same time, using bank statistics, the mixture parameters continue to be stabilized by the KL-anchored dynamic shrinkage mechanism. Specifically, the parameter update keeps the same anchored form as Eq. (14):

$$
\begin{array}{r} \pmb {\mu} _ {k} ^ {\mathcal {B}} = \frac {\beta_ {k} ^ {\mathcal {B}} \pmb {v} _ {k} ^ {\mathcal {B}} + (1 - \beta_ {k} ^ {\mathcal {B}}) \pmb {\mu} _ {k} ^ {\prime}}{\| \beta_ {k} ^ {\mathcal {B}} \pmb {v} _ {k} ^ {\mathcal {B}} + (1 - \beta_ {k} ^ {\mathcal {B}}) \pmb {\mu} _ {k} ^ {\prime} \|}, \\ \mathcal {A} _ {d} (\kappa_ {k} ^ {\mathcal {B}}) = \| \beta_ {k} ^ {\mathcal {B}} \pmb {v} _ {k} ^ {\mathcal {B}} + (1 - \beta_ {k} ^ {\mathcal {B}}) \pmb {\mu} _ {k} ^ {\prime} \|, \end{array}\tag{18}
$$

where $\begin{array} { r } { \pmb { v } _ { k } ^ { B } = \frac { \sum _ { j \in { \cal { B } } } \gamma _ { j , k } \bar { \pmb { z } } _ { j , k } \mathbf { f } _ { j } } { \sum _ { j \in { \cal { B } } } \gamma _ { j , k } \bar { \pmb { z } } _ { j , k } } } \end{array}$ and $\begin{array} { r } { \beta _ { k } ^ { B } = \frac { \sum _ { j \in \{ 3 } } \gamma _ { j , k } \bar { \bf z } _ { j , k } } { \sum _ { j \in \{ 3 } \} \gamma _ { j , k } \bar { \bf z } _ { j , k } + \alpha _ { k } } .  \end{array}$ According to this, we establish a sample-wise version MOON<sup>†</sup> in Tab. 8. As shown, MOON<sup>†</sup> could still achieve excellent performance. In practice, cold-starting the memory bank B with a held-out set may lead to better results.

More backbones and architectures. We further extend our evaluation to include 4 additional CLIP backbones and 3 other VLMs in Tab. 9, which demonstrate the universal effectiveness of our MOON across diverse model architectures, scales, and types. App. I.2 and I.3 provide more details.

## 6. Conclusion

In this work, we systematically revisit test-time transduction for VLMs under realistic class imbalance from the perspective of PLE, and reveal the brittleness and underlying limitations of existing transductive methods. Therefore, we propose MOON, which is based on a mixture of vMF distributions and dynamically adjusts shrinkage strength at both the instance and class levels to mitigate negative transfer. Extensive experiments validate that MOON achieves effective, efficient and practical adaptation.

## Acknowledgment

Ziyue Qiao is supported by the National Natural Science Foundation of China (No. 62406056) and the Guangdong Basic and Applied Basic Research Foundation (No. 2024A1515140114). The authors are grateful to the anonymous reviewers for their efforts and insightful suggestions to improve this paper.

## Impact Statement

This work advances test-time adaptation for vision-language models by addressing realistic class imbalance and distribution shifts. Our approach leverages a mixture of von Mises-Fisher distributions with dynamic, data-driven shrinkage to suppress unreliable predictions and mitigate negative transfer. We provide theoretical grounding through penalized likelihood estimation, and demonstrate empirical effectiveness and efficiency across multiple datasets and backbones. By enabling robust and efficient adaptation, our method improves the reliability of deployed vision-language systems in practical applications such as image classification, retrieval, and zero-shot reasoning, without requiring access to model weights or additional training data.

## References

Abdul Samadh, J., Gani, M. H., Hussein, N., Khattak, M. U., Naseer, M. M., Shahbaz Khan, F., and Khan, S. H. Align your prompts: Test-time prompting with distribution alignment for zero-shot generalization. Advances in Neural Information Processing Systems, 36: 80396–80413, 2023.

Banerjee, A., Dhillon, I. S., Ghosh, J., Sra, S., and Ridgeway, G. Clustering on the unit hypersphere using von mises-fisher distributions. Journal of Machine Learning Research, 6(9), 2005.

Bossard, L., Guillaumin, M., and Van Gool, L. Food-101– mining discriminative components with random forests. In European conference on computer vision, pp. 446–461. Springer, 2014.

Cao, Z., Ma, L., Long, M., and Wang, J. Partial adversarial domain adaptation. In Proceedings of the European conference on computer vision (ECCV), pp. 135–150, 2018.

Chen, X., Huang, J., Liu, Z., Jiang, Q., Huang, F., Jiang, J., and Wang, Z. Test-time distillation for continual model adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7593– 7604, 2026.

Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2818–2829, 2023.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3606–3613, 2014.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dobler, M., Marsden, R. A., Raichle, T., and Yang, B. A lost¨ opportunity for vision-language models: a comparative study of online test-time adaptation for vision-language models. In European Conference on Computer Vision, pp. 117–133. Springer, 2024.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.

Fei-Fei, L., Fergus, R., and Perona, P. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In 2004 conference on computer vision and pattern recognition workshop, pp. 178–178. IEEE, 2004.

Feng, C.-M., Yu, K., Liu, Y., Khan, S., and Zuo, W. Diverse data augmentation with diffusions for effective test-time prompt tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2704–2714, 2023.

Fuchs, C., Zanella, M., and De Vleeschouwer, C. Online gaussian test-time adaptation of vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 128–137, 2025.

Gong, T., Jeong, J., Kim, T., Kim, Y., Shin, J., and Lee, S.- J. NOTE: Robust continual test-time adaptation against temporal correlation. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Gopal, S. and Yang, Y. Von mises-fisher clustering models. In International conference on machine learning, pp. 154– 162. PMLR, 2014.

Govindarajan, H., Siden, P., Roll, J., and Lindsten, F. Dino´ as a von mises-fisher mixture model. arXiv preprint arXiv:2405.10939, 2024.

Han, Z., Yang, J., Wang, G., Li, J., Xu, Q., Shou, M. Z., and Zhang, C. Dota: Distributional test-time adaptation of vision-language models. arXiv preprint arXiv:2409.19375, 2024.

Hasnat, M. A., Bohne, J., Milgram, J., Gentric, S., and´ Chen, L. von mises-fisher mixture model-based deep learning: Application to face verification. arXiv preprin arXiv:1706.04264, 2017.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Helber, P., Bischke, B., Dengel, A., and Borth, D. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.

Huang, J., Chen, X., Liu, Z., Sun, Y., Jiang, J., and Wang, Z. What drives test-time adaptation for clip? a controlled empirical study from an update perspective. arXiv preprint arXiv:2606.14299, 2026.

Ilharco, G., Wortsman, M., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., et al. Openclip. Zenodo, 2021.

Kalantidis, Y., Tolias, G., et al. Label propagation for zeroshot classification with vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23209–23218, 2024.

Karmanov, A., Guan, D., Lu, S., El Saddik, A., and Xing, E. Efficient test-time adaptation of vision-language models. The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pp. 554–561, 2013.

Li, Q., Liu, Z., Luo, W., Luo, T., and Hou, C. Correcting visual blur induced by attention distraction to reduce hallucinations: Algorithm and theory. In Forty-third International Conference on Machine Learning, 2026a.

Li, Q., Liu, Z., Xu, T., Luo, T., and Hou, C. Adaptive disentangled representation learning for incomplete multi-view multi-label classification. arXiv preprint arXiv:2601.05785, 2026b.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Liu, J., Song, L., and Qin, Y. Prototype rectification for few-shot learning. In European conference on computer vision, pp. 741–756. Springer, 2020.

Liu, Z., Wei, Y., Feng, L., Su, X., Xia, X., Guan, W., Xie, Z., and Yang, S. Do all individual layers help? an empirical study of task-interfering layers in vision-language models. arXiv preprint arXiv:2602.01167, 2026.

Lozier, D. W. Nist digital library of mathematical functions. Annals of Mathematics and Artificial Intelligence, 38(1): 105–119, 2003.

Lu, Y., Liu, J., Zhang, Y., Liu, Y., and Tian, X. Prompt distribution learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5206–5215, 2022.

Maji, S., Rahtu, E., Kannala, J., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

Martin, S., Boudiaf, M., Chouzenoux, E., Pesquet, J.-C., and Ayed, I. Towards practical few-shot query sets: Transductive minimum description length inference. Advances in Neural Information Processing Systems, 35:34677– 34688, 2022.

Martin, S., Huang, Y., Shakeri, F., Pesquet, J.-C., and Ben Ayed, I. Transductive zero-shot and few-shot clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28816–28826, 2024.

Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pp. 722–729. IEEE, 2008.

Ochal, M., Patacchiola, M., Vazquez, J., Storkey, A., and Wang, S. Few-shot learning with class imbalance. IEEE Transactions on Artificial Intelligence, 4(5):1348–1358, 2023.

Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658, 2023.

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pp. 3498–3505. IEEE, 2012.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Razaviyayn, M., Hong, M., and Luo, Z.-Q. A unified convergence analysis of block successive minimization methods for nonsmooth optimization. SIAM Journal on Optimization, 23(2):1126–1153, 2013.

Shu, M., Nie, W., Huang, D.-A., Yu, Z., Goldstein, T., Anandkumar, A., and Xiao, C. Test-time prompt tuning for zero-shot generalization in vision-language models. Advances in Neural Information Processing Systems, 35: 14274–14289, 2022.

Soomro, K., Zamir, A. R., and Shah, M. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

Sun, Q., Fang, Y., Wu, L., Wang, X., and Cao, Y. Evaclip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023.

Veilleux, O., Boudiaf, M., Piantanida, P., and Ben Ayed, I. Realistic evaluation of transductive few-shot learning. Advances in Neural Information Processing Systems, 34: 9290–9302, 2021.

Wang, D., Shelhamer, E., Liu, S., Olshausen, B., and Darrell, T. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726, 2020.

Wang, Z., Dai, Z., Poczos, B., and Carbonell, J. Characteriz-´ ing and avoiding negative transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11293–11302, 2019.

Wang, Z., Liang, J., Sheng, L., He, R., Wang, Z., and Tan, T. A hard-to-beat baseline for training-free clip-based adaptation. arXiv preprint arXiv:2402.04087, 2024.

Xiao, J., Hays, J., Ehinger, K. A., Oliva, A., and Torralba, A. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pp. 3485–3492. IEEE, 2010.

Yuan, L., Xie, B., and Li, S. Robust test-time adaptation in dynamic scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15922–15932, 2023.

Zanella, M. and Ben Ayed, I. On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23783–23793, 2024.

Zanella, M., Gerin, B., and Ayed, I. Boosting vision-´ language models with transduction. Advances in Neural Information Processing Systems, 37:62223–62256, 2024.

Zanella, M., Fuchs, C., De Vleeschouwer, C., and Ben Ayed, I. Realistic test-time adaptation of vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 25103–25112, 2025.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.

Zhang, R., Zhang, W., Fang, R., Gao, P., Li, K., Dai, J., Qiao, Y., and Li, H. Tip-adapter: Training-free adaption of clip for few-shot classification. In European conference on computer vision, pp. 493–510. Springer, 2022.

Zhang, T., Wang, J., Guo, H., Dai, T., Chen, B., and Xia, S.-T. Boostadapter: Improving test-time adaptation via regional bootstrapping. arXiv preprint arXiv:2410.15430, 2024a.

Zhang, Y., Zhu, W., Tang, H., Ma, Z., Zhou, K., and Zhang, L. Dual memory networks: A versatile adaptation approach for vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 28718–28728, 2024b.

Zhang, Y., Kim, Y., Choi, Y.-G., Kim, H., Liu, H., and Hong, S. Backpropagation-free test-time adaptation via probabilistic gaussian alignment. arXiv preprint arXiv:2508.15568, 2025.

Zhao, H., Liu, Y., Alahi, A., and Lin, T. On pitfalls of test-time adaptation. arXiv preprint arXiv:2306.03536, 2023.

Zou, H. The adaptive lasso and its oracle properties. Journal of the American statistical association, 101(476):1418– 1429, 2006.

## A. Overall Procedure of MOON

The overall procedure of our proposed MOON is presented in Alg. 1. The BSUM-style iterative optimization of MOON can be conceptualized as a generalized Expectation-Maximization (EM) algorithm: fixing parameters $( \mu _ { k } , \kappa _ { k } )$ to update assignments z corresponds to the E-step, while fixing z and updating $( \mu _ { k } , \kappa _ { k } )$ corresponds to the M-step.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Overall procedure of MOON

Require: Visual feature embeddings $\{\mathbf{f}_i\}_{i=1}^N, \mathbf{f}_i \in \mathcal{S}^{d-1}$, textual class embeddings $\{\mathbf{t}_k\}_{k=1}^K, \mathbf{t}_k \in \mathcal{S}^{d-1}$; Fixed hyperparameters (no need for tuning): VLM temperature $\tau$, Number of neighbors in Laplacian term $m$, Iterations $T$.

Ensure: Latent label assignments as final predictions $\mathbf{z} \in [0,1]^{N \times K}$.

# Initialization

1: Compute zero-shot logits $\hat{\mathbf{y}} = \{\hat{\mathbf{y}}_{i,k}\}_{k=1}^K, \hat{\mathbf{y}}_{i,k} = \frac{\exp(\mathbf{f}_i^\top \mathbf{t}_k / \tau)}{\sum_j \exp(\mathbf{f}_i^\top \mathbf{t}_j / \tau)}$; initialize assignments $\mathbf{z} \leftarrow \hat{\mathbf{y}}$. ▷ See Eq. (1)

2: Initialize vMF parameters $\boldsymbol{\mu} = \{\boldsymbol{\mu}_k\}_{k=1}^K$ and $\boldsymbol{\kappa} = \{\kappa_k\}_{k=1}^K$; prior anchors $\boldsymbol{\mu}'$ and $\boldsymbol{\kappa}'$. ▷ See Eq. (7)

3: Build a m-NN affinity graph $\mathbf{W} = [\omega_{i,j}] \in \mathcal{R}^{N \times N}$, where $\omega_{i,j} = \begin{cases} \mathbf{f}_i^\top \mathbf{f}_j, &amp; \text{if } \mathbf{f}_j \text{ is the } m\text{-nearest neighbors of } \mathbf{f}_i \\ 0, &amp; \text{otherwise} \end{cases}$.

4: Compute class-level adjustment weights $\boldsymbol{\alpha} = \{\alpha_k\}_{k=1}^K$ with confidence $\boldsymbol{\lambda} = \{\lambda_k\}_{k=1}^K$. ▷ See Eq. (10) and (11)

5: Initialize instance-level adjustment weights $\boldsymbol{\gamma} = \{\gamma_i\}_{i=1}^N$. ▷ See Eq. (9)

# BSUM-style iterative optimization

6: for $t = 1$ to $T$ do

# Block update with respect to assignments $\mathbf{z}$

7: Compute vMF log-likelihood scores: log $p_{i,k}^{\mathrm{vMF}} \leftarrow \kappa_k \boldsymbol{\mu}_k^\top \mathbf{f}_i + \log C_d(\kappa_k)$. ▷ See Eq. (6)

8: Update z by single-pass linear approximation: $\mathbf{z}_i^{(t+1)} = \frac{\hat{\mathbf{y}}_i \odot \exp(\log \mathbf{p}_i^{\mathrm{vMF}} + \sum_j \omega_{ij} \mathbf{z}_j^{(t)})}{(\hat{\mathbf{y}}_i \odot \exp(\log \mathbf{p}_i^{\mathrm{vMF}} + \sum_j \omega_{ij} \mathbf{z}_j^{(t)}))^{\top} \mathbb{1}_K}$. ▷ See Eq. (12)

9: Update instance-level adjustment weights: $\gamma_i = 1 - \frac{H(\mathbf{z}_i)}{\log K}$.

# Compute shrinkage strengths

10: Compute shrinkage strengths $\beta = \{\beta_k\}_{k=1}^K$: $n_k \leftarrow \sum_{i=1}^N \gamma_i z_{i,k}, \quad \beta_k \leftarrow \frac{n_k}{n_k + \alpha_k}$.

(In practice, we use hard label counts for stability: $n_k = \sum_i \gamma_i$ [arg max_j $z_{i,j} = k$.]

# Block update with respect to distribution parameters ($\boldsymbol{\mu},\boldsymbol{\kappa}$)

11: Compute empirical estimates from standard MLE: $\mathbf{v}_k \leftarrow \frac{\sum_i \gamma_i z_{i,k} \mathbf{f}_i}{\sum_i \gamma_i z_{i,k}}$.

12: Update $\boldsymbol{\mu}$ and $\mathcal{A}_d(\boldsymbol{\kappa})$ by closed-form anchor shrinkage: $\boldsymbol{\mu}_k = \frac{\beta_k v_k + (1 - \beta_k) u_k'}{\|\beta_k v_k + (1 - \beta_k) u_k' \|}, A_d(\kappa_k) = \|β_k v_k + (1 - β_k) u_k'\|$, ▷ See Eq. (14)

13: Inversely estimate $\boldsymbol{\kappa}$ using Banerjee approximation: $\kappa_k = A_d^{-1}(\kappa_k)$. ▷ See Eq. (15)

14: end for

15:return z.
</div>

In implementation, we modify the shrinkage strength $\beta _ { k }$ by replacing the soft assignment predictions with hard ones on the vertices of the probability simplex $\Delta _ { K }$ following StatA (Zanella et al., 2025), which is:

$$
\beta_ {k} \approx \frac {\sum_ {i = 1} ^ {N} \mathbb {1} [ k = \operatorname{argmax} _ {r} \gamma_ {i , k} z _ {i , r} ]}{\sum_ {i = 1} ^ {N} \mathbb {1} [ k = \operatorname{argmax} _ {r} \gamma_ {i , k} z _ {i , r} ] + \alpha_ {k}}.\tag{19}
$$

This design has been proven to be more robust in practice, with experimental results shown in App. J.

Complexity analysis. We provide a theoretical complexity analysis of our vMF-based MOON and Gaussian-based stateof-the-art method StatA, further demonstrating the significant efficiency advantage of vMF mixtures that has been largely overlooked in prior works. Formally, let N, K, d denote the number of samples, classes, and feature dimensions, respectively. Let m be the number of neighbors and T the number of iterations.

(1) Graph construction: Both methods share the initial cost of constructing the affinity graph. This requires $O ( N ^ { 2 } d )$ for dense similarity computation, which dominates the pre-processing cost.

(2) Per-iteration cost: For MOON, each iteration is dominated by two parts: (i) likelihood computation and parameter aggregation, which scales with $O ( N K d )$ ; and (ii) sparse graph smoothing, which scales with $O ( N m K )$ . Thus, the per-iteration complexity is $O ( N K ( d + m ) )$ . In contrast, StatA incurs higher costs due to two factors: (i) it performs $L _ { G }$ inner loops for assignment updates (typically $L _ { G } = 5 )$ , inflating the smoothing cost to $O ( L _ { G } N m K )$ ; and (ii) its parameter update involves computing per-class covariance matrices, adding an $O ( K d ^ { 2 } )$ term.

(3) Total cost: Excluding the shared graph construction, the total asymptotic complexities are:

$$
\mathcal {O} _ {\mathrm{MOON}} \approx T \cdot N K (d + m) \quad \text { vs. } \quad \mathcal {O} _ {\mathrm{StatA}} \approx T \cdot (N K d + L _ {G} N m K + K d ^ {2}).\tag{20}
$$

MOON achieves superior efficiency by avoiding the quadratic complexity $O ( d ^ { 2 } )$ in parameter updates and eliminating the need for inner assignment loops.

Convergence guarantee of optimization algorithm. Our algorithm can be analyzed within the standard BSUM framework. Let $\mathcal { L } ( z , \Theta )$ denote the objective and $\Theta = \{ \mu _ { k } , \kappa _ { k } \} _ { k = 1 } ^ { K }$ . Given that: (1) Affinity matrix W is PSD; (2) Anchor weights $\alpha _ { k }$ and $\gamma _ { i } > 0 ; ( 3 )$ Fixing z, the parameter subproblem in Θ has a unique minimizer.

Then each outer iteration of our algorithm is a valid BSUM step:

• z-update: The only non-convex part is the Laplacian term. Since W is PSD, this term is concave and can be upperbounded by its first-order Taylor expansion at the current iteration (proofs in App. G.1). Minimizing this tight surrogate yields Eq. (12), i.e., the update is an exact minimizer of the majorized subproblem.

• Θ update: Fixing z, our objective is strictly convex, and Eqs. (13)-(14) are the corresponding closed-form exact updates. This step further decreases the objective.

Therefore, each outer iteration satisfies

$$
\mathcal {L} (z ^ {(t + 1)}, \Theta^ {(t + 1)}) \leq \mathcal {L} (z ^ {(t)}, \Theta^ {(t)}).\tag{21}
$$

Since $\mathcal { L } _ { a }$ is lower-bounded, the objective sequence is monotonically non-increasing and thus convergent. And, every limit point of the iterate sequence is a coordinate-wise minimum. And, as BSUM doe not require solving subproblems to full convergence at each outer iteration, single-pass z-update preserves convergence guarantee.

## B. Revisiting StatA

StatA (Zanella et al., 2025) first discuss transduction under the realistic class imbalanced scenarios at test-time. Unlike our MOON which operates on the unit hypersphere, StatA assumes that the visual feature representations $\{ \mathbf { f } _ { i } \} _ { i = 1 } ^ { N }$ follow a Gaussian Mixture Model (GMM) in the Euclidean space. To handle performance degradation, it introduces a KL-divergencebased regularization term acting as a statistical anchor. The optimization objective is to minimize the PLE objective penalized by the deviation from zero-shot prior anchors leveraged from text $\{ \mathcal { N } ( \mu _ { k } ^ { \prime } , \Sigma _ { k } ^ { \prime } ) \} _ { k = 1 } ^ { K }$

$$
\begin{array}{l} \mathcal {L} _ {\text {StatA}} (\mathbf {z}, \theta) = \underbrace {- \sum_ {i = 1} ^ {N} \mathbf {z} _ {i} ^ {\top} \log \mathcal {N} (f _ {i} | \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k})} _ {\text {negative log - likelihood (NLL)}} + \mathcal {R} (\mathbf {z}) + \alpha \underbrace {\sum_ {k = 1} ^ {K} \mathrm{KL} \big (\mathcal {N} (\boldsymbol {\mu} _ {k} ^ {\prime} , \boldsymbol {\Sigma} _ {k} ^ {\prime}) \| \mathcal {N} (\boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k}) \big)} _ {\text {statistical anchor}}, \\ \text {where} \quad \mathcal {R} (\mathbf {z}) = \underbrace {- \sum_ {i , j} \omega_ {i j} \mathbf {z} _ {i} ^ {\top} \mathbf {z} _ {j}} _ {\text {Laplacian reg.}} + \underbrace {\sum_ {i = 1} ^ {N} \mathrm{KL} (\mathbf {z} _ {i} \| \hat {\mathbf {y}} _ {i})} _ {\text {text supervision}}. \end{array}\tag{22}
$$

where $\boldsymbol { \theta } = \{ \mu _ { k } , \Sigma _ { k } \} _ { k = 1 } ^ { K }$ are the mixture parameters, and $\alpha > 0$ is a hyperparameter as anchor term weight. The anchor distribution is initialized as: $\pmb { \mu } _ { k } ^ { \prime } = \mathbf { t } _ { k }$ and $\begin{array} { r } { \Sigma ^ { \prime } = \operatorname { D i a g } \left( \frac { \sum _ { i , k } \hat { y } _ { i , k } ( \mathbf { f } _ { i } - { \pmb \mu } _ { k } ) ( \mathbf { f } _ { i } - { \pmb \mu } _ { k } ) ^ { \top } } { \sum _ { i , k } \hat { y } _ { i , k } } \right) } \end{array}$

The algorithm is performed via BSUM-style iterative optimization, which alternates between assignments z block and distribution parameters θ block:

• Assignment update: Fixing θ, the assignments $z _ { i , k }$ are updated based on the Gaussian likelihoods $\begin{array} { r l } { p _ { i , k } } & { { } \propto } \end{array}$ $\begin{array} { r } { \frac { 1 ^ { \cup } } { \sqrt { | \Sigma _ { k } | } } \exp \big ( - \frac { 1 ^ { \mathtt { s } } } { 2 } ( \mathbf { f } _ { i } - \pmb { \mu } _ { k } ) ^ { \top } \pmb { \Sigma } _ { k } ^ { - 1 } ( \mathbf { f } _ { i } - \pmb { \mu } _ { k } ) \big ) } \end{array}$ , similar rule as Eq. (12).

• Parameter update: Fixing z, the parameters are updated in closed form. Crucially, the update for the mean $\pmb { \mu } _ { k }$ and covariance $\Sigma _ { k }$ exhibits an anchor shrinkage behavior:

$$
\pmb {\mu} _ {k} = \beta_ {k} \mathbf {v} _ {k} + (1 - \beta_ {k}) \pmb {\mu} _ {k} ^ {\prime}, \quad \pmb {\Sigma} _ {k} = \beta_ {k} T _ {k} + (1 - \beta_ {k}) (\Sigma^ {\prime} + \mathrm{Diag} ((\pmb {\mu} _ {k} ^ {\prime} - \pmb {\mu} _ {k}) ^ {2})),\tag{23}
$$

with empirical estimates as:

$$
\boldsymbol {v} _ {k} = \frac {\sum_ {i = 1} ^ {N} z _ {i , k} \mathbf {f} _ {i}}{\sum_ {i = 1} ^ {N} z _ {i , k}}; \quad \boldsymbol {T} _ {k} = \frac {\sum_ {i = 1} ^ {N} z _ {i , k} \mathrm{Diag} ((\mathbf {f} _ {i} - \boldsymbol {\mu} _ {k}) ^ {2})}{\sum_ {i = 1} ^ {N} z _ {i , k}}.\tag{24}
$$

Here, $\begin{array} { r } { n _ { k } = \sum _ { i } z _ { i , k } } \end{array}$ is the soft count, and $\begin{array} { r } { \beta _ { k } = \frac { n _ { k } } { n _ { k } + \alpha } \approx \frac { \sum _ { i } 1 \left[ k = \mathrm { a r g m a x } _ { r } z _ { i , r } \right] } { \sum _ { i } 1 \left[ k = \mathrm { a r g m a x } _ { r } z _ { i , r } \right] + \alpha } \in [ 0 , 1 ] } \end{array}$ denotes the shrinkage strength.

Limitations. As shown in Eq. (23), the shrinkage strength $\beta _ { k }$ , controlled by soft count $n _ { k }$ and anchor weight $\alpha ,$ , balances the trade-off between empirical estimate $\mathbf { v } _ { k }$ and the prior anchor $\pmb { \mu } _ { k } ^ { \prime }$ . However, anchor weight α is a fixed scalar hyperparameter shared across all classes and samples. This implies a static shrinkage strength modeling that ignores the varying reliability of different classes (e.g., outliers vs. effective classes) or instances, rendering StatA suboptimal in both accuracy and robustness under realistic class imbalance, as discussed in Sec. 1.

## C. Experimental Details

## C.1. Datasets

We evaluate our proposed MOON and other baselines on 11 widely-used public datasets for fine-grained visual classification. These datasets cover a diverse range of domains, including generic objects, scenes, textures, satellite imagery, and specific fine-grained categories. Specifically, the benchmark includes: ImageNet (Deng et al., 2009), SUN397 (Xiao et al., 2010), Aircraft (Maji et al., 2013), EuroSAT (Helber et al., 2019), StanfordCars (Krause et al., 2013), Food101 (Bossard et al., 2014), Pets (Parkhi et al., 2012), Flowers102 (Nilsback & Zisserman, 2008), Caltech101 (Fei-Fei et al., 2004), DTD (Cimpoi et al., 2014), and UCF101 (Soomro et al., 2012). Detailed statistics for these datasets are provided in Tab. 10.

## C.2. Baselines

We compare our MOON against a comprehensive set of baselines, which are categorized into: (1) Transductive methods: EM-Dirichlet (Dirichlet) (Martin et al., 2024), ZLaP (Kalantidis et al., 2024), GDA-CLIP (Wang et al., 2024), TransCLIP (Zanella et al., 2024), ADAPT (Zhang et al., 2025), and StatA (Zanella et al., 2025)<sup>6</sup>; (2) Online TTA methods: TENT (Wang et al., 2020), TDA (Karmanov et al., 2024), DMN (Zhang et al., 2024b), and OGA (Fuchs et al., 2025). We also incorporate another TTA method MTA (Zanella & Ben Ayed, 2024) that requires per-image augmentation.

The details and experimental configurations of these baselines are listed as below:

• ZLaP (CVPR’24): introduces a non-parametric framework that leverages the graph structure of unlabeled data via label propagation, utilizing geodesic distances on the data manifold to address the modality gap in VLMs. The number of nearest neighbors m is set to 5, scale parameter of RBF kernel function $\gamma$ is set to 5.0, and the clamping factor α is fixed at 0.3. We don’t specifically scale the similarity matrix.

• EM-Dirichlet (CVPR’24): frames transduction on the unit simplex by modeling class-conditional feature distributions with a Dirichlet law, solving the MLE problem via a hyperparameter-free Block Majorization-Minimization algorithm. The temperature T in the probabilities is fixed to 30.

• GDA-CLIP (ICLR’24): applies Gaussian Discriminant Analysis (GDA) by assuming a shared covariance matrix for class features, estimating parameters via closed-form solutions and ensembling them with zero-shot logits. The ensemble weight is set to α by default.

• TransCLIP (NeurIPS’24): formulates adaptation as a regularized MLE with a text-guided KL-divergence penalty, employing an iterative optimization procedure that decouples sample assignments and parameter updates. Here, text-guided KL divergence penalty λ is set to 1, and the number of nearest neighbors m is set to 3.

• ADAPT (NeurIPS’25): presents a backpropagation-free method that reframes adaptation as Gaussian probabilistic inference with closed-form updates, utilizing a knowledge bank to efficiently support online and transductive settings. We set the bank size L to 12, and the momentum coefficient for parameter update α to 0.9.

• StatA (CVPR’25): addresses realistic scenarios with variable effective classes by employing a statistical anchor regularization within a Gaussian Mixture Model (GMM) to dynamically constrain features near text-derived priors. The anchor term weight α is set to 1, with the number of nearest neighbors m set to 3. We use hard $\beta _ { k }$ by default.

• TENT (ICLR’21): adapts the model by minimizing the Shannon entropy of predictions on target data, specifically updating the affine parameters of Batch Normalization layers to align internal statistics online. We set the learning rate to 1e-3, and perform 5 adaptation steps for each batch.

• TDA (CVPR’24): utilizes a lightweight key-value cache system with entropy-based filtering and introduces a negative cache mechanism to explicitly penalize unlikely classes using negative pseudo-labeling. All the configuration is kept the same as those set in the original paper on ImageNet.

• DMN (CVPR’24): integrates a static memory for pre-trained knowledge and a dynamic memory for historical test features, employing a cross-attention strategy to refine decision boundaries based on temporal context. All the configuration is kept the same as those set in the original paper on ImageNet.

• OGA (CVPRW’25): reframes online adaptation as a Maximum A Posteriori (MAP) estimation problem using multivariate Gaussian distributions and zero-shot priors to calibrate predictions without gradient backpropagation. THe memory update threshold τ is set to 0.01, with cache memory capacity set to 8.

• MTA (CVPR’24): proposes a training-free strategy that leverages MeanShift on augmented views to identify distribution modes, jointly optimizing a learnable inlierness score to robustly aggregate visual information. All the configuration is kept the same as those set in the original paper on ImageNet.

## C.3. Prompts

Following (Zhang et al., 2022), we adopt default, fixed prompt templates to initialize text embeddings for all methods, as illustrated in Tab. 10.

Table 10. Dataset information and prompt templates.

<table><tr><td>Name</td><td>Other name</td><td># K</td><td># N</td><td>Description</td><td>Prompt template</td></tr><tr><td>SUN397</td><td>SUN397</td><td>397</td><td>19,850</td><td>Scenes classification</td><td>&quot;a photo of a [ ].&quot;</td></tr><tr><td>Aircraft</td><td>FGVCAircraft</td><td>100</td><td>3,333</td><td>Aircraft classification</td><td>&quot;a photo of a [ ], a type of aircraft.&quot;</td></tr><tr><td>EuroSAT</td><td>EuroSAT</td><td>10</td><td>8,100</td><td>Satellite images classification</td><td>&quot;a centered satellite photo of [ ].&quot;</td></tr><tr><td>StanfordCars</td><td>Cars</td><td>196</td><td>8,041</td><td>Cars classification</td><td>&quot;a photo of a [ ].&quot;</td></tr><tr><td>Food101</td><td>Food101</td><td>101</td><td>30,300</td><td>Food classification</td><td>&quot;a photo of [ ], a type of food.&quot;</td></tr><tr><td>Pets</td><td>OxfordPets</td><td>37</td><td>3,669</td><td>Pets classification</td><td>&quot;a photo of [ ], a type of pet.&quot;</td></tr><tr><td>Flowers102</td><td>OxfordFlowers</td><td>102</td><td>2,463</td><td>Flowers classification</td><td>&quot;a photo of a [ ], a type of flower.&quot;</td></tr><tr><td>Caltech101</td><td>Caltech101</td><td>101</td><td>2,465</td><td>Objects classification</td><td>&quot;a photo of a [ ].&quot;</td></tr><tr><td>DTD</td><td>DTD</td><td>47</td><td>1,692</td><td>Textures classification</td><td>&quot;[ ] texture.&quot;</td></tr><tr><td>UCF101</td><td>UCF101</td><td>101</td><td>3,783</td><td>Actions classification</td><td>&quot;a photo of a person doing [ ].&quot;</td></tr><tr><td>ImageNet</td><td>ImageNet-1K</td><td>1000</td><td>50,000</td><td>Objects classification</td><td>&quot;a photo of a [ ].&quot;</td></tr></table>

## C.4. Data Sampler

In this section, we describe the sampling strategies for constructing realistic test-time scenarios, following the protocols in StatA (Zanella et al., 2025).

Batch adaptation. To simulate realistic class sparsity where the label distribution within a batch is partial, we construct test batches with a limited number of effective classes. Specifically, given a batch size B and a total of K classes, we first determine the number of effective classes $K _ { e f f }$ , which is either fixed or uniformly sampled from $[ K _ { e f f } ^ { \operatorname* { m i n } } , K _ { e f f } ^ { \operatorname* { m a x } } ]$ . We then randomly select a subset of classes $\mathcal { C } _ { b a t c h }$ with size $K _ { e f f }$ and aggregate all their corresponding samples. The final test batch is formed by randomly sampling $B$ instances without replacement from this restricted pool, ensuring that the batch contains only a fraction of the total categories.

Online adaptation. We generate non-i.i.d. data streams using a Dirichlet-based framework to evaluate robustness against temporal correlation (Yuan et al., 2023). The data stream is divided into slots, where the allocation of each class across slots follows a Dirichlet distribution Dir(ξ · 1). The scalar ξ controls the correlation intensity: large values approximate an i.i.d. stream, while small values concentrate classes into fewer slots to create high temporal correlation. Additionally, we consider a separate sequential setting (simulating $\xi  0 )$ , where classes are randomly permuted and all samples from a class appear contiguously before transitioning to the next, representing the most extreme temporal correlation.

## C.5. Implementation Details

Unless otherwise specified, we employ CLIP ViT-B/16 as the default backbone and evaluate performance using the Top-1 accuracy. Consistent with our black-box assumption, we utilize a fixed set of hyperparameters across all experiments without per-task tuning: we set the nearest neighbors $m = 3$ and the number of iterations to 10, while inheriting the temperature parameter τ directly from pre-trained VLM. For stability and robustness, we employ hard assignments for the shrinkage strength $\beta _ { k }$ , and dynamically update the instance-level weights $\gamma _ { i }$ at each iteration. All experiments are conducted on a single NVIDIA RTX 4090 24GB GPU. To ensure statistical reliability given the stochastic data sampling, all reported results represent the average of 1,000 independent runs for batch adaptation and 100 runs for online adaptation, initialized with a fixed random seed of 1.

## D. Generality of KL-Anchored PLE for Exponential Families

In this section, we provide a formal proof that KL-based distribution anchor in the penalized likelihood estimation (PLE) formulation, i.e., $\mathcal { R } ( \mathbf { M } )$ in Eq. (2), yields an adaptive shrinkage behavior that enables convex combination update in the mean-parameter space for any (regular) exponential-family class-conditional mixture model.

## D.1. Problem Setup

Consider a K-class latent-variable mixture model with unlabeled samples $\{ x _ { i } \} _ { i = 1 } ^ { N }$ and soft assignments $\mathbf { z } _ { i , k } \in [ 0 , 1 ]$ satisfying $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \mathbf { z } _ { i , k } = 1 } \end{array}$ . For each class $k \in \{ 1 , \ldots , K \}$ , assume the class-conditional density function belongs to a regular minimal exponential family:

$$
p (x \mid \eta) = h (x) \exp \bigl (\eta^ {\top} T (x) - A (\eta) \bigr),\tag{25}
$$

where $\eta$ is the natural parameter, $T ( x )$ is the sufficient statistic, and $A ( \eta )$ is the log-partition function. For a regular minimal exponential family, $A ( \eta )$ is strictly convex and the mapping $\nabla A$ is one-to-one, relating natural parameters to mean parameters $\mu = \mathbb { E } _ { \eta } [ T ( X ) ] = \nabla A ( \eta )$

Given soft assignments, define the class-wise soft counts and soft sufficient-statistic sums:

$$
n _ {k} = \sum_ {i = 1} ^ {N} \mathbf {z} _ {i, k}, \qquad S _ {k} = \sum_ {i = 1} ^ {N} \mathbf {z} _ {i, k} T (x _ {i}).\tag{26}
$$

Let $q _ { k } ( x ) = p ( x \mid \eta _ { k } ^ { \prime } )$ denote a fixed anchor distribution for class k. Conditioned on $\{ { \bf z } _ { i , k } \}$ , the M-step minimizes the following KL-anchored PLE objective over $\{ \eta _ { k } \} _ { k = 1 } ^ { K }$

$$
\mathcal {L} (\{\eta_ {k} \}) = - \sum_ {i = 1} ^ {N} \sum_ {k = 1} ^ {K} \mathbf {z} _ {i, k} \log p (x _ {i} \mid \eta_ {k}) + \alpha \sum_ {k = 1} ^ {K} \mathrm{KL} \Bigl (q _ {k} \| p (\cdot \mid \eta_ {k}) \Bigr), \quad \alpha > 0.\tag{27}
$$

## D.2. Closed-Form KL for Exponential Families

Lemma D.1 (KL divergence within an exponential family). $L e t q ( \cdot ) = p ( \cdot \mid \eta ^ { \prime } ) a n d p ( \cdot ) = p ( \cdot \mid \eta )$ be members of the same exponential family (25). Then

$$
\operatorname{KL} \left(p (\cdot \mid \eta^ {\prime}) \| p (\cdot \mid \eta)\right) = A (\eta) - A \left(\eta^ {\prime}\right) - \left(\eta - \eta^ {\prime}\right) ^ {\top} \mu^ {\prime}, \quad \mu^ {\prime} = \mathbb {E} _ {\eta^ {\prime}} [ T (X) ] = \nabla A \left(\eta^ {\prime}\right).\tag{28}
$$

Proof. By definition,

$$
\operatorname{KL} \left(p (\cdot \mid \eta^ {\prime}) \| p (\cdot \mid \eta)\right) = \mathbb {E} _ {\eta^ {\prime}} \left[ \log p (X \mid \eta^ {\prime}) - \log p (X \mid \eta) \right].\tag{29}
$$

Using (25), we have

$$
\log p (X \mid \eta^ {\prime}) - \log p (X \mid \eta) = \left((\eta^ {\prime}) ^ {\top} T (X) - A (\eta^ {\prime})\right) - \left(\eta^ {\top} T (X) - A (\eta)\right),\tag{30}
$$

since log $h ( X )$ cancels. Taking expectation under $p ( \cdot \mid \eta ^ { \prime } )$ yields

$$
\operatorname{KL} \left(p (\cdot \mid \eta^ {\prime}) \| p (\cdot \mid \eta)\right) = \left(\eta^ {\prime} - \eta\right) ^ {\top} \mathbb {E} _ {\eta^ {\prime}} [ T (X) ] + A (\eta) - A \left(\eta^ {\prime}\right)\tag{31}
$$

$$
= A (\eta) - A \left(\eta^ {\prime}\right) - \left(\eta - \eta^ {\prime}\right) ^ {\top} \mu^ {\prime},\tag{32}
$$

where $\mu ^ { \prime } = \mathbb { E } _ { \eta ^ { \prime } } [ T ( X ) ] = \nabla A ( \eta ^ { \prime } )$

## D.3. KL Anchoring Implies Convex Combination in Mean-Parameter Space

Theorem D.2 (KL-anchored M-step yields convex combination in mean space). Assume each class-conditional model $p ( \cdot \mid \eta _ { k } )$ belongs to a regular minimal exponential family (25). Fixing soft assignments $\{ { \bf z } _ { i , k } \}$ , the M-step objective (27) is strictly convex in each $\eta _ { k }$ and admits a unique minimizer η<sup>⋆</sup>. Moreover, the corresponding mean parameter satisfies

$$
\nabla A (\eta_ {k} ^ {\star}) = \frac {S _ {k} + \alpha \mu_ {k} ^ {\prime}}{n _ {k} + \alpha}, \qquad \mu_ {k} ^ {\prime} = \nabla A (\eta_ {k} ^ {\prime}).\tag{33}
$$

Equivalently, for $n _ { k } > 0$ with empirical mean parameter $\hat { \mu } _ { k } = S _ { k } / n _ { k }$

$$
\nabla A (\eta_ {k} ^ {\star}) = \beta_ {k} \hat {\mu} _ {k} + (1 - \beta_ {k}) \mu_ {k} ^ {\prime}, \qquad \beta_ {k} = \frac {n _ {k}}{n _ {k} + \alpha} \in [ 0, 1 ].\tag{34}
$$

Proof. The objective (27) decomposes across classes: $\begin{array} { r } { \mathcal { L } ( \{ \eta _ { k } \} ) = \sum _ { k = 1 } ^ { K } \mathcal { L } _ { k } ( \eta _ { k } ) \ + } \end{array}$ const, where, using log p(x $\mid \eta ) =$ log $h ( x ) + \eta ^ { \top } T ( x ) - A ( \eta )$ ,

$$
- \sum_ {i = 1} ^ {N} \mathbf {z} _ {i, k} \log p (x _ {i} \mid \eta_ {k}) = - \sum_ {i = 1} ^ {N} \mathbf {z} _ {i, k} \left(\log h (x _ {i}) + \eta_ {k} ^ {\top} T (x _ {i}) - A (\eta_ {k})\right)\tag{35}
$$

$$
= - \eta_ {k} ^ {\top} \sum_ {i = 1} ^ {N} \mathbf {z} _ {i, k} T (x _ {i}) + \Big (\sum_ {i = 1} ^ {N} \mathbf {z} _ {i, k} \Big) A (\eta_ {k}) + \text { const }\tag{36}
$$

$$
= - S _ {k} ^ {\top} \eta_ {k} + n _ {k} A (\eta_ {k}) + \text { const. }\tag{37}
$$

For the KL anchor term, apply Lemma D.1 with $q _ { k } ( \cdot ) = p ( \cdot \mid \eta _ { k } ^ { \prime } )$ and $p ( \cdot \mid \eta _ { k } )$

$$
\mathrm{KL} \left(q _ {k} \| p (\cdot | \eta_ {k})\right) = A \left(\eta_ {k}\right) - A \left(\eta_ {k} ^ {\prime}\right) - \left(\eta_ {k} - \eta_ {k} ^ {\prime}\right) ^ {\top} \mu_ {k} ^ {\prime}, \quad \mu_ {k} ^ {\prime} = \nabla A \left(\eta_ {k} ^ {\prime}\right).\tag{38}
$$

Combining (37) and (38) (and dropping constants independent of $\eta _ { k } )$ , we obtain

$$
\mathcal {L} _ {k} (\eta_ {k}) = - S _ {k} ^ {\top} \eta_ {k} + n _ {k} A (\eta_ {k}) + \alpha \left(A (\eta_ {k}) - \left(\eta_ {k}\right) ^ {\top} \mu_ {k} ^ {\prime}\right) + \text { const }\tag{39}
$$

$$
= - \left(S _ {k} + \alpha \mu_ {k} ^ {\prime}\right) ^ {\top} \eta_ {k} + (n _ {k} + \alpha) A (\eta_ {k}) + \text { const }.\tag{40}
$$

Since $A ( \eta )$ is strictly convex for a regular minimal exponential family and $n _ { k } + \alpha > 0$ , it follows that $\mathcal { L } _ { k } ( \eta _ { k } )$ is strictly convex in $\eta _ { k }$ and thus has a unique minimizer $\eta _ { k } ^ { \star }$

Taking the gradient of (40) and setting it to zero yields the first-order optimality condition:

$$
- (S _ {k} + \alpha \mu_ {k} ^ {\prime}) + (n _ {k} + \alpha) \nabla A (\eta_ {k} ^ {\star}) = 0,
$$

(41)

which implies (33). When $n _ { k } > 0$ , substituting $S _ { k } = n _ { k } \hat { \mu } _ { k }$ into (33) gives (34) with $\beta _ { k } = n _ { k } / ( n _ { k } + \alpha ) \in [ 0 , 1 ]$

## D.4. Corollaries for Realistic Test-Time Scenarios

Theorem D.2 immediately yields two properties that are particularly relevant to realistic test-time settings with class imbalance and sparse effective label coverage.

Corollary D.3 (No deviation for outlier classes). If a class is entirely outlier (absent classes) in the current batch in the sense that $n _ { k } = 0$ , then the optimal mean parameter satisfies

$$
\nabla A (\eta_ {k} ^ {\star}) = \mu_ {k} ^ {\prime} = \nabla A (\eta_ {k} ^ {\prime}).\tag{42}
$$

Moreover, since ∇A is injective for a regular minimal exponential family, it follows that

$$
\eta_ {k} ^ {\star} = \eta_ {k} ^ {\prime}.\tag{43}
$$

That $i s ,$ the optimal parameter for an outlier (absent) class is exactly its anchor parameter, and will not deviate due to the absence of evidence.

Proof. Setting $n _ { k } = 0$ and $S _ { k } = \mathbf { 0 }$ in (33) gives $\nabla A ( \eta _ { k } ^ { \star } ) = \mu _ { k } ^ { \prime }$ . Injectivity of $\nabla A$ in a regular minimal exponential family implies $\eta _ { k } ^ { \star } = \eta _ { k } ^ { \prime }$ □

Corollary D.4 (Bounded deviation for rare classes). For any class with $n _ { k } > 0 ,$ , the deviation from the anchor in meanparameter space is bounded by

$$
\left\| \nabla A (\eta_ {k} ^ {\star}) - \mu_ {k} ^ {\prime} \right\| = \beta_ {k} \left\| \hat {\mu} _ {k} - \mu_ {k} ^ {\prime} \right\| \leq \frac {n _ {k}}{n _ {k} + \alpha} \left\| \hat {\mu} _ {k} - \mu_ {k} ^ {\prime} \right\|.\tag{44}
$$

Thus, when a class appears rarely (small $n _ { k } ) ,$ , its update magnitude away from the anchor is linearly shrunk by $\beta _ { k }$

Proof. Equation (34) implies $\nabla A ( \eta _ { k } ^ { \star } ) - \mu _ { k } ^ { \prime } = \beta _ { k } ( { \hat { \mu } } _ { k } - \mu _ { k } ^ { \prime } )$ . Taking norms on both sides yields (44).

Remarks. Theorem D.2 formalizes a key mechanism behind KL-anchored PLE: regardless of the specific exponentialfamily choice, the anchor regularization transforms the M-step into a strictly convex problem whose first-order condition yields an adaptive shrinkage that enables convex combination in mean-parameter space. This property is particularly beneficial in realistic test-time adaptation, where many classes may be missing or under-represented within a batch (e.g., $K _ { \mathrm { e f f } } \ll K$ and $\xi  0 )$ , and naive maximum-likelihood estimation tends to overfit to locally biased statistics.

## E. Details of von Mises-Fisher Distributions

## E.1. Introduction to vMF distributions

Let $x \in \mathbb { R } ^ { d }$ be a random vector on the unit hypersphere $\mathbb { S } ^ { d - 1 } , \operatorname { i . e . , } \| x \| _ { 2 } = 1$ . The probability density function of von Mises-Fisher (vMF) distribution on $\mathbb { S } ^ { d - 1 }$ is defined by

$$
p (x; \boldsymbol {\mu}, \kappa) = \mathcal {C} _ {d} (\kappa) \exp \bigl (\kappa   \boldsymbol {\mu} ^ {\top} x \bigr), \qquad \boldsymbol {\mu} \in \mathbb {S} ^ {d - 1},   \kappa \geq 0,\tag{45}
$$

where $\pmb { \mu }$ is the mean direction vector and κ is the concentration scalar. The normalization constant is given by

$$
\mathcal {C} _ {d} (\kappa) = \frac {\kappa^ {\nu}}{(2 \pi) ^ {d / 2} I _ {\nu} (\kappa)}, \qquad \nu = \frac {d}{2} - 1,\tag{46}
$$

where $I _ { \nu } ( \cdot )$ denotes the modified Bessel function of the first kind: $\begin{array} { r } { I _ { \nu } \left( \kappa \right) = \left( \frac { 1 } { 2 } \kappa \right) ^ { \nu } \sum _ { k = 0 } ^ { \infty } \frac { \left( \frac { 1 } { 4 } \kappa ^ { 2 } \right) ^ { k } } { k ! \Gamma \left( \nu + k + 1 \right) } , \Gamma ( \cdot ) } \end{array}$ denotes Gamma function.

In our model, given a normalized visual feature embedding $\mathbf { f } _ { i } \in \mathbb { S } ^ { d - 1 }$ , we parameterize each class k by vMF parameters $\mathcal { V } _ { k } = ( \mu _ { k } , \kappa _ { k } )$ . The log-likelihood is

$$
\log p _ {i, k} ^ {\mathrm{vMF}} = \log \mathcal {C} _ {d} (\kappa_ {k}) + \kappa_ {k} \pmb {\mu} _ {k} ^ {\top} \mathbf {f} _ {i}.\tag{47}
$$

Numerical approximation. Taking logarithm of Eq. (46) yields

$$
\log \mathcal {C} _ {d} (\kappa) = \nu \log \kappa - \frac {d}{2} \log (2 \pi) - \log I _ {\nu} (\kappa).\tag{48}
$$

Since $I _ { \nu } ( \kappa )$ is transcendental, we adopt the large-κ asymptotic<sup>7</sup>

$$
\log I _ {\nu} (\kappa) \approx \kappa - \frac {1}{2} \log (2 \pi \kappa),\tag{49}
$$

which leads to

$$
\log \mathcal {C} _ {d} (\kappa) \approx \frac {d - 1}{2} \log \kappa - \kappa - \frac {d - 1}{2} \log (2 \pi).\tag{50}
$$

In implementation, since our assignment update Eq. (12) involves a softmax over classes; terms independent of κ (e.g., $- \frac { d - \mathrm { i } } { 2 } \log ( 2 \pi )$ for fixed $d )$ cancel out<sup>8</sup>. Therefore, we use the simplified approximation form

$$
\log \mathcal {C} _ {d} (\kappa) = \frac {d - 1}{2} \log \kappa - \kappa .\tag{51}
$$

## E.2. Derivation of KL divergence between two multivariate vMF distributions

Consider two vMF multivariate distributions $p ( x ) = \mathcal { V } _ { p } ( x ; \pmb { \mu } _ { p } , \kappa _ { p } )$ and $q ( x ) = \mathcal { V } _ { q } ( x ; \pmb { \mu } _ { q } , \kappa _ { q } )$ on $\mathbb { S } ^ { d - 1 }$ . The Kullback-Leibler (KL) divergence is

$$
\operatorname{KL} (p \| q) = \mathbb {E} _ {x \sim p} [ \log p (x) - \log q (x) ].\tag{52}
$$

Using the vMF log-density from (45), we have

$$
\log p (x) - \log q (x) = \log \frac {\mathcal {C} _ {d} (\kappa_ {p})}{\mathcal {C} _ {d} (\kappa_ {q})} + \kappa_ {p} \pmb {\mu} _ {p} ^ {\top} x - \kappa_ {q} \pmb {\mu} _ {q} ^ {\top} x.\tag{53}
$$

Taking expectation w.r.t. $x \sim p$ yields

$$
\mathrm{KL} (p \| q) = E _ {x \sim p} \left[ \log \frac {\mathcal {C} _ {d} (\kappa_ {p})}{\mathcal {C} _ {d} (\kappa_ {q})} + \kappa_ {p} \pmb {\mu} _ {p} ^ {\top} x - \kappa_ {q} \pmb {\mu} _ {q} ^ {\top} x \right]\tag{54}
$$

$$
= \log \frac {\mathcal {C} _ {d} (\kappa_ {p})}{\mathcal {C} _ {d} (\kappa_ {q})} + E _ {x \sim p} [ \kappa_ {p} \pmb {\mu} _ {p} ^ {\top} x ] - E _ {x \sim p} [ \kappa_ {q} \pmb {\mu} _ {q} ^ {\top} x ]\tag{55}
$$

$$
= \log \frac {\mathcal {C} _ {d} (\kappa_ {p})}{\mathcal {C} _ {d} (\kappa_ {q})} + \kappa_ {p} \pmb {\mu} _ {p} ^ {\top} \mathbb {E} _ {p} [ x ] - \kappa_ {q} \pmb {\mu} _ {q} ^ {\top} \mathbb {E} _ {p} [ x ].\tag{56}
$$

A standard vMF identity states that the mean of $x \sim \mathrm { v M F } ( \mu _ { p } , \kappa _ { p } )$ is aligned with $\mu _ { p }$ :

$$
\mathbb {E} _ {p} [ x ] = \mathcal {A} _ {d} (\kappa_ {p})   \boldsymbol {\mu} _ {p}, \qquad \mathcal {A} _ {d} (\kappa) \triangleq \frac {I _ {\frac {d}{2}} (\kappa)}{I _ {\frac {d}{2} - 1} (\kappa)},\tag{57}
$$

Substituting (57) into (56), and using $\| \mu _ { p } \| _ { 2 } = 1$ , we obtain

$$
\mathrm{KL} (p \| q) = \log \frac {\mathcal {C} _ {d} (\kappa_ {p})}{\mathcal {C} _ {d} (\kappa_ {q})} + \kappa_ {p} \pmb {\mu} _ {p} ^ {\top} (\mathcal {A} _ {d} (\kappa_ {p}) \pmb {\mu} _ {p}) - \kappa_ {q} \pmb {\mu} _ {q} ^ {\top} (\mathcal {A} _ {d} (\kappa_ {p}) \pmb {\mu} _ {p})\tag{58}
$$

$$
= \log \frac {\mathcal {C} _ {d} (\kappa_ {p})}{\mathcal {C} _ {d} (\kappa_ {q})} + \kappa_ {p} \mathcal {A} _ {d} (\kappa_ {p}) (\boldsymbol {\mu} _ {p} ^ {\top} \boldsymbol {\mu} _ {p}) - \kappa_ {q} \mathcal {A} _ {d} (\kappa_ {p}) (\boldsymbol {\mu} _ {q} ^ {\top} \boldsymbol {\mu} _ {p})\tag{59}
$$

$$
= \log \frac {\mathcal {C} _ {d} (\kappa_ {p})}{\mathcal {C} _ {d} (\kappa_ {q})} + \kappa_ {p} \mathcal {A} _ {d} (\kappa_ {p}) - \kappa_ {q} \mathcal {A} _ {d} (\kappa_ {p}) \boldsymbol {\mu} _ {q} ^ {\top} \boldsymbol {\mu} _ {p}.\tag{60}
$$

In our objective in Eq. (8), the anchor term uses $\mathrm { K L } ( \mathcal { V } _ { k } ^ { \prime } | | \mathcal { V } _ { k } )$ with $\mathcal { V } _ { k } ^ { \prime } = ( \mu _ { k } ^ { \prime } , \kappa _ { k } ^ { \prime } )$ (anchor) and $\mathcal { V } _ { k } = ( \mu _ { k } , \kappa _ { k } )$ (empirical estimate). Applying (60) to $p = \mathcal { V } _ { k } ^ { \prime } ( \pmb { \mu } _ { k } ^ { \prime } , \kappa _ { k } ^ { \prime } )$ and $q = \mathcal { V } _ { k } ( { \pmb \mu } _ { k } , \kappa _ { k } )$ gives

$$
\mathrm{KL} (\mathcal {V} _ {k} ^ {\prime} \| \mathcal {V} _ {k}) = \log \frac {\mathcal {C} _ {d} (\kappa_ {k} ^ {\prime})}{\mathcal {C} _ {d} (\kappa_ {k})} + \kappa_ {k} ^ {\prime} \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) - \kappa_ {k} \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime})   \boldsymbol {\mu} _ {k} ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime}.\tag{61}
$$

Therefore, the anchor term R(M) is given by

$$
\mathcal {R} (\mathbf {M}) = \sum_ {k = 1} ^ {K} \Big (\log \mathcal {C} _ {d} (\kappa_ {k} ^ {\prime}) - \log \mathcal {C} _ {d} (\kappa_ {k}) + \kappa_ {k} ^ {\prime} \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) - \kappa_ {k} \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime})   \boldsymbol {\mu} _ {k} ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime} \Big).\tag{62}
$$

When optimizing w.r.t. $( \mu _ { k } , \kappa _ { k } )$ , the terms depending on $( \mu _ { k } , \kappa _ { k } )$ reduce to

$$
- \log \mathcal {C} _ {d} (\kappa_ {k}) - \kappa_ {k} \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \boldsymbol {\mu} _ {k} ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime},\tag{63}
$$

up to constants independent of $( \mu _ { k } , \kappa _ { k } )$

## F. Derivation of the variable initialization of κ

In Eq. (7), we initialize $\pmb { \mu } _ { k } ^ { \prime }$ from the zero-shot text prototype $\mathbf { t } _ { k } .$ , and estimate $\boldsymbol { \mathcal { A } } _ { d } ( \kappa _ { k } ^ { \prime } )$ via a mean-squared-distance approximation on the unit hypersphere. We provide the detailed derivation of the latter in this section.

Given $\pmb { \mu } _ { k } ^ { \prime }$ fixed, we compute the soft, class-weighted mean squared Euclidean distance $\mathrm { M S E } _ { k }$ as

$$
\mathrm{MSE} _ {k} \triangleq \frac {\sum_ {i = 1} ^ {N} z _ {i , k} \| \mathbf {f} _ {i} - \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2} ^ {2}}{\sum_ {i = 1} ^ {N} z _ {i , k}} = \frac {\sum_ {i = 1} ^ {N} z _ {i , k} \| \mathbf {f} _ {i} - \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2} ^ {2}}{N _ {k}}, \quad N _ {k} \triangleq \sum_ {i = 1} ^ {N} z _ {i, k}.\tag{64}
$$

On the unit hypersphere, both f<sub>i</sub> and $\pmb { \mu } _ { k } ^ { \prime }$ are $\ell _ { 2 } \cdot$ -normalized, i.e., $\| \mathbf { f } _ { i } \| _ { 2 } = \| \pmb { \mu } _ { k } ^ { \prime } \| _ { 2 } = 1$ . Hence,

$$
\| \mathbf {f} _ {i} - \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2} ^ {2} = \| \mathbf {f} _ {i} \| _ {2} ^ {2} + \| \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2} ^ {2} - 2 (\mathbf {f} _ {i} ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime}) = 2 (1 - \cos \theta_ {i}),\tag{65}
$$

where cos $\theta _ { i } \triangleq \mathbf { f } _ { i } ^ { \intercal } \pmb { \mu } _ { k } ^ { \prime }$ . Taking the weighted average over i for class k gives

$$
\mathrm{MSE} _ {k} = 2 \Big (1 - \mathbb {E} _ {k} [ f ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime} ] \Big), \qquad \mathbb {E} _ {k} [ f ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime} ] \triangleq \frac {\sum_ {i} z _ {i , k} \mathbf {f} _ {i} ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime}}{\sum_ {i} z _ {i , k}}.\tag{66}
$$

Therefore,

$$
\mathbb {E} _ {k} [ f ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime} ] = 1 - \frac {\mathrm{MSE} _ {k}}{2}.\tag{67}
$$

For a vMF distribution on $\mathbb { S } ^ { d - 1 }$ with density $p ( f ) = \mathcal { C } _ { d } ( \kappa ) \exp ( \kappa \mu ^ { \top } f )$ , it is well-known that<sup>9</sup>

$$
\mathbb {E} _ {f \sim \mathrm{vMF} (\mu , \kappa)} [ \mu^ {\top} f ] = \mathcal {A} _ {d} (\kappa) \triangleq \frac {I _ {d / 2} (\kappa)}{I _ {d / 2 - 1} (\kappa)},\tag{68}
$$

where $I _ { \nu } ( \cdot )$ is the modified Bessel function of the first kind.

Combining (67) and (68), we obtain the approximation

$$
\mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \approx \mathbb {E} _ {k} [ f ^ {\top} \pmb {\mu} _ {k} ^ {\prime} ] = 1 - \frac {\mathrm{MSE} _ {k}}{2} = 1 - \frac {\sum_ {i} z _ {i , k} \| \mathbf {f} _ {i} - \pmb {\mu} _ {k} ^ {\prime} \| _ {2} ^ {2}}{2 \sum_ {i} z _ {i , k}}.\tag{69}
$$

Finally, $\kappa _ { k } ^ { \prime }$ is approximated as the inversion of $\boldsymbol { \mathcal { A } } ( \kappa _ { k } )$ using Eq. (15).

## G. Derivations of the variable update

## G.1. With respect to assignments z

Following the derivations from TransCLIP (Zanella et al., 2024), we derive the update for assignments $\mathbf { z } = \{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$ under the simplex constraint $\mathbf { z } _ { i } \in \Delta ^ { K }$ with our instance-level, entropy-based weight $\gamma _ { i }$ . Note that this derivation is based on the setting in Eq. (9) that $\gamma _ { i }$ is computed solely from the fixed pseudo label $\hat { \mathbf { y } } _ { i }$ . Although in actual implementation, we dynamically update $\gamma _ { i }$ with current assignment $\mathbf { z } _ { i }$ , we treat it more as an engineering trick.

Fixing the distribution parameters $( \mu _ { k } , \kappa _ { k } )$ , the z-dependent part from the objective in Eq. (8) can be written as

$$
\min _ {\mathbf {z} \in (\Delta^ {K}) ^ {N}} \sum_ {i = 1} ^ {N} \gamma_ {i} \Big (- \mathbf {z} _ {i} ^ {\top} \log \mathbf {p} _ {i} ^ {\mathrm{vMF}} + \mathrm{KL} (\mathbf {z} _ {i} \| \hat {\mathbf {y}} _ {i}) \Big) - \sum_ {i, j} \gamma_ {i} \omega_ {i j} \mathbf {z} _ {i} ^ {\top} \mathbf {z} _ {j}.\tag{70}
$$

Since $\omega _ { i j } = \mathbf { f } _ { i } ^ { \top } \mathbf { f } _ { j } \geq 0$ , the affinity matrix $\mathbf { W } = [ \omega _ { i j } ] \in \mathbb { R } ^ { N \times N }$ is positive semi-definite (PSD). This makes the Laplacian term concave with respect to z. Therefore, we adopt a BSUM-style approximation by taking the tight linear upper bound of $\mathbf { z } _ { i }$ at iteration t.

Constructing linear upper bound. To construct such a linear upper bound, we first rewrite the Laplacian term in a matrix form. Let $\mathbf { z } \in \mathbb { R } ^ { N K }$ denote the concatenation of $\{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$ (stacked by samples), and $\mathbf { G } = \mathrm { d i a g } ( \gamma ) \in \mathbf { \bar { \mathbb { R } } } ^ { N \times N }$ be the diagonal matrix of instance-level weights $\gamma .$ Then, the weighted Laplacian term can be written as

$$
- \sum_ {i, j} \gamma_ {i} \omega_ {i j} \mathbf {z} _ {i} ^ {\top} \mathbf {z} _ {j} = - \sum_ {i, j} (\mathbf {G W}) _ {i j} \mathbf {z} _ {i} ^ {\top} \mathbf {z} _ {j} = \mathbf {z} ^ {\top} \boldsymbol {\Psi} \mathbf {z},\tag{71}
$$

where

$$
\boldsymbol {\Psi} \triangleq - (\mathbf {G W}) \otimes \mathbf {I} _ {K},\tag{72}
$$

⊗ denotes the Kronecker product, and ${ \mathbf { I } } _ { K }$ is the $K \times K$ identity matrix. For notational simplicity, we use ${ \mathbf { I } } _ { K }$ since each $\mathbf { z } _ { i } \in \mathbb { R } ^ { K }$ . This is also the standard lifting used in TransCLIP and StatA.

When W is PSD and $\gamma \succeq \mathbf { 0 }$ , the symmetrized weight matrix $\mathbf { \Gamma } _ { 2 } ^ { 1 } \left( \mathbf { G } \mathbf { W } + ( \mathbf { G } \mathbf { W } ) ^ { \top } \right)$ is also $\mathrm { P S D ^ { 1 0 } }$ , which implies that $\Psi$ is negative semi-definite (NSD). As a results, $\mathbf { z } ^ { \top }$ Ψz is concave with respect to z.

For a concave quadratic function $q ( \mathbf { z } ) = \mathbf { z } ^ { \top }$ Ψz with $\Psi \preceq \mathbf { 0 }$ , its first-order Taylor expansion at the current iterate $\mathbf { z } ^ { ( t ) }$ provides a tight global upper bound:

$$
\mathbf {z} ^ {\top} \boldsymbol {\Psi} \mathbf {z} \leq (\mathbf {z} ^ {(t)}) ^ {\top} \boldsymbol {\Psi} \mathbf {z} ^ {(t)} + (\nabla q (\mathbf {z} ^ {(t)})) ^ {\top} (\mathbf {z} - \mathbf {z} ^ {(t)}), \quad \nabla q (\mathbf {z}) = (\boldsymbol {\Psi} + \boldsymbol {\Psi} ^ {\top}) \mathbf {z}.\tag{73}
$$

In particular, if $\Psi$ is symmetric (or using its symmetric part), the gradient simplifies to $\nabla q ( \mathbf { z } ) = 2 \Psi \mathbf { z } .$ , and the bound becomes

$$
\mathbf {z} ^ {\top} \boldsymbol {\Psi} \mathbf {z} \leq (\mathbf {z} ^ {(t)}) ^ {\top} \boldsymbol {\Psi} \mathbf {z} ^ {(t)} + 2 (\boldsymbol {\Psi} \mathbf {z} ^ {(t)}) ^ {\top} (\mathbf {z} - \mathbf {z} ^ {(t)}).\tag{74}
$$

This upper bound is tight in the BSUM sense, i.e., it equals the original quadratic term at ${ \mathbf z } = { \mathbf z } ^ { ( t ) }$ . Moreover, replacing the quadratic coupling by (73) or (74) yields a linear surrogate that decouples across $\left\{ \mathbf { z } _ { i } \right\}$ under simplex constraints, enabling an efficient BSUM update.

Therefore, by fixing the neighbors $\{ \mathbf { z } _ { j } ^ { ( t ) } \} _ { j }$ and upper-bounding the bilinear term, we rewrite the Laplacian term as

$$
- \sum_ {i, j} \gamma_ {i} \omega_ {i j} \mathbf {z} _ {i} ^ {\top} \mathbf {z} _ {j} \approx - \sum_ {i, j} \gamma_ {i} \omega_ {i j} \mathbf {z} _ {i} ^ {\top} \mathbf {z} _ {j} ^ {(t)} + \text { const } = - \sum_ {i} \gamma_ {i} \mathbf {z} _ {i} ^ {\top} \sum_ {j} \omega_ {i j} \mathbf {z} _ {j} ^ {(t)} + \text { const }\tag{75}
$$

Substituting (75) into (70), the problem becomes separable over i.

Per-sample subproblem and cancellation of $\gamma _ { i }$ . For each sample i, we obtain the subproblem

$$
\mathbf {z} _ {i} ^ {(t + 1)} \in \arg \min _ {\mathbf {z} _ {i} \in \Delta^ {K}} \gamma_ {i} \Big (- \mathbf {z} _ {i} ^ {\top} \log \mathbf {p} _ {i} ^ {\mathrm{vMF}} + \mathrm{KL} (\mathbf {z} _ {i} \| \hat {\mathbf {y}} _ {i}) - \mathbf {z} _ {i} ^ {\top} \sum_ {j} \omega_ {i j} \mathbf {z} _ {j} ^ {(t)} \Big).\tag{76}
$$

Since $\gamma _ { i } > 0$ is a constant multiplier in (76), it does not affect the minimizer:

$$
\arg \min _ {\mathbf {z} _ {i} \in \Delta^ {K}} \gamma_ {i}   g _ {i} (\mathbf {z} _ {i}) = \arg \min _ {\mathbf {z} _ {i} \in \Delta^ {K}} g _ {i} (\mathbf {z} _ {i}), \qquad (\gamma_ {i} > 0),\tag{77}
$$

which explains why the assignment update in Eq. (12) does not explicitly depend on $\gamma _ { i }$ .

Closed-form update. Expanding $\begin{array} { r } { \mathrm { K L } ( \mathbf { z } _ { i } \| \hat { \mathbf { y } } _ { i } ) = \sum _ { k } z _ { i , k } \log z _ { i , k } - z _ { i , k } } \end{array}$ log $\hat { y } _ { i , k }$ and omitting $\mathbf { z } _ { i }$ -independent constants, the effective subproblem is

$$
\min _ {\mathbf {z} _ {i} \in \Delta^ {K}} \sum_ {k = 1} ^ {K} z _ {i, k} \log z _ {i, k} - \sum_ {k = 1} ^ {K} z _ {i, k} s _ {i, k} ^ {(t)}, \quad s _ {i, k} ^ {(t)} \triangleq \log \hat {y} _ {i, k} + \log p _ {i, k} ^ {\mathrm{vMF}} + \sum_ {j} \omega_ {i j} z _ {j, k} ^ {(t)}.\tag{78}
$$

Introducing a Lagrange multiplier $\lambda _ { i }$ (different from the class confidence in Eq. (11)) for constraint $\textstyle \sum _ { k } z _ { i , k } = 1$ , solving the Karush-Kuhn-Tucker (KKT) conditions yields

$$
\log z _ {i, k} + 1 - s _ {i, k} ^ {(t)} + \lambda_ {i} = 0 \quad \Longrightarrow \quad z _ {i, k} \propto \exp \bigl (s _ {i, k} ^ {(t)} \bigr).\tag{79}
$$

Since $\mathbf { z } _ { i } \in \Delta _ { k }$ , we obtain the final form after normalization

$$
z _ {i, k} ^ {(t + 1)} = \frac {\exp (s _ {i , k} ^ {(t)})}{\sum_ {r = 1} ^ {K} \exp (s _ {i , r} ^ {(t)})} = \frac {\hat {y} _ {i , k} \exp (\log p _ {i , k} ^ {\mathrm{vMF}} + \sum_ {j} \omega_ {i j} z _ {j , k} ^ {(t)})}{\sum_ {r = 1} ^ {K} \hat {y} _ {i , r} \exp (\log p _ {i , r} + \sum_ {j} \omega_ {i j} z _ {j , r} ^ {(t)})}.\tag{80}
$$

Expressed in vector form, Eq. (80) is given by

$$
\mathbf {z} _ {i} ^ {(t + 1)} = \frac {\hat {\mathbf {y}} _ {i} \odot \exp (\log \mathbf {p} _ {i} ^ {\mathrm{vMF}} + \sum_ {j} \omega_ {i j} \mathbf {z} _ {j} ^ {(t)})}{(\hat {\mathbf {y}} _ {i} \odot \exp (\log \mathbf {p} _ {i} ^ {\mathrm{vMF}} + \sum_ {j} \omega_ {i j} \mathbf {z} _ {j} ^ {(t)})) ^ {\top} \mathbb {1} _ {K}}.\tag{81}
$$

This has the same form as StatA, with log $p _ { i , k } ^ { \mathrm { v M F } } = \log \mathcal { C } _ { d } ( \kappa _ { k } ) + \kappa _ { k } \mu _ { k } ^ { \top } \mathbf { f } _ { i }$ instantiated by our vMF likelihood in Eq. (6).

## G.2. With respect to parameters µ and κ

In this subsection, we derive the closed-form updates for the vMF parameters $\{ \mu _ { k } , \kappa _ { k } \} _ { k = 1 } ^ { K }$ by fixing the soft assignments z. Recall that all feature vectors are ℓ<sub>2</sub>-normalized, i.e., $\| \mathbf { f } _ { i } \| _ { 2 } = 1$

Update of $\pmb { \mu } _ { k }$ . We first consider the mean direction vector $\pmb { \mu } _ { k }$ . Collecting all terms in the PLE objective Eq. (8) that depend on $\pmb { \mu } _ { k } .$ , we obtain

$$
J (\boldsymbol {\mu} _ {k}) = - \sum_ {i} \gamma_ {i} \mathbf {z} _ {i, k} \left(\kappa_ {k} \boldsymbol {\mu} _ {k} ^ {\top} \mathbf {f} _ {i}\right) + \alpha \left(\kappa_ {k} \mathcal {A} _ {d} \left(\kappa_ {k} ^ {\prime}\right) \boldsymbol {\mu} _ {k} ^ {\top} \boldsymbol {\mu} _ {k} ^ {\prime}\right), \quad \text { s   .   t   . } \| \boldsymbol {\mu} _ {k} \| _ {2} = 1.\tag{82}
$$

Minimizing this objective is equivalent to maximizing

$$
\max _ {\boldsymbol {\mu} _ {k}} \kappa_ {k} \boldsymbol {\mu} _ {k} ^ {\top} \left(\sum_ {i} \gamma_ {i} \mathbf {z} _ {i, k} \mathbf {f} _ {i} + \alpha \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \boldsymbol {\mu} _ {k} ^ {\prime}\right), \quad \text { s.t. } \| \boldsymbol {\mu} _ {k} \| _ {2} = 1,\tag{83}
$$

Let

$$
R _ {k} \triangleq \sum_ {i} \gamma_ {i} \mathbf {z} _ {i, k} \mathbf {f} _ {i}, \qquad T _ {k} \triangleq \alpha \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \boldsymbol {\mu} _ {k} ^ {\prime},\tag{84}
$$

and define the combined resultant vector

$$
R _ {k} ^ {\mathrm{tot}} \triangleq R _ {k} + T _ {k}.\tag{85}
$$

Then, the objective in (83) reduces to maximizing $\mu _ { k } ^ { \top } R _ { k } ^ { \mathrm { t o t } }$ under a unit-norm constraint. The optimum is achieved when $\pmb { \mu } _ { k }$ aligns with $R _ { k } ^ { \mathrm { t o t } }$ , yielding

$$
\boldsymbol {\mu} _ {k} = \frac {R _ {k} ^ {\mathrm{tot}}}{\| R _ {k} ^ {\mathrm{tot}} \| _ {2}} = \frac {\sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} \mathbf {f} _ {i} + \alpha \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \boldsymbol {\mu} _ {k} ^ {\prime}}{\| \sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} \mathbf {f} _ {i} + \alpha \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2}}.\tag{86}
$$

Update of $\kappa _ { k }$ . Considering concentration parameter $\kappa _ { k }$ . Similarly, when fixing $\pmb { \mu } _ { k }$ , the terms involving $\kappa _ { k }$ in Eq. (8) can be written as

$$
\mathcal {L} (\kappa_ {k}) = - (N _ {k} + \alpha) \log \mathcal {C} _ {d} (\kappa_ {k}) - \kappa_ {k} \boldsymbol {\mu} _ {k} ^ {\top} \left(\sum_ {i} \gamma_ {i} \mathbf {z} _ {i, k} \mathbf {f} _ {i} + \alpha \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \boldsymbol {\mu} _ {k} ^ {\prime}\right),\tag{87}
$$

where $\begin{array} { r } { N _ { k } = \sum _ { i } \gamma _ { i } \mathbf { z } _ { i , k } } \end{array}$ denotes the soft assignment count (with weights $\gamma _ { i } )$

From the update of $\pmb { \mu } _ { k }$ in (86), the direction of $\pmb { \mu } _ { k }$ coincides with that of $R _ { k } ^ { \mathrm { t o t } }$ , and thus

$$
\boldsymbol {\mu} _ {k} ^ {\top} R _ {k} ^ {\mathrm{tot}} = \| R _ {k} ^ {\mathrm{tot}} \| _ {2}.\tag{88}
$$

Denoting $\bar { R } _ { k } ^ { \mathrm { t o t } } \triangleq \| R _ { k } ^ { \mathrm { t o t } } \| _ { 2 }$ , the objective (87) becomes

$$
\mathcal {L} (\kappa_ {k}) = - (N _ {k} + \alpha) \log \mathcal {C} _ {d} (\kappa_ {k}) - \kappa_ {k} \bar {R} _ {k} ^ {\mathrm{tot}}.\tag{89}
$$

Taking the derivative with respect to $\kappa _ { k }$ and setting it to zero yields

$$
\frac {\partial L (\kappa_ {k})}{\partial \kappa} = - (N _ {k} + \alpha) \frac {\partial}{\partial \kappa} \log \mathcal {C} _ {d} (\kappa_ {k}) - \frac {\partial}{\partial \kappa} \kappa_ {k} \bar {R} _ {k} ^ {\mathrm{tot}}\tag{90}
$$

$$
= - (N _ {k} + \alpha) \frac {\partial}{\partial \kappa} \log \mathcal {C} _ {d} (\kappa_ {k}) - \bar {R} _ {k} ^ {\mathrm{tot}}
$$

$$
= (N _ {k} + \alpha) \mathcal {A} _ {d} (\kappa) - \bar {R} _ {k} ^ {\mathrm{tot}}\tag{91}
$$

(92)

(93)

where we used the standard vMF identity $\begin{array} { r } { \frac { \partial } { \partial \kappa } \log \mathcal { C } _ { d } ( \kappa ) = - \mathcal { A } _ { d } ( \kappa ) } \end{array}$ . Therefore, $\kappa _ { k }$ satisfies

$$
\mathcal {A} _ {d} (\kappa_ {k}) = \frac {\bar {R} _ {k} ^ {\mathrm{tot}}}{N _ {k} + \alpha} = \frac {\| \sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} \mathbf {f} _ {i} + \alpha \mathcal {A} _ {d} (\kappa_ {k} ^ {\prime}) \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2}}{\sum_ {i} \gamma_ {i} \mathbf {z} _ {i , k} + \alpha}.\tag{94}
$$

Then, $\kappa _ { k }$ is approximated as the inverse of $\boldsymbol { \mathcal { A } } _ { d } ( \kappa _ { k } )$ using Eq. (15).

## H. Equivalence to the Adaptive Shrinkage Form for Parameter Updates

In this section, we show that the parameter updates in Eq. (13) can be written in an equivalent adaptive shrinkage form, i.e., Eq. (14). For class $k ,$ define

$$
N _ {k} \triangleq \sum_ {i} \gamma_ {i} \mathbf {z} _ {i, k}, \qquad R _ {k} \triangleq \sum_ {i} \gamma_ {i} \mathbf {z} _ {i, k} \mathbf {f} _ {i}, \qquad \pmb {v} _ {k} \triangleq \frac {R _ {k}}{N _ {k}}, \qquad \beta_ {k} \triangleq \frac {N _ {k}}{N _ {k} + \alpha}.\tag{95}
$$

Here, we set α as a fixed scalar for simplicity.

Remark. To make the equivalence more transparent, we first present the proof under the mild simplification $\mathcal { A } _ { d } ( \kappa _ { k } ^ { \prime } ) = 1$ i.e., distribution anchor provides a deterministic direction (which is intuitive and natural). The same derivation holds for general $\mathcal { A } _ { d } ( \kappa _ { k } ^ { \prime } ) \neq 1$ by replacing α with $\alpha \mathcal { A } _ { d } ( \kappa _ { k } ^ { \prime } )$ ).

With respect to $\pmb { \mu } _ { k }$ . Let

$$
V _ {k} ^ {\text { ours }} \triangleq R _ {k} + \alpha \boldsymbol {\mu} _ {k} ^ {\prime} = \sum_ {i} \gamma_ {i} \mathbf {z} _ {i, k} \mathbf {f} _ {i} + \alpha \boldsymbol {\mu} _ {k} ^ {\prime}, \quad V _ {k} ^ {\text { StatA }} \triangleq \beta_ {k} \mathbf {v} _ {k} + (1 - \beta_ {k}) \boldsymbol {\mu} _ {k} ^ {\prime}.\tag{96}
$$

Eq. (13) updates $\pmb { \mu } _ { k }$ by normalizing $V _ { k } ^ { \mathrm { o u r s } }$ , i.e.,

$$
\boldsymbol {\mu} _ {k} = \frac {V _ {k} ^ {\text { ours }}}{\| V _ {k} ^ {\text { ours }} \| _ {2}}.\tag{97}
$$

We now show that this is equivalent to normalizing $V _ { k } ^ { \mathrm { S t a t A } }$ . Indeed, substituting the definitions in (95) yields

$$
V _ {k} ^ {\mathrm{StatA}} = \left(\frac {N _ {k}}{N _ {k} + \alpha}\right) \frac {R _ {k}}{N _ {k}} + \left(\frac {\alpha}{N _ {k} + \alpha}\right) \boldsymbol {\mu} _ {k} ^ {\prime}\tag{98}
$$

$$
= \frac {R _ {k}}{N _ {k} + \alpha} + \frac {\alpha \pmb {\mu} _ {k} ^ {\prime}}{N _ {k} + \alpha}\tag{99}
$$

$$
= \frac {1}{N _ {k} + \alpha} (R _ {k} + \alpha \pmb {\mu} _ {k} ^ {\prime})\tag{100}
$$

$$
= \frac {1}{N _ {k} + \alpha} V _ {k} ^ {\mathrm{ours}}.\tag{101}
$$

Therefore, $V _ { k } ^ { \mathrm { S t a t A } }$ and $V _ { k } ^ { \mathrm { o u r s } }$ are colinear and share the same direction. After normalization, we obtain the exact equivalence

$$
\frac {V _ {k} ^ {\mathrm{ours}}}{\| V _ {k} ^ {\mathrm{ours}} \| _ {2}} \equiv \frac {V _ {k} ^ {\mathrm{StatA}}}{\| V _ {k} ^ {\mathrm{StatA}} \| _ {2}} = \frac {\beta_ {k} \boldsymbol {v} _ {k} + (1 - \beta_ {k}) \boldsymbol {\mu} _ {k} ^ {\prime}}{\| \beta_ {k} \boldsymbol {v} _ {k} + (1 - \beta_ {k}) \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2}},\tag{102}
$$

which proves the equivalence between Eq. (13) and (14).

With respect to $\kappa _ { k }$ . Eq. (13) computes

$$
\mathcal {A} _ {d} (\kappa_ {k}) = \frac {\| R _ {k} + \alpha \pmb {\mu} _ {k} ^ {\prime} \| _ {2}}{N _ {k} + \alpha} = \frac {\| V _ {k} ^ {\mathrm{ours}} \| _ {2}}{N _ {k} + \alpha}.\tag{103}
$$

According to Eq. (101), we have

$$
\left\| \beta_ {k} \boldsymbol {v} _ {k} + (1 - \beta_ {k}) \boldsymbol {\mu} _ {k} ^ {\prime} \right\| _ {2} = \left\| V _ {k} ^ {\text {StatA}} \right\| _ {2} = \left\| \frac {1}{N _ {k} + \alpha} V _ {k} ^ {\text {ours}} \right\| _ {2}\tag{104}
$$

$$
= \frac {1}{N _ {k} + \alpha} \| V _ {k} ^ {\mathrm{ours}} \| _ {2}\tag{105}
$$

$$
\underline {{\| R _ {k} + \alpha \boldsymbol {\mu} _ {k} ^ {\prime} \| _ {2}}}
$$

$$
N _ {k} + \alpha\tag{106}
$$

$$
= \mathcal {A} _ {d} (\kappa_ {k}).\tag{107}
$$

Thus, the concentration update in Eq. (13) is also exactly equivalent to the form in Eq. (14).

## I. Additional Experimental Results

## I.1. Results with different batch sizes

To verify the robustness of MOON concerning data availability, we conduct experiments with varying batch sizes as reported in Tab. 11. Across three different batch sizes and two realistic scenarios, our method consistently outperforms all baselines and significantly improves upon the zero-shot CLIP initialization. This indicates the effectiveness of MOON, regardless of whether the incoming data batch is sparse or abundant, ensuring reliable adaptation performance across different data scales.

Table 11. Results with different batch sizes. The best and second-best results are marked in bold and underlined, respectively. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.

<table><tr><td rowspan="2"> $K_{\text{eff}}$ </td><td rowspan="2">Method</td><td colspan="11">(a) Batch Size: 126.</td><td rowspan="2">Avg.</td></tr><tr><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td></tr><tr><td></td><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td> $72.0_{+5.4}$ </td><td> $66.5_{+4.0}$ </td><td> $26.6_{+1.9}$ </td><td> $48.2_{-0.1}$ </td><td> $72.7_{+7.1}$ </td><td> $88.7_{+2.8}$ </td><td> $92.1_{+3.0}$ </td><td> $75.4_{+4.7}$ </td><td> $93.7_{+0.5}$ </td><td> $47.1_{+4.2}$ </td><td> $70.0_{+2.5}$ </td><td> $68.5_{+3.3}$ </td></tr><tr><td>MOON</td><td> $82.3_{+15.7}$ </td><td> $76.2_{+13.7}$ </td><td> $27.1_{+2.4}$ </td><td> $46.2_{-2.1}$ </td><td> $75.7_{+10.1}$ </td><td> $93.8_{+7.9}$ </td><td> $92.1_{+3.0}$ </td><td> $75.9_{+5.2}$ </td><td> $94.8_{+1.6}$ </td><td> $47.1_{+3.6}$ </td><td> $72.9_{+5.4}$ </td><td> $71.3_{+6.1}$ </td></tr><tr><td rowspan="2">High</td><td>StatA</td><td> $69.4_{+2.8}$ </td><td> $64.9_{+2.4}$ </td><td> $23.6_{-1.1}$ </td><td> $47.2_{-1.1}$ </td><td> $68.0_{+2.4}$ </td><td> $87.0_{+1.1}$ </td><td> $88.2_{-0.9}$ </td><td> $72.0_{+1.3}$ </td><td> $94.0_{+0.8}$ </td><td> $46.9_{+3.4}$ </td><td> $68.2_{+0.7}$ </td><td> $66.3_{+1.1}$ </td></tr><tr><td>MOON</td><td> $76.4_{+9.8}$ </td><td> $69.6_{+7.1}$ </td><td> $21.7_{-3.0}$ </td><td> $46.2_{-2.1}$ </td><td> $68.6_{+3.0}$ </td><td> $88.9_{+3.0}$ </td><td> $86.4_{-2.7}$ </td><td> $71.3_{+0.6}$ </td><td> $93.7_{+0.5}$ </td><td> $40.2_{-3.3}$ </td><td> $67.1_{-0.4}$ </td><td> $66.4_{+1.1}$ </td></tr><tr><td colspan="14">(b) Batch Size: 256.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td> $72.0_{+5.4}$ </td><td> $66.7_{+4.2}$ </td><td> $27.1_{+2.4}$ </td><td> $56.0_{+7.7}$ </td><td> $74.1_{+8.5}$ </td><td> $88.9_{+3.0}$ </td><td> $92.9_{+3.8}$ </td><td> $76.0_{+5.3}$ </td><td> $93.6_{+0.4}$ </td><td> $47.0_{+3.5}$ </td><td> $70.5_{+3.0}$ </td><td> $69.5_{+4.3}$ </td></tr><tr><td>MOON</td><td> $82.9_{+16.3}$ </td><td> $77.8_{+15.3}$ </td><td> $27.9_{+3.2}$ </td><td> $49.4_{+1.1}$ </td><td> $76.6_{+11.0}$ </td><td> $94.1_{+8.2}$ </td><td> $92.9_{+3.8}$ </td><td> $76.1_{+5.4}$ </td><td> $95.2_{+2.0}$ </td><td> $48.4_{+4.9}$ </td><td> $73.7_{+6.2}$ </td><td> $72.3_{+7.0}$ </td></tr><tr><td rowspan="2">High</td><td>StatA</td><td> $71.1_{+4.5}$ </td><td> $66.3_{+3.8}$ </td><td> $24.2_{-0.5}$ </td><td> $55.5_{+7.2}$ </td><td> $70.6_{+5.0}$ </td><td> $87.6_{+1.7}$ </td><td> $88.9_{-0.2}$ </td><td> $73.7_{+3.0}$ </td><td> $94.1_{+0.9}$ </td><td> $47.0_{+3.5}$ </td><td> $69.9_{+2.4}$ </td><td> $68.1_{+2.9}$ </td></tr><tr><td>MOON</td><td> $80.5_{+13.9}$ </td><td> $73.2_{+10.7}$ </td><td> $23.2_{-1.5}$ </td><td> $49.1_{+0.8}$ </td><td> $71.8_{+6.2}$ </td><td> $90.4_{+4.5}$ </td><td> $88.2_{-0.9}$ </td><td> $73.1_{+2.4}$ </td><td> $94.1_{+0.9}$ </td><td> $42.7_{-0.8}$ </td><td> $69.6_{+2.1}$ </td><td> $68.7_{+3.5}$ </td></tr><tr><td colspan="14">(c) Batch Size: 500.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td></td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td> $71.5_{+4.9}$ </td><td> $65.5_{+3.0}$ </td><td> $27.8_{+3.1}$ </td><td> $59.3_{+11.0}$ </td><td> $74.9_{+9.3}$ </td><td> $88.3_{+2.4}$ </td><td> $93.1_{+4.0}$ </td><td> $76.8_{+6.1}$ </td><td> $93.1_{-0.1}$ </td><td> $47.1_{+3.6}$ </td><td> $69.9_{+2.4}$ </td><td> $69.8_{+4.5}$ </td></tr><tr><td>MOON</td><td> $82.4_{+15.8}$ </td><td> $76.7_{+14.2}$ </td><td> $28.6_{+3.9}$ </td><td> $55.5_{+7.2}$ </td><td> $77.0_{+11.4}$ </td><td> $93.7_{+7.8}$ </td><td> $92.7_{+3.6}$ </td><td> $76.5_{+5.8}$ </td><td> $95.0_{+1.8}$ </td><td> $49.0_{+5.5}$ </td><td> $73.4_{+5.9}$ </td><td> $72.8_{+7.5}$ </td></tr><tr><td rowspan="2">High</td><td>StatA</td><td> $72.1_{+5.5}$ </td><td> $67.3_{+4.8}$ </td><td> $25.1_{+0.4}$ </td><td> $60.0_{+11.7}$ </td><td> $72.3_{+6.7}$ </td><td> $88.2_{+2.3}$ </td><td> $90.3_{+1.2}$ </td><td> $75.5_{+4.8}$ </td><td> $93.8_{+0.6}$ </td><td> $47.2_{+3.7}$ </td><td> $70.7_{+3.2}$ </td><td> $69.3_{+4.1}$ </td></tr><tr><td>MOON</td><td> $82.3_{+15.7}$ </td><td> $75.5_{+13.0}$ </td><td> $24.3_{-0.4}$ </td><td> $55.8_{+7.5}$ </td><td> $73.8_{+8.2}$ </td><td> $91.2_{+5.3}$ </td><td> $89.3_{+0.2}$ </td><td> $74.5_{+3.8}$ </td><td> $94.7_{+1.5}$ </td><td> $44.1_{+0.6}$ </td><td> $70.8_{+3.3}$ </td><td> $70.6_{+5.3}$ </td></tr></table>

Table 12. Overview of the VLMs employed in our experiments.

<table><tr><td>Model name</td><td>Full name</td><td>#Param</td><td>Detailed #Param</td><td>FLOPS (B)</td><td>Resolution</td><td>Pretrained</td></tr><tr><td>CLIP</td><td>clip-resnet50 / RN50</td><td>102M</td><td>102,007,137</td><td>18.18</td><td>224x224</td><td>WIT</td></tr><tr><td>CLIP</td><td>clip-resnet101 / RN101</td><td>120M</td><td>119,688,033</td><td>25.50</td><td>224x224</td><td>WIT</td></tr><tr><td>CLIP</td><td>clip-vit-base-patch16 / ViT-B/16</td><td>150M</td><td>149,620,737</td><td>41.09</td><td>224x224</td><td>WIT</td></tr><tr><td>CLIP</td><td>clip-vit-base-patch32 / ViT-B/32</td><td>151M</td><td>151,277,313</td><td>14.78</td><td>224x224</td><td>WIT</td></tr><tr><td>CLIP</td><td>clip-vit-large-patch14 / ViT-L/14</td><td>428M</td><td>427,616,513</td><td>175.33</td><td>224x224</td><td>WIT</td></tr><tr><td>OpenCLIP</td><td>ViT-B-16</td><td>150M</td><td>149,620,737</td><td>41.09</td><td>224x224</td><td>DataComp</td></tr><tr><td>SigLIP</td><td>ViT-SO400M-14-SigLIP-384</td><td>878M</td><td>877,960,498</td><td>723.48</td><td>384x384</td><td>WebLI</td></tr><tr><td>EVA-CLIP</td><td>EVA01-g-14</td><td>1.14B</td><td>1,136,435,841</td><td>547.36</td><td>224x224</td><td>LAION</td></tr></table>

## I.2. Results with other backbones

Following the discussion in Sec. 5.3, we extend our evaluation to four additional CLIP visual backbones, including ResNet (He et al., 2016) architectures (RN50, RN101) and Vision Transformers (ViT) (Dosovitskiy et al., 2021) of varying scales (ViT-B/32, ViT-L/14), as detailed in Tab. 13, 14 and 15. These experiments cover the full range of realistic batch and online adaptation settings. MOON demonstrates remarkable universality, achieving the highest accuracy in nearly all evaluated scenarios (winning in 33 out of 36 cases) with only negligible margins in the few exceptions. This consistent superiority across diverse architectures and model capacities confirms that our vMF-based dynamic shrinkage mechanism is a generalized solution, capable of enhancing VLM performance regardless of the specific underlying visual encoder.

## I.3. Results with other VLM architectures

To validate the generalizability of our MOON beyond standard CLIP models, we evaluate three additional VLMs with diverse architectures, parameter scales, and training procedures: OpenCLIP (151M) (Cherti et al., 2023), SigLIP (878M) (Zhai et al., 2023), and EVA-CLIP (1.1B) (Sun et al., 2023), all implemented using the OpenCLIP codebase (Ilharco et al., 2021). Details regarding the VLMs in our experiment is presented in Tab. 12. Experiments conducted with a batch size of 1,000 across three realistic scenarios (Tables 16, 17, and 18) reveal that MOON consistently enhances zero-shot performance and achieves state-of-the-art results in nearly all nine settings. Notably, the relative improvements are particularly distinct for smaller models (e.g., OpenCLIP), suggesting our method effectively compensates for weaker initial representations, while maintaining substantial gains on challenging large-scale datasets like ImageNet and SUN397. Collectively, these findings demonstrate the universal effectiveness of MOON across diverse model architectures, scales, and pre-training paradigms.

## I.4. Results on full dataset with all classes

We further evaluate the extreme scenario where the model adapts to the full dataset containing all classes simultaneously, representing a dense label distribution that deviates from our sparsity assumption. As shown in Tab. 19, MOON trails the strongest baseline StatA by a marginal gap (∼2%). This behavior is consistent with our analysis in the main text: our dynamic shrinkage mechanism is inherently designed with an inductive bias to favor sparse effective class sets, which is less optimal when the ground-truth distribution is uniform. However, unlike many specialized methods that collapse when assumptions are violated, MOON maintains competitive high-accuracy performance without severe degradation. Considering its significant computational efficiency, our method offers a robust trade-off, serving as a reliable solution even in scenarios with maximal class presence.

## I.5. Results on random class scenarios

To simulate a highly unpredictable deployment environment, we introduce a challenging ”Random” scenario where the number of effective classes $K _ { e f f }$ varies stochastically between 1 and min{N, K} for each adaptation step. This setting effectively aggregates the characteristics of varying sparsity levels into a single dynamic evaluation, with results demonstrated in Tab. 20. Under these volatile conditions, MOON exhibits remarkable stability, achieving performance comparable to the state-of-the-art StatA across varying batch sizes. The fact that our method matches the strongest baseline in accuracy while operating with significantly lower latency and a simpler optimization procedure (as detailed in Appendix A) further underscores its practicality for handling real-world data streams with unknown and fluctuating statistics.

Table 13. Results on four additional CLIP backbones, batch adaptation with batch size of 64. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.  
(a) ResNet-50.

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>58.2</td><td>58.9</td><td>17.0</td><td>36.2</td><td>55.8</td><td>77.4</td><td>85.7</td><td>66.1</td><td>85.7</td><td>42.8</td><td>61.8</td><td>58.7</td></tr><tr><td rowspan="2">Very Low</td><td>StatA</td><td>68.2+10.0</td><td>63.7+4.8</td><td>21.1+4.1</td><td>43.3+7.1</td><td>71.3+15.5</td><td>87.1+9.7</td><td>93.1+7.4</td><td>74.1+8.0</td><td>90.1+4.4</td><td>45.3+2.5</td><td>66.2+4.4</td><td>65.8+7.1</td></tr><tr><td>MOON</td><td>79.7+21.5</td><td>77.5+18.6</td><td>25.4+8.4</td><td>41.8+5.6</td><td>74.8+19.0</td><td>94.9+17.5</td><td>95.5+9.8</td><td>75.2+9.1</td><td>92.2+6.5</td><td>54.4+11.6</td><td>74.1+12.3</td><td>71.4+12.7</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>65.0+6.8</td><td>62.8+3.9</td><td>17.8+0.8</td><td>31.7-4.5</td><td>67.1+11.3</td><td>83.6+6.2</td><td>88.2+2.5</td><td>71.7+5.6</td><td>89.0+3.3</td><td>44.9+2.1</td><td>64.0+2.2</td><td>62.3+3.7</td></tr><tr><td>MOON</td><td>79.5+21.3</td><td>77.9+19.0</td><td>21.9+4.9</td><td>35.2-1.0</td><td>72.8+17.0</td><td>92.9+15.5</td><td>94.0+8.3</td><td>75.6+9.5</td><td>91.7+6.0</td><td>49.2+6.4</td><td>71.6+9.8</td><td>69.3+10.6</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>61.1+2.9</td><td>59.5+0.6</td><td>16.4-0.6</td><td>27.3-8.9</td><td>62.1+6.3</td><td>78.7+1.3</td><td>81.4-4.3</td><td>64.1-2.0</td><td>87.9+2.2</td><td>43.8+1.0</td><td>62.0+0.2</td><td>58.6-0.1</td></tr><tr><td>MOON</td><td>73.6+15.4</td><td>71.3+12.4</td><td>18.6+1.6</td><td>31.9-4.3</td><td>67.2+11.4</td><td>87.6+10.2</td><td>88.5+2.8</td><td>70.5+4.4</td><td>89.8+4.1</td><td>42.2-0.6</td><td>67.5+5.7</td><td>64.4+5.7</td></tr><tr><td colspan="14">(b) ResNet-101.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>61.3</td><td>59.0</td><td>17.9</td><td>32.9</td><td>63.2</td><td>80.7</td><td>86.9</td><td>64.3</td><td>89.9</td><td>37.3</td><td>61.1</td><td>59.5</td></tr><tr><td rowspan="2">Very Low</td><td>StatA</td><td>73.0+11.7</td><td>66.5+7.5</td><td>22.5+4.6</td><td>30.4-2.5</td><td>76.2+13.0</td><td>89.5+8.8</td><td>95.2+8.3</td><td>74.6+10.3</td><td>91.9+2.0</td><td>42.9+5.6</td><td>65.1+4.0</td><td>66.2+6.7</td></tr><tr><td>MOON</td><td>80.0+18.7</td><td>74.5+15.5</td><td>24.8+6.9</td><td>30.8-2.1</td><td>77.9+14.7</td><td>94.1+13.4</td><td>93.9+7.0</td><td>72.4+8.1</td><td>93.2+3.3</td><td>46.1+8.8</td><td>70.3+9.2</td><td>68.9+9.4</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>71.2+9.9</td><td>65.9+6.9</td><td>20.0+2.1</td><td>29.6-3.3</td><td>73.1+9.9</td><td>88.1+7.4</td><td>92.9+6.0</td><td>74.9+10.6</td><td>92.8+2.9</td><td>42.9+5.6</td><td>64.4+3.3</td><td>65.1+5.6</td></tr><tr><td>MOON</td><td>79.8+18.5</td><td>74.5+15.5</td><td>21.8+3.9</td><td>30.0-2.9</td><td>76.4+13.2</td><td>93.1+12.4</td><td>92.5+5.6</td><td>72.6+8.3</td><td>94.0+4.1</td><td>42.6+5.3</td><td>68.4+7.3</td><td>67.8+8.3</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>67.0+5.7</td><td>62.7+3.7</td><td>18.6+0.7</td><td>28.7-4.2</td><td>69.6+6.4</td><td>84.9+4.2</td><td>88.8+1.9</td><td>70.1+5.8</td><td>92.1+2.2</td><td>40.9+3.6</td><td>63.6+2.5</td><td>62.5+3.0</td></tr><tr><td>MOON</td><td>74.5+13.2</td><td>69.1+10.1</td><td>19.7+1.8</td><td>29.9-3.0</td><td>72.7+9.5</td><td>89.0+8.3</td><td>89.0+2.1</td><td>68.2+3.9</td><td>92.6+2.7</td><td>37.5+0.2</td><td>65.3+4.2</td><td>64.3+4.8</td></tr><tr><td colspan="14">(c) ViT-B/32.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>62.0</td><td>62.1</td><td>19.1</td><td>45.4</td><td>60.2</td><td>80.4</td><td>87.3</td><td>66.6</td><td>91.4</td><td>42.7</td><td>63.5</td><td>61.9</td></tr><tr><td rowspan="2">Very Low</td><td>StatA</td><td>68.1+6.1</td><td>65.6+3.5</td><td>23.0+3.9</td><td>53.2+7.8</td><td>71.9+11.7</td><td>85.8+5.4</td><td>94.3+7.0</td><td>74.9+8.3</td><td>93.4+2.0</td><td>45.2+2.5</td><td>64.5+1.0</td><td>67.3+5.4</td></tr><tr><td>MOON</td><td>80.9+18.9</td><td>78.4+16.3</td><td>25.7+6.6</td><td>52.2+6.8</td><td>77.8+17.6</td><td>94.8+14.4</td><td>94.7+7.4</td><td>75.4+8.8</td><td>95.1+3.7</td><td>53.6+10.9</td><td>70.4+6.9</td><td>72.6+10.7</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>67.2+5.2</td><td>65.7+3.6</td><td>21.9+2.8</td><td>50.1+4.7</td><td>69.3+9.1</td><td>84.5+4.1</td><td>92.5+5.2</td><td>75.3+8.7</td><td>93.2+1.8</td><td>46.1+3.4</td><td>64.6+1.1</td><td>66.4+4.5</td></tr><tr><td>MOON</td><td>80.3+18.3</td><td>78.8+16.7</td><td>23.8+4.7</td><td>45.6+0.2</td><td>77.0+16.8</td><td>93.7+13.3</td><td>93.1+5.8</td><td>76.4+9.8</td><td>94.8+3.4</td><td>47.9+5.2</td><td>69.6+6.1</td><td>71.0+9.1</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>65.5+3.5</td><td>64.8+2.7</td><td>20.0+0.9</td><td>45.4±0.0</td><td>65.0+4.8</td><td>82.6+2.2</td><td>89.4+2.1</td><td>70.6+4.0</td><td>92.9+1.5</td><td>46.7+4.0</td><td>64.4+0.9</td><td>64.3+2.4</td></tr><tr><td>MOON</td><td>75.6+13.6</td><td>73.6+11.5</td><td>20.3+1.2</td><td>40.8-4.6</td><td>70.8+10.6</td><td>89.1+8.7</td><td>89.6+2.3</td><td>71.1+4.5</td><td>93.6+2.2</td><td>42.0-0.7</td><td>67.1+3.6</td><td>66.7+4.8</td></tr><tr><td colspan="14">(d) ViT-L/14.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>73.5</td><td>67.7</td><td>32.5</td><td>60.3</td><td>76.9</td><td>90.9</td><td>93.5</td><td>79.5</td><td>95.2</td><td>53.5</td><td>74.9</td><td>72.6</td></tr><tr><td rowspan="2">Very Low</td><td>StatA</td><td>78.9+5.4</td><td>71.3+3.6</td><td>40.4+7.9</td><td>71.4+11.1</td><td>84.4+7.5</td><td>94.2+3.3</td><td>97.1+3.6</td><td>82.9+3.4</td><td>97.0+1.8</td><td>55.3+1.8</td><td>77.1+2.2</td><td>77.3+4.7</td></tr><tr><td>MOON</td><td>85.8+12.3</td><td>80.1+12.4</td><td>41.3+8.8</td><td>72.4+12.1</td><td>86.2+9.3</td><td>97.9+7.0</td><td>97.5+4.0</td><td>83.5+4.0</td><td>98.1+2.9</td><td>64.7+11.2</td><td>80.8+5.9</td><td>80.8+8.2</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>78.2+4.7</td><td>71.6+3.9</td><td>38.4+5.9</td><td>65.6+5.3</td><td>82.4+5.5</td><td>93.1+2.2</td><td>96.3+2.8</td><td>82.8+3.3</td><td>96.1+0.9</td><td>55.4+1.9</td><td>76.8+1.9</td><td>76.1+3.5</td></tr><tr><td>MOON</td><td>86.4+12.9</td><td>80.8+13.1</td><td>39.7+7.2</td><td>65.2+4.9</td><td>85.2+8.3</td><td>97.2+6.3</td><td>97.5+4.0</td><td>83.7+4.2</td><td>97.7+2.5</td><td>60.7+7.2</td><td>79.7+4.8</td><td>79.4+6.8</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>76.6+3.1</td><td>70.0+2.3</td><td>36.4+3.9</td><td>62.6+2.3</td><td>80.6+3.7</td><td>92.1+1.2</td><td>93.9+0.4</td><td>80.8+1.3</td><td>95.6+0.4</td><td>54.6+1.1</td><td>77.1+2.2</td><td>74.5+2.0</td></tr><tr><td>MOON</td><td>83.3+9.8</td><td>76.3+8.6</td><td>36.3+3.8</td><td>61.4+1.1</td><td>83.2+6.3</td><td>95.4+4.5</td><td>94.9+1.4</td><td>81.5+2.0</td><td>96.7+1.5</td><td>54.5+1.0</td><td>78.2+3.3</td><td>76.5+3.9</td></tr></table>

Table 14. Results on four additional CLIP backbones, batch adaptation with batch size of 1,000. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.  
(a) ResNet-50.

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td><img src="images/8700e270da0d3ac042b0170475456c1b2540798470268cdb8bc2572163f69939.jpg"/></td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>58.2</td><td>58.9</td><td>17.0</td><td>36.2</td><td>55.8</td><td>77.4</td><td>85.7</td><td>66.1</td><td>85.7</td><td>42.8</td><td>61.8</td><td>58.7</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>65.2+7.0</td><td>61.5+2.6</td><td>18.6+1.6</td><td>51.2+15.0</td><td>67.2+11.4</td><td>80.9+3.5</td><td>89.1+3.4</td><td>70.7+4.6</td><td>88.5+2.8</td><td>46.8+4.0</td><td>65.3+3.5</td><td>64.1+5.4</td></tr><tr><td>MOON</td><td>78.2+20.0</td><td>75.1+16.2</td><td>21.3+4.3</td><td>43.3+7.1</td><td>70.6+14.8</td><td>87.5+10.1</td><td>89.7+4.0</td><td>73.4+7.3</td><td>91.1+5.4</td><td>47.8+5.0</td><td>72.2+10.4</td><td>68.2+9.5</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>65.4+7.2</td><td>63.1+4.2</td><td>16.5-0.5</td><td>51.7+15.5</td><td>65.4+9.6</td><td>81.0+3.6</td><td>84.4-1.3</td><td>70.0+3.9</td><td>88.3+2.6</td><td>47.2+4.4</td><td>66.0+4.2</td><td>63.5+4.8</td></tr><tr><td>MOON</td><td>77.5+19.3</td><td>74.1+15.2</td><td>18.1+1.1</td><td>43.2+7.0</td><td>67.5+11.7</td><td>84.3+6.9</td><td>86.5+0.8</td><td>71.3+5.2</td><td>89.3+3.6</td><td>43.6+0.8</td><td>69.2+7.4</td><td>65.9+7.2</td></tr><tr><td rowspan="2">Very High</td><td>StatA</td><td>63.5+5.3</td><td>62.4+3.5</td><td>14.8-2.2</td><td>51.7+15.5</td><td>60.8+5.0</td><td>77.8+0.4</td><td>83.5-2.2</td><td>66.2+0.1</td><td>87.9+2.2</td><td>46.6+3.8</td><td>64.5+2.7</td><td>61.8+3.1</td></tr><tr><td>MOON</td><td>74.7+16.5</td><td>70.5+11.6</td><td>16.7-0.3</td><td>43.2+7.0</td><td>62.3+6.5</td><td>79.6+2.2</td><td>85.9+0.2</td><td>68.3+2.2</td><td>86.1+0.4</td><td>42.5-0.3</td><td>64.8+3.0</td><td>63.1+4.4</td></tr><tr><td colspan="14">(b) ResNet-101.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>61.3</td><td>59.0</td><td>17.9</td><td>32.9</td><td>63.2</td><td>80.7</td><td>86.9</td><td>64.3</td><td>89.9</td><td>37.3</td><td>61.1</td><td>59.5</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>70.5+9.2</td><td>65.3+6.3</td><td>20.5+2.6</td><td>33.6+0.7</td><td>73.9+10.7</td><td>85.4+4.7</td><td>91.1+4.2</td><td>73.1+8.8</td><td>92.2+2.3</td><td>43.2+5.9</td><td>66.5+5.4</td><td>65.0+5.5</td></tr><tr><td>MOON</td><td>77.6+16.3</td><td>72.6+13.6</td><td>21.5+3.6</td><td>33.3+0.4</td><td>75.3+12.1</td><td>88.9+8.2</td><td>89.6+2.7</td><td>71.2+6.9</td><td>94.2+4.3</td><td>41.6+4.3</td><td>68.7+7.6</td><td>66.8+7.3</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>71.4+10.1</td><td>66.2+7.2</td><td>18.6+0.7</td><td>32.8-0.1</td><td>72.2+9.0</td><td>85.1+4.4</td><td>87.9+1.0</td><td>71.9+7.6</td><td>92.2+2.3</td><td>42.5+5.2</td><td>66.5+5.4</td><td>64.3+4.8</td></tr><tr><td>MOON</td><td>77.7+16.4</td><td>71.2+12.2</td><td>19.2+1.3</td><td>33.1+0.2</td><td>72.6+9.4</td><td>86.3+5.6</td><td>87.6+0.7</td><td>69.1+4.8</td><td>93.1+3.2</td><td>38.7+1.4</td><td>65.9+4.8</td><td>65.0+5.5</td></tr><tr><td rowspan="2">Very High</td><td>StatA</td><td>70.1+8.8</td><td>65.4+6.4</td><td>16.9-1.0</td><td>32.9+0.0</td><td>68.2+5.0</td><td>82.4+1.7</td><td>87.2+0.3</td><td>68.7+4.4</td><td>91.3+1.4</td><td>41.9+4.6</td><td>63.8+2.7</td><td>62.6+3.1</td></tr><tr><td>MOON</td><td>75.9+14.6</td><td>67.9+8.9</td><td>17.9+0.0</td><td>33.1+0.2</td><td>68.7+5.5</td><td>82.7+2.0</td><td>87.1+0.2</td><td>65.8+1.5</td><td>90.2+0.3</td><td>37.7+0.4</td><td>62.2+1.1</td><td>62.7+3.2</td></tr><tr><td colspan="14">(c) ViT-B/32.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>62.0</td><td>62.1</td><td>19.1</td><td>45.4</td><td>60.2</td><td>80.4</td><td>87.3</td><td>66.6</td><td>91.4</td><td>42.7</td><td>63.5</td><td>61.9</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>65.9+3.9</td><td>63.3+1.2</td><td>21.9+2.8</td><td>51.3+5.9</td><td>69.3+9.1</td><td>82.2+1.8</td><td>90.3+3.0</td><td>74.1+7.5</td><td>92.6+1.2</td><td>47.4+4.7</td><td>66.1+2.6</td><td>65.9+4.0</td></tr><tr><td>MOON</td><td>79.1+17.1</td><td>76.0+13.9</td><td>22.7+3.6</td><td>52.1+6.7</td><td>74.4+14.2</td><td>89.1+8.7</td><td>90.3+3.0</td><td>74.0+7.4</td><td>94.2+2.8</td><td>47.0+4.3</td><td>69.2+5.7</td><td>69.8+7.9</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>67.0+5.0</td><td>65.0+2.9</td><td>20.2+1.1</td><td>51.1+5.7</td><td>68.5+8.3</td><td>82.7+2.3</td><td>88.5+1.2</td><td>73.7+7.1</td><td>92.5+1.1</td><td>49.5+6.8</td><td>66.9+3.4</td><td>66.0+4.1</td></tr><tr><td>MOON</td><td>79.3+17.3</td><td>75.1+13.0</td><td>20.2+1.1</td><td>52.1+6.7</td><td>71.6+11.4</td><td>86.5+6.1</td><td>87.9+0.6</td><td>71.8+5.2</td><td>93.7+2.3</td><td>43.8+1.1</td><td>67.9+4.4</td><td>68.2+6.3</td></tr><tr><td rowspan="2">Very High</td><td>StatA</td><td>66.6+4.6</td><td>66.0+3.9</td><td>18.8-0.3</td><td>51.0+5.6</td><td>65.1+4.9</td><td>81.5+1.1</td><td>88.0+0.7</td><td>70.6+4.0</td><td>91.9+0.5</td><td>49.5+6.8</td><td>66.5+3.0</td><td>65.1+3.2</td></tr><tr><td>MOON</td><td>76.9+14.9</td><td>72.3+10.2</td><td>18.4-0.7</td><td>52.1+6.7</td><td>66.9+6.7</td><td>82.6+2.2</td><td>87.5+0.2</td><td>68.6+2.0</td><td>91.6+0.2</td><td>43.0+0.3</td><td>65.3+1.8</td><td>65.9+4.0</td></tr><tr><td colspan="14">(d) ViT-L/14.</td></tr><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>73.5</td><td>67.7</td><td>32.5</td><td>60.3</td><td>76.9</td><td>90.9</td><td>93.5</td><td>79.5</td><td>95.2</td><td>53.5</td><td>74.9</td><td>72.6</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>76.2+2.7</td><td>69.4+1.7</td><td>39.1+6.6</td><td>71.0+10.7</td><td>81.9+5.0</td><td>91.7+0.8</td><td>94.8+1.3</td><td>81.9+2.4</td><td>95.6+0.4</td><td>56.9+3.4</td><td>77.6+2.7</td><td>76.0+3.4</td></tr><tr><td>MOON</td><td>85.0+11.5</td><td>79.2+11.5</td><td>38.9+6.4</td><td>67.4+7.1</td><td>84.0+7.1</td><td>95.9+5.0</td><td>95.4+1.9</td><td>83.3+3.8</td><td>97.7+2.5</td><td>59.4+5.9</td><td>80.2+5.3</td><td>78.8+6.2</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>77.2+3.7</td><td>70.9+3.2</td><td>36.8+4.3</td><td>71.2+10.9</td><td>82.0+5.1</td><td>92.3+1.4</td><td>94.3+0.8</td><td>81.9+2.4</td><td>95.3+0.1</td><td>58.7+5.2</td><td>78.8+3.9</td><td>76.3+3.7</td></tr><tr><td>MOON</td><td>85.3+11.8</td><td>78.3+10.6</td><td>35.3+2.8</td><td>67.2+6.9</td><td>83.6+6.7</td><td>94.6+3.7</td><td>94.2+0.7</td><td>82.5+3.0</td><td>96.9+1.7</td><td>55.9+2.4</td><td>79.0+4.1</td><td>77.5+5.0</td></tr><tr><td rowspan="2">Very High</td><td>StatA</td><td>77.3+3.8</td><td>71.6+3.9</td><td>33.7+1.2</td><td>71.2+10.9</td><td>79.5+2.6</td><td>91.7+0.8</td><td>94.1+0.6</td><td>80.7+1.2</td><td>94.9-0.3</td><td>59.0+5.5</td><td>78.7+3.8</td><td>75.7+3.1</td></tr><tr><td>MOON</td><td>84.5+11.0</td><td>76.0+8.3</td><td>32.6+0.1</td><td>67.2+6.9</td><td>80.9+4.0</td><td>92.4+1.5</td><td>94.0+0.5</td><td>80.3+0.8</td><td>95.1-0.1</td><td>54.8+1.3</td><td>76.6+1.7</td><td>75.9+3.3</td></tr></table>

Table 15. Results on four additional CLIP backbones, online adaptation with batch size of 128. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.  
(a) ResNet-50.

<table><tr><td>Scenario</td><td>Method</td><td><img src="images/d6b7deb59b0bebb60436a304667d0c4ebfaf8af897e437b82983bd3b984f9edb.jpg"/></td><td>SUN397</td><td>Aircert</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td><img src="images/b83f64e1e7a02e60964b0d99a901ba38b06856a47e5d53ae3d053b2d59b8765c.jpg"/></td><td><img src="images/90f1bdb8fba8fa0de104912e80d084122658950787503e27afa26f577a6b079f.jpg"/></td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>58.2</td><td>58.9</td><td>17.0</td><td>36.2</td><td>55.8</td><td>77.4</td><td>85.7</td><td>66.1</td><td>85.7</td><td>42.8</td><td>61.8</td><td>58.7</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>54.6-3.6</td><td>56.6-2.3</td><td>15.1-1.9</td><td>39.7+3.5</td><td>57.6+1.8</td><td>79.4+2.0</td><td>85.1-0.6</td><td>60.7-5.4</td><td>87.8+2.1</td><td>44.4+1.6</td><td>61.7-0.1</td><td>58.4-0.3</td></tr><tr><td>MOON</td><td>58.2±0.0</td><td>60.4+1.5</td><td>16.2-0.8</td><td>39.5+3.3</td><td>57.7+1.9</td><td>85.2+7.8</td><td>89.5+3.8</td><td>66.7+0.6</td><td>87.8+2.1</td><td>43.0+0.2</td><td>62.9+1.1</td><td>60.6+1.9</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>59.6+1.4</td><td>60.8+1.9</td><td>17.7+0.7</td><td>43.5+7.3</td><td>65.9+10.1</td><td>84.5+7.1</td><td>90.6+4.9</td><td>68.1+2.0</td><td>89.3+3.6</td><td>45.5+2.7</td><td>64.5+2.7</td><td>62.8+4.1</td></tr><tr><td>MOON</td><td>70.3+12.1</td><td>72.9+14.0</td><td>20.9+3.9</td><td>41.0+4.8</td><td>69.7+13.9</td><td>92.5+15.1</td><td>93.7+8.0</td><td>72.0+5.9</td><td>90.5+4.8</td><td>48.3+5.5</td><td>70.6+8.8</td><td>67.5+8.8</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>64.7+6.5</td><td>62.6+3.7</td><td>18.5+1.5</td><td>43.6+7.4</td><td>68.5+12.7</td><td>85.8+8.4</td><td>92.2+6.5</td><td>70.1+4.0</td><td>89.8+4.1</td><td>45.9+3.1</td><td>65.2+3.4</td><td>64.3+5.6</td></tr><tr><td>MOON</td><td>78.3+20.1</td><td>77.1+18.2</td><td>22.2+5.2</td><td>41.3+5.1</td><td>72.9+17.1</td><td>93.4+16.0</td><td>94.2+8.5</td><td>73.6+7.5</td><td>91.4+5.7</td><td>49.7+6.9</td><td>72.8+11.0</td><td>69.7+11.0</td></tr><tr><td rowspan="2">Separate</td><td>StatA</td><td>66.6+8.4</td><td>62.6+3.7</td><td>19.8+2.8</td><td>44.3+8.1</td><td>69.5+13.7</td><td>85.6+8.2</td><td>93.8+8.1</td><td>71.9+5.8</td><td>90.2+4.5</td><td>46.0+3.2</td><td>65.3+3.5</td><td>65.1+6.4</td></tr><tr><td>MOON</td><td>79.4+21.2</td><td>76.7+17.8</td><td>24.7+7.7</td><td>40.3+4.1</td><td>73.3+17.5</td><td>92.8+15.4</td><td>94.1+8.4</td><td>74.9+8.8</td><td>92.3+6.6</td><td>53.0+10.2</td><td>73.8+12.0</td><td>70.5+11.8</td></tr><tr><td colspan="14">(b) ResNet-101.</td></tr><tr><td>Scenario</td><td>Method</td><td><img src="images/6e2e6d9b7e5d5905e09c26a196e56fbb6cedb3b7a2c14955f99b675b791703b8.jpg"/></td><td><img src="images/28cb4e4a4cb3b232aabf0ef5b4737f64afa98c558b78ff3d70059df4611354ec.jpg"/></td><td><img src="images/2b6a5b0fc997afcc1514323f188dc3ff98c316024934e14a10687e058f24fff1.jpg"/></td><td><img src="images/64f6c28278157be00d3c509277bea9f4b87834507723d5c5c091c6c62dd08d0d.jpg"/></td><td><img src="images/aa4c7a837bc2e33c9907893b0989c43dc2b89925c052d0fd1f4fd7bf696fedb5.jpg"/></td><td><img src="images/01e00b9646316fac2eafd27c2f58a36dd713ae6d2d20feb8ea46cb38e6c619bf.jpg"/></td><td>Pets</td><td><img src="images/769bb2150c524cbc7ab607614224b77493f86466d134a93ef4c1d2f1baab020c.jpg"/></td><td><img src="images/139d9c709f1cf44277a9267d5a73c04cd6a4117b686bf4e252577f4778ebd486.jpg"/></td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>61.3</td><td>59.0</td><td>17.9</td><td>32.9</td><td>63.2</td><td>80.7</td><td>86.9</td><td>64.3</td><td>89.9</td><td>37.3</td><td>61.1</td><td>59.5</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>60.5-0.8</td><td>59.3+0.3</td><td>16.9-1.0</td><td>32.7-0.2</td><td>65.5+2.3</td><td>84.9+4.2</td><td>91.0+4.1</td><td>67.8+3.5</td><td>92.2+2.3</td><td>41.1+3.8</td><td>62.8+1.7</td><td>61.3+1.8</td></tr><tr><td>MOON</td><td>61.8+0.5</td><td>60.3+1.3</td><td>17.6-0.3</td><td>32.4-0.5</td><td>65.9+2.7</td><td>87.5+6.8</td><td>89.9+3.0</td><td>65.7+1.4</td><td>91.8+1.9</td><td>37.7+0.4</td><td>61.9+0.8</td><td>61.1+1.6</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>66.1+4.8</td><td>64.2+5.2</td><td>19.7+1.8</td><td>33.3+0.4</td><td>72.2+9.0</td><td>88.1+7.4</td><td>94.1+7.2</td><td>72.1+7.8</td><td>93.2+3.3</td><td>42.9+5.6</td><td>65.2+4.1</td><td>64.6+5.1</td></tr><tr><td>MOON</td><td>71.9+10.6</td><td>70.5+11.5</td><td>21.0+3.1</td><td>33.0+0.1</td><td>74.1+10.9</td><td>92.6+11.9</td><td>92.4+5.5</td><td>69.6+5.3</td><td>93.9+4.0</td><td>41.4+4.1</td><td>67.8+6.7</td><td>66.2+6.7</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>70.5+9.2</td><td>65.9+6.9</td><td>20.6+2.7</td><td>33.5+0.6</td><td>74.1+10.9</td><td>88.7+8.0</td><td>94.4+7.5</td><td>73.1+8.8</td><td>93.4+3.5</td><td>43.0+5.7</td><td>65.7+4.6</td><td>65.7+6.2</td></tr><tr><td>MOON</td><td>78.2+16.9</td><td>74.1+15.1</td><td>21.8+3.9</td><td>33.4+0.5</td><td>76.3+13.1</td><td>93.2+12.5</td><td>92.8+5.9</td><td>70.8+6.5</td><td>94.6+4.7</td><td>42.5+5.2</td><td>69.5+8.4</td><td>67.9+8.4</td></tr><tr><td rowspan="2">Separate</td><td>StatA</td><td>71.4+10.1</td><td>65.7+6.7</td><td>22.1+4.2</td><td>32.2-0.7</td><td>74.9+11.7</td><td>88.5+7.8</td><td>94.2+7.3</td><td>73.9+9.6</td><td>93.4+3.5</td><td>41.9+4.6</td><td>65.7+4.6</td><td>65.8+6.3</td></tr><tr><td>MOON</td><td>78.8+17.5</td><td>74.0+15.0</td><td>23.8+5.9</td><td>33.8+0.9</td><td>76.8+13.6</td><td>92.7+12.0</td><td>92.7+5.8</td><td>72.1+7.8</td><td>95.2+5.3</td><td>44.1+6.8</td><td>70.3+9.2</td><td>68.6+9.1</td></tr><tr><td colspan="14">(c) ViT-B/32.</td></tr><tr><td>Scenario</td><td>Method</td><td><img src="images/6ffe18604aeee6819a89531fabbee697f91b7e8414d77d419991e186f3fdf663.jpg"/></td><td><img src="images/1425faeaefbff6737ebea052b14705276e8e2bbd83826b417f818f0c4ac6a5db.jpg"/></td><td><img src="images/1a8cc70f50d755874461c3009a111332ee8da05c999e7903fef7914be8d4dd99.jpg"/></td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td><img src="images/f7fdb04d23fbdaeab67f5bd0c7a9c6e1d241196455a83233a94b2aa7f8fafa46.jpg"/></td><td><img src="images/f1089e797c16147316f2c188a1faa4b21e7d0980beddf1e811d5f6683e53c96d.jpg"/></td><td>[XDA0]</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>62.0</td><td>62.1</td><td>19.1</td><td>45.4</td><td>60.2</td><td>80.4</td><td>87.3</td><td>66.6</td><td>91.4</td><td>42.7</td><td>63.5</td><td>61.9</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>61.4-0.6</td><td>62.7+0.6</td><td>19.2+0.1</td><td>51.0+5.6</td><td>61.8+1.6</td><td>82.6+2.2</td><td>91.0+3.7</td><td>69.0+2.4</td><td>92.9+1.5</td><td>46.4+3.7</td><td>64.4+0.9</td><td>63.9+2.0</td></tr><tr><td>MOON</td><td>62.4+0.4</td><td>63.6+1.5</td><td>18.2-0.9</td><td>49.9+4.5</td><td>62.2+2.0</td><td>87.4+7.0</td><td>90.3+3.0</td><td>67.8+1.2</td><td>92.6+1.2</td><td>42.3-0.4</td><td>64.1+0.6</td><td>63.7+1.8</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>64.6+2.6</td><td>64.8+2.7</td><td>21.4+2.3</td><td>49.9+4.5</td><td>68.1+7.9</td><td>84.4+4.0</td><td>92.8+5.5</td><td>72.5+5.9</td><td>93.5+2.1</td><td>46.4+3.7</td><td>65.5+2.0</td><td>65.8+3.9</td></tr><tr><td>MOON</td><td>72.9+10.9</td><td>74.3+12.2</td><td>22.2+3.1</td><td>50.7+5.3</td><td>73.7+13.5</td><td>93.1+12.7</td><td>93.1+5.8</td><td>72.2+5.6</td><td>94.2+2.8</td><td>47.4+4.7</td><td>69.2+5.7</td><td>69.4+7.5</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>66.9+4.9</td><td>64.9+2.8</td><td>22.0+2.9</td><td>50.1+4.7</td><td>69.9+9.7</td><td>84.6+4.2</td><td>93.2+5.9</td><td>73.5+6.9</td><td>93.7+2.3</td><td>46.3+3.6</td><td>65.6+2.1</td><td>66.4+4.6</td></tr><tr><td>MOON</td><td>79.7+17.7</td><td>77.7+15.6</td><td>23.4+4.3</td><td>51.1+5.7</td><td>76.6+16.4</td><td>93.8+13.4</td><td>93.6+6.3</td><td>73.6+7.0</td><td>94.7+3.3</td><td>48.4+5.7</td><td>70.4+6.9</td><td>71.2+9.3</td></tr><tr><td rowspan="2">Separate</td><td>StatA</td><td>67.0+5.0</td><td>63.8+1.7</td><td>22.9+3.8</td><td>44.9-0.5</td><td>70.4+10.2</td><td>84.1+3.7</td><td>92.8+5.5</td><td>74.6+8.0</td><td>94.0+2.6</td><td>45.1+2.4</td><td>65.0+1.5</td><td>65.9+4.0</td></tr><tr><td>MOON</td><td>80.3+18.3</td><td>77.3+15.2</td><td>25.4+6.3</td><td>49.1+3.7</td><td>77.0+16.8</td><td>93.1+12.7</td><td>93.4+6.1</td><td>75.0+8.4</td><td>95.1+3.7</td><td>50.6+7.9</td><td>70.4+6.9</td><td>71.5+9.6</td></tr><tr><td colspan="14">(d) ViT-L/14.</td></tr><tr><td>Scenario</td><td>Method</td><td><img src="images/879d6018cd08b166781365153caf5921b65f2d971dd732d71031af73be7f298a.jpg"/></td><td><img src="images/8652ca5223c6c06108baa49597ba10788f5bf7acba12dde83cf96497c3704c63.jpg"/></td><td><img src="images/f9df416f71494101d4a4fb715864397e212fd855a3cc7c7ff8cddb85ae710860.jpg"/></td><td><img src="images/d390ca2854e05f90996c70090c19d0636744b530025fbb1e0da5becacb15a0a4.jpg"/></td><td><img src="images/976e02064f6f65e609f5680f7279fb0a32034397cd5ac1eb8a4631545b6c2f52.jpg"/></td><td><img src="images/dd8d4ddb04c7fc9902e90b6be5531526eeca020d80dc9eafb9c8f84a2a4e7ede.jpg"/></td><td>[WCT2]</td><td><img src="images/8d827a48bd41a323b3a174338020dcbf142ec01987ab3adbd254877ebdd6d0b5.jpg"/></td><td><img src="images/bd5ceeef8a6f1ca7681fcda27df6966c77f3033cbcc877f4df1002500cab1b53.jpg"/></td><td>[AKST]</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>73.5</td><td>67.7</td><td>32.5</td><td>60.3</td><td>76.9</td><td>90.9</td><td>93.5</td><td>79.5</td><td>95.2</td><td>53.5</td><td>74.9</td><td>72.6</td></tr><tr><td rowspan="2">Low</td><td>StatA</td><td>73.3-0.2</td><td>68.2+0.5</td><td>34.1+1.6</td><td>68.8+8.5</td><td>77.7+0.8</td><td>92.0+1.1</td><td>95.0+1.5</td><td>80.2+0.7</td><td>95.6+0.4</td><td>55.4+1.9</td><td>76.9+2.0</td><td>74.3+1.7</td></tr><tr><td>MOON</td><td>73.8+0.3</td><td>68.3+0.6</td><td>32.2-0.3</td><td>67.8+7.5</td><td>77.7+0.8</td><td>94.5+3.6</td><td>95.2+1.7</td><td>79.6+0.1</td><td>96.0+0.8</td><td>54.0+0.5</td><td>75.9+1.0</td><td>74.1+1.5</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>75.8+2.3</td><td>70.6+2.9</td><td>38.3+5.8</td><td>69.8+9.6</td><td>81.9+5.0</td><td>93.2+2.3</td><td>96.3+2.8</td><td>81.5+2.0</td><td>95.8+0.6</td><td>55.6+2.1</td><td>77.6+2.7</td><td>76.0+3.4</td></tr><tr><td>MOON</td><td>81.3+7.8</td><td>77.3+9.6</td><td>37.9+5.4</td><td>69.0+8.7</td><td>83.9+7.0</td><td>97.2+6.3</td><td>96.8+3.3</td><td>82.0+2.5</td><td>97.1+1.9</td><td>58.8+5.3</td><td>79.6+4.7</td><td>78.3+5.7</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>77.6+4.1</td><td>71.1+3.4</td><td>39.6+7.1</td><td>68.9+8.6</td><td>82.9+6.0</td><td>93.5+2.6</td><td>96.5+3.0</td><td>81.9+2.4</td><td>95.7+0.5</td><td>55.5+2.0</td><td>77.5+2.6</td><td>76.4+3.8</td></tr><tr><td>MOON</td><td>85.6+12.1</td><td>80.3+12.6</td><td>39.7+7.2</td><td>69.4+9.1</td><td>85.2+8.3</td><td>97.5+6.6</td><td>97.0+3.5</td><td>82.6+3.1</td><td>97.4+2.2</td><td>60.3+6.8</td><td>80.4+5.5</td><td>79.6+7.0</td></tr><tr><td rowspan="2">Separate</td><td>StatA</td><td>77.6+4.1</td><td>70.5+2.8</td><td>41.3+8.8</td><td>66.3+6.0</td><td>83.2+6.3</td><td>93.5+2.6</td><td>96.3+2.8</td><td>82.0+2.5</td><td>95.8+0.6</td><td>54.5+1.0</td><td>76.8+1.9</td><td>76.1+3.6</td></tr><tr><td>MOON</td><td>85.7+12.2</td><td>80.4+12.7</td><td>41.4+8.9</td><td>68.1+7.8</td><td>85.2+8.3</td><td>97.4+6.5</td><td>96.8+3.3</td><td>83.4+3.9</td><td>97.7+2.5</td><td>62.6+9.1</td><td>80.5+5.6</td><td>79.9+7.3</td></tr></table>

Table 16. Results on OpenCLIP (151M), batch adaptation with batch size of 1,000. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td><img src="images/a6727ea73d1e6caede63a2844792181a3ffb08a2eb6eee8f3fa03ced7448807a.jpg"/></td><td><img src="images/1ddc7e3775330a38e5a5e9e0ecb4a25cbb251835ad7a660f0d2d53f8c4395f8c.jpg"/></td><td><img src="images/257ee60678bc30f959a53da9c0f443d23b5c994346fb30f679af0e48e17ba00d.jpg"/></td><td><img src="images/cb0e0746c624f074e3ddde07d6e57e46134ce8025855d45bf3c1d045352ddde0.jpg"/></td><td><img src="images/acef0e1a41ce8b235fd19dd0eafb9dbf0e555629a45d70d322ace0d66062e088.jpg"/></td><td><img src="images/aa6b94fcb510f14f4c9d2c0f2aa5cccba567acf9b775449452a886180ea5acc1.jpg"/></td><td><img src="images/7bb6a7f8ce36408d43c6c5574b441fa1b5fa569db8912d34a6af3846bfc939a8.jpg"/></td><td><img src="images/61874a4c1118f8bec2eb836d33861c2287393478e07944af00388b3e55a88e78.jpg"/></td><td><img src="images/81e9318a7cb0c6dcd054180b9f54449cbd50d3222551d446a3a12ba4badcc167.jpg"/></td><td>DTD</td><td><img src="images/2505fc9faf2002bd789978d6f626b9e655e70050f63af14004ae1e0d26e2edc3.jpg"/></td><td>Avg.</td></tr><tr><td></td><td>OpenCLIP</td><td>73.0</td><td>69.9</td><td>29.7</td><td>56.4</td><td>89.9</td><td>87.5</td><td>92.8</td><td>75.4</td><td>96.7</td><td>58.3</td><td>67.5</td><td>72.5</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td> $73.4_{+0.4}$ </td><td> $68.8_{-1.1}$ </td><td> $32.6_{+2.8}$ </td><td> $62.3_{+5.9}$ </td><td> $91.2_{+1.3}$ </td><td> $87.4_{-0.2}$ </td><td> $93.9_{+1.0}$ </td><td> $79.7_{+4.2}$ </td><td> $96.4_{-0.3}$ </td><td> $59.7_{+1.4}$ </td><td> $70.2_{+2.7}$ </td><td> $74.1_{+1.7}$ </td></tr><tr><td>MOON</td><td> $83.5_{+10.5}$ </td><td> $80.0_{+10.1}$ </td><td> $32.6_{+2.9}$ </td><td> $61.5_{+5.1}$ </td><td> $93.9_{+4.0}$ </td><td> $93.3_{+5.8}$ </td><td> $95.3_{+2.5}$ </td><td> $78.4_{+3.0}$ </td><td> $98.1_{+1.4}$ </td><td> $65.1_{+6.8}$ </td><td> $74.0_{+6.6}$ </td><td> $77.8_{+5.3}$ </td></tr><tr><td rowspan="2">High</td><td>StatA</td><td> $74.1_{+1.1}$ </td><td> $70.1_{+0.3}$ </td><td> $32.1_{+2.4}$ </td><td> $62.4_{+6.0}$ </td><td> $91.4_{+1.5}$ </td><td> $88.2_{+0.7}$ </td><td> $93.7_{+0.9}$ </td><td> $79.7_{+4.3}$ </td><td> $96.7_{+0.1}$ </td><td> $62.1_{+3.7}$ </td><td> $71.1_{+3.6}$ </td><td> $74.7_{+2.2}$ </td></tr><tr><td>MOON</td><td> $83.7_{+10.7}$ </td><td> $80.2_{+10.4}$ </td><td> $31.0_{+1.2}$ </td><td> $61.5_{+5.1}$ </td><td> $93.9_{+4.0}$ </td><td> $91.7_{+4.2}$ </td><td> $93.7_{+0.8}$ </td><td> $77.5_{+2.0}$ </td><td> $97.9_{+1.2}$ </td><td> $61.3_{+2.9}$ </td><td> $72.3_{+4.8}$ </td><td> $76.8_{+4.3}$ </td></tr><tr><td rowspan="2">Very High</td><td>StatA</td><td> $74.6_{+1.6}$ </td><td> $71.3_{+1.4}$ </td><td> $30.9_{+1.1}$ </td><td> $62.4_{+6.0}$ </td><td> $91.2_{+1.3}$ </td><td> $88.2_{+0.7}$ </td><td> $93.6_{+0.8}$ </td><td> $77.8_{+2.3}$ </td><td> $96.7_{\pm 0.0}$ </td><td> $62.6_{+4.3}$ </td><td> $71.4_{+3.9}$ </td><td> $74.6_{+2.1}$ </td></tr><tr><td>MOON</td><td> $83.1_{+10.1}$ </td><td> $79.1_{+9.2}$ </td><td> $28.8_{-0.9}$ </td><td> $61.4_{+5.0}$ </td><td> $93.1_{+3.2}$ </td><td> $89.3_{+1.7}$ </td><td> $93.4_{+0.6}$ </td><td> $75.6_{+0.1}$ </td><td> $96.6_{-0.1}$ </td><td> $60.5_{+2.1}$ </td><td> $70.4_{+2.9}$ </td><td> $75.6_{+3.1}$ </td></tr></table>

Table 17. Results on SigLIP (878M), batch adaptation with batch size of 1,000. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td><img src="images/7ec48b57fcc4c3f52eb4b0f5c0271bd2cf81df2bd200f1316de3aa6c8b803b25.jpg"/></td><td><img src="images/1f4661d5dd1eecb306d47b9165a2fa1034d41d57aee46f7aa87bd81182eb013d.jpg"/></td><td>Aircraft</td><td><img src="images/ddf104892b37543e20e5cce0b8f53d68b725f14a733203552cfdbe449670f215.jpg"/></td><td><img src="images/28477c17b18049f219270fa8fc2fc3cceb57f28ed48932b455d15d5b936b3c9d.jpg"/></td><td><img src="images/998183482c50f6a57892da1e7606f50c334431479abc88680782392498d42b08.jpg"/></td><td>Pets</td><td><img src="images/a2f463fb6bb370c2385a9f370ab9588ca9f5d40e65f154bebf6cef11a65a0da5.jpg"/></td><td><img src="images/07bf0402968c0beef1555da0996460ac837795408e06980397ed719a3073bcc3.jpg"/></td><td>[70TW]</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>SigLIP</td><td>82.3</td><td>75.4</td><td>60.2</td><td>57.1</td><td>94.7</td><td>94.7</td><td>96.5</td><td>92.7</td><td>98.2</td><td>64.8</td><td>83.7</td><td>81.8</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td> $82.7_{+0.5}$ </td><td> $76.1_{+0.7}$ </td><td> $64.7_{+4.5}$ </td><td> $60.4_{+3.3}$ </td><td> $95.4_{+0.7}$ </td><td> $94.0_{-0.7}$ </td><td> $95.4_{-1.1}$ </td><td> $90.4_{-2.3}$ </td><td> $97.6_{-0.6}$ </td><td> $68.0_{+3.2}$ </td><td> $86.0_{+2.2}$ </td><td> $82.8_{+1.0}$ </td></tr><tr><td>MOON</td><td> $91.4_{+9.1}$ </td><td> $86.0_{+10.6}$ </td><td> $65.8_{+5.6}$ </td><td> $59.4_{+2.3}$ </td><td> $96.8_{+2.1}$ </td><td> $98.1_{+3.4}$ </td><td> $98.3_{+1.8}$ </td><td> $93.5_{+0.7}$ </td><td> $98.8_{+0.6}$ </td><td> $71.9_{+7.1}$ </td><td> $87.0_{+3.3}$ </td><td> $\mathbf{86.1}_{+4.2}$ </td></tr><tr><td rowspan="2">High</td><td>StatA</td><td> $82.8_{+0.6}$ </td><td> $77.3_{+1.9}$ </td><td> $64.1_{+3.9}$ </td><td> $60.8_{+3.7}$ </td><td> $95.5_{+0.8}$ </td><td> $94.8_{+0.1}$ </td><td> $95.0_{-1.5}$ </td><td> $90.5_{-2.2}$ </td><td> $97.8_{-0.4}$ </td><td> $68.6_{+3.8}$ </td><td> $85.6_{+1.9}$ </td><td> $83.0_{+1.1}$ </td></tr><tr><td>MOON</td><td> $91.5_{+9.2}$ </td><td> $86.6_{+11.2}$ </td><td> $64.0_{+3.8}$ </td><td> $60.1_{+3.0}$ </td><td> $96.6_{+2.0}$ </td><td> $97.1_{+2.4}$ </td><td> $97.2_{+0.7}$ </td><td> $93.5_{+0.7}$ </td><td> $98.7_{+0.5}$ </td><td> $66.6_{+1.7}$ </td><td> $85.3_{+1.6}$ </td><td> $\mathbf{85.2}_{+3.3}$ </td></tr><tr><td rowspan="2">Very High</td><td>StatA</td><td> $82.8_{+0.5}$ </td><td> $77.6_{+2.2}$ </td><td> $60.4_{+0.2}$ </td><td> $60.8_{+3.7}$ </td><td> $95.4_{+0.7}$ </td><td> $94.8_{+0.1}$ </td><td> $94.9_{-1.6}$ </td><td> $90.4_{-2.3}$ </td><td> $97.7_{-0.5}$ </td><td> $69.0_{+4.1}$ </td><td> $84.5_{+0.7}$ </td><td> $82.6_{+0.7}$ </td></tr><tr><td>MOON</td><td> $91.0_{+8.8}$ </td><td> $85.2_{+9.8}$ </td><td> $58.8_{-1.4}$ </td><td> $60.1_{+3.0}$ </td><td> $96.2_{+1.5}$ </td><td> $95.6_{+0.9}$ </td><td> $97.0_{+0.5}$ </td><td> $93.4_{+0.6}$ </td><td> $97.6_{-0.6}$ </td><td> $65.1_{+0.3}$ </td><td> $82.7_{-1.1}$ </td><td> $\mathbf{83.9}_{+2.0}$ </td></tr></table>

Table 18. Results on EVA-CLIP (1.1B), batch adaptation with batch size of 1,000. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td><img src="images/5b748ae7e43912b2145557b1b15eb001827c85b21079035faee1a64a5627f4ca.jpg"/></td><td>SUN397</td><td><img src="images/9ed0a08205d67a8c2aa17659e1c9ba66110b411c37b1ecf52f82064c5aa42b2f.jpg"/></td><td><img src="images/0579e316d82116429f0db3caf2ea431c48cef24292d1831f45c6327383b97ee6.jpg"/></td><td><img src="images/0e3e07b13dcc6060a55abe4b273212c639ccc5e186a5278a765e82eb8bbf2ff0.jpg"/></td><td><img src="images/dbaad9304dcdbc898b9d7e0f871f62a5e1e29eecaec162d3b92d57791299b7a2.jpg"/></td><td>Pets</td><td><img src="images/64db749e30883632968f2f5a05db091cd0cbf29013a40ed605ae065f26a9dd9b.jpg"/></td><td><img src="images/536dc47227b8513efa9138291291a9b4c735d92521d641ba3baacb4cbefc53d1.jpg"/></td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td></td><td>EVA-CLIP</td><td>78.0</td><td>72.9</td><td>33.6</td><td>70.1</td><td>91.2</td><td>91.0</td><td>94.2</td><td>75.0</td><td>97.3</td><td>60.6</td><td>78.0</td><td>76.5</td></tr><tr><td rowspan="2">Medium</td><td>StatA</td><td>77.8-0.2</td><td>71.5-1.5</td><td>36.9+3.3</td><td>72.7+2.6</td><td>91.8+0.6</td><td>90.8-0.2</td><td>93.9-0.3</td><td>77.8+2.7</td><td>97.1-0.2</td><td>62.4+1.8</td><td>79.7+1.7</td><td>77.5+1.0</td></tr><tr><td>MOON</td><td>81.1+3.1</td><td>75.6+2.7</td><td>36.5+2.8</td><td>72.7+2.5</td><td>93.1+1.8</td><td>93.2+2.2</td><td>94.8+0.6</td><td>76.4+1.4</td><td>97.7+0.4</td><td>64.8+4.2</td><td>80.4+2.5</td><td>78.7+2.2</td></tr><tr><td rowspan="2">High</td><td>StatA</td><td>78.3+0.4</td><td>73.0+0.1</td><td>36.2+2.6</td><td>73.2+3.1</td><td>92.2+1.0</td><td>91.6+0.6</td><td>94.2±0.0</td><td>78.4+3.4</td><td>97.4+0.1</td><td>64.8+4.2</td><td>80.3+2.4</td><td>78.2+1.6</td></tr><tr><td>MOON</td><td>81.3+3.3</td><td>76.2+3.3</td><td>35.0+1.4</td><td>72.7+2.6</td><td>93.1+1.9</td><td>92.7+1.7</td><td>94.2±0.0</td><td>76.7+1.7</td><td>97.8+0.5</td><td>63.5+2.9</td><td>80.1+2.1</td><td>78.5+2.0</td></tr><tr><td rowspan="2">Very High</td><td>StatA</td><td>78.7+0.8</td><td>73.8+0.8</td><td>35.4+1.8</td><td>73.2+3.1</td><td>92.3+1.0</td><td>91.6+0.6</td><td>94.2±0.0</td><td>77.7+2.7</td><td>97.4+0.1</td><td>65.5+5.0</td><td>80.7+2.7</td><td>78.2+1.7</td></tr><tr><td>MOON</td><td>81.2+3.2</td><td>76.1+3.2</td><td>33.6-0.1</td><td>72.7+2.6</td><td>92.9+1.6</td><td>91.8+0.8</td><td>94.1-0.1</td><td>76.0+0.9</td><td>97.5+0.2</td><td>63.1+2.5</td><td>79.5+1.5</td><td>78.0+1.5</td></tr></table>

Table 19. Results on full dataset with all classes. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.

<table><tr><td> $K_{\text{eff}}$ </td><td>Method</td><td><img src="images/c345b8352d9270fe6ff6d2ce4d93f9b833b6feff4b83f1195aff7c246cd5460d.jpg"/></td><td>SUN397</td><td><img src="images/889e50218103be45e0bfc1632facc865af63b133d80da1bdcc092965bca4d0a0.jpg"/></td><td><img src="images/777fa766821fb54a47701977e08673c2fb41c9ba8940b793275ffc28970731cb.jpg"/></td><td>StanfordCars</td><td><img src="images/a71987aaf33d45a28e760da5efdccb9113dde1ac4932302f806d4a40d2fee955.jpg"/></td><td>Pets</td><td><img src="images/49ce93f6789f3633e080c077f5f9cd1acd15af18610dc4e598052067967d0968.jpg"/></td><td><img src="images/3f18780ad9eb96ffb9eaa68b96c8fcf68b39ae7bc969b63bca590a292fd898a3.jpg"/></td><td>DTD</td><td><img src="images/b19e4cb19e7026ed602fd490220d8df79651b0cd2339c693077ddd40298063c8.jpg"/></td><td>Avg.</td></tr><tr><td></td><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>All</td><td>StatA</td><td> $69.9_{+3.3}$ </td><td> $68.7_{+6.2}$ </td><td> $24.7_{\pm 0.0}$ </td><td> $67.3_{+19.0}$ </td><td> $68.0_{+2.4}$ </td><td> $87.1_{+1.2}$ </td><td> $92.4_{+3.3}$ </td><td> $75.2_{+4.5}$ </td><td> $94.2_{+1.0}$ </td><td> $48.4_{+4.9}$ </td><td> $73.5_{+6.0}$ </td><td> $69.9_{+4.7}$ </td></tr><tr><td></td><td>MOON</td><td> $68.7_{+2.1}$ </td><td> $65.4_{+2.9}$ </td><td> $24.4_{-0.3}$ </td><td> $59.1_{+10.8}$ </td><td> $67.2_{+1.6}$ </td><td> $86.7_{+0.8}$ </td><td> $90.2_{+1.1}$ </td><td> $72.8_{+2.1}$ </td><td> $93.4_{+0.2}$ </td><td> $45.0_{+1.5}$ </td><td> $70.9_{+3.4}$ </td><td> $67.6_{+2.4}$ </td></tr></table>

Table 20. Results on random class scenarios, where K<sub>ef</sub> is randomly sampled within [1, min{N, K}]. Subscript green indicates improvement, red indicates decline, and gray indicates no change compared with zero-shot performance.

<table><tr><td colspan="13">(a) Batch Size: 64.</td></tr><tr><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85.9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>StatA</td><td> $68.7_{+2.1}$ </td><td> $64.8_{+2.3}$ </td><td> $24.1_{-0.6}$ </td><td> $50.6_{+2.3}$ </td><td> $68.9_{+3.3}$ </td><td> $87.1_{+1.2}$ </td><td> $90.8_{+1.7}$ </td><td> $72.1_{+1.4}$ </td><td> $93.8_{+0.6}$ </td><td> $46.6_{+3.1}$ </td><td> $68.7_{+1.2}$ </td><td> $66.9_{+1.7}$ </td></tr><tr><td>MOON</td><td> $75.0_{+8.4}$ </td><td> $69.0_{+6.5}$ </td><td> $22.5_{-2.2}$ </td><td> $47.8_{-0.5}$ </td><td> $69.6_{+4.0}$ </td><td> $89.2_{+3.3}$ </td><td> $90.0_{+0.9}$ </td><td> $71.7_{+1.0}$ </td><td> $93.5_{+0.3}$ </td><td> $42.5_{-1.0}$ </td><td> $68.8_{+1.3}$ </td><td> $67.2_{+2.0}$ </td></tr><tr><td colspan="13">(b) Batch Size: 128.</td></tr><tr><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85,9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>StatA</td><td> $68.6_{+2.0}$ </td><td> $64.3_{+1.8}$ </td><td> $23.6_{-1.1}$ </td><td> $53.3_{+5.0}$ </td><td> $67.5_{+1.9}$ </td><td> $86.9_{+1.0}$ </td><td> $91.1_{+2.0}$ </td><td> $71.5_{+0.8}$ </td><td> $93.8_{+0.6}$ </td><td> $46.9_{+3.4}$ </td><td> $67.9_{+0.4}$ </td><td> $66.8_{+1.6}$ </td></tr><tr><td>MOON</td><td> $74.1_{+7.5}$ </td><td> $67.3_{+4.8}$ </td><td> $22.0_{-2.7}$ </td><td> $50.6_{+2.3}$ </td><td> $66.4_{+0.8}$ </td><td> $87.9_{+2.0}$ </td><td> $90.7_{+1.6}$ </td><td> $70.2_{-0.5}$ </td><td> $93.2_{\pm 0.0}$ </td><td> $44.1_{+0.6}$ </td><td> $66.4_{-1.1}$ </td><td> $66.6_{+1.4}$ </td></tr><tr><td colspan="13">(c) Batch Size: 256.</td></tr><tr><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>859</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>StatA</td><td> $68.2_{+1.6}$ </td><td> $64.1_{+1.6}$ </td><td> $24.1_{-0.6}$ </td><td> $55.3_{+7.0}$ </td><td> $66.3_{+0.7}$ </td><td> $87.4_{+1.5}$ </td><td> $91.9_{+2.8}$ </td><td> $73.2_{+2.5}$ </td><td> $93.8_{+0.6}$ </td><td> $47.4_{+3.9}$ </td><td> $68.7_{+1.2}$ </td><td> $67.3_{+2.1}$ </td></tr><tr><td>MOON</td><td> $72.9_{+6.3}$ </td><td> $64.7_{+2.2}$ </td><td> $23.3_{-1.4}$ </td><td> $51.9_{+3.6}$ </td><td> $64.8_{-0.8}$ </td><td> $89.0_{+3.1}$ </td><td> $91.7_{+2.6}$ </td><td> $72.2_{+1.5}$ </td><td> $93.5_{+0.3}$ </td><td> $46.5_{+3.0}$ </td><td> $67.9_{+0.4}$ </td><td> $67.1_{+1.9}$ </td></tr><tr><td colspan="13">(d) Batch Size: 500.</td></tr><tr><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>85 9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>StatA</td><td> $68.1_{+1.5}$ </td><td> $64.2_{+1.7}$ </td><td> $24.9_{+0.2}$ </td><td> $54.5_{+6.2}$ </td><td> $67.2_{+1.6}$ </td><td> $87.5_{+1.6}$ </td><td> $92.5_{+3.4}$ </td><td> $74.2_{+3.5}$ </td><td> $93.5_{+0.3}$ </td><td> $47.1_{+3.6}$ </td><td> $69.4_{+1.9}$ </td><td> $67.6_{+2.3}$ </td></tr><tr><td>MOON</td><td> $70.9_{+4.3}$ </td><td> $62.4_{-0.1}$ </td><td> $24.7_{\pm 0.0}$ </td><td> $52.7_{+4.4}$ </td><td> $66.5_{+0.9}$ </td><td> $89.8_{+3.9}$ </td><td> $91.8_{+2.7}$ </td><td> $73.1_{+2.4}$ </td><td> $93.7_{+0.5}$ </td><td> $47.6_{+4.1}$ </td><td> $69.1_{+1.6}$ </td><td> $67.5_{+2.3}$ </td></tr><tr><td colspan="13">(e) Batch Size: 1,000.</td></tr><tr><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td>B5 9</td><td>89.1</td><td>70.7</td><td>93.2</td><td>43.5</td><td>67.5</td><td>65.2</td></tr><tr><td>StatA</td><td> $67.5_{+0.9}$ </td><td> $65.2_{+2.7}$ </td><td> $25.4_{+0.7}$ </td><td> $55.0_{+6.7}$ </td><td> $68.0_{+2.4}$ </td><td> $87.2_{+1.3}$ </td><td> $93.0_{+3.9}$ </td><td> $75.3_{+4.6}$ </td><td> $93.3_{+0.1}$ </td><td> $47.8_{+4.3}$ </td><td> $70.7_{+3.2}$ </td><td> $68.0_{+2.8}$ </td></tr><tr><td>MOON</td><td> $67.9_{+1.3}$ </td><td> $64.0_{+1.5}$ </td><td> $25.4_{+0.7}$ </td><td> $54.7_{+6.4}$ </td><td> $68.1_{+2.5}$ </td><td> $89.7_{+3.8}$ </td><td> $92.0_{+2.9}$ </td><td> $74.0_{+3.3}$ </td><td> $94.1_{+0.9}$ </td><td> $48.1_{+4.6}$ </td><td> $70.7_{+3.2}$ </td><td> $68.1_{+2.8}$ </td></tr><tr><td colspan="13">(f) Batch Size: 2000.</td></tr><tr><td>Method</td><td>ImageNet</td><td>SUN397</td><td>Aircraft</td><td>EuroSAT</td><td>StanfordCars</td><td>Food101</td><td>Pets</td><td>Flowers102</td><td>Caltech101</td><td>DTD</td><td>UCF101</td><td>Avg.</td></tr><tr><td>CLIP</td><td>66.6</td><td>62.5</td><td>24.7</td><td>48.3</td><td>65.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>StatA</td><td> $68.1_{+1.5}$ </td><td> $66.1_{+3.6}$ </td><td> $26.3_{+1.6}$ </td><td> $56.7_{+8.4}$ </td><td> $69.6_{+4.0}$ </td><td> $86.7_{+0.8}$ </td><td> $93.0_{+3.9}$ </td><td> $77.0_{+6.3}$ </td><td> $93.3_{+0.1}$ </td><td> $47.4_{+3.9}$ </td><td> $71.5_{+4.0}$ </td><td> $68.7_{+3.5}$ </td></tr><tr><td>MOON</td><td> $68.8_{+2.2}$ </td><td> $66.2_{+3.7}$ </td><td> $26.3_{+1.6}$ </td><td> $55.0_{+6.7}$ </td><td> $69.9_{+4.3}$ </td><td> $89.4_{+3.5}$ </td><td> $92.0_{+2.9}$ </td><td> $75.3_{+4.6}$ </td><td> $94.2_{+1.0}$ </td><td> $47.7_{+4.2}$ </td><td> $71.4_{+3.9}$ </td><td> $68.7_{+3.5}$ </td></tr></table>

![](images/2d43ec1aa503a1eb51273151b68fd34befb94a8c8d1e7c67678d9b6230aeb9d9.jpg)

Figure 4. Hyperparameter sensitivity analysis. Results are reported on batch adaptation, Medium scenario, batch size of 1,000. Ours wlo α wlo γ w/o update μ w/o update K  
![](images/b16f19a3084485b0a5643d6d35edcb4cf54bef5fa905916f964cd5644f92eb1b.jpg)  
Figure 5. Detailed ablation study on components, over various batch sizes and scenarios. Each reported performance is averaged over all datasets and runs.

## J. Additional Analyses

Hyperparameter sensitivity. We analyze the sensitivity of MOON to the existing hyperparameters, including iteration number T and the number of neighbors in Laplacian term m in Fig. 4. The results show that MOON is robust to both hyperparameters. Moreover, the performance improves rapidly within the first few iterations and outperforms StatA nearly saturates after T = 3, suggesting fast practical convergence. Together with the dynamic shrinkage mechanism, MOON thus requires no task-specific hyperparameter tuning. We set T = 10 and m = 3 by default for all experiments.

Fine-grained ablation analysis. We present a fine-grained ablation study across batch sizes and sparsity levels in Fig. 5. Consistent with our design motivation, the impact of the class-level weight α diminishes as the effective class set becomes denser (e.g., in the Very High scenario). This confirms that α functions precisely as intended: effectively suppressing outlier classes in sparse settings while relaxing constraints when the distribution approaches uniformity. Conversely, the instance-level weight γ shows marginal influence at small batch sizes due to high statistical variance but becomes increasingly significant at larger batch sizes, where it can leverage stable batch statistics to filter unreliable samples effectively. Furthermore, the performance gain from iterative parameter updates tends to saturate in scenarios with large batches and dense classes, suggesting that abundant data naturally provides sufficient empirical evidence for reliable estimation. Overall, the full MOON framework consistently yields optimal performance.

Table 21. Implementation of shrinkage strength $\beta _ { k }$ .  
(a) Batch adaptation, with batch size of 64.

<table><tr><td>Scenario</td><td>Very Low</td><td>Low</td><td>Medium</td><td>Avg.</td></tr><tr><td>MOON w/ soft  $\beta_{k}$ </td><td>75.3</td><td>73.3</td><td>68.6</td><td>72.4</td></tr><tr><td>MOON w/ hard  $\beta_{k}$ </td><td>74.6</td><td>73.1</td><td>69.4</td><td>72.4</td></tr></table>

(b) Batch adaptation, with batch size of 1,000.

<table><tr><td>Scenario</td><td>Medium</td><td>High</td><td>Very High</td><td>Avg.</td></tr><tr><td>MOON w/ soft  $\beta_{k}$ </td><td>72.1</td><td>70.3</td><td>68.1</td><td>70.2</td></tr><tr><td>MOON w/ hard  $\beta_{k}$ </td><td>72.9</td><td>71.3</td><td>69.2</td><td>71.1</td></tr></table>

(c) Online adaptation, with batch size of 128.

<table><tr><td>Scenario</td><td>Low</td><td>Medium</td><td>High</td><td>Separate</td><td>Avg.</td></tr><tr><td>MOON w/ soft  $\beta_{k}$ </td><td>64.9</td><td>71.8</td><td>73.6</td><td>74.2</td><td>71.1</td></tr><tr><td>MOON w/ hard  $\beta_{k}$ </td><td>66.5</td><td>71.8</td><td>73.4</td><td>73.9</td><td>71.4</td></tr></table>

Implementation of $\beta _ { k }$ . We investigate the implementation strategy for the shrinkage strength $\beta _ { k }$ by comparing the standard soft assignment against the hard assignment (i.e., discretization via argmax) on the probability simplex. As shown in Tab. 21, employing hard assignments consistently yields superior robustness and stability. This advantage stems from the inherent property of the softmax operation, which produces non-zero residual probabilities for all classes. In a soft assignment regime, these residuals can accumulate to form misleading counts for absent classes, thereby weakening the necessary shrinkage. By adopting hard assignments, we effectively eliminate this background noise, ensuring that $\beta _ { k }$ accurately reflects the true class presence and enforces strict anchoring for outlier categories.

## K. Limitations and Future Work

While our MOON demonstrates robust performance and efficiency, there remain promising avenues for future exploration. First, vMF distributions inherently assume isotropy on the hypersphere. Explicitly modeling the anisotropy of VLM representations, for instance, by exploring Fisher-Bingham distributions or other non-isotropic spherical models, could potentially capture more complex feature geometries. Second, MOON can be more deeply integrated with memory banks or caches, enabling more efficient and effective adaptation in sample-wise, online-TTA mode. Finally, the construction of the affinity graph still entails a quadratic complexity with respect to the batch size. Incorporating approximate nearest neighbor search strategies could be beneficial, especially for scaling to large-scale offline adaptation tasks.