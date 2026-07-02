# Condensing Large-Scale Datasets Directly with Minimal Information Loss

Xinyi Shang<sup>1∗</sup>, Peng Sun<sup>2,3∗</sup>, Bei Shi<sup>4,∗</sup>, Zixuan Wang<sup>2</sup>, and Tao Lin<sup>3,†</sup>

<sup>1</sup> University College London, United Kingdom

<sup>2</sup> Zhejiang University, China

<sup>3</sup> Westlake University, China

<sup>4</sup> University of Macau, China

Abstract. Recent advancements in scaling dataset distillation rely heavily on decoupled information extraction pipelines, comprising <sup>Squeeze</sup>, <sup>Recover</sup>, and <sup>Relabel</sup> stages. Despite their scalability to large-scale datasets, these methods sufer from prohibitive computational overhead and poor cross-architecture generalization. In this paper, we reveal the root cause of these bottlenecks: the implicit dual-compression process, from data to model and back to images, inherently induces severe information loss. Crucially, we empirically and theoretically demonstrate that this loss creates a distribution shift that fundamentally compromises the widely adopted <sup>Relabel</sup> strategy, transforming the pre-trained model into an unreliable labeler that yields sub-optimal labels. To overcome these critical flaws, we propose <sup>CIM</sup>, a novel, metric-driven framework that abandons the flawed dual-compression paradigm. Instead, <sup>CIM</sup> explicitly quantifies and minimizes the information gap between the original and synthetic datasets. By directly aligning the data distributions, our approach ensures high-fidelity information condensation and inherently satisfies the prerequisites for efective relabeling. Extensive experiments demonstrate that <sup>CIM</sup> establishes a new state-of-the-art. Notably, it distills ImageNet-1K at an IPC =10 in merely 80 minutes on a single RTX-4090 GPU, achieving an unprecedented 48.7% Top-1 accuracy on ResNet-18 and significantly outperforming previous SOTA approaches, such as NRR-DD and DELT, by 2.6% and 2.9%, respectively. Our code is available at https://github.com/LINs-lab/CIM.

Keywords: Dataset Distillation · Minimal Information Loss · Relabel

## 1 Introduction

Dataset distillation (DD) [45] aims to condense the knowledge from a massive training dataset into a remarkably small synthetic set, preserving comparable generalization performance. By substituting the original data with this compact proxy, dataset distillation drastically reduces training overhead and storage costs.

\* Equal contribution.

<sup>†</sup> Corresponding author. Email: lintao@westlake.edu.cn

Consequently, it has emerged as an enabling technique for diverse applications, including continual learning [29, 57], neural architecture search [39, 44], and privacy-preserving learning [3, 8, 32, 47].

To scale dataset distillation to large-scale datasets like ImageNet-1K [7] and ImageNet-21K [28], a recent line of research focuses on information extraction pipelines [34, 36, 41, 42, 49]. Most notably, the pioneering SRe<sup>2</sup>L [49] introduces a decoupled three-stage paradigm: (i) <sup>Squeeze</sup> the dataset information into a pre-trained model, (ii) <sup>Recover</sup> this information back into the image space to form synthetic data, and (iii) <sup>Relabel</sup> the distilled images using the pre-trained model to inject label-space knowledge. By avoiding costly unrolled optimization, this paradigm yields strong empirical results.

Despite their success, existing extraction-based pipelines sufer from two critical bottlenecks: poor cross-architecture generalization and prohibitive computational overhead, particularly during the <sup>Recover</sup> stage. We identify the root cause as the severe information loss induced by the implicit dual-compression process: first from data to model parameters (<sup>Squeeze</sup>), and then from model back to synthetic images (<sup>Recover</sup>). Furthermore, while the <sup>Relabel</sup> stage improves performance, our theoretical and empirical analyses reveal a hidden vulnerability: its eficacy is strictly conditional on distribution alignment. When the dual-compression process causes the synthetic samples to drift away from the original data distribution, the pre-trained model acts as an unreliable labeler, yielding sub-optimal labels. Therefore, ensuring distributional proximity is crucial to fully unlock the benefits of the <sup>Relabel</sup> strategy.

Motivated by these insights, we abandon the flawed dual-compression paradigm and propose <sup>CIM</sup>, a novel framework that explicitly condenses information by minimizing the information loss between real and distilled datasets (<sup>CIM</sup>). Unlike prior extraction-based methods, which rely on complex inversion to recover informative synthetic images from a pre-trained model, <sup>CIM</sup> is explicitly metric-driven. Specifically, we formulate a principled metric to comprehensively quantify the information gap, and consequently the distribution shift, between the synthetic samples and their real counterparts. By directly optimizing the distilled set to minimize this gap within a unified objective, <sup>CIM</sup> ensures highfidelity information retention and strict distribution alignment. This design not only eliminates the computationally expensive recovery process but also natively satisfies the conditions required for efective relabeling. Our contributions are summarized as follows:

1. We conduct a rigorous revisiting of information extraction-based dataset distillation, revealing that the inherent dual-compression process leads to severe information loss. This loss not only degrades cross-architecture generalization and computational eficiency but also causes a distribution shift that fundamentally compromises the widely adopted <sup>Relabel</sup> strategy.

2. We introduce <sup>CIM</sup>, a novel, metric-driven framework that explicitly quantifies and minimizes the information gap between the original and distilled datasets, achieving more faithful, eficient, and aligned information condensation.

3. Extensive experiments verify that <sup>CIM</sup> establishes a new state-of-the-art across various scale datasets. Notably, our framework distills ImageNet-1K at IPC = 10 in merely 80 minutes on a single RTX-4090 GPU, achieving an unprecedented 48.7% Top-1 accuracy on ResNet-18, outperforming previous SOTA approaches, such as NRR-DD and DELT, by 2.6% and 2.9%, respectively.

## 2 Related Work

The primary aim of dataset distillation is to condense a large dataset into a smaller, yet highly representative subset, preserving its core semantic and statistical characteristics. Wang et al. [45] first introduce the dataset distillation as a bi-level meta-learning optimization problem. The outer loop aims at optimizing the meta-dataset, while the inner loop focuses on training models using the distilled dataset. Existing methods can be roughly divided into three paradigms for solving this complex bi-level optimization problem.

Uni-level optimization-based paradigm. Tackling this bi-level problem is complex, especially when optimizing proxy models via gradient descent, which involves unraveling an intricate computational graph. studies [24, 59] have proposed approximating model training using kernel ridge regression, which provides a closed-form solution for optimal weights, thereby reducing training costs and improving performance. Despite these advancements, such methods still struggle with extensive computational demands or limitations due to the approximations in convex relaxation.

Matching-based paradigm. Another strategy involves emulating the behaviors of the original dataset in the distilled one. They focus on minimizing disparities between surrogate models trained on both synthetic and original datasets. The key metrics for this are matching gradients [17,23,53,57], features [43], distribution [56, 58], and training trajectories [1, 5, 6, 11, 13, 50]. Trajectory and gradient matching, in particular, has shown impressive results with low IPC. However, these methods often tailor the distilled dataset to specific network architectures, limiting their generalizability. Cazenavette et al. [2] address this by proposing the GLaD that synthesizes more realistic images to enhance generalization. These methods incur substantial computational overhead due to the frequent calculation of discrepancies between the distilled and original datasets, requiring numerous iterations for optimization until convergence. Therefore, computational and memory challenges remain, particularly when scaling to large datasets.

Information extraction-based paradigm. The SRe<sup>2</sup>L framework [49], as the first work eficiently scalable to ImageNet-1K, introduces a novel decoupled bi-level learning paradigm. This involves three stages: 1) <sup>Squeeze</sup> relevant information from the original dataset into a pre-trained model, 2) <sup>Recover</sup> this information into the image space, 3) <sup>Relabel</sup> the distilled images by using the pre-trained model to further distill knowledge into the label space. Its eficiency and efectiveness have garnered community attention, spurring a series of research eforts. Shao et al. [34] note that SRe<sup>2</sup>L is limited to specific backbones and layers, impacting the generalization of the distilled dataset. They advocate for using diverse backbones for more precise and efective distillation. Yin et al. [48] further enhance $\mathrm { S R e ^ { 2 } L }$ with curriculum data augmentation. [22] use Wasserstein distance to create more representative images. Sun et al. [41] introduce an optimization-free approach RDED that achieves notable diversity and realism in distilled datasets. Recently, DELT [36] mitigates the issue of low within-class diversity of $\mathrm { S R e ^ { 2 } I }$ by varying the number of iterations for diferent IPCs during the data synthesis phase. Moreover, NRR-DD [42] addresses the shortcoming of $\mathrm { S R e ^ { 2 } L }$ in often emphasizing instance-specific features by efectively capturing both class-general and instance-specific features. EDC [35] introduces a unified framework grounded in both empirical and theoretical foundations. Building on the distillation process introduced by $\mathrm { S R e ^ { 2 } L }$ [49], it incorporates two key enhancements: soft categoryaware matching and dynamic adjustments to the learning rate schedule. These additions can further optimize the distillation process.

## 3 On the Pitfalls of Information Extraction Paradigm

In this section, we revisit the information extraction paradigm for dataset distillation in depth. We highlight two critical aspects: 1) The primary challenge is significant information loss, which hurts the quality and diversity in distilled images. 2) We empirically and theoretically demonstrate the critical attributes necessary for retaining the eficacy of <sup>Relabel</sup> and seek an approach to efectively distill information from original images while adhering to these attributes.

## 3.1 Preliminary

Dataset distillation. Given a large-scale dataset $\mathcal { T } = \{ \mathbf { x } _ { i } , y _ { i } \} _ { i = 1 } ^ { | \mathcal { T } | }$ which consists of $| \tau |$ samples, dataset distillation aims to synthesize a smaller set $\begin{array} { r } { \boldsymbol { S } = ( S _ { X } , S _ { Y } ) = } \end{array}$ $\{ \widetilde { \mathbf { x } } _ { j } , \widetilde { y } _ { j } \} _ { j = \cdot } ^ { | { S } | }$ with |S| synthetic samples such that models trained on $\tau$ will have similar performance as models trained on $s \colon$

$$
\mathbb {E} _ {\mathbf {x} \sim P _ {\mathcal {D}}} [ \ell (\phi_ {\pmb {\theta} _ {\mathcal {T}}} (\mathbf {x}), y) ] \simeq \mathbb {E} _ {\mathbf {x} \sim P _ {\mathcal {D}}} [ \ell (\phi_ {\pmb {\theta} _ {\mathcal {S}}} (\mathbf {x}), y) ],\tag{1}
$$

where $P _ { D }$ is the test real distribution, x is a data sample, ℓ is the loss function, i.e., cross-entropy loss. Here, $\pmb { \theta } _ { T }$ and $\theta _ { \mathcal { S } }$ denotes the parameters of the neural network $\phi$ trained on $\tau$ and $s ,$ respectively.

A closer look at information extraction paradigm. As the first efective yet eficient solution to allow the dataset distillation on diverse-scale datasets such as ImageNet-1K [7], information extraction-based methods have attracted attention and inspired many subsequent works. These methods typically employ a threestage distillation process, indirectly transferring information from the original to the distilled images. The first stage involves condensing information from the complete dataset into pre-trained neural network models via <sup>Squeeze</sup>, followed by the extraction of this information into distilled images using <sup>Recover</sup> [22,49]. Upon the distilled data samples, <sup>Relabel</sup> applies a pre-trained model to further distill knowledge into the label space. However, this paradigm encounters several key challenges:

1. The <sup>Recover</sup> stage necessitates batch normalization in pre-trained models [16] to align the statistical features between distilled and original images [22, 49].

2. Significant information loss during both <sup>Squeeze</sup> and <sup>Recover</sup> stages leads to distilled images with minimal content, adversely afecting performance, particularly in low IPC settings [41].

3. Distilled images often exhibit unrealistic textures or semantics, tailored to specific networks, thereby limiting their generalization ability [34].

4. Despite outperforming other paradigms (e.g., matching-based methods) in terms of eficiency [49], the <sup>Recover</sup> stage is still computationally demanding, requiring numerous optimization iterations [48].

In the meanwhile, the influence of <sup>Relabel</sup> is under-explored [34, 49]. These challenges motivate us to explore a method to simultaneously address four key problems by directly condensing information from the original dataset for image distillation, abandoning traditional decoupled <sup>Squeeze</sup> and <sup>Recover</sup> stages, and to further explore the role of <sup>Relabel</sup>.

## 3.2 Does <sup>Relabel</sup> Always Help Distilled Images?

<sup>Relabel</sup> has become a widely adopted and efective technique in dataset distillation, as demonstrated in recent works [33, 41, 49]. The core idea is to use a model pre-trained on the full real dataset to re-assign labels to the distilled images, and then train a student model on these relabeled distilled samples.

![](images/eb593adb55ccbb6122c340f2e8030b4cd983423fba0f015f2d8ff1f298ba0d83.jpg)

To evaluate the efect of <sup>Relabel</sup>, we compare state-of-the-art DD methods with and without this strategy. As reported in Tab. 7 (App. E), removing <sup>Relabel</sup> leads to a substantial performance degradation. For example, on

Fig. 1: Applying <sup>Relabel</sup> to ADD [53] and DataDAM [30]. We evaluate the distilled images with IPC = 10 during the distillation process. The results indicate that <sup>Relabel</sup> only assists the early-stage distilled datasets.

ImageNet-1k, the accuracy of SRe<sup>2</sup>L and G-VBSM drops to 1.1% and 0.8%, respectively, when <sup>Relabel</sup> is disabled.

Given its efectiveness, a natural question is whether <sup>Relabel</sup> can be used as a plug-and-play component for DD methods that do not originally rely on it. To answer this, we apply <sup>Relabel</sup> to distilled images produced by representative non-<sup>Relabel</sup> methods, such as ADD [53] and DataDAM [30]. The results are shown in Fig. 1. We observe that performance gains are mainly limited to the very early distillation stage (i.e., when the “distilled” images remain close to the initial real images used for initialization). In contrast, as distillation proceeds, relabeling the increasingly optimized distilled images provides little benefit and can even hurt performance.

This behavior suggests a mismatch between the relabeling model and the evolved distilled samples. Specifically, the model used for <sup>Relabel</sup> is trained exclusively on the original real dataset, whereas distilled images gradually deviate from the real-image distribution during optimization. Such a distribution shift alters semantic and/or textural cues, making the pre-trained relabeler less reliable on distilled samples (see App. K for qualitative evidence). Consequently, inaccurate relabels accumulate, and the downstream student model sufers.

A theoretical perspective. Beyond the above empirical observation, we theoretically reveal that a model well-trained on the original, undisturbed samples (i.e., the model used for <sup>Relabel</sup>) may only provide suboptimal labels for shifted distilled samples. Concretely, standard DD typically starts from a subset of real samples $\{ \mathbf { x } _ { j } \} _ { j = 1 } ^ { | \mathcal { S } | }$ drawn from the full dataset T , and iteratively optimizes them into distilled samples $\{ \widetilde { \mathbf { x } } _ { j } \} _ { j = \cdot } ^ { | { \cal S } | }$ with $\widetilde { \mathbf { x } } _ { j } = \mathbf { x } _ { j } + \epsilon _ { j }$ , where $\epsilon _ { j }$ denotes the learned perturbation. A shift occurs because the optimization updates $\epsilon _ { j }$ may change both semantics and textures of $\mathbf { x } _ { j }$ , causing the distilled distribution to deviate from the original one.

Proposition 1. For two original Gaussian distributions $\mathcal { N } _ { 1 } ( \mu _ { 1 } , \sigma ^ { 2 } )$ and $\mathcal { N } _ { 2 } ( \mu _ { 2 } , \sigma ^ { 2 } )$ we first define their shifted versions $\mathcal { N } _ { 1 } ^ { s } ( \mu _ { 1 } + s _ { 1 } , \sigma ^ { 2 } )$ and $\mathcal { N } _ { 2 } ^ { s } ( \mu _ { 2 } + s _ { 2 } , \sigma ^ { 2 } )$ . Then, the optimal classification model $f _ { o p t }$ for $\mathcal { N } _ { 1 } ^ { s }$ and $\mathcal { N } _ { 2 } ^ { s }$ achieves the following suboptimal classification accuracy for N<sub>1</sub> and $\bar { \mathcal { N } } _ { 2 }$ :

$$
P _ {a c c} = \frac {1}{2} \left(\varPhi \left(\frac {\mu_ {2} + s _ {2} - \mu_ {1} + s _ {1}}{2 \sigma}\right) + 1 - \varPhi \left(\frac {\mu_ {1} + s _ {1} - \mu_ {2} + s _ {2}}{2 \sigma}\right)\right)\tag{2}
$$

$$
P _ {a c c} \leq \frac {1}{2} \left(\varPhi \left(\frac {\mu_ {2} - \mu_ {1}}{2 \sigma}\right) + 1 - \varPhi \left(\frac {\mu_ {1} - \mu_ {2}}{2 \sigma}\right)\right)\tag{3}
$$

Here, $\varPhi ( \cdot )$ is the cumulative distribution function (CDF) of the standard normal distribution.

That is, a relabeler trained on the original dataset may provide unreliable labels when applied to distribution-shifted distilled samples. Therefore, to maximize the eficacy of <sup>Relabel</sup>, it is necessary to ensure the distilled data distribution remains close to that of the original real dataset. Only under such alignment can distilled samples be recognized and labeled correctly by the pre-trained relabeler.

Subset selection from real data. To achieve these goals, a straightforward approach involves directly selecting images from the original dataset to construct the distilled dataset. Numerous works [12, 41, 42, 46] have explored methods for selecting diverse and representative key samples from the original dataset to create a distilled dataset. These methods can ensure that the distilled data can be accurately identified by the pre-trained model and maximize the efectiveness of <sup>Relabel</sup>. A notable contribution is RDED [41], which extracts key patches from each image based on high realism scores to construct a distilled dataset.

![](images/968b43361b64a779d01a98ea31e2974bc1d6f444e191f6c7e1b72166b9bf06d0.jpg)  
(a) SRe<sup>2</sup>L [49]

![](images/5ffbb64e5176536d340dc5e25684517b5d6d233c83d1c2dd945d0fa1714b81fe.jpg)  
(b) G-VBSM [34]

![](images/cd468b37993f9e351172b6e781b11b73779b957f32243e751106a752296b391f.jpg)  
(c) RDED [40]  
Fig. 2: Visualization of Feature Distributions for the Original and Distilled Datasets. The distilled datasets are optimized using state-of-the-art distillation techniques: SRe<sup>2</sup>L [49], G-VBSM [34], and RDED [40]. Orange, green, and blue points depict the first three classes of CIFAR-10, while ⋆ points represent the corresponding distilled datasets with images per class (IPC) of 50. The lighter shades represent the original dataset.

To further validate whether the sample selected by RDED can achieve the distribution as the original dataset, we visualize the feature distributions <sup>1</sup> of the distilled dataset alongside those of the original dataset in Fig. 2. Specifically, we compare the state-of-the-art methods, including SRe<sup>2</sup>L [49], G-VBSM [34], and RDED [41]. Among these, RDED demonstrates the closest alignment with the feature distribution of the original dataset. Additionally, as shown in Tab. 1 and Tab. 2, RDED achieves superior performance compared to the other two SOTA methods. Therefore, by default, <sup>CIM</sup> adopts the selection mechanism of RDED<sup>2</sup>. Note that the subset selection strategy itself is not the focus of this work. Our proposed <sup>CIM</sup> is selection-method-agnostic and can incorporate various selection strategies, as demonstrated in Tab. 6.

## 4 Methodology

The following section begins by introducing the formal definitions of efective information for a given sample and the resulting information gap between any two given samples. Furthermore, we propose a method based on minimizing the information gap between the selected sample subsets and the distilled samples to ensure the preservation of information integrity in the distilled set. The distillation process of our <sup>CIM</sup> is depicted in Fig. 3, with the detailed algorithm outlined in Alg. 1 in Appendix.

On the information gap of two samples. For the selected IPC subsets of key images, we aim to compress each of them into a more compact pixel space, i.e., distilled image x, thus forming S<sub>X</sub> . However, given the constraints of limited pixel space storage, using a naive solution—e.g., directly resizing and concatenating multiple images from a subset into one—results in a significant reduction in the fineness and detail of the original images. To efectively capture and condense the salient information from the original samples into the distilled ones, we aim to enable each distilled sample $\widetilde { \mathbf { x } }$ to encapsulate the information of a selected subset from $\mathcal { T } _ { X }$ . We begin by formally defining the efective information of a data sample as follows.

Definition 1 (Observation-based Efective Information). Let $\mathbf { x } _ { i }$ represent a sample from any domain $( e . g .$ , image). Define an observer group $\mathcal { R } = \{ \xi _ { j } \}$ where each observer $\xi _ { j }$ is capable of extracting or interpreting features from $\mathbf { x } _ { i }$ and $| \mathcal { R } | \geq 1$ . The efective information of the sample $\mathbf { x } _ { i } ,$ , as observed by the group $\mathcal { R } _ { : }$ , is conceptualized as the distribution $\mathcal { P } _ { \mathbf { x } _ { i } | \mathcal { R } } ( z )$ . This distribution is formulated as:

$$
\mathcal {P} _ {\mathbf {x} _ {i} | \mathcal {R}} (z) := \left\{z \mid z = \xi_ {j} (\mathbf {x} _ {i}), \forall \xi_ {j} \in \mathcal {R} \right\},\tag{4}
$$

where z denotes the set of features or interpretations extracted from $\mathbf { x } _ { i }$ by an observer $\xi _ { j }$ within $\mathcal { R }$ .

The Def. 1 posits that the efective information of a sample encompasses the set of its features as perceived or extracted by a diverse set of observers. Samples are considered to have similar efective information if they result in comparable feature sets across the observers in $\mathcal { R }$ . We consequently define the efective information gap between two given samples $\mathbf { x } _ { i }$ and $\mathbf { x } _ { j }$

Definition 2 (Pairwise Efective Information Gap). Let $\mathbf { x } _ { i }$ and $\mathbf { x } _ { j }$ represent two samples from any domain, and given an observer group $\mathcal { R } = \{ \xi _ { j } \}$ . This efective information gap is formulated as the diference between their efective information:

$$
I _ {G} (\mathbf {x} _ {i}, \mathbf {x} _ {j}; \mathcal {R}) = \mathrm{D} _ {\mathrm{KL}} (\mathcal {P} _ {\mathbf {x} _ {i} | \mathcal {R}} | | \mathcal {P} _ {\mathbf {x} _ {j} | \mathcal {R}}).\tag{5}
$$

Compressing information into distilled set. However, directly minimizing the information gap between distilled set $\boldsymbol { \mathcal { S } } _ { \boldsymbol { X } }$ and original set $\mathcal { D } _ { X }$ through Def. 2 is intractable, due to Eq. 5 is a sample-level metric that cannot be directly applied to the set. Thanks to the widely adapted data augmentation techniques $\mathcal { A }$ to enhance the diversity of each sample $\widetilde { \mathbf { x } }$ in practice, we relax the estimation of $\mathrm { E q . ~ 5 }$ through calculating the efective information gap between a set of N original samples

![](images/f585d287677411bb417c31b1f440f0668e6bb0b056ec49325600df3d5d687520.jpg)  
Fig. 3: Distillation Process of Our <sup>CIM</sup>. First, IPC subsets are selected from the original data $\tau ,$ where each subset contains images denoted as $\{ \mathbf { x } _ { j } \} _ { j = 1 } ^ { N } ;$ Then, for each image $\widetilde { \mathbf { x } }$ in the initial distilled data $s ,$ , the RandomCrop is applied to generate views $\{ \widetilde { \mathbf { x } } ^ { n } \} _ { n = 1 } ^ { N }$ . The information gap $I _ { G } ( \mathbf { x } _ { j } , \widetilde { \mathbf { x } } ^ { n } )$ is then minimized for each view. This process is iteratively performed for every distilled image $\widetilde { \mathbf { x } } ;$

$\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N }$ and a distilled sample $\widetilde { \mathbf { x } } _ { j }$ using N augmented views of the distilled samples, i.e.,

$$
\mathbb {E} _ {(\mathbf {x} _ {i}, \widetilde {\mathbf {x}} _ {j} ^ {(i)}) \sim (\{\mathbf {x} _ {i} \} _ {i = 1} ^ {N}, \mathcal {A} (\widetilde {\mathbf {x}} _ {j}))} I _ {G} (\mathbf {x} _ {i}, \widetilde {\mathbf {x}} _ {j} ^ {(i)}; \mathcal {R}) \text {s.t.} \quad \mathcal {A} (\widetilde {\mathbf {x}} _ {j}) = \{\widetilde {\mathbf {x}} _ {j} ^ {(1)}, \widetilde {\mathbf {x}} _ {j} ^ {(2)}, \ldots , \widetilde {\mathbf {x}} _ {j} ^ {(N)} \}.\tag{6}
$$

However, we still cannot directly distill sample $\widetilde { \mathbf { x } } _ { j }$ through Eq. 6 due to the intractable KL divergence estimation in Eq. 5, and thus we derive Thm. 1 that bounds Eq. 5 to enable it computationally tractable.

Theorem 1 (Pairwise Efective Information Gap). Let $\mathbf { x } _ { i }$ and $\mathbf { x } _ { j }$ represent two samples from any domain, and given an observer group ${ \mathcal { R } } = \{ \xi _ { j } \}$ . The efective information gap is upper-bounded as

$$
I _ {G} (\mathbf {x} _ {i}, \mathbf {x} _ {j}; \mathcal {R}) \leq \mathbb {E} _ {\xi_ {k} \sim \mathcal {R}} \| \xi_ {k} (\mathbf {x} _ {i}) - \xi_ {k} (\mathbf {x} _ {j}) \| ^ {2} \mathrm{s.t.} k \in [ 1, | \mathcal {R} | ].\tag{7}
$$

Therefore, we can capture a distilled sample $\widetilde { \mathbf { x } } _ { j }$ through combining Eq. 6 and Thm. 1, namely,

$$
\arg \min _ {\widetilde {\mathbf {x}} _ {j}} \mathbb {E} _ {(\mathbf {x} _ {i}, \widetilde {\mathbf {x}} _ {j} ^ {(i)}) \sim (\{\mathbf {x} _ {i} \} _ {i = 1} ^ {N}, \mathcal {A} (\widetilde {\mathbf {x}} _ {j}))} \mathbb {E} _ {\xi_ {k} \sim \mathcal {R}} \| \xi_ {k} (\mathbf {x} _ {i}) - \xi_ {k} (\widetilde {\mathbf {x}} _ {j} ^ {(i)}) \| ^ {2}\tag{8}
$$

To reduce computational complexity, we utilize RandomCrop to generate the augmentations $\boldsymbol { \mathcal { A } } ( \widetilde { \mathbf { x } } _ { j } )$ and concatenate resized real images to initialize distilled sample $\widetilde { \mathbf { x } } _ { j }$ . We further consider a specific scenario where the observer group consists of only one pre-trained model across various transformations<sup>3</sup>. This strategy allows us to employ multiple observers without requiring an excessive number of pre-trained models. Our loss function is defined as follows to achieve Eq. 8:

$$
\mathcal {L} _ {\Delta \widetilde {\mathbf {x}} _ {j}} = \mathbb {E} _ {(\mathbf {x} _ {i}, \widetilde {\mathbf {x}} _ {j} ^ {(i)}) \sim (\{\mathbf {x} _ {i} \} _ {i = 1} ^ {N}, \mathcal {A} (\widetilde {\mathbf {x}} _ {j} + \Delta \widetilde {\mathbf {x}} _ {j}))} \mathbb {E} _ {\zeta_ {k} \sim \mathcal {G}} \left\| \zeta_ {k} \circ \phi_ {\boldsymbol {\theta} _ {\mathcal {T}}} (\mathbf {x} _ {i}) - \zeta_ {k} \circ \phi_ {\boldsymbol {\theta} _ {\mathcal {T}}} (\widetilde {\mathbf {x}} _ {j} ^ {(i)}) \right\| ^ {2}\tag{9}
$$

where $\mathcal { R } = \{ \xi _ { k } \ | \ \xi _ { k } = \zeta _ { k } \circ \phi _ { \pmb \theta _ { T } } , \forall \zeta _ { k } \sim \mathcal { G } \}$ and $\mathcal { G }$ denotes the transformation group, $\mathcal { A } ( \widetilde { \mathbf { x } } _ { j } + \Delta \widetilde { \mathbf { x } } _ { j } ) = \{ \widetilde { \mathbf { x } } _ { j } ^ { ( 1 ) } , \widetilde { \mathbf { x } } _ { j } ^ { ( 2 ) } , \ldots , \widetilde { \mathbf { x } } _ { j } ^ { ( N ) } \}$ . By minimizing Eq. 9, we find the optimal ∆x<sup>⋆</sup><sub>j</sub> and capture the image $\widetilde { \mathbf { x } } _ { j } \gets \widetilde { \mathbf { x } } _ { j } + \varDelta \widetilde { \mathbf { x } } _ { j } ^ { \star }$ as the distilled image.

Balancing the semantic and textural information. Directly generating the distilled image $\widetilde { \mathbf { x } } _ { j }$ through aligning its efective information with the original image set $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { N }$ in the last layer of model $\phi _ { \pmb { \theta } _ { T } }$ may lead to a substantial loss of texture information. The reason is that the model $\phi _ { \pmb { \theta } _ { T } }$ tends to extract semantic information from input images, which often results in a notable drop in the texture details. To balance between semantic richness and texture preservation in the distilled image $\widetilde { \mathbf { x } } _ { j } ,$ we leverage intermediate model features instead of the last-layer logits, which helps in retaining more textural details (see Section 5.3 for the validation of its robustness).

<sup>Relabel</sup> across Various Transformations. We propose an improved relabeling strategy for our informative distilled images $\widetilde { \mathbf { x } } ,$ which provides more diverse and informative knowledge in the label space compared to the basic one-shot labeling approach. The standard <sup>Relabel</sup> technique [49], inspired by [52], suggests that a random image crop may contain a diferent object than the one originally labeled, leading to inaccurate or misleading training data. This highlights the limitations of the one-shot labeling strategy in expressing suficient knowledge for cropped images.

Our extended <sup>Relabel</sup> approach considers a wider range of image transformations beyond random cropping. Similar to the soft labeling approach in [37], we generate transformed-view-level soft label $\widetilde { y } _ { k } \ = \ \phi _ { \pmb { \theta } _ { T } } ( \zeta _ { k } ( \widetilde { \mathbf { x } } ) )$ , where $\zeta _ { k } ( \widetilde { \mathbf { x } } )$ represents the k-th transformation applied to the distilled image x.

Therefore, we can train the model $\phi _ { \pmb { \theta } _ { S } }$ on the distilled data by minimizing:

$$
\mathcal {L} = - \sum_ {j} \sum_ {k} \| \phi_ {\pmb {\theta} _ {S}} (\zeta_ {k} (\widetilde {\mathbf {x}} _ {j})) - \widetilde {y} _ {(j, k)} \| ^ {2}.\tag{10}
$$

The whole distillation process for an entire dataset by using our <sup>CIM</sup> is illustrated in Alg. 1 in Appendix.

## 5 Experiment

In this section, we evaluate the performance of our proposed <sup>CIM</sup> over various datasets and neural architectures. First, we demonstrate the superior results of <sup>CIM</sup> on real-world datasets, cross-architecture generalization and eficiency. We next perform comprehensive ablation studies to evaluate the impact of each component in our proposed method, as well as to analyze the influence of hyperparameter choices and subset selection strategies. Finally, we demonstrate the superior performance of our approach, <sup>CIM</sup>, in continual learning applications.

## 5.1 Experimental Setting

Datasets and neural network architectures. We conduct experiments on varying scales and resolutions of images.

– Small-scale: we evaluate on two datasets, including CIFAR-10 (32 × 32) [19] and CIFAR-100 (32 × 32) [18].

– Large-scale: we also use two large-scale high-resolution datasets including Tiny-ImageNet (64 × 64) [20] and ImageNet-1K (224 × 224) [7].

Following prior dataset distillation works [13, 49, 58], we employ ConvNet [13], ResNet-18 [14], and MobileNet-V2 [31], ViT-T/16 [9], ShufleNet-v2-x2.0 [25], DenseNet-121 [15], as our backbone networks. Specifically, for ConvNet, we use Conv-3 on CIFAR-10/100 and use Conv-4 on Tiny-ImageNet and ImageNet-1K. More details about the used datasets and architectures can be found in $\mathrm { A p p }$ . G.

Baselines. We compare our method with several SOTA distillation methods that can scale to large high-resolution datasets, including G-VBSM [34], SRe<sup>2</sup>L [49], RDED [41], CDA [48], WMDD [21], Teddy [51], CUDD [10], EDC [35], DWA [10], CV-DD [4], INFER [54], GIFT [33], NRR-DD [42], DELT [36] . The results of additional baseline methods, including DataDAM [30], ADD [53], IDM [58], CDA [48], WMDD [21], DATM [13], DREAM [23], and FreD [38], are presented in App. G. Comprehensive details regarding these approaches are also provided in the same appendix.

Table 1: Comparison with baseline approaches on CIFAR-10 (“CF-10”) and CIFAR-100 (“CF-100”). In the table, bold indicates the best result. Underline means the second-best result. IPC refers to the Images Per Class within distilled datasets.

<table><tr><td colspan="2"></td><td colspan="4">ConvNet</td><td colspan="4">ResNet-18</td><td colspan="4">ResNet-50</td></tr><tr><td>Dataset</td><td>IPC</td><td>G-VBSM</td><td>SRe2L</td><td>RDED</td><td>CIM (Ours)</td><td>G-VBSM</td><td>SRe2L</td><td>RDED</td><td>CIM (Ours)</td><td>G-VBSM</td><td>SRe2L</td><td>RDED</td><td>CIM (Ours)</td></tr><tr><td rowspan="3">CF-10</td><td>1</td><td> $21.2 \pm 0.2$ </td><td> $22.2 \pm 1.1$ </td><td> $28.9 \pm 0.4$ </td><td> $\mathbf{37.4 \pm 0.4}$ </td><td> $17.0 \pm 1.0$ </td><td> $19.9 \pm 0.9$ </td><td> $27.8 \pm 0.4$ </td><td> $\mathbf{32.3 \pm 0.2}$ </td><td> $17.2 \pm 0.9$ </td><td> $20.2 \pm 0.5$ </td><td> $24.8 \pm 0.5$ </td><td> $\mathbf{31.8 \pm 0.7}$ </td></tr><tr><td>10</td><td> $38.6 \pm 0.8$ </td><td> $39.6 \pm 0.4$ </td><td> $\underline{56.0 \pm 0.1}$ </td><td> $\mathbf{61.4 \pm 0.4}$ </td><td> $36.3 \pm 0.7$ </td><td> $39.4 \pm 0.9$ </td><td> $\underline{47.3 \pm 0.5}$ </td><td> $\mathbf{66.2 \pm 0.9}$ </td><td> $33.8 \pm 1.1$ </td><td> $37.2 \pm 0.6$ </td><td> $\underline{45.1 \pm 0.7}$ </td><td> $\mathbf{62.9 \pm 0.3}$ </td></tr><tr><td>50</td><td> $62.7 \pm 0.5$ </td><td> $57.7 \pm 0.4$ </td><td> $\underline{71.1 \pm 0.2}$ </td><td> $\mathbf{74.7 \pm 0.2}$ </td><td> $64.5 \pm 0.6$ </td><td> $62.8 \pm 1.2$ </td><td> $\underline{76.4 \pm 0.4}$ </td><td> $\mathbf{85.1 \pm 0.3}$ </td><td> $61.5 \pm 0.6$ </td><td> $61.6 \pm 0.2$ </td><td> $\underline{74.1 \pm 0.6}$ </td><td> $\mathbf{84.2 \pm 0.3}$ </td></tr><tr><td rowspan="3">CF-100</td><td>1</td><td> $13.4 \pm 0.3$ </td><td> $12.9 \pm 0.1$ </td><td> $21.8 \pm 0.4$ </td><td> $\mathbf{27.4 \pm 0.4}$ </td><td> $13.4 \pm 0.5$ </td><td> $11.5 \pm 0.4$ </td><td> $4.6 \pm 0.1$ </td><td> $\mathbf{31.1 \pm 0.7}$ </td><td> $12.6 \pm 0.6$ </td><td> $10.1 \pm 0.1$ </td><td> $4.5 \pm 0.2$ </td><td> $\mathbf{28.1 \pm 0.2}$ </td></tr><tr><td>10</td><td> $38.7 \pm 0.8$ </td><td> $34.2 \pm 0.3$ </td><td> $\underline{47.0 \pm 0.3}$ </td><td> $\mathbf{49.3 \pm 0.3}$ </td><td> $47.0 \pm 0.4$ </td><td> $42.7 \pm 0.5$ </td><td> $53.4 \pm 0.3$ </td><td> $\mathbf{61.7 \pm 0.2}$ </td><td> $47.5 \pm 0.5$ </td><td> $44.2 \pm 0.5$ </td><td> $\underline{54.0 \pm 0.3}$ </td><td> $\mathbf{62.9 \pm 0.4}$ </td></tr><tr><td>50</td><td> $53.8 \pm 0.4$ </td><td> $52.2 \pm 0.3$ </td><td> $\mathbf{55.3 \pm 0.2}$ </td><td> $\mathbf{55.1 \pm 0.2}$ </td><td> $60.0 \pm 0.1$ </td><td> $57.4 \pm 0.2$ </td><td> $\underline{64.0 \pm 0.0}$ </td><td> $\mathbf{67.0 \pm 0.1}$ </td><td> $62.2 \pm 0.3$ </td><td> $60.6 \pm 0.2$ </td><td> $\underline{65.8 \pm 0.3}$ </td><td> $\mathbf{67.9 \pm 0.1}$ </td></tr></table>

Table 2: Comparison with the state-of-the-art dataset distillation methods on Tiny-ImageNet (“TN-IN”) and ImageNet-1K (“IN-1K”) on ResNet-18.

<table><tr><td>Dataset</td><td>IPC</td><td>SRe2L</td><td>G-VBSM</td><td>CDA</td><td>WMDD</td><td>RDED</td><td>Teddy</td><td>GIFT</td><td>NRR-DD</td><td>DELT</td><td>CIM (Ours)</td></tr><tr><td rowspan="3">TN-IN</td><td>1</td><td> $13.5 \pm 0.2$ </td><td> $9.0 \pm 0.1$ </td><td>-</td><td> $7.6 \pm 0.2$ </td><td> $15.4 \pm 0.6$ </td><td>-</td><td> $\underline{15.9 \pm 0.3}$ </td><td> $13.5 \pm 0.2$ </td><td> $9.3 \pm 0.5$ </td><td> $\mathbf{25.1 \pm 0.4}$ </td></tr><tr><td>10</td><td> $43.6 \pm 0.5$ </td><td> $37.7 \pm 0.3$ </td><td>-</td><td> $41.8 \pm 0.1$ </td><td> $48.4 \pm 0.3$ </td><td>-</td><td> $\underline{49.2 \pm 0.1}$ </td><td> $45.2 \pm 0.2$ </td><td> $43.0 \pm 0.1$ </td><td> $\mathbf{53.3 \pm 0.1}$ </td></tr><tr><td>50</td><td> $53.4 \pm 0.3$ </td><td> $52.2 \pm 0.0$ </td><td> $48.7 \pm 0.1$ </td><td> $\underline{59.4 \pm 0.5}$ </td><td> $57.4 \pm 0.2$ </td><td> $55.2 \pm 0.1$ </td><td> $\underline{58.1 \pm 0.1}$ </td><td> $\mathbf{61.2 \pm 0.1}$ </td><td> $55.7 \pm 0.5$ </td><td> $58.6 \pm 0.1$ </td></tr><tr><td rowspan="4">IN-1K</td><td>1</td><td> $2.8 \pm 0.2$ </td><td> $2.1 \pm 0.1$ </td><td>-</td><td> $3.2 \pm 0.3$ </td><td> $5.9 \pm 0.1$ </td><td>-</td><td>-</td><td> $\mathbf{11.6 \pm 0.2}$ </td><td>-</td><td> $\mathbf{7.3 \pm 0.3}$ </td></tr><tr><td>10</td><td> $31.1 \pm 0.1$ </td><td> $35.7 \pm 0.2$ </td><td>-</td><td> $38.2 \pm 0.2$ </td><td> $41.1 \pm 0.2$ </td><td> $34.1 \pm 0.1$ </td><td> $43.2 \pm 0.1$ </td><td> $\underline{46.1 \pm 0.0}$ </td><td> $45.8 \pm 0.0$ </td><td> $\mathbf{48.7 \pm 0.2}$ </td></tr><tr><td>50</td><td> $49.5 \pm 0.2$ </td><td> $51.6 \pm 0.2$ </td><td> $53.5 \pm 0.1$ </td><td> $57.6 \pm 0.5$ </td><td> $55.3 \pm 0.2$ </td><td> $52.5 \pm 0.1$ </td><td> $56.5 \pm 0.1$ </td><td> $\underline{60.1 \pm 0.1}$ </td><td> $59.2 \pm 0.1$ </td><td> $\mathbf{60.4 \pm 0.1}$ </td></tr><tr><td>100</td><td> $54.3 \pm 0.2$ </td><td> $56.1 \pm 0.1$ </td><td> $58.0 \pm 0.1$ </td><td> $\underline{60.7 \pm 0.0}$ </td><td> $58.6 \pm 0.1$ </td><td> $56.5 \pm 0.0$ </td><td> $59.3 \pm 0.2$ </td><td>-</td><td> $\mathbf{62.4 \pm 0.1}$ </td><td> $\mathbf{62.4 \pm 0.1}$ </td></tr></table>

Implementation details. By default, <sup>CIM</sup> employs the subset selection mechanism of RDED. Notably, the proposed <sup>CIM</sup> is selection-method-agnostic and can seamlessly integrate alternative selection strategies. All distilled datasets synthesized from these baselines are evaluated using the same post-training process. All the hyper-parameters used in our Alg. 1 are general, insensitive, and easy-implemented for all datasets and network architectures (c.f. Sec. 5.3 and App. I for validation). We employ a generalized configuration for $\tau ^ { \prime }$ (c.f. App. C for definition), where the size of subset $\vert \mathcal { T } ^ { \prime } \vert$ is set as 300. We set the number $N = 4$ of images squeezed in a distilled image and number $M = 2 0 0$ of compression iteration (c.f. Alg. 1 for definition). More implementation details are provided in App. G.

## 5.2 Comparison with the SOTA Methods

Small-scale datasets. Following previous research [1, 6, 58], we set IPC to 1, 10, and 50 to compare with baselines on varying datasets and networks. As the results reported in Tab. 1, our method <sup>CIM</sup> outperforms other methods on varying datasets and neural networks with diferent IPC. It is noteworthy that prior information extraction-based solutions like $\mathrm { S R e ^ { 2 } L }$ struggle in scenarios involving small distilled datasets such as CIFAR-100 with $\tt I P C = 1$ or CIFAR-10 with all IPC, further verifying our claims proposed in Sec. 3.1. Detailed comparison among more datasets and baselines can be found in App. H. Furthermore, we visualize the distilled images of various methods in App. K.

Large-scale datasets. To evaluate the practicality of <sup>CIM</sup> in real-world settings, we further conduct experiments on the large-scale and high-resolution benchmarks Tiny-ImageNet and ImageNet-1K. The corresponding results are reported in Tab. 2. In addition, we compare <sup>CIM</sup> with several state-of-the-art methods on ResNet-50, as shown in Tab. 3. Note that we exclude some methods listed in

Table 3: Comparison with baselines on ImageNet-1K using ResNet-50.

<table><tr><td>IPC</td><td>G-VBSM</td><td>SRe2L</td><td>RDED</td><td>Teddy</td><td>CUDD</td><td>CDA</td><td>EDC</td><td>DWA</td><td>INFER</td><td>CV-DD</td><td>CIM (Ours)</td></tr><tr><td>10</td><td> $42.6 \pm 0.4$ </td><td> $38.3 \pm 0.5$ </td><td> $46.2 \pm 0.3$ </td><td> $37.2 \pm 0.1$ </td><td> $46.2 \pm 0.3$ </td><td>-</td><td> $\underline{54.1 \pm 0.2}$ </td><td> $43.0 \pm 0.1$ </td><td> $38.3 \pm 0.2$ </td><td> $51.3 \pm 0.2$ </td><td> $\underline{54.3 \pm 0.4}$ </td></tr><tr><td>50</td><td> $60.3 \pm 0.1$ </td><td> $58.4 \pm 0.1$ </td><td> $62.5 \pm 0.1$ </td><td> $58.5 \pm 0.1$ </td><td> $63.6 \pm 0.2$ </td><td> $61.3 \pm 0.2$ </td><td> $\underline{64.3 \pm 0.2}$ </td><td> $62.3 \pm 0.3$ </td><td> $63.4 \pm 0.2$ </td><td> $63.9 \pm 0.1$ </td><td> $\underline{65.9 \pm 0.1}$ </td></tr><tr><td>100</td><td> $64.1 \pm 0.1$ </td><td> $62.9 \pm 0.1$ </td><td> $65.5 \pm 0.1$ </td><td>-</td><td> $\underline{66.7 \pm 0.1}$ </td><td> $65.1 \pm 0.1$ </td><td>-</td><td> $65.7 \pm 0.1$ </td><td>-</td><td>-</td><td> $\underline{67.6 \pm 0.1}$ </td></tr></table>

Table 4: Top-1 accuracy (%) on ImageNet-1k for cross-architecture generalization. We utilize ResNet-18 for distilling the original dataset. Subsequently, the distilled data is transferred to other architectures, with IPC = 10.

<table><tr><td>Verifier</td><td>EfficientNet-B0</td><td>ResNet-18</td><td>MobileNet-V2</td><td>ViT-T/16</td><td>ShuffleNet-v2-x2.0</td><td>DenseNet-121</td></tr><tr><td> $SRe^{2}L$ </td><td>35.2 ± 0.0</td><td>35.0 ± 0.3</td><td>27.6 ± 0.0</td><td>3.2 ± 0.0</td><td>38.4 ± 0.3</td><td>43.2 ± 0.1</td></tr><tr><td>G-VBSM</td><td>31.5 ± 0.0</td><td>30.2 ± 1.2</td><td>26.0 ± 0.4</td><td>3.3 ± 0.1</td><td>31.8 ± 0.9</td><td>39.6 ± 0.1</td></tr><tr><td>RDED</td><td>41.1 ± 0.0</td><td>38.4 ± 0.7</td><td>34.8 ± 0.0</td><td>8.5 ± 0.1</td><td>43.2 ± 0.3</td><td>47.4 ± 0.3</td></tr><tr><td>Ours</td><td>48.5 ± 0.2</td><td>48.7 ± 0.3</td><td>42.3 ± 0.2</td><td>10.8 ± 0.0</td><td>50.8 ± 0.2</td><td>54.4 ± 0.2</td></tr></table>

Tab. 2 from the ResNet-50 comparison, since their distilled datasets are not available on larger networks, making a fair reproduction infeasible. Furthermore, beyond the commonly used setting of IPC =50, we also report results under larger budgets (e.g., IPC =100) to assess scalability. It is obvious that our <sup>CIM</sup> achieves the best performance on most scenarios, demonstrating the efectiveness.

Cross-architecture generalization. An important property of the distilled datasets is their good generalization capability across unseen architectural models. Here we evaluate the generalizability of our distilled datasets when IPC =10. As reported in Tab. 4, our distilled dataset achieves the best performance on unseen networks, which reflects the good generalizability of the data and labels distilled by our method. Our success stems from our <sup>CIM</sup> efectively keeps both textural and semantic information in distilled images.

Eficiency comparison. Eficiency is also a key factor during the distillation process. Here, we use a single RTX-4090 GPU for two methods to conduct experiments on Tiny-ImageNet. The reason why we mainly compare with $\mathrm { S R e ^ { 2 } I }$ and G-VBSM is based on their outstanding as eficient optimization-based methods currently. We evaluate the distillation eficiency by recording the run-time cost and peak GPU memory usage of distilling the image. As evidenced in Tab. 5, our <sup>CIM</sup> achieves superior eficiency in comparison to SOTA methods, with the exception of RDED, which benefits from an optimization-free paradigm [41], demonstrating a notable advantage of eficacy and eficiency. Significantly, our algorithm can ofer a versatile peak memory capacity, enabling adjustments to batch size dynamically without sacrificing performance. This eficiency is attributed to the fact that our Alg. 1 can independently<sup>4</sup> optimize images, allowing us to distill them one by one. More comparisons are in App. H.

![](images/9a4eebf667d3fcaf48ccee8152fa757b957609f0f964c962c663507cb6429b98.jpg)  
(a) Number M

![](images/669882814b4c7e402c415aab08e01c45c56105a29c5693b2106b49ac9eecd3df.jpg)  
(b) Number N

![](images/c2b5df3e033746b946a4b6d01f37e397b585365f2c915c3cb1df45c4abc5043a.jpg)  
(c) Alignment Layer

![](images/5845c364461800b88a7836eb52af155e3deb87fa5c4a5ebbd0b817a3706219b7.jpg)  
(d) Number K  
Fig. 4: Ablation study on each component in our <sup>CIM</sup>. We evaluate our <sup>CIM</sup> with diferent number M of compression iterations (4a), number N of images squeezed in one distilled image (4b), feature alignment layer (4c), and the information gap with respect to the number K of iterations(4d). The yellow •, red •, blue • and deep blue • denote CIFAR-10, CIFAR-100, Tiny-ImageNet and ImageNet-1k respectively.

Table 5: Eficiency comparison on varying networks. Following $\mathrm { S R e ^ { 2 } L }$ , Time Cost is the consumption (s) when generating 100 images simultaneously, and the peak value of GPU memory usage is measured with a batch size of 100.

<table><tr><td rowspan="2">Architecture</td><td colspan="4">Time Cost (s)</td><td colspan="4">Peak Memory (GB)</td></tr><tr><td> $SRe^2L$ </td><td>G-VBSM</td><td>RDED</td><td>Ours</td><td> $SRe^2L$ </td><td>G-VBSM</td><td>RDED</td><td>Ours</td></tr><tr><td>Conv-4</td><td>51.68</td><td>259.84</td><td>1.682</td><td>13.02</td><td>1.36</td><td>4.94</td><td>0.74</td><td>0.65</td></tr><tr><td>ResNet-18</td><td>191.14</td><td>259.84</td><td>2.78</td><td>25.34</td><td>3.62</td><td>4.94</td><td>0.92</td><td>1.56</td></tr><tr><td>MobileNet-V2</td><td>114.05</td><td>259.84</td><td>4.07</td><td>18.81</td><td>1.27</td><td>4.94</td><td>0.29</td><td>0.64</td></tr></table>

## 5.3 Ablation Study

In this section, we set the default IPC = 10 and employ ConvNet as the network backbone to examine how the components used in our <sup>CIM</sup> influence the quality of distilled dataset (see App. I for more investigation).

Influence of compression iteration number M. The number of distillation iterations, denoted as M, impacts two aspects: 1) A higher M enhances <sup>CIM</sup>’s ability to generate higher-quality images; 2) A lower M ensures a faster execution of our Alg. 1. Consequently, choosing an optimal iteration number M represents a balance between quality and speed. As illustrated in Fig. 4a, an iteration count of M = 200 ofers a well-rounded compromise for various datasets. Additionally, it is noteworthy that our <sup>CIM</sup> exhibits robustness to variations in M . Specifically, setting M beyond 200 yields negligible diferences in performance.

Influence of size of compressed images N. Though we can compress more original images $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { N }$ into each distilled image x by increasing N to benefit the feature diversity (c.f. Sec. 4), it also results in less information preserved from each original image x<sub>i</sub>. Fig. 4b showcase that the validation performance rises to the highest on selected four datasets when $N = 4$

Why and how to choose the feature alignment layer? The experimental results in Fig. 4c illustrate the impact of the feature alignment layer, alongside the discussion in Sec. 4. As depicted in Fig. 4c, high performance is often achieved through alignment at the middle layer. This outcome likely stems from the varied information encoded at diferent network depths: shallow layers capture more textural details, whereas logit and deep layers are more adept at encoding semantic information. Thus, aligning at the middle layer enables the distilled dataset to achieve an optimal balance of these information types.

![](images/f55a461516fe97b6fcf45796333733d5a5cc1379dcf5b797e237c68d5dc5a616.jpg)  
(a) CIFAR-10

![](images/c063af2d1c1193952ac207002e19554dc4668883b58cb04539e96c2430d47eff.jpg)  
(b) CIFAR-100

![](images/6338582f232a773c3c56f13a645a12cdf5face7636e727de207298cdc6db7b09.jpg)  
(c) Tiny-ImageNet  
Fig. 5: Application of continual learning on various datasets when $\tt T P C = 1 0$

Table 6: Comparison of <sup>CIM</sup> using various selection strategies with $\mathtt { I P C } = 1 0 .$ By default, <sup>CIM</sup> uses the RDED selection method.

<table><tr><td></td><td>Random</td><td>K-means</td><td>Herding</td><td>Ours</td></tr><tr><td>CIFAR-10</td><td>64.5 ± 0.2</td><td>65.5 ± 1.2</td><td>66.2 ± 0.8</td><td>66.6 ± 0.1</td></tr><tr><td>CIFAR-100</td><td>60.4 ± 0.3</td><td>60.4 ± 0.2</td><td>61.0 ± 0.1</td><td>61.8 ± 0.1</td></tr><tr><td>Tiny-ImageNet</td><td>51.4 ± 0.5</td><td>51.3 ± 0.2</td><td>51.3 ± 0.2</td><td>53.2 ± 0.1</td></tr><tr><td>ImageNet-1k</td><td>45.3 ± 0.0</td><td>44.8 ± 0.3</td><td>45.3 ± 0.1</td><td>48.6 ± 0.2</td></tr></table>

Influence of iteration number K on the information gap. Fig. 4d illustrates the efect of iteration count K on the information gap. An increased iteration count K efectively reduces the information gap, enabling the model to capture and retain more features from the original images in the distilled dataset. However, as K nears 200, further iterations ofer minimal gains. This trend suggests that an appropriate choice of K balances eficient information retention with computational cost across various datasets.

Influence of subset selection. Importantly, the selection mechanism is not intrinsic to our framework. To assess alternative strategies, we replaced RDED’s selection mechanism with approaches such as Random Selection, K-means Clustering [12], and Herding [46]. A ResNet-18 model was trained on datasets distilled using these various selection strategies, with results summarized in Tab. 6. Notably, the results indicate that even with Random Selection, our framework achieves competitive performance. Additional results for the case where IPC = 1 are provided in Tab. 15 in Appendix.

Application: Continual learning. Following prior studies [49,56] that leverage synthetic datasets in continual learning to assess the quality of synthetic data, we employ the GDumb framework [27] for continual learning setup. In comparison to SRe<sup>2</sup>L, our study implements a 5-step class-incremental learning approach using ResNet-18 across CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets, with IPC = 10. The results of these experiments are depicted in Fig. 5. It is evident that our results substantially improve upon the baseline methods.

## 6 Conclusion

In this paper, we demonstrate that the primary limitation of current state-of-theart dataset distillation methods at various scales is the issue of huge information loss. To address this, we introduce the <sup>CIM</sup> technique, which efectively compresses critical data from images into distilled forms with minimal information loss. Our extensive experiments reveal that <sup>CIM</sup> markedly surpasses existing SOTA dataset distillation techniques across a range of dataset sizes and network architectures. Additionally, we highlight the eficiency of <sup>CIM</sup> by showcasing its ability to distill the ImageNet-1k dataset in just 80 minutes.

## Acknowledgment

This work was supported in part by the National Science and Technology Major Project (No. 2022ZD0115101), Research Center for Industries of the Future (RCIF) at Westlake University, Westlake Education Foundation, and Westlake University Center for High-performance Computing.

## References

1. Cazenavette, G., Wang, T., Torralba, A., Efros, A.A., Zhu, J.Y.: Dataset distillation by matching training trajectories. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4750–4759 (2022)

2. Cazenavette, G., Wang, T., Torralba, A., Efros, A.A., Zhu, J.Y.: Generalizing dataset distillation via deep generative prior. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3739–3748 (2023)

3. Chen, D., Kerkouche, R., Fritz, M.: Private set generation with discriminative information. Advances in Neural Information Processing Systems 35, 14678–14690 (2022)

4. Cui, J., Li, Z., Ma, X., Bi, X., Luo, Y., Shen, Z.: Dataset distillation via committee voting. arXiv preprint arXiv:2501.07575 (2025)

5. Cui, J., Wang, R., Si, S., Hsieh, C.J.: Dc-bench: Dataset condensation benchmark. Advances in Neural Information Processing Systems 35, 810–822 (2022)

6. Cui, J., Wang, R., Si, S., Hsieh, C.J.: Scaling up dataset distillation to imagenet-1k with constant memory. In: International Conference on Machine Learning. pp. 6565–6590. PMLR (2023)

7. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)

8. Dong, T., Zhao, B., Lyu, L.: Privacy for free: How does dataset condensation help privacy? In: International Conference on Machine Learning. pp. 5378–5396. PMLR (2022)

9. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale (2021)

10. Du, J., Hu, J., Huang, W., Zhou, J.T., et al.: Diversity-driven synthesis: Enhancing dataset distillation through directed weight adjustment. Advances in neural information processing systems 37, 119443–119465 (2024)

11. Du, J., Jiang, Y., Tan, V.Y., Zhou, J.T., Li, H.: Minimizing the accumulated trajectory error to improve dataset distillation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3749–3758 (2023)

12. Forgy, E.W.: Cluster analysis of multivariate data: eficiency versus interpretability of classifications. biometrics 21, 768–769 (1965)

13. Guo, Z., Wang, K., Cazenavette, G., Li, H., Zhang, K., You, Y.: Towards lossless dataset distillation via dificulty-aligned trajectory matching. arXiv preprint arXiv:2310.05773 (2023)

14. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

15. Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.Q.: Densely connected convolutional networks (2018)

16. Iofe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International conference on machine learning. pp. 448–456. pmlr (2015)

17. Kim, J.H., Kim, J., Oh, S.J., Yun, S., Song, H., Jeong, J., Ha, J.W., Song, H.O.: Dataset condensation via eficient synthetic-data parameterization. In: International Conference on Machine Learning. pp. 11102–11118. PMLR (2022)

18. Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features from tiny images (2009)

19. Krizhevsky, A., Nair, V., Hinton, G.: Cifar-10 and cifar-100 datasets. URl: https://www. cs. toronto. edu/kriz/cifar. html 6(1), 1 (2009)

20. Le, Y., Yang, X.: Tiny imagenet visual recognition challenge. CS 231N 7(7), 3 (2015)

21. Liu, H., Li, Y., Xing, T., Dalal, V., Li, L., He, J., Wang, H.: Dataset distillation via the wasserstein metric (2024)

22. Liu, H., Xing, T., Li, L., Dalal, V., He, J., Wang, H.: Dataset distillation via the wasserstein metric. arXiv preprint arXiv:2311.18531 (2023)

23. Liu, Y., Gu, J., Wang, K., Zhu, Z., Jiang, W., You, Y.: Dream: Eficient dataset distillation by representative matching. arXiv preprint arXiv:2302.14416 (2023)

24. Loo, N., Hasani, R., Amini, A., Rus, D.: Eficient dataset distillation using random feature approximation. Advances in Neural Information Processing Systems 35, 13877–13891 (2022)

25. Ma, N., Zhang, X., Zheng, H.T., Sun, J.: Shuflenet v2: Practical guidelines for eficient cnn architecture design (2018)

26. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of machine learning research 9(11) (2008)

27. Prabhu, A., Torr, P.H., Dokania, P.K.: Gdumb: A simple approach that questions our progress in continual learning. In: European conference on computer vision. pp. 524–540. Springer (2020)

28. Ridnik, T., Ben-Baruch, E., Noy, A., Zelnik-Manor, L.: Imagenet-21k pretraining for the masses. arXiv preprint arXiv:2104.10972 (2021)

29. Rosasco, A., Carta, A., Cossu, A., Lomonaco, V., Bacciu, D.: Distilled replay: Overcoming forgetting through synthetic samples. In: International Workshop on Continual Semi-Supervised Learning. pp. 104–117 (2021)

30. Sajedi, A., Khaki, S., Amjadian, E., Liu, L.Z., Lawryshyn, Y.A., Plataniotis, K.N.: Datadam: Eficient dataset distillation with attention matching. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17097–17107 (2023)

31. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4510–4520 (2018)

32. Shang, X., Lu, Y., Huang, G., Wang, H.: Federated learning on heterogeneous and long-tailed data via classifier re-training with federated features. arXiv preprint arXiv:2204.13399 (2022)

33. Shang, X., Sun, P., Lin, T.: Gift: Unlocking full potential of labels in distilled dataset at near-zero cost. In: International Conference on Learning Representations (2025)

34. Shao, S., Yin, Z., Zhou, M., Zhang, X., Shen, Z.: Generalized large-scale data condensation via various backbone and statistical matching. arXiv preprint arXiv:2311.17950 (2023)

35. Shao, S., Zhou, Z., Chen, H., Shen, Z.: Elucidating the design space of dataset condensation. In: Advances in neural information processing systems (2024)

36. Shen, Z., Sherif, A., Yin, Z., Shao, S.: Delt: A simple diversity-driven earlylate training for dataset distillation. CVPR (2025)

37. Shen, Z., Xing, E.: A fast knowledge distillation framework for visual recognition. In: European Conference on Computer Vision. pp. 673–690. Springer (2022)

38. Shin, D., Shin, S., Moon, I.c.: Frequency domain-based dataset distillation. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)

39. Such, F.P., Rawal, A., Lehman, J., Stanley, K., Clune, J.: Generative teaching networks: Accelerating neural architecture search by learning to generate synthetic training data. In: International Conference on Machine Learning. pp. 9206–9216 (2020)

40. Sun, P., Shi, B., Yu, D., Lin, T.: On the diversity and realism of distilled dataset: An eficient dataset distillation paradigm. arXiv preprint arXiv:2312.03526 (2023)

41. Sun, P., Shi, B., Yu, D., Lin, T.: On the diversity and realism of distilled dataset: An eficient dataset distillation paradigm. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

42. Tran, M.T., Le, T., Le, X.M., Do, T.T., Phung, D.: Enhancing dataset distillation via non-critical region refinement. CVPR (2025)

43. Wang, K., Zhao, B., Peng, X., Zhu, Z., Yang, S., Wang, S., Huang, G., Bilen, H., Wang, X., You, Y.: Cafe: Learning to condense dataset by aligning features. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12196–12205 (2022)

44. Wang, R., Cheng, M., Chen, X., Tang, X., Hsieh, C.J.: Rethinking architecture selection in diferentiable nas (2021)

45. Wang, T., Zhu, J.Y., Torralba, A., Efros, A.A.: Dataset distillation. arXiv preprint arXiv:1811.10959 (2018)

46. Welling, M.: Herding dynamical weights to learn. In: Proceedings of the 26th Annual International Conference on Machine Learning. pp. 1121–1128 (2009)

47. Xiong, Y., Wang, R., Cheng, M., Yu, F., Hsieh, C.J.: Feddm: Iterative distribution matching for communication-eficient federated learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16323– 16332 (2023)

48. Yin, Z., Shen, Z.: Dataset distillation in large data era. arXiv preprint arXiv:2311.18838 (2023)

49. Yin, Z., Xing, E., Shen, Z.: Squeeze, recover and relabel: Dataset condensation at imagenet scale from a new perspective. arXiv preprint arXiv:2306.13092 (2023)

50. Yu, R., Liu, S., Wang, X.: Dataset distillation: A comprehensive review. arXiv preprint arXiv:2301.07014 (2023)

51. Yu, R., Liu, S., Ye, J., Wang, X.: Teddy: Eficient large-scale dataset distillation via taylor-approximated matching. In: European Conference on Computer Vision. pp. 1–17. Springer (2024)

52. Yun, S., Oh, S.J., Heo, B., Han, D., Choe, J., Chun, S.: Re-labeling imagenet: from single to multi-labels, from global to localized labels. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2340–2350 (2021)

53. Zhang, L., Zhang, J., Lei, B., Mukherjee, S., Pan, X., Zhao, B., Ding, C., Li, Y., Xu, D.: Accelerating dataset distillation via model augmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11950–11959 (2023)

54. Zhang, X., Du, J., Liu, P., Zhou, J.T.: Breaking class barriers: Eficient dataset distillation via inter-class feature compensator. ICLR (2025)

55. Zhao, B., Bilen, H.: Dataset condensation with diferentiable siamese augmentation. In: International Conference on Machine Learning (2021)

56. Zhao, B., Bilen, H.: Dataset condensation with distribution matching. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 6514–6523 (2023)

57. Zhao, B., Mopuri, K.R., Bilen, H.: Dataset condensation with gradient matching. arXiv preprint arXiv:2006.05929 (2020)

58. Zhao, G., Li, G., Qin, Y., Yu, Y.: Improved distribution matching for dataset condensation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7856–7865 (2023)

59. Zhou, Y., Nezhadarya, E., Ba, J.: Dataset distillation using neural feature regression. Advances in Neural Information Processing Systems 35, 9813–9827 (2022)

## A Limitations

Although our <sup>CIM</sup> significantly outperforms existing SOTA methods, its primary limitation, as discussed in Section 5.2, is that it cannot surpass the optimizationfree paradigm (e.g., RDED [41]) in terms of eficiency. This constraint limits its applicability in diverse real-world scenarios.

## B Proof of Proposition 1

Proof. Define the optimal classification model $f _ { o p t } \colon$ Here, we aim to classify the two distributions $\mathcal { N } _ { 1 } ^ { s } ( \mu _ { 1 } + s _ { 1 } , \sigma ^ { 2 } )$ and $\mathcal { N } _ { 2 } ^ { s } ( \mu _ { 2 } + s _ { 2 } , \sigma ^ { 2 } )$ using a decision boundary. Given that both distributions have the same variance, the optimal decision boundary will be linear, and we can employ Bayesian decision theory. The decision boundary is given by the condition:

$$
P (\mathcal {N} _ {1} ^ {s} | x) = P (\mathcal {N} _ {2} ^ {s} | x)\tag{11}
$$

Applying Bayes’ rule, we can transform the above equation into a form of log likelihood ratio:

$$
\log \left(\frac {P (x | \mathcal {N} _ {1} ^ {s}) P (\mathcal {N} _ {1} ^ {s})}{P (x | \mathcal {N} _ {2} ^ {s}) P (\mathcal {N} _ {2} ^ {s})}\right) = 0\tag{12}
$$

Given

$$
P (x | \mathcal {N} _ {i} ^ {s}) = \frac {1}{\sqrt {2 \pi \sigma^ {2}}} e ^ {- \frac {(x - (\mu_ {i} + s _ {i})) ^ {2}}{2 \sigma^ {2}}},\tag{13}
$$

we can further simplify the equation. Note here we assume the prior probabilities $P ( \mathcal { N } _ { 1 } ^ { s } )$ ) and $P ( \mathcal { N } _ { 2 } ^ { s } )$ are equal. If they are not equal, we would need to consider these prior probabilities. For simplicity, we assume they are equal here. Thus, we can ignore the prior probabilities and focus on the likelihood ratio:

$$
\frac {(x - (\mu_ {1} + s _ {1})) ^ {2}}{2 \sigma^ {2}} = \frac {(x - (\mu_ {2} + s _ {2})) ^ {2}}{2 \sigma^ {2}}\tag{14}
$$

Expanding and simplifying the above equation, we can find the value of $x ,$ which will be the decision boundary for classification:

$$
x (\mu_ {1} + s _ {1} - \mu_ {2} - s _ {2}) = \frac {(\mu_ {1} + s _ {1}) ^ {2} - (\mu_ {2} + s _ {2}) ^ {2}}{2}\tag{15}
$$

$$
x = \frac {(\mu_ {1} + s _ {1}) ^ {2} - (\mu_ {2} + s _ {2}) ^ {2}}{2 (\mu_ {1} + s _ {1} - \mu_ {2} - s _ {2})}\tag{16}
$$

$$
x = \frac {(\mu_ {1} + s _ {1} + \mu_ {2} + s _ {2})}{2}\tag{17}
$$

This value of x defines the position of the linear decision boundary $f _ { o p t }$ . This result provides an explicit expression for the optimal linear decision boundary, allowing us to classify based on the means and variances of the two distributions.

## C Selecting Key Subsets from Original Dataset

Our goal is to devise an efective method for identifying the crucial samples that are instrumental in benefiting the <sup>Relabel</sup> process. Motivated by the Summary in Section 3.1 and Proposition 1—which suggests that each chosen sample pair $\left( \mathbf { x } , y \right)$ is supposed to receive an informative and accurate label $\phi _ { \pmb { \theta } _ { T } } ( \mathbf { x } )$ from the pre-trained model during the <sup>Relabel</sup> phase—the goal then relaxes to find the subsets of $\tau$ that include samples that are relabeled most accurately by the pre-trained model $\phi _ { \pmb { \theta } _ { T } }$ . Thus, we introduce a loss-based importance score s for each sample pair $\left( \mathbf { x } , y \right)$ , defined as $s = - \ell ( \phi _ { \pmb { \theta } _ { \tau } } ( \mathbf { x } ) , y )$ . The key sample selection procedure is elaborated below.

Let $\mathcal { T } _ { c } : = \{ ( \mathbf { x } , y ) ~ | ~ ( \mathbf { x } , y ) \in \mathcal { T } , y = c \}$ represents the subset of the dataset $\tau$ containing only those samples $\left( \mathbf { x } , y \right)$ that are labeled with the class c. For each class data $\mathcal { T } _ { c } .$ we identify the key samples x based on their importance scores $s ,$ forming IPC subsets of key samples as

$$
\begin{array}{c} \mathcal {S} _ {c} = \left\{\{\mathbf {x} _ {(j, i)}, y _ {(j, i)} \} _ {j = 1} ^ {N} \right\} _ {i = 1} ^ {\mathrm{IPC}} \\ \text {s.t.} s _ {(j, i)} = - \mathrm{CE} (\phi_ {\boldsymbol {\theta} _ {\mathcal {T}}} (\mathbf {x} _ {(j, i)}), y _ {(j, i)}) \geq \bar {s}, \end{array}\tag{18}
$$

where s¯ denotes a predetermined threshold<sup>5</sup>, N indicates the number of samples within each selected subset, and CE denotes the CrossEntropyLoss. We further simplify the whole procedure due to the computational overhead and diversity issue, as explained below

1. computing the scores s for all samples x in data $\mathcal { T } _ { c }$ presents a significant computational challenge;

2. focusing solely on samples that closely align with the true label can lead to a lack of diversity.

Therefore, we utilize a pre-selection strategy inspired by [41], which involves selecting a subse ${ } ^ { 6 } \ T _ { c } ^ { \prime } \subset \ T _ { c }$ uniformly at random to serve as a proxy for the entire $\mathcal { T } _ { c } .$ . Such a pre-selection strategy not only promotes diversity in the data but also lessens the computational load [41], thereby laying the groundwork for our subsequent score-based sample selection process.

## D Proof of Theorem 1

We initiate our derivation by examining the Kullback-Leibler divergence between the efective information distributions of $x _ { i }$ and ${ \hat { x } } _ { i }$ as observed by the group R:

$$
\begin{array}{l} \mathrm{D} _ {\mathrm{KL}} (\mathcal {P} _ {x _ {i} | \mathcal {R}} | | \mathcal {P} _ {\hat {x} _ {i} | \mathcal {R}}) = \mathbb {E} _ {z} \left[ \log \frac {\mathcal {P} _ {x _ {i} | \mathcal {R}} (z)}{\mathcal {P} _ {\hat {x} _ {i} | \mathcal {R}} (z)} \right] \\ \leq \mathbb {E} _ {z} \left[ | \log \mathcal {P} _ {x _ {i} | \mathcal {R}} (z) - \log \mathcal {P} _ {\hat {x} _ {i} | \mathcal {R}} (z) | \right]. \end{array} \tag {19}
$$

By applying kernel density estimation to $p ( \cdot )$ , we obtain:

$$
\mathbb {E} _ {z} \left[ | \log \mathcal {P} _ {x _ {i} | \mathcal {R}} (z) - \log \mathcal {P} _ {\hat {x} _ {i} | \mathcal {R}} (z) | \right]\tag{20}
$$

$$
= \mathbb {E} _ {\xi_ {j}} \left[ | \log \sum_ {\xi_ {k}} \left[ \frac {1}{(2 \pi) ^ {\frac {1}{d}}} \exp (\frac {- \| \xi_ {j} (x _ {i}) - \xi_ {k} (x _ {i}) \| ^ {2}}{2}) \right] \right.\tag{21}
$$

$$
- \log \sum_ {\xi_ {k}} \left[ \frac {1}{(2 \pi) ^ {\frac {1}{d}}} \exp (\frac {- \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {k} (\hat {x} _ {i}) \| ^ {2}}{2}) \right] | \Biggr ] ,\tag{22}
$$

where d denotes the dimension of x and a primary term exists within log(·). Therefore, we approximate:

$$
\approx \mathbb {E} _ {\xi_ {j}} \left[ \left| \sum_ {\xi_ {k}} \left[ \log \left(\frac {1}{(2 \pi) ^ {\frac {1}{d}}} \exp \left(\frac {- \| \xi_ {j} (x _ {i}) - \xi_ {k} (x _ {i}) \| ^ {2}}{2}\right)\right) \right] \right. \right.\tag{23}
$$

$$
- \left. \sum_ {\xi_ {k}} \left[ \log \left(\frac {1}{(2 \pi) ^ {\frac {1}{d}}} \exp \left(\frac {- \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {k} (\hat {x} _ {i}) \| ^ {2}}{2}\right)\right) \right] \right|\tag{24}
$$

$$
= \mathbb {E} _ {\xi_ {j}} \left[ \left| \sum_ {\xi_ {k}} \left[ \frac {- \| \xi_ {j} (x _ {i}) - \xi_ {k} (x _ {i}) \| ^ {2}}{2} \right] - \sum_ {\xi_ {k}} \left[ \frac {- \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {k} (\hat {x} _ {i}) \| ^ {2}}{2} \right] \right| \right]\tag{25}
$$

$$
= \mathbb {E} _ {\xi_ {j}} \left[ \left| \sum_ {\xi_ {k}} \left[ \frac {- \| \xi_ {j} (x _ {i}) - \xi_ {k} (x _ {i}) \| ^ {2}}{2} \right] - \sum_ {\xi_ {k}} \left[ \frac {- \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {k} (\hat {x} _ {i}) \| ^ {2}}{2} \right] \right| \right]\tag{26}
$$

$$
\leq \mathbb {E} _ {\xi_ {j}} \left[ \sum_ {\xi_ {k}} \left[ \left| \frac {- \| \xi_ {j} (x _ {i}) - \xi_ {k} (x _ {i}) \| ^ {2}}{2} - \frac {- \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {k} (\hat {x} _ {i}) \| ^ {2}}{2} \right| \right] \right]\tag{27}
$$

$$
\leq \mathbb {E} _ {\xi_ {j}} \left[ \sum_ {\xi_ {k}} \left[ \left| \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {k} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {j} (x _ {i}) - \xi_ {k} (x _ {i}) \| ^ {2} \right| \right] \right]\tag{28}
$$

$$
= \mathbb {E} _ {\xi_ {j}} \left[ \sum_ {\xi_ {k} \neq \xi_ {j}} \left[ \left| \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {k} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {j} (x _ {i}) - \xi_ {k} (x _ {i}) \| ^ {2} \right| \right] \right.\tag{29}
$$

$$
= \mathbb {E} _ {\xi_ {j}} \Bigg [ \sum_ {\xi_ {k} \neq \xi_ {j}} \Bigg (\big | \| \xi_ {j} (\hat {x} _ {i}) \| ^ {2} - 2 \langle \xi_ {j} (\hat {x} _ {i}), \xi_ {k} (\hat {x} _ {i}) \rangle + \| \xi_ {k} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {j} (x _ {i}) \| ^ {2}\tag{30}
$$

$$
\left. \left. + 2 \langle \xi_ {j} (x _ {i}), \xi_ {k} (x _ {i}) \rangle - \| \xi_ {k} (x _ {i}) \| ^ {2} \right|\right) \Bigg ]\tag{31}
$$

$$
\leq \left(| \mathcal {R} | - 1\right) \cdot \mathbb {E} _ {\xi_ {j}} \left[ \left| \| \xi_ {j} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {j} (x _ {i}) \| ^ {2} \right| \right] + \sum_ {\xi_ {k} \neq \xi_ {j}} \left[ \left| \| \xi_ {k} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {k} (x _ {i}) \| ^ {2} \right| \right]\tag{32}
$$

$$
+ \mathbb {E} _ {\xi_ {j}} \left[ \sum_ {\xi_ {k} \neq \xi_ {j}} [ | 2 \langle \xi_ {j} (x _ {i}), \xi_ {k} (x _ {i}) \rangle - 2 \langle \xi_ {j} (\hat {x} _ {i}), \xi_ {k} (\hat {x} _ {i}) \rangle | ] \right]\tag{33}
$$

$$
\approx (| \mathcal {R} | - 1) \cdot \mathbb {E} _ {\xi_ {j}} \left[ \left| \| \xi_ {j} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {j} (x _ {i}) \| ^ {2} \right| \right] + \sum_ {\xi_ {k} \neq \xi_ {j}} \left[ \left| \| \xi_ {k} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {k} (x _ {i}) \| ^ {2} \right| \right] + o (1)\tag{34}
$$

$$
\approx (| \mathcal {R} | - 1) \cdot \mathbb {E} _ {\xi_ {j}} \left[ \left| \| \xi_ {j} (\hat {x} _ {i}) \| ^ {2} - \| \xi_ {j} (x _ {i}) \| ^ {2} \right| \right]\tag{+λ}
$$

(35)

To minimize this term, we can use an alternative one:

$$
\mathbb {E} _ {\xi_ {j}} \left[ \| \xi_ {j} (\hat {x} _ {i}) - \xi_ {j} (x _ {i}) \| ^ {2} \right].\tag{36}
$$

## E Detailed Analysis for Existing Information Extraction-Based Approaches

Results without relabel. We evaluate the data with and without employing relabeling. The results are presented in Table 7. The visualizations of the images synthesized by state-of-the-art methods are presented in Section K.

Table 7: Testing was conducted with/without the application of relabeling technique. The experiment utilized the ResNet-18 model, with an IPC value set to 10.

<table><tr><td>Relabel</td><td colspan="4">Without Relabel</td><td colspan="4">With Relabel</td></tr><tr><td>Dataset</td><td>G-VBSM</td><td> $SRe^{2}L$ </td><td>RDED</td><td>Ours</td><td>G-VBSM</td><td> $SRe^{2}L$ </td><td>RDED</td><td>Ours</td></tr><tr><td>CIFAR-10</td><td>32.3 ± 0.7</td><td>10.9 ± 0.5</td><td>35.2 ± 0.6</td><td>51.6 ± 0.6</td><td>36.3 ± 0.7</td><td>39.4 ± 0.9</td><td>47.3 ± 0.5</td><td>66.2 ± 0.9</td></tr><tr><td>CIFAR-100</td><td>10.2 ± 0.2</td><td>1.2 ± 0.2</td><td>21.5 ± 0.3</td><td>40.7 ± 0.3</td><td>47.0 ± 0.4</td><td>42.7 ± 0.5</td><td>53.4 ± 0.3</td><td>61.7 ± 0.2</td></tr><tr><td>Tiny-ImageNet</td><td>0.6 ± 0.1</td><td>0.6 ± 0.0</td><td>14.7 ± 0.4</td><td>27.0 ± 0.5</td><td>37.7 ± 0.3</td><td>43.6 ± 0.5</td><td>48.4 ± 0.3</td><td>53.3 ± 0.1</td></tr><tr><td>ImageNet-1k</td><td>0.8 ± 0.1</td><td>1.1 ± 0.0</td><td>19.7 ± 0.3</td><td>22.0 ± 1.8</td><td>35.7 ± 0.2</td><td>31.1 ± 0.1</td><td>41.1 ± 0.2</td><td>48.7 ± 0.2</td></tr></table>

## F Detailed Framework of Our <sup>CIM</sup>

The structure of our <sup>CIM</sup> framework is outlined in Algorithm 1.

## G Experiment Details

Datasets. In addition to the datasets described in Section 5.1, we note that prevalent dataset distillation techniques struggle to scale to large, high-resolution datasets. In Table 8, we present some information about the dataset, including the number of classes, the number of images per class in the training set, and the test set.

Table 8: Details about the datasets

<table><tr><td>Dataset</td><td colspan="3">Num of Classes IPC of Training Set IPC of Test Set</td></tr><tr><td>CIFAR-10</td><td>10</td><td>5000</td><td>1000</td></tr><tr><td>CIFAR-100</td><td>100</td><td>500</td><td>100</td></tr><tr><td>Tiny-ImageNet</td><td>200</td><td>500</td><td>50</td></tr><tr><td>ImageNet-1k</td><td>1000</td><td>732 - 1300</td><td>50</td></tr></table>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 An efficient framework for dataset distillation.

Input: Original full dataset T, a corresponding pre-trained observer model  $\phi_{\theta_{T}}$  and initial S = 0.

Parameters: The number N for squeezing images, the number M of compression iterations, the size of  $T_{c}'$ .

for  $T_{c}' \subset T_{c} \subset T$  do

{Stage 1. Selecting Subsets of Key Samples}

for  $(\mathbf{x}_{i}, y_{i}) \in \mathcal{T}_{c}'$  do

Calculate  $s_{i} = -\ell(\phi_{\theta_{T}}(\mathbf{x}_{i}), y_{i})$ 

end for

Select top- $(N \times \text{IPC})$  images  $\{\mathbf{x}_{i}\}_{i=1}^{N \times \text{IPC}}$  via  $s_{i}$ 

{Stage 2. Compressing Effective Information}

for j = 1 to IPC do

Initialize distilled images as  $\widetilde{x}_{j}$ 

for m = 1 to M do

 $\Delta x \leftarrow \Delta x - \nabla_{\Delta x} L_{\Delta x}$ 

end for

{Stage 3. RELABEL}

Relabel  $\widetilde{x}_{j} \leftarrow \widetilde{x}_{j} + \Delta x^{\star}$  with  $\widetilde{y}_{j}$ $S = S \cup \{(\widetilde{x}_{j}, \widetilde{y}_{j})\}$ 

end for

end for

Output: Small distilled dataset S
</div>

Models. The experiment employed a multitude of pre-trained models, and we delineated their accuracies in Table 9. These results are furnished solely for reference purposes.

Baselines. We benchmark our proposed <sup>CIM</sup> against a range of SOTA distillation techniques capable of handling large, high-resolution datasets.

G-VBSM [34] surpasses one-sided methods like SRe<sup>2</sup>L by creating synthetic datasets with richer information and better generalization across various backbones, layers, and statistics.

SRe<sup>2</sup>L [49] is a novel entrant that eficiently handles ImageNet-1K, significantly outpacing other methods in managing large, high-resolution datasets and serving as our primary comparison point.

– RDED [41] enables the compression of large-scale, high-resolution datasets while maintaining diversity and realism, significantly reducing the time required for training neural networks like ResNet-18 on ImageNet-1K.

– DataDAM [30] eficiently distills images across multiple resolutions and scales by matching spatial attention maps between real and distilled samples at various layers within families of randomly initialized neural networks.

– ADD [53] demonstrates efective scalability across varying dataset resolutions, enhancing distillation speed through model augmentation.

Table 9: Accuracy of pre-trained models.

<table><tr><td>Dataset</td><td>Model</td><td>Size</td><td>Accuracy</td></tr><tr><td rowspan="2">CIFAR-10</td><td>ResNet-18</td><td>32 × 32</td><td>93.86</td></tr><tr><td>Conv-3</td><td>32 × 32</td><td>82.24</td></tr><tr><td rowspan="2">CIFAR-100</td><td>ResNet-18</td><td>32 × 32</td><td>72.27</td></tr><tr><td>Conv-3</td><td>32 × 32</td><td>61.27</td></tr><tr><td rowspan="2">Tiny-ImageNet</td><td>ResNet-18</td><td>64 × 64</td><td>61.98</td></tr><tr><td>Conv-4</td><td>64 × 64</td><td>49.73</td></tr><tr><td rowspan="2">ImageNet-1k</td><td>ResNet-18</td><td>224 × 224</td><td>69.31</td></tr><tr><td>Conv-4</td><td>64 × 64</td><td>43.6</td></tr></table>

– IDM [58] presents an eficient dataset condensation technique utilizing distribution matching, ofering a scalable alternative to computationally demanding optimization-focused methods [1, 57].

CDA [48] achieves superior accuracy on large-scale datasets like ImageNet-1K and 21K and significantly narrows the performance gap compared to full-data training counterparts.

– WMDD [21] presents a novel dataset distillation approach that employs the Wasserstein distance to enhance distribution matching, achieving state-ofthe-art performance by efectively capturing the essential representations of extensive datasets in synthetic forms.

– DATM [13] stands out by occasionally surpassing the training performance of the full original dataset, for instance, achieving IPC = 100 on CIFAR-100.

– DREAM [23] introduces an eficient technique while also delivering the most remarkable results.

– FreD [38] is a novel parameterization method for dataset distillation that operates in the frequency domain, significantly reducing the budget for synthesizing a small-sized synthetic dataset while preserving the original dataset’s information and consistently improving the performance of existing distillation methods.

Evaluating main results. For both dataset distillation and performance evaluation, we employ identical neural network architectures. Consistent with previous studies [1, 6, 58], we use Conv-3 for CIFAR-10 and CIFAR-100 distillation tasks and Conv-4 for Tiny-ImageNet (with the exception of DREAM, which utilizes Conv-3) and ImageNet-1K distillation. In line with [1, 6], MTT and TESLA apply a reduced resolution for distilling 224 × 224 images. According to [49], for retrieving and evaluating distilled datasets, $\mathrm { S R e ^ { 2 } L }$ and <sup>CIM</sup> adopt ResNet-18.

Evaluating the distilled dataset. We detail the hyperparameter configurations for distilling datasets in Table 10a. Consistent with recent works [34, 48, 49], the evaluation on the distilled dataset follows the parameters outlined in Table 10b. Furthermore, we implement Diferentiable Siamese Augmentation (DSA) as described by [55] to enhance images during both the distillation and evaluation phases of our experiments.

Table 10: Hyperparameter setting.  
(a) Data Synthesis.

<table><tr><td>Config</td><td>Value</td><td>Explanation</td></tr><tr><td>Iteration</td><td>200</td><td>NA</td></tr><tr><td>Optimizer</td><td>AdamW</td><td> $\beta_1, \beta_2=(0.9, 0.999)$ </td></tr><tr><td>Learning Rate</td><td>0.01</td><td>NA</td></tr><tr><td>Initialization</td><td>RDED</td><td>Initialized using images from training dataset</td></tr><tr><td>Factor</td><td>2</td><td>NA</td></tr><tr><td>Mipc</td><td>300</td><td>NA</td></tr><tr><td>Depth</td><td>Deep/Mid</td><td>Deep for ConvNet and ResNet(modified), Mid for ResNet</td></tr><tr><td colspan="3">(b) Evaluation.</td></tr><tr><td>Config</td><td>Value</td><td>Explanation</td></tr><tr><td>Epochs</td><td>300/1000</td><td>300 for ImageNet-1k, 1000 for default</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>NA</td></tr><tr><td>Learning Rate</td><td>0.001</td><td>NA</td></tr><tr><td>Batch Size</td><td>10/50/100/200</td><td>10 for 0 &lt; Num of Images ≤ 10, 50 for 10 &lt; Num of Images ≤ 500, 100 for 500 &lt; Num of Images ≤ 20000, 200 for 20000 &lt; Num of Images</td></tr><tr><td>Scheduler</td><td>MultiStepLR</td><td>milestones=[2 × epochs // 3, 5 × epochs // 6] gamma=0.2</td></tr><tr><td>Augmentation</td><td>DSA strategy</td><td>color, crop, cutout, flip, scale, rotate</td></tr></table>

Diferentiable Siamese Augmentation (DSA). We employ DSA (Diferentiable Siamese Augmentation) as a method for image augmentation. To enhance clarity, we outline the DSA operations utilized in Table 11, along with their corresponding transformations and probabilities.

Factor Crop. We have integrated a cropping method termed ’factor crop’, which is applied prior to DSA. This technique enhances sample diversity by precisely extracting regions from specific areas of the images and resizing them to their original dimensions. As an augmentation method, it bolsters the model’s generalizability, as depicted in Figure 6. This approach functions as a substitute for RandomCrop, thereby preserving the semantic integrity of compressed images within the distilled dataset.

Table 11: Diferentiable Siamese Augmentation(DSA) and ratios

<table><tr><td>DSA</td><td>Transform</td><td>Ratio</td></tr><tr><td>Color</td><td>Color Jitter</td><td>Brightness=1.0Saturation=2.0Contrast=0.5</td></tr><tr><td>Crop</td><td>Random Crop</td><td>Crop Pad=0.125</td></tr><tr><td>Cutout</td><td>Random Cutout</td><td>Cutout=0.5</td></tr><tr><td>Flip</td><td>Random Horizontal Flip</td><td>Flip=0.5</td></tr><tr><td>Scale</td><td>Random Scale</td><td>Scale=1.2</td></tr><tr><td>Rotate</td><td>Random Rotation</td><td>Rotate=15.0</td></tr></table>

![](images/ae94c5cc1469de0a1b6ed2826a96b41a5bd42848c2d896c042128bf3aed122cc.jpg)  
Fig. 6: Visualization of factor crop.

## H Experiment Results

Comparison with more datasets and baselines. In addition to the experiments discussed in Section 5.2, we further benchmark our proposed <sup>CIM</sup> against a broader set of baselines, encompassing recent contributions [22, 34, 40, 50]. The outcomes, presented in Table 12 and Table 13, consistently afirm the superior performance of <sup>CIM</sup> in dataset distillation tasks.

Table 12: Comparison with more SOTA dataset distillation methods.

<table><tr><td colspan="2">Architecture</td><td colspan="5">ConvNet</td><td colspan="5">ResNet-18</td></tr><tr><td>Dataset</td><td>IPC</td><td>DataDAM</td><td>ADD</td><td>IDM</td><td>RDED</td><td>CIM (Ours)</td><td>G-VBSM</td><td>CDA</td><td>WMDD</td><td>RDED</td><td>CIM (Ours)</td></tr><tr><td rowspan="3">CIFAR-10</td><td>1</td><td> $32.0 \pm 1.2$ </td><td>49.2</td><td> $45.6 \pm 0.7$ </td><td> $23.5 \pm 0.3$ </td><td> $\mathbf{37.4} \pm \mathbf{0.4}$ </td><td>-</td><td>-</td><td>-</td><td> $22.9 \pm 0.4$ </td><td> $\mathbf{32.3} \pm \mathbf{0.2}$ </td></tr><tr><td>10</td><td> $54.2 \pm 0.8$ </td><td>67.1</td><td> $58.6 \pm 0.1$ </td><td> $50.2 \pm 0.3$ </td><td> $\mathbf{61.4} \pm \mathbf{0.4}$ </td><td> $53.5 \pm 0.6$ </td><td>-</td><td>-</td><td> $37.1 \pm 0.3$ </td><td> $\mathbf{66.2} \pm \mathbf{0.9}$ </td></tr><tr><td>50</td><td> $67.0 \pm 0.4$ </td><td>73.8</td><td> $67.5 \pm 0.1$ </td><td> $68.4 \pm 0.1$ </td><td> $\mathbf{74.7} \pm \mathbf{0.2}$ </td><td> $59.2 \pm 0.4$ </td><td>-</td><td>-</td><td> $62.1 \pm 0.1$ </td><td> $\mathbf{85.1} \pm \mathbf{0.3}$ </td></tr><tr><td rowspan="3">CIFAR-100</td><td>1</td><td> $14.5 \pm 0.5$ </td><td>29.8</td><td> $20.1 \pm 0.3$ </td><td> $19.6 \pm 0.3$ </td><td> $\mathbf{27.4} \pm \mathbf{0.4}$ </td><td> $25.9 \pm 0.5$ </td><td>-</td><td>-</td><td> $11.0 \pm 0.3$ </td><td> $\mathbf{31.1} \pm \mathbf{0.7}$ </td></tr><tr><td>10</td><td> $34.8 \pm 0.5$ </td><td>45.6</td><td> $45.1 \pm 0.1$ </td><td> $48.1 \pm 0.3$ </td><td> $\mathbf{49.3} \pm \mathbf{0.3}$ </td><td> $59.5 \pm 0.4$ </td><td>-</td><td>-</td><td> $42.6 \pm 0.2$ </td><td> $\mathbf{61.7} \pm \mathbf{0.2}$ </td></tr><tr><td>50</td><td> $49.4 \pm 0.3$ </td><td>52.6</td><td> $50.0 \pm 0.2$ </td><td> $\mathbf{57.0} \pm \mathbf{0.1}$ </td><td> $55.1 \pm 0.2$ </td><td> $65.0 \pm 0.5$ </td><td>-</td><td>-</td><td> $62.6 \pm 0.1$ </td><td> $\mathbf{67.0} \pm \mathbf{0.1}$ </td></tr><tr><td rowspan="3">Tiny ImageNet</td><td>1</td><td> $8.3 \pm 0.4$ </td><td>-</td><td> $10.1 \pm 0.2$ </td><td> $12.0 \pm 0.1$ </td><td> $\mathbf{25.4} \pm \mathbf{0.2}$ </td><td>-</td><td>-</td><td> $7.6 \pm 0.2$ </td><td> $9.7 \pm 0.4$ </td><td> $\mathbf{25.1} \pm \mathbf{0.4}$ </td></tr><tr><td>10</td><td> $18.7 \pm 0.3$ </td><td>-</td><td> $21.9 \pm 0.2$ </td><td> $39.6 \pm 0.1$ </td><td> $\mathbf{42.2} \pm \mathbf{0.2}$ </td><td>-</td><td>-</td><td> $41.8 \pm 0.1$ </td><td> $41.9 \pm 0.2$ </td><td> $\mathbf{53.3} \pm \mathbf{0.1}$ </td></tr><tr><td>50</td><td> $28.7 \pm 0.3$ </td><td>-</td><td> $27.7 \pm 0.3$ </td><td> $\mathbf{47.6} \pm \mathbf{0.2}$ </td><td> $47.0 \pm 0.2$ </td><td>-</td><td> $48.7$ </td><td> $\mathbf{59.4} \pm \mathbf{0.5}$ </td><td> $58.2 \pm 0.1$ </td><td> $58.6 \pm 0.1$ </td></tr><tr><td rowspan="3">ImageNet-1k</td><td>1</td><td> $2.0 \pm 0.1$ </td><td>-</td><td>-</td><td>-</td><td> $\mathbf{5.8} \pm \mathbf{0.1}$ </td><td>-</td><td>-</td><td> $3.2 \pm 0.3$ </td><td> $6.6 \pm 0.2$ </td><td> $\mathbf{7.3} \pm \mathbf{0.3}$ </td></tr><tr><td>10</td><td> $6.3 \pm 0.0$ </td><td>-</td><td>-</td><td>-</td><td> $\mathbf{24.5} \pm \mathbf{0.1}$ </td><td> $31.4 \pm 0.5$ </td><td>-</td><td> $38.2 \pm 0.2$ </td><td> $42.0 \pm 0.1$ </td><td> $\mathbf{48.7} \pm \mathbf{0.2}$ </td></tr><tr><td>50</td><td> $15.5 \pm 0.2$ </td><td>-</td><td>-</td><td>-</td><td> $\mathbf{37.8} \pm \mathbf{0.1}$ </td><td> $51.8 \pm 0.4$ </td><td> $53.5$ </td><td> $57.6 \pm 0.5$ </td><td> $56.5 \pm 0.1$ </td><td> $\mathbf{60.4} \pm \mathbf{0.1}$ </td></tr></table>

Table 13: Comparison with other SOTA dataset distillation methods that we reproduced.

<table><tr><td colspan="2">Architecture</td><td colspan="4">ConvNet</td></tr><tr><td>Dataset</td><td>IPC</td><td>DATM</td><td>DREAM</td><td>FreD</td><td>CIM (Ours)</td></tr><tr><td rowspan="3">CIFAR-10</td><td>1</td><td></td><td> $40.8 \pm 1.2$ </td><td> $33.2 \pm 1.2$ </td><td> $\mathbf{50.0} \pm \mathbf{0.3}$ </td></tr><tr><td>10</td><td> $60.1 \pm 0.3$ </td><td> $64.2 \pm 0.1$ </td><td> $46.6 \pm 0.6$ </td><td> $\mathbf{72.2} \pm \mathbf{0.2}$ </td></tr><tr><td>50</td><td> $63.1 \pm 1.0$ </td><td> $72.2 \pm 0.1$ </td><td> $46.3 \pm 0.4$ </td><td> $\mathbf{78.3} \pm \mathbf{0.1}$ </td></tr><tr><td rowspan="3">CIFAR-100</td><td>1</td><td>-</td><td> $22.4 \pm 0.4$ </td><td> $15.5 \pm 0.2$ </td><td> $\mathbf{42.7} \pm \mathbf{0.3}$ </td></tr><tr><td>10</td><td> $27.7 \pm 0.3$ </td><td> $41.3 \pm 0.6$ </td><td>-</td><td> $\mathbf{54.8} \pm \mathbf{0.2}$ </td></tr><tr><td>50</td><td> $42.8 \pm 0.2$ </td><td> $48.3 \pm 0.2$ </td><td>-</td><td> $\mathbf{56.6} \pm \mathbf{0.1}$ </td></tr><tr><td rowspan="3">Tiny ImageNet</td><td>1</td><td>-</td><td>-</td><td> $5.8 \pm 0.2$ </td><td> $\mathbf{31.7} \pm \mathbf{0.4}$ </td></tr><tr><td>10</td><td> $28.8 \pm 0.2$ </td><td>-</td><td>-</td><td> $\mathbf{46.3} \pm \mathbf{0.2}$ </td></tr><tr><td>50</td><td> $36.1 \pm 0.1$ </td><td>-</td><td>-</td><td> $\mathbf{47.4} \pm \mathbf{0.1}$ </td></tr><tr><td rowspan="3">ImageNet-1k</td><td>1</td><td>-</td><td>-</td><td>-</td><td> $\mathbf{14.5} \pm \mathbf{0.3}$ </td></tr><tr><td>10</td><td>-</td><td>-</td><td>-</td><td> $\mathbf{24.0} \pm \mathbf{0.5}$ </td></tr><tr><td>50</td><td>-</td><td>-</td><td>-</td><td> $\mathbf{37.3} \pm \mathbf{0.1}$ </td></tr></table>

Table 14: Eficiency comparison with SOTA methods with Conv-4 on Tiny-ImageNet.

<table><tr><td rowspan="2">Architecture</td><td colspan="3">Time Cost (s)</td><td colspan="3">Peak Memory (GB)</td></tr><tr><td>DREAM</td><td>DATM</td><td>Ours</td><td>DREAM</td><td>DATM</td><td>Ours</td></tr><tr><td>Conv-4</td><td>33906.17</td><td>12470.90</td><td>13.02</td><td>15.92</td><td>20.16</td><td>0.65</td></tr></table>

## H.1 Eficiency Comparison

Beyond assessing performance in Section 5.2, we expand our evaluation to include additional baselines. Results presented in Table 14 underscore the exceptional eficiency of our proposed <sup>CIM</sup>, which also requires the least GPU memory.

## I Ablation

Selection. To compare our selection strategy with others, namely Random, Kmeans [12], and Herding [46], we trained ResNet-18 using datasets distilled through various strategies, as delineated in Tables 15.

## J Continual Learning

In comparison to $\mathrm { S R e ^ { 2 } L }$ , our study implements a five-step class-incremental learning approach using ResNet-18 across CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets, each with an IPC setting of 10. The results of these experiments are depicted in Figures 7, 8, and 9 for CIFAR-10, CIFAR-100, and Tiny-ImageNet, respectively.

Table 15: Comparision with other selection strategies, with IPC = 1

<table><tr><td></td><td>Random</td><td>K-means</td><td>Herding</td><td>Ours</td></tr><tr><td>CIFAR-10</td><td>28.3 ± 0.4</td><td>30.3 ± 0.4</td><td>31.0 ± 0.3</td><td>31.3 ± 0.5</td></tr><tr><td>CIFAR-100</td><td>25.1 ± 0.0</td><td>25.5 ± 0.3</td><td>26.9 ± 0.2</td><td>30.6 ± 0.1</td></tr><tr><td>Tiny-ImageNet</td><td>19.5 ± 0.5</td><td>19.6 ± 1.0</td><td>19.5 ± 0.2</td><td>25.5 ± 0.0</td></tr><tr><td>ImageNet-1k</td><td>5.0 ± 0.0</td><td>5.6 ± 0.1</td><td>5.6 ± 0.1</td><td>7.1 ± 0.2</td></tr></table>

![](images/78d48233f113059ca99b124048615fb83f644597af3cd650a7f9895fcf98f01d.jpg)  
Fig. 7: Visualization of continual learning on CIFAR-10 with $\mathtt { I P C } = 1 0$

## K Visualization

Baselines. Within the scope of CIFAR-10 distillation under the IPC = 10 setting, we illustrate the visual representations of distilled datasets. This includes visualizations for ADD [53] in Figure 12, DataDAM [30] in Figure 13, $\mathrm { S R e ^ { 2 } L }$ [49] in Figure 15, and DREAM [23] in Figure 14. Distilled images of each method are generated starting from actual images, showcased in Figures 10 and 11.

A simple squeezing-based method. The image-squeezing process entails resizing and concatenating images to facilitate dataset distillation. For example, consider the manipulation of 4 images, each originally sized at 224 × 224 pixels. The initial step involves downsizing each image to 112 × 112 pixels. Subsequently, these reduced images are merged into a single composite image, efectively reverting to the original resolution of 224 × 224 pixels. This approach underpins a simplistic, squeezing-based dataset distillation method, whereby N randomly selected original images are compressed into one distilled image to compose a condensed dataset. In the context of distilling CIFAR-10 with an IPC = 10 configuration, we exhibit the visual outcomes of this process for diverse settings: N = 1 in Figure 16, N = 4 in Figure 17, N = 9 in Figure 18, N = 16 in Figure 19, and N = 25 in Figure 20.

Our proposed <sup>CIM</sup>. With an IPC setting of 10, we illustrate the distilled datasets generated by our proposed <sup>CIM</sup>. These include visualizations for CIFAR-10 in Figure 21, CIFAR-100 in Figure 22, Tiny-ImageNet in Figure 23, and ImageNet-1k in Figure 24.

![](images/2eb341467bb5b1bb307bf33a4c29ba1b5714398aa68e5249d7efbe1b77906cc8.jpg)  
Fig. 8: Visualization of continual learning on CIFAR-100 with $\mathtt { I P C } = 1 0$

![](images/970b1e121a85f90d2b15195b36d7066e5ca65a3fed6907007de8ff1d23f062bf.jpg)  
Fig. 9: Visualization of continual learning on Tiny-ImageNet with $\mathtt { I P C } = 1 0$

![](images/c6efe8487596753985cd17d565f25c364d18ccb07cede0fad83413817a8b1fae.jpg)  
Fig. 10: Visualization of initialized images before distilling on CIFAR-10.

![](images/6c773a6a925edcf4fa2a356f58160de3099f6155f4a3d768412316ad9c152c28.jpg)  
Fig. 11: Visualization of initialized data before distilling on CIFAR-10 data showcases the mixture of 4 images per initial instance.

![](images/48fff1bc247b2a7c8cc0dcbc8e04556e53e256f7230be535c26f8d5b6bc2b983.jpg)  
Fig. 12: Synthetic data visualization on CIFAR-10 from ADD [53].

![](images/9d012a7b88090d422bef859560fa64459a17935207b8ee4d0d9b48508de65452.jpg)  
Fig. 13: Synthetic data visualization on CIFAR-10 from DataDAM [30].

![](images/c0c0277b1e53918a21dd70ba0b566df49d1513e6e4a349df00a59e2388e8df6a.jpg)  
Fig. 14: Synthetic data visualization on CIFAR-10 from DREAM [23].

![](images/e2b4a31f157354a9f5e61f0947303b3c67884880bcc0a428263bf4ba4e51d593.jpg)  
Fig. 15: Synthetic data visualization on CIFAR-10 from $\mathrm { S R e ^ { 2 } L }$ [49].

![](images/fba9cbe68c67b1346d83e23017a341b83aa53bc6500ea04fc0079439608fa7d5.jpg)  
Fig. 16: Initialized real data visualization on CIFAR-10 with N = 1.

![](images/64e69352c8a0bc4560e38b4cdfb5b56c3dcb4972c465bc5129d5db0d522500ec.jpg)  
Fig. 17: Initialized real data visualization on CIFAR-10 with N = 4.

![](images/8b3fbb490a10f12e72776e254c2a25139e31937d59be2c2383b1c37ab89de63a.jpg)  
Fig. 18: Initialized real data visualization on CIFAR-10 with N = 9.

![](images/77d4c5ca5b4e6064f74295eb47dfe1bfd766bb6f2dc8a370c6922480abdd4246.jpg)  
Fig. 19: Initialized real data visualization on CIFAR-10 with N = 16.

![](images/b7918e49e89b3da49055439b864c1fde2e8bb6c807261be640d9a9bec89d5e81.jpg)  
Fig. 20: Initialized real data visualization on CIFAR-10 with N = 25.

![](images/02b0ab1214157280a1c841189ece5520d96ef35f2f7e6c4888ad3b641160c19a.jpg)  
Fig. 21: Synthetic data visualization on CIFAR-10 from <sup>CIM</sup> (Ours).

![](images/80ac0ab55b0278f4ba701f3cf7df3a038bc7a52ac1c1cd0fdf7ebada8c627303.jpg)  
Fig. 22: Synthetic data visualization on CIFAR-100 from <sup>CIM</sup> (Ours).

![](images/7de30f1fdd7b0eab81fa22c0301941009f449697d863a4e05649df62a81c7304.jpg)  
Fig. 24: Synthetic data visualization on ImageNet-1k from <sup>CIM</sup> (Ours).

![](images/2466ab87f6e27e0f2adee4058d3a5da8d0df3973507b5b16cc38cee86ebf564f.jpg)  
Fig. 23: Synthetic data visualization on Tiny-ImageNet from <sup>CIM</sup> (Ours).