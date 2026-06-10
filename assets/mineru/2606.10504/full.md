# Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundation and Algorithm

Trong Khiem Tran \* 1 2 Anh Duc Chu \* 2 Quang Hung Pham 2 Phi Le Nguyen 2 Trong Nghia Hoang 1

## Abstract

Cross-modal knowledge distillation (CMKD) studies how a (large) teacher model trained on one type of data (e.g., images) can guide a (smaller) student model building on another type of data (e.g., text/audio). Existing CMKD methods often require paired multi-modal data with aligned semantics, but obtaining such paired data are often costly and impractical. To mitigate this limitation, we develop a new CMKD framework for the more challenging setting where paired data are unavailable. In particular, we establish a crossmodal distributional relationship between teacher and student models which reveals two fundamental quantities governing effective distillation: feature alignment and label alignment. These quantities characterize semantic discrepancy between modalities at the levels of representation and prediction distributions, respectively. Motivated by this insight, we propose a principled framework, with theoretical guarantees, that enables effective cross-modal knowledge distillation by aligning distributions rather than individual samples. Extensive experiments across a wide range of multimodal benchmarks show that our framework is highly effective in both unpaired and paired data settings, improving significantly over prior work.

## 1. Introduction

Knowledge distillation (KD) was first introduced by (Bucilua et al., 2006) and (Hinton et al., 2015) as a mechanism for transferring predictive knowledge from a complex

1School of Electrical Engineering and Computer Science, Washington State University, Pullman, US 2School of Information and Communications Technology, Hanoi University of Science and Technology, Hanoi, Vietnam. Correspondence to: Trong Nghia Hoang <trongnghia.hoang@wsu.edu>, Trong Khiem Tran <khiem.tran@wsu.edu>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

teacher model to a simpler student model. The key motivation is that highly over-parameterized models are often beneficial during training, as the additional capacity provides flexibility for making mistakes and correcting them. The learned knowledge however can often be compressed into a significantly smaller model with minimal loss in predictive performance, making deployment much more efficient. This can be achieved by making the student model mimic the behavior of the teacher model via matching soft predictions.

In classical in-modal settings, both the teacher and student models operate on the same input representation and data modalities. An existing rich literature has advanced in-modal knowledge distillation by improving how knowledge is transferred from teacher to student models. These include contrastive losses for representation alignment (Tian et al., 2020), variance reduction via Bayes-optimal teachers (Menon et al., 2021), gradient-aware adaptive distillation (Zhu & Wang, 2021), decoupled treatment of target and non-target classes (Zhao et al., 2022), multi-level feature review mechanisms (Chen et al., 2021), and relation-based losses capturing inter- and intra-class structure (Huang et al., 2022; Yang et al., 2023). A more comprehensive review of classical in-modal KD methods is provided in Section 5.1.

In contrast, cross-modal knowledge distillation (CMKD) considers more challenging and practical settings where teacher and student models operate on different data modalities, such as transferring knowledge from image-based models to text- or audio-based models. In its distillation process, the teacher and student inputs correspond to different modalities of the same underlying instance, which typically requires paired multimodal data during training. Recent advances have demonstrated the broad applicability of CMKD across diverse domains, including selective bidirectional distillation for bridging modality gaps (C2KD) (Huo et al., 2024), self-supervised representation learning from unlabeled videos (Sarkar & Etemad, 2022), vision–language self-distillation via cross-attention mechanisms (COSMOS) (Kim et al., 2024), and weakly paired transfer between microscopy images and transcriptomics data (XKD) (Bendidi et al., 2025). Despite this progress, existing CMKD methods remain largely dependent on paired or weakly paired multimodal data with sample-level correspondence, which is often costly, difficult to obtain, or impractical in real-world settings where modalities are collected independently or asynchronously. This raises the following fundamental challenge:

## How can we distill knowledge from the teacher model to the student without using sample-level pairing ?

A natural extension to the unpaired setting is to adapt featurebased knowledge distillation methods by replacing samplelevel feature matching with distribution-level feature alignment. However, prior work (Huo et al., 2024) has shown that direct adoption of conventional feature-based distillation methods, such as FitNet (Romero et al., 2014) and ReviewKD (Chen et al., 2021), to cross-modal settings remains ineffective even when paired data are available. This limitation arises because effective cross-modal transfer requires aligning not only representation structures but also predictive semantics across modalities. In the unpaired setting, where semantic sample-level correspondence is entirely absent, this challenge becomes even more severe, as further confirmed by our experimental results (Section 4.2).

To address this challenge, we introduce a principled framework that establishes a provable bound on the generalized student error via two fundamental quantities: (i) feature alignment and (ii) label alignment. These quantities characterize semantic discrepancy across modalities at the levels of representation and prediction distributions via measuring the cross-modal gap between the teacher model and the student, given a shared representation space. Larger cross-modal discrepancies make effective knowledge transfer substantially more difficult and can lead to suboptimal distillation. This characterization provides actionable guidance for minimizing cross-modal discrepancy, leading to a theoretically grounded framework for algorithm design. Our key contributions are summarized as follows:

1. Theoretical Analysis. We develop a theoretical bound that decomposes the distilled student model’s generalized error into three components: (i) the teacher’s generalized error, which acts as a fixed overhead; (ii) the representation distributional discrepancy between the teacher and student models – feature alignment; and (iii) the prediction distributional discrepancy between the teacher and student models – label alignment. To the best of our knowledge, this is the first generalization bound for CMKD that reveals a synergistic influence of teacher quality, feature alignment, and label alignment on the student’s generalization (Section 2).

2. Algorithm Design. Motivated by our theoretical analysis, we develop a practical framework for cross-modal knowledge distillation that focuses on distribution-level alignment which does not require training examples of sample-level pairing. In particular, the developed method enables effective knowledge transfer via minimizing both feature and label alignment. To achieve this, we construct an optimizable surrogate that serves as a medium for selectively distilling teacher knowledge relevant to the student’s knowledge. This surrogate is optimized in a bi-level optimization manner by optimizing the student’s objective with calibration to feature alignment and label alignment (Section 3).

3. Empirical Evaluation. We evaluate our approach on four multimodal benchmarks: AVE (Tian et al., 2018) for event localization; CREMA-D (Cao et al., 2014) and RAVDESS (Livingstone & Russo, 2018) for emotion recognition; and VGGSound (Chen et al., 2020), a large-scale benchmark containing over 200,000 videos spanning more than 300 classes. Across all datasets, our method consistently outperforms recent state-of-the-art (SOTA) baselines in both unpaired and paired CMKD settings, demonstrating its effectiveness and robustness (Section 4).

For interested readers, we also provide a comprehensive literature review of existing KD methods in both in-modal and cross-modal settings in Section 5.

## 2. Theoretical Analysis

We begin by formalizing the problem setting and introducing key notations (Section 2.1). We then establish generalization bounds for the student model under both infinite-data (Section 2.2) and finite-data (Section 2.3) settings.

## 2.1. Problem Setting and Notations

Let $M _ { T } \ \triangleq \ ( \theta , p _ { T } ( y \mid z = \theta ( \mathbf { x } ^ { T } ) ) )$ denote the teacher model which comprises a feature extractor $\theta ( { \pmb x } ^ { T } )$ and a prediction head $p _ { T } ( y \mid z = \theta ( \mathbf { x } ^ { T } ) )$ ). The teacher model is $D ^ { T } = \stackrel { \cdot } { ( } x _ { i } ^ { T } , y _ { i } \stackrel { _ { T } } { ) _ { i = 1 } ^ { n _ { T } } } \sim ( \mathcal { X } ^ { T } \times \mathcal { Y } )$

Our goal is to distill relevant knowledge from $M _ { T }$ into a student model $M _ { S } \triangleq ( \phi , p _ { S } ( y \mid \phi ( \pmb { x } ^ { S } ) ) )$ with feature extractor $\phi ( \pmb { x } ^ { S } )$ and solution head $p _ { S } ,$ so that it can gener-$\left( \boldsymbol { x } _ { i } ^ { S } , \boldsymbol { y } _ { i } \right) _ { i = 1 } ^ { n _ { S } } \sim D ^ { \tilde { S } }$ $( \pmb { x } _ { i } ^ { S } , \pmb { y } _ { i } ) \in \pmb { \chi } ^ { S } \times \pmb { \ y }$ to unseen data sampled from $( \mathcal { X } ^ { S } , \mathcal { Y } )$ .

We assume the training input to the student model is sampled from new data modalities $\ d \chi ^ { S } \ne \ d \chi ^ { T }$ which were not previously seen by the teacher model during its pretraining. To facilitate representation alignment during distillation, we configure the teacher and student feature extractors, $\theta : \mathcal { X } ^ { T }  \mathcal { Z }$ and $\phi : \mathcal { X } ^ { S }  \mathcal { Z }$ , to map their respective inputs into a shared embedding space $\mathcal { Z }$ .

Definition 2.1 (Generalized Error). We have the generalized error for the teacher model and the student model under the feature maps ϕ and θ as:

$$
\operatorname{err} _ {T} (\theta) \triangleq \mathbb {E} _ {D ^ {T} \left(\boldsymbol {x} ^ {T}, y\right)} \left[ - \log p _ {T} \left(y \mid \theta \left(\boldsymbol {x} ^ {T}\right)\right) \right], \tag {1}
$$

$$
\operatorname{err} _ {S} (\phi) \triangleq \mathbb {E} _ {D ^ {S} \left(\boldsymbol {x} ^ {S}, y\right)} \left[ - \log p _ {S} \left(y \mid \phi \left(\boldsymbol {x} ^ {S}\right)\right) \right], \tag {2}
$$

which are the expected (generalized) teacher and student prediction losses over the corresponding data distributions.

Definition 2.2 (Feature Distribution). The feature distributions $D ^ { S } ( z )$ and $D ^ { T } ( z )$ denote the push-forward distributions induced by the student and teacher feature maps, $z = \phi ( \pmb { x } ^ { S } )$ and $z = \theta ( \pmb { x } ^ { T } )$ , on the marginal input distributions $\dot { D } ^ { S } ( \pmb { x } ^ { S } )$ and $D ^ { T } ( \pmb { x } ^ { T } )$ , respectively.

Let $D ^ { S } ( y \mid z )$ and $D ^ { T } ( y \mid z )$ denote the student and teacher feature-label conditional distributions induced by the feature maps $z = \phi ( { \pmb x } ^ { S } )$ and $z = \theta ( \pmb { x } ^ { T } )$ ), respectively.

## 2.2. Asymptotic Performance Bound

We will now provide the generalized performance bound for cross-modal distillation in the asymptotic regime where key quantities (see below) in the bound are computed with respect to the true data distributions. This is equivalent to assuming an infinite amount of data, such that empirical estimates coincide with their population counterparts.

Our main result characterizes the generalized student loss $\mathrm { e r r } _ { S } ( \phi )$ in terms of the following key quantities:

Overhead. The generalized teacher loss $\mathrm { e r r } _ { T } ( \theta )$ .

Feature Alignment (FA). A function of the distributional distance between the student and teacher representation distributions, $D ^ { S } ( z )$ and $D ^ { T } ( z )$ , induced by ϕ and θ.

Label Alignment (LA). A function of the distributional distance between the student and teacher predictions, which captures the semantic gap between the student predictor $p _ { S } ( y \mid z )$ and the teacher predictor $p _ { T } ( y \mid z )$ .

An informal statement of our result is stated below.

Theorem 2.3 (Informal Statement). Given teacher and student feature maps θ and ϕ, we have:

$$
\begin{array}{l} \operatorname{err} _ {S} (\phi) \leq \operatorname{err} _ {T} (\theta) \\ + \text {   Feature   Alignment   } + \text {   Label   Alignment   }. \tag {3} \\ \end{array}
$$

This result reveals how the generalization quality of the teacher model and the interplay between feature alignment and label alignment during distillation will influence the generalized student performance. The teacher generalized performance essentially acts as a fixed overhead. The remaining terms show that feature alignment alone is insufficient for effective cross-modal distillation, as aggressive representation alignment may inadvertently enlarge the semantic discrepancy between the teacher and student prediction distributions. In particular, when alignment induces representations that enlarge this predictive gap, the student model may overfit to misaligned semantic structures, ultimately degrading generalization performance.

This insight is made precise via the below formal definitions and theorem statement (Theorem 2.6).

Definition 2.4 (Feature Alignment). Let $\Delta$ denote the set of cost metrics δ on the pre-trained representation space such that the cross-entropy of the teacher prediction,

$$
\ell_ {\tau} (\boldsymbol {z}) \triangleq - \mathbb {E} _ {D ^ {T} (y | \boldsymbol {z})} \left[ \log p _ {T} (y \mid \boldsymbol {z}) \right], \tag {4}
$$

is $\tau _ { \delta } { \ - } 1$ Lipschitz with respect to $\delta \colon$

$$
\left| \ell_ {\tau} (\boldsymbol {z} _ {1}) - \ell_ {\tau} (\boldsymbol {z} _ {2}) \right| \leq \tau_ {\delta} \cdot \delta \big (\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2} \big). \tag {5}
$$

The feature alignment under the student feature map ϕ is

$$
\mathbf {F A} (\phi , \theta) \triangleq \min _ {\delta \in \Delta} \left\{\tau_ {\delta} \cdot W _ {\delta} \left(D ^ {T} (\boldsymbol {z}), D ^ {S} (\boldsymbol {z})\right) \right\}. \tag {6}
$$

where $W _ { \delta }$ is Wasserstein-1 distance with cost metric $\delta .$

Definition 2.5 (Label Alignment). Let $\kappa ( y , z )$ denote the label transport kernel between the teacher and student predictors, defined as $\kappa ( y , z ) \triangleq D ^ { T } ( y \mid z ) / D ^ { S } ( y \mid z )$ .

The label alignment between the teacher predictor $p _ { T }$ and the student predictor $p _ { S }$ is then defined as

$$
\mathbf {L A} (p _ {S}, p _ {T}) \triangleq - \mathbb {E} _ {D ^ {S} (\boldsymbol {z}, y)} \left[ \log \left(\frac {p _ {S} (y \mid \boldsymbol {z})}{p _ {T} (y \mid \boldsymbol {z}) ^ {\kappa (y , \boldsymbol {z})}}\right) \right]. \tag {7}
$$

The formal statement of our result can now be stated below.

Theorem 2.6 (Formal Statement). Plugging the above definition into the informal statement in Theorem 2.3,

$$
\operatorname{err} _ {S} (\phi) \leq \operatorname{err} _ {T} (\theta) + \boldsymbol {F A} (\phi , \theta) + \boldsymbol {L A} (p _ {S}, p _ {T}) . \tag {8}
$$

A detailed proof is provided in Appendix A.

Theorem 2.6 formalizes the earlier intuition by providing a principled characterization of the semantic gap in crossmodal knowledge distillation (CMKD) through both feature alignment (FA) and label alignment (LA).

## 2.3. Finite-Data Performance Bound

In this section, we analyze the finite-sample regime, where only $n _ { S }$ student samples and $n _ { T }$ teacher samples are available. In particular, we revisit the earlier asymptotic bound by replacing its population-level quantities with their empirical counterparts, leading to the following key quantities characterizing the generalized student loss $\mathrm { e r r } _ { S } ( \phi )$ :

Empirical Label Alignment $( { \bf L A } _ { e } )$ . We define the empirical label alignment $( { \bf L A } _ { e } )$ as a Monte Carlo estimate of the exact label alignment (LA) computed from $n _ { S }$ samples:

$$
\mathbf {L} \mathbf {A} _ {e} (p _ {S}, p _ {T}) \triangleq - \frac {1}{n _ {S}} \sum_ {i = 1} ^ {n _ {S}} \log \left(\frac {p _ {S} (y _ {i} \mid \boldsymbol {z} _ {i})}{p _ {T} (y _ {i} \mid \boldsymbol {z} _ {i}) ^ {\kappa (y _ {i} , \boldsymbol {z} _ {i})}}\right). \tag {9}
$$

Empirical Feature Alignment $( \mathbf { F } \mathbf { A } _ { e } )$ . We define the empirical feature alignment $( { \bf F } { \bf A } _ { e } )$ as an empirical estimate of the population-level feature alignment (FA) computed from $n _ { S }$ student samples and $n _ { T }$ teacher samples:

$$
\mathbf {F A} _ {e} (\phi , \theta) \triangleq \min _ {\delta \in \Delta} \left\{\tau_ {\delta} W _ {\delta} \left(D _ {n _ {T}} ^ {T} (\boldsymbol {z}), D _ {n _ {S}} ^ {S} (\boldsymbol {z})\right) \right\}. \tag {10}
$$

The formal statement of our finite-data result is stated below.

Theorem 2.7 (Formal Statement). Given the teacher and student feature maps θ and ϕ, the following holds with the probability at least 1 − 3δ where $\delta \in ( 0 , 1 / 3 ) .$ :

$$
\begin{array}{l} \operatorname{err} _ {S} (\phi) \leq \operatorname{err} _ {T} (\theta) + \mathbf {F A} _ {e} (\phi , \theta) + \mathbf {L A} _ {e} (p _ {S}, p _ {T}) \\ + \tau_ {\delta} \sqrt {\frac {\log (2 / \delta)}{2}} \left(\frac {1}{\sqrt {n _ {S}}} + \frac {1}{\sqrt {n _ {T}}}\right) \\ + O \left(n _ {S} ^ {- 1 / s _ {1}}\right) + O \left(n _ {T} ^ {- 1 / s _ {2}}\right) \\ + O \left(2 \sqrt {\frac {2 d \log (n _ {S} / d)}{n _ {S}}} + \sqrt {\frac {\log (1 / \delta)}{2 n _ {S}}}\right), \tag {11} \\ \end{array}
$$

where $s _ { 1 }$ and $s _ { 2 }$ are any constants larger than the upper Wasserstein dimensions (Weed & Bach, 2019) of the student and teacher representation distributions, respectively; and d denotes the VC dimension (Mohri et al., 2012) characterizing the complexity of the student hypothesis class. A detailed proof is provided in Appendix B.

Intuitively, Theorem 2.7 extends the earlier analysis to the practical finite-sample regime, where only a limited number of teacher and student samples are available. Notably, the appearance of the VC dimension reveals a fundamental trade-off between alignment and model complexity: while a more expressive student model may better align with the teacher, increasing its complexity d also enlarges the generalization gap. Consequently, for a fixed number of student samples nS, increasing model capacity does not necessarily improve performance, as the risk of overfitting becomes increasingly dominant, as reflected by the term $O ( \sqrt { d / n _ { S } } )$ .

## 3. Algorithm Design

We now present our proposed Cross-Modal Knowledge Distillation framework, UCMKD (Universal Cross Modal Knowledge Distillation), designed to address the unpaireddata challenge based on the theoretical insights developed in Section 2. Our framework adopts the bi-level optimization approach in (Finn et al., 2017) that minimizes the generalized student loss (outer optimization) while calibrating the cross-modal semantic gap between teacher and student models (inner optimization). As suggested by our theoretical analysis, this semantic gap is characterized via two complementary quantities: feature alignment (FA) and label alignment (LA). Accordingly, UCMKD operates through the two-stage workflow illustrated in Figure 1(b). In stage 1, we learn the student encoder ϕ by minimizing the representation distributional discrepancy between teacher and student models in the latent space (Section 3.1). In stage 2, we optimize both the encoder $\phi$ and prediction head $p _ { S } ( y \mid z )$ to minimize the prediction distributional discrepancy between teacher and student models (Section 3.2).

Algorithm 1 Universal Cross-Modal KD (UCMKD)  
1: input: teacher model $M_{T} \triangleq (\theta, p_{T}(y \mid z = \theta(\boldsymbol{x}^{T})))$ ,
student and teacher datasets: $D^{S}$ and $D^{T}$ ,
numbers of outer update epochs $n_{0}$ ,
inner adaptation epochs ( $n_{1}$ and $n_{2}$ ),
learning rate $\eta$ ,
and hyper-parameters $\lambda_{1}, \lambda_{2}$ .
2: output: student model $M_{S} \triangleq (\phi, p_{S}(y \mid z = \phi(\boldsymbol{x}^{S})))$ 3: initialize the student encoder and predictor $\phi^{0}, p_{S}^{0}$ .
4: for t = 1 to $n_{0}$ do
5: $\phi_{tmp} \leftarrow \phi^{t-1}$ and $p_{tmp} \leftarrow p_{S}^{t-1}$ 6: for r = 1 to $n_{1}$ do
7: $\phi_{tmp} \leftarrow \phi_{tmp} - \eta \lambda_{1} \nabla_{\phi} \ell_{\mathbf{FA}}(\phi_{tmp})$ # Eq. (14)
8: end for
9: for r = 1 to $n_{2}$ do
10: # see Eq. (15)
11: $\phi_{tmp} \leftarrow \phi_{tmp} - \lambda_{2} \eta \nabla_{\phi} \ell_{\mathbf{LA}}(\phi_{tmp}, p_{tmp})$ 12: $p_{tmp} \leftarrow p_{tmp} - \lambda_{2} \eta \nabla_{p} \ell_{\mathbf{LA}}(\phi_{tmp}, p_{tmp})$ 13: end for
14: $\phi^{t} = \phi^{t-1} - \eta \nabla_{\phi} err_{S}(\phi_{tmp}, p_{tmp})$ # Eq. (2)
15: $p_{S}^{t} = p_{S}^{t-1} - \eta \nabla_{p} err_{S}(\phi_{tmp}, p_{tmp})$ # Eq. (2)
16: end for
17: return distilled student model $M_{S} = (\phi^{n_{0}}, p_{S}^{n_{0}})$

To elaborate on this design, we note that directly optimizing the student objective together with the semantic crossmodal gap from Theorem 2.6, as in conventional CMKD approaches, is often unstable due to the highly coupled interaction between the prediction map $p _ { S } ( y \mid z )$ and the feature map $z = \phi ( \pmb { x } ^ { S } )$ (see Table 6). In particular, update to the feature extractor ϕ simultaneously modifies the alignment between teacher and student distributions while reshaping the student representation landscape $D ^ { S } ( z )$ on which the predictor $p _ { S } ( y \mid z )$ is learned. This moving-target effect substantially complicates the optimization landscape and can lead to unstable convergence. To address this challenge, we adopt a hybrid bi-level optimization approach with a structured two-stage inner-update procedure that calibrates the student loss with respect to the cross-modal semantic gap. In particular, the inner optimization is decomposed into minimizing feature alignment loss (stage 1) and minimizing label alignment loss (stage 2). Figure 1 provides an overview of the proposed framework, which optimizes the student loss via inner adaptation steps used to compute the outer optimization loss in a meta-learning fashion. The pseudo-code of the full algorithm is provided in Algorithm 1.

![](images/9b0daf1b4efccdf8825cff24f4dffbf2c1a5a8850ac2ba19abc73786704b3603.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Teacher Modalities x^T"] --> B["Encoder θ"]
  B --> C["Latent space (Z)"]
  C --> D["Encoder φ"]
  D --> E["Student Modalities x^S"]
  F["Feature Alignment Section 3.1"] --> G["D^T(z) D^S(z)"]
  H["Label Alignment Section 3.2"] --> I["p_T(y | z) p_S(y | z)"]
    J["student's generalized error ≤ teacher's generalized error + feature alignment + label alignment (Section 2)"]
  GIJ["G & I & J"] --> K["Feature Alignment Section 3.1"]
    style A fill:#f9f,stroke:#333
    style F fill:#ccf,stroke:#333
    style H fill:#cfc,stroke:#333
    style J fill:#fcc,stroke:#333
```
</details>

Figure 1. Overview of our UCMKD framework: The teacher and student encoders map inputs from different modalities into a shared latent space Z. The cross-modal generalization bound decomposes into two distributional quantities: Feature Alignment, a Wasserstein distance between the latent distributions $\mathcal { D } ^ { T } ( z )$ and $\mathcal { D } ^ { S } ( z )$ – Section 3.1; and Label Alignment, a distance measure between the induced predictive distributions $p _ { T } ( y \mid z )$ and $p _ { S } ( y \mid z ) - \xi$ ection 3.2. Theorems 2.6 and 2.7 bound the student’s generalized error by the sum of teacher error, feature alignment, and label alignment, motivating distribution-level alignment without sample-level pairing.

## 3.1. Feature Alignment Loss

To compute the feature alignment (FA), a direct approach is to minimize the optimal transport distance between the empirical student and teacher representation distributions in the shared latent space Z:

$$
W _ {\delta} \left(D _ {n _ {S}} ^ {S} (\boldsymbol {z}), D _ {n _ {T}} ^ {T} (\boldsymbol {z})\right) = \min _ {\pi \in \Pi} \sum_ {i = 1} ^ {n _ {S}} \sum_ {j = 1} ^ {n _ {T}} \pi_ {i j} \delta \left(\boldsymbol {z} _ {i} ^ {S}, \boldsymbol {z} _ {j} ^ {T}\right) \tag {12}
$$

where Π denote the set of coupling π between $D _ { n _ { S } } ^ { S } ( z )$ and $D _ { n _ { T } } ^ { T } ( z ) ^ { 1 }$ and $\pi _ { i j } \triangleq \pi ( z _ { i } ^ { S } , z _ { j } ^ { T } )$ . This can be naively achieved via solving a linear program with a computational complexity $O ( \operatorname* { m a x } ( n _ { S } , n _ { T } ) ^ { 3 } )$ (Peyre & Cuturi ´ , 2020).

To reduce this complexity, we instead adopt the following entropic-regularized formulation:

$$
\pi^ {*} = \arg \min _ {\pi \in \Pi} \left(\sum_ {i = 1} ^ {n _ {S}} \sum_ {j = 1} ^ {n _ {T}} \pi_ {i j} \delta \left(\boldsymbol {z} _ {i} ^ {S}, \boldsymbol {z} _ {j} ^ {T}\right) + \epsilon H (\pi)\right). \tag {13}
$$

This regularized problem can be solved efficiently using the Sinkhorn algorithm with quadratic complexity (Peyre &´ Cuturi, 2020). In practice, we use the Euclidean transport cost $\delta \triangleq \ell _ { 2 }$ and the default regularization parameter $\epsilon = 0 . 1$ . The resulting feature alignment loss is defined as

$$
\ell_ {\mathbf {F A}} (\theta , \phi) \triangleq W _ {\ell_ {2}} \left(D _ {N _ {S}} ^ {S} (\boldsymbol {z}), D _ {N _ {T}} ^ {T} (\boldsymbol {z})\right), \tag {14}
$$

where the contribution of feature alignment is controlled by the hyperparameter $\lambda _ { 1 }$ (Algorithm 1). Here, $D _ { n _ { S } } ^ { S } ( z )$ and

$D _ { n _ { T } } ^ { T } ( z )$ denote the push-forward of the empirical student and teacher input distributions under the corresponding feature maps ϕ and θ, respectively.

## 3.2. Label Alignment Loss

The label alignment loss is set to be the empirical $\mathbf { L A } _ { e }$ term previously defined in Eq. (9):

$$
\ell_ {\mathbf {L A}} \left(p _ {S}, p _ {T}\right) \triangleq - \frac {1}{n _ {S}} \sum_ {i = 1} ^ {n _ {S}} \log \left(\frac {p _ {S} (y _ {i} \mid \boldsymbol {z} _ {i})}{p _ {T} (y _ {i} \mid \boldsymbol {z} _ {i}) ^ {\kappa (y _ {i} , \boldsymbol {z} _ {i})}}\right) \tag {15}
$$

where $\kappa ( y , z ) \triangleq D ^ { T } ( y \mid z ) / D ^ { S } ( y \mid z )$ .

Intuitively, $\kappa ( y , z )$ quantifies the agreement between teacher and student prediction distributions, thereby acting as a gating mechanism for selective knowledge distillation. A practical estimation procedure for the transport kernel is provided in Appendix D. When the teacher’s prediction conflicts with the student’s target semantics, particularly when the teacher assigns negligible probability to the target label, the kernel approaches zero, $\kappa ( y , z ) \simeq 0$ . In this regime, the alignment loss naturally reduces to the standard supervised loss of the student model:

$$
\lim _ {\kappa \rightarrow 0} \ell_ {\mathbf {L A}} = \mathbb {E} _ {D ^ {S} (\boldsymbol {z}, y)} [ - \log p _ {S} (y \mid \boldsymbol {z}) ] = \operatorname{err} _ {S} (\phi). \tag {16}
$$

This mechanism ensures that, under semantic disagreement, the student model prioritizes its own supervised signal rather than inheriting potentially unreliable teacher guidance. Furthermore, the formulation naturally enables a pseudo-labeling strategy (Nguyen et al., 2020) for minimizing the semantic prediction discrepancy between teacher and student models without requiring paired data, thus improving robustness in unpaired settings.

## 4. Empirical Evaluation

In this section, we present comprehensive experimental results validating the effectiveness of our method under both unpaired-data (Section 4.2) and paired-data (Section 4.3) settings. We further provide ablation studies analyzing parameter sensitivity and performance under data-scarcity scenarios (Section 4.4). Due to limited space, additional experimental results and ablation studies are deferred to Appendix G.

## 4.1. Implementation Details

We evaluate our proposed cross-modal distillation method UCMKD on 4 multi-modal datasets which include: (1) AVE (Tian et al., 2018) is an audio-visual dataset for audio-visual event localization, which has 28 classes; (2) CREMA-D (Cao et al., 2014) is an audio-visual dataset for speech emotion recognition, with 6 categorizations; (3) RAVDESS (Livingstone & Russo, 2018) is an audio-visual dataset containing 1,440 emotional utterances with 8 different emotion classes; (4) VGGsound (Chen et al., 2020) is a large-scale video dataset containing about 200 000 videos and more than 300 classes covering daily life activities.

Unpaired Data Simulation. To simulate the unpaired setting, we apply a stochastic permutation to the original multimodal dataset $\{ \mathcal { X } _ { 1 } , \mathcal { X } _ { 2 } , \mathcal { Y } \}$ . Specifically, we break the instance-level correspondence between modalities by randomly shuffling the indices of one modality relative to the other, resulting in two independent subsets, $\{ \mathcal { X } _ { 1 } , \mathcal { V } \}$ and $\{ \mathcal { X } _ { 2 } , \mathcal { V } \}$ . This procedure removes sample-level alignment while preserving the marginal distributions of each modality.

Hyperparameters. We use the same hyperparameter configuration across all baselines and train each network for 100 epochs with an initial learning rate of 1e−2. Following (Peng et al., 2022), we adopt ResNet-18 (He et al., 2015) as the backbone architecture for both visual and audio modalities. Additional implementation and hyperparameter details are provided in Appendix F.

## 4.2. Experiment Results on Unpaired Setting

Given the limited literature on knowledge distillation under unpaired settings, we compare our method (UCMKD) against: (1) Cross-Entropy, which serves as a nondistillation baseline; and (2) Feature Distillation, which performs distillation through latent-space representation alignment. Table 1 reports the prediction accuracy of all methods across the evaluated tasks. UCMKD consistently achieves the best performance, with an average improvement of approximately 14.3% over Cross-Entropy and 7.5% over Feature Distillation. Notably, UCMKD outperforms Vanilla KD (Hinton et al., 2015) on 6 out of 8 tasks, despite Vanilla KD benefiting from paired multimodal data, an advantage unavailable to UCMKD in our unpaired setting. These results demonstrate the effectiveness of our method in transferring cross-modal knowledge without requiring explicit sample-level correspondence.

![](images/cb46a52fdf5ae8fa48f0aaa9a6c6c6a0150006cb64794aedcb7a8fa5800b380a.jpg)

<details>
<summary>heatmap</summary>

| LA weight | RAVDESS(A-V) Accuracy (%) | RAVDESS(A-V) Accuracy (%) | RAVDESS(A-V) Accuracy (%) | RAVDESS(A-V) Accuracy (%) | RAVDESS(V-A) Accuracy (%) | RAVDESS(V-A) Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.2 | 79.92 | 74.03 | 71.33 | 76.42 | 71.53 | 78.02 |
| 0.4 | 74.83 | 76.82 | 75.92 | 72.53 | 75.33 | 77.12 |
| 0.6 | 75.82 | 75.03 | 79.42 | 71.53 | 74.33 | 76.02 |
| 0.8 | 76.82 | 70.43 | 76.02 | 71.83 | 74.33 | 73.83 |
| 1.0 | 76.82 | 72.33 | 73.83 | 71.23 | 75.52 | 76.12 |
(a) RAVDESS(A-V)
(b) RAVDESS(V-A)
</details>

Figure 2. Heatmap of the performance of our method (UCMKD) on RAVDESS (Livingstone & Russo, 2018) across different values of the hyper-parameters λ1 and λ2 (Algorithm 1) under (a) audioto-visual (A → V ) and (b) visual-to-audio (V → A) settings.

## 4.3. Experiment Results on Paired Setting

In this section, we evaluate UCMKD under the paired-data knowledge distillation (KD) setting and compare it against Vanilla KD (Hinton et al., 2015) and several state-of-the-art baselines, including C2KD (Huo et al., 2024), DKD (Zhao et al., 2022), RKD (Park et al., 2019), RLD (Sun et al., 2024), FitNet (Romero et al., 2014), and Review (Chen et al., 2021). As shown in Table 2, our method consistently outperforms competing baselines on the AVE, RAVDESS, and CREMA-D datasets. In particular, our framework achieves the highest accuracy on 5 out of 8 experimental tasks. This is consistent with the unpaired-setting results in Section 4.2 and further demonstrate the universally robustness of the proposed framework across both paired- and unpaired-data cross-modal distillation settings.

## 4.4. Ablation Studies

Data Scarcity Scenario. To evaluate the robustness of our method under data-scarcity conditions, we conduct experiments using reduced training sets. Specifically, we subsample the original training data at ratios of 0.3, 0.4, and 0.5, while retaining the full test set to ensure a consistent evaluation benchmark. Tables 3, 4, and 5 report the performance on the RAVDESS dataset of our method and two unpaired-setting baselines: Cross-Entropy and Feature Distillation. We also include Vanilla KD (Hinton et al., 2015) under the paired setting using the same ResNet-18 backbone. The results show that our method consistently achieves the strongest performance across all data-constrained scenarios.

Parameter Sensitivity. We conduct a sensitivity analysis to evaluate the effect of the hyperparameters $\lambda _ { 1 }$ and $\lambda _ { 2 } .$ , which control the relative contributions of the Feature Alignment (FA) and Label Alignment (LA) losses, respectively. Across these experiments, the inner-loop optimization steps are fixed to $n _ { 1 } = n _ { 2 } = 1$ (see Algorithm 1). Figure 2 presents the resulting performance heatmaps (audio-to-visual and visual-to-audio) on the RAVDESS dataset (Liu et al., 2022) according to the hyper-parameter grid with different choices for $\lambda _ { 1 }$ and $\lambda _ { 2 }$ within $0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0$ . The best performance for audio-to-visual $( A \to V )$ distillation is achieved at $( \lambda _ { 1 } , \lambda _ { 2 } ) \ : = \ : ( 0 . 2 , 1 . 0 )$ , while visual-to-audio $( V  A )$

Table 1. Prediction accuracy on the AVE, RAVDESS, CREMA-D, and VGGSound datasets under the unpaired setting, comparing Cross-Entropy, Feature-Based Knowledge Distillation, and our proposed UCMKD framework. For reference, we also report Vanilla KD (Hinton et al., 2015) trained under paired supervision. $A  { \bar { V } }$ and $V  A$ denote distillation from audio to visual and visual to audio modalities, respectively. Despite operating without paired supervision, UCMKD consistently outperforms unpaired baselines and surpasses the paired Vanilla KD baseline on 6 out of 8 experimental tasks.

<table><tr><td rowspan="2">Method</td><td colspan="2">AVE</td><td colspan="2">RAVDESS</td><td colspan="2">CREMA-D</td><td colspan="2">VGGsound</td></tr><tr><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td></tr><tr><td>Teacher</td><td>52.74</td><td>30.35</td><td>79.92</td><td>77.72</td><td>65.46</td><td>70.97</td><td>56.78</td><td>44.43</td></tr><tr><td>Cross Entropy</td><td> $27.70 \pm 2.35$ </td><td> $50.08 \pm 2.88$ </td><td> $65.47 \pm 3.69$ </td><td> $70.66 \pm 1.16$ </td><td> $71.51 \pm 0.88$ </td><td> $61.96 \pm 0.83$ </td><td> $41.68 \pm 2.32$ </td><td> $54.40 \pm 2.34$ </td></tr><tr><td>Feature KD</td><td> $31.01 \pm 1.04$ </td><td> $48.51 \pm 1.49$ </td><td> $65.37 \pm 3.20$ </td><td> $69.80 \pm 2.96$ </td><td> $69.22 \pm 1.38$ </td><td> $61.69 \pm 0.69$ </td><td> $41.07 \pm 1.84$ </td><td> $52.08 \pm 0.83$ </td></tr><tr><td>Vanilla KD</td><td> $29.85 \pm 1.85$ </td><td> $49.01 \pm 2.47$ </td><td> $67.17 \pm 3.86$ </td><td> $73.10 \pm 1.63$ </td><td> $72.18 \pm 1.15$ </td><td> $62.45 \pm 0.56$ </td><td> $43.35 \pm 0.29$ </td><td> $51.71 \pm 0.69$ </td></tr><tr><td>UCMKD</td><td> $34.16 \pm 1.12$ </td><td> $52.24 \pm 1.08$ </td><td> $73.83 \pm 1.25$ </td><td> $74.43 \pm 2.15$ </td><td> $71.64 \pm 0.86$ </td><td> $66.67 \pm 1.24$ </td><td> $43.10 \pm 0.38$ </td><td> $56.84 \pm 0.47$ </td></tr></table>

Table 2. Prediction accuracy on the AVE, RAVDESS, CREMA-D, and VGGSound datasets under the paired setting, comparing Vanilla KD (Hinton et al., 2015), C2KD (Huo et al., 2024), DKD (Zhao et al., 2022), RKD (Park et al., 2019), RLD (Sun et al., 2024), FitNet (Romero et al., 2014), Review (Chen et al., 2021), and our UCMKD framework. $A  V$ and $V $ A denote distillation from audio to visual and visual to audio modalities, respectively. UCMKD achieves the best performance on 5 out of 8 experimental tasks.

<table><tr><td rowspan="2">Method</td><td colspan="2">AVE</td><td colspan="2">RAVDESS</td><td colspan="2">CREMA-D</td><td colspan="2">VGGsound</td></tr><tr><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td></tr><tr><td>Vanilla KD</td><td> $29.85 \pm 1.85$ </td><td> $49.01 \pm 2.47$ </td><td> $67.17 \pm 3.86$ </td><td> $73.10 \pm 1.63$ </td><td> $\mathbf{72.18} \pm \mathbf{1.15}$ </td><td> $62.45 \pm 0.56$ </td><td> $43.35 \pm 0.29$ </td><td> $51.71 \pm 0.69$ </td></tr><tr><td>RLD</td><td> $22.80 \pm 1.22$ </td><td> $42.87 \pm 0.82$ </td><td> $56.64 \pm 1.98$ </td><td> $63.94 \pm 0.85$ </td><td> $43.54 \pm 4.59$ </td><td> $53.04 \pm 0.67$ </td><td> $32.73 \pm 0.66$ </td><td> $44.66 \pm 0.70$ </td></tr><tr><td>RKD</td><td> $27.86 \pm 0.70$ </td><td> $42.54 \pm 0.54$ </td><td> $41.60 \pm 6.33$ </td><td> $39.50 \pm 1.53$ </td><td> $44.44 \pm 2.79$ </td><td> $62.50 \pm 0.87$ </td><td> $37.30 \pm 0.49$ </td><td> $50.71 \pm 0.40$ </td></tr><tr><td>DKD</td><td> $22.80 \pm 0.65$ </td><td> $34.08 \pm 1.66$ </td><td> $62.67 \pm 4.27$ </td><td> $62.27 \pm 2.57$ </td><td> $30.02 \pm 6.11$ </td><td> $57.40 \pm 0.38$ </td><td> $35.70 \pm 0.28$ </td><td> $43.85 \pm 0.29$ </td></tr><tr><td>C2KD</td><td> $33.33 \pm 0.73$ </td><td> $47.15 \pm 1.61$ </td><td> $56.41 \pm 2.42$ </td><td> $\mathbf{82.78} \pm \mathbf{0.41}$ </td><td> $71.50 \pm 0.11$ </td><td> $64.43 \pm 0.42$ </td><td> $40.90 \pm 0.30$ </td><td> $\mathbf{61.90} \pm \mathbf{0.27}$ </td></tr><tr><td>FitNet</td><td> $25.87 \pm 1.95$ </td><td> $49.25 \pm 1.61$ </td><td> $68.08 \pm 0.75$ </td><td> $69.96 \pm 3.43$ </td><td> $70.11 \pm 1.32$ </td><td> $65.01 \pm 0.01$ </td><td> $37.90 \pm 0.39$ </td><td> $57.10 \pm 0.79$ </td></tr><tr><td>Review</td><td> $22.30 \pm 0.62$ </td><td> $48.92 \pm 0.65$ </td><td> $54.91 \pm 3.20$ </td><td> $71.50 \pm 2.00$ </td><td> $63.89 \pm 1.68$ </td><td> $61.02 \pm 0.54$ </td><td> $38.20 \pm 0.47$ </td><td> $57.90 \pm 0.79$ </td></tr><tr><td>UCMKD</td><td> $33.50 \pm 1.82$ </td><td> $53.07 \pm 0.51$ </td><td> $\mathbf{76.06} \pm \mathbf{2.28}$ </td><td> $75.13 \pm 0.65$ </td><td> $70.43 \pm 0.66$ </td><td> $\mathbf{66.75} \pm \mathbf{1.36}$ </td><td> $\mathbf{43.70} \pm \mathbf{0.182}$ </td><td> $55.98 \pm 0.38$ </td></tr></table>

Table 3. Prediction accuracy on the RAVDESS dataset using the ResNet-18 backbone with a training subset ratio of $0 . 5 ,$ comparing our method UCMKD against the Cross-Entropy, Feature-Based Knowledge Distillation, and Vanilla KD baselines.

<table><tr><td>Method</td><td>A→V</td><td>V→A</td></tr><tr><td>CE</td><td>59.87 ± 0.76</td><td>71.86 ± 2.30</td></tr><tr><td>Feature KD</td><td>55.48 ± 2.85</td><td>69.16 ± 5.35</td></tr><tr><td>Vanilla KD</td><td>57.91 ± 4.93</td><td>69.03 ± 4.68</td></tr><tr><td>UCMKD</td><td>72.76 ± 5.14</td><td>74.35 ± 1.27</td></tr></table>

Table 4. Prediction accuracy on the RAVDESS dataset using the ResNet-18 backbone with a training subset ratio of 0.4, comparing our method UCMKD against the Cross-Entropy, Feature-Based Knowledge Distillation, and Vanilla KD baselines.

<table><tr><td>Method</td><td>A→V</td><td>V→A</td></tr><tr><td>CE</td><td>57.14 ± 3.64</td><td>69.93 ± 3.75</td></tr><tr><td>Feature KD</td><td>50.88 ± 3.49</td><td>71.96 ± 0.84</td></tr><tr><td>Vanilla KD</td><td>57.68 ± 1.50</td><td>71.63 ± 0.91</td></tr><tr><td>UCMKD</td><td>69.56 ± 2.17</td><td>78.72 ± 0.80</td></tr></table>

Table 5. Prediction accuracy on the RAVDESS dataset using the ResNet-18 backbone with a training subset ratio of $0 . 3 ,$ comparing our method against the Cross-Entropy, Feature-Based Knowledge Distillation, and Vanilla KD baselines.

<table><tr><td>Method</td><td>A→V</td><td>V→A</td></tr><tr><td>CE</td><td>54.51 ± 4.10</td><td>68.96 ± 2.20</td></tr><tr><td>Feature KD</td><td>50.71 ± 6.76</td><td>63.97 ± 2.20</td></tr><tr><td>Vanilla KD</td><td>50.98 ± 4.40</td><td>68.46 ± 0.60</td></tr><tr><td>UCMKD</td><td>69.03 ± 2.50</td><td>77.32 ± 1.34</td></tr></table>

distillation performs best at $( \lambda _ { 1 } , \lambda _ { 2 } ) ~ = ~ ( 1 . 0 , 0 . 8 )$ . Notably, even under less favorable hyperparameter configurations, our method consistently outperforms the baselines on $A  V$ and remains competitive on $V  A$ . These results demonstrate the robustness and stability of our UCMKD across a broad range of loss-weight configurations.

Informativeness of Theoretical Bounds. To empirically validate the informativeness (i.e., tightness) of the theoretical results developed in Section 2, we evaluate the gap between the theoretical bounds and the observed empirical performance across multiple datasets. Figure 3 summarizes the resulting tightness analysis for both the infinite-data setting (Theorem 2.6) and the finite-data regime (Theorem 2.7). Overall, the bounds remain reasonably tight across all datasets, with an average gap of 24.5%. Notably, on the large-scale VGGSound dataset containing over 300K+ samples, the gap decreases to 11%, suggesting that the bounds become increasingly informative as data coverage grows. This observation is consistent with the theoretical behavior characterized in Theorems 2.6 and 2.7.

![](images/5afad497ba85f68b59a213ca3bf796aff11cb61e3f41c1586c99c76c7ba2161b.jpg)

<details>
<summary>stacked bar chart</summary>

Comparison between Student Loss and Theoretical Upper Bound
| Model | Feature Alignment Loss | Label Alignment Loss | Teacher Loss | Student Loss (LHS) |
|---|---|---|---|---|
| AVE | 0.32 | 1.45 | 0.00 | 1.41 |
| RAVDESS | 0.23 | 1.28 | 0.17 | 1.22 |
| CREMAD | 0.19 | 1.26 | 0.00 | 1.01 |
| VGG | 0.30 | 2.58 | 0.00 | 2.32 |
RHS=1.66, LHS=1.41
RHS=1.74, LHS=1.22
RHS=1.35, LHS=1.01
RHS=2.58, LHS=2.32
</details>

Figure 3. Informativeness of the theoretical bound across the AVE, RAVDESS, CREMA-D, and VGGSound datasets. The proposed bound remains reasonably tight with an average gap of 24.5%.

Table 6. Component-wise ablation results on the AVE and RAVDESS datasets comparing FA-only (bi-level), LA-only (bilevel), both (w/o bi-level), and our UCMKD framework.

<table><tr><td rowspan="2">Method</td><td colspan="2">AVE</td><td colspan="2">RAVDESS</td></tr><tr><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td></tr><tr><td>FA-only (bilevel)</td><td>31.01</td><td>48.51</td><td>65.37</td><td>69.80</td></tr><tr><td>LA-only (bilevel)</td><td>30.02</td><td>48.92</td><td>67.90</td><td>69.33</td></tr><tr><td>Both (w/o bilevel)</td><td>28.11</td><td>48.26</td><td>66.47</td><td>69.53</td></tr><tr><td>UCMKD</td><td>34.16</td><td>52.24</td><td>73.83</td><td>74.43</td></tr></table>

Component-wise Contribution. Table 6 presents an ablation study on the contributions of feature alignment (FA), label alignment (LA), and the bi-level optimization framework. We observe that FA-only and LA-only variants each achieve competitive performance individually. However, directly combining both alignment losses without the bilevel formulation leads to noticeable performance degradation, highlighting the instability of naive simultaneous optimization (see Section 3). In contrast, UCMKD, which integrates both alignment objectives within the proposed bi-level framework, consistently achieves the strongest performance across all settings. These findings align with the theoretical insights of Theorem 2.6 and demonstrate the importance of both alignment components and the bi-level optimization approach.

Alternative Distribution Distances. We note that using the $\ell _ { 2 }$ cost for OT is standard practice in prior works (Damodaran et al., 2018; Tran et al., 2026). Moreover, our theoretical bound is agnostic to the specific choice of cost metric. To further assess this design choice, Table 7 compares the performance of several alternative cost metrics. Overall, all metrics achieve comparable results. Among them, $\ell _ { 2 }$ achieves marginally better results on 3 out of 4 tasks, making it a reasonable default choice in practice.

Scalability with Larger Backbones. To further evaluate scalability under more realistic settings, we conduct additional experiments using ViT-based architectures, with ViT-B as the teacher and ViT-S as the student following the protocol in (Addepalli et al., 2024). These models are representative of modern large-scale vision architectures. As shown in Table 8, UCMKD consistently achieves the best student performance across all datasets and transfer directions. These results demonstrate that the proposed framework remains effective when transitioning from ResNet-based backbones to substantially larger ViT-based architectures. Combined with the complexity analysis in Appendix E, these findings provide further evidence of the scalability of our approach in practical large-scale settings.

Table 7. Comparison of different transport cost metrics, including $\ell _ { 1 } , \ell _ { 2 } ,$ and angular distance, on RAVDESS and CREMA-D.

<table><tr><td>Dataset</td><td> $\ell_1$ </td><td> $\ell_2$ </td><td>Angular</td></tr><tr><td>RAVDESS (A→V)</td><td>74.23</td><td>73.83</td><td>71.03</td></tr><tr><td>RAVDESS (V→A)</td><td>73.65</td><td>74.44</td><td>73.03</td></tr><tr><td>CREMA-D (A→V)</td><td>70.67</td><td>71.68</td><td>71.07</td></tr><tr><td>CREMA-D (V→A)</td><td>63.52</td><td>66.59</td><td>62.41</td></tr></table>

Table 8. Scalability evaluation using a ViT-based architecture with ViT-B (patch16-224, 86M parameters) as the teacher and ViT-S (patch16-224, 22M parameters) as the student.

<table><tr><td rowspan="2">Method</td><td colspan="2">AVE</td><td colspan="2">RAVDESS</td></tr><tr><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td></tr><tr><td>Teacher</td><td>75.87</td><td>70.15</td><td>90.41</td><td>89.11</td></tr><tr><td>CE</td><td>51.19</td><td>53.73</td><td>65.63</td><td>66.13</td></tr><tr><td>Feature KD</td><td>50.96</td><td>56.22</td><td>69.83</td><td>67.73</td></tr><tr><td>UCMKD</td><td>56.97</td><td>58.21</td><td>80.32</td><td>72.43</td></tr></table>

## 5. Related Works

In this section, we provide the literature review of the most relevant works on unimodal KD (Section 5.1) and crossmodal KD (Section 5.2). The detailed formulations of both mentioned settings are provided in Appendix C.

## 5.1. In-Modal Knowledge Distillation

In-Modal Knowledge Distillation (KD) transfers the knowledge of a pretrained teacher model to a student model by minimizing discrepancies between their output predictions or intermediate representations. The seminal work of (Hinton et al., 2015) formulates this as minimizing the Kullback–Leibler (KL) divergence between teacher and student soft predictions, enabling the compression of large models while largely preserving predictive performance. Follow-up work has explored a broad range of alternative mechanisms for effective knowledge transfer. For example, CRD (Tian et al., 2020) introduces contrastive objectives for representation-level distillation, while

SCKD (Zhu & Wang, 2021) adaptively adjusts the distillation process according to gradient similarity between teacher and student models. DKD (Zhao et al., 2022) decomposes KD into target-class and non-target-class distillation to improve flexibility, and Review (Chen et al., 2021) leverages multi-level teacher representations through a feature-review mechanism. DIST (Huang et al., 2022) further develops correlation-based objectives to capture inter-class and intraclass structural relations, while L2D (Yang et al., 2023) extends such relation-based distillation to multi-label classification. SHAKE (Li & JIN, 2022) bridges offline and online KD through auxiliary shadow heads, (Lv et al., 2024) replaces the KL divergence with Wasserstein-distance-based objectives, and RLD (Sun et al., 2024) dynamically refines teacher logits using label information to suppress misleading supervision from incorrect teacher predictions.

Despite their strong empirical performance, these methods predominantly assume that teacher and student models operate on identical training data, thereby overlooking the challenges of Cross-Modal Knowledge Distillation (CMKD), particularly under unpaired settings where explicit samplelevel correspondence is unavailable.

## 5.2. Cross-Modal Knowledge Distillation

Cross-Modal Knowledge Distillation (CMKD) extends traditional KD to settings where teacher and student models operate on different data modalities. Early work by Gupta et al. (2015) transfers supervision from a labeled modality to an unlabeled paired modality, while Roheda et al. (2018) employs GANs to transfer knowledge across missing and available modalities. Xue et al. (2021) adapts multimodal networks to unlabeled modalities through pseudo-label sampling from unimodal teachers, and Lee et al. (2023) develops decomposed cross-modal distillation for RGB-based detection using optical-flow supervision.

Other work explores cross-modal transfer for dense indoor prediction (Yun et al., 2023) and theoretical understanding of modality interactions through modality Venn diagrams and modality-focusing hypotheses (Xue et al., 2022). More recent approaches aim to improve cross-modal transfer under increasingly complex settings. C2KD (Huo et al., 2024) introduces selective bidirectional distillation to bridge modality gaps, while (Sarkar & Etemad, 2022) develops a self-supervised framework for cross-modal distillation from unlabeled videos. COSMOS (Kim et al., 2024) further incorporates text-cropping and cross-attention mechanisms for vision-language models, and XKD (Bendidi et al., 2025) explores weakly paired microscopy and transcriptomics data.

Despite these advances, existing CMKD methods still rely heavily on paired or weakly paired multi-modal data, which are often costly or unavailable in realistic settings. Addressing this has been the focus of this paper.

## 6. Limitations and Future Works

While our framework provides a principled formulation and effective algorithm for unpaired CMKD, several directions remain open. First, although our theoretical analysis permits arbitrary transport costs $\delta ( \cdot , \cdot )$ , we instantiate the framework using the Euclidean metric for simplicity. Learning the transport geometry via adaptive metric learning may further tighten the alignment objective. Second, the proposed bi-level optimization introduces additional computational overhead due to second-order updates or their approximations. Improving optimization efficiency through implicit differentiation or reduced unrolling would further enhance scalability. Third, our results on the large-scale VGGSound benchmark suggest that the framework remains effective beyond small curated datasets, motivating future exploration in settings with foundation models where cross-modal transfer must operate without explicit sample-level pairing. Finally, since the proposed framework relies on distribution-level alignment rather than paired supervision, extending it to cross-modal generative modeling under unpaired settings is another promising direction for future follow-up.

## 7. Conclusion

This paper studies cross-modal knowledge distillation under the challenging unpaired setting and proposes UCMKD, a principled framework built upon two key components: Feature Alignment and Label Alignment. We establish both infinite-sample and finite-sample generalization bounds for the student model and develop a practical meta-learningstyle optimization framework to realize these objectives. Extensive experiments on AVE, RAVDESS, CREMA-D, and VGGSound demonstrate consistent improvements in unpaired settings while remaining competitive with SOTA methods when paired data are available. The proposed formulation also naturally extends beyond prediction task to general distribution-matching losses, opening promising directions for large-scale multi-modal transfer and unsupervised cross-modal generative modeling.

## Acknowledgement

This work utilized GPU compute resources at SDSC and ACES through allocation CIS230391 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services and Support (ACCESS) program (Boerner et al., 2023), which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. Trong Nghia Hoang is supported by National Science Foundation CAREER Award IIS-2544071. The authors also acknowledge the compute support from Modal.

## Impact Statement

Our work provides a framework for cross-modal knowledge distillation without requiring paired data, enabling multimodal transfer in settings where synchronized datasets are unavailable. However, distribution-level alignment might propagate biases or spurious correlations inherited from the teacher model. Careful inspection of the teacher model is important when such systems are deployed in practice.

## References

Addepalli, S., Asokan, A. R., Sharma, L., and Babu, R. V. Leveraging vision-language models for improving domain generalization in image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024.  
Bendidi, I., Mesbahi, Y. E., Denton, A. K., Suri, K., Kenyon-Dean, K., Genovesio, A., and Noutahi, E. A cross modal knowledge distillation & data augmentation recipe for improving transcriptomics representations through morphological features. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=w7lm8AjzH6.  
Boerner, T. J., Deems, S., Furlani, T. R., Knuth, S. L., and Towns, J. Access: Advancing innovation: Nsf’s advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing 2023: Computing for the Common Good, PEARC ’23, pp. 173–176, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450399852. doi: 10.1145/ 3569951.3597559. URL https://doi.org/10. 1145/3569951.3597559.  
Bucilua, C., Caruana, R., and Niculescu-Mizil, A. Model compression. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’06, pp. 535–541, New York, NY, USA, 2006. Association for Computing Machinery. ISBN 1595933395. doi: 10.1145/ 1150402.1150464. URL https://doi.org/10. 1145/1150402.1150464.  
Cao, H., Cooper, D., Keutmann, M., Gur, R., Nenkova, A., and Verma, R. Crema-d: Crowd-sourced emotional multimodal actors dataset. IEEE transactions on affective computing, 5:377–390, 10 2014. doi: 10.1109/TAFFC. 2014.2336244.  
Chen, H., Xie, W., Vedaldi, A., and Zisserman, A. Vggsound: A large-scale audio-visual dataset. ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 721–725,

2020. URL https://api.semanticscholar. org/CorpusID:216522760.

Chen, P., Liu, S., Zhao, H., and Jia, J. Distilling knowledge via knowledge review. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5006–5015, 2021. doi: 10.1109/CVPR46437.2021. 00497.

Damodaran, B., Kellenberger, B., Flamary, R., Tuia, D., and Courty, N. Deepjdot: Deep joint distribution optimal transport for unsupervised domain adaptation. In Computer Vision – ECCV 2018 - 15th European Conference, 2018, Proceedings, pp. 467–483. Springer, October 2018. ISBN 9783030012243. doi: 10.1007/ 978-3-030-01225-0\ 28. 15th European Conference on Computer Vision, ECCV 2018 ; Conference date: 08-09- 2018 Through 14-09-2018.

Finn, C., Abbeel, P., and Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, 2017. URL https://api.semanticscholar. org/CorpusID:6719686.

Gupta, S., Hoffman, J., and Malik, J. Cross modal distillation for supervision transfer. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2827–2836, 2015. URL https://api. semanticscholar.org/CorpusID:6832420.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2015. URL https: //api.semanticscholar.org/CorpusID: 206594692.

Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network, 2015. URL https: //arxiv.org/abs/1503.02531.

Huang, T., You, S., Wang, F., Qian, C., and Xu, C. Knowledge distillation from a stronger teacher. ArXiv, abs/2205.10536, 2022. URL https: //api.semanticscholar.org/CorpusID: 248986690.

Huo, F., Xu, W., Guo, J., Wang, H., and Guo, S. C2kd: Bridging the modality gap for cross-modal knowledge distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16006– 16015, 2024. doi: 10.1109/CVPR52733.2024.01515.

Kim, S., Xiao, R., Georgescu, M.-I., Alaniz, S., and Akata, Z. Cosmos: Cross-modality self-distillation for vision language pre-training. 2025 IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14690–14700, 2024. URL https: //api.semanticscholar.org/CorpusID: 274436198.  
Koltchinskii, V. and Panchenko, D. Rademacher processes and bounding the risk of function learning. In Gine, E., ´ Mason, D. M., and Wellner, J. A. (eds.), High Dimensional Probability II, pp. 443–457, Boston, MA, 2000. Birkhauser Boston. ISBN 978-1-4612-1358-1.¨  
Lee, P., Kim, T., Shim, M., Wee, D., and Byun, H. Decomposed cross-modal distillation for rgb-based temporal action detection. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2373– 2383, 2023. doi: 10.1109/CVPR52729.2023.00235.  
Li, L. and JIN, Z. Shadow knowledge distillation: Bridging offline and online knowledge transfer. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum? id=prQT0gN81oG.  
Liu, T., Lam, K.-M., Zhao, R., and Qiu, G. Deep cross-modal representation learning and distillation for illumination-invariant pedestrian detection. IEEE Transactions on Circuits and Systems for Video Technology, 32 (1):315–329, 2022. doi: 10.1109/TCSVT.2021.3060162.  
Liu, X., LI, L., Li, C., and Yao, A. NORM: Knowledge distillation via n-to-one representation matching. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=CRNwGauQpb6.  
Livingstone, S. R. and Russo, F. A. The ryerson audiovisual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english. PLoS ONE, 13(5):e0196391, 2018. doi: 10.1371/journal.pone.0196391.  
Lv, J., Yang, H., and Li, P. Wasserstein distance rivals kullback-leibler divergence for knowledge distillation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https: //openreview.net/forum?id=1qfdCAXn6K.  
Menon, A. K., Rawat, A. S., Reddi, S., Kim, S., and Kumar, S. A statistical perspective on distillation. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 7632–7642. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/ v139/menon21a.html.  
Mohri, M., Rostamizadeh, A., and Talwalkar, A. Foundations of Machine Learning. Adaptive Computation and Machine Learning series. MIT Press, 2012. ISBN 9780262018258. URL https://books.google. com.vn/books?id=maz6AQAAQBAJ.  
Nguyen, C. V., Hassner, T., Archambeau, C., and Seeger, M. W. Leep: A new measure to evaluate transferability of learned representations. In International Conference on Machine Learning, 2020. URL https://api.semanticscholar. org/CorpusID:211572839.  
Park, W., Kim, D., Lu, Y., and Cho, M. Relational knowledge distillation. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3962–3971, 2019. URL https: //api.semanticscholar.org/CorpusID: 131765296.  
Peng, X., Wei, Y., Deng, A., Wang, D., and Hu, D. Balanced multimodal learning via on-the-fly gradient modulation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8228–8237, 2022. URL https://api.semanticscholar. org/CorpusID:247779156.  
Peyre, G. and Cuturi, M. Computational optimal trans- ´ port, 2020. URL https://arxiv.org/abs/1803. 00567.  
Roheda, S., Riggan, B. S., Krim, H., and Dai, L. Crossmodality distillation: A case for conditional generative adversarial networks. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 2926–2930, 2018. doi: 10.1109/ICASSP. 2018.8462082.  
Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. Fitnets: Hints for thin deep nets. CoRR, abs/1412.6550, 2014. URL https://api. semanticscholar.org/CorpusID:2723173.  
Sarkar, P. and Etemad, A. Xkd: Cross-modal knowledge distillation with domain alignment for video representation learning. ArXiv, abs/2211.13929, 2022. URL https://api.semanticscholar. org/CorpusID:254017839.  
Sun, W., Chen, D., Lyu, S., Chen, G., Chen, C., and Wang, C. Knowledge distillation with refined logits. 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1110–1119, 2024. URL https://api.semanticscholar. org/CorpusID:271865571.  
Tian, Y., Shi, J., Li, B., Duan, Z., and Xu, C. Audio-visual event localization in unconstrained videos. In The European Conference on Computer Vision (ECCV), September 2018.  
Tian, Y., Krishnan, D., and Isola, P. Contrastive representation distillation. In International Conference on Learning Representations, 2020. URL https://openreview. net/forum?id=SkgpBJrtvS.  
Tran, T. K., Dao, M. C., Nguyen, P. L., Truong, T. N., and Hoang, T. N. Rethinking cross-modal fine-tuning: Optimizing the interaction between feature alignment and target fitting. In The 29th International Conference on Artificial Intelligence and Statistics, 2026. URL https: //openreview.net/forum?id=YXPoM9GI12.  
Weed, J. and Bach, F. Sharp asymptotic and finite-sample rates of convergence of empirical measures in wasserstein distance. Bernoulli, 25(4A):2620–2648, 2019.  
Xue, Z., Ren, S., Gao, Z., and Zhao, H. Multimodal knowledge expansion. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 834–843, 2021. doi: 10.1109/ICCV48922.2021.00089.  
Xue, Z., Gao, Z., Ren, S., and Zhao, H. The modality focusing hypothesis: Towards understanding crossmodal knowledge distillation. In International Conference on Learning Representations, 2022. URL https://api.semanticscholar. org/CorpusID:252668904.  
Yang, P., Xie, M.-K., Zong, C.-C., Feng, L., Niu, G., Sugiyama, M., and Huang, S.-J. Multi-label knowledge distillation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17225–17234, 2023. doi: 10.1109/ICCV51070.2023.01584.  
Yun, H., Na, J., and Kim, G. Dense 2d-3d indoor prediction with sound via aligned cross-modal distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7863–7872, 2023.  
Zhao, B., Cui, Q., Song, R., Qiu, Y., and Liang, J. Decoupled knowledge distillation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11943–11952, 2022. doi: 10.1109/ CVPR52688.2022.01165.  
Zhu, Y. and Wang, Y. Student customized knowledge distillation: Bridging the gap between student and teacher. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5037–5046, 2021. doi: 10.1109/ ICCV48922.2021.00501.

## A. Proof for the infinity data points case.

We have the generalized error for the teacher model and the student model as:

$$
\operatorname{err} _ {T} = \mathbb {E} _ {D ^ {T} \left(\boldsymbol {x} ^ {T}, y\right)} \left[ - \log p _ {T} (y \mid \boldsymbol {z} = \theta \left(\boldsymbol {x} ^ {T}\right)) \right] \tag {17}
$$

$$
\operatorname{err} _ {S} = \mathbb {E} _ {D ^ {S} \left(\boldsymbol {x} ^ {S}, y\right)} \left[ - \log p _ {S} \left(y \mid \boldsymbol {z} = \phi \left(\boldsymbol {x} ^ {S}\right)\right) \right] \tag {18}
$$

We denote the joint distributions:

$$
D ^ {T} \left(\theta \left(\boldsymbol {x} ^ {T}\right), y\right) := D ^ {T} (\boldsymbol {z}, y) = D ^ {T} (\boldsymbol {z}) D ^ {T} (y \mid \boldsymbol {z}) \tag {19}
$$

$$
D ^ {S} \left(\phi \left(\boldsymbol {x} ^ {S}\right), y\right) := D ^ {S} (\boldsymbol {z}, y) = D ^ {S} (\boldsymbol {z}) D ^ {S} (y \mid \boldsymbol {z}) \tag {20}
$$

Then, we can rewrite the generalized error of the teacher and the student model as:

$$
\operatorname{err} _ {T} = \mathbb {E} _ {D ^ {T} (\boldsymbol {z}, y)} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] = \mathbb {E} _ {D ^ {T} (\boldsymbol {z})} \mathbb {E} _ {D ^ {T} (y \mid \boldsymbol {z})} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] \tag {21}
$$

$$
\operatorname{err} _ {S} = \mathbb {E} _ {D ^ {S} (\boldsymbol {z}, y)} \left[ - \log \left(p _ {S} (y \mid \boldsymbol {z})\right) \right] = \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \mathbb {E} _ {D ^ {S} (y \mid \boldsymbol {z})} \left[ - \log \left(p _ {S} (y \mid \boldsymbol {z})\right) \right] \tag {22}
$$

We have:

$$
\operatorname{err} _ {S} - \operatorname{err} _ {T} = \mathbf {A} + \mathbf {B} \tag {23}
$$

where:

$$
\mathbf {A} := \quad \operatorname{err} _ {S} - \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \mathbb {E} _ {D ^ {T} (y | \boldsymbol {z})} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] \tag {24}
$$

$$
\mathbf {B} := \quad \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \mathbb {E} _ {D ^ {T} (y | \boldsymbol {z})} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] - \operatorname{err} _ {T} \tag {25}
$$

## 1. Bounding A. We have:

$$
\mathbf {A} = \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \mathbb {E} _ {\mathcal {D} ^ {S} (y | \boldsymbol {z})} \left[ - \log \left(p _ {S} (y \mid \boldsymbol {z})\right) \right] - \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \mathbb {E} _ {D ^ {T} (y | \boldsymbol {z})} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] \tag {26}
$$

$$
= \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \left[ \mathbb {E} _ {D ^ {S} (y | \boldsymbol {z})} \left[ - \log \left(p _ {S} (y \mid \boldsymbol {z})\right) \right] - \mathbb {E} _ {D ^ {T} (y | \boldsymbol {z})} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] \right] \tag {27}
$$

$$
= \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \left[ - \sum_ {y \in \mathcal {Y}} \left(D ^ {S} (y \mid \boldsymbol {z}) \log \left(p _ {S} (y \mid \boldsymbol {z})\right) - D ^ {T} (y \mid \boldsymbol {z}) \log \left(p _ {T} (y \mid \boldsymbol {z})\right)\right) \right] \tag {28}
$$

$$
= \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \mathbb {E} _ {D ^ {S} (y | \boldsymbol {z})} \left[ - \log \left(p _ {S} (y \mid \boldsymbol {z})\right) + \frac {D ^ {T} (y \mid \boldsymbol {z})}{D ^ {S} (y \mid \boldsymbol {z})} \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] \tag {29}
$$

With a mild assumption $D ^ { S } ( y \mid z ) > 0$ , we denote the label transport kernel $\begin{array} { r } { \kappa ( y , z ) \triangleq \frac { D ^ { T } ( y | z ) } { D ^ { S } ( y | z ) } } \end{array}$ DT (y|z)DS (y|z) . Thus, term A then can be expressed as:

$$
\mathbf {A} = \mathbb {E} _ {D ^ {S} (\boldsymbol {z}, y)} \left[ - \log \left(\frac {p _ {S} (y \mid \boldsymbol {z})}{p _ {T} (y \mid \boldsymbol {z}) ^ {\kappa (y , \boldsymbol {z})}}\right) \right] \tag {30}
$$

## 2. Bounding B. We have:

$$
\mathbf {B} = \quad \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \mathbb {E} _ {D ^ {T} (y | \boldsymbol {z})} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] - \mathbb {E} _ {D ^ {T} (\boldsymbol {z})} \mathbb {E} _ {D ^ {T} (y | \boldsymbol {z})} \left[ - \log \left(p _ {T} (y \mid \boldsymbol {z})\right) \right] \tag {31}
$$

$$
= \mathbb {E} _ {D ^ {S} (\boldsymbol {z})} \left[ \ell_ {\tau} (\boldsymbol {z}) \right] - \mathbb {E} _ {D ^ {T} (\boldsymbol {z})} \left[ \ell_ {\tau} (\boldsymbol {z}) \right] \tag {32}
$$

where $\ell _ { \tau } ( z ) \triangleq \mathbb { E } _ { D ^ { T } ( y | z ) } \Big \lceil - \log ( p _ { T } ( y \mid z ) ) \Big \rceil$ is the cross-entropy of the teacher prediction as Definition 2.4. For any cost metric $\delta \in \Delta$ such that $| \ell _ { \tau } ^ { - } ( z _ { 1 } ) - \ell _ { \tau } ( z _ { 2 } ) | \leq ^ { - } \tau _ { \delta } \cdot \delta ( z _ { 1 } , z _ { 2 } )$ , the Kantorovich-Rubinstein duality ascertains that:

$$
\mathbf {B} \leq \tau_ {\delta} \mathbf {W} _ {\delta} \left(D ^ {S} (\boldsymbol {z}), D ^ {T} (\boldsymbol {z})\right) \tag {33}
$$

where $\mathbf { W } _ { \delta }$ denotes the Wasserstein −1 distance with the cost metric δ. Combine Eq. (29) and Eq. (33) we have:

$$
\operatorname{err} _ {S} \leq \operatorname{err} _ {T} + \tau_ {\delta} \mathbf {W} _ {\delta} \left(D ^ {S} (\boldsymbol {z}), D ^ {T} (\boldsymbol {z})\right) + \mathbb {E} _ {D ^ {S} (\boldsymbol {z}, y)} \left[ - \log \left(\frac {p _ {S} (y \mid \boldsymbol {z})}{p _ {T} (y \mid \boldsymbol {z}) ^ {\kappa (y , \boldsymbol {z})}}\right) \right] \tag {34}
$$

Using Definition 2.4 and Definition 2.5, finally, we complete our proof:

$$
\operatorname{err} _ {S} \leq \operatorname{err} _ {T} + \mathbf {F A} (\theta , \phi) + \mathbf {L A} (p _ {S}, p _ {T}) \tag {35}
$$

## B. Proof for the finite data points case.

## 1. Rademacher Bounds.

We start with the Rademacher bound (Koltchinskii & Panchenko, 2000), which is stated as follows.

Rademacher Bounds. Let $\mathcal { F }$ is the family of functions mapping from Z to [0, 1]. Then for any $0 < \delta < 1$ , with probability at least 1 − δ over sample $S = \{ z _ { 1 } , \cdot \cdot \cdot , z _ { n } \}$ , the following holds for all $f \in { \mathcal { F } } ;$ :

$$
\mathbb {E} [ f ] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} f (z _ {i}) + 2 \mathcal {R} _ {n} (\mathcal {F}) + \sqrt {\frac {\log (1 / \delta)}{2 n}} \tag {36}
$$

$$
\mathbb {E} [ f ] \leq \frac {1}{n} \sum_ {i = 1} ^ {n} f (z _ {i}) + 2 \hat {\mathcal {R}} _ {S} (\mathcal {F}) + 3 \sqrt {\frac {\log (2 / \delta)}{2 n}} \tag {37}
$$

Where $\mathcal { R } _ { n } ( \mathcal { F } )$ and $\hat { \mathcal { R } } _ { S } ( \mathcal { F } )$ are the Rademacher complexity and the empirical Rademacher complexity.

## 2. Bounding Wasserstein distance.

Feature Alignment (FA) is formulated as Wasserstein Distance with the momentum $p = 1$ , cost metric δ, and high dimension $d > 1$ . For clear notation, we introduce two true probability distributions ν and $\mu$ with their empirical distributions $\hat { \nu } _ { n }$ and $\hat { \mu } _ { m }$ which provided by n and m data points, respectively. Using the triangle inequality, the Wasserstein distance term can be expressed as:

$$
\mathbf {W} (\nu , \mu) \leq \mathbf {W} \left(\hat {\nu} _ {n}, \hat {\mu} _ {m}\right) + \mathbf {W} \left(\nu , \hat {\nu} _ {n}\right) + \mathbf {W} \left(\mu , \hat {\mu} _ {m}\right) \tag {38}
$$

Where $\mathbf { W } ( \hat { \nu } _ { n } , \hat { \mu } _ { m } )$ can be seen as the empirical estimation of $\mathbf W (  { \boldsymbol \nu } ,  { \boldsymbol \mu } )$ , which can be directly computed using standard numerical methods. Next, we will explore $\mathbf { W } ( \nu , \hat { \nu } _ { n } )$ term as well as $\mathbf { W } ( \mu , \hat { \mu } _ { m } )$ term in context Wasserstein−1 Distance.

For all $n > 0$ and finite momentum $1 \leq p < \infty$ , (Weed & Bach, 2019) stated that:

$$
\mathbb {P} \left(\mathbf {W} (\nu , \hat {\nu} _ {n}) - \mathbb {E} \left[ \mathbf {W} (\nu , \hat {\nu} _ {n}) \right] \geq t\right) \leq \exp (- 2 n t ^ {2}) \tag {39}
$$

Thus, with the probability at least $1 - \exp ( - 2 n t ^ { 2 } )$ , we have:

$$
\mathbf {W} (\nu , \hat {\nu} _ {n}) \leq \mathbb {E} \left[ \mathbf {W} (\nu , \hat {\nu} _ {n}) \right] + t \tag {40}
$$

We then define $d _ { p } ^ { * } ( \nu )$ as the upper Wasserstein dimensions (Weed & Bach, 2019). Using Theorem 1 in (Weed & Bach, 2019), given the finite momentum $1 \leq p < \infty$ and $s _ { 1 } > d _ { p } ^ { * } ( \nu )$ is the upper Wasserstein dimension, exist a constant $C _ { 1 } > 0$ , such that:

$$
\mathbb {E} \left[ \mathbf {W} (\nu , \hat {\nu} _ {n}) \right] \leq C _ {1} n ^ {- 1 / s _ {1}} \tag {41}
$$

Thus, denote $\delta \triangleq \exp ( - 2 n t ^ { 2 } )$ and $0 < \delta < 1$ , with the probability at least $1 - \delta ,$ , we have:

$$
\mathbf {W} (\nu , \hat {\nu} _ {n}) \leq C _ {1} n ^ {- 1 / s _ {1}} + \sqrt {\frac {\log (2 / \delta)}{2 n}} \tag {42}
$$

Deriving the same steps, with the probability at least $1 - \delta ,$ , we have:

$$
\mathbf {W} (\mu , \hat {\mu} _ {m}) \leq C _ {2} m ^ {- 1 / s _ {2}} + \sqrt {\frac {\log (2 / \delta)}{2 m}} \tag {43}
$$

Finally, with the probability at least 1 − 2δ, the Wasserstein distance can be bounded by:

$$
\mathbf {W} (\nu , \mu) \leq \mathbf {W} \left(\hat {\nu} _ {n}, \hat {\mu} _ {m}\right) + C _ {1} n ^ {- 1 / s _ {1}} + C _ {2} m ^ {- 1 / s _ {2}} + \sqrt {\frac {\log (2 / \delta)}{2 n}} + \sqrt {\frac {\log (2 / \delta)}{2 m}} \tag {44}
$$

where $C _ { 1 } , C _ { 2 } , s _ { 1 } , s _ { 2 }$ are positive constants, $s _ { 1 } , s _ { 2 }$ are larger than the upper Wasserstein dimensions of ν and $\mu ,$ respectively.

## 3. Bounding Label Alignment.

$\begin{array} { r } { f ( z , y ) \triangleq - \log \left( \frac { p _ { S } ( y | z ) } { p _ { T } ( y | z ) ^ { \kappa ( y , z ) } } \right) } \end{array}$

$$
\mathbf {L A} \triangleq \mathbb {E} _ {D ^ {S} (\boldsymbol {z}, y)} \left[ - \log \left(\frac {p _ {S} (y \mid \boldsymbol {z})}{p _ {T} (y \mid \boldsymbol {z}) ^ {\kappa (y , \boldsymbol {z})}}\right) \right] = \mathbb {E} _ {D ^ {S} (\boldsymbol {z}, y)} [ f (\boldsymbol {z}, y) ] \tag {45}
$$

With the mild assumption that the class function $f \in { \mathcal { F } }$ is upper-bounded by a constraint $C _ { 3 } > 0 ,$ , we can scale the function $f$ to [0, 1] by dividing by $C _ { 3 }$ and denote the new class function as $\mathcal { F } / C _ { 3 }$ . Using Rademacher bound (Koltchinskii & Panchenko, 2000), given $0 < \delta < 1$ , with the the probability at least $1 - \delta$ over m provided sample, we have:

$$
\frac {\mathbb {E} [ f ]}{C _ {3}} \leq \frac {\hat {E} [ f ]}{C _ {3}} + 2 \mathcal {R} _ {m} (\mathcal {F} / C _ {3}) + \sqrt {\frac {\log (1 / \delta)}{2 m}} \tag {46}
$$

where $\begin{array} { r } { \hat { E } [ f ] \triangleq \frac { 1 } { m } \sum _ { i = 1 } ^ { m } f ( z _ { i } , y _ { i } ) } \end{array}$ and $\mathcal { R } _ { m } ( \mathcal { F } / C _ { 3 } )$ is Rademacher complexity. By using the property $\alpha \cdot { \mathcal { R } } ( { \mathcal { G } } ) = { \mathcal { R } } ( \alpha \cdot { \mathcal { G } } )$ , we have:

$$
\mathbb {E} [ f ] \leq \hat {E} [ f ] + 2 \mathcal {R} _ {m} (\mathcal {F}) + C _ {3} \sqrt {\frac {\log (1 / \delta)}{2 m}} \tag {47}
$$

Let $\Pi _ { \mathcal { F } } : \mathbb { N }  \mathbb { N }$ be the growth function. Applying Massart’s lemma to $\mathcal { R } _ { m } ( { \mathcal { F } } )$ (Mohri et al., 2012), we have:

$$
\mathcal {R} _ {m} (\mathcal {F}) \leq C _ {3} \sqrt {\frac {2 \log \Pi_ {\mathcal {F}} (m)}{m}} \tag {48}
$$

Let $d \triangleq \operatorname { V C d i m } ( { \mathcal { F } } )$ be the VC dimension of the hypothesis class function ${ \mathcal F } .$ For all $m \in \mathbb { N } ,$ using Sauer’s lemma (Mohri et al., 2012) we have:

$$
\Pi_ {\mathcal {F}} (m) \leq \sum_ {i = 0} ^ {d} \binom {m} {i} \tag {49}
$$

Then, for all $d \leq n$ , we have:

$$
\Pi_ {\mathcal {F}} (m) \leq \left(\frac {e m}{d}\right) ^ {d} \tag {50}
$$

Finally, given $0 < \delta < 1$ , the function class $f$ is upper-bounded by a constraint $C _ { 3 } > 0$ , the V-C dimension $d ,$ with the probability at least $1 - \delta ,$ , we have:

$$
\mathbb {E} [ f ] \leq \hat {E} [ f ] + 2 C _ {3} \sqrt {\frac {2 d \log (m / d)}{m}} + C _ {3} \sqrt {\frac {\log (1 / \delta)}{2 m}} \tag {51}
$$

## 4. Bounding the generalized student error on Offline CMKD.

In the Offline CMKD setting, the teacher error is fixed due to the fixed teacher backbone during the distillation process. We can treat the teacher’s error as the fixed overhead, then combining E.q (44), and $\operatorname { E . q } \left( 4 6 \right)$ , given $0 \leq \delta \leq 1 / 3$ , the teacher and the student empirical distribution $D _ { n _ { T } } ^ { T } ( z )$ and $D _ { n s } ^ { S } ( z )$ provided by $n _ { T }$ and $n _ { S }$ data points respectively. Let the Monte Carlo estimation of the Label Alignment $\mathrm { ( L A ) }$ be $\mathbf { L A } _ { e } ( p _ { S } , p _ { T } ) ,$ , s1 and $s _ { 2 }$ are larger than the upper-bound Wasserstein dimensions (Weed & Bach, 2019) of the student and teacher representation distribution, respectively, with probability at

least 1 − 3δ, we have:

$$
\operatorname{err} _ {S} \leq \quad \operatorname{err} _ {T} + \mathbf {L A} _ {e} \left(p _ {S}, p _ {T}\right) + 2 C _ {3} \sqrt {\frac {2 d \log \left(n _ {S} / d\right)}{n _ {S}}} + C _ {3} \sqrt {\frac {\log (1 / \delta)}{2 n _ {S}}} \tag {52}
$$

$$
+ \tau_ {\delta} \Big (\mathbf {W} \big (D _ {n _ {T}} ^ {T} (\boldsymbol {z}), D _ {n _ {S}} ^ {S} (\boldsymbol {z}) \big) + C _ {1} n _ {S} ^ {- 1 / s _ {1}} + C _ {2} n _ {T} ^ {- 1 / s _ {2}} + \sqrt {\frac {\log (2 / \delta)}{2 n _ {S}}} + \sqrt {\frac {\log (2 / \delta)}{2 n _ {T}}} \Big)
$$

We completed our proof.

## C. Knowledge Distillation additional formulation

In this section, we provide additional detailed formulation about Knowledge Distillation in both common settings: unimodal KD (Section C.1) and cross-modal KD (Section C.2).

## C.1. In-Modal Knowledge Distillation

Formally, we consider the K -classes classification problem where both the teacher model and the student model receive the same input modality X and produce the logit prediction over K classes. Let $h _ { \theta } ( X )$ and $h _ { \phi } ( X )$ be the pre-softmax logit of the teacher model and the student model, respectively. Given a temperature T , we have the softened predictions as

$$
f _ {\theta} (X; T) = \operatorname{softmax} (h _ {\theta} (X) / T) \tag {53}
$$

$$
f _ {\phi} (X; T) = \operatorname{softmax} \left(h _ {\phi} (X) / T\right)
$$

The student model is trained to minimize the weighted combination of cross entropy loss with respect to the ground-truth labels Y and the distillation loss:

$$
\mathcal {L} = \lambda \mathrm{CE} (f _ {\phi} (X), Y) + (1 - \lambda) T ^ {2} \cdot \mathrm{KL} \big (f _ {\theta} (X; T) | | f _ {\phi} (X; T) \big) \tag {54}
$$

where KL is the Kullback–Leibler divergence and $\lambda \in [ 0 , 1 ]$ . The objective of the combined loss function is to encourage the small, simple student model to mimic the behavior of large, complex teacher model, thus enabling the compression of the large models while preserving performance (Hinton et al., 2015).

## C.2. Cross-Modal Knowledge Distillation

Cross-modal Knowledge Distillation generalizes the unimodal framework to heterogeneous modalities, allowing a teacher with access to a stronger modality to guide a student with a weaker one. We consider two modalities, denoted by $X _ { 1 }$ and $X _ { 2 }$ processed by the teacher and student models, respectively, and a single label Y for both. In this setting, $X _ { 1 }$ and $X _ { 2 }$ come from the same instance and have the same label, thus called paired data setting. The combined objective function extends from Eq. (54) as (Liu et al., 2022):

$$
\mathcal {L} = \lambda \mathrm{CE} (f _ {\phi} (X _ {2}), Y) + (1 - \lambda) T ^ {2} \cdot \mathrm{KL} \big (f _ {\theta} (X _ {1}; T) | | f _ {\phi} (X _ {2}; T) \big) \tag {55}
$$

## D. Practical Estimation of the Label Transport Kernel

The theoretical analysis introduces a label-transport kernel $\kappa ( y , z ) \triangleq D ^ { T } ( y \mid z ) / D ^ { S } ( y \mid z )$ to modulate label alignment in the absence of paired samples. In practical implementation, κ is only required under the student conditional $y \sim D ^ { S } ( \cdot \mid z )$ . For supervised classification, the empirical conditional induced by the labeled student dataset is a Dirac distribution ${ \hat { D } } ^ { S } ( y \mid z _ { i } ) = \delta ( y = y _ { i } )$ when given data point $( { \pmb x } _ { i } , y _ { i } )$ ) under the feature map $z _ { i } = \phi ( \pmb { x } _ { i } )$ . Consequently, the kernel is evaluated only at the realized label $y _ { i }$ , such that

$$
\hat {\kappa} _ {i} = \kappa (y _ {i}, \boldsymbol {z} _ {i}) = \frac {\hat {D} ^ {T} (y _ {i} \mid \boldsymbol {z} _ {i})}{\hat {D} ^ {S} (y _ {i} \mid \boldsymbol {z} _ {i})} = \hat {D} ^ {T} (y _ {i} \mid \boldsymbol {z} _ {i}) \tag {56}
$$

Given a good pre-trained teacher model, we further adopt a plug-in estimator ${ \hat { D } } ^ { T } ( y _ { i } \mid z _ { i } ) = p _ { T } ( y _ { i } \mid z _ { i } )$ (pseudo label sampling (Nguyen et al., 2020)). Therefore, $\hat { \kappa } _ { i } = p _ { T } ( y _ { i } \mid z _ { i } )$ acts as a teacher–student label-compatibility score: when the teacher assigns low probability to the student’s ground-truth label, distillation is downweighted to mitigate negative transfer.

Table 9. Per-epoch training time comparison between UCMKD and Feature-based Knowledge Distillation. The relative overhead is computed as the ratio between the training time of UCMKD and Feature Distill.

<table><tr><td rowspan="2">Method</td><td colspan="2">AVE</td><td colspan="2">CREMA-D</td><td colspan="2">RAVDESS</td><td colspan="2">VGGSound</td></tr><tr><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td></tr><tr><td>UCMKD</td><td>31.8s</td><td>44.13s</td><td>34.8s</td><td>63.6s</td><td>103.3s</td><td>83.8s</td><td>748.2s</td><td>692.3s</td></tr><tr><td>Feature KD</td><td>14.7s</td><td>16.3s</td><td>22.5s</td><td>31.9s</td><td>59.4s</td><td>53.0s</td><td>624.8s</td><td>237.9s</td></tr><tr><td>Relative overhead</td><td>2.16×</td><td>2.76×</td><td>1.54×</td><td>1.99×</td><td>1.73×</td><td>1.58×</td><td>1.20×</td><td>2.91×</td></tr></table>

Table 10. Scalability evaluation using a ViT-based architecture with ViT-L (ViT-L/16, 300M+ parameters) as the teacher and ViT-S (patch16-224, 22M parameters) as the student.

<table><tr><td rowspan="2">Method</td><td colspan="2">AVE</td><td colspan="2">RAVDESS</td></tr><tr><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td></tr><tr><td>Teacher</td><td>72.64</td><td>76.87</td><td>95.50</td><td>90.57</td></tr><tr><td>CE</td><td>51.19</td><td>53.73</td><td>65.63</td><td>66.13</td></tr><tr><td>Feature KD</td><td>56.47</td><td>54.48</td><td>74.53</td><td>62.24</td></tr><tr><td>UCMKD</td><td>59.45</td><td>61.19</td><td>88.42</td><td>77.42</td></tr></table>

## E. Complexity Analysis

As discussed in Section 6, the bilevel optimization in UCMKD introduces additional training cost. However, this overhead mainly comes from a constant-factor increase in the number of student forward/backward passes. It does not introduce a new dependence on the number of training samples beyond the cost already required for computing the alignment objective. Therefore, the scaling behavior with respect to dataset size remains comparable to feature-based knowledge distillation baselines.

Empirical training cost. We first report the per-epoch training time of UCMKD and Feature-based KD in Table 9. Across datasets and transfer directions, UCMKD increases the per-epoch training time by a moderate constant factor, ranging from $1 . 2 0 \times \mathrm { t o } 2 . 9 1 \times$ . This overhead is acceptable given the performance gains over existing baselines reported in Tables 2 and 1.

Theoretical complexity. We now analyze the training complexity of UCMKD compared with feature-based KD. Let N denote the number of training samples. We assume the same student and teacher backbones across methods. Let $F _ { s }$ and $F _ { t }$ denote the forward costs of the student and teacher models, respectively, and let $B _ { s }$ denote the backward cost of the student model. Let $W ( N )$ denote the cost of computing the Wasserstein distance, $\mathrm { e . g . }$ , using Sinkhorn iterations with entropic regularization (Peyre & Cuturi ´ , 2020). For feature-based KD, the per-epoch cost can be written as

$$
C _ {\mathrm{FKD}} = N (F _ {s} + F _ {t} + B _ {s}) + W (N). \tag {57}
$$

For UCMKD, we perform one feature-alignment(FA) step and one label-alignment(LA) step per iteration, as used in our experiments. This requires additional student forward/backward passes, while the teacher only needs to be evaluated once. The resulting cost is

$$
C _ {\mathrm{UCMKD}} = 3 N (F _ {s} + B _ {s}) + N F _ {t} + W (N). \tag {58}
$$

Therefore,

$$
C _ {\mathrm{UCMKD}} <   3 C _ {\mathrm{FKD}}, \tag {59}
$$

This shows that UCMKD introduces at most a constant-factor overhead compared with feature-based $\mathrm { K D , }$ rather than changing the asymptotic dependence on N . Importantly, the optimal transport term $W ( N )$ is preserved rather than multiplied by the bilevel procedure. Thus, the main dataset-size-dependent alignment cost remains unchanged. Consequently, UCMKD preserves the scalability of KD-style training while providing substantially improved performance.

Table 11. Comparison with recent feature-based KD baselines in the unpaired cross-modal setting.

<table><tr><td rowspan="2">Method</td><td colspan="2">AVE</td><td colspan="2">RAVDESS</td><td colspan="2">CREMA-D</td></tr><tr><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td><td>A→V</td><td>V→A</td></tr><tr><td>NORM</td><td>27.68</td><td>50.49</td><td>64.97</td><td>70.12</td><td>70.16</td><td>60.48</td></tr><tr><td>REVIEW</td><td>27.12</td><td>46.27</td><td>62.61</td><td>68.42</td><td>69.76</td><td>60.62</td></tr><tr><td>UCMKD</td><td>34.16</td><td>52.24</td><td>73.83</td><td>74.43</td><td>71.64</td><td>66.67</td></tr></table>

Table 12. Hyperparameter configurations for multimodal datasets AVE, CREMA-D, RAVDESS, VGGsound. FA epoch and LA epoch denote the number of epochs minimizing Feature Alignment(FA) and Label Alignment(LA) in a single distillation epoch. λ1 and λ2 are the weights of Feature Alignment loss and Label Alignment loss in Algorithm 1.

<table><tr><td>Hyperparameter</td><td>AVE</td><td>CREMA-D</td><td>RAVDESS</td><td>VGGsound</td></tr><tr><td>Backbone</td><td>ResNet-18</td><td>ResNet-18</td><td>ResNet-18</td><td>ResNet-18</td></tr><tr><td>Batch size</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Epoch</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Optimizer</td><td>SGD</td><td>SGD</td><td>SGD</td><td>SGD</td></tr><tr><td>Learning rate</td><td>1e-2</td><td>1e-2</td><td>1e-2</td><td>1e-2</td></tr><tr><td> $\lambda_1$ </td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td> $\lambda_2$ </td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>FA epoch</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>LA epoch</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></table>

## F. Implementation detail and Hyperparameters

In this section, we provide more implementation details and hyperparameters to reproduce our empirical results. To ensure fair comparisons, we adopt the same hyperparameters for all compared baselines. The specific hyperparameters are provided in Table 12. All experiments are run on an NVIDIA RTX A6000 GPU, and results are averaged over 5 independent runs. Our official implementation can be found at https://github.com/Duckduck-05/UCMKD.

## G. Additional Experimental Results and Ablation Studies

In this section, we present detailed experimental results, including standard deviations, and additional ablation studies of the proposed method that support the empirical analysis in Section 4.

Scalability with larger backbones. Table 13 reports the prediction accuracy on RAVDESS and CREMA-D using ResNet-50 as the backbone, further demonstrating that our method is not tied to a specific architecture and remains effective under a stronger convolutional network. Across the four transfer directions, our method achieves the best performance in three cases, showing consistent robustness across datasets and modality-transfer settings. To further evaluate scalability in more realistic settings, we conduct additional experiments using a ViT-based architecture (ViT-L as the teacher and ViT-S as the student), which is representative of modern large-scale vision models. As shown in Table 10, UCMKD consistently achieves the best student performance across all datasets and transfer directions. These results show that UCMKD remains effective when moving from ResNet-based backbones to substantially larger ViT-based architectures. Together with the complexity analysis (See Appendix E), this provides further evidence that our approach is scalable in realistic settings.

Compare with modern feature-based KD methods. We further evaluate UCMKD against two recent feature-based knowledge distillation methods, REVIEW (Chen et al., 2021) and NORM (Liu et al., 2023). Although these methods were originally designed for paired data knowledge distillation, they can be adapted to the unpaired cross-modal setting and serve as strong feature-based baselines, as discussed in Section 1. As shown in Table 11, UCMKD consistently outperforms both REVIEW and NORM across all datasets and transfer directions. These results further demonstrate that simply applying modern feature-based KD objectives is insufficient for the unpaired cross-modal distillation problem.

Robustness under distributional mismatch. We further evaluate UCMKD under increasingly challenging unpaired settings with distributional mismatch between the teacher and student modalities. Specifically, we consider three levels of difficulty: Easy, which follows the random permutation setting used in the main text; Medium, which introduces marginal mismatch by sampling teacher and student modalities from disjoint sample pools; and Hard, which further incorporates modality-specific noise and label imbalance between the teacher and student datasets. As shown in Table 14, both CE and Feature KD suffer substantial performance degradation as the mismatch becomes stronger. In contrast, UCMKD shows a more gradual decline and consistently achieves the best performance across all settings and transfer directions. These results demonstrate that UCMKD is more robust to marginal mismatch, domain shift, and label imbalance in realistic unpaired cross-modal distillation scenarios.

Table 13. Prediction accuracy with standard deviation on RAVDESS and CREMA-D with backbone ResNet-50 across different unpaired setting baselines: Cross Entropy, Feature KD, our method, and paired setting Vanilla KD (Hinton et al., 2015). Our method achieves the best performance on 3 out of 4 tasks.

<table><tr><td rowspan="2">Method</td><td colspan="2">RAVDESS</td><td colspan="2">CREMA-D</td></tr><tr><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td><td> $A \rightarrow V$ </td><td> $V \rightarrow A$ </td></tr><tr><td>Teacher</td><td>63.64</td><td>70.13</td><td>65.46</td><td>74.06</td></tr><tr><td>CE</td><td>52.25 ± 0.8</td><td>57.58 ± 2.9</td><td>71.33 ± 1.4</td><td>61.16 ± 0.4</td></tr><tr><td>Feature KD</td><td>50.25 ± 0.9</td><td>72.03 ± 2.9</td><td>69.71 ± 0.2</td><td>62.37 ± 0.2</td></tr><tr><td>Vanilla KD</td><td>66.67 ± 2.86</td><td>72.60 ± 1.56</td><td>73.20 ± 1.65</td><td>62.54 ± 0.17</td></tr><tr><td>Ours</td><td>70.83 ± 4.2</td><td>74.13 ± 3.5</td><td>72.36 ± 1.2</td><td>66.94 ± 0.9</td></tr></table>

Table 14. Performance under increasingly challenging unpaired settings with marginal mismatch, domain shift, and label imbalance on RAVDESS.

<table><tr><td rowspan="2">Method</td><td colspan="3">RAVDESS A→V</td><td colspan="3">RAVDESS V→A</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>Easy</td><td>Medium</td><td>Hard</td></tr><tr><td>Feature KD</td><td>65.37</td><td>56.34</td><td>54.55</td><td>69.80</td><td>42.86</td><td>39.06</td></tr><tr><td>CE</td><td>65.47</td><td>61.04</td><td>51.45</td><td>70.66</td><td>63.24</td><td>44.96</td></tr><tr><td>UCMKD</td><td>73.83</td><td>67.64</td><td>64.23</td><td>74.43</td><td>67.93</td><td>51.56</td></tr></table>