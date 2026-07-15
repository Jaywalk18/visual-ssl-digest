# Label-Free Target-Domain Adaptation for Unconstrained Event-Image Feature Matching via Dual-Stage Distillation

Zhonghua Yi<sup>1</sup>, Hao Shi<sup>4,1</sup>, Qi Jiang<sup>1</sup>, Yufan Zhang<sup>3</sup>, Kailun Yang<sup>2</sup>, and Kaiwei Wang<sup>1,∗</sup> <sup>1</sup>Zhejiang University, <sup>2</sup>Hunan University, <sup>3</sup>National University of Defense Technology, <sup>4</sup>Ant Group

## Abstract

Building pixel-level correspondence between event and image data is a fundamental task for multi-sensor systems. However, existing cross-modal matching methods are largely restricted by their re liance on either matching labels or strictly aligned hardware, which limits them to unlabeled and unconstrained real-world scenarios where neither matching ground truth nor prior sensor relationships are available. To address this, we propose a novel two-stage training paradigm. First, we leverage large-scale data to perform label-agnostic distillation pretraining, upgrading optimization ob jectives with distribution-based and contrastive losses to learn highly generalizable representations. Second, to tackle unlabeled and unconstrained downstream data, we introduce an epipolar guided self-distillation framework. By utilizing consistency ver ification to isolate robust matches and incorporating geometric confidence derived from an external epipolar prior, our model can efectively self-evolve directly on target domains without any supervision. Furthermore, we introduce a rigorous cross-modal evalua tion benchmark based on TUM-VIE, featuring physically separated cameras with distinct intrinsic parameters and resolutions. Exten sive experiments demonstrate that our proposed method achieves state-of-the-art performance on both MVSEC and TUM-VIE pose estimation tasks. The source code and benchmark will be made pub licly available at https://github.com/ZhonghuaYi/nexus2-oficial.

## Keywords

Event Cameras, Distillation, Cross-modal Feature Matching

## 1 Introduction

Cross-modal feature matching [11, 23, 30] has recently attracted significant research interest, focusing on establishing pixel-level correspondences between multi-modal data. Among these tasks event-image feature matching [37] is particularly challenging. Due to the unique nature of event cameras, their data is represented as asynchronous point clouds in 3D spatiotemporal space, making it significantly more dificult to match with standard 2D images.

Unlabeled and unconstrained multi-sensor systems [42] are ubiq uitous in real-world applications, since they bypass the need for expensive tracking equipment, especially in distributed sensor networks where relative sensor poses are inherently dynamic and unfixed. As illustrated in Fig. 1(a), our target scenario focuses on these physically decoupled, heterogeneous sensors with unknown spatial relationships. In such environments, the model must learn to establish correspondences on target data without access to any ground-truth matching labels through pose and depth (unlabeled), or predefined sensor relationships (unconstrained).

However, previous methods struggle to generalize to such setups and generally fall into two categories. The first category, shown in Fig. 1(b), relies on large-scale end-to-end training using synthetic multimodal data with matching labels. These methods require image datasets with known camera poses and pixel-wise depth to synthesize large amounts of multimodal data, thereby generalizing to event-image matching [23]. The second category, such as EI-Nexus [37] (Fig. 1(c)), relaxes the need for synthetic labels by distilling knowledge from aligned event-image pairs. However, these alignment-dependent approaches are fundamentally bottlenecked by their reliance on specialized hardware (e.g., coaxial DAVIS cameras) to provide pixel-perfect spatial correspondence. Consequently, they cannot adapt to target downstream datasets where such strict hardware alignment is unavailable.

To bridge this gap, we propose a two-stage training paradigm (Fig. 1(d)) that enables model optimization on unlabeled and unconstrained event-image data. In the pretraining stage, while sharing a similar distillation philosophy with EI-Nexus, we significantly enhance the model’s foundational robustness. Specifically, we scale up the training to a massive corpus of real-world data and upgrade the distillation objectives from simple pixel regression to a local score distribution loss and a contrastive descriptor loss. This ensures the model learns highly generalizable cross-modal features rather than overfitting to specific spatial configurations.

Our most significant departure from prior work lies in the second stage: epipolar-guided self-distillation. Unlike EI-Nexus, which is restricted to its pretraining environment, our framework allows the model to continuously self-evolve directly on unaligned target domains. We employ a teacher-student framework initialized with the pretrained model. We feed homography-warped, challenging cross-modal pairs into the student network while providing the original, unwarped event-image pairs to the teacher network. Then we perform consistency verification between the teacher’s and student’s matching predictions to extract high-consistency matches. Furthermore, by incorporating an epipolar geometric prior, we calculate geometric confidences based on the teacher’s matches. This confidence score guides the self-distillation process, encouraging the network to focus more on matching predictions that strictly conform to the underlying epipolar geometry. This self-distillation strategy significantly boosts the pretrained model’s performance on downstream data, entirely without the need for matching labels.

Furthermore, existing evaluation datasets for event-image matching sufer from notable limitations. They are typically based on DAVIS cameras [21, 41], where events and frames share identical camera intrinsics, resolutions, and optical axes, or they inappropriately evaluate 2D homography estimation on 3D scenes with parallax [23], violating basic geometric constraints. Consequently, these datasets fail to accurately reflect feature matching performance in genuine multi-sensor systems. To address this critical gap, we construct a novel and rigorous evaluation benchmark based on the TUM-VIE dataset [14]. Unlike previous setups, it features spatially ofset event and frame cameras with heterogeneous properties (e.g., distinct intrinsics and resolutions), serving as a realistic platform for unlabeled and unconstrained multi-sensor applications.

![](images/85c0adbcfbba356f2c9b61a205f078c6fd0279b0708656406268273d599f4c83.jpg)  
Figure 1: (a) In label-free event-image feature matching, unknown spatial relationships and absent geometric priors prevent obtaining matching labels. (b) Supervised synthesis paradigms rely on simulated data and existing matching labels, rendering them incapable of label-free fine-tuning. (c) Alignment-dependent approaches fully rely on pixel-aligned data, failing to adapt to target datasets where such hardware relations are unavailable. (d) Our framework first performs label-agnostic pretraining for robust zero-shot features. Driven by self-distillation, the model could self-evolve directly on the unlabeled target domain.

Extensive experiments on both traditional aligned and physi cally decoupled datasets demonstrate that our framework achieves state-of-the-art performance across MVSEC [37] and TUM-VIE pose estimation benchmarks. Notably, our label-agnostic pretrained model exhibits superior zero-shot generalization, outperforming supervised baselines on the MVSEC dataset. Furthermore, our self distillation efectively bridges severe domain gaps in unlabeled and unconstrained systems and yields a significant performance boost, achieving up to 21<sup>.</sup>9% improvement on MVSEC and 16<sup>.</sup>6% on the TUM-VIE benchmark compared to our zero-shot foundation.

In summary, our key contributions are as follows:

• We propose a novel two-stage training paradigm for event image feature matching, enabling robust model deployment on unconstrained and unlabeled downstream data.

• We introduce an epipolar-guided self-distillation framework that leverages external geometric priors to supervise the network’s self-evolution on unaligned target domains without matching labels.

• We construct a rigorous cross-modal pose estimation bench mark based on the TUM-VIE dataset, featuring physically decoupled sensors with heterogeneous properties.

• Our method achieves state-of-the-art performance on both MVSEC and TUM-VIE datasets, while maintaining a light weight architecture.

## 2 Related Work

## 2.1 Image Feature Matching

Visual feature matching is a fundamental cornerstone of computer vision. Over the past few years, the field has witnessed a paradigm shift from handcrafted features [19, 24] to deep-learningbased architectures. Detector-based methods [10, 22, 39], such as SuperPoint [3], extract sparse keypoints and establish robust correspondences via Mutual Nearest Neighbor (MNN) or Graph Neural Networks [13, 18, 25]. Concurrently, semi-dense matching paradigms [27, 31, 33], estimate sub-pixel matches through a coarse-to-fine scheme. Dense matching methods [5, 6, 16, 32] leverage transformers and dense correlation volumes to achieve remarkable accuracy in texture-less and illumination-variant RGB scenes. However, these single-modal paradigms are inherently restricted to image-to-image matching and fail to bridge the profound domain gap between asynchronous events and static frames.

## 2.2 Cross-Modal Feature Matching

Cross-modal feature matching [11, 34, 36] has long been an active research area, primarily focusing on establishing reliable correspondences between images captured by diferent sensory modalities, such as RGB-IR [2, 30] pairs. Investigating the cross-modal link between events and images is a relatively recent endeavor. The primary obstacle is that asynchronous events lack the structural regularity of standard frames, presenting a formidable challenge.

Early event-tracking approaches often rely on strictly aligned event-image data streams [1, 8, 20]. These methods typically detect keypoints on an initial intensity frame and subsequently track them continuously using event data with high temporal resolution [12, 21]. However, these are not genuine cross-modal feature matching algorithms. Since their operational paradigm fundamentally requires pre-existing, pixel-perfect spatial alignment between events and images to even initialize the tracking process, they do not actually solve the correspondence problem across unknown spatial gaps. Consequently, they are inapplicable to physically de coupled multi-sensor systems with unknown spatial relationships.

To explicitly address unconstrained multi-modal matching, recent learning-based architectures have emerged. Notably, MIN IMA [23] tackles this by employing a multi-modal synthesis engine to generate diverse modality representations from the MegaDepth dataset [17]. By coupling this synthesized data with the original ground-truth correspondence labels, it successfully achieves end to-end supervised training for cross-modal matching. Furthermore, EI-Nexus [37] leverages the pre-established pixel-wise correspon dences inherently provided by strictly aligned event-image pairs. Through knowledge distillation, it aligns an event-based keypoint detector with a pretrained image-based detector, thereby realizing efective event-image feature matching. In contrast to these ap proaches, our work initiates with the integration of generalizable pretraining on large-scale datasets, followed by self-supervised re finement on unlabeled and unconstrained data predicated on the pretrained model, thereby enabling the efective exploitation of downstream data.

## 2.3 Self-supervised Correspondence

Since pixel-level correspondences are dificult to obtain in the real world, self-supervised methods have been an important part of correspondence estimation. RIPE [15] utilizes reinforcement learn ing driven by epipolar geometry rewards on unlabeled image pairs to extract robust keypoints without depth priors. Sonata [35] em ploys self-distillation and spatial masking to prevent networks from relying on low-level geometric shortcuts. In the Tracking Any-Point (TAP) task, BootsTAP [4] refines long-term trajectories using a self-supervised student-teacher framework based on spatial transformation equivariance. GMRW [26] formulates tracking on a space-time graph, training a global matching transformer via contrastive random walks and cycle consistency. TAPNext [40] treats tracking as a causal, self-supervised sequential masked token decoding task, whereas LEAP-Track [38] improves global matching eficiency using curriculum-based sparse attention. While these methods primarily target single-modal or tracking scenarios, we extend the self-supervised paradigm to cross-modal matching by leveraging epipolar geometry to guide the self-distillation process, efectively resolving the spatial misalignment between events and images without external supervision.

## 3 Method

## 3.1 Problem Formulation

Cross-Modal Feature Matching. Given a pair of observations $( X _ { A } , X _ { B } )$ from distinct modalities $M _ { A }$ and $M _ { B }$ capturing the same 3D scene, the goal is to establish a set of pixel-level correspondences $C = \{ ( p _ { A } ^ { i } , p _ { B } ^ { j } ) \}$ . Feature extraction networks $f _ { A } , f _ { B }$ predict keypoint locations $P \in \mathbb { R } ^ { N \times 2 }$ and <sup>??</sup>-dimensional descriptors $D \in \mathbb { R } ^ { N \times d }$ for both modalities. A correspondence is valid $\operatorname { i f } p _ { A } ^ { i }$ and $p _ { B } ^ { j }$ are projec tions of the same 3D point, typically determined by maximizing the similarity $s ( D _ { A } ^ { i } , D _ { B } ^ { j } )$ in a shared latent descriptor space.

Event-Image Matching. We instantiate the modalities as the event domain E and image domain I. Due to hardware heterogeneity, the event <sup>??</sup> and image <sup>??</sup> often possess distinct resolutions and intrinsic parameters. The task is to learn an event feature extractor $f _ { \theta }$ such that predicted event descriptors $D _ { E }$ and image descriptors $D _ { I }$ are modality-invariant. For any matching pair $( p _ { E } , p _ { I } )$ , the objective is:

$$
s (f _ {\theta} (E), g _ {\phi} (I)) \to 1,\tag{1}
$$

where $s ( \cdot , \cdot )$ is a similarity metric $( e . g .$ , cosine similarity).

Unlabeled and Unconstrained Matching Challenge. Traditional optimization of <sup>??</sup> heavily relies on either explicit groundtruth correspondences $C ^ { * }$ (derived from depth and pose T) or strict hardware-level spatial alignment provided by the dataset. However, in real-world distributed or mobile platforms, sensors are often physically decoupled and unconstrained, rendering $C ^ { * }$ and aligned data unavailable. Therefore, our objective is to formulate an optimization strategy for <sup>??</sup> using only unlabeled, spatially unconstrained event-image pairs, leveraging implicit geometric consistency to bridge the massive modality gap without external supervision.

## 3.2 Method Overview

Directly optimizing an event feature extractor $f _ { \theta }$ on unconstrained, unlabeled data is hard due to the massive modality gap and lack of spatial alignment. To address this, we propose a two-stage bootstrapping framework that first establishes a robust baseline and subsequently drives it to self-evolve on target downstream data. Stage 1: Label-Agnostic pretraining (Sec. 3.3). The goal of this stage is to acquire a highly generalizable initialization $\theta _ { p r e }$ . Inspired by EI-Nexus [37], we utilize aligned event-image data to guide the initial cross-modal learning. However, unlike EI-Nexus, which fundamentally assumes aligned hardware is available during target deployment, we strictly confine the use of aligned data to this pretraining phase on a large-scale dataset. Using a frozen image feature extractor $g _ { \phi }$ as a teacher, we distill knowledge into $f _ { \theta }$ via our proposed local score distribution and contrastive descriptor losses. Unlike methods relying on synthetic events, our pretraining utilizes real-world DAVIS recordings, ensuring the baseline captures authentic event dynamics and sensor noise, which is essential for robust zero-shot generalization.

Stage 2: Epipolar-Guided Self-Distillation (Sec. 3.4). Given the initialization $\theta _ { p r e } ,$ this stage adapts the network to specific unconstrained and unlabeled downstream scenarios without matching labels. We employ a teacher-student self-distillation scheme where the student receives homography-augmented inputs. To establish reliable pseudo-labels, we perform cross-modal consistency checks between teacher and student predictions. Furthermore, to prevent confirmation bias and error accumulation, we introduce an epipolar geometric prior. By estimating the fundamental matrix from initial matches, we compute geometric confidence scores that dynamically weight the self-distillation loss. This mechanism acts as a strict geometric filter, forcing the model to adhere to 3D physical constraints during self-evolution in unconstrained environments.

## 3.3 Label-Agnostic pretraining via Distillation

The first stage aims to learn a robust, hardware-agnostic event representation via cross-modal distillation on large-scale aligned data (Fig. 2).

![](images/341e6c7d6f9501bb8295c1cbc6d4e92cda7f68c440a89769e92397d164839f79.jpg)  
Figure 2: Overview of the label-agnostic pretraining. Knowl edge is transferred from the frozen image teacher to the event student.

Data Selection and Quality Filtering. We utilize COESOT [28], which provides large-scale, strictly aligned event-image pairs captured by DAVIS346 cameras. To prevent the student network from learning corrupted pseudo-labels caused by motion blur or HDR artifacts in standard images, we employ a Laplacian variance-based filtering strategy. Pairs with a variance below a specific threshold are excluded to ensure the structural integrity of the sharp edges required for distillation.

Distillation Architecture. We adopt the teacher-student paradigm using a frozen SuperPoint [3] as the image teacher $g _ { \phi } .$ . The student event network $f _ { \theta }$ is designed with a backbone and two heads (score and descriptor) that match the teacher’s output dimensions: a $H _ { c } \times$ $W _ { c } \times 6 5$ local score distribution and a dense descriptor map. This structural alignment enables seamless knowledge transfer from the image domain to the event domain.

Distillation Objectives. Instead of naive pixel-wise regression in EI-Nexus [37], which limits generalization, we introduce three localized distillation losses:

1) Latent Feature Loss ${ \bf \Pi } ( \mathcal { L } _ { f e a t } ) :$ To align intermediate semantic representations, we apply an $L _ { 2 }$ distance between the event and image backbones:

$$
\mathcal {L} _ {f e a t} = \left\| F _ {E} - F _ {I} \right\| _ {2} ^ {2}.\tag{2}
$$

2) Local Score Distribution Loss $( \mathcal { L } _ { s c o r e } ) \mathrm { : }$ To enhance keypoint robustness, we minimize the Kullback-Leibler (KL) divergence be tween the 8 × 8 local probability distributions $S _ { I }$ and $S _ { E } \colon$

$$
\mathcal {L} _ {K L} = \frac {1}{H _ {c} W _ {c}} \sum_ {h, w} D _ {K L} (S _ {I} ^ {(h, w)} \parallel S _ {E} ^ {(h, w)}).\tag{3}
$$

This forces the student to learn relative keypoint likelihoods within local neighborhoods rather than absolute pixel values.

3) Contrastive Descriptor Loss $( \mathcal { L } _ { d e s c } ) { : }$ We employ a contrastive hinge loss to establish a modality-invariant descriptor space. For a positive pair $( d _ { E } , d _ { I } )$ and its corresponding hard negative $d _ { I } ^ { - }$ , the loss is:

$$
\mathcal {L} _ {h i n g e} = 1 - s (d _ {E}, d _ {I}) + \max (0, s (d _ {E}, d _ {I} ^ {-}) - t h _ {n e g}),\tag{4}
$$

where $s ( \cdot , \cdot )$ is cosine similarity. This objective explicitly aligns positive cross-modal pairs $( s \to 1 )$ while repelling hard negatives beyond a margin $t h _ { n e g } ,$ , ensuring high discriminative power.

The total pretraining loss is $\begin{array} { r } { \mathcal { L } _ { p r e } = \mathcal { L } _ { f e a t } + \mathcal { L } _ { K L } + \mathcal { L } _ { h i n g e } . } \end{array}$

## 3.4 Epipolar-Guided Self-Distillation

To adapt the sub-optimal baseline model to downstream scenarios, where multi-modal data is completely unconstrained and unlabeled, we propose an epipolar-guided self-distillation framework. Our core insight is to drive the network to self-evolve by extracting high-quality pseudo ground-truth matches through asymmetric augmentations and geometric consistency verification. The whole self-distillation framework is shown in Fig. 3.

Framework and Notation. We adopt a teacher-student architecture. Let the original image coordinate system be <sup>??</sup> , and the original event coordinate system be <sup>??</sup>. To construct challenging asymmetric inputs, we apply a random homography H to the event stream <sup>??</sup>, yielding a warped stream $E ^ { H } = \{ ( \mathbf { p } _ { k } ^ { H } , t _ { k } , p _ { k } ) \}$ , where $\mathbf { p } _ { k } ^ { H }$ is the warped event coordinates, calculated via perspective projection: $\mathbf { p } _ { k } ^ { H } = \mathrm { p r o j } ( \mathbf { H } \mathbf { p } _ { k } )$ .

Crucially, the image branches for both the teacher and the student are initialized with the SuperPoint weights and kept strictly frozen $( g _ { \phi } )$ , ensuring the image feature space serves as a highly stable and consistent anchor. Let $\theta _ { t }$ denote the parameters of the teacher’s event branch, which are updated via Exponential Moving Average (EMA) [29] from the student parameters <sup>??</sup>. The forward passes for the teacher network, processing the unwarped original data, are formulated as:

$$
(^ {I} S ^ {t}, ^ {I} D ^ {t}) = g _ {\phi} (I), (^ {E} S ^ {t}, ^ {E} D ^ {t}) = f _ {\theta_ {t}} (E).\tag{5}
$$

For the student network, its image branch processes the same original image to yield outputs $( ^ { I } S ^ { s } , { } ^ { I } D ^ { s } )$ . However, its event branch processes the augmented event stream to generate predictions strictly within the transformed coordinate system $E ^ { H }$ :

$$
\left(^ {E ^ {H}} S ^ {s}, ^ {E ^ {H}} D ^ {s}\right) = f _ {\theta} \left(E ^ {H}\right).\tag{6}
$$

By extracting sparse keypoints from the score maps and interpolating their corresponding descriptors from the descriptor maps, we obtain the complete sets of image and event keypoints. For the teacher network, we define these extracted sets as $\mathcal { K } _ { I } ^ { t }$ and $\mathcal { K } _ { E } ^ { t }$ :

$$
\mathcal {K} _ {I} ^ {t} = \left\{\left(^ {I} P _ {i} ^ {t}, ^ {I} S _ {i} ^ {t}, ^ {I} d _ {i} ^ {t}\right) \right\} _ {i = 1} ^ {n _ {t} ^ {I}}, \quad \mathcal {K} _ {E} ^ {t} = \left\{\left(^ {E} P _ {i} ^ {t}, ^ {E} S _ {i} ^ {t}, ^ {E} d _ {i} ^ {t}\right) \right\} _ {i = 1} ^ {n _ {t} ^ {E}}.\tag{7}
$$

These two sets are then fed into a parameter-free Mutual Nearest Neighbor (MNN) matcher. We formulate this matching process as a function that yields the teacher’s initial cross-modal match set $M ^ { t } ;$

$$
\mathcal {M} ^ {t} = \mathrm{MNN} (\mathcal {K} _ {E} ^ {t}, \mathcal {K} _ {I} ^ {t}) = \left\{\left(^ {E} P _ {k} ^ {t}, ^ {E} S _ {k} ^ {t}, ^ {E} d _ {k} ^ {t}, ^ {I} P _ {k} ^ {t}, ^ {I} S _ {k} ^ {t}, ^ {I} d _ {k} ^ {t}\right) \right\} _ {k = 1} ^ {m _ {t}}.\tag{8}
$$

Similarly, for the student network, we obtain K<sup>??</sup><sub>??</sub> , $\mathcal { K } _ { E ^ { H } } ^ { s }$ and $\mathbf { \nabla } { M ^ { s } } =$ $\mathrm { M N N } ( \mathcal { K } _ { \scriptscriptstyle F H } ^ { s } , \mathcal { K } _ { I } ^ { s } )$ .

Cross-Modal Consistency verification. To construct reliable pseudo-labels, we design a dual-threshold consistency verification between the teacher and student predictions. First, we identify identical matched image keypoints across both networks. Given the disparate inputs furnished to the student and teacher, their resultant matched image keypoints will not achieve perfect congruence. Hence, it is important to identify the same image keypoints within the matching predictions derived from both models. We find the nearest neighbor in the student’s image keypoints for a given teacher’s image point ${ } ^ { I } P _ { i } ^ { t }$ and require their distance to be less than a threshold $t h _ { i }$ . If this spatial consistency holds, we assume the student’s image point inherits the stable features of ${ } ^ { I } P _ { i } ^ { t }$

![](images/83696e0dd73e22e06a25e5e6a5e26c9198384db43028a3d5e6c8237124fcc91e.jpg)  
Figure 3: Epipolar-guided self-distillation procedure. The pretrained model is copied as the teacher and student network, while the teacher is updated through EMA. Student receives homography-augmented event inputs. The predicted matches from Student and Teacher are further verified by checking consistency. The Teacher’s prediction was further calculated using the epipolar constraint to obtain geometric confidence, guiding the self-distillation process.

Next, we map the teacher’s matched event keypoint $^ E P _ { i } ^ { t }$ from the <sup>??</sup> coordinate system to the student’s $E ^ { H }$ coordinate system using the known homography matrix H:

$$
{ } ^ { E ^ { H } } P _ { i } ^ { t \rightarrow s } = \mathbf { H } \cdot { } ^ { E } P _ { i } ^ { t } .\tag{9}
$$

We then check if this projected point aligns with the student’s matched event point $E ^ { \star } { } _ { P _ { i } ^ { s } }$ by requiring their distance to be less than $t h _ { e }$ . If this spatial consensus is met, we establish $^ { E ^ { H } } P _ { i } ^ { t  s }$ as a high-quality pseudo ground-truth event matching point for the corresponding image keypoint in the student’s transformed space. Epipolar Geometric Confidence. Relying solely on network out put consistency introduces the risk of error accumulation (confirma tion bias) as the student evolves. To mitigate this, we introduce an external, purely geometric filter. Without relying on specific camera intrinsics, we apply RANSAC [7] on the teacher’s overall matching results to estimate a Fundamental matrix F. For each matched pair, we calculate its epipolar distance <sup>??????</sup> relative to F. A smaller epipo lar distance indicates a higher likelihood that the match conforms to a rigid 3D scene structure. We convert this error into a normalized geometric confidence score $C _ { i } = e x p ( - e r r _ { i } ) \in ( 0 , 1 ]$

Self-Supervised Distillation Objectives. Finally, we calculate the cross-modal distillation loss by enforcing the student’s event predictions at the pseudo-ground-truth location $^ { E ^ { H } } P _ { i } ^ { t  s }$ to mimic the corresponding stable image outputs.

Unlike the pretraining stage, the downstream unaligned data lacks a spatial alignment prior. Consequently, formulating a KL divergence over local spatial distributions is ill-posed. Instead, we apply a direct regression loss for the scores:

$$
\mathcal {L} _ {i} ^ {s c o r e} = \big \| ^ {E ^ {H}} S ^ {s} \big (^ {E ^ {H}} P _ {i} ^ {t \to s}) - ^ {I} S _ {i} ^ {t} \big \| _ {2}.\tag{10}
$$

For the descriptors, we employ a contrastive hinge loss. The pseudoground-truth match serves as the positive pair, while the most similar non-corresponding image descriptor $\cdot ^ { I } \dot { d } _ { j } ^ { t }$ serves as the mined hard negative:

$$
\begin{array}{l} \mathcal {L} _ {i} ^ {d e s c} = \left(1 - ^ {E ^ {H}} D ^ {s} (^ {E ^ {H}} P _ {i} ^ {t \to s}) \cdot {} ^ {I} d _ {i} ^ {t}\right) \\ \quad + \max \left(0, ^ {E ^ {H}} D ^ {s} (^ {E ^ {H}} P _ {i} ^ {t \to s}) \cdot {} ^ {I} d _ {j} ^ {t} - t h _ {n e g}\right). \end{array}\tag{11}
$$

The total self-supervised loss is dynamically weighted by the geometric confidence, forcing the network to focus primarily on matches that adhere to the underlying epipolar geometry:

$$
\mathcal {L} = \sum_ {i = 1} ^ {N} C _ {i} \bigg (\mathcal {L} _ {i} ^ {s c o r e} + \mathcal {L} _ {i} ^ {d e s c} \bigg).\tag{12}
$$

## 4 Experiments

## 4.1 TUM-VIE E↔I Matching Benchmark

Existing event-image matching benchmarks sufer from two major limitations: (1) relying on synthetic 2D homographies [9, 23] that ignore scene depth and lack genuine 3D parallax, or (2) utilizing idealized DAVIS sensors [37, 41] where modalities share identical intrinsics and coaxial axes, failing to reflect modern hardware heterogeneity. To address these gaps, we introduce the TUM-VIE Event-Image Matching Benchmark, a rigorous testbed based on the TUM-VIE dataset [14]. It features a physically decoupled, heterogeneous stereo rig with two standard cameras (1024 × 1024) and two high-resolution Prophesee event cameras (1280 × 720). Due to spatial ofsets and distinct intrinsic properties, the modalities are naturally misaligned, providing a highly realistic environment with complex 3D parallax.

Table 1: Quantitative evaluation of cross-modal relative pose estimation on the MVSEC and TUM-VIE pose estimation benchmarks. Modality denotes whether the method performs pseudo Image-to-Image (I↔I, using event-to-image conversion) or direct Event-to-Image (E↔I) matching. “pretrain” and “ft.” represent our label-agnostic pre-training and self-supervised fine-tuning, respectively. Best results are highlighted in bold.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Modality</td><td rowspan="2">Data</td><td rowspan="2">Sensor Relation</td><td rowspan="2">Scheme</td><td colspan="3">MVSEC AUC@</td><td colspan="3">TUM-VIE AUC@</td></tr><tr><td> $5^{\circ}$ </td><td> $10^{\circ}$ </td><td> $20^{\circ}$ </td><td> $5^{\circ}$ </td><td> $10^{\circ}$ </td><td> $20^{\circ}$ </td></tr><tr><td>SuperPoint [3]</td><td>I↔I</td><td>Label-Free</td><td>Unconstrained</td><td>Self-Supervised</td><td>0.31</td><td>1.50</td><td>5.08</td><td>0.00</td><td>0.06</td><td>0.22</td></tr><tr><td>LoFTR (outdoor) [27]</td><td>I↔I</td><td>Label-Required</td><td>Unconstrained</td><td>Supervised</td><td>0.23</td><td>0.77</td><td>2.94</td><td>0.08</td><td>0.19</td><td>0.51</td></tr><tr><td>LoFTR (indoor)</td><td>I↔I</td><td>Label-Required</td><td>Unconstrained</td><td>Supervised</td><td>0.00</td><td>0.16</td><td>0.47</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoMa (outdoor) [6]</td><td>I↔I</td><td>Label-Required</td><td>Unconstrained</td><td>Supervised</td><td>0.71</td><td>3.34</td><td>8.01</td><td>0.92</td><td>4.09</td><td>10.88</td></tr><tr><td>RoMa (indoor)</td><td>I↔I</td><td>Label-Required</td><td>Unconstrained</td><td>Supervised</td><td>0.64</td><td>2.19</td><td>6.77</td><td>0.14</td><td>1.04</td><td>4.87</td></tr><tr><td>MINIMA (LG) [23]</td><td>E↔I</td><td>Label-Required</td><td>Unconstrained</td><td>Supervised</td><td>1.15</td><td>3.20</td><td>7.78</td><td>1.07</td><td>5.25</td><td>13.90</td></tr><tr><td>MINIMA (LoFTR)</td><td>E↔I</td><td>Label-Required</td><td>Unconstrained</td><td>Supervised</td><td>2.19</td><td>6.95</td><td>14.47</td><td>0.83</td><td>3.50</td><td>9.83</td></tr><tr><td>MINIMA (RoMa)</td><td>E↔I</td><td>Label-Required</td><td>Unconstrained</td><td>Supervised</td><td>1.22</td><td>3.98</td><td>9.80</td><td>4.38</td><td>13.83</td><td>26.10</td></tr><tr><td>EI-Nexus (MVSEC ver.) [37]</td><td>E↔I</td><td>Label-Free</td><td>Aligned</td><td>Distillation</td><td>3.79</td><td>11.98</td><td>24.24</td><td>3.72</td><td>12.18</td><td>24.66</td></tr><tr><td>EI-Nexus (EC ver.)</td><td>E↔I</td><td>Label-Free</td><td>Aligned</td><td>Distillation</td><td>2.55</td><td>8.48</td><td>18.70</td><td>2.19</td><td>8.22</td><td>17.02</td></tr><tr><td>Ours (pretrain)</td><td>E↔I</td><td>Label-Free</td><td>Aligned</td><td>Distillation</td><td>4.47</td><td>13.18</td><td>26.44</td><td>3.46</td><td>12.63</td><td>24.49</td></tr><tr><td>Ours (MVSEC ft.)</td><td>E↔I</td><td>Label-Free</td><td>Unconstrained</td><td>Self-Distillation</td><td>5.63</td><td>16.70</td><td>32.23</td><td>1.98</td><td>7.86</td><td>17.15</td></tr><tr><td>Ours (TUM-VIE ft.)</td><td>E↔I</td><td>Label-Free</td><td>Unconstrained</td><td>Self-Distillation</td><td>4.04</td><td>12.25</td><td>25.33</td><td>4.77</td><td>15.35</td><td>28.56</td></tr></table>

We formulate the evaluation as a cross-modal relative pose (ex trinsic) estimation task. We select 4 challenging sequences recorded under two distinct calibration configurations, encompassing a total of 8 unique cross-modal extrinsic transformations. By randomly sampling 200 timestamps, we construct a suite of 800 pose estimation tasks. By evaluating recovered extrinsics against precise ground-truth annotations, this benchmark definitively measures matching robustness under genuine hardware heterogeneity and varied spatial alignments.

## 4.2 Experimental Setup

4.2.1 Baselines. To comprehensively evaluate our method, we select representative state-of-the-art baselines across three distinct paradigms: Single-Modal Image Matching Models: To assess the zeroshot generalization of standard image matchers on cross-modal tasks, we convert the asynchronous events into pseudo-images. Specifically, we map the polarity of the latest triggered event at each pixel into an RGB format. We evaluate a diverse spectrum of models: SuperPoint [3] (MNN as matcher) representing detectorbased methods, LoFTR [27] representing semi-dense matchers, and RoMa [6] representing dense matchers.

MINIMA [23]: It is a recent robust cross-modal matching framework. It synthesizes heterogeneous modalities on the MegaDepth dataset and strictly relies on ground-truth matching labels for training. We report results using its provided LightGlue (LG) [18], LoFTR, and RoMa variants. Crucially, due to its absolute reliance on precise matching labels, MINIMA cannot be fine-tuned or adapted on our unlabeled downstream event-image data.

EI-Nexus [37]: It focuses specifically on event-image matching by distilling knowledge from an image detector (SuperPoint) to an event detector. We evaluate its oficially released models trained on the MVSEC [41] and EC [21] datasets and use MNN as the matcher. However, EI-Nexus fundamentally assumes that the event and image streams are densely and perfectly aligned during training. Consequently, it is incapable of training on unconstrained downstream data where spatial relationships are unknown.

4.2.2 Datasets. COESOT [28]. We utilize the COESOT dataset for pretraining. While originally designed for object tracking, it provides a massive corpus of roughly aligned event streams and intensity frames. By applying our Laplacian variance filter (thresh old set to 100) to discard scenes with severe blur, we extract approximately 170k high-quality event-image pairs for our model’s zero-shot cross-modal generalization.

MVSEC [41]. We utilize MVSEC as one of our downstream datasets without access to matching labels, following the evaluation pipeline as EI-Nexus [37]. It provides 346 × 260 aligned event-image data. TUM-VIE. As introduced in Sec. 4.1, it features spatially ofset sensors with high-resolution disparities (1280 × 720 for events vs. 1024 × 1024 for images). We strictly reserve 4 complex sequences exclusively for our new evaluation benchmark, while the remaining unannotated and unaligned sequences serve as the target domain data for our self-supervised fine-tuning.

4.2.3 Implementation Details. All experiments are implemented in PyTorch and conducted on an NVIDIA RTX 4090 GPU.

Architecture and Input. To process the asynchronous events, the input point cloud is discretized into a 16-channel voxel grid. This is concatenated with a 1-channel binary event mask to form a 17- channel input tensor. Our event branch is a lightweight VGG-based architecture with only 1<sup>.</sup>48<sup>??</sup> parameters, comprising a backbone, a score head, and a descriptor head.

In the pretraining stage, we use a frozen SuperPoint [3] as the image keypoint extractor. The contrastive margin <sup>??</sup> (for $t h _ { n e g } )$ is set to 0<sup>.</sup>2. We optimize the student network using the Adam optimizer with a learning rate of $5 \times 1 0 ^ { - 4 }$ for 20 epochs. To ensure training eficiency and focus, supervision is applied exclusively to spatial pixels where events occurred, efectively filtering out superfluous background information.

During self-distillation, the student is initialized with pretrained weights, while the teacher is updated via EMA. We randomly sample pairs from distinct timestamps within a sequence to simulate unaligned downstream data. The Adam optimizer is employed with a learning rate of $1 \times 1 0 ^ { - 4 }$ . For the consistency check, we set the image domain threshold $t h _ { i } = 1 . 5$ pixels and the event-domain projection threshold $t h _ { e } ~ = ~ 5 . 0$ pixels, balancing geometric strictness with pseudo-label density. For dataset-specific adaptation, we fine-tune on MVSEC for 5 epochs (full resolution, EMA decay 0<sup>.</sup>999) and on TUM-VIE for 2 epochs (512 × 512 random crops, EMA decay 0<sup>.</sup>9999). Batch sizes are set to 32 in pretraining and 8 in self-distillation.

![](images/081387fa87009d43a84cca21ee209483bc5f47c95f55d3d00da00b20be30172e.jpg)  
Figure 4: Qualitative results comparison on the TUM-VIE validation dataset. Our pretrained model achieves better pose estimation results than previous methods without accessing the data in the target domain. After self-distillation training in the target domain, our model achieves self-evolving.

## 4.3 Main Results

The quantitative evaluation of relative pose estimation is summa rized in Tab. 1. The visualization results on TUM-VIE are shown in Fig. 4. We report the Area Under the Curve (AUC) of the pose error at thresholds of 5<sup>◦</sup>, 10<sup>◦</sup>, and $2 0 ^ { \circ }$ . Our analysis focuses on addressing the fundamental challenges of cross-modal matching and validating the efectiveness of our proposed two-stage framework.

Inefectiveness of Direct Modality Translation. As shown in the top section of Tab. 1, converting events into pseudo-images to utilize single-modal matchers (I↔I) yields critically poor performance across both datasets. For instance, the advanced RoMa (outdoor) model achieves only 8<sup>.</sup>01 AUC@20<sup>◦</sup> on MVSEC and 10<sup>.</sup>88 on TUM-VIE. This demonstrates that direct event-image methods discard essential spatiotemporal dynamics, rendering standard image domain knowledge inefective under severe modality gaps. Native cross-modal representation learning (E↔I) is strictly necessary. Comparison with Label-Supervised and Alignment-Dependent Baselines. Our method demonstrates significant superiority by lifting restrictive training assumptions. First, the supervised MINIMA series sufers from severe domain overfitting due to its reliance on explicit matching labels. For example, MINIMA-RoMa’s performance collapses from 26<sup>.</sup>10 on TUM-VIE to 9<sup>.</sup>80 on MVSEC, revealing a critical vulnerability to distinct sensor setups.

In addition, we compare against EI-Nexus, which relaxes label requirements but depends strictly on aligned event-image pairs for distillation. While our pretrained model already outperforms EI-Nexus on MVSEC (26<sup>.</sup>44 vs. 24<sup>.</sup>24 AUC@20<sup>◦</sup>) through more generalizable objectives, EI-Nexus’s primary bottleneck is its inability to fine-tune on unaligned downstream data. In contrast, our self-distillation is entirely alignment-agnostic, enabling direct optimization on unconstrained target data. This adaptation successfully bridges hardware-induced domain gaps, elevating our performance to 32<sup>.</sup>23 on MVSEC and 28<sup>.</sup>56 on TUM-VIE. Ultimately, our 1<sup>.</sup>48<sup>??</sup> lightweight model achieves state-of-the-art accuracy without requiring matching labels or strict hardware alignment during downstream training.

Domain Shift and the Necessity of Self-Distillation. The crossdataset evaluation results prove the efectiveness of our two-stage design paradigm. While our pretrained model exhibits robust zeroshot generalization, the necessity of dataset-specific adaptation becomes evident when facing severe hardware-induced domain shifts. For example, when our model is fine-tuned on MVSEC (MVSEC ft.), it surges to state-of-the-art levels on MVSEC (32<sup>.</sup>23 AUC@20<sup>◦</sup>, ↑ 21<sup>.</sup>9% improvement), but its performance degrades when tested directly on TUM-VIE (17<sup>.</sup>15). This performance drop across disparate multi-sensor systems is a universal challenge, stemming from distinct sensor intrinsics, noise profiles, and uncalibrated spatial ofsets. This explicitly justifies the core motivation of our Stage 2 epipolarguided self-distillation. Because our adaptation scheme is entirely label-free and alignment-agnostic, we can seamlessly deploy and fine-tune it directly on the target unlabeled and unconstrained data. By doing so (TUM-VIE ft.), the network successfully bridges this specific hardware gap, driving the AUC@20<sup>◦</sup> from 24<sup>.</sup>49 up to 28<sup>.</sup>56 (↑ 16<sup>.</sup>6% improvement) and securing the top performance.

## 4.4 Ablation Studies

We conduct extensive ablation studies on the MVSEC dataset to verify our architectural designs and training strategies.

pretraining Configurations. Tab. 2 confirms the efectiveness of our Stage 1 design. Using baseline (EI-Nexus) objectives achieves the lowest generalization. Upgrading from naive regression $( L _ { r e g } )$ to distribution-based KL divergence $( L _ { K L } )$ and contrastive hinge loss $( L _ { h i n g e } )$ significantly boosts generalization by establishing a more discriminative cross-modal feature space. Furthermore, data quality is critical: utilizing “Filtered” COESOT data instead of the “Full” unfiltered set increases performance from 22<sup>.</sup>04 to 26<sup>.</sup>44 AUC@20<sup>◦</sup>. This demonstrates that removing motion-blurred and textureless pairs is essential for stable distillation.

Table 2: Ablation study on the impact of data filtering and various combinations of distillation objectives during pretraining. Best results are highlighted in bold.

<table><tr><td></td><td>Data</td><td> $L^{score}$ </td><td> $L^{desc}$ </td><td> $L_{feat}$ </td><td>AUC@5</td><td>AUC@10</td><td>AUC@20</td></tr><tr><td>Baseline</td><td>Filtered</td><td> $L_{reg}$ </td><td> $L_{reg}$ </td><td>√</td><td>3.11</td><td>8.80</td><td>20.39</td></tr><tr><td>1</td><td>Filtered</td><td> $L_{KL}$ </td><td> $L_{reg}$ </td><td>√</td><td>3.26</td><td>10.51</td><td>21.50</td></tr><tr><td>2</td><td>Filtered</td><td> $L_{KL}$ </td><td> $L_{hinge}$ </td><td>×</td><td>3.29</td><td>11.19</td><td>24.40</td></tr><tr><td>3</td><td>Full</td><td> $L_{KL}$ </td><td> $L_{hinge}$ </td><td>√</td><td>3.20</td><td>10.52</td><td>22.04</td></tr><tr><td>4</td><td>Filtered</td><td> $L_{KL}$ </td><td> $L_{hinge}$ </td><td>√</td><td>4.47</td><td>13.18</td><td>26.44</td></tr></table>

![](images/46a78ecd65c4965c664b87567d61f43654bc33cd8126111d9372d36acba3fc92.jpg)

Figure 5: Ablation study on self-distillation against various variants, including distilling without confidence weighting, and weighting based purely on descriptor similarity.  
![](images/51666904cb0fa23ec7282c2efa52c31d77a752b2869b1de956c971e661f67d82.jpg)  
Figure 6: Threshold sensitivity during self-distillation.

Self-Distillation Strategies and Confidence Metrics. We evalu ate three settings in Stage 2 (Fig. 5) to verify our design: (1) w/o Conf., self-distillation without any confidence weighting; (2) Desc. Sim. Conf., weighting labels via teacher-predicted descriptor similarity; and (3) Full paradigm, our final method with epipolar-guided confi dence. Results show that naive distillation (w/o Conf.) improves over the pretrained baseline, demonstrating the efectiveness of our base self-distillation strategy. In addition, relying on the teacher’s own descriptor similarity (Desc. Sim. Conf.) severely degrades performance (30<sup>.</sup>1 AUC@20<sup>◦</sup>). This reveals a critical circular dependency: using the network’s internal features to validate its own predictions reinforces confirmation bias, making it unable to rectify incorrect matches. In contrast, our Full paradigm achieves the highest accuracy of 32<sup>.</sup>2 AUC@20<sup>◦</sup>, by utilizing epipolar geometry as an objective metric.

Sensitivity of Thresholds. Fig. 6 analyzes the dual thresholds in Stage 2. For the image-domain (<sup>??ℎ</sup>?? ), 1<sup>.</sup>5 pixels outperforms a strict 0<sup>.</sup>9 setting, proving that accommodating 8-connected neighborhood quantization errors is vital for robust pseudo-labeling. For the eventdomain $( t h _ { e } )$ , we observe a fundamental trade-of between pseudolabel quality and quantity. Stringent thresholds $( t h _ { e } \textless 5 )$ starve the student network of supervisory signals, while overly relaxed settings $( t h _ { e } ~ > ~ 5 )$ introduce geometric noise. The performance peaks at $5 . 0 ,$ which optimally balances geometric strictness with the volume of valid training data.

## 5 Conclusion

In this paper, we presented a novel two-stage training paradigm for event-to-image feature matching, specifically designed for unconstrained and unlabeled multi-sensor systems. By combining label-agnostic pretraining with epipolar-guided self-distillation, our framework efectively eliminates the long-standing dependence on expensive matching labels and strict hardware alignment. To provide a more realistic evaluation, we established a rigorous benchmark based on the TUM-VIE dataset, featuring genuine 3D parallax and heterogeneous sensor properties. Extensive experiments demonstrate that our lightweight model not only exhibits strong zero-shot generalization but also successfully self-evolves on unla beled target domains, achieving state-of-the-art performance across MVSEC and TUM-VIE pose estimation tasks.

Despite the promising results, our current approach follows a detector-based paradigm, which relies on sparse keypoint extraction and matching. In the future, we aim to extend this framework to dense matching architectures. Exploring more efective dense cross-modal correspondence will be a key direction to further enhance matching robustness and coverage, particularly in scenarios with repetitive patterns or sparse textures.

## References

[1] Ignacio Alzugaray and Margarita Chli. 2020. Haste: multi-hypothesis asynchro nous speeded-up tracking of events. In 31st British Machine Vision Virtual Conference (BMVC 2020). ETH Zurich, Institute of Robotics and Intelligent Systems, 744.

[2] Yuxin Deng and Jiayi Ma. 2022. ReDFeat: Recoupling detection and description for multimodal feature learning. IEEE Transactions on Image Processing 32 (2022), 591–602.

[3] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. 2018. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 224–236.

[4] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Joao Carreira, et al. 2024. Bootstap: Bootstrapped training for tracking-any-point. In Proceedings of the Asian Conference on Computer Vision. 3257–3274.

[5] Johan Edstedt, Ioannis Athanasiadis, Mårten Wadenbäck, and Michael Felsberg. 2023. DKM: Dense kernelized feature matching for geometry estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 17765–17775.

[6] Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, and Michael Fels berg. 2024. Roma: Robust dense feature matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 19790–19800.

[7] Martin A Fischler and Robert C Bolles. 1981. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM 24, 6 (1981), 381–395.

[8] Daniel Gehrig, Henri Rebecq, Guillermo Gallego, and Davide Scaramuzza. 2020. EKLT: Asynchronous photometric feature tracking using events and frames. International Journal of Computer Vision 128, 3 (2020), 601–618.

[9] Mathias Gehrig, Willem Aarents, Daniel Gehrig, and Davide Scaramuzza. 2021. Dsec: A stereo event camera dataset for driving scenarios. IEEE Robotics and Automation Letters 6, 3 (2021), 4947–4954.

[10] Pierre Gleize, Weiyao Wang, and Matt Feiszli. 2023. Silk: Simple learned keypoints. In Proceedings of the IEEE/CVF international conference on computer vision. 22499– 22508.

[11] Xingyi He, Hao Yu, Sida Peng, Dongli Tan, Zehong Shen, Hujun Bao, and Xiaowei Zhou. 2025. Matchanything: Universal cross-modality image matching with large scale pre-training. arXiv preprint arXiv:2501.07556 (2025).

[12] Javier Hidalgo-Carrió, Guillermo Gallego, and Davide Scaramuzza. 2022. Event aided direct sparse odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5781–5790.

[13] Hanwen Jiang, Arjun Karpur, Bingyi Cao, Qixing Huang, and André Araujo. 2024. Omniglue: Generalizable feature matching with foundation model guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19865–19875.

[14] Simon Klenk, Jason Chui, Nikolaus Demmel, and Daniel Cremers. 2021. Tum vie: The tum stereo visual-inertial event dataset. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 8601–8608.

[15] Johannes Künzel, Anna Hilsmann, and Peter Eisert. 2025. Ripe: Reinforcement learning on unlabeled image pairs for robust keypoint extraction. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4868–4877.

[16] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. 2024. Grounding image matching in 3d with mast3r. In European conference on computer vision. Springer, 71–91.

[17] Zhengqi Li and Noah Snavely. 2018. Megadepth: Learning single-view depth prediction from internet photos. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2041–2050.

[18] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. 2023. Lightglue: Local feature matching at light speed. In Proceedings of the IEEE/CVF international conference on computer vision. 17627–17638.

[19] David G Lowe. 2004. Distinctive image features from scale-invariant keypoints. International journal of computer vision 60, 2 (2004), 91–110.

[20] Nico Messikommer, Carter Fang, Mathias Gehrig, and Davide Scaramuzza. 2023. Data-driven feature tracking for event cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5642–5651.

[21] Elias Mueggler, Henri Rebecq, Guillermo Gallego, Tobi Delbruck, and Davide Scaramuzza. 2017. The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM. The International journal of robotics research 36, 2 (2017), 142–149.

[22] Guilherme Potje, Felipe Cadar, André Araujo, Renato Martins, and Erickson R Nascimento. 2024. Xfeat: Accelerated features for lightweight image matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2682–2691.

[23] Jiangwei Ren, Xingyu Jiang, Zizhuo Li, Dingkang Liang, Xin Zhou, and Xiang Bai. 2025. Minima: Modality invariant image matching. In Proceedings of the Computer Vision and Pattern Recognition Conference. 23059–23068.

[24] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. 2011. ORB: An eficient alternative to SIFT or SURF. In 2011 International conference on computer

vision. Ieee, 2564–2571.

[25] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. 2020. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4938–4947.

[26] Ayush Shrivastava and Andrew Owens. 2024. Self-supervised any-point tracking by contrastive random walks. In European Conference on Computer Vision. Springer, 267–284.

[27] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. 2021. LoFTR: Detector-free local feature matching with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8922–8931.

[28] Chuanming Tang, Xiao Wang, Ju Huang, Bo Jiang, Lin Zhu, Shifeng Chen, Jianlin Zhang, Yaowei Wang, and Yonghong Tian. 2025. Revisiting color-event based tracking: A unified network, dataset, and metric. Pattern Recognition (2025), 112718.

[29] Antti Tarvainen and Harri Valpola. 2017. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems 30 (2017).

[30] Önder Tuzcuoğlu, Aybora Köksal, Buğra Sofu, Sinan Kalkan, and A Aydin Alatan. 2024. Xoftr: Cross-modal feature matching transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4275–4286.

[31] Qing Wang, Jiaming Zhang, Kailun Yang, Kunyu Peng, and Rainer Stiefelhagen. 2022. Matchformer: Interleaving attention in transformers for feature matching. In Proceedings of the Asian conference on computer vision. 2746–2762.

[32] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. 2024. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20697–20709.

[33] Yifan Wang, Xingyi He, Sida Peng, Dongli Tan, and Xiaowei Zhou. 2024. Eficient LoFTR: Semi-dense local feature matching with sparse-like speed. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21666– 21675.

[34] Peihao Wu, Yongxiang Yao, Wenfei Zhang, Dong Wei, Yi Wan, Yansheng Li, and Yongjun Zhang. 2025. Mapglue: Multimodal remote sensing image matching. arXiv preprint arXiv:2503.16185 (2025).

[35] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. 2025. Sonata: Self-supervised learning of reliable point representations. In Proceedings of the Computer Vision and Pattern Recognition Conference. 22193–22204.

[36] Yibin Ye, Xichao Teng, Hongrui Yang, Shuo Chen, Yuli Sun, Yijie Bian, Tao Tan, Zhang Li, and Qifeng Yu. 2025. 3MOS: a multi-source, multi-resolution, and multi-scene optical-SAR dataset with insights for multi-modal image matching. Visual Intelligence 3, 1 (2025), 19.

[37] Zhonghua Yi, Hao Shi, Qi Jiang, Kailun Yang, Ze Wang, Diyang Gu, Yufan Zhang, and Kaiwei Wang. 2025. EI-Nexus: Towards Unmediated and Flexible Inter Modality Local Feature Extraction and Matching for Event-Image Data. In Proceedings of the Winter Conference on Applications of Computer Vision. 1979–1988

[38] Chenzhi Zhao, Wufan Wang, Bo Zhang, and Wendong Wang. 2026. Learning to LEAP: Eficient Dense Point Tracking by Focusing Where It Matters. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. 13108–13116.

[39] Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter CY Chen, Qingsong Xu, and Zhengguo Li. 2023. Aliked: A lighter keypoint and descriptor extraction network via deformable transformation. IEEE Transactions on Instrumentation and Measurement 72 (2023), 1–16

[40] Artem Zholus, Carl Doersch, Yi Yang, Skanda Koppula, Viorica Patraucean, Xu Owen He, Ignacio Rocco, Mehdi SM Sajjadi, Sarath Chandar, and Ross Goroshin. 2025. Tapnext: Tracking any point (tap) as next token prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9693– 9703.

[41] Alex Zihao Zhu, Dinesh Thakur, Tolga Özaslan, Bernd Pfrommer, Vijay Kumar, and Kostas Daniilidis. 2018. The multivehicle stereo event camera dataset: An event camera dataset for 3D perception. IEEE Robotics and Automation Letters 3, 3 (2018), 2032–2039.

[42] David Zuñiga-Noël, Jose-Raul Ruiz-Sarmiento, Ruben Gomez-Ojeda, and Javier Gonzalez-Jimenez. 2019. Automatic multi-sensor extrinsic calibration for mobile robots. IEEE Robotics and Automation Letters 4, 3 (2019), 2862–2869.

## 6 Details of TUM-VIE E↔I Matching Benchmark

To evaluate cross-modal feature matching in physically decoupled and heterogeneous sensor systems, we construct a rigorous bench mark based on the TUM-VIE dataset [14]. This benchmark provides a realistic experimental platform characterized by spatially ofset event and frame cameras with distinct intrinsics, resolutions, and genuine 3D parallax.

Benchmark Structure and Splits. The benchmark follows the two calibration statuses, A and B, provided by the original TUM-VIE dataset, which represent diferent extrinsic configurations between the event and image sensors. We utilize these predefined statuses to evaluate the model’s adaptation to varying hardware setups. As summarized in Table 3, we carefully partition the sequences into training and testing splits for each calibration status:

• Calibration A: We utilize 5 sequences for training, including mocap-desk and the loop-floor series. The test split consists of skate-easy and mocap-desk2.

• Calibration B: This status includes a larger pool of 12 train ing sequences, such as the ofice-maze, running, and bike series. We reserve skate-hard and bike-hard for evaluation.

Evaluation Protocol. By leveraging these diverse sequences across diferent calibration states, our benchmark assesses the model’s ability to establish correspondences under unconstrained spatial ofsets and unlabeled data. The inclusion of various motion patterns (e.g., mocap, skate, and bike) and illumination conditions (e.g., dark sequences) ensures a comprehensive evaluation of feature robustness. This setup allows for a rigorous comparison of self-supervised and label-free matching algorithms in genuine multi-sensor environments.

## 7 Architecture Details of the Event Branch

The event branch is designed as a compact VGG-style encoder tailored for sparse event data, with 1<sup>.</sup>48<sup>??</sup> parameters in total. It aims to extract discriminative features that are spatially aligned with the image branch while maintaining computational eficiency.

## 7.1 Input Representation

Each event stream is first converted into a voxelized spatiotemporal representation. In our default configuration, we utilize 16 temporal bins. To explicitly guide the network toward regions with valid observations, we append a binary event-support mask as an additional input channel. This results in a 17-channel input tensor $\mathbf { V } \in \mathbb { R } ^ { 1 7 \times H \times W }$ . For the pretraining stage, we operate on a single temporal slice to ensure eficient cross-modal alignment.

## 7.2 Backbone Architecture

The backbone is responsible for extracting shared hierarchical features from the input event voxel and mask. As detailed in Table 4, it consists of four convolutional stages that progressively reduce the spatial resolution to 1/8 of the original input.

Stages 1–2: These stages utilize 64 filters with $3 \times 3$ kernels. To rapidly expand the receptive field and reduce computational re dundancy, each stage is followed by $\mathrm { ~ a ~ } 2 \times 2$ max-pooling layer, decreasing the resolution to $H / 2$ and $H / 4 ,$ , respectively.

Table 3: Train and test splits of TUM-VIE event-image matching benchmark. We sample two sequences for test under each calibration status.

<table><tr><td>Calibration</td><td>Train Split</td><td>Test Split</td></tr><tr><td>A</td><td>mocap-deskloop-floor0loop-floor1loop-floor2loop-floor3</td><td>skate-easymocap-desk2</td></tr><tr><td>B</td><td>mocap-1d-transmocap-3d-transmocap-6dofmocap-shakemocap-shake2office-mazerunning-easyrunning-hardfloor2-darkslidebike-easybike-dark</td><td>skate-hardbike-hard</td></tr></table>

Stage 3: This stage increases the channel dimensionality to 128. A final $2 \times 2$ max-pooling layer is applied, resulting in a condensed feature map at $1 / 8$ of the input resolution.

Stage 4: This stage serves as the final feature integrator, consisting of two $3 \times 3$ convolutions and a $1 \times 1$ convolution, all with 128 output channels. It maintains the $H / 8 \times W / 8$ resolution to produce the latent representation F.

## 7.3 Prediction Heads

Two task-specific heads are attached to the refined feature map F to produce keypoint detections and descriptors.

Score Head: This head projects F into a 256-dimensional latent space before mapping it to keypoint logits. The resulting coarse logits map is i $\phantom { + } | \mathop { \mathbb { R } } ^ { 6 5 \times H / 8 \times W / 8 } \phantom { + }$ . Subsequently, the logits map is transformed to score map $\mathbf { S } \in \mathbb { R } ^ { 6 5 \times H / 8 \times W / 8 }$ by applying a softmax transformation across the 65-dimensional vector. Then the last “dustbin” element is discarded, and the resultant scores are reallocated to the original $8 \times 8$ pixel grid. Finally, we apply the input event-support mask to the score map, ensuring that detections are strictly confined to regions with physical event activity. After Non-Maximum Suppression (NMS) and border filtering, we utilize a top-k strategy to extract keypoints.

Descriptor Head: This head transforms F into a dense descriptor map $\mathbf { D } { \bf \bar { \Psi } } \in \ \mathbb { R } ^ { 2 5 6 \times H / 8 \times W / 8 }$ . The descriptors are $L _ { 2 } \cdot$ -normalized. For sparse matching, we sample descriptors at the detected keypoint locations, while the dense map is reserved for downstream crossmodal distillation and fine-tuning.

Table 4: Detailed architectural configuration of the Event Branch. <sup>??</sup> and <sup>??</sup> denote the input image height and width. The architecture is divided into a shared Backbone for feature extraction and two task-specific Heads.

<table><tr><td>Module</td><td>Sub-module / Layer</td><td>Kernel</td><td>Output Channels</td><td>Spatial Scale</td></tr><tr><td>Input</td><td>Event Voxel + Mask</td><td>-</td><td>17</td><td> $H \times W$ </td></tr><tr><td rowspan="9">Backbone</td><td>Stage 1: Conv</td><td> $3 \times 3$ </td><td>64</td><td> $H \times W$ </td></tr><tr><td>Stage 1: Max Pooling</td><td> $2 \times 2$ </td><td>64</td><td> $H/2 \times W/2$ </td></tr><tr><td>Stage 2: Conv</td><td> $3 \times 3$ </td><td>64</td><td> $H/2 \times W/2$ </td></tr><tr><td>Stage 2: Max Pooling</td><td> $2 \times 2$ </td><td>64</td><td> $H/4 \times W/4$ </td></tr><tr><td>Stage 3: Conv</td><td> $3 \times 3$ </td><td>128</td><td> $H/4 \times W/4$ </td></tr><tr><td>Stage 3: Max Pooling</td><td> $2 \times 2$ </td><td>128</td><td> $H/8 \times W/8$ </td></tr><tr><td>Stage 4: Conv</td><td> $3 \times 3$ </td><td>128</td><td> $H/8 \times W/8$ </td></tr><tr><td>Stage 4: Conv</td><td> $3 \times 3$ </td><td>128</td><td> $H/8 \times W/8$ </td></tr><tr><td>Stage 4: Conv</td><td> $1 \times 1$ </td><td>128</td><td> $H/8 \times W/8$ </td></tr><tr><td rowspan="2">Score Head</td><td>Intermediate Latent</td><td> $3 \times 3$ </td><td>256</td><td> $H/8 \times W/8$ </td></tr><tr><td>Logit Projection*</td><td> $1 \times 1$ </td><td>65</td><td> $H/8 \times W/8$ </td></tr><tr><td rowspan="2">Descriptor Head</td><td>Intermediate Latent</td><td> $3 \times 3$ </td><td>256</td><td> $H/8 \times W/8$ </td></tr><tr><td>Dense Descriptor Map</td><td> $1 \times 1$ </td><td>256</td><td> $H/8 \times W/8$ </td></tr></table>

The 65 channels follow the SuperPoint [3] structure (8×8 grid + 1 dustbin), subsequently reshaped to $H \times W .$

## 8 Training Details

## 8.1 Implementation Details for Label-Agnostic Pretraining

8.1.1 Data Processing and Augmentation. We conduct the initial pretraining on a filtered subset of the COESOT dataset [28], retaining only high-quality samples where the event stream and intensity frames are accurately aligned without severe motion blur (filtered via Laplacian variance). For each sample, we extract events within a 100 ms temporal window immediately preceding the timestamp of the corresponding image frame. The event coordinates are normal ized to the image resolution and voxelized into a grid with 16 bins. To handle the sparsity of event data, an additional binary mask channel is incorporated to indicate valid event locations.

To foster geometric robustness while maintaining cross-modal consistency, we apply synchronized geometric augmentations to both modalities. These include random cropping to a 256 × 256 resolution and independent horizontal and vertical flips with a probability of 0<sup>.</sup>5. Consistent with our goal of learning fundamental structural representations, no photometric distortions or event specific point perturbations are applied during this stage.

8.1.2 Architecture and Loss Functions. The model adopts a twobranch architecture. The image branch is initialized with pretrained weights (e.g., SuperPoint [3]) and remains strictly frozen to provide a stable inductive bias. The event branch is optimized to align its output space with the image branch. Both branches extract sparse keypoints and 256-dimensional local descriptors, retaining up to 1024 keypoints per modality for matching. The pretraining is supervised by a composite loss function as described in the main text.

8.1.3 Optimization Hyperparameters. We optimize the network for 20 epochs using the AdamW optimizer. The specific hyperparame ter configurations are detailed in Table 5. The model’s cross-dataset generalization is monitored by evaluating the pose estimation performance on the MVSEC dataset [41] after each epoch.

Table 5: Hyperparameters for label-agnostic pretraining.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base Learning Rate</td><td> $5 \times 10^{-4}$ </td></tr><tr><td>Minimum Learning Rate</td><td> $1 \times 10^{-7}$ </td></tr><tr><td>Learning Rate Schedule</td><td>Cosine Annealing</td></tr><tr><td>Weight Decay</td><td> $1 \times 10^{-5}$ </td></tr><tr><td>Betas ( $\beta_1, \beta_2$ )</td><td>(0.9, 0.999)</td></tr><tr><td>Epsilon ( $\epsilon$ )</td><td> $1 \times 10^{-8}$ </td></tr><tr><td>Effective Batch Size</td><td>32</td></tr><tr><td>Training Epochs</td><td>20</td></tr><tr><td>Max Keypoints (N)</td><td>1024</td></tr><tr><td>Descriptor Dimension</td><td>256</td></tr></table>

## 9 Implementation Details for Self-Distillation

The self-supervised training protocol employs a teacher-student architecture. The student is actively optimized via gradient descent, while the teacher acts as a gradient-free target network updated by an Exponential Moving Average (EMA). To ensure a stable start, EMA updates begin only after a warm-up period of 1000 iterations.

## 9.1 Continuous-Space Homography Augmentation

To cultivate modality-invariant features, we apply an asymmetric, continuous-space homography augmentation exclusively to the student’s event stream, leaving the image unperturbed as a fixed geometric anchor. The planar homography is generated from a centered patch (ratio 0<sup>.</sup>8) incorporating perspective distortion (amplitude 0<sup>.</sup>2), isotropic scaling (0<sup>.</sup>1), and in-plane rotation (up to ±1<sup>.</sup>0 rad). To ensure physically plausible perturbations and avoid resampling artifacts, we enforce a strict field-of-view constraint and apply the transformation directly to the raw, asynchronous event coordinates rather than rasterized frames. Finally, the teacher’s matching predictions are dynamically warped via the forward homography H to generate pseudo-labels directly within the student’s augmented coordinate space, formulating the self-supervision as a rigorous, pixel-perfect consistency evaluation.

## 9.2 Consistency Filtering and Objectives

Because the downstream data lacks ground-truth labels, self-supervision is derived entirely from teacher-student agreement. After mapping the teacher’s predictions into the student’s coordinate frame via H, we enforce a rigorous multi-step filtering pipeline to extract high-confidence pseudo-labels:

(1) Mutual Consistency & Mask Validity: Matches must be mutually consistent across the image and event branches of the teacher. Additionally, the predicted event keypoints are verified against the binary event-support mask to discard detections in unobserved regions.

(2) Spatial Distance Filtering: We enforce spatial proximity constraints between corresponding features. The image-side distance threshold is set to 1<sup>.</sup>5 pixels, while the event-side threshold is 5<sup>.</sup>0 pixels.

(3) Epipolar RANSAC Guidance: We estimate a Fundamental matrix using RANSAC (with an inlier threshold of 1<sup>.</sup>0 pixel and 0<sup>.</sup>999 confidence) on the filtered matches to compute geometric confidence scores, further down-weighting outliers.

The overall optimization objective combines an event-score regression term with a descriptor consistency loss. For the descriptor loss, we employ a positive alignment and a hardest-negative hinge constraint.

## 9.3 Dataset-Specific Configurations (MVSEC vs. TUM-VIE)

While the core framework is shared, several hyperparameters are tailored to the distinct characteristics of the target datasets, as summarized in Table 6.

Table 6: Optimization hyperparameters and dataset-specific configurations for Self-Distillation. Diferences between the MVSEC and TUM-VIE setups are explicitly highlighted.

<table><tr><td>Hyperparameter</td><td>MVSEC Setup</td><td>TUM-VIE Setup</td></tr><tr><td colspan="3">Shared Optimization Parameters</td></tr><tr><td>Optimizer</td><td colspan="2">AdamW</td></tr><tr><td>Base Learning Rate</td><td colspan="2"> $1 \times 10^{-4}$ </td></tr><tr><td>Minimum Learning Rate</td><td colspan="2"> $1 \times 10^{-6}$  (Cosine Annealing)</td></tr><tr><td>Weight Decay &amp; Betas ( $\beta_1, \beta_2$ )</td><td colspan="2"> $1 \times 10^{-4}$  &amp; (0.9, 0.999)</td></tr><tr><td>Gradient Clipping (Max Norm)</td><td colspan="2">1.0</td></tr><tr><td>Mixed Precision</td><td colspan="2">Enabled (AMP)</td></tr><tr><td> $th_i$ </td><td colspan="2">1.5 px</td></tr><tr><td> $th_e$ </td><td colspan="2">5.0 px</td></tr><tr><td>Effective Batch Size</td><td colspan="2">8</td></tr><tr><td>EMA Warm-up Iterations</td><td colspan="2">1000</td></tr><tr><td colspan="3">Dataset-Specific Parameters</td></tr><tr><td>Training Epochs / Total Steps</td><td>5 / ~10k</td><td>2 / ~9k</td></tr><tr><td>EMA Momentum</td><td>0.999</td><td>0.9999</td></tr><tr><td>Event Temporal Slice</td><td>400 ms</td><td>20ms</td></tr><tr><td>Training Resolution</td><td>346 × 260</td><td>512 × 512</td></tr></table>

Justification for Dataset-Specific Parameters. The disparity in training epochs (5 for MVSEC and 2 for TUM-VIE) is explicitly set to ensure a comparable number of total optimization steps (approximately 10k) across diferent dataset sizes. Furthermore, the diference in the event temporal slice (400 ms for MVSEC versus 20 ms for TUM-VIE) is adapted to the inherent event density of each dataset. Scenes in MVSEC typically exhibit lower event densities, requiring a longer accumulation window to form valid structural contours. Conversely, the high event density in TUM-VIE allows a concise 20 ms slice to capture rich spatial details while efectively avoiding motion blur.

Table 7: Impact of pre-training datasets. Under the same training settings, COESOT demonstrates better generalization than MVSEC. Furthermore, our pretraining also scales the generalization performance as the input data increases.

<table><tr><td rowspan="2">Dataset</td><td colspan="3">MVSEC AUC</td><td colspan="3">TUM-VIE AUC</td></tr><tr><td> $5^{\circ}$ </td><td> $10^{\circ}$ </td><td> $20^{\circ}$ </td><td> $5^{\circ}$ </td><td> $10^{\circ}$ </td><td> $20^{\circ}$ </td></tr><tr><td>MVSEC</td><td>4.47</td><td>13.45</td><td>27.14</td><td>2.31</td><td>9.45</td><td>19.66</td></tr><tr><td>Filtered COESOT</td><td>4.47</td><td>13.18</td><td>26.44</td><td>3.46</td><td>12.63</td><td>24.49</td></tr><tr><td>Filtered COESOT + MVSEC</td><td>5.66</td><td>15.44</td><td>28.48</td><td>3.49</td><td>12.41</td><td>24.75</td></tr></table>

## 10 Additional Results

## 10.1 Impact of Pretraining Datasets

In the experimental setup of the main text, we explicitly designate MVSEC and TUM-VIE as the unlabeled and unconstrained downstream target datasets. To strictly evaluate the model’s zero-shot generalization and self-supervised adaptation capabilities in authentic, unseen scenarios, we utilized exclusively the large-scale Filtered COESOT dataset during the pretraining phase. To further investigate the impact of pretraining data distribution and scale on the model’s feature representation capabilities, we provide supplementary comparative experiments in Table 7 using diferent pretraining datasets. Notably, all ablation experiments strictly employ the same loss function configurations (i.e., local score distribution loss and contrastive descriptor loss) and training hyperparameters as described in the main text.

Based on the experimental results in Table 7, we observe a critical dependency on both the diversity and scale of the pretraining corpus. When the model is pretrained exclusively on the scenehomogeneous MVSEC dataset, it achieves competitive performance on the homologous MVSEC test set (reaching an AUC@20<sup>◦</sup> of 27<sup>.</sup>14). However, when directly evaluated on the TUM-VIE dataset—which introduces heterogeneous sensor properties and a significant domain shift—its performance experiences a substantial drop, dropping to an AUC@20<sup>◦</sup> of 19<sup>.</sup>66. In stark contrast, pretraining solely on the Filtered COESOT dataset yields a strong zero-shot score of 24<sup>.</sup>49 on TUM-VIE. This notable diference demonstrates that relying purely on scene-homogeneous data is insuficient to endow the model with cross-domain generalization capabilities, even when optimized with our proposed distillation objectives. Consequently, this corroborates the rationale and necessity of utilizing the diverse COESOT dataset as our foundational pretraining corpus in the main text.

Beyond the critical need for scene diversity, our results demonstrate that scaling the pretraining data efectively further enhances the model’s capability. To evaluate architectural scalability, we constructed a joint pretraining corpus combining Filtered COESOT with MVSEC. This multi-source dataset not only improves the AUC@20<sup>◦</sup> on MVSEC to 28<sup>.</sup>48 but also pushes the zero-shot generalization on the entirely unseen TUM-VIE dataset to a new high of 24<sup>.</sup>75. This consistent performance gain underscores the eficacy of our proposed cross-modal distillation paradigm in absorbing and leveraging diverse data sources. Ultimately, it proves that by continuously scaling up aligned real-world pretraining data, the generalization capacity of the model’s foundational features can be systematically and directly enhanced.

Table 8: Synergy between pretraining and self-distillation. We compare the relative pose estimation performance on MVSEC before (×) and after (<sup>✓</sup>) self-distillation using difer ent pretrained initializations. The results indicate that our self-distillation consistently improves performance, and a stronger pretrained foundational model unlocks a higher upper bound for the final adaptation.

<table><tr><td rowspan="2">Pretraining Dataset</td><td rowspan="2">Self-Distillation</td><td colspan="3">MVSEC AUC</td></tr><tr><td> $5^{\circ}$ </td><td> $10^{\circ}$ </td><td> $20^{\circ}$ </td></tr><tr><td>Filtered COESOT</td><td>×</td><td>4.47</td><td>13.18</td><td>26.44</td></tr><tr><td>Filtered COESOT</td><td>√</td><td>5.63</td><td>16.70</td><td>32.23</td></tr><tr><td>Filtered COESOT + MVSEC</td><td>×</td><td>5.66</td><td>15.44</td><td>28.48</td></tr><tr><td>Filtered COESOT + MVSEC</td><td>√</td><td>5.86</td><td>17.53</td><td>32.87</td></tr></table>

## 10.2 Synergy Between Pretraining and Self-Distillation

In Table 8, we further investigate the relationship between the quality of the pretrained initialization (Stage 1) and the final per formance achieved after applying epipolar-guided self-distillation (Stage 2). We evaluate two distinct pretrained models—one trained exclusively on Filtered COESOT and another trained on the com bined Filtered COESOT + MVSEC corpus. Both models are evaluated on the MVSEC target domain before and after the self-distillation phase.

The results reveal a compelling synergy between our pretraining and self-distillation stages. Most notably, applying the self distillation framework yields substantial and consistent improvements regardless of the initial pretraining dataset. For instance, the baseline model pretrained solely on COESOT improves its AUC@20<sup>◦</sup> from 26<sup>.</sup>44 to 32<sup>.</sup>23, demonstrating the robustness of our label-free self-evolution paradigm in bridging hardware-induced domain gaps. Beyond this consistent enhancement, the experiments demonstrate that a superior initialization unlocks a correspond ingly higher optimization ceiling. When the model is initialized with stronger foundational features (such as those pretrained on the combined COESOT and MVSEC corpus), it naturally begins with a higher zero-shot baseline (28<sup>.</sup>48 vs. 26<sup>.</sup>44). Furthermore, after undergoing the same self-distillation process, it achieves an even higher final performance of 32<sup>.</sup>87 AUC@20<sup>◦</sup>. This constructive syn ergy indicates that our self-distillation framework does not merely saturate at a fixed performance bottleneck; instead, it actively lever ages the richer geometric priors embedded in a superior pretrained model to extract higher-quality pseudo-labels, ultimately pushing the upper bound for downstream adaptation.

## 10.3 Additional Qualitative Results

This section provides extended qualitative evaluations for cross modal pose estimation on the MVSEC (Fig. 7) and TUM-VIE (Fig. 8) datasets. These visualizations further characterize the performance of our framework in unlabeled and unconstrained scenarios.

Geometric Consistency: Sparse vs. Dense Matching. Visual analysis reveals a fundamental performance discrepancy between dense correspondence frameworks (e.g., RoMa [6] and the MINIMAvariant [23]) and sparse matching architectures. While dense matchers generate spatially continuous correspondence fields, their geometric precision often degrades in textureless or motion-blurred regions of the event stream. This leads to a high ratio of nondiscriminative matches that negatively impact the robustness of RANSAC-based pose solvers. Conversely, sparse architectures leverage keypoints distilled from robust image anchors. By prioritizing feature saliency and geometric stability, these sparse models maintain a higher inlier ratio, ensuring more precise relative pose estimation under challenging cross-modal conditions.

Comparative Analysis with Distillation-based Baselines. While our framework shares a distillation-based sparse matching philosophy with EI-Nexus, it ofers distinct advantages in representation learning and domain adaptation. First, the integration of local score distribution and contrastive descriptor losses in the pretraining stage yields a more robust zero-shot foundation than the regressionbased targets utilized in EI-Nexus. The visualizations indicate that our keypoints exhibit superior cross-modal pose estimation performance, even before target-domain fine-tuning.

Second, our framework facilitates a self-distillation phase that addresses the inherent limitations of label-required training. Unlike EI-Nexus, which is constrained by its alignment-required design, our model undergoes unsupervised optimization directly on the target domain that does not provide matching labels or aligned events and frames. By incorporating epipolar geometric constraints into the self-supervision loop, the model efectively accounts for the spatial ofsets in physically decoupled systems. This results in correspondences that are more accurately aligned with the underlying 3D structure and more uniformly distributed across the scene, as evidenced by the improved pose estimation metrics in the qualitative plots.

![](images/68e25abc20b29e6cd9088f010571ee857191ef6072ea7993b066358c6123e922.jpg)  
Figure 8: Qualitative results of pose estimation on the TUM-VIE dataset.