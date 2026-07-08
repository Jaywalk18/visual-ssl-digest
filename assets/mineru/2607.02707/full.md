# VLRC: Vision-Language Reprojection Consistency as a scalable signal for better feed-forward 3D pretraining

Marwane Hariat<sup>1</sup> Gianni Franchi<sup>2</sup> David Filliat<sup>2</sup> Antoine Manzanera<sup>1</sup>\*

<sup>1</sup>U2IS, ENSTA – Institut Polytechnique de Paris, Palaiseau, France <sup>2</sup>Pole Recherche, Agence Minist ˆ erielle pour l’IA de D ´ efense, Palaiseau, France ´ {marwane.hariat, antoine.manzanera, gianni.franchi}@ensta.fr {david.filliat}@polytechnique.edu

![](images/c887b831fa0636521169052b042e888709a108e0bba61836cb7044a1d740aaa6.jpg)  
Figure 1. SS3D+VLRC trained on YouTube8M web videos reconstructs both the exterior (left) and interior (right) of the Sagrada Familia from two casual videos: one recorded outside and one inside. It accurately localizes the entrance in 3D using the prompt (“Show me the entrance of the cathedral”). The reconstruction uses self-supervised estimates of depth, camera pose, and intrinsics. Depth maps and camera trajectories are visualized, each camera being shown along with its corresponding viewpoint. Point clouds are rendered with Open3D.

## Abstract

Feed-forward 3D models are commonly trained using either expensive geometric supervision or self-supervised photometric objectives, both of which provide incomplete learning signals. We introduce Vision-Language Reprojection Consistency (VLRC), a scalable auxiliary objective that exploits frozen vision-language representations as semantic multi-view supervision. Given a predicted 3D reconstruction, VLRC reprojects dense vision-language features across views and enforces feature consistency between corresponding image locations, requiring no additional 3D annotations. The objective integrates seamlessly with both self-supervised monocular reconstruction and supervisedpretrained feed-forward 3D models during unlabeled adaptation. By aligning geometry with language-grounded features, VLRC not only improves depth and camera estimation but also enables more coherent multi-view semantic fusion for open-vocabulary 3D scene understanding. Experiments on indoor and outdoor benchmarks demonstrate consistent gains in 3D reconstruction accuracy and zero-shot openvocabulary 3D semantic segmentation.

## 1. Introduction

Estimating 3D structure from monocular videos is a central problem in computer vision. Recent large-scale feedforward 3D models have shown that neural networks can directly predict rich geometric quantities, including depth, camera motion, intrinsics, and 3D structure. A first line of work relies on large-scale geometric supervision, such as depth, camera pose, intrinsics, or point clouds. Methods such as Depth Anything [28], VGGT [45], π<sup>3</sup> [48], DUSt3R [47], MapAnything [22], and CUT3R [46] leverage large amounts of annotated or reconstructed 3D data, such as ground-truth depth, calibrated multi-view captures, LiDAR scans, synthetic data, or pseudo-labels distilled from stronger reconstruction systems. This form of supervision enables impressive feed-forward 3D prediction across diverse benchmarks, but it has several drawbacks: it remains expensive to obtain, can inherit biases from the datasets, sensors, reconstruction pipelines, or teacher models used to generate it, and is difficult to extend continually as models encounter new in-the-wild video domains.

A second line of work aims to reduce this dependence on explicit 3D supervision through self-supervised learning from monocular video. In particular, SS3D [17] shows that a single feed-forward model can jointly learn depth, camera motion, and intrinsics from large-scale raw web videos using structure-from-motion reprojection losses. Such selfsupervised objectives are attractive because they can scale to unannotated videos, but they are appearance-driven and can become ambiguous under illumination changes, lowtexture regions, repeated patterns, specularities, motion blur, occlusions, and dynamic objects.

Despite their differences, both supervised and selfsupervised feed-forward 3D models rely on incomplete training signals: supervised training is limited by the cost, bias, and limited continual scalability of annotations or pseudo-labels, while self-supervised training is limited by the ambiguity of photometric consistency. This motivates an additional scalable signal that can complement both forms of 3D pretraining.

Vision-language models such as CLIP [30, 37, 56], SigLIP [44, 53], and VL-JEPA [6] provide a promising source of such a signal. By aligning images with language at web scale, they learn visual features enriched with language-grounded semantic priors beyond visual appearance. These features have been widely used for openvocabulary 3D semantic segmentation, but they are typically lifted and fused only after the 3D structure has been estimated. As a result, the quality of the 3D semantic representation depends directly on the quality of the predicted 3D structure: if the 3D prediction is inaccurate, the same physical point may receive inconsistent vision-language features across views, leading to noisy 3D semantics.

Our key observation is that this inconsistency can itself be used as a training signal. If a predicted 3D structure induces inconsistent dense vision-language features across multiple views of the same scene, then the predicted geometry is likely misaligned with the underlying 3D scene structure. Enforcing this consistency provides a scalable signal that complements both supervised and self-supervised 3D pretraining: it does not require ground-truth 3D annotations, and it is less tied to low-level RGB appearance than photometric reprojection, providing useful gradients in ambiguous regions such as textureless surfaces or illuminationvarying areas. In addition, we show that this signal can be integrated into existing unlabeled post-training pipelines, such as SelfEvo-style adaptation [20], where it provides an additional vision-language consistency term for adapting pretrained 3D models to new video domains.

We introduce Vision-Language Reprojection Consistency (VLRC), a general auxiliary objective for large-scale feed-forward 3D pretraining. Given a model that predicts 3D structure from monocular videos, VLRC uses the induced 3D reconstruction to reproject dense visionlanguage-aligned features across views and enforces multiview consistency in feature space. This objective requires no additional 3D annotations. Instead, it reuses frozen vision-language representations as a scalable feature-space signal that complements existing 3D training objectives. VLRC improves 3D learning in two complementary ways. First, it adds a feature-space supervision signal for 3D estimation: the predicted 3D structure must explain not only RGB appearance or available 3D targets, but also the multiview consistency of high-level vision-language features. Second, by enforcing this consistency, VLRC aligns the predicted 3D structure with dense semantic representations, enabling more coherent fusion of 2D vision-language features into 3D. As a result, VLRC improves both core 3D estimation and downstream 3D semantic understanding.

To summarize, our main contributions are:

1. We introduce Vision-Language Reprojection Consistency (VLRC), a general auxiliary objective that uses predicted 3D structure to enforce multi-view consistency of dense vision-language features.

2. We demonstrate that VLRC improves core 3D estimates, including depth, camera motion, intrinsics, and induced 3D reconstruction, across multiple datasets and across both self-supervised and supervised feed-forward 3D settings.

3. We show that VLRC produces 3D structure that is better aligned with dense vision-language features, improving downstream zero-shot open-vocabulary 3D semantic segmentation.

## 2. Related work

Large-scale supervised feed-forward methods. Recent progress in feed-forward 3D estimation has been driven by large-scale supervision from depth, camera calibration, reconstructed point clouds, or teacher-generated pseudolabels. MiDaS [38] showed that mixing heterogeneous labeled depth datasets with scale- and shift-invariant losses yields robust zero-shot relative depth. Depth Anything [28, 52] further scaled monocular depth learning by combining labeled data with pseudo-labels generated on large unlabeled image collections, using strong visual backbones such as DINOv2 [33]. Beyond monocular depth, DUSt3R [47] predicts dense point maps from image pairs using crossview transformer reasoning initialized from CroCo [49], while MASt3R [24] adds dense matching features for im proved correspondence and reconstruction. More recent feed-forward models predict richer 3D structure from multiple views: VGGT [45] jointly predicts cameras, depth, point maps, and tracking features; MapAnything [22] accepts optional geometric inputs such as intrinsics, poses, or depth; and CUT3R [46] introduces a continuous reconstruction formulation with memory for temporally coherent video reconstruction.

These supervised or pseudo-supervised models demonstrate the power of large-scale 3D targets, but they remain dependent on annotated datasets, reconstruction pipelines, sensors, or teacher models. Their predictions may inherit the biases of these sources, and adapting them continually to new in-the-wild video domains can require additional target generation, as observed in SelfEvo [20]. Our work is complementary: rather than introducing a new feed-forward architecture, VLRC provides an auxiliary vision-language reprojection signal that can be added to existing 3D models to improve geometry and support unlabeled adaptation.

Self-supervised and continual feed-forward 3D adaptation. Self-supervised monocular 3D learning estimates depth, camera motion, and intrinsics by synthesizing one frame from another and comparing it to the observed image, following the view-synthesis formulation popularized by Zhou et al. [60]. This removes the need for groundtruth 3D, but the supervision remains primarily photometric and is therefore ambiguous under illumination changes, low texture, repeated patterns, specularities, motion blur, occlusions, and dynamic objects. Many works address specific violations of this assumption: Marsal et al. [32] model brightness changes; Godard et al. [13] use minimum reprojection over multiple source frames to handle occlusions; Li et al. [26] estimate residual 3D motion for dynamic objects; Hariat et al. [14] identify moving regions using discrepancies between optical flow and depth-based reprojection; and Shu et al. [40] and Hariat et al. [15] introduce feature- or contour-based cues to better supervise weakly textured regions.

Although effective, these mechanisms are often specialized to individual failure modes. VLRC instead provides a unified feature-space signal: when predicted depth, pose, or intrinsics induce inconsistent dense vision-language features across views, the model receives a reprojection penalty in a representation space that is less tied to raw RGB appearance. This is motivated by recent evidence that visionlanguage-aligned features are sensitive to multi-view geometric inconsistencies [2]. Since these features are learned from large-scale image-text data, they encode higher-level semantic cues that complement photometric consistency.

A related line of work studies continual learning [10] and self-distillation [5, 18] for annotation-free adaptation.

In feed-forward 3D reconstruction, SelfEvo [20] adapts a pretrained multi-view model such as VGGT on unlabeled videos: a teacher processes the full input sequence and produces stop-gradient reconstruction targets, while a student receives a reduced-context sequence and learns to recover the same reconstruction; the teacher is updated as an exponential moving average of the student. VLRC is complementary to this strategy, providing an additional visionlanguage reprojection signal that can be combined with SelfEvo-style post-training.

Open-vocabulary 3D semantic segmentation. 3D scene understanding has moved from closed-set geometric recognition toward open-vocabulary representations built from large-scale vision-language models. Early 3D recognition methods such as VoteNet [36] reason directly on point clouds using deep point-set features and Hough voting. Recent open-vocabulary methods instead transfer 2D vision-language features into 3D. PLA [8] and Part-SLIP [29] project 3D points into multiple views, extract CLIP features, and aggregate them into point-level representations. PointCLIP [54] learns view aggregation and selection strategies, related to multi-view selection schemes such as [42]. OpenScene [35] fuses multi-view image features and distills them into a 3D network, while later works use image- or region-level descriptions from large visionlanguage models to associate language with 3D points [21, 59]. Casper3D [16] further converts noisy multi-view 2D foundation-model embeddings into latent 3D semantic representations using a Bayesian inverse strategy.

Most of these methods assume reliable geometry, such as ground-truth depth, calibrated poses, intrinsics, or highquality reconstructed point clouds. This assumption is not guaranteed when geometry is predicted by feed-forward 3D models from monocular videos: a reconstruction may be sufficient for RGB reprojection while still producing inconsistent multi-view vision-language features. VLRC addresses this issue during 3D training by encouraging the predicted geometry to align dense vision-language features across views, enabling more coherent 2D-to-3D feature lifting and stronger text-queryable 3D semantic segmentation.

## 3. Method

## 3.1. Problem setup

Let $f _ { \theta }$ be a pretrained feed-forward multi-view reconstruction model

$$
f _ {\theta}: \{I _ {i} \} _ {i = 1} ^ {N} \longrightarrow \{(D _ {i}, T _ {i}, K _ {i}) \} _ {i = 1} ^ {N}
$$

that maps a short clip of N consecutive RGB frames to perframe depth, pose, and intrinsics. We denote:

$D _ { t } ( p ) \in \mathbb { R } _ { + }$ : the predicted depth at pixe $p ,$ ${ { T } _ { t } } \ = \ \left\lceil \begin{array} { c c } { { { R } _ { t } } } & { { { \mathbf { t } } _ { t } } } \\ { { { \mathbf { 0 } } } } & { { 1 } } \end{array} \right\rceil$ : the camera pose to world with $R _ { t } \in$ SO(3) and $\mathbf { t } _ { t } \in \mathbb { R } ^ { 3 }$

$K _ { t } \in \mathbb { R } ^ { 3 \times 3 }$ : the camera intrinsics matrix.

3D structure. These predictions induce a 3D reconstruction of the scene. Let $p = ( u , v , 1 ) ^ { \top }$ be a homogeneous pixel coordinate. A pixel $p$ is first back-projected to 3D in camera coordinates as

$$
X _ {t} (p) = D _ {t} (p) K _ {t} ^ {- 1} p \quad \in \mathbb {R} ^ {3}.\tag{1}
$$

It is then transformed into world coordinates as:

$$
Y _ {t} (p) = R _ {t} X _ {t} (p) + \mathbf {t} _ {t}.\tag{2}
$$

Aggregating such points over frames yields the predicted 3D reconstruction.

Supervised 3D pre-training. In a supervised feedforward 3D setting, the model is trained using ground-truth or pseudo-ground-truth 3D targets. Let $D _ { i } ^ { \star } , T _ { i } ^ { \star }$ , and $K _ { i } ^ { \star }$ denote the target depth, camera pose, and intrinsics for frame i. For notational simplicity, we write the multi-task loss terms using a generic robust discrepancy $\| \cdot \| _ { \epsilon }$ . In practice, the exact form of each term is model-dependent and may include scale-invariant, shift-invariant, normalized, or task-specific variants.

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{3D}} ^ {\sup} = \sum_ {i = 1} ^ {N} \| \mathbf {D} _ {i} - \mathbf {D} _ {i} ^ {\star} \| _ {\epsilon} + \sum_ {i = 1} ^ {N} \| \mathbf {T} _ {i} - \mathbf {T} _ {i} ^ {\star} \| _ {\epsilon} + \sum_ {i = 1} ^ {N} \| \mathbf {K} _ {i} - \mathbf {K} _ {i} ^ {\star} \| \\ = \mathcal {L} _ {\text {depth}} + \mathcal {L} _ {\text {camera}} + \mathcal {L} _ {\text {intr}}. \end{array}\tag{3}
$$

Additional supervised terms, such as point-map or tracking losses, can be included when they are part of the base reconstruction model.

Self-supervised training. In the self-supervised setting, the model uses predicted depth, camera motion, and intrinsics to warp a nearby source frame $I _ { s }$ into the coordinate frame of a target frame $I _ { t }$ . The pixel $p _ { t }$ is first backprojected in the target camera coordinate system as in Eq 1, transformed to the source camera, and then projected into the source image. The resulting sampled source color gives the synthesized target image $\hat { \hat { I } } _ { t }$ , which is compared to the observed target image $I _ { t } \mathbf { : }$

$$
\begin{array}{c} T _ {t \to s} = \left[ \begin{array}{c c} R _ {t \to s} & \mathbf {t} _ {t \to s} \\ \mathbf {0} & 1 \end{array} \right] \in S E (3), \\ X _ {t} (p _ {t}) = D _ {t} (p _ {t}) K _ {t} ^ {- 1} p _ {t}, \\ p _ {s} = \pi \bigg (K _ {s} T _ {t \to s} \left[ \begin{array}{c} X _ {t} (p _ {t}) \\ 1 \end{array} \right] \bigg), \\ \hat {I} _ {t} (p _ {t}) = I _ {s} (p _ {s}), \\ \mathcal {L} _ {\text { photo}} = \sum_ {p _ {t} \in \Omega} \rho \Big (\hat {I} _ {t} (p _ {t}) - I _ {t} (p _ {t}) \Big), \\ \mathcal {L} _ {\text { 3D }} ^ {\text { self }} = \mathcal {L} _ {\text { photo }} + \mathcal {L} _ {\text { reg }} \end{array}\tag{4}
$$

where $\mathcal { L } _ { \mathrm { r e g } }$ denotes regularization terms such as depth smoothness or other model-specific priors.

Here, $T _ { t  s } ~ = ~ T _ { s } ^ { - 1 } T _ { t } ~ \in ~ S E ( 3 )$ is the relative transformation from the target camera to the source camera, $\pi ( \cdot )$ denotes perspective projection, and $I _ { s } ( p _ { s } )$ is obtained with differentiable bilinear sampling. The function $\rho ( \cdot )$ is a robust photometric penalty, typically combining an RGB comparison term with an SSIM comparison term. This objective encourages the predicted depth, pose, and intrinsics to explain the target image through view synthesis in color space.

We use $ { \mathcal { L } } _ { \mathrm { 3 D } }$ to denote the base reconstruction objective, either $\mathcal { L } _ { \mathrm { 3 D } } ^ { \mathrm { s u p } }$ defined in Eq. 3, $\mathcal { L } _ { \mathrm { 3 D } } ^ { \mathrm { s e l f } }$ defined in Eq. 4, or another model-specific objective. In Sec. 4, we instantiate this objective in two settings: self-supervised video pretraining with SS3D [17] and supervised feed-forward reconstruction with VGGT [45].

## 3.2. Vision-Language Reprojection Consistency

The reprojection formulation in Eq. 4 enforces RGB consistency through photometric losses. More generally, the same geometry-induced correspondences can be applied to dense C vision-language-aligned features. We call this auxiliary signal Vision-Language Reprojection Consistency.

Let $F _ { i } = \phi _ { \mathrm { V L M } } ( I _ { i } )$ be a dense vision-language aligned feature map extracted from frame $I _ { i } .$ . Using the same projected source coordinate $p _ { s }$ as in Eq. 4, we synthesize the target feature map as:

$$
\hat {F} _ {t} (p _ {t}) = F _ {s} (p _ {s}).
$$

We then enforce feature-level consistency between $\hat { F } _ { t }$ and $F _ { t }$ , as explained on Fig. 2, yielding the final objective:

$$
\mathcal {L} _ {\mathrm{VLRC}} = \sum_ {(t, s) \in \mathcal {P}} \sum_ {p _ {t} \in \Omega} m _ {t, s} (p _ {t}) \left[ 1 - \cos \Bigl (\hat {F} _ {t} (p _ {t}), F _ {t} (p _ {t}) \Bigr) \right],\tag{5}
$$

$$
\mathcal {L} _ {\mathrm{final}} = \mathcal {L} _ {\mathrm{3D}} + \lambda_ {\mathrm{VLRC}} \mathcal {L} _ {\mathrm{VLRC}},\tag{6}
$$

![](images/76f8a811d0e8280cc5507a873dd48e56bc668501b15d7445bbef55ec7e2edaf4.jpg)  
Figure 2. Reprojection of a point $p \in \mathbb { R } ^ { 3 } \colon$ : the head of a cyclist - The 3d point centered on the head of the cyclist is reprojected on the successive frames and the corresponding Dense VLM features are extracted with a frozen encoder and compared across the reprojected correspondences using a cosine dissimilarity loss. VLRC encourages the predicted geometry to align multi-view language-grounded features consistently across views. P denotes the set of target-source frame pairs used for reprojection.

where Ω is the set of pixels, λ<sub>VLRC</sub> controls the strength of the vision-language reprojection signal. Gradients are back-propagated through the differentiable reprojection operation into the 3D model, while the vision-language encoder remains frozen. Here $m _ { t , i }$ <sub>s</sub> masks invalid projections and unreliable correspondences. Additional details are provided in Sec. 4.1.

## 3.3. Open-Vocabulary 3D Semantic Segmentation

We use the geometry learned with VLRC to lift dense vision-language features into 3D for zero-shot openvocabulary semantic segmentation. As seen in Section 3.1, given a video sequence $\{ I _ { i } \} _ { i = 1 } ^ { n }$ , the reconstruction model predicts depth, camera poses, and intrinsics, which induces a 3D point cloud in world coordinate of cardinality M:

$$
\mathcal {P} = \{P _ {j} \} _ {j = 1} ^ {M}, \qquad P _ {j} \in \mathbb {R} ^ {3}.
$$

In parallel, a dense vision-language encoder $\mathcal { E } ^ { 2 D }$ extracts a pixel-level feature map for each image:

$$
F _ {i} = \mathcal {E} ^ {2 D} (I _ {i}), \qquad F _ {i} (u, v) \in \mathbb {R} ^ {d}.
$$

For each 3D point $P _ { j }$ in world coordinate, we project it into every frame where it is visible. Let $T _ { \mathrm { w }  i }$ denote the transformation from the world coordinate system to camera i, and let K be the camera intrinsics. The projection of $P _ { j }$

into image $I _ { i }$ is given by

$$
\tilde {q} _ {i j} = K   T _ {\mathrm{w} \to i} \left[ \begin{array}{c} P _ {j} \\ 1 \end{array} \right], \qquad q _ {i j} = \pi (\tilde {q} _ {i j}) = (u _ {i j}, v _ {i j}),
$$

We then retrieve the corresponding dense VLM feature by bilinear sampling:

$$
f _ {i j} = F _ {i} (q _ {i j}).
$$

The 3D feature associated with point $P _ { j }$ is obtained by aggregating its multi-view features:

$$
f ^ {3 D} (P _ {j}) = \frac {\sum_ {i = 1} ^ {N} \alpha_ {i j} m _ {i j} f _ {i j}}{\sum_ {i = 1} ^ {N} \alpha_ {i j} m _ {i j}},
$$

where $m _ { i j } ~ \in ~ \{ 0 , 1 \}$ indicates whether $P _ { j }$ is visible and validly projected in frame i, and $\alpha _ { i j }$ is a view-dependent confidence weight.

Open-vocabulary segmentation can be performed in two modes. In the first mode, we assume a predefined set of semantic categories $\mathcal { C } = \{ c _ { k } \} _ { k = 1 } ^ { C }$ . Given a VLM text encoder ${ \mathcal { E } } ^ { T }$ , we compute one text embedding per class,

$$
t _ {k} = \mathcal {E} ^ {T} (\mathrm{prompt} (c _ {k})),
$$

and assign each 3D point to the closest text embedding:

$$
\hat {y} _ {j} = \arg \max _ {k} \cos \bigl (f ^ {3 D} (P _ {j}), t _ {k} \bigr).
$$

This produces a zero-shot semantic label for every point in the predicted reconstruction.

In the second mode, the user provides an arbitrary freeform text query q, such as “where is the Eifel-tower?” in Fig. 3 or “Show me the entrance of the cathedral” in Fig. 1. We encode the query as

$$
t _ {q} = \mathcal {E} ^ {T} (q),
$$

and compute a query relevance score for each 3D point:

$$
s _ {j} (q) = \cos \bigl (f ^ {3 D} (P _ {j}), t _ {q} \bigr).
$$

The resulting scores define a text-conditioned 3D activation heat map. For binary localization, we threshold the scores,

$$
\hat {z} _ {j} (q) = \mathbf {1} [ s _ {j} (q) > \tau ],
$$

where $\tau$ is a similarity threshold. This produces a queryspecific binary segmentation or localization mask in 3D.

Thus, the same fused 3D VLM representation supports both predefined-vocabulary semantic segmentation and free-form text-query localization. For clarity, we present here the simple uniform averaging case, $\alpha _ { i j } \ = \ 1$ for all $i , j$ , following [21]. We also evaluate additional fusion strategies in Sec. 4.

## 4. Experiments

Our experiments are organized around two questions: (i) Does VLRC improve core 3D estimates? (ii) Does the resulting geometry improve multi-view VLM feature fusion for open-vocabulary 3D segmentation?

## 4.1. Implementation Details

We evaluate VLRC in two regimes: self-supervised SS3D fine-tuning and VGGT/SelfEvo-style adaptation. In both cases, VLRC is added as an auxiliary feature-space reprojection loss.

Self-supervised setting. For the self-supervised regime, we add VLRC to SS3D [17], a monocular-video model trained with the photometric SfM objective in Eq. 4 to predict depth, camera motion, and intrinsics. We start from the SS3D checkpoint pretrained on YouTube8M [1] and finetune it on the target datasets using the combined objective in Eq. 6.

Following [14], we compute the validity mask $m _ { t , s }$ in Eq. 5 by comparing depth-pose reprojection correspondences with optical-flow correspondences. Pixels with large discrepancies, typically caused by dynamic objects, occlusions, or unreliable geometry, are excluded from the VLRC loss.

Supervised-pretrained setting. For the supervisedpretrained regime, we start from VGGT-1B [45] and follow the SelfEvo-style unlabeled post-training protocol [20]. A teacher processes the full sequence and provides stopgradient reconstruction targets, while a reduced-context student is trained to match them. We add VLRC to this self-distillation objective. No pixel masking is used in this setting.

Dense VLM features. Unless otherwise stated, we use dense CLIP-derived vision-language features from [30]. The VLM encoder is frozen throughout training. Feature maps are upsampled to the image resolution using Feat-Up [11], and VLRC is computed with cosine dissimilarity between reprojected source features and target-frame features. We provide ablations over the choice of VLM backbone and feature representation in Table 3.

Optimization. During training, images are resized so that their shortest side is 518 pixels, followed by a $5 1 8 \times 5 1 8$ crop. We use $\lambda _ { \mathrm { V L R C } } ~ = ~ 0 . 0 1$ for SS3D fine-tuning and $\lambda _ { \mathrm { V L R C } } = 0 . 1$ for VGGT/SelfEvo-style adaptation. Unless otherwise stated, both regimes are trained for 20 epochs using Adam [23] with $\beta _ { 1 } = 0 . 9 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . The learning rate follows a cosine decay schedule from $1 0 ^ { - 5 } { \mathrm { t o } } 1 0 ^ { - 8 }$ All experiments are implemented in PyTorch [34].

## 4.2. Datasets and Evaluation Protocols

We evaluate VLRC along two axes: core 3D estimation and open-vocabulary 3D semantic segmentation. For self-supervised 3D estimation, we compare SS3D+VLRC against SS3D on depth, camera motion, and intrinsics. Depth is evaluated on KITTI [12] and NYUv2 [41]; camera motion on Sintel [3] and TUM-RGBD [43]; and intrinsics on Sintel. For supervised-pretrained adaptation, we evaluate VGGT/SelfEvo+VLRC on depth using KITTI and Sintel. For open-vocabulary 3D segmentation, we evaluate on ScanNet200 [7] and introduce a KITTI-based zero-shot protocol, described in Appendix A.

## 4.3. Depth Estimation

Tables 1 and 2 show that VLRC improves SS3D depth estimation on both KITTI and NYUv2, outperforming prior self-supervised baselines. Table 4 shows that VLRC also improves SelfEvo-style adaptation of VGGT on Sintel and KITTI, indicating that the signal complements both photometric self-supervision and supervised-pretrained unlabeled adaptation. More results on camera motion and intrinsics are given in Appendix B.

Table 3 studies the feature backbone used in VLRC when fine-tuning SS3D + VLRC on KITTI. CLIP-Seg outperforms DINOv2 and slightly improves over TIPSv2. While TIPSv2 is designed to improve dense patch-text alignment through spatially aware vision-language pretraining, CLIP-Seg builds on CLIP with an explicit dense segmentation decoder.

## 4.4. Open-Vocabulary Semantic Segmentation

Most existing open-vocabulary 3D semantic segmentation protocols rely on ground-truth geometry, making it difficult to evaluate whether a reconstruction model produces geometry that is well aligned with dense VLM features. We therefore evaluate VLRC under two complementary settings.

For the ScanNet200 protocol, we use Casper3D [16] as the downstream 3D semantic representation model. Casper3D fuses multi-view 2D features into a viewinvariant 3D representation. To test the effect of VLRC on geometry-aware feature fusion, we pretrain Casper3D on NYUv2, a dataset visually and geometrically close to ScanNet, using reconstructions from SS3D, SS3D+VLRC, SelfEvo, and SelfEvo+VLRC. We then fine-tune Casper3D on ScanNet following the original protocol. This evaluates whether VLRC-improved geometry leads to stronger downstream 3D semantic representations.

Second, many 3D estimation methods provide checkpoints or reported results on KITTI. We therefore introduce a KITTI-based zero-shot open-vocabulary 3D semantic segmentation protocol. This setting is designed to compare reconstruction models under an identical evaluation pipeline. Each method first predicts a 3D point cloud. Each 3D point is then projected into a fixed number of adjacent frames, where dense CLIP logits are extracted and aggregated across views. We compute CLIP logits using Cityscapes semantic categories as text prompts, matching the label space of the SegFormer pseudo labels. Further details are provided in Appendix A. This protocol directly evaluates the alignment between the predicted geometry and dense CLIP features. Fig. 5 illustrates typical failure cases when VLRC is not used. Without VLRC, the predicted geometry induces less reliable cross-view correspondences, causing dense VLM activations to be fused at inconsistent 3D locations. This produces noisy, fragmented, or misplaced prompt responses in the reconstructed scene. In contrast, adding VLRC encourages the geometry to preserve feature-level consistency across views, resulting in cleaner and more spatially coherent open-vocabulary 3D localization.

Results are shown in Table 5. VLRC improves Scan-Net200 performance under the Casper3D protocol and substantially improves KITTI zero-shot segmentation, indicating better alignment between predicted 3D structure and dense vision-language features.

Qualitative results for arbitrary free-form text queries are shown in Fig. 4, Fig. 1, and Fig. 3. Fig. 4 uses SS3D+VLRC fine-tuned on KITTI, while Fig. 1 and Fig. 3 show SS3D+VLRC trained on YouTube8M web videos.

![](images/fbef7a06b6ed51aa82eab306097186faccf91925e59021a9014d3eb8ed68c4da.jpg)  
Figure 3. Zero-shot open-vocabulary 3D semantic localization from casual monocular videos. For each scene, we show Top: one input frame of the video, Middle: 3D reconstruction from SS3D + VLRC fine-tuned on web-videos, Bottom: the CLIPbased 3D localization from text prompts that are respectively “Where is the Castle?” and “Where is the Eiffel Tower?”. For CLIP-based 3D localization, we retain only 3D points with text similarity above a threshold of τ = 0.5.

![](images/cb97288a56c896906aa472b8cc5e5223356c87115accc638b3dc4746d337eddd.jpg)  
(a) Prompt: “Where are the cars?”

![](images/d66ff261b93bd5a8164a2f5eaef698852fe27b5477c8bc3a8b81ff91dca3273d.jpg)  
(b) Prompt: “Where is the person?”  
Figure 5. Effect of VLRC on open-vocabulary 3D localization. Without VLRC, cars are not well located and the person is missed. Adding VLRC produces cleaner and more spatially coherent 3D activations.

![](images/66310cf9c78f7e32cb065d6e27678cc91ccab9db0475982e04ad1a8751335b00.jpg)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Self-Supervised</td><td colspan="4">Lower is better ↓</td><td colspan="3">Higher is better ↑</td></tr><tr><td>Abs Rel</td><td>Sq Rel</td><td>RMSE</td><td>RMSE log</td><td> $\delta_1$ </td><td> $\delta_2$ </td><td> $\delta_3$ </td></tr><tr><td>Monodepth2 [13]</td><td>√</td><td>0.110</td><td>0.831</td><td>4.642</td><td>0.187</td><td>0.883</td><td>0.962</td><td>0.982</td></tr><tr><td>MonoViT [55]</td><td>√</td><td>0.099</td><td>0.708</td><td>4.372</td><td>0.175</td><td>0.900</td><td>0.967</td><td>0.984</td></tr><tr><td>HR-Depth [31]</td><td>√</td><td>0.109</td><td>0.792</td><td>4.632</td><td>0.185</td><td>0.884</td><td>0.962</td><td>0.983</td></tr><tr><td>RA-Depth [19]</td><td>√</td><td>0.096</td><td>0.613</td><td>4.216</td><td>0.171</td><td>0.903</td><td>0.968</td><td>0.985</td></tr><tr><td>DIFFNet [57]</td><td>√</td><td>0.102</td><td>0.764</td><td>4.483</td><td>0.180</td><td>0.896</td><td>0.965</td><td>0.983</td></tr><tr><td>Hariat et al. [15]</td><td>√</td><td>0.082</td><td>0.604</td><td>4.108</td><td>0.162</td><td>0.928</td><td>0.968</td><td>0.985</td></tr><tr><td>SS3D</td><td>√</td><td>0.064</td><td>0.530</td><td>3.212</td><td>0.138</td><td>0.946</td><td>0.977</td><td>0.986</td></tr><tr><td>Ours: SS3D + VLRC</td><td>√</td><td>0.060</td><td>0.496</td><td>2.908</td><td>0.133</td><td>0.950</td><td>0.978</td><td>0.986</td></tr></table>

Table 1. KITTI FT - Fine-tuning on KITTI [12] and evaluating on KITTI.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Self-Supervised</td><td colspan="3">Lower is better ↓</td><td colspan="3">Higher is better ↑</td></tr><tr><td>Abs Rel</td><td>RMSE</td><td>RMSE log</td><td> $\delta_1$ </td><td> $\delta_2$ </td><td> $\delta_3$ </td></tr><tr><td>MovingIndoor [58]</td><td>√</td><td>0.208</td><td>0.712</td><td>0.086</td><td>0.674</td><td>0.900</td><td>0.968</td></tr><tr><td>StructDepth [25]</td><td>√</td><td>0.140</td><td>0.540</td><td>0.060</td><td>0.817</td><td>0.955</td><td>0.988</td></tr><tr><td>MonoIndoor++ [27]</td><td>√</td><td>0.132</td><td>0.517</td><td>N/A</td><td>0.834</td><td>0.961</td><td>0.990</td></tr><tr><td>IndoorDepth [9]</td><td>√</td><td>0.126</td><td>0.494</td><td>0.054</td><td>0.845</td><td>0.965</td><td>0.991</td></tr><tr><td>Hariat et al. [15]</td><td>√</td><td>0.115</td><td>0.458</td><td>0.054</td><td>0.859</td><td>0.970</td><td>0.992</td></tr><tr><td>SS3D</td><td>√</td><td>0.090</td><td>0.418</td><td>0.049</td><td>0.866</td><td>0.970</td><td>0.992</td></tr><tr><td>Ours: SS3D + VLRC</td><td>√</td><td>0.082</td><td>0.407</td><td>0.044</td><td>0.867</td><td>0.971</td><td>0.992</td></tr></table>

Table 2. NYU FT - Fine-tuning on NYUv2 [41] and evaluating on NYUv2.

Figure 4. Qualitative open-vocabulary 3D results on KITTI. Each row shows, from left to right: Input frame, Point cloud, Prompt heatmap, and Depth. The heatmaps are generated from text queries that are respectively from top to bottom: “Where are the trees?”, “Where are the cars?”, and “Where is the person?”.

<table><tr><td rowspan="2">Method</td><td rowspan="2">AbsRel ↓</td><td rowspan="2">Δ AbsRel</td><td rowspan="2">δ1 (%) ↑</td><td rowspan="2">Δ δ1</td><td rowspan="2">Method</td><td colspan="2">ScanNet200</td><td colspan="2">KITTI</td></tr><tr><td>mIoU ↑</td><td>mAcc ↑</td><td>mIoU ↑</td><td>mAcc ↑</td></tr><tr><td>SS3D (Baseline)</td><td>0.064</td><td>-</td><td>94.6</td><td>-</td><td colspan="5">ScanNet200: Casper3D protocol</td></tr><tr><td>w/ DINOv2 [33]</td><td>0.065</td><td>+0.001</td><td>94.3</td><td>-0.3</td><td>Casper3D [16]</td><td>11.0</td><td>18.1</td><td>-</td><td>-</td></tr><tr><td>w/ TIPSv2 [4]</td><td>0.061</td><td>-0.003</td><td>95.0</td><td>+0.3</td><td>Casper3D pretrained w/ SelfEvo geometry</td><td>11.1</td><td>18.3</td><td>-</td><td>-</td></tr><tr><td>w/ CLIP-Seg [30]</td><td>0.060</td><td>-0.004</td><td>95.0</td><td>+0.4</td><td>Casper3D pretrained w/ SelfEvo+VLRC geometry</td><td>12.1</td><td>19.0</td><td>-</td><td>-</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="5">KITTI: zero-shot geometry-to-semantics protocol</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>SS3D</td><td>-</td><td>-</td><td>17.5</td><td>28.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>SS3D + VLRC</td><td>-</td><td>-</td><td>24.0</td><td>39.3</td></tr></table>

Table 3. Effect of the feature backbone used in VLRC. Finetuning and evaluation are performed on KITTI. ∆ values are computed relative to SS3D. Green indicates improvement and red indicates degradation.

<table><tr><td rowspan="2">Method</td><td colspan="2">Sintel</td><td colspan="2">KITTI</td></tr><tr><td>Abs Rel ↓</td><td>δ &lt; 1.25 ↑</td><td>Abs Rel ↓</td><td>δ &lt; 1.25 ↑</td></tr><tr><td>VGGT [45]</td><td>0.227</td><td>0.684</td><td>0.059</td><td>0.961</td></tr><tr><td>SelfEvo (VGGT)</td><td>0.212</td><td>0.692</td><td>0.042</td><td>0.979</td></tr><tr><td>SelfEvo + VLRC</td><td>0.209</td><td>0.700</td><td>0.038</td><td>0.980</td></tr></table>

Table 4. VLRC complements SelfEvo-style post-training. Adding VLRC to SelfEvo improves depth prediction on Sintel and KITTI, showing that VLRC provides complementary supervision during unlabeled adaptation.  
Table 5. Open-vocabulary 3D semantic segmentation. We evaluate VLRC under two complementary protocols: ScanNet200 with Casper3D fine-tuning, and a KITTI zero-shot geometry-tosemantics protocol. VLRC improves performance in both settings, indicating better alignment between predicted 3D structure and dense vision-language features.

## 5. Conclusion

We introduced Vision-Language Reprojection Consistency, a general auxiliary objective for feed-forward 3D learning. Instead of using vision-language models only after geometry has been estimated, VLRC uses dense VLM features as a training signal: predicted depth, camera motion, and intrinsics induce cross-view correspondences, and the model is encouraged to make language-aligned features consistent across these reprojected views. This provides a scalable feature-space supervision signal that does not require additional 3D annotations and complements both photometric self-supervision and supervised-pretrained 3D reconstruction models.

Across self-supervised SS3D fine-tuning and VGGT/SelfEvo-style unlabeled adaptation, VLRC improves core 3D estimates . We further show that geometry trained with VLRC better supports multi-view aggregation of dense VLM features, improving open-vocabulary 3D semantic segmentation on both indoor and outdoor protocols. These results suggest that aligning geometry with vision-language representations during training is a promising direction for building scalable 3D models that are geometrically accurate, and more compatible with open-vocabulary semantic understanding.

Future work. An interesting direction for future work is to jointly adapt the VLM together with the 3D reconstruction model, so that the dense vision-language features themselves become more geometry-aware.

## References

[1] Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, and Sudheendra Vijayanarasimhan. Youtube-8m: A largescale video classification benchmark. arXiv preprint arXiv:1609.08675, 2016. 6, 1, 3

[2] Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring multi-view consistency in generated images. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6034–6044, 2025. 3

[3] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In European conference on computer vision, pages 611–625. Springer, 2012. 6, 1

[4] Bingyi Cao, Koert Chen, Kevis-Kokitsi Maninis, Kaifeng Chen, Arjun Karpur, Ye Xia, Sahil Dua, Tanmaya Dabral, Guangxing Han, Bohyung Han, et al. Tipsv2: Advancing vision-language pretraining with enhanced patch-text align ment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29325–29335, 2026. 8

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 3

[6] Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, and Pascale Fung. Vl-jepa: Joint embedding predictive architecture for vision-language. arXiv preprint arXiv:2512.10942, 2025. 2

[7] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 6

[8] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Pla: Language-driven openvocabulary 3d scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7010–7019, 2023. 3

[9] Chao Fan, Zhenyu Yin, Yue Li, and Feiqing Zhang. Deeper into self-supervised monocular indoor depth estimation. arXiv preprint arXiv:2312.01283, 2023. 8

[10] Enrico Fini, Victor G Turrisi Da Costa, Xavier Alameda-Pineda, Elisa Ricci, Karteek Alahari, and Julien Mairal. Selfsupervised models are continual learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9621–9630, 2022. 3

[11] Stephanie Fu, Mark Hamilton, Laura Brandt, Axel Feldman, Zhoutong Zhang, and William T Freeman. Featup: A modelagnostic framework for features at any resolution. arXiv preprint arXiv:2403.10516, 2024. 6

[12] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The in-

ternational journal of robotics research, 32(11):1231–1237, 2013. 6, 8

[13] Clement Godard, Oisin Mac Aodha, Michael Firman, and´ Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3828–3838, 2019. 3, 8

[14] Marwane Hariat, Antoine Manzanera, and David Filliat. Rebalancing gradient to improve self-supervised co-training of depth, odometry and optical flow predictions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1267–1276, 2023. 3, 6, 2

[15] Marwane Hariat, Antoine Manzanera, and David Filliat. Improved monocular depth prediction using distance transform over pre-semantic contours with self-supervised neural networks. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21868–21879, 2025. 3, 8

[16] Marwane Hariat, Gianni Franchi, David Filliat, and Antoine Manzanera. Lightweight 3d feature pretraining by bayesian inversion of 2d foundation models. 2026. 3, 7, 8

[17] Marwane Hariat, Gianni Franchi, David Filliat, and Antoine Manzanera. Ss3d: End2end self-supervised 3d from web videos. arXiv preprint arXiv:2604.22686, 2026. 2, 4, 6

[18] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´ vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 3

[19] Mu He, Le Hui, Yikai Bian, Jian Ren, Jin Xie, and Jian Yang. Ra-depth: Resolution adaptive self-supervised monocular depth estimation. In European Conference on Computer Vision, pages 565–581. Springer, 2022. 8

[20] Nan Huang, Pengcheng Yu, Weijia Zeng, James M Rehg, Angjoo Kanazawa, Haiwen Feng, and Qianqian Wang. Selfimproving 4d perception via self-distillation. arXiv preprint arXiv:2604.08532, 2026. 2, 3, 6

[21] Li Jiang, Shaoshuai Shi, and Bernt Schiele. Open-vocabulary 3d semantic segmentation with foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21284–21294, 2024. 3, 6

[22] Nikhil Keetha, Norman Muller, Johannes Sch ¨ onberger,¨ Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414, 2025. 1, 3

[23] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 6

[24] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with mast3r. In Computer Vision – ECCV 2024: 18th European Conference, page 71–91. Springer-Verlag, 2024. 3

[25] Boying Li, Yuan Huang, Zeyu Liu, Danping Zou, and Wenxian Yu. Structdepth: Leveraging the structural regularities for self-supervised indoor depth estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12663–12673, 2021. 8

[26] Hanhan Li, Ariel Gordon, Hang Zhao, Vincent Casser, and Anelia Angelova. Unsupervised monocular depth learning

in dynamic scenes. In Conference on Robot Learning, pages 1908–1917. PMLR, 2021. 3

[27] Runze Li, Pan Ji, Yi Xu, and Bir Bhanu. Monoindoor++: Towards better practice of self-supervised monocular depth estimation for indoor environments. IEEE Transactions on Circuits and Systems for Video Technology, 33(2):830–846, 2022. 8

[28] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 1, 2

[29] Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. Partslip: Low-shot part segmentation for 3d point clouds via pretrained imagelanguage models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21736–21746, 2023. 3

[30] Timo Luddecke and Alexander Ecker. Image segmenta-¨ tion using text and image prompts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7086–7096, 2022. 2, 6, 8

[31] Xiaoyang Lyu, Liang Liu, Mengmeng Wang, Xin Kong, Lina Liu, Yong Liu, Xinxin Chen, and Yi Yuan. Hr-depth: High resolution self-supervised monocular depth estimation. In Proceedings of the AAAI conference on artificial intelli gence, pages 2294–2301, 2021. 8

[32] Remi Marsal, Florian Chabot, Ang´ elique Loesch, and´ Hichem Sahbi. Brightflow: Brightness-change-aware unsupervised learning of optical flow. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2061–2070, 2023. 3

[33] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 8

[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 6

[35] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 815–824, 2023. 3

[36] Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9277–9286, 2019. 3

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2

[38] Rene Ranftl, Katrin Lasinger, David Hafner, Konrad´ Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 2

[39] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems, 31, 2018. 2

[40] Chang Shu, Kun Yu, Zhixiang Duan, and Kuiyuan Yang. Feature-metric loss for self-supervised learning of depth and egomotion. In European Conference on Computer Vision, pages 572–588. Springer, 2020. 3

[41] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012. 6, 8

[42] Dan Song, Xinwei Fu, Ning Liu, Wei-Zhi Nie, Wen-Hui Li, Lan-Jun Wang, You Yang, and An-An Liu. Mv-clip: Multiview clip for zero-shot 3d shape recognition. IEEE Transactions on Circuits and Systems for Video Technology, 35(9): 8767–8779, 2025. 3

[43] Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram¨ Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573–580. IEEE, 2012. 6, 1

[44] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 2

[45] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 1, 3, 4, 6, 8, 2

[46] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10510–10522, 2025. 1, 3

[47] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 1, 2

[48] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025. 1

[49] Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Bregier, Yohann Cabon, Vaibhav Arora, Leonid Ants-´ feld, Boris Chidlovskii, Gabriela Csurka, and Jer´ ome Re-ˆ vaud. Croco: self-supervised pre-training for 3d vision tasks

by cross-view completion. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NeurIPS’22), 2022. 2

[50] Felix Wimbauer, Weirong Chen, Dominik Muhle, Christian Rupprecht, and Daniel Cremers. Anycam: Learning to recover camera poses and intrinsics from casual videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16717–16727, 2025. 1, 2

[51] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34: 12077–12090, 2021. 1

[52] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024. 2

[53] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 2

[54] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8552–8562, 2022. 3

[55] Chaoqiang Zhao, Youmin Zhang, Matteo Poggi, Fabio Tosi, Xianda Guo, Zheng Zhu, Guan Huang, Yang Tang, and Stefano Mattoccia. Monovit: Self-supervised monocular depth estimation with a vision transformer. In 2022 international conference on 3D vision (3DV), pages 668–678. IEEE, 2022. 8

[56] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European conference on computer vision, pages 696–712. Springer, 2022. 2

[57] Hang Zhou, David Greenwood, and Sarah Taylor. Selfsupervised monocular depth estimation with internal feature fusion. In British Machine Vision Conference (BMVC), 2021. 8

[58] Junsheng Zhou, Yuwang Wang, Kaihuai Qin, and Wenjun Zeng. Moving indoor: Unsupervised video depth learning in challenging environments. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8618– 8627, 2019. 8

[59] Mingquan Zhou, Chen He, Ruiping Wang, and Xilin Chen. Ov3d-cg: Open-vocabulary 3d instance segmentation with contextual guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5305–5314, 2025. 3

[60] Tinghui Zhou, Matthew Brown, Noah Snavely, and David G Lowe. Unsupervised learning of depth and ego-motion from video. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1851–1858, 2017. 3

# VLRC: Vision-Language Reprojection Consistency as a scalable signal for better feed-forward 3D pretraining

Supplementary Material

The supplementary material includes multiple details and insights that complement the main paper.

## A. KITTI Open-Vocabulary 3D Segmentation Protocol

Our new protocol is based on the sequences of the KITTI Odometry dataset To obtain the 3D segmentation map, for each scene we first consider a temporal window of 10 frames centered around the target frame. Using the predicted depth, intrinsic parameters, and camera poses, of the considered pretrained feed-forward multi-view reconstruction model we back-project the target frame into a 3D point cloud. Each 3D point is then reprojected into all 10 neighboring frames, and we aggregate the corresponding logits with respect to each class with the prompt template ‘‘a photo of a [CLASS].’’ and by averaging them across views. The final semantic label for each 3D point is obtained by taking the arg max over the averaged logits.

For evaluation, we use Velodyne LiDAR points as reference 3D points. Since KITTI Odometry does not provide dense semantic annotations, we assign pseudo-semantic labels by applying SegFormer-B5 [51] to the RGB frames and projecting the LiDAR points into the labeled images.

To facilitate reproducibility, we provide the code for generating the pseudo labels, constructing the 3D point-level evaluation set, and running the KITTI open-vocabulary 3D segmentation protocol.

This protocol directly evaluates the alignment between the predicted geometry and dense CLIP features. As shown in Tab. A.1, SS3D + VLRC outperforms AnyCam[50] and VGGT[45], and approaches the performance of DUSt3R[47], which are supervised methods. In contrast, SS3D+VLRC is trained fully self-supervised. DUSt3R remains a strong reference because on top of being supervised, it relies on heavier test-time optimization. Qualitative comparisons are shown on Fig. A.1.

We also observe that computing CLIP logits over SAM masks, rather than individual pixels, provides an additional improvement.

## B. Additional 3D Estimation Results

For pose and intrinsic results, we retrain SS3D + VLRC on Youtube8M[1] and we evaluate the performances in zeroshot for camera motion on Sintel [3] and TUM-RGBD [43];

<table><tr><td>Method</td><td>mIoU (%)↑</td><td>mAcc (%)↑</td></tr><tr><td>DUSt3R</td><td>26.2</td><td>43.4</td></tr><tr><td>AnyCam</td><td>15.2</td><td>24.0</td></tr><tr><td>VGGT</td><td>18.2</td><td>32.2</td></tr><tr><td>SS3D</td><td>17.5</td><td>28.3</td></tr><tr><td>Ours</td><td>24.0</td><td>39.3</td></tr></table>

Table A.1. 3D semantic segmentation results. Mean IoU (mIoU) and mean class accuracy (mAcc).

and intrinsics on Sintel.

Pose Estimation. We further report zero-shot pose estimation results in Table B.1. We evaluate on Sintel, which contains synthetic scenes with complex motion and strong appearance changes, and on dynamic TUM-RGBD, which captures challenging real-world motion. We compare against AnyCam, a recent camera-pose method designed to leverage strong pretrained supervised depth and flow estimators. Despite being trained fully self-supervised, SS3D+VLRC achieves competitive or stronger performance across these benchmarks. The gains over SS3D show that the VRLC signal improves not only depth, but also the camera-motion component of the unified 3D estimator.

Intrinsic Estimation. Table B.2 reports intrinsic estimation results. SS3D+VLRC predicts camera intrinsics directly from raw monocular videos, without external calibration cues or privileged information. Despite this fully self-supervised setting, our method matches AnyCam in absolute focal error (AFE) and achieves a substantially lower relative focal error (RFE). It also brings clear improvements over the original SS3D baseline.

Overall, the consistent gains across depth, pose, and intrinsics demonstrate that vision-language reprojection improves the core geometric quantities required for 3D reconstruction. Table 3 further shows that CLIP features outperform DINO features in our reprojection framework, suggesting that the language-aligned semantic information encoded by CLIP provides a stronger supervision signal than purely visual features.

We also provide qualitative results in Fig. 4, showing predicted depth maps and the corresponding 3D reconstructions on several sequences of KITTI Odometry dataset. For clearer visualization, we remove sky regions using an offthe-shelf sky segmentation network before rendering the point clouds.

![](images/97c79bf85143f4e3dda2a344811d00254253f4799d6495d3639606069e68a72c.jpg)  
Figure A.1. Comparison of labeled point clouds obtained by reprojection from AnyCam[50], SS3D + VLRC, and DUSt3R, with KITTI ground truth shown for reference. For visualization, reprojections are computed from ground-truth labels rather than CLIP predictions. AnyCam exhibits strong geometric distortions, while DUSt3R benefits from supervised training and heavier test-time alignment/bundle adjustment. In contrast, our method remains fully self-supervised and lightweight at inference.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td rowspan="2">No Superv.</td><td rowspan="2">Approx. Runtime</td><td colspan="3">Sintel</td><td colspan="3">TUM-RGBD (dynamics)</td></tr><tr><td>ATE↓</td><td> $RPE_{trans} \downarrow$ </td><td> $RPE_{rot} \downarrow$ </td><td>ATE↓</td><td> $RPE_{trans} \downarrow$ </td><td> $RPE_{rot} \downarrow$ </td></tr><tr><td rowspan="3"></td><td>AnyCam[50]</td><td>√</td><td>&lt; 20sec</td><td>0.099</td><td>0.045</td><td>0.567</td><td>0.095</td><td>0.025</td><td>1.050</td></tr><tr><td>SS3D[17]</td><td>√</td><td>&lt; 20sec</td><td>0.090</td><td>0.043</td><td>0.601</td><td>0.092</td><td>0.026</td><td>1.064</td></tr><tr><td>Ours: SS3D + VLRC</td><td>√</td><td>&lt; 20sec</td><td>0.088</td><td>0.041</td><td>0.587</td><td>0.090</td><td>0.021</td><td>1.038</td></tr></table>

Table B.1. Pose estimation comparison to AnyCam[50] in zero shot. Absolute trajectory error (ATE) and relative pose error for translation (RPEtrans) and rotation (RPErot) on the Sintel and TUM-RGBD datasets.

<table><tr><td>Method</td><td>AFE (px)↓</td><td>RFE (%)↓</td></tr><tr><td>UniDepth</td><td>447.4</td><td>35.7</td></tr><tr><td>Dust3r</td><td>434.0</td><td>36.4</td></tr><tr><td>AnyCam</td><td>252.2</td><td>18.1</td></tr><tr><td>SS3D</td><td>256.6</td><td>16.7</td></tr><tr><td>Ours: SS3D + VLRC</td><td>255.5</td><td>16.5</td></tr></table>

Table B.2. Intrinsic parameter estimation on Sintel. Mean absolute focal error (AFE) and mean relative focal error (RFE).

## C. Additional Implementation Details

We provide additional details for retraining SS3D+VLRC on YouTube8M. We follow the same training protocol as the authors of SS3D[17]. See the paper for more details. Here is an overview.

Architecture: we use the VGGT [45] architecture for our pipeline, keeping only the depth, pose and intrinsic heads. Preprocessing. Shot detection, frame-rate normalization, and frame filtering are performed with PyAV.

Validity masking. We use CoopNet [14] to identify unreliable pixels, including occlusions and moving objects.

Training of student: To construct each training batch, we first randomly select three clusters and then sample a total of 24 images from them. Instead of applying standard backpropagation, we follow the method proposed in [39] to compute Pareto-optimal gradients. During training, images are resized such that their shortest side is 518 pixels, after which a 518 × 518 crop is extracted.

Hyperparameters: We used $\lambda _ { \mathrm { d i s t i l l } } = 0 . 2$

Web-video qualitative results. For the qualitative webvideo results in the main paper, we use SS3D+VLRC retrained on YouTube8M [1] following the self-supervised SS3D training protocol, with VLRC added as an auxiliary feature-space reprojection loss. No 3D annotations, camera poses, or depth supervision are used during this retraining.