# Cross4D-JEPA: Dense Cross-modal Correspondence Distillation for 4D Point Cloud Representation Learning

Trung Thanh Nguyen $^{1}$ , Hai Nguyen-Truong $^{2}$ , Tu Vo $^{3}$ , Hoang M. Truong $^{4}$ , and Tuan-Anh Vu $^{5*}$ $^{1}$ Nagoya University, Japan $^{2}$ Northeastern University, USA $^{3}$ KC Machine Learning Lab, Korea $^{4}$ University of Science, Vietnam National University Ho Chi Minh City, Vietnam $^{5}$ University of California, Los Angeles, USA

## Abstract

Automatic understanding of dynamic 4D point clouds, the 3D-point sequences captured over time by depth sensors and LiDAR, is central to robotics and embodied perception. Yet annotating them densely is expensive, making self-supervised pretraining the natural route to transferable representations. Existing pretext tasks, however, are almost entirely intra-modal, and the few methods that transfer knowledge from 2D foundation models rely on a single global embedding per clip, discarding the rich per-patch semantics that these models compute. To address this gap, we propose Cross4D-JEPA, a teacher-student method that distills a frozen 2D foundation model, an image model DI-NOv2, or a video model V-JEPA 2, into a 4D point encoder. The proposed method combines (1) a dense cross-modal correspondence that maps every 3D point to the teacher patch feature it projects to, and (2) a per-point objective that trains the student to match these features in latent space with no masking, negatives, or decoder. We evaluate Cross4D-JEPA on four benchmarks, MSR-Action3D, De-formingThings4D, NTU-RGB+D60, and HOI4D, against intra-modal and global cross-modal baselines. Experimental results show that, under a matched protocol, the proposed method consistently outperforms intra-modal and global cross-modal baselines across the four benchmarks and is competitive with heavier published 4D methods; further analysis attributes this gain primarily to the granularity of the correspondence rather than the teacher modality. Beyond recognition accuracy, the dense representation learned by Cross4D-JEPA transfers across domains, improves label efficiency, and improves full-label fine-tuning under the same training budget, while a $13 \times$ smaller encoder matches a heavyweight pooling backbone.

## 1. Introduction

Automatic understanding of dynamic 4D point clouds, the sequences of 3D points captured over time by depth sensors, LiDAR, and multi-view rigs, is central to robotics and embodied perception $[8, 9, 40]$ , where recognizing actions, interactions, and deformations $[14–16, 31]$ underlies downstream decisions. However, dense temporal labels are expensive to obtain, making self-supervised pretraining the natural route to transferable 4D representations $[10, 32, 48]$ .

Yet existing pretext tasks for 4D point clouds are almost entirely intra-modal. Spatio-temporal contrastive learning $[6, 10, 33]$ , masked structure prediction $[32, 48]$ , and complete-to-partial geometry distillation $[45]$ all supervise the encoder from the point clouds alone, and never inject the appearance and object semantics that large 2D models already capture. The one method that transfers from 2D video into a point encoder, CrossVideo $[18]$ , contrastively matches a single global embedding per clip. This signal is coarse, collapsing a structured 2D feature map into a single vector and discarding where each feature came from.

This is a missed opportunity, as the richest 2D foundation models are precisely those with spatially dense, per-patch features that a single global embedding collapses. DINOv2 [22] produces a patch-level feature map whose local descriptors support dense correspondence and segmentation without any fine-tuning. Static cross-modal distillation already exploits this property, as CrossJEPA [25] and Concerto [44] distill an image model into a static point encoder by predicting target embeddings in latent space [2], the “beyond-masking” recipe that also underlies Point-JEPA [29]. The open question is how to carry such knowledge into dynamic point clouds, and whether to do so at the granularity of one embedding per clip, as in prior video-to-point transfer, or densely, per point.

To close this gap, we propose Cross4D-JEPA, a teacher-student method that distills a frozen 2D foundation model, an image model DINOv2, or a video model V-JEPA 2 [3], into a geometry-only 4D point encoder. The proposed method has two components. First, it builds a dense cross-modal correspondence. For each frame, we render the point cloud, record the per-pixel nearest-point index, and pull every teacher patch feature back onto the 3D point it represents, giving an exact, occlusion-aware geometric correspondence. Second, a per-point objective trains the student to predict, in the teacher's latent space, the teacher embedding for each point, without masking, negatives, or a decoder. With the teacher frozen and its per-point targets cached once, pretraining incurs no additional teacher forward pass and stays within single-GPU hours. The main contributions of this work are as follows:

\- We propose Cross4D-JEPA, a dense cross-modal correspondence distillation method that distills a frozen 2D foundation model into a 4D point encoder per point, rather than via a single global embedding per clip, without masking, negatives, or a decoder.

\- We show through controlled analysis that the transfer of Cross4D-JEPA is driven by correspondence granularity, not teacher modality, as the image teacher DINOv2 [22] and the video teacher V-JEPA 2 [3] are statistically tied under global supervision.

\- We evaluate the proposed Cross4D-JEPA on four benchmarks, MSR-Action3D [14], DeformingThings4D [15], NTU-RGB+D60 [31], and HOI4D [16]. The results show that it consistently outperforms intra-modal and global cross-modal baselines, improves label efficiency and full-label fine-tuning performance under the same budget, and matches a heavyweight pooling backbone with $13 \times$ fewer parameters.

## 2. Related Work

Self-supervised learning on static point clouds. Masked autoencoding underpins much of 3D self-supervised learning, from Point-MAE [23] and Point-M2AE [42] to cross-modal variants that distill 2D knowledge into point encoders such as I2P-MAE [43], ACT [7], and ReCon [28], all of which rely on masking and decoders. Point-JEPA [29] instead adopts the joint-embedding predictive paradigm [2], predicting target embeddings in latent space. These methods operate on static point clouds, whereas the proposed Cross4D-JEPA adds the temporal axis and a dense per-point correspondence to a frozen 2D foundation teacher.

Self-supervised learning on 4D point clouds. For dynamic sequences, spatio-temporal backbones such as PST-Net [9], P4TRANSFORMER [8], and PPTr [40] support a range of pretext objectives. These include spatio-temporal contrastive learning [6, 10, 33, 35], clip-order prediction [38], masked structure prediction [32, 48], and intramodal distillation [34, 45]. The main cross-modal 4D method, CrossVideo [18], applies contrastive losses to global clip embeddings. The proposed Cross4D-JEPA instead introduces a dense, cross-modal, and non-contrastive objective that predicts per-patch features for a frozen 2D foundation model at each 3D point, without negatives.

Cross-modal 2D-to-3D distillation. A separate line distills frozen 2D foundation models into 3D encoders via joint-embedding prediction, in which CrossJEPA [25] and Concerto [44] supervise a static point encoder using object-or scene-level embeddings from cached targets. A complementary line transfers 2D features through an explicit 2D↔3D correspondence, using calibrated cameras and Li-DAR for contrastive distillation [17, 19, 30] or per-scene feature-field optimization [12, 13, 24]. Cross-modal distillation has also been used for action recognition, transferring knowledge across modalities at the clip level [21]. Unlike these approaches, which require calibrated camera-LiDAR alignment or per-scene optimization, the proposed Cross4D-JEPA recovers the correspondence from geometry alone by rendering, so it needs no calibration and caches each target once; it learns a feed-forward 4D encoder rather than a static model or a per-scene field, and extends the cached-target formulation to dynamic point clouds with a dense per-point objective.

Video foundation models and latent world models. In 2D video, masked modeling [37, 39] and joint-embedding prediction [3, 5] predict masked or future latents, and recent variants act as latent world models for forecasting [41, 46, 47]. We adopt such models as teachers, namely a video model V-JEPA 2 [3] and an image model DINOv2 [22], and find that the per-frame image teacher transfers densely as well as or better, while latent forecasting does not drive the gain. By contrast, the proposed Cross4D-JEPA distills a general 3D-geometry encoder for deformable, non-driving motion, rather than the 2D-video or driving-LiDAR encoders these works produce.

## 3. Method

Problem formulation. We denote a point-cloud clip as $P = \{P_{f}\}_{f=1}^{T}$ , with per-frame point set $P_{f} = \{P_{f,n}\}_{n=1}^{N} \in R^{N \times 3}$ , where T is the number of frames and N the number of points per frame. Self-supervised 4D representation learning seeks an encoder $f_{\theta}$ that maps a clip to a set of spatio-temporal tokens $f_{\theta}(P) = \{(\mathbf{t}_{j}, c_{j})\}_{j=1}^{M}$ , where $t_{j} \in R^{C}$ is a token feature and $c_{j} \in R^{3}$ its spatial center, usable for downstream recognition without labels. We learn $f_{\theta}$ by distilling a frozen 2D foundation model $g_{\phi}$ , reading for each visible point a teacher feature at the pixel it projects to, forming a per-point target, and training $f_{\theta}$ to predict that target in the teacher's latent space. Unlike prior cross-modal transfer from 2D [18], which supervises a single global embedding per clip, the supervision here is dense, with one target per point.

Method overview. Cross4D-JEPA is a two-stream, asymmetric architecture (Fig. 1) with three stages. (1) the frozen teacher $g_{\phi}$ renders each frame and produces a per-point target $y_{f,n}$ , computed once and cached; (2) the trainable 4D student $f_{\theta}$ maps the geometry-only clip to spatio-temporal tokens $\{(\mathbf{t}_j, c_j)\}_{j=1}^M$ ; (3) a lightweight per-token head $\psi$ predicts each token's target $\hat{y}_j = \psi(\mathbf{t}_j)$ in the teacher's latent space, matched with a cosine loss and no masking, negatives, or decoder. The asymmetry is deliberate, as the geometry-only student must reproduce, at every point, what the appearance-rich teacher sees at the corresponding pixel, rather than summarizing a clip into one vector.

![](images/7461e5d2dfe9f7fcd4299a49ecbbb48b11fd6d50b7877ec4a24f021b0878371a.jpg)  
Figure 1. Overview of Cross4D-JEPA. A frozen 2D foundation model (top) encodes each rendered frame, and its patch features are pulled back onto the 3D points they depict, giving a per-point target cached once. A trainable 4D encoder (bottom) maps the clip to spatio-temporal tokens, and a per-token head predicts each token's target in the teacher's latent space under a cosine loss, with no masking, negatives, or decoder. The distillation is dense, per point rather than per clip.

The bridge between the 2D teacher and the 4D student is rendering, despite their different inputs. The forward render $I_{f} = \mathcal{R}(P_{f};\pi)$ maps the 3D geometry into the teacher's 2D input space, and the inverse point-to-pixel correspondence $\mathrm{pix}_{\pi}$ carries the teacher's patch features back onto the 3D points, letting an image model supervise a geometry-only encoder. The teacher operates per frame, while the student ingests the entire clip, aggregating the per-point, per-frame targets $\{y_{f,n}\}$ into a spatio-temporal representation. The central design choice is that this supervision is dense, with each token supervised by the teacher feature at the surface point it represents.

## 3.1. Dense cross-modal correspondence

Since the 2D teacher operates on images, we first view each frame's geometry from a single virtual viewpoint, a fixed pinhole projection $\pi = (K, [R|t])$ with intrinsics $K$ and pose $[R|t]$ . For each frame $f$ , we render the points under $\pi$ into a depth-shaded image with a dependency-free, $z$ -buffered splatter as:

$$
I _ {f} = \mathcal {R} (P _ {f}; \pi) \in \mathbb {R} ^ {3 \times H \times W},\tag{1}
$$

placing the viewpoint at the clip centroid and backing it off to the clip's extent to keep the whole subject within the frame, regardless of dataset units. We use a single fixed viewpoint per clip, which keeps the rendering deterministic; adding cameras raises target coverage but not probe accuracy (supplementary). The frozen teacher maps this image to a patch-token feature map, which we compress with a fixed random projection $W \in R^{D \times d_{p}}$ to bound the cache size as:

$$
\begin{array}{r l} & {\Phi_ {f} = g _ {\phi} (I _ {f}) \in \mathbb {R} ^ {h _ {p} \times w _ {p} \times D},} \\ & {\widetilde {\Phi} _ {f} = \Phi_ {f} W \in \mathbb {R} ^ {h _ {p} \times w _ {p} \times d _ {p}},} \end{array}\tag{2}
$$

where $h_p \times w_p$ is the teacher's spatial patch grid, equal to $H / 14$ per side for the image teacher DINOv2 [22]. For the video teacher V-JEPA 2 [3], we run the teacher on a short rendered clip and read its spatio-temporal patch tokens per frame, which fits the same per-point construction.

To pull these features onto the geometry, let $\operatorname{pix}_{\pi}: R^{3} \to \{1, \ldots, h_{p}\} \times \{1, \ldots, w_{p}\}$ map a 3D point to the patch it projects to under $\pi$ , and let $V_{f} \subseteq \{1, \ldots, N\}$ be the points visible in frame f, read from an occlusion-aware z-buffer with deterministic tie-breaking. The per-point target is the projected teacher feature at the corresponding patch as:

$$
y _ {f, n} = \widetilde {\Phi} _ {f} \left[ \operatorname{pix} _ {\pi} (P _ {f, n}) \right] \in \mathbb {R} ^ {d _ {p}}, \qquad n \in \mathcal {V} _ {f}.\tag{3}
$$

Occluded points carry no target and are excluded from the loss. Since $g_{\phi}$ is frozen and the viewpoint is deterministic, all targets $\{y_{f,n}\}$ are precomputed once and cached in FP16, adding no teacher forward pass to the training loop.

## 3.2. Student encoder and per-point prediction

The student $f_{\theta}$ is a standard 4D point backbone; we use a hierarchical PointNet++-style [26, 27] spatiotemporal encoder, POINT4D, and report the official P4TRANSFORMER [8] backbone where noted. It produces the tokens $\{(\mathbf{t}_{j}, c_{j})\}_{j=1}^{M}$ of the problem formulation, each assigned to a frame $f(j)$ , with $c_{j}$ recovered in the original (un-normalized) coordinates. As the tokens are farthest-point-sampled (FPS) [27] centroids that need not coincide with input points, we match each token to its nearest input point in the same frame as:

$$
n ^ {\star} (j) = \underset {n \in \{1, \dots , N \}} {\arg \min} \left\| c _ {j} - P _ {f (j), n} \right\| _ {2},\tag{4}
$$

and predict its target with a lightweight linear head $\psi: R^{C} \to R^{d_{p}}$ , giving the prediction $\hat{y}_{j} = \psi(\mathbf{t}_{j}) \in \mathbb{R}^{d_{p}}$ . In our POINT4D tokenizer, the centroids are FPS samples of the input points, so this match is near-degenerate (sub-voxel distance) and needs no distance threshold or soft assignment.

## 3.3. Training objective

Let $\mathcal{M}=\left\{j:n^{\star}(j)\in\mathcal{V}_{f(j)}\right\}$ be the set of tokens whose matched point is visible. The objective is the mean negative cosine similarity between each prediction and its stop-gradient target, computed in latent space with no decoder and no negatives as:

$$
\mathcal {L} _ {\text { dense }} = \frac {1}{| \mathcal {M} |} \sum_ {j \in \mathcal {M}} \Bigl (1 - \cos \bigl (\hat {y} _ {j}, \operatorname{sg} (y _ {f (j), n ^ {*} (j)}) \bigr) \Bigr),\tag{5}
$$

where $\cos(a,b)=a^{\top}b/(\|a\|_{2}\|b\|_{2})$ and sg blocks gradients to the frozen targets. The frozen teacher provides a fixed, non-trivial target, which rules out the trivial-constant solution of EMA-target JEPAs [2, 5]. Because this fixed target already prevents collapse, an auxiliary anti-collapse penalty is unnecessary; we include a VICReg-style [4] variance term only for completeness:

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{var}} = \frac {1}{d _ {p}} \sum_ {k = 1} ^ {d _ {p}} \mathrm{ReLU} (1 - \sigma_ {k}), \\ \sigma_ {k} = \sqrt {\mathrm{Var} _ {j \in \mathcal {M}} [ \hat {y} _ {j , k} ] + \varepsilon}, \end{array}\tag{6}
$$

where $\hat{y}_{j,k}$ is the k-th coordinate of $\hat{y}_{j}$ and $\varepsilon$ is a small constant for numerical stability. We minimize $L = L_{dense} + \lambda L_{var}$ with AdamW, linear warmup, and cosine decay; removing this term ( $\lambda=0$ ) leaves results statistically unchanged (Supplementary Material), confirming the cosine target alone suffices.

## 3.4. Teacher and variants

The proposed Cross4D-JEPA is teacher-agnostic, and we consider the following choices and ablations.

\- Teacher. The default is the image foundation model DINOv2 [22], whose patch features are spatially dense and semantically meaningful; we also evaluate a video teacher, V-JEPA 2 [3], applied to short rendered clips.

\- Teacher input. For all four datasets the teacher sees the rendered depth-shaded image (Eq. (1)), never the dataset's RGB; the point-to-pixel correspondence is thus exact by construction (the render's $z$ -buffer index) and needs no camera calibration or RGB-depth alignment, including on the RGB-D benchmarks (NTU-RGB+D 60 [31], HOI4D [16]).

\- Ablated variants. We vary two design axes. Supervision granularity ranges from a single per-clip embedding to dense per-point targets, and teacher modality ranges from an image to a video model. We additionally examine a latent-forecasting formulation that predicts a future view, a view-invariant latent field, a latent-dynamics rollout, and a multi-view consistency term.

## 4. Experiments

## 4.1. Datasets and protocols

Datasets. We evaluate the proposed method on four 4D point-cloud benchmarks spanning human, animal, and real-RGB domains, as follows:

\- MSR-Action3D (MSR) [14]: the primary benchmark for this study, 567 depth-sensor videos over 20 actions, under the standard cross-subject split of 270 train and 297 test.

\- DeformingThings4D-Animals (DT4D) [15]: synthetic 4D sequences of deforming animal meshes performing non-rigid motions, 1,772 animations over 38 animal categories, under a deterministic cross-identity split of 1,413 train and 359 test.

\- NTU-RGB+D 60 (NTU) [31]: real-RGB action recognition, 56,880 videos over 60 actions, under the standard cross-subject split of 40,320 train and 16,560 test.

\- HOI4D [16]: real RGB-D 4D semantic segmentation, 2,221 annotated sequences over $\sim$ 40 categories, under the official split of 1,776 train and 445 test.

Evaluation protocols. We report two protocols as follows: (1) Linear probing: a linear head on the frozen encoder's mean-pooled features, the primary measure of representation quality; (2) Label-efficient fine-tuning: end-to-end fine-tuning on a fraction $f \in \{0.1, 0.25, 0.5, 1.0\}$ of labels, comparing an encoder initialized from Cross4D-JEPA pretraining against the identical backbone trained from scratch. Unless noted, we report mean±std over 3 seeds, and the gaps we highlight exceed this seed spread.

Baseline methods. To isolate the effect of supervision granularity, we compare three settings that share the same POINT4D backbone, pretraining data, and probe, differing only in the distillation target, as follows:

\- Random: the same POINT4D backbone with random initialization and no distillation, serving as the lower bound.

\- Global: cross-modal distillation from a single per-clip teacher embedding, the same-granularity counterpart of - Dense (ours): the per-point correspondence distillation supervises each point with its own teacher feature.

Table 1. Effect of supervision granularity. Comparison of the proposed dense distillation with the baselines in linear-probe accuracy on three datasets. Entries are mean±std over seeds; chance denotes the random-guess accuracy. All distillation here uses DINOv2-large as the default teacher.

<table><tr><td>Dataset</td><td>chance</td><td>random</td><td>global</td><td>dense (ours)</td></tr><tr><td>MSR-Action3D</td><td>0.05</td><td> $0.42 \pm .02$ </td><td> $0.64 \pm .06$ </td><td> $\mathbf{0.81} \pm .02$ </td></tr><tr><td>DeformingThings4D</td><td>0.03</td><td> $0.24 \pm .02$ </td><td> $0.42 \pm .03$ </td><td> $\mathbf{0.49} \pm .02$ </td></tr><tr><td>NTU-RGB+D 60</td><td>0.02</td><td> $0.11 \pm .02$ </td><td> $0.23 \pm .02$ </td><td> $\mathbf{0.40} \pm .01$ </td></tr></table>

Table 2. Granularity versus teacher modality in linear-probe accuracy on the MSR-Action3D dataset with the POINT4D backbone, given as mean±std.

<table><tr><td>Teacher</td><td>modality</td><td>global</td><td>dense (ours)</td></tr><tr><td>DINOv2</td><td>image</td><td>0.64 ± .06</td><td>0.81 ± .02</td></tr><tr><td>DINOv3</td><td>image</td><td>0.64 ± .02</td><td>0.87 ± .03</td></tr><tr><td>V-JEPA 2</td><td>video</td><td>0.61 ± .08</td><td>0.88 ± .01</td></tr><tr><td>VideoMAEv2</td><td>video</td><td>0.63 ± .07</td><td>0.81 ± .01</td></tr></table>

prior 2D-to-point transfer [18].

Implementation details. The student is POINT4D, a dependency-free PointNet++-style $[26, 27]$ spatio-temporal encoder (3.3M params, T=24 frames, N=2048 points), supervised by a frozen DINOv2-large teacher for 100 epochs (AdamW $[20]$ , per-point cosine objective Eq. (5)) and evaluated by a linear probe on mean-pooled features. DINOv2-large is the default teacher throughout; the teacher-modality study (Sec. 4.2.1) additionally evaluates the video models V-JEPA 2 $[3]$ and VideoMAEv2 $[39]$ , as well as the stronger image model DINOv3 $[36]$ . For all four datasets, the teacher sees only rendered depth-shaded images (never the dataset RGB), giving calibration-free per-point targets. The full architecture, optimization, and computation, plus ablations of the random-projection dimension $d_{p}$ , render resolution, and the VICReg term, are in the Supplementary Material.

## 4.2. Dense distillation is the lever

Table 1 reports the linear-probe accuracy of the random, global, and dense encoders, given as mean±std. On the MSR-Action3D dataset, the proposed dense distillation achieves 0.81, surpassing the global baseline by 0.17 and the random encoder by 0.39 with the same probe. This 0.17 gain exceeds the inter-seed spread — the dense and global error bars do not overlap — confirming that distilling where each 2D feature belongs, rather than one vector per clip, makes the representation linearly separable. The trend is consistent across all three datasets: dense improves over global by 0.17, 0.07, and 0.17 on MSR-Action3D, DeformingThings4D, and NTU-RGB+D, with non-overlapping error bars. The lower absolute accuracies on the latter two reflect their larger label spaces of 38 and 60 classes compared to MSR-Action3D's 20. Yet, all results remain 16 to $20 \times$ the chance, and the dense-over-global ordering holds across all three datasets. Figure 2 provides the qualitative counterpart of this ordering.

![](images/d5655abbe442d8e48e9b56c50b9842a7210b1dfc36758fd45c4d086f5b794082.jpg)  
Figure 2. Dense distillation paints coherent body-part semantics onto 3D points. Qualitative counterpart of Tab. 1 on four MSR-Action3D actions (rows). Each point is colored by the PCA-to-RGB of its nearest token from the dense (ours), global, and random encoders, shown next to the input. Only the dense encoder gives the same body part a consistent color across poses without point-level supervision, whereas the global encoder collapses to a near-uniform per-clip color, and random is unstructured.

## 4.2.1. Effect of teacher modality

Table 2 varies the teacher to test whether the gain depends on its modality rather than on the dense correspondence. The dense objective is teacher-agnostic: under global supervision, every teacher sits in the 0.61 to 0.64 range regardless of modality, with the image and video teachers tied within their error bars, whereas dense supervision lifts each one into the 0.81 to 0.88 range, so a particular teacher is not what drives the transfer. Modality acts only as a secondary lever once supervision is dense, where the video teacher V-JEPA 2 reaches 0.88 and the stronger image teacher DINOv3 reaches 0.87, both above DINOv2 and VideoMAEv2 at 0.81; this 0.07 spread among teachers is far smaller than the global-to-dense step of at least 0.17, so the secondary gain tracks the teacher's dense-feature quality rather than its modality, and we adopt DINOv2 as the default for its clean per-patch correspondence. We similarly rule out the forecasting pretext tasks common in prior 4D self-supervised learning, such as future-view and latent-dynamics prediction: the variants we explored never beat dense distillation's 0.81, topping out at 0.33 on a lightweight backbone and remaining statistically indistinguishable from one another on P4TRANSFORMER, confirming that the per-point correspondence is the operative mechanism.

Table 3. Granularity × backbone. Linear-probe accuracy on the MSR-Action3D dataset with the DINOv2 teacher. The bold cells are comparable, but only dense supervision with POINT4D yields per-point features, at 13× fewer parameters.

<table><tr><td>Backbone</td><td>global</td><td>dense (ours)</td></tr><tr><td>POINT4D (3.3M)</td><td>0.64</td><td>0.81</td></tr><tr><td>P4TRANSFORMER (42M)</td><td>0.80</td><td> $\sim 0.60$ </td></tr></table>

## 4.2.2. Effect of backbone resolution

Table 3 shows that the granularity lever interacts with the encoder's token resolution. The POINT4D backbone, with 3.3M parameters, keeps tokens close to individual surface points and can absorb per-point targets, so dense supervision lifts it to 0.81 while global supervision leaves it at 0.64. The official P4TRANSFORMER, with 42M parameters, instead pools each clip toward a single classification token, which inverts the ordering: global supervision reaches 0.80, but under dense supervision the probe falls to $\sim$ 0.60, consistent with its pooled tokenization discarding the per-point targets, as detailed in the supplementary material. Each backbone is therefore best at its own token granularity, and the two bold cells are directly comparable. The practical payoff is that dense distillation lets a 13 $\times$ smaller backbone match the heavyweight pooling backbone's accuracy while retaining the per-point features shown in Fig. 2. The dense $\gg$ global ordering thus reflects a per-point-preserving backbone, not a claim that global supervision is generally weak.

## 4.3. Label-efficient fine-tuning

Table 4 reports fine-tuning from the dense encoder against training from scratch at four label fractions on the MSR-

Table 4. Label-efficient fine-tuning on the MSR-Action3D dataset with the POINT4D backbone, given as mean±std.

<table><tr><td>Labels</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>From scratch</td><td> $0.29 \pm .11$ </td><td> $0.63 \pm .02$ </td><td> $0.76 \pm .01$ </td><td> $0.82 \pm .01$ </td></tr><tr><td>Dense (ours)</td><td> $\mathbf{0.46} \pm .04$ </td><td> $\mathbf{0.73} \pm .02$ </td><td> $\mathbf{0.88} \pm .02$ </td><td> $\mathbf{0.92} \pm .01$ </td></tr><tr><td> $\Delta$ </td><td>+0.17</td><td>+0.10</td><td>+0.12</td><td>+0.10</td></tr></table>

Table 5. Cross-domain transfer. Linear-probe accuracy of a frozen encoder pretrained on the source row and probed on the target column with no target-domain pretraining. The diagonal cells are in-domain, where the source and target match. The bottom row, random, is a randomly initialized POINT4D encoder probed on each target and serves as the lower baseline that every cross-domain cell exceeds.

<table><tr><td>Source ↓ / Target →</td><td>MSR</td><td>DT4D</td><td>NTU</td></tr><tr><td>MSR-Action3D</td><td>0.81</td><td>0.44</td><td>0.46</td></tr><tr><td>DeformingThings4D</td><td>0.67</td><td>0.49</td><td>0.32</td></tr><tr><td>NTU-RGB+D 60</td><td>0.68</td><td>0.44</td><td>0.40</td></tr><tr><td>random</td><td>0.59</td><td>0.27</td><td>0.25</td></tr></table>

Action3D dataset. The dense initialization wins across all fractions, and the gain is largest when labels are scarce, at +0.17 with 10%, where the pretrained part structure serves as a proxy for the missing supervision. Unlike pretraining gains that usually fade once labels are plentiful, the advantage persists at full supervision, still adding +0.10 at 100% and reaching 0.92 against 0.82 with non-overlapping error bars, which shows that the dense correspondence encodes structure that label supervision alone does not recover and thereby raises the fully-supervised ceiling. The +0.17 at 10% also exceeds the +0.14 from global distillation, so the benefit again traces to per-point granularity.

## 4.4. Generality and robustness

Cross-domain generality. Table 5 probes a frozen dense encoder pretrained on one dataset on another, with no target-domain pretraining, and it beats a random encoder on every source-to-target pair across the human, animal, and real-RGB domains. The gains are largest for the harder targets, for example, 0.46 against 0.25 for MSR-to-NTU and 0.44 against 0.27 for MSR-to-DT4D, and even on MSR, where a random encoder already reaches $\sim$ 0.59, the transferred encoders still improve to $\sim$ 0.67. Figure 3 shows the same generality qualitatively. Dense distillation thus learns transferable structure rather than dataset-specific shortcuts.

Zero-shot 4D semantic segmentation. The most direct test of a dense objective is per-point labeling itself. Table 6 reports HOI4D 4D semantic segmentation with a frozen encoder and a linear per-point head. Transferred

![](images/884d1ecbac95c4e8179611a48a9fb4359c8a69d1473b31bf9bed9c0709820ab1.jpg)  
Figure 3. Dense features generalize across domains. Per-point features of the frozen dense encoder pretrained on MSR-Action3D and applied to DeformingThings4D-Animals, HOI4D, and NTU-RGB+D, with each point colored by its per-clip PCA-to-RGB. Part-coherent regions emerge across the animal, hand-object, and real-RGB human domains, not only on the MSR-Action3D source.

Table 6. HOI4D 4D semantic segmentation. Per-point linear probe on a frozen encoder reporting mIoU/mAcc/OA in % on a subsampled 24-frame/2048-point split with $\sim$ 40 classes.  
![](images/1f153d6292a28b11569f09015c630086f135c5d7a455bc70867eb6a246bf9026.jpg)  
Figure 4. Dense distillation recovers per-point HOI4D segmentation. One HOI4D validation clip, with each point colored by its ground-truth or in-domain-predicted semantic class under a shared palette, where matching colors indicate correct predictions, and gray is the ignored background.

zero-shot from MSR-Action3D, the dense encoder reaches 16.2 mIoU, 3.5× random init, and 3.7× the global encoder, which itself sits at the random floor, with part-coherent regions emerging on the unseen HOI4D geometry under no target-domain labels (Fig. 3). Pretrained in-domain, dense rises to 23.6 against 7.2 for global, and the dense encoder's per-point predictions closely track the ground truth on a held-out clip, whereas global and random collapse to a few classes (Fig. 4). Per-point supervision is thus what enables the spatio-temporal tokens to carry transferable per-point semantics, whereas clip pooling discards them. The effect also persists under full supervision, where initializing an end-to-end segmenter from the dense encoder raises the fully-supervised ceiling from 39.0 to 40.9 mIoU (+1.9, 3 seeds; in the Supplementary Material).

![](images/671327d3c38ecf4de3b304854bb0e3d627e976c73bd67f38d3a96731489324fb.jpg)  
Figure 5. Temporal robustness of the dense features. One MSR-Action3D motion across frames t=0 to 20, each point colored by a single shared PCA-to-RGB of the dense encoder's per-point features. Body-part regions keep a consistent color through the motion. The claim is region-level color stability, not a frozen segmentation, as MSR-Action3D resamples points per frame.

Table 7. Comparison with published 4D self-supervised learning. MSR-Action3D linear-probe accuracy. Top: published numbers on their native P4TRANSFORMER backbone. Bottom: each pretext reimplemented on the POINT4D backbone with identical data and probe, at its native training budget.

<table><tr><td>Method</td><td>Backbone</td><td>LP acc.</td></tr><tr><td colspan="3">Published (native backbone; not matched)</td></tr><tr><td>MaST-Pre [32]</td><td>P4TRANSFORMER</td><td>0.64</td></tr><tr><td>Uni4D [48]</td><td>P4TRANSFORMER</td><td>0.85</td></tr><tr><td colspan="3">Matched on the POINT4D backbone (same data + probe)</td></tr><tr><td>random init</td><td>POINT4D</td><td>0.42</td></tr><tr><td>Uni4D pretext</td><td>POINT4D</td><td>0.25</td></tr><tr><td>MaST-Pre pretext</td><td>POINT4D</td><td>0.33</td></tr><tr><td>i-JEPA (intra-modal latent)</td><td>POINT4D</td><td>0.55</td></tr><tr><td>Cross4D-JEPA (dense)</td><td>POINT4D</td><td>0.81</td></tr></table>

Temporal robustness. On an in-domain clip, the dense features are stable over time, keeping each body part a consistent color across the frames of a motion, as Fig. 5 shows. The per-point representation thus tracks parts through the deformation rather than drifting frame to frame.

Test-time robustness. Under test-time input corruption with no retraining, the dense features are far more resilient than a random encoder to temporal subsampling, still reaching 0.35 at 6 of 24 frames against 0.04, and degrade gracefully under point dropout and coordinate jitter. The full sweeps are in the Supplementary Material.

## 4.5. Comparison with 4D self-supervised methods

Table 7 places the proposed method in the context of published 4D Self-Supervised Learning (SSL) methods, which pretrain a 4D encoder on unlabeled point sequences. These methods differ in their backbones, corpora, and protocols so that a raw head-to-head comparison would confound the granularity effect studied here; we therefore report two complementary views. First, against published numbers on their native P4TRANSFORMER backbone, dense distillation on the lighter POINT4D already exceeds MaST-Pre [32] and approaches Uni4D [48] despite a weaker encoder, and its fine-tuned ceiling (0.92, Tab. 4) is competitive with from-scratch supervision, although no fully-supervised state-of-the-art claim is made. Second, under a matched protocol that shares the POINT4D backbone, pretraining data, and probe, dense distillation reaches 0.81 and outperforms every intra-modal pretext by a wide margin, namely an i-JEPA-style masked-latent objective (0.55) and reimplementations of MaST-Pre (0.33) and Uni4D (0.25). These pretexts are native to P4TRANSFORMER; reimplemented on POINT4D at their published budget and without tuning their unpublished loss weights, an untuned intra-modal pretext can leave features no more linearly separable than random initialization, so their absolute values warrant caution. The consistent gap, rather than the exact baseline values, is the robust conclusion: on a per-point backbone, it is the 2D-foundation teacher, not the self-supervised framework, that drives the transfer.

## 5. Discussion

Why dense supervision transfers and global does not. We attribute the gap between dense and global supervision to where the teacher's structure is forced to land. A single per-clip embedding routes all of the teacher's spatial structure through a single vector, which a student can match with a coarse, pose-agnostic summary; thus, on a per-point backbone, we see global distillation plateau near its global-supervision ceiling ( $\approx 0.64$ ). Dense supervision instead requires the student to reproduce the teacher's feature at each surface point, placing localized appearance and part semantics onto the geometry rather than averaging them away; we find this per-point structure is what makes actions linearly separable (0.81 vs. 0.64 at identical teacher, backbone, and budget). We read the same effect as the reason granularity, not modality, is the dominant lever: supervision granularity sets the first-order effect ( $0.64 \rightarrow 0.81$ ), whereas the teacher's dense-feature quality, not its modality, adds only a secondary gain once dense correspondence is in place — a stronger image teacher matches the video teacher.

Dense distillation needs a backbone that preserves per-point resolution. We observe that the same objective on P4TRANSFORMER reaches only $\approx 0.60$ (Sec. 4.2.2).

We read this as consistent with the backbone's pooled tokenization rather than an optimization failure: bounding the feature scale fixes training but not the probe, while the pooling toward a single classification token leaves little per-point structure for the dense objective. That same pooling conversely makes P4TRANSFORMER strong under global supervision (0.80): supervision granularity must match the granularity the backbone preserves, so we regard “dense >> global” as specific to a per-point backbone. The payoff is concrete, as this pairing matches the heavyweight pooling configuration’s accuracy with 13× fewer parameters while retaining the per-point features that a pooled clip token discards.

Relation to distilled feature fields. We view lifting a 2D foundation model's features into 3D via correspondence as a 4D analog of distilled feature fields, yet more scalable: the field is obtained in one feed-forward pass of a geometry encoder rather than via per-scene optimization, and we cache the targets once, adding no teacher forward pass to training.

## 6. Conclusion

We proposed Cross4D-JEPA, a cross-modal method that distills a frozen 2D foundation model into a 4D point encoder densely, by rendering the geometry, recovering an occlusion-aware $2D\leftrightarrow3D$ correspondence, and supervising each encoder token with the teacher patch feature of the surface point it represents, in latent space, with no masking, contrastive negatives, or decoder. Across four 4D benchmarks, dense distillation consistently outperforms intramodal and global cross-modal baselines, transfers across domains, improves label efficiency, and matches a heavy-weight pooling backbone at a fraction of its size. The central insight is that correspondence granularity, not teacher modality, drives the transfer: supervising where each 2D feature belongs, per point rather than per clip, places localized part semantics onto the geometry and makes the representation broadly useful.

Limitations. We center the multi-seed study on MSR-Action3D with a dependency-free POINT4D backbone; the official P4TRANSFORMER underperforms the dense objective, since its aggressive pooling discards the per-point structure dense distillation needs (supplementary material). On the large NTU-RGB+D 60, we pretrain on a clip subset to fit the in-memory target cache. Dense distillation presupposes a teacher with spatially dense features and a render that places the subject in the frame; for depth-only data, we approximate real appearance with the rendered views.

Future work. The per-point targets inherit the teacher's semantic feature space, visible as coherent within-frame regions (Fig. 2). Because the pipeline is teacher-agnostic and its render-based correspondence needs no calibration, substituting a dense language-aligned teacher (e.g., a dense

CLIP variant) is a direct drop-in that reuses the same cached machinery to enable open-vocabulary 3D queries over point sequences. We note that these features capture within-frame semantics but are not yet pose-invariant: across large deformations, geometry alone remains a strong baseline for cross-frame correspondence, so we leave open the question of learning pose-invariant dense 3D descriptors, for example, with correspondence-aware objectives or a finer backbone. Beyond rendered or depth views, we see recovering large-scale dynamic point clouds from real video with reconstruction pipelines such as Stereo4D [11] and DePT3R [1] as a promising route to scale dense cross-modal pretraining beyond curated meshes.

## Acknowledgements

This work used the Delta system at the National Center for Supercomputing Applications (NCSA) through allocation CIS260951 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## A. Additional implementation and robustness detail

This section collects the dataset statistics, training defaults, and the design-choice and test-time robustness analyses referenced from the main paper.

Datasets. MSR-Action3D: 567 depth-sensor videos over 20 actions, standard cross-subject split (270 train / 297 test). DeformingThings4D-Animals: synthetic 4D sequences of deforming animal meshes, 1,772 animations over 38 animal categories, deterministic cross-identity split (1,413 / 359). NTU-RGB+D 60: real-RGB action recognition, 56,880 videos over 60 actions, standard cross-subject split (40,320 / 16,560). HOI4D: real RGB-D 4D semantic segmentation, 2,221 annotated sequences over \~40 categories, official split (1,776 / 445).

Architecture and training defaults. Table 8 lists the per-dataset clip settings. The student POINT4D (3.3M params) is a PointNet++-style encoder with two set-abstraction levels and a depth-4, 4-head spatio-temporal transformer at feature dimension 256. The frozen DINOv2-large teacher's patch tokens are randomly projected to $d_p = 384$ and cached in fp16; rendered images are $126 \times 126$ ( $9 \times 9$ DINOv2 patches), one fixed teacher camera per clip. We pretrain with AdamW (learning rate $5 \times 10^{-4}$ , weight decay $10^{-4}$ , 5% linear warmup then cosine decay), optimizing the per-point cosine loss with a small VICReg term. The linear probe runs 300 epochs on mean-pooled features without standardization; light point augmentation (scaling, z-rotation, jitter) is applied only during downstream finetuning. All runs use a single NVIDIA A100 or H200 GPU, and the encoder is identical across pretraining and downstream tasks, thereby isolating the self-supervised gain.

Table 8. Training defaults.

<table><tr><td>Dataset</td><td>Frames T</td><td>Points N</td><td>Stride</td><td>Pretrain ep.</td></tr><tr><td>MSR-Action3D</td><td>24</td><td>2048</td><td>1</td><td> $\sim 100$ </td></tr><tr><td>DeformingThings4D</td><td>16</td><td>2048</td><td>1</td><td> $\sim 60$ </td></tr><tr><td>NTU-RGBD 60</td><td>24</td><td>2048</td><td>2</td><td> $\sim 100$ </td></tr><tr><td>HOI4D</td><td>24</td><td>2048</td><td>1</td><td> $\sim 60$ </td></tr></table>

Table 9. Efficiency. Parameters and FLOPs for one clip $(24 \times 2048 \times 3)$ . P4TRANSFORMER is shown for context.

<table><tr><td>Component</td><td>Params (M)</td><td>FLOPs (G)</td></tr><tr><td>POINT4D encoder (ours)</td><td>3.28</td><td>11.5</td></tr><tr><td>+ dense per-token head</td><td>0.10</td><td>0.30</td></tr><tr><td>P4TRANSFORMER (context)</td><td>42.0</td><td>77.1</td></tr></table>

Full-split NTU pretraining. The main-table NTU-RGB+D result pretrains on a 1,000-clip subset to bound the in-memory target cache. Using the token-granularity cache, we also pretrain on the full 40,320-clip Cross-Subject training split: the dense encoder's linear probe reaches 0.43 (Cross-Subject) and 0.40 (Cross-View) at a single seed, at or above the subset's 0.40. The subset is therefore a conservative estimate, and the dense-over-global advantage is not an artifact of the reduced corpus.

Efficiency and computation. Table 9 reports parameters and FLOPs per clip; the student is small, and the frozen teacher is cached (no teacher forward pass in the training loop). Pretraining is cache-amortized: teacher patch features and the per-point correspondence are rendered and cached once per dataset, then read as FP16. Because targets are stored at the student's token granularity (not per raw point), the cache is tiny — 7 MB for MSR-Action3D, 25–35 MB for HOI4D and DeformingThings4D — and end-to-end pretraining (target precompute + 100 epochs, all clips) completes in $\sim$ 35 minutes on a single GPU.

Label efficiency (plot). Figure 6 plots the MSR-Action3D label-efficiency result reported in the main paper: the dense-distilled initialization helps most when labels are scarce and remains ahead at full supervision.

Sensitivity to design choices. The dense result is stable across the random-projection dimension $d_{p}$ and the render resolution (Table 10): every setting lands in 0.81–0.85, well above the 0.64 global baseline, with higher render resolution helping slightly. The method is not hyperparameterfragile. Replacing the random projection with a learned PCA projection of the same dimensionality leaves the probe essentially unchanged (0.82 vs. 0.81), consistent with the Johnson–Lindenstrauss property that a random projection approximately preserves the cosine geometry used by the loss; we therefore keep the cheaper, learning-free random projection.

![](images/20fbd18af3f4a658e7b8129bdf9948a11ffc908db412b340345e196182418916.jpg)  
Figure 6. Label efficiency on MSR-Action3D. Action-recognition accuracy versus the fraction of labeled training clips, for a from-scratch encoder (orange) and the same architecture initialized from Cross4D-JEPA pretraining (blue); error bars are over seeds and the shaded band is the pretraining gain. The dense-distilled initialization helps most when labels are scarce (+17 points at 10%) and remains ahead at full supervision (91.9% vs. 82.4%).

Table 10. Sensitivity of MSR-Action3D dense linear-probe accuracy to the projection type, the random-projection dimension $d_{p}$ , and render image size (one seed; default random $d_{p}=384$ , img 126). All settings remain in the 0.81–0.85 range, far above the 0.64 global baseline.

<table><tr><td>Setting</td><td>LP acc.</td></tr><tr><td>default ( $d_p$ =384, img 126)</td><td>0.81</td></tr><tr><td> $d_p$ =768</td><td>0.807</td></tr><tr><td>PCA projection ( $d_p$ =384)</td><td>0.82</td></tr><tr><td>render img 98</td><td>0.807</td></tr><tr><td>render img 168</td><td>0.847</td></tr></table>

Feature standardization helps the random baseline, not the method. Our linear probe uses no feature standardization. Table 11 reports both settings: z-scoring the frozen features before the probe lifts the random-init baseline from 0.40 to 0.59 (uninformative features become more separable once normalized), but slightly lowers the dense encoder $(0.80 \rightarrow 0.79)$ . The no-standardization choice is therefore conservative — it does not help our method, only deflates the baseline it is measured against — and dense $\gg$ random holds under both (+0.39 off, +0.19 on).

Table 11. Probe feature standardization on/off (MSR-Action3D linear-probe acc., single seed). Standardization inflates the random baseline by +0.19 but slightly hurts the dense encoder; we report the conservative std-off setting throughout. Dense $\gg$ random under both.

<table><tr><td>Encoder</td><td>std off (default)</td><td>std on</td></tr><tr><td>random init</td><td>0.40</td><td>0.59</td></tr><tr><td>dense (ours)</td><td>0.80</td><td>0.79</td></tr></table>

Table 12. Cross-domain transfer matrix (linear-probe acc.): a frozen encoder pretrained on the source (row) is probed on the target (column) with no target-domain pretraining. Diagonal (italic) is in-domain. Every cross-domain cell outperforms the target's random baseline across the human (MSR), animal (DT4D), and real-RGB (NTU) domains.

<table><tr><td>Source ↓ / Target →</td><td>MSR</td><td>DT4D</td><td>NTU</td></tr><tr><td>MSR</td><td>0.826</td><td>0.435</td><td>0.463</td></tr><tr><td>DT4D</td><td>0.666</td><td>0.490</td><td>0.324</td></tr><tr><td>NTU</td><td>0.680</td><td>0.440</td><td>0.400</td></tr><tr><td>random</td><td>0.593</td><td>0.265</td><td>0.254</td></tr></table>

Cross-domain transfer (full matrix). A frozen dense encoder pretrained on one dataset and linear-probed on another — with no target-domain pretraining — beats a random encoder on every source→target pair (Table 12). Gains are largest for the harder targets (MSR → NTU 0.463 vs. 0.254, MSR → DT4D 0.435 vs. 0.265, NTU → DT4D 0.440); on MSR, where random init already reaches ∼0.59, cross-domain encoders still beat it (≈ 0.67–0.68). Dense distillation thus learns transferable structure rather than dataset-specific shortcuts.

Test-time input corruption. With no retraining, we probe the frozen dense vs. random encoders under test-time input perturbations (Table 13). Dense pretraining is far more robust to temporal subsampling — at 6 of 24 frames it still reaches 0.35 while a random encoder collapses to 0.04 — and degrades gracefully under moderate point dropout and coordinate jitter. The one regime where dense loses its edge is severe corruption (25% of points, $\sigma=0.05$ ): because the dense features genuinely encode geometry, destroying the geometry hurts them more than the already-uninformative random features. The advantage is thus a property of the learned structure — present exactly where structure survives.

Table 13. Test-time robustness (MSR-Action3D linear-probe acc.; the probe head is trained on clean features and evaluated on perturbed test inputs, no retraining). Single representative run per encoder — the clean (“none”) values lie within the per-seed range behind the main result, and the point is the degradation trend, not the absolute clean value. Dense degrades gracefully and is far more robust to temporal subsampling than random; under severe corruption (rightmost of each block), the geometry-dependent dense features lose their edge.

<table><tr><td></td><td>none</td><td>mild</td><td>mod.</td><td>severe</td></tr><tr><td colspan="5">temporal subsample (24/18/12/6 frames)</td></tr><tr><td>dense</td><td>0.85</td><td>0.81</td><td>0.69</td><td>0.35</td></tr><tr><td>random</td><td>0.40</td><td>0.13</td><td>0.06</td><td>0.04</td></tr><tr><td colspan="5">point dropout (keep 100/75/50/25%)</td></tr><tr><td>dense</td><td>0.85</td><td>0.84</td><td>0.70</td><td>0.16</td></tr><tr><td>random</td><td>0.40</td><td>0.40</td><td>0.40</td><td>0.39</td></tr><tr><td colspan="5">coordinate jitter (σ=0/.01/.02/.05)</td></tr><tr><td>dense</td><td>0.85</td><td>0.84</td><td>0.76</td><td>0.25</td></tr><tr><td>random</td><td>0.40</td><td>0.40</td><td>0.40</td><td>0.32</td></tr></table>

Table 14. VICReg ablation (MSR-Action3D dense linear-probe acc., mean±std over 3 seeds). Removing the variance term ( $\lambda=0$ ) leaves the probe unchanged; the per-point cosine target is the primary anti-collapse mechanism.

<table><tr><td>Variance term</td><td>LP acc.</td></tr><tr><td>default (λ=1)</td><td>0.836 ± 0.019</td></tr><tr><td>λ=0 (off)</td><td>0.821 ± 0.008</td></tr></table>

The dense target is what prevents collapse. The per-point cosine correspondence target is the primary anti-collapse mechanism: removing the VICReg variance term ( $\lambda=0$ ) leaves the MSR linear probe statistically unchanged (Table 14; overlapping 3-seed error bars) and far above the 0.64 global baseline. The variance term is an inessential backstop, not the source of the gain.

Number of teacher cameras and target coverage. The single-camera render leaves $\sim 49\%$ of MSR points with a teacher target (34% on HOI4D, 16% on DeformingThings4D — its $360^{\circ}$ animal meshes expose one side per camera; occluded points are masked from the loss). Adding cameras raises coverage ( $49 \rightarrow 74 \rightarrow 79\%$ for $1/2/4$ ) but not accuracy (Table 15; flat within the seed-noise floor), so one camera suffices for single-view depth data.

Per-point correspondence (DeformingThings4D). Full numbers for the per-point correspondence probe of the main paper (frozen encoder, nearest-feature matching to ground-truth mesh vertices; Table 16). Dense > random > global at every frame gap, with global below random; raw XYZ is a geometry oracle (the meshes deform slowly), so the informative comparison is among learned encoders.

Table 15. Teacher cameras vs. coverage and accuracy (MSR-Action3D dense LP, mean±std over 3 seeds). This is an independent camera sweep with its own seed set, so the single-camera value (0.828) differs from the main result (0.813) within the per-seed range. Coverage rises with cameras; accuracy is flat (overlapping error bars).

<table><tr><td># cameras</td><td>coverage</td><td>LP acc.</td></tr><tr><td>1 (default)</td><td>49%</td><td>0.828 ± 0.018</td></tr><tr><td>2</td><td>74%</td><td>0.818 ± 0.011</td></tr><tr><td>4</td><td>79%</td><td>0.823 ± 0.009</td></tr></table>

Table 16. Per-point temporal correspondence on DeformingThings4D (acc@τ, τ=0.02 of bbox diagonal; mean±std over 3 seeds). Δ is the frame gap. Raw XYZ (geometry oracle) shown for reference.

<table><tr><td>Init</td><td> $\Delta=8$ </td><td> $\Delta=15$ </td></tr><tr><td>random</td><td>0.028 ± .003</td><td>0.059 ± .023</td></tr><tr><td>global</td><td>0.015 ± .002</td><td>0.017 ± .003</td></tr><tr><td>dense (ours)</td><td>0.082 ± .004</td><td>0.165 ± .010</td></tr><tr><td>raw XYZ (oracle)</td><td>0.152</td><td>0.366</td></tr></table>

Table 17. Fully-supervised HOI4D 4D semantic segmentation (end-to-end fine-tuning, mIoU in %, mean±std over 3 seeds). Dense pretraining raises the supervised ceiling over from-scratch training.

<table><tr><td>Initialization</td><td>mIoU</td></tr><tr><td>from scratch</td><td>39.0 ± 0.5</td></tr><tr><td>dense (ours)</td><td>40.9 ± 0.3</td></tr></table>

Fully-supervised HOI4D segmentation ceiling. Beyond the frozen probe of the main paper, fine-tuning the encoder end-to-end on HOI4D under full supervision (Table 17) shows that dense pretraining also raises the supervised ceiling: initializing from the dense encoder reaches 40.9 mIoU versus 39.0 from scratch (+1.9), the same ceiling-raising effect seen in action recognition.

## B. Detailed “what is not the lever” decompositions

This section reports the detailed numbers behind the main paper's analysis of what is not the lever: neither teacher modality nor the latent-forecasting (world-model) components is the mechanism that drives cross-modal transfer to 4D point clouds — the per-point correspondence is.

Table 18. Teacher modality under global supervision (linear-probe acc.) on both backbones. The two teachers are tied within the per-seed spread on both backbones; cf. dense distillation at 0.81/0.88 (image/video) in the main paper.

<table><tr><td>Global teacher</td><td>POINT4D</td><td>P4TRANSFORMER</td></tr><tr><td>V-JEPA 2 (video)</td><td>0.61</td><td>0.800</td></tr><tr><td>DINOv2 (image)</td><td>0.64</td><td>0.804</td></tr></table>

Table 19. Exploratory decomposition of the latent-forecasting (world-model) components (single seed, lightweight SIMPLE4D, V-JEPA 2 global target, MSR-Action3D linear-probe acc.). The whole regime is far below dense distillation (0.81); on P4TRANSFORMER with 3 seeds, the differences are within the noise floor. Reported for completeness only.

<table><tr><td>Variant</td><td>LP acc.</td></tr><tr><td>full (field + rollout + consistency)</td><td>0.331</td></tr><tr><td>- consistency (field + rollout)</td><td>0.258</td></tr><tr><td>direct predictor (no field)</td><td>0.222</td></tr><tr><td>field, no rollout</td><td>0.171</td></tr><tr><td>co-temporal ( $\Delta=0$ )</td><td>0.160</td></tr></table>

Teacher modality (global supervision). With supervision reduced to a single global embedding per clip, a video teacher (V-JEPA 2) and an image teacher (DINOv2) give statistically indistinguishable linear-probe accuracy — and this tie holds on both backbones we tested (POINT4D in the main paper and P4TRANSFORMER here; Table 18), so it is not a backbone artifact. The dense formulation instead reaches 0.81 with the image teacher, so the gain comes from granularity, not modality.

Latent-forecasting components (global regime). Table 19 decomposes the world-model variants we explored. These are single-seed runs on a lightweight voxel encoder (SIMPLE4D) in the global-target regime; we include them for completeness and do not build the method on them. Two points stand out. First, the entire regime tops out at 0.33 — well below dense distillation's 0.81 — so no arrangement of these components is competitive with dense correspondence. Second, on the stronger P4TRANSFORMER backbone with multiple seeds, the pairwise differences among these variants fell within the $\sim$ 0.04 single-seed noise floor, i.e. they are not robust. We therefore treat the dense correspondence, not the forecasting machinery, as the contribution.

Table 20. Dense distillation across backbones (MSR-Action3D linear-probe acc.). The official P4TRANSFORMER's pooling discards per-point structure, so the same dense objective underperforms POINT4D regardless of normalization or token granularity. 3-seed mean±std where available.

<table><tr><td>Backbone (dense distillation)</td><td>LP acc.</td></tr><tr><td>Point4D (main)</td><td>0.813 ± 0.021</td></tr><tr><td>P4Transformer, official pooling</td><td>~0.59</td></tr><tr><td>+ final LayerNorm</td><td>0.600 ± 0.031</td></tr><tr><td>+ finer tokens (less pooling)</td><td>0.607 ± 0.022</td></tr></table>

## C. Dense distillation on P4TRANSFORMER

We applied the same dense correspondence distillation to the official P4TRANSFORMER backbone (its CUDA point-4D-convolution ops). Across four variants, the linear probe lands near 0.60 — far below POINT4D's 0.813 under the identical objective, teacher, and budget (Table 20). A final LayerNorm fixes a feature-magnitude explosion (the scale-invariant cosine loss otherwise lets the unnormalized transformer output drift to standard deviation $\sim 80$ ) but does not move the probe; eval-time feature standardization only reaches 0.62; and a less-pooled (finer-token) configuration does not help either. This is most consistent with the backbone's pooled tokenization — designed to funnel a clip toward a single classification token — which leaves little per-point structure for the dense objective, rather than with optimization or feature scale (cf. the main paper's discussion). We keep the dependency-free POINT4D as the main backbone for this reason.

## D. Additional qualitative visualizations

Each map colors every input point by a 3-component PCA (mapped to RGB) of its nearest encoder token's feature — the same rule used for the main paper's Fig. 2 — with the encoder frozen.

## References

[1] Vivek Alumootil and Tuan-Anh Vu. DePT3R: Joint dense point tracking and 3d reconstruction of dynamic scenes in a single forward pass. Computing Research Repository, arXiv Preprints, arXiv:2512.13122, pages 1–11, 2025. 9

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. 1, 2, 4

[3] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus,

![](images/da10caccb4db051ee717f0901fa428da8cbad2be3d227d9deea611a6f745f2e4.jpg)  
Figure 7. The dense distillation pipeline on one MSR-Action3D frame: the input point cloud is (i) rendered to a depth-shaded image and fed to DINOv2, giving (ii) DINOv2 patch features (PCA→RGB; the subject separates cleanly from the background), which are (iii) back-projected onto the 3D points via the render correspondence. (ii) and (iii) share one PCA color space, so the subject's colors match in both.

![](images/74faf458a2cd91c4b9fd7c372a5a54910749c4d8a809bf375a085198dc46841e.jpg)

Figure 8. Dense vs. global vs. random per-point features (MSR-Action3D, six actions). Extended version of the main paper figure: each input point is colored by the PCA-to-RGB of its nearest encoder token. The dense encoder forms consistent body-part regions (distinct torso, legs, arms, head) repeated across actions; the global-distilled encoder collapses toward a blobber, near-uniform per-clip coloring; and random init is unstructured — a qualitative view of why global distillation discards the per-point structure a dense task needs (cf. the HOI4D segmentation result in the main paper). Rows are encoders, columns are actions; each encoder row has its own PCA color space (fit over all clips), so hues are comparable across actions within a row but not across the encoder rows — the comparison is spatial coherence.

Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. Computing Research Repository, arXiv Preprints, arXiv:2506.09985, pages 1–48, 2025. 1, 2, 3, 4, 5

[4] Adrien Bardes, Jean Ponce, and Yann LeCun. VI-CReg: Variance-invariance-covariance regularization for self-supervised learning. In Proceedings of the 10th International Conference on Learning Representations, pages 1–23, 2022. 4

[5] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Computing Research Repository, arXiv Preprints, arXiv:2404.08471, pages 1–28, 2024. 2, 4

[6] Yujin Chen, Matthias Nießner, and Angela Dai. 4DContrast: Contrastive learning with dynamic correspondences for 3D scene understanding. In Proceedings of the 2022 European Conference on Computer Vision, pages 1–17, 2022. 1, 2

[7] Runpei Dong, Zekun Qi, Linfeng Zhang, Junbo Zhang, Jianjian Sun, Zheng Ge, Li Yi, and Kaisheng Ma. Autoencoders as cross-modal teachers: Can pretrained 2D image transformers help 3D representation learning? In Proceedings of the 11th International Conference on Learning Representations, pages 1–20, 2023. 2

[8] Hehe Fan, Yi Yang, and Mohan Kankanhalli. Point 4D transformer networks for spatio-temporal modeling in point cloud videos. In Proceedings of the 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14199-14208, 2021. 1, 2, 3

[9] Hehe Fan, Xin Yu, Yuhang Ding, Yi Yang, and Mohan Kankanhalli. PSTNet: Point spatio-temporal convolution on point cloud sequences. In Proceedings of the 9th International Conference on Learning Representations, pages 1–23, 2021. 1, 2

[10] Siyuan Huang, Yichen Xie, Song-Chun Zhu, and Yixin Zhu. Spatio-temporal self-supervised representation learning for 3d point clouds. In Proceedings of the 2021 IEEE/CVF International Conference on Computer Vision, pages 6515–6525, 2021. 1, 2

[11] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4D: Learning how things move in 3D from internet stereo videos. In In proceedings of the 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10497–10509, 2025. 9

[12] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: Language embedded radiance fields. In Proceedings of the 2023 IEEE/CVF International Conference on Computer Vision, pages 19672-19682, 2023. 2

[13] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing NeRF for editing via feature field distillation. Advances in Neural Information Processing Systems, 35:1–26, 2022. 2

[14] Wanqing Li, Zhengyou Zhang, and Zicheng Liu. Action recognition based on a bag of 3D points. In Proceedings of the 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, pages 9–14, 2010. 1, 2, 4

[15] Yang Li, Hikari Takehara, Takafumi Taketomi, Bo Zheng, and Matthias Nießner. 4DComplete: Non-rigid motion estimation beyond the observable surface. In Proceedings of the 2021 IEEE/CVF International Conference on Computer Vision, pages 12686–12696, 2021. 2, 4

[16] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. HOI4D: A 4D egocentric dataset for category-level human-object interaction. In Proceedings of the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20981–20990, 2022. 1, 2, 4

[17] Youquan Liu, Lingdong Kong, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, and Ziwei Liu. Segment any point cloud sequences by distilling vision foundation models. Advances in Neural Information Processing Systems, 36(1617):37193–37229, 2023. 2

[18] Yunze Liu, Changxi Chen, Zifan Wang, and Li Yi. CrossVideo: Self-supervised cross-modal contrastive learning for point cloud video understanding. In Proceedings of the 2024 IEEE International Conference on Robotics and Automation, pages 12436–12442, 2024. 1, 2, 5

[19] Yueh-Cheng Liu, Yu-Kai Huang, Hung-Yueh Chiang, Hung-Ting Su, Zhe-Yu Liu, Chin-Tang Chen, Ching-Yu Tseng, and Winston H. Hsu. Learning from 2D: Contrastive pixel-to-point knowledge transfer for 3D pretraining. Computing Research Repository, arXiv Preprints, arXiv:2104.04687, pages 1–10, 2021. 2

[20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of the 7th International Conference on Learning Representations, pages 1–19, 2019. 5

[21] Trung Thanh Nguyen, Yasutomo Kawanishi, Vijay John, Takahiro Komamizu, and Ichiro Ide. View-aware cross-modal distillation for multi-view action recognition. In Proceedings of the 2026 IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7769–7778, 2026. 2

[22] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, pages 1–32, 2024. 1, 2, 3, 4

[23] Yatian Pang, Wenxiao Wang, Francis E. H. Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In Proceedings of the 2022 European Conference on Computer Vision, pages 604–621, 2022. 2

[24] Songyou Peng, Kyle Genova, Chiyu Max Jiang, Andrea Tagliasacchi, Marc Pollefeys, and Thomas Funkhouser.

OpenScene: 3D scene understanding with open vocabularies. In Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 815–824, 2023. 2

[25] Avishka Perera, Kumal Hewagamage, Saeedha Nazar, Kavishka Abeywardana, Hasitha Gallella, Ranga Rodrigo, and Mohamed Afham. CrossJEPA: Cross-modal joint-embedding predictive architecture for efficient 3D representation learning from 2D images. Computing Research Repository, arXiv Preprints, arXiv:2511.18424, pages 1–24, 2025. 1, 2

[26] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. PointNet: Deep learning on point sets for 3D classification and segmentation. In Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition, pages 652–660, 2017. 3, 5

[27] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. PointNet++: Deep hierarchical feature learning on point sets in a metric space. Advances in Neural Information Processing Systems, 30:5105–5114, 2017. 3, 4, 5

[28] Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Contrast with reconstruct: Contrastive 3D representation learning guided by generative pretraining. In Proceedings of the 40th International Conference on Machine Learning, pages 28223–28243, 2023. 2

[29] Ayumu Saito, Prachi Kudeshia, and Jiju Poovvancheri. Point-JEPA: A joint embedding predictive architecture for self-supervised learning on point cloud. In Proceedings of the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7348–7357, 2025. 1, 2

[30] Corentin Sautier, Gilles Puy, Spyros Gidaris, Alexandre Boulch, Andrei Bursuc, and Renaud Marlet. Image-to-lidar self-supervised distillation for autonomous driving data. In Proceedings of the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9881–9891, 2022. 2

[31] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. NTU RGB+D: A large scale dataset for 3D human activity analysis. In Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition, pages 1010-1019, 2016. 1, 2, 4

[32] Zhiqiang Shen, Xiaoxiao Sheng, Hehe Fan, Longguang Wang, Yulan Guo, Qiong Liu, Hao Wen, and Xi Zhou. Masked spatio-temporal structure prediction for self-supervised learning on point cloud videos. In Proceedings of the 2023 IEEE/CVF International Conference on Computer Vision, pages 16534–16543, 2023. 1, 2, 7, 8

[33] Zhiqiang Shen, Xiaoxiao Sheng, Longguang Wang, Yulan Guo, Qiong Liu, and Xi Zhou. PointCMP: Contrastive mask prediction for self-supervised learning on point cloud videos. In Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1212-1222, 2023. 1, 2

[34] Xiaoxiao Sheng, Zhiqiang Shen, and Gang Xiao. Contrastive predictive autoencoders for dynamic point cloud self-supervised learning. In Proceedings of the 37th AAAI Conference on Artificial Intelligence and 35th Conference

on Innovative Applications of Artificial Intelligence and 13th Symposium on Educational Advances in Artificial Intelligence, pages 9802–9810, 2023. 2

[35] Xiaoxiao Sheng, Zhiqiang Shen, Gang Xiao, Longguang Wang, Yulan Guo, and Hehe Fan. Point contrastive prediction with semantic clustering for self-supervised learning on point cloud videos. In Proceedings of the 2023 IEEE/CVF International Conference on Computer Vision, pages 16469-16478, 2023. 2

[36] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3. Computing Research Repository, arXiv Preprints, arXiv:2508.10104, pages 1–67, 2025. 5

[37] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in Neural Information Processing Systems, 36(732):10078–10093, 2022. 2

[38] Haiyan Wang, Liang Yang, Xuejian Rong, Jinglun Feng, and Yingli Tian. Self-supervised 4D spatio-temporal feature learning via order prediction of sequential point cloud clips. In Proceedings of the 2021 IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3761–3770, 2021. 2

[39] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. VideoMAE V2: Scaling video masked autoencoders with dual masking. In Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14549–14560, 2023. 2, 5

[40] Hao Wen, Yunze Liu, Jingwei Huang, Bo Duan, and Li Yi. Point primitive Transformer for long-term 4D point cloud video understanding. In Proceedings of the 2022 European Conference on Computer Vision, Part XXIX, pages 19–35, 2022. 1, 2

[41] Haichao Zhang, Yijiang Li, Shwai He, Tushar Nagarajan, Mingfei Chen, Jianglin Lu, Ang Li, and Yun Fu. ThinkJEPA: Empowering latent world models with large vision-language reasoning model. Computing Research Repository, arXiv Preprints, arXiv:2603.22281, pages 1–21, 2026. 2

[42] Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, and Hongsheng Li. Point-M2AE: Multi-scale masked autoencoders for hierarchical point cloud pre-training. Advances in Neural Information Processing Systems, 36(1962):27061–27074, 2022. 2

[43] Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, and Hongsheng Li. Learning 3D representations from 2D pre-trained models via image-to-point masked autoencoders. In Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21769–21780, 2023. 2

[44] Yujia Zhang, Xiaoyang Wu, Yixing Lao, Chengyao Wang, Zhuotao Tian, Naiyan Wang, and Hengshuang Zhao. Con-

certo: Joint 2D-3D self-supervised learning emerges spatial representations. Advances in Neural Information Processing Systems, 38:1–25, 2025. 1, 2

[45] Zhuoyang Zhang, Yuhao Dong, Yunze Liu, and Li Yi. Complete-to-partial 4D distillation for self-supervised point cloud sequence representation learning. In Proceedings of 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17661-17670, 2023. 1, 2

[46] Haoran Zhu and Anna Choromanska. Self-supervised JEPA-based world models for LiDAR occupancy completion and forecasting. Computing Research Repository, arXiv Preprints, arXiv:2602.12540, pages 1–9, 2026. 2

[47] Haoran Zhu, Zhenyuan Dong, Kristi Topollai, Beiyao Sha, and Anna Choromanska. AD-L-JEPA: Self-supervised representation learning with joint embedding predictive architecture for automotive LiDAR object detection. In Proceedings of the 40th AAAI Conference on Artificial Intelligence, pages 13925–13933, 2026. 2

[48] Zhi Zuo, Chenyi Zhuang, Pan Gao, Jie Qin, Hao Feng, and Nicu Sebe. Uni4D: A unified self-supervised learning framework for point cloud videos. In Proceedings of the 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1116–1126, 2025. 1, 2, 7, 8