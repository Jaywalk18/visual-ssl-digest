# Wat3R: Underwater 3D Geometry Learning without Annotations

Jiangwei Ren, Xingyu Jiang<sup>†</sup>, Zijie Song, Wei Xu, Hongkai Lin, Dingkang Liang, and Xiang Bai

Huazhong University of Science and Technology {jwren,jiangxy998,dkliang,xbai}@hust.edu.cn

Abstract. Estimating 3D geometry in underwater environments presents unique challenges due to light attenuation, scattering, and the absence of large-scale, high-quality 3D annotations. Pioneering methods rely on massive dense annotations that are impractical in underwater settings. In this paper, we propose Wat3R, a cross-domain semi-supervised learning framework designed to adapt feed-forward 3D reconstruction models from air to underwater scenes. Uniquely, our method eliminates the need for any annotated underwater data following a teacher-student architecture, that learns robust geometry representations merely on abundant unlabeled real underwater video footage. We also design a cross-view consistency loss that leverages geometric cues from other views to compensate for the information degradation in the current view caused by water attenuation and scattering. Furthermore, considering the lack of comprehensive evaluation benchmarks, we construct Water3D, a diverse dataset covering various water bodies and underwater scenarios, designed for geometric task evaluation. Experimental results demonstrate that Wat3R outperforms current state-of-the-art methods in underwater multi-view depth estimation and point cloud reconstruction. The dataset and code are available at https://github.com/LSXI7/Wat3R .

Keywords: Underwater Vision, Geometry Estimation, VGGT, Crossdomain Semi-supervised Learning

## 1 Introduction

Underwater visual geometry estimation aims to recover the 3D structures, including camera poses, depths, and point clouds, from multi-view underwater imagery. This capability underpins practical applications ranging from underwater robot navigation and obstacle avoidance, to marine mapping, terrain modeling, and underwater archaeology [18,25]. Unlike on-land scenes, underwater environments present unique challenges caused by the light absorption and scattering, as well as view-dependent degradation. These physical dificulties further lead to the critical scarcity of large-scale, high-quality 3D annotations [29] in this domain, posing great challenges in training a well-performing model.

![](images/7cc0435bbd806cfdd515ab3e354131083e12dc9e7f2cb81b42be38f64b14ca88.jpg)  
Fig. 1: Wat3R reconstructs from the open-domain underwater images in a feedforward manner without requiring any underwater 3D annotations. Our Wat3R achieves significant enhancement in both single-view and multi-view tasks. Statistic results also reveal the superior performance of our Wat3R against the SOTA.

In recent years, 3D vision has witnessed a paradigm shift from classical multiview geometry pipelines to feed-forward neural reconstruction models. Advanced methods like DUSt3R [41], VGGT [40] and their successors [6, 19, 39, 47] have demonstrated remarkable performance in recovering 3D geometry in a single forward pass. These models learn strong geometric priors from massive on-land datasets with dense 3D ground truths, a.k.a. camera poses and depths. However, directly deploying these powerful models in underwater scenarios results in poor generalization due to the significant domain shift. And the dificulty in obtaining dense 3D labels for real underwater data also prevents the direct training of efective, generalizable models tailored to underwater environments.

To relieve the above dilemma, we propose Wat3R, a semi-supervised framework that adapts VGGT to the underwater domain without any underwater 3D annotations. As shown in Fig. 1, our method enables feed-forward reconstruction across underwater scenes with complex water conditions. Following a teacherstudent learning paradigm, we first simulate various underwater degradations on existing annotated on-land datasets (the original training set of VGGT) as the required labeled data, which helps to initialize strong geometric priors for underwater environments. To further refine and generalize the model to a real underwater domain, we also collect a large amount of real underwater video footage as the unlabeled training set. Besides, to address the visual degradation caused by water attenuation and scattering, we introduce the Cross-view Consistent Loss, which integrates geometric cues from other views to compensate for the information degradation in the current view. Considering the incomplete annotations and insuficient coverage of existing underwater benchmarks, we construct Water3D, which contains a wide spectrum of underwater conditions and scenarios with correct camera poses and depths as shown in Fig. 2. Extensive experiments on public datasets and our Water3D demonstrate that our Wat3R significantly outperforms strong baselines and recent feed-forward alternatives, delivering robust performance even in poor visibility underwater regions.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">#Scenes</td><td colspan="2">Annotations</td></tr><tr><td>Depth</td><td>Pose</td></tr><tr><td>SeaThru [3]</td><td>5</td><td>√</td><td>-</td></tr><tr><td>SQUID [4]</td><td>4</td><td>√</td><td>-</td></tr><tr><td>FLSea VI [29]</td><td>12</td><td>√</td><td>-</td></tr><tr><td>FLSea Stereo [29]</td><td>5</td><td>√</td><td>-</td></tr><tr><td>SeaThru-NeRF [20]</td><td>4</td><td>-</td><td>√</td></tr><tr><td>Water3D (Ours)</td><td>42</td><td>√</td><td>√</td></tr></table>

![](images/f5e32a3bf150df1d975b95b257faf498bd2fc11c32814d23fd28f0bde8f90064.jpg)  
Fig. 2: Overview of our constructed Water3D dataset. Left: point cloud visualization and background color distribution of Water3D. Right: comparison of existing underwater datasets in terms of scene scale and available annotations. Our Water3D consists of various underwater conditions with both depth and pose annotations.

In summary, our contributions are as follows:

– We propose Wat3R, the first semi-supervised VGGT-based framework that generalizes to diverse underwater scenes without requiring underwater 3D annotations. It ofers a practical paradigm for geometric learning in other complex environments, just leveraging unlabeled real videos.

– We introduce a Cross-View Consistent Loss, which makes it easy to learn geometry cues from heavily degraded regions by aggregating the information from other related views as compensation.

We construct Water3D, an underwater multi-view dataset with comprehensive 3D annotations. It contains diverse underwater conditions.

## 2 Related Work

## 2.1 Feed-forward 3D Reconstruction Models

Classical 3D reconstruction is built on multi-view geometry and optimizationbased pipelines. Wherein the camera poses and scene structure are recovered through feature matching, epipolar geometry, and global refinement [1, 9–11,

14, 32, 33]. These systems achieve high accuracy and scalability. However, they rely on complex multi-stage pipelines and iterative optimization, leading to high computational cost and limited robustness in challenging imaging conditions like underwater scenes. Recent work reforms 3D reconstruction toward feedforward inference. DUSt3R [41] initiates this paradigm by directly regressing dense geometry from image pairs without known camera poses. MASt3R [19] strengthens this 3D prior for correspondence and alignment. Subsequent methods [6, 28, 39, 42, 47] progressively extend this framework toward multi-view inference, higher eficiency, and SLAM-oriented applications. VGGT [40] further improves reconstruction accuracy through large-scale training and multi-task joint optimization. More recently, MapAnything [17] expands the input space by incorporating multiple geometric inputs, while Depth Anything 3(DA3) [22] advances large-scale generalization with unified depth–ray modeling. However, the success of these models critically depends on massive labeled 3D datasets. In underwater scenes, obtaining those geometric annotations is dificult due to the poor visibility and unknown underwater imaging, which fundamentally limits the applicability of data-hungry reconstruction models.

## 2.2 Underwater Vision and Geometry Perception

Underwater visual perception is strongly influenced by the physical image formation process, where light attenuation and scattering significantly degrade visual signals. A large body of work [3,15,26] has focused on underwater image restoration and enhancement, aiming to remove color distortion and scattering efects to improve visual quality. Physically grounded underwater imaging models [2] therefore play a central role in characterising underwater visual environments. Several reconstruction and perception frameworks [20, 37, 49] explicitly incorporate these models to jointly reason about scene geometry and underwater light transport. Recent work [45] further shows that incorporating underwater imaging priors can improve LMMs for the understanding of underwater scenes.

Due to the scarcity of large-scale underwater datasets with accurate 3D annotations, several studies explore synthetic data generation using physics-based rendering engines [27,43] or generative models [23,50]. However, these approaches are typically designed for stereo or monocular perception tasks and often exhibit domain discrepancies when transferred to real underwater environments. NeRF [20,34,52] and 3D Gaussian Splatting methods [21,38,46] have also been applied to underwater scene reconstruction. These methods benefit from the compatibility between volumetric rendering formulations and underwater imaging models, enabling joint modeling of scene geometry and light transport. Nevertheless, they generally rely on known camera poses or sparse geometric priors and require optimization-based reconstruction. In contrast, our work focuses on feed-forward multi-view geometry learning in underwater scenes without requiring underwater 3D annotations. For this purpose, we introduce a cross-domain semi-supervised learning framework together with a cross-view consistency loss to address underwater degradation for better training.

![](images/370ababaf52441ff2be166ce6a20d5ce6ef2599b11bfd3346af0c45c94f20fcb.jpg)  
Fig. 3: Overview of our Wat3R framework. Our pipeline follows a Mean Teacher semi-supervised paradigm, where the teacher network produces pseudo-labels for depth, camera parameters, and point maps to supervise the student network. Training leverages labeled synthetic underwater data together with unlabeled real underwater videos, enabling adaptation without underwater 3D annotations. Additional per-view and cross-view consistency losses enforce multi-view geometric coherence.

## 3 Method

This paper aims to recover camera poses, depths and point clouds from collected underwater views in a single feed-forward pass. To achieve this, several key challenges we may face: (i) How to train a model under limited or no 3D annotations for underwater scenes. (ii) How to restrain the information loss caused by unknown underwater degradation. In the following, we will alleviate these concerns by applying cross-domain semi-supervised learning onto the advanced pretrained visual geometry model. Specifically, we first introduce the problem definition and notation of the base model VGGT [40], then describe the overall framework and training strategy, and finally detail our consistency-aware losses.

## 3.1 Problem Definition and Notation

Given a sequence of underwater RGB frames $( I _ { i } ) _ { i = 1 } ^ { N }$ , the target of our model is to predict a set of geometric attributes $y _ { i }$ for each view i, formulated as:

$$
(y _ {i}) _ {i = 1} ^ {N} = f \big ((I _ {i}) _ {i = 1} ^ {N} \big),\tag{1}
$$

where $y _ { i } = ( g _ { i } , D _ { i } , P _ { i } )$ denotes the predicted geometric attributes of view i corresponding to the camera parameters, depth map, and point map, respectively. $g _ { i } \in \mathbb { R } ^ { 9 }$ encodes the camera pose between view i and the first camera, containing the rotation quaternion, translation vector, and field of view. $D _ { i } \in \mathbb { R } ^ { H \times W }$ denotes the depth map; and $P _ { i } \in \mathbb { R } ^ { 3 \times H \times W }$ is the predicted 3D point map expressed in the coordinate frame of the first camera. Following this feed-forward framework, VGGT [40] is known as the pioneering work, which is trained on large-scale 3D-annotated on-land datasets and has shown remarkable performance. Therefore, we chose VGGT as the base model such that it can provide strong geometric priors when adapting to underwater environments. However, it is impractical to directly train VGGT for underwater adaptation due to the lack of 3D annotated training sets.

## 3.2 Cross-domain Semi-supervised Learning

Considering the lack of underwater 3D annotations, we investigate a semi-supervised learning framework that adapts a pretrained multi-view geometry model from land to underwater environments. Our key idea is to transfer geometric priors to underwater scenes through consistency-driven teacher–student training. As shown in Fig. 3, the framework follows a Mean Teacher [35] design, where the teacher provides stable pseudo geometric supervision while the student learns from both labeled and unlabeled data.

To be specific, we first simulate various underwater degradations on existing annotated on-land datasets (the original training set of VGGT) as the required labeled data, which helps to initialize strong geometric priors for underwater environments. We then collect a large amount of real underwater video footage as the unlabeled training set, that helps to further refine and generalize the model to the real underwater domain. During training, the student network is updated through backpropagation, while the teacher follows an exponential moving average (EMA) of the student’s parameters with the update rule:

$$
\theta_ {t} ^ {k} = (1 - \lambda) \cdot \theta_ {t} ^ {k - 1} + \lambda \cdot \theta_ {s} ^ {k},\tag{2}
$$

where $\theta _ { s } ^ { k }$ and $\theta _ { t } ^ { k }$ are the student parameters and teacher parameters in iteration $k ,$ respectively. λ is the smoothing coeficient.

Synthesized Underwater Data (Labeled). We generate underwater-style training images using a simplified version of the revised underwater imaging model [2]. The image formation is modeled as:

$$
I = J e ^ {- \beta^ {D} z} + B ^ {\infty} (1 - e ^ {- \beta^ {B} z}),\tag{3}
$$

where J and I denote the clear image and its underwater degradation and z is the depth of the scene. $B ^ { \infty } \in \mathbb { R } ^ { 3 }$ denotes the background light, while $\beta ^ { D } \in \mathbb { R } ^ { 3 }$ and $\beta ^ { \bar { B } } \in \mathbb { R } ^ { 3 }$ model the attenuation of direct transmission and backscatter, respectively. All three parameters are randomly sampled within [0, 1]. The multi-view datasets often contain incomplete or noisy depth annotations, which can create partial water-rendering artifacts when used directly in the image formation model. Therefore, we estimate dense depth maps using the monocular depth model DA3MONO-LARGE [22] for rendering the underwater appearance, while keeping the original geometric annotations as the supervised targets. To control the visible range of each scene, the depth maps are linearly rescaled to a random physical range. The attenuation coeficients are sampled while enforcing the physical constraint that red attenuates faster than green and blue [3]. To simulate spatially varying attenuation, the sampled coeficients are modulated by a smooth random spatial map obtained by repeatedly smoothing Gaussian noise. Several visual results of synthetic underwater can be seen in the bottom left of Fig. 3, which shows the good simulation of diferent underwater degradations.

Real Underwater Video Data (Unlabeled). To capture the diverse visual conditions of real underwater environments, we construct a large-scale video collection combining a curated scientific dataset with in-the-wild footage. We collect around 10, 000 raw underwater videos from public sources, but only 5,504 clips are left after manual filtering. We retain clips with continuous shots, visible static structures, reasonable camera motion, suficient view overlap, and usable image quality. We remove surface or aerial videos, edited clips with frequent cuts, near duplicates, severe blur or compression, dominant overlays, and clips dominated by open water or moving objects. Frames are sampled every ten frames to reduce temporal redundancy, yielding 359k training images in total.

Sequence-level Training Augmentation. As a semi-supervised learning framework, data augmentation is critical for stable training. Following single image based task [7, 51], the teacher and student models respectively receive weak and strong augmentations of the same frame. However, the unsupervised training on unlabeled video data often results in limited viewpoint variation. Therefore, sequence-level augmentation is necessary to increase geometric diversity and prevent model collapse. Specifically, we first sample 24−36 frames and shufle the order before feeding them to the teacher. For the student branch, the same frames are randomly subsampled to 2−12 images, shufled again, and each image is rotated by 0<sup>◦</sup>, 90<sup>◦</sup>, 180<sup>◦</sup>, or 270<sup>◦</sup> to avoid pose collapse. We additionally apply color jittering, grayscale conversion, and Gaussian blur.

## 3.3 Consistency-aware Training Objective

3D reconstruction data generated by SfM pipelines or rendering engines inherently satisfy strong multi-view geometric consistency. To preserve this structural property during training, we introduce additional consistency objectives that encourage coherent geometric predictions. Our training objective consists of three components: a supervised loss following VGGT, a per-view consistency loss using teacher predictions as pseudo-labels, and a cross-view consistency loss enforcing multi-view geometric coherence. The overall training objective is defined as:

$$
\mathcal {L} _ {\mathrm{total}} = \mathcal {L} _ {\mathrm{supervised}} + \lambda_ {p} \mathcal {L} _ {\mathrm{per-view}} + \lambda_ {c} \mathcal {L} _ {\mathrm{cross-view}}.\tag{4}
$$

Supervised Loss. The supervised loss is applied when training on labeled synthetic data, which is computed between the student predictions and the groundtruth annotations $\{ g ^ { g t } , D ^ { g t } , P ^ { g t } \}$ . Here we directly use the loss in VGGT [40]:

$$
\mathcal {L} _ {\mathrm{supervised}} = \mathcal {L} _ {\mathrm{camera}} (g ^ {s}, g ^ {g t}) + \mathcal {L} _ {\mathrm{depth}} (D ^ {s}, D ^ {g t}) + \mathcal {L} _ {\mathrm{point}} (P ^ {s}, P ^ {g t}),\tag{5}
$$

where the depth loss includes regression, confidence, and gradient terms; the point loss consists of regression, confidence, and normal terms; and the camera loss is computed using Huber loss. Details can refer to VGGT [40].

Per-view Geometry Consistency Loss. We use this loss when learning from unlabeled real data. It shares the same formulation as the supervised loss by regarding the predictions of the teacher model, i.e., $\{ g ^ { t } , D ^ { t } , P ^ { t } \}$ , as pseudo ground truth. Notably, point maps are projected from depth and camera parameters for more stable geometric supervision [40]. The per-view loss is defined as:

$$
\mathcal {L} _ {\mathrm{per-view}} = \mathcal {L} _ {\mathrm{camera}} (g ^ {s}, g ^ {t}) + \mathcal {L} _ {\mathrm{depth}} (D ^ {s}, D ^ {t}) + \mathcal {L} _ {\mathrm{point}} (P ^ {s}, P ^ {t}).\tag{6}
$$

Cross-view Geometry Consistency Loss. Underwater scattering and attenuation often lead to poor visibility and limited geometric information. To address this challenge, we introduce a cross-view geometry consistency loss that explicitly integrates geometric cues from other views to compensate for the information degradation in the current view. Given N teacher frames, we first obtain a coarse foreground mask $M _ { i } ^ { \mathrm { f g } }$ for each frame i using a simple two-cluster K-means on the depth map, keeping the closer cluster as foreground. For every frame i, we backproject valid teacher pixels into 3D and reproject them into all other teacher views. Define the projected pixel and the visibility indicator as:

$$
\begin{array}{l} V _ {i \to j} (x) = \mathbf {1} \big (| D _ {j} ^ {t} (u _ {i \to j} (x)) - D _ {i \to j} ^ {t} (x) | <   \delta \land u _ {i \to j} (x) \in \Omega_ {j} \big), \\ u _ {i \to j} (x) = \pi_ {j} \big (\pi_ {i} ^ {- 1} (x, D _ {i} ^ {t} (x)) \big). \end{array}\tag{7}
$$

Here, $V _ { i \to j } ( x )$ indicates whether the pixel x in frame i remains visible and depthconsistent after being projected into frame $j , \ 1 ( \cdot )$ returns 1 if it is true. $D _ { i } ^ { t }$ denotes the teacher depth for view $i . \ \varOmega _ { j }$ is the image domain of view $j . \ u _ { i \to j } ( x )$ indicates the projection from view i to view $j$ . Wherein $\pi _ { j }$ is the projection operator that projects the depth of view $j$ into the world coordinate system using the camera pose, and $\pi _ { i } ^ { - 1 }$ is the back-projection operator. Then the statistic region of view i is computed as the depth-consistent part of its foreground mask:

$$
M _ {i} ^ {\text { static }} (x) = \mathbf {1} \left(\sum_ {j \neq i} V _ {i \rightarrow j} (x) \geq k\right) M _ {i} ^ {\text { fg }} (x).\tag{8}
$$

We use the conservative setting $k = N - 2$ in our experiments, where N denotes the number of teacher frames used to construct the mask. This mask is not an additional supervision source; it only selects reliable pixels for applying cross-view supervision. The foreground term suppresses distant background water regions, while the multi-view consistency term removes dynamic or geometrically unstable regions. When the scene contains mostly open water, severe turbidity, or strong object motion, the mask naturally becomes sparse and therefore prevents the model from enforcing unreliable cross-view constraints.

During student training, we randomly select n frames from the same sequence (in arbitrary order), forming an index set S. For any pair $( i , j ) \in S$ , depth from teacher frame i is projected into view $j \colon$ :

$$
D _ {i \rightarrow j} ^ {t} (x) = \pi_ {j} \left(\pi_ {i} ^ {- 1} (x, D _ {i} ^ {t} (x))\right),\tag{9}
$$

and compared with the student depth $D _ { j } ^ { s }$ at frame $j$ under the static mask $M _ { i } ^ { \mathrm { s t a t i c } }$ . The geometry-consistent loss averages over all ordered pairs:

$$
\mathcal {L} _ {\mathrm{cross-view}} = \frac {1}{| \mathcal {S} | ^ {2}} \sum_ {i \in \mathcal {S}} \sum_ {j \in \mathcal {S}, j \neq i} \mathcal {L} _ {\mathrm{L1}} \big (D _ {j} ^ {s}, D _ {i \rightarrow j} ^ {t}, M _ {j} ^ {\mathrm{static}} \big).\tag{10}
$$

Table 1: Multi-view depth estimation. We report results under two evaluation protocols: (1) Shufle 10-view evaluation, and (2) Full subsequence evaluation (100 frames, ordered). The best and second of each category are masked as Bold and Underline, respectively. Shaded rows denote two-stage pipelines, where input images are first enhanced by the corresponding underwater image enhancement method and then fed into VGGT model. Values in <sub>(</sub>red<sub>)</sub> indicate the percentage improvement over VGGT.

<table><tr><td rowspan="2">Methods</td><td colspan="4">Sea-thru [3]</td><td colspan="4">FLSea Stereo [29]</td></tr><tr><td>Rel↓</td><td> $\delta_1 \uparrow$ </td><td> $\log_{10} \downarrow$ </td><td>RMSE↓</td><td>Rel↓</td><td> $\delta_1 \uparrow$ </td><td> $\log_{10} \downarrow$ </td><td>RMSE↓</td></tr><tr><td colspan="9">Shuffle 10-view Evaluation (stride=10)</td></tr><tr><td>WaterSplatting [21] 3DV&#x27;25</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.427</td><td>0.415</td><td>0.157</td><td>1.476</td></tr><tr><td>Fast3r [47] CVPR&#x27;25</td><td>0.277</td><td>0.713</td><td>0.083</td><td>0.631</td><td>0.290</td><td>0.529</td><td>0.141</td><td>1.355</td></tr><tr><td>MapAnything [17] 3DV&#x27;26</td><td>0.216</td><td>0.800</td><td>0.060</td><td>0.454</td><td>0.146</td><td>0.841</td><td>0.061</td><td>0.798</td></tr><tr><td> $\pi 3$  [42] ICLR&#x27;26</td><td>0.185</td><td>0.909</td><td>0.044</td><td>0.358</td><td>0.139</td><td>0.856</td><td>0.053</td><td>0.837</td></tr><tr><td>DA3 [22] ICLR&#x27;26</td><td>0.187</td><td>0.892</td><td>0.046</td><td>0.333</td><td>0.141</td><td>0.851</td><td>0.056</td><td>0.872</td></tr><tr><td>VGGT [40] CVPR&#x27;25</td><td>0.190</td><td>0.891</td><td>0.047</td><td>0.380</td><td>0.137</td><td>0.849</td><td>0.059</td><td>0.760</td></tr><tr><td>+ Semi-UIR [15] CVPR&#x27;23</td><td>0.201</td><td>0.846</td><td>0.052</td><td>0.387</td><td>0.136</td><td>0.861</td><td>0.056</td><td>0.784</td></tr><tr><td>+ PSPL [26] TIP&#x27;25</td><td>0.209</td><td>0.836</td><td>0.055</td><td>0.419</td><td>0.160</td><td>0.817</td><td>0.063</td><td>0.911</td></tr><tr><td>Wat3R</td><td>0.167(+12.1%)</td><td>0.946(+6.2%)</td><td>0.038(+19.1%)</td><td>0.290(+23.7%)</td><td>0.119(+13.1%)</td><td>0.885(+4.2%)</td><td>0.048(+18.6%)</td><td>0.720(+5.3%)</td></tr><tr><td colspan="9">Full Subsequence Evaluation (100 frames, ordered)</td></tr><tr><td>Fast3r [47] CVPR&#x27;25</td><td>0.274</td><td>0.711</td><td>0.081</td><td>0.622</td><td>0.299</td><td>0.521</td><td>0.146</td><td>1.363</td></tr><tr><td>MapAnything [17] 3DV&#x27;26</td><td>0.227</td><td>0.788</td><td>0.063</td><td>0.488</td><td>0.147</td><td>0.841</td><td>0.062</td><td>0.795</td></tr><tr><td> $\pi 3$  [42] ICLR&#x27;26</td><td>0.187</td><td>0.923</td><td>0.041</td><td>0.358</td><td>0.139</td><td>0.856</td><td>0.053</td><td>0.836</td></tr><tr><td>DA3 [22] ICLR&#x27;26</td><td>0.183</td><td>0.922</td><td>0.040</td><td>0.311</td><td>0.139</td><td>0.855</td><td>0.055</td><td>0.861</td></tr><tr><td>VGGT [40] CVPR&#x27;25</td><td>0.193</td><td>0.900</td><td>0.045</td><td>0.373</td><td>0.136</td><td>0.851</td><td>0.058</td><td>0.759</td></tr><tr><td>Wat3R</td><td>0.170(+11.9%)</td><td>0.953(+5.9%)</td><td>0.036(+20.0%)</td><td>0.294(+21.2%)</td><td>0.120(+11.8%)</td><td>0.884(+3.9%)</td><td>0.048(+17.2%)</td><td>0.715(+5.8%)</td></tr></table>

By now, we have introduced the main process of the proposed Wat3R, a cross-domain semi-supervised learning framework to generalize VGGT to the challenging underwater domain without requiring any underwater annotations.

## 4 Experiments

Next, we evaluate our Wat3R on four underwater tasks: multi-view depth estimation (Sec. 4.2), point map estimation (Sec. 4.3), camera pose estimation (Sec. 4.4), and monocular depth estimation (Sec. 4.5).

## 4.1 Implementation Details

We train the model on 4 NVIDIA RTX 4090 GPUs for 19,200 steps, beginning with a 1,000-step linear warm-up. The peak learning rates are set to $5 \times 1 0 ^ { - 6 }$ for the ViT backbone and $5 \times 1 0 ^ { - 5 }$ for the downstream head. To improve training eficiency and reduce GPU memory consumption, we employ gradient checkpointing and bfloat16 mixed-precision training. The ratio of unlabeled to labeled samples is set to 1:3. For the first 6,400 steps, training is conducted exclusively on labeled data. The unsupervised loss weight $\lambda _ { u }$ is then gradually increased, reaching its peak value of 0.5 at step 12,800. During training, we randomly sample 2 to 12 images, with the longest side of each image resized to 518 pixels.

Datasets. To ensure applicability to underwater vision scenarios, we conduct experiments on five public datasets together with our own constructed Water3D:

![](images/74c40224c0806981913b20926a6f4121d814bcfa79bc1b7f063a2d815a60098c.jpg)  
Fig. 4: 3D reconstruction results on our constructed Water3D. DA3 [22], MapAnything [17], VGGT [40] and our Wat3R are used for comparison. Wat3R obtains more complete and physically consistent reconstructions with reduced artifacts.

– FLSea-VI [29] contains forward-looking monocular underwater sequences captured in shallow Mediterranean and Red Sea environments. We evaluate on the Coral Table Loop and U-Canyon sequences.

– FLSea-Stereo [29] comprises 7,341 synchronized stereo image pairs from 5 shallow-water scenes, including canyon and flat sandy seabed environments, collected from the FLSea-VI dataset in diferent dives and scenes.

– SQUID [4] provides a quantitative underwater stereo dataset consisting of 57 stereo image pairs with stereo-derived ground-truth depth, collected at diferent seasons, depths, and water types in natural marine environments.

– Sea-thru [3] is an underwater RGBD dataset composed of 1,205 images from 5 natural scenes, acquired in two optically diferent water bodies under natural illumination, spanning depths, water types, and scene structures.

– SeaThru-NeRF [20] consists of 5 real underwater scenes annotated with camera poses only. This dataset is originally used for NeRF evaluation.

Water3D consists of 42 real underwater scenes, where 20 are from UVEB [44], and 22 scenes are collected alongside our training data (but not used for training). Camera poses and depths are reconstructed using COLMAP [32], wherein the matcher is changed to the advanced MINIMA [30], which is further fine-tuned with our synthetic underwater data. The final results are manually checked to make sure the correctness of annotations, where only the scenes with stable registration and no obvious collapse or severe outliers are preserved. See supplement for more details.

## 4.2 Multi-view Depth Estimation

We first evaluate the depth estimation from multi-view input. Two video datasets with depth annotations are used, i.e., Sea-thru [3] and FLSea-Stereo [29]. We compare our Wat3R with state-of-the-art models trained on large-scale annotated data. Besides, the underwater Splatting method [21] is also evaluated. To verify whether the underwater image enhancement (UIE) method is efective for 3D geometry estimation, we also take two enhancement methods into a two-stage pipeline. It first enhances underwater images to make them closer to the on-land domain, then feeds them into a 3D model. Following [40, 41, 47], long sequences are divided into multiple sub-scenes, each containing at most 100 images. And we uniformly sample 10 images from each sequence and randomly shufle their order for testing. In addition, we follow the video-based depth evaluation protocol of [8] to evaluate full subsequences. We evaluate metric depth by aligning predicted depth maps to the ground-truth scale using a least-squares estimation of scale and shift. Performance is measured using standard depth metrics, and results are reported in Tab. 1. We report absolute relative error (Rel), absolute error in log-scale $\left( \log _ { 1 0 } \right)$ , root mean squared error (RMSE), and the percentage of inlier pixels $\delta _ { i }$ under thresholds of $1 . 2 5 ^ { i }$ to evaluate overall depth accuracy. The frame order of input video is randomly shufled during training, so the model does not rely on temporal continuity or video-specific supervision.

![](images/fa89955b3f3a79113e8a0e4d4bdf89fd4eab787379983f1d85032336d7d17551.jpg)  
Fig. 5: Qualitative comparison on in-the-wild images with Wat3R, DA3 [22], MapAnything [17], and VGGT [40]. For a fair comparison, all methods take the same input. No post-processing is applied, allowing full point clouds to be visualized.

Tab. 1 reveals that our model achieves clear and consistent improvements compared to the original VGGT [40]. Specifically, WaterSplatting [21] is a Splatting method designed for underwater scenes. But it relies on initialized point clouds, producing large errors on FLSea-Stereo and even failing on the challenging Sea-thru dataset. π3 [42] and DA3 [22] obtain competitive results due to their improved 3D representations, and their large training set contains some underwater scenes. The shaded rows show that UIE almost brings no improvement for multi-view depth estimation. The enhancement process may easily introduce information loss or view-inconsistent changes, failing to recover new structural cues. In contrast, Wat3R learns underwater-aware geometry directly during training, leading to more reliable 3D perception in challenging underwater scenes. These results highlight our method’s ability to achieve superior accuracy and robustness, particularly in challenging underwater multi-view scenarios. These high-fidelity results are visually demonstrated in Fig. 4 and Fig. 5.

Table 2: Performance comparison on the Water3D dataset. The best and second of each category are masked as Bold and Underline, respectively. Values in <sub>(</sub>red<sub>)</sub> indicate the percentage improvement over VGGT [40].

<table><tr><td rowspan="2">Methods</td><td colspan="2">Accuracy ↓</td><td colspan="2">Completeness ↓</td><td colspan="2">Overall ↓</td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>Fast3r [47] CVPR&#x27;25</td><td>1.216</td><td>0.531</td><td>2.444</td><td>0.784</td><td>1.830</td><td>0.658</td></tr><tr><td>MapAnything [17] 3DV&#x27;26</td><td>0.643</td><td>0.284</td><td>0.666</td><td>0.277</td><td>0.655</td><td>0.281</td></tr><tr><td>π3 [42] ICLR&#x27;26</td><td>0.491</td><td>0.168</td><td>0.413</td><td>0.184</td><td>0.452</td><td>0.176</td></tr><tr><td>DA3 [22] ICLR&#x27;26</td><td>0.679</td><td>0.187</td><td>0.528</td><td>0.160</td><td>0.604</td><td>0.174</td></tr><tr><td>VGGT [40] CVPR&#x27;25</td><td>0.486</td><td>0.193</td><td>0.762</td><td>0.191</td><td>0.624</td><td>0.192</td></tr><tr><td>Wat3R (Point)</td><td>0.444(+8.6%)</td><td>0.148(+23.3%)</td><td>0.409(+46.3%)</td><td>0.165(+13.6%)</td><td>0.427(+31.6%)</td><td>0.157(+18.2%)</td></tr><tr><td>Wat3R (Depth+Cam)</td><td>0.446(+8.2%)</td><td>0.162(+16.1%)</td><td>0.366(+52.0%)</td><td>0.143(+25.1%)</td><td>0.406(+34.9%)</td><td>0.153(+20.3%)</td></tr></table>

## 4.3 Point Map Estimation

We further evaluate the quality of reconstructed multi-view point maps using our underwater dataset Water3D. For each sequence, 20 images are uniformly sampled for evaluation. Following [40, 42, 47], we first perform a coarse Sim(3) alignment using the Umeyama algorithm [36], followed by refinement with the Iterative Closest Point (ICP) algorithm. We report Accuracy, Completeness, and their Overall average as metrics in Tab. 2. The direct point-map output and the reconstruction derived from predicted depth and cameras both achieve strong performance. Their consistent results indicate that adaptation preserves geometric agreement among the point, depth, and camera heads. This is enabled by our consistency-aware objectives, which couple predictions across heads and views even without ground-truth annotations for real underwater videos.

## 4.4 Camera Pose Estimation

We now evaluate camera pose estimation on the SeaThru-NeRF [20]. Following prior work [40, 41], performance is measured using AUC under diferent angular thresholds, where AUC is calculated from the minimum of Relative Translation Accuracy (RTA) and Relative Rotation Accuracy (RRA).

As shown in Tab. 3, DA3 [22] achieves the best overall performance across all thresholds. This is mainly due to its depth-ray representation, which provides a minimal yet suficient formulation for jointly modeling scene geometry and camera motion. Geometry-driven pose inference exhibits natural robustness in underwater environments. Our cross-view consistency learning enables the model to extract complementary geometric cues across views, reducing its sensitivity to water-induced degradation and yielding more stable pose estimates.

## 4.5 Monocular Depth Estimation

We evaluate monocular depth estimation to assess the model’s understanding of underwater degradation. In this setting, no complementary cues from other views are available. Following [23,50], we evaluate performance on four datasets:

Table 3: Pose estimation AUC at diferent thresholds on the SeaThru-NeRF [20]. The best and second of each category are masked as Bold and Underline, respectively. Values in $( \mathbf { r e d } )$ indicate the percentage improvement over VGGT [40].

<table><tr><td>Methods</td><td>AUC@5°</td><td>AUC@15°</td><td>AUC@30°</td></tr><tr><td>Fast3r [47] CVPR&#x27;25</td><td>0.040</td><td>0.239</td><td>0.498</td></tr><tr><td>π3 [42] ICLR&#x27;26</td><td>0.216</td><td>0.635</td><td>0.809</td></tr><tr><td>DA3 [22] ICLR&#x27;26</td><td>0.731</td><td>0.901</td><td>0.950</td></tr><tr><td>MapAnything [17] 3DV&#x27;26</td><td>0.074</td><td>0.458</td><td>0.707</td></tr><tr><td>VGGT [40] CVPR&#x27;25</td><td>0.392</td><td>0.707</td><td>0.843</td></tr><tr><td>Wat3R</td><td>0.540(+37.8%)</td><td>0.820(+16.0%)</td><td>0.906(+7.5%)</td></tr></table>

Table 4: Monocular depth estimation. The best and second of each category are masked as Bold and Underline, respectively. The top three methods (Udepth [49], UW-Depth [13], WaterMono [12]) are underwater-specific models. WaterMono [12] is trained on FLSea VI [29]; its results are shown in gray. Values in $( \mathbf { r e d } )$ indicate the percentage improvement over VGGT [40].

<table><tr><td rowspan="2">Methods</td><td colspan="2">FLSea VI [29]</td><td colspan="2">FLSea Stereo [29]</td><td colspan="2">SQUID [4]</td><td colspan="2">Sea-thru [3]</td></tr><tr><td>Rel↓</td><td> $\delta_1 \uparrow$ </td><td>Rel↓</td><td> $\delta_1 \uparrow$ </td><td>Rel↓</td><td> $\delta_1 \uparrow$ </td><td>Rel↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td>Udepth [49] ICRA&#x27;23</td><td>0.212</td><td>0.683</td><td>0.279</td><td>0.550</td><td>0.312</td><td>0.547</td><td>0.166</td><td>0.832</td></tr><tr><td>UW-Depth [13] ICRA&#x27;24</td><td>0.330</td><td>0.447</td><td>0.400</td><td>0.427</td><td>0.491</td><td>0.343</td><td>0.184</td><td>0.792</td></tr><tr><td>WaterMono [12] TIM&#x27;25</td><td>0.074</td><td>0.949</td><td>0.206</td><td>0.684</td><td>0.261</td><td>0.544</td><td>0.145</td><td>0.829</td></tr><tr><td>DAv2 [48] NeurIPS&#x27;24</td><td>0.069</td><td>0.958</td><td>0.134</td><td>0.849</td><td>0.099</td><td>0.900</td><td>0.089</td><td>0.940</td></tr><tr><td>Fast3r [47] CVPR&#x27;25</td><td>0.198</td><td>0.720</td><td>0.213</td><td>0.679</td><td>0.368</td><td>0.522</td><td>0.125</td><td>0.911</td></tr><tr><td>MapAnything [17] 3DV&#x27;26</td><td>0.086</td><td>0.951</td><td>0.146</td><td>0.837</td><td>0.104</td><td>0.898</td><td>0.104</td><td>0.960</td></tr><tr><td>π3 [42] ICLR&#x27;26</td><td>0.081</td><td>0.944</td><td>0.148</td><td>0.831</td><td>0.234</td><td>0.640</td><td>0.113</td><td>0.949</td></tr><tr><td>DA3 [22] ICLR&#x27;26</td><td>0.090</td><td>0.936</td><td>0.154</td><td>0.824</td><td>0.151</td><td>0.802</td><td>0.111</td><td>0.950</td></tr><tr><td>VGGT [40] CVPR&#x27;25</td><td>0.107</td><td>0.888</td><td>0.159</td><td>0.809</td><td>0.194</td><td>0.703</td><td>0.113</td><td>0.942</td></tr><tr><td>Wat3R</td><td>0.061(+43.0%)</td><td>0.971(+9.3%)</td><td>0.120(+24.5%)</td><td>0.886(+9.5%)</td><td>0.107(+44.8%)</td><td>0.893(+27.0%)</td><td>0.090(+20.4%)</td><td>0.976(+3.6%)</td></tr></table>

FLSea-VI [29], FLSea-Stereo [29], SQUID [4] and Sea-thru [3], using Absolute Relative Error (Rel) and $\delta _ { 1 }$ as metrics. The results are summarized in Tab. 4. Although our model is not trained with single-image supervision, it achieves the best performance across four datasets. The cross-view supervision helps separate underwater degradation from scene geometry. The learned geometric prior then transfers to single-view input. Fig. 6 presents qualitative monocular results.

## 4.6 Ablation Studies

Tab. 5 presents an ablation study to analyze the contribution of each component in our framework. Starting from the pretrained VGGT model as the baseline, we progressively introduce key elements of our method, including real underwater video data, sequence-level augmentation, the proposed cross-view consistency loss $\mathcal { L } _ { \mathrm { c r o s s - v i e w } }$ , and the static mask $M ^ { \mathrm { s t a t i c } }$ . For reference, we include a supervised variant trained only on synthetic underwater data to highlight the efect of unlabeled real underwater videos and the proposed consistency constraints.

The results show three clear trends. (1) Training with synthetic underwater rendering already improves performance over the VGGT baseline by partially bridging the domain gap between terrestrial and underwater imagery. (2) Incorporating real underwater video data together with strong augmentation consistently improves performance, demonstrating that large-scale unlabeled videos provide efective supervision signals for adapting geometry models to underwater scenes. (3) The proposed cross-view consistency loss and static mask further improve performance by enforcing multi-view geometric coherence and suppressing unstable regions caused by scattering or dynamic content. Note that our masking strategy primarily deals with severe underwater degradation and dynamic objects. For simpler and static scenes like Sea-thru, the gains are not significant.

![](images/58281fd0f42d987c7b897bb6a4f58e2ec27cac374ecc0d6a4b5a5b82a9731919.jpg)  
Fig. 6: Monocular depth estimation using Wat3R, DA3 [22], and VGGT [40]. The first two rows are sampled from the SQUID [4] dataset, the third row from FLSea-Stereo [29], and the fourth row from FLSea-VI [29].

Table 5: Ablation studies. The multi-view depth estimation is evaluated. The best and second of each category are masked as Bold and Underline, respectively.

<table><tr><td rowspan="2"></td><td rowspan="2">Syn Water</td><td rowspan="2">Real Video</td><td rowspan="2">Strong Aug.</td><td rowspan="2"> $\mathcal{L}_{\text{cross-view}}$ </td><td rowspan="2"> $M^{\text{static}}$ </td><td colspan="2">Sea-thru</td><td colspan="2">FLSea Stereo</td></tr><tr><td>Rel↓</td><td> $\delta_1 \uparrow$ </td><td>Rel↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td rowspan="6">VGGT</td><td></td><td></td><td></td><td></td><td></td><td>0.190</td><td>0.891</td><td>0.137</td><td>0.849</td></tr><tr><td>✓</td><td></td><td></td><td></td><td></td><td>0.173</td><td>0.920</td><td>0.135</td><td>0.860</td></tr><tr><td>✓</td><td>✓</td><td></td><td></td><td></td><td>0.181</td><td>0.906</td><td>0.165</td><td>0.793</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td></td><td>0.172</td><td>0.936</td><td>0.126</td><td>0.871</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>0.167</td><td>0.949</td><td>0.126</td><td>0.869</td></tr><tr><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>0.174</td><td>0.929</td><td>0.130</td><td>0.871</td></tr><tr><td>Wat3R</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>0.167</td><td>0.946</td><td>0.119</td><td>0.885</td></tr></table>

## 4.7 Robustness Analysis and Failure Cases

To evaluate the robustness of the model under extreme conditions, we report results under progressively stronger underwater degradation on both synthetic and real scenes in Fig. 7. On synthetically degraded DTU [16] dataset, the point-map error of all methods increases as attenuation and scattering intensify, whereas

![](images/31bdba50ababde0c0169a2a8e71be3e1f5756fba803fe1da68289193b87a4184.jpg)  
Fig. 7: Robustness and failure case analysis. Left: synthetic underwater degradation with increasing visual corruption. Right: real underwater scenes grouped by dificulty. Wat3R shows more robust to moderate degradation.

Wat3R degrades more gradually and maintains a clear margin over the competing methods. This advantage also extends to real Water3D scenes. Severe visibility loss reduces the stable visual evidence available for geometry recovery. Nevertheless, Wat3R deteriorates more slowly. Its robustness stems from training on diverse synthetic and real underwater conditions, together with consistency objectives that encourage geometry to remain stable under appearance changes.

## 4.8 Discussion on Possible Limitations

Although Wat3R is designed for underwater geometry reconstruction without requiring underwater annotations, several limitations remain. Handling highly dynamic elements, such as moving marine life or divers, may require additional modeling of temporal dynamics and motion cues. In open-water, highly turbid or very deep scenes, reliable matched structures can become sparse. Our conservative static mask then suppresses unreliable regions, but this also reduces the cross-view learning signal and limits performance in such extreme cases.

## 5 Conclusion

In this paper, we introduced Wat3R, a first semi-supervised VGGT framework for underwater geometry learning. Wat3R learns robust geometry representations merely on unlabeled real underwater video, without the need for any annotated underwater data. And a cross-view consistency loss is designed to address underwater degradation for better training. Besides, we construct Water3D, the first comprehensive evaluation benchmark containing diverse underwater conditions. Experimental results demonstrate that Wat3R significantly outperforms the state-of-the-art in underwater multi-view geometry estimation.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant U2341227, 62406117, and U25B2078).

## References

1. Agarwal, S., Furukawa, Y., Snavely, N., Simon, I., Curless, B., Seitz, S.M., Szeliski, R.: Building rome in a day. CACM 54(10), 105–112 (2011)

2. Akkaynak, D., Treibitz, T.: A revised underwater image formation model. In: CVPR. pp. 6723–6732 (2018)

3. Akkaynak, D., Treibitz, T.: Sea-thru: A method for removing water from underwater images. In: CVPR. pp. 1682–1691 (2019)

4. Berman, D., Levy, D., Avidan, S., Treibitz, T.: Underwater single image color restoration using haze-lines and a new quantitative dataset. IEEE TPAMI (2020)

5. Bleyer, M., Rhemann, C., Rother, C.: Patchmatch stereo-stereo matching with slanted support windows. In: BMVC. vol. 11, pp. 1–11 (2011)

6. Cabon, Y., Stofl, L., Antsfeld, L., Csurka, G., Chidlovskii, B., Revaud, J., Leroy, V.: Must3r: Multi-view network for stereo 3d reconstruction. In: CVPR. pp. 1050– 1060 (2025)

7. Chen, D., Liu, Z., Yang, C., Wang, D., Yan, Y., Xu, Y., Ji, X.: Conformalsam: Unlocking the potential of foundational segmentation models in semi-supervised semantic segmentation with conformal prediction. In: ICCV. pp. 24045–24055 (2025)

8. Chen, S., Guo, H., Zhu, S., Zhang, F., Huang, Z., Feng, J., Kang, B.: Video depth anything: Consistent depth estimation for super-long videos. In: CVPR. pp. 22831– 22840 (2025)

9. Crandall, D., Owens, A., Snavely, N., Huttenlocher, D.: Discrete-continuous optimization for large-scale structure from motion. In: CVPR. pp. 3001–3008 (2011)

10. Crandall, D.J., Owens, A., Snavely, N., Huttenlocher, D.P.: Sfm with mrfs: Discrete-continuous optimization for large-scale structure from motion. IEEE TPAMI 35(12), 2841–2853 (2012)

11. Cui, H., Gao, X., Shen, S., Hu, Z.: Hsfm: Hybrid structure-from-motion. In: CVPR. pp. 1212–1221 (2017)

12. Ding, Y., Li, K., Mei, H., Liu, S., Hou, G.: Watermono: Teacher-guided anomaly masking and enhancement boosting for robust underwater self-supervised monocular depth estimation. IEEE TIM (2025)

13. Ebner, L., Billings, G., Williams, S.: Metrically scaled monocular depth estimation through sparse priors for underwater robots. In: ICRA. pp. 3751–3757 (2024)

14. Hartley, R., Zisserman, A.: Multiple view geometry in computer vision. Cambridge university press (2003)

15. Huang, S., Wang, K., Liu, H., Chen, J., Li, Y.: Contrastive semi-supervised learning for underwater image restoration via reliable bank. In: CVPR. pp. 18145–18155 (2023)

16. Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H.: Large scale multi-view stereopsis evaluation. In: CVPR. pp. 406–413 (2014)

17. Keetha, N., Müller, N., Schönberger, J., Porzi, L., Zhang, Y., Fischer, T., Knapitsch, A., Zauss, D., Weber, E., Antunes, N., et al.: Mapanything: Universal feed-forward metric 3d reconstruction. In: 3DV (2026)

18. Kim, A., Eustice, R.: Pose-graph visual slam with geometric model selection for autonomous underwater ship hull inspection. In: IROS. pp. 1559–1565 (2009)

19. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r. In: ECCV. pp. 71–91. Springer (2024)

20. Levy, D., Peleg, A., Pearl, N., Rosenbaum, D., Akkaynak, D., Korman, S., Treibitz, T.: Seathru-nerf: Neural radiance fields in scattering media. In: CVPR. pp. 56–65 (2023)

21. Li, H., Song, W., Xu, T., Elsig, A., Kulhanek, J.: Watersplatting: Fast underwater 3d scene reconstruction using gaussian splatting. In: 3DV. pp. 969–978 (2025)

22. Lin, H., Chen, S., Liew, J., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. In: ICLR (2026)

23. Lin, H., Liang, D., Qi, Z., Bai, X.: A unified image-dense annotation generation model for underwater scenes. In: CVPR. pp. 961–970 (2025)

24. Lindenberger, P., Sarlin, P.E., Pollefeys, M.: Lightglue: Local feature matching at light speed. In: ICCV. pp. 17627–17638 (2023)

25. Liu, R., Fan, S., Wang, W., Yang, Y.: Underwater visual slam with depth uncertainty and medium modeling. In: ICCV. pp. 970–980 (2025)

26. Liu, Y., Jiang, Q., Li, X., Luo, T., Ren, W.: Toward better than pseudo-reference in underwater image enhancement. IEEE TIP (2025)

27. Lv, Q., Dong, J., Li, Y., Chen, S., Yu, H., Zhang, S., Wang, W.: Uwstereo: A large synthetic dataset for underwater stereo matching. IEEE TCSVT (2025)

28. Maggio, D., Lim, H., Carlone, L.: VGGT-SLAM: Dense rgb slam optimized on the sl (4) manifold. In: NeurIPS (2025)

29. Randall, Y.: Flsea: Underwater visual-inertial and stereo-vision forward-looking datasets. Master’s thesis, University of Haifa (Israel) (2023)

30. Ren, J., Jiang, X., Li, Z., Liang, D., Zhou, X., Bai, X.: Minima: Modality invariant image matching. In: CVPR. pp. 23059–23068 (2025)

31. Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M.: From coarse to fine: Robust hierarchical localization at large scale. In: CVPR. pp. 12716–12725 (2019)

32. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: CVPR. pp. 4104–4113 (2016)

33. Snavely, N., Seitz, S.M., Szeliski, R.: Photo tourism: exploring photo collections in 3d. ACM TOG pp. 835–846 (2006)

34. Tang, Y., Zhu, C., Wan, R., Xu, C., Shi, B.: Neural underwater scene representation. In: CVPR. pp. 11780–11789 (2024)

35. Tarvainen, A., Valpola, H.: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In: NeurIPS. vol. 30 (2017)

36. Umeyama, S.: Least-squares estimation of transformation parameters between two point patterns. IEEE TPAMI (1991)

37. Wang, C., Xu, H., Jiang, G., Yu, M., Luo, T., Chen, Y.: Underwater monocular depth estimation based on physical-guided transformer. IEEE TGRS 62, 1–16 (2024)

38. Wang, H., Anantrasirichai, N., Zhang, F., Bull, D.: Uw-gs: Distractor-aware 3d gaussian splatting for enhanced underwater scene reconstruction. In: WACV. pp. 3280–3289 (2025)

39. Wang, H., Agapito, L.: 3d reconstruction with spatial memory. In: 3DV. pp. 78–89 (2025)

40. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: CVPR. pp. 5294–5306 (2025)

41. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: CVPR. pp. 20697–20709 (2024)

42. Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: π<sup>3</sup>: Permutation-equivariant visual geometry learning. In: ICLR (2026)

43. Wu, Z., Wang, Y., Wen, Y., Zhang, Z., Wu, B., Tang, H.: Stereoadapter: Adapting stereo depth estimation to underwater scenes. In: ICRA (2026)

44. Xie, Y., Kong, L., Chen, K., Zheng, Z., Yu, X., Yu, Z., Zheng, B.: Uveb: A largescale benchmark and baseline towards real-world underwater video enhancement. In: CVPR. pp. 22358–22367 (2024)

45. Xu, W., Wang, C., Liang, D., Zhao, Z., Jiang, X., Zhang, P., Bai, X.: Nautilus: A large multimodal model for underwater scene understanding. In: NeurIPS (2025)

46. Yang, D., Leonard, J.J., Girdhar, Y.: Seasplat: Representing underwater scenes with 3d gaussian splatting and a physically grounded image formation model. In: ICRA. pp. 7632–7638 (2025)

47. Yang, J., Sax, A., Liang, K.J., Henaf, M., Tang, H., Cao, A., Chai, J., Meier, F., Feiszli, M.: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In: CVPR. pp. 21924–21935 (2025)

48. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. In: NeurIPS. vol. 37, pp. 21875–21911 (2024)

49. Yu, B., Wu, J., Islam, M.J.: Udepth: Fast monocular depth estimation for visuallyguided underwater robots. In: ICRA (2023)

50. Zhang, F., You, S., Li, Y., Fu, Y.: Atlantis: Enabling underwater depth estimation with stable difusion. In: CVPR. pp. 11852–11861 (2024)

51. Zhao, Z., Yang, L., Long, S., Pi, J., Zhou, L., Wang, J.: Augmentation matters: A simple-yet-efective approach to semi-supervised semantic segmentation. In: CVPR. pp. 11350–11359 (2023)

52. Zhou, J., Liang, T., Zhang, D., Liu, S., Wang, J., Wu, E.Q.: Waterhe-nerf: Waterray matching neural radiance fields for underwater scene reconstruction. Information Fusion 115, 102770 (2025)

## Supplementary Material for “Wat3R: Underwater 3D Geometry Learning without Annotations”

In this appendix, we provide additional analyses and implementation details of our method:

– analysis of the cross-view supervision mechanism in Appendix A;

– construction pipeline of the Water3D dataset in Appendix B;

– visualization of the proposed physics-based synthetic underwater rendering process in Appendix C;

– further analysis of Wat3R, including condition-wise results, comparisons with underwater-specific 3DGS and classical SfM pipelines, and additional ablations in Appendix D;

impact of underwater image enhancement on 3D reconstruction in Appendix E; additional qualitative results for monocular depth estimation and multi-view 3D reconstruction in Appendix F.

## A Analysis of Cross-view Supervision

To better understand cross-view supervision, we illustrate its pipeline and the efect of the static mask in dynamic scenes in Fig. A1. We compare diferent supervision sources for the cross-view alignment in Tab. A1 The first row removes the cross-view loss entirely, serving as the baseline. The second row applies selfsupervision by constructing the cross-view constraint using the student predictions themselves. The third row uses the teacher predictions as the supervision target. The results show that the introduction of cross-view supervision improves performance compared to the baseline without loss. However, using student predictions as the target leads to limited improvement due to the instability of the predictions during training. This demonstrates that the EMA teacher produces more reliable geometric targets for enforcing cross-view consistency.

## B Water3D Dataset Details Construction

In this section, we introduce the construction pipeline of Water3D, from raw video streams to reconstructed point clouds. First, images are extracted from the raw videos at 10 frames per second. We use the hloc toolbox [31] to build the COLMAP [32] reconstruction pipeline. To increase the accuracy of hloc, we adopt the $\mathrm { M I N I M A _ { L i g h t G l u e } \ [ 2 4 , 3 0 ] }$ model finetuned with our synthetic underwater data. The resulting correspondences are passed to COLMAP to perform structure-from-motion reconstruction, producing camera intrinsics, extrinsics, and sparse point clouds. For dense reconstruction, we employ COLMAP’s PatchMatch Stereo [5] algorithm. The parameters are adjusted to better handle weak-texture regions in underwater environments. Multi-view geometric consistency is enforced during depth estimation, and unreliable depth predictions are filtered based on photometric and geometric constraints.

![](images/b5ea557cf87d9b2b95008df00044c15354ccb72cf73e370aeb5a373e461584be.jpg)  
Fig. A1: Cross-view geometric supervision with static region filtering. Teacher depth is reprojected from view i to view j to supervise student predictions. A multi-view static mask M <sup>static</sup> is constructed using a consistency threshold k to filter dynamic or inconsistent regions. The bottom row shows an example of a dynamic scene where moving objects are filtered by the static mask.

Due to the presence of suspended particles in underwater scenes, raw depth maps often contain significant outliers. We project the depth maps into 3D space and use a KD-tree for nearest-neighbor analysis to remove outliers with low local density. The filtered points are then re-projected to refine the depth maps.

## C Visualization of Synthetic Underwater Rendering

To better illustrate the behavior of our physics-based underwater rendering process, we visualize several examples generated under diferent parameter settings in Fig. A2. The synthesized images encourage the model to learn the relationship between geometric structure and underwater light degradation.

## D More Understanding of Our Wat3R and Water3D

In this section, we present additional analyses to better understand the behavior of Wat3R. First, we report condition-wise results on Water3D to evaluate our method under diferent underwater environments. Then, we compare our method with classical SfM pipelines to illustrate the dificulty of reliable reconstruction in underwater environments.

Table A1: Efect of Diferent Supervision Sources for the Cross-view Loss. We report Absolute Relative Error (Rel↓) and $\delta _ { 1 }$ ↑ for both multi-view depth and monocular depth estimation on FLSea Stereo [29].

<table><tr><td rowspan="2">Cross-view supervision</td><td colspan="2">Multi-view Depth</td><td colspan="2">Monocular Depth</td></tr><tr><td>Rel↓</td><td> $\delta_1 \uparrow$ </td><td>Rel↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td>None</td><td>0.126</td><td>0.871</td><td>0.127</td><td>0.871</td></tr><tr><td>Student prediction</td><td>0.122</td><td>0.881</td><td>0.123</td><td>0.880</td></tr><tr><td>Teacher prediction</td><td>0.119</td><td>0.885</td><td>0.120</td><td>0.886</td></tr></table>

![](images/9b3327794205dcd4904e5f8a4f71bab8151bec882d63f1f43df96abb3f86fd7e.jpg)  
Fig. A2: Synthetic underwater rendering examples. Starting from an on-land RGB image and depth map, we generate underwater images using our physics-based rendering model. The vertical axis varies the simulated water color, representing different underwater environments, while the horizontal axis shows decreasing visibility caused by increasing attenuation and scattering.

## D.1 Condition-wise Results on Water3D

We also report condition-wise results on Water3D in Tab. A2. The scenes are grouped into shallow sea, deep sea, lake, and river environments according to the dominant capture condition. Wat3R consistently achieves the lowest overall mean and median errors across all groups. River scenes remain the most challenging because they often contain stronger turbidity, less stable texture, and narrower visible range.

## D.2 Comparison with Optimization-based Underwater Pipelines

To address whether the improvement holds beyond feed-forward baselines, we further compare with WaterSplatting [21] and COLMAP-based SfM/MVS pipelines on FLSea Stereo in Tab. A3. For COLMAP, we test SuperPoint+LightGlue (SPLG) and MINIMA matchers. The COLMAP variants can obtain low error on successfully reconstructed scenes, especially with MINIMA, but they fail on most 10-view testing sequences and therefore lack robustness under sparse underwater inputs. WaterSplatting also sufers from inaccurate initialization and sparse-view reconstruction failures. In contrast, Wat3R produces valid predictions for all scenes and achieves the best overall robustness.

Table A2: Condition-wise reconstruction error on Water3D. Best in bold.

<table><tr><td rowspan="2">Methods</td><td colspan="4">Overall Mean / Median Error ↓</td></tr><tr><td>Shallow Sea</td><td>Deep Sea</td><td>Lake</td><td>River</td></tr><tr><td>DA3 (ICLR&#x27;26)</td><td>0.852 / 0.254</td><td>0.924 / 0.167</td><td>0.863 / 0.056</td><td>3.645 / 3.033</td></tr><tr><td>VGGT (CVPR&#x27;25)</td><td>0.857 / 0.226</td><td>1.690 / 0.223</td><td>0.877 / 0.079</td><td>3.650 / 2.440</td></tr><tr><td>Wat3R (ours)</td><td>0.489 / 0.177</td><td>0.283 / 0.101</td><td>0.084 / 0.046</td><td>0.703 / 0.513</td></tr></table>

Table A3: Comparison with underwater-specific and classical reconstruction pipelines on FLSea Stereo. We evaluate multi-view depth under the same 10-view setting as the main paper.

<table><tr><td>Methods</td><td>Rel↓</td><td>δ1 ↑</td><td>RMSE↓</td><td>Failure rate↓</td></tr><tr><td>Wat3R (ours)</td><td>0.119</td><td>0.885</td><td>0.720</td><td>0/152</td></tr><tr><td>WaterSplatting (3DV&#x27;25)</td><td>0.427</td><td>0.415</td><td>1.476</td><td>99/152</td></tr><tr><td>COLMAP+SPLG</td><td>0.129</td><td>0.847</td><td>0.584</td><td>91/152</td></tr><tr><td>COLMAP+MINIMA</td><td>0.105</td><td>0.896</td><td>0.456</td><td>99/152</td></tr></table>

## D.3 Additional Ablations

Tab. A4 further isolates the efect of sequence-level strong augmentation. Applying strong augmentation to synthetic data alone does not improve over syntheticonly supervised training, indicating that the augmentation mainly benefits the semi-supervised real-video branch where viewpoint diversity is limited. To verify that our adaptation strategy is not tied to VGGT, we also apply the same framework to π3 [42]. As shown in Tab. A5, the adapted model improves both Sea-thru and FLSea Stereo, suggesting that the proposed cross-domain semi-supervised training can transfer to other feed-forward geometry backbones.

## D.4 Wat3R v.s. Colmap

We analyze the robustness of diferent reconstruction approaches in challenging underwater environments. During the construction of Water3D, we collected nearly 100 underwater video scenes and attempted to reconstruct them using an improved COLMAP [32] pipeline to obtain ground-truth geometry. However, despite careful tuning and dense multi-view inputs, reliable reconstructions could only be obtained for 42 scenes after manual inspection, which were finally retained in the dataset. Several representative examples are shown in Fig. A3.

Table A4: Additional ablation on synthetic data and strong augmentation on FLSea Stereo. We report multi-view depth metrics to isolate the efect of synthetic underwater data, sequence-level strong augmentation (SA), and unlabeled real videos.

<table><tr><td>Methods</td><td>VGGT</td><td>+syn</td><td>+syn+SA</td><td>+syn+real+SA</td></tr><tr><td>Rel↓</td><td>0.136</td><td>0.131</td><td>0.132</td><td>0.126</td></tr><tr><td>δ1↑</td><td>0.851</td><td>0.868</td><td>0.864</td><td>0.873</td></tr></table>

Table A5: Applying the proposed adaptation framework to π3 [42]. We replace VGGT with π3 as the base feed-forward geometry model and evaluate multi-view depth on Sea-thru and FLSea Stereo.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Sea-thru</td><td colspan="3">FLSea Stereo</td></tr><tr><td>Rel↓</td><td>δ1↑</td><td>RMSE↓</td><td>Rel↓</td><td>δ1↑</td><td>RMSE↓</td></tr><tr><td>π3 (ICLR&#x27;26)</td><td>0.185</td><td>0.909</td><td>0.358</td><td>0.139</td><td>0.856</td><td>0.837</td></tr><tr><td>Wat3R (π3 as base model)</td><td>0.169</td><td>0.933</td><td>0.310</td><td>0.117</td><td>0.899</td><td>0.763</td></tr></table>

The examples highlight the dificulty of applying classical SfM pipelines to underwater imagery. In the upper example of Fig. A3, COLMAP reconstructs only a small portion of the ship hull surface while the remaining structure becomes fragmented with scattered outliers. Moreover, the diver’s oxygen tank visible in the upper-left corner is incorrectly reconstructed behind the ship hull, indicating significant geometric inconsistency. Another example is shown in the lower row. Although the images appear visually clear, the camera motion in the sequence is limited. Under such a small camera baseline, COLMAP produces a degenerate reconstruction where the recovered point cloud collapses into an almost planar structure and fails to recover the true scene geometry.

In contrast, learning-based approaches show significantly stronger robustness in these scenarios. Even when COLMAP fails to produce valid reconstructions, our method can still generate coherent and structurally consistent point clouds that better reflect the underlying scene geometry.

## E Using v.s. Not Using underwater Image enhancement

Many recent underwater vision works [15, 26] focus on restoring underwater images to resemble terrestrial scenes. Such restoration aims to reduce water-induced interference and provide higher-quality inputs for downstream tasks. Therefore, we additionally investigate the role of underwater image enhancement (UIE) in 3D reconstruction. We further evaluate our method using the latest underwater image restoration approaches together with models trained on diferent scenes, as shown in Tab. A6.

![](images/23101a14b6b2022ae5d497d8f9064ca5f42b8acab746389c9fe81bb8578b3001.jpg)  
Fig. A3: Examples of challenging scenes from Water3D. During the construction of the dataset, we use an improved COLMAP [32] pipeline to reconstruct point clouds from raw underwater videos. However, many scenes still lead to unstable or degenerate reconstructions due to scattering efects, low texture, and limited camera motion. For comparison, we also show the results of Wat3R, VGGT [40], and DA3 [22]. While COLMAP fails to produce reliable reconstructions in these cases, our method is able to generate more coherent and geometrically consistent point clouds.

However, this two-stage pipeline does not necessarily benefit 3D reconstruction. Most UIE methods are designed to improve visual clarity rather than geometric consistency, and they are typically trained independently of downstream geometric tasks. As a result, the enhanced images may not provide reliable cues for multi-view geometry estimation. Instead of explicitly removing water efects, we simulate underwater conditions by adding water efects to terrestrial images during training. This strategy allows the model to implicitly learn water removal while understanding underwater geometry.

Table A6: Performance change with underwater image enhancement. We report Absolute Relative Error (Rel↓) for multi-view depth estimation. Applying UIE as a preprocessing step does not lead to clear improvements in geometric estimation.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Sea-thru [3] (Rel↓)</td><td colspan="3">FLSea Stereo [29] (Rel↓)</td></tr><tr><td>VGGT</td><td>MapAnything</td><td>DA3</td><td>VGGT</td><td>MapAnything</td><td>DA3</td></tr><tr><td>Original</td><td>0.190</td><td>0.216</td><td>0.187</td><td>0.137</td><td>0.146</td><td>0.141</td></tr><tr><td>Semi-UIR [15] CVPR&#x27;23</td><td>0.201</td><td>0.218</td><td>0.195</td><td>0.136</td><td>0.148</td><td>0.147</td></tr><tr><td>PSPL [26] TIP&#x27;25</td><td>0.209</td><td>0.230</td><td>0.207</td><td>0.160</td><td>0.151</td><td>0.149</td></tr></table>

## F Additional Visualization

We provide additional geometric prediction results for in-the-wild multi-view inputs in Fig. A4, multi-view inputs on our Water3D in Fig. A5, and monocular inputs in Fig. A6.It can be seen that our model achieves robust and high-quality geometric predictions across several scenarios.

![](images/c5f8bd71012f9e4c200273cefc9b19754c929deb80b49f1a5ddf277005eb05a5.jpg)  
Fig. A4: Qualitative comparison of in-the-wild multi-view 3D reconstruction. We compare Wat3R with DA3 [22], MapAnything [17], and VGGT [40] on challenging underwater scenes. Our method produces more coherent and complete 3D structures, while competing methods sufer from fragmented geometry or missing structures.

![](images/76778f4c1aaa780f394a19e463cfa29278b35410decb626d6e3b62123d58273d.jpg)  
Fig. A5: Qualitative results on diverse underwater scenes from Water3D. The examples illustrate the diversity of underwater environments in our dataset. Compared with VGGT [40], Wat3R produces more complete and coherent reconstructions, even recovering structures that are missing in the COLMAP-based ground truth.

![](images/5cf1f03617195782c5a57a121a9a886129c8c2c8bd2e3f6b0382c65c52b4f93e.jpg)  
Fig. A6: Qualitative results on monocular depth estimation using Wat3R, DA3 [22] and VGGT [40]. Rows 1–2: SQUID [4]; rows 3–4: SeaThru [3] (brightness increased for visualization only); rows 5–6: FLSea Stereo [29]; rows 7–8: FLSea VI [29].