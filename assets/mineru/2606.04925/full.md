# Scene-Centric Unsupervised Video Panoptic Segmentation

Christoph Reich\* 1,2,5,6

Oliver Hahn\* 2,3

Nikita Araslanov 1,5

Laura Leal-Taixé 3

Christian Rupprecht 4

Daniel Cremers† 1,5,6

Stefan Roth† 2,6,7

1TU Munich 2TU Darmstadt 3NVIDIA 4University of Oxford 5MCML 6ELIZA 7hessian.AI \*equal contribution †equal advising

https://visinf.github.io/videocups

Cityscapes-VPS   
![](images/e3cc88399b6d70c5c29e688b2de412f8ea47a3038e1586f00d369b093640cb11.jpg)

![](images/454fd3aae4e8d185d3bc18069dd97ffe020d490d73f3863b15399dcafdec5528.jpg)

![](images/a8b66949a1429e5ee6bbf3c94541e9d30a86a19ea5e5446a1cd954cc57d63455.jpg)

![](images/39d052d7a03ed43da4be23cd2589ff99610f21ecaa1122b28617bde12894ddfb.jpg)

![](images/ab7792843826cb400f16f4003a4af0f7de7759961089787cdf7dcc1318025e0f.jpg)

![](images/d73f7786e39ed07d24a3d1bd7e4fb87241fa72bfde842b0194ed3afb45830c71.jpg)  
t2

t0   
![](images/2d1b118755b395008057ed8072c8bf6e8b5daf644a3c95b26a3ee0030cec03a4.jpg)

![](images/d18f46a7eef8440794fd140ff485c338e60a3506d63bbbec1dfdea9fc4303d82.jpg)

![](images/38e7abbd17d6db3d4743d3cf7a01b06ff5113242d02fa0659d5781115bdb1436.jpg)

![](images/5383dea2d01330437a3662638b1e32b4d29bd67e4f7683f794922bc751e11cdd.jpg)

![](images/eb4632c7f8fb5a78b887aa9db5add6a578b5bc6ea761e30288eecae93f10c687.jpg)

![](images/08cd9b2597d94427e31db6c11ae1337d1bda8e1963ce7871f19422036683fe08.jpg)

![](images/811c7c676eb081a32e85675ecf532f8039351872bac62b6d13409c0b61df86ce.jpg)

![](images/4b2a8b5231a676dd37cceeef8b4d7a339820b4c793d0192f96c7b6c0de45fdeb.jpg)

![](images/aa94d5ea354532b594fb70ec52d9a87feb0f2fd7cc654a69e67986059abc01cf.jpg)

![](images/7eab5336c5002022fe3f791308c2795997b97e823dabd500f310e6f1803d8c12.jpg)  
Figure 1. Results and overview of our unsupervised video panoptic segmentation approach VideoCUPS. Top: Panoptic video predictions by VideoCUPS across four datasets. Bottom: We use self-supervised representations, motion, and depth cues from monocular videos to generate scene-centric panoptic video pseudo-labels and train a video panoptic segmentation model using a novel Video DropLoss.

# Abstract

Video panoptic segmentation (VPS) aims to jointly detect, segment, and track all objects while partitioning the video into semantically consistent regions. We introduce the task setting of unsupervised VPS, omitting any human supervision. Existing unsupervised scene understanding works mainly focused on image segmentation tasks; the video domain remains underexplored. We propose VideoCUPS, the first unsupervised VPS approach. VideoCUPS generates temporally consistent panoptic video pseudo-labels from scene-centric videos by exploiting unsupervised depth, motion, and visual cues. Training on these pseudo-labels using a novel Video DropLoss yields an accurate, unsupervised VPS model. To benchmark progress, we introduce a comprehensive evaluation protocol and four competitive baselines, extending state-of-the-art unsupervised panoptic image and instance video segmentation models to VPS. VideoCUPS outperforms all baselines and demonstrates strong label-efficient learning. With VideoCUPS, our evaluation protocol, and baselines, we provide a strong foundation for future research on unsupervised VPS.

# 1. Introduction

Video Panoptic Segmentation (VPS) [57, 114] is a holistic scene understanding task that extends panoptic segmentation [59] from the spatial to the spatio-temporal domain, unifying instance and semantic segmentation over time. Specifically, VPS aims to detect, segment, classify, and temporally associate individual object instances, while also assigning each pixel a semantic category. This comprehensive segmentation task enables parsing of complex, dynamic real-world environments and has a wide range of applications, such as autonomous driving, robotics, video editing, and medical imaging [see 119, 126, for an overview].

Advances in panoptic video understanding have been driven by supervised learning, relying on significant amounts of human-annotated data for training [19, 29, 57, 69, 83, 90, 114]. However, acquiring dense pixel-level instance and semantic annotations for images is highly resource intensive [23]. Extending labeling efforts to the temporal domain poses even more challenges, including limited scalability and label quality [21, 114, 117]. Despite the success of densely annotated large-scale datasets, such as SA-1B [60], there is a natural interest in more efficient and scalable annotation-free alternatives [41, 48, 93, 111].

Unsupervised learning has emerged as a powerful paradigm, showcasing significant progress in scene understanding tasks such as unsupervised semantic [40, 42, 91], instance [92, 109, 111], and panoptic segmentation [41, 81]. Among these, U2Seg [81] established the first approach for unsupervised image panoptic segmentation by combining semantic pseudo-labels from STEGO [42] and instance masks from CutLER’s MaskCut component [109]. CUPS [41] extended the paradigm to scene-centric1 imagery by using motion and depth cues from real-world stereo video to generate panoptic pseudo-labels for images, overcoming the need for object-centric imagery required for training U2Seg. These advances have focused on unsupervised image segmentation. In contrast, the unsupervised panoptic segmentation of videos remains underexplored, while offering broader applications for dynamic real-world environments and a more natural human perception of the world by also perceiving the temporal domain. We therefore introduce the task setting of unsupervised video panoptic segmentation, which aims to panoptically segment videos without any human supervision.

To approach unsupervised VPS for the first time, we introduce VideoCUPS: scene-Centric Unsupervised Video Panoptic Segmentation. While building on CUPS [41], which addresses unsupervised image panoptic segmentation, VideoCUPS directly produces temporally consistent video panoptic pseudo-labels. Additionally, our VPS pseudo-labeling method operates solely on monocular videos as input, thereby overcoming the need for stereo video during VPS training. To enable effective training on our pseudo-labels, we introduce a Video DropLoss and selfenhanced video copy-paste augmentation. For assessing the accuracy of VideoCUPS, we establish four competitive baselines, built using CUPS [41], U2Seg [81], and Video-CutLER [110]. VideoCUPS, together with the proposed baselines and evaluation protocol, forms a foundation for future work on unsupervised panoptic video understanding.

Specifically, we make the following contributions: (i) We introduce the task setting of unsupervised video panoptic segmentation and propose a unified evaluation protocol spanning four established VPS datasets. To enable comparison, we extend the Segmentation and Tracking Quality (STQ) to the unsupervised setting by incorporating pseudo-semantic matching. Moreover, we construct four competitive VPS baselines that combine state-ofthe-art unsupervised semantic, video instance, and panoptic image segmentation models with unsupervised tracking. (ii) We generate high-quality video panoptic pseudolabels solely from monocular scene-centric videos using self-supervised visual, depth, and motion cues. Using a novel Video DropLoss and self-enhanced video copy-paste augmentation, we train on our pseudo-labels, leading to the first unsupervised VPS approach. (iii) VideoCUPS consistently outperforms all unsupervised baselines across a wide range of scene-centric video datasets. Additionally, we show that VideoCUPS provides a strong foundation for approaching VPS using label-efficient learning.

# 2. Related Work

Unsupervised segmentation methods have been shaped by advances in self-supervised learning (SSL) and unsupervised low-level vision, particularly in motion and depth estimation. We first review these developments before discussing unsupervised segmentation approaches.

Self-supervised representation learning aims to learn expressive and transferable visual representations from unlabeled data [30]. A variety of pretext tasks have been proposed to achieve this [3, 30], enabling feature extractors that generalize across downstream tasks [82, 93]. The advent of Vision Transformers (ViTs) [26] has further shaped SSL by facilitating large-scale training and enabling novel pretext designs [48, 124]. Contemporary methods typically optimize ViTs through contrastive learning [6, 17, 18, 47], negative-free objectives [7, 13, 16, 37], clustering [5, 11, 12, 105], masked modeling [39, 48, 80], or a combination of these [82, 93, 124]. Recent SSL frameworks, such as the DINO family [13, 82, 93], provide semantically rich, dense features, suited for unsupervised segmentation [42, 109].

Unsupervised optical flow aims to estimate apparent motion directly from video without ground truth [2, 123]. While classical formulations were inherently unsupervised [8, 49, 74], early deep learning approaches relied on synthetic datasets to provide supervision [25, 76, 97]. Inspired by traditional formulations and motivated by the synthetic-to-real gap, deep learning-based unsupervised optical flow has been introduced [2, 54, 78, 84, 123]. Recent unsupervised deep optical flow methods provide accurate motion estimation, efficient inference, and strong generalization across diverse real-world domains [70, 75, 96].

Unsupervised monocular depth estimation aims to estimate depth of monocular imagery by learning from stereo images or monocular videos [32, 34, 125]. Learning depth from monocular videos is done by novel-view synthesis and photometric consistency [106, 125]. Novel-view synthesis, however, assumes a static scene and breaks for dynamics [35, 99, 120]. Recent approaches use auto-masking [35], semantic/instance cues [14, 15, 31, 38, 61, 67, 116], or multi-view [31, 113, 118] to compensate for dynamic objects. Other methods, such as DynamoDepth [99], jointly learn depth, motion, and/or motion segmentation, decomposing the scene into static and dynamic parts [50, 68].

Unsupervised instance segmentation aims to detect and segment objects in images without human supervision [94]. Recent approaches [100, 104, 108, 109, 111] train classagnostic detectors using pseudo-labels derived from SSL features of object-centric imagery. TokenCut [112] obtains foreground masks from DINO features using normalized cuts [89]. CutLER [109] extends this by iteratively cutting multiple pseudo-masks per image, and is further improved by [4, 92, 111]. A complementary direction exploits motion cues for object discovery [22, 36, 55, 71, 86, 95, 100, 122]. Recently, unsupervised extensions to video have emerged. VideoCutLER [110] trains on synthetic videos from image pseudo-masks, FlowCut [87] enforces motion-based temporal consistency, and AutoQ-VIS [73] improves pseudolabels via automatic quality assessment.

Unsupervised semantic segmentation aims to divide images into semantically meaningful regions without any human annotations. Early deep learning approaches [20, 44, 53] used representation learning, encouraging embeddings to capture dense semantic similarity. Leveraging self-supervised DINO [13] features as an inductive bias, STEGO [42] distills and clusters features to obtain unsupervised semantic segmentations. Building on the STEGO framework, subsequent methods [40, 52, 56, 88, 91] refine the distillation and probing process. Other unsupervised segmentation approaches [24, 79, 103] alternatively use vision-language diffusion features. To the best of our knowledge, there are no extensions of unsupervised semantic segmentation methods to video to date.

Unsupervised panoptic segmentation has recently emerged as a natural next step, following advances in unsupervised semantic and instance segmentation. While panoptic segmentation of images and videos has been extensively studied in the supervised setting [see 29, 126, for an overview], we are only aware of two unsupervised image panoptic segmentation approaches, U2Seg [81] and CUPS [41]. U2Seg combines CutLER’s MaskCut [109] and STEGO [42] to create pseudo-labels for panoptic training, but inherits MaskCut’s object-centric bias, significantly limiting accuracy on scene-centric data [41]. CUPS overcomes this by grouping unsupervised scene flow from stereo into rigid instances [95] and combining these with unsupervised semantics [91] to train a panoptic network. In our work, we employ both U2Seg and CUPS in competitive baselines and propose the first approach to directly perform unsupervised video panoptic segmentation. While we, similar to CUPS [41], use self-supervised representations, motion, and depth cues, VideoCUPS requires only monocular video for VPS pseudo-labeling, captures non-rigid instance motions, and directly generates panoptic video pseudo-labels.

# 3. Method: Unsupervised VPS

First, we generate panoptic video pseudo-labels (cf . Sec. 3.1 and Fig. 2) from monocular videos. Second, we train a VPS model (cf . Sec. 3.2) using these pseudo-labels, a novel Video DropLoss, and self-enhanced video copypaste augmentations, leading to the first unsupervised VPS model. Third, to enable evaluation of VideoCUPS and future approaches, we present an evaluation protocol for the unsupervised VPS setting (cf . Sec. 3.3).

# 3.1. Generating VPS pseudo-labels

To generate temporally coherent panoptic video pseudolabels, we adopt a bottom-up strategy (cf . Fig. 2). Initially, we produce semantic and instance pseudo-labels for individual frames, which are then refined through temporal consistency processing along the video sequence.

From motion and depth to instance pseudo-labels. Drawing inspiration from Gestalt principles [62, 63, 115], we adopt the common fate, proximity, and similarity principle—neighborhoods that move together belong together—to derive class-agnostic instance pseudo-masks from monocular videos. Accordingly, we defined objects as entities capable of moving. We obtain per-frame instance pseudo-labels across an entire video clip as follows. Given two consecutive monocular frames, we obtain unsupervised optical flow $\mathbf { f } \in \mathbb { R } ^ { 2 \times \mathrm { H } \times \mathrm { W } }$ using SMURF [96]. Monocular depth d $\in \mathbb { R } ^ { \mathrm { H \times W } }$ is estimated by DynamoDepth [99]. Alongside depth, DynamoDepth also estimates dense motion probabilities m $\in \ [ 0 , 1 ] ^ { \mathrm { { \dot { H } } \times W } }$ , decomposing the scene into static $( m _ { h , w }  0 )$ and dynamic regions $( m _ { h , w }  1 )$ .

We employ a variant of region growing [1, 43] to extract a variable number of instance pseudo-masks. Specifically, we threshold m at $\alpha = 0 . 1 5$ to obtain instance seeds. Next, we iteratively merge pixels within a Chebyshev neighborhood r based on their relative depth and flow difference. In particular, for pixel $\textbf { x } = \mathbf { \Psi } ( h , w )$ with $m _ { \mathbf { x } } \ > \ \alpha _ { : }$ we merge pixels within the Chebyshev neighborhood $\mathcal { N } _ { r } ( \mathbf { x } ) =$ $\{ \mathbf { y } \mid | \mathbf { y } - \mathbf { x } | | _ { \infty } \leq r \wedge m _ { \mathbf { y } } > \alpha , \mathbf { y } \neq \mathbf { x } \}$ to x if

$$
\frac {\left| d _ {\mathbf {x}} - d _ {\mathbf {y}} \right|}{\left| d _ {\mathbf {x}} \right|} <   \tau_ {d} \quad \text { and } \quad \frac {\left| \left| f _ {\mathbf {x}} - f _ {\mathbf {y}} \right| \right| _ {2}}{\left| \left| f _ {\mathbf {x}} \right| \right| _ {2}} <   \tau_ {f}, \tag {1}
$$

with $\mathbf { y } \in \mathcal { N } _ { r } ( \mathbf { x } )$ . Merging proceeds iteratively until convergence and can be parallelized for efficiency. The resulting set of l class-agnostic pseudo-instance masks M ∈ $\{ 0 , 1 \} ^ { l \times \mathrm { H } \times \mathrm { W } }$ groups pixels that share consistent relative depth and motion. Unlike the rigid-motion pseudo-labeling in CUPS [41], we do not assume rigidity but exploit smoothness, enabling us to also capture non-rigidly moving instances, such as pedestrians in motion (cf . Fig. 5).

From SSL features to semantic pseudo-labels. We derive an unsupervised semantic segmentation model S by distilling DINO [13] features into a lower-dimensional embedding via a contrastive objective, leveraging monocular depth as an auxiliary cue. Clustering with stochastic cosinedistance k-means yields $\mathcal { S } : \mathbb { R } ^ { 3 \times \tilde { \mathrm { H } } \times \mathrm { W } } \to \{ 0 , 1 \} ^ { \mathrm { c } _ { \mathrm { p } } \times \mathrm { H } \times \mathrm { W } }$ , mapping an input image I to dense semantic pseudo-labels with $\mathrm { c _ { p } }$ semantic pseudo-classes, consistent across the entire dataset. While unsupervised semantic segmentation approaches typically operate at low resolutions (e.g., 3202), close to that used for SSL pre-training [13], we use depthguided semantic inference [41] to obtain high-resolution semantic predictions. Specifically, we infer a semantic prediction $\mathbf { P } ^ { \mathrm { l o w } }$ at lower resolution and $\mathbf { P } ^ { \mathrm { h i g h } }$ at a higher resolution using sliding-window inference. $\mathbf { P } ^ { \mathrm { l o w } }$ captures coarse, nearfield semantics and $\mathbf { P } ^ { \mathrm { h i g h } }$ preserves fine details via slidingwindow inference and soft aggregation. $\mathbf { P } ^ { \mathrm { l o w } }$ is upsampled to the resolution of $\mathbf { P } ^ { \mathrm { h i g h } }$ and both are fused with a per-pixel depth weight $\alpha _ { h , w } = ( d _ { h , w } + 1 ) ^ { - 1 }$ using the monocular depth prediction d from DynamoDepth:

![](images/092c852d6080567ad8ec13c73d334cfafde227d19b0d2e0a18255deb6635303b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Instance Pseudo-Labeling"] --> B["Flow Network"]
    A --> C["Depth Network"]
    B --> D["Optical Flow"]
    C --> E["Motion Probability"]
    D --> F["Region Growing"]
    E --> F
    F --> G["Propagation & Tracking"]
    G --> H["Align"]
    H --> I["Temporal Smoothing"]
    I --> J["Temporal Inference"]
    J --> K["Depth-guided Semantic Inference"]
    K --> L["Semantic Prediction"]
    L --> M["Semantic Network"]
    M --> N["Input Video"]
    style A fill:#f9f,stroke:#333
    style J fill:#ccf,stroke:#333
```
</details>

Figure 2. VideoCUPS pseudo-label generation. Instance pseudo-labeling applies motion-based region growing segmentation using unsupervised optical flow from SMURF [96] and depth from DynamoDepth [99]. Semantic pseudo-labeling uses a k-means clustering prediction of distilled DINO features [13], combined with a depth-guided inference [41]. Temporal tracking propagates and tracks the instance pseudo-labels, temporally smoothes the semantic pseudo-labels, and aligns the two signals into video panoptic pseudo-labels.

$$
\mathbf {P} ^ {*} = \boldsymbol {\alpha} \odot \mathbf {P} ^ {\text { low }} + (1 - \boldsymbol {\alpha}) \odot \mathbf {P} ^ {\text { high }}. \tag {2}
$$

We further apply regularized Frank-Wolfe inference [66] for dense CRFs [64], enabling fast spatial regularization. Building on [41], we adapt DepthG [91] retrained with the monocular depth from DynamoDepth [99], ensuring consistency with our unsupervised, monocular setting.

From image to video pseudo-labels. A key component of our pseudo-label generation is the temporal processing and fusion of frame-wise semantic and instance information.

Instance propagation and tracking extends the framewise, class-agnostic instance pseudo-labels to the video domain using optical-flow-based mask propagation and IoUbased association. Given three consecutive frames $\mathbf { I } _ { t - 1 }$ , $\mathbf { I } _ { t } ,$ and $\mathbf { I } _ { t + 1 }$ , we estimate the forward and backward optical flows f fwt−1,t, f bwt−1,t $\mathbf { f } _ { t - 1 , t } ^ { \mathrm { f w } } , \mathbf { f } _ { t - 1 , i } ^ { \mathrm { b w } }$ and $\mathbf { f } _ { t , t + 1 } ^ { \mathrm { f w } } , \mathbf { f } _ { t , t + 1 } ^ { \mathrm { b w } }$ , using SMURF. We then perform backward warping of the instance pseudolabels $\mathbf { M } _ { t }$ and $\mathbf { M } _ { t + 1 }$ to their respective previous frames, resulting in $\hat { \mathbf { M } } _ { t  t - 1 }$ and $\hat { \mathbf { M } } _ { t + 1  t }$ while ignoring occluded pixels identified via forward–backward flow consistency [101]. To match the instance IDs between frame t − 1 and t, we compute an IoU cost matrix between all instance masks in $\mathbf { M } _ { t - 1 }$ and $\hat { \mathbf { M } } _ { t  t - 1 }$ . Hungarian matching is applied to all pairs with IoU $> \tau _ { m } = 0 . 4$ , and the resulting associations are used to update the instance IDs in $\mathbf { M } _ { t }$ . For instances in $\mathbf { M } _ { t - 1 }$ without a match, we attempt recovery using $\dot { \mathbf { M } } _ { t + 1 \neq t }$ under the same threshold $\tau _ { m }$ , thereby resurrecting instances lost in frame t by warping back masks from $t + 1$ . The remaining masks in $\mathbf { M } _ { t + 1  t }$ are assigned new IDs, and the temporal window advances by one frame until the end of the video clip. Finally, we filter out short-lived instances that appear in less than 2 frames of the clip.

Temporal semantic smoothing enforces temporal consistency of semantic pseudo-labels by aggregating neighboring predictions. For each frame t, we obtain $\hat { \mathbf { P } } _ { t - 1  t }$ and $\hat { \mathbf { P } } _ { t + 1  t }$ , the warped pseudo-labels from adjacent frames using flow. The temporally smoothed label P˜ t is obtained via pixel-wise majority vote over $\{ \hat { \mathbf { P } } _ { t - 1  t } , \mathbf { P } _ { t } ^ { * } , \hat { \mathbf { P } } _ { t + 1  t } \}$ using a three-frame sliding window.

Aligning semantic and instance pseudo-labels per video clip results in the final video panoptic pseudo-labels. We align the semantic and instance signals by assigning a consistent semantic pseudo-class to all masks of an instance ID across an entire clip, determined by a majority vote over all semantic pseudo-labels within the instance masks.

Once all video panoptic pseudo-labels are obtained, we aim to retrieve the split of the semantic pseudo-classes into pseudo “thing” and “stuff” classes. We aggregate pixel distributions across all clips by computing the ratio of each semantic pseudo-class frequency within the instance masks relative to its overall frequency. We designate semantic pseudo-classes with a high ratio above a threshold $\psi ^ { \mathrm { t s } }$ as “thing”, and those below as “stuff”.

# 3.2. Learning from VPS pseudo-labels

Using our panoptic video pseudo-labels, we aim to train a model to perform unsupervised VPS. In particular, given an input video of T frames, the model predicts a panoptic video segmentation $\mathbf { P } = ( \mathbf { S } , \mathbf { R } )$ , composed of the predicted pseudo-classes $\mathbf { S } \in \{ 1 , 2 , \ldots , c _ { \mathsf { p } } \} ^ { \mathsf { T } \times \mathrm { H } \times \mathrm { W } }$ and np $n _ { p }$ binary video instance masks $\mathbf { R } \in \mathsf { \{ 0 , \dot { 1 } \} } ^ { n _ { p } \times \mathrm { T } \times \mathrm { H } \times \mathrm { W } }$ for “thing” object instances. Since our pseudo-labels capture only moving “thing” instances (e.g., moving cars), we train the VPS model sparsely to generalize to static objects $( e . g .$ , parked cars). We introduce a Video DropLoss, extending the DropLoss [109] to video, and a self-enhanced video copy-paste augmentation to improve small-object detection.

Video DropLoss. Given two consecutive video frames of our pseudo-labeled clips, we infer “thing” video instance detections $\mathbf { D } _ { j }$ (masks & semantic class) with their tracking latent representation $\mathbf { E } _ { j }$ from our model. Given a sparse set of pseudo “thing” video instance labels $\hat { \bf D } _ { i }$ (masks & pseudo-class, derived from M & P˜ ) and their track $\mathrm { i } \mathrm { \hat { d } } _ { i } ,$ we supervise “thing” detections with our Video DropLoss:

$$
\mathcal {L} _ {\mathrm{VDrop}} = \mathbb {1} \left(\operatorname{IoU} _ {j} ^ {\max} > \tau_ {\mathrm{IoU}}\right) \mathcal {L} _ {\mathrm{d}} \left(\mathbf {D} _ {j}, \hat {\mathbf {D}} _ {i}\right) \mathcal {L} _ {\mathrm{t}} \left(\mathbf {E} _ {j}, \hat {\mathrm{id}} _ {i}\right), \tag {3}
$$

where $\mathcal { L } _ { \mathrm { t } }$ denotes the tracking loss [121] and $\mathcal { L } _ { \mathrm { d } }$ the “thing” detection loss [59]. This Video DropLoss pseudosupervises only “thing” instance predictions $\mathbf { D } _ { j }$ and their tracking representation $\mathbf { E } _ { j }$ that sufficiently overlap with a pseudo-instance $\hat { \bf D } _ { i } \ ( i . e . \ \mathrm { I o U } _ { j } ^ { \mathrm { m a x } } \ > \ \tau _ { \mathrm { I o U } } )$ . Our Video DropLoss enables learning from our sparse pseudo-labels while providing the freedom to predict objects and their tracks that are not covered by our pseudo-labels $( e . g .$ , static objects). Semantics of “stuff” regions are supervised using a standard cross-entropy loss.

Self-enhanced video copy-paste augmentation. To improve the “thing” detection and tracking accuracy of the VPS model on small objects, we introduce a self-enhanced video copy-paste augmentation. Copy-pasting instance masks [27, 28, 33] has been shown to be particularly effective when training with sparse pseudo-labels [41, 92, 109]. Instead of copy-pasting instance masks derived from the pseudo-labels onto another image for augmentation, CUPS [41] has shown that it is beneficial to derive the instance mask from the model’s prediction itself. The intuition behind this is that the network gradually discovers more “thing” objects than captured by the pseudo-labels. We extend this idea to the video domain. In particular, given a training batch, we perform inference and extract confident “thing” video instances from the model’s VPS prediction. We apply random scaling and horizontal flipping to the video instance masks and paste the augmented masks into clips of the training batch. We paste masks using random trajectories, ensuring diverse motion patterns. Finally, we train our model on the batch of augmented clips.

# 3.3. Unsupervised VPS evaluation protocol

In the absence of supervision, our predicted semantic pseudo-classes do not align with the ground-truth semantic class IDs [20, 40–42, 52, 56, 88, 91]. Therefore, a mapping between pseudo and ground-truth categories is required before using standard evaluation metrics. We present a simple, hyperparameter-free matching strategy for aligning the pseudo-classes while strictly preserving the separation between “thing” and “stuff” categories.

Specifically, given a video of length T, we obtain an unsupervised VPS prediction $\mathbf { P } = ( \mathbf { S } , \mathbf { R } )$ . Only for evaluation, we have given the ground-truth VPS label $\bar { \bf P } = ( \bar { \bf S } , \bar { \bf R } )$ , with the semantic ground truth $\bar { \bf S } \in \{ 1 , 2 , \ldots , { c _ { \mathrm { g t } } } \}$ T×H×W and the corresponding $n _ { \mathrm { g t } }$ binary video instance masks $\bar { \mathbf { R } } \in$ {0, 1}ngt×T×H×W. $\{ 0 , 1 \} ^ { n _ { \mathrm { g t } } \times \mathrm { T } \times \mathrm { \bar { H } } \times \mathrm { W } }$

Panoptic segmentation [57, 59, 114] distinguishes between “thing” categories for which instance masks are predicted and “stuff” categories for which only semantics are predicted. To adhere to this strict separation between both, we extract the set of semantic pseudo “thing” categories $\mathbb { S } _ { \mathtt { p } } ^ { \mathtt { T h } } \subset \{ 1 , \dotsc , c _ { \mathtt { p } } \}$ (i.e., categories with video instance predictions) and semantic pseudo “stuff” categories $\mathbb { S } _ { \mathfrak { p } } ^ { \mathrm { S t } } ~ \subset$ $\{ 1 , \ldots , c _ { \mathfrak { p } } \}$ , with $\mathbb { S } _ { \mathfrak { p } } ^ { \mathrm { T h } } \cap \mathbb { S } _ { \mathfrak { p } } ^ { \mathrm { S t } } = \emptyset \mathrm { a n d } \mathbb { S } _ { \mathfrak { p } } ^ { \mathrm { T h } } \cup \mathbb { S } _ { \mathfrak { p } } ^ { \mathrm { S t } } = \{ 1 , \dots , \overset { \cdot } { c _ { \mathfrak { p } } } \}$ . Similarly, we know the ground-truth semantic “thing” categories $\dot { \mathbb { S } } _ { \mathrm { g t } } ^ { \mathrm { T h } } \subset \{ 1 , \dots , { c _ { \mathrm { g t } } } \}$ and semantic “stuff” categories $\mathbb { S } _ { \mathrm { g t } } ^ { \mathrm { S t } } \subset \{ 1 , \ldots , c _ { \mathrm { g t } } \}$ , with $\mathbb { S } _ { \mathrm { g t } } ^ { \mathrm { T h } } \cap \mathbb { S } _ { \mathrm { g t } } ^ { \mathrm { S t } } = \emptyset$ and $\mathbb { S } _ { \mathrm { g t } } ^ { \mathrm { T h } } \cup \mathbb { S } _ { \mathrm { g t } } ^ { \mathrm { S t } } =$ $\{ \breve { 1 } , \ldots , c _ { \mathrm { g t } } \}$ . For each category type, we construct a cost matrix $\mathbf { A } ^ { \mathrm { T h } } \in \mathbb { N } ^ { | \mathbb { S } _ { \mathrm { p } } ^ { \mathrm { T h } } | \times | \mathbb { S } _ { \mathrm { g t } } ^ { \mathrm { T h } } | }$ and $\mathbf { A } ^ { \mathrm { S t } } \in \mathbb { N } ^ { | \mathbb { S } _ { \mathrm { p } } ^ { \mathrm { S t } } | \times | \mathbb { S } _ { \mathrm { g t } } ^ { \mathrm { S t } } | }$ | that accumulates the number of overlapping pixels between every pseudo and ground-truth class across all videos in the validation set. We independently apply Hungarian matching [65] to both matrices, maximizing pixel overlap, and establish an initial correspondence by matching one groundtruth class with a pseudo-class. If there exist more pseudo than ground-truth classes, unmatched pseudo-classes are assigned to the ground-truth class with the highest overlap.

After alignment, we follow the established protocol by Weber et al. [114] from the supervised literature. In particular, we compute the Segmentation and Tracking Quality (STQ), composed of the Association Quality (AQ) and Segmentation Quality (SQ). STQ measures accuracy on full videos at the pixel level, requires no threshold-based matching for validating video instance detections, and considers both precision and recall, different from other VPS metrics [51, 57, 114]. More details and a discussion on other VPS metrics are provided in the supplement.

# 4. Experiments

We evaluate the unsupervised VPS accuracy of VideoCUPS within its training domain and its generalization (Sec. 4.1). To assess VideoCUPS’s accuracy, we also report four baselines. Next, we provide label-efficient learning results (Sec. 4.2). Finally, we analyze the impact of our core components (Sec. 4.3). Additional results are in the supplement.

Datasets. We train VideoCUPS on video pseudo-labels generated from the Cityscapes training sequences (2 975 clips of 30 frames each) and evaluate it on the Cityscapes-VPS val set [57]. To assess generalization, we conduct cross-domain evaluations on KITTI-STEP [114] and Waymo [77, 98], and further test out-of-domain (OOD) generalization on MOTS [107]. While Cityscapes-VPS, KITTI-STEP, and Waymo focus on understanding driving scenes, MOTS addresses human-centric segmentation and tracking in indoor and outdoor settings. For all crossdomain datasets, we ensure compatibility of their label spaces with the Cityscapes category definitions through matching (cf . Sec. 3.3). Note, we ignore extremely small instances in Waymo; more details are in the supplement.

Table 1. Unsupervised VPS on Cityscapes-VPS val. We compare VideoCUPS to our unsupervised VPS baselines, using STQ, AQ, and SQ (all in %, ↑). VideoCUPS achieves state-of-the-art accuracy on Cityscapes-VPS val. † denotes CUPS retrained using monocular videos. 

<table><tr><td>Method</td><td>Training data</td><td>Pseudo-classes</td><td>STQ</td><td>AQ</td><td>SQ</td></tr><tr><td>Supervised [58, 121]</td><td>Cityscapes &amp; Cityscapes-VPS</td><td>-</td><td>42.0</td><td>27.0</td><td>65.3</td></tr><tr><td>DepthG [91] + VideoCutLER [110]</td><td>Cityscapes &amp; ImageNet</td><td>27</td><td>9.9</td><td>3.4</td><td>28.2</td></tr><tr><td>U2Seg [81] + SORT [9]</td><td>COCO &amp; ImageNet</td><td>800 + 27</td><td>11.4</td><td>5.6</td><td>23.0</td></tr><tr><td>CUPS [41] + SORT [9]</td><td>Cityscapes (stereo videos)</td><td>27</td><td>20.6</td><td>13.3</td><td>31.8</td></tr><tr><td>CUPS† [41] + SORT [9]</td><td>Cityscapes (monocular videos)</td><td>27</td><td>17.8</td><td>10.6</td><td>29.9</td></tr><tr><td>VideoCUPS (Ours)</td><td>Cityscapes (monocular videos)</td><td>27</td><td>22.2</td><td>15.3</td><td>32.3</td></tr></table>

![](images/e753a204721f4656ebe257e69cb3bbedf120be44b4df85a7ccc148461ca47715.jpg)

<details>
<summary>text_image</summary>

Road
Sidewalk
Building
Wall
Fence
Pole
T. Light
T. Sign
Veget.
Terrain
Sky
Person
Rider
Car
Truck
Bus
Train
M.cycle
Bicycle
Ground-truth
DepthG+
VideoCutLER
U2Seg + SORT
CUPS + SORT
CUPS† + SORT
t0
t1
t2
t0
t1
t2
</details>

Figure 3. Qualitative unsupervised VPS examples. We compare VideoCUPS to our baselines DepthG [91] + VideoCutLER [110], U2Seg [81] + SORT, CUPS [41] + SORT [9], and CUPS† [41] + SORT [9] on Cityscapes-VPS val. We highlight regions of improvement.

Evaluation details. We follow the evaluation protocol outlined in Sec. 3.3 and report the Segmentation and Tracking Quality (STQ) [114] alongside the Association Quality (AQ) and Segmentation Quality (SQ), all in %.

Implementation details. We generate video pseudo-labels using $c _ { \mathrm { p } } = 2 7$ pseudo-classes, following CUPS [41]. To adhere to our purely unsupervised and monocular setup, we retrain DynamoDepth [99] with a DINO ResNet-18 [13, 46], instead of an ImageNet-supervised ResNet-18, and also retrain DepthG with monocular depth from DynamoDepth. Our region growing uses $\tau _ { d } ~ = ~ 0 . 0 2 , ~ \tau _ { f } ~ = ~ 0 . 0 4$ , and $r \ = \ 8 .$ To ensure fairness to our baselines U2Seg [81] and CUPS [41], which employ a Panoptic Cascade Mask R-CNN [10, 58], we use the closest video extension Panoptic Cascade MaskTrack R-CNN [10, 58, 121] with a DINO ResNet-50 [13, 46, 109]. We train using AdamW [72], our self-enhanced video copy-paste augmentation, and Video DropLoss (with $\tau _ { \mathrm { I o U } } ~ = ~ 0 . 5 )$ for eight epochs. We refer to the supplement for further details.

Unsupervised VPS baselines. As there are no existing unsupervised VPS approaches, we construct four competitive baselines. DepthG + VideoCutLER combines the unsupervised semantic segmentation approach DepthG [91] with the class-agnostic video instance segmentation method VideoCutLER [110]. We adopt the “thing”/“stuff” separation and the semantic-instance fusion scheme from our pseudo-labeling. Since running VideoCutLER on long videos leads to memory exhaustion, we split them into clips of 30 frames with a 5-frame temporal overlap. Instance IDs are aligned across clips using IoU overlap. U2Seg [81] + SORT and CUPS $[ 4 l J + S O R T$ combine existing state-of-the-art approaches to unsupervised panoptic image segmentation with SORT [9], a well-established unsupervised multi-object tracker. SORT assigns temporally consistent IDs to the “thing” detections of the respective model. We use the proposed hyperparameters by Bewley et al. [9]. As CUPS utilizes stereo video for training, we also provide a monocular variant of CUPS using monocular depth from DynamoDepth to assess the impact of using stereo cues. This variant is denoted as $C U P S ^ { \dagger } + S O R T$ .

Supervised upper bound. To contextualize our unsupervised results, we train a supervised equivalent of VideoCUPS. Following the protocol in supervised VPS [57, 114], we initialize with a pre-trained backbone (DINO [13]) and pre-train on Cityscapes [23] panoptic image annotations. Next, we fine-tune for VPS on Cityscapes-VPS [57].

# 4.1. Unsupervised VPS results

In-domain results. In Tab. 1, we compare VideoCUPS against our proposed baselines DepthG + VideoCutLER, $\mathrm { U 2 S e g ~ + ~ \ S O R T } .$ , and CUPS + SORT (w/ and w/o stereo) on the Cityscapes-VPS validation set. VideoCUPS significantly outperforms DepthG + VideoCutLER and U2Seg + SORT, increasing STQ by 12.3 % and 10.8 % points, respectively. We attribute the lower STQ of both baselines to their training on object-centric data. Instead, VideoCUPS can train directly on scene-centric videos. CUPS + SORT requires stereo video for video training, limiting applicability. While being monocular, VideoCUPS reaches an STQ of 22.2 %, outperforming the stereo CUPS + SORT baselines (20.6 % STQ). In comparison to the monocular variant, CUPS† + SORT, VideoCUPS leads to an improved STQ of 4.4 %. This demonstrates that our pseudo-labeling approach leverages monocular and scene-centric video more effectively, while CUPS requires stereo cues to achieve competitive results. These findings are also reflected in the qualitative comparison in Fig. 3.

Table 2. Generalization results. Video panoptic segmentation results, comparing VideoCUPS to our unsupervised VPS baselines, using STQ, AQ, and SQ (all in %, ↑). We evaluate generalization to the Waymo and KITTI-STEP datasets as well as to the OOD dataset MOTS. VideoCUPS consistently outperforms all of the proposed baselines. † denotes CUPS retrained using monocular videos. 

<table><tr><td rowspan="2">Method</td><td colspan="3">KITTI-STEP</td><td colspan="3">Waymo</td><td colspan="3">MOTS (OOD)</td></tr><tr><td>STQ</td><td>AQ</td><td>SQ</td><td>STQ</td><td>AQ</td><td>SQ</td><td>STQ</td><td>AQ</td><td>SQ</td></tr><tr><td>Supervised [58, 121]</td><td>53.9</td><td>59.9</td><td>48.4</td><td>22.3</td><td>12.6</td><td>39.4</td><td>20.5</td><td>12.7</td><td>33.1</td></tr><tr><td>DepthG [91] + VideoCutLER [110]</td><td>13.2</td><td>8.7</td><td>20.1</td><td>7.9</td><td>2.6</td><td>23.9</td><td>14.5</td><td>6.8</td><td>30.7</td></tr><tr><td>U2Seg [81] + SORT [9]</td><td>24.0</td><td>21.1</td><td>27.2</td><td>10.4</td><td>4.8</td><td>22.6</td><td>14.9</td><td>7.2</td><td>30.8</td></tr><tr><td>CUPS [41] + SORT [9]</td><td>34.2</td><td>37.7</td><td>31.1</td><td>17.5</td><td>9.9</td><td>30.8</td><td>16.7</td><td>10.4</td><td>27.0</td></tr><tr><td> $CUPS^†$  [41] + SORT [9]</td><td>32.9</td><td>35.4</td><td>30.5</td><td>16.6</td><td>9.3</td><td>29.8</td><td>14.9</td><td>7.8</td><td>28.3</td></tr><tr><td>VideoCUPS (Ours)</td><td>37.3</td><td>43.6</td><td>32.0</td><td>18.4</td><td>10.7</td><td>31.6</td><td>18.6</td><td>10.5</td><td>33.0</td></tr></table>

![](images/c97a25d69b6ac3e0fe1b7ee4c0ef8805cbe00b9a3d4b9dee297055b0656f99cc.jpg)

<details>
<summary>line</summary>

| w/o fine-tune | VideoCUPS (Ours) STQ | DINO STQ | Supervised (random init.) AQ | Supervised (Cityscapes pre-training init.) AQ | Supervised (random init.) S | Supervised (Cityscapes pre-training init.) S |
| ------------- | --------------------- | -------- | ----------------------------- | -------------------------------------------- | --------------------------- | -------------------------------------------- |
| 510           | 22                    | 33       | 18                            | 20                                           | 40                          | 45                                           |
| 20            | 35                    | 35       | 20                            | 22                                           | 45                          | 50                                           |
| 40            | 38                    | 37       | 22                            | 24                                           | 50                          | 55                                           |
| 60            | 39                    | 38       | 23                            | 25                                           | 55                          | 60                                           |
| 100           | 40                    | 39       | 24                            | 26                                           | 60                          | 65                                           |
| 224           | -                     | -        | -                             | -                                            | -                           | -                                            |
</details>

Fraction of Cityscapes-VPS [57] train labels used for training in %.   
Figure 4. Label-efficient learning. We fine-tune VideoCUPS and a DINO-initialized model on varying fractions of labeled Cityscapes-VPS train clips and report STQ, AQ, & SQ (all in %, ↑) on Cityscapes-VPS val. We also report models trained on the full Cityscapes-VPS train set, without any pre-training (rand. init.) and with supervised image pre-training on Cityscapes. 224 % denotes using both Cityscapes & Cityscapes-VPS. For training on Cityscapes-VPS subsets, we report the average and standard deviation over three different subsets.

Domain generalization results. In Tab. 2, we assess cross-domain generalization on KITTI-STEP, Waymo, and MOTS (OOD). VideoCUPS consistently outperforms all four baselines across datasets, achieving improvements of up to 3.3 % STQ on KITTI-STEP. We observe that STQ is higher for KITTI-STEP compared to Waymo and MOTS, as fewer instances need to be detected and tracked [77, 114]. These results showcase that unsupervised training generalizes effectively across domains. While the supervised model is still more accurate than unsupervised approaches, we observe that supervised learning is more susceptible to domain shifts, particularly on Waymo and MOTS.

# 4.2. Label-efficient learning

Achieving high-quality video panoptic segmentation ultimately depends on adapting to a predominantly humandefined semantic taxonomy, which remains beyond the reach of fully unsupervised approaches (cf . Tab. 1 & 2). A promising direction is unsupervised pre-training to acquire robust spatio-temporal and segmentation priors, followed by fine-tuning on a small set of labelled examples. This approach enables efficient adaptation to human-defined tasks while minimizing the need for extensive annotations.

In Fig. 4, we explore this scenario by comparing the unsupervised VideoCUPS-initialized model to the same architecture initialized with DINO [13] and trained with varying fractions of Cityscapes-VPS labels. We also report our supervised upper bound, using supervised panoptic image pre-training on Cityscapes and full Cityscapes-VPS finetuning. To assess the impact of the supervised image pretraining, we also report a randomly initialized model (with He init. [45]) trained on the full Cityscapes-VPS training set. Note that the Cityscapes and Cityscapes-VPS training splits are disjoint, containing 2 975 images / 2 975 labels and 400 clips / 2 400 labels, respectively.

Fine-tuning VideoCUPS with different fractions of VPS labels consistently outperforms the DINO pre-trained model. In particular, when using 10 % of Cityscapes-VPS labels, VideoCUPS improves by 4.6 % STP over DINO. While the delta reduces for larger fractions of annotations, VideoCUPS still outperforms DINO by 3.5 % STQ when using 100 % of labels. In comparison to the randomly initialized supervised model trained on 100 % of the Cityscapes-VPS labels, VideoCUPS requires only 10 % of the labels to reach the same STQ. Training VideoCUPS on all Cityscapes-VPS labels, almost closes the gap to the supervised model trained on both Cityscapes & Cityscapes-VPS (38.1 % vs. 40.4 % STQ), despite the latter using 124 % more labels. These results show that our unsupervised training is a strong initialization for learning with limited labels.

![](images/411b3bf85a6acfea02cd5f05850a25f90af8614716e593088e4cfe171fe1db86.jpg)

<details>
<summary>text_image</summary>

Road
Sidewalk
Building
Wall
Fence
Pole
T. Light
T. Sign
Veget.
Terrain
Sky
Person
Rider
Car
Truck
Bus
Train
M.cycle
Bicycle
CUPS VPS extd.
Ours
t0
t1
t2
t0
t1
t2
</details>

Figure 5. Qualitative pseudo-label examples. We compare VideoCUPS pseudo-labels to CUPS pseudo-labels extended to video with our approach. While CUPS benefits from stereo cues for improved depth, resulting in better semantics, our monocular pseudo-labels discover more instances, maintain longer tracks (t1 & t2; right), and capture more non-rigid motion (t1; left). We visualize matched pseudo-classes.

Table 3. Video pseudo-label generation ablation, analyzing the contribution of individual components, using STQ, AQ, and SQ (all in %, ↑) for pseudo-labels generated on Cityscapes-VPS val. 

<table><tr><td>Pseudo-label configuration</td><td>Mono.</td><td>STQ</td><td>AQ</td><td>SQ</td></tr><tr><td>Vanilla semantics + region growing instances</td><td>√</td><td>9.3</td><td>2.6</td><td>32.5</td></tr><tr><td>+ Depth-guided semantic inference [41]</td><td>√</td><td>9.4</td><td>2.7</td><td>32.6</td></tr><tr><td>+ Instance propagation &amp; tracking</td><td>√</td><td>12.0</td><td>4.4</td><td>32.4</td></tr><tr><td>+ Temporal semantic smoothing (full config.)</td><td>√</td><td>12.1</td><td>4.5</td><td>32.3</td></tr><tr><td>Video-extended CUPS pseudo labels [41]</td><td>✗</td><td>11.6</td><td>3.9</td><td>35.0</td></tr></table>

# 4.3. Analyzing VideoCUPS

VideoCUPS pseudo-label analysis. Table 3 presents an ablation of our pseudo-label generation, evaluating the contribution of each core component on Cityscapes-VPS val. Starting from combining the unsupervised semantic prediction of DepthG (vanilla semantics) with region-growing object proposals, we incrementally add depth-guided semantic inference [41], instance propagation, and temporal semantic smoothing. Each component contributes to improving the final STQ of our pseudo-labels, while our instance propagation aids the most. Temporal semantic smoothing results in only a minor increase in STQ. We attribute this partly to the limited temporal quality of the Cityscapes-VPS labels, as noted by Zhou et al. [124] and Woo et al. [117].

For reference, we also compare against CUPS pseudolabels generated using stereo video. In particular, we use the CUPS panoptic image pseudo-labels and extend these by our tracking and temporal smoothing (cf . Sec. 3.1) to obtain temporally consistent video pseudo-labels. Despite the absence of strong stereo cues, our purely monocular pseudo-labels achieve a higher STQ (12.1 % vs. 11.6 %). Only in SQ, the stereo pseudo-labels from CUPS improve over our monocular pseudo-labels. We attribute this to the lower-quality depth cues of our monocular approach, resulting in weaker depth-guided semantic inference. As a qualitative reference, we provide examples of our VideoCUPS

Table 4. VideoCUPS training ablation, analyzing the contribution of our core training components, using STQ, AQ, and SQ (all in %, ↑) on Cityscapes-VPS val. 

<table><tr><td>Training configuration</td><td>STQ</td><td>AQ</td><td>SQ</td></tr><tr><td>Vanilla training</td><td>17.8</td><td>10.0</td><td>31.8</td></tr><tr><td>+ Video DropLoss ( $\mathcal{L}_{\text{VDrop}}$ )</td><td>21.5</td><td>14.4</td><td>32.1</td></tr><tr><td>+ Video copy-paste augmentation</td><td>21.7</td><td>14.8</td><td>31.8</td></tr><tr><td>+ Self-enhance copy-paste augmentation (full config.)</td><td>22.2</td><td>15.3</td><td>32.3</td></tr></table>

pseudo-labels as well as our video extension of the CUPS pseudo-labels in Fig. 5.

VideoCUPS training analysis. In Tab. 4, we analyze the contribution of individual training components on Cityscapes-VPS. Starting from a vanilla training setup, adding our Video DropLoss improves STQ and AQ by mitigating instances missed by pseudo-labeling. Adding video copy-paste augmentation further improves STQ. Adding our self-enhanced copy-paste augmentation (full config.) achieves the highest STQ, aiding in the detection and tracking of small objects, as indicated by the improved AQ.

# 5. Conclusion

We introduced the task setting of unsupervised video panoptic segmentation and defined a comprehensive evaluation protocol across multiple scene-centric datasets. Our proposed method, VideoCUPS, is the first to approach this problem, showcasing that unsupervised panoptic video understanding can be achieved entirely without human supervision. VideoCUPS relies solely on monocular videos for VPS pseudo-labeling, removing the need for stereo. Compared with four proposed baselines built from state-of-theart unsupervised panoptic image and video instance segmentation methods, VideoCUPS consistently outperforms these baselines across various scene-centric VPS datasets. We further demonstrate that VideoCUPS provides a strong initialization for learning from limited annotated VPS examples. Together, our task definition, evaluation protocol, baselines, and method establish a foundation for future research on unsupervised panoptic video understanding.

Acknowledgments. This project was partially supported by the European Research Council (ERC) Advanced Grant SIM-ULACRON (grant agreement No. 884679), DFG project CR 250/26-1 “4D-YouTube”, and GNI Project “AICC”. This project was also partially supported by the ERC under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 866008). Additionally, this work has been co-funded by the LOEWE initiative (Hesse, Germany) within the emergenCITY center [LOEWE/1/12/519/03/05.001(0016)/72] and by the Deutsche Forschungsgemeinschaft (German Research Foundation, DFG) under Germany’s Excellence Strategy (EXC 3066/1 “The Adaptive Mind”, Project No. 533717223). Christoph Reich is supported by the Konrad Zuse School of Excellence in Learning and Intelligent Systems (ELIZA) through the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the German Federal Ministry of Education and Research. Christian Rupprecht is supported by an Amazon Research Award. Finally, we acknowledge the support of the European Laboratory for Learning and Intelligent Systems (EL-LIS) and thank Simone Schaub-Meyer for insightful discussions.

# References

[1] Rolf Adams and Leanne Bischof. Seeded region growing. IEEE Trans. Pattern Anal. Mach. Intell., 16(6):641–647, 1994. 3   
[2] Aria Ahmadi and Ioannis Patras. Unsupervised convolutional neural networks for motion estimation. In ICIP, pages 1629–1633, 2016. 2   
[3] Saleh Albelwi. Survey on self-supervised learning: Auxiliary pretext tasks and contrastive learning methods in imaging. Entropy, 24(4):551, 2022. 2   
[4] Shahaf Arica, Or Rubin, Sapir Gershov, and Shlomi Laufer. CuVLER: Enhanced unsupervised object discoveries through exhaustive self-supervised transformers. In CVPR, pages 23105–23114, 2024. 2   
[5] Yuki Markus Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In ICLR, 2020. 2   
[6] Philip Bachman, R. Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In NeurIPS, pages 15509–15519, 2019. 2   
[7] Adrien Bardes, Jean Ponce, and Yann LeCun. VI-CRegL: Self-supervised learning of local visual features. In NeurIPS, pages 8799–8810, 2022. 2   
[8] John L. Barron, David J. Fleet, and Steven S. Beauchemin. Performance of optical flow techniques. Int. J. Comput. Vis., 12(1):43–77, 1994. 2   
[9] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. Simple online and realtime tracking. In ICIP, pages 3464–3468, 2016. 6, 7, iii, v, vi, vii, viii, ix   
[10] Zhaowei Cai and Nuno Vasconcelos. Cascade R-CNN: Delving into high quality object detection. In CVPR, pages 6154–6162, 2018. 6, ii   
[11] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and

Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, pages 132–149, 2018. 2   
[12] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, pages 9912–9924, 2020. 2   
[13] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jé- gou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pages 9650–9660, 2021. 2, 3, 4, 6, 7, ii, iv, v   
[14] Vincent Casser, Soeren Pirk, Reza Mahjourian, and Anelia Angelova. Depth prediction without the sensors: Leveraging structure for unsupervised learning from monocular videos. In AAAI, pages 8001–8008, 2019. 2   
[15] Vincent Casser, Soeren Pirk, Reza Mahjourian, and Anelia Angelova. Unsupervised monocular depth and ego-motion learning with structure and semantics. In CVPRW, pages 381–388, 2019. 2   
[16] Xinlei Chen and Kaiming He. Exploring simple Siamese representation learning. In CVPR, pages 15750–15758, 2021. 2   
[17] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv:2003.04297 [cs.CV], 2020. 2   
[18] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In CVPR, pages 9640–9649, 2021. 2   
[19] Bowen Cheng, Anwesa Choudhuri, Ishan Misra, Alexander Kirillov, Rohit Girdhar, and Alexander G. Schwing. Mask2Former for video instance segmentation. arXiv:2112.10764 [cs.CV], 2021. 1   
[20] Jang Hyun Cho, Utkarsh Mall, Kavita Bala, and Bharath Hariharan. PiCIE: Unsupervised semantic segmentation using invariance and equivariance in clustering. In CVPR, pages 16794–16804, 2021. 3, 5, v   
[21] Anwesa Choudhuri, Girish Chowdhary, and Alexander G. Schwing. Context-aware relative object queries to unify video instance and panoptic segmentation. In CVPR, pages 6377–6386, 2023. 1   
[22] Subhabrata Choudhury, Laurynas Karazija, Iro Laina, Andrea Vedaldi, and Christian Rupprecht. Guess What Moves: Unsupervised video and image segmentation by anticipating motion. In BMVC, 2022. 3   
[23] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Scharwächter, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In CVPR, pages 3213–3223, 2016. 1, 2, 6, i, ii   
[24] Paul Couairon, Mustafa Shukor, Jean-Emmanuel Haugeard, Matthieu Cord, and Nicolas Thome. Diff-Cut: Catalyzing zero-shot semantic segmentation with diffusion features and recursive normalized cut. In NeurIPS, pages 13548–13578, 2024. 3, viii   
[25] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Häusser, Caner Hazırba¸s, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, and Thomas Brox. FlowNet: Learning optical flow with convolutional networks. In ICCV, pages 2758–2766, 2015. 2

[26] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16×16 words: Transformers for image recognition at scale. In ICLR, 2021. 2   
[27] Nikita Dvornik, Julien Mairal, and Cordelia Schmid. Modeling visual context is key to augmenting object detection datasets. In ECCV, pages 364–380, 2018. 5   
[28] Debidatta Dwibedi, Ishan Misra, and Martial Hebert. Cut, Paste and Learn: Surprisingly easy synthesis for instance detection. In ICCV, pages 1301–1310, 2017. 5   
[29] Omar Elharrouss, Somaya Al-Maadeed, Nandhini Subramanian, Najmath Ottakath, Noor Almaadeed, and Yassine Himeur. Panoptic segmentation: A review. arXiv:2111.10250 [cs.CV], 2021. 1, 3   
[30] Linus Ericsson, Henry Gouk, Chen Change Loy, and Timothy M. Hospedales. Self-supervised representation learning: Introduction, advances, and challenges. IEEE Trans. Signal Process., 39(3):42–62, 2022. 2   
[31] Ziyue Feng, Liang Yang, Longlong Jing, Haiyan Wang, YingLi Tian, and Bing Li. Disentangling object motion and occlusion for unsupervised multi-frame monocular depth. In ECCV, pages 228–244, 2022. 2   
[32] Ravi Garg, Vijay Kumar Bg, Gustavo Carneiro, and Ian Reid. Unsupervised CNN for single view depth estimation: Geometry to the rescue. In ECCV, pages 740–756, 2016. 2   
[33] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. In CVPR, pages 2918– 2928, 2021. 5   
[34] Clément Godard, Oisin Mac Aodha, and Gabriel J. Brostow. Unsupervised monocular depth estimation with leftright consistency. In CVPR, pages 270–279, 2017. 2   
[35] Clément Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J. Brostow. Digging into self-supervised monocular depth estimation. In ICCV, pages 3827–3837, 2019. 2   
[36] Xinrui Gong, Oliver Hahn, Christoph Reich, Krishnakant Singh, Simone Schaub-Meyer, Daniel Cremers, and Stefan Roth. Motion-refined DINOSAUR for unsupervised multiobject discovery. In ICCVW, pages 220–230, 2025. 3   
[37] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Remi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS, pages 21271–21284, 2020. 2   
[38] Vitor Guizilini, Rui Hou, Jie Li, Rares Ambrus, and Adrien Gaidon. Semantically-guided representation learning for self-supervised monocular depth. In ICLR, 2020. 2   
[39] Agrim Gupta, Jiajun Wu, Jia Deng, and Li Fei-Fei. Siamese masked autoencoders. In NeurIPS, pages 40676–40693, 2023. 2   
[40] Oliver Hahn, Nikita Araslanov, Simone Schaub-Meyer, and Stefan Roth. Boosting unsupervised semantic segmentation

with principal mask proposals. Trans. Mach. Learn. Res., 2024. 1, 3, 5, v   
[41] Oliver Hahn, Christoph Reich, Nikita Araslanov, Daniel Cremers, Christian Rupprecht, and Stefan Roth. Scenecentric unsupervised panoptic segmentation. In CVPR, pages 24485–24495, 2025. 1, 2, 3, 4, 5, 6, 7, 8, i, ii, iii, v, vi, vii, viii, ix   
[42] Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T. Freeman. Unsupervised semantic segmentation by distilling feature correspondences. In ICLR, 2022. 1, 2, 3, 5, ii, v   
[43] Robert M. Haralick and Linda G. Shapiro. Image segmentation techniques. Comput. Vis. Graph. Image Process., 29 (1):100–132, 1985. 3   
[44] Robert Harb and Patrick Knöbelreiter. InfoSeg: Unsupervised semantic image segmentation with mutual information maximization. In GCPR, pages 18–32, 2021. 3   
[45] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In ICCV, pages 1026–1034, 2015. 7   
[46] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 6, ii   
[47] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, pages 9729–9738, 2020. 2   
[48] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 16000–16009, 2022. 1, 2   
[49] Berthold K. P. Horn and Brian G. Schunck. Determining optical flow. Artif. Intell., 17(1–3):185–203, 1981. 2   
[50] Tak-Wai Hui. RM-Depth: Unsupervised learning of recurrent monocular depth in dynamic scenes. In CVPR, pages 1675–1684, 2022. 2   
[51] Juana Valeria Hurtado, Rohit Mohan, Wolfram Burgard, and Abhinav Valada. MOPT: Multi-object panoptic tracking. In CVPRW, 2020. 5, iii   
[52] Aleksandar Jevtic, Christoph Reich, Felix Wimbauer, ´ Oliver Hahn, Christian Rupprecht, Stefan Roth, and Daniel Cremers. Feed-forward SceneDINO for unsupervised semantic scene completion. In ICCV, pages 6784–6796, 2025. 3, 5   
[53] Xu Ji, Joao F. Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In ICCV, pages 9865–9874, 2019. 3   
[54] Rico Jonschkowski, Austin Stone, Jonathan T. Barron, Ariel Gordon, Kurt Konolige, and Anelia Angelova. What matters in unsupervised optical flow. In ECCV, pages 557– 572, 2020. 2   
[55] Laurynas Karazija, Subhabrata Choudhury, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Unsupervised multi-object segmentation by predicting probable motion patterns. In NeurIPS, pages 2128–2141, 2022. 3   
[56] Chanyoung Kim, Woojung Han, Dayun Ju, and Seong Jae Hwang. EAGLE: Eigen aggregation learning for object-

centric unsupervised semantic segmentation. In CVPR, pages 3523–3533, 2024. 3, 5, v   
[57] Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Video panoptic segmentation. In CVPR, pages 9856–9865, 2020. 1, 5, 6, 7, i, iii   
[58] Alexander Kirillov, Ross B. Girshick, Kaiming He, and Piotr Dollár. Panoptic feature pyramid networks. In CVPR, pages 6399–6408, 2019. 6, 7, ii, vi   
[59] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic segmentation. In CVPR, pages 9404–9413, 2019. 1, 5   
[60] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, pages 4015–4026, 2023. 1   
[61] Marvin Klingner, Jan-Aike Termöhlen, Jonas Mikolajczyk, and Tim Fingscheidt. Self-supervised monocular depth estimation: Solving the dynamic object problem by semantic guidance. In ECCV, pages 582–600, 2020. 2   
[62] Kurt Koffka. Principles of Gestalt psychology. Routledge, 1935. 3   
[63] Wolfgang Köhler. Gestalt psychology. Psychol. Forsch., 31 (1):XVIII–XXX, 1967. 3   
[64] Philipp Krähenbühl and Vladlen Koltun. Efficient inference in fully connected CRFs with Gaussian edge potentials. In NIPS, pages 109–117, 2011. 4, ii   
[65] Harold W. Kuhn. The Hungarian method for the assignment problem. Nav. Res. Logist., 2(1-2):83–97, 1955. 5   
[66] D. Khuê Lê-Huu and Karteek Alahari. Regularized Frank-Wolfe for dense CRFs: Generalizing mean field and beyond. In NeurIPS, pages 1453–1467, 2021. 4, ii   
[67] Seokju Lee, Sunghoon Im, Stephen Lin, and In So Kweon. Learning monocular depth in dynamic scenes via instanceaware projection consistency. In AAAI, pages 1863–1872, 2021. 2   
[68] Hanhan Li, Ariel Gordon, Hang Zhao, Vincent Casser, and Anelia Angelova. Unsupervised monocular depth learning in dynamic scenes. In CoRL, pages 1908–1917, 2021. 2   
[69] Xiangtai Li, Wenwei Zhang, Jiangmiao Pang, Kai Chen, Guangliang Cheng, Yunhai Tong, and Chen Change Loy. Video K-Met: A simple, strong, and unified baseline for video segmentation. In CVPR, pages 18847–18857, 2022. 1   
[70] Gal Lifshitz and Dan Raviv. Cost function unrolling in unsupervised optical flow. IEEE Trans. Pattern Anal. Mach. Intell., 46(2):869–880, 2024. 2   
[71] Runtao Liu, Zhirong Wu, Stella Yu, and Stephen Lin. The emergence of objectness: Learning zero-shot segmentation from videos. In NeurIPS, pages 13137–13152, 2021. 3   
[72] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 6, ii   
[73] Kaixuan Lu, Mehmet Onurcan Kaya, and Dim P. Papadopoulos. AutoQ-VIS: Improving unsupervised video instance segmentation via automatic quality assessment. In ICCVW, 2025. 3   
[74] Bruce D. Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In IJCAI, pages 674–679, 1981. 2

[75] Rémi Marsal, Florian Chabot, Angélique Loesch, and Hichem Sahbi. BrightFlow: Brightness-change-aware unsupervised learning of optical flow. In WACV, pages 2061– 2070, 2023. 2   
[76] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In CVPR, pages 4040–4048, 2016. 2   
[77] Jieru Mei, Alex Zihao Zhu, Xinchen Yan, Hang Yan, Siyuan Qiao, Liang-Chieh Chen, and Henrik Kretzschmar. Waymo Open Dataset: Panoramic video panoptic segmentation. In ECCV, pages 53–72, 2022. 5, 7, i, iii, v, viii   
[78] Simon Meister, Junhwa Hur, and Stefan Roth. UnFlow: Unsupervised learning of optical flow with a bidirectional census loss. In AAAI, pages 7251–7259, 2018. 2   
[79] Koichi Namekata, Amirmojtaba Sabour, Sanja Fidler, and Seung Wook Kim. EmerDiff: Emerging pixel-level semantic knowledge in diffusion models. In ICLR, 2024. 3   
[80] Duy Kien Nguyen, Yanghao Li, Vaibhav Aggarwal, Martin R. Oswald, Alexander Kirillov, Cees G. M. Snoek, and Xinlei Chen. R-MAE: Regions meet masked autoencoders. In ICLR, 2024. 2   
[81] Dantong Niu, Xudong Wang, Xinyang Han, Long Lian, Roei Herzig, and Trevor Darrell. Unsupervised universal image segmentation. In CVPR, pages 22744–22754, 2024. 1, 2, 3, 6, 7, ii, iii, v, vi, vii, viii, ix   
[82] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024. 2   
[83] Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan L. Yuille, and Liang-Chieh Chen. VIP-DeepLab: Learning visual perception with depth-aware video panoptic segmentation. In CVPR, pages 3997–4008, 2021. 1   
[84] Zhe Ren, Junchi Yan, Bingbing Ni, Bin Liu, Xiaokang Yang, and Hongyuan Zha. Unsupervised deep learning for optical flow estimation. In AAAI, 2017. 2   
[85] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet large scale visual recognition challenge. Int. J. Comput. Vis., 115(13):211–252, 2015. 2, ii   
[86] Sadra Safadoust and Fatma Güney. Multi-object discovery by low-dimensional object motion. In ICCV, pages 734– 744, 2023. 3   
[87] Alp Eren Sari and Paolo Favaro. FlowCut: Unsupervised video instance segmentation via temporal mask matching. arXiv:2505.13174 [cs.CV], 2025. 3   
[88] Hyun Seok Seong, WonJun Moon, SuBeen Lee, and Jae-Pil Heo. Leveraging hidden positives for unsupervised semantic segmentation. In CVPR, pages 19540–19549, 2023. 3, 5, v   
[89] Jianbo Shi and Jitendra Malik. Normalized cuts and image segmentation. IEEE Trans. Pattern Anal. Mach. Intell., 22 (8):888–905, 2000. 2

[90] Inkyu Shin, Dahun Kim, Qihang Yu, Jun Xie, Hong-Seok Kim, Bradley Green, In So Kweon, Kuk-Jin Yoon, and Liang-Chieh Chen. Video-kMaX: A simple unified approach for online and near-online video panoptic segmentation. In WACV, pages 229–239, 2024. 1   
[91] Leon Sick, Dominik Engel, Pedro Hermosilla, and Timo Ropinski. Unsupervised semantic segmentation through depth-guided feature correlation and sampling. In CVPR, pages 3637–3646, 2024. 1, 3, 4, 5, 6, 7, ii, iii, iv, v, vi, vii, viii, ix   
[92] Leon Sick, Dominik Engel, Sebastian Hartwig, Pedro Hermosilla, and Timo Ropinski. CutS3D: Cutting semantics in 3D for 2D unsupervised instance segmentation. In ICCV, pages 21265–21275, 2025. 1, 2, 5   
[93] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seung Eun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, et al. DINOv3. arXiv:2508.10104 [cs.CV], 2025. 1, 2, iv, v   
[94] Oriane Siméoni, Éloi Zablocki, Spyros Gidaris, Gilles Puy, and Patrick Pérez. Unsupervised object localization in the era of self-supervised ViTs: A survey. Int. J. Comput. Vis., 133(2):781–808, 2025. 2   
[95] Leonhard Sommer, Philipp Schröppel, and Thomas Brox. SF2SE3: Clustering scene flow into SE(3)-motions via proposal and selection. In GCPR, pages 215–229, 2022. 3   
[96] Austin Stone, Daniel Maurer, Alper Ayvaci, Anelia Angelova, and Rico Jonschkowski. SMURF: Self-teaching multi-frame unsupervised RAFT with full-image warping. In CVPR, pages 3887–3896, 2021. 2, 3, 4   
[97] Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume. In CVPR, pages 8934–8943, 2018. 2   
[98] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, et al. Scalability in perception for autonomous driving: Waymo Open Dataset. In CVPR, pages 2443–2451, 2020. 5, i, v, viii   
[99] Yihong Sun and Bharath Hariharan. Dynamo-Depth: Fixing unsupervised depth estimation for dynamical scenes. In NeurIPS, pages 54987–55005, 2023. 2, 3, 4, 6, ii, v, viii   
[100] Yihong Sun and Bharath Hariharan. MOD-UV: Learning mobile object detectors from unlabeled videos. In ECCV, pages 289–307, 2024. 2, 3   
[101] Narayanan Sundaram, Thomas Brox, and Kurt Keutzer. Dense point trajectories by GPU-accelerated large displacement optical flow. In ECCV, pages 438–451, 2010. 4   
[102] Zachary Teed and Jia Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In ECCV, pages 402–419, 2022. ii   
[103] Junjiao Tian, Lavisha Aggarwal, Andrea Colaco, Zsolt Kira, and Mar González-Franco. Diffuse, attend, and segment: Unsupervised zero-shot segmentation using stable diffusion. In CVPR, pages 3554–3563, 2024. 3   
[104] Wouter Van Gansbeke, Simon Vandenhende, and Luc Van Gool. Discovering object masks with transformers for

unsupervised semantic segmentation. arXiv:2206.06363 [cs.CV], 2022. 2   
[105] Shashanka Venkataramanan, Valentinos Pariza, Mohammadreza Salehi, Lukas Knobel, Spyros Gidaris, Elias Ramzi, Andrei Bursuc, and Yuki M. Asano. Franca: Nested matryoshka clustering for scalable visual representation learning. arXiv:2507.14137 [cs.CV], 2025. 2   
[106] Sudheendra Vijayanarasimhan, Susanna Ricco, Cordelia Schmid, Rahul Sukthankar, and Katerina Fragkiadaki. SfM-Net: Learning of structure and motion from video. arXiv:1704.07804 [cs.CV], 2017. 2   
[107] Paul Voigtlaender, Michael Krause, Aljosa O ˘ sep, Jonathon ˘ Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, and Bastian Leibe. MOTS: Multi-object tracking and segmentation. In CVPR, pages 7942–7951, 2019. 5, i, v, ix   
[108] Xinlong Wang, Zhiding Yu, Shalini De Mello, Jan Kautz, Anima Anandkumar, Chunhua Shen, and José M. Álvarez. FreeSOLO: Learning to segment objects without annotations. In CVPR, pages 14156–14166, 2022. 2   
[109] Xudong Wang, Rohit Girdhar, Stella X. Yu, and Ishan Misra. Cut and learn for unsupervised object detection and instance segmentation. In CVPR, pages 3124–3134, 2023. 1, 2, 3, 4, 5, 6, ii   
[110] Xudong Wang, Ishan Misra, Ziyun Zeng, Rohit Girdhar, and Trevor Darrell. VideoCutLER: Surprisingly simple unsupervised video instance segmentation. In CVPR, pages 22755–22764, 2024. 2, 3, 6, 7, iii, v, vi, vii, viii, ix   
[111] XuDong Wang, Jingfeng Yang, and Trevor Darrell. Segment anything without supervision. In NeurIPS, pages 138731–138755, 2024. 1, 2   
[112] Yangtao Wang, Xi Shen, Yuan Yuan, Yuming Du, Maomao Li, Shell Xu Hu, James L. Crowley, and Dominique Vaufreydaz. TokenCut: Segmenting objects in images and videos with self-supervised transformer and normalized cut. IEEE Trans. Pattern Anal. Mach. Intell., 45(12): 15790–15801, 2023. 2   
[113] Jamie Watson, Oisin Mac Aodha, Victor Prisacariu, Gabriel Brostow, and Michael Firman. The temporal opportunist: Self-supervised multi-frame monocular depth. In CVPR, pages 1164–1174, 2021. 2   
[114] Mark Weber, Jun Xie, Maxwell D. Collins, Yukun Zhu, Paul Voigtlaender, Hartwig Adam, Bradley Green, Andreas Geiger, Bastian Leibe, Daniel Cremers, Aljosa Osep, Laura Leal-Taixé, and Liang-Chieh Chen. STEP: Segmenting and tracking every pixel. In NeurIPS, 2021. 1, 5, 6, 7, i, ii, iii, v, vi, vii   
[115] Max Wertheimer. Experimentelle Studien über das Sehen von Bewegung. Zeitschrift für Psychologie, 61:161–165, 1912. 3   
[116] Felix Wimbauer, Nan Yang, Lukas Von Stumberg, Niclas Zeller, and Daniel Cremers. MonoRec: Semi-supervised dense reconstruction in dynamic environments from a single moving camera. In CVPR, pages 6112–6122, 2021. 2   
[117] Sanghyun Woo, Dahun Kim, Joon-Young Lee, and In So Kweon. Learning to associate every segment for video panoptic segmentation. In CVPR, pages 2705–2714, 2021. 1, 8, i

[118] Sungmin Woo, Wonjoon Lee, Woo Jin Kim, Dogyoon Lee, and Sangyoun Lee. ProDepth: Boosting self-supervised multi-frame monocular depth with probabilistic fusion. In ECCV, pages 201–217, 2024. 2   
[119] Guohuan Xie, Syed Ariff Syed Hesham, Wenya Guo, Bing Li, Ming-Ming Cheng, Guolei Sun, and Yun Liu. A comprehensive survey on video scene parsing: Advances, challenges, and prospects. arXiv:2506.13552 [cs.CV], 2025. 1   
[120] Gengshan Yang and Deva Ramanan. Learning to segment rigid motions from two frames. In CVPR, pages 1266– 1275, 2021. 2   
[121] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In ICCV, pages 5187–5196, 2019. 5, 6, 7, ii   
[122] Yanchao Yang, Brian Lai, and Stefano Soatto. DyStaB: Unsupervised object segmentation via dynamic-static bootstrapping. In CVPR, pages 2826–2836, 2021. 3   
[123] Jason J. Yu, Adam W. Harley, and Konstantinos G. Derpanis. Back to basics: Unsupervised learning of optical flow via brightness constancy and motion smoothness. In ECCVW, pages 3–10, 2016. 2   
[124] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image BERT pretraining with online tokenizer. In ICLR, 2022. 2, 8, i   
[125] Tinghui Zhou, Matthew Brown, Noah Snavely, and David G. Lowe. Unsupervised learning of depth and egomotion from video. In CVPR, pages 6612–6619, 2017. 2   
[126] Tianfei Zhou, Fei Zhang, Boyu Chang, Wenguan Wang, Ye Yuan, Ender Konukoglu, and Daniel Cremers. Image segmentation in foundation model era: A survey. arXiv: 2408.12957 [CV.cv], 2024. 1, 3

# Scene-Centric Unsupervised Video Panoptic Segmentation Supplementary Material

Christoph Reich\* 1,2,5,6

Oliver Hahn\* 2,3

Nikita Araslanov 1,5

Laura Leal-Taixé 3

Christian Rupprecht 4

Daniel Cremers† 1,5,6

Stefan Roth† 2,6,7

1TU Munich 2TU Darmstadt 3NVIDIA 4University of Oxford 5MCML 6ELIZA 7hessian.AI \*equal contribution †equal advising

https://visinf.github.io/videocups

In this supplement, we first provide additional implementation details, including dataset information, to aid reproducibility (Sec. A). We discuss our evaluation protocol and choice of metric in Section B. Next, we provide additional quantitative and qualitative results and analysis (Sec. C). Finally, we provide a comprehensive discussion on the limitations of VideoCUPS and outline potential future research directions (Sec. D).

# A. Reproducibility

This section provides further information about the datasets used and details on our implementation to ensure reproducibility. To further ensure reproducibility and provide a foundation for future work on unsupervised VPS, our code is available at https://github.com/visinf/cups/ tree/main/videocups.

# A.1. Datasets

Here, we provide additional details on the datasets utilized for training and evaluation.

Cityscapes [23] is a dataset of urban driving scenes composed of 5 000 high-resolution images at 1 024×2 048 pixels. The dataset is split into 2 975 training, 500 validation, and 1 525 test images, each annotated at the pixel level with panoptic labels. While Cityscapes provides different levels of semantic annotations (27, 19, and 7 semantic categories), the Cityscapes evaluation protocol employs 19 categories for evaluation. These 19 categories are composed of 8 “thing” and 11 “stuff” categories. Every annotated training image is extracted from a 30-frame video clip. Following prior work [41], we utilize these 2 975 training clips (86 275 video frames) for generating pseudo-labels and training.

Cityscapes-VPS [57] extends the Cityscapes dataset with panoptic video annotations. In particular, Cityscapes-VPS offers VPS annotations of the 500 Cityscapes validation clips. Cityscapes-VPS provides annotations for every fifth frame of each 30-frame clip using 19 categories matching Cityscapes. The annotations are obtained using a semiautomated annotation process with human correction. Still, multiple works pointed out labeling errors [117, 124]. The 500 Cityscapes-VPS dataset provides a split into 400 training, 50 validation, and 50 test clips. We perform evaluation on the 50 validation video sequences, following the originally proposed setting.

KITTI-STEP [114] provides panoptic video annotations for the KITTI-MOTS dataset [107, 128] and comprises 12 training, 9 validation, and 29 test videos. While Cityscapes-VPS provides more video clips and “thing” detections per frame, KITTI-STEP provides significantly longer video clips, on average 381 annotated frames per sequence, and longer tracks (average track length 51 frames). This is significantly longer than the 6 annotated frames of each Cityscapes-VPS clip. The semantic taxonomy of KITTI-STEP matches the 19-class taxonomy of Cityscapes, however, provides fewer “thing” classes—only persons and cars are annotated instance-wise. To compensate for this during evaluation, we perform semantic matching using the “thing” and “stuff” separation of Cityscapes and ignore video instance predictions of semantic classes different than “person” and “car”. For evaluation, we use the validation split.

Waymo [77, 98] comprises panoramic video panoptic annotations for 2 860 clips, covering a broad range of street scenes under diverse conditions (e.g., night, rain, etc.). The dataset provides five camera views and is split into 2 002 training, 286 validation, and 572 test clips. We use the forward-facing view with a resolution of 1 080 × 1 920 pixels. Following established practice [41], we map Waymo’s semantic labels to the Cityscapes taxonomy, resulting in 16 categories, and report results on the validation split. As Waymo includes a substantial number of very small “thing” detections, we remove instances with an average track size below 400 pixels. This mitigates the impact of extremely fine-grained annotations, which current unsupervised approaches cannot segment reliably, and ensures comparability to Cityscapes-VPS.

MOTS [107] is used to assess scenes-centric VPS accuracy on videos different from autonomous driving scenarios. We utilize the four MOTChallenge training video sequences, each composed of 2 866 frames, for evaluation. These frames entail a resolution of 1 080 × 1 920 or 480 × 640 pixels. MOTS provides two annotated categories “person” and “background”, while providing video instance segmentation for each “person” instance. We consider “person” as a “thing” category and “background” as a “stuff” category.

# A.2. Implementation details

We implement VideoCUPS using PyTorch [131] and train using PyTorch Lightning [127]. We utilized Detectron2 [138] for implementing Panoptic Cascade MaskTrack R-CNN and Kornia [133] for augmentations. Our implementation is partly built upon the code from previous work [41, 42, 91].

Pre-trained models. Our full pipeline utilizes SMURF [102], Dynamo-Depth [99], and DepthG [91]. To ensure full compliance with our purely unsupervised and monocular setup, we retrain Dynamo-Depth and DepthG. While SMURF has already been trained using monocular, unlabeled videos, DepthG uses a supervised depth model, and Dynamo-Depth initializes training with an ImageNet [85] supervised backbone. In particular, we retrain Dynamo-Depth [99] with a DINO ResNet-18 [13, 46], instead of an ImageNet-supervised [85] ResNet-18. DepthG is retrained with the monocular depth estimates of our retrained Dynamo-Depth model.

Pseudo-label generation. We generate panoptic video pseudo-labels using $c _ { \mathrm { p } } \quad = \quad 2 7$ pseudo-classes on the Cityscapes training sequences, following CUPS [41] and use a thing-stuff threshold of $\psi ^ { \mathrm { t s } } ~ = ~ 0 . 0 1 ~ ( c f$ . Sec. C.1). Semantic pseudo-labeling uses the prediction of our retrained DepthG, and depth-guided semantic inference follows the same setting as proposed in [41]. We post-process pseudo-labels with a CRF [64] using regularized Frank-Wolfe inference [66] and use the original hyperparameters. SMURF and our retrained Dynamo-Depth are used for pseudo-labeling. Our region growing uses $\alpha = 0 . 1 5$ , $\tau _ { d } = 0 . 0 2 , \tau _ { f } = 0 . 0 4$ , and $r = 8 ( c f$ . Tab. 8). Instance propagation and tracking uses a sliding window of length three and an IoU-threshold of $\tau _ { m } = 0 . 4$ in Hungarian matching (cf . Tab. 7). Temporal semantic smoothing likewise uses a three-frame sliding window.

Training and evaluation. To ensure fairness to our baselines U2Seg [81] and CUPS [41], which employ a Panoptic Cascade Mask R-CNN [10, 58], we use the closest video extension, the Panoptic Cascade MaskTrack R-CNN [10, 58, 121]. Following CutLER [109], U2Seg [81], and CUPS [41], we utilize a ResNet-50 [46] backbone with DINO [13] initialization, pre-trained self-supervised for two epochs on ImageNet [85]. Building on CUPS [41], we train using AdamW [72] with a base learning rate of $2 \times 1 0 ^ { - 5 }$ , our self-enhanced video copy-paste augmentation, and our Video DropLoss (with $\tau _ { \mathrm { I o U } } = 0 . 5 )$ for eight epochs. Our self-enhanced video copy-paste augmentation starts using model predictions after one epoch. During the first epoch, we copy-paste pseudo-labels. We paste between one and eight “thing” video detections into another video clip. Training was performed on four NVIDIA A100 GPUs (80 GB) using a batch size of 24. Evaluation of VideoCUPS and our unsupervised baselines is performed using the native resolution of each dataset (e.g., $1 0 2 4 \times 2 0 4 8$ for Cityscapes [23]).

# B. Unsupervised VPS Evaluation Protocol

As we train in a fully unsupervised fashion, our model only predicts pseudo-classes. These need to be mapped to the ground-truth categories. For this, we presented a simple and hyperparameter-free approach in Sec. 3.3. After mapping pseudo-categories to ground-truth categories, we utilize the established Segmentation and Tracking Quality (STQ) [114]. In the following, we provide details on the STQ and discuss other VPS metrics.

# B.1. Segmentation and Tracking Quality

After mapping pseudo-categories to ground-truth categories (cf . Sec. 3.3), we are equipped with the VPS predictions per clip $\mathbf { P } _ { i } = ( \check { \mathbf { S } } _ { i } , \mathbf { R } _ { i } )$ , where $\check { \mathbf { S } } _ { i } \in \{ 1 , 2 , \dotsc , c _ { \mathrm { g t } } \}$ T×H×W denotes the matched semantic predictions obtained using the pseudo-semantics S; i is the clip index, and $\mathbf { R } _ { i } \in$ $\{ 0 , \overset { \cdot } { 1 } \} ^ { n _ { p } \times \mathrm { T } \times \mathrm { H } \times \mathrm { W } }$ indicates the per-frame presence of $n _ { p }$ predicted “thing” video instances. For evaluation, we use the ground-truth VPS labels $\bar { { \bf P } } _ { i } = ( \bar { \bf S } _ { i } , \bar { \bf R } _ { i } )$ , with the semantic ground truth $\bar { \bf S } _ { i } \in \frac { \ d } { \ d t } \{ 1 , 2 , \dotsc , { \it c } _ { \mathrm { g t } } \} ^ { \mathrm { T } } \times \mathrm { H } \times \mathrm { W }$ and the corresponding $n _ { \mathrm { g t } }$ binary video instance masks $\bar { \bf R } _ { i } \in$ $\{ 0 , 1 \} ^ { n _ { \mathrm { g t } } \times \mathbf { \dot { T } } \times \mathbf { H } \times \mathbf { \dot { W } } }$ . The Segmentation and Tracking Quality is computed as

$$
\mathrm{STQ} = (\mathrm{AQ} \cdot \mathrm{SQ}) ^ {\frac {1}{2}}, \tag {4}
$$

where AQ is the Association Quality, and SQ is the Segmentation Quality. Specifically, the AQ measures how accurately instances were detected and tracked over time, while SQ measures how well pixel semantics were predicted, effectively decoupling segmentation and association.

More specifically, the Segmentation Quality, SQ, is defined as the mean Intersection over Union over the groundtruth classes $c \in \{ 1 , 2 , \ldots , c _ { \mathrm { g t } } \}$ computed as

$$
\mathrm{SQ} = \frac {1}{c _ {\mathrm{gt}}} \sum_ {c \in \{1, \dots , c _ {\mathrm{gt}} \}} \frac {\mathrm{TP} _ {c}}{\mathrm{TP} _ {c} + \mathrm{FP} _ {c} + \mathrm{FN} _ {c}}, \tag {5}
$$

with

$$
\mathrm{TP} _ {c} = \sum_ {i, t, h, w} \mathbb {1} \left[ \check {\mathbf {S}} _ {i, t, h, w} = c \right] \mathbb {1} \left[ \bar {\mathbf {S}} _ {i, t, h, w} = c \right], \tag {6}
$$

$$
\mathrm{FP} _ {c} = \sum_ {i, t, h, w} \mathbb {1} \left[ \check {\mathbf {S}} _ {i, t, h, w} = c \right] \mathbb {1} \left[ \bar {\mathbf {S}} _ {i, t, h, w} \neq c \right], \tag {7}
$$

$$
\mathrm{FN} _ {c} = \sum_ {i, t, h, w} \mathbb {1} \left[ \check {\mathbf {S}} _ {i, t, h, w} \neq c \right] \mathbb {1} \left[ \bar {\mathbf {S}} _ {i, t, h, w} = c \right], \tag {8}
$$

computed over the temporal and spatial dimensions as well as all evaluated clips.

The Association Quality, AQ, is computed using the “thing” video detections Ri and the ground truth $\bar { \mathbf { R } } _ { i }$ . First, the true positive $\mathrm { T P A } _ { i } ( g , p )$ , false positive $\mathrm { F P A } _ { i } ( g , p )$ , and false negative $\mathrm { F N A } _ { i } ( g , p )$ association areas for the predicted video instances $g \in \{ 1 , 2 , \ldots , n _ { p } \}$ and the groundtruth video instances $p \in \{ 1 , 2 , \ldots , n _ { \mathrm { g t } } \}$ are computed per clip i as

$$
\mathrm{TPA} _ {i} (g, p) = \sum_ {t, h, w} \bar {\mathbf {R}} _ {i, g, t, h, w} \mathbf {R} _ {i, p, t, h, w}, \tag {9}
$$

$$
\mathrm{FPA} _ {i} (g, p) = \sum_ {t, h, w} \mathbf {R} _ {i, p, t, h, w} - \mathrm{TPA} _ {i} (g, p), \tag {10}
$$

$$
\mathrm{FNA} _ {i} (g, p) = \sum_ {t, h, w} \bar {\mathbf {R}} _ {i, g, t, h, w} - \mathrm{TPA} _ {i} (g, p). \tag {11}
$$

Next, the pairwise association Intersection over Union Io $\mathsf { U } _ { i } ^ { \mathrm { A } } ( g , p )$ is computed using

$$
\mathrm{IoU} _ {i} ^ {\mathrm{A}} (g, p) = \frac {\mathrm{TPA} _ {i} (g , p)}{\mathrm{TPA} _ {i} (g , p) + \mathrm{FPA} _ {i} (g , p) + \mathrm{FNA} _ {i} (g , p)}. \tag {12}
$$

Finally, using $\mathrm { T P A } _ { i } ( g , p )$ and $\mathrm { I o U } _ { i } ^ { \mathrm { A } } ( g , p )$ , the Association Quality is computed as

$$
\mathrm{AQ} = \sum_ {i} \frac {\sum_ {g = 1} ^ {n _ {\mathrm{gt} , i}} \frac {1}{| \bar {\mathbf {R}} _ {i , g} |} \sum_ {p = 1} ^ {n _ {p , i}} \mathrm{TPA} _ {i} (g , p) \mathrm{IoU} _ {i} ^ {\mathrm{A}} (g , p)}{n _ {\mathrm{gt} , i}}, \tag {13}
$$

where $\left| { \bar { \bf R } } _ { i , g } \right|$ denotes the total area (i.e., the number of pixels) of the ground-truth video instances $g , n _ { \mathrm { g t } , i }$ the number of ground-truth instances of clip i, and $n _ { p , i }$ the number of predicted instances of clip i.

# B.2. Discussion

Existing work in the supervised domain offers alternative metrics for evaluating VPS, including the Video Panoptic Quality (VPQ) [57] and Panoptic Tracking Quality (PTQ) [51]. However, we adopt STQ as our primary metric in our proposed evaluation protocol for several reasons. First, STQ provides a clear separation between segmentation and association quality, yielding more interpretable insights into model behavior. Second, STQ evaluates entire videos at the per-pixel level and avoids the need for temporal windowing. In contrast, VPQ relies on fixed window sizes, which introduce sensitivity to the chosen hyperparameter and do not scale to full-length videos [114, 130]. Third, STQ does not use a threshold-based matching of “thing” predictions, in contrast to both PTQ and VPQ, making it more robust across object scales and crowded scenes. For these reasons, we refrain from incorporating VPQ into our evaluation protocol, as STQ provides the most stable, interpretable, and hyperparameter-free evaluation. In doing so, we follow KITTI-STEP [114], the 2D Video Panoptic Segmentation Challenge at CVPRW 2023, and the Waymo panoramic VPS dataset [77]. For a more detailed discussion of different VPQ metrics, we refer to Weber et al. [114].

Table 5. VPQ vs. STQ. We compare VideoCUPS to our unsupervised VPS baselines using VPQ and STQ (all in %, ↑) on KITTI-STEP. Both in VPQ and STQ, VideoCUPS outperforms the baselines. † denotes CUPS retrained using monocular videos. 

<table><tr><td>Method</td><td>VPQ</td><td>STQ</td></tr><tr><td>DepthG [91] + VideoCutLER [110]</td><td>14.3</td><td>13.2</td></tr><tr><td>U2Seg [81] + SORT [9]</td><td>19.0</td><td>24.0</td></tr><tr><td>CUPS [41] + SORT [9]</td><td>20.4</td><td>34.2</td></tr><tr><td> $CUPS^†$  [41] + SORT [9]</td><td>20.0</td><td>32.9</td></tr><tr><td>VideoCUPS (Ours)</td><td>21.1</td><td>37.3</td></tr></table>

Nevertheless, we report VPQ for VideoCUPS and all baselines on KITTI-STEP in Tab. 5 for completeness. VideoCUPS consistently outperforms all proposed baselines in both VPQ and STQ. We observe that the relative accuracy gap between the methods is smaller for VPQ than for STQ. We attribute this to the fundamental differences between the two metrics: VPQ is dominated by perframe/per-window mask quality, whereas STQ explicitly captures both recognition and temporal consistency. Additionally, STQ does not penalize for the recovery/correcting of tracks [see 114, for more details]. Therefore, STQ better reflects progress in unsupervised VPQ.

# C. Additional Results

Here, we provide additional qualitative and quantitative results extending our experiments in the main paper (cf . Sec. 4).

# C.1. Pseudo-label thing-stuff threshold analysis

The thing–stuff threshold $\psi ^ { \mathrm { t s } }$ introduced in Sec. 3.1 partitions the semantic pseudo-classes into pseudo-thing and pseudo-stuff classes based on their frequency inside the instance masks across the training data. Table 6 reports the effect of varying $\psi ^ { \mathrm { t s } }$ on pseudo-labels generated for the Cityscapes-VPS validation split. Very low thresholds assign many semantic pseudo-classes (e.g., 10 for $\psi ^ { \mathrm { t s } } = 0 . 0 0 2 5 )$ to the “thing” subset, leading to a form of over-clustering of true instance categories. Conversely, high thresholds reduce the number of thing pseudo-classes (e.g., 3 for $\psi ^ { \mathrm { t s } } = 0 . 0 3 )$ , which degrades results, as measured by the STQ. The best results are obtained at $\psi ^ { \mathrm { t s } } ~ = ~ 0 . 0 1$ , yielding five pseudothing classes and an STQ of 12.1 %.

# C.2. Tracking threshold analysis

In Tab. 7, we analyze the influence of the IoU-threshold $\tau _ { m }$ used for tracking and instance propagation (cf . Section 3.1).

Table 6. Pseudo-label thing-stuff threshold analysis. We evaluate pseudo-labels generated on Cityscapes-VPS val using different values for the thing-stuff threshold $\psi ^ { \mathrm { t s } }$ using STQ (in %, ↑). 

<table><tr><td> $\psi^{\text{ts}} \rightarrow$ </td><td>0.0025</td><td>0.005</td><td>0.01</td><td>0.02</td><td>0.03</td></tr><tr><td>STQ</td><td>7.5</td><td>11.5</td><td>12.1</td><td>11.3</td><td>10.7</td></tr></table>

Table 7. Video pseudo-label instance propagation and tracking threshold $\tau _ { m }$ analysis, using different IoU thresholds evaluated on pseudo-labels generated on Cityscapes-VPS val, using STQ, AQ, and SQ (all in %, ↑). 

<table><tr><td>Pseudo-label configuration</td><td>STQ</td><td>AQ</td><td>SQ</td></tr><tr><td> $\tau_{m} = 0.3$ </td><td>11.8</td><td>4.4</td><td>32.2</td></tr><tr><td> $\tau_{m} = 0.4$ </td><td>12.1</td><td>4.5</td><td>32.3</td></tr><tr><td> $\tau_{m} = 0.5$ </td><td>12.0</td><td>4.4</td><td>32.3</td></tr></table>

Table 8. Instance pseudo-labeling hyperparameter analysis. We analyse our region growing hyperparameters (instance seed threshold $\alpha ,$ relative depth threshold $\tau _ { d } ,$ relative flow threshold $\tau _ { f } ,$ and neighborhood radius r) on Cityscapes val and report STQ (in %, ↑). 

<table><tr><td rowspan="2"></td><td colspan="3"> $\leftarrow \alpha \rightarrow$ </td><td colspan="3"> $\leftarrow \tau_d \rightarrow$ </td><td colspan="3"> $\leftarrow \tau_f \rightarrow$ </td><td colspan="3"> $\leftarrow r \rightarrow$ </td></tr><tr><td>0.05</td><td>0.15</td><td>0.25</td><td>0.01</td><td>0.02</td><td>0.04</td><td>0.01</td><td>0.04</td><td>0.07</td><td>2</td><td>8</td><td>14</td></tr><tr><td>STQ</td><td>12.1</td><td>12.1</td><td>11.3</td><td>11.6</td><td>12.1</td><td>12.0</td><td>12.0</td><td>12.1</td><td>12.0</td><td>12.1</td><td>12.1</td><td>12.1</td></tr></table>

In particular, we generate pseudo-labels on the Cityscapes-VPS validation split and evaluate the pseudo-labels following the experimental setup from Tab. 3. Overall, our pseudo-labeling is robust to different $\tau _ { m }$ values. Nonetheless, setting $\tau _ { m } = 0 . 4$ yields a slightly better STQ than 0.3 and 0.5.

# C.3. Instance pseudo-labeling analysis

In Tab. 8, we provide an analysis of our instance pseudolabeling hyperparameters. We again generate pseudo-labels on the Cityscapes-VPS validation split and evaluate the pseudo-labels following the experimental setup from Tab. 3. We observe a robust behaviour of our pseudo-labeling w.r.t. the relative motion threshold $\tau _ { f }$ and the neighbourhood radius r. Both the instance seed threshold α and the relative depth threshold $\tau _ { d }$ still exhibit a relatively robust behaviour, while less robust than $\tau _ { f }$ and r.

# C.4. Dynamic vs. static analysis

Table 9 analyses the accuracy of VideoCUPS and our pseudo-labels on dynamic and static “thing” instances only. In particular, we utilize Cityscapes ground-truth motion masks [134] and compute STQ for moving “thing” instances (STQD) and static “thing” instances (STQS). When computing STQD, we ignore all “stuff” regions and static “thing” instances. Similarly, for STQS we ignore all “stuff” regions and dynamic “thing” instances. While our pseudolabels only capture dynamic objects, VideoCUPS improves accuracy on both static and dynamic objects. These results demonstrate the effectiveness of our Video DropLoss in enabling the network to detect and track objects missed by our pseudo-labels. Note that Cityscapes-VPS contains significantly more and smaller static than dynamic instances [134], resulting in lower $\mathrm { { \bf S T Q ^ { S } } }$ , which is also observed for the supervised upper bound.

Table 9. Dynamic vs. static analysis. We report STQ for dynamic $( \mathrm { S T Q } ^ { \mathrm { D } } )$ and static “thing” objects (STQS) only, ignoring stuff pixels. Both metrics in % (↑) on Cityscapes val. 

<table><tr><td>Approach</td><td>STQ $^{\text{D}}$ </td><td>STQ $^{\text{S}}$ </td></tr><tr><td>Supervised</td><td>42.8</td><td>28.6</td></tr><tr><td>Pseudo-labels</td><td>16.8</td><td>5.9</td></tr><tr><td>VideoCUPS</td><td>23.9</td><td>18.8</td></tr></table>

Table 10. Pseudo-labeling oracle. We analyze pseudo-labels generated using supervised depth, flow & motion masks and our unsupervised pseudo-labels on Cityscapes val. using STQ, AQ, and SQ (all in %, ↑). 

<table><tr><td>Pseudo-labels</td><td>STQ</td><td>AQ</td><td>SQ</td></tr><tr><td>Supervised</td><td>17.3</td><td>8.7</td><td>34.3</td></tr><tr><td>Unsupervised (Ours)</td><td>12.1</td><td>4.5</td><td>32.3</td></tr></table>

Table 11. SSL features for semantic pseudo-labeling analysis. We compare our modified version of DepthG [91] using DINO [13] and DINOv3 [93], evaluating semantic image segmentation using mIoU (in %, ↑) for unsupervised clustering and supervised linear probing on Cityscapes val. 

<table><tr><td>SSL-Features</td><td>Unsupervised mIoU</td><td>Supervised mIoU</td></tr><tr><td>DINO [13]</td><td>23.2</td><td>28.6</td></tr><tr><td>DINOv3 [93]</td><td>22.0</td><td>41.0</td></tr></table>

# C.5. Pseudo-labeling oracle

In Tab. 10, we provide an oracle experiment by using supervised cues to generate pseudo labels. In particular, we use supervised depth [139], flow [137], and motion masks [134] for pseudo-labeling. These supervised cues significantly improve pseudo-label accuracy, demonstrating the potential benefit of more accurate unsupervised depth, flow, and motion segmentation to improve unsupervised VPS. Note that this only improves the moving-object masks while still using unsupervised semantics.

# C.6. Analysing SSL features for semantic pseudolabeling

We analyze the effect of different SSL feature representations on the unsupervised semantic segmentation component used for our pseudo-labeling (cf . Sec. 3.1). We experiment using our DepthG [91] variant, adapted to the unsupervised and monocular setting by replacing supervised depth with monocular predictions from Dynamo-

Depth [99]. Both VideoCUPS and the original DepthG employ DINO [13] ViT-Base/8 features. We additionally evaluate DINOv3 [93] ViT-Base/16 features under the standard unsupervised semantic image segmentation protocol [20, 40, 42, 56, 88, 91] and report the mean Intersection over Union in Tab. 11.

Despite stronger segmentation results from supervised linear probing, DINOv3 yields inferior unsupervised segmentation mIoU compared to DINO (cf . Tab. 11). This aligns with prior observations for DINO compared to DI-NOv2 [40]. We attribute the drop to the substantially larger patch sizes in DINOv2/v3, which result in a reduced spatial resolution. While the representations become more discriminative in a supervised setting, their coarse spatial granularity appears detrimental for unsupervised clustering. We use DINO(v1) features in our experiments to ensure a fair comparison with U2Seg [81] and CUPS [41].

# C.7. Class-level analysis

Table 12 provides class-wise Segmentation Quality results of VideoCUPS and our baselines. Note that SQ only measures the segmentation accuracy, not detection and tracking accuracy. We observe that rare classes are still a significant challenge for all unsupervised approaches. For example, the predictions of VideoCUPS only capture five out of the eight “thing” classes. CUPS + SORT and CUPS† + SORT only capture four “thing” classes. Notably, while scoring a significantly lower overall SQ and aligning not well with ground-truth instances (cf . Tab. 1), U2Seg + SORT predicts all “thing” and “stuff” classes, most likely due to the significant overclustering with 827 pseudo-classes. On average, VideoCUPS outperforms our proposed baselines. In comparison to the supervised upper bound, missed classes account for most of the accuracy gap between supervised and unsupervised approaches, including VideoCUPS. For frequent classes (e.g., “Road”, “Sky”, or “Car”), our unsupervised VPS almost matches the results of the supervised upper bound. Fine-tuning on just a few VPS annotations (10 % of Cityscapes-VPS train) can adapt VideoCUPS to predict all semantic classes.

# C.8. Qualitative results

In addition to the Cityscapes-VPS qualitative results in Sec. 4.1, we present further visual comparisons. We compare VideoCUPS to the proposed baselines DepthG [91] + VideoCutLER [110], U2Seg [81] + SORT [9], CUPS [41] + SORT, and CUPS† [41] + SORT, across KITTI-STEP [114], Waymo [77, 98], and MOTS [107]. We also include qualitative out-of-domain (OOD) results on DAVIS [132]. Importantly, we apply no post-processing to avoid confounding the evaluation. We deliberately do not filter small masks or discard short-lived instance tracks, as this would introduce additional inference-time hyperparameters.

Figure 7 presents a qualitative comparison on KITTI-STEP. DepthG + VideoCutLER detects only a limited set of instances. U2Seg + SORT increases the number of predicted instances but frequently produces artifact-like instance predictions (e.g., erroneous arrow on the road; top example). CUPS + SORT yields a large number of instances with stable temporal identities, while its monocular variant, CUPS† + SORT, misses several smaller background instances. In contrast, VideoCUPS consistently discovers both near and far objects, producing accurate masks and temporally robust tracks.

Figure 8 compares all methods on the Waymo dataset. DepthG + VideoCutLER captures only prominent foreground objects and frequently merges distant instances into single masks (e.g., car 1, left). U2Seg + SORT predicts good semantics but continues to merge multiple objects and exhibits noticeable artifacts in the instance predictions. CUPS + SORT achieves strong semantic segmentation results and recovers many instances with stable temporal identities. Under the pronounced domain shift and in cluttered scenes, both CUPS + SORT and VideoCUPS occasionally predict small false instance predictions (e.g., lamppost; right example). The monocular variant, CUPS† + SORT, detects fewer objects and generates coarser instance masks (e.g., person 7, right). In contrast, VideoCUPS provides accurate semantics and numerous precise instance masks (e.g., person 6 and car 4, right) with consistent tracking across the entire sequence.

A qualitative assessment on the OOD dataset, MOTS, is provided in Fig. 9. DepthG + VideoCutLER recovers many of the foreground pedestrians but frequently merges multiple individuals into a single instance mask. U2Seg + SORT predicts pedestrian instances reliably, yet suffers from artifacts (e.g., store signs in the right example). CUPS + SORT outputs precise instance masks but occasionally fails to maintain tracks (e.g., person 2; left example). The monocular variant, CUPS† + SORT, yields coarser masks and more artifacts overall. Overall, VideoCUPS delivers the strongest qualitative results among all evaluated methods: it provides accurate instance masks with stable temporal associations, struggling only with very small distant objects.

We further assess the generalization ability of VideoCUPS. Figure 10 shows qualitative results on the DAVIS [132] dataset, using the class assignments learned on Cityscapes-VPS to map pseudo-classes to ground-truth categories for visualization. VideoCUPS generalizes well to this unseen domain and correctly handles unseen semantic concepts, such as forest and mountains (top examples). In addition, VideoCUPS produces accurate instance masks with consistent tracking over time.

Table 12. Class-level results on Cityscapes-VPS val. We compare VideoCUPS to the unsupervised VPS baselines, using the class-wise segmentation quality (SQ, in %, ↑). † denotes CUPS retrained using monocular videos. \* denotes “thing” classes with spatio-temporal instance annotations. For reference, we also report the class-wise scores of VideoCUPS fine-tuned with 10 % of the Cityscapes-VPS annotations. 

<table><tr><td>Method</td><td>Road</td><td>Sidewalk</td><td>Building</td><td>Wall</td><td>Fence</td><td>Pole</td><td>Traffic Light</td><td>Traffic Sign</td><td>Vegetation</td><td>Terrain</td><td>Sky</td><td>Person*</td><td>Rider*</td><td>Car*</td><td>Truck*</td><td>Bus*</td><td>Train*</td><td>Motorcycle*</td><td>Bicycle*</td><td>Mean (SQ)</td></tr><tr><td>Supervised [58]</td><td>88.7</td><td>73.0</td><td>83.1</td><td>47.9</td><td>57.6</td><td>51.1</td><td>47.9</td><td>63.8</td><td>82.4</td><td>58.2</td><td>85.7</td><td>72.0</td><td>58.7</td><td>84.8</td><td>65.7</td><td>82.5</td><td>26.2</td><td>47.2</td><td>64.9</td><td>65.3</td></tr><tr><td>DepthG [91] + VideoCutLER [110]</td><td>85.4</td><td>17.8</td><td>67.5</td><td>1.3</td><td>9.5</td><td>8.5</td><td>7.2</td><td>24.9</td><td>81.3</td><td>25.0</td><td>78.0</td><td>53.6</td><td>0.0</td><td>74.6</td><td>2.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>28.2</td></tr><tr><td>U2Seg [81] + SORT [9]</td><td>77.4</td><td>0.4</td><td>50.6</td><td>7.2</td><td>1.2</td><td>1.0</td><td>0.4</td><td>0.1</td><td>77.7</td><td>22.5</td><td>74.9</td><td>15.1</td><td>2.9</td><td>48.0</td><td>2.7</td><td>32.8</td><td>0.5</td><td>4.5</td><td>17.8</td><td>23.0</td></tr><tr><td>CUPS [41] + SORT [9]</td><td>81.1</td><td>16.2</td><td>63.2</td><td>1.3</td><td>4.9</td><td>30.6</td><td>17.6</td><td>40.7</td><td>80.9</td><td>36.3</td><td>81.9</td><td>47.9</td><td>0.0</td><td>61.1</td><td>0.0</td><td>23.3</td><td>0.0</td><td>0.0</td><td>18.2</td><td>31.8</td></tr><tr><td> $CUPS^†$ [41] + SORT [9]</td><td>81.3</td><td>13.2</td><td>62.2</td><td>1.5</td><td>5.0</td><td>30.3</td><td>19.7</td><td>38.1</td><td>81.1</td><td>30.1</td><td>81.9</td><td>45.4</td><td>0.0</td><td>62.8</td><td>0.0</td><td>12.4</td><td>0.0</td><td>0.0</td><td>2.3</td><td>29.9</td></tr><tr><td>VideoCUPS (Ours)</td><td>81.4</td><td>17.7</td><td>67.9</td><td>1.0</td><td>9.9</td><td>27.0</td><td>8.8</td><td>39.1</td><td>80.9</td><td>32.5</td><td>83.4</td><td>53.1</td><td>0.0</td><td>67.8</td><td>0.0</td><td>10.1</td><td>0.0</td><td>0.6</td><td>31.5</td><td>32.3</td></tr><tr><td>VideoCUPS w/ 10% VPS ann. (Ours)</td><td>85.6</td><td>43.9</td><td>83.0</td><td>16.8</td><td>35.7</td><td>37.4</td><td>37.5</td><td>52.7</td><td>83.0</td><td>19.5</td><td>88.2</td><td>70.0</td><td>44.8</td><td>82.9</td><td>24.8</td><td>5.1</td><td>21.6</td><td>18.2</td><td>56.5</td><td>47.7</td></tr></table>

![](images/5764948434539327c6315851801aec7f6810a2937bf88d8969d36a06f554786c.jpg)

<details>
<summary>natural_image</summary>

Street scene with cars on a tree-lined road, no visible text or symbols
</details>

![](images/d4eba3dcd547157666342d118678d3841359ccf88db0442caa39578d39ef33d9.jpg)

<details>
<summary>natural_image</summary>

Street scene with cars and trees, no visible text or symbols
</details>

Figure 6. VideoCUPS partial occlusion example on KITTI-STEP. While VideoCUPS struggles by design with full occlusion, VideoCUPS is still able to track objects through some partial occlusions (cf . object 154). Zoom in for details.

# D. Limitations and Future Work

Moving objects assumption. We show that unsupervised VPS is feasible by combining self-supervised visual representations with motion and depth cues. A current limitation is the requirement for independently moving objects to obtain initial video instance pseudo-labels. Although this assumption holds in many real-world scenarios, predominantly static objects, e.g., a painting mounted on a wall, remain challenging to segment in the unsupervised setting. MaskCut-based approaches such as U2Seg or VideoCut-LER can, in principle, discover such objects, but they require object-centric imagery and exhibit poor results on scene-centric data. Integrating motion-based segmentation with MaskCut-based pseudo-labeling may enable the segmentation of predominantly static objects while still scaling to scene-centric videos.

Dependency on driving scenes. While Most of our results are reported on driving scenes, VideoCUPS can be applied to non-driving-specific scenarios. Our approach only requires an agent moving through space, and target instances are movable, a common setting in robotics. Still, we require accurate unsupervised depth and motion, as well as VPS annotations for evaluation, which are mostly available for driving scenes (e.g., KITTI-STEP [114]). We show domain generalization of VideoCUPS beyond driving scenes on MOTS (cf . Tab. 2, Fig. 9) and DAVIS (cf . Fig. 10).

Occlusions. Motion segmentation, used for pseudolabeling, can only detect non-occluded objects. Additionally, partial occlusions, e.g., a car behind a pole, can lead to two detections of the same object cut by the partial occlusion. Subsequently, VideoCUPS, trained using these pseudo-labels, struggles to detect partially occluded objects correctly and fails to track temporarily fully occluded objects. Still, though our self-enhanced video copy-paste augmentation VideoCUPS can handle some degree of partial occlusions (cf . Fig. 6). Enhancing the pasting strategy of video copy-paste augmentations by systematically introducing partial and full occlusions, as well as by extending training to longer clips, might offer potential avenues to mitigate this limitation.

Unsupervised semantic taxonomy. Unsupervised segmentation approaches, including VideoCUPS, learn a segmentation taxonomy from unsupervised cues and imposed hyperparameters. While we demonstrate that VideoCUPS learns a taxonomy that significantly correlates with human-defined taxonomies, ideally, unsupervised approaches would learn a flexible, hierarchical taxonomy capable of expressing and discovering novel semantic categories. Creating more flexible approaches and benchmarks that treat unsupervised VPS as an open-vocabulary task would provide a path to overcoming this limitation. Additionally, unsupervised taxonomies require matching to a ground-truth taxonomy for validation. While most likely not suitable for evaluation, exploring ground-truthfree alignment between taxonomies could provide a powerful way to adapt unsupervised segmentation models to new taxonomies and to analyze the structure of the learned taxonomy [129, 135, 136].

Scaling to multiple datasets. Existing scene-centric unsupervised panoptic methods, such as CUPS, rely on stereo video during training. In contrast, VideoCUPS’s pseudo-

![](images/b152649c255e22e17fe51961a30c35879cb5acbb34a55cd443eff407147b49c7.jpg)  
Figure 7. KITTI-STEP—Qualitative unsupervised VPS examples. We compare our proposed method VideoCUPS to the proposed baselines DepthG [91] + VideoCutLER [110], U2Seg [81] + SORT, CUPS [41] + SORT [9], and CUPS† [41] + SORT [9] on KITTI-STEP [114] val.

![](images/883f646678fbdadaa91824ab0f0a0e11795a3063036d7b2637e06317b3e35e38.jpg)

<details>
<summary>text_image</summary>

Road
Sidewalk
Building
Wall
Fence
Pole
T. Light
T. Sign
Veget.
Terrain
Sky
Person
Rider
Car
Truck
Bus
Train
M.cycle
Bicycle
Ground-truth
DepthG+
VideoCutLER
U2Seg + SORT
CUPS + SORT
CUPS + SORT
VideoCUPS (Ours)
t0
t1
t2
t0
t1
t2
</details>

Figure 8. Waymo—Qualitative unsupervised VPS examples. We compare our proposed method VideoCUPS to the proposed baselines DepthG [91] + VideoCutLER [110], U2Seg [81] + SORT, CUPS [41] + SORT [9], and CUPS† [41] + SORT [9] on Waymo [77, 98] val.

labeling uses monocular videos. This provides an initial step toward scaling unsupervised panoptic video understanding to larger video datasets. Still, achieving true scalability to causal and monocular videos requires progress in two domains. First, while stereo depth estimation is robust and generalizes well, unsupervised monocular depth estimation is still limited. Current models, including Dynamo-Depth [99], are typically trained on a single dataset and typically do not generalize well to different cameras and other datasets/domains. Robust and generalizable unsupervised monocular depth estimation, including a static and dynamic scene decomposition, would enable more highquality pseudo-labels, enabling scaling VideoCUPS. Second, unsupervised semantic segmentation approaches must produce consistent pseudo semantics across diverse datasets and a large set of pseudo-categories. Current DINO-based unsupervised semantic segmentation approaches, including DepthG [91], typically train a segmentation head for a specific dataset. The resulting pseudo-categories do not necessarily align with pseudo-categories obtained when training on another dataset. Additionally, diffusion-based unsupervised semantic segmentation approaches, such as DiffCut [24], use language supervision and often provide pseudo-categories only consistent within a single image, requiring per-image matching for validation. Obtaining an approach that can express a globally consistent, hierarchical, and large-scale taxonomy of pseudo-categories is needed for scaling unsupervised VPS and unsupervised scene understanding in general.

# References

[127] William A. Falcon and The PyTorch Lightning team. Py-Torch Lightning. https://github.com/Lightning-AI/ pytorch-lightning, 2019. ii   
[128] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? The KITTI vision benchmark suite. In CVPR, pages 3354–3361, 2012. i   
[129] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. Position: The Platonic representation hypothesis. In ICML, pages 20617–20642, 2024. vi   
[130] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-scale video panoptic

![](images/be989b650a876fde0358e22b04b6489a478ae52d3279311349a0264c77be97f0.jpg)

<details>
<summary>text_image</summary>

Background
Ground-truth
DepthG + VideoCUTLER
U2Seg + SORT
CUPS + SORT
CUPS† + SORT
VideoCUPS (Ours)
t0
t1
t2
t0
t1
t2
Person
</details>

Figure 9. MOTS—Qualitative unsupervised VPS examples. We compare our proposed method VideoCUPS to the proposed baselines DepthG [91] + VideoCutLER [110], U2Seg [81] + SORT, CUPS [41] + SORT [9], and CUPS† [41] + SORT [9] on MOTS [107] val.

![](images/dc1d16fd3127693a385c836138490d06b092c30055ff9060f1d71f93ed88a3d3.jpg)

<details>
<summary>text_image</summary>

Road
Sidewalk
Building
Wall
Fence
Pole
T. Light
T. Sign
Veget.
Terrain
Sky
Person
Rider
Car
Truck
Bus
Train
M.cycle
Bicycle
t0
t1
t2
t0
t1
t2
</details>

Figure 10. Qualitative unsupervised VPS examples on DAVIS [132]. We provide qualitative samples for VideoCUPS inference on DAVIS videos using the Cityscapes-VPS class assignments for visualization purposes. VideoCUPS generalizes to the unseen dataset and even to unseen semantic concepts.

segmentation in the wild: A benchmark. In CVPR, pages 21033–21043, 2022. iii

[131] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison,

Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In NeurIPS, pages 8024–8035, 2019. ii

[132] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, pages 724–732, 2016 v, ix   
[133] Edgar Riba, Dmytro Mishkin, Daniel Ponsa, Bradski Ethan, and Gary Bradski. Kornia: An open source differentiable computer vision library for PyTorch. In WACV, pages 8024–8035, 2020. ii   
[134] Mennatullah Siam, Alex Kendall, and Martin Jagersand. Video class agnostic segmentation benchmark for autonomous driving. In CVPR, pages 2825–2834, 2021. iv   
[135] Rishi Jha, Collin Zhang, Vitaly Shmatikov, and John X. Morris. Harnessing the universal geometry of embeddings. In NeurIPS, 2025. vi   
[136] Dominik Schnaus, Nikita Araslanov, and Daniel Cremers. It’s a (blind) match! Towards vision-language correspondence without parallel data. In CVPR, pages 24983–24992, 2025. vi   
[137] Yihan Wang, Lahav Lipson, and Jia Deng. SEA-RAFT: Simple, efficient, accurate RAFT for optical flow. In ECCV, pages 36–54, 2024. iv   
[138] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github. com/facebookresearch/detectron2, 2019. ii   
[139] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. In NeurIPS, pages 21875–21911, 2024. iv