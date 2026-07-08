# Geometric Reciprocity: Unlocking Self-Supervision for Stereoscopic Video Generation

Jingyi Lu <sup>1</sup>

## Abstract

Monocular-to-stereo conversion synthesizes stereoscopic content from 2D videos for immersive 3D experiences. In modern Depth-Image-Based Rendering (DIBR) approaches, stereo inpainting of disocclusions is the critical bottleneck. Training-based methods achieve superior quality but rely on scarce stereo pairs or synthetic data with domain gaps. We address this through the first self-supervised framework learning from monocular videos via cycle consistency. Our key contribution is the Geometric Reciprocity Theorem (GRT): under the nearest-neighbor DIBR formulation, the disocclusion mask when synthesizing a target view equals the mask of pixels lost when warping back from target to source, enabling analytical computation of test-time disocclusion masks directly from monocular images. This yields train-test consistency for the stated warping formulation, supporting self-supervised learning from unlimited monocular videos and substantial improvements over training-free and supervised state-of-the-art methods. Project page: https://visual-ai.github.io/grt/

## 1. Introduction

The demand for immersive 3D experiences on VR/AR devices, 3D cinema, and stereoscopic displays has made stereoscopic video generation a fundamental problem. The task is to convert monocular videos to stereoscopic format by synthesizing one view from the other (conventionally right-eye from left-eye, or vice versa).

Modern approaches predominantly adopt the Depth-Image-Based Rendering (DIBR) framework (Wang et al., 2024; Shi et al., 2024; Dai et al., 2024; Zhao et al., 2024; Huang et al.,

Kai Han <sup>1</sup>

2025; Shvetsova et al., 2026), which decomposes the problem into three sequential stages: estimating depth from the input frame, warping the image to synthesize an initial target view, and inpainting the resulting disocclusions. These disocclusions are regions newly visible in the target view that were occluded in the source, creating geometrically structured missing patterns that general-purpose inpainting methods cannot handle (see Section A7). This domain gap has made the stereo inpainting stage the critical bottleneck.

Due to the scarcity of training data, recent training-free methods (Wang et al., 2024; Dai et al., 2024) manipulate pretrained diffusion models for zero-shot stereo inpainting, but lack stereo-specific geometric priors and produce inferior results. Training-based approaches (Zhao et al., 2024; Huang et al., 2025; Shvetsova et al., 2026) learn these priors yet face a fundamental challenge in training data quality and availability. Existing methods either rely on scarce and often proprietary real stereo pairs with error-prone stereo matching for disocclusion identification (Zhao et al., 2024; Shvetsova et al., 2026), or resort to synthetic data that suffers from inevitable domain gaps (Huang et al., 2025) (see Section A8 for more details).

To address the data bottleneck, we propose the first selfsupervised framework that learns stereo inpainting from monocular videos alone, eliminating the need for stereo pairs or synthetic data. Our approach exploits the inherent bidirectional symmetry of stereo view relationships, enabling right-left-right cycle consistency where generating the left view from right through DIBR and then reconstructing the right from the synthesized left should recover the original input.

While cycle consistency provides self-supervision in principle, naive implementation requires multiple sequential model inferences and backpropagation through nondifferentiable warping operations, making video-scale training computationally prohibitive. We resolve this through a geometric insight: the stereo warping process itself encodes all information needed to compute cycle consistency losses without executing the cycle. Through rigorous geometric analysis, we discover that in the right-left-right cycle, the intermediate steps of inpainting the left view, estimating its depth, and performing pixel warping operations are redundant under the nearest-neighbor DIBR formulation. We formalize this as the Geometric Reciprocity Theorem (GRT): for any right (target) view to be synthesized from a left (source) view, the disocclusion mask required equals the mask of pixels that would be lost when warping back from target to source. This theorem reveals that the cycle consistency supervision signal can be computed analytically from the target view alone using only its depth estimate, with no synthesis, no intermediate views, and no cycling required.

Table 1. Comparison of training data construction approaches. Our Geometric Reciprocity Theorem enables constructing scalable, high-quality training data with train-inference consistent masks from arbitrary real-world monocular videos.

<table><tr><td>Preferable Properties</td><td>Data Scalability</td><td>Real-World Data</td><td>Mask Quality</td><td>Train-Inference Consistency</td><td>Training Efficiency</td></tr><tr><td>Real Stereo Pairs</td><td>✕</td><td>√</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Synthetic Data</td><td>√</td><td>✕</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Cycle Consistency (Ours)</td><td>√</td><td>√</td><td>-</td><td>√</td><td>✕</td></tr><tr><td>Geometric Reciprocity (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

GRT fundamentally transforms training data construction. By treating any monocular image as a target view, we directly compute its inference-time disocclusion mask via GRT, yielding the mask induced by the same DIBR geometry used at test time. The image itself then serves as ground truth for these disocclusions, enabling self-supervised training from unlimited monocular videos with train-inference consistency that substantially outperforms both training-free methods and supervised state-of-the-art. We summarize the above comparisons in Table 1.

Our contributions are: (1) the first self-supervised stereo inpainting framework learning from monocular videos without requiring stereo pairs or synthetic data, (2) the Geometric Reciprocity Theorem enabling analytical cycle consistency computation without executing the cycle, (3) the first comprehensive datasets (ImageNet-GRT, Kinetics-GRT, and DAVIS-GRT) with geometrically consistent disocclusion masks for training and evaluation, and (4) state-of-theart performance surpassing both training-free and supervised methods. Code, precomputed masks, and trained weights will be released at https://github.com/ Visual-AI/GRT.

## 2. Related work

## 2.1. Monocular-to-Stereo Video Conversion

Monocular-to-stereo video conversion synthesizes a righteye view from standard 2D video (assumed as left-eye input) for immersive 3D experiences on stereoscopic displays, 3D cinemas, and VR/AR devices. Early work (Xie et al., 2016) attempted direct regression using end-to-end CNNs but was limited by lack of robust depth priors. Modern approaches predominantly adopt Depth-Image-Based Rendering (DIBR), which first estimates per-frame depth maps with pretrained models like Depth Anything (Yang et al., 2024), then warps the left view to synthesize an initial right view. This warping uncovers disocclusions—areas occluded in the left view—framing monocular-to-stereo conversion as stereo inpainting: filling newly visible regions.

Recent solutions tackle this challenge in various ways. Training-free methods leverage pretrained models without fine-tuning. StereoDiffusion (Wang et al., 2024) performs zero-shot inpainting by manipulating latents in pretrained image diffusion models. StereoCrafter-Zero (Shi et al., 2024) uses noisy restart and iterative refinement on video diffusion models. SVG (Dai et al., 2024) contributes a Frame Matrix architecture for video consistency while remaining training-free.

Training-based methods demonstrate superior quality and consistency by learning stereo-specific priors from datasets. StereoCrafter (Zhao et al., 2024) fine-tunes Stable Video Diffusion using auto-regressive strategies for temporal coherence. Restereo (Huang et al., 2025) jointly addresses stereo generation and video restoration via training on synthetically degraded data. M2SVid (Shvetsova et al., 2026) improves efficiency through single-step feed-forward prediction conditioned on both original left and warped right views. However, without ready-to-use stereo inpainting datasets, these methods construct training data through preprocessing pipelines. Methods using real stereo pairs face stereo matching errors and copyright restrictions, while those using synthetic data face domain gaps between rendered and real videos, limiting generalization.

## 2.2. Monocular Depth Estimation

Monocular Depth Estimation (MDE) infers dense depth maps from single images. Early methods trained on singledomain datasets like KITTI (Geiger et al., 2012) or NYU-D (Silberman et al., 2012) performed well in specific scenarios but lacked generalization. Recent foundation models achieve zero-shot capabilities through multi-dataset aggregation (MiDaS (Ranftl et al., 2020)) and massive unlabeled datasets (Depth Anything (Yang et al., 2024), Depth-Pro (Bochkovskii et al., 2025)). Advances include generative models like Marigold (Ke et al., 2024), Vision Transformer architectures (Ranftl et al., 2021), and robustness improvements under adverse conditions (Kong et al., 2023; Zheng et al., 2023; Sun et al., 2025). For DIBR-based monocular-to-stereo conversion, predicted depth maps govern warping by defining per-pixel horizontal displacement, with closer objects shifted more than distant ones to create stereoscopic effects.

## 2.3. Cycle Consistency

Cycle consistency enables unsupervised learning by enforcing bidirectional consistency $( A \to B \to A )$ CycleGAN (Zhu et al., 2017) demonstrated this for unpaired image-to-image translation. Applications span domain adaptation (Hoffman et al., 2018; Singh, 2021), temporal learning (Dwibedi et al., 2019; Yang et al., 2021), 3D dense correspondence (Zhou et al., 2016), and super-resolution (Yuan et al., 2018). TrajectoryCrafter (Yu et al., 2025a) extends cycle consistency to multi-view synthesis through explicit forward-backward reprojection, requiring two full rendering passes. In contrast, our GRT reveals the analytical redundancy in this cycle and enables direct disocclusion mask derivation from target-view geometry alone.

## 3. Method

We present a self-supervised approach for training stereo inpainting networks using monocular videos (images). Building on the DIBR framework (Section 3.1), we identify data scarcity and domain gaps as key bottlenecks (Section 3.2). We introduce cycle consistency (Section 3.3) as a self-supervision mechanism and prove the Geometric Reciprocity Theorem (GRT) (Section 3.4), which eliminates the computational overhead of cycle consistency while preserving equivalent geometric constraints, enabling selfsupervised learning from monocular videos (images) without requiring paired stereo data (Section 3.5).

## 3.1. Preliminaries

Stereoscopic video generation synthesizes stereo pairs from monocular input for immersive 3D visualization. Given a monocular frame as the left-eye view $I _ { L }$ , the task is to generate the corresponding right-eye view $I _ { R } .$ . While the reverse direction is equally valid, we adopt the left-to-right formulation for clarity. Modern approaches employ the Depth-Image-Based Rendering (DIBR) framework, which reduces stereoscopic generation to a stereo inpainting problem. First, a depth estimation model D predicts disparity, which is inversely proportional to depth:

$$
d _ {L} = \mathcal {D} (I _ {L}), \quad d _ {L} = \frac {b f}{Z _ {L}},\tag{1}
$$

where $d _ { L } ( x , y )$ denotes the disparity at pixel $( x , y )$ $Z _ { L } ( x , y )$ is the corresponding depth, b is the baseline, and f is the focal length. We adopt the convention that disparity values are positive, with larger values indicating closer objects. Second, warping projects the input to the target viewpoint. Specifically, $W _ { L  R }$ establishes pixel correspondences between views. For each pixel $( x , y )$ in the left view, its corresponding position $( x ^ { \prime } , y ^ { \prime } )$ in the right view is computed as:

$$
x ^ {\prime} = x - d _ {L} (x, y), \quad y ^ {\prime} = y,\tag{2}
$$

where the horizontal coordinate is shifted by the disparity value, while the vertical coordinate remains unchanged due to rectified stereo geometry. During warping, multiple source pixels may map to the same target location (collision), or some target pixels may receive no mapping (disocclusion). The warping operation outputs the warped image $\tilde { I } _ { R }$ and the disocclusion mask $M _ { \mathrm { d i s } } ^ { L  }$

$$
\tilde {I} _ {R}, M _ {\mathrm{dis}} ^ {L \rightarrow R} = W _ {L \rightarrow R} (I _ {L}, d _ {L}),\tag{3}
$$

where $M _ { \mathrm { d i s } } ^ { L  R } ( x , y ) = 1$ indicates pixels that are disoccluded in $I _ { R }$ and require inpainting. Third, a stereo inpainting network fills these holes:

$$
\hat {I} _ {R} = G (\tilde {I} _ {R}, M _ {\mathrm{dis}} ^ {L \rightarrow R}),\tag{4}
$$

where G represents the stereo inpainting network.

## 3.2. Motivation

The primary bottleneck in DIBR-based stereoscopic generation is training the stereo inpainting network $G ,$ stemming from the unique characteristics of stereoscopic disocclusion patterns and the scarcity of suitable training data.

Domain gap in disocclusion patterns. Stereoscopic disocclusion masks $M _ { \mathrm { d i s } } ^ { L  R }$ exhibit geometrically structured patterns fundamentally different from typical inpainting masks (random strokes, rectangular regions, or object instances). This domain gap causes general inpainting methods and training-free methods (Wang et al., 2024; Dai et al., 2024) to produce suboptimal results, lacking stereo-specific geometric priors for plausible disocclusion filling. Data scarcity for supervised training. Supervised approaches (Zhao et al., 2024; Huang et al., 2025) require training pairs $( I _ { R } , M _ { \mathrm { d i s } } ^ { L  R } )$ , where the stereo inpainting network takes masked right view $I _ { R } { \odot } \big ( 1 { - } M _ { \mathrm { { d i s } } } ^ { L  R } \big )$ and disocclusion mask $M _ { \mathrm { d i s } } ^ { L  R }$ as input, using $I _ { R }$ for supervision. However, no ready-to-use datasets exist. To construct such data from real stereo pairs, StereoCrafter (Zhao et al., 2024) collects real stereo video pairs and generates training data by aligning left to right views to identify disocclusion regions $\dot { M } _ { \mathrm { d i s } } ^ { L  R }$ . However, this alignment introduces stereo matching errors that severely degrade data quality. Moreover, high-quality stereo videos predominantly come from copyrighted 3D films with restricted access, further limiting availability. Synthetic data approaches, such as Restereo (Huang et al., 2025) using Kubric (Greff et al., 2022), avoid stereo matching errors but suffer significant domain gaps between rendered and real-world videos, limiting real-world generalization.

Cycle Consistency  
![](images/026a1c5e5281ade6c9dbb8ea6b5e4befc5de981f683d8539129a26ced57a6f8f.jpg)  
Figure 1. Cycle consistency framework. Given right view $I _ { R } ,$ , the complete cycle synthesizes left view through depth estimation (D), forward warping $( W _ { R  L } )$ , and inpainting (G), then reconstructs the right view via depth estimation on the synthesized left view, backward warping $( W _ { L  R } )$ , and inpainting. The reconstruction loss $\mathcal { L } _ { \mathrm { c y c l e } } = \| \hat { I } _ { R } ^ { \mathrm { r e c o n } } - I _ { R } \|$ provides self-supervision.

To overcome these limitations, we propose the first selfsupervised approach leveraging abundant monocular videos (images) for training stereo inpainting networks. Our method demonstrates that, based on cycle consistency, disocclusion masks $M _ { \mathrm { d i s } } ^ { L  R }$ can be directly derived from right views $I _ { R }$ to construct training data, eliminating the need for stereo pairs or synthetic data.

## 3.3. Cycle Consistency

We introduce right-left-right cycle consistency as a selfsupervision mechanism for stereoscopic generation. As illustrated in Figure 1, the principle is geometric and bidirectional: synthesizing the left view from the right, then reconstructing the right from this synthesized left, should recover the original input. Given a frame $I _ { R } ,$ we first synthesize a pseudo left-eye view through the DIBR framework:

$$
d _ {R} = \mathcal {D} (I _ {R}),\tag{5}
$$

$$
\tilde {I} _ {L}, M _ {\mathrm{dis}} ^ {R \rightarrow L} = W _ {R \rightarrow L} (I _ {R}, d _ {R}),\tag{6}
$$

$$
\hat {I} _ {L} = G (\tilde {I} _ {L}, M _ {\mathrm{dis}} ^ {R \rightarrow L}).\tag{7}
$$

The right view is then reconstructed from $\hat { I } _ { L }$ :

$$
\hat {d} _ {L} = \mathcal {D} (\hat {I} _ {L}),\tag{8}
$$

$$
\tilde {I} _ {R}, M _ {\mathrm{dis}} ^ {L \rightarrow R} = W _ {L \rightarrow R} (\hat {I} _ {L}, \hat {d} _ {L}),\tag{9}
$$

$$
\hat {I} _ {R} ^ {\mathrm{recon}} = G (\tilde {I} _ {R}, M _ {\mathrm{dis}} ^ {L \to R}).\tag{10}
$$

The cycle consistency loss enforces that reconstruction matches input:

$$
\mathcal {L} _ {\text { cycle }} = \left\| \hat {I} _ {R} ^ {\text { recon }} - I _ {R} \right\|.\tag{11}
$$

While cycle consistency enables self-supervised training without paired stereo data, naive implementation is computationally prohibitive. End-to-end differentiability requires backpropagating through warping operations $W ( \cdot , \cdot )$ that depend on estimated disparity $\mathcal { D } ( \cdot )$ , yet warping involves discrete pixel coordinate mapping and scattered writes to irregular locations, operations fundamentally non-differentiable in nature. Differentiable rendering approximations introduce substantial overhead and complexity. More critically, the cycle demands four sequential model inferences per training step: disparity estimation from $I _ { R } ,$ , left view inpainting, disparity re-estimation from $\hat { I } _ { L }$ , and final right view inpainting. This doubles memory requirements and training costs while accumulating errors through multiple non-linear transformations, making video-scale training computationally infeasible.

## 3.4. Geometric Reciprocity Theorem

We resolve all aforementioned challenges through a key geometric insight: the stereo warping process itself inherently encodes all information needed to enforce cycle consistency. Building on this insight, we mathematically prove that the entire gradient-intensive left view synthesis steps, including inpainting the left view, estimating its depth, and performing pixel warping operations, do not affect the cycle consistency loss $\mathcal { L } _ { \mathrm { c y c l e } }$ and can therefore be eliminated entirely. We formalize this result as follows:

Geometric Reciprocity Theorem (GRT). Under the nearest-neighbor DIBR formulation, for any right (target) view $I _ { R }$ synthesized from a left (source) view, the disocclusion mask required for synthesis is mathematically equivalent to the mask of pixels that would be lost when warping from right (target) to left (source):

$$
M _ {\mathrm{dis}} ^ {L \rightarrow R} = M _ {\mathrm{lost}} ^ {R \rightarrow L},\tag{12}
$$

where $M _ { \mathrm { l o s t } } ^ { R  L }$ identifies pixels in $I _ { R }$ that would be lost due to boundary violations or depth occlusions during right-toleft warping with estimated disparity $d _ { R } \ = \ { \mathcal { D } } ( I _ { R } )$ . By treating any monocular image as a right (target) view, GRT enables us to directly compute the inference-time disocclusion mask that would emerge when synthesizing it from a left (source) view, using only depth estimation from $I _ { R } .$ . The image itself then serves as ground truth for these disocclusions, enabling self-supervised training from monocular videos (images) while bypassing the entire synthesis pipeline.

![](images/1d8f2baf3149b2d91da183d6e51347679b1a837382799e368de5f2391e5c2af4.jpg)  
Figure 2. Progressive simplification of cycle consistency to Geometric Reciprocity. (i) Inpainted regions in $\hat { I } _ { L }$ do not affec $\hat { I } _ { R } ^ { \mathrm { r e c o n } }$ allowing us to skip left view inpainting (marked with ×) and directly use $\tilde { I } _ { L }$ . (ii) Right-to-left warping transfers disparity from $d _ { R }$ to $\tilde { I } _ { L }$ , allowing us to skip left view disparity estimation and directly reuse $\tilde { d } _ { L }$ . (iii) Transferred disparity ensures perfect round-trips for all validly warped pixels, enabling analytical computation of $M _ { \mathrm { d i s } } ^ { L  }$ as pixels lost during right-to-left warping and eliminating all warping operations (marked with ×). The final result reveals that $M _ { \mathrm { d i s } } ^ { L  R } = M _ { \mathrm { l o s t } } ^ { R  L }$ can be computed directly from $( I _ { R } , d _ { R } )$ alone.

## 3.4.1. PROOF OF GRT

We prove the GRT by progressively simplifying the cycle consistency framework through three geometric observations visualized in Figure 2. Remarkably, we demonstrate that each computational step in the naive cycle can be eliminated without affecting the final result, revealing an elegant mathematical structure underlying stereoscopic generation. Note that this proof assumes nearest-neighbor warping; we extend to soft interpolation warping in Section A9.

Eliminating left view inpainting. We first observe that the inpainted left view content, $\hat { I } _ { L } \stackrel { - } { = } G ( \tilde { I } _ { L } , M _ { \mathrm { d i s } } ^ { R  L } )$ , is geometrically irrelevant to the cycle consistency loss. Consider pixels where $M _ { \mathrm { d i s } } ^ { R  L } = 1$ : these regions were disoccluded during right-to-left warping and thus have no physical counterpart in the original $I _ { R }$ . When warping back from left to right, these inpainted pixels cannot produce valid mappings since they represent content absent from the actual scene geometry. Consequently, the disocclusion mask $M _ { \mathrm { d i s } } ^ { L  R }$ computed from the incomplete warped view $\tilde { I } _ { L }$ is identical to that computed from the fully inpainted $\hat { I } _ { L }$ . This geometric property allows us to bypass the entire left view inpainting step:

$$
\tilde {I} _ {R}, M _ {\mathrm{dis}} ^ {L \rightarrow R} = W _ {L \rightarrow R} (\tilde {I} _ {L}, \mathcal {D} (\tilde {I} _ {L})).\tag{13}
$$

Eliminating left view disparity estimation. Having established that we can work directly with $\tilde { I } _ { L }$ in the previous step, we recognize that disparity estimation from the left view is redundant. During forward warping, each pixel $( x _ { R } , y _ { R } )$ in $I _ { R }$ projects to position $( x _ { L } , y _ { L } )$ where:

$$
x _ {L} = x _ {R} + d _ {R} (x _ {R}, y _ {R}), \quad y _ {L} = y _ {R}.\tag{14}
$$

Treating disparity as a single-channel image, this projection simultaneously transfers both color and disparity to non-

disoccluded pixels:

$$
\begin{array}{r} \tilde {I} _ {L} (x _ {L}, y _ {L}) = I _ {R} (x _ {R}, y _ {R}), \\ \tilde {d} _ {L} (x _ {L}, y _ {L}) = d _ {R} (x _ {R}, y _ {R}), \end{array}\tag{15}
$$

where $( x _ { L } , y _ { L } )$ receives projection from $( x _ { R } , y _ { R } )$ . Note that $\tilde { d } _ { L }$ contains valid disparity values only at nondisoccluded pixels transferred from $I _ { R } ,$ , while disoccluded regions (where $M _ { \mathrm { d i s } } ^ { R  L } = 1 )$ remain undefined. The crucial observation is that the previous step eliminated the need for complete left view information. We only need disparity values for non-disoccluded pixels in $\tilde { I } _ { L }$ , precisely those already transferred from $I _ { R }$ . Disoccluded regions do not participate in backward warping and thus do not require disparity estimation. Therefore, we directly use the transferred disparity $\tilde { d } _ { L }$ instead of re-estimating via $\mathcal { D } ( \tilde { I } _ { L } )$ , eliminating another computational bottleneck:

$$
\tilde {I} _ {R}, M _ {\mathrm{dis}} ^ {L \rightarrow R} = W _ {L \rightarrow R} (\tilde {I} _ {L}, \tilde {d} _ {L}).\tag{16}
$$

Eliminating warping operations. Our final observation is that even the warping operations themselves can be eliminated. After removing both left view inpainting and disparity estimation, the cycle reduces to forward-backward warping: estimate disparity $d _ { R }$ from $I _ { R } ,$ , forward warp to obtain $\tilde { I } _ { L }$ and $\tilde { d } _ { L }$ , then backward warp to compute $\tilde { I } _ { R }$ and $M _ { \mathrm { d i s } } ^ { L  R }$ . We observe that pixels completing the round-trip warping return to their exact original positions:

$$
\begin{array}{r l} & x _ {R} ^ {\prime} = x _ {L} - \tilde {d} _ {L} (x _ {L}, y _ {L}) \\ & \qquad = [ x _ {R} + d _ {R} (x _ {R}, y _ {R}) ] - [ d _ {R} (x _ {R}, y _ {R}) ] \\ & \qquad = x _ {R}. \end{array}\tag{17}
$$

This reveals the fundamental structure: pixels completing the forward-backward cycle remain unchanged, while those failing to complete it form the disocclusion mask. From a self-supervised learning perspective, this structure naturally provides supervision. Pixels completing the cycle correspond to regions where ground truth exists in $I _ { R } ,$ while $\bar { M } _ { \mathrm { d i s } } ^ { L  R }$ identifies regions requiring inpainting. Rather than explicitly performing warping operations to identify these regions, we determine which pixels from $I _ { R }$ fail to complete the round-trip. A pixel $( x _ { R } , y _ { R } )$ is lost under two mutually exclusive conditions:

(a) Boundary violation: The pixel projects outside the image domain during forward warping:

$$
M _ {\mathrm{oob}} ^ {R \rightarrow L} (x _ {R}, y _ {R}) = \mathbb {I} [ x _ {R} + d _ {R} (x _ {R}, y _ {R}) \notin [ 0, W) ],\tag{18}
$$

where $W$ is image width and $\mathbb { I } [ \cdot ]$ is the indicator function.

(b) Depth occlusion: Multiple pixels from $I _ { R }$ map to the same location $( x _ { L } , y _ { L } )$ during forward warping. Only the closest pixel is retained using a depth buffer:

$$
B [ x _ {L}, y _ {L} ] = \max _ {(x _ {R}, y _ {R}) \to (x _ {L}, y _ {L})} d _ {R} (x _ {R}, y _ {R}),\tag{19}
$$

where $( x _ { R } , y _ { R } ) \  \ ( x _ { L } , y _ { L } )$ denotes all pixels from $I _ { R }$ projecting to $( x _ { L } , y _ { L } )$ with $y _ { R } = y _ { L }$ . A pixel is occluded if its disparity is less than this maximum:

$$
M _ {\mathrm{occl}} ^ {R \rightarrow L} (x _ {R}, y _ {R}) = \mathbb {I} [ d _ {R} (x _ {R}, y _ {R}) <   B [ x _ {L}, y _ {L} ] ],\tag{20}
$$

where $( x _ { L } , y _ { L } )$ is the projected location from $( x _ { R } , y _ { R } )$ . The complete lost pixel mask combines both cases:

$$
M _ {\text { lost }} ^ {R \to L} = M _ {\text { oob }} ^ {R \to L} \vee M _ {\text { occl }} ^ {R \to L}.\tag{21}
$$

Both conditions depend only on $( I _ { R } , d _ { R } )$ and require no warping operations, only analytical computation. This remarkable result establishes the GRT: by eliminating all three computational steps (left view inpainting, left view disparity estimation, and warping operations), we arrive at the elegant equivalence $M _ { \mathrm { d i s } } ^ { L  \bar { R } } = \dot { M } _ { \mathrm { l o s t } } ^ { R  L }$ . Note that if we start from the left-right-left cycle instead, we can derive the symmetric result $M _ { \mathrm { d i s } } ^ { \Breve { R }  L } = \dot { M } _ { \mathrm { l o s t } } ^ { L  R }$

## 3.4.2. EQUIVALENCE TO FULL SUPERVISION

Unlike most self-supervised methods (Zhu et al., 2017; Yang et al., 2021) that provide weak, indirect supervision for downstream tasks, GRT fundamentally differs by producing the same mask supervision as paired stereo data under the stated DIBR formulation. By treating any monocular image as a target view $I _ { R } ,$ GRT directly computes $M _ { \mathrm { d i s } } ^ { L  R } \stackrel {  } { = } M _ { \mathrm { l o s t } } ^ { R  L }$ from disparity estimation alone. This is the same mask that emerges when a source view synthesizes $I _ { R }$ via DIBR (Figure 3, top), yet requires neither source view synthesis nor paired stereo data. As shown in the green-highlighted regions of Figure 3, the maskconditioned inpainting process during training matches that at inference, ensuring train-inference consistency.

## 3.4.3. APPLICATION OF GRT

GRT enables self-supervised training data construction by computing inference-identical disocclusion masks from monocular images (Figure 3, bottom). Each mask can be computed once offline and reused during training. Given a target view $I _ { R } ,$ we estimate $d _ { R } = \mathcal { D } ( I _ { R } )$ and compute $M _ { \mathrm { d i s } } ^ { L  R } = M _ { \mathrm { l o s t } } ^ { R  L }$ to construct triplets $( \dot { X } _ { \mathrm { i n p u t } } , M _ { \mathrm { d i s } } ^ { L  R } , I _ { R } )$ where $X _ { \mathrm { i n p u t } } = I _ { R } \odot ( 1 - M _ { \mathrm { d i s } } ^ { L  R } )$ with $I _ { R }$ as ground truth. The training objective is:

$$
\mathcal {L} = \| G (X _ {\mathrm{input}}, M _ {\mathrm{dis}} ^ {L \to R}) - I _ {R} \|.\tag{22}
$$

Training and evaluation datasets. We construct the first comprehensive datasets for stereo inpainting from widelyadopted benchmarks: ImageNet-GRT and Kinetics-GRT from ImageNet (Russakovsky et al., 2015) and Kinetics (Kay et al., 2017) for training. Building on the established equivalence (Section 3.4.2), we create DAVIS-GRT from DAVIS (Perazzi et al., 2016) for evaluation (greenhighlighted regions, Figure 3) to faithfully reflect real-world inference performance (details in Section A2).

![](images/92e10a8d68404c502deb40a38f9202692ac293f5a2a788141e4be55f497fc64b.jpg)  
Figure 3. Equivalence to Full Supervision. Top: inference synthesizes $I _ { R }$ from $I _ { L }$ via DIBR and obtains $M _ { \mathrm { d i s } } ^ { L  R }$ for inpainting. Bottom: GRT treats a monocular frame as $I _ { R }$ and computes the same mask analytically from $I _ { R }$ alone for training and DAVIS-GRT evaluation.

## 3.5. Model Architecture

We adopt LaMa (Suvorov et al., 2022), which leverages Fast Fourier Convolutions, for image stereo inpainting and ProPainter (Zhou et al., 2023) for video stereo inpainting. ProPainter features recurrent flow completion and sparse video transformers to handle temporal consistency. Both models are fine-tuned on ImageNet-GRT and Kinetics-GRT respectively, using a combined loss of L1 and perceptual components.

## 4. Experiments

## 4.1. Experimental Setup

Baselines. For image stereo inpainting, we compare against StereoDiffusion (Wang et al., 2024), ZeroStereo (Wang et al., 2025), and Mono2Stereo (Yu et al., 2025b). For video stereoscopic generation, we evaluate against SVG (Dai et al., 2024) and StereoCrafter (Zhao et al., 2024). We denote our methods as Ours-Image and Ours-Video.

Datasets and Metrics. We conduct evaluations on two complementary benchmarks. DAVIS-GRT isolates the stereo inpainting sub-task under geometrically consistent masks with clean ground truth, while Inria 3DMovie evaluates the full left-to-right stereoscopic generation pipeline on real stereo footage. On DAVIS-GRT, which comprises 50 video clips with 3,455 frames, we report PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018) to assess stereo inpainting quality, along with CLIP Temporal Consistency (CTC), which measures frame-to-frame CLIP feature cosine similarity, to evaluate temporal coherence. Image-based methods are evaluated per-frame on this dataset. Per-frame inference time is reported at 512×512 resolution. On the Inria 3DMovie Dataset (Alahari et al., 2013), which contains 36 stereo video pairs, we evaluate video stereo generation methods using SIoU (Yu et al., 2025b) and MEt3R (Asim et al., 2025) to measure viewing comfort and geometric consistency, reflecting the quality of stereoscopic viewing experience.

## 4.2. Quantitative Results

DAVIS-GRT. As shown in Table 2, both Ours-Image and Ours-Video achieve superior performance across all metrics compared to existing methods, while being significantly faster with inference times of 0.05s and 0.24s per frame, respectively. This improvement stems from our selfsupervised training paradigm that ensures train-inference alignment through GRT-derived data, enabling more accurate geometric understanding without relying on imperfect stereo pair collections.

Inria 3DMovie Dataset. Leveraging improved stereo inpainting capabilities, our methods deliver more comfortable viewing experiences. As shown in Table 3, Ours-Video demonstrates substantially superior geometric consistency and viewing comfort compared to video stereoscopic generation baselines.

Ground Trut  
![](images/c5f3666a8f8dfa39b08194027cf03de845dde81b0f106a8e3e9a611b63f3e4c1.jpg)  
SVG

Table 2. DAVIS-GRT stereo inpainting results. Video methods additionally report temporal consistency (CTC).

<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CTC↑</td><td>Time (s)↓</td></tr><tr><td colspan="6">Image Stereo Inpainting</td></tr><tr><td>Mono2Stereo</td><td>28.20</td><td>0.9250</td><td>0.0485</td><td>-</td><td>2.4</td></tr><tr><td>ZeroStereo</td><td>28.50</td><td>0.9280</td><td>0.0470</td><td>-</td><td>4.1</td></tr><tr><td>StereoDiffusion</td><td>29.77</td><td>0.9360</td><td>0.0411</td><td>-</td><td>10.4</td></tr><tr><td>Ours-Image</td><td>35.52</td><td>0.9800</td><td>0.0129</td><td>-</td><td>0.05</td></tr><tr><td colspan="6">Video Stereo Inpainting</td></tr><tr><td>SVG</td><td>27.33</td><td>0.9052</td><td>0.0552</td><td>0.9721</td><td>3.0</td></tr><tr><td>StereoCrafter</td><td>28.95</td><td>0.9501</td><td>0.0445</td><td>0.9755</td><td>0.6</td></tr><tr><td>Ours-Video</td><td>34.06</td><td>0.9733</td><td>0.0210</td><td>0.9770</td><td>0.24</td></tr></table>

Table 3. Evaluation on Inria 3DMovie.

<table><tr><td>Method</td><td>SIoU↑</td><td>MEt3R↓</td></tr><tr><td>SVG</td><td>0.2160</td><td>0.2245</td></tr><tr><td>StereoCrafter</td><td>0.2201</td><td>0.1961</td></tr><tr><td>Ours-Video</td><td>0.2516</td><td>0.0973</td></tr></table>

## 4.3. Qualitative Results

Visual comparisons in Figure 4 demonstrate the effectiveness of our approach. We compare against StereoDiffusion, the top-performing image baseline, as well as all video stereo inpainting methods. Our GRT-based training enables precise geometric understanding, producing natural textures in disoccluded regions with smooth boundary transitions.

## 4.4. Effectiveness of GRT Training Data

To validate the broad applicability of GRT-derived training data, we fine-tune several foundation models using our datasets. For images, we adapt LaMa (Suvorov et al., 2022) and pretrained Stable Diffusion (Rombach et al., 2022) inpainting models (SD1.5, SD2.1, and SDXL); for videos, we fine-tune ProPainter (Zhou et al., 2023) and StereoCrafter. As shown in Table 4, all models achieve substantial improvements across metrics after fine-tuning on GRT data. Notably, even StereoCrafter, purpose-built for stereoscopic generation, benefits considerably, demonstrating that GRT provides higher-quality supervision than existing stereo pair collection strategies. These results confirm our selfsupervised training paradigm is broadly applicable across diverse architectures.

Warped Left  
StereoDiffusion  
StereoCrafter  
Ours-Video  
Figure 4. Qualitative comparison. Our method produces more natural textures and smoother boundaries.  
Table 4. GRT fine-tuning results across image and video inpainting backbones.

<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CTC↑</td></tr><tr><td colspan="5">Image Stereo Inpainting</td></tr><tr><td>LaMa</td><td>31.75</td><td>0.9660</td><td>0.0239</td><td>-</td></tr><tr><td>LaMa + GRT</td><td>35.52</td><td>0.9800</td><td>0.0129</td><td>-</td></tr><tr><td>SD1.5 Inpainting</td><td>27.38</td><td>0.9152</td><td>0.0982</td><td>-</td></tr><tr><td>SD1.5 Inpainting + GRT</td><td>30.70</td><td>0.9574</td><td>0.0281</td><td>-</td></tr><tr><td>SD2.1 Inpainting</td><td>27.37</td><td>0.9143</td><td>0.0981</td><td>-</td></tr><tr><td>SD2.1 Inpainting + GRT</td><td>28.12</td><td>0.9472</td><td>0.0419</td><td>-</td></tr><tr><td>SDXL Inpainting</td><td>23.40</td><td>0.8842</td><td>0.1426</td><td>-</td></tr><tr><td>SDXL Inpainting + GRT</td><td>29.45</td><td>0.9431</td><td>0.0481</td><td>-</td></tr><tr><td colspan="5">Video Stereo Inpainting</td></tr><tr><td>ProPainter</td><td>31.03</td><td>0.9648</td><td>0.0318</td><td>0.9764</td></tr><tr><td>ProPainter + GRT</td><td>34.06</td><td>0.9733</td><td>0.0210</td><td>0.9770</td></tr><tr><td>StereoCrafter</td><td>28.95</td><td>0.9501</td><td>0.0445</td><td>0.9755</td></tr><tr><td>StereoCrafter + GRT</td><td>30.85</td><td>0.9612</td><td>0.0298</td><td>0.9760</td></tr></table>

## 5. Conclusion

We present a self-supervised framework for training stereo inpainting networks from monocular videos, addressing the data bottleneck in monocular-to-stereo conversion. Our key contribution is the Geometric Reciprocity Theorem (GRT): under the stated DIBR formulation, the disocclusion mask for a target view equals the pixels lost when warping back to source. This enables computing training masks from depth alone, eliminating stereo pairs or synthetic data. We achieve state-of-the-art quality with scalable training from real-world videos.

## Acknowledgements

This work is supported by Hong Kong Research Grants Council – General Research Fund (Grant Nos. 17213825 and 17211024), Hong Kong Innovation and Technology Commission – Innovation and Technology Fund (Project No. ITS/488/24FP), and HKU Seed Fund for PI Research.

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

Alahari, K., Seguin, G., Sivic, J., and Laptev, I. Pose estimation and segmentation of people in 3d movies. In ICCV, 2013.

Asim, M., Wewer, C., Wimmer, T., Schiele, B., and Lenssen, J. E. Met3r: Measuring multi-view consistency in generated images. In CVPR, 2025.

Bochkovskii, A., Delaunoy, A., Germain, H., Santos, M., Zhou, Y., Richter, S. R., and Koltun, V. Depth pro: Sharp monocular metric depth in less than a second. In ICLR, 2025.

Chen, S., Guo, H., Zhu, S., Zhang, F., Huang, Z., Feng, J., and Kang, B. Video depth anything: Consistent depth estimation for super-long videos. In CVPR, 2025.

Dai, P., Tan, F., Xu, Q., Futschik, D., Du, R., Fanello, S., Qi, X., and Zhang, Y. SVG: 3D stereoscopic video generation via denoising frame matrix. arXiv preprint arXiv:2407.00367, 2024.

Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., and Zisserman, A. Temporal cycle-consistency learning. In CVPR, 2019.

Geiger, A., Lenz, P., and Urtasun, R. Are we ready for autonomous driving? the kitti vision benchmark suite. In CVPR, 2012.

Greff, K., Belletti, F., Beyer, L., Doersch, C., Du, Y., Duckworth, D., Fleet, D. J., Gnanapragasam, D., Golemo, F., Herrmann, C., et al. Kubric: A scalable dataset generator. In CVPR, 2022.

Hoffman, J., Tzeng, E., Park, T., Zhu, J.-Y., Isola, P., Saenko, K., Efros, A., and Darrell, T. Cycada: Cycle-consistent adversarial domain adaptation. In ICML, 2018.

Huang, X., Singh, A. K., Dubost, F., Vasconcelos, C. N., Khattar, S., Shi, L., Theobalt, C., Oztireli, C., and Singh, G. ReStereo: Diffusion stereo video generation and restoration. arXiv preprint arXiv:2506.06023, 2025.

Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, P., et al. The Kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R. C., and Schindler, K. Repurposing diffusion-based image generators for monocular depth estimation. In CVPR, 2024.

Kong, L., Xie, S., Hu, H., Ng, L. X., Cottereau, B., and Ooi, W. T. RoboDepth: Robust out-of-distribution depth estimation under corruptions. In NeurIPS, 2023.

Lin, H., Chen, S., Liew, J., Chen, D. Y., Li, Z., Shi, G., Feng, J., and Kang, B. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Min, J., Kim, J., Min, C.-H., Kim, M., Jeon, Y., and Choi, M. DepthFocus: Controllable depth estimation for seethrough scenes. arXiv preprint arXiv:2511.16993, 2025.

Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., and Sorkine-Hornung, A. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, 2016.

Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., and Koltun, V. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE TPAMI, 2020.

Ranftl, R., Bochkovskiy, A., and Koltun, V. Vision transformers for dense prediction. In ICCV, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. ImageNet large scale visual recognition challenge. IJCV, 2015.

Shi, J., Wang, Q., Li, Z., Idoughi, R., and Wonka, P. StereoCrafter-Zero: Zero-shot stereo video generation with noisy restart. arXiv preprint arXiv:2411.14295, 2024.

Shvetsova, N., Bhat, G., Truong, P., Kuehne, H., and Tombari, F. M2SVid: End-to-end inpainting and refinement for monocular-to-stereo video conversion. In International Conference on 3D Vision (3DV), 2026.

Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.

Singh, A. CLDA: Contrastive learning for semi-supervised domain adaptation. In NeurIPS, 2021.

Sun, B., Jin, M., Yin, B., and Hou, Q. Depth anything at any condition. arXiv preprint arXiv:2507.01634, 2025.

Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., and Lempitsky, V. Resolution-robust large mask inpainting with Fourier convolutions. In WACV, 2022.

Wang, L., Frisvad, J. R., Jensen, M. B., and Bigdeli, S. A. Stereodiffusion: Training-free stereo image generation using latent diffusion models. In CVPR, 2024.

Wang, X., Yang, H., Xu, G., Cheng, J., Lin, M., Deng, Y., Zang, J., Chen, Y., and Yang, X. ZeroStereo: Zero-shot stereo matching from single images. In ICCV, 2025.

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image quality assessment: From error visibility to structural similarity. IEEE TIP, 2004.

Xie, J., Girshick, R., and Farhadi, A. Deep3d: Fully automatic 2d-to-3d video conversion with deep convolutional neural networks. In ECCV, 2016.

Yang, C., Lamdouar, H., Lu, E., Zisserman, A., and Xie, W. Self-supervised video object segmentation by motion grouping. In ICCV, 2021.

Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., and Zhao, H. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024.

Yu, M., Hu, W., Xing, J., and Shan, Y. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, 2025a.

Yu, S., Chen, Y., Qi, Z., Xie, Z., Wang, Y., Wang, L., Shan, Y., and Lu, H. Mono2stereo: A benchmark and empirical study for stereo conversion. In CVPR, 2025b.

Yuan, Y., Liu, S., Zhang, J., Zhang, Y., Dong, C., and Lin, L. Unsupervised image super-resolution using cycle-incycle generative adversarial networks. In CVPRW, 2018.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Zhao, S., Hu, W., Cun, X., Zhang, Y., Li, X., Kong, Z., Gao, X., Niu, M., and Shan, Y. StereoCrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3D from monocular videos. arXiv preprint arXiv:2409.07447, 2024.

Zheng, Y., Zhong, C., Li, P., Gao, H.-a., Zheng, Y., Jin, B., Wang, L., Zhao, H., Zhou, G., Zhang, Q., et al. STEPS: Joint self-supervised nighttime image enhancement and depth estimation. In ICRA, 2023.

Zhou, S., Li, C., Chan, K. C., and Loy, C. C. Propainter: Improving propagation and transformer for video inpainting. In ICCV, 2023.

Zhou, T., Krahenbuhl, P., Aubry, M., Huang, Q., and Efros, A. A. Learning dense correspondence via 3d-guided cycle consistency. In CVPR, 2016.

Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV, 2017.

## A1. Code and Data Release

The project repository is https://github.com/ Visual-AI/GRT. We will release source code, precomputed GRT masks, and trained weights under the Apache 2.0 license.

## A2. Dataset Details

We construct three datasets by applying the Geometric Reciprocity Theorem (GRT) (Section 3.4) to widely-adopted vision benchmarks. For each image or video frame, we treat it as a right (target) view, estimate its monocular depth, and compute a binary disocclusion mask $M _ { \mathrm { d i s } } ^ { L  R }$ following the GRT procedure. This yields self-supervised training samples and test-aligned evaluation samples for stereo inpainting, where each RGB image or frame is paired with its corresponding GRT-derived disocclusion mask.

Depth estimation and GRT mask generation. We use Depth Anything V2-Large (Yang et al., 2024) to estimate monocular depth for each image or frame, following previous stereoscopic video generation literature. The predicted relative (inverse) depth is converted to a disparity map and linearly rescaled to [0, αW ], where W is the image width and α = 0.1 for training and α = 0.06 for evaluation. We then apply the GRT procedure (Section 3.4) to compute $M _ { \mathrm { d i s } } ^ { L  \bar { R } }$ by identifying lost pixels through boundary violations and depth occlusions. On average, disoccluded regions comprise approximately 10% of pixels for training data. Note that our framework flexibly accommodates alternative depth estimators, such as Video Depth Anything (Chen et al., 2025) for temporally consistent depth, Depth Anything V3 (Lin et al., 2025) for metric depth, or DepthFocus (Min et al., 2025) for multi-layer depth to handle transparent surfaces. We adopt Depth Anything V2 to follow prior works and ensure fair comparisons.

ImageNet-GRT. ImageNet-GRT uses the complete ImageNet-1K training split (Russakovsky et al., 2015), containing ∼1.28M images across 1,000 object categories. We utilize all images without filtering and disregard class labels, treating ImageNet purely as a diverse source of natural imagery.

Kinetics-GRT. Kinetics-GRT uses the Kinetics-400 training split (Kay et al., 2017), comprising ∼240K videos across 400 human action categories with rich temporal dynamics and camera motion. We decode all frames at their original resolution and frame rate while preserving temporal order.

DAVIS-GRT. DAVIS-GRT uses all 50 videos (3,455 frames) from DAVIS 2016 (Perazzi et al., 2016), a video segmentation benchmark known for high-quality annotations and challenging scenarios. DAVIS-GRT serves exclusively for testing; no DAVIS frames are used during training or model selection, ensuring unbiased assessment of generalization.

## A3. Pseudocode

We provide pseudocode in Figure A1 implementing the Geometric Reciprocity Theorem (GRT) presented in Section 3.4. Given a disparity map, the function computes the disocclusion mask by identifying pixels that would be lost during warping due to boundary violations or depth occlusions.

## A4. Model Architecture and Training

Our model builds upon the LaMa architecture (Suvorov et al., 2022) with Fast Fourier Convolutions (FFC) for image stereo inpainting, and ProPainter (Zhou et al., 2023) for video stereo inpainting.

## A4.1. Image Stereo Inpainting

Architecture. We use the Big LaMa-Fourier generator (27M parameters): 3 downsampling blocks, 9 FFC-based residual blocks, and 3 upsampling blocks. The network takes 4-channel input (masked RGB + mask) and outputs 3-channel RGB. FFC blocks provide image-wide receptive fields by processing features in both spatial (local) and frequency (global) domains.

Training. We fine-tune the model on ImageNet-GRT with realistic disocclusion masks generated via GRT for 30 epochs at 256 × 256 resolution. We use combined L1 and LPIPS loss with Adam optimizer (learning rate 10<sup>−4</sup>), batch size 32, and cosine annealing scheduler on two NVIDIA V100 GPUs.

## A4.2. Video Stereo Inpainting

Architecture. We adopt ProPainter which comprises three key components: (1) Recurrent Flow Completion (RFC) network that completes corrupted optical flows using deformable alignment with 8× downsampled features; (2) Dual-domain propagation combining global image propagation (with flow consistency check) and local feature propagation (flow-guided deformable alignment); (3) Mask-guided sparse video Transformer with 8 blocks, window size $5 \times 9 ,$ and extended size equal to half of the window size. The Transformer applies sparse attention only to query windows intersecting mask regions and uses temporal stride of 2 for key/value space to reduce redundancy.

Training. We fine-tune from pretrained ProPainter weights on Kinetics-GRT for 10,000 steps at 432 × 240 resolution. We use combined L1 reconstruction loss and T-PatchGAN adversarial loss (weight 0.01) with Adam optimizer (learning rate $1 0 ^ { - 4 } )$ and batch size 8. The model processes local sequences of length 10 during training and length 20 during inference on two NVIDIA V100 GPUs.

```python
def compute_grt_mask(disparity, direction='R2L'):
    """
    Compute GRT-based disocclusion mask via geometric warping.

    Args:
    disparity: [B,H,W] disparity map (positive values)
    direction: 'R2L' for right-to-left warp (finds L->R disocclusions)
    'L2R' for left-to-right warp (finds R->L disocclusions)

    Returns:
    mask: [B,H,W] binary mask (True = disoccluded pixel)

    """
    B, H, W = disparity.shape

    # Compute target positions after warping
    x = torch.arange(W)[None, None, :]  # [1,1,W]
    if direction == 'R2L':
    x_target = round(x + disparity)  # Right-to-left warp
    else:  # L2R
    x_target = round(x - disparity)  # Left-to-right warp

    # Mask 1: Out-of-bounds pixels (boundary violations)
    mask_oob = (x_target < 0) | (x_target >= W)

    # Mask 2: Occluded pixels (depth ordering conflicts)
    # When multiple pixels map to same target, keep only the closest (max disparity)
    valid_idx = torch.where(~mask_oob)
    x_tgt = x_target[valid_idx]
    d_src = disparity[valid_idx]

    # Find winning disparity at each target position
    d_winner = scatter_max(d_src, index=x_tgt, dim_size=W)  # [B,H,W]

    # Mark projected pixels that lost the depth competition
    mask_occ = torch.zeros_like(mask_oob)
    is_occluded = (d_src < d_winner[valid_idx])
    mask_occ[valid_idx][is_occluded] = True

    # Combine both invalidation conditions
    return mask_oob | mask_occ
```  
Figure A1. Pseudocode implementing the Geometric Reciprocity Theorem (GRT). Given a disparity map, the function computes the disocclusion mask by identifying pixels lost during warping due to boundary violations (mask oob) or depth occlusions (mask occ).

## A5. Additional Analyses

Naive cycle consistency. GRT precomputes masks offline and reduces online optimization to standard inpainting, while naive cycle consistency requires multiple sequential model inferences and gradient approximations through warping. A controlled LaMa comparison is summarized in Table A1.

Table A1. Efficiency and quality comparison with naive cycle consistency.

<table><tr><td>Method</td><td>Steps/s↑</td><td>Memory↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Naive Cycle Consistency</td><td>0.6</td><td>19 GB</td><td>32.14</td><td>0.9663</td><td>0.0271</td></tr><tr><td>GRT</td><td>1.9</td><td>11 GB</td><td>33.10</td><td>0.9718</td><td>0.0232</td></tr></table>

Mask agreement. We compare analytically computed GRT masks with cycle-consistency masks under different depth and warping settings. The high agreement in Table A2 indicates that the derived masks capture the same disocclusion structure in practice.

Table A2. Pixel-level mask agreement with GRT masks.

<table><tr><td>Setting</td><td>Agreement (%)↑</td></tr><tr><td>Cycle consistency, depth-consistent</td><td>100.0</td></tr><tr><td>Cycle consistency, independently estimated</td><td>99.1</td></tr><tr><td>Bilinear warping vs. nearest-neighbor GRT</td><td>99.5</td></tr></table>

Depth estimator robustness. GRT is depth-estimatoragnostic and only requires a disparity map as input. Table A3 shows that performance degrades gradually with weaker depth backbones but remains close.

Table A3. Depth estimator sensitivity on Inria 3DMovie.

<table><tr><td>Depth Estimator</td><td>SIoU↑</td><td>MEt3R↓</td></tr><tr><td>Depth Anything V2-Large</td><td>0.2516</td><td>0.0973</td></tr><tr><td>Depth Anything V2-Base</td><td>0.2489</td><td>0.1012</td></tr><tr><td>Depth Anything V1-Small</td><td>0.2451</td><td>0.1058</td></tr></table>

GRT mask components. The boundary-violation and depth-occlusion terms are jointly required to identify geometrically lost pixels. Ablating either component degrades downstream inpainting quality.

Data scaling. Because GRT requires no paired stereo capture, it can benefit from larger monocular video corpora. Table A5 shows monotonic gains as the training set grows.

## A6. Limitations

GRT inherits the assumptions of DIBR-based stereoscopic generation. It relies on rectified stereo geometry and the quality of the input disparity map; transparent or reflective surfaces, inaccurate depth discontinuities, and non-Lambertian effects can therefore lead to incorrect masks or visually implausible inpainting. The nearest-neighbor formulation gives exact mask consistency under the theorem statement, while bilinear interpolation introduces a small approximation gap as discussed in Section A9.

Table A4. Ablation of GRT mask components on LaMa trained with ImageNet-GRT.

<table><tr><td>Mask Configuration</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Boundary violation only</td><td>32.89</td><td>0.9694</td><td>0.0200</td></tr><tr><td>Depth occlusion only</td><td>32.54</td><td>0.9676</td><td>0.0216</td></tr><tr><td>Full GRT mask</td><td>35.52</td><td>0.9800</td><td>0.0129</td></tr></table>

Table A5. Dataset scale ablation for ProPainter trained on GRT data.

<table><tr><td>Training Data Scale</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>CTC↑</td></tr><tr><td>0% (pretrained)</td><td>31.03</td><td>0.9648</td><td>0.0318</td><td>0.9764</td></tr><tr><td>10% (~24K videos)</td><td>31.88</td><td>0.9672</td><td>0.0274</td><td>0.9765</td></tr><tr><td>50% (~120K videos)</td><td>33.45</td><td>0.9710</td><td>0.0241</td><td>0.9768</td></tr><tr><td>100% (~240K videos)</td><td>34.06</td><td>0.9733</td><td>0.0210</td><td>0.9770</td></tr><tr><td>Kinetics-700 (~650K videos)</td><td>34.83</td><td>0.9751</td><td>0.0188</td><td>0.9774</td></tr></table>

## A7. Discussion of General Inpainting Models

General-purpose image inpainting models are designed to fill large contiguous missing regions by hallucinating plausible content from surrounding context. While these models are powerful, stereo disocclusions present a fundamentally different mask pattern. Disocclusion masks are thin, elongated, and scattered along depth discontinuities at object boundaries, with significantly smaller total area compared to typical inpainting masks. More critically, disocclusions reveal previously hidden background content that is often visually and semantically distinct from the adjacent occluding foreground. For example, a disocclusion behind a person might expose a wall or furniture with no visual similarity to the person’s appearance. This domain gap in both mask geometry and content semantics causes general inpainting models to produce inferior results on stereo disocclusions. As we demonstrate in Figure A2, while these models succeed on large contiguous masks, they fail on the thin scattered masks characteristic of stereo disocclusions.

## A8. Training Data Challenges

Constructing high-quality training data remains a significant bottleneck for stereo inpainting. Methods relying on real stereo pairs, such as StereoCrafter (Zhao et al., 2024), face a fundamental paradox: they rely on stereo matching algorithms to extract pixel correspondences from stereo pairs and subsequently derive disocclusion masks, yet stereo matching itself is prone to errors and particularly unreliable in disoccluded regions. As shown in Fig. A3, stereo matching produces distorted masks with alignment errors.

General Inpainting

Stereo Inpainting

GRT (Ours)  
Masked Input  
Result  
Masked Input  
Result  
![](images/94a6e608835ea68af8e87ef69f1757d2de34de326827d9ead2aec28ad88288bc.jpg)

![](images/ab9140dc9f242e4fed24e85ad772db078632779ee71fb3e104dcef263386f060.jpg)  
Figure A2. Performance of SDXL Inpainting on different mask patterns. The model handles large contiguous masks well (General Inpainting) but struggles with thin scattered disocclusion masks along object boundaries (Stereo Inpainting).

Original  
Mask  
Masked Input  
![](images/075383b9a04702206d0ba745357cd076757b18b31048f1c3745f78aa489d82bf.jpg)

Mask  
Masked Input  
![](images/bbedfc42f345ba8267d2d7df054b6a0e16440fd782ff5c35e1ec7c7bedef6e05.jpg)

![](images/605506fa078b5788cd1dda2fb8562efa951d36d071406002933b5ae5f5e74dba.jpg)  
Figure A3. Stereo matching methods produce distorted disocclusion masks and misaligned inpainting triplets. Our GRT approach generates geometrically accurate masks at test-time.

StereoCrafter confirms this limitation, acknowledging the prevalence of large stereo matching errors that force them to employ complex manual alignment and aggressive filtering strategies (e.g., discarding samples where warping PSNR < 25 dB), resulting in substantial data waste. Alternatively, approaches utilizing synthetic data (Huang et al., 2025) provide precise geometry but suffer from inevitable Sim2Real domain gaps. Our GRT-based approach resolves these dilemmas by deriving geometrically consistent masks directly from monocular data, avoiding both error-prone matching and synthetic domain shifts.

## A9. Extension to Soft Interpolation Warping

The GRT proof in Section 3.4.1 assumes nearest-neighbor warping, where each pixel (x<sub>R</sub>, y<sub>R</sub>) maps to discrete coordinates $( x _ { L } , y _ { L } ) = ( { \mathrm { r o u n d } } ( x _ { R } + d _ { R } ( x _ { R } , y _ { R } ) ) , y _ { R } )$ . In practice, bilinear warping is sometimes preferred for smoother synthesis. We show that GRT extends to this soft interpolation setting by reformulating the problem at the renderer level rather than the pixel level, though this introduces a subtle train-test consistency trade-off.

Renderer-based formulation. Under bilinear warping, we treat each pixel $( x _ { R } , y _ { R } )$ in the right view $I _ { R }$ as a renderer that carries appearance and geometry information. Each renderer projects to continuous coordinates:

$$
x _ {L} ^ {\prime} = x _ {R} + d _ {R} (x _ {R}, y _ {R}), \quad y _ {L} ^ {\prime} = y _ {R},\tag{23}
$$

and distributes its contribution to a $2 \times 2$ neighborhood in the left view via bilinear weights $w _ { x _ { R } , y _ { R } } ( x _ { L } , y _ { L } )$ . This splatting process synthesizes the left view as a weighted blend of renderer contributions.

Forward warping with occlusion handling. For each left-view pixel $( x _ { L } , y _ { L } )$ ), let $S ( x _ { L } , y _ { L } )$ denote all renderers whose bilinear kernels overlap it. To handle occlusions, we maintain a depth buffer that tracks the maximum disparity among contributing renderers:

$$
d _ {L} ^ {\max} (x _ {L}, y _ {L}) = \max _ {(x _ {R}, y _ {R}) \in S (x _ {L}, y _ {L})} d _ {R} (x _ {R}, y _ {R}).\tag{24}
$$

Only renderers at maximum disparity are retained, as nearer objects occlude farther ones. Let $S _ { \operatorname* { m a x } } ( x _ { L } , y _ { L } )$ denote renderers at maximum disparity:

$$
\begin{array}{c} S _ {\max} (x _ {L}, y _ {L}) = \{(x _ {R}, y _ {R}) \in S (x _ {L}, y _ {L}): \\ d _ {R} (x _ {R}, y _ {R}) = d _ {L} ^ {\max} (x _ {L}, y _ {L}) \}. \end{array}\tag{25}
$$

The warped left view is synthesized as:

$$
\tilde {I} _ {L} (x _ {L}, y _ {L}) = \frac {\sum_ {(x _ {R} , y _ {R}) \in S _ {\max}} w _ {x _ {R} , y _ {R}} \cdot I _ {R} (x _ {R} , y _ {R})}{\sum_ {(x _ {R} , y _ {R}) \in S _ {\max}} w _ {x _ {R} , y _ {R}}},\tag{26}
$$

where we abbreviate $S _ { \operatorname* { m a x } } ( x _ { L } , y _ { L } )$ as $S _ { \mathrm { m a x } }$ for notational simplicity.

Lost renderer criteria. A renderer $( x _ { R } , y _ { R } )$ is lost if it fails to contribute to any left-view pixel. This occurs under two conditions. First, boundary violation occurs when the renderer’s bilinear kernel support lies entirely outside the valid image domain:

$$
M _ {\mathrm{oob}} ^ {R \rightarrow L} (x _ {R}, y _ {R}) = \mathbb {I} \left[\left\lfloor x _ {L} ^ {\prime} \right\rfloor , \left\lceil x _ {L} ^ {\prime} \right\rceil \notin [ - 1, W ] \right],\tag{27}
$$

where the range [−1, W ] ensures at least one neighbor in [0, W ) has non-zero weight. Second, depth occlusion occurs when the renderer is occluded at all left-view locations in its support. Let $S ^ { * } ( x _ { R } , y _ { R } )$ denote valid left-view pixels overlapping $( x _ { R } , y _ { R } ) \mathrm { { ^ { \circ } s } }$ bilinear kernel. The occlusion mask is:

$$
\begin{array}{c} M _ {\mathrm{occl}} ^ {R \to L} (x _ {R}, y _ {R}) = \mathbb {I} \big [ d _ {R} (x _ {R}, y _ {R}) <   d _ {L} ^ {\max} (x _ {L}, y _ {L}), \\ \forall (x _ {L}, y _ {L}) \in S ^ {*} (x _ {R}, y _ {R}) \big ]. \end{array}\tag{28}
$$

The complete lost mask combines both conditions:

$$
M _ {\mathrm{lost}} ^ {R \rightarrow L} = M _ {\mathrm{oob}} ^ {R \rightarrow L} \vee M _ {\mathrm{occl}} ^ {R \rightarrow L}.\tag{29}
$$

GRT validity at the renderer level. The Geometric Reciprocity relation $M _ { \mathrm { d i s } } ^ { L  R } = M _ { \mathrm { l o s t } } ^ { R  L }$ continues to hold when formulated at the renderer level. The three-step simplification in Section 3.4.1 remains geometrically valid: disoccluded regions lack scene correspondence, each renderer co-transfers disparity with its appearance, and renderers completing the round-trip return to their original positions. Therefore, lost renderers in the forward pass correspond exactly to disoccluded regions in the backward pass.

Train-test consistency. While GRT holds mathematically at the renderer level, bilinear warping introduces a subtle inconsistency. During training, $\dot { M } _ { \mathrm { l o s t } } ^ { R  L }$ is computed from discrete renderers in the original image $I _ { R }$ . During testing, the synthesized view $\hat { I } _ { L }$ contains blended pixel values, and backward warping treats each blended pixel as a single renderer rather than the multiple discrete renderers that created it. This approximation causes slight differences in disocclusion masks, though the effect is minor for typical stereo baseline ranges.

Implementation. We adopt nearest-neighbor warping to maintain consistency with the theorem statement and to improve computational efficiency. Both variants produce visually similar results, while nearest-neighbor warping keeps training masks aligned with the stereoscopic generation pipeline.