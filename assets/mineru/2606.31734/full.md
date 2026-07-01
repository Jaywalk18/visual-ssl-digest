# <sup>MemLearner</sup>: Learning to Query Context Memory for Video World Models

Jiwen Yu<sup>1‡</sup>, Jianxiong Gao<sup>2‡</sup>, Jianhong Bai<sup>3‡</sup>, Yiran Qin<sup>1</sup>, Kaiyi Huang<sup>1‡</sup>, Quande Liu<sup>4</sup>, Xintao Wang<sup>4†</sup>, Pengfei Wan<sup>4</sup>, Kun Gai<sup>4</sup>, and Xihui Liu<sup>1†</sup>

<sup>1</sup> The University of Hong Kong 2 Fudan University 3 Zhejiang University 4 Kuaishou Technology

![](images/94428925599b0241143099b9064c2623013096ce630a5931c55616adbb685ddf.jpg)  
Fig. 1: Teaser Demonstration. We propose a novel memory mechanism for video world models through learning-based adaptive context querying. Compared to prior rule-based context retrieval methods [36, 58, 66], our approach handles scenes with occlusions and dynamic objects. This figure highlights dynamic objects, representative occluders, video generation trajectory, and key frames.

Abstract. Video World Models are interactive video generation models that predict future world states based on user actions and history video frames. A critical challenge in video world models is the lack of memory, causing inconsistent generated scenes over extended durations. Previous methods explored rule-based context frame retrieval as memory, but they fail to generalize in scenarios with scene occlusions and dynamic objects. We propose <sup>MemLearner</sup>, a learning-based adaptive context query method using query tokens to bridge context and predicted tokens. By leveraging the video generation model itself for context querying, <sup>Mem-</sup> <sup>Learner</sup> exploits pre-trained visual priors without training additional modules from scratch, and incorporates eficient strategies for training and inference. We collect a dataset of long videos with scene occlusions and dynamic objects, paired with camera pose annotations, and propose a multi-dataset training strategy leveraging both annotated rendered and unannotated real-world videos. Extensive experiments demonstrate that <sup>MemLearner</sup> significantly outperforms prior video world models in terms of scene consistency and memory, particularly under challenging occlusion and dynamic scenarios.

Keywords: Video World Model · Interactive Video Generation · Video Difusion Model

## 1 Introduction

World Models [22] understand and simulate world dynamics by taking historical world states and interactive actions as input to predict new states. World states can be represented as language [8], latent representations [4, 22, 72], 3D/4D [34, 64], or videos [6,43]. Among these, Video World Models are particularly promising [60,67,68] as videos photorealistically capture real-world dynamics, and abundant internet video data ofers significant scaling potential. Despite significant progress made in generating short video clips, Video World Models still face significant challenges of scene consistency over extended durations, caused by insuficient memory mechanisms.

The memory issue in Video World Models arises when later generated scenes become inconsistent with earlier ones due to limited context windows. Existing methods address this challenge by diferent memory representations: 3D reconstruction [40,42,47,56,65,70], compressed feature representations [29,45,71,73], and retrieving key context frames [12,36,58,66]. Context retrieval is particularly promising as it eliminates additional costs and potential errors caused by 3D reconstruction or history compression.

However, existing context retrieval methods are rule-based, relying on FOV overlap [58, 66] or point cloud estimation and surfel matching [36], facing several fundamental limitations in complex scenes with occlusions and dynamic objects. For example, FOV-based context retrieval [66] cannot account for occluding walls between camera views, point-cloud-based retrieval methods cannot accurately reconstruct moving objects. Moreover, these hard-coded retrieval rules fail to take generalized and dynamic environments with state changes into consideration. These limitations motivate a paradigm shift: rather than using hand-crafted rules to retrieve context, we propose a new memory mechanism that enables the network to learn to adaptively query information from historical frames through end-to-end training.

We formulate history-conditioned video generation as predicting future frames based on historical context frames. To design a learning-based context query method, we introduce query tokens (Q tokens) as an information bridge between context tokens (C tokens) and predicted tokens (P tokens) (Fig. 2 (a)): Q tokens attend to C tokens to adaptively extract context information, while P tokens attend to Q tokens as the generation condition. A key design choice is to leverage the video generation model itself for context querying, rather than introducing a separate module. Specifically, we feed all C, Q, and P tokens together into the video generation model (Fig. 2 (c), detailed in Sec. 3.2), exploiting the model’s pre-trained visual priors without additional scratch-trained modules. To further reduce the computational cost of context querying over long video sequences, we propose two eficient strategies for training and inference (Sec. 3.3).

![](images/3d7d522300c573ebb1f59e753251d78fa7ce5490dc9a0c704ba786e0cc5f3414.jpg)  
Fig. 2: Architecture clarification. (a) Interaction mechanism among C, Q, and P tokens; (b)&(c) Two designs for context querying, where the alternative design in (b) fails in experiments (Sec. 5.4), and the adopted design in (c) leverages the prior knowledge of the video generation model itself and performs efectively.

Training learning-based context querying requires a long video dataset with occlusion, dynamic objects, camera pose annotations, and suficient diversity. However, no existing datasets [39,53,66,74] fully meet these criteria. Real-world videos (e.g., YouTube) provide diverse and dynamic content but lack precise camera pose annotations, while rendered videos (e.g., Unreal Engine) ofer accurate poses but sufer from limited visual realism and diversity. To address this, we make two contributions. First, we collect a rendered dataset with customized occlusions and dynamic objects (Sec. 4.1). Second, we propose a multi-dataset training strategy that assigns a dedicated camera encoder to each dataset type, where unannotated data is fed with zero camera parameters. By isolating distinct annotation qualities into separate encoders, the model simultaneously leverages the strengths of diverse data sources without mutual interference (Sec. 4.2).

Our contributions can be summarized as follows:

– We propose <sup>MemLearner</sup>, a learning-based adaptive context query method that utilizes query tokens to extract valid memory from context tokens, enabling memory-augmented Video World Models with improved consistency and generalization.

We collect a customized video world model dataset based on Unreal Engine, incorporating occlusion and dynamic objects, and propose a multi-dataset training strategy to leverage the advantages of both rendered and real videos.

– Extensive experiments demonstrate that <sup>MemLearner</sup> significantly outperforms previous Video World Models in terms of scene consistency and memory, particularly under occlusion and dynamic object scenarios.

## 2 Related Work

## 2.1 Video World Models

World Model. The World Model [22] refers to a system that takes historical world state and current action as input to predict future world state. Its primary purpose is to understand and simulate the evolving process of the world. The world state can be represented in various forms, such as language [8], semantic representations [4, 72], 3D/4D [34, 64], and videos [6, 43]. Recently, due to video photorealism, large data volumes, and breakthroughs in video generation models, Video World Models are considered a promising path to World Models [60,67,68].

Video Generation Models. Mainstream video generation models now use the Difusion Transformer (DiT) [43,44] architecture, enabling highly realistic video creation [7, 17, 31, 33, 43, 48, 52, 62]. Other architectures include next-token prediction [14, 32, 54, 59] and hybrid approaches [11, 18, 37], but fall short of DiT in generation quality. Interactive control in video generation includes camera pose [5,23,55], trajectory [19,55], and action control [51,69,75], with applications in film production, game video control, and robotic simulation. Streaming video generation conditions on previously generated frames to create new content, with both frame-by-frame [11,49,71] and chunk-by-chunk methods [1,66,69] primarily using DiT. Long video generation sufers from error accumulation, which several methods address [13, 26, 38, 41, 61]. Since our work focuses on learnable memory rather than ultra-long generation, we do not discuss error accumulation.

## 2.2 Memory Mechanism for Video World Models

Memory capability in video world models refers to generating consistent long videos, especially when revisiting scenes or objects. It is a fundamental prerequisite for planning, interaction and state modeling, as these core capabilities rely on retaining past visual memories. However, many methods struggle with this [16, 28, 49, 51], due to limited context windows that cannot retain suficient historical information. To address this, additional conditional information is necessary. Existing approaches fall into three paradigms: (1) 3D as memory [40, 42, 47, 56, 65, 70]: reconstructing 3D representations from historical frames and rendering new initial frames as conditions; (2) Feature as memory [29, 73]: extracting semantic features from historical frames or maintaining learnable features injected into the generation model; (3) Context as memory [12, 25, 36, 45, 50, 58, 66, 71]: directly using historical frames as conditions. Among these, context memory is the most straightforward. However, incorporating all historical contexts introduces prohibitive computational overhead.

Some works propose context retrieval to select historical contexts relevant to current generation. Conditioning on only retrieved frames significantly reduces computational overhead. Existing methods typically rely on rule-based retrieval [36,58,66], which lack generalization across videos and scenes. We propose a learning-based context query method enabling the model to learn to query useful historical information, thereby achieving learnable memory.

![](images/5947ddde614592b3ba4fb2c6489a87708caf4096dc9e63d28b9962a0eaeeb0ee.jpg)  
Fig. 3: Model Architecture. Our video generation model adopts a Difusion Transformer. We concatenate C, Q, P tokens for context-conditioned long video generation. An optional camera encoder supports interactive control.

## 2.3 Learnable Query Tokens in Various Domains

Learnable query tokens have been widely adopted for compressing visual context across various domains: In multimodal language models, Perceiver Resampler [2] and Q-Former [35] use learnable queries to aggregate visual features for language model consumption; In video understanding, adaptive frame selection methods [3, 9, 57] employ learnable strategies to identify informative frames for recognition and reasoning tasks. Our work addresses a fundamentally diferent problem: memory issue in video world models, where query tokens serve as an information bridge between context tokens and predicted tokens, adaptively extracting fine-grained, token-level information across multiple historical frames to support consistent video generation under occlusion and dynamic scenarios.

## <sub>3</sub> MemLearner

## 3.1 Preliminaries

Our approach is built upon a latent video difusion model with a causal 3D VAE [30] and Difusion Transformer (DiT) [44]. As shown in Fig. 3, each DiT block sequentially contains spatial (2D) attention, spatial-temporal (3D) attention, cross-attention, and feed-forward networks (FFN). The 3D VAE encoder compresses a video frame sequence x temporally and spatially into latent representations ${ \bf z } = E ( { \bf x } )$ , and the DiT is trained to predict noise scaled and added to z via a standard difusion objective. We integrate camera control [5, 55] by injecting camera poses $\mathbf { c a m } = [ \bar { R } , t ] \in \mathbb { R } ^ { f \times ( 3 \times 4 ) }$ into the model via a one-layer MLP encoder $\mathcal { E } _ { c } ( \cdot )$ , added to features between the 2D and 3D attention layers. Note that camera control only guides the trajectory of generated videos; our learning-based context query method does not rely on camera poses, as validated in Sec. 5.5. For long video generation, we follow a chunk-by-chunk autoregressive paradigm: historical context frame latents h are concatenated with the to-begenerated latents z along the frame dimension and fed jointly into the model, with the difusion loss applied only to z. This preserves the model’s generative priors without architectural modifications.

## 3.2 Learning to Query Context

Introducing Adaptive Query Tokens. Our key insight is that diferent predicted frames require diferent guidance information from context frames, and even within the generation of a single frame, diferent denoising stages of the diffusion process need to emphasize diferent aspects of historical information. We verify this via attention similarity analysis in Appendix C.2: query tokens exhibit markedly diferent attention distributions across predicted frames and difusion timesteps, with early timesteps attending more broadly to context while later timesteps focus on fine-grained local correspondences. Naive history compression or rule-based keyframe retrieval fail to tackle complex scenarios such as occlusion and dynamic objects. To enable adaptive memory, we introduce learnable query tokens Q that bridge context and generation. These query tokens dynamically extract relevant information from context tokens C and guide the generation of predicted tokens P. Importantly, this querying mechanism can be learned end-to-end via indirect supervision from the difusion loss on P alone, without explicit supervision for a dedicated querying module.

Architecture Design Insights. The intuitive design introduces an additional context query module, as in Fig. 2 (b). However, our experiments show that when jointly training the architecture, the from-scratch context query module fails to learn useful information. Attention similarity analysis (Appendix C.2) confirms that this module produces near-zero similarity between query and context tokens, indicating a failure to establish meaningful context modeling; this in turn hinders gradient propagation to the video generation model, which consequently ignores the module’s output and degrades to a text-to-video model (Sec. 5.4). As in Fig. 2 (c), our design avoids a separate from-scratch module and instead leverages the pre-trained video generation model itself for context querying. This ofers clear advantages: (1) it exploits the model’s prior knowledge, reducing data and compute requirements; (2) end-to-end training naturally learns context query capability without requiring additional input-output design or supervision losses for a separate module.

Architecture Details. Our work is based on a latent video difusion model. We denote the video latent tokens as $Z = \{ \mathbf { C } , \mathbf { Q } , \mathbf { P } \}$ , where C, Q and P denote context, query, and predicted tokens, respectively. The difusion model input is the noisy token at timestep $t ,$ written as $Z _ { t } = \{ \mathbf { C } , \mathbf { Q } , \mathbf { P } _ { t } \}$ , where the context tokens and query tokens remain unperturbed, while the predicted tokens are noised only at the input (and not at any subsequent layer) via randomly sampled Gaussian noise to obtain $\mathbf { P } _ { t }$ , with the scaled and added noise being $\epsilon ^ { P } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ The training loss for the entire architecture is:

$$
\mathcal {L} (\theta) = \mathbb {E} [ | | \epsilon_ {\theta} (Z _ {t}, \mathbf {c a m}, \mathbf {p}, t) | | - \epsilon^ {P} ],\tag{1}
$$

![](images/88f38c833f7a31082b5ef0b23b168cfb2e05b51b61d9c27395e50918fe8043cc.jpg)  
Fig. 4: Eficiency Strategies. (a) Strategy#1: Context querying in shallow Query Layers with C, Q, P tokens; deep Generative Layers use only Q, P tokens. (b) Strategy#2: Remove unnecessary attention computation for improved eficiency.

where θ denotes all learnable parameters. Note that supervision is applied only to the noise predicted on predicted tokens. We concatenate C, Q and P along the frame dimension and input them into the video DiT, where diferent tokens interact with each other in the 3D attention. For a 3D attention layer with input $F _ { i n }$ and output $F _ { o u t }$ , we have $F _ { i n } = \{ { \bf C } , { \bf Q } , { \bf P } \}$ and $F _ { o u t } = \{ \mathbf { C } _ { o u t } , \mathbf { Q } _ { o u t } , \mathbf { P } _ { o u t } \}$ For brevity, we use $\mathbf { C } , \mathbf { Q } .$ and P to denote the features corresponding to diferent tokens. We define the linear computations in the attention block as $q ( \cdot ) , k ( \cdot )$ $v ( \cdot )$ , and $o ( \cdot )$ , which correspond to the operations for query, key, value, and output. A standard 3D attention computation is:

$$
\begin{array}{r} F _ {o u t} = F _ {i n} + o (\mathbf {s m} (q (F _ {i n}) k (F _ {i n}) ^ {\top}) v (F _ {i n})), \\ = F _ {i n} + g (F _ {i n}, F _ {i n}, F _ {i n}), \end{array}\tag{2}
$$

where $\mathbf { s m } ( \cdot )$ denotes the softmax operation, and $g ( \cdot , \cdot , \cdot )$ is a shorthand notation indicating the inputs for query, key, and value operations, respectively. Considering that the context frame length may be very large, this computation in Eq. 2 is expensive and ineficient.

## 3.3 Eficient Strategies

We present two simple yet efective strategies to improve the eficiency of $\operatorname { E q }$ . 2 during both training and inference.

Strategy #1: Query Only in Early Layers. Information querying is analogous to encoding, which requires fewer parameters and computations than the generation model $( \mathrm { e . g . }$ , video VAEs use far fewer parameters than video generation models). Therefore, context querying can be performed only in the early shallow layers, which our experimental results validate. Specifically, assuming the difusion transformer has $n + m$ layers in total, we divide them into two types as shown in Fig. 4 (a): n shallow Query Layers near the input and m deep Generative Layers near the output, where $n \ll m$ . Interactions among $\mathbf { C } , \mathbf { Q } .$ , and P occur only in Query Layers, while Generative Layers process only Q and P.

Strategy #2: Exclude Unnecessary Computations. We exclude unnecessary attention computations by retaining only three essential patterns as shown in Fig. 4 (b): (1) Q as queries attending to P as keys/values, enabling Q to understand what information to extract from context based on the prediction target; (2) Q as queries attending to C as keys/values, enabling Q to extract information from context; (3) P as queries attending to P and Q as keys/values, enabling P to extract information from both. All other attention computations are excluded, particularly those where C serve as queries, significantly reducing computational overhead. We provide experimental analysis of performance under diferent attention patterns in Sec. 5.5.

Combining the two strategies, we present the revised Eq. 2. For Query Layers:

$$
\begin{array}{l} \mathbf {C} _ {o u t} = \mathbf {C}, \\ \mathbf {Q} _ {o u t} = \mathbf {Q} + g (\mathbf {Q}, \{\mathbf {C}, \mathbf {P} \}, \{\mathbf {C}, \mathbf {P} \}), \\ \mathbf {P} _ {o u t} = \mathbf {P} + g (\mathbf {P}, \{\mathbf {P}, \mathbf {Q} \}, \{\mathbf {P}, \mathbf {Q} \}). \end{array}\tag{3}
$$

For Generative Layers, features of C are removed:

$$
\begin{array}{l} \mathbf {Q} _ {o u t} = \mathbf {Q} + g (\mathbf {Q}, \mathbf {P}, \mathbf {P}), \\ \mathbf {P} _ {o u t} = \mathbf {P} + g (\mathbf {P}, \{\mathbf {P}, \mathbf {Q} \}, \{\mathbf {P}, \mathbf {Q} \}). \end{array}\tag{4}
$$

## 4 Dataset

## 4.1 Dataset Collection

To validate our method, we require a dataset with specific characteristics: long videos, accurate per-frame camera annotations, occlusion relationships, dynamic objects, and suficient diversity. However, existing long video datasets do not meet these requirements, as shown in Tab. 1. To address this, we collect a rendered dataset based on Unreal Engine with the following two key designs: (1) Customized scenes and dynamic objects. We curate a diverse collection of scenes with occlusion relationships, including factories, streets, and natural landscapes, as well as dynamic objects such as humans and animals with various actions. By combining diferent scenes and dynamic objects, we construct diverse dynamic scenarios. (2) Automated trajectory generation. To enable automated trajectory generation with collision avoidance in dynamic scenes, we implement a blueprint script in Unreal Engine. This script can randomly generate camera trajectories of arbitrary length with automatic obstacle avoidance, and the traversal range is customizable. This significantly improves our data collection eficiency. Using this pipeline, we collect 100 long videos across 13 scenes, with each video averaging over 18000 frames, totaling 16.7 hours of video content. Additional details about data collection can be found in Appendix A.

## 4.2 Multi-Dataset Training Strategy

Video datasets can be categorized into three types based on camera pose annotation accuracy and visual realism: (1) Rendered data [66] with precise pose annotations but non-photorealistic style and limited diversity; (2) Real-world data with estimated poses [53], ofering photorealistic style but less precise annotations and limited dynamics due to the filtering required for accurate pose estimation; and (3) Real-world data without accurate pose annotations [39], providing photorealistic style, rich content diversity, and better dynamics, but lacking accurate camera annotations.

Table 1: Comparison of long video datasets for video world models. No existing dataset fully satisfies all four requirements: precise per-frame camera pose annotations, occlusion relationships, dynamic objects, and revisit scenarios. We collect a dataset meeting these requirements. <sup>1</sup>HQ subset only. <sup>2</sup>Available subset only. <sup>3</sup>Insuficient annotation accuracy. <sup>4</sup>Not all frames are annotated. <sup>5</sup>Primarily street walking videos with limited occlusions. <sup>6</sup>Filtered videos with reduced dynamics. <sup>7</sup>Few revisits without specialized design, especially for real-world data.  
<sup>✔</sup>: Fully satisfied <sup>●</sup>: Partially satisfied <sup>✗</sup>: Not satisfied

<table><tr><td>Dataset</td><td>Source</td><td>Style</td><td>Duration</td><td>Cam. Pose</td><td>Occlusion</td><td>Dynamic</td><td>Revisit</td></tr><tr><td>CaM [66]</td><td>Simulator</td><td>Rendered</td><td>7.0 h</td><td>√</td><td>✗</td><td>✗</td><td>√</td></tr><tr><td>SpatialVid [53]</td><td>YouTube</td><td>Real-World</td><td>1146.06  $h^{1}$ </td><td>√</td><td> $\bullet^{5}$ </td><td> $\bullet^{6}$ </td><td> $\times^{7}$ </td></tr><tr><td>Sekai-real [39]</td><td>YouTube</td><td>Real-World</td><td>304.0  $h^{1}$ </td><td> $\bullet^{3}$ </td><td> $\bullet^{5}$ </td><td>√</td><td> $\times^{7}$ </td></tr><tr><td>OmniWorld [74]</td><td>Simulator</td><td>Rendered</td><td>11.14  $h^{2}$ </td><td> $\times^{4}$ </td><td>√</td><td>√</td><td> $\bullet^{7}$ </td></tr><tr><td>Ours</td><td>Simulator</td><td>Rendered</td><td>16.7 h</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

To leverage the complementary strengths of these data sources, we propose a multi-dataset training strategy that assigns a dedicated camera encoder to each dataset type. Specifically, rendered data with precise poses and estimated-pose data each use a separate camera encoder to process their respective camera annotations, while data without reliable poses is fed with zero camera parameters (i.e., R = 0, t = 0) through a third dedicated camera encoder. By isolating diferent annotation qualities into separate encoders, datasets with varying pose accuracy do not interfere with each other during training, allowing the model to benefit from precise pose supervision of rendered data, the realism of estimated-pose data, and the diversity of unannotated real-world data simultaneously. During inference, we use only the camera encoder trained on precisely annotated data to ensure reliable camera control.

## 5 Experiments

## 5.1 Implementation Details.

In our implementation, P tokens correspond to 77 video frames per generation. The number of Q tokens equals that of P tokens. Rather than using learnable parameters, Q tokens are initialized as randomly sampled noise, which better aligns with the input distribution of difusion models. During training and inference, C tokens can be up to 9 times the length of P tokens, with the last frame of the C token sequence always matching the first frame of the P token sequence to ensure continuity. During training, the length of C tokens is uniformly sampled from 0 to 9 times the P token length, where 0 corresponds to training only the image-to-video capability. We perform full fine-tuning of the video generation model on the collected dataset.

Our primary experiments are built upon an internal 1B-parameter text-tovideo difusion transformer with 28 layers, of which 5 serve as Query Layers. We select this model for its stronger visual quality than open-source models of comparable scale and far lower computational costs than larger alternatives $( e . g . , 5 \mathrm { B }$ 14B), enabling more comprehensive ablation studies with higher-quality baselines. To verify that <sup>MemLearner</sup> generalizes across architectures, we additionally apply our method to Wan2.1 (T2V-1.3B) [52], an open-source video DiT. As detailed in Appendix C.1, <sup>MemLearner</sup> achieves consistent improvements over the CaM [66] baseline on the open-source model, confirming that our approach is architecture-agnostic and benefits from pre-trained priors regardless of the specific backbone.

Videos are generated at $6 4 0 \times 3 5 2$ resolution, producing 77 frames with a causal 3D VAE that applies a temporal compression ratio of 4, yielding 20-frame video latents. We configure P tokens to represent 20 video latent frames, with Q tokens matching the P token count, while C tokens can represent up to 180 video latent frames. Training is conducted for over 20,000 iterations with a batch size of 8 and a learning rate of $5 \times 1 0 ^ { - 5 }$ . We explore two dataset configurations: (1) Rendered-only: 50% from our collected dataset and 50% from CaM [66]; (2) Mixed: 75% rendered data (combining configuration 1) and 25% real-world data (12.5% Sekai-real [39] and 12.5% SpatialVid [53]). For sampling, we apply Classifier-Free Guidance [24] with 50 steps for text conditioning.

## 5.2 Evaluation Methods

For evaluation, we reserve a 5% held-out test set via random dataset partitioning with no video overlap between training and test splits. Our evaluation protocol includes: (1) FID/FVD to assess video visual quality; (2) PSNR/LPIPS to measure memory capability by quantifying pixel-wise diferences between frames. Following prior work [66], we adopt two evaluation strategies for memory assessment: (1) Ground truth comparison (GT Comp.): measuring whether predicted frames align with ground truth given context from ground truth frames. With suficient context, generated results should match the ground truth; (2) Revisit comparison (Revisit Comp.): comparing newly generated revisit frames against previously generated ones in long video sequences. For implementation, we test on simple camera trajectories where the camera rotates n degrees with $n \sim \mathcal { U } ( 9 0 ^ { \circ } , 1 8 0 ^ { \circ } )$ and returns, enabling straightforward identification of corresponding frame pairs. We then compute PSNR/LPIPS between all corresponding frame pairs from the outbound and return paths and average the results, thereby evaluating consistency throughout the entire generation process rather than just between the first and last frames.

To ensure evaluation diversity, we conduct experiments across multiple datasets with distinct characteristics: our collected dataset with occlusions and dynamic objects (Tab. 2, Fig. 7), the CaM dataset [66] without occlusions or dynamics (Tab. 3), and the real-world SpatialVID dataset [53] (Appendix C.3), where

![](images/9d4c8f51f1f22a442c4af3c95877b47d1fb0848877caeda4ea2a13c8f007e7d1.jpg)  
Fig. 5: Qualitative Results. <sup>MemLearner</sup> efectively handles both indoor (a, b) and outdoor (c, Fig. 1) scenes with occlusions and dynamic objects.

![](images/d178bb5e4dee1cfc3f2120546681f1f54b2537106b4f15b52c3d68ac8b539c33.jpg)  
(3) Prompt: a rustic wooden cabin beside neatly stacked

![](images/a9d0a1c70bf4d7f0f795c8aee6e706d058a97cae93bf515bb9cc8b42834640ba.jpg)  
(4) Prompt: a cozy bedroom with a neatly made bed,

Fig. 6: Real-World Qualitative Results. By incorporating real-world videos into training dataset, <sup>MemLearner</sup> generalizes to real-world scenes.

<sup>MemLearner</sup> consistently outperforms baselines across all three evaluation settings. We further provide a user study in Appendix C.4, in which <sup>MemLearner</sup> is preferred over other SOTAs in terms of visual quality and scene consistency.

## 5.3 Qualitative Results

We present qualitative examples to illustrate <sup>MemLearner</sup> generation quality. As shown in Fig. 5, our method handles diverse indoor/outdoor scenes with rich occlusions and dynamic objects. We provide video demos on our project page for direct viewing. Since our mixed training uses real-world data, <sup>Mem-</sup> <sup>Learner</sup> generalizes to real-world domains without adaptation. Fig. 6 shows the model preserves scene layout and object appearance across revisits, confirming the learned query mechanism generalizes to real environments.

## 5.4 Comparison Results

In this section, we compare baselines and SOTAs: (1) DFoT [49], a long video generation model without memory design; (2) FramePack [71], which hierarchically compresses history frames as conditions to retain limited memory; (3) VMem [36], which achieves memory capability via a point-cloud-based retrieval rule for context retrieval. (4) Context-as-Memory [66], which selects relevant conditioning frames via FOV-overlap computation to provide efective memory; (5) Separate Module, i.e., the design in Fig. 2 (b), where an additional five-layer

![](images/413e76d5175ac9ad836d7a4e25d0b67784182dcb5b0b3ebfee24819e448db354.jpg)  
Fig. 7: Qualitative Comparison. <sup>MemLearner</sup> achieves optimal memory and visual quality, demonstrating learning-based adaptive context query efectiveness. Other methods show inconsistent memory performance.

Table 2: Quantitative Comparison. <sup>MemLearner</sup> outperforms all metrics. Notably, the Separate Module (Fig. 2 (b)) lacks context learning capability, indicating joint training of a scratch module with pre-trained video DiT is inefective.

<table><tr><td rowspan="2">Methods</td><td colspan="4">GT Comp.</td><td colspan="5">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td><td>fps↑</td></tr><tr><td>DFoT [49]</td><td>16.98</td><td>0.4796</td><td>147.09</td><td>998.43</td><td>16.14</td><td>0.5481</td><td>151.38</td><td>1021.46</td><td>1.59</td></tr><tr><td>FP [71]</td><td>16.42</td><td>0.5104</td><td>143.97</td><td>967.97</td><td>15.86</td><td>0.5837</td><td>154.11</td><td>1037.58</td><td>1.40</td></tr><tr><td>VMem [36]</td><td>19.59</td><td>0.3872</td><td>129.94</td><td>850.17</td><td>17.30</td><td>0.4187</td><td>141.82</td><td>968.45</td><td>0.73</td></tr><tr><td>CaM [66]</td><td>19.85</td><td>0.3475</td><td>125.35</td><td>848.61</td><td>17.61</td><td>0.3934</td><td>137.87</td><td>948.63</td><td>0.97</td></tr><tr><td>Fig. 2 (b)</td><td>9.16</td><td>0.6567</td><td>145.63</td><td>930.54</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.48</td></tr><tr><td>Ours</td><td>21.23</td><td>0.2904</td><td>112.75</td><td>835.98</td><td>18.57</td><td>0.3230</td><td>101.57</td><td>847.52</td><td>0.54</td></tr></table>

Transformer (with the same layer structure as the video DiT) serves as a context query module. Its output channels are aligned with the video DiT’s internal features, eliminating the need for a patchify layer; (6) our proposed <sup>MemLearner</sup>.

For fair comparison, all methods are implemented using our codebase and dataset. We evaluate both memory-related metrics (PSNR/LPIPS) and visual quality metrics (FID/FVD). Results are summarized in Tab. 2 and Fig. 7. <sup>Mem-</sup> <sup>Learner</sup> achieves the best performance across all metrics and benchmarks. Separate Module achieves poor performance on GT Comp., as it fails to learn conditioning on context information and behaves like a text-to-video model. This demonstrates that jointly training a from-scratch context query module with a pre-trained video generation model cannot efectively learn context query capability; instead, the video generation model ignores the context query module. In contrast, <sup>MemLearner</sup>, which leverages the video generation model itself for context querying, is significantly more efective.

In addition to our collected dataset, we also conduct quantitative evaluations on CaM dataset [66], which lacks occlusion and dynamic object scenes as shown in Tab. 1. Results in Tab. 2 and Tab. 3 show CaM and our method perform comparably on CaM dataset (no occlusions/dynamics), while CaM degrades

Table 3: Quantitative comparison on CaM dataset [66] (no occlusions/dynamics). Small performance gaps, widening sharply on our occlusion/dynamic dataset (Tab. 2).  
![](images/ee4b400f86e587978c244cec4ce4435a1a0681eaa2f0f7399dfcb1f7abbf6868.jpg)

<table><tr><td rowspan="2">Methods</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>VMem [36]</td><td>19.84</td><td>0.3564</td><td>17.98</td><td>0.3633</td></tr><tr><td>CaM [66]</td><td>20.22</td><td>0.3003</td><td>18.11</td><td>0.3414</td></tr><tr><td>Ours</td><td>20.35</td><td>0.2975</td><td>18.29</td><td>0.3374</td></tr></table>

Fig. 8: Attention computation settings for ablation study in Tab. 4. Performance impact of attention scope when Q tokens act as queries.

Table 4: Ablation of Attention Compu tation: Settings (a)-(c) in Fig. 8. Critical computation is Q queries attending to P keys/values.  
Table 5: Ablation of camera embedding. Even without camera poses for C tokens, the model learns to query useful information with no significant performance drop.

<table><tr><td rowspan="2">Setting</td><td colspan="2">GT Comp.</td><td colspan="3">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>fps↑</td></tr><tr><td>(a)</td><td>21.23</td><td>0.2904</td><td>18.57</td><td>0.3230</td><td>0.54</td></tr><tr><td>(b)</td><td>21.19</td><td>0.2949</td><td>18.53</td><td>0.3342</td><td>0.51</td></tr><tr><td>(c)</td><td>17.27</td><td>0.4657</td><td>16.34</td><td>0.5107</td><td>0.56</td></tr></table>

<table><tr><td rowspan="2">Add Camera Emb.</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>All tokens</td><td>21.23</td><td>0.2904</td><td>18.57</td><td>0.3230</td></tr><tr><td>Only P tokens</td><td>21.17</td><td>0.2946</td><td>18.38</td><td>0.3384</td></tr></table>

sharply on our dataset and our method remains robust. This directly validates our approach’s efectiveness for handling occlusions and dynamic objects.

## 5.5 Ablation Study

Ablation of Attention Computation. As shown in Fig. 8, we compare three attention variants: (a) standard setting; (b) augmenting (a) with Q tokens as queries attending to Q tokens as keys/values; (c) removing from (a) the pattern where Q tokens as queries attend to P tokens as keys/values. Results in Tab. 4 show that (b) matches (a) in performance, the added computation is non-critical and removable. In contrast, (c) exhibits significant performance degradation, demonstrating that it is essential for Q tokens to extract guidance from P tokens to determine what information to query from C tokens. We further analyze the efect of including C tokens as queries in Appendix C.5, which introduces substantial computational overhead with negligible benefit.

Ablation of Query Layer Number. We ablate Query Layer count impact in Tab. 6. Performance saturates at 5 layers, with computational overhead rising thereafter. 5 layers balance computational cost and memory performance.

Ablation of Camera Embedding. When injecting camera pose embeddings, providing them to P tokens is essential for interactive video generation. However, it is unclear whether C tokens and Q tokens also require camera pose information. We investigate this in Tab. 5, comparing two settings: injecting camera poses only into P tokens versus all tokens. For C tokens, we inject the corresponding camera pose, while Q tokens receive the same as P tokens. Surprisingly, the results show that even without providing camera poses to C and Q tokens, the

Table 6: Ablation of Query Layer Count: 5 layers balance computational cost and memory performance.

<table><tr><td rowspan="2">#Layer</td><td colspan="2">GT Comp.</td><td colspan="3">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>fps↑</td></tr><tr><td>1</td><td>18.16</td><td>0.4056</td><td>17.25</td><td>0.4144</td><td>0.61</td></tr><tr><td>3</td><td>19.43</td><td>0.3625</td><td>17.83</td><td>0.3808</td><td>0.58</td></tr><tr><td>5</td><td>21.23</td><td>0.2904</td><td>18.57</td><td>0.3230</td><td>0.54</td></tr><tr><td>10</td><td>21.36</td><td>0.2913</td><td>18.50</td><td>0.3214</td><td>0.46</td></tr><tr><td>20</td><td>21.37</td><td>0.2891</td><td>18.66</td><td>0.3217</td><td>0.36</td></tr></table>

Table 7: Mixed Dataset Training Ablation: Real-world data alone cannot learn memory capability (insuficient revisit).

<table><tr><td rowspan="2">Datasets</td><td colspan="2">Revisit Comp.</td><td colspan="2">Real Quality</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td>CaM [66]</td><td>18.35</td><td>0.3263</td><td>167.47</td><td>1368.01</td></tr><tr><td>Real [39, 53]</td><td>15.32</td><td>0.6019</td><td>104.16</td><td>828.95</td></tr><tr><td>CaM+Real</td><td>18.27</td><td>0.3312</td><td>111.33</td><td>849.98</td></tr><tr><td>CaM+Ours</td><td>18.57</td><td>0.3230</td><td>161.83</td><td>1379.42</td></tr><tr><td>CaM+Ours+Real</td><td>18.49</td><td>0.3251</td><td>115.47</td><td>837.91</td></tr></table>

memory performance does not degrade significantly. This suggests that video generation models have the potential to learn geometric correspondences implicitly without explicit 3D information such as camera poses.

Ablation of Datasets. In Tab. 7, we ablate the impact of training with different dataset mixtures. ‘CaM’ refers to the dataset proposed in Context-as-Memory [66], ‘Real’ denotes the mixture of SpatialVID [53] and Sekai-real [39] real-world datasets, and ‘Ours’ is our proposed dataset. When combining rendered datasets ‘CaM’ and ‘Ours’, we use a 1:1 ratio, while rendered and real datasets are mixed at a 3:1 ratio. The ‘Real Quality’ metric measures the FID/FVD distance between generated videos and real-world videos. We observe that training solely on real-world data fails to learn memory capabilities, as evidenced by poor performance across all metrics, indicating insuficient revisit patterns in these real-world datasets. Models trained purely on rendered data produce videos whose distribution deviates from real videos. Incorporating real data into training efectively improves the realism of generated videos, as further illustrated by qualitative comparisons in Appendix C.6.

## 6 Conclusion

In this work, we propose a paradigm shift for context memory in video world models, transitioning from rule-based context retrieval to learning-based context query. To validate our method, we collect rendered data with accurate camera annotations, customized occlusions and dynamic objects, and suficient revisit patterns. We further propose a multi-dataset training strategy to efectively leverage both rendered and real-world videos.

Limitations and Future Work. While <sup>MemLearner</sup> advances context memory in video world models, the memory problem remains far from solved: (1) Expanding model capacity and training data scale is imperative. At our current 1B model scale, scenes with many simultaneously interacting characters in large environments remain challenging, as the model lacks suficient capacity to track numerous dynamic entities. We observe that generation quality degrades notably when more than five characters interact within the same scene, producing inconsistent appearances or missing objects upon revisit. Scaling to larger models and collecting long video data with richer dynamic interactions are necessary to address this. (2) Current research, including ours, focuses on full context storage and eficient information retrieval/querying [10,20,21,36,45,58,66], yet memory should not scale linearly with generation time. Notably, context compression is orthogonal to our contribution: one can first compress context representations and then apply our learned query mechanism on the compressed tokens. Exploring compression, summarization, updating, editing and selective forgetting are key future directions for practical, intelligent memory systems.

## Acknowledgements

This work is supported by the Research Grant Council of Hong Kong through Early Career Scheme under grant 27214925. We thank Yao Teng, Xiaoshi Wu, and Haoran He for constructive discussions and suggestions.

## References

1. ai, S., Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W.Q., Luo, W., Kang, X., Sun, Y., Cao, Y., Huang, Y., Lin, Y., Fang, Y., Tao, Z., Zhang, Z., Wang, Z., Liu, Z., Shi, D., Su, G., Sun, H., Pan, H., Wang, J., Sheng, J., Cui, M., Hu, M., Yan, M., Yin, S., Zhang, S., Liu, T., Yin, X., Yang, X., Song, X., Hu, X., Zhang, Y., Li, Y.: Magi-1: Autoregressive video generation at scale (2025), https://arxiv.org/abs/2505.13211

2. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35, 23716– 23736 (2022)

3. Arnab, A., Iscen, A., Caron, M., Fathi, A., Schmid, C.: Temporal chain of thought: Long-video understanding by thinking in frames. arXiv preprint arXiv:2507.02001 (2025)

4. Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., et al.: V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985 (2025)

5. Bai, J., Xia, M., Fu, X., Wang, X., Mu, L., Cao, J., Liu, Z., Hu, H., Bai, X., Wan, P., et al.: Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647 (2025)

6. Ball, P.J., Bauer, J., Belletti, F., Brownfield, B., Ephrat, A., Fruchter, S., Gupta, A., Holsheimer, K., Holynski, A., Hron, J., Kaplanis, C., Limont, M., McGill, M., Oliveira, Y., Parker-Holder, J., Perbet, F., Scully, G., Shar, J., Spencer, S., Tov, O., Villegas, R., Wang, E., Yung, J., Baetu, C., Berbel, J., Bridson, D., Bruce, J., Buttimore, G., Chakera, S., Chandra, B., Collins, P., Cullum, A., Damoc, B., Dasagi, V., Gazeau, M., Gbadamosi, C., Han, W., Hirst, E., Kachra, A., Kerley, L., Kjems, K., Knoepfel, E., Koriakin, V., Lo, J., Lu, C., Mehring, Z., Moufarek, A., Nandwani, H., Oliveira, V., Pardo, F., Park, J., Pierson, A., Poole, B., Ran, H., Salimans, T., Sanchez, M., Saprykin, I., Shen, A., Sidhwani, S., Smith, D., Stanton, J., Tomlinson, H., Vijaykumar, D., Wang, L., Wingfield, P., Wong, N., Xu, K., Yew, C., Young, N., Zubov, V., Eck, D., Erhan, D., Kavukcuoglu, K.,

Hassabis, D., Gharamani, Z., Hadsell, R., van den Oord, A., Mosseri, I., Bolton, A., Singh, S., Rocktäschel, T.: Genie 3: A new frontier for world models (2025)

7. Bao, F., Xiang, C., Yue, G., He, G., Zhu, H., Zheng, K., Zhao, M., Liu, S., Wang, Y., Zhu, J.: Vidu: a highly consistent, dynamic and skilled text-to-video generator with difusion models. arXiv preprint arXiv:2405.04233 (2024)

8. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020)

9. Buch, S., Nagrani, A., Arnab, A., Schmid, C.: Flexible frame selection for eficient video reasoning. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 29071–29082 (2025)

10. Cai, S., Yang, C., Zhang, L., Guo, Y., Xiao, J., Yang, Z., Xu, Y., Yang, Z., Yuille, A., Guibas, L., Agrawala, M., Jiang, L., Wetzstein, G.: Mixture of contexts for long video generation. In: arXiv (2025)

11. Chen, B., Monso, D.M., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffusion forcing: Next-token prediction meets full-sequence difusion. arXiv preprint arXiv:2407.01392 (2024)

12. Chen, T., Hu, X., Ding, Z., Jin, C.: Learning world models for interactive video generation. arXiv preprint arXiv:2505.21996 (2025)

13. Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., Hsieh, C.J.: Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283 (2025)

14. Cui, Y., Chen, H., Deng, H., Huang, X., Li, X., Liu, J., Liu, Y., Luo, Z., Wang, J., Wang, W., Wang, Y., Wang, C., Zhang, F., Zhao, Y., Pan, T., Li, X., Hao, Z., Ma, W., Chen, Z., Ao, Y., Huang, T., Wang, Z., Wang, X.: Emu3.5: Native multimodal models are world learners (2025), https://arxiv.org/abs/2510.26583

15. Damen, D., Doughty, H., Farinella, G.M., Fidler, S., Furnari, A., Kazakos, E., Moltisanti, D., Munro, J., Perrett, T., Price, W., Wray, M.: Scaling egocentric vision: The epic-kitchens dataset. In: Proceedings of the European Conference on Computer Vision (ECCV) (2018)

16. Decart, E.: Oasis: A universe in a transformer. https://oasis-model.github.io/ (2024)

17. DeepMind, G.: Veo 2: Our state-of-the-art video generation model. https:// deepmind.google/technologies/veo/veo-2/ (2024)

18. Deng, H., Pan, T., Diao, H., Luo, Z., Cui, Y., Lu, H., Shan, S., Qi, Y., Wang, X.: Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169 (2024)

19. Fu, X., Liu, X., Wang, X., Peng, S., Xia, M., Shi, X., Yuan, Z., Wan, P., Zhang, D., Lin, D.: 3dtrajmaster: Mastering 3d trajectory for multi-entity motion in video generation. In: ICLR (2025)

20. Gu, Y., Mao, W., Shou, M.Z.: Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325 (2025)

21. Guo, Y., Yang, C., Yang, Z., Ma, Z., Lin, Z., Yang, Z., Lin, D., Jiang, L.: Long context tuning for video generation. arXiv preprint arXiv:2503.10589 (2025)

22. Ha, D., Schmidhuber, J.: Recurrent world models facilitate policy evolution 31 (2018)

23. He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., Yang, C.: Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101 (2024)

24. Ho, J., Salimans, T.: Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598 (2022)

25. Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., Hold-Geofroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al.: Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040 (2025)

26. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009 (2025)

27. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

28. Kanervisto, A., Bignell, D., Wen, L.Y., Grayson, M., Georgescu, R., Valcarcel Macua, S., Tan, S.Z., Rashid, T., Pearce, T., Cao, Y., et al.: World and human action models towards gameplay ideation. Nature 638(8051), 656–663 (2025)

29. Kim, S.W., Zhou, Y., Philion, J., Torralba, A., Fidler, S.: Learning to simulate dynamic environments with gamegan. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1231–1240 (2020)

30. Kingma, D.P., Welling, M., et al.: Auto-encoding variational bayes (2013)

31. Kling: Kling ai: Next-generation ai creative studio. https://app.klingai.com/ (2024)

32. Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Schindler, G., Hornung, R., Birodkar, V., Yan, J., Chiu, M.C., et al.: Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125 (2023)

33. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)

34. Labs, W.: Generating worlds. https://www.worldlabs.ai/blog/generatingworlds (2024)

35. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International conference on machine learning. pp. 19730–19742. PMLR (2023)

36. Li, R., Torr, P., Vedaldi, A., Jakab, T.: Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903 (2025)

37. Li, T., Tian, Y., Li, H., Deng, M., He, K.: Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838 (2024)

38. Li, W., Pan, W., Luan, P.C., Gao, Y., Alahi, A.: Stable video infinity: Infinitelength video generation with error recycling. arXiv preprint arXiv:2510.09212 (2025)

39. Li, Z., Li, C., Mao, X., Lin, S., Li, M., Zhao, S., Xu, Z., Li, X., Feng, Y., Sun, J., Li, Z., Zhang, F., Ai, J., Wang, Z., Wu, Y., He, T., Pang, J., Qiao, Y., Jia, Y., Zhang, K.: Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675 (2025)

40. Li, Z., Yu, H.X., Liu, W., Yang, Y., Herrmann, C., Wetzstein, G., Wu, J.: Wonderplay: Dynamic 3d scene generation from a single image and actions. In: Proceedings of the IEEE/CVF international conference on computer vision (2025)

41. Liu, K., Hu, W., Xu, J., Shan, Y., Lu, S.: Rolling forcing: Autoregressive long video difusion in real time. arXiv preprint arXiv:2509.25161 (2025)

42. Ma, B., Gao, H., Deng, H., Luo, Z., Huang, T., Tang, L., Wang, X.: You see it, you got it: Learning 3d creation on pose-free videos at scale. arXiv preprint arXiv:2412.06699 (2024)

43. OpenAI: Creating video from text. https://openai.com/index/sora/ (2024)

44. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023)

45. Po, R., Nitzan, Y., Zhang, R., Chen, B., Dao, T., Shechtman, E., Wetzstein, G., Huang, X.: Long-context state-space video world models (2025), https://arxiv. org/abs/2505.20171

46. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning (2021)

47. Ren, X., Shen, T., Huang, J., Ling, H., Lu, Y., Nimier-David, M., Müller, T., Keller, A., Fidler, S., Gao, J.: Gen3c: 3d-informed world-consistent video generation with precise camera control. arXiv preprint arXiv:2503.03751 (2025)

48. Runway: Runway : Tools for human imagination. https://runwayml.com/ (2024)

49. Song, K., Chen, B., Simchowitz, M., Du, Y., Tedrake, R., Sitzmann, V.: Historyguided video difusion. arXiv preprint arXiv:2502.06764 (2025)

50. Sun, W., Zhang, H., Wang, H., Wu, J., Wang, Z., Wang, Z., Wang, Y., Zhang, J., Wang, T., Guo, C.: Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614 (2025)

51. Valevski, D., Leviathan, Y., Arar, M., Fruchter, S.: Difusion models are real-time game engines. arXiv preprint arXiv:2408.14837 (2024)

52. Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)

53. Wang, J., Yuan, Y., Zheng, R., Lin, Y., Gao, J., Chen, L.Z., Bao, Y., Zhang, Y., Zeng, C., Zhou, Y., Long, X., Zhu, H., Zhang, Z., Cao, X., Yao, Y.: Spatialvid: A large-scale video dataset with spatial annotations (2025), https://arxiv.org/ abs/2509.09676

54. Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al.: Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 (2024)

55. Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., Shan, Y.: Motionctrl: A unified and flexible motion controller for video generation. In: ACM SIGGRAPH 2024 Conference Papers (2024)

56. Wu, T., Yang, S., Po, R., Xu, Y., Liu, Z., Lin, D., Wetzstein, G.: Video world models with long-term spatial memory (2025), https://arxiv.org/abs/2506.05284

57. Wu, Z., Xiong, C., Ma, C.Y., Socher, R., Davis, L.S.: Adaframe: Adaptive frame selection for fast video recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1278–1287 (2019)

58. Xiao, Z., Lan, Y., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., Pan, X.: Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369 (2025)

59. Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021)

60. Yang, S., Walker, J.C., Parker-Holder, J., Du, Y., Bruce, J., Barreto, A., Abbeel, P., Schuurmans, D.: Position: Video as the new language for real-world decision making. In: Proceedings of the 41st International Conference on Machine Learning (2024)

61. Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., Chen, S.H.Y.: Longlive: Real-time interactive long video generation (2025)

62. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)

63. Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., et al.: Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800 (2024)

64. Yu, H.X., Duan, H., Herrmann, C., Freeman, W.T., Wu, J.: Wonderworld: Interactive 3d scene generation from a single image. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5916–5926 (June 2025)

65. Yu, H.X., Duan, H., Hur, J., Sargent, K., Rubinstein, M., Freeman, W.T., Cole, F., Sun, D., Snavely, N., Wu, J., et al.: Wonderjourney: Going from anywhere to everywhere. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6658–6667 (2024)

66. Yu, J., Bai, J., Qin, Y., Liu, Q., Wang, X., Wan, P., Zhang, D., Liu, X.: Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141 (2025)

67. Yu, J., Qin, Y., Che, H., Liu, Q., Wang, X., Wan, P., Zhang, D., Gai, K., Chen, H., Liu, X.: A survey of interactive generative video. arXiv preprint arXiv:2504.21853 (2025)

68. Yu, J., Qin, Y., Che, H., Liu, Q., Wang, X., Wan, P., Zhang, D., Liu, X.: Position: Interactive generative video as next-generation game engine. arXiv preprint arXiv:2503.17359 (2025)

69. Yu, J., Qin, Y., Wang, X., Wan, P., Zhang, D., Liu, X.: Gamefactory: Creating new games with generative interactive videos (2025)

70. Yu, W., Xing, J., Yuan, L., Hu, W., Li, X., Huang, Z., Gao, X., Wong, T.T., Shan, Y., Tian, Y.: Viewcrafter: Taming video difusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048 (2024)

71. Zhang, L., Agrawala, M.: Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626 (2025)

72. Zhou, G., Pan, H., LeCun, Y., Pinto, L.: Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983 (2024)

73. Zhou, S., Du, Y., Yang, Y., Han, L., Chen, P., Yeung, D.Y., Gan, C.: Learning 3d persistent embodied world models. arXiv preprint arXiv:2505.05495 (2025)

74. Zhou, Y., Wang, Y., Zhou, J., Chang, W., Guo, H., Li, Z., Ma, K., Li, X., Wang, Y., Zhu, H., Liu, M., Liu, D., Yang, J., Fu, Z., Chen, J., Shen, C., Pang, J., Zhang, K., He, T.: Omniworld: A multi-domain and multi-modal dataset for 4d world modeling (2025), https://arxiv.org/abs/2509.12201

75. Zhu, F., Wu, H., Guo, S., Liu, Y., Cheang, C., Kong, T.: Irasim: Learning interactive real-robot action simulators. arXiv preprint arXiv:2406.14540 (2024)

## A Details of Collected Dataset

3D Scenes and Dynamic Objects. We collect 13 diverse 3D scene assets from Fab.com<sup>5</sup>. To minimize the domain gap between rendered data and real-world videos, we prioritize photorealistic scene assets. We also incorporate stylized scenes to enhance dataset diversity. The collected scenes encompass various environments including streets, shopping malls, rural areas, indoor and outdoor spaces. To increase scene dynamics, we introduce dynamic objects including human characters with diverse appearances and animals such as dogs, camels, and horses. By randomly combining these scenes with dynamic objects, we construct diverse dynamic 3D scenarios for training data generation.

Automated Camera Trajectory Generation. To automate camera trajectory generation, we implement a Blueprint script in Unreal Engine. The script introduces a camera actor with randomized movement controlled by the script logic. In play mode, this actor autonomously navigates the scene using Unreal Engine’s built-in navigation APIs for obstacle avoidance and random exploration. The script logs the camera pose at each timestep to a file. Subsequently, we parse this log file to generate sequence files compatible with Unreal Engine’s renderer, enabling direct video rendering. For rendering, the camera is configured with a focal length of 24mm, an aperture of 10, and a field of view (FOV) of 52.67 degrees.

Data Preprocessing and Text Annotation. The rendered video frames are captioned using a pre-trained multimodal large language model [63]. We generate captions every 77 frames. During training, we randomly sample 77 frames from a long video sequence and use the caption of the nearest annotated segment as the text condition for training.

## B Details of Internal Model Architecture

Base Text-to-Video Generation Model Our base T2V generation model adopts a latent difusion transformer architecture, shown in Figure 9. Videos are first encoded into latent representations via a 3D-VAE, which then serve as the input to a difusion transformer. Previous approaches using UNets or transformers often append a separate 1D temporal attention module for temporal modeling. However, this spatially-temporally decoupled design is suboptimal. We adopt 3D self-attention to directly model spatiotemporal tokens, leading to improved coherence and quality in generated videos. We derive scale parameters from the difusion timestep and apply RMSNorm to spatiotemporal tokens before each attention and feed-forward network (FFN) layer.

3D Attention Implementation Details. Our designed 3D attention eficiently processes three types of tokens: context, query, and predicted tokens. We provide the corresponding pseudocode in Figure 10.

![](images/1bec892a055f6a432f50364bdef87d818c813898940ecd7925c4f230db9dfd2e.jpg)  
Fig. 9: Overview of the base text-to-video generation model.

## C Supplementary Experimental Results

## C.1 Results on Open-Source Video Models

To verify that MemLearner generalizes across diferent models, we apply our method to Wan 2.1 (T2V-1.3B) [52], an open-source video Difusion Transformer [43, 44]. Wan 2.1 shares a similar architecture with our internal model: each Transformer block contains a 3D attention module, making it directly compatible with our context query mechanism. We follow the same training protocol as described in Sec. 5.1 of the main text and compare against CaM [66] and VMem [36].

As shown in Tab. 8, MemLearner consistently outperforms both baselines on Wan 2.1, confirming that our learning-based context query approach is architectureagnostic. We note that the absolute performance of all methods on Wan 2.1 is lower than on our internal model (Tab. 2 in the main text), which we attribute to the relatively weaker generation capability of this open-source model at the 1.3B scale.

<table><tr><td rowspan="2">Methods</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>VMem [36]</td><td>18.24</td><td>0.4282</td><td>16.87</td><td>0.4738</td></tr><tr><td>CaM [66]</td><td>18.37</td><td>0.4192</td><td>16.91</td><td>0.4640</td></tr><tr><td>Ours</td><td>19.25</td><td>0.3913</td><td>17.17</td><td>0.4472</td></tr></table>

Table 8: Quantitative comparison on Wan 2.1 (T2V-1.3B) [52].

## C.2 Attention Visualization Results

We visualize the attention similarity between query tokens and context tokens to validate two key claims in the main text.

```python
# Extract tokens from hidden states
def extract_tokens(hs):
    """Split hs into P, Q, C tokens based on frame indices"""
    return P_tokens, Q_tokens, C_tokens

# 3D Attention Computation
if layer_type == "Generative":
    # Extract P and Q tokens only
    Q_query, P_query = extract_tokens(query)
    Q_key, P_key = extract_tokens(key)
    Q_value, P_value = extract_tokens(value)

    # Q attends to P; P attends to P+Q
    Q_out = flash_attention(Q_query, P_key, P_value)
    P_out = flash_attention(P_query, concat(P_key, Q_key), concat(P_value, Q_value))

elif layer_type == "Query":
    # Extract C, Q, P tokens
    Q_query, P_query, C_query = extract_tokens(query)
    Q_key, P_key, C_key = extract_tokens(key)
    Q_value, P_value, C_value = extract_tokens(value)

    # Q attends to P+C; P attends to P+Q
    Q_out = flash_attention(Q_query, concat(P_key, C_key), concat(P_value, C_value))
    P_out = flash_attention(P_query, concat(P_key, Q_key), concat(P_value, Q_value))

# Concatenate and project output
output = concat(P_out, Q_out)
hidden_states = output_projection(output)
```  
Fig. 10: Pseudocode for 3D attention computation. Generative Layers process P and Q tokens, while Query Layers process all C, Q, P tokens with diferent attention patterns.

Adaptive Attention Across Frames and Timesteps. As shown in the left half of Fig. 11, query tokens exhibit markedly diferent attention distributions across predicted frames and difusion timesteps. Early difusion timesteps attend more broadly to context tokens, capturing global scene layout, while later timesteps focus on fine-grained local correspondences. This confirms that diferent predicted frames and denoising stages require diferent information from context, justifying our adaptive query design over naive rule-based context retrieval.

Failure of the Separate Module. As shown in the right half of Fig. 11, the separate context query module (Fig. 2(b) in the main text) produces near-zero attention similarity between query and context tokens, indicating a failure to establish meaningful context modeling. This lack of efective information flow hinders gradient propagation to the video generation model, which consequently ignores the separate context query module’s output and degrades to a text-to-video model, consistent with the quantitative results in Tab. 2 in the main text.

![](images/3edcb0fbc4a3513a4f0a0462815214de49a81ee1991b85cc5e01fe51892bff5e.jpg)  
Fig. 11: Attention visualization results.

## C.3 Evaluation on Real-World SpatialVID Dataset

To evaluate generalization to real-world scenarios, we conduct quantitative experiments on the SpatialVID dataset [53]. We select SpatialVID for two reasons: (1) it contains long videos with sequences up to 900 frames, and (2) it provides per-frame camera pose annotations, enabling controlled evaluation. We randomly sample 1,000 long sequences from the dataset for testing, ensuring no overlap with training data.

As shown in Tab. 9, MemLearner consistently outperforms baselines across all metrics. However, the performance gaps are relatively small compared to those on our collected dataset (Tab. 2 in the main text). This is expected: SpatialVID filters out videos with large camera motions to ensure accurate pose estimation, and real-world videos inherently lack frequent revisit patterns (as illustrated in Tab. 1 of the main text). These characteristics make SpatialVID a relatively easy benchmark for evaluating memory capability.

<table><tr><td rowspan="2">Methods</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>VMem [36]</td><td>21.76</td><td>0.2844</td><td>18.29</td><td>0.3301</td></tr><tr><td>CaM [66]</td><td>21.83</td><td>0.2674</td><td>18.41</td><td>0.3285</td></tr><tr><td>Ours</td><td>22.46</td><td>0.2576</td><td>18.75</td><td>0.3078</td></tr></table>

Table 9: Quantitative comparison on the real-world SpatialVID dataset [53].

## C.4 User Study

In Table 10, we also conduct a user study, and the results clearly show that users overwhelmingly prefer our method for both visual quality and scene consistency.

<table><tr><td></td><td>DFoT [49]</td><td>FramePack [71]</td><td>VMem [36]</td><td>CaM [66]</td><td>Ours</td></tr><tr><td>Quality (%)</td><td>20.79</td><td>14.53</td><td>49.00</td><td>54.13</td><td>69.51</td></tr><tr><td>Consistency (%)</td><td>9.11</td><td>15.09</td><td>41.88</td><td>45.58</td><td>72.93</td></tr></table>

Table 10: We randomly selected one video clip to be predicted from each of the 13 scenes and compared <sup>MemLearner</sup> with others. 27 users chose their preferred video from a randomly ordered set, with multiple selections allowed. The table shows user preference rates.

## C.5 Additional Attention Computational Comparison

We provide a supplementary comparison of computational cost and performance when including context tokens as queries in 3D attention computations. As shown in Table 11 and Figure 12, incorporating context tokens as queries introduces substantial computational overhead while ofering negligible or even degraded performance.

![](images/a8d5abec352ecbef09e3105960cc0d13de2b303a243f4607505bf92c48ae6b88.jpg)

![](images/5e1e4b8c6169b7c177d55aec11da61574f1350427a11a649b68368bd80f028ce.jpg)

![](images/53796b5ba1523c2096ca9eccd2f03074612225688356ae3b2496e013d3664af0.jpg)

![](images/cf973eb3daa56bb29abb2542566705bf1360485068ae819ca75658520fdeacc3.jpg)  
Fig. 12: Additional attention computation settings. Horizontal tokens serve as queries; vertical tokens serve as keys/values.

<table><tr><td rowspan="2">Setting</td><td colspan="2">GT Comp.</td><td colspan="3">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>Speed (fps)↑</td></tr><tr><td>(a)*5+(b)*23</td><td>21.23</td><td>0.2904</td><td>18.57</td><td>0.3230</td><td>0.54</td></tr><tr><td>(c)*5+(b)*23</td><td>21.15</td><td>0.2976</td><td>18.43</td><td>0.3291</td><td>0.46</td></tr><tr><td>(c)*28</td><td>21.27</td><td>0.2914</td><td>18.62</td><td>0.3212</td><td>0.24</td></tr><tr><td>(d)*28</td><td>21.34</td><td>0.2895</td><td>18.67</td><td>0.3227</td><td>0.28</td></tr></table>

Table 11: Additional attention computation comparison. Settings $\overline { { ( \mathrm { a } ) \ – ( \mathrm { d } ) } }$ are shown in Fig. 12. The model has 28 layers total. In the notation, ${ } ^ { 4 \cdot } ( \mathrm { a } ) ^ { \ast } 5 + ( \mathrm { b } ) ^ { \ast } 2 3 ^ { , }$ indicates that the first 5 Query Layers use setting (a) and the remaining 23 Generative Layers use setting (b). This is the default setting for <sup>MemLearner</sup>. Other settings follow the same notation convention.

## C.6 Additional Dataset Ablation Results

Figure 13 shows additional qualitative results on how incorporating real-world training data improves the realism of generated videos.

## C.7 Additional Comparison Results

In Figure 14, we provide additional qualitative comparisons across diferent methods.

## C.8 Comparison with a Separately Trained Query Module

A natural alternative to our end-to-end design is to train the context query module separately with explicit supervision. Following this direction (separate pre-training, auxiliary supervision, and progressive fine-tuning), we pre-train a separate query module with an L1 loss between the query (Q) and predicted (P) tokens, and then jointly fine-tune it together with the video generation model. As shown in Tab. 12, this alternative remains far below MemLearner. We attribute this to the dificulty of designing explicit supervision for context querying: what constitutes “correctly queried memory” is unknown, so any hand-crafted supervision target is likely suboptimal. In contrast, MemLearner avoids this issue by learning what to query end-to-end from the difusion loss alone.

<table><tr><td rowspan="2">Setting</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td><td rowspan="2">fps↑</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Alternative</td><td>16.56</td><td>0.4929</td><td>15.71</td><td>0.6057</td><td>0.48</td></tr><tr><td>Ours</td><td>21.23</td><td>0.2904</td><td>18.57</td><td>0.3230</td><td>0.54</td></tr></table>

Table 12: Comparison with a separately trained query module using auxiliary L1 supervision and progressive fine-tuning.

## C.9 Ablation on Query Token Initialization

We further investigate how the initialization of query tokens afects performance. By default, MemLearner initializes the Q tokens as randomly sampled noise, which aligns with the input distribution of difusion models. We also try initializing the Q tokens from a copy of the noisy predicted (P) tokens. As shown in Tab. 13, the two initialization strategies perform comparably. This is because the Q-to-P attention lets the query tokens learn what the predicted tokens need at each forward pass; the query guidance is thus acquired through attention rather than from the specific initial values, confirming that the initialization is not critical.

<table><tr><td rowspan="2">Q Init.</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Noisy P tokens</td><td>21.17</td><td>0.2938</td><td>18.67</td><td>0.3227</td></tr><tr><td>Noise (Ours)</td><td>21.23</td><td>0.2904</td><td>18.57</td><td>0.3230</td></tr></table>

Table 13: Ablation on query token initialization. Noise initialization and noisy-Ptoken initialization perform comparably.

## C.10 Comparison with Geometry-Based Retrieval

To further compare against geometry/position-based retrieval, we evaluate VRAG [12], which difers from FOV-based retrieval (e.g., CaM [66]) by leveraging geometric position cues to alleviate the wall-occlusion problem. As shown in Tab. 14, Mem-Learner still outperforms VRAG across all metrics, demonstrating the advantage of learning-based context query over geometry-based retrieval.

<table><tr><td rowspan="2">Methods</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>VRAG [12]</td><td>19.61</td><td>0.3782</td><td>17.22</td><td>0.4008</td></tr><tr><td>Ours</td><td>21.23</td><td>0.2904</td><td>18.57</td><td>0.3230</td></tr></table>

Table 14: Comparison with the geometry-based retrieval method VRAG [12].

## C.11 Zero-Shot Transfer to Real-World Epic-Kitchens

To validate generalization to out-of-distribution real-world scenarios, we conduct a zero-shot transfer evaluation on the Epic-Kitchens dataset [15], a large-scale egocentric video dataset that difers substantially from our training data. As shown in Tab. 15, MemLearner consistently outperforms CaM [66] and VMem [36], confirming that our learned memory mechanism generalizes to real-world scenarios and is not overfitted to the specific occlusion and dynamic-object patterns of our rendered training data.

<table><tr><td rowspan="2">Epic-Kitchens Methods</td><td colspan="2">GT Comp.</td><td colspan="2">Revisit Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>VMem [36]</td><td>18.85</td><td>0.4208</td><td>16.94</td><td>0.4450</td></tr><tr><td>CaM [66]</td><td>19.31</td><td>0.3940</td><td>17.28</td><td>0.4206</td></tr><tr><td>Ours</td><td>20.19</td><td>0.3114</td><td>18.35</td><td>0.3375</td></tr></table>

Table 15: Zero-shot transfer comparison on the real-world Epic-Kitchens dataset [15].

## C.12 Additional Video Quality Evaluation: VBench and CLIP Similarity

Since PSNR/LPIPS primarily assess pixel-level consistency, we additionally evaluate video quality with VBench [27] and semantic consistency with CLIP similarity [46]. We report five VBench dimensions, namely Background Consistency (BG), Temporal Flickering (TF), Motion Smoothness (MS), Aesthetic Quality (AQ), and Imaging Quality (IQ), together with CLIP similarity (CLIP). As shown in Tab. 16, MemLearner consistently outperforms CaM [66] and VMem [36] across all metrics, confirming that its advantage holds beyond pixel-level metrics.

<table><tr><td>Method</td><td>BG↑</td><td>TF↑</td><td>MS↑</td><td>AQ↑</td><td>IQ↑</td><td>CLIP↑</td></tr><tr><td>VMem [36]</td><td>0.9618</td><td>0.9504</td><td>0.9638</td><td>0.5883</td><td>0.6357</td><td>0.3219</td></tr><tr><td>CaM [66]</td><td>0.9630</td><td>0.9539</td><td>0.9647</td><td>0.5895</td><td>0.6442</td><td>0.3190</td></tr><tr><td>Ours</td><td>0.9684</td><td>0.9572</td><td>0.9661</td><td>0.5905</td><td>0.6491</td><td>0.3255</td></tr></table>

Table 16: Additional video quality evaluation on VBench [27] dimensions (BG: Background Consistency; TF: Temporal Flickering; MS: Motion Smoothness; AQ: Aesthetic Quality; IQ: Imaging Quality) and CLIP similarity.

![](images/c993eae100af1058224f936fac93887748e502dd2b9f3787ea277b091e2b9980.jpg)  
(1) Prompt: an ornate festival pier filled with glowing lanterns, where dragon-shaped lights and tall pavilions sway softly under the twilight sky dotted with emerging stars ...

![](images/a3a9a296fe6e2d6602f9218c64944cbe9df43a3a1dbffed081be64609b2c7f06.jpg)

![](images/3d227ca1721813754f51ba8ad2226aaa59731be03eab4ce7f6dab4810d66b644.jpg)

![](images/2bec4d0d7b4e1613d95b11e28ec4a7c1f4e2ded6eb7c1eb6c849ea08727477b1.jpg)  
(2) Prompt: A modern two-story house with light siding, stone accents, two garage doors, and a tidy lawn sits along a quiet suburban street ...

![](images/627bcad7cf7dc76fd0d5d07b14a11767f6e38c639b9c190065302b22d7cb28fe.jpg)

![](images/15bf664fcf2c334676d327519eac1ba6f4c8a329b7bd07ed1363df1e22408193.jpg)  
(3) Prompt: a rustic wooden cabin beside neatly stacked logs in a green alpine meadow, surrounded by dense evergreens and steep mountain peaks ...

![](images/7b9980bb992cd4c2a6d23375af2ee15367ef39a83673635b0dda25a9b4f2147b.jpg)  
(4) Prompt: a cozy modern bedroom with a neatly made olivegreen bed, a floral feature wall with a sunburst mirror, a simple nightstand, and a mustard chair beside a small table ...

![](images/d684d12091f9318ae1c84d8be17ee5c22cc4e29dd2d717bd69a930d2b0da7f53.jpg)  
(5) Prompt: a formal garden with a circular stone pedestal, symmetrical tall hedges, blooming pink roses, and stone urns framing the entrance to a stairway ...

Fig. 13: Additional Dataset Ablation Results. “Rendered” denotes results trained only on rendered videos, while “+Real” indicates results after adding real-world videos to the training set.

![](images/ab18b06f45ed9e6eb5e7a5335e6f50316098b6559e342d9c16b1dfd6fe425169.jpg)

![](images/8f16c9cd0509061a1e1150eb9b0ab5eb570586b80960a28e85e2edbd0ccbbce0.jpg)  
Fig. 14: Additional Qualitative Comparison Results.