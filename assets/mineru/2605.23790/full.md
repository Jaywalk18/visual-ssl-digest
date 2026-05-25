# Exploring deep learning for Event-Based Saliency Prediction with a Transformer-based model

Romaric Mazna1[0009−0004−4658−0685], Jean Martinet1[0000−0001−8821−5556], and Sai Deepesh Pokala1[0009−0008−2837−2988]

1i3S/CNRS, Université Côte d’Azur {firstname.surname}@univ-cotedazur.fr

Abstract. Saliency prediction has been extensively studied in RGB images and videos as a computational model of human visual attention. In contrast, predicting saliency from event-based data remains largely unexplored, despite the biological inspiration and favorable sensing properties of event cameras. Two obstacles have held this direction back: the absence of large-scale event saliency datasets, and the lack of a strong baseline. In this paper, we introduce SEST (Swin Event-based Saliency Transformer), a transformer-based model for saliency prediction from event data, bridging the data scarcity barrier through event-native pretraining and synthetic supervision. SEST leverages a self-supervised pretrained event-based Swin Transformer backbone combined with a CNN decoder to produce dynamic saliency maps. To address the scarcity of annotated event-based saliency data, we introduce two new benchmark datasets, N-DHF1K and N-UCF Sports, generated from large-scale RGB saliency benchmarks. Experimental results show that SEST clearly outperforms existing event-based saliency methods and narrows the performance gap with state-of-the-art RGB models. Zero-shot evaluation on a real event camera dataset further demonstrates that our model trained on synthetic data remains transferable on real event streams. To the best of our knowledge, this work is the first to apply deep learning to event-based saliency prediction, opening a new research direction at the intersection of event-based vision and neuromorphic visual attention. Code and data are available at: https://github.com/romageek/sest.

Keywords: Saliency · Visual attention · Event-based data · Transformer

# 1 Introduction

Visual attention refers to the ability of the human brain to selectively focus on the most relevant parts of a scene while ignoring less important details. Saliency prediction aims to model this phenomenon and has many applications in computer vision such as image captioning, image and video compression, and video summarization. Over the past years, several models have been developed to predict saliency in static and dynamic RGB data. Large-scale datasets with dense human fixation annotations have enabled the training of powerful convolutional, recurrent and transformer-based architectures capable of capturing complex spatial and temporal attention patterns. However, these advances rely only on visually dense input.

Recently, event cameras have seen a growing interest due to their advantages such as high dynamic range, low latency, energy efficiency and high temporal resolution. Contrary to conventional RGB cameras, where images are recorded at a fixed rate and resolution, event cameras mimic the functioning of the human retina by detecting and recording brightness changes at pixel-level, thus removing redundancy in the generated data. While deep learning has been successfully applied to several event-based tasks such as object detection, optical flow and semantic segmentation, it has not yet addressed visual attention or saliency prediction.

Research on saliency prediction in event-based data is still at an early stage. In the context of RGB data, different architectures from low-level visual features to deep learning methods have been developed to predict saliency. However, in the context of event-based data, the few models that have been developed are either hand-designed (proto-object models, per-pixel saliency pipelines) or use spiking architectures without gradient-based training. This can be explained by the lack of available human fixation datasets. Currently, only one event-based saliency dataset [5] exists, comprising 598 clips of 8s each across a limited set of 6 imbalanced scene categories recorded with a stationary camera. Most of the few studies on saliency prediction in event-based data have relied on synthesizing a few samples of images/videos from RGB saliency datasets into events to evaluate their models. This lack of data has limited research in saliency prediction for event-based data. We argue this barrier can be bridged by combining self-supervised event pretraining with synthetic supervision. We thus leverage a Swin Transformer pretrained exclusively on event data [41], followed by a CNN decoder to produce saliency maps.

Furthermore, we introduce two new event-based saliency datasets, N-DHF1K and N-UCF Sports. We transform videos from video saliency datasets DHF1K [38] and UCF Sports [28] into events using the ESIM library [13]. The DHF1K dataset consists of 1K videos with more than 600K frames and per-frame fixation annotations from 17 observers across 150 scene categories covering diverse motions and activities. The UCF Sports dataset consists of 150 videos chosen from the UCF Sports Action dataset [36]. This scale and diversity enable far more robust saliency models than the only available real event-based dataset.

From a neuromorphic perspective, saliency prediction in event-based data aligns naturally with the principles of sparse, asynchronous, and temporally precise neural computation. Unlike frame-based saliency models that operate on dense visual representations, event-based saliency emerges from spatio-temporal contrast and motion cues, closely resembling bottom-up attentional mechanisms observed in biological vision. Although the proposed architecture is implemented using conventional deep learning components, it exploits event sparsity, temporal binning, and multi-scale temporal integration, making it compatible with future spiking adaptations for neuromorphic deployments.

The contributions of this work are the following:

1. We present the first deep-learning approach to event-based saliency prediction, opening a research direction the field has not yet explored, and providing a baseline architecture (SEST) that the community can build on.   
2. We introduce two event-based saliency datasets, N-DHF1K and N-UCF Sports, generated from large-scale RGB benchmarks to address the scarcity of annotated event saliency data.   
3. We demonstrate that SEST significantly outperforms existing event-based saliency methods and transfers to real event streams in a zero-shot setting, validating the utility of our synthetic benchmarks.

The paper is structured as follows. Sec. 2 reviews related work, Sec. 3 details the proposed architecture, Sec. 4 describes the experimental setup, Sec. 5 presents and discusses the results, and Sec. 6 concludes the paper.

# 2 Related work

# 2.1 Saliency prediction in RGB data

There have been various studies aimed at developing computational models for saliency prediction in RGB data. Early traditional approaches were based on the feature integration theory [37]. The most commonly known work is the Itti and Koch model [18], which combines multi-scale image features into a single topographical saliency map. Le Meur et al. [25] refined this approach by incorporating additional human visual system (HVS) features, such as contrast sensitivity, perceptual decomposition, visual masking, and center-surround interactions. Other successful models include GBVS [15], which constructs a fully connected graph over multiscale feature maps, and Hou et al. [16], which introduces the Incremental Coding Length approach to maximize entropy gain in feature selection. Several static saliency models have also been proposed [2,12,22,32,33], all relying on multi-scale feature computation to generate saliency maps. With advancements in deep learning, deep neural networks have been employed to predict saliency maps. Early CNN-based saliency models include EDN [35] and Deep-Fix [23], the latter incorporating VGG-16 features with location-based convolutional layers. Later approaches, such as SalGAN [34], leveraged GANs to enhance saliency predictions in static images through adversarial training, while EML-NET [20] introduced a disjoint encoder-decoder architecture. More recently, [9] compared Vision Transformer (ViT) self-attention maps with saliency maps, demonstrating strong similarities, particularly under self-supervised training. With the availability of large video eye tracking datasets, several works focused on dynamic saliency modeling to leverage temporal context. Early deep video saliency models used convolutional architectures with recurrent units, including ConvLSTM-based architectures such as SalEMA [26] and DeepVS [21], and attentive ConvLSTM models like ACLNet [38] and SalSAC [40]. Other works using convolutional architectures include TASED Net [29], HD2S [1], which use 3D CNNs to capture spatiotemporal features directly through volumetric convolutions, thereby avoiding recurrent units. Finally, more recent approaches increasingly rely on attention mechanisms and transformers to capture dependencies over a larger time-frame. Models such as STSANet [39] and STRA-Net [24] introduce spatiotemporal self-attention and residual attentive modules to improve feature integration across space and time. Fully convolutional encoder-decoder architectures such as ViNet [19], TSFP [6], UNISAL [10] integrate multi-scale visual features for saliency prediction. Finally, transformer-based models including SalFoM [30], TMFINet [42], THTDNet [31] further extend this approach by leveraging global context modeling, showing strong performance on complex dynamic scenes.

# 2.2 Saliency prediction in event-based data

Few models have been developed in the context of event-based data. Iacono et al. (2019) [17] adapted a proto-object attention model for event-based data to enhance iCub humanoid robot visual processing capabilities. Using a bottom-up attention mechanism, the model processes event-based data through three layers: center-surround filtering, border ownership cells, and grouping cells. These layers decompose the visual scene into proto-objects, with saliency determined by contrast and edges. D’Angelo et al. (2022) [8] later implemented the protoobject attention model on SpiNNaker neuromorphic computing platform [11], effectively taking advantage of the asynchronous output of event cameras to reduce both latency and computational cost. Studies by Gruel et al. in 2022 [14] and by Bulzomi et al. in 2023 [3] demonstrate the advantages of neuromorphic attention for event-based data. Gruel’s work focused on leveraging attention for gesture recognition, while Bulzomi applied attention to pedestrian detection using a static event camera. Both approaches employed a spiking neural network that dynamically adapts to incoming event-based data, focusing on regions of high activity as meaningful areas while filtering irrelevant background noise. More recently, Simon Chane et al. [5] developed a pipeline used to compute a saliency score for each event as they occur. Their approach was tested on a dataset they captured from an event-based camera recording a dynamic visual scene.

# 3 Approach

The proposed network (Figure 1) is based on a pretrained Swin-T Transformer as encoder to extract multi-level spatiotemporal features from events input, followed by a CNN decoder to decode features into saliency maps. We deliberately keep the architecture minimal to establish a clean baseline; methodological elaboration is left to future work.

# 3.1 Event Representation

We convert the raw event stream, represented as a set of asynchronous events $\{ ( x _ { k } , y _ { k } , t _ { k } , p _ { k } ) \}$ , into a voxel grid representation. Events within each temporal bin are accumulated per pixel and per polarity, producing a count-based voxel grid of shape [T, 2, H, W], H=W=224. During training, samples are processed in batches of size B, giving $\mathbf { X } \in \mathbb { R } ^ { B \times T \times 2 \times H \times \mathbf { \breve { W } } }$ . This representation preserves both the temporal structure and the polarity asymmetry of the event stream.

![](images/459559625b2c3209dcb8a5b1a39ba8993a24b7a60ef02785f58a822fdb7241c5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Event frames"] --> B["Patch Partition"]
    B --> C["Conv Embedding"]
    C --> D["x2"]
    D --> E["Conv3d"]
    E --> F["Concatenation"]
    F --> G["1024 x T x 56 x 56"]
    G --> H["Conv3d BatchNorm3d LeakyRelu"]
    H --> I["Conv3d Upsample"]
    I --> J["Learnable center-bias prior"]
    J --> K["1 x T x 224 x 224"]
    K --> L["Hi4 x Wi4 x 48"]
    L --> M["Hi4 x Wi4 x C"]
    M --> N["Hi8 x Wi8 x 2C"]
    N --> O["Hi16 x Wi16 x 4C"]
    O --> P["Hi32 x Wi32 x BC"]
    P --> Q["Conv3d Upsample"]
    Q --> R["256 x T x 56 x 56"]
    R --> S["Conv3d BatchNorm3d LeakyRelu"]
    S --> T["Concatenation"]
    T --> U["1024 x T x 56 x 56"]
    U --> V["Conv3d Upsample"]
    V --> W["Conv3d Upsample"]
    W --> X["Conv3d + Upsample"]
    X --> Y["Conv3d + Upsample"]
    Y --> Z["Conv3d + Upsample"]
    Z --> AA["Conv3d + Upsample"]
    AA --> AB["Conv3d + Upsample"]
    AB --> AC["Conv3d + Upsample"]
    AC --> AD["Conv3d + Upsample"]
    AD --> AE["Conv3d + Upsample"]
    AE --> AF["Conv3d + Upsample"]
    AF --> AG["Conv3d + Upsample"]
```
</details>

Fig. 1. Overview of the proposed Swin Event-based Saliency Transformer (SEST) architecture.

# 3.2 Pretrained Swin Transformer

Yang et al. [41] introduced the first transformer solely trained with event-based data using a self-supervised scheme building on the DINOv2 framework. We leverage this pretrained backbone through end-to-end fine-tuning, inheriting rich event-specific representations learned from large-scale self-supervised training. Full pretraining details are described in [41].

# 3.3 3D Feature Extraction and Fusion

While the backbone is pretrained on 2D event images, dynamic saliency prediction requires reasoning over temporal sequences of event bins. To preserve the integrity of the pretrained 2D spatial representations, we treat each temporal bin as an independent spatial sample during encoding, while the temporal axis is explicitly recovered after feature extraction for subsequent 3D spatiotemporal reasoning.

Concretely, the batched input $\mathbf { X } \in \mathbb { R } ^ { B \times T \times 2 \times H \times W }$ is reshaped into a flattened spatial batch $\mathbf { X ^ { \prime } } \in \mathbb { R } ^ { ( \tilde { B } \cdot T ) \times 2 \times H \times W }$ before being fed into the encoder. This decoupling allows us to fully exploit the pretrained event representations without modification, while enabling the decoder to model cross-bin temporal dependencies that are essential for dynamic saliency prediction.

After processing, we extract hierarchical features from the four stages of the Swin-T Transformer. These features are reshaped back into 5D tensors $\{ \mathbf { F } _ { i } \} _ { i = 1 } ^ { 4 } ,$ , where $\mathbf { F } _ { i } \in \mathbb { R } ^ { B \times T \times C _ { i } \times H _ { i } \times W _ { i } }$ corresponds to the features from the $i ^ { \mathrm { t h } }$ Swin stage, restoring the temporal axis for subsequent 3D convolutional operations.

To align feature dimensions across scales, each stage output is processed by a $3 \times 3 \times 3$ Conv3D layer projecting channels to a fixed depth $d = 2 5 6$ . The $3 \times 3 \times 3$ kernel captures local spatiotemporal context across adjacent temporal bins at each scale, compensating for the temporal independence assumed during the 2D encoding stage and introducing cross-bin awareness before fusion. Features from stages 2, 3, and 4 are spatially upsampled to $5 6 \times 5 6$ via trilinear interpolation to match the resolution of stage 1:

$$
\mathbf {F} _ {i} ^ {\prime} = \left\{ \begin{array}{l l} \operatorname{Conv3D} _ {i} (\mathbf {F} _ {i}), & i = 1 \\ \text { Upsample } (\operatorname{Conv3D} _ {i} (\mathbf {F} _ {i}))  , & i \in \{2, 3, 4 \} \end{array} \right. \tag {1}
$$

The resulting features $\{ \mathbf { F } _ { i } ^ { \prime } \} _ { i = 1 } ^ { 4 }$ are concatenated along the channel dimension to form a unified spatiotemporal volume:

$$
\mathbf {U} = \operatorname{Concat} \left(\{\mathbf {F} _ {i} ^ {\prime} \} _ {i = 1} ^ {4}\right) \in \mathbb {R} ^ {B \times T \times 1 0 2 4 \times 5 6 \times 5 6} \tag {2}
$$

# 3.4 Feature Refinement and Saliency Map Reconstruction

The concatenated volume U aggregates complementary information across four levels of abstraction: fine-grained boundary and texture information from stage 1 through to high-level semantic content at stage 4. A 3D convolutional refinement block integrates these representations jointly across channel and temporal dimensions:

$$
\mathbf {Z} = \text { LeakyReLU } (\mathrm{BN} (\operatorname{Conv3D} (\mathbf {U}))) \tag {3}
$$

Unlike a 2D convolution, the 3D kernel explicitly models how multi-scale features co-evolve across time, capturing for instance that a salient object boundary at temporal bin t should remain spatially consistent with the semantic saliency response at the same location in adjacent bins. The channel dimension is simultaneously reduced from 1024 to $d \ : = \ : 2 5 6$ , into a compact representation Z ∈ RB×T ×256×56×56. $\mathbf { Z } \in \mathbb { R } ^ { B \times T }$

The refined features are projected to the target channel depth and upsampled to the original input resolution to produce a dense prediction:

$$
\hat {\mathbf {Y}} = \text { Upsample } (\text { Conv3D } (\mathbf {Z}))  , \quad \hat {\mathbf {Y}} \in \mathbb {R} ^ {B \times T \times 1 \times H \times W} \tag {4}
$$

To model spatial attention bias, we follow [7] and allow the network to learn a center-bias prior directly from training data rather than imposing a pre-defined prior. A learnable weight matrix $\mathbf { M } _ { b } \in \mathbb { R } ^ { H \times W }$ , initialized from a uniform distribution $\mathbf { M } _ { b } \sim \mathcal { U } ( 0 , 1 )$ , modulates the dense prediction via element-wise multiplication.

$$
\hat {\mathbf {Y}} _ {\text { biased }} = \hat {\mathbf {Y}} \odot (1 + \mathbf {M} _ {b}) \tag {5}
$$

By learning this bias from data rather than assuming a fixed Gaussian prior, the model adapts to the statistical regularities of event-based data specifically.

The final saliency map is obtained by applying Gaussian blur smoothing followed by a Sigmoid activation.

# 4 Experiments

# 4.1 Saliency datasets

We converted two existing video saliency datasets (DHF1K [38], and UCF Sports [28]) into event datasets (N-DHF1K and N-UCF Sports) using ESIM library [13] with positive and negative contrast thresholds of 0.09 and a 3 ms refractory period. We used their saliency and fixation maps as ground truths to train our model on the synthetic events.

DHF1K Dynamic Human Fixations 1K (DHF1K) is a large-scale dataset of gaze recordings from 17 observers who were asked to freely view the videos. It consists of 1K videos with diverse content, varied motion and various objects. The original dataset was partitioned into 600 training, 100 validation, and 300 test videos. Since the ground truth labels for the official test are hidden for benchmarking, we reserved 100 videos from the training set to serve our test set. This resulted in a final split of 500 training videos, 100 validation videos, and 100 test videos.

UCF Sports UCF Sports is a human fixations dataset on 150 videos from the UCF Sports Action dataset. The videos covered 9 action sports such as swinging, diving, and lifting. Contrary to DHF1K, where observers freely observed the videos, observers were asked to identify occurring actions within the UCF videos. Our final data split consists of 103 training videos, 15 validation videos, and a test set of 32 videos.

Although the proposed datasets are synthetically generated from RGB videos, this approach enables scalable supervision using high-quality human fixation data that is currently scarce in native event-based form. More importantly, prior work has shown that event representations generated from photorealistic videos preserve motion, contrast, and temporal saliency cues relevant to downstream tasks [13, 41].

# 4.2 Protocol

Implementation details SEST is implemented using the $\mathrm { P y }$ Torch framework. We use a batch size of 20 and a learning rate of 0.006. We train our model on a single NVIDIA H100 GPU for 30 epochs with an AdamW optimizer [27]. We implement early stopping based on the validation loss (patience = 3) and reduce the learning rate on plateau by a factor of 0.1 (patience = 1). We trained and evaluated our model using various temporal window sizes by varying the number of bins $( T \in \{ 7 , 1 0 , 1 4 , 2 1 \} )$ across the two datasets. The duration of each bin was fixed to match the sampling period of the original source videos. For N-UCF Sports (10Hz), the bin duration was set to 100ms, while for N-DHF1K (30Hz), it was set to 33.33ms. Consequently, the total observation window for each experiment scales linearly with the number of bins.

Loss function We considered different metrics to design our loss function, following common practice in the literature [4]. Let Y denote the ground truth saliency map and $\hat { Y }$ the predicted saliency map. The loss function is defined as:

$$
\mathcal {L} (\hat {Y}, Y) = \mathcal {L} _ {\mathrm{KL}} (\hat {Y}, Y) + \alpha_ {1} \mathcal {L} _ {\mathrm{CC}} (\hat {Y}, Y) + \alpha_ {2} \mathcal {L} _ {\mathrm{BCE}} (\hat {Y}, Y) \tag {6}
$$

with $\alpha _ { 1 } = 0 . 5$ and $\alpha _ { 2 } = 0 . 7$ set empirically. $\mathcal { L } _ { \mathrm { K L } } , \mathcal { L } _ { \mathrm { C C } }$ and $\mathcal { L } _ { \mathrm { B C E } }$ are Kullback-Leibler divergence, Pearson Correlation Coefficient, and Binary Cross Entropy respectively. ${ \mathcal { L } } _ { \mathrm { K L } }$ and ${ \mathcal { L } } _ { \mathrm { C C } }$ supervise the predicted saliency at the distribution level, while $\mathcal { L } _ { \mathrm { B C E } }$ enforces pixel-wise localization, balancing distributionmatching with spatial accuracy. We negate the Pearson correlation so that minimizing ${ \mathcal { L } } _ { \mathrm { C C } }$ maximizes the correlation between $Y$ and $\hat { Y }$ .

${ \mathcal { L } } _ { \mathrm { K L } }$ implements the Kullback-Leibler Divergence, which quantifies the dissimilarity between two probability distributions:

$$
\mathcal {L} _ {\mathrm{KL}} (\hat {Y}, Y) = \sum_ {x} Y (x) \log (\frac {Y (x)}{\hat {Y} (x)}) \tag {7}
$$

${ \mathcal { L } } _ { \mathrm { C C } }$ implements the Pearson Correlation Coefficient that measures the linear relationship between two images:

$$
\mathcal {L} _ {\mathrm{CC}} (\hat {Y}, Y) = - \frac {\sum_ {i} (\hat {Y} _ {i} - \bar {\hat {Y}}) (Y _ {i} - \bar {Y})}{\sqrt {\sum_ {i} (\hat {Y} _ {i} - \bar {\hat {Y}}) ^ {2}} \sqrt {\sum_ {i} (Y _ {i} - \bar {Y}) ^ {2}}} \tag {8}
$$

LBCE implements the Binary Cross Entropy that measures the pixel-wise dissimilarity between two images:

$$
\mathcal {L} _ {\mathrm{BCE}} (\hat {Y}, Y) = - \sum_ {i} \left[ Y _ {i} \log (\hat {Y} _ {i}) + (1 - Y _ {i}) \log (1 - \hat {Y} _ {i}) \right] \tag {9}
$$

Evaluation metrics We evaluate our model performance by employing the following metrics: AUC-Judd (AUC-J), Pearson Correlation Coefficient (CC), Similarity (SIM), and Normalized Scanpath Saliency (NSS). These metrics have been discussed in detail in [4].

# 5 Results and Analysis

# 5.1 Performance results

We compared the saliency prediction performance of our event-based model, SEST, on the N-DHF1K and N-UCF Sports datasets to the performance of two state-of-the-art event-based models (SNNevProto [8] and evST [5]) and several state-of-art RGB-based models from DHF1K and UCF Sports benchmarks1 (see Table 1).

![](images/4b7f14f53414726acd5d6e052312beda82e63063bfd808cb2dd25a82c9da241e.jpg)

<details>
<summary>text_image</summary>

Video Frames
Event Frames
GT
SNNevProto
evST
SEST (This work)
Time
Time
Time
601-02
601-03
601-04
601-05
601-06
601-07
601-08
601-09
601-10
601-11
601-12
601-13
601-14
601-15
601-16
601-17
601-18
601-19
601-20
601-21
601-22
601-23
601-24
601-25
601-26
601-27
601-28
601-29
601-30
601-31
601-32
601-33
601-34
601-35
601-36
601-37
601-38
601-39
601-40
601-41
601-42
601-43
601-44
601-45
601-46
601-47
601-48
601-49
601-50
601-51
601-52
601-53
601-54
601-55
601-56
601-57
601-58
601-59
601-60
601-61
601-62
601-63
601-64
601-65
601-66
601-67
601-68
601-69
601-70
601-71
601-72
601-73
601-74
601-75
601-76
601-77
601-78
601-79
601-80
601-81
601-82
601-83
601-84
601-85
601-86
601-87
601-88
601-89
601-90
601-91
601-92
601-93
601-94
601-95
601-96
601-97
601-98
601-99
602-02
</details>

Fig. 2. Illustration of qualitative results for three samples: UCF Sports Run-side-004 (rows 1-2), DHF1K 548 (rows 3-4), UCF Sports Swing-Bench-003 (rows 5-6).

N-DHF1K The best overall performance on this dataset is achieved by RGBbased models, with SalFoM leading across all four metrics. Other RGB-based models such as TMFI, THTD-Net, and STSANet follow closely. In contrast, existing event-based models show significantly inferior performance, with very low CC and NSS values. Our model, SEST, substantially improves performance within the event domain. Across different time bin configurations, SEST achieves AUC-J values up to 0.9197, CC up to 0.4907, SIM up to 0.3969, and NSS up to 2.3956, clearly outperforming all other event-based approaches by a large margin.

N-UCF Sports Similar to the case of DHF1K, RGB models achieve the highest overall performance on this dataset too. Among them, the best performance is generally achieved by STSANet and TMFI, with AUC-J values of 0.936 and strong NSS scores. In comparison, existing event-based models exhibit noticeably lower performance, with AUC-J values typically in the range of 0.77 to 0.81 and NSS scores mostly below 1.5, highlighting the gap that remains between RGBbased methods and current event-based methods. However, SEST consistently outperforms all other event-based models across all four metrics. In particular, our model attains the highest AUC-J value among the event-based approaches, along with improved CC, SIM and NSS scores. Notably, the 14-bin configuration of our model provides the best overall performance.

These results show that although a performance gap still exists between RGB and event-based methods, SEST significantly narrows this gap and sets a new state of the art for event-based saliency prediction on the N-DHF1K and N-UCF Sports datasets.

Table 1. Quantitative comparison on N-DHF1K and N-UCF Sports datasets. 

<table><tr><td colspan="7">N-DHF1K</td><td colspan="7">N-UCF Sports</td></tr><tr><td>Domain</td><td>Model</td><td>#bins</td><td>AUC-J↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>Domain</td><td>Model</td><td>#bins</td><td>AUC-J↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td></tr><tr><td rowspan="5">RGB</td><td>SalFoM [30]</td><td>-</td><td>0.9222</td><td>0.5692</td><td>0.4208</td><td>3.3536</td><td rowspan="5">RGB</td><td>TMFI [42]</td><td>-</td><td>0.936</td><td>0.707</td><td>0.565</td><td>3.863</td></tr><tr><td>TMFI [42]</td><td>-</td><td>0.9153</td><td>0.5461</td><td>0.4068</td><td>3.1463</td><td>STSANet [39]</td><td>-</td><td>0.936</td><td>0.705</td><td>0.560</td><td>3.908</td></tr><tr><td>THTD-Net [31]</td><td>-</td><td>0.9152</td><td>0.5479</td><td>0.4062</td><td>3.1385</td><td>SalSAC [40]</td><td>-</td><td>0.926</td><td>0.671</td><td>0.534</td><td>3.523</td></tr><tr><td>STSANet [39]</td><td>-</td><td>0.9125</td><td>0.5288</td><td>0.3829</td><td>3.0103</td><td>ViNet [19]</td><td>-</td><td>0.924</td><td>0.673</td><td>0.522</td><td>3.620</td></tr><tr><td>TSFP-Net [6]</td><td>-</td><td>0.9116</td><td>0.5168</td><td>0.3921</td><td>2.9665</td><td>TSFP-Net [6]</td><td>-</td><td>0.923</td><td>0.685</td><td>0.561</td><td>3.698</td></tr><tr><td rowspan="12">Event</td><td>evST [5]</td><td>21</td><td>0.5464</td><td>0.0274</td><td>0.1106</td><td>0.1420</td><td rowspan="12">Event</td><td>evST [5]</td><td>21</td><td>0.7756</td><td>0.2681</td><td>0.1464</td><td>1.3312</td></tr><tr><td>evST [5]</td><td>14</td><td>0.5408</td><td>0.0232</td><td>0.1208</td><td>0.1091</td><td>evST [5]</td><td>14</td><td>0.7779</td><td>0.2658</td><td>0.1382</td><td>1.3692</td></tr><tr><td>evST [5]</td><td>10</td><td>0.5413</td><td>0.0246</td><td>0.1290</td><td>0.1120</td><td>evST [5]</td><td>10</td><td>0.7793</td><td>0.2594</td><td>0.1308</td><td>1.3798</td></tr><tr><td>evST [5]</td><td>7</td><td>0.5380</td><td>0.0217</td><td>0.1317</td><td>0.0898</td><td>evST [5]</td><td>7</td><td>0.7787</td><td>0.2559</td><td>0.1263</td><td>1.3798</td></tr><tr><td>SNNevProto [8]</td><td>21</td><td>0.6427</td><td>0.0753</td><td>0.1092</td><td>0.3999</td><td>SNNevProto [8]</td><td>21</td><td>0.7890</td><td>0.2325</td><td>0.1336</td><td>1.0205</td></tr><tr><td>SNNevProto [8]</td><td>14</td><td>0.6201</td><td>0.0579</td><td>0.1185</td><td>0.2886</td><td>SNNevProto [8]</td><td>14</td><td>0.8195</td><td>0.2741</td><td>0.1369</td><td>1.2204</td></tr><tr><td>SNNevProto [8]</td><td>10</td><td>0.6036</td><td>0.0443</td><td>0.1266</td><td>0.2054</td><td>SNNevProto [8]</td><td>10</td><td>0.7941</td><td>0.2237</td><td>0.1144</td><td>1.0474</td></tr><tr><td>SNNevProto [8]</td><td>7</td><td>0.5884</td><td>0.0375</td><td>0.1342</td><td>0.1725</td><td>SNNevProto [8]</td><td>7</td><td>0.8316</td><td>0.2834</td><td>0.1262</td><td>1.3286</td></tr><tr><td>SEST (this work)</td><td>21</td><td>0.8956</td><td>0.4284</td><td>0.3014</td><td>2.1209</td><td>SEST (this work)</td><td>21</td><td>0.8605</td><td>0.4381</td><td>0.3090</td><td>2.1259</td></tr><tr><td>SEST (this work)</td><td>14</td><td>0.9009</td><td>0.4814</td><td>0.3641</td><td>2.2769</td><td>SEST (this work)</td><td>14</td><td>0.8943</td><td>0.5237</td><td>0.3800</td><td>2.6590</td></tr><tr><td>SEST (this work)</td><td>10</td><td>0.8947</td><td>0.4907</td><td>0.3969</td><td>2.2328</td><td>SEST (this work)</td><td>10</td><td>0.8942</td><td>0.5073</td><td>0.3462</td><td>2.6022</td></tr><tr><td>SEST (this work)</td><td>7</td><td>0.9197</td><td>0.4661</td><td>0.3634</td><td>2.3956</td><td>SEST (this work)</td><td>7</td><td>0.8841</td><td>0.4555</td><td>0.3214</td><td>2.4537</td></tr></table>

Qualitative analysis Figure 2 presents qualitative comparisons across three representative sequences: a high-motion athletics scene, a low-motion pedestrian scene, and a gymnastics scene. In all three cases, SEST produces spatially coherent, temporally consistent saliency maps that closely match the ground truth. SNNevProto generates diffuse activations dominated by background structure, failing to isolate salient subjects, while evST produces noisy, unstructured responses that deteriorate further in high-activity scenes. These observations highlight two consistent advantages of SEST over prior methods: robustness to varying event density, and temporal consistency across bins, both essential properties for reliable event-based saliency prediction in practice.

Generalizability and Transferability on real event dataset We evaluated the cross-dataset generalizability of SEST in two complementary settings (Table 2). First, we assess synthetic-to-synthetic transfer by training on one synthetic dataset and testing on the other. The results show that training on N-DHF1K generalizes better to N-UCF Sports than the other way around. This suggests that N-DHF1K, which contains more diverse scenes and motion patterns, provides a richer training distribution that transfers more effectively to N-UCF Sports. Conversely, the more domain-specific sports content in N-UCF Sports leads to reduced generalization when evaluated on the broader N-DHF1K dataset.

Second, to assess generalization beyond synthetic data, we conduct a zeroshot evaluation on 100 sequences (sampled at 30Hz, equally distributed across the 6 scene categories) from the real event camera dataset of [5], using 14 temporal bins and no fine-tuning. Despite the domain shift, SEST remains competitive: while evST obtains a higher NSS (1.4141), SEST achieves substantially higher CC and SIM (0.4725 and 0.4708), confirming generalizable saliency distributions. SNNevProto, in contrast, fails to generalise on AUC-J, CC, and NSS. SEST’s lower NSS likely reflects its sensitivity to fixation-level spatial displacement under domain shift, and the low-motion nature of the real dataset may favor evST’s asynchronous per-pixel design. This highlights a limitation of the current real dataset and motivates N-DHF1K and N-UCF Sports as essential resources, given their broader motion dynamics, for training robust event-based saliency models.

Table 2. Cross-dataset generalization. The top rows report synthetic-to-synthetic transfer between N-UCF Sports and N-DHF1K. The bottom rows report zero-shot evaluation on the real event camera dataset [5] (no fine-tuning). 

<table><tr><td>Model (Train / Test)</td><td>#bins</td><td>AUC-J ↑</td><td>CC ↑</td><td>SIM ↑</td><td>NSS ↑</td></tr><tr><td colspan="6">Synthetic-to-synthetic transfer</td></tr><tr><td>SEST (N-UCF Sports / N-DHF1K)</td><td>14</td><td>0.7831</td><td>0.2598</td><td>0.2246</td><td>1.2331</td></tr><tr><td>SEST (N-DHF1K / N-UCF Sports)</td><td>14</td><td>0.9015</td><td>0.5286</td><td>0.4256</td><td>2.6873</td></tr><tr><td colspan="6">Zero-shot on real event data [5]</td></tr><tr><td>SNNevProto [8] (- / real event data)</td><td>14</td><td>0.5763</td><td>0.0480</td><td>0.3059</td><td>0.1126</td></tr><tr><td>evST [5] (-/ real event data)</td><td>14</td><td>0.7848</td><td>0.2750</td><td>0.1396</td><td>1.4141</td></tr><tr><td>SEST (N-DHF1K/ real event data)</td><td>14</td><td>0.7725</td><td>0.4725</td><td>0.4708</td><td>1.1434</td></tr></table>

# 5.2 Computational Load

We evaluated the inference time for a single saliency map prediction across all three models on both GPU and CPU (Table 3), using an NVIDIA RTX A5500 (16GB) and an Intel Core i9-12950HX. Despite its larger memory footprint, SEST achieves the fastest inference on both hardware configurations (6.62ms GPU, 121.68ms CPU). The high inference times of competing methods reflect their differing design goals: SNNevProto was optimized for the SpiNNaker neuromorphic platform [11], where it achieves 16ms [8], while evST was designed for per-pixel saliency updates at microsecond precision rather than dense map reconstruction [5], making direct latency comparison with frame-based approaches inherently limited. Table 3 should therefore be read as a characterization of SEST’s computational profile on general-purpose hardware, not as a ranking of overall efficiency.

Table 3. Comparative inference times. 

<table><tr><td>Model</td><td>GPU Inference time (ms)</td><td>CPU Inference time (ms)</td><td>Parameters (M)</td><td>Model size (MB)</td></tr><tr><td>SNNevProto [8]</td><td>62.69</td><td>1396.17</td><td>-</td><td>-</td></tr><tr><td>evST [5]</td><td>-</td><td>2109.68</td><td>-</td><td>-</td></tr><tr><td>SEST (this work)</td><td>6.62</td><td>121.68</td><td>45.1</td><td>180.247</td></tr></table>

# 5.3 Ablation study

In this section, we assess the impact of two design choices on the N-UCF Sports test set: the learnable center bias component and the use of 2D convolutions in place of 3D convolutions within the decoder.

Training without center bias learning To evaluate the contribution of the spatial prior, we conducted an ablation study by training the architecture without the learnable center-bias component across all temporal configurations. The comparative results are detailed in Table 4. Our analysis indicates that the effectiveness of the center-bias is highly dependent on the temporal window length. At moderate temporal windows (10-14 bins), the center-bias significantly improves the CC and NSS metrics. We attribute this improvement to the module’s implementation as a multiplicative spatial gain, $\hat { Y } \odot ( 1 + \mathbf { M } _ { b } )$ , which allows the network to adaptively amplify salient features within statistically probable regions of interest.

However, at 21 bins, the effectiveness diminishes. We interpret this as a conflict between the learned global spatial prior and the increased local variance of longer sequences; as the temporal window expands, the salient target is more likely to deviate from the statistically learned center-bias regions, leading to a suppression of valid features. These results suggest the center-bias is most effective for short-to-medium temporal integration.

Replacing Conv3d with Conv2d : We also trained a variant of the model in which all 3D convolutions in the decoder are replaced by 2D convolutions, deferring the recovery of the temporal axis to the very last layer. When trained on 14 bins, this variant yields degraded performance (AUC-J: 0.8483, CC: 0.4072, SIM: 0.2415, NSS: 1.9544), as expected, since Conv2d treats each time bin independently and cannot exploit temporal correlations until the output. Reducing the input to a single bin provides only a marginal improvement (AUC-J: 0.8410, CC: 0.4179, SIM: 0.2946, NSS: 2.0522), still falling short of the Conv3d baseline. These results indicate that 2D convolutions are not well suited to the task given our pretrained backbone and underline the importance of explicitly modeling the temporal information of event-based data, a property that future work should further exploit to improve performance.

Table 4. SEST performance on N-UCF Sports without center bias learning. 

<table><tr><td>Model</td><td>#bins</td><td>AUC-J ↑</td><td>CC ↑</td><td>SIM ↑</td><td>NSS ↑</td></tr><tr><td>SEST (No Center bias)</td><td>21</td><td>0.8954</td><td>0.5190</td><td>0.3498</td><td>2.6149</td></tr><tr><td>SEST (No Center bias)</td><td>14</td><td>0.8847</td><td>0.4978</td><td>0.3795</td><td>2.5049</td></tr><tr><td>SEST (No Center bias)</td><td>10</td><td>0.8861</td><td>0.4799</td><td>0.3516</td><td>2.4906</td></tr><tr><td>SEST (No Center bias)</td><td>7</td><td>0.8999</td><td>0.4908</td><td>0.3595</td><td>2.4978</td></tr></table>

# 6 Conclusion

In this work, we addressed the largely unexplored problem of saliency prediction from event-based data. We proposed SEST, a transformer-based approach that leverages self-supervised pretrained event representations to model spatiotemporal saliency dynamics directly from event streams. To support supervised training and evaluation, we introduced two new event-based saliency datasets, N-DHF1K and N-UCF Sports, derived from large-scale RGB saliency benchmarks. While synthetically generated, these datasets enable scalable experimentation across diverse motion patterns and scene dynamics. Experimental results demonstrate that SEST significantly outperforms existing event-based saliency methods and narrows the performance gap with state-of-the-art RGB models. Zero-shot evaluation on a real event camera dataset further confirms that our synthetic benchmarks provide a transferable supervisory signal for real-world generalization.

Despite these encouraging results, the current architecture leverages temporal information only late in the decoder and remains too heavy for edge applications where event cameras typically excel. Future work will focus on explicit temporal attention over event bins and on lighter models through architectural redesign and compression techniques. By showing deep learning is viable for this task, we hope to open a new research direction at the intersection of event-based vision, saliency, and neuromorphic computing.

Acknowledgments. This work was supported by the project NAMED (ANR-23- CE45-0025-01) of the French National Research Agency (ANR). Training experiments presented in this paper were carried out using the Grid’5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations (see https://www.grid5000.fr).

# References

1. Bellitto, G., Proietto Salanitri, F., Palazzo, S., Rundo, F., Giordano, D., Spampinato, C.: Hierarchical domain-adapted feature learning for video saliency prediction. International Journal of Computer Vision (2021)   
2. Bruce, N., Tsotsos, J.: Saliency based on information maximization. NIPS (2005)   
3. Bulzomi, H., Gruel, A., Martinet, J., Fujita, T., Nakano, Y., Bendahan, R.: Object detection for embedded systems using tiny spiking neural networks: Filtering noise through visual attention. In: 2023 18th International Conference on Machine Vision and Applications (MVA). pp. 1–5. IEEE (2023)

4. Bylinskii, Z., Judd, T., Oliva, A., Torralba, A., Durand, F.: What Do Different Evaluation Metrics Tell Us About Saliency Models? IEEE TPAMI (2019)   
5. Chane, C.S., Niebur, E., Benosman, R., Ieng, S.H.: An event-based implementation of saliency-based visual attention for rapid scene analysis. IEEE Trans. on Cognitive and Developmental Systems (2024)   
6. Chang, Q., Zhu, S.: Temporal-spatial feature pyramid for video saliency detection (2021), https://arxiv.org/abs/2105.04213   
7. Cornia, M., Baraldi, L., Serra, G., Cucchiara, R.: A deep multi-level network for saliency prediction. In: ICPR (2016)   
8. D’Angelo, G., Perrett, A., Iacono, M., Furber, S., Bartolozzi, C.: Event driven bio-inspired attentive system for the iCub humanoid robot on SpiNNaker. Neuromorphic Computing and Engineering (2022)   
9. Djilali, Y.A.D., McGuinness, K., O’Connor, N.E.: Vision transformers are inherently saliency learners. In: BMVC (2023)   
10. Droste, R., Jiao, J., Noble, J.A.: Unified image and video saliency modeling. In: ECCV (2020)   
11. Furber, S., Bogdan, P.: Spinnaker-a spiking neural network architecture. Now Publishers (2020)   
12. Gao, D., Vasconcelos, N.: Discriminant saliency for visual recognition from cluttered scenes. NIPS 17 (2004)   
13. Gehrig, D., Gehrig, M., Hidalgo-Carrió, J., Scaramuzza, D.: Video to events: Recycling video datasets for event cameras. In: CVPR (June 2020)   
14. Gruel, A., Vitale, A., Martinet, J., Magno, M.: Neuromorphic event-based spatiotemporal attention using adaptive mechanisms. In: AICAS. IEEE (2022)   
15. Harel, J., Koch, C., Perona, P.: Graph-based visual saliency. NIPS 19 (2006)   
16. Hou, X., Zhang, L.: Dynamic visual attention: Searching for coding length increments. NIPS 21 (2008)   
17. Iacono, M., D’Angelo, G., Glover, A., Tikhanoff, V., Niebur, E., Bartolozzi, C.: Proto-object based saliency for event-driven cameras. In: IROS. IEEE (2019)   
18. Itti, L., Koch, C.: Computational modelling of visual attention. Nature reviews neuroscience 2(3) (2001)   
19. Jain, S., Yarlagadda, P., Jyoti, S., Karthik, S., Subramanian, R., Gandhi, V.: Vinet: Pushing the limits of visual modality for audio-visual saliency prediction. In: IROS. IEEE (2021)   
20. Jia, S., Bruce, N.D.: Eml-net: An expandable multi-layer network for saliency prediction. Image and vision computing 95 (2020)   
21. Jiang, L., Xu, M., Liu, T., Qiao, M., Wang, Z.: Deepvs: A deep learning based video saliency prediction approach. In: ECCV (September 2018)   
22. Kootstra, G., Nederveen, A., De Boer, B.: Paying attention to symmetry. In: BMVC (2008)   
23. Kruthiventi, S.S., Ayush, K., Babu, R.V.: Deepfix: A fully convolutional neural network for predicting human eye fixations. IEEE Trans. on Image Processing (2017)   
24. Lai, Q., Wang, W., Sun, H., Shen, J.: Video saliency prediction using spatiotemporal residual attentive networks. IEEE Trans. on Image Processing (2019)   
25. Le Meur, O., Le Callet, P., Barba, D., Thoreau, D.: A coherent computational approach to model bottom-up visual attention. IEEE TPAMI (2006)   
26. Linardos, P., Mohedano, E., Nieto, J.J., O’Connor, N.E., Giró-i-Nieto, X., McGuinness, K.: Simple vs complex temporal recurrences for video saliency prediction. In: BMVC (2019)

27. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)   
28. Mathe, S., Sminchisescu, C.: Actions in the eye: Dynamic gaze datasets and learnt saliency models for visual recognition. IEEE TPAMI (2014)   
29. Min, K., Corso, J.J.: Tased-net: Temporally-aggregating spatial encoder-decoder network for video saliency detection. In: ICCV (2019)   
30. Moradi, M., Moradi, M., Rundo, F., Spampinato, C., Borji, A., Palazzo, S.: Salfom: Dynamic saliency prediction with video foundation models. In: International Conference on Pattern Recognition. pp. 33–48. Springer (2024)   
31. Moradi, M., Palazzo, S., Spampinato, C.: Transformer-based video saliency prediction with high temporal dimension decoding. In: VISIGRAPP 2024 (2024)   
32. Murray, N., Vanrell, M.e.a.: Saliency estimation using a non-parametric low-level vision model. In: CVPR (2011)   
33. Navalpakkam, V., Itti, L.: An integrated model of top-down and bottom-up attention for optimizing detection speed. In: CVPR’06. vol. 2, pp. 2049–2056. IEEE (2006)   
34. Pan, J., Ferrer, C.C., et al.: Salgan: Visual saliency prediction with generative adversarial networks. Preprint arXiv:1701.01081 (2017)   
35. Pan, J., Sayrol, E., et al.: Shallow and deep convolutional networks for saliency prediction. In: CVPR (2016)   
36. Rodriguez, M.D., Ahmed, J., Shah, M.: Action mach a spatio-temporal maximum average correlation height filter for action recognition. In: CVPR (2008)   
37. Treisman, A.M., Gelade, G.: A feature-integration theory of attention. Cognitive psychology 12(1) (1980)   
38. Wang, W., Shen, J., Guo, F., Cheng, M.M., Borji, A.: Revisiting Video Saliency: A Large-Scale Benchmark and a New Model. In: CVPR (2018)   
39. Wang, Z., Liu, Z., Li, G., Wang, Y., Zhang, T., Xu, L., Wang, J.: Spatio-temporal self-attention network for video saliency prediction. IEEE Trans. on Multimedia 25 (2023)   
40. Wu, X., Wu, Z., Zhang, J., Ju, L., Wang, S.: Salsac: A video saliency prediction model with shuffled attentions and correlation-based convlstm. In: Proceedings of the AAAI conference on artificial intelligence. vol. 34, pp. 12410–12417 (2020)   
41. Yang, Y., Pan, L., Liu, L.: Event camera data dense pre-training. In: European Conference on Computer Vision. pp. 292–310. Springer (2024)   
42. Zhou, X., Wu, S., Shi, R., Zheng, B., Wang, S., Yin, H., Zhang, J., Yan, C.: Transformer-based multi-scale feature integration network for video saliency prediction. IEEE Transactions on Circuits and Systems for Video Technology (2023)