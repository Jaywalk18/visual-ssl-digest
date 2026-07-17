# ScanFocus: A Coarse-to-Fine Framework for Spatio-Temporal Video Grounding

Kai Chen<sup>∗</sup> , Ming Dai<sup>∗</sup> , Wenxuan Cheng , and Wankou Yang<sup>†</sup>

Southeast University

Abstract. Spatio-Temporal Video Grounding (STVG) aims to retrieve the visual trajectory of a specific object from a video stream as described by a natural language expression. However, most advanced methods struggle to balance global context modeling with precise boundary localization. Due to the prohibitive computational costs of processing long videos, these approaches typically resort to low-rate temporal downsampling and implicit motion modeling. This inevitably suppresses highfrequency boundary cues and neglects the explicit inter-frame dependencies required for precise boundary delineation. To address these limitations, we present ScanFocus, a novel coarse-to-fine framework that decouples the STVG task into a global spatio-temporal scan and a local boundary focus. Specifically, we utilize a unified vision-language fusion encoder combined with a lightweight Deformable Semantic-Motion Fusion module to eficiently align multimodal features and generate coarse proposals. To recover the suppressed fine-grained details, we introduce the Semantic-Guided Temporal Aggregator (SGTA) in the refinement stage. By densely sampling around coarse boundaries, SGTA explicitly models short-term temporal interactions under semantic guidance, capturing rapid motion changes for precise timestamp regression. Extensive experiments on three widely used benchmarks demonstrate the performance superiority of our proposed method over previous approaches. Code will be released at https://github.com/TenMinutes209/ScanFocus

Keywords: Spatio-Temporal Video Grounding · Coarse-to-fine · Multimodal Fusion

## 1 Introduction

The objective of Spatio-Temporal Video Grounding [59] is to retrieve the visual trajectory of a specific object from a video stream as described by a natural language expression. This dual-domain localization task demands not only robust spatial object detection but also precise temporal boundary delineation, making it a sophisticated frontier for evaluating cross-modal reasoning capabilities in dynamic environments.

![](images/60ef31a2bea184cb9ef79415671d47538053a58c355d200ec26523c13b6f212b.jpg)  
Fig. 1: Comparison of temporal boundary localization paradigms. (a) Existing Transformer-based methods often produce ambiguous boundaries due to the suppression of high-frequency temporal cues caused by global downsampling. (b) Our proposed method adopts a coarse-to-fine framework that first generates a coarse interval at a low frame rate, followed by boundary dense sampling to recover fine-grained details for precise localization.

Recently, capitalizing on the powerful attention mechanisms and compact pipelines of Transformers [44], researchers have introduced many Transformerbased architectures to address this challenge, achieving unprecedented results. By leveraging sophisticated encoder-decoders to comprehensively fuse visual and linguistic features, these methods [19,20,26,33,40,53] have significantly promoted the synergistic recognition of multimodal information and established new stateof-the-art (SOTA) performance in spatio-temporal alignment.

Despite these advances, existing Transformer-based STVG methods are inherently limited by the suppression of high-frequency cues. Due to the prohibitive computational costs of processing long video sequences, existing approaches [19, 20, 26, 53] typically resort to low-rate temporal subsampling designed for global semantic alignment. Consequently, high-frequency temporal boundary cues are discarded, and critical frames corresponding to the groundtruth boundaries are often entirely skipped during sampling. This physically prevents the model from accessing exact boundary features, leading to temporally ambiguous predictions, as visually explicated in Fig. 1(a).

This ambiguity is further exacerbated by a lack of explicit temporal modeling. To maintain eficiency under global modeling, current methods primarily focus on frame-level cross-modal interaction—fusing appearance, motion, and linguistic features within each individual frame. They rely heavily on motion features implicitly encoded by the backbone, lacking a dedicated mechanism to capture the fine-grained temporal dependencies between frames for precise boundary delineation. Moreover, forcing the encoder to simultaneously accommodate multi-modal features under the static MDETR [27] paradigm creates a significant optimization bottleneck. This overloaded fusion traps the model in suboptimal solutions, limiting high-precision temporal localization.

To empirically validate this temporal localization bottleneck, we conduct an oracle analysis on the HC-STVGv1 dataset (for other datasets, please kindly refer to Supplementary Material). As illustrated in Fig. 2, we compare our full model’s predictions against an Oracle setting where the spatial tube sequences are evaluated using Ground Truth (GT) timestamps. We observe a staggering performance leap: for instance, the vIoU@0.3 surges from 67.5% to 94.0%, and the vIoU@0.5 jumps from 42.2% to 86.9%. This substantial gap demonstrates that our framework already possesses highly accurate spatial grounding capabilities, confirming that the primary constraint on STVG is the temporal ambiguity.

![](images/a3522a98c992400af605b9c1b1f25894e8d0e915f5353036061bd287834f7d3c.jpg)  
Fig. 2: Oracle Experiment on HCSTVG-v1. Using GT timestamps instead of predicted intervals substantially improves performance, identifying temporal localization as the main bottleneck.

To address these limitations, in this paper, we introduce ScanFocus, a novel coarse-to-fine framework that decouples STVG into a global spatio-temporal scan and a local boundary focus. As illustrated in Fig. 1(b), mirroring the human visual strategy of global scanning followed by local scrutiny, our framework integrates sparse global retrieval with dense boundary refinement. This design efectively recovers suppressed high-frequency cues while establishing explicit inter-frame dependencies for precise temporal delineation. Specifically, we first utilize a unified vision-language fusion encoder combined with a Deformable Semantic-Motion Fusion encoder [63] to eficiently align multimodal features, followed by dual DETR-style decoders [9] to generate spatial tubes and coarse temporal intervals, respectively. Furthermore, to capture the high-frequency cues suppressed in the global scan, we introduce a Local Boundary Focus stage. Here, we perform dense boundary sampling around the predicted coarse boundaries and employ a Semantic-Guided Temporal Aggregator module to explicitly model short-term temporal dependencies within local windows. As a result, we recover the boundary details and refine the coarse proposals into precise start-end timestamps through the refine decoder. To sum up, our contributions include:

We propose ScanFocus, a coarse-to-fine framework that decouples STVG into a global spatio-temporal scan and a local boundary focus. This architecture efectively mitigates the suppression of high-frequency cues in global modeling by integrating sparse retrieval with dense boundary refinement.

– To resolve the lack of explicit inter-frame dependencies in existing methods, we introduce the Semantic-Guided Temporal Aggregator within the refinement stage. This module explicitly models short-term temporal interactions within densely sampled boundary windows, enabling precise regression of start and end timestamps.

– We conduct extensive experiments on three widely used benchmarks to demonstrate the performance superiority of our proposed method over previous SOTA approaches, validating the efectiveness of our strategy.

## 2 Related Work

## 2.1 Spatio-Temporal Video Grounding

Spatio-Temporal Video Grounding [59] aims to precisely localize a target of interest both spatially and temporally within an untrimmed video sequence, based on a given natural language description. Existing STVG methodologies have primarily evolved through several distinct paradigms. Early approaches [42, 58, 59] typically adopted a two-stage strategy, which first leveraged pre-trained object detectors to generate a redundant set of region proposals and subsequently selected the optimal candidate that best aligns with the textual query. However, these methods are often constrained by the quality of the external detector and sufer from high computational overhead. To mitigate these limitations, subsequent advancements [19, 20, 26, 33, 40, 50, 53] have shifted toward a unified one-stage framework. Inspired by the success of Transformers [44], these methods employ an end-to-end encoder-decoder architecture to directly regress the spatio-temporal tubelets without relying on any pre-defined proposals. By facilitating tighter cross-modal interaction and eliminating the heuristic proposal generation process, this paradigm has established new state-of-the-art performance. Most recently, an emerging paradigm has integrated Multimodal Large Language Models (MLLMs) [1,4,22,23,62] into STVG [21,30,45,60], harnessing their profound reasoning capabilities to further enhance fine-grained spatio-temporal understanding. However, due to the inherent ambiguity of temporal boundaries and the inevitable suppression of high-frequency boundary cues caused by temporal downsampling, existing models frequently struggle with imprecise temporal localization. To address these challenges, we propose a coarse-to-fine strategy to improve the temporal grounding performance.

## 2.2 Vision-Language Modeling

The paradigm of Vision-Language Modeling (VLM) is designed to bridge the semantic gap between the visual and textual modalities, enabling a unified understanding of heterogeneous data. Recently, it has been widely adopted in the field of visual grounding [10,12–15,27,28,31,61], visual question answering [3,25,51], visual captioning [18,38,39,54], video temporal grounding [2,8,16,17,29,32,34– 36,52,56], etc. In STVG, current mainstream frameworks [19,20,53] heavily rely on the MDETR [27] pipeline for cross-modal fusion. Despite their success, these methods typically struggle with an optimization bottleneck caused by forcing a spatially-aligned encoder to simultaneously model both temporal dynamics and motion cues. Such a paradigm fails to explicitly decouple the disparate feature spaces required for spatial and temporal grounding, leading to sub-optimal performance due to the compromised representation capability of the overloaded multimodal encoder. In contrast, our approach eliminates the reliance on this complex encoder-decoder architecture. By leveraging a vision-language fusion encoder [5, 48] to establish robust vision-language alignments, we reformulate the intricate tri-modal interaction into a streamlined cross-modal fusion task.

## 2.3 Video Temporal Grounding

The objective of video temporal grounding (VTG) [2] is to establish a finegrained alignment between linguistic descriptions and their corresponding temporal segments in untrimmed videos. VTG has been extensively studied through its two sub-tasks: moment retrieval (MR) and highlight detection (HD). Early methodologies [17, 52, 56] primarily relied on proposal-based paradigms, which first generated candidate segments via sliding windows or anchor mechanisms and then performed cross-modal ranking. While efective, these approaches often sufer from high redundancy and struggle with coarse temporal boundaries due to their reliance on predefined heuristics. In contrast, recent advancements have predominantly shifted toward end-to-end DETR-based frameworks [6,7,11, 24,29,32,35–37,46,57]. By utilizing learnable queries to directly regress temporal boundaries, these methods eliminate the need for hand-crafted proposals and facilitate global cross-modal reasoning. Subsequent studies, such as R2-Tuning [34] and FlashVTG [8], further refined this paradigm by incorporating multi-scale temporal modeling to capture events of diverse durations. However, STVG [59] presents a more formidable challenge than VTG, as it extends the grounding task from a single temporal axis to a joint spatio-temporal manifold, requiring a synergistic understanding of both object appearance and motion evolution.

## 3 Method

In this section, we first formulate the spatio-temporal video grounding task and review the prevalent MDETR-based framework in Sec. 3.1. In Sec. 3.2, we give an overview of our framework. Then, we detail the Global Spatio-Temporal Scan in Sec. 3.3 and the Local Boundary Focus in Sec. 3.4. In Sec. 3.5, we describe the training objectives and optimization strategy.

## 3.1 Preliminary

Formulation. Given a video sequence $V \in \mathbb { R } ^ { T \times H \times W \times 3 }$ and a natural language query D, the goal of STVG is to precisely output a spatio-temporal tube (a bounding box sequence with temporal boundaries) grounding D in V .

![](images/698abb45195975055aea4c2f666ab5f4d0d2d48bc33f60d5c94b08306128fb38.jpg)  
Fig. 3: Overview architecture of our proposed ScanFocus. The framework follows a coarse-to-fine paradigm, decoupling the task into two stages: 1) Global Spatio-Temporal Scan: We first utilize a unified vision-language fusion encoder combined with a lightweight Semantic-Motion Fusion Encoder to eficiently align multimodal features. Dual DETR-style decoders are then employed to generate coarse spatial tubes and temporal intervals. 2) Local Boundary Focus: To recover high-frequency cues suppressed in the coarse stage, we perform dense sampling around the predicted coarse boundaries. The Semantic-Guided Temporal Aggregator explicitly models short-term dependencies within these local windows, which are finally fed into dual refine decoders to predict precise start and end timestamps.

As illustrated in Fig. 4(a), prevailing STVG frameworks [19, 20, 26, 53] typically build upon the MDETR [27] architecture. They first extract static appearance $\left( F _ { a } \right)$ , linguistic $\left( F _ { t } \right)$ , and temporal motion $\left( F _ { m } \right)$ features using independent MDETR pre-trained backbones. Subsequently, a heavy MDETR pretrained standard transformer encoder is applied to project and couple these heterogeneous representations within a shared latent space:

$$
\hat {F} _ {a}, \hat {F} _ {m}, \hat {F} _ {t} = \mathrm{Encoder} (\mathrm{Concat} (\mathrm{Proj} (F _ {a}, F _ {m}, F _ {t}))),\tag{1}
$$

where Proj(·) denotes linear projections for channel alignment.

Finally, task-specific decoders are employed to predict the spatio-temporal locations. The MDETR pre-trained spatial locator regresses bounding box sequences B, while an auxiliary temporal decoder concurrently predicts the temporal boundaries (s, e).

## 3.2 Overview

To address the limitations described above, we propose ScanFocus. Fig. 3 illustrates the overall architecture. Diferent from previous methods that rely on heavy static pre-training, ScanFocus adopts a coarse-to-fine philosophy. The <sup>ROIPool Proj.</sup>pipeline begins with a unified vision-language fusion encoder initialized by video-??<sub>??</sub> <sub>Temporal</sub>language pre-training to extract aligned features, followed by a lightweight Deformable Semantic-Motion Encoder to eficiently fuse semantic and motion in-S Slice&Interp.formation. The encoded features are then processed in two cascaded stages: 1) The Global Spatio-Temporal Scan stage (Sec. 3.3), which utilizes dual decoders to generate coarse spatial tubes and temporal intervals from sparse global queries; 2) The Local Boundary Focus stage (Sec. 3.4), which performs dense boundary sampling and employs a Semantic-Guided Temporal Aggregator to capture fine-grained inter-frame dependencies for precise boundary refinement.

![](images/4754e52e44cc8f5603127de4a5232a3abae9b40b0ce809c279bdcd77e720a5e5.jpg)  
Fig. 4: Comparison of multi-modal fusion mechanisms. (a) Existing methods fuse vision, language, and motion simultaneously. This fully-coupled tri-modal alignment is inherently challenging and yields suboptimal representations on limited ROIPool Proj. Adddownstream datasets. (b) We adopt a decoupled strategy leveraging a general vision-<sub>Temporal??</sub>language encoder to extract visual and linguistic features pre-aligned on massive up-<sup>Self-attn</sup> stream datasets, followed by a lightweight Semantic-Motion Fusion Encoder to further <sup>????</sup>align these robust semantic representations with motion dynamics.

## 3.3 Coarse: Global Spatio-Temporal Scan

Feature Extraction. As shown in Fig. 3, given the redundancy in adjacent video frames and the need for eficient global modeling, we first perform a sparse sampling strategy on the input video. Specifically, we uniformly sample $T _ { c }$ frames from the original video sequence with a low sampling rate $f _ { c }$ to construct the coarse input, denoted as $\bar { V _ { c } } \in \mathbb { R } ^ { T _ { c } \times H \times W \times 3 }$

Subsequently, to bypass the optimization bottleneck of task-specific pretraining (Fig. 4(a)), we leverage general pre-trained foundation models for feature extraction, as illustrated in Fig. 4(b), we adopt a unified Vision-Language Fusion encoder (VLF) to extract appearance and linguistic features from these sampled frames. Unlike traditional vision-language paradigms that employ separate unimodal encoders to extract independent appearance and linguistic features followed by a cross-modal fusion module, our VLF is inherently unified. Pre-trained on massive-scale image-text pairs, it implicitly aligns visual and textual representations within a shared semantic space during the encoding phase, eliminating the domain discrepancy often observed in late-fusion architectures. Specifically, we use BEiT-3 [48] to extract appearance, linguistic, and [CLS] features, denoted as $F _ { a } \in \mathbb { R } ^ { T _ { c } \times N _ { a } \times C _ { a } } , F _ { l } \in \mathbb { R } ^ { \hat { T } _ { c } \times L \times C _ { a } }$ , and $\bar { F } _ { c } \in \mathbb { R } ^ { T _ { c } \times 1 \times \mathsf { \bar { C } } _ { a } }$ , where $N _ { a }$ and L denote the token counts, and Ca $C _ { a }$ represents the feature dimension.

For motion feature extraction, following previous methods in [19, 20], we introduce an video encoder to explicitly extract motion features. Specifically, we employ a frozen pre-trained VideoMAE [43] to generate the motion representation $F _ { m } \in \mathbb { R } ^ { \tilde { T } \times N _ { m } \times C _ { m } }$ , where $N _ { m }$ and $C _ { m }$ are the motion token count and channel dimension, respectively.

Semantic-Motion Fusion. Previous STVG methods [19, 20, 26, 53] often rely on heavy, fully-coupled Transformer encoders for multi-modal interaction. In contrast, leveraging the inherent cross-modal alignment of our VLF encoder, we propose a lightweight Deformable Semantic-Motion Fusion module that reformulates this interaction as a multi-scale deformable attention task. Specifically, we project all features into a shared dimension $C$ to form a multi-level feature pyramid $\mathcal { X } = \{ F _ { a } , F _ { m } , F _ { t } , F _ { c } \}$ , and define the concatenated query sequence as $\mathbf { Z } \in \mathbb { R } ^ { N _ { t o t a l } \times C }$ . For each query token $z _ { q } \in \mathbf { Z }$ , the fused representation is computed by sparsely sampling from $\mathcal { X } \mathrm { : }$

$$
\operatorname{Fusion} \left(z _ {q}\right) = \sum_ {l = 1} ^ {4} \sum_ {k = 1} ^ {K} A _ {l q k} \cdot \mathbf {W} _ {v} \cdot \Phi \left(\mathcal {X} ^ {l}; p _ {q} + \Delta p _ {l q k}\right),\tag{2}
$$

where $p _ { q } , \varDelta p _ { l q k }$ , and $A _ { l q k }$ denote the reference point, learnable ofset, and attention weight, respectively. This mechanism enables linguistic tokens to sparsely attend to salient motion or appearance regions with linear complexity. Finally, after $L _ { e n c }$ layers of stacked fusion, the unified sequence is disentangled into enriched modality streams:

$$
F _ {a} ^ {\prime}, F _ {m} ^ {\prime}, F _ {t} ^ {\prime}, F _ {c} ^ {\prime} = \mathrm{Split} \left(\varPhi_ {\mathrm{D}} \left(\mathrm{Concat} (F _ {a}, F _ {m}, F _ {t}, F _ {c})\right)\right),\tag{3}
$$

where $\varPhi _ { \mathrm { D } }$ denotes the stacked deformable encoder layers. Split(·) recovers the individual modality streams based on their original token lengths.

Global Spatio-Temporal Decoder. Following established practices [19, 20], we employ separate spatial and temporal decoders. To accelerate convergence and improve localization, we propose a semantic-injection strategy to explicitly incorporate linguistic and global contexts into the queries prior to decoding. Specifically, we initialize learnable queries $Q _ { s } , Q _ { t } \in \bar { \mathbb { R } } ^ { T _ { c } \times C }$ for the spatial and temporal branches. We then sequentially inject semantic information from the text features $F _ { t } ^ { ' }$ and the global context token $F _ { c } ^ { ' }$ into these queries via standard cross-attention layers, yielding semantically conditioned queries $Q _ { s } ^ { i }$ and $Q _ { t } ^ { i }$ . This initialization ensures that the queries are target-aware before interacting with the visual $( F _ { a } ^ { ' } )$ and motion $( \boldsymbol { F } _ { m } ^ { \prime } )$ features. With these conditioned queries, the decoders perform iterative reasoning through K stacked layers. Finally, taskspecific prediction heads generate the coarse outputs: a 3-layer MLP predicts the bounding box sequence $\mathbf { B } _ { c } \in \mathbb { R } ^ { T _ { c } \times 4 }$ , while a temporal head outputs framewise start/end probabilities to form the coarse temporal interval $\tau _ { c } = ( t _ { c s } , t _ { c e } )$

## 3.4 Refine: Local Boundary Focus

Dense Boundary Sampling. To recover the fine-grained motion details that suppressed by the sparse global scan, we introduce a boundary-focused dense<sup>Standard</sup> <sup>Transformer</sup> <sup>Encoder</sup> General Pretrained sampling strategy. Given the predicted coarse interval<sup>????</sup> <sup>=</sup> <sup>????,</sup> <sup>????,</sup> <sup>????</sup> ???? ???? ???? ???? $\tau _ { c } .$ , we construct two local<sub>Randomly</sub> <sub>Initialized</sub> observation windows centered at $t _ { c s }$ and $t _ { c e }$ . Distinct from the global stage, we sample <sub>mp.</sub> <sub>Enc.</sub> $N _ { w }$ frames within each window using a high sampling rateV-L Encoder Temporal Encoder Enc. Tex. Enc. $f _ { r } = k _ { r } \cdot f _ { c }$ (where $k _ { r } ~ > ~ 1$ is the density factor) to ensure dense temporal coverage. This yields two dense frame sequences: $\nu _ { r s }$ and $\nu _ { r e }$ , both of shape $\mathbb { R } ^ { N _ { w } \times H \times \smile }$

To maintain semantic consistency and maximize parameter eficiency, we strategically reuse the feature extraction and fusion modules from the coarse stage (Sec. 3.3). Specifically, $\gamma _ { r s }$ and $\nu _ { r e }$ are independently processed by the shared vision-language fusion encoder, Video Encoder, and the Semantic-MotionROIPool Proj. Add Fusion module. This yields refined multi-modal feature sequences<sub>Temporal</sub> <sup>Hadamard</sup> <sup>P.</sup><sub>??</sub> <sup>????</sup> $F _ { r s } , F _ { r e }$ , encoding rich boundary details for precise delineation.<sub>??</sub>

Semantic-Guided Temporal Aggregator. While the shared encoder efectively extracts frame-wise representations, precise boundary delineation inherently requires capturing fine-grained temporal evolution, which relies on explicit inter-frame reasoning. To bridge this gap, we introduce SGTA. Leveraging the reduced temporal footprint of the local windows $( N _ { w } )$ , SGTA performs eficient, dense temporal modeling to capture high-frequency motion

![](images/67a752eb000383b8897dfaafed370f75a49c1fd94bfef41d7f9475ae8f1572db.jpg)  
Fig. 5: Detailed architecture of SGTA module.

cues omitted in the coarse stage. As shown in Fig. 5, taking the start branch as an example, SGTA first generates a dense spatial prior for the local window. Given the coarse bounding box sequence $\mathbf { B } _ { c }$ and the predicted start timestamp $t _ { c s }$ , we retrieve the subset of coarse boxes corresponding to the temporal span of the refinement window. Since this window contains $N _ { w }$ frames sampled at a high rate $f _ { r }$ , we perform temporal linear interpolation on this sparse subset to upsample it to the target length $N _ { w }$ . This eficiently generates the dense spatial prior $\bar { \mathbf B } _ { s t } \in \mathbb R ^ { N _ { w } \times 4 }$ without requiring heavy external object detectors. With the refined feature sequence $F _ { r s }$ obtained from the shared encoder, we isolate its distinct modality streams: appearance $\left( F _ { r s } ^ { a } \right)$ , motion $\left( F _ { r s } ^ { m } \right)$ , and text $\left( F _ { r s } ^ { t } \right)$ . To rigorously align the visual representations with the tracked object, we employ RoI Pooling on the appearance features guided by the spatial prior $\mathbf { B } _ { s t }$ . This yields the object-aligned appearance sequence $F _ { p s } ^ { a }$ , which focuses exclusively on the target regions:

$$
F _ {p s} ^ {a} = \mathrm{ROIPool} (F _ {r s} ^ {a}, \mathbf {B} _ {s t}).\tag{4}
$$

Subsequently, to explicitly align the temporal dynamics with the target object and linguistic query, we utilize these semantic features to guide the motion representation. To filter out irrelevant background dynamics, the motion features

$F _ { r s } ^ { m }$ are modulated by the object-aligned appearance features $F _ { p s } ^ { a } ,$ efectively highlighting motion patterns associated with the visual referent. Furthermore, the text features $F _ { r s } ^ { t }$ are injected as a linguistic bias. The final semantic-guided motion feature $F _ { s g }$ is formulated as:

$$
F _ {s g} = \left(\phi_ {a} (F _ {p s} ^ {a}) \odot F _ {r s} ^ {m}\right) \oplus \phi_ {t} (F _ {r s} ^ {t}),\tag{5}
$$

where $\odot$ and ⊕ denote the Hadamard product and element-wise addition, respectively. $\phi _ { a }$ and $\phi _ { t }$ represent learnable linear projections. Finally, to capture high-frequency cues and model explicit inter-frame dependencies, we flatten $F _ { s g }$ into a unified sequence of length $N _ { w } \times N _ { m }$ (where $N _ { m }$ denotes the number of motion tokens) and process it through a stack of $L _ { t }$ temporal self-attention layers:

$$
F _ {r s} ^ {\prime} = \mathrm{MHSA} _ {\times L _ {t}} (\text { Flatten } (F _ {s g})).\tag{6}
$$

This flattening structure ensures that all tokens within the local window are mutually perceptible, enabling fine-grained global reasoning across the boundary region. The resulting $F _ { r s } ^ { ' }$ efectively aggregates dense boundary details, serving as the input for the subsequent prediction heads.

Refine Decoder. Similar to the coarse stage, a DETR-style decoder $\mathrm { D e c } _ { r s }$ is employed to probe the aggregated features $F _ { r s } ^ { ' }$ . The decoded embeddings are then directly projected by task-specific prediction heads to generate the framewise boundary probability $\mathbf { P } _ { s t }$ and the auxiliary action confidence $\mathbf { A } _ { s t }$ . The end branch follows an identical procedure to obtain $\mathbf { P } _ { e d }$ and $\mathbf { A } _ { e d }$

## 3.5 Optimization

To ensure optimization stability and prevent fine-grained gradients from perturbing well-learned global priors, we adopt a decoupled two-stage training paradigm. For coarse stage, we first train the global modules (backbone, fusion encoder, and coarse decoders) using $\mathcal { L } _ { c }$ . The spatial decoder is supervised by L1 and GIoU losses given the ground-truth box sequence $\mathbf { B } ^ { \ast }$ , while the temporal decoder uses KL divergence to align the predicted distributions $\hat { P } _ { s / e }$ with Gaussian-smoothed boundary labels $t _ { s / e } ^ { * }$

$$
\mathcal {L} _ {c} = \lambda_ {b o x} (\mathcal {L} _ {L 1} (\mathbf {B} _ {c}, \mathbf {B} ^ {*}) + \mathcal {L} _ {I o U} (\mathbf {B} _ {c}, \mathbf {B} ^ {*})) + \lambda_ {t m p} (\mathcal {L} _ {K L} (\hat {P} _ {s}, t _ {s} ^ {*}) + \mathcal {L} _ {K L} (\hat {P} _ {e}, t _ {e} ^ {*})).\tag{7}
$$

For refine stage, we freeze the coarse stage parameters and exclusively optimize the refinement modules (SGTA and Refine Decoders). We formulate boundary localization and action detection as binary classification tasks, optimized via Binary Cross-Entropy (BCE) loss:

$$
\mathcal {L} _ {r} = \lambda_ {r e f} \mathcal {L} _ {B C E} (\mathbf {P} _ {s t / e d}, \mathbf {y} _ {s t / e d} ^ {*}) + \lambda_ {a c t} \mathcal {L} _ {B C E} (\mathbf {A} _ {s t / e d}, \mathbf {y} _ {a c t} ^ {*}),\tag{8}
$$

where $\mathbf { P } _ { s t / e d }$ and $\mathbf { A } _ { s t / e d }$ are the predicted boundary and action probabilities, and $\mathbf { y } ^ { * }$ are the corresponding binary ground-truth labels. The hyperparameters $\lambda$ balance their respective loss terms.

## 4 Experiments

## 4.1 Experiment Settings

Datasets. The commonly used datasets in spatio-temporal video grounding are HC-STVGv1/v2 and VidSTG. HC-STVGv1 comprises 5,660 untrimmed video clips, which are partitioned into 4,500 training and 1,160 testing video-text pairs. HC-STVGv2 significantly expands upon this scale to include a total of 16,544 samples, subdivided into 10,131 samples for training, 2,000 for validation, and 4,413 for testing. Due to the unavailability of ground-truth annotations for the test set, we report results on the validation set, following prior work. VidSTG provides a more substantial corpus for evaluation, encompassing 99,943 videotext pairs derived from 5,436 source videos. This dataset is characterized by its high linguistic diversity, featuring a mix of 44,808 declarative sentences and 55,135 interrogative queries. The data is organized into training, validation, and testing subsets with 80,684, 8,956, and 10,303 sentences respectively, which are mapped across 5,436, 602, and 732 unique video sequences.

Metrics. Following prior work, we employ m\_tIoU, m\_vIoU, and vIoU@R as evaluation metrics. m\_tIoU measures the ability of temporal grounding by computing the average temporal Intersection-over-Union (tIoU) score across all the test set. m\_vIoU assesses spatial grounding quality by computing the average IoU across space and time between predicted and annotated spatio-temporal tubes, and vIoU@R measures the performance using ratios of samples with vIoU greater than R in test sets. For detailed metrics, please kindly refer to [53].

Implementation. Our model employs BEiT-3 [48] for vision-language modeling and VideoMAE [43] for motion representation. Inputs are resized to 384×384. We sample 64 coarse frames and 8 refine frames $\left( k _ { r } = 2 \right)$ . Detailed network configurations and loss weights are provided in the Supplementary Material.

## 4.2 Quantitative Results

Results on HC-STVGv1/v2. To validate the eficacy of our method, we compare it against state-of-the-art approaches on the HC-STVGv1 and v2 datasets. Table 1 and Table 2 present the comparison results on the respective test sets.

As shown in Table 1, ScanFocus significantly outperforms existing methods across all metrics on HC-STVGv1. Notably, on the strict localization metrics vIoU@0.3 and vIoU@0.5, our method achieves 67.5% and 42.2%, surpassing the previous best method TA-STVG [20] by 4.4% and 5.4%, respectively. Similarly, on the HC-STVGv2 dataset (Table 2), ScanFocus establishes new state-of-theart performance, achieving a 2.0% improvement in tiou and a 2.6% improvement in vIoU@0.5 over TA-STVG. These consistent improvements demonstrate the superiority of our coarse-to-fine paradigm in recovering high-frequency boundary cues for precise temporal delineation.

Table 1: Comparison on HC-STVGv1. The best results are highlighted in bold.

<table><tr><td>Methods</td><td>m_tIoU</td><td>m_vIoU</td><td>vIoU@0.3</td><td>vIoU@0.5</td></tr><tr><td>STVGBert [40]</td><td>-</td><td>20.4</td><td>29.4</td><td>11.3</td></tr><tr><td>TubeDETR [53]</td><td>43.7</td><td>32.4</td><td>49.8</td><td>23.5</td></tr><tr><td>STCAT [26]</td><td>49.4</td><td>35.1</td><td>57.7</td><td>30.1</td></tr><tr><td>SGFDN [47]</td><td>46.9</td><td>35.8</td><td>56.3</td><td>37.1</td></tr><tr><td>STVGFormer [33]</td><td>-</td><td>36.9</td><td>62.2</td><td>34.8</td></tr><tr><td>VG-DINO [50]</td><td>-</td><td>38.3</td><td>62.5</td><td>36.1</td></tr><tr><td>CG-STVG [19]</td><td>52.8</td><td>38.4</td><td>61.5</td><td>36.3</td></tr><tr><td>TA-STVG [20]</td><td>53.0</td><td>39.1</td><td>63.1</td><td>36.8</td></tr><tr><td>ScanFocus (Ours)</td><td>55.5</td><td>41.8</td><td>67.5</td><td>42.2</td></tr></table>

Table 2: Comparison on HC-STVGv2. The best results are highlighted in bold.

<table><tr><td>Methods</td><td>m_tIoU</td><td>m_vIoU</td><td>vIoU@0.3</td><td>vIoU@0.5</td></tr><tr><td>PCC [55]</td><td>-</td><td>30.0</td><td>-</td><td>-</td></tr><tr><td>2D-Tan [41]</td><td>-</td><td>30.4</td><td>50.4</td><td>18.8</td></tr><tr><td>MMN [49]</td><td>-</td><td>30.3</td><td>49.0</td><td>25.6</td></tr><tr><td>TubeDETR [53]</td><td>53.9</td><td>36.4</td><td>58.8</td><td>30.6</td></tr><tr><td>STVGFormer [33]</td><td>58.1</td><td>38.7</td><td>65.5</td><td>33.8</td></tr><tr><td>VG-DINO [50]</td><td>-</td><td>39.9</td><td>67.1</td><td>34.5</td></tr><tr><td>CG-STVG [19]</td><td>60.0</td><td>39.5</td><td>64.5</td><td>36.3</td></tr><tr><td>TA-STVG [20]</td><td>60.4</td><td>40.2</td><td>65.8</td><td>36.7</td></tr><tr><td>ScanFocus (Ours)</td><td>62.4</td><td>41.7</td><td>68.4</td><td>39.3</td></tr></table>

Table 3: Comparison on VidSTG. The best results are highlighted in bold.

<table><tr><td rowspan="2">Methods</td><td colspan="4">Declarative Sentences</td><td colspan="4">Interrogative Sentences</td></tr><tr><td>m_tIoU</td><td>m_vIoU</td><td>vIoU@0.3</td><td>vIoU@0.5</td><td>m_tIoU</td><td>m_vIoU</td><td>vIoU@0.3</td><td>vIoU@0.5</td></tr><tr><td>STGRN [59]</td><td>48.5</td><td>19.8</td><td>25.8</td><td>14.6</td><td>47.0</td><td>18.3</td><td>21.1</td><td>12.8</td></tr><tr><td>OMRN [58]</td><td>50.7</td><td>23.1</td><td>32.6</td><td>16.4</td><td>49.2</td><td>20.6</td><td>28.4</td><td>14.1</td></tr><tr><td>STGVT [42]</td><td>-</td><td>21.6</td><td>29.8</td><td>18.9</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>STVGBert [40]</td><td>-</td><td>24.0</td><td>30.9</td><td>18.4</td><td>-</td><td>22.5</td><td>26.0</td><td>16.0</td></tr><tr><td>TubeDETR [53]</td><td>48.1</td><td>30.4</td><td>42.5</td><td>28.2</td><td>46.9</td><td>25.7</td><td>35.7</td><td>23.2</td></tr><tr><td>SGFDN [47]</td><td>45.1</td><td>28.3</td><td>41.7</td><td>29.1</td><td>44.8</td><td>25.8</td><td>36.9</td><td>23.9</td></tr><tr><td>STVGFormer [33]</td><td>-</td><td>33.7</td><td>47.2</td><td>32.8</td><td>-</td><td>28.5</td><td>39.9</td><td>26.2</td></tr><tr><td>STCAT [26]</td><td>50.8</td><td>33.1</td><td>46.2</td><td>32.6</td><td>49.7</td><td>28.2</td><td>39.2</td><td>26.6</td></tr><tr><td>CG-STVG [19]</td><td>51.4</td><td>34.0</td><td>47.7</td><td>33.1</td><td>49.9</td><td>29.0</td><td>40.5</td><td>27.5</td></tr><tr><td>TA-STVG [20]</td><td>51.7</td><td>34.4</td><td>48.2</td><td>33.5</td><td>50.2</td><td>29.5</td><td>41.5</td><td>28.0</td></tr><tr><td>SpaceVLLM-7B [45]</td><td>47.7</td><td>27.4</td><td>39.1</td><td>26.2</td><td>48.5</td><td>25.4</td><td>35.9</td><td>22.2</td></tr><tr><td>ASTG [60]</td><td>45.6</td><td>29.2</td><td>40.3</td><td>27.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ScanFocus (Ours)</td><td>53.3</td><td>36.5</td><td>50.6</td><td>36.0</td><td>51.4</td><td>30.7</td><td>42.5</td><td>29.4</td></tr></table>

Results on VidSTG. To further evaluate the generalization capability of Scan-Focus, we conduct experiments on the VidSTG dataset, which is significantly more challenging due to its diverse linguistic expressions (i.e., declarative and interrogative sentences) and complex temporal variations. As summarized in Table 3, ScanFocus consistently outperforms existing state-of-the-art methods across both sentence types.

## 4.3 Ablation Studies

In this section, we perform ablation experiments to investigate the impact of key components in ScanFocus. Unless otherwise specified, all experiments are conducted on the HC-STVGv1 dataset with an input resolution of 224 × 224. For brevity, tiou and viou are used to represent m\_tIoU and m\_vIoU.

Component-wise Efectiveness. As shown in Table 4, the baseline global scan model achieves a tIoU of 50.9%. Simply introducing the Dense Sampling (DS) strategy with refine decoders improves tIoU to 52.4% by recovering highfrequency temporal cues; however, the slight drop in vIoU@0.5 suggests that raw dense frames may introduce redundancy without explicit modeling. Finally, incorporating the proposed SGTA module yields the best performance across all metrics, demonstrating its ability to efectively aggregate dense boundary information through semantic-guided reasoning.

Table 4: Component-wise efectiveness analysis. Evaluated on HC-STVGv1. DS means refine stage with multi-modal fusion and start/end decoders.  
Table 5: Impact of Window Size $N _ { w } .$ Performance of ScanFocus varies with diferent window sizes.

<table><tr><td>Method</td><td>tIoU</td><td>vIoU</td><td>vIoU@0.3</td><td>vIoU@0.5</td></tr><tr><td>Coarse</td><td>50.9</td><td>38.2</td><td>60.7</td><td>38.0</td></tr><tr><td>+ DS</td><td>52.4</td><td>39.1</td><td>62.6</td><td>37.9</td></tr><tr><td>+ SGTA</td><td>53.7</td><td>40.0</td><td>64.0</td><td>39.5</td></tr></table>

<table><tr><td> $N_w$ </td><td>tIoU</td><td>vIoU</td><td>GFLOPs</td></tr><tr><td>4</td><td>52.7</td><td>39.2</td><td>10.7</td></tr><tr><td>6</td><td>53.2</td><td>39.6</td><td>16.1</td></tr><tr><td>8</td><td>53.7</td><td>40.0</td><td>21.5</td></tr><tr><td>10</td><td>53.6</td><td>39.9</td><td>26.9</td></tr><tr><td>12</td><td>53.5</td><td>39.8</td><td>32.2</td></tr></table>

Impact of Refine Window Size $N _ { w }$ . We evaluate the influence of refine window size $N _ { w } \in \{ 4 , 6 , 8 , 1 0 , 1 2 \}$ to find the optimal temporal receptive field for boundary regression. Given the high sampling rate $f _ { r }$ , this window size $N _ { w }$ determines the balance between local detail and temporal span. As shown in Table 5, the tIoU and vIoU initially increases and peaks at $N _ { w } = 8$ . This trend suggests that a moderate window is essential for capturing suficient temporal context to disambiguate the transition between action and background. However, performance begins to degrade when $N _ { w }$ exceeds 8. This degradation is likely because overly long windows introduce redundant background frames and irrelevant motion noise, which dilutes the boundary-specific semantic features within the SGTA module. In terms of eficiency, GFLOPs increase linearly with $N _ { w } ,$ confirming that our decoupled design maintains stable computational scalability. Given that $N _ { w } = 8$ achieves the best trade-of between localization precision and resource consumption, we adopt it as our default configuration.

Components of SGTA Design. To further investigate the internal mechanism of the Semantic-Guided Temporal Aggregator, we perform an exhaustive ablation study on its key components: the Semantic Guide and the Temporal Attention mechanism. The results are summarized in Table 6. As observed, removing the Temporal Attention component leads to a noticeable performance degradation, with tIoU dropping from 53.7% to 52.8%. This decline underscores the necessity of explicit inter-frame dependency modeling for capturing longrange temporal dynamics in video sequences. Furthermore, the absence of the Semantic Guide results in a decrease across all metrics, confirming that semantic cues play a pivotal role in filtering out background noise and focusing the model on task-relevant temporal boundaries. The full SGTA configuration achieves the best performance, validating that the synergy between semantic grounding and temporal relation modeling is essential for precise boundary localization.

Table 6: Ablation on SGTA Design. Evaluated on the HC-STVGv1 dataset. TA and SG denote Temporal Attention and Semantic Guide, respectively.  
Table 7: Ablation on fusion mechanism. Deformable Semantic-Motion Fusion achieves superior performance with significantly lower complexity.

<table><tr><td>Configuration</td><td>tIoU</td><td>vIoU</td><td>vIoU@0.3</td><td>vIoU@0.5</td></tr><tr><td>Full SGTA</td><td>53.7</td><td>40.0</td><td>64.0</td><td>39.5</td></tr><tr><td>w/o TA.</td><td>52.8</td><td>39.3</td><td>63.1</td><td>38.7</td></tr><tr><td>w/o SG.</td><td>53.1</td><td>39.5</td><td>63.4</td><td>39.0</td></tr></table>

<table><tr><td>Mechanism</td><td>tIoU</td><td>vIoU</td><td>vIoU@0.5</td><td>GFLOPs</td></tr><tr><td>Standard SA</td><td>53.3</td><td>39.8</td><td>36.5</td><td>516</td></tr><tr><td>Deform. SA</td><td>53.7</td><td>40.0</td><td>39.5</td><td>260</td></tr></table>

Efectiveness of Deformable Semantic-Motion Fusion. As shown in Table 7, our Deformable Semantic-Motion Fusion significantly outperforms the standard baseline while reducing the computational overhead by 50%. Notably, our approach achieves a substantial gain in the stringent vIoU@0.5 metric. This demonstrates that by adaptively focusing on sparse sampling points rather than dense global dependencies, our method efectively mitigates spatial-temporal noise and provides superior localization precision for STVG task.

## 4.4 Qualitative Results

To intuitively demonstrate the efectiveness of our coarse-to-fine framework, we present qualitative grounding results in Fig. 6. As illustrated, the initial coarse stage often produces ambiguous temporal boundaries. Guided by explicit interframe reasoning within the local boundary focus, our refine stage successfully corrects these initial predictions by expanding or trimming the temporal windows, achieving more precise alignment with the Ground Truth. For more visualization examples, please refer to the Supplementary Material.

## 5 Conclusion

We propose ScanFocus, a novel coarse-to-fine framework that resolves temporal ambiguity in STVG by decoupling the task into a global spatio-temporal scan and a local boundary focus. To capture suppressed high-frequency cues and model explicit inter-frame dependencies, we introduce SGTA for fine-grained temporal modeling with dense boundary sampling. Extensive experiments across three benchmarks demonstrate that our method significantly outperforms existing SOTA approaches, establishing a highly efective paradigm for STVG.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China under Nos. 62276061 and 62436002. This work is also supported by Research

Text: the man in the hat raises his hand turns around and walks to the door.  
![](images/474c5485ecdc1268fd220e47ec39656bc07655d1459d713474afb98a7f4d2845.jpg)

Text: the man in the red suit closes his whip and hands the man in the white shirt away.  
![](images/c59f47b02971517865cbebca2ebb9abe45c41de33f789aa5c4cf29927dcaf19c.jpg)  
Fig. 6: Qualitative results of ScanFocus. The coarse stage often generates ambiguous boundaries. Our refine stage successfully corrects these initial proposals to align precisely with the Ground Truth.

Fund for Advanced Ocean Institute of Southeast University (Major Program MP202404).

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

2. Anne Hendricks, L., Wang, O., Shechtman, E., Sivic, J., Darrell, T., Russell, B.: Localizing moments in video with natural language. In: ICCV. pp. 5803–5812 (2017)

3. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D.: Vqa: Visual question answering. In: ICCV. pp. 2425–2433 (2015)

4. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

5. Bao, H., Dong, L., Piao, S., Wei, F.: Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254 (2021)

6. Barrios, W., Soldan, M., Ceballos-Arroyo, A.M., Heilbron, F.C., Ghanem, B.: Localizing moments in long video via multimodal guidance. In: ICCV. pp. 13667– 13678 (2023)

7. Cao, M., Yang, T., Weng, J., Zhang, C., Wang, J., Zou, Y.: Locvtp: Video-text pre-training for temporal localization. In: ECCV. pp. 38–56. Springer (2022)

8. Cao, Z., Zhang, B., Du, H., Yu, X., Li, X., Wang, S.: Flashvtg: Feature layering and adaptive score handling network for video temporal grounding. In: WACV. pp. 9226–9236. IEEE (2025)

9. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: Endto-end object detection with transformers. In: ECCV. pp. 213–229. Springer (2020)

10. Chen, W., Chen, L., Wu, Y.: An eficient and efective transformer decoder-based framework for multi-task visual grounding. In: ECCV. pp. 125–141. Springer (2024)

11. Chen, Y.W., Tsai, Y.H., Yang, M.H.: End-to-end multi-modal video temporal grounding. NeurIPS 34, 28442–28453 (2021)

12. Dai, M., Chen, K., Cheng, W., Zhuang, J., Feng, Z., Zhu, P., Yang, W.: Gc3vg: Generalized multi-task visual grounding with coarse-to-fine consistency constraints. TCSVT (2025)

13. Dai, M., Cheng, W., Liu, J.J., Yang, L., Feng, Z., Yang, W., Wang, J.: Improving generalized visual grounding with instance-aware joint learning. TPAMI (2025)

14. Dai, M., Cheng, W., Liu, J.j., Yang, S., Cai, W., Sun, Y., Yang, W.: Deris: Decoupling perception and cognition for enhanced referring image segmentation through loopback synergy. In: ICCV. pp. 19936–19946 (2025)

15. Dai, M., Yang, L., Xu, Y., Feng, Z., Yang, W.: Simvg: A simple framework for visual grounding with decoupled multi-modal fusion. NeurIPS 37, 121670–121698 (2024)

16. Dai, M., Yang, S., Duan, B., Yang, W., Wang, J.: Momentseg: Moment-centric sampling for enhanced video pixel understanding. arXiv preprint arXiv:2510.09274 (2025)

17. Gao, J., Sun, C., Yang, Z., Nevatia, R.: Tall: Temporal activity localization via language query. In: ICCV. pp. 5267–5275 (2017)

18. Gu, X., Chen, G., Wang, Y., Zhang, L., Luo, T., Wen, L.: Text with knowledge graph augmented transformer for video captioning. In: CVPR. pp. 18941–18951 (2023)

19. Gu, X., Fan, H., Huang, Y., Luo, T., Zhang, L.: Context-guided spatio-temporal video grounding. In: CVPR. pp. 18330–18339 (2024)

20. Gu, X., Shen, Y., Luo, C., Luo, T., Huang, Y., Lin, Y., Fan, H., Zhang, L.: Knowing your target: Target-aware transformer makes better spatio-temporal video grounding. arXiv preprint arXiv:2502.11168 (2025)

21. Gu, X., Zhang, H., Fan, Q., Niu, J., Zhang, Z., Zhang, L., Chen, G., Chen, F., Wen, L., Zhu, S.: Thinking with bounding boxes: Enhancing spatio-temporal video grounding via reinforcement fine-tuning. arXiv preprint arXiv:2511.21375 (2025)

22. Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)

23. Guo, D., Wu, F., Zhu, F., Leng, F., Shi, G., Chen, H., Fan, H., Wang, J., Jiang, J., Wang, J., et al.: Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062 (2025)

24. Hao, J., Sun, H., Ren, P., Wang, J., Qi, Q., Liao, J.: Can shufling video benefit temporal bias problem: A novel training framework for temporal grounding. In: ECCV. pp. 130–147. Springer (2022)

25. Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., Chen, X.: In defense of grid features for visual question answering. In: CVPR. pp. 10267–10276 (2020)

26. Jin, Y., Yuan, Z., Mu, Y., et al.: Embracing consistency: A one-stage approach for spatio-temporal video grounding. NeurIPS 35, 29192–29204 (2022)

27. Kamath, A., Singh, M., LeCun, Y., Synnaeve, G., Misra, I., Carion, N.: Mdetrmodulated detection for end-to-end multi-modal understanding. In: ICCV. pp. 1780–1790 (2021)

28. Kang, W., Liu, G., Shah, M., Yan, Y.: Segvg: Transferring object bounding box to segmentation for visual grounding. In: ECCV. pp. 57–75. Springer (2024)

29. Lei, J., Berg, T.L., Bansal, M.: Detecting moments and highlights in videos via natural language queries. NeurIPS 34, 11846–11858 (2021)

30. Li, H., Chen, J., Wei, Z., Huang, S., Hui, T., Gao, J., Wei, X., Liu, S.: Llava-st: A multimodal large language model for fine-grained spatial-temporal understanding. In: CVPR. pp. 8592–8603 (2025)

31. Li, Z., Li, Y., Li, Q., Wang, P., Guo, D., Lu, L., Jin, D., Zhang, Y., Hong, Q.: Lvit: language meets vision transformer in medical image segmentation. IEEE transactions on medical imaging 43(1), 96–107 (2023)

32. Lin, K.Q., Zhang, P., Chen, J., Pramanick, S., Gao, D., Wang, A.J., Yan, R., Shou, M.Z.: Univtg: Towards unified video-language temporal grounding. In: ICCV. pp. 2794–2804 (2023)

33. Lin, Z., Tan, C., Hu, J.F., Jin, Z., Ye, T., Zheng, W.S.: Collaborative static and dynamic vision-language streams for spatio-temporal video grounding. In: CVPR. pp. 23100–23109 (2023)

34. Liu, Y., He, J., Li, W., Kim, J., Wei, D., Pfister, H., Chen, C.W.: r 2-tuning: Eficient image-to-video transfer learning for video temporal grounding. In: ECCV. pp. 421–438. Springer (2024)

35. Moon, W., Hyun, S., Lee, S., Heo, J.P.: Correlation-guided query-dependency calibration for video temporal grounding. arXiv preprint arXiv:2311.08835 (2023)

36. Moon, W., Hyun, S., Park, S., Park, D., Heo, J.P.: Query-dependent video representation for moment retrieval and highlight detection. In: CVPR. pp. 23023–23033 (2023)

37. Mun, J., Cho, M., Han, B.: Local-global video-text interactions for temporal grounding. In: CVPR. pp. 10810–10819 (2020)

38. Ren, Z., Huang, Z., Wei, Y., Zhao, Y., Fu, D., Feng, J., Jin, X.: Pixellm: Pixel reasoning with large multimodal model. In: CVPR. pp. 26374–26383 (2024)

39. Shen, Y., Gu, X., Xu, K., Fan, H., Wen, L., Zhang, L.: Accurate and fast compressed video captioning. In: ICCV. pp. 15558–15567 (2023)

40. Su, R., Yu, Q., Xu, D.: Stvgbert: A visual-linguistic transformer based framework for spatio-temporal video grounding. In: ICCV. pp. 1533–1542 (2021)

41. Tan, C., Lin, Z., Hu, J.F., Li, X., Zheng, W.S.: Augmented 2d-tan: A twostage approach for human-centric spatio-temporal video grounding. arXiv preprint arXiv:2106.10634 (2021)

42. Tang, Z., Liao, Y., Liu, S., Li, G., Jin, X., Jiang, H., Yu, Q., Xu, D.: Humancentric spatio-temporal video grounding with visual transformers. TCSVT 32(12), 8238–8249 (2021)

43. Tong, Z., Song, Y., Wang, J., Wang, L.: Videomae: Masked autoencoders are dataeficient learners for self-supervised video pre-training. NeurIPS 35, 10078–10093 (2022)

44. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. NeurIPS 30 (2017)

45. Wang, J., Zhang, Z., Liu, Z., Li, Y., Ge, J., Xie, H., Zhang, Y.: Spacevllm: Endowing multimodal large language model with spatio-temporal video grounding capability. arXiv preprint arXiv:2503.13983 (2025)

46. Wang, L., Mittal, G., Sajeev, S., Yu, Y., Hall, M., Boddeti, V.N., Chen, M.: Protege: Untrimmed pretraining for video temporal grounding by video temporal grounding. In: CVPR. pp. 6575–6585 (2023)

47. Wang, W., Liu, J., Su, Y., Nie, W.: Eficient spatio-temporal video grounding with semantic-guided feature decomposition. In: ACMMM. pp. 4867–4876 (2023)

48. Wang, W., Bao, H., Dong, L., Bjorck, J., Peng, Z., Liu, Q., Aggarwal, K., Mohammed, O.K., Singhal, S., Som, S., et al.: Image as a foreign language: Beit pretraining for vision and vision-language tasks. In: CVPR. pp. 19175–19186 (2023)

49. Wang, Z., Wang, L., Wu, T., Li, T., Wu, G.: Negative sample matters: A renaissance of metric learning for temporal grounding. In: AAAI. vol. 36, pp. 2613–2623 (2022)

50. Wasim, S.T., Naseer, M., Khan, S., Yang, M.H., Khan, F.S.: Videogrounding-dino: Towards open-vocabulary spatio-temporal video grounding. In: CVPR. pp. 18909– 18918 (2024)

51. Weng, Y., Han, M., He, H., Chang, X., Zhuang, B.: Longvlm: Eficient long video understanding via large language models. In: ECCV. pp. 453–470. Springer (2024)

52. Xu, H., He, K., Plummer, B.A., Sigal, L., Sclarof, S., Saenko, K.: Multilevel language and vision integration for text-to-clip retrieval. In: AAAI. vol. 33, pp. 9062– 9069 (2019)

53. Yang, A., Miech, A., Sivic, J., Laptev, I., Schmid, C.: Tubedetr: Spatio-temporal video grounding with transformers. In: CVPR. pp. 16442–16453 (2022)

54. You, Q., Jin, H., Wang, Z., Fang, C., Luo, J.: Image captioning with semantic attention. In: CVPR. pp. 4651–4659 (2016)

55. Yu, Y., Wang, X., Hu, W., Luo, X., Li, C.: 2rd place solutions in the hc-stvg track of person in context challenge 2021. arXiv preprint arXiv:2106.07166 3(7) (2021)

56. Zhang, S., Peng, H., Fu, J., Luo, J.: Learning 2d temporal adjacent networks for moment localization with natural language. In: AAAI. vol. 34, pp. 12870–12877 (2020)

57. Zhang, Y., Chen, X., Jia, J., Liu, S., Ding, K.: Text-visual prompting for eficient 2d temporal video grounding. In: CVPR. pp. 14794–14804 (2023)

58. Zhang, Z., Zhao, Z., Lin, Z., Huai, B., Yuan, N.J.: Object-aware multibranch relation networks for spatio-temporal video grounding. arXiv preprint arXiv:2008.06941 (2020)

59. Zhang, Z., Zhao, Z., Zhao, Y., Wang, Q., Liu, H., Gao, L.: Where does it exist: Spatio-temporal video grounding for multi-form sentences. In: CVPR. pp. 10668– 10677 (2020)

60. Zhao, H., Ong, Y.S., Zhou, J.T.: Agentic spatio-temporal grounding via collaborative reasoning. arXiv preprint arXiv:2602.13313 (2026)

61. Zhu, C., Zhou, Y., Shen, Y., Luo, G., Pan, X., Lin, M., Chen, C., Cao, L., Sun, X., Ji, R.: Seqtr: A simple yet universal network for visual grounding. In: ECCV. pp. 598–615. Springer (2022)

62. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025)

63. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159 (2020)