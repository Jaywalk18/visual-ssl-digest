# MobileSAM2: Lightweight Segment Anything for Spatial Intelligence

Kai Jiang<sup>1</sup> , Jiaxing Huang<sup>1,⋆</sup> , Jingyi Zhang<sup>2</sup>, Weiying Xie<sup>3</sup>, Yunsong Li<sup>3</sup>, Yufei Wang<sup>4</sup>, Aoran Xiao<sup>2</sup>, and Dacheng Tao<sup>2</sup>

<sup>1</sup> Hong Kong Polytechnic University, HK, China

<sup>2</sup> Nanyang Technological University, Singapore

<sup>3</sup> Xidian University, China

<sup>4</sup> SparcAI Inc., USA

xdjiangkai@foxmail.com, jiaxing.huang@polyu.edu.hk

Abstract. The recent large video foundation model, SAM2, enables segment anything in both images and videos, serving as a powerful base model for various applications. However, many of such use cases require to operate on resource-constrained devices like mobile phones and laptops. In this work, we aim to make SAM2 more mobile-friendly by distilling the heavyweight SAM2 into a lightweight model, facilitating segment anything in both images and videos on mobile devices. To this end, we propose Hypergraphical Knowledge Distill (HyperKD), which introduces the idea of hypergraph into knowledge distillation, aiming to efectively model and transfer SAM2’s generalizable and comprehensive knowledge. HyperKD consists of Temporal HyperKD and Granularity HyperKD that construct hypergraphs to explicitly model and extract the generalizable temporal knowledge and the comprehensive multi-granularity knowledge from SAM2 respectively, which are then distilled into the lightweight student model by aligning it with the constructed hypergraphs. Besides, we present MobileSAM2, a new family of lightweight SAM2 that balances eficiency and efectiveness via searching the best model architectures with HyperKD during model size reduction. Extensive experiments validate MobileSAM2 across multiple benchmarks and show promising generalization performance on embodied AI tasks.

Keywords: Segment anything model · Knowledge distillation · Lightweight model · Spatial intelligence · Embodied AI

## 1 Introduction

Large vision foundation models [26, 35, 53, 54] trained with large-scale visual data have achieved significant success across various areas of computer vision. A prominent example is Segment Anything Model (SAM) [35], the first foundation model for image segmentation, which learns over 11 million images with 1.1 billion mask annotations and thus enables segmenting anything. Recently, SAM2 [54] has extended SAM to video segmentation while keeping the image segmentation ability intact. SAM2 learns rich temporal and segmentation knowledge from an unprecedented dataset with 113.8K videos and 40.9M mask annotations (e.g., SA-V manual and internal data [54]) and outperforms traditional methods by substantial margins, showing great flexibility in segmenting anything in both still images and video streams and demonstrating strong generalization and zero-shot

![](images/f9c49afe5d8a73c0e32054940140dc94e5d0a68163fd36e1d7746f12829bd548.jpg)  
Fig. 1: Video Segmentation Comparison<sup>5</sup>. With the proposed HyperKD and the searched model architectures, our MobileSAM2 works efectively with limited parameters.

Given its impressive capabilities, SAM2 has been employed as a powerful foundation model for a wide range of computer vision applications, such as video editing [5, 42], trafic surveillance [78, 82], autonomous vehicles [12, 49], robotics [11, 27], etc. On the other hand, many of these use cases require to operate on resource-constrained devices [31, 36, 58] like mobile phones, laptops, drones, robots, etc. In this work, we aim to make SAM2 more mobile-friendly by distilling it into a lightweight model, facilitating segmenting anything on mobile devices for both images and videos.

We begin by examining the key factors behind SAM2’s success. Compared to previous video segmentation methods, SAM2, for the first time, scales up video training data to an unprecedented level, both in the number of videos and the granularity of mask annotations, learning from which two critical types of knowledge including (1) Temporal Knowledge: SAM2 captures rich temporal correlations across video frames by learning from over 100K videos (∼50 times of the previous largest dataset), ultimately acquiring generalizable temporal knowledge that enables robust segmentation along video frames; and (2) Multi-Granularity Knowledge: SAM2 captures diverse granularity correlations across multiple granularity levels of visual concepts from over 40M multi-level mask annotations (e.g, objects as well as parts and subparts), finally learning comprehensive multi-granularity knowledge that enables comprehensive and coarse-to-fine understanding and segmentation of various visual concepts in videos. Inspired by these insights, we believe that one efective method of distilling SAM2’s knowledge is to transfer such two types of knowledge comprehensively, such that the lightweight student model can inherit SAM2’s robust and comprehensive video segmentation capabilities.

To this end, we propose Hypergraphical Knowledge Distill (HyperKD), which introduces the idea of hypergraph into knowledge distillation, aiming to efectively model and transfer SAM2’s generalizable and comprehensive knowledge into a lightweight model. Since these two types of knowledge are implicitly encoded in SAM2’s learnt parameters and thus dificult to directly access, HyperKD constructs hypergraphs to explicitly model and extract the generalizable temporal knowledge and the comprehensive multi-granularity knowledge from SAM2, which are then distilled into the lightweight student model by aligning it with the constructed hypergraphs.

We instantiate HyperKD with two examples, named Temporal HyperKD and Granularity HyperKD. Temporal HyperKD first considers patches/objects within a single video frame as nodes and the distances between their embeddings as edges, and then constructs a hypergraph by linking multiple related nodes across frames via hyperedges that are formed by fusing edges from each individual frame. Granularity HyperKD first treats segmentation entities within a single segmentation granularity level as nodes and the distances between their embeddings as edges, and then builds a hypergraph by linking multiple related nodes across segmentation granularity levels (e.g., objects, parts and subparts) via hyperedges that are formed by fusing edges from each individual segmentation granularity level.

Besides, based on our proposed HyperKD, we introduce MobileSAM2, a new family of lightweight segment anything models, which strikes a favorable trade-of between model size and performance. Specifically, we adopt a progressive model contraction strategy [56] to iteratively scale down a large model, generating a series of lightweight segment anything models by searching and identifying model architectures that best retain SAM2’s temporal and multi-granularity knowledge (i.e., those can minimize HyperKD losses) during the model contraction process.

Our MobileSAM2 ofers three desirable features: (1) Robust Temporal Segmentation: Temporal HyperKD models and transfers SAM2’s temporal knowledge, enabling MobileSAM2 to capture rich correlations across video frames and achieve robust segmentation along video frames; (2) Comprehensive Multi-Granularity Segmentation: Granularity HyperKD enables MobileSAM2 to capture diverse correlations across multiple segmentation granularity levels, allowing for comprehensive and coarse-to-fine understanding and segmentation; (3) Eficiency&Efectiveness: By searching for the best model architectures via HyperKD, the resulting MobileSAM2 strikes an efective balance between model size and performance.

The main contributions of this work are threefold. First, we introduce the idea of hypergraph for knowledge distillation to explicitly model and transfer

SAM2’s generalizable and comprehensive knowledge into a lightweight model. To our knowledge, this is the first work that explores hypergraph for video segmentation knowledge distillation. Second, we design HyperKD, including Temporal HyperKD and Granularity HyperKD that distill temporal and multigranularity knowledge efectively. Third, we present MobileSAM2, a new family of lightweight segment anything models that efectively balances model size and performance. Fourth, extensive experiments demonstrate MobileSAM2’s superior performance across multiple benchmarks.

## 2 Related Works

## 2.1 Segment Anything Model

Segmentation Anything Models (SAMs) have shown impressive zero-shot generalization and interactive segmentation across diverse datasets. Models like SAM [35], SEEM [86], and Semantic-SAM [38] advance segmentation by using a promptable architecture that processes spatial (points, boxes, masks) and/or semantic (text) prompts to generate segmentation masks. Fast-SAM [84], MobileSAM [79], and TinySAM [57] enhance inference speed, while HQ-SAM [32] improves segmentation quality. SAM2 [54] extends SAM [35] to video segmentation with a memory-based transformer, enabling robust and comprehensive segmentation across video frames by storing object information and handling past interactions. However, SAM2’s large parameters and high computational demands make deployment on resource-constrained devices challenging. Compressing and accelerating SAM2 is a largely under-explored problem. We focus on distilling SAM2 into a lightweight model, enabling eficient segmentation on mobile devices for both images and videos.

## 2.2 Knowledge Distillation

Knowledge distillation (KD) [20] aims to train compact and eficient student models under the guidance of large teacher models. By learning from the teacher’s soft labels, the student can outperform models trained only on hard labels. KD methods are broadly categorized into logit-based, feature-based, and relationbased approaches. Logit-based KD [6, 23, 30, 47, 81, 83] minimizes a distance metric, such as KL divergence, between the predicted probabilities of the student and teacher, typically derived from the softmax of logit outputs. Feature-based KD [7, 8, 21, 22, 40, 41, 43, 55, 60, 61, 75] uses intermediate feature maps or refined information as knowledge. Relation-based KD [39,45,50,51,62,72] aligns instance correlations between the student and teacher networks. In this work, we aim to distill SAM2’s generalizable temporal and multi-granularity knowledge learnt from over 100K videos and 40M annotations by capturing and encapsulating coarseto-fine semantic relationships across frames, achieving robust and comprehensive video segmentation in a lightweight MobileSAM2.

## 2.3 Hypergraph

Hypergraph [1, 14–16, 18, 19, 19, 28, 34, 37, 65] is an advanced data structure that captures complex, high-order associations by allowing hyperedges to connect multiple nodes simultaneously. This capability makes hypergraphs particularly efective for modeling intricate relationships that traditional pairwise graph structures cannot adequately represent. Hypergraphs have been widely applied across domains such as social network analysis [71, 76], drug-target interaction modeling [29,63], and brain network analysis [67,87], where multi-way interactions are crucial for understanding underlying structures. In this work, we design a hypergraphical knowledge distillation method that constructs temporal and granularity hypergraphs to distill SAM2’s generalizable knowledge learnt from a vast dataset. This approach efectively uncovers coarse-to-fine, high-order semantic relationships of various visual concepts across video frames, resulting in a lightweight MobileSAM2 with robust and comprehensive video segmentation capabilities.

## 3 Methodology

This section presents MobileSAM2, a new family of tiny and eficient segment anything models with efective and eficient distillation on video segmentation. We first introduce preliminaries in Section 3.1. Then we introduce the Hypergraphical Knowledge Distillation (HyperKD) framework for small segment anything model training in Section 3.2. After that, we design a new tiny model family that balances eficiency and efectiveness by progressively scaling down a large seed model in Section 3.4.

## 3.1 Preliminaries

Segment Anything Model 2 (SAM2) [54] is a unified model for video and image segmentation. SAM2 is designed to segment anything by utilizing a “Promptable Segmentation” scheme, where the model generates a segment of spatio-temporal mask (i.e., a ‘masklet’) based on given prompts on any frames of the video, such as points, boxes, or masks. SAM2 can segment the object of interest in single image and across video frames when appropriate prompts are provided. Additionally, SAM2 support “interactive segmentation”, which allows users to iteratively refine segments and scale up training data through modelin-the-loop annotation. This process results in a more powerful SAM2 that can handle diverse segmentation tasks with improved flexibility and accuracy.

SAM2 typically works with two stages. It first encodes an input frame and a set of prompts into feature embeddings and then predicts the expected segmentation mask conditioned on these features and the memory context from previously observed frames. Specifically, SAM2 consists of an image encoder Encoder<sup>I</sup>, a memory module MemModule, a prompt encoder Encoder<sup>P</sup> , and a mask decoder Decoder. Given an input video $\boldsymbol { x } ^ { I } \in \mathbb { R } ^ { T \times H \times W \times 3 }$ , where T is the number of frames, and a prompt $x ^ { P }$ on the first frame, SAM2 first encodes $x ^ { I }$ and $x ^ { P }$ into embeddings as follows:

![](images/dfbecf594c0e2c7212f872819b581a565bb63b74771614ff84fd2bfaaaa9bf00.jpg)  
Fig. 2: Overview of HyperKD. HyperKD constructs hypergraphs to explicitly model and extract the generalizable temporal knowledge and the comprehensive multigranularity knowledge from SAM2, which are then distilled into lightweight MobileSAM2 by aligning it with the constructed hypergraphs. Specifically, Temporal HyperKD considers objects within a single frame as nodes and constructs hypergraphs by linking multiple related nodes across frames via hyperedges, as shown along the temporal dimension. Granularity HyperKD treats segmentation entities within a single granu larity level as nodes and builds hypergraphs by linking multiple relevant nodes across granularity levels $( \mathrm { e . g . }$ , objects, parts and subparts) via hyperedges, as shown along the granularity dimension. In this way, the trained MobileSAM2 captures rich and diverse hypergraphical correlations across multiple frames and granularity levels, ultimately achieving robust and comprehensive video understanding and segmentation.

$$
z ^ {I} = \mathbf {E n c o d e r} ^ {I} (x ^ {I}), z ^ {P} = \mathbf {E n c o d e r} ^ {P} (x ^ {P}).\tag{1}
$$

The memory module then conditions the i-th frame embedding $z ^ { I } ( i )$ on the past N frames embeddings $\{ z ^ { I } ( j ) \} _ { j = i - N } ^ { i }$ and predictions $\{ m ( j ) \} _ { j = i - N } ^ { i }$ as follows:

$$
z _ {m} ^ {I} (i) = \text { MemModule } \left(z ^ {I} (i) \mid \{z ^ {I} (j) \} _ {j = i - N} ^ {i - 1}, \{m (j) \} _ {j = i - N} ^ {i - 1}\right),\tag{2}
$$

where $\{ z ^ { I } ( i ) \} _ { i = t - N } ^ { t - 1 }$ and $\{ m ( i ) \} _ { i = t - N } ^ { t - 1 }$ interact with $z ^ { I } ( t )$ , leading to a conditioned frame embedding $z _ { m } ^ { I }$

The mask decoder Decoder then decodes the conditioned frame embedding $z _ { m } ^ { I } ( t )$ conditioned on the prompt embedding $z ^ { P }$ :

$$
m (i), c (i) = \mathbf {D e c o d e r} \left(z _ {m} ^ {I} (i) \mid z ^ {P}\right),\tag{3}
$$

where $z ^ { P }$ interacts with $z _ { m } ^ { I } ( i )$ , producing binary segmentation predictions $m \in$ $\mathbb { R } ^ { T \times M \times H \times W }$ that include M valid masks, each corresponding to diferent levels of segmentation granularity, along with the corresponding confidence scores $c \in \overset { \smile } { \mathbb { R } ^ { T \times M } }$

Naïve Solution with FitNet [55]. In this paper, we adopt SAM2 [64] as the pretrained teacher model, using Hiera-L [56] as image encoder. We employ the FitNet approach [55] for knowledge distillation, where a teacher model guides a student model by aligning the studen $\mathrm { \Omega } ^ { \prime } \mathrm { s }$ intermediate feature representations from the image encoder with those from the teacher. Given an unlabeled video $x ^ { I }$ the teacher model first generates feature embeddings through its image encoder, which serve as alignment targets for the corresponding layers in the student model’s image encoder. This alignment of intermediate representations enhances knowledge transfer in our distillation setup. The unsupervised training of the student model on unlabeled data is formulated as:

$$
\mathrm{Loss} = \| z _ {t} ^ {I} - z _ {s} ^ {I} \| _ {1},\tag{4}
$$

which aligns the intermediate feature embedding $z _ { t } ^ { I }$ and $z _ { s } ^ { I }$ from the teacher’s image encoder Encode $\mathbf { c } _ { t } ^ { I }$ and the student’s image encoder Encode $\cdot _ { s } ^ { I }$ , respectively.

## 3.2 Hypergraphical Knowledge Distillation

We observe that SAM2’s success is driven by its captured two key types of knowledge from large-scale, multi-granular video data: temporal knowledge for robust segmentation across frames and multi-granularity knowledge for detailed understanding of visual concepts at multiple levels. Motivated by this, we propose Hypergraphical Knowledge Distillation (HyperKD), which uses hypergraphs in knowledge distillation to efectively capture and encapsulate above two types of knowledge into a lightweight model. Since SAM2’s temporal and multi-granularity knowledge are implicitly encoded in its parameters, HyperKD explicitly models them through hypergraphs, which the student model learns to emulate. We instantiate HyperKD with two methods: Temporal HyperKD that links related nodes across frames to capture temporal consistency, and Granularity HyperKD that connects nodes across segmentation levels to convey multi-level understanding.

Temporal HyperKD Temporal HyperKD works in a two-step manner. The first step is temporal hypergraph extraction with node and hyperedge construction that aims to uncover the implicitly encoded generalizable temporal knowledge in SAM2. The second step is temporal hypergraph encapsulation, which encapsulates the extracted temporal hypergraph into the student model, enabling the student model to generate more temporally consistent segmentation.

Temporal Hypergraph Extraction. With the generalizable temporal knowledge contained in video embedding $z _ { t } ^ { I }$ from teacher model, we formulate the proposed Patch-wise Temporal Hypergraph and Instance-wise Temporal Hypergraph as $\mathcal { G } _ { t } ^ { P a } = ( \mathcal { V } _ { t } ^ { P a } , \mathcal { E } _ { t } ^ { \tilde { P a } } )$ and $\bar { \mathcal { G } } _ { t } ^ { \bar { I } n s } = ( \bar { \mathcal { V } } _ { t } ^ { I n s } , \mathcal { E } _ { t } ^ { I n s } )$ , which are capable of capturing temporal semantic relationships and associations across frames. For $\mathcal { G } _ { t } ^ { \bar { P } a }$ , we deconstruct the video embedding $z _ { t } ^ { I }$ as grid-based patches to constitute vertex set $\mathcal { V } _ { t } ^ { P a }$ . For $\mathcal { G } _ { t } ^ { I n s }$ , we constitute vertex set $\mathcal { V } _ { t } ^ { I n s }$ by pooling the video embedding $z _ { t } ^ { I }$ according to high confidence segmentation entries of each frame in teacher model prediction $m _ { t }$

Without loss of generality, to establish hyperedges that model relationships among vertices in $\nu ,$ we apply an ϵ-ball distance threshold around each vertex. An ϵ-ball forms a hyperedge that includes all vertices within a certain distance ϵ from a central vertex. The set of hyperedges $\mathcal E _ { t } ^ { P a }$ or $\mathcal { E } _ { t } ^ { I n s }$ are therefore constructed as follows:

$$
\mathcal {E} = \{\text { ball } (v, \epsilon) \mid v \in \mathcal {V} \},\tag{5}
$$

where ba $\mathrm { { . l l } } ( v , \epsilon ) = \{ u \ | \ \mathrm { { d i s t } } ( x _ { u } ^ { I } - x _ { v } ^ { I } ) < \epsilon , u \in \mathcal { V } \}$ represents the neighboring vertex set of a given vertex v, V is $\mathcal { V } _ { t } ^ { P a }$ or $\mathcal { V } _ { t } ^ { I n s }$ , and dist $( \bar { x _ { u } ^ { I } } - x _ { v } ^ { I } )$ is the distance function used to determine proximity. In practical, the hypergraph $\mathcal { G }$ is represented by its incidence matrix $H$ , where $H ( u , v ) = \mathrm { { d i s t } } ( x _ { u } ^ { I } - x _ { v } ^ { I } ) = \bar { 1 } - \langle x _ { u } ^ { I } , x _ { v } ^ { I } \rangle / ( \| x _ { u } ^ { I } \| \| x _ { v } ^ { I } \| )$ if vertex u is part of the hyperedge centered at vertex $v ,$ and $H ( u , v ) = 0$ otherwise.

Temporal Hypergraph Encapsulation encapsulates both patch-level and instance-level generalizable temporal knowledge captured by the extracted hypergraph $\mathcal G _ { t } ^ { P a }$ and $\mathcal { G } _ { t } ^ { I n s }$ into the student model, thereby preserving SAM2’s generalizable temporal knowledge within the student model. Specifically, similar to the construction of $\mathcal G _ { t } ^ { P a }$ and $\mathcal { G } _ { t } ^ { I n s }$ , we construct $\mathcal { G } _ { s } ^ { P a }$ and $\mathcal { G } _ { s } ^ { I n s }$ for video embeddings $z _ { s } ^ { I }$ from student model. Then, we encapsulate the extracted temporal hypergraph $\mathcal { G } _ { t } ^ { P a }$ and $\mathcal { G } _ { t } ^ { I n s }$ into student model by minimizing:

$$
\mathcal {L} _ {\mathrm{THKD}} = \| H _ {t} ^ {P a} - H _ {s} ^ {P a} \| _ {\mathrm{F}} + \| H _ {t} ^ {I n s} - H _ {s} ^ {I n s} \| _ {\mathrm{F}},\tag{6}
$$

where $H _ { t } ^ { P a } , H _ { t } ^ { I n s } , H _ { s } ^ { P a }$ , and $H _ { s } ^ { I n s }$ refer to the incidence matrices of $\mathcal { G } _ { t } ^ { P a } , \mathcal { G } _ { t } ^ { I n s }$ 2 $\mathcal { G } _ { s } ^ { P a }$ , and $\mathcal { G } _ { s } ^ { I n s }$ , respectively.

In this way, Temporal HyperKD ensures that both patch-level and instancelevel temporal hypergraphs capture neighborhood relationships within the feature space, efectively distilling SAM2’s generalizable temporal knowledge into the student model, enabling the student model to maintain temporal consistency and achieve robust segmentation across video frames.

Granularity HyperKD As Temporal HyperKD distills only the generalizable temporal knowledge, we further design Granularity HyperKD to extract a granularity hypergraph and encapsulate it into the student model to improve segmentation precision across diferent levels of detail. Specifically, the granularity hypergraph captures hierarchical relationships between multi-granularity level segmentation entities, complementing the temporal hypergraph by providing orthogonal and multi-granularity knowledge.

Granularity Hypergraph Extraction. Given the i-th frame embedding $z _ { t } ^ { I } ( i ) \in \mathbb { R } ^ { D \times h \times w }$ and its predicated binary segmentation masks $m ( i ) \in \mathbb { R } ^ { M \times H \times \smile }$ with M levels of segmentation granularity from teacher model, we instantiate granularity hypergraph as $\mathcal { G } _ { t } ^ { G r a n } = ( \mathcal { V } _ { t } ^ { G r a n } , \mathcal { E } _ { t } ^ { G r a n } )$ , in which each node $v \in \mathcal { V } _ { t } ^ { G r a n }$ 2 is initialized by pooling the frame embedding according to predicated binary segmentation masks as follows:

$$
\mathcal {V} _ {t} ^ {\text {   Gran   }} = \left\{\text { MaskedAveragePooling } \left(z _ {t} ^ {I} (i, j), m (i, j)\right) \mid j = 1, \dots , M \right\},\tag{7}
$$

and the set of hyperedges $\mathcal { E } _ { t } ^ { G r a n }$ can be constructed with $\mathrm { E q . 5 }$

Granularity Hypergraph Encapsulation encapsulates the orthogonal and multi-granularity knowledge in the extracted granularity hypergraph into the student model. This process complements the temporal hypergraph and enhances segmentation precision by enabling the model to segment objects at various levels of detail, such as objects, parts, and subparts. Specifically, similar to the construction of $\mathcal { G } _ { t } ^ { G r a n }$ , we construct $\mathcal { G } _ { s } ^ { G r a n }$ for the video embeddings $z _ { s } ^ { I }$ from the student model, using the predictions $m ( i )$ from the teacher model. We then encapsulate the extracted granularity hypergraph into the student model, following a similar approach as in Temporal Hypergraph Encapsulation, by minimizing the following loss function:

$$
\mathcal {L} _ {\mathrm{GHKD}} = \| H _ {t} ^ {G r a n} - H _ {s} ^ {G r a n} \| _ {\mathrm{F}},\tag{8}
$$

where $H _ { t } ^ { G r a n }$ and $H _ { s } ^ { G r a n }$ are the incidence matrices of $\mathcal { G } _ { t } ^ { G r a n }$ and $\mathcal { G } _ { s } ^ { G r a n }$ , respectively.

## 3.3 Overall Objective

In summary, the overall training loss of the proposed HyperKD can be formulated as:

$$
\mathrm{Loss} = \| z _ {t} ^ {I} - z _ {s} ^ {I} \| _ {1} + \alpha \cdot \mathcal {L} _ {T H K D} + \beta \cdot \mathcal {L} _ {G H K D},\tag{9}
$$

where α and $\beta$ are weighting coeficients.

## 3.4 Model Architectures

We introduce MobileSAM2, a lightweight version of SAM2 that strikes a balance between model size and performance. While SAM2’s memory module, prompt encoder, and mask decoder are lightweight (under 12M parameters), its image encoder, based on Hiera-L [56], has over 212M parameters, making it too heavy for resource-constrained devices. Thus, the key instantiate MobileSAM2 is reducing the size of the image encoder while preserving SAM2’s robust and comprehensive video segmentation capabilities.

Motivated by this observation, we present a new family of Hierarchical Vision Transformer [56] for SAM2 by scaling down a large model seed with a progressive model contraction approach [13]. Specifically, we begin with a manually designed Hiera backbone that serves as image encoder, which maintaining the core principles of the original Hiera design. Then, we establish a parameterized search space of contraction factors that includes critical architectural components such as embedding dimensions, layer counts, attention head configurations, and expansion ratios. The progressive model contraction process iteratively identifies promising architectures, ultimately resulting in a smaller Hiera variant that retains competitive performance while minimizing computational complexity and memory usage, making it suitable for eficient deployment. Hiera begins with a patch embedding layer that processes input images into tokens, followed by several stages where each stage progressively increases the channel dimensions while reducing spatial resolution through downsampling. The architecture emphasizes eficient parameter usage by employing lightweight MultiScaleBlock layers in early stages, transitioning to windowed self-attention in later stages. We consider the following contraction factors to form a model:

Table 1: Architectures of our searched lightweight MobileSAM2<sup>6</sup>.

<table><tr><td>Contraction Factors</td><td> $\Gamma_{\text{Emb}}$ </td><td> $\Gamma_{\text{Blk}}$ </td><td> $\Gamma_{\text{WinSiz}}$ </td><td> $\Gamma_{\text{HExp}}$ </td><td>Params. (M)</td></tr><tr><td>MobileSAM2-5M</td><td>[48, 96, 192, 384]</td><td>[1, 2, 5, 2]</td><td>[8, 4, 14, 7]</td><td>2</td><td>5.84</td></tr><tr><td>MobileSAM2-10M</td><td>[64, 128, 256, 512]</td><td>[1, 2, 5, 2]</td><td>[8, 4, 14, 7]</td><td>2</td><td>10.37</td></tr><tr><td>MobileSAM2-23M</td><td>[96, 192, 384, 768]</td><td>[1, 3, 9, 1]</td><td>[8, 8, 14, 7]</td><td>2</td><td>23.74</td></tr></table>

$I _ { \mathrm { E m b } } { : }$ Embedding dimension of each stage. Decreasing it results in a thinner network with fewer heads in multi-head self-attention.  
${ \itGamma } _ { \mathrm { B l k } } :$ The number of blocks in each stage. The depth of the model is decreased by reducing these values.  
$I _ { \mathrm { W i n S i z } } \mathrm { . }$ Window size in the each stage. Smaller window size lead to fewer operations and model parameters during self-attention.

$\begin{array} { r } { { \cal I } _ { \mathrm { H E x p } } { : } \qquad } \end{array}$ Expansion ratio of attention heads in multi-head attention at each stage. Reducing the dimensionality of each attention head directly decreases the computation burden of self-attention, leading to lower computation cost.

We scale down the above factors by the progressive model contraction approach and search and identify a new family of lightweight SAM2, named MobileSAM2, as shown in Table 1.

## 4 Experiments

Table 2 show the benchmarking of our methods with state-of-the-art knowledge distillation methods, including FitNets [55], CIRKD [70], and FAKD [77], over 5 widely used video segmentation datasets including 2 general video object segmentation (VOS) dataset (i.e., MOSE [10], DAVIS 2017 [64]), 1 long-term VOS dataset (LVOS [24]), and 2 segment anything in videos dataset (SA-V) [54] (i.e., dataset SA-V-val [54] and SA-V-test [54]).

## 4.1 Implementation Details

We use our designed MobileSAM2 as the lightweight student model and SAM2- L [64] (with Hiera-L [56] as the image encoder) as the pretrained teacher model. Since SAM2’s memory module (i.e., memory attention, memory encoder, and memory bank), prompt encoder and mask decoder are relatively lightweight (under 12M parameters), we optimize MobileSAM2 by training its image encoder while keeping its remaining modules (i.e., memory module, prompt encoder and mask decoder copied from the pretrained SAM2-L teacher model) frozen. We use only ∼ 10% SA-V data [54] (i.e., 11K Videos) for eficient distillation training. Following SAM2 [54], we employ the AdamW [44] optimizer with an initial learning rate of $5 \times 1 0 ^ { - 4 }$ . All models are trained for 5 epochs on an A100 (80GB) GPU with a batch size of 5, using 8-frame sequences per sample. The training time is 65 hours. For each iteration, we use a 4 × 4 grid of spatial prompts for mask prediction by the teacher model. We set weighting coeficients α = 1 and $\beta = 1$ to balance temporal consistency and hierarchical knowledge transfer.

Table 2: VOS comparison. With the proposed HyperKD and the searched model architectures, our MobileSAM2 works efectively with limited parameters on video segmentation. Note SAM2 Large [54] serves as the teacher model for all knowledge distill methods. For fair comparsion, all methods process video sequences.

<table><tr><td rowspan="2">Model</td><td rowspan="2" colspan="2">Distillation Method Params. (M)</td><td colspan="5">J&amp;F</td></tr><tr><td colspan="5">MOSE val DAVIS 2017 val LVOS val SA-V val SA-V test</td></tr><tr><td colspan="8">100% Training Data with 256 A100 GPUs: 113.8K Videos (SA-V manual and internal data [54])</td></tr><tr><td>SAM2 Base+ [54]</td><td>-</td><td>68.7</td><td>72.8</td><td>88.8</td><td>75.8</td><td>72.2</td><td>74.7</td></tr><tr><td>SAM2 Large [54]</td><td>-</td><td>212.1</td><td>74.6</td><td>89.2</td><td>81.7</td><td>74.5</td><td>76.0</td></tr><tr><td colspan="8">~ 10% Training Data with 1 A100 GPU: 11K Videos from SA-V manual data [54])</td></tr><tr><td rowspan="2">SAM2-TinyViT-5M</td><td>No Distill</td><td>5.4</td><td>26.0</td><td>30.4</td><td>30.8</td><td>28.4</td><td>27.7</td></tr><tr><td>FitNet [55]</td><td>5.4</td><td>33.8</td><td>38.7</td><td>37.7</td><td>34.3</td><td>35.1</td></tr><tr><td rowspan="6">MobileSAM2-5M</td><td>No Distill</td><td>5.8</td><td>29.7</td><td>36.6</td><td>33.8</td><td>32.7</td><td>33.0</td></tr><tr><td>FitNet [55]</td><td>5.8</td><td>36.6</td><td>41.5</td><td>39.7</td><td>38.2</td><td>39.3</td></tr><tr><td>CIRKD [70]</td><td>5.8</td><td>36.8</td><td>43.3</td><td>41.2</td><td>38.3</td><td>38.6</td></tr><tr><td>CAT-KD [21]</td><td>5.8</td><td>35.0</td><td>41.9</td><td>40.3</td><td>38.3</td><td>39.8</td></tr><tr><td>FAKD [77]</td><td>5.8</td><td>37.7</td><td>42.5</td><td>41.0</td><td>39.6</td><td>39.9</td></tr><tr><td>HyperKD (Ours)</td><td>5.8</td><td>44.6</td><td>51.0</td><td>48.8</td><td>49.2</td><td>49.4</td></tr><tr><td rowspan="2">SAM2-TinyViT-11M</td><td>No Distill</td><td>11.2</td><td>27.9</td><td>29.6</td><td>32.5</td><td>28.9</td><td>30.8</td></tr><tr><td>FitNet [55]</td><td>11.2</td><td>36.4</td><td>40.0</td><td>39.8</td><td>37.8</td><td>39.0</td></tr><tr><td rowspan="6">MobileSAM2-10M</td><td>No Distill</td><td>10.4</td><td>30.9</td><td>35.7</td><td>34.4</td><td>33.7</td><td>34.2</td></tr><tr><td>FitNet [55]</td><td>10.4</td><td>38.9</td><td>44.3</td><td>42.2</td><td>41.6</td><td>41.4</td></tr><tr><td>CIRKD [70]</td><td>10.4</td><td>40.8</td><td>45.8</td><td>43.7</td><td>42.4</td><td>43.2</td></tr><tr><td>CAT-KD [21]</td><td>10.4</td><td>39.9</td><td>44.3</td><td>42.9</td><td>41.3</td><td>41.8</td></tr><tr><td>FAKD [77]</td><td>10.4</td><td>41.3</td><td>46.7</td><td>44.0</td><td>42.4</td><td>43.3</td></tr><tr><td>HyperKD (Ours)</td><td>10.4</td><td>48.7</td><td>54.2</td><td>51.8</td><td>49.9</td><td>50.0</td></tr><tr><td rowspan="2">SAM2-TinyViT-21M</td><td>No Distill</td><td>21.2</td><td>30.3</td><td>35.2</td><td>34.5</td><td>32.0</td><td>34.7</td></tr><tr><td>FitNet [55]</td><td>21.2</td><td>44.9</td><td>49.0</td><td>47.0</td><td>45.8</td><td>45.6</td></tr><tr><td rowspan="6">MobileSAM2-23M</td><td>No Distill</td><td>23.7</td><td>39.5</td><td>44.7</td><td>44.8</td><td>42.4</td><td>44.3</td></tr><tr><td>FitNet [55]</td><td>23.7</td><td>57.4</td><td>61.1</td><td>60.8</td><td>59.4</td><td>59.9</td></tr><tr><td>CIRKD [70]</td><td>23.7</td><td>60.4</td><td>62.1</td><td>61.5</td><td>60.7</td><td>60.5</td></tr><tr><td>CAT-KD [21]</td><td>23.7</td><td>60.2</td><td>62.1</td><td>61.1</td><td>59.7</td><td>59.9</td></tr><tr><td>FAKD [77]</td><td>23.7</td><td>60.6</td><td>65.0</td><td>63.4</td><td>62.0</td><td>62.7</td></tr><tr><td>HyperKD (Ours)</td><td>23.7</td><td>65.8</td><td>72.1</td><td>69.0</td><td>66.4</td><td>67.8</td></tr></table>

## 4.2 Results

In line with SAM2 [54], our proposed MobileSAM2 is designed for general, interactive, promptable video segmentation tasks, while we also address the semi-supervised video object segmentation (VOS) setting, where the prompt is a ground-truth mask on the first frame, as this is a common protocol in the field. We compare MobileSAM2 with existing state-of-the-art knowledge distillation methods as well as SAM2 equipped with the state-of-the-art lightweight backbone (Tiny-ViT [66]) in Table 2, reporting accuracy based on standard protocols. We evaluate three versions of MobileSAM2 with varying model sizes (5M, 10M, and 23M), each ofering diferent size-accuracy trade-ofs.

Results on general VOS dataset. Table 2 presents the video object segmentation results on two general VOS datasets: MOSE and DAVIS 2017. MobileSAM2 achieves substantial performance improvements over the baseline across these datasets. Specifically, MobileSAM2 outperforms state-of-the-art methods by a large margin in J&F score, highlighting the efectiveness of our proposed HyperKD in exploring eficient lightweight model architectures as well as distilling knowledge from the pretrained SAM2 model. The superior performance of MobileSAM2 is largely attributed to HyperKD’s ability to capture generalizable temporal knowledge and comprehensive multi-granularity knowledge, enabling it to focus on cross-temporal and multi-granularity spatial cues of objects. This not only benefits the architecture search for lightweight MobileSAM2 but also enhances the distillation process from SAM2 to lightweight MobileSAM2 . All methods show performance gains. However, MobileSAM2 achieves the most significant improvements, demonstrating its capability to align closely with the SAM2’s representation by efectively distilling both temporal and multigranularity knowledge from SAM2.

Table 3: Ablation studies of HyperKD with Temporal HyperKD and Granular HyperKD. The experiments are conducted on video object segmentation (VOS) over LVOS val.

<table><tr><td rowspan="2"></td><td colspan="2">Temporal HyperKD</td><td rowspan="2">Granular HyperKD</td><td rowspan="2">J&amp;F</td></tr><tr><td>Patch-wise</td><td>Instance-wise</td></tr><tr><td>MobileSAM2-23M (Baseline/no distill)</td><td>-</td><td>-</td><td>-</td><td>44.8</td></tr><tr><td></td><td>√</td><td></td><td></td><td>62.9</td></tr><tr><td></td><td></td><td>√</td><td></td><td>64.4</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td>64.4</td></tr><tr><td></td><td></td><td></td><td>√</td><td>63.1</td></tr><tr><td>MobileSAM2-23M</td><td>√</td><td>√</td><td>√</td><td>69.0</td></tr></table>

Results on long-term VOS dataset. Table 2 also reports the experiments on long term VOS dataset, LVOS. MobileSAM2 surpasses all competing methods by a significant margin, demonstrating HyperKD’s efectiveness and robustness in capturing SAM2’s generalizable temporal and multi-granularity knowledge for long-term video segmentation. The substantial performance gains achieved by MobileSAM2 on LVOS highlight the eficiency of its lightweight architecture, explored by HyperKD’s guidance, as well as the efectiveness of distilling knowledge from SAM2 by HyperKD.

Results on segment anything in videos dataset. We evaluate the efectiveness of our MobileSAM2 on segment anything in videos dataset, i.e., SA-V val and SA-V test, which measure performance for open-world segments of “any” object class [54]. Table 2 reports the VOS result, showcasing significant improvements over the baseline and outperforming state-of-the-arts thereby highlighting the superiority of MobileSAM2.

## 4.3 Ablation Study

In Table 3, we conduct ablation studies to assess the individual contributions of our proposed MobileSAM2 on the LVOS dataset. The baseline model, without distillation from SAM2, does not perform well due to its limited ability to capture high-order temporal and multi-granularity semantic relationships of visual concepts. By contrast, including Patch-wise Temporal HyperKD significantly improves the baseline, indicating that Patch-wise Temporal HyperKD efectively captures patch-wise temporal dependencies across frames. Additionally, applying

Table 4: Parameter analysis on hypergraph construction threshold ϵ in HyperKD on LVOS.

<table><tr><td>Model Architecture</td><td colspan="5">MobileSAM2-23M</td></tr><tr><td>ε</td><td>0.5</td><td>0.75</td><td>1.0</td><td>1.25</td><td>1.5</td></tr><tr><td>J&amp;F</td><td>65.2</td><td>68.4</td><td>69.0</td><td>68.1</td><td>62.0</td></tr></table>

Table 5: Comparison of our HyperKD with traditional two-vertices graph-based knowledge distillation over LVOS.

<table><tr><td colspan="2">Model Architecture</td><td colspan="3">MobileSAM2-23M</td></tr><tr><td>Distillation Method</td><td>No Distill</td><td>Context Matters [69]</td><td>IntRA-KD [25]</td><td>HyperKD (Ours)</td></tr><tr><td>J&amp;F</td><td>44.8</td><td>60.7</td><td>60.9</td><td>69.0</td></tr></table>

Instance-wise Temporal HyperKD brings further improvements, showing that it enhances instance-level temporal knowledge across frames. Moreover, introducing Granularity HyperKD further boosts performance, demonstrating that it provides complementary multi-granularity knowledge that efectively benifits robust and comprehensive video segmentation capabilities.

## 4.4 Discussion

Parameter Study. In the construction of hyper graph, the predefined ϵ-ball distance threshold is used to establish hyperedges that model relationships among vertices in hypergraph. We studied ϵ by changing it from 0.5 to 1.5 with a step of 0.25. Table 4 reports the experiments over LVOS dataset. It is also observed that the performance of MobileSAM2 is relatively stable across from 0.75 to 1.25, with only minor variations, while there is a performance decline at the thresholds of 0.5 and 1.5. A higher threshold creates a more connected hypergraph, increasing the risk of over-smoothing, while a lower threshold may result in an under-connected hypergraph that fails to capture high-order relationships. To balance these factors, our HyperKD uses a distance threshold of 1.0 for hypergraph construction, chosen empirically to maintain connectivity without excessive smoothing.

HyperKD v.s. Graph-based Knowledge Distillation. We compare our HyperKD with the prior graph-based knowledge distillation method, Context Matters [69] and IntRA-KD [25]. As shown in Table 5, HyperKD outperforms Context Matters and IntRA-KD, primarily because the graphs of Context Matters and IntRA-KD cannot efectively capture high-order semantic relationships among visual concepts, which our hypergraph-based approach successfully achieves.

Distance Metrics for Constructing Hypergraph. We explore diferent feature distance metrics for hypergraph construction, conducting experiments with the following metrics: 1) Cosine Similarity [9], 2) Euclidean Distance [9], and 3) Manhattan Distance [9]. The results in Tab. 7 show that HyperKD performs efectively and consistently across all metrics. Among them, cosine similarity yields the best results, largely because it aligns with the attention mechanism, making it a natural choice for the distillation of generalizable temporal knowledge and comprehensive multi-granularity knowledge in SAM2.

Table 6: Eficiency comparison on mobile GPU.

<table><tr><td>Model</td><td>Params. (M)</td><td>Inference FPS</td><td>LVOS</td></tr><tr><td>SAM2 Base+</td><td>68.7</td><td>Out of Memory</td><td>-</td></tr><tr><td>SAM2 Large</td><td>212.1</td><td>Out of Memory</td><td>-</td></tr><tr><td>SAM2-TinyViT-5M</td><td>5.4</td><td>13.1</td><td>37.7</td></tr><tr><td>MobileSAM2-5M</td><td>5.8</td><td>13.3</td><td>48.8</td></tr><tr><td>SAM2-TinyViT-11M</td><td>11.2</td><td>11.3</td><td>39.8</td></tr><tr><td>MobileSAM2-10M</td><td>10.4</td><td>10.8</td><td>51.8</td></tr><tr><td>SAM2-TinyViT-21M</td><td>21.2</td><td>8.0</td><td>47.0</td></tr><tr><td>MobileSAM2-23M</td><td>23.7</td><td>8.2</td><td>69.0</td></tr></table>

Table 7: Analysis on hypergraph construction in HyperKD with diferent distance metrics on LVOS.

<table><tr><td>Model Architecture</td><td colspan="3">MobileSAM2-23M</td></tr><tr><td>Distance Metrics</td><td>Cosine Similarity (Ours)</td><td>Euclidean Distance</td><td>Manhattan Distances</td></tr><tr><td>J&amp;F</td><td>69.0</td><td>67.2</td><td>65.7</td></tr></table>

Eficiency Analysis. We evaluate model eficiency in inference speed for all models on a single RTX 4060 (8GB) mobile GPU using PyTorch 2.3.1 and CUDA 12.1 with automatic mixed precision (bfloat16). We compiled the image encoder with torch.compile for all models. FPS measurements for video object segmentation are taken with a batch size of 1, following the common protocol. Table 6 reports the results on LVOS dataset, demonstrating that MobileSAM2 achieves a significant reduction in computational overhead while maintaining competitive performance compared to the original SAM2, making it more mobilefriendly for resource-constrained devices.

Applicability of HyperKD to Other Foundation Models. We conduct experiments on TrackAnything (TAM) [73] as shown in Table 8. It demonstrates HyperKD’s applicability beyond SAM2: applied to the structurally independent TrackAnything (TAM) [73] under the same limited-data setting (∼10%), HyperKD improves TAM-TinyViT-21M from 58.9 to 63.1 J&F and MobileTAM-23M from 60.8 to 66.5 J&F on DAVIS-2017. More critically, it disentangles the distillation objective from the architecture design: since TAM shares no architectural lineage with SAM2 or MobileSAM2, the consistent gains confirm that the performance improvements stem from the HyperKD objective itself, not from co-adaptation with a specific backbone. Together, these findings establish HyperKD as a generalizable distillation framework, and validate that the gains on MobileSAM2 reflect genuine knowledge transfer rather than artifacts of the searched architecture.

Application of MobileSAM2 on Embodied AI. With the advent of reasoning models and agents [17, 74, 80], to verify the application of MobileSAM2 on embodied AI systems, we integrate MobileSAM2 into CaP-Agent0 [17] as the core perception model and evaluate manipulation performance on LIBERO-PRO [85].

Table 8: Applicability of HyperKD for TrackAnything (TAM) [73]. We report J&F scores on DAVIS-2017.

<table><tr><td>Model</td><td colspan="3">Distillation Method Params. (M) DAVIS-2017 J&amp;F</td></tr><tr><td></td><td colspan="3">100% Training Data</td></tr><tr><td>TrackAnything (TAM) [73]</td><td>-</td><td>636</td><td>73.1</td></tr><tr><td></td><td colspan="3">~ 10% Training Data</td></tr><tr><td rowspan="2">TAM-TinyViT-21M</td><td>No Distill</td><td>21.2</td><td>58.9</td></tr><tr><td>HyperKD (Ours)</td><td>21.2</td><td>63.1</td></tr><tr><td rowspan="2">MobileTAM-23M</td><td>No Distill</td><td>23.7</td><td>60.8</td></tr><tr><td>HyperKD (Ours)</td><td>23.7</td><td>66.5</td></tr></table>

Table 9: Application of MobileSAM2 on Embodied AI.

<table><tr><td rowspan="2">Method</td><td colspan="2">Libero-Object</td><td colspan="2">Libero-Goal</td><td colspan="2">Libero-Spatial</td><td rowspan="2">Avg.</td></tr><tr><td>Pos (Avg.)</td><td>Task (Avg.)</td><td>Pos (Avg.)</td><td>Task (Avg.)</td><td>Pos (Avg.)</td><td>Task (Avg.)</td></tr><tr><td>OpenVLA [33]</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\pi 0$  [3]</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\pi 0.5$  [2]</td><td>17</td><td>1</td><td>38</td><td>0</td><td>20</td><td>1</td><td>12.8</td></tr><tr><td>SAM3 [4]-CaP-Agent0 [17]</td><td>27</td><td>31</td><td>29</td><td>16</td><td>13</td><td>23</td><td>23.2</td></tr><tr><td>SAM2-TinyViT-21M - CaP-Agent0</td><td>21</td><td>27</td><td>23</td><td>13</td><td>11</td><td>21</td><td>19.3</td></tr><tr><td>MobileSAM2-23M - CaP-Agent0</td><td>24</td><td>30</td><td>28</td><td>14</td><td>12</td><td>23</td><td>21.8</td></tr></table>

As shown in Table 9, the full SAM3-powered CaP-Agent0 achieves 23.2% average success rate, outperforming training-based VLA baselines like $\pi _ { 0 . 5 }$ [2] (12.8%). At a comparable lightweight parameter scale, SAM2-TinyViT-21M built on the state-of-the-art Tiny-ViT [66] backbone drops to 19.3%. In comparison, our MobileSAM2-23M reaches 21.8% average success rate, retaining competitive performance and outperforming SAM2-TinyViT-21M on spatial reasoning tasks and embodied AI tasks. These results demonstrate that MobileSAM2 is an eficient perceptual backbone for resource-constrained robot platforms, enabling practical on-board deployment with minimal performance sacrifice.

## 5 Conclusion

This paper presents a new family of tiny and eficient video foundation models, MobileSAM2, distilled from large SAM2, using the proposed HyperKD. HyperKD consists of Temporal HyperKD and Granularity HyperKD that construct hypergraphs to explicitly model and extract the generalizable temporal knowledge and the comprehensive multi-granularity knowledge from SAM2 respectively, facilitating the knowledge distillation from SAM2 to MobileSAM2 as well as the model architecture searching of MobileSAM2. Extensive experiments on multiple video segmentation datasets demonstrate that MobileSAM2 consistently outperforms state-of-the-art techniques by clear margins.

## Acknowledgment

[Re Prof Tao] This project is supported by the National Research Foundation, Singapore, under its NRF Professorship Award No. NRF-P2024-001. This work is also supported by PolyU Internal Fund.

## References

1. Bai, S., Zhang, F., Torr, P.H.: Hypergraph convolution and hypergraph attention. Pattern Recognition 110, 107637 (2021)

2. Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M.R., Finn, C., Fusai, N., Galliker, M.Y., et al.: pi<sub>0.5</sub>: a vision-language-action model with open-world generalization. In: 9th Annual Conference on Robot Learning (2025)

3. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al.: pi<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164 (2024)

4. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H.K., Dollár, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: Sam 3: Segment anything with concepts (2025), https://arxiv.org/abs/2511.16719

5. Ceylan, D., Huang, C.H.P., Mitra, N.J.: Pix2video: Video editing using image difusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 23206–23217 (2023)

6. Chen, D., Mei, J.P., Wang, C., Feng, Y., Chen, C.: Online knowledge distillation with diverse peers. In: Proceedings of the AAAI conference on artificial intelligence. vol. 34, pp. 3430–3437 (2020)

7. Chen, D., Mei, J.P., Zhang, H., Wang, C., Feng, Y., Chen, C.: Knowledge distillation with the reused teacher classifier. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11933–11942 (2022)

8. Chen, P., Liu, S., Zhao, H., Jia, J.: Distilling knowledge via knowledge review. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5008–5017 (2021)

9. Deza, E., Deza, M.M., Deza, M.M., Deza, E.: Encyclopedia of distances. Springer (2009)

10. Ding, H., Liu, C., He, S., Jiang, X., Torr, P.H., Bai, S.: Mose: A new dataset for video object segmentation in complex scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20224–20234 (2023)

11. Dupont, P.E., Nelson, B.J., Goldfarb, M., Hannaford, B., Menciassi, A., O’Malley, M.K., Simaan, N., Valdastri, P., Yang, G.Z.: A decade retrospective of medical robotics research from 2010 to 2020. Science robotics 6(60), eabi8017 (2021)

12. Faisal, A., Kamruzzaman, M., Yigitcanlar, T., Currie, G.: Understanding autonomous vehicles. Journal of transport and land use 12(1), 45–72 (2019)

13. Feichtenhofer, C.: X3d: Expanding architectures for eficient video recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 203–213 (2020)

14. Feng, Y., Huang, J., Du, S., Ying, S., Yong, J.H., Li, Y., Ding, G., Ji, R., Gao, Y.: Hyper-yolo: When visual object detection meets hypergraph computation. arXiv preprint arXiv:2408.04804 (2024)

15. Feng, Y., Liu, S., Han, X., Du, S., Wu, Z., Hu, H., Gao, Y.: Hypergraph foundation model. arXiv preprint arXiv:2503.01203 (2025)

16. Feng, Y., You, H., Zhang, Z., Ji, R., Gao, Y.: Hypergraph neural networks. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 3558–3565 (2019)

17. Fu, M., Yu, J., El-Refai, K., Kou, E., Xue, H., Huang, H., Xiao, W., Wang, G., Li, F.F., Shi, G., et al.: Cap-x: A framework for benchmarking and improving coding agents for robot manipulation. arXiv preprint arXiv:2603.22435 (2026)

18. Gao, Y., Feng, Y., Ji, S., Ji, R.: Hgnn+: General hypergraph neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(3), 3181–3199 (2022)

19. Gao, Y., Zhang, Z., Lin, H., Zhao, X., Du, S., Zou, C.: Hypergraph learning: Methods and practices. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(5), 2548–2566 (2020)

20. Gou, J., Yu, B., Maybank, S.J., Tao, D.: Knowledge distillation: A survey. International Journal of Computer Vision 129(6), 1789–1819 (2021)

21. Guo, Z., Yan, H., Li, H., Lin, X.: Class attention transfer based knowledge distillation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11868–11877 (2023)

22. Heo, B., Kim, J., Yun, S., Park, H., Kwak, N., Choi, J.Y.: A comprehensive overhaul of feature distillation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1921–1930 (2019)

23. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015)

24. Hong, L., Chen, W., Liu, Z., Zhang, W., Guo, P., Chen, Z., Zhang, W.: Lvos: A benchmark for long-term video object segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13480–13492 (2023)

25. Hou, Y., Ma, Z., Liu, C., Hui, T.W., Loy, C.C.: Inter-region afinity distillation for road marking segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12486–12495 (2020)

26. Huang, J., Zhang, J., Jiang, K., Lu, S.: Open-vocabulary object detection via language hierarchy. arXiv preprint arXiv:2410.20371 (2024)

27. Javaid, M., Haleem, A., Singh, R.P., Suman, R.: Substantial capabilities of robotics in enhancing industry 4.0 implementation. Cognitive Robotics 1, 58–75 (2021)

28. Jiang, J., Wei, Y., Feng, Y., Cao, J., Gao, Y.: Dynamic hypergraph neural networks. In: IJCAI. pp. 2635–2641 (2019)

29. Jin, S., Hong, Y., Zeng, L., Jiang, Y., Lin, Y., Wei, L., Yu, Z., Zeng, X., Liu, X.: A general hypergraph learning algorithm for drug multi-task predictions in microto-macro biomedical networks. PLOS Computational Biology 19(11), e1011597 (2023)

30. Jin, Y., Wang, J., Lin, D.: Multi-level logit distillation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24276– 24285 (2023)

31. Kamath, V., Renuka, A.: Deep learning based object detection for resource constrained devices: Systematic review, future trends and challenges ahead. Neurocomputing 531, 34–60 (2023)

32. Ke, L., Ye, M., Danelljan, M., Tai, Y.W., Tang, C.K., Yu, F., et al.: Segment anything in high quality. Advances in Neural Information Processing Systems 36 (2024)

33. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al.: Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246 (2024)

34. Kim, S., Lee, S.Y., Gao, Y., Antelmi, A., Polato, M., Shin, K.: A survey on hypergraph neural networks: an in-depth and step-by-step guide. In: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. pp. 6534–6544 (2024)

35. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. arXiv preprint arXiv:2304.02643 (2023)

36. Kornaros, G.: Hardware-assisted machine learning in resource-constrained iot environments for security: review and future prospective. IEEE Access 10, 58603–58622 (2022)

37. Lee, G., Bu, F., Eliassi-Rad, T., Shin, K.: A survey on hypergraph mining: Patterns, tools, and generators. arXiv preprint arXiv:2401.08878 (2024)

38. Li, F., Zhang, H., Sun, P., Zou, X., Liu, S., Yang, J., Li, C., Zhang, L., Gao, J.: Semantic-sam: Segment and recognize anything at any granularity. arXiv preprint arXiv:2307.04767 (2023)

39. Li, G., Li, X., Wang, Y., Zhang, S., Wu, Y., Liang, D.: Knowledge distillation for object detection via rank mimicking and prediction-guided feature imitation. In: Proceedings of the AAAI conference on artificial intelligence. vol. 36, pp. 1306–1313 (2022)

40. Li, Z., Ye, J., Song, M., Huang, Y., Pan, Z.: Online knowledge distillation for eficient pose estimation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11740–11750 (2021)

41. Lin, S., Xie, H., Wang, B., Yu, K., Chang, X., Liang, X., Wang, G.: Knowledge distillation via the target-aware transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10915–10924 (2022)

42. Liu, S., Zhang, Y., Li, W., Lin, Z., Jia, J.: Video-p2p: Video editing with crossattention control. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8599–8608 (2024)

43. Liu, Y., Chen, K., Liu, C., Qin, Z., Luo, Z., Wang, J.: Structured knowledge distillation for semantic segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2604–2613 (2019)

44. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

45. Mei, K., Delbracio, M., Talebi, H., Tu, Z., Patel, V.M., Milanfar, P.: Conditional difusion distillation (2023)

46. Miles, R., Yucel, M.K., Manganelli, B., Saà-Garriga, A.: Mobilevos: Real-time video object segmentation contrastive learning meets knowledge distillation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10480–10490 (June 2023)

47. Mirzadeh, S.I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., Ghasemzadeh, H.: Improved knowledge distillation via teacher assistant. In: Proceedings of the AAAI conference on artificial intelligence. vol. 34, pp. 5191–5198 (2020)

48. Mordeson, J.N., Nair, P.S.: Fuzzy graphs and fuzzy hypergraphs, vol. 46. Physica (2012)

49. Parekh, D., Poddar, N., Rajpurkar, A., Chahal, M., Kumar, N., Joshi, G.P., Cho, W.: A review on autonomous vehicles: Progress, methods and challenges. Electronics 11(14), 2162 (2022)

50. Park, W., Kim, D., Lu, Y., Cho, M.: Relational knowledge distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3967–3976 (2019)

51. Peng, B., Jin, X., Liu, J., Li, D., Wu, Y., Liu, Y., Zhou, S., Zhang, Z.: Correlation congruence for knowledge distillation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5007–5016 (2019)

52. Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., Sorkine-Hornung, A.: A benchmark dataset and evaluation methodology for video object segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 724–732 (2016)

53. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763. PMLR (2021)

54. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.Y., Girshick, R., Dollár, P., Feichtenhofer, C.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024), https://arxiv.org/abs/2408. 00714

55. Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C., Bengio, Y.: Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550 (2014)

56. Ryali, C., Hu, Y.T., Bolya, D., Wei, C., Fan, H., Huang, P.Y., Aggarwal, V., Chowdhury, A., Poursaeed, O., Hofman, J., et al.: Hiera: A hierarchical vision transformer without the bells-and-whistles. In: International Conference on Machine Learning. pp. 29441–29454. PMLR (2023)

57. Shu, H., Li, W., Tang, Y., Zhang, Y., Chen, Y., Li, H., Wang, Y., Chen, X.: Tinysam: Pushing the envelope for eficient segment anything model. arXiv preprint arXiv:2312.13789 (2023)

58. Shuvo, M.M.H., Islam, S.K., Cheng, J., Morshed, B.I.: Eficient acceleration of deep learning inference on resource-constrained edge devices: A review. Proceedings of the IEEE 111(1), 42–91 (2022)

59. Su, C.P., Tseng, C.H., Pu, B., Zhao, L., Yang, J., Chen, Z., Lee, S.J.: Ea-kd: Entropy-based adaptive knowledge distillation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 731–740 (October 2025)

60. Tian, Y., Zhang, C., Guo, Z., Zhang, X., Chawla, N.: Learning mlps on graphs: A unified view of efectiveness, robustness, and eficiency. In: The Eleventh International Conference on Learning Representations (2022)

61. Tian, Y., Krishnan, D., Isola, P.: Contrastive multiview coding. arXiv preprint arXiv:1906.05849 (2019)

62. Tung, F., Mori, G.: Similarity-preserving knowledge distillation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1365–1374 (2019)

63. Viñas, R., Joshi, C.K., Georgiev, D., Lin, P., Dumitrascu, B., Gamazon, E.R., Liò, P.: Hypergraph factorization for multi-tissue gene expression imputation. Nature machine intelligence 5(7), 739–753 (2023)

64. Voigtlaender, P., Luiten, J., Torr, P.H., Leibe, B.: Siam r-cnn: Visual tracking by re-detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6578–6588 (2020)

65. Wang, Y., Kleinberg, J.: From graphs to hypergraphs: Hypergraph projection and its remediation. arXiv preprint arXiv:2401.08519 (2024)

66. Wu, K., Zhang, J., Peng, H., Liu, M., Xiao, B., Fu, J., Yuan, L.: Tinyvit: Fast pretraining distillation for small vision transformers. In: European conference on computer vision. pp. 68–85. Springer (2022)

67. Xiao, L., Wang, J., Kassani, P.H., Zhang, Y., Bai, Y., Stephen, J.M., Wilson, T.W., Calhoun, V.D., Wang, Y.P.: Multi-hypergraph learning-based brain functional connectivity analysis in fmri data. IEEE transactions on medical imaging 39(5), 1746–1758 (2019)

68. Xu, N., Yang, L., Fan, Y., Yue, D., Liang, Y., Yang, J., Huang, T.: Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327 (2018)

69. Yang, A., Lin, S., Yeh, C.H., Shu, M., Yang, Y., Chang, X.: Context matters: Distilling knowledge graph for enhanced object detection. IEEE Transactions on Multimedia (2023)

70. Yang, C., Zhou, H., An, Z., Jiang, X., Xu, Y., Zhang, Q.: Cross-image relational knowledge distillation for semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12319–12328 (2022)

71. Yang, D., Qu, B., Yang, J., Cudré-Mauroux, P.: Lbsn2vec++: Heterogeneous hypergraph embedding for location-based social networks. IEEE Transactions on Knowledge and Data Engineering 34(4), 1843–1855 (2020)

72. Yang, J., Martinez, B., Bulat, A., Tzimiropoulos, G., et al.: Knowledge distillation via softmax regression representation learning. International Conference on Learning Representations (ICLR) (2021)

73. Yang, J., Gao, M., Li, Z., Gao, S., Wang, F., Zheng, F.: Track anything: Segment anything meets videos (2023)

74. Yao, H., Huang, J., Wu, W., Zhang, J., Wang, Y., Liu, S., Wang, Y., Song, Y., Feng, H., Shen, L., et al.: Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. Advances in Neural Information Processing Systems 38, 29918–29952 (2026)

75. Yim, J., Joo, D., Bae, J., Kim, J.: A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4133–4141 (2017)

76. Young, J.G., Petri, G., Peixoto, T.P.: Hypergraph reconstruction from network data. Communications Physics 4(1), 135 (2021)

77. Yuan, J., Phan, M.H., Liu, L., Liu, Y.: Fakd: Feature augmented knowledge distillation for semantic segmentation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 595–605 (2024)

78. Zhang, B., Zhang, J.: A trafic surveillance system for obtaining comprehensive information of the passing vehicles based on instance segmentation. IEEE Transactions on Intelligent Transportation Systems 22(11), 7040–7055 (2020)

79. Zhang, C., Han, D., Qiao, Y., Kim, J.U., Bae, S.H., Lee, S., Hong, C.S.: Faster segment anything: Towards lightweight sam for mobile applications. arXiv preprint arXiv:2306.14289 (2023)

80. Zhang, J., Huang, J., Yao, H., Liu, S., Zhang, X., Lu, S., Tao, D.: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 1859–1869 (October 2025)

81. Zhang, R., Shen, J., Liu, T., Liu, J., Bendersky, M., Najork, M., Zhang, C.: Do not blindly imitate the teacher: Using perturbed loss for knowledge distillation. arXiv preprint arXiv:2305.05010 (2023)

82. Zhang, X., Feng, Y., Angeloudis, P., Demiris, Y.: Monocular visual trafic surveillance: A review. IEEE Transactions on Intelligent Transportation Systems 23(9), 14148–14165 (2022)

83. Zhang, Y., Xiang, T., Hospedales, T.M., Lu, H.: Deep mutual learning. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 4320–4328 (2018)

84. Zhao, X., Ding, W., An, Y., Du, Y., Yu, T., Li, M., Tang, M., Wang, J.: Fast segment anything. arXiv preprint arXiv:2306.12156 (2023)

85. Zhou, X., Xu, Y., Tie, G., Chen, Y., Zhang, G., Chu, D., Zhou, P., Sun, L.: Liberopro: Towards robust and fair evaluation of vision-language-action models beyond memorization. [arXiv preprint arXiv:2510.03827] (2025)

86. Zou, X., Yang, J., Zhang, H., Li, F., Li, L., Gao, J., Lee, Y.J.: Segment everything everywhere all at once. arXiv preprint arXiv:2304.06718 (2023)

87. Zu, C., Gao, Y., Munsell, B., Kim, M., Peng, Z., Zhu, Y., Gao, W., Zhang, D., Shen, D., Wu, G.: Identifying high order brain connectome biomarkers via learning on hypergraph. In: Machine Learning in Medical Imaging: 7th International Workshop, MLMI 2016, Held in Conjunction with MICCAI 2016, Athens, Greece, October 17, 2016, Proceedings 7. pp. 1–9. Springer (2016)

## 6 Appendix

## 6.1 Model Architectures

Our proposed MobileSAM2 architecture is detailed in Table 10. This architecture follows a hierarchical structure, starting with a patch embedding layer and progressing through four stages. To create a compact family of MobileSAM2 models, we employ contraction factors $\{ T _ { \mathrm { E m b } } , T _ { \mathrm { B l k } } , T _ { \mathrm { W i n S i z } } , T _ { \mathrm { H E x p } } \}$ , which enable us to scale down the model size. We begin with a base model containing 27M parameters, then generate a set of candidate models by adjusting these contraction factors. From these candidates, we select models that meet specific constraints on parameter count and throughput. We evaluate these selected models on approximately 2% of the SA-V manual dataset [54] for training and validate them on the LVOS [24] validation set. The models with the highest validation accuracy are further refined in subsequent steps until the target performance is achieved.

Table 10: Architectures of our searched lightweight MobileSAM2<sup>7</sup>.

<table><tr><td></td><td>Stage</td><td>Block</td><td>Configuration</td></tr><tr><td rowspan="5">MobileSAM2</td><td>Patch Embedding</td><td>Conv</td><td>Embed Dim  $\Gamma_{\text{Emb\_1}}$ </td></tr><tr><td>Stage 1</td><td>Multi-Scale Block [66]</td><td> $\begin{bmatrix} \text{Inupt Dim } \Gamma_{\text{Emb\_1}}, \text{ Outupt Dim } \Gamma_{\text{Emb\_2}} \\ \text{Window Size } \Gamma_{\text{WinSiz\_1}}, \text{ Attention Heads } 1 \end{bmatrix} \times 1$ </td></tr><tr><td>Stage 2</td><td>Multi-Scale Block [66]</td><td> $\begin{bmatrix} \text{Inupt Dim } \Gamma_{\text{Emb\_2}}, \text{ Outupt Dim } \Gamma_{\text{Emb\_2}} \\ \text{Window Size } \Gamma_{\text{WinSiz\_2}}, \text{ Attention Heads } \bar{\Gamma}_{\text{HExp}}^{2} \\ \text{Inupt Dim } \Gamma_{\text{Emb\_2}}, \text{ Outupt Dim } \Gamma_{\text{Emb\_3}} \\ \text{Window Size } \Gamma_{\text{WinSiz\_2}}, \text{ Attention Heads } \bar{\Gamma}_{\text{HExp}} \end{bmatrix} \times 1$ </td></tr><tr><td>Stage 3</td><td>Multi-Scale Block [66]</td><td> $\begin{bmatrix} \text{Inupt Dim } \Gamma_{\text{Emb\_3}}, \text{ Outupt Dim } \Gamma_{\text{Emb\_3}} \\ \text{Window Size } \Gamma_{\text{WinSiz\_3}}, \text{ Attention Heads } \bar{\Gamma}_{\text{HExp}}^{2} \\ \text{Inupt Dim } \Gamma_{\text{Emb\_3}}, \text{ Outupt Dim } \Gamma_{\text{Emb\_4}} \\ \text{Window Size } \Gamma_{\text{WinSiz\_3}}, \text{ Attention Heads } \bar{\Gamma}_{\text{HExp}}^{2} \end{bmatrix} \times 1$ </td></tr><tr><td>Stage 4</td><td>Multi-Scale Block [66]</td><td> $\begin{bmatrix} \text{Inupt Dim } \Gamma_{\text{Emb\_4}}, \text{ Outupt Dim } \Gamma_{\text{Emb\_4}} \\ \text{Window Size } \Gamma_{\text{WinSiz\_4}}, \text{ Attention Heads } \bar{\Gamma}_{\text{HExp}}^{3} \end{bmatrix} \times 2$ </td></tr></table>

## 6.2 Implementation Details

We use approximately 10% of the SA-V dataset (∼ 11K videos) as our training data for eficient distillation. Following SAM2, we adopt the AdamW optimizer with an initial learning rate of $5 \times 1 0 ^ { - 4 }$ . All models are trained for 5 epochs on an A100 (80GB) $\operatorname { G P U }$ with a batch size of 5 and 8-frame sequences per sample. Each training iteration uses a $4 \times 4$ grid of spatial prompts for mask prediction by the teacher model, and we set the weighting coeficients $\alpha = 1$ and beta = 1 to balance temporal consistency and hierarchical knowledge transfer. The total training time is approximately 65 hours.

In our progressive model contraction framework, we start with a hierarchical ViT seed (Hiera-B) and progressively shrink one architectural axis at a time - embedding dimension $\left( { \cal I } _ { \mathrm { E m b } } \right)$ , number of blocks $( { \cal T } _ { \mathrm { B l k } } )$ , window size $\left( { \cal T } _ { \mathrm { W i n S i z } } \right)$ and head expansion $\left( { \cal I } _ { \mathrm { H E x p } } \right)$ . Each reduction step produces lighter candidate models, all trained with the same distillation recipe, and evaluated on both accuracy and parameter count. The best candidate under a parameter budget is selected after each iteration, repeating this process until the desired eficiency or diminishing returns are observed. This approach avoids expensive large-scale searches, explicitly linking architectural choices to HyperKD and downstream validation rather than merely model size. Our lightweight architecture search resulted in approximately 15 candidate models; each candidate was quickly evaluated via a proxy: the image encoder was trained for a single epoch on roughly 10% of SA-V, and candidates were selected based on accuracy measured on LVOS - each proxy run required ${ \sim } 8$ hours. LVOS was intentionally chosen, as our objective is long-term temporal consistency (a core strength of SAM2), and LVOS serves as the most relevant benchmark for this goal.

## 6.3 Evaluation Metrics

We use two widely adopted evaluation metrics, region similarity (J) and contour accuracy (F), following the DAVIS benchmark [52].

– Region Similarity (J): This metric calculates the Intersection over Union (IoU) between the ground truth (G) and the prediction (M), defined as:

$$
J = \frac {M \cap G}{M \cup G}.\tag{10}
$$

– Contour Accuracy (F): This metric evaluates the precision of the segmentation boundary by computing the harmonic mean of contour recall $\left( \mathrm { P _ { c } } \right)$ and contour precision $\mathrm { ( R _ { c } ) }$ , defined as:

$$
\mathrm{F} = \frac {2 \cdot \mathrm {P_ {c}} \cdot \mathrm {R_ {c}}}{\mathrm {P_ {c}} + \mathrm {R_ {c}}}.\tag{11}
$$

Following the YouTube-VOS benchmark [68], we separately report performance on seen and unseen categories. We calculate the final J&F score as follows:

$$
\mathrm {J\ & F = \frac {J_ {\mathrm{s}} + F_ {\mathrm{s}} + J_ {\mathrm{u}} + F_ {\mathrm{u}}}{4}},\tag{12}
$$

where subscripts s and u denote scores for seen and unseen categories, respectively. This separate evaluation for seen and unseen categories provides a clearer measure of the model’s generalization ability in video object segmentation.

## 6.4 Additional Discussions

Parameter Studies To investigate the influence of diferent loss function components in HyperKD, we perform a parameter study on the weighting coeficients α and $\beta$ in loss function Eq. (9). We adjust α and $\beta$ over the range $[ 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 0 ^ { 1 } , 1 0 ^ { 2 } ]$ , while keeping the other parameter fixed at 1. The results are summarized in Table 11 and Table 12.

In Table 11, increasing α from $1 0 ^ { - 2 }$ to 1 significantly improves performance, indicating that the enforcement of temporal consistency is crucial for video segmentation. At $\alpha = 1$ , the model achieves the best J&F score of 69.0. For larger α values $( \alpha = 1 0 \mathrm { o r } \alpha = 1 0 ^ { 2 } )$ , performance degrades because an excessive emphasis on temporal consistency might distort the boundaries of objects.

Table 11: Parameter Study on α in loss function Eq. (9). We adjust α while keeping $\beta = 1$ . The best performance is achieved when $\alpha = 1$

<table><tr><td>Model Architecture</td><td colspan="5">MobileSAM2-23M</td></tr><tr><td> $\alpha$ </td><td> $10^{-2}$ </td><td> $10^{-1}$ </td><td>1</td><td> $10^{1}$ </td><td> $10^{2}$ </td></tr><tr><td>J&amp;F</td><td>60.2</td><td>65.7</td><td>69.0</td><td>66.8</td><td>61.4</td></tr></table>

In Table 12, similar to α, performance improves as $\beta$ increases to 1, reaching a peak of 69.0. For very small $\beta$ values $( \beta = 1 0 ^ { - 2 } \ \mathrm { { o r } } \ \beta = 1 0 ^ { - 1 } )$ , the model lacks suficient hierarchical feature guidance, leading to lower accuracy. For large $\beta$ values $( \beta = 1 0 ^ { 1 } \mathrm { o r } \beta = 1 0 2 )$ , performance drops, possibly due to overregularization from hierarchical distillation.

Table 12: Parameter Study on $\beta$ in loss function Eq. (9). We adjust $\beta$ while keeping $\alpha = 1$ . The best performance is achieved when $\beta = 1$

<table><tr><td>Model Architecture</td><td colspan="5">MobileSAM2-23M</td></tr><tr><td> $\beta$ </td><td> $10^{-2}$ </td><td> $10^{-1}$ </td><td>1</td><td> $10^{1}$ </td><td> $10^{2}$ </td></tr><tr><td>J&amp;F</td><td>59.8</td><td>64.1</td><td>69.0</td><td>67.2</td><td>62.0</td></tr></table>

Comparison with Diferent Hypergraph Construction Methods To investigate the impact of hypergraph formulation on knowledge distillation, we compare Fuzzy Hypergraph [48] and our proposed HyperKD-based hypergraph construction. The results are shown in Table 13. Without any hypergraph-based knowledge distillation, the MobileSAM2-23M model achieves only 44.8 J&F. This shows that direct feature distillation without hypergraph modeling is insuficient for efective lightweight segmentation. The Fuzzy Hypergraph [48] method improves the performance to 62.4 J&F, indicating that using hypergraphs to model multi-way relationships enhances knowledge transfer. However, Fuzzy Hypergraph [48] relies on predefined edge weights and lacks explicit high-order structure, which limits its ability to capture complex relationships in video segmentation.

HyperKD achieves the highest J&F score of 69.0, significantly outperforming Fuzzy Hypergraph by +6.6. This indicates that HyperKD’s hypergraph formulation efectively captures high-order relations, leading to better temporal and granularity-aware knowledge transfer.

Table 13: Comparison of Diferent Hypergraph Construction Methods. We evaluate the impact of diferent hypergraph formulations on the LVOS validation set (J&F). HyperKD outperforms Fuzzy Hypergraph.

<table><tr><td colspan="2">Model Architecture</td><td colspan="2">MobileSAM2-23M</td></tr><tr><td>Hypergraph Construction Method</td><td>No Distill</td><td>Fuzzy Hypergraph [48]</td><td>HyperKD (Ours)</td></tr><tr><td>J&amp;F</td><td>44.8</td><td>62.4</td><td>69.0</td></tr></table>

Comparison to non-SAM2 Architecture Baseline in Semi-supervised VOS We conduct experiments to compare MobileSAM2 with MobileVOS shown in Table below. Under the same 11K SA-V dataset training and semi-supervised VOS protocol, MobileSAM2 surpasses MobileVOS [46] with a stronger accuracysize trade-of, enabled by HyperKD and architectural refinements that transfer SAM2’s temporal and multi-granularity knowledge with minimal loss, validating MobileSAM2’s efectiveness and novelty.

Table 14: Comparison to non-SAM2 architecture baseline in semi-supervised VOS. For fair comparsion, all methods process video sequences..

<table><tr><td rowspan="2">Model</td><td rowspan="2" colspan="2">Distillation Method Params. (M)</td><td colspan="5">J&amp;F</td></tr><tr><td colspan="5">MOSE val DAVIS2017 val LOVS val SA-V val SA-V test</td></tr><tr><td colspan="8">~ 10% Training Data with 1 A100 GPU: 11K Videos from SA-V manual data</td></tr><tr><td>MobileNetV2 w/o ASPP</td><td>MobileVOS</td><td>1.9</td><td>35.6</td><td>40.7</td><td>36.8</td><td>37.4</td><td>37.1</td></tr><tr><td>MobileNetV2</td><td>MobileVOS</td><td>2.5</td><td>36.2</td><td>41.3</td><td>37.5</td><td>38.1</td><td>38.0</td></tr><tr><td>ResNet18</td><td>MobileVOS</td><td>8.1</td><td>37.8</td><td>42.5</td><td>41.2</td><td>40.6</td><td>41.0</td></tr><tr><td>MobileSAM2-5M</td><td>HyperKD (Ours)</td><td>5.8</td><td>44.6</td><td>51.0</td><td>48.8</td><td>49.2</td><td>49.4</td></tr><tr><td>MobileSAM2-10M</td><td>HyperKD (Ours)</td><td>10.4</td><td>48.7</td><td>54.2</td><td>51.8</td><td>49.9</td><td>50.0</td></tr><tr><td>MobileSAM2-23M</td><td>HyperKD (Ours)</td><td>23.7</td><td>65.8</td><td>72.1</td><td>69.0</td><td>66.4</td><td>67.8</td></tr></table>

Comparison to Recent Distillation Baseline We compare HyperKD to EA-KD [59] under identical datasets/protocols below, indicating more efective transfer of temporal and multi-granularity knowledge.

Analysis of Data Proportion in HyperKD Table 16 shows the impact of the proportion of training data on HyperKD. J&F scores increase from 45.2% (1% data) to 73.0% (20% data), with the largest gain between 5% and 10% (+13.2%). In particular, with 10% data, MobileSAM2-23M achieves 69.0% J&F, reaching 94.5% of the performance at 20%, demonstrating HyperKD’s data eficiency. Beyond 15%, gains diminish (+1.7%), indicating that HyperKD efectively extracts core knowledge with moderate data, reducing reliance on large-scale datasets. These results show that HyperKD enables near-optimal performance with only 10-15% of data, making it ideal for low-resource training, real-time applications, and mobile deployment.

Table 15: Comparison to Recent Distillation Baselines. We evaluate the impact of diferent hypergraph formulations on the LVOS validation set (J&F).

<table><tr><td colspan="2">Model Architecture</td><td colspan="2">MobileSAM2-23M</td></tr><tr><td>Distillation Method</td><td>No Distill</td><td>EA-KD [59]</td><td>HyperKD (Ours)</td></tr><tr><td>LOVS (J&amp;F)</td><td>44.8</td><td>63.6</td><td>69.0</td></tr></table>

Table 16: Impact of Training Data Proportion on HyperKD Performance. We report J&F scores (higher is better) on the LVOS validation set. HyperKD achieves competitive performance even with limited data.

<table><tr><td>Model Architecture</td><td>MobileSAM2-23M</td></tr><tr><td>Training Data Proportion</td><td>1% 5% 10% 15% 20%</td></tr><tr><td>J&amp;F</td><td>45.2 55.8 69.0 71.3 73.0</td></tr></table>

Comparison with lightweight non-distillation models As shown in Table 17, we conduct experiments to compare MobileSAM2 with FastSAM [84] and MobileSAM [79], using the same evaluation datasets and evaluation protocols. As shown in Table below, fastSAM with 68M parameters, achieves a J&F score of 48.9%. MobileSAM with 5.78M parameters attains a considerably lower J&F score of 32.2%. In contrast, our MobileSAM2 achieves a J&F score of 48.8%, 51.8%, and 69.0% with only 5.84M, 10.37M, and 23.74M parameters, respectively, achieving a much better trade-of between model size and performance. The superior performance of MobileSAM2 largely attributed to our carefully designed HyperKD and architectural improvements, which efectively model and transfer SAM2’s rich temporal knowledge and diverse multi-granularity knowledge without excessive loss of accuracy. These results validate both the efectiveness and novelty of MobileSAM2 for lightweight segmentation.

Analysis of task-specific segmentation loss We evaluated HyperKD both with and without task-specific segmentation losses (i.e., cross-entropy using available masks) to assess their impact on optimization stability and performance. As shown in Table 18, removing task-specific losses improves performance by +1.8 J&F. We attribute this to the fact that task losses, derived from a limited 10% labeled dataset, may introduce noise and conflict with the rich structural knowledge captured by the teacher. In contrast, HyperKD’s hypergraph-based supervision provides multi-level and high-quality guidance (temporal and granularity) that stabilizes optimization and reduces the need for additional task-specific objectives.

Table 17: Comparison with lightweight non-distillation models. We report Parameters (MB) and J&F scores (higher is better) on the LVOS validation set.

<table><tr><td>Model Architecture</td><td>Params (MB)</td><td>J&amp;F</td></tr><tr><td>FastSAM [84]</td><td>68</td><td>48.9</td></tr><tr><td>MobileSAM [79]</td><td>5.78</td><td>32.2</td></tr><tr><td>MobileSAM2-5M</td><td>5.84</td><td>48.8</td></tr><tr><td>MobileSAM2-10M</td><td>10.37</td><td>51.8</td></tr><tr><td>MobileSAM2-23M</td><td>23.74</td><td>69.0</td></tr></table>

Table 18: Analysis of task-specific segmentation loss. We evaluate the impact of using task-specific segmentation loss on the LVOS validation set (J&F).

<table><tr><td colspan="2">Model Architecture</td><td colspan="2">MobileSAM2-23M</td></tr><tr><td>Distillation Method</td><td>No Distill</td><td>HyperKD (w/ task-specific loss)</td><td>HyperKD (w/o task-specific loss)</td></tr><tr><td>J&amp;F</td><td>44.8</td><td>67.2</td><td>69.0</td></tr></table>

Study Limitations The progressive contraction strategy used to derive lightweight student models involves manually designed search spaces (e.g., embedding dimensions, number of blocks, attention head sizes). It lacks the automation and optimality guarantees of neural architecture search, potentially limiting the performance ceiling of MobileSAM2.

Visualization of High-Order Learning in HyperKD Fig. 3 visualizes the hypergraph afinity structure via query-pixel similarity. For a selected query point on an object (marked in green), we compute the cosine similarity in the teacher’s feature space between that point and every other spatial position, producing a heatmap that reveals which pixels the hypergraph considers semantically connected. The teacher’s hypergraph cleanly groups the queried object, with high similarity concentrated on the object’s full extent and minimal leakage to the background. The student distilled via HyperKD produces a closely matching similarity map, confirming that the core hypergraph structure—pixel-level semantic grouping—is successfully transferred from the 224M teacher to the 23M student.

## 6.5 Qualitative Results

We provide more qualitative illustrations of our MobileSAM2-24M. As shown in Figure Figure 4, MobileSAM2 produces segmentation across multiple video consistently.

![](images/52a7dc9267346c0d450c5d6463a3ededd69379db55be94680ae12f02c3ad690e.jpg)  
Fig. 3: Visualization of High-Order Learning in HyperKD.

![](images/01b414a43394a00e39453d44edbcc1c3a237cd52448ec4ed35eb6169b54fc12f.jpg)  
Fig. 4: Qualitative illustrations of MobileSAM2-24M, where each row refers to frames from one video and each masklet has one unique color.