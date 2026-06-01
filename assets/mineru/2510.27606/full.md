# Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning

Yuhong Liu1,2, Beichen Zhang1,3, Yuhang Zang1B, Yuhang Cao1, Long Xing1

Xiaoyi Dong1, Haodong Duan1, Dahua Lin1,3, Jiaqi Wang1, 4B

1Shanghai AI Laboratory 2Shanghai Jiao Tong University

3The Chinese University of Hong Kong 4Shanghai Innovation Institute

{liuyuhong1,zangyuhang}@pjlab.org.cn BCorresponding Authors.

Code: https://github.com/InternLM/Spatial-SSRL

Model: https://huggingface.co/internlm/Spatial-SSRL-7B

https://huggingface.co/internlm/Spatial-SSRL-Qwen3VL-4B

Data: https://huggingface.co/datasets/internlm/Spatial-SSRL-81k

![](images/741e4cfaa89db981ab0a947f85151e403d059b24c71cc790632c5f4f3cedb3c6.jpg)

<details>
<summary>natural_image</summary>

Person sitting on a green bench overlooking a waterfront with a cat and boat nearby (no visible text or symbols)
</details>

Consider the real-world 3D locations of the objects. Which object has a lower location?

A. Orange cat B. White boat

# Qwen2.5-VL-7B:

To determine which object has a lower location..

1. Identify the objects in question: …

2. Compare the relative positions:

3. Determine the lower location: - The ground (where the cat is sitting) is lower than the water (where the boat is floating).

The final answer is: A. orange cat

![](images/c062df507fd3bbf2388fae849a3f7593813f2c1ae9c75361cbd2442c8dfb1a6c.jpg)

# Spatial-SSRL-7B:

To determine which object is lower…

1. The orange cat is sitting on a bench, which is at a certain height above the ground.

2. The white boat is floating on the water, which is lower than the ground.

Since the water is lower than the bench, the white boat is at a lower location than the orange cat.

The final answer is: B

![](images/cf3efc6c7213d3861928fe6e736962c8cca32e59d604f59c468ab56602117c56.jpg)

![](images/8871d27583fa9d3c502edad799f772d6149d8107a38096ffe66c3faac709594c.jpg)

<details>
<summary>text_image</summary>

Street photo of a cyclist with helmet and bicycle, showing visible store signboards and vehicle license plate
</details>

Consider the real-world 3D locations and orientations of the objects. Which side of the person on bicycle is facing the “Santa Cruz County Bank” sign? A. Back B. Right C. Front D. Left

# Qwen2.5-VL-7B:

To determine which side of the person…

1. Identify the sign:… 2. Observe the person: …

3. Determine the orientation of the person: The person is facing forward…

4. Consider the relative positions: The sign is on the right side of the image, and the person is facing forward, which means the right side of the person is facing the sign.

The final answer is: B. Right

# Spatial-SSRL-7B:

To determine which side of the person on the bicycle is facing the sign…

1. The "Santa Cruz County Bank" sign is on the right side of the image.

2. The person on the bicycle is facing the camera, which means their back is towards the sign.

3. Therefore, the person's back is facing the sign. The final answer is: A

![](images/b6be1a9b7062817601020f2dd39041e3df9156a91e396c766bfc423667fd68e1.jpg)

<details>
<summary>radar</summary>

| Category       | Spatial-SSRL-7B | Qwen2.5-VL-7B | Qwen2.5-VL-7B (CoT) |
| -------------- | ---------------- | ------------- | ------------------- |
| Spatial457     | 53.34            | 44.88         | 44.67               |
| VSI-Bench      | 39.29            | 38.08         | 32.69               |
| ViewSpatial    | 37.81            | 36.83         | 36.66               |
| What'sUp       | 90.61            | 86.95         | 70.61               |
| Qspatial-plus  | 52.48            | 46.53         | 46.53               |
| SpatialEval    | 64.03            | 62.37         | 61.12               |
| 3DSRBench      | 56.53            | 53.39         | 54.55               |
</details>

(a) Comparison of Responses to Spatial Understanding Problems   
(b) Comparison of Performance   
Figure 1. We present Spatial-SSRL, a self-supervised reinforcement learning paradigm for spatial understanding. (a) Qualitative examples: the baseline answers are wrong (red), whereas our model predicts correctly (green) for 3D locations and orientations. (b) Quantitative results on seven spatial benchmarks show consistent improvements of Spatial-SSRL-7B against Qwen2.5-VL-7B and its CoT variant.

# Abstract

Spatial understanding remains a weakness of Large Vision-Language Models (LVLMs). Existing supervised fine-tuning (SFT) and recent reinforcement learning with verifiable rewards (RLVR) pipelines depend on costly supervision, specialized tools, or constrained environments that limit scale. We introduce Spatial-SSRL, a self-supervised RL paradigm that derives verifiable signals directly from ordinary RGB or RGB-D images. Spatial-SSRL automatically formulates five pretext tasks that capture 2D and 3D spatial structure: shuffled patch reordering, flipped patch recognition,

cropped patch inpainting, regional depth ordering, and relative 3D position prediction. These tasks provide groundtruth answers that are easy to verify and require no human or LVLM annotation. Training on our tasks substantially improves spatial reasoning while preserving general visual capabilities. On seven spatial understanding benchmarks in both image and video settings, Spatial-SSRL delivers average accuracy gains of 4.63% (3B) and 3.89% (7B) over the Qwen2.5-VL baselines. Our results show that simple, intrinsic supervision enables RLVR at scale and provides a practical route to stronger spatial intelligence in LVLMs.

![](images/57906269534617e16f2ebfc28cd0793f0ef1f8a6b651146d5bb5908c985ec499.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) Extrinsic Supervisory Signals
        A["Data Collection"] --> B["Annotated Images"]
        B --> C["Limited Raw Data"]
        C --> D["Additional Annotation"]
        D --> E["Specialized Models"]
        E --> F["High Cost"]
        E --> G["Tool-Heavy"]
        F --> H["Model Optimization"]
        G --> H
        H --> I["Extrinsic Supervision"]
        I --> J["Policy model"]
    end

    subgraph (b) Intrinsic Supervisory Signals
        K["Data Collection"] --> L["Scalable Raw Data"]
        L --> M["Data Augmentation"]
        M --> N["Model Optimization"]
        N --> O["Template QAs"]
        N --> P["Augmented Data"]
        O --> Q["Intrinsic Supervision"]
        P --> Q
        Q --> R["Policy model"]
    end

    A --> C
    B --> D
    C --> E
    D --> F
    E --> G
    F --> H
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S["Verifiable"]
    R --> T["QA-Acc =100%"]
    R --> U["Scalable"]
```
</details>

Figure 2. (a) Prior pipelines boost spatial understanding by injecting extrinsic supervision from expert tools or synthetic environments, which inflates cost and limits scalability. (b) Our Spatial-SSRL replaces these dependencies with intrinsic self-supervision, yielding a scalable, lightweight, low-cost, and naturally verifiable pipeline.

# 1. Introduction

Spatial understanding is pivotal for Large Vision-Language Models (LVLMs) to analyze complex real-world scenes. As shown in Fig. 1 (a), the ability to reason over depth, distance, azimuth, and relative object positions enables faithful reconstruction of 3D environments and unlocks applications such as autonomous driving [55, 59], robot manipulation [3, 14, 29, 54], and embodied navigation [8, 20]. Although LVLMs report near-saturated results on various tasks (e.g., visual question answering [6, 39], image captioning [67, 69], object segmentation [16, 75], math reasoning [43, 71]), their spatial understanding remains substantially below human performance [34, 41, 49, 70].

Early data-centric Supervised Fine-Tuning (SFT) approaches [4, 9] advance spatial understanding by synthesizing spatial question-answer pairs from 2D images [4] or constructing single-image 3D scene graphs with a depth plugin [9]. However, SFT tends to memorize datasetspecific patterns [5, 10], inherits errors from detectors and monocular depth, and often depends on expensive proprietary models for question-answering curation. Recent Reinforcement Learning with Verifiable Reward (RLVR) methods [33, 37, 64] improve the generalization of spatial understanding over SFT by optimizing with verifiable rewards [18, 30], but are constrained to specific environments (e.g., 3D scans) and demand substantial pipeline engineering with limited domain coverage. As shown in Fig. 2 (a), a key open challenge is to retain the optimization benefits of RL while scaling verifiable supervision to ordinary images across diverse domains without manual labels, specialized assets, or costly tooling.

To tackle this challenge, we draw on visual selfsupervised learning (SSL), which learns visual representations from intrinsic structure in data via pretext objectives such as contrastive alignment and permutation/jigsaw tasks [7, 25, 51] without manual labels. Our key idea is that intrinsic consistency signals in ordinary 2D or RGB-D images (e.g., relative depth, geometric consistency, correspondence, or invariance under view transformations) naturally supervise spatial understanding. Because SSL targets are defined by the pretext itself, their correctness is deterministically verifiable during training, making SSL well aligned with the RLVR training paradigm [76]. Unlike prior SSL approaches that are primarily used for visual encoder pretraining [57], we repurpose SSL objectives as verifiable reward functions to directly optimize LVLM behavior via RL, shifting supervision from representation quality to spatial understanding while remaining broadly applicable across diverse image domains.

Drawing on this insight, we introduce Spatial-SSRL, a Self-Supervised Reinforcement Learning paradigm for improving the spatial understanding of LVLMs. According to Fig. 2 (b), our framework defines a suite of verifiable selfsupervised tasks constructed solely from RGB-D inputs, organized into two categories: depth-free and depth-based. Depth-free tasks target 2D structure, including relative position between regions, permutation ordering, and crossview correspondence. Depth-based tasks exploit depth to supervise relative depth or distance ranking and 3D relation consistency under perspective transformations. Each task is posed as a question–answer prompt to the LVLM, with a deterministic verifier producing binary or scalar rewards. We then optimize the model with Group Relative Policy Optimization (GRPO) [53], using the verifiable SSL task reward and an initial cold-start stage to stabilize RL training.

Our Spatial-SSRL approach is cost-effective, scalable, and naturally compatible with the RLVR paradigm. Data curation is fully automated: it requires only raw RGB or RGB-D images and uses no human labels or auxiliary proprietary models. Inputs can be collected at scale from commodity depth sensors and public sources [11, 52, 56]. Because supervision comes from intrinsic image consistency rather than expensive task-specific annotations or tool chains, the pipeline generalizes across domains, is reproducible end-to-end, and is easy to extend with new pretext tasks without changing the data source.

As shown in Fig. 1 (b), Spatial-SSRL yields substantial gains in seven spatial understanding benchmarks [28, 32, 36, 44, 58, 60, 68]. We observe consistent accuracy improvements across benchmarks in both image and video settings. Representative sub-tasks include relative orientation [44, 60], depth ordering [44], and multi-view relative location [28, 32]. Our largest gain is +8.67% on Spatial457 [60], which requires precise perception, strong spatial interpretation, and multi-step reasoning. Despite the targeted RL fine-tuning, the models retain their performance on general visual question-answering (VQA), multi-image understanding, and hallucination diagnostics [17, 21, 39], showing no regression in non-spatial capabilities. Comprehensive ablations in Sec. 4.3 further indicate that both depth-free (2D layout, permutation, correspondence) and depth-based (relative depth, 3D relation) pretexts contribute to the RLVR optimization.

Our main contributions are: 1) We introduce Spatial-SSRL, a self-supervised reinforcement learning paradigm for improving LVLM spatial understanding. Spatial-SSRL is cost-effective, scalable, naturally compatible with the RLVR paradigm, and easy to extend to new SSL tasks. 2) We design a suite of self-supervised tasks for spatial perception, covering both depth-free and depth-based objectives. Ablation studies show that each task contributes to improved spatial understanding. 3) We provide new insights into curating high-quality, challenging SSL data for RL training, opening a new avenue for effectively combining RLVR and SSL for improving spatial understanding.

# 2. Related Work

LVLM Spatial Understanding. Previous studies have been carried out to improve the spatial understanding of LVLMs. The most straightforward approach is to optimize LVLMs on high-quality question-answer (QA) pairs curated via manual annotation [3] or proprietary models [50]. However, these approaches lack scalability due to their high cost. A more affordable alternative uses public datasets (e.g., ScanNet [11]) with abundant meta-information [48], but the scale and richness of the curated data are inherently constrained by the datasets employed.

To construct large-scale spatial QAs in a cost-effective manner, two dominant paradigms have emerged: tool-based and simulation-based approaches. Tool-based approaches incorporate tools within their pipelines. The tools are either open-source models that struggle with spatial understanding [9], or expert models [4, 12, 27] for object detection, segmentation, and depth estimation, resulting in an overly complex pipeline with additional computational cost (depicted in Fig. 2). Simulation-based approaches render 3D scenes and synthesize QAs [72, 74], whose quality remains unsatisfactory due to divergence from real-world scenarios. In contrast, Spatial-SSRL provides a novel paradigm featuring a tool-free pipeline, real-world consistency, costeffectiveness, and high scalability.

Self-Supervised Learning. Self-supervised learning (SSL), which obtains supervision from the inherent structure of the data itself without relying on manual labels [22], has been remarkably effective for visual representation learning. Early approaches learn visual features via contrastive learning [7, 25] or self-supervised tasks such as rotation [19], jigsaw [47], and temporal ordering [46]. With the rise of LLMs and LVLMs, SSL has become prevalent in pre-training. Pre-training of autoregressive LMs (GPT [2]), masked language models (BERT [13]), and masked autoencoders (MAE [26]) all follow the paradigm of adding masks to parts of the data and learning to predict the masked content. However, strategies for using SSL in the LVLM posttraining phase remain limited.

Recently, several approaches appear to enhance LVLMs via self-supervised post-training [24, 62, 65]. Jigsaw-R1 [62] uses jigsaw puzzles for RL. Visual Jigsaw [65] is a concurrent work that constructs jigsaw tasks across three different modalities. Our work covers a broader set of self-supervised tasks, with Jigsaw as only one component. SSL4RL [24] focuses solely on 2D tasks, while our work targets spatial understanding and designs verifiable supervisory signals based on both 2D and RGB-D images.

Reinforcement Learning with Verifiable Rewards. Following the success of Deepseek-R1 [23], many recent works have applied Group Relative Policy Optimization [53] and demonstrated the potential of reinforcement learning with verifiable rewards (RLVR) [42, 48, 61, 63, 73]. For example, Visual-RFT improves performance on image classification, detection, and grounding [42], while other approaches incentivize math reasoning via RLVR [61, 73].

In the realm of spatial understanding, existing methods mainly rely on annotation-rich public datasets and toolheavy pipelines [33, 37, 48] to curate training data for RL. Such approaches are unable to fully exploit the advantages of RL because the quality and scale of their data are limited by the specialized assets they employ. To address this problem, we introduce a novel paradigm that scales verifiable supervision and enhances LVLM spatial understanding.

# 3. Spatial-SSRL

The overview of Spatial-SSRL is shown in Fig. 3. Our framework consists of two stages: self-supervised task design (Sec. 3.1) and reinforcement learning (Sec. 3.2). The design is guided by three principles that distinguish our approach from prior work: (i) zero human or LLM supervision: all ground-truth labels are derived deterministically from image structure; (ii) tool-free scalability: no external detection, segmentation, or rendering pipelines are required; and (iii) natural verifiability: supervision signals are verifiable and suitable for RL reward computation.

![](images/6939d015b7ebba0e0c8a7884f6bab3fe6f680d37a734ebcecf5788915e38c057.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["(a) Self-Supervised Task Formulation"] --> B["Cropped Patch Inpainting"]
    B --> C["Flipped Patch Recognition"]
    C --> D["Shuffled Patch Reordering"]
    D --> E["Regional Depth Ordering"]
    E --> F["Relative Position Prediction"]
    F --> G["Verifiable Reward"]
    G --> H["Group Relative Policy Optimization"]
    H --> I["Reference Model"]
    
    subgraph A
        J["RGB Images"]
        K["RGB-D Images"]
        L["Data Collection"]
    end
    
    subgraph B
        M["Cropped Patch Inpainting"]
        N["Cropped Patch Recognition"]
        O["Cropped Patch Reordering"]
        P["Cropped Patch Inpainting"]
        Q["Cropped Patch Recognition"]
    end
    
    subgraph C
        R["A B C D"]
        S["A B C D"]
        T["A B C D"]
        U["A B C D"]
        V["A B C D"]
        W["A B C D"]
    end
    
    subgraph D
        X["A B C D"]
        Y["A B C D"]
        Z["A B C D"]
        AA["A B C D"]
        AB["A B C D"]
        AC["A B C D"]
    end
    
    subgraph E
        AD["Regional Depth Ordering"]
        AE["Regional Depth Ordering"]
        AF["Regional Depth Ordering"]
        AG["Regional Depth Ordering"]
        AH["Regional Depth Ordering"]
        AI["Regional Depth Ordering"]
        AJ["Regional Depth Ordering"]
    end
    
    subgraph F
        AK["Relative Position Prediction"]
        AL["Relative Position Prediction"]
        AM["Relative Position Prediction"]
        AN["Relative Position Prediction"]
    end
    
    subgraph G
        AO["Verifiable Reward: R = 0.9 × R_Acc + 0.1 × R_Fmt\nR_Acc = {1, if ans correct\n0, otherwise}\nR_Fmt = {1, if format correct\n0, otherwise}"]
    end
    
    subgraph H
        AP["Group Relative Policy Optimization"] --> AQ["Reference Model"]
    end
    
    subgraph I
        AR["Reference Model"] --> AS["Reference Model"]
    end
    
    subgraph J
        AT["81k images & QA pairs"] --> AU["Policy model"]
    end
    
    subgraph K
        AU --> AV["Data Collection"]
    end
    
    subgraph L
        AW["Format Prompt: First THINK about the reasoning process within <think> tags. Then put the final answer in \boxed{.}"]
    end
    
    subgraph M
        AX["QR code: 0, 1"] --> AY["QR code: 0, 1"] & AX
    end
```
</details>

Figure 3. Overview of Spatial-SSRL. (a) Self-supervised data curation: from raw RGB and RGB-D images, we automatically construct five pretext tasks (patch reordering, patch flip detection, cropped-patch inpainting, regional depth ordering, and relative 3D position prediction), requiring no human or LLM annotations. (b) RL training: the model is optimized with Group Relative Policy Optimization (GRPO) using a verifiable reward function that evaluates answer correctness, and a format reward that elicits format compliance.

Task Design. To comprehensively enhance spatial understanding, we design five complementary pretext tasks in total. Shuffled patch reordering requires recovering global 2D layout consistency from permuted patches. Flipped patch recognition demands sensitivity to mirror symmetries and local orientation cues. Cropped patch inpainting tests texture-context matching and fine-grained structural reasoning. Regional depth ordering evaluates ordinal depth perception across image regions. Relative 3D position prediction assesses egocentric spatial relations (left-/right, front/back) conditioned on object orientation. These tasks jointly encourage both 2D layout understanding and 3D spatial reasoning, providing unambiguous targets for RLVR optimization. Importantly, the Spatial-SSRL framework is modular and extensible: additional self-supervised

tasks can be integrated without modifying the RL pipeline, enabling future work to expand spatial coverage.

# 3.1. Self-Supervised Task Design

We organize our five self-supervised tasks into two complementary categories that target different aspects of spatial intelligence. The three depth-free tasks (Sec. 3.1.1) operate solely on RGB images and emphasize 2D spatial relationships, including patch-level layout, structural consistency, and fine-grained texture-context correspondence. The two depth-based tasks (Sec. 3.1.2) use depth information to supervise 3D scene understanding, focusing on ordinal depth perception and egocentric spatial relations. Together, these tasks provide comprehensive coverage of spatial reasoning capabilities.

# 3.1.1. Depth-free Tasks

RGB images contain rich intrinsic structural information that can serve as self-supervision without external annotation. By applying deterministic augmentation operations solely to pixel-level content, we construct Question-Answer (QA) pairs for three complementary tasks: Shuffled patch reordering, Flipped patch recognition, and Cropped patch inpainting. Each task exploits different aspects of image structure to promote spatial understanding.

Shuffled Patch Reordering. Motivation: Recovering the original order of shuffled patches requires understanding global 2D layout consistency and relative positional relationships, skills that directly transfer to reasoning about object arrangements in real scenes.

Formulation: Given an image $I \in \mathbb { R } ^ { H \times W }$ , we partition where it into $\begin{array} { r } { P _ { H } = \frac { H } { M } } \end{array}$ $M \times N$ grid of patches, each of size and $\begin{array} { r } { P _ { W } = \frac { W } { N } } \end{array}$ H  W . We denote the patch grid $P _ { H } \times P _ { W }$ , as $\mathcal { X } \ = \ \{ \boldsymbol { x } _ { i , j } \in \ \mathbb { R } ^ { P _ { H } \times P _ { W } } \ \stackrel { \cdot \cdot } { | } \ i \ \in \ [ 0 , M ) , j \ \in \ [ 0 , N ) \}$ , where $x _ { i , j } = I ( i P _ { H } : ( i + 1 ) P _ { H } , j P _ { W } : ( j + 1 ) P _ { W } )$ . For convenience, we flatten X into a sequence $\hat { \mathcal { X } } \quad =$ $\left[ \hat { x } _ { 0 } , \hat { x } _ { 1 } , \dots , \hat { x } _ { M \times N - 1 } \right]$ with $\hat { x } _ { k } = x _ { i , j }$ and $k = i N + j$ . We then apply a random permutation $\pi : \{ 0 , \ldots , M \times N { - } 1 \} \mapsto$ $\{ 0 , \dots , M \times N - 1 \}$ to generate the shuffled sequence $\hat { \mathcal { X } } _ { \mathrm { s h u f f e } } ~ = ~ [ \hat { x } _ { \pi ( 0 ) } , \hat { x } _ { \pi ( 1 ) } , \dots , \hat { x } _ { \pi ( M \times N - 1 ) } ]$ , from which we reconstruct the shuffled input image $I _ { \mathrm { i n p u t } }$ .

Task objective: The model is prompted to predict the correct patch ordering that restores the original image. Since π is bijective, its inverse $\pi ^ { - 1 }$ exists and provides the groundtruth answer:

$$
\pi^ {- 1} = [ \pi^ {- 1} (0), \pi^ {- 1} (1), \dots , \pi^ {- 1} (M \times N - 1) ]. \tag {1}
$$

Difficulty enhancement: To increase task complexity and prevent trivial edge-matching solutions, we optionally mask one random patch $\hat { x } _ { \pi ( t ) }$ by setting all pixels to white: $\hat { x } _ { \pi ( t ) } ( 0 : P _ { H } , 0 : P _ { W } ) = 2 5 5$ . This forces the model to rely on global layout rather than local patch boundaries. Implementation details are provided in Appendix A.2.

Flipped Patch Recognition. Motivation: Detecting subtle orientation violations requires sensitivity to local geometry, mirror symmetries, and directional cues such as text, faces, and shadows. These capabilities are essential for understanding viewpoint-dependent spatial relations.

Formulation: Given an image $I \in \mathbb { R } ^ { H \times W }$ , we partition it into patches using the same grid structure as shuffled patch reordering, yielding $\hat { \mathcal X } = \{ \hat { x } _ { k } \}$ with $k \in [ 0 , M \times N -$ 1].

We randomly select one patch $\hat { x } _ { t }$ and apply a flip operation f : ˆxt 7→ xˆflit $f : { \hat { x } } _ { t } \mapsto { \check { x } } _ { t } ^ { \mathrm { f l i p } }$ p defined as:

$$
\left\{ \begin{array}{l l} x _ {\text { vert }}, & \text { with   probability   0.5 }, \\ x _ {\text { horz }}, & \text { with   probability   0.5 }, \end{array} \right. \tag {2}
$$

where $x _ { \mathrm { v e r t } } ( r , c ) = x ( P _ { H } - 1 - r , c )$ (vertical flip) and $x _ { \mathrm { h o r z } } ( r , c ) = x ( r , P _ { W } - 1 - c )$ (horizontal flip). The input image $I _ { \mathrm { i n p u t } }$ is reconstructed from the modified sequence $\hat { \mathcal { X } } ^ { \prime } = [ \hat { x } _ { 0 } , \dotsc , \hat { x } _ { t } ^ { \mathrm { H i p } } , \dotsc , \hat { x } _ { M \times N - 1 } ]$ .

Task objective: The model must identify both the index t of the flipped patch and its flip direction $d \in \{ \mathrm { v e r t } , \mathrm { h o r z } \}$ . The ground-truth answer is the tuple [t, d].

Cropped Patch Inpainting. Motivation: Identifying which patch correctly fills a masked region requires analyzing texture continuity, semantic context, and structural consistency between local content and its surroundings. These skills generalize to understanding spatial coherence.

Formulation: Given an image $I \in \mathbb { R } ^ { H \times W }$ , we extract a square patch of side length $s = \operatorname* { m i n } ( H / 2 , W / 2 )$ whose top-left corner $( x _ { 0 } , y _ { 0 } )$ is sampled uniformly:

$$
\left(x _ {0}, y _ {0}\right) \sim \mathcal {U} ([ 0, H - s ] \times [ 0, W - s ]). \tag {3}
$$

The cropped region is $\mathcal { R } = [ x _ { 0 } , x _ { 0 } + s ] \times [ y _ { 0 } , y _ { 0 } + s ]$ , and we denote the extracted patch as $I _ { \mathrm { c r o p } } = I ( { \mathcal { R } } )$ . We construct the masked input image $I _ { \mathrm { i n p u t } }$ by zeroing out the cropped region:

$$
I _ {\text { input }} (u, v) = \left\{ \begin{array}{l l} 0, & (u, v) \in \mathcal {R}, \\ I (u, v), & \text { otherwise }. \end{array} \right. \tag {4}
$$

Task objective: The model is presented with $I _ { \mathrm { i n p u t } }$ and four candidate patches in a multiple-choice format, and must select the correct patch $I _ { \mathrm { c r o p } }$ that fills the masked region.

Distractor construction: To prevent trivial solutions based on low-level texture matching, we construct three challenging distractors that share substantial visual similarity with the ground-truth patch. The three distractors are: (1) a 90◦ rotated version of $I _ { \mathrm { c r o p } } ; ( 2 )$ the interior subregion $I _ { \mathrm { i n t } } = I _ { \mathrm { c r o p } } ( \mathcal { R } _ { \mathrm { i n t } } )$ , where $\begin{array} { r } { \mathcal { R } _ { \mathrm { i n t } } \stackrel { \cdot } { = } \left[ \frac { s } { 4 } , \frac { 3 s } { 4 } - 1 \right] \times \left[ \frac { s } { 4 } , \frac { 3 s } { 4 } - 1 \right] } \end{array}$ and (3) the exterior region $I _ { \mathrm { e x t } } = I ( \mathcal { R } _ { \mathrm { e x t } } )$ , where $\mathcal { R } _ { \mathrm { e x t } } ~ =$ $\left[ x _ { 0 } - \theta s , x _ { 0 } + ( 1 + \theta ) s \right] \times \left[ y _ { 0 } - \theta s , y _ { 0 } + ( 1 + \theta ) s \right]$ ] with $\theta \in \{ 0 . 2 5 , 0 . 5 \}$ . For $\mathcal { R } _ { \mathrm { e x t } }$ extending beyond image boundaries, we clip to $[ 0 , H ] \times [ 0 , W ]$ . All distractors are resized to $s \times s$ to prevent size-based discrimination, forcing the model to attend to fine-grained structural and semantic consistency. Please refer to Appendix A.4 for more details.

# 3.1.2. Depth-based Tasks

Depth maps encode per-pixel distances from the camera, providing rich geometric supervision for 3D spatial understanding without requiring manual annotation. We design two complementary depth-based tasks that supervise spatial reasoning from different perspectives: Regional depth ordering adopts a camera-centric view and tests ordinal depth perception across regions, while Relative position prediction adopts an object-centric view and evaluates egocentric spatial relations conditioned on object orientation. Together, these two tasks promote robust 3D scene comprehension.

Regional Depth Ordering. Motivation: Ranking regions by distance from the camera requires integrating depth cues, perspective understanding, and ordinal reasoning, which are foundational skills for tasks such as occlusion reasoning and 3D scene reconstruction.

Formulation: Given an image $I \in \mathbb { R } ^ { H \times W }$ and its normalized depth map $D \in \mathbb { R } ^ { H \times W }$ , we select three disjoint regions $R _ { 1 } , R _ { 2 } , R _ { 3 } \subseteq [ 0 , H ] \times [ 0 , W ]$ ] with increasing depth $( \mathrm { i . e . , } R _ { 1 }$ is closest to the camera, $R _ { 3 }$ is farthest). To ensure unambiguous ordering, we enforce the following constraints:

$$
r (R _ {i}) = \max _ {(x, y) \in R _ {i}} D (x, y) - \min _ {(x, y) \in R _ {i}} D (x, y) <   r _ {\max}, \tag {5}
$$

$$
d \left(R _ {i}, R _ {i + 1}\right) = \min _ {(x, y) \in R _ {i + 1}} D (x, y) - \max _ {(x, y) \in R _ {i}} D (x, y) > d _ {\text { min }}, \tag {6}
$$

where $r ( R _ { i } )$ denotes the depth range within region $R _ { i }$ and $d ( R _ { i } , R _ { i + 1 } )$ denotes the depth gap between consecutive regions. We set $r _ { \mathrm { m a x } } = 0 . 1 5$ and $d _ { \operatorname* { m i n } } = 0 . 0 5$ to guarantee well-separated regions with consistent internal depth. We then apply a random permutation $\hat { \pi } : \{ 1 , 2 , 3 \} \mapsto \{ 1 , 2 , 3 \}$ to assign visual labels, marking each region’s center with label $\hat { \pi } ( R _ { i } )$ on the image to construct $I _ { \mathrm { i n p u t } }$ .

Task objective: The model must order the labeled regions from closest to farthest. The ground-truth answer is the sequence $\hat { \pi } = [ \hat { \pi } ( 1 ) , \hat { \pi } ( 2 ) , \hat { \pi } ( 3 ) ]$ ].

Relative Position Prediction. Motivation: Determining spatial relations from an object’s perspective (e.g., “the cup is to the left of the book”) requires mental rotation, egocentric coordinate transformation, and integration of orientation cues with depth information.

Formulation: Given an image with depth map $D ,$ we select two pixel locations $R _ { 1 }$ and $R _ { 2 }$ . We assume an object is located at $R _ { 1 }$ with a specified orientation and the model is required to predict the relative position of $R _ { 2 }$ from the object’s viewpoint (e.g., left, right-front). The task is posed as a multiple-choice question with four options. The construction process and coordinate definitions are shown in Fig. 8.

Coordinate transformation: Let $( x _ { 1 } , y _ { 1 } , z _ { 1 } )$ and $\left( x _ { 2 } , y _ { 2 } , z _ { 2 } \right)$ denote the camera-frame coordinates of $R _ { 1 }$ and $R _ { 2 }$ , where $z _ { i } ~ = ~ D ( x _ { i } , y _ { i } )$ . The object’s orientation is defined by a rotation angle θ (counterclockwise from the camera’s z-axis to the object’s forward direction). We assume the object’s orientation is parallel to the ground plane, effectively projecting the problem onto the xz-plane and ignoring vertical displacement, as y-coordinates do not reliably encode real-world height due to perspective projection.

The orientation θ is sampled uniformly from $\{ l e f t ( 9 0 ^ { \circ } )$ , towards camera (180◦), right (270◦), away from camera $( 0 ^ { \circ } ) \}$ . To compute the relative position of $R _ { 2 }$ in the object’s coordinate frame, we apply a 2D rigid transformation (translation followed by rotation) to $( x _ { 2 } , z _ { 2 } )$ :

$$
\left[ \begin{array}{c} \tilde {x} _ {2} \\ \tilde {z} _ {2} \\ 1 \end{array} \right] = \underbrace {\left[ \begin{array}{c c c} \cos \theta & \sin \theta & 0 \\ - \sin \theta & \cos \theta & 0 \\ 0 & 0 & 1 \end{array} \right]} _ {\text {Rotation}} \underbrace {\left[ \begin{array}{c c c} 1 & 0 & - x _ {1} \\ 0 & 1 & - z _ {1} \\ 0 & 0 & 1 \end{array} \right]} _ {\text {Translation}} \left[ \begin{array}{c} x _ {2} \\ z _ {2} \\ 1 \end{array} \right]. \tag {7}
$$

Task objective: The relative position is determined by the signs of $( \tilde { x } _ { 2 } , \tilde { z } _ { 2 } )$ in the object’s coordinate frame. We define the directional labels as:

$$
\tilde {p} _ {x} = \left\{ \begin{array}{l l} \text { Right }, & \tilde {x} _ {2} > \delta_ {x} \\ \text { Left }, & \tilde {x} _ {2} <   - \delta_ {x}, \\ \text { None }, & \text { otherwise } \end{array} \right. \tag {8}
$$

$$
\tilde {p} _ {z} = \left\{ \begin{array}{l l} \text {   Front,   } & \tilde {z} _ {2} > \delta_ {z} \\ \text {   Back,   } & \tilde {z} _ {2} <   - \delta_ {z}, \\ \text {   None,   } & \text {   otherwise   } \end{array} \right. \tag {9}
$$

where $\delta _ { x }$ and $\delta _ { z }$ are thresholds that enforce unambiguous spatial separation. We discard instances where both $\tilde { p } _ { x } = \tilde { p } _ { z } = \mathrm { N o n e }$ , ensuring all valid answers describe a clear spatial relation along at least one axis. The groundtruth answer is the tuple $( \tilde { p } _ { x } , \tilde { p } _ { z } )$ . Additional details are provided in Appendix A.6.

# 3.1.3. Dataset Construction

Data sources. We collect raw RGB images from COCO [38] and RGB-D images from DIODE [56] and MegaDepth [35]. These datasets provide real-world imagery spanning diverse indoor and outdoor scenes, object categories, and viewpoints. Critically, we use only the raw images and depth maps (where available), discarding all human-provided annotations such as bounding boxes, segmentations, or captions. This ensures our pipeline remains fully self-supervised and reproducible without dependence on costly annotation infrastructure.

Spatial-SSRL-81k dataset. By applying the automated augmentation and task construction procedures described in Sec. 3.1, we generate a dataset of 81,053 questionanswer pairs, denoted Spatial-SSRL-81k. The dataset balances depth-free and depth-based tasks in roughly equal proportions and exhibits diverse question formats, including ordering tasks, multiple-choice questions with image options, and multiple-choice questions with text options. Importantly, because all supervision is derived deterministically from image structure, Spatial-SSRL-81k achieves 100% ground-truth accuracy, which is unattainable by prior pipelines that rely on noisy detections or model-generated annotations. Detailed statistics, task distributions, and example QA pairs are provided in Appendix A.

# 3.2. RL Training with Verifiable Rewards

We optimize LVLMs on the Spatial-SSRL-81k dataset using Group Relative Policy Optimization (GRPO) [53], a policy-gradient RL algorithm well-suited for verifiable reward signals. Our training procedure has two stages: an SFT cold-start phase followed by GRPO optimization.

Cold-start with SFT. Our five self-supervised tasks vary substantially in difficulty and output format (ordering sequences, multiple-choice with image or text options). In preliminary experiments, we observe that directly applying RL from a pretrained checkpoint leads to training instability and reward collapse, as the base model fails to generate valid formatted responses (success rate < 5% for complex tasks like relative position prediction). To mitigate this, we first perform a brief SFT warm-up on a small subset of approximately 3,600 samples (∼4.4% of the full dataset). This cold-start phase familiarizes the model with task formats and answer structures while preserving the benefits of RL-based optimization in the subsequent stage.

GRPO Optimization. Following cold-start, we apply GRPO to optimize the policy on all five tasks jointly. We append a format prompt to each question, instructing the model to generate its reasoning process within ⟨think⟩ . . . ⟨/think⟩ tags and provide the final answer in \boxed{}. Our structured output format encourages step-by-step reasoning and enables reliable answer extraction for reward computation.

Reward Design. The GRPO reward function has two binary components: an accuracy reward $r _ { \mathrm { a c c } }$ and a format reward $r _ { \mathrm { f m t } } .$ . The accuracy reward is set to $r _ { \mathrm { a c c } } = 1$ if and only if the model’s predicted answer exactly matches the groundtruth (determined via self-supervision), and $r _ { \mathrm { a c c } } = 0$ otherwise. The format reward is $r _ { \mathrm { f m t } } = 1$ if the output strictly adheres to the prescribed format (reasoning enclosed in think tags, answer in boxed command), and $r _ { \mathrm { f m t } } = 0$ otherwise. The overall reward is a weighted combination, $r = 0 . 9 { \cdot } r _ { \mathrm { a c c } } { + 0 . 1 { \cdot } r _ { \mathrm { f m t } } }$ . We assign higher weight to accuracy than format because format compliance typically stabilizes quickly after cold-start (> 95% compliance), whereas accuracy improvements require the full RL phase.

# 4. Experiments

# 4.1. Experimental Settings

Base models and training. We train Spatial-SSRL-3B and Spatial-SSRL-7B from Qwen2.5-VL-3B and Qwen2.5- VL-7B [1], and train Spatial-SSRL-4B from Qwen3-VL-4B on our Spatial-SSRL-81k. In the cold-start stage, we train for 5 epochs on the SFT data with a learning rate of $1 \times 1 0 ^ { - 5 }$ . During GRPO, we apply KL regularization with weight 0.01. For each training sample, we generate a rollout group of 5 responses with temperature 1.0. The training uses a global batch size of 128 and a learning rate of $1 \times 1 0 ^ { - 6 }$ for 360 steps.

Evaluation Protocol. We evaluate all models using VLMEvalKit [15], an open-source toolkit that provides standardized evaluation protocols and metrics for visionlanguage models. For benchmarks not natively supported by VLMEvalKit, we adapt the official evaluation code to accommodate the Qwen2.5-VL architecture while strictly preserving the original metrics and evaluation procedures. When evaluating models with reasoning capabilities (Spatial-SSRL-3B and Spatial-SSRL-7B), we append the same format prompt used during GRPO training (Sec. 3.2) to elicit structured reasoning, ensuring consistency between training and inference. For fair comparison, baseline models without reasoning are evaluated using their standard prompts without modification. Additional evaluation details, including prompt templates and per-benchmark configurations, are provided in Appendix B.

# 4.2. Main Results

# 4.2.1. Spatial Understanding

Benchmark overview. We evaluate spatial understanding across seven diverse benchmarks spanning image and video modalities. Image benchmarks include Spatial457 [60], 3DSRBench [44], SpatialEval [58], QSpatial-plus [36], What’sUp [28], and ViewSpatial [32], while VSI-Bench [68] focuses on understanding egocentric videos. The tasks in these benchmarks cover a variety of types (distance estimation, object occlusion, collision prediction, object orientation, multi-view understanding, etc.) and require strong 2D and 3D visual-spatial intelligence of LVLMs.

Overall performance improvements. As shown in Tab. 1 and Tab. 2, Spatial-SSRL improves spatial understanding on both Qwen2.5-VL and Qwen3-VL architectures. Importantly, Spatial-SSRL achieves consistent improvements on Qwen2.5-VL across all seven benchmarks for both 3B and 7B model scales. The 3B model yields an average gain of +4.63%, with particularly strong improvements on Spatial457 (+12.37%), and VSI-Bench (+5.65%). The 7B model also achieves an average gain of +3.89%. The substantial gains on Spatial457, which requires complex pose estimation and multi-step spatial reasoning, demonstrate that Spatial-SSRL enhances not only spatial perception but also higher-level spatial reasoning capabilities.

Reasoning capability enhancement. A critical observation from Tab. 1 is that the baseline models exhibit performance degradation when prompted to generate explicit reasoning chains. Specifically, Qwen2.5-VL-3B drops from 45.91% to 44.85% average accuracy with reasoning enabled, while Qwen2.5-VL-7B shows mixed results with notable degradation on What’sUp (86.95% → 70.61%) and VSI-Bench (38.08% → 32.69%). This phenomenon, consistent with prior work [64, 74], indicates that base models lack effective spatial reasoning and their generated reasoning steps introduce noise rather than providing useful inference. In contrast, Spatial-SSRL models with reasoning consistently outperform baselines without reasoning across all benchmarks, demonstrating that our self-supervised RL paradigm successfully teaches the model to generate productive spatial reasoning rather than spurious correlations.

Table 1. Performance of Qwen2.5-VL-based models on spatial understanding benchmarks. The open-source models and our model are evaluated on seven benchmarks, and the average results are provided in the last column. For a comprehensive comparison, we test two settings of the baseline model: one that does not generate the reasoning process and one that does. We compute the improvements based on the results of our models and the baseline models without reasoning. The qualitative analysis of some cases is shown in Appendix C. 

<table><tr><td rowspan="2">Models</td><td rowspan="2">Reasoning</td><td colspan="6">Image</td><td>Video</td><td rowspan="2">Avg.</td></tr><tr><td>Spatial457</td><td>3DSRBench</td><td>SpatialEval</td><td> $QSpatial_{plus}$ </td><td>What&#x27;sUp</td><td>ViewSpatial</td><td>VSI-Bench</td></tr><tr><td colspan="10">Representative Spatial Models</td></tr><tr><td>SpatialLadder-3B [33]</td><td>√</td><td>40.19</td><td>48.91</td><td>41.57</td><td>37.62</td><td>66.59</td><td>39.06</td><td>41.67</td><td>45.09</td></tr><tr><td>SpaceR-7B [48]</td><td>√</td><td>53.66</td><td>53.91</td><td>61.57</td><td>48.51</td><td>86.22</td><td>37.34</td><td>40.54</td><td>54.54</td></tr><tr><td colspan="10">Baselines &amp; Our Model (3B)</td></tr><tr><td>Qwen2.5-VL-3B</td><td>×</td><td>33.70</td><td>50.30</td><td>54.65</td><td>33.66</td><td>85.85</td><td>35.38</td><td>27.84</td><td>45.91</td></tr><tr><td>Qwen2.5-VL-3B</td><td>√</td><td>34.82</td><td>49.47</td><td>50.55</td><td>30.66</td><td>83.54</td><td>34.23</td><td>30.67</td><td>44.85</td></tr><tr><td>Spatial-SSRL-3B</td><td>√</td><td>46.07</td><td>51.72</td><td>59.59</td><td>39.60</td><td>86.71</td><td>36.62</td><td>33.49</td><td>50.54</td></tr><tr><td>Improvement</td><td>N/A</td><td>+12.37</td><td>+1.42</td><td>+4.94</td><td>+5.95</td><td>+0.86</td><td>+1.24</td><td>+5.65</td><td>+4.63</td></tr><tr><td colspan="10">Baselines &amp; Our Model (7B)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>×</td><td>44.67</td><td>53.39</td><td>62.37</td><td>46.53</td><td>86.95</td><td>36.83</td><td>38.08</td><td>52.69</td></tr><tr><td>Qwen2.5-VL-7B</td><td>√</td><td>44.88</td><td>54.55</td><td>61.12</td><td>46.53</td><td>70.61</td><td>36.66</td><td>32.69</td><td>49.58</td></tr><tr><td>Spatial-SSRL-7B</td><td>√</td><td>53.34</td><td>56.53</td><td>64.03</td><td>54.46</td><td>90.61</td><td>37.81</td><td>39.29</td><td>56.58</td></tr><tr><td>Improvement</td><td>N/A</td><td>+8.67</td><td>+3.14</td><td>+1.66</td><td>+7.93</td><td>+3.66</td><td>+0.98</td><td>+1.21</td><td>+3.89</td></tr></table>

Table 2. Performance of Qwen-3-VL-based models on spatial understanding and general VQA benchmarks. We evaluate the models on the seven benchmarks in Tab. 1 and the four General VQA benchmarks in Tab. 3, and show the average results. The detailed results are shown in Appendix B.3. 

<table><tr><td>Models</td><td>CoT</td><td>Spatial Und.</td><td>CoT</td><td>Gnr.-VQA</td></tr><tr><td>Qwen3-VL-4B</td><td>✗</td><td>60.14</td><td>✗</td><td>69.10</td></tr><tr><td>Qwen3-VL-4B</td><td>√</td><td>60.23</td><td>N/A</td><td>N/A</td></tr><tr><td>Spatial-SSRL-4B</td><td>√</td><td>61.43 (+1.29)</td><td>✗</td><td>70.28 (+1.18)</td></tr></table>

Analysis of improvement patterns. The improvement distribution across benchmarks reveals meaningful patterns. The largest gains appear on benchmarks requiring 3D spatial reasoning (Spatial457, 3DSRBench), validating that both depth-free and depth-based pretext tasks contribute to 3D understanding. Moderate gains on 2D-centric benchmarks (SpatialEval, What’sUp) suggest that depthfree tasks like patch reordering and inpainting effectively enhance 2D spatial layout comprehension. The relatively smaller but consistent improvements on ViewSpatial indicate that orientation and multi-perspective understanding benefit from our pretext tasks as well.

Cross-modal generalization. Despite training exclusively on static RGB and RGB-D images, Spatial-SSRL improves video spatial understanding on VSI-Bench by +5.65% (3B) and +1.21% (7B). The ability to transfer from image-based self-supervision to video understanding suggests that the pretext tasks foster internal spatial representations that are modality-agnostic and grounded in geometric principles.

# 4.2.2. General Visual Capabilities

Evaluation setup. To verify that spatial-focused training does not degrade general visual understanding, we evaluate Spatial-SSRL on two complementary benchmark suites. First, we assess general VQA capabilities on MMBench-v1. $\mathbf { - E N _ { \mathrm { t e s t } } }$ [39] (general visual understanding), BLINK [17] (multi-image understanding), Hallusion-Bench [21] (hallucination), and RealWorldQA [66] (realworld scene understanding). These benchmarks test fundamental visual perception without requiring extensive spatial analysis, so we evaluate both baseline and Spatial-SSRL models using standard prompts without reasoning instructions. Second, we evaluate fine-grained visual perception on OCRBench [40], $\mathrm { C h a r t Q A } _ { \mathrm { t e s t } }$ [45], and SeedBench2- plus [31], which require accurate recognition of dense text, charts, and detailed visual information. For these benchmarks, we apply reasoning prompts to Spatial-SSRL to maintain consistency with spatial evaluation.

General VQA results. As shown in Tab. 2 and Tab. 3, Spatial-SSRL not only preserves but actually improves general visual capabilities despite being trained exclusively on self-supervised spatial tasks. The 3B model achieves an average gain of +2.02% across the four general VQA benchmarks. The 7B model shows a smaller but consistent gain of +0.57% on average, with improvements on MMBench1.1 (+0.37%), HallusionBench (+0.92%), and RealWorldQA (+0.65%). The consistent gains across diverse visual reasoning tasks suggest that self-supervised spatial training provides a beneficial effect for general visual understanding. We attribute this to two factors: (1) the pretext tasks require holistic image understanding to reason about patch relationships and structural coherence, which transfers to general scene comprehension; and (2) the reasoning format training encourages systematic visual analysis that benefits non-spatial reasoning as well.

Table 3. General visual capability comparisons. The benchmarks are organized into two categories: General VQA and OCR and Chart Understanding. The first category covers a wide range of fundamental visual capabilities, such as knowledge application, and hallucination recognition, multi-image understanding, etc. The second category targets the understanding of images with charts or rich textual details. The average accuracy of both categories are computed and provided. 

<table><tr><td rowspan="2">Models</td><td colspan="5">General VQA</td><td colspan="4">OCR and Chart Understanding</td></tr><tr><td>MMBench1.1</td><td>BLINK</td><td>Hallusion</td><td>RealWorldQA</td><td>Avg.</td><td>OCRBench</td><td>ChartQA</td><td>SeedBench2plus</td><td>Avg.</td></tr><tr><td colspan="10">3B Models</td></tr><tr><td>Qwen2.5-VL-3B</td><td>77.25</td><td>48.97</td><td>46.03</td><td>65.23</td><td>59.37</td><td>82.6</td><td>84.08</td><td>68.69</td><td>78.46</td></tr><tr><td>Spatial-SSRL-3B</td><td>78.29</td><td>50.92</td><td>49.66</td><td>66.67</td><td>61.39 (+2.02)</td><td>84.1</td><td>83.08</td><td>68.56</td><td>78.58 (+0.12)</td></tr><tr><td colspan="10">7B Models</td></tr><tr><td>Qwen2.5-VL-7B</td><td>82.23</td><td>55.87</td><td>52.26</td><td>68.63</td><td>64.75</td><td>88.5</td><td>86.44</td><td>70.93</td><td>81.96</td></tr><tr><td>Spatial-SSRL-7B</td><td>82.60</td><td>56.23</td><td>53.18</td><td>69.28</td><td>65.32 (+0.57)</td><td>89.6</td><td>88.08</td><td>71.85</td><td>83.18 (+1.22)</td></tr></table>

Fine-grained perception results. On OCR and chart understanding benchmarks, Spatial-SSRL maintains strong performance with modest improvements: +0.12% average for 3B models and +1.22% average for 7B models. Notably, the 7B model boosts performance on OCRBench (+1.1%), ChartQA (+1.64%), and SeedBench2-plus (+0.92%). The minor degradation on ChartQA for the 3B model (-1.0%) is within typical evaluation variance and does not indicate systematic capability loss. We attribute the overall preservation to the fact that several pretext tasks, particularly cropped patch inpainting and flipped patch recognition, require attention to fine-grained visual details, texture continuity, and local structural consistency, which align with the demands of OCR and chart understanding.

# 4.3. Ablation Studies

Overall observations. As shown in Tab. 4, all training configurations improve upon the baseline across most evaluation dimensions, validating that each pretext task contributes positively to spatial understanding. The model trained on all five tasks achieves the best performance on four out of seven benchmarks and competitive performance on others, demonstrating that depth-free and depth-based supervision provide complementary learning signals that synergize when combined. Comparing single-task models to the combined model reveals that no individual task dominates across all dimensions, underscoring the importance of diverse self-supervised objectives for spatial understanding. Individual task contributions. Among depth-free tasks, Shuffled Patch Reordering excels at 2D spatial layout and reasoning, likely because recovering patch order requires global structural understanding and multi-step inference. Flipped Patch Recognition achieves the strongest maze reasoning performance, suggesting that detecting subtle orientation violations is beneficial to logical reasoning. Cropped Patch Inpainting shows balanced improvements across general VQA and 3D height estimation, indicating that texturecontext matching transfers broadly. Among depth-based tasks, Regional Depth Ordering achieves the strongest 3D height understanding and strong location reasoning, validating its direct supervision of ordinal depth perception. Relative Position Prediction achieves the best general VQA and strong multi-object reasoning, likely because egocentric coordinate transformation requires mental rotation and perspective-taking that transfer to general reasoning. Combining both depth-based tasks achieves the best multi-object performance, demonstrating complementarity between ordinal depth and egocentric reasoning.

Depth-free vs. depth-based supervision. Depth-free tasks drive stronger improvements on general VQA and 2D spatial layout, validating that RGB-only supervision effectively enhances 2D understanding. Notably, depth-free tasks also improve 3D reasoning (e.g., Flip achieves 48.63% on 3DSR-MultiObj., competitive with depth-based tasks), suggesting that 2D structural reasoning provides useful inductive biases for 3D inference. Depth-based tasks, as expected, excel at 3D-centric benchmarks: averaging 61.45% on the three 3DSR subsets vs. 57.99% for depth-free tasks. The 3.46% gap demonstrates that explicit depth supervision is crucial for robust 3D understanding, though 2D tasks provide non-trivial 3D benefits.

Task complementarity and synergy. Combining all five tasks yields improvements beyond any single-category model on key metrics. This synergy indicates that 2D and 3D supervision address different failure modes and provide mutual regularization. According to our observations, diverse task coverage is more important than individual task optimization, as no single task suffices. Practitioners aiming to maximize specific capabilities (e.g., 3D height for robotics) can prioritize corresponding tasks (Regional Depth), while applications requiring holistic spatial intelligence should use diverse task combinations.

Table 4. Task ablation on benchmark subsets. Each row represents a training configuration and its performance across seven evaluation dimensions. All models are trained based on Qwen2.5-VL-7B. The five columns under training tasks illustrate the tasks used for each setting. Gnr-VQA averages the four general VQA benchmarks from Tab. 3. Spatial subtasks are tested on the subsets of the spatial benchmarks in Tab. 1. Bold indicates best performance; double-underline and underline indicate second and third best. 

<table><tr><td colspan="5">Trained Tasks</td><td colspan="7">Benchmark Subsets</td></tr><tr><td colspan="3">Depth-free</td><td colspan="2">Depth-based</td><td>General Tasks</td><td colspan="2">2D-Spatial Tasks</td><td colspan="4">3D-Spatial Tasks</td></tr><tr><td>Crop</td><td>Shuf.</td><td>Flip</td><td>Depth</td><td>Pos.</td><td>Gnr-VQA</td><td>Spa457-2D</td><td>SpaEval-Maze</td><td>Spa457-Pose</td><td>3DSR-Height</td><td>3DSR-Locat.</td><td>3DSR-MultiObj.</td></tr><tr><td colspan="12">Baseline Model</td></tr><tr><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>64.75</td><td>56.52</td><td>35.13</td><td>33.92</td><td>52.61</td><td>67.84</td><td>46.70</td></tr><tr><td colspan="12">Models on Depth-free Tasks</td></tr><tr><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>65.39</td><td>59.64</td><td>37.67</td><td>40.40</td><td>58.99</td><td>69.22</td><td>47.57</td></tr><tr><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>65.30</td><td>61.30</td><td>42.80</td><td>40.61</td><td>58.48</td><td>69.51</td><td>48.10</td></tr><tr><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>65.38</td><td>60.86</td><td>43.60</td><td>39.69</td><td>56.59</td><td>71.24</td><td>48.63</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>65.38</td><td>61.68</td><td>40.80</td><td>41.36</td><td>56.67</td><td>72.06</td><td>47.79</td></tr><tr><td colspan="12">Models on Depth-based Tasks</td></tr><tr><td>X</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>64.99</td><td>61.60</td><td>28.73</td><td>40.22</td><td>63.48</td><td>72.94</td><td>46.99</td></tr><tr><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>65.64</td><td>60.86</td><td>39.07</td><td>40.20</td><td>58.91</td><td>72.20</td><td>48.85</td></tr><tr><td>X</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>65.29</td><td>62.06</td><td>36.07</td><td>41.67</td><td>62.97</td><td>71.94</td><td>49.28</td></tr><tr><td colspan="12">Models on All Tasks</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>65.32</td><td>62.94</td><td>42.53</td><td>42.74</td><td>58.91</td><td>74.11</td><td>48.85</td></tr></table>

# 5. Conclusion

We introduce Spatial-SSRL, a self-supervised reinforcement learning paradigm that derives verifiable supervision directly from intrinsic image structure. Our key insight is that ordinary RGB and RGB-D images contain rich self-supervised signals for spatial understanding that are naturally compatible with reinforcement learning through verifiable rewards. Through comprehensive experiments on seven spatial understanding benchmarks, we demonstrate that Spatial-SSRL delivers substantial improvements: +4.63% average accuracy for 3B models and +3.89% for 7B models, with particularly strong gains on complex spatial reasoning benchmarks. Critically, Spatial-SSRL not only enhances spatial capabilities but also improves general visual understanding while preserving fine-grained perception. In future work, we plan to extend our framework to video-native pretext tasks (e.g., optical flow prediction and temporal coherence) to strengthen video spatial reasoning beyond the current cross-modal transfer.

# References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun

Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 7

[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3   
[3] Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 9490–9498. IEEE, 2025. 2, 3   
[4] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024. 2, 3   
[5] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025. 2   
[6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? In NeurIPs, 2024. 2   
[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020. 2, 3   
[8] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu

Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453, 2024. 2   
[9] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatial-RGPT: Grounded spatial reasoning in vision-language models. In NeurIPs, 2024. 2, 3   
[10] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025. 2   
[11] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 2, 3   
[12] Nianchen Deng, Lixin Gu, Shenglong Ye, Yinan He, Zhe Chen, Songze Li, Haomin Wang, Xingguang Wei, Tianshuo Yang, Min Dou, et al. Internspatial: A comprehensive dataset for spatial reasoning in vision-language models. arXiv preprint arXiv:2506.18385, 2025. 3   
[13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171– 4186, 2019. 3   
[14] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. In arXiv preprint arXiv:2303.03378, 2023. 2   
[15] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 7   
[16] Yongchao Feng, Yajie Liu, Shuai Yang, Wenrui Cai, Jinqing Zhang, Qiqi Zhan, Ziyue Huang, Hongxi Yan, Qiao Wan, Chenguang Liu, et al. Vision-language model for object detection and segmentation: A review and evaluation. arXiv preprint arXiv:2504.09480, 2025. 2   
[17] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024. 3, 8   
[18] Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and Yi Wu. On designing effective rl reward at training time for llm reasoning. arXiv preprint arXiv:2410.15115, 2024. 2   
[19] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018. 3

[20] Dylan Goetting, Himanshu Gaurav Singh, and Antonio Loquercio. End-to-end navigation with vision language models: Transforming spatial reasoning into question-answering. arXiv preprint arXiv:2411.05755, 2024. 2   
[21] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024. 3, 8   
[22] Jie Gui, Tuo Chen, Jing Zhang, Qiong Cao, Zhenan Sun, Hao Luo, and Dacheng Tao. A survey on self-supervised learning: Algorithms, applications, and future trends. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9052–9071, 2024. 3   
[23] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3   
[24] Xiaojun Guo, Runyu Zhou, Yifei Wang, Qi Zhang, Chenheng Zhang, Stefanie Jegelka, Xiaohan Wang, Jiajun Chai, Guojun Yin, Wei Lin, et al. Ssl4rl: Revisiting self-supervised learning as intrinsic reward for visual-language reasoning. arXiv preprint arXiv:2510.16416, 2025. 3   
[25] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 2, 3   
[26] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable ´ vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 3   
[27] Yining Hong, Chunru Lin, Yilun Du, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. 3d concept learning and reasoning from multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9202–9212, 2023. 3   
[28] Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s” up” with vision-language models? investigating their struggle with spatial reasoning. arXiv preprint arXiv:2310.19785, 2023. 3, 7   
[29] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2   
[30] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model posttraining. arXiv preprint arXiv:2411.15124, 2024. 2   
[31] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024. 8

[32] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in visionlanguage models. arXiv preprint arXiv:2505.21500, 2025. 3, 7   
[33] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531, 2025. 2, 3, 8   
[34] Yun Li, Yiming Zhang, Tao Lin, XiangRui Liu, Wenxiao Cai, Zheng Liu, and Bo Zhao. Sti-bench: Are mllms ready for precise spatial-temporal world understanding? arXiv preprint arXiv:2503.23765, 2025. 2   
[35] Zhengqi Li and Noah Snavely. Megadepth: Learning singleview depth prediction from internet photos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2041–2050, 2018. 6   
[36] Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models. arXiv preprint arXiv:2409.09788, 2024. 3, 7   
[37] Zhenyi Liao, Qingsong Xie, Yanhao Zhang, Zijian Kong, Haonan Lu, Zhenyu Yang, and Zhijie Deng. Improved visual-spatial reasoning via r1-zero-like training. arXiv preprint arXiv:2504.00883, 2025. 2, 3   
[38] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 6   
[39] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is your multi-modal model an all-around player? In ECCV, 2024. 2, 3, 8   
[40] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), 2024. 8   
[41] Zihan Liu, Zhikang Niu, Qiuyang Xiao, Zhisheng Zheng, Ruoqi Yuan, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Jianze Liang, Xie Chen, Leilei Sun, Dahua Lin, and Jiaqi Wang. Star-bench: Probing deep spatio-temporal reasoning as audio 4d intelligence. arXiv preprint arXiv:2510.24693, 2025. 2   
[42] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-RFT: Visual reinforcement fine-tuning. In ICCV, 2025. 3   
[43] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 2   
[44] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In

Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025. 3, 7   
[45] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 8   
[46] Ishan Misra, C Lawrence Zitnick, and Martial Hebert. Shuffle and learn: unsupervised learning using temporal order verification. In European conference on computer vision, pages 527–544. Springer, 2016. 3   
[47] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European conference on computer vision, pages 69–84. Springer, 2016. 3   
[48] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025. 3, 8   
[49] Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428, 2025. 2   
[50] Juneyoung Ro, Namwoo Kim, and Yoonjin Yoon. How well do vision-language models understand cities? a comparative study on spatial reasoning from street-view images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6476–6485, 2025. 3   
[51] Rodrigo Santa Cruz, Basura Fernando, Anoop Cherian, and Stephen Gould. Deeppermnet: Visual permutation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3949–3957, 2017. 2   
[52] Hamed Sarbolandi, Damien Lefloch, and Andreas Kolb. Kinect range sensing: Structured-light versus time-of-flight kinect. Computer vision and image understanding, 139:1– 20, 2015. 2   
[53] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 3, 6   
[54] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15768–15780, 2025. 2   
[55] Kexin Tian, Jingrui Mao, Yunlong Zhang, Jiwan Jiang, Yang Zhou, and Zhengzhong Tu. Nuscenes-spatialqa: A spatial understanding and reasoning benchmark for visionlanguage models in autonomous driving. arXiv preprint arXiv:2504.03164, 2025. 2   
[56] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019. 2, 6

[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 2   
[58] Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, Sharon Li, and Neel Joshi. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. Advances in Neural Information Processing Systems, 37:75392–75421, 2024. 3, 7   
[59] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic vision-language dataset for autonomous driving with counterfactual reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22442–22452, 2025. 2   
[60] Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24669–24679, 2025. 3, 7   
[61] Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025. 3   
[62] Zifu Wang, Junyi Zhu, Bo Tang, Zhiyu Li, Feiyu Xiong, Jiaqian Yu, and Matthew B Blaschko. Jigsaw-r1: A study of rule-based visual reinforcement learning with jigsaw puzzles. arXiv preprint arXiv:2505.23590, 2025. 3   
[63] Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025. 3   
[64] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965, 2025. 2, 7   
[65] Penghao Wu, Yushan Zhang, Haiwen Diao, Bo Li, Lewei Lu, and Ziwei Liu. Visual jigsaw post-training improves mllms. arXiv preprint arXiv:2509.25190, 2025. 3   
[66] X.AI. Grok-1.5 vision preview, 2024. 8   
[67] Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, and Dahua Lin. Caprl: Stimulating dense image caption capabilities via reinforcement learning. arXiv preprint arXiv:2509.22647, 2025. 2   
[68] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 3, 7, 4   
[69] Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Yue Cao, Xinlong Wang, and Jingjing Liu. Capsfusion: Rethinking image-text data at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14022–14032, 2024. 2

[70] Jirong Zha, Yuxuan Fan, Xiao Yang, Chen Gao, and Xinlei Chen. How to enable llm with 3d capacity? a survey of spatial reasoning in llm. arXiv preprint arXiv:2504.05786, 2025. 2   
[71] Beichen Zhang, Yuhong Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Haodong Duan, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Booststep: Boosting mathematical capability of large language models via improved single-step reasoning. arXiv preprint arXiv:2501.03226, 2025. 2   
[72] Haoyu Zhang, Meng Liu, Zaijing Li, Haokun Wen, Weili Guan, Yaowei Wang, and Liqiang Nie. Spatial understanding from videos: Structured prompts meet simulation data. arXiv preprint arXiv:2506.03642, 2025. 3   
[73] Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. arXiv preprint arXiv:2504.05812, 2025. 3   
[74] Wanyue Zhang, Yibin Huang, Yangbin Xu, JingJing Huang, Helu Zhi, Shuo Ren, Wang Xu, and Jiajun Zhang. Why do mllms struggle with spatial understanding? a systematic analysis from data to architecture. arXiv preprint arXiv:2509.02359, 2025. 3, 7   
[75] Zhixiong Zhang, Shuangrui Ding, Xiaoyi Dong, Songxin He, Jianfan Lin, Junsong Tang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Sec: Advancing complex video object segmentation via progressive concept construction. arXiv preprint arXiv:2507.15852, 2025. 2   
[76] Zizhuo Zhang, Jianing Zhu, Xinmu Ge, Zihua Zhao, Zhanke Zhou, Xuan Li, Xiao Feng, Jiangchao Yao, and Bo Han. Co-reward: Self-supervised reinforcement learning for large language model reasoning via contrastive agreement. arXiv preprint arXiv:2508.00410, 2025. 2

# A. Method Details

# A.1. Statistics of Spatial-SSRL-81k

Spatial-SSRL-81k consists of 81,053 samples in total. All samples belong to the self-supervised tasks in Sec. 3.1. The examples of all tasks are shown in Fig. 4, Fig. 5, Fig. 6, Fig. 7, and Fig. 6. Considering that Shuffled Patch Reordering and Flipped Patch Recognition share similar image layouts and structures, we regard them as one large task and evenly mix them with three other tasks, each containing approximately 20k samples. It also benefits the even distribution of the two task categories: Depth-free and Depth-Based. The number of samples for each task is demonstrated in Tab. 5. The formulation details of each task are provided in Sec. A.2, Sec. A.3, Sec. A.4, Sec. A.5, Sec. A.6.

Table 5. The size of all five tasks in the RL training dataset. 

<table><tr><td>Category</td><td>Task</td><td>Size</td></tr><tr><td rowspan="3">Depth-free</td><td>Shuffled Patch Reordering</td><td>16,028</td></tr><tr><td>Flipped Patch Recognition</td><td>4,005</td></tr><tr><td>Cropped Patch Inpainting</td><td>20,200</td></tr><tr><td rowspan="2">Depth-based</td><td>Regional Depth Ordering</td><td>20,620</td></tr><tr><td>Relative Position Prediction</td><td>20,200</td></tr></table>

# A.2. Shuffled Patch Reordering

![](images/149cae7ca1f492ac8208cc1134e82e89d261088bff59e5d2b1b602789a96eb1e.jpg)

<details>
<summary>natural_image</summary>

Four-panel street scene showing a police officer on a motorcycle crossing a road, with other vehicles and pedestrians in the background (no visible text or symbols)
</details>

Q: The image has been divided into 4 shuffled patches labeled 0, 1, 2, and 3. Based on visual clues such as continuity, alignment, and context, answer the correct arrangement of the patches to restore the original image, where the format is 'TopLeft-TopRight-BottomLeft-BottomRight'.

A: 0-2-3-1

![](images/8499d643666a252dec66d2cee2321f70f7b5bb71ec53f4f153a2e61eb1ec655a.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo sequence showing a person lying on a bed with patterned bedding, no visible text or symbols
</details>

Q: The image has been divided into 4 shuffled horizontal strips labeled 0, 1, 2, and 3. Based on visual clues such as continuity, alignment, and context, answer the correct arrangement of the strips to restore the original image, where the format is 'Left-Middle1-Middle2-Right'.

A: 1-2-0-3

Figure 4. Examples of the task Shuffled Patch Reordering.

We split the images into M × N patches, where M is the number of patches in the vertical direction and N is the number of patches in the horizontal direction. The examples are provided in Fig. 4. For instance, the example on the bottom is patchified with M = 1, N = 4.

Besides, we may also apply a mask to one patch on some task samples. The details such as M, N , mask, etc., and the corresponding number of each type of samples are listed in Tab. 6.

Table 6. The details of Shuffled Patch Reordering samples. 

<table><tr><td>Patchify Strategy</td><td>Mask</td><td>Size</td></tr><tr><td>M=2, N=2</td><td>✕</td><td>4,000</td></tr><tr><td>M=2, N=2</td><td>√</td><td>4,028</td></tr><tr><td>Horizontal (M=1, N=3 or 4)</td><td>✕</td><td>4,991</td></tr><tr><td>Vertical (M=3 or 4, N=1)</td><td>✕</td><td>3,009</td></tr></table>

# A.3. Flipped Patch Recognition

![](images/b41d4ebe0764cb817c2016074f19955e52b587c3780cb5ff566121bc7d1299b5.jpg)

<details>
<summary>natural_image</summary>

Four-panel image showing a dog in a leafy outdoor setting, with close-ups of animals in natural grass (no text or symbols visible)
</details>

Q: The image has been divided into 4 labeled 0, 1, 2, and 3. One of the four patches is flipped either horizontally or vertically. Based on visual clues such as continuity, alignment, and context, answer the correct patch that is flipped and the direction the flip, where the format is 'Label-Direction'. The direction can only be 0(flipped vertically) or 1(flipped horizontally).

A: 3-0

![](images/ce118e2d86b8c95c5bf1972d7c50f888793b7d2eb870383b41b2224782d0f7ff.jpg)

<details>
<summary>natural_image</summary>

Side view of a red and white train on railway tracks with overhead power lines (no visible text or symbols)
</details>

Q: The image has been divided into 4 labeled 0, 1, 2, and 3. One of the four patches is flipped either horizontally or vertically. Based on visual clues such as continuity, alignment, and context, answer the correct patch that is flipped and the direction the flip, where the format is 'Label-Direction'. The direction can only be 0(flipped vertically) or 1(flipped horizontally).

A: 2-1

Figure 5. Examples of the task Flipped Patch Recognition.

The examples of this task are provided in Fig. 5.

# A.4. Cropped Patch Inpainting

![](images/3497bc7e2f5008ea2ccfa2010b7968c4b78edd793ceec7df970b7a65bc7c92de.jpg)

Q: Which image is the missing part in the first image <image1>? Based on visual clues such as alignment, image content, and positional relationship, select one of the four options <image2><image3><image4><image5> as the final answer. The final answer should be chosen from 'A', 'B', 'C', and 'D'.

A: A (<image2>)

<image1>   
![](images/34163a0f4b4b08984ba42a2eec0f017e2e083b9617a573aa369947af7d5ddd76.jpg)

A. <image2>   
![](images/4b2d29e0dbd254d6c95e9e0be969fca7139beed9f5b9ae060acd8c76ad146f1b.jpg)  
C. <image4>

![](images/eed994726e1e326281469e44a85c317d79283363a48bf60520ed17e046e9ad36.jpg)

B. <image3>   
![](images/192462d747721e5ba9f40706192cccef2579e0eb2642def62773cfccce7e367d.jpg)  
D. <image5>

![](images/db5ba888d3ca92c2a1416617442991688bcfcfe252d31e1ea888385f4cd00003.jpg)

Q: Which image is the missing part in the first image <image1>? Based on visual clues such as alignment, image content, and positional relationship, select one of the four options <image2><image3><image4><image5> as the final answer. The final answer should be chosen from 'A', 'B', 'C', and 'D'.

A: D (<image5>)

Figure 6. Examples of the task Cropped Patch Inpainting.

The examples are provided in Fig. 6. Among the four options in a problem, the correct option is the patch directly cropped from the blackened area of <image1>, retaining its original size. The other three options also come from <image1> to make the problem more challenging, and they are reshaped to the same size as the ground-truth patch.

As illustrated in Sec. 3.1.1, the three distractors are constructed in the following three forms: internal regions of the ground-truth, external regions of the ground-truth (θ = 0.25 or 0.5), and rotations of the ground-truth (90◦ clockwise or counterclockwise). We ensure that all three distractors are distinct from each other and the possibility of employing each method in formulating one problem is shown in Tab. 7.

Table 7. The probability of adopting each method for constructing a distractor of one QA sample in Cropped Patch Inpainting task. 

<table><tr><td>Method</td><td>Parameter Value</td><td>Probability</td></tr><tr><td>Internal Region</td><td>N/A</td><td>0.2</td></tr><tr><td rowspan="2">External Region</td><td> $\theta = 0.25$ </td><td>0.2</td></tr><tr><td> $\theta = 0.5$ </td><td>0.2</td></tr><tr><td rowspan="2">Rotation</td><td>90° Clockwise</td><td>0.2</td></tr><tr><td>90° Counterclockwise</td><td>0.2</td></tr></table>

# A.5. Regional Depth Ordering

![](images/366c5eba018e22a482cc8ee7e8b508c30ea1d85d26061f861d402052a5843799.jpg)

<details>
<summary>natural_image</summary>

Snow-covered park scene with trees and a large tree, no visible text or symbols
</details>

Q: The original image has three regions marked as 1, 2, and 3. Consider the content, positional relationships, depths of the three regions and other cues, and sort the depths of the three regions from smallest to largest from the camera, where the format of the answer is 'Smallest-Middle-Largest'.

A: 3-1-2   
![](images/d3ea196f7561867d019250a181808a68f8171f847b563a0c40be689acde21a00.jpg)

<details>
<summary>natural_image</summary>

Outdoor scene with snow-covered ground, tall grasses, and a paved path (no visible text or symbols)
</details>

Q: The original image has three regions marked as 1, 2, and 3. Consider the content, positional relationships, depths of the three regions and other cues, and sort the depths of the three regions from smallest to largest from the camera, where the format of the answer is 'Smallest-Middle-Largest'.

A: 2-1-3

Figure 7. Examples of the task Regional Depth Ordering.

The examples of this task are provided in Fig. 7.

# A.6. Relative Position Prediction

Figure 8 depicts the detailed construction procedure of this task, offering the definition of the variables used in Sec. 3.1.1 and explaining how the ground-truth answer can be derived through our automated pipeline. The examples are provided in Fig. 9.

During formulation, we define the parameters θ to represent the orientation of the object and $\delta _ { x } , \delta _ { z }$ as thresholds to avoid ambiguity of the ground-truth answers. $\delta _ { x }$ is in the direction of $\vec { x } _ { o b j }$ and $\delta _ { z }$ is in the direction of $\vec { z } _ { o b j }$ . We use pixels as the unit for measuring in the direction parallel to the image plane $( \mathbf { e . g . } , \vec { x } _ { c a m } )$ , and normalized depth

![](images/a67230e5b7b4eb276f0da80707d62e574d219469a4a80f3f322bd48d806fb71a.jpg)

<details>
<summary>text_image</summary>

Camera
x1
x2
x1
y1
y2
z1
z2
zObject
Object perspective
(x1,y1,z1)cam,(x2,y2,z2)cam
(x̃2,ȳ2, Ž2)obj
Facing left: θ = 90°
Q: Assume that there is an object at position ①, facing to the left.
From the perspective of the object, at what direction is region ② located?
A: Right-front.
</details>

Figure 8. The construction procedure of the task Relative position prediction. We define two coordinate systems based on the camera and the hypothesized object respectively. The z-axis represents the orientation. The x-axis represents the right side. The y-axis is always vertically downward. $( x _ { i } , y _ { i } , z _ { i } )$ is the coordinate of position i in the camera system while $( \tilde { x } _ { i } , \tilde { y } _ { i } , \tilde { z } _ { i } )$ is defined in the object system.   
![](images/9a63773b9d25b5e3fe6347fba3643b2309ecc54ad0497268220a7e928344c9b0.jpg)

<details>
<summary>natural_image</summary>

Exterior view of the Palace of Westminster with Big Ben clock tower and Gothic-style church spires under a partly cloudy sky (no signage)
</details>

Q: I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '1' and it's facing to me. According to the camera, where is the region marked '2'? A. Front B. Right C. Left D. Back. Consider cues such as depth, orientation, and 3D spatial relationship. The final answer should be chosen from 'A', 'B', 'C', and 'D’.

A: C (Left)   
![](images/80477ef05dcbd6d4e953cc0cf6f9023ea46af3b839c1eb1307446b0089b865d0.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a historic European city square featuring the St. Peter's Tower and surrounding buildings (no signage)
</details>

I've taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position '1' and it's facing to me. According to the camera, where is the region marker '2'? A. Right-Front B. Right-Back C. Left-Front D. Left-Back. Consider cues such as depth, orientation, and 3D spatial relationship and ignore the height difference between the two regions. The final answer should be chosen from 'A', 'B', 'C', and 'D’.

A: A (Right-Front)   
Figure 9. Examples of the task Relative Position Prediction   
Table 8. The values of parameters in formulating Relative Position Prediction samples. px stands for pixels and nd is the normalized depth in the raw RGB-D images, ranging from 0 to 1. 

<table><tr><td>Object Orientation</td><td> $\theta$ </td><td> $\delta_{x}$ </td><td> $\delta_{z}$ </td></tr><tr><td rowspan="2"> $\vec{z}_{obj} \perp$  Image Plane</td><td> $0^{\circ}$ </td><td>150 (px)</td><td>0.25 (nd)</td></tr><tr><td> $180^{\circ}$ </td><td>150 (px)</td><td>0.25 (nd)</td></tr><tr><td rowspan="2"> $\vec{z}_{obj} //$  Image Plane</td><td> $90^{\circ}$ </td><td>0.25 (nd)</td><td>150 (px)</td></tr><tr><td> $270^{\circ}$ </td><td>0.25 (nd)</td><td>150 (px)</td></tr></table>

(0-1) as the unit for measuring in the direction perpendicular to the image plane $( \mathbf { e . g . } , \vec { z } _ { c a m } )$ . Since the orientation of the object changes due to the value of θ, $\delta _ { x }$ and $\delta _ { z }$ are determined by their direction (parallel or perpendicular to the image plane). We set the thresholds as 150 pixels in the direction parallel to the image plane and 0.25 in the direction perpendicular to the image plane. The detailed values are shown in Tab. 8.

# A.7. Question Templates and Format Prompt

We provide the question templates for the five selfsupervised tasks. The words in green are alternative content determined in the automated construction procedure.

# Shuffled Patch Reordering

# 1. M = 2, N = 2, w.o. mask:

Question: The image has been divided into 4 shuffled patches labeled 0, 1, 2, and 3. Based on visual clues such as continuity, alignment, and context, answer the correct arrangement of the patches to restore the original image, where the format is ‘TopLeft-TopRight-BottomLeft-BottomRight’.

# 2. M = 2, N = 2, with mask:

Question: The image has been divided into 4 shuffled patches labeled 0, 1, 2, and 3. One of the four patches is masked completely by white pixels. Based on visual clues such as continuity, alignment, and context, answer the correct arrangement of the patches to restore the original image, where the format is ‘TopLeft-TopRight-BottomLeft-BottomRight’.

# 3. M = 1, N = 3 or 4, w.o. mask:

Question: The image has been divided into 3(4) shuffled horizontal strips labeled 0, 1, 2, and 3. Based on visual clues such as continuity, alignment, and context, answer the correct arrangement of the strips to restore the original image, where the format is ‘Left-Middle-Right’(‘Left-Middle1-Middle2-Right’).

# 4. M = 3 or 4, N = 1, w.o. mask:

Question: The image has been divided into 3(4) shuffled vertical strips labeled 0, 1, 2, and 3. Based on visual clues such as continuity, alignment, and context, answer the correct arrangement of the strips to restore the original image, where the format is ‘Top-Middle-Bottom’(‘Top-Middle1-Middle2-Bottom’).

# Flipped Patch Recognition

Question: The image has been divided into 4 labeled 0, 1, 2, and 3. One of the four patches is flipped either horizontally or vertically. Based on visual clues such as continuity, alignment, and context, answer the correct patch that is flipped and the direction the flip, where the format is ’Label-Direction’. The direction can only be 0(flipped vertically) or 1(flipped horizontally).

# Cropped Patch Inpainting

Question: Which image is the missing part in the first image <image1>? Based on visual clues such as alignment, image content, and positional relationship, select one of the four options <image2><image3><image4><image5> as the final answer. The final answer should be chosen from ‘A’, ‘B’, ‘C’, and ‘D’.

# Regional Depth Ordering

Question: The original image has three regions marked as 1, 2, and 3. Consider the content, positional relationships, depths of the three regions and other cues, and sort the depths of the three regions from smallest to largest from the camera, where the format of the answer is ‘Smallest-Middle-Largest’.

# Regional Depth Ordering

Question: I’ve taken an image and there are two regions marked as 1, and 2 on the image. Assume that there is a camera at position ‘1’(‘2’) and it’s facing to the left of the image. According to the camera, where is the region marked ‘2’(‘1’)? A. Front (Right-Front) B. Right(Right-Back) C. Back(Left-Front) D. Left(Left-Back). Consider cues such as depth, orientation, and 3D spatial relationship. The final answer should be chosen from ‘A’, ‘B’, ‘C’, and ‘D’.

During GRPO training, a format prompt is appended at the end of the task questions to enable the model to generate reasoning content. By following a fixed format, it’s also easier to extract the final answers from the model’s responses for computing the accuracy reward.

# Format Prompt for Training

You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within ⟨think⟩ ⟨/think⟩ tags. The final answer MUST BE put in \boxed{}.

# B. Evaluation Details

# B.1. Spatial Benchmarks

Benchmarks supported in VLMEvalkit. VLMEvalkit supports 3DSRBench, SpatialEval, QSpatial-plus, and Spatial457. The former two benchmarks are multiple-choice problems with a non-CoT original prompt. Therefore, we use the official code in VLMEvalkit to evaluate the reasoning-free settings in Tab. 1 and add our training format prompt to evaluate all reasoning-required settings.

QSpatial-plus targets the quantitative prediction of 3D distances and requires a strict output format encompassing scalar and distance unit to facilitate its final score computation. Instead of employing our format prompt, we follow the official prompts (including both non-reasoning and reasoning version) during the evaluation of all models.

Spatial457 is also not in the form of multiple-choice questions, and its prompt requires CoT response. So we use the original prompt for reasoning-required baseline settings in Tab. 1, and our format prompt for Spatial-SSRL-3B (and 7B) to ensure consistency with training. For reasoning-free baseline settings, the prompt to enable direct outputs of the answers is: Please directly give the answer.

Other Benchmarks. What’sUp, ViewSpatial, and VSI-Bench are not supported in VLMEvalkit. Our evaluation implementation makes minor modification to the official code to adapt it to Qwen2.5-VL architecture while strictly preserving the original metrics and evaluation procedures.

What’sUp contains multiple-choice problems targeting the recognition of unambiguous 2D spatial relation of two objects in an image (e.g., A mug under a table). The evaluation metric is exact matching of the option letter. We apply the following format prompt for evaluating non-reasoning settings in Tab. 1: Based on the image, choose the correct option from the list below., and append our training format prompt for testing reasoning settings.

ViewSpatial aims at evaluating multi-perspective spatial reasoning, requiring the model’s capability of 3D reconstruction and perspective transformation. Similarly, all problems are in the form of multiple-choice questions. We use the official code in our experiment. However, the original prompt doesn’t explicitly instruct the model to generate a reasoning process, but it also doesn’t guide the model to directly output the final answer. To accommodate it to our experiment settings, we define both the non-reasoning and reasoning format prompts as follows:

• Prompt for Reasoning-free settings: Reply only to the corresponding option.\nAnswer:   
• Prompt for Reasoning settings: The final answer should be the option letter from the given choices.\n + Format Prompt for Training

VSI-Bench targets spatial understanding of egocentric videos. It contains multiple-choice answers and numerical answers format. We follow the metrics proposed by the benchmark [68], which uses exact matching for multiplechoice answers and Mean Relative Accuracy (MRA) for numerical answers. Given a numerical model prediction yˆ and its corresponding ground-truth value, MRA is defined with a confidence threshold set $\mathcal { C } = \{ 0 . 0 5 , 0 . 1 0 , . . . , 0 . 5 \}$ : M RA = P 1( |yˆ−y| $M R A = \sum _ { x \in { \mathcal { C } } } \mathbb { 1 } ( { \frac { | { \hat { y } } - y | } { y } } \ < \ x )$ . To balance efficiency and video quality, set max frames to 128 for each video input.

Both non-reasoning and reasoning format prompts for VSI-Bench are given as follows:

• Prompt for Reasoning-free settings: Answer directly with a number(integer or decimal). / Answer directly with the option letter from the given choices.   
• Prompt for Reasoning settings: The final answer should be a number(integer or decimal).\n + Format Prompt for Training / The final answer should be the option letter from the given choices.\n + Format Prompt for Training

# B.2. General Visual Benchmarks

Evaluation Implementation. All the benchmarks in Sec. 4.2.2 for testing models’ general visual capabilities are supported in VLMEValkit. And we use it to implement the entire evaluation of all models on these benchmarks.

Baseline Models. The baseline models are evaluated by directly applying the original prompt provided in the toolkit.

Our Models. For our models (Spatial-SSRL-3B and Spatial-SSRL-7B), we employ the original prompt in general VQA benchmarks as we discover that the reasoning process hardly yields benefits in such problems only requiring simple visual perception, and we append the training format prompt for OCR and chart understanding benchmarks for consistency with training since they demand some basic analysis (e.g., numeric comparison, calculation) as well as fine-grained comprehension of rich visual details, which shares similarities to our depth-free tasks.

# B.3. Results on Qwen3-VL-based Models

We train Spatial-SSRL-4B, initialized from Qwen3-VL-4B-Instruct, on our dataset Spatial-SSRL-81k solely composed of self-supervised QA samples. To make the evaluation consistent with Tab. 1 and Tab. 3 in Sec. 4, we evaluate both the non-reasoning and reasoning variants of the baseline model for spatial understanding benchmarks and compare them with Spatial-SSRL-4B. The results are shown in

Table 9. Performance of Qwen3-VL-4B (baseline model) and Spatial-SSRL-4B on spatial understanding. 

<table><tr><td rowspan="2">Models</td><td rowspan="2">Reasoning</td><td colspan="6">Image</td><td>Video</td><td rowspan="2">Avg.</td></tr><tr><td>Spatial457</td><td>3DSRBench</td><td>SpatialEval</td><td> $QSpatial_{plus}$ </td><td>What&#x27;sUp</td><td>ViewSpatial</td><td>VSI-Bench</td></tr><tr><td>Qwen3-VL-4B</td><td>✘</td><td>53.43</td><td>56.46</td><td>63.04</td><td>63.37</td><td>98.78</td><td>39.09</td><td>46.82</td><td>60.14</td></tr><tr><td>Qwen3-VL-4B</td><td>✓</td><td>55.25</td><td>55.83</td><td>71.69</td><td>61.39</td><td>96.83</td><td>41.79</td><td>38.82</td><td>60.23</td></tr><tr><td>Spatial-SSRL-4B</td><td>✓</td><td>57.12</td><td>59.48</td><td>72.38</td><td>59.41</td><td>97.44</td><td>42.07</td><td>42.13</td><td>61.43 (+1.29)</td></tr></table>

Tab. 9. The results of baseline models and our model on general VQA benchmarks are provided in Tab. 10.

Table 10. Performance of Qwen3-VL-4B (baseline model) and Spatial-SSRL-4B on general VQA. Our model has achieved an average accuracy gain of 1.18%. 

<table><tr><td>Models</td><td>MMBench</td><td>BLINK</td><td>Hallusion</td><td>RealWorld</td><td>Avg.</td></tr><tr><td>Qwen3</td><td>84.10</td><td>64.81</td><td>56.38</td><td>71.11</td><td>69.10</td></tr><tr><td>Ours</td><td>84.26</td><td>65.23</td><td>62.23</td><td>69.41</td><td>70.28</td></tr></table>

# C. Qualitative Analysis

In this section, we provide some qualitative examples to compare the performance of Spatial-SSRL-7B and our base model Qwen2.5-VL-7B on spatial understanding problems. These examples cover a broad range of spatial intelligence: 3D location understanding, depth comparison, orientation recognition, multi-object relationship perception, and reasoning on viewpoint transformation.

Height Higher   
![](images/c5360efcb356415c86437e4ec3c5cd7e0b5721123cf990a354a4748287d8f7ab.jpg)

<details>
<summary>natural_image</summary>

Large black-and-white striped inflatable kite flying in the sky, with spectators and buildings in the background (no visible text or symbols)
</details>

Q: Consider the real-world 3D locations of the objects. Which object has a higher location? A. yellow bear kite B. building

# Qwen2.5-VL-7B:

To determine which object has a higher location, we need to compare the vertical positions of the yellow bear kite and the buildings in the background.\n\n1. The yellow bear kite is flying in the sky, and its position is above the ground.\n2. The buildings in the background are on the ground level, indicating they are at a lower elevation compared to the kite.\n\nSince the kite is in the air and the buildings are on the ground, the kite must be at a higher location than the buildings.\n\nFinal answer: \\boxed{A. yellow bear kite}

# Spatial-SSRL-7B:

<think>\nTo determine which object is higher, let's analyze the image:\n\n1. The yellow bear kite is in the foreground, closer to the ground.\n2. The buildings in the background are further away from the viewer.\n\nSince the buildings are further away, they are higher in the real-world 3D space compared to the yellow bear kite, which is closer to the ground.\n\nTherefore, the building is higher than the yellow bear kite.\n</think>\n\\boxed{B}

Location Above   
![](images/662ec67065a033f6ca790d3791cfe322e6845846774c30c739d75fdc1ee2929e.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a train station platform with overhead electric lines and passengers (no signage)
</details>

Q: "Consider the real-world 3D locations of the objects. Is the clock directly underneath the train station roof top? A. yes B. no

# Qwen2.5-VL-7B:

<think>\nThe image shows a train at a station with a platform on the right side. Above the platform, there is a structure that appears to be part of the train station roof. There is a clock mounted on this structure. The clock is positioned above the platform, not directly underneath it. Therefore, the clock is not directly underneath the train station roof.\n</think>\n\\boxed{B}

# Spatial-SSRL-7B:

<think>\nTo determine if the clock is directly underneath the train station roof top, I need to assess the spatial relationship between the clock and the roof. In the image, the clock is mounted on the roof structure, and it appears to be positioned at a height that aligns with the roof's underside. The clock is not hanging from the roof but is rather part of the roof structure itself, indicating it is directly underneath the roof.\n</think>\n\\boxed{A}

Location Closer to Camera   
![](images/41356db5a948fb98746923064d10445363d61f89cfb3b2ddc0260f9a5d0dbe65.jpg)

<details>
<summary>natural_image</summary>

Harbor scene with a large red and white cargo ship docked near a dock, surrounded by boats and parked cars (no visible text or signage)
</details>

Q: "Consider the real-world 3D location of the objects. Which object is further away from the camera? A. boat B. fire hydrant

# Qwen2.5-VL-7B:

<think>\nTo determine which object is further away from the camera, we need to consider the perspective and depth in the image. The boat appears larger and more detailed, suggesting it is closer to the viewer. The fire hydrant, on the other hand, is smaller and less detailed, indicating it is farther away.\n\nTherefore, the fire hydrant is f ur t her away fro m t h e camera.\n</think>\n\\boxed{B. fire hydrant}

# Spatial-SSRL-7B:

<think>\nTo determine which object is further away from the camera, I need to assess the depth cues in the image. The boat appears to be in the water, which is typically further away than objects on the land. The fire hydrant is on the ground, closer to the viewer. Therefore, the boat is further away from the camera.\n</think>\n\\boxed{A. boat}

Figure 10. Qualitative examples of spatial understanding

Orientation Viewpoint   
![](images/27fc7b9e4a6c7f6d395486712c8d819b6626176d59b1cd5fc4c6349e57e43d73.jpg)

<details>
<summary>natural_image</summary>

Exterior view of the Tower Bridge in London with a river and boats on the water (no signage or text visible)
</details>

Q: Consider the real-world 3D locations and orientations of the objects. Which side of the a i r p l a n e is f a c i n g the camera? A. front B.left C. back D. right

# Qwen2.5-VL-7B:

To determine which side of the airplane is facing the camera, let's analyze the image step by step:\n\n1. The airplane is positioned on the water near the bridge.\n2. The orientation of the airplane relative to the bridge and the surrounding structures can help us understand its position.\n3. The airplane appears to be facing towards the left side of the image from our perspective.\n\nGiven this analysis, the airplane is facing towards the left side of the image, which means the front of the airplane is facing the camera.\n\nThe final answer is: \\boxed{B}

# Spatial-SSRL-7B:

<think>\nTo determine which side of the airplane is facing the camera, I need to consider the orientation of the airplane relative to the viewer. In the image, the airplane appears to be positioned with its nose pointing towards the camera. This is because the airplane's wings and tail are visible, and the perspective suggests that the front of the airplane is facing the viewer.\n\nTherefore, the airplane is facing the camera from the front.\n</think>\n\\boxed{A}

Multi-obj. Orientation   
![](images/7c406d86932156f4c58ca16103e3faaa382483c37f3c1c31398b3d6f1d3ced15.jpg)

<details>
<summary>natural_image</summary>

Child sleeping in a car while holding a teddy bear (no visible text or symbols)
</details>

Q: Consider the real-world 3D orientations of the objects. Are the kid and the teddy bear facing same or similar directions, or very different directions? A. very different directions B. same or similar directions

# Qwen2.5-VL-7B:

<think>\nTo determine if the kid and the teddy bear are facing the same or similar directions, let's analyze their positions:\n\n1. The kid is sleeping with their head slightly tilted to the side, which means they are not facing directly forward but rather at an angle.\n2. The teddy bear is positioned in front of the kid, and its face is oriented towards the camera, which suggests it is facing forward.\n\nGiven these observations:\n- The kid's head is tilted, indicating a direction that is not directly forward.\n- The teddy bear is facing forward, which is a different direction from the kid's tilted position.\n\nTherefore, the kid and the teddy bear are facing very different directions.\n</think>\n\\boxed{A. very different directions}

# Spatial-SSRL-7B:

<think>\nTo determine the orientation of the kid and the teddy bear, I need to consider their positions relative to each other. The kid is facing forward, towards the camera, with their head slightly tilted. The teddy bear is positioned in front of the kid, also facing forward. Both the kid and the teddy bear are oriented in the same direction, which is towards the camera.\n\nTherefore, the final answer is that they are facing the same or similar directions.\n</think>\n\\boxed{B}

Perspective Transformation   
![](images/60e977bbd70f7874d37636e9f470504606b3eb831ed52ee1f8a5815a33b3f8bc.jpg)

<details>
<summary>natural_image</summary>

A small dog standing on pavement next to a white bus and a blue-labeled vehicle (no visible text or symbols)
</details>

Q: Consider the real-world 3D locations and orientations of the objects. If I stand at the recreational vehicle's position facing where it is facing, is the dog in front of me or behind me? A. behind B. in front of

# Qwen2.5-VL-7B:

<think>\nTo determine if the dog is in front of or behind the recreational vehicle (RV) when standing at its position, we need to consider the orientation and relative positions of the RV and the dog.\n\n1. The RV is positioned with its front facing towards the left side of the image.\n2. The dog is standing on the ground in the foreground, closer to the viewer than the RV.\n3. If you were to stand at the RV's position and face forward, the dog would be in your line of sight and closer to you than the RV.\n\nGiven this information, the dog is in front of the RV when you are standing at its position.\n\nThe final answer is: \\boxed{B}"

# Spatial-SSRL-7B:

<think>\nTo determine the position of the dog relative to the recreational vehicle, I need to consider the orientation of the RV and the dog's position in the image.\n\n1. The recreational vehicle is facing away from the camera, as indicated by the direction of its wheels and the orientation of the trailer.\n2. The dog is positioned in the foreground, closer to the camera than the RV.\n3. Since the dog is in front of the RV from the camera's perspective, it is behind the RV from the RV's perspective.\n\nTherefore, the dog is behind the recreational vehicle if I stand at the RV's position facing where it is facing.\n</think>\n\\boxed{A}

Figure 11. Qualitative examples of spatial understanding