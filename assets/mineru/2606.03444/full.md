# PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization

Ying Tang 1 Dong Li 1 Youjia Zhang 1 Zikai Song 1 Junqing Yu 1 Wei Yang 1 , †

https://github.com/robotyingtang/PRISM-VFM

# Abstract

Unifying the complementary strengths of diverse Vision Foundation Models (VFMs) into a single efficient model is highly desirable but challenged by the negative transfer inherent in monolithic distillation. To address these feature conflicts, we introduce PRISM, a novel dual-stream Mixture-of-Experts (MoE) framework that synergizes VFMs via modular specialization. We propose a two-stage paradigm: (1) expertise deconstruction, where a teacher-conditional router guides experts to specialize in distinct representational subspaces to mitigate interference, followed by (2) dynamic recomposition, where the router learns to assemble these experts into tailored computational pathways for downstream tasks. Experiments on PASCAL-Context and NYUD-v2 show that PRISM establishes a new state of the art, validating that sparse, emergent specialization is a scalable approach for integrating diverse visual knowledge.

# 1. Introduction

A practical goal in computer vision is to consolidate multiple Vision Foundation Models (VFMs) into a single deployable student. Modern VFMs provide complementary cues, such as high-level semantics from CLIP (Radford et al., 2021), spatial structures from SAM (Kirillov et al., 2023), and finegrained texture or correspondence from DINOv2 (Oquab et al., 2023); however, deploying them as a model ensemble is costly in memory, latency, and engineering complexity. The desired student should absorb these complementary visual dimensions into one parameter space while avoiding

1School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan, China. Correspondence to: Wei Yang <weiyangcs@hust.edu.cn>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

the negative transfer caused by naive dense distillation.

However, compressing such heterogeneous knowledge into a unitary network introduces a fundamental optimization paradox. When shared parameters are forced to satisfy contradictory supervision signals, the model suffers from severe gradient conflict. For instance, while DINO encourages feature variance to distinguish local textures, CLIP often suppresses such variance to achieve semantic invariance. In standard dense architectures, these opposing gradient vector fields lead to destructive interference, causing the shared weights to collapse into a compromised average that fails to excel in either dimension.

To mitigate this interference, pioneering approaches such as SAK (Lu et al., 2025) adopt a divide-and-conquer strategy, assigning independent parameter branches to different teachers. While effective, this paradigm relies on the rigid assumption of hard boundaries, that visual knowledge can be explicitly sliced into disjoint domains. We argue that this is an oversimplification. In reality, visual knowledge exhibits soft boundaries with intricate overlaps. For example, both CLIP and DINO encode representations of a cat, yet they focus on different frequency bands (semantic identity vs. local texture) of the same entity. Static partitioning ignores this nuance, creating parameter redundancy and hindering the positive transfer of shared concepts. Consequently, a robust student requires a dynamic architecture capable of automatically perceiving the nature of the knowledge: sharing parameters for consensus and branching out only when functional conflicts arise.

In this work, we propose PRISM (Projecting Representations into Independent Specialized Modules), a framework that shifts from “manual partitioning” to self-organized specialization. Rather than explicitly assigning layers or modules to particular teachers or tasks, PRISM adopts a dual-stream gated Mixture-of-Experts (MoE) architecture for conflict-aware partial sharing. A Universal Anchor stream preserves stable shared representations and captures consensus knowledge, while a Conditioned MoE stream provides plasticity by routing tokens to sparse experts conditioned on layer, token, and teacher/task context. Driven by context-modulated routing and locality-aware decorrelation, PRISM dynamically shares compatible knowledge, separates conflicting signals, and recombines specialized experts when knowledge partially overlaps, enabling emergent expert specialization while maintaining parameter efficiency.

Our contributions are summarized as follows:

• We conceptually reframe multi-teacher distillation as a dynamic consensus-conflict trade-off, identifying the limitations of static “hard boundary” partitioning in prior arts.   
• We propose PRISM, a dual-stream MoE architecture that leverages Context-modulated routing to achieve implicit and emergent knowledge decomposition.   
• To enable effective expert specialization, we introduce a locality-aware decorrelation mechanism that prevents semantic short-circuiting in shallow layers, acting as a critical inductive bias.

# 2. Related Work

Knowledge Distillation of VFMs As large-scale generalists, Vision Foundation Models (VFMs) demonstrate superior performance across diverse domains(e.g., vision (Song et al., 2026; Ye et al., 2025; Song et al., 2025) and multimodal tasks (Li et al., 2024a; Wang et al., 2026; Zhang et al., 2026)) with minimal tuning. Notable examples include CLIP (Radford et al., 2021) for vision-language understanding, DINOv2 (Oquab et al., 2023) for fine-grained representation learning, and SAM (Kirillov et al., 2023) for promptable segmentation. Many of them share attention-based designs (Song et al., 2022; 2023; 2024). Despite their efficacy, the substantial computational demands of these models have led to the widespread adoption of knowledge distillation (Vandenhende et al., 2021; Ye & Xu, 2023; Zong et al., 2024) for VFM compression and efficiency (Zhang & Yang, 2021; Ishihara et al., 2021; Yu et al., 2024).

Multi-teacher distillation has also been studied before the VFM era, and typically by aggregating soft targets or intermediate features from multiple relatively homogeneous teachers (Fukuda et al., 2017; You et al., 2017; Liu et al., 2020). More recently, research has shifted towards distilling multiple VFMs into a single student to synergize their heterogeneous strengths(Zhou et al., 2025; Ranzinger et al., 2024a; Shi et al., 2024). SAM-CLIP (Wang et al., 2024a) merges CLIP’s semantic knowledge into SAM via continual learning. RADIO (Ranzinger et al., 2024b) distills CLIP, DINOv2, and SAM simultaneously to enhance downstream performance, a direction further refined by RADIOv2.5 (Heinrich et al., 2025), introducing improved training recipes and scaling laws for more robust feature aggregation. Similarly, DUNE (Sarıyıldız et al., 2025) bridges the gap between 2D and 3D perception by distilling a universal encoder from heterogeneous teachers, while Theia (Shang et al., 2024) incorporates Depth Anything (Yang et al., 2024a) for robot learning. Furthermore, UNIC (Sariyildiz et al., 2024) utilizes multi-teacher distillation to consolidate specialized experts into a universal classification backbone. Most existing multi-teacher distillation methods (Ranzinger et al., 2024b; Shang et al., 2024) employ straightforward distillation into a dense backbone. We argue this leads to inherent feature interference, particularly between conflicting domains such as semantics and geometry. Unlike these approaches, PRISM adaptively transfers knowledge by retaining unique representation biases through a dynamic MoE mechanism to maximize strengths across multiple tasks.

Multi-Task Learning Architectures Multi-Task Learning (MTL)(Caruana, 1997) aims to train a single model capable of handling multiple tasks simultaneously (Ruder, 2017; Chen et al., 2020; Ishihara et al., 2021; Li et al., 2025; 2026). Research generally diverges into two categories: multi-task optimization (Chen et al., 2018; Guo et al., 2018; Kendall et al., 2018; Liu et al., 2021b;a; Li et al., 2024b) and model architecture design (Bruggemann et al. ¨ , 2021; Ye & Xu, 2022a;b). In terms of architectures, existing methods are typically categorized into encoder-focused (Liu et al., 2019; Wang et al., 2025) and decoder-focused (Ye & Xu, 2022a) approaches. Recently, knowledge distillation (Zeng et al., 2026) has been integrated into MTL to bridge the gap between multi-task students and single-task teachers (Ranzinger et al., 2024b; Heinrich et al., 2025; Lu et al., 2025). Notably, Xu et al. (Xu et al., 2023) proposed directly distilling a small multi-task student from a large multi-task teacher.

Traditional MTL architectures typically rely on static sharing designs, which can be broadly categorized into encoderfocused (Liu et al., 2019) and decoder-focused (Ye & Xu, 2022a) approaches. Although such hard or partially shared parameterizations are simple and effective, static sharing can lead to negative transfer when task objectives conflict.

To improve flexibility, recent works have explored dynamic architectures based on Mixture-of-Experts (MoE). For instance, Mod-Squad (Chen et al., 2023) modularizes experts with an information-theoretic objective to balance cooperation and specialization, while TaskExpert (Ye & Xu, 2023) dynamically assembles representations through a memorybased MoE mechanism in the decoding stage. While promising, these methods do not explicitly address the semantic conflicts that may arise in multi-teacher distillation. In this regard, PRISM introduces conflict-aware routing for partial sharing, and remains complementary to standard MoE regularizers such as load balancing and routing entropy (Shazeer et al., 2017; Fedus et al., 2022). It also differs from FiLMbased MoE variants such as MoFME (Zhang et al., 2024): in PRISM, FiLM conditions routing decisions rather than replacing expert computation.

![](images/f6d4afea3180e209d93b61a3009734caf9c4dd0a6b91bf6d57985b00181065a2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["Block 1"]
    B --> C["Block 2"]
    C --> D["Ldecorr"]
    D --> E["Dual-Stream Conditioned MoE"]
    E --> F["Block N"]
    F --> G["Ldistill"]
    G --> H["Condition ID (Teacher ID)"]
    H --> I["VFM Teacher 1"]
    H --> J["VFM Teacher 2"]
    H --> K["VFM Teacher N"]
    I --> L["Universal Anchor"]
    J --> L
    K --> L
    L --> M["Standard Feed-Forward Network"]
    M --> N["Gate λ"]
    N --> O["+"]
    O --> P["LayerNorm"]
    Q["Teacher ID"] --> R["Multi-Head Self-Attention (MSA)"]
    R --> S["LayerNorm"]
    T["Task ID"] --> U["FiLM Layer"]
    U --> V["Router"]
    V --> W["Specialized Delta"]
    W --> X["Expert 1"]
    W --> Y["Expert 2"]
    W --> Z["Expert N"]
    W --> AA["Shared Expert Bias Compensator"]
    W --> AB["Summation"]
    AC["Task Loss"] --> AD["Head"]
    AE["Stage2"] --> AF["Head"]
    AF --> AG["Head"]
```
</details>

Figure 1. Overview of the PRISM framework. (Top) Two-Stage Training Pipeline. In Stage 1, the student (Dual-Stream Conditioned MoE) mimics multiple frozen VFM teachers. The Context ID (Teacher ID) conditions the routing, driving emergent knowledge decomposition. The $\bar { \mathcal { L } } _ { \mathrm { d e c o r r } }$ is applied to shallow layers to prevent rank collapse. In Stage 2, the model recombines experts for downstream tasks using the Task ID as context. (Bottom) Dual-Stream Architecture. The PRISM block replaces standard FFNs with two parallel paths: a Universal Anchor for shared consensus, and a Specialized Delta for conflict resolution. A FiLM-based Router modulates features based on context before dispatching tokens to sparse experts. A learnable gate λ dynamically fuses the two streams.

In the specific context of distilling Vision Foundation Models (VFMs), recent approaches like RADIO (Ranzinger et al., 2024b) and SAK (Lu et al., 2025) have emerged. SAK represents the state-of-the-art in static architecture, employing a “Teacher-Agnostic Stem” plus “Teacher-Specific Adapters.” While this reduces interference, its reliance on rigid, manual branching assumes hard boundaries between tasks, leading to structural redundancy when knowledge overlaps.

Positioning PRISM. Bridging these paradigms, PRISM proposes a dynamic encoder-focused architecture tailored for multi-teacher distillation. Unlike Mod-Squad which relies on loss constraints, or SAK which relies on static branches, PRISM leverages a context-modulated router to structurally resolve gradient conflicts. By treating experts as a basis for emergent knowledge decomposition, PRISM automatically shares parameters for consensus and branches out for conflicts token-by-token. This fine-grained mechanism solves the “Negative Transfer” problem more efficiently than rigid branching, offering a “best-of-both-worlds” solution between stability and plasticity.

# 3. Methodology

To resolve the gradient conflict in Multi-Teacher Distillation, we propose PRISM, a framework centered on the principle of “Decompose-then-Recombine”. As illustrated in Figure 1, PRISM transforms a standard Vision Transformer (ViT) into a Dual-Stream Conditioned MoE architecture. PRISM introduces a dynamic trade-off mechanism: a Universal Anchor stream maintains stability for shared consensus, while a Specialized Delta stream provides plasticity for conflict resolution via context-modulated routing. The training proceeds in two stages: emergent knowledge decomposition from multiple VFM teachers (Stage 1), followed by knowledge recombination for downstream tasks (Stage 2), regularized by a locality-aware decorrelation loss.

# 3.1. Problem: Gradient Conflict in MTD

We formulate the task of Multi-Teacher Distillation (MTD) as a multi-objective optimization problem. Let T = $\{ T _ { 1 } , T _ { 2 } , \dots , T _ { K } \}$ denote a set of K heterogeneous vision foundation models (VFMs), serving as teachers. The student model S, parameterized by Θ, aims to minimize the joint distillation loss over the training distribution X :

$$
\mathcal {L} _ {\text { total }} (\mathbf {x}; \Theta) = \sum_ {k = 1} ^ {K} \gamma_ {k} \mathcal {L} _ {\text { distill }} (S (\mathbf {x}), T _ {k} (\mathbf {x})) \tag {1}
$$

where $\gamma _ { k }$ is a scalar coefficient balancing the contribution of each teacher.

Gradient conflict in dense architectures. Consider a standard dense layer (e.g., the FFN in a ViT block) with parameters $\theta \in \Theta$ . During back-propagation, the parameter update $\Delta \theta$ is proportional to the aggregate gradient vector $\begin{array} { r } { \mathbf { g } _ { \mathrm { t o t a l } } = \sum _ { k = 1 } ^ { K } \gamma _ { k } \mathbf { g } _ { k } } \end{array}$ , where $\mathbf { g } _ { k } \ = \ \nabla _ { \theta } \mathcal { L } _ { \mathrm { d i s t i l l } } ( T _ { k } )$ . A fundamental optimization dilemma arises when teachers provide contradictory supervision for the same input features. Mathematically, this conflict manifests as a negative cosine similarity between task gradients:

$$
\mathcal {C} _ {i, j} = \cos (\mathbf {g} _ {i}, \mathbf {g} _ {j}) = \frac {\left\langle \mathbf {g} _ {i} , \mathbf {g} _ {j} \right\rangle}{\| \mathbf {g} _ {i} \| \| \mathbf {g} _ {j} \|} <   0 \tag {2}
$$

In a standard dense architecture where parameters are globally shared, severe conflict $( \mathcal { C } _ { i , j } \ll 0 )$ leads to destructive interference, where $\mathbf { g } _ { i } \approx - \mathbf { g } _ { j } .$ Consequently, the magnitude of the aggregate gradient $\| \mathbf { g } _ { t o t a l } \|$ diminishes, and the optimization settles into a “compromised” equilibrium that is suboptimal for all tasks. We refer to this phenomenon as gradient averaging.

# 3.2. Motivation: Decomposition via Gradient Orthogonalization

To mitigate the gradient averaging dilemma, we argue that the student parameter space should distinguish between consensus and conflict components. Consensus knowledge, such as generic low-level structures, can be safely shared, whereas conflicting teacher-specific signals should update separated or weakly coupled parameters.

Specifically, for two teacher objectives $T _ { i }$ and $T _ { j }$ with conflicting gradients, i.e., $\cos ( \mathbf { g } _ { i } , \mathbf { g } _ { j } ) < 0$ , PRISM encourages their routed effective gradients on each sparse expert $E _ { n }$ to have small interaction:

$$
\langle \tilde {\mathbf {g}} _ {i, n}, \tilde {\mathbf {g}} _ {j, n} \rangle \approx 0, \tag {3}
$$

where $\tilde { \bf g } _ { i , n }$ denotes the gradient component that is actually routed to expert $E _ { n }$ under teacher context $T _ { i } .$ . This objective can be achieved either by reducing co-activation of the same expert or by making the residual gradients on co-active experts weakly aligned. It motivates our dual-stream design: the Universal Anchor preserves shared consensus, while the Conditioned MoE reduces effective interference through sparse, context-dependent dispatching.

# 3.3. Architecture: Dual-Stream Conditioned MoE

Guided by the orthogonality principle, we introduce the PRISM block to replace standard FFNs. Unlike SAK (Lu et al., 2025) which decomposes the network into a static “Teacher-Agnostic Stem” and “Teacher-Specific Adapters”, PRISM relaxes this hard constraint into a continuous, dynamic trade-off between Stability and Plasticity. Formally, for an input x, the block output is:

$$
\mathbf {y} = \mathbf {x} + \underbrace {\lambda \cdot \mathcal {F} _ {\mathrm{anc}} (\mathrm{LN} (\mathbf {x}))} _ {\text { Stability   Stream }} + \underbrace {(1 - \lambda) \cdot \mathcal {F} _ {\mathrm{moe}} (\mathbf {x} , c)} _ {\text { Plasticity   Stream }} \tag {4}
$$

where $\lambda \in [ 0 , 1 ]$ is a learnable gating scalar. This gate allows the model to dynamically trade off between reusing common knowledge and invoking specialized experts. Empirical analysis (see Sec. 4.3) reveals that λ naturally starts high in shallow layers (preferring stability) and decreases in deeper layers (favoring specialization), validating our hierarchical design hypothesis.

Universal Anchor. The first stream, ${ \mathcal { F } } _ { \mathrm { a n c } } ,$ is a standard dense MLP shared across all contexts. It captures taskagnostic, low-frequency patterns. We term this the Stability Stream, as it ensures the model maintains a robust optimization trajectory regardless of routing fluctuations.

Specialized Delta. The second stream, ${ \mathcal { F } } _ { \mathrm { m o e } } ,$ is a sparse Mixture-of-Experts. This stream offers a unified pool of experts accessed via Context-Modulated Routing. We term this the Plasticity Stream. By conditioning on context c, it provides the necessary plasticity to resolve gradient conflicts, allowing the model to emergently decide whether to invoke specific experts or combine them, handling Soft Boundaries that static methods miss.

Network Instantiation. We adopt ViT-Base (ViT-B/16) as the backbone. We replace the FFNs with Dual-Stream blocks in the 2nd, 5th, 8th, 11th layers. Each MoE layer contains $N = 1 5$ sparse experts with a Top-3 routing strategy, plus one internal shared expert for bias compensation.

# 3.4. Context-Modulated Routing Mechanism

To guide the Plasticity Stream, we employ a routing mechanism that directs token flow based on both image content and task intent.

Context-Aware Feature Modulation. Standard routers rely solely on input tokens x, which is insufficient for distinguishing tasks with identical inputs. We inject the Context ID c (Teacher ID in Stage 1, Task ID in Stage 2) via Featurewise Linear Modulation (FiLM):

$$
\hat {\mathbf {x}} = (1 + \gamma (c)) \odot \text { LayerNorm } (\mathbf {x}) + \beta (c) \tag {5}
$$

where $\gamma ( c )$ and $\beta ( c )$ are learned affine transformations. This re-orients the feature space, enabling the router to make distinct decisions for CLIP vs. DINO contexts.

Sparse Dispatching. The router $G ( \cdot )$ maps $\hat { \bf x }$ to expert weights via Top-K Softmax. The MoE output is:

$$
\mathcal {F} _ {\mathrm{moe}} (\mathbf {x}, c) = E _ {\text { shared }} (\mathbf {x}) + \sum_ {i \in \text { TopK }} G (\hat {\mathbf {x}}) _ {i} E _ {i} (\mathbf {x}) \tag {6}
$$

The internal shared expert $E _ { \mathrm { s h a r e d } }$ captures common biases specific to the conflict subspace, further stabilizing the sparse routing.

# 3.5. Training Objectives and Strategy

Locality-Aware Decorrelation Loss. A critical prerequisite for effective expert specialization is the diversity of input tokens. However, in multi-teacher distillation, we identify a detrimental phenomenon we call “semantic shortcircuiting”. Driven by strong high-level supervision (e.g., from CLIP), the student model tends to bypass low-level feature extraction, causing shallow layers to prematurely converge to global semantic representations.

Mathematically, this manifests as rank collapse, where feature variance across tokens diminishes rapidly. If input tokens are homogeneous, the router lacks discriminative signals to dispatch them to different experts, leading to routing collapse. To counteract this, we explicitly inject a locality inductive bias by applying a Locality-Aware Decorrelation Loss $( { \mathcal { L } } _ { \mathrm { d e c o r r } } )$ to the shallow blocks. We penalize high cosine similarity between spatially distant pixels while preserving local correlations:

$$
\mathcal {L} _ {\text { decorr }} = \frac {1}{| \mathcal {P} |} \sum_ {(i, j) \in \mathcal {P}} \max (0, \cos (\mathbf {z} _ {i}, \mathbf {z} _ {j}) - \epsilon) \cdot \mathbb {I} (d _ {i j} > r) \tag {7}
$$

where $\mathbf { z } _ { i }$ is the feature vector at position i, $d _ { i j }$ is the spatial Euclidean distance, and r is a locality radius. This constraint forces shallow layers to encode rich, localized structural variations, providing high-quality, high-rank “raw materials” for the deep-layer experts.

Training Strategy. Our training pipeline consists of two distinct stages, as shown in Figure 1.

Stage 1: Emergent Knowledge Decomposition. The student learns from K frozen VFM teachers. For each iteration, we randomly sample a teacher $T _ { k }$ and use its ID as context c. The total loss is:

$$
\mathcal {L} _ {\text { stage1 }} = \mathcal {L} _ {\text { aux }} + \alpha \mathcal {L} _ {\text { distill }} + \beta \mathcal {L} _ {\text { decorr }} \tag {8}
$$

Stage 2: Knowledge Recombination. In this stage, we fine-tune the model to adapt to downstream tasks while preserving the knowledge acquired in Stage 1. The total objective function is formulated as:

$$
\mathcal {L} _ {\text { stage2 }} = \mu \mathcal {L} _ {\text { distill }} + \sum_ {t \in \mathbb {T}} w _ {t} \mathcal {L} _ {t} \tag {9}
$$

where $\mathcal { L } _ { t }$ is the task-specific loss for task t. The hyperparameter $\mu$ balances the distillation loss and the task losses, with a default value of 1.0, while $w _ { t }$ adjusts the importance of each task. We set fixed $w _ { t }$ values following the standard practice in MTL(Maninis et al., 2019; Kanakis et al., 2020).

# 4. Experiments

# 4.1. Experimental Setup

Datasets and Protocol. Following SAK (Lu et al., 2025), we perform Stage 1 pre-training on ImageNet-1k (Deng et al., 2009), followed by Stage 2 fine-tuning and evaluation on two standard multi-task benchmarks. PASCAL-Context (Mottaghi et al., 2014) covers five scene understanding tasks: Semantic Segmentation (SemSeg), Human Parsing (Parsing), Saliency, Surface Normal (Normal), and Boundary Detection (Boundary). NYUD-v2 (Silberman et al., 2012) focuses on indoor scenes with four tasks: Sem-Seg, Depth Estimation, Normal, and Boundary.

Evaluation Metrics. We adopt standard metrics for each task: mean Intersection over Union (mIoU) for SemSeg and Parsing; maximal F-measure (maxF) for Saliency; optimal dataset scale F-measure (odsF) for Boundary; mean Error (mErr) for Normal Estimation; and RMSE for Depth Estimation. Lower values are better for mErr and RMSE. VFM Teachers. We distill knowledge from three frozen Vision Foundation Models (VFMs) with ViT-L backbones, unless otherwise specified: (1) DINOv2-L (Oquab et al., 2023): Provides robust local features and fine-grained correspondence. (2) CLIP-L (Radford et al., 2021): Offers high-level semantic understanding and language-aligned representations. (3) SAM-L (Kirillov et al., 2023): Contributes precise geometric and boundary cues from promptable segmentation. We extract teacher features from layers 5, 11, 17, and 23.

Baselines. We compare PRISM against three categories of methods: (1) Task-Aware MTL Methods, including encoder-decoder architectures like InvPT++ (Ye & Xu, 2024), TaskPrompter (Ye & Xu, 2022b), BFCI (Zhang et al., 2025), MLoRE (Yang et al., 2024b), and SEM (Huang et al., 2024); (2) Unified Foundation Models, such as RA-DIO/RADIOv2.5 (Ranzinger et al., 2024b; Heinrich et al., 2025), UNIC (Sariyildiz et al., 2024), and Theia (Shang et al., 2024), which distill multiple VFMs into a single backbone; and (3) State-of-the-Art, specifically SAK (Lu et al., 2025), which uses explicit architectural branching for multi-teacher distillation.

Implementation Details. We use ViT-B/16 as the backbone.

Table 1. Comparison with state-of-the-art methods on PASCAL-Context (ViT-B backbone). PRISM achieves the best overall performance $( \Delta _ { m } = 2 . 2 9 \% )$ and outperforms SAK on all 5 tasks. 

<table><tr><td>Model</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td><td> $\Delta_m$ % ↑</td></tr><tr><td>Single-task baseline</td><td>80.25</td><td>70.54</td><td>84.54</td><td>13.57</td><td>74.22</td><td>0.00</td></tr><tr><td>Multi-task baseline</td><td>76.76</td><td>65.26</td><td>84.39</td><td>13.98</td><td>70.37</td><td>-4.04</td></tr><tr><td>InvPT(Ye &amp; Xu, 2022a)</td><td>77.33</td><td>66.62</td><td>85.14</td><td>13.78</td><td>73.20</td><td>-2.28</td></tr><tr><td>InvPT++(Ye &amp; Xu, 2024)</td><td>76.95</td><td>66.89</td><td>85.12</td><td>13.54</td><td>73.30</td><td>-1.92</td></tr><tr><td>TaskPrompter(Ye &amp; Xu, 2022b)</td><td>79.00</td><td>67.00</td><td>85.05</td><td>13.47</td><td>73.50</td><td>-1.24</td></tr><tr><td>TaskExpert(Ye &amp; Xu, 2023)</td><td>78.45</td><td>67.38</td><td>84.96</td><td>13.55</td><td>72.30</td><td>-1.73</td></tr><tr><td>BFCI(Zhang et al., 2025)</td><td>77.98</td><td>68.19</td><td>85.06</td><td>13.48</td><td>72.98</td><td>-1.31</td></tr><tr><td>MLoRE(Yang et al., 2024b)</td><td>79.26</td><td>67.82</td><td>85.31</td><td>13.65</td><td>74.69</td><td>-0.83</td></tr><tr><td>RADIO(Ranzinger et al., 2024b)</td><td>78.06</td><td>68.13</td><td>85.18</td><td>13.59</td><td>72.64</td><td>-1.53</td></tr><tr><td>RADIOv2.5(Heinrich et al., 2025)</td><td>81.75</td><td>71.49</td><td>81.26</td><td>16.10</td><td>-</td><td>-</td></tr><tr><td>UNIC(Sariyildiz et al., 2024)</td><td>75.90</td><td>62.85</td><td>81.84</td><td>15.78</td><td>-</td><td>-</td></tr><tr><td>Theia(Shang et al., 2024)</td><td>76.51</td><td>67.53</td><td>84.38</td><td>14.56</td><td>70.34</td><td>-4.33</td></tr><tr><td>SAK(Lu et al., 2025)</td><td>81.88</td><td>74.30</td><td>84.79</td><td>14.02</td><td>74.09</td><td>0.83</td></tr><tr><td>PRISM (Ours)</td><td>82.20</td><td>75.34</td><td>84.81</td><td>13.47</td><td>75.92</td><td>2.29</td></tr></table>

In stage 1, the model is trained for 30 epochs on ImageNet-1K. In stage 2, the model is trained for 40000 iterations on PASCAL-Context and NYUD-v2 using the AdamW optimizer. For Locality-Aware Decorrelation, we apply Ldecorr to the first two layers, setting the hyperparameters in Eq. 8 as $\alpha = 0 . 9$ and $\beta = 0 . 1$ .

# 4.2. Main Results

Table 1 and Table 2 summarize the quantitative comparisons on PASCAL-Context and NYUD-v2, respectively.

Performance on PASCAL-Context. PRISM establishes a new state-of-the-art, achieving an average improvement $( \Delta _ { m } )$ of 2.29%, clearly outperforming the previous best SAK (0.83%) and other strong competitors. Regarding superiority over SAK, PRISM surpasses it across all five tasks. Specifically, on Semantic Segmentation, PRISM achieves 82.20 mIoU (+0.32 vs. SAK), and on Human Parsing, it reaches 75.34 mIoU (+1.04). This suggests that our Context-Modulated Routing effectively preserves high-level semantic knowledge without rigid task-specific branching. Crucially, for geometric precision, PRISM improves Normal Estimation from 14.02 to 13.47 mErr and Boundary Detection from 74.09 to 75.92 odsF, indicating that emergent experts capture shared geometric structures more effectively than physically separated adapters.

Performance on NYUD-v2. On the indoor scene benchmark, PRISM remains highly competitive with SAK. It improves Semantic Segmentation (60.22 vs. 59.93 mIoU) and Depth Estimation (0.4883 vs. 0.4942 RMSE), indicating effective transfer of semantic and geometry-aware knowledge. Meanwhile, SAK retains an advantage on Surface Normal and Boundary Detection, likely because its dedicated adapters provide stronger task-specific locality for indoor geometric and high-frequency cues. This reflects a dataset-dependent balance between flexible cross-teacher recombination and specialized local adaptation.

Table 2. Comparison with state-of-the-art methods on NYUDv2 (ViT-B backbone). PRISM surpasses SAK in semantic and depth estimation tasks. 

<table><tr><td>Model</td><td>Semseg mIoU ↑</td><td>Depth RMSE ↓</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td><td> $\Delta_{m}$ % ↑</td></tr><tr><td>Single-task baseline</td><td>51.15</td><td>0.5792</td><td>19.77</td><td>77.35</td><td>0.00</td></tr><tr><td>Multi-task baseline</td><td>49.27</td><td>0.5823</td><td>19.92</td><td>75.88</td><td>-1.72</td></tr><tr><td>InvPT(Ye &amp; Xu, 2022a)</td><td>50.30</td><td>0.5367</td><td>19.00</td><td>77.60</td><td>2.47</td></tr><tr><td>InvPT++(Ye &amp; Xu, 2024)</td><td>49.79</td><td>0.5318</td><td>18.90</td><td>77.10</td><td>2.40</td></tr><tr><td>TaskPrompter(Ye &amp; Xu, 2022b)</td><td>50.40</td><td>0.5402</td><td>18.91</td><td>77.60</td><td>2.49</td></tr><tr><td>ECS(Shoouri et al., 2023)</td><td>50.46</td><td>0.5332</td><td>18.42</td><td>77.89</td><td>3.53</td></tr><tr><td>BFCI(Zhang et al., 2025)</td><td>51.14</td><td>0.5186</td><td>18.92</td><td>77.98</td><td>3.89</td></tr><tr><td>TSP(Wang et al., 2024b)</td><td>51.22</td><td>0.5301</td><td>18.78</td><td>76.90</td><td>3.26</td></tr><tr><td>SEM(Huang et al., 2024)</td><td>51.34</td><td>0.5222</td><td>18.95</td><td>77.60</td><td>3.67</td></tr><tr><td>RADIO(Ranzinger et al., 2024b)</td><td>55.03</td><td>0.5186</td><td>18.49</td><td>77.97</td><td>6.33</td></tr><tr><td>RADIOv2.5(Heinrich et al., 2025)</td><td>57.19</td><td>0.4980</td><td>20.04</td><td>-</td><td>-</td></tr><tr><td>UNIC(Sariyildiz et al., 2024)</td><td>42.21</td><td>0.6172</td><td>22.78</td><td>-</td><td>-</td></tr><tr><td>Theia(Shang et al., 2024)</td><td>51.80</td><td>0.5367</td><td>19.70</td><td>76.08</td><td>1.83</td></tr><tr><td>SAK(Lu et al., 2025)</td><td>59.93</td><td>0.4942</td><td>17.60</td><td>78.60</td><td>11.11</td></tr><tr><td>PRISM (Ours)</td><td>60.22</td><td>0.4883</td><td>17.81</td><td>76.59</td><td>10.59</td></tr></table>

Table 3. Scaling Results on PASCAL-Context with ViT-L. PRISM obtains the best overall $\Delta _ { m }$ . 

<table><tr><td>Model</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td><td> $\Delta_m$ % ↑</td></tr><tr><td>Single-task baseline</td><td>81.61</td><td>72.77</td><td>83.80</td><td>13.87</td><td>75.24</td><td>0.00</td></tr><tr><td>SAK</td><td>84.01</td><td>76.99</td><td>84.65</td><td>13.82</td><td>76.27</td><td>2.30</td></tr><tr><td>PRISM</td><td>84.34</td><td>77.83</td><td>84.67</td><td>13.43</td><td>76.23</td><td>3.16</td></tr></table>

Scaling to ViT-L. We further evaluate whether PRISM remains effective with a larger student backbone. As shown in Table 3, following the same ImageNet-1K pretraining and PASCAL-Context fine-tuning protocol as SAK, PRISM with ViT-L achieves a $\Delta _ { m }$ of 3.16%, compared with 2.30% for SAK. This suggests that the proposed decompositionand-recombination mechanism is not limited to ViT-B, but continues to improve conflict-aware partial sharing at a larger model scale.

![](images/d7260cdcb39d00833d4eea174a73ec33c89979c97af44d53f2a58b7d846fd2ca.jpg)  
(a) Gradient topology (Layer 11)

![](images/b39a11f228a0beb9b788c96d668049763dbcb837fef46a16e2d0e26d90260a03.jpg)

<details>
<summary>area</summary>

| Comparison       | Standard FFN | MoE Experts |
| ---------------- | ------------ | ----------- |
| dinov2 vs. clip  | 0.3          | 0.6         |
| dinov2 vs. sam   | 0.4          | 1.2         |
| clip vs. sam      | 0.1          | 0.4         |
</details>

(b) Cosine similarity distribution   
Figure 2. Visualization of effective VFM conflict reduction. (a) The joint distribution of gradient norms. Magnitudes are independently normalized to [0, 1] to compare geometric tendencies. Standard FFNs (diamonds) show broad simultaneous updates from multiple VFMs, indicating dense parameter entanglement. In contrast, MoE experts (circles) form an L-shaped topology, suggesting that many sparse experts are predominantly updated by one VFM condition. (b) The density of cosine similarities between VFM gradients. Sparse experts (blue) concentrate around zero, indicating reduced effective interaction, whereas the shared FFN (red) exhibits broader correlations caused by simultaneous multi-teacher updates.

# 4.3. Gradient-Based Conflict Analysis

To validate whether PRISM resolves the optimization conflict among different Vision Foundation Models (VFMs), we analyzed the gradient dynamics. Let θ denote the parameter vector (either in Sparse Experts or the Shared FFN). We define $g _ { v } = \nabla _ { \theta } \mathcal { L } _ { v }$ as the gradient derived from the distillation loss $\mathcal { L } _ { v }$ corresponding to a specific VFM v ∈ {DINOv2, CLIP, SAM}.

Gradient Topology. We visualized the joint distribution of gradient L2-norms for VFM pairs (Figure 2a). To compare geometric tendencies despite scale disparities, we applied independent normalization, scaling gradients of each module to [0, 1] by their respective maximums.

The shared FFN parameters (red diamonds) receive broad simultaneous updates from multiple VFMs. They are often distributed in the interior region rather than concentrated near a single axis, indicating that dense parameters are jointly affected by heterogeneous teacher signals. Such simultaneous updates create the optimization condition under which gradient averaging can occur. In contrast, the MoE experts (blue circles) exhibit an L-shaped topology. Many experts lie close to one axis, meaning that their dominant update comes from one VFM condition while the update from another is weak. For pairs involving SAM, some experts deviate from the axes due to partially shared structural cues, but they remain more separated than the dense FFN parameters. This pattern suggests that PRISM decomposes VFM knowledge into sparse expert subspaces and reduces effective cross-teacher interference.

Cosine Similarity. We further quantified interference by calculating the neuron-level cosine similarity: $\cos ( \theta ) =$ $( g _ { v _ { A } } \cdot g _ { v _ { B } } ) / ( \| g _ { v _ { A } } \| \| g _ { v _ { B } } \| + \epsilon )$ .

Figure 2b further shows that sparse experts and shared FFNs follow different interaction patterns. The shared FFN (red) has a broad distribution, reflecting the fact that the same dense parameters are updated by multiple VFMs. In contrast, the sparse experts (blue) place more mass near zero cosine similarity, indicating that routed teacher-specific updates become weakly interacting on many expert parameters. Positive modes correspond to compatible teacher signals that can be safely shared, while negative modes usually occur with highly imbalanced gradient magnitudes, where one teacher provides the dominant update. Overall, the transition from broad dense coupling to sparse, near-axis expert updates supports the intended mechanism of PRISM: reducing effective interference while preserving useful shared signals.

Table 4. Component Analysis on PASCAL-Context (ViT-S Student). We evaluate the impact of the Dual-Stream design, FiLM routing, and Decorrelation Loss. The full PRISM model achieves the best trade-off. 

<table><tr><td>Configuration</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td></tr><tr><td>(1) w/o MoE</td><td>78.04</td><td>67.55</td><td>84.97</td><td>14.31</td><td>70.16</td></tr><tr><td>(2) w/o Anchor</td><td>78.46</td><td>67.98</td><td>84.41</td><td>14.65</td><td>69.08</td></tr><tr><td>(3) w/o FiLM</td><td>78.87</td><td>69.26</td><td>84.64</td><td>14.46</td><td>70.63</td></tr><tr><td>(4) w/o  $\mathcal{L}_{decorr}$ </td><td>79.80</td><td>61.80</td><td>85.50</td><td>14.32</td><td>70.86</td></tr><tr><td>(5) PRISM (Full)</td><td>79.19</td><td>69.25</td><td>85.01</td><td>14.28</td><td>70.78</td></tr></table>

# 4.4. Ablation Study

To dissect the contribution of each component in PRISM, we conduct comprehensive ablation studies using a ViT-S student and ViT-B teachers. Stage 1 is trained on 10% of ImageNet-1k, followed by fine-tuning on PASCAL-Context.

Impact of Dual-Stream Architecture. We first validate the necessity of the hybrid design. As shown in Row 1 in Table 4, the dense baseline (w/o MoE) struggles with capacity, yielding the lowest segmentation performance at 78.04 mIoU. Critically, removing the Universal Anchor (Row 2) results in the worst surface normal estimation error (14.65) among all configurations. This specific degradation confirms that while sparse experts handle task divergence, the shared anchor is indispensable for maintaining coherent structural and geometric representations.

Table 5. Controlled Comparison on PASCAL-Context (ViT-S Student). Wide ViT-S controls for static dense capacity scaling, while vanilla MoE controls for generic sparse routing. 

<table><tr><td>Model</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td></tr><tr><td>vanilla MoE</td><td>72.96</td><td>61.77</td><td>83.34</td><td>15.03</td><td>66.73</td></tr><tr><td>Wide ViT-S</td><td>78.15</td><td>67.71</td><td>84.88</td><td>14.24</td><td>70.35</td></tr><tr><td>PRISM</td><td>79.19</td><td>69.25</td><td>85.01</td><td>14.28</td><td>70.78</td></tr></table>

Effectiveness of Context-Modulated Routing. Next, we investigate the routing mechanism. Replacing the FiLMbased router with simple task embedding concatenation (Row 3) leads to sub-optimal routing, dropping Semantic Segmentation by 0.32 mIoU and increasing normal error compared to the full model. This indicates that static task IDs are insufficient; the router requires the input-conditional feature recalibration provided by FiLM to align features with expert subspaces dynamically.

Necessity of Locality-Aware Decorrelation. The decorrelation loss is critical for balanced specialization. Removing $\mathcal { L } _ { d e c o r r }$ keeps several coarse metrics strong, but causes Human Parsing to drop sharply from 69.25 to 61.80 mIoU. Human Parsing is particularly sensitive because it requires fine-grained separation of adjacent body parts with thin and ambiguous boundaries. Additional routing analysis in Appendix C shows that $\mathcal { L } _ { d e c o r r }$ reorganizes Parsing away from geometry-dominated routing and toward more taskappropriate semantic/boundary sharing.

Beyond Capacity Scaling and Plain MoE. To isolate the source of PRISM’s improvements, we compare it with two controlled baselines that account for potential confounding factors in capacity and sparsity. First, we construct an iso-FLOPs Wide ViT-S baseline by increasing the FFN expansion ratio to 5×, matching the active computation of PRISM’s Universal Anchor, shared experts, and Top-3 routed experts. We initialize this widened model with Net2Net tiling (Chen et al., 2015) to ensure stable convergence, thereby testing whether the gains simply come from static dense capacity scaling. Second, we compare with a vanilla sparse MoE of comparable active capacity, obtained by removing PRISM-specific components, including the Universal Anchor, FiLM-conditioned routing, shared expert, and dual-stream fusion. This control examines whether the gains arise merely from introducing sparse expert routing. As shown in Table 5, PRISM consistently outperforms both baselines, suggesting that its advantage comes from conflict-aware dual-stream routing and dynamic knowledge disentanglement rather than increased dense capacity or Table 6. Cost-performance Comparison on ViT-S. PRISM-lite keeps the same PRISM design but reduces the expert MLP ratio from 4.0 to 1.0.

(a) Efficiency. 

<table><tr><td>Model</td><td>Total Params (M)</td><td>Active Params (M)</td><td>GFLOPs</td><td>Latency (ms)</td></tr><tr><td>SAK ViT-S</td><td>26.40</td><td>26.40</td><td>53.20</td><td>20.25</td></tr><tr><td>PRISM-lite</td><td>48.05</td><td>38.06</td><td>57.51</td><td>47.51</td></tr></table>

(b) Performance on PASCAL-Context. 

<table><tr><td>Model</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td><td> $\Delta_m$ % ↑</td></tr><tr><td>SAK ViT-S</td><td>78.66</td><td>68.46</td><td>84.66</td><td>14.33</td><td>70.28</td><td>0.43</td></tr><tr><td>PRISM-lite</td><td>78.97</td><td>69.60</td><td>84.71</td><td>14.39</td><td>69.73</td><td>0.61</td></tr></table>

generic MoE sparsity alone.

Cost-Performance Trade-off. PRISM introduces extra offline cost because Stage 1 requires forwarding frozen VFM teachers. However, this cost is paid only during distillation, while inference uses a single student without running multiple VFMs. To examine whether the design remains useful under a lighter setting, we evaluate PRISM-lite, which keeps the same dual-stream architecture, FiLM routing, and losses, but reduces the expert MLP ratio from 4.0 to 1.0. As shown in Table 6, PRISM-lite achieves higher overall $\Delta _ { m }$ than SAK under comparable GFLOPs, although sparse dispatch introduces higher latency.

# 4.5. Analysis of Emergent Specialization

To verify our “Decompose-then-Recombine” hypothesis and validate the existence of soft boundaries, we probe the routing dynamics of the MoE layer (Layer 11) on the validation set by visualizing the expert activation probability P (Experti|Condition) in Figure 3.

Phase 1: Emergent Decomposition. We first observe the emergence of teacher specialization and partial consensus in the routing patterns. The experts partition into distinct clusters: $E _ { 1 4 }$ acts as a semantic specialist dominated by CLIP (20.7%), while $E _ { 2 }$ serves as a geometric specialist dominated by SAM (14.2%), confirming that PRISM successfully separates conflicting supervision signals into more specialized parameter subspaces. Crucially, the router also identifies experts representing partial consensus, such as $E _ { 1 2 }$ , which is co-activated by DINOv2 and CLIP for shared texture semantics but ignored by SAM. This highlights a fundamental advantage over SAK. While SAK’s static architecture forces a rigid dichotomy between global consensus and task specificity, often leading to redundant learning of shared features, PRISM naturally identifies these soft boundaries. By consolidating shared knowledge into specific experts like $E _ { 1 2 } .$ , our method achieves greater parameter efficiency.

![](images/804a4014e881c7bbff87b0fdd9166a8ece58e79c285b6c8943789c03b2e0cfba.jpg)

<details>
<summary>heatmap</summary>

| | DINOv2 (%) | CLIP (%) | SAM (%) |
|---|---|---|---|
| E0 | 0.2 | 16.0 | 1.7 |
| E1 | 2.7 | 4.6 | 10.0 |
| E2 | 5.4 | 5.3 | 14.2 |
| E3 | 0.6 | 12.0 | 5.9 |
| E4 | 1.9 | 3.0 | 14.8 |
| E5 | 14.7 | 1.5 | 9.2 |
| E6 | 1.0 | 4.0 | 14.6 |
| E7 | 19.1 | 1.8 | 1.8 |
| E8 | 2.2 | 11.6 | 0.4 |
| E9 | 6.2 | 6.8 | 3.6 |
| E10 | 9.0 | 5.1 | 9.3 |
| E11 | 1.7 | 7.7 | 6.2 |
| E12 | 18.8 | 0.0 | 5.9 |
| E13 | 16.4 | 0.1 | 2.3 |
| E14 | 0.1 | 20.7 | 0.0 |
</details>

(a) VFM Conditional Affinity (Stage 1)

![](images/162c2c44e0aa062fc243662084bf8c963b671bfefbdfe0b58df6c53da3e60b8e.jpg)

<details>
<summary>heatmap</summary>

| Category | E0 (%) | E1 (%) | E2 (%) | E3 (%) | E4 (%) | E5 (%) | E6 (%) | E7 (%) | E8 (%) | E9 (%) | E10 (%) | E11 (%) | E12 (%) | E13 (%) | E14 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Semseg | 5.2% | 5.8% | 7.3% | 7.3% | 7.1% | 5.3% | 5.6% | 9.3% | 4.3% | 6.1% | 7.4% | 5.4% | 9.0% | 7.8% | 7.2% |
| Parsing | 7.7% | 3.8% | 6.5% | 5.7% | 7.2% | 6.1% | 6.1% | 9.1% | 3.6% | 4.5% | 11.5% | 4.0% | 10.9% | 6.1% | 7.2% |
| Saliency | 6.8% | 5.0% | 8.0% | 6.9% | 6.4% | 8.5% | 7.0% | 10.2% | 3.8% | 5.3% | 6.8% | 7.7% | 6.0% | 6.5% | 5.1% |
| Normal | 6.1% | 4.0% | 7.6% | 9.6% | 6.2% | 9.0% | 7.0% | 12.3% | 3.4% | 3.9% | 6.6% | 9.6% | 5.0% | 6.0% | 3.8% |
| Boundary | 5.8% | 5.4% | 7.7% | 5.6% | 6.4% | 5.7% | 5.8% | 10.2% | 4.8% | 6.8% | 6.8% | 7.6% | 8.6% | 7.6% | 5.2% |
</details>

(b) Task Conditional Affinity (Stage 2)   
Figure 3. Visualization of Emergent Knowledge Decomposition and Recombination at Layer 11. (a) When conditioned on Teachers, experts show clear specialization (e.g., $E _ { 1 4 }$ for CLIP, E7 for DINOv2), supporting effective conflict reduction through expert specialization. (b) When conditioned on Tasks, the router recombines these experts. For example, SemSeg (Semantic) utilizes both DINO-experts and CLIP-experts, whereas Normal (Geometric) ignores CLIP-experts $( E _ { 1 4 } )$ in favor of DINO-experts $( E _ { 7 } )$ .

Phase 2: Knowledge Recombination. In the second stage, we analyze how these specialized primitives are utilized for downstream tasks. Unlike the sparse activation seen during teacher training, tasks exhibit composite routing patterns that recombine knowledge. Semantic Segmentation recruits a hybrid coalition, activating both the partial consensus expert $E _ { 1 2 } ( 9 . 0 \% )$ for spatial correspondence and the semantic expert $E _ { 1 4 }$ (7.2%) for categorical reasoning. Conversely, Surface Normal Estimation heavily relies on texture-aware experts while significantly reducing reliance on the CLIPspecific $E _ { 1 4 }$ to 3.8%. This validates that PRISM moves beyond rigid partitioning to decompose knowledge into granular primitives, which are then dynamically recombined based on task needs.

Learnable Stability-Plasticity Trade-off. Finally, we track the trajectory of the learnable gate λ across layers to understand the adaptive balance between shared and specialized features. In shallow layers such as Layer 2, λ converges to approximately 0.7, retaining high reliance on the Universal Anchor for stability. In deeper layers such as layers 5, 8 and 11, λ shifts to approximately 0.5, indicating a greater dependence on Specialized Experts. This trend aligns with the intuition that conflicting semantic concepts primarily arise in deeper representations, necessitating stronger expert

intervention to resolve interference.

# 5. Conclusion

In this work, we propose PRISM, a dynamic framework that addresses the optimization paradox in multi-teacher distillation through a “Decompose-then-Recombine” paradigm. By introducing a Dual-Stream Conditioned MoE, our architecture reduces effective cross-teacher interference through sparse expert specialization while preserving shared consensus via a Universal Anchor. Extensive experiments demonstrate that this emergent specialization leads to state-of-theart performance across dense prediction benchmarks, effectively harmonizing the distinct strengths of heterogeneous Vision Foundation Models. Diagnostic analyses further show that context-modulated routing captures soft boundaries between knowledge domains and supports dynamic knowledge recombination. One remaining trade-off lies in balancing semantic abstraction and geometric locality: PRISM learns a shared pool of specialized experts for heterogeneous VFM knowledge, but the optimal allocation between high-level semantic cues and local geometric cues can vary across datasets and tasks. Future work may explore task-adaptive routing or lightweight local refinements to further tune this balance.

# Acknowledgements

This work was supported by the National Natural Science Foundation of China (Nos. 62272184 and 62402189), the China Postdoctoral Science Foundation (Nos. 2024M751012, 2025T180429, and GZC20230894), the Postdoctor Project of Hubei Province (No. 2024HBB-HCXB014), the Natural Science Foundation of Hubei Province (No. JCZRMS202600758), and the CIPS-SMP-Zhipu Large Model Fund (No. CIPS-SMP20250306). The computational work was performed on the highperformance computing platform at Huazhong University of Science and Technology.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Bruggemann, D., Kanakis, M., Obukhov, A., Georgoulis, S., ¨ and Van Gool, L. Exploring relational context for multitask dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 15869– 15878, 2021.   
Caruana, R. Multitask learning. Machine learning, 28(1): 41–75, 1997.   
Chen, T., Goodfellow, I., and Shlens, J. Net2net: Accelerating learning via knowledge transfer. arXiv preprint arXiv:1511.05641, 2015.   
Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich, A. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning, pp. 794–803. PMLR, 2018.   
Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., and Anguelov, D. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems, 33: 2039–2050, 2020.   
Chen, Z., Shen, Y., Ding, M., Chen, Z., Zhao, H., Learned-Miller, E. G., and Gan, C. Mod-squad: Designing mixtures of experts as modular multi-task learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11828–11837, 2023.   
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database.

In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.   
Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022. URL https://jmlr. org/papers/v23/21-0998.html.   
Fukuda, T., Suzuki, M., Kurata, G., Thomas, S., Cui, J., and Ramabhadran, B. Efficient knowledge distillation from an ensemble of teachers. In Interspeech 2017, pp. 3697–3701, 2017. doi: 10.21437/Interspeech.2017-614.   
Guo, M., Haque, A., Huang, D.-A., Yeung, S., and Fei-Fei, L. Dynamic task prioritization for multitask learning. In Proceedings of the European conference on computer vision (ECCV), pp. 270–287, 2018.   
Heinrich, G., Ranzinger, M., Yin, H., Lu, Y., Kautz, J., Tao, A., Catanzaro, B., and Molchanov, P. Radiov2. 5: Improved baselines for agglomerative vision foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22487–22497, 2025.   
Huang, H., Huang, Y., Lin, L., Tong, R., Chen, Y.-W., Zheng, H., Li, Y., and Zheng, Y. Going beyond multi-task dense prediction with synergy embedding models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28181–28190, 2024.   
Ishihara, K., Kanervisto, A., Miura, J., and Hautamaki, V. Multi-task learning with attention for end-to-end autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2902–2911, 2021.   
Kanakis, M., Bruggemann, D., Saha, S., Georgoulis, S., Obukhov, A., and Van Gool, L. Reparameterizing convolutions for incremental multi-task learning without task interference. In European Conference on Computer Vision (ECCV), 2020.   
Kendall, A., Gal, Y., and Cipolla, R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7482–7491, 2018.   
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023.   
Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Li, W., Zhou, H., Yu, J., Song, Z., and Yang, W. Coupled mamba: Enhanced multimodal fusion with coupled state space model. Advances in Neural Information Processing Systems, 37:59808–59832, 2024a.   
Li, W., Song, Z., Zhou, H., Zhang, Y., Yu, J., and Yang, W. Lora-mixer: Coordinate modular lora experts through serial attention routing. arXiv preprint arXiv:2507.00029, 2025.   
Li, W., Song, Z., Zhang, J., Zhao, T., Lin, J., Wang, Y., and Yang, W. Large language model as token compressor and decompressor, 2026.   
Li, W.-H., Liu, X., and Bilen, H. Universal representations: A unified look at multiple task and domain learning. International Journal of Computer Vision, 132(5):1521–1545, 2024b.   
Liu, B., Liu, X., Jin, X., Stone, P., and Liu, Q. Conflictaverse gradient descent for multi-task learning. Advances in Neural Information Processing Systems, 34:18878– 18890, 2021a.   
Liu, L., Li, Y., Kuang, Z., Xue, J.-H., Chen, Y., Yang, W., Liao, Q., and Zhang, W. Towards impartial multitask learning. In International conference on learning representations, 2021b.   
Liu, S., Johns, E., and Davison, A. J. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1871–1880, 2019.   
Liu, Y., Zhang, W., and Wang, J. Adaptive multi-teacher multi-level knowledge distillation. Neurocomputing, 415: 106–113, 2020. doi: 10.1016/j.neucom.2020.07.048.   
Lu, Y., Cao, S., and Wang, Y.-X. Swiss army knife: Synergizing biases in knowledge from vision foundation models for multi-task learning. In The Thirteenth International Conference on Learning Representations, 2025.   
Maninis, K.-K., Radosavovic, I., and Kokkinos, I. Attentive single-tasking of multiple tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.   
Mottaghi, R., Chen, X., Liu, X., Cho, N.-G., Lee, S.-W., Fidler, S., Urtasun, R., and Yuille, A. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 891–898, 2014.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.   
Ranzinger, M., Barker, J., Heinrich, G., Molchanov, P., Catanzaro, B., and Tao, A. Phi-s: Distribution balancing for label-free multi-teacher distillation. arXiv preprint arXiv:2410.01680, 2024a.   
Ranzinger, M., Heinrich, G., Kautz, J., and Molchanov, P. Am-radio: Agglomerative vision foundation model reduce all domains into one. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12490–12500, 2024b.   
Ruder, S. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.   
Sariyildiz, M. B., Weinzaepfel, P., Lucas, T., Larlus, D., and Kalantidis, Y. Unic: Universal classification models via multi-teacher distillation. arXiv preprint arXiv:2408.05088, 2024.   
Sarıyıldız, M. B., Weinzaepfel, P., Lucas, T., de Jorge, P., Larlus, D., and Kalantidis, Y. Dune: Distilling a universal encoder from heterogeneous 2d and 3d teachers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 30084–30094, 2025.   
Shang, J., Schmeckpeper, K., May, B. B., Minniti, M. V., Kelestemur, T., Watkins, D., and Herlant, L. Theia: Distilling diverse vision foundation models for robot learning. arXiv preprint arXiv:2407.20179, 2024.   
Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum? id=B1ckMDqlg.   
Shi, M., Liu, F., Wang, S., Liao, S., Radhakrishnan, S., Zhao, Y., Huang, D.-A., Yin, H., Sapra, K., Yacoob, Y., et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024.   
Shoouri, S., Yang, M., Fan, Z., and Kim, H.-S. Efficient computation sharing for multi-task visual scene understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17130–17141, 2023.   
Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pp. 746–760. Springer, 2012.

Song, Z., Yu, J., Chen, Y.-P. P., and Yang, W. Transformer tracking with cyclic shifting window attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8791–8800, 2022.   
Song, Z., Luo, R., Yu, J., Chen, Y.-P. P., and Yang, W. Compact transformer tracker with correlative masked modeling. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 2321–2329, 2023.   
Song, Z., Tang, Y., Luo, R., Ma, L., Yu, J., Chen, Y.-P. P., and Yang, W. Autogenic language embedding for coherent point tracking. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 2021–2030, 2024.   
Song, Z., Luo, R., Ma, L., Tang, Y., Chen, Y.-P. P., Yu, J., and Yang, W. Temporal coherent object flow for multiobject tracking. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 6978–6986, 2025.   
Song, Z., Yu, J., Chen, Y.-P. P., Yang, W., and Wang, X. Hypergraph-state collaborative reasoning for multi-object tracking, 2026.   
Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., and Van Gool, L. Multi-task learning for dense prediction tasks: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(7):3614– 3633, 2021.   
Wang, D., Zhang, Y., Yu, J., Chen, Y.-P. P., Xu, C., and Song, Z. Seeing further and wider: Joint spatio-temporal enlargement for micro-video popularity prediction. arXiv preprint arXiv:2604.20311, 2026.   
Wang, H., Vasu, P. K. A., Faghri, F., Vemulapalli, R., Farajtabar, M., Mehta, S., Rastegari, M., Tuzel, O., and Pouransari, H. Sam-clip: Merging vision foundation models towards semantic and spatial understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3635–3647, 2024a.   
Wang, S., Li, J., Zhao, Z., Lian, D., Huang, B., Wang, X., Li, Z., and Gao, S. Tsp-transformer: Task-specific prompts boosted transformer for holistic scene understanding. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 925–934, 2024b.   
Wang, X., Tang, C., Yue, X., and Li, W.-H. 3d-aware multitask learning with cross-view correlations for dense scene understanding. arXiv preprint arXiv:2511.20646, 2025.   
Xu, Y., Yang, Y., and Zhang, L. Multi-task learning with knowledge distillation for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21550–21559, 2023.

Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., and Zhao, H. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10371–10381, 2024a.   
Yang, Y., Jiang, P.-T., Hou, Q., Zhang, H., Chen, J., and Li, B. Multi-task dense prediction via mixture of low-rank experts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 27927– 27937, 2024b.   
Ye, H. and Xu, D. Inverted pyramid multi-task transformer for dense scene understanding. In European Conference on Computer Vision, pp. 514–530. Springer, 2022a.   
Ye, H. and Xu, D. Taskprompter: Spatial-channel multitask prompting for dense scene understanding. In The Eleventh International Conference on Learning Representations, 2022b.   
Ye, H. and Xu, D. Taskexpert: Dynamically assembling multi-task representations with memorial mixture-ofexperts. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21828–21837, 2023.   
Ye, H. and Xu, D. Invpt++: Inverted pyramid multi-task transformer for visual scene understanding. IEEE transactions on pattern analysis and machine intelligence, 46 (12):7493–7508, 2024.   
Ye, L., Zhang, Y., Wu, Y., Chen, Y.-P. P., Yu, J., Yang, W., and Song, Z. Mvp: Winning solution to smp challenge 2025 video track. arXiv preprint arXiv:2507.00950, 2025.   
You, S., Xu, C., Xu, C., and Tao, D. Learning from multiple teacher networks. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1285–1294, 2017. doi: 10.1145/3097983.3098135.   
Yu, J., Dai, Y., Liu, X., Huang, J., Shen, Y., Zhang, K., Zhou, R., Adhikarla, E., Ye, W., Liu, Y., et al. Unleashing the power of multi-task learning: A comprehensive survey spanning traditional, deep, and pretrained foundation model eras. arXiv preprint arXiv:2404.18961, 2024.   
Zeng, G., Yu, J., Chen, Y.-P. P., Chen, X., Yang, W., and Song, Z. Curevo: Curriculum-guided self-evolution for video understanding. arXiv preprint arXiv:2604.26707, 2026.   
Zhang, J., Fan, J., Ye, P., Zhang, B., Ye, H., Li, B., Cai, Y., and Chen, T. Bridgenet: Comprehensive and effective feature interactions via bridge feature for multi-task dense predictions. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(5):3657–3672, 2025.

Zhang, R., Luo, Y., Liu, J., Yang, H., Dong, Z., Gudovskiy, D., Okuno, T., Nakata, Y., Keutzer, K., Du, Y., and Zhang, S. Efficient deweather mixture-of-experts with uncertainty-aware feature-wise linear modulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 16812–16820, 2024. doi: 10.1609/aaai.v38i15.29622.   
Zhang, Y. and Yang, Q. A survey on multi-task learning. IEEE transactions on knowledge and data engineering, 34(12):5586–5609, 2021.   
Zhang, Y., Zhang, X., Sheng, J., Li, W., Yu, J., Chen, Y.-P. P., Yang, W., and Song, Z. Semantic-aware logical reasoning via a semiotic framework, 2026.   
Zhou, J., Zhang, H., Yuan, J., Ye, P., Chen, T., Jiang, H., Chen, M., and Zhang, Y. All-in-one: Transferring vision foundation models into stereo matching. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 10797–10805, 2025.   
Zong, Z., Ma, B., Shen, D., Song, G., Shao, H., Jiang, D., Li, H., and Liu, Y. Mova: Adapting mixture of vision experts to multimodal context. Advances in Neural Information Processing Systems, 37:103305–103333, 2024.

# A. Effective Cross-Teacher Interaction

In the main paper, we use gradient topology and cosine-similarity distributions to show that PRISM reduces effective cross-teacher interference. Here we provide a more detailed pair-by-layer-by-checkpoint analysis. This analysis directly measures whether two teacher conditions update the same sparse experts, how much their token-level routing overlaps, and whether the remaining co-active experts receive aligned or conflicting gradients.

For teacher contexts a and b, we measure the effective sparse-path interaction at layer l as

$$
I _ {l} ^ {(a, b)} = \mathbb {E} \left[ \mathbf {1} (E _ {l} ^ {a} \cap E _ {l} ^ {b} \neq \emptyset) \cdot \frac {1}{| E _ {l} ^ {a} \cap E _ {l} ^ {b} |} \sum_ {e \in E _ {l} ^ {a} \cap E _ {l} ^ {b}} \cos (g _ {l, e} ^ {a}, g _ {l, e} ^ {b}) \right]. \tag {10}
$$

This quantity is zero when no sparse expert is co-activated, and otherwise measures the residual gradient interaction on the actually shared sparse parameters.

Co-active experts across checkpoints   
![](images/52aa5d6c138a3ea906eacf36a2aba30b2348da994ab470f64af9492ad75d06ee.jpg)  
Figure 4. Co-active sparse experts across checkpoints. Each cell shows the number of experts that are activated under both teacher conditions for a teacher pair at a given MoE layer. Co-activation drops sharply during training: pair/layer cases with zero co-active experts increase from 4/12 at the early checkpoint to 7/12 by mid/late/final. Earlier layers become exactly separated more often, while deeper layers retain limited overlap, most notably DINOv2–SAM at Layer 11.

Figure 4 first shows the number of co-active experts across checkpoints. Co-activation drops sharply during training: the number of pair/layer cases with zero co-active sparse experts increases from 4/12 at the early checkpoint to 7/12 by the mid, late, and final checkpoints. This supports the intended sparse-path separation mechanism. Earlier layers become exactly separated more often, while deeper layers preserve limited overlap, most notably DINOv2–SAM at Layer 11, which is consistent with their partially shared structural cues.

Figure 5 provides a representative DINOv2–CLIP case. The top panel reports the mean cosine similarity on co-active sparse experts, while the bottom panel reports the cosine similarity on the shared branch. In Layer 5, the sparse-expert cosine decreases from 0.235 at the early checkpoint to 0.189 at the middle checkpoint, and then becomes negative at late/final checkpoints $( - 0 . 1 6 7 / - 0 . 1 6 6 )$ . Cross markers indicate checkpoints where no co-active sparse experts exist, i.e., the effective sparse interaction is zero.

dinov2-clip across layers   
![](images/adf2c1d7ba4c9fe6e9015a9960f006424503d16e41d640c183250a56d10b73af.jpg)

<details>
<summary>line</summary>

| Group | L2    | L5    | L8    | L11   |
|-------|-------|-------|-------|-------|
| no co-active sparse experts | 0.23  | 0.19  | 0.02  | 0.02  |
| no co-active sparse experts | -0.27 | -0.18 | 0.00  | 0.01  |
| no co-active sparse experts | -0.27 | -0.18 | 0.01  | 0.02  |
</details>

![](images/cadee51ec4feeeaeda1d81c43874fe9cb2ed33b8c351eab1c7d527e7f5b171c0.jpg)

<details>
<summary>line</summary>

| Checkpoint | L2    | L5    | L8    | L11   |
| ---------- | ----- | ----- | ----- | ----- |
| early      | 0.0   | 0.07  | 0.08  | 0.3   |
| mid        | -0.05 | -0.12 | 0.03  | -0.08 |
| late       | -0.08 | -0.25 | 0.0   | -0.05 |
| final      | -0.1  | -0.28 | 0.02  | -0.06 |
</details>

Figure 5. Representative gradient-interaction trend for DINOv2–CLIP across checkpoints from the same training run. Top: mean cosine similarity of gradients received by co-active sparse experts at different checkpoints. Bottom: cosine similarity on the shared branch. Cross markers denote checkpoints where no co-active sparse experts exist, i.e., effective sparse interaction is zero in practice. A representative example is Layer 5, where the sparse-expert cosine changes from 0.235 at early to 0.189 at mid, then becomes negative at late/final $( - 0 . 1 6 7 / \dot { - } 0 . 1 6 6 \dot { ) }$ , indicating substantially reduced cross-teacher interference.

Figure 6 gives the complete characterization over all three teacher pairs, four MoE layers, and four checkpoints. PRISM reduces sparse-path interference through two complementary regimes. In many cases, routing overlap and co-activation decrease directly. In cases where overlap persists, such as some deeper DINOv2–SAM settings, the residual sparse-expert gradient cosine remains close to zero. Meanwhile, the shared expert retains small but nonzero compatible gradients, indicating structured decomposition rather than indiscriminate suppression of all shared information.   
![](images/f5e9b41313fce60affba3b8fd5deeabf4fda702162e938446c3223957d58825f.jpg)

<details>
<summary>heatmap</summary>

| Teacher pair | layer | early | mid | late | final |
|---|---|---|---|---|---|
| L2 | clip-sam | 0.0 | 0.0 | 0.0 | 0.0 |
| L2 | dinov2-clip | 6.7 | 0.0 | 0.0 | 0.0 |
| L2 | dinov2-sam | 0.0 | 0.0 | 0.0 | 0.0 |
| L5 | clip-sam | 0.0 | 0.0 | 0.0 | 0.0 |
| L5 | dinov2-clip | 8.9 | 5.4 | 2.0 | 2.0 |
| L5 | dinov2-sam | 0.0 | 0.0 | 0.0 | 0.0 |
| L8 | clip-sam | 2.2 | 1.0 | 1.0 | 0.7 |
| L8 | dinov2-clip | 11.2 | 4.0 | 4.0 | 4.0 |
| L8 | dinov2-sam | 2.5 | 0.0 | 0.0 | 0.0 |
| L11 | clip-sam | 5.5 | 4.2 | 3.7 | 3.6 |
| L11 | dinov2-clip | 8.3 | 4.5 | 3.7 | 3.6 |
| L11 | dinov2-sam | 11.0 | 11.0 | 11.0 | 11.0 |
</details>

(a) Co-active sparse experts across checkpoints

![](images/6408529a1e58b55df972f1ccc90a23faa76d1b7a43042d1370c07e71b50bca79.jpg)

<details>
<summary>heatmap</summary>

| Checkpoint | early | mid | late | final |
|---|---|---|---|---|
| Row 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| Row 2 | 0.349 | 0.000 | 0.000 | 0.000 |
| Row 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| Row 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| Row 5 | 0.520 | 0.089 | 0.011 | 0.018 |
| Row 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| Row 7 | 0.001 | 0.002 | 0.001 | 0.001 |
| Row 8 | 0.571 | 0.020 | 0.021 | 0.024 |
| Row 9 | 0.002 | 0.000 | 0.000 | 0.000 |
| Row 10 | 0.035 | 0.033 | 0.032 | 0.031 |
| Row 11 | 0.274 | 0.082 | 0.074 | 0.073 |
The values in the table represent the decimal values for each checkpoint and time period (e.g., 'early' or 'mid'). The color scale on the right indicates the magnitude of the value at each checkpoint and time period.
</details>

(b)Token overlap across checkpoints

![](images/97b0e3bea3e017103558da31ebfe5d91a50c6d65653fefc242a16902ed690ff2.jpg)

<details>
<summary>heatmap</summary>

| Teacher pair | layer | early | mid | late | final |
|---|---|---|---|---|---|
| L2 | clip-sam | 0.000 | 0.000 | 0.000 | 0.000 |
| L2 | dinov2-clip | -0.044 | 0.000 | 0.000 | 0.000 |
| L2 | dinov2-sam | 0.000 | 0.000 | 0.000 | 0.000 |
| L5 | clip-sam | 0.000 | 0.000 | 0.000 | 0.000 |
| L5 | dinov2-clip | 0.107 | 0.012 | 0.011 | 0.002 |
| L5 | dinov2-sam | 0.000 | 0.000 | 0.000 | 0.000 |
| L8 | clip-sam | -0.011 | 0.000 | 0.000 | 0.000 |
| L8 | dinov2-clip | 0.002 | 0.009 | 0.004 | 0.005 |
| L8 | dinov2-sam | -0.001 | 0.000 | 0.000 | 0.000 |
| L11 | clip-sam | 0.003 | 0.006 | 0.000 | -0.002 |
| L11 | dinov2-clip | -0.012 | -0.001 | 0.004 | 0.003 |
| L11 | dinov2-sam | -0.004 | -0.002 | -0.000 | 0.000 |
</details>

(c) Sparse-expert gradient cosine

![](images/72eacc08fb712f49ab850cab8ce61cb664f3e9a528eda1144da6875effa5e708.jpg)

<details>
<summary>heatmap</summary>

| Checkpoint | early | mid | late | final |
|---|---|---|---|---|
| Row 1 | 0.072 | 0.084 | 0.116 | 0.127 |
| Row 2 | 0.014 | 0.191 | 0.158 | 0.174 |
| Row 3 | -0.129 | -0.026 | 0.030 | 0.029 |
| Row 4 | 0.013 | 0.018 | 0.008 | 0.011 |
| Row 5 | 0.027 | 0.054 | -0.001 | 0.049 |
| Row 6 | 0.055 | 0.021 | 0.028 | 0.021 |
| Row 7 | -0.015 | -0.031 | -0.021 | -0.024 |
| Row 8 | 0.040 | 0.056 | 0.082 | 0.070 |
| Row 9 | -0.024 | -0.000 | -0.010 | -0.005 |
| Row 10 | -0.085 | -0.012 | -0.003 | -0.002 |
| Row 11 | 0.015 | 0.007 | 0.004 | -0.003 |
| Row 12 | 0.036 | 0.006 | 0.023 | 0.023 |
</details>

(d) Shared-expert gradient cosine   
Figure 6. Pair × layer × checkpoint characterization of cross-teacher interaction. We show (a) co-active sparse experts, (b) token overlap, (c) gradient cosine on the actually co-active sparse experts, and (d) gradient cosine on the shared expert. PRISM reduces sparse-path interference either by decreasing routing overlap/co-activation or, when overlap persists, by driving the residual sparse-expert gradient cosine near zero. The shared expert retains small but nonzero compatible gradients, indicating structured decomposition rather than global suppression.

# B. Stability Across Random Seeds

We next examine whether expert specialization is a stable phenomenon or a seed-specific artifact. Since MoE expert indices are permutation-invariant, we align non-reference seeds to the first seed with Hungarian matching before comparison. Figure 7 shows that the same qualitative teacher–expert affinity structure consistently re-emerges across random seeds.

In early and middle layers, teacher-preferred partitions are sharp: different teachers activate different expert groups after alignment. In the deepest layer, routing becomes more mixed and shared, which is expected because high-level representations contain more overlapping semantic and structural information. This seed-level stability supports that the observed specialization is a robust outcome of the PRISM training objective and architecture.

Teacher-Expert Affinity Across Seeds   
![](images/1713958458ca185646054172c7fd64be5e361d4ce885b133bba9e548d0d296ab.jpg)  
Figure 7. Teacher–expert affinity across random seeds during the decompose stage. Each heatmap shows the routing-frequency affinity matrix for one MoE layer. Non-reference seeds are aligned to the first seed using Hungarian matching. The top axis shows aligned comparison slots $S _ { k } ,$ while the bottom axis shows the original expert IDs $E _ { j }$ in each seed. Cells with zero or negligible affinity are left blank for readability. Earlier and middle layers exhibit sharper teacher-preferred partitioning, whereas the deepest layer is more mixed/shared; importantly, the same qualitative specialization structure consistently re-emerges across seeds up to permutation.

# C. Effect of Locality-Aware Decorrelation on Human Parsing

The component ablation in the main paper shows that removing $\mathcal { L } _ { d e c o r r }$ causes a large drop in Human Parsing. This section provides additional evidence explaining why this task benefits strongly from locality-aware decorrelation. Human Parsing requires fine-grained separation among adjacent body parts, where local boundaries are thin and often ambiguous. Therefore, a routing pattern dominated by coarse geometry can hurt part-level discrimination. We conduct this analysis as an additional diagnostic evaluation to localize the effect of $\mathcal { L } _ { d e c o r r } \mathrm { : }$ while its absolute mIoU values may differ from the main ablation table due to the diagnostic protocol, all comparisons within this section use the same protocol.

![](images/1c293176bea9905bf00fc92a11e0d0e70012735c69c460b28bd5149cb2fbfe8d.jpg)

<details>
<summary>natural_image</summary>

Collage of three-panel image showing a cyclist in different poses: seated, swimming, and walking (no text or symbols present)
</details>

Figure 8. Qualitative comparison on Human Parsing. We show three representative examples for illustration. Compared with the variant without $\mathcal { L } _ { d e c o r r } ,$ adding $\mathcal { L } _ { d e c o r r }$ generally yields more coherent part layouts and reduces fragmented predictions, especially around lower-body regions and articulated limbs. This is consistent with the improvements in boundary-band evaluation and the largest per-class gains on limb categories.

Figure 8 shows representative qualitative examples. With $\mathcal { L } _ { d e c o r r } ,$ , predictions become more coherent and less fragmented, especially around lower-body regions and articulated limbs. This qualitative trend matches the quantitative improvements in Tables 7, 8, and 9.

![](images/c93690a438e3c586169972cefc139c67c61d6e96f2da4a4d2eb721010ea97786.jpg)

<details>
<summary>heatmap</summary>

| Compared task | 2 | 5 | 8 | 11 |
|---|---|---|---|---|
| Semseg | 0.000 | 0.000 | 0.000 | 0.162 |
| Saliency | 0.000 | 0.102 | 0.371 | 0.240 |
| Normal | 1.000 | 1.000 | 1.000 | 0.827 |
| Boundary | 0.000 | 0.000 | 0.000 | 0.159 |
</details>

(a) Routing overlap w/o Ldecorr

![](images/648f94b0f54ff2385ac1f7819cce7fa204dee83b84d847c5a9e11a57cd5f3d41.jpg)

<details>
<summary>heatmap</summary>

| | 2 | 5 | 8 | 11 |
|---|---|---|---|---|
| Semseg | 0.828 | 0.781 | 0.803 | 0.685 |
| Saliency | 0.171 | 0.384 | 0.550 | 0.565 |
| Normal | 0.230 | 0.552 | 0.554 | 0.579 |
| Boundary | 0.819 | 0.755 | 0.783 | 0.805 |
</details>

(b) Routing overlap w/ Ldecorr

![](images/8e8a9a87b6e3db4509d72baa5fb83a7a93e22c620112943610e6102b18b3b257.jpg)

<details>
<summary>heatmap</summary>

| Category | E0 (%) | E1 (%) | E2 (%) | E3 (%) | E4 (%) | E5 (%) | E6 (%) | E7 (%) | E8 (%) | E9 (%) | E10 (%) | E11 (%) | E12 (%) | E13 (%) | E14 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Semseg - | 5.0% | 9.3% | 3.1% | 25.2% | 0.0% | 6.6% | 8.3% | 20.7% | 0.0% | 4.0% | 0.0% | 8.2% | 4.3% | 0.0% | 5.3% |
| Parsing - | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 33.3% | 0.0% | 33.3% | 0.0% | 0.0% | 33.3% | 0.0% |
| Saliency - | 17.3% | 9.5% | 13.8% | 0.4% | 0.0% | 4.3% | 3.1% | 0.6% | 0.0% | 25.1% | 0.0% | 24.6% | 0.1% | 0.0% | 0.9% |
| Normal - | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 33.3% | 0.0% | 33.3% | 0.0% | 0.0% | 33.3% | 0.0% |
| Boundary - | 5.0% | 7.1% | 4.1% | 27.2% | 1.5% | 6.5% | 3.1% | 23.3% | 0.0% | 2.0% | 0.0% | 4.9% | 12.1% | 0.0% | 3.1% |
</details>

(c) L2 task affinity (w/o $L _ { d e c o r r } )$

![](images/22b2bf2c25124922556e6e52aacf711392d5ca79528e8b2ebaeb5f812a2d3961.jpg)

<details>
<summary>heatmap</summary>

| Category | E0 (%) | E1 (%) | E2 (%) | E3 (%) | E4 (%) | E5 (%) | E6 (%) | E7 (%) | E8 (%) | E9 (%) | E10 (%) | E11 (%) | E12 (%) | E13 (%) | E14 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Semseg - | 0.0% | 0.5% | 0.0% | 7.1% | 8.7% | 6.6% | 0.3% | 22.1% | 7.3% | 6.7% | 0.2% | 0.0% | 12.9% | 0.0% | 27.5% |
| Parsing - | 0.0% | 2.7% | 0.0% | 3.8% | 8.2% | 5.7% | 0.0% | 23.9% | 12.2% | 3.0% | 0.0% | 0.0% | 11.1% | 0.1% | 29.2% |
| Saliency - | 0.0% | 29.1% | 0.0% | 6.5% | 20.4% | 3.7% | 1.4% | 0.0% | 12.1% | 11.4% | 15.3% | 0.0% | 0.1% | 0.0% | 0.0% |
| Normal - | 0.0% | 4.7% | 0.8% | 2.3% | 10.6% | 24.1% | 0.2% | 0.3% | 33.3% | 7.6% | 9.5% | 0.0% | 0.0% | 4.1% | 2.4% |
| Boundary - | 0.0% | 0.1% | 0.0% | 3.1% | 6.7% | 2.2% | 0.0% | 25.4% | 11.4% | 3.3% | 0.0% | 0.2% | 17.2% | 0.5% | 30.0% |
</details>

(d) L2 task affinity (w/ Ldecorr)   
Figure 9. Why Human Parsing benefits more from $\mathcal { L } _ { d e c o r r } .$ (a,b) Parsing-condition routing overlap across MoE layers. Each cell shows the token-level routing Jaccard overlap between Human Parsing and another downstream task under the corresponding task condition. Without $\mathcal { L } _ { d e c o r r } ,$ Parsing is almost fully aligned with Normal in early/mid layers, while its overlap with SemSeg and Boundary is nearly zero until Layer 11. With $\mathcal { L } _ { d e c o r r } .$ , this pattern changes substantially: Parsing–Normal overlap decreases, while overlap with SemSeg and Boundary increases markedly. (c,d) Layer-2 task affinity, i.e., expert usage distribution. Without $\mathcal { L } _ { d e c o r r } .$ , Parsing and Normal concentrate on nearly the same sparse experts; with $\mathcal { L } _ { d e c o r r }$ , Parsing redistributes toward experts that are also used by SemSeg/Boundary. Together, these results suggest that $\mathcal { L } _ { d e c o r r }$ reorganizes Parsing away from an overly geometry-dominated routing pattern toward a more task-appropriate combination of semantic and boundary-sensitive sharing.

Figure 9 further explains the routing mechanism behind this gain. Without $\mathcal { L } _ { d e c o r r } ,$ Human Parsing is almost fully aligned with Normal in early and middle layers, and both tasks concentrate on nearly the same sparse experts. With $\mathcal { L } _ { d e c o r r } ,$ Parsing–Normal overlap decreases substantially, while overlap with SemSeg and Boundary increases. This suggests that $\mathcal { L } _ { d e c o r r }$ does not merely enforce stronger global orthogonality; it reorganizes Parsing toward a task-appropriate mixture of semantic and boundary-sensitive sharing.

Table 7 reports overall Human Parsing and boundary-band metrics. The gain from $\mathcal { L } _ { d e c o r r }$ is not limited to global mIoU; it remains clear near part boundaries, where boundary-band mIoU improves by +5.59 and foreground-only boundary-band mIoU improves by +5.86.

Table 7. Overall Human Parsing analysis. Boundary-band metrics are computed near ground-truth part boundaries. 

<table><tr><td>Metric</td><td>w/o  $\mathcal{L}_{\text{decorr}}$ </td><td>w/  $\mathcal{L}_{\text{decorr}}$ </td><td> $\Delta$ </td></tr><tr><td>Human Parsing mIoU (all)</td><td>63.69</td><td>71.18</td><td>+7.48</td></tr><tr><td>Human Parsing mIoU (fg only)</td><td>58.54</td><td>67.08</td><td>+8.54</td></tr><tr><td>Boundary-band mIoU (all)</td><td>42.26</td><td>47.85</td><td>+5.59</td></tr><tr><td>Boundary-band mIoU (fg only)</td><td>40.74</td><td>46.60</td><td>+5.86</td></tr><tr><td>Boundary-band pixel acc.</td><td>62.59</td><td>66.99</td><td>+4.40</td></tr></table>

The class-wise results in Table 8 show that the largest gains occur on fine-grained limb categories, such as upper arm, lower arm/hand, and lower leg/foot. These are precisely the categories where neighboring body parts are semantically similar and separated by thin local boundaries.

Table 8. Per-class IoU on Human Parsing. 

<table><tr><td>Class</td><td>w/o  $\mathcal{L}_{\text{decorr}}$ </td><td>w/  $\mathcal{L}_{\text{decorr}}$ </td><td> $\Delta$ </td></tr><tr><td>Background</td><td>94.58</td><td>95.72</td><td>+1.14</td></tr><tr><td>Head</td><td>86.31</td><td>90.43</td><td>+4.12</td></tr><tr><td>Torso</td><td>71.88</td><td>77.92</td><td>+6.04</td></tr><tr><td>Upper arm</td><td>48.97</td><td>62.24</td><td>+13.28</td></tr><tr><td>Lower arm / hand</td><td>53.10</td><td>63.53</td><td>+10.43</td></tr><tr><td>Upper leg</td><td>50.91</td><td>57.59</td><td>+6.67</td></tr><tr><td>Lower leg / foot</td><td>40.09</td><td>50.80</td><td>+10.71</td></tr></table>

Table 9 confirms the same trend under boundary-band evaluation. The strongest boundary-region gains again appear on articulated limb parts, supporting that $\mathcal { L } _ { d e c o r r }$ improves local part separation rather than only coarse semantic recognition.

Table 9. Boundary-band per-class IoU on Human Parsing. 

<table><tr><td>Class</td><td>w/o  $\mathcal{L}_{decorr}$ </td><td>w/  $\mathcal{L}_{decorr}$ </td><td> $\Delta$ </td></tr><tr><td>Background</td><td>51.36</td><td>55.37</td><td>+4.01</td></tr><tr><td>Head</td><td>55.95</td><td>60.46</td><td>+4.52</td></tr><tr><td>Torso</td><td>44.89</td><td>49.21</td><td>+4.32</td></tr><tr><td>Upper arm</td><td>34.10</td><td>40.83</td><td>+6.72</td></tr><tr><td>Lower arm / hand</td><td>41.02</td><td>45.60</td><td>+4.58</td></tr><tr><td>Upper leg</td><td>34.78</td><td>41.14</td><td>+6.36</td></tr><tr><td>Lower leg / foot</td><td>33.72</td><td>42.35</td><td>+8.63</td></tr></table>

# D. Sensitivity Analysis

We evaluate the sensitivity of PRISM to routing sparsity, expert capacity, the shared expert, and the application depth of $\mathcal { L } _ { d e c o r r }$ on PASCAL-Context with a ViT-S student. As shown in Table 10, nearby configurations remain competitive, indicating that PRISM is not a brittle single-point design. The default setting provides the best overall balance, especially on Human Parsing and Normal Estimation.

The routing sparsity results show that Top-1 underuses expert combinations, while Top-4 increases capacity but hurts Boundary, suggesting that overly dense routing weakens specialization. Varying the number of experts from 12 to 18 produces stable but slightly lower results than the default N = 15. Removing the shared expert also degrades performance, confirming that a small shared component inside the sparse path stabilizes routing. Finally, applying $\mathcal { L } _ { d e c o r r }$ across L2,5 gives the best multi-task trade-off, whereas removing it severely hurts Human Parsing.

Table 10. Sensitivity analysis on PASCAL-Context with ViT-S. 

<table><tr><td>Configuration</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td></tr><tr><td>N = 15, Top-1</td><td>77.16</td><td>66.77</td><td>84.43</td><td>14.79</td><td>68.47</td></tr><tr><td>N = 15, Top-2</td><td>78.97</td><td>68.91</td><td>84.68</td><td>14.66</td><td>69.02</td></tr><tr><td>N = 15, Top-4</td><td>78.81</td><td>68.92</td><td>84.45</td><td>14.62</td><td>66.43</td></tr><tr><td>N = 12, Top-3</td><td>78.21</td><td>68.56</td><td>84.89</td><td>14.78</td><td>68.84</td></tr><tr><td>N = 18, Top-3</td><td>78.34</td><td>68.68</td><td>84.37</td><td>14.70</td><td>69.01</td></tr><tr><td>w/o Shared Expert</td><td>78.72</td><td>67.73</td><td>84.20</td><td>14.88</td><td>68.01</td></tr><tr><td>w/o  $\mathcal{L}_{decorr}$ </td><td>79.80</td><td>61.80</td><td>85.50</td><td>14.32</td><td>70.86</td></tr><tr><td> $\mathcal{L}_{decorr}$  on L2</td><td>79.12</td><td>68.96</td><td>84.53</td><td>14.51</td><td>68.27</td></tr><tr><td> $\mathcal{L}_{decorr}$  on L2,5,8</td><td>78.80</td><td>68.32</td><td>84.37</td><td>14.45</td><td>68.24</td></tr><tr><td>PRISM Default</td><td>79.19</td><td>69.25</td><td>85.01</td><td>14.28</td><td>70.78</td></tr></table>

# E. Reduced-Teacher Distillation

We further test whether PRISM still helps when fewer teachers are distilled. Table 11 reports a reduced-teacher setting using only DINOv2 and SAM. Under the same student and training setup, PRISM clearly outperforms a dense widened student. This indicates that PRISM’s benefit does not rely on the full CLIP+DINOv2+SAM teacher set; even with fewer heterogeneous teachers, sparse context-dependent routing remains useful for separating and recombining complementary knowledge.

Table 11. Reduced-teacher setting with $\mathbf { D I N O v } 2 + \mathbf { S A M }$ on PASCAL-Context. 

<table><tr><td>Model</td><td>Semseg mIoU ↑</td><td>Parsing mIoU ↑</td><td>Saliency maxF ↑</td><td>Normal mErr ↓</td><td>Boundary odsF ↑</td></tr><tr><td>Wide ViT</td><td>66.13</td><td>61.21</td><td>84.16</td><td>14.73</td><td>67.08</td></tr><tr><td>PRISM</td><td>77.80</td><td>68.90</td><td>84.50</td><td>14.43</td><td>69.48</td></tr></table>

# F. Additional Efficiency Details

PRISM introduces additional offline training cost because Stage 1 requires forwarding frozen VFM teachers. This cost is paid only during distillation; after training, inference uses a single student model without running multiple VFMs. Table 12 summarizes the end-to-end training cost for the main ViT-B PASCAL setting.

Table 12. End-to-end training cost of PRISM on PASCAL-Context with ViT-B. GPU-hours include teacher forwarding. 

<table><tr><td>Stage</td><td>Schedule</td><td>Train / Val Batch size</td><td>Hardware</td><td>GPU-hours</td></tr><tr><td>Stage 1</td><td>30 epochs</td><td>20 / 20</td><td>RTX 5090 32GB</td><td>499.2</td></tr><tr><td>Stage 2</td><td>40k iters</td><td>2 / 8</td><td>RTX 5090 32GB</td><td>56.1</td></tr><tr><td>Total</td><td>-</td><td>-</td><td>-</td><td>555.3</td></tr></table>

# G. Additional Analysis of Routing Dynamics

We present a detailed, layer-by-layer visualization of the routing dynamics. In the following figures, each heatmap spans the full page width to reveal fine-grained activation patterns.

![](images/534d8d9ebce56471d7b6732241b17653537ed9e2716fd40b29b19512bb2bdee3.jpg)

<details>
<summary>heatmap</summary>

|        | E0   | E1   | E2   | E3   | E4   | E5   | E6   | E7   | E8   | E9   | E10  | E11  | E12  | E13  | E14  |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| DINOv2 | 33.3%| 0.2% | 0.0% | 3.1% | 0.0% | 0.3% | 31.6%| 1.8% | 0.0% | 0.0% | 0.0% | 2.7% | 22.1%| 4.9% | 0.0% |
| CLIP   | 0.0% | 9.5% | 0.1% | 1.0% | 0.0% | 1.2% | 0.0% | 0.4% | 0.0% | 0.0% | 32.9%| 24.1%| 0.0% | 1.3% | 29.5%|
| SAM    | 0.0% | 0.0% | 0.0% | 0.0% | 33.3%| 0.0% | 0.0% | 0.0% | 33.3%| 33.3%| 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
</details>

(a) VFM Affinity (Layer 2)

![](images/37444d71c382fdb0f2cc42fc1858e5b6d8249ff8275a8b8a1941a1b221a15ae9.jpg)

<details>
<summary>heatmap</summary>

|        | E0   | E1   | E2   | E3   | E4   | E5   | E6   | E7   | E8   | E9   | E10  | E11  | E12  | E13  | E14  |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| Semseg | 24.6%| 1.6% | 0.5% | 7.4% | 0.0% | 5.0% | 11.0%| 4.9% | 0.0% | 0.0% | 5.3% | 12.7%| 16.6%| 6.2% | 4.3% |
| Parsing | 25.8%| 2.0% | 0.4% | 6.1% | 0.0% | 4.0% | 11.8%| 4.3% | 0.0% | 0.0% | 5.8% | 11.0%| 20.1%| 6.0% | 2.6% |
| Saliency | 2.6% | 10.0%| 4.9% | 8.3% | 0.0% | 8.9% | 0.4% | 11.9%| 0.0% | 0.0% | 9.4% | 30.0%| 0.2% | 4.9% | 8.2% |
| Normal | 0.4% | 3.1% | 32.8%| 5.1% | 0.0% | 9.8% | 0.1% | 20.1%| 0.0% | 0.0% | 9.3% | 2.8% | 0.1% | 10.1%| 6.3% |
| Boundary | 26.0%| 1.6% | 0.2% | 6.0% | 0.0% | 3.4% | 13.3%| 5.4% | 0.0% | 0.0% | 3.0% | 11.0%| 21.1%| 6.7% | 2.2% |
</details>

(b) Task Affinity (Layer 2)   
Figure 10. Routing Dynamics at Shallow Layer (Layer 2). Top: VFM Affinity. Bottom: Task Affinity. At this early stage, the activation blocks are broad and diffuse. The router does not yet distinguish sharply between teachers or tasks, indicating that Layer 2 processes universal, shared visual primitives (Global Consensus).

![](images/b656bd580e94aeba255f91a18ccdd06f579d8e057d14a80b5ae28a9f7bb4cbff.jpg)

<details>
<summary>heatmap</summary>

|        | E0   | E1   | E2   | E3   | E4   | E5   | E6   | E7   | E8   | E9   | E10  | E11  | E12  | E13  | E14  |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| DINOv2 | 15.8%| 3.5% | 0.0% | 0.0% | 0.0% | 14.4%| 33.3%| 0.0% | 8.5% | 9.1% | 3.7% | 7.5% | 4.3% | 0.0% | 0.0% |
| CLIP   | 0.0% | 0.0% | 33.3%| 33.0%| 33.2%| 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| SAM    | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 33.3%| 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 33.3%| 33.3%|
</details>

(a) VFM Affinity (Layer 5)

![](images/586f462fdb77c5850f1f1e7dd13cfcb5373bd0c085e7a24bec0221cd0e963ce5.jpg)

<details>
<summary>heatmap</summary>

|        | E0   | E1   | E2   | E3   | E4   | E5   | E6   | E7   | E8   | E9   | E10  | E11  | E12  | E13  | E14  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Semseg | 4.6% | 14.3% | 2.9% | 7.8% | 0.1% | 16.7% | 7.6% | 5.6% | 6.6% | 5.2% | 8.2% | 5.1% | 7.0% | 0.0% | 8.3% |
| Parsing | 5.6% | 12.3% | 1.8% | 6.0% | 0.0% | 20.5% | 9.8% | 4.9% | 7.3% | 4.7% | 8.4% | 4.7% | 6.1% | 0.0% | 7.7% |
| Saliency | 2.6% | 5.7% | 26.6% | 7.2% | 18.0% | 2.9% | 0.9% | 4.6% | 4.7% | 4.4% | 7.1% | 4.0% | 5.4% | 0.0% | 5.9% |
| Normal | 9.2% | 4.8% | 30.4% | 11.6% | 27.8% | 2.1% | 1.0% | 0.6% | 3.3% | 2.4% | 2.2% | 0.9% | 1.8% | 0.0% | 1.9% |
| Boundary | 4.1% | 24.8% | 1.1% | 7.5% | 0.0% | 8.1% | 1.8% | 0.2% | 9.3% | 8.2% | 9.8% | 9.5% | 9.1% | 0.0% | 6.5% |
</details>

(b) Task Affinity (Layer 5)   
Figure 11. Routing Dynamics at Middle Layer (Layer 5). Transitioning to mid-level features, we observe the onset of separation. While some overlaps persist, the router begins to form distinct clusters for different contexts, preparing for the semantic split in deeper layers.

![](images/a56d02d15c4341f8f343032b750bad6878b1e86dceb5f36d4618c665e0004d72.jpg)

<details>
<summary>heatmap</summary>

| | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 | E13 | E14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2 | 6.9% | 14.4% | 0.0% | 5.6% | 5.0% | 27.2% | 0.0% | 12.8% | 0.0% | 20.2% | 0.0% | 0.7% | 0.0% | 0.3% | 6.9% |
| CLIP | 7.1% | 2.2% | 0.3% | 2.8% | 10.1% | 0.0% | 12.4% | 0.0% | 15.7% | 0.0% | 0.0% | 13.7% | 27.8% | 5.2% | 2.6% |
| SAM | 0.0% | 0.0% | 28.6% | 0.0% | 0.0% | 0.0% | 4.8% | 0.0% | 15.1% | 0.0% | 31.3% | 0.0% | 0.0% | 20.3% | 0.0% |
The chart displays a heatmap of percentage values for each category at each time point in the table. The values are estimated based on the grid layout and are labeled numerically on the cells.
</details>

(a) VFM Affinity (Layer 8)

![](images/54de73b6f2a7ff3962d09c4f6e351c5043a3d5cb7546b08216d22956ded48abe.jpg)

<details>
<summary>heatmap</summary>

|        | E0   | E1   | E2   | E3   | E4   | E5   | E6   | E7   | E8   | E9   | E10  | E11  | E12  | E13  | E14  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Semseg | 11.6% | 8.4% | 7.3% | 12.6% | 13.9% | 6.1% | 0.1% | 4.6% | 3.4% | 0.8% | 7.3% | 8.4% | 2.2% | 2.4% | 10.8% |
| Parsing | 11.6% | 5.9% | 7.5% | 12.1% | 13.3% | 6.6% | 0.3% | 9.5% | 5.6% | 2.7% | 7.1% | 7.4% | 2.4% | 3.6% | 4.5% |
| Saliency | 11.1% | 11.3% | 6.7% | 10.4% | 13.5% | 3.6% | 1.1% | 1.3% | 6.9% | 0.0% | 6.5% | 10.3% | 2.4% | 9.3% | 5.6% |
| Normal | 12.3% | 10.5% | 3.8% | 8.6% | 14.2% | 3.5% | 0.9% | 1.1% | 6.6% | 0.0% | 0.3% | 12.7% | 7.4% | 6.0% | 12.1% |
| Boundary | 9.2% | 8.5% | 7.1% | 11.0% | 14.9% | 9.2% | 0.2% | 8.4% | 7.3% | 3.8% | 5.4% | 7.1% | 2.1% | 1.5% | 4.4% |
</details>

(b) Task Affinity (Layer 8)   
Figure 12. Routing Dynamics at Deep Layer (Layer 8). In the deep layers, the experts exhibit sharp, context-dependent specialization resembling Layer 11. Distinct, non-overlapping clusters form for each teacher (Top), supporting that effective conflict reduction is mainly handled in the network’s later stages.