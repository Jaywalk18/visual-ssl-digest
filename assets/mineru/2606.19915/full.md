# SpatialSV: Internalizing Interpretable 3D Spatial Awareness in MLLMs via Task-Oriented Visual Supervision

Jiayu Tang , Yuchen Zhou , Chao Gou∗

School of Intelligent Systems Engineering, Sun Yat-sen University

tangjy59@mail2.sysu.edu.cn, zhouych37@mail2.sysu.edu.cn, gouchao@mail.sysu.edu.cn

## Abstract

Unlocking the spatial intelligence of multimodal large language models (MLLMs) is crucial for understanding and interacting with the 3D world. Prevailing approaches typically inject spatial priors via external tools, which impose significant inference overhead, or rely on latent feature distillation, which remains uninterpretable and lacks finegrained geometric constraints. To address these issues, we propose SpatialSV, a framework designed to internalize robust 3D spatial awareness within MLLMs while simultaneously offering inherent interpretability. Deviating from passive feature imitation, SpatialSV employs task-oriented visual supervision, compelling the model to actively lift its 2D visual features into explicit 3D representations, including depth maps, camera poses, and point clouds. Crucially, this 2D-to-3D lifting process provides a transparent window into the model’s representations: the resulting 3D reconstructions serve as an intuitive proxy for visualizing and diagnosing the quality of the model’s intrinsic spatial knowledge. Extensive experiments across multiple models and benchmarks demonstrate the effectiveness of SpatialSV in enhancing and interpreting MLLMs’ spatial intelligence. Furthermore, the framework exhibits strong generalization in semisupervised settings, validating its potential to leverage unlabeled visual data for scalable, interpretable spatial representation learning.

## 1 Introduction

Spatial intelligence refers to the ability to understand, interpret and reason about the spatial relationships and properties of objects and scenes. This ability is fundamental for realworld tasks ranging from autonomous driving [Zhou et al., 2025; Tang et al., 2026] to robotic manipulation [Zhou et al., 2023; Song et al., 2025], where understanding and interacting with the 3D environments is essential. Despite significant progress in spatial reasoning and scene understanding, even the most advanced multimodal large language models (MLLMs) face challenges in maintaining cross-view consistency and reasoning about occluded objects [Yin et al., 2025; Yang et al., 2025a]. These limitations highlight a fundamental weakness of MLLMs—the reliance on 2D visual-textual data and autoregressive training paradigms—which fails to yield robust and consistent internal 3D spatial representations. This absence of intrinsic 3D spatial awareness is a major bottleneck toward genuine spatial intelligence.

![](images/e1ca2346cf527d3da6da3c9d9b63ce7c6d1d007f3ea6ebc4c236bc2bd52b6627.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Prevaling Methods"] --> B["Spatial Question"]
  B --> C["External Tools"]
  C --> D["Spatial Priors"]
  D --> E["MLLM"]
  E --> F["Without internal spatial awareness, I have to rely on to understand +"]
  F --> G["Quality & Efficiency Issues"]
    
  H["SpatialSV (Training Stage)"] --> I["3D Task-oriented supervision"]
  I --> J["Point Clouds"]
  I --> K["Depth Maps"]
  I --> L["Camera Poses"]
  J --> M["Internalizing Spatial Awareness"]
  K --> M
  L --> M
  M --> N["I am learning how to represent the scene +"]
    
  O["SpatialSV (Inference Stage)"] --> P["Free from Any Tool or Prior"]
  P --> Q["Spatial Question"]
  Q --> R["MLLM"]
  R --> S["I can represent + internally Interpretability 2D→3D lift"]
  S --> T["Accordingly, my answer is ....."]
```
</details>

Figure 1: (a) Prevailing methods incorporate external spatial priors as input to MLLMs, suffering from prior quality and inference efficiency issues. (b) SpatialSV internalizes 3D spatial awareness in MLLMs via task-oriented visual supervision, which enables spatial understanding with interpretable intrinsic spatial representations.

To address this limitation, one major line of research focuses on providing MLLMs with external spatial priors to compensate for their insufficient intrinsic representations, as illustrated in Figure 1 (a). Among these, prompt-based methods rely on external models or tools to construct prompts enriched with spatial priors, [Yang et al., 2025b; Li et al., 2025d; Qi et al., 2025; Zhu et al., 2025b; Liu et al., 2025b]. However, such methods introduce substantial inference overhead and are sensitive to errors from external models. A more direct alternative is to explicitly encode 3D data such as point clouds and depth maps, and project them to the languagealigned space of large language models [Chen et al., 2024b;

Li et al., 2025b; Wang et al., 2025c; Zhu et al., 2025a]. However, these methods typically depend on highly customized encoding and alignment modules and will introduce both training and inference complexity. Moreover, the strong 2D pretraining bias of MLLMs poses an additional obstacle to the explicit integration of 3D data.

Instead of emphasizing the construction and utilization of external spatial representations, we argue that MLLMs should internalize spatial awareness to form an internal spatial mental model [Johnson-Laird, 1980; Yin et al., 2025]. Imaging in a practical embodied AI system where acquiring 3D data online is often costly and challenging with strict real-time constraints, MLLMs must rely on their own spatial awareness to construct internal representations of the environment, while maintaining a lightweight architecture to enable fast inference. To this end, recent works attempt to inject spatial awareness into MLLMs by distilling features from 3D vision foundation models (VFMs) [Huang et al., 2025a; Chen et al., 2025b]. However, the representation learned via distillation is inherently hard to interpret, limiting a deeper understanding of the internal spatial mental modeling mechanism [Johnson-Laird, 1983] within MLLMs.

Inspired by the probing techniques [El Banani et al., 2024; Chen et al., 2025a], we conduct a pilot study to investigate the quality of spatial representations across different MLLMs under two paradigms: pure autoregressive text supervision and feature distillation. The probing results, as illustrated in Figure 2, reveal the following observations: (1) The quality of an MLLM’s intrinsic spatial representation is positively correlated with its level of spatial intelligence. (2) Introducing 3Daware supervision improves the quality of spatial representations. These findings validate the effectiveness of 3D-aware supervision in enhancing the spatial intelligence of MLLMs. However, the performance gap between the depth probing results and the ground-truth further exposes the limitations of the distillation paradigm. This is because distilling features from 3D VFMs constitutes a coarse-grained alignment process that lacks explicit and structured spatial constraints. In addition, aligning feature dimensions during distillation inevitably incurs information loss in the target representations, further weakening the robustness of the supervision signal.

These insights raise a central question: can we identify a more intuitive and fine-grained form of 3D-aware supervision? In response to this question, we propose SpatialSV, a framework that internalizes interpretable and robust 3D spatial awareness in MLLMs via task-oriented visual supervision, as illustrated in Figure 1 (b). Specifically, we perform 2D-to-3D lifting of MLLMs’ visual features and align them with explicit spatial representations such as depth maps and point clouds, so as to unlock their intrinsic spatial intelligence. Inspired by 3D VFMs which leverage overlapping yet complementary downstream 3D tasks to learn geometrically consistent representations, we incorporate a set of complementary 3D tasks, including depth estimation, point-cloud reconstruction, and ray-map prediction. Technically, we extract multi-layer hidden visual features from the MLLM and employ projection layers together with decoupled DPT modules to perform multi-task prediction. By combining the coarse-grained guidance of feature distillation with fine-grained, task-oriented constraints, SpatialSV facilitates robust intrinsic spatial representation learning of MLLMs. Extensive experiments across multiple MLLMs and benchmarks demonstrate the superior performance of SpatialSV in enhancing spatial intelligence.

![](images/e646073977d4f8d6644d4511248a3cefd8ae6f32c829bb6728d0821ecd41e46a.jpg)

<details>
<summary>line chart</summary>

| Model | Baseline | Pure-text | Distillation |
| --- | --- | --- | --- |
| Owen2.5-VL-3B | 60 | 60 | 60 |
| Owen2.5-VL-7B | 55 | 55 | 55 |
| LLaVA-OneVision-1.5-4B | 50 | 50 | 50 |
| LLaVA-OneVision-1.5-8B | 45 | 45 | 45 |
| LLaVA-NeXT-Video-7B | 40 | 40 | 40 |
| InternVL3-1B | 35 | 35 | 35 |
| InternVL3-2B | 30 | 30 | 30 |
| InternVL3-8B | 25 | 25 | 25 |
</details>

![](images/41becb5a19e43e72acecea0a51c9b929948b9db14533fb39892de1db4edf59c4.jpg)

<details>
<summary>text_image</summary>

Baseline Pure-Text Distillation GT. Raw Image
(b)
</details>

Figure 2: Depth probing results. (a) Quantitative results: the correlation between the quality of internal representations and the level of spatial intelligence. We compare 8 MLLMs under three variants: an untuned baseline, a text supervision variant, and a distillation variant. (b) Qualitative results: visualized depth maps of three variants for Qwen2.5-VL-3B, along with the ground-truths and raw images.

Additionally, the 3D lifting results derived from MLLMs serve as an intuitive and interpretable manifestation of models’ internal spatial representations. Through both quantitative and qualitative analyses, we investigate a distinct correlation between 3D lifting results and intra-model spatial intelligence as well as sample difficulties, highlighting the inherent interpretability of SpatialSV. Furthermore, we apply SpatialSV to a semi-supervised setting where 50% of the samples have no textual annotations and achieve performance comparable to full text supervision. This is particularly important for scenarios where 3D text annotations are scarce. In summary, our contributions include:

• We propose SpatialSV, a novel framework that internalizes robust 3D spatial awareness in MLLMs through task-oriented visual supervision, with the 3D lifting results serving as an intuitive and interpretable proxy for intra-model spatial representations.  
• We introduce a probe-based representation analysis framework for MLLMs, revealing the limitations of existing supervision paradigms and offering insights into improving supervision signals.  
• Extensive experiments demonstrate SpatialSV’s strong cross-model and cross-dataset generalization in enhancing the spatial intelligence of MLLMs, and highlight its inherent interpretabiliy, as well as the potential for exploiting unlabeled visual data.

![](images/7b4cc0ee9e50883600aabb2f28135c3b59a58444c15105e06888ff18405d22fc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Intrinsic Spatial Repre."] --> B["3D-aware Supervision"]
  B --> C["Representation Interpretation"]
  C --> D["3D Task-oriented Repre."]
  D --> E["3D-aware Supervision"]
  E --> F["3D-task-oriented Repre."]
  F --> G["Multi-modal Large Language Model"]
  G --> H["Language Instruction"]
    
    subgraph Inputs
  I1["v^l4"] --> I2["x^l4"]
  I3["v^l3"] --> I4["x^l3"]
  I5["v^l2"] --> I6["x^l2"]
  I7["v^l1"] --> I8["x^l1"]
  I9["Layer l4-th"] --> I10["..."]
  I11["Layer l3-th"] --> I12["..."]
  I13["Layer l2-th"] --> I14["..."]
  I15["Layer l1-th"] --> I16["..."]
    end
    
    subgraph Outputs
  J["v"] --> K["v3D"] --> L["v3D"] --> M["v3D"] --> N["m3D"]
    end
    
    subgraph Legend
  O["Ray maps"] --> P["Point cloud maps"]
  Q["Depth maps"] --> R[Vanilla Text Supervision <answer>C. Light purple sofa</answer>
    end
```
</details>

Figure 3: The schematic illustration of SpatialSV, a framework that internalizes interpretable and robust 3D spatial awareness in MLLMs via task-oriented visual supervision. We perform 2D-to-3D lifting of the MLLM’s multi-layer hidden visual features and align them with explict 3D representations, including depth maps, ray maps, and point clouds. In this way, SpatialSV facilitates robust spatial representation learning and offers representation interpretability.

## 2 Method

## 2.1 Preliminaries

A typical MLLM consists of a visual encoder $\mathcal { E } _ { v }$ and a decoder-only large language model (LLM) F. In the context of multi-view spatial understanding, the multimodal input comprises N multi-view images $\mathcal { T } \overset { = } { = } \{ I _ { i } \} _ { i = 1 } ^ { N }$ along with a language instruction. The visual encoder transforms the images into visual features ${ \pmb v } ^ { 0 } = \mathcal { E } _ { v } ( \mathcal { T } ) \in \mathbb { R } ^ { N _ { v } \times d }$ aligned with the LLM’s input space, where $N _ { v }$ is the number of visual tokens, and d is the feature dimension. The LLM then performs cross-modal feature interaction between the visual features and instruction tokens, producing L layers of multi-$\{ \pmb { v } ^ { i } , \pmb { x } ^ { i } \} _ { i = 1 } ^ { \pmb { \mathscr { L } } }$ $\boldsymbol { v } ^ { i }$ $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } ^ { i } }$ the i-th layer of visual and textual features, respectively.

During the fine-tuning stage, the LLM applies causal modeling to the final-layer multimodal features and autoregressively predicts the next text token. The model is optimized by minimizing the standard cross-entropy loss:

$$
\mathcal {L} _ {\text { text }} = - \sum_ {t = 1} ^ {T} \log p _ {\theta} (\boldsymbol {x} _ {t} ^ {L} \mid \boldsymbol {x} _ {<   t} ^ {L}, \boldsymbol {v} ^ {L}), \tag {1}
$$

where $p _ { \theta }$ is the probability distribution conditioned on the previous multimodal context, and T is the length of the text token sequence. Notably, only the textual features are explicitly supervised under this training paradigm.

Following prior work [Huang et al., 2025a], we inject spatial awareness into the MLLM by explicitly supervising its visual features. Specifically, we introduce a 3D vision foundation model $\mathcal { F } _ { 3 D }$ to extract target features that encodes geometric priors, and apply 2D average pooling to match the size of MLLM’s visual features: $v _ { 3 \mathrm { D } } = f _ { \mathrm { p o o l } } ( \mathcal { F } _ { 3 \mathrm { D } } ( \mathcal { T } ) ) \ \in$ R $N _ { v } \times d _ { \mathrm { 3 D } }$ , where $d _ { 3 \mathrm { D } }$ is the dimension of target features. In addition, we project MLLM’s last-layer visual features into the target feature space to ensure dimension compatibility: $v _ { \mathrm { p r o j } } \stackrel { \smile } { = } \phi ( \pmb { v } ^ { L } ) \in \mathring { \mathbb { R } } ^ { N _ { v } \times d _ { \mathrm { 3 D } } }$ . Finally, the model is optimized by maximizing the cosine similarity between $v _ { \mathrm { p r o j } }$ and ${ \pmb v } _ { \mathrm { 3 D } }$ in a distillation paradigm:

$$
\mathcal {L} _ {\text { distill }} = - S (\boldsymbol {v} _ {\text { proj }}, \boldsymbol {v} _ {\text { 3D }}), \tag {2}
$$

where $S ( \cdot , \cdot )$ denotes cosine similarity. However, this distillation supervision suffers from inherent limitations in the interpretability of representations. In contrast, an intuitive analysis of MLLMs’ internal representation quality and its correlation with spatial intelligence is crucial for understanding the spatial mental modeling mechanism within MLLMs and uncovering their capability boundaries.

## 2.2 Probe-based Spatial Representation Analysis

To acquire deeper insights into the intrinsic spatial intelligence of MLLMs, we investigate the correlation between representation quality and intelligence level under different supervision paradigms. Inspired by prior work [El Banani et al., 2024; Chen et al., 2025a] which investigate the 3D-awareness within visual models, we introduce a probe-based representation analysis framework for MLLMs.

Specifically, we evaluate multiple MLLMs, including Qwen2.5-VL [Bai et al., 2025], LLaVA-NeXT-Video [Zhang et al., 2024], InternVL3 [Zhu et al., 2025c], and LLaVA-OneVision-1.5 [Li et al., 2024a], on the multi-view spatial understanding benchmark MindCube-Tiny [Yin et al., 2025]. For each MLLM, we consider three variants: (i) an untuned version, (ii) a version fine-tuned solely with autoregressive text supervision, and (iii) a version that additionally incorporates visual distillation supervision beyond text supervision.

To quantify spatial intelligence, we compute each model’s question-answering accuracy on MindCube-Tiny. To access the quality of intrinsic representations, we attach a DPT [Ranftl et al., 2021] module on top of MLLM’s visual features for depth estimation. During this process, all MLLM parameters are frozen and only the DPT module is trainable. The depth map annotations for multi-view images in MindCube are obtained using DepthAnything-v3 [Lin et al., 2025], and the

DPT head is trained on the training split. After training, we evaluate the depth estimation quality on MindCube-Tiny using the RMSE metric, which serves as a proxy for the quality of MLLM’s internal spatial representations.

Figure 2 presents both quantitative and qualitative results, from which we draw two key observations: (1) for a given MLLM, introducing visual distillation supervision substantially improves both intrinsic representation quality and spatial intelligence; (2) as the quality of intrinsic representations improves, the model’s spatial intelligence consistently increases. These findings strongly validate the effectiveness of 3D-aware visual supervision in enhancing the spatial intelligence of MLLMs. However, from the depth visualizations in Figure 2, we can observe a noticeable performance gap between the probing depth maps obtained under the distillation paradigm and the ground-truths. Specifically, the probing results appear overly blurred and suffer from significant loss of geometric details. This can be attributed to the inherent limitations of feature distillation supervision: (1) distillation at the feature level is inherently coarse-grained and lacks explicit, structured geometric constraints; (2) feature dimension alignment (e.g., via 2D average pooling) inevitably leads to information loss in the target representations. Collectively, these insights motivate a central question: can we identify a fine-grained and intuitive form of 3D-aware visual supervision that can not only enhance but also interpret the intrinsic spatial intelligence of MLLMs?

## 2.3 SpatialSV

To overcome the inherent limitations of feature distillation supervision, we propose SpatialSV, a framework that internalizes interpretable and robust 3D spatial awareness in MLLMs via task-oriented visual supervision, as depicted in Figure 3.

3D VFMs [Wang et al., 2025b; Lin et al., 2025] facilitate robust and geometrically consistent representation learning via overlapping yet complementary 3D downstream tasks, spanning point cloud reconstruction, camera pose estimation, depth estimation, and point tracking. Motivated by these work, we select three types of explicit spatial representations that are highly compatible with MLLM’s visual feature maps, namely depth maps, ray maps and point cloud maps. These representations encode per-pixel spatial semantics, e.g., depth, camera pose and 3D coordinates, and are therefore more intuitive, structured and fine-grained than abstract feature maps. To obtain the ground-truth data, we leverage 3D VFMs to estimate multi-view depth maps and camera parameters, which are further transformed into ray maps and point clouds (see Supp.A.1 for detailed computations). We align MLLM’s visual features with these 3D task outputs to improve the quality of internal spatial representations.

Concretely, we employ a set of two-layer projectors $\Phi _ { \mathrm { 3 D } }$ to lift MLLM’s multi-layer visual features into a shared 3D feature space. Among the hidden visual features of an MLLM, higher layers capture richer cross-modal semantics, whereas lower layers preserve more low-level visual information. Accordingly, we select visual features from four different layers to capture complementary spatial and semantic representations, with the set of selected layer indexes as $\{ l _ { i } \} _ { i = 1 } ^ { 4 }$ , and the projector set $\Phi _ { 3 \mathrm { D } } = \{ \phi _ { i } \} _ { i = 1 } ^ { 4 }$ . The projectors map the selected layer-wise features into multi-task shared 3D features:

$$
\boldsymbol {v} _ {\mathrm{3D}} ^ {i} = \phi_ {i} (\boldsymbol {v} ^ {l _ {i}}), i \in [ 1, 4 ], \tag {3}
$$

where $\boldsymbol { v } _ { 3 \mathrm { D } } ^ { i } \in \mathbb { R } ^ { N _ { v } \times d _ { 3 \mathrm { D } } ^ { i } }$ is the i-th selected layer of 3D-lifted features, and $d _ { \mathrm { 3 D } } ^ { i }$ is the corresponding dimension. Subsequently, we introduce task-decoupled DPT modules FDPT [Ranftl et al., 2021] which receive the lifted features as input and independently predict per-pixel depth, camera pose, and 3D coordinates:

$$
\left\{\boldsymbol {m} _ {\mathrm{3D}} ^ {t}, \boldsymbol {c} ^ {t} \right\} = \mathcal {F} _ {\mathrm{DPT}} ^ {t} \left(\left\{\boldsymbol {v} _ {\mathrm{3D}} ^ {i} \right\} _ {i = 1} ^ {4}\right), \tag {4}
$$

where $t \in$ {depth, ray, pointcloud} denotes the task type; $ { m _ { \mathrm { 3 D } } ^ { t } }$ $\boldsymbol { c } ^ { t } \in \mathbb { R } ^ { N _ { v } }$ the confidence of mt3D. Specifically, mdep.3D $ { m _ { \mathrm { 3 D } } ^ { t } }$ $m _ { 3 \mathrm { D } } ^ { \mathrm { d e p . } } ~ \in ~ \mathbb { R } ^ { N _ { v } \times 1 }$ corresponds to per-pixel depth values; $ { m _ { 3 \mathrm { D } } } ^ { \mathrm { r a y } } \ \in \ \mathbb { R } ^ { N _ { v } \times 6 }$ denotes per-pixel ray representations, where the first three channels encode the ray origin and the last three channels encode the ray direction; and $m _ { 3 \mathrm { D } } ^ { \mathrm { p o i . } } ~ \in ~ \mathbb { R } ^ { N _ { v } \times 3 }$ represents perpixel 3D spatial coordinates. Following [Wang et al., 2025b; Lin et al., 2025], we then compute task-specific losses between the MLLM’s 3D-lifted predictions and the corresponding ground-truth annotations:

$$
\mathcal {L} _ {\mathrm{3D}} ^ {t} = \mathcal {L} _ {2} \left(\boldsymbol {y} _ {\mathrm{3D}} ^ {t}, \boldsymbol {m} _ {\mathrm{3D}} ^ {t}\right) + \mathcal {L} _ {\text {conf}} \left(\boldsymbol {y} _ {\mathrm{3D}} ^ {t}, \boldsymbol {m} _ {\mathrm{3D}} ^ {t}; \boldsymbol {c} ^ {t}\right)) + \tag {5}
$$

$$
\mathbb {1} _ {\{t = \text {dep.} \}} \mathcal {L} _ {\text {grad}} + \mathbb {1} _ {\{t = \text {poi.} \}} \mathcal {L} _ {\text {norm}},
$$

where $\boldsymbol { y } _ { \mathrm { 3 D } } ^ { t }$ is the ground-truth of task $t ; \mathbb { I } _ { \{ \cdot \} }$ denotes the indicator function; ${ \mathcal { L } } _ { \mathrm { c o n f } }$ is the confidence loss; $\mathcal { L } _ { \mathrm { g r a d } }$ is the gradient loss applied to the depth residual, penalizing blurred, misplaced, or spurious depth transitions; and ${ \mathcal { L } } _ { \mathrm { n o r m } }$ is the surface normal loss that enforce the alignment between the local surface geometry of predicted points with the ground-truth. See Supp.A.2 for the detailed loss computation procedure.

Consequently, the overall training objective integrates an autoregressive text loss, a coarse-grained feature distillation loss, and fine-grained 3D-aware task-oriented losses:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { text }} + \mathcal {L} _ {\text { distill }} + \mathcal {L} _ {\mathrm{3D}} ^ {\text { dep. }} + \mathcal {L} _ {\mathrm{3D}} ^ {\text { ray }} + \mathcal {L} _ {\mathrm{3D}} ^ {\text { poi. }}. \tag {6}
$$

This approach combines coarse-grained guidance with finegrained geometric constraints, which promotes robust intrinsic spatial representation learning within MLLMs. Moreover, during inference, the 3D-lifted predictions derived from the MLLM serve as an intuitive proxy to assess representation quality and delineate the model’s capability boundaries, as discussed in Sec. 3.3.

## 3 Experiments

## 3.1 Experimental Settings

Datasets. For training, we use 10k training samples from MindCube [Yin et al., 2025] to fine-tune the model. For evaluation, we focus on two benchmarks, MindCube-Tiny [Yin et al., 2025] and VSI-Bench [Yang et al., 2025a], both of which target spatial understanding from limited ego-centric views. We evaluate models on MindCube-Tiny and the multiplechoice split of VSI-Bench.

Evaluation Metrics. We use Accuracy to evaluate the overall performance on multiple-choice questions, computed by exact matching between model predictions and ground-truths.

<table><tr><td rowspan="2">Method</td><td colspan="4">MindCube-Tiny</td><td colspan="7">VSI-Bench</td></tr><tr><td>Rotation</td><td>Among</td><td>Around</td><td>Overall↑</td><td>Rel.Dir. Hard</td><td>Rel.Dir. Medium</td><td>Rel.Dir. Easy</td><td>Rel. Dist.</td><td>App. Order</td><td>Route Plan</td><td>Overall↑</td></tr><tr><td colspan="12">Baseline</td></tr><tr><td>Chance Level (Random)</td><td>26.5</td><td>24.5</td><td>23.8</td><td>24.6</td><td>26.5</td><td>25.7</td><td>24.9</td><td>24.1</td><td>23.3</td><td>28.4</td><td>27.7</td></tr><tr><td>Chance Level (Frequency)</td><td>34.5</td><td>34.8</td><td>33.3</td><td>34.3</td><td>25.2</td><td>33.6</td><td>50.2</td><td>25.1</td><td>25.2</td><td>29.4</td><td>29.0</td></tr><tr><td colspan="12">Qwen2.5-VL Family</td></tr><tr><td>Qwen2.5-VL-3B</td><td>34.5</td><td>36.0</td><td>32.3</td><td>34.5</td><td>34.3</td><td>33.3</td><td>29.0</td><td>31.7</td><td>22.7</td><td>27.8</td><td>29.6</td></tr><tr><td>Qwen2.5-VL-3B + Text.</td><td>33.5</td><td>49.7</td><td>74.8</td><td>55.3</td><td>35.7</td><td>35.7</td><td>26.3</td><td>31.1</td><td>23.6</td><td>29.9</td><td>30.1</td></tr><tr><td>Qwen2.5-VL-3B + Distillation.</td><td>34.5</td><td>51.5</td><td>81.3</td><td>58.6</td><td>35.9</td><td>34.1</td><td>30.9</td><td>32.1</td><td>22.8</td><td>32.0</td><td>30.6</td></tr><tr><td>Qwen2.5-VL-3B + SpatialSV</td><td>36.0</td><td>56.2</td><td>84.5</td><td>62.3</td><td>34.6</td><td>37.8</td><td>34.6</td><td>33.5</td><td>23.3</td><td>37.1</td><td>32.2</td></tr><tr><td>Qwen2.5-VL-7B</td><td>35.0</td><td>29.7</td><td>26.8</td><td>29.6</td><td>28.4</td><td>24.9</td><td>45.2</td><td>31.8</td><td>29.5</td><td>31.4</td><td>30.8</td></tr><tr><td>Qwen2.5-VL-7B + Text.</td><td>35.5</td><td>51.0</td><td>73.5</td><td>55.9</td><td>27.4</td><td>31.2</td><td>51.2</td><td>31.4</td><td>30.1</td><td>34.5</td><td>32.4</td></tr><tr><td>Qwen2.5-VL-7B + Distillation.</td><td>35.0</td><td>53.8</td><td>76.5</td><td>58.3</td><td>29.0</td><td>32.5</td><td>50.2</td><td>34.8</td><td>29.1</td><td>37.6</td><td>33.7</td></tr><tr><td>Qwen2.5-VL-7B + SpatialSV</td><td>37.5</td><td>58.5</td><td>81.8</td><td>62.8</td><td>29.5</td><td>35.2</td><td>48.4</td><td>37.2</td><td>30.1</td><td>42.8</td><td>35.4</td></tr><tr><td colspan="12">LLaVA-OneVision-1.5 Family</td></tr><tr><td>LLaVA-OneVision-1.5-8B</td><td>33.5</td><td>31.0</td><td>32.5</td><td>32.0</td><td>28.7</td><td>34.9</td><td>48.9</td><td>36.1</td><td>28.8</td><td>28.9</td><td>33.5</td></tr><tr><td>LLaVA-OneVision-1.5-8B + Text.</td><td>35.0</td><td>49.5</td><td>55.5</td><td>49.1</td><td>34.1</td><td>35.5</td><td>52.1</td><td>37.3</td><td>29.9</td><td>30.4</td><td>35.5</td></tr><tr><td>LLaVA-OneVision-1.5-8B + Distillation.</td><td>36.5</td><td>52.7</td><td>58.3</td><td>51.8</td><td>36.5</td><td>35.5</td><td>50.2</td><td>38.3</td><td>28.3</td><td>32.0</td><td>35.7</td></tr><tr><td>LLaVA-OneVision-1.5-8B + SpatialSV</td><td>38.0</td><td>57.5</td><td>60.3</td><td>55.2</td><td>38.3</td><td>36.5</td><td>49.3</td><td>42.1</td><td>29.6</td><td>40.2</td><td>38.1</td></tr><tr><td colspan="12">LLaVA-NeXT-Video Family</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>34.5</td><td>41.7</td><td>34.5</td><td>38.1</td><td>27.1</td><td>32.5</td><td>50.7</td><td>35.9</td><td>27.5</td><td>31.4</td><td>32.9</td></tr><tr><td>LLaVA-NeXT-Video-7B + Text.</td><td>33.0</td><td>52.3</td><td>78.8</td><td>57.9</td><td>27.9</td><td>35.7</td><td>45.6</td><td>35.5</td><td>29.8</td><td>32.5</td><td>33.6</td></tr><tr><td>LLaVA-NeXT-Video-7B + Distillation.</td><td>36.5</td><td>55.2</td><td>79.5</td><td>60.2</td><td>29.5</td><td>33.6</td><td>48.4</td><td>37.5</td><td>29.5</td><td>34.5</td><td>34.4</td></tr><tr><td>LLaVA-NeXT-Video-7B + SpatialSV</td><td>39.5</td><td>61.8</td><td>82.3</td><td>64.9</td><td>29.2</td><td>33.9</td><td>52.5</td><td>37.8</td><td>30.4</td><td>44.9</td><td>35.9</td></tr><tr><td colspan="12">InternVL3 Family</td></tr><tr><td>InternVL3-2B</td><td>31.0</td><td>33.3</td><td>24.8</td><td>30.1</td><td>26.5</td><td>31.5</td><td>48.4</td><td>30.1</td><td>24.4</td><td>34.0</td><td>30.3</td></tr><tr><td>InternVL3-2B + Text.</td><td>32.0</td><td>54.3</td><td>78.0</td><td>58.5</td><td>29.0</td><td>34.1</td><td>45.6</td><td>32.0</td><td>30.1</td><td>29.9</td><td>32.4</td></tr><tr><td>InternVL3-2B + Distillation.</td><td>35.0</td><td>56.5</td><td>79.5</td><td>60.6</td><td>29.5</td><td>33.3</td><td>47.0</td><td>33.0</td><td>28.0</td><td>31.4</td><td>32.4</td></tr><tr><td>InternVL3-2B + SpatialSV</td><td>37.5</td><td>57.2</td><td>82.8</td><td>62.4</td><td>31.4</td><td>35.2</td><td>47.9</td><td>35.1</td><td>28.8</td><td>41.2</td><td>34.6</td></tr><tr><td>InternVL3-8B</td><td>34.5</td><td>39.3</td><td>33.3</td><td>36.5</td><td>23.6</td><td>34.1</td><td>46.1</td><td>34.1</td><td>32.9</td><td>32.5</td><td>33.1</td></tr><tr><td>InternVL3-8B + Text.</td><td>33.5</td><td>68.8</td><td>82.5</td><td>67.5</td><td>25.7</td><td>33.6</td><td>50.2</td><td>34.5</td><td>32.0</td><td>29.9</td><td>33.5</td></tr><tr><td>InternVL3-8B + Distillation.</td><td>35.5</td><td>70.8</td><td>81.5</td><td>68.5</td><td>28.4</td><td>36.0</td><td>47.5</td><td>35.5</td><td>31.7</td><td>34.0</td><td>34.5</td></tr><tr><td>InternVL3-8B + SpatialSV</td><td>38.5</td><td>77.2</td><td>86.0</td><td>73.7</td><td>30.6</td><td>37.3</td><td>50.7</td><td>36.2</td><td>33.2</td><td>42.8</td><td>36.6</td></tr></table>

Table 1: Comparison of our approach (SpatialSV) with the baseline models, the pure-text supervision variants, and the feature distillation varients on MindCube-Tiny and VSI-Bench. The best results within each model type are bolded. More detailed results across 8 MLLMs are provided in Supp.C.1.

For the 3D-lifted depth estimation task, we follow [El Banani et al., 2024] to report RMSE as the evaluation metric.

Implementation Details. We adopt multiple MLLMs as the baselines for SpatialSV, including Qwen2.5-VL [Bai et al., 2025], LLaVA-OneVision-1.5 [Li et al., 2024a], LLaVA-NeXT-Video [Zhang et al., 2024], and InternVL3 [Zhu et al., 2025c]. For all models, we fine-tune the vision–language projector, the large language model, all 2D-to-3D projectors, and the task-decoupled DPT modules, while freezing the visual encoder. All models are optimized using AdamW with a batch size of 16 and a warm-up ratio of 0.03. We use DepthAnything-v3 [Lin et al., 2025] by default to obtain the ground-truth supervision signals. More detailed hyperparameter settings of all models are provided in Supp.B.2. All experiments are conducted on 8 NVIDIA H800 GPUs.

## 3.2 Comparison with Baselines

Comparison with Other Supervision Paradigms. We evaluate our approach against the baselines, the pure-text supervision variants, and the feature distillation variants on 6 MLLMs spanning 4 model families (more detailed results present in Supp.C.1). As shown in Table 1, SpatialSV consistently achieves the best performance, demonstrating its strong cross-model generalization capability. Specifically, on MindCube-Tiny, SpatialSV yield overall gains ranging from 3.42% to 12.66% over the pure-text variants, and from 2.97% to 7.81% over the distillation variants. On VSI-Bench, we observe an overall improvement by 4.36% to 6.79% compared with the distillation variants. Notably, SpatialSV achieves an average improvement of 10.0% over the pure-text variants on the relative distance task, while on the more challenging route planning task, the improvement reaches 32.6%. This indicates that SpatialSV effectively enhances the intrinsic spatial representations for 3D scenes within MLLMs, which is critical for estimating spatial attributes such as distance and direction. Moreover, it is worth noting that no VSI-Bench-related sample is incorporated in training data; nevertheless, SpatialSV still achieves significant improvements, highlighting its remarkable crossdataset generalization ability.

Comparison on Multiple Benchmarks. We further evaluate our approach on several additional datasets, including 6 spatial understanding benchmarks, namely Ego3D-Bench [Gholami et al., 2025], Spatial457 [Wang et al., 2025d], ViewSpatial-Bench [Li et al., 2025a], 3DSR-Bench [Ma et al., 2025], SP-Bench [Li et al., 2025c], TopViewRS [Li et al., 2024b], and 2 general benchmarks, CVBench [Zhu et al., 2025d] and MMBench [Liu et al., 2024]. Detailed information about these datasets is provided in Supp.B.1. As shown in Table 2, our method consistently outperforms baseline models and pure-text variants across all benchmarks, indicating that SpatialSV enhances spatial intelligence while also preserving the general understanding capability. This further underscores SpatialSV’s strong cross-dataset generalization.

<table><tr><td>Method</td><td>Ego3D-Bench</td><td>Spatial457</td><td>ViewSpatial-Bench</td><td>3DSR-Bench</td><td>SPBench</td><td>TopViewRS</td><td>CVBench</td><td>MMBench</td><td>Avg.</td></tr><tr><td>Qwen2.5-VL-3B</td><td>32.4</td><td>33.9</td><td>35.8</td><td>49.6</td><td>39.3</td><td>43.3</td><td>70.7</td><td>76.3</td><td>47.7</td></tr><tr><td>Qwen2.5-VL-3B+Text.</td><td>31.8</td><td>33.4</td><td>36.3</td><td>49.3</td><td>40.2</td><td>43.4</td><td>70.2</td><td>76.2</td><td>47.6</td></tr><tr><td>Qwen2.5-VL-3B+SpatialSV</td><td>34.7</td><td>34.8</td><td>39.5</td><td>52.3</td><td>44.5</td><td>43.8</td><td>71.4</td><td>76.9</td><td>49.8</td></tr><tr><td>Qwen2.5-VL-7B</td><td>35.8</td><td>43.7</td><td>37.2</td><td>53.9</td><td>45.3</td><td>43.4</td><td>78.0</td><td>77.4</td><td>51.8</td></tr><tr><td>Qwen2.5-VL-7B+Text.</td><td>36.2</td><td>43.4</td><td>37.6</td><td>53.7</td><td>46.4</td><td>43.4</td><td>77.6</td><td>77.7</td><td>52.0</td></tr><tr><td>Qwen2.5-VL-7B+SpatialSV</td><td>39.3</td><td>44.3</td><td>40.1</td><td>55.2</td><td>47.8</td><td>43.6</td><td>78.2</td><td>79.3</td><td>53.5</td></tr></table>

Table 2: Comparison of our approach (SpatialSV) with the baseline models and the pure-text supervision variants on multiple spatial understanding and general benchmarks. The best results are bolded.

![](images/647cf0c3e87993ff56e13b7395e1401a66d2be60accc931af747c564f587fb89.jpg)

<details>
<summary>scatterplot</summary>

| Model | 3D Lifting Results Quality (RMSE %, lower for better) | Spatial Intelligence Level (Acc. %, higher for better) |
| :--- | :--- | :--- |
| Qwen2.5-VL-3B | 7.0 | 74 |
| Qwen2.5-VL-7B | 10.0 | 63 |
| LLaVA-OneVision-1.5-4B | 12.5 | 63 |
| LLaVA-OneVision-1.5-8B | 12.5 | 63 |
| LLaVA-NeXT-Video-7B | 7.5 | 65 |
| InternVL3-1B | 17.5 | 51 |
| InternVL3-2B | 17.5 | 55 |
| InternVL3-8B | 7.0 | 74 |
</details>

Figure 4: Correlation between the quality of the SpatialSV-based 3D lifting results and intra-model spatial intelligence level.

![](images/03538f926835bec31b0aa1419def5b1c12e032e70dca6fd56b45e7740c37b62d.jpg)

<details>
<summary>bar chart</summary>

| Data Bins Conditioned on 3D Lifting Results Quality | Qwen2.5-VL-3B | Qwen2.5-VL-7B | LLaVA-OneVision-1.5-4B | LLaVA-OneVision-1.5-8B | LLaVA-NeXT-Video-7B | InternVL3-1B | InternVL3-2B | InternVL3-3B |
| -------------------------------------------------- | ------------- | ------------- | ---------------------- | ---------------------- | ------------------- | -------------- | -------------- | -------------- |
| Bin1                                               | 48            | 50            | 36                     | 48                     | 56                  | 48             | 50             | 56             |
| Bin2                                               | 58            | 60            | 54                     | 60                     | 60                  | 58             | 60             | 70             |
| Bin3                                               | 70            | 62            | 58                     | 58                     | 68                  | 62             | 64             | 76             |
| Bin4                                               | 78            | 76            | 60                     | 60                     | 76                  | 68             | 72             | 90             |
</details>

Figure 5: Correlation between the SpatialSV-based 3D lifting results with model-specific sample perferences. Samples are partitioned into 4 bins based on the quality of 3D lifting results.

## 3.3 Interpretability Analysis

In this section, we investigate the inherent interpretability of SpatialSV-based 3D lifting results. We conduct both quantitative and qualitative analyses to reveal their correlations with intra-model spatial intelligence and model-specific sample preferences.

Correlation with Intra-Model Spatial Intelligence. We use the depth estimation performance to quantify the quality of 3D lifting results, and VQA performance to measure MLLMs’ intrinsic spatial intelligence, evaluating 8 MLLMs on MindCube-Tiny. In Figure 4, we observe a strong correlation between the quality of 3D lifting results and intramodel spatial intelligence, which is aligned with the findings in Sec.2.2. Qualitatively, Figure 6 shows that MLLMs exhibit varying representation capability under the same spatial context. In the first case, the depth maps and point clouds derived from LLaVA-OneVision-8B lack necessary details of the key object small table, leading to its deviation from the correct option involving this object. In the second case, the failure of Qwen2.5-VL-7B can be partly attributed to its ignorance of chairs, which could help establish cross-view correspondences. In contrast, models that answer correctly can represent these critical objects in their 3D lifting results. These strongly validate the effectiveness of SpatialSV-based 3D lifting results as a proxy of intra-model spatial representations.

<table><tr><td rowspan="2">feat.</td><td rowspan="2">dep.</td><td rowspan="2">poi.</td><td rowspan="2">ray.</td><td colspan="2">Qwen2.5VL-3B</td><td colspan="2">LLaVA-NV-7B</td></tr><tr><td>MC.</td><td>VSI.</td><td>MC.</td><td>VSI.</td></tr><tr><td>-</td><td>-</td><td>-</td><td>-</td><td>55.3</td><td>30.1</td><td>57.9</td><td>33.6</td></tr><tr><td>√</td><td>-</td><td>-</td><td>-</td><td>57.4</td><td>30.2</td><td>60.3</td><td>34.0</td></tr><tr><td>-</td><td>√</td><td>-</td><td>-</td><td>59.4</td><td>31.3</td><td>62.2</td><td>34.8</td></tr><tr><td>-</td><td>-</td><td>√</td><td>-</td><td>58.2</td><td>31.5</td><td>62.4</td><td>33.9</td></tr><tr><td>-</td><td>-</td><td>-</td><td>√</td><td>56.3</td><td>30.0</td><td>60.2</td><td>34.2</td></tr><tr><td>-</td><td>√</td><td>√</td><td>√</td><td>61.4</td><td>32.3</td><td>63.7</td><td>35.6</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>62.3</td><td>32.2</td><td>64.9</td><td>35.9</td></tr></table>

Table 3: Ablation of different supervision signals on MindCube-Tiny (MC.) and VSI-Bench (VSI.). Feat., dep., poi., and ray. denote features, depth maps, ray maps and point cloud maps, respectively.

Correlation with Model-Specific Sample Preferences. For each MLLM, we partition the samples in MindCube-Tiny into four bins according to the quality of estimated depth maps. From Bin1 to Bin4, the samples correspond to progressively high-quality 3D lifting results. In Figure 5, QA accuracy exhibit a clear upward trend across all models, suggesting that samples with unfaithful 3D lifting results tend to be more challenging for the models. As shown in Figure 6, Qwen2.5- VL-7B succeeds in the first sample—accurately representing the spatial scene and providing the correct answer—but fails in the second sample in both aspects. This further highlights the correlation between SpatialSV-based 3D lifting results and model-specific sample preferences.

## 3.4 Ablation Study

Different 3D-Aware Visual Supervision Signals. As reported in Table 3, we examine different 3D-aware visual supervision signals, including 3D VFM features, depth maps, point maps and ray maps. Each supervision signal yields performance gains to varying degrees, with depth maps contributing the most, likely due to their intuitive spatial semantics and relatively low alignment difficulty. When all taskoriented supervision signals are jointly applied, the model significantly outperforms any single-task supervision, highlighting the effectiveness of overlapping yet complementary multi-task supervision for learning robust spatial representations. Furthermore, incorporating VFM features brings additional gains, suggesting that coarse-grained feature alignment and fine-grained task-oriented spatial constraints can mutually guide and reinforce each other.

<table><tr><td rowspan="2">Layer Indexes</td><td rowspan="2">Projector Numbers</td><td colspan="4">MindCube-Tiny</td></tr><tr><td>Rotation</td><td>Among</td><td>Around</td><td>Overall</td></tr><tr><td>(36, 36, 36, 36)</td><td>4</td><td>32.5</td><td>54.7</td><td>81.3</td><td>59.8</td></tr><tr><td>(25, 26, 27, 28)</td><td>4</td><td>36.5</td><td>55.7</td><td>84.3</td><td>62.0</td></tr><tr><td>(4, 12, 20, 28)</td><td>1</td><td>35.0</td><td>55.2</td><td>81.8</td><td>60.7</td></tr><tr><td>(4, 12, 20, 28)</td><td>4</td><td>36.0</td><td>56.2</td><td>84.5</td><td>62.3</td></tr></table>

Table 4: Ablation of different visual layers and projector numbers on MindCube-Tiny for Qwen2.5-VL-3B. The best is bolded.

<table><tr><td rowspan="2">50% data</td><td rowspan="2">50% data</td><td colspan="4">MindCube-Tiny</td></tr><tr><td>Rotation</td><td>Among</td><td>Around</td><td>Overall</td></tr><tr><td> $\mathcal{L}_{\text{text}}$ </td><td>-</td><td>27.5</td><td>42.0</td><td>64.8</td><td>47.2</td></tr><tr><td> $\mathcal{L}_{\text{text}} + \mathcal{L}_{\text{spatial}}$ </td><td>-</td><td>29.5</td><td>46.7</td><td>70.5</td><td>51.8</td></tr><tr><td> $\mathcal{L}_{\text{text}} + \mathcal{L}_{\text{spatial}}$ </td><td> $\mathcal{L}_{\text{spatial}}$ </td><td>29.0</td><td>49.5</td><td>73.0</td><td>53.9</td></tr><tr><td> $\mathcal{L}_{\text{text}}$ </td><td> $\mathcal{L}_{\text{text}}$ </td><td>33.5</td><td>49.7</td><td>74.8</td><td>55.3</td></tr></table>

Table 5: Semi-supervised learning with SpatialSV.

Different Supervised Visual Feature Layers and Numbers of 2D-to-3D Projectors. As shown in Table 4, we explore different configurations of visual feature layers and projector numbers on Qwen2.5-VL-3B, which contains 37 visual feature layers (e.g., $L = 3 7 )$ . Regarding the selection of visual layers, choosing $\{ l _ { i } \} _ { i = 1 } ^ { 4 } = \{ 4 , 1 2 , 2 0 , 2 8 \}$ yields the best performance, achieving a 4.18% improvement over supervising only the final visual layer. This is because overly deep layers tend to lack low-level visual details crucial for 3D reconstruction, thereby hindering the learning of 3D spatial representations. In contrast, combining shallow visual details with deep cross-modal semantics which are complementary proves feasible. For projector numbers, employing layer-wise decoupled projectors leads to a 2.64% performance gain compared to using a single shared projector. This can be attributed to the semantic heterogeneity across different visual layers, which necessitates distinct 2D-to-3D mapping functions.

Semi-Supervised Learning with SpatialSV. Given the scarcity of high-quality 3D vision-language data, we explore directly exploiting unlabeled visual data. Since task-oriented visual supervision is independent of text autoregressive supervision, SpatialSV is naturally suited to semi-supervised learning. Specifically, we split the training data into two nonoverlapping subsets and consider 4 configurations: (i) applying pure-text supervision $( \mathcal { L } _ { \mathrm { t e x t } } )$ to 50% data; (ii) applying $\mathcal { L } _ { \mathrm { t e x t } }$ and SpatialSV $( \mathcal { L } _ { \mathrm { s p a t i a l } } )$ to 50% data; (iii) applying both losses to 50% data and only $\mathcal { L } _ { \mathrm { s p a t i a l } }$ to the remaining unlabeled 50%; (iv) applying $\mathcal { L } _ { \mathrm { t e x t } }$ to 100% data. From Table 5, We observe that the semi-supervised setting in (iii) achieves performance close to full text supervision (53.9 vs. 55.3), while yielding a remarkable 14.19% improvement over setting (i). These results highly validate the potential of SpatialSV for effectively exploiting unlabeled raw visual data. Notably, SpatialSV can directly leverage off-the-shelf 3D VFMs to generate 3D task annotations as semi-supervised signals, which is much less costly and easier to obtain than constructing spatial question-answering datasets.

![](images/2b8bb2789551ebac5409364ca76b9a1c2d558c2d09a7f10e6629349458100bbb.jpg)

<details>
<summary>text_image</summary>

Question: From the viewpoint presented in image 1, what is to the right of the office chair with blue seat?
A. Single sofa alongside a small table B. Table topped with a microwave C. Kitchenette D. Blue rubbish bin
The small table almost invisible
Qwen2.5VL-7B + SpatialSV:
A. Single sofa alongside a small table LLaVA-OneVision-8B + SpatialSV:
D. Blue rubbish bin
</details>

![](images/11fe69c1ee762aaaea57ff92fb91cc92bba1cee51b74f744b1f3572f061a41df.jpg)

<details>
<summary>text_image</summary>

Question: From the viewpoint presented in image 3, what is to the left of
the white table?
A. TV	B. Glass door	C. Gridded glass wall	D. Paneled glass wall
Ignore the chairs
InternVL-8B + SpatialSV:
C. Gridded glass wall	Qwen2.5VL-7B + SpatialSV:
B. Glass door
</details>

Figure 6: Qualitative results on MindCube-Tiny.

## 4 Conclusion

We propose SpatialSV, a novel framework that internalizes robust and interpretable 3D spatial awareness into MLLMs through task-oriented visual supervision. SpatialSV performs 2D-to-3D lifting of MLLMs’ multi-layer visual features to align with explicit 3D representations, with the 3D-lifting results serving as intuitive and interpretable proxy of models’ internal representations. Extensive experiments across multiple models and benchmarks demonstrate SpatialSV’s effectiveness in enhancing and interpreting spatial intelligence. Semi-supervised learning results further validate SpatialSV’s potential to leverage unlabeled visual data for scalable spatial representation learning.

## A Technical Details about SpatialSV

## A.1 Construction of Visual Supervision Signals

We leverage the off-the-shelf 3D vision foundation models (VFMs) to obtain 3D representations, including 3D feature maps, depth maps, ray maps and point cloud maps, to serve as visual supervision signals for SpatialSV. The 3D VFMs we use include VGGT [Wang et al., 2025b] and DepthAnythingv3 [Lin et al., 2025]. Specifically, we input multi-view images $\{ I _ { i } \} _ { i = 1 } ^ { N } , I _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ into a 3D VFM $\mathcal { F } _ { \mathrm { 3 D } }$ to obtain 3D features, depth maps, and the camera parameters:

$$
\left(\left\{\boldsymbol {f} _ {i} ^ {l} \right\} _ {l = 1} ^ {L _ {\mathrm{3D}}}, \boldsymbol {y} _ {i} ^ {\text { dep. }}, \boldsymbol {g} _ {i}\right) _ {i = 1} ^ {N _ {v}} = \mathcal {F} _ {\mathrm{3D}} \left(\left\{I _ {i} \right\} _ {i = 1} ^ {N _ {v}}\right), \tag {7}
$$

where $\{ f _ { i } ^ { l } \} _ { l = 1 } ^ { L _ { 3 \mathrm { D } } }$ and ydi $\pmb { y } _ { i } ^ { \mathrm { d e p . } } \in \mathbb { R } ^ { H \times W \times 1 }$ denote the 3D features and depth map corresponding to the i-th input image; ${ \pmb g } _ { i } = [ { \bf q } _ { i } , { \bf t } _ { i } , { \bf \dot { f } } _ { i } ] \in \mathbb { R } ^ { 9 }$ denotes the camera parameters, which is the concatenation of the rotation quaternion $\mathbf { q } _ { i } \in \mathbb { R } ^ { 4 }$ , the translation vector $\mathbf { t } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , and the field of view $\mathbf { f } _ { i } ~ \in ~ \mathbb { R } ^ { 2 }$ . We can recover the the standard intrinsic matrix Ki from $\mathbf { f } _ { i }$ and the principal point which is fixed at the image center $\textstyle { \bigl ( } { \frac { W } { 2 } } , { \frac { H } { 2 } } { \bigr ) }$ $\mathbf { R } _ { i } ^ { - } = \mathbf { \bar { \mathit { R o t } } } ( \mathbf { q } _ { i } )$ and the camera center $\mathbf { t } _ { i }$ .

Following [Huang et al., 2025a], we extract the final-layer 3D features $f ^ { L _ { 3 \mathrm { D } } }$ as distillation supervision signals. Due to the orthogonality constraint of rotation matrices which may lead to unstable optimization, we do not directly lift the 2D visual features of MLLMs into camera parameters. Instead, we follow [Lin et al., 2025] to adopt ray maps as a substitute, which reduces the learning difficulty of the prediction task.

To obtain the per-view ground-truth ray map ${ \pmb y } _ { i } ^ { r a y . } \in \qquad $ $\mathbb { R } ^ { H \times W \times 6 }$ , we employ the camera parameters to compute the per-pixel ray direction:

$$
\mathbf {d} _ {i} = \mathbf {R} _ {i} \mathbf {K} _ {i} ^ {- 1} \mathbf {p} (u, v) \in \mathbb {R} ^ {H \times W \times 3}, \tag {8}
$$

where $\mathbf { p } ( u , v ) = [ u , v , 1 ] ^ { T }$ is the homogeneous pixel coordi-$\mathbf { \Delta } y _ { i } ^ { r a y . } = [ \mathbf { t } _ { i } , \mathbf { d } _ { i } ]$ the per-view ground-truth point cloud map via:

$$
\boldsymbol {y} _ {i} ^ {\text { poi. }} (u, v) = \boldsymbol {t} _ {i} + \boldsymbol {y} _ {i} ^ {\text { dep. }} (u, v) \cdot \boldsymbol {d} _ {i} (u, v). \tag {9}
$$

Consequently, the 3D outputs $\{ \pmb { y } _ { i } ^ { \mathrm { d e p . } } , \pmb { y } _ { i } ^ { \mathrm { r a y . } } , \pmb { y } _ { i } ^ { \mathrm { p o i . } } \} _ { i = 1 } ^ { N }$ } i=1 serve as the task-oriented visual supervision signals of SpatialSV.

## A.2 3D Task-Oriented Loss Functions

In this section, we detail the computation procedure of loss terms present in $\operatorname { E . q . } \ ( 5 )$ of the main paper, namely the confidence loss ${ \mathcal { L } } _ { \mathrm { c o n f } } ,$ the depth-map gradiant loss $\mathcal { L } _ { \mathrm { g r a d } } .$ , and the point-cloud normal loss $\bar { \mathcal { L } } _ { \mathrm { { n o r m } } }$ .

The confidence loss term aims to encourage high influence only at reliable regions and down-weight uncertain regions, which can be computed as:

$$
\begin{array}{l} \mathcal {L} _ {\text { conf }} (\boldsymbol {y} _ {3 \mathrm{D}} ^ {t}, \boldsymbol {m} _ {3 \mathrm{D}} ^ {t}; \boldsymbol {c} ^ {t})) = \\ \frac {1}{| \Omega |} \sum_ {i \in \Omega} (| | \boldsymbol {y} _ {\mathrm{3D}} ^ {t, i} - \boldsymbol {m} _ {\mathrm{3D}} ^ {t, i} | | _ {2} \cdot \boldsymbol {c} ^ {t} - \lambda_ {c} \cdot \log (\boldsymbol {c} ^ {t})), \tag {10} \\ \end{array}
$$

where Ω is the set of valid pixels; $| | \cdot | | _ { 2 }$ is the $\mathcal { L } _ { 2 }$ norm; and $\lambda _ { c }$ is a regularization weight that prevent the model from predicting infinite uncertainty to minimize the loss.

The depth-map gradient loss encourages the predicted depth map to respect local depth variation, preserving edge structures and reducing depth ambiguity across neighbors, which can be formulated as:

$$
\mathcal {L} _ {\text { grad }} = \sum_ {k \in \{x, y \}} \frac {1}{| \Omega |} \sum_ {i \in \Omega} \left(| \nabla_ {k} \Delta_ {i} | \cdot \boldsymbol {c} ^ {t} - \lambda_ {c} \cdot \log (\boldsymbol {c} ^ {t})\right) \tag {11}
$$

where $\Delta _ { i } = \boldsymbol { y } _ { \mathrm { 3 D } } ^ { t , i } - m _ { \mathrm { 3 D } } ^ { t , i }$ denotes the difference map; $\nabla _ { x }$ and $\nabla _ { y }$ are horizontal and vertical spatial gradients, which ensure the local structure of the prediction matches the ground-truth even if the absolute depth is incorrect.

The point-cloud normal loss enforces geometry consistency of the reconstructed surface by aligning per-pixel surface normals derived from the prediction with the groundtruth, which can be computed via:

$$
\mathcal {L} _ {\text {normal}} = \frac {1}{| \Omega |} \sum_ {i \in \Omega} ((1 - \left\langle \mathbf {n} (\boldsymbol {m} _ {3 \mathrm{D}} ^ {t}) _ {i}, \mathbf {n} (\boldsymbol {y} _ {3 \mathrm{D}} ^ {t}) _ {i} \right\rangle) \cdot \boldsymbol {c} ^ {t} - \tag {12}
$$

$$
\lambda_ {c} \cdot \log (\pmb {c} ^ {t}))
$$

where $\langle \cdot , \cdot \rangle$ denotes the dot product, with the term $1 - \langle \mathbf { n } _ { 1 } , \mathbf { n } _ { 2 } \rangle$ approximating $1 - c o s ( \theta )$ and penalizing angular deviation.

## B Implementation Details of Experiments

## B.1 Dataset Details

In this section, we detail the information of all datasets used in our experiments, including eight spatial understanding benchmarks, namely MindCube [Yin et al., 2025], VSI-Bench [Yang et al., 2025a], Ego3D-Bench [Gholami et al., 2025], Spatial457 [Wang et al., 2025d], ViewSpatial-Bench [Li et al., 2025a], 3DSR-Bench [Ma et al., 2025], SP-Bench [Li et al., 2025c], TopViewRS [Li et al., 2024b], along with two general benchmarks, CVBench [Zhu et al., 2025d] and MMBench [Liu et al., 2024].

MindCube targets key challenges such as cross-view object consistency and reasoning about occluded or invisible objects, and comprises three types of camera motion: Rotation, Among, and Around. In our experiments, we use the 10k training samples in MindCube to fine-tune the models, and leverage the 1.2k samples in MindCube-Tiny for evaluation.

VSI-Bench emphasizes the understanding of spatial relations and multi-object correspondence, covering spatial relation tasks $( \mathrm { e . g . }$ , relative distance and relative direction), spatiotemporal tasks (e.g., object appearance order), and complex spatial manipulation tasks (e.g., route planning). We use its multiple-choice set for evaluation.

Ego3D-Bench is designed to evaluate spatial reasoning from ego-centric, multi-view outdoor data. It comprises over 8.6k QA pairs across five tasks spanning absolute distance, relative distance, localization, motion reasoning, and travel time. We use its multiple-choice set for evaluation.

Spatial457 is a synthetic benchmark that emphasizes four core capabilities spanning multi-object recognition, 2D locations, 3D locations, and 3D orientation. It introduces three spatial relation types—2D spatial relations in the camera view, 6D spatial relations that combine 3D position and orientation from a target object’s perspective, and collision relations for forward/backward motion—captured in seven question types across five difficulty levels.

ViewSpatial-Bench focuses on multi-perspective spatial localization. It comprises 5.7k QA pairs across 1.3k scenes, covering task types that test spatial localization from both camera and human viewpoints.

3DSR-Bench features 12 question types that probe 3D properties such as object height, inter-object location, orientation, and multi-object relations, emphasizing 3D grounding, camera extrinsics, object poses, depth, and multi-object reasoning.

SP-Bench is a two-part dataset consisting of SPBench-SI (Single-Image) and SPBench-MV (Multi-View). It targets four spatial reasoning tasks—absolute distance, relative distance, object size, and relative direction—with questions provided in numerical and multiple-choice formats. We employ SPBench-SI for evaluation.

TopViewRS targets top-view spatial understanding through 11k multiple-choice questions posed on indoor top-view maps. It defines four tasks spanning top-view recognition, top-view localization, static spatial seasoning and dynamic spatial reasoning.

CVBench is designed to evaluate cross-video relational reasoning, comprising 1.3k diversified videos accompanied by 1k carefully crafted multiple-choice QA pairs that span three hierarchical tiers: cross-video object association, crossvideo event association and cross-video complex reasoning.

MMBench comprises 3.2k data samples across 20 leaf abilities organized under a hierarchical taxonomy that spans perception (coarse and fine-grained, including single- and cross-instance) and reasoning (e.g., attribute, spatial, social, nature, physical, logic, plus complex structuralized imagetext understanding and future prediction).

## B.2 Detailed Hyperparameter Settings

We apply SpatialSV to 8 MLLMs across 4 model families, including Qwen2.5-VL-3B, Qwen2.5-VL-7B [Bai et al., 2025], LLaVA-NeXT-Video-7B [Zhang et al., 2024], LLaVA-OneVision-1.5-4B, LLaVA-OneVision-1.5-8B [Li et al., 2024a], InternVL3-1B, InternVL3-2B, and InternVL3- 8B [Zhu et al., 2025c]. For all models, we fine-tune the vision–language projector, the large language model, all 2D-to-3D projectors, and the task-decoupled DPT modules, while freezing the visual encoder. All models are optimized using AdamW with a batch size of 16 and a warm-up ratio of 0.03. Table 6 presents the settings of learning rates and supervised visual layer indexes for each model. All experiments are conducted on 8 NVIDIA H800 GPUs.

## C More Experimental Results

## C.1 Comparison with Other Supervision Methods

As shown in Table 8 , we present a detailed comparison with other supervision paradigms, especially the distillation paradigm based on 3D VFM features [Huang et al., 2025a], across 8 MLLMs on MindCube-Tiny and VSI-Bench. Notably, SpatialSV consistently outperforms both the pure-text and feature distillation variants, indicating that SpatialSV can inject robust spatial awareness into the models via 3D taskoriented fine-grained spatial constraints.

<table><tr><td>Model</td><td> $lr_{llm}$ </td><td> $lr_{mm}$ </td><td> $lr_{3D}$ </td><td>Supervised Visual Layers</td></tr><tr><td>Qwen2.5-VL-3B</td><td>2e-7</td><td>1e-5</td><td>1e-5</td><td>(4, 12, 20, 28)</td></tr><tr><td>Qwen2.5-VL-7B</td><td>2e-7</td><td>1e-5</td><td>1e-5</td><td>(3, 10, 17, 24)</td></tr><tr><td>LLaVA-OV-1.5-4B</td><td>2e-7</td><td>1e-5</td><td>1e-5</td><td>(4, 12, 20, 28)</td></tr><tr><td>LLaVA-OV-1.5-8B</td><td>2e-7</td><td>1e-5</td><td>1e-5</td><td>(4, 12, 20, 28)</td></tr><tr><td>LLaVA-NV-7B</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>(4, 11, 18, 25)</td></tr><tr><td>InternVL3-1B</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>(3, 9, 15, 21)</td></tr><tr><td>InternVL3-2B</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>(3, 9, 15, 21)</td></tr><tr><td>InternVL3-3B</td><td>1e-5</td><td>1e-5</td><td>1e-5</td><td>(3, 9, 15, 21)</td></tr></table>

Table 6: The settings of learning rates and supervised visual layers for different MLLMs. $l r _ { \mathrm { { l l m } } } , l r _ { \mathrm { { m m } } } ,$ , and $\boldsymbol { l } \boldsymbol { r } _ { 3 \mathrm { D } }$ denote the learning rates of the language model, the vision-language projector, and the 2D-to-3D components including projectors and DPT heads, respectively.

![](images/fe1d18b9a48d3c26c04d13c039d92b0f6397d23d74ac5595b4079c13549ef917.jpg)

<details>
<summary>text_image</summary>

Question: From the viewpoint presented in image 4,
what is to the right of the red chair and plastic container?
A. Leather loveseat with three seat cushions B. TV
C. Two single sofas D. Leather loveseat
Qwen2.5VL-7B + SpatialSV:
C. Two single sofas
LLaVA-NeXT-Video-7B + SpatialSV:
A. Leather loveseat with three seat cushions
Qwen2.5VL-3B + SpatialSV:
D. Leather loveseat
InternVL-2B + SpatialSV:
C. Two single sofas
</details>

Figure 7: Qualitative results on MindCube-Tiny (Example1).

## C.2 Ablation on the Selection of 3D VFMs

In this section, we conduct an ablation on the selection of 3D VFMs that are leveraged to obtain the visual supervision signals. As shown in Table 7, we compare the performance of Qwen2.5-VL-3B and LLaVA-NeXT-Video-7B when VGGT and DepthAnything-v3 are respectively used as sources of the target representations. Both configurations exhibit comparable performance, indicating the robustness of our approach to the source of supervision signals.

## C.3 More Qualitative Results

We provide additional qualitative results that demonstrate varying internal representation capabilities and VQA performance of different MLLMs on specific samples. The qualitative results are shown in Figure 7-9.

![](images/42f478fd09868775374c4e47fd8f9d001e14dfe3a3d55f3a4acb65a7e3772093.jpg)  
Figure 8: Qualitative results on MindCube-Tiny (Example2).

<table><tr><td rowspan="2">MLLM</td><td rowspan="2">3D VFM</td><td colspan="4">MindCube-Tiny</td></tr><tr><td>Rotation</td><td>Among</td><td>Around</td><td>Overall</td></tr><tr><td rowspan="2">Qwen2.5VL-3B</td><td>VGGT</td><td>34.5</td><td>56.8</td><td>83.5</td><td>62.0</td></tr><tr><td>DAMv3</td><td>36.0</td><td>56.2</td><td>84.5</td><td>62.3</td></tr><tr><td rowspan="2">LLaVA-NV-7B</td><td>VGGT</td><td>38.0</td><td>62.8</td><td>82.8</td><td>65.3</td></tr><tr><td>DAMv3</td><td>39.5</td><td>61.8</td><td>82.3</td><td>64.9</td></tr></table>

Table 7: Ablation on the selection of 3D VFMs.

## D Related Work

The significant advancements of multimodal large language models [Bai et al., 2025; Zhu et al., 2025c; Zhou et al., 2026] in visual understanding and reasoning has motivated growing efforts to endow them with strong Spatial Intelligence. which is crucial for real-world applications such as autonomous driving [Zhou et al., 2025; Huang et al., 2025b; Tang et al., 2026] and embodied interaction [Hu et al., 2025; Song et al., 2025; Yang et al., 2025c]. According to the way of incorporating spatial knowledge, existing methods can be broadly categorized into two lines of research: external spatial prior–based approaches and internalized spatial awareness–based approaches.

External spatial prior–based approaches aim to compensate for the limited intrinsic spatial awareness of MLLMs by constructing external inputs that encode rich spatial priors. Among these, prompt-based methods rely on auxiliary models or tools to generate various forms of linguistic or visual prompts, including chains of thought [Zhang et al., 2025], cognitive maps [Yang et al., 2025a; Gholami et al., 2025], object markers [Qi et al., 2025], bird’s-eye-view (BEV) maps [Zhu et al., 2025b], motion trajectories [Li et al., 2025d], and 3D bounding boxes [Liu et al., 2025b]. For example, Ego3D-VLM [Gholami et al., 2025] employs off-the-shelf referring expression detection and depth estimation models to construct cognitive maps as model inputs. MindJourney [Yang et al., 2025b] leverages multi-round interactions between a video diffusion model and a VLM to generate viewpoints containing spatial evidence as prompts. However, such methods rely heavily on the performance of external models, making them sensitive to upstream errors while significantly increasing inference overhead. Another line of approaches within this paradigm explicitly incorporate 3D inputs through multimodal fusions. [Xu et al., 2024; Chen et al., 2024b] require explicit point cloud encoding and alignment with the textual modality, while [Zheng et al., 2025; Zhu et al., 2025a] transform point clouds into 3D positional encodings. [Cheng et al., 2024; Liu et al., 2025a] encode depth maps as additional inputs. However, these methods rely on highly customized modules and precise cross-modal alignment while also increases computational costs, posing challenges for practical deployment.

![](images/da55c9484227e0987e23a5f3f607d408c9e6ad03c25d74de2a93933e03cee036.jpg)  
Figure 9: Qualitative results on MindCube-Tiny (Example3).

Internalized spatial awareness–based approaches emphasize the learning of internal spatial representation within MLLMs. Early approaches [Chen et al., 2024a] leverage large-scale 2D spatial vision–language datasets and train models solely with textual supervision, which proves insufficient for injecting spatial awareness due to the absence of 3D prior guidance. Ross3D [Wang et al., 2025a] introduces masked cross-view and global-view reconstruction in the latent space as auxiliary supervision, which remains confined to the 2D domain and struggle with the learning of 3D representations. More recent approaches [Huang et al., 2025a; Chen et al., 2025b], employ representations from 3D vision foundation models (VFMs) [Wang et al., 2025b;

<table><tr><td rowspan="2">Method</td><td colspan="4">MindCube-Tiny</td><td colspan="7">VSI-Bench</td></tr><tr><td>Rotation</td><td>Among</td><td>Around</td><td>Overall↑</td><td>Rel.Dir. Hard</td><td>Rel.Dir. Medium</td><td>Rel.Dir. Easy</td><td>Rel. Dist.</td><td>App. Order</td><td>Route Plan</td><td>Overall↑</td></tr><tr><td colspan="12">Baseline</td></tr><tr><td>Chance Level (Random)</td><td>26.5</td><td>24.5</td><td>23.8</td><td>24.6</td><td>26.5</td><td>25.7</td><td>24.9</td><td>24.1</td><td>23.3</td><td>28.4</td><td>27.7</td></tr><tr><td>Chance Level (Frequency)</td><td>34.5</td><td>34.8</td><td>33.3</td><td>34.3</td><td>25.2</td><td>33.6</td><td>50.2</td><td>25.1</td><td>25.2</td><td>29.4</td><td>29.0</td></tr><tr><td colspan="12">Qwen2.5-VL Family</td></tr><tr><td>Qwen2.5-VL-3B</td><td>34.5</td><td>36.0</td><td>32.3</td><td>34.5</td><td>34.3</td><td>33.3</td><td>29.0</td><td>31.7</td><td>22.7</td><td>27.8</td><td>29.6</td></tr><tr><td>Qwen2.5-VL-3B + Text.</td><td>33.5</td><td>49.7</td><td>74.8</td><td>55.3</td><td>35.7</td><td>35.7</td><td>26.3</td><td>31.1</td><td>23.6</td><td>29.9</td><td>30.1</td></tr><tr><td>Qwen2.5-VL-3B + Distillation.</td><td>34.5</td><td>51.5</td><td>81.3</td><td>58.6</td><td>35.9</td><td>34.1</td><td>30.9</td><td>32.1</td><td>22.8</td><td>32.0</td><td>30.6</td></tr><tr><td>Qwen2.5-VL-3B + SpatialSV</td><td>36.0</td><td>56.2</td><td>84.5</td><td>62.3</td><td>34.6</td><td>37.8</td><td>34.6</td><td>33.5</td><td>23.3</td><td>37.1</td><td>32.2</td></tr><tr><td>Qwen2.5-VL-7B</td><td>35.0</td><td>29.7</td><td>26.8</td><td>29.6</td><td>28.4</td><td>24.9</td><td>45.2</td><td>31.8</td><td>29.5</td><td>31.4</td><td>30.8</td></tr><tr><td>Qwen2.5-VL-7B + Text.</td><td>35.5</td><td>51.0</td><td>73.5</td><td>55.9</td><td>27.4</td><td>31.2</td><td>51.2</td><td>31.4</td><td>30.1</td><td>34.5</td><td>32.4</td></tr><tr><td>Qwen2.5-VL-7B + Distillation.</td><td>35.0</td><td>53.8</td><td>76.5</td><td>58.3</td><td>29.0</td><td>32.5</td><td>50.2</td><td>34.8</td><td>29.1</td><td>37.6</td><td>33.7</td></tr><tr><td>Qwen2.5-VL-7B + SpatialSV</td><td>37.5</td><td>58.5</td><td>81.8</td><td>62.8</td><td>29.5</td><td>35.2</td><td>48.4</td><td>37.2</td><td>30.1</td><td>42.8</td><td>35.4</td></tr><tr><td colspan="12">LLaVA-OneVision-1.5 Family</td></tr><tr><td>LLaVA-OneVision-1.5-4B</td><td>26.5</td><td>35.2</td><td>32.3</td><td>32.8</td><td>37.3</td><td>35.5</td><td>46.1</td><td>37.0</td><td>24.3</td><td>29.9</td><td>33.9</td></tr><tr><td>LLaVA-OneVision-1.5-4B + Text.</td><td>35.0</td><td>49.0</td><td>38.3</td><td>43.1</td><td>38.9</td><td>34.1</td><td>48.9</td><td>38.6</td><td>25.1</td><td>30.9</td><td>34.9</td></tr><tr><td>LLaVA-OneVision-1.5-4B + Distillation.</td><td>34.5</td><td>51.2</td><td>42.3</td><td>45.4</td><td>40.0</td><td>35.5</td><td>47.9</td><td>39.0</td><td>24.6</td><td>33.0</td><td>35.3</td></tr><tr><td>LLaVA-OneVision-1.5-4B + SpatialSV</td><td>34.5</td><td>56.5</td><td>51.5</td><td>51.2</td><td>40.5</td><td>35.2</td><td>52.5</td><td>42.0</td><td>25.4</td><td>36.1</td><td>37.1</td></tr><tr><td>LLaVA-OneVision-1.5-8B</td><td>33.5</td><td>31.0</td><td>32.5</td><td>32.0</td><td>28.7</td><td>34.9</td><td>48.9</td><td>36.1</td><td>28.8</td><td>28.9</td><td>33.5</td></tr><tr><td>LLaVA-OneVision-1.5-8B + Text.</td><td>35.0</td><td>49.5</td><td>55.5</td><td>49.1</td><td>34.1</td><td>35.5</td><td>52.1</td><td>37.3</td><td>29.9</td><td>30.4</td><td>35.5</td></tr><tr><td>LLaVA-OneVision-1.5-8B + Distillation.</td><td>36.5</td><td>52.7</td><td>58.3</td><td>51.8</td><td>36.5</td><td>35.5</td><td>50.2</td><td>38.3</td><td>28.3</td><td>32.0</td><td>35.7</td></tr><tr><td>LLaVA-OneVision-1.5-8B + SpatialSV</td><td>38.0</td><td>57.5</td><td>60.3</td><td>55.2</td><td>38.3</td><td>36.5</td><td>49.3</td><td>42.1</td><td>29.6</td><td>40.2</td><td>38.1</td></tr><tr><td colspan="12">LLaVA-NeXT-Video Family</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>34.5</td><td>41.7</td><td>34.5</td><td>38.1</td><td>27.1</td><td>32.5</td><td>50.7</td><td>35.9</td><td>27.5</td><td>31.4</td><td>32.9</td></tr><tr><td>LLaVA-NeXT-Video-7B + Text.</td><td>33.0</td><td>52.3</td><td>78.8</td><td>57.9</td><td>27.9</td><td>35.7</td><td>45.6</td><td>35.5</td><td>29.8</td><td>32.5</td><td>33.6</td></tr><tr><td>LLaVA-NeXT-Video-7B + Distillation.</td><td>36.5</td><td>55.2</td><td>79.5</td><td>60.2</td><td>29.5</td><td>33.6</td><td>48.4</td><td>37.5</td><td>29.5</td><td>34.5</td><td>34.4</td></tr><tr><td>LLaVA-NeXT-Video-7B + SpatialSV</td><td>39.5</td><td>61.8</td><td>82.3</td><td>64.9</td><td>29.2</td><td>33.9</td><td>52.5</td><td>37.8</td><td>30.4</td><td>44.9</td><td>35.9</td></tr><tr><td colspan="12">InternVL3 Family</td></tr><tr><td>InternVL3-1B</td><td>31.5</td><td>37.3</td><td>32.0</td><td>34.6</td><td>30.8</td><td>33.9</td><td>47.0</td><td>26.6</td><td>24.6</td><td>33.5</td><td>30.2</td></tr><tr><td>InternVL3-1B + Text.</td><td>33.5</td><td>52.3</td><td>80.0</td><td>58.4</td><td>29.0</td><td>34.7</td><td>44.7</td><td>27.3</td><td>25.4</td><td>30.9</td><td>30.0</td></tr><tr><td>InternVL3-1B + Distillation.</td><td>35.5</td><td>53.5</td><td>77.0</td><td>58.3</td><td>29.8</td><td>35.2</td><td>43.8</td><td>28.2</td><td>25.1</td><td>32.0</td><td>30.4</td></tr><tr><td>InternVL3-1B + SpatialSV</td><td>35.0</td><td>56.8</td><td>78.5</td><td>60.4</td><td>30.0</td><td>34.7</td><td>50.2</td><td>28.2</td><td>24.9</td><td>38.7</td><td>31.4</td></tr><tr><td>InternVL3-2B</td><td>31.0</td><td>33.3</td><td>24.8</td><td>30.1</td><td>26.5</td><td>31.5</td><td>48.4</td><td>30.1</td><td>24.4</td><td>34.0</td><td>30.3</td></tr><tr><td>InternVL3-2B + Text.</td><td>32.0</td><td>54.3</td><td>78.0</td><td>58.5</td><td>29.0</td><td>34.1</td><td>45.6</td><td>32.0</td><td>30.1</td><td>29.9</td><td>32.4</td></tr><tr><td>InternVL3-2B + Distillation.</td><td>35.0</td><td>56.5</td><td>79.5</td><td>60.6</td><td>29.5</td><td>33.3</td><td>47.0</td><td>33.0</td><td>28.0</td><td>31.4</td><td>32.4</td></tr><tr><td>InternVL3-2B + SpatialSV</td><td>37.5</td><td>57.2</td><td>82.8</td><td>62.4</td><td>31.4</td><td>35.2</td><td>47.9</td><td>35.1</td><td>28.8</td><td>41.2</td><td>34.6</td></tr><tr><td>InternVL3-8B</td><td>34.5</td><td>39.3</td><td>33.3</td><td>36.5</td><td>23.6</td><td>34.1</td><td>46.1</td><td>34.1</td><td>32.9</td><td>32.5</td><td>33.1</td></tr><tr><td>InternVL3-8B + Text.</td><td>33.5</td><td>68.8</td><td>82.5</td><td>67.5</td><td>25.7</td><td>33.6</td><td>50.2</td><td>34.5</td><td>32.0</td><td>29.9</td><td>33.5</td></tr><tr><td>InternVL3-8B + Distillation.</td><td>35.5</td><td>70.8</td><td>81.5</td><td>68.5</td><td>28.4</td><td>36.0</td><td>47.5</td><td>35.5</td><td>31.7</td><td>34.0</td><td>34.5</td></tr><tr><td>InternVL3-8B + SpatialSV</td><td>38.5</td><td>77.2</td><td>86.0</td><td>73.7</td><td>30.6</td><td>37.3</td><td>50.7</td><td>36.2</td><td>33.2</td><td>42.8</td><td>36.6</td></tr></table>

Table 8: Detailed comparison of our approach (SpatialSV) with the baseline models, the pure-text supervision variant, and the feature distillation variant on MindCube-Tiny and VSI-Bench across 8 MLLMs. The best results within each model type are bolded.

Lin et al., 2025] as supervision signals for feature distillation. Nevertheless, imitation at the feature level is inherently coarse-grained and lacks structured and explicit geometric constraints. Moreover, the intrinsic representations learned by these methods remain uninterpretable, which hinders a deeper understanding of the spatial mental modeling mechanisms within MLLMs.

Our approach follows the second paradigm, focusing on internalizing spatial awareness in MLLMs to maintain a streamlined model architecture and facilitate practical deployment.

More importantly, we introduce explicit 3D representations as supervision signals, enabling task-oriented, fine-grained spatial constraints while preserving the interpretability of the internalized spatial awareness.

## References

[Bai et al., 2025] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.  
[Chen et al., 2024a] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.  
[Chen et al., 2024b] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26428–26438, 2024.  
[Chen et al., 2025a] Yue Chen, Xingyu Chen, Anpei Chen, Gerard Pons-Moll, and Yuliang Xiu. Feat2gs: Probing visual foundation models with gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6348–6361, 2025.  
[Chen et al., 2025b] Zhangquan Chen, Manyuan Zhang, Xinlei Yu, Xufang Luo, Mingze Sun, Zihao Pan, Yan Feng, Peng Pei, Xunliang Cai, and Ruqi Huang. Think with 3d: Geometric imagination grounded spatial reasoning from limited views. arXiv preprint arXiv:2510.18632, 2025.  
[Cheng et al., 2024] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.  
[El Banani et al., 2024] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3d awareness of visual foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21795–21806, 2024.  
[Gholami et al., 2025] Mohsen Gholami, Ahmad Rezaei, Zhou Weimin, Sitong Mao, Shunbo Zhou, Yong Zhang, and Mohammad Akbari. Spatial reasoning with visionlanguage models in ego-centric multi-view scenes. arXiv preprint arXiv:2509.06266, 2025.  
[Hu et al., 2025] Wenbo Hu, Yining Hong, Yanjun Wang, Leison Gao, Zibu Wei, Xingcheng Yao, Nanyun Peng, Yonatan Bitton, Idan Szpektor, and Kai-Wei Chang. 3dllm-mem: Long-term spatial-temporal memory for embodied 3d large language model. arXiv preprint arXiv:2505.22657, 2025.  
[Huang et al., 2025a] Xiaohu Huang, Jingjing Wu, Qunyi Xie, and Kai Han. Mllms need 3d-aware representation supervision for scene understanding. arXiv preprint arXiv:2506.01946, 2025.  
[Huang et al., 2025b] Zhijian Huang, Chengjian Feng, Feng Yan, Baihui Xiao, Zequn Jie, Yujie Zhong, Xiaodan Liang, and Lin Ma. Robotron-drive: All-in-one large multimodal model for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8011–8021, 2025.  
[Johnson-Laird, 1980] Philip N Johnson-Laird. Mental models in cognitive science. Cognitive science, 4(1):71–115, 1980.  
[Johnson-Laird, 1983] Philip Nicholas Johnson-Laird. Mental models: Towards a cognitive science of language, inference, and consciousness. Number 6. Harvard University Press, 1983.  
[Li et al., 2024a] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.  
[Li et al., 2024b] Chengzu Li, Caiqi Zhang, Han Zhou, Nigel Collier, Anna Korhonen, and Ivan Vulic. Topviewrs: ´ Vision-language models as top-view spatial reasoners. arXiv preprint arXiv:2406.02537, 2024.  
[Li et al., 2025a] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025.  
[Li et al., 2025b] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model. arXiv preprint arXiv:2510.12276, 2025.  
[Li et al., 2025c] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in visionlanguage models. arXiv preprint arXiv:2510.08531, 2025.  
[Li et al., 2025d] Pengteng Li, Pinhao Song, Wuyang Li, Weiyu Guo, Huizai Yao, Yijie Xu, Dugang Liu, and Hui Xiong. See&trek: Training-free spatial prompting for multimodal large language model. arXiv preprint arXiv:2509.16087, 2025.  
[Lin et al., 2025] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.  
[Liu et al., 2024] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your  
multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.  
[Liu et al., 2025a] Yang Liu, Ming Ma, Xiaomin Yu, Pengxiang Ding, Han Zhao, Mingyang Sun, Siteng Huang, and Donglin Wang. Ssr: Enhancing depth perception in visionlanguage models via rationale-guided spatial reasoning. arXiv preprint arXiv:2505.12448, 2025.  
[Liu et al., 2025b] Yifan Liu, Fangneng Zhan, Kaichen Zhou, Yilun Du, Paul Pu Liang, and Hanspeter Pfister. Abstract 3d perception for spatial intelligence in visionlanguage models. arXiv preprint arXiv:2511.10946, 2025.  
[Ma et al., 2025] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025.  
[Qi et al., 2025] Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428, 2025.  
[Ranftl et al., 2021] Rene Ranftl, Alexey Bochkovskiy, and ´ Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.  
[Song et al., 2025] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15768–15780, 2025.  
[Tang et al., 2026] Jiayu Tang, Yuchen Zhou, Chen Xiong, and Chao Gou. Letp: Coupling attention localization and cognitive reasoning for ego-centric multi-task driving scene perception. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 19797–19801. IEEE, 2026.  
[Wang et al., 2025a] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. arXiv preprint arXiv:2504.01901, 2025.  
[Wang et al., 2025b] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.  
[Wang et al., 2025c] Xiaoyan Wang, Zeju Li, Yifan Xu, Jiaxing Qi, Zhifei Yang, Ruifei Ma, Xiangde Liu, and Chao Zhang. Spatial 3d-llm: Exploring spatial awareness in 3d vision-language models. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2025.  
[Wang et al., 2025d] Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24669–24679, 2025.  
[Xu et al., 2024] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In European Conference on Computer Vision, pages 131–147. Springer, 2024.  
[Yang et al., 2025a] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.  
[Yang et al., 2025b] Yuncong Yang, Jiageng Liu, Zheyuan Zhang, Siyuan Zhou, Reuben Tan, Jianwei Yang, Yilun Du, and Chuang Gan. Mindjourney: Test-time scaling with world models for spatial reasoning. arXiv preprint arXiv:2507.12508, 2025.  
[Yang et al., 2025c] Yuncong Yang, Han Yang, Jiachen Zhou, Peihao Chen, Hongxin Zhang, Yilun Du, and Chuang Gan. 3d-mem: 3d scene memory for embodied exploration and reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17294–17303, 2025.  
[Yin et al., 2025] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025.  
[Zhang et al., 2024] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.  
[Zhang et al., 2025] Haoyu Zhang, Meng Liu, Zaijing Li, Haokun Wen, Weili Guan, Yaowei Wang, and Liqiang Nie. Spatial understanding from videos: Structured prompts meet simulation data. arXiv preprint arXiv:2506.03642, 2025.  
[Zheng et al., 2025] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8995–9006, 2025.  
[Zhou et al., 2023] Yuchen Zhou, Guang Tan, Mengtang Li, and Chao Gou. Learning from easy to hard pairs: Multistep reasoning network for human-object interaction detection. In Proceedings of the 31st ACM International Conference on Multimedia, pages 4368–4377, 2023.  
[Zhou et al., 2025] Yuchen Zhou, Jiayu Tang, Xiaoyan Xiao, Yueyao Lin, Linkai Liu, Zipeng Guo, Hao Fei, Xiaobo  
Xia, and Chao Gou. Where, what, why: Towards explainable driver attention prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2675–2685, 2025.  
[Zhou et al., 2026] Yuchen Zhou, Jiayu Tang, Shuo Yang, Xiaoyan Xiao, Yuqin Dai, Wenhao Yang, Chao Gou, Xiaobo Xia, and Tat-Seng Chua. Logic unseen: Revealing the logical blindspots of vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 29062–29070, 2026.  
[Zhu et al., 2025a] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d capabilities. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4295–4305, 2025.  
[Zhu et al., 2025b] Fangrui Zhu, Hanhui Wang, Yiming Xie, Jing Gu, Tianye Ding, Jianwei Yang, and Huaizu Jiang. Struct2d: A perception-guided framework for spatial reasoning in large multimodal models. arXiv preprint arXiv:2506.04220, 2025.  
[Zhu et al., 2025c] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.  
[Zhu et al., 2025d] Nannan Zhu, Yonghao Dong, Teng Wang, Xueqian Li, Shengjun Deng, Yijia Wang, Zheng Hong, Tiantian Geng, Guo Niu, Hanyan Huang, et al. Cvbench: Benchmarking cross-video synergies for complex multimodal reasoning. arXiv preprint arXiv:2508.19542, 2025.