# Why Far Looks Up: Probing Spatial Representation in Vision-Language Models

Cheolhong Min1, Jaeyun Jung1, Daeun Lee1, Hyeonseong Jeon1, Yu Su2, Jonathan Tremblay3, Chan Hee Song3,†,‡, and Jaesik Park1,†

1 Seoul National University

2 The Ohio State University

3 NVIDIA

lusong@nvidia.com, {cheolhong.min, jaesik.park}@snu.ac.kr

Abstract. Vision-language models (VLMs) achieve strong performance on spatial reasoning benchmarks, yet it remains unclear whether this reflects structured 3D understanding or reliance on statistical shortcuts in natural images. We introduce a representation-level analysis framework that constructs minimal contrastive pairs to measure how spatial axes are organized and disentangled within VLM embeddings. Our analysis across multiple model families reveals a consistent vertical-distance entanglement: models conflate vertical image position with distance, mirroring the perspective bias of natural photographs. This bias produces a significant accuracy gap between perspective-consistent and counter-heuristic examples, and intensifies under data scaling even as overall benchmark accuracy improves. We further show that models with similar benchmark scores can exhibit different internal representations, and that these differences predict accuracy and robustness across diverse spatial reasoning benchmarks. To isolate this bias from evaluation-set skew, we introduce SpatialTunnel, a synthetic benchmark designed to expose spatial shortcut biases by removing common correlations present in natural images. Experiments suggest that the entanglement is model-intrinsic, and that models with well-separated spatial axes exhibit greater robustness, indicating that well-structured spatial representations lead to more reliable spatial reasoning across diverse benchmarks. Code and benchmark are available on the project website.

Keywords: Vision-Language Models · Spatial Understanding · Representation Analysis

# 1 Introduction

Spatial reasoning is a core capability for Vision-Language Models (VLMs), particularly as these systems are increasingly deployed in robotics [29, 32, 41, 54], embodied agents [1, 48, 51], and multimodal assistants [2, 24, 42] that observe and interact with physical environments. Although modern VLMs are primarily trained on 2D image–text pairs [3, 4,16,39], they achieve strong performance on spatial reasoning benchmarks [20, 23, 55], and recent work continues to improve these results through scaling and spatial training data [12, 13, 50, 53, 65]. These advances suggest that current models possess meaningful spatial understanding. However, it remains unclear whether strong benchmark accuracy reflects robust spatial reasoning or the exploitation of statistical regularities in natural images.

![](images/efa0c2c26c447952f533f27f01a76969d827d536541bdcd90bf7acdfb87ab889.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Existing datasets"] --> B["Bookcase"]
    B --> C["than"]
    C --> D["farther"]
    D --> E["or"]
    E --> F["closer"]
    F --> G["Is"]
    G --> H["Table"]
    H --> I["Is"]
    
    J["Weak Spatial VLMs"] --> K["Vertical-Distance Entanglement"]
    K --> L["far"]
    L --> M["I use 'above' as a proxy for 'far'"]
    
    N["Strong Spatial VLMs"] --> O["far"]
    O --> P["above"]
    P --> Q["I look for actual depth cues"]
    
    R["Responses"] --> S[": Table is closer. ✓"]
    S --> T[": Blue cube is farther. ✗"]
    
    U["Existing datasets"] --> V["Bookcase"]
    V --> W["than"]
    W --> X["farther"]
    X --> Y["or"]
    Y --> Z["closer"]
    Z --> AA["Is"]
    AA --> AB["Table"]
    AB --> AC["Is"]
    
    AD["Existing datasets"] --> AE["Bookcase"]
    AE --> AF["than"]
    AF --> AG["farther"]
    AG --> AH["or"]
    AH --> AI["closer"]
    AI --> AJ["Is"]
    AJ --> AK["Table"]
    AK --> AL["Is"]
    
    AM["Weak Spatial VLMs"] --> AN["Vertical-Distance Entanglement"]
    AN --> AO["far"]
    AO --> AP["I use 'above' as a proxy for 'far'"]
    
    AQ["Strong Spatial VLMs"] --> AR["Vertical-Distance Entanglement"]
    AR --> AS["far"]
    AS --> AT["I look for actual depth cues"]
    
    AU["Responses"] --> AV[": Table is closer. ✓"]
    AV --> AW[": Blue cube is closer. ✓"]
```
</details>

Fig. 1: Many VLMs answer spatial questions via a perspective-driven shortcut, e.g., objects located higher in the image are further away in 3D. By confusing 2D vertical position with 3D distance, models fail systematically on counter examples. Our SpatialTunnel benchmark and contrastive probing expose this vertical-distance entanglement. In contrast, strong spatial VLMs show disentangled axes and consistent correctness across both real and synthetic settings.

Many spatial relations can be partially inferred from correlations that arise naturally in photographic data rather than from explicit reasoning about 3D spatial structure. For example, perspective in everyday photographs introduces a consistent relationship between vertical image position and depth: objects appearing higher in the image are often farther from the camera, as in Figure 1. Such correlations allow models to rely on shortcuts that substitute vertical cues for depth reasoning, achieving high benchmark accuracy while internally conflating distinct spatial dimensions.

This limitation highlights a broader challenge in evaluating spatial understanding in VLMs. Behavioral benchmarks measure whether a model produces correct answers, but they provide limited insight into how those answers are obtained. Two models may achieve similar performance while relying on different internal mechanisms: one encoding spatial relations in a structured, separable manner, and another depending on correlated cues present in natural imagery which become brittle under distribution shift. Distinguishing these possibilities requires examining how spatial information is represented inside the model, rather than relying on output-level performance.

Recent work has revealed persistent spatial reasoning failures through controlled benchmarks [30, 61, 62] and has begun probing internal model behavior such as attention dynamics [11]. However, these efforts primarily assess individual task performance or local mechanisms, leaving the global geometric organization of spatial relations in representation space largely unexplored.

We address this gap from two complementary angles. First, we analyze how spatial relations along three core 3D axes – horizontal (left / right), vertical (above / below), and depth (close / far) – are organized within VLM internal embeddings, using controlled contrastive examples that vary only the spatial relation between objects while holding confounds such as object identity fixed. Second, we introduce SpatialTunnel, a synthetic benchmark designed to remove perspective-driven biases in spatial evaluation. Its tunnel geometry decouples vertical image position from depth, enabling balanced assessment beyond the correlations present in natural image benchmarks.

Across multiple VLM families, our experiments reveal that horizontal relations form stable, opposing directions in representation space, whereas vertical and depth relations are frequently entangled, suggesting reliance on perspectivedriven cues. Moreover, models with more structured spatial representations perform better across diverse spatial reasoning benchmarks, including EmbSpatial-Bench [20], CV-Bench [55], and BLINK [23]. Evaluations on SpatialTunnel further expose biases hidden under standard benchmark settings, and models with more structured representations exhibit greater robustness when these correlations are removed. Together, these results suggest that benchmark accuracy alone may overestimate the spatial reasoning capabilities of current VLMs. Our contributions are threefold:

– Representation-level analysis of spatial reasoning. We introduce a framework for analyzing how spatial relations are organized within VLM embeddings, diagnosing whether models encode structured spatial reasoning or rely on shortcut cues.

– Spatial representations predict robustness. We show that models with similar benchmark performance can exhibit markedly different internal spatial representations, and that models with more structured spatial representations exhibit greater robustness and generalization.

– A bias-controlled synthetic benchmark for spatial reasoning. We construct a synthetic dataset that decouples vertical image position from depth, revealing shortcut biases hidden under standard benchmark settings.

# 2 Related Work

Spatial Understanding Datasets and Benchmarks. Recent benchmarks have revealed persistent weaknesses in VLM spatial reasoning despite strong semantic performance. Controlled evaluations such as What’s Up [30] and COM-FORT [62] show that models frequently fail on basic positional distinctions and frame-of-reference consistency. To probe deeper spatial competence, subsequent work has expanded along several axes: egocentric and cross-video reasoning [20, 55], 6DoF diagnostic tasks [57], and multi-step spatial referring [65]. In parallel, simulation-based datasets [45, 50, 54] provide large-scale supervision for physical dynamics, yet spatial performance often plateaus with data scaling [61]. While these efforts effectively measure whether models succeed or fail, they do not examine what cues models rely on internally—in particular, none isolate the entanglement between vertical image position and perceived depth that arises from perspective projection. Our work targets this gap by constructing controlled synthetic environments and contrastive splits that systematically expose this bias.

Probing Internal Representations of Vision-Language Models. Recent work has moved beyond behavioral evaluation to examine the internal states of VLMs. Linear probing studies show that vision encoders inherently represent monocular depth cues [15] and bind geometric coordinates to object activations in early layers [31], while unified extraction frameworks [47] facilitate systematic comparison across model families. On the mechanistic side, ADAPTVIS [11] analyzes attention dynamics during spatial reasoning, and Spatial Forcing [37] explicitly aligns intermediate layers with 3D structure. However, these approaches primarily detect the presence of individual spatial primitives or adjust local attention behavior; they do not examine how different spatial dimensions are jointly organized—in particular, whether depth and vertical cues occupy separable or entangled directions in representation space. We address this gap through controlled contrastive analysis of internal embeddings, directly measuring the geometric relationship between spatial axes to reveal entanglement that isolated probing cannot detect.

# 3 Perspective Projection Bias in Spatial Understanding

Vision-language models are increasingly expected to reason about 3D spatial relationships from a single RGB image, e.g., answering questions such as “Is the chair closer to the camera than the table?” However, monocular images provide only a 2D projection of the 3D scene, requiring models to infer spatial structure from indirect visual cues. A central question is whether current VLMs genuinely learn such 3D reasoning, or instead rely on the visual cues that happen to correlate with depth in image space.

In this section, we analyze how VLMs perform spatial reasoning across multiple model families and benchmarks. Our analysis reveals a systematic bias arising from perspective projection: models frequently use an object’s vertical position in the image as a proxy for its distance from the camera. We term this phenomenon vertical-distance entanglement, where image-plane vertical position becomes conflated with depth. Across multiple models and benchmarks, we show that this bias consistently emerges and leads to systematic errors in spatial reasoning.

# 3.1 What is Vertical-Distance Entanglement?

Perspective projection and vertical position. From the observer’s viewpoint, objects farther away on a common ground surface appear higher in the image. This phenomenon gives rise to the classical elevation cue: for objects lying on the ground plane, those nearer to the horizon line are perceived as being farther from the observer [15] (see Appendix A for details).

Entanglement as a shortcut. We hypothesize that VLMs exploit this correla-

![](images/b897124c05df1ce8f506dacf7a988910b1105485f9c944173b2e4b9a5d635135.jpg)

<details>
<summary>text_image</summary>

Consistent
farther object
appears higher
VS
Counter
farther object
appears lower
</details>

Fig. 2: Consistent vs. counter examples. Consistent: Farther object appears higher in the image; Counter: Farther object appears lower.

tion as a shortcut: when asked about relative depth, they partially rely on the vertical positions of objects rather than reasoning about 3D structure. We refer to this phenomenon as vertical-distance entanglement, indicating the tendency of a model to treat above ≈ far and below ≈ close when answering depth-related questions.

Consistent and counter-heuristic examples. To systematically analyze this entanglement, we categorize depth-related samples into two groups, consistent and counter (Figure 2). The classification is based on whether the groundtruth spatial relationship aligns with the vertical-position heuristic.

We implement this by comparing the vertical center coordinates of the two queried objects in pixel space: if the farther object has a smaller y-coordinate (i.e., higher in the image), the example is consistent; otherwise, it is counter. If a model exhibits no entanglement, its accuracy should be comparable on both groups. Conversely, a systematic accuracy gap between the two groups would constitute evidence that the model relies on the vertical-position shortcut.

# 3.2 Experimental Setup

Models. We evaluate three VLM families spanning different architectures: Molmo-7B-O-0924 [16], NVILA-Lite-2B [40], and Qwen2.5-VL-3B-Instruct [4]. To analyze how spatial fine-tuning affects entanglement, we train variants of each model at multiple data scales (80k, 400k, 800k, and 2M samples); base models refer to the original pretrained weights without additional fine-tuning. We also include RoboRefer-2B-SFT [65], which shares the NVILA-Lite-2B base but is trained on more than 20M samples including RGB and RGB-D images, and Qwen3-VL-235B-A22B-Instruct [3] as a large-scale reference.

Training data. Recent work has attributed VLMs’ limited spatial understanding to a lack of spatial reasoning data during training, motivating several spatialfocused datasets [9, 19, 45, 50, 60, 65]. To study the effect of data scaling within and across model families, we uniformly mix five existing spatial understanding datasets (i.e., SAT [45], RoboSpatial [50], SPAR-7M [60], RefSpatial [65], PRISM [19]) and subsample at four target scales (80k, 400k, 800k, and 2M) for supervised fine-tuning (see Appendix B.2 and B.3 for details).

Table 1: Distribution of consistent, counter, and ambiguous examples. Existing spatial benchmarks are skewed toward consistent examples, mirroring the natural statistics of perspective projection in real-world images. 

<table><tr><td>Type</td><td>EmbSpatial-Bench</td><td>CV-Bench-3D</td><td>Definition</td></tr><tr><td>Consistent</td><td>976 (80.9%)</td><td>363 (60.5%)</td><td>GT aligns with heuristic</td></tr><tr><td>Counter</td><td>129 (10.7%)</td><td>65 (10.8%)</td><td>GT contradicts heuristic</td></tr><tr><td>Ambiguous</td><td>101 (8.4%)</td><td>172 (28.7%)</td><td> $\Delta y < 5\%$  of image height</td></tr></table>

# 3.3 Evidence from Existing Benchmarks

We first examine whether vertical-distance entanglement is observable on established spatial reasoning benchmarks that use real-world images: EmbSpatial-Bench [20] and the 3D-spatial split of CV-Bench [55].

Data distribution is skewed toward consistent examples. We classify all depth-related questions in both benchmarks into consistent, counter, and ambiguous categories following the criteria defined in Section 3.1. As shown in Table 1, consistent examples account for 80.9% of EmbSpatial-Bench and 60.5% of CV-Bench-3D, while counter examples constitute only about 10% in each. This heavy skew reflects the natural statistics of real-world photographs: in most everyday scenes, farther objects do appear higher in the image.

Models systematically fail on counter examples. We evaluate a range of VLMs spanning different architectures and scales on the two benchmarks, reporting accuracy separately for consistent and counter subsets (Table 2). Across all models and all training scales, accuracy on consistent examples significantly exceeds that on counter examples. For instance, Qwen2.5-VL fine-tuned on 2M samples achieves 60.9% on the consistent split of EmbSpatial-Bench but only 24% on counter examples, yielding a 36.9 percentage-point gap. This pattern holds regardless of model family (Molmo, NVILA, or Qwen2.5-VL), model size, or the amount of spatial fine-tuning data, suggesting that vertical-distance entanglement is a widespread phenomenon rather than an artifact of any single architecture, training recipe, or data scale.

# 4 Behavioral Analysis with a Synthetic Dataset

The accuracy gap in Section 3 indicates that VLMs systematically fail on counter examples in real-world datasets. However, real photographs conflate multiple depth cues (e.g., vertical position, apparent size, and occlusion), making it difficult to isolate the contribution of any single cue. To enable controlled interventions, we introduce SpatialTunnel, a synthetic dataset that decouples an object’s vertical image-plane position from its 3D depth by design, allowing the two factors to be manipulated independently.

Table 2: Accuracy on consistent vs. counter examples across models and benchmarks. All models exhibit a substantial accuracy gap, with consistent examples outperforming counter examples. Results are reported on depth-related questions from EmbSpatial-Bench and CV-Bench-3D. Indented rows denote fine-tuned variants at the given spatial data scale. 

<table><tr><td rowspan="2">Model</td><td colspan="2">EmbSpatial-Bench</td><td colspan="2">CV-Bench-3D</td></tr><tr><td>Consistent</td><td>Counter</td><td>Consistent</td><td>Counter</td></tr><tr><td>Molmo-7B-O-0924 [16]</td><td>63.5</td><td>34.9 (-28.6)</td><td>93.1</td><td>75.4 (-17.7)</td></tr><tr><td>+ 80k</td><td>60.6</td><td>29.5 (-31.1)</td><td>80.2</td><td>56.9 (-23.3)</td></tr><tr><td>+ 400k</td><td>62.7</td><td>27.1 (-35.6)</td><td>89.5</td><td>56.9 (-32.6)</td></tr><tr><td>+ 800k</td><td>65.2</td><td>34.1 (-31.1)</td><td>88.7</td><td>70.8 (-17.9)</td></tr><tr><td>+ 2M</td><td>65.3</td><td>39.5 (-25.8)</td><td>90.6</td><td>72.3 (-18.3)</td></tr><tr><td>NVILA-Lite-2B [40]</td><td>49.0</td><td>27.1 (-21.9)</td><td>74.4</td><td>40.0 (-34.4)</td></tr><tr><td>+ 80k</td><td>57.7</td><td>15.5 (-42.2)</td><td>71.6</td><td>50.8 (-20.8)</td></tr><tr><td>+ 400k</td><td>61.1</td><td>34.1 (-27.0)</td><td>81.3</td><td>58.5 (-22.8)</td></tr><tr><td>+ 800k</td><td>63.2</td><td>38.8 (-24.4)</td><td>84.6</td><td>67.7 (-16.9)</td></tr><tr><td>+ 2M</td><td>60.7</td><td>41.1 (-19.6)</td><td>97.2</td><td>93.8 (-3.4)</td></tr><tr><td>RoboRefer-2B-SFT [65]</td><td>87.0</td><td>59.7 (-27.3)</td><td>98.9</td><td>95.4 (-3.5)</td></tr><tr><td>Qwen2.5-VL-3B-Instruct [4]</td><td>54.7</td><td>32.6 (-22.1)</td><td>75.5</td><td>55.4 (-20.1)</td></tr><tr><td>+ 80k</td><td>50.6</td><td>30.2 (-20.4)</td><td>69.7</td><td>60.0 (-9.7)</td></tr><tr><td>+ 400k</td><td>52.6</td><td>27.1 (-25.5)</td><td>65.8</td><td>58.5 (-7.3)</td></tr><tr><td>+ 800k</td><td>55.8</td><td>26.4 (-29.4)</td><td>61.2</td><td>58.5 (-2.7)</td></tr><tr><td>+ 2M</td><td>60.9</td><td>24.0 (-36.9)</td><td>62.0</td><td>53.8 (-8.2)</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct [3]</td><td>73.3</td><td>41.7 (-31.6)</td><td>98.1</td><td>90.8 (-7.3)</td></tr></table>

# 4.1 SpatialTunnel Benchmark

To evaluate spatial relations in a controlled manner, we require an environment with two key properties: (i) objects can be positioned arbitrarily, enabling queries over any spatial relation (e.g., left/right, above/below, near/far); and (ii) an object’s vertical position can be adjusted independently of its depth, allowing us to construct image groups that differ only in vertical placement while preserving depth ordering.

To satisfy these requirements, we build a tunnel-shaped synthetic scene in Blender [5] (Figure 3). Each scene consists of a single-point-perspective corridor whose walls, ceiling, and floor are symmetric about the camera’s optical axis, where objects are placed anywhere on the interior tunnel surfaces. Because objects near the top and bottom of the image can be equidistant from the camera, the common heuristic “higher in the image ⇒ farther” no longer holds. We parameterize each object by its depth z and an angular position θ on the tunnel cross-section. Holding z fixed while varying θ moves the object up/down and left/right in the image without changing its depth ordering, enabling matched counterfactual pairs that flip vertical arrangement while preserving depth.

![](images/22a061433a392aa4024809ab706d512312a49184cb36c1332ec9c105279fec88.jpg)

<details>
<summary>text_image</summary>

θ₁
θ₂
</details>

![](images/49c89f145b471d5760d9f26efb58c5636cbc46ade9df9e0f7ec1400cddce2806.jpg)

<details>
<summary>text_image</summary>

θ₁ = 0°
θ₂ = 90°
θ₁ = 90°
θ₂ = 157.5°
</details>

![](images/6e18004d5f387ae1ec32d80121c045e637847202ceae75438e6189f408f1a35b.jpg)

<details>
<summary>text_image</summary>

θ₁ = 180°
θ₂ = 45°
θ₁ = 337.5°
θ₂ = 202.5°
</details>

Fig. 3: SpatialTunnel holds the two objects at fixed depths while sweeping their angular positions around the tunnel cross-section, so that 2D image-plane layout varies independently of depth ordering.

We construct a synthetic benchmark suite, SpatialTunnel, that enables controlled spatial interventions in a single-point-perspective corridor. Specifically, we place two objects at predetermined depths and sweep each object along the tunnel cross-section, discretizing the interior into 16 angular positions (see Figure 3). This yields a $1 6 \times 1 6$ Cartesian grid over $( \theta _ { 1 } , \theta _ { 2 } )$ , enabling heatmap-style diagnostics of model behavior across configurations (see Figure 4). To increase visual diversity and improve robustness, we randomize object appearance (color, size, and shape) and scene lighting across renders. Additional synthetic variants for other spatial cues $( e . g .$ , object size) and auxiliary analyses are provided in the Appendix C.4.

# 4.2 Experimental Setup on SpatialTunnel

Given a rendered RGB image containing two objects, the model is asked a binary depth-comparison question. In our setup, an object is always placed farther from the camera than the other, and the VLM is asked to answer the questions like $^ { * } I s \ \{ o b j _ { 1 } \}$ closer to / farther from the camera than $\left\{ o b j _ { 2 } \right\} \ell ^ { , }$ Following prior work [28, 58, 62], we define a local probability by extracting the logits for Yes and No at the first generated token. We then compute the predicted probability as

$$
p = \sigma \left(\ell_ {\text { Yes }} - \ell_ {\text { No }}\right).
$$

The correctness score for a single query is defined as $v = p$ if the ground-truth answer is Yes, and $v = 1 - p$ if it is No. We report the following metrics for all VLMs described in Section 3.2. Following the definition in Section 3.1, samples are partitioned into consistent and counter subsets. We report four metrics: (1) Mean accuracy (v), the mean correctness score across all images and questions; (2) Consistent accuracy $( v _ { \mathrm { c o n s } } )$ , the mean correctness score on consistent examples; (3) Counter accuracy $( v _ { \mathrm { c t r } } )$ , the mean correctness score on counter examples; and (4) Accuracy gap $( \varDelta = v _ { \mathrm { c o n s } } - v _ { \mathrm { c t r } } )$ , the accuracy difference between the two subsets, quantifying the vertical-distance entanglement. A model with no directional bias would yield $\varDelta \approx 0$ .

![](images/bdc21e74ed9198685627a2d88e17f3ca8e2c3ab6f50725fa823a4c7964d49e60.jpg)  
Fig. 4: Mean accuracy heatmaps on SpatialTunnel for Molmo-7B. Each cell indexes a joint angular configuration $( \theta _ { 1 } , \theta _ { 2 } )$ of the two objects (red = higher accuracy; blue = lower). Gray indicates configurations outside the subset. From base → 400k → 2M training samples, accuracy on (a) perspective-consistent cells improves steadily. In contrast, (b) counter cells remain substantially harder, with the largest drop at 400k and a partial recovery at 2M.

# 4.3 Results on SpatialTunnel: Vertical-Distance Entanglement

Consistent with Section 3.3, we observe that the vertical-distance entanglement is universal. Across all base and fine-tuned models, accuracy is consistently higher on the consistent subset than on the counter subset, yielding a positive accuracy gap ∆. Table 3 and Figure 4 summarize model behavior on SpatialTunnel.

For example, base Qwen2.5-VL-3B achieves $v _ { \mathrm { c o n s } } ~ = ~ 0 . 7 7 6$ but only $v _ { \mathrm { c t r } } ~ = ~ 0 . 3 6 0$ , indicating strong reliance on the vertical-position shortcut. While base NVILA-Lite-2B produces a narrower gap, its sub-0.5 overall accuracy suggests near-random performance rather than meaningful depth understanding. Figure 4 visualizes positional bias at the cell level for Molmo-7B variants. If predictions were insensitive to 2D placement, accuracy would be approximately uniform across the grid. Instead, most models show pronounced contrast between consistent and counter regions. The results suggest that large-scale spatial training reduces this reliance. RoboRefer [65], trained on more than 20M QA pairs, achieves the smallest gap $( \varDelta = + 0 . 0 4 6 )$ among models performing above chance. Qwen3-VL-235B attains the highest mean accuracy $( v = 0 . 9 0 8 )$ with a similarly small gap $( \varDelta = + 0 . 0 6 8 )$ , indicating that very large-scale pretraining can substantially alleviate this bias even without targeted spatial fine-tuning.

Table 3: Consistent vs. Counter accuracy on SpatialTunnel. v: mean correctness score; $v _ { \mathrm { c o n s } }$ and $v _ { \mathrm { c t r } } \colon$ scores on consistent and counter subsets; $\varDelta = v _ { \mathrm { c o n s } } - v _ { \mathrm { c t r } }$ . 

<table><tr><td>Model</td><td> $\mathbf{v}$ </td><td> $\mathbf{v}_{\text{cons}}$ </td><td> $\mathbf{v}_{\text{ctr}}$ </td><td> $\Delta$ </td></tr><tr><td>Molmo-7B-O-0924</td><td>0.528</td><td>0.565</td><td>0.487</td><td>+0.078</td></tr><tr><td>+ 80k</td><td>0.496</td><td>0.507</td><td>0.486</td><td>+0.021</td></tr><tr><td>+ 400k</td><td>0.501</td><td>0.593</td><td>0.409</td><td>+0.184</td></tr><tr><td>+ 800k</td><td>0.531</td><td>0.628</td><td>0.430</td><td>+0.198</td></tr><tr><td>+ 2M</td><td>0.666</td><td>0.703</td><td>0.630</td><td>+0.073</td></tr><tr><td>NVILA-Lite-2B</td><td>0.488</td><td>0.504</td><td>0.471</td><td>+0.033</td></tr><tr><td>+ 80k</td><td>0.499</td><td>0.562</td><td>0.438</td><td>+0.124</td></tr><tr><td>+ 400k</td><td>0.669</td><td>0.804</td><td>0.538</td><td>+0.267</td></tr><tr><td>+ 800k</td><td>0.646</td><td>0.728</td><td>0.571</td><td>+0.157</td></tr><tr><td>+ 2M</td><td>0.812</td><td>0.875</td><td>0.749</td><td>+0.127</td></tr><tr><td>RoboRefer-2B-SFT</td><td>0.793</td><td>0.816</td><td>0.770</td><td>+0.046</td></tr><tr><td>Qwen2.5-VL-3B</td><td>0.570</td><td>0.776</td><td>0.360</td><td>+0.416</td></tr><tr><td>+ 80k</td><td>0.512</td><td>0.585</td><td>0.440</td><td>+0.145</td></tr><tr><td>+ 400k</td><td>0.503</td><td>0.588</td><td>0.418</td><td>+0.171</td></tr><tr><td>+ 800k</td><td>0.499</td><td>0.600</td><td>0.398</td><td>+0.202</td></tr><tr><td>+ 2M</td><td>0.500</td><td>0.648</td><td>0.353</td><td>+0.295</td></tr><tr><td>Qwen3-VL-235B</td><td>0.908</td><td>0.948</td><td>0.880</td><td>+0.068</td></tr></table>

![](images/80c51ff63d4b3eab36e606f98dbe8289194102a4ff629db92b10a5085fd649d3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Question Pair"] --> B["+ Is A to the left or right of B ?"]
    A --> C["(GT : Left)"]
    A --> D["+ Is B to the left or right of A ?"]
    A --> E["(GT : Right)"]
    F["Delta vectors"] --> G["δ_far"]
    F --> H["δ_close"]
    F --> I["..."]
    G --> J["(RoboRefer-2B-SFT)"]
    H --> J
    I --> J
    J --> K["Spatial representation"]
```
</details>

Fig. 5: Contrastive probing for representation-level spatial analysis. Given a spatial-relation VQA, we construct a minimal question pair by swapping the object order, which flips the ground-truth relation. We extract the final-token hidden state at an intermediate layer for each question and compute a delta vector as their difference, isolating the relational displacement in embedding space. Aggregated across samples, these vectors summarize the model’s internal spatial representations and enable diagnosing systematic confounds among spatial cues.

# 5 Representation Analysis via Contrastive Probing

Sections 3–4 established vertical-distance entanglement as a model-intrinsic phenomenon through behavioral evaluation. We now turn to internal representations to examine how spatial axes are encoded and what distinguishes models that exhibit robust spatial reasoning.

# 5.1 Beyond Benchmark Accuracy

Behavioral accuracy alone can be a misleading indicator of spatial understanding. Table 4 reports performance across five spatial reasoning tasks spanning different formats, dimensionalities, and difficulty levels. Beyond EmbSpatial-Bench and the 3D-spatial split of CV-Bench (CV-3D) used in Table 2, we additionally include the 2D-spatial split of CV-Bench (CV-2D) and the spatial relationship and relative depth splits of BLINK [23]. Detailed task descriptions are provided in the Appendix B.4.

Fine-tuned variants of Molmo, NVILA, and Qwen show inconsistent patterns across benchmarks. For example, NVILA (2M) achieves 93.8% on CV-3D Depth but only 62.9% on BLINK Spatial Relation, while Qwen (2M) scores 78.3% on BLINK Spatial Relation but drops to 52.2% on CV-3D Distance. No single accuracy figure reliably indicates how well these models have internalized 3D spatial concepts. In contrast, RoboRefer-SFT-2B and Qwen3-VL-235B achieve consistently high performance across all benchmarks. As we show in Section 5.4, these models also exhibit the most structured internal representations under our probing framework, including high axis coherence, well-separated PCA clusters, and low entanglement. This suggests that representation quality underlies robust spatial reasoning across benchmarks, motivating us to look beyond accuracy and analyze model-internal representations directly.

Table 4: Performance comparison across spatial understanding benchmarks. Fine-tuned variants of Molmo, NVILA, and Qwen exhibit inconsistent performance fluctuations depending on the benchmark, while RoboRefer and Qwen3-VL-235B, which show strong representation structure under our probing framework (Section 5.4), achieve consistently high performance across all evaluated tasks. Bold denotes best. BLINK confidence intervals are reported in Appendix B.5. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">EmbSpatial Overall</td><td rowspan="2">CV-2D Relation</td><td rowspan="2" colspan="2">CV-3D Depth Distance</td><td rowspan="2" colspan="2">BLINK Rel. Depth Spat. Rel.</td></tr><tr></tr><tr><td>Molmo-7B-O-0924 [16]</td><td>60.7</td><td>76.3</td><td>84.5</td><td>68.5</td><td>78.2</td><td>70.6</td></tr><tr><td>+ 80k</td><td>52.9</td><td>62.3</td><td>71.0</td><td>67.5</td><td>72.6</td><td>60.8</td></tr><tr><td>+ 400k</td><td>64.9</td><td>84.3</td><td>80.0</td><td>70.8</td><td>72.6</td><td>68.5</td></tr><tr><td>+ 800k</td><td>69.1</td><td>90.0</td><td>82.0</td><td>70.8</td><td>75.0</td><td>61.5</td></tr><tr><td>+ 2M</td><td>74.3</td><td>93.7</td><td>87.3</td><td>81.3</td><td>71.0</td><td>69.2</td></tr><tr><td>NVILA-Lite-2B [40]</td><td>54.0</td><td>58.6</td><td>69.2</td><td>52.3</td><td>64.5</td><td>67.1</td></tr><tr><td>+ 80k</td><td>65.1</td><td>78.9</td><td>66.2</td><td>60.8</td><td>53.2</td><td>74.1</td></tr><tr><td>+ 400k</td><td>62.1</td><td>83.2</td><td>74.3</td><td>67.0</td><td>71.8</td><td>63.6</td></tr><tr><td>+ 800k</td><td>69.7</td><td>85.5</td><td>78.2</td><td>71.3</td><td>57.3</td><td>65.0</td></tr><tr><td>+ 2M</td><td>69.4</td><td>91.4</td><td>93.8</td><td>87.2</td><td>70.2</td><td>62.9</td></tr><tr><td>RoboRefer-SFT-2B [65]</td><td>92.0</td><td>96.5</td><td>95.7</td><td>90.5</td><td>84.7</td><td>79.7</td></tr><tr><td>Qwen2.5-VL-3B [4]</td><td>62.3</td><td>67.4</td><td>70.3</td><td>60.2</td><td>68.6</td><td>83.9</td></tr><tr><td>+ 80k</td><td>57.3</td><td>59.7</td><td>64.7</td><td>61.5</td><td>58.1</td><td>79.7</td></tr><tr><td>+ 400k</td><td>58.6</td><td>58.2</td><td>62.0</td><td>54.5</td><td>58.9</td><td>78.3</td></tr><tr><td>+ 800k</td><td>60.9</td><td>59.4</td><td>58.7</td><td>51.2</td><td>58.1</td><td>79.0</td></tr><tr><td>+ 2M</td><td>65.7</td><td>68.8</td><td>58.5</td><td>52.2</td><td>53.2</td><td>78.3</td></tr><tr><td>Qwen3-VL-235B [3]</td><td>82.0</td><td>96.5</td><td>93.3</td><td>91.0</td><td>84.7</td><td>90.2</td></tr></table>

# 5.2 Contrastive Probing

Given an image, we construct a pair of questions that differ only in the ordering of the queried objects, such as swapping “Is A to the left or right of $B ? ^ { \dag \dag }$ with “Is B to the left or right of $A ? { } ^ { p }$ . Consequently, the ground-truth answer for the swapped query becomes the spatial inverse of the original; for instance, a left relationship is inverted to right. For probing, we extract hidden states at a fixed intermediate layer $L ^ { * }$ per model family. Let $h _ { q } \in \mathbb { R } ^ { d }$ denote the final-token hidden state at layer L∗ for question q. For a question pair $\left( q _ { 1 } , q _ { 2 } \right)$ , we define delta vector $\delta = h _ { q _ { 2 } } - h _ { q _ { 1 } }$ as the representation-space displacement induced by the swap. By repeating this across many images, we obtain a set of delta vectors per spatial category (e.g., above, below, far, close, $l e f t ,$ , right). This procedure aims to isolate the latent encoding of spatial directions by neutralizing common visual components. We define two metrics over these delta vectors:

Axis coherence. For each spatial axis (horizontal, vertical, distance), we pool the delta vectors from both opposing categories (e.g., far and close for the distance axis). To align directions, we negate the deltas from the opposing category so that all vectors point toward the canonical direction:

$$
\tilde {\delta} ^ {(i)} = \left\{ \begin{array}{l l} \delta^ {(i)} & \text { if   category   is   canonical   (e.g.,   far) } \\ - \delta^ {(i)} & \text { if   category   is   opposite   (e.g.,   close) } \end{array} \right. \tag {1}
$$

Axis coherence is the mean pairwise cosine similarity over the sign-corrected set:

$$
\mathrm{Coh} _ {\text { axis }} = \frac {2}{N (N - 1)} \sum_ {i <   j} \cos (\tilde {\delta} ^ {(i)}, \tilde {\delta} ^ {(j)}). \tag {2}
$$

High coherence indicates that the model encodes the axis as a stable, consistent direction in representation space.

VD-Entanglement Index. To quantify the degree of vertical-distance entanglement at the representation level, we compute the mean delta vector $\mu _ { c }$ for each category $c \in$ {above, below, far, close} and define the VD-Entanglement Index (VD-EI):

$$
\begin{array}{l} \mathrm{VD} - \mathrm{EI} = \frac {1}{4} \left[ \cos \left(\mu_ {\text {above}}, \mu_ {\text {far}}\right) + \cos \left(\mu_ {\text {below}}, \mu_ {\text {close}}\right) \right. \tag {3} \\ \left. - \cos (\mu_ {\mathrm{above}}, \mu_ {\mathrm{close}}) - \cos (\mu_ {\mathrm{below}}, \mu_ {\mathrm{far}}) \right]. \\ \end{array}
$$

The first two terms measure the similarity between perspective-aligned pairs (above↔far, below↔close); the last two measure perspective-opposing pairs. A positive value indicates that vertical and distance representations are directionally coupled in the manner predicted by perspective projection; zero indicates independence. We extract hidden states from EmbSpatial-Bench [20] images at a fixed intermediate layer per model family, following previous approaches [10, 25, 49] (see Appendix D.3 and D.4 for details of layer selection).

# 5.3 Distance Coherence and Counter Accuracy

Distance coherence is the weakest axis. Across all models and training scales in Table 5, CohD is the lowest among the three axes. Fine-tuning substantially increases vertical coherence (e.g., Molmo: $0 . 2 3 \  \ 0 . 5 7 ;$ ; Qwen: $0 . 2 9  0 . 5 9 )$ , but CohD grows by a comparatively smaller margin.

Distance coherence growth accompanies counter accuracy improvement. Figure 6a plots CohD against counter accuracy on EmbSpatial-Bench, the same dataset from which the coherence values are derived. At early scaling steps $( e . g . , 8 0 \mathrm { k } )$ , counter accuracy sometimes drops before CohD has meaningfully increased. However, once spatial fine-tuning data reaches sufficient scale, a consistent pattern emerges across NVILA (from 80k onward) and Molmo (from 400k onward): as $\mathrm { C o h } _ { \mathrm { D } }$ increases, counter accuracy rises in tandem, tracing an upward trajectory in Figure 6a. In contrast, Qwen’s $\mathrm { C o h _ { D } }$ remains nearly flat throughout scaling (0.043 → 0.052), and its counter accuracy declines, widening the consistent-counter gap. This suggests that as models form a more coherent distance representation through spatial data scaling, they become more robust to the vertical-position shortcut. Conversely, when distance coherence stagnates, continued scaling does not resolve the entanglement.

![](images/97846a06428331571a980ff1d4d7e5e978c81c54b2fb05aeb62d5d67109a01af.jpg)  
(a) Counter Accuracy vs. Distance Coherence

![](images/8576ffc3cd83e2cd510d5d64d1381e2a73fd48f5d5f7ab89da0375e04a314387.jpg)

<details>
<summary>scatter</summary>

| Model Group | VD-Entanglement Index (VD-EI) | Distance Coherence (Co/h) |
|---|---|---|
| NVILA-Lite-2B | 0.56 | 0.09 |
| NVILA-Lite-2B | 0.60 | 0.075 |
| NVILA-Lite-2B | 0.63 | 0.07 |
| NVILA-Lite-2B | 0.64 | 0.08 |
| RoboRefer-2B-SFT | 0.39 | 0.17 |
| RoboRefer-2B-SFT | 0.64 | 0.08 |
The data points are not explicitly labeled in the chart but correspond to the labels on the x-axis. The y-axis is labeled 'Distance Coherence (Co/h)' with a value of 0.17. There is no additional data series or legend present.
</details>

(b) Distance Coherence vs. VD-EI   
Fig. 6: Internal probing analysis of spatial representations. (a) Positive correlation between behavioral accuracy on counter examples and internal distance coherence (CohD). (b) Comparing distance coherence (CohD) against geometric entanglement (VD-EI) within the NVILA family; RoboRefer occupies a unique region of high coherence and low entanglement. Unlabeled points denote base models, and numeric labels (e.g., 80k) indicate data-mix fine-tuned variants.

Cross-domain validity of distance coherence. To examine whether CohD reflects a reusable representation rather than a benchmark-specific artifact, we measure CohD on SpatialTunnel and compare it against counter accuracy on two other benchmarks. CohD computed on SpatialTunnel correlates with counter accuracy on both EmbSpatial-Bench and CV-Bench-3D $( \rho = 0 . 7 5 9$ and 0.804, respectively; both $p < 1 0 ^ { - 3 } )$ . This cross-domain alignment supports the view that CohD captures predictive signal beyond in-domain computation artifacts (see Appendix D.5 for additional details).

# 5.4 What Characterizes Strong Spatial Representations

We compare the NVILA-Lite-2B scaling series and RoboRefer-2B-SFT [65], which share the same base architecture, to identify what representation profile accompanies robust spatial reasoning. Figure 7 visualizes the delta vectors via PCA. In the base model (i.e., NVILA-Lite-2B), distance delta vectors are collapsed near the origin without forming a distinguishable axis. Fine-tuning with 2M samples initiates directional spreading, but vertical and distance clusters remain overlapped. RoboRefer exhibits three cleanly separated clusters, each aligned with a distinct principal component. Figure 6b quantifies this contrast. The NVILA scaling trajectory yields only marginal gains in CohD while VD-EI remains high (0.54–0.64). RoboRefer occupies a distinct region: the highest CohD (0.182) and the lowest VD-EI (0.362) in the family, corresponding to 59.7% counter accuracy on EmbSpatial-Bench versus 41.1% for NVILA (2M).

![](images/cf5eb4e8722630f060d2de9080e111af5f6edca518521ce1e04e44441e3345d8.jpg)  
Fig. 7: PCA of delta vectors across models. Each point is a delta vector colored by axis (orange: horizontal, green: vertical, purple: distance), with darker/lighter shades distinguishing opposing categories within each axis (e.g., left vs. right). Molmo (2M), NVILA (2M), and Qwen (2M) show separation along the horizontal and vertical axes, but distance delta vectors remain poorly distinguished. RoboRefer and Qwen3 exhibit three clearly separated clusters, each aligned with a distinct principal component.

These results suggest that high CohD, together with low VD-EI as a complementary signal, accompanies robust spatial reasoning across benchmarks. Within the NVILA scaling series, incremental increases in fine-tuning scale yield only modest gains in CohD. In contrast, RoboRefer and Qwen3 exhibit well-separated axes and strong overall performance, showing such structure can emerge under substantially richer training regimes. RoboRefer reflects a different training regime (e.g., additional supervision and much larger training), so we treat it as an illustrative reference rather than attributing its gains to a single factor. Overall, CohD can serve as a practical diagnostic for whether training improves spatial representations.

Table 5: Axis coherence and VD-Entanglement Index. Coh measures directional consistency within axes. CohD is consistently the lowest across all models. Bold denotes best. 

<table><tr><td>Model</td><td> $\text{Coh}_\text{H}$ </td><td> $\text{Coh}_\text{V}$ </td><td> $\text{Coh}_\text{D}$ </td><td>VD-EI</td></tr><tr><td>Molmo-7B</td><td>0.143</td><td>0.228</td><td>0.075</td><td>0.279</td></tr><tr><td>+ 80k</td><td>0.122</td><td>0.332</td><td>0.072</td><td>0.388</td></tr><tr><td>+ 400k</td><td>0.236</td><td>0.597</td><td>0.096</td><td>0.459</td></tr><tr><td>+ 800k</td><td>0.247</td><td>0.559</td><td>0.107</td><td>0.514</td></tr><tr><td>+ 2M</td><td>0.239</td><td>0.574</td><td>0.112</td><td>0.474</td></tr><tr><td>NVILA-2B</td><td>0.323</td><td>0.289</td><td>0.052</td><td>0.539</td></tr><tr><td>+ 80k</td><td>0.295</td><td>0.497</td><td>0.070</td><td>0.606</td></tr><tr><td>+ 400k</td><td>0.242</td><td>0.574</td><td>0.095</td><td>0.589</td></tr><tr><td>+ 800k</td><td>0.278</td><td>0.498</td><td>0.089</td><td>0.591</td></tr><tr><td>+ 2M</td><td>0.241</td><td>0.553</td><td>0.104</td><td>0.550</td></tr><tr><td>RoboRefer-2B</td><td>0.649</td><td>0.830</td><td>0.182</td><td>0.362</td></tr><tr><td>Qwen2.5-3B</td><td>0.367</td><td>0.293</td><td>0.043</td><td>0.457</td></tr><tr><td>+ 80k</td><td>0.386</td><td>0.315</td><td>0.040</td><td>0.456</td></tr><tr><td>+ 400k</td><td>0.450</td><td>0.452</td><td>0.042</td><td>0.451</td></tr><tr><td>+ 800k</td><td>0.473</td><td>0.538</td><td>0.045</td><td>0.429</td></tr><tr><td>+ 2M</td><td>0.485</td><td>0.586</td><td>0.052</td><td>0.472</td></tr></table>

# 6 Conclusion

We introduced a representation-level diagnostic framework that reveals verticaldistance entanglement as a pervasive, model-intrinsic bias across VLM families and scales. Our analysis shows that models with more structured spatial representations – characterized by high distance coherence and low VD-Entanglement Index – not only exhibit stronger counter-heuristic robustness but also achieve higher accuracy across diverse spatial reasoning benchmarks. To isolate this bias from evaluation-set confounds, we introduced the synthetic benchmark SpatialTunnel, which removes perspective-driven correlations present in natural images. Together, these results demonstrate that representational structure, rather than benchmark accuracy alone, provides a reliable indicator of robust spatial reasoning in vision-language models.

# References

1. Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., et al.: Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691 (2022) 1   
2. Anthropic: Claude opus 4 & claude sonnet 4 system card. Tech. rep., Anthropic (May 2025), https : / / www - cdn . anthropic . com / 4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf 1   
3. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025) 2, 5, 7, 11, 22   
4. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 2, 5, 7, 11, 21   
5. Blender Online Community: Blender - a 3D modelling and rendering package. Blender Foundation (2025), https://www.blender.org/ 7   
6. Brazil, G., Kumar, A., Straub, J., Ravi, N., Johnson, J., Gkioxari, G.: Omni3d: A large benchmark and model for 3d object detection in the wild. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 13154– 13164 (2023) 24   
7. Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A., Zhang, Y.: Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158 (2017) 24   
8. Chang, A.X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., et al.: Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012 (2015) 23   
9. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14455–14465 (2024) 5   
10. Chen, H., Lin, J., Chen, X., Fan, Y., Jin, X., Su, H., Dong, J., Fu, J., Shen, X.: Rethinking visual layer selection in multimodal llms. arXiv e-prints pp. arXiv–2504 (2025) 12, 32   
11. Chen, S., Zhu, T., Zhou, R., Zhang, J., Gao, S., Niebles, J.C., Geva, M., He, J., Wu, J., Li, M.: Why is spatial reasoning hard for VLMs? an attention mechanism perspective on focus areas. In: Forty-second International Conference on Machine Learning (2025), https://openreview.net/forum?id=k7vcuqLK4X 3, 4   
12. Chen, S., Uy, M.A., Song, C.H., Ladhak, F., Murali, A., Qu, Q., Birchfield, S., Blukis, V., Tremblay, J.: SpaceTools: Tool-augmented spatial reasoning via double interactive RL. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026), https://arxiv.org/abs/2512.04069, to appear 2   
13. Cheng, A.C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., Liu, S.: Spatialrgpt: Grounded spatial reasoning in vision-language models. In: Advances in Neural Information Processing Systems (NeurIPS) (2024) 2   
14. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5828–5839 (2017) 22, 24

15. Danier, D., Aygün, M., Li, C., Bilen, H., Mac Aodha, O.: Depthcues: Evaluating monocular depth perception in large vision models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 20049–20059 (2025) 4, 5, 21   
16. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al.: Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 91–104 (2025) 2, 5, 7, 11, 21   
17. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 13142–13153 (2023) 23   
18. Deitke, M., VanderBilt, E., Herrasti, A., Weihs, L., Ehsani, K., Salvador, J., Han, W., Kolve, E., Kembhavi, A., Mottaghi, R.: Procthor: Large-scale embodied ai using procedural generation. Advances in Neural Information Processing Systems 35, 5982–5994 (2022) 22   
19. Deshpande, A., Deng, Y., Ray, A., Salvador, J., Han, W., Duan, J., Zeng, K.H., Zhu, Y., Krishna, R., Hendrix, R.: Graspmolmo: Generalizable task-oriented grasping via large-scale synthetic data generation. arXiv preprint arXiv:2505.13441 (2025) 5, 6, 23, 24   
20. Du, M., Wu, B., Li, Z., Huang, X.J., Wei, Z.: Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). pp. 346–355 (2024) 2, 3, 4, 6, 12, 24   
21. Eppner, C., Mousavian, A., Fox, D.: Acronym: A large-scale grasp dataset based on simulation. In: 2021 IEEE International Conference on Robotics and Automation (ICRA). pp. 6222–6227. IEEE (2021) 23   
22. Eppner, C., Murali, A., Garrett, C., O’Flaherty, R., Hermans, T., Yang, W., Fox, D.: scene\_synthesizer: A python library for procedural scene generation in robot manipulation. Journal of Open Source Software 10(105), 7561 (2025) 23   
23. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive. In: European Conference on Computer Vision. pp. 148–166. Springer (2024) 2, 3, 10, 25   
24. Google: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities (2025), https://arxiv.org/ abs/2507.06261 1, 27   
25. Gurnee, W., Tegmark, M.: Language models represent space and time. In: Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., Sun, Y. (eds.) International Conference on Learning Representations. vol. 2024, pp. 2483– 2503 (2024), https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 0a6059857ae5c82ea9726ee9282a7145-Paper-Conference.pdf 12, 32   
26. Hata, K., Savarese, S.: Cs231a course notes 1: Camera models. url: https://web. stanford. edu/class/cs231a/course\_ notes/01-camera-models. pdf (2015) 20   
27. Hoiem, D., Savarese, S.: Representations and techniques for 3D object recognition and scene interpretation. Springer Nature (2022) 21   
28. Hu, J., Levy, R.: Prompting is not a substitute for probability measurements in large language models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 5040–5060 (2023) 8

29. Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Galliker, M.Y., Ghosh, D., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A.Z., Shi, L.X., Smith, L., Springenberg, J.T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling, A., Wang, H., Yu, L., Zhilinsky, U.: π0.5: a vision-language-action model with openworld generalization (2025), https://arxiv.org/abs/2504.16054 1   
30. Kamath, A., Hessel, J., Chang, K.W.: What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP) (2023) 3   
31. Kang, R., Chen, H., Gkioxari, G., Perona, P.: Linear mechanisms for spatiotemporal reasoning in vision language models. arXiv preprint arXiv:2601.12626 (2026) 4   
32. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E.P., Sanketi, P.R., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., Finn, C.: Openvla: An open-source vision-language-action model. In: Agrawal, P., Kroemer, O., Burgard, W. (eds.) Proceedings of the 8th Conference on Robot Learning (CoRL). Proceedings of Machine Learning Research, vol. 270, pp. 2679–2713. PMLR (06–09 Nov 2025), https://proceedings.mlr.press/v270/kim25c.html 1   
33. Kolve, E., Mottaghi, R., Han, W., VanderBilt, E., Weihs, L., Herrasti, A., Deitke, M., Ehsani, K., Gordon, D., Zhu, Y., et al.: Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474 (2017) 24   
34. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision 123(1), 32–73 (2017) 24   
35. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision 128(7), 1956–1981 (2020) 23   
36. Lazarow, J., Griffiths, D., Kohavi, G., Crespo, F., Dehghan, A.: Cubify anything: Scaling indoor 3d object detection. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22225–22233 (2025) 23   
37. Li, F., Song, W., Zhao, H., Wang, J., Ding, P., Wang, D., Zeng, L., Li, H.: Spatial forcing: Implicit spatial representation alignment for vision-language-action model. arXiv preprint arXiv:2510.12276 (2025) 4   
38. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014) 24   
39. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023) 2   
40. Liu, Z., Zhu, L., Shi, B., Zhang, Z., Lou, Y., Yang, S., Xi, H., Cao, S., Gu, Y., Li, D., et al.: Nvila: Efficient frontier visual language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4122– 4134 (2025) 5, 7, 11, 21   
41. NVIDIA, :, Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L.J., Fang, Y., Fox, D., Hu, F., Huang, S., Jang, J., Jiang, Z., Kautz, J., Kundalia, K., Lao, L., Li, Z., Lin, Z., Lin, K., Liu, G., Llontop, E., Magne, L., Mandlekar, A.,

Narayan, A., Nasiriany, S., Reed, S., Tan, Y.L., Wang, G., Wang, Z., Wang, J., Wang, Q., Xiang, J., Xie, Y., Xu, Y., Xu, Z., Ye, S., Yu, Z., Zhang, A., Zhang, H., Zhao, Y., Zheng, R., Zhu, Y.: Gr00t n1: An open foundation model for generalist humanoid robots (2025), https://arxiv.org/abs/2503.14734 1   
42. OpenAI: Openai gpt-5 system card (2025), https://arxiv.org/abs/2601.03267 1, 27   
43. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021) 32   
44. Raistrick, A., Mei, L., Kayan, K., Yan, D., Zuo, Y., Han, B., Wen, H., Parakh, M., Alexandropoulos, S., Lipson, L., et al.: Infinigen indoors: Photorealistic indoor scenes using procedural generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21783–21794 (2024) 23   
45. Ray, A., Duan, J., II, E.L.B., Tan, R., Bashkirova, D., Hendrix, R., Ehsani, K., Kembhavi, A., Plummer, B.A., Krishna, R., Zeng, K.H., Saenko, K.: SAT: Dynamic spatial aptitude training for multimodal language models. In: Second Conference on Language Modeling (2025), https://openreview.net/forum?id=DW8U8ZWa1U 4, 5, 6, 22, 24   
46. Savva, M., Chang, A.X., Hanrahan, P.: Semantically-enriched 3d models for common-sense knowledge. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. pp. 24–31 (2015) 23   
47. Sheta, H., Huang, E.H., Wu, S., Alenabi, I., Hong, J., Lin, R., Ning, R., Wei, D., Yang, J., Zhou, J., et al.: From behavioral performance to internal competence: Interpreting vision-language models with vlm-lens. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. pp. 886–895 (2025) 4   
48. Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J., Garg, A.: Progprompt: Generating situated robot task plans using large language models. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 11523–11530 (2023). https://doi.org/10.1109/ ICRA48891.2023.10161317 1   
49. Skean, O., Arefin, M.R., Zhao, D., Patel, N.N., Naghiyev, J., LeCun, Y., Shwartz-Ziv, R.: Layer by layer: Uncovering hidden representations in language models. In: Forty-second International Conference on Machine Learning (2025), https: //openreview.net/forum?id=WGXb7UdvTX 12, 32   
50. Song, C.H., Blukis, V., Tremblay, J., Tyree, S., Su, Y., Birchfield, S.: Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15768–15780 (2025) 2, 4, 5, 6, 22, 24   
51. Song, C.H., Wu, J., Washington, C., Sadler, B.M., Chao, W.L., Su, Y.: Llmplanner: Few-shot grounded planning for embodied agents with large language models. In: ICCV (2023) 1   
52. Szeliski, R.: Computer vision: algorithms and applications. Springer Nature (2022) 20   
53. Tan, H., Zhou, E., Li, Z., Xu, Y., Ji, Y., Chen, X., Chi, C., Wang, P., Jia, H., Ao, Y., Cao, M., Chen, S., Li, Z., Liu, M., Wang, Z., Rong, S., Lyu, Y., Zhao, Z., Co, P., Li, Y., Han, Y., Xie, S., Yao, G., Wang, S., Zhang, L., Yang, X., Jiao, Y., Shi, D., Xie, K., Nie, S., Men, C., Lin, Y., Wang, Z., Huang, T., Zhang, S.: Robobrain 2.5: Depth in sight, time in mind. arXiv preprint arXiv:2601.14352 (2026) 2

54. Team, G.R., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Armstrong, T., Balakrishna, A., Baruch, R., Bauza, M., Blokzijl, M., et al.: Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020 (2025) 1, 4   
55. Tong, S., II, E.L.B., Wu, P., Woo, S., IYER, A.J., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., Pan, X., Fergus, R., LeCun, Y., Xie, S.: Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024), https:// openreview.net/forum?id=Vi8AepAXGy 2, 3, 4, 6, 24, 32   
56. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023) 32   
57. Wang, X., Ma, W., Zhang, T., de Melo, C.M., Chen, J., Yuille, A.: Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24669– 24679 (2025) 4   
58. Wang, Y., Shi, F.: Logical forms complement probability in understanding language model (and human) performance. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 16862–16877 (2025) 8   
59. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12–22 (2023) 22   
60. Zhang, J., Chen, Y., Zhou, Y., Xu, Y., Huang, Z., Mei, J., Chen, J., Yuan, Y.J., Cai, X., Huang, G., et al.: From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976 (2025) 5, 6, 22, 24   
61. Zhang, W., Huang, Y., Xu, Y., Huang, J., Zhi, H., Ren, S., Xu, W., Zhang, J.: Why do mllms struggle with spatial understanding? a systematic analysis from data to architecture (2025), https://arxiv.org/abs/2509.02359 3, 4   
62. Zhang, Z., Hu, F., Lee, J., Shi, F., Kordjamshidi, P., Chai, J., Ma, Z.: Do visionlanguage models represent space and how? evaluating spatial frame of reference under ambiguities. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=84pDoCD4lH 3, 8   
63. Zheng, J., Zhang, J., Li, J., Tang, R., Gao, S., Zhou, Z.: Structured3d: A large photo-realistic dataset for structured 3d modeling. In: European Conference on Computer Vision. pp. 519–535. Springer (2020) 22   
64. Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ade20k dataset. International journal of computer vision 127(3), 302–321 (2019) 24   
65. Zhou, E., An, J., Chi, C., Han, Y., Rong, S., Zhang, C., Wang, P., Wang, Z., Huang, T., Sheng, L., Zhang, S.: Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https://openreview.net/forum? id=OGxalNUHbJ 2, 4, 5, 6, 7, 9, 11, 13, 22, 23, 24

# Appendices

The following appendices provide supplementary material that complements the main text.

– Appendix A Ground-Plane Geometry and Vertical Image Position.   
– Appendix B Additional Details on Experiment Setup.   
– Appendix C Additional Details on SpatialTunnel.   
– Appendix D Additional Details on Contrastive Probing.

# A Ground-Plane Geometry and Vertical Image Position

We derive how perspective projection on a flat ground plane induces a monotonic relationship between an object’s depth and its vertical image coordinate under a standard pinhole camera model [26, 52].

Setup. Consider a pinhole camera with focal length $f ,$ whose optical axis is aligned with the world Z-axis, and whose center is at height $H _ { c } > 0$ above a horizontal ground plane [26].

World coordinates are $( X , Y , Z )$ , with the ground plane defined as $Y = 0$ , and the camera is located at $( 0 , H _ { c } , 0 )$ , looking along the positive Z-axis. In the camera coordinate system, a point on the ground plane has coordinates

$$
(X _ {c}, Y _ {c}, Z _ {c}) = (X, - H _ {c}, Z),
$$

where $Z > 0$ denotes the depth of the point.

Perspective projection. Under the pinhole camera model, the projection of $\left( X _ { c } , Y _ { c } , Z _ { c } \right)$ onto the image plane at distance f along the $Z _ { c } .$ -axis is given by [26]

$$
u = f \frac {X _ {c}}{Z _ {c}}, v = f \frac {Y _ {c}}{Z _ {c}}.
$$

Substituting the ground-plane coordinates $( X _ { c } , Y _ { c } , Z _ { c } ) = ( X , - H _ { c } , Z )$ yields

$$
u = f \frac {X}{Z}, v = - f \frac {H _ {c}}{Z}.
$$

Thus, for points on the ground plane with fixed camera height $H _ { c } ,$ , the vertical image coordinate satisfies

$$
v (Z) = - f \frac {H _ {c}}{Z}.
$$

Depth–height relationship. From the expression above, the magnitude of the vertical coordinate obeys

$$
| v (Z) | \propto \frac {1}{Z},
$$

so that increasing depth Z decreases the magnitude |v|. Adopting the standard image convention that the v-axis increases downward, we define the image-frame coordinate as

$$
v _ {\mathrm{img}} (Z) = - v (Z) = \frac {f H _ {c}}{Z} > 0,
$$

which confirms that ground-plane points appear below the principal point. Under the zero-tilt assumption, the horizon coincides with the principal point at $v _ { \mathrm { i m g } } =$ 0 [27], and as $Z \to \infty .$ , we have $v _ { \mathrm { i m g } } ( Z ) \to 0 ^ { + }$ , meaning that points farther along the ground plane project closer to the horizon line and therefore appear higher in the image. Consequently, for objects resting on a common ground plane, greater depth corresponds to a higher vertical position in the image, which is precisely the classical elevation monocular depth cue exploited in both human perception and recent depth-cue benchmarks [15].

# B Additional Details on Experiment Setup

In this section, we detail the models, training data sources, data mix composition, and benchmarks used in the experiments described in Section 3.

# B.1 Models

We conduct experiments on the following vision-language models, each capable of spatial reasoning.

– Molmo-7B-O-0924 [16]: Molmo-7B-O-0924 is a 7B-parameter open visionlanguage model from the Molmo family, trained on the PixMo dataset of around one million carefully curated image–text pairs, using an OLMo-7B backbone with OpenAI CLIP as the vision encoder.   
– NVILA-Lite-2B [40]: NVILA-Lite-2B is a compact 2B-parameter visual language model built on the NVILA architecture, using a scale-then-compress design that processes high-resolution images and long videos efficiently by compressing visual tokens for faster inference and lower compute cost while maintaining competitive accuracy on standard benchmarks.   
– Qwen2.5-VL-3B-Instruct [4]: Qwen2.5-VL-3B-Instruct is a 3B-parameter multi-modal vision-language model from the Qwen2.5-VL family, designed to process images, documents, and videos together with text. It combines a Vision Transformer encoder with a Qwen2.5-series language decoder to support instruction-following tasks such as OCR, document understanding, and general visual reasoning.

– RoboRefer-2B-SFT [65]: RoboRefer-2B-SFT is a 2B-parameter visionlanguage model for robotics that is supervised-finetuned on RefSpatial and related instruction-following and referring datasets to handle spatial referring instructions in complex 3D scenes. In the second SFT step, RefSpatial is reused with both RGB and RGB-D inputs so that the image encoder learns robust spatial understanding from RGB alone while using depth as an auxiliary training signal, enabling both RGB-only and RGB-D inference at test time.   
– Qwen3-VL-235B-A22B-Instruct [3]: Qwen3-VL-235B-A22B-Instruct is a large open-weight Mixture-of-Experts vision–language model (235B parameters, 22B active) that combines text generation with visual understanding over images and video. It is an instruction-tuned Qwen3-VL variant designed for general-purpose multimodal tasks such as visual question answering, document parsing, and multilingual OCR in chat-style interactions.

# B.2 Training Data Sources

A number of benchmarks and datasets for spatial understanding in VLMs have been proposed by the community. Rather than generating training data from scratch, we leverage existing datasets and compose data mixes at varying scales to train the models. Below we describe each dataset used in our experiments.

– SAT [45]: SAT is a synthetic spatial reasoning dataset built in the ProcTHOR-10K [18] interactive 3D indoor simulation environment, using about 22K procedurally generated apartment scenes composed of 1K object assets and rendered into 2D views. It contains 175K automatically generated question–answer pairs over 20K scenes, constructed from perfect 3D geometry and simulator metadata without human annotation, and is split into 127K static spatial QAs (relative position, depth, counting) and 48K dynamic spatial QAs grouped into Egocentric Movement, Object Movement, Allocentric Perspective, Goal Aiming, and Action Consequence, where actions in the simulator change spatial relationships across frames.   
– RoboSpatial [50]: RoboSpatial is a large-scale 2D/3D spatial reasoning dataset built from real indoor and tabletop environments, where egocentric RGB images are paired with 3D scans instead of a synthetic simulator. The data are collected as 3D scene scans and corresponding first-person images, and then automatically annotated with around 3M spatial relations over 1M images and 5K scans, capturing rich object–object and object–space relationships relevant for robotics. The benchmark defines tasks such as spatial affordance prediction (where an object can be placed or an action can be executed), spatial relationship prediction (e.g., left/right, in front/behind, on/under), and robot manipulation tasks that test whether models can use these spatial cues to guide real-world actions.   
– SPAR-7M [60]: SPAR-7M is a large-scale spatial reasoning dataset built from indoor 3D scenes (e.g., ScanNet [14], ScanNet++ [59], Structured3D [63])

with around 7M QA pairs and 33 tasks covering a wide range of spatial perception and reasoning skills. For our experiments, we sample from the following tasks: (1) Multi-view object spatial relation, which requires describing object–camera spatial relations in a multi-view setting (2) Single-view object spatial relation, a single-view multiple-choice task that asks models to select the correct relative position between two objects (3) Single-view spatial imagination, which evaluates single-view spatial imagination by asking models to verbally infer observer-centric relations beyond the directly visible configuration (4) Object count, which focuses on numerical reasoning by predicting object counts in the scene and (5) Multi-view spatial imagination, a multi-view spatial imagination task that requires describing how object–object relations change as the camera moves.

– RefSpatial [65]: RefSpatial is a large-scale spatial referring dataset built from 2D web images (OpenImages [35]) and 3D embodied videos (CA-1M [36]), plus simulated scenes from Infinigen [44] with Objaverse [17] assets. It contains 2.5M RGB-D samples and 20M QA pairs over 31 spatial relations and up to 5 reasoning steps, covering qualitative/quantitative spatial QA and point-based location/placement supervision. The tasks include object location (pointing to a described object), free-space placement (pointing to a feasible placement location), and multi-step spatial reasoning with explicit intermediate steps on simulated scenes. In our work, we sample from all Ref-Spatial sources but use only the RGB images and associated annotations, discarding depth maps.

PRISM [19]: PRISM is a large-scale synthetic task-oriented grasping dataset built in a procedurally generated tabletop simulation using ShapeNet-Sem [8, 46] objects, ACRONYM [21] grasp annotations, and SceneSynthesizer-based scene composition [22], where 2,300+ object instances are rendered in heavily randomized scenes with calibrated RGB-D views, natural language task instructions, and associated 6-DoF grasp poses. GPT-based pipelines generate and match grasp-centric descriptions and manipulation tasks to appropriate grasps, yielding hundreds of thousands of samples for training and evaluating language-conditioned grasp prediction on both seen and novel objects.

# B.3 Data Mix Composition

We construct four training data mixes of increasing scale using the five spatial datasets described in Section B.2. For the 80k through 800k mixes, each dataset contributes an equal number of samples. For the 2M mix, the per-dataset allocation is adjusted to accommodate differences in total dataset size, with smaller datasets (e.g., SAT) included in full while larger ones (e.g., RefSpatial) are subsampled. Table 6 summarizes the per-dataset sample counts at each scale.

Within each dataset, samples are drawn proportionally across all constituent sub-files (e.g., the six task categories of SAT, the seven QA types of RefSpatial). Detailed per-file sampling ratios and the ready-to-use data mix configurations will be publicly released.

Table 6: Per-dataset sample counts for each data mix scale. The 80k–800k mixes use uniform allocation across datasets. The 2M mix uses all available samples from smaller datasets and subsamples larger ones (RefSpatial at ∼3.3%, SAT and PRISM in full). 

<table><tr><td>Dataset</td><td>Available</td><td>80k</td><td>400k</td><td>800k</td><td>2M</td></tr><tr><td>SAT [45]</td><td>172,384</td><td>15,997</td><td>79,997</td><td>159,998</td><td>172,384</td></tr><tr><td>RoboSpatial [50]</td><td>300,000</td><td>15,999</td><td>79,998</td><td>159,999</td><td>300,000</td></tr><tr><td>SPAR-7M [60]</td><td>608,538</td><td>15,997</td><td>79,997</td><td>159,996</td><td>608,538</td></tr><tr><td>RefSpatial [65]</td><td>16,363,656</td><td>15,994</td><td>79,996</td><td>159,997</td><td>540,397</td></tr><tr><td>PRISM [19]</td><td>378,844</td><td>16,000</td><td>80,000</td><td>160,000</td><td>378,844</td></tr><tr><td>Total</td><td></td><td>79,987</td><td>399,988</td><td>799,990</td><td>2,000,163</td></tr></table>

# B.4 Benchmarks

We evaluate on the following benchmarks which are designed to test VLM’s spatial understanding ability.

– EmbSpatial-Bench [20]: EmbSpatial-Bench is introduced to systematically evaluate and improve large vision-language models’ spatial understanding for embodied tasks, addressing the gap that most existing Visual Spatial Reasoning benchmarks are 2D, dataset-centric (e.g., COCO [38]/VG [34]), and object-centric rather than agent-centric, thus misaligned with real navigation and manipulation settings. To better reflect embodied scenarios, the authors construct EmbSpatial-Bench from 3D indoor environments in MP3D [7], AI2-THOR [33], and ScanNet [14], rendering egocentric RGB-D views and using camera parameters and 3D coordinates to automatically derive 2D bounding boxes and spatial relation triplets between objects. They define six fundamental relations (above, below, left, right, close, far) in the agent’s egocentric coordinate system to cover all three axes, convert these relations into multiple-choice QA pairs, and apply automatic filtering based on bounding box statistics followed by human verification to ensure object recognizability, relation correctness, and plausibility of distractor options.

CV-Bench [55]: CV-Bench (CV-Bench-2D, CV-Bench-3D) is a vision-centric multiple-choice benchmark built by repurposing standard vision datasets ADE20K [64], COCO [38], and Omni3D [6] into VQA-style examples that probe fundamental 2D and 3D understanding. All images come from these real-world datasets rather than a synthetic simulator, and each instance is manually inspected, resulting in 2,638 high-quality examples with naturallanguage questions and four-way answer choices. The 2D split focuses on classic perception skills such as spatial relationship reasoning and object counting, while the 3D split targets depth ordering and relative distance understanding derived from the rich 3D annotations of Omni3D.

Table 7: BLINK performance with Wilson 95% confidence intervals. Point estimates (accuracy, %) are shown with Wilson 95% CIs for BLINK Rel. Depth (n=124) and Spat. Rel. (n=143). Bold marks the best point estimate within each column. 

<table><tr><td>Model</td><td>Rel. Depth (n=124)</td><td>Spat. Rel. (n=143)</td></tr><tr><td>Molmo-7B-O-0924</td><td>78.2 [70.2, 84.6]</td><td>70.6 [62.7, 77.5]</td></tr><tr><td>+ 80k</td><td>72.6 [64.1, 79.7]</td><td>60.8 [52.7, 68.5]</td></tr><tr><td>+ 400k</td><td>72.6 [64.1, 79.7]</td><td>68.5 [60.5, 75.6]</td></tr><tr><td>+ 800k</td><td>75.0 [66.7, 81.8]</td><td>61.5 [53.4, 69.1]</td></tr><tr><td>+ 2M</td><td>71.0 [62.4, 78.2]</td><td>69.2 [61.2, 76.2]</td></tr><tr><td>NVILA-Lite-2B</td><td>64.5 [55.8, 72.4]</td><td>67.1 [59.1, 74.3]</td></tr><tr><td>+ 80k</td><td>53.2 [44.5, 61.8]</td><td>74.1 [66.4, 80.6]</td></tr><tr><td>+ 400k</td><td>71.8 [63.3, 78.9]</td><td>63.6 [55.5, 71.1]</td></tr><tr><td>+ 800k</td><td>57.3 [48.5, 65.6]</td><td>65.0 [56.9, 72.4]</td></tr><tr><td>+ 2M</td><td>70.2 [61.6, 77.5]</td><td>62.9 [54.8, 70.4]</td></tr><tr><td>RoboRefer-SFT-2B</td><td>84.7 [77.3, 90.0]</td><td>79.7 [72.4, 85.5]</td></tr><tr><td>Qwen2.5-VL-3B</td><td>68.6 [59.9, 76.1]</td><td>83.9 [77.0, 89.0]</td></tr><tr><td>+ 80k</td><td>58.1 [49.3, 66.4]</td><td>79.7 [72.4, 85.5]</td></tr><tr><td>+ 400k</td><td>58.9 [50.1, 67.1]</td><td>78.3 [70.9, 84.3]</td></tr><tr><td>+ 800k</td><td>58.1 [49.3, 66.4]</td><td>79.0 [71.6, 84.9]</td></tr><tr><td>+ 2M</td><td>53.2 [44.5, 61.8]</td><td>78.3 [70.9, 84.3]</td></tr><tr><td>Qwen3-VL-235B</td><td>84.7 [77.3, 90.0]</td><td>90.2 [84.2, 94.1]</td></tr></table>

BLINK [23]: BLINK is a benchmark that repurposes 14 classic computer vision datasets into 3,807 visually prompted multiple-choice questions to assess fine-grained visual perception abilities such as relative depth estimation, spatial reasoning, visual correspondence, forensics detection, and multi-view understanding in multimodal large language models. The images span abstract diagrams, synthetic scenes, and real-world photographs, covering diverse settings from object-centric views to outdoor landscapes without relying on a single simulation engine, and each task is constructed by overlaying simple visual markers and natural-language questions on existing annotated datasets. Among its tasks, the relative depth (Rel. Depth) and spatial relation (Spat. Rel.) settings require models to compare which of two or more marked points is closer or to reason about geometric and positional relations between regions in the image, providing a focused probe of low-level 3D and spatial understanding beyond object recognition.

# B.5 BLINK Confidence Intervals

To contextualize performance differences on small BLINK subsets, we report Wilson 95% confidence intervals for the two BLINK splits used in Table 4: Rel. Depth (n=124) and Spat. Rel. (n=143).

# C Additional Details on SpatialTunnel

This section presents additional details for SpatialTunnel, including scene setup, the VQA protocol, proprietary-model results, and the object-size variant.

# C.1 Scene Generation Details

All scenes in SpatialTunnel are rendered using Blender.

Object placement. The tunnel has a square cross-section of $2 \mathrm { m } \times 2 \mathrm { m }$ . Each scene contains two objects placed at different depths, with obj1 always farther from the camera than obj2. To vary image-plane layout while preserving groundtruth depth, each object is swept independently over 16 discrete angular positions on the tunnel cross-section. Holding depth fixed while varying θ changes the object’s image-plane position without altering its distance from the camera.

This construction yields matched image pairs that differ only in 2D layout while preserving the underlying depth ordering.

Randomization. For each scene, we independently randomize the following factors:

– Shape. Each object is instantiated as either a sphere or a cube.   
Color. Each object is assigned one of seven colors: red, green, blue, yellow, cyan, magenta, or black. Materials are implemented with a Principled BSDF shader. Surface roughness is sampled uniformly from [0.05, 1.0]. Objects are constrained to have distinct (color, shape) combinations.   
– Size. In the phase-variation setting, the base sizes are $s _ { 1 } = 0 . 2$ for $\mathrm { \ o b j _ { 1 } }$ and $s _ { 2 } = 0 . 1$ 1 for $\mathrm { o b j } _ { 2 } ,$ , each multiplied by an independent random scale factor in [1.0, 1.5]. In the size-variation setting, sizes are controlled systematically as described in Section C.4.   
– Lighting. We use a Nishita sky texture in Blender. The sun rotation is sampled uniformly from [1.25π, 1.75π] radians, and the background intensity is fixed to 0.15.

# C.2 VQA Protocol

Given a rendered image containing two objects $( \mathrm { o b j } _ { 1 }$ and $\operatorname { o b j } _ { 2 } )$ , the model is asked a binary depth-comparison question. To control for wording effects and answer-polarity bias, we instantiate four question templates per image, varying both the queried object order and the direction of comparison:

1. ${ } ^ { 4 6 } T s$ the $\{ o b j _ { 1 } \}$ closer to the camera than the $\left\{ o b j _ { 2 } \right\} \ell ^ { , * }$ GT: No   
2. $^ { a } T s \ t h e \ \{ o b j _ { 2 } \}$ closer to the camera than the $\left\{ o b j _ { 1 } \right\} \ell ^ { \prime \prime }$ GT: Yes   
3. $^ { a } T s \ t h e \ \{ o b j _ { 2 } \}$ farther from the camera than the $\left\{ o b j _ { 1 } \right\} \ell ^ { \prime \prime }$ GT: No   
4. “Is the $\{ o b j _ { 1 } \}$ farther from the camera than the $\left\{ o b j _ { 2 } \right\} \ell ^ { , }$ GT: Yes

For each joint angular configuration $( \theta _ { 1 } , \theta _ { 2 } )$ , we render 12 scene instances with independently randomized shape, color, size, and lighting, for a total of $1 6 \times 1 6 \times 1 2 = 3 , 0 7 2$ images. With four question templates per image, this yields 12,288 question-image pairs. Unless otherwise noted, responses are evaluated using the probability-based protocol described in Section 4.2. We then average the four template-level correctness scores to obtain a single score for each configuration cell.

# C.3 Proprietary Model Results

We additionally evaluate three proprietary configurations on SpatialTunnel: GPT-5.2 [42] in its default configuration, GPT-5.2 with reasoning enabled, and Gemini-2.5-Pro [24]. Because token-level logits are not exposed by the Azure API endpoints we tested, we evaluate models using final Yes/No outputs and report exact-match accuracy. We instantiate four question templates per image and average the resulting accuracies. Table 8 summarizes these proprietary results alongside the open-source baselines that are recomputed with exact-match accuracy.

Under the default setting, GPT-5.2 attains a mean exact-match accuracy of 0.613 , with $A c c _ { \mathrm { c o n } } = 0 . 6 7 3$ and $A c c _ { \mathrm { c t r } } = 0 . 5 5 2$ , yielding a gap of $\varDelta = 0 . 1 2 0$ . The positive gap indicates better performance on perspective-consistent cells than on counter cells, matching the directional bias observed in the open-source models.

Enabling reasoning improves GPT-5.2 from 0.613 to 0.953 mean accuracy and reduces the gap from $\varDelta = 0 . 1 2 0$ to $\varDelta = 0 . 0 5 8$ . Gemini-2.5-Pro also performs strongly, achieving 0.919 mean accuracy with a slightly negative gap, $\varDelta = - 0 . 0 2 8$ . Taken together, the proprietary-model results show that enabling reasoning in GPT-5.2 both improves exact-match accuracy and reduces the consistent-counter gap on SpatialTunnel, while Gemini-2.5-Pro likewise exhibits a near-zero gap.

# C.4 Extending the Analysis to Object Size

Section 4 showed that many VLMs rely on vertical image position as a shortcut for depth. We next examine whether the same failure mode extends to another cue, object size. In natural images, larger objects often appear closer to the camera. If a model relies on this cue, its depth judgments should degrade when relative size conflicts with the true depth ordering.

To test this, we construct a size-controlled variant of SpatialTunnel in which the two object sizes, denoted by $s _ { 1 }$ for $\mathrm { \ o b j _ { 1 } }$ and $s _ { 2 }$ for $\mathrm { \ o b j _ { 2 } }$ , are anti-correlated under the constraint $s _ { 1 } + s _ { 2 } = 0 . 4$ . Specifically, we sweep $s _ { 1 }$ over 10 equal intervals, yielding 11 values in total, and set $s _ { 2 } = 0 . 4 - s _ { 1 }$ . The object depths are held fixed throughout, with $\mathrm { \ o b j _ { 1 } }$ always farther from the camera than $\mathrm { \ o b j _ { 2 } }$ . As $s _ { 1 }$ increases, the scene moves from a size-consistent regime, where the farther object is smaller, to a size-conflicting regime, where the farther object is larger than the nearer one.

Table 8: Exact-match response accuracy on SpatialTunnel under final-output evaluation. All models in this table are scored using discrete Yes/No outputs and averaged over the four question templates. These values are therefore not directly comparable to the logit-based correctness scores v reported in Section 4. 

<table><tr><td>Model</td><td> $Acc_{all}$ </td><td> $Acc_{con}$ </td><td> $Acc_{ctr}$ </td><td>Δ</td></tr><tr><td colspan="5">Molmo-7B</td></tr><tr><td>base</td><td>0.581</td><td>0.718</td><td>0.434</td><td>+0.284</td></tr><tr><td>+ 80k</td><td>0.500</td><td>0.500</td><td>0.500</td><td>+0.000</td></tr><tr><td>+ 400k</td><td>0.502</td><td>0.693</td><td>0.308</td><td>+0.384</td></tr><tr><td>+ 800k</td><td>0.551</td><td>0.768</td><td>0.316</td><td>+0.452</td></tr><tr><td>+ 2M</td><td>0.731</td><td>0.795</td><td>0.673</td><td>+0.121</td></tr><tr><td colspan="5">NVILA-Lite-2B</td></tr><tr><td>base</td><td>0.468</td><td>0.510</td><td>0.419</td><td>+0.091</td></tr><tr><td>+ 80k</td><td>0.454</td><td>0.545</td><td>0.364</td><td>+0.181</td></tr><tr><td>+ 400k</td><td>0.794</td><td>0.971</td><td>0.607</td><td>+0.364</td></tr><tr><td>+ 800k</td><td>0.851</td><td>0.979</td><td>0.727</td><td>+0.252</td></tr><tr><td>+ 2M</td><td>0.977</td><td>0.998</td><td>0.955</td><td>+0.043</td></tr><tr><td>RoboRefer-2B-SFT</td><td>0.919</td><td>0.956</td><td>0.879</td><td>+0.076</td></tr><tr><td colspan="5">Qwen2.5-VL-3B</td></tr><tr><td>base</td><td>0.493</td><td>0.653</td><td>0.342</td><td>+0.310</td></tr><tr><td>+ 80k</td><td>0.495</td><td>0.610</td><td>0.388</td><td>+0.222</td></tr><tr><td>+ 400k</td><td>0.498</td><td>0.576</td><td>0.423</td><td>+0.153</td></tr><tr><td>+ 800k</td><td>0.504</td><td>0.606</td><td>0.406</td><td>+0.200</td></tr><tr><td>+ 2M</td><td>0.506</td><td>0.681</td><td>0.335</td><td>+0.346</td></tr><tr><td>Qwen3-VL-235B</td><td>0.929</td><td>0.958</td><td>0.909</td><td>+0.050</td></tr><tr><td>GPT-5.2 (default)</td><td>0.613</td><td>0.673</td><td>0.552</td><td>+0.120</td></tr><tr><td>GPT-5.2 (reasoning)</td><td>0.953</td><td>0.980</td><td>0.922</td><td>+0.058</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.919</td><td>0.905</td><td>0.933</td><td>-0.028</td></tr></table>

Figure 8 shows a representative scene rendered under six $( s _ { 1 } , s _ { 2 } )$ configurations. As $s _ { 1 }$ increases from left to right, obj1 grows while $\mathrm { \ o b j _ { 2 } }$ shrinks correspondingly. At the left endpoint $( s _ { 1 } { = } 0 . 1 0 , s _ { 2 } { = } 0 . 3 0 )$ , apparent size agrees with the true depth ordering. At the right endpoint $( s _ { 1 } { = } 0 . 3 0 , s _ { 2 } { = } 0 . 1 0 )$ , the farther object appears substantially larger than the nearer one, which creates a strong cue that contradicts ground-truth depth.

We evaluate the same open-source VLMs as in Section 3 and report mean logit-based accuracy v over all configurations. Table 9 also reports accuracy at the two endpoints of the size sweep, $v _ { s _ { 1 } = 0 . 1 }$ and $v _ { s _ { 1 } = 0 . 3 }$ . We define the size-bias gap as

$$
\varDelta_ {s} = v _ {s _ {1} = 0. 1} - v _ {s _ {1} = 0. 3},
$$

which quantifies the size-distance entanglement. Larger positive values of $\varDelta _ { s }$ indicate stronger reliance on apparent size as a proxy for depth.

![](images/f4d733b23029ed4e2c49e9da0528c3e087db27e95dccf7e7a452e7a33aac8345.jpg)

![](images/0170af7aa2abb065fab2bef1e405f71179f084494fa246fa1d7aca19455ee993.jpg)

![](images/77403394c143e881d21751e1b4860fbba252782c912c1e688e5183251ab0327e.jpg)

![](images/a0550078f5d574eb8913b75b2fb14c261a63c10dad96126e4de9c9386dd8840f.jpg)

![](images/54dad9b340beb3a6de588a835598dfaafa6358f96f7c7078cbaea2543dd3fdab.jpg)

![](images/df6a30fb5344c77d0467d63613f1956d803f98250ba32776391f1fd0bd5844aa.jpg)

Fig. 8: Object-size variation in SpatialTunnel. A representative scene rendered under six $( s _ { 1 } , s _ { 2 } )$ configurations with $s _ { 1 } + s _ { 2 } = 0 . 4$ , where $s _ { 1 }$ and $s _ { 2 }$ denote the sizes of $\mathrm { \ o b j _ { 1 } }$ and $\mathrm { o b j } _ { 2 } ,$ respectively. $\mathrm { \ o b j _ { 1 } }$ is always farther from the camera than $\mathrm { \ o b j _ { 2 } . }$ As $s _ { 1 }$ increases from left to right, the farther object grows while the nearer object shrinks, moving from a size-consistent to a size-conflicting configuration.   
![](images/b1dd1c94fb66f570598cb27847883946b81d8b409540eea59b6c3839994f7fc6.jpg)  
Fig. 9: Correctness as a function of object size. Mean logit-based correctness, averaged over all question templates, as a function of obj1 size (bottom axis) and $\mathrm { \ o b j _ { 2 } }$ size (top axis), with $s _ { 1 } + s _ { 2 } = 0 . 4$ . Each curve corresponds to one training-data variant. Molmo and NVILA show clear degradation as the farther object becomes larger than the nearer one, whereas Qwen remains near chance throughout, indicating weak depth reasoning in this setting.

Figure 9 plots mean correctness as a function of $s _ { 1 }$ (bottom axis) and $s _ { 2 } =$ $0 . 4 - s _ { 1 }$ (top axis) for each model family. Models that rely on the size cue show clear performance degradation as $s _ { 1 }$ increases, that is, as the farther object becomes larger than the nearer one. This confirms that apparent size acts as a confounding cue for depth, just as vertical position does in the main text.

Analysis. The object-size results closely mirror the vertical-distance entanglement results in the main text. In both settings, performance is higher when a simple cue agrees with the true depth ordering, and lower when that cue is put in conflict with depth.

1. Qwen is largely insensitive to size, but remains near chance. All Qwen variants cluster around chance performance $( v \approx 0 . 5 0 )$ and exhibit negligible gaps $( | \varDelta _ { s } | < 0 . 0 2 )$ ). This weak sensitivity should not be interpreted as robustness. Rather, it reflects limited depth discrimination in this setting.   
2. Fine-tuning improves mean accuracy but often amplifies size-based shortcut reliance in Molmo and NVILA. For Molmo-7B (2M), mean accuracy rises to $v = 0 . 8 0 1$ , but the size-bias gap also grows to $\varDelta _ { s } = + 0 . 2 4 6$ .

Table 9: Object-size variant results. Mean logit-based accuracy v across all $( s _ { 1 } , s _ { 2 } )$ configurations, accuracy at the two endpoints of the size sweep, and the size-bias gap $\varDelta _ { s }$ . Larger positive $\varDelta _ { s }$ indicates stronger reliance on apparent size as a depth cue. 

<table><tr><td>Model</td><td> $\mathbf{v}$ </td><td> $\mathbf{v}_{\mathbf{s}_1=0.1}$ </td><td> $\mathbf{v}_{\mathbf{s}_1=0.3}$ </td><td> $\Delta_s$ </td></tr><tr><td colspan="5">Qwen2.5-VL-3B</td></tr><tr><td>base</td><td>0.510</td><td>0.507</td><td>0.515</td><td>-0.008</td></tr><tr><td>+ 80k</td><td>0.513</td><td>0.511</td><td>0.516</td><td>-0.005</td></tr><tr><td>+ 400k</td><td>0.500</td><td>0.494</td><td>0.510</td><td>-0.016</td></tr><tr><td>+ 800k</td><td>0.498</td><td>0.489</td><td>0.505</td><td>-0.016</td></tr><tr><td>+ 2M</td><td>0.522</td><td>0.528</td><td>0.509</td><td>+0.018</td></tr><tr><td colspan="5">Molmo-7B</td></tr><tr><td>base</td><td>0.589</td><td>0.609</td><td>0.553</td><td>+0.057</td></tr><tr><td>+ 80k</td><td>0.516</td><td>0.525</td><td>0.503</td><td>+0.022</td></tr><tr><td>+ 400k</td><td>0.605</td><td>0.656</td><td>0.532</td><td>+0.124</td></tr><tr><td>+ 800k</td><td>0.606</td><td>0.648</td><td>0.551</td><td>+0.097</td></tr><tr><td>+ 2M</td><td>0.801</td><td>0.888</td><td>0.643</td><td>+0.246</td></tr><tr><td colspan="5">NVILA-Lite-2B</td></tr><tr><td>base</td><td>0.515</td><td>0.530</td><td>0.487</td><td>+0.043</td></tr><tr><td>+ 80k</td><td>0.526</td><td>0.555</td><td>0.482</td><td>+0.073</td></tr><tr><td>+ 400k</td><td>0.727</td><td>0.815</td><td>0.586</td><td>+0.229</td></tr><tr><td>+ 800k</td><td>0.686</td><td>0.779</td><td>0.557</td><td>+0.222</td></tr><tr><td>+ 2M</td><td>0.828</td><td>0.895</td><td>0.689</td><td>+0.207</td></tr><tr><td>RoboRefer-2B-SFT</td><td>0.804</td><td>0.804</td><td>0.743</td><td>+0.061</td></tr></table>

Similarly, NVILA-Lite-2B (2M) reaches $v = 0 . 8 2 8$ with $\varDelta _ { s } = + 0 . 2 0 7$ . This is the same qualitative pattern observed for vertical position: fine-tuning improves aggregate performance, yet also increases sensitivity to a correlated cue.

3. RoboRefer mitigates size bias while retaining high accuracy. RoboRefer achieves $v = 0 . 8 0 4 .$ , comparable to the strongest fine-tuned checkpoints, while exhibiting a much smaller gap $( \varDelta _ { s } = + 0 . 0 6 1 )$ . This mirrors the trend in the vertical-distance analysis, where RoboRefer also shows a relatively small gap, suggesting greater robustness to shortcut depth cues.

Taken together with the vertical-position intervention, the size-variation results show the same qualitative pattern: accuracy is higher when a simple cue agrees with the true depth ordering, and lower when that cue conflicts with depth. For the models we study, depth judgments therefore remain sensitive to multiple cues that often correlate with depth, including image height and apparent size. These cues can support performance when they are aligned with the underlying 3D layout, but accuracy may decline when that correlation is weakened or reversed. Thus, high average accuracy on depth queries should be interpreted with some caution, as it may not always reflect equally robust 3D spatial reasoning.

# D Additional Details on Contrastive Probing

In this section, we detail the swap pair construction methodology for each spatial category, describe the layer selection methodology, examine the cross-domain consistency of distance coherence, and present full heatmap and PCA visualizations for all model families.

# D.1 Swap Pair Construction

For horizontal and vertical pairs, we construct minimal contrastive pairs by symmetrically swapping the two queried objects: the question “Is A to the left or right of B?” becomes “Is B to the left or right of A?”, which flips the groundtruth label while keeping all other visual context fixed.

For distance (far/close) pairs, the construction differs due to the format of EmbSpatial-Bench depth questions, which are presented as four-choice questions rather than direct relational queries. We identify the target object from the correct answer option, and sample the reference object uniformly at random from the remaining distractor options. The original question asks whether the target is far or close relative to the reference; the swapped question reverses these roles. This yields the same label-flip structure as the horizontal and vertical cases, while adapting to the available annotation format.

# D.2 A Brief Illustration of VD-EI

VD-EI is positive when perspective-aligned pairs (above↔far, below↔close) are more similar than perspective-opposing pairs, and is near zero when the aligned and opposing terms largely cancel. In the extreme, VD-EI tends to be largest when aligned cosines are high while opposing cosines are negative (anti-aligned).

# D.3 Layer Selection Methodology

This section provides supplementary details on layer selection criteria and permodel justifications. The probing code extracts all layers; the user should select the appropriate L∗ from the saved outputs.

Selection criteria. For each model family, we select a single representative layer L∗ at which to compute all probing metrics reported in the main text. The selection is guided by the following criteria, applied in order of priority (i.e., if criteria conflict, earlier criteria take precedence):

1. Axis coherence plateau. Coherence across all three spatial axes (horizontal, vertical, distance) is at or near its peak, indicating that stable axis-level structure has formed in the representation space.

2. VD-EI stability. The VD-Entanglement Index should be at a meaningful plateau rather than in a transient region, ensuring that the selected layer captures the entanglement phenomenon we aim to analyze. When criteria conflict (e.g., VD-EI oscillates), we prioritize criterion (1) and select a layer from the shared high-coherence region across all three axes.   
3. Avoidance of final layers. The selected layer should not fall in the last few layers of the model, as these tend to be optimized for next-token prediction rather than rich representational encoding.

Supporting evidence for intermediate-layer selection. The preference for intermediate layers over final layers is well supported by prior work across both language and vision-language models.

Spatial representations in LLMs have been shown to form in early layers and plateau around the model midpoint; for instance, layer 50 out of 80 (63%) in Llama-2-70b [56] was identified as the primary analysis layer for spatial probing [25]. A systematic study across 32 probing tasks further confirms that intermediate layers encode richer representations than final layers, which become increasingly specialized for output generation [49]. In the multi-modal setting, visual layer selection experiments demonstrate that middle CLIP-ViT [43] layers outperform deep layers for spatial reasoning; on CV-Bench [55], layer 18 out of 24 outperforms the penultimate layer by 3%, suggesting that vision-centric tasks such as spatial and positional reasoning benefit from mid-depth features rather than the deepest ones [10].

Per-model selection. We describe the layer selection rationale for each model family below, with specific coherence and VD-EI values drawn from the full layer-wise trajectories.

Molmo-7B-O-0924 (32 layers): L∗ = 23 (72%). Coherence across all three axes peaks in the L20–25 range, with horizontal ∼0.24, vertical ∼0.55, and distance ∼0.10 at L23. VD-EI reaches a plateau between L15 and L23 (0.5–0.7 for fine-tuned variants, 0.25 for base model), with L23 at the upper end of this range. PCA visualizations confirm clearer cluster separation at L23 compared to neighboring layers.

NVILA-Lite-2B (28 layers): L∗ = 20 (71%). Vertical coherence plateaus between L18 and L25 (0.40–0.60 for fine-tuned variants, 0.80 for RoboRefer). Horizontal coherence is stable across L15–27 (0.20–0.30 for fine-tuned variants, 0.65 for RoboRefer). Distance coherence peaks between L18 and L25 (0.03–0.05 for fine-tuned variants, 0.18 for RoboRefer). VD-EI plateaus at L17–26 (0.5–0.6 for fine-tuned variants), while RoboRefer shows around 0.25. PCA visualizations are shown at L=25, where cluster separation is more visually distinguishable.

Qwen2.5-VL-3B-Instruct (36 layers): L∗ = 28 (78%). Horizontal coherence reaches ∼0.37 at L28. Vertical coherence is ∼0.35 at L28, with a peak of

0.55 at L33. Distance coherence remains low at ∼0.035. VD-EI peaks at L20–22 (0.50), dips around L25, and rebounds to 0.50 at L28. L28 balances meaningful entanglement with reasonable coherence while remaining below the outputspecialized final layers.

Qwen3-VL-235B-A22B-Instruct (94 layers): $L ^ { * } = 8 7 \ ( 9 3 \% )$ . Coherence across all axes forms very late in this model. Horizontal coherence peaks at L83–87 (0.65), vertical peaks at L87–90 (0.63), and distance peaks at L85– 90 (0.23). VD-EI oscillates between 0.3 and 0.7 in the upper layers without a clear plateau. The selected depth of 93% notably exceeds the 71–75% range observed in smaller models, likely reflecting architectural differences: Qwen3- VL-235B-A22B-Instruct is a 94-layer Mixture-of-Experts model with 235B total parameters (22B active), which may delay the formation of stable spatial representations to later layers. We select $L ^ { * } = 8 7$ from the shared high-coherence region across all three axes, despite VD-EI oscillation in the upper layers. Importantly, the resulting cross-model $\mathrm { C o h } _ { \mathrm { D } }$ ranking is robust to alternative valid layer choices (see below).

# D.4 Robustness to Alternative Layer Choices.

Our layer selection follows the predefined protocol above and is applied independently per model family before cross-model comparison. Candidate ranges are defined as the union of layers where $\mathrm { C o h } _ { \mathrm { H } } , \mathrm { C o h } _ { \mathrm { V } }$ , and $\mathrm { C o h _ { D } }$ are near peak; when the criteria differ, we prioritize high axis coherence. To quantify sensitivity, we sample 1K random layers within each candidate range (without refitting) and recompute the cross-model CohD ranking; the resulting rankings show high agreement with the reported ordering (Spearman $\rho = 0 . 9 2 8 )$ ).

# D.5 Cross-Domain Consistency of Distance Coherence

Figure 10 compares $\mathrm { C o h } _ { D }$ computed on the synthetic dataset (SpatialTunnel) with that computed on EmbSpatial-Bench. The absolute values differ between the two domains, as CohD in SpatialTunnel is generally higher; however, the relative ordering of models within each family is largely consistent.

For the NVILA family (highlighted in Figure 10), the ranking is preserved across both datasets: RoboRefer $\mathrm { > N V I L A 2 M > N V I L A }$ 400k $\approx { \mathrm { N V I L A } }$ 800k $> \mathrm { N V I L A 8 0 k } > \mathrm { N V I L A }$ base. The Molmo family exhibits minor rank swaps among adjacent checkpoints, but the overall trend of increasing coherence with training scale, from 80k to 2M, is shared across domains. For the Qwen family, the Qwen2.5-VL-3B scale variants cluster tightly in both settings, while Qwen3- 235B shows a markedly different profile.

These results suggest that the absolute magnitude of $\mathrm { C o h } _ { D }$ can be influenced by the environment, but it provides a reliable relative comparison when models are evaluated under the same data condition. We publicly release both evaluation datasets and the probing pipeline so that future users can benchmark new models under identical conditions and compare against the values reported in this paper, enabling CohD to serve as a reproducible measure of spatial representation quality.

![](images/76534da7f8692c02132767654fc980b2623bb1e1c8c0643997d52a9645db5407.jpg)

<details>
<summary>bar</summary>

| Model          | Distance Coherence in Synthetic Dataset | Distance Coherence in Real Dataset |
| -------------- | --------------------------------------- | ----------------------------------- |
| Molmo base     | 0.10                                    | 0.08                                |
| Molmo 80k      | 0.06                                    | 0.07                                |
| Molmo 400k     | 0.09                                    | 0.10                                |
| Molmo 800k     | 0.10                                    | 0.11                                |
| Molmo 2M       | 0.18                                    | 0.11                                |
| NVILA base     | 0.05                                    | 0.04                                |
| NVILA 80k      | 0.06                                    | 0.07                                |
| NVILA 400k     | 0.11                                    | 0.08                                |
| NVILA 800k     | 0.10                                    | 0.07                                |
| NVILA 2M       | 0.20                                    | 0.10                                |
| RoboRefer      | 0.27                                    | 0.17                                |
| Qwen base      | 0.10                                    | 0.04                                |
| Qwen 80k       | 0.09                                    | 0.04                                |
| Qwen 400k      | 0.09                                    | 0.04                                |
| Qwen 800k      | 0.10                                    | 0.05                                |
| Qwen 2M        | 0.10                                    | 0.05                                |
| Qwen3-235B     | 0.55                                    | 0.22                                |
</details>

Fig. 10: Distance Coherence measured on synthetic (SpatialTunnel) vs. real (EmbSpatial-Bench) datasets. Gray bars denote CohD computed on SpatialTunnel; red dots denote CohD on EmbSpatial-Bench. Although the absolute magnitudes differ across domains, the relative ordering within each model family is largely preserved. The red box highlights the NVILA family, where the ranking is identical across both datasets.

# D.6 Heatmap and PCA Results

This section presents the cross-category similarity heatmaps and PCA visualizations for all model families. Similarity is computed as the cosine similarity between each category’s mean delta vector.

As shown in the heatmap results (Figure 11, 12, 13), the similarity between opposing categories on the same axis (e.g., left and right on the horizontal axis) is consistently near −1, indicating that the model encodes opposite spatial directions as antiparallel vectors in representation space. Additionally, similarity between horizontal and the other axes (i.e., vertical and distance) is close to zero, suggesting that the horizontal axis is encoded independently from both vertical and distance representations. However, between the vertical and distance axes, the perspective-aligned pairs above↔far and below↔close exhibit meaningful positive similarity in the range of 0.1–0.65 across all models. This confirms that vertical and distance representations are directionally coupled, consistent with the entanglement phenomenon described in the main text.

![](images/fa934635da2a54ef76bacff9c818496c10f14107b92acbb627b3126791eeeac0.jpg)

<details>
<summary>heatmap</summary>

MOLMO (vanilla) — Delta Heatmap L23 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.9755 | 0.0826 | -0.0989 | -0.1166 | 0.0739 |
| | -0.9755 | 1.0 | -0.1241 | 0.1397 | 0.0809 | -0.0401 |
| | 0.0826 | -0.1241 | 1.0 | -0.984 | 0.3081 | -0.3054 |
| | -0.0989 | 0.1397 | -0.984 | 1.0 | -0.3212 | 0.3072 |
| | -0.1166 | 0.0809 | 0.3081 | -0.3212 | 1.0 | -0.9076 |
| | 0.0739 | -0.0401 | -0.3054 | 0.3072 | -0.9076 | 1.0 |
</details>

![](images/ea8ee1374947942a5020f700d82e86c838cb2a4ee7f5d9ac74a98221b815dc2a.jpg)

![](images/860790aae6c38481d649317ffb4603f8de3306e2fa2fb5d5d014b5cb4bd02f09.jpg)

![](images/4376f196ec3d145129dd92af6b8d6d8a42049bf53f778dc1d8d561de0050aace.jpg)

![](images/cf109d2a47d38191902610648aa8b44f0dcb2ddaaee6b5002817c08abe0d3a7c.jpg)

<details>
<summary>heatmap</summary>

MOLMO (2m) — Delta Heatmap L23 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.9862 | 0.0926 | -0.1094 | -0.04 | -0.0187 |
| | -0.9862 | 1.0 | -0.0914 | 0.1102 | -0.0072 | 0.0583 |
| | 0.0926 | -0.0914 | 1.0 | -0.9967 | 0.4923 | -0.4447 |
| | -0.1094 | 0.1102 | -0.9967 | 1.0 | -0.4865 | 0.4372 |
| | -0.04 | -0.0072 | 0.4923 | -0.4865 | 1.0 | -0.9317 |
| | -0.0187 | 0.0583 | -0.4447 | 0.4372 | -0.9317 | 1.0 |
</details>

Fig. 11: Cross-category similarity heatmaps for the Molmo family. Each cell shows the cosine similarity between mean delta vectors of two categories. Variants range from vanilla (base Molmo-7B) to 2M (SFT with 2M-sample data mix).

![](images/50d9c7a8eb416066513047665453daa4d4f5db0dbe5e66021b28713eb1a43823.jpg)

<details>
<summary>heatmap</summary>

NVILA (vanilla) — Delta Heatmap L20 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.9946 | 0.1269 | -0.1586 | -0.0479 | 0.0083 |
| | -0.9946 | 1.0 | -0.1317 | 0.1627 | 0.014 | 0.0226 |
| | 0.1269 | -0.1317 | 1.0 | -0.9891 | 0.5845 | -0.5811 |
| | -0.1586 | 0.1627 | -0.9891 | 1.0 | -0.5912 | 0.5759 |
| | -0.0479 | 0.014 | 0.5845 | -0.5912 | 1.0 | -0.8439 |
| | 0.0083 | 0.0226 | -0.5811 | 0.5759 | -0.8439 | 1.0 |
</details>

![](images/e13bc829645eafc94c6239f7bb424013c126de89c0947eb21f46b8822e7776a7.jpg)

<details>
<summary>heatmap</summary>

NVILA (80k) — Delta Heatmap L20 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| left | 1.0 | -0.9888 | 0.2513 | -0.2853 | -0.0739 | -0.0319 |
| right | -0.9888 | 1.0 | -0.2655 | 0.2998 | 0.0342 | 0.0692 |
| above | 0.2513 | -0.2655 | 1.0 | -0.994 | 0.6428 | -0.6188 |
| below | -0.2853 | 0.2998 | -0.994 | 1.0 | -0.6386 | 0.6139 |
| far | -0.0739 | 0.0342 | 0.6428 | -0.6386 | 1.0 | -0.9187 |
| close | -0.0319 | 0.0692 | -0.6188 | 0.6139 | -0.9187 | 1.0 |
</details>

![](images/895e21f0094a907af164574e37614d0fbea257d13db3ba48d023d30f65b5ff86.jpg)

<details>
<summary>heatmap</summary>

NVILA (400k) — Delta Heatmap L20 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.9883 | 0.0935 | -0.1388 | -0.1374 | 0.0076 |
| | -0.9883 | 1.0 | -0.0906 | 0.1367 | 0.094 | 0.0314 |
| | above | 0.0935 | -0.0906 | 1.0 | -0.9958 | 0.6687 |
| | below | -0.1388 | 0.1367 | -0.9958 | 1.0 | -0.6624 |
| | far | -0.1374 | 0.094 | 0.6687 | -0.6624 | 1.0 |
| | close | 0.0076 | 0.0314 | -0.6188 | 0.6166 | -0.925 |
| ...
</details>

![](images/1a57be8b4fad2a8f25b2ee6c93d07b31096fd0a8c80f01a28decaa0a4d88e317.jpg)

<details>
<summary>heatmap</summary>

NVILA (800k) — Delta Heatmap L20 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| left | 1.0 | -0.9906 | 0.1516 | -0.1972 | -0.0762 | -0.006 |
| right | -0.9906 | 1.0 | -0.1386 | 0.1846 | 0.0586 | 0.0244 |
| above | 0.1516 | -0.1386 | 1.0 | -0.9958 | 0.6469 | -0.5713 |
| below | -0.1972 | 0.1846 | -0.9958 | 1.0 | -0.6395 | 0.5639 |
| far | -0.0762 | 0.0586 | 0.6469 | -0.6395 | 1.0 | -0.9229 |
| close | -0.006 | 0.0244 | -0.5713 | 0.5639 | -0.9229 | 1.0 |
</details>

![](images/f1ccf0b57ea76ac60d412b50f8e5ad8114b9a24b3d4dec1117df94a6351445a5.jpg)

![](images/d337b4c84cc93bd15d52d317078caccb56a493ccf02b0c26a5091612f94b5e54.jpg)

<details>
<summary>heatmap</summary>

NVILA (roborefer) — Delta Heatmap L20 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.9979 | 0.092 | -0.1132 | -0.1467 | 0.1256 |
| | -0.9979 | 1.0 | -0.0795 | 0.1019 | 0.1301 | -0.1122 |
| | 0.092 | -0.0795 | 1.0 | -0.998 | 0.382 | -0.3681 |
| | -0.1132 | 0.1019 | -0.998 | 1.0 | -0.3897 | 0.3716 |
| | -0.1467 | 0.1301 | 0.382 | -0.3897 | 1.0 | -0.9596 |
| | 0.1256 | -0.1122 | -0.3681 | 0.3716 | -0.9596 | 1.0 |
</details>

Fig. 12: Cross-category similarity heatmaps for the NVILA family. Variants include NVILA-Lite-2B from vanilla (base) through 2M (SFT), plus RoboRefer (RoboRefer-2B-SFT).

![](images/466d984b5d49c01844ac956c573bb14378be68c25904d0eb78d73016b1a55323.jpg)

<details>
<summary>heatmap</summary>

QWEN (vanilla) — Delta Heatmap L28 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.9922 | 0.173 | -0.2745 | -0.0945 | 0.0054 |
| | -0.9922 | 1.0 | -0.1849 | 0.2875 | 0.0718 | 0.0051 |
| | 0.173 | -0.1849 | 1.0 | -0.9817 | 0.4403 | -0.4082 |
| | -0.2745 | 0.2875 | -0.9817 | 1.0 | -0.4438 | 0.427 |
| | -0.0945 | 0.0718 | 0.4403 | -0.4438 | 1.0 | -0.8291 |
| | 0.0054 | 0.0051 | -0.4082 | 0.427 | -0.8291 | 1.0 |
</details>

![](images/c24933047a4ef3529be3905e78a83b7452e0be2b690105a148a84377df0944d0.jpg)

<details>
<summary>heatmap</summary>

QWEN (80k) — Delta Heatmap L28 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.9923 | 0.1669 | -0.2652 | -0.0881 | -0.0065 |
| | -0.9923 | 1.0 | -0.1822 | 0.2812 | 0.0657 | 0.0167 |
| above | 0.1669 | -0.1822 | 1.0 | -0.9824 | 0.4606 | -0.4194 |
| below | -0.2652 | 0.2812 | -0.9824 | 1.0 | -0.4606 | 0.4347 |
| far | -0.0881 | 0.0657 | 0.4606 | -0.4606 | 1.0 | -0.8237 |
| close | -0.0065 | 0.0167 | -0.4194 | 0.4347 | -0.8237 | 1.0 |
</details>

![](images/b1f797a9177e55d967a6fac848c739ea20e368269427f47590d4e1951f83dfcf.jpg)

<details>
<summary>heatmap</summary>

QWEN (400k) — Delta Heatmap L28 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| left | 1.0 | -0.9936 | 0.1783 | -0.2752 | -0.0796 | -0.0339 |
| right | -0.9936 | 1.0 | -0.1956 | 0.2933 | 0.0579 | 0.0433 |
| above | 0.1783 | -0.1956 | 1.0 | -0.9858 | 0.488 | -0.4032 |
| below | -0.2752 | 0.2933 | -0.9858 | 1.0 | -0.4851 | 0.4181 |
| far | -0.0796 | 0.0579 | 0.488 | -0.4851 | 1.0 | -0.8153 |
| close | -0.0339 | 0.0433 | -0.4032 | 0.4181 | -0.8153 | 1.0 |
</details>

![](images/f0ded240dcb011e097325b8d22ee408662b6f635ade56d1737f13fc6883a8fbf.jpg)

<details>
<summary>heatmap</summary>

QWEN (800k) — Delta Heatmap L28 (all pairs)
| | left | right | above | below | far | close |
|---|---|---|---|---|---|---|
| | 1.0 | -0.994 | 0.1731 | -0.2631 | -0.0711 | -0.0477 |
| | -0.994 | 1.0 | -0.1905 | 0.2815 | 0.0484 | 0.058 |
| | 0.1731 | -0.1905 | 1.0 | -0.9891 | 0.4773 | -0.3862 |
| | -0.2631 | 0.2815 | -0.9891 | 1.0 | -0.4759 | 0.4003 |
| | -0.0711 | 0.0484 | 0.4773 | -0.4759 | 1.0 | -0.8197 |
| | -0.0477 | 0.058 | -0.3862 | 0.4003 | -0.8197 | 1.0 |
</details>

![](images/59b43219028c774b0775c5f98b20ddbbb705054fcd73ef83ac492f60ea54ebca.jpg)

![](images/9a5506f57fe2ec4b68ce50af644d03e0293cbf38055b3d81f904f0a27bcc37f6.jpg)

<details>
<summary>heatmap</summary>

QWEN (qwen3_235b) — Delta Heatmap L87 (all pairs)
| | left | right | above | under | far | close |
|---|---|---|---|---|---|---|
| left | 1.0 | -0.9984 | 0.0126 | -0.0221 | -0.0331 | -0.0008 |
| right | -0.9984 | 1.0 | -0.0155 | 0.0253 | 0.0279 | 0.0055 |
| above | 0.0126 | -0.0155 | 1.0 | -0.9966 | 0.134 | -0.1067 |
| under | -0.0221 | 0.0253 | -0.9966 | 1.0 | -0.1271 | 0.0997 |
| far | -0.0331 | 0.0279 | 0.134 | -0.1271 | 1.0 | -0.9509 |
| close | -0.0008 | 0.0055 | -0.1067 | 0.0997 | -0.9509 | 1.0 |
</details>

Fig. 13: Cross-category similarity heatmaps for the Qwen family. Variants include Qwen2.5-VL-3B-Instruct (vanilla through 2M) and Qwen3-VL-235B-A22B-Instruct.

![](images/102f723356c9fba01771f8a309d79216546cb96fd16f4b2ea7e578d770902fff.jpg)

<details>
<summary>scatter</summary>

| PC1 (15.7%) | PC2 (7.2%) | Category |
| ----------- | ---------- | -------- |
| -15         | 0          | below    |
| -10         | 2.5        | above    |
| -5          | 5          | far      |
| 0           | 7.5        | left     |
| 5           | 5          | close    |
| 10          | 2.5        | far      |
| 15          | 0          | above    |
</details>

![](images/402a0c135d259d2232e6758c222fc393547f06d73c3a7a921cb3ca8ef29fb25f.jpg)

<details>
<summary>scatter</summary>

| PC1 (25.4%) | PC2 (6.9%) | Category |
| ----------- | ---------- | -------- |
| -10         | 2          | below    |
| -5          | 0          | left     |
| 0           | -2         | right    |
| 5           | 0          | above    |
| 10          | 2          | close    |
</details>

![](images/f2051f92c6659c0d32c6979d0d4731d08912a042845ff26d75bce998ab03e06d.jpg)

![](images/abaae609495e7280844adf8c71dfaa25cc65509c0da092b39f8eab3accf234bd.jpg)

![](images/a057ec0dbdb1e37d8e7862bb0ff59c2f2595e0fa580f549a2e62af17afa8104f.jpg)

<details>
<summary>scatter</summary>

| PC1 | PC2 | Category |
| --- | --- | --- |
| -20 | 0 | above |
| -15 | 5 | above |
| -10 | 10 | above |
| -5 | 15 | above |
| 0 | 20 | above |
| 5 | 25 | above |
| 10 | 30 | above |
| 15 | 35 | above |
| 20 | 40 | above |
| -20 | -5 | far |
| -15 | -10 | far |
| -10 | -15 | far |
| -5 | -20 | far |
| 0 | -25 | far |
| 5 | -30 | far |
| 10 | -35 | far |
| 15 | -40 | far |
| 20 | -45 | far |
| -20 | 10 | left |
| -15 | 15 | left |
| -10 | 20 | left |
| -5 | 25 | left |
| 0 | 30 | left |
| 5 | 35 | left |
| 10 | 40 | left |
| 15 | 45 | left |
| 20 | 50 | left |
| -20 | -10 | below |
| -15 | -15 | below |
| -10 | -20 | below |
| -5 | -25 | below |
| 0 | -30 | below |
| 5 | -35 | below |
| 10 | -40 | below |
| 15 | -45 | below |
| 20 | -50 | below |
| -20 | 5 | close |
| -15 | 10 | close |
| -10 | 15 | close |
| -5 | 20 | close |
| 0 | 25 | close |
| 5 | 30 | close |
| 10 | 35 | close |
| 15 | 40 | close |
| 20 | 45 | close |
| -20 | 15 | right |
| -15 | 20 | right |
| -10 | 25 | right |
| -5 | 30 | right |
| 0 | 35 | right |
| 5 | 40 | right |
| 10 | 45 | right |
| 15 | 50 | right |
| 20 | 55 | right |
| -20 | -10 | above |
| -15 | -15 | above |
| -10 | -20 | above |
| -5 | -25 | above |
| 0 | -30 | above |
| 5 | -35 | above |
| 10 | -40 | above |
| 15 | -45 | above |
| 20 | -50 | above |
| -20 | 10 | far |
| -15 | 15 | far |
| -10 | 20 | far |
| -5 | 25 | far |
| 0 | 30 | far |
| 5 | 35 | far |
| 10 | 40 | far |
| 15 | 45 | far |
| 20 | 50 | far |
| -20 | -15 | close |
| -15 | -20 | close |
| -10 | -25 | close |
| -5 | -30 | close |
| 0 | -35 | close |
| 5 | -40 | close |
| 10 | -45 | close |
| 15 | -50 | close |
| 20 | -55 | close |
| -20 | 15 | right |
| -15 | 20 | right |
| -10 | 25 | right |
| -5 | 30 | right |
| 0 | 35 | right |
| 5 | 40 | right |
| 10 | 45 | right |
| 15 | 50 | right |
| 20 | 55 | right |
</details>

Fig. 14: 2D PCA of delta vectors for the Molmo family. Each point represents a per-sample delta vector, colored by spatial category. Opposing categories (e.g., left vs. right) separate along shared principal components, while far/close overlap with above/below, reflecting vertical-distance entanglement.

![](images/fb49c5f73f9fb0be6bb7c6184f2260aad40045b22159c397347b2f51f77dd328.jpg)

<details>
<summary>scatter</summary>

| PC1 (25.3%) | PC2 (14.4%) | Category |
| ----------- | ----------- | -------- |
| -40         | 0           | left     |
| -20         | 0           | right    |
| 0           | 0           | above    |
| 20          | 0           | below    |
| 40          | 0           | close    |
</details>

![](images/ac553b3d685f63cf92a6ba53dd49a44f7bed28cccba6a57dadcc677db369ee5f.jpg)

<details>
<summary>scatter</summary>

| PC1 (31.9%) | PC2 (16.3%) | Category |
| ----------- | ----------- | -------- |
| -60         | 40          | left     |
| -40         | 20          | right    |
| -20         | 0           | above    |
| 0           | -20         | far      |
| 20          | -40         | close    |
| 40          | -20         | below    |
| 60          | 0           | far      |
</details>

![](images/771cf53601aeb1af1ca5b08d64c99a9e0e2ae7ea9e993b7daa62f389e604eb68.jpg)

<details>
<summary>scatter</summary>

| PC1 (32.6%) | PC2 (10.2%) | Category |
| ----------- | ----------- | -------- |
| -80         | 0           | above    |
| -60         | 20          | far      |
| -40         | 40          | left     |
| -20         | 60          | below    |
| 0           | 80          | far      |
| 20          | 100         | right    |
| 40          | 120         | above    |
| 60          | 140         | below    |
| 80          | 160         | far      |
</details>

![](images/c7e8a142c0badac168501fa3e319ecbeac08a7cccd47c5df92212ef48434a6f0.jpg)

<details>
<summary>scatter</summary>

| PC1 (31.4%) | PC2 (13.3%) | Category |
| ----------- | ----------- | -------- |
| -100        | 0           | below    |
| -75         | 25          | left     |
| -50         | 50          | right    |
| -25         | 75          | above    |
| 0           | 100         | close    |
| 25          | 75          | far      |
| 50          | 50          | below    |
| 75          | 25          | left     |
| 100         | 0           | right    |
</details>

![](images/38f64aba3cda641c7406dbe6db90f9a3966d858b7a40043ec5dd1b851f2857bf.jpg)

<details>
<summary>scatter</summary>

| PC1 (26.8%) | PC2 (14.0%) | Category |
| ----------- | ----------- | -------- |
| -75         | 0           | below    |
| -50         | 25          | left     |
| -25         | 50          | right    |
| 0           | 75          | above    |
| 25          | 50          | far      |
| 50          | 25          | close    |
| 75          | 0           | above    |
</details>

![](images/f775a0ea3b62c7fb2b6ffe1deb23c126827038e9e707d1ce4313a6d017ba0421.jpg)

<details>
<summary>scatter</summary>

| PC1 (35.8%) | PC2 (28.7%) | Category |
| ----------- | ----------- | -------- |
| -80         | 80          | left     |
| -60         | 60          | left     |
| -40         | 40          | left     |
| -20         | 20          | left     |
| 0           | 0           | far      |
| 20          | -20         | far      |
| 40          | -40         | far      |
| 60          | -60         | far      |
| 80          | -80         | far      |
| -80         | 80          | above    |
| -60         | 60          | above    |
| -40         | 40          | above    |
| -20         | 20          | above    |
| 0           | 0           | above    |
| 20          | -20         | above    |
| 40          | -40         | above    |
| 60          | -60         | above    |
| 80          | -80         | above    |
| -80         | 80          | close    |
| -60         | 60          | close    |
| -40         | 40          | close    |
| -20         | 20          | close    |
| 0           | 0           | close    |
| 20          | -20         | close    |
| 40          | -40         | close    |
| 60          | -60         | close    |
| 80          | -80         | close    |
| -80         | 80          | far      |
| -60         | 60          | far      |
| -40         | 40          | far      |
| -20         | 20          | far      |
| 0           | 0           | far      |
| 20          | -20         | far      |
| 40          | -40         | far      |
| 60          | -60         | far      |
| 80          | -80         | far      |
| -80         | 80          | below    |
| -60         | 60          | below    |
| -40         | 40          | below    |
| -20         | 20          | below    |
| 0           | 0           | below    |
| 20          | -20         | below    |
| 40          | -40         | below    |
| 60          | -60         | below    |
| 80          | -80         | below    |
| -80         | 80          | left     |
| -60         | 60          | left     |
| -40         | 40          | left     |
| -20         | 20          | left     |
| 0           | 0           | left     |
| 20          | -20         | left     |
| 40          | -40         | left     |
| 60          | -60         | left     |
| 80          | -80         | left     |
</details>

Fig. 15: 2D PCA of delta vectors for the NVILA family. RoboRefer shows notably tighter distance-axis clusters (far /close) separated from vertical categories, consistent with its higher CohD and lower VD-EI.

![](images/db651f8ebb37c5df7b65e639da29f121a07847f61befe08d99eb94081a49349e.jpg)

<details>
<summary>scatter</summary>

| PC1 (24.8%) | PC2 (12.3%) | Category |
| ----------- | ----------- | -------- |
| -60         | 40          | left     |
| -40         | 20          | left     |
| -20         | 0           | far      |
| 0           | -20         | far      |
| 20          | -40         | far      |
| 40          | -60         | far      |
| 60          | -80         | far      |
| -60         | 40          | above    |
| -40         | 20          | above    |
| -20         | 0           | above    |
| 0           | -20         | above    |
| 20          | -40         | above    |
| 40          | -60         | above    |
| 60          | -80         | above    |
| -60         | 40          | close    |
| -40         | 20          | close    |
| -20         | 0           | close    |
| 0           | -20         | close    |
| 20          | -40         | close    |
| 40          | -60         | close    |
| 60          | -80         | close    |
| -60         | 40          | below    |
| -40         | 20          | below    |
| -20         | 0           | below    |
| 0           | -20         | below    |
| 20          | -40         | below    |
| 40          | -60         | below    |
| 60          | -80         | below    |
</details>

![](images/709a940bbaa1f4f11160ff625e50d6ac93497a66c17fbb479c1fd4d2e16d7624.jpg)

![](images/e89781e86850df221f36ba9d6c3035e1f65ac1950947b4814b87c4c01e7e8ab7.jpg)

<details>
<summary>scatter</summary>

| PC1 (28.4%) | PC2 (15.8%) | Category |
| ----------- | ----------- | -------- |
| -40         | 30          | left     |
| -20         | 20          | below    |
| 0           | 0           | far      |
| 20          | -10         | close    |
| 40          | -30         | left     |
</details>

![](images/d42047e241957601833e082f11df3186d2521b311dd804f904414162cb91a2b3.jpg)

<details>
<summary>scatter</summary>

| PC1 (29.5%) | PC2 (19.2%) | Category |
| ----------- | ----------- | -------- |
| -40         | 40          | left     |
| -20         | 20          | below    |
| 0           | 0           | far      |
| 20          | -20         | close    |
| 40          | -40         | left     |
</details>

![](images/55a07ea7ab4aab3073116a1ce025144ebc96f919bf93635ee68fca130557b22b.jpg)

<details>
<summary>scatter</summary>

| PC1 (31.4%) | PC2 (21.5%) | Category |
| ----------- | ----------- | -------- |
| -40         | 40          | left     |
| -20         | 20          | below    |
| 0           | 0           | far      |
| 20          | -20         | above    |
| 40          | -40         | close    |
</details>

![](images/b7242f0296be15ccb4b92c6cd339eef22a4f3c3d793ab66ed4ffe70680f6221b.jpg)

<details>
<summary>scatter</summary>

| PC1 (38.5%) | PC2 (19.2%) | Category |
| ----------- | ----------- | -------- |
| -60         | 40          | above    |
| -40         | 20          | above    |
| -20         | 0           | above    |
| 0           | -20         | above    |
| 20          | -40         | above    |
| 40          | -20         | above    |
| 60          | 0           | above    |
| -60         | 0           | right    |
| -40         | 10          | right    |
| -20         | 20          | right    |
| 0           | 30          | far      |
| 20          | 40          | far      |
| 40          | 50          | far      |
| 60          | 60          | far      |
| -60         | 10          | left     |
| -40         | 20          | left     |
| -20         | 30          | left     |
| 0           | 40          | left     |
| 20          | 50          | left     |
| 40          | 60          | left     |
| 60          | 70          | left     |
| -60         | -10         | below    |
| -40         | -20         | below    |
| -20         | -30         | below    |
| 0           | -40         | below    |
| 20          | -50         | below    |
| 40          | -60         | below    |
| 60          | -70         | below    |
| -60         | -15         | close     |
| -40         | -25         | close     |
| -20         | -35         | close     |
| 0           | -45         | close     |
| 20          | -55         | close     |
| 40          | -65         | close     |
| 60          | -75         | close     |
</details>

Fig. 16: 2D PCA of delta vectors for the Qwen family. Variants include Qwen2.5-VL-3B-Instruct and Qwen3-VL-235B-A22B-Instruct. Qwen3-VL-235B exhibits markedly cleaner cluster separation across all three axes.

![](images/cca26d25ed30fbd9e2a100b0dea9990f1758b1672a91e8a85c86fb8ff0edb06e.jpg)

<details>
<summary>scatter_3d</summary>

| PC1 (15.7%) | PC2 (7.2%) | PC3 (5.6%) |
| ----------- | ---------- | ---------- |
| -15         | 0          | 7.5        |
| -10         | 5          | 5.0        |
| -5          | 10         | 2.5        |
| 0           | 15         | 0.0        |
| 5           | 20         | -2.5       |
| 10          | 25         | -5.0       |
| 15          | 30         | -7.5       |
| 20          | 35         | -10.0      |
| 25          | 40         | -12.5      |
| 30          | 45         | -15.0      |
| 35          | 50         | -17.5      |
| 40          | 55         | -20.0      |
| 45          | 60         | -22.5      |
| 50          | 65         | -25.0      |
| 55          | 70         | -27.5      |
| 60          | 75         | -30.0      |
| 65          | 80         | -32.5      |
| 70          | 85         | -35.0      |
| 75          | 90         | -37.5      |
| 80          | 95         | -40.0      |
| 85          | 100        | -42.5      |
| 90          | 105        | -45.0      |
| 95          | 110        | -47.5      |
| 100         | 115        | -50.0      |
| 105         | 120        | -52.5      |
| 110         | 125        | -55.0      |
| 115         | 130        | -57.5      |
| 120         | 135        | -60.0      |
| 125         | 140        | -62.5      |
| 130         | 145        | -65.0      |
| 135         | 150        | -67.5      |
| 140         | 155        | -70.0      |
| 145         | 160        | -72.5      |
| 150         | 165        | -75.0      |
| 155         | 170        | -77.5      |
| 160         | 175        | -80.0      |
| 165         | 180        | -82.5      |
| 170         | 185        | -85.0      |
| 175         | 190        | -87.5      |
| 180         | 195        | -90.0      |
| 185         | 200        | -92.5      |
| 190         | 205        | -95.0      |
| 195         | 210        | -97.5      |
| 200         | 215        | -100.0     |
| 205         | 220        | -102.5     |
| 210         | 225        | -105.0     |
| 215         | 230        | -107.5     |
| 220         | 235        | -110.0     |
| 225         | 240        | -112.5     |
| 230         | 245        | -115.0     |
| 235         | 250        | -117.5     |
| 240         | 255        | -120.0     |
| 245         | 260        | -122.5     |
| 250         | 265        | -125.0     |
| 255         | 270        | -127.5     |
| 260         | 275        | -130.0     |
| 265         | 280        | -132.5     |
| 270         | 285        | -135.0     |
| 275         | 290        | -137.5     |
| 280         | 295        | -140.0     |
| 285         | 300        | -142.5     |
| 290         | 305        | -145.0     |
| 295         | 310        | -147.5     |
| 300         | 315        | -150.0     |
| 305         | 320        | -152.5     |
| 310         | 325        | -155.0     |
| 315         | 330        | -157.5     |
| 320         | 335        | -160.0     |
| 325         | 340        | -162.5     |
| 330         | 345        | -165.0     |
| 335         | 350        | -167.5     |
| 340         | 355        | -170.0     |
| 345         | 360        | -172.5     |
| 350         | 365        | -175.0     |
| 355         | 370        | -177.5     |
| 360         | 375        | -180.0     |
| 365         | 380        | -182.5     |
| 370         | 385        | -185.0     |
| 375         | 390        | -187.5     |
| 380         | 395        | -190.0     |
| 385         | 400        | -192.5     |
| 390         | 405        | -195.0     |
| 395         | 410        | -197.5     |
| 400         | 415        | -200.0     |
| 405         | 420        | -202.5     |
| 410         | 425        | -205.0     |
| 415         | 430        | -207.5     |
| 420         | 435        | -210.0     |
| 425         | 440        | -212.5     |
| 430         | 445        | -215.0     |
| 435         | 450        | -217.5     |
| 440         | 455        | -220.0     |
| 445         | 460        | -222.5     |
| 450         | 465        | -225.0     |
| 455         | 470        | -227.5     |
| 460         | 475        | -230.0     |
| 465         | 480        | -232.5     |
| 470         | 485        | -235.0     |
| 475         | 490        | -237.5     |
| 480         | 495        | -240.0     |
| 485         | 500        | -242.5     |
| 490         | 505        | -245.0     |
| 495         | 510        | -247.5     |
| 500         | 515        | -250.0     |
| 505         | 520        | -252.5     |
| Note: The data is already in CSV format with the original table as shown in the code itself. The actual data will vary due to the random nature of the data generation process.
</details>

![](images/cedcd0f7c8363b52f06cb35967d35edf10c03b91fd25030da057ecacf581105b.jpg)

![](images/c720e97337371ca14137387778dabdceea78ce2bee0091416de048d9f39df459.jpg)

<details>
<summary>scatter</summary>

| PC1       | PC2       | PC3       | Category |
| --------- | --------- | --------- | -------- |
| -20       | 0         | 10        | above    |
| -15       | 5         | 5         | above    |
| -10       | 10        | 0         | above    |
| -5        | 15        | -5        | above    |
| 0         | 20        | -10       | above    |
| 5         | 25        | -15       | above    |
| 10        | 30        | -20       | above    |
| 15        | 35        | -25       | above    |
| 20        | 40        | -30       | above    |
| 25        | 45        | -35       | above    |
| 30        | 50        | -40       | above    |
| 35        | 55        | -45       | above    |
| 40        | 60        | -50       | above    |
| 45        | 65        | -55       | above    |
| 50        | 70        | -60       | above    |
| 55        | 75        | -65       | above    |
| 60        | 80        | -70       | above    |
| 65        | 85        | -75       | above    |
| 70        | 90        | -80       | above    |
| 75        | 95        | -85       | above    |
| 80        | 100       | -90       | above    |
| 85        | 105       | -95       | above    |
| 90        | 110       | -100      | above    |
| 95        | 115       | -105      | above    |
| 100       | 120       | -110      | above    |
| 105       | 125       | -115      | above    |
| 110       | 130       | -120      | above    |
| 115       | 135       | -125      | above    |
| 120       | 140       | -130      | above    |
| 125       | 145       | -135      | above    |
| 130       | 150       | -140      | above    |
| 135       | 155       | -145      | above    |
| 140       | 160       | -150      | above    |
| 145       | 165       | -155      | above    |
| 150       | 170       | -160      | above    |
| 155       | 175       | -165      | above    |
| 160       | 180       | -170      | above    |
| 165       | 185       | -175      | above    |
| 170       | 190       | -180      | above    |
| 175       | 195       | -185      | above    |
| 180       | 200       | -190      | above    |
| 185       | 205       | -195      | above    |
| 190       | 210       | -200      | above    |
| 195       | 215       | -205      | above    |
| 200       | 220       | -210      | above    |
| 205       | 225       | -215      | above    |
| 210       | 230       | -220      | above    |
| 215       | 235       | -225      | above    |
| 220       | 240       | -230      | above    |
| 225       | 245       | -235      | above    |
| 230       | 250       | -240      | above    |
| 235       | 255       | -245      | above    |
| 240       | 260       | -250      | above    |
| 245       | 265       | -255      | above    |
| 250       | 270       | -260      | above    |
| 255       | 275       | -265      | above    |
| 260       | 280       | -270      | above    |
| 265       | 285       | -275      | above    |
| 270       | 290       | -280      | above    |
| 275       | 295       | -285      | above    |
| 280       | 300       | -290      | above    |
| 285       | 305       | -295      | above    |
| 290       | 310       | -300      | above    |
| 295       | 315       | -305      | above    |
| 300       | 320       | -310      | above    |
| 305       | 325       | -315      | above    |
| 310       | 330       | -320      | above    |
| 315       | 335       | -325      | above    |
| 320       | 340       | -330      | above    |
| 325       | 345       | -335      | above    |
| 330       | 350       | -340      | above    |
| 335       | 355       | -345      | above    |
| 340       | 360       | -350      | above    |
| 345       | 365       | -355      | above    |
| 350       | 370       | -360      | above    |
| 355       | 375       | -365      | above    |
| 360       | 380       | -370      | above    |
| 365       | 385       | -375      | above    |
| 370       | 390       | -380      | above    |
| 375       | 395       | -385      | above    |
| 380       | 400       | -390      | above    |
| 385       | 405       | -395      | above    |
| 390       | 410       | -400      | above    |
| 395       | 415       | -405      | above    |
| 400       | 420       | -410      | above    |
| 405       | 425       | -415      | above    |
| 410       | 430       | -420      | above    |
| 415       | 435       | -425      | above    |
| 420       | 440       | -430      | above    |
| 425       | 445       | -435      | above    |
| 430       | 450       | -440      | above    |
| 435       | 455       | -445      | above    |
| 440       | 460       | -450      | above    |
| 445       | 465       | -455      | above    |
| 450       | 470       | -460      | above    |
| 455       | 475       | -465      | above    |
| 460       | 480       | -470      | above    |
| 465       | 485       | -475      | above    |
| 470       | 490       | -480      | above    |
| 475       | 495       | -485      | above    |
| 480       | 500       | -490      | above    |
| 485       | 505       | -495      | above    |
| 490       | 510       | -500      | below    |
| ... (additional values in the chart) are not provided in the code.
</details>

![](images/c21244302f9f4008e6ff3b3ffe801d34e14cace135bc713c8cb8117e58d24ab9.jpg)

<details>
<summary>scatter_3d</summary>

| PC1 (35.9%) | PC2 (11.4%) | PC3 (5.7%) |
| ----------- | ----------- | ---------- |
| -20         | 0           | 15         |
| -10         | 5           | 10         |
| 0           | 10          | 5          |
| 10          | 15          | 0          |
| 20          | 20          | -5         |
| 30          | 25          | -10        |
| 40          | 30          | -15        |
| 50          | 35          | -20        |
| 60          | 40          | -25        |
| 70          | 45          | -30        |
| 80          | 50          | -35        |
| 90          | 55          | -40        |
| 100         | 60          | -45        |
| 110         | 65          | -50        |
| 120         | 70          | -55        |
| 130         | 75          | -60        |
| 140         | 80          | -65        |
| 150         | 85          | -70        |
| 160         | 90          | -75        |
| 170         | 95          | -80        |
| 180         | 100         | -85        |
| 190         | 105         | -90        |
| 200         | 110         | -95        |
| 210         | 115         | -100       |
| 220         | 120         | -105       |
| 230         | 125         | -110       |
| 240         | 130         | -115       |
| 250         | 135         | -120       |
| 260         | 140         | -125       |
| 270         | 145         | -130       |
| 280         | 150         | -135       |
| 290         | 155         | -140       |
| 300         | 160         | -145       |
| 310         | 165         | -150       |
| 320         | 170         | -155       |
| 330         | 175         | -160       |
| 340         | 180         | -165       |
| 350         | 185         | -170       |
| 360         | 190         | -175       |
| 370         | 195         | -180       |
| 380         | 200         | -185       |
| 390         | 205         | -190       |
| 400         | 210         | -195       |
| 410         | 215         | -200       |
| 420         | 220         | -205       |
| 430         | 225         | -210       |
| 440         | 230         | -215       |
| 450         | 235         | -220       |
| 460         | 240         | -225       |
| 470         | 245         | -230       |
| 480         | 250         | -235       |
| 490         | 255         | -240       |
| 500         | 260         | -245       |
| 510         | 265         | -250       |
| 520         | 270         | -255       |
| 530         | 275         | -260       |
| 540         | 280         | -265       |
| 550         | 285         | -270       |
| 560         | 290         | -275       |
| 570         | 295         | -280       |
| 580         | 300         | -285       |
| 590         | 305         | -290       |
| 600         | 310         | -295       |
| 610         | 315         | -300       |
| 620         | 320         | -305       |
| 630         | 325         | -310       |
| 640         | 330         | -315       |
| 650         | 335         | -320       |
| 660         | 340         | -325       |
| 670         | 345         | -330       |
| 680         | 350         | -335       |
| 690         | 355         | -340       |
| 700         | 360         | -345       |
| 710         | 365         | -350       |
| 720         | 370         | -355       |
| 730         | 375         | -360       |
| 740         | 380         | -365       |
| 750         | 385         | -370       |
| 760         | 390         | -375       |
| 770         | 395         | -380       |
| 780         | 400         | -385       |
| 790         | 405         | -390       |
| 800         | 410         | -395       |
| 810         | 415         | -400       |
| 820         | 420         | -405       |
| 830         | 425         | -410       |
| 840         | 430         | -415       |
| 850         | 435         | -420       |
| 860         | 440         | -425       |
| 870         | 445         | -430       |
| 880         | 450         | -435       |
| 890         | 455         | -440       |
| 900         | 460         | -445       |
| 910         | 465         | -450       |
| 920         | 470         | -455       |
| 930         | 475         | -460       |
| 940         | 480         | -465       |
| 950         | 485         | -470       |
| 960         | 490         | -475       |
| 970         | 495         | -480       |
| 980         | 500         | -485       |
| 990         | 505         | -490       |
| 1000        | 510         | -495       |
| ... (additional data points) are not provided in the code. The actual data points may vary due to the use of the delta vectors in the chart. The numbers in the table represent the delta vectors in each category. There is only one data series in this case. The values in the table represent the delta vectors in each category. There is no additional data series in this case.
</details>

![](images/5153aab47c963bded80d196b69ca29f58c483e9ba1223573ba42af005b399485.jpg)

<details>
<summary>scatter</summary>

| PC1       | PC2       | PC3       | Category |
| --------- | --------- | --------- | -------- |
| -10       | 5         | 10        | left     |
| 0         | 10        | 15        | right    |
| 10        | 5         | 10        | above    |
| -5        | 0         | 5         | below    |
| 5         | -5        | 0         | far      |
| -15       | -10       | -5        | close    |
| 20        | -15       | -10       | far      |
| -20       | 0         | -15       | above    |
| 25        | 5         | -10       | below    |
| -25       | 10        | -5        | far      |
| 30        | 15        | 0         | above    |
| -30       | 10        | 5         | below    |
| 35        | 15        | 10        | far      |
| -35       | 15        | 15        | above    |
| 40        | 20        | 20        | below    |
| -40       | 20        | 25        | far      |
| 45        | 25        | 20        | above    |
| -45       | 25        | 25        | below    |
| 50        | 30        | 30        | far      |
| -50       | 30        | 25        | above    |
| 55        | 35        | 35        | below    |
| -55       | 35        | 30        | far      |
| 60        | 40        | 40        | above    |
| -60       | 40        | 35        | below    |
| 65        | 45        | 45        | far      |
| -65       | 45        | 40        | above    |
| 70        | 50        | 50        | below    |
| -70       | 50        | 45        | far      |
| 75        | 55        | 50        | above    |
| -75       | 55        | 45        | below    |
| 80        | 60        | 55        | far      |
| -80       | 60        | 60        | above    |
| 85        | 65        | 60        | below    |
| -85       | 65        | 65        | far      |
| 90        | 70        | 65        | above    |
| -90       | 70        | 70        | below    |
| 95        | 75        | 70        | far      |
| -95       | 75        | 75        | above    |
| 100       | 80        | 75        | below    |
| -100      | 80        | 80        | far      |
| 105       | 85        | 75        | above    |
| -105      | 85        | 80        | below    |
| 110       | 90        | 80        | far      |
| -110      | 90        | 85        | above    |
| 115       | 95        | 85        | below    |
| -115      | 95        | 90        | far      |
| 120       | 100       | 90        | above    |
| -120      | 100       | 95        | below    |
| 125       | 105       | 95        | far      |
| -125      | 105       | 100       | above    |
| 130       | 110       | 100       | below    |
| -130      | 110       | 105       | far      |
| 135       | 115       | 105       | above    |
| -135      | 115       | 110       | below    |
| 140       | 120       | 110       | far      |
| -140      | 120       | 115       | above    |
| 145       | 125       | 115       | below    |
| -145      | 125       | 120       | far      |
| 150       | 130       | 120       | above    |
| -150      | 130       | 125       | below    |
| 155       | 135       | 125       | far      |
| -155      | 135       | 130       | above    |
| 160       | 140       | 130       | below    |
| -160      | 140       | 135       | far      |
| 165       | 145       | 135       | above    |
| -165      | 145       | 140       | below    |
| 170       | 150       | 140       | far      |
| -170      | 150       | 145       | above    |
| 175       | 155       | 145       | below    |
| -175      | 155       | 150       | far      |
| 180       | 160       | 150       | above    |
| -180      | 160       | 155       | below    |
| 185       | 165       | 155       | far      |
| -185      | 165       | 160       | above    |
| 190       | 170       | 160       | below    |
| -190      | 170       | 165       | far      |
| 195       | 175       | 165       | above    |
| -195      | 175       | 170       | below    |
| 200       | 180       | 170       | far      |
| -200      | 180       | 175       | above    |
| ...       ... <--> (various) between PCX and PCY
</details>

Fig. 17: 3D PCA of delta vectors for the Molmo family. A distinct distance axis does not clearly emerge, although delta vectors in the horizontal and vertical axes appear more well-clustered with data scaling.

![](images/0e44e037998b02f805c98ca316d602ddad9e0c44c49a95c778731f68a63daa72.jpg)  
Fig. 18: 3D PCA of delta vectors for the NVILA family. RoboRefer’s distance clusters (far/close) occupy a distinct subspace from vertical categories, unlike the finetuned variants.

QWEN(vanilla)-L28   
Delta Vectors by Category   
![](images/612c4d3034d7cb950c46e3693fe97a4b47c62613e18e77ab8fbd2cb66d5fc61d.jpg)

<details>
<summary>scatter</summary>

| PC1       | PC2       | PC3       |
| --------- | --------- | --------- |
| -60       | -40       | 40        |
| -40       | -20       | 20        |
| -20       | 0         | 0         |
| 0         | 20        | -20       |
| 20        | 40        | -40       |
| 40        | 60        | -60       |
| 60        | 80        | -80       |
| 80        | 100       | -100      |
| 100       | 120       | -120      |
| 120       | 140       | -140      |
| 140       | 160       | -160      |
| 160       | 180       | -180      |
| 180       | 200       | -200      |
| 200       | 220       | -220      |
| 220       | 240       | -240      |
| 240       | 260       | -260      |
| 260       | 280       | -280      |
| 280       | 300       | -300      |
| 300       | 320       | -320      |
| 320       | 340       | -340      |
| 340       | 360       | -360      |
| 360       | 380       | -380      |
| 380       | 400       | -400      |
| 400       | 420       | -420      |
| 420       | 440       | -440      |
| 440       | 460       | -460      |
| 460       | 480       | -480      |
| 480       | 500       | -500      |
| 500       | 520       | -520      |
| 520       | 540       | -540      |
| 540       | 560       | -560      |
| 560       | 580       | -580      |
| 580       | 600       | -600      |
| 600       | 620       | -620      |
| 620       | 640       | -640      |
| 640       | 660       | -660      |
| 660       | 680       | -680      |
| 680       | 700       | -700      |
| 700       | 720       | -720      |
| 720       | 740       | -740      |
| 740       | 760       | -760      |
| 760       | 780       | -780      |
| 780       | 800       | -800      |
| 800       | 820       | -820      |
| 820       | 840       | -840      |
| 840       | 860       | -860      |
| 860       | 880       | -880      |
| 880       | 900       | -900      |
| 900       | 920       | -920      |
| 920       | 940       | -940      |
| 940       | 960       | -960      |
| 960       | 980       | -980      |
| 980       | 1000      | -1000     |
| 100        | 125        | -125      |
| 125       | 15        | -15       |
| 15        | -5        | -5        |
| 175       | -15       | -15       |
| 2     | -35       | -35       |
| -5        | -5        | -5        |
| -75       | -75       | -75       |
| -1     | -1     | -1        |
| -7        | -7        | -7        |
| -17.5     | -17.5     | -17.5     |
| -2     | -3     | -3        |
| -17.5     | -17.5     | -17.5     |
| -2     | -5        | -5        |
| -17.5     | -17.5     | -17.5     |
| -2     | -7.5      | -7.5      |
| -17.5     | -3     | -7        |
| -2     | -1     | -7        |
| -17.5     | -7.5      | -17.5     |
| -2     | -1.75     | -17.5     |
| -17.5     | -7.5      | -17.5     |
| -2     | -3.75     | -17.5     |
| -17.5     | -1     | -3.75     |
| -2     | -7.5      | -1     |
| -17.5     | -3.75     | -7.5      |
| -2     | -1     | -3.75     |
| -17.5     | -7.5      | -1     |
| -2     | -3.75     | -7.5      |
| -1    | -1.75     | -1.75     |
| -1.75     | -7.5      | -7.5      |
| -2     | -3.75     | -3.75     |
| -1.75     | -1.75     | -1.75     |
| -2     | -7.5      | -7.5      |
| -1    | -3.75     | -3.75     |
| -1.75     | -1.75     | -1.75     |
| -2     | -7.5      | -7.5      |
| -1.75     | -3.75     | -3.75     |
| -2     | -1.75     | -1.75     |
| -1    | -7.5      | -1.75     |
| -1.75     | -3.75     | -7.5      |
| -2     | -1.75     | -3.75     |
| -1    | -7.5      | -1.75     |
| -1.75     | -3.75     | -7.5      |
| -2     | -1.75     | -3.75     |
| -1    | -7.5      | -1.75     |
| -1.75     | -3.75     | -7.3      |
| -2     | -1.75     | -3.75     |
| -1    | -7.5      | -1.75     |
| -1.75     | -3.75     | -7.3      |
| -2     | -1.75     | -3.75     |
| -1    | -7.5      | -1.75     |
| -    |   nan        |           |
|           \text{PC3}   = (-6)   \text{PC2}   = (4)   \text{PC3}   = (9)   \text{PC2}   = (3)   \text{PC3}   = (9)   \text{PC2}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC2}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (9)   \text{PC3}   = (\text{PC3} / \text{PC3})    \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \text{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{PC3} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} ,\mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm{Pc} / \mathrm {Pc} / \mathrm {Pc} / \mathrm {Pc} / \mathrm {Pc} / \mathrm {Pc} / \mathrm {Pc} / \mathrm {Pc} / \mathrm {Pc} / \mathrm {Pc}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\shortcirc}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf{\scriptsize{*}}\textbf {\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\scriptsize*}\textbf{\script size}
</details>

QWEN (80k)-L28   
Delta Vectors by Category   
![](images/a8bb0414d66308f4eb6965b7e11a9f24d9fbbe4854ad68ab1f922d8e4c5102b2.jpg)

<details>
<summary>scatter</summary>

| PC1 (25.3%) | PC2 (12.8%) | PC3 (6.7%) |
| ----------- | ----------- | ---------- |
| -60         | -40         | 40         |
| -40         | -20         | 20         |
| -20         | 0           | 0          |
| 0           | 20          | -20        |
| 20          | 40          | -40        |
| 40          | 60          | -60        |
| 60          | 80          | -80        |
| 80          | 100         | -100       |
| 100         | 120         | -120       |
| 120         | 140         | -140       |
| 140         | 160         | -160       |
| 160         | 180         | -180       |
| 180         | 200         | -200       |
| 200         | 220         | -220       |
| 220         | 240         | -240       |
| 240         | 260         | -260       |
| 260         | 280         | -280       |
| 280         | 300         | -300       |
| 300         | 320         | -320       |
| 320         | 340         | -340       |
| 340         | 360         | -360       |
| 360         | 380         | -380       |
| 380         | 400         | -400       |
| 400         | 420         | -420       |
| 420         | 440         | -440       |
| 440         | 460         | -460       |
| 460         | 480         | -480       |
| 480         | 500         | -500       |
| 500         | 520         | -520       |
| 520         | 540         | -540       |
| 540         | 560         | -560       |
| 560         | 580         | -580       |
| 580         | 600         | -600       |
| 600         | 620         | -620       |
| 620         | 640         | -640       |
| 640         | 660         | -660       |
| 660         | 680         | -680       |
| 680         | 700         | -700       |
| 700         | 720         | -720       |
| 720         | 740         | -740       |
| 740         | 760         | -760       |
| 760         | 780         | -780       |
| 780         | 800         | -800       |
| 800         | 820         | -820       |
| 820         | 840         | -840       |
| 840         | 860         | -860       |
| 860         | 880         | -880       |
| 880         | 900         | -900       |
| 900         | 920         | -920       |
| 920         | 940         | -940       |
| 940         | 960         | -960       |
| 960         | 980         | -980       |
| 980         | 1000        | -1000      |
| 1000        | 1020        | -1020      |
| 115        | 115        | -115       |
| 135        | 135        | -135       |
| 155        | 155        | -155       |
| 175        | 175        | -175       |
| 195        | 195        | -195       |
| 215        | 215        | -215       |
| 235        | 235        | -235       |
| 255        | 255        | -255       |
| 275        | 275        | -275       |
| 295        | 295        | -295       |
| 315        | 315        | -315       |
| 335        | 335        | -335       |
| 355        | 355        | -355       |
| 375        | 375        | -375       |
| 395        | 395        | -395       |
| 415        | 415        | -415       |
| 435        | 435        | -435       |
| 455        | 455        | -455       |
| 475        | 475        | -475       |
| 495        | 495        | -495       |
| 515        | 515        | -515       |
| 535        | 535        | -535       |
| 555        | 555        | -555       |
| 575        | 575        | -575       |
| 595        | 595        | -595       |
| 615        | 615        | -615       |
| 635        | 635        | -635       |
| 655        | 655        | -655       |
| 675        | 675        | -675       |
| 695        | 695        | -695       |
| 715        | 715        | -715       |
| 735        | 735        | -735       |
| 755        | 755        | -755       |
| 775        | 775        | -775       |
| 795        | 795        | -795       |
| 815        | 815        | -815       |
| 835        | 835        | -835       |
| 855        | 855        | -855       |
| 875        | 875        | -875       |
| 895        | 895        | -895       |
| 915        | 915        | -915       |
| 935        | 935        | -935       |
| 955        | 955        | -955       |
| 975        | 975        | -975       |
| 995        | 995        | -995       |
| Note: The values in the CSV data are not explicitly provided in the code. The code does not include the original data points from the 'left' to 'right' series. The values in the 'close' series are not explicitly provided in the code but are inferred from the 'below' series. There is only one data series represented by the 'left' series. The other five series are not explicitly labeled in the code but are inferred from the 'above' series. The values in the 'above' series are estimated based on the 'close' series. The values in the 'below' series are estimated based on the 'far' series. The values in the 'close' series are estimated based on the 'near' series. The values in the 'above' series are estimated based on the 'close' series. The values in the 'below' series are estimated based on the 'near' series. The values in the 'close' series are estimated based on the 'close' series. The values in the 'above' series are estimated based on the 'near' series. The values in the 'below' series are estimated based on the 'close' series. The values in the 'close' series are estimated based on the 'near' series. The values in the 'above' series are estimated based on the 'close' series. The values in the 'below' series are estimated based on the 'near' series. The values in the 'close' series are estimated based on the 'close' series. The values in the 'above' series are estimated based on the 'near' series. The values in the 'below" series are estimated based on the 'close' series. The values in the 'close' series are estimated based on the 'near' series. The values in the 'above' series are estimated based on the 'close' series. The values in the 'below" series are estimated based on the 'near' series. The values in the 'close' series are estimated based on the 'close' series. The values in the 'above' series are estimated based on the 'near' series. The values in the 'below" series are estimated based on the 'close' series. The values in the 'close' series are estimated based on the 'near' series. The values in the 'above' series are estimated based on the 'close' period from the code.
</details>

QWEN (400k)－L28   
Delta Vectors by Category   
![](images/c2ca2b41f08dd91cc4ca788f23b433e727ca3d39b1b809b0857a5a51f7aa7c18.jpg)

QWEN (800k)-L28   
Delta Vectors by Category   
![](images/387df2e97767d606d8cca0c8e916cbdd20c90996e59c4fbba32bf054709db488.jpg)

<details>
<summary>scatter</summary>

| PC1 (29.5%) | PC2 (19.2%) | PC3 (6.3%) |
| ----------- | ----------- | ---------- |
| -40         | 0           | 40         |
| -20         | 20          | 20         |
| 0           | 40          | 0          |
| 20          | 20          | -20        |
| 40          | 0           | -40        |
</details>

QWEN (2m)- L28   
Delta Vectors by Category   
![](images/d63c70bd9157cfb2fd5a716bc70db007ba1e0ddbf5d4e2f5966124965ec045da.jpg)

<details>
<summary>scatter</summary>

| PC1 (31.4%) | PC2 (21.5%) | PC3 (6.2%) |
| ----------- | ----------- | ---------- |
| -60         | -40         | 40         |
| -40         | -20         | 20         |
| -20         | 0           | 0          |
| 0           | 20          | -20        |
| 20          | 40          | -40        |
| 40          | 60          | -60        |
| 60          | 80          | -80        |
</details>

QWEN (qwen3\_235b)- L87   
Delta Vectors by Category   
![](images/216523ad6a8bde11d822bb8bbe83b1bf8f02b3256486c993e00749b0a5433921.jpg)

<details>
<summary>scatter</summary>

| PC1       | PC2       | PC3       |
| --------- | --------- | --------- |
| -60       | -40       | 40        |
| -40       | -20       | 20        |
| -20       | 0         | 0         |
| 0         | 20        | -20       |
| 20        | 40        | -40       |
| 40        | 60        | -60       |
| 60        | 80        | -80       |
| 80        | 100       | -100      |
| 100       | 120       | -120      |
| 120       | 140       | -140      |
| 140       | 160       | -160      |
| 160       | 180       | -180      |
| 180       | 200       | -200      |
| 200       | 220       | -220      |
| 220       | 240       | -240      |
| 240       | 260       | -260      |
| 260       | 280       | -280      |
| 280       | 300       | -300      |
| 300       | 320       | -320      |
| 320       | 340       | -340      |
| 340       | 360       | -360      |
| 360       | 380       | -380      |
| 380       | 400       | -400      |
| 400       | 420       | -420      |
| 420       | 440       | -440      |
| 440       | 460       | -460      |
| 460       | 480       | -480      |
| 480       | 500       | -500      |
| 500       | 520       | -520      |
| 520       | 540       | -540      |
| 540       | 560       | -560      |
| 560       | 580       | -580      |
| 580       | 600       | -600      |
| 600       | 620       | -620      |
| 620       | 640       | -640      |
| 640       | 660       | -660      |
| 660       | 680       | -680      |
| 680       | 700       | -700      |
| 700       | 720       | -720      |
| 720       | 740       | -740      |
| 740       | 760       | -760      |
| 760       | 780       | -780      |
| 780       | 800       | -800      |
| 800       | 820       | -820      |
| 820       | 840       | -840      |
| 840       | 860       | -860      |
| 860       | 880       | -880      |
| 880       | 900       | -900      |
| 900       | 920       | -920      |
| 920       | 940       | -940      |
| 940       | 960       | -960      |
| 960       | 980       | -980      |
| 980       | 1000      | -1000     |
| 1000      | 1020      | -1020     |
| 115        | 115        | -115      |
| 135        | 135        | -135      |
| 155        | 155        | -155      |
| 175        | 175        | -175      |
| 195        | 195        | -195      |
| 215        | 215        | -215      |
| 235        | 235        | -235      |
| 255        | 255        | -255      |
| 275        | 275        | -275      |
| 295        | 295        | -295      |
| 315        | 315        | -315      |
| 335        | 335        | -335      |
| 355        | 355        | -355      |
| 375        | 375        | -375      |
| 395        | 395        | -395      |
| 415        | 415        | -415      |
| 435        | 435        | -435      |
| 455        | 455        | -455      |
| 475        | 475        | -475      |
| 495        | 495        | -495      |
| 515        | 515        | -515      |
| 535        | 535        | -535      |
| 555        | 555        | -555      |
| 575        | 575        | -575      |
| 595        | 595        | -595      |
| 615        | 615        | -615      |
| 635        | 635        | -635      |
| 655        | 655        | -655      |
| 675        | 675        | -675      |
| 695        | 695        | -695      |
| 715        | 715        | -715      |
| 735        | 735        | -735      |
| 755        | 755        | -755      |
| 775        | 775        | -775      |
| 795        | 795        | -795      |
| 815        | 815        | -815      |
| 835        | 835        | -835      |
| 855        | 855        | -855      |
| 875        | 875        | -875      |
| 895        | 895        | -895      |
| 915        | 915        | -915      |
| 935        | 935        | -935      |
| 955        | 955        | -955      |
| 975        | 975        | -975      |
| 995        | 995        | -995      |
| Note: The 'PC1' values are not explicitly provided in the code. The 'PC2' values are not explicitly provided in the code. There is only one data series in the code.
</details>

Fig. 19: 3D PCA of delta vectors for the Qwen family. Variants include Qwen2.5-VL-3B-Instruct and Qwen3-VL-235B-A22B-Instruct. Qwen3-VL-235B shows clear three-way separation among horizontal, vertical, and distance axes in 3D space.