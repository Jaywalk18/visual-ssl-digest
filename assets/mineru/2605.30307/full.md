# Grounded 3D-Aware Spatial Vision-Language Modeling

An-Chieh Cheng1,3\* Yang Fu1 Yatai Ji3 Ligeng Zhu3 Guanqi Zhan3 Zhuoyang Zhang2,3 Zhaojing Yang1 Song Han2,3 Yao Lu3 Pavlo Molchanov3 Vidya Nariyambut Murali3 Jan Kautz3 Xiaolong Wang1 Hongxu Yin3 Sifei Liu3

1UCSD 2MIT 3NVIDIA

https://www.anjiecheng.me/gr3d

Visual Grounded Spatial CoT   
![](images/e4c71d250832afcf1d7d5a868291cf99caf727fdead6dfb62f8926b774186ed1.jpg)

<details>
<summary>text_image</summary>

light beer bottle
dark beer bottle
</details>

What is the distance between the leftmost bottle and the rightmost bottle?   
The left bottle is at [...] and the right bottle is at [...] . The distance is 1.05 meters.

Region-Prompted 3D Detection   
![](images/791ca6b6b2d9c54fb397fda5f73d73c45524eae64330e45ab977e715c20e8edb.jpg)

<details>
<summary>text_image</summary>

Street photo with visible store signboards and a magnified inset showing a shopping cart with a pink icon.
</details>

Detect the 3D bounding box of   
<3d\_box center="-0.27, 1.31, 31.11" dimensions="2.05, 1.46, ...</3d\_box>

3D Detection with Visual Grounded CoT   
![](images/af9f886ccf0970555914e23fb1719236cbff0a6bc3213d5ebd7d8771b517a5e2.jpg)

<details>
<summary>text_image</summary>

bed
pillow on pillow
toon
toon max
windows
windows
blinds
blinds
top
top
toon
toon max
growth
in
后
后
</details>

Detect the 3D bounding boxes of all objects.   
[...] , [...] , [...] ,... <3d\_box center="0.68, 0.05, 2.15" dimensions="0.12...

Region-Prompted Spatial VQA   
![](images/4cb94b25ff90e732eda3b86f8067f7aa8c53cc48f6737365a7578346f8a7d1ac.jpg)

<details>
<summary>natural_image</summary>

City skyline with modern skyscrapers under a cloudy sky (no visible text or signage)
</details>

Which building is taller, is it

![](images/f603ff85c6a7a793cedd55dca5bb2a779120210607fa6c81122b9916a86ab0b9.jpg)

![](images/c26763a0bfd0bd76212f7e32c81830ee70dd3fbd46f8d397c7f1cdb49518c13f.jpg)  
Region [0] is with more height.

3D Grounding with Visual Grounded CoT   
![](images/102e71d4f85fde1b07bac218cb82d3123bcf9ec1b99d3f0f7e76a8ac942431df.jpg)

<details>
<summary>text_image</summary>

the bottle the
the bottle that could be leaking flu
</details>

Detect the 3D bounding box of the bottle that could be leaking fluid.   
The bottle is at [x1 ,y1 ,x2 ,y2 ] <3d\_box center="0.31, 0.21, 0.83"...

Region-Prompted 3D Point Prediction   
![](images/cd07a4b8ec2fc1b0408033ab14d104c3206525b6ffc5dd8455bb77aee5055408.jpg)

<details>
<summary>natural_image</summary>

Kitchen countertop with stainless steel countertop, kitchenware, and a microwave (no visible text or symbols)
</details>

Please detect the 3D location of point   
x: -0.34, y: -0.17, z: 0.83

Figure 1. Overview. GR3D bridges 2D pixel-space and 3D metric-space by integrating multiple grounding capabilities into visual chainof-thoughts, enabling complex spatial understanding through grounded 2D perception followed by 3D inference.

# Abstract

We present GR3D, a spatial vision language model equipped with three complementary grounding capabilities—explicit 2D grounding, implicit 2D grounding, and monocular 3D grounding—within a single framework. GR3D introduces an implicit grounding mechanism that identifies entity mentions during generation and inserts the corresponding region tokens into the text stream, allowing the model to reference visual evidence on the fly when producing spatial chain-of-thought responses. In parallel, a

region-prompted monocular 3D grounding design predicts 3D bounding boxes in the camera view from grounded region queries, supported by intrinsic-aware normalization and dense geometric supervision. Together, these grounding capabilities enable GR3D to decompose complex spatial understanding problems into grounded 2D perception followed by 3D inference. GR3D achieves consistent improvements across grounded and non-grounded spatial benchmarks, demonstrating grounding as an effective inductive bias for strengthening spatial understanding in VLMs. These grounding capabilities collectively enhance general spatial understanding beyond the grounding task itself.

# 1. Introduction

Vision–language models (VLMs) have rapidly evolved into general-purpose perception–language systems [1–8], capable of understanding scenes, following open-ended instructions, and supporting diverse multimodal tasks. As these models begin to serve as the core of embodied agents that must act, manipulate, and navigate in the physical world [9–17], their spatial competence becomes crucial. Embodied intelligence requires models not only to recognize what is present, but also to understand where objects are and how they are arranged in space—capabilities essential for grounding language into actions such as where to reach, step, or orient [18–20]. Without reliable spatial grounding, the link between high-level instructions and physical interaction remains brittle, limiting the scalability of VLMs toward real-world embodied perception and control.

Rapid progress in spatial VLMs has substantially advanced 2D spatial understanding and even 3D perception [21–29]. Yet grounding—the ability to reliably associate linguistic mentions with concrete visual regions and connect 2D evidence with 3D structure—remains limited. Two challenges, in particular, are under-addressed. (i) Implicit 2D grounding is scarce: most systems support explicit “point to X” grounding but lack mechanisms or data for automatically detecting entities mentioned in free-form text and integrating their corresponding visual evidence during generation. Constructing such supervision is difficult, as it requires aligning textual mentions to latent visual regions and interleaving region information into the language stream. (ii) Monocular 3D grounding is inherently illposed: from a single view, object scale, depth, and intrinsics are entangled, and 3D prediction requires first identifying which instance the text refers to before estimating its 3D extent and pose. Existing approaches often bypass this intermediate localization step [30], rely on multi-view supervision [31], or are limited by the scarcity of 3D box annotations [32].

To address these limitations, we introduce (GR3D), a spatial VLM that integrates grounding as a core mechanism for learning spatial representations. GR3D jointly supports three complementary grounding capabilities within a unified architecture: explicit 2D grounding, which predicts object regions through the language head in a structured textual format; implicit 2D grounding, which links linguistic mentions to visual evidence through dynamic region insertion; and monocular 3D grounding, which extends region understanding into 3D by predicting bounding boxes and camera intrinsics under dense geometric supervision. Together, these mechanisms establish a fine-grained alignment between language, image regions, and geometry, enabling consistent 2D and 3D spatial reasoning.

While explicit 2D grounding predicts the location of queried objects, it cannot handle free-form reasoning where spatial cues are implicit. Real-world spatial queries—e.g., describing relations, distances, or navigation targets—require first recognizing and localizing the entities mentioned before reasoning about the query itself. GR3D bridges this gap with an implicit 2D grounding mechanism that performs streaming region insertion: as the model generates responses, it dynamically predicts the visual region corresponding to each mentioned entity, encodes the region into a token, and injects it directly into the ongoing language stream. This enables reasoning to evolve directly over grounded visual evidence, yielding coherent spatial predictions without any separate detection phase.

Inferring 3D structure from a single view introduces both linguistic and geometric ambiguities, such as determining which instance a description refers to and estimating its depth, scale, and pose without multi-view cues. GR3D addresses these challenges through a regionprompted 3D grounding formulation: each grounded 2D region is treated as a query for 3D inference, supported by intrinsic-aware normalization and dense geometric supervision derived from depth estimation. This design aligns semantic localization and geometric prediction within a consistent camera-view framework, enabling the model to infer coherent 3D structure directly from grounded 2D evidence and to generalize across diverse scenes and viewpoints. Crucially, by receiving region tokens produced by implicit 2D grounding, the 3D predictor naturally plugs into CoT-driven reasoning—allowing the model to first resolve “which object” via grounded language generation and then infer “what 3D structure” for that object. This decomposition makes monocular 3D grounding applicable to both instance-level referring tasks and open-set category-level 3D detection.

Integrating explicit 2D grounding, implicit 2D grounding, and monocular 3D grounding positions GR3D as a flexible spatial understanding framework spanning 2D/3D and single-/multi-view settings. Through this groundingcentered formulation, the model learns to localize, reference, and reason over spatial structure in a unified manner. Implicit grounding enhances CoT accuracy and spatial consistency on CVBench [33], ERQA [30] and SAT [34], while region-prompted 3D grounding with dense point supervision achieves state-of-the-art performance on Omni3D. Moreover, we observe key insights: (i) grounding improves general spatial understanding even without explicit localization; (ii) dense geometric supervision provides scalable structure cues; (iii) combining implicit grounding with region-prompted 3D inference unlocks a versatile decomposition pipeline that supports referring-instance 3D grounding, category-level 3D detection, and multi-object scene grounding. Together, these results show that embedding grounding within the model architecture strengthens both spatial perception and grounded reasoning.

![](images/6922caac368b3129a78b779ae3a7dd01199cba1d1ce70193bf0008eb43cb99fe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Large Language Model"] --> B["Vision Encoder"]
    A --> C["Spatial PE"]
    B --> D["MLP Projector"]
    C --> E["MLP Region Extractor"]
    D --> F["Tokenizer"]
    E --> G["Detect the 3D BBox of the orange chair."]
    F --> H["2D BBox / Point [x1, y1, x2, y2"]]
    G --> I["3D BBox <x, y, z, h, w, l, p, r, y>"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#ffc,stroke:#333
    style I fill:#ffc,stroke:#333
```
</details>

Figure 2. Method overview. GR3D builds on Region-VLMs by adding streaming region insertion for visual Chain-of-Thought reasoning. During CoT, the model repeatedly predicts a region, extracts its visual embedding, and reinserts a region token into the text sequence, enabling step-by-step spatial reasoning with dynamically refreshed visual cues.

# 2. Method

GR3D is designed to address two major limitations of current Spatial VLMs: the lack of an implicit grounding mechanism that allows models to automatically associate linguistic mentions with visual evidence during reasoning, and the difficulty of performing monocular 3D grounding from a single image with entangled depth and scale cues. To overcome these, we first construct a foundational Spatial VLM (Sec. 2.1) that provides geometry-aware features for both single- and multi-view inputs. Building on this foundation, we introduce explicit and implicit 2D grounding (Sec. 2.2) to link linguistic expressions with visual evidence, and extend them to monocular 3D grounding through region prompts, intrinsic normalization, and dense geometric supervision (Sec. 2.3). Finally, we describe our data construction pipeline (Sec. 2.4) that generates large-scale implicit grounding annotations and balanced 2D–3D supervision to facilitate training.

# 2.1. Foundational Spatial VLM

Objective. We follow the design principle of SR-3D [22] to construct a foundational Spatial VLM based on the NVILA-8B-Lite [3] architecture. This model provides a unified spatial representation that supports both single- and multi-view spatial understanding, serving as the base for subsequent grounding modules. At this stage, no grounding capability is included; the focus is on building a geometry-aware representation layer compatible with language reasoning.

Single-view Setup. The base NVILA encoder extracts dense visual tokens from an RGB image for single-view inputs. To make these tokens spatially aware, we augment them with 2D positional embeddings derived from their pixel coordinates and relative depth cues. Each visual token therefore carries both appearance and geometric context. Unlike language tokens, which are processed sequentially, these enriched visual tokens retain their spatial arrangement within the image grid. They are passed through the multimodal projector before being fed to the language model. This projection preserves spatial locality while remaining compatible with NVILA’s multimodal fusion pipeline.

In addition, we preserve the region-prompt design used in SR-3D: specific image regions can be encoded as individual query tokens by pooling features within a given bounding box. This structure allows downstream modules to reference localized spatial content directly, maintaining full alignment with the NVILA token structure and positional hierarchy. Overall, the single-view formulation provides a strong spatially-structured feature space for both regionlevel interaction and text-aligned representation.

Multi-view Setup. Our framework naturally extends from single-view to multi-view inputs by embedding all image tokens with depth- and pixel-based positional cues in a unified spatial feature space. The first view is processed exactly as in the single-view case, and all subsequent views are transformed into the first-frame coordinate system. Please see the supplementary materials for our multi-view results.

# 2.2. Grounding in the 2D Plane

Grounding on the 2D plane aims to teach the model to associate linguistic mentions with localized image evidence. We introduce both explicit and implicit forms of grounding, designed to strengthen the spatial reasoning capacity of the vision–language model.

# 2.2.1. Explicit 2D Grounding

For explicit 2D grounding, we adopt a simple and general formulation. Given a natural-language instruction, the model predicts 2D bounding boxes directly in HTML-style textual format (e.g., <bbox>[x1, y1, x2, y2]</bbox>), using its standard language generation head without any additional detection branch. This unified design integrates grounding seamlessly into the vision– language interface, without introducing task-specific architectural components.

# 2.2.2. Implicit 2D Grounding

Consider a global spatial reasoning query such as: $^ { * } I n$ the kitchen, how far is the second bottle on the shelf from the small brown teddy bear on top of the washing machine in the laundry room?” Traditional spatial VLMs attempt to answer such questions directly from global image features, relying on large-scale question–answer pairs to memorize spatial relationships. However, this departs from how humans perceive scenes: we first identify where each mentioned object is before reasoning about their relations. Our implicit grounding mechanism explicitly introduces this intermediate step of entity localization during generation, aligning the model’s behavior with human visual reasoning.

Streaming Region Insertion. Given an input instruction, the model generates its response in a chain-ofthought (CoT) fashion. When an entity (e.g., “the second bottle on the shelf”) is mentioned, the model first predicts its corresponding 2D bounding box coordinates, e.g., $[ x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ]$ . Immediately after, the corresponding image region is encoded through the region encoder, and its embedding—a region token—is inserted directly into the text stream at that position. The generation then continues conditioned on both the textual and visual context. The same procedure repeats for subsequent entities, producing a temporally aligned reasoning trajectory that alternates between language and vision.

Training and Inference Paradigm. During training, the bounding box coordinates are directly predicted by the language head and optimized through teacher forcing, as they are treated as part of the textual output sequence. Once the coordinates are produced, the corresponding region token— derived from the ground-truth region—is inserted into the generation stream. This token is detached from the computation graph (i.e., no gradient flows through it) but serves as a strong conditional cue for subsequent token prediction. During inference, the process becomes fully autoregressive. The model first predicts coordinates, then encodes the predicted region to obtain its embedding, which is inserted back into the ongoing sequence before the next generation step. The subsequent reasoning, such as relational comparison or distance estimation, is thus conditioned on both the textual context and dynamically inserted region evidence.

Comparison and Interpretation. Our stream-based grounding can be viewed abstractly as analogous to a twostep process, i.e., first grounding entities with a VLM, and then performing region-conditioned reasoning with a spatial VLM with a region encoder equipped. Unlike this staged formulation, our approach unifies both phases in a single generative stream. The model learns when and what to ground based on linguistic context, and its reasoning naturally unfolds on grounded evidence without explicit stage transitions. This results in a fluid, interpretable reasoning process that tightly couples perception and cognition while avoiding the discontinuities of discrete grounding modules.

# 2.3. Monocular 3D Grounding via Region Prompt

Monocular 3D grounding aims to enable single-view models to infer 3D structure from natural language and visual cues. This task faces two major challenges. First, linguistic ambiguity: textual references often under-specify which instance is being mentioned, requiring the model to implicitly identify the target entity before 3D reasoning. Second, geometric ambiguity: the coupling between object scale, depth, and camera intrinsics makes single-view estimation inherently uncertain. We address these through several components below that align semantic localization and geometric inference within a unified generative framework.

Region-prompt Formulation. Given a localized 2D region, the model treats this region as a spatial query for 3D reasoning. The region’s visual features are pooled and encoded into a region token, which is fused into the text stream to guide 3D box prediction. Since the model already possesses implicit 2D grounding capability, this step focuses solely on extending that capacity from 2D to 3D—mapping a grounded region to its corresponding 3D representation. This formulation simplifies 3D grounding by conditioning inference on a given region, enabling the model to estimate position, scale, and orientation directly without performing explicit multi-step localization.

3D Box Representation. Each 3D bounding box is expressed in a unified, language-based format compatible with 2D HTML-style outputs, eliminating the need for task-specific heads. The box is parameterized by its center $( x _ { c } , y _ { c } , z _ { c } )$ , size $( w , h , l )$ , and orientation $( \theta _ { p } , \theta _ { r } , \theta _ { y } )$ , where $( \theta _ { p } , \theta _ { r } , \theta _ { y } )$ are normalized Euler angles (pitch/roll/yaw). To ensure consistency across datasets, we standardize orientations by selecting the rotation variant that minimizes the angular deviation between the local PCA axes of the region and the global coordinate axes (X, Y, Z)—that is, the variant closest to the identity basis rather than a mirrored alternative. This compact decomposition makes the representation transferable: the center term aligns naturally with depth-based supervision (see below), while the dimension and rotation terms capture view-invariant geometry. The format promotes stability, interpretability, and seamless integration into the generative language interface.

Intrinsic Normalization. To mitigate scale and depth ambiguity, we introduce an intrinsic-aware normalization strategy that rescales images according to focal length, yielding a consistent field of view across datasets. Concretely, given focal length $f _ { x }$ , we normalize the spatial scale by $\begin{array} { r } { \breve { W } ^ { \prime } = \frac { 1 0 0 0 } { f _ { x } } } \end{array}$ · W and $\begin{array} { r } { H ^ { \prime } = \frac { 1 0 0 0 } { f _ { x } } } \end{array}$ 1000 · H, aligning the apparent object size in the feature space and supporting robust 3D inference without explicitly regressing intrinsics.

Points and Direct Grounding Supervision. We supervise monocular 3D grounding with complementary signals beyond sparse 3D-box labels. (i) Region→3D: when a 2D box is available, the model predicts its 3D box directly from the region prompt. (ii) Pure text→3D: when no 2D box exists, the model localizes the mentioned entity via its built-in textual grounding and regresses its 3D box, enabling coverage of text-only data. In addition, we construct an auxiliary dense region-to-3D supervision: from ground-truth or predicted depth maps, we randomly sample valid surface points per image (e.g., 100 per image) and train the model to predict their 3D coordinates conditioned on the corresponding region prompt. This depth-driven signal scales supervision well beyond limited 3D-box annotations. Finally, to tolerate modest grounding noise, we apply lightweight 2D bounding-box augmentation (jitters in size and location), improving robustness while preserving semantic locality.

<table><tr><td rowspan="2">Method</td><td colspan="2">SUN-RGBD [35]</td><td colspan="2">ARKITSCENES [36]</td><td colspan="2">OBJECTRON [37]</td><td colspan="2">HYPERSIM [38]</td><td colspan="2">KITTI [39]</td><td colspan="2">NUSCENES [40]</td><td rowspan="2"> $AP_{3D} \uparrow$ </td></tr><tr><td> $AP_{15} \uparrow$ </td><td>mAP ↑</td><td> $AP_{15} \uparrow$ </td><td>mAP ↑</td><td> $AP_{15} \uparrow$ </td><td>mAP ↑</td><td> $AP_{15} \uparrow$ </td><td>mAP ↑</td><td> $AP_{15} \uparrow$ </td><td>mAP ↑</td><td> $AP_{15} \uparrow$ </td><td>mAP ↑</td></tr><tr><td colspan="14">Vision Specialist Models</td></tr><tr><td>ImVoxelNet [41]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>9.4</td></tr><tr><td>SMOKE [42]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>10.4</td></tr><tr><td>Cube R-CNN [32]</td><td>-</td><td>15.33</td><td>-</td><td>41.73</td><td>-</td><td>50.84</td><td>-</td><td>7.48</td><td>-</td><td>32.50</td><td>-</td><td>30.06</td><td>23.26</td></tr><tr><td>OVMono3D [43] $_{w/Cube\ R-CNN}$ </td><td>-</td><td>15.20</td><td>-</td><td>41.60</td><td>-</td><td>58.87</td><td>-</td><td>7.75</td><td>-</td><td>25.45</td><td>-</td><td>24.33</td><td>22.98</td></tr><tr><td>DetAny3D [44] $_{w/Cube\ R-CNN}$ </td><td>26.62</td><td>18.96</td><td>59.55</td><td>46.13</td><td>72.51</td><td>54.42</td><td>11.43</td><td>7.17</td><td>44.28</td><td>31.61</td><td>41.01</td><td>30.97</td><td>24.92</td></tr><tr><td colspan="14">Vision Language Models</td></tr><tr><td>Qwen3-VL-4B [45]</td><td>28.28</td><td>17.60</td><td>63.97</td><td>46.33</td><td>61.60</td><td>43.13</td><td>11.56</td><td>6.44</td><td>17.39</td><td>11.25</td><td>7.48</td><td>4.89</td><td>-</td></tr><tr><td>Qwen3-VL-8B [45]</td><td>28.28</td><td>17.77</td><td>62.32</td><td>45.23</td><td>61.63</td><td>43.59</td><td>11.62</td><td>7.23</td><td>5.23</td><td>3.32</td><td>11.52</td><td>7.56</td><td>-</td></tr><tr><td>GR3D-8B (Ours)</td><td>43.49</td><td>31.64</td><td>67.49</td><td>52.52</td><td>71.68</td><td>54.32</td><td>16.42</td><td>10.87</td><td>22.18</td><td>14.75</td><td>22.98</td><td>16.59</td><td>25.40</td></tr></table>

Table 1. Comparison on the Omni3D [32] benchmark between GR3D, vision specialists, and recent VLMs. We report $\mathrm { A P _ { 1 5 } }$ and mAP for each dataset domain. GR3D outperforms all recent VLMs and vision specialists, especially on the indoor domain. 

<table><tr><td>Method</td><td> $AP_{2D}^{sun}$ </td><td> $AP_{2D}^{park}$ </td><td> $AP_{2D}^{obj}$ </td><td> $AP_{2D}^{hyp}$ </td><td> $AP_{2D}^{kit}$ </td><td> $AP_{2D}^{nus}$ </td></tr><tr><td>Cube R-CNN [32]</td><td>15.07</td><td>40.22</td><td>49.24</td><td>11.05</td><td>36.14</td><td>34.64</td></tr><tr><td>Qwen3-VL-8B [45]</td><td>8.06</td><td>22.44</td><td>30.06</td><td>3.08</td><td>1.54</td><td>2.56</td></tr><tr><td>GR3D-8B (Ours)</td><td>38.86</td><td>46.17</td><td>51.66</td><td>28.53</td><td>20.49</td><td>22.16</td></tr></table>

Table 2. 2D detection results on the Omni3D benchmark. We report the mean Average Precision (mAP) for each dataset domain.

Together, region-prompt grounding, structured 3D box representation, intrinsic normalization, and scalable training signals address both linguistic and geometric ambiguities of monocular 3D grounding. These components jointly provide a camera-relative spatial understanding that generalizes across datasets and supports future extensions to multi-view and embodied reasoning tasks.

# 2.4. Data Construction and Composition

Data Construction for Grounding. To construct the implicit grounding corpus, we start from RefSpatial [23], which includes 2D samples from OpenImages [53], 3D video data from CA-1M [54], and synthetic scenes. RefSpatial contains diverse image–text pairs, but it lacks regionlevel annotations for all the mentioned entities. To obtain them, we use Florence-2 [55] to generate candidate 2D bounding boxes and class labels for each textual mention, producing dense but noisy region annotations.

<table><tr><td>Methods</td><td>Acc. (%)</td></tr><tr><td>Human</td><td>98.3</td></tr><tr><td>GPT-4V-Turbo [7]</td><td>66.9</td></tr><tr><td>GPT-4o [58]</td><td>64.5</td></tr><tr><td>LLaVA-v1.5-7B-xtuner [59]</td><td>50.8</td></tr><tr><td>CogVLM-7B [60]</td><td>50.8</td></tr><tr><td>LLaVA-v1.5-7B [61]</td><td>51.6</td></tr><tr><td>LLaVA-InternLM2-7B [62]</td><td>52.4</td></tr><tr><td>SpatialRGPT-8B* [21]</td><td>87.9</td></tr><tr><td>SR3D-8B* [22]</td><td>90.3</td></tr><tr><td>GR3D-8B (Ours)</td><td>94.4</td></tr></table>

![](images/0a7fac83db5ece07786939a8e596761a2feccb653804a909bf81d389972c1c73.jpg)

<details>
<summary>text_image</summary>

Which point is closer to the camera?
(A) A is closer (B) B is closer
The point labeled A is at [0.76, 0.13,
0.80, 0.20] and the point labeled
B is at [0.44, 0.73, 0.48, 0.79].
(B) B is closer
</details>

Figure 3. Results on the BLINK-Depth benchmark for point-level region spatial understanding. Left: comparison with VLM baselines. Right: visualization of one sample. Our method surpasses prior Region-VLMs (\*), which require manual annotated masks.

We then refine these annotations through a VLM for verification and a rephrasing pipeline. This process (i) verifies one-to-one alignment between textual mentions and detected regions, removing unmatched or ambiguous cases, and (ii) rewrites generic class names into concise, instancelevel descriptions based on image context. The resulting corpus provides high-quality implicit grounding supervision that links textual mentions to corresponding visual evidence with precise instance semantics.

For explicit grounding, we augment samples that contain ground-truth boxes by generating short instance-level referring expressions with a vision–language model and validating their existence in the image. Only verified matches are retained. Together, these procedures yield reliable implicit and explicit grounding data, while depth, point, and 3D-box supervision follow the setup in Sec. 2.3.

Data Composition and Distribution. Our training data is composed of publicly available sources: 97K grounded CoT samples, 780K 3D detection samples from Omni3D [32] and EmbodiedScan [56], and 272K pointmap reconstruction samples from DepthLM [57]. We do not use any proprietary or in-house data and the scale of our 3D detection data is comparable to prior works such as VST [26], ensuring performance gains are not simply due to data size.

<table><tr><td rowspan="3">Method</td><td colspan="9">SPATIAL</td><td colspan="4">GENERAL</td><td></td><td></td></tr><tr><td colspan="3">BLINK [46]</td><td colspan="3">CVBench [33]</td><td>RWQA [47]</td><td>ERQA [30]</td><td>SAT [34]</td><td>EMB [48]</td><td>ChartQA [49]</td><td>MME [50]</td><td>POPE [51]</td><td>AI2D [52]</td><td></td></tr><tr><td>Dep.</td><td>Spa.</td><td>Avg.</td><td>Rel.</td><td>Dep.</td><td>Dis.</td><td>Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NVILA-Lite-8B</td><td>73.38</td><td>79.72</td><td>50.51</td><td>93.38</td><td>92.83</td><td>91.00</td><td>86.31</td><td>65.35</td><td>36.25</td><td>62.60</td><td>68.90</td><td>84.80</td><td>1692</td><td>88.10</td><td>91.01</td></tr><tr><td>GR3D-8B (Stage 1)</td><td>87.90</td><td>83.21</td><td>54.35</td><td>96.92</td><td>98.16</td><td>95.50</td><td>87.23</td><td>68.75</td><td>40.25</td><td>76.00</td><td>81.01</td><td>84.48</td><td>1656</td><td>88.23</td><td>91.81</td></tr><tr><td>GR3D-8B (Stage 2)</td><td>87.90</td><td>80.41</td><td>53.26</td><td>96.46</td><td>98.00</td><td>96.00</td><td>87.26</td><td>65.23</td><td>38.50</td><td>70.60</td><td>77.58</td><td>84.00</td><td>1626</td><td>87.00</td><td>91.54</td></tr></table>

Table 3. Performance comparison on general visual question answering and spatial reasoning benchmarks.

# 3. Experiments

In this section, we begin by describing the implementation details (Sec. 3.1), including the training stages and datasets used. We then present the main results of our model, highlighting its 3D detection performance in Sec. 3.2. Next, we assess whether the model preserves its general VLM and spatial capabilities in Sec. 3.3. In Sec. 3.4, we evaluate the visual grounded CoT enabled by our implicit grounding approach. Finally, Sec. 3.5 provides additional analysis and ablation studies of the model’s 3D detection performance.

# 3.1. Implementation Details

Our model is trained in two stages as detailed in following.

Stage 1: Spatial Pretraining. The goal of this stage is to strengthen the model’s spatial understanding and 2D grounding capabilities, which later improves its 3D detection performance, as shown in our analysis. We initialize the visual encoder, projector, and LLM from NVILA-Lite 8B, while the spatial positional encoding module is newly initialized. Training is performed on a data mixture similar to SR-3D, augmented with 2D grounding data and regionto-3D detection data from Sec. 2.4. During this stage, we freeze the visual encoder and train the remaining modules.

Stage 2: Detection CoT Finetuning. After pretraining, the model already possesses strong 2D grounding and basic 3D detection abilities. We then fine-tune it on CoT-oriented detection data, including detection data in CoT format (curated from Omni3D by first grounding in 2D and then predicting 3D boxes). Since the visual features are already well-formed after Stage 1, we fine-tune only the LLM to learn the reasoning and text-generation structure.

# 3.2. 3D Object Detection

We evaluate our model on the Omni3D test set, following the benchmark protocol and hyperparameters used in DetAny3D. The Omni3D benchmark reports Average Precision (AP), where predictions are matched to ground-truth using 3D IoU with thresholds ranging from 0.05 to 0.50.

For comparison, we include both vision-specialist baselines (e.g., ImVoxelNet [41], Cube R-CNN [32], OV-Mono3D [43], and DetAny3D [44]) and VLM-based baselines (e.g., Qwen3VL-4B [45] and Qwen3VL-8B [45]. Our main results are shown in Table 1 and Fig. 4, where our model outperforms all VLM baselines. Compared with vision specialists, our model achieves competitive results overall and delivers notably better performance on indoor datasets.

We further analyze why existing VLMs perform worse on 3D detection. First, unlike our approach, they do not disentangle 3D detection into a two-step process—2D grounding followed by 3D box prediction. As we show in the analysis (Sec. 3.5), 2D grounding provides a stable geometric anchor that leads to more reliable and consistent 3D predictions. Second, existing VLMs struggle with handling camera intrinsics. Qwen3VL is highly sensitive to input resolution, since pixel dimensions implicitly encode the focal length used in its geometric reasoning. This makes its 3D predictions unstable under changes in image size. VST [26] partially addresses this by normalizing focal length in a manner similar to ours. However, it still requires FoVs to be passed as text prompts. Representing metric geometric parameters in textual form is difficult for the model to parse and integrate reliably, which limits its 3D understanding across scenes and camera setups.

Since our method explicitly separates 2D grounding from 3D prediction, we also evaluate 2D grounding performance on the Omni3D benchmark. As shown in Table 2, our model exceeds region proposals generated by Cube R-CNN and the Qwen3-VL family. For Qwen3-VL models, which do not perform explicit 2D grounding, we evaluate using 2D boxes projected from their predicted 3D outputs.

# 3.3. Visual Question Answering

We investigate two key questions: (1) whether Stage 1 spatial pre-training effectively improves spatial reasoning performance, and (2) whether Stage 2 detection CoT finetuning negatively affects the model’s general VQA capabilities. We evaluate two variants of our model: one after spatial pre-training and one after CoT finetuning. The results are presented in Table 3. After spatial pre-training, the model shows a clear improvement on spatial-related VQA benchmarks, confirming the effectiveness of this stage. In contrast, Stage 2 finetuning focuses on learning the structure of CoT reasoning, and the results indicate that it does not significantly reduce general VQA performance. Most benchmarks remain similar to the Stage 1 model, suggesting that the model maintains strong general-purpose abilities.

# 3.4. Implicit Grounding CoT

We aim to evaluate two aspects of our implicit grounding approach: (1) how accurate the grounding is, and (2)

![](images/5f2449f4c1580c2c9a0b088a94ab20ba22565827fbeee6af245a2c18b957ff40.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing 3D bounding boxes and animal images in various settings (no text or symbols)
</details>

Figure 4. Qualitative results on 3D object detection. Our model produces accurate 3D bounding boxes on in-the-wild samples.

whether the grounding genuinely contributes to correct answers rather than producing hallucinated reasoning.

To study this, we evaluate our model on the MM-GCoT [63] benchmark, which provides three key metrics: answer accuracy (A-Acc), grounding accuracy (G-Acc), and answer–grounding consistency (Consist.). A-Acc measures the correctness of the textual answer. G-Acc follows the Acc@0.5 protocol, where a prediction is considered correct if its IoU with the ground-truth box exceeds 0.5. The consistency metric measures the percentage of predictions where both the answer and the grounding box are correct. We show results in Table 4, where our method outperforms baselines in all these metrics.

To further evaluate the performance in spatial reasoning scenarios, we conduct experiments on BLINK-Depth using the same grounding-based CoT formulation. As shown in Table 3, our method surpasses prior Region-VLMs, which are typically strong on this benchmark but require manually annotated masks as input. In contrast, our model achieves higher performance while performing grounding automatically. We additionally provide qualitative examples demonstrating that our model can accurately localize tiny regions and successfully handle point-level areas.

# 3.5. Analysis and Ablation Study

2D→3D vs Direct 3D Prediction. As shown in Table 5, first grounding the target region in 2D and then predicting its 3D bounding box leads to a clear improvement over direct 3D prediction. This two-step design is more visioncentric, as it explicitly forces the model to learn objectspecific visual features before performing 3D reasoning. It also naturally decomposes the task into two subproblems—2D grounding and 3D inference—where the former benefits from significantly larger amounts of training data across generic detection and grounding datasets. Leveraging this abundant 2D supervision allows the model to establish stronger spatial priors, which in turn improves downstream 3D detection performance.

Do spatial pretraining help 3D detection? Table 5 further supports this assumption by showing that spatial pretraining noticeably improves performance in the outdoor domain. The Omni3D dataset is highly imbalanced [44], with far fewer outdoor training samples compared to indoor scenes. As a result, models trained from scratch struggle to generalize in outdoor settings. Spatial pretraining provides a strong remedy by injecting generic 2D spatial and grounding knowledge, enabling the model to better transfer its learned priors to the 3D detection task. This demonstrates that leveraging 2D supervision is especially beneficial when 3D data is limited or unevenly distributed.

Effect of Intrinsic Normalization. Intrinsic normalization yields a modest, yet consistent improvement. Although its impact is smaller than the two factors discussed above, normalizing intrinsics helps reduce systematic biases when the model encounters cameras with different focal lengths. Without this normalization, the model may lead to small but noticeable localization offsets in the predicted 3D boxes.

Contribution of Pointmap Reconstruction. We further analyze the effect of pointmap reconstruction as an auxiliary task for 3D detection. This supervision strengthens the model’s ability to align region-level visual features with their corresponding 3D geometry. To isolate this effect from 2D grounding quality, we use ground-truth 2D boxes as our model input and evaluate only the 3D prediction. This separation is enabled by our disentangled pipeline and allows us to directly measure the reconstruction capability. As shown in Fig. 5, increasing the amount of pointmap supervision yields a clear scaling trend on SUN-RGBD: more pointmap data consistently improves 3D detection performance.

<table><tr><td rowspan="2"></td><td colspan="3">ATTRIBUTE</td><td colspan="3">JUDGEMENT</td><td colspan="3">OBJECT</td><td colspan="3">AVERAGE</td></tr><tr><td> $Acc_A \uparrow$ </td><td> $Acc_G \uparrow$ </td><td>Cons. ↑</td><td> $Acc_A \uparrow$ </td><td> $Acc_G \uparrow$ </td><td>Cons. ↑</td><td> $Acc_A \uparrow$ </td><td> $Acc_G \uparrow$ </td><td>Cons. ↑</td><td> $Acc_A \uparrow$ </td><td> $Acc_G \uparrow$ </td><td>Cons. ↑</td></tr><tr><td>Qwen2.5-VL-7B [64](AF)</td><td>73.6</td><td>72.5</td><td>59.8</td><td>87.9</td><td>56.3</td><td>51.5</td><td>57.8</td><td>64.1</td><td>59.1</td><td>73.1</td><td>64.3</td><td>56.8</td></tr><tr><td>Qwen2.5-VL-7B [64](GF)</td><td>48.8</td><td>82.6</td><td>45.7</td><td>80.6</td><td>72.8</td><td>62.4</td><td>26.7</td><td>62.3</td><td>32.6</td><td>52.0</td><td>72.6</td><td>46.9</td></tr><tr><td>LLaVA-7B [1](AF)</td><td>68.6</td><td>9.2</td><td>8.8</td><td>83.0</td><td>11.2</td><td>11.5</td><td>58.4</td><td>9.1</td><td>9.9</td><td>70.0</td><td>9.8</td><td>10.1</td></tr><tr><td>LLaVA-7B [1](GF)</td><td>59.7</td><td>6.3</td><td>5.6</td><td>82.5</td><td>0.5</td><td>0.6</td><td>35.9</td><td>5.5</td><td>9.7</td><td>59.4</td><td>4.1</td><td>5.3</td></tr><tr><td>LLaVA-GCoT-7B [63]</td><td>72.8</td><td>66.7</td><td>56.1</td><td>88.3</td><td>61.7</td><td>56.9</td><td>62.3</td><td>61.7</td><td>61.3</td><td>74.5</td><td>63.3</td><td>58.1</td></tr><tr><td>GR3D-8B (Ours)</td><td>78.9</td><td>77.3</td><td>66.7</td><td>85.0</td><td>79.6</td><td>70.4</td><td>71.1</td><td>65.7</td><td>66.1</td><td>78.3</td><td>74.2</td><td>67.7</td></tr></table>

Table 4. Results on the MM-GCoT benchmark. “AF” and “GF” correspond to answer-first and grounding-first prompting settings. AccA, AccG, and Cons. refer to answer accuracy, grounding accuracy, and consistency between them. 

<table><tr><td>2D→3D</td><td>PT</td><td>Cam</td><td> $AP_{15}^{sun} \uparrow$ </td><td> $AP_{3D}^{sun} \uparrow$ </td><td> $AP_{15}^{kit} \uparrow$ </td><td> $AP_{3D}^{kit} \uparrow$ </td></tr><tr><td>-</td><td>-</td><td>-</td><td>30.19</td><td>20.27</td><td>10.08</td><td>6.22</td></tr><tr><td>√</td><td>-</td><td>-</td><td>42.29</td><td>29.87</td><td>15.61</td><td>10.03</td></tr><tr><td>√</td><td>√</td><td>-</td><td>41.24</td><td>30.95</td><td>21.55</td><td>14.35</td></tr><tr><td>√</td><td>√</td><td>√</td><td>43.49</td><td>31.64</td><td>22.18</td><td>14.75</td></tr></table>

Table 5. Ablation study on the key components of GR3D-8B. “PT” denotes pretraining, “2D→3D” denotes 2D grounding followed by 3D prediction, and “Cam” denotes using normalized intrinsics.

# 4. Related Work

Spatial Vision Language Models. Recent work has rapidly expanded the spatial capabilities of VLMs across 2D grounding, monocular spatial reasoning, and multi-view 3D scene understanding [25, 26, 34, 48, 65–83]. Representative 2D spatial VLMs such as SpatialVLM [65], SpatialPin [66], VST [26], and SpatialLadder [80] focus on image-plane relations including relative position, direction, and distance using explicit spatial cues. SRGPT [21] improves fine-grained single-view spatial perception by introducing a region branch for more precise region-level querying. SR-3D [22] extends this idea by preserving and enabling multi-view spatial reasoning through a unified visual tokens space. Other multi-view spatial VLMs [27, 28] incorporate 3D cues or cross-view alignment for scene reasoning. Despite this progress, implicit 2D grounding and monocular 3D grounding from a single image remain underexplored [26, 45, 84]. In contrast, our approach jointly addresses both problems without requiring any spatial annotations at inference time.

Monocular 3D Grounding. Traditional 3D object detection has long focused on single-dataset, closed-set scenarios [22, 41, 42, 85–90], achieving strong performance but suffering from poor generalization to new environments. Initial efforts to overcome this [32, 91] utilized multi-dataset training to create universal detectors. The Omni3D [32], for instance, aggregated a wide variety of 3D datasets and proposed a universal model trained jointly on them. However, these models still confined to a predefined list of object classes seen during training. More recent work [43, 44, 92] have turned their focus to the openvocabulary setting. OVMono3D [43] proposes a two-stage “detect-then-lift” pipeline: it first employs an off-the-shelf 2D open-vocabulary detector [93] to generate 2D proposals, then feeds these region features into a specialized 3D head to regress 3D parameters. DetAny3D [44] proposes a more integrated, promptable architecture that directly fuses features from 2D foundation models and uses 2D prompts, e.g., points, boxes, or text to query the model for 3D outputs. Instead of treating 3D grounding as a standalone detection task, GR3D predicts 3D boxes as part of a unified VLM framework that also includes dynamic implicit 2D grounding. This unified approach allows GR3D to leverage grounding as a key driver to enhance general spatial alignment and geometry-consistent reasoning, improving performance on both grounded and non-grounded spatial tasks.

![](images/382796bd01bc14bd66ebc2abb010339f83784fae39d305666c6b4a20c6b6901d.jpg)

<details>
<summary>line</summary>

| Pointmap Data Scale (%) | AP₁₅₃D | AP₃D | AR₃D |
| ----------------------- | ------ | ---- | ---- |
| 0                       | 52     | 38   | 50   |
| 5                       | 53     | 39   | 51   |
| 100                     | 59     | 43   | 54   |
</details>

Figure 5. Scaling behavior when increasing pointmap prediction data on SUN-RGBD. More pointmap supervision leads to better 3D detection performance.

Thinking with Images. Our work is also realted to the recent line of “Thinking with Images” work [94–101]. Different from these approaches, GR3D avoids explicit visual thought processes and external tools, offering a more efficient and unified design through implicit 2D grounding and native 3D reasoning within the VLM’s generative flow.

# 5. Conclusion

We introduced GR3D, a spatial VLM that integrates explicit 2D grounding, implicit grounding for CoT reasoning, and monocular 3D grounding within a single framework. By enabling the model to reference visual evidence during generation and by coupling region-grounded queries with 3D box prediction, GR3D decomposes spatial understanding into grounded 2D perception followed by 3D inference. GR3D delivers consistent gains across various benchmarks, showing that grounding serves as an effective inductive bias for better spatial understanding in VLMs.

Acknowledgements. This project was supported, in part, by NSF CAREER Award IIS-2240014, gifts from Amazon, Meta, and Qualcomm.

# References

[1] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 8   
[2] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, 2024. 2   
[3] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. In CVPR, 2025. 3   
[4] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. In ICLR, 2024.   
[5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv:2308.12966, 2023.   
[6] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites. Science China Information Sciences, 2024.   
[7] OpenAI. Gpt-4 technical report, 2023. arXiv:2303.08774. 5   
[8] Gemini Team. Gemini: a family of highly capable multimodal models. arXiv:2312.11805, 2023. 2   
[9] Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets, et al. Habitat 2.0: Training home assistants to rearrange their habitat. In NeurIPS, 2021. 2   
[10] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In CVPR, 2024.   
[11] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In CoRL, 2023.   
[12] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In RSS, 2024.   
[13] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. RSS, 2025.   
[14] Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, et al. Egovla: Learning visionlanguage-action models from egocentric human videos. arXiv preprint arXiv:2507.12440, 2025.   
[15] NVIDIA Research. GR00T N1.5: An Improved Open Foundation Model for Generalist Humanoid Robots.

https://research.nvidia.com/labs/gear/ gr00t-n1\_5, 2025.   
[16] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.   
[17] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025. 2   
[18] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In CoRL, 2022. 2   
[19] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. In RSS, 2023.   
[20] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, et al. Robovqa: Multimodal long-horizon reasoning for robotics. In ICRA, 2024. 2   
[21] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In NeurIPS, 2024. 2, 5, 8   
[22] An-Chieh Cheng, Yang Fu, Yukang Chen, Zhijian Liu, Xiaolong Li, Subhashree Radhakrishnan, Song Han, Yao Lu, Jan Kautz, Pavlo Molchanov, et al. 3d aware region prompted vision language model. arXiv preprint arXiv:2509.13317, 2025. 3, 5, 8, 1, 2   
[23] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308, 2025. 5, 1   
[24] BAAI RoboBrain Team. Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029, 2025. 1   
[25] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. In CoRL, 2024. 8, 1   
[26] Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, et al. Visual spatial tuning. arXiv preprint arXiv:2511.05491, 2025. 5, 6, 8   
[27] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. In ICCV, 2025. 8   
[28] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In CVPR, 2025. 8

[29] Ting Huang, Zeyu Zhang, and Hao Tang. 3d-r1: Enhancing reasoning in 3d vlms for unified scene understanding. arXiv:2507.23478, 2025. 2   
[30] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 2, 6   
[31] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 2   
[32] Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3d: A large benchmark and model for 3d object detection in the wild. In CVPR, 2023. 2, 5, 6, 8   
[33] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024. 2, 6   
[34] Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. Sat: Dynamic spatial aptitude training for multimodal language models. In COLM, 2025. 2, 6, 8   
[35] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In CVPR, 2015. 5   
[36] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021. 5   
[37] Adel Ahmadyan, Liangkai Zhang, Artsiom Ablavatski, Jianing Wei, and Matthias Grundmann. Objectron: A large scale dataset of object-centric videos in the wild with pose annotations. In CVPR, 2021. 5   
[38] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021. 5   
[39] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal of robotics research, 2013. 5   
[40] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020. 5   
[41] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Imvoxelnet: Image to voxels projection for monocular and multi-view general-purpose 3d object detection. In WACV, 2022. 5, 6, 8

[42] Zechen Liu, Zizhang Wu, and Roland Toth. Smoke: Single- ´ stage monocular 3d object detection via keypoint estimation. In CVPRW, 2020. 5, 8   
[43] Jin Yao, Hao Gu, Xuweiyi Chen, Jiayun Wang, and Zezhou Cheng. Open vocabulary monocular 3d object detection. arXiv preprint arXiv:2411.16833, 2024. 5, 6, 8   
[44] Hanxue Zhang, Haoran Jiang, Qingsong Yao, Yanan Sun, Renrui Zhang, Hao Zhao, Hongyang Li, Hongzi Zhu, and Zetong Yang. Detect anything 3d in the wild. In CVPR, 2025. 5, 6, 7, 8   
[45] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 5, 6, 8, 1, 3, 4   
[46] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In ECCV, 2024. 6   
[47] xAI. RealWorldQA: A benchmark dataset for real-world spatial understanding. https://huggingface.co/ datasets/visheratin/realworldqa, 2024. 6   
[48] Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large visionlanguage models. In ACL, 2024. 6, 8   
[49] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In ACL, 2022. 6   
[50] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In NeurIPS, 2025. 6   
[51] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023. 6   
[52] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A Diagram is Worth a Dozen Images. In ECCV, 2016. 6   
[53] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 2020. 5   
[54] Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, and Afshin Dehghan. Cubify anything: Scaling indoor 3d object detection. In CVPR, 2025. 5   
[55] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In CVPR, 2024. 5   
[56] Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai

Chen, Tianfan Xue, et al. Embodiedscan: A holistic multimodal 3d perception suite towards embodied ai. In CVPR, 2024. 5, 2   
[57] Zhipeng Cai, Ching-Feng Yeh, Hu Xu, Zhuang Liu, Gregory Meyer, Xinjie Lei, Changsheng Zhao, Shang-Wen Li, Vikas Chandra, and Yangyang Shi. Depthlm: Metric depth from vision language models. In ICLR, 2026. 5   
[58] OpenAI. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/, 2024. 5, 2   
[59] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. https://github.com/InternLM/ xtuner, 2023. 5   
[60] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. In NeurIPS, 2024. 5, 2   
[61] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024. 5   
[62] Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024. 5   
[63] Qiong Wu, Xiangcong Yang, Yiyi Zhou, Chenxin Fang, Baiyang Song, Xiaoshuai Sun, and Rongrong Ji. Grounded chain-of-thought for multimodal large language models. arXiv preprint arXiv:2503.12799, 2025. 7, 8   
[64] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv:2502.13923, 2025. 8, 1, 2   
[65] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024. 8, 1   
[66] Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. In NeurIPS, 2024. 8   
[67] Wenxiao Cai, Iaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. In ICRA, 2025.   
[68] Wufei Ma, Haoyu Chen, Guofeng Zhang, Celso M de Melo, Alan Yuille, and Jieneng Chen. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In ICCV, 2025.   
[69] Yihong Tang, Ao Qu, Zhaokai Wang, Dingyi Zhuang, Zhaofeng Wu, Wei Ma, Shenhao Wang, Yunhan Zheng, Zhan Zhao, and Jinhua Zhao. Sparkle: Mastering basic spatial capabilities in vision language models elicits generalization to spatial reasoning. In Findings of EMNLP, 2025.   
[70] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual understanding and reasoning. In WACV, 2025.   
[71] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching

spatial understanding to 2d and 3d vision-language models for robotics. In CVPR, 2025.   
[72] Mingjie Xu, Mengyang Wu, Yuzhi Zhao, Jason Chun Lok Li, and Weifeng Ou. Llava-spacesgg: Visual instruct tuning for open-vocabulary scene graph generation with enhanced spatial relations. In WACV, 2025.   
[73] Damiano Marsili, Rohun Agrawal, Yisong Yue, and Georgia Gkioxari. Visual agentic ai for spatial reasoning with a dynamic api. In CVPR, 2025.   
[74] Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yaochen Hu, Lingfeng Zhang, Yingxue Zhang, Shuang Wu, Tongtong Cao, Guowei Huang, et al. Spatialcot: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning. arXiv:2501.10074, 2025.   
[75] Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models. In EMNLP, 2024.   
[76] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, 2025. 2   
[77] Yunze Man, Liang-Yan Gui, and Yu-Xiong Wang. Situational awareness matters in 3d vision language reasoning. In CVPR, 2024.   
[78] Xiongkun Linghu, Jiangyong Huang, Xuesong Niu, Xiaojian Shawn Ma, Baoxiong Jia, and Siyuan Huang. Multimodal situated reasoning in 3d scenes. In NeurIPS, 2024.   
[79] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv:2505.23747, 2025.   
[80] Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision-language models. arXiv preprint arXiv:2510.08531, 2025. 8   
[81] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In ICLR, 2026.   
[82] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. In NeurIPS, 2023.   
[83] Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, et al. Omnivinci: Enhancing architecture and data for omni-modal understanding llm. In ICLR, 2026. 8   
[84] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. SEED-Bench: Benchmarking Multimodal Large Language Models. In CVPR, 2024. 8   
[85] Xiaozhi Chen, Kaustav Kundu, Ziyu Zhang, Huimin Ma, Sanja Fidler, and Raquel Urtasun. Monocular 3d object detection for autonomous driving. In CVPR, 2016. 8

[86] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. TPAMI, 2024.   
[87] Tingting Liang, Hongwei Xie, Kaicheng Yu, Zhongyu Xia, Zhiwei Lin, Yongtao Wang, Tao Tang, Bing Wang, and Zhi Tang. Bevfusion: A simple and robust lidar-camera fusion framework. NeurIPS, 2022.   
[88] Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin. Fcos3d: Fully convolutional one-stage monocular 3d object detection. In ICCV, 2021.   
[89] Xuewu Lin, Tianwei Lin, Zixiang Pei, Lichao Huang, and Zhizhong Su. Sparse4d: Multi-view 3d object detection with sparse spatial-temporal fusion. arXiv preprint arXiv:2211.10581, 2022.   
[90] Renrui Zhang, Han Qiu, Tai Wang, Ziyu Guo, Ziteng Cui, Yu Qiao, Hongsheng Li, and Peng Gao. Monodetr: Depthguided transformer for monocular 3d object detection. In ICCV, 2023. 8   
[91] Zhuoling Li, Xiaogang Xu, SerNam Lim, and Hengshuang Zhao. Unimode: Unified monocular 3d object detection. In CVPR, 2024. 8   
[92] Zhenyu Wang, Yali Li, Taichi Liu, Hengshuang Zhao, and Shengjin Wang. Ov-uni3detr: Towards unified openvocabulary 3d object detection via cycle-modality propagation. In ECCV, 2024. 8   
[93] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV. Springer, 2024. 8, 2   
[94] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 8, 3   
[95] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. MM-ReAct: Prompting ChatGPT for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023. 3   
[96] Kaitao Chen, Shaohao Rui, Yankai Jiang, Jiamin Wu, Qihao Zheng, Chunfeng Song, Xiaosong Wang, Mu Zhou, and Mianxin Liu. Think twice to see more: Iterative visual reasoning in medical vlms. arXiv preprint arXiv:2510.10052, 2025. 3   
[97] D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. ViperGPT: Visual inference via python execution for reasoning. In ICCV, 2023. 3   
[98] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. In ICLR, 2026.   
[99] Xingyu Fu, Minqian Liu, Zhengyuan Yang, John Corring, Yijuan Lu, Jianwei Yang, Dan Roth, Dinei Florencio, and Cha Zhang. Refocus: Visual editing as a chain of thought for structured image understanding. In ICML, 2025. 3

[100] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. NeurIPS, 2024. 3   
[101] Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. In CVPR, 2024. 8   
[102] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014. 1, 2   
[103] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In CVPR, 2016. 1, 2   
[104] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1, 2   
[105] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1   
[106] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In CVPR, 2025. 1   
[107] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2   
[108] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024. 2   
[109] Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. In ICLR, 2025. 2   
[110] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv:2406.16852, 2024. 2   
[111] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, 2024. 2   
[112] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-onevision: Easy visual task transfer. TMLR, 2025. 2

[113] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In CVPR, 2022. 2, 3   
[114] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In CVPR, 2023. 2   
[115] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. arXiv preprint arXiv:2305.11172, 2023. 2   
[116] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2   
[117] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv:2306.15195, 2023. 2   
[118] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv:2409.12191, 2024. 2, 3   
[119] Ya-Qi Yu, Minghui Liao, Jihao Wu, Yongxin Liao, Xiaoyu Zheng, and Wei Zeng. Texthawk: Exploring efficient finegrained perception of multimodal large language models. arXiv preprint arXiv:2404.09204, 2024. 2   
[120] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In ECCV, 2020. 2, 3   
[121] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multiimage spatial intelligence. In ICLR, 2026. 3   
[122] Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. In NeurIPS, 2025. 3   
[123] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. NeurIPS, 2024. 3   
[124] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. GRIT: Teaching mllms to think with images. In NeurIPS, 2025. 3   
[125] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-ofthought reasoning for vision-language-action models. In CVPR, 2025. 3   
[126] Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, and Ivan Vulic. Visual planning: ´ Let’s think only with images. In ICLR, 2026. 4

# Grounded 3D-Aware Spatial Vision-Language Modeling Supplementary Material

![](images/0b6a7af999ed7d81a30a9b3a4a561684302bcc4ca4ae7fa31095a1e8b0c97ba5.jpg)

<details>
<summary>natural_image</summary>

Comparison of 3D object detection bounding boxes and 3D model views for Qwen3-VL 8B and Ours models, showing street scenes and spatial overlays (no text or symbols)
</details>

Figure 6. Qualitative comparison on 3D object detection between our model and Qwen3-VL 8B [45]. Our model produces more accurate 3D bounding boxes with fewer missed objects, demonstrating stronger spatial grounding and detection reliability.

# Table of Contents

1. More Results on 3D Detection   
2. More Results on 2D Grounding 1   
3. More Results on Multi-View Understanding 1   
4. Implementation Details 3   
5. More Related Work 3   
6. Discussions 4

# 1. More Results on 3D Detection

We show a qualitative comparison on 3D object detection between GR3D and Qwen3-VL-8B [45]. As shown in Fig. 6, when multiple objects are present, GR3D produces clearly better results due to our detect-then-lift technique. For indoor scenes, GR3D also predicts 3D boxes with more accurate orientations compared to Qwen3-VL-8B.

# 2. More Results on 2D Grounding

Our approach decomposes 3D detection into two steps: first grounding the target in 2D, then predicting its 3D properties based on the grounded region. Because accurate 2D grounding is essential for the first step, we evaluate our model on two grounding benchmarks. We first report results on RefSpatial [23], a benchmark designed for spatial referring that includes queries about vacant regions, spatial relations (e.g., “left of”, “between”), and fine-grained spatial logic. As shown in Table 6, our model achieves strong spatial referring performance and outperforms several baselines, including RoboRefer [23], demonstrating its ability to reason about complex spatial relations in 2D. We further evaluate on the widely used RefCOCO, RefCOCO+, and RefCOCOg datasets [102, 103] to measure general referring capability. These benchmarks contain diverse referring expressions involving object names, attributes, and relational descriptions. Results in Table 8 show that GR3D performs comparably to vision-specialized models and is on par with top VLMs such as InternVL-3.5 [104] or Qwen2.5-VL [64], confirming that our strong 2D grounding ability generalizes well to both spatial and standard referring benchmarks.

<table><tr><td>Method</td><td>LOCATION</td><td>PLACEMENT</td><td>UNSEEN</td></tr><tr><td>Gemini-2.5-Pro [105]</td><td>46.9</td><td>24.2</td><td>27.1</td></tr><tr><td>SpaceLLaVA-13B [65]</td><td>5.8</td><td>4.3</td><td>4.0</td></tr><tr><td>RoboPoint-13B [25]</td><td>22.8</td><td>9.2</td><td>8.4</td></tr><tr><td>Molmo-7B [106]</td><td>21.9</td><td>12.8</td><td>12.2</td></tr><tr><td>Molmo-72B [106]</td><td>45.7</td><td>14.7</td><td>21.2</td></tr><tr><td>RoboBrain-2.0-7B [24]</td><td>36.0</td><td>29.0</td><td>32.5</td></tr><tr><td>RoboRefer-8B [23]</td><td>52.0</td><td>53.0</td><td>37.7</td></tr><tr><td>GR3D-8B</td><td>63.0</td><td>50.0</td><td>41.5</td></tr></table>

Table 6. Performance comparison on RefSpatial [23].

# 3. More Results on Multi-View Understanding 3.1. Multi-View Extension of Our Framework

Our framework naturally extends from single-view to multiview settings through a unified spatial embedding design similar to SR-3D [22]. All image tokens, regardless of the view they originate from, are mapped into the same spatial feature space using depth-based and pixel-based positional cues. This allows the model to maintain consistent geometric relationships across views without requiring explicit point cloud reconstruction or global world coordinates.

<table><tr><td rowspan="2">Methods</td><td>Obj. Count</td><td>Abs. Dist.</td><td>Obj. Size</td><td>Room Size</td><td>Rel. Dist.</td><td>Rel. Dir.</td><td>Route Plan</td><td>Appr. Order</td><td></td></tr><tr><td colspan="4">Quantitative</td><td colspan="4">Qualitative</td><td>Avg.</td></tr><tr><td>Random</td><td>-</td><td>-</td><td>-</td><td>-</td><td>25.0</td><td>36.1</td><td>28.3</td><td>25.0</td><td>-</td></tr><tr><td>Human Level $^{\dagger}$ </td><td>94.3</td><td>47.0</td><td>60.4</td><td>45.9</td><td>94.7</td><td>95.8</td><td>95.8</td><td>100</td><td>79.2</td></tr><tr><td colspan="10">Proprietary Models (API)</td></tr><tr><td>GPT-4o [58]</td><td>46.2</td><td>5.3</td><td>43.8</td><td>38.2</td><td>37.0</td><td>41.3</td><td>31.5</td><td>28.5</td><td>34.0</td></tr><tr><td>Gemini-1.5 Flash [107]</td><td>49.8</td><td>30.8</td><td>53.5</td><td>54.4</td><td>37.7</td><td>41.0</td><td>31.5</td><td>37.8</td><td>42.1</td></tr><tr><td>Gemini-1.5 Pro [107]</td><td>56.2</td><td>30.9</td><td>64.1</td><td>43.6</td><td>51.3</td><td>46.3</td><td>36.0</td><td>34.6</td><td>45.3</td></tr><tr><td colspan="10">Open-source Models</td></tr><tr><td>InternVL2-2B [108]</td><td>24.9</td><td>22.0</td><td>35.0</td><td>33.8</td><td>44.2</td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL2-8B [108]</td><td>31.3</td><td>29.0</td><td>48.9</td><td>44.2</td><td>38.0</td><td>33.4</td><td>28.9</td><td>46.4</td><td>37.5</td></tr><tr><td>InternVL2-40B [108]</td><td>41.3</td><td>26.2</td><td>48.2</td><td>27.5</td><td>47.6</td><td>32.7</td><td>27.8</td><td>44.7</td><td>37.0</td></tr><tr><td>LongVILA-8B [109]</td><td>29.1</td><td>9.1</td><td>16.7</td><td>0.0</td><td>29.6</td><td>30.7</td><td>32.5</td><td>25.5</td><td>21.6</td></tr><tr><td>VILA-1.5-8B [2]</td><td>17.4</td><td>21.8</td><td>50.3</td><td>18.8</td><td>32.1</td><td>34.8</td><td>31.0</td><td>24.8</td><td>28.9</td></tr><tr><td>VILA-1.5-40B [2]</td><td>22.4</td><td>24.8</td><td>48.7</td><td>22.7</td><td>40.5</td><td>25.7</td><td>31.5</td><td>32.9</td><td>31.2</td></tr><tr><td>LongVA-7B [110]</td><td>38.0</td><td>16.6</td><td>38.9</td><td>22.2</td><td>33.1</td><td>43.3</td><td>25.4</td><td>15.7</td><td>29.2</td></tr><tr><td>LLaVA-Video-7B [111]</td><td>48.5</td><td>14.0</td><td>47.8</td><td>24.2</td><td>43.5</td><td>42.4</td><td>34.0</td><td>30.6</td><td>35.6</td></tr><tr><td>LLaVA-Video-72B [111]</td><td>48.9</td><td>22.8</td><td>57.4</td><td>35.3</td><td>42.4</td><td>36.7</td><td>35.0</td><td>48.6</td><td>40.9</td></tr><tr><td>LLaVA-OneVision-7B [112]</td><td>47.7</td><td>20.2</td><td>47.4</td><td>12.3</td><td>42.5</td><td>35.2</td><td>29.4</td><td>24.4</td><td>32.4</td></tr><tr><td>LLaVA-OneVision-72B [112]</td><td>43.5</td><td>23.9</td><td>57.6</td><td>37.5</td><td>42.5</td><td>39.9</td><td>32.5</td><td>44.6</td><td>40.2</td></tr><tr><td>SR-3D-8B</td><td>54.9</td><td>53.8</td><td>74.5</td><td>65.1</td><td>63.5</td><td>81.8</td><td>33.5</td><td>75.9</td><td>62.9</td></tr><tr><td>GR3D-8B</td><td>69.6</td><td>55.2</td><td>76.8</td><td>65.6</td><td>70.5</td><td>86.3</td><td>35.5</td><td>81.2</td><td>67.6</td></tr></table>

Table 7. We finetune our model on multi-view datasets [56, 113] following SR-3D [22], and then evaluate multi-view global spatial scene understanding on VSI-Bench [76]. Methods marked with † are evaluated on the Tiny subset. GR3D outperforms all state-of-the-art baselines, demonstrating strong spatial recognition capability.

<table><tr><td rowspan="2">Model Name</td><td colspan="3">REFCOCO</td><td colspan="3">REFCOCO+</td><td colspan="2">REFCOCOG</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td></tr><tr><td colspan="9">Vision Specialists</td></tr><tr><td>Grounding-DINO-L [93]</td><td>90.6</td><td>93.2</td><td>88.2</td><td>82.8</td><td>89.0</td><td>75.9</td><td>86.1</td><td>87.0</td></tr><tr><td>UNINEXT-H [114]</td><td>92.6</td><td>94.3</td><td>91.5</td><td>85.2</td><td>89.6</td><td>79.8</td><td>88.7</td><td>89.4</td></tr><tr><td>ONE-PEACE [115]</td><td>92.6</td><td>94.2</td><td>89.3</td><td>88.8</td><td>92.2</td><td>83.2</td><td>89.2</td><td>89.3</td></tr><tr><td colspan="9">Vision Language Models</td></tr><tr><td>InternVL3-1B [116]</td><td>85.8</td><td>90.1</td><td>81.7</td><td>76.6</td><td>84.1</td><td>69.2</td><td>82.8</td><td>82.6</td></tr><tr><td>InternVL3.5-1B [104]</td><td>85.4</td><td>89.7</td><td>80.2</td><td>77.7</td><td>85.5</td><td>69.5</td><td>81.9</td><td>81.6</td></tr><tr><td>InternVL3-2B [116]</td><td>89.8</td><td>92.6</td><td>86.4</td><td>84.0</td><td>89.2</td><td>76.5</td><td>87.6</td><td>87.2</td></tr><tr><td>InternVL3.5-2B [104]</td><td>88.7</td><td>91.6</td><td>84.8</td><td>82.7</td><td>88.4</td><td>76.6</td><td>85.6</td><td>85.5</td></tr><tr><td>Qwen2.5-VL-3B [64]</td><td>89.1</td><td>91.7</td><td>84.0</td><td>82.4</td><td>88.0</td><td>74.1</td><td>85.2</td><td>85.7</td></tr><tr><td>Shikra-7B [117]</td><td>87.0</td><td>90.6</td><td>80.2</td><td>81.6</td><td>87.4</td><td>72.1</td><td>82.3</td><td>82.2</td></tr><tr><td>CogVLM-G [60]</td><td>92.8</td><td>94.8</td><td>89.0</td><td>88.7</td><td>92.9</td><td>83.4</td><td>89.8</td><td>90.8</td></tr><tr><td>Qwen2-VL-7B [118]</td><td>91.7</td><td>93.6</td><td>87.3</td><td>85.8</td><td>90.5</td><td>79.5</td><td>87.3</td><td>87.8</td></tr><tr><td>Qwen2.5-VL-7B [64]</td><td>90.0</td><td>92.5</td><td>85.4</td><td>84.2</td><td>89.1</td><td>76.9</td><td>87.2</td><td>87.2</td></tr><tr><td>TextHawk2 [119]</td><td>91.9</td><td>93.0</td><td>87.6</td><td>86.2</td><td>90.0</td><td>80.4</td><td>88.2</td><td>88.1</td></tr><tr><td>InternVL3.5-8B [104]</td><td>92.4</td><td>94.7</td><td>88.7</td><td>87.9</td><td>92.4</td><td>82.4</td><td>89.6</td><td>89.4</td></tr><tr><td>GR3D-8B</td><td>91.8</td><td>94.5</td><td>88.8</td><td>87.5</td><td>91.4</td><td>81.0</td><td>89.5</td><td>89.7</td></tr></table>

Table 8. We evaluate GR3D ’s 2D grounding on RefCOCO, Ref-COCO+, and RefCOCOg [102, 103]. Baseline numbers are taken from [104, 118]. GR3D achieves grounding accuracy comparable to vision specialists [93, 114, 115] models and performs on par with top VLMs such as InternVL3.5 [104].

For multi-view inputs, the first view is processed exactly

as in the single-view case and is treated as the reference frame. Unlike SR-3D, which assumes a global world coordinate system and expresses all views in that space, our approach keeps everything in the coordinate frame of the first camera. Each additional view is transformed into this reference coordinate system using its intrinsics and extrinsics, so all depth-derived 3D locations and pixel-coordinate cues are expressed in the same spatial frame. After this transformation, tokens from different views that observe the same physical point occupy nearby positions in the unified embedding space. This allows the model to reason about 3D structure, occlusion, and cross-view consistency directly from the spatial tokens.

# 3.2. Results on VSI-Bench

To validate this design, we finetune our stage-1 model on multi-view datasets [56, 113] following SR-3D [22], and then evaluate multi-view global spatial scene understanding on VSI-Bench [76]. As shown in Table 7, GR3D achieves strong performance with an average score of 67.6 and surpasses all state-of-the-art baselines, showing that our method can effectively handle multi-view inputs.

# 3.3. Results on ScanRefer, ScanQA, MMSI, SPAR

To further evaluate the 3D grounding capabilities of GR3D on multi-view datasets, we conduct studies leveraging Scan-Refer [120] benchmark. However, ScanRefer assumes access to a pre-aligned world coordinate space, which is not directly compatible with the settings of Qwen3-VL [45] and ours. We therefore adapt ScanRefer into a frame/2D box grounding followed by 3D detection in the camera coordinate space, and compare against Qwen3-VL under the same input conditions. Our method outperforms Qwen3-VL-8B and is competitive with methods that use pre-aligned 3D input. We also report results on ScanQA [113], MMSI-Bench [121] and SPAR-Bench [122], showing consistent improvements.

<table><tr><td rowspan="2"></td><td colspan="2">SCANREFER</td><td colspan="3">SCANQA</td><td rowspan="2"></td><td rowspan="2">MMSI</td><td rowspan="2">SPAR</td></tr><tr><td>@0.25</td><td>@0.5</td><td>B4</td><td>C</td><td>EM</td></tr><tr><td>SPAR</td><td>48.8</td><td>43.1</td><td>15.3</td><td>90.7</td><td>27.7</td><td>GPT-4o</td><td>30.3</td><td>38.1</td></tr><tr><td>3D-LLaVA</td><td>51.2</td><td>40.6</td><td>17.1</td><td>92.6</td><td>-</td><td>InternVL2.5-8B</td><td>28.7</td><td>36.3</td></tr><tr><td rowspan="2">Video-3D LLM</td><td rowspan="2">58.1</td><td rowspan="2">51.7</td><td rowspan="2">16.2</td><td rowspan="2">102.1</td><td rowspan="2">30.1</td><td>Qwen2.5-VL-7B</td><td>25.9</td><td>33.1</td></tr><tr><td>Qwen3-VL-8B</td><td>31.1</td><td>39.8</td></tr><tr><td>Qwen3-VL-8B</td><td>37.7</td><td>33.2</td><td>-</td><td>-</td><td>-</td><td>NVILA-8B</td><td>28.1</td><td>34.1</td></tr><tr><td rowspan="2">GR3D-8B</td><td rowspan="2">52.0</td><td rowspan="2">46.1</td><td rowspan="2">18.1</td><td rowspan="2">105.1</td><td rowspan="2">29.2</td><td>SR-3D-8B</td><td>25.8</td><td>32.1</td></tr><tr><td>GR3D-8B</td><td>29.2</td><td>43.7</td></tr></table>

Table 9. Performance comparison on ScanRefer [120], ScanQA [113], MMSI-Bench [121], and SPAR-Bench [122].

# 4. Implementation Details

# 4.1. Model Architecture

Following NVILA-Lite, we use SigLIP as the vision encoder with an input resolution of 448 and a patch size of 14, paired with a Qwen-2-7B [118] LLM backbone. For training the stage-1 model, we follow SR-3D and enable dynamic tiling with up to 12 tiles per image. We also adopt SR-3D’s dynamic tiling region extractor, which provides a larger effective receptive field for regions and improves the model’s ability to handle small objects. During the first stage, the vision encoder is frozen and only the remaining modules are trained. For the second CoT detection stage, the LLM is fine-tuned to learn the reasoning structure and the autoregressive 3D prediction format.

# 4.2. Training Hyper-parameters

Both stages use the same optimization schedule: a warmup ratio of 0.03 and a cosine learning-rate scheduler. In the stage-1 stage, we train all non-visual modules with AdamW and a base learning rate of $5 \times 1 0 ^ { - 5 }$ , while keeping the SigLIP encoder frozen. The second CoT detection stage fine-tunes only the Qwen-2-7B LLM with a smaller learning rate of $1 . 5 \times 1 0 ^ { - 5 }$ to stabilize chain-of-thoughts text generation. Training the stage-1 model takes approximately 4 days on 8 nodes of A100 servers, while the second stage takes about 4 hours on the same compute setup.

# 4.3. Data Composition

The data composition for both training stages is summarized in Table 10. Most of our training data follow NVILA’s data recipe, though we use only a subset due to computational constraints. Part of the spatial data is inherited from SR-3D, while many of the 2D grounding datasets are newly introduced to the model and trained for the first time on our weights. For the 3D detection data used in stage 1, we follow DetAny3D’s filtering rules on Omni3D to select highquality training objects, and convert each scene into multiturn conversations with up to 10 rounds. For the CoT detection data used in stage 2, we construct multi-object reasoning sequences by selecting up to 20 objects for each target.

<table><tr><td colspan="2">Stage-1 Data</td></tr><tr><td>Hybrid</td><td>ShareGPT4V-SFT, Molmo, The Cauldron, Cambrian, LLaVA-OneVision</td></tr><tr><td>Captioning</td><td>MSR-VTT, Image Paragraph Captioning, ShareGPT4V-100K</td></tr><tr><td>Reasoning</td><td>CLEVR, NLVR, VisualMRC</td></tr><tr><td>Document</td><td>DocVQA, UniChart-SFT, ChartQA</td></tr><tr><td>OCR</td><td>TextCaps, OCRVQA, ST-VQA, POIE, SORIE, SynthDoG-en, TextOCR-GPT4V, ArxivQA, LLaVAR</td></tr><tr><td>General VQA</td><td>ScienceQA, VQAv2, ViQuAE, Visual Dialog, GQA, Geo170K, LRV-Instruction, RefCOCO, GeoQA, OK-VQA, TabMVP, EstVQA</td></tr><tr><td>Diagram &amp; Dialogue</td><td>DVQA, AI2D, Shikra, UniMM-Chat</td></tr><tr><td>Instruction</td><td>LRV-Instruction, SVIT, MMC-Instruction, MM-Instruction</td></tr><tr><td>Text-only</td><td>FLAN-1M, MathInstruct, Dolly, GSM8K-ScRel-SFT</td></tr><tr><td>Knowledge</td><td>WordART, WIT, STEM-QA</td></tr><tr><td>Medical</td><td>PathVQA, Slake, MedVQA</td></tr><tr><td>Region</td><td>RegionGPT</td></tr><tr><td>Spatial &amp; 2D Grounding</td><td>RefCOCO, MGrounding, Molmo, Groma, Spatial-RGPT, RefSpatial, SAT, EmbSpatial, DepthLM</td></tr><tr><td>Detection</td><td>Omni3D, EmbodiedScan</td></tr><tr><td colspan="2">Stage-2 Data</td></tr><tr><td>Detection</td><td>Omni3D-CoT</td></tr><tr><td>Spatial</td><td>RefSpatial-CoT, MMG-CoT, EmbSpatial-CoT, Vis-CoT</td></tr></table>

Table 10. Data recipe for training GR3D.

# 5. More Related Work

A related line of research, recently formalized as Thinking with Images [94], focuses on improving complex VLM reasoning by decomposing problems into explicit, intermediate steps, treating vision as a dynamic workspace. Many such methods act as “commanders” orchestrating external visual tools [95, 96] or as “visual programmers” that generate code for custom analysis and edits [97–99]. Others generate intermediate visual representations to guide reasoning, often called Visual Chain of Thought (V-CoT) [123]. These V-CoT methods may interleave text with explicit visual groundings [124], sketch visual artifacts [100], generate subgoal images for robotics [125], or perform planning entirely through visual state sequences [126]. While these methods enhance transparency and performance on complex tasks, they still focus on 2D image space, rely on coarse region-selection cues or external tools, and rarely integrate these reasoning steps with a 3D spatial framework. In contrast, our GR3D framework bypasses the need for an explicit, step-by-step visual thought process. It achieves a more seamless integration by performing implicit 2D grounding and unified 3D reasoning natively within the VLM’s generative flow.

![](images/18b4500ae8c5797b0b0668cee1d847990ff28684408de4dd3342ef17154dcd4c.jpg)  
Figure 7. A failure case analysis of GR3D compared to Qwen3-VL-8B.

# 6. Discussions

# 6.1. Standard VLM without PE

Our method can be applied to standard VLMs, but 3D priors further improve performance. Using positional embeddings, Omni3D mAP (averaged over 6 datasets) improves from 22.9 to 25.4 compared to a standard VLM without positional embeddings, showing their benefit as a simple and effective 3D prior.

# 6.2. Hallucinations

We do not observe frequent hallucinated 2D boxes. The main failures are missing or ambiguous 2D grounding, which leads to incorrect predictions. We show an example above and compare them with Qwen3-VL-8B [45].

# 6.3. Effect of Intrinsic Estimation Errors

The effect of intrinsic normalization is modest. Since the normalization only determines the resolution size, it does not require highly accurate intrinsic estimates. In practice, off-the-shelf intrinsic estimators are sufficiently accurate: using GeoCalib for focal length prediction on Omni3D results in only a 1.2 mAP drop (averaged over 6 datasets).

# 6.4. CoT Data Robustness

We conduct an ablation study on the impact of data quality by training with a noisier corpus, which leads to performance drops from 74.2 to 62.8 on MM-GCoT’s grounding accuracy. Human evaluation on 200 randomly sampled instances from the filtered corpus shows that 95.5% of the generated bounding boxes are accurate.

# 6.5. Latency Analysis

We implement multimodal prefix caching to ensure that the inference pipeline runs at a speed comparable to standard autoregressive generation. For Region Insertion, the process only extracts relevant areas from already encoded image features and passes them through a lightweight MLP projector, without re-encoding the image. We provide a latency analysis that compares our method with other baselines, tested on the same input image using a single A100 GPU. Our model is fastest among VLMs due to a more efficient dynamic tiling–based vision encoder (vs. AnyRes). The additional cost per inserted region is only 0.01 s, which is a small fraction of the total 2.7 s inference time.

<table><tr><td></td><td>DetAny3D</td><td>VST-7B</td><td>Qwen3-VL-8B</td><td>GR3D-8B</td></tr><tr><td>Latency (s)</td><td>0.98</td><td>2.76</td><td>3.23</td><td>2.72</td></tr></table>

# 6.6. Limitations

Our approach has two main limitations. First, the inference speed is slower compared to vision specialists. This is mainly due to the use of a large language model backbone, our two-stage “2D grounding first” pipeline, and the fact that 3D bounding boxes are generated autoregressively as text tokens, all of which introduce additional latency. Second, current 3D detection datasets are still limited. Popular datasets such as Omni3D cover only a narrow set of environments, camera configurations, and object categories, which restricts the diversity and scale of 3D supervision our model can learn from. As a result, further progress will benefit from larger and more diverse 3D datasets with broader scene coverage and richer object annotations.