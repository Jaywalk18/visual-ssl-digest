# Reinforcing Dual-Path Reasoning in Spatial Vision Language Models

Yatai Ji1,2,\* An-Chieh Cheng2,3 Yang Fu2,3 Yukang Chen2 Han Zhang2 Zhaojing Yang3 Wei Huang1,2 Ka Chun Cheung2 Song Han2 Vidya Nariyambut Murali2 Pavlo Molchanov2 Simon See2 Jan Kautz2 Hongxu Yin2 Ping Luo1,† Sifei Liu2,†

1 The University of Hong Kong 2 NVIDIA 3 University of California, San Diego

Project page: https://sr-real.github.io/

## Abstract

Spatial VLMs have made substantial progress in geometric perception, yet complex spatial reasoning—requiring multi-step inference that chains depth cues, distance comparisons, and scene relations—remains challenging. Moreover, different spatial queries demand fundamentally different strategies: some are best resolved through purely linguistic, step-by-step deduction over scene relations, while others benefit from first explicitly grounding objects in 3D space before performing quantitative inference. No existing framework simultaneously addresses both strategies within a unified spatial VLM. We present Dual-Path Spatial Reasoning via Reinforcement Learning for Spatial VLMs (SR-ReaL), a unified framework that equips a spatial VLM with two complementary reasoning paths: Language-Only Reasoning (LOR), which performs step-by-step linguistic deduction, and Detect-Then-Reason (DTR), which first detects 3D geometric cues (e.g., object centers or bounding boxes) via region tokens before explicit geometric inference. SR-ReaL trains both paths through a two-stage pipeline. In the cold-start stage, we construct structured chain-of-thought supervision for LOR and DTR, expose a region-to-3D grounding interface that links visual region tokens to predicted 3D coordinates, and blend 2D/3D grounding data with general-purpose VQA for stable initialization. The subsequent RL stage jointly optimizes both paths using GRPO with accuracy and format rewards; for DTR, a discrete center-based detection reward further refines geometric alignment during reasoning.

Across diverse spatial benchmarks, SR-ReaL significantly outperforms spatial VLM baselines: (i) a single RL-trained checkpoint supports both reasoning paths, with DTR excelling in region-aware tasks through precise 3D localization and LOR enhancing general spatial reasoning; (ii) jointly training both paths fosters mutual reinforcement, with each mode benefiting from the other’s supervision; (iii) high-quality, blended cold-start data is crucial for stable RL optimization and cross-domain transfer; and (iv) the model generalizes across datasets and domains without per-task tuning.

## Contents

Introduction 3  
2 Related Work 4

3 Methods 5

3.1 Spatial VLM Foundation 5  
3.2 Region-to-3D Detection 5  
3.3 Cold-Start Supervised Fine-tuning 6  
3.4 Reinforcement Learning Stage 7

4 Experiments 8

4.1 Experimental Setup 8  
4.2 Main Results . 9  
4.3 Qualitative Results 11  
4.4 Ablation Analysis 11

5 Conclusion 12

A Implementation Details  
B Preliminary knowledge of GRPO  
C CoT Data Construction 1  
D More Results 2  
E More Visualization 3  
F Limitations 3  
G Broader Impact 5

Qwen2.5-VL: D

SR-3D: A

![](images/d9cff75a1f472dac199af46deffc14190b56698f850ccac72ba0d5ad4513b58f.jpg)

![](images/441c9464bf972b93328c24e3a05941d1fac17f63710978e0cdbee911dfcdab97.jpg)

(w/o reasoning)

## SpatialReasoner:

<think>The 3Dlocationof papers is (-0.2,0.9,0.8),and the observer is (-0.4,0.7,1.O).The vectorbetweenthem is (0.2,0.2,-0.2).The left direction is (-0.9,-0.2,0.2). cosine -0.90 → 154.6°→behind. The front direction is (- 0.2,-0.2，-1.O)..cosine0.17→80→right.Theabove directionis(0.2,-0.7-0.6)..osine-0.14→98°→below. Thus,finalorder isright,below,behind.</think>

<answer>right, below, behind</answer>

![](images/f98de9b87f82f3b902db9d9e9db28558ba9ab7fa847442a2fd18463f51ebb0ac.jpg)

## SpatialLadder:

<think> Let me think through this step by step. The imageshowsa desk with various items... keyboard, papers,cup.Thequestionasksforthepapers'position before/aftermovingtothecenterofthebooksand facing the monitor.

1.Before: paperson the desk,in front of the keyboard. 2.After:papersstillonthedesk..slightlymore centered. Fromtheobserver'sperspective,thepapersareinfront of the keyboard when viewed fromthe side... givinga leftfrontrelation.Therefore,thepapersaretotheleftofthe observerandinfront.</think>

<answer>A</answer>

![](images/0d4dbf7592e7cadabd2158d4615fa4c89934977ce01962b6dbbc395eec3fd1c8.jpg)

(w/ reasoning)

## Spatial Query

Question: When the observer moves to the center of the books (green bbox) and faces themonitor (bluebbox)，whatisthe spatial relationship between the papers (red bbox)and the observer?

A. left, front C. right, below, front

B. below, behind D.right,behind

![](images/2ebbcb31cc66d9e13a9f968951ffc3f8d97df0372ac2f12457c962b878629523.jpg)

<details>
<summary>natural_image</summary>

Office desk with papers, a mug, and a chair (no visible text or symbols)
</details>

## SR-ReaL: Language Only Reasoning

<think>When moving to the center of the booksand facing the monitor,thepapersarenowtotherightof theobserver.The papersarealsoonthe desk,which is belowthe observer's eye level.Finally，thepapersare infrontoftheobserver.So,the final position of the papers isright,below,front.</think>

<answer>C</answer>

![](images/88aa585d572f6b87275dddbdf24b12b214933325038551bb6b5cf09075b61ef9.jpg)

## SR-ReaL: Detect-then-Reasoning

<detect> Find 3 relevant objects:

<3d\_boxceter="[-0.07-0.30,1.55]">papers (red)</3d> <3d\_boxeter"[-66,-.63,.61]">ooks(green)</d <3do[.]e)/ </detect>

<think>The papers (red box) are located at Ptar [-O.07,-0.30, 1.55],ilersii=[6] and the facing direction is towards Pface=[0.72,-0.56,1.41]. ThevectortotargetisVrel=Ptar-Pbs=[O.59,.33,-006].he obeverdtce 0.14].

-Thepapersarettherigtoftheoseerslinefsigtfd -Positive△y (O.33)in Vrel,Sopapersarebelow. -Thedotproductof thesevectorsispositive,confirming the target isintheforwarddirection. Thus,thepapersareright,below,andfrontof the observer.</think>

<answer>C</answer>

![](images/691fe00391bb92d7c47e9be215fd38dd79cd7dff46ff2dc2999395f138fa2657.jpg)  
Figure 1: A spatial imagination query where SR-ReaL resolves the task under both reasoning paths—Language-Only Reasoning (LOR) and Detect-Then-Reason (DTR) (right-most column). Prior art (left-most column) fail on such examples due to inaccurate or insufficient geometric deduction.

(our enhanced reasoning)

## 1. Introduction

Large Vision-Language Models (VLMs) have rapidly advanced in interpreting and reasoning over visual content, driven by increasingly capable architectures (Bai et al., 2023; Chen et al., 2023; Liu et al., 2024; Lin et al., 2024; Liu et al., 2025c; Ye et al., 2025; Ji et al., 2023, 2025) and proprietary systems (OpenAI, 2024; Google DeepMind, 2023), yet their spatial abilities remain persistently limited—studies reveal weaknesses in understanding 3D layout, depth, occlusion, and viewpoint-dependent relations (Kamath et al., 2023; Liu et al., 2023; Wang et al., 2024a; Shiri et al., 2024). This has motivated spatial VLMs: models that explicitly encode geometric structure and depth cues, enabling more accurate spatial understanding of 3D scenes (Chen et al., 2024; Guo et al., 2024; Cheng et al., 2024; Zhu et al., 2024; Cheng et al., 2025; Ma et al., 2025b). Yet answering challenging spatial questions requires more than recognizing geometric relations—it demands combining multiple cues, applying geometric rules, and reasoning through intermediate steps, a capability we call spatial reasoning.

Spatial reasoning goes substantially beyond spatial understanding in both structure and difficulty. While many queries depend on a single local relation, more challenging ones require multi-step inference that chains relations, integrates global context with localized cues, or performs quantitative 3D comparison. Crucially, different spatial queries call for different reasoning strategies: some are best resolved through purely linguistic, step-by-step deduction over scene relations, while queries involving depth, distance, or precise object localization benefit from first grounding objects in 3D space before performing inference. Recent work has applied reinforcement learning (RL) to elicit such multi-step reasoning in language and multimodal models, including R1-style approaches (DeepSeek-AI, 2025; Huang et al., 2025; Meng et al., 2025; Yang et al., 2025b; Zhang et al., 2025b; Wang et al., 2025b) and spatial RL pipelines (Wang et al., 2025a; Ma et al., 2025b; Li et al., 2025; Ma et al., 2025a). However, these systems are built on generic VLMs that lack strong spatial perception and thus cannot leverage the rich geometric structure provided by spatial VLMs. Moreover, no existing method jointly supports both linguistic and geometry-grounded reasoning paths within a single spatial VLM, nor provides the structured supervision to develop them in a unified, mutually reinforcing framework.

We address this problem with SR-ReaL, a unified spatial reasoning model that equips a spatial VLM with two complementary reasoning paths: Language-Only Reasoning (LOR), which performs step-by-step linguistic deduction, and Detect-Then-Reason (DTR), which injects explicit 3D cues (e.g., centers or oriented boxes) before carrying out quantitative inference. Figure 1 shows a spatial query with our LOR and DTR solutions. To enable flexible use of both modes, SR-ReaL trains the model through a two-stage pipeline: a cold-start supervised stage that introduces linguistic and geometry-aware traces while adding the grounding capability missing in the base model, followed by a reinforcement learning stage that refines reasoning using structured feedback. This unified formulation ties the two reasoning paths naturally to both the cold-start

and RL procedures.

In the cold-start stage, we construct two structured chain-of-thought (CoT) datasets aligned with LOR and DTR. The LOR dataset provides purely linguistic traces: given a spatial question–answer pair, a large vision-language model is prompted to generate concise step-by-step explanations that derive the answer from scene relations. The DTR dataset augments this format with explicit geometry by including detected or annotated 3D centers or bounding boxes together with reasoning steps that perform quantitative comparison or simple geometric computation. To enable DTR, we utilize a region-to-3D grounding paradigm with the spatial VLM’s region branch (Cheng et al., 2024, 2025): given a visual region token, the model predicts its 3D center or bounding box, linking region references to geometric quantities through a unified interface. Beyond these two CoT datasets, the cold-start stage also mixes 2D/3D grounding data, region-prompted QA, and general multimodal data so the model acquires grounding and reasoning without losing basic vision–language abilities. Training solely on CoT-style data degrades general multimodal behavior, whereas this blended initialization yields more stable RL optimization and stronger cross-domain transfer.

During RL, SR-ReaL trains both reasoning paths using a DAPO-style Group-Relative Policy Optimization (GRPO) with online filtering. The two paths are prompt-guided: the system prompt explicitly specifies whether to use LOR or DTR, and selecting DTR triggers the model to incorporate extracted region tokens for 3D localization, producing the required 3D predictions before reasoning. The reward combines task accuracy with a format reward and, for DTR, additionally includes a discretized detection reward that facilitates 3D grounding. This setup enables both reasoning modes to be jointly optimized under a unified RL framework while ensuring their compatibility and mutual enhancement.

We evaluate SR-ReaL across a comprehensive suite of spatial benchmarks spanning three categories: (i) region-grounded datasets with localized object references; (ii) global benchmarks requiring whole-scene reasoning; and (iii) out-ofdistribution (OOD) sets with shifted imagery or question styles. The results demonstrate that SR-ReaL is highly versatile, effectively supporting both reasoning modes to adapt to diverse scenarios. It consistently outperforms the spatial understanding baseline across multiple benchmarks, with DTR specifically showing superior performance on region-based tasks due to precise 3D localization. Furthermore, we find that jointly training both reasoning modes fosters mutual reinforcement, and improved 3D grounding in DTR directly enhances reasoning accuracy. Our ablation studies also highlight that RL significantly boosts generalization capabilities, while the quality of cold-start data ensures stable RL optimization and robust transfer performance.

These findings underscore the potential of SR-ReaL in elevating spatial VLMs from perception to reasoning. We demonstrate that with proper grounding and initialization, RL effectively strengthens both linguistic and geometric reasoning skills. In summary, our contributions are: (i) SR-ReaL, a unified dual-path framework that advances spatial reasoning in spatial VLMs via structured RL training; (ii) a structured CoT formulation with two parallel paths—LOR (linguistic) and DTR (geometry-aware) with region-to-3D grounding and a discrete center-based detection reward; and (iii) demonstration of significant performance gains and strong cross-domain generalization, highlighting how region grounding, data quality, and RL supervision jointly determine spatial reasoning behavior.

## 2. Related Work

Spatial VLMs. Early image-based spatial VLMs focus on object-centric properties such as position, size, orientation, and pairwise distance or direction (Kamath et al., 2023; Liu et al., 2023; Rajabi and Kosecka, 2023; Ranasinghe et al., 2024; Shiri et al., 2024; Lee et al., 2025; Wang et al., 2024a; Tang et al., 2025; Liu et al., 2025b,a; Ma et al., 2025b,a), with SpatialVLM (Chen et al., 2024) scaling spatial QA in 2D and SpatialRGPT (Cheng et al., 2024) introducing region prompting and depth cues for finer-grained QA. More recent models extend to multi-view or video inputs via 3D positional cues or cross-frame alignment—LLaVA-3D (Zhu et al., 2024), Video-3D-LLM (Zheng et al., 2025), and SR-3D (Cheng et al., 2025)—enabling multi-view 3D reasoning but assuming known camera poses.

RL for VLM Reasoning. Early reasoning advances rely on supervised CoT learning (Muennighoff et al., 2025; Thawakar et al., 2025). Building on CoT initialization, RL has emerged as a general mechanism: DeepSeek-R1 (DeepSeek-AI, 2025) shows rule-based RL alone induces long-form reasoning without human-labelled trajectories. In the multimodal setting, Vision-R1 and MM-Eureka (Huang et al., 2025; Meng et al., 2025) adapt R1-style GRPO to mathematical and visual tasks, R1-OneVision (Yang et al., 2025b) converts visual content into structured text before applying RL, and R1-VL/VL-Rethinker (Zhang et al., 2025b; Wang et al., 2025b) introduce step-wise reward shaping to improve multimodal reasoning reliability.

![](images/4f68c75f44f8ba8a565648326f16114613c222fbfced7cb5990a0689391274b9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Stage 1: Cold-Start SFT
        A["Spatial CoT"]
        B["3D Grounding w Global Question"]
        C["3D Grounding w Region Prompt"]
        D["Region-prompted General QAs"]
        E["General Multimodal QAs"]
    end

    subgraph Stage 2: Reinforcement Learning
        F["User: When the observer shifts to the box <mask> and toward dresser <mask>, what is the positioning of tv <mask>?"]
    end

  F --> G["Vision Backbone"]
  G --> H["LLM"]
  H --> I["sample"]
  I --> J["rollouts"]
  J --> K["×0.7"]
  J --> L["×0.1"]
  J --> M["×0.2"]
  J --> N["r_accuracy = I(y = y)"]
  J --> O["r_format = {1, valid format 0, otherwise}"]
  J --> P["Δd = ||Center_pred - Center_GT||_2"]
  J --> Q["r_detect = {1.0, Δd < 0.1 0.8, 0.1 < Δd < 0.3 ...}"]
  J --> R["r_n"]
  R --> S["All rollouts have equal rewards"]
  S --> T["filter"]
  T --> U["×"]
  U --> V["Group Computation"]
  V --> W["A1 A2 ... An"]
  W --> X["optimize"]
  X --> Y["User: When the observer shifts to the box <mask> and toward dresser <mask>, what is the positioning of tv <mask>?"]
  Y --> Z["sample"]
  Z --> AA["rollouts"]
  AA --> AB["×0.7"]
  AA --> AC["×0.1"]
  AA --> AD["×0.2"]
  AA --> AE["r_accuracy = I(y = y)"]
  AA --> AF["r_format = {1, valid format 0, otherwise}"]
  AA --> AG["Δd = ||Center_pred - Center_GT||_2"]
  AA --> AH["r_detect = {1.0, Δd < 0.1 0.8, 0.1 < Δd < 0.3 ...}"]
  AA --> AI["r_n"]
```
</details>

Figure 2: Two-stage Training Pipeline: In Stage 1 (cold-start SFT), we fine-tune the spatial VLM with spatial CoT, 2D/3D grounding (global and region prompts), and region-prompted multimodal QA to initialize spatial reasoning ability. In Stage 2, we apply RL on multiple-choice and filling spatial QA, optimizing grouped rollouts of LOR/DTR trajectories with accuracy, format, and 3D-center detection rewards.

Spatial Reasoning VLMs. Recent work has focused on equipping vision–language models with stronger spatial reasoning by combining structured spatial supervision, explicit 3D cues, and reinforcement learning. 3D-R1 (Wang et al., 2025a) extends R1-style training to multi-view 3D scenes with known camera poses, strengthening 3D spatial reasoning through detailed multi-view textual grounding. SpatialReasoner (Ma et al., 2025a) and SpatialLLM (Ma et al., 2025b) introduce explicit 3D scene representations that unify perception, spatial computation, and reasoning within a single model, improving their ability to handle diverse 3D spatial questions despite relying solely on single-view RGB inputs. Visual Spatial Tuning (Wu et al., 2025) adopts a progressive pipeline, first improving spatial perception through supervised tasks and then refining spatial CoT via RL. SpatialLadder (Li et al., 2025) employs a three-stage schedule—early 3D localization, structured spatial understanding, and RL-based refinement—to enhance spatial reasoning. ViGoRL (Sarch et al., 2025) instead applies multi-turn RL anchored to 2D image coordinates, encouraging the model to iteratively crop, localize, and reason on spatial and visual search tasks. In contrast, SR-ReaL conducts RL in a region-based spatial VLM with single-view settings, introducing an internal region-to-3D grounding interface and showing how RL shapes the model’s choice between two reasoning paths—LOR and DTR—within one unified architecture.

## 3. Methods

## 3.1. Spatial VLM Foundation

We build a spatial vision–language model following the SR-3D framework (Cheng et al., 2025), reimplementing its joint text–image architecture. Specifically, the model supports interleaved text, image, and region tokens for spatially grounded reasoning. We follow the single-view setup, which applies to single-image or multi-image settings.

For spatial encoding, we adopt SR-3D’s positional embedding formulation, which encodes 2D coordinates and the depth map while incorporating camera intrinsics and extrinsics to provide 3D-aware positional bias. Then the pixel-wise 3D position map is encoded and added to the corresponding visual embeddings, effectively fusing geometric information into the visual features.

We further introduce the region-prompt interface that allows region tokens (e.g., mask or bounding-box references) to be injected alongside text tokens. Through cross-attention, these region tokens interact with vision features, enabling localized reasoning over specific spatial regions. Note that the original SR-3D does not include region-level 3D signal prediction or grounding; in the following sections, we extend this capability through additional 2D and 3D grounding supervision.

## 3.2. Region-to-3D Detection

3D spatial localization is a critical capability for spatial reasoning, which enables our detect-then-reason paradigm to perform structured, explicit geometric calculations based on coordinates, yielding more accurate results. However, directly predicting 3D positions from text is highly challenging. To address this, we use region tokens as a bridge, leveraging 2D position priors to facilitate 3D localization. Instead of localizing directly from language prompts, we input region prompts to the LLM to generate corresponding 3D coordinates, effectively disentangling semantic parsing from 3D perception. The region prompt is marked as <mask> in the text to keep consistent with Cheng et al. (2025), which is replaced by the visual token of the corresponding image region. In Table 5, we show that direct grounding (remove the region prompt) leads to significant performance drop for the DTR inference path.

![](images/dec778c5185ee7a8d79aaf4a40c008a829688fa64af58542c9eea58db9f55e7b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["INPUT SOURCE"] --> B["COT SYNTHESIS"]
  B --> C["REASONING OUTPUT"]
  A --> D["1 Image + Language"]
  D --> E["Q: Describe the window (red bbox) compared to the chair (blue bbox) in terms of spatial location."]
  D --> F["A: D. right, above, farther"]
  B --> G["2 Image + Geometry"]
  G --> H["Q: Identify the object that is spatially farther to box (red point): the green point books or the blue point book."]
  G --> I["A: books (green point)"]
  B --> J["3D geometry: boxx (red point): {'center': [-0.63, -0.59, 1.72"], 'size': ..., 'rotation': ...}]
  J --> K["Indoor & Outdoor lamp: center: [0, -0.2, 1.7"] size: [0.2, 0.36, 0.2] rotation: ...]
  J --> L["truck: center: [-1.37, -0.2, 35.2"] size: [6.37, 2.98, 2.88] rotation: ...]
  B --> M["3D scene graph"]
  M --> N["Generate spatial QA from scene graphs for navigation, manipulation, and affordance. Navigation Manipulation Affordance"]
  N --> O["? Questions"]
  O --> P["Complex Reasoning"]
  P --> Q["Q: Could the lamp fit on the console table's lowe's shelf? Choose the right response: A. No. B. Yes. C. Maybe. think: The lamp is 0.36 units tall, and the shelf space is about 0.5 units. Since 0.36 < 0.5 and it fits horizontally, it can fit. A: B (Yes)"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#fcf,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#fcf,stroke:#333
    style J fill:#cff,stroke:#333
    style K fill:#cff,stroke:#333
    style L fill:#cff,stroke:#333
    style M fill:#ffc,stroke:#333
    style N fill:#cfc,stroke:#333
    style O fill:#cfc,stroke:#333
    style P fill:#cfc,stroke:#333
```
</details>

Figure 3: CoT Data Construction: We generate step-by-step CoT of two spatial reasoning paths: Language-Only Reasoning (top), Detect-Then-Reason with geometry-grounded deduction (middle). Complex Spatial tasks Construction: Using multimodal scene-graph datasets that provide both visual and geometric annotations, we prompt LVLM to generate higher-level reasoning task data (bottom).

## 3.3. Cold-Start Supervised Fine-tuning

The whole pipeline is shown in Figure 2. We introduce a cold-start phase that equips the base model with essential reasoning capabilities before reinforcement learning. In the following sections, we describe how we construct CoT data for both language-only reasoning (LOR) and detect-then-reason (DTR), and how we blend them with general supervision to initialize the model effectively.

CoT Data Construction—LOR. We first construct language-only CoT data to enhance the model’s ability for textual spatial reasoning. As depicted in the upper part of Figure 3, given a spatial question and its ground-truth answer, we prompt a LVLM to generate a step-by-step reasoning trace that logically derives the correct answer. The reasoning is required to be concise yet explicit, ensuring that each step connects observed spatial relations (e.g., orientation or distance) to the final decision. Each trace ends with a definitive statement linking reasoning to the answer (e.g., “Therefore, the answer is A (the chair is closer)”). This dataset establishes a linguistic foundation for spatial reasoning without any geometric supervision.

CoT Data Construction—DTR. To incorporate explicit geometry, we extend the above process with detected 3D cues. For each spatial question, we either use available 3D annotations or estimate object centers (??, ??, ??) / 3D bounding boxes. These cues are then integrated into a structured reasoning chain that performs explicit spatial computation (e.g., distances, coordinate comparison, geometric rules) to derive the answer. As shown in the middle part in Figure 3, for every 2D region annotated in SPAR (Zhang et al., 2025a), we retrieve its 3D location by projecting all EmbodiedScan (Wang et al., 2024b) 3D object annotations into the image plane using the known camera parameters. The matched 3D annotation is then used as the ground-truth 3D coordinate. For multi-view questions, we select one frame as main perspective and align each object to the reference coordinate system. After prompting the expert model, each output is organized as “<detect> ... </detect><think> ... </think><answer> ... </answer>”. Unlike the language-only CoT, this version enforces mathematical formulation and deduction. Region tokens from the base spatial VLM’s region branch align text mentions with visual regions (mask/box), supporting accurate spatial grounding during both detection and reasoning.

Complex Spatial Task Generation. Beyond template-style spatial questions, we further generate complex spatial tasks involving navigation, object interaction, and layout reasoning, which aims to improve our model’s reasoning generalization in diverse spatial scenarios. Using multimodal scene-graph datasets that provide both visual and geometric annotations, we prompt LVLM to generate spatially grounded questions, identify relevant objects, and then produce the corresponding answers and detailed CoT traces. Each example forms a tuple of (image, question, coordinates, CoT, answer), extending reasoning coverage from simple spatial comparison to higher-level reasoning over structured 3D scenes. The data pipeline is shown in the bottom part in Figure 3.

Quality Control. Expert-generated CoT traces may contain hallucinations or invalid reasoning. We apply two-stage filtering: answer matching retains only samples whose conclusion matches the ground truth, and an LLM verifier then checks logical consistency of the rationale and correctness of intermediate spatial computation. We further spot-check sampled batches to verify annotation format and region–object alignment. Data proportions are reported in Appendix C.

Blended Supervision for Stable Cold-Start. Training exclusively on CoT data leads to rapid degeneration of the model’s general multimodal capabilities, including fundamental spatial skills and stable output formatting. We observe that without diverse supervision, the model tends to overfit to CoT-style reasoning and shows weaker cross-domain transfer. To mitigate this, we introduce a blended fine-tuning stage. First, since our spatial VLM backbone lacks grounding capability, we incorporate both 2D and 3D grounding data to build this competence. For 2D grounding, we use the RefCOCO (Kazemzadeh et al., 2014) dataset, where the model learns to predict 2D bounding boxes given natural language descriptions. For the region-to-3D supervision required by DTR, we integrate data from CA-1M, Omni3D, and OmniNOCS (Lazarow et al., 2024; Brazil et al., 2023; Krishnan et al., 2024b), enabling the model to output 3D bounding boxes—defined by object centers, dimensions, and orientations. To further enhance general spatial perception capabilities, we incorporate additional region-prompted spatial fine-tuning data. To further preserve linguistic and commonsense priors, we also blend general multimodal SFT data containing standard non-spatial Q&A pairs. Together, these sources form the complete cold-start dataset of roughly one million samples, used to fine-tune the model for two epochs before reinforcement learning.

## 3.4. Reinforcement Learning Stage

We train the model with GRPO, a rule-based reinforcement learning method well suited to multiple-choice supervision. Both SPAR and the OpenImages-derived dataset (described later) provide multiple-choice answers, enabling correctness to be used directly as the scalar accuracy reward.

Reward Design. Following standard practice in VLM RL, each rollout receives two rewards: a format reward and an accuracy reward. The format reward checks whether the output follows the required “think–answer” structure for LOR, and “detect–think–answer” structure for DTR. The accuracy reward is computed based on the task type. For multiple-choice questions, a rollout receives a positive score if the selected option matches the ground-truth label and zero otherwise. For filling questions, we use an exponentially smoothed relative error to calculate the reward:

$$
r _ {a c c} (x) = \exp \left(- 2 \cdot \frac {| x - x _ {\mathrm{gt}} |}{| x _ {\mathrm{gt}} | + \epsilon}\right), \tag {1}
$$

where ?? is the predicted value, ??gt is the ground-truth value, and ?? is a small constant for numerical stability.

DTR Detection Reward. Cold-start learning equips the model with a region-to-3D interface: when region tokens appear in the prompt, the model predicts the corresponding 3D center or bounding box using the learned pattern. RL preserves this output format and evaluates the predicted centers. For each DTR rollout, we extract the predicted 3D centers, compute their distances ?? to ground-truth annotations, and assign a discretized detection reward:

$$
r _ {\text { detect }} = \max \left(0, 1 - \left\lfloor \frac {d}{0 . 2} \right\rfloor \times 0. 2\right). \tag {2}
$$

Table 1: Results on spatial reasoning benchmarks. I.e., SPAR-Bench, where SR-3D as its base model, and Ours-LOR / Ours-DTR denote our final model evaluated with language-only and detect-then-reason inference, respectively. Our method also generalizes and improves the performance of the out-of-domain benchmarks, such as EmbSpatial and SAT. SAT contains global scene questions without region grounding, so we use LOR inference only. Bold numbers denote the best results in each column. Underlined numbers denote the second-best results. More detailed results on SPAR-Bench can be found in Appendix D.

<table><tr><td rowspan="2"></td><td colspan="4">SPAR-Bench</td><td rowspan="2">EmbSpatial</td><td rowspan="2">SAT</td></tr><tr><td>Low</td><td>Medium</td><td>High</td><td>Avg.</td></tr><tr><td colspan="7">General Models</td></tr><tr><td>InternVL2.5-8B (Chen et al., 2023)</td><td>21.7</td><td>31.1</td><td>36.3</td><td>29.7</td><td>59.8</td><td>57.3</td></tr><tr><td>LLaVA-OneVision-1.5-8B (Li et al., 2024)</td><td>31.9</td><td>26.0</td><td>42.4</td><td>35.5</td><td>67.2</td><td>64.0</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>17.5</td><td>29.5</td><td>41.8</td><td>30.2</td><td>70.4</td><td>62.0</td></tr><tr><td>Qwen3-VL-8B (Team, 2025)</td><td>35.7</td><td>30.0</td><td>46.7</td><td>39.6</td><td>79.0</td><td>69.3</td></tr><tr><td>SpatialRGPT (Cheng et al., 2024)</td><td>26.0</td><td>22.0</td><td>32.1</td><td>28.0</td><td>60.9</td><td>44.0</td></tr><tr><td>NVILA-8B-Lite (Base) (Liu et al., 2025c)</td><td>24.9</td><td>28.6</td><td>40.3</td><td>32.3</td><td>67.3</td><td>63.3</td></tr><tr><td colspan="7">Spatial Reasoning Models</td></tr><tr><td>Cosmos-Reason1-7B (Lin et al., 2025)</td><td>23.4</td><td>19.4</td><td>24.2</td><td>22.8</td><td>65.2</td><td>60.7</td></tr><tr><td>ViGoRL (Sarch et al., 2025)</td><td>25.0</td><td>18.1</td><td>18.6</td><td>21.1</td><td>71.8</td><td>61.3</td></tr><tr><td>SpatialLadder (Li et al., 2025)</td><td>25.8</td><td>33.2</td><td>42.6</td><td>34.4</td><td>59.8</td><td>16.7</td></tr><tr><td>SpaceR (Ouyang et al., 2025)</td><td>32.3</td><td>33.0</td><td>47.6</td><td>39.2</td><td>69.4</td><td>64.7</td></tr><tr><td>VST (Yang et al., 2025a)</td><td>53.3</td><td>25.4</td><td>53.7</td><td>48.9</td><td>70.4</td><td>69.3</td></tr><tr><td colspan="7">Ours</td></tr><tr><td>SR-3D (Base) (Cheng et al., 2025)</td><td>24.1</td><td>37.6</td><td>40.1</td><td>33.4</td><td>72.5</td><td>63.0</td></tr><tr><td>Ours-LOR</td><td>58.5</td><td>47.7</td><td>67.3</td><td>60.5</td><td>79.2</td><td>68.7</td></tr><tr><td>Ours-DTR</td><td>61.1</td><td>46.9</td><td>68.3</td><td>61.9</td><td>81.3</td><td>-</td></tr></table>

In multi-view scenarios, we identify which frame the model selects as a reference and calculate the reward with the object’s 3D ground-truth center point projected to that specific reference frame.

Online Filtering. We apply an online filtering mechanism similar to DAPO: rollout groups whose samples receive identical total reward are removed because they provide no relative advantage. The remaining groups are resampled to maintain the batch size. In this way, training samples have greater advantages, enabling more efficient model training and improving overall performance.

Training Data. The RL stage uses two data sources. (1) SPAR: single-view and multi-view questions provide region references and 2D/3D positions for accuracy and detection rewards. (2) OpenImages-derived data: to broaden spatial generalization beyond SPAR, we construct an additional dataset based on SRGPT data pipelines. For each OpenImages image, we build a 3D scene graph by combining segmentation masks with monocular depth lifting to approximate per-object point clouds, from which we estimate oriented 3D bounding boxes as detection supervision. We then randomly select two or more objects from each scene graph and generate multiple-choice spatial questions describing their geometric relationships with LLM. This expanded dataset supplies both global spatial reasoning signals and 3D supervision for DTR rewards, enabling RL to improve linguistic reasoning and geometry-aware detection behavior jointly.

## 4. Experiments

## 4.1. Experimental Setup

Training Data–Cold-Start. We train a single unified model from SR-3D that supports both reasoning modes (LOR and DTR). The Cold-Start stage is built from four components, totaling approximately 1M samples. Each component is summarized below:

• CoT–LOR. We construct 30k CoT samples following the LOR reasoning pathway. SPAR (Zhang et al., 2025a) is used as the primary source, providing 10k concise reasoning traces on single-view and multi-view data. The remaining

Table 2: Comparisons of LOR/DTR inference on representative SPAR-Bench subtasks for single-view and multi-view scenarios. We report accuracies on depth\_prediction\_oo (Depth), distance\_infer\_center\_oo (Distance), obj\_spatial\_relation\_oo (Relation), and spatial\_imagination\_oo (Imagination).

<table><tr><td rowspan="2"></td><td colspan="4">Single-View</td><td colspan="4">Multi-view</td></tr><tr><td>Depth</td><td>Distance</td><td>Relation</td><td>Imagination</td><td>Depth</td><td>Distance</td><td>Relation</td><td>Imagination</td></tr><tr><td>Ours-LOR</td><td>30.5</td><td>75.0</td><td>71.4</td><td>49.3</td><td>25.9</td><td>67.0</td><td>78.9</td><td>69.5</td></tr><tr><td>Ours-DTR</td><td>35.4</td><td>80.0</td><td>73.9</td><td>52.0</td><td>30.4</td><td>67.0</td><td>77.8</td><td>69.7</td></tr></table>

Table 3: Results on OOD benchmarks: BLINK (Spatial), RealWolrdQA, and CVBench.

<table><tr><td></td><td>BLINK(s)</td><td>RWQA</td><td>CVBench</td></tr><tr><td>SR-3D (base)</td><td>83.9</td><td>68.1</td><td>88.9</td></tr><tr><td>Ours-LOR (Full)</td><td>80.4</td><td>59.5</td><td>88.5</td></tr><tr><td>Ours-direct (Full)</td><td>87.4</td><td>64.6</td><td>88.1</td></tr></table>

20k are drawn for complex spatial task reasoning construction, including indoor CA-1M (Lazarow et al., 2024) and outdoor NuScenes (Caesar et al., 2020) scenes.

• CoT–DTR. We curate 10k CoT samples requiring explicit region-based detection followed by quantitative analysis and reasoning. These samples are also derived from SPAR, aligning with the DTR pathway used later in RL optimization.  
• Grounding Data. To support spatial localization, we include 3D grounding from Omni3D (Brazil et al., 2023), OmniNOCS (Krishnan et al., 2024a) and 2D grounding from RefCOCO (Kazemzadeh et al., 2014). These examples provide region-to-3D point supervision crucial for DTR.  
• Region-Prompted VQA. We incorporate region-aware QA pairs from SRGPT (Cheng et al., 2024), which help the model build localized spatial understanding tied to visual regions.  
• General-Purpose VQA. To maintain broad multimodal capability and prevent overfitting to spatial reasoning, we include non-spatial QA data from LLaVA-1.5 (Liu et al., 2024).

TrainingData–RL. The RL stage uses roughly 200k spatial questions spanning single-view and multi-view settings, in both multiple-choice and filling formats: ∼100k global questions (LOR) and ∼100k region-grounded questions with 2D/3D object coordinates (DTR), sourced from SPAR and an OpenImages-derived dataset (see Section 3.4).

Evaluation Benchmarks. Our model supports both LOR and DTR inference paradigms on the same checkpoint. (i) Region-Grounded benchmarks consist of questions tied to specific image regions, allowing us to evaluate both LOR and DTR inference. We use SPAR-Bench (Zhang et al., 2025a), which contains multiple-choice and filling questions on single-view and multi-view scenarios, covering 20 subtasks including distance, spatial relationships, and viewpoint changes. We also include EmbSpatial-Bench (Du et al., 2024), which specifically targets positional relationships in embodied environments. (ii) Global-Only benchmarks contain only image-level questions without region grounding, such as the dynamic spatial benchmark SAT (Ray et al., 2025); here we use LOR inference only. (iii) OOD benchmarks evaluate robustness under distribution shift. We include BLINK (Fu et al., 2024) for perception-heavy tasks, RealWorldQA (xAI, 2024) for real-world driving and everyday scenes, and CVBench (Tong et al., 2024) which focuses on spatial counting and relationships.

## 4.2. Main Results

Performance on Standard Benchmarks. We first evaluate our model on standard spatial benchmarks including SPAR-Bench, EmbSpatial, and SAT, as shown in Table 1 and Table 2. On SPAR-Bench, which features diverse question types (single/multi-image, multiple-choice/filling), our model achieves significant improvements over the base model. Specifically, Ours-DTR attains an average accuracy of 61.9, surpassing the base SR-3D (33.4) by a large margin (+28.5). Similarly, on EmbSpatial, which focuses on embodied positional relationships, Ours-DTR reaches 81.3 compared to 72.5 for the base. On the global benchmark SAT, where no region prompts are available, our model using LOR (68.7) maintains better than the base (63.0), showing that RL tuning preserves global spatial reasoning. Table 2 further details the performance on representative subtasks of SPAR-Bench. Across both single-view and multi-view settings, the geometry-aware DTR mode consistently outperforms LOR on quantitative tasks, such as Depth Prediction (+4.9 on Single-View) and Distance Inference <think>The table (red box) is to the left of the pedestal fan (blue box). The table is also positioned further back in the room compared to the pedestal fan, meaning it is farther away from the observer. The vertical positions of the table and the pedestal fan are relatively similar, so there is no clear above or below relationship. Therefore, the correct answer is 'left, , farther', which corresponds to option D.</think>

Spatial Relationship  
![](images/4be2ecde933ba3df9adcc118067f64d27ddf09e41cf10feaf8554f9ea1ea9b61.jpg)

<details>
<summary>natural_image</summary>

Interior view of a laboratory or lab space with equipment and a chair (no visible text or symbols)
</details>

How does the table (red bbox) compare in position to the pedestal fan (blue bbox) in the given image?  
A. left, below, closer B. right, above, farther C. right, above, closer D. left, , farther  
Answer: D

<answer>D</answer>

LOR

<detect>Find 2 relevant objects:

<3d\_box center=\"-0.77,0.5,3.68\">table (red bbox)</3d\_box>

<3d\_box center=\"0.97,0.47,2.21\">pedestal fan (blue bbox)</3d\_box></detect>

<think> I will analyze their positions using the provided 3D coordinates.

First, I'll compare their left-right positions. The table's x-coordinate is -0.77 and the fan's is 0.97. Since -0.77 < 0.97, the table is to the \*\*left\*\* of the fan.

Next, I'll compare their above-below positions. The table's y-coordinate is 0.5 and the fan's is 0.47. The difference is only 0.03, which is very small. Visually, both objects are on the floor, so their vertical difference is negligible, making this relationship indistinguishable.

Finally, I'll compare their farther-closer positions. The table's z-coordinate is 3.68 and the fan's is 2.21. Since 3.68 > 2.21, the table is \*\*farther\*\* away than the fan.

Combining these findings, the table is to the left of and farther than the fan, with no significant height difference. This corresponds to \"left, , farther\". So the answer is 'D'.</think>

<answer>D</answer>

DTR

Distance Prediction  
![](images/49ab3b582570a60c15d9c1808112f679ceb63b0ebbc1dfe2ef9526c0a82be668.jpg)

<details>
<summary>natural_image</summary>

Interior view of a conference room with tables, chairs, and carpet (no visible text or signage)
</details>

![](images/ffc91bff95bdfeaab0961efd0330d736458c6c5bb1518c8d54335e49b80bc4d8.jpg)  
How far away, in meters, is the center of chair (red point) from the center of chair (blue point) in the images? Calculate or judge based on the 3D center points of these objects. Provide only one number for your response.  
Answer: 2.9

<think>The red point marks a chair at the far end of the room, while the blue point marks a chair closer to the camera, near the center of the table. Considering the depth and distance between the two chairs, the distance between the red and blue points appears to be approximately 1.9 meters. So the answer is 1.9</think>

<answer>1.9</answer>

LOR

<detect>After aligning to the reference frame 0, find 2 relevant objects:

<3d\_box center=\"-1.19,-1.09,4.39\">chair (red point)</3d\_box>

<3d\_box center=\"0.39,-0.19,2.39\">chair (blue point)</3d\_box></detect>

<think>The goal is to calculate the distance between the center of the chair marked by the red point and the center of the chair marked by the blue point. All coordinates are provided relative to the camera in Frame 0. First, we identify the 3D coordinates for each object from the scene graph. The red point is located at \`center3d: [-1.19, -1.09, 4.39]\`, and the blue point is at \`center3d: [0.39, -0.19, 2.39]\`. To find the distance between these two points in 3D space, we use the Euclidean distance formula:

Distance = √ ((0.39 – (-1.19))^2 + (-0.19 - (-1.09))^2 + (2.39 - 4.39)^2) = √ ((1.58)^2 + (0.90)^2 + (-2.00)^2) = √(7.3024) = 2.70 meters.

So, the distance between the center of the chair (red point) and the center of the chair (blue point) is approximately 2.70 meters. </think>

<answer>2.7</answer>

DTR

Navigation Planning  
![](images/dffb5372be8f3ace2daf8a1f459c94a1fb82fd7f0919c64bb962ce7a0fcd198c.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern living room with white furniture, TV, and large windows (no visible text or symbols)
</details>

Outline a path from the right door of the house to the door of the left room.

<think>I looked at the image to find the right door and the left room door. The right door is on the far right, and the left room door is on the far left. I then considered the furniture in the room, like the sofas and coffee table, which are mostly in the middle and towards the back. To get from the right door to the left door, you would need to walk around the furniture. The path would likely start near the right door, move forward and slightly to the left to go around the sofas and coffee table, and then continue towards the left side of the room until you reach the left room door. The correct answer, therefore, is "Start near the right door, move forward and left to go around the sofas and coffee table, and then continue towards the left room door."</think>

<answer>Start near the right door, move forward and left to go around the sofas and coffee table, and then continue towards the left room door.</answer>

LOR

Figure 4: Visualization examples of our model. On the fundamental spatial question (spatial relationship and distance), we compare the reasoning paths of LOR and DTR. On the complex spatial task (navigtion), our model also demonstrates reasoning generalization capability.

(+5.0 on Single-View). This confirms that accessing explicit 3D coordinates enables more precise calculation compared to purely linguistic reasoning. For Spatial Relation and Imagination tasks, DTR remains highly competitive, often exceeding LOR, demonstrating that explicit detection benefits general spatial reasoning as well.

Generalization on OOD Benchmarks. We further assess the model’s robustness on out-of-distribution (OOD) benchmarks—BLINK(s), RealWorldQA, and CVBench—which involve unseen imagery and question styles (Table 3). Although trained on region-grounded spatial QA, the model maintains strong generalization under these shifts. While applying CoT reasoning to these distributions leads to lower performance, we find that the model maintains high accuracy when using “direct inference” (i.e., answering directly without CoT). For example, direct inference achieves 87.4% on BLINK(s) and 88.1% on CVBench, significantly surpassing the CoT mode. This result confirms that our training framework enhances complex spatial reasoning without catastrophic forgetting, preserving the model’s original strong capabilities for general spatial perception and QA tasks.

Table 4: Ablation on LOR and DTR training. We compare models trained with LOR-only, DTR-only, and mixed data. Columns indicate the inference mode used during evaluation (LOR/DTR).

<table><tr><td rowspan="2"></td><td colspan="2">SPAR-Bench</td><td colspan="2">EmbSpatial</td></tr><tr><td>-LOR</td><td>-DTR</td><td>-LOR</td><td>-DTR</td></tr><tr><td>Ours (LOR)</td><td>58.0</td><td>-</td><td>75.9</td><td>-</td></tr><tr><td>Ours (DTR)</td><td>-</td><td>57.2</td><td>-</td><td>71.4</td></tr><tr><td>Ours (Full)</td><td>58.7</td><td>60.8</td><td>77.6</td><td>78.8</td></tr></table>

Table 5: Ablation on DTR designs. We validate the impact of RL detection reward and region-to-3D mechanisms. All metrics are evaluated under DTR inference. SPAR-Det denotes 3D localization error on a validation set of SPAR.

<table><tr><td></td><td>SPAR-Bench</td><td>EmbSpatial</td><td>SPAR-Det</td></tr><tr><td>w/o detect reward</td><td>59.9</td><td>76.0</td><td>0.78</td></tr><tr><td>w/o region-to-3D</td><td>59.3</td><td>74.8</td><td>0.67</td></tr><tr><td>Ours</td><td>60.6</td><td>78.5</td><td>0.45</td></tr></table>

## 4.3. Qualitative Results

Figure 4 shows visualization examples of our model’s reasoning. In the first case of spatial relation task, our model can reason effectively through either linguistic deduction or explicit coordinate comparison. The second case involves a challenging multi-view distance estimation problem. LOR relies on visual estimation and fails to gauge the metric distance accurately across viewpoints. In contrast, DTR aligns the chair from the second view to the first view’s reference frame, localizes both objects within the same coordinate system, and computes the distance explicitly, yielding a precise answer. The final example involves multi-step planning for a complex navigation task. Our model first localizes the start and end points, then identifies obstacles along the path, and finally generates the correct route plan. This capability is primarily attributed to the CoT data for complex problems constructed from CA-1M and NuScenes during the cold-start phase, enabling the model to handle more sophisticated spatial tasks. More additional examples are provided in the Appendix E.

## 4.4. Ablation Analysis

Impact of Joint LOR & DTR Training. Table 4 investigates the benefits of unifying LOR and DTR training. We compare our full model against variants trained exclusively on LOR or DTR data. The results show that joint training not only supports both inference modes within a single model but also fosters mutual reinforcement: the model trained on both (Ours-Full) consistently outperforms the single-mode baselines across all metrics. For instance, LOR inference improves from 58.0 to 58.7 on SPAR-Bench when DTR data is added, suggesting that learning explicit geometry enhances the model’s underlying spatial representation. Conversely, DTR inference benefits from LOR training (improving from 57.2 to 60.8), likely because pure DTR training can lead to an over-reliance on quantitative calculation at the expense of qualitative spatial perception and reasoning. This confirms our hypothesis that LOR and DTR are complementary paradigms, and a unified model can effectively leverage both.

Ablation on DTR Designs. Table 5 validates the critical components of our DTR mechanism: the RL detection reward and the region-to-3D interface. Here, we evaluate all models using DTR inference and report 3D localization error on a validation set (SPAR-Det) consisting 400 instances. Removing the discrete detection reward leads to a significant drop in localization accuracy (error increases from 0.45 to 0.78) and a corresponding decline in reasoning performance, demonstrating that precise 3D grounding is essential for correct geometric deduction. Similarly, ablating the region-to-3D interface (i.e., predicting coordinates directly from text without visual region tokens) degrades performance, highlighting the significance of 2D position priors. These results confirm that both explicit supervision for intermediate detection steps and the region-to-3D mechanism are vital for the success of the detect-then-reason paradigm.

Impact of Cold-Start SFT and RL Phases. Table 6 disentangles the effects of the SFT and RL stages in LOR training. After cold-start SFT, SPAR-Bench performance increases due to the CoT data from SPAR, while metrics on unseen tasks like SAT and EmbSpatial decline. Adding the RL stage improves both reasoning and generalization, enabling the model to solve

Table 6: Effect of Cold-Start/RL stages. The ablation is conducted with LOR-only training on selection questions. “Cold-Start only” removes RL, “RL only” skips cold-start, and “Full pipeline” uses both stages. “SI”: single-iview, “MI”: multi-view.

<table><tr><td></td><td>SPAR(SI)</td><td>SPAR(MI)</td><td>SAT</td><td>EmbSpatial</td></tr><tr><td>SR-3D (Base)</td><td>38.97</td><td>41.49</td><td>63.00</td><td>72.50</td></tr><tr><td>Cold-Start only</td><td>56.53</td><td>45.89</td><td>62.67</td><td>65.66</td></tr><tr><td>RL only</td><td>62.34</td><td>64.22</td><td>65.33</td><td>76.75</td></tr><tr><td>Full pipeline</td><td>72.21</td><td>69.39</td><td>64.67</td><td>76.90</td></tr></table>

Table 7: Ablation of Cold-Start training data in LOR training setup.

<table><tr><td>CoTspar</td><td>CoTCA</td><td>MM-data</td><td>Region-data</td><td>SPAR(SI)</td><td>EmbSpatial</td></tr><tr><td>√</td><td>×</td><td>×</td><td>×</td><td>69.30</td><td>64.34</td></tr><tr><td>√</td><td>√</td><td>×</td><td>×</td><td>67.13</td><td>68.81</td></tr><tr><td>√</td><td>√</td><td>√</td><td>×</td><td>69.16</td><td>76.55</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>72.21</td><td>76.90</td></tr></table>

EmbSpatial and achieving the strongest results on SPAR and EmbSpatial. We also test an “RL only” variant that applies RL directly to the spatial base model. Although it improves metrics and even achieves the best SAT performance, its generated CoTs are often illogical or inconsistent with the answers, indicating that RL alone cannot bootstrap coherent reasoning (see Appendix E for examples).

Analysis of Cold-Start Data Components. In Table 7, each data group contributes positively to the final performance. Cold-starting with only SPAR CoT data yields strong SPAR-Bench results but poor generalization, including a low 64.34 on EmbSpatial and occasional chaotic reasoning on OOD tasks. Adding CoT data from other sources (e.g., CA-1M) and general-purpose multimodal data progressively improves generalization, leading to large gains on EmbSpatial. Incorporating region-related fine-tuning data further boosts performance and yields the best results across all benchmarks. Thus, diverse data sources during cold start are essential for strong generalization and spatial perception.

## 5. Conclusion

SR-REAL offers a novel approach to spatial reasoning in Vision–Language Models, demonstrating the effectiveness of reinforcement learning in enhancing both linguistic and geometric reasoning. Our framework integrates two complementary reasoning paths, LOR and DTR, leveraging region-to-3D grounding and discrete detection rewards to improve spatial accuracy. SR-REAL excels across a range of spatial benchmarks and achieves strong cross-domain generalization.

## A. Implementation Details

Stage 1: Cold-Start SFT. We fine-tune the SR-3D base model on the blended cold-start dataset (∼1M samples) for 2 epochs. Training uses a learning rate of $5 \times 1 0 ^ { - 6 }$ with cosine decay scheduling and a batch size of 128.

Stage 2: Reinforcement Learning. We apply GRPO-based RL for 200 steps on the ∼200k spatial RL dataset. The rollout batch size is 512 and the learning rate is $1 \times 1 0 ^ { - 6 }$ with cosine decay. Both stages are trained on 32 NVIDIA A100 GPUs.

## B. Preliminary knowledge of GRPO

In this work, we employ Group Relative Policy Optimization (GRPO) as our core reinforcement learning algorithm. GRPO is a policy optimization method that eliminates the need for a value function critic—common in algorithms like PPO—thereby reducing memory usage and computational overhead during training.

The fundamental principle of GRPO involves estimating the advantage of a sampled response by comparing its reward against a group of other responses generated from the same input query. Formally, for a given question-answer pair query $q ,$ the old behavior policy $\pi _ { \theta _ { \mathrm { o l d } } }$ samples a group of $G$ outputs $\{ o _ { i } \} _ { i = 1 } ^ { G }$ . The optimization process consists of the following key components:

Group-Relative Advantage Estimation. Instead of relying on a learned value function to predict the baseline, GRPO calculates the advantage $\hat { A } _ { i , t }$ for the ??-th response by normalizing its reward $r _ { i }$ with respect to the group’s statistics. The advantage for each token ?? in the response $o _ { i }$ is defined as:

$$
\hat {A} _ {i, t} = \frac {r _ {i} - \operatorname{mean} (\{r _ {i} \} _ {i = 1} ^ {G})}{\operatorname{std} (\{r _ {i} \} _ {i = 1} ^ {G})}, \tag {3}
$$

where $r _ { i }$ is the reward obtained for the ??-th output. This normalization effectively serves as a baseline, encouraging the model to reinforce outputs that perform better than the group average.

Objective Function. GRPO maximizes a surrogate objective that incorporates importance sampling ratios and the PPO-style clipping mechanism to ensure stable updates. Additionally, a Kullback-Leibler (KL) divergence penalty is added directly to the objective to prevent the trained policy $\pi \theta$ from deviating excessively from the reference policy $\pi _ { \mathrm { r e f } } .$ . The full objective function is given by:

$$
\mathcal {J} _ {\mathrm{GRPO}} (\theta) = \mathbb {E} _ {q \sim \mathcal {D}, \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (\cdot | q)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| o _ {i} |} \sum_ {t = 1} ^ {| o _ {i} |} \right.
$$

$$
\left(\min \left(\rho_ {i, t} (\theta) \hat {A} _ {i, t}, \operatorname{clip} \left(\rho_ {i, t} (\theta), 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {i, t}\right) \right. \tag {4}
$$

$$
\left. \left. - \beta D _ {\mathrm{KL}} (\pi_ {\theta} | | \pi_ {\mathrm{ref}})\right) \right],
$$

where $\rho _ { i , t } ( \boldsymbol { \theta } )$ is the probability ratio between the current and old policies:

$$
\rho_ {i, t} (\theta) = \frac {\pi_ {\theta} (o _ {i , t} \mid q , o _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (o _ {i , t} \mid q , o _ {i , <   t})}. \tag {5}
$$

Here, ?? is the clipping parameter, $\beta$ is the coefficient for the KL penalty, and $D _ { \mathrm { K I } }$ measures the divergence between the current policy and the reference policy at the token level. This formulation allows GRPO to efficiently optimize the policy using group-based relative feedback without maintaining a separate critic model.

## C. CoT Data Construction

CoT Data Composition. Figure S1 illustrates the task distribution of our CoT instruction data. We construct approximately 10k LOR samples and 10k DTR samples sourced from the SPAR dataset, covering fundamental spatial perception tasks including depth estimation, distance prediction, object spatial relations, spatial imagination, and position matching. In addition, we construct 20k complex spatial task CoT samples drawn from the CA-1M and NuScenes datasets, targeting higher-level reasoning scenarios such as navigation, object manipulation, and spatial planning. Together, these three components provide balanced coverage across both basic geometric reasoning and complex scene-level inference.

![](images/40fc8a08e5e1d377ca255b8acaf8b6582b04262e80adf82b8a83e40e8716a9f3.jpg)  
Figure S1: Task distribution of CoT instruction data. The LOR and DTR subsets (10k each) are sourced from SPAR and cover fundamental spatial tasks including depth, distance, spatial relations, spatial imagination, and position matching. The complex task subset (20k) is drawn from CA-1M and NuScenes and covers higher-level scenarios such as navigation, object manipulation, and spatial planning.

Table S1: Detailed results on SPAR-Bench. We report scores of 20 dimensions, together with the overall average.

<table><tr><td>Model</td><td>camera_motion_infer</td><td>depth_prediction_oc</td><td>depth_prediction_oc_mv</td><td>depth_prediction_oo</td><td>depth_prediction_oo_mv</td><td>distance_infer_center_oo</td><td>distance_infer_center_oo_mv</td><td>distance_prediction_oc</td><td>distance_prediction_oc_mv</td><td>distance_prediction_oo</td><td>distance_prediction_oo_mv</td><td>obj_spatial_relation_oc_mv</td><td>obj_spatial_relation_oo</td><td>obj_spatial_relation_oo_mv</td><td>position_matching</td><td>spatial_imagination_oc</td><td>spatial_imagination_oc_mv</td><td>spatial_imagination_oo</td><td>spatial_imagination_oo_mv</td><td>view_change_infer</td><td>Avg.</td></tr><tr><td>Gemini-2.5-Pro</td><td>28.5</td><td>56.1</td><td>42.7</td><td>18.3</td><td>16.2</td><td>73.2</td><td>64.3</td><td>49.5</td><td>53.3</td><td>44.8</td><td>38.0</td><td>52.5</td><td>71.2</td><td>74.8</td><td>35.6</td><td>39.5</td><td>43.0</td><td>30.8</td><td>65.0</td><td>15.6</td><td>45.4</td></tr><tr><td>InternVL2.5-8B Chen et al. (2023)</td><td>26.2</td><td>26.2</td><td>27.2</td><td>12.1</td><td>15.1</td><td>57.6</td><td>52.7</td><td>24.8</td><td>25.2</td><td>22.3</td><td>21.4</td><td>39.5</td><td>35.7</td><td>30.2</td><td>57.0</td><td>25.0</td><td>29.7</td><td>25.5</td><td>30.8</td><td>10.7</td><td>29.7</td></tr><tr><td>LLaVA-OneVision-1.5-8B Li et al. (2024)</td><td>29.8</td><td>41.8</td><td>35.1</td><td>19.1</td><td>16.2</td><td>62.4</td><td>58.9</td><td>43.6</td><td>42.1</td><td>30.7</td><td>25.4</td><td>43.8</td><td>45.1</td><td>48.2</td><td>43.5</td><td>30.1</td><td>34.3</td><td>22.8</td><td>35.0</td><td>5.1</td><td>35.5</td></tr><tr><td>Qwen3-VL-8B Team (2025)</td><td>25.2</td><td>51.4</td><td>54.7</td><td>17.4</td><td>17.5</td><td>70.3</td><td>67.9</td><td>22.8</td><td>37.2</td><td>50.5</td><td>38.9</td><td>50.8</td><td>63.7</td><td>40.4</td><td>48.6</td><td>28.2</td><td>32.6</td><td>34.4</td><td>31.7</td><td>16.4</td><td>39.6</td></tr><tr><td>NVILA-8B-Lite Liu et al. (2025c)</td><td>25.8</td><td>27.4</td><td>33.9</td><td>19.8</td><td>17.2</td><td>60.0</td><td>58.3</td><td>27.6</td><td>26.9</td><td>24.0</td><td>23.5</td><td>29.2</td><td>39.8</td><td>49.0</td><td>53.4</td><td>25.5</td><td>33.7</td><td>23.5</td><td>44.5</td><td>7.1</td><td>32.3</td></tr><tr><td>SpatialRGPT-VILA1.5-8B Cheng et al. (2024)</td><td>26.2</td><td>29.5</td><td>29.3</td><td>17.3</td><td>17.1</td><td>51.8</td><td>53.3</td><td>23.4</td><td>29.3</td><td>32.1</td><td>32.0</td><td>39.8</td><td>18.4</td><td>27.4</td><td>26.2</td><td>27.7</td><td>27.9</td><td>17.5</td><td>24.9</td><td>13.6</td><td>28.0</td></tr><tr><td>RynnBrain Dang et al. (2026)</td><td>30.5</td><td>70.3</td><td>55.5</td><td>11.3</td><td>14.2</td><td>82.4</td><td>81.8</td><td>68.2</td><td>62.7</td><td>51.1</td><td>36.5</td><td>44.2</td><td>63.7</td><td>42.9</td><td>53.9</td><td>34.1</td><td>36.0</td><td>29.1</td><td>33.9</td><td>10.6</td><td>45.4</td></tr><tr><td>ViGoRL Sarch et al. (2025)</td><td>1.2</td><td>40.4</td><td>30.0</td><td>17.7</td><td>17.7</td><td>60.0</td><td>31.5</td><td>34.8</td><td>32.1</td><td>13.3</td><td>12.3</td><td>5.5</td><td>29.9</td><td>5.3</td><td>34.1</td><td>16.9</td><td>0.0</td><td>20.2</td><td>2.2</td><td>19.1</td><td>21.1</td></tr><tr><td>SpaceR Ouyang et al. (2025)</td><td>40.0</td><td>39.9</td><td>40.7</td><td>17.9</td><td>15.4</td><td>64.7</td><td>55.4</td><td>45.4</td><td>41.7</td><td>28.7</td><td>28.8</td><td>58.8</td><td>54.1</td><td>50.4</td><td>39.7</td><td>32.8</td><td>41.6</td><td>27.2</td><td>40.9</td><td>19.5</td><td>39.2</td></tr><tr><td>VST Yang et al. (2025a)</td><td>34.0</td><td>72.5</td><td>54.8</td><td>33.4</td><td>22.9</td><td>88.2</td><td>82.4</td><td>73.0</td><td>66.3</td><td>63.3</td><td>37.6</td><td>49.0</td><td>66.8</td><td>47.4</td><td>31.6</td><td>39.8</td><td>43.3</td><td>30.1</td><td>36.7</td><td>10.7</td><td>48.9</td></tr><tr><td>SR-3D (Base) Cheng et al. (2025)</td><td>30.8</td><td>25.6</td><td>36.5</td><td>15.5</td><td>12.3</td><td>56.2</td><td>56.8</td><td>19.8</td><td>26.8</td><td>28.2</td><td>32.0</td><td>46.8</td><td>45.3</td><td>42.1</td><td>54.5</td><td>28.2</td><td>27.0</td><td>25.2</td><td>32.2</td><td>27.9</td><td>33.4</td></tr><tr><td>Ours-LOR</td><td>43.8</td><td>80.4</td><td>69.2</td><td>30.5</td><td>25.9</td><td>75.0</td><td>67.0</td><td>78.1</td><td>70.4</td><td>63.6</td><td>49.7</td><td>80.5</td><td>71.4</td><td>78.9</td><td>70.5</td><td>55.6</td><td>54.4</td><td>49.3</td><td>69.5</td><td>29.2</td><td>60.5</td></tr><tr><td>Ours-DTR</td><td>43.8</td><td>81.1</td><td>69.1</td><td>35.4</td><td>30.4</td><td>80.0</td><td>67.0</td><td>79.0</td><td>72.5</td><td>65.9</td><td>55.2</td><td>80.5</td><td>73.9</td><td>77.8</td><td>68.2</td><td>54.8</td><td>55.5</td><td>52.0</td><td>69.7</td><td>29.1</td><td>61.9</td></tr></table>

Prompts. Our model supports two distinct inference pathways, where the specific reasoning mode is determined by the input prompt. The system prompts corresponding to Language-Only Reasoning (LOR) and Detect-Then-Reason (DTR) are presented in Table S4. In contrast to LOR, the DTR paradigm explicitly instructs the model to output 3D detection information enclosed within a <detect> block prior to thinking. Furthermore, Tables S5, S6, and S7 illustrate the system prompts used to query the LVLM during the construction of LOR CoT data, DTR CoT data, and complex spatial reasoning scenarios, respectively.

## D. More Results

Detailed results on SPAR-Bench. Table S1 reports per-dimension scores across all 20 subtasks on SPAR-Bench, complementing the aggregated results presented in the main paper. Ours-DTR achieves the best overall average (61.9) and leads on most depth- and distance-related dimensions, consistent with its explicit 3D grounding capability. Ours-LOR performs competitively across relation and spatial imagination tasks. Both modes substantially outperform the SR-3D base model across nearly all dimensions.

Training mechanisms during RL. In Table S2, we analyze the impact of standard GRPO enhancement mechanisms—specifically the online filter (as detailed in the main text) and KL coefficient decay—on reinforcement learning for spatial tasks. KL coefficient decay targets the KL divergence term in the GRPO objective, which acts as a constraint to keep the updated policy close to the reference model. During training, we anneal this coefficient following a cosine decay schedule. Comparing Rows 2 and 3, we observe that incorporating the online filter not only boosts the model’s final performance but also significantly enhances training efficiency: the model converges to the reported results in just 120 steps, whereas standard RL requires 300 steps. Furthermore, a comparison of the last two rows reveals that while KL coefficient decay leads to a performance drop on SPAR-BENCH, it yields improvements on benchmarks that diverge from the training distribution, such as SAT and EmbSpatial, thereby enhancing generalization.

Grounding data. Table S3 illustrates the critical role of grounding data during the cold-start phase. As shown in the second row, the absence of auxiliary grounding supervision leads to inaccurate localization, resulting in performance degradation across all benchmarks under DTR inference. This decline is particularly pronounced on EmbSpatial and CVBench. However, the impact on SPAR-Bench is relatively minor; this is because the cold-start dataset already includes DTR CoT samples derived specifically from SPAR, which provide implicit localization guidance even without separate grounding data.

## E. More Visualization

Qualitative examples. Figure S2 presents additional qualitative examples demonstrating the model’s capability across both reasoning pathways. The model yields accurate results under both LOR and DTR paradigms. In the first example on distance measurement, LOR incorrectly estimates distance due to inaccurate object localization, while DTR accurately computes the distance through precise spatial coordinate detection. In the last example, which involves a manipulation task—specifically, identifying obstacles when moving a large pot from the stove to the sink—the model correctly infers that the manipulation trajectory intersects with the intermediate object (the countertop), thereby deriving the correct answer.

Failure cases when inappropriate cold-start. Tables 6 and 7 in the main text demonstrate the critical impact of the cold-start phase on the model’s final performance, highlighting that diverse cold-start data facilitates more stable reasoning on out-of-domain (OOD) questions. Table S8 provides a detailed breakdown of failure modes. When RL is applied directly without a cold-start phase, the model frequently exhibits reasoning-answer inconsistency: it may select the correct option, yet the intermediate Chain-of-Thought (CoT) process is factually incorrect or logically misaligned with the answer. Furthermore, when the cold-start is restricted to CoT data based on SPAR, followed by RL on SPAR dataset, the model maintains accurate reasoning on in-domain queries but sometimes produces chaotic or incoherent CoT on OOD tasks. These findings underscore that incorporating a diverse mixture of data during both the cold-start and RL stages is essential for improving reasoning generalization.

Table S2: Ablation of different mechanisms during LOR training. ‘Filter’ indicates online filtering, and ‘KLcos’ means the coefficient of the KL constraint term in GRPO undergoes cosine decay.

<table><tr><td></td><td>SPAR (SI)</td><td>SPAR (MI)</td><td>SAT</td><td>EmbSpatial</td></tr><tr><td>Cold-Start</td><td>56.53</td><td>45.89</td><td>62.67</td><td>65.66</td></tr><tr><td>RL</td><td>69.88</td><td>66.96</td><td>64</td><td>76.2</td></tr><tr><td>RL+filter</td><td>72.21</td><td>69.39</td><td>64.67</td><td>76.90</td></tr><tr><td>RL+filter+KLcos</td><td>71.48</td><td>68.35</td><td>66.67</td><td>77.66</td></tr></table>

Table S3: Ablation of Grounding data in cold-start for DTR inference.

<table><tr><td></td><td>SPAR (SI)</td><td>EmbSpatial</td><td>CVBench (3D)</td></tr><tr><td>Cold-Start</td><td>59.58</td><td>68.54</td><td>90.08</td></tr><tr><td>w/o Ground-data</td><td>59.36</td><td>66.32</td><td>83.58</td></tr></table>

## F. Limitations

While SR-ReaL demonstrates strong spatial reasoning performance, several limitations remain. First, our framework is built upon the SR-3D spatial VLM, which requires depth maps and camera intrinsics/extrinsics at inference time. This limits <think>I looked at the image and the object information. The question asks which object is the farthest away. I checked the positions of the desk, chair, dresser, and pillow. The desk is on the left side of the image, the chair is in the middle, and the dresser and pillow are on the right side. By looking at the image, the desk is clearly the most distant object from the camera's viewpoint. The chair and dresser are closer, and the pillow is the closest. Therefore, the desk is the farthest object. So the answer is A.</think> LOR

Distance Calculation  
![](images/2aa6eccd3049279c9cf47e9317552a851d9fc7108a32a3bab064ca3923f90c5e.jpg)

<details>
<summary>natural_image</summary>

Purple sleep bag resting on a desk with books and a small dog, no visible text or symbols
</details>

Which object from the list is situated at the greatest distance from your point of view within the image? Select from the following choices. A. Desk B. chair C. Dresser D. pillow Answer: D

<answer>A</answer>

LOR

<detect>Find 4 relevant objects:

<3d\_box center=\"-0.83,0.3,1.78\">desk</3d\_box><3d\_box center=\"-0.21,0.41,1.61\">chair</3d\_box>

<3d\_box center=\"0.23,0.47,1.41\">dresser</3d\_box><3d\_box center=\"-0.83,-0.33,1.91\">pillow</3d\_box></detect>

<think>The observer's position is the origin of the coordinate system, which is at \`[0, 0, 0]\`. The distance from the observer to an object is determined by the magnitude of its position vector.

First, we identify the position vectors for each object from the scene graph:\n- Desk: \`P\_desk = [-0.83, 0.3, 1.78]\` …

Next, we calculate the squared distance from the origin to each object's position vector:

\- Desk: \`(-0.83)^2 + (0.3)^2 + (1.78)^2 = 0.6889 + 0.09 + 3.1784 = 4.0573\`

\- Pillow: \`(-0.83)^2 + (-0.33)^2 + (1.91)^2 = 0.6889 + 0.1089 + 3.6481 = 4.4459\`

Comparing the squared distances, the largest value is 4.4459, which corresponds to the pillow's position. Therefore, the pillow is the farthest object.

So the answer is 'D'.</think>

<answer>D</answer>

DTR

Distance Comparison  
![](images/4b4f9faad3c60483aa3c582711abda313943b13f9d06d1d8e695375d3ddd8b14.jpg)

<details>
<summary>natural_image</summary>

Interior scene with a desk, chair, lamp, and scattered items (no visible text or symbols)
</details>

Analyze the image and conclude whether desk (green point) or suitcase (blue point) is closer to trash can (red point). Calculate or judge based on the 3D center points of these objects.

A. suitcase (blue point)

B. desk (green point)

Answer: A

<think>The red and blue points are located on the floor next to each other. The red point is on a trash can, and the blue point is on a suitcase. The green point is on a desk, which is positioned against the wall behind the trash can and suitcase. Visually, the trash can and suitcase are very close to each other, while the desk is further away. Therefore, the suitcase (blue point) is closer to the trash can (red point). So the answer is A.</think>

<answer>A</answer>

LOR

<detect>Find 3 relevant objects:

<3d\_box center=\"0.27,-0.11,2.31\">trash can (red point)</3d\_box>

<3d\_box center=\"-0.6,-0.16,2.02\">desk (green point)</3d\_box>

<3d\_box center=\"0.96,-0.23,2.32\">suitcase (blue point)</3d\_box></detect>

<think>The user wants to determine whether the desk (green point) or the suitcase (blue point) is closer to the trash can (red point). This can be solved by calculating the 3D Euclidean distance between the red and green points, and between the red and blue points, and then comparing these distances.

The coordinates are:- Red point (trash can): P\_red = [0.27, -0.11, 2.31] …

First, we calculate the squared distance between the red and green points:

Distance(red, green)^2 = (0.27 - (-0.6))^2 + (-0.11 - (-0.16))\^2 + (2.31 - 2.02)^2 = 0.7569 + 0.0025 + 0.0841 = 0.8435

Next, we calculate the squared distance between the red and blue points:

Distance(red, blue)^2 = (0.27 - 0.96)^2 + (-0.11 - (-0.23))^2 + (2.31 - 2.32)^2 = 0.4761 + 0.0144 + 0.0001 = 0.4906

Comparing the two distances, 0.4906 is less than 0.8435. Therefore, the suitcase (blue point) is closer to the trash can (red point). So the answer is 'A'.</think>

<answer>A</answer>

DTR

Manipulation  
![](images/c40145b59760bfd4c4932acb3711efc989da1b87f72af7c5725951c83e86ac4c.jpg)

<details>
<summary>natural_image</summary>

Interior view of a kitchen with open shelves, mugs, and a green tray (no visible text or symbols)
</details>

Moving a large pot from the stove to the sink requires navigating around which immediate major obstacle?

<think>The pot is on the stove, which is located to the left of the sink. To move the pot from the stove to the sink, you would need to move it across the countertop, which is the surface between the stove and the sink. Therefore, the countertop is the main obstacle you would encounter.</think>

<answer> the countertop </answer>

LOR

Figure S2: More visualization cases of SR-REAL.

Table S4: The system prompts for two reasoning paths of SR-REAL.

<table><tr><td>Language-Only Reasoning:You are a helpful assistant. The user asks a question, and then you solve it.Please first think deeply about the question based on the given image, and then provide the final answer. The reasoning process and answer are enclosed within &lt;think&gt; &lt;/think&gt; and &lt;answer&gt; &lt;answer here &lt;answer&gt; .Question: {Question}</td></tr><tr><td>Detect-then-Reason:You are a helpful assistant. The user asks a question, and then you solve it.Please first detect 3D centers of relevant objects, then think deeply about the question based on the given image, and finally provide the answer. The detection, reasoning and answer are enclosed within &lt;detect&gt; &lt;/detect&gt;, &lt;think&gt; &lt;/think&gt; and &lt;answer&gt; &lt;/answer&gt; tags, respectively, i.e., &lt;detect&gt; detection here &lt;/detect&gt;, &lt;think&gt; reasoning process here &lt;answer&gt; answer here &lt;answer&gt;.Question: {Question}</td></tr></table>

Table S5: The prompt for LVLM to produce LOR CoT.

<table><tr><td># ROLE: Spatial Reasoning Chain-of-thinking GeneratorGiven a multiple-choice spatial reasoning question, a scene image, and the correct answer option, generate a **step-by-step reasoning chain** that logically derives the correct choice. Your output must demonstrate precise spatial reasoning based on the provided scene context.</td></tr><tr><td>## INPUT CONTEXT:You will receive the following:1. **Image:** An image depicting the 3D scene.2. **Question:** A multiple-choice question about the image.3. **Answer:** The ground truth answer index to the question. (e.g., &quot;A&quot;, &quot;B&quot;, &quot;C&quot;, or &quot;D&quot;)</td></tr><tr><td>## TASK:Your primary task is to analyze 3D visual scenes within the image, understand the &#x27;Question&#x27; and &#x27;Answer&#x27; accurately, then generate chain-of-thinking for how to get the correct answer.* Use the &#x27;Image&#x27; to understand scene context, object appearances, and spatial relationship.* Use the &#x27;Answer&#x27; to validate your reasoning process, your reasoning must be able to arrive at the correct answer.</td></tr><tr><td>## GUIDELINES &amp; CONSTRAINTS:* The reasoning process must be consistent with the spatial information in the image.* The reasoning process must clearly solve the question, and derive the correct answer.* The word number of reasoning process should be between 100 and 200.* No need for explicit output of step 1, step 2 ...</td></tr><tr><td>## A REFERENCE FOR REASONING STRUCTURE1. Scene interpretation2. Analysis: Position (horizontal/vertical/depth), relationship, distance ...3. Inference: solve the question, describe the thinking process4. Conclusion: matching correct answer</td></tr><tr><td>## OUTPUT FORMAT:Provide the output in the string format (the generated reasoning process).</td></tr></table>

applicability to settings where such geometric metadata is unavailable, such as unconstrained in-the-wild images without depth sensors. Second, the DTR path relies on region tokens derived from 2D bounding box or mask annotations. When such region annotations are absent—as in global benchmarks like SAT—DTR cannot be applied, and the model falls back to LOR. Automatically generating reliable region proposals for arbitrary queries remains an open challenge. Third, despite gains on OOD benchmarks, we observe that applying chain-of-thought reasoning to perception-heavy tasks (e.g., BLINK, RealWorldQA) can hurt performance compared to direct inference. This suggests the model has not fully learned when to engage multi-step reasoning versus respond directly, an important direction for future work. Finally, the cold-start data construction relies on a proprietary LVLM (Gemini-2.5-Pro) for CoT generation, introducing a dependency on external APIs and potential quality variance across reasoning traces.

## G. Broader Impact

Positive impacts. Improving spatial reasoning in vision-language models has broad beneficial applications. Enhanced spatial understanding supports embodied AI systems, assistive robotics, autonomous driving, and augmented reality, all of which can improve accessibility and quality of life. More capable spatial reasoning may also accelerate progress in scientific domains that require interpreting 3D spatial data, such as medical imaging analysis and remote sensing.

Potential negative impacts. As with general-purpose vision-language models, improvements in spatial understanding could be misused in surveillance systems or automated physical-space monitoring without appropriate consent. We encourage responsible deployment and advocate for clear usage guidelines when applying spatially-aware models in sensitive or privacy-sensitive environments. The reliance on proprietary data generation pipelines may also concentrate capability development among well-resourced organizations, potentially widening access gaps.

## Table S6: The prompt for LVLM to produce DTR CoT.

\# ROLE: Spatial Reasoning Chain-of-thinking Generator

Given a multiple-choice spatial reasoning question, a scene image, the correct answer option, and center 3D coordinates of relevant objects, generate a \*\*step-by-step reasoning chain\*\* that logically derives the correct choice. Your output must demonstrate precise spatial reasoning based on the provided scene context and object positions.

\## INPUT CONTEXT:

You will receive the following:

1. \*\*Image:\*\* An image depicting the 3D scene.

2. \*\*Question:\*\* A multiple-choice question about the image. The qustion is about some objects annotated by bounding boxes or points in the image.

3. \*\*Answer:\*\* The ground truth answer index to the question. (e.g., "A", "B", "C", or "D")

4. \*\*Scene Graph:\*\* A dictionary containing center 3D coordinates of object identifiers in the format: {’red point’: [-0.1, 0.2, 1.4]}

\## COORDINATE SYSTEM (Camera Coordinates):

\* \*\*+x axis:\*\* Left to Right

\* \*\*+y axis:\*\* Top to Bottom

\* \*\*+z axis:\*\* Near to Far (depth)

\* Note that the x y z coordinate is from the perspective of camera, if the question changes the observer to a different position and direction, the right, above, front of observer can not correspond with original x y z.

\## TASK:

Your primary task is to analyze 3D visual scenes within the image, understand the ‘Question‘ and ‘Answer‘ accurately, combine center positions of objects, then generate chain-of-thinking for how to get the correct answer.

\* Use the ‘Image‘ to understand scene context, object appearances, and spatial relationship.

\* Use the ‘Scene Graph‘ to get precise 3D positions of objects mentioned in the question. The scene graph is about annotations (bbox or point), and you need to match the corresponding objects in the question.

\* Use the ‘Answer‘ to validate your reasoning process, your reasoning must be able to arrive at the correct answer.

\## GUIDELINES & CONSTRAINTS:

\* The reasoning process must be consistent with the spatial information in the image.

\* The reasoning process must clearly solve the question, and derive the correct answer.

\* The word number of reasoning process should be between 100 and 200.

\* Keep the mathematical calculations brief, no need to output too many formulas (<3).

\* Mainly use the visual content for relationship judgement, with 3D coordinate calculations as supplementary or for verification.

\* No need for explicit output of step 1, step 2 ... or 1. 2. ...

\## A REFERENCE FOR REASONING STRUCTURE

1. Scene interpretation

2. Analysis: Position (horizontal/vertical/depth), relationship, distance ..

3. Computaion: calculate or compare the 3D coordinates to support spatial reasoning

4. Inference: solve the question, describe the thinking process

5. Conclusion: matching correct answer, e.g., "So the answer is ’A’."

\## OUTPUT FORMAT:

Provide the reasoning process in the string format: \* (String) The generated explanation of how the answer was derived, referencing image context and 3D coordinates of objects. The word number of reasoning process should be less than 200.

## Table S7: The prompt for LVLM to generate complex spatial task questions.

\# ROLE: Spatial Reasoning Evaluator Question Generator

\## PERSONA:

You are an expert evaluator specializing in designing challenging spatial reasoning tasks for advanced AI models. You are creative, meticulous, and focused on probing deep spatial understanding beyond simple object recognition.

\## INPUT:

You will receive an image or a video.

\## TASK:

Generate exactly 6 highly challenging spatial reasoning questions based on the provided image or video. These questions are intended to evaluate the capabilities of a state-of-the-art vision-language model. The questions must necessitate multi-step reasoning, inference about implicit spatial relationships, object interactions, potential future states, or navigation planning within the scene.

\## QUESTION CATEGORIES & DISTRIBUTION:

Generate questions covering the following categories. Unless a specific type is requested by the user later, adhere to this distribution:

\* \*Example Concept:\* "Outline a path from [Object A] to [Object B], staying only on [Surface Type] and going around [Obstacle]."

\* \*Example Concept:\* "If attempting to move from [Location X] to [Location Y] following [Simple Rule], what is the first major obstacle encountered?"

\* \*\*Manipulation (2 questions):\*\* Focus on object affordances, interaction plausibility, planning sequences of actions for rearrangement, or predicting outcomes of physical interactions.

\* \*Example Concept:\* "Could the [Object C] realistically be picked up with one hand and placed securely on top of [Object D]?"

\* \*Example Concept:\* "Describe the steps to rearrange [Objects E, F, G] from their current positions to form a [Target Configuration]."

\* \*Example Concept:\* "Where is the most likely place to find a [Common Object Type] in this scene, and is it currently visible?"

\* \*\*Scene Arrangement/Layout Reasoning (1 question):\*\* Focus on the plausibility of object placements, spatial relationships between objects, or understanding the functional layout of the scene.

\* \*Example Concept:\* "Is the placement of [Object H] relative to [Object I] typical for this type of environment? Why or why not?"

\* \*Example Concept:\* "If you added a [New Object], where could it fit without disturbing the main function of the area near [Existing Object J]?"

\## CONSTRAINTS:

1. \*\*Deep Reasoning Required:\*\* Questions MUST demand more than simple identification or direct observation. They should require inferring relationships, predicting outcomes, or planning sequences.

2. \*\*Visual Grounding Essential:\*\* Questions MUST require information \*directly\* from the image/video. They cannot be answerable using only common sense or external knowledge \*without\* visual context.

3. \*\*Integration of Knowledge:\*\* Questions SHOULD ideally require the model to integrate visual information with general world knowledge (e.g., physics, object properties, typical layouts) to arrive at the answer.

4. \*\*3D Spatial Computation Necessary:\*\* Questions MUST require reasoning about 3D spatial attributes (coordinates, dimensions, orientation) and involve numerical calculations (distances, angles, volumetric comparisons) derived from the visual data.

5. \*\*Minimal Explicit Information:\*\* Phrase questions concisely, providing only necessary context. Avoid giving away information that the evaluated model should infer from the visual input.

6. \*\*Conciseness:\*\* Each question must be a maximum of 25 words. (Slightly increased from 20 for flexibility).

7. \*\*Creativity:\*\* Strive for novel and non-obvious questions that genuinely test advanced spatial understanding.

\## INTERNAL PROCESS GUIDELINE:

\* To aid in generating complex questions, you may first internally generate a detailed description or caption of the image/video, focusing on objects, spatial relationships, potential interactions, and environmental context. Do NOT include this caption in your final output.

\## OUTPUT FORMAT:

\* Provide the 6 questions as a numbered list.

\* Indicate the category for each question in parentheses, e.g., ‘1. Question text (Navigation)‘

Table S8: The failure cases when inappropriate cold-start.

## Failure cases when RL only:

Question: What changes in the observer’s perception of the spatial relationship between monitor (red bbox) and pillow (yellow bbox) when observer relocates to pillow (green bbox) and face window (blue bbox)? Calculate or judge based on the 3D center points of these objects. Please use the world coordinate system to determine the up and down position relationship of objects. Pick the right response from the available choices.

A. right, above, closer  
B. right, , farther  
C. left, above,  
D. , above, closer

Response: <think>The observer’s perspective changes as they move from the pillow to the window. Initially, the monitor is to the right of the pillow and below it. When the observer moves to the window, the monitor is now below the observer’s current position, making it closer to the observer. Therefore, the observer’s perception of the spatial relationship between the monitor and the pillow changes to right, above, closer.</think> <answer>B</answer>

## Failure cases in out-of-domain questions when cold-start with only SPAR cot:

question: Which motion direction should the spoon be moved in order to slice some cheese? Choices: A. orange. B. red. C. blue. D. light blue. Please answer directly with only the letter of the correct option and nothing else.

Response: <think>The red arrow points towards the lid of the container, indicating the direction of<think>The red arrow points towards the lid, suggesting the direction of<think>The red arrow<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th <th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th<th

## References

Junjie Bai et al. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. URL https://arxiv.org/abs/2308.12966. 3  
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Ming-Hsuan Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. CoRR, abs/2502.13923, 2025. 8  
Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3d: A large benchmark and model for 3d object detection in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. URL https://github.com/facebookresearch/omni3d. Code and dataset release: facebookresearch/omni3d. arXiv:2207.10660. 7, 9  
Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11618–11628, 2020. doi: 10.1109/CVPR42600.2020.01164. URL https://openaccess.thecvf.com/content\_CVPR\_2020/html/Caesar\_ nuScenes\_A\_Multimodal\_Dataset\_for\_Autonomous\_Driving\_CVPR\_2020\_paper.html. 9  
Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas J. Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. arXiv preprint arXiv:2401.12168, 2024. URL https://arxiv.org/abs/2401.12168. 3, 4  
Wenshan Chen et al. Internvl: Scaling up vision foundation models and aligning for generic vision-language understanding. arXiv preprint arXiv:2305.05662, 2023. URL https://arxiv.org/abs/2305.05662. 3, 8, 2  
An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language models. arXiv preprint arXiv:2406.01584, 2024. URL https://arxiv.org/abs/2406.01584. 3, 4, 8, 9, 2  
An-Chieh Cheng, Yang Fu, Yukang Chen, Zhijian Liu, Xiaolong Li, Subhashree Radhakrishnan, Song Han, Yao Lu, Jan Kautz, Pavlo Molchanov, Hongxu Yin, Xiaolong Wang, and Sifei Liu. 3d aware region prompted vision language model. arXiv preprint arXiv:2509.13317, 2025. URL https://arxiv.org/abs/2509.13317. 3, 4, 5, 6, 8, 2  
Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, Minghao Zhu, Xiao Lin, Yang Bai, Qian Jiang, Yaxi Zhao, Minghua Zeng, Junlong Gao, Yuming Jiang, Jun Cen, Siteng Huang, Liuyi Wang, Wenqiao Zhang, Chengju Liu, Jianfei Yang, Shijian Lu, and Deli Zhao. Rynnbrain: Open embodied foundation models. CoRR, abs/2602.14979, 2026. 2  
DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3, 4  
Mengfei Du, Binhao Wu, Zejun Li, Xuanjing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 2024. 9  
Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024. 9  
Google DeepMind. Gemini: A family of highly capable multimodal models, 2023. URL https://deepmind.google/gemini/. 3  
Qiushan Guo et al. Regiongpt: Towards region understanding vision language model. arXiv preprint arXiv:2403.02330, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/papers/Guo\_RegionGPT\_Towards\_Region\_ Understanding\_Vision\_Language\_Model\_CVPR\_2024\_paper.pdf. 3  
Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025. 3, 4  
Yatai Ji, Rongcheng Tu, Jie Jiang, Weijie Kong, Chengfei Cai, Wenzhe Zhao, Hongfa Wang, Yujiu Yang, and Wei Liu. Seeing what you miss: Vision-language pre-training with semantic completion learning. In CVPR, pages 6789–6798. IEEE, 2023. 3  
Yatai Ji, Shilong Zhang, Jie Wu, Peize Sun, Weifeng Chen, Xuefeng Xiao, Sidi Yang, Yujiu Yang, and Ping Luo. IDA-VLM: towards movie understanding via id-aware large vision-language model. In ICLR. OpenReview.net, 2025. 3  
Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s "up" with vision-language models? investigating their struggle with spatial reasoning. In EMNLP, 2023. 3, 4  
Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. GitHub repository: https://github.com/lichengunc/refer, 2014. 7, 9  
Akshay Krishnan, Abhijit Kundu, Kevis-Kokitsi Maninis, James Hays, and Matthew Brown. Omninocs: A unified NOCS dataset and model for 3d lifting of 2d objects. In ECCV (75), volume 15133 of Lecture Notes in Computer Science, pages 127–145. Springer, 2024a. 9  
Akshay Krishnan, Abhijit Kundu, Kevis-Kokitsi Maninis, James Hays, and Matthew Brown. Omninocs: A unified nocs dataset and model for 3d lifting of 2d objects. In European Conference on Computer Vision, pages 127–145. Springer, 2024b. 7  
Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, and Afshin Dehghan. Cubify anything: Scaling indoor 3d object detection. arXiv preprint arXiv:2412.04458, 2024. 7, 9  
Phillip Y Lee, Jihyeon Je, Chanho Park, Mikaela Angelina Uy, Leonidas Guibas, and Minhyuk Sung. Perspective-aware reasoning in vision-language models via mental imagery simulation. arXiv preprint arXiv:2504.17207, 2025. 4  
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. CoRR, abs/2408.03326, 2024. 8, 2  
Hongxing Li, Dingming Li, Zixuan Wang, Yuchen Yan, Hang Wu, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. Spatialladder: Progressive training for spatial reasoning in vision–language models. arXiv preprint arXiv:2510.08531, 2025. 3, 5, 8  
Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699, 2024. 3  
Tsung-Yi Lin, Ming-Yu Liu, et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025. 8  
Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651, 2023. 3, 4  
Haotian Liu et al. Llava-1.5: Improved reasoning, grounding, and chart understanding. arXiv preprint arXiv:2402.16835, 2024. URL https://arxiv.org/abs/2402.16835. 3, 9  
Yang Liu, Ming Ma, Xiaomin Yu, Pengxiang Ding, Han Zhao, Mingyang Sun, Siteng Huang, and Donglin Wang. Ssr: Enhancing depth perception in vision-language models via rationale-guided spatial reasoning. arXiv preprint arXiv:2505.12448, 2025a. 4  
Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yaochen Hu, Lingfeng Zhang, Yingxue Zhang, Shuang Wu, Tongtong Cao, Guowei Huang, Helong Huang, Guangjian Tian, Weichao Qiu, Xingyue Quan, Jianye Hao, and Yuzheng Zhuang. Spatialcot: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning, 2025b. URL https://arxiv.org/abs/2501.10074. 4  
Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4122–4134, 2025c. 3, 8, 2  
Wufei Ma, Yu-Cheng Chou, Qihao Liu, Xingrui Wang, Celso de Melo, Jieneng Chen, Jianwen Xie, and Alan Yuille. Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. arXiv preprint arXiv:2504.20024, 2025a. 3, 4, 5  
Wufei Ma, Luoxin Ye, Celso de Melo, Alan L Yuille, and Jieneng Chen. Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models. In CVPR, 2025b. 3, 4, 5  
Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 3, 4  
Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025. 4  
OpenAI. Gpt-4o, 2024. URL https://openai.com. 3  
Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025. 8, 2  
Navid Rajabi and Jana Kosecka. Towards grounded visual spatial reasoning in multi-modal vision language models. arXiv preprint arXiv:2308.09778, 2023. 4  
Kanchana Ranasinghe, Satya Narayan Shukla, Omid Poursaeed, Michael S Ryoo, and Tsung-Yu Lin. Learning to localize objects improves spatial reasoning in visual-llms. In CVPR, pages 12977–12987, 2024. 4  
Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, and Kate Saenko. Sat: Spatial aptitude training for multimodal language models, 2025. 9  
Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, and Katerina Fragkiadaki. Grounded reinforcement learning for visual reasoning. arXiv preprint arXiv:2505.23678, 2025. 5, 8, 2  
Fatemeh Shiri, Xiao-Yu Guo, Mona Far, Xin Yu, Reza Haf, and Yuan-Fang Li. An empirical analysis on spatial reasoning capabilities of large multimodal models. In EMNLP, 2024. 3, 4  
Yihong Tang, Ao Qu, Zhaokai Wang, Dingyi Zhuang, Zhaofeng Wu, Wei Ma, Shenhao Wang, Yunhan Zheng, Zhan Zhao, and Jinhua Zhao. Sparkle: Mastering basic spatial capabilities in vision language models elicits generalization to spatial reasoning, 2025. URL https://arxiv.org/abs/2410.16162. 4  
Qwen Team. Qwen3-vl technical report. CoRR, abs/2511.21631, 2025. 8, 2  
Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, Hisham Cholakkal, Ivan Laptev, Mubarak Shah, Fahad Shahbaz Khan, and Salman Khan. Llamav-o1: Rethinking step-by-step visual reasoning in llms. arXiv preprint arXiv:2501.06186, 2025. 4  
Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 9  
Chenyang Wang et al. 3d-r1: Reinforcing 3d spatial reasoning and understanding in large multimodal models. arXiv preprint arXiv:2506.12322, 2025a. 3, 5  
Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025b. 3, 4  
Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, Sharon Li, and Neel Joshi. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. Advances in Neural Information Processing Systems, 37: 75392–75421, 2024a. 3, 4  
Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, Xihui Liu, Cewu Lu, Dahua Lin, and Jiangmiao Pang. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied AI. In CVPR, pages 19757–19767. IEEE, 2024b. 7  
Q. Wu et al. Visual spatial tuning: Bootstrapping spatial intelligence in vlms via 3d-aware perception and reinforcement learning. arXiv preprint arXiv:25xx.xxxxx, 2025. 5  
xAI. Realworldqa: A benchmark for real-world spatial understanding. https://huggingface.co/datasets/xai-org/RealworldQA, 2024. Accessed: 2025-11-13. 9  
Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, Yi Lin, and Hengshuang Zhao. Visual spatial tuning. CoRR, abs/2511.05491, 2025a. 8, 2  
Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, Bo Zhang, and Wei Chen. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025b. 3, 4  
Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, et al. Omnivinci: Enhancing architecture and data for omni-modal understanding llm. arXiv preprint arXiv:2510.15870, 2025. 3  
Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, Xingyue Quan, Hang Xu, and Li Zhang. From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976, 2025a. URL https://arxiv.org/abs/2503.22976. 7, 8, 9  
Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. arXiv preprint arXiv:2503.12937, 2025b. 3, 4  
Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8995–9006, 2025. 4  
Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024. URL https://arxiv.org/abs/2409.18125. 3, 4