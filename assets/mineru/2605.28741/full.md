# Self-Prophetic Decoding to Unlock Visual Search in LVLMs

Zhendong He \* 1 Qiyuan Dai \* 2 Guanbin Li 1 Liang Lin 1 Sibei Yang 1

# Abstract

Large Vision-Language Models (LVLMs) are rapidly evolving toward true multimodal reasoning, with visual search representing a concrete instantiation of the thinking-with-images paradigm. However, LVLM visual search faces two key challenges: incompatibility among intrinsic capabilities after post-training, and interference in long multi-step reasoning contexts. To address these, we identify two novel insights. First, selfregulation between pre- and post-training LVLMs leverages the intrinsic single-step capabilities of the pre-training model to mitigate capability deterioration and long-context interference. Second, probability-based prophetic sampling, replacing naive prompting, provides a probabilistic interface where the pre-training model acts as a prophet and the post-training model selectively accepts prophetic tokens under its output distribution, preserving coherent multi-step reasoning. Building on these insights, we introduce SeProD, a selfprophetic decoding framework that leverages intrinsic single-step capabilities to enable coherent multi-step reasoning in a training-free, plug-andplay manner. Experiments show that SeProD consistently improves multiple visual-search LVLMs across all 12 splits of 4 visual search benchmarks, as well as across general VQA benchmarks, without added computational overhead, thanks to its parallel prophetic acceptance mechanism.

# 1. Introduction

Large Vision-Language Models (LVLMs) are rapidly evolving from simple perception toward true multimodal reasoning. Recently, the thinking-with-images paradigm (Xu et al., 2025; Hu et al., 2024; Lai et al., 2025; Cheng et al., 2025; Fan et al., 2025; Liang et al., 2025) exemplifies the

\*Equal contribution 1School of Computer Science and Engineering, Sun Yat-sen University 2ShanghaiTech University. Correspondence to: Sibei Yang <yangsb3@mail.sysu.edu.cn>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

![](images/72f29c73f81b1991c0321698efb3af955a4d0145701c8b2bc53a3d0991f9ae1e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) External Tool Augmentation
        A["Thought"] --> B["Call Tools"]
        B --> C["Thought"]
        C --> D["Call Tools"]
        D --> E["Answer"]
        F["Input I_L + Param."] --> G["Expert models Detection Draw Segmentation"]
        G --> H["Output Image or Text"]
        H --> I["Interface: Normal New Input I_L able New Image or Prompt"]
    end

    subgraph (b) Intrinsic Model Extension
        J["Reasoning"] --> K["Grounding"]
        K --> L["Reasoning"]
        L --> M["Grounding"]
        M --> N["Answer"]
        O["New Reasoning"] --> P["New Grounding"]
        P --> Q["New Reasoning"]
        Q --> R["New Grounding"]
        R --> S["Context from Post-training LVLM"]
        T["Input I_L + Query"] --> U["Existing Intrinsic capabilities Counting Spatial Understanding"]
        U --> V["Output Tokens"]
        V --> W["Interface: SeProD Forward in Post-training LVLM"]
        W --> X["Tokens"]
    end

    subgraph (c) Self-prophetic Decoding
        Y["Original LVLM"] --> Z["Remain unchanged"]
        AA["Pre-training LVLM and Interface"] --> AB["Existing Intrinsic capabilities"]
        AB --> AC["OCR"]
        AC --> AD["Output Tokens"]
        AD --> AE["Context from Post-training LVLM"]
        AF["Input I_L + Query"] --> AG["Existing Intrinsic capabilities"]
        AG --> AH["OCR"]
        AH --> AI["Output Tokens"]
        AI --> AJ["Interface: SeProD Forward in Post-training LVLM"]
        AJ --> AK["Tokens"]
    end
```
</details>

Figure 1. Overview of paradigms for enabling visual search in LVLMs. (a) External tool augmentation. LVLMs call visual tools and fuse tool outputs into subsequent reasoning, but the interface is rigid and fragments multi-step reasoning. (b) Intrinsic model extensions. LVLMs natively activate zoom-in and region grounding in a single forward pass, but visual-search post-training introduces incompatibilities among these intrinsic capabilities. (c) Our SeProD remains naturally compatible with LVLMs by operating at the pre-training level, while providing a flexible probabilistic interface to seamlessly integrate and coordinate these abilities during post-training visual search. Consequently, SeProD preserves intrinsic single-step capabilities like (a), while enabling coherent multi-step reasoning akin to (b).

generation and integration of visual evidence as intermediate reasoning steps, extending beyond text-only chainof-thought (Chen et al., 2024b; Zheng et al., 2023; Mitra et al., 2024). In particular, visual search (Zhang et al., 2025b) constitutes a concrete and well-defined instantiation of thinking-with-images, in which correctly answering questions requires actively searching, grounding, and zooming in to verify specific regions within usually high-resolution images. This makes visual search a natural and tractable starting point for studying multimodal reasoning.

Recent efforts to endow LVLMs with visual search broadly fall into two categories: external tool augmentation and intrinsic model extension. The former employs functioncalling or tool usage to provide LVLMs with tool-specific capabilities, the outputs of which are then integrated to LVLMs’ subsequent reasoning (Li et al., 2025; Hu et al., 2024; Su et al., 2025; Wu et al., 2025). While enabling direct use of well-developed tools, it is constrained by rigid interfaces and the risk of errors from treating multi-step, context-dependent reasoning as multiple isolated, singlestep task executions, as shown in Fig. 1(a). In contrast, intrinsic model extensions aim to directly activate LVLMs’ native capabilities, such as zoom-in and region-of-interest grounding, during reasoning (Wang et al., 2025a; Zheng et al., 2025; Zhang et al., 2025b; Wang et al., 2025c; Lai et al., 2025). This allows closer integration between reasoning and intrinsic abilities, supporting cross-step knowledge sharing within a single forward pass, as shown in Fig. 1(b).

![](images/c0d9c5978cdcefdf225f82434ffb5f1577bf513e5700a7057127dba74722c5df.jpg)

<details>
<summary>bar</summary>

| Metric | Pre-training Model (%) | Post-training Model (%) |
| :--- | :--- | :--- |
| Grounding RefCOCO_val | 90.0 | 40.9 |
| OCR | 40.9 | 38.7 |
| Spatial | 48.7 | 37.8 |
| Counting | 30.3 | 29.0 |
Red arrows indicate statistical changes (-49.3, -2.3, -10.9, -1.0). The chart includes percentage values (e.g., 90.0, -49.3) above each pair of bars.
</details>

![](images/5a44c566e41f02b6ca1a04e138c7007e8435645d26b1990d0e53f3151a2d6a0b.jpg)

<details>
<summary>bar</summary>

| VisualProbe | Accuracy (%) |
| :--- | :--- |
| Easy | 61.70 + 5.67 |
| Medium | 48.88 + 2.24 |
| Hard | 37.74 + 5.66 |
(a) Increase from top to bottom is the white car on the third bridge facing left or right?
Turn 0: relevant to the question
... I can see three main bridges...
Turn 1: relevant to the question
... There is a white car visible on the third bridge from the top...
Turn 2: relevant to the question
... It seems I focused on the wrong bridge last time...
Turn 3: relevant to the question
... I can see a white car on the bridge... I will try to zoom in on...
Turn 4: relevant to the question
... My previous attempts didn't isolate the car correctly. Let me carefully re-examine the image again...
Turn 5: I see three bridges. The white car is on the third bridge...
Keep all turns (from turn 0 to turn 5)
Answer: Left ✗
Remove irrelevant turns (turn 2 and turn 4)
Answer: Right ✓
</details>

![](images/267ce65e64c5b06cbd27f0f0d344b0a1322ef2b215440a21530c90a706b8143a.jpg)

<details>
<summary>line</summary>

| Probability Value | Original | Naive | SeProQ |
| ----------------- | -------- | ----- | ------ |
| 0.75              | 0%       | 0%    | 0%     |
| 0.85              | 0%       | 0%    | 0%     |
| 0.95              | 0%       | 0%    | 0%     |
| >0.95             | >30%     | >35%  | >40%   |
</details>

Figure 2. (a) The degradation of intrinsic capabilities at a single step after visual-search post-training. Performance drops on grounding, OCR, spatial understanding, and counting when evaluated at a specific reasoning turn. (b) Interference accumulation in long multi-step trajectories. Masking irrelevant context recovers correct predictions, indicating sensitivity to early-step errors. (c) Distribution curves of the original visual-search LVLM, the na¨ıve method, and our SeProD, shown in blue, orange, and green, demonstrate that SeProD, by accepting only tokens aligned with the native distribution, preserves output consistency with the original model and thereby promotes coherent multi-step reasoning. Please refer to Appendix Sec. E for experimental details.

In this work, we focus on visual search, particularly complex scenarios that unfold over multiple interdependent reasoning steps, which motivates us to follow intrinsic model extensions. However, current intrinsic model extensions face two crucial challenges: (1) Incompatibility among intrinsic capabilities after visual-search post-training. Grounding, answering, and high-level reasoning are not jointly optimized in visual search. Limited instruction-following data restricts generalization, and reinforcement learning lacks supervision on intermediate steps. The optimization therefore favors task completion over maintaining robustness of individual steps that depend on different intrinsic abilities. Consequently, capabilities that originally functioned independently begin to interfere with each other, and some deteriorate or are even catastrophically forgotten. We assess single-step reasoning by selecting a specific step from the inference trajectory and comparing models (Bai et al., 2025b; Lai et al., 2025) with and without visual search post-training. As shown in Fig. 2(a), post-training reduces grounding, OCR recognition, spatial understanding, and counting performance by 49.3%, 2.3%, 10.9% and 3.0%, respectively. (2) Interference in long multi-step reasoning contexts. Long multi-step reasoning contexts, while essential for complex visual search, can in turn propagate interference, as mistaken intermediate predictions disrupting later reasoning and irrelevant information impairing the final answer, particularly when it mainly relies on visual evidence from the last step. As shown in the left panel of Fig. 2(b), removing irrelevant content from the long reasoning trajectory yields improvements of 5.56%, 2.24%, and 5.66% on the three splits of VisualProbe-test (Lai et al., 2025), respectively. An example is provided in the right panel of Fig. 2(b).

To overcome these challenges, we introduce our first key insight: the self-regulation of pre- and post-training LVLMs. On one hand, since distinct intrinsic capabilities of pretraining LVLMs1, e.g. grounding, zoom-in, and OCR recognition, already exist, they can be leveraged as internal guides to help the post-training LVLM integrate and coordinate these abilities, effectively addressing the challenge of incompatibility among intrinsic functions. On the other hand, post-training LVLMs can exploit cross-step shared context to dynamically steer the reasoning trajectory and focus of pre-training LVLMs for each single step. Simultaneously, feedback from these focused, single-step capabilities can mitigate the challenge of interference from long contexts while preserving reasoning coherence.

Despite these advantages, a new technical challenge arises: (3) How to design an effective interface to enable interaction between pre- and post-training LVLMs? A na¨ıve approach is to feed textual prompts from the pre-training model as additional input prefixes or to overwrite the posttraining model’s current output. However, this reduces the interaction to a tool-like interface and introduces two issues. First, the prompts often fail to directly steer the posttraining LVLM’s outputs, limiting their effect on subsequent reasoning. Second, even when they do influence the outputs, their single-step nature can disrupt contextual coherence, destabilizing multi-step reasoning, as shown in Appendix Fig. 8. To address this, we propose our second key insight: probability-based prophetic sampling as the interface, as shown in Fig. 1(c). In prophetic sampling, the pre-training LVLM acts as a “prophet,” and the post-training LVLM accepts tokens generated by this prophet if their probability under the adjusted distribution derived from posttraining itself and the prophet exceeds a predefined threshold; otherwise, tokens are sampled from the post-training model. This mechanism allows the post-training LVLM to treat pre-training outputs directly as generated tokens rather than inputs, propagating accepted prefixes through subsequent decoding. Meanwhile, by only accepting tokens that align with its native distribution, the post-training model preserves its output consistency and maintains coherent multi-step reasoning, as evidenced by the blue and green distribution curves in Fig. 2(c).

To realize these insights, we propose a self-prophetic decoding (SeProD) framework for visual search, harnessing LVLMs’ intrinsic capabilities to improve multi-step reasoning in a training-free, plug-and-play manner. Specifically, the post-training LVLM serves as the search model, dynamically steering the reasoning trajectory and executing visual search in a single forward pass. At each reasoning step, the current input and partial output are fed to the pre-training LVLM, acting as a prophet, which generates single-step, capability-specific prophetic prefixes. Then, the search model evaluates all prophetic prefixes in parallel, selectively accepting those that align with its output distribution, ensuring controlled and coherent propagation. Such interaction between the search and prophet models iterates over the reasoning process. Notably, prefix evaluation is fully parallelizable, and the prophet model can be a smaller counterpart, incurring minimal overhead and potentially improving efficiency. Moreover, because this sampling is imposed solely at inference and is compatible with any intrinsic-extended LVLM for visual search, Se-ProD is inherently training-free and plug-and-play.

In summary, our contributions are multi-faceted:

• We identify a new insight that self-regulation between pre- and post-training LVLMs leverages intrinsic singlestep capabilities to mitigate deterioration and long-context interference caused by visual search training.   
• We are the first to propose a probability-based prophetic interface that treats pre-training outputs as prophetic predictions, accepting them approximately under the post-training distribution to preserve coherence reasoning.   
• We introduce a self-prophetic decoding framework (Se-ProD) that enables the activation of intrinsic single-step capabilities while supporting multi-step coherent reasoning in a training-free, plug-and-play manner.   
• SeProD consistently improves multiple visual-search LVLMs, achieving state-of-the-art results on all 12 splits across 4 visual search benchmarks. It also yields gains on

several general VQA benchmarks, while introducing no additional computational overhead.

# 2. Related Work

Large Vision-Language Models (LVLMs) constitute a central line of research in multimodal learning, aiming to align visual representations with large language models for unified understanding and reasoning. Early works, such as BLIP-2 (Li et al., 2023) and LLaVA (Liu et al., 2023), established a paradigm that combines pre-trained vision encoders with large language models, enabling effective alignment between visual and linguistic modalities. To support more flexible image inputs and finer-grained visual perception, recent studies adopt the AnyRes(Li et al., 2024) strategy to overcome resolution constraints in input images. Building upon these advances, a series of strong LVLMs have emerged, including LLaVA (Li et al., 2024; Xu et al., 2024), Qwen-VL (Wang et al., 2024a; Bai et al., 2025b;a), and InternVL (Chen et al., 2024a; Gao et al., 2024).

Multimodal Reasoning. To enable sophisticated reasoning, existing methods generally follow two main categories. The first category transforms visual inputs into structured or symbolic representations (Yang et al., 2019; Zheng et al., 2023; Shao et al., 2024; Chen et al., 2024b), such as object relationship graph, captions and scene graphs, and then performing symbolic or language-based reasoning over these intermediate representations. The second category follows the thinking-with-images paradigm, in which LVLMs perform iterative, multi-step reasoning by dynamically interacting with visual inputs at inference time. Within this paradigm, some approaches (Sur´ıs et al., 2023; Hu et al., 2024; Su et al., 2025) rely on external tools, such as visual experts or code execution modules, to progressively refine visual evidence during reasoning. In contrast, other methods (Shen et al., 2025a; Fan et al., 2025; Zhang et al., 2025b; Wang et al., 2025a; Zheng et al., 2025; Wang et al., 2025c; Lai et al., 2025) stimulate the model’s intrinsic capabilities, such as grounding, enabling it to capture fine-grained visual cues and support complex reasoning patterns.

Visual Search is a challenging yet essential multimodal capability within the thinking-with-images paradigm. It requires models to answer questions over high-resolution images by actively identifying and localizing regions relevant to fine-grained visual details. Some existing works (Wu & Xie, 2023; Liu et al., 2025; Li et al., 2025; Shen et al., 2025b; Zhang et al., 2025a) augment LVLMs with external modules or tools to perform visual search. In contrast, other approaches enable LVLMs to exploit their intrinsic grounding capability for region localization and zoom-in operations through end-to-end training (Zhang et al., 2025b; Wang et al., 2025c;a; Zheng et al., 2025; Lai et al., 2025). For example, ZoomEye (Shen et al., 2025b) employs a tree search algorithm to simulate zoom-in and zoom-out operations, allowing the model to inspect image details and identify key information. DeepEyes (Zheng et al., 2025) and Mini-o3 (Lai et al., 2025) further leverage reinforcement learning to enable models to select zoom-in regions based on their intrinsic grounding capability, thereby acquiring task-relevant visual details. However, they are constrained by rigid interfaces, incompatibilities among intrinsic abilities, or interference from long contexts. SeProD addresses these limitations by enabling self-regulation and adopting probability-based sampling as the interface.

![](images/9f4bbb86b62a91936acb547370b80fbe3ce49eb8e367e4d0baf06e1fabb1ae31.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Search Model"] --> B["Turn 1"]
    B --> C["New Reasoning"]
    C --> D["New Grounding"]
    D --> E["Turn 2"]
    E --> F["New Reasoning"]
    F --> G["New Grounding"]
    G --> H["Turn i+1 R_{i+1} New Reasoning"]
    H --> I["Final Answer A_i"]
    I --> J["Proofing Mode"]
    J --> K["Grounding Mode"]
    K --> L["Self-Prophetic Decoding (SeProD)"]
    L --> M["Evaluate in Parallel"]
    L --> N["Decode Token-by-token"]
    L --> O["Search Model X"]
    L --> P["Extract Probabilities in Parallel"]
    P --> Q["Search Model O_i (Candidate Prefixes)"]
    P --> R["Prophet Model P_p(O_i)"]
    
    subgraph Prophet Model
        S["Input I_i World"] --> T["Grounding Mode"]
        T --> U["Answering Mode"]
        U --> V["Proofing Mode"]
        V --> W["Final Answer A_i"]
        W --> X["Proofing Mode"]
        X --> Y["H_{i-1}"]
        Y --> Z["Resulting Node"]
        Z --> AA["Proofing Mode"]
        AA --> AB["Final Answer A_i"]
        AB --> AC["Proofing Mode"]
        AC --> AD["Final Answer A_i"]
        AD --> AE["H_{i-1}"]
        AE --> AF["Resulting Node"]
        AF --> AG["Final Answer A_i"]
        AG --> AH["H_{i-1}"]
        AH --> AI["Resulting Node"]
        AI --> AJ["Final Answer A_i"]
        AJ --> AK["H_{i-1}"]
        AK --> AL["Resulting Node"]
        AL --> AM["H_{i-1}"]
        AM --> AN["Resulting Node"]
        AN --> AO["H_{i-1}"]
        AO --> AP["Resulting Node"]
        AP --> AQ["H_{i-1}"]
        AQ --> AR["Resulting Node"]
        AR --> AS["H_{i-1}"]
        AS --> AT["Resulting Node"]
        AT --> AU["H_{i-1}"]
        AU --> AV["Resulting Node"]
        AV --> AW["H_{i-1}"]
        AW --> AX["Resulting Node"]
        AX --> AY["H_{i-1}"]
        AY --> AZ["Resulting Node"]
        AZ --> BA["H_{i-1}"]
        BA --> BB["Resulting Node"]
        BB --> BC["H_{i-1}"]
        BC --> BD["Resulting Node"]
        BD --> BE["H_{i-1}"]
        BE --> BF["Resulting Node"]
        BF --> BG["H_{i-1}"]
        BG --> BH["Resulting Node"]
        BH --> BI["H_{i-1}"]
        BI --> BJ["Resulting Node"]
        BJ --> BK["H_{i-1}"]
        BK --> BL["Resulting Node"]
        BL --> BM["H_{i-1}"]
        BM --> BN["Resulting Node"]
        BN --> BO["H_{i-1}"]
        BO --> BP["Resulting Node"]
        BP --> BQ["H_{i-1}"]
        BQ --> BR["Resulting Node"]
        BR --> BS["H_{i-1}"]
        BS --> BT["Resulting Node"]
        BT --> BU["H_{i-1}"]
        BU --> BV["Resulting Node"]
        BV --> BW["H_{i-1}"]
        BW --> BX["Resulting Node"]
        BX --> BY["H_{i-1}"]
        BY --> BZ["Resulting Node"]
        BZ --> CA["H_{i-1}"]
        CA --> CB["Resulting Node"]
        CB --> CC["H_{i-1}"]
        CC --> CD["Resulting Node"]
        CD --> CE["H_{i-1}"]
        CE --> CF["Resulting Node"]
        CF --> CG["H_{i-1}"]
        CG --> CH["Resulting Node"]
        CH --> CI["H_{i-1}"]
        CI --> CJ["Resulting Node"]
        CJ --> CK["H_{i-1}"]
        CK --> CL["Resulting Node"]
        CL --> CM["H_{i-1}"]
        CM --> CN["Resulting Node"]
        CN --> CO["H_{i-1}"]
        CO --> CP["Resulting Node"]
        CP --> CQ["H_{i-1}"]
        CQ --> CR["Resulting Node"]
        CR --> CS["H_{i-1}"]
        CS --> CT["Resulting Node"]
        CT --> CU["H_{i-1}"]
        CU --> CV["Resulting Node"]
        CV --> CW["H_{i-1}"]
        CW --> CX["Resulting Node"]
        CX --> CY["H_{i-1}"]
        CY --> CZ["Resulting Node"]
        CZ --> DA["H_{i-1}"]
        DA --> DB["Resulting Node"]
        DB --> DC["H_{i-1}"]
        DC --> DD["Resulting Node"]
        DD --> DE["H_{i-1}"]
        DE --> DF["Resulting Node"]
        DF --> DG["H_{i-1}"]
        DG --> DH["Resulting Node"]
        DH --> DI["H_{i-1}"]
        DI --> DJ["Resulting Node"]
        DJ --> DK["H_{i-1}"]
        DK --> DL["Resulting Node"]
        DL --> DV["H_{i-1}"]
        DV --> DW["Resulting Node"]
    end
    
    style Prophet Model fill:#f9f,stroke:#333
    style Self-Prophetic Decoding fill:#ccf,stroke:#333
```
</details>

Figure 3. The overall framework of SeProD. (1) Pair of search model and prophet model. The post-training LVLM serves as the search model, responsible for steering the global multi-turn reasoning process. Its pre-training counterpart acts as the prophet model, exploiting native intrinsic capabilities to produce single-step prophetic prefixes. The two models are coupled through bidirectional signals: search-to-prophet specifies task-relevant focus at each turn, while prophet-to-search provides guidance for subsequent reasoning. (2) Self-prophetic decoding incorporates prophetic prefixes into the search model’s decoding process. Candidate prefixes are evaluated in parallel and accepted only when they fall under the adjusted distributions of both models. Once accepted, they are absorbed as native tokens of the search model and propagated through subsequent token-by-token decoding, preserving the original generation dynamics.

# 3. Method

The overall framework of self-prophetic decoding (SeProD) is illustrated in Fig. 3. SeProD enhances post-training LVLMs for visual search by integrating a pre-training counterpart that provides complementary intrinsic capabilities and single-step regulation via distribution-approximately prophetic sampling. First, we formalize the post-training LVLM as the search model and its pre-training counterpart as the prophet model (see Sec. 3.1). The search model dynamically steers the overall reasoning trajectory, while the prophet generates single-step, capability-specific prophetic prefixes, with the two models interacting iteratively. Second, in the search-to-prophet interaction, the search model guides the prophet’s focus at each reasoning step, allowing it to generate prophetic prefixes for tasks focusing on distinct abilities such as grounding and counting (see Sec. 3.2). Third, in the prophet-to-search interaction, the prophetic prefixes are selectively incorporated into the search model via probability-based prophetic sampling, directly shaping its outputs while preserving the native distribution, thereby maintaining coherent multi-step reasoning (see Sec. 3.3).

# 3.1. Definition of Search and Prophet Models

We define a pair of search and prophet models in SeProD. The search model refers to any LVLM that has undergone post-training for visual search, e.g., Mini-o3 (Lai et al., 2025), DeepEyes (Zheng et al., 2025), and Pixel Reasoner (Wang et al., 2025a), or other intrinsic-extended LVLMs. Its prophet model is the corresponding pretraining counterpart, i.e., the same base LVLMs before visual search post-training. This pairing captures a complementary asymmetry: post-training equips the search model with long-horizon reasoning and visual-search priors, while the prophet preserves strong single-step capabilities.

Search Model continuously adjusts and steers the global reasoning trajectory. Specifically, given an image I, typically at high resolution, and a query Q, the search model performs multi-turn reasoning and generation in a single forward pass. At each reasoning turn i, it generates the output (Ii, Ci) conditioned on the entire history Hi−1:

$$
H _ {i - 1} = \{(I, Q), (I _ {1}, C _ {1}), \dots , (I _ {i - 1}, C _ {i - 1}) \}. \tag {1}
$$

Here, $I _ { i - 1 }$ denotes the cropped then zoomed-in image produced from the previous step $i - 1$ , and $C _ { i - 1 }$ is the corresponding textual output. The output $C _ { i }$ is generated autoregressively as follows,

$$
p _ {s} (C _ {i} \mid H _ {i - 1}) = \prod_ {j = 1} ^ {L _ {c}} p _ {s} (c _ {i, j} \mid H _ {i - 1}, c _ {i, <   j}), \tag {2}
$$

where $c _ { i , j }$ and $c _ { i , < j }$ denote the j-th token and its preceding tokens in $C _ { i } ,$ respectively, and $L _ { c }$ is the token length of $C _ { i }$ . For simplicity, and without loss of generality, we assume that all $C _ { i }$ share the same token length. In particular, each $C _ { i }$ and its corresponding $I _ { i }$ are generated in two modes:

• Grounding mode: $C _ { i } = ( R _ { i } , G _ { i } )$ , where $R _ { i }$ are reasoning tokens and $G _ { i }$ is a grounding prediction, specifying a reference to a previously observed image (from the original image up to turn $i - 1 )$ and a region of interest coordinate within it. The output image $I _ { i }$ is obtained by cropping the referenced image at this region and zooming in.   
• Answering mode: $C _ { i } \ = \ ( R _ { i } , A _ { i } )$ , where $A _ { i }$ contains the final answer. In this mode, the current turn terminates, and no further output image is generated. For notational convenience, we set image output $I _ { i } = I _ { i - 1 }$ in this case.

Notably, while each reasoning step yields only two output modes, these merely represent the form of the output; in reality, each step may engage distinct capabilities, such as grounding, OCR recognition, counting, and more.

Prophet Model, fortunately defined as the search model’s base LVLM prior to visual-search post-training, perfectly leverages its native intrinsic abilities to serve as a complementary aid to the search model. Moreover, we select this counterpart because its general output distribution closely aligns with the search model, enhancing the acceptance rate in the self-prophetic decoding of Sec. 3.3. Specifically, given an image I and a query $Q$ , the prophet model generates the output O directly in a single turn as follows,

$$
p _ {p} (O \mid I, Q) = \prod_ {i = 1} ^ {L _ {d}} p _ {p} (o _ {i} \mid I, Q, o _ {<   i}), \tag {3}
$$

where $o _ { i }$ denotes the token at position i, $o _ { < i }$ denotes all preceding tokens, and $L _ { d }$ is total token sequence length.

# 3.2. Search-to-Prophet Single-Step Focus

In the search-to-prophet interaction, while the search model encompasses a multi-step reasoning trajectory, it provides step-specific guidance, directing the prophet’s focus at each turn and enabling the generation of prophetic prefixes tailored to distinct task-specific abilities.

Specifically, at the i-th reasoning turn of the search model, the prophet model takes the target’s output image $I _ { i }$ as input. The textual output $C _ { i }$ is omitted to avoid biasing the draft toward the target’s reasoning trajectory, preserving its independent, single-step capabilities. To accommodate the two modes of $C _ { i } -$ grounding and answering—we introduce two distinct queries: a grounding query, which verifies the validity of the current cropped image and steers subsequent reasoning, and an answering query, which produces answer drafts. The prophet model’s prediction $O _ { i }$ at this turn is then formulated as follows,

$$
p _ {p} \left(O _ {i} \mid I _ {i}, Q ^ {p}\right) = \prod_ {j = 1} ^ {L _ {d}} p _ {p} \left(o _ {i, j} \mid I _ {i}, Q ^ {p}, o _ {i, <   j}\right), \tag {4}
$$

$$
Q ^ {p} = \left\{ \begin{array}{l l} Q ^ {g}, & \text {if C_{i} = (R_{i} ,G)}, \\ Q, & \text {otherwise.} \end{array} \right.
$$

Here, $Q ^ { p }$ denotes the prophet model’s query. When the search model’s output $C _ { i }$ is in grounding mode, $Q ^ { p }$ is set to the grounding verification query $Q ^ { g } ;$ otherwise, it defaults to the original query $Q$ for question answering. The prophet model consistently uses the current turn’s output image $I _ { i }$ as its visual input. Next, we provide a detailed specification for these two modes.

Grounding Verification for Next-Step Steering first determines whether the current image crop contains the relevant regions for answering the query and, based on this feedback, generates prophetic prefixes for the search model’s next turn. Specifically, it evaluates the presence of regions-of-interest in the current image $I _ { i } ,$ yielding a binary response, true or false. If true, more detailed information about the region is included in the prophet model’s output $O _ { i } ,$ serving as prophetic prefixes for generating the reasoning token $R _ { i + 1 }$ in the next turn $i + 1$ (see Sec. 3.3 Eq. (5)). If false, the search model is prompted to relocate the potential region.

Answer Drafting for Final Correction directly produces the prophet model’s answer $O _ { i } .$ , which is used as prophetic prefixes for the search model’s final answer $A _ { i }$ via propheticto-search sampling. In this mode, $A _ { i }$ is not generated beforehand; it is obtained on-the-fly through prophetic sampling for efficiency.

In summary, the prophet model’s outputs $O _ { i }$ (with their probabilities $p _ { p } ( O _ { i } \mid I _ { i } , Q ^ { p } ) )$ serve distinct roles for the search model: supplying prophetic prefixes for next-turn reasoning tokens $R _ { i + 1 }$ in grounding mode, and prophetic prefixes for the current-turn answer $A _ { i }$ in answering mode.

# 3.3. Prophet-to-Search Probabilistic Decoding

In the prophet-to-search interaction, the prophet model’s output at the current turn, derived from its intrinsic singlestep capabilities during the search-to-prophet phase, in turn feeds back to guide the search model’s subsequent reasoning, grounding and answering. As discussed in Sec. 1, a straightforward interface that directly feeds the prophet model’s output as an additional input to the search model either fails to meaningfully influence its generation or overly constrains, disrupting reasoning coherence. We thus propose a novel probability-based self-prophetic decoding to address these issues. Inspired by LLM speculative decoding (Leviathan et al., 2023) for inference acceleration, we propose a novel probability-based self-prophetic decoding, introduced for the first time to LVLM reasoning and visual search.

Specifically, at the i-th reasoning turn, the prophet model’s output $O _ { i }$ (defined at Eq. (4)) serves as prophetic prefixes for the search model to generate $R _ { i + 1 } \thinspace \thinspace \mathrm { o r } \ : A _ { i }$ . For notational simplicity, we denote the resulting output after self-prophetic decoding as $X$ , with $X \in \{ R _ { i + 1 } , A _ { i } \}$ . Generation of X proceeds in two steps. First, the search model evaluates all tokens in $O _ { i }$ and their probabilities $p _ { p } ( O _ { i } \mid I _ { i } , Q ^ { p } )$ in parallel, selectively accepting those approximately consistent with both the search and prophet models’ distributions. This evaluation accelerates the search model’s inference, as the tokens in $O _ { i }$ are pre-collected and can be evaluated in parallel, rather than being generated sequentially. Second, for the first token that is not accepted, an additional token is sampled from the search model’s distribution, and generation continues iteratively until X is complete. The sampling of the j-th token of X during decoding is formulated as follows:

$$
x _ {j} \sim \left\{ \begin{array}{l l} p _ {p} (x _ {j} \mid I _ {i}, Q ^ {p}, x _ {<   j}), & \text { if } j <   \min \{j \mid s _ {j} <   \tau \}, \\ p _ {s} (x _ {j} \mid H _ {i}, x _ {<   j}), & \text { otherwise }, \end{array} \right.
$$

where

$$
s _ {j} = p _ {s} (o _ {i, j} \mid H _ {i}, o _ {i, <   j}) ^ {\alpha} p _ {p} (o _ {i, j} \mid I _ {i}, Q ^ {p}, o _ {i, <   j}) ^ {1 - \alpha} \tag {5}
$$

Here, $s _ { j }$ measures the consistency of the $j \mathrm { - t h }$ token $o _ { i , j }$ in $O _ { i }$ with both the search and prophet model distributions, specifically $p _ { s } ( o _ { i , j } \mid H _ { i } , o _ { i , < j } )$ and $p _ { p } ( o _ { i , j } \mid I _ { i } , Q ^ { p } , o _ { i , < j } )$ , with higher values indicating stronger agreement. And α is a balancing factor, initialized to 0.5 and automatically adjusted based on the normalized relative rank of token $o _ { i , j }$ in the search model’s logits. A higher rank leads to a larger $\alpha ,$ promoting stronger adherence to the search model’s native distribution. τ serves as a consistency threshold hyperparameter that determines the acceptance boundary for draft tokens $O _ { i }$ . When the score $s _ { j } \geq \tau$ , the search model accepts the prophet model’s output $o _ { i , j }$ . Conversely, min $\{ j \ | \ s _ { j } < \tau \}$ identifies the index of the first rejected token, from which the search model resumes generation solely based on its native distribution $p _ { s } ( x _ { j } \mid H _ { i } , x _ { < j } )$ .

Notably, the proposed probability-based self-prophetic decoding allows the search model to accept the prophet model’s outputs as valid prefixes conforming to its own distribution, thereby fully leveraging single-round capabilities while maintaining coherence across multi-turn reasoning. Using the proposed decoding, the search model generates the output X. As before, in grounding mode, X corresponds to the next-turn reasoning tokens $R _ { i + 1 }$ , while in answering mode it represents the current-turn answer $A _ { i }$ . The searchto-prophet and prophet-to-search interactions then alternate iteratively until the final answer is produced.

# 4. Experiment

Benchmarks. We evaluate SeProD using two categories of benchmarks. (1) Visual Search. We use V\* Bench (Wu & Xie, 2023), HR-Bench (Wang et al., 2024b), and Visual-Probe test (Lai et al., 2025). These benchmarks focus on high-resolution visual reasoning and require precise localization of small visual entities. They test whether an LVLM can extract task-relevant information from complex visual scenes and handle fine-grained spatial distinctions. (2) General VQA. We use MME-RealWorld (Zhang et al., 2025c) as the comprehensive general VQA benchmark. In addition, we include ScienceQA (Lu et al., 2022), OCRBench (Liu et al., 2024), and CVBench (Tong et al., 2024) as benchmarks that aim to reflect intrinsic LVLM capabilities such as OCR recognition, spatial understanding, and counting. Details of the benchmarks are provided in Appendix Sec. D.

Baselines. To verify the transferability and robustness of our framework, we select three representative post-training LVLMs, i.e., Pixel Reasoner (Wang et al., 2025a), Deep-Eyes (Zheng et al., 2025), and Mini-o3 (Lai et al., 2025), as baselines. These models vary substantially in both architecture and post-training strategies. By applying our approach to each of them without modification, we demonstrate that it can be seamlessly integrated into diverse model families and reasoning pipelines.

Implementation Details. We use Qwen2.5-VL-3B (Bai et al., 2025b) as the default prophet model for efficiency, unless explicitly stated otherwise. We use the official pipelines of each baseline and apply SeProD during inference. The consistency threshold τ is fixed to 0.3 and set $\alpha = 0 . 5 - r$ , where r denotes the normalized rank of the corresponding token in the search model’s output logits. We adopt either the standard accuracy or the averaged accuracy. Specifically, VisualProbe test is evaluated using avg@32 for all baselines, while for $\mathbf { V } ^ { \ast }$ Bench and HR-Bench, we use standard accuracy for Pixel Reasoner and DeepEyes. Following the evaluation metrics of Mini-o3, avg@32 and avg@8 are used for the two benchmark respectively for Mini-o3.

# 4.1. Comparison with State-of-the-Art Methods

Tab. 1 presents a comparative analysis of SeProD against state-of-the-art approaches on all high-resolution benchmarks. We compare the performance of the method both with and without external tools. Across all the benchmarks, our approach consistently yields notable gains over the baselines, underscoring its effectiveness in enhancing posttraining LVLMs via probability-guided prophetic sampling, which enables a more synergistic interaction between preand post-training models.

Table 1. Comparison with state-of-the-art methods on high-resolution benchmarks. Performance of SeProD evaluated on VisualProbetest, V\* Bench, and HR-Bench across multiple baseline LVLMs. SeProD consistently improves over the original models on all benchmarks and subsets, including both open-ended reasoning and multiple-choice settings. The gains are particularly pronounced on challenging scenarios with long reasoning trajectories and strong spatial or cross-instance perception requirements. Results marked with † are reproduced using the official code. “7B” denotes the prophet model size; otherwise, it defaults to 3B. 

<table><tr><td rowspan="2">Method</td><td colspan="3">VisualProbe</td><td colspan="3">V* Bench</td><td colspan="3">HR-Bench 4K</td><td colspan="3">HR-Bench 8K</td></tr><tr><td>Hard</td><td>Medium</td><td>Easy</td><td>Attr.</td><td>Spatial</td><td>Overall</td><td>FSP</td><td>FCP</td><td>Overall</td><td>FSP</td><td>FCP</td><td>Overall</td></tr><tr><td colspan="13">Methods with external tool</td></tr><tr><td>SEAL (Wu &amp; Xie, 2023)</td><td>-</td><td>-</td><td>-</td><td>74.8</td><td>76.3</td><td>75.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ZoomEyes (Shen et al., 2025b)</td><td>-</td><td>-</td><td>-</td><td>93.9</td><td>85.5</td><td>90.6</td><td>84.3</td><td>55.0</td><td>69.6</td><td>88.5</td><td>50.0</td><td>69.3</td></tr><tr><td>RAP (Wang et al., 2025b)</td><td>-</td><td>-</td><td>-</td><td>90.4</td><td>96.1</td><td>91.1</td><td>73.8</td><td>40.5</td><td>57.1</td><td>72.3</td><td>35.3</td><td>53.8</td></tr><tr><td>DyFo (Li et al., 2025)</td><td>-</td><td>-</td><td>-</td><td>80.0</td><td>82.9</td><td>81.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FOCUS (Zhong et al., 2025)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>90.6</td><td>-</td><td>-</td><td>79.3</td><td>-</td><td>-</td><td>76.3</td></tr><tr><td colspan="13">Methods without external tool</td></tr><tr><td>Chain-of-Focus (Zhang et al., 2025b)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>88.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Simple-o3 (Wang et al., 2025c)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>90.4</td><td>-</td><td>-</td><td>76.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Pixel Reasoner $^{\dagger}$ (Wang et al., 2025a)</td><td>28.7</td><td>29.0</td><td>58.7</td><td>88.7</td><td>84.2</td><td>86.9</td><td>84.8</td><td>60.5</td><td>72.6</td><td>76.3</td><td>52.3</td><td>64.3</td></tr><tr><td>Pixel Reasoner + Ours</td><td>30.2</td><td>30.4</td><td>61.7</td><td>90.4</td><td>85.5</td><td>88.5</td><td>86.0</td><td>61.3</td><td>73.6</td><td>77.5</td><td>52.8</td><td>65.1</td></tr><tr><td>Δ</td><td>+1.5</td><td>+1.4</td><td>+3.0</td><td>+1.7</td><td>+1.3</td><td>+1.6</td><td>+1.2</td><td>+0.8</td><td>+1.0</td><td>+1.2</td><td>+0.5</td><td>+0.8</td></tr><tr><td>DeepEyes $^{\dagger}$ (Zheng et al., 2025)</td><td>38.4</td><td>30.5</td><td>61.2</td><td>90.4</td><td>86.8</td><td>89.0</td><td>90.8</td><td>55.3</td><td>73.0</td><td>85.3</td><td>54.5</td><td>69.9</td></tr><tr><td>DeepEyes + Ours</td><td>41.9</td><td>32.3</td><td>64.7</td><td>91.3</td><td>90.8</td><td>91.1</td><td>91.8</td><td>55.8</td><td>73.8</td><td>85.8</td><td>58.0</td><td>71.9</td></tr><tr><td>Δ</td><td>+3.5</td><td>+1.8</td><td>+3.5</td><td>+0.9</td><td>+4.0</td><td>+2.1</td><td>+1.0</td><td>+0.5</td><td>+0.8</td><td>+0.5</td><td>+3.5</td><td>+2.0</td></tr><tr><td>Mini-o3 $^{\dagger}$ (Lai et al., 2025)</td><td>47.2</td><td>49.0</td><td>66.7</td><td>88.8</td><td>82.4</td><td>86.3</td><td>90.4</td><td>64.0</td><td>77.2</td><td>88.9</td><td>57.0</td><td>73.0</td></tr><tr><td>Mini-o3 + Ours</td><td>50.5</td><td>51.5</td><td>69.6</td><td>90.8</td><td>86.8</td><td>89.2</td><td>91.5</td><td>65.2</td><td>78.3</td><td>90.2</td><td>59.7</td><td>75.0</td></tr><tr><td>Δ</td><td>+3.3</td><td>+2.5</td><td>+2.9</td><td>+2.0</td><td>+4.4</td><td>+2.9</td><td>+1.1</td><td>+1.2</td><td>+1.1</td><td>+1.3</td><td>+2.7</td><td>+2.0</td></tr><tr><td>Mini-o3 + Ours (7B)</td><td>51.5</td><td>52.2</td><td>71.3</td><td>92.0</td><td>88.0</td><td>90.4</td><td>92.0</td><td>65.2</td><td>78.6</td><td>90.5</td><td>59.9</td><td>75.2</td></tr><tr><td>Δ</td><td>+4.3</td><td>+3.2</td><td>+4.6</td><td>+3.2</td><td>+5.6</td><td>+4.1</td><td>+1.6</td><td>+1.2</td><td>+1.4</td><td>+1.6</td><td>+2.9</td><td>+2.2</td></tr></table>

Results on VisualProbe test. SeProD substantially improves the performance of all baselines on VisualProbe test, which is designed to encourage reflective, trial-and-error reasoning (Lai et al., 2025). Notably, on the easy subset, SeProD achieves improvements of 3.0%, 3.5%, and 2.9% over the three baselines, respectively. On the more challenging hard subset, SeProD yields gains of 3.5% and 3.3% for DeepEyes and Mini-o3, respectively. Due to the presence of numerous distracting elements and open-ended answers, the VisualProbe is substantially more challenging than V\*Bench and HR-Bench. The larger performance gains achieved by SeProD on VisualProbe demonstrate its effectiveness in enhancing the search model’s ability to locate and perceive fine-grained targets in complex visual search scenarios.

Results on V\* Bench and HR-Bench. Performance gains are also observed on multiple-choice benchmarks such as V\* Bench and HR-Bench (4K and 8K), as well as on each of their respective subsets. Our approach improves accuracy across all baselines, confirming its robustness beyond

open-ended reasoning tasks. Notably, SeProD enables both DeepEyes and Mini-o3 to perform particularly well on tasks involving multiple instances. On the Spatial Understanding subset of V\*Bench, it improves performance by 4.0% and 4.4%, respectively, while on the FCP subset of HR-Bench 8K, the corresponding gains are 3.5% and 2.7%. These gains suggest that SeProD effectively leverages the intrinsic spatial reasoning capability of the prophet model to guide inference, thereby enabling the search model to better identify and reason about complex spatial relations. Similarly, the consistent performance improvements observed for Deep-Eyes and Mini-o3 on the FCP subset of the HR Bench-8K benchmark indicate that the prophet model produces more precise outcomes for cross-instance perception. This guidance helps the search model refocus on identifying relationships among multiple instances, leading to more reliable predictions in challenging high-resolution scenarios.

Results on General VQA benchmarks. Using Mini-o3 as the search model, Qwen2.5-VL-3B as the prophet model, we compare the performance of Qwen2.5-VL-7B, Deep-Eyes, Mini-o3, and SeProD on a diverse set of general VQA benchmarks. As shown in Tab. 2, SeProD not only improves the search model’s performance on visual search tasks but also achieves better performance on the general VQA benchmarks. The performance gains on MME-RealWorld and CVBench are particularly notable. By integrating SeProD, the intrinsic capabilities of the search model are further activated and reinforced, resulting in a more substantial improvement on these benchmarks.

Table 2. Results on general VQA benchmarks. Performance comparison of baselines and SeProD on general VQA benchmarks. SeProD consistently improves the visual search LVLM and achieves superior performance. 

<table><tr><td>Method</td><td>MME-RW</td><td>ScienceQA</td><td>OCRBench</td><td>CVBench</td></tr><tr><td>Qwen2.5-VL</td><td>57.3</td><td>69.4</td><td>81.5</td><td>73.9</td></tr><tr><td>DeepEyes</td><td>64.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Mini-o3</td><td>65.5</td><td>84.5</td><td>83.8</td><td>74.4</td></tr><tr><td>Ours</td><td>67.7</td><td>85.4</td><td>85.3</td><td>78.4</td></tr></table>

Table 3. Computational Overhead of SeProD. Acceptance rate of prophetic-prefix tokens under SeProD and the resulting inference speedup relative to the baseline. Results indicate high acceptance rates and slight speedups. 

<table><tr><td rowspan="2">Metric</td><td colspan="3">VisualProbe</td></tr><tr><td>Hard</td><td>Medium</td><td>Easy</td></tr><tr><td>Acceptance rate</td><td>74.6%</td><td>74.2%</td><td>80.7%</td></tr><tr><td>Speedup</td><td>1.06x</td><td>1.03x</td><td>1.07x</td></tr></table>

Computational Overhead. We evaluate the computational cost incurred by SeProD across the entire inference pipeline. Specifically, we conduct an empirical analysis using Minio3 on the VisualProbe test dataset, and report both the proportion of prophetic-prefix tokens accepted by the search model and the speedup ratio. In terms of implementation, we use NVIDIA H20 GPUs to run the search model and the prophet model. For each subset of VisualProbe test, inference is conducted 4 times with a batch size of 4. The results are reported in Tab. 3. Across all subsets of the VisualProbe test, SeProD achieves high acceptance rates without introducing additional computational overhead, owing to the parallel evaluation of prefixes in the search model.

# 4.2. Ablation Study

In this section, we employ Mini-o3 as the base model to conduct ablation experiments, aiming to isolate the effects of individual components in SeProD and examine how varying the scale of the prophet model impacts overall performance.

Component Effectiveness. As shown in Tab. 4, (1) the na¨ıve approach performs worse than the search model itself most of the time, indicating that simply feeding the pretraining model’s output as textual prompts or overriding the post-training model’s current output does not lead to meaningful improvements; instead, it may interfere with or even degrade the model’s existing capabilities. (2) We evaluate Table 4. Ablation results. Effectiveness of individual components in SeProD (left table), as well as the impact of different thresholds (right table). The results show that each component contributes to the performance gains. SeProD achieves the best results, and SeProD is robust to the choice of threshold. “S w/o A” and “S w/o G” denote SeProD without Answer Drafting and Grounding Verification, respectively.

<table><tr><td rowspan="2">Method</td><td colspan="3">VisualProbe</td></tr><tr><td>Hard</td><td>Medium</td><td>Easy</td></tr><tr><td>naïve</td><td>46.2</td><td>47.4</td><td>67.0</td></tr><tr><td>S w/o A</td><td>49.5</td><td>50.1</td><td>68.1</td></tr><tr><td>S w/o G</td><td>49.9</td><td>50.4</td><td>68.7</td></tr><tr><td>Ours</td><td>50.5</td><td>51.5</td><td>69.6</td></tr></table>

<table><tr><td rowspan="2">τ</td><td colspan="3">VisualProbe</td></tr><tr><td>Hard</td><td>Medium</td><td>Easy</td></tr><tr><td>0.2</td><td>50.0</td><td>51.0</td><td>69.0</td></tr><tr><td>0.25</td><td>50.3</td><td>50.9</td><td>69.3</td></tr><tr><td>0.3</td><td>50.5</td><td>51.5</td><td>69.6</td></tr><tr><td>0.35</td><td>50.0</td><td>51.2</td><td>69.2</td></tr></table>

Grounding Verification and Answer Draft independently. Both modules provide complementary strengths. Grounding Verification improves spatial grounding precision and facilitates deeper exploration of task-relevant regions, while the Answer module enhances the semantic plausibility of candidate responses. Their integration yields the largest performance gain, highlighting the importance of jointly verifying the model’s grounding and supplying semantically plausible candidate answers. (3) In addition, we investigate the effect of different threshold values τ on SeProD. The results show that a wide range of reasonable thresholds can consistently yield performance gains. This demonstrates that our approach is robust to hyperparameter choices and that the improvements arise from its intrinsic effectiveness rather than from hyperparameter tuning.

Table 5. Effect of alpha configuration. Fixed α improves the baseline model, but underperforms dynamic α due to its inability to adapt to token-level uncertainty and confidence. 

<table><tr><td rowspan="2">α</td><td colspan="3">VisualProbe</td></tr><tr><td>Hard</td><td>Medium</td><td>Easy</td></tr><tr><td>0.3</td><td>49.3</td><td>49.8</td><td>68.6</td></tr><tr><td>0.4</td><td>48.7</td><td>50.1</td><td>67.9</td></tr><tr><td>0.5</td><td>48.6</td><td>49.6</td><td>67.6</td></tr><tr><td>0.6</td><td>49.1</td><td>50.0</td><td>68.8</td></tr><tr><td>0.7</td><td>48.7</td><td>49.1</td><td>69.0</td></tr><tr><td>0.5 - r</td><td>50.5</td><td>51.5</td><td>69.6</td></tr></table>

Alpha configuration. We compare a dynamically adjusted α with a fixed α. As shown in Tab. 5, a fixed α improves the performance of the baseline model, thanks to the probabilitybased prophetic sampling. However, its gains are smaller than those of a dynamic α, because a fixed α cannot adaptively adjust to the uncertainty of individual tokens, nor can it leverage the search model’s confidence for each token to dynamically allocate weights.

Prophet Model Scale. Tab. 1 further evaluates a 7B prophet model for comparison with the default 3B setting. The 7B prophet yields additional performance gains, particularly on the more challenging VisualProbe benchmark. Beyond the inherent advantage of larger model capacity, a potential contributing factor is that the search model is also initialized from the same 7B model, resulting in closer output distributions between the search model and the prophet model.

# 5. Conclusion

In this paper, we introduce SeProD, a training-free and plug-and-play framework that aligns pre- and post-training LVLMs to address capability degradation and long-context interference in visual search. By leveraging a probabilitybased prophetic interface, SeProD activates precise singlestep abilities while preserving contextual coherence. Experiments show that our approach consistently improves multiple intrinsic-extended LVLMs and achieves state-of-the-art performance across all visual search benchmarks, and further enhances performance on general VQA benchmarks, without introducing additional computational overhead.

# Impact Statement

This work aims to advance the field of machine learning by improving the coherence of multi-step multimodal reasoning in large vision-language models. The proposed framework focuses on inference mechanisms for coordinating intrinsic model capabilities, without introducing new data sources, training procedures, or application-specific objectives. We do not foresee direct negative societal consequences arising uniquely from this work. Potential risks and ethical considerations are therefore aligned with those already well established for Large Vision-Language Models such as issues related to bias, misuse, or over-reliance on automated reasoning systems.

# Acknowledgment

This work is supported by the National Natural Science Foundation of China under Grant No.62576365 and in part by the National Natural Science Foundation of China under Grant No. 62322608.

# References

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang,

X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025b. URL https://arxiv.org/abs/2502.13923.   
Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024a.   
Chen, Z., Zhou, Q., Shen, Y., Hong, Y., Sun, Z., Gutfreund, D., and Gan, C. Visual chain-of-thought prompting for knowledge-based visual reasoning. Proceedings of the AAAI Conference on Artificial Intelligence, 38 (2):1254–1262, Mar. 2024b. doi: 10.1609/aaai.v38i2. 27888. URL https://ojs.aaai.org/index. php/AAAI/article/view/27888.   
Cheng, Z., Chen, Q., Xu, X., Wang, J., Wang, W., Fei, H., Wang, Y., Wang, A. J., Chen, Z., Che, W., and Qin, L. Visual thoughts: A unified perspective of understanding multimodal chain-of-thought, 2025. URL https:// arxiv.org/abs/2505.15510.   
Fan, Y., He, X., Yang, D., Zheng, K., Kuo, C.-C., Zheng, Y., Narayanaraju, S. J., Guan, X., and Wang, X. E. Grit: Teaching mllms to think with images, 2025. URL https://arxiv.org/abs/2505.15879.   
Gao, Z., Chen, Z., Cui, E., Ren, Y., Wang, W., Zhu, J., Tian, H., Ye, S., He, J., Zhu, X., Lu, L., Lu, T., Qiao, Y., Dai, J., and Wang, W. Mini-internvl: A flexibletransfer pocket multimodal model with 5 URL https: //arxiv.org/abs/2410.16261.   
Gu, S., Lugmayr, A., Danelljan, M., Fritsche, M., Lamour, J., and Timofte, R. Div8k: Diverse 8k resolution image dataset. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pp. 3512–3516, 2019. doi: 10.1109/ICCVW.2019.00435.   
Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N. A., and Krishna, R. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models, 2024. URL https://arxiv. org/abs/2406.09403.   
Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C.,

Lo, W.-Y., Dollar, P., and Girshick, R. Segment´ anything, 2023. URL https://arxiv.org/abs/ 2304.02643.   
Lai, X., Li, J., Li, W., Liu, T., Li, T., and Zhao, H. Minio3: Scaling up reasoning patterns and interaction turns for visual search, 2025. URL https://arxiv.org/ abs/2509.07969.   
Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding, 2023. URL https://arxiv.org/abs/2211.17192.   
Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C. Llavaonevision: Easy visual task transfer, 2024. URL https: //arxiv.org/abs/2408.03326.   
Li, G., Xu, J., Zhao, Y., and Peng, Y. Dyfo: A trainingfree dynamic focus visual search for enhancing lmms in fine-grained visual understanding, 2025. URL https: //arxiv.org/abs/2504.14920.   
Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. URL https://arxiv.org/abs/2301.12597.   
Liang, X., Guo, X., Jin, Z., Pan, W., Shang, P., Cai, D., Lin, B., and Ye, J. Enhancing spatial reasoning through visual and textual thinking, 2025. URL https://arxiv. org/abs/2507.20529.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning, 2023. URL https://arxiv.org/abs/2304. 08485.   
Liu, X., Hu, Y., Zou, Y., Wu, L., Xu, J., and Zheng, B. Hide: Rethinking the zoom-in method in high resolution mllms via hierarchical decoupling, 2025. URL https: //arxiv.org/abs/2510.00054.   
Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.-C., Liu, C.-L., Jin, L., and Bai, X. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67 (12), December 2024. ISSN 1869-1919. doi: 10.1007/ s11432-024-4235-6. URL http://dx.doi.org/ 10.1007/s11432-024-4235-6.   
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering, 2022. URL https: //arxiv.org/abs/2209.09513.

Mitra, C., Huang, B., Darrell, T., and Herzig, R. Compositional chain-of-thought prompting for large multimodal models, 2024. URL https://arxiv.org/ abs/2311.17076.   
Shao, H., Qian, S., Xiao, H., Song, G., Zong, Z., Wang, L., Liu, Y., and Li, H. Visual cot: Advancing multimodal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning, 2024. URL https://arxiv.org/abs/2403.16999.   
Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., Xu, R., and Zhao, T. Vlm-r1: A stable and generalizable r1-style large visionlanguage model, 2025a. URL https://arxiv.org/ abs/2504.07615.   
Shen, H., Zhao, K., Zhao, T., Xu, R., Zhang, Z., Zhu, M., and Yin, J. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration, 2025b. URL https://arxiv.org/ abs/2411.16044.   
Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., and Cheng, Y. Openthinkimg: Learning to think with images via visual tool reinforcement learning, 2025. URL https://arxiv.org/ abs/2505.08617.   
Sur´ıs, D., Menon, S., and Vondrick, C. Vipergpt: Visual inference via python execution for reasoning, 2023. URL https://arxiv.org/abs/2303.08128.   
Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S. C., Yang, J., Yang, S., Iyer, A., Pan, X., Wang, Z., Fergus, R., LeCun, Y., and Xie, S. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024. URL https://arxiv.org/abs/ 2406.16860.   
Wang, H., Su, A., Ren, W., Lin, F., and Chen, W. Pixel reasoner: Incentivizing pixel-space reasoning with curiositydriven reinforcement learning, 2025a. URL https: //arxiv.org/abs/2505.15966.   
Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.   
Wang, W., Ding, L., Zeng, M., Zhou, X., Shen, L., Luo, Y., and Tao, D. Divide, conquer and combine: A trainingfree framework for high-resolution image perception in multimodal large language models. arXiv preprint, 2024b. URL https://arxiv.org/abs/2408.15556.

Wang, W., Jing, Y., Ding, L., Wang, Y., Shen, L., Luo, Y., Du, B., and Tao, D. Retrieval-augmented perception: High-resolution image perception meets visual rag, 2025b. URL https://arxiv.org/abs/ 2503.01222.   
Wang, Y., Chen, Q., Li, Z., Wang, S., Guo, S., Zhang, Z., and Wei, Z. Simple o3: Towards interleaved vision-language reasoning, 2025c. URL https://arxiv.org/abs/ 2508.12109.   
Wu, M., Yang, J., Jiang, J., Li, M., Yan, K., Yu, H., Zhang, M., Zhai, C., and Nahrstedt, K. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use, 2025. URL https://arxiv.org/ abs/2505.19255.   
Wu, P. and Xie, S. V\*: Guided visual search as a core mechanism in multimodal llms, 2023. URL https: //arxiv.org/abs/2312.14135.   
Xu, R., Yao, Y., Guo, Z., Cui, J., Ni, Z., Ge, C., Chua, T.-S., Liu, Z., Sun, M., and Huang, G. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images, 2024. URL https://arxiv.org/abs/2403.11703.   
Xu, Y., Li, C., Zhou, H., Wan, X., Zhang, C., Korhonen, A., and Vulic, I. Visual planning: Let’s think only with ´ images, 2025. URL https://arxiv.org/abs/ 2505.11409.   
Yang, S., Li, G., and Yu, Y. Dynamic graph attention for referring expression comprehension, 2019. URL https: //arxiv.org/abs/1909.08164.   
Yu, L., Poirson, P., Yang, S., Berg, A. C., and Berg, T. L. Modeling context in referring expressions. In European conference on computer vision, pp. 69–85. Springer, 2016.   
Zhang, J., Khayatkhoei, M., Chhikara, P., and Ilievski, F. Mllms know where to look: Training-free perception of small visual details with multimodal llms, 2025a. URL https://arxiv.org/abs/2502.17422.   
Zhang, X., Gao, Z., Zhang, B., Li, P., Zhang, X., Liu, Y., Yuan, T., Wu, Y., Jia, Y., Zhu, S.-C., and Li, Q. Adaptive chain-of-focus reasoning via dynamic visual search and zooming for efficient vlms, 2025b. URL https:// arxiv.org/abs/2505.15436.   
Zhang, Y.-F., Zhang, H., Tian, H., Fu, C., Zhang, S., Wu, J., Li, F., Wang, K., Wen, Q., Zhang, Z., Wang, L., Jin, R., and Tan, T. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans?, 2025c. URL https://arxiv. org/abs/2408.13257.

Zheng, G., Yang, B., Tang, J., Zhou, H.-Y., and Yang, S. Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models, 2023. URL https://arxiv.org/abs/2310.16436.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing ”thinking with images” via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.14362.

Zhong, L., Rosenthal, F., Sicking, J., Huger, F., Bagdonat, ¨ T., Gottschalk, H., and Schwinn, L. Focus: Internal mllm representations for efficient fine-grained visual question answering, 2025. URL https://arxiv.org/abs/ 2506.21710.

In this appendix, we provide comprehensive information, including examples and failure cases of SeProD, failure of na¨ıve textual interfaces, details on the benchmarks used in this paper, and implementation details for analysis in the introduction.

• Sec. A - Examples of SeProD   
• Sec. B - Failure Cases of SeProD   
• Sec. C - Failure of Na¨ıve Textual Interfaces   
• Sec. D - More Details on the Benchmarks Used in This Paper   
• Sec. E - Implementation Details for Analysis in the Introduction

# A. Examples of SeProD

In this section, we present some examples illustrating how SeProD leverages the prophet model to help the search model obtain better results. Specifically, Fig. 4 shows the case where the prophet model’s output serves as prophetic prefixes to generate the reasoning token in the next turn. Fig. 5 demonstrates how the search model is guided to perform a further zoom-in to obtain a more accurate answer.

![](images/7550fb8a4f948226528ea194a6fb4f02d53cea8c1c021407eb45e546699ba20e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Search Model Output"] --> B["Turn 0: ...I can see some small figures of people, particularly in the road areas..."]
    A --> C["Grounding: [0.4,0.5,0.8,0.9"]]
    D["Prophet Model Output"] --> E["False. (Need more zoom in)"]
    D --> F["Grounding mode"]
    G["New Search Model Output"] --> H["Turn 0: ...I can see some small figures of people, particularly in the road areas... I need to zoom in further."]
    G --> I["Grounding: [0.5,0.6,0.8,0.9"]]
    J["Turn 1: ...The person appears to be located near the center of this cropped image, slightly to the right."]
    J --> K["Grounding: [0.4,0.3,0.7,0.7"]]
    L["Turn 5: ...I need to get a better view to see their shoes more clearly."]
    L --> M["Grounding: [0.5,0.2,1.0,0.7"]]
    N["Next turn (Turn 6): The user... The person appears in the central region... I will zoom in to get a clearer view..."]
    N --> O["Grounding: [0.2,0.3,0.8,0.8"]]
    P["Question: What color are the shoes of the person in the purple T-shirt?"]
    Q["Ground Truth: White."] --> R["Original Search Model Output"]
    S["The brown text is used to prompt the search model."]
    T["The green text denotes the prophetic prefixes accepted by the search model."] --> U["Prophet Model Output"]
    V["New Search Model Output"] --> W["Turn 0: ...I can see some small figures of people, particularly in the road areas... I need to zoom in further."]
    X["Turn 1: ...I may have zoomed into the wrong area."]
    Y["Turn 5: ...I need to get a better view to see their shoes more clearly."]
    Z["Next turn (Turn 6): The user... The person appears in the central region... I will zoom in to get a clearer view..."]
```
</details>

Figure 4. An example of SeProD. At turn 0, the region localized by the search model through its grounding capability is provided as input, and the prophet model determines that a further zoom-in operation is required, prompting the search model to zoom in more precisely. At turn 1, the prophet model judges that the region obtained by the search model does not contain the target of interest and instructs it to search for an alternative, correct region. At turn 5, the prophet model identifies the target in the image and outputs detailed information (i.e., the person appears in the central region of the image) as prophetic prefixes for the next turn (turn 6).

![](images/7a5316f8a441f2d730ebec299922416051938f8c28838f45129bf0ef59f820c9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Search Model Output"] --> B["Turn 0: ...I'll start by focusing on the upper right quadrant of the image to better see the details. Grounding: [0.5,0.5,1.0,0.8"]]
    A --> C["Turn 5: ...He is wearing a hat that is clearly green in color... The region of image used to answer"]
    A --> D["Turn 6 (an extra turn): ...The man is bent over, wearing a green shirt and a white cap... Evaluate answer from the prophet model in parallel"]
    E["Prophet Model Output"] --> F["False. (Need more zoom in)"]
    E --> G["False. (Need more zoom in)"]
    E --> H["...wearing a white hat. Answering mode"]
    I["New Search Model Output"] --> J["Turn 0: ...I'll start by focusing on the upper right quadrant of the image to better see the details. I need to zoom in further. Grounding: [0.5,0.5,0.8,0.8"]]
    I --> K["Turn 5: ...He is wearing a hat that is clearly green in color... I need to zoom in further. Grounding: [0.6,0.1,0.9,0.7"]]
    I --> L["Turn 6 (an extra turn): ...I get a possible answer from prophet model... Answer: The man in green on the right side of the lawn is wearing a white hat."]
    M["Question: What color hat is the man in green on the right side of the lawn in the upper right corner of the square wearing? Ground Truth: white"] --> N["Original Search Model Output"]
    O["The brown text is used to prompt the search model. The green text denotes the prophetic prefixes accepted by the search model."] --> P["Prophet Model Output"]
    Q["Answer: Green"] --> R["The region of image used to answer"]
    S["Answer: Green"] --> T["The region of image used to answer"]
    U["Answer: Green"] --> V["The region of image used to answer"]
    W["Answer: Green"] --> X["The region of image used to answer"]
```
</details>

Figure 5. An example of SeProD. At turns 0 and 5, the regions localized by the search model through its grounding capability are provided as input, and the prophet model prompts the search model to perform further zoom-in operations. At turn 6, the prophet model takes the final image as input and generates the answer, which is used as prophetic prefixes. When the search model needs to generate the final answer, the prophetic prefixes are evaluated in parallel.

# B. Failure Cases of SeProD

In this section, we present representative failure cases of SeProD, which primarily arise from two sources: grounding errors in the search model and insufficient clarity in cropped image regions.

# B.1. Grounding Errors in the Search Model

As shown in Fig. 6, at the final turn (turn 8), the search model ultimately localizes a misleading and incorrect region for answering, which causes the prophet model to generate its response based only on this erroneous region.

# B.2. Insufficient Clarity in Cropped Image Regions

As shown in Fig. 7, at the final turn (turn 3), the search model localizes the correct region. However, the target text in the image is inherently unclear and ambiguous, which leads the prophet model to produce an incorrect answer.

![](images/46b9b33e115b72cdc0929f17345727271eb875e12037201c4f2dea896bfc3896.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a historic cathedral with twin spires and a clock tower, featuring a central statue and plaza (no signage)
</details>

Question: Is the person sitting on the chair in the lower left corner of the picture an old man or a young man?

Ground Truth: old person

![](images/a01a391c0c8cb6414d6486d41d39c60f994aff1e6e47e0cbe56682bcbf687ade.jpg)

<details>
<summary>natural_image</summary>

Person sitting on a stone bench outdoors, wearing a hat and coat (no visible text or symbols)
</details>

# Original Search Model Output

# Turn 0:

…I will focus my attention on the lower-left quadrant of the image.

Grounding: [0.0,0.5,0.5,1.0]

# Prophet Model Output

![](images/b13aed892a0f0d7f0c6c31e770a92f740984ff829d0735ce8ea24365a99be03b.jpg)

False. (Need more zoom in)

Grounding mode

# New Search Model Output

# Turn 0:

…I will focus my attention on the lower-left quadrant of the image. I need to zoom in fu

Grounding: [0.0,0.6,0.5,1.0]

# Turn 2:

…but the people are still a bit far away and it's not very clear. It seems I have included the wrong area previously…

Grounding: [0.2,0.5,0.7,0.9]

![](images/6d061077b4d5a1fcf555f2bbab76eab843a70dea8c560fe8aee7e6ca479e48fd.jpg)

False. (No main object here)

Grounding mode

# Turn 2:

…I may have zoomed into the wrong area.

Grounding: [0.3,0.4,0.8,0.9]

# Turn 8:

…The statue's features point to a young man.

Evaluate answer from the prophet model in parallel

The region of image used to answer

![](images/e8a7e2021dc9e89eb5d155f7a24227ed95512de6c91a3a98b8e11889c59d4b1b.jpg)

![](images/a2f008f981c19af7ac2f45cba8ab61ff4e961e0c5d476819c3083e4d40817230.jpg)

A young man.

Answering mode

# Turn 8:

…The statue's features point to a young man. I get a possible answer from prophet model

Answer: A young man.

The brown text is used to prompt the search model.

The red text denotes the prophetic prefixes accepted by the search model but wrong.

Figure 6. A failure case of SeProD. At the final turn (turn 8), the search model ultimately localizes a misleading and incorrect region for answering, which causes the prophet model to generate its response based only on this erroneous region.

# C. Failure of Na¨ıve Textual Interfaces

In this section, we present a concrete example illustrating the failure cases of the na¨ıve approach. Fig. 8 covers two scenarios: (1) failure to directly steer the outputs of the post-trained LVLM, and (2) disruption of contextual coherence, which destabilizes multi-step reasoning.

# D. Details on the Benchmarks Used in This Paper

# D.1. Benchmarks for Visual Search Tasks

V\* Bench (Wu & Xie, 2023) is a benchmark constructed on 191 high-resolution images sampled from the SA-1B dataset (Kirillov et al., 2023). For each image, a multiple-choice question is provided, where exactly one option is correct.

![](images/74cef5eb2eade60d8d78d49ce3ae628c27f677c217b38afb130036b5b19ef39a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Search Model Output"] --> B["Turn 0: ...I will start by zooming into the left part of the image to better identify the trees and any potential billboards. Grounding: [0.0,0.3,0.4,0.8"]]
    A --> C["Turn 1: ...I need to zoom in on that red sign to read the text on it. Grounding: [0.0,0.4,0.5,0.8"]]
    D["Prophet Model Output"] --> E["False. (Need more zoom in)"]
    D --> F["Yes. The number is on the..."]
    D --> G["Grounding mode"]
    H["New Search Model Output"] --> I["Turn 0: ...and any potential billboards. I need to zoom in further. Grounding: [0.0,0.4,0.3,0.7"]]
    H --> J["Next turn (Turn 2): The user wants to find the first phone number on a red billboard. In Observation 2, the number is on the central region... Grounding: [0.2,0.3,0.7,0.8"]]
    K["Answer: 138818073"] --> L["Turn 8: ... I get a possible answer from prophet model... Answer: 138818073"]
    M["The brown text is used to prompt the search model. The green text denotes the prophetic prefixes accepted by the search model. The red text denotes the prophetic prefixes accepted by the search model but wrong."] --> N["The region of image used to answer"]
    O["Turn 3: ...The phone number is &quot;138818073&quot;. It is clear and legible and no further grounding is necessary."] --> P["Evaluate answer from the prophet model in parallel"]
```
</details>

Figure 7. A failure case of SeProD. At the final turn (turn 3), the search model localizes the correct region. However, the target text in the image is inherently unclear and ambiguous, which leads the prophet model to produce an incorrect answer.

The benchmark comprises two task categories: attribute recognition and spatial relationship reasoning. The attribute recognition split includes 115 images and targets the identification of object-level properties. The spatial reasoning split contains 76 images and focuses on inferring relative spatial configurations between object pairs.

HR-Bench (Wang et al., 2024b) is bulit from 200 high-resolution images curated from the DIV8K dataset (Gu et al., 2019). The benchmark is evenly partitioned into two subsets: Fine-grained Single-instance Perception (FSP) and Fine-grained Cross-instance Perception (FCP), each containing 100 images. The FSP subset targets fine-grained understanding of individual objects, covering tasks such as attribute identification, visual prompting, and OCR recognition. In contrast, the FCP subset focuses on relational reasoning, including map interpretation, chart analysis, and spatial reasoning. HR-Bench is released in two configurations, namely HR-Bench 8K and HR-Bench 4K. The 8K version retains the original image resolution, whereas the 4K version is obtained by cropping regions centered on target objects. Each image is paired with a single multiple-choice question consisting of four candidate answers and one ground-truth option. The answer options are rotated across images, yielding 800 total samples.

VisualProbe Test (Lai et al., 2025) is divided into three difficulty levels, namely easy, medium, and hard, which contain

# Are the two people sitting under the tree in the middle men or women?

![](images/9641e6440380e86afcb6d563f29f885a2dc8e7dd4b7af15df522d8b0d365937f.jpg)

<details>
<summary>text_image</summary>

Ground Truth: men
</details>

# Situation 1: fail to directly steer

# Final turn:

… which clearly indicates that this person is a woman… I get a possible answer as suggestions from another model: The two people sitting under the tree appear to be men.

# Answer:

One of the people sitting under the tree is a man, and the other one is a woman.

# Situation 2: destabilizing multistep reasoning

# Turn i:

The user… I get a possible answer as suggestions from another model: The two people sitting under the tree appear to be men.

# Turn i+1:

…I may have zoomed in on the wrong area…I need to zoom in again

The green text: a salient and correct answer derived from the pre-training LVLM. The red text: an incorrect answer or action produced by the post-training LVLM.

Figure 8. Failure of na¨ıve textual interfaces. Pre-training LVLM’s output weakly steers the post-training model and breaks coherence across steps, leading to unstable multi-step reasoning.

141, 268, and 106 samples, respectively. Compared to previous visual search benchmarks, VisualProbe places greater emphasis on small target objects and a large number of distracting elements. These characteristics require models to perform iterative exploration and rely on trial-and-error reasoning. Each sample consists of a single image paired with an open-ended question.

# D.2. Benchmarks for General VQA

MME-RealWorld (Zhang et al., 2025c) comprises 13,366 high-resolution images with an average resolution of 2,000 × 1,500 pixels, accompanied by 29,429 annotations spanning 43 sub-tasks.

ScienceQA (Lu et al., 2022) is a benchmark designed for multiple-choice science question answering. It spans three academic domains, including social science, natural science, and language science. The test split comprises 4,241 samples.

OCRBench (Liu et al., 2024) is designed to systematically assess the OCR recognition performance of models. The benchmark is organized into five task categories, covering text recognition, scene Text-Centric VQA, document-Oriented VQA, key information extraction, and handwritten mathematical expression recognition. In total, OCRBench contains 1,000 samples.

CVBench (Tong et al., 2024) is derived from several established vision benchmarks and comprises 2,638 examples. 2D visual understanding is evaluated through spatial relationships and object counting, while 3D visual understanding is assessed using depth order and relative distance.

# E. Implementation Details for Analysis in the Introduction

Examining the intrinsic capabilities of post-training LVLMs. This paragraph provides the implementation details corresponding to Fig. 2(a). We use the RefCOCO (Yu et al., 2016) validation set $\mathsf { R e f C O C O } _ { v a l }$ to evaluate the grounding ability of pre-training and post-training LVLMs. The metric is IoU@0.5. We use the corresponding OCR recognition, spatial understanding, and counting questions from VisualProbe test (Lai et al., 2025) to evaluate the associated capabilities. Specifically, we select the current state-of-the-art Mini-o3 (Lai et al., 2025) as the post-training LVLM. We choose Qwen2.5- VL-7B (Bai et al., 2025b), which is used to initialize Mini-o3, as the pre-training model. For grounding evaluation of Mini-o3, we treat the cropped image used for its final answer as the predicted result. For the comparison of OCR recognition, spatial understanding, and counting abilities between the two models, we first obtain the answers and inference trajectories of Mini-o3 by applying the corresponding VisualProbe test questions. Subsequently, Qwen2.5-VL-7B is provided with the image corresponding to the fifth turn in the Mini-o3 inference trajectory, together with the original question, to obtain its prediction. If the inference trajectory contains fewer than five turns, the final answering turn of Mini-o3 is used instead.

Testing interference in long multi-step reasoning contexts. This paragraph provides the implementation details corresponding to Fig. 2(b). We report the performance of the state-of-the-art Mini-o3 on three subsets of VisualProbe test before and after removing irrelevant context, as well as the corresponding improvements. Specifically, we define irrelevant context as the reasoning rounds in which the model explores erroneously magnified regions. For each sample, given the reasoning trajectory generated by Mini-o3, we remove the irrelevant context and the final answering round. We then feed the revised trajectory back into the model, prompting it to generate a new answering round and obtain a new prediction.

Probability distribution curves. This paragraph provides the implementation details corresponding to Fig. 2(c). We obtain the probability distribution curves for the original visual-search LVLM, the na¨ıve method, and our SeProD. Specifically, we select Mini-o3 as the search model and Qwen2.5-VL-3B (Bai et al., 2025b) as the prophet model, and conduct experiments on the VisualProbe-test dataset. For each method, we compute the average probability of the final answer tokens under the mini-o3 and randomly sample 100 instances to visualize the resulting probability distributions.