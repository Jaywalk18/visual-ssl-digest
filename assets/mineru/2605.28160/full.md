# Look on Demand: A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning

Yang Zhang 1 Xiaoshuai Sun 1 2 Rui Zhao 3 Wujin Sun 4 Yidong Chen 3 4 Jiayi Ji 1 Qian Chen 5 Rongrong Ji 1

# Abstract

Existing multimodal reasoning approaches mainly follow two paradigms: pre-reasoning visual-totext conversion or reasoning in a unified vision–language space. The former compresses finegrained visual details through static textualization, while the latter suffers from linguistic dominance, weakening faithfulness to visual evidence. In this work, we argue that a central challenge is how and when visual evidence is introduced into the reasoning process. Motivated by this insight, we propose a multimodal reasoning framework in which a language model maintains a reasoning state and dynamically schedules visual evidence acquisition, deciding both when to query an independent perception module and when to terminate reasoning. Experiments across multiple multimodal reasoning benchmarks show that CSMR consistently outperforms representative baselines in accuracy under zero-shot settings. Further analysis demonstrates that these gains stem from reasoning-state-driven visual querying and early termination. The code is available at https: //github.com/YangZhang2511/CSMR

# 1. Introduction

In recent years, the rapid advancement of large-scale Vision– Language Models (VLMs) has significantly accelerated research on multimodal reasoning. A central goal of this line of work is to enable models to perform complex reasoning by effectively integrating visual and linguistic infor-

1Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, 361005, P.R. China. 2Sino-Russian ResearchCenter for Digital Economy. 3School of Informatics, Xiamen University, China. 4Institute of Artificial Intelligence, Xiamen University, China. 5School of Information Engineering, Xiamen Ocean Vocational College, Xiamen 361102, China. Correspondence to: Xiaoshuai Sun <xssun@xmu.edu.cn>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

![](images/bef88fdb94842b8f03544139b57a76442b8280de06b4e75297aca77e0122ef74.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) Text-Centric PreReasoning Paradigm
        A["Image"] --> B["Visual-to-Text model"]
        C["Question"] --> D["LLM"]
        D --> E["Final Answer"]
    end

    subgraph (b) Unified Multimodal Representation Paradigm
        F["Image"] --> G["Image Encoding"]
        H["Question"] --> I["Text Encoding"]
        J["LLM"] --> K["Final Answer"]
        L["Question"] --> M["LLM"]
        N["Text Encoding"] --> O["Sufficient evidence"]
        P["LLM"] --> Q["Final Answer"]
        R["Question"] --> S["LLM"]
        T["Text Encoding"] --> U{No: Visual Question}
        V["LLM"] --> W["Final Answer"]
        X["Question"] --> Y["LLM"]
        Z["Text Encoding"] --> AA{Sufficient evidence}
        AB["LLM"] --> AC["Final Answer"]
        AD["Question"] --> AE["LLM"]
        AF["Text Encoding"] --> AG{Sufficient evidence}
        AH["LLM"] --> AI["Final Answer"]
        AJ["Question"] --> AK["LLM"]
        AL["Text Encoding"] --> AM{Sufficient evidence}
        AN["LLM"] --> AO["Final Answer"]
        AP["Question"] --> AQ["LLM"]
        AR["Text Encoding"] --> AS{Sufficient evidence}
        AT["LLM"] --> AU["Final Answer"]
        AV["Question"] --> AW["LLM"]
        AX["Text Encoding"] --> AY{Sufficient evidence}
        AZ["LLM"] --> BA["Final Answer"]
        BB["Question"] --> BC["LLM"]
        BD["Text Encoding"] --> BE{Sufficient evidence}
        BF["LLM"] --> BG["Final Answer"]
        BH["Question"] --> BI["LLM"]
        BJ["Text Encoding"] --> BK{Sufficient evidence}
        BL["LLM"] --> BM["Final Answer"]
        BN["Question"] --> BO["LLM"]
        BP["Text Encoding"] --> BQ{Sufficient evidence}
        BR["LLM"] --> BS["Final Answer"]
        BT["Question"] --> BU["LLM"]
        BV["Text Encoding"] --> BW{Sufficient evidence}
        BX["LLM"] --> BY["Final Answer"]
        BZ["Question"] --> CA["LLM"]
        CB["Text Encoding"] --> CC{Sufficient evidence}
        CD["LLM"] --> CE["Final Answer"]
        CF["Question"] --> CG["LLM"]
        CH["Text Encoding"] --> CI{Sufficient evidence}
        CJ["LLM"] --> CK["Final Answer"]
        CL["Question"] --> CM["LLM"]
        CN["Text Encoding"] --> CO{Sufficient evidence}
        CP["LLM"] --> CQ["Final Answer"]
        CR["Question"] --> CS["LLM"]
        CT["Text Encoding"] --> CU{Sufficient evidence}
        DU["LLM"] --> DV{Sufficient evidence}
        DW["Question"] --> DX["LLM"]
        DY["Text Encoding"] --> DY{Sufficient evidence}
        DY --> DYB{Sufficient evidence}
    end

    subgraph (c) Our Framework
        E
        F
        G
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
        U
    end

    subgraph (c) Our Framework
        M
        N
        O
    end
```
</details>

Figure 1. Illustration of two dominant multimodal reasoning paradigms and our framework.

mation. Depending on how visual evidence is introduced during reasoning, existing approaches can be broadly categorized into two paradigms. The first paradigm converts visual inputs into textualized visual evidence before reasoning, and subsequently performs the entire reasoning process purely in the language space (Yang et al., 2022; Salaberria et al., 2023; Zheng et al., 2023), as illustrated in Fig. 1(a). This design directly leverages the strong reasoning capabilities of large language models (LLMs). However, since the reasoning trajectory has not yet unfolded at the time of one-shot visual-to-text conversion, the initial textualization can only capture coarse semantics, inevitably compressing or discarding fine-grained evidence that may become decisive at later stages (Zhang et al., 2024a; Bigverdi et al., 2025). The second paradigm introduces visual evidence directly into the reasoning process at the representation level (Mitra et al., 2024; Li & Ma, 2025; Lei et al., 2025; Man et al., 2025), enabling end-to-end reasoning within a unified vision–language embedding space, as illustrated in Fig.1(b). By avoiding explicit visual-to-text conversion, this paradigm retains direct access to raw visual representations, thereby preventing the irreversible loss of fine-grained visual evidence, and has become a prevalent approach in recent multimodal reasoning research. However, as discussed in Section 4, this paradigm exhibits a structural bias that causes visual representations to be influenced by linguistic priors, resulting in weakened grounding in image evidence.

Taken together, these observations suggest that the two paradigms correspond to two typical failure modes: either important visual evidence is lost during the conversion of visual inputs into text, or visual representations are undermined in their faithfulness to image evidence. These observations indicate that the central challenge in multimodal reasoning lies in determining when and how visual evidence should be introduced and integrated into the reasoning process. Inspired by the working memory theory of Baddeley et al. (Baddeley, 2020), we propose CSMR, a cognitive scheduling framework for visual evidence acquisition in multimodal reasoning. As illustrated in Fig. 1(c), an LLM serves as the cognitive core that controls the reasoning process and dynamically invokes a VLM during inference to obtain necessary visual evidence. Because the acquisition of visual evidence is explicitly driven by the reasoning state, it is no longer treated as a one-time conversion, but rather as a process that can be repeatedly invoked and progressively validated. Even if the visual evidence returned by the perception module is insufficient, the reasoning core can continue to invoke the perception module until sufficient visual evidence is obtained to support the reasoning process. Meanwhile, the structural decoupling of perception and reasoning prevents the extracted visual evidence from being influenced by linguistic priors. Our contributions are threefold:

1.We analyze a structural limitation in unified multimodal reasoning: visual representations are influenced by linguistic priors, resulting in weakened grounding in image evidence.

2.We propose CSMR, in which a language model maintains an explicit reasoning state and uses it to govern the reasoning process, dynamically invoking an independent visual perception module to acquire task-relevant visual evidence.

3.We conduct extensive evaluations on multiple multimodal reasoning benchmarks, demonstrating consistent improvements in accuracy over representative baselines.

# 2. Related Work

# 2.1. Text-Centric Pre-Reasoning Paradigm

In this paradigm, visual inputs are converted into textual representations prior to reasoning, and all subsequent reasoning is performed purely in the language space without further access to the original images. Early approaches rely on global image descriptions for vision-to-text conversion (Yang et al., 2022; Salaberria et al., 2023), while later works improve task alignment by introducing question-relevant captions (Hu et al., 2023) or more abstract intermediate textual representations, such as candidate-answer-based reasoning (Shao et al., 2023). DDCoT (Zheng et al., 2023) further decomposes problems into sub-questions, queries visual evidence via external tools, and integrates the resulting textual evidence for final reasoning. Despite their effectiveness, these approaches rely on static and pre-planned textualization of visual inputs, preventing the model from incrementally supplementing or revising visual evidence during reasoning, which limits performance on tasks requiring fine-grained or iterative visual verification.

# 2.2. Unified Multimodal Representation Paradigm

This paradigm performs end-to-end reasoning by fusing visual and textual features into a unified multimodal representation space. By jointly modeling both modalities, these methods aim to enhance cross-modal interaction during reasoning. Existing approaches can be roughly grouped into three categories. Early methods, such as T5 (Raffel et al., 2020), fuse visual and textual features but generate text-only reasoning outputs. Subsequent works introduce image-related intermediate structures or prompting signals to guide reasoning in the shared space, including CCoT (Mitra et al., 2024) and SCAFFOLD (Lei et al., 2025). More recent methods further intensify multimodal interaction by jointly generating visual and textual information during reasoning, enabling multimodal reasoning chains, as exemplified by ICoT (Gao et al., 2025) and AIMCoT (Li & Ma, 2025). Despite these advances, this paradigm is prone to hallucinations (Deng et al., 2024; Liu et al., 2025b; Wang et al., 2025; Yang et al., 2025); this paper further analyzes the underlying causes of this issue. As analyzed in Section 4, under joint multimodal training objectives, dominant linguistic priors may bias visual representations toward the language space, undermining faithfulness to the original image evidence.

In addition, some interactive or tool-augmented reasoning approaches (Chen et al., 2023; Yang et al., 2023; Gupta & Kembhavi, 2023) mainly reason by imitating demonstration trajectories, making them essentially trajectory imitationdriven. Thus, both query generation and answer prediction tend to follow demonstrated patterns rather than being guided by the internal reasoning state of the current task.

# 3. Preliminaries

In this paper, we focus on the single-stream paradigm, which is the dominant architecture in contemporary VLMs such as LLaVA (Li et al., 2024) and Qwen-VL (Team, 2025). A typical single-stream VLM consists of a vision encoder $E _ { v }$ and a decoder-only LLM, which process visual and textual information within a unified token sequence. Given an image I and a textual question q, the vision encoder encodes the image into visual embeddings:

$$
x ^ {\text { vis }} = E _ {v} (I). \tag {1}
$$

The question is converted into an instruction-style prompt $\mathrm { P r o m p t _ { i n s t r u c t } }$ and encoded into textual embeddings:

$$
x ^ {\text { text }} = E _ {t} (\text { Prompt } _ {\text { instruct }} (q)). \tag {2}
$$

where $E _ { t } ( \cdot )$ denotes the text embedding function of the language model, which maps the input text into a sequence of embeddings. The resulting visual and textual embeddings are concatenated to form a joint multimodal representation:

$$
X = \left[ x ^ {\text { vis }}; x ^ {\text { text }} \right]. \tag {3}
$$

Finally, L autoregressively generates the answer sequence:

$$
p _ {\Theta} (a _ {t} \mid X, a _ {<   t}) = \mathrm{LLM} (X, a _ {<   t}), \quad t = 1, \dots , T, \tag {4}
$$

producing the final output

$$
a _ {\text { final }} = (a _ {1}, a _ {2}, \dots , a _ {T}). \tag {5}
$$

During answer generation, visual and textual information is integrated solely through the LLM’s self-attention mechanism. Formally, the attention weight assigned to the j-th token at layer ℓ is

$$
A _ {i j} ^ {(\ell)} = \frac {\exp \left(s _ {i j} ^ {(\ell)}\right)}{\sum_ {k \in [ X ; a _ {<   i} ]} \exp \left(s _ {i k} ^ {(\ell)}\right)}, \tag {6}
$$

where the attention score is computed by

$$
s _ {i j} ^ {(\ell)} = \frac {Q _ {i} K _ {j} ^ {\top}}{\sqrt {d}}, \tag {7}
$$

with $Q _ { i }$ denoting the query vector at the i-th decoding position, $K _ { j }$ the key vector of the j-th token, and d the dimensionality of the query and key vectors.

# 4. Empirical Analysis

This section analyzes why the unified multimodal representation paradigm systematically undermines faithful visual grounding in the vision encoder $E _ { v }$ . We show that this issue stems from two structural factors: (i) the training objective lacks explicit constraints on visual faithfulness, and (ii) the language-prior-dominated self-attention mechanism biases the optimization of the vision encoder.

# 4.1. Limitation of the Standard Training Objective

The standard optimization objective does not enforce that the vision encoder $E _ { v }$ produce representations faithfully grounded in the image. Formally, given an input image I and a question q, the training objective is

$$
\Theta^ {*} = \arg \max _ {\Theta} p _ {\Theta} (a _ {\text { final }} \mid I, q), \tag {8}
$$

where $\Theta = \{ \theta _ { v } , \theta _ { l } \}$ denote the parameters of the $E _ { v }$ and $L$ Under this objective, the model may achieve a low training loss without fully relying on visual evidence. This is because the LLM L can leverage strong linguistic priors to produce outputs that align with the target answer $a _ { \mathrm { f i n a l } }$ . To verify this point, we conducted an experiment in the $\mathbf { A } _ { \mathbf { l } }$ ppendix A. The results show that the model retains surprisingly high accuracy even without images, indicating a strong reliance on linguistic priors.

# 4.2. Language-Dominant Attention Causes Misleading Visual Updates

The vision encoder is prone to misleading updates during training because visual tokens have limited influence on the final prediction. This weakness arises from two attentionrelated issues: (i) text tokens consistently gain higher attention weights than visual tokens, and (ii) the reasoning paradigm produces long textual chains that dilute the already limited attention assigned to visual tokens.

Intrinsic Attention Bias Toward Text Tokens. During answer generation, the model concentrates most attention on text tokens while assigning substantially less to visual tokens. This systematic bias originates from the internal attention mechanism of the LLM L. As described in Section 3, when generating the next token $a _ { i + 1 }$ , the model allocates attention over the entire sequence $[ X ; a _ { < i + 1 } ]$ . The attention assigned to visual tokens is:

$$
A _ {\text { vis }} ^ {(\ell)} = \frac {\sum_ {j \in x ^ {\text { vis }}} \exp \left(s _ {i j} ^ {(\ell)}\right)}{\sum_ {j \in x ^ {\text { vis }}} \exp \left(s _ {i j} ^ {(\ell)}\right) + \sum_ {k \in [ x ^ {\text { text }} ; a _ {<   i + 1} ]} \exp \left(s _ {i k} ^ {(\ell)}\right)}. \tag {9}
$$

Visual tokens typically receive lower attention scores for two reasons: (1) the query token $a _ { i }$ originates from the text domain, and its query vector $Q _ { i }$ is optimized during pretraining to better match text tokens; (2) the representation space is shaped by large-scale text corpora, making textual tokens densely structured while visual tokens are comparatively sparse and semantically weaker in this space, which systematically reduces $Q _ { i } – K _ { j }$ similarity.

To substantiate this analysis, we analyze the attention distribution between visual and textual tokens when generating the first answer token in Qwen3-VL-8B on a subset of ScienceQA (Appendix A.2). As shown in Fig. 2, across all 35 Transformer layers, the average pre-softmax attention score on text tokens is consistently higher than that on image tokens (about 2.5× on average). Softmax normalization further magnifies this logit gap, systematically reducing the attention mass assigned to visual tokens (Chen et al., 2024a; Zhang et al., 2025; Zhu et al., 2026). Our findings align with prior work (Deng et al., 2025), which shows that visionlanguage models tend to rely more on textual inputs when they conflict with visual evidence.

![](images/a54b34b1e53a57c9692cba45cb293b0bd3a6cae187a90060004f234f4a1bb90e.jpg)

<details>
<summary>line</summary>

| Layer Index | Mean Attention of Text Tokens | Mean Attention of Image Tokens |
| ----------- | ----------------------------- | ------------------------------- |
| 0           | 0.00024                       | 0.00008                         |
| 1           | 0.00027                       | 0.00009                         |
| 2           | 0.00017                       | 0.00017                         |
| 3           | 0.00010                       | -0.00004                        |
| 4           | 0.00014                       | 0.00006                         |
| 5           | 0.00013                       | 0.00004                         |
| 6           | 0.00012                       | 0.00006                         |
| 7           | 0.00011                       | 0.00004                         |
| 8           | 0.00012                       | 0.00002                         |
| 9           | 0.00011                       | -0.00004                        |
| 10          | 0.00012                       | 0.00002                         |
| 11          | 0.00011                       | 0.00001                         |
| 12          | 0.00012                       | 0.00002                         |
| 13          | 0.00011                       | 0.00003                         |
| 14          | 0.00012                       | 0.00004                         |
| 15          | 0.00011                       | -0.00002                        |
| 16          | 0.00015                       | 0.00008                         |
| 17          | 0.00016                       | 0.00012                         |
| 18          | 0.00015                       | 0.00012                         |
| 19          | 0.00014                       | 0.00012                         |
| 20          | 0.00015                       | 0.00012                         |
| 21          | 0.00013                       | 0.00012                         |
| 22          | 0.00014                       | 0.00012                         |
| 23          | 0.00015                       | 0.00012                         |
| 24          | 0.00014                       | 0.00012                         |
| 25          | 0.00013                       | 0.00012                         |
| 26          | 0.00012                       | 0.00012                         |
| 27          | 0.00013                       | 0.00012                         |
| 28          | 0.00014                       | 0.00-(-)                        |
| 29          | 0.00-(-)                      | -(-)                            |
| 30          | 0.      | -(-)                            |
| 31          | -(-)                          | -(-)                            |
| 32          | -(-)                          | -(-)                            |
| 33          | -(-)                          | -(-)                            |
| 34          | -(-)                          | -(-)                            |
| 35          | -(-)                          | -(-)                            |
</details>

Figure 2. Layer-wise Mean Attention Scores (Text vs. Image). We report the average pre-softmax attention scores of the first generated token across all 35 Transformer layers on a ScienceQA subset. Text tokens consistently receive higher attention than visual tokens, indicating a systematic attention bias toward text.

To rule out the possibility that the above phenomenon is merely an artifact of average-attention dilution caused by the large number of visual tokens, we further aggregate the total pre-softmax attention scores over textual and visual tokens, respectively. The results show that visual tokens still receive lower total pre-softmax attention scores than textual tokens, as reported in Appendix A.2. In addition, to verify that this phenomenon is not specific to the Qwen-VL family, we conduct further experiments on LLaVA-1.6-7B and observe a consistent attention bias toward textual tokens. More details are provided in Appendix A.3.

Attention Dilution Caused by Long Textual Chains. CoT-style generation typically produces long intermediate textual sequences (Zheng et al., 2023; Tan et al., 2024). As more text tokens are appended while the set of visual tokens remains fixed, the denominator in Eq. 9 increases, thereby diluting the relative attention mass assigned to visual tokens. This trend is consistent with recent findings (Chu et al., 2025; Sun et al., 2025; Jian et al., 2025; Liu et al., 2025a).

Consequently, $E _ { v }$ is updated primarily by language-priordominated gradients, as visual tokens exert minimal influence on the generation process.

# 5. Methodology

# 5.1. Overview

We consider the standard multimodal reasoning setting: given an image I and a question q, the goal is to produce a final answer $a _ { \mathrm { f i n a l } }$ grounded in I. As illustrated in Fig. 3, our framework consists of two modules: (1) a Cognitive Reasoning Core (CRC) instantiated as an LLM that conducts iterative reasoning and decides when additional visual evidence is needed; and (2) a Primary Visual Perception Module (PVP) that returns textualized visual evidence from I upon request. During inference, the CRC iteratively issues visual queries to the PVP when necessary, and integrates the returned information to produce $a _ { \mathrm { f i n a l } }$ .

# 5.2. Cognitive Reasoning Core

The CRC serves as the central module of the framework. It is responsible for maintaining the global reasoning state and for determining whether the currently acquired evidence is sufficient to support a final decision, or whether additional perceptual evidence needs to be obtained.

Design of the CRC. We instantiate the CRC as an LLM rather than a VLM for three main reasons. First, from an extensibility perspective, a modality-agnostic reasoning core naturally generalizes to additional perceptual modalities (e.g., video or audio). Second, LLMs possess sufficient capacity to support such cognitive scheduling, as prior studies have shown their ability to actively identify missing information and seek additional evidence during multi-step reasoning under uncertainty (Hu et al., 2024). Finally, at comparable model scales, LLMs typically exhibit stronger abstract reasoning and logical control, whereas VLMs must balance additional cross-modal alignment objectives during training, often leading to trade-offs in pure reasoning capacity (Li et al., 2025; Zhou et al., 2025).

Inputs and Reasoning State. We define a step as one CRC invocation. At step t, the CRC takes as input a fixed instruction prompt $\mathrm { P r o m p t } _ { \mathrm { C R C } } ( q )$ (which contains the question q and task specification) and a reasoning state $h _ { t - 1 }$ that contains all previous interactions. In this work, we represent $h _ { t - 1 }$ as the concatenation of (1) the cumulative reasoning trace $\{ r _ { \tau } \} _ { \tau = 1 } ^ { t - 1 }$ produced by the CRC, and (2) the collection τ=1of textualized visual evidence $\{ a _ { \tau } ^ { v } \} _ { \tau = 1 } ^ { t - 1 }$ returned by the PVP. At t = 1, h0 is empty.

Decision Mechanism. At step t, the CRC produces an intermediate output

$$
r _ {t} = \text { LLM } (\text { Prompt } _ {\text { CRC }} (q), h _ {t - 1})  . \tag {10}
$$

The output $r _ { t }$ expresses one of two intents: (i) issuing a new visual query for additional information, or (ii) producing the final answer to q. We introduce a routing function $g ( \cdot )$ that deterministically parses $r _ { t }$ into a structured decision:

$$
o _ {t} ^ {c} = \left\{ \begin{array}{l l} q _ {t} ^ {v}, & \text { if   } g (r _ {t}) \in \mathcal {Y} _ {\text { query }}, \\ a _ {\text { final }}, & \text { if   } g (r _ {t}) \in \mathcal {Y} _ {\text { answer }}, \end{array} \right. \tag {11}
$$

where $\mathcal { \mathrm { V } } _ { \mathrm { q u e r y } }$ and $\mathcal { V } _ { \mathrm { a n s w e r } }$ are two disjoint sets of parsed outputs corresponding to a visual query and a final answer, respectively. Please note that the routing function is used solely to parse the output of the CRC and does not influence the reasoning process of the CRC. Specifically, we constrain the CRC through prompting to follow the corresponding predefined formats, while $g ( \cdot )$ simply applies regular expression matching to determine whether the output is a query $q _ { t } ^ { v }$ or an answer $a _ { \mathrm { f i n a l } }$ .

![](images/140c42211f8a836e176af614b54a2942111e9031333a6731cad8efc8ace5e957.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text Input: Question: What can we infer about the person in the picture? Options: A. They have 20/20 vision. B. They are sad or angry. C. They are warm."] --> B["1"]
    B --> C["LLM"]
    C --> D{Sufficient evidence in RS?}
    D -->|Yes| E["Final Prediction"]
    D -->|No| F["Visual Query"]
    F --> G["PVP: Consists of a VLM Provides textualized visual evidence upon request from the CRC."]
    G --> H["6"]
    H --> I["CRC: Reasoning State (RS)"]
    I --> J["7"]
    J --> K["8 ..."]
    K --> L["Sufficient evidence in RS?"]
    L --> M["3"]
    M --> N["Updated"]
    N --> O["Image Input: C. They are warm."]
    M --> P["Yes: Generate prediction based on the RS"]
    P --> Q["4"]
    Q --> R["PVP: Consists of a VLM Provides textualized visual evidence upon request from the CRC."]
    R --> S["5"]
    S --> T["Image Input: C. They are warm."]
    
    U["RS-Controlled Reasoning Process"] --> V["At the start of reasoning, CRC issues an initial visual query: What is the dominant color or texture of the person's facial features?"]
    V --> W["PVP generates textualized visual evidence: The person's facial features are dominated by light, pale skin tones ..."]
    W --> X["Update RS from PVP feedback."]
    X --> Y["CRC uses the RS to identify missing evidence and issue a new visual query: Does the person appear to be in a warm environment ?"]
    Y --> Z["When the RS indicates sufficient evidence, CRC outputs the final prediction."]
```
</details>

Figure 3. Overview of the CSMR architecture and its reasoning workflow. The left panel illustrates the overall structure of the CSMR, which consists of a CRC and a PVP. Given an input image and a question, the CRC maintains the current reasoning state and generates targeted visual queries to invoke the PVP when necessary. The PVP independently analyzes the original image and returns textualized visual evidence that answers the issued query. This evidence is then integrated into the CRC’s reasoning state to support subsequent reasoning. The right panel presents a concrete example of reasoning. The CRC progressively generates visual queries based on the current reasoning state. Once the obtained textualized visual evidence is deemed sufficient, the CRC directly produces the final answer.

Reasoning state Update. If $g ( r _ { t } ) \in \mathcal { V } _ { \mathrm { a n s w e r } }$ , the loop terminates and outputs $a _ { \mathrm { f i n a l } }$ . Otherwise, after receiving the PVP evidence $a _ { t } ^ { v }$ , we update the reasoning state by

$$
h _ {t} = \operatorname{update} (h _ {t - 1}, r _ {t}, a _ {t} ^ {v}), \tag {12}
$$

where update(·) aggregates the newly generated trace and visual evidence into the state. In this work, we simply implement update(·) by concatenation, which preserves the full interaction history without introducing additional trainable parameters.

# 5.3. Primary Visual Perception

The PVP provides textualized visual evidence whenever the CRC issues a visual query. It takes as input the query $q _ { t } ^ { v }$ and the image I, which are embedded into a prompting template PromptPVP that instructs the model to answer $q _ { t } ^ { v }$ based on the visual content in I. The PVP then invokes an (independent) VLM to produce:

$$
a _ {t} ^ {v} = \text { VLM } (\text { Prompt } _ {\text { PVP }} (I, q _ {t} ^ {v})) \tag {13}
$$

Textualized evidence is explicit and thus easier to verify than latent embeddings, which facilitates detecting visually inconsistent intermediate outputs.

# 5.4. Reasoning Process

Given $( I , q )$ , the CRC starts from $h _ { 0 } = \emptyset$ and iterates Eq. 10. If $g ( r _ { t } ) \in \mathcal { V } _ { \mathrm { q u e r y } }$ , the CRC issues $q _ { t } ^ { v }$ to the PVP, which returns $a _ { t } ^ { v }$ via Eq. 13; the CRC then updates $h _ { t }$ and proceeds to the next step. The loop terminates when $g ( r _ { t } ) \in \mathcal { 1 }$ answer or when the accumulated context reaches the maximum token budget T max.

Table 1. Main results on M3CoT, ScienceQA, and LLaVA-Bench In-the-Wild (LLaVA-W) with Qwen2 series backbones. 

<table><tr><td>Perception Backbone</td><td>Reasoning Backbone</td><td>Method</td><td>M3CoT ACC.</td><td>ScienceQA ACC.</td><td>LLaVA-W ROUGE-L</td></tr><tr><td rowspan="5">Qwen2-VL-7B</td><td rowspan="5">Qwen2-VL-7B</td><td>No-CoT</td><td>43.6</td><td>56.3</td><td>32.7</td></tr><tr><td>Multimodal CoT (Zhang et al., 2024b)</td><td>40.1</td><td>51.3</td><td>30.7</td></tr><tr><td>CCoT (Mitra et al., 2024)</td><td>43.3</td><td>56.4</td><td>29.4</td></tr><tr><td>SCAFFOLD (Lei et al., 2025)</td><td>41.7</td><td>53.7</td><td>31.8</td></tr><tr><td>ICoT (Gao et al., 2025)</td><td>44.1</td><td>56.8</td><td>34.2</td></tr><tr><td rowspan="3">Qwen2-VL-7B</td><td rowspan="3">Qwen2-7B</td><td>Caption</td><td>40.9</td><td>67.7</td><td>29.1</td></tr><tr><td>DDCoT (Zheng et al., 2023)</td><td>39.0</td><td>71.9</td><td>26.4</td></tr><tr><td>CSMR (Ours)</td><td>45.7</td><td>78.2</td><td>34.3</td></tr></table>

Table 2. Ablation study on the M3CoT benchmark, analyzing the impact of different CRC control strategies on multimodal reasoning performance. All experiments are conducted under the zero-shot setting. 

<table><tr><td>Method</td><td>M3CoT Acc.</td><td>Time s/sample.</td></tr><tr><td>CSMR</td><td>45.7</td><td>24.34</td></tr><tr><td>w/ Single-Query CRC</td><td>40.0</td><td>12.35</td></tr><tr><td>w/ Pre-planned Visual Queries</td><td>40.1</td><td>23.36</td></tr><tr><td>w/ Fixed-Step CRC</td><td>42.7</td><td>72.48</td></tr></table>

# 6. Experiments

# 6.1. Datasets

We evaluate on three representative benchmarks covering multi-step reasoning, science-domain multimodal reasoning, and open-ended instruction following: M3CoT (Chen et al., 2024b), ScienceQA (Saikh et al., 2022), and LLaVA-Bench In-the-Wild (LLaVA-W) (Liu et al., 2024b). M3CoT emphasizes concise, visually grounded multi-step reasoning; ScienceQA focuses on cross-modal scientific problem solving across multiple subjects and grade levels; and LLaVA-W targets open-ended visual understanding with long-form answers annotated by GPT-4V.

# 6.2. Baselines

No-CoT directly generates answers from the given image and question without intermediate reasoning. Caption directly converts the image into a natural language description and conducts reasoning based on the generated caption. CCoT (Mitra et al., 2024) constructs a scene graph from the image to capture object-level and relational details, which is then used as structured input to guide reasoning. DD-CoT (Zheng et al., 2023) decomposes the original question into simpler sub-questions, solved by a VQA model when visual input is needed, and integrates the results for the final prediction. SCAFFOLD (Lei et al., 2025) adopts a non-parametric approach by introducing spatial anchors into images that can be explicitly referenced in language, thereby enhancing the alignment and coordination between visual evidence and textual reasoning. ICoT (Gao et al., 2025) introduces interleaved visual–textual intermediate reasoning steps, enabling fine-grained multimodal reasoning by explicitly grounding textual rationales in image regions.

# 6.3. Implementation Details

Following the experimental settings of prior works (Gao et al., 2025; Li & Ma, 2025), we conduct experiments on the Qwen2 (Yang et al., 2024) series models. Specifically, Qwen2-VL-7B-Instruct is adopted as the perception backbone, while Qwen2-7B-Instruct serves as the reasoning backbone. For baseline methods, the results of No-CoT, CCoT, SCAFFOLD, and ICoT are directly taken from (Gao et al., 2025), where perception and reasoning are handled by a single unified model. Meanwhile, under a unified backbone setting, we implement DDCoT and Caption as comparison baselines. DDCoT is the most closely related to CSMR in terms of methodological paradigm, while the Caption method represents a simple vision-to-text baseline. The prompt template used by CSMR is provided in the Appendix B, while DDCoT adopts the prompt released in its original paper. For the Caption baseline, we employ a VLM to generate a textual description of the image and then answer the question based on the generated caption. All experiments are conducted under a zero-shot setting on two NVIDIA L40 GPUs. Additional implementation details and hyperparameter settings are reported in the Appendix C.

# 6.4. Main Results

Table 1 summarizes the performance of different methods on three multimodal reasoning benchmarks. As shown in the table, CSMR achieves the best performance across all benchmarks. Specifically, compared to ICoT, a strong representative of the unified multimodal representation paradigm, CSMR consistently attains superior results, suggesting that tightly coupled cross-modal fusion may not be a necessary condition for strong multimodal reasoning. Within the textcentric pre-reasoning paradigm, CSMR also significantly outperforms DDCoT and Caption, demonstrating the advantage of cognitively scheduled reasoning over approaches that convert all visual information into text prior to reasoning. Notably, all performance gains are achieved without any additional training or fine-tuning, highlighting the structural benefits of our approach.

Table 3. Results on M3CoT under different perception–reasoning backbone configurations. 

<table><tr><td>Perception Backbone</td><td>Reasoning Backbone</td><td>Method</td><td>M3CoT ACC.</td><td>Time s/sample</td></tr><tr><td rowspan="3">Qwen2-VL-7B</td><td rowspan="3">Qwen3-8B</td><td>Caption</td><td>43.1</td><td>5.2</td></tr><tr><td>DDCoT (Zheng et al., 2023)</td><td>43.3</td><td>52.9</td></tr><tr><td>CSMR (Ours)</td><td>48.1</td><td>20.7</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td rowspan="3">Qwen3-8B</td><td>Caption</td><td>45.2</td><td>6.8</td></tr><tr><td>DDCoT (Zheng et al., 2023)</td><td>50.3</td><td>54.6</td></tr><tr><td>CSMR (Ours)</td><td>53.6</td><td>23.1</td></tr></table>

# 6.5. Ablation Study

To analyze the effect of different design choices, we conduct a series of ablation studies. First, removing dynamic visual querying by restricting the CRC to a single visual query results in a substantial drop in performance, with accuracy decreasing from 45.7% to 40.0%. This indicates that iterative, feedback-driven visual acquisition plays a critical role, and that a single-pass perception strategy is insufficient for the evaluated multi-step reasoning tasks. Second, we restrict the CRC to generate all visual queries for the entire reasoning process at the first step, instead of dynamically issuing queries during later reasoning steps. This constraint leads to a notable performance drop, with accuracy decreasing from 45.7% to 40.1%. This suggests that pre-planned visual queries, without intermediate reasoning feedback, often fail to align with the actual reasoning trajectory, highlighting the importance of dynamically acquiring visual information during step-by-step reasoning. Third, we fix the number of CRC–PVP interaction steps to seven, as the vast majority of samples require fewer than seven interaction steps in practice. Under this setting, performance further degrades, with accuracy dropping to 42.7%. This result indicates that flexible termination is essential for effective multimodal reasoning. It improves inference efficiency by avoiding unnecessary reasoning steps. More importantly, fixing the number of reasoning steps may introduce redundant information that interferes with the reasoning process and ultimately reduces accuracy. Overall, these ablation results demonstrate that the effectiveness of CSMR critically depends on the dynamic control mechanism of the CRC, rather than on static or pre-defined reasoning and querying strategies.

![](images/3917ce738d72395a60dee12665e021b81507551e9c82c91fd930d7a938d9a3e6.jpg)

<details>
<summary>bar_stacked</summary>

| Category | No Hallucination | Hallucination |
| :--- | :--- | :--- |
| CSRV | 0.33 | 0.67 |
| DDCoT | 0.24 | 0.76 |
</details>

Figure 4. Comparison of hallucination rates between DDCoT and CSMR on M3CoT. Hallucinations are identified by GPT-5 based on inconsistencies between generated dialogues and image content. CSMR exhibits a lower hallucination rate than DDCoT.

# 6.6. Hallucination Analysis

This section investigates whether CSMR can effectively reduce hallucinations in multimodal reasoning. To ensure a fair comparison, we focus on DDCoT and CSMR, as both methods involve multiple invocations of the visual perception model during the reasoning process. We adopt M3CoT as the evaluation benchmark because its questions exhibit a strong reliance on visual evidence, thereby reducing the influence of linguistic priors. Specifically, we randomly sample 200 instances from M3CoT using the same random seed as in the main experiments. We then apply DDCoT and CSMR to this subset, respectively, while preserving their full multi-round reasoning dialogues. Finally, we employ GPT-5 as an automatic evaluator to assess each dialogue and identify whether it contains textual descriptions that are inconsistent with the visual evidence in the image, i.e., whether hallucinations occur. The experimental results are shown in Figure 4. As observed, compared to DDCoT, CSMR achieves a 9-percentage-point increase in the proportion of samples without hallucinations, indicating a clear advantage in hallucination control.

<table><tr><td>Text Input:Question:What is the primary business of the truck?Options:A. Transporting clothing.B. Selling food.C. Selling books.D. Transporting cake.</td><td>CSMR:CRC:What is the primary activity of the truck in the image?PVP:The primary activity of the truck in the image is serving customers, as it appears to be a mobile bakery or dessert truck and there are people standing in front of it, possibly ordering food.CRC:Is the primary activity of the truck serving customers or transporting goods?</td></tr><tr><td rowspan="2">Image Input:<img src="images/726b10c56dc696b3ee94e9638369e9d8f3ba443fc9ee5df2fa2ef1a2130b9c5b.jpg"/></td><td>PVP:The primary activity of the truck appears to be serving customers, as it has an open window and items inside, suggesting that it is selling goods, likely cakes.CRC:The Correct Option: B. Selling food</td></tr><tr><td>DDCOT:Sub-questions:(1) What is the type of goods the truck is carrying?(2) What is the purpose of the truck&#x27;s operation?(3) What is the business associated with the goods being transported or sold?Sub-answers:(1) The truck is carrying goods related to baking, as indicated by the sign &quot; Southern Cake Queen&quot;.(2) The purpose of the truck&#x27;s operation is to sell baked goods, specifically cakes, as indicated by the sign on the side of the truck.(3) The business associated with the goods being transported or sold is &quot;Southern Cake Queen&quot;.Rationale:Based on the description and the image provided, the truck is a mobile food truck with a sign that reads &quot; Southern Bake Queen&quot; or &quot; Southern Cake Queen&quot;, indicating that it sells baked goods, specifically cakes.Output:D. Transporting cake.</td></tr></table>

Figure 5. Comparison of reasoning paths between DDCoT and CSMR. CSMR constructs a progressive, evidence-conditioned reasoning trajectory by dynamically generating sub-questions, while DDCoT relies on static and parallel sub-question decomposition, which leads to semantic drift and misaligned decision focus.

# 6.7. Case Study

To further illustrate the difference between the reasoning trajectories induced by CSMR and DDCoT, we analyze a concrete case study. As shown in Fig. 5, CSMR follows a dynamic, evidence-driven reasoning trajectory. Its reasoning process first acquires the necessary visual–semantic information by posing targeted questions, and then formulates more specific reasoning-oriented tasks, such as distinguishing between selling goods and transporting cargo. This questioning strategy forms a clear semantic progression, allowing each reasoning step to progressively narrow the decision space while continuously aligning with the core semantics of the original question, thereby effectively suppressing semantic drift during reasoning. In contrast, DDCoT induces a static reasoning trajectory by committing to a set of parallel questions before any concrete visual evidence is obtained. Lacking conditional dependencies, these questions tend to expand along the same semantic dimension. In this case, the reasoning process remains confined to identifying the type of goods carried by the truck, rather than shifting to the core discriminative dimension of selling versus transporting, which ultimately leads to semantic drift and hinders convergence to the correct conclusion.

# 6.8. Effectiveness Regimes of CSMR

This set of experiments examines how the relative reasoning capability between the perception backbone and the reasoning backbone influences the performance gains of

CSMR over DDCoT and the Caption baseline. We focus on DDCoT because it is the most closely related paradigm, while Caption serves as a simple vision-to-text reference. We consider two backbone configurations. In the first setting, Qwen2-VL-7B is paired with Qwen3-8B, where the reasoning backbone exhibits substantially stronger reasoning capability than the perception backbone. In the second setting, Qwen3-VL-8B is paired with Qwen3-8B; under image–text benchmarks, these two models can be regarded as providing comparable effective reasoning capability (Bai et al., 2025). As shown in Table 3, CSMR outperforms DD-CoT by 4.8 accuracy points under the Qwen2-VL-7B and Qwen3-8B configuration. When switching to Qwen3-VL-8B and Qwen3-8B, this gain narrows to 3.3 points. Across both settings, the Caption baseline consistently yields substantially lower accuracy.

These results suggest that CSMR achieves the greatest performance gains when the reasoning backbone exhibits substantially stronger reasoning capability than the perception backbone. From a structural perspective, under this regime, CSMR centralizes global reasoning control in the reasoning backbone, allowing it to directly govern the acquisition and utilization of perceptual evidence. When the perception backbone already possesses strong autonomous reasoning ability, DDCoT can sufficiently leverage this capability, reducing the additional gains from centralized control. In terms of efficiency, although Caption remains the fastest due to its single-pass design, CSMR is consistently more efficient than DDCoT across both configurations. This efficiency gain stems from the fact that in CSMR, the reasoning backbone actively decides when sufficient information has been gathered and halts further reasoning.

# 7. Conclusion and Future Work

We revisit a fundamental question in multimodal reasoning: how visual evidence should participate in reasoning. Existing paradigms either compress visual evidence before reasoning or weaken it through linguistic dominance in unified representations. We address this with a cognitive scheduling framework that decouples perception from reasoning and dynamically acquires visual evidence during reasoning. Experiments show consistent gains across benchmarks, suggesting that effective multimodal reasoning may depend more on principled evidence scheduling than tighter multimodal fusion. Additionally, the multi-model collaboration and dynamic visual querying in CSMR introduce additional inference overhead. Future work will explore techniques such as quantization to improve efficiency while maintaining reasoning performance (Jin et al., 2025; Ma et al., 2024a;b; 2025).

# Impact Statement

Recent advances in vision-language models (VLMs) and large language models (LLMs) have significantly increased the demand for stronger reasoning capability, better scalability, and lower training cost in multimodal reasoning tasks. This also raises two important concerns: (1) since the current implementation of CSMR is training-free, whether its performance ceiling may be limited in scenarios requiring extremely high accuracy or strong domain adaptation; and (2) compared with the single-pass inference of unified VLMs, whether the decoupled reasoning framework of CSMR may introduce higher inference overhead. In this work, we discuss both of these concerns.

For concern (1), we argue that the “training-free” property is not an intrinsic limitation of CSMR, but rather a design choice of the current implementation. Structurally, CSMR decouples the reasoning module (CRC) from the perception module (PVP), allowing them to be optimized either independently or jointly. In particular, the core capability of CRC is to actively identify missing information and dynamically acquire evidence based on the current reasoning state. Such information-seeking reasoning behavior can be further modeled and optimized through training. Meanwhile, PVP can also be fine-tuned to improve perception accuracy in specific domains. Therefore, CSMR does not fundamentally limit the achievable performance ceiling; instead, it provides a structured reasoning framework that is complementary to parameter learning.

For concern (2), we acknowledge that some unified VLMbased chain-of-thought (CoT) methods are more lightweight in terms of single-pass inference cost. However, the primary goal of CSMR is not to minimize inference cost, but to provide a decoupled capability scaling path. Experimental results show that overall performance can be consistently improved by strengthening the reasoning module while keeping the perception module unchanged. From this perspective, CSMR offers higher training efficiency: when stronger capability is required, only the LLM needs to be upgraded, without retraining the entire VLM, which is typically less costly than training a unified multimodal model of comparable scale. In addition, the decoupled structure also provides strong modality scalability. When introducing new modalities such as video or audio, CSMR only requires attaching the corresponding perception modules, whereas unified models often require joint modeling and retraining across all modalities, leading to substantially higher training costs.

# Acknowledgements

This work was supported by National Key R&D Program of China (No. 2023YFB4502804), the National Natural Science Foundation of China (No. U22B2051, No. U25B2066, No. 62302411).

# References

Baddeley, A. Working memory. Memory, pp. 71–111, 2020.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

Bigverdi, M., Luo, Z., Hsieh, C.-Y., Shen, E., Chen, D., Shapiro, L. G., and Krishna, R. Perception tokens enhance visual reasoning in multimodal language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3836– 3845, June 2025.

Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models, 2024a. URL https: //arxiv.org/abs/2403.06764.

Chen, Q., Qin, L., Zhang, J., Chen, Z., Xu, X., and Che, W. M3cot: A novel benchmark for multi-domain multistep multi-modal chain-of-thought, 2024b. URL https: //arxiv.org/abs/2405.16473.   
Chen, Z., Zhou, Q., Shen, Y., Hong, Y., Zhang, H., and Gan, C. See, think, confirm: Interactive prompting between vision and language models for knowledge-based visual reasoning. arXiv preprint arXiv:2301.05226, 2023.   
Chu, X., Chen, X., Wang, G., Tan, Z., Huang, K., Lv, W., Mo, T., and Li, W. Qwen look again: Guiding visionlanguage reasoning models to re-attention visual information, 2025. URL https://arxiv.org/abs/2505. 23558.   
Deng, A., Chen, Z., and Hooi, B. Seeing is believing: Mitigating hallucination in large vision-language models via clip-guided decoding, 2024. URL https: //arxiv.org/abs/2402.15300.   
Deng, A., Cao, T., Chen, Z., and Hooi, B. Words or vision: Do vision-language models have blind faith in text?, 2025. URL https://arxiv.org/abs/2503.02199.   
Gao, J., Li, Y., Cao, Z., and Li, W. Interleaved-modal chain-of-thought. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 19520– 19529, June 2025.   
Gupta, T. and Kembhavi, A. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14953–14962, 2023.   
Hu, Y., Hua, H., Yang, Z., Shi, W., Smith, N. A., and Luo, J. Promptcap: Prompt-guided image captioning for vqa with gpt-3. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2963–2975, 2023.   
Hu, Z., Liu, C., Feng, X., Zhao, Y., Ng, S.-K., Luu, A. T., He, J., Koh, P. W., and Hooi, B. Uncertainty of thoughts: Uncertainty-aware planning enhances information seeking in large language models, 2024. URL https://arxiv.org/abs/2402.03271.   
Jian, P., Wu, J., Sun, W., Wang, C., Ren, S., and Zhang, J. Look again, think slowly: Enhancing visual reflection in vision-language models, 2025. URL https://arxiv. org/abs/2509.12132.   
Jin, Y., Li, J., Gu, T., Liu, Y., Zhao, B., Lai, J., Gan, Z., Wang, Y., Wang, C., Tan, X., and Ma, L. Efficient multimodal large language models: a survey. Visual Intelligence, 3(1), December 2025. ISSN 2731-9008. doi: 10.1007/s44267-025-00099-6. URL http://dx.doi. org/10.1007/s44267-025-00099-6.

Lei, X., Yang, Z., Chen, X., Li, P., and Liu, Y. Scaffolding coordinates to promote vision-language coordination in large multi-modal models. In Rambow, O., Wanner, L., Apidianaki, M., Al-Khalifa, H., Eugenio, B. D., and Schockaert, S. (eds.), Proceedings of the 31st International Conference on Computational Linguistics, pp. 2886–2903, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025. coling-main.195/.   
Li, B., Zhang, K., Zhang, H., Guo, D., Zhang, R., Li, F., Zhang, Y., Liu, Z., and Li, C. Llavanext: Stronger llms supercharge multimodal capabilities in the wild, May 2024. URL https://llava-vl.github.io/blog/ 2024-05-10-llava-next-stronger-llms/.   
Li, T., Zhang, J., Rao, Y., and Cheng, Y. Unveiling the compositional ability gap in vision-language reasoning model, 2025. URL https://arxiv.org/abs/ 2505.19406.   
Li, X. and Ma, J. Aimcot: Active information-driven multimodal chain-of-thought for vision-language reasoning, 2025. URL https://arxiv.org/abs/2509. 25699.   
Liu, C., Xu, Z., Wei, Q., Wu, J., Zou, J., Wang, X. E., Zhou, Y., and Liu, S. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models, 2025a. URL https://arxiv.org/abs/2505. 21523.   
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024a.   
Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. Llavanext: Improved reasoning, ocr, and world knowledge, 2024b.   
Liu, X., Deng, A., and Zach, C. Energy-guided decoding for object hallucination mitigation, 2025b. URL https: //arxiv.org/abs/2507.07731.   
Ma, Y., Li, H., Zheng, X., Ling, F., Xiao, X., Wang, R., Wen, S., Chao, F., and Ji, R. Affinequant: Affine transformation quantization for large language models. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum? id=of2rhALq8l.   
Ma, Y., Li, H., Zheng, X., Ling, F., Xiao, X., Wang, R., Wen, S., Chao, F., and Ji, R. Outlier-aware slicing for post-training quantization in vision transformer. In Fortyfirst International Conference on Machine Learning,

2024b. URL https://openreview.net/forum? id=Uh5XN9d2J4.   
Ma, Y., Jin, T., Zheng, X., Wang, Y., Li, H., Wu, Y., Jiang, G., Zhang, W., and Ji, R. Ompq: Orthogonal mixed precision quantization, 2025. URL https://arxiv. org/abs/2109.07865.   
Man, Y., Huang, D.-A., Liu, G., Sheng, S., Liu, S., Gui, L.-Y., Kautz, J., Wang, Y.-X., and Yu, Z. Argus: Visioncentric reasoning with grounded chain-of-thought, 2025. URL https://arxiv.org/abs/2505.23766.   
Mitra, C., Huang, B., Darrell, T., and Herzig, R. Compositional chain-of-thought prompting for large multimodal models, 2024. URL https://arxiv.org/ abs/2311.17076.   
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21 (140):1–67, 2020.   
Saikh, T., Ghosal, T., Mittal, A., Ekbal, A., and Bhattacharyya, P. Scienceqa: A novel resource for question answering on scholarly articles. Int. J. Digit. Libr., sep 2022.   
Salaberria, A., Azkune, G., de Lacalle, O. L., Soroa, A., and Agirre, E. Image captioning for effective use of language models in knowledge-based visual question answering. Expert Systems with Applications, 212:118669, 2023.   
Shao, Z., Yu, Z., Wang, M., and Yu, J. Prompting large language models with answer heuristics for knowledgebased visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14974–14983, June 2023.   
Sun, H.-L., Sun, Z., Peng, H., and Ye, H.-J. Mitigating visual forgetting via take-along visual conditioning for multi-modal long CoT reasoning. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5158–5171, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251- 0. doi: 10.18653/v1/2025.acl-long.257. URL https: //aclanthology.org/2025.acl-long.257/.   
Tan, C., Wei, J., Gao, Z., Sun, L., Li, S., Guo, R., Yu, B., and Li, S. Z. Boosting the power of small multimodal reasoning models to match larger models with self-consistency training, 2024. URL https://arxiv.org/abs/ 2311.14109.

Team, Q. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388.   
Wang, C., Chen, X., Zhang, N., Tian, B., Xu, H., Deng, S., and Chen, H. MLLM can see? dynamic correction decoding for hallucination mitigation. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=4z3IguA4Zg.   
Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J., Yang, J., Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P., Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S., Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Liu, X., Fan, Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z., Zhang, Z., Guo, Z., and Fan, Z. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.   
Yang, T., Li, Z., Cao, J., and Xu, C. Understanding and mitigating hallucination in large vision-language models via modular attribution and intervention. In The Thirteenth International Conference on Learning Representations, 2025.   
Yang, Z., Gan, Z., Wang, J., Hu, X., Lu, Y., Liu, Z., and Wang, L. An empirical study of gpt-3 for few-shot knowledge-based vqa. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pp. 3081–3089, 2022.   
Yang, Z., Li, L., Wang, J., Lin, K., Azarnasab, E., Ahmed, F., Liu, Z., Liu, C., Zeng, M., and Wang, L. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.   
Zhang, D., Yang, J., Lyu, H., Jin, Z., Yao, Y., Chen, M., and Luo, J. Cocot: Contrastive chain-of-thought prompting for large multimodal models with multiple image inputs. arXiv preprint arXiv:2401.02582, 2024a.   
Zhang, X., Quan, Y., Shen, C., Yuan, X., Yan, S., Xie, L., Wang, W., Gu, C., Tang, H., and Ye, J. From redundancy to relevance: Information flow in LVLMs across reasoning tasks. In Chiruzzo, L., Ritter, A., and Wang, L. (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 2289–2299, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025. naacl-long.115. URL https://aclanthology. org/2025.naacl-long.115/.

Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models, 2024b. URL https://arxiv.org/ abs/2302.00923.   
Zheng, G., Yang, B., Tang, J., Zhou, H.-Y., and Yang, S. Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 5168–5191. Curran Associates, Inc., 2023.   
Zhou, C., Wang, M., Ma, Y., Wu, C., Chen, W., Qian, Z., Liu, X., Zhang, Y., Wang, J., Xu, H., Luo, F., Chen, X., Hao, X., Li, H., Zhang, A., Wang, W., Zhang, K., Jia, G., Li, L., Lu, Z., Lu, Y., and Guo, Y. From perception to cognition: A survey of vision-language interactive reasoning in multimodal large language models, 2025. URL https://arxiv.org/abs/2509.25373.   
Zhu, Z., Liang, J., Jiang, S., Fu, J., Liu, M., Sun, G., Ng, S.- K., and Qin, B. Analyzing reasoning consistency in large multimodal models under cross-modal conflicts, 2026. URL https://arxiv.org/abs/2601.04073.

Table 4. Accuracy on the selected ScienceQA subset (232 samples, grades 8–12) under different input configurations. 

<table><tr><td>Backbone</td><td>Question</td><td>Option</td><td>Image</td><td>Hint</td><td>Accuracy</td></tr><tr><td rowspan="4">Qwen3-VL</td><td>√</td><td>√</td><td>√</td><td>√</td><td>92.67%</td></tr><tr><td>√</td><td>√</td><td>√</td><td>✗</td><td>81.90%</td></tr><tr><td>√</td><td>√</td><td>✗</td><td>√</td><td>68.10%</td></tr><tr><td>√</td><td>√</td><td>✗</td><td>✗</td><td>57.33%</td></tr></table>

# A. Supplementary Experiment on ScienceQA

# A.1. Linguistic-Prior-Dominated Training

To validate our claim that a VLM can achieve a low training loss primarily by relying on linguistic priors, we conduct a controlled empirical study on a subset of the ScienceQA dataset. Specifically, we select 232 samples from grades 8–12 that include an image, question, option, and hint text. The hint text provides auxiliary textual information to help the model understand the question. Compared to lower-grade questions, these samples typically involve more complex reasoning and are more likely to require visual evidence, making them a more challenging testbed for language-only inference.

We adopt Qwen3-VL-8B, a recent large scale vision–language model, as the backbone to ensure that the observed behavior is not an artifact of limited model capacity. We evaluate four input configurations by selectively removing image and/or hint information. As shown in Table 4, even when no visual input is provided, the model achieves 57.33% accuracy without hints and 68.10% accuracy with hints, both substantially above random chance.

This observation suggests that, under current standard supervised training objectives, models may reduce training loss by relying heavily on linguistic priors, even in the absence of visual inputs. Consequently, during joint multimodal training, there is limited incentive for the visual encoder to maintain strict faithfulness to image evidence.

![](images/88b8959f279eaba33a07341d377ec3970ea08db1c40c7fc62553e48833f9011a.jpg)

<details>
<summary>line</summary>

| Layer Index | Total Attention of Text Tokens | Total Attention of Image Tokens |
| ----------- | ------------------------------ | -------------------------------- |
| 1           | 950                            | 220                              |
| 2           | 670                            | 200                              |
| 3           | 1080                           | 450                              |
| 4           | 360                            | -100                             |
| 5           | 450                            | -50                              |
| 6           | 570                            | 100                              |
| 7           | 540                            | 180                              |
| 8           | 170                            | -50                              |
| 9           | 440                            | 120                              |
| 10          | 70                             | -150                             |
| 11          | 310                            | 20                               |
| 12          | 270                            | 30                               |
| 13          | 420                            | 130                              |
| 14          | 290                            | -20                              |
| 15          | 420                            | 100                              |
| 16          | 230                            | -50                              |
| 17          | 610                            | 210                              |
| 18          | 310                            | 30                               |
| 19          | 610                            | 240                              |
| 20          | 540                            | 180                              |
| 21          | 450                            | 130                              |
| 22          | 530                            | 160                              |
| 23          | 550                            | 140                              |
| 24          | 590                            | 220                              |
| 25          | 520                            | 170                              |
| 26          | 430                            | 130                              |
| 27          | 500                            | 160                              |
| 28          | 390                            | 80                               |
| 29          | 610                            | 250                              |
| 30          | 660                            | 300                              |
| 31          | 620                            | 280                              |
| 32          | 510                            | 210                              |
| 33          | 570                            | 270                              |
| 34          | 360                            | 150                              |
| 35          | 240                            | -5                               |
</details>

Figure 6. Layer-wise total pre-softmax attention scores allocated to text tokens and visual tokens during the generation of the first output token. Results are averaged over all attention heads and all samples in the ScienceQA subset using Qwen3-VL-8B. Across most layers, text tokens receive substantially higher attention mass than visual tokens, revealing a strong text-dominant attention bias during answer generation.

# A.2. Attention Distribution During Answer Generation

To further analyze the model’s internal attention distribution during answer generation, we conduct an attention study on the ScienceQA subset selected in the above section, using the recent large-scale vision–language model Qwen3-VL-8B. Specifically, we focus on the attention allocation of the self-attention module over the input sequence when generating the first output token in the autoregressive decoding stage.

![](images/bd7856323fdae147d13185d4a458d2b371a2ecfeed06e33b632290e79f000834.jpg)

<details>
<summary>line</summary>

| Layer Index | Mean Attention of Text Tokens | Mean Attention of Image Tokens |
| ----------- | ----------------------------- | ------------------------------- |
| 0           | 1.18                          | 0.16                            |
| 1           | 1.72                          | 0.10                            |
| 2           | 0.55                          | 0.12                            |
| 3           | 0.35                          | 0.11                            |
| 4           | 0.40                          | 0.10                            |
| 5           | 0.50                          | 0.09                            |
| 6           | 0.65                          | 0.08                            |
| 7           | 0.63                          | 0.08                            |
| 8           | 0.78                          | 0.07                            |
| 9           | 1.02                          | 0.06                            |
| 10          | 1.02                          | 0.06                            |
| 11          | 1.18                          | 0.06                            |
| 12          | 1.20                          | 0.06                            |
| 13          | 1.15                          | 0.05                            |
| 14          | 1.55                          | 0.05                            |
| 15          | 1.30                          | 0.05                            |
| 16          | 1.28                          | 0.05                            |
| 17          | 0.93                          | 0.06                            |
| 18          | 0.92                          | 0.06                            |
| 19          | 0.92                          | 0.06                            |
| 20          | 0.96                          | 0.06                            |
| 21          | 0.89                          | 0.06                            |
| 22          | 0.73                          | 0.06                            |
| 23          | 0.56                          | 0.06                            |
| 24          | 0.87                          | 0.06                            |
| 25          | 0.54                          | 0.06                            |
| 26          | 0.84                          | 0.06                            |
| 27          | 0.51                          | 0.06                            |
| 28          | 0.53                          | 0.06                            |
| 29          | 0.65                          | 0.06                            |
| 30          | 0.44                          | 0.05                            |
| 31          | 1.25                          | 0.03                            |
</details>

Figure 7. Layer-wise Mean Attention Scores (Text vs. Image). We report the average attention scores for the first generated token across all 32 Transformer layers on a subset of ScienceQA using LLaVA-1.6-7B. Text tokens consistently receive higher attention than visual tokens, indicating a systematic attention bias toward text.

For each sample, we concatenate the question, image, hint, and options as the model input, and extract the self-attention weights from every transformer layer. We then average the attention matrices across all attention heads within each layer to obtain a layer-level attention distribution; additionally, we average across the sample dimension to reduce variance due to individual example differences. Finally, we partition the input tokens by modality into visual tokens and text tokens, and compute the mean attention score for each group.

Meanwhile, we aggregate the total pre-softmax attention scores over text tokens and visual tokens at each transformer layer. Fig. 6 shows the layer-wise attention distribution averaged over all samples, where text tokens consistently receive higher total attention than visual tokens.

# A.3. Generalizability Verification of Attention Distribution Analysis

To examine whether the above attention analysis is affected by the visual token compression mechanism in the Qwen-VL family, we further conduct experiments on LLaVA-1.6-7B (Liu et al., 2024a). Since Qwen-VL models compress visual tokens and thus substantially reduce the number of visual tokens, the observed attention patterns may partly depend on this compression design. In other words, whether the same conclusion holds for models without visual token compression remains to be further verified. Unlike Qwen-VL models, LLaVA-1.6-7B does not adopt a comparable visual token compression mechanism, and therefore its input sequence contains a much larger number of visual tokens.

It is worth noting that the pre-softmax attention scores in LLaVA-1.6-7B are unnormalized logits and can take negative values. Therefore, directly summing the pre-softmax scores over tokens from different modalities does not admit a clear probabilistic interpretation. For this reason, we instead analyze the average attention score per token. Meanwhile, to investigate how the difference in token counts affects the overall attention allocation, we further report the total attention scores assigned to text tokens and visual tokens.

The experimental results are shown in Fig. 7 and Fig. 8. Specifically, Fig. 7 reports the average attention score per token, while Fig. 8 shows the total attention scores assigned to text and visual tokens. From the perspective of per-token attention, text tokens receive substantially higher attention than visual tokens, indicating that the model still tends to focus more on textual information at the unit-token level. From the perspective of total attention, although the number of visual tokens is much larger than that of text tokens, visual tokens dominate the total attention only in the early layers. As the layer depth increases, the total attention assigned to text tokens gradually surpasses that assigned to visual tokens. These results further suggest that, even in models without visual token compression, a clear text-oriented attention bias still emerges.

Layer-wise Total Attention Scores (Text vs. Image)   
![](images/35d050a21e34a9d696cfe4f607d2a84567bc1e2941ad1d6a979dccf42b563cbc.jpg)

<details>
<summary>line</summary>

| Layer Index | Total Attention of Text Tokens | Total Attention of Image Tokens |
| ----------- | ------------------------------ | ------------------------------- |
| 0           | 0.47                           | 0.53                            |
| 1           | 0.68                           | 0.30                            |
| 2           | 0.22                           | 0.41                            |
| 3           | 0.14                           | 0.38                            |
| 4           | 0.16                           | 0.32                            |
| 5           | 0.20                           | 0.30                            |
| 6           | 0.26                           | 0.26                            |
| 7           | 0.27                           | 0.27                            |
| 8           | 0.31                           | 0.23                            |
| 9           | 0.41                           | 0.18                            |
| 10          | 0.41                           | 0.18                            |
| 11          | 0.47                           | 0.17                            |
| 12          | 0.48                           | 0.16                            |
| 13          | 0.46                           | 0.15                            |
| 14          | 0.62                           | 0.14                            |
| 15          | 0.52                           | 0.15                            |
| 16          | 0.51                           | 0.15                            |
| 17          | 0.37                           | 0.21                            |
| 18          | 0.36                           | 0.21                            |
| 19          | 0.36                           | 0.21                            |
| 20          | 0.38                           | 0.21                            |
| 21          | 0.36                           | 0.21                            |
| 22          | 0.29                           | 0.22                            |
| 23          | 0.35                           | 0.19                            |
| 24          | 0.22                           | 0.21                            |
| 25          | 0.34                           | 0.20                            |
| 26          | 0.33                           | 0.21                            |
| 27          | 0.21                           | 0.20                            |
| 28          | 0.26                           | 0.19                            |
| 29          | 0.18                           | 0.18                            |
| 30          | 0.50                           | 0.07                            |
| 31          | -                              | -                               |
</details>

Figure 8. Layer-wise Sum Attention Scores (Text vs. Image). We report the total attention scores for the first generated token across all 32 Transformer layers on a subset of ScienceQA using LLaVA-1.6-7B. In most layers, text tokens receive higher attention scores than visual tokens, indicating a systematic attention bias toward text.

# B. Prompt Templates

The prompt template of the CRC is shown in Listing 1. The PVP answers the visual queries generated by the CRC by directly analyzing the image. Since the PVP does not involve decision-making or interaction control, we omit its prompt for brevity.

# Listing 1. CRC Prompt Template

[Role and Input]   
You are solving a multimodal reasoning task and are required to actively acquire visual evidence from an image by asking visual questions.   
You will receive:   
- An input question related to the image.   
[Phase 0: Integrated Problem Analysis]   
In this phase, analyze the input question to identify the key visual evidence required for answering it.   
[Phase 1: Visual Questioning]   
In this phase, ask visual questions about the image to obtain information necessary for answering the input question.   
Each question should be grounded in the image content and aim to reduce uncertainty in the reasoning process.   
After receiving the corresponding visual evidence, incorporate it into your latest analysis.   
[Phase 2: Answer Decision]   
Based on the updated analysis, determine whether the available visual evidence is sufficient to answer the input question with confidence.   
- If not, return to Phase 1 and ask another visual question.   
- If so, output the final prediction to the input question.

# C. Hyperparameter Settings

Table 5. Hyperparameter settings for reasoning and perception backbones. 

<table><tr><td>Hyperparameter</td><td>Reasoning Backbone</td><td>Perception Backbone</td></tr><tr><td>Temperature</td><td>0.3</td><td>0.7</td></tr><tr><td>Top-p</td><td>0.9</td><td>0.9</td></tr><tr><td>Top-k</td><td>30</td><td>80</td></tr><tr><td>Max tokens</td><td>2048</td><td>512</td></tr><tr><td> $T_{max}$ </td><td>6000</td><td>6000</td></tr><tr><td>Repetition penalty</td><td>1.0</td><td>1.0</td></tr><tr><td>Max model length</td><td>8192</td><td>8192</td></tr></table>