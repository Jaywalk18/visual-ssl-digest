# LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams

Zhenyu Yang , Kairui Zhang , Bing Wang , Shengsheng Qian , Member, IEEE, and Changsheng Xu , Fellow, IEEE

Abstract—Despite the remarkable progress of Video Large Language Models (Video-LLMs), current online architectures still struggle to simultaneously process continuous video streams, decide autonomously when to respond, and preserve long-horizon contextual memory. These obstacles undermine real-time responsiveness and cause severe forgetting throughout prolonged interactions. In this work, we introduce LiveStarPro, a live streaming assistant that is designed for proactive video understanding over long-horizon streams. The design of LiveStarPro rests on three complementary components. The first component is Streaming Verification Decoding (SVeD), an inference framework that identifies the appropriate response timing through singlepass perplexity verification, thereby eliminating the dependency on explicit silence tokens. The second component is Streaming Causal Attention Masks (SCAM), a training strategy that enforces incremental video-language alignment over variablelength streams so that dynamic verification becomes feasible. The third component is Tree-Structured Hierarchical Memory (TSHM), a recursive memory architecture that organizes evicted historical information into event chains and consequently enables efficient retrieval from effectively unbounded video streams without saturating the active context window. To facilitate a comprehensive evaluation under realistic online conditions, we further present OmniStarPro, a large-scale benchmark that spans 15 diverse real-world scenarios and that extends to hourscale streams for the assessment of long-term recall. Extensive experiments demonstrate that LiveStarPro consistently surpasses existing methods, attaining a 28.9% improvement in semantic correctness and an 18.2% reduction in timing error relative to prior online Video-LLMs, while its streaming key-value cache further yields a 1.58× inference speedup over the same model without caching. The model and the code are publicly available at https://github.com/sotayang/LiveStarPro.

Index Terms—Online video understanding, multimodal large language models, streaming inference, vision-language benchmarks.

## I. INTRODUCTION

HE swift advancement of Large Vision-Language Models (LVLMs) [1]–[5] has substantially transformed the field of Video Large Language Models (Video-LLMs) [6]– [10]. Recent architectures exhibit notable competence across complex multimodal tasks, owing to sustained progress in spatial-temporal reasoning [11]–[13], memory-augmented processing [14]–[16], and long-context comprehension [17]–[19]. Nevertheless, these achievements have been obtained predominantly under offline settings, in which models operate over finite and pre-recorded video sequences.

Driven by such progress, growing research attention has been directed toward live streaming assistants for online video understanding [20]–[25]. Unlike their offline counterparts, these agents must remain always-on, reason continually as time unfolds, and deliver feedback in real time. As depicted in Fig. 1(a), a central difficulty in this setting arises from the need to process continuous frame-by-frame inputs while autonomously deciding the most appropriate moment to respond [25]–[27]. To operate effectively, such models must be endowed with proactive temporal decision-making, so that responses are emitted only at contextually relevant moments rather than as repetitive outputs or default refusals (e.g., “I do not know”) for every incoming frame. As an early attempt, VideoLLM-online [20] proposed a streaming EOS (End-Of-Sequence) prediction mechanism (Fig. 1(b)) that continually consumes the video stream and emits EOS tokens to mark silence intervals conditioned on user queries. Building upon this idea, subsequent studies such as VideoLLM-MoD [21] and LION-FS [22] refined the EOS-based framework to improve computational efficiency and response accuracy.

Despite these improvements, the EOS-driven paradigm remains fundamentally limited, giving rise to four structural deficiencies. (1) Severe Data Imbalance: The frames requiring silence far outnumber those triggering a response [23]. A 90-second stream sampled at 2 FPS produces 180 frames; if only 6 events deserve a spoken response, the remaining 174 frames map to silence, a response-to-silence ratio of approximately 1:29. (2) Temporal Inconsistency: As shown in Fig. 1(b), temporally adjacent and visually near-identical frames frequently receive contradictory targets, where one triggers a detailed narration while the next demands an immediate EOS, hindering convergence during fine-tuning. (3) Objective Misalignment: Standard pre-training optimizes semantic alignment between visual features and textual descriptions, whereas the silence mechanism forces a mapping from rich visual evidence to a null EOS token, conflicting with the meaningful cross-modal correspondence underlying pre-training. (4) Vocabulary Degradation: Treating the EOS token as an ordinary vocabulary entry lets its high frequency contaminate the semantic space, introducing ambiguity and distorting the natural probability distribution over meaningful tokens. Together, these deficiencies impede optimization and weaken video-language alignment, ultimately eroding the core video understanding capability of the model. We therefore contend that silence should not be a learned prediction target, but a derived state verified through the model’s confidence. It shifts the central problem from predicting silence to verifying relevance, leading to Challenge 1: How to establish an effective response-silence framework during both training and inference while preserving video-language alignment?

![](images/a47a0535d53608adc2c227c852bde8058ff070fa5cfaa4c98d18141243c27f18.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph a_Real_time_Narration["(a) Real-time Narration"]
  A1["Current Frame"] --> B1["Deco"]
  B1 --> C1["Inference"]
  C1 --> D1["Temporal Inconsistency"]
  D1 --> E1["Training"]
  E1 --> F1["Video-LLM"]
  F1 --> G1["Data Imbalance"]
  G1 --> H1["Deco"]
  H1 --> I1["Inference"]
    end

    subgraph b_Existing_Methods["(b) Existing Methods"]
  J1["Current Frame"] --> K1["Deco"]
  K1 --> L1["Inference"]
  L1 --> M1["Decode"]
  M1 --> N1["Narration"]
    end

    subgraph c_LiveStarPro_Ours["(c) LiveStarPro (Ours)"]
  O1["LiveStarPro"] --> P1["Pre-trained Consistent"]
  P1 --> Q1["Mask"]
  Q1 --> R1["Training"]
  R1 --> S1["Narration"]
  S1 --> T1["Mask"]
  T1 --> U1["Training"]
  U1 --> V1["Mask"]
  V1 --> W1["Mask"]
  W1 --> X1["Mask"]
  X1 --> Y1["Mask"]
  Y1 --> Z1["Mask"]
  Z1 --> AA["Mask"]
  AA --> AB["Mask"]
  AB --> AC["Mask"]
  AC --> AD["Mask"]
  AD --> AE["Mask"]
  AE --> AF["Mask"]
  AF --> AG["Mask"]
  AG --> AH["Mask"]
  AH --> AI["Mask"]
  AI --> AJ["Mask"]
  AJ --> AK["Mask"]
  AK --> AL["Mask"]
  AL --> AM["Mask"]
  AM --> AN["Mask"]
  AN --> AO["Mask"]
  AO --> AP["Mask"]
  AP --> AQ["Mask"]
  AQ --> AR["Mask"]
  AR --> AS["Mask"]
  AS --> AT["Mask"]
  AT --> AU["Mask"]
  AU --> AV["Mask"]
  AV --> AW["Mask"]
    end

    subgraph d_Training_Perspective["(d) Training Perspective"]
  AX["Training Perspective"] --> AY["Nar"]
  AY --> AZ["Nar"]
  AZ --> BA["Nar"]
  BA --> BB["Nar"]
  BB --> BC["Nar"]
  BC --> BD["Nar"]
  BD --> BE["Nar"]
  BE --> BF["Nar"]
  BF --> BG["Nar"]
    end

    subgraph e_Full_Narration["(e) Full Narration"]
  BH["Full Narration"] --> BI"At the start, a gray kitten is seen running across a green grassy field. Shortly after, another gray kitten enters the frame and approaches the first one. Then, the scene shifts, and they are now indoors.""
    end
```
</details>

Fig. 1. Illustration of online video understanding. (a) Taking the RNG task as an example, online video understanding requires Video-LLMs to continuously process unbounded video streams and respond only at appropriate moments. (b) Existing EOS-based methods suffer from data imbalance and temporal inconsistency, leading to unstable training and suboptimal online inference. (c)-(e) LiveStarPro establishes an effective proactive response-silence framework through training (SCAM), inference (SVeD), and memory (TSHM), enabling coherent and context-aware real-time narration without compromising basic video understanding capabilities.

Beyond response timing, deploying an always-on assistant exposes fundamental limitations in the memory management of existing online Video-LLMs, which predominantly rely on simple sliding window mechanisms suffering from three inherent obstacles. (1) Context Window Saturation: Irrespective of the context size (e.g., 8192 tokens), a continuous and potentially unbounded stream eventually exceeds the model capacity, forcing a trade-off between recent observations and longterm history. (2) Catastrophic Forgetting: Common eviction strategies such as First-In-First-Out (FIFO) discard historical content indiscriminately once the buffer is full, permanently removing access to past events and preventing answers to queries about distant history (e.g., “What did the person pick up an hour ago?”). (3) Retrieval Inaccessibility: Existing streaming architectures treat the context buffer as a passive container without an explicit mechanism to recall evicted information. Consequently, historical events outside the active window become inaccessible during inference, preventing the model from relating current observations to relevant long-term context. These obstacles motivate Challenge 2: How can longhorizon video streams be processed with efficient long-term memory retrieval while maintaining contextual consistency beyond a fixed context window?

In parallel, the progress of online Video-LLMs is further constrained by the limited diversity of training data and the narrow scope of existing evaluation benchmarks, which jointly fail to capture the full spectrum of real-world streaming applications. Most representative models [20]–[22] rely heavily on first-person videos drawn from Ego4D [28]. Although StreamMind [23] broadens the domain to sports through SoccerNet [29], the overall coverage of real-world contexts remains sparse. A comparable limitation is observed in current evaluation protocols. Recent benchmarks such as SVBench [30], OVO-Bench [31], and StreamBench [27] have introduced synchronous evaluation settings, yet they remain largely restricted to video question answering. As a result, they leave a broad range of practical online tasks unassessed, including live streaming narration, temporal grounding, and multi-turn interactive understanding.

To address these challenges, we present LiveStarPro, a proactive live streaming assistant for long-horizon video streams across diverse scenarios. LiveStarPro produces context-aware responses at semantically appropriate moments by combining adaptive streaming decoding with hierarchical memory management. For Challenge 1, we establish a proactive response-silence paradigm coupling two synergistic innovations: a Streaming Verification Decoding (SVeD) mechanism that uses single-pass verification to determine the optimal response timing and suppress redundant outputs through strategic silence, and a stream-oriented training strategy built on Streaming Causal Attention Masks (SCAM) that aligns progressively revealed frames with their captions to instill the incremental video-language alignment SVeD requires. For Challenge 2, we propose a Tree-Structured Hierarchical Memory (TSHM): a Peak-End strategy distills the active context by prioritizing salient keyframes via perplexity verification, while evicted history is offloaded into a recursive event tree that attaches new events as children of semantically similar nodes, thereby modeling the temporal progression and causal dependencies of the stream; when a response is triggered, a context-aware retrieval mechanism re-injects relevant event chains of visual tokens and captions to support long-term reasoning. Finally, to mitigate the data limitations, we introduce OmniStarPro, a comprehensive dataset for training and benchmarking that encompasses diverse real-world scenarios and evaluation tasks for online video understanding, and that augments the short-horizon setting with a long-form partition dedicated to long-term memory recall. Extensive experiments across three benchmarks demonstrate that LiveStarPro attains state-of-the-art results for online video understanding.

The preliminary version of our work has been published in the proceedings of the Conference on Neural Information Processing Systems (NeurIPS) 2025 [32]. This journal version extends it as follows. (1) Whereas the preliminary version targets instantaneous response timing over minute-scale streams, this version advances toward long-horizon online understanding, where an always-on assistant must process effectively unbounded hour-scale streams without succumbing to catastrophic forgetting, coupling proactive response timing with structured long-term memory for coherence over prolonged interactions. (2) We substantially extend the memory design: the Peak-End compression is reformulated as the Short-Term Working Memory of a more complete Tree-Structured Hierarchical Memory, which adds a Long-Term Retrieval Memory organized as a Recursive Event Tree, a memory-augmented generation mechanism based on hierarchical beam descent, and a theoretical analysis establishing bounded active memory and sublinear retrieval over unbounded streams. (3) We extend OmniStar to OmniStarPro by curating a new longform partition (OmniStarPro-Long) of videos ranging from ten minutes to beyond one hour and three memory-centric tasks that probe recall of evicted historical content, in addition to the original 15 real-world scenarios and 5 shorthorizon tasks. (4) We additionally update the experiments with evaluations on long-term recall, ablations of memory compression and retrieval, and efficiency analyses that raise the average semantic-correctness improvement from 19.5% to 28.9% over existing online Video-LLMs, expand the related work in Section II, and publicly release our code, dataset, and resources at https://github.com/sotayang/LiveStarPro.

Our main contributions can be summarized as follows:

• We present LiveStarPro, a proactive live streaming assistant that conducts continual real-time comprehension over long-horizon streams, sustains coherent contextual reasoning across diverse online tasks, and emits responses only at semantically appropriate moments.  
• We propose Streaming Verification Decoding (SVeD), a novel inference framework that determines the optimal response timing through a single forward-pass verification. It decouples silence determination from vocabulary generation, thereby avoiding the pitfalls of EOS-based approaches while preserving real-time responsiveness.  
• We design a training strategy based on Streaming Causal Attention Masks (SCAM) that, through interleaved frame-caption sequences, trains LiveStarPro to incrementally align variable-length video inputs with linguistic outputs, supporting the dynamic verification logic

of SVeD.

• We introduce a Tree-Structured Hierarchical Memory (TSHM) for long-horizon streaming video, coupling Peak-End compression for the active window with a Recursive Event Tree for long-term storage, enabling the model to retrieve coherent event chains and reuse historical visual-textual information evicted from the active context.  
• We construct OmniStarPro, a comprehensive dataset spanning 15 real-world scenarios and 5 short-horizon tasks, further contributing a long-form partition with 3 memory-centric tasks for long-term recall. Extensive experiments demonstrate the state-of-the-art performance of LiveStarPro, with an average improvement of 28.9% in semantic correctness and an 18.2% reduction in timing difference relative to existing online Video-LLMs.

## II. RELATED WORK

## A. Video Large Language Models

The steady maturation of Large Language Models (LLMs) [33]–[39] has propelled parallel progress in Video-LLMs [6]–[10], enabling them to address demanding tasks such as video captioning [40]–[42], question answering [43]– [48], and temporal grounding [49]–[51] within offline settings. Representative open-source systems, such as LLaVA-NeXT-Video [52], Video-LLaVA [53], and VILA [54], together with closed-source systems such as GPT-4o [35] and Gemini 1.5 Pro [55], generally treat video as a pre-recorded and finite file. Yet seamless human-computer interaction demands capabilities beyond static video analysis: an effective assistant should process real-time streams while exercising autonomous temporal decision-making, responding at contextually appropriate moments without explicit user prompts [20], [25]. Existing Video-LLMs [56]–[59] remain constrained by the dynamic nature of continuous streams and often lack the flexibility to alternate between passive observation and active response.

## B. Online Video Understanding

Driven by real-time applications such as live streaming [60] and wearable devices [28], recent work investigates architectures for online video understanding [20], [23], [25], [61], [62], where systems must process frames incrementally and decide when to respond. VideoLLM-online [20] pioneered a streaming EOS (End-Of-Sequence) token mechanism to mark silence intervals, later refined for efficiency by VideoLLM-MoD [21] and LION-FS [22]. As StreamMind [23] notes, however, EOS reliance is structurally flawed: the imbalance between silent and active frames biases training, while the semantic conflict between rich visual inputs and the meaningless EOS token erodes visual-language alignment [63]. Reliable evaluation is equally critical. Offline benchmarks [44], [64]–[74] probe abilities from action recognition to longterm reasoning [75]–[79], yet present complete, pre-segmented videos [80] unlike unsegmented live streams [81]–[83]. Recent streaming benchmarks such as SVBench [30], OVO-Bench [31], and StreamBench [27] evaluate models during synchronous playback, but rely on Video Question Answering alone and overlook tasks like continuous narration or real-time grounding. Their scenario coverage is also narrow, since heavy reliance on Ego4D [28] restricts evaluation primarily to firstperson perspectives. To support a more holistic evaluation, we introduce a comprehensive dataset and benchmark tailored to online agents that encompasses diverse real-world scenarios together with a synergistic set of streaming tasks.

![](images/b6dc05531504fde18fff3a5fc784e34fea3d582e72b12fb22c8ce35e374d6bc3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Input"] --> B["VLM"]
  B --> C["Decoder"]
  C --> D["Update context"]
  D --> E["Decer"]
  E --> F["Swap Frm^t_j and Dec^k"]
  F --> G["Dec^k+1"]
  G --> H["Append Dec^{k+1}"]
    
  I["Frame-1 ~ Frame-5 Caption-1"] --> J["Frame-by-Frame Input"]
  K["Frame-6 ~ Frame-11 Caption-2"] --> J
  L["Frame-17 ~ Frame-21 Caption-4"] --> J
    
  M["Decoding step^t_i"] --> N["Decoding Layer"]
  O["New Frm^t_j"] --> P["VLM"]
  Q["Decoding step^t_i"] --> R["PPL"]
  S["Decoding step^t_i"] --> T["PPL"]
  U["Decoding step^t_i"] --> V["logits"]
  W["Decoding step^t_i"] --> X["PPL"]
  Y["Decoding step^t_i"] --> Z["logits"]
  AA["Decoding step^t_i"] --> AB["PPL"]
  AC["Decoding step^t_i"] --> AD["logits"]
  AE["Decoding step^t_i"] --> AF["PPL"]
  AG["VLM"] --> AH["VLM"]
  AH --> AI["Decoder Layer"]
  AJ["Decer"] --> AK["Decer"]
    
    style A fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#ccf,stroke:#333
    style N fill:#ccf,stroke:#333
    style O fill:#ccf,stroke:#333
    style Q fill:#ccf,stroke:#333
    style R fill:#ccf,stroke:#333
    style S fill:#ccf,stroke:#333
    style T fill:#ccf,stroke:#333
    style U fill:#ccf,stroke:#333
    style V fill:#ccf,stroke:#333
    style W fill:#ccf,stroke:#333
    style X fill:#ccf,stroke:#333
    style Y fill:#ccf,stroke:#333
    style Z fill:#ccf,stroke:#333
    style AA fill:#ccf,stroke:#333
    style AB fill:#ccf,stroke:#333
    style AC fill:#ccf,stroke:#333
    style AD fill:#ccf,stroke:#333
    style AE fill:#ccf,stroke:#333
```
</details>

Fig. 2. Overview of the streaming verification decoding (SVeD) inference framework: A dynamic response-silence decoding framework designed to determine optimal response timing for online video understanding.

## C. Memory Management for Long-Form Video

A central bottleneck in processing continuous video streams is reconciling the finite context window of LLMs with arbitrarily long visual inputs. Early studies pursued token reduction through techniques such as spatial-temporal pooling [10], [26] and token merging [72]. More elaborate strategies were subsequently developed to capture extended temporal dependencies. Within streaming settings, VideoStreaming [25] propagates memory from earlier clips to guide the encoding of the current clip, while StreamChat [27] and TimeChat-Online [84] employ hierarchical memory or differential token dropping to filter redundant information on the fly. For offline long-form analysis, MovieChat [72], MA-LMM [16], and VideoLLaMB [85] rely on external memory banks that store and retrieve compressed historical features. Most of these mechanisms, however, are tailored to offline processing in which global context is fully accessible, or they apply uniform compression that does not distinguish salient moments from redundant ones in a live stream. Moreover, the external memory banks adopted by these methods are organized as flat collections that require an exhaustive scan with a retrieval cost that grows linearly with the number of stored units and a footprint that grows linearly with the stream duration. Drawing on the Peak-End rule [86] from cognitive psychology, we introduce a Tree-Structured Hierarchical Memory that both compresses the active context by retaining salient frames and organizes evicted history into a structured retrieval bank [87], [88] with sublinear expected retrieval, thereby granting access to long-term dependencies in infinite streams without saturating the context window.

## III. THE LIVESTARPRO FRAMEWORK

## A. Overview

To advance real-time online video understanding, we introduce LiveStarPro, a proactive live streaming assistant for long-horizon streams. As illustrated in Fig. 1, the architecture comprises three sub-modules that jointly resolve response timing, streaming alignment, and long-horizon memory management. Streaming Verification Decoding (SVeD) is a dynamic inference mechanism that autonomously identifies the appropriate response moment through a lightweight singlepass perplexity verification, removing the dependency on explicit silence tokens and delivering proactive context-aware responses with minimal latency. Streaming Causal Attention Masks (SCAM) constitute an instruction-tuning paradigm tailored to the incremental behavior of SVeD; by imposing specialized causal masks on interleaved frame-caption sequences, it aligns evolving visual prefixes with linguistic semantics so that understanding is continually updated as new frames arrive. Tree-Structured Hierarchical Memory (TSHM) manages memory for long-horizon streaming video, coupling a short-term working memory compressed through the Peak-End strategy with a long-term Recursive Event Tree, thereby enabling selective recall of historical event chains beyond the active context window and mitigating catastrophic forgetting throughout prolonged interactions.

## B. Inference Framework

A central challenge in online video understanding lies in the trade-off between responsiveness and computational efficiency. Unlike offline models that operate over complete videos, a streaming assistant must autonomously decide when to update its response as new frames arrive. A naive frameby-frame decoding scheme guarantees high responsiveness, yet it incurs prohibitive latency together with substantial narrative redundancy. To overcome this difficulty, we devise an inference framework centered on Streaming Verification Decoding (SVeD) that shifts the paradigm from continuous generation toward efficient verification.

1) Semantic Verification via Autoregressive Generation: A straightforward way to determine response timing verifies the semantic validity of historical outputs against each incoming frame. Let $[ D e c ] ^ { t _ { i } }$ denote the caption most recently emitted at timestamp $t _ { i } .$ . For a new frame at time $t _ { j } ,$ , the model performs full autoregressive decoding to generate a candidate $[ D e c ] _ { \mathrm { c a n d } } ^ { t _ { j } } ,$ decide whether the content has evolved or remains redundant. Formally, the response policy is:

$$
\operatorname{Output} \left(t _ {j}\right) = \left\{ \begin{array}{l l} \text { silent }, & \mathcal {S} \left(\left[ D e c \right] ^ {t _ {i}}, \left[ D e c \right] _ {\text { cand }} ^ {t _ {j}}\right) \geq \tau \\ \left[ D e c \right] _ {\text { cand }} ^ {t _ {j}}, & \mathcal {S} \left(\left[ D e c \right] ^ {t _ {i}}, \left[ D e c \right] _ {\text { cand }} ^ {t _ {j}}\right) <   \tau \end{array} \right. \tag {1}
$$

where $ { \boldsymbol { S } } ( \cdot , \cdot )$ is a semantic similarity function and τ a predefined threshold: a semantically consistent caption is suppressed as silence, otherwise the candidate is released.

This generate-then-compare strategy, however, requires a complete autoregressive decoding for every frame before a decision, incurring O(L) latency per frame with sequence length L, which is impractical for low-latency real-time streaming.

2) Streaming Verification Decoding: To overcome this efficiency bottleneck, we propose Streaming Verification Decoding (SVeD), which replaces the generate-then-compare paradigm with a verify-then-generate formulation. Rather than treating silence as a generation target, we regard it as a verification state. SVeD introduces a lightweight decoding gate that governs the transition between “watching” and “speaking.” The underlying mechanism requires only a single forward pass to verify the semantic validity of the existing caption with respect to the newly arriving visual frames. At any decoding step $t _ { i } ,$ the perplexity (PPL) of the generated caption [Dec] acts as a confidence measure:

$$
\mathrm{PPL} ^ {t _ {i}} ([ D e c ]) = \sqrt [ N ]{\frac {1}{P ([ D e c ] \mid [ C t x ^ {<   t _ {i}} ] , [ F r m ^ {t _ {i}} ])}} \tag {2}
$$

where N is the token length of [Dec]. For each subsequent incoming frame $[ F r m ^ { t _ { j } } ]$ , rather than autoregressively generating new tokens, SVeD simply re-evaluates the PPL of the previous caption [Dec] under the updated context.

Based on this metric, the gate operates on a threshold logic: if the verification perplexity exceeds a scaled reference value $( \mathrm { P P L } ^ { t _ { j } } ( [ D e c ] ) \ > \ \stackrel {  } { \alpha } \cdot \mathrm { P P L } ^ { t _ { i } } ( [ D e c ] ) )$ , it indicates a significant divergence between the visual input and the current description, prompting the gate to activate and generate a fresh caption. Conversely, if the perplexity remains stable, the gate suppresses generation to avoid redundancy. In this suppression state, to ensure the context window reflects the passage of time, we perform a logical Swap operation: the existing caption [Dec] is moved to the end of the context buffer [Ctx], effectively extending its validity to the current timestamp. This mechanism, detailed in Algorithm 1, ensures adaptive response timing and narrative coherence while significantly reducing computational overhead.

3) Streaming Key-Value Cache: An efficient implementation of SVeD must accommodate frequent context updates, including appends and swaps, without recurring computation. Conventional LLM caching mechanisms are largely static and therefore unable to cope with the dynamic sequence modifications that our framework induces. We accordingly devise an effective Streaming Key-Value (KV) Cache with a dual-level organization that maintains an intra-dialogue KV cache for frame-level processing and an inter-dialogue streaming cache for long-term context across conversations. This design meets two critical requirements: it preserves cache sequence integrity under the logical swap operations of SVeD, removing recomputation of historical representations when the caption position changes; and it accommodates the dynamic length adaptation of Peak-End memory compression, permitting strategic pruning of redundant tokens while retaining temporal coherence. As reported in Tab. VII, this strategy accelerates inference by 1.58× over configurations without KV caching while incurring negligible performance loss, rendering SVeD highly suitable for online deployment.

Algorithm 1 Streaming Verification Decoding (SVeD)  
Require: Video frame stream $\{[Frm^{t}]\}_{t=1}^{T}$ , Sensitivity threshold $\alpha$ Ensure: Dynamically generated caption [Dec]

1: Initialize [Dec], [Ctx] ← ∅

2: Initialize reference timestamp $t_{i} \leftarrow 0$ 3: for each incoming frame $[Frm^{t_{j}}]$ do

4: Append $[Frm^{t_{j}}]$ to [Ctx]

5: if [Dec] ≠ ∅ then

6: Compute verification perplexity:

7: $PPL^{t_{j}}([Dec]) = \sqrt[N]{1/P([Dec] | [Ctx^{\leq t_{j}}])}$ 8: if $PPL^{t_{j}}([Dec]) > \alpha \cdot PPL^{t_{i}}([Dec])$ then

9: Activate decoding:

10: Generate new tokens $[Dec]_{new}$ using $[Ctx^{\leq t_{j}}]$ 11: Update [Dec] ← [Dec] new

12: Append [Dec] to [Ctx]

13: $t_{i} \leftarrow t_{j}$ {Update reference timestamp}

14: else

15: Suppress: Swap the last two elements in [Ctx]
{Move [Dec] to the end}

16: end if

17: else

18: Perform initial decoding to generate [Dec]

19: Append [Dec] to [Ctx]

20: $t_{i} \leftarrow t_{j}$ 21: end if

22: end for

## C. Training Strategy

Although the SVeD framework supplies an efficient means of determining contextually appropriate response timing, its effectiveness depends on the model’s capacity to accurately estimate the probability of a caption under a progressively evolving visual context. Foundation models pre-trained on static image-text pairs typically lack the temporal granularity that the incremental verification logic of inference requires. To close this gap, we devise a stream-oriented training strategy centered on Streaming Causal Attention Masks (SCAM) that reformulates the training objective to match the dynamic inference requirements of LiveStarPro.

![](images/ae56b3844b004bede0d5fe8e38234198adc5de579b3590face592d011c5400d3.jpg)

<details>
<summary>bar chart</summary>

| Semantic Clip | Vision Token | Text Token | Generated Token | Masked Token | Interleaved Sequence |
| ------------- | ------------ | ---------- | --------------- | ------------ | --------------------- |
| 1             | 0            | 1          | 0               | 0            | 1                     |
| 2             | 0            | 0          | 0               | 0            | 2                     |
| 3             | 0            | 0          | 0               | 0            | 3                     |
| 4             | 0            | 0          | 0               | 0            | 4                     |
| 5             | 0            | 0          | 0               | 0            | 5                     |
</details>

Fig. 3. Overview of Streaming Causal Attention Masks (SCAM). SCAM organizes frames and captions into interleaved sequences and performs progressive per-time-step training, masking preceding captions within each semantic clip to align training with streaming inference.

1) Streaming Video-Language Alignment: Existing Video-LLMs generally build upon foundation models pre-trained on static image-text pairs [1], [2], [89]. Such models commonly optimize a static alignment objective:

$$
\max P ([ T x t _ {i} ] \mid [ I m g _ {i} ] / [ V i d _ {i} ]), \tag {3}
$$

which proves ill-suited to online scenarios in which the visual context accumulates incrementally and demands dynamic alignment with the linguistic outputs. To resolve this difficulty, we reformulate the training objective so as to model the probability of the current semantic description conditioned on the evolving historical context:

$$
\max P ([ T x t ^ {k} ] \mid [ C t x ^ {<   t _ {i}} ], [ F r m ^ {t _ {i}} ]), \forall t _ {i} \in C _ {k}, \tag {4}
$$

where $[ F r m ^ { t _ { i } } ]$ represents the frame at timestamp $t _ { i } ,$ and $[ \boldsymbol { C } t \boldsymbol { x } ^ { < t _ { i } } ]$ denotes the accumulated multimodal history. Here, $C _ { k }$ defines a semantic clip, namely a continuous sequence of frames that share the same semantic event description $[ T x t ^ { k } ]$ . Crucially, this objective departs in a fundamental manner from EOS-based approaches [20], [21] that compel the model to predict a silence token (i.e., max $P ( \mathrm { E O S ~ } | \dots ) )$ for non-response frames. By circumventing this trivial mapping, LiveStarPro sustains a consistent focus on meaningful visual-linguistic correlations and thereby furnishes a robust probability distribution for the verification stage of SVeD.

2) Interleaved Frame-Caption Sequences: To support frame-by-frame processing throughout training, we organize the data as Interleaved Frame-Caption Sequences. This format reproduces the streaming inference procedure in which the model continually ingests visual inputs and refreshes its understanding. Concretely, for a semantic clip $C _ { k } .$ , each frame $[ F r m ^ { t _ { i } } ]$ is paired with the corresponding caption $\left[ C a p ^ { k } \right]$ . Because multiple frames within the same event share identical semantics, naive repetition could induce overfitting. To counter this tendency, we adopt a stochastic caption sampling strategy: for every frame, a caption $[ C a p _ { j } ^ { k } ]$ is drawn at random from a predefined pool of M paraphrased variants, which encourages the model to learn robust semantic representations rather than to memorize specific string patterns.

3) Streaming Causal Attention Masks: Autoregressive training on these interleaved sequences raises distinctive challenges, particularly with respect to information leakage and context management. Since the frames within a single semantic clip $C _ { k }$ share identical caption targets, a standard causal mask would permit the model to copy trivially the caption generated for a preceding frame in the same clip and consequently to disregard the current visual input. To counter this behavior, we propose Streaming Causal Attention Masks (SCAM), as illustrated in Fig. 3. SCAM adapts the standard attention mechanism so as to enforce a selective visibility policy: it masks the attention weights that correspond to previously generated captions within the current semantic clip $C _ { k }$ and thereby compels the model to depend solely on the visual features of the current frame $[ F r m ^ { t _ { i } } ]$ together with the accumulated history. At the same time, in order to preserve narrative coherence across scene transitions, SCAM retains full visibility of the terminal captions from all preceding semantic clips $\{ C _ { 1 } , \ldots , C _ { k - 1 } \}$ . This explicit demarcation of semantic boundaries allows the model to reason about the event history while intra-event redundancy is suppressed. The optimized objective under SCAM is defined as:

$$
\max P ([ C a p _ {j} ^ {k} ] \mid [ C t x ^ {<   t _ {i}} \{M a s k ^ {\leq t _ {i}} \} ], [ F r m ^ {t _ {i}} ]), \forall t _ {i} \in C _ {k}. \tag {5}
$$

Here, $\boldsymbol { M a s k } ^ { \le t _ { i } }$ realizes these constraints in mathematical form and ensures that LiveStarPro learns to generate captions that are grounded in visual evidence rather than in linguistic repetition.

## D. Tree-Structured Hierarchical Memory (TSHM)

Although SVeD and SCAM endow LiveStarPro with robust capabilities for real-time inference and alignment, a critical deployment bottleneck arises from the memory accumulation and computational latency of long-duration video streams. Streaming inputs from live broadcasts, surveillance systems, and robotic cameras routinely extend well beyond the hour scale. Whatever the context capacity, a perpetual stream will inevitably saturate the memory budget of the model. Conventional strategies such as First-In-First-Out (FIFO) or sliding window approaches are prone to catastrophic forgetting, since they discard historical events indiscriminately once observations drift past the fixed temporal horizon. To overcome these limitations, we propose Tree-Structured Hierarchical Memory (TSHM). Inspired by human cognitive architectures, TSHM arranges memory into two complementary tiers: a highresolution Short-Term Working Memory that retains recent fine-grained details, and a compressed Long-Term Retrieval Memory that preserves salient historical events in a structured and queryable format, as illustrated in Fig. 4.

1) Short-Term Working Memory: Peak-End Compression: The processing of modern streaming videos, which frequently span extended durations at high frame rates, imposes substantial computational challenges on long-horizon understanding. To mitigate this, we devise a memory compression mechanism that draws on the Peak-End Rule [86], according to which human memory predominantly retains salient moments together with summary-level representations of experiences. Our design distills the active context window, designated the Short-Term Working Memory, through the explicit modeling of these two complementary signals. Specifically, the Peaks are identified by keyframe selection that is grounded in semantic confidence. During the SVeD phase, the verification perplexity of each frame is computed as a semantic divergence score, defined as $S ( t ) = { \mathrm { P P L } } ^ { t } ( [ D e c ] )$ . Frames with lower perplexity values signal a stronger semantic alignment with the ongoing description and are accordingly regarded as salient keyframes. In parallel, the End component is instantiated by the caption of each semantic clip, which functions as a temporal summary that captures the aggregated semantics of the entire event.

![](images/e58f2d16a821b7ab1c646e46c77efed9cea0003c7fb5dd1ed613ffac98edcba4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["(a) Short-Term Working Memory"] --> B["Context PPL"]
  A --> C["Compact Memory"]
  A --> D["Memory Unit"]
  B --> E["Nar."]
  C --> F["Nar."]
  D --> G["Nar."]
  E --> H["1.85 2.46 2.11"]
  F --> I["1.73 2.68 2.02 2.29"]
  G --> J["1.85 1.73 2.02 2.51 1.94"]
  H --> K["Compress"]
  I --> L["New Frame"]
  J --> M["New Frame"]
  K --> N["Long-Term Retrieval Memory"]
  L --> N
  M --> N
  N --> O["Offload"]
  O --> P["Initial State"]
  P --> Q["C"]
  Q --> R["F"]
  R --> S["Attach"]
  S --> T["C"]
  T --> U["F"]
  U --> V["New Branch"]
  V --> W["C"]
  W --> X["F"]
  X --> Y["Trigger"]
  Y --> Z["C"]
  Z --> AA["F"]
  AA --> AB["Retrieval Top-K"]
  AB --> AC["Video-LLM"]
  AC --> AD["Current Query Embedding"]
  AD --> AE["Video-LLM"]
  AE --> AF["Video-LLM"]
  AF --> AG["Nar."]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#cfc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#fcc,stroke:#333
    style J fill:#fcc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#fcc,stroke:#333
    style N fill:#fcc,stroke:#333
    style O fill:#ffc,stroke:#333
    style P fill:#ffc,stroke:#333
    style Q fill:#ffc,stroke:#333
    style R fill:#ffc,stroke:#333
    style S fill:#ffc,stroke:#333
    style T fill:#ffc,stroke:#333
    style U fill:#ffc,stroke:#333
    style V fill:#ffc,stroke:#333
    style W fill:#ffc,stroke:#333
    style X fill:#ffc,stroke:#333
    style Y fill:#ffc,stroke:#333
    style Z fill:#ffc,stroke:#333
    style AA fill:#ffc,stroke:#333
    style AB fill:#ffc,stroke:#333
    style AC fill:#ffc,stroke:#333
    style AD fill:#ffc,stroke:#333
```
</details>

Fig. 4. Overview of Tree-Structured Hierarchical Memory (TSHM). (a) Short-term frames are compressed via Peak-End rule, with evicted units offloaded to long-term storage. (b) The Recursive Event Tree organizes units by attaching similar events as children (Sim $\geq \tau )$ or creating new branches. (c) Context-aware retrieval fetches relevant event chains to augment generation.

The accumulation of tokens within the short-term working memory is governed by a dynamic pruning strategy. Once the total token count attains the context budget $L _ { m a x }$ , a pruning cycle is launched independently for each semantic clip $C _ { k }$ . For every clip, a dynamic threshold $\tau _ { k }$ is computed as the median divergence score of its frames, and only the following subset of frames is retained:

$$
\mathcal {T} _ {\text { keep }} ^ {k} = \{t \in C _ {k} \mid S (t) \leq \tau_ {k} \}. \tag {6}
$$

This operation removes approximately 50% of the frames with higher semantic divergence while consistently preserving the clip-level summary caption. As further frames arrive and additional pruning cycles are triggered, older semantic clips are progressively condensed into a compact representation consisting of a small number of salient peak frames together with their corresponding summary captions. Should the shortterm working memory remain saturated after all clips have been reduced to this compact form, the oldest semantic units are evicted and transferred to the long-term retrieval memory for persistent storage.

By prioritizing peak-level visual evidence and summarylevel semantic representations, this compression strategy affords adaptive scaling for long-duration streams. As demonstrated in Tab. VII, LiveStarPro attains superior semantic correctness and lower timing difference relative to Uniform Dropout and FIFO-based forgetting strategies, which validates both its computational efficiency and its capacity to sustain coherent long-range narratives.

2) Long-Term Retrieval Memory: Recursive Event Tree: Even with the pruning mechanisms introduced above, an unbounded video stream will inevitably exceed the capacity of a finite context buffer. Rather than permanently discarding evicted memory units, we archive them in a tree-structured long-term memory that preserves temporal continuity and causal relationships among events. This memory is organized as a recursive event tree, where each node corresponds to a memory unit $U _ { i } = \{ c _ { i } , v _ { i } , \mathcal { E } _ { i } , \mathcal { C } _ { i } \}$ . Here, $c _ { i }$ denotes the event caption, $v _ { i }$ represents the visual tokens of the corresponding peak frame, $\mathcal { E } _ { i }$ is a semantic embedding used for indexing, and $\mathcal { C } _ { i }$ stores the list of child nodes. When a newly evicted unit $U _ { n e w }$ is inserted into the long-term memory, its placement is determined by measuring semantic similarity between $\mathcal { E } _ { n e w }$ and the embeddings of existing nodes. Specifically, we identify the node $U _ { b e s t }$ with the highest similarity score. If this score exceeds a predefined threshold σ, $U _ { n e w }$ is attached as a child of $U _ { b e s t }$ , indicating that the new event represents a refinement or continuation of an existing event thread. Otherwise, $U _ { n e w }$ is initialized as a new root node, forming an independent event branch.

To ensure that higher-level nodes serve as effective semantic summaries of their corresponding event subtrees, the embedding of the parent node is updated upon each insertion using a momentum-based aggregation scheme:

$$
\mathcal {E} _ {\text { parent }} \leftarrow \text { Normalize } \left((1 - \beta) \cdot \mathcal {E} _ {\text { parent }} + \beta \cdot \mathcal {E} _ {\text { child }}\right), \tag {7}
$$

where $\beta$ controls the update rate. This design allows parent embeddings to gradually evolve toward the semantic centroid of their descendant events, thereby supporting robust and discriminative retrieval.

3) Memory-Augmented Generation via Retrieval: The hierarchical organization of the long-term memory enables a unified retrieval-augmented generation framework that supports both explicit querying and implicit contextual reasoning. When the SVeD gate activates response generation, LiveStarPro constructs a query vector q in accordance with the current task setting. In explicit question answering, q is derived directly from the textual embedding of the user query, which permits the targeted retrieval of specific historical events or facts. By contrast, for streaming narration without an explicit user prompt, we formulate an implicit query from the aggregated visual embeddings of recent short-term frames, which captures the current visual context. This implicit visual query retrieves relevant historical events from the long-term memory and thereby allows the model to identify recurring entities or semantically related scenes and to maintain long-range narrative coherence.

Crucially, because each parent embedding is maintained as the semantic centroid of its subtree (Eq. 7), the query need not be scored against every stored unit. Instead, LiveStarPro performs a hierarchical beam descent: starting from the set of root nodes, it scores only the immediate children of the current frontier by cosine similarity, retains the k most similar nodes as the next frontier, and recurses toward the leaves. Formally, the frontier at depth d is updated as

$$
\mathcal {F} _ {d + 1} = \operatorname{Top} - k _ {U _ {i} \in \text { Child } (\mathcal {F} _ {d})} \frac {q \cdot \mathcal {E} _ {i}}{\| q \| \| \mathcal {E} _ {i} \|}, \tag {8}
$$

and the descent terminates at the leaf level, yielding the index set I of the retrieved units. Since only the children along the top-k root-to-leaf paths are evaluated, this procedure inspects $O ( k b \log _ { b } n )$ nodes rather than all n units, which realizes the sublinear retrieval cost analyzed in Sec. III-D4. When the tree degenerates into isolated roots $( \sigma \to 1 )$ , the frontier spans the entire root level and the descent gracefully reduces to the exhaustive $O ( n )$ scan of a flat index.

For each retrieved node, the associated event context is further gathered by traversing the corresponding tree paths, which include its parent and immediate children. The resulting memory set,

$$
\mathcal {M} _ {\text { retrieved }} = \left\{\left(c _ {j}, v _ {j}\right) \mid j \in \mathcal {I} \cup \operatorname{Path} (\mathcal {I}) \right\},
$$

is injected back into the attention window during generation. By explicitly integrating retrieved historical captions and visual tokens with current observations, LiveStarPro effectively bridges long-term memory with ongoing perception, enabling coherent question answering together with temporally consistent streaming narration under a bounded context budget.

4) Theoretical Analysis: We characterize the memory footprint and retrieval cost of TSHM as a qualitative complexity argument under stated structural assumptions rather than a worst-case guarantee. Let T denote the elapsed stream duration measured in semantic clips, and let n denote the number of memory units in the long-term retrieval memory at query time.

a) Bounded active memory.: The short-term working memory operates under a fixed token budget $L _ { m a x }$ . Whenever the budget is reached, the dynamic pruning rule retains the $\mathcal { T } _ { k e e p } ^ { k }$ satisfying $\begin{array} { l c l } { S ( t ) } & { \le } & { \tau _ { k } } \end{array}$ with $\tau _ { k }$ the per-clip median, removing about half of the frames of every clip while preserving the summary caption. Each clip is thus at most halved per pruning cycle, contributing at most $\lceil F _ { k } / 2 ^ { r } \rceil + 1$ units after r cycles, where $F _ { k }$ is its initial frame count. The active memory therefore never exceeds $L _ { m a x }$ regardless of stream length, guaranteeing a constant per-step inference cost independent of T .

b) Expected logarithmic retrieval under balanced growth.: A query traverses the recursive event tree from roots toward leaves. Under the assumption that σ and $\beta$ induce a bounded branching factor b and balanced subtree growth, a tree storing n units has height $h \ = \ O ( \log _ { b } n )$ . Since retrieval evaluates the top-k nodes along root-to-leaf paths and then expands the parent and immediate children of each, the expected number of similarity evaluations per query is $O ( k b \log _ { b } n )$ , sublinear in the stored units, matching the moderate branching factor and shallow height observed on OmniStarPro-Long (Tab. IX). Balanced growth is not guaranteed: a highly skewed stream that collapses most events into one thread can degenerate the tree into a chain, while setting σ so high that every unit forms an independent root reduces the structure to a flat index with linear $O ( n )$ cost, which delineates the role of $\sigma$ in trading retrieval accuracy against efficiency.

c) Comparison with flat memory.: A flat memory bank storing every evicted unit incurs $O ( n )$ retrieval and a footprint growing linearly with T . TSHM instead bounds the active footprint by $L _ { m a x } ,$ supports $O ( \log n )$ expected retrieval under the balanced-growth assumption, and preserves causal event chains through parent and child traversal, which helps explain the graceful recall degradation of LiveStarPro as the memory span grows on OmniStarPro-Long.

## IV. DATASET: OMNISTARPRO

To enable rigorous evaluation and training of LiveStarPro, we construct OmniStarPro, a dataset for online video understanding that overcomes prior benchmarks with expertannotated streams and a unified streaming protocol probing temporal perception and contextual awareness under strict causal constraints. It comprises two complementary partitions. OmniStarPro-Live inherits the five synergistic shorthorizon tasks of the conference benchmark and emphasizes instantaneous response timing over minute-scale streams. OmniStarPro-Long is a newly curated collection of streams from ten minutes to beyond one hour, introducing three memory-centric tasks that probe recall of historical content after its eviction from the active context window. This longform emphasis targets applications such as live streaming, surveillance, and cinematic tools.

## A. Dataset Construction Pipeline

The OmniStarPro construction follows a rigorous multistage process that guarantees multimodal consistency, content diversity, and real-time narration fluency. As depicted in Fig. 5, we take OmniStarPro-RNG as a running example and construct the remaining tasks analogously: steps (1)-(3) cover data collection and preprocessing shared across the five shorthorizon tasks of the OmniStarPro-Live partition, whereas steps (4)-(6) assemble a specific online task dataset.

1) Balanced Data Collection: Data collection for the OmniStarPro-Live partition is initiated through the official API of YouTube, from which we gather 120,598 short videos (≤ 6 minutes) together with the associated metadata. We choose YouTube for its heterogeneous repository spanning broad categories and cultural distributions with detailed classification tags, concentrating on 15 real-world scenarios that encompass 46 specific categories (Fig. 6a). To curb geographic bias, we apply stratified sampling conditioned on source countries via two mechanisms: downsampling overrepresented regions, and capping selection at 20 videos per channel, which prevents content homogenization while preserving platform authenticity. For the OmniStarPro-Long partition, we separately harvest 9,260 long-form videos exceeding ten minutes from the same scenarios as the raw pool for memory-centric curation.

2) Multimodal Quality Filtering: To safeguard cross-modal content quality, we implement a three-stage hierarchical filtering process. The social validation phase retains only videos with substantial audience engagement (more than 5,000 social interactions). For audio, we adopt OpenAI’s Whisper1 speech

![](images/d0a8a9066941518de6d8a4c076365c82fadbc435986bfa6412e3beb3784bbd31.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["3. Quality Selection"] --> B["2. Filtering"]
  B --> C["1. Video Collection"]
  C --> D["4. Constructing Captions"]
  D --> E["5. Combine Captions"]
  E --> F["6. Manual Annotation"]

    subgraph A[Quality Selection]
        A1["Number of likes ≥ 5000"]
        A2["per 10 seconds ≤ 2 words"]
        A3["resolution ≥ 720p"]
        A4["Aesthetic score ≥ 5"]
        A5["Optical flow analyse"]
    end

    subgraph B[Filtering]
        B1["Time duration ≤ 300s"]
        B2["Country Downsampling"]
        B3["Diversity Per channel ≤ 20"]
    end

    subgraph C[Video Collection]
        C1["Topic"]
        C2["YouTube"]
    end

    subgraph D[Constructing Captions]
        D1["Prompt"]
        D2["Qwen2-VL"]
        D3["A young boy in a red jacket..."]
        D4["A young boy in a red jacket..."]
        D5["The boy is wiping ice off..."]
        D6["After cleaning, the car..."]
        D7["N Seconds"]
        D8["Caption 1"]
        D9["Caption 2"]
        D10["Caption 3"]
        D11["•••"]
        D12["•••"]
        D13["•••"]
        D14["•••"]
        D15["•••"]
    end

    subgraph E[Combine Captions]
        E1["A young boy in a red jacket..."]
        E2["A young boy in red jacket..."]
        E3["The boy is wiping ice off..."]
        E4["N Seconds"]
        E5["•••"]
        E6["•••"]
        E7["•••"]
        E8["•••"]
        E9["•••"]
        E10["•••"]
        E11["•••"]
        E12["•••"]
        E13["•••"]
        E14["•••"]
        E15["•••"]
    end

    subgraph F[Manual Annotation]
        F1["Narration 1"]
        F2["Narration 2"]
        F3["Narration X"]
        F4["I-2S"]
        F5["M-NS"]
    end

    subgraph G[Manual Annotation]
        G1["a) AI-Generated Narration Error Detection"]
        G2["b) Missing Temporal Annotation Identification"]
        G3["c) Event Sequence Error Correction"]
        G4["d) Second-Level Timestamp Verification"]
        G5["e) Narration Segmentation Analysis"]
        G6["f) Violent Content Filtering"]
        G7["g) Dialogue Excessiveness Assessment."]
    end
```
</details>

Fig. 5. Overview of the pipeline of a rigorous multi-stage process. Steps (1)-(3) involve data collection and preprocessing, and steps (4)-(6) involve constructing an online task dataset, using the OmniStarPro-RNG task as an example. Other online tasks are constructed in a similar manner.

recognition system under a strict lexical density constraint of at most two words per 10-second interval, since the evaluated Video-LLMs typically cannot process speech and excessively dense speech obscures their understanding of online videos. The visual quality assessment comprises three sequential analyses. First, FFmpeg2-based resolution screening preserves only HD content (≥720p). Subsequently, optical flow3 analysis removes videos that exhibit either excessive motion blur (≥15% high-variance frames) or prolonged static segments (≥85% temporal occupancy within any 30-second window). Finally, an aesthetic assessment model4 is employed to exclude videos that score below 5/10 on visual composition metrics. This multi-tiered filtering architecture progressively reduces the initial corpus from 120,598 candidate videos to a refined set of 21,544 high-quality videos.  
3) Temporal-Aware Frame Processing: We operate on a per-second basis, generating an initial textual caption for each video second with Qwen2-VL [2]. To curb temporal redundancy while preserving narrative fluency, we introduce a dynamic segmentation algorithm that detects coherent story segments by analyzing semantic similarity between consecutive captions. The core of this approach relies on an adaptive thresholding mechanism (θ = 0.9) together with a LIFO (Last-In-First-Out) stack architecture. Specifically, timestamps are pushed onto the stack while the cosine similarity between successive captions is computed continuously. When adjacent captions exceed the semantic coherence threshold, the algorithm collapses them into a unified segment, which preserves temporal continuity while eliminating redundant descriptions. The final narration, produced by Qwen2-VL, combines two processes: (1) the merging of redundant captions within each coherent segment, and (2) the use of historical narrations to maintain contextual fluency between adjacent segments.

2https://ffmpeg.org/

3https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial code/ video/optical flow/optical flow.py

4https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.3/tools/scoring

4) Multidimensional Annotation: To guarantee high-quality annotations, we engaged a panel of 30 domain experts to perform multilayered annotation across seven critical dimensions: a) AI-Generated Narration Error Detection, b) Missing Temporal Annotation Identification, c) Event Sequence Error Correction, d) Second-Level Timestamp Verification, e) Narration Segmentation Analysis, f) Violent Content Filtering, and g) Dialogue Excessiveness Assessment. Through iterative consistency validation rounds, inter-annotator discrepancies were resolved and robust annotation agreement was established. This rigorous quality control yielded 20,137 validated videotext pairs with real-time narrations, an 87.8% retention rate relative to the original dataset.  
5) Long-form Stream Curation: The OmniStarPro-Long partition is assembled separately to support memory-centric evaluation over hour-scale content. Beginning from the raw pool of 9,260 long-form videos, we apply the same multimodal quality filtering and expert verification used for the short-horizon partition, which yields 2,108 high-quality longform streams. These streams are stratified into three duration tiers: 1,396 of ten to thirty minutes, 331 of thirty to sixty minutes, and 381 beyond one hour, averaging 34.7 minutes. For each stream, per-second dense captions are produced and consolidated into coherent event segments through the dynamic semantic fusion procedure of StreamingCoT [90], which furnishes reliable second-level temporal references. Building on these references, expert annotators compose 12,704 queries for the LMR, CDQ, and TBR tasks and record, for every query, the ground-truth timestamp of the supporting evidence together with the resulting memory span. This protocol ensures that a substantial proportion of the queries depend on content lying well beyond the active context window, thereby isolating the capacity for genuine long-term recall.

## B. Task Definitions

To assess the capabilities of online agents comprehensively, we formulate two task families. The OmniStarPro-Live partition retains five short-horizon tasks that emulate realworld streaming interactions, whereas the OmniStarPro-Long partition contributes three memory-centric tasks that operate over long-form streams.

(1) Real-time Narration Generation (RNG) requires the model to act as a live commentator producing coherent realtime descriptions of evolving content. Unlike offline captioning, it is penalized for latency and must decide when to speak or stay silent to avoid redundancy. (2) Online Temporal Grounding (OTG) evaluates event localization within a continuous stream: given a query, the model must identify the start and end timestamps of the relevant segment as soon as it occurs, without access to future frames. (3) Frame-level Dense QA (FDQ) probes fine-grained perception by querying the model at dense intervals about detailed visual attributes or actions in the current frame, assessing its capacity to sustain high-resolution awareness throughout the stream. (4) Contextual Online QA (COQ) assesses short-term memory and causal reasoning through questions that depend on recent history, requiring the model to synthesize the current frame with the preceding context. (5) Multi-turn Interactive QA (MIQ) emulates a continuous user-agent dialogue, challenging the model to handle coreference resolution and maintain a consistent persona over a long session while the background video continues to play.

(1) Long-range Memory Recall (LMR) targets retrieval of factual details that appeared early in a long-form stream and have since been evicted from the active context window. Queried at a late timestamp about an attribute of a past entity or event, the model must recover it without the intervening frames. Because the query-evidence distance typically exceeds the active window, this task isolates genuine long-term recall rather than recent-context perception. (2) Cross-event Difference Query (CDQ) compares two events separated by a long interval within the same stream: given a pair of widely spaced moments, the model must report the change in a salient attribute such as the count, state, or spatial arrangement of the entities. Requiring simultaneous retrieval and contrast of two distant memories, it probes access to multiple historical entries rather than an explicit logical chain. (3) Temporal Backtracking (TBR) requires the model to locate the most recent past occurrence of a queried event and report its timestamp or an associated attribute. Unlike OTG, which localizes an event in the present, this task backtracks through evicted history to recover an event that has left the active window, exposing the catastrophic forgetting of fixed-window architectures.

## C. Statistics and Comparison

OmniStarPro sets itself apart from existing benchmarks through its scale, its diversity, and its dedicated emphasis on online streaming constraints across a wide temporal spectrum.

1) Scenario Diversity: As illustrated in Fig. 6a, OmniStarPro covers 15 diverse real-world scenarios, including Travel & Events, Sports, News & Politics, and Gaming. Each scenario is further subdivided into 2 to 4 fine-grained categories by the native annotation system of YouTube, resulting in 46 specific categories in total.

2) Video Length Distribution: Fig. 6b and Fig. 6c present the distribution of video durations across the two partitions. Within the OmniStarPro-Live partition, 45.54% of the videos exceed 100 seconds in length, with the majority falling within the 51 to 150-second range. The OmniStarPro-Long partition extends this spectrum substantially, with 2,108 streams averaging 34.7 minutes (stratified as above). This coverage of hourscale content confirms that OmniStarPro supports the joint evaluation of instantaneous response and long-term recall, far surpassing standard short-video benchmarks.

3) Memory-Span Distribution: For the three memorycentric tasks, we additionally report the distribution of the memory span, defined as the temporal interval between a query and its supporting evidence. Across the 12,704 longform queries, the memory span averages 18.6 minutes and reaches a maximum of 71.3 minutes, and 73.4% of the queries exhibit spans that exceed the active context window of the model, which guarantees that the OmniStarPro-Long partition genuinely stresses long-term retrieval rather than recent-context perception.

4) Annotation Scale: The OmniStarPro-Live partition contains 20,137 expert-annotated video streams that are partitioned into 19,137 training and 1,000 evaluation instances, with temporally dense annotations that average 14.5 QA pairs and 8.2 caption segments per video. The OmniStarPro-Long partition supplies an additional 2,108 long-form streams together with 12,704 memory-centric queries for the LMR, CDQ, and TBR tasks, each accompanied by a ground-truth evidence timestamp that supports span-aware evaluation. Together, the two partitions comprise 22,245 streams and ensure rigorous evaluation across both short-horizon and long-horizon online settings.

## V. EXPERIMENTS

## A. Experimental Setup

1) Datasets and Benchmarks: We validate LiveStarPro across both offline and online settings. For real-world online evaluation, we adopt our OmniStarPro benchmark over the five short-horizon tasks and the three memory-centric longform tasks. We further report results on SVBench [30] for streaming QA and build an Ego4D Narration Stream [28] benchmark from dense timestamped narrations to gauge egocentric understanding. To endow LiveStarPro with streaming capabilities, we design a two-phase progressive training paradigm that draws on 83K samples. Phase I (Temporal Alignment Pretraining) focuses on establishing frame-level semantic correspondences using 63K curated segments from ActivityNet Captions [65] (9K selected from 20K raw samples), Shot2Story [91] (33K selected from 43K raw samples), Ego4D Narration Stream [28] (20K selected from 113K raw samples), and MVBench [44] (1K selected from 4K raw samples). Phase II (Multi-Task Online Adaptation) utilizes 20K samples from OmniStarPro to specialize the model for the five online tasks via multi-objective alignment.

2) Implementation Details: LiveStarPro is built upon the InternVideo2.5 architecture [10], [92] and comprises an InternViT [1] vision encoder, an MLP projector, and an

![](images/709520938164d91bfd5b6b14c78fee5de9742e1e0179e11d59718aa2a53c6471.jpg)

<details>
<summary>pie chart</summary>

| Category | Value |
|---|---|
| Fashion & Style | 100 |
| Beauty & Personal | 85 |
| Entertainment | 70 |
| Comedy | 65 |
| Pop Art & Icon | 60 |
| Travel & Sports | 55 |
| Drama | 50 |
| Animal & Animal | 45 |
| Music | 40 |
| Cars & Vehicles | 35 |
| Car Reviewes | 30 |
| Motorcycle | 25 |
| Car Maintenance | 20 |
| Music Videos | 15 |
| Concerts | 10 |
| H-22: Tourists | 5 |
| Pet Care | 0 |
| Vehicle | 0 |
| Sports | 0 |
| Auto & Vehicles | 0 |
| Film & Animation | 0 |
| Social Animated Movies | 0 |
| Personal & Family | 0 |
| Movie Reviews | 0 |
| Restaurants | 0 |
| Children's Toys | 0 |
| Gaming (Video) | 0 |
| TV Productions | 0 |
| DIY Products | 0 |
| Fashion & Style | 100 |
| Makeup & Tuxitions | 95 |
| Personal & Personal | 90 |
| Analysis | 85 |
| Current News | 80 |
| Fashion/Stores | 75 |
| Movie Traps | 70 |
| Celebrity Gorsip | 65 |
| Variety Shows | 60 |
| Funny Clips | 55 |
| Stand-up Comedy | 50 |
| Guts | 45 |
| Family Life | 40 |
| Personal Comedy | 35 |
| Entertainment | 30 |
| Comedy | 25 |
| Travel & Sports | 20 |
| Drama | 15 |
| Animal & Animal | 10 |
| Music | 5 |
| Cars & Vehicles | 0 |
| Film & Animation | 0 |
| Automotive & Vehicles | 0 |
| Pet Care | 0 |
| Video | 0 |
| Sports | 0 |
| Outdoor & Tourism | 0 |
| School | 0 |
| Business | 0 |
| Retail/Bride | 0 |
| Home Goods | 0 |
| Travel Guides | 0 |
| Travelers | 0 |
| Traveling Games | 0 |
| Entertainment (Music) | 0 |
| Entertainment (Entertainment) | 0 |
The chart displays a single data point for the 'Comedy' category. The values are estimated based on the chart's visual representation.
</details>

(a)

![](images/1cce5c61186791b978394fb551d295603b74b93160c098a5387d69f0f4e4e777.jpg)

<details>
<summary>bar chart</summary>

| Duration (seconds) | Number of streams |
| ------------------ | ----------------- |
| 0-50               | 3,020             |
| 51-100             | 7,946             |
| 101-150            | 4,445             |
| 151-200            | 2,102             |
| 201-250            | 1,124             |
| 251-300            | 1,464             |
| >300               | 36                |
</details>

(b)

![](images/86c9e0303e1edf8d78f62bda435af547778f0ebf29f20e060bd19c96c4d4401e.jpg)

<details>
<summary>bar chart</summary>

| Duration (minutes) | Number of streams | Percentage |
| ------------------ | ----------------- | ---------- |
| 10-30              | 1,396             | 66%        |
| 30-60              | 331               | 16%        |
| >60                | 381               | 18%        |
</details>

(c)  
Fig. 6. Distributions of video data. (a) Distribution of video categories across 15 real-world scenarios. (b) Duration distribution of the OmniStarPro-Live partition at the second level. (c) Duration distribution of the OmniStarPro-Long partition at the minute level.

InternLM2.5-7B [93] language model. InternViT extracts video frame embeddings at 1-4 FPS, with each frame represented by 16 tokens. For efficiency, we fine-tune the model under a static resolution strategy, which processes multiminute content within an 8K-token context window. Full finetuning ran on 8× NVIDIA A800 GPUs.

3) Settings.: During training, we trained the models for 1 epoch with a learning rate of $4 \times 1 0 ^ { - 5 }$ using the AdamW optimizer $( \beta _ { 1 } ~ = ~ 0 . 9 , \beta _ { 2 } ~ = ~ 0 . 9 9 9 ,$ weight decay = 0.05). We utilized a per-device batch size of 1 and gradient accumulation over 4 steps to achieve an effective global batch size of 32. A cosine learning rate scheduling was adopted with a warmup ratio of 0.03. Input frames were uniformly resized to $4 4 8 \times 4 4 8$ , with a patch downsampling ratio of 0.5. The vision encoder was frozen during training, while the MLP projector and language model components were fully updated. Each training sequence contains up to 8192 tokens, consisting of interleaved frame and language tokens following the InternVL2.5 conversational template. We optimized the model using the standard autoregressive cross-entropy loss computed over the language tokens; critically, the loss was computed only on assistant response tokens, and inter-frame language segments were excluded via our SCAM strategy. For inference, the tunable scaling factor α in SVeD was set to 1.03 by default. Regarding the TSHM memory module, the pruning window W in the short-term working memory was set to 60 frames, and the size of the paraphrased caption pool for streaming video-language alignment was set to M = 1 by default to ensure optimal temporal alignment.

## B. Online Experiments

To approximate real-world online conditions, we conduct inference-time evaluations of Video-LLMs on the OmniStarPro test set. Unlike the Ego4D Narration Stream benchmark, which runs offline for lack of inference-result scoring, and semi-online benchmarks such as SVBench that rely on fixed decoding timestamps, OmniStarPro lets Video-LLMs autonomously decide when to respond or remain silent across five tasks in a fully online setting. This setting assesses both the temporal accuracy and the semantic consistency of responses against the ground truth.

Evaluation Metrics. We adopt the following metrics to assess model performance as an online video assistant. For each video, we denote the set of ground-truth semantic clips as $G = \{ g _ { 1 } , \dots , g _ { N } \}$ , where each clip $g _ { i }$ is associated with a temporal interval $[ t _ { \mathrm { s t a r t } , i } , t _ { \mathrm { e n d } , i } ]$ and a ground-truth caption. The model produces a set of responses $\boldsymbol { R } = \{ r _ { 1 } , \ldots , r _ { M } \}$ , where each response $r _ { j }$ is generated at time $t _ { \mathrm { r e s p } , j }$ . For each semantic clip gi, we define the set of matched responses as

$$
M _ {i} = \left\{r _ {j} \in R \mid t _ {\text { resp }, j} \in \left[ t _ {\text { start }, i}, t _ {\text { end }, i} \right] \right\}.
$$

• TimDiff (Timing Difference) measures the temporal deviation between model-generated responses and groundtruth semantic clips. For each clip, missing responses are penalized by assigning the full clip duration, while multiple responses incur cumulative latency penalties. Formally, TimDiff is defined as

$$
\begin{array}{l} \operatorname{TimDiff} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\mathbb {I} [ | M _ {i} | = 0 ] \cdot (t _ {\text { end }, i} - t _ {\text { start }, i}) \right. \\ \left. + \mathbb {I} \left[ \left| M _ {i} \right| > 0 \right] \cdot \sum_ {r _ {j} \in M _ {i}} \left(t _ {\text { resp }, j} - t _ {\text { start }, i}\right)\right), \\ \end{array}
$$

where I[·] denotes the indicator function. Lower TimDiff values indicate closer temporal alignment between responses and visual events.

• TimRedun (Timing Redundancy) evaluates response redundancy by measuring the deviation from the ideal case of producing exactly one response per semantic clip. It is computed as the average absolute difference between the number of generated responses and one across all clips:

$$
\text { TimRedun } = \frac {1}{N} \sum_ {i = 1} ^ {N} \left| | M _ {i} | - 1 \right|.
$$

• TimCover (Timing Coverage) measures the proportion of semantic clips that receive at least one valid response. It is defined as

$$
\operatorname{TimCover} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {I} [ | M _ {i} | > 0 ],
$$

TABLE I QUANTITATIVE COMPARISON ON THE OMNISTARPRO-RNG TASK UNDER OFFLINE AND ONLINE SETTINGS. PPL IS PROVIDED FOR REFERENCE ONLY DUE TO VOCABULARY VARIATIONS. “-” DENOTES TEST INAPPLICABILITY.

<table><tr><td rowspan="2">Method</td><td colspan="4">Offline Evaluation</td><td colspan="5">Online Evaluation</td></tr><tr><td>PPL↓</td><td>TokAcc↑</td><td>SemCor↑</td><td>SumFluen↑</td><td>TimDiff↓</td><td>TimRedun↓</td><td>TimCover↑</td><td>SemCor↑</td><td>SumFluen↑</td></tr><tr><td>Human</td><td>-</td><td>-</td><td>6.73</td><td>7.17</td><td>1.08</td><td>1.24</td><td>0.84</td><td>6.09</td><td>6.81</td></tr><tr><td colspan="10">Offline Video-LLMs / LVLMs</td></tr><tr><td>GPT-4V [94]</td><td>-</td><td>-</td><td>4.97</td><td>5.37</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GPT-4o [35]</td><td>-</td><td>-</td><td>5.03</td><td>5.45</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LLaVA-Video [52]</td><td>12.42</td><td>0.53</td><td>3.40</td><td>2.88</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InternVideo2.5 [92]</td><td>6.91</td><td>0.56</td><td>4.32</td><td>3.61</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InternVL2.5 [1]</td><td>9.81</td><td>0.51</td><td>3.40</td><td>2.94</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MiniCPM-V 2.6 [3]</td><td>9.46</td><td>0.57</td><td>4.34</td><td>4.13</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-VL [95]</td><td>13.80</td><td>0.59</td><td>4.42</td><td>4.24</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="10">Online Assistants</td></tr><tr><td>VideoLLM-online [20]</td><td>9.73</td><td>0.49</td><td>3.01</td><td>0.69</td><td>2.67</td><td>2.15</td><td>0.80</td><td>1.68</td><td>0.59</td></tr><tr><td>VideoLLM-MoD [21]</td><td>9.93</td><td>0.48</td><td>2.89</td><td>0.65</td><td>2.54</td><td>2.49</td><td>0.90</td><td>1.66</td><td>0.55</td></tr><tr><td>MMDuet [24]</td><td>5.69</td><td>0.60</td><td>4.29</td><td>3.40</td><td>2.32</td><td>0.62</td><td>0.51</td><td>1.93</td><td>2.69</td></tr><tr><td>LiveStar</td><td>5.14</td><td>0.62</td><td>4.62</td><td>4.55</td><td>1.91</td><td>0.95</td><td>0.71</td><td>3.19</td><td>4.25</td></tr><tr><td>LiveStarPro</td><td>5.21</td><td>0.65</td><td>4.76</td><td>4.64</td><td>1.89</td><td>1.01</td><td>0.78</td><td>3.27</td><td>4.41</td></tr></table>

TABLE II EVALUATION RESULTS OF ONLINE VIDEO-LLMS ON OMNISTARPRO. “–” IN OTG MEANS NO GENERATION NEEDED FOR SCORING; “–” IN COQ AND MIQ INDICATES REAL-TIME QA.

<table><tr><td rowspan="2">Method</td><td colspan="5">Online Evaluation (SemCor↑/TimDiff↓)</td><td rowspan="2">FPS↑(5 min)</td></tr><tr><td>RNG</td><td>OTG</td><td>FDQ</td><td>COQ</td><td>MIQ</td></tr><tr><td>Human</td><td>6.09/1.08</td><td>-/1.81</td><td>9.12/1.01</td><td>7.96/-</td><td>7.83/-</td><td>-</td></tr><tr><td>VideoLLM-online</td><td>1.68/2.67</td><td>-/9.69</td><td>2.35/2.15</td><td>4.01/-</td><td>3.83/-</td><td>3.37</td></tr><tr><td>VideoLLM-MoD</td><td>1.66/2.54</td><td>-/9.83</td><td>2.11/2.23</td><td>3.99/-</td><td>3.75/-</td><td>3.41</td></tr><tr><td>MMDuet</td><td>1.93/2.32</td><td>-/4.42</td><td>4.78/2.65</td><td>5.71/-</td><td>5.62/-</td><td>0.91</td></tr><tr><td>LiveStar</td><td>3.19/1.91</td><td>-/3.57</td><td>6.44/1.80</td><td>5.85/-</td><td>5.78/-</td><td>3.82</td></tr><tr><td>LiveStarPro</td><td>3.27/1.89</td><td>-/3.61</td><td>6.61/1.77</td><td>5.97/-</td><td>5.81/-</td><td>3.96</td></tr></table>

where a clip contributes a score of 1 if it contains at least one response and 0 otherwise.

• SemCor (Semantic Correctness) assesses semantic alignment between model responses and ground truth using GPT-4o. For each semantic clip gi, if $| M _ { i } | \ge 1$ , we select the earliest response in $M _ { i }$ as the representative output; otherwise, an empty response is assigned. GPT-4o scores the response–clip pair along three dimensions: (1) Semantic Accuracy, (2) Language Quality, and (3) Information Completeness, each on a 0–10 scale. The final SemCor score is obtained by averaging the three dimensions.  
• SumFluen (Summarize Fluency) evaluates the holistic fluency and narrative quality of the model’s complete output sequence. All responses are concatenated in temporal order to form a model narrative, which is compared against the concatenated ground-truth narrative using GPT-4o. Evaluation is conducted along five dimensions: (1) Writing Logicality, (2) Language Fluency, (3) Writing Conciseness, (4) Semantic Consistency, and (5) Narrative Completeness.

Validity of LLM-as-Judge. SemCor and SumFluen use GPT-4o as an automated judge [96], [97]. Since all methods are scored with identical prompts and multi-dimensional rubrics, the judge applies a consistent scoring function; thus, although absolute scores may carry stylistic bias, the rank-order and relative improvements across methods remain reliable.

Results. The right portion of Tab. I reports the online evaluation results of different models on OmniStarPro-RNG, where each model follows its own response-silence strategy. The results show that LiveStarPro achieves both reduced response latency and improved semantic accuracy. Notably, VideoLLMonline and VideoLLM-MoD generate outputs for nearly every frame, resulting in the highest TimCover scores; however, this aggressive behavior leads to inferior performance on other metrics. Conversely, MMDuet produces substantially fewer responses, yielding the lowest TimRedun but at the expense of reduced performance on other evaluation criteria. These trends are further illustrated in Tab. II, where LiveStarPro surpasses prior online Video-LLMs across the OmniStarPro tasks and improves over its conference predecessor LiveStar on most tasks, with comparable timing on OTG, while maintaining the fastest inference speed. Relative to the second-best prior online Video-LLM, it achieves a 28.9% improvement in SemCor, an 18.2% reduction in TimDiff, and a 16.1% increase in FPS.

Long-form Memory Evaluation. To assess long-term recall, we further evaluate online Video-LLMs on the OmniStarPro-Long partition across the three memory-centric tasks. Following the span-aware protocol, we report recall accuracy under three memory-span buckets, namely short (<10 minutes), medium (10–30 minutes), and long (>30 minutes), which respectively correspond to evidence within, near the boundary of, and well beyond the active context window. Here recall accuracy denotes the proportion of queries whose generated answer matches the ground-truth attribute or timestamp of the supporting evidence, judged by GPT-4o against the per-query ground truth, rather than a top-k retrieval recall. As summarized in Tab. III, all sliding-window baselines exhibit a pronounced degradation as the memory span grows, with accuracy that collapses toward chance once the supporting evidence is evicted from the active window. By contrast, LiveStarPro retains a substantially higher recall across all three buckets and degrades far more gracefully on the long bucket, which directly evidences the benefit of the hierarchical memory of TSHM for genuine long-term retrieval rather than recent-context perception. These results also serve as a principled proxy for comparison with offline memory architectures such as MA-LMM [16], MovieChat [72], and VideoLLaMB [85]: all three methods organize evicted features in flat external banks identical in structure to the flat k-NN baseline in Tab. VIII, which obtains 21.3% on the long bucket. The recursive event tree in TSHM achieves 37.2% on the same partition, demonstrating a clear advantage over the flatretrieval paradigm that these prior works rely on.

TABLE III RECALL ACCURACY (%) ON THE OMNISTARPRO-LONG PARTITION UNDER THREE MEMORY-SPAN BUCKETS. “S”, “M”, AND “L” DENOTE SHORT (<10 MIN), MEDIUM (10–30 MIN), AND LONG (>30 MIN) SPANS.

<table><tr><td rowspan="2">Method</td><td colspan="3">LMR</td><td colspan="3">CDQ</td><td colspan="3">TBR</td></tr><tr><td>S</td><td>M</td><td>L</td><td>S</td><td>M</td><td>L</td><td>S</td><td>M</td><td>L</td></tr><tr><td>VideoLLM-online</td><td>41.2</td><td>18.6</td><td>6.4</td><td>33.7</td><td>14.1</td><td>5.2</td><td>38.5</td><td>16.0</td><td>5.8</td></tr><tr><td>VideoLLM-MoD</td><td>42.0</td><td>19.3</td><td>6.9</td><td>34.5</td><td>14.8</td><td>5.6</td><td>39.1</td><td>16.7</td><td>6.1</td></tr><tr><td>MMDuet</td><td>47.8</td><td>24.5</td><td>9.1</td><td>39.6</td><td>19.2</td><td>7.4</td><td>44.2</td><td>21.3</td><td>8.0</td></tr><tr><td>LiveStar</td><td>59.5</td><td>33.0</td><td>21.1</td><td>51.3</td><td>28.1</td><td>17.7</td><td>56.0</td><td>30.2</td><td>19.9</td></tr><tr><td>LiveStarPro</td><td>63.4</td><td>49.7</td><td>37.2</td><td>55.1</td><td>42.8</td><td>31.5</td><td>59.8</td><td>46.3</td><td>34.6</td></tr></table>

## C. Offline Experiments

In contrast to online evaluation, where models autonomously determine response timing, our offline experiments on Ego4D, OmniStarPro-RNG, and SVBench operate under pre-defined decoding schedules. To ensure fair comparison, we strictly follow the evaluation protocols and metrics established in prior work [20]–[22], despite their non-generative nature, where evaluation is limited to ground-truth verification. For OmniStarPro, we further go beyond conventional perplexity and token-level accuracy checks by enabling full model generation with online-style scoring, which constitutes a key step toward comprehensive assessment of online capabilities.

Evaluation Metrics. For standardized offline benchmarking, we adopt widely used metrics from prior studies, including PPL, TokAcc (Token Accuracy, previously termed LM-Correctness), TimeDiff, and Fluency [20]–[22]. In addition, we evaluate model outputs generated under offline-prescribed timing using online-style generation metrics, namely SemCor and SumFluen, both of which are scored with GPT-4o.

Results. The left side of Tab. I reports offline evaluation results on OmniStarPro-RNG under fixed decoding timing conditions. The results indicate that LiveStarPro outperforms all other online assistants as well as open-source offline Video-LLMs/LVLMs across all evaluated metrics, although a noticeable gap remains relative to human performance and GPT-4V/4o. Notably, the offline portion of Tab. I also offers a partial controlled signal for the benefit of the SVeD+SCAM paradigm: the InternVideo2.5 backbone evaluated without streaming fine-tuning achieves a SemCor of 4.32, whereas LiveStarPro—built on the same backbone—attains 4.62 under identical offline decoding conditions, a gain that cannot be attributed to backbone differences and therefore reflects the improved video-language alignment afforded by the SCAM training objective. Moving to dialogue and streaming capabilities, Tab. IV presents the evaluation on SVBench. In the zero-shot setting, LiveStarPro attains an average of 52.20, which exceeds both its InternVideo2.5 backbone (49.16) and the streaming counterpart LiveStar (49.76), and approaches the strongest general-purpose open-source LVLMs such as Qwen2.5-VL (55.21) and MiniCPM-V 2.6 (53.15); the remaining gap is expected, since these baselines are trained on substantially larger instruction-tuning corpora than the streamingoriented data used here. Once fine-tuned on the SVBench training set, LiveStar† improves its zero-shot average from 49.76 to 57.41 (a 15.37% relative gain), thereby surpassing all open-source Video-LLMs and approaching the closedsource GPT-4V, which indicates that the streaming-oriented design does not preclude competitive interactive QA when indomain supervision is available. Finally, as shown in Tab. V, LiveStarPro consistently exceeds the performance of other online assistants on the Ego4D benchmark, attaining an 18.1% higher TokAcc relative to the second-best LION-FS.

## D. Conventional Offline Experiments

Although LiveStarPro is optimized for proactive online streaming, we must verify that its specialized design does not compromise general video understanding. Conventional benchmarks such as MVBench [44], LongVideoBench [98], and VideoMME [99] operate under an offline paradigm in which models access the complete video file without realtime latency constraints, revealing whether the model retains spatial-temporal reasoning and long-context comprehension after streaming fine-tuning.

Results. Table VI reports the performance of LiveStarPro on three representative offline benchmarks, alongside state-ofthe-art offline LVLMs and online-capable models. LiveStarPro attains 69.8% on MVBench, 56.3% on LongVideoBench, and 60.8% on VideoMME (w/o subtitles); relative to the InternVideo2.5 backbone it incurs a modest decline rather than a collapse, indicating that the proposed memory and training strategy largely preserve, rather than fully retain, offline reasoning ability. Compared with representative offline models such as LLaVA-OneVision, LiveStarPro reaches higher accuracy on MVBench and stays close to recent large-scale visionlanguage models such as Qwen2.5-VL and InternVL2.5, while still trailing the strongest of them on long-context benchmarks. Among methods that support online inference, LiveStarPro consistently outperforms prior approaches and exceeds VideoChat-Online by 8.0 percentage points on VideoMME. Taken together, these results suggest that the hierarchical memory and the streaming-oriented training strategy keep the offline degradation commonly observed in online video assistants within an acceptable margin, while delivering the proactive streaming capability that the offline models lack.

## E. Ablation Study

Impact of Response-Silence Threshold. LiveStarPro adopts a dynamic response-silence decoding scheme that is governed by an adaptive threshold. Specifically, the decoding

TABLE IV EVALUATION RESULTS OF VARIOUS MODELS ON SVBENCH IN DIALOGUE AND STREAMING EVALUATION. † DENOTES FINE-TUNING ON THE SVBENCH TRAINING SET.

<table><tr><td rowspan="2">Method</td><td colspan="6">Dialogue Evaluation</td><td colspan="6">Streaming Evaluation</td><td rowspan="2">AVG</td></tr><tr><td>SA</td><td>CC</td><td>LC</td><td>TU</td><td>IC</td><td>OS</td><td>SA</td><td>CC</td><td>LC</td><td>TU</td><td>IC</td><td>OS</td></tr><tr><td colspan="14">Closed-source LVLMs</td></tr><tr><td>GPT-4V</td><td>56.03</td><td>62.61</td><td>69.09</td><td>65.36</td><td>53.73</td><td>60.30</td><td>56.37</td><td>61.41</td><td>65.80</td><td>59.18</td><td>57.16</td><td>57.93</td><td>59.12</td></tr><tr><td>GPT-4o</td><td>58.26</td><td>64.76</td><td>70.75</td><td>67.68</td><td>55.82</td><td>62.57</td><td>57.99</td><td>63.52</td><td>67.72</td><td>60.18</td><td>59.25</td><td>59.97</td><td>61.27</td></tr><tr><td colspan="14">Open-source Video-LLMs/LVLMs</td></tr><tr><td>LLaVA-NeXT-Video</td><td>37.71</td><td>44.59</td><td>52.05</td><td>41.80</td><td>36.58</td><td>41.40</td><td>34.29</td><td>39.68</td><td>47.65</td><td>35.33</td><td>36.68</td><td>36.12</td><td>38.76</td></tr><tr><td>InternVL2.5</td><td>43.73</td><td>50.70</td><td>56.61</td><td>55.03</td><td>43.46</td><td>48.73</td><td>40.44</td><td>48.34</td><td>52.84</td><td>46.93</td><td>45.10</td><td>45.04</td><td>46.89</td></tr><tr><td>InternVideo2.5</td><td>46.83</td><td>53.48</td><td>58.22</td><td>58.91</td><td>47.02</td><td>51.73</td><td>41.76</td><td>49.72</td><td>53.25</td><td>48.44</td><td>47.10</td><td>46.58</td><td>49.16</td></tr><tr><td>MiniCPM-V 2.6</td><td>51.70</td><td>59.50</td><td>65.33</td><td>61.72</td><td>50.09</td><td>56.63</td><td>46.44</td><td>52.73</td><td>58.35</td><td>53.48</td><td>48.32</td><td>49.67</td><td>53.15</td></tr><tr><td>Qwen2.5-VL</td><td>52.54</td><td>59.85</td><td>65.52</td><td>64.64</td><td>51.23</td><td>57.57</td><td>48.21</td><td>56.12</td><td>60.31</td><td>56.33</td><td>52.46</td><td>52.84</td><td>55.21</td></tr><tr><td colspan="14">Online Assistants</td></tr><tr><td>LiveStar</td><td>46.43</td><td>53.75</td><td>59.36</td><td>57.29</td><td>45.64</td><td>51.37</td><td>43.56</td><td>51.52</td><td>55.71</td><td>50.79</td><td>47.77</td><td>48.15</td><td>49.76</td></tr><tr><td>LiveStarPro</td><td>47.49</td><td>56.58</td><td>63.86</td><td>59.70</td><td>48.92</td><td>53.72</td><td>44.33</td><td>55.49</td><td>59.83</td><td>52.87</td><td>50.48</td><td>50.68</td><td>52.20</td></tr><tr><td>LiveStar $^{\dagger}$ </td><td>54.06</td><td>61.08</td><td>66.43</td><td>66.06</td><td>52.67</td><td>58.95</td><td>52.19</td><td>59.00</td><td>62.85</td><td>58.35</td><td>54.95</td><td>55.87</td><td>57.41</td></tr></table>

TABLE V OFFLINE EVALUATION ON THE EGO4D NARRATION STREAM BENCHMARK. $^ { 6 6 } - ^ { 9 9 }$ DENOTES INCOMPATIBILITY WITH THE FLUENCY METRIC DUE TO THE ABSENCE OF EOS TOKENS.

<table><tr><td>Method</td><td>PPL↓</td><td>TimeDiff↓</td><td>Fluency↑</td><td>TokAcc↑</td></tr><tr><td>VideoLLM-online</td><td>2.43</td><td>2.04</td><td>45.1%</td><td>48.1%</td></tr><tr><td>VideoLLM-MoD</td><td>2.41</td><td>2.04</td><td>45.2%</td><td>48.9%</td></tr><tr><td>LION-FS</td><td>2.09</td><td>2.15</td><td>46.1%</td><td>52.4%</td></tr><tr><td>MMDuet</td><td>4.51</td><td>1.97</td><td>-</td><td>39.3%</td></tr><tr><td>LiveStar</td><td>1.97</td><td>1.76</td><td>-</td><td>61.1%</td></tr><tr><td>LiveStarPro</td><td>1.95</td><td>1.63</td><td>-</td><td>61.9%</td></tr></table>

TABLE VIPERFORMANCE COMPARISON ON CONVENTIONAL OFFLINE VIDEOUNDERSTANDING BENCHMARKS.

<table><tr><td>Model</td><td>MVBench</td><td>LongVideoBench</td><td>VideoMME (w/o sub.)</td></tr><tr><td colspan="4">Offline Video-LLMs / LVLMs</td></tr><tr><td>GPT-4V</td><td>43.7</td><td>59.1</td><td>59.9</td></tr><tr><td>GPT-4o</td><td>64.6</td><td>66.7</td><td>71.9</td></tr><tr><td>Gemini-1.5-Pro</td><td>60.5</td><td>64.0</td><td>71.9</td></tr><tr><td>LLaVA-NeXT-Video</td><td>53.1</td><td>49.1</td><td>46.5</td></tr><tr><td>LLaVA-OneVision</td><td>56.7</td><td>56.3</td><td>58.2</td></tr><tr><td>VideoLLaMA2</td><td>54.6</td><td>-</td><td>47.9</td></tr><tr><td>Qwen2.5-VL</td><td>69.6</td><td>56.0</td><td>65.1</td></tr><tr><td>InternVL2.5</td><td>72.0</td><td>60.0</td><td>64.2</td></tr><tr><td colspan="4">Online Assistants</td></tr><tr><td>MovieChat</td><td>55.1</td><td>56.3</td><td>38.2</td></tr><tr><td>VideoChat-Online</td><td>64.9</td><td>-</td><td>52.8</td></tr><tr><td>Dispider</td><td>-</td><td>-</td><td>57.2</td></tr><tr><td>LiveStarPro</td><td>69.8</td><td>56.3</td><td>60.8</td></tr></table>

threshold is formulated as $\alpha \cdot \mathrm { P P L } ^ { t _ { i } } ( [ D e c ] )$ , where $\alpha \geq 1$ denotes a tunable scaling factor applied to the timestepdependent perplexity. To systematically examine the sensitivity of the model to $\alpha ,$ we perform an empirical study over the range [1.0, 1.1] on the OmniStarPro-RNG benchmark. Results from online evaluation (Fig. 7) highlight the pivotal role of α in balancing timing difference, timing redundancy, and timing coverage. The experiments indicate that optimal performance is achieved within a narrow interval of $\alpha = 1 . 0 2 – 1 . 0 4 .$ , and we select $\alpha = 1 . 0 3$ as the default setting. The narrowness of this optimal range reflects a well-understood property of perplexity-based thresholds: because perplexity is computed relative to the LM’s own probability distribution, its absolute scale varies across model families and domains, and α essentially normalizes a relative drift rather than an absolute magnitude. In deployment, α can therefore be calibrated efficiently on a small held-out stream (typically fewer than 100 frames) to identify the optimal operating point for a given domain, and this calibration cost is negligible compared to the main training procedure.

![](images/ef4a466d58e990c5464f817a5f19dec8c60c0e9a85e97fd6a2d0d6a9296832e5.jpg)

<details>
<summary>line chart</summary>

| Tunable Threshold | TimDiff | TimRedun | TimCover |
| ----------------- | ------- | -------- | -------- |
| 1.00              | 0.50    | 2.00     | 0.90     |
| 1.01              | 0.70    | 1.75     | 0.88     |
| 1.02              | 0.95    | 1.55     | 0.85     |
| 1.03              | 1.10    | 1.35     | 0.82     |
| 1.04              | 1.30    | 1.15     | 0.78     |
| 1.05              | 1.45    | 1.00     | 0.75     |
| 1.06              | 1.60    | 0.90     | 0.72     |
| 1.07              | 1.75    | 0.85     | 0.70     |
| 1.08              | 2.00    | 0.75     | 0.68     |
| 1.09              | 2.10    | 0.72     | 0.65     |
| 1.10              | 2.20    | 0.68     | 0.62     |
</details>

Fig. 7. Ablation study on the impact of response-silence threshold.

Effect of Memory and Caching Strategies. Tab. VII summarizes ablation results for different memory management schemes and KV cache strategies. For memory strategies: (1) Uniform dropout leads to a 4.70% decrease in SemCor due to the indiscriminate removal of critical recent frames; (2) FIFObased forgetting disrupts temporal reasoning by discarding historical event captions, resulting in a 9.42% increase in TimDiff and a 3.76% reduction in SemCor; (3) Our Peak-End Memory Compression retains semantic clip summaries and keyframes via precomputed PPL-based dropout, achieving the best SemCor and the lowest TimDiff among short-term strategies; (4) Furthermore, incorporating TSHM for long-term retrieval boosts SemCor to 3.27, demonstrating that preserving evicted history in a recursive tree effectively mitigates catastrophic forgetting with only a marginal FPS trade-off. For KV caching, disabling both inter-/intra-dialogue caching or intra-dialogue caching alone yields suboptimal throughput. In contrast, for the short-term Peak-End configuration, enabling both strategies improves inference throughput by 1.53× over no caching (3.82 vs. 2.50 FPS) and by 1.31× over intradialogue caching alone (3.82 vs. 2.92 FPS) under 5-minute video inference; the full LiveStarPro with TSHM further reaches 3.96 FPS, a 1.58× speedup over no caching. This avoids redundant recomputation of historical representations, thereby supporting low-latency inference without compromising output quality.

TABLE VII ABLATION STUDY ON MEMORY STRATEGIES AND KV CACHE FOR INFERENCE ON OMNISTARPRO-RNG.

<table><tr><td>Memory Strategy</td><td>KV Cache</td><td>SemCor↑</td><td>TimDiff↓</td><td>FPS↑</td></tr><tr><td rowspan="2">Uniform DropoutFIFO Forgetting</td><td rowspan="2">Both</td><td>3.04</td><td>2.01</td><td>3.77</td></tr><tr><td>3.07</td><td>2.09</td><td>3.91</td></tr><tr><td rowspan="3">Peak-End (Short-term)</td><td rowspan="2">Neitherw/o Inter-Dialog</td><td>3.19</td><td>1.95</td><td>2.50</td></tr><tr><td>3.17</td><td>1.87</td><td>2.92</td></tr><tr><td>Both</td><td>3.19</td><td>1.91</td><td>3.82</td></tr><tr><td>TSHM (Long-term)</td><td>Both</td><td>3.27</td><td>1.89</td><td>3.96</td></tr></table>

TABLE VIII LONG-TERM RETRIEVAL DIAGNOSTICS ON LMR OF OMNISTARPRO-LONG. RECALL ACCURACY (%) ON THE SHORT/LONG MEMORY-SPAN BUCKETS AND AVERAGE PER-QUERY RETRIEVAL LATENCY (MS).

<table><tr><td>Long-term Memory</td><td>Recall (S)↑</td><td>Recall (L)↑</td><td>Latency (ms)↓</td></tr><tr><td>None (sliding window)</td><td>41.2</td><td>6.4</td><td>-</td></tr><tr><td>Flat k-NN bank</td><td>58.7</td><td>21.3</td><td>38.6</td></tr><tr><td>Recursive Event Tree</td><td>63.4</td><td>37.2</td><td>12.4</td></tr></table>

Long-term Retrieval Diagnostics. We further dissect the long-term retrieval memory of TSHM on the LMR task of the OmniStarPro-Long partition, where the supporting evidence of a query frequently resides beyond the active context window. Tab. VIII contrasts the recursive event tree against a flat knearest-neighbor memory bank that stores every evicted unit without hierarchical organization; the recall accuracy of the recursive event tree therefore coincides with the LMR row of Tab. III, while the two baselines isolate the contribution of the hierarchical organization. The flat index attains a comparable recall on the short memory-span bucket, yet it degrades markedly on the long bucket and incurs a retrieval latency that grows linearly with the number of stored units. By contrast, the recursive event tree sustains a substantially higher recall on the long bucket and a sublinear retrieval latency, which empirically corroborates the logarithmic-retrieval property established in Sec. III-D4. The benefit originates from the parent and child traversal, which recovers coherent event chains rather than isolated units.

Sensitivity of Tree Hyper-parameters. The structure of the recursive event tree is governed by the similarity threshold σ, which decides whether an evicted unit is attached as a child or initialized as a new root, and by the momentum factor $\beta ,$ which controls how rapidly a parent embedding evolves toward the centroid of its descendants. Tab. IX reports the long-bucket recall together with the resulting average branching factor and tree height as σ and β vary. A small σ produces shallow and wide trees that collapse semantically distinct events into a single branch and thereby dilute retrieval precision, whereas an excessively large σ fragments the memory into many isolated roots and reverts to a flat index. A moderate momentum factor allows parent embeddings to summarize their subtrees without erasing discriminative detail. The configuration σ = 0.75 and β = 0.3 yields the most favorable balance between branching factor and height and attains the highest long-bucket recall, and we adopt it as the default setting.

TABLE IX SENSITIVITY OF THE RECURSIVE EVENT TREE TO σ AND β. “BF” AND “H” DENOTE THE AVERAGE BRANCHING FACTOR AND TREE HEIGHT.

<table><tr><td>σ</td><td>β</td><td>BF</td><td>H</td><td>Recall (L)↑</td></tr><tr><td>0.65</td><td>0.3</td><td>6.8</td><td>3.1</td><td>31.9</td></tr><tr><td>0.75</td><td>0.1</td><td>3.9</td><td>5.4</td><td>34.7</td></tr><tr><td>0.75</td><td>0.3</td><td>4.1</td><td>5.2</td><td>37.2</td></tr><tr><td>0.75</td><td>0.5</td><td>4.0</td><td>5.3</td><td>35.6</td></tr><tr><td>0.85</td><td>0.3</td><td>2.2</td><td>8.7</td><td>33.1</td></tr></table>

## F. Case Study

We conduct qualitative comparisons on the RNG task between our LiveStarPro model and representative online video understanding baselines, namely VideoLLM-online and MMDuet. The qualitative results are illustrated in Fig. 8. These examples reveal that VideoLLM-online and MMDuet frequently exhibit limited contextual reasoning, hallucinated descriptions, and inadequate fine-grained recognition. By contrast, LiveStarPro consistently produces responses that are more accurate, more firmly grounded, and more temporally appropriate, owing to the effective integration of long-range context with streaming visual evidence. These qualitative comparisons underscore the advantage of LiveStarPro on fine-grained and context-sensitive online video understanding, while they also expose the limitations of existing LVLMs under realistic streaming settings.

## VI. CONCLUSION

This paper introduces LiveStarPro, a live streaming assistant that is conceived to deliver proactive responsiveness through adaptive streaming decoding. Around a streaming response– silence paradigm, we contribute three coordinated innovations: (1) a Streaming Verification Decoding (SVeD) mechanism that determines the appropriate response timing during realtime inference through single-pass perplexity verification, (2) a streaming video–language alignment framework that incorporates Streaming Causal Attention Masks (SCAM) during training, and (3) a Tree-Structured Hierarchical Memory (TSHM) that organizes evicted history into retrievable event chains for reasoning over long-horizon streams. Extensive experiments across three benchmark datasets demonstrate that LiveStarPro attains state-of-the-art performance in online video understanding while preserving practical deployment efficiency, sustaining throughput of approximately 3 FPS on

![](images/b5ff11d2d0102240ddecf26c4a29ee573190a9bd81b52f3a1edb5f23bfdfeb40.jpg)

<details>
<summary>text_image</summary>

00:00
00:02
00:04
00:06
00:08
</details>

<table><tr><td>Methods</td><td>00:00</td><td>00:02</td><td>00:04</td><td>00:06</td><td>00:08</td></tr><tr><td>Ground Truth</td><td>Hands are under a black faucet with orange liquid on them. A light-purple fluffy scrunchie is on the wrist. A red duck ornament and a small potted plant are beside the sink.</td><td>The hand sweeps upward under the black faucet. White foam flows from the raised hand onto the light plush wristband.</td><td>The scene shifts to a woman in a red shirt and yellow headband at the sink, washing her face.</td><td>The camera switches again. One hand adjusts the cuff or wristband on the other hand. Beige clothesand pink-and-white checkered cuffs are visible.</td><td>Finally, four plush wristbands of different colors are neatly arranged on the bathroom wall.</td></tr><tr><td>VideoLLM-online</td><td>You hold a bottle.</td><td>You rub your hands.</td><td>You rub your hands.</td><td>You rub your hands.</td><td>You rub your hands.</td></tr><tr><td>MMDuet</td><td>Slient. (nothing output)</td><td>Slient. (nothing output)</td><td>Apply the cream to the face.</td><td>Slient. (nothing output)</td><td>Slient. (nothing output)</td></tr><tr><td>LiveStarPro</td><td>A hand with a pink wristband appears under the faucet scene ,with a duck-shaped ornamentand a potted plant nearby.</td><td>The pink-wristband hand is under running water; the white liquid on the palm is gradually washed away.</td><td>The scene switches to a bathroom. A woman in a pink T-shirt stands at the sink washing her face.</td><td>The scene switches to the woman&#x27;s hands. She wears pink-and-white wristbands, and one hand adjusts the cuff of the other.</td><td>The scene switches to a graywall, where the woman arranges four wristbands of different colors.</td></tr></table>

![](images/2d3d7f0a76d1047df65141cc5025698a5f60a34d7859e9a3f5161f99a303cb48.jpg)

## Takeaway

This case shows that VideoLLM-online and MMDuet perform poorly on the RNG task. VideoLLM-online’s answers are brief, repetitive, and limited to a first-person viewpoint, while MMDuet often misses key points and sometimes outputs nothing. In contrast, LiveStarPro provides more timely and precise understanding of the video and soccasionally captures details beyond the annotations.

![](images/d260a203aaeb65b10b2fc0d0ba25ca332969e0ca8ebc49dcf82628319b1092fc.jpg)  
Fig. 8. Comparison on the RNG task. LiveStarPro is timely and precise, while VideoLLM-online is repetitive and MMDuet often misses key points.

hour-long streams. On the OmniStarPro-Long partition, LiveStarPro further sustains reliable recall across all three memorycentric tasks (long-range memory recall, cross-event difference query, and temporal backtracking), confirming that TSHM effectively mitigates catastrophic forgetting for evidence that lies well beyond the active context window. By advancing a new response–silence paradigm together with the OmniStarPro benchmark, this work lays groundwork for robust, scalable models capable of long-horizon online video understanding.

## REFERENCES

[1] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu et al., “Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling,” arXiv preprint arXiv:2412.05271, 2024.  
[2] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin, “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.  
[3] Y. Yao, T. Yu, A. Zhang, C. Wang, J. Cui, H. Zhu, T. Cai, H. Li, W. Zhao, Z. He et al., “Minicpm-v: A gpt-4v level mllm on your phone,” arXiv preprint arXiv:2408.01800, 2024.  
[4] P. Zhang, X. Dong, Y. Zang, Y. Cao, R. Qian, L. Chen, Q. Guo, H. Duan, B. Wang, L. Ouyang et al., “Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output,” arXiv preprint arXiv:2407.03320, 2024.  
[5] T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Zhang, D. Rojas, G. Feng, H. Zhao et al., “Chatglm: A family of large language models from glm-130b to glm-4 all tools,” arXiv preprint arXiv:2406.12793, 2024.  
[6] K. Ataallah, X. Shen, E. Abdelrahman, E. Sleiman, D. Zhu, J. Ding, and M. Elhoseiny, “Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens,” arXiv preprint arXiv:2404.03413, 2024.  
[7] M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” arXiv preprint arXiv:2306.05424, 2023.  
[8] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,” arXiv preprint arXiv:2305.06355, 2023.  
[9] A. Yang, A. Nagrani, P. H. Seo, A. Miech, J. Pont-Tuset, I. Laptev, J. Sivic, and C. Schmid, “Vid2seq: Large-scale pretraining of a visual language model for dense video captioning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10 714–10 726.  
[10] Y. Wang, K. Li, Y. Li, Y. He, B. Huang, Z. Zhao, H. Zhang, J. Xu, Y. Liu, Z. Wang et al., “Internvideo: General video foundation models via generative and discriminative learning,” arXiv preprint arXiv:2212.03191, 2022.  
[11] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao et al., “Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms,” arXiv preprint arXiv:2406.07476, 2024.  
[12] Z. Liu, Y. Dong, Z. Liu, W. Hu, J. Lu, and Y. Rao, “Oryx mllm: Ondemand spatial-temporal understanding at arbitrary resolution,” arXiv preprint arXiv:2409.12961, 2024.  
[13] S. Ren, L. Yao, S. Li, X. Sun, and L. Hou, “Timechat: A time-sensitive multimodal large language model for long video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 313–14 323.  
[14] H. Zhang, Y. Wang, Y. Tang, Y. Liu, J. Feng, J. Dai, and X. Jin, “Flash-vstream: Memory-based real-time understanding for long video streams,” arXiv preprint arXiv:2406.08085, 2024.  
[15] E. Song, W. Chai, T. Ye, J.-N. Hwang, X. Li, and G. Wang, “Moviechat+: Question-aware sparse memory for long video question answering,” arXiv preprint arXiv:2404.17176, 2024.  
[16] B. He, H. Li, Y. K. Jang, M. Jia, X. Cao, A. Shah, A. Shrivastava, and S.-N. Lim, “Ma-lmm: Memory-augmented large multimodal model for long-term video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 504–13 514.  
[17] X. Wang, D. Song, S. Chen, C. Zhang, and B. Wang, “Longllava: Scaling multi-modal llms to 1000 images efficiently via hybrid architecture,” 2024. [Online]. Available: https://arxiv.org/abs/2409.02889  
[18] F. Xue, Y. Chen, D. Li, Q. Hu, L. Zhu, X. Li, Y. Fang, H. Tang, S. Yang, Z. Liu, Y. He, H. Yin, P. Molchanov, J. Kautz, L. Fan, Y. Zhu, Y. Lu, and S. Han, “Longvila: Scaling long-context visual language models for long videos,” null, 2024.  
[19] P. Zhang, K. Zhang, B. Li, G. Zeng, J. Yang, Y. Zhang, Z. Wang, H. Tan, C. Li, and Z. Liu, “Long context transfer from language to vision,” arXiv preprint arXiv:2406.16852, 2024.  
[20] J. Chen, Z. Lv, S. Wu, K. Q. Lin, C. Song, D. Gao, J.-W. Liu, Z. Gao, D. Mao, and M. Z. Shou, “Videollm-online: Online video large language model for streaming video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 407–18 418.  
[21] S. Wu, J. Chen, K. Q. Lin, Q. Wang, Y. Gao, Q. Xu, T. Xu, Y. Hu, E. Chen, and M. Z. Shou, “Videollm-mod: Efficient video-language streaming with mixture-of-depths vision computation,” Advances in Neural Information Processing Systems, vol. 37, pp. 109 922–109 947, 2024.  
[22] W. Li, B. Hu, R. Shao, L. Shen, and L. Nie, “Lion-fs: Fast & slow video-language thinker as online video assistant,” arXiv preprint arXiv:2503.03663, 2025.  
[23] X. Ding, H. Wu, Y. Yang, S. Jiang, D. Bai, Z. Chen, and T. Cao, “Streammind: Unlocking full frame rate streaming video dialogue through eventgated cognition,” arXiv preprint arXiv:2503.06220, 2025.  
[24] Y. Wang, X. Meng, Y. Wang, J. Liang, J. Wei, H. Zhang, and D. Zhao, “Videollm knows when to speak: Enhancing time-sensitive video comprehension with video-text duet interaction format,” arXiv preprint arXiv:2411.17991, 2024.  
[25] R. Qian, X. Dong, P. Zhang, Y. Zang, S. Ding, D. Lin, and J. Wang, “Streaming long video understanding with large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 119 336– 119 360, 2024.  
[26] X. Zhou, A. Arnab, S. Buch, S. Yan, A. Myers, X. Xiong, A. Nagrani, and C. Schmid, “Streaming dense video captioning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 243–18 252.  
[27] H. Xiong, Z. Yang, J. Yu, Y. Zhuge, L. Zhang, J. Zhu, and H. Lu, “Streaming video understanding and multi-round interaction with memory-enhanced knowledge,” arXiv preprint arXiv:2501.13468, 2025.  
[28] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu et al., “Ego4d: Around the world in 3,000 hours of egocentric video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18 995–19 012.  
[29] S. Giancola, M. Amine, T. Dghaily, and B. Ghanem, “Soccernet: A scalable dataset for action spotting in soccer videos,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 1711–1721.  
[30] Z. Yang, Y. Hu, Z. Du, D. Xue, S. Qian, J. Wu, F. Yang, W. Dong, and C. Xu, “Svbench: A benchmark with temporal multi-turn dialogues for streaming video understanding,” arXiv preprint arXiv:2502.10810, 2025.  
[31] Y. Li, J. Niu, Z. Miao, C. Ge, Y. Zhou, Q. He, X. Dong, H. Duan, S. Ding, R. Qian et al., “Ovo-bench: How far is your videollms from real-world online video understanding?” arXiv preprint arXiv:2501.05510, 2025.  
[32] Z. Yang, K. Zhang, Y. Hu, B. Wang, S. Qian, B. Wen, F. Yang, T. Gao, W. Dong, and C. Xu, “Livestar: Live streaming assistant for real-world online video understanding,” Advances in Neural Information Processing Systems, vol. 38, pp. 31 266–31 304, 2026.  
[33] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.  
[34] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.  
[35] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.  
[36] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27 730–27 744, 2022.  
[37] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al., “Improving language understanding by generative pre-training,” Unknown, 2018.  
[38] X. Wang, J. Wu, Z. Lin, F. Zhang, D. Zhang, and L. Nie, “Video dataflywheel: Resolving the impossible data trinity in video-language understanding,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.  
[39] Y. Zhou, H. Zhang, S.-I. Park, B. Yoo, and X. Qi, “Object-centric representation learning for video scene understanding,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 8410– 8423, 2024.  
[40] L. Chen, X. Wei, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, Z. Tang, L. Yuan et al., “Sharegpt4video: Improving video understanding and generation with better captions,” Advances in Neural Information Processing Systems, vol. 37, pp. 19 472–19 495, 2024.  
[41] L. Xu, Y. Zhao, D. Zhou, Z. Lin, S. K. Ng, and J. Feng, “Pllava: Parameter-free llava extension from images to videos for video dense captioning,” arXiv preprint arXiv:2404.16994, 2024.  
[42] M. M. Islam, N. Ho, X. Yang, T. Nagarajan, L. Torresani, and G. Bertasius, “Video recap: Recursive captioning of hour-long videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 198–18 208.  
[43] D. Ko, J. S. Lee, W. Kang, B. Roh, and H. J. Kim, “Large language models are temporal and causal reasoners for video question answering,” arXiv preprint arXiv:2310.15747, 2023.  
[44] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo et al., “Mvbench: A comprehensive multi-modal video understanding benchmark,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 195–22 206.  
[45] M. Maaz, H. Rasheed, S. Khan, and F. Khan, “Videogpt+: Integrating image and video encoders for enhanced video understanding,” arXiv preprint arXiv:2406.09418, 2024.  
[46] A. Yang, A. Miech, J. Sivic, I. Laptev, and C. Schmid, “Learning to answer visual questions from web videos,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 3202–3218, 2025.  
[47] Y. Li, X. Wang, J. Xiao, W. Ji, and T.-S. Chua, “Transformer-empowered invariant grounding for video question answering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 11, pp. 9510– 9522, 2025.  
[48] J. Li, P. Wei, W. Han, S.-C. Zhu, and L. Fan, “Intentqa: Intent question answering in videos by cognitive context reasoning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–18, 2026.  
[49] Y. Guo, J. Liu, M. Li, D. Cheng, X. Tang, D. Sui, Q. Liu, X. Chen, and K. Zhao, “Vtg-llm: Integrating timestamp knowledge into video llms for enhanced video temporal grounding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, 2025, pp. 3302– 3310.  
[50] Y. Xu, Y. Sun, Z. Xie, B. Zhai, and S. Du, “Vtg-gpt: Tuning-free zeroshot video temporal grounding with gpt,” Applied Sciences, vol. 14, no. 5, p. 1894, 2024.  
[51] Y. Wang, X. Meng, J. Liang, Y. Wang, Q. Liu, and D. Zhao, “Hawkeye: Training video-text llms for grounding text in videos,” arXiv preprint arXiv:2403.10228, 2024.  
[52] Y. Zhang, B. Li, h. Liu, Y. j. Lee, L. Gui, D. Fu, J. Feng, Z. Liu, and C. Li, “Llava-next: A strong zero-shot video understanding model,” April 2024. [Online]. Available: https://llava-vl.github.io/blog/2024-04- 30-llava-next-video/  
[53] B. Lin, B. Zhu, Y. Ye, M. Ning, P. Jin, and L. Yuan, “Video-llava: Learning united visual representation by alignment before projection,” arXiv preprint arXiv:2311.10122, 2023.  
[54] J. Lin, H. Yin, W. Ping, P. Molchanov, M. Shoeybi, and S. Han, “Vila: On pre-training for visual language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 689–26 699.  
[55] M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.  
[56] J. Liu, S. Chen, X. He, L. Guo, X. Zhu, W. Wang, and J. Tang, “Valor: Vision-audio-language omni-perception pretraining model and dataset,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 2, pp. 708–724, 2025.  
[57] W. Wu, X. Wang, H. Luo, J. Wang, Y. Yang, and W. Ouyang, “Cap4video++: Enhancing video understanding with auxiliary captions,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 7, pp. 5223–5237, 2025.  
[58] P. Jin, H. Li, L. Yuan, S. Yan, and J. Chen, “Hierarchical banzhaf interaction for general video-language representation learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 3, pp. 2125–2139, 2025.  
[59] X. Wang, J. Wu, Z. Lin, F. Zhang, D. Zhang, and L. Nie, “Video dataflywheel: Resolving the impossible data trinity in video-language understanding,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 4, pp. 2912–2923, 2025.  
[60] J. Gao, Y. Lian, Z. Zhou, Y. Fu, and B. Wang, “Livechat: A largescale personalized dialogue dataset automatically constructed from live streaming,” arXiv preprint arXiv:2306.08401, 2023.  
[61] Z. Yang, K. Zhang, S. Qian, W. Dong, and C. Xu, “Don’t pause: Streaming video-language synchrony for online video understanding,” arXiv preprint arXiv:2606.06991, 2026.  
[62] K. Zhang, Z. Yang, B. Wang, S. Qian, and C. Xu, “Querystream: Advancing streaming video understanding with query-aware pruning  
and proactive response,” in The Fourteenth International Conference on Learning Representations, 2026.  
[63] B. Zhu, B. Lin, M. Ning, Y. Yan, J. Cui, H. Wang, Y. Pang, W. Jiang, J. Zhang, Z. Li et al., “Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment,” arXiv preprint arXiv:2310.01852, 2023.  
[64] K. Mangalam, R. Akshulakov, J. Malik et al., “Egoschema: A diagnostic benchmark for very long-form video language understanding,” Advances in Neural Information Processing Systems, vol. 36, 2024.  
[65] Z. Yu, D. Xu, J. Yu, T. Yu, Z. Zhao, Y. Zhuang, and D. Tao, “Activitynetqa: A dataset for understanding complex web videos via question answering,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, 2019, pp. 9127–9134.  
[66] L. Li, Y.-C. Chen, Y. Cheng, Z. Gan, L. Yu, and J. Liu, “Hero: Hierarchical encoder for video+ language omni-representation pre-training,” arXiv preprint arXiv:2005.00200, 2020.  
[67] V. Patraucean, L. Smaira, A. Gupta, A. Recasens, L. Markeeva, D. Banarse, S. Koppula, M. Malinowski, Y. Yang, C. Doersch et al., “Perception test: A diagnostic benchmark for multimodal video models,” Advances in Neural Information Processing Systems, vol. 36, 2024.  
[68] A. Zadeh, M. Chan, P. P. Liang, E. Tong, and L.-P. Morency, “Socialiq: A question answering benchmark for artificial social intelligence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 8807–8817.  
[69] D. Xu, Z. Zhao, J. Xiao, F. Wu, H. Zhang, X. He, and Y. Zhuang, “Video question answering via gradually refined attention over appearance and motion,” in Proceedings of the 25th ACM international conference on Multimedia, 2017, pp. 1645–1653.  
[70] J. Lei, L. Yu, M. Bansal, and T. L. Berg, “Tvqa: Localized, compositional video question answering,” arXiv preprint arXiv:1809.01696, 2018.  
[71] J. Xiao, X. Shang, A. Yao, and T.-S. Chua, “Next-qa: Next phase of question-answering to explaining temporal actions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 9777–9786.  
[72] E. Song, W. Chai, G. Wang, Y. Zhang, H. Zhou, F. Wu, H. Chi, X. Guo, T. Ye, Y. Zhang et al., “Moviechat: From dense token to sparse memory for long video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 221–18 232.  
[73] W. Wang, Z. He, W. Hong, Y. Cheng, X. Zhang, J. Qi, S. Huang, B. Xu, Y. Dong, M. Ding et al., “Lvbench: An extreme long video understanding benchmark,” arXiv preprint arXiv:2406.08035, 2024.  
[74] Y. Jang, Y. Song, Y. Yu, Y. Kim, and G. Kim, “Tgif-qa: Toward spatiotemporal reasoning in visual question answering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2758–2766.  
[75] E. Song, W. Chai, T. Ye, J.-N. Hwang, X. Li, and G. Wang, “Moviechat+: Question-aware sparse memory for long video question answering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 1, pp. 374–389, 2026.  
[76] J. Li, M. Gao, X. He, S. Tang, W.-S. Zheng, J. Xiao, M. Wang, T.-S. Chua, and Y. Zhuang, “Momentor++: Advancing video large language models with fine-grained long video reasoning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 6, pp. 6208– 6224, 2026.  
[77] K. Zhang, Z. Yang, M. Han, Y. Zhuge, H. Hao, C. Li, Z. Li, and X. Chang, “Selongvlm: Empowering long video language models with self-corrective clip selection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–16, 2026.  
[78] S. Tian, R. Wang, H. Guo, P. Wu, Y. Dong, X. Wang, J. Yang, H. Zhang, H. Zhu, and Z. Liu, “Ego-r1: Agentic chain-of-tool-thought for ultralong egocentric video reasoning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–16, 2026.  
[79] S. A. Peirone, F. Pistilli, A. Alliegro, T. Tommasi, and G. Averta, “Hier-egopack: Hierarchical egocentric video understanding with diverse task perspectives,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1917–1931, 2026.  
[80] Z. Yang, Z. Du, S. Qian, and C. Xu, “Never seen before: Benchmarking genuine zero-shot composed image retrieval with consistent videosourced datasets,” arXiv preprint arXiv:2606.07032, 2026.  
[81] L. Hong, Z. Liu, W. Chen, C. Tan, Y. Feng, X. Zhou, P. Guo, J. Li, Z. Chen, S. Gao, W. Zhang, and W. Zhang, “Lvos: A benchmark for large-scale long-term video object segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 1, pp. 946–961, 2026.  
[82] S. Yang, W. Yu, W. Yang, X. Liu, H. Tan, L. Lan, and N. Xiao, “Wildvideo: Benchmarking lmms for understanding video-language interaction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 10, pp. 9330–9344, 2025.  
[83] J. Wu, W. Liu, Y. Liu, M. Liu, L. Nie, Z. Lin, and C. W. Chen, “A survey on video temporal grounding with multimodal large language model,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 2, pp. 1521–1541, 2026.  
[84] L. Yao, Y. Li, Y. Wei, L. Li, S. Ren, Y. Liu, K. Ouyang, L. Wang, S. Li, S. Li et al., “Timechat-online: 80% visual tokens are naturally redundant in streaming videos,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 10 807–10 816.  
[85] Y. Wang, Y. Song, C. Xie, Y. Liu, and Z. Zheng, “Videollamb: Long streaming video understanding with recurrent memory bridges,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 24 170–24 181.  
[86] D. Kahneman, “Evaluation by moments: Past and future,” Choices, values, and frames, pp. 693–708, 2000.  
[87] Z. Yang, D. Xue, S. Qian, W. Dong, and C. Xu, “Ldre: Llm-based divergent reasoning and ensemble for zero-shot composed image retrieval,” in Proceedings of the 47th International ACM SIGIR conference on research and development in information retrieval, 2024, pp. 80–90.  
[88] Z. Yang, S. Qian, D. Xue, J. Wu, F. Yang, W. Dong, and C. Xu, “Semantic editing increment benefits zero-shot composed image retrieval,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 1245–1254.  
[89] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892– 34 916, 2023.  
[90] Y. Hu, Z. Yang, S. Wang, S. Qian, B. Wen, F. Yang, T. Gao, and C. Xu, “Streamingcot: A dataset for temporal dynamics and multimodal chainof-thought reasoning in streaming videoqa,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 13 464–13 470.  
[91] M. Han, L. Yang, X. Chang, and H. Wang, “Shot2story20k: A new benchmark for comprehensive understanding of multi-shot videos,” arXiv preprint arXiv:2312.10300, 2023.  
[92] Y. Wang, X. Li, Z. Yan, Y. He, J. Yu, X. Zeng, C. Wang, C. Ma, H. Huang, J. Gao et al., “Internvideo2. 5: Empowering video mllms with long and rich context modeling,” arXiv preprint arXiv:2501.12386, 2025.  
[93] Z. Cai, M. Cao, H. Chen, K. Chen, K. Chen, X. Chen, X. Chen, Z. Chen, Z. Chen, P. Chu et al., “Internlm2 technical report,” arXiv preprint arXiv:2403.17297, 2024.  
[94] Z. Yang, L. Li, K. Lin, J. Wang, C.-C. Lin, Z. Liu, and L. Wang, “The dawn of lmms: Preliminary explorations with gpt-4v (ision),” arXiv preprint arXiv:2309.17421, vol. 9, no. 1, p. 1, 2023.  
[95] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.  
[96] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing et al., “Judging llm-as-a-judge with mt-bench and chatbot arena,” Advances in neural information processing systems, vol. 36, pp. 46 595–46 623, 2023.  
[97] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, “G-eval: Nlg evaluation using gpt-4 with better human alignment,” in Proceedings of the 2023 conference on empirical methods in natural language processing, 2023, pp. 2511–2522.  
[98] H. Wu, D. Li, B. Chen, and J. Li, “Longvideobench: A benchmark for long-context interleaved video-language understanding,” Advances in Neural Information Processing Systems, vol. 37, pp. 28 828–28 857, 2024.  
[99] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang et al., “Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 24 108–24 118.