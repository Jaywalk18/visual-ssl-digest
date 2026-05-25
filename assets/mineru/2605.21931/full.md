# EvoVid: Temporal-Centric Self-Evolution for Video Large Language Models

Shiqi Huang1, Ziyue Wang2, Zhongrong Zuo2, Han Qiu2, Qi She2, Bihan Wen1∗

1School of Electrical and Electronic Engineering, Nanyang Technological University 2ByteDance

# Abstract

Recent Video Large Language Models (Video-LLMs) have demonstrated strong capabilities in video reasoning through reinforcement learning (RL). However, existing RL pipelines rely heavily on human-annotated tasks and solutions, making them costly to scale and fundamentally constrained by human expertise. Selfevolving frameworks have recently emerged as a promising alternative through autonomous Questioner–Solver self-play. Unfortunately, these approaches are primarily designed for static modalities such as text and images, fundamentally failing to capture the temporal dynamics that are central to video reasoning. In this work, we propose EvoVid, a temporal-centric self-evolving framework that enables Video-LLMs to improve directly from raw, unannotated videos. Specifically, we introduce two complementary temporal-centric rewards: a temporal-aware Questioner reward that encourages temporally dependent question generation through temporal perturbation sensitivity, and a temporal-grounded Solver reward that provides automatic temporal supervision via inherent video segment localization. Extensive experiments across four base models and six benchmarks demonstrate consistent improvements over both base models and existing self-evolving baselines, achieving competitive performance with supervised methods. These results highlight temporal-centric self-evolution as an effective and scalable paradigm for video understanding and reasoning.

# 1 Introduction

Recent Video Large Language Models (Video-LLMs) have demonstrated strong capabilities in video understanding and reasoning, often enabled by reinforcement learning (RL) [1–5]. Despite this progress, these advancements still rely on human-annotated tasks and solutions to construct verifiable reward signals, making them costly to scale and inherently bounded by human expertise [6, 7]. This bottleneck is especially acute in the video domain, where the intricate interplay of spatial variance and temporal causality makes exhaustive, high-quality annotation exceptionally difficult [8].

To bypass this reliance on external supervision, self-evolving models have recently emerged as a promising alternative. Instead of learning from predefined annotations, these frameworks improve through iterative Questioner–Solver self-play, where both training tasks and learning signals are generated inherently during the RL process [7, 9–11]. While the self-evolving paradigm has shown strong effectiveness in Large Language Models (LLMs) [6, 7, 9] and is gaining increasing attention in Multimodal Large Language Models (MLLMs) [10, 11], existing frameworks are primarily designed for static modalities such as text and images, failing to account for the temporal dynamics inherent to video. Consequently, directly extending these methods to video often results in temporally insensitive and modality-agnostic self-evolution.

![](images/210045c6b6375f893b3237f93c2041649f274e30e4ae0cfa6aa9622141b7f930.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Human-annotated Tasks and Solutions"] --> B["Solver"]
    B --> C["Costly to scale and bounded by human expertise"]
    C --> D["Image: Software icon with dollar bill, clock, group of people, dollar sign"]
    D --> E["Arrow pointing to the robot"]
```
</details>

![](images/4d4435ce18933d1f58edf33439952fa974e57efc73003f2417e465f651c19ea1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image-based Self-Evolution"] --> B["Questioner"]
    A --> C["Solver"]
    B --> D["Temporally-insensitive Questioner and Solver"]
    C --> D
    D --> E["Generic, single-frame answerable"]
    E --> F["Self-play loop"]
    F --> B
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
```
</details>

![](images/df7072b85786da0639f171cab7a0be6053afde914b86af4e673e333245133635.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Questioner"] --> B["Self-play loop"]
    C["Solver"] --> B
    B --> D["Temporal-Aware Questioner"]
    B --> E["Temporal-Grounded Solver"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#ccf,stroke:#333
```
</details>

Figure 1: Comparison between supervised RL, VANILLA self-evolving frameworks, and EvoVid. Left: Supervised RL relies on human-annotated tasks and solutions to construct reward signals, making training costly and inherently bounded by human expertise. Middle: VANILLA self-evolving frameworks, primarily designed for static modalities, i.e., images, generate generic and often singleframe answerable questions, leading to temporally insensitive Questioner–Solver self-play. Right: EvoVid introduces video-based self-evolution through a temporal-aware Questioner and a temporalgrounded Solver, enabling temporal-centric self-evolution directly from raw, unannotated videos.

As shown in Figure 1, existing VANILLA self-evolving frameworks typically generate generic questions that can be answered from a single frame or static appearance cues alone. This limitation largely stems from the use of generic rewards, such as difficulty and diversity [10], which provide little incentive to capture temporal dependencies. Consequently, the generated questions remain largely temporally insensitive, and the resulting supervision fails to encourage reasoning over temporal order, causality, or state transitions. As a result, both the Questioner and Solver become insensitive to the temporal structure that is fundamental to video understanding. This reveals a key limitation of current self-evolving frameworks: they lack temporal-centric and verifiable learning signals that explicitly exploit the dynamic nature of video.

To address this limitation, we propose EvoVid, a temporal-centric self-evolving framework for video understanding and reasoning through a temporal-aware Questioner and a temporal-grounded Solver. Specifically, we design two complementary rewards targeting the Questioner and Solver respectively. First, we propose a temporal-aware Questioner reward that encourages the generation of temporally dependent questions. Our central insight is that meaningful video questions should rely on the correct temporal ordering of frames, making temporal sensitivity a natural intrinsic supervision signal. We therefore compare Solver responses on the original video and temporally perturbed variants, rewarding questions whose answers significantly degrade under temporal perturbation. This transforms temporal sensitivity into a label-free and verifiable learning signal, guiding the Questioner to generate temporally intensive queries.

Second, we introduce a temporal-grounded Solver reward that provides explicit temporal supervision without requiring human annotations. During question generation, the Questioner only observes a sampled temporal window from the video and generates questions conditioned on this partial context. The sampled window therefore naturally defines an inherent temporal segment associated with the generated question. The Solver is then trained to predict both the answer and the relevant temporal segment from the full video, while being rewarded for accurately localizing the sampled window. In this way, temporal grounding supervision emerges automatically from the self-evolving process itself, establishing grounded reasoning without requiring manually annotated timestamps.

Together, the temporal-aware Questioner and temporal-grounded Solver form a co-evolution process in which a single base model alternates between the two roles: the Questioner generates temporally dependent questions, while the Solver answers the questions and localizes the corresponding temporal segments. Both roles are jointly optimized using Group Relative Policy Optimization (GRPO) [12]. By integrating temporal perturbation sensitivity and temporal grounding supervision into the selfevolving loop, EvoVid enables Video-LLMs to progressively acquire stronger temporal reasoning capabilities directly from raw, unannotated videos.

Our main contributions are three-fold:

• We propose EvoVid, the first self-evolving framework for Video-LLMs which improves video understanding and reasoning directly from raw, unannotated videos without relying on human supervision.

• We introduce two temporal-centric and modality-inherent verifiable rewards: a temporalaware Questioner reward that preserves temporal perturbation sensitivity, and a temporalgrounded Solver reward that leverages inherent temporal segment supervision.   
• We conduct extensive experiments across four MLLMs and six video understanding and reasoning benchmarks, demonstrating consistent improvements and validating the effectiveness of video-based self-evolution.

# 2 Related Work

Reinforcement Learning for Video-LLMs. Recent work has explored reinforcement learning (RL) for Video-LLMs through a range of video-specific optimization strategies. These include encouraging temporal sensitivity by perturbing frame order [1], enhancing spatio-temporal perception via structured or rule-based reward fine-tuning [2], and stabilizing training with difficulty-aware reward formulations [3]. Other approaches focus on temporally grounded tasks, such as leveraging IoU-based rewards for temporal video grounding with annotated segments [4], or incorporating self-verification processes to assess search completeness in long-form video understanding [13]. More recent efforts further investigate token-level credit assignment to better capture reasoningrelevant information during optimization [5]. Despite these advances, existing methods largely rely on human-labeled supervision such as annotated QA pairs, temporal segments, or task-specific heuristics to construct reward signals. This dependence limits scalability and constrains the model’s ability to generalize beyond the scope of human-provided annotations. In contrast, our approach derives verifiable reward signals directly from the temporal dynamics of raw video, enabling label-free optimization and supporting self-evolving reasoning without human supervision.

Self-evolving Models. Self-evolution [14, 15] has emerged as a powerful paradigm for eliciting reasoning behavior in Large Language Models (LLMs) [6, 16, 17] without human labels, by utilizing intrinsic output consistency [7, 9] rather than external supervision. This paradigm has recently been extended to Vision-Language Models (VLMs) [18], typically instantiated as a questioner–solver framework, where both components co-evolve under internal verification signals such as voting-based aggregation [10, 19, 20], game outcomes [21], and continuous or trajectory-level supervision [11, 22]. In parallel, some approaches exploit privileged 3D structure as a programmable verifier to enhance spatial reasoning [23]. However, these methods are primarily designed for text or static images, where verification signals lack inherent sensitivity to temporal structure. Consequently, directly extending them to video often leads to temporally agnostic question generation and limited performance gains. Our work addresses this limitation by introducing temporal-centric verifiable signals that are intrinsically tied to the time dimension of video data.

# 3 Method

# 3.1 Self-evolving Framework

Problem Formulation. In our work, the self-evolving model uses no question-answer annotations, metadata, or external reward models. The paradigm builds upon two roles initialized from a single pretrained base model: a Questioner and a Solver. Given an unlabeled video v, the Questioner $\pi _ { Q }$ generates a question $q \sim \pi _ { Q } ( \cdot \mid v )$ , while the Solver $\pi _ { S }$ produces M candidate answers $\lbrace \hat { a } _ { m } \rbrace _ { m = 1 } ^ { M } ,$ where $\hat { a } _ { m } \sim \pi _ { S } ( \cdot \mid v , q )$ . A pseudo target answer $a ^ { * }$ is obtained via majority voting:

$$
a ^ {*} = \arg \max _ {a} \sum_ {m = 1} ^ {M} \mathbb {1} \left[ \hat {a} _ {m} = a \right]. \tag {1}
$$

The goal is to jointly optimize $\pi _ { Q }$ and $\pi _ { S }$ through RL process such that the questioner produces informative and learnable questions, while the solver improves its ability to answer them.

Preliminaries. In existing self-evolving frameworks [7, 9, 10], both policies are optimized with rewards derived from solver responses. The solver’s confidence on (v, q) is the empirical agreement rate of its M candidates with the majority-voted answer $a ^ { * } { } ;$ :

![](images/fad2bb2c1d74f896ca0737e389dfd0bc50b229e36d13810b7a9d0cf2a4587d99.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Questioner π_Q"] --> B["What does the person do after leaving the room?"]
    B --> C["Original"]
    C --> D["Solver π_S"]
    D --> E["Solver Responses"]
    E --> F["Temporal-aware Questioner Reward r_temp^Q"]
    F --> G["Update Solver π_S"]
    
    H["Questioner π_Q"] --> I["Generate questions based on this segment..."]
    I --> J["Original"]
    J --> K["Solver π_S"]
    K --> L["Predicted segment"]
    L --> M["Temporal-grounded Solver Reward r_temp^S"]
    M --> G
    
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style I fill:#cff,stroke:#333
    style K fill:#cff,stroke:#333
    style L fill:#ffc,stroke:#333
    style M fill:#ffc,stroke:#333
```
</details>

Figure 2: Overview of EvoVid. Questioner $\pi _ { Q }$ and Solver $\pi _ { S }$ co-evolve through two temporal-centric rewards. Questioner Training: with original and shuffled frames to derive t $\pi _ { S }$ frozen, tempora $\pi _ { Q }$ generates questions ware Questioner reward $\pi _ { S }$ responds usingolver Training: $r _ { \mathrm { t e m p } } ^ { Q }$ with $\pi _ { Q }$ frozen, the Questioner generates questions from a sampled $K$ -frame window, and $\pi _ { S }$ predicts both the answer and temporal segment, which is compared with the sampled window to derive the temporal-grounded Solver reward $r _ { \mathrm { t e m p } } ^ { S } .$ . Preliminary rewards are omitted for simplicity.

$$
s (v, q) = \frac {1}{M} \sum_ {m = 1} ^ {M} \mathbb {1} \left[ \hat {a} _ {m} = a ^ {*} \right] \tag {2}
$$

The questioner is rewarded by a difficulty term:

$$
r _ {\text { diff }} ^ {Q} = \min \big (s (v, q), 1 - s (v, q) \big), \tag {3}
$$

which favors questions of moderate difficulty, along with a diversity penalty $r _ { \mathrm { d i v } } ^ { Q }$ that discourages reward is:

$$
r ^ {Q} = \max \left(0, r _ {\text { diff }} ^ {Q} - r _ {\text { div }} ^ {Q}\right). \tag {4}
$$

The solver is trained using a pseudo target derived from Eq. (1) and receives a reward composed of an answer correctness term $\mathrm { a c c } \in \{ 0 , 1 \}$ and a format term fmt $\in \{ 0 , 1 \}$ }:

$$
r ^ {S} = (1 - w) \mathrm{acc} + w \mathrm{fmt}. \tag {5}
$$

Both the questioner and solver are optimized using GRPO [12], which normalizes rewards within each batch and incorporates a KL regularization term for stable updates. More details and training algorithm can be found in the Appendix A and B.

# 3.2 Temporal-aware Questioner

The reward signals in existing self-evolving frameworks, introduced in §3.1, are largely modalityagnostic and do not explicitly capture the temporal structure of video. In particular, the preliminary Questioner rewards $r _ { \mathrm { d i f f } } ^ { Q }$ and $r _ { \mathrm { d i v } } ^ { Q }$ depend only on $( q , s ( v , q ) )$ ), providing no signal that emphasizes temporal dependencies across frames. As a result, the Questioner is confined to a temporally insensitive regime and tends to behave similarly to image-based question generation, producing questions that can be answered from a single frame. It therefore receives little incentive to explore temporal relationships such as order, causality, or state transitions, ultimately limiting the effectiveness and expressiveness of the learned supervision.

To address this, we introduce a temporal-aware Questioner reward that explicitly leverages the intrinsic supervision provided by the ordering of video frames. Given a question $q ,$ we evaluate the Solver on both the original video v and a temporally perturbed version v˜ obtained by applying a permutation σ to the frame sequence, $i . e . , \tilde { v } = \sigma ( v )$ . Let $s ( v , q )$ and $s ( \tilde { v } , q )$ denote the corresponding confidence scores defined in Eq. (2). A temporally meaningful question should depend on the correct frame order and therefore lead to degraded Solver confidence under temporal perturbation. We define the temporal-aware reward as:

$$
r _ {\text { temp }} ^ {Q} = \max \big (0, s (v, q) - s (\tilde {v}, q) \big). \tag {6}
$$

This reward favors questions whose answers become less consistent when temporal order is disrupted, thereby encouraging the Questioner to generate queries that are sensitive to temporal structure. The final Questioner reward is:

$$
r _ {\text { total }} ^ {Q} = r ^ {Q} + \lambda_ {q} r _ {\text { temp }} ^ {Q}, \tag {7}
$$

where $\lambda _ { q }$ controls the strength of temporal supervision. By explicitly introducing temporal perturbations during training, the proposed reward promotes the generation of questions that require reasoning over action sequences and event relations, guiding learning toward temporally and causally informative content.

# 3.3 Temporal-grounded Solver

While the temporal-aware Questioner reward improves the temporal sensitivity of generated questions, the Solver itself remains weakly supervised with respect to temporal structure. In the preliminary rewards, Solver is optimized only for answer correctness, without explicit guidance on when relevant evidence occurs in the video. Since the base reward $r ^ { S }$ is invariant to the temporal location of evidence, a correct answer that ignores temporal locality is rewarded identically to one that accurately localizes supporting evidence. Consequently, Solver receives no incentive to align its predictions with specific temporal segments, limiting its ability to perform temporally grounded reasoning.

To address this, we introduce a temporally grounded supervision signal by exploiting the inherent characteristic of video modality within the self-play pipeline. During question generation used for Solver training, each video is sampled as a contiguous K-frame window, which defines a temporal segment $\boldsymbol { \mathcal { W } } \bar { \mathbf { \Sigma } } = [ t _ { s } , t _ { e } ]$ in the original video. Since Questioner generates queries conditioned on this window, W naturally serves as an automatically constructed pseudo ground-truth segment. In addition to predicting the answer, the Solver is required to output a temporal segment $\hat { \mathcal { W } } = [ \hat { t } _ { s } , \hat { t } _ { e } ]$ in format <segmen $> \hat { t } _ { s } \dag - \hat { t } _ { e } \dag < /$ segment>, and is rewarded based on its alignment with W using Intersection-over-Union (IoU):

$$
r _ {\text { temp }} ^ {S} = \mathrm{IoU} (\hat {\mathcal {W}}, \mathcal {W}). \tag {8}
$$

To ensure meaningful grounding, this reward is applied only when the predicted answer is correct. The final Solver reward is:

$$
r _ {\text {total}} ^ {S} = r ^ {S} + \lambda_ {s} r _ {\text {temp}} ^ {S}, \tag {9}
$$

where $\lambda _ { s }$ balances answer correctness and temporal grounding. By aligning predictions with automatically derived temporal segments, Solver is encouraged to learn temporally localized reasoning without requiring human annotations.

# 4 Experiments

# 4.1 Experimental Setup

Training Data. During self-evolving training, we aggregate a diverse collection of open-domain video data. The raw videos are sourced from the training splits of LLaVA-Video-178K [24], STAR [25], CLEVRER [26], PerceptionTest [27], NeXT-QA [28], and Video-Holmes [29], resulting in a total of 5,778 videos. Importantly, we do not use any annotations from these datasets during training, only the raw videos are utilized.

Evaluation Benchmarks. We evaluate our model on six video benchmarks spanning both reasoningfocused and general-purpose video understanding tasks. The reasoning benchmarks include Video-Holmes [29], VSIBench [30], VideoMMMU [31], and MMVU [32], which assess the model’s reasoning capabilities in video understanding. The general-purpose benchmarks, TempCompass [33] and VideoMME [34], evaluate performance across a mixture of perception and reasoning tasks. For MMVU, we follow previous works [1, 3, 5] to evaluate on its multiple-choice questions. For VideoMME, no subtitles are used during evaluation.

Table 1: Performance on video reasoning and general benchmarks across four MLLMs. For each base model, we compare the untrained Base Model against the frozen-questioner ( Questioner) baseline, where Solver is trained on questions emitted by an untrained Questioner, and three iterations of EvoVid co-evolution. Bold marks the best result for each model. Blue superscripts on the Iter-3 row report the absolute improvement over the corresponding base model. 

<table><tr><td rowspan="2">Methods</td><td colspan="4">Video Reasoning Benchmark</td><td colspan="2">Video General Benchmark</td></tr><tr><td>Video-Holmes</td><td>VSI-Bench</td><td>VideoMMMU</td><td>MMVU</td><td>TempCompass</td><td>VideoMME</td></tr><tr><td colspan="7">Qwen2.5-VL-3B-Instruct</td></tr><tr><td>Base Model (w/o training)</td><td>26.8</td><td>25.1</td><td>30.6</td><td>48.5</td><td>28.9</td><td>48.0</td></tr><tr><td>EvoVid (* Questioner)</td><td>27.8</td><td>26.3</td><td>35.0</td><td>53.3</td><td>34.5</td><td>48.3</td></tr><tr><td>EvoVid (Iter 1)</td><td>28.0</td><td>29.5</td><td>34.4</td><td>51.7</td><td>33.2</td><td>47.0</td></tr><tr><td>EvoVid (Iter 2)</td><td>26.9</td><td>28.4</td><td>36.9</td><td>53.1</td><td>46.3</td><td>48.9</td></tr><tr><td>EvoVid (Iter 3)</td><td> $27.2 ^{↑0.4}$ </td><td> $29.5 ^{↑4.4}$ </td><td> $36.7 ^{↑6.1}$ </td><td> $54.6 ^{↑6.1}$ </td><td> $48.2 ^{↑19.3}$ </td><td> $48.6 ^{↑0.6}$ </td></tr><tr><td colspan="7">Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Base Model (w/o training)</td><td>27.8</td><td>27.7</td><td>47.8</td><td>59.2</td><td>72.2</td><td>53.1</td></tr><tr><td>EvoVid (* Questioner)</td><td>28.4</td><td>29.3</td><td>46.6</td><td>60.5</td><td>72.5</td><td>51.5</td></tr><tr><td>EvoVid (Iter 1)</td><td>28.6</td><td>28.5</td><td>48.9</td><td>60.6</td><td>72.1</td><td>51.4</td></tr><tr><td>EvoVid (Iter 2)</td><td>29.9</td><td>30.9</td><td>48.3</td><td>60.5</td><td>73.1</td><td>53.8</td></tr><tr><td>EvoVid (Iter 3)</td><td> $29.3 ^{↑1.5}$ </td><td> $31.7 ^{↑4.0}$ </td><td> $50.0 ^{↑2.2}$ </td><td> $62.4 ^{↑3.2}$ </td><td> $73.2 ^{↑1.0}$ </td><td> $53.6 ^{↑0.5}$ </td></tr><tr><td colspan="7">Qwen3-VL-4B-Instruct</td></tr><tr><td>Base Model (w/o training)</td><td>29.9</td><td>40.1</td><td>46.7</td><td>60.3</td><td>70.9</td><td>49.6</td></tr><tr><td>EvoVid (* Questioner)</td><td>30.7</td><td>40.6</td><td>45.1</td><td>61.1</td><td>70.6</td><td>49.6</td></tr><tr><td>EvoVid (Iter 1)</td><td>30.4</td><td>41.4</td><td>47.3</td><td>61.0</td><td>70.8</td><td>49.7</td></tr><tr><td>EvoVid (Iter 2)</td><td>29.1</td><td>43.1</td><td>47.3</td><td>62.4</td><td>70.9</td><td>49.7</td></tr><tr><td>EvoVid (Iter 3)</td><td> $31.1 ^{↑1.2}$ </td><td> $42.8 ^{↑2.7}$ </td><td> $47.6 ^{↑0.9}$ </td><td> $62.6 ^{↑2.3}$ </td><td> $71.7 ^{↑0.8}$ </td><td> $51.4 ^{↑1.8}$ </td></tr><tr><td colspan="7">Qwen3-VL-8B-Instruct</td></tr><tr><td>Base Model (w/o training)</td><td>36.2</td><td>37.4</td><td>46.8</td><td>64.8</td><td>73.6</td><td>50.3</td></tr><tr><td>EvoVid (* Questioner)</td><td>36.3</td><td>35.3</td><td>47.8</td><td>63.7</td><td>73.4</td><td>50.4</td></tr><tr><td>EvoVid (Iter 1)</td><td>36.0</td><td>38.0</td><td>49.0</td><td>63.8</td><td>74.3</td><td>50.3</td></tr><tr><td>EvoVid (Iter 2)</td><td>35.2</td><td>38.3</td><td>48.1</td><td>67.0</td><td>73.7</td><td>51.4</td></tr><tr><td>EvoVid (Iter 3)</td><td> $36.6 ^{↑0.4}$ </td><td> $39.8 ^{↑2.4}$ </td><td> $49.1 ^{↑2.3}$ </td><td> $66.2 ^{↑1.4}$ </td><td> $74.3 ^{↑0.7}$ </td><td> $51.2 ^{↑0.9}$ </td></tr></table>

Implementation Details. We apply our method to four base models: Qwen2.5-VL-3B-Instruct [35], Qwen2.5-VL-7B-Instruct [35], Qwen3-VL-4B-Instruct [36], and Qwen3-VL-8B-Instruct [36]. Both the Questioner $\pi _ { Q }$ and Solver $\pi _ { S }$ are initialized from the same base model and trained separately using LoRA fine-tuning. The input is limited to at most 16 frames with a resolution of $1 2 8 \times \mathsf { \bar { 1 4 } } \times \mathsf { 1 \bar { 4 } }$ . During GRPO [12] optimization, we use a global batch size of 128 with group size $G { = } 8$ and a KL coefficient of $1 0 ^ { - 2 }$ for both Questioner and Solver training. For Questioner training, each candidate question is evaluated by sampling $M = 1 0$ answers from the current Solver to compute the reward. The co-evolution process runs for three rounds, alternating between Questioner and Solver, with each phase trained for 20 steps. We set the reward weights to $\lambda _ { q } = 0 . 1$ for the temporal-aware Questioner reward and $\lambda _ { s } = 0 . 3$ for the temporal-grounded Solver reward, with a grounding window of $K = 8$ frames. The Questioner and Solver are optimized using AdamW with learning rates of $1 0 ^ { - 6 }$ and $2 \times 1 0 ^ { - 6 }$ for 20 steps across three co-evolution rounds. Experiments are conducted with 8× H100 GPUs. At evaluation time, we adopt the official Qwen2.5-VL decoding settings (top\_ $_ { \mathrm { - } } \mathrm { p } = 0 . 0 0 1$ , temperature = 0.01). More details can be found in the Appendix A.

# 4.2 Main Results

Table 1 presents the results on six video reasoning and general benchmarks using four MLLMs, including Qwen2.5-VL-Instruct 3B/7B and Qwen3-VL-Instruct 4B/8B. Consistent performance improvements are observed across different base models throughout the training iterations of our proposed self-evolving framework EvoVid. The gains across all benchmarks validate the effectiveness of our approach despite the absence of human-curated questions or annotations. Through a fully autonomous learning process, the model progressively improves its video understanding and reasoning capabilities by independently generating and solving training questions. For the Frozen Questioner setting, where the solver is trained for three iterations (60 steps) using questions generated by an untrained base model, the achievable improvements are naturally constrained by the quality of the initial policy. Among all models, Qwen2.5-VL-3B exhibits the most significant gains on temporally challenging benchmarks, particularly TempCompass (+19.3%), VideoMMMU (+6.1%), and MMVU (+6.1%), highlighting the effectiveness of our framework even when starting from a relatively weaker initialization. Across all model scales, larger improvements are consistently observed on video reasoning benchmarks that require temporal understanding, aligning with our design motivation of explicitly optimizing temporally grounded reward signals.

Table 2: Comparison with supervised baselines and VANILLA self-evolving baseline. We evaluate EvoVid on Qwen2.5-VL-7B against three reference approaches: supervised fine-tuning (SFT), supervised RL with GRPO (VANILLA GRPO), and the label-free VANILLA self-evolving framework without temporal-inherent rewards. † denotes methods trained on human-curated datasets with both questions and annotations. In contrast, both VANILLA self-evolving and EvoVid are trained directly from raw videos without human supervision. We also initialize EvoVid from the SFT checkpoint, with results reported in the last row. Bold and underline indicate the best and second-best results. 

<table><tr><td rowspan="2">Methods</td><td colspan="4">Video Reasoning Benchmark</td><td colspan="2">Video General Benchmark</td><td rowspan="2">Avg.</td></tr><tr><td>Video-Holmes</td><td>VSI-Bench</td><td>VideoMMMU</td><td>MMVU</td><td>TempCompass</td><td>VideoMME</td></tr><tr><td>Base Model (w/o training)</td><td>27.8</td><td>27.7</td><td>47.8</td><td>59.2</td><td>72.2</td><td>53.1</td><td>47.9</td></tr><tr><td> $SFT^†$ </td><td>31.1</td><td>31.8</td><td>47.4</td><td>61.3</td><td>69.2</td><td>52.8</td><td>49.0</td></tr><tr><td> $SFT^† + VANILLA GRPO^†$ </td><td>38.8</td><td>32.7</td><td>48.3</td><td>62.1</td><td>71.3</td><td>54.5</td><td>51.3</td></tr><tr><td> $VANILLA Self-evolving$ </td><td>28.9</td><td>28.9</td><td>49.4</td><td>59.2</td><td>72.9</td><td>52.5</td><td>48.6</td></tr><tr><td>EvoVid (Iter 3)</td><td>29.3</td><td>31.7</td><td>50.0</td><td>62.4</td><td>73.2</td><td>53.6</td><td>50.0</td></tr><tr><td> $SFT^† + EvoVid (Iter 3)$ </td><td>33.3</td><td>35.0</td><td>48.6</td><td>63.5</td><td>72.6</td><td>54.0</td><td>51.2</td></tr></table>

Table 3: Reward ablation on proposed temporal Questioner and Solver rewards. We coablation based on Qwen2.5-VL-7B at Iter 3 on the proposed temporal-aware Questioner reward and temporal-grounded Solver reward $r _ { \mathrm { t e m p } } ^ { S } .$ temp. The first row represents VANILLA Self-evolving with $r _ { \mathrm { t e m p } } ^ { Q }$ only preliminary rewards present in §3.1. 

<table><tr><td colspan="2">Reward</td><td colspan="4">Video Reasoning Benchmark</td><td colspan="2">Video General Benchmark</td><td rowspan="2">Avg.</td></tr><tr><td> $r_{temp}^{Q}$ </td><td> $r_{temp}^{S}$ </td><td>Video-Holmes</td><td>VSI-Bench</td><td>VideoMMMU</td><td>MMVU</td><td>TempCompass</td><td>VideoMME</td></tr><tr><td>×</td><td>×</td><td>28.9</td><td>28.9</td><td>49.4</td><td>59.2</td><td>72.9</td><td>52.5</td><td>48.6</td></tr><tr><td>√</td><td>×</td><td>29.7</td><td>29.6</td><td>49.1</td><td>62.1</td><td>72.8</td><td>52.7</td><td>49.3</td></tr><tr><td>×</td><td>√</td><td>30.1</td><td>30.7</td><td>49.3</td><td>61.0</td><td>71.8</td><td>52.5</td><td>49.2</td></tr><tr><td>√</td><td>√</td><td>29.3</td><td>31.7</td><td>50.0</td><td>62.4</td><td>73.2</td><td>53.6</td><td>50.0</td></tr></table>

# 4.3 Comparison with Supervised Baselines and VANILLA Self-evolving Baseline

Table 2 compares EvoVid based on Qwen2.5-VL-7B against supervised fine-tuning (SFT), supervised RL with GRPO (VANILLA GRPO), and the VANILLA self-evolving baseline. As shown in the results, EvoVid surpasses SFT by an additional average gain of +1.0%, where SFT is trained for one epoch on the Video-R1-CoT-165k dataset [1]. Applying an additional 1k steps of VANILLA GRPO further improves performance over SFT by 2.3%, while our method achieves a 2.1% improvement over the base model without relying on any human annotations. Compared with the VANILLA self-evolving baseline, our approach increases the average performance from 48.6% to 50.0%, achieving the best results across all evaluated datasets. Notably, among all baselines, including supervised methods, EvoVid achieves the best performance on VideoMMMU, MMVU, and TempCompass, demonstrating the effectiveness of the self-evolving framework for video understanding through the integration of temporal-centric rewards for both the Questioner and Solver. We further initialize EvoVid with Video-R1 SFT checkpoint [1]. The resulting model achieves performance competitive with the supervised SFT + VANILLA GRPO baseline while requiring significantly fewer RL training steps. This highlights the potential of our framework to further enhance pretrained Video-LLMs through self-evolving temporal supervision without introducing additional human annotations.

# 4.4 Ablation Studies

Reward Ablation. Table 3 isolates the contributions of the two temporal-inherent rewards by incrementally adding each component to the VANILLA Self-evolving baseline. Using only the temporal-aware Questioner rewardmarks by +0.7%, while using only t $r _ { \mathrm { t e m p } } ^ { Q }$ improves the average perforporal-grounded Solver reward across the six bench-increases the average $r _ { \mathrm { t e m p } } ^ { S }$ to 49.2%, corresponding to a +0.6% gain. Combining both rewards produces the full model, achieving 50.0% average performance, which surpasses VANILLA self-evolving by +1.4%. The complementary improvements indicate that the gains from the two rewards are additive rather than redundant, consistent with their distinct optimization roles on different agents and temporal aspects. Specifically, the temporal-aware Questioner reward guides the questioner through frame-order sensitivity, whereas the temporal-grounded Solver reward enhances the solver via temporally consistent segment localization.

![](images/77cb19cb67b9d7331c8ecfec4c7c862411bd99db68cb275868bfe8e0655ea584.jpg)

<details>
<summary>bar</summary>

Shuffle Strategy in Questioner Reward
| Shuffle Type | Average Score on Video Benchmarks |
| :--- | :--- |
| No Shuffle | 48.6 |
| Random Shuffle | 49.3 |
| Reverse | 48.8 |
| Block Shuffle | 49.1 |
</details>

Figure 3: Effect of different shuffle strategies in Questioner reward on performance. We use random shuffle by default.

![](images/053fc880acc220aa72ed0f9e3ed72a0d23b3ab779d13d8400ae596d359dfba78.jpg)

<details>
<summary>bar</summary>

| K Value | Average Score on Video Benchmarks |
| ------- | ---------------------------------- |
| K=4     | 49.1                               |
| K=8     | 49.2                               |
| K=12    | 48.9                               |
| K=16    | 48.2                               |
</details>

Figure 4: Effect of different window sizes in Solver reward on performance. We use $K = 8$ by default.

Shuffle Strategy in Questioner Reward. Figure 3 illustrates the impact of different frame-shuffling strategies on model performance. We compare a no-shuffle baseline with three temporal perturbation strategies: random shuffle (default), reverse-order frames, and block-wise random shuffle (block = 4). All shuffling strategies outperform the no-shuffle baseline (48.6%), indicating that temporally degraded contrastive signals are effective for encouraging temporal sensitivity during training. Among the variants, simply reversing the frame order yields only a modest improvement of +0.2%, as it perturbs only the temporal direction while preserving much of the original temporal structure. Both block shuffle and fully random shuffle achieve stronger and comparable gains, although block shuffle performs slightly worse because it retains short-range temporal coherence while mainly disrupting long-range dependencies. In contrast, random shuffle removes temporal structure more thoroughly, providing a stronger temporal contrastive signal. We use random shuffle in all experiments by default.

Window Size in Solver Reward. Figure 4 compares the effect of different window sizes K used for question generation and temporal-grounded solver supervision. We sweep $K \in \{ 4 , 8 , 1 2 , 1 6 \}$ on 16-frame videos. For moderate window sizes $( K = \mathsf { \bar { f } } 4 , 8 , 1 2 \} )$ ), performance remains stable, achieving 48.9–49.2 on average with only 0.3-point variation, indicating that the method is not highly sensitive to the exact window size within an informative regime. Performance drops at $K = 1 { \dot { 6 } } .$ where the sampled segment spans the full video and the IoU-based grounding bonus becomes degenerate (48.2%). This result confirms that the gain mainly comes from informative temporal segment supervision rather than merely introducing a <segment> output. Overall, the results suggest that any reasonable window size below the full video length can provide effective grounding supervision. We adopt $K = 8$ as the default setting in all experiments.

# 4.5 Qualitative Analysis of Generated Questions

Figure 5 visualizes the word distributions of generated questions between the VANILLA Self-evolving baseline and our temporal-aware Questioner, revealing a sharp lexical contrast. Compared with the baseline, which is dominated by generic and appearance-oriented terms such as person, object, and shape, our model produces substantially richer temporal and reasoning-related vocabulary. In particular, words such as primary, frame, consider, change, reason and interaction appear much more frequently, indicating that the generated questions increasingly focus on temporal transitions, causal relations, and dynamic scene understanding. The increased diversity of temporal and relational concepts suggests that the proposed temporal-aware Questioner reward successfully encourages the Questioner to generate more temporally dependent and reasoning-intensive queries, rather than relying on general and static visual attributes alone.

![](images/0abf38ea21f8b21828d9d95c5fc057f4676e88149ca5c592a5499f44f38b05c2.jpg)

<details>
<summary>text_image</summary>

Baseline
event
hand
sport
broom
people
order
player
hair
hair
color
beef
person
property
first
right
change
position
table
interact
body
man
volume
cylinder
perform
preparing
move
frame
correct
change
left
child
right
form
body
person
activity
</details>

![](images/6c25a2f779165f7caf2c8ec98822f810eed6fe84e0cabf473d11a49b8fae75ed.jpg)

<details>
<summary>text_image</summary>

Ours
purpose
number
reason
primary
form
suggest
final
role
player
time
yellow
ingredient
other
case
inferred
assuming
cylinder
setup
pattern
change
feature
activity
position
behavior
red
blue
total
immediately
state
approimate
preparation
indicate woman
table
event
space
order
start
aside_text
child
aspect
person
type
blue
green
foot
consider
food stage
setting
</details>

Figure 5: Qualitative analysis of generated questions. The word cloud visualization shows that EvoVid generates questions with substantially richer temporal and reasoning-oriented vocabulary compared to the VANILLA Self-evolving baseline.

# 4.6 Iteration Scaling

Figure 6 extends the co-evolution process beyond the default three iterations. Three of the four backbones (Qwen2.5-VL-7B, Qwen3-VL-4B, and Qwen3-VL-8B) exhibit a generally stable increasing trend, where performance improves progressively with additional iterations despite minor fluctuations at certain stages. Overall, the curves show a clear upward trajectory, indicating that temporal-centric self-evolution continues to provide meaningful gains during iterative training. In contrast, Qwen2.5-VL-3B behaves less consistently, reaching its peak performance at iteration 3 before showing slight degradation in later iterations. We attribute this instability to the limited capacity of Qwen2.5-VL-3B model, which may hinder its ability to sustain long-term co-evolution optimization. Nevertheless, the results suggest that the default three-iteration setting used in our experiments demonstrates balanced performance-cost tradeoff while maintaining stable training across different model scales.

![](images/d3109475bb342c61e292820cd0861c8f5829e464e1b6ac12714dbe916a2357a7.jpg)

<details>
<summary>line</summary>

| Model          | Average Score on Video Benchmarks |
| -------------- | ----------------------------------- |
| Base Model     | 34.70                               |
| Iter 1         | 37.30                               |
| Iter 2         | 40.10                               |
| Iter 3         | 40.60                               |
| Iter 4         | 39.20                               |
| Iter 5         | 39.20                               |
| Iter 6         | 38.80                               |
</details>

Figure 6: Iteration scaling. Performance under extended training iterations and model scales.

# 4.7 Question Evolution

Table 4 investigates the evolution of Questions generated by Questioner across iterations. We sample 300 questions generated at each iteration, denoted as $\mathcal { D } _ { \mathrm { I t e r } }$ , and evaluate Solver accuracy using GPT-5.4-derived responses as oracle answers. Reading across rows, the accuracy decreases substantially (e.g., from 33.0% to 13.5% for the base model), indicating that Questioner progressively constructs a more challenging curriculum over iterations. Reading down columns, each successive Solver consistently outperforms its predecessor, demonstrating that Questioner and Solver co-evolve jointly, with Solver capability improving alongside the increasing difficulty of generated questions.

Table 4: Question evolution. Solver performance evaluated on questions generated by Questioner across different iterations. 

<table><tr><td></td><td> $\mathcal{D}_{\text{Iter 1}}$ </td><td> $\mathcal{D}_{\text{Iter 2}}$ </td><td> $\mathcal{D}_{\text{Iter 3}}$ </td></tr><tr><td>Base Model</td><td>33.0</td><td>19.0</td><td>13.5</td></tr><tr><td>Solver@Iter 1</td><td>35.5</td><td>20.5</td><td>17.0</td></tr><tr><td>Solver@Iter 2</td><td>36.5</td><td>23.5</td><td>18.5</td></tr><tr><td>Solver@Iter 3</td><td>38.0</td><td>24.5</td><td>22.0</td></tr></table>

# 5 Conclusion

In this work, we present EvoVid, a temporal-centric self-evolving framework for video understanding and reasoning. Driven by the intrinsic temporal dynamics of video, we design a temporal-aware Questioner and a temporal-grounded Solver that co-evolve through iterative training, yielding consistent improvements across diverse benchmarks and model scales. Our work points to a new paradigm for Video-LLMs, enabling scalable and fully autonomous learning.

# References

[1] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025. 1, 3, 5, 7, 15   
[2] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025. 3   
[3] Jinyoung Park, Jeehye Na, Jinyoung Kim, and Hyunwoo J Kim. Deepvideo-r1: Video reinforcement fine-tuning via difficulty-aware regressive grpo. arXiv preprint arXiv:2506.07464, 2025. 3, 5, 15   
[4] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025. 3   
[5] Ziyue Wang, Sheng Jin, Zhongrong Zuo, Jiawei Wu, Han Qiu, Qi She, Hao Zhang, and Xudong Jiang. Video-ktr: Reinforcing video reasoning via key token attribution. arXiv preprint arXiv:2601.19686, 2026. 1, 3, 5, 15   
[6] Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025. 1, 3   
[7] Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-zero: Self-evolving reasoning llm from zero data. arXiv preprint arXiv:2508.05004, 2025. 1, 3   
[8] Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and Yueting Zhuang. Self-supervised spatiotemporal learning via video clip order prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10334–10343, 2019. 1   
[9] Ziyi Yang, Weizhou Shen, Chenliang Li, Ruijun Chen, Fanqi Wan, Ming Yan, Xiaojun Quan, and Fei Huang. Spell: Self-play reinforcement learning for evolving long-context language models. arXiv preprint arXiv:2509.23863, 2025. 1, 3   
[10] Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, and Yonghui Yang. Visplay: Self-evolving vision-language models from images. arXiv preprint arXiv:2511.15661, 2025. 1, 2, 3   
[11] Omkar Thawakar, Shravan Venkatraman, Ritesh Thawkar, Abdelrahman Shaker, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, and Fahad Khan. Evolmm: Self-evolving large multimodal models with continuous rewards. arXiv preprint arXiv:2511.16672, 2025. 1, 3   
[12] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 4, 6   
[13] Junwen Pan, Qizhe Zhang, Rui Zhang, Ming Lu, Xin Wan, Yuan Zhang, Chang Liu, and Qi She. Timesearchr: Adaptive temporal search for long-form video understanding via self-verification reinforcement learning. arXiv preprint arXiv:2511.05489, 2025. 3   
[14] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models, 2024. URL https://arxiv. org/abs/2401.01335, 2401. 3   
[15] Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, Yongbin Li, Zhi Jin, Fei Huang, Dacheng Tao, and Jingren Zhou. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387, 2024. 3   
[16] Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, Vijai Mohan, and Jason Chen. Language self-play for data-free training. arXiv preprint arXiv:2509.07414, 2025. 3   
[17] Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, and Jason Weston. Spice: Self-play in corpus environments improves reasoning. arXiv preprint arXiv:2510.24684, 2025. 3   
[18] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding vision-language model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025. 3

[19] Han Wang, Yi Yang, Jingyuan Hu, Minfeng Zhu, and Wei Chen. V-zero: Self-improving multimodal reasoning with zero annotation. arXiv preprint arXiv:2601.10094, 2026. 3   
[20] Wen Wen, Tianwu Zhi, Kanglong Fan, Yang Li, Xinge Peng, Yabin Zhang, Yiting Liao, Junlin Li, and Li Zhang. Self-evolving vision-language models for image quality assessment via voting and ranking. arXiv preprint arXiv:2509.25787, 2025. 3   
[21] Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi, Yueqian Lin, Yiran Chen, Hai Helen Li, Kun Wan, and Wentian Zhao. Vision-zero: Scalable vlm self-improvement via strategic gamified self-play. arXiv preprint arXiv:2509.25541, 2025. 3   
[22] Meghana Sunil, Manikandarajan Venmathimaran, and Muthu Subash Kavitha. ireasoner: Trajectory-aware intrinsic reasoning supervision for self-evolving large multimodal models. arXiv preprint arXiv:2601.05877, 2026. 3   
[23] Zongxia Li, Hongyang Du, Chengsong Huang, Xiyang Wu, Lantao Yu, Yicheng He, Jing Xie, Xiaomin Wu, Zhichao Liu, Jiarui Zhang, et al. Mm-zero: Self-evolving multi-model vision language models from zero data. arXiv preprint arXiv:2603.09206, 2026. 3   
[24] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 5, 14   
[25] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. arXiv preprint arXiv:2405.09711, 2024. 5, 14   
[26] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019. 5, 14   
[27] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748– 42761, 2023. 5, 14   
[28] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 5, 14   
[29] Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. Video-holmes: Can mllm think like holmes for complex video reasoning? arXiv preprint arXiv:2505.21374, 2025. 5, 14   
[30] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 5, 14   
[31] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025. 5, 14   
[32] Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu, Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen, Chuhan Li, Zhijian Xu, et al. Mmvu: Measuring expert-level multi-discipline video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8475–8489, 2025. 5, 15   
[33] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772, 2024. 5, 15   
[34] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118, 2025. 5, 16   
[35] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6   
[36] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 6

# Appendix

# A Implementation Details

# A.1 Reward Implementation

The full Questioner and Solver rewards are defined in §3. Here we give the implementation-level details that were left implicit in the main paper.

Format reward. For both the Questioner and Solver, we apply a format gate before assigning any reward. Questioner outputs that do not contain a parseable <type>/<question>/<answer> triple receive zero reward. Similarly, Solver outputs without a \boxed answer receive zero accuracy reward, and outputs lacking a parseable <segment $> t _ { s } - t _ { e } < /$ segment> yield zero IoU bonus, preventing malformed predictions from corrupting the gradient. Concretely, the indicator 1Qfmt $\mathbb { 1 } _ { \mathrm { f m t } } ^ { Q } \in$ {0, 1} in Algorithm 1 denotes the Questioner format gate, while $r _ { \mathrm { f m t } } \in \{ 0 , 1 \}$ } in the Solver branch serves as the Solver format gate. The format weight w in Eq. (13) further balances correctness against format compliance.

Confidence and the difficulty term. The Solver confidence $s ( v , q )$ is computed exactly as in Eq. (2) of the main paper, with $M { = } 1 0$ Solver samples per question. We re-state the difficulty term for clarity:

$$
r _ {\text { diff }} ^ {Q} = \min \big (s (v, q), 1 - s (v, q) \big). \tag {10}
$$

Diversity penalty. For each video v, the Questioner produces a group of G rollouts $\{ q _ { g } \} _ { g = 1 } ^ { G }$ (with G=8 following the GRPO group size). We compute pairwise BLEU similarities among the group and run agglomerative clustering with average linkage at a fixed BLEU threshold of $\tau _ { \mathrm { B L E U } } { = } 0 . 5$ to obtain clusters $\{ \mathcal { C } _ { k } \}$ . The diversity penalty for a question q in cluster $\mathcal { C } _ { k }$ is

$$
r _ {\mathrm{div}} ^ {Q} = \lambda_ {d} \cdot \frac {| \mathcal {C} _ {k} |}{G}, \quad \lambda_ {d} = 1, \tag {11}
$$

so that questions inside large clusters (i.e. the Questioner is repeating itself across the group) are penalized more heavily. The combined preliminary Questioner reward used in the main paper (§3.1) is then $r ^ { Q } = \operatorname* { m a x } \left( 0 , \dot { r } _ { \mathrm { d i f f } } ^ { Q } - r _ { \mathrm { d i v } } ^ { Q } \right)$ .

Final Questioner Questioner reward $r _ { \mathrm { t e m p } } ^ { Q }$ olver rewards. Combining the prelimina and the temporal-grounded Solver reward $r _ { \mathrm { t e m p } } ^ { S }$ s with the temporal-awarefrom §3, the rewards used

$$
r _ {\text { total }} ^ {Q} = \mathbb {1} _ {\mathrm{fmt}} ^ {Q} \cdot \left(r ^ {Q} + \lambda_ {q} r _ {\text { temp }} ^ {Q}\right), \tag {12}
$$

$$
r _ {\text { total }} ^ {S} = (1 - w) \mathrm{acc} + w \mathrm{fmt} + \lambda_ {s} r _ {\text { temp }} ^ {S}, \tag {13}
$$

where a gating ·acc is applied to $r _ { \mathrm { t e m p } } ^ { S }$ to ensure that grounding is rewarded only when the predicted answer matches the pseudo-target. We use $\lambda _ { q } { = } 0 . 1$ and $\lambda _ { s } \mathrm { = } 0 . 3$ throughout (§4); w=0.1 keeps format a small fraction of the Solver reward.

# A.2 Prompt Templates

Questioner prompt. The Questioner is given a prompt that constrains the output schema and the admissible question types. We require a single <type>/<question>/<answer> triple per rollout, where the type is exactly one of {multiple choice, numerical, regression}. The strict format makes the rollouts trivially parseable for the reward function and prevents the model from emitting auxiliary commentary that would dilute the supervision signal. We include one in-context example so the format constraint is grounded for the policy. The full prompt is reproduced verbatim below.

# Questioner Prompt

You are an intelligent Question Generator. Your task is to create a question based on the given video. Requirements (must follow exactly):

1. Watch the video carefully and understand all details across the frames.   
2. Generate exactly one question that is directly related to the video content.   
3. Choose the question type from only one of: multiple choice (Yes/No or four options A/B/C/D, one correct), numerical (a specific numeric answer), or regression (a continuous value such as a measurement, quantity, or coordinate).   
4. The question must require analysis or reasoning, not just description.   
5. Provide the correct answer. Include units if applicable.   
6. Output strictly in the three-block format below, with nothing else.

```txt
Output format:
<type>X</type>
<question>Y</question>
<answer>Z</answer>
where X ∈ {multiple choice, numerical, regression}. 
```

Solver prompt. The Solver receives a thin chain-of-thought wrapper around the Questionergenerated query, with an additional sentence that instructs the Solver to emit the relevant time segment in a parseable form for the temporal-grounded Solver reward.

# Solver Prompt

{question} Please reason step by step based on the question and video, and put your final answer within \boxed{}. Additionally, predict the video time segment (in seconds) that is most relevant to answering this question and output it as <segment>Xs–Ys</segment> (e.g. <segment>2.5s–5.0s</segment>).

The IoU bonus $\lambda _ { s } \cdot u$ · acc is computed only when a parseable <segment> tag is present and the answer is correct, and is masked out otherwise (Eq. (13)).

# A.3 Hyperparameters

Table 5 lists all hyperparameters used in the main experiments. We do not tune any hyperparameter on the evaluation benchmarks.

# B Algorithm and Pipeline

Algorithm 1 summarizes the complete co-evolution loop of EvoVid, which consists of three key phases: Phase 1: Questioner Training, Phase 2: Solver Dataset Construction, and Phase 3: Solver Training. In Phase 1, the Questioner is optimized to generate temporally dependent queries under the proposed reward signals. In Phase 2, the updated Questioner generates one question per video conditioned on a uniformly sampled K-frame contiguous window $[ t _ { s } , t _ { e } ]$ across the full training set. For each question, the frozen Solver produces M=10 candidate answers, from which we derive a majority-vote pseudo-answer $a ^ { * }$ and a confidence score $s ( v , q )$ . We retain only samples with confidence in $s \in [ 0 . 3 , 0 . 8 ]$ to form the Solver training set $\mathcal { D } _ { t } .$ , where the lower bound filters out questions that are too difficult for the current Solver, and the upper bound removes those that are already trivial, maintaining a balanced training curriculum. In Phase 3, the Solver is trained to answer the generated questions and predict the corresponding temporal segments. The window timestamps $[ t _ { s } , t _ { e } ]$ travel with the example into Solver training and serve as the pseudo-ground-truth segment W in the temporal-grounded reward. These three phases are executed iteratively, enabling the Questioner and Solver to co-evolve and progressively improve the model’s temporal reasoning capability without relying on human supervision.

Table 5: Training hyperparameters used in the main experiments. 

<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Base models</td><td>4 backbones</td><td>Qwen2.5-VL-3B/7B-Instruct, Qwen3-VL-4B/8B-Instruct</td></tr><tr><td>Parameter-efficient FT</td><td>LoRA</td><td>rank 32, α=64, dropout 0.05, all linear projections</td></tr><tr><td>Training GPUs</td><td>4×H100, FSDP</td><td>one worker per GPU</td></tr><tr><td>Rollout GPUs</td><td>4×H100, vLLM</td><td>tensor parallel 1, one server per GPU</td></tr><tr><td>GRPO group size G</td><td>Questioner / Solver</td><td>8 / 8</td></tr><tr><td>Solver samples per question</td><td>M</td><td>10</td></tr><tr><td>Co-evolution iterations</td><td>T</td><td>3 (default; ablated up to 8)</td></tr><tr><td>Steps per phase</td><td>Questioner / Solver</td><td>20 / 20</td></tr><tr><td>Frames per video</td><td> $T_f$ </td><td>16</td></tr><tr><td>Visual tokens per frame</td><td></td><td>128 (14×14 patches)</td></tr><tr><td>Questioner window length</td><td>K</td><td>8 contiguous frames (default)</td></tr><tr><td>Temporal-aware Questioner weight</td><td> $λ_q$ </td><td>0.1</td></tr><tr><td>Temporal-grounded Solver weight</td><td> $λ_s$ </td><td>0.3</td></tr><tr><td>Diversity penalty weight</td><td> $λ_d$ </td><td>1.0</td></tr><tr><td>Solver format weight</td><td>w</td><td>0.1</td></tr><tr><td>BLEU clustering threshold</td><td> $τ_{BLEU}$ </td><td>0.5</td></tr><tr><td>Score-band filter</td><td> $[s_{min}, s_{max}]$ </td><td>[0.3, 0.8]</td></tr><tr><td>Optimizer</td><td>AdamW</td><td> $β_1=0.9, β_2=0.95$ </td></tr><tr><td>Learning rate</td><td>Questioner / Solver</td><td> $1×10^{-6}/2×10^{-6}$ </td></tr><tr><td>Global batch size</td><td></td><td>128</td></tr><tr><td>KL coefficient</td><td>β (GRPO)</td><td> $10^{-2}$ </td></tr><tr><td>Decoding (eval)</td><td>official Qwen2.5-VL</td><td>top_p=0.001, temperature=0.01</td></tr></table>

# C Dataset Details

# C.1 Training Videos

Following §4, we aggregate raw video files from the training splits of LLaVA-Video-178K [24] (2,636), STAR [25] (991), CLEVRER [26] (920), PerceptionTest [27] (640), NeXT-QA [28] (358), and Video-Holmes [29] (233), totaling 5,778 unique videos. We use only the raw video files; all human-written QA pairs, captions, and temporal-segment annotations bundled with these datasets are discarded before training and never enter the pipeline. The mixture spans synthetic physics (CLEVRER), instructional (LLaVA-Video-178K, NeXT-QA), egocentric (PerceptionTest), and narrative (Video-Holmes, STAR) clips, ensuring that the temporal-inherent rewards see a broad range of visual dynamics rather than a single domain.

# C.2 Evaluation Benchmarks

We evaluate on six public video benchmarks spanning temporal reasoning, spatial–temporal understanding, expert-domain reasoning, and general-purpose video QA. We follow each benchmark’s official protocol unless noted otherwise.

Video-Holmes [29]. A reasoning-centric benchmark comprising 1,837 questions over 270 narrative videos, where each correct answer requires multi-step inference across non-adjacent clips (e.g., motive attribution, causal event linking, and hidden information recovery). It mainly evaluates models’ temporal and causal reasoning capabilities.

VSI-Bench [30]. A spatial-reasoning benchmark for MLLMs built on 288 real-world room-tour videos and roughly 5,000 question–answer pairs. Its eight tasks (object count, absolute and relative distances, relative direction, route planning, object size, room size, and appearance order) span three reasoning families: configurational, measurement estimation, and spatiotemporal. Each task requires aggregating evidence across many frames rather than from any single salient view.

VideoMMMU [31]. An expert-domain benchmark with 900 questions over 300 college-level instructional videos covering six disciplines (Art, Business, Science, Medicine, Humanities, Engineering). The questions are organized along three cognitive stages: perception, comprehension, and adaptation, which mirror a human-like progression from observation to applied reasoning. Strong performance therefore demands both the domain knowledge to interpret the demonstration and the temporal grounding that aligns each step of the demonstration to the question being asked.

Algorithm 1 EvoVid: Temporal-Centric Self-Evolution for Video-LLMs.   
INPUT: Initial models $\pi_{Q}, \pi_{S}$ (initialized from a single base policy); unlabeled video set V; group size G; Solver samples M; window length K; score band $[s_{\min}, s_{\max}] = [0.3, 0.8]$ ; reward weights $\lambda_{q}, \lambda_{s}, w$ .

OUTPUT: Evolved policies $\pi_{Q}$ and $\pi_{S}$ .

1: for each self-play iteration do

2: // -- Phase 1: Questioner Training --
3: for each Questioner training step do

4: Sample a video $v \in V$ and a question group $\{q_{i}\}_{i=1}^{G} \sim \pi_{Q}(\cdot \mid v)$ ;

5: for each question $q_{i}$ do

6: if Format_Check( $q_{i}$ ) is invalid (e.g., missing <type>/<question>/<answer> tags) then

7: $r_{i} \leftarrow 0$ ;

8: else

9: Sample M answers $\{\hat{a}_{m}\}_{m=1}^{M} \sim \pi_{S}(\cdot \mid v, q_{i})$ ;

10: Get pseudo-label $a^{*} \leftarrow$ MajorityVote( $\{\hat{a}_{m}\}$ );

11: Compute Solver confidence $s(v, q_{i}) \leftarrow \frac{1}{M} \sum_{m=1}^{M} 1\{\hat{a}_{m} = a^{*}\}$ ;

12: Sample shuffled-frame answers $\{\tilde{a}_{m}\}_{m=1}^{M} \sim \pi_{S}(\cdot \mid \sigma(v), q_{i})$ ;

13: Compute shuffled confidence $s(\sigma(v), q_{i}) \leftarrow \frac{1}{M} \sum_{m=1}^{M} 1\{\tilde{a}_{m} = a^{*}\}$ ;

14: Compute difficulty $r_{diff}^{Q} \leftarrow \min(s(v, q_{i}), 1 - s(v, q_{i}))$ ;

15: Compute diversity penalty $r_{div}^{Q}(q_{i})$ via intra-group BLEU clustering;

16: Compute temporal-aware reward $r_{temp}^{Q} \leftarrow \max(0, s(v, q_{i}) - s(\sigma(v), q_{i}))$ ;

17: Final reward: $r_{i} \leftarrow \max(0, r_{\text{diff}}^{Q} - r_{\text{div}}^{Q}) + \lambda_{q} r_{\text{temp}}^{Q}$ ;

18: Update $\pi_{Q}$ via GRPO using rewards $\{r_{i}\}_{i=1}^{G}$ ;

19: // -- Phase 2: Solver Dataset Construction --
20: Initialize curated dataset $D \leftarrow \emptyset$ ;

21: for each video $v \in V$ do

22: Sample window $[t_{s}, t_{e}]$ of K contiguous frames; let $v^{w}$ denote the windowed clip;

23: Sample one question $q \sim \pi_{Q}(\cdot \mid v^{w})$ ; /* Questioner sees only $v^{w}$ here */

24: Sample M answers $\{\hat{a}_{m}\}_{m=1}^{M} \sim \pi_{S}(\cdot \mid v, q)$ ;

25: Get pseudo-label $a^{*} \leftarrow$ MajorityVote( $\{\hat{a}_{m}\}$ );

26: Compute confidence $s(v, q) \leftarrow \frac{1}{M} \sum_{m=1}^{M} 1\{\hat{a}_{m} = a^{*}\}$ ;

27: if $s(v, q) \in [s_{\min}, s_{\max}]$ then

28: Add $(v, [t_{s}, t_{e}], q, a^{*})$ to D;

29: // -- Phase 3: Solver Training --
30: for each minibatch $(v, [t_{s}, t_{e}], q, a^{*}) \in D$ do

31: Sample G Solver responses $\{(\hat{a}_{m}, [\hat{t}_{s}^{(m)}, \hat{t}_{e}^{(m)}])\}_{m=1}^{G} \sim \pi_{S}(\cdot \mid v, q)$ ;

32: Compute correctness acc $_{m} \leftarrow 1\{\hat{a}_{m} = a^{*}\}$ and format reward fmt $_{m}$ ;

33: Compute IoU $u_{m} \leftarrow IoU([\hat{t}_{s}^{(m)}, \hat{t}_{e}^{(m)}], [t_{s}, t_{e}])$ ; /* window-as-pseudo-segment */

34: Final reward: $r_{m} \leftarrow (1-w) acc_{m} + w fmt_{m} + \lambda_{s} u_{m} acc_{m}$ ;

35: Update $\pi_{S}$ via GRPO using rewards $\{r_{m}\}_{m=1}^{G}$ ;

36: return $\pi_{Q}, \pi_{S}$ .

MMVU [32]. A multi-discipline video benchmark of 3,000 expert-annotated question–answer pairs over 1,529 specialized-domain videos that span 27 subjects across four core disciplines (Science, Healthcare, Humanities & Social Sciences, Engineering). Following prior work [1, 3, 5], we evaluate on the multiple-choice subset to keep our numbers directly comparable with the published baselines.

TempCompass [33]. A diagnostic benchmark for fine-grained temporal perception, structured along five basic temporal axes: action, speed, direction, attribute change, and event order, and ten finer sub-aspects such as relative speed, camera direction, and combined change.

VideoMME [34]. A general-purpose video-understanding benchmark of 2,700 questions over 900 videos drawn from six domains and 30 sub-class video types. Each video is bucketed by duration into a short (<2 min), medium (4–15 min), or long (30–60 min) split. We report the average across the three duration splits without subtitles, so the model must rely solely on the visual stream.

# D Temporal Keyword Frequency

The word cloud analysis in §4 (Figure 5) shows a lexical shift in the Questioner’s vocabulary toward temporal and reasoning words. Figure 7 additional presents the frequency change for temporally grounded vocabularies between baseline and EvoVid. Compared to the baseline, our model significantly increases the usage of key temporal and causal terms such as sequence, primary, change, after, and before. Notably, high-level reasoning cues (e.g., sequence, change) exhibit the largest gains, indicating stronger emphasis on temporal progression and event dynamics. The broader coverage of temporal keywords, including less frequent terms like transition, during, and while, further suggests improved diversity in temporal reasoning patterns. Overall, this shift validates that our approach encourages more temporally aware and causally grounded question generation.

![](images/a9054f2887449c4813a6c2bdf0279d7600a1cf77bc44860712d2b4bae4576c8b.jpg)

<details>
<summary>bar</summary>

Temporal-keyword frequency: Baseline vs Ours
| Termor | Baseline (%) | Ours (%) |
| :--- | :--- | :--- |
| sequence | 15.4 | 25.7 |
| primary | 1.9 | 16.6 |
| frame | 1.4 | 7.5 |
| change | 1.1 | 5.9 |
| after | 3.6 | 5.4 |
| before | 0.6 | 3.8 |
| when | 0.8 | 1.9 |
| appear | 2.6 | 1.8 |
| end | 0.6 | 1.7 |
| moment | 0.0 | 1.3 |
| while | 0.0 | 1.1 |
| during | 0.3 | 1.0 |
| transition | 0.2 | 0.8 |
| then | 0.2 | 0.4 |
| start | 0.1 | 0.4 |
| disappear | 0.2 | 0.1 |
</details>

Figure 7: Temporal keyword frequency comparison.

# E Limitations and Broader Impacts

# E.1 Limitations

Despite the promising performance of our video-based self-evolving framework, several limitations remain. First, the current framework is trained with only 16 video frames, which may restrict its ability to model long-range temporal dependencies. Extending the paradigm to handle longer videos and more complex real-world video streams can be potential directions. In addition, our method is primarily motivated by intrinsic supervision signals centered on temporal sensitivity and temporal grounding. Future work could further explore other important aspects of video reasoning, including fine-grained spatial understanding and multi-event dependency modeling.

# E.2 Broader Impacts

This work presents a video-centric self-evolving framework that enables Video-LLMs to improve temporal reasoning directly from raw, unannotated videos without relying on large-scale human supervision. Our work focuses primarily on methodological research and benchmark evaluation, and does not introduce new negative social impacts. However, it may inherit or amplify biases present in large-scale internet video data or pretrained models, potentially leading to unfair or unreliable behavior across different environments and populations.