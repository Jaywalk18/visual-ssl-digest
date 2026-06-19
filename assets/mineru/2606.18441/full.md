# Reasoning as Intersection: Consensus-Frame Alignment for Visual Focus in Video-MLLMs

Chengwen Liu1,∗ Zhe Huang2,∗ Jisheng Dang1 Hong Peng1,† Qi Tian3 Tat-Seng Chua4

1School of Information Science and Engineering, Lanzhou University, Lanzhou, China

2Beijing University of Posts and Telecommunications, Beijing, China

3Cloud and AI BU, Huawei, Shenzhen, China

4School of Computing, National University of Singapore, Singapore

∗Equal contribution. †Corresponding author: Hong Peng.

## Abstract

Reinforcement learning has improved the reasoning ability of large language models, but applying outcome-only rewards to video multimodal large language models (Video-MLLMs) provides limited guidance on which visual evidence should support the answer. Inspired by multisensory integration, where consistent cues can enhance the salience and reliability of perceptual estimates, we introduce Consensus Frame GRPO (CF-GRPO), a temporal-annotationfree process-level reward framework for evidence-aware video reasoning. CF-GRPO constructs a consensus frame prior from intrinsic video cues, including temporal coverage, scene-transition cues, and query-conditioned visual relevance. It then computes a model-side frame-use score from visual and response representations and optimizes their agreement through the Consensus Frame Reward (CFR). With salience-aware sparse aggregation and distribution sharpening, CFR provides a high-contrast reward signal without requiring human temporal annotations. Experiments show that VideoCFR achieves competitive performance across complex video reasoning benchmarks and improves several metrics over representative Video-MLLM and RL baselines, while the consensus prior provides an interpretable view of the evidence frames emphasized during training. The implementation is available at https://github.com/1Pansy/VideoCFR.

Keywords: Video-MLLMs, video reasoning, reinforcement learning, process-level reward, frame evidence alignment.

## 1 Introduction

Recent advances in reinforcement learning (RL), including group relative policy optimization (GRPO), have made post-training an effective way to align large language models (LLMs) and improve their reasoning abilities [1,2]. This direction has also been adapted to video multimodal large language models (Video-MLLMs), where recent methods use answer correctness, temporal consistency, localization, tool-use, or task-specific verifiable rewards to improve video reasoning and grounding [3–9]. These studies show that RL can improve Video-MLLM behavior when the reward is better matched to video tasks.

Despite this progress, video reasoning remains underspecified when supervision is dominated by final-answer correctness. A video contains redundant, temporally distributed, and heterogeneous visual evidence. A rolloutlevel answer reward can indicate whether the response is correct, but it does not specify which frames should support the answer, whether the model has relied on visually relevant evidence, or whether a visually salient but irrelevant frame has dominated the response. Temporal rewards and localization rewards address important aspects of video understanding, but they usually supervise temporal order, answer validity, or explicit grounding targets rather than the frame-level evidence used during generation. Therefore, Video-MLLM RL needs a process signal that is closer to the visual evidence used by the model while still preserving standard outcome supervision.

In Figure 1, we illustrate this motivation. In long or untrimmed videos, a single sampling strategy or a diffuse frame-use pattern can anchor the response to frames that are plausible but not decisive for the question. In the example, the model must identify the evidence frame containing the relevant price tag and compare it with the question requirement; attending to another visually similar frame leads to an incorrect answer. This failure mode is not simply a lack of more frames. It is a mismatch between the evidence needed by the question and the frames emphasized by the generation process.

Existing evidence-selection and video-grounding methods also recognize the importance of informative frames, using keyframe selection, token compression, semanticvisual consensus, or spatio-temporal grounding to reduce redundancy or localize events [10–15]. These methods are closely related, but our goal is different. We do not aim to choose a final input subset at inference time or add a grounding head with temporal annotations. Instead, we ask whether RL training itself can receive a framelevel signal that encourages the generated response to be associated with candidate evidence frames.

![](images/c0eccdf276b140e21aafc5c48e80e17665b7a83c25fd44161b0ddedfb3683e00.jpg)

<details>
<summary>text_image</summary>

wrong evidence
wrong evidence
consensus anchor
</details>

![](images/66b1198f8a797d209d667ba3608d26fe4b1ae249bf3fff0ad1c2cf3ad0815a23.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Visual cue"] --> B["Conjunctive binding + Pattern completion"]
  C["Temporal cue"] --> D["Stable belief = intersection (∩) of all constraints"]
  E["Semantic relevance"] --> D
  F["Nonlinear mixed selectivity"] --> G["Consensus evidence Robust representation supported by complementary cues, resilient to noise."]
  H["Interference: Visual cue, Temporal cue, Semantic relevance"] --> I["Stable belief = intersection (∩) of all constraints"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#ccf,stroke:#333
    style H fill:#ccf,stroke:#333
```
</details>

![](images/412cc41a162764df22d11a7b1a10b910da3abcd33e0817c85b0d0a6b7b907bb0.jpg)

<details>
<summary>text_image</summary>

Baselines: single sampling strategy
Sampling
probability
true evidence window
every k frames
Frame index
</details>

![](images/0ceee2306444f9e20fea62d67cc68fc7e8bbd9ec9d30e50527aca528e19bcbea.jpg)

<details>
<summary>line chart</summary>

| Prior probability | High P(i) (consensus) |
| ----------------- | --------------------- |
| Peak              | High                  |
</details>

![](images/02a437b5af4b79943cbcdb911621060b961b5a8fbf4551b3a9bf2a05f1d83d9c.jpg)

<details>
<summary>bar chart</summary>

OURS VS BASELINES
| Method | TempCompass Accuracy | MVBench Accuracy |
| :--- | :--- | :--- |
| UTR | 0.65 | 0.25 |
| LVID | 0.35 | 0.45 |
| LVA | 0.45 | 0.55 |
| VILA | 0.55 | 0.65 |
| Ours | 1.0 | 0.75 |
</details>

![](images/567559c7f1d8fa6f362809d88c2071c9f5e71158b201001fe6328bf4ee623641.jpg)

<details>
<summary>text_image</summary>

Question
According to the price tags, what's the highest
listed price shown in the video?
→ Requires multi-frame comparison and an evidence frame.

Wrong reasoning
3740 (missed a digit) → wrong-frame anchor
1980(misread tag) → hallucination

Consensus frame ✓
Consensus anchor: f*
Model answer grounded in frame f*
Final:44800
</details>

Figure 1: Motivation of consensus-frame alignment. In long-video QA, single-source sampling or diffuse frame use can anchor the response to visually plausible but incorrect frames. VideoCFR constructs a consensus prior from temporal coverage, scene-transition, and query-relevance cues, and uses this prior as training-time evidence guidance for aligning generation with frames that support the answer.

To obtain such a signal without manual temporal labels, our design uses a consensus prior over frames. The motivation is consistent with multisensory integration in biological perception, where information from multiple channels can improve event salience and estimate reliability under uncertainty [16–19]. For video reasoning, we instantiate this idea through intrinsic video cues rather than through human annotations: uniform temporal coverage preserves the global sequence, scene transitions capture visual changes, and query-conditioned visual relevance identifies frames related to the question. Their agreement defines a soft consensus frame prior. This prior is not treated as ground-truth temporal annotation; it is a temporal-annotation-free estimate of which candidate frames are more likely to contain useful evidence.

We introduce Consensus Frame GRPO (CF-GRPO), which supplements standard answer rewards with the Consensus Frame Reward (CFR). CFR encourages agreement between the consensus frame prior and a model-side frameuse score extracted from visual and response representations. This design is also related to process supervision, which provides feedback on intermediate reasoning behavior rather than only final answers [20–23]. However, video evidence alignment requires feedback over temporally distributed visual frames, whereas existing process rewards mainly evaluate textual reasoning steps or imageconditioned reasoning traces. CFR therefore converts process supervision into a frame-level reward for video RL without requiring human temporal annotations.

CF-GRPO changes the training objective rather than only filtering frames before inference. During training, the consensus prior is compared with the model-side frame-use distribution, and their overlap provides a scalar reward that can be optimized together with accuracy, structural, and temporal rewards. To make this reward more discriminative, CFR uses salience-aware sparse aggregation to preserve high-response visual regions and temperature sharpening to reduce overly diffuse frame scores. Experiments show that VideoCFR obtains competitive performance on complex video reasoning benchmarks, and ablations indicate that the consensus prior, sparse aggregation, and sharpening each contribute to the final performance. Visual analyses further show that high-attention responseframe events are more concentrated on consensus frames, which is consistent with the intended evidence-alignment effect.

Our main contributions are as follows:

• We propose Consensus Frame GRPO, a temporalannotation-free process-level RL framework for video reasoning. It augments outcome rewards with framelevel evidence alignment while avoiding expensive human temporal annotations.  
• We develop the Consensus Frame Reward mechanism. It constructs a consensus prior from temporal coverage, scene-transition, and query-conditioned visual

relevance, then aligns this prior with a model-side frame-use score through salience-aware sparse aggregation and distribution sharpening.

• Our VideoCFR model achieves competitive performance on complex video reasoning benchmarks, and ablation and visualization analyses show that the proposed reward provides useful evidence-level training signals and interpretable frame-level priors.

## 2 Related Work

## 2.1 Reinforcement Learning for Video-MLLMs

RL has become an effective post-training technique for aligning model behavior and improving reasoning in language models and multimodal models [1, 2]. In the video domain, recent work adapts GRPO-style optimization to Video-MLLMs by defining verifiable rewards for video QA, temporal ordering, and grounding. Video-R1 introduces T-GRPO, which combines rule-based answer rewards with an ordered-versus-shuffled temporal reward to reduce reliance on single-frame shortcuts [3]. DeepVideo-R1 studies optimization instability in video reinforcement fine-tuning and introduces Regressive GRPO with difficulty-aware data augmentation [4]. VideoChat-R1 applies reinforcement fine-tuning to spatio-temporal perception tasks, including temporal grounding and object tracking, showing that task-specific reward design can improve grounded video QA [5]. Temporal-RLT further analyzes reward design and data efficiency for VideoLLM reinforcement learning [7]. Recent studies also investigate data-efficient video RL and tool-augmented long-video reasoning, indicating that video-specific reward design is becoming a central issue in Video-MLLM post-training [8, 9].

Several related studies broaden this direction by improving training recipes, model scales, or temporal reward construction. TinyLLaVA-Video-R1 investigates whether smaller multimodal models can acquire video reasoning ability through reinforcement learning [24]. VideoRFT focuses on reinforced fine-tuning for video reasoning capability [6]. VIDEO-UTR studies temporal reward construction for scalable Video-MLLM training [25]. These methods are closely related to ours because they share the same post-training setting and aim to make reward signals more suitable for video. However, their supervision is mainly expressed through final answer correctness, temporal consistency, or task-specific grounding labels. Such rewards indicate whether a response is correct or temporally consistent, but they provide limited direct guidance on which visual evidence frames should support the generated answer. CFR addresses this missing signal by introducing a frame-level evidence-alignment reward while retaining standard outcome and temporal rewards.

## 2.2 Process Supervision and Reward Modeling

Process supervision provides feedback on intermediate reasoning states rather than only on final answers. In text reasoning, process- and outcome-based supervision have been compared on mathematical problem solving, and step-level reward models have been shown to provide informative training and selection signals [20,21]. This line of work motivates the use of denser feedback when finalanswer supervision is under-specified. For multimodal reasoning, VisualPRM extends process reward modeling to visual tasks by training a multimodal reward model and evaluating step-wise correctness in reasoning traces [22]. Perceptual-evidence anchored RL further emphasizes that multimodal reasoning rewards can benefit from explicitly connecting reasoning behavior with perceptual evidence [23].

Despite their relevance, existing process-supervision methods do not directly solve the video evidence-alignment problem considered here. Textual process rewards usually supervise logical steps, while multimodal PRMs typically evaluate reasoning traces or image-conditioned steps. Video reasoning additionally requires identifying temporally distributed evidence from redundant frame sequences. Directly collecting human labels for which frames support each answer would be expensive and dataset-dependent. Our method therefore constructs process feedback from intrinsic video cues. The consensus prior is not treated as ground-truth temporal annotation; instead, it defines a temporal-annotation-free frame-level prior that is used to regularize the model-side frame-use score during RL.

## 2.3 Video Evidence Selection and Spatio-Temporal Grounding

Another closely related direction addresses video redundancy by selecting, compressing, or explicitly grounding visual evidence before generation. Video-MLLMs have explored compact frame representations, long-context transfer, and video-oriented training recipes to process richer temporal context [26–30]. These approaches improve the capacity to ingest video context, but they do not by themselves specify which frames should be emphasized during RL optimization.

Keyframe selection methods make this evidence problem explicit. Adaptive Keyframe Sampling formulates long-video understanding as an input-side frame selection problem under a limited frame budget [10]. FOCUS similarly studies efficient keyframe selection for long-video understanding [11]. SeViCES uses semantic-visual evidence consensus to identify informative frames and refine answers without retraining the backbone model [12]. Efficient Frame Selection further trains a selector with RL so that retained frames improve downstream video understanding [13]. These methods show that relevance, coverage, and semantic-visual agreement are useful criteria for reducing video redundancy.

![](images/cc29971503a0498da4fbbb6a6ff244e8abf98b4760bd5462847eec6bd3b96767.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Input Video"] -->|①| B["Uniform candidates + Multi-source cues"]
  B --> C["Multi-strategy Frame Consensus Selector"]
  C --> D["Question (e.g.: What is the person cooking?)"]
  D --> E["Reward Design"]
  E --> F["Optimized model (Consensus-aware)"]
  F --> G["Benefits"]
    
    subgraph B_Consensus_Aware_GRPO_Optimization["B: Consensus-Aware GRPO Optimization"]
  H["Visual signal"] --> I["Temporal signal"]
  I --> J["Temporal-aligned"]
  J --> K["Visual-aligned"]
  K --> L["Semantic similarity"]
  L --> M["Interpretable: intersection of cues"]
  M --> N["High-P(i) candidates"]
    end
    
    subgraph R_Reward_Design["R: Reward Design"]
  O["Accuracy reward"] --> P["Temporal Consistency reward"]
  P --> Q["Structural consistency reward"]
  Q --> R["Consensus Frame Coverage reward (CFR)"]
  R --> S["R = w1R_acc + w2R_temp + w3R_struct + w4R_CFR"]
  S --> T["Policy Update Feedback Loop"]
  T --> U["GRPO (RL fine-tuning)"]
  U --> V["Consensus Bonus (P,α) guides training"]
  V --> W["Evidence Frame Guidance"]
    end
```
</details>

Figure 2: Overview of the CF-GRPO framework. Panel A constructs a multi-source consensus prior from uniform coverage, scene transitions, and query-conditioned semantic relevance. Panel B incorporates CFR into GRPO, rewarding overlap between the consensus prior and the model-side frame-use distribution while preserving accuracy, temporal, and structural rewards.

(A) Multi-source Selection  
![](images/a5504e43491a04364b01cb103c275d061a880a3c671a652e3686214de74ae8ce.jpg)

<details>
<summary>chart content</summary>

| Chart Type | Metric | Value |
| :--- | :--- | :--- |
| (a1) Scene-cut Prior | scene-change score | - |
| (a1) Scene-cut Prior | HSV histogram distance | - |
| (a2) Semantic Prior | similarity score | - |
| (a2) Semantic Prior | SigLIP text-image similarity | - |
| (a3) Multi-source overlay view | S_uniform = {f1...f16}, S_scene = {(f5,f10,f11,f14,f15), S_sem = {f7,f8,f11,f15} | - |
| (a3) Multi-source overlay view | f1 | - |
| (a3) Multi-source overlay view | f2 | - |
| (a3) Multi-source overlay view | f3 | - |
| (a3) Multi-source overlay view | f4 | - |
| (a3) Multi-source overlay view | f5 | - |
| (a3) Multi-source overlay view | f6 | - |
| (a3) Multi-source overlay view | f7 | - |
| (a3) Multi-source overlay view | f8 | - |
| (a3) Multi-source overlay view | f9 | - |
| (a3) Multi-source overlay view | f10 | - |
| (a3) Multi-source overlay view | f11 | - |
| (a3) Multi-source overlay view | f12 | - |
| (a3) Multi-source overlay view | f13 | - |
| (a3) Multi-source overlay view | f14 | - |
| (a3) Multi-source overlay view | f15 | - |
| (a3) Multi-source overlay view | f16 | - |
</details>

(B) Consensus Weighting& Normalization→P(i)  
![](images/c328adb2d7b9cd55fe2bb1f3f5de362eff711c338f5ada20d074db575b3866b1.jpg)

<details>
<summary>text_image</summary>

(b1) Probability Formulas
w_i = w_base + 1(i \in S_\text{scene})\lambda_\text{scene} + 1(i \in S_\text{sem})\lambda_\text{sem}
w_base = 0.1, \lambda_\text{scene} = 1.0, \lambda_\text{sem} = 3.0
P(i) = \frac{w_i}{\sum_j w_j}
(b2) Weight contributions
(Semantic prior is often peaked)
frame index
(b3) Normalized prior
(Final prior distribution)
frame index
(b4) Final P(i) Output
P(i) = 0.02 0.02 0.02 0.08 ... 0.08 0.05 0.15 0.02
frame_sets={uniform:[...], scene:[...], semantic[...]}
Training signal
</details>

Figure 3: Multi-source consensus prior construction. Candidate frames are characterized through uniform coverage, scene-cut responses, and query-conditioned semantic relevance. These cues are fused through weighted scoring and normalization to obtain $P ( i )$ , a soft temporal-annotation-free prior for downstream evidence alignment rather than a ground-truth temporal label.

Spatio-temporal grounding methods pursue a related but different goal: they aim to localize when and where queried events or objects occur. SpaceVLLM introduces spatio-temporal aware queries and a query-guided space decoder, together with grounding data, to improve explicit localization in videos [15]. Such methods provide strong grounding capabilities, but they usually require architectural components, grounding annotations, or task-specific localization objectives.

CFR is complementary to both evidence selection and grounding. It does not select a final input subset at inference time, and it does not add a grounding head to the Video-MLLM. Instead, it converts multiple evidence cues, including temporal coverage, scene transitions, and queryconditioned visual relevance, into a consensus prior for RL training. The reward then encourages agreement between this prior and the model-side frame-use score extracted from visual and response representations. Therefore, our contribution lies in process-level reward design for evidence alignment, rather than input compression, training-free keyframe selection, or architecture-level grounding.

## 3 Method

## 3.1 Overview of Consensus Frame GRPO

As illustrated in Figure 2, CF-GRPO augments standard Video-MLLM RL with a temporal-annotation-free process reward. The goal is not to replace answer correctness, but to reduce reward underspecification by adding a frame-level signal about whether the generated response is aligned with candidate visual evidence.

In Panel A, we construct a consensus prior by fusing multiple complementary cues, including uniform temporal coverage, scene transitions, and query-conditioned visual relevance. This prior highlights frames that contain informative evidence while retaining broad temporal coverage.

In Panel B, the consensus prior is used to compute an auxiliary reward during Video-MLLM training. A model-side frame-use score is extracted from the similarity between visual frame representations and response hidden states. The Consensus Frame Reward (CFR) rewards agreement between this score and the consensus prior, alongside standard accuracy, structure, and temporal consistency rewards. This optimization provides evidence-level feedback without requiring human temporal labels.

Preliminaries We consider an original video sequence with T frames and uniformly sample K frames as visual input. Unless otherwise noted, the frame index i refers to one of these K sampled frames, and all frame-level distributions are defined over the sampled frame set. The model generates responses conditioned on a query q and the sampled video frames. Training employs GRPO, a policy optimization method for grouped RL fine-tuning, where policies are updated based on grouped preferences over multiple rollouts.

## 3.2 Consensus Prior Construction

As illustrated in Figure 3, the construction of our consensus prior follows a two-stage pipeline: multi-source selection and consensus weighting.

Multi-source Selection To identify candidate evidence frames, we combine signals from complementary sources, as depicted in Figure 3(A). To capture scene transitions, we compute the scene-cut prior shown in Figure 3(a1) by analyzing the Bhattacharyya distance between HSV histograms of adjacent sampled frames. Peaks in this metric signify substantial visual shifts, and a preset scene-change criterion is applied to isolate distinct scene boundaries from minor fluctuations. Additionally, we integrate the semantic prior in Figure 3(a2), which is derived from a pretrained visual-language encoder. By evaluating imagetext similarity between sampled frames and the user query, we prioritize the top-ranked frames that are most directly relevant to the query. The selection process is grounded by the uniform baseline in Figure 3(a3), which maintains temporal coverage across the video.

Consensus Weighting & Normalization The selected signals are fused into a probabilistic distribution, as shown in Figure 3(B). We assign importance weights wi to each frame i based on the contributions from the uniform base, scene transitions, and semantic relevance, as shown in Figure 3(b1, b2):

$$
w _ {i} = w _ {\text { base }} + \mathbb {I} (i \in \mathcal {S} _ {\text { scene }}) \cdot \lambda_ {\text { scene }} + \mathbb {I} (i \in \mathcal {S} _ {\text { sem }}) \cdot \lambda_ {\text { sem }}, \tag {1}
$$

where $w _ { \mathrm { b a s e } }$ prevents probability collapse, and $w _ { \mathrm { b a s e } } ,$ $\lambda _ { \mathrm { s c e n e } } .$ and $\lambda _ { \mathrm { s e m } }$ are hyperparameters. These weights are then normalized to produce the final consensus prior $P ( i )$ , as shown in Figure 3(b3, b4):

$$
P (i) = \frac {w _ {i}}{\sum_ {j = 1} ^ {K} w _ {j}}. \tag {2}
$$

This resulting distribution $P ( i )$ serves as a temporalannotation-free evidence prior for reward construction.

## 3.3 Model-side Frame-use Score Extraction and Sharpening

To align generation with the consensus prior, we require a frame-level score that is comparable to $P ( i )$ . We compute this model-side frame-use score from the similarity between visual frame representations and response hidden states. Since the model processes visual information as a sequence of patch tokens, these fine-grained signals are aggregated into frame-level vectors before the similarity calculation. We aggregate frame features using salienceaware sparse aggregation to preserve sparse signals. Each sampled frame is encoded into M visual patch tokens, yielding $H _ { \mathrm { v i s } } \in \mathbb { R } ^ { K \times M \times D }$ . The aggregated frame vector is computed by channel-wise max pooling:

![](images/4152624fd4f17c0c22d3507a2d077711aa460910eb579ffba339276df66290c5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Uniform base"] --> B["Scene transitions"]
  B --> C["Question relevance"]
  C --> D["Multi-source consensus prior P(i)"]
  D --> E["Guide by CFR"]
  E --> F["The consensus score: R_raw = Σ P(i)·α(i)"]
  F --> G["Policy optimization"]
  G --> H["Model-side frame usage α(i)"]
  H --> I["Response hidden states"]
  I --> J["Frame vectors"]
  J --> K["Aggregation"]
  K --> L["Semantic alignment"]
  L --> M["Alignment effect (α peaks move toward P peaks)"]
  M --> N["Overlap (P,α) ↑ (0.18→0.42)"]
  N --> O["Higher overlap → Better evidence grounding"]
```
</details>

Figure 4: Consensus Frame Reward as a process signal. CFR estimates a model-side frame-use distribution from response-frame similarity and rewards its overlap with the consensus prior $P ( i )$ . The resulting scalar reward guides GRPO toward generation behavior that is more aligned with candidate evidence frames.

$$
[ \mathbf {h} _ {i} ^ {v} ] _ {d} = \max _ {m = 1} ^ {M} H _ {\mathrm{vis}} [ i, m, d ], \quad d = 1, \dots , D, \tag {3}
$$

capturing high-response regions without dilution by background noise.

For alignment, given the response hidden states $H _ { \mathrm { r e s } } \in$ $\mathbb { R } ^ { L \times D }$ , we compute raw weight scores:

$$
S _ {i} = \frac {1}{L} \sum_ {t = 1} ^ {L} \frac {\mathbf {h} _ {t} ^ {\mathrm{res}} \cdot (\mathbf {h} _ {i} ^ {v}) ^ {\top}}{\| \mathbf {h} _ {t} ^ {\mathrm{res}} \| \| \mathbf {h} _ {i} ^ {v} \|}. \tag {4}
$$

Here, $S _ { i }$ quantifies the representation-level association between the i-th frame and the generated response. By averaging the cosine similarity across all response tokens t, we obtain a global frame-use score for the entire generation process.

The raw scores are sharpened with temperature $\tau$ by applying a softmax over the K sampled frames:

$$
\boldsymbol {w} _ {\mathrm{attn}} (i) = \frac {\exp (S _ {i} / \tau)}{\sum_ {j = 1} ^ {K} \exp (S _ {j} / \tau)}. \tag {5}
$$

This enhances the signal-to-noise ratio for reliable policy gradients.

Consensus Frame Reward. As shown in Figure 4, the consensus frame reward is the dot product:

$$
R _ {\mathrm{cf}} = \mathbf {P} \cdot \boldsymbol {w} _ {\mathrm{attn}} = \sum_ {i = 1} ^ {K} P (i) \cdot \boldsymbol {w} _ {\mathrm{attn}} (i), \tag {6}
$$

rewarding agreement between the temporal-annotationfree consensus prior and the model-side frame-use score.

Because both vectors are normalized over the K sampled frames, the raw overlap is a bounded scalar whose magnitude depends on the frame budget; when visualizing training dynamics, we therefore report a scaled version of this overlap as a diagnostic rather than as a separate reward term.

## 3.4 Reward Design and Optimization

The final component of our framework is the optimization objective, shown in Figure 2 (B). We use a composite reward function that preserves standard outcome supervision while adding evidence-level feedback. The reward contains four terms: accuracy for answer correctness, structural constraints for response format and conciseness, temporal consistency for ordered video reasoning, and CFR for agreement with candidate evidence frames. The total reward integrates these standard and process-oriented terms:

$$
R _ {\text { total }} (o) = R _ {\text { acc }} + R _ {\text { struct }} + \lambda \cdot R _ {\text { cf }} + R _ {\text { temp }}, \tag {7}
$$

where $R _ { \mathrm { a c c } }$ evaluates answer accuracy, and $R _ { \mathrm { s t r u c t } }$ (formatand-length reward) penalizes excessive generation length and incorrect response formats to encourage conciseness and structural compliance. The temporal reward $R _ { \mathrm { t e m p } }$ encourages sensitivity to frame order:

$$
R _ {\text { temp }} = \left\{ \begin{array}{l l} \gamma & \text { if   } \mathrm{Acc} _ {\text { ordered }} \geq \mathrm{Acc} _ {\text { shuffled }} \text {   and   } o \text {   is   correct } \\ 0 & \text { otherwise }, \end{array} \right. \tag {8}
$$

Here, $\mathrm { A c c } _ { \mathrm { o r d e r e d } }$ and $\mathrm { A c c } _ { \mathrm { s h u f f l e d } }$ denote verifiable answer scores obtained from the ordered and shuffled frame sequences for the same sample. This reward incentivizes the model to perform better on temporally ordered inputs than on shuffled sequences when the answer is correct.

Optimization uses GRPO [2], grouping rollouts and updating the policy π via preference optimization. Let

$$
\rho_ {i, t} (\theta) = \frac {\pi_ {\theta} (o _ {i , t} \mid v , q , o _ {i , <   t})}{\pi_ {\theta_ {\mathrm{old}}} (o _ {i , t} \mid v , q , o _ {i , <   t})}, \tag {9}
$$

where v denotes the sampled video input and D denotes the training distribution. For each video-question pair $( v , q )$ , a group of outputs is sampled from the old policy. Following GRPO, the clipped objective is:

$$
\mathcal {J} _ {\mathrm{CF-GRPO}} (\theta) = \mathbb {E} _ {\mathcal {D}, \pi_ {\theta_ {\text {old}}}} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| o _ {i} |} \sum_ {t = 1} ^ {| o _ {i} |} \ell_ {i, t} (\theta) \right], \tag {10}
$$

$$
\begin{array}{l} \ell_ {i, t} (\theta) = \min \left(\rho_ {i, t} (\theta) \hat {A} _ {i}, \operatorname{clip} _ {\epsilon} \left(\rho_ {i, t} (\theta)\right) \hat {A} _ {i}\right) \\ - \beta \mathbb {D} _ {\mathrm{KL}} [ \pi_ {\theta} \| \pi_ {\text { ref }} ] _ {i, t}, \\ \end{array}
$$

where cli $\ L _ { \epsilon } ( x ) = \mathrm { c l i p } ( x , 1 - \epsilon , 1 + \epsilon )$ , and $\hat { A } _ { i }$ is estimated from the group-normalized completion reward based on $r ( o _ { i } ) = R _ { \mathrm { t o t a l } } ( o _ { i } )$ . The term $\mathbb { D } _ { \mathrm { K L } } [ \pi _ { \theta } \lVert \pi _ { \mathrm { r e f } } ] _ { i , t }$ denotes the token-level KL estimator used in $\mathrm { G R P O }$ . Thus, the policy update is token-wise, while the reward and advantage are defined for the complete rollout. This loop optimizes a consensus-aware policy while keeping the model architecture unchanged.

## 4 Experiments

## 4.1 Experimental Setup

Training Details Training begins with one epoch of SFT on the Video-R1 CoT-165k dataset and continues with VideoCFR optimization using the proposed CF-GRPO algorithm on 8 NVIDIA A800 GPUs with a perdevice batch size of 4. The training data incorporates both image and video reasoning samples. The video portion comprises 100K video-question-answer pairs sampled from MSRVTT-QA, MSVD-QA, and ActivityNet-QA, augmented with synthetic temporal reasoning queries. For efficiency, training limits video frames to a maximum of 16, with each frame processed under a dynamically adapted pixel budget up to $1 2 8 \times 2 8 \times 2 8$ . During inference, the pixel budget increases to $2 5 6 \times 2 8 \times 2 8$ , and the sampled frame budget ranges from 16 to 64 for better performance. Group size G is set to 8. Training runs for 5 epochs, taking approximately 48 hours.

Implementation Hyperparameters We use the AdamW optimizer with a learning rate of $5 \times 1 0 ^ { - 6 }$ and weight decay of 0.01. The KL divergence hyperparameter β is 0.04. Gradient norm is clipped to 5 for stability. For the CFR mechanism, the reward scaling factor is $\lambda = 3 . 0$ , and the sharpening temperature is τ = 0.1. Hierarchical prior weights are $w _ { \mathrm { b a s e } } = 0 . 1 , \lambda _ { \mathrm { s c e n e } } = 1 . 0$ , and $\lambda _ { \mathrm { s e m } } = 3 . 0$ The temporal reward parameter is $\gamma = 0 . 3$ .

Benchmarks and Baselines We evaluate on a suite of video reasoning benchmarks (VSI-Bench [31], VideoM-MMU [32], MMVU(MC) [33]) and general video understanding benchmarks (MVBench [34], TempCompass [35], VideoMME(w/o sub) [36]). The comparison includes standard Video-MLLMs and recent RL-enhanced baselines when their reported metrics match these benchmarks.

## 4.2 Main Results

We compare VideoCFR with representative Video-MLLMs and recent RL-enhanced baselines in Table 1. The table separates standard Video-MLLMs from RL-based methods to make the comparison axes clear. Overall, VideoCFR obtains competitive results across both reasoning-oriented and general video-understanding benchmarks, suggesting that adding an evidence-level CFR signal is useful beyond answer-only supervision.

Comparison with Video-MLLM and RL Baselines Existing RL approaches, such as Video-R1-7B, VideoChat-R1, and Temporal-RLT, already improve over many standard baselines by introducing video-specific reward designs.

VideoCFR further improves several benchmark scores under comparable model scale, suggesting that frame-level evidence alignment provides an additional training signal. For example, on VideoMMMU, our 32-frame model achieves 52.4%, improving over the reported Video-R1-7B result by 5.2% and the reported VideoChat-R1 result by 3.7%. Compared with the recent MARC-3B compression model, VideoCFR reports higher accuracy on all six benchmarks, although this comparison should be interpreted as accuracy context because MARC-3B is optimized for one-frame-equivalent token compression with a smaller backbone. These results support the role of evidence-level reward design in Video-MLLM post-training, while the per-benchmark gains remain dependent on frame budget and task type.

Scaling with Temporal Context Increasing the sampled frame budget changes performance in a benchmarkdependent manner. VideoCFR benefits from more frames on several benchmarks: increasing from 16 to 64 frames improves VideoMME (55.1% → 61.1%), VSI-Bench (31.8% $ 3 4 . 8 \%$ ), and TempCompass $( 7 0 . 8 \%  7 2 . 9 \% )$ . However, VideoMMMU and MVBench peak at smaller frame budgets. This pattern suggests that CFR can help the model use additional temporal context when the benchmark benefits from richer evidence, but additional frames are not uniformly beneficial across all tasks.

## 4.3 Ablation and Control Studies

We separate reward-component ablations from broader training and data controls. Table 2 isolates the components inside the CFR mechanism under the same 16-frame VideoCFR setting, while Table 3 varies the training recipe, the use of image QA data, and the use of an external frame-selection method. This separation avoids conflating internal reward design with changes in training stages or input-frame selection.

Reward-Component Ablation The ablation starts from the source of the reward signal by contrasting the multi-source consensus prior with a uniform prior. The uniform prior assumes an equiprobable distribution across frames $( P ( i ) = 1 / K )$ . Since this makes the overlap with any normalized model-side frame-use distribution constant, this variant removes the informative frame-prior signal while preserving the same training setting. In contrast, our consensus prior integrates scene-transition and semantic relevance cues into a non-uniform evidence prior. As detailed in Table 2, replacing the consensus prior with a uniform distribution reduces all three reasoning benchmarks and MVBench, most notably by 3.2% on VideoM-MMU, although it yields small gains on TempCompass and VideoMME. Removing the semantic prior reduces four of six benchmarks, while removing the scene prior reduces all six benchmarks. These results indicate that the consensus prior is most beneficial on reasoning-oriented evaluations, and that combining complementary evidence sources generally provides a stronger signal than relying on a single cue.

Table 1: Video model performance comparison. Results are reported as accuracy (%). For MARC-3B, the frame entry follows the original paper and denotes one-frame-equivalent visual tokens after compression.

<table><tr><td rowspan="2">MODELS</td><td rowspan="2">FRAMES</td><td colspan="3">VIDEO REASONING BENCHMARK</td><td colspan="3">VIDEO GENERAL BENCHMARK</td></tr><tr><td>VSI-BENCH</td><td>VIDEOMMMU</td><td>MMVU(MC)</td><td>MVBENCH</td><td>TEMPCOMPASS</td><td>VIDEOMME(w/o sub)</td></tr><tr><td>LLAMA-VID [26]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>41.9</td><td>45.6</td><td>-</td></tr><tr><td>VIDEOLLAMA2 [27]</td><td>-</td><td>-</td><td>-</td><td>44.8</td><td>54.6</td><td>-</td><td>47.9</td></tr><tr><td>LONGVA-7B [28]</td><td>-</td><td>29.2</td><td>23.9</td><td>-</td><td>-</td><td>56.9</td><td>52.6</td></tr><tr><td>VILA-1.5-8B [37]</td><td>-</td><td>28.9</td><td>20.8</td><td>-</td><td>-</td><td>58.8</td><td>-</td></tr><tr><td>VILA-1.5-40B [37]</td><td>-</td><td>31.2</td><td>34.0</td><td>-</td><td>-</td><td>-</td><td>60.1</td></tr><tr><td>VIDEO-UTR-7B [25]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>58.8</td><td>59.7</td><td>52.6</td></tr><tr><td>LLAVA-ONEVISION-7B [29]</td><td>-</td><td>32.4</td><td>33.8</td><td>49.2</td><td>56.7</td><td>-</td><td>58.2</td></tr><tr><td>KANGEROO-8B [30]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>61.1</td><td>62.5</td><td>56.0</td></tr><tr><td>QWEN2.5-VL-7B [38]</td><td>-</td><td>-</td><td>47.4</td><td>61.3</td><td>59.4</td><td>69.2</td><td>52.8</td></tr><tr><td>VIDEO-R1-7B [3]</td><td>16</td><td>30.3</td><td>47.2</td><td>63.5</td><td>62.4</td><td>70.8</td><td>54.3</td></tr><tr><td>VIDEOCHAT-R1 [5]</td><td>16</td><td>28.9</td><td>48.7</td><td>65.8</td><td>64.2</td><td>73.5</td><td>57.7</td></tr><tr><td>DEEPVIDEO-R1 [4]</td><td>-</td><td>33.0</td><td>40.7</td><td>59.0</td><td>49.6</td><td>63.1</td><td>51.1</td></tr><tr><td>TINYLLAVA-VIDEO-R1 [24]</td><td>16</td><td>-</td><td>-</td><td>46.9</td><td>-</td><td>49.5</td><td>46.6</td></tr><tr><td>VIDEORFT [6]</td><td>32</td><td>-</td><td>-</td><td>51.1</td><td>62.1</td><td>-</td><td>-</td></tr><tr><td>TEMPORAL-RLT [7]</td><td>32</td><td>-</td><td>-</td><td>65.0</td><td>-</td><td>-</td><td>57.6</td></tr><tr><td>MARC-3B [14]</td><td>1</td><td>27.6</td><td>33.1</td><td>52.0</td><td>45.8</td><td>55.3</td><td>39.4</td></tr><tr><td>VIDEOCFR (OURS)</td><td>16</td><td>31.8</td><td>50.5</td><td>66.4</td><td>66.1</td><td>70.8</td><td>55.1</td></tr><tr><td>VIDEOCFR (OURS)</td><td>32</td><td>33.1</td><td>52.4</td><td>65.9</td><td>64.5</td><td>72.8</td><td>58.9</td></tr><tr><td>VIDEOCFR (OURS)</td><td>64</td><td>34.8</td><td>50.6</td><td>66.7</td><td>63.9</td><td>72.9</td><td>61.1</td></tr></table>

Table 2: Ablation study of VideoCFR reward components. Results are reported as accuracy (%).

<table><tr><td rowspan="2">MODELS</td><td rowspan="2">FRAMES</td><td colspan="3">VIDEO REASONING BENCHMARK</td><td colspan="3">VIDEO GENERAL BENCHMARK</td></tr><tr><td>VSI-BENCH</td><td>VIDEOMMMU</td><td>MMVU(MC)</td><td>MVBENCH</td><td>TEMPCOMPASS</td><td>VIDEOMME(w/o sub)</td></tr><tr><td>VIDEOCFR (OURS)</td><td>16</td><td>31.8</td><td>50.5</td><td>66.4</td><td>66.1</td><td>70.8</td><td>55.1</td></tr><tr><td>UNIFORM PRIOR</td><td>16</td><td>31.4</td><td>47.3</td><td>63.5</td><td>63.1</td><td>71.0</td><td>55.8</td></tr><tr><td>w/o SEMANTIC PRIOR</td><td>16</td><td>31.8</td><td>48.9</td><td>64.3</td><td>64.1</td><td>70.9</td><td>54.4</td></tr><tr><td>w/o SCENE PRIOR</td><td>16</td><td>31.0</td><td>46.1</td><td>65.8</td><td>63.2</td><td>70.7</td><td>54.1</td></tr><tr><td>w/o SPARSE AGGREGATION</td><td>16</td><td>32.2</td><td>47.3</td><td>64.9</td><td>63.6</td><td>70.8</td><td>55.4</td></tr><tr><td>w/o SHARPENING</td><td>16</td><td>30.6</td><td>49.1</td><td>64.9</td><td>63.2</td><td>71.1</td><td>54.2</td></tr></table>

The same table also evaluates the model-side signal construction. Removing sparse aggregation lowers VideoM-MMU, MMVU(MC), and MVBench, which are benchmarks where localized or fine-grained visual evidence can affect the answer. Removing distribution sharpening reduces most reasoning and general benchmarks, including VSI-Bench, VideoMMMU, MMVU(MC), MVBench, and VideoMME. The remaining small metric gains in a few columns indicate that these components are not uniformly beneficial for every benchmark, but the overall pattern supports their role in making CFR more informative for complex reasoning tasks.

Training, Data, and Frame-Selection Controls We further evaluate the effects of training stages, training data, and external frame selection in Table 3. SFT only applies cold-start supervised fine-tuning to the base model. SFT + GRPO is a pure reinforcement-learning control without temporal or consensus-frame rewards, while SFT + GRPO + Temporal retains the temporal reward but excludes the consensus-frame reward. VideoCFR w/o SFT applies the VideoCFR reward design directly to the base model without cold-start supervised fine-tuning. VideoCFR improves over SFT on all six benchmarks, over $\mathrm { S F T } + \mathrm { G R P O } +$ Temporal on five of six benchmarks, matches or improves over VideoCFR w/o SFT on five of six benchmarks, and improves over $\mathrm { S F T } + \mathrm { G R P O }$ on four of six benchmarks. These results indicate that cold-start SFT and CFR-based RL provide complementary gains while the effect remains benchmark-dependent.

AKS [10] is used as an external frame-selection method whose selected frames are fed into the same model. Under the matched 16-frame setting, AKS changes performance in a benchmark-dependent manner: it improves some SFT results but does not reproduce the gains obtained by VideoCFR, especially on VideoMMMU, MMVU(MC), and MVBench. This suggests that the gains of VideoCFR cannot be explained by input-side frame selection alone. VideoCFR w/o Image-QA trains without image QA data and uses only video QA, leading to clear drops on VideoM-MMU and MMVU(MC), which indicates that image QA data remains useful for preserving fine-grained visual recognition during video RL training.

Table 3: Control study on training stages, training data, and external frame selection. Results are reported as accuracy (%).

<table><tr><td rowspan="2">VARIANT</td><td rowspan="2">FRAMES</td><td colspan="3">VIDEO REASONING BENCHMARK</td><td colspan="3">VIDEO GENERAL BENCHMARK</td></tr><tr><td>VSI-BENCH</td><td>VIDEOMMMU</td><td>MMVU(MC)</td><td>MVBENCH</td><td>TEMPCOMPASS</td><td>VIDEOMME(w/o sub)</td></tr><tr><td>SFT</td><td>16</td><td>30.2</td><td>44.6</td><td>59.2</td><td>57.1</td><td>69.4</td><td>51.9</td></tr><tr><td>SFT + GRPO</td><td>16</td><td>32.7</td><td>48.3</td><td>62.1</td><td>61.1</td><td>71.3</td><td>54.5</td></tr><tr><td>SFT + GRPO + TEMPORAL</td><td>16</td><td>31.2</td><td>46.8</td><td>63.2</td><td>60.8</td><td>71.1</td><td>53.4</td></tr><tr><td>VIDEOCFR w/o SFT</td><td>16</td><td>31.8</td><td>49.5</td><td>63.8</td><td>60.4</td><td>70.9</td><td>53.8</td></tr><tr><td>SFT + AKS</td><td>16</td><td>32.6</td><td>45.0</td><td>59.0</td><td>57.8</td><td>65.9</td><td>51.9</td></tr><tr><td>VIDEOCFR</td><td>16</td><td>31.8</td><td>50.5</td><td>66.4</td><td>66.1</td><td>70.8</td><td>55.1</td></tr><tr><td>VIDEOCFR + AKS</td><td>16</td><td>31.7</td><td>46.9</td><td>63.5</td><td>63.4</td><td>69.4</td><td>55.3</td></tr><tr><td>VIDEOCFR w/o IMAGE-QA</td><td>16</td><td>31.9</td><td>43.4</td><td>61.4</td><td>63.9</td><td>70.5</td><td>54.1</td></tr></table>

![](images/72d10c42efedd866d49b40f5395c0269d5aabdbd2f403e839e7f989ccae5e357.jpg)

<details>
<summary>line chart</summary>

| Frame Index | Average aggregation | Salience-Aware Sparse Aggregation |
| ----------- | ------------------- | ---------------------------------- |
| 0           | 0.0608              | 0.0615                             |
| 1           | 0.0610              | 0.0625                             |
| 2           | 0.0608              | 0.0620                             |
| 3           | 0.0609              | 0.0618                             |
| 4           | 0.0611              | 0.0625                             |
| 5           | 0.0611              | 0.0625                             |
| 6           | 0.0613              | 0.0625                             |
| 7           | 0.0615              | 0.0622                             |
| 8           | 0.0617              | 0.0612                             |
| 9           | 0.0622              | 0.0623                             |
| 10          | 0.0630              | 0.0625                             |
| 11          | 0.0635              | 0.0625                             |
| 12          | 0.0640              | 0.0630                             |
| 13          | 0.0645              | 0.0635                             |
| 14          | 0.0655              | 0.0635                             |
| 15          | 0.0660              | 0.0640                             |
</details>

Figure 5: Effect of feature aggregation. Salience-aware sparse aggregation better preserves sparse high-response cues than average aggregation, yielding more discriminative frame-level alignment.

## 4.4 Diagnostic Analysis

The following analyses examine whether the internal signals of CFR behave consistently with the intended evidence-alignment mechanism. They do not introduce additional training variants; instead, they inspect training dynamics, frame-use distributions, spatial focus, and attention-event concentration.

Effect of Sparse Feature Aggregation We analyze the feature aggregation strategy used to extract framelevel semantics from sparse visual tokens. Average aggregation can attenuate high-activation visual features by mixing them with background tokens, while salienceaware sparse aggregation preserves the strongest local responses for reward computation. As illustrated in Figure 5, average aggregation shows a systematic temporal bias and misses early-stage visual semantics. In contrast, salience-aware sparse aggregation maintains sharper responses throughout the sequence and exhibits distinct peaks at information-rich moments such as Frames 1 and 4–6. This supports its role in amplifying task-relevant signals before computing the model-side frame-use score.

![](images/61fb1241ab3097dc2fb7410ced9854cdddc29482c27e8624219393392717eca2.jpg)

<details>
<summary>bar chart</summary>

| Frame Index | Probability |
| ----------- | ----------- |
| 0.0         | 0.06        |
| 2.5         | 0.06        |
| 5.0         | 0.06        |
| 7.5         | 0.06        |
| 10.0        | 0.06        |
| 12.5        | 0.06        |
| 15.0        | 0.06        |
</details>

![](images/2d813d3eb592f1c284ef9272e09c51c7d0f892be4d4cc91bf1f4ee557823663f.jpg)

<details>
<summary>bar chart</summary>

| Frame Index | Value  |
| ----------- | ------ |
| 0.0         | 0.047  |
| 2.5         | 0.061  |
| 5.0         | 0.053  |
| 7.5         | 0.061  |
| 10.0        | 0.058  |
| 12.5        | 0.072  |
| 15.0        | 0.092  |
</details>

Figure 6: Effect of distribution sharpening. Lowtemperature sharpening produces a more peaked frame-use distribution than a high-entropy softmax, strengthening the process signal for optimization.

Effect of Distribution Sharpening We examine distribution sharpening by comparing the low-temperature softmax with a standard softmax. As visualized in Figure 6, the standard softmax produces a more diffuse distribution, which weakens the contrast among candidate frames. The low-temperature variant provides a highercontrast frame-use distribution. This is important for CFR because a nearly uniform model-side distribution would make the overlap with the consensus prior less discriminative and would provide weaker policy-gradient feedback.

Training Dynamics and Consensus Alignment We visualize the training dynamics of VideoCFR-7B in Figure 7. The accuracy reward increases and stabilizes during training, while completion length gradually grows and then plateaus. The scaled consensus score rises from 0.47 to over 0.62, indicating that the CFR term is not merely a static auxiliary score but is optimized during RL training. The simultaneous stabilization of accuracy and completion length suggests that the increase in consensus alignment is not simply caused by unbounded response-length growth. Completion length is still reported only as a training diagnostic and should not be interpreted as a direct measure of reasoning quality.

![](images/9dd156fee713e1680153ee88dbe672a3e584f6620ca1beafd3e6b308a325b477.jpg)

<details>
<summary>line chart</summary>

| Global Steps | Accuracy_reward | Completion_Length | Scaled Consensus Score |
| ------------ | --------------- | ----------------- | ---------------------- |
| 0            | 0.65            | 280               | 0.47                   |
| 200          | 0.62            | 350               | 0.48                   |
| 400          | 0.65            | 390               | 0.52                   |
| 600          | 0.67            | 400               | 0.56                   |
| 800          | 0.70            | 405               | 0.59                   |
| 1000         | 0.71            | 410               | 0.62                   |
| 1200         | 0.73            | 405               | 0.63                   |
</details>

Figure 7: Training dynamics of VideoCFR-7B. The curves show the progression of reasoning accuracy, completion length, and scaled consensus alignment during RL training.

![](images/acccb32c8adf80e1c905a89d806b1fb9016c6ab214819bb8292d986030225dcf.jpg)

<details>
<summary>natural_image</summary>

Plate of fried pizza with cheese and meat pieces, placed on a metal mesh surface (no text or symbols visible)
</details>

(a) Video Frame

![](images/1d7720631b747363dfc37f5f4b07c3ba1cc718cb52b7531f133f787e16d03f16.jpg)

<details>
<summary>natural_image</summary>

Colorful abstract illustration of a purple, textured surface on a white plate with geometric background (no text or symbols)
</details>

(b) Average Aggregation

![](images/b3a326c2a94f6589ab3272426ac8c4e9543dd7a0df8c70b49354504fa69a7c1a.jpg)

<details>
<summary>natural_image</summary>

Colorful food photo showing a plated dish with a hot pot and a flame (no text or symbols visible)
</details>

(c) Sparse Aggregation  
Figure 8: Spatial focus visualization on multiple objects. (a) Video frame. (b) Average aggregation heatmap showing fragmented attention. (c) Salience-aware sparse aggregation heatmap covering the pizza and bacon while retaining attention on the background fries.

Spatial Focus under Sparse Aggregation To inspect the learned frame-use behavior at the spatial level, we visualize heatmaps for a food-recognition query in Figure 8. The visualization shows higher response-associated scores on the primary objects—bacon and pizza—while retaining sensitivity to peripheral details such as the fries in the background. Compared with average aggregation, salience-aware sparse aggregation produces a less fragmented pattern and suppresses several background regions. This pattern supports the intended role of sparse aggregation: it preserves localized high-response visual regions that may be diluted by averaging, so the frame-use score used by CFR is less dominated by background tokens.

Attention-Event Enrichment on Consensus Frames. We further analyze whether high-attention response-frame events are concentrated on consensus frames. For each output-token position, we collect global top-5% frametoken attention events and compute their enrichment relative to the corresponding frame-slot baseline, where 1.0× indicates proportional allocation. This metric asks whether consensus frames receive more high-attention events than expected from their share of available frame slots.

As shown in Figure 9(a), non-consensus frames mostly concentrate around or below this baseline over normalized output-token positions. In contrast, Figure 9(b) shows that consensus frames cover a broader above-baseline region, indicating that high-attention events are more likely to occur on frames favored by the consensus prior. The aggregate statistics in Figure 9(c) provide the same pattern at the frame-group level: consensus frames occupy 14.1% of frame slots but account for 19.5% of top-attention events, with a mean enrichment of 1.48×, whereas nonconsensus frames remain below baseline with a mean enrichment of 0.94×. These results do not replace quantitative benchmark evaluation, but they are consistent with the interpretation that CFR encourages response generation to align with frames identified as candidate visual evidence.

## 4.5 Qualitative Case Studies

To connect the diagnostic patterns above with concrete answer behavior, we present qualitative cases. These cases are not used as substitutes for benchmark evaluation; they illustrate how evidence alignment can affect model focus when the relevant visual cue is localized, temporally brief, or visually confusable with irrelevant content.

Robustness to Visual Distractions Figure 10 examines a medical visualization in which the relevant cue is spatially localized. The baseline model (Video-R1-7B) focuses on the gums, which are visually salient but not decisive for the queried procedure, and selects an incorrect procedure option. This behavior is consistent with a wrong-evidence failure mode: the response can be anchored by locally salient content and common language associations, such as linking teeth or gums with contouring, rather than by procedure-specific evidence.

In contrast, VideoCFR-7B assigns higher evidence to the surgical marking lines and selects the correct procedure option. Figure 11 provides a mechanism-level view of this behavior. Mean pooling can attenuate localized signals such as surgical markings by averaging them with background pixels, whereas sparse aggregation preserves localized high-response regions for reward computation. The consensus reward then encourages the model-side frame-use score to align with candidate evidence frames identified through scene-cut and semantic cues. This case is therefore consistent with the intended interaction between sparse signal preservation and consensus-frame alignment.

![](images/0feee72eb263aa4e55406991ed3aed05eb061ca8b656391520892aea1d66e3b3.jpg)

<details>
<summary>line chart</summary>

| Output token position (%) | Event enrichment |
| ------------------------- | ---------------- |
| 0                         | 1.0              |
| 50                        | 1.0              |
| 100                       | 1.0              |
</details>

![](images/7c703d245e07dd02dc12aea95ca4f2ff1e73528b224ac11c06369c4500ec865d.jpg)

<details>
<summary>line chart</summary>

| Output token position (%) | Event enrichment |
| ------------------------- | ---------------- |
| 0                         | 0.0              |
| 50                        | 1.0              |
| 100                       | 1.0              |
</details>

![](images/b0108bef1fcf40042ec4a3d25d1df93a1778ff7ddd5b8a73c933f7528236ee66.jpg)

<details>
<summary>violin chart</summary>

| Frame group   | Event enrichment |
| ------------- | ---------------- |
| Non-consensus | 0.94x            |
| Consensus     | 1.48x            |
</details>

Figure 9: Top-attention event enrichment over normalized output-token positions. (a) Density distribution for non-consensus frames. (b) Density distribution for consensus frames. (c) Global enrichment statistics across frame groups. Enrichment is measured relative to the frame-slot baseline, where 1.0× denotes proportional allocation.

Case Study: Avoiding Visual Distraction via Consensus Frame Reward (CFR)  
![](images/34a39277669a4e245a0bff5d32e41401f9a4083f1a51489b28a2445c8e404e14.jpg)

<details>
<summary>text_image</summary>

Baseline Video-R1
(1) Evidence
Block
Q: Which surgery is demonstrated in the video?
A: Tooth extraction
B: Maxillofacial surgery
C: Root canal
D: Periodontal
E: Gum contouring or gingivectomy
(2)
Question
Block
(3) Model
Output
Block
Video-R1-7B ✗
Prediction: E (Gum contouring / gingivectomy)
"focus on gums... related to gum contouring"
</details>

![](images/51eb87aef45333f6cb9ff3255f4017e70960f97735c7198bc79bb579f7a28cc1.jpg)

<details>
<summary>text_image</summary>

Ours Video-CFR
surgical
marking lines
Q: Which surgery is demonstrated in the video?
A: Tooth extraction
B: Maxillofacial surgery
C: Root canal
D: Periodontal
E: Gum contouring or gingivectomy
Video-CFR-7B ✓
Prediction: B (Maxillofacial surgery)
"lines indicating possible surgical incisions or movements"
</details>

Figure 10: Case study on visual distraction. The baseline (Video-R1) attends to the salient but irrelevant gums and predicts an incorrect option. VideoCFR assigns higher evidence to the surgical marking lines and predicts the correct procedure.

Temporal-Spatial Focusing Across a Sequence Figure 12 examines whether the model uses non-uniform temporal evidence in a video sequence explaining biological mechanisms. The model assigns lower weight to less informative introductory content and higher weight to frames containing relevant mechanisms. Frame 0, an introductory title page, receives the lowest score (0.0879), while Frame 6, which depicts protein transport, receives the highest score (0.1523). This non-uniform distribution is consistent with the effect of distribution sharpening, which is intended to make candidate evidence frames more distinguishable.

The spatial heatmaps provide complementary evidence for localized visual focus. In Frame 3, high-activation regions concentrate on antibody structures and relevant textual annotations, while the white background receives lower activation. Together with the temporal peaks at Frames 3 and 6, this case suggests that VideoCFR can combine temporal frame selection with spatially localized evidence use when multiple frames support the target biological category. This behavior is aligned with the diagnostic results in Figure 9, where consensus frames receive above-baseline high-attention events.

(a) Evidence frames + zoom  
![](images/32a8ef6a592d54430e47653f3137bf6b302b0d9218deb9d34f8467c2f3091eba.jpg)

![](images/d499ca60872740aab0b1a99fc7b5efb5f0a941acd640546d7a474d2fbd5ce214.jpg)

![](images/d42d4082c7ac51332968213f435d1783f7d48da591c2734609442fe68a21dcf8.jpg)

![](images/b649dc0ad1fe1de2391da055d71a2d1ece751eaacb0bc305209f61fa3785a22b.jpg)  
Ours (Video-R1 + CFR): Detail-Oriented Evidence  
(b) Baseline (video-R1): Visual Distraction

Reasoning excerpt (trimmed)

·linesand markings suggest surgical procedure  
·possible surgical incisions or movements  
· correct answer is B

Reasoning excerpt (trimmed)

·detailed view of the gums..  
· aligns with gum contouring/gingivectomy  
·answer should be E

(c) CFR success (with highlighted reasoning)

·lines and markings suggest surgical procedure  
·possible surgical incisions or movements  
·correct answer isB

(d) Mechanism schematic  
![](images/cdc463541d9a6f3eece8f8a47412e8b6b7a1dc72516fa66d80d434a8f7a0bc2c.jpg)  
feature-map  
Mean pooling

![](images/a584fc4d3b19a27d4eb1bae96fc6bc4e725c30c3c6154ee53445c0ab5ca4ec70.jpg)  
spikesaveraged out (flattened,owactivation)  
Max pooling

![](images/cfff49128a03f7d7c4ab7cc0e9874bf83a0700308259bcfddf3fca88cb25daef.jpg)  
spikes preserved (high activation retained)  
CFR encourages focusing on frames containing high-signal surgical markings.

Figure 11: Mechanism schematic for evidence aggregation. The comparison of evidence frames and reasoning excerpts illustrates how salience-aware sparse aggregation preserves localized surgical markings, whereas mean pooling attenuates them through averaging.  
![](images/b9c1351b45cfca66174fe8223db669c3a09ea29c22209641109c176790939baa.jpg)

<details>
<summary>heatmap</summary>

| Frame | Score |
|-------|-------|
| Frame 0 | 0.0879 |
| Frame 1 | 0.1064 |
| Frame 2 | 0.1113 |
| Frame 3 | 0.1387 |
| Frame 4 | 0.1396 |
| Frame 5 | 0.1289 |
| Frame 6 | 0.1523 |
| Frame 7 | 0.1348 |
</details>

Figure 12: Visualization of temporal attention and spatial saliency. The top row displays representative consensus frames from the video with their corresponding attention scores. Higher scores correspond to frames containing task-relevant content (e.g., Frame 6 showing protein transport mechanisms), while lower scores are assigned to less relevant introductory frames. The heatmaps illustrate the model’s spatial focus under salience-aware sparse aggregation.

Key-Frame Evidence Chaining Figure 13 examines a case where the answer depends on a small set of temporally separated visual cues. The selected evidence frames form a sequence that progresses from an approaching moonlike body through a high-speed collision to large-scale destruction. The corresponding rationale compares this evidence with the candidate options and selects the lunarimpact answer rather than distractor categories such as meteorite attack or warfare. This case is consistent with the qualitative role of CFR as an evidence-alignment signal: the answer is supported by a non-uniform chain of relevant frames instead of a frame-independent description of the video.

## 5 Limitations

Although CF-GRPO provides a process-level reward for evidence-aware video reasoning without human temporal annotations, it still has several limitations.

At the evidence-prior level, the consensus prior is an estimate of candidate evidence rather than ground-truth temporal supervision. It is constructed from temporal coverage, scene-transition cues, and query-conditioned visual relevance, so it may be less reliable when the necessary evidence is implicit, visually subtle, or distributed across many frames. This constraint also limits the modality coverage of the current prior, which uses visual and text-query cues only. Videos that require audio, subtitles, speech content, or cross-modal event timing may require additional consensus signals beyond the frame-level visual evidence considered here. At the optimization level, CF-GRPO introduces additional computation during training because it computes consensus priors, model-side frame-use scores, and RL updates over grouped rollouts. In addition, increasing the sampled frame budget is not uniformly beneficial across benchmarks, which suggests that framebudget selection should be adapted to the task rather than treated as a universal scaling rule. At the evaluation level, our diagnostic analyses use internal frame-use and attention-event statistics to inspect evidence alignment. These analyses help interpret the training behavior, but they are not a substitute for evaluations with humanannotated temporal evidence labels.

Future work could explore longer-context video architectures, additional modalities, cached priors, distilled rewards, or online approximations to broaden CF-GRPO toward more comprehensive multimodal video reasoning.

Question: What is the most likely cause of the disaster in the video?

A. Warfare. B. Meteorite attack. C. Lunar impact D. Alien attack.

Stage 1: Initial Key Frame (Reference: Frame 3)  
![](images/aaa5d979ac17a1ef9edfd381fc93394ebd0d73fe3d005e52d531467fdb183390.jpg)  
Initial key-frame cue: a large moon-like body approaching Earth.

![](images/cf358f8e16bd46c29a0329bbe556927d7fb33548b288da27161601005f5b4b79.jpg)

Stage 2: Impact Key Frame (Reference: Frame 5)  
![](images/0620a29ab4a7080bd58db86b7438bc16b8fd21a095630aea39c3d81f6af72483.jpg)  
Impact evidence: high-speed collision causing widespread destruction.

![](images/146717e3e32bc55cee2a2624a66cfef15ae59fbed12f708ee2dc602860254258.jpg)

Stage 3: Final Confirmation (Reference: Frame 9)  
![](images/39843d2ece6bdb09c28a2b118c58a43530c65ee3dfef70eff51a497da2102e9f.jpg)  
Final confirmation: object scale and destruction best match a lunar impact.

I focus on the most informative key frames rather than describing every frame uniformly. The video first shows a large celestial body moving toward Earth, which provides the core clue.

The following key frames further reveal a high-speed impact, dense debris, and widespread destruction. These cues indicate a natural collision event instead of human warfare or alien activity.

I then compare the visual evidence with the candidate options. The object is far larger than a typical meteorite, and the destruction scale is consistent with a moon-like body colliding with Earth.

Therefore, the key-frame evidence chain: large celestial body → Earth collision → global destruction → most strongly supports C. Lunar impact

![](images/66bc7f4f5b21be5911953f1308fdee0e49fb290634f43dac2182ca6e13290306.jpg)

Final Answer: C. Lunar impact

![](images/d48e47bd756ac3924a9ffe3e1676a00c3ccb13d0e8979dc4b72fe95ee03e13b3.jpg)

Video-CFR is more focused on decisive key-frame evidence.

Figure 13: Qualitative case on key-frame evidence chaining. The visualized rationale links an initial moon-like object, the subsequent impact, and the final destruction pattern before selecting the lunar-impact answer.

## 6 Conclusion

We introduce Consensus Frame GRPO, a temporalannotation-free process-level RL framework for Video-MLLMs. By constructing a consensus prior from intrinsic video cues and rewarding its agreement with a model-side frame-use score, CF-GRPO supplements outcome rewards with evidence-level feedback. Experiments show that VideoCFR improves several reasoning-oriented metrics and remains competitive on general video-understanding benchmarks, and ablations indicate that the consensus prior, salience-aware aggregation, and sharpening mechanism each contribute to the final performance. These results suggest that reward design for video reasoning should consider not only answer correctness, but also whether the generation process is aligned with consensus visual evidence.

## References

[1] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in Neural Information Processing Systems, vol. 35, pp. 27 730– 27 744, 2022.  
[2] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, “DeepSeekMath: Pushing the limits of mathematical

reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[3] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, J. Wu, X. Zhang, B. Wang, and X. Yue, “Video-R1: Reinforcing video reasoning in MLLMs,” in Advances in Neural Information Processing Systems, vol. 38, 2025.  
[4] J. Park, J. Na, J. Kim, and H. J. Kim, “DeepVideo-R1: Video reinforcement fine-tuning via difficultyaware regressive GRPO,” in Advances in Neural Information Processing Systems, vol. 38, 2025.  
[5] X. Li, Z. Yan, D. Meng, L. Dong, X. Zeng, Y. He, Y. Wang, Y. Qiao, Y. Wang, and L. Wang, “VideoChat-R1: Enhancing spatio-temporal perception via reinforcement fine-tuning,” arXiv preprint arXiv:2504.06958, 2025.  
[6] Q. Wang, Y. Yu, Y. Yuan, R. Mao, and T. Zhou, “VideoRFT: Incentivizing video reasoning capability in MLLMs via reinforced fine-tuning,” arXiv preprint arXiv:2505.12434, 2025.  
[7] H. Li, S. Han, Y. Liao, J. Luo, J. Gao, S. Yan, and S. Liu, “Reinforcement learning tuning for VideoLLMs: Reward design and data efficiency,” arXiv preprint arXiv:2506.01908, 2025.  
[8] X. Wang, Z. Wu, L. Huang, Y. Zheng, and P. Peng, “Incentivizing versatile video reasoning in MLLMs via data-efficient reinforcement learning,” in Proceedings  
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 5444–5454.  
[9] H. Zhang, X. Gu, J. Li, C. Ma, S. Bai, C. Zhang, B. Zhang, Z. Zhou, D. He, and Y. Tang, “Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 32 903–32 914.  
[10] X. Tang, J. Qiu, L. Xie, Y. Tian, J. Jiao, and Q. Ye, “Adaptive keyframe sampling for long video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 29 118–29 128.  
[11] Z. Zhu, H. Xu, Y. Luo, Y. Liu, K. Sarkar, Z. Yang, and Y. You, “FOCUS: Efficient keyframe selection for long video understanding,” arXiv preprint arXiv:2510.27280, 2025.  
[12] Y. Sheng, Y. Hao, C. Li, S. Wang, and X. He, “Se-ViCES: Unifying semantic-visual evidence consensus for long video understanding,” arXiv preprint arXiv:2510.20622, 2025.  
[13] Y. Qin, H. Li, W. Mu, and Y. He, “Efficient frame selection for long video understanding via reinforcement learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 16 944–16 953.  
[14] P. Wu, Z. Yu, Y. Liu, C.-H. Wu, E. Zhou, and J. Shen, “MARC: Memory-augmented RL token compression for efficient video understanding,” in International Conference on Learning Representations, 2026.  
[15] J. Wang, Z. Zhang, Z. Liu, Y. Li, J. Ge, H. Xie, and Y. Zhang, “SpaceVLLM: Endowing multimodal large language model with spatio-temporal video grounding capability,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 12, 2026, pp. 9912–9920.  
[16] B. E. Stein and T. R. Stanford, “Multisensory integration: Current issues from the perspective of the single neuron,” Nature Reviews Neuroscience, vol. 9, no. 4, pp. 255–266, 2008.  
[17] M. O. Ernst and M. S. Banks, “Humans integrate visual and haptic information in a statistically optimal fashion,” Nature, vol. 415, no. 6870, pp. 429–433, 2002.  
[18] D. Alais and D. Burr, “The ventriloquist effect results from near-optimal bimodal integration,” Current Biology, vol. 14, no. 3, pp. 257–262, 2004.  
[19] D. C. Knill and A. Pouget, “The bayesian brain: The role of uncertainty in neural coding and computation,” Trends in Neurosciences, vol. 27, no. 12, pp. 712–719, 2004.  
[20] J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving, and I. Higgins, “Solving math word problems with process- and outcomebased feedback,” arXiv preprint arXiv:2211.14275, 2022.  
[21] H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe, “Let’s verify step by step,” in International Conference on Learning Representations, 2024.  
[22] W. Wang, Z. Gao, L. Chen, Z. Chen, J. Zhu, X. Zhao, Y. Liu, Y. Cao, S. Ye, X. Zhu, L. Lu, H. Duan, Y. Qiao, J. Dai, and W. Wang, “VisualPRM: An effective process reward model for multimodal reasoning,” arXiv preprint arXiv:2503.10291, 2025.  
[23] C. Zhang, H. Qiu, Q. Zhang, Y. Xu, Z. Zeng, S. Yang, P. Shi, L. Ma, and J. Zhang, “Perceptual-evidence anchored reinforced learning for multimodal reasoning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 41 111–41 120.  
[24] X. Zhang, S. Wen, W. Wu, and L. Huang, “TinyLLaVA-Video-R1: Towards smaller LMMs for video reasoning,” arXiv preprint arXiv:2504.09641, 2025.  
[25] E. Yu, K. Lin, L. Zhao, Y. Wei, Z. Zhu, H. Wei, J. Sun, Z. Ge, X. Zhang, J. Wang et al., “Unhackable temporal rewarding for scalable video MLLMs,” arXiv preprint arXiv:2502.12081, 2025.  
[26] Y. Li, C. Wang, and J. Jia, “LLaMA-VID: An image is worth 2 tokens in large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 323–340.  
[27] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao et al., “VideoL-LaMA 2: Advancing spatial-temporal modeling and audio understanding in Video-LLMs,” arXiv preprint arXiv:2406.07476, 2024.  
[28] P. Zhang, K. Zhang, B. Li, G. Zeng, J. Yang, Y. Zhang, Z. Wang, H. Tan, C. Li, and Z. Liu, “Long context transfer from language to vision,” arXiv preprint arXiv:2406.16852, 2024.  
[29] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “LLaVA-OneVision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.  
[30] J. Liu, Y. Wang, H. Ma, X. Wu, X. Ma, X. Wei, J. Jiao, E. Wu, and J. Hu, “Kangaroo: A powerful video-language model supporting long-context video input,” arXiv preprint arXiv:2408.15542, 2024.  
[31] J. Yang, S. Yang, A. W. Gupta, R. Han, F.-F. Li, and S. Xie, “Thinking in space: How multimodal large language models see, remember, and recall spaces,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 10 632–10 643.  
[32] K. Hu, P. Wu, F. Pu, Wang Xiao, Y. Zhang, X. Yue, B. Li, and Z. Liu, “Video-MMMU: Evaluating knowledge acquisition from multi-discipline professional videos,” arXiv preprint arXiv:2501.13826, 2025.  
[33] Y. Zhao, H. Zhang, L. Xie, T. Hu, G. Gan, Y. Long, Z. Hu, W. Chen, C. Li, Z. Xu et al., “MMVU: Measuring expert-level multi-discipline video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 8475–8489.  
[34] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo et al., “MVBench: A comprehensive multi-modal video understanding benchmark,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 22 195–22 206.  
[35] Y. Liu, S. Li, Y. Liu, Y. Wang, S. Ren, L. Li, S. Chen, X. Sun, and L. Hou, “TempCompass: Do video LLMs really understand videos?” arXiv preprint arXiv:2403.00476, 2024.  
[36] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang et al., “Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 24 108–24 118.  
[37] J. Lin, H. Yin, W. Ping, P. Molchanov, M. Shoeybi, and S. Han, “VILA: On pre-training for visual language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 26 689–26 699.  
[38] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.