# MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models

Hanyang Yu 1,‡, Haitao Lin 2,†, Jingbo Zhang 2, Wenyao Zhang 2,

Chenghao Gu 4,‡, Heng Li 1, Ping Tan 1,†

1The Hong Kong University of Science and Technology, 2Tencent Robotics X, 3Tsinghua University

‡Work done during an internship at Tencent Robotics X, †Corresponding author

World Action Models (WAMs) present a promising paradigm for robotic control via video prediction. However, current WAMs suffer from fundamental spatial bottlenecks: standard text inputs introduce referential ambiguity in cluttered scenes, while unstructured RGB predictions lack semantic grounding and remain biased by task-irrelevant backgrounds. To overcome these limitations, we introduce MaskWAM, an object-centric world-action model. By jointly integrating masks as both explicit inputs and predictions via a unified Mixture of Transformers (MoT), MaskWAM unlocks robust policy generalization. This design provides two key benefits: (1) predicting future masks yields object-centric semantic supervision that suppresses visual noise, significantly enhancing even standard text-conditioned WAMs; and (2) coupling this predictive supervision with first-frame visual prompts, such as target object masks, establishes a precise spatial anchor that substantially reduces language ambiguity. Crucially, as WAMs are inherently vision-driven architectures, direct mask conditioning yields substantially stronger guidance than text alone, establishing a precise and robust paradigm for manipulating unseen objects. Evaluations on LIBERO, RoboTwin, and real-world tasks demonstrate that MaskWAM significantly outperforms baselines in both language-clear and language-ambiguous tasks.

Date: June 12, 2026

Project Page: https://hanyangyu1021.github.io/maskwam.github.io/

Keywords: World-Action Models, Mask Visual Prompting, Object-Centric Manipulation

## 1 Introduction

World Action Models (WAMs) [4, 23, 26, 40, 59, 62] have emerged as a promising alternative to Vision-Language-Action (VLA) policies [6, 7, 31, 35, 44, 46, 53, 61] by coupling action generation with future video prediction. Rather than directly mapping observations and language to actions, WAMs treat the modeling of future observations as a powerful proxy task [59]. This paradigm is highly appealing because it fundamentally enhances the underlying representation of the model. By capturing physical dynamics and task-relevant temporal structures, these predictive representations more effectively utilize spatio-temporal priors, thereby facilitating better transfer to downstream control tasks.

However, deploying WAMs in real-world tasks presents two critical challenges in target representation, specifically regarding how the model encodes the target object and its spatial location: (1) Lack of explicit grounding in pure visual predictions. While forecasting RGB frames yields implicit spatial priors, visual reconstruction supervision remains weakly structured, treating all regions equally rather than prioritizing task-relevant objects. Consequently, the target can remain entangled with background clutter, degrading policy performance. To achieve reliable target isolation, the policy therefore requires auxiliary, explicit semantic supervision. (2) Inherent descriptive limits of textual conditioning. Although textual instructions provide essential semantic context, they lack sufficient precision for exact spatial anchoring. Language is inherently limited in describing complex spatial relationships or identifying the intended object among visually similar distractors, as illustrated in Fig. 1, leading to suboptimal policy guidance.

To overcome these spatial bottlenecks, we propose MaskWAM, an end-to-end world-action model that achieves precise control by unifying future mask prediction and visual prompting. This unified framework directly addresses the aforementioned challenges through two synergistic mechanisms. (1) Semantic supervision via mask prediction: To tackle the lack of explicit grounding, MaskWAM predicts future masks alongside RGB frames. This provides the necessary highly semantic supervision signal to abstract away environmental noise and force the policy to inherently prioritize task-relevant regions. (2) Spatial anchoring via visual prompting: To overcome the descriptive limits of textual conditioning, MaskWAM integrates first-frame visual prompts such as target masks. This establishes a precise spatial anchor that substantially reduces referential ambiguity, providing reliable guidance for the policy even in severely cluttered scenes.

![](images/6e07a4cf0650bba44f60e6aac798c736d6fdd7aaf53919c680ac5f9404137e0a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["observation"] --> B["instruction"]
  B --> C["open the drawer and put the pen into it"]
  C --> D["give me the red bottle at ..."]
  D --> E["ambiguous"]
  E --> F["VLA / WAM text-prompt only"]
  F --> G["unconstrained attention"]
  G --> H["RGB"]
  H --> I["?"]
  I --> J["Language clear"]
    
  K["observation"] --> L["zero-pad mask"]
  L --> M["open the drawer and put the pen into it"]
  M --> N["first-frame mask"]
  N --> O["give me the red bottle at ..."]
  O --> P["clear"]
  P --> Q["MaskWAM text-prompt / visual-prompt"]
  Q --> R["focused attention"]
  R --> S["mask"]
  S --> T["precise control"]
    
  U["simulation performance"] --> V["LIBERO"]
  V --> W["Robotwin 2.0"]
  W --> X["Success Rate (%)"]
  X --> Y["Overall: 51.7% for λγ-mask, 32.6% for FastWAM-coord, 17.4% for λγ-coord, 84.9% for MaskWAM (Ours)"]
    
  Y --> Z["In distribution"]
  Y --> AA["Distractors"]
  Y --> AB["Novel Instances"]
  Y --> AC["Lighting"]
    
  AD["Real-World Tasks"] --> AE["Higher Precision"]
  AD --> AF["Better Generalization"]
```
</details>

Figure 1 Overview. We introduce MaskWAM, an end-to-end WAM that unifies mask prompting and mask prediction, supporting both text prompts and visual prompts. MaskWAM achieves high-precision manipulation and strong generalization in both simulation and real-world tasks.

Specifically, our architecture processes the current RGB alongside an optional visual prompt anchored exclusively on the first frame of the episode. A frozen Video VAE [48] compresses these inputs into latents, which are combined with noisy future latents, robot proprioception, and T5 language embeddings [42]. This unified stream is processed by a Mixture of Transformers (MoT) [29] to jointly predict future RGB, masks, and actions via flow-matching. Crucially, explicitly predicting future masks prevents representational shortcuts, forcing the model to both actively attend to the initial visual anchor and prioritize task-relevant regions. Furthermore, since WAMs are inherently vision-driven, direct mask conditioning aligns naturally with their visual prediction process. Our experiments further show that this design provides more effective fine-grained guidance than the evaluated visual-prompted VLA baseline and text-conditioned WAMs or VLAs.

In summary, our main contributions are as follows: (1) We propose MaskWAM, a world-action model that integrates masks as explicit inputs and outputs, jointly modeling future RGB frames, task-relevant masks, and action chunks within a unified architecture. (2) We demonstrate the dual benefits of this design: predicting masks acts as a powerful semantic regularizer steering attention to target regions, while first-frame visual prompting provides a precise spatial anchor for target disambiguation, driving robust generalization across unseen objects. (3) Comprehensive evaluations show that MaskWAM achieves state-of-the-art success rates of 98.4% on LIBERO and 92.2% on RoboTwin. In real-world deployments, it attains 84.3% on languageclear tasks and 84.9% on challenging language-ambiguous scenarios, outperforming the strongest baseline by a 33.2%.

## 2 Related Work

Video World Models for Robotic Manipulation. Recent manipulation systems increasingly move beyond direct action regression toward models that explicitly predict future observations alongside future actions [2, 9–11, 13, 14, 21, 51, 52, 58, 61, 63–66]. Video-diffusion world models leverage large-scale generative priors to forecast plausible futures, providing richer supervision than behavioral cloning and a natural mechanism for reasoning about consequences before acting. However, most WAM formulations [4, 18, 23, 25, 28, 30, 37, 40, 56, 67] still supervise the visual future mainly through RGB reconstruction. This leaves a representation gap between high-dimensional pixels and the lower-dimensional spatial structure that governs control. MaskWAM addresses this gap by adding future mask prediction for task-relevant regions, jointly trained with RGB futures and actions.

Visual Prompting and Intermediate Representations for Robot Policies. To achieve robust spatial grounding, existing approaches either utilize visual cues as static inputs or predict them as intermediate representations. The former approach conditions policies on visual prompts (e.g., point clicks, bounding boxes, masks) to resolve referential ambiguity and explicitly isolate task-relevant objects [3, 15, 19, 27, 33, 34, 43, 45, 55, 57]. Concurrently, the latter integrates intermediate representations like keypoints [16, 20, 49, 60], trajectories [3, 17, 24], masks [36], object flows [50, 54], or affordance maps [1, 22, 38, 39] to bridge the gap between raw vision and continuous control. Unlike methods requiring per-frame conditional masks [27] or treating them as isolated targets [36], MaskWAM unifies both paradigms. By combining first-frame visual conditioning with future mask prediction, it enforces explicit semantic supervision and steers attention toward the prompted target for robust spatial grounding.

## 3 Method

Overview. Given a current RGB observation $\mathbf { I } _ { 0 } \in \mathbb { R } ^ { 3 \times H \times W }$ , a proprioceptive state ${ \bf s } _ { 0 } \in \mathbb { R } ^ { D }$ , a language instruction ℓ, and an optional first-frame target mask $\mathbf { M } _ { 0 } \in \{ 0 , 1 \} ^ { \tilde { H } \times \tilde { W } }$ , MaskWAM jointly predicts an action chunk $\mathbf { a } _ { 1 : K } \in \mathbb { R } ^ { K \times D }$ , future RGB frames $\mathbf { I } _ { 1 : T } \in \mathbb { R } ^ { T \times 3 \times H \times W }$ , and future masks $\mathbf { M } _ { 1 : T }$ . Formally, the model parameterizes the joint distribution

$$
p _ {\theta} \left(\mathbf {a} _ {1: K}, \mathbf {I} _ {1: T}, \mathbf {M} _ {1: T} \mid \mathbf {I} _ {0}, \mathbf {M} _ {0}, \mathbf {s} _ {0}, \ell\right), \tag {1}
$$

where K is the action chunk size and D is the action dimension. Built on Wan 2.2 [48], MaskWAM extends conventional world-action models beyond RGB-space future prediction by introducing masks as explicit spatial representations of task-relevant regions. The action branch is implemented as a Mixture of Transformers (MoT) action expert, allowing video and action tokens to interact through joint attention as in Fig. 2.

## 3.1 Future Masks as Predictive Targets

Unified RGB and mask encoding. To use masks as prediction targets while preserving the visual priors of the pretrained video backbone, we represent each mask as an RGB-compatible image rather than introducing a separate mask-specific encoder. Specifically, a task-relevant mask is rendered as a three-channel image with a fixed color palette, while the background is assigned a separate color. The rendered mask sequence has the same temporal horizon and spatial resolution as the corresponding RGB observations. Both RGB frames and rendered mask frames are then encoded by the same causal 3D VAE encoder E, producing RGB latents $\mathbf { z } _ { v } \in \mathbb { R } ^ { C \times L \times H ^ { \prime } \times W ^ { \prime } }$ and mask latents $\mathbf { z } _ { m } \in \mathbb { R } ^ { C \times L \times H ^ { \prime } \times W ^ { \prime } }$ , where $C , L , H ^ { \prime }$ , and $W ^ { \prime }$ denote the latent channel size, temporal length, spatial height, and spatial width, respectively.

The RGB and mask latents are concatenated along the channel dimension before being fed into the diffusion backbone:

$$
\mathbf {z} = \left[ \mathbf {z} _ {v}; \mathbf {z} _ {m} \right] \in \mathbb {R} ^ {2 C \times L \times H ^ {\prime} \times W ^ {\prime}}. \tag {2}
$$

This design aligns mask information with the pretrained visual latent space and allows each latent position to encode both appearance and task-relevant spatial guidance. By treating masks as RGB-compatible inputs, MaskWAM can reuse the original VAE and transformer interface with minimal architectural changes, while still providing an explicit spatial representation for future prediction and action generation.

Patch embedding expansion. To process concatenated RGB and mask latents, we expand the input patch embedding of the pretrained video backbone from C to 2C input channels. The original visual channels inherit the pretrained weights, while the added mask channels are initialized to zero. This preserves the initial behavior of the pretrained model and allows mask information to be gradually incorporated during fine-tuning. When mask prediction is enabled, the output head is also expanded from C to 2C channels to predict both RGB and mask latent velocities.

![](images/804da872497404308d587d2b98b34d232ddfd686b577e49bff3385db891efb6f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Training data"] --> B["VAE Encoder"]
  B --> C["Obs"]
  C --> D["Img latent"]
  C --> E["Noisy latent"]
  C --> F["Noise"]
  D --> G["Mask latent"]
  E --> H["Noisy latent"]
  F --> I["Noisy latent"]
  G --> J["2C"]
  H --> J
  I --> J
  J --> K["N x H x W x 2C"]
  K --> L["DiT Block"]
  L --> M["x N"]
  M --> N["denoised future latent"]
  N --> O["VAE Decoder"]
  O --> P["Future RGBs"]
  P --> Q["Future Masks"]
  Q --> R["Actions"]
  R --> S["Action Noise"]
  S --> T["τv=1.0, τa=0.0, random"]
  T --> U["Visual prompt (optional)"]
  U --> V["Pick up the grey bottle in the lower right corner of the box"]
```
</details>

![](images/c266e78b876c43aac723caf03a40a9f922c9ba5ac23159d84586c7e970c77992.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph b_Attention_mask["\"(b) Attention mask\""]
  A1["Input: z0 to ai"] --> B1["Image: grid of squares"]
  A2["Input: z1 to ai"] --> B2["Image: grid of squares"]
  A3["Input: z2 to ai"] --> B3["Image: grid of squares"]
  A4["Input: ai"] --> B4["Image: grid of squares"]
  B1 --> C1["VAE Encoder"]
  B2 --> C2["VAE Encoder"]
  B3 --> C3["VAE Encoder"]
  C1 --> D1["text instruction"]
  C2 --> D2["Text Instruction"]
  C3 --> D3["Text Instruction"]
  D1 --> E1["RGB-Mask"]
  D2 --> E2["KV-cache"]
  D3 --> E3["Action"]
  E1 --> F1["Action Noise"]
  E2 --> F2["Action Noise"]
  E3 --> F3["Action Noise"]
  F1 --> G1["Action"]
  F2 --> G2["Action"]
  F3 --> G3["Action"]
    end

    subgraph c_Inference_pipeline["\"(c) Inference pipeline\""]
  H1["Input: z0 to ai"] --> I1["Image: grid of squares"]
  H2["Input: z1 to ai"] --> I2["Image: grid of squares"]
  H3["Input: z2 to ai"] --> I3["Image: grid of squares"]
  H4["Input: ai"] --> I4["Image: grid of squares"]
  I1 --> J1["Text instruction"]
  I2 --> J2["Text instruction"]
  I3 --> J3["Text instruction"]
  J1 --> K1["RGB-Mask"]
  J2 --> K2["KV-cache"]
  K1 --> L1["Action Noise"]
  K2 --> L2["Action Noise"]
  L1 --> M1["Action"]
  L2 --> M2["Action"]
  M1 --> N1["Actions"]
  M2 --> N2["Actions"]
    end

    subgraph d_Inference_pipeline["\"(d) Inference pipeline\""]
  O1["Input: z0 to ai"] --> P1["Image: grid of squares"]
  O2["Input: z1 to ai"] --> P2["Image: grid of squares"]
  O3["Input: z2 to ai"] --> P3["Image: grid of squares"]
  O4["Input: ai"] --> P4["Image: grid of squares"]
  P1 --> Q1["Text instruction"]
  P2 --> Q2["Text instruction"]
  P3 --> Q3["Text instruction"]
  Q1 --> R1["RGB-Mask"]
  Q2 --> R2["KV-cache"]
  R1 --> S1["Action Noise"]
  R2 --> S2["Action Noise"]
  S1 --> T1["Tv"]
  S2 --> T2["Ta"]
  T1 --> U1["Actions"]
  T2 --> U2["Actions"]
    end
```
</details>

Figure 2 MaskWAM architecture. $( a )$ Training: Noisy RGB and mask latents are channel-concatenated and denoised by a unified DiT. The model jointly optimizes future RGB, future masks, and action chunking under a joint flow-matching framework with decoupled noise schedules $\tau _ { v }$ and $\tau _ { a } . \quad ( b )$ Attention mask: A block-wise causal attention mask enables unified RGB, mask, and action training. (c) Inference: Conditioned on optional first-frame masks to resolve spatial ambiguities, the model leverages KV-caching to efficiently generate actions from partially denoised latents.

## 3.2 Initial Masks as Policy Conditions

At deployment, ${ { \bf { M } } _ { 0 } }$ acts as an optional first-frame anchor that can be obtained from a text phrase, click, bounding box, or coarse mask via a segmentation model such as SAM-3 [8]. This anchor designates the target object or contact region in language-ambiguous scenes, such as duplicate objects or sub-object manipulation. MaskWAM does not require a prompt to benefit from future mask prediction, but the correspondence between the first-frame mask and the predicted mask future makes visual prompting a natural application: the prompt initializes the mask channel at the first frame, and the world model propagates that target specification into future RGB, future masks, and actions.

Unified Prompting via Mask Dropout. To enable a single policy to seamlessly handle both language-clear tasks and language-ambiguous tasks, we employ an input mask dropout strategy during training. Specifically, the initial target mask ${ { \bf { M } } _ { 0 } }$ is replaced with a zero tensor with probability $p = 0 . 5$ . This unified training paradigm ensures perfect compatibility across different instruction regimes. Consequently, at deployment, the exact same model can dynamically incorporate an initial mask for precise target disambiguation, or rely exclusively on standard RGB and language inputs when the text instruction is inherently unambiguous.

## 3.3 Training Objective: Joint Flow Matching

Building upon the fused RGB and mask latent stream, MaskWAM utilizes a MoT architecture comprising two interacting branches: a visual branch that jointly denoises the visual and mask representations, and a lightweight action expert dedicated to denoising action chunks. Language features, extracted via a frozen T5 text encoder, are injected into the visual branch through cross attention. Concurrently, the proprioceptive state and noisy actions are processed through projection layers to serve as inputs for the action expert. This expert subsequently utilizes joint attention mechanisms over the visual context provided by the visual branch to predict action velocities.

Consequently, we train MaskWAM end to end using a unified flow matching objective that decouples the noise scheduling across two distinct domains. First, within the visual domain, the RGB and mask streams are synchronized under a shared visual timestep $\tau _ { v } .$ This ensures that visual appearance and structural masks remain spatiotemporally aligned throughout the generative process. Second, within the action domain, we sample an independent timestep $\tau _ { a } .$ . This decoupling forces the action expert to perform trajectory denoising conditioned upon visual contexts at various noise levels. Crucially, this temporal independence establishes the theoretical foundation for the partial denoising inference strategy similar to [40]. The overall training objective is formulated as the sum of the individual flow matching losses:

$$
\mathcal {L} = \mathcal {L} _ {\text { video }} + \mathcal {L} _ {\text { mask }} + \mathcal {L} _ {\text { act }}, \tag {3}
$$

where $L _ { \mathrm { v i d e o } } , \ L _ { \mathrm { m a s k } }$ , and $L _ { \mathrm { a c t } }$ denote the flow-matching losses for RGB videos, mask representations, and action trajectories, respectively.

Implementation Details. During training, the Video VAE and text encoder remain frozen. The architecture processes $3 8 4 \times 3 2 0$ RGB frames (T-shape) with a prediction horizon of $T = 8$ . At deployment, the initial spatial anchor ${ { \bf { M } } _ { 0 } }$ is generated via SAM3 [8] only once at the start of the episode, eliminating the need for real-time tracking and repeated visual prompting during subsequent predictions. To minimize latency, MaskWAM employs a partial-denoising decoding strategy. Rather than fully denoising future frames, the model performs only one denoising step on the joint RGB-mask stream to extract intermediate, task-aware visual latents. The action expert then jointly attends to these partially denoised latents, language instructions, and proprioception to generate the action chunk as in Fig. $2 ( \mathrm { c ) }$ . This avoids the high test-time cost of full video generation while preserving the mask-grounded world state. More details are provided in the appendix.

## 4 Experiments

Benchmarks and Baselines. We evaluate MaskWAM on the LIBERO benchmark [32] against leading baselines, including WorldVLA, GR00T-N1, π0, π0.5, Motus, and FastWAM, to show that auxiliary mask prediction improves policy learning even without test-time visual prompts. To assess multitask generalization, we evaluate MaskWAM on RoboTwin 2.0 [12] against $\pi _ { 0 }$ and FastWAM under randomized instructions, environments, and object placements.

Performance on LIBERO. As shown in Table 1, MaskWAM establishes a new state-of-the-art with a 98.4% average success rate. It consistently outperforms recent VLAs like $\pi _ { 0 . 5 }$ and advanced WAMs such as Motus and FastWAM. Compared to our RGB-only variant, the auxiliary mask prediction objective lifts performance from 97.3% to 98.4%, showing that mask supervision can enhance base policy learning without visual prompts at deployment. Attention maps in the right panel of Table 1 explain this improvement: while the RGB-only model often highlights spurious backgrounds, mask supervision forces MaskWAM to attend precisely to taskrelevant regions, yielding superior visual grounding and execution robustness.

Performance on RoboTwin 2.0. As shown in Table 2, MaskWAM achieves a state-of-the-art 92.2% average success rate across six randomized RoboTwin 2.0 tasks, outperforming $\pi _ { 0 }$ and FastWAM by 19.4% and 4.5%, respectively. Ablations highlight the advantage of joint prediction: the Mask-only variant (88.8%) natively outperforms RGB-only (87.3%), but unifying both modalities maximizes performance to 92.2%. This confirms that while RGB futures capture essential dynamics, auxiliary mask futures force the model to focus on taskrelevant regions. Notably, the good performance of our Mask-only ablation conceptually aligns with the findings of the concurrent Mask World Model [55], while our full MaskWAM further demonstrates that our joint representation achieves the best performance.

## 4.1 Evaluation in Real-world Tasks

Tasks and Evaluation Protocol. To evaluate the capability and generalizability of MaskWAM, we design eight challenging real-world manipulation tasks. As illustrated in Figure 3, these are categorized into standard language-clear settings (Tasks 1-4) and language-ambiguous settings requiring explicit spatial prompting (Tasks 5-8). For data collection, we gather an average of 100 human demonstrations per task. For languageclear tasks, each model is evaluated over 100 trials per task. For language-ambiguous generalization, each model is evaluated over 60 trials per task under each generalization setting. Detailed descriptions for each task and their environmental setups are deferred to the Appendix.

Table 1 Success rates (%) on the LIBERO benchmark. The right panel visualizes action-to-video attention maps of RGB-based WAM and MaskWAM, showing that mask supervision encourages the model to attend more consistently to task-relevant regions.

<table><tr><td>Method</td><td>Type</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg</td></tr><tr><td>WorldVLA [10]</td><td>VLA</td><td>87.6</td><td>96.2</td><td>83.4</td><td>60.0</td><td>81.8</td></tr><tr><td>GR00T-N1 [5]</td><td>VLA</td><td>94.4</td><td>97.6</td><td>93.0</td><td>90.6</td><td>93.9</td></tr><tr><td> $\pi_0$  [6]</td><td>VLA</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.1</td></tr><tr><td> $\pi_{0.5}$  [7]</td><td>VLA</td><td>98.6</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.8</td></tr><tr><td>Motus [4]</td><td>WAM</td><td>96.8</td><td>99.8</td><td>96.6</td><td>97.6</td><td>97.7</td></tr><tr><td>FastWAM [59]</td><td>WAM</td><td>98.2</td><td>100.0</td><td>97.0</td><td>95.2</td><td>97.6</td></tr><tr><td>Ours (RGB-only)</td><td>WAM</td><td>96.8</td><td>99.6</td><td>97.0</td><td>95.8</td><td>97.3</td></tr><tr><td>Ours (Mask-only)</td><td>WAM</td><td>97.2</td><td>99.8</td><td>97.4</td><td>96.0</td><td>97.6</td></tr><tr><td>Ours</td><td>WAM</td><td>98.8</td><td>100.0</td><td>98.2</td><td>96.4</td><td>98.4</td></tr></table>

![](images/7e75c87133aeadb6a88853d221f2238e1a64d6a1b99927f954463573717d87e3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["RGB-only WAM"] --> B["Output 1"]
  A --> C["Output 2"]
  A --> D["Output 3"]
  E["MaskWAM"] --> F["Output 1"]
  E --> G["Output 2"]
  E --> H["Output 3"]
  I["Output 4"] --> J["Output 5"]
  K["Output 6"] --> L["Output 7"]
  M["Output 8"] --> N["Output 9"]
```
</details>

Table 2 Success rates (%) on the RoboTwin 2.0 benchmark. We use 500 randomized episodes per task for training. RGB-only denotes MaskWAM with only RGB prediction, and Mask-only denotes MaskWAM with only mask prediction.

<table><tr><td>Method</td><td>Hammer</td><td>Bell</td><td>Card</td><td>Burger</td><td>Stand</td><td>Shoe</td><td>Average</td></tr><tr><td> $\pi_0$  [6]</td><td>68</td><td>72</td><td>81</td><td>79</td><td>63</td><td>74</td><td>72.8</td></tr><tr><td>FastWAM [59]</td><td>83</td><td>87</td><td>92</td><td> $\underline{94}$ </td><td>80</td><td>90</td><td>87.7</td></tr><tr><td>Ours (RGB-only)</td><td>82</td><td>87</td><td>91</td><td>93</td><td>79</td><td> $\underline{92}$ </td><td>87.3</td></tr><tr><td>Ours (Mask-only)</td><td> $\underline{85}$ </td><td> $\underline{90}$ </td><td> $\underline{93}$ </td><td>93</td><td> $\underline{81}$ </td><td>91</td><td> $\underline{88.8}$ </td></tr><tr><td>Ours</td><td>88</td><td>93</td><td>95</td><td>97</td><td>85</td><td>95</td><td>92.2</td></tr></table>

Performance on Language-clear Tasks. To evaluate standard in-distribution performance, we first test the policies on the four fundamental text-conditioned tasks. As shown in Table 3, MaskWAM achieves highly competitive performance, surpassing state-of-the-art VLA models like $\pi _ { 0 }$ and $\pi _ { 0 . 5 } ,$ , as well as prior WAM architectures such as FastWAM. Crucially, compared to our own RGB-only variant, integrating future mask prediction consistently improves success rates across the evaluated tasks, lifting performance from 86% to 91% on Task 1 and from 76% to 81% on Task 3. This indicates that jointly predicting masks does not compromise the base language-clear policy, but rather stabilizes and enhances it by providing explicit semantic grounding.

Table 3 Real-robot evaluation on language-clear tasks. Success rates are reported in %. We collect an average of 100 demonstrations per task, and each model is evaluated over 100 trials per task with diverse object placements.

<table><tr><td>Method</td><td>Type</td><td>Task 1</td><td>Task 2</td><td>Task 3</td><td>Task 4</td><td>Avg</td></tr><tr><td> $\pi_0$  [6]</td><td>VLA</td><td>57</td><td>54</td><td>54</td><td>58</td><td>55.8</td></tr><tr><td> $\pi_{0.5}$  [7]</td><td>VLA</td><td>83</td><td>55</td><td>74</td><td>77</td><td>72.3</td></tr><tr><td>FastWAM [59]</td><td>WAM</td><td>88</td><td>76</td><td>77</td><td>75</td><td>79.0</td></tr><tr><td>Ours (RGB-only)</td><td>WAM</td><td>86</td><td>77</td><td>76</td><td>78</td><td>79.3</td></tr><tr><td>Ours</td><td>WAM</td><td>91</td><td>82</td><td>81</td><td>83</td><td>84.3</td></tr></table>

Performance on Language-ambiguous Tasks. To evaluate how explicit spatial prompting resolves target uncertainty, we conduct a comprehensive assessment detailed in Fig. 4, spanning four distinct settings: one foundational in-distribution evaluation and three demanding zero-shot generalization axes. (1) In-Distribution: Establishing foundational performance, MaskWAM achieves a 92.9% average success rate. Notably, when utilizing textual coordinate prompts, the language-driven π0-coord outperforms the visiondriven FastWAM-coord with a 41.7% success rate compared to 26.3%. This performance inversion supports our hypothesis that spatial disambiguation in WAMs is better injected natively via the visual modality of masks rather than text.

Building upon this robust visual grounding, we evaluate zero-shot generalization across the remaining three axes. (2) Distractors: Introducing unseen, task-irrelevant objects to test spatial grounding amidst visual clutter, MaskWAM maintains a 90.4% success rate with minimal degradation, substantially outperforming the π0-mask baseline at 52.9%. (3) Novel Instances: Replacing the training target with unseen objects of varying geometries or related categories (e.g., substituting a trained bowl with a novel cup), MaskWAM achieves a

![](images/8ebec690b045a0da032ca1e8ab8e3858f8ecfd92cdf118818c883393a539e059.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Task 1: stack three green bowls to the middle."] --> B["Task 2: pick up the cup and place it on the peg."]
  B --> C["Task 3: open the drawer, put the pen into it and then close it."]
  C --> D["Task 4: fold the blue towel into a square."]
  D --> E["Task 5: grasp the white bowl"]
  E --> F["Task 6: pick up the yellow cup"]
  F --> G["Task 7: pick and place the bottle"]
  G --> H["Task 8: grasp the Cosmetics"]
```
</details>

Figure 3 Real-robot tasks. Tasks 1-4 indicate standard language-clear cases, while Tasks 5-8 introduce languageambiguous scenarios.

74.6% success rate, significantly surpassing the 44.6% achieved by π0-mask and demonstrating robust categorylevel structural skill transfer. (4) Lighting: Altering environmental illumination to test robustness against appearance shifts, MaskWAM exhibits strong invariance with an 81.7% success rate. Across all settings, our mask-prediction mechanism consistently outperforms both visual-prompt VLA and textual coordinate baselines, confirming its efficacy in resolving language ambiguity and enhancing the visual generalization. Qualitative visualizations are provided in the appendix.

![](images/82046f08d714ec5cfc0cc1c30e22b77f5b310563a6a5b9d072862f2e3503d8f7.jpg)

<details>
<summary>bar chart</summary>

| Category | π₀-mask (%) | π₀-coord (%) | FastWAM-coord (%) | MaskWAM (Ours) (%) |
| :--- | :--- | :--- | :--- | :--- |
| In Distribution | 63 | 42 | 26 | 93 |
| Distractors | 53 | 30 | 11 | 90 |
| Novel Instances | 45 | 25 | 13 | 74 |
| Lighting | 46 | 33 | 19 | 81 |
| Overall | 51.7 | 32.6 | 17.4 | 84.9 |
Out-of-Distribution (Generalization)
</details>

Figure 4 Generalization performance under language ambiguity with explicit spatial prompting. We evaluate policy robustness in scenes where language instructions alone are insufficient to specify the target, demonstrating how precise spatial cues (e.g., visual masks or textual coordinate prompts) resolve target uncertainty.

## 4.2 Ablations

To validate the core architectural designs of MaskWAM, we perform controlled ablations across both simulated and real-world environments. We investigate the necessity of joint modality prediction and explicit visual prompting by addressing three critical questions:

Q1: Is paired RGB and mask prediction essential for robust policy learning? Evaluated on LIBERO (Table 1), we compare our full MaskWAM against single-objective variants. While isolated RGBonly (97.3%) and Mask-only (97.6%) models perform well, our joint model attains a SOTA 98.4%. This synergy is critical for spatial precision, yielding absolute improvements of +2.0% (Spatial ) and +1.2% (Goal ) over the RGB-only baseline. Crucially, jointly optimizing both modalities overcomes RGB’s vulnerability to visual noise and the mask’s lack of textural context. By forcing the latent space to better align visual dynamics with object-centric semantics, paired prediction acts as an essential representational regularizer.

Q2: Is future mask prediction necessary for grounding visual prompts? Evaluated in languageambiguous real-world tasks (Table 4), providing a mask prompt without future mask prediction (Ours-nopred) reduces the success rate to 21.6%. This indicates that a visual prompt alone does not guarantee effective grounding: without the auxiliary prediction objective, the model fails to reliably use the visual guidance. In contrast, our full model achieves 84.9%, showing that future mask prediction is essential for grounding policy attention on the visual prompt.

Q3: How do different prompt formulations impact performance? Evaluated in our languageambiguous setup (Table 4), we compare mask conditioning against spatial coordinate text (Ours-coord). While coordinate text struggles with spatial alignment (18.2% success), our mask formulation (84.9%) provides dense spatial-semantic priors to resolve referential ambiguities. As visualized in the right panel of Table 4, the text-coordinate-prompted model suffers from dispersed attention distracted by background clutter. In contrast, our mask-conditioned policy maintains a sharply localized, temporally consistent focus on the target object. Furthermore, ablation studies detailed in the Appendix confirm the model’s robustness against varying input mask qualities.

Table 4 Ablation of mask conditioning and prediction. The left panel isolates first-frame mask conditioning and future mask prediction; Ours-no-pred only removes the future mask target and loss from MaskWAM. The right panel compares coordinate-augmented text prompts with mask-based visual prompts. “Coord. Prompt” denotes coordinate-augmented textual prompting.

<table><tr><td colspan="4">Language-ambiguous Tasks (ID &amp; OOD)</td></tr><tr><td>Setting</td><td>Ours-no-pred</td><td>Ours-coord</td><td>Ours</td></tr><tr><td>Future Mask</td><td>-</td><td>√</td><td>√</td></tr><tr><td>Mask Prompt</td><td>√</td><td>-</td><td>√</td></tr><tr><td>Coord. Prompt</td><td>-</td><td>√</td><td>-</td></tr><tr><td>Avg. Succ.</td><td>21.6</td><td>18.2</td><td>84.9</td></tr></table>

![](images/6f719520ceb5b91bb18475c3b5ae9e6d21d769dcbea229a44e185ca4b040caa9.jpg)

<details>
<summary>text_image</summary>

RGB
Ours-coord
Ours
t = 0
timestep
</details>

## 5 Conclusion

We introduced MaskWAM, an end-to-end world-action model that augments standard RGB prediction with auxiliary future mask prediction. By integrating mask modeling into the latent space of a unified DiT, MaskWAM encourages the policy to ground its actions in task-critical regions. This formulation improves generalization across distractors, object variations, and lighting changes, while enabling precise target disambiguation through optional first-frame mask prompts. Overall, MaskWAM shows that predicting what matters is just as crucial as predicting what happens for robust robotic manipulation.

## 6 Limitations and Future Work

While MaskWAM demonstrates strong performance, it has several limitations. First, MaskWAM relies on mask supervision during training and segmentation-derived prompts during deployment. Automating reliable mask extraction in cluttered real-world environments remains non-trivial. Second, due to computational constraints, we defer large-scale RGB-mask-action pretraining to future work, which holds significant potential for further enhancing mask-aware visual dynamics and downstream policy robustness.

## References

[1] Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, and Deepak Pathak. Affordances from human videos as a versatile representation for robotics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13778–13790, 2023.

[2] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. arXiv preprint arXiv:2409.16283, 2024.  
[3] Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In European Conference on Computer Vision, pages 306–324. Springer, 2024.  
[4] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model. 2025.  
[5] Johannes Bjorck et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint, 2025.  
[6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.  
[7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0.5: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.  
[8] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.  
[9] Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Xin Li, Hao Luo, Fan Wang, Fan Wang, and Deli Zhao. Rynnvla-002: A unified vision-language-action and world model. arXiv preprint arXiv:2511.17502, 2025.  
[10] Jun Cen, Zhihao Li, Yuze Hu, Ange Yao, Yichun Yang, Junran Peng, and Ruizhen Xu. Worldvla: Towards autoregressive action model with world knowledge. arXiv preprint, 2025.  
[11] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.  
[12] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.  
[13] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation, 2023. URL https://arxiv.org/ abs/2302.00111.  
[14] Yao Feng, Hengkai Tan, Xinyi Mao, Chendong Xiang, Guodong Liu, Shuhe Huang, Hang Su, and Jun Zhu. Vidar: Embodied video diffusion model for generalist manipulation, 2025. URL https://arxiv.org/abs/2507.12898.  
[15] Jiayuan Gu, Sean Kirmani, Paul Wohlhart, Yao Lu, Montserrat Gonzalez Arenas, Kanishka Rao, Wenhao Yu, Chuyuan Fu, Keerthana Gopalakrishnan, Zhuo Xu, Priya Sundaresan, Peng Xu, Hao Su, Karol Hausman, Chelsea Finn, Quan Vuong, and Ted Xiao. Rt-trajectory: Robotic task generalization via hindsight trajectory sketches, 2023.  
[16] Siddhant Haldar and Lerrel Pinto. Point policy: Unifying observations and actions with key points for robot manipulation. arXiv preprint arXiv:2502.20391, 2025.  
[17] Cheng-Chun Hsu, Bowen Wen, Jie Xu, Yashraj Narang, Xiaolong Wang, Yuke Zhu, Joydeep Biswas, and Stan Birchfield. Spot: Se (3) pose trajectory diffusion for object-centric manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 4853–4860. IEEE, 2025.  
[18] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.  
[19] Haifeng Huang, Xinyi Chen, Yilun Chen, Hao Li, Xiaoshen Han, Zehan Wang, Tai Wang, Jiangmiao Pang, and Zhou Zhao. Roboground: Robotic manipulation with grounded vision-language priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22540–22550, 2025.  
[20] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024.  
[21] Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, Loic Magne, Ajay Mandlekar, Avnish Narayan, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Yinzhen Xu, Xiaohui Zeng, Kaiyuan Zheng, Ruijie Zheng, Ming-Yu Liu, Luke Zettlemoyer, Dieter Fox, Jan Kautz, Scott Reed, Yuke Zhu, and Linxi Fan. Dreamgen: Unlocking generalization in robot learning through video world models, 2025. URL https://arxiv.org/abs/2505.12705.  
[22] Yuanchen Ju, Kaizhe Hu, Guowei Zhang, Gu Zhang, Mingrun Jiang, and Huazhe Xu. Robo-abc: Affordance generalization beyond categories via semantic correspondence for robot manipulation. In European Conference on Computer Vision, pages 222–239. Springer, 2024.  
[23] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.  
[24] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.  
[25] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.  
[26] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.  
[27] Puhao Li, Yingying Wu, Ziheng Xi, Wanlin Li, Yuzhe Huang, Zhiyuan Zhang, Yinghan Chen, Jianan Wang, Song-Chun Zhu, Tengyu Liu, et al. Controlvla: Few-shot object-centric adaptation for pre-trained vision-languageaction models. arXiv preprint arXiv:2506.16211, 2025.  
[28] Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, and Carl Vondrick. Video generators are robot policies. arXiv preprint arXiv:2508.00795, 2025.  
[29] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wentau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.  
[30] Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, Si Liu, Jianlan Luo, Liliang Chen, Shuicheng Yan, Maoqing Yao, and Guanghui Ren. Genie envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025.  
[31] Haitao Lin, Hanyang Yu, Jingshun Huang, He Zhang, Yonggen Ling, Ping Tan, Xiangyang Xue, and Yanwei Fu. Universal pose pretraining for generalizable vision-language-action policies. arXiv preprint arXiv:2602.19710, 2026.  
[32] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776– 44791, 2023.  
[33] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. Moka: Open-world robotic manipulation through mark-based visual prompting. In Robotics: Science and Systems, 2024.  
[34] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. Moka: Open-world robotic manipulation through mark-based visual prompting. arXiv preprint arXiv:2403.03174, 2024.  
[35] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.  
[36] Yunfan Lou, Xiaowei Chi, Xiaojie Zhang, Zezhong Qian, Chengxuan Li, Rongyu Zhang, Yaoxu Lyu, Guoyu Song, Chuyao Fu, Haoxuan Xu, et al. Mask world model: Predicting what matters for robust robot policy learning. arXiv preprint arXiv:2604.19683, 2026.  
[37] Teli Ma, Jia Zheng, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang, and Shuo Yang. Dit4dit: Jointly modeling video dynamics and actions for generalizable robot control. arXiv preprint arXiv:2603.10448, 2026.  
[38] Tushar Nagarajan, Christoph Feichtenhofer, and Kristen Grauman. Grounded human-object interaction hotspots  
from video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8688–8697, 2019.  
[39] Soroush Nasiriany, Sean Kirmani, Tianli Ding, Laura Smith, Yuke Zhu, Danny Driess, Dorsa Sadigh, and Ted Xiao. Rt-affordance: Affordances are versatile intermediate representations for robot manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8249–8257. IEEE, 2025.  
[40] Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692, 2025.  
[41] Qwen Team. Qwen3-vl: A frontier multimodal large language model. https://github.com/QwenLM/Qwen3-VL, 2025. Accessed: 2026-01-22.  
[42] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.  
[43] Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, Chelsea Finn, and Karol Hausman. Open-world object manipulation using pre-trained vision-language models. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 3397–3417. PMLR, 2023.  
[44] Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. Vla-jepa: Enhancing vision-language-action model with latent world model. arXiv preprint arXiv:2602.10098, 2026.  
[45] Priya Sundaresan, Suneel Belkhale, Dorsa Sadigh, and Jeannette Bohg. Kite: Keypoint-conditioned policies for semantic manipulation, 2023.  
[46] HY Team, Xumin Yu, Zuyan Liu, Ziyi Wang, He Zhang, Yongming Rao, Fangfu Liu, Yani Zhang, Ruowen Zhao, Oran Wang, et al. Hy-embodied-0.5: Embodied foundation models for real-world agents. arXiv preprint arXiv:2604.07430, 2026.  
[47] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.  
[48] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.  
[49] Shengjie Wang, Jiacheng You, Yihang Hu, Jiongye Li, and Yang Gao. Skil: Semantic keypoint imitation learning for generalizable data-efficient manipulation. arXiv preprint arXiv:2501.14400, 2025.  
[50] Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel. Any-point trajectory modeling for policy learning. arXiv preprint arXiv:2401.00025, 2023.  
[51] John Won, Kyungmin Lee, Huiwon Jang, Dongyoung Kim, and Jinwoo Shin. Dual-stream diffusion for worldmodel augmented vision-language-action model, 2025. URL https://arxiv.org/abs/2510.27607.  
[52] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation, 2023.  
[53] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al. A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692, 2026.  
[54] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. arXiv preprint arXiv:2407.15208, 2024.  
[55] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 2475–2499. PMLR, 2025.  
[56] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, and Joel Jang. World action models are zero-shot  
policies, 2026. URL https://arxiv.org/abs/2602.15922.  
[57] Hang Yu, Juntu Zhao, Yufeng Liu, Kaiyu Li, Cheng Ma, Di Zhang, Yingdong Hu, Guang Chen, Junyuan Xie, Junliang Guo, et al. Point what you mean: Visually grounded instruction policy. arXiv preprint arXiv:2512.18933, 2025.  
[58] Hanyang Yu, Xiaoxiao Long, and Ping Tan. Lm-gaussian: Boost sparse-view 3d gaussian splatting with large model priors. arXiv preprint arXiv:2409.03456, 2024.  
[59] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.  
[60] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721, 2024.  
[61] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, and Xin Jin. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. CoRR, abs/2507.04447, 2025. doi: 10.48550/ARXIV.2507.04447. URL https://doi.org/10.48550/arXiv.2507.04447.  
[62] Wenyao Zhang, Bozhou Zhang, Zekun Qi, Wenjun Zeng, Xin Jin, and Li Zhang. Disentangled robot learning via separate forward and inverse dynamics pretraining. arXiv preprint arXiv:2604.16391, 2026.  
[63] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models, 2025. URL https://arxiv.org/abs/2503. 22020.  
[64] Ruijie Zheng, Jing Wang, Scott Reed, Johan Bjorck, Yu Fang, Fengyuan Hu, Joel Jang, Kaushil Kundalia, Zongyu Lin, Loic Magne, Avnish Narayan, You Liang Tan, Guanzhi Wang, Qi Wang, Jiannan Xiang, Yinzhen Xu, Seonghyeon Ye, Jan Kautz, Furong Huang, Yuke Zhu, and Linxi Fan. Flare: Robot learning with implicit world modeling, 2025. URL https://arxiv.org/abs/2505.15659.  
[65] Pengfei Zhou, Liliang Chen, Shengcong Chen, Di Chen, Wenzhi Zhao, Rongjun Jin, Guanghui Ren, and Jianlan Luo. Act2goal: From world model to general goal-conditioned policy, 2025. URL https://arxiv.org/abs/2512. 23541.  
[66] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024.  
[67] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets, 2025. URL https: //arxiv.org/abs/2504.02792.

## Appendix

## A Details about Real-world Experiments

## A.1 Real-world Hardware Setup.

We evaluate our model using a Dual-arm Xtrainer robotic platform, as shown in Figure 5. For visual perception, we employ a multi-camera configuration: a RealSense D455 depth camera serves as the head-mounted Eye-on-Base sensor for global scene understanding, while a RealSense D405 depth camera is integrated as an Eye-on-Hand sensor to provide localized, high-resolution visual feedback.

![](images/ed6c1a9ef363a81232b07762a89900c933c130ec35cf3290f66be61efa7a9b0a.jpg)

<details>
<summary>text_image</summary>

head camera
(D455)
left arm
wrist cameras
(D405)
right arm
</details>

Figure 5 Real-robot platform Our real-world experiment is based on a Dual-arm Xtrainer robotic platform

## A.2 Task settings and evaluation in Real-world Tasks

Task Settings. To evaluate the capability and generalizability of MaskWAM, we design eight representative and challenging real-world manipulation tasks, grouped into language-clear and language-ambiguous settings as shown in Figure 3. The language-clear tasks include: (1) Stacking three green nested bowls; (2) Hanging a mug onto a designated peg on a wooden stand; (3) Long-horizon Interaction, which involves opening a drawer, placing a pen inside, and closing the drawer; and (4) Deformable Object Manipulation, which requires folding a fabric towel. The language-ambiguous tasks include: (5) Grasping Bowl, grasping a specified bowl among three visually similar bowls; (6) Picking up Cup, picking up the specified cup; (7) Picking and Placing Bottle, picking and placing the designated bottle from a small box; and (8) Grasping Cosmetics, grasping a specified cosmetic item from 16 tightly arranged red cosmetic products in a box. We collect an average of 100 demonstrations per task. For language-clear tasks, each model is evaluated over 100 trials per task. For language-ambiguous tasks, each model is evaluated over 60 trials per task. The execution success rate is reported as the primary performance metric.

## A.3 Baseline Details for Language-Ambiguous Tasks

In Section 4.1, we compare MaskWAM with both VLA-based baselines $\left( \pi _ { 0 } \mathbf { - m a s k } , \pi _ { 0 } \mathbf { - c o o r d } \right)$ and WAM-based variants (FastWAM-coord). These baselines examine whether spatial information can be provided through language instructions or visual prompts. We provide the detailed baseline designs below.

For coordinate-text baselines (π0-coord and FastWAM-coord), we first compute the centroid of the targetobject mask and convert it into an explicit normalized pixel coordinate in the language instruction. For example, the command can be written as: “grasp the white bottle whose target center is at $x = 0 . 4 3$ from the left and $y = 0 . 4 0$ from the top in the front view.” We find that such coordinate-based text prompts provide a certain degree of coarse spatial control for $\pi _ { 0 } { \mathrm { - } } { \mathsf { c o o r d } }$ , which achieves reasonable success rates on relatively simple tasks, such as Tasks 5 and 6, where the objects are large and well separated on the table. In contrast, FastWAM-coord remains largely insensitive to these coordinate-augmented instructions, even when such commands are included during training, and therefore achieves relatively low success rates. For more precision-demanding tasks, such as Tasks 7 and 8, both text-prompt baselines fail to complete the task. This suggests that even when precise coordinate information is provided in text, language conditioning remains insufficient for fine-grained manipulation. Moreover, in real-world deployment, such accurate target coordinates are usually unavailable; users can typically provide only coarse descriptions such as “left”, “right”, “front”, or “back”, which further weakens this form of spatial control.

![](images/85097c196daf78a87019c724afe98493c1c3c2b3dd2b593c92b115a8c8ef39f9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image 1"] --> B["Language Annotation"]
  C["Image 2"] --> D["Mask Annotation"]
  E["Image 3"] --> D
  F["Image 4"] --> D
  G["Image 5"] --> D
  H["Image 6"] --> D
  I["Image 7"] --> J["Text Condition"]
  K["Image 8"] --> L["Text Condition"]
  M["Image 9"] --> L
  N["Image 10"] --> O["Text Condition"]
  P["Image 11"] --> O
  Q["Image 12"] --> R["Text Condition"]
  S["Image 13"] --> R
  T["Image 14"] --> U["Text Condition"]
  V["VLM"] --> W["Task-relevant Objects"]
  X["SAM3"] --> Y["Text Condition"]
  Z["Robot"] --> AA["Text Condition"]
  AB["Drawer"] --> AC["Text Condition"]
  AD["pen"] --> AE["Text Condition"]
  AF["Robot"] --> AG["Text Condition"]
  AH["Mouse"] --> AI["Text Condition"]
  AJ["Mouse"] --> AK["Text Condition"]
  AL["Mouse"] --> AM["Text Condition"]
  AN["Mouse"] --> AO["Text Condition"]
  AP["Human Verification"] --> AQ["Labeled Dataset"]
  AR["Point Condition"] --> AS["Human Verification"]
  AT["Mask Sequence (round n)"] --> AU["Labeled Dataset"]
  AV["✓"] --> AW["Labeled Dataset"]
  AX["X"] --> AY["Human Verification"]
  AZ["X"] --> AY
  BA["×"] --> AY
```
</details>

Figure 6 Annotation pipeline. We obtain language labels, initialize SAM3 from task-relevant prompts, and propagate masks with human verification.

We further investigate whether a mask image can serve as a visual prompt for guiding $\pi _ { 0 }$ . In the $\pi _ { 0 }$ -mask variant, the policy receives both the RGB image and the corresponding mask image. For the front view, both inputs are normalized to [−1, 1] and passed through the SigLIP encoder [47] to obtain RGB and mask embeddings, respectively. To encourage the network to use the input mask, we fuse the two embeddings through direct addition:

$$
\mathbf {e} _ {\text { fused }} = \mathbf {e} _ {\text { rgb }} + \mathbf {e} _ {\text { mask }}. \tag {4}
$$

The fused embedding is then fed into the VLM transformer layers. We find that $\pi _ { 0 } .$ -mask performs better than $\pi _ { 0 } \cdot$ -coord, but still falls short on high-precision tasks.

Overall, MaskWAM performs best, followed by $\pi _ { 0 } .$ -mask, $\pi _ { 0 } \cdot$ -coord, and FastWAM-coord. In this comparison, WAM-based models appear less sensitive to coordinate-augmented language prompts, while MaskWAM benefits more from mask-based visual prompting. We attribute this trend to the vision-centric prediction backbone of WAMs, which allows dense visual prompts to align naturally with the visual prediction process.

## B Dataset Annotation and Mask-Prompt Robustness

## B.1 Language and Mask Annotation

We build an automated annotation pipeline as shown in Figure 6. The pipeline differs between language-clear and language-ambiguous tasks to avoid oracle leakage.

Language-clear tasks: Qwen3-VL [41] parses the instruction to identify task-relevant objects, and SAM3 [8] segments and tracks masks across the episode.

Language-ambiguous tasks: Since text alone cannot disambiguate the target, a human annotator provides a point prompt on the first frame. SAM3 propagates this point into a full mask and tracks it across the episode.

![](images/42b5adde1a5463f45ecb24bf6a2caa3ccc62ddd076efcf3d9c3474d4911c6fec.jpg)

<details>
<summary>text_image</summary>

Original
IoU: 1.00
Erosion-S (10%)
IoU: 0.82
Erosion-L (30%)
IoU: 0.40
Dilation-S (10%)
IoU: 0.84
Dilation-M (30%)
IoU: 0.55
Dilation-L (60%)
IoU: 0.35
Shift-S (10%)
IoU: 0.72
Shift-M (30%)
IoU: 0.32
Shift-L (50%)
IoU: 0.10
Dropout (25%)
IoU: 0.70
Dropout (50%)
IoU: 0.50
Dropout (75%)
IoU: 0.23
Original
IoU: 1.00
Erosion-S (20%)
IoU: 0.72
Erosion-L (40%)
IoU: 0.38
Dilation-S (10%)
IoU: 1.00
Dilation-M (30%)
IoU: 0.67
Dilation-L (60%)
IoU: 0.41
Shift-S (10%)
IoU: 0.81
Shift-M (30%)
IoU: 0.42
Shift-L (50%)
IoU: 0.14
Dropout (25%)
IoU: 0.61
Dropout (50%)
IoU: 0.41
Dropout (75%)
IoU: 0.24
</details>

Figure 7 Examples of imperfect first-frame mask prompts. We visualize representative corrupted masks generated from the online SAM3 mask, including erosion, dilation, spatial shift, and region dropout.

All first-round annotations are human-verified. Low-quality episodes are corrected with point prompts on keyframes, and SAM3 re-propagates the corrected masks. This loop repeats until quality is satisfactory. With this pipeline, 91% of episodes require no human correction, and annotating 50 episodes takes approximately 3 minutes. For language-ambiguous tasks, the initial point prompt adds marginal overhead (5–10 seconds per episode).

## B.2 Robustness to Noisy First-frame Mask Prompts

In our real-world experiments, the first-frame prompt mask is generated online by SAM3 [8] at the beginning of each episode. To evaluate whether MaskWAM relies on perfectly clean mask prompts, we further study its sensitivity to imperfect first-frame masks. We keep the trained MaskWAM checkpoint fixed and perturb only the first-frame prompt mask $M _ { 0 }$ during inference. All subsequent observations, language instructions, proprioceptive states, and model weights remain unchanged.

We evaluate two representative real-world language-ambiguous tasks: Task 5, where the target object is relatively large and well separated, and Task 8, where the target object must be selected from densely arranged visually similar cosmetic items. For each task and each mask condition, we run 20 trials with diverse initial object placements. The default condition uses the online SAM3 mask, which is the same setting as our main real-world evaluation. We then apply four types of synthetic corruption to this online mask, as illustrated in Fig. 7. Erosion simulates under-segmentation, dilation simulates over-segmentation, shift simulates spatial mislocalization caused by inaccurate prompting or segmentation, and region dropout simulates partial mask missing caused by occlusion, reflection, or segmentation failure.

As shown in Fig. 8, MaskWAM is robust to moderate first-frame prompt corruption. Starting from the online SAM3 mask used in our main real-world evaluation, mild erosion, dilation, shift, and region dropout only lead to limited performance degradation, especially when the corrupted mask remains sufficiently overlapped with the original target mask. This suggests that MaskWAM does not require a perfectly clean prompt mask, as long as the mask still provides an approximate target identity and spatial anchor. In contrast, performance drops more clearly under severe corruption, such as large spatial shifts or heavy region dropout, where the prompt either becomes mislocalized or loses most of the target area. These results indicate that the first-frame mask primarily serves as a target identity and spatial anchoring signal: MaskWAM tolerates moderate mask noise, but severe target-region corruption weakens object-centric grounding.

![](images/96a613f1a0f29de1f4691db6498c0c2037acc327c170b154013f8955b4719de0.jpg)

<details>
<summary>line chart</summary>

| Mask IoU | Online SAM3 | Erosion | Other |
| -------- | ----------- | ------- | ----- |
| 0.00     | 85.0        | 100.0   | 85.0  |
| 0.25     | 91.0        | 100.0   | 96.0  |
| 0.50     | 92.0        | 100.0   | 98.0  |
| 0.75     | 95.0        | 100.0   | 99.0  |
| 1.00     | 100.0       | 100.0   | 100.0 |
</details>

![](images/a20866dd6a9750906f3baeaa12e49e9ae1bb77d0f43c7519dc4936e09bef1365.jpg)

<details>
<summary>line chart</summary>

| Mask IoU | Shift | Dropout |
| -------- | ----- | ------- |
| 0.00     | 52    | -       |
| 0.25     | 69    | 75      |
| 0.50     | -     | 80      |
| 0.75     | 75    | 83      |
| 1.00     | -     | 83      |
</details>

Figure 8 Robustness to noisy first-frame mask prompts. We perturb only the first-frame prompt mask at inference time while keeping the model checkpoint fixed. The online SAM3 mask is used as the default prompt, and all corrupted masks are generated from this default mask. The x-axis denotes the mask IoU with respect to the online SAM3 mask before perturbation, and the y-axis denotes the real-world success rate.

## C Ablations on Alternative Mask-Conditioning Designs

Here we discuss several design choices we considered before settling on the final MaskWAM formulation described in Section 3. These negative results suggest that masks should not be treated as a lightweight side signal, but should be aligned with the pretrained visual latent space and coupled with an explicit future prediction objective.

Direct mask downsampling or a separate mask encoder. We first experimented with simpler ways of obtaining mask latents. One option was to directly downsample binary masks to the spatial resolution of the video latent. Another option was to train a lightweight 3D CNN mask encoder from scratch. Both designs led to worse performance than encoding rendered mask frames with the pretrained RGB video VAE. We found that the choice of VAE is important: directly downsampled masks lack the semantic and structural priors of the pretrained visual latent space, while a separately learned mask encoder introduces a representation mismatch between RGB latents and mask latents. In contrast, using the same RGB VAE for rendered masks maps both RGB and mask observations into a shared latent space, making channel-level fusion more stable.

ControlNet-style mask injection. We also tried injecting mask information through a separate mask encoder and zero-initialized residual adapters, similar in spirit to ControlNet-style conditioning. This design did not work well in our setting. Although the side branch provides an additional mask condition, the mask signal remains auxiliary and is not explicitly tied to the model’s future prediction objective. As a result, the policy can still underuse the mask, especially in language-ambiguous tasks where the target object must be distinguished from visually similar distractors. This motivated us to make the mask part of the main RGB-mask latent stream and supervise it directly through future mask prediction.

Zero-initialized gated mask fusion for VLA. For the visual-prompted VLA baseline, we further tested a zeroinitialized gated fusion design. Specifically, the RGB image and mask image are encoded separately, and the

Table 5 Generalization performance under language ambiguity with explicit spatial prompting. Success rates are reported in ${ \% } ,$ and each model is evaluated over 60 trials per task under each generalization setting. Additional qualitative examples of these generalization settings are shown in Figure. 9.

<table><tr><td rowspan="2">Generalization Setting</td><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="4">Language ambiguity</td><td rowspan="2">Average (%)</td></tr><tr><td>Task 5</td><td>Task 6</td><td>Task 7</td><td>Task 8</td></tr><tr><td rowspan="4">In Domain</td><td> $\pi_0$ -mask</td><td>VLA</td><td>96.7</td><td>93.3</td><td>38.3</td><td>23.3</td><td>62.9</td></tr><tr><td> $\pi_0$ -coord</td><td>VLA</td><td>53.3</td><td>88.3</td><td>20.0</td><td>5.0</td><td>41.7</td></tr><tr><td>FastWAM-coord</td><td>WAM</td><td>35.0</td><td>55.0</td><td>13.3</td><td>1.7</td><td>26.3</td></tr><tr><td>Ours</td><td>WAM</td><td>100.0</td><td>100.0</td><td>88.3</td><td>83.3</td><td>92.9</td></tr><tr><td rowspan="4">Distractors</td><td> $\pi_0$ -mask</td><td>VLA</td><td>86.7</td><td>75.0</td><td>31.7</td><td>18.3</td><td>52.9</td></tr><tr><td> $\pi_0$ -coord</td><td>VLA</td><td>46.7</td><td>61.7</td><td>11.7</td><td>0.0</td><td>30.0</td></tr><tr><td>FastWAM-coord</td><td>WAM</td><td>15.0</td><td>23.3</td><td>5.0</td><td>0.0</td><td>10.8</td></tr><tr><td>Ours</td><td>WAM</td><td>100.0</td><td>98.3</td><td>85.0</td><td>78.3</td><td>90.4</td></tr><tr><td rowspan="4">Novel Instances</td><td> $\pi_0$ -mask</td><td>VLA</td><td>71.7</td><td>65.0</td><td>26.7</td><td>15.0</td><td>44.6</td></tr><tr><td> $\pi_0$ -coord</td><td>VLA</td><td>41.7</td><td>51.7</td><td>8.3</td><td>0.0</td><td>25.4</td></tr><tr><td>FastWAM-coord</td><td>WAM</td><td>18.3</td><td>28.3</td><td>6.7</td><td>0.0</td><td>13.3</td></tr><tr><td>Ours</td><td>WAM</td><td>86.7</td><td>76.7</td><td>70.0</td><td>65.0</td><td>74.6</td></tr><tr><td rowspan="4">Lighting</td><td> $\pi_0$ -mask</td><td>VLA</td><td>75.0</td><td>68.3</td><td>28.3</td><td>13.3</td><td>46.3</td></tr><tr><td> $\pi_0$ -coord</td><td>VLA</td><td>48.3</td><td>66.7</td><td>15.0</td><td>3.3</td><td>33.3</td></tr><tr><td>FastWAM-coord</td><td>WAM</td><td>31.7</td><td>36.7</td><td>8.3</td><td>0.0</td><td>19.2</td></tr><tr><td>Ours</td><td>WAM</td><td>93.3</td><td>91.7</td><td>73.3</td><td>68.3</td><td>81.7</td></tr></table>

fused visual embedding is computed as

$$
e _ {\mathrm{fused}} = e _ {\mathrm{rgb}} + g \cdot e _ {\mathrm{mask}},
$$

where the learnable gate g is initialized to zero to preserve the original pretrained behavior at the beginning of fine-tuning. However, we observed that the model tended to keep the mask pathway weak and largely ignored the mask embedding during training. This suggests that preserving the pretrained behavior through zero initialization, while useful for stability, does not by itself guarantee that the model will learn to use the new mask signal for fine-grained spatial grounding.

![](images/0cd3418134725aadd7c4ecdf816a7d692ac41e0380834429dd541cbe278a1d34.jpg)

<details>
<summary>text_image</summary>

Setting
In-domain
In-domain
Out-of-domain
Out-of-domain
Out-of-domain
Train object
Test object
Train object
Test object
Train object
Test object
Train object
Test object
</details>

Figure 9 Additional in-domain and out-of-domain real-world demonstrations. We show representative indomain and out-of-domain rollouts, demonstrating that MaskWAM maintains precise mask-based target grounding under varied object appearances, placements, and scene layouts.

PredictePredicted Masks n map  
![](images/180f2552344a317e73c061e22204927992c58170520834013aeba4764670e6e3.jpg)

<details>
<summary>natural_image</summary>

Grid of 16-panel medical imaging sequence showing green circular objects on white surfaces, with corresponding thermal or heatmaps below (no text or symbols)
</details>

PredictePredicted Masks n map  
![](images/279bb54bddc156f4269c92f774bcebe883cbbd3584e5170d291e0f80ec57222e.jpg)

<details>
<summary>natural_image</summary>

Grid of 16-panel medical imaging sequence showing a robotic arm performing on a lab, with white arrows indicating motion or force vectors (no text or symbols present)
</details>

PredictePredicted Masks n map  
![](images/78ba0335340112e6d9378cf48d773abb5384b5ff9b5ad3821bad3c6f218c99fa.jpg)

<details>
<summary>natural_image</summary>

Grid of 16-panel medical imaging sequence showing robotic arm, white tool, and thermal imaging views (no text or symbols)
</details>

PredictePredicted Masks  map  
![](images/2ca6e09448e633d1d596cb7a6d6fe76d235d11904948a502a48a95a2da8cf4f9.jpg)

<details>
<summary>natural_image</summary>

Grid of 16-panel medical imaging sequence showing robotic arm, tissue images, and thermal imaging overlays (no text or symbols)
</details>

Figure 10 Visualization of predicted RGB frames, masks, and attention masks in real data. For visualization only, we decode full future RGB and mask sequences offline; during real-world deployment, MaskWAM uses partial denoising and does not generate full future sequences at test time.

![](images/c1ecaa1f1f54587eb5b28eb59bd7ca1b6f0d3c572f0d8fdb30c5f6ab2600fb54.jpg)  
Figure 11 Visualization of predicted RGB frames, masks, and attention masks in Robotwin benchmark.

![](images/5aa9acb0a592a96245ae7279447d23882bb2adaf686f7d3d36f19ceb95cdcf29.jpg)  
Figure 12 Comparison of attention maps between text-prompted and visual-prompted WAMs. The text-prompted WAM fails to reliably distinguish different textual instructions, leading to scattered attention patterns over task-irrelevant regions. In contrast, with the first-frame mask prompt and future mask prediction, MaskWAM produces more focused attention on task-relevant regions.