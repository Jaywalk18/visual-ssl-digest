# O-MARC: Omni Memory-Augmented Compression Distillation for Efficient Video Understanding

Peiran Wu1, Yunze Liu\*2, Chi-Hao Wu2, Chen Chen3, Junxiao Shen1,2, 1University of Bristol, 2Memories.ai Research, 3University of Central Florida, Project Web: O-MARC

# Abstract

Omnimodal large language models enable unified audio video understanding, but long joint token sequences make inference costly, and existing benchmarks do not fully isolate audio visual association in noisy user generated videos. We introduce UGC-AVQA, a public UGC benchmark with 1,000 videos and 4,816 QA pairs, where an audio removal test ensures that benchmark questions require both acoustic and visual evidence. To reduce inference cost, we propose OMAC, a training free plug in compression method that preserves salient visual memory and temporally grounded audio anchors. To further make compact models robust to compressed inputs, we introduce O-MARC, a compression distillation framework for learning with memory compressed multimodal contexts. On Qwen2.5-Omni-3B, O-MARC improves the average score across four benchmarks to 45.8, outperforming full token inference at 44.1 and OmniZip at 41.0. OMAC also keeps inference efficient, reducing latency by 34.6% (1.53× speedup) and memory by 34.7% compared with full token inference.

# 1 Introduction

Omnimodal large language models have recently emerged as a promising direction for unified audio, visual, and language understanding (Cheng et al., 2024; Xu et al., 2025; Liu et al., 2026). Unlike vision-language models that primarily rely on visual inputs (Wang et al., 2024; Liu et al., 2023), omnimodal models jointly process video frames, audio streams, and textual instructions, enabling reasoning over both what is seen and what is heard. This capability is particularly important for real-world videos, where speech, ambient sounds, scene transitions, and visual actions often provide complementary evidence for understanding complex events. Despite these advances, efficient omnimodal inference remains a significant challenge. Since audio and video are both token-intensive modalities, their joint representation can easily produce long multimodal sequences, leading to substantial memory overhead and slow inference, especially for long videos (Shao et al., 2025a,b; Shen et al., 2026).

Token compression is a natural solution for reducing this cost, but extending it to audiovisual inputs is non-trivial. Unlike visual-only compression, audiovisual compression must account for two modalities with different redundancy patterns and temporal characteristics. Video tokens often exhibit spatial and temporal redundancy (Wu et al., 2025; Song et al., 2024), whereas audio tokens may encode brief but decisive cues. Crucially, the importance of each modality is often defined in relation to the other: a sound becomes meaningful only when grounded in the visible scene, and a visual event may remain ambiguous without its corresponding audio. Compressing audio and video in isolation can therefore remove cross-modal evidence that is necessary for accurate reasoning (Tao et al., 2025; Ding et al., 2026).

Another important challenge lies in evaluation. Existing audiovisual benchmarks have covered broad omnimodal understanding, long video reasoning, and temporal alignment, but they do not explicitly isolate joint audio and visual association in noisy user generated videos (Li et al., 2025; Hong et al., 2025; Zhou et al., 2025). For example, video understanding benchmarks such as VideoMME (Fu et al., 2025), TempCompass (Liu et al., 2024), and LongVideoBench (Wu et al., 2024) have been widely adopted to evaluate long video reasoning. Nevertheless, the models commonly assessed on these benchmarks are predominantly vision based and rarely make use of audio signals. To fill this gap, we introduce UGC-AVQA, a benchmark constructed from public UGC short videos. Each question requires evidence from both modalities. To ensure that the benchmark genuinely evaluates audiovisual reasoning, we further apply an audio removal difficulty check: a sample is kept for evaluation only if a strong model fails when the audio track is removed. This procedure filters out questions that can be solved from visual priors alone.

To address this efficiency problem, we propose Omni Memory Augmented Compression, a plug in method inspired by video memory compression, particularly MARC (Wu et al., 2025). OMAC keeps compact visual memory and audio anchors, then uses the visual memory distribution to allocate audio budget without modifying the backbone model. Beyond training free inference, we further introduce O-MARC, an omni memory augmented compression distillation framework that teaches omnimodal models to reason under compressed multimodal contexts. Experiments show that OMAC improves over direct token compression with comparable inference cost across 3B, 7B, and 30B scale models. O-MARC further delivers strong gains in the efficient 3B setting, and when scaled to the 7B model it outperforms the contemporary training based baseline on the overall average.

Our contributions are threefold. First, we introduce UGC-AVQA, a public UGC benchmark for audio visual association. Second, we propose OMAC as a plug in compression method for audiovisual inputs. Third, we develop O-MARC as a compression distillation framework for robust and efficient omnimodal reasoning.

# 2 Related Work

# 2.1 Omni Large Language Model

In recent years, multimodal systems have undergone continuous development, evolving from large language models to vision language models (Liu et al., 2023) and, more recently, video large language models (Li et al., 2024a; Bai et al., 2025). However, just using visual information is insufficient for achieving multimodal interaction that closely resembles human perception and communication. This limitation highlights the importance of omni models (Xu et al., 2025; Cui et al., 2026). By incorporating both visual and audio modalities during training and inference, omni models can capture richer cross-modal information, identify fine-grained audiovisual cues that conventional video large language models may overlook, and develop a more comprehensive understanding of inter-modal relationships (Fu et al., 2024; Ge et al., 2025; Li et al., 2024b). From early omni-modal systems such as GPT-4o (Hurst et al., 2024) to recent models in the Qwen-Omni (Xu et al., 2025) and Gemini series (Comanici et al., 2025), it is clear that multimodal research is increasingly moving beyond vision-centric large language models towards omni models that support more integrated audio-visual understanding.

<table><tr><td>Dataset</td><td>Videos</td><td>QA pairs</td><td>Focus</td></tr><tr><td>WorldSense (Hong et al., 2025)</td><td>1662</td><td>3172</td><td>Real world omni reasoning</td></tr><tr><td>OmniVideoBench (Li et al., 2025)</td><td>628</td><td>1000</td><td>Long video reasoning</td></tr><tr><td>Daily-Omni (Zhou et al., 2025)</td><td>684</td><td>1197</td><td>Temporal alignment</td></tr><tr><td>UGC-AVQA</td><td>1000</td><td>4816</td><td>UGC audio visual association</td></tr></table>

Table 1: Dataset comparison. UGC-AVQA targets audio visual association in public UGC videos, complementing broader omnimodal and long video benchmarks.

# 2.2 Omni Token Compression

With the development of omni models, these models also encounter the issue of extremely long token sequences when performing audio-visual inference, similar to that faced by large language models for video. Recent research on this issue has focused on token compression to improve the inference efficiency of large multimodal language models (Song et al., 2024; Wu et al., 2025). This approach is highly effective, as multimodal inputs typically contain a significant amount of redundant information (Bolya et al., 2022; Chen et al., 2024; Shang et al., 2025). Token compression methods for visual or text data have been extensively studied, but their application in audio-visual scenarios has received little attention. Furthermore, we believe that some compression methods such as omnizip (Tao et al., 2025) must sacrifice some performance in order to improve efficiency, or require large amounts of training data to achieve performance gains such as omnisift (Ding et al., 2026).

# 3 UGC-AVQA Dataset

We introduce UGC-AVQA, a user generated video dataset for evaluating fine grained audio visual association in realistic short videos. The dataset is built with human annotated captions and QA pairs, and 30% of both benchmark candidates and generated training examples are reviewed by trained human annotators and corrected when necessary. Unlike broad video QA benchmarks, UGC-AVQA focuses on questions that require models to connect what is heard with what is seen across events, scenes, and temporal changes. This is especially important for UGC videos, where camera motion, background speech, environmental sounds, and abrupt transitions make compressed omnimodal reasoning difficult.

![](images/832b0eb8ddd30e5873b118e641b8ec597894e44eb1462727020db52d8dde3f71.jpg)

<details>
<summary>text_image</summary>

UGC Video Collection
1000 Videos
</details>

![](images/e57ca27eef4ff21cbf39d81285cb282e94a318722aa65533a6ddc9fac4b3f3f5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["User"] --> B["Omni Detail Caption"]
    A --> C["Video"]
    B --> D["Audio Visual Question"]
    C --> D
    D --> E["Audio Visual Event Progression"]
    D --> F["Audio Visual Scene or Temporal"]
    D --> G["Cross Scene Audio Visual Alignment"]
    D --> H["Fine Grained Audio and Visual Contrast"]
```
</details>

![](images/399e179a4de4477eb1e248ac5a985bfe90a64ce85eab33d9045e02ff8e11101e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Quality Control"] --> B["Gemini 3.1 flash"]
    B --> C["Answers the 4 tasks: (1) All Wrong: Hard Sample\n(2) Else: Easy Sample"]
    C --> D["Hard Samples\n(UGC AVQA Benchmark)\n206 videos, 1648 QA"]
    D --> E["Human Check"]
```
</details>

![](images/b2411251735dceefa6778e52bfa9dc3d0dccb55a004aafbb31a47ca82fffb517.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["794 Videos"] --> B["(1) Event Progression\n(2) Scene Shift\n(3) Audio Scene Shift\n(4) Key Multimodal Difference"]
    B --> C["Easy Sampled\n(Distillation Data)\n794 videos, 3168 new QA"]
    C --> D["Human Check"]
    E["Omni Detail Caption"] --> F["Gemini 3.1 flash"]
```
</details>

Figure 1: UGC-AVQA construction pipeline. We collect public UGC videos, manually annotate detailed captions and audio visual questions, filter hard benchmark samples with an audio removal test, and review generated QA pairs with trained human annotators.

Table 1 compares UGC-AVQA with recent audio visual benchmarks. WorldSense covers broad real world omnimodal understanding, OmniVideoBench emphasizes long video reasoning, and Daily-Omni includes temporal alignment. UGC-AVQA instead makes audio visual dependence the central target through event progression, scene or temporal transition, cross scene alignment, and fine grained contrast. A detailed category comparison is provided in Appendix A.3.

# 3.1 Collection, Annotation, and Compliance

We collect 1,000 videos from publicly accessible short video platforms and release only the original video URLs, source metadata, detailed captions, and QA annotations, without redistributing raw video files. This preserves provenance and access through the original platforms. The dataset and derived annotations are intended for research on audiovisual reasoning and should be used consistently with the access conditions of the original sources.

Each video is annotated with an omni detail caption covering visual entities, scene transitions, actions, acoustic events, and temporal relations. Human annotators then write multiple choice questions in four categories: Audio-Visual Event Progression, Audio-Visual Scene or Temporal Transition, Cross-Scene Audio-Visual Alignment, and Fine-Grained Audio-Visual Contrast. Representative cases are shown in Appendix A.4.

# 3.2 Difficulty-Guided Split Construction

A central goal of UGC-AVQA is to prevent benchmark questions from being solved by visual priors alone. As shown in Figure 1, we remove the audio track from each candidate sample and evaluate Gemini-3.1-Flash on the visual only input. A sample is included in the benchmark only if the model fails under this audio ablated setting. We then review 30% of retained candidates with trained human annotators and correct them when necessary.

The resulting benchmark split contains 206 videos and 1,648 QA pairs. The remaining 794 videos form the training split, where Geminiassisted annotation creates 3,168 QA pairs across four aligned categories: Event Progression, Scene Shift, Audio-Scene Match, and Key Multimodal Difference. We also review 30% of generated training QA pairs with trained human annotators and correct them when necessary. This yields a difficulty guided curriculum, with easier examples used for training and harder audio dependent examples reserved for evaluation.

# 3.3 Evaluation Protocol

All QA pairs are formatted as multiple choice questions. We report overall and category level accuracy. Since UGC-AVQA is filtered through the audio removal difficulty check, strong performance requires models to preserve acoustic evidence together with visual and temporal context, making it a targeted stress test for audiovisual association under memory compression.

![](images/bfc8f1d761da5372e3a974d83a6360d4460c05a5e132ade34b589b8a715a83ed.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Large Language Model"] --> B["OMAC"]
    B --> C["Video Encoder"]
    B --> D["Audio Encoder"]
    B --> E["Text Encoder"]
    C --> F["Input Frames"]
    D --> G["Frame Summarization"]
    E --> H["Temporal coverage + query score"]
    E --> I["Retained Frame memories"]
    F --> J["Selected Key Frames"]
    G --> K["Selected Key Frames"]
    H --> L["Selected Key Frames"]
    I --> M["Selected Key Frames"]
    J --> N["Token contrast within each frame"]
    K --> O["Token contrast within each frame"]
    L --> P["Contrast score"]
    M --> Q["Contrast score"]
    N --> R["Audio Memory"]
    O --> S["Audio Memory"]
    P --> T["Audio Memory"]
    Q --> U["Audio Memory"]
    R --> V["Audio Token sequence"]
    S --> W["Audio Token sequence"]
    T --> X["Merge nearby tokens into anchors"]
    U --> Y["Merge nearby tokens into anchors"]
    V --> Z["Drop lower score tokens"]
    W --> AA["Merge nearby tokens into anchors"]
    X --> AB["Merge nearby tokens into anchors"]
    Y --> AC["Merge nearby tokens into anchors"]
    Z --> AD["Allocate audio budget across time"]
    AA --> AE["Allocate audio budget across time"]
    AB --> AF["Frames with more visual memory keep more audio"]
    AC --> AG["Frames with more visual memory keep more audio"]
    AD --> AH["Visual memory over time"]
    AE --> AI["Visual memory over time"]
```
</details>

Figure 2: OMAC for training-free compression. OMAC keeps informative visual and acoustic cues, forms compact frame memory tokens, and allocates more audio capacity to time regions that receive more visual memory.

# 4 Omni Memory Augmented Compression

We propose Omni Memory Augmented Compression (OMAC), a method that operates without additional training and compresses long audiovisual inputs by preserving compact memory carriers rather than treating the context as a flat token stream. Let $\mathcal { A } = \{ a _ { i } \} _ { i = 1 } ^ { N _ { a } }$ denote the audio tokens, $\mathcal { V } = \{ v _ { t , p } \}$ the video tokens, where $t \in \{ 1 , \ldots , T \}$ indexes frames and $p \in \{ 1 , \ldots , P \}$ indexes visual positions, and $\mathbf { q } \in \mathbb { R } ^ { d }$ the query embedding. OMAC outputs a compressed multimodal sequence with audio memory $\mathcal { M } _ { a }$ and visual memory $\mathcal { M } _ { v }$ embedded in the original temporal order. It retains frames that are relevant to the query and keeps audio anchors, forms compact frame memory tokens inside selected frames, and uses the resulting visual memory distribution to adjust the temporal audio budget. The compression strength is controlled by $\rho _ { a }$ for audio and $\rho _ { v }$ for video, while other local selection and merging hyperparameters are fixed across experiments.

# 4.1 Frame Scoring & Visual Memory

We first summarize each frame into a frame representation

$$
\bar {\mathbf {v}} _ {t} = \frac {1}{P} \sum_ {p = 1} ^ {P} \mathbf {v} _ {t, p}, \tag {1}
$$

where $\mathbf { v } _ { t , p }$ is the embedding of the p-th token in frame t. This frame summary acts as a coarse visual memory descriptor for the whole frame. At this stage, each frame can be viewed as a candidate memory slot for the full frame. We then score each frame by its query relevance:

$$
s _ {t} = \cos (\bar {\mathbf {v}} _ {t}, \mathbf {q}). \tag {2}
$$

Frames with high scores are selected as key frames, and the number of retained frames is controlled by the video compression ratio $\rho _ { v }$ . In practice, OMAC uses a selection rule with temporal coverage so that retained frame memories remain both relevant to the query and well spread across the video. In other words, OMAC first chooses which frame memories are worth keeping at high fidelity.

Inside each selected frame, OMAC currently performs contrast filtering within the frame. We compute a token contrast score

$$
\alpha_ {t, p} = 1 - \cos (\mathbf {v} _ {t, p}, \mathbf {c} _ {t}), \quad \mathbf {c} _ {t} = \frac {1}{P} \sum_ {p = 1} ^ {P} \mathbf {v} _ {t, p}, (3)
$$

where $\mathbf { c } _ { t }$ is the centroid of frame t. A token receives a high score when it departs from the average content of the frame, which makes it a better carrier of local evidence than repetitive background content. After normalizing $\alpha _ { t , p } ,$ we keep the tokens with the largest scores in the selected frame as explicit visual memory. These retained tokens form the fine grained evidence inside the memory slot for that frame.

We also form a small frame memory token by weighted pooling over the selected tokens:

$$
\mathbf {z} _ {t} = \sum_ {p \in \mathcal {K} _ {t}} \frac {\exp (\hat {\alpha} _ {t , p})}{\sum_ {j \in \mathcal {K} _ {t}} \exp (\hat {\alpha} _ {t , j})} \mathbf {v} _ {t, p}, \tag {4}
$$

where $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ is the set of retained high score tokens in frame $t ,$ and $\hat { \alpha } _ { t , p }$ is the normalized token contrast score. This compact token serves as an additional frame memory carrier and is inserted back into the original token order using one available token slot in the selected frame. In this way, OMAC preserves both explicit key tokens and a compact summary memory. Therefore, the visual branch has a two level memory structure: frame summaries decide which moments are retained as coarse memory units, and the selected frames are further represented by distinctive local tokens plus a small frame memory token.

# 4.2 Audio Memory Budget Adjustment

For audio, OMAC uses a compression strategy built around anchors. Let $\ell _ { i }$ denote the importance score of audio token $a _ { i }$ . We keep high score tokens as audio anchors and merge nearby dropped tokens into them. These anchors are not merely retained tokens; after merging, they become compact acoustic memory slots that summarize nearby evidence. At a coarser scale, OMAC also groups audio tokens into aligned temporal segments, so the audio branch can be viewed as combining fine grained anchor memory with memory allocation over local time regions. If $\mathcal G ( j )$ is the set of tokens assigned to anchor $a _ { j }$ , then the updated anchor feature $\tilde { \mathbf { a } } _ { j }$ is

$$
\tilde {\mathbf {a}} _ {j} = \frac {\mathbf {a} _ {j} + \sum_ {i \in \mathcal {G} (j)} \omega_ {i j} \mathbf {a} _ {i}}{1 + \sum_ {i \in \mathcal {G} (j)} \omega_ {i j}}, \tag {5}
$$

where $\mathbf { a } _ { j }$ is the original feature of anchor $a _ { j } .$ , and $\omega _ { i j }$ is the merge weight for token $a _ { i }$ . The number of retained audio anchors is primarily controlled by the audio compression ratio $\rho _ { a }$ . This gives a compact audio memory while keeping dominant acoustic evidence such as speech cues, sound events, and local temporal context.

OMAC then uses the resulting visual memory allocation over time to lightly adjust the temporal audio budget. After visual compression, each selected frame retains a different amount of visual evidence, including explicit kept tokens and a small number of frame memory tokens. We therefore interpret the visual memory retained at frame t as a temporal importance signal. Let $\pi ( i )$ map audio token $a _ { i }$ to its aligned frame index, and let $w _ { t }$ be the visual importance weight of frame $t ,$ derived from the amount of visual memory retained for that frame after visual compression. This induces a coarse temporal partition of the audio stream, where each aligned time segment acts as a candidate temporal memory region. If $B _ { a }$ is the total retained audio budget and $n _ { t }$ is the number of audio tokens aligned to frame $t ,$ then the target audio budget for frame t is

$$
b _ {t} = B _ {a} \cdot \frac {n _ {t} w _ {t}}{\sum_ {\tau = 1} ^ {T} n _ {\tau} w _ {\tau}}. \tag {6}
$$

This step does not increase the total audio budget. It only reallocates more audio capacity to moments that receive more visual memory and compresses visually lighter intervals more aggressively. From the memory perspective, this means OMAC spends more acoustic memory on temporal regions that the visual branch has already identified as worthy of higher memory capacity, which helps the compressed sequence preserve event structure across modalities rather than isolated local tokens.

# 5 Omni Memory-Augmented RL Token Compression Distillation

To complement the training free OMAC compression strategy, we further introduce O-MARC, an Omni Memory-Augmented RL Token Compression method built on GRPO (Feng et al., 2026; Guo et al., 2025; Wu et al., 2026). As shown in Figure 3 O-MARC combines memory compression with compression self-distillation: for each query $q ,$ we sample two rollouts from the current policy $\pi _ { \theta } ,$ , including a full token rollout without memory compression as the teacher branch and a compressed rollout with memory compression enabled as the student branch. Since both branches are generated online by the current policy, the entire training procedure remains on-policy.

Let $R _ { i } ^ { \mathrm { f u l l } }$ and $R _ { i } ^ { \mathrm { { c o m p } } }$ denote the scalar rewards of the i-th teacher and student rollout, respectively. We use their discrepancy to measure compressioninduced degradation:

$$
\Delta_ {i} = \frac {\mathrm{ReLU} (R _ {i} ^ {\mathrm{full}} - R _ {i} ^ {\mathrm{comp}})}{| R _ {i} ^ {\mathrm{full}} | + \tau}, \tag {7}
$$

where ReLU(·) keeps only positive degradation, and $\tau$ is a small constant for numerical stability. A larger $\Delta _ { i }$ indicates that compression causes a larger reward drop on the same query.

We then reshape the GRPO advantage using the teacher student reward gap. Let $A _ { i }$ denote the original GRPO advantage of the i-th rollout under the standard GRPO pipeline. We define a compressionaware distillation weight

$$
w _ {i} = \mathrm{ReLU} (A _ {i}) \cdot \Delta_ {i}, \tag {8}
$$

where $w _ { i }$ becomes large only when the sample is both beneficial under GRPO and strongly harmed by compression. We then use it to construct the shaped advantage

$$
\tilde {A} _ {i} = A _ {i} + \lambda w _ {i}, \tag {9}
$$

![](images/2e44ad82a2bcf6d24fc140be4e4e486050b795e417d7c4609dfe7816f83e7f8a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Video"] --> B["OMAC"]
    B --> C["Full-token teacher rollout"]
    B --> D["Compressed student rollout"]
    C --> E["Reward Function/ Reward Model"]
    D --> E
    E --> F["GRPO Advantage/ Compute"]
    E --> G["Compression-aware/ Reward Gap"]
    F --> H["Advantage Shaping"]
    G --> H
    H --> I["W_i = ReLU(A_i)·Δ_i / (Δ_i = A_i + λw_i)"]
    E --> J["Δ_1, Δ_2, ..., Δ_G"]
    J --> K["Δ_i = ReLU(R_i^full - R_i^comp) / |R_i^full| + τ"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
    style F fill:#cfc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#fcc,stroke:#333
```
</details>

Figure 3: O-MARC for training-based compression. The full token branch and compressed branch are sampled from the current policy, and their reward gap shapes the GRPO advantage for robust compression training.

where ${ \tilde { A } } _ { i }$ is the final advantage used for policy optimization, and λ is a hyperparameter controlling the shaping strength. Therefore, samples that are already beneficial under GRPO and are strongly degraded by compression receive larger positive updates.

We denote the clipped policy ratio by

$$
\rho_ {i} (\theta) = \operatorname{clip} \left(\frac {\pi_ {\theta} (o _ {i} \mid q)}{\pi_ {\theta_ {\text {old}}} (o _ {i} \mid q)}, 1 - \epsilon , 1 + \epsilon\right). \tag {10}
$$

Here, $\pi _ { \theta _ { \mathrm { o l d } } }$ denotes the rollout policy used to generate the current batch, and ϵ is the clipping threshold.

Finally, we replace the original advantage in the C-GRPO objective with ${ \tilde { A } } _ { i }$ :

$$
\mathcal {L} _ {\mathrm{C-GRPO}} = \mathbb {E} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \rho_ {i} (\theta) \tilde {A} _ {i} - \beta \mathrm{KL} \left(\pi_ {\theta} \| \pi_ {\text {ref}}\right) \right]. \tag {11}
$$

Relative to standard GRPO, our only modification is to replace the original advantage $A _ { i }$ with the compression-aware shaped advantage ${ \tilde { A } } _ { i }$ .

In this way, the full-token teacher branch does not supervise the compressed branch through tokenlevel probability matching; instead, it distills its preference through advantage shaping, explicitly biasing policy optimization toward behaviors that remain robust under memory compression.

# 6 Experiment

# 6.1 Experimental Settings

Inference Setting. For all video experiments, we sample frames at 1 FPS and use at most 32 frames per video. The maximum single frame size is 50,174. For the audio stream, we use a sample rate of SAMPLE\_RATE=16000. Unless otherwise specified, compression experiments keep the retained ratio at 30% and evaluate the resulting compressed multimodal context on downstream QA benchmarks. All results were run twice and averaged.

Training Setting. For O-MARC distillation, we use Qwen2.5-Omni-3B as the base model and train on 4 NVIDIA RTX PRO 6000 GPUs with 96GB memory each. The training videos follow the same preprocessing protocol as evaluation: 1 FPS sampling, at most 32 frames per video, and a maximum single frame size of 50,174. We train with num\_train\_epochs=1, batch\_size=1, learning rate $1 \times 1 0 ^ { - 5 }$ , and num\_generations=4. This 3B base model is aligned with our goal of building an efficient omnimodal model: Qwen2.5-Omni-7B and Qwen3-Omni-30B-A3B provide useful reference points, but their model scales are substantially larger and therefore less consistent with our efficiency oriented setting. For more training details you can see in appendix A.1.

# 6.2 Benchmarks

The benchmark study first asks whether UGC-AVQA exposes a meaningful gap in audio visual association. Proprietary models remain strongest, with Gemini-3.1-pro and Gemini-3.1-flash-lite reaching 69.1 and 67.1 average accuracy. Among open source models, the much larger Qwen3-Omni-30B-A3B reaches 65.0, while Qwen2.5-Omni-3B only reaches 49.0. This gap supports our dataset motivation: reliable UGC audio visual QA requires more than generic visual recognition or unimodal

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Rank</td><td rowspan="2">Avg.</td><td>AVEP</td><td>AVST</td><td>CSAVA</td><td>FGAVC</td></tr><tr><td colspan="4">UGC-AVQA</td></tr><tr><td colspan="7">Proprietary Models (API)</td></tr><tr><td>Gemini-3.1-pro</td><td>1</td><td>69.1</td><td>68.7</td><td>71.4</td><td>68.5</td><td>68.0</td></tr><tr><td>Gemini-3.1-flash-lite</td><td>2</td><td>67.1</td><td>68.5</td><td>68.0</td><td>64.3</td><td>67.5</td></tr><tr><td colspan="7">Open-source Models</td></tr><tr><td>Gemma-4-E4B-it</td><td>4</td><td>59.7</td><td>58.5</td><td>61.7</td><td>55.6</td><td>63.1</td></tr><tr><td>Gemma-4-E2B-it</td><td>9</td><td>45.6</td><td>46.1</td><td>45.2</td><td>42.0</td><td>49.0</td></tr><tr><td>Gemma-3n-E4B-it</td><td>5</td><td>54.6</td><td>58.0</td><td>57.0</td><td>47.6</td><td>55.6</td></tr><tr><td>Gemma-3n-E2B-it</td><td>7</td><td>53.0</td><td>55.8</td><td>55.6</td><td>46.4</td><td>54.4</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>6</td><td>54.2</td><td>52.7</td><td>54.4</td><td>51.2</td><td>58.7</td></tr><tr><td>Qwen2.5-Omni-3B</td><td>8</td><td>49.0</td><td>44.7</td><td>52.4</td><td>48.3</td><td>50.5</td></tr><tr><td>Qwen3-Omni-30B-A3B</td><td>3</td><td>65.0</td><td>65.3</td><td>66.8</td><td>60.9</td><td>67.0</td></tr></table>

Table 2: Benchmark results on UGC-AVQA. We report average accuracy and category accuracy for proprietary and open source omnimodal models. AVEP, AVST, CSAVA, and FGAVC denote event progression, scene or temporal reasoning, cross scene alignment, and fine grained contrast.

priors.

The category breakdown further shows where models struggle. CSAVA is the lowest category for several open source models, including Qwen3- Omni-30B-A3B at 60.9 and Gemma-3n-E4B-it at 47.6. This suggests that cross scene audio visual alignment is a core difficulty of UGC-AVQA, and motivates compression methods that preserve temporally aligned acoustic and visual evidence rather than only reducing visual redundancy.

# 6.3 Main Results of OMAC and O-MARC

The main comparison evaluates two claims: whether OMAC is a stronger plug in compressor than direct token pruning, and whether O-MARC can train models to reason better under compressed memory. For training free compression, OMAC consistently improves over OmniZip (Tao et al., 2025). On Qwen3-Omni-30B-A3B, the average score increases from 48.5 to 50.9, with UGC-AVQA rising from 53.9 to 56.9 and OmniVideo from 31.7 to 35.5. On Qwen2.5-Omni-7B, OMAC improves the average score from 43.5 to 45.0 and raises WorldSense from 39.6 to 42.4. These gains indicate that preserving visual memory and audio anchors keeps more reasoning evidence than direct pruning at the same retained ratio.

The larger improvement comes from compression aware training. With only 3k training samples, O-MARC on Qwen2.5-Omni-3B improves the average score from 42.8 with OMAC to 45.8 and surpasses the full token 3B baseline. On UGC-AVQA, it raises accuracy from 49.2 to 57.9, showing that the compact model learns to use compressed multimodal memory rather than merely tolerate it. Scaling the same training framework to Qwen2.5-Omni-7B further improves the average score from 45.0 to 47.2 with 3k samples, and to 51.1 with 100k samples.

We also compare with OmniSIFT (Ding et al., 2026), a contemporary compression training work released in May. OmniSIFT uses 107k training samples; for a comparable data scale, our 100k plus setting first uses 104k samples for SFT alignment and then applies RL on 3k UGC samples. Under the same 7B backbone, O-MARC achieves a higher average score than OmniSIFT, 51.1 versus 48.7, and substantially higher UGC-AVQA accuracy, 64.6 versus 56.4. OmniSIFT is slightly stronger on WorldSense, but O-MARC is stronger on DailyOmni, UGC-AVQA, and OmniVideo. This comparison suggests that explicitly training with memory compressed audio visual evidence is especially beneficial for tasks requiring fine grained multimodal association.

# 6.4 Ablation Studies

Compression Ratio. This ablation tests whether OMAC remains useful as the token budget changes. At 55% compression, OMAC reaches 49.5 average accuracy, above both full token inference at 49.0 and OmniZip at 48.6. At 70% compression, OMAC still reaches 49.2, while OmniZip drops to 47.1. The pattern supports the role of memory tokens: they are most valuable when the budget is limited and the model must keep only task relevant evidence.

Video Size. The video size ablation asks whether OMAC is robust when the visual input is shortened. On WorldSense, OMAC improves over OmniZip from 40.2 to 41.7 with 64 frames, and from 39.7 to 41.2 with 32 frames. The gain remains stable even under the smaller visual budget, indicating that OMAC does not simply benefit from more frames, but from allocating the retained memory more effectively.

Efficiency. The cost experiment verifies that the accuracy gains do not come from a heavier inference path. Compared with full token inference, OMAC reduces latency from 3.81s to 2.49s and memory from 24.2GB to 15.8GB. Its cost is almost the same as OmniZip, which uses 2.47s and 15.7GB, but OMAC gives stronger accuracy in the main comparison and ablations. Thus, OMAC keeps the efficiency profile of token compression while preserving more reasoning evidence.

Guidance Phase. This design ablation studies how audio and visual compression should interact. Audio guided visual compression, as in OmniZip, reaches 47.7 on DailyOmni. Directly using visual information to guide audio compression improves the score to 48.2, showing that visual context is useful for locating important acoustic regions. The full OMAC design reaches 49.9 by first constructing visual and audio memory separately, then using the retained visual memory to adjust the audio budget.

<table><tr><td rowspan="2">Method</td><td colspan="2">Settings</td><td>DailyOmni</td><td colspan="5">UGC-AVQA</td><td>OmniVideo</td><td>WorldSense</td><td>Average</td></tr><tr><td>Retained Ratio</td><td>Frames</td><td>Overall Accuracy</td><td>AVEP</td><td>AVST</td><td>CSAVA</td><td>FGAVC</td><td>Overall Accuracy</td><td>Avg. Score</td><td>Overall Accuracy</td><td>ACC</td></tr><tr><td colspan="12">Qwen3-Omni-30B-A3B</td></tr><tr><td>Full Tokens</td><td>100%</td><td>32</td><td>69.9</td><td>65.3</td><td>66.8</td><td>60.9</td><td>67.0</td><td>65.0</td><td>39.4</td><td>52.5</td><td>56.7</td></tr><tr><td>OmniZip</td><td>30%</td><td>32</td><td>61.1</td><td>51.5</td><td>56.6</td><td>54.4</td><td>53.4</td><td>53.9</td><td>31.7</td><td>47.4</td><td>48.5</td></tr><tr><td>OMAC (Ours)</td><td>30%</td><td>32</td><td>63.2</td><td>57.3</td><td>59.0</td><td>53.9</td><td>57.5</td><td>56.9</td><td>35.5</td><td>48.0</td><td>50.9</td></tr><tr><td colspan="12">Qwen2.5-Omni-7B</td></tr><tr><td>Full Tokens</td><td>100%</td><td>32</td><td>56.3</td><td>52.7</td><td>54.4</td><td>51.2</td><td>58.3</td><td>54.2</td><td>34.6</td><td>43.6</td><td>47.2</td></tr><tr><td>OmniZip</td><td>30%</td><td>32</td><td>51.8</td><td>51.5</td><td>53.6</td><td>50.0</td><td>55.8</td><td>52.7</td><td>30.0</td><td>39.6</td><td>43.5</td></tr><tr><td>OMAC (Ours)</td><td>30%</td><td>32</td><td>53.6</td><td>52.4</td><td>54.9</td><td>49.3</td><td>56.3</td><td>53.2</td><td>30.9</td><td>42.4</td><td>45.0</td></tr><tr><td>OmniSIFT* (100k+ train)</td><td>30%</td><td>32</td><td>59.0</td><td>52.9</td><td>55.8</td><td>55.1</td><td>61.7</td><td>56.4</td><td>35.0</td><td>44.4</td><td>48.7</td></tr><tr><td>O-MARC (3k+ train)</td><td>30%</td><td>32</td><td>54.6</td><td>56.3</td><td>64.1</td><td>51.2</td><td>65.3</td><td>59.2</td><td>31.5</td><td>43.4</td><td>47.2</td></tr><tr><td>O-MARC* (100k+ train)</td><td>30%</td><td>32</td><td>60.4</td><td>65.7</td><td>68.2</td><td>58.3</td><td>66.3</td><td>64.6</td><td>35.2</td><td>44.0</td><td>51.1</td></tr><tr><td colspan="12">Qwen2.5-Omni-3B</td></tr><tr><td>Full Tokens</td><td>100%</td><td>32</td><td>53.8</td><td>44.7</td><td>52.4</td><td>48.3</td><td>50.5</td><td>49.0</td><td>31.6</td><td>42.2</td><td>44.1</td></tr><tr><td>OmniZip</td><td>30%</td><td>32</td><td>47.7</td><td>44.7</td><td>49.0</td><td>44.9</td><td>49.8</td><td>47.1</td><td>29.3</td><td>39.7</td><td>41.0</td></tr><tr><td>OMAC (Ours)</td><td>30%</td><td>32</td><td>49.9</td><td>47.6</td><td>52.4</td><td>47.1</td><td>49.5</td><td>49.2</td><td>30.8</td><td>41.2</td><td>42.8</td></tr><tr><td>O-MARC (3k+ train)</td><td>30%</td><td>32</td><td>52.3</td><td>52.4</td><td>61.4</td><td>55.8</td><td>61.9</td><td>57.9</td><td>31.0</td><td>42.1</td><td>45.8</td></tr></table>

Table 3: Main comparison of token compression methods. We compare full token inference, OmniZip, OMAC, O-MARC, and the contemporary OmniSIFT baseline under the same frame budget. DailyOmni and WorldSense report accuracy, OmniVideo reports average score, and UGC-AVQA reports both category and overall accuracy. Average ACC is computed over the main scores of DailyOmni, UGC-AVQA, OmniVideo, and WorldSense. Note(\*): Omnisift is a concurrent project that was open-sourced in May.

<table><tr><td>Method</td><td>Compression Ratio</td><td>Avg.</td></tr><tr><td>Baseline</td><td>0%</td><td>49.0</td></tr><tr><td>OmniZip</td><td>55%</td><td>48.6</td></tr><tr><td>OmniZip</td><td>70%</td><td>47.1</td></tr><tr><td>OMAC</td><td>55%</td><td>49.5</td></tr><tr><td>OMAC</td><td>70%</td><td>49.2</td></tr></table>

Table 4: Effect of compression ratio on UGC-AVQA. OMAC is compared with OmniZip and the full token baseline under moderate and strong compression. 

<table><tr><td>Method</td><td>Frames</td><td>Size</td><td>Avg.</td></tr><tr><td>Baseline</td><td>64</td><td>128 × 28 × 28</td><td>43.6</td></tr><tr><td>Baseline</td><td>32</td><td>64 × 28 × 28</td><td>42.2</td></tr><tr><td>OmniZip</td><td>64</td><td>128 × 28 × 28</td><td>40.2</td></tr><tr><td>OmniZip</td><td>32</td><td>64 × 28 × 28</td><td>39.7</td></tr><tr><td>OMAC</td><td>64</td><td>128 × 28 × 28</td><td>41.7</td></tr><tr><td>OMAC</td><td>32</td><td>64 × 28 × 28</td><td>41.2</td></tr></table>

Table 5: Effect of video size on WorldSense. We compare OMAC and OmniZip with different frame counts and maximum visual token sizes on long video examples. 

<table><tr><td>Method</td><td>Frames</td><td>Max Size</td><td>Time(s)</td><td>Memory(GB)</td></tr><tr><td>Baseline</td><td>64</td><td> $128 \times 28 \times 28$ </td><td>3.81</td><td>24.2</td></tr><tr><td>OmniZip</td><td>64</td><td> $128 \times 28 \times 28$ </td><td>2.47</td><td>15.7</td></tr><tr><td>OMAC</td><td>64</td><td> $128 \times 28 \times 28$ </td><td>2.49</td><td>15.8</td></tr></table>

Table 6: Inference cost on long video. We report latency and GPU memory on WorldSense.

<table><tr><td>Method</td><td>Frames</td><td>Joint Way</td><td>DailyOmni</td></tr><tr><td>Baseline</td><td>32</td><td>NA</td><td>53.8</td></tr><tr><td>OmniZip</td><td>32</td><td>A-V</td><td>47.7</td></tr><tr><td>OMAC(*)</td><td>32</td><td>V-A</td><td>48.2</td></tr><tr><td>OMAC</td><td>32</td><td>V-A-VA</td><td>49.9</td></tr></table>

Table 7: Guidance and intervention phase influence. A-V denotes audio guided visual compression, V-A denotes visual guided audio compression, and V-A-VA denotes the full OMAC design that first builds visual and audio memory and then uses visual memory to adjust the audio budget.

This supports our final design choice: the audio budget should be guided by visual memory after salient visual evidence has been selected, rather than by raw visual signals alone.

# 7 Conclusion

We present a unified study of efficient omnimodal reasoning through benchmark construction, memory compression, and compression aware training. UGC-AVQA targets audiovisual association in public UGC videos and uses audio removal filtering to ensure that evaluation requires both acoustic and visual evidence. OMAC introduces a training free memory compression mechanism that preserves salient visual memory and temporally grounded audio anchors at nearly the same inference cost as direct pruning. Building on this compressed memory, O-MARC trains compact models to reason under a reduced token budget. Experiments show that OMAC consistently improves over OmniZip across model scales, while O-MARC achieves the strongest gains in the efficient 3B setting and outperforms the contemporary OmniSIFT when trained with comparable 100k plus data. These results indicate that audiovisual compression should be viewed not only as an inference acceleration technique, but also as a trainable condition for robust and efficient omnimodal reasoning.

# Limitations

UGC-AVQA is built from publicly accessible short video platforms, so its coverage naturally reflects the style, language, and content distribution of currently available UGC videos. Since we release video links rather than raw files, a small portion of the data may become unavailable over time as platform content changes. Future work can extend the benchmark to more languages, longer videos, and additional real world domains while keeping the same audio visual dependency focused evaluation protocol.

# References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.   
Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2022. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461.   
Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.   
Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and 1 others. 2024. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476.   
Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.   
Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, and 1 others. 2026. Minicpm-o 4.5: Towards real-time full-duplex omnimodal interaction. arXiv preprint arXiv:2604.27393.

Yue Ding, Yiyan Ji, Jungang Li, Xuyang Liu, Xinlong Chen, Junfei Wu, Bozhou Li, Bohan Zeng, Yang Shi, Yushuo Guan, and 1 others. 2026. Omnisift: Modality-asymmetric token compression for efficient omni-modal large language models. arXiv preprint arXiv:2602.04804.   
Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. 2026. Video-r1: Reinforcing video reasoning in mllms. Advances in Neural Information Processing Systems, 38:99114–99137.   
Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, and 1 others. 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118.   
Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Yuhang Dai, Meng Zhao, Yi-Fan Zhang, Shaoqi Dong, Yangze Li, Xiong Wang, and 1 others. 2024. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211.   
Yuying Ge, Yixiao Ge, Chen Li, Teng Wang, Junfu Pu, Yizhuo Li, Lu Qiu, Jin Ma, Lisheng Duan, Xinyu Zuo, and 1 others. 2025. Arc-hunyuan-video-7b: Structured video comprehension of real-world shorts. arXiv preprint arXiv:2507.20939.   
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.   
Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. 2025. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326.   
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.   
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and 1 others. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.   
Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Wentao Wang, Zhenghao Song, Dingling Zhang, and 1 others. 2025. Omnivideobench: Towards audio-visual understanding evaluation for omni mllms. arXiv preprint arXiv:2510.10689.

Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, and 1 others. 2024b. Baichuan-omni technical report. arXiv preprint arXiv:2410.08565.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.   
Kai Liu, Jungang Li, Yuchong Sun, Shengqiong Wu, Daoan Zhang, Wei Zhang, Sheng Jin, Sicheng Yu, Geng Zhan, Jiayi Ji, and 1 others. 2026. Javisgpt: A unified multi-modal llm for sounding-video comprehension and generation. Advances in Neural Information Processing Systems, 38:142289–142324.   
Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. 2024. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772.   
Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. 2025. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22857–22867.   
Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. 2025a. Holitom: Holistic token merging for fast video large language models. arXiv preprint arXiv:2505.21334.   
Kele Shao, Keda Tao, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, and Huan Wang. 2025b. When tokens talk too much: A survey of multimodal long-context token compression across images, videos, and audios. arXiv preprint arXiv:2507.20198.   
Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, and 1 others. 2026. Fastvid: Dynamic density pruning for fast video large language models. Advances in Neural Information Processing Systems, 38:123553–123581.   
Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, and 1 others. 2024. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232.   
Keda Tao, Kele Shao, Bohan Yu, Weiqiang Wang, Huan Wang, and 1 others. 2025. Omnizip: Audio-guided dynamic token compression for fast omnimodal large language models. arXiv preprint arXiv:2511.14582.   
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024. Qwen2- vl: Enhancing vision-language model’s perception

of the world at any resolution. arXiv preprint arXiv:2409.12191.   
Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A benchmark for longcontext interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857.   
Peiran Wu, Yunze Liu, Miao Liu, and Junxiao Shen. 2026. St-think: How multimodal large language models reason about 4d worlds from ego-centric videos. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5174–5183.   
Peiran Wu, Zhuorui Yu, Yunze Liu, Chi-Hao Wu, Enmin Zhou, and Junxiao Shen. 2025. Marc: Memoryaugmented rl token compression for efficient video understanding. arXiv preprint arXiv:2510.07915.   
Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, and 1 others. 2025. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765.   
Ziwei Zhou, Rui Wang, Zuxuan Wu, and Yu-Gang Jiang. 2025. Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities. arXiv preprint arXiv:2505.17862.

# A Appendix

# A.1 Training Details

Our GRPO training uses four sampled rollouts per prompt with a maximum completion length of 1024 and a maximum context length of 32768. We optimize for 2 epochs with learning rate 1 × 10−5, batch size 1 per device, and 4-way generation under bfloat16 and DeepSpeed ZeRO-2. The reward is the sum of two equally weighted terms, namely answer accuracy and format correctness. For multimodal preprocessing, we use 1 FPS video sampling, cap the input to at most 32 frames, disable audio output generation, and enable audio-in-video reasoning.

As shown in Figure 4, the overall reward improves steadily throughout training, while the compression gap ratio exhibits a downward trend despite local fluctuations. This pattern is consistent with the goal of O-MARC: the policy becomes stronger overall and, at the same time, more robust under memory compression.

# A.2 Annotator Instructions

Human annotators were recruited from the authors’ institution as student research assistants with experience in multimodal annotation, and were compensated according to local institutional standards for research assistance. Annotators were instructed that the videos and derived annotations would be used only for research on audiovisual reasoning. Annotators were asked to inspect both the visual stream and the audio track, write an omni detail caption covering visible entities, actions, scene transitions, speech, non-speech sounds, and temporal relations, and avoid adding private identity information beyond what is already visible or audible in the public video. For each video, annotators then wrote multiple choice questions whose answers require both modalities. The four target categories were defined as follows: Audio-Visual Event Progression asks what audio event follows or precedes a visual event; Audio-Visual Scene or Temporal Transition asks how audio changes across visual scene changes; Cross-Scene Audio-Visual Alignment asks which audio segment corresponds to a visual moment across scenes; and Fine-Grained Audio-Visual Contrast asks how subtle acoustic cues differ across visual states. Annotators were instructed to provide one correct option, plausible distractors, and evidence from the caption for each answer. During quality control, reviewers removed questions with ambiguous answers, visually solvable questions, malformed options, or answers unsupported by the video.

![](images/38051ca784006f8c6ad57b2a7cf4c5723c93e3157855a5af98a8fc0b8e886ef2.jpg)

<details>
<summary>line</summary>

| train/global_step | train/reward |
| ----------------- | ------------ |
| 0                 | 1.2          |
| 500               | 1.5          |
| 1k                | 1.55         |
| 1.5k              | 1.6          |
| 2k                | 1.65         |
| 2.5k              | 1.68         |
| 3k                | 1.67         |
| 3.5k              | 1.66         |
</details>

(a) Training reward.

![](images/fd3fdad9d1ce9a495ba4d624168bc2d8f91d789bed1bd4960789b5fd7ee2b8cd.jpg)

<details>
<summary>line</summary>

| Step | train/compression_reward/gap_ratio_mean |
| ---- | -------------------------------------- |
| 0    | 0.1                                    |
| 500  | 0.08                                   |
| 1k   | 0.06                                   |
| 1.5k | 0.07                                   |
| 2k   | 0.05                                   |
| 2.5k | 0.04                                   |
| 3k   | 0.03                                   |
| 3.5k | 0.02                                   |
</details>

(b) Compression reward gap ratio.   
Figure 4: Training dynamics of O-MARC. The total reward rises steadily during GRPO training, while the compression reward gap ratio gradually decreases, indicating that the compressed rollout becomes better aligned with the full-token teacher rollout over time.

# A.3 Question Taxonomy Comparison

We further compare the question taxonomy of UGC-AVQA with the annotation files used by existing audiovisual benchmarks. UGC-AVQA is organized around four association centric categories: Audio-Visual Event Progression, Audio-Visual Scene or Temporal Transition, Cross-Scene Audio-Visual Alignment, and Fine-Grained Audio-Visual Contrast. These categories are designed to test whether a model can jointly inspect what is heard and what is seen, rather than only localize a relevant visual span or recognize an isolated audio event.

Table 8 shows that existing benchmarks provide valuable coverage of general video understanding, but their categories are not primarily defined by cross-modal dependence. WorldSense includes audio recognition and audio counting, while many categories still focus on visual attributes, spatial relations, or object counts. OmniVideoBench contains multimodal reasoning steps, yet the taxonomy is organized around general abilities such as counting, causal reasoning, and temporal understanding. DailyOmni includes AV Event Alignment, but it is one part of a broader daily video taxonomy. In contrast, UGC-AVQA makes audio visual association the central evaluation target.

# A.4 Examples and Necessity

The distinction is also visible in question examples. A WorldSense question may ask the position of an object relative to a person, which mainly tests spatial perception. An OmniVideoBench question may ask how many objects appear when a narration segment is mentioned, where audio helps locate the moment and vision provides the answer. A DailyOmni question may ask which audio event is synchronized with a visual shot, testing local alignment. UGC-AVQA instead asks questions such as what happens in the audio after a specific visual action, what audio accompanies a visual scene, or how the vocal tone changes between two visual moments. Such questions require the model to maintain both streams and reason over their relation.

<table><tr><td>Benchmark</td><td>Representative categories</td><td>Category focus</td></tr><tr><td>UGC-AVQA</td><td>Event progression, scene or temporal transition, cross-scene alignment, fine-grained contrast</td><td>Audio and visual evidence must be compared, aligned, or contrasted across time.</td></tr><tr><td>WorldSense</td><td>Object counting, spatial relation, attribute recognition, event sorting, temporal localization, audio recognition</td><td>Broad world understanding across visual, audio, temporal, and semantic tasks.</td></tr><tr><td>OmniVideoBench</td><td>Counting, causal reasoning, temporal understanding, reference reasoning, fine-grained perception, sentiment analysis</td><td>General long video reasoning, often using speech as a cue to locate or disambiguate visual evidence.</td></tr><tr><td>DailyOmni</td><td>Event sequence, AV event alignment, context understanding, reasoning, inference, comparative</td><td>Daily video understanding with event order, scene context, and general audiovisual matching.</td></tr></table>

Table 8: Comparison of question taxonomies. UGC-AVQA focuses on audio visual association, while existing benchmarks emphasize broader video understanding skills.

(a) Audio-Visual Event Progression   
![](images/f805d8eb957f938ce423bd780a0113c5a36687b555a58eeab55f92e5041fa77c.jpg)

![](images/51db4236208517c5172af2b745122739bb458cdeb36734b7f4d7436eefc2bafc.jpg)

![](images/9a51fd658ff19293db35ba48dea812d69def39619aa8ae6d7bc59c54b4bf13f0.jpg)  
Q: How do the voices change when theair fryer begins to spin?   
A.They becomequieter   
B.They express surprise   
C. They start singing   
D. They stop talking

(b) Cross-Scene Audio-Visual Alignment   
![](images/d5b9bd6e5e40597f3824446559e27e07f15de42f4cea3295042483d5e76c1a4d.jpg)

![](images/43b743f86b65276d59b55858ff22bb1f4917713f7995bab725cffdb7651c2ffe.jpg)

![](images/33905a52e960a4dca9a154d7d4916460f94a62856713df8d98320eb4ddc0284e.jpg)  
Q: Which pronoun sign is being tapped to the wall when the teacher's audio voice first warns against using informal terms?   
A. usted   
B.ustedes   
C.vosotros   
D. Rusty

(c) Fine-Grained Audio-Visual Contrast   
![](images/85da0e4969bc578c48038b11f91264a036c7b88bf1d3e8fbf85b78d8deace016.jpg)

![](images/462710895cfd2960a51aa29dda5c998f31c31358b850f9acfc8222e39a98f2cc.jpg)

![](images/56f501d18acbc950f0dfb87b70dcdfc3729d9a09e10f176149846fc2faf2918e.jpg)  
Q: How does the keyboard backlight color differ between the man's initial question and the subsequent explanation？   
A. Yellow to blue   
B.Blue to purple   
C. Purple to blue   
D. Purple to yellow

(d) Audio-Visual Scene or Temporal Transition   
![](images/7b3b8eec9eb8209498b3d575fad1f7543ed0cbd0b52f1614dca68a43ecb5b731.jpg)

![](images/c96c9e39faf4d158bb3c27f9765511eb558ad5de41f00365b910d466011f13bc.jpg)

![](images/4b616231883d11c3d61133a15a992af61ea2e3d530a4da4f35336d0e05d21914.jpg)  
Q: What sound immediately precedes the camera panning toward the box？   
A. Soft puring   
B.Woman askinga question   
C. Rustling of paper   
D.Thudding sound

Figure 5: Representative UGC-AVQA cases. The figure will show one example from each of the four UGC-AVQA categories: audio visual event progression, scene or temporal transition, cross-scene audio visual alignment, and fine-grained audio visual contrast.

This focus makes UGC-AVQA necessary for evaluating efficient omnimodal models. Compression methods may preserve visually salient frames while discarding brief audio cues, or keep audio anchors without enough visual context to interpret them. A benchmark centered on cross-modal dependency can reveal these failures more directly than broad video QA. The audio removal filtering used in UGC-AVQA further strengthens this property: a question is retained only when the answer cannot be reliably recovered from the muted video, making the benchmark a targeted test of audio visual association rather than visual prior matching.

Figure 5 illustrates the four types of association tested by UGC-AVQA. Together, these cases highlight why the dataset is important for studying au-

diovisual compression: a model must preserve the right acoustic cue, the right visual moment, and their temporal relation under a reduced token budget.