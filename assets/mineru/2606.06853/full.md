# MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models

Yifan Xu1,2, Chao Zhang2\*, Ruifei Ma2, Fei Gao2, Zhifei Yang3, Jiaxing Qi1, Zhipeng Chen4 1School of Computer Science and Engineering, Beihang University 2Beijing Digital Native Digital City Research Center 3School of Computer Science, Peking University 4School of Artificial Intelligence, Beijing University of Posts and Telecommunications

yifan xu@buaa.edu.cn, ariczhang2009@gmail.com https://motion-enhancer.github.io/

# Abstract

The new era has witnessed a remarkable capability to extend Vision-Language Models (VLMs) for tackling tasks of video understanding. While current VLMs excel at event- or story-level understanding, their ability to capture fine-grained motion details remains limited, primarily due to their focus on high-level static semantic structures and macro-event logic. In contrast, Video Diffusion Models (VDMs) are adept at modeling dynamic motion patterns, benefiting from large-scale video data and the intrinsic requirement of temporal generation. In this paper, we introduce MotionEnhancer, a novel approach that leverages motion priors distilled from a powerful video diffusion model as auxiliary supervision to enhance the motion understanding capability of a VLM via attention alignment. MotionEnhancer comprises two simple parameterfree modules, Motion-sensitive Head Selection (MHS) and Motion-salient Text Token Identification (MTTI), to directly extract and optimize motion-related attentions from the VDM in a computation-only manner. MotionEnhancer provides a scalable solution for motion understanding without additional training parameters, modifications to existing architectures, or tool calling. Extensive experiments demonstrate that MotionEnhancer can achieve consistent improvements over state-of-the-art VLMs on two motion-level video understanding benchmarks, especially on motion-related metrics.

# 1. Introduction

In recent years, Vision-Language Models (VLMs) have become the mainstream framework for video understanding, advancing tasks like video captioning and question answering through multimodal alignment and semantic reasoning [1, 4, 5, 12, 27, 34]. Unlike static images, videos contain sequential frames that reflect scene dynamics over time. The temporal relationships among these frames reveal how objects move, interact, and transform over time. Effectively modeling these temporal relationships is crucial for capturing object movement and interaction. Therefore, VLMs must understand not just individual frames, but also the dynamic changes across them.

![](images/6b818b718ccd0dfa90e1b8d819ecf74f6f379e95295168894708c093495cc8cf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["A man is kneeling on a surfboard."] --> B["VDM"]
    B --> C["MotionEnhancer"]
    C --> D["Motion-centric Attention Refinement"]
    D --> E["VLM"]
    E --> F["Standard SFT Result"]
    F --> G["VLM T2V Attention"]
    H["Can you describe this video?"] --> E
    I["Motion Prior"] --> J["Motion Enhancer"]
    J --> K["Guide"]
    K --> E
```
</details>

(A) High-level Overview of MotionEnhancer   
![](images/d201687c5f0d570a77f64db7a626653d970412d790a58ea036747730419754ff.jpg)

<details>
<summary>text_image</summary>

Distinctions Between Different Heads
VDM
head1 head2 head3 head4
Distinctions Between Different Text Tokens
man is kneeling surfboard
</details>

(B) Observation on VDM Attention   
Figure 1. (A) High-level overview of MotionEnhancer, which incorporates motion priors from the VDM as guidance during supervised fine-tuning of the VLM for improved motion understanding. (B) Observation of VDM attention. We observe distinct patterns in the attention maps across different transformer heads and text tokens in the VDM, which motivates our refinement of motion-centric attention.

Most video understanding VLMs follow a common pipeline: extract key frames, encode them with image encoders, and feed the features into multimodal models for alignment and reasoning [2, 4, 34]. As designed for understanding, the core competency of VLMs lies in capturing the overall meaning of a video by integrating information across frames, which prioritizes understanding high-level semantic structures such as static conceptual relationships and macro-event logic. This task-driven approach enables strong performance on event- and story-level understanding, making VLMs well-suited for holistic tasks like video captioning. However, they often overlook fine-grained motion details between frames, leading to a mismatch with the needs of motion-level understanding (for theoretical analysis, please see Sec. 3.1).

Meanwhile, Video Diffusion Models (VDMs) excel at generating visually realistic and temporally coherent videos [3, 10, 26, 31, 36]. During step-by-step denoising, VDMs learn the complex spatiotemporal patterns in videos, including physical laws of object motion, dynamic scene transitions, and inter-frame dependencies. This inherent capability to capture real-world motion dynamics equips VDMs with a deep modeling of video motion patterns, making them serve as implicit simulators with learned interactive dynamics from large-scale video data. Thus, video generative models unlock exciting possibilities for enhancing VLMs by providing more accurate motion modeling. Since attention mechanisms [25] are central to VDMs, their textto-vision attention naturally encodes motion priors. Inspired by this, we propose MotionEnhancer, as shown in Fig. 1(A), which leverages motion priors distilled from a powerful VDM as auxiliary supervision to enhance the motion understanding capability of a VLM via simple attention alignment. This idea aligns with advances in 2D visionlanguage tasks [9]. A key challenge arises: How can we efficiently extract motion-focused attention signals from video QA pairs using a VDM?

As shown in Fig. 1(B), attention maps from the VDM exhibit distinct patterns across transformer heads and text tokens. To refine motion-centric attention, we propose two parameter-free modules: Motion-sensitive Head Selection (MHS) and Motion-salient Text Token Identification (MTTI). MHS draws on the findings of SparseVideoGen [29] and evaluates temporal attention maps using diagonal concentration, spatial continuity, and high-value region ratios to select motion-relevant heads. MTTI computes frame-wise averages and inter-frame differences to identify text tokens responsive to both smooth and abrupt motion. Notably, both modules require no extra training parameters. They can directly extract and optimize motion-related attention maps from the VDM in a computation-only manner. By aligning motion priors from the VDM with the textto-vision attention in the VLM, our approach significantly improves the motion perception and reasoning ability of VLMs on motion-level benchmarks. Crucially, it transfers capabilities from pretrained generative models rather than requiring access to their original training data, enabling an efficient form of capability transfer and enhancement that reduces dependence on large-scale video re-collection. Our work demonstrates a new kind of cross-paradigm model interaction: using internal signals of one model family (e.g., generative VDMs) to guide another (e.g., discriminative VLMs). Our focus is on introducing a simple and generalizable approach, rather than developing complex or specialized module designs for attention extraction or alignment.

Our main contributions can be summarized as follows:

• We propose MotionEnhancer, a novel framework that leverages motion priors distilled from a powerful VDM as auxiliary supervision to enhance the motion understanding capability of a VLM through simple text-to-vision attention alignment. We also provide theoretical analysis to verify the feasibility of MotionEnhancer.   
• To obtain effective motion priors, we design two parameter-free modules, MHS and MTTI, specifically adapted to the temporal characteristics of videos. They directly extract and optimize motion-related attention maps from the VDM in a computation-only manner.   
• Extensive experiments on two motion-level video understanding benchmarks demonstrate the superiority of MotionEnhancer. Our results show that the attention alignment strategy can be successfully extended to video tasks with only minor adaptations.

# 2. Related Work

# 2.1. VLMs for Video Understanding

Recent advances in VLMs have prompted their adaptation to video understanding. Most video VLMs include a visual encoder, modality alignment, and a Large Language Model (LLM) backbone. A common strategy treats videos as image sequences by sampling key frames for encoding [7, 12]. Some enhance input flexibility through new positional embeddings and dynamic resolutions [2, 4], while others use Q-Former [13], spatial-temporal patchification [16], or adaptive pooling [30] to compress and accelerate video encoding. TE Fusion [8] groups frames and applies group-level self-attention for temporal modeling. Motion-Sight [6] improves motion modeling via object spotlighting and motion blur.

Despite progress, current VLMs remain limited in finegrained motion understanding or rely on extra modules and external tools. In contrast, we enhance VLM motion understanding by leveraging motion priors from a video diffusion model, without extra parameters or architecture changes.

![](images/3b3b907bc97f098dc4ad888776968647f3aab3fa02ccf054bfa85b380bc9980a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Video"] --> B["Visual Language Model (VLM)"]
    B --> C["Self Attention"]
    C --> D["A man in sunglasses exits the car."]
    D --> E["L_AR"]
    E --> F["Video Diffusion Model (VDM)"]
    F --> G["3D Full Attention"]
    G --> H["concat"]
    H --> I["Multi head self attention"]
    I --> J["Q"]
    I --> K["K"]
    I --> L["V"]
    G --> M["N Attn. Maps"]
    M --> N["Zvision Q"]
    N --> O["Z_text"]
    O --> P["Zvision K"]
    P --> Q["Text-to-Video Attn."]
    Q --> R["X_V2v"]
    R --> S["Frame1 Frame2 Frame3 Frame4"]
    S --> T["Frame1 Frame2 Frame3 Frame4"]
    T --> U["Frame4 Frame3 Frame4"]
    U --> V["Frame4 Frame3 Frame4"]
    V --> W["Frame1 Frame2 Frame3 Frame4"]
    W --> X["Frame1 Frame2 Frame3 Frame4"]
    X --> Y["Frame1 Frame2 Frame3 Frame4"]
    Y --> Z["Frame1 Frame2 Frame3 Frame4"]
    Z --> AA["Frame1 Frame2 Frame3 Frame4"]
    AA --> AB["Frame1 Frame2 Frame3 Frame4"]
    AB --> AC["Frame1 Frame2 Frame3 Frame4"]
    AC --> AD["Frame1 Frame2 Frame3 Frame4"]
    AD --> AE["Frame1 Frame2 Frame3 Frame4"]
    AE --> AF["Frame1 Frame2 Frame3 Frame4"]
    AF --> AG["Frame1 Frame2 Frame3 Frame4"]
    AG --> AH["Frame1 Frame2 Frame3 Frame4"]
    AH --> AI["Frame1 Frame2 Frame3 Frame4"]
    AI --> AJ["Frame1 Frame2 Frame3 Frame4"]
    AJ --> AK["Frame1 Frame2 Frame3 Frame4"]
    AK --> AL["Frame1 Frame2 Frame3 Frame4"]
    AL --> AM["Frame1 Frame2 Frame3 Frame4"]
    AM --> AN["Frame1 Frame2 Frame3 Frame4"]
    AN --> AO["Frame1 Frame2 Frame3 Frame4"]
    AO --> AP["Frame1 Frame2 Frame3 Frame4"]
    AP --> AQ["Frame1 Frame2 Frame3 Frame4"]
    AQ --> AR["Frame1 Frame2 Frame3 Frame4"]
    AR --> AS["Frame1 Frame2 Frame3 Frame4"]
    AS --> AT["Frame1 Frame2 Frame3 Frame4"]
    AT --> AU["Frame1 Frame2 Frame3 Frame4"]
    AU --> AV["Frame1 Frame2 Frame3 Frame4"]
    AV --> AW["Frame1 Frame2 Frame3 Frame4"]
    AW --> AX["Frame1 Frame2 Frame3 Frame4"]
    AX --> AY["Frame1 Frame2 Frame3 Frame4"]
    AY --> AZ["Frame1 Frame2 Frame3 Frame4"]
    AZ --> BA["Frame1 Frame2 Frame3 Frame4"]
    BA --> BB["Frame1 Frame2 Frame3 Frame4"]
    BB --> BC["Frame1 Frame2 Frame3 Frame4"]
    BC --> BD["Frame1 Frame2 Frame3 Frame4"]
    BD --> BE["Frame1 Frame2 Frame3 Frame4"]
    BE --> BF["Frame1 Frame2 Frame3 Frame4"]
    BF --> BG["Frame1 Frame2 Frame3 Frame4"]
    BG --> BH["Frame1 Frame2 Frame3 Frame4"]
    BH --> BI["Frame1 Frame2 Frame3 Frame4"]
    BI --> BJ["Frame1 Frame2 Frame3 Frame4"]
    BJ --> BK["Frame1 Frame2 Frame3 Frame4"]
    BK --> BL["Frame1 Frame2 Frame3 Frame4"]
    BL --> BM["Frame1 Frame2 Frame3 Frame4"]
    BM --> BN["Frame1 Frame2 Frame3 Frame4"]
    BN --> BO["Frame1 Frame2 Frame3 Frame4"]
    BO --> BP["Frame1 Frame2 Frame3 Frame4"]
    BP --> BQ["Frame1 Frame2 Frame3 Frame4"]
    BQ --> BR["Frame1 Frame2 Frame3 Frame4"]
    BR --> BS["Frame1 Frame2 Frame3 Frame4"]
    BS --> BT["Frame1 Frame2 Frame3 Frame4"]
    BT --> BU["Frame1 Frame2 Frame3 Frame4"]
    BU --> BV["Frame1 Frame2 Frame3 Frame4"]
    BV --> BW["Frame1 Frame2 Frame3 Frame4"]
    BW --> BX["Frame1 Frame2 Frame3 Frame4"]
    BX --> BY["Frame1 Frame2 Frame3 Frame4"]
    BY --> BZ["Frame1 Frame2 Frame3 Frame4"]
    BZ --> CA["Frame1 Frame2 Frame3 Frame4"]
    CA --> CB["Frame1 Frame2 Frame3 Frame4"]
    CB --> CC["Frame1 Frame2 Frame3 Frame4"]
    CC --> CD["Frame1 Frame2 Frame3 Frame4"]
    CD --> CE["Frame1 Frame2 Frame3 Frame4"]
    CE --> CF["Frame1 Frame2 Frame3 Frame4"]
    CF --> CG["Frame1 Frame2 Frame3 Frame4"]
    CG --> CH["Frame1 Frame2 Frame3 Frame4"]
    CH --> CI["Frame1 Step Function: Visual Language Model (VLM) - Self Attention - Please describe this video."]
    CI --> CJ["Image Input Video - DDIM Inversion - Visual Language Model (VLM) - Self Attention - A man in sunglasses exits the car."]
    CJ --> CK["Noise Latents - A man wearing sunglasses is getting out of the car."]
    CK --> CL["Video Diffusion Model (VDM) - 3D Full Attention - Concat / Multi head self attention / VDM - Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / Vmax / GND - A man in sunglasses exits the car."]
```
</details>

Figure 2. Framework of MotionEnhancer. Our method leverages motion priors distilled from a powerful VDM as auxiliary supervision to enhance the motion understanding capability of a VLM through attention alignment. Attention maps extracted from the VDM during DDIM sampling are filtered by the Motion-sensitive Head Selection (MHS) and Motion-salient Text Token Identification (MTTI) modules to identify motion-relevant attentions. The resulting text-to-vision attentions are then used to guide the VLM during supervised fine-tuning.

# 2.2. Diffusion Model Guidance for VLMs

Diffusion models generate high-quality visual representations by progressively denoising latent features, capturing fine-grained semantics and structure [3, 22]. This requires precise cross-modal modeling, allowing diffusion models to enhance VLMs’ feature extraction and reasoning. For instance, DIVA [28] uses generative feedback from frozen diffusion models to optimize CLIP features. GenHancer [19] introduces lightweight denoisers and class-token-based reconstruction. Lavender [9] aligns VLM attention with Stable Diffusion to transfer visual expertise.

However, these methods focus mainly on images. In this work, we explore a simple yet effective way to enhance VLM’s ability for video motion understanding. We show that attention map alignment between diffusion models and VLMs extends easily to video, requiring minimal temporal adaptation and no extra backbone changes.

# 3. MotionEnhancer Theory

We start by comparing VLM’s learned distribution with the one required by motion understanding, and show that the latter naturally aligns with VDM’s generative distribution.

# 3.1. Distributional Mismatch Between VLMs and the Requirements of Motion Understanding

A VLM is trained with an autoregressive objective

$$
\mathcal {L} _ {\mathrm{AR}} = - \sum_ {i} \log p _ {\theta} (r _ {i} \mid \mathbf {V}, r _ {<   i}), \tag {1}
$$

where $\mathbf { V } ~ = ~ ( V _ { 1 } , \ldots , V _ { F } )$ is the video and $r _ { i }$ is the i-th response token. Internally, its text-to-video attention can be interpreted as a discriminative conditional distribution

$$
A ^ {V L M} (t, s, f) \approx p _ {\theta} (t \mid V _ {f} (s), s _ {<   i}), \tag {2}
$$

where t is a text token and $V _ { f } ( s )$ is the visual feature at spatial position s in frame f . This defines a $\boldsymbol { \mathbf { \rho } } _ { \boldsymbol { p } } ( t | V )$ distribution, answering: “Given the visual input (often a single frame), how likely is this token?” Crucially, the model can satisfy this objective using static appearance cues, such as context and background, without modeling temporal dynamics or how visual evidence changes over time.

In contrast, motion understanding requires a different form of reasoning. Motion-level questions q like “Does the person start running then stop?” or “In which direction does the camera move?” depend on identifying where and when motion-related evidence appears across frames [11, 18]. This corresponds to a latent-variable factorization:

$$
p ^ {\star} (a \mid \mathbf {V}, q) = \sum_ {\mathbf {E}} p ^ {\star} (a \mid \mathbf {E}, q) p ^ {\star} (\mathbf {E} \mid \mathbf {V}, q), \tag {3}
$$

where E denotes motion-related latent evidence (object trajectories, motion segments, frame intervals) and a is the answer. To reveal the structure of $p ^ { \star } ( \mathbf { E } | \mathbf { V } , q )$ , we follow common practices in grounded captioning and VideoQA [15, 21], and decompose the global evidence E into pertoken evidence subsets:

$$
\mathbf {E} = E _ {t} \mid t \in \mathcal {T} (q, a), \tag {4}
$$

where ${ \mathcal { T } } ( q , a )$ is the set of semantic units, such as verbs and motion-related tokens, in the question or answer space. Under this factorization, the evidence distribution becomes:

$$
p ^ {\star} (\mathbf {E} \mid \mathbf {V}, q) = \prod_ {t \in \mathcal {T} (q, a)} p ^ {\star} (E _ {t} \mid \mathbf {V}, t, q), \tag {5}
$$

reflecting the principle that the evidence relevant to each semantic unit t is conditionally independent given the video.

Now, each $E _ { t }$ must specify the spatiotemporal regions in V where the concept t is visually grounded. Assuming $E _ { t }$ is selected by scoring each video location $( s , f )$ with a relevance function $r _ { t } ( s , f )$ , we can rewrite:

$$
p ^ {\star} (E _ {t} \mid \mathbf {V}, t, q) \propto \prod_ {(s, f) \in E _ {t}} r _ {t} (s, f), \tag {6}
$$

and the relevance score itself is a normalized conditional likelihood [21]:

$$
r _ {t} (s, f) = p ^ {\star} (v _ {s, f} \mid t), \tag {7}
$$

which is the probability that the patch $v _ { s , f }$ is an instance of the concept t. Thus, $p ^ { \star } ( \mathbf { E } | \mathbf { V } , q )$ is governed by a set of concept-conditioned visual distributions $p ^ { \star } ( v _ { s , f } | t )$ , where $t \in { \mathcal { T } } ( q , a )$ and $( s , f ) \in \mathbf { V }$ . This defines an evidenceseeking $p ( V | t )$ distribution: “Given a semantic concept t, where in the video is its visual evidence, and how does it evolve over time?”

For semantic units that encode actions or dynamics, the associated latent evidence E must encode frame-to-frame transitions. At the level of per-location features, this evidence can be expressed or alternated as motion:

$$
\mathbf {E} [ \text { Motion } (s, f) ] \propto \| V _ {f + 1} (s) - V _ {f} (s) \|. \tag {8}
$$

In summary, the distribution learned by VLMs $( p ( t | V ) )$ is misaligned with the $p ( V | t )$ structure required for motion understanding. This mismatch explains VLMs’ strong appearance bias and limited temporal sensitivity, motivating the pursuit of the evidence-seeking distribution $p ( V | t )$ .

# 3.2. Why VDM Attention is a Reliable Source of Motion Priors?

A VDM naturally yields the concept-conditioned visual distribution for motion reasoning. During generation, it predicts the clean latent video $\mathbf { z } _ { 0 }$ from noisy inputs $\mathbf { z } _ { k } .$ , conditioned on text tokens. At each step, the model uses attention to determine how the visual content for a token t should appear across spatial and temporal positions. This attention guides what to generate at each location $( s , f )$ , effectively forming a generative, concept-conditioned map [23]:

$$
A ^ {V D M} (t, s, f) \approx p _ {\phi} (v _ {s, f} \mid t, \mathbf {z} _ {k}), \tag {9}
$$

reflecting where the semantics of token t are visually realized in the generated video.

Unlike discriminative VLMs that can rely on static cues, VDMs must generate temporally coherent videos, ensuring adjacent frames form plausible motion. When a token t corresponds to dynamic content—like actions or camera movement—the denoising objective enforces accurate modeling of feature changes across frames. Regions with higher temporal variation (larger $| V _ { f + 1 } ( s ) - V _ { f } ( s ) | \rangle$ ) are harder to reconstruct and thus receive more modeling focus. As a result, the cross-attention for motion-related tokens becomes sensitive to temporal changes, highlighting regions whose dynamics align with the semantics of t.

This gives VDM attention two key properties. First, it approximates the evidence-seeking distribution $p ^ { \star } ( V | t )$ by highlighting where concept t is grounded across space and time. Second, it is naturally motion-calibrated, with attention shifts reflecting actual motion magnitudes. Together, these properties make VDM-distilled attention effective for improving motion-enhanced VLMs via attention alignment.

# 4. MotionEnhancer Methodology

We propose leveraging motion priors from a VDM as auxiliary supervision to enhance VLMs’ motion understanding via attention alignment (Fig. 2). Using DDIM inversionsampling, we extract attention maps from the VDM and apply two parameter-free modules, Motion-sensitive Head Selection (MHS) and Motion-salient Text Token Identification (MTTI), to identify motion-relevant heads and tokens. These motion priors guide attention alignment in the VLM.

While we use Qwen2.5-VL [2], InternVL3 [37], and CogVideoX-1.5-5B [31], our method is general and applicable to other VLMs and DiT-based VDMs.

# 4.1. Video Diffusion-based Attention Extraction

An input video is compressed by a 3D causal VAE into a latent $z _ { \mathrm { v i s i o n } } ~ \in ~ \mathrm { R } ^ { F \times \bar { H } \times W }$ , where F , H, and W denote frames, height, and width, with total sequence length $S = F \times H \times W$ . We apply 5-step DDIM inversion (details in supplementary) to extract noise, which is concatenated with text embeddings ${ \mathit { z } } _ { \mathrm { t e x t } }$ to form the multimodal sequence $z _ { \mathrm { m m } }$ . CogVideoX is trained with zero terminal SNR, following LDM’s noise schedule [22], improving generation quality but still allowing some sampling deviations [33]. To address this, we reconstruct from noise using classifier-free guidance, incorporating cross-stream memory from a parallel DDIM-inverted path. During the 5-step denoising, attention maps [25] are computed and stored at each step:

$$
A _ {\mathrm{mm}} = \text { Softmax } \left(\frac {Q _ {\mathrm{mm}} K _ {\mathrm{mm}} ^ {T}}{\sqrt {d}}\right), \tag {10}
$$

where $Q _ { \mathrm { m m } }$ and $K _ { \mathrm { m m } }$ are the queries and keys from $z _ { \mathrm { m m } } ,$ and d is the feature dimension. The final attention map Amm is obtained by applying both layer-wise and timestep-wise average pooling to the raw attention outputs.

# 4.2. Motion-centric Attention Refinement

For each transformer head in the VDM, we extract its attention map $A _ { \mathrm { m m } }$ . Since not all heads capture temporal motion [29], we propose Motion-sensitive Head Selection (MHS) to select and aggregate motion-relevant heads. Additionally, Motion-salient Text Token Identification (MTTI) filters out motion-irrelevant text, helping the VLM focus on meaningful text-video motion connections.

Motion-sensitive Head Selection (MHS). To identify motion-relevant heads, we leverage the observation that attention weights often form diagonal patterns in framelevel maps, indicating temporal continuity in fixed regions [20, 29]. We quantify this pattern without introducing trainable parameters. For each head, a motion mask $\mathcal { M } \in \mathbf { \bar { R } } ^ { S \times S }$ captures this diagonal structure. We then evaluate each vision-to-vision attention map $A _ { \mathrm { v } 2 \mathrm { v } } ~ \in ~ \mathrm { R } ^ { S \times S }$ using three metrics: concentration, spatial coherence, and prevalence.

(1) Diagonal Focus Coefficient (DFC). DFC measures the proportion of attention concentrated within the diagonal mask compared to regions outside the mask. For the attention map $A _ { \mathrm { v 2 v } }$ , DFC is defined as:

$$
\mathrm{DFC} = \frac {\sum_ {(i , j) \in \mathcal {M}} A _ {\mathrm{v2v}} ^ {2} [ i , j ]}{\sum_ {(i , j) \notin \mathcal {M}} A _ {\mathrm{v2v}} ^ {2} [ i , j ]}. \tag {11}
$$

A higher DFC indicates that more attention is focused along the diagonal, suggesting stronger motion relevance.

(2) Temporal Continuity Score (TCS). TCS measures the persistence of high diagonal attention, indicating consistent spatial focus across frames. We set the threshold τ as the average attention value. For each spatial location s, we extract a cross-frame submatrix $A _ { s } \in \bar { \mathrm { R } ^ { F \times F } }$ from $A _ { \mathrm { v 2 v } }$ , capturing attention between s across all frame pairs. In each row f of $A _ { s }$ , we identify maximal contiguous segments where attention exceeds τ . Let $\operatorname { L e n } ( s ) = l _ { 1 } , l _ { 2 } , . . . , l _ { m }$ be these segment lengths. TCS is the mean segment length across all s:

$$
\mathrm{TCS} = \frac {1}{S} \sum_ {s = 1} ^ {S} \frac {1}{m} \sum_ {i = 1} ^ {m} l _ {i}, \tag {12}
$$

where S is the total number of spatial locations, and if Len(s) is empty, the segment length is set to 0. A higher TCS indicates more sustained attention to consistent regions across frames, reflecting stronger motion continuity.

(3) Diagonal Saliency Ratio (DSR). DSR quantifies how frequently high attention appears in the diagonal region D. Using the same threshold τ as in TCS, we count the number of entries $n _ { \mathrm { h i g h } }$ in D where attention $\geq \tau$ . Let |D| be the total number of entries in D. Then, DSR is defined as:

$$
\mathrm{DSR} = \frac {n _ {\text { high }}}{| D |}. \tag {13}
$$

Larger DSR means high attention is more widespread along the diagonal, not just at isolated spots.

After computing DFC, TCS, and DSR for all heads, we standardize each metric using its mean and standard deviation. Each head is assigned a composite score by summing its normalized metrics, and the top 50% are selected as motion-related heads. We then average-pool the attention maps $A _ { \mathrm { m m } }$ from these selected heads.

Motion-salient Text Token Identification (MTTI). After aggregating motion-aware heads, we extract the text-tovision attention region, yielding $A _ { \mathsf { t } 2 \mathrm { v } } \in \mathrm { R } ^ { T \times S }$ . Since not all text tokens relate to motion, we assess each token’s temporal dynamics. By average pooling over spatial dimensions $( H \times W )$ , we obtain $\bar { A _ { \mathrm { t 2 f } } } \bar { \in } \mathrm { R } ^ { T \bar { \times } F }$ , where each row captures a token’s attention across frames. For each token t, we compute the mean of its attention and the mean of its first-order differences. The motion score is defined as:

$$
\mathrm{MS} (t) = \operatorname{Mean} _ {f} (A _ {\mathrm{t2f}} ^ {t}) + \frac {1}{F - 1} \sum_ {f = 1} ^ {F - 1} | A _ {\mathrm{t2f}} ^ {t} (f + 1) - A _ {\mathrm{t2f}} ^ {t} (f) |. \tag {14}
$$

The mean attention value captures a token’s overall importance, while the mean first-order difference reflects its temporal fluctuation—higher for dynamic events, lower for static elements. We rank tokens by motion scores and select the top 50% for alignment, ensuring they are both salient and temporally dynamic. The resulting VDM attention is $A _ { \mathsf { V D M } } \in \mathbf { R } ^ { T ^ { \prime } \times \bar { S } }$ , where $T ^ { \prime }$ is the number of selected tokens.

# 4.3. Attention Alignment

The VLMs employ self-attention to model semantic and spatial relationships between text and visual tokens across multiple heads and layers. Following the same procedure as with the VDM, we apply average pooling across heads and layers to obtain an attention matrix $A _ { \mathsf { V L M } } \in \mathrm { R } ^ { T ^ { \prime } \times S }$ , where each row reflects how a text token attends to visual patches.

At this point, we have the text-to-vision attention maps from both the VDM and VLM, denoted as $A _ { \mathrm { V D M } }$ and

Table 1. Quantitative results of MotionBench. \* denotes results we reproduced using their open-source code, while other results are taken from the original benchmark. 

<table><tr><td>Model</td><td>Frames</td><td>Overall</td><td>Average</td><td>MR</td><td>LM</td><td>CM</td><td>MO</td><td>AO</td><td>RC</td></tr><tr><td colspan="10">Small Size Series</td></tr><tr><td>Qwen2.5-VL-3B* [2]</td><td>1fps</td><td>53.56</td><td>49.45</td><td>59.54</td><td>53.11</td><td>38.44</td><td>70.14</td><td>40.46</td><td>35.00</td></tr><tr><td>Qwen2.5-VL-3B + MotionEnhancer (Ours)</td><td>1fps</td><td> $56.60^{\uparrow 3.04}$ </td><td> $52.51^{\uparrow 3.06}$ </td><td>63.06</td><td>61.72</td><td>47.01</td><td>68.84</td><td>43.16</td><td>31.25</td></tr><tr><td>InternVL3-2B* [37]</td><td>8</td><td>53.96</td><td>49.69</td><td>60.01</td><td>57.69</td><td>43.90</td><td>70.00</td><td>40.27</td><td>26.25</td></tr><tr><td>InternVL3-2B + MotionEnhancer (Ours)</td><td>8</td><td> $55.50^{\uparrow 1.54}$ </td><td> $51.35^{\uparrow 1.66}$ </td><td>61.57</td><td>57.51</td><td>46.23</td><td>71.30</td><td>42.00</td><td>29.50</td></tr><tr><td colspan="10">Medium Size Series</td></tr><tr><td>MiniCPM-V2.6-7B [32]</td><td>64</td><td>52</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GLM4-9B + TE Fusion [8]</td><td>16</td><td>58</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-VL-7B* [2]</td><td>1fps</td><td>52.81</td><td>48.29</td><td>59.00</td><td>54.58</td><td>35.58</td><td>71.30</td><td>38.54</td><td>30.75</td></tr><tr><td>Qwen2.5-VL-7B + MotionSight* [6]</td><td>1fps</td><td>55.30</td><td>51.56</td><td>59.88</td><td>57.33</td><td>47.01</td><td>73.91</td><td>40.46</td><td>30.75</td></tr><tr><td>Qwen2.5-VL-7B + MotionEnhancer (Ours)</td><td>1fps</td><td> $57.04^{\uparrow 4.23}$ </td><td> $52.92^{\uparrow 4.63}$ </td><td>63.40</td><td>61.54</td><td>47.27</td><td>70.29</td><td>43.55</td><td>31.50</td></tr><tr><td>InternVL3-8B* [37]</td><td>8</td><td>54.88</td><td>50.81</td><td>60.42</td><td>58.06</td><td>43.64</td><td>70.29</td><td>43.93</td><td>28.50</td></tr><tr><td>InternVL3-8B +MotionEnhancer (Ours) [37]</td><td>8</td><td> $57.69^{\uparrow 2.81}$ </td><td> $53.22^{\uparrow 2.41}$ </td><td>64.14</td><td>60.07</td><td>48.83</td><td>75.94</td><td>40.85</td><td>29.50</td></tr><tr><td colspan="10">Large Size Series</td></tr><tr><td>PLLaVA-34B [30]</td><td>16</td><td>52</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-VL-72B* [2]</td><td>1fps</td><td>58.30</td><td>54.32</td><td>64.00</td><td>60.30</td><td>48.60</td><td>73.20</td><td>46.80</td><td>33.00</td></tr></table>

AVLM. We first interpolate AVLM to match the dimensions of AVDM, then use a 3-layer MLP as the aligner network.

$$
\mathcal {L} _ {\mathrm{MSE}} = \left| \left| \text { Aligner } (A _ {\mathrm{VLM}}) - A _ {\mathrm{VDM}} \right| \right| _ {2}, \tag {15}
$$

where ||·||2 denotes L2-norm. Note that only the previously selected text tokens are involved in this alignment step.

The total loss is as below and optimized in a supervised fine-tuning (SFT) manner, with λ as a balance factor:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\mathrm{AR}} + \lambda \mathcal {L} _ {\mathrm{MSE}}. \tag {16}
$$

# 4.4. Discussion

Here, we discuss two questions: (1) Why do we use average pooling for VLM heads instead of motion-based head selection? VDMs are generative and their heads specialize in spatial or temporal aspects. In contrast, VLMs focus on understanding and lack such clear specialization. Thus, average pooling is more suitable for VLMs. As a result, VLM transformer heads are more general-purpose and do not show the clear specialization found in VDMs.

(2) Can the text token identification strategy select only motion-related tokens? While aimed at selecting motionrelevant tokens, motion is often carried by verbs and their associated subjects or objects. Our method mainly filters out unrelated tokens (e.g., function words like ”the”, ”which”) rather than isolating only verbs.

# 5. Experiments

# 5.1. Experimental Setups

Training Data. We leverage all 5k video QA pairs from MotionBench-Train [8] and sample 20k pairs from MotionVid-QA [6], totally forming 25k pairs for training.

Motion-level Benchmarks. We evaluate our approach on two motion-level video understanding benchmarks: MotionBench [8] and FAVOR-Bench [24].

MotionBench includes 5,385 videos and 8,052 QA pairs on six motion-focused tasks: Motion Recognition (MR), Location-related Motion (LM), Action Order (AO), Repetition Count (RC), Motion-related Objects (MO), and Camera Motion (CM). We use its official Dev set for evaluation.

FAVOR-Bench’s close-ended test set contains 1,776 videos and 8,184 QA pairs spanning six dimensions: Action Sequence (AS), Holistic Action Classification (HAC), Single Action Detail (SAD), Multiple Action Details (MAD), Camera Motion (CM), and Non-Subject Motion (NSM).

Evaluation metrics. We report results as accuracy scores for each task type. ‘Overall’ accuracy is computed across all questions, reflecting total performance, while the ‘Average’ metric refers to the mean accuracy over all types, ensuring equal consideration for each type of motion problem regardless of how many samples each contains.

# 5.2. Implementation Details

We use CogVideoX-1.5-5B [31] as the VDM, and Qwen2.5-VL (3B, 7B) [2] and InternVL3 (2B, 8B) [37] as VLMs. Attention maps are extracted from the frozen VDM via 5-step DDIM sampling after 5-step inversion. This process is fully offline before VLM SFT, and the priors can be reused across VLMs and ablations. In practice, one-time extraction takes 20-30 seconds on an A100 GPU. During SFT, the vision tower, merger, and LLM backbone are trainable. We use AdamW [17] with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , ϵ = 1e−8, and weight decay 0.1, and a cosine scheduler with 0.03 warmup ratio. Learning rates are 1e−5 for the LLM and merger, and 2e−6 for the vision tower. The loss factor λ is set to 1. Training runs for one epoch with batch size 8 on eight A100 GPUs (80GB) using DeepSpeed.

Table 2. Quantitative results of FAVOR-Bench. \* denotes results we reproduced using their open-source code, while other results are taken from the original benchmark. (For more VLMs, please see supplementary materials.) 

<table><tr><td>Model</td><td>Frames</td><td>Overall</td><td>Average</td><td>AS</td><td>HAC</td><td>SAD</td><td>MAD</td><td>CM</td><td>NSM</td></tr><tr><td colspan="10">Small Size Series</td></tr><tr><td>VideoLLaMA3-2B [34]</td><td>1fps</td><td>32.98</td><td>34.61</td><td>28.97</td><td>36.60</td><td>34.90</td><td>38.01</td><td>28.56</td><td>40.62</td></tr><tr><td>Qwen2.5VL-3B* [2]</td><td>1fps</td><td>37.43</td><td>38.07</td><td>38.45</td><td>38.16</td><td>39.35</td><td>43.40</td><td>23.72</td><td>45.31</td></tr><tr><td>Qwen2.5VL-3B + MotionEnhancer (Ours)</td><td>1fps</td><td> $44.53^{\uparrow 7.10}$ </td><td> $43.94^{\uparrow 5.87}$ </td><td>45.01</td><td>51.59</td><td>44.40</td><td>48.96</td><td>28.37</td><td>45.31</td></tr><tr><td>InternVL3-2B* [37]</td><td>8</td><td>39.27</td><td>39.11</td><td>37.66</td><td>43.28</td><td>40.49</td><td>44.98</td><td>29.21</td><td>39.06</td></tr><tr><td>InternVL3-2B + MotionEnhancer (Ours)</td><td>8</td><td> $43.71^{\uparrow 4.44}$ </td><td> $45.35^{\uparrow 6.24}$ </td><td>38.53</td><td>54.57</td><td>42.60</td><td>51.78</td><td>33.02</td><td>51.56</td></tr><tr><td colspan="10">Medium Size Series</td></tr><tr><td>LLaVA-Video-7B-Qwen2 [35]</td><td>64</td><td>38.60</td><td>39.94</td><td>36.14</td><td>41.27</td><td>41.28</td><td>44.48</td><td>29.58</td><td>46.88</td></tr><tr><td>VideoChat-Flash-Qwen2-7B [14]</td><td>1fps</td><td>43.82</td><td>44.86</td><td>41.90</td><td>48.41</td><td>42.84</td><td>50.95</td><td>35.07</td><td>50.00</td></tr><tr><td>VideoLLaMA3-7B [34]</td><td>1fps</td><td>41.46</td><td>41.46</td><td>40.20</td><td>44.13</td><td>42.42</td><td>48.30</td><td>31.53</td><td>42.19</td></tr><tr><td>Qwen2.5VL-7B* [2]</td><td>1fps</td><td>42.61</td><td>42.58</td><td>41.64</td><td>47.83</td><td>44.89</td><td>47.55</td><td>28.28</td><td>45.31</td></tr><tr><td>Qwen2.5VL-7B + MotionSight* [6]</td><td>1fps</td><td>45.47</td><td>45.99</td><td>46.23</td><td>51.59</td><td>45.01</td><td>50.04</td><td>29.95</td><td>53.12</td></tr><tr><td>Qwen2.5VL-7B + MotionEnhancer (Ours)</td><td>1fps</td><td> $46.88^{\uparrow 4.27}$ </td><td> $47.01^{\uparrow 4.43}$ </td><td>49.34</td><td>50.62</td><td>45.37</td><td>53.20</td><td>30.42</td><td>53.12</td></tr><tr><td>InternVL3-8B* [37]</td><td>8</td><td>45.82</td><td>46.35</td><td>45.39</td><td>48.54</td><td>47.59</td><td>51.45</td><td>33.58</td><td>51.56</td></tr><tr><td>InternVL3-8B + MotionEnhancer (Ours)</td><td>8</td><td> $48.94^{\uparrow 3.12}$ </td><td> $49.25^{\uparrow 2.90}$ </td><td>47.17</td><td>57.11</td><td>46.57</td><td>56.35</td><td>36.74</td><td>51.56</td></tr><tr><td colspan="10">Large Size Series</td></tr><tr><td>LLaVA-Video-72B-Qwen2 [35]</td><td>64</td><td>46.08</td><td>46.49</td><td>48.35</td><td>47.50</td><td>45.25</td><td>51.70</td><td>33.02</td><td>53.12</td></tr><tr><td>Qwen2.5-VL-72B* [2]</td><td>1fps</td><td>48.14</td><td>48.17</td><td>50.28</td><td>46.98</td><td>48.13</td><td>51.78</td><td>40.28</td><td>51.56</td></tr></table>

Table 3. Ablation study of MHS and MTTI using Qwen2.5VL-7B. These results confirm that MHS and MTTI are complementary, and combining them yields the highest gains. 

<table><tr><td rowspan="2">Idx</td><td colspan="2">Variants</td><td colspan="2">MotionBench</td><td colspan="2">FAVOR-Bench</td></tr><tr><td>MHS</td><td>MTTI</td><td>Over.</td><td>Aver.</td><td>Over.</td><td>Aver.</td></tr><tr><td>1</td><td>✗</td><td>✗</td><td>54.83</td><td>51.51</td><td>44.83</td><td>44.54</td></tr><tr><td>2</td><td>√</td><td>✗</td><td>56.60</td><td>52.51</td><td>46.65</td><td>46.55</td></tr><tr><td>3</td><td>✗</td><td>√</td><td>55.80</td><td>51.31</td><td>45.47</td><td>45.99</td></tr><tr><td>4</td><td>√</td><td>√</td><td>57.04</td><td>52.92</td><td>46.88</td><td>47.01</td></tr></table>

# 5.3. Comparison with State-of-the-art Method

Baselines. Our compared methods are categorized based on model size: small, medium and large size series, including popular open-source VLMs and improvements on them. We perform comparison in small and medium series, while showing the larger series as the upper limit of this task.

MotionBench. Tab. 3 presents the evaluation results on MotionBench. Incorporating MotionEnhancer consistently yields substantial improvements for Qwen2.5-VL across both 3B and 7B variants. For the 3B backbone, MotionEnhancer elevates the category average by 3.1%, with pro-

nounced gains in motion-relevant metrics. For the 7B backbone, the enhancement effect becomes more significant, with category average increasing by 4.6%, and MR and CM improving by 4.4% and 11.7%, respectively. We observe similar improvements with InternVL3 backbone, with an overall accuracy of 1.51% for the 2B backbone and 2.81% for the 8B backbone. These results indicate that MotionEnhancer effectively augments temporal motion modeling.

FAVOR-Bench. Tab. 4 reports the results on FAVOR-Bench, further validating the effectiveness of MotionEnhancer for fine-grained motion understanding. Across backbone sizes, Qwen2.5-VL-3B and 7B derive consistent and substantial gains from MotionEnhancer, and surpass other models in their respective series. On the 3B backbone, MotionEnhancer improves category average by 5.6% and overall score by 7.1%, with remarkable gains in HAC and MAD. On the 7B backbone, category average rises by 4.5% and overall score by 4.2%, with consistent gains in AS, HAC, and MAD. We observe similar improvements with InternVL3 backbone, with an overall accuracy of 4.44% for the 2B backbone and 3.12% for the 8B backbone.

Notably, Qwen2.5-VL-3B+MotionEnhancer surpasses Qwen2.5-VL-7B on both benchmarks, while Qwen2.5- VL-7B+MotionEnhancer achieves performance comparable to Qwen2.5-VL-72B and other large-scale baselines. These results highlight that MotionEnhancer scales robustly across backbone sizes, enabling compact models to attain performance levels typically associated with substantially larger architectures.

![](images/fdf76dd6bf2b18c5081ff9f0501f845ff929db1c861bbd1b57b77152f45ff6fb.jpg)

<details>
<summary>text_image</summary>

Question: What is the sequence of actions performed by the person's hands?
A. Pulls the grill first, then uses the spray bottle
B. Waves the towel first, then sprays with the bottle
C. Uses the spray bottle first, then pulls the grill
D. Uses the spray bottle with both hands simultaneously
Qwen2.5VL-7B: B. Waves the towel first, then sprays with the bottle
Qwen2.5VL-7B + MotionEnhancer: A. Pulls the grill first, then uses the spray bottle
Question: What happens to the basketball after it goes through the hoop?
A. It is caught by the man in the black shirt
B. It drops to the left side of the screen
C. It rolls to the right side of the screen
D. It bounces off the hoop and moves upwards
Qwen2.5VL-7B: A. It is caught by the man in the black shirt
Qwen2.5VL-7B + MotionEnhancer: B. It drops to the left side of the screen
</details>

![](images/dbfdf1ebeacbf46bbe99d37ba1bd56b55a019e60cd2342da8f0e636730a030cb.jpg)

<details>
<summary>text_image</summary>

Question: What tiny action did the person in red clothes do before picking up the two yellow gear parts?
A. Swap the positions of two gear parts C. Slapping gear parts
B. Swap the positions of the two spanners D. Tapping gear parts

Qwen2.5VL-7B: C. Slapping gear parts

Qwen2.5VL-7B + MotionEnhancer: A. Swap the positions of two gear parts

Question: How does the person with the backpack move?
A. Towards the left, across the right side of the screen C. Randomly moves back and forth
B. Stays in a stationary position D. Towards the right, away from the group

Qwen2.5VL-7B: A. Towards the left, across the right side of the screen
Qwen2.5VL-7B + MotionEnhancer: D. Towards the right, away from the group
</details>

(A) Motion Understanding Examples

![](images/aa7ed0deb64a407e3daea609c45c59884cf03c61e53802691e30d308adbfd059.jpg)

<details>
<summary>text_image</summary>

Token Selection Result: In the center of the screen is a woman wearing a dark top. She first bows her head slightly, then turns to the left and bends over. Afterward, she straightens up and looks into the camera to explain. Following this, she turns to the right, bends down, and places an item from her right hand into the cabinet.
VLM Attention Maps
VDM's Text Token Attention Maps
w/o
Motion
Enhancer
Bow
with
Motion
Enhancer
Place
(B) Attention Map and Text Selection Examples
</details>

![](images/82c40a8e141bc6343bb24bbb906a99395ab309eeacc0939ab9320ac5ca418351.jpg)

<details>
<summary>text_image</summary>

Description: The main subject of the image is a little boy. He is facing towards the right side of the frame, holding an egg in his left hand and a paintbrush in his right hand, which he is using to apply paint to it. Then, he opens his mouth wide and closes it with force, puts the brush down to dip it in the paint, and then picks it up to continue painting.
Token: open
Description: Two men are standing side by side facing the camera, with the man on the left wearing a white shirt and the man on the right wearing a blue shirt; both are facing towards the right of the screen. Then, the cameras zooms in, focusing on the face of the man in the blue shirt at the center of the screen. Initially, his eyes are looking towards the right of the screen, then he glances to his own right, and subsequently opens his mouth to take a breath.
Token: looking
(C) Challenging Cases of VDM's Attention Maps
</details>

Figure 3. Qualitative examples of MotionEnhancer. (More examples can be found in supplementary materials.)

# 5.4. Ablation Study

Table 3 presents the ablation results of MHS and MTTI on Qwen2.5VL-7B. The baseline (Index 1) includes neither module, and is trained only on our 25k dataset, using average pooling for both motion heads and text tokens during alignment. It achieves Overall/Average scores of 54.83/51.51 on MotionBench and 44.83/44.54 on FAVOR-Bench. Adding only MHS (Index 2) brings significant improvements: +1.77 Overall on MotionBench and +1.82 Overall on FAVOR-Bench, showing that selecting motionrelevant heads effectively enhances temporal motion modeling. Applying only MTTI (Index 3) also consistently improves over the baseline, indicating that aligning with motion-related text tokens benefits motion understanding. However, its gain is smaller than that of MHS, since MTTI benefits from the prior motion filtering provided by MHS. Combining both MHS and MTTI (Index 4) achieves the best performance, with Overall/Average scores of 57.04/52.92 on MotionBench and 46.88/47.01 on FAVOR-Bench. These results confirm that MHS and MTTI are complementary, and together yield the largest gains.

# 5.5. Qualitative Examples and Limitations

Qualitative Examples. We provide qualitative examples of motion understanding, attention, and challenging cases. In Fig. 3(A), MotionEnhancer helps better answer motionrelated questions. Fig. 3(B) shows that MHS improves motion-focused attention, while MTTI effectively filters out irrelevant tokens (see Sec. 4.4). After alignment, the VLM attends more to motion cues and related objects.

Limitations and future improvements. In practice, after training, we observe fewer wrong-to-correct cases for videos in which the main subject fills the frame and remains static. To understand the reason, we visualize the VDM attention maps of these challenging videos in Fig. 3(C), and find that the attention becomes diffuse and less focused, failing to highlight specific objects or motions. This limitation stems from a bias inherent in the VDM training data. Since the VDM is mainly trained on videos containing small objects, it struggles to model large, static subjects that occupy the entire frame. Future work could explore more refined motion extraction methods and introduce data preprocessing strategies to mitigate this bias.

# 6. Conclusion

In this work, we introduce MotionEnhancer, which leverages motion priors distilled from a powerful VDM as auxiliary supervision to enhance VLM’s motion understanding capability via attention alignment. Extensive experiments show that MotionEnhancer consistently improves over state-of-the-art VLMs on two motion-level video understanding benchmarks, especially on motion-related metrics, demonstrating MotionEnhancer provides a scalable solution for motion understanding without requiring extra training parameters and architectural modifications. Moreover, motion latents extracted from large-scale videos via VDMs can serve as a motion-aware pretraining signal for downstream tasks that are highly sensitive to temporal dynamics (e.g., robotic arm grasping), offering strong VLM initialization that improves sample efficiency and temporal generalization. We leave this for future exploration.

# References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.   
[4] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024.   
[5] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.   
[6] Yipeng Du, Tiehan Fan, Kepan Nan, Rui Xie, Penghao Zhou, Xiang Li, Jian Yang, Zhenheng Yang, and Ying Tai. Motionsight: Boosting fine-grained motion understanding in multimodal llms. arXiv preprint arXiv:2506.01674, 2025.   
[7] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024.   
[8] Wenyi Hong, Yean Cheng, Zhuoyi Yang, Weihan Wang, Lefan Wang, Xiaotao Gu, Shiyu Huang, Yuxiao Dong, and Jie Tang. Motionbench: Benchmarking and improving fine-grained video motion understanding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8450–8460, 2025.   
[9] Chen Jin, Ryutaro Tanno, Amrutha Saseendran, Tom Diethe, and Philip Teare. Diffusion instruction tuning. arXiv preprint arXiv:2502.06814, 2025.   
[10] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.   
[11] Jie Lei, Licheng Yu, Tamara Berg, and Mohit Bansal. Tvqa+: Spatio-temporal grounding for video question answering. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 8211–8225, 2020.   
[12] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.

[13] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730– 19742. PMLR, 2023.   
[14] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024.   
[15] Yicong Li, Junbin Xiao, Chun Feng, Xiang Wang, and Tat-Seng Chua. Discovering spatio-temporal rationales for video question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13869– 13878, 2023.   
[16] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024.   
[17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[18] Yujie Lu, Yale Song, William Wang, Lorenzo Torresani, and Tushar Nagarajan. Vited: Video temporal evidence distillation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8501–8511, 2025.   
[19] Shijie Ma, Yuying Ge, Teng Wang, Yuxin Guo, Yixiao Ge, and Ying Shan. Genhancer: Imperfect generative models are secretly strong vision-centric enhancers. arXiv preprint arXiv:2503.19480, 2025.   
[20] Yue Ma, Yulong Liu, Qiyuan Zhu, Ayden Yang, Kunyu Feng, Xinhua Zhang, Zhifeng Li, Sirui Han, Chenyang Qi, and Qifeng Chen. Follow-your-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning. arXiv preprint arXiv:2506.05207, 2025.   
[21] Effrosyni Mavroudi and Rene Vidal. Weakly-supervised ´ generation and grounding of visual descriptions with conditional generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15544–15554, 2022.   
[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image ¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
[23] Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Ture. What the daam: Interpreting stable dif- ¨ fusion using cross attention. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5644–5659, 2023.   
[24] Chongjun Tu, Lin Zhang, Pengtao Chen, Peng Ye, Xianfang Zeng, Wei Cheng, Gang Yu, and Tao Chen. Favor-bench: A comprehensive benchmark for fine-grained video motion understanding. arXiv preprint arXiv:2503.14935, 2025.   
[25] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia

Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[26] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
[27] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024.   
[28] Wenxuan Wang, Quan Sun, Fan Zhang, Yepeng Tang, Jing Liu, and Xinlong Wang. Diffusion feedback helps clip see better. arXiv preprint arXiv:2407.20171, 2024.   
[29] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.   
[30] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024.   
[31] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.   
[32] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.   
[33] Hidir Yesiltepe and Pinar Yanardag. Dynamic view synthesis as an inverse problem. arXiv preprint arXiv:2506.08004, 2025.   
[34] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.   
[35] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.   
[36] Zhenghao Zhang, Junchao Liao, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2063–2073, 2025.   
[37] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

# MotionEnhancer: Leveraging Video Diffusion for Motion-Enhanced Vision-Language Models

Supplementary Material

# 1. Preliminaries

# 1.1. DDIM Framework

Denoising Diffusion Implicit Models (DDIM) [21] provide an efficient and flexible sampling framework for diffusion models. Traditional diffusion models, such as Denoising Diffusion Probabilistic Models (DDPM) [9], rely on a stochastic generation process, where random noise is gradually removed through a series of probabilistic denoising steps. This approach, while effective, often requires a large number of iterations to generate high-quality samples, resulting in high computational cost. DDIM, in contrast, introduces a deterministic alternative to the sampling process. It leverages a non-Markovian forward process that adds Gaussian noise to the original data $\mathbf { x } _ { \mathrm { 0 } }$ according to a predefined noise schedule $\alpha _ { t }$ . Specifically, the forward process can be formulated as:

$$
\mathbf {x} _ {t} = \sqrt {\alpha_ {t}} \mathbf {x} _ {0} + \sqrt {1 - \alpha_ {t}} \epsilon , \quad \epsilon \sim \mathcal {N} (0, \mathbf {I}), \tag {1}
$$

where $\alpha _ { t }$ decreases monotonically over time, gradually corrupting the data.

The key innovation of DDIM lies in its reverse process. Instead of sampling new noise at every step, DDIM uses a deterministic mapping to reconstruct the data. The denoising step is given by:

$$
\mathbf {x} _ {t - 1} = \sqrt {\alpha_ {t - 1}} \hat {\mathbf {x}} _ {0} (\mathbf {x} _ {t}) + \sqrt {1 - \alpha_ {t - 1}} \epsilon_ {\theta} (\mathbf {x} _ {t}, t), (2)
$$

where $\hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } )$ is the predicted clean data:

$$
\hat {\mathbf {x}} _ {0} (\mathbf {x} _ {t}) = \frac {\mathbf {x} _ {t} - \sqrt {1 - \alpha_ {t}} \epsilon_ {\theta} (\mathbf {x} _ {t} , t)}{\sqrt {\alpha_ {t}}}. \tag {3}
$$

By removing the stochasticity from the reverse process, DDIM enables faster and more controllable sampling. This deterministic formulation allows the model to generate high-quality samples with significantly fewer steps compared to DDPM, while preserving the overall data structure and ensuring boundary consistency. As a result, DDIM has emerged as a widely favored option in diffusion-based generative modeling, especially within applications where both efficiency and fidelity are critical requirements.

# 1.2. DDIM Inversion

DDIM inversion [18] extends the DDIM framework by enabling the mapping of real data $\mathbf { x } _ { \mathrm { 0 } }$ back into its corresponding latent noise $\mathbf { x } _ { T }$ . This process is achieved through an iterative application of the reversed DDIM denoising step, effectively tracing the generative trajectory in the opposite direction.

Formally, starting from the observed data x0, the inversion procedure sequentially estimates the intermediate noisy representations $\mathbf { x } _ { t }$ for $t = 0 , \ldots , T - 1$ via:

$$
\mathbf {x} _ {t + 1} = \sqrt {\alpha_ {t + 1}} \hat {\mathbf {x}} _ {0} (\mathbf {x} _ {t}) + \sqrt {1 - \alpha_ {t + 1}} \epsilon_ {\theta} (\mathbf {x} _ {t}, t). \tag {4}
$$

After T iterations, this process yields a latent code $\mathbf { x } _ { T } =$ ${ \mathcal { G } } ^ { - 1 } ( \mathbf { x } _ { 0 } )$ , which encodes the original data within the noise space of the diffusion model.

A notable feature of DDIM inversion is its path determinism: if the noise prediction function $\epsilon _ { \theta }$ remains fixed, the entire sequence is reversible. In other words, if one starts from $\mathbf { x } _ { 0 } ,$ , performs the inversion step, and then follows with the standard forward generative process, the original input will be reconstructed exactly:

$$
\mathbf {x} _ {0} \xrightarrow {\text { invert }} \mathbf {x} _ {T} \xrightarrow {\text { reconstruct }} \mathbf {x} _ {0}. \tag {5}
$$

This deterministic and bijective mapping ensures that each data point has a unique latent representation and that the model’s internal states can be precisely analyzed.

Such inversion capability is especially valuable for investigating the internal mechanisms of diffusion models, enabling controllable editing, semantic manipulation, and deeper interpretability. It provides a systematic approach to relate observable samples with their latent origins, thus opening avenues for fine-grained analysis and intervention in generative modeling tasks.

# 1.3. Introduction of Our Used Models

We utilize CogVideoX-5B as our Video Diffusion Model (VDM) and Qwen2.5-VL (3B and 7B) and InternVL3 (2B and 8B) as our Vision-Language Models (VLM). Below is a concise overview of these models, covering their architectural designs and operational workflows.

CogVideoX: A DiT-based Video Generative Model. CogVideoX [26] is a diffusion-transformer model [19] for text-to-video generation that tackles the twin challenges of high-resolution fidelity and long-range temporal coherence in a single, end-to-end pipeline. Its first stage is a 3D causal VAE that compresses both spatial and temporal axes with an 8×8×4 reduction factor, turning seconds of raw pixels into a compact latent cube while preserving fine detail and motion continuity; the causal design guarantees that decoding can proceed frame-by-frame for streaming applications. The latent volume is patched into spatio-temporal tokens and simply concatenated with T5 text embeddings [20], eliminating cumbersome cross-attention modules and allowing the same transformer blocks to process both modalities. Inside each transformer layer, 3D full attention lets every patch attend to any other patch in its causal window, capturing large motions without drift. Separate expert adaptive LayerNorm branches scale and shift visual and textual features before they merge, aligning modalities without extra parameters.

Qwen2.5-VL: Flagship VLM in Qwen Series. Qwen2.5-VL [2] is a VLM that ingests images and text at their native resolution without ever cropping them into fixed grids. At its core is a redesigned Vision Transformer (ViT) [4] trained from scratch, which incorporates window attention in most layers—with only four layers using full attention—to reduce computational complexity from quadratic to linear relative to input patches, enabling native handling of dynamic resolutions without normalization artifacts. The resulting dense visual tokens are streamed into a lightweight vision-language merger that groups neighboring patches and projects the ensemble through a two-layer MLP, yielding a compact set of vision embeddings that align seamlessly with the text token space while retaining fine-grained spatial cues. For video, Qwen2.5- VL introduces Multimodal Rotary Position Embedding (MRoPE) tied to absolute timestamps: temporal position IDs are explicitly synchronized with real-world clock time, so the model perceives frame-rate variations naturally and reasons about event order without external preprocessing.

InternVL3: Native Multimodal Pre-training at Scale. InternVL3 [30] departs from the dominant “language-first, vision-second” paradigm by performing native multimodal pre-training from the outset: a single unified stage interleaves large-scale text corpora with diverse vision–language data, allowing linguistic and visual capacities to co-evolve without later alignment. Architecturally, the model keeps the clean ViT–MLP–LLM stack: an image-agnostic ViT feeds a lightweight two-layer projector that directly maps raw visual tokens into the language model’s embedding space, eliminating resolution-specific modules or fixed-size crops. To gracefully accommodate long visual contexts, InternVL3 adopts Variable Visual Position Encoding (V2PE) [8] that modulates positional increments for visual tokens, keeping the overall sequence length within the LLM’s native window. Operationally, all components—vision encoder, projector, and language backbone—are jointly optimized on the combined corpus; subsequent post-training refines conversational quality through supervised fine-tuning followed by Mixed Preference Optimization (MPO) [24], a unified objective that blends preference, quality, and generation losses. Test-time computation is amplified by a visual process-supervised critic that selects the best among multiple decoded chains, pushing reasoning quality without enlarging the model.

Algorithm 1: MotionEnhancer Pipeline   
Input: Video V; Question I; Target response s;
Pretrained VLM; Pretrained VDM
Output: Fine-tuned VLM; Total loss $L_{total}$ Step 1: VDM-based Attention Extraction $z_{vision} \leftarrow VDM.VAE.encode(V)$ ; // Encode video, $z_{vision} \in R^{F \times H \times W}$ $z_{noise} \leftarrow DDIM_Inversion(z_{vision}, step = 5)$ ;

// Obtain DDIM-inverted noise $z_{text} \leftarrow VDM.TextEncoder(I)$ ; // Encode text $z_{mm} \leftarrow [z_{text}, z_{noise}]$ ; // Form multimodal sequence

foreach DDIM denoising step do

Compute attention: $A_{mm} \leftarrow Softmax\left(\frac{Q_{mm}K_{mm}^{T}}{\sqrt{d}}\right)$ ;

Average $A_{mm}$ for all layers;

Store $A_{mm}$ for all heads;

Step 2: Motion-sensitive Head Selection

foreach head h in VDM do

Extract vision-to-vision attention $A_{v2v}^{(h)}$ from $A_{mm}$ ; // $[S \times S]$ , $S = H \times W$ Compute DFC, TCS, and DSR

score h = Norm(DFC) + Norm(TCS) + Norm(DSR) ; // Compute head score $A_{mm}^{H_m} \leftarrow Avg(\{A_{mm}^{(h)}\}_{h \in H_m})$ ; // Select top-50% heads $H_m$ , and aggregate motion heads

Step 3: Motion-salient Text Token Identification $A_{t2v} \leftarrow extract text-to-vision region(A_{mm}^{H_m})$ ;

// Extract text-to-vision attention

foreach text token t do $A_{t,t2f} \leftarrow Avg_{(h,w)}A_{t2v}[t, f, h, w]$ ; // Pool over spatial dimensions $m_t \leftarrow Mean_f(A_{t,t2f})$ ; // Compute mean $d_t \leftarrow \frac{1}{F-1}\sum_{f=1}^{F-1}|A_{t,t2f}[f+1] - A_{t,t2f}[f]|$ ;

// Compute first-order diff mean $MS(t) = m_t + d_t$ ; // Motion score

Select top-50% tokens $T_m$ by $MS(t)$ $A_{VDM} \leftarrow A_{t2v}[T_m, :]$ ; // Obtain VDM motion prior

Step 4: VLM Attention Extraction

Extract VLM attention maps $A_{VLM,raw}$ for $(V, I, s)$ Average pool across heads/layers and obtain $A_{VLM}$ $A_{VLM} \leftarrow A_{VLM}[T_m, :]$ ; // Select same tokens

Interpolate $A_{VLM}$ to match $A_{VDM}$ Step 5: Supervised Fine-tuning with Attention Alignment $A_{VLM,aligned} \leftarrow MLP(A_{VLM})$ // Align VLM attention $L_{MSE} = ||A_{VLM,aligned} - A_{VDM}||_2$ ; // Compute attention alignment loss $L_{AR} = -\sum_{i=1}^{|s|} log P(s[i]|V, I, \theta, s[1 : i-1])$ ;

// Compute auto-regressive loss $L_{total} = L_{AR} + \lambda L_{MSE}$ ; // Compute total loss

Update trainable parameters in VLM with $L_{total}$

# 2. Algorithm

The MotionEnhancer pipeline is shown in Algorithm 1. (1) Step 1: VDM-based Attention Extraction. The input video is encoded into latent representations using a pretrained VDM. During the DDIM inversion and denoising process, multi-head attention maps are extracted from the VDM across all layers. (2) Step 2: Motion-sensitive Head Selection. For each attention head, motion relevance is quantitatively evaluated using the Diagonal Focus Coefficient, Temporal Continuity Score, and Diagonal Saliency Ratio. The top-50% heads most sensitive to temporal motion are selected, and their attention maps are aggregated. (3) Step 3: Motion-salient Text Token Identification. The aggregated attention maps are analyzed to compute temporal attention statistics for each text token. Tokens exhibiting high temporal attention and significant inter-frame variation are selected as motion-salient text tokens. The VDM attention maps filtered by both MHS and MTTI are used to construct motion priors, focusing on motion-relevant regions and tokens. (4) Step 4: VLM Attention Extraction. For the same video-text input, attention maps are extracted from the VLM and pooled across heads and layers. The attention corresponding to the selected motion-salient tokens is retained and interpolated if necessary. (5) Step 5: Supervised Fine-tuning with Attention Alignment During fine-tuning, the VLM is optimized by minimizing the L2 distance between its own attention maps and the motion priors from the VDM, while also being supervised with the standard auto-regressive loss for video question answering. All trainable parameters in the VLM are updated with respect to the combined loss, enabling the model to acquire enhanced motion understanding without modifying its architecture or adding new trainable modules.

# 3. Detailed Theoretical Analysis

This section provides a comprehensive theoretical extension from the main text, offering deeper insights into the distributional properties of VLMs and VDMs that form the foundation of MotionEnhancer.

# 3.1. Preliminaries and Notation

Let a video be denoted as

$$
V = (V _ {1}, V _ {2}, \dots , V _ {F}), \tag {6}
$$

where $V _ { f }$ is the f -th frame and F is the number of frames. Each frame is discretized into spatial locations $s \in S ;$ w e denote the visual feature at location s and frame f as $v _ { s , f }$ .

A VLM is trained with an autoregressive objective:

$$
\mathcal {L} _ {\mathrm{AR}} = - \sum_ {i} \log p _ {\theta} (r _ {i} \mid V, r _ {<   i}), \tag {7}
$$

where θ denotes the VLM parameters and $r _ { i }$ is the i-th response token.

Internally, the VLM produces text-to-video attention scores. For a token t, the attention weight at location $( s , f )$ can be interpreted as a discriminative conditional distribution:

$$
A ^ {\mathrm{VLM}} (t, s, f) \approx p _ {\theta} \big (t \mid v _ {s, f}, r _ {<   i} \big), \tag {8}
$$

which reflects a recognition-style distribution $p _ { \theta } ( t | V )$ learned under the autoregressive training paradigm.

By contrast, motion understanding requires discovering evidence in the video that supports a motion-related token t and its associated answer a. As argued in Sec. 3, this can be expressed via latent evidence E and its factorization over semantic units $t \in T ( q , a )$ as:

$$
p ^ {\star} (a \mid V, q) = \sum_ {E} p ^ {\star} (a \mid E, q)   p ^ {\star} (E \mid V, q), \tag {9}
$$

$$
E = \{E _ {t} \mid t \in T (q, a) \}, \tag {10}
$$

$$
p ^ {\star} (E \mid V, q) = \prod_ {t \in T (q, a)} p ^ {\star} (E _ {t} \mid V, t, q), \tag {11}
$$

where $E _ { t }$ identifies the spatiotemporal regions that visually realize token t.

Within each $E _ { t } .$ , the relevance of a location $( s , f )$ can be represented by a concept-conditioned visual likelihood:

$$
r _ {t} (s, f) = p ^ {\star} (v _ {s, f} \mid t), \tag {12}
$$

which naturally leads to an evidence-seeking distribution $p ^ { \star } ( V \mid t ) ;$ : “given concept t, where and how is it visually realized over space and time?”

For motion-related tokens (e.g., verbs, motion modifiers), the latent evidence must encode frame-to-frame transitions, which, at the feature level, can be summarized by a motion magnitude:

$$
\text { Motion } (s, f) \propto \| V _ {f + 1} (s) - V _ {f} (s) \|. \tag {13}
$$

# 3.2. Distributional Mismatch in More Detail

We now formalize the mismatch between $p _ { \theta } ( t \mid V )$ (what the VLM is trained on) and $p ^ { \star } ( V \mid t )$ (what motion understanding requires). Using Bayes’ rule at the level of perlocation features:

$$
p ^ {\star} (t \mid v _ {s, f}) = \frac {p ^ {\star} (v _ {s , f} \mid t) p ^ {\star} (t)}{p ^ {\star} (v _ {s , f})}. \tag {14}
$$

If the VLM were trained with sufficiently rich supervision and under a well-specified model, $A ^ { \mathrm { V L M } } ( t , s , f )$ could in principle approximate $\boldsymbol { p } ^ { \star } ( t \mid \boldsymbol { v } _ { s , f } )$ . However, in realistic settings, the supervision usually consists of high-level QA or captions rather than dense, per-location labels. And the set of questions and responses $( q , a )$ is highly imbalanced: many queries can be answered from a static context without modeling temporal changes. Also, the model capacity and optimization are shaped to maximize accuracy, not to invert the distribution to recover $\boldsymbol { p } ^ { \star } ( v _ { s , f } \mid t )$ .

Hence, a VLM can achieve high training and evaluation performance by relying mostly on static appearance cues. Formally, decompose the visual feature as:

$$
v _ {s, f} = v _ {s, f} ^ {\text { stat }} + v _ {s, f} ^ {\text { dyn }}, \tag {15}
$$

where $v ^ { \mathrm { s t a t } }$ captures static background and identity cues, and $v ^ { \mathrm { d y n } }$ captures temporal variations. If most training questions can be answered using $v ^ { \mathrm { s t a t } }$ alone, then gradients of $\mathcal { L } _ { \mathrm { A R } }$ with respect to $v ^ { \mathrm { d y n } }$ are small or sparse, and the model converges to a solution where:

$$
p _ {\theta} (t \mid v _ {s, f}) \approx p _ {\theta} (t \mid v _ {s, f} ^ {\text { stat }}), \tag {16}
$$

i.e., the conditional depends only weakly on temporal $\mathrm { { d y } \mathrm { { - } } }$ namics.

On the other hand, motion understanding requires $\mathbf { e s - }$ timating where and when motion-related evidence occurs, which is precisely governed by $\boldsymbol { p } ^ { \star } ( v _ { s , f } \mid t )$ and its temporal variation across $f .$ In particular, the expectation

$$
\mathbb {E} \big [ \mathrm{Motion} (s, f) \mid t \big ] \propto \mathbb {E} \Big [ \left| \left| V _ {f + 1} (s) - V _ {f} (s) \right| \right| \Big | t \Big ] \tag {17}
$$

is a functional of the evidence-seeking distribution $p ^ { \star } ( V \mid$ t), not of the recognition distribution $p ^ { \star } ( t \mid V )$ .

Thus, unless the training signal explicitly forces the VLM to approximate or invert $p ^ { \star } ( V \mid t )$ , there is a structural mismatch: the model is optimized for predicting tokens given videos, while fine-grained motion reasoning requires discovering the video given a concept. This explains why VLMs exhibit appearance bias and weak temporal sensitivity, as observed empirically.

# 3.3. VDM Attention as Evidence-Seeking Posterior

Video diffusion models (VDMs) are trained to generate videos conditioned on text. At the denoising step $k ,$ the model receives noisy latent video $z _ { k }$ and text tokens $t ,$ and outputs a cleaner latent that approximates the true video latent $z _ { \mathrm { 0 } }$ . The cross-attention at this step is computed as:

$$
A ^ {\mathrm{VDM}} (t, s, f) = \operatorname{Softmax} \left(\frac {Q (t) K (s , f) ^ {\top}}{\sqrt {d}}\right), \tag {18}
$$

where $Q ( t )$ is the query for token t and $K ( s , f )$ is the key derived from the latent at location $( s , f )$ . Aggregating across layers and timesteps, as described in the main paper, yields a stable attention estimate.

Under mild assumptions (e.g., linear attention maps and sufficiently expressive $Q / K$ projections), this attention can be interpreted as a normalized relevance score between t and $v _ { s , f }$ conditioned on the current noisy latent:

$$
A ^ {\mathrm{VDM}} (t, s, f) \approx p _ {\phi} (v _ {s, f} \mid t, z _ {k}), \tag {19}
$$

where $\phi$ denotes VDM parameters. This resembles the concept-conditioned likelihood $p ^ { \star } ( v _ { s , f } | t )$ required for motion reasoning.

Two properties make VDM attention particularly suited as a motion prior: Evidence-seeking nature. Since the model must generate video content that visually realizes the text token t, the attention highlights locations where the semantics of t should appear. This approximates the evidenceseeking distribution $p ^ { \star } ( V | t )$ : regions receiving higher attention are those where the generated video is more likely to exhibit the concept. Motion calibration. The denoising objective penalizes reconstruction errors, which are larger in regions with strong motion (large $\| V _ { f + 1 } ( s ) - V _ { f } ( s ) \| )$ because these regions are harder to predict from noisy latents. Consequently, gradients concentrate on dynamic areas, making cross-attention for motion-related tokens sensitive to temporal variations. Locations with higher motion magnitude receive higher or more fluctuating attention across timesteps, aligning with the notion of motion evidence.

In the idealized limit where the VDM perfectly models the true data distribution, $A ^ { \mathrm { V D M } } ( t , s , f )$ converges to a calibrated approximation of $p ^ { \star } ( v _ { s , f } \mid t )$ . Even in practice, it serves as a high-quality, motion-aware prior that complements the VLM’s appearance-focused attention.

# 3.4. Attention Alignment as Approximate Posterior Matching

MotionEnhancer introduces an auxiliary MSE loss between VDM and VLM attention:

$$
\mathcal {L} _ {\mathrm{MSE}} = \left\| \text { Aligner } (A ^ {\mathrm{VLM}}) - A ^ {\mathrm{VDM}} \right\| _ {2} ^ {2}, \tag {20}
$$

and optimizes

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\mathrm{AR}} + \lambda \mathcal {L} _ {\mathrm{MSE}}. \tag {21}
$$

For a given motion-salient token t, we treat each attention row as a discrete distribution over locations:

$$
\pi_ {\theta} ^ {\mathrm{VLM}} (s, f \mid t) := A ^ {\mathrm{VLM}} (t, s, f), \tag {22}
$$

$$
\pi_ {\phi} ^ {\mathrm{VDM}} (s, f \mid t) := A ^ {\mathrm{VDM}} (t, s, f). \tag {23}
$$

Ignoring the Aligner for a moment (or assuming it is near identity), we may view $\mathcal { L } _ { \mathrm { M S E } }$ as minimizing an $L ^ { \bar { 2 } }$ distance between these distributions. In expectation over data, we approximately solve:

$$
\min _ {\theta} \mathbb {E} _ {(V, q, a)} \left[ - \log p _ {\theta} (a \mid V, q) \right] +
$$

$$
\lambda \mathbb {E} _ {t \in T _ {\mathrm{mot}} (q, a)} \left[ \left| \left| \pi_ {\theta} ^ {\mathrm{VLM}} (\cdot | t) - \pi_ {\phi} ^ {\mathrm{VDM}} (\cdot | t) \right| \right| _ {2} ^ {2} \right]. \tag {24}
$$

This expanded objective has two important interpretations: Posterior matching in attention space. If we interpret $\pi _ { \theta } ^ { \mathrm { V L M } } ( \cdot \mid t )$ as an approximate posterior over evidence locations for token t, and $\pi _ { \phi } ^ { \mathrm { V D M } } ( { \bar { \cdot } } \mid t )$ as an approximate concept-conditioned distribution from the generative model, then $\mathcal { L } _ { \mathrm { M S E } }$ acts to push the VLM distribution toward the motion-sensitive evidence-seeking distribution encoded by the VDM. This reduces the discrepancy between the recognition distribution (where a VLM can rely heavily on static cues) and the generative distribution that better respects motion. Constrained optimization and regularization. The optimization in Eq. 24 can equivalently be viewed as:

$$
\min _ {\theta} \quad \mathbb {E} \left[ - \log p _ {\theta} (a \mid V, q) \right] \tag {25}
$$

$$
\mathrm{s.t.} \mathbb {E} \big [ \| \pi_ {\theta} ^ {\mathrm{VLM}} - \pi_ {\phi} ^ {\mathrm{VDM}} \| _ {2} ^ {2} \big ] \leq \epsilon (\lambda),
$$

where ϵ(λ) decreases as λ increases. Thus, the feasible set of VLM parameters is restricted to those producing attention maps that are compatible with motion priors distilled from the VDM.

In practice, the Aligner is implemented as a small MLP. Its limited capacity enables mild, smooth transformations (e.g., rescaling or slight warping) to account for architectural differences between VLM and VDM, while preventing degenerate solutions in which the VLM attention remains unchanged and the Aligner absorbs the full discrepancy. In this sense, the Aligner functions as a preconditioner rather than a substitute for matching attention distributions.

# 3.5. MHS and MTTI as Projection onto a Motion Subspace

Both MHS and MTTI modules act as selection operators that isolate the motion-relevant components of the VDM attention. Let A denote the full text-to-video attention tensor from the VDM.

Head-level projection. MHS selects a subset of heads exhibiting strong temporal structure. At a high level, this can be written as a projection:

$$
A _ {\text { head }} = \Pi_ {\text { head }} (\mathcal {A}), \tag {26}
$$

where $\Pi _ { \mathrm { h e a d } }$ keeps only motion-sensitive heads and discards others.

Token-level projection. Similarly, MTTI selects text tokens whose attention varies across frames. This yields a second projection:

$$
A _ {\mathrm{mot}} = \Pi_ {\mathrm{token}} (A _ {\mathrm{head}}). \tag {27}
$$

Overall effect. Together, MHS and MTTI produce a compact motion-focused attention map:

$$
A _ {\mathrm{mot}} = \Pi_ {\mathrm{token}} \big (\Pi_ {\mathrm{head}} (\mathcal {A}) \big), \tag {28}
$$

Table 1. The experimental system and hardware setups. 

<table><tr><td colspan="2">System &amp; Hardware Overview</td></tr><tr><td>CPU</td><td>Intel(R) Xeon(R) Platinum8375C CPU @ 2.90GHz</td></tr><tr><td>GPU</td><td>8×NVIDIA A100 Tensor Core GPU</td></tr><tr><td>Memory</td><td>1T DRAM</td></tr><tr><td>Operating System</td><td>Ubuntu 22.04.4 LTS</td></tr><tr><td>CUDA Version</td><td>12.1</td></tr><tr><td>NVIDIA Driver</td><td>530.30.02</td></tr><tr><td>ML Framework</td><td>Python 3.10.12 Pytorch 2.5.1</td></tr></table>

<table><tr><td colspan="2">GPU Specifications</td></tr><tr><td>CUDA Cores</td><td>6912</td></tr><tr><td>Memory Capacity</td><td>80GB</td></tr><tr><td>Memory Bandwidth</td><td>1935GB/s</td></tr></table>

Table 2. Statistical information of the two used benchmarks, including each task type and quantity. 

<table><tr><td rowspan="2">MotionBench</td><td>MR</td><td>LM</td><td>AO</td><td>RC</td><td>MO</td><td>CM</td></tr><tr><td>1478</td><td>546</td><td>519</td><td>400</td><td>690</td><td>385</td></tr><tr><td rowspan="2">FAVOR-Bench</td><td>AS</td><td>HAC</td><td>SAD</td><td>MAD</td><td>CM</td><td>NSM</td></tr><tr><td>2637</td><td>1541</td><td>1662</td><td>1205</td><td>1075</td><td>64</td></tr></table>

which represents the projection of the original VDM attention onto a motion subspace. This projected attention is the signal aligned with the VLM in MotionEnhancer.

# 4. Discussion

# 4.1. Dependence on VDM quality

MotionEnhancer’s effectiveness depends on the informativeness of the VDM’s motion priors. As discussed in the main content, when a VDM produces diffuse attention (e.g., large static subjects), the alignment signal weakens. Nevertheless, using such priors can yield improvements, and our paper’s focus is on how to transfer those priors rather than on improving their intrinsic quality. To mitigate this, we will employ solutions like confidence gating (low-motionconfidence cases get minimal alignment) or ensemble attention maps from multiple VDMs to reduce model-specific artifacts.

# 4.2. Role of λ in Alignment

Tuning λ effectively trades off trust in the VDM prior versus the VLM’s own training signal. $\lambda = 0$ means no prior injection and standard VLM. Moderate λ presents motionsensitive heads and tokens are guided by priors, while static reasoning remains mostly unaffected. Very large λ denotes that VLM may overfit the VDM’s specific biases, potentially harming robustness.

Table 3. Quantitative results of MotionBench. \* denotes results we reproduced using their open-source code, while other results are taken from the original benchmark. 

<table><tr><td>Model</td><td>Frames</td><td>Overall</td><td>Average</td><td>MR</td><td>LM</td><td>CM</td><td>MO</td><td>AO</td><td>RC</td></tr><tr><td colspan="10">Small Size Series</td></tr><tr><td>Qwen2.5-VL-3B* [2]</td><td>1fps</td><td>53.56</td><td>49.45</td><td>59.54</td><td>53.11</td><td>38.44</td><td>70.14</td><td>40.46</td><td>35.00</td></tr><tr><td>Qwen2.5-VL-3B + MotionEnhancer (Ours)</td><td>1fps</td><td> $56.60^{\uparrow 3.04}$ </td><td> $52.51^{\uparrow 3.06}$ </td><td>63.06</td><td>61.72</td><td>47.01</td><td>68.84</td><td>43.16</td><td>31.25</td></tr><tr><td>InternVL3-2B* [30]</td><td>8</td><td>53.96</td><td>49.69</td><td>60.01</td><td>57.69</td><td>43.90</td><td>70.00</td><td>40.27</td><td>26.25</td></tr><tr><td>InternVL3-2B + MotionEnhancer (Ours)</td><td>8</td><td> $55.50^{\uparrow 1.54}$ </td><td> $51.35^{\uparrow 1.66}$ </td><td>61.57</td><td>57.51</td><td>46.23</td><td>71.30</td><td>42.00</td><td>29.50</td></tr><tr><td colspan="10">Medium Size Series</td></tr><tr><td>MiniCPM-V2.6-7B [27]</td><td>64</td><td>52</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CogVLM2-Video-8B [10]</td><td>24</td><td>41</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GLM4-9B + TE Fusion [11]</td><td>16</td><td>58</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-VL-7B* [2]</td><td>1fps</td><td>52.81</td><td>48.29</td><td>59.00</td><td>54.58</td><td>35.58</td><td>71.30</td><td>38.54</td><td>30.75</td></tr><tr><td>Qwen2.5-VL-7B + MotionSight* [5]</td><td>1fps</td><td>55.30</td><td>51.56</td><td>59.88</td><td>57.33</td><td>47.01</td><td>73.91</td><td>40.46</td><td>30.75</td></tr><tr><td>Qwen2.5-VL-7B + MotionEnhancer (Ours)</td><td>1fps</td><td> $57.04^{\uparrow 4.23}$ </td><td> $52.92^{\uparrow 4.63}$ </td><td>63.40</td><td>61.54</td><td>47.27</td><td>70.29</td><td>43.55</td><td>31.50</td></tr><tr><td>InternVL3-8B* [30]</td><td>8</td><td>54.88</td><td>50.81</td><td>60.42</td><td>58.06</td><td>43.64</td><td>70.29</td><td>43.93</td><td>28.50</td></tr><tr><td>InternVL3-8B +MotionEnhancer (Ours) [30]</td><td>8</td><td> $57.69^{\uparrow 2.81}$ </td><td> $53.22^{\uparrow 2.41}$ </td><td>64.14</td><td>60.07</td><td>48.83</td><td>75.94</td><td>40.85</td><td>29.50</td></tr><tr><td colspan="10">Large Size Series</td></tr><tr><td>PLLaVA-34B [25]</td><td>16</td><td>52</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LLaVA-NeXT-Video-34B [13]</td><td>32</td><td>48</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-VL-72B* [2]</td><td>1fps</td><td>58.30</td><td>54.32</td><td>64.00</td><td>60.30</td><td>48.60</td><td>73.20</td><td>46.80</td><td>33.00</td></tr></table>

Table 4. Quantitative results of FAVOR-Bench. \* denotes results we reproduced using their open-source code, while other results are taken from the original benchmark. 

<table><tr><td>Model</td><td>Frames</td><td>Overall</td><td>Average</td><td>AS</td><td>HAC</td><td>SAD</td><td>MAD</td><td>CM</td><td>NSM</td></tr><tr><td colspan="10">Small Size Series</td></tr><tr><td>VideoLLaMA3-2B [28]</td><td>1fps</td><td>32.98</td><td>34.61</td><td>28.97</td><td>36.60</td><td>34.90</td><td>38.01</td><td>28.56</td><td>40.62</td></tr><tr><td>InternVL2.5-2B [3]</td><td>8</td><td>22.90</td><td>23.45</td><td>18.70</td><td>28.23</td><td>23.71</td><td>27.47</td><td>19.16</td><td>23.44</td></tr><tr><td>Qwen2.5VL-3B* [2]</td><td>1fps</td><td>37.43</td><td>38.07</td><td>38.45</td><td>38.16</td><td>39.35</td><td>43.40</td><td>23.72</td><td>45.31</td></tr><tr><td>Qwen2.5VL-3B + MotionEnhancer (Ours)</td><td>1fps</td><td> $44.53^{\uparrow 7.10}$ </td><td> $43.94^{\uparrow 5.87}$ </td><td>45.01</td><td>51.59</td><td>44.40</td><td>48.96</td><td>28.37</td><td>45.31</td></tr><tr><td>InternVL3-2B* [30]</td><td>1fps</td><td>39.27</td><td>39.11</td><td>37.66</td><td>43.28</td><td>40.49</td><td>44.98</td><td>29.21</td><td>39.06</td></tr><tr><td>InternVL3-2B + MotionEnhancer (Ours)</td><td>1fps</td><td> $43.71^{\uparrow 4.44}$ </td><td> $45.35^{\uparrow 6.24}$ </td><td>38.53</td><td>54.57</td><td>42.60</td><td>51.78</td><td>33.02</td><td>51.56</td></tr><tr><td colspan="10">Medium Size Series</td></tr><tr><td>LLaVA-Video-7B-Qwen2 [29]</td><td>64</td><td>38.60</td><td>39.94</td><td>36.14</td><td>41.27</td><td>41.28</td><td>44.48</td><td>29.58</td><td>46.88</td></tr><tr><td>VideoChat-Flash-Qwen2-7B [14]</td><td>1fps</td><td>43.82</td><td>44.86</td><td>41.90</td><td>48.41</td><td>42.84</td><td>50.95</td><td>35.07</td><td>50.00</td></tr><tr><td>VideoLLaMA3-7B [28]</td><td>1fps</td><td>41.46</td><td>41.46</td><td>40.20</td><td>44.13</td><td>42.42</td><td>48.30</td><td>31.53</td><td>42.19</td></tr><tr><td>Video-LLaVA-7B [15]</td><td>8</td><td>25.37</td><td>25.09</td><td>24.91</td><td>21.54</td><td>25.45</td><td>30.54</td><td>26.23</td><td>21.88</td></tr><tr><td>LLaVA-NeXT-Video-7B [13]</td><td>8</td><td>23.45</td><td>22.27</td><td>21.27</td><td>22.45</td><td>26.05</td><td>26.72</td><td>23.07</td><td>14.06</td></tr><tr><td>Tarsier-7B [23]</td><td>8</td><td>17.46</td><td>20.50</td><td>12.55</td><td>21.16</td><td>17.87</td><td>17.93</td><td>22.23</td><td>31.25</td></tr><tr><td>Qwen2.5VL-7B* [2]</td><td>1fps</td><td>42.61</td><td>42.58</td><td>41.64</td><td>47.83</td><td>44.89</td><td>47.55</td><td>28.28</td><td>45.31</td></tr><tr><td>Qwen2.5VL-7B + MotionSight* [5]</td><td>1fps</td><td>45.47</td><td>45.99</td><td>46.23</td><td>51.59</td><td>45.01</td><td>50.04</td><td>29.95</td><td>53.12</td></tr><tr><td>Qwen2.5VL-7B + MotionEnhancer (Ours)</td><td>1fps</td><td> $46.88^{\uparrow 4.27}$ </td><td> $47.01^{\uparrow 4.43}$ </td><td>49.34</td><td>50.62</td><td>45.37</td><td>53.20</td><td>30.42</td><td>53.12</td></tr><tr><td>InternVL3-8B* [30]</td><td>1fps</td><td>45.82</td><td>46.35</td><td>45.39</td><td>48.54</td><td>47.59</td><td>51.45</td><td>33.58</td><td>51.56</td></tr><tr><td>InternVL3-8B + MotionEnhancer (Ours)</td><td>1fps</td><td> $48.94^{\uparrow 3.12}$ </td><td> $49.25^{\uparrow 2.90}$ </td><td>47.17</td><td>57.11</td><td>46.57</td><td>56.35</td><td>36.74</td><td>51.56</td></tr><tr><td colspan="10">Large Size Series</td></tr><tr><td>LLaVA-NeXT-Video-34B [13]</td><td>8</td><td>30.44</td><td>32.58</td><td>31.70</td><td>31.99</td><td>32.31</td><td>22.99</td><td>29.58</td><td>46.88</td></tr><tr><td>LLaVA-Video-72B-Qwen2 [29]</td><td>64</td><td>46.08</td><td>46.49</td><td>48.35</td><td>47.50</td><td>45.25</td><td>51.70</td><td>33.02</td><td>53.12</td></tr><tr><td>Qwen2.5-VL-72B* [2]</td><td>1fps</td><td>48.14</td><td>48.17</td><td>50.28</td><td>46.98</td><td>48.13</td><td>51.78</td><td>40.28</td><td>51.56</td></tr><tr><td>InternVL2.5-78B [3]</td><td>8</td><td>38.54</td><td>38.36</td><td>38.38</td><td>40.62</td><td>39.05</td><td>43.65</td><td>29.40</td><td>39.06</td></tr></table>

![](images/b1cb587105269156389f42e31ce838ee718a5f97a0d87d5a95a9aa86d8dfe050.jpg)

<details>
<summary>text_image</summary>

leokimvideo
leokimvideo
leokimvideo
2 Steps
5 Steps
10 Steps
leokimvideo
leokimvideo
leokimvideo
</details>

(a) DDIM Inversion/Denoising Steps   
In the center of the screen is the front grill emblem of a Honda car, with a red doll stuck underneath the \"H\". On the right side of the screen, a hand reaches in, attempting to pull the doll out, then manipulating it, trying to remove it.

![](images/4e3ddd1f90440888fc0a09d90dad871d60660f45ae2c44b0bfb5b5f3e9e26ebc.jpg)

<details>
<summary>text_image</summary>

25%
leokimvideo
leokimvideo
leokimvideo
50%
leokimvideo
leokimvideo
75%
leokimvideo
leokimvideo
leokimvideo
</details>

(b) MHS Selection Proportion   
Figure 1. Visualization of different DDIM inversion/denoising steps and MHS selection proportion (left top: original video).

# 5. Experiments

# 5.1. System & Hardware Setups

The system and hardware details with GPU specifications of our experimental setup are provided in Tab. 1.

# 5.2. Benchmark Details

Basic information. In our experiments, we select two public benchmarks, namely MotionBench [11] and FAVOR-Bench [22], containing questions of different types from different video sources. They have been used in previous VLM for video understanding studies, making them suitable for evaluating the performance of our method. The details of these benchmarks are as follows:

• MotionBench is a benchmark with 5,385 videos and a total of 8,052 question-answer pairs. It is divided into a development set with 4,018 samples and a test set with 4,036 samples. MotionBench evaluates six core capabilities for understanding fine details in motion: Motion Recognition (MR), Location-related Motion (LM), Action Order (AO), Repetition Count (RC), Motion-related Objects (MO), and Camera Motion (CM). This allows for a comprehensive evaluation of motion-level perception. The videos come from diverse sources, including the web, existing public datasets, and synthetic videos created using Unity3. This mix ensures broad coverage of real-world applications. All data underwent careful human annotation and a multi-stage quality control process.   
• FAVOR-Bench is a benchmark of 1,776 carefully selected videos covering diverse domains, each with detailed manual annotations of various motions. It evaluates models through both close-ended and open-ended tasks. For close-ended evaluation, it contains 8,184 challenging question-answer pairs across six tasks: Action Sequence (AS), Holistic Action Classification (HAC), Single Action Detail (SAD), Multiple Action Details (MAD), Camera Motion (CM), and Non-Subject Motion

(NSM). For open-ended evaluation, FAVOR-Bench offers both a novel cost-efficient LLM-free caption assessment method and a GPT-assisted evaluation approach.

Benchmark Statistics. We provide statistical information of the MotionBench dev set and FAVOR-Bench close-ended evaluation in Tab. 2, including each task type and its corresponding quantity.

# 5.3. Comparison with More VLMs

Given the page constraint of the main text, the comparative results of additional VLMs on the FAVOR-Bench benchmark are presented in Tab. 3 and Tab. 4. Empowered by MotionEnhancer, Qwen2.5-VL and InternVL3 not only outperform their counterparts within the same model size category but also attain performance comparable to that of larger-scale VLMs.

# 5.4. More Ablations

We conduct ablation studies on DDIM inversion/denoising steps and MHS/MTTI selection ratio. Tab. 5 shows that the number of DDIM steps has a clear impact on performance. Using only 2 steps leads to obvious degradation on both benchmarks, especially on FAVOR-Bench, indicating that overly coarse inversion/denoising fails to recover reliable motion-sensitive attention from the VDM. Increasing the number of steps consistently improves the results, as finer inversion and denoising better preserve temporally meaningful motion cues. However, more steps also introduce higher computational cost. The visual differences under different step settings are shown in Fig. 1(A), and we use 5 steps because the reconstructed dynamics are good for human eyes at low cost.

The attention head and text token selection ratio also affects performance by controlling the purity and coverage of the motion prior. Fig. 1(B) and Tab. 5 show the visual and quantitative changes under different ratios. A small ratio (25%) makes the prior overly sparse and may discard useful motion heads or motion-related text tokens, resulting in incomplete motion supervision. In contrast, a large ratio (75%) introduces more non-motion attention, which weakens the motion signal and hurts alignment quality. The 50% setting gives the best trade-off between retaining sufficient motion information and filtering out irrelevant noise, leading to the most balanced overall performance. Therefore, we use 50% as a stable default.

Table 5. Ablation of DDIM inversion/denoising steps and MHS/ MTTI selection ratio. All results are trained with 25k data. 

<table><tr><td rowspan="2">Model Variant</td><td colspan="2">MotionBench</td><td colspan="2">FAVOR-Bench</td></tr><tr><td>Over.</td><td>Aver.</td><td>Over.</td><td>Aver.</td></tr><tr><td>Qwen2.5VL-7B</td><td>54.83</td><td>51.51</td><td>44.83</td><td>44.54</td></tr><tr><td>- 2 steps for DDIM</td><td>52.76</td><td>48.68</td><td>24.01</td><td>25.16</td></tr><tr><td>- 5 steps for DDIM</td><td>57.04</td><td>52.92</td><td>46.88</td><td>47.01</td></tr><tr><td>- 10 steps for DDIM</td><td>57.51</td><td>53.35</td><td>49.02</td><td>47.99</td></tr><tr><td>- 25% head/token select</td><td>55.33</td><td>51.00</td><td>47.12</td><td>47.28</td></tr><tr><td>- 50% head/token select</td><td>57.04</td><td>52.92</td><td>46.88</td><td>47.01</td></tr><tr><td>- 75% head/token select</td><td>54.88</td><td>50.69</td><td>45.69</td><td>44.10</td></tr></table>

# 5.5. MotionEnhancer VS. More Training Data

Tab. 6 analyzes the effectiveness of MotionEnhancer compared to direct data scaling on MotionBench and FAVOR-Bench using Qwen2.5-VL-7B as the baseline VLM. Notably, MotionVid-QA [5] is a large dataset containing 133k samples. While trained solely on MotionVid-QA, we observe clear improvements over the baseline VLM (row 2). Crucially, despite this substantial difference in training data volume, MotionEnhancer achieves superior performance on MotionBench (both Overall and Average scores) while delivering comparable performance on FAVOR-Bench with only 25k training data. These results highlight MotionEnhancer’s ability to achieve competitive or superior outcomes with significantly less training data, underscoring its data efficiency and effectiveness in enhancing motion understanding for VLMs.

# 5.6. Results on Conventional Benchmarks

SEED-Bench [12] is a large-scale benchmark with 19K human-annotated multiple-choice questions across 12 spatial and temporal dimensions on images and videos, designed to objectively evaluate the generative comprehension capabilities of multimodal large language models. Video-MME [6] is the first comprehensive benchmark for evaluating multimodal large language models on video understanding, featuring 900 manually curated videos across diverse domains and durations, with 2,700 expert-annotated multiple-choice questions. It uniquely incorporates subtitles and audio to assess models’ capabilities in handling complex, long-form, and multimodal video content.

As shown in Tab. 7, across both benchmarks, our method demonstrates that enhancing motion modeling does not come at the cost of traditional video understanding performance. Instead, MotionEnhancer yields consistent improvements or preserves strong baselines, confirming its compatibility with general-purpose video comprehension tasks. For Qwen2.5-VL-7B, MotionEnhancer slightly reduces the SeedBench “All” score, but improves the video subset from 61.3 to 62.1 and maintains competitive image-level performance. On VideoMME, the model equipped with MotionEnhancer achieves an overall score of 61.4, while achieving competitive results on short, medium, and long video categories—demonstrating that temporal enhancements remain stable across video durations. For InternVL3-8B, the trend is similarly positive, where MotionEnhancer preserves strong SeedBench results. Importantly, on VideoMME, MotionEnhancer improves the overall score from 62.1 to 62.3, with consistent benefits across short and medium segments. This suggests that MotionEnhancer strengthens temporal reasoning without materially harming static-image understanding.

Table 6. Experimental results of MotionEnhancer VS. More Training Data on MotionBench and FAVOR-Bench using Qwen2.5-VL-7B as our backbone. 

<table><tr><td rowspan="2">Model</td><td colspan="2">MotionBench</td><td colspan="2">FAVOR-Bench</td></tr><tr><td>Over.</td><td>Aver.</td><td>Over.</td><td>Aver.</td></tr><tr><td>Qwen2.5VL-7B</td><td>52.81</td><td>48.29</td><td>42.61</td><td>42.58</td></tr><tr><td>+ MotionVid-QA (133k)</td><td>55.70</td><td>51.25</td><td>47.12</td><td>47.28</td></tr><tr><td>+ MotionEnhancer (25k)</td><td>57.04</td><td>52.92</td><td>46.88</td><td>47.01</td></tr></table>

Overall, these results indicate that MotionEnhancer not only boosts performance on motion-centric tasks but also maintains or even subtly improves general video understanding ability. This demonstrates that injecting motionaware priors does not introduce instability or unwanted trade-offs, reinforcing MotionEnhancer’s suitability as a lightweight and broadly applicable enhancement for videolanguage models.

# 6. Visulization Examples

# 6.1. A Complete Training Sample

To demonstrate the motion-aware token selection process, we provide a detailed visualization of a training instance in Fig. 3. The video depicts an elderly man performing a sequence of actions: raising his arms, clenching his fists, bowing his head, and speaking continuously. These actions are annotated with descriptive text, from which we extract tokens using the MTTI. Tokens such as “raise”, “bow”, “lowering”, and “speaking” receive high MS values, indicating strong correlation with motion dynamics. In contrast, static or contextual tokens like “his”, “the”, or “glasses” are assigned low scores and are filtered out. This selection process ensures that only semantically and dynamically relevant tokens contribute to the motion prior.

Table 7. Performance on SEED-Bench and VideoMME (w/o sub). \* denotes results we reproduced using their open-source code, while other results are taken from the original benchmark. 

<table><tr><td rowspan="2">Model</td><td colspan="3">SEED-Bench</td><td colspan="4">VideoMME (w/o sub)</td></tr><tr><td>All</td><td>Image</td><td>Video</td><td>Overall</td><td>Short</td><td>Medium</td><td>Long</td></tr><tr><td>GPT-4V [1]</td><td>67.3</td><td>69.1</td><td>60.5</td><td>59.9</td><td>70.5</td><td>55.8</td><td>53.5</td></tr><tr><td>LLaVA-1.5-13B [17]</td><td>61.6</td><td>68.2</td><td>42.7</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Weitu-VL-1.0-13B</td><td>69.2</td><td>74.2</td><td>50.5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPHINXv2-1k-13B [16]</td><td>67.5</td><td>74.8</td><td>39.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VideoLLaMA3-7B [28]</td><td>-</td><td>-</td><td>-</td><td>66.2</td><td>80.1</td><td>63.7</td><td>54.9</td></tr><tr><td>VideoChat-Flash-7B [14]</td><td>-</td><td>-</td><td>-</td><td>65.3</td><td>78.0</td><td>67.8</td><td>55.6</td></tr><tr><td>VITA1.5-7B [7]</td><td>-</td><td>-</td><td>-</td><td>56.1</td><td>67.0</td><td>54.2</td><td>47.1</td></tr><tr><td>Qwen2.5-VL-7B* [2]</td><td>74.1</td><td>77.5</td><td>61.3</td><td>60.7</td><td>70.9</td><td>59.6</td><td>51.6</td></tr><tr><td>Qwen2.5-VL-7B+MotionEnhancer (Ours)</td><td>72.9</td><td>75.9</td><td>62.1</td><td>61.4</td><td>72.8</td><td>61.1</td><td>50.2</td></tr><tr><td>InternVL3-8B* [30]</td><td>72.7</td><td>76.2</td><td>60.4</td><td>62.1</td><td>71.8</td><td>61.9</td><td>52.6</td></tr><tr><td>InternVL3-8B+MotionEnhancer (Ours)</td><td>72.4</td><td>75.8</td><td>60.6</td><td>62.3</td><td>73.0</td><td>62.2</td><td>51.7</td></tr></table>

Furthermore, we visualize the aggregated text-to-vision attention maps for the selected tokens in Fig. 2 and Fig. 4. These maps highlight spatial regions in the video frames that correspond to the described actions, confirming that the model attends to the correct locations where motion occurs. This not only validates the effectiveness of MTTI but also provides interpretable evidence of how MotionEnhancer grounds language in dynamic visual contexts.

# 6.2. DDIM Inversion and Reconstruction

Fig. 5 illustrates the DDIM inversion and reconstruction pipeline, a core component of our attention extraction mechanism. The process begins with the original video frames, which are encoded into latent representations using the VDM’s VAE. These latents are then inverted through a deterministic DDIM process to obtain a noise sequence that corresponds to the original video. This reversibility is crucial for ensuring that the attention maps extracted during the denoising process are semantically aligned with the original video content.

# 6.3. Video Understanding

We present qualitative comparisons on MotionBench (Fig. 6) and FAVOR-Bench (Fig. 7) between the baseline Qwen2.5-VL-7B model and its MotionEnhancerenhanced version across a range of motion-centric tasks. These case studies underscore two key contributions of MotionEnhancer: Improved temporal grounding: the model better aligns textual descriptions with the actual sequence of events. Enhanced motion sensitivity: the model becomes more responsive to both foreground and background motion cues. Importantly, these improvements are achieved with-

![](images/a85ba5dc7feb6fccef46398f095e490dc9d4b55456cafc1753aa495996eb725b.jpg)

<details>
<summary>text_image</summary>

Head 10 (2.30)
</details>

![](images/b89c7c294ba126744634b2bcfcfe2d9bb1fe1fb3af637a3cac2b94903671587a.jpg)

<details>
<summary>text_image</summary>

Head 11 (-1.84)
</details>

Figure 2. Zoom-in view of two specific heads. We selected the first 100 spatial location tokens for visualization. It is demonstrated that heads with high scores exhibit a diagonal pattern, which is consistent with the main paper.

out modifying the VLM architecture, highlighting the flexibility and generalizability of our attention alignment approach.

# References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
[3] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024.   
[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,

Raw Video   
![](images/ed7db4370ed6a6379df1ccb2261b25a5d9efa2670b33cb8bbba285ea777c7a2c.jpg)

<details>
<summary>natural_image</summary>

Portrait of an older man wearing glasses and a plaid shirt, seated indoors with blurred background elements (no visible text or symbols)
</details>

![](images/e4d5b73afd499c3141803808a3f6ed91eb189b4ad2d5afc9b637158ff9e96858.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man in a plaid shirt seated indoors, gesturing with hands (no visible text or symbols)
</details>

![](images/c3348f7ac87ab312002f6e802585d1ae00515e1a07712f10a229374572faff7c.jpg)

<details>
<summary>natural_image</summary>

Man in plaid shirt giving thumbs-up gesture indoors (no visible text or symbols)
</details>

![](images/be94fe1941678466178be5f03dd094f4303f9fd3c80bccc7c45209e0185aa37d.jpg)

<details>
<summary>natural_image</summary>

Man in blue plaid shirt making fists in a dimly lit room (no visible text or symbols)
</details>

![](images/1580dace58573b498c50f110372458bca3296431fae88ffb6a9c4b93a47282ab.jpg)

<details>
<summary>natural_image</summary>

Man in a plaid shirt gesturing with hands, indoors near a window (no visible text or symbols)
</details>

![](images/974538c5d5025e3290a5a034e504902201ee8b883d4ddaf296cf678d6bdb3a89.jpg)

<details>
<summary>natural_image</summary>

Portrait of an elderly man wearing glasses and a plaid shirt, seated indoors with no visible text or symbols.
</details>

![](images/7b3935a1a0d12d0ef85ee37f6853c7e97cf709ce46e4385c6bc4d2e71982eb3f.jpg)

<details>
<summary>natural_image</summary>

Portrait of an elderly man wearing glasses and a plaid shirt, seated indoors with no visible text or symbols.
</details>

![](images/ce50cdbe7e16171bdcc6178029affc532ef4dbc6696087aed8a56bf8b1cae432.jpg)

<details>
<summary>natural_image</summary>

Portrait of an elderly man wearing glasses and a plaid shirt, smiling indoors with blurred background elements (no visible text or symbols)
</details>

# Selected Text Token by MTTI

An elderly bald man wearing glasses, with a black shirt and a blue and black plaid shirt over it, is  sitting towards the right of the center of the picture. He first raises his bent arms, with his palms open and facing his body, placed in front of his chest. Then he clenches his fists,  bows his head in  coordination  with the hand gesture, and raises his head again before relaxing and lowering his arms from the bottom of the frame. Throughout the process, he is continuously speaking, and at the end, he even smiles and moves his eyebrows.

Motion Score 

<table><tr><td rowspan="2">Selected</td><td>Word</td><td>raise</td><td>man</td><td>open</td><td>bow</td><td>lowering</td><td>speaking</td></tr><tr><td>Score</td><td>0.73</td><td>0.85</td><td>0.83</td><td>0.74</td><td>0.61</td><td>1.29</td></tr><tr><td rowspan="2">Unselected</td><td>Word</td><td>even</td><td>his</td><td>from</td><td>in</td><td>the</td><td>glasses</td></tr><tr><td>Score</td><td>-1.25</td><td>-0.42</td><td>-1.67</td><td>-0.85</td><td>-0.35</td><td>-1.50</td></tr></table>

VDM's Text Token Attention Maps   
![](images/474b7f6f7aa7c4ef8e1438d263ebf69275fa14e4522e88def2332837a554103e.jpg)

<details>
<summary>text_image</summary>

elderly
clenches
</details>

Figure 3. A complete training sample. We first show the selected text token by MTTI (in red). We then present the motion scores of some text tokens based on the mean value and the first-order difference mean. We further illustrate two aggregated text-to-vision attention maps.

![](images/052a2254e2d7fc9033942906db8790a4cca334b1d3c6963d5d7414884dfa3a50.jpg)  
Figure 4. Vision-to-vision attention maps of different heads in VDM, with each head bearing a total score of DFC, TCS, and DSR. Heads in red are selected for aggregation.

![](images/b5a52e260cecedab4f97fef2912e0f8554746614c4bec28e4ace414c7dcd064e.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a car with a crowd of people, one in foreground and two in the background (no visible text or symbols)
</details>

![](images/664b2e5fb0ffda7c94c1af4f47a3daa080c8a4448d3c4b75ab9d66914178781f.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a car exterior, a close-up of a textured surface with purple and pink speckles, and a close-up of a car's dashboard (no visible text or symbols)
</details>

![](images/393f8dc862317dc71877a12dc000361607d9ee9e90d2f60d0dd7ad38e2ebe07a.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person driving a car, with dense crowd of people in the background (no visible text or symbols)
</details>

![](images/b4ceb18015171e1cc09dfcd20b26eb5814339b4863d55347f684891fd9f1f9a8.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person standing beside a blue car, with a close-up of a crowd of people in the background (no visible text or symbols)
</details>

![](images/309c4c44688bb64ec825b032afc8dce5c1729203ec2057f8e701911ae86b0dea.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person standing beside a blue car, a close-up of a textured surface, and a man walking near a car (no visible text or symbols)
</details>

![](images/a243fbf1a9532e8bea5cd254f6e3c8fdc8aa91edeb6c4e3c539bcf8c7dd04acc.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a man standing indoors, a close-up of a car with a person walking nearby (no visible text or symbols)
</details>

On the left side of the screen, there is a car with its door open. A man wearing sunglasses is getting out of the car. His body is facing towards the right of the screen, and with his left hand, he closes the car door and walks towards the camera.

![](images/f74104ce8f16776ba19f607bf5ed870e879aba45695f0cf2074e0054bec49341.jpg)

<details>
<summary>natural_image</summary>

Two-panel photo showing a cooking process: top shows stir-frying noodles with chopsticks, bottom shows a pan of food being mixed (no text or symbols visible)
</details>

![](images/e46f19a9bedc4a42170151aa0bb4ede8d59d400babf1b3b8706e58ae01d95961.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo showing cooking process: stir-fried noodles in a pot, then chopped ingredients in a bowl, and finally boiling on a stove (no text or symbols visible)
</details>

![](images/746dcd3cfb61e468eca9503c4856f5c0c46a6f1d4d9b92468362b2f8ea33ebe9.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo showing a cooking process: stir-fried noodles, then chopped granular material, and finally boiling in a pan (no text or symbols visible)
</details>

![](images/fff861eb76185734fd5514e62070470744206b48b0e9682c2a51596ee80730c3.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo showing cooking steps: stir-fried noodles, granular texture with colorful speckles, and boiling in a pan (no text or symbols visible)
</details>

![](images/ade7915bf6a6ecd28c6f861b349ae7d44097b53cf33bd7c285117299fc145c6d.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo showing food preparation: stir-fried noodles in a pot, then chopped ingredients in a bowl, and finally boiling on a stove (no text or symbols visible)
</details>

![](images/6fe9565d96ed85efab1cedae98ac327a81e53023a0a77e29e60301b94e821c6b.jpg)

<details>
<summary>natural_image</summary>

Three-panel photo showing a pan of shredded food, a close-up of granular material, and a close-up of fine noodles in a glass bowl (no text or symbols visible)
</details>

The center of the screen shows a pot of soaked bean sprouts being cooked. At the top right of the screen, a black spoon is stirring the bean sprouts, lifting a spoonful and then setting it back down, gently manipulating the bean sprouts.

![](images/8e3ba3a59c5486858ce5ab4d39b6f6228043aaa08ffb150d3586a542c7ee87e9.jpg)

<details>
<summary>natural_image</summary>

Two-panel image: top shows a person walking on a grassy field with a bucket; bottom shows a large crowd of people in a field, reflected in water.
</details>

![](images/966772258dfd3f2975933c7e8e773f019e432d0ab6ed7956d6f5892bd6b1cacf.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person spraying water in a grassy field, with a close-up of the crowd and a person walking nearby (no visible text or symbols)
</details>

![](images/e47a87b54bbaa1166fdf4d7bab85b2a5b26a29d7466129ec6801cb4763cd0ee7.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person walking on a grassy field, with a green field and a close-up of the crowd in the background (no text or symbols visible)
</details>

![](images/66fc205725213d04f425381879ef9c95c538c38b84f3276b895e8c3b1d35c76b.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person walking on grass, with a ring and crowd in the background (no text or symbols)
</details>

![](images/2b7993675008482553a821cb1cc28c06e0b89feedaa230f6f2d50e72bec80050.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person in a field, with a magnifying glass and a close-up of a flower field (no text or symbols visible)
</details>

![](images/1e3ab77ac8942dd86d47f2863e8f0e91f3d301d2a802973d16f9cd7d32ef0b4b.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a person walking in a field, a close-up of a large pile of flowers, and a person walking outdoors (no text or symbols visible)
</details>

On the left side of the screen, a person is walking forward on a patch of grass with their back to the camera. Their right hand is placed behind the body, holding a hat, while the left arm swings back and forth on the left side of the body.

![](images/c18372f743ca3fb7a99644028a936526b07f585d603bd516cf806486f1a43ab5.jpg)

<details>
<summary>natural_image</summary>

Close-up of hands cleaning a car window with a red spray bottle (no text or symbols visible)
</details>

![](images/01a7c6173b6873f8313667a4e29f1df97b68418d29e037aca4c7be621c6d6a90.jpg)

<details>
<summary>natural_image</summary>

Close-up of a red-handled tool applying paint to a car interior, showing texture and color variations (no text or symbols visible)
</details>

![](images/1549f6760472e73d79ad29fbed5b3525a5fd8071bc0530a3b0afcbf9eec5a6da.jpg)

<details>
<summary>natural_image</summary>

Close-up of a car's grille with red-handled tool, showing texture and color variations (no text or symbols)
</details>

![](images/1c4780781f5048b1b025c945b0b5d5f17605db4be423ec35adad0c50d60804b2.jpg)

<details>
<summary>natural_image</summary>

Close-up of a car door trim cleaning process with red-handled tool, showing texture and color variations (no text or symbols)
</details>

![](images/5e0ed66d4db77510f49599cf31df325a63dfb96d9c15b4731456d0249db85c49.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a red object being placed on a car's side panel, with a close-up of the textured surface (no text or symbols visible)
</details>

![](images/55aefd1fd8a65c0609757cfe1368a2959a7323a65f59283274e00de3aa54e263.jpg)

<details>
<summary>natural_image</summary>

Close-up of hands cleaning a textured surface with a red tool, no visible text or symbols
</details>

In the center of the screen is the front grill emblem of a Honda car, with a red doll stuck underneath the \"H\". On the right side of the screen, a hand reaches in, attempting to pull the doll out, then manipulating it, trying to remove it.

Figure 5. Our DDIM inversion and reconstruction process. Grouped in sets of three rows: the first row is the original video, the second row is the latent noise after DDIM Inversion, and the third row is the reconstructed video.

![](images/531d9936616794851ff4efae9c29364165767f18d74eabaf4578f19f5981a1d6.jpg)

![](images/acbc03b320d240eb98c6a392179bdfc65739ea8c52806735027d6cc098fe0d1d.jpg)

![](images/f1e4cf287c77fce3ddb9990447d591df3976e84f58697eb6cea227bd4191aa52.jpg)

![](images/454793af4080e80c0351ea976ea87de71893bac26e097a7880f85dde4a4a3c7f.jpg)

![](images/908771da688768771398ac5500424c250b8e6943b4a6cae5d60dc6189c161fa0.jpg)

![](images/180c0685a49c92d8d530dfb9315d8cda371f522b59e80687a029b5015ec7ac32.jpg)

Task Type: Motion Recognition

Question: What does the woman do just before closing her eyes?

A. She keeps her eyes open wide.   
B. She smiles widely.

C. She removes a cotton swab herself.   
D. She furrows her brow.

Qwen2.5VL-7B: B. She smiles widely.

Qwen2.5VL-7B + MotionEnhancer: D. She furrows her brow.

![](images/22d1d5ca543686df1958161a4af700bfadc92c1ce61882b7dc2e5cbe190f9cd4.jpg)

![](images/b11ccc56ef0b0c4d1755873bd639faf783f127a3f9f84c9b3efc9b6cf00ae2c4.jpg)

![](images/a0153a47911772cffb32114c8e3703bec61eeb3ec776c95286a82dd6cee1b593.jpg)

![](images/bf70325e726eafd511c8760b63c91d273f1a5779d05568169d3f427e823e4431.jpg)

![](images/65531ee13684a77d288e5f6533f21587958ba40dbdff5d22d711829bf0a277d1.jpg)

![](images/641ac7dc009d4d035d021b18674cec33a613873ecc13b10bf00e4cdbada5c9ed.jpg)

Task Type: Location-related Motion

Question: What action does the hand perform first when it appears on screen?

A. Places a shiitake mushroom on the right side of the pizza   
B. Places green pepper slices on the pizza

C. Sprinkles cheese crumbs on the pizza   
D. Puts blueberries on the pizza

Qwen2.5VL-7B: C. Sprinkles cheese crumbs on the pizza

Qwen2.5VL-7B + MotionEnhancer: A. Places a shiitake mushroom on the right side of the pizza

![](images/96bd2af2eacc0c6312d458d1ef02fc0b313e822006d2b020a94c613c1053fdf0.jpg)

![](images/af94aa4b1fcf9685cca620b0500e8d4e48b0d103193a60412c210d60487111e0.jpg)

![](images/c61258fd9ed8b09a54c460268dfd003c263a96cb1dd31e6c5d78c506a629b898.jpg)

![](images/f833450ff8787b8ed9502271009280c22bfa3afb1733981a59ff2c5b577c647f.jpg)

![](images/a32467da4267e8eacf52d47d7eecfab5e16629ef069bfd6452b79655a4626e6a.jpg)

![](images/b62b32d172d8e355fc325e3a1a21bfdcf81dc1bed218d47d8a8281fb29cc6599.jpg)

Task Type: Camera Motion

Question: What sequence describes the actions involving the camera and hand movements?

A. Hand retracts, then camera pans left   
B. Hand extends, then camera zooms out

C. Camera moves, then fingers extend and retract   
D. Fingers extend, then hand waves

Qwen2.5VL-7B: A. Hand retracts, then camera pans left

Qwen2.5VL-7B + MotionEnhancer: C. Camera moves, then fingers extend and retract

![](images/8d6221818d5889ae2e3ecc72087096ad448d9c6e6d02bf7ba872fd6176e54332.jpg)

![](images/39a849f275031b28c78c088ca2f4f57e626eda7c3bc0e8cf7eb89c76ce5c5b27.jpg)

![](images/f1bcd1605e00514b39f3d802da2bdd6d5d826394ef4b7980193b310ce0937d5f.jpg)

![](images/88f4eeb8f82d3a41ad03f233afac80cfd49b1e35fead9fc29539bdb1b49c528a.jpg)

![](images/851c3b55ca5ae80fe82f3e63423215f73d21dca57a944d4df4b126b97a039a77.jpg)

![](images/796497172b0b07bf3bfa6300ead080861592249a23c380700a31521aa74e0688.jpg)

Task Type: Motion-related Objects

Question: What is the person wearing green clothes wearing on their left hand?

A. Wristwatch   
B. Wrist guard

C. Bracelet   
D. Ring

Qwen2.5VL-7B: D. Ring

Qwen2.5VL-7B + MotionEnhancer: A. Wristwatch

![](images/580bd696e0cb8b72ad8e8dfcab67bc3718dfd1915b8478f93d79dc1f0f79753a.jpg)

![](images/e11e59cfa3d2b6c87ad17ac36ff2a61462dae40e9f5224aa4f61cf1d35df94b6.jpg)

![](images/abd86c2b8ca744de5b560f8295beb3991a732f36c6a354f5322fa8332bf9ec38.jpg)

![](images/a7e0b228dd7f09fee2d0ff25d1d03be0442ec2f568b579189ae13d6dc75dfa8c.jpg)

![](images/0fecdccfb8c8dc8c202a7ca861f58808606016af2726a0c3f2f0f13ef26521af.jpg)

![](images/764da6e42517babc2185b551853ef3b3284c20337749451d9b5f00af55b6bebc.jpg)

Task Type: Action Order

Question: Please describe the detailed breakdown of the action in the video.

A. Jump up, lower left arm, swing right arm, catch the ball.   
B. Catch the ball, drop left arm, jump, swing right arm

C. Jump up, drop left arm, swing right arm, catch the ball.   
D. Catch the ball, Swing right arm, Lower left arm, Jump

Qwen2.5VL-7B: C. Jump up, drop left arm, swing right arm, catch the ball.

Qwen2.5VL-7B + MotionEnhancer: A. Jump up, lower left arm, swing right arm, catch the ball.

![](images/72c075d1d9936d0fbd5d1dbababa335a7fba72db946024d78b0b24671bd0f833.jpg)

![](images/51e2d33a16a03cd619b028e60e3ec7a61cdedb5fdbd38993cf63b754cb1c441e.jpg)

![](images/eaface67b18737d21349fc7b67cd9045bd9e58db0c1028424ff04455e46efc4c.jpg)

![](images/76c2a975538d1a6a90b3d80de13fae434690c927f34273a4f407cb5fc0e912f2.jpg)

![](images/84b3c7e3110aa7bf6ec2ddfa20634e2b52d59bb8705b27706516f9bc53d3ba05.jpg)

![](images/a1ce290b81340b941f7f0b9dd6d71d9eb0db0cc57a418255cce7a5e41c0a217a.jpg)

![](images/db7ccddccab3185036eeaa6f02d979f6e48306e100870150b3d67c8c9381ea65.jpg)

Task Type: Repetition Count

Question: Please count the number of repeated actions in the video.

A. 2

C. 4   
D. 8

Qwen2.5VL-7B: B. 6

Qwen2.5VL-7B + MotionEnhancer: A. 2

Figure 6. Qualitative results on MotionBench.

![](images/ff0a8f78de685aa74f8a2354591ba0006566f300b0fcdf14340d2e19d36f070e.jpg)

![](images/e7174fa65871e229378c253c460bf592699ffc0d43f66883eeb58b7f39ff4daa.jpg)

![](images/705a13d152e32612d4719707e40a4fd983146b99c20efeb5124276b9dfbe524a.jpg)

![](images/aac6f7f390f02e7fd2fa75f1a5de8a64fe16fe4f96f7321a24a41735cedce130.jpg)

![](images/0ffe7f8488f8beb54ea6092f4415a72c0d162beb3736a4429771f924bda134e4.jpg)

![](images/a287ebd2c38d5ab04d4570dc25f036b4a2963ef9f9f693e8f9473f2b2147297a.jpg)  
Task Type: Action Sequence

Question: After picking up the computer and walking to the right, which of the following action sequences does the woman in the red and black plaid shirt perform next? C. Stop → Look to the right → Walk to the front left C.Stop →Look to the right →Walk to the front left

A. Walk to the front left → Stop → Look to the right   
B. Look to the right → Stop → Pick up the computer

D. Walk backward → Stop → Turn left and look forward   
E. Look to the right → Turn left and look forward → Stop

Qwen2.5VL-7B: E. Look to the right → Turn left and look forward → Stop

Qwen2.5VL-7B + MotionEnhancer: C. Stop → Look to the right → Walk to the front left

![](images/d2501ecda4d6f815e230cdf7e2677159fb56452f018108c65e320766fd6a1796.jpg)

![](images/c3bf7734d791938e653b7e0a22ea75c8854a0b3e5a59e3b7943fba0adea240b0.jpg)

![](images/0d92b453778ec165c76b343606456a68e1496df4a27ee3ed64090a8e0ddc026d.jpg)

![](images/edc5a2aa6a7bcc540b8a19c8f2a4e751a6a2365d81dd2e81e7a8701d73fcbb98.jpg)

![](images/18e782e006055cc65668cf746b0538d0e0f7abab72d3f888c8e17dd7e09432c7.jpg)

![](images/9cdd523e699de13a57c7b8f0002c867bb7f17a02f81550b329771b1559a6fba1.jpg)

Task Type: Holistic Action Classification)

Question：What is the core behavior demonstrated by the man in white clothing throughout

the video?

A. Repeatedly pouring water and adjusting the position of the cup   
B. Continuously walking back and forth between different locations

C. Rapidly opening and closing drawers while retrieving items   
D. Moving a chair then sitting down to organize documents   
E. Concentratedly flipping through contents in a folder.

Qwen2.5VL-7B: B. Continuously walking back and forth between different locations

Qwen2.5VL-7B + MotionEnhancer: D. Moving a chair then sitting down to organize documents

![](images/9d25615d1105568304e2b04075b51e41f61aff116ffb7ee201fccef16533726f.jpg)

![](images/4ecab157d30365f5b16fd55a0bee33a960b729885eea9ccee760d49c0d09d0b9.jpg)

![](images/a16cef85a1af4f46d3d9412c97a9794c37fddd61870b6dcb8b73f7b7dd95c114.jpg)

![](images/efba06cbac6929baa68f81c9e2ae447e6eb1827fd2237f24de5fed1a72c9e48a.jpg)

![](images/ea098d34b68228e5dd7a6ec5e059d382991763456f7ca46d5f5b3f2b43ec7c27.jpg)

![](images/acb87aeef9f25a7f546c695a0ccf2f53d953e73a9c283add15b205dc7b918bcb.jpg)

Task Type: Single Action Detail

QUestion: When the boy in green clothes performs a rollover maneuver, what is the primary dynamic movement of

the body?

A. Using both hands to push off the mattress and quickly sit up

B. Swinging legs in the air to make tapping movements

C. Repeatedly hitting the pillow with the head three times

D. Keeping the upper body stationary while only moving the legs

E. The body transitions from lying on the back to a side-lying position.

Qwen2.5VL-7B: C. Repeatedly hitting the pillow with the head three times

Qwen2.5VL-7B + MotionEnhancer: E. The body transitions from lying on the back to a side-lying position.

![](images/d282aaa32522a7ea776f3f85340d01a1bb616d266de67ef9bb53e7ddec4fb1d8.jpg)

![](images/6cc34342ec6cdc92a7dd5f14cac4c41a97b9b606a0bf3ea65d24de7faa1ac41a.jpg)

![](images/03e9b028a0a217552b61412002cab077f8aef7cfc606bd72ae8e7286da8fde65.jpg)

![](images/54917a1d466976b2ead3b0f6c40373acaed2eaff80f2f01ed4ed320e565ef506.jpg)

![](images/4a6d8c86b6d0d838ccdbe0590be0dedee50c8a5f01912960905a8d6f6c8e3c6d.jpg)

![](images/054a76a90e6e7a18118735c7c3c8a595079e283cbccc549e415d7548d80ad96d.jpg)

Task Type: Multiple Action Details

Question: In the video, what did the man in the black and green striped short-

sleeve do?

A. Put the phone in his pocket

B. Stared at the phone screen and continuously typed

C. Ran forward, took out the phone to answer a call, and then walked around

D. Put something in his mouth to eat

E. Continued to run forward and touched the camera lens.

Qwen2.5VL-7B: A. Put the phone in his pocket

Qwen2.5VL-7B + MotionEnhancer: C. Ran forward, took out the phone to answer a call, and then walked around

![](images/2342acdbf099411909b794a5b795d55ecf2912fe78b61949703c28d0461030e9.jpg)

![](images/adfaf0f49d0284a030cd2d1e76728e58987f839e16583a18693af5cf9b64ad90.jpg)

![](images/cca811f36f66003005d9c9c040c974567b868ec194c650cae761cabc88ee4dfd.jpg)

![](images/212c915577e1d8b1083ecacb4c6409d1cdb47a91f1240458c73fcb7d3c41007d.jpg)

![](images/92c4916ec8908417ef3a71452335eec129a85052b3b8d980535eae2b2ff278a4.jpg)

![](images/27fa10c211def4e3a220dca53f7a4a1919d5e55e70b05c45af8846264aa4d9a1.jpg)

Task Type: Camera Motion

Question: When the woman wearing glasses and a dark blue shirt stands up, what is the main movement of the camera?

A. Camera shake

B. Upward movement

C. Left and right movement

D. Rotation

E. Fast zoom

Qwen2.5VL-7B: E. Fast zoom

Qwen2.5VL-7B + MotionEnhancer: B. Upward movement

![](images/19bc7fa8f793ec441e27aa3a01955822ba613dddbbbf44f9d21488e2b959e0fa.jpg)

![](images/82aac9605ce81d5d591423a5622d2f5e4c08c7df04592178ff6a686215787d0e.jpg)

![](images/59ff3d8b7d28f0e5755950051f49397d325aeaca10fddfcbe0f44c597b5f8b0d.jpg)

![](images/0d0e8a0e13253c5cc5a7fae90b804bfc9d2200a7f01c070c0cf1429ddca82c1a.jpg)

![](images/8c5f92063a20d591c8b64c7918d1fa431835453a5b307b47cc3359ce051ced5a.jpg)

![](images/f3bc65f55ce657e32e4c25ae6c73a997bf8f5697762bea85310f9b60e55044c9.jpg)

Task Type: Non-Subject Motion

Question: What are all the non-dominant natural environment movements in the video?

A. Petals falling

B. Waterfall pouring

C. Treetops swaying and waterfall pouring

D. Petals falling and waterfall pouring

E. The hem of a character's clothing fluttering

Qwen2.5VL-7B: A. Petals falling

Qwen2.5VL-7B + MotionEnhancer: D. Petals falling and waterfall pouring

Figure 7. Qualitative results on FAVOR-Bench.

Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.   
[5] Yipeng Du, Tiehan Fan, Kepan Nan, Rui Xie, Penghao Zhou, Xiang Li, Jian Yang, Zhenheng Yang, and Ying Tai. Motionsight: Boosting fine-grained motion understanding in multimodal llms. arXiv preprint arXiv:2506.01674, 2025.   
[6] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108– 24118, 2025.   
[7] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Yangze Li, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025.   
[8] Junqi Ge, Ziyi Chen, Jintao Lin, Jinguo Zhu, Xihui Liu, Jifeng Dai, and Xizhou Zhu. V2pe: Improving multimodal long-context capability of vision-language models with variable visual position encoding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21070–21084, 2025.   
[9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
[10] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024.   
[11] Wenyi Hong, Yean Cheng, Zhuoyi Yang, Weihan Wang, Lefan Wang, Xiaotao Gu, Shiyu Huang, Yuxiao Dong, and Jie Tang. Motionbench: Benchmarking and improving fine-grained video motion understanding for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8450–8460, 2025.   
[12] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.   
[13] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.   
[14] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024.   
[15] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.

[16] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.   
[17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023.   
[18] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023.   
[19] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.   
[20] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.   
[21] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.   
[22] Chongjun Tu, Lin Zhang, Pengtao Chen, Peng Ye, Xianfang Zeng, Wei Cheng, Gang Yu, and Tao Chen. Favor-bench: A comprehensive benchmark for fine-grained video motion understanding. arXiv preprint arXiv:2503.14935, 2025.   
[23] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024.   
[24] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.   
[25] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024.   
[26] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.   
[27] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.   
[28] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.   
[29] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.

[30] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.