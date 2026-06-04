# AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation

Haobo Li 1 2 Yanhong Zeng 2 3 B Yunhong Lu 4 2 Jiapeng Zhu 2 Hao Ouyang 2 Qiuyu Wang 2 Ka Leong Cheng 2 Yujun Shen 2 Zhipeng Zhang 1 5 B

https://aad-1.github.io/

![](images/511f083312af1dd2bb13b463abd29465eb4a974fa965874696873ca114a7776a.jpg)

<details>
<summary>natural_image</summary>

Underwater scene with two diver swimming near coral reefs and turquoise fish swimming in the ocean (no text or symbols)
</details>

frame=0

![](images/2040cb4ab844c2cce1e43aa88050cac12432ca9a546257f598aa690a347b8fea.jpg)  
group of fish swimming over a cora

![](images/e38aeaa22fa9981fabfff23c0db6d81c33bb89b0235e1722b45d0e065bd1bea6.jpg)

![](images/c229828dbd7e4e5a2c22272f8f00ad4a2ff378b1cf14c5c06ec107fc28f41cc1.jpg)  
frame=320

![](images/2d454326bf9313bd34ef0570a4ba28fed249d29179deef68e0873f9d6937500b.jpg)

![](images/21c08677bd7338f9b37ab844500f0241854364d24bf088608ee6baf490ffc97e.jpg)  
frame=0

![](images/6edddae59dcec546a6ada5cf8b10245d68c17ebc70d0d65c28e46a02ff8b556e.jpg)

![](images/b3bd4bdf02cd035b2879db96bb025f200dae25db5ce98260f65feb77a30fca09.jpg)  
frame=160

![](images/cba6aef59b8762d1712e76c9c8535e8a6f1fdc677c2b427c7e38f5e248c88339.jpg)

![](images/e0c30a9a58b19bc49af419b580cdfa4de27ac7e915487b19f538c06f8b864d71.jpg)  
frame=320   
Figure 1. We propose AAD-1, an Asymmetric Adversarial Distillation framework for One-step autoregressive video generation. Given a single conditioning image, AAD-1 generates videos autoregressively while maintaining both high visual quality and motion fidelity over long horizons, requiring only one sampling step per chunk.

# Abstract

We present AAD-1, an Asymmetric Adversarial Distillation framework for One-step autoregressive image-to-video generation. State-of-the-art methods adopt adversarial distillation but suffer from motion collapse and training instability, resulting in static videos. AAD-1 addresses these challenges through two key designs in architecture and training strategy. Our key architectural insight is to break the symmetry between generator and discriminator. While the generator remains causal to preserve autoregressive sampling capability, the discriminator attends bidirectionally over the full spatiotemporal context and produces a single holistic realism score for the entire video sequence. This asymmetric design enables the discriminator to effectively detect global temporal failures and long-range drift that cause motion collapse in autoregressive generation. To

1AutoLab, SAI, SJTU 2Ant Group 3Department of Automation, Tsinghua University 4Zhejiang University 5Anyverse Dynamics. Correspondence to: Zhipeng Zhang <zhipeng.zhang.cv@outlook.com>, Yanhong Zeng <zengyh1900@gmail.com>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

stabilize training, we introduce a phased strategy that first uses distribution matching to bootstrap a stable one-step generator, providing a warm-up phase that brings the student distribution closer to the teacher before adversarial distillation begins. Extensive experiments on VBench demonstrate that AAD-1 achieves state-of-the-art performance in one-step autoregressive video generation.

# 1. Introduction

Fast autoregressive video diffusion post-training has emerged as a promising paradigm that adapts pretrained bidirectional video diffusion models (Wan et al., 2025; Kong et al., 2024; Lin et al., 2024), which are limited to generating fixed-length short clips, into few-step autoregressive models that support indefinitely long video generation (Teng et al., 2025; Yuan et al., 2025). This paradigm has attracted significant research interest due to its value for real-time streaming applications (e.g., gaming) and world modeling (Brooks et al., 2024; Ball et al., 2025; Feng et al., 2024).

Training fast autoregressive video diffusion models presents substantial challenges. Recent state-of-the-art methods integrate self-rollout training, where models learn from their own generated trajectories (Lin et al., 2025b; Huang et al., 2025) rather than ground-truth contexts, overcoming the exposure bias in Teacher Forcing (Ho et al., 2022) or Diffusion Forcing (Chen et al., 2024). However, self-rollout training requires performing causal adaptation and step distillation simultaneously, imposing the burden of learning both autoregressive dynamics and accelerated sampling concurrently. This coupled optimization proves particularly challenging, with existing approaches requiring four or more sampling steps to maintain acceptable quality.

In this work, we target the extremely challenging one-step autoregressive image-to-video generation. While adversarial distillation is a leading approach for one-step distillation (Lin et al., 2025a), two critical challenges limit current methods. (1) Architectural limitation. Existing methods adopt symmetric discriminator architectures that mirror the generator’s causal structure with frame-wise discrimination, as shown in Figure 2-(a) (Lin et al., 2025b). However, a causal discriminator evaluating frame t can only attend to contexts up to block t − 1 without future information, causing inherent insensitivity to accumulated temporal degradation. While individual frames appear realistic when conditioned on preceding frames, the overall sequence gradually loses motion fidelity, leading to motion collapse where videos become stuck at the initial frame (Lin et al., 2025b). Aggregating all tokens for a video-level logit (Figure 2-(b)) offers partial improvement, yet causal attention fundamentally limits capturing long-range dependencies. (2) Training instability. When training from scratch, early one-step predictions lie far from the data distribution, and under self-rollout training, this gap compounds across time, destabilizing training dynamics (Cheng et al., 2025).

To address these challenges, we propose AAD-1, an Asymmetric Adversarial Distillation framework for One-Step autoregressive video generation with two key innovations in architecture and training. (1) Bidirectional discriminator with holistic discrimination. To overcome the architectural limitation, we employ a bidirectional discriminator with video-level holistic discrimination. While the generator remains causal to preserve autoregressive sampling, as shown in Figure 2-(c), the discriminator attends bidirectionally over the full spatiotemporal volume and produces a single realism score for the entire sequence. This asymmetric design provides two critical advantages: (a) the discriminator can detect global temporal failures such as motion collapse that manifest gradually across the sequence, and (b) it can penalize long-range drift by comparing any frame against both past and future context. Our extensive ablations demonstrate that both components are essential, removing either bidirectional attention or video-level scoring substantially degrades motion quality, with causal or frame-wise variants reverting to motion collapse behaviors. (2) Phased training with distribution matching warm-up. To stabilize adversarial distillation, we introduce a warmup stage that leverages frame-wise distribution matching. Specifically, we use DMD to bootstrap a stable one-step generator that produces on-manifold predictions, establishing a foundation for subsequent adversarial refinement. This warm-up phase provides the adversarial stage with initial predictions sufficiently close to real data that the discriminator can provide meaningful gradients, preventing the training instability observed when optimizing from scratch.

![](images/d95b5019417ea16cda244292a46a05b62af429d8cf6bcf4ac52ddd9a53721380.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) Causal backbone.
        A1["Input 1"] --> B1["Causal DiT"]
        A2["Input 2"] --> B1
        A3["Input 3"] --> B1
        B1 --> C1["Output"]
        B1 --> D1["Q"]
        B2["Input 1"] --> E1["Causal DiT"]
        B3["Input 2"] --> E1
        B4["Input 3"] --> E1
        E1 --> F1["Output"]
        E1 --> G1["Q"]
        H1["Input 1"] --> I1["Causal DiT"]
        H2["Input 2"] --> I1
        H3["Input 3"] --> I1
        I1 --> J1["Output"]
        I1 --> K1["Q"]
    end

    subgraph (b) Causal backbone.
        L1["Causal DiT"] --> M1["Causal DiT"]
        M1 --> N1["Output"]
        M1 --> O1["Q"]
        P1["Causal DiT"] --> Q1["Causal DiT"]
        Q1 --> R1["Output"]
        Q1 --> S1["Q"]
    end

    subgraph (c) Bidirectional backbone
        T1["Causal DiT"] --> U1["Causal DiT"]
        U1 --> V1["Output"]
        U1 --> W1["Q"]
        X1["Causal DiT"] --> Y1["Causal DiT"]
        Y1 --> Z1["Output"]
        Y1 --> AA["Q"]
    end
```
</details>

Figure 2. Discriminator Architecture Comparison. We compare three configurations: (a) Causal backbone with frame-wise logits, providing dense local feedback but lacking global temporal context; (b) Causal backbone with video-level logit, aggregating information causally but still constrained by unidirectional attention; and (c) Bidirectional backbone with video-level logit (AAD-1), which attends to the full spatiotemporal context. The bidirectional attention in (c) enables holistic discrimination that can detect gradual motion degradation and long-range drift across the entire sequence, which causal architectures are hard to capture.

We conduct extensive experiments on VBench, demonstrating that AAD-1 achieves state-of-the-art performance in one-step autoregressive video generation with superior visual quality and motion fidelity. Our contributions are:

• We identify critical architectural and training limitations in existing one-step autoregressive video generation that lead to motion collapse and training instability.   
• We propose an asymmetric adversarial distillation framework featuring a bidirectional discriminator with video-level holistic discrimination and a phased training strategy with distribution matching warm-up.   
• We achieve state-of-the-art one-step autoregressive video generation on VBench.

# 2. Related Work

Autoregressive video diffusion models. Autoregressive video diffusion models generate video sequences frame-byframe, where each frame is synthesized through a diffusion process conditioned on preceding frames (Chen et al., 2025; Zhang & Agrawala, 2025; Wan et al., 2025). Standard training strategies include Teacher Forcing (TF) (Wan et al., 2025; Ho et al., 2022), which conditions on clean historical frames with shared noise schedules, and Diffusion Forcing (DF) (Chen et al., 2024; 2025; Teng et al., 2025), which uses independently noised contexts. To enable efficient streaming inference, recent methods adapt pretrained bidirectional models by introducing block-causal attention patterns (Yin et al., 2025; Lin et al., 2025b). These patterns apply bidirectional self-attention within local temporal windows while maintaining causal dependencies across blocks, thereby supporting KV-cache reuse during sequential generation.

To further address the train-test distribution gap, several approaches integrate self-rollout training (also termed Self Forcing (Huang et al., 2025) or Student Forcing (Lin et al., 2025b)), where models learn from their own generated trajectories rather than solely from ground-truth data (Liu et al., 2025; Cui et al., 2025). These methods typically perform distillation simultaneously, requiring the model to learn both autoregressive dynamics and accelerated sampling concurrently (Lu et al., 2025b; Hong et al., 2025; Yin et al., 2024b). However, this joint optimization presents significant training challenges, with existing approaches typically requiring four or more sampling steps to maintain acceptable quality (Yang et al., 2025). In contrast, our work targets single-step autoregressive video generation, achieving robust streaming generation with minimal inference cost.

Accelerating video diffusion models. Diffusion distillation aims to compress multi-step sampling processes into fewer iterations while preserving generation quality. Existing approaches can be categorized into trajectory-level and distribution-level methods. Trajectory-level techniques approximate the sampling trajectories of teacher models through progressive distillation that iteratively halves the number of steps (Salimans & Ho, 2022), consistency models that map arbitrary trajectory points to their origins (Song et al., 2023), or rectified flow methods that straighten sampling paths (Liu et al., 2022). Distribution-level methods, by contrast, directly match the output distributions between student and teacher models. Representative approaches include adversarial distillation, which employs discriminators to align the distributions of real and generated data (Lin et al., 2025a; Xu et al., 2024; Sauer et al., 2024), and score distillation methods that minimize the reverse KL divergence using the score functions of real and fake distributions (Wang et al., 2023; Yin et al., 2024b;a; Lu et al., 2025a).

In the video domain, existing work largely adapts image distillation techniques to bidirectional models that generate short clips of fixed duration (Shao et al., 2025; Cheng et al., 2025; Mao et al., 2025). APT2 represents the most relevant prior work, applying adversarial distillation to autoregressive video generation (Lin et al., 2025b). Our work differs from APT2 in four aspects. First, APT2 relies on a closed-source model, whereas our study is built on the publicly available Wan 2.1 backbone (Wan et al., 2025) and reports key implementation details of the training recipe. Second, APT2 uses a causal discriminator with frame-wise discrimination; in contrast, we use a bidirectional discriminator with a video-level logit, so the training-time critic can evaluate a complete rollout with future context. Third, we explicitly separate one-step initialization and adversarial refinement through a DMD warm-up stage, which avoids the instability of cold-start adversarial training. Fourth, we provide controlled ablations of backbone visibility and logit granularity, showing that the causal/frame-wise design is prone to static-video collapse while bidirectional video-level discrimination gives more stable long-horizon generation.

# 3. Preliminaries

Video notation and sliding-window causal streaming. We denote a video clip by $x _ { 1 : T } = ( x _ { 1 } , \ldots , x _ { T } )$ , where each frame $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { H \times \dot { W } \times \dot { C } }$ (height, width, channels). Let c denote optional conditioning (e.g., text). In sliding-window causal streaming, frames are generated one at a time. At step t the model produces frame $\hat { x } _ { t }$ conditioned on (i) the previous L frames (sliding window of size L) and (ii) a set of S sink frames $x _ { 1 : S }$ that are always retained from the beginning of the sequence:

$$
\hat {x} _ {t} \sim p _ {\theta} (\cdot | x _ {1: S}, \hat {x} _ {t - L: t - 1}, c). \tag {1}
$$

The sink frames provide a fixed anchor to the start of the video and help maintain long-range consistency, while the sliding window captures recent temporal context. We write $x _ { \mathrm { c t x } , t } = ( x _ { 1 : S } , \hat { x } _ { t - L : t - 1 } )$ to denote the visual context (and omit the subscript t when it is clear). The window slides forward after each frame is generated. Despite these mechanisms, errors can still compound over long sequences, leading to temporal drift.

Distribution matching distillation (DMD). DMD transfers a strong teacher diffusion model $p _ { \mathrm { T } }$ to a fast student causal generator $G _ { \theta }$ by minimizing a distributionlevel divergence. Given noise $z _ { t } \sim \mathcal { N } ( 0 , I )$ , visual context $x _ { \mathrm { c t x } , t } .$ , and text conditioning $c ,$ the student produces $\hat { x } _ { t } ~ = ~ G _ { \theta } ( z _ { t } , x _ { \mathrm { c t x } , t } , c )$ . The DMD objective encourages $p _ { G _ { \theta } }$ ≈ pT using score-based distribution-matching gradients derived from real and fake score estimates. DMD is stable for few-step distillation, but quality can degrade when pushed to a single step.

Adversarial distillation. GAN-based distillation trains a causal generator $G _ { \theta }$ together with a discriminator $D _ { \psi }$ that distinguishes real frames from generated ones. The standard adversarial objective is

$$
\begin{array}{l} \min _ {G _ {\theta}} \max _ {D _ {\psi}} \mathbb {E} _ {x} [ \log D _ {\psi} (x) ] \tag {2} \\ + \mathbb {E} _ {z _ {t}} \big [ \log \big (1 - D _ {\psi} (G _ {\theta} (z _ {t}, x _ {\mathrm{ctx}, t}, c)) \big) \big ]. \\ \end{array}
$$

For causal streaming, the generator $G _ { \theta }$ must remain strictly causal, producing each frame ${ \hat { x } } _ { t } = G _ { \theta } ( z _ { t } , x _ { \mathrm { c t x } , t } , c )$ using the visual context defined above. The discriminator $D _ { \psi }$ can be either causal (past-only) or bidirectional (accessing future frames during training). This paper studies how discriminator design and a three-stage asymmetric adversarial distillation recipe affect one-step causal generation quality.

# 4. Asymmetric Adversarial Distillation

We study one-step autoregressive image-to-video generation for streaming video applications. Our training pipeline has three stages: (i) ODE initialization via Diffusion Forcing on teacher denoising trajectories under noisy context, (ii) onestep DMD warmup under self-rollout context by matching real and fake scores, and (iii) asymmetric adversarial refinement with a causal generator trained against a bidirectional discriminator with video-level discrimination, see Figure 3.

Causal architecture adaptation. We follow the notation in Section 3. In particular, the student causal generator produces one chunk in a single forward pass, $\hat { x } _ { t } =$ $G _ { \theta } ( z _ { t } , x _ { \mathrm { c t x } , t } , c )$ , and is deployed autoregressively with a sliding-window visual context $x _ { \mathrm { c t x } , t } = ( x _ { 1 : S } , \hat { x } _ { t - L : t - 1 } )$ .

Stage I: ODE initialization. Following prior work on causal video generation (Yin et al., 2025; Huang et al., 2025), we first use a bidirectional teacher (Wan 2.1 T2V (Wan et al., 2025)) to generate denoising trajectories as supervision targets. We then train the causal student generator $G _ { \theta }$ to regress these teacher trajectories. To align with the few-step inference target (e.g., 1 or 2 steps), we restrict the regression supervision to those specific discrete timesteps used in the downstream stages, rather than the full ODE trajectory. This is implemented via a Diffusion Forcing (Chen et al., 2024) objective where context chunks are noised at levels corresponding to this discrete schedule. Let $\tilde { x } _ { \mathrm { c t x } , t }$ denote the noisy context and $S _ { \phi } ^ { \mathrm { { O D E } } } ( \cdot )$ the ODE-based teacher sampler, the optimization function is defined as:

$$
\mathcal {L} _ {\mathrm{ODE}} (\theta) = \mathbb {E} _ {t, z _ {t}} \left[ \left\| G _ {\theta} (z _ {t}, \tilde {x} _ {\mathrm{ctx}, t}, c) - \mathcal {S} _ {\phi} ^ {\mathrm{ODE}} (z _ {t}, \tilde {x} _ {\mathrm{ctx}, t}, c) \right\| _ {2} ^ {2} \right]. \tag {3}
$$

Autoregressive video generation requires adapting pretrained bidirectional video models into autoregressive generators by replacing bidirectional full-attention with blockwise causal attention. This stage provides stable initialization for subsequent one-step distillation.

Stage II: distribution matching warmup. We employ Self-Forcing Distribution Matching Distillation (Huang et al., 2025) to holistically align the student’s autoregressive distribution $p _ { \theta }$ with the teacher’s distribution. This framework utilizes three models: the causal student $G _ { \theta }$ , a frozen bidirectional teacher $S _ { \mathrm { r e a l } }$ (Real Score), and a dynamically updated bidirectional model $s _ { \mathrm { f a k e } }$ (Fake Score). During training, we first perform autoregressive self-rollout to generate a full clip $\hat { x } _ { 1 : T }$ from the student $p _ { \theta }$ using self-rollout context

$$
\hat {x} _ {\mathrm{ctx}, t} = (x _ {1: S}, \hat {x} _ {t - L: t - 1}):
$$

$$
\hat {x} _ {t} = G _ {\theta} (z _ {t}, \hat {x} _ {\mathrm{ctx}, t}, c), \quad t = 1, \dots , T. \tag {4}
$$

To match distributions, we perturb the entire generated sequence to a random noise level τ to obtain $\hat { x } _ { 1 : T , \tau }$ . The Fake Score model $s _ { \mathrm { f a k e } }$ is trained to estimate the score of the generated distribution via denoising score matching:

$$
\mathcal {L} _ {\text { score }} (\phi) = \mathbb {E} _ {\hat {x} \sim p _ {\theta}, \tau , \epsilon} \left[ \| s _ {\text { fake }} (\hat {x} _ {1: T, \tau}, \tau , c) - \epsilon \| _ {2} ^ {2} \right]. \tag {5}
$$

Concurrently, the generator $G _ { \theta }$ is updated to minimize the distribution divergence using the gradients derived from the discrepancy between real and fake scores:

$$
\begin{array}{l} \nabla_ {\theta} \mathcal {L} _ {\mathrm{DMD}} = - \mathbb {E} _ {\hat {x} \sim p _ {\theta}, \tau} \left[ \left(s _ {\text { real }} (\hat {x} _ {1: T, \tau}, \tau , c) \right. \right. \\ \left. - s _ {\text {fake}} \left(\hat {x} _ {1: T, \tau}, \tau , c)\right) ^ {\top} \nabla_ {\theta} \hat {x} _ {1: T} \right]. \tag {6} \\ \end{array}
$$

Compared to teacher forcing distillation, this self-rollout distribution matching effectively bridges the train-test gap.

Stage III: asymmetric adversarial refinement. We refine the one-step generator with adversarial training. We construct a discriminator $D _ { \psi }$ using the Wan 2.1 T2V (Wan et al., 2025) backbone initialized from pre-trained weights. Following the APT (Lin et al., 2025a) architecture, we insert cross-attention heads at the 19th, 29th, and 39th transformer layers to aggregate spatiotemporal features into a scalar score. Unlike APT which operates on clean inputs, we apply Gaussian noise to the discriminator inputs according to a randomly sampled timestep τ . This noise injection is essential for stabilizing the training of our asymmetric generator-discriminator pair. We sample a generated clip $\hat { x } _ { 1 : T }$ by rolling out the causal generator autoregressively:

$$
\hat {x} _ {t} = G _ {\theta} (z _ {t}, (x _ {1: S}, \hat {x} _ {t - L: t - 1}), c), \quad t = 1, \dots , T. \tag {7}
$$

We train a discriminator $D _ { \psi }$ on full clips (hence bidirectional during training), while keeping $G _ { \theta }$ strictly causal. Let $x _ { 1 : T , \tau } = \alpha _ { \tau } x _ { 1 : T } + \sigma _ { \tau } \epsilon$ and $\hat { x } _ { 1 : T , \tau } = \alpha _ { \tau } \hat { x } _ { 1 : T } + \sigma _ { \tau } \epsilon .$ , with $\epsilon \sim \mathcal { N } ( 0 , I )$ , denote real and generated clips perturbed at timestep τ , which is also provided to the discriminator. Using the standard logistic GAN objective, we optimize

$$
\begin{array}{l} \mathcal {L} _ {D} (\psi) = - \mathbb {E} _ {x \sim p _ {\text {data}}, \tau} \left[ \log D _ {\psi} (x _ {1: T, \tau}, \tau , c) \right] \tag {8} \\ \left. - \mathbb {E} _ {\hat {x} \sim p _ {\theta}, \tau} \left[ \log \left(1 - D _ {\psi} \left(\hat {x} _ {1: T, \tau}, \tau , c\right)\right) \right], \right. \\ \end{array}
$$

$$
\mathcal {L} _ {G} (\theta) = - \mathbb {E} _ {\hat {x} \sim p _ {\theta}, \tau} \left[ \log D _ {\psi} \left(\hat {x} _ {1: T, \tau}, \tau , c\right) \right]. \tag {9}
$$

To stabilize training, we employ approximated R1 and R2 regularizations (Lin et al., 2025a), penalizing the discriminator’s sensitivity to small perturbations on real and generated samples, respectively:

$$
\begin{array}{l} \mathcal {L} _ {\text { reg }} (\psi) = \mathbb {E} _ {x, \tau} \left[ \| D _ {\psi} (x _ {1: T, \tau}, \tau , c) - D _ {\psi} (x _ {1: T, \tau} + \delta , \tau , c) \| ^ {2} \right] \\ + \mathbb {E} _ {\hat {x}, \tau} \left[ \| D _ {\psi} (\hat {x} _ {1: T, \tau}, \tau , c) - D _ {\psi} (\hat {x} _ {1: T, \tau} + \delta , \tau , c) \| ^ {2} \right], \tag {10} \\ \end{array}
$$

![](images/47d9d39b9fe5239970388c790eed01fbbd5ad4b5fb2fea8bc4f63b5ce8892821.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Generator (Causal DiT)"] -->|adapt causal attention| B["x N blocks"]
    B --> C["Flow-Matching Loss"]
    C --> D["Clean image"]
    C --> E["Noisy chunks"]
    D --> F["Pred clean chunks"]
    E --> G["Pred clean chunks"]
```
</details>

![](images/4784a26132f6ef0314c07c760ed241905969888c5c65f35b02063a5c7b515116.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["DMD Loss"] --> B["Generator (Causal DiT)"]
    B --> C["Real Score"]
    B --> D["Fake Score"]
    C --> E["-"]
    D --> E
```
</details>

![](images/cc08e92b78f301676e9ab135781079d2910f3eaa82695e470d4a43d28feac3b8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Generator Loss"] --> B["Generator (Causal DiT)"]
    B --> C["Pred video"]
    C --> D["Discriminator (Bidirectional DiT)"]
    D --> E["Discriminator (Bidirectional DiT)"]
    E --> F["Real Video"]
    F --> G["Head"]
    G --> H["Concat Query"]
    H --> I["LayerNorm & Linear"]
    I --> J["Real/Fake"]
    J --> K["Discriminator Loss"]
    K --> L["Head Architecture"]
    L --> M["Linear"]
    L --> N["Add&Norm"]
    L --> O["Cross Attention"]
    O --> P["Learnable query tokens"]
    Q["Bidirectional DiT block"] --> R["Head"]
    S["Bidirectional DiT block"] --> T["Head"]
    U["Bidirectional DiT block"] --> V["Head"]
    W["Discriminator Architecture"] --> X["Discriminator Loss"]
```
</details>

Figure 3. Training Pipeline. We train a one-step autoregressive generator $G _ { \theta }$ through three stages. (a) Stage I: ODE initialization replaces bidirectional attention in pre-trained video models with block-wise causal attention, trained by diffusion-forcing with flowmatching loss. (b) Stage II: One-step DMD Warmup distills a strong diffusion teacher under self-rollout training by matching real and fake scores, bringing the student distribution close to the teacher. (c) Stage III: Asymmetric Adversarial Refinement autoregressively rolls out $G _ { \theta }$ and trains it against a bidirectional discriminator. The discriminator uses bidirectional DiT blocks where a single group of learnable query tokens are used to aggregate full video context for video-level discrimination.

where $\delta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ is a small perturbation applied at the same discriminator timestep τ . The discriminator is optimized with $\mathcal { L } _ { D } + \lambda \mathcal { L } _ { \mathrm { r e g } }$ . The bidirectional discriminator aggregates full video context through learnable query tokens, providing stronger temporal consistency signals including sensitivity to long-horizon drift.

Rationale for staged training design. Directly training an asymmetric setup (causal $G _ { \theta }$ with a bidirectional $D _ { \psi } )$ is empirically unstable in the 1-step regime. The ODE and DMD stages move the student close to the teacher distribution, after which adversarial refinement can focus on improving visual quality and temporal coherence. Furthermore, since the teacher distribution and the real data distribution are inherently misaligned, adopting a DMD2-style joint DMD+GAN loss (Yin et al., 2024a) causes the two objectives to conflict: the DMD loss pulls the generator toward the teacher while the GAN loss pulls it toward real data, resulting in unstable training dynamics (Tong et al., 2025; Cheng et al., 2025). Separating them into sequential stages avoids this instability. We find this three-stage design crucial for stable training and high-quality results.

Long-video generation mechanisms. To enable stable infinite streaming, we adopt a Sink Token + Sliding Window attention mechanism (Xiao et al., 2023). We dedicate the first few tokens as “sink tokens” that always participate in attention to preserve global identity information, combined with a local sliding window for recent motion context. Furthermore, we implement Relative RoPE (similar to StreamingLLM (Xiao et al., 2023)) to handle positional encoding extrapolation, ensuring that the relative distances between query and key embeddings remain within the training distribution regardless of the absolute frame index.

Implementation details. We employ the 14B Wan 2.1 T2V model as our backbone. For Image-to-Video (I2V), we encode the conditioning frame into the first KV cache position as a standalone chunk, while subsequent generation uses a chunk size of 4. We set the attention sink size to 1 and local window size to 9. Stages 1 and 2 follow Self Forcing (Huang et al., 2025). Specifically, we train the Stage 1 ODE model for 2,000 steps. In Stage 2, we set the update frequency ratio between the generator and the fake score model to 1:5. We train the DMD generator for only 100 steps and employ early stopping, as prolonged training empirically leads to motion collapse. In Stage 3, we initialize the discriminator with the Wan 2.1 T2V backbone and an APT-style head (Lin et al., 2025a), inserting crossattention blocks at layers 19, 29, and 39. We utilize the approximated R1 and R2 regularizations as described in the Method section to stabilize the 14B model, setting the regularization weight λ = 20 with a perturbation scale of $\sigma _ { \mathrm { r e g } } = 0 . 0 5$ . Additionally, we apply timestep-dependent Gaussian noise to the discriminator inputs, sampling $\tau \sim$ U[0, 1000] to match the generator’s noise schedule. For the generator, we use a learning rate of $4 \times 1 0 ^ { - 7 }$ with EMA decay 0.98; for the discriminator, we do not apply EMA and set the backbone learning rate to $1 \times 1 0 ^ { - 6 }$ and the head learning rate to $2 \times 1 0 ^ { - 6 }$ . We use a batch size of 256 via gradient accumulation for training stability and train the generator for 200 steps.

Table 1. Quantitative comparison on VBench-I2V (Huang et al., 2024). We compare our method against autoregressive baselines using 4-NFE sampling (CausVid (Yin et al., 2025) and Self Forcing (Huang et al., 2025)), and include the bidirectional model Wan 2.1 I2V (Wan et al., 2025) with 100-NFE sampling (50 steps with CFG guidance) as reference. Our model with full three-stage training achieves state-of-the-art performance among autoregressive methods using only a single sampling step. The best result in each column is shown in bold, and the second-best result is underlined. 

<table><tr><td rowspan="2">Method</td><td colspan="6">Quality</td><td colspan="2">Condition</td></tr><tr><td>Subject Consistency↑</td><td>Background Consistency↑</td><td>Motion Smoothness↑</td><td>Dynamic Degree↑</td><td>Aesthetic Quality↑</td><td>Imaging Quality↑</td><td>I2V Subject↑</td><td>I2V Background↑</td></tr><tr><td>Bidirectional</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wan 2.1 I2V (100 NFE)</td><td>93.88</td><td>94.86</td><td>98.14</td><td>51.09</td><td>64.97</td><td>70.12</td><td>96.80</td><td>98.59</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CausVid (4 NFE)</td><td>83.45</td><td>89.37</td><td>98.61</td><td>33.80</td><td>61.55</td><td>70.60</td><td>92.91</td><td>83.34</td></tr><tr><td>Self Forcing (4 NFE)</td><td>91.77</td><td>93.41</td><td>98.55</td><td>34.93</td><td>60.96</td><td>71.50</td><td>95.79</td><td>91.18</td></tr><tr><td>Ours (1 NFE, Stage-II)</td><td>92.14</td><td>92.13</td><td>98.04</td><td>50.30</td><td>58.64</td><td>69.37</td><td>96.56</td><td>95.12</td></tr><tr><td>Ours (1 NFE, Stage-III)</td><td>94.34</td><td>95.08</td><td>98.22</td><td>41.46</td><td>60.07</td><td>71.49</td><td>98.65</td><td>97.83</td></tr></table>

# 5. Experiments

We evaluate the effectiveness of our proposed Asymmetric Adversarial Distillation framework on large-scale video generation benchmarks. We focus on two key aspects: (1) the quality and stability of few-step streaming generation compared to autoregressive and diffusion baselines, and (2) the impact of discriminator architecture design on training stability and motion quality.

# 5.1. Comparison with State-of-the-Art Methods

We evaluate I2V short-video generation under the official VBench standard protocol, producing 5-second clips at a unified 480p resolution. We compare against representative diffusion and autoregressive baselines in Table 1, including Wan 2.1 (Wan et al., 2025), CausVid (Yin et al., 2025), and Self Forcing (Huang et al., 2025). For CausVid and Self Forcing, we follow their published evaluation settings and report zero-shot results. Table 1 reports per-aspect VBench metrics on both generation quality and conditioning faithfulness. Overall, our method achieves strong I2V conditioning performance and imaging quality. Figures 4 and 5 provide qualitative comparisons and user preferences, respectively.

As shown in Table 1, our one-step model achieves competitive generation quality compared to multi-step autoregressive baselines while requiring only a single forward pass. In particular, the Stage-III model achieves the best autoregressive performance in subject consistency (94.34), background consistency (95.08), and I2V subject faithfulness (98.65), while also reaching 97.83 on I2V background faithfulness and 71.49 on imaging quality. Compared with CausVid and Self Forcing, our method substantially improves scene coherence and conditioning preservation, indicating that the proposed asymmetric adversarial distillation effectively stabilizes long-horizon generation. We also observe a clear trade-off between Stage-II and Stage-III training: Stage-II yields stronger motion magnitude (Dynamic Degree 50.30), whereas Stage-III provides better consistency and faithfulness overall. Figures 4 and 5 further support these findings: our method reduces identity drift and receives higher user preference scores in perceptual comparisons.

We further assess perceptual quality via a side-by-side user study on motion realism and image quality. Figure 5 shows that our method is preferred over both Self Forcing and CausVid, indicating stronger perceived quality.

# 5.2. Ablation Studies

We investigate optimal training strategies for one-step causal generation. We first examine the necessity of the stagewise DMD training pipeline in Figure 6, and then ablate discriminator topology at the 14B scale to understand what forms of adversarial supervision lead to stable long-horizon motion. Finally, we analyze why a full-step causal teacher can be unreliable as supervision due to drift.

For the Causal Backbone settings in our ablation, we initialize the discriminator from the Stage 2 DMD-trained generator, ensuring both models start from the same distribution. We also enforce the exact same block-wise causal attention mask. Regarding the logit heads: for video-wise logits, the learnable query token performs cross-attention over the entire spatiotemporal sequence to aggregate global features; for frame-wise logits, the query token performs cross-attention restricted to individual frame tokens independently, lacking global temporal aggregation capabilities.

![](images/6e981fc707773c24b78777b70a8b1c73ff05faf89f2c5a7d4a9a8880137a3d8c.jpg)

<details>
<summary>text_image</summary>

Prompt: a group of jellyfish swimming in an aquarium
Ours
CauseVid
Self Forcing
frame=0 frame=64 frame=128 frame=192 frame=256 frame=320
</details>

Figure 4. Qualitative comparison. We compare our method against autoregressive baselines using 4-NFE sampling (CausVid (Yin et al., 2025) and Self Forcing (Huang et al., 2025)). Given a conditioning image of a swimming jellyfish, our method synthesizes vivid motion while maintaining visual fidelity and identity consistency over long horizons (up to 320 frames), whereas baselines exhibit identity drift.

![](images/37f39d78c7e5444fd1c72704c2c19b4a1e471079f62024e47c6dd54bbbf78aa6.jpg)

<details>
<summary>bar_stacked</summary>

Preference Study Results
| Study | Ours (%) | Ours preferred (%) | Tie (%) | Other preferred (%) |
| :--- | :--- | :--- | :--- | :--- |
| CausVid | 72.5 | 16.2 | 11.2 | 0 |
| Self Forcing | 45.0 | 31.0 | 24.0 | 0 |
</details>

Figure 5. User Preference Study. Win rates of our method against baselines (Self Forcing, CausVid). Our method is preferred in the majority among these methods.

![](images/58c97a8faa43aacb49722fa52f17fd2cf24bfb8c8372a810fdb42890febaae7c.jpg)

<details>
<summary>natural_image</summary>

Six-panel image showing a street scene with neon light effects, labeled 'initial frame', 'w DMD warmup', and 'w/o DMD warmup' (no text or symbols on the scenes themselves)
</details>

Figure 6. Stage-wise ablation of DMD warmup. DMD warmup helps stabilize subsequent adversarial refinement and prevents severe visual degradation.

Ablation on DMD warmup. We ablate the DMD warmup stage to verify whether adversarial refinement alone can reliably train a one-step autoregressive generator. As shown in Table 2 and Figure 6, removing DMD warmup leaves the initial generator distribution too far from the data distribution, making the subsequent GAN objective unstable and causing severe visual degradation. With DMD warmup, the generator starts from a much better one-step solution, preserving scene structure and object appearance before adversarial training improves temporal realism.

Table 2. Ablation on DMD warmup. DMD warmup improves one-step generation quality before adversarial refinement. 

<table><tr><td>Method</td><td>Aesthetic Quality↑</td><td>Imaging Quality↑</td></tr><tr><td>w/o DMD warmup</td><td>53.63</td><td>62.81</td></tr><tr><td>w/ DMD warmup</td><td>58.64</td><td>69.37</td></tr></table>

Analysis of discriminator architectures. We systematically analyze the impact of discriminator topology (Table 3), with qualitative examples shown in Figure 7, through the lens of our theoretical proofs in Appendix A. The evaluation is conducted on 100 videos randomly sampled from the VBench-I2V benchmark and our dataset. We measure Dynamic Degree on 5-second videos, while Drift Score is evaluated on 20-second rollouts to better capture longhorizon error accumulation. The primary driver of performance is the backbone’s causality. As proven in Proposition A.1, causal discriminators effectively suffer from linear error accumulation. A causal backbone prevents the futureanchored gradients necessary to critique early decisions based on global outcomes (Proposition A.2). For causal backbones, granularity is critical: frame-wise heads produce completely static videos (Dynamic Degree 1.08), while video-wise heads restore motion (42.07) but still exhibit severe drift. We attribute the motion collapse in frame-wise discrimination to a trivial solution: since the discriminator only evaluates the marginal distribution $p ( x _ { t } )$ of each frame independently, and any previous frame $x _ { t - 1 }$ is itself a perfectly realistic image, the generator can achieve a high discriminator score by simply copying $G ( x _ { < t } ) = x _ { t - 1 }$ , producing static video. Video-wise heads avoid this failure mode by enforcing temporal coherence across the sequence.

For bidirectional backbones, both granularity settings perform comparably, with video-wise logits achieving slightly

Prompt: Drone wide-angle flyover of Cancun beach at sunset, orange–pink–purple sky reflecting on turquoise water above golden sand.

![](images/e49ee28b2b5c3e45532d59118afda7adb93408daa597368fe7ae1988e93e3b2c.jpg)

<details>
<summary>natural_image</summary>

Three-panel sequence showing a beach sunset scene with visible sun flare and video frames, no text or symbols present.
</details>

Prompt: A brown horse is trotting along a dirt path leading towards a small village surrounded by rolling hills and lush green fields.

![](images/a52d9e42a8d79fdf30f3f4ea1348cc6fd9708aa91dffb2cc59f6750335e5245c.jpg)

<details>
<summary>text_image</summary>

Causal DiT
Frame level
Causal DiT
Video level
Bid. DiT
Video level
</details>

Figure 7. Qualitative ablation study. We compare generated motion under four settings: (a) Causal backbone w/ frame-wise logits results in completely static videos; (b) Causal backbone w/ video-wise logit and (c) Bidirectional backbone w/ frame-wise logits are both prone to drift, exhibiting erratic camera movement, excessive speed, or color shifts. (d) Bidirectional backbone w/ video-wise logit (Ours) achieves the best performance with stable generation.

Table 3. Ablation on Discriminators. We compare Causal vs. Bidirectional visibility and Frame-wise vs. Video-wise granularity. Causal + Frame-wise produces completely static videos (Dynamic Degree 1.08); Causal + Video-wise has high dynamics but severe drift. Bidirectional backbones provide stable supervision, with Video-wise logits achieving the best drift mitigation. 

<table><tr><td>Backbone</td><td>Logit Granularity</td><td>Drift Score↓</td><td>VBench Dynamics↑</td></tr><tr><td>Causal DiT</td><td>Frame-wise</td><td>N/A</td><td>1.08</td></tr><tr><td>Causal DiT</td><td>Video-wise</td><td>7.10</td><td>42.07</td></tr><tr><td>Bidirectional DiT</td><td>Frame-wise</td><td>4.38</td><td>39.04</td></tr><tr><td>Bidirectional DiT</td><td>Video-wise</td><td>4.02</td><td>39.29</td></tr></table>

better drift mitigation (4.02 vs. 4.38). We hypothesize that bidirectional attention already enables deep feature interaction across the entire spatiotemporal volume within the bidirectional DiT backbone, which makes the head’s aggregation strategy less critical.

Drift in a full-step causal teacher. To isolate the limitations of causal supervision itself, we construct a full-step causal teacher by adapting a Wan 2.1 T2V model (Wan et al., 2025) into a causal generator using the 1.3B variant. Specifically, we replace bidirectional attention with a block-wise causal mask, allowing tokens within a frame chunk to attend bi-directionally while preventing attention to future chunks. We train this causal teacher using Diffusion Forcing (Chen et al., 2024), which conditions the current chunk’s denoising process on noisy versions of previous chunks to bridge the train–test gap. At inference time, the model generates videos autoregressively in a chunk-wise manner.

![](images/93eb75257cf3761beeb8bb3d3ac2c4738da1efb8b430f8047654b679cfd0cdda.jpg)  
Figure 8. Drift in Causal Video Diffusion Model. Long-horizon rollout from the full-step causal teacher.

However, even when this full-step causal teacher converges, we observe severe autoregressive error accumulation: long-horizon rollouts exhibit geometric distortion and identity loss (Figure 8), suggesting a drifting distribution pdrift(x1:T ). Using such a drifting causal teacher directly as a discriminator $D ( x _ { 1 : T } )$ ) can therefore provide flawed supervision, since the drifting trajectory remains high-likelihood under the teacher itself. This motivates our asymmetric adversarial distillation with a bidirectional discriminator that can provide future-anchored critiques.

![](images/729cdcb0581d239f4d810e8357a7f389f04c9dabf20f268def99b00ab5f9e127.jpg)  
Figure 9. Effect of regularization coefficient λ. Without regularization (λ = 0), training collapses. Excessive regularization (λ = 50) introduces grid-like patterns. The optimal setting (λ = 20) balances stability and visual quality.

Analysis of regularization coefficient. Beyond architectural choices, we find that the regularization coefficient λ plays a critical role in training stability. As illustrated in Figure 9, setting $\lambda = 0 ( \mathrm { i . e }$ ., removing the regularization term entirely) leads to rapid training collapse, where the generator produces degenerate outputs. Conversely, an overly large coefficient (λ = 50) introduces visible grid-like artifacts in the generated frames, likely due to over-regularization suppressing fine-grained texture details. We empirically find that λ = 20 strikes a good balance, maintaining stable adversarial training while preserving visual fidelity.

# 6. Conclusion

We proposed AAD-1, an asymmetric adversarial distillation framework for one-step autoregressive video generation. By employing a bidirectional discriminator with videolevel holistic discrimination and a phased training strategy with distribution matching warm-up, AAD-1 effectively addresses motion collapse and training instability. Extensive experiments on VBench demonstrate that AAD-1 achieves state-of-the-art performance with superior visual quality and motion fidelity. We hope our work provides valuable insights for efficient autoregressive video generation.

# Limitations

Despite its strong chunk-wise one-step autoregressive generation, our method has limitations in fast motion, complex structures, and long-horizon extrapolation.

Fast motion. The one-step setting can struggle in fastmoving scenes, where large inter-frame motion must be predicted by a single denoising pass rather than refined across multiple sampling steps. In such cases, we observe blurry frames, distorted structures, or degraded temporal coherence, reflecting the difficulty of compressing iterative diffusion sampling into very few steps (Yin et al., 2024b; Lin et al., 2025a). Improving one-step objectives for large motion remains important for robust streaming generation.

Complex structures. Compared with APT2-style onestep-per-image generation (Lin et al., 2025b), where each step can focus on local synthesis for a single image, our chunk-wise one-step setting requires the generator to synthesize multiple latent frames within a chunk in a single forward pass. This makes preserving fine-grained details and subtle local dynamics more challenging, especially for complex and highly structured content such as human faces and hands. These challenges suggest a need for training objectives and generation strategies that better capture complex local structure under chunk-wise one-step generation.

Long-horizon extrapolation. Our adversarial refinement is trained on 5-second clips due to data and compute constraints, as high-quality long-video training data remains scarce and expensive to curate. Although the model can extrapolate beyond this horizon, long rollouts may exhibit drift and quality degradation as errors accumulate over autoregressive chunks, consistent with long-horizon autoregressive video generation challenges (Lin et al., 2025b). We hypothesize that longer-video adversarial training could alleviate this issue by exposing the generator to long-range temporal failures and accumulated rollout errors.

# Acknowledgements

This work was supported in part by the Natural Science Foundation of China under Grant No. 62503323, the Ant Group Research Intern Program, and the Ant Group Postdoctoral Programme.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, specifically in efficient video generation. While our method enables faster autoregressive video synthesis, we acknowledge potential dual-use concerns common to generative models, including the creation of misleading or harmful content. We encourage the development of detection mechanisms and responsible deployment practices alongside this technology. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here beyond these standard considerations for generative AI systems.

# References

Ball, P. J., Bauer, J., Belletti, F., Brownfield, B., Ephrat, A., Fruchter, S., Gupta, A., Holsheimer, K., Holynski, A., Hron, J., Kaplanis, C., Limont, M., McGill, M., Oliveira, Y., Parker-Holder, J., Perbet, F., Scully, G., Shar, J., Spencer, S., Tov, O., Villegas, R., Wang, E., Yung, J., Baetu, C., Berbel, J., Bridson, D., Bruce, J., Buttimore, G., Chakera, S., Chandra, B., Collins, P., Cullum, A., Damoc, B., Dasagi, V., Gazeau, M., Gbadamosi, C., Han, W., Hirst, E., Kachra, A., Kerley, L., Kjems, K., Knoepfel, E., Koriakin, V., Lo, J., Lu, C., Mehring, Z., Moufarek, A., Nandwani, H., Oliveira, V., Pardo, F., Park, J., Pierson, A., Poole, B., Ran, H., Salimans, T., Sanchez, M., Saprykin, I., Shen, A., Sidhwani, S., Smith, D., Stanton, J., Tomlinson, H., Vijaykumar, D., Wang, L., Wingfield, P., Wong, N., Xu, K., Yew, C., Young, N., Zubov, V., Eck, D., Erhan, D., Kavukcuoglu, K., Hassabis, D., Gharamani, Z., Hadsell, R., van den Oord, A., Mosseri, I., Bolton, A., Singh, S., and Rocktaschel, T. Genie 3: A¨ new frontier for world models. 2025.   
Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., and Ramesh, A. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simul   
Chen, B., Mart´ı Monso, D., Du, Y., Simchowitz, M., ´ Tedrake, R., and Sitzmann, V. Diffusion forcing: Nexttoken prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081– 24125, 2024.   
Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al. Skyreelsv2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.   
Cheng, J., Ma, B., Ren, X., Jin, H. H., Yu, K., Zhang, P., Li, W., Zhou, Y., Zheng, T., and Lu, Q. Phased one-step adversarial equilibrium for video diffusion models. arXiv preprint arXiv:2508.21019, 2025.   
Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., and Hsieh, C.-J. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.   
Feng, R., Zhang, H., Yang, Z., Xiao, J., Shu, Z., Liu, Z., Zheng, A., Huang, Y., Liu, Y., and Zhang, H. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024.   
Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D. J. Video diffusion models. Advances in

neural information processing systems, 35:8633–8646, 2022.   
Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., Hold-Geoffroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.   
Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.   
Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.   
Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song, S. L., Rajbhandari, S., and He, Y. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.   
Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. Hunyuanvideo: A systematic framework for large video generative ors.models. arXiv preprint arXiv:2412.03603, 2024.   
Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He, X., Ye, Y., Yuan, S., Chen, L., et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.   
Lin, S., Xia, X., Ren, Y., Yang, C., Xiao, X., and Jiang, L. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316, 2025a.   
Lin, S., Yang, C., He, H., Jiang, J., Ren, Y., Xia, X., Zhao, Y., Xiao, X., and Jiang, L. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025b.   
Liu, K., Hu, W., Xu, J., Shan, Y., and Lu, S. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.   
Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.   
Lu, Y., Ren, Y., Xia, X., Lin, S., Wang, X., Xiao, X., Ma, A. J., Xie, X., and Lai, J.-H. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16818– 16829, 2025a.

Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., Cheng, K. L., Zhu, J., Cao, H., Zhang, Z., Zhu, X., et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025b.   
Mao, X., Jiang, Z., Wang, F.-Y., Zhang, J., Chen, H., Chi, M., Wang, Y., and Luo, W. Osv: One step is enough for high-quality image to video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12585–12594, 2025.   
Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.   
Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103. Springer, 2024.   
Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.   
Shao, S., Yi, H., Guo, H., Ye, T., Zhou, D., Lingelbach, M., Xu, Z., and Xie, Z. Magicdistillation: Weak-to-strong video distillation for large-scale few-step synthesis. arXiv preprint arXiv:2503.13319, 2025.   
Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. 2023.   
Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W., Luo, W., et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.   
Tong, S., Ma, N., Xie, S., and Jaakkola, T. Flow map distillation without data. arXiv preprint arXiv:2511.19428, 2025.   
Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36:8406–8441, 2023.   
Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

Xu, Y., Zhao, Y., Xiao, Z., and Hou, T. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8196–8206, 2024.   
Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al. Longlive: Realtime interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.   
Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., and Freeman, B. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.   
Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6613–6623, 2024b.   
Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22963–22974, 2025.   
Yuan, H., Chen, W., Cen, J., Yu, H., Liang, J., Chang, S., Lin, Z., Feng, T., Liu, P., Xing, J., et al. Lumos-1: On autoregressive video generation from a unified model perspective. arXiv preprint arXiv:2507.08801, 2025.   
Zhang, L. and Agrawala, M. Frame context packing and drift prevention in next-frame-prediction video diffusion models. arXiv preprint arXiv:2504.12626, 2025.

# A. Theoretical Analysis of Ablation Settings

Notation. Let $x _ { 1 : T }$ denote a video clip (conditioned on context c). We denote the data distribution by $p ( x _ { 1 : T } )$ and the causal generator’s rollout distribution by $q ( x _ { 1 : T } )$ ). Let $x _ { < t } \triangleq x _ { 1 : t - 1 }$ . For distributions $P , Q$ with densities $p , q$ , the KL divergence is $\mathrm { K L } ( P \| Q ) \triangleq \mathbb { E } _ { x \sim P } [ \log ( p ( x ) / q ( x ) ) ]$ ].

# A.1. On-Policy Error Accumulation in Causal Rollouts

Proposition A.1 (Linear Error Accumulation). Let $\begin{array} { r } { p ( x _ { 1 : T } ) = \prod _ { t = 1 } ^ { T } p _ { t } ( x _ { t } \mid x _ { < t } ) } \end{array}$ and $\begin{array} { r } { q ( x _ { 1 : T } ) = \prod _ { t = 1 } ^ { T } q _ { t } ( x _ { t } \mid x _ { < t } ) } \end{array}$ be two autoregressive distributions. If the expected on-policy conditional KL divergence is bounded by ε at each step, i.e.,

$$
\forall t, \quad \mathbb {E} _ {x _ {<   t} \sim q} \left[ \mathrm{KL} \left(q _ {t} (\cdot \mid x _ {<   t}) \| p _ {t} (\cdot \mid x _ {<   t})\right) \right] \leq \varepsilon , \tag {11}
$$

then the joint KL divergence satisfies $\mathrm { K L } ( q ( x _ { 1 : T } ) | | p ( x _ { 1 : T } ) ) \leq T \varepsilon$ .

Proof. We expand the KL divergence definition using the chain rule for autoregressive models.

$$
\begin{array}{l} \mathrm{KL} (q \| p) = \int q (x _ {1: T}) \log \frac {\prod_ {t = 1} ^ {T} q _ {t} (x _ {t} \mid x _ {<   t})}{\prod_ {t = 1} ^ {T} p _ {t} (x _ {t} \mid x _ {<   t})} d x _ {1: T} \\ = \sum_ {t = 1} ^ {T} \int q (x _ {1: T}) \log \frac {q _ {t} (x _ {t} \mid x _ {<   t})}{p _ {t} (x _ {t} \mid x _ {<   t})} d x _ {1: T}. \\ \end{array}
$$

Consider the t-th term in the summation. We decompose $q ( x _ { 1 : T } ) = q ( x _ { < t } ) q _ { t } ( x _ { t } \mid x _ { < t } ) q ( x _ { > t } \mid x _ { \leq t } )$ and integrate out the future variables $\boldsymbol { x } _ { > t } \boldsymbol { : }$

$$
\begin{array}{l} \int q (x _ {1: T}) \log \frac {q _ {t} (x _ {t} \mid x _ {<   t})}{p _ {t} (x _ {t} \mid x _ {<   t})} d x _ {1: T} \\ = \int q (x _ {<   t}) \left[ \int q _ {t} (x _ {t} \mid x _ {<   t}) \log \frac {q _ {t} (x _ {t} \mid x _ {<   t})}{p _ {t} (x _ {t} \mid x _ {<   t})} d x _ {t} \right] d x _ {<   t} \\ = \mathbb {E} _ {x _ {<   t} \sim q} \Big [ \mathrm{KL} \big (q _ {t} (\cdot | x _ {<   t}) \| p _ {t} (\cdot | x _ {<   t}) \big) \Big ]. \\ \end{array}
$$

Substituting this back into the sum and applying the bound from Eq. (11), we obtain:

$$
\mathrm{KL} (q \| p) = \sum_ {t = 1} ^ {T} \mathbb {E} _ {x _ {<   t} \sim q} [ \mathrm{KL} _ {t} ] \leq \sum_ {t = 1} ^ {T} \varepsilon = T \varepsilon .
$$

Remark. This result highlights that controlling the one-step error ε on the generator’s own induced distribution (on-policy matching) is sufficient to bound the sequence-level drift linearly in T . Our Stage III self-rollout training explicitly targets this on-policy minimization.

# A.2. Analysis of backbone visibility

Proposition A.2 (Future-Anchored Gradients in Bidirectional Backbones). Let $s _ { t } ( x _ { 1 : T } ) = \operatorname { H e a d } ( H _ { t } )$ be the discriminator logit for frame t, where $H _ { t }$ is the backbone representation.

1. Causal backbone: If the backbone is causal, $H _ { t }$ depends only on $x _ { < t }$ . Thus, $\begin{array} { r } { \frac { \partial s _ { t } } { \partial x _ { > t } } = 0 } \end{array}$ ∂st = 0. ∂x>t   
$H _ { t }$ $x _ { 1 : T }$ $\frac { \partial s _ { t } } { \partial x _ { > t } } \neq 0$

Proof. Case (i): causal backbone. A causal backbone enforces a mask $M _ { i j } = 0$ for $j > i .$ . The representation $H _ { t }$ at index t is computed as a function of inputs $x _ { 1 } , \ldots , x _ { t }$ only. Formally, $H _ { t } = f _ { t } ( x _ { \leq t } )$ . For any suffix variation $x _ { > t } ^ { \prime } \neq x _ { > t }$ , we have $H _ { t } ( x _ { \le t } , x _ { > t } ) = H _ { t } ( x _ { \le t } , x _ { > t } ^ { \prime } )$ , implying $s _ { t }$ is invariant to future frames. Consequently, gradients cannot propagate from future content violations back to time t.

Case (ii): bidirectional backbone. A bidirectional backbone allows attention to all tokens. The representation is a function of the full sequence: $H _ { t } = g _ { t } ( x _ { 1 : T } )$ . A perturbation in the future $x _ { > t }$ alters $H _ { t }$ via the attention mechanism, changing $s _ { t } .$ By the chain rule, ∂st∂x $\begin{array} { r } { \frac { \partial s _ { t } } { \partial x _ { > t } } = \bar { \frac { \partial s _ { t } } { \partial H _ { t } } } \frac { \partial H _ { t } } { \partial x _ { > t } } } \end{array}$ 0st= , which is non-zero. This mechanism allows the discriminator to act as an ”anchor,” penalizing step t if it is inconsistent with the (ground-truth) future $x _ { > t }$ provided during offline training. □

Note on causal backbone with a video-wise head. In the setting with a causal backbone and a video-wise head, the final score $S = \mathrm { P o o l } ( \{ s _ { t } \} _ { t = 1 } ^ { T } )$ depends on all frames. However, the feature extraction $H _ { t }$ remains causal. The future dependency is ”late fusion” (gradients flow from $S \mathrm { t o } H _ { t }$ based on pooling weights, but $H _ { t }$ itself does not contain future features). In contrast, a bidirectional backbone provides ”early fusion,” enriching $H _ { t }$ with future context directly.

# A.3. Analysis of logit granularity

Proposition A.3 (Video-wise Heads Subsume Frame-wise Heads). Let the backbone outputs be $H = [ H _ { 1 } , \dots , H _ { T } ]$ ]. A frame-wise head queries only $H _ { t }$ to score frame $t ,$ while a video-wise head queries $H _ { 1 : T }$ . The class of functions implementable by video-wise heads strictly includes those implementable by frame-wise heads.

Proof. Consider a standard attention mechanism $\mathsf { A t t n } ( Q , K , V )$ . The frame-wise head for frame t computes $y _ { t } ^ { \mathrm { f r a m e } } =$ $\mathrm { A t t n } ( Q _ { t } , H _ { t } W _ { K } , H _ { t } W _ { V } )$ . The video-wise head computes $y ^ { \mathrm { v i d e o } } = \mathrm { A t t n } ( Q _ { \mathrm { g l o b a l } } , H W _ { K } , H W _ { V } )$ with a mixing mask M . We can emulate the frame-wise behavior in the video-wise architecture by constructing a block-diagonal mask M in the video-wise head such that query tokens corresponding to time t can only attend to keys at time t (setting $M _ { i , j } = - \infty$ if tokens $( i ) \in t , t o k e n s ( j ) \notin t )$ . Under this masking, the softmax normalizes only over single-frame tokens, recovering the exact computation of the frame-wise head (assuming shared weights). Since the video-wise head can instantiate this block-diagonal masking pattern while also allowing cross-frame attention patterns, it is strictly more expressive. □

# B. Additional Quantitative Results

Drift score. Following Reward Forcing (Lu et al., 2025b), we quantify long-horizon visual drift by computing the standard deviation of imaging-quality scores along the temporal horizon. Specifically, we evaluate imaging quality over temporal segments of each long rollout and average the resulting standard deviation across videos. A lower Drift Score indicates more stable visual quality over time.

We provide additional quantitative results on VBench-I2V (Huang et al., 2024) to complement the main paper. Beyond the standard 1-NFE, 480p, 5-second setting, we evaluate a 2-NFE variant, 20-second rollouts, and zero-shot 720p generation.

The 2-NFE variant is included as an inference-budget reference. It uses the same three-stage training pipeline as AAD-1, including ODE initialization, DMD warmup, and asymmetric adversarial refinement. For adversarial stabilization, we follow Self Forcing (Huang et al., 2025) and add timestep-dependent Gaussian noise to the discriminator inputs. For generated rollouts corresponding to a given generator output timestep, the discriminator noise level is sampled from the associated timestep interval, keeping the noised discriminator inputs consistent with the generator’s output distribution. As shown in Table 4, the slightly larger sampling budget improves motion smoothness and dynamic degree while maintaining strong I2V subject and background faithfulness.

The 20-second and 720p settings are evaluated in a zero-shot manner from the standard AAD-1 model, without additional training on longer videos or higher-resolution data. These results help illustrate how different inference settings affect temporal consistency, motion dynamics, visual quality, and image-to-video condition preservation.

# C. Training Cost and Memory

We provide additional details on the training cost and memory footprint of our method. Full training takes approximately 3.5 days on 64 NVIDIA H20 $\mathrm { G P U s } ,$ including about 0.5 day for Stage I, 1 day for Stage II, and 2 days for Stage III. To reduce memory usage, we employ Ulysses-style context parallelism (Jacobs et al., 2023) with context parallel size 8 together with PyTorch activation checkpointing. Under the same Stage III setup, namely 64 H20 GPUs, 8 GPUs per node, and Ulysses-style context parallelism with $\mathrm { c p } = 8$ , the bidirectional discriminator adversarial training reaches a peak total GPU memory usage of approximately 1040 GB and requires about 49 hours of training, while the causal discriminator adversarial training baseline uses approximately 830 GB and requires about 65 hours. The bidirectional discriminator incurs a higher memory cost because it processes the full sequence jointly; however, it can exploit FlashAttention-3 (Shah et al., 2024) for efficient full-sequence attention, whereas the causal discriminator relies on FlexAttention to implement causal masking, which results in slower training in practice.

Table 4. Additional quantitative results on VBench-I2V (Huang et al., 2024). Wan 2.1 I2V (Wan et al., 2025), sampled with 100 NFE, is included as a bidirectional reference. All AAD-1 variants are evaluated under different inference settings. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Setting</td><td colspan="6">Quality</td><td colspan="2">Condition</td></tr><tr><td>Subject Consistency↑</td><td>Background Consistency↑</td><td>Motion Smoothness↑</td><td>Dynamic Degree↑</td><td>Aesthetic Quality↑</td><td>Imaging Quality↑</td><td>I2V Subject↑</td><td>I2V Background↑</td></tr><tr><td colspan="10">Bidirectional reference</td></tr><tr><td>Wan 2.1 I2V</td><td>100 NFE</td><td>93.88</td><td>94.86</td><td>98.14</td><td>51.09</td><td>64.97</td><td>70.12</td><td>96.80</td><td>98.59</td></tr><tr><td colspan="10">AAD-1 variants</td></tr><tr><td>AAD-1</td><td>480p, 5s, 1 NFE</td><td>94.34</td><td>95.08</td><td>98.22</td><td>41.46</td><td>60.07</td><td>71.49</td><td>98.65</td><td>97.83</td></tr><tr><td>AAD-1</td><td>480p, 5s, 2 NFE</td><td>94.03</td><td>95.52</td><td>98.99</td><td>50.04</td><td>59.46</td><td>71.00</td><td>98.06</td><td>98.50</td></tr><tr><td>AAD-1</td><td>480p, 20s, 1 NFE</td><td>84.31</td><td>89.30</td><td>98.93</td><td>60.98</td><td>55.48</td><td>68.61</td><td>97.43</td><td>97.25</td></tr><tr><td>AAD-1</td><td>720p, 5s, 1 NFE</td><td>94.52</td><td>95.63</td><td>98.76</td><td>24.39</td><td>61.03</td><td>72.29</td><td>98.30</td><td>98.70</td></tr></table>

# D. Inference Efficiency

We report latency and throughput on a single H100 GPU following the Self-Forcing protocol. Since runtime depends strongly on model size, we compare 1 NFE and 4 NFE inference at matched parameter scales. As shown in Table 5, reducing the sampling budget from 4 NFE to 1 NFE consistently lowers latency and improves throughput within each scale.

Table 5. Inference efficiency. Latency and throughput are measured on a single H100 GPU. 

<table><tr><td rowspan="2">NFE</td><td colspan="2">1.3B</td><td colspan="2">14B</td></tr><tr><td>Latency (s)↓</td><td>Throughput (FPS)↑</td><td>Latency (s)↓</td><td>Throughput (FPS)↑</td></tr><tr><td>1</td><td>0.289</td><td>43.37</td><td>1.134</td><td>14.33</td></tr><tr><td>4</td><td>0.714</td><td>17.70</td><td>2.822</td><td>5.71</td></tr></table>