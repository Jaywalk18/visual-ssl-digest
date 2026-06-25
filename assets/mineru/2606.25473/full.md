# Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Difusion Distillation in Streaming Video Generation and Interactive World Models

Kaiwen Zheng<sup>1,3</sup>, Guande He<sup>2</sup>, Min Zhao<sup>1</sup>, Jintao Zhang<sup>1</sup>, Huayu Chen<sup>1</sup>, Jianfei Chen<sup>1</sup>, Chen-Hsuan Lin<sup>3</sup>, Ming-Yu Liu<sup>3</sup>, Jun Zhu<sup>1\*</sup>, Qianli Ma<sup>3</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>UT Austin <sup>3</sup>NVIDIA

<sup>\*</sup>Corresponding Author

zkwthu@gmail.com; dcszj@tsinghua.edu.cn; mingyul@nvidia.com https://github.com/NVlabs/rcm

## Abstract

Autoregressive video difusion with causal difusion transformers has emerged as a major paradigm for realtime streaming video generation and action-conditioned interactive world models. In this work, we extend rCM, an advanced difusion distillation framework, to autoregressive video difusion. The core philosophy of rCM lies in the complementarity between forward and reverse divergences, represented by consistency models (CMs) and distribution matching distillation (DMD), respectively, in difusion distillation. This philosophy naturally carries over to the autoregressive setting, where teacher-forcing (TF) provides an ofline, forward-divergence causal training paradigm, while self-forcing (SF) corresponds to an on-policy, reverse-divergence refinement.

Our contributions are: (1) through extensive experiments, we show that teacher-forcing CM is currently the best complement to self-forcing DMD as an initialization strategy (2) we present the first implementation of teacher-forcing-based continuous-time CMs (e.g., sCM/MeanFlow) for autoregressive video difusion, enabled by our custom-mask FlashAttention-2 JVP kernel, achieving 10× faster convergence compared to discrete-time CMs (dCMs) (3) we introduce Causal-rCM, a leading, unified, and scalable algorithm-infrastructure open recipe for difusion distillation and causal training (4) we achieve state-of-the-art streaming video generation performance in both frame-wise and chunk-wise settings, using only synthetic data for training.

Notably, our distilled 2-step causal Wan2.1-1.3B model achieves a VBench-T2V score of 84.63 with only 1 or 2 sampling steps. We further apply Causal-rCM to Cosmos 3, an advanced omnimodal world foundation model for physical AI with action-conditioned generation capability, enabling an interactive world model.

VBench-T2V Performance (based on Wan2.1-1.3B)  
![](images/246e6608767303e4c5a809711eca56a22e36f4031c4e8bf7f73176ff76b56589.jpg)  
Following commen practice, we report VBench results using the official GPT-4o-auamented prompts

Figure 1: State-of-the-art performance of Causal-rCM for streaming video generation (1-step: 84.63). Causal-rCM achieves leading VBench-T2V scores across 1-step, 2-step, and 4-step generation, under both frame-wise and chunk-wise autoregressive regimes.

## Contents

1 Introduction 2
2 Background 4
2.1 Diffusion Models 4
2.2 Diffusion Distillation 5
2.3 Autoregressive Video Diffusion 6
3 Causal-rCM: A Leading, Unified and Scalable Algorithm-Infrastructure Open Recipe for Diffusion Distillation and Causal Training 7
3.1 Algorithms 7
3.1.1 Teacher-Forcing, Teacher-Forcing dCM and Self-Forcing DMD 8
3.1.2 JVP-based Causal Distillation with Teacher-Forcing sCM/MeanFlow 9
3.1.3 Extension to Noisy Context and Custom Step Schedule 10
3.2 Infrastructure 11
3.2.1 Main Components 11
3.2.2 Compatibility Design 12
4 Experiments 13
4.1 Setup 13
4.2 Results 14
4.2.1 Streaming Video Generation 14
4.2.2 Interactive World Model 16
5 Related Work 17
6 Limitations and Future 19
A Theoretical Analysis of TrigFlow-sCM and RF-sCM 26
B FlashAttention-2 JVP Kernel with Custom Masks 30

## 1. Introduction

Video difusion models are widely recognized as a form of world simulators (Brooks et al., 2024; Bao et al., 2024; Kong et al., 2024; Wan et al., 2025; Ali et al., 2025; Gao et al., 2025; Seedance et al., 2026; NVIDIA, 2026). Instead of denoising all frames jointly with a bidirectional-attention difusion transformer, autoregressive (AR) video difusion (Jin et al., 2025; Teng et al., 2025; Chen et al., 2025) performs next-frame or next-chunk prediction with causal-attention difusion transformers. This mirrors the shift from masked difusion (Sahoo et al., 2024; Shi et al., 2024; Zheng et al., 2025) to block difusion (Arriola et al., 2025) in the discrete difusion regime. In this paradigm, the model is autoregressive across frames or chunks, while difusion denoising is performed within each frame or chunk. This enables streaming long video generation (Huang et al., 2025; Yang et al., 2026; Chen et al., 2026), interactive world models (Hong et al., 2025; HunyuanWorld, 2025; He et al., 2025; Robbyant Team et al., 2026), and embodied AR video difusion for closed-loop robot control (Feng et al., 2025; Li et al., 2026; Ye et al., 2026).

Common causal training paradigms, such as teacher-forcing (TF) and difusion-forcing (DF) (Chen et al., 2024), sufer from error accumulation and quality degradation over time during AR difusion inference, commonly known as exposure bias (Schmidt, 2019; Ning et al., 2024). The recent self-forcing paradigm (Huang et al., 2025; Lin et al., 2025) resolves this issue by using on-policy training to tackle the training-inference gap, coupled with distribution matching distillation (DMD) (Yin et al., 2024,) or adversarial GAN losses (Lin et al.,

2025) for difusion step distillation. Self-forcing approaches have pushed AR video difusion toward practical low-latency, real-time, and long-horizon generation in streaming and interactive settings.

![](images/34b47f8db8028cd27f6fa4750f0fc59bd9b4cb462718389937350608a689578d.jpg)  
Figure 2: A unified divergence perspective of rCM (Zheng et al., 2025) and Causal-rCM.

However, self-forcing with DMD or GAN objectives is sensitive to initialization and sufers from mode collapse, as DMD-style objectives are based on reverse-KL divergence and optimize student-generated rollouts. Existing AR difusion systems therefore introduce diferent initialization strategies before self-forcing, such as ODE-pair regression (Yin et al., 2025; Huang et al., 2025; He et al., 2025; Zhu et al., 2026), difusion-forcing-style causal adaptation (Huang et al., 2025; Robbyant Team et al., 2026), or hybrid TF/DF initialization (Hong et al., 2025). These designs suggest that a stable ofline causal objective is crucial before on-policy distribution matching, but the connection between initialization, causal training paradigms, and distillation losses remains underexplored.

In this work, we introduce Causal-rCM, extending rCM (score-regularized consistency model) (Zheng et al., 2025) to AR video difusion. In rCM, the key insight is the forward-reverse complementarity at the level of distillation objectives: CMs act as forward-divergence, trajectory-preserving objectives, while DMD acts as a reverse-divergence, distribution-matching objective. In AR difusion, an analogous complementarity arises at the level of causal training paradigms, where teacher-forcing provides an ofline, mode-covering training signal and self-forcing provides an on-policy refinement signal under autoregressive rollouts. Based on this correspondence, Causal-rCM uses teacher-forcing CM for few-step causal distillation on ofline causal contexts and teacher trajectories, and self-forcing DMD to directly optimize the inference-time few-step distribution.

## Relation to Prior Art

CMs are widely used as initialization or regularization for DMD- and GAN-based difusion distillation (Lin et al., 2025: Zheng et al. 2025). Notably for AR diffusion, APT2 (Lin et al. 2025) has adopted teacher-forcing-based CM as initialization for the self-forcing stage, with later theoretical support through the lens of frame-level injectivity (Zhu et al., 2026). Causal-rCM difers from previous works by (1) providing a unified divergence perspective on diferent causal training paradigms, distillation losses, and their synergy, echoing the high-level principle of rCM; (2) conducting a holistic and systematic investigation of diferent initialization strategies for self-forcing DMD, uncovering their pros and cons; (3) providing the first implementation of teacher-forcing based continuous-time consistency models (sCM (Lu and Song, 2024), MeanFlow (Geng et al., 2025)) with our custom-mask FlashAttention-2 JVP kernel, achieving 10× faster convergence compared to discrete-time CMs (dCMs); (4) introducing a leading, unified, and scalable algorithm-infrastructure open recipe for difusion distillation and causal training, achieving state-of-the-art performance in AR difusion distillation.

## Forward-Reverse Objective Complementarity

The broader philosophy of jointly leveraging forward and reverse objectives has appeared across difusion mid-training, difusion reinforcement learning, and difusion distillation. Forward or ofline objectives, such as difusion losses, teacher-forcing losses, and CM losses on real data or teacher trajectories, provide stable training signals and preserve mode coverage. Reverse or on-policy objectives, such as DMD, adversarial losses, and reward-driven optimization on generated samples, directly improve the generated distribution but are more sensitive to initialization and coverage. As summarized in Table 1, recent methods including DDO (Zheng et al., 2025), DifusionNFT (Zheng et al., 2025), DDRL (Ye et al., 2025), and rCM (Zheng et al., 2025) all benefit from this complementarity. Causal-rCM instantiates the same principle in AR difusion distillation: teacher-forcing CM serves as the forward/ofline component, while self-forcing DMD serves as the reverse/on-policy component.

Table 1: Forward–reverse objective complementarity across difusion mid-training, distillation, and RL.

<table><tr><td>Method</td><td>Domain</td><td>Forward Component (Pretrain / Offline)</td><td>Reverse Component (Posttrain / On-policy)</td><td>Effect / Takeaway</td></tr><tr><td>DDO (Zheng et al., 2025)</td><td>diffusion / AR mid-training</td><td>diffusion loss on real data</td><td>anti-likelihood diffusion loss on self-generated negatives</td><td>new record FIDs on ImageNet without auxiliary data/model</td></tr><tr><td>DiffusionNFT (Zheng et al., 2025)</td><td>diffusion RL</td><td>forward-process diffusion objective</td><td>reward-ranked positive / negative generated samples</td><td>25× efficiency</td></tr><tr><td>DDRL (Ye et al., 2025)</td><td>diffusion RL</td><td>forward-KL / diffusion-loss regularization to offline data</td><td>GRPO-style reward optimization on generated rollouts</td><td>alleviating reward hacking and diversity collapse</td></tr><tr><td>rCM (Zheng et al., 2025)</td><td>diffusion distillation</td><td>(s)CM loss on data / teacher trajectories</td><td>DMD loss on student-generated samples</td><td>alleviating mode collapse</td></tr><tr><td>Causal-rCM</td><td>AR diffusion distillation</td><td>teacher-forcing CM on offline causal contexts</td><td>self-forcing DMD on autoregressive student rollouts</td><td>TF-CM initializes SF with causal structure and mode coverage</td></tr></table>

Notes. The complementarity can be realized either in a single joint stage or in a forward-to-reverse order across separate stages. We use “on-policy” to emphasize self-generated samples or rollouts; in difusion RL, such data can be online but of-policy in the strict RL sense.

## 2. Background

## 2.1. Difusion Models

Difusion models (DMs) (Ho et al., 2020; Song et al., 2020) learn continuous data distributions by gradually perturbing clean data $\pmb { x } _ { 0 } \sim p _ { \mathrm { d a t a } }$ with Gaussian noise, which generates a trajectory $\{ \pmb { x } _ { t } \} _ { t = 0 } ^ { T }$ along with associated marginals $\{ q _ { t } \} _ { t = 0 } ^ { T } .$ , and then learning to reverse this process. The forward process follows a closed-form transition kernel $q _ { t | 0 } ( \pmb { x } _ { t } | \pmb { x } _ { 0 } ) = \mathcal { N } ( \alpha _ { t } \pmb { x } _ { 0 } , \sigma _ { t } ^ { 2 } \pmb { I } )$ with predefined noise schedule $\alpha _ { t } , \sigma _ { t } ,$ , enabling reparameterization as ${ \pmb x } _ { t } = \alpha _ { t } { \pmb x } _ { 0 } + \sigma _ { t } { \pmb \epsilon } , { \pmb \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , { \pmb I } )$ . The sampling process of DMs can follow the probability flow ordinary diferential equation (PF-ODE) $\begin{array} { r } { \mathrm { d } \pmb { x } _ { t } = \left[ f ( t ) \pmb { x } _ { t } - \frac { 1 } { 2 } g ^ { 2 } ( t ) \nabla _ { \pmb { x } _ { t } } \log q _ { t } ( \pmb { x } _ { t } ) \right] \mathrm { d } t } \end{array}$ , where $\begin{array} { r } { f ( t ) = \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } , g ^ { 2 } ( t ) = \frac { \mathrm { d } \sigma _ { t } ^ { 2 } } { \mathrm { d } t } - 2 \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } \sigma _ { t } ^ { 2 } } \end{array}$ and $\nabla _ { \pmb { x } _ { t } } \log q _ { t } ( \pmb { x } _ { t } )$ is the score function (Song et al., 2020). A key property of DMs is the theoretical equivalence of diferent parameterizations: the network may predict the score $( \nabla _ { \pmb { x } _ { t } } \log q _ { t } ( \pmb { x } _ { t } ) )$ , the noise (??), the clean data $\mathbf { \Pi } ( \pmb { x } _ { 0 } )$ , or the velocity (??), with optimal predictors being analytically interconvertible (Zheng et al., 2023). With velocity parameterization ?? (Zheng et al., 2023), DMs are trained by minimizing the mean square error (MSE) $\mathbb { E } _ { { \pmb x } _ { 0 } \sim p _ { \mathrm { d a t a } } , { \epsilon } , t } [ { \pmb w } ( t ) \lVert { \pmb v } _ { \theta } ( { \pmb x } _ { t } , t ) - { \pmb v } \rVert _ { 2 } ^ { 2 } ]$ , where the regression target is ${ \pmb v } = \dot { \alpha } _ { t } { \pmb x } _ { 0 } + \dot { \sigma } _ { t } { \pmb \epsilon }$ (denote $\dot { f } _ { t } : = \mathrm { d } f _ { t } / \mathrm { d } t )$ and the PF-ODE is simplified to $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \pmb { v } _ { \theta } ( \pmb { x } _ { t } , t ) } \end{array}$ , commonly known as flow matching (Lipman et al., 2022). A notable special case, rectified flow (RF) (Liu et al., 2022), employs the schedule $\alpha _ { t } = 1 - t , \sigma _ { t } = t$ , which simplifies the velocity target to ${ \pmb v } = { \pmb \epsilon } - { \pmb x } _ { 0 }$

## 2.2. Difusion Distillation

## Consistency Distillation

Consistency models (CMs) (Song et al., 2023) aim to learn a consistency function $\pmb { f } _ { \theta } : ( \pmb { x } _ { t } , t ) \mapsto \pmb { x } _ { 0 }$ which maps the point $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ at arbitrary time ?? on the teacher PF-ODE trajectory to the initial point $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . Given a freeform student network ${ \cal F } _ { \theta } ( { \pmb x } , t )$ , the consistency function is usually parameterized as ${ \pmb f } _ { \theta } ( { \pmb x } , t ) = c _ { \mathrm { s k i p } } ( t ) { \pmb x } +$ $c _ { \mathrm { o u t } } ( t ) { \cal F } _ { \theta } ( c _ { \mathrm { i n } } ( t ) { \bf x } , c _ { \mathrm { n o i s e } } ( t ) )$ , with $c _ { \mathrm { s k i p } } ( 0 ) = 1$ and $c _ { 0 \mathrm { u t } } ( 0 ) = 0 ~ ( \mathrm { e . g . , } ~ f _ { \theta } ( { \pmb x } , t ) = { \pmb x } - t { \pmb F } _ { \theta } ( { \pmb x } , t )$ under the RF schedule). This parameterization naturally satisfies the boundary condition ${ \pmb f } _ { \theta } ( { \pmb x } , 0 ) \equiv { \pmb x }$ . Here, $f _ { \theta }$ is the direct counterpart of the data predictor (denoiser) in $\mathrm { D M } s ,$ while $F _ { \theta } ( { \boldsymbol { x } } , t )$ corresponds to the velocity predictor ${ \pmb v } _ { \theta }$

The CM objective enforces consistent student outputs at adjacent timesteps $t - \Delta t$ and ?? along the teacher trajectory. Discrete-time CMs (dCMs) minimize the following objective with $\Delta t > 0$ :

$$
\mathcal {L} _ {\mathrm{dCM}} (\theta) = \mathbb {E} _ {\boldsymbol {x} _ {0} \sim p _ {\mathrm{data}}, \boldsymbol {\epsilon}, t} \left[ w (t) d \left(\boldsymbol {f} _ {\theta} (\boldsymbol {x} _ {t}, t), \boldsymbol {f} _ {\theta^ {-}} (\hat {\boldsymbol {x}} _ {t - \Delta t}, t - \Delta t)\right) \right],\tag{1}
$$

where $w ( \cdot )$ is a positive weighting function, $d ( \cdot , \cdot )$ is a distance metric, ??<sup>−</sup> is the stop-gradient version of $\theta ,$ and $\hat { \pmb { x } } _ { t - \Delta t }$ is obtained by solving the teacher PF-ODE from $( x _ { t } , t )$ to $t - \Delta t$ with numerical solvers.

Continuous-time CMs (sCM) (Lu and Song, 2024) take the limit $\Delta t \to 0$ in dCM to obtain a more accurate objective. When $d ( \pmb { x } , \pmb { y } ) = \| \pmb { x } - \pmb { y } \| _ { 2 } ^ { 2 }$ , the instantaneous CM loss become $\begin{array} { r } { \mathbb { E } _ { \pmb { x } _ { 0 } \sim p _ { \mathrm { d a t a } } , \epsilon , t } \left[ \pmb { w } ( t ) \pmb { f } _ { \theta } ( \pmb { x } _ { t } , t ) ^ { \top } \frac { \mathrm { d } \pmb { f } _ { \theta } - ( \pmb { x } _ { t } , t ) } { \mathrm { d } t } \right] } \end{array}$ where $\begin{array} { r } { \frac { \mathrm { d } f _ { \theta ^ { - } } ( { \pmb x } _ { t } , t ) } { \mathrm { d } t } = \nabla _ { { \pmb x } _ { t } } { \pmb f } _ { \theta ^ { - } } ( { \pmb x } _ { t } , t ) \frac { \mathrm { d } { \pmb x } _ { t } } { \mathrm { d } t } + \partial _ { t } { \pmb f } _ { \theta ^ { - } } ( { \pmb x } _ { t } , t ) } \end{array}$ is the tangent of $\pmb { f } _ { \theta } \mathrm { a t } \left( \pmb { x } _ { t } , t \right)$ along the teacher ODE trajectory $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } = \pmb { v } _ { \mathrm { t e a c h e r } } ( \pmb { x } _ { t } , t ) } \end{array}$ . This tangent can be eficiently computed by forward-mode automatic diferentiation, Jacobian-vector product (JVP): $\begin{array} { r } { \frac { \mathrm { d } f _ { \theta ^ { - } } ( \mathbf { x } _ { t } , t ) } { \mathrm { d } t } = \mathrm { J } \mathrm { V P } ( f _ { \theta ^ { - } } , ( \mathbf { x } _ { t } , t ) , ( \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } , 1 ) ) } \end{array}$ . sCM further applies MSE reformulation and tangent normalization, reducing the loss to

$$
\mathcal {L} _ {\mathrm{sCM}} (\theta) = \mathbb {E} _ {\boldsymbol {x} _ {0} \sim p _ {\text { data }}, \epsilon , t \sim p _ {G}} \left[ \left\| \boldsymbol {F} _ {\theta} (\boldsymbol {x} _ {t}, t) - \boldsymbol {F} _ {\theta^ {-}} (\boldsymbol {x} _ {t}, t) - \frac {\boldsymbol {g}}{\| \boldsymbol {g} \| _ {2} ^ {2} + c} \right\| _ {2} ^ {2} \right], \quad \boldsymbol {g} = w (t) \frac {\mathrm{d} \boldsymbol {f} _ {\theta^ {-}} (\boldsymbol {x} _ {t} , t)}{\mathrm{d} t}\tag{2}
$$

MeanFlow (Geng et al., 2025) can be viewed as combining sCM with consistency trajectory models (CTMs) (Kim et al., 2023) under the RF schedule. CTMs extend CMs by adding another time condition $s < t$ and defining a consistency trajectory function $\mathbf { \delta } f _ { \theta } : ( \mathbf { \boldsymbol { x } } _ { t } , t , s ) \mapsto \mathbf { \boldsymbol { x } } _ { s } ,$ which maps the point $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ to a less noisy point $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { s } }$ on the teacher ODE trajectory. The infinitesimal jump from ?? to $t ,$ i.e., $f _ { \theta } ( x _ { t } , t , t )$ , reduces to the difusion denoiser and serves as an anchor for applying the difusion loss. This anchor enhances training stability, preserves multi-step sampling, and enables training few-step models from scratch. Thus, CTMs can be viewed as an interpolation between DMs and CMs. In the continuous-time case, CTMs can be optimized with an objective similar to sCM:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{sCTM}} (\theta) = \mathbb {E} _ {\boldsymbol {x} _ {0} \sim p _ {\mathrm{data}}, \epsilon , t, s} \left[ \left\| \boldsymbol {F} _ {\theta} (\boldsymbol {x} _ {t}, t, s) - \boldsymbol {F} _ {\theta^ {-}} (\boldsymbol {x} _ {t}, t, s) - \frac {\boldsymbol {g}}{\| \boldsymbol {g} \| _ {2} ^ {2} + c} \right\| _ {2} ^ {2} \right], \quad \boldsymbol {g} = w (t) \frac {\mathrm{d} \boldsymbol {f} _ {\theta^ {-}} (\boldsymbol {x} _ {t} , t , s)}{\mathrm{d} t} \\ = \mathbb {E} _ {\boldsymbol {x} _ {0} \sim p _ {\mathrm{data}}, \epsilon , t, s} \left[ \frac {\| \Delta_ {\theta} \| _ {2} ^ {2}}{\| \Delta_ {\theta^ {-}} \| _ {2} ^ {2} + c} \right], \quad \Delta_ {\theta} = \boldsymbol {F} _ {\theta} (\boldsymbol {x} _ {t}, t, s) - \boldsymbol {F} _ {\theta^ {-}} (\boldsymbol {x} _ {t}, t, s) - \frac {\mathrm{d} \boldsymbol {f} _ {\theta^ {-}} (\boldsymbol {x} _ {t} , t , s)}{\mathrm{d} t} \end{array}\tag{3}
$$

Under the RF schedule, we have $\pmb { f } _ { \theta ^ { - } } ( \pmb { x } _ { t } , t , s ) : = \pmb { x } _ { t } - ( t - s ) \pmb { F } _ { \theta ^ { - } } ( \pmb { x } _ { t } , t , s )$ , and $\begin{array} { r } { \frac { \mathrm { d } \pmb { f } _ { \theta ^ { - } } ( \pmb { x } _ { t } , t , s ) } { \mathrm { d } t } = \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } - \pmb { F } _ { \theta ^ { - } } ( \pmb { x } _ { t } , t , s ) - } \end{array}$ $\begin{array} { r } { ( t - s ) \frac { \mathrm { d } F _ { \theta ^ { - } } ( { \pmb x } _ { t } , t , s ) } { \mathrm { d } t } } \end{array}$ . Since $F _ { \theta }$ is the velocity predictor ${ \pmb v } _ { \pmb { \theta } _ { } }$ , if we take $\frac { \mathrm { d } { \pmb x } _ { t } } { \mathrm { d } t }$ as the ground-truth velocity $\pmb { v } = \epsilon - \pmb { x } _ { 0 } ,$ then

$$
\Delta_ {\theta} = \boldsymbol {v} _ {\theta} (\boldsymbol {x} _ {t}, t, s) - \boldsymbol {v} + (t - s) \mathrm{JVP} (\boldsymbol {v} _ {\theta^ {-}}, (\boldsymbol {x} _ {t}, t, s), (\boldsymbol {v}, 1, 0))\tag{4}
$$

which recovers the MeanFlow objective. Alternatively, we can set $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { t } } { \mathrm { d } t } \ = \ \pmb { v } _ { \mathrm { t e a c h e r } } ( \pmb { x } _ { t } , t ) } \end{array}$ and use the same formulation for distillation, rather than training from scratch.

## Distribution Matching Distillation

Distribution matching distillation (DMD) (Yin et al., 2024,) is a simple and efective type of score distillation (Wang et al., 2023; Zhou et al., 2024). Given a few-step student generator $\pmb { x } _ { 0 } ^ { \theta } = \pmb { G } _ { \theta } ( z ) , z \sim p ( z )$ with prior distribution ??(??), DMD aims to match the student distribution ??<sub>??</sub> with the teacher distribution ?? by minimizing the reverse-KL divergence on their difused marginals:

$$
\mathcal {L} _ {\mathrm{DMD-raw}} (\theta) = \mathbb {E} _ {t} \left[ D _ {\mathrm{KL}} (p _ {\theta} ^ {t} \parallel p _ {\mathrm{teacher}} ^ {t}) \right], \quad \boldsymbol {x} _ {t} \sim q _ {t | 0} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0} ^ {\theta}).\tag{5}
$$

The gradient of this objective can be written as a score diference between the student and teacher distributions:

$$
\nabla_ {\theta} \mathcal {L} _ {\mathrm{DMD-raw}} (\theta) = \mathbb {E} _ {\pmb {z}, \pmb {\epsilon}, t} \left[ w (t) \left(\nabla_ {\pmb {x} _ {t}} \log p _ {\theta} ^ {t} (\pmb {x} _ {t}) - \nabla_ {\pmb {x} _ {t}} \log p _ {\mathrm{teacher}} ^ {t} (\pmb {x} _ {t})\right) ^ {\top} \frac {\mathrm{d} \pmb {x} _ {t}}{\mathrm{d} \theta} \right].\tag{6}
$$

The teacher score is provided by the pretrained DM, while the student score is intractable for a few-step generator. DMD therefore trains an auxiliary fake score network ?? on student-generated samples with ${ \mathcal { L } } _ { \mathrm { f a k e } } ( \phi ) =$ $\mathbb { E } _ { z , \epsilon , t } \left[ \lambda ( t ) \left| \left| \pmb { f } _ { \phi } ( \pmb { x } _ { t } , t ) - \pmb { x } _ { 0 } ^ { \theta } \right| \right| _ { 2 } ^ { 2 } \right]$ , which serves as a proxy for the student score. In denoiser parameterization, the score diference can be written, up to a time-dependent scalar absorbed into $w ( t )$ , as the diference between the fake and teacher denoisers. With the adaptive normalization trick in DMD, the student can be updated with the following stop-gradient MSE objective:

$$
\mathcal {L} _ {\mathrm{DMD}} (\theta) = \mathbb {E} _ {\boldsymbol {x} _ {0} ^ {\theta} \sim p _ {\theta}, \epsilon , t \sim p _ {D}} \left[ \left\| \boldsymbol {x} _ {0} ^ {\theta} - \mathsf {s g} \left[ \boldsymbol {x} _ {0} ^ {\theta} - \frac {\boldsymbol {f} _ {\mathrm{fake}} (\boldsymbol {x} _ {t} , t) - \boldsymbol {f} _ {\mathrm{teacher}} (\boldsymbol {x} _ {t} , t)}{\mathrm{mean} (\mathrm{abs} (\boldsymbol {x} _ {0} ^ {\theta} - \boldsymbol {f} _ {\mathrm{teacher}} (\boldsymbol {x} _ {t} , t)))} \right] \right\| _ {2} ^ {2} \right]\tag{7}
$$

DMD alternates between student $( \mathcal { L } _ { \mathrm { D M D } } ( \theta ) )$ and critic $( { \mathcal { L } } _ { \mathrm { f a k e } } ( \phi ) )$ phases, forming an adversarial training dynamic similar to GANs.

## 2.3. Autoregressive Video Difusion

![](images/e3c20e6d8d240af3e97214e9d6e9827e2a12efb637e89c6b19fec6f6674b8997.jpg)  
(a) Teacher-Forcing (TF)

![](images/d0c4a673b1cdf39f9f640c385f5020a0945a976807d915689630ab101e0d10f0.jpg)  
(b) Diffusion-Forcing (DF)

![](images/4726dc5029e4130195fb845b3d29c49f30b3eb439667a20f7d919a9d383da3e4.jpg)  
(c) Self-Forcing (SF)  
Figure 3: Illustration of causal training paradigms, adapted from Self-Forcing (Huang et al., 2025).

Autoregressive (AR) video difusion factorizes video generation along the temporal dimension. Given a video latent sequence $\pmb { x } _ { 0 } = [ \pmb { x } _ { 0 } ^ { 1 } , \ldots , \pmb { x } _ { 0 } ^ { N } ]$ divided into frames or chunks, an AR model generates each block conditioned on previous blocks: $\begin{array} { r } { p _ { \theta } ( \pmb { x } _ { 0 } ) = \prod _ { i = 1 } ^ { N } p _ { \theta } ( \pmb { x } _ { 0 } ^ { i } | \pmb { x } _ { 0 } ^ { < i } ) } \end{array}$ . Within each temporal block, the model $p _ { \theta } ( \pmb { x } _ { 0 } ^ { i } | \pmb { x } _ { 0 } ^ { < i } )$ still performs difusion denoising, e.g., under the RF schedule, $\begin{array} { r } { \pmb { x } _ { t } ^ { i } = ( 1 - t ) \pmb { x } _ { 0 } ^ { i } + t \pmb { \epsilon } ^ { i } } \end{array}$ with velocity target ${ \pmb v } ^ { i } = \epsilon ^ { i } - { \pmb x } _ { 0 } ^ { i }$ . Diferent from bidirectional video difusion, which denoises all frames jointly with full temporal attention, AR video difusion uses causal attention so that each frame or chunk only attends to past context. This enables KV caching like LLMs and makes the model naturally suitable for streaming and interactive generation.

Fig. 3 illustrates the three causal training paradigms: teacher-forcing (TF), difusion-forcing (DF), and self-forcing (SF).

In TF, the model predicts the current noisy block while attending to clean ground-truth history, i.e., ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } ^ { i } , t | { \pmb x } _ { 0 } ^ { < i } )$ TF is stable and parallelizable via a specific attention mask, but it creates a training-inference gap: during inference, the model must condition on its own generated history rather than ground-truth context.

DF assigns independent noise levels to diferent frames or chunks and trains the model under a block-causal attention mask, i.e., ${ \pmb v } _ { \theta } ( { \pmb x } _ { t _ { i } } ^ { i } , t _ { i } | { \pmb x } _ { t _ { < i } } ^ { < i } )$ . This exposes the model to noisy histories and improves robustness. However, the training-inference gap remains: perturbing ground-truth videos with synthetic noise does not match the errors and artifacts accumulated from model-generated rollouts at inference.

SF directly simulates AR inference during training. The student rolls out chunks sequentially with KV caching, $\tilde { \pmb { x } } _ { 0 } ^ { i } = G _ { \theta } ( z ^ { i } | \tilde { \pmb { x } } _ { 0 } ^ { < i } )$ , and the loss is applied to the self-generated video. Therefore, SF trains the model under its own inference-time context distribution, directly addressing the exposure bias induced by the training-inference gaps in TF and DF. SF must be combined with reverse-type on-policy objectives, such as DMD or GAN losses.

## 3. Causal-rCM: A Leading, Unified and Scalable Algorithm-Infrastructure Open Recipe for Difusion Distillation and Causal Training

## 3.1. Algorithms

![](images/96a5f2df32776d80478950d507e3b02f49091fcef1dd840b20d0ad7d2f21be3d.jpg)  
Figure 4: Comparison between Causal-rCM and other approaches.

To extend rCM to autoregressive difusion, we pair its two distillation objectives (CM, DMD) with two causal training paradigms, teacher-forcing (TF) and self-forcing (SF), respectively. This preserves the forward-reverse correspondence of rCM in the autoregressive setting: TF-CM provides an ofline, forward-type consistency objective, whereas SF-DMD provides an on-policy, reverse-type distribution-matching objective.

TF-CM requires an autoregressive difusion teacher that is evaluated under the same clean-context setting as the student during TF-based distillation. Such a causal teacher can be trained from scratch, or adapted from a pretrained bidirectional difusion model, with TF or DF. It is arguably more reasonable to use TF because it exposes the teacher to clean historical frames, matching the context distribution used in TF-based distillation. The CM component can be instantiated either as the simple dCM or as more advanced continuous-time variants such as sCM and MeanFlow. For SF-DMD, following prior work (Huang et al., 2025; Lin et al., 2025), we use a bidirectional teacher and a bidirectional fake-score network to provide real and fake score estimates on self-generated rollouts (Fig. 3(c)) and apply the DMD loss (Eqn. 7).

Unlike rCM, which combines CM and DMD in a joint-training style, Causal-rCM applies TF-CM and SF-DMD sequentially. The full pipeline consists of three stages: (1) TF converts the bidirectional difusion model into an autoregressive difusion model, which serves as both the causal teacher and the student initialization for the subsequent TF-CM stage; (2) TF-CM distills the causal teacher into a few-step causal student, which serves as the student initialization for the subsequent SF-DMD stage; and (3) SF-DMD refinement further optimizes the student on its own autoregressive rollouts, reducing the training-inference gap and exposure bias. As summarized in Fig. 4, Causal-rCM provides a simple and strong recipe that avoids cumbersome ODE-pair knowledge distillation (KD) (Luhman and Luhman, 2021) and GAN-style post-training, while introducing a novel TF-sCM implementation and achieving state-of-the-art performance.

## 3.1.1. Teacher-Forcing, Teacher-Forcing dCM and Self-Forcing DMD

The core operation of TF-based training is to replace a standard single-state forward with a packed causal forward over concatenated clean context and noisy targets. Concretely, for a velocity predictor, instead of evaluating ${ \pmb v } _ { \theta } ( { \pmb x } _ { t } , t )$ , we evaluate

$$
\left[ \pmb {v} _ {\theta} \left([ \pmb {x} _ {0} ^ {\mathrm{clean}}, \pmb {x} _ {t} ^ {\mathrm{noisy}} ], [ \pmb {0} ^ {\mathrm{clean}}, t ^ {\mathrm{noisy}} ]; M _ {\mathrm{TF}}\right) \right] _ {\mathrm{noisy}},\tag{8}
$$

where $M _ { \mathrm { T F } }$ is the TF attention mask, the clean part provides ground-truth causal context at timestep $0 ,$ and the loss is applied only to the noisy part. The mask ensures that each noisy block attends only to its allowed clean history and its own noisy tokens, matching the TF pattern in Fig. 3. Such TF-mask attention can be implemented with custom-mask attention operators such as FlexAttention (Dong et al., 2024) or MagiAttention (Zewei and Yunpeng, 2025). An alternative is a two-pass implementation: first cache the clean tokens under a block-causal attention mask, and then perform a second forward pass in which noisy tokens attend to the cached clean context. However, this design requires the clean-token KV cache to be retained in the computational graph, making it less compatible with activation checkpointing and more memory-intensive.

With a difusion regression target ${ \pmb v } = { \pmb \epsilon } - { \pmb x } _ { 0 }$ under the RF schedule, the ordinary TF objective is

$$
\mathcal {L} _ {\mathrm{TF}} (\theta) = \mathbb {E} _ {\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, t} \left[ w (t) \left\| \left[ \boldsymbol {v} _ {\theta} \left(\left[ \boldsymbol {x} _ {0} ^ {\text {clean}}, \boldsymbol {x} _ {t} ^ {\text {noisy}} \right], \left[ \boldsymbol {0} ^ {\text {clean}}, \boldsymbol {t} ^ {\text {noisy}} \right]; \boldsymbol {M} _ {\mathrm{TF}}\right) \right] _ {\text {noisy}} - \boldsymbol {v} \right\| _ {2} ^ {2} \right].\tag{9}
$$

This gives a full-step causal difusion model. For TF-dCM, the clean context remains fixed, while the noisy part is moved along the causal teacher PF-ODE trajectory. Let $\hat { \pmb x } _ { t - \Delta t } ^ { \mathrm { n o i s y } }$ be obtained by solving the causal teacher ODE from $\pmb { x } _ { t } ^ { \mathrm { n o i s y } }$ at ?? to $t - \Delta t$ under the same TF mask. The student minimizes

$$
\begin{array}{r l} & {\mathcal {L} _ {\mathrm{TF-dCM}} (\theta) = \mathbb {E} _ {\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, t} \Bigg [ w (t) d \Bigg (\left[ \boldsymbol {f} _ {\theta} \left([ \boldsymbol {x} _ {0} ^ {\mathrm{clean}}, \boldsymbol {x} _ {t} ^ {\mathrm{noisy}} ], [ \boldsymbol {0} ^ {\mathrm{clean}}, \boldsymbol {t} ^ {\mathrm{noisy}} ]; \boldsymbol {M} _ {\mathrm{TF}}\right) \right] _ {\mathrm{noisy}},} \\ & {\qquad \qquad \qquad \left. \left[ \boldsymbol {f} _ {\theta^ {-}} \left([ \boldsymbol {x} _ {0} ^ {\mathrm{clean}}, \hat {\boldsymbol {x}} _ {t - \Delta t} ^ {\mathrm{noisy}} ], [ \boldsymbol {0} ^ {\mathrm{clean}}, \boldsymbol {t} - \boldsymbol {\Delta t} ^ {\mathrm{noisy}} ]; \boldsymbol {M} _ {\mathrm{TF}}\right) \right] _ {\mathrm{noisy}}\right) \Bigg ].} \end{array}\tag{10}
$$

SF-DMD is applied after TF-CM. The student first performs a temporal AR rollout with KV caching. At chunk $i ,$ the model generates the current clean chunk conditioned on the cached states of previous generated chunks:

$$
\tilde {\boldsymbol {x}} _ {0} ^ {i} = \boldsymbol {G} _ {\theta} (\boldsymbol {z} ^ {i} \mid \mathrm{KV} ^ {<   i}), \quad \mathrm{KV} ^ {<   i} = \mathrm{KV} (\tilde {\boldsymbol {x}} _ {0} ^ {<   i}).\tag{11}
$$

After $\tilde { \mathbf { x } } _ { 0 } ^ { i }$ is generated, it is fed once more into the causal transformer through a cache-update forward pass, which appends its clean-token key/value states to the cache:

$$
\mathrm{KV} ^ {\leq i} = \text { Append } \left(\mathrm{KV} ^ {<   i}, \mathrm{KV} _ {\theta^ {-}} (\mathfrak {s g} [ \tilde {\boldsymbol {x}} _ {0} ^ {i} ])\right).\tag{12}
$$

Within each chunk, $G _ { \theta }$ is implemented by few-step self-rollout denoising from pure noise $z ^ { i } ;$ :

$$
\tilde {\boldsymbol {x}} _ {t _ {N}} ^ {i} = \boldsymbol {z} ^ {i} \xrightarrow {\theta^ {-}} \tilde {\boldsymbol {x}} _ {t _ {N - 1}} ^ {i} \xrightarrow {\theta^ {-}} \dots \xrightarrow {\theta^ {-}} \tilde {\boldsymbol {x}} _ {t _ {1}} ^ {i} \xrightarrow {\theta} \tilde {\boldsymbol {x}} _ {0} ^ {i}, \quad 0 <   t _ {1} <   t _ {2} <   \dots <   t _ {N} = 1.\tag{13}
$$

In each training iteration, the number of simulation steps ?? is randomly sampled from $[ 1 , N _ { \mathrm { m a x } } ]$ . Each transition can be instantiated as CM-style reverse denoising followed by forward noising, e.g., under the RF schedule,

$$
\tilde {\boldsymbol {x}} _ {t _ {n - 1}} ^ {i} = (1 - t _ {n - 1}) \boldsymbol {f} _ {\theta} \left(\tilde {\boldsymbol {x}} _ {t _ {n}} ^ {i}, t _ {n} \mid \mathrm{KV} ^ {<   i}\right) + t _ {n - 1} \boldsymbol {\epsilon} _ {n}, \quad \boldsymbol {\epsilon} _ {n} \sim \mathcal {N} (\boldsymbol {0}, \boldsymbol {I}).\tag{14}
$$

The final output $\tilde { \mathbf { \boldsymbol { x } } } _ { 0 } = [ \tilde { \mathbf { \boldsymbol { x } } } _ { 0 } ^ { 1 } , \dots , \tilde { \mathbf { \boldsymbol { x } } } _ { 0 } ^ { N _ { \mathrm { c h u n k } } } ]$ enters the DMD loss. Following standard practice (Yin et al., 2024; Huang et al., 2025), we apply gradient truncation to make SF-DMD memory-eficient. The intermediate denoising steps and previous-chunk KV caches are detached (indicated by $\theta ^ { - } )$ . Only the final denoising step $t _ { 1 } \to 0$ of each chunk is kept diferentiable (indicated by ??), which the DMD loss is back-propagated through.

## 3.1.2. JVP-based Causal Distillation with Teacher-Forcing sCM/MeanFlow

TF-sCM uses the same packed causal forward as TF and TF-dCM, but replaces the finite-step consistency target with a continuous-time tangent target. The clean context is kept fixed, while the noisy tokens move along the causal teacher ODE. Under the RF schedule, define the causal teacher velocity on the noisy branch as

$$
\pmb {v} _ {\mathrm{teacher}} ^ {\mathrm{TF}} = \left[ \pmb {v} _ {\mathrm{teacher}} \left([ \pmb {x} _ {0} ^ {\mathrm{clean}}, \pmb {x} _ {t} ^ {\mathrm{noisy}} ], [ \pmb {0} ^ {\mathrm{clean}}, \pmb {t} ^ {\mathrm{noisy}} ]; \pmb {M} _ {\mathrm{TF}}\right) \right] _ {\mathrm{noisy}}.\tag{15}
$$

The RF consistency map on the noisy branch is

$$
\left[ \boldsymbol {f} _ {\theta} ^ {\mathrm{TF-RF}} \right] _ {\mathrm{noisy}} = \boldsymbol {x} _ {t} ^ {\mathrm{noisy}} - t \left[ \boldsymbol {v} _ {\theta} \left(\left[ \boldsymbol {x} _ {0} ^ {\mathrm{clean}}, \boldsymbol {x} _ {t} ^ {\mathrm{noisy}} \right], \left[ \boldsymbol {0} ^ {\mathrm{clean}}, \boldsymbol {t} ^ {\mathrm{noisy}} \right]; \boldsymbol {M} _ {\mathrm{TF}}\right) \right] _ {\mathrm{noisy}}.\tag{16}
$$

Its continuous-time tangent along the causal teacher trajectory is

$$
\begin{array}{r l} & {\pmb {h} _ {\mathrm{TF-sCM}} = \pmb {v} _ {\mathrm{teacher}} ^ {\mathrm{TF}} - \left[ \pmb {v} _ {\theta^ {-}} \left([ \pmb {x} _ {0} ^ {\mathrm{clean}}, \pmb {x} _ {t} ^ {\mathrm{noisy}} ], [ \pmb {0} ^ {\mathrm{clean}}, \pmb {t} ^ {\mathrm{noisy}} ]; \pmb {M} _ {\mathrm{TF}}\right) \right] _ {\mathrm{noisy}}} \\ & {\qquad - t \left[ \mathsf {J V P} \left(\pmb {v} _ {\theta^ {-}}, \left([ \pmb {x} _ {0} ^ {\mathrm{clean}}, \pmb {x} _ {t} ^ {\mathrm{noisy}} ], [ \pmb {0} ^ {\mathrm{clean}}, \pmb {t} ^ {\mathrm{noisy}} ]\right), ([ \pmb {0} ^ {\mathrm{clean}}, \pmb {v} _ {\mathrm{teacher}} ^ {\mathrm{TF}} ], [ \pmb {0} ^ {\mathrm{clean}}, \pmb {1} ^ {\mathrm{noisy}} ]); \pmb {M} _ {\mathrm{TF}}\right) \right] _ {\mathrm{noisy}}.} \end{array}\tag{17}
$$

Here the JVP is computed through the same TF-masked packed forward as the primal prediction. The tangent of the clean context is zero, and only the noisy branch follows the teacher velocity.

The TF-sCM objective is then

$$
\mathcal {L} _ {\mathrm{TF-sCM}} (\theta) = \mathbb {E} _ {\boldsymbol {x} _ {0}, \boldsymbol {\epsilon}, t} \left[ \left\| \Delta \boldsymbol {v} _ {\theta} ^ {\mathrm{TF}} - \frac {w (t) \boldsymbol {h} _ {\mathrm{TF-sCM}}}{w ^ {2} (t) \| \boldsymbol {h} _ {\mathrm{TF-sCM}} \| _ {2} ^ {2} + c} \right\| _ {2} ^ {2} \right],\tag{18}
$$

where

$$
\Delta \boldsymbol {v} _ {\theta} ^ {\mathrm{TF}} = \left[ \boldsymbol {v} _ {\theta} \left([ \boldsymbol {x} _ {0} ^ {\text {clean}}, \boldsymbol {x} _ {t} ^ {\text {noisy}} ], [ \mathbf {0} ^ {\text {clean}}, \boldsymbol {t} ^ {\text {noisy}} ]; \boldsymbol {M} _ {\mathrm{TF}}\right) - \boldsymbol {v} _ {\theta^ {-}} \left([ \boldsymbol {x} _ {0} ^ {\text {clean}}, \boldsymbol {x} _ {t} ^ {\text {noisy}} ], [ \mathbf {0} ^ {\text {clean}}, \boldsymbol {t} ^ {\text {noisy}} ]; \boldsymbol {M} _ {\mathrm{TF}}\right) \right] _ {\text {noisy}}.\tag{19}
$$

A subtle but important design choice is to use the RF-native form of sCM, rather than wrapping the RF velocity model into TrigFlow and applying the TrigFlow-sCM objective as in rCM (Zheng et al., 2025). Although diferent difusion noise schedules, such as TrigFlow and RF, are analytically convertible up to a time-dependent scaling (Zheng et al., 2023), they generally induce diferent normalized MSE objectives for sCM (Appendix A). In the bidirectional setting, rCM finds the TrigFlow wrapper beneficial for stability. However, in our causal TF setting, the TrigFlow-wrapped TF-sCM results in degraded generation quality, whereas the RF-native TF-sCM produces more smooth outputs.

## 3.1.3. Extension to Noisy Context and Custom Step Schedule

![](images/d6121953ed78a59b3ed5aec2a2cfae43d55aeb2cfbf2c2df6fc0842d6c1150ec.jpg)  
(a) TF with noisy context

![](images/cb652ea7df388da3e2a816a6b9cef325a4559d0fa16d340bac6d9c435f9c6bec.jpg)  
(b) SF with noisy context and custom step schedule  
Figure 5: Adaptation to acceleration techniques: noisy context and custom step schedule.

Noisy context and custom step schedules (Liu et al., 2026) are two simplest and most efective inference acceleration techniques for AR video difusion distillation. Both TF and SF can naturally incorporate them, as illustrated in Fig. 5.

## Noisy Context

Unlike LLMs, AR video difusion must maintain a denoising-time-aware KV cache: standard clean-context AR inference requires an additional clean-context encoding pass after the denoising steps of each chunk, so an ??-step causal difusion model efectively costs ?? + 1 number of function evaluations (NFEs) per chunk. Noisy context removes this extra pass by reusing the KV states from the last denoising step as the context for subsequent chunks, reducing the efective latency from ?? + 1 to ?? NFEs. Besides acceleration, noisy context can improve long-horizon robustness, as residual noise acts as a low-pass filter that suppresses accumulated high-frequency artifacts while preserving coarse motion dynamics (Huang et al., 2025).

In TF, noisy context is incorporated by replacing the clean history in the packed TF forward with noisy historical tokens at the corresponding context timestep, while the loss remains applied only to the current target block. In SF, noisy context is used directly during AR rollout. Although introducing noisy context in the TF stages would better align with inference, we find it suficient in practice to apply it only in the final SF stage.

## Custom Step Schedule

The number of denoising steps can also vary across chunks. In text-to-video generation, the first chunk is typically more demanding because it establishes the global scene, layout, and appearance, whereas later chunks mainly extend the video conditioned on previous context. We therefore allow a chunk-dependent step schedule

$$
[ N _ {1}, N _ {2}, \dots , N _ {N _ {\mathrm{chunk}}} ],\tag{20}
$$

where $N _ { i }$ denotes the number of denoising steps for chunk ??. For example, a nominal 2-step model can use $[ 4 , 2 , 2 , \dots ]$ , allocating extra computation only to the first chunk.

For SF-DMD training, we cycle the rollout length by the training iteration. For example, for a target schedule $[ 4 , 2 , 2 , \dots ]$ , SF-DMD repeatedly cycles through $[ 1 , 1 , 1 , \ldots ] \ \to \ [ 2 , 2 , 2 , \ldots ] \ \to \ [ 3 , 2 , 2 , \ldots ] \ \to \ [ 4 , 2 , 2 , \ldots ] \ \to$ $[ 1 , 1 , 1 , \ldots ] \to \cdots$ . This cycling strategy is important because SF-DMD only back-propagates through the final denoising step of each chunk. Cycling the rollout length makes diferent denoising intervals appear as the final diferentiable step across iterations, rather than supervising only the last interval of the maximum-step sampler.

Table 2: Implementation-level comparison of autoregressive video difusion codebases.

<table><tr><td rowspan="2">Codebase</td><td colspan="2">Recipe Scope</td><td colspan="4">Algorithmic Recipes</td><td colspan="4">Bidirectional Infra</td><td colspan="5">Causal Infra</td></tr><tr><td>Bi.</td><td>Causal</td><td>TF</td><td>DF</td><td>SF</td><td>Replayed</td><td>FSDP2</td><td>CP/SP</td><td>SAC</td><td>JVP</td><td>FSDP2</td><td>CP/SP</td><td>SAC</td><td>JVP</td><td>KV Cache</td></tr><tr><td>Self-Forcing (Huang et al., 2025)</td><td>✕</td><td>√</td><td>√</td><td>√</td><td>√</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td> $\triangle^{v1}$ </td><td>✕</td><td> $\triangle^{AC}$ </td><td>✕</td><td>√post</td></tr><tr><td>FastVideo (Hao-AI Lab, 2026)</td><td>√</td><td>√</td><td>✕</td><td>√</td><td>√</td><td>✕</td><td>√</td><td> $\checkmark^{F-U}$ </td><td>√</td><td>✕</td><td>√</td><td>✕</td><td>√</td><td>✕</td><td>√post</td></tr><tr><td>FastGen (Nie et al., 2026)</td><td>√</td><td>√</td><td>✕</td><td>√</td><td>√</td><td>✕</td><td>√</td><td>✕</td><td> $\triangle^{AC}$ </td><td>✕</td><td>√</td><td>✕</td><td> $\triangle^{AC}$ </td><td>✕</td><td>√post</td></tr><tr><td>(Causal-)rCM</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td> $\checkmark^{F-U}$ </td><td>√</td><td>√</td><td>√</td><td> $\checkmark^{F-U}$ </td><td>√</td><td>√</td><td>√pre/post</td></tr></table>

Notes. <sup>✓</sup>: supported; <sup>✗</sup>: not found; △: partial, unclear, or path-dependent.  
TF: teacher-forcing implemented as [clean frames, noisy frames] concatenation with a special causal mask.  
DF: difusion-forcing with ordinary block-causal masking.  
SF: self-forcing with self-rollout / KV-cache-style training execution.  
Replayed: replayed back-propagation technique that avoids storing the entire computation graph during self-rollout.  
FSDP2: Fully Sharded Data Parallel v2. <sup>v1</sup>: FSDP1-only support.  
CP/SP: context/sequence parallel. <sup>T</sup>: temporal/frame axis; <sup>F</sup>: flattened video-token axis, e.g., flattened T×H×W patch tokens. <sup>U</sup>: DeepSpeed-Ulysses; <sup>R</sup>: Ring Attention; <sup>UR</sup>: Ulysses–Ring hybrid (USP).  
SAC: selective activation checkpointing. <sup>AC</sup>: activation checkpointing, but not clearly op-level SAC.  
JVP: Jacobian-vector-product. Base operator for continuous-time consistency model (sCM/MeanFlow).  
KV Cache: causal self-attention KV cache. <sup>pre</sup>: K is cached before RoPE; <sup>post</sup>: K is cached after RoPE.

## 3.2. Infrastructure

Causal-rCM is designed as an algorithm-infrastructure recipe. Its main infrastructure goal is to make causal training paradigms (TF, DF, and SF), continuous-time JVP-based CMs, and large-scale parallel training mutually compatible. Achieving this requires careful co-design of attention-mask specification, KV caching, FSDP2, context parallelism, activation checkpointing, FlashAttention-2 JVP kernels, and replayed back-propagation. Table 2 summarizes the resulting system-level coverage and highlights the infrastructure advantages of Causal-rCM over other widely used codebases.

## 3.2.1. Main Components

## FlashAttention-2 JVP Kernel with Custom Masks

Continuous-time CMs require the tangent of the network output along the teacher ODE. Computing this tangent with a generic torch.func.jvp over unfused attention is impractical for large video transformers due to the materialization of large attention intermediates and the resulting memory overhead. To enable JVP through fused attention under TF masks, we build on the FlashAttention2-JVP kernel in rCM (Zheng et al., 2025) and extend it to support custom masks. The TF mask is represented as admissible query-key ranges rather than materialized dense matrices. The details are presented in Appendix B.

## Parallelisms

We use FSDP2 (Zhao et al., 2023) as a ZeRO-3-style sharding backend: parameters, gradients, and optimizer states are partitioned across data-parallel ranks, and each module materializes full parameters only for its local computation. This reduces per-GPU model-state memory and makes it feasible to train large video DiTs with student, teacher, fake-score, and EMA networks in the same distillation pipeline. We use distributed checkpointing (DCP) to save and restore the sharded model and optimizer states directly across ranks, avoiding the need to gather full model states on a single process.

We use flattened Ulysses-style context parallelism (CP) (Jacobs et al., 2023) to shard the long video-token sequence across ranks. Specifically, the spatiotemporal video tokens are first flattened into a single sequence, and CP partitions this flattened sequence dimension across P devices. Before attention, each GPU holds a shard of size [B, H, L/P, C] for QKV. An all-to-all operation then redistributes QKV to [B, H/P, L, C] for local attention, followed by another all-to-all to restore the sequence partition of the attention output ??. A key design choice is to make CP transparent to the outer algorithm: the network interface always takes and returns the global full sequence, independent of CP size, while the network internally handles local sequence shards, all-to-all attention, and output gathering.

## Activation Checkpointing

We use selective activation checkpointing (SAC) to reduce activation memory by recomputing only selected parts of the network during backward. Unlike vanilla region-based torch.utils.checkpoint, SAC provides finer-grained control over which operations are recomputed and which intermediates are preserved. In practice, we apply SAC mainly to compute-heavy stateless regions such as attention and MLP blocks, while leaving lightweight or stateful operations outside checkpointed regions.

## KV Cache

The KV cache is used by causal rollout execution and inference. We distinguish three cache modes: disabled mode for ordinary packed training, append mode for committing a generated chunk into the cache, and readonly mode for generating the current chunk while attending to previously committed chunks. Cached K/V tensors are detached by construction, which prevents gradients from propagating through previous chunks and keeps SF-DMD memory bounded. The cache also records chunk boundaries, so a readonly forward can expose only the prefix needed by the current block. This supports both standard AR rollout and variants such as noisy context, where the final denoising forward can be reused as the context state.

We support both pre-RoPE and post-RoPE key caching. Post-RoPE caching is simple and eficient because cached keys can be reused directly. Pre-RoPE caching is useful when the same cached content may need diferen positional treatment, e.g., for length extrapolation or alternative position indexing (Yesiltepe et al., 2026; Yi et al., 2025; Li et al., 2026; Kim et al., 2026). The implementation keeps this choice inside the attention context so that the high-level rollout code does not need to distinguish the two cases.

## Replayed Back-propagation

SF-DMD generates on-policy videos through AR rollout. In the standard execution with gradient-truncation, the final diferentiable denoising steps of all chunk are kept in the computational graph, which can be memoryintensive for long videos. We therefore provide an optional replayed back-propagation mode (Hong et al., 2025) as a memory-saving implementation. The rollout is first constructed without gradients, while storing the final noisy input, timestep, detached KV cache, and DMD target for each chunk. Then, each chunk’s final denoising step is recomputed with gradients enabled, and its gradient is back-propagated separately with gradient accumulation. This trades additional computation for lower activation memory. We deliberately reserve this replayed path for SF-DMD: TF, DF, and TF-CM remain packed, since replaying diferentiable prefix-KV computation ofers limited additional benefit once SAC is enabled.

## 3.2.2. Compatibility Design

A major goal of Causal-rCM is to make advanced causal training features composable. In practice, many components that work independently can conflict when used together. We therefore implement compatibility at the level of execution semantics rather than as independent feature switches.

## SAC × FlexAttention.

Packed TF/DF/TF-CM training relies on custom-mask attention. In the FlexAttention path, the attention pattern is specified by a mask\_mod function and lowered by the PyTorch compiler into a specialized fused attention kernel. To make this compatible with SAC we use torch>=2 . 10 together with

$$
\text { torch\_inductor.config.wrap\_inductor\_compiled\_regions } = \text { True }
$$

which exposes Inductor-compiled FlexAttention calls to SAC as explicit checkpointable regions, internally represented as inductor\_compiled\_code.

## SAC × self-forcing.

SF-DMD rollout is stateful because KV caches and causal metadata evolve across chunks. We make this compatible with SAC by separating persistent cache storage from per-forward causal state: historical K/V tensors are stored as detached context, while each forward constructs a fresh CausalInferenceState describing the current chunk, cache range, and append/read-only mode for future recomputation. The inference state is not reused through in-place updates, so checkpoint recomputation reconstructs the same causal context as the original forward. Cache-append forwards are kept outside checkpointed execution, so recomputation never replays cache mutation; checkpointed regions only read a fixed causal context.

## JVP × FSDP2.

Following rCM, we implement JVP at the layer level, rather than applying a global torch.func.jvp to an FSDP2-wrapped model. Each layer exposes a paired primal-tangent interface, taking (??, ????) as input and returning (??, ????). This corresponds to an FSDP2(JVP) design instead of JVP(FSDP2). FSDP2 continues to manage parameter materialization, sharding, and gradient reduction at layer boundaries, while tangent propagation is performed locally within each layer’s forward computation.

## JVP × Ulysses CP.

Ulysses CP extends naturally to JVP because tangent tensors follow the same communication pattern as their primal counterparts. Specifically, tQ, tK, tV are all-to-all exchanged together with Q, K, V, the local attention computation is replaced by our custom-mask FlashAttention-2 JVP kernel, and the resulting tO is returned through the same output all-to-all as O. We reuse the JVP-compatible distributed-attention design from rCM, while adding custom-mask support for packed TF/DF/TF-CM training.

## KV cache × Ulysses CP.

For rollout execution, cached K/V tensors must be compatible with Ulysses CP. We use a post-all-to-all KV cache, where the cache is stored in the same [B, H/P, L, C] layout as exposed to local attention. Each CP rank directly reuses its head-sharded, full-sequence cached K/V states. This avoids repeatedly converting old cache entries between global and CP-local layouts.

## 4. Experiments

## 4.1. Setup

Table 3: Training configurations for Causal-rCM on Wan2.1 T2V.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">Stage 1</td><td colspan="2">Stage 2</td><td>Stage 3</td></tr><tr><td>Wan2.1-1.3B TF/DF</td><td>Wan2.1-14B TF/DF</td><td>Wan2.1-1.3B TF-dCM</td><td>Wan2.1-1.3B TF-sCM</td><td>Wan2.1-1.3B SF-DMD</td></tr><tr><td>Global batch size</td><td>256</td><td>64</td><td>32</td><td>32</td><td>64</td></tr><tr><td>Context parallel size</td><td>1</td><td>8</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Student optimizer</td><td>AdamW lr =  $10^{-5}$  $\beta = (0.9, 0.999)$ wd = 0.01</td><td>AdamW lr =  $10^{-5}$  $\beta = (0.9, 0.999)$ wd = 0.01</td><td>AdamW lr =  $2 \times 10^{-6}$  $\beta = (0, 0.999)$ wd = 0.01</td><td>AdamW lr =  $2 \times 10^{-6}$  $\beta = (0, 0.999)$ wd = 0.01</td><td>AdamW lr =  $2 \times 10^{-6}$  $\beta = (0, 0.999)$ wd = 0.01AdamW lr =  $4 \times 10^{-7}$  $\beta = (0, 0.999)$ wd = 0.01</td></tr><tr><td>Fake-score optimizer</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CFG scale</td><td>-</td><td>-</td><td>3.0</td><td>3.0</td><td>5.0</td></tr><tr><td>Time sampling / weighting</td><td>TF:  $p_G$ = UniformShift(5), shared t, Gaussian-bell weight; DF:  $p_G$ = UniformShift(5), random per-chunk t, no weight</td><td>TF:  $p_G$ = UniformShift(5), shared t, Gaussian-bell weight; DF:  $p_G$ = UniformShift(5), random per-chunk t, no weight</td><td>uniform RF grid with shift = 3, steps = 48, skip = 1</td><td> $p_G$ = LogitNormal ( $\mu = -0.8, \sigma = 1.6$ )</td><td> $p_D$ = UniformShift(5)</td></tr><tr><td>Specific hyperparameters</td><td>-</td><td>-</td><td>-</td><td>tangent warmup = 1000</td><td rowspan="2">max rollout steps = 4 student update freq. = 6 varies</td></tr><tr><td>Training iterations</td><td>30k</td><td>30k</td><td>10k</td><td>1k</td></tr></table>

## Models and Datasets.

We conduct the main streaming video generation experiments on Wan2.1 T2V (Wan et al., 2025) at 480p resolution. Videos are generated at 832 × 480 spatial resolution with 81 RGB frames, corresponding to 21 latent frames after VAE temporal compression. Training uses the synthetic T2V data provided by rCM (Zheng et al., 2025), generated by the bidirectional Wan2.1-14B teacher with 100-step Euler sampling, shift 3.0, and CFG scale 5.0. We use Wan2.1-1.3B as the main student model and use Wan2.1-14B teachers for distillation.

We evaluate two causal chunk patterns. The frame-wise setting, denoted by c1-1, uses one initial latent frame and then one-latent-frame chunks. The chunk-wise setting, denoted by c3-3, uses one initial latent chunk of three frames and then three-latent-frame chunks. The same chunk pattern is used consistently for packed TF/DF/TF-CM masks, SF-DMD rollout, KV-cache inference, and streaming evaluation.

## Training.

Causal-rCM uses a three-stage training recipe. We report the main hyperparameters in Table 3. For TF-CM, we use 14B causal teachers trained with TF. For SF-DMD, we use 14B birectional teacher and fake score networks.

For few-step SF-DMD, we use RF sampling schedules with a maximum of 4 denoising steps. The 4-step sampler uses intermediate times [15/16, 5/6, 5/8]. The 2-step sampler uses 4 steps for the first chunk and 2 steps for later chunks, with schedule [[15/16, 5/6, 5/8], [5/6]]. The 2-step noisy-context variant uses schedule [[15/16, 5/6, 5/8], [5/8]] and reuses the final denoising forward as the context cache. The 1-step variant uses 4 steps for the first chunk and 1 step for later chunks, with schedule [[15/16, 5/6, 5/8], []].

## Evaluation Metrics.

For streaming quality, we evaluate text-to-video generation with VBench-T2V (Huang et al., 2024), reporting the total score as well as the quality and semantic sub-scores.

For inference eficiency, we report the number of function evaluations (NFE), throughput in frames per second (FPS), first-chunk latency, and second-chunk latency. All eficiency measurements are conducted with batch size 1 on a single H100 GPU. The reported FPS and latency include both difusion sampling and VAE decoding.

## 4.2. Results

## 4.2.1. Streaming Video Generation

## Main Results.

Table 4 compares Causal-rCM against bidirectional Wan2.1 and streaming video generation baselines, including Self-Forcing (Huang et al., 2025), LongLive (Yang et al., 2026), Causal Forcing (Zhu et al., 2026), and AnyFlow (Gu et al., 2026). We report both frame-wise and chunk-wise results. Causal-rCM achieves state-ofthe-art streaming quality while supporting 4-step, 2-step, 2-step noisy-context, and 1-step inference schedules.

Table 4: Main streaming video generation results on Wan2.1 T2V.

<table><tr><td>Method</td><td>NFE</td><td>Total Score↑</td><td>Quality Score↑</td><td>Semantic Score↑</td><td>Throughput↑(FPS)</td><td>First Latency↓(s)</td><td>Second Latency↓(s)</td><td>SF-DMD iters</td></tr><tr><td colspan="9">Bidirectional</td></tr><tr><td>Wan2.1-1.3B</td><td>50×2</td><td>82.78</td><td>83.44</td><td>80.13</td><td>0.72</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Wan2.1-14B</td><td>50×2</td><td>83.35</td><td>83.97</td><td>80.88</td><td>0.18</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="9">Frame-wise (c1-1)</td></tr><tr><td>Causal Forcing (4-step)</td><td>5</td><td>81.56</td><td>82.59</td><td>77.44</td><td>8.3</td><td>0.40</td><td>0.46</td><td>-</td></tr><tr><td>Causal-rCM (4-step)</td><td>5</td><td>84.29</td><td>85.27</td><td>80.36</td><td>8.3</td><td>0.40</td><td>0.46</td><td>1200</td></tr><tr><td>Causal-rCM (2-step)</td><td>3</td><td>84.63</td><td>85.46</td><td>81.31</td><td>12.2</td><td>0.40</td><td>0.31</td><td>3000</td></tr><tr><td>Causal-rCM (2-step, noisy ctx)</td><td>2</td><td>83.11</td><td>83.55</td><td>81.37</td><td>15.9</td><td>0.40</td><td>0.23</td><td>1500</td></tr><tr><td>Causal-rCM (1-step)</td><td>2</td><td>84.63</td><td>85.54</td><td>81.01</td><td>15.9</td><td>0.40</td><td>0.23</td><td>3000</td></tr><tr><td colspan="9">Chunk-wise (c3-3)</td></tr><tr><td>Self-Forcing (4-step)</td><td>5</td><td>83.76</td><td>84.53</td><td>80.68</td><td>17.4</td><td>0.57</td><td>0.64</td><td>-</td></tr><tr><td>LongLive (4-step)</td><td>5</td><td>83.62</td><td>84.36</td><td>80.69</td><td>17.4</td><td>0.57</td><td>0.64</td><td>-</td></tr><tr><td>Causal Forcing (4-step)</td><td>5</td><td>83.96</td><td>84.94</td><td>80.04</td><td>17.4</td><td>0.57</td><td>0.64</td><td>-</td></tr><tr><td>AnyFlow (4-step)</td><td>5</td><td>84.31</td><td>85.15</td><td>80.94</td><td>17.4</td><td>0.57</td><td>0.64</td><td>-</td></tr><tr><td>Causal-rCM (4-step)</td><td>5</td><td>84.37</td><td>85.02</td><td>81.73</td><td>17.4</td><td>0.57</td><td>0.64</td><td>1250</td></tr><tr><td>Causal-rCM (2-step)</td><td>3</td><td>84.30</td><td>85.04</td><td>81.36</td><td>22.2</td><td>0.57</td><td>0.49</td><td>2500</td></tr><tr><td>Causal-rCM (2-step, noisy ctx)</td><td>2</td><td>84.24</td><td>84.96</td><td>81.36</td><td>25.6</td><td>0.57</td><td>0.41</td><td>1750</td></tr><tr><td>Causal-rCM (1-step)</td><td>2</td><td>84.01</td><td>84.71</td><td>81.22</td><td>25.6</td><td>0.57</td><td>0.41</td><td>3000</td></tr></table>

## Performance under Custom Step Schedule and Noisy Context.

Table 4 shows an interesting behavior under custom step schedules. In the frame-wise setting, the 1-step and 2-step Causal-rCM models outperform the 4-step variant, which is counter-intuitive at first glance. We attribute this to the nature of the frame-wise setting: each AR chunk contains only a single latent frame and therefore has no internal temporal structure to denoise. In this case, allocating many denoising steps to every future chunk can over-emphasize autoregressive feedback errors, especially considering the gradient truncating strategy of SF-DMD. Empirically, we observe that 4-step frame-wise SF-DMD is more prone to camera drift, e.g., a consistent leftward camera rotation across samples, and can only be trained stably for about 1k iterations. In contrast, using 1 or 2 steps for later chunks largely suppresses this drift and allows stable training for around 3k iterations. Since each future chunk contains only one latent frame, 1–2 denoising steps are already suficient to generate the frame, and the reduced rollout depth improves stability.

The trend is diferent in the chunk-wise setting, where each chunk contains three latent frames and therefore has non-trivial internal temporal correlation. Here, a deeper 4-step sampler provides a better denoising trajectory for modeling motion and intra-chunk consistency, leading to the best overall score. This suggests that the optimal step schedule depends on the temporal span of each AR chunk: frame-wise generation benefits more from shallow, stable rollout, while chunk-wise generation benefits from additional denoising depth.

Noisy context further improves inference eficiency by eliminating the extra clean-context KV encoding pass, reducing the efective cost from ?? + 1 to ?? NFEs per chunk. Comparing 2-step sampling with noisy context against 1-step sampling, we find that 1-step sampling is better in the frame-wise setting, while 2-step sampling with noisy context is better in the chunk-wise setting. This is consistent with the above observation. For single-frame chunks, the extra denoising step brings limited benefit, while the residual noise in the context can directly afect fine-grained details in frame-level prediction. For three-frame chunks, the chunk contains a higher-dimensional and more redundant spatiotemporal token group. In this regime, Gaussian perturbations are less likely to destroy the entire chunk-level structure uniformly, and much of the motion and coarse semantic context can still be preserved (Hoogeboom et al., 2023). Therefore, 2-step sampling with noisy context can retain the benefit of an additional denoising step for intra-chunk temporal coherence.

## Comparison between TF-dCM and TF-sCM.

![](images/08a7acdfb6966861309d698441f80da9e87e578d7a3d6319b044c8788e479818.jpg)  
Figure 6: Training curves of TF-dCM and TF-sCM.

Fig. 6 compares TF-dCM and TF-sCM before the final SF-DMD stage. TF-sCM consistently provides a stronger initialization with over 10× fewer training iterations. In the frame-wise setting, TF-sCM reaches above 81.8 VBench-T2V score within 1-2k iterations, already surpassing TF-dCM trained for 10k iterations. The gap is even clearer in the chunk-wise setting, where TF-sCM reaches above 83 within 1-2k iterations, while TF-dCM improves more slowly and remains lower after much longer training.

## Ablation Studies on Initialization Strategies.

Table 5 ablates the initialization strategies of SF-DMD. We compare causal difusion initializations from DF and TF, ODE-pair knowledge distillation variants (DF-KD and TF-KD), and teacher-forcing consistency initializations (TF-dCM and TF-sCM). The corresponding training curves are shown in Fig. 7.

Table 5: Ablation of initialization strategies for 4-step SF-DMD.

<table><tr><td>Initialization</td><td>Total Score↑</td><td>Quality Score↑</td><td>Semantic Score↑</td><td>SF-DMD iterations</td></tr><tr><td colspan="5">Frame-wise (c1-1)</td></tr><tr><td>DF</td><td>83.11</td><td>83.85</td><td>80.16</td><td>2000</td></tr><tr><td>TF</td><td>82.62</td><td>83.62</td><td>78.61</td><td>1000</td></tr><tr><td>DF-KD</td><td>80.59</td><td>80.41</td><td>81.32</td><td>2000</td></tr><tr><td>TF-KD</td><td>83.49</td><td>84.50</td><td>79.43</td><td>1250</td></tr><tr><td>TF-dCM</td><td>84.29</td><td>85.27</td><td>80.36</td><td>1200</td></tr><tr><td>TF-sCM</td><td>83.84</td><td>84.67</td><td>80.55</td><td>1000</td></tr><tr><td colspan="5">Chunk-wise (c3-3)</td></tr><tr><td>DF</td><td>84.80</td><td>85.58</td><td>81.65</td><td>1500</td></tr><tr><td>TF</td><td>84.95</td><td>85.82</td><td>81.47</td><td>1000</td></tr><tr><td>DF-KD</td><td>83.61</td><td>84.10</td><td>81.68</td><td>1500</td></tr><tr><td>TF-KD</td><td>83.79</td><td>84.41</td><td>81.30</td><td>1000</td></tr><tr><td>TF-dCM</td><td>84.33</td><td>85.22</td><td>80.75</td><td>3200</td></tr><tr><td>TF-sCM</td><td>84.37</td><td>85.02</td><td>81.73</td><td>1250</td></tr></table>

![](images/b475e2bb9a44d3aaa264c600f7f099d94f2833deecb70bfa0615fb40eb9dc798.jpg)  
(a) Frame-wise

![](images/f8ae472887cbcbdb2c97763c12389b36f64cc660141ae2aad232df671470ab84.jpg)  
(b) Chunk-wise  
Figure 7: SF-DMD training curves with diferent initialization strategies.

In the frame-wise setting, TF-CM initialization achieves the best overall performance, with DF and TF-KD also providing competitive alternatives. Although TF-sCM starts from a stronger initial model, TF-dCM is more stable during SF-DMD and supports longer refinement, leading to a higher peak score. In the chunk-wise setting, DF/TF initialization achieves the highest VBench-T2V scores, close to 85. However, as shown in Fig. 8, these models often produce over-smoothed and over-saturated textures, such as water, hair, and leaves, with noticeably fewer fine-grained details. Considering both VBench scores and qualitative inspection, TF-CM initialization is still the most reliable choice. Among the two TF-CM variants, TF-sCM slightly outperforms TF-dCM while requiring fewer SF-DMD iterations.

## 4.2.2. Interactive World Model

We further apply Causal-rCM to Cosmos 3 (NVIDIA, 2026), an omnimodal world model based on a two-tower Mixture-of-Transformers architecture. Cosmos 3 separates an understanding tower (UND) for text and prompt reasoning from a generation tower (GEN) for vision, action, and sound tokens, while sharing the multimodal attention layers and unified 3D mRoPE across modalities. In the original generator mode, GEN tokens use bidirectional self-attention for multimodal denoising. To support interactive world modeling, we convert the GEN vision stream into a temporal-causal autoregressive difusion stack (Fig. 9).

We treat each latent video frame as a vision supertoken, which contains all spatial latent tokens of that frame. Temporal-causal attention is applied at the supertoken level: future vision supertokens are masked from past and current ones, while spatial tokens within the same vision supertoken remain fully bidirectional.

The same causal stack supports text-to-video, image-to-video, and forward-dynamics (action-conditioned) modeling. In text-to-video, all vision supertokens are generated from text conditioning. In image-to-video and forward dynamics, the first vision supertoken is provided as clean context, and the model predicts future vision supertokens autoregressively. For forward dynamics, action supertokens are treated as input conditions. A null action supertoken is used for the first frame, and real action supertokens are aligned by unified 3D mRoPE to the next generated vision supertoken, so that action $A _ { i }$ controls the transition from state $V _ { i }$ to $V _ { i + 1 }$

![](images/46eb9b4496e586dbba3b66594ca780b987a8643346a1bdd4b67c00e2685d38b5.jpg)  
(a) DF + SF-DMD

![](images/b0bed3248630adaee91cf1a7de313688014b159e3d40d055174fd5fb6f9d79aa.jpg)  
(b) TF + SF-DMD

![](images/08bb67b409e07694ebc2b2afc08e973f1ceceef7266c81aa84b44195f29e226d.jpg)  
(c) TF-dCM + SF-DMD

![](images/6be940ebfb29b71ae5fa662568c68bff2d9dbe4bb3f1f992c32aba22e2f65088.jpg)  
(d) TF-sCM + SF-DMD

Figure 8: Visualizations of chunk-wise SF-DMD under diferent initialization strategies. DF/TF initialization leads to higher VBench-T2V scores while sufering from overly smooth textures and lacking fine-grained details.  
![](images/17e77ed7755d4d718f6c00ddde171d85228984379a635c5e57c33d1772a426d7.jpg)  
Figure 9: From Cosmos 3 to interactive Cosmos 3. Cosmos 3 uses causal self-attention for UND tokens, full cross-attention from GEN to UND tokens, and bidirectional self-attention within GEN tokens. Interactive Cosmos 3 preserves the UND-GEN attention structure but replaces GEN self-attention with temporal-causal attention over latent-frame supertokens. In the forward-dynamics layout, $V _ { i }$ denotes a vision supertoken, $A _ { i }$ controls the transition to $V _ { i + 1 } ,$ and a null action token is inserted before $V _ { 0 }$ to keep a uniform token layout.

As shown in Fig. 10, the interactive Cosmos 3 model supports streaming control: given the same initial scene, the generated future frames follow distinct trajectories under left-turn, right-turn, and stay-forward controls.

## 5. Related Work

## Diferential information and JVPs in generative modeling.

Diferential information has played an important role in difusion ODEs beyond standard first-order denoising supervision. High-order denoising score matching shows that first-order score matching is insuficient for maximum-likelihood difusion ODE training, and controls higher-order score errors to tighten the likelihood gap (Lu et al., 2022). Subsequent work improves difusion ODE likelihood estimation and training with velocity parameterization, variance reduction, and high-order flow-matching objectives (Zheng et al., 2023). DPM-Solver-v3 further uses empirical model statistics of a pretrained difusion model to derive improved ODE solve coeficients, and also reveals numerical issues related to time derivatives in difusion networks (Zheng et al., 2023). More recently, sCM, MeanFlow, AYF, and FACM use JVPs as a direct training signal for continuous-time consistency or flow-map objectives (Lu and Song, 2024; Geng et al., 2025; Sabour et al., 2025; Peng et al., 2025). rCM scales JVP-based consistency distillation to large image and video difusion models by making

![](images/839376dccff0fcf1838b178a08ae43ad64a1b26debe9c405a4c56fb0a1edae3c.jpg)  
Figure 10: Cosmos 3 interactive generation on autonomous-driving scenarios conditioned on the action of the vehicle ego-motion.

JVP computation compatible with FlashAttention, FSDP, and context parallelism, and combines it with DMD regularization (Zheng et al., 2025). Causal-rCM extends this line to autoregressive video difusion, applying JVP-based teacher-forcing sCM under clean causal contexts as a structured initialization for self-forcing DMD.

## Forward-reverse complementarity in distillation objectives.

A growing set of few-step methods can be viewed as combining a coverage-preserving forward component with a quality- or reward-seeking reverse component. For text-to-image generation, recent practical studies standardize large-scale few-step distillation recipes for strong text-conditioned teachers, and empirically compare sCM with MeanFlow (Pu et al., 2025). Flow-map methods distill teacher ODE behavior more directly. FreeFlow (Tong et al., 2025) performs data-free flow-map distillation by sampling from the prior and querying teacher dynamics on student-induced flow-map states, with an additional correction objective to mitigate compounding errors. In contrast, ??-Flow (Chen et al., 2025) is more explicitly on-policy by matching teacher velocities along the student policy’s own ODE trajectory. Distribution-matching methods improve few-step quality but may sacrifice diversity; recent variants therefore introduce role separation, RL signals, or adversarial flow objectives to balance mode coverage and mode seeking (Jiang et al., 2025; Wu et al., 2026; Cheng et al., 2025; Lin et al., 2026). This complementarity is especially explicit in recent long-video work: Cai et al. (2026) pair a supervised global flow-matching head for long-range structure with a local DMD head for short-window fidelity, while HiAR (Zou et al., 2026) observes that self-rollout reverse-KL distillation can amplify low-motion shortcuts and adds a forward-KL regularizer to preserve motion diversity.

## Video and autoregressive difusion distillation.

Video distillation must handle temporal consistency and long-horizon error accumulation in addition to perframe visual quality. Self-Forcing (Huang et al., 2025) and APT2 (Lin et al., 2025) are representative works that propose self-forcing as an on-policy distillation paradigm for mitigating exposure bias in AR generation. In particular, APT2 initializes self-forcing with teacher-forcing consistency distillation, but relies on a relatively cumbersome GAN objective during the self-forcing stage. Concurrent to our work, Causal Forcing++ (Zhao et al., 2026) also combines teacher-forcing consistency with self-forcing DMD, while we implement JVP-based continuous-time consistency under teacher forcing, and provide a systematic algorithm-and-infrastructure open recipe with holistic evaluation. Apart from the CM route, Transition Matching Distillation matches multi-step video denoising trajectories with few-step transition processes using conditional flow heads, followed by distribution matching on flow-head rollouts (Nie et al., 2026). AnyFlow shifts video distillation from endpoint consistency to arbitrary-interval flow-map transitions and uses backward simulation for on-policy distillation in both bidirectional and causal architectures (Gu et al., 2026). Other recent work studies from-scratch few-step video training with eficient solution-flow objectives (Park et al., 2026), video-specific distillation losses fo oversaturation and temporal collapse (You et al., 2026). Adversarial refinement has also been explored for one-step AR video generation, e.g., by augmenting DMD with a noised-latent GAN loss (Feng et al., 2026) or by using asymmetric adversarial distillation after distribution-matching warm-up (Li et al., 2026). The distilled models are orthogonal to attention-level acceleration and could be further combined with sparse attention techniques (Zhang et al., 2025, 2026), as demonstrated by TurboDifusion (Zhang et al., 2025), which combines rCM with attention acceleration and quantization.

Table 6: A high-level view of CM/CTM distillation recipes.

<table><tr><td>Route</td><td>Setting</td><td>Ultimate pipeline</td><td>Related works</td></tr><tr><td>CM</td><td>Bidirectional</td><td>dCM → sCM → DMD/GAN (+ CM/on-policy CM)</td><td>APT (Lin et al., 2025): dCM → GAN; rCM (Zheng et al., 2025): sCM + DMD.</td></tr><tr><td>CM</td><td>Causal</td><td>TF-dCM → TF-sCM → SF-DMD/SF-GAN (+ TF-CM/SF-CM)</td><td>APT2 (Lin et al., 2025): TF-dCM → SF-GAN; CF++ (Zhao et al., 2026): TF-dCM → SF-DMD; Causal-rCM (ours): TF-dCM/TF-sCM → SF-DMD.</td></tr><tr><td>CTM</td><td>Bidirectional</td><td>MeanFlow (FD) → MeanFlow (JVP) → DMD/GAN (+ MeanFlow/on-policy MeanFlow)</td><td>Transition Matching (Nie et al., 2026): MeanFlow (FD) → DMD2-v with flow-head rollout; AnyFlow (Gu et al., 2026): MeanFlow (FD) → DMD + on-policy MeanFlow (FD).</td></tr><tr><td>CTM</td><td>Causal</td><td>TF-MeanFlow (FD) → TF-MeanFlow (JVP) → SF-DMD/SF-GAN (+ TF-MeanFlow/SF-MeanFlow)</td><td>AnyFlow (Gu et al., 2026): TF-MeanFlow (FD) → SF-DMD + SF-MeanFlow (FD).</td></tr></table>

Notes. MeanFlow (FD) denotes finite-diference-estimated MeanFlow, and MeanFlow (JVP) denotes the exact JVP-based MeanFlow. →: diferent stages; +: joint training.

## 6. Limitations and Future

## Limitations.

Although Causal-rCM provides an efective algorithm-infrastructure recipe for autoregressive difusion distillation, several limitations remain. First, frame-wise T2V training with long rollout depth is still fragile. In this setting, the 4-step SF-DMD model tends to develop camera drift after extended training, e.g., a consistent directional camera bias, and therefore cannot be trained for a long duration. This issue could be eliminated in action-conditioned interactive settings, where actions provide an explicit motion prior and reduce the ambiguity of camera evolution. Second, the best initialization before SF-DMD does not always translate into the best final model. TF-sCM gives a stronger pre-SF-DMD initialization than TF-dCM, but in the frame-wise setting, TF-dCM can be more stable under long SF-DMD refinement and achieve a higher final peak. This suggests that initialization quality and refinement stability are not fully aligned. Third, fully joint optimization like rCM remains challenging. In our causal setting, joint training tends to lower the VBench ceiling, so we currently use a staged pipeline. This could be attributed to the distribution gap between the causal teacher and the bidirectional teacher. Finally, the current custom-mask FlashAttention JVP kernel is implemented in Triton. As a result, the per-iteration speed of TF-sCM is only comparable to TF-dCM with standard FlashAttention-2, lacking behind more advanced kernels like FlashAttention-3/4.

## Future directions.

A natural next step is to make the staged recipe more systematic. Table 6 summarizes our high-level view: current distillation methods can be interpreted as subsets of two ultimate pipelines, a CM route and a CTM route, each with bidirectional and causal variants. Discrete-time methods (dCM, MeanFlow with finite diference estimation) could be the warmup stage for continuous-time JVP ones (sCM, MeanFlow) to enhance stability

Beyond algorithmic design, future work should improve the underlying systems stack. Better kernels for custom attention, JVP, and KV-cache execution, together with runtime features such as torch.compile, CUDA Graphs, and NVFP4 could further reduce overhead and make large-scale training and inference more eficient.

## References

[1] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tifany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025. 2

[2] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block difusion: Interpolating between autoregressive and difusion language models. arXiv preprint arXiv:2503.09573, 2025. 2

[3] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with difusion models. arXiv preprint arXiv:2405.04233, 2024. 2

[4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. 2024. URL https://openai. com/research/video-generation-models-as-world-simulators, 3, 2024. 2

[5] Shengqu Cai, Weili Nie, Chao Liu, Julius Berner, Lvmin Zhang, Nanye Ma, Hansheng Chen, Maneesh Agrawala, Leonidas Guibas, Gordon Wetzstein, et al. Mode seeking meets mean seeking for fast long video generation. arXiv preprint arXiv:2602.24289, 2026. 18

[6] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Difusion forcing: Next-token prediction meets full-sequence difusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 2

[7] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025. 2

[8] Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, and Sai Bi. pi-flow: Policy-based few-step generation via imitation distillation. arXiv preprint arXiv:2510.14974, 2025. 18

[9] Yukang Chen, Luozhou Wang, Wei Huang, Shuai Yang, Bohan Zhang, Yicheng Xiao, Ruihang Chu, Weian Mao, Qixin Hu, Shaoteng Liu, Yuyang Zhao, Huizi Mao, Ying-Cong Chen, Enze Xie, Xiaojuan Qi, and Song Han. Longlive2.0: An nvfp4 parallel infrastructure for long video generation. arXiv preprint arXiv, 2026. 2

[10] Zhenglin Cheng, Peng Sun, Jianguo Li, and Tao Lin. Twinflow: Realizing one-step generation on large models with self-adversarial flows. arXiv preprint arXiv:2512.05150, 2025. 18

[11] Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2(3):4, 2024. 8

[12] Jiaqi Feng, Justin Cui, Yuanhao Ban, and Cho-Jui Hsieh. One-forcing: Towards stable one-step autoregressive video generation. arXiv preprint arXiv:2605.23458, 2026. 19

[13] Yao Feng, Chendong Xiang, Xinyi Mao, Hengkai Tan, Zuyue Zhang, Shuhe Huang, Kaiwen Zheng, Haitian Liu, Hang Su, and Jun Zhu. Vidarc: Embodied video difusion model for closed-loop control. arXiv preprint arXiv:2512.17661, 2025. 2

[14] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025. 2

[15] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 3, 5, 17

[16] Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, and Mike Zheng Shou. Anyflow: Any-step video difusion model with on-policy flow map distillation. arXiv preprint arXiv:2605.13724, 2026. 14, 19

[17] Hao-AI Lab. FastVideo: A unified inference and post-training framework for accelerated video generation, 2026. URL https://github.com/hao-ai-lab/FastVideo. 11

[18] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Size Wu, Wei Li, Xuchen Song, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025. 2, 3

[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4

[20] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geofroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025. 2, 3, 12

[21] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple difusion: End-to-end difusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 15

[22] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 6, 7, 9, 11, 14, 18

[23] Yubo Huang, Hailong Guo, Fangtai Wu, Weiqiang Wang, Shifeng Zhang, Shijie Huang, Qijun Gan, Lin Liu, Sirui Zhao, Enhong Chen, Jiaming Liu, and Steven Hoi. Live avatar: Streaming real-time audio-driven avatar generation with infinite length. arXiv preprint arXiv:2512.04677, 2025. 2, 3, 10

[24] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 14

[25] Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025. 2

[26] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023. 11

[27] Dengyang Jiang, Dongyang Liu, Zanyi Wang, Qilong Wu, Liuzhuozheng Li, Hengzhuang Li, Xin Jin, David Liu, Changsheng Lu, Zhen Li, et al. Distribution matching distillation meets reinforcement learning. arXiv preprint arXiv:2511.13649, 2025. 18

[28] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for eficient video generative modeling. In International Conference on Learning Representations, volume 2025, pages 23378–23402, 2025. 2

[29] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of difusion. arXiv preprint arXiv:2310.02279, 2023. 5

[30] Youngrae Kim, Qixin Hu, C-C Jay Kuo, and Peter A Beerel. Memrope: Training-free infinite video generation via evolving memory tokens. arXiv preprint arXiv:2603.12513, 2026. 12

[31] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2

[32] Haobo Li, Yanhong Zeng, Yunhong Lu, Jiapeng Zhu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Yujun Shen, and Zhipeng Zhang. Aad-1: Asymmetric adversarial distillation for one-step autoregressive video generation. arXiv preprint arXiv:2606.03972, 2026. 19

[33] Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging limited-horizon training and open-ended testing in autoregressive video difusion. arXiv preprint arXiv:2602.07775, 2026. 12

[34] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026. 2

[35] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Difusion adversarial post-training for one-step video generation. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 37959–37974. PMLR, 2025. 2, 3, 19

[36] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025. 2, 3, 7, 18, 19

[37] Shanchuan Lin, Ceyuan Yang, Zhijie Lin, Hao Chen, and Haoqi Fan. Continuous adversarial flow models. arXiv preprint arXiv:2604.11521, 2026. 18

[38] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4

[39] Jinxiu Liu, Xuanming Liu, Kangfu Mei, Yandong Wen, Ming-Hsuan Yang, and Weiyang Liu. Streaming autoregressive video generation via diagonal distillation. In ICLR, 2026. 10

[40] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4

[41] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 3, 5, 17, 29

[42] Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Maximum likelihood training for score-based difusion odes by high order denoising score matching. In International conference on machine learning, pages 14429–14460. PMLR, 2022. 17

[43] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021. 8

[44] Weili Nie, Julius Berner, Chao Liu, and Arash Vahdat. Nvidia fastgen: Fast generation from difusion models, 2026. URL https://github.com/NVlabs/FastGen. 11

[45] Weili Nie, Julius Berner, Nanye Ma, Chao Liu, Saining Xie, and Arash Vahdat. Transition matching distillation for fast video generation. arXiv preprint arXiv:2601.09881, 2026. 18, 19

[46] Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, and Itir Onal Ertugrul. Elucidating the exposure bias in difusion models. In International Conference on Learning Representations, volume 2024, pages 15167–15189, 2024. 2

[47] NVIDIA. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026. URL https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf. 2, 16

[48] Dogyun Park, Yanyu Li, Sergey Tulyakov, and Anil Kag. Eflow: Fast few-step video generator training from scratch via eficient solution flow. arXiv preprint arXiv:2603.27086, 2026. 19

[49] Yansong Peng, Kai Zhu, Yu Liu, Pingyu Wu, Hebei Li, Xiaoyan Sun, and Feng Wu. Facm: Flow-anchored consistency models. arXiv preprint arXiv:2507.03738, 2025. 17

[50] Yifan Pu, Yizeng Han, Zhiwei Tang, Jiasheng Tang, Fan Wang, Bohan Zhuang, and Gao Huang. Few-step distillation for text-to-image generation: A practical guide. arXiv preprint arXiv:2512.13006, 2025. 18

[51] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026. 2, 3

[52] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. arXiv preprint arXiv:2506.14603, 2025. 17

[53] Subham Sekhar Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. arXiv preprint arXiv:2406.07524, 2024. 2

[54] Florian Schmidt. Generalization in generation: A closer look at exposure bias. In Proceedings of the 3rd Workshop on Neural Generation and Translation, pages 157–167, 2019. 2

[55] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026. 2

[56] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked difusion for discrete data. arXiv preprint arXiv:2406.04329, 2024. 2

[57] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. arXiv preprin arXiv:2011.13456, 2020. 4

[58] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252. PMLR, 2023. 5

[59] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 2

[60] Shangyuan Tong, Nanye Ma, Saining Xie, and Tommi Jaakkola. Flow map distillation without data. arXiv preprint arXiv:2511.19428, 2025. 18

[61] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 13

[62] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36:8406–8441, 2023. 6

[63] Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. Diversity-preserved distribution matching distillation for fast visual synthesis. arXiv preprint arXiv:2602.03139, 2026. 18

[64] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. In ICLR, 2026. 2, 14

[65] Haotian Ye, Kaiwen Zheng, Jiashu Xu, Puheng Li, Huayu Chen, Jiaqi Han, Sheng Liu, Qinsheng Zhang, Hanzi Mao, Zekun Hao, et al. Data-regularized reinforcement learning for difusion models at scale. arXiv preprint arXiv:2512.04332, 2025. 4

[66] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. 2

[67] Hidir Yesiltepe, Tuna Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 40256–40265, 2026. 12

[68] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081.202512

[69] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024. 2, 6, 9

[70] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 2, 6

[71] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video difusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025. 3

[72] Yuyang You, Yongzhi Li, Jiahui Li, Yadong Mu, Quan Chen, and Peng Jiang. Adaptive video distillation: Mitigating oversaturation and temporal collapse in few-step generation. arXiv preprint arXiv:2603.21864, 2026. 19

[73] Tao Zewei and Huang Yunpeng. Magiattention: A distributed attention towards linear scalability for ultra-long context, heterogeneous mask training. https://github.com/SandAI-org/MagiAttention/, 2025. 8, 30

[74] Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, et al. Sla: Beyond sparsity in difusion transformers via fine-tunable sparselinear attention. arXiv preprint arXiv:2509.24006, 2025. 19

[75] Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodifusion: Accelerating video difusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025. 19

[76] Jintao Zhang, Haoxu Wang, Kai Jiang, Kaiwen Zheng, Youhe Jiang, Ion Stoica, Jianfei Chen, Jun Zhu, and Joseph E Gonzalez. Sla2: Sparse-linear attention with learnable routing and qat. arXiv preprint arXiv:2602.12675, 2026. 19

[77] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive difusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026. 18, 19

[78] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023. 11

[79] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpm-solver-v3: Improved difusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36:55502–55542, 2023. 17

[80] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Improved techniques for maximum likelihood estimation for difusion odes. In International Conference on Machine Learning, pages 42363–42389. PMLR, 2023. 4, 9, 17, 26

[81] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Difusionnft: Online difusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025. 4

[82] Kaiwen Zheng, Yongxin Chen, Huayu Chen, Guande He, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Direct discriminative optimization: Your likelihood-based visual generative model is secretly a gan discriminator. arXiv preprint arXiv:2503.01103, 2025. 4

[83] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked difusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. In International Conference on Learning Representations, volume 2025, pages 63186–63227, 2025. 2

[84] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale difusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025. 3, 4, 9, 11, 13, 18, 19, 26, 29

[85] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained difusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024. 6

[86] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive difusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026. 3, 14

[87] Kai Zou, Dian Zheng, Hongbo Liu, Tiankai Hang, Bin Liu, and Nenghai Yu. Hiar: Eficient autoregressive long video generation via hierarchical denoising. arXiv preprint arXiv:2603.08703, 2026. 18

## A. Theoretical Analysis of TrigFlow-sCM and RF-sCM

This section compares two implementations of continuous-time consistency distillation for an RF-native velocity predictor: (i) applying a TrigFlow wrapper to the RF velocity predictor and then using the TrigFlow-sCM objective (Zheng et al., 2025), and (ii) directly writing the sCM objective under the RF schedule. Despite that diferent difusion noise schedules $( \mathrm { e . g . }$ , TrigFlow and RF) are equivalent and mutually convertible (Zheng et al., 2023) up to a scaling factor, we show that they result in generally diferent normalized MSE training objectives for sCM. The diference comes from the input-output scaling of the TrigFlow wrapper, the tangent normalization, and the finite-precision evaluation order of JVPs.

## RF and TrigFlow coordinates.

Let $u \in [ 0 , 1 ]$ denote the RF time and let $\tau \in [ 0 , \pi / 2 ]$ denote the TrigFlow time. Define

$$
C = \cos \tau , \qquad S = \sin \tau , \qquad Z = C + S, \qquad b = Z ^ {- 1}, \qquad u = \frac {S}{Z}.
$$

The RF and TrigFlow forward processes are

$$
\boldsymbol {x} _ {u} = (1 - u) \boldsymbol {x} _ {0} + u \boldsymbol {\epsilon}, \quad \boldsymbol {x} _ {\tau} = C \boldsymbol {x} _ {0} + S \boldsymbol {\epsilon}.
$$

They are related by a time-dependent state scaling:

$$
\pmb {x} _ {\tau} = Z \pmb {x} _ {u}, \qquad \pmb {x} _ {u} = b \pmb {x} _ {\tau}.\tag{21}
$$

Let ${ \pmb v } _ { \theta } ( { \pmb x } _ { u } , u )$ be an RF velocity predictor and let $\mathbf { V } ( \pmb { x } _ { u } , u ) = \pmb { v } _ { \mathrm { t e a c h e r } } ( \pmb { x } _ { u } , u )$ denote the RF teacher velocity. The direct RF consistency map is

$$
\boldsymbol {f} _ {\theta} ^ {\mathrm{RF}} (\boldsymbol {x} _ {u}, u) = \boldsymbol {x} _ {u} - u \boldsymbol {v} _ {\theta} (\boldsymbol {x} _ {u}, u).\tag{22}
$$

TrigFlow wrapper as input-output transforms.

The TrigFlow wrapper around the RF velocity predictor can be written as

$$
\pmb {F} _ {\theta} ^ {\mathrm{trig}} (\pmb {x} _ {\tau}, \tau) = (C - S) \pmb {x} _ {u} + b \pmb {v} _ {\theta} (\pmb {x} _ {u}, u), \qquad \pmb {x} _ {u} = b \pmb {x} _ {\tau}, \quad u = \frac {S}{Z}.\tag{23}
$$

Equivalently, the wrapper first applies the input transform $( { \pmb x } _ { \tau } , \tau ) \mapsto ( { \pmb x } _ { u } , u )$ , evaluates the RF velocity predictor, and then applies the output transform

$$
\boldsymbol {v} _ {\theta} (\boldsymbol {x} _ {u}, u) \mapsto (C - S) \boldsymbol {x} _ {u} + b \boldsymbol {v} _ {\theta} (\boldsymbol {x} _ {u}, u).
$$

Under the TrigFlow preconditioning

$$
\pmb {f} _ {\theta} ^ {\mathrm{trig}} (\pmb {x} _ {\tau}, \tau) = C \pmb {x} _ {\tau} - S \pmb {F} _ {\theta} ^ {\mathrm{trig}} (\pmb {x} _ {\tau}, \tau),\tag{24}
$$

substituting Eqn. 23 gives

$$
\begin{array}{r l} & {\pmb {f} _ {\theta} ^ {\mathrm{trig}} (\pmb {x} _ {\tau}, \tau) = C Z \pmb {x} _ {u} - S [ (C - S) \pmb {x} _ {u} + b \pmb {v} _ {\theta} (\pmb {x} _ {u}, u) ]} \\ & {\qquad = (C ^ {2} + C S - C S + S ^ {2}) \pmb {x} _ {u} - \frac {S}{Z} \pmb {v} _ {\theta} (\pmb {x} _ {u}, u)} \\ & {\qquad = \pmb {x} _ {u} - u \pmb {v} _ {\theta} (\pmb {x} _ {u}, u) = \pmb {f} _ {\theta} ^ {\mathrm{RF}} (\pmb {x} _ {u}, u).} \end{array}\tag{25}
$$

Therefore, the TrigFlow wrapper and the direct RF parameterization define the same consistency map after the change of variables in Eqn. 21.

Direct RF-sCM tangent.

The RF teacher ODE is

$$
\frac {\mathrm{d} \pmb {x} _ {u}}{\mathrm{d} u} = \mathbf {V} (\pmb {x} _ {u}, u).
$$

For the stop-gradient network $\theta ^ { - }$ , define the RF JVP

$$
\mathbf {J} _ {\theta^ {-}} ^ {\mathrm{RF}} = \operatorname{JVP} \left(\boldsymbol {v} _ {\theta^ {-}}; (\boldsymbol {x} _ {u}, u), (\mathbf {V}, 1)\right) = \nabla_ {\boldsymbol {x} _ {u}} \boldsymbol {v} _ {\theta^ {-}} \mathbf {V} + \partial_ {u} \boldsymbol {v} _ {\theta^ {-}}.\tag{26}
$$

The tangent of the RF consistency map is

$$
\begin{array}{c} \mathbf {h} _ {\mathrm{RF}} = \frac {\mathrm{d}}{\mathrm{d} u} \left[ \boldsymbol {x} _ {u} - u \boldsymbol {v} _ {\theta^ {-}} (\boldsymbol {x} _ {u}, u) \right] \\ = \mathbf {V} - \boldsymbol {v} _ {\theta^ {-}} - u \mathbf {J} _ {\theta^ {-}} ^ {\mathrm{RF}}. \end{array}\tag{27}
$$

The direct RF-sCM objective can thus be written as

$$
\mathcal {L} _ {\mathrm{RF-sCM}} = \mathbb {E} \left[ \left\| \Delta \boldsymbol {v} - \frac {w _ {\mathrm{RF}} (u) \mathbf {h} _ {\mathrm{RF}}}{w _ {\mathrm{RF}} ^ {2} (u) \| \mathbf {h} _ {\mathrm{RF}} \| _ {2} ^ {2} + c} \right\| _ {2} ^ {2} \right], \qquad \Delta \boldsymbol {v} = \boldsymbol {v} _ {\theta} - \boldsymbol {v} _ {\theta^ {-}}.\tag{28}
$$

$\Delta v$ is zero in the forward value, but it still indicates the output coordinate with respect to which gradients are taken.

TrigFlow JVP through the input and output transforms.

We next rewrite the TrigFlow-sCM JVP in the RF coordinates. Along the TrigFlow teacher trajectory,

$$
\frac {\mathrm{d} u}{\mathrm{d} \tau} = b ^ {2}.\tag{29}
$$

Using $\pmb { x } _ { u } = b \pmb { x } _ { \tau }$ and $\begin{array} { r } { \frac { \mathrm { d } \pmb { x } _ { \tau } } { \mathrm { d } \tau } = \pmb { F } _ { \mathrm { t e a c h e r } } ^ { \mathrm { t r i g } } ( \pmb { x } _ { \tau } , \tau ) } \end{array}$ , where

$$
\pmb {F} _ {\mathrm{teacher}} ^ {\mathrm{trig}} (\pmb {x} _ {\tau}, \tau) = (C - S) \pmb {x} _ {u} + b \mathbf {V},
$$

we obtain

$$
\begin{array}{c} \frac {\mathrm{d} \boldsymbol {x} _ {u}}{\mathrm{d} \tau} = \frac {\mathrm{d}}{\mathrm{d} \tau} (b \boldsymbol {x} _ {\tau}) = \dot {b} \boldsymbol {x} _ {\tau} + b \frac {\mathrm{d} \boldsymbol {x} _ {\tau}}{\mathrm{d} \tau} \\ = \dot {b} Z \boldsymbol {x} _ {u} + b [ (C - S) \boldsymbol {x} _ {u} + b \mathbf {V} ]. \end{array}\tag{30}
$$

Since

$$
\dot {b} = \frac {\mathrm{d}}{\mathrm{d} \tau} \frac {1}{C + S} = - \frac {C - S}{Z ^ {2}} = - (C - S) b ^ {2},
$$

the explicit state terms cancel and

$$
\frac {\mathrm{d} \pmb {x} _ {u}}{\mathrm{d} \tau} = b ^ {2} \mathbf {V}.\tag{31}
$$

Thus the JVP direction entering the RF velocity predictor inside the TrigFlow wrapper is

$$
\left(\frac {\mathrm{d} \boldsymbol {x} _ {u}}{\mathrm{d} \tau}, \frac {\mathrm{d} u}{\mathrm{d} \tau}\right) = b ^ {2} (\mathbf {V}, 1).\tag{32}
$$

In exact arithmetic,

$$
\operatorname{JVP} \left(\boldsymbol {v} _ {\theta^ {-}}; (\boldsymbol {x} _ {u}, u), b ^ {2} (\mathbf {V}, 1)\right) = b ^ {2} \mathbf {J} _ {\theta^ {-}} ^ {\mathrm{RF}}.\tag{33}
$$

The output transform in Eqn. 23 contains explicit ??-dependent coeficients. Therefore the JVP of the wrapped

velocity is

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} \tau} \boldsymbol {F} _ {\theta^ {-}} ^ {\text {trig}} = \frac {\mathrm{d}}{\mathrm{d} \tau} \left[ (C - S) \boldsymbol {x} _ {u} + b \boldsymbol {v} _ {\theta^ {-}} (\boldsymbol {x} _ {u}, u) \right] \\ \qquad = - Z \boldsymbol {x} _ {u} + (C - S) \frac {\mathrm{d} \boldsymbol {x} _ {u}}{\mathrm{d} \tau} + \dot {b}   \boldsymbol {v} _ {\theta^ {-}} + b   \text {JVP} \left(\boldsymbol {v} _ {\theta^ {-}}; (\boldsymbol {x} _ {u}, u), \left(\frac {\mathrm{d} \boldsymbol {x} _ {u}}{\mathrm{d} \tau}, \frac {\mathrm{d} u}{\mathrm{d} \tau}\right)\right) \\ \qquad = - Z \boldsymbol {x} _ {u} + (C - S) b ^ {2} (\mathbf {V} - \boldsymbol {v} _ {\theta^ {-}}) + b ^ {3} \mathbf {J} _ {\theta^ {-}} ^ {\text {RF}}. \end{array}\tag{34}
$$

Now diferentiate the TrigFlow consistency map in Eqn. 24:

$$
\begin{array}{r l} & {\mathbf {h} _ {\mathrm{trig}} = \frac {\mathrm{d}}{\mathrm{d} \tau} \left[ C \pmb {x} _ {\tau} - S \pmb {F} _ {\theta^ {-}} ^ {\mathrm{trig}} (\pmb {x} _ {\tau}, \tau) \right]} \\ & {\qquad = - S \pmb {x} _ {\tau} + C \frac {\mathrm{d} \pmb {x} _ {\tau}}{\mathrm{d} \tau} - C \pmb {F} _ {\theta^ {-}} ^ {\mathrm{trig}} - S \frac {\mathrm{d}}{\mathrm{d} \tau} \pmb {F} _ {\theta^ {-}} ^ {\mathrm{trig}}.} \end{array}\tag{35}
$$

Substituting ${ \pmb x } _ { \tau } = Z { \pmb x } _ { u }$ L2

$$
\frac {\mathrm{d} \pmb {x} _ {\tau}}{\mathrm{d} \tau} = (C - S) \pmb {x} _ {u} + b \mathbf {V}, \qquad \pmb {F} _ {\theta^ {-}} ^ {\mathrm{trig}} = (C - S) \pmb {x} _ {u} + b \pmb {v} _ {\theta^ {-}},
$$

and Eqn. 34, we get

$$
\begin{array}{r l} & {\mathbf {h} _ {\mathrm{trig}} = - S Z \pmb {x} _ {u} + C b (\mathbf {V} - \pmb {v} _ {\theta^ {-}}) - S \left[ - Z \pmb {x} _ {u} + (C - S) b ^ {2} (\mathbf {V} - \pmb {v} _ {\theta^ {-}}) + b ^ {3} \mathbf {J} _ {\theta^ {-}} ^ {\mathrm{RF}} \right]} \\ & {\qquad = \left[ C b - S (C - S) b ^ {2} \right] (\mathbf {V} - \pmb {v} _ {\theta^ {-}}) - S b ^ {3} \mathbf {J} _ {\theta^ {-}} ^ {\mathrm{RF}}.} \end{array}\tag{36}
$$

Using

$$
C b - S (C - S) b ^ {2} = b ^ {2}, \qquad S b ^ {3} = u b ^ {2},
$$

we obtain the compact relation

$$
\mathbf {h} _ {\mathrm{trig}} = b ^ {2} \left(\mathbf {V} - \pmb {v} _ {\theta^ {-}} - u \mathbf {J} _ {\theta^ {-}} ^ {\mathrm{RF}}\right) = b ^ {2} \mathbf {h} _ {\mathrm{RF}}.\tag{37}
$$

This derivation makes explicit that the TrigFlow wrapper introduces no new RF-JVP structure: after the input and output transforms, the same RF combination

$$
\mathbf {V} - \boldsymbol {v} _ {\theta^ {-}} - u \operatorname{JVP} \left(\boldsymbol {v} _ {\theta^ {-}}; (\boldsymbol {x} _ {u}, u), (\mathbf {V}, 1)\right)
$$

appears. The only exact-arithmetic diference at the tangent level is the factor $b ^ { 2 } = Z ^ { - 2 }$

TrigFlow-sCM objective in RF velocity coordinates.

The TrigFlow-sCM objective is applied in the TrigFlow velocity coordinate. From Eqn. 23,

$$
\Delta \boldsymbol {F} ^ {\mathrm{trig}} = \boldsymbol {F} _ {\theta} ^ {\mathrm{trig}} - \boldsymbol {F} _ {\theta^ {-}} ^ {\mathrm{trig}} = b (\boldsymbol {v} _ {\theta} - \boldsymbol {v} _ {\theta^ {-}}) = b \Delta \boldsymbol {v}.\tag{38}
$$

Therefore, with

$$
\mathbf {g} _ {\mathrm{trig}} = w _ {\mathrm{trig}} (\tau) \mathbf {h} _ {\mathrm{trig}} = w _ {\mathrm{trig}} (\tau) b ^ {2} \mathbf {h} _ {\mathrm{RF}},
$$

where $w _ { \mathrm { t r i g } } ( \tau )$ is taken as cos $\tau = C$ in sCM (Lu and Song, 2024) and rCM (Zheng et al., 2025), the TrigFlowsCM loss becomes

$$
\begin{array}{r l} & {\mathcal {L} _ {\mathrm{TrigFlow-sCM}} = \mathbb {E} \left[ \left\| \Delta \pmb {F} ^ {\mathrm{trig}} - \frac {\mathbf {g} _ {\mathrm{trig}}}{\| \mathbf {g} _ {\mathrm{trig}} \| _ {2} ^ {2} + c} \right\| _ {2} ^ {2} \right]} \\ & {\qquad = \mathbb {E} \left[ \left\| b \Delta \pmb {v} - \frac {w _ {\mathrm{trig}} (\tau) b ^ {2} \mathbf {h} _ {\mathrm{RF}}}{w _ {\mathrm{trig}} ^ {2} (\tau) b ^ {4} \| \mathbf {h} _ {\mathrm{RF}} \| _ {2} ^ {2} + c} \right\| _ {2} ^ {2} \right]} \\ & {\qquad = \mathbb {E} \left[ \frac {1}{Z ^ {2}} \left\| \Delta \pmb {v} - \frac {w _ {\mathrm{trig}} (\tau) Z ^ {3} \mathbf {h} _ {\mathrm{RF}}}{w _ {\mathrm{trig}} ^ {2} (\tau) \| \mathbf {h} _ {\mathrm{RF}} \| _ {2} ^ {2} + c Z ^ {4}} \right\| _ {2} ^ {2} \right].} \end{array}\tag{39}
$$

By contrast, direct RF-sCM is Eqn. 28. Hence the two objectives share the same zero-consistency condition,

$$
\mathbf {h} _ {\mathrm{trig}} = 0 \quad \Longleftrightarrow \quad \mathbf {h} _ {\mathrm{RF}} = 0,
$$

but they are not, in general, the same normalized MSE objective.

Efect of tangent normalization.

The distinction is easiest to see when $c = 0$ . Eqn. 28 gives the RF normalized tangent target

$$
\mathbf {T} _ {\mathrm{RF}} = \frac {w _ {\mathrm{RF}} \mathbf {h} _ {\mathrm{RF}}}{w _ {\mathrm{RF}} ^ {2} \| \mathbf {h} _ {\mathrm{RF}} \| _ {2} ^ {2}} = \frac {\mathbf {h} _ {\mathrm{RF}}}{w _ {\mathrm{RF}} \| \mathbf {h} _ {\mathrm{RF}} \| _ {2} ^ {2}}.
$$

Eqn. 39 gives the TrigFlow target expressed in the RF output coordinate:

$$
\mathbf {T} _ {\mathrm{trig} \rightarrow \mathrm{RF}} = \frac {w _ {\mathrm{trig}} Z ^ {3} \mathbf {h} _ {\mathrm{RF}}}{w _ {\mathrm{trig}} ^ {2} \| \mathbf {h} _ {\mathrm{RF}} \| _ {2} ^ {2}} = \frac {Z ^ {3}}{w _ {\mathrm{trig}}} \frac {\mathbf {h} _ {\mathrm{RF}}}{\| \mathbf {h} _ {\mathrm{RF}} \| _ {2} ^ {2}}.
$$

Thus, if ?? is kept general,

$$
\mathbf {T} _ {\mathrm{trig} \rightarrow \mathrm{RF}} = \frac {Z ^ {3} w _ {\mathrm{RF}}}{w _ {\mathrm{trig}}} \mathbf {T} _ {\mathrm{RF}}.
$$

The loss gradient with respect to the RF velocity predictor satisfies

$$
\nabla_ {\theta} \mathcal {L} _ {\mathrm{TrigFlow-sCM}} = \frac {Z w _ {\mathrm{RF}}}{w _ {\mathrm{trig}}} \nabla_ {\theta} \mathcal {L} _ {\mathrm{RF-sCM}}, \qquad (c = 0).\tag{40}
$$

When $c > 0 ,$ the diference is not reducible to a simple scalar reweighting, because the TrigFlow denominator becomes $w _ { \mathrm { t r i g } } ^ { 2 } \lVert \mathbf { h } _ { \mathrm { R F } } \rVert _ { 2 } ^ { 2 } + c Z ^ { 4 }$ , whereas the RF denominator is $w _ { \mathrm { R F } } ^ { 2 } \lVert \mathbf { h } _ { \mathrm { R F } } \rVert _ { 2 } ^ { 2 } + c$ . Therefore, tangent normalization breaks the strict equivalence of the two normalized MSE losses.

## Finite-precision JVP evaluation.

The relation in Eqn. 33 is exact only in real arithmetic. In floating-point computation,

$$
\mathrm{fl} \left[ \mathrm{JVP} \left(\boldsymbol {v} _ {\theta^ {-}}; (\boldsymbol {x} _ {u}, u), b ^ {2} (\mathbf {V}, 1)\right) \right] \neq b ^ {2} \mathrm{fl} \left[ \mathrm{JVP} \left(\boldsymbol {v} _ {\theta^ {-}}; (\boldsymbol {x} _ {u}, u), (\mathbf {V}, 1)\right) \right]\tag{41}
$$

in general, and the JVP rearrangement (Lu and Song, 2024) further absorbs the coeficient $w _ { \mathrm { t r i g } } ( \tau ) = \cos \tau$ into JVP computation. Placing the scales inside the JVP direction propagates the scaled tangent through every layer of the network, while factoring them outside first evaluates an unscaled tangent and only then rescales the result. These two evaluation orders can difer because of rounding, mixed-precision casts, fused kernels, activation checkpointing, and custom FlashAttention JVP implementations.

Moreover, the TrigFlow wrapper contains explicit input-output transform terms whose cancellations are algebraically exact but not necessarily bitwise exact. For example, the derivation of Eqn. 37 cancels the statedependent terms from diferentiating $C \pmb { x } _ { \tau } , S \pmb { F } _ { \theta - } ^ { \mathrm { t r i g } }$ , and the wrapper coeficients. A direct RF implementation computes the compact expression

$$
\mathbf {h} _ {\mathrm{RF}} = \mathbf {V} - \boldsymbol {v} _ {\theta^ {-}} - u \mathbf {J} _ {\theta^ {-}} ^ {\mathrm{RF}}
$$

without these intermediate transform terms. Consequently, even when the exact-arithmetic tangent relation $\mathbf { h } _ { \mathrm { t r i g } } = b ^ { 2 } \mathbf { h } _ { \mathrm { R F } }$ holds, the two implementations are not expected to be bitwise equivalent under practical large-scale mixed-precision training. This numerical distinction can be amplified by the normalized target $\mathbf { g } / ( \lVert \mathbf { g } \rVert _ { 2 } ^ { 2 } + c )$ , especially when $\lVert \mathbf { g } \rVert _ { 2 }$ is small or the stabilizing constant ?? is small.

## B. FlashAttention-2 JVP Kernel with Custom Masks

For TF-sCM, the student network is evaluated on a packed sequence that concatenates clean context tokens and noisy target tokens under a TF attention mask. The Jacobian-vector-product (JVP) must be computed through exactly the same masked attention operator as the primal forward pass. A dense additive mask is conceptually simple but memory-ineficient for long video sequences. We therefore represent the custom mask as a sparse list of admissible query-key rectangles in the MagiAttention (Zewei and Yunpeng, 2025) style, and stream only those rectangles inside the FlashAttention-2 loop.

Let $\pmb { M } \in \{ 0 , - \infty \} ^ { N _ { q } \times N _ { k } }$ be a custom attention mask, where ${ \cal M } _ { a b } = 0$ means that query token ?? may attend to key token ??, and $M _ { a b } = - \infty$ otherwise. The masked attention output is

$$
\mathbf {O} = \mathrm{softmax} \bigg (\frac {\mathbf {Q K} ^ {\top}}{\sqrt {d}} + M \bigg) \mathbf {V}.\tag{42}
$$

For JVP, given tangents (tQ, tK, tV), the score tangent is

$$
\mathbf {t S} = \frac {\mathbf {t Q K} ^ {\top} + \mathbf {Q t K} ^ {\top}}{\sqrt {d}}.\tag{43}
$$

The mask is a discrete routing object and has no tangent. Therefore, masked-out entries are assigned zero tangent contribution:

$$
\mathbf {S} _ {a b} = - \infty , \quad \mathbf {t S} _ {a b} = 0 \qquad \mathrm{if} M _ {a b} = - \infty .
$$

Equivalently, the JVP is taken through the masked attention map

$$
(\mathbf {Q}, \mathbf {K}, \mathbf {V}) \mapsto \operatorname{softmax} (\mathbf {S} + M) \mathbf {V},
$$

with ?? fixed.

For a row $^ { a , }$ let $p _ { a b }$ denote the masked softmax probability over allowed keys. The attention tangent is

$$
\mathbf {t O} _ {a} = \sum_ {b} p _ {a b} \mathbf {t V} _ {b} + \sum_ {b} p _ {a b} \left(\mathbf {t S} _ {a b} - \sum_ {c} p _ {a c} \mathbf {t S} _ {a c}\right) \mathbf {V} _ {b},\tag{44}
$$

where all sums are over valid keys under ??. The kernel computes this expression in the same online-softmax pass as the primal FlashAttention computation. For a streamed block, define the unnormalized probability

$$
\tilde {\mathbf {P}} _ {i j} = \exp (\mathbf {S} _ {i j} - m _ {\mathrm{new}}), \qquad \tilde {\mathbf {H}} _ {i j} = \tilde {\mathbf {P}} _ {i j} \odot \mathbf {t} \mathbf {S} _ {i j}.
$$

Besides the standard FlashAttention accumulators $( m , \ell , \mathbf O )$ , we maintain three JVP accumulators:

$$
\mathbf {A} = \sum_ {j} \tilde {\mathbf {P}} _ {i j} \mathbf {t V} _ {j}, \qquad \mathbf {B} = \sum_ {j} \tilde {\mathbf {H}} _ {i j} \mathbf {V} _ {j}, \qquad r = \sum_ {j} \mathrm{rowsum} (\tilde {\mathbf {H}} _ {i j}).
$$

After normalization, the tangent output is

$$
\mathbf {t O} _ {i} = \operatorname{diag} \left(\ell_ {i}\right) ^ {- 1} \left(\mathbf {A} _ {i} + \mathbf {B} _ {i} - \operatorname{diag} \left(r _ {i}\right) \mathbf {O} _ {i}\right),\tag{45}
$$

where $\mathbf { O } _ { i }$ in the last term is the normalized primal output. The same online rescaling factor used for the primal accumulators is applied to $\mathbf { A } _ { i } , \mathbf { B } _ { i } , r _ { i } ,$ so the JVP remains numerically aligned with the FlashAttention-2 softmax normalization.

## Sparse custom-mask representation.

The custom mask is represented as a set of query groups and their admissible key ranges. Each query group ?? contains a contiguous query interval $\mathcal { Q } _ { g } = [ q _ { g } ^ { 0 } , q _ { g } ^ { 1 } )$ and a list of valid key intervals

$$
\mathcal {K} _ {g} = \left\{\left[ k _ {g, r} ^ {0}, k _ {g, r} ^ {1}\right) \right\} _ {r = 1} ^ {R _ {g}}.
$$

The kernel launches tasks $( g , i )$ , where ?? is a query tile inside $\mathcal { Q } _ { g }$ . Each task streams only the key ranges in $\qquad \mathcal { K } _ { g } .$ This range-list view covers both dense/full attention and structured causal masks: dense attention has one query group with one full key range, while teacher-forcing or block-causal masks are decomposed into a small number of full query-key rectangles. Importantly, the same sparse schedule is used for both the primal score S and its tangent tS, ensuring that the tangent corresponds to the exact masked attention operator used in the forward pass.

We present the full algorithm in Algo. 1.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 FlashAttention-2 Forward Pass with JVP Computation and Custom Mask

Require: Matrices Q, K, V, their tangents tQ, tK, tV, block sizes  $B_{r}$ ,  $B_{c}$ , and a custom mask M represented by query groups  $\{Q_{g}\}$ , key ranges  $\{K_{g}\}$ , and task list  $\mathcal{T} = \{(g, i)\}$ .

1: Split Q, tQ into query tiles of size  $B_{r} \times d$  and K, tK, V, tV into key/value tiles of size  $B_{c} \times d$ .

2: Allocate output O, log-sum-exp L, and output tangent tO.

3: for each task  $(g, i) \in \mathcal{T}$  in parallel do

4: Let  $I_{i}$  be the query-token indices of tile i and let  $I_{i}^{g} = I_{i} \cap Q_{g}$ .

5: Load  $Q_{i}$ ,  $tQ_{i}$  from HBM to SRAM, with rows outside  $I_{i}^{g}$  masked out.

6: Initialize  $m_{i} \leftarrow (-\infty)^{B_{r}}$ ,  $\ell_{i} \leftarrow 0^{B_{r}}$ ,  $O_{i} \leftarrow 0^{B_{r} \times d}$ .

7: Initialize JVP accumulators  $r_{i} \leftarrow 0^{B_{r}}$ ,  $A_{i} \leftarrow 0^{B_{r} \times d}$ ,  $B_{i} \leftarrow 0^{B_{r} \times d}$ .

8: for each allowed key range  $[k_{0}, k_{1}) \in K_{g}$  do

9: for each key/value tile j intersecting  $[k_{0}, k_{1})$  do

10: Load  $K_{j}$ ,  $tK_{j}$ ,  $V_{j}$ ,  $tV_{j}$  from HBM to SRAM.

11: Let  $J_{j}$  be the key-token indices of tile j and define the tile-valid mask

 $B_{ij}^{g} = \{(a, b) : a \in I_{i}^{g}, b \in J_{j} \cap [k_{0}, k_{1})\}$ .

12: Compute scores and score tangents  $S_{ij} = Q_{i}K_{j}^{\top}$ ,  $tS_{ij} = tQ_{i}K_{j}^{\top} + Q_{i}tK_{j}^{\top}$ .

13: Apply the custom mask  $S_{ij} \leftarrow where(B_{ij}^{g}, S_{ij}, -\infty)$ ,  $tS_{ij} \leftarrow where(B_{ij}^{g}, tS_{ij}, 0)$ .

14: Compute  $m_{new} = max(m_{i}, rowmax(S_{ij}))$ .

15: Compute  $\tilde{P}_{ij} = exp(S_{ij} - m_{new})$ .

16: Compute  $\ell_{new} = e^{m_{i}-m_{new}} \cdot \ell_{i} + rowsum(\tilde{P}_{ij})$ .

17: Rescale primal and JVP accumulators:

 $O_{i} \leftarrow diag(e^{m_{i}-m_{new}})O_{i}$ ,

 $A_{i} \leftarrow diag(e^{m_{i}-m_{new}})A_{i}$ ,

 $B_{i} \leftarrow diag(e^{m_{i}-m_{new}})B_{i}$ ,

 $r_{i} \leftarrow e^{m_{i}-m_{new}} \cdot r_{i}$ .

18: Update primal accumulator  $O_{i} \leftarrow O_{i} + \tilde{P}_{ij}V_{j}$ .

19: Update value-tangent accumulator  $A_{i} \leftarrow A_{i} + \tilde{P}_{ij}tV_{j}$ .

20: Compute  $\tilde{H}_{ij} = \tilde{P}_{ij} \odot tS_{ij}$ .

21: Update score-tangent accumulators:

 $r_{i} \leftarrow r_{i} + rowsum(\tilde{H}_{ij})$ ,

 $B_{i} \leftarrow B_{i} + \tilde{H}_{ij}V_{j}$ .

22: Update  $m_{i} \leftarrow m_{new}, l_{i} \leftarrow l_{new}$ .

23: end for

24: end for

25: Normalize primal output:

 $O_{i} \leftarrow diag(\ell_{i})^{-1}O_{i}, L_{i} \leftarrow m_{i} + log(\ell_{i}).$ 

26: Compute the JVP epilogue:

 $tO_{i} = diag(\ell_{i})^{-1}(A_{i} + B_{i} - diag(r_{i})O_{i}).$ 

27: Write  $O_{i}, L_{i}, tO_{i}$  to HBM for rows in  $I_{i}^{g}$ .

28: end for

29: return O, L, tO.
</div>