# Proprio: Latent Self-Scoring and Inference-Time Refinement for Physically Plausible Video Generation

Mariam Hassan 1, Kaouther Messaoud 2, Wuyang Li 1, Alexandre Alahi 1

1 École Polytechnique Fédérale de Lausanne (EPFL), 2 Télécom Paris, IP Paris Project Page: https://vita-epfl.github.io/Proprio/

![](images/56abad1340be1fe189d67715a1bf5b4f6bdcf1c98e7bddb2e52f6f772bc226a0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image-to-Video"] --> B["Text-to-Video"]
    B --> C["Self-Scoring and Refining"]
    C --> D["Refine: Update initial noise to reduce the score"]
    C --> E["Select: Pick latent with best score"]
    D --> F["Latent Video Generator"]
    E --> G["Latent Video Generator"]
    F --> H["Initial Noise"]
    G --> I["High Score (Worse)"]
    H --> J["Low Score (better)"]
    I --> K["Text-to-Video"]
    J --> L["Text-to-Video"]
```
</details>

Figure 1: Proprio enables a video generator to evaluate and improve its own generation. Left: qualitative refinement comparisons for TurboWan2.2. Right: Proprio computes a latent self-score from the model’s denoising residual and uses it for either sample selection or inference-time refinement.

# Abstract

Modern video generative models produce visually impressive results, yet frequently violate basic physical principles. We propose Proprio, a training-free framework that enables a frozen video generator to assess and improve the physical plausibility of its own outputs. Inspired by proprioception, the biological sense of one’s own movement, Proprio treats the model’s flow residual under controlled latent perturbations as a self-scoring signal. Samples that are better explained by the generator’s learned dynamics induce smaller and more stable residuals. We aggregate this signal across timesteps and perturbations, focus it on motion-relevant regions with a dynamic spatiotemporal mask, and use it for best-of-N search, gradient-based self-refinement, or both. Across text-to-video and image-to-video benchmarks, Proprio consistently improves physical plausibility, outperforming VLM-based scoring, and external world-model baselines in several settings. With TurboWan2.2, Proprio improves Physics-IQ from 32.2 to 37.5 (+16.5%) and VideoPhy2-hard physical commonsense from 45.6 to 55.0 (+20.6%). Human evaluation further shows that raters prefer Proprio-selected or refined videos for physical plausibility in roughly two-thirds of comparisons. These results suggest that frozen video generators contain actionable internal signals for evaluating and improving the physical plausibility of their own outputs.

# 1 Introduction

Diffusion models [14, 25] have achieved remarkable progress, emerging as dominant paradigms for high-fidelity video generation [8, 1, 34]. Despite their impressive visual quality, these models still struggle to maintain physical plausibility, often producing videos that violate basic real-world constraints. Such failures not only limit perceptual realism but also undermine the utility of these models as reliable world models for downstream applications such as robotics, planning, and simulation.

To address this, existing efforts typically rely on external sources of feedback to provide physical assessment and supervision. Some work inject physical structure through curated datasets, physical priors, or differentiable simulation [35, 24]. Another line of work guides generation using external evaluators, including vision-language models, or pretrained latent world models such as V-JEPA [37, 9, 43, 41, 3, 7]. These methods provide useful signals for selecting physically plausible generations and have shown promising results in improving physical realism.

While effective, this external-feedback paradigm faces two key limitations. (1) External evaluators introduce their own biases and failures: VLMs struggle to assess physical plausibility because their judgments are often entangled with text-image alignment objectives and visual appearance [29]. Pretrained world models provide a more dynamics-oriented signal, but operate on decoded videos and map them into their own representation space, creating a representation mismatch with the generator’s latent space. (2) Misalignment with the generator’s internal dynamics: external evaluators are trained with different training objectives, so their feedback is not directly aligned with the generator’s sampling process and may miss generator-specific inconsistencies across denoising steps.

These limitations motivate an intrinsic perspective on physical plausibility: Can a video generator leverage its own denoising dynamics as a self-contained signal for producing more physically plausible videos? Here, denoising dynamics refers to the model-predicted update direction at each noise level. We focus on its prediction error, which we call the flow residual: the discrepancy between the predicted denoising or velocity direction and the target induced by a controlled perturbation.

Conceptually, the flow residual can be interpreted as an approximate likelihood signal [20], providing inherent cues for assessing physical validity, as previously explored in curated simulated video pairs [40]. In contrast, we reinterpret this likelihood-preference signal as a generator-native objective. Since video generators are trained on large-scale real video datasets, their learned distributions implicitly capture regularities of real-world dynamics. It is therefore natural to treat consistency with such learned dynamics as an intrinsic proxy for physical plausibility. Empirically, we test this intuition with a small-scale diagnostic check. Given a set of real videos, we pair each video with a generated counterpart that exhibits a physical failure using a video model, and compare their residual scores under that model (see Appendix A.1). In 82.2% of the pairs, the residual signal prefers the ground-truth video over the model’s own physically implausible generation, suggesting the model’s internal denoising dynamics can distinguish natural video dynamics from its own failure cases.

Motivated by this, we propose Proprio, a generator-native, training-free framework that converts internal denoising signals into an internal proxy for physical plausibility. Inspired by proprioception, the biological ability to sense one’s motion without external observation, Proprio enables a frozen video generator to assess and improve its own outputs. Specifically, we define a noise-conditioned selfconsistency signal based on the model’s denoising residual under controlled latent perturbations. We robustly estimate the score by aggregating residuals across multiple timesteps with inverse-variance weighting, which downweights noisy estimates, and by applying a dynamic spatiotemporal mask to emphasize motion-relevant regions where physical failures occur. Beyond scoring, Proprio uses the score as a self-refinement objective. It refines generations by backpropagating through the frozen sampler and updating the initial noise to produce a more internally consistent video.

We evaluate Proprio in three inference-time strategies: best-of-N search, self-refinement, and hybrid search-and-refine approach (see Fig. 1). For image-to-video (I2V) and text-to-video (T2V) generation, Proprio improves physical accuracy over VLM-based, and outperforms external pretrained worldmodel in several settings. These results suggest that a frozen video generator can be used to generate videos and to guide selection and refinement of its outputs. Our contributions are as follows:

• Generator-native self-scoring: We introduce Proprio, a training-free framework that reuses a frozen generator’s denoising residual as an intrinsic score for physical plausibility.

• Inference-time self-alignment: We turn this score into an optimization objective for a frozen generator, refining its outputs by updating only the initial noise, and optionally combine this refinement with best-of-N selection.

• Empirical validation: We demonstrate consistent gains on I2V and T2V physics benchmarks, showing that a single frozen generator can act as both sampler and self-evaluator, guiding selection and refinement toward more physically plausible dynamics.

![](images/e751823f7ff4d3b80ba808270df95b39487345c6871b196fed25835cf7330e8e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Latent z₀"] --> B["Conditions c"]
    C["Variance Weighting wₜ"] --> D["Score S(x)"]
    E["Dynamic Mask M"] --> D
    F["Multi-step Perturbation ε₀ε₁...εₙ t=0.3T t=0.4T ... t=0.6T zₜ=z₀+εσₜ"] --> G["Latent Video Generator"]
    H["Lermotion Lₘₒₜᵢₒₜ"] --> D
    I["Lₑglobal Lₗₒₐₗ"] --> G
    J["Shared Model"] --> G
    K["Latent Update Latent Video Generator"] --> L["Refine (μ,σ) to minimize S and L_KL"]
    M["Vae Decoder"] --> L
    N["N Candidate Latents Rank by S(x)"] --> O["Initial noise parameters ζ₀~N(0,I) ζₖ=μ+σ·ζ₀ μ,σ"]
    P["Score and Regularisation ℒ=S(ζₖ)+βL_KL"] --> Q["Adam update for μ,σ"]
    R["Physically Plausible Video"] --> S["Output"]
```
</details>

Figure 2: Proprio method overview. For a generated latent video, Proprio computes a per-video self-scoring signal by perturbing the latent at multiple timesteps, measuring denoising residuals with the frozen generator, weighting timesteps by inverse variance, and emphasizing informative motion regions with a dynamic spatiotemporal mask. The resulting score can either be used for ranking and selection, self-refining or both. The VAE encoder is omitted for brevity.

# 2 Related Work

Physical plausibility in video generation. Recent video generative models [8, 6, 19, 34, 36, 38, 12, 31] have achieved impressive visual fidelity, yet they still frequently violate basic physical principles [18, 29]. Previous work explored three lines of work to solve problem. The first improves physical fidelity during training by introducing physics-aware data, supervision, or learning objectives [35, 9, 35]. The second incorporates explicit physical structure or simulation into the generation process [24]. The final approach addresses the problem at inference time through external guidance, for example using VLM-based planning or reward signals from pretrained world models. VLIPP and GPT4motion use a VLM as physics reward [37, 26], while WMReward uses VJEPA-2 as an external latent world model to provide rewards for search and guided sampling [41].

Inference-time alignment for image generation. Inference-time alignment for diffusion models can be categorized into search-based optimization-based methods. Search-based approaches improve alignment by exploring multiple denoising trajectories and selecting or resampling candidates according to a reward, including Best-of-N selection, SMC-style resampling, and value-based decoding [21, 22, 27, 44, 16]. Optimization-based approaches directly modify the sampling trajectory or noise variables to increase the target reward, for example through reward guidance or direct noise optimization [33, 11, 39, 2, 10]. Collectively, these works show that substantial gains can be obtained at test time without retraining the generator. However, they typically rely on externally defined reward models or prompt-alignment objectives.

Latent reward modeling. Recent work has explored training video generation models as latent reward models, motivated by the limitations of VLM-based evaluators. In particular, learning rewards models by directly training on noisy latent states has been shown to better match latentspace generation and avoid the overhead of pixel-space VLM rewards [23, 28, 30]. Alternatively, LikePhys [40] curates a synthetic benchmark and evaluates models using preference Plausibility Preference Error (PPE) metric on curated valid-invalid video pairs. Unlike these approaches, which either train dedicated latent reward models or depend on curated simulated pairs, our method directly reuses the frozen generator’s own denoising residual as a self-score for physical plausibility.

# 3 Method

We introduce Proprio, a training-free inference-time framework with two complementary stages: self-scoring and self-refinement (see Fig. 2). Self-scoring asks whether a generated latent video is consistent with the frozen generator’s learned dynamics: samples that the model can explain under controlled perturbations should induce smaller and more stable flow residuals. Since large video generators are trained on natural video dynamics, this learned-flow consistency can serve as an intrinsic proxy for physical plausibility. Self-refinement then uses this score as an objective to improve a sample by updating only a structured parameterization of its initial sampling noise.

The self-scoring signal is built from four components. (1) We perturb each generated latent at multiple scheduler timesteps and noise realizations, probing the sample across different noise regimes. (2) We measure the model’s flow prediction residual, reusing the generator’s own training objective as a self-consistency signal. (3) We optionally apply a dynamic spatiotemporal mask to focus the score on motion-relevant regions. (4) We aggregate residuals across timesteps with inverse-variance weighting, giving more weight to reliable estimates and downweighting perturbation-sensitive ones. The resulting score supports three inference-time modes: best-of-N search, self-refinement, and hybrid search-and-refine. We briefly review the flow-matching formulation in Appendix F.

# 3.1 Flow Prediction under Scheduler Perturbations

Let c denote the conditioning input (e.g., text prompt or input image). Let $z _ { 0 } \in \mathbb { R } ^ { T \times H \times W \times C }$ denote the latent representation of a generated video $x _ { 0 }$ . We consider a pretrained video generator with parameters θ, which predicts a velocity (flow) field

$$
\hat {v} _ {\theta}: \mathbb {R} ^ {T \times H \times W \times C} \times \mathcal {T} \times \mathcal {C} \rightarrow \mathbb {R} ^ {T \times H \times W \times C}, \tag {1}
$$

where T denotes the set of scheduler timesteps.

Modern video generators define perturbations through a scheduler rather than a fixed analytical path. Given a timestep $t \in \mathcal T$ and noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , we construct a perturbed latent

$$
z _ {t} = \mathcal {P} (z _ {0}, \epsilon , t), \tag {2}
$$

where P denotes the forward perturbation process.

The model predicts a flow field $\hat { v } _ { \theta } ( z _ { t } , t , c )$ , which is trained to match a target transport of the form

$$
v ^ {\star} (z _ {0}, \epsilon) \approx \epsilon - z _ {0}. \tag {3}
$$

This corresponds to the standard velocity parameterization and enforces local consistency of the learned transport under perturbations of real data.

# 3.2 Intrinsic Self-Scoring via Flow Residuals

Given a latent sample $z _ { \mathrm { 0 } }$ , we probe its consistency by sampling ϵ and evaluating the model at perturbed states $z _ { t }$ . We define the flow residual

$$
r (z _ {0}, \epsilon , t; c) = \hat {v} _ {\theta} (z _ {t}, t, c) - v ^ {\star} (z _ {0}, \epsilon), \tag {4}
$$

and the per-timestep score

$$
\ell_ {t} (z _ {0}, \epsilon ; c) = \| r (z _ {0}, \epsilon , t; c) \| _ {2} ^ {2}. \tag {5}
$$

This quantity corresponds to the flow-matching regression error evaluated at $z _ { 0 } .$ , providing a local surrogate for alignment with the learned data distribution. Intuitively, it measures how consistently the learned transport explains the sample under perturbations: samples near well-modeled regions yield small residuals, while deviations from the learned distribution produce larger discrepancies.

Dynamic Spatiotemporal Masking. The score $\ell _ { t }$ aggregates errors across the entire latent volume, including large static regions that are less informative for physical plausibility. To focus the signal on informative motion regions, we introduce a dynamic spatiotemporal mask $\dot { M } \in [ 0 , 1 ] ^ { T \times H \times \bigstar }$ .

Let u index spatiotemporal locations. Refer to Appendix D for mask details. The masked score is

$$
\ell_ {t} ^ {\text { motion }} (z _ {0}, \epsilon ; c) = \frac {\sum_ {u} M _ {u} \| r _ {u} (z _ {0} , \epsilon , t ; c) \| _ {2} ^ {2}}{\sum_ {u} M _ {u} + \delta}, \tag {6}
$$

where $\delta > 0$ ensures numerical stability.

This can be interpreted as importance weighting over spatiotemporal regions, emphasizing locations with higher expected relevance to motion and physical interactions, thereby improving sensitivity to physically meaningful deviations.

Multi-Perturbation, Multi-Timestep Estimation. A single evaluation of $\ell _ { t }$ is sensitive to both timestep and noise. We therefore aggregate across multiple perturbations and timesteps.

Let $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { K } \}$ and $\epsilon ^ { ( n ) } \sim \mathcal { N } ( 0 , I )$ for $n = 1 , \ldots , N$ . For each timestep $t _ { k } .$ , we compute

$$
\mu_ {k} = \frac {1}{N} \sum_ {n = 1} ^ {N} \ell_ {t _ {k}} (z _ {0}, \epsilon^ {(n)}; c), \tag {7}
$$

$$
v _ {k} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(\ell_ {t _ {k}} (z _ {0}, \epsilon^ {(n)}; c) - \mu_ {k}\right) ^ {2}. \tag {8}
$$

The mean $\mu _ { k }$ estimates the residual strength at timestep $t _ { k }$ , while the variance $v _ { k }$ measures how stable that estimate is across perturbations. This separates the magnitude of the inconsistency from the reliability of the measurement.

Variance-Weighted Aggregation. Different timesteps can provide signals of different reliability: some noise levels expose meaningful inconsistencies, while others produce residuals that are highly sensitive to the sampled perturbation. We therefore weight timestep estimates by inverse variance:

$$
w _ {k} = \frac {1}{v _ {k} + \varepsilon}, \quad \tilde {w} _ {k} = \frac {w _ {k}}{\sum_ {j = 1} ^ {K} w _ {j}}, \tag {9}
$$

where $\varepsilon > 0$ is a small constant for numerical stability.

We use $v _ { k }$ to measure the stability of the score at timestep $t _ { k } \mathrm { : }$ larger $v _ { k }$ indicates that the score is more sensitive to the perturbation noise. Inverse-variance weighting therefore assigns higher weight to more stable timesteps and lower weight to noisier ones.

The final score is

$$
S (z _ {0}; c) = \sum_ {k = 1} ^ {K} \tilde {w} _ {k} \mu_ {k}. \tag {10}
$$

For the motion-aware variant, we define $S _ { \mathrm { m o t i o n } }$ analogously using $\ell _ { t } ^ { \mathrm { m o t i o n } }$ , and optionally combine both:

$$
S _ {\text { hybrid }} (z _ {0}; c) = \lambda S _ {\text { global }} (z _ {0}; c) + (1 - \lambda) S _ {\text { motion }} (z _ {0}; c), \tag {11}
$$

where $\lambda \in [ 0 , 1 ]$ . The global score preserves broad sample consistency, while the motion-aware score emphasizes dynamic regions; the hybrid score balances these two views.

# 3.3 Inference-Time Selection and Refinement

The Proprio score defines a generator-native objective over samples. It can therefore be used not only for selection, but also for refinement: rather than simply choosing the lowest-scoring candidate, we can directly optimize the sampling noise to reduce the score. We thus propose an inference-time self-refinement procedure that updates the initial noise while keeping the generator fixed.

The Proprio score defines a generator-native objective over samples. Given candidates $\{ z _ { 0 } ^ { ( i ) } \}$ , we select $\hat { i } = \arg \operatorname* { m i n } _ { i } S ( z _ { 0 } ^ { ( i ) } ; c )$ . This provides a simple test-time scaling mechanism: generate multiple candidates and keep the one that is most internally consistent according to the frozen generator.

Beyond selection, the same score can be used for self-refinement. Instead of modifying the generator weights, we optimize a structured parameterization of the initial sampling noise. Let $\zeta _ { 0 } \sim \mathcal { N } ( 0 , I )$ denote the base noise. We introduce per-channel parameters $\boldsymbol { \mu } \in \mathbb { R } ^ { C }$ and $\bar { \sigma } \in \mathbb { R } _ { + } ^ { C }$ , and define

$$
\zeta = \mu + \sigma \odot \zeta_ {0}, \tag {12}
$$

where $\boldsymbol { \mu } \in \mathbb { R } ^ { C }$ and $\sigma \in \mathbb { R } _ { + } ^ { C }$ . We denote by $z _ { 0 } ( \zeta ; c )$ the latent generated from noise $\zeta$ under conditioning c.

We optimize

$$
\mathcal {L} (\mu , \sigma) = S (z _ {0} (\zeta ; c); c) + \beta \mathcal {L} _ {\mathrm{KL}}, \tag {13}
$$

where $\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } } = \frac { 1 } { C } \sum _ { d = 1 } ^ { C } \frac { 1 } { 2 } \left( \mu _ { d } ^ { 2 } + \sigma _ { d } ^ { 2 } - 1 - 2 \log \sigma _ { d } \right) } \end{array}$

Table 1: Inference-time search on VideoPhy2. We report Semantic Adherence (SA), Physical Commonsense (PC), and Joint accuracy on the standard and hard splits. Higher is better. 

<table><tr><td rowspan="3">Method</td><td colspan="6">Wan2.2 5B</td><td colspan="6">Turbo Wan14B</td><td colspan="6">Hunyuan Video</td></tr><tr><td colspan="3">Standard</td><td colspan="3">Hard</td><td colspan="3">Standard</td><td colspan="3">Hard</td><td colspan="3">Standard</td><td colspan="3">Hard</td></tr><tr><td>SA</td><td>PC</td><td>Joint</td><td>SA</td><td>PC</td><td>Joint</td><td>SA</td><td>PC</td><td>Joint</td><td>SA</td><td>PC</td><td>Joint</td><td>SA</td><td>PC</td><td>Joint</td><td>SA</td><td>PC</td><td>Joint</td></tr><tr><td>Random</td><td>65.4</td><td>58.1</td><td>48.6</td><td>31.7</td><td>44.4</td><td>19.4</td><td>67.2</td><td>68.3</td><td>56.1</td><td>32.2</td><td>47.8</td><td>23.3</td><td>61.1</td><td>75.0</td><td>50.0</td><td>29.4</td><td>61.1</td><td>20.0</td></tr><tr><td>Qwen 7B [4]</td><td>67.0</td><td>63.7</td><td>52.0</td><td>35.0</td><td>42.2</td><td>18.9</td><td>65.6</td><td>69.4</td><td>55.0</td><td>32.8</td><td>46.1</td><td>22.2</td><td>61.7</td><td>78.3</td><td>52.8</td><td>31.7</td><td>64.4</td><td>23.9</td></tr><tr><td>WMReward [41]</td><td>62.2</td><td>66.1</td><td>51.1</td><td>33.9</td><td>47.8</td><td>20.6</td><td>63.9</td><td>73.3</td><td>53.9</td><td>30.0</td><td>53.9</td><td>20.0</td><td>57.2</td><td>78.3</td><td>48.3</td><td>27.2</td><td>70.0</td><td>21.7</td></tr><tr><td>Ours -  $S_{global}$ </td><td>67.6</td><td>64.8</td><td>52.5</td><td>30.0</td><td>50.0</td><td>18.9</td><td>65.9</td><td>73.7</td><td>56.4</td><td>33.3</td><td>53.3</td><td>22.2</td><td>56.7</td><td>77.2</td><td>46.1</td><td>25.0</td><td>70.8</td><td>18.8</td></tr><tr><td>Ours -  $S_{motion}$ </td><td>66.5</td><td>68.2</td><td>55.3</td><td>34.4</td><td>51.7</td><td>23.3</td><td>64.8</td><td>74.3</td><td>55.9</td><td>33.9</td><td>57.8</td><td>23.9</td><td>58.3</td><td>79.4</td><td>50.6</td><td>22.9</td><td>70.8</td><td>18.8</td></tr><tr><td>Ours -  $S_{hybrid}$ </td><td>66.5</td><td>69.8</td><td>55.3</td><td>33.3</td><td>52.8</td><td>22.8</td><td>64.2</td><td>74.9</td><td>56.4</td><td>33.3</td><td>58.3</td><td>23.9</td><td>58.9</td><td>78.3</td><td>50.6</td><td>28.9</td><td>71.7</td><td>25.0</td></tr></table>

The Gaussian-affine parameterization gives refinement enough flexibility to improve the sample while keeping the optimized noise close to the generator’s original Gaussian prior. Starting from µ = 0 and σ = 1, we optimize (µ, log σ) with Adam for a small number of steps while keeping the generator fixed. In practice, we return the sample achieving the lowest unregularized Proprio score during optimization. The full algorithm is provided in Appendix C.

# 4 Experiments

We evaluate Proprio in three inference-time settings: search, refinement, and search-and-refine (Sec. 3). Experiments are conducted on both text-to-video (T2V) and image-to-video (I2V) generation, using VideoPhy2 [5] and Physics-IQ [29], respectively, two benchmarks designed to assess physical plausibility and reasoning in generated videos. We include additional experiments in Appendix A.

Benchmarks For T2V, we use VideoPhy2 [5], which reports three metrics: Semantic Adherence (SA), Physical Commonsense (PC), and Joint (SA=1 and PC=1). We follow the default evaluation protocol where scores greater than or equal to 4 are mapped to 1 and all others to 0. Following previous works [17], we randomly sample 360 prompts equally across standard and hard set. For I2V, we use Physics-IQ [29], which directly reports a physics score.

Baselines. For search, we compare against (i) random selection, (ii) a VLM-based scorer using Qwen-7B [4], and (iii) WMReward [41], which ranks samples using VJEPA2 surprise scores. We follow their official implementation, scoring full video frames and selecting the lowest surprisescore sample using Vjepa2-G. For refinement, we compare against WMReward’s guidance-based optimization method using the recommended guidance of 0.005 and update frequency of 5.

Implementation details. Unless otherwise stated, we generate 16 candidate samples per prompt. For fair comparison, all methods score the same candidate set. Experiments are run on 2 H100 GPUs. We use each model’s default frame rate: 24 FPS for Wan2.2 and Hunyuan Video 1.5, and 16 FPS for Turbo Wan2.2. Refer to Appendix E for more details.

# 4.1 Improving Generation via Inference-Time Search

We evaluate inference-time scaling by generating 16 candidate samples and selecting the best one using Proprio. We conduct experiments across three video models, Wan2.2 5B [34], TurboDiffusion Wan2.2 14B [42], and Hunyuan Video 1.5 [36], in both (T2V) and (I2V) settings.

Text-to-Video. As shown in Table 1, Proprio consistently outperforms random selection and the VLM-based scorer on VideoPhy2, while WMReward remains a stronger baseline than Qwen-7B. Across models, Qwen-7B provides limited benefit: although it occasionally improves semantic adherence (SA), these gains do not translate into stronger physical commonsense (PC) or joint performance, consistent with prior observations [41]. Compared to WMReward, Proprio achieves further improvements in most settings, particularly on PC and joint metrics. We observe a clear tradeoff between SA and PC: global scoring tends to better preserve SA, whereas motion-aware and hybrid scoring more strongly improve physical consistency. Among these, the hybrid variant is the most robust, consistently achieving the best balance between SA and PC and the strongest joint performance. While improving physical plausibility can reduce SA, Proprio exhibits a milder tradeoff than prior approaches, with smaller SA degradation relative to the gains in PC and joint scores. Overall, these results indicate that generator-native self-scoring provides a more effective signal for selecting physically plausible T2V samples than generic VLM-based evaluation, and is competitive with or stronger than WMReward.

Image-to-Video. As shown in Table 2, the I2V setting exhibits a different trend. The motion and hybrid variants of Proprio perform similarly, with motion-based scoring slightly stronger sometimes. This is consistent with the fact that I2V candidates are anchored to the same input image and are therefore already semantically aligned, making motion dynamics the primary factors distinguishing samples. While WMReward performs strongly across models, Proprio slightly improves over it on Wan2.2 5B and Turbo and achieves the best overall result on Hunyuan Video with the motionaware score. Overall, these results show that generator-native self-scoring provides a strong and competitive criterion for selecting physically plausible I2V generations.

Table 2: Inference-time search results on Physics-IQ across three models. Higher is better. 

<table><tr><td>Method</td><td>Wan2.2 5B</td><td>Turbo</td><td>Hunyuan Video</td></tr><tr><td>Random</td><td>29.45</td><td>31.25</td><td>31.63</td></tr><tr><td>Qwen 7B [4]</td><td>28.51</td><td>31.32</td><td>32.23</td></tr><tr><td>WMReward [41]</td><td>32.40</td><td>34.67</td><td>34.75</td></tr><tr><td>Ours -  $S_{global}$ </td><td>31.47</td><td>34.48</td><td>33.73</td></tr><tr><td>Ours -  $S_{motion}$ </td><td>32.25</td><td>34.65</td><td>36.09</td></tr><tr><td>Ours -  $S_{hybrid}$ </td><td>32.46</td><td>35.13</td><td>35.13</td></tr></table>

# 4.2 Refining Generations with Inference-Time Optimization

We evaluate inference-time refinement using Proprio as the optimization signal. To make gradientbased optimization tractable, we use a distilled model (TurboDiffusion Wan 14B) and backpropagate through a reduced number of denoising steps (4 instead of 50), significantly lowering computational cost while preserving a useful refinement signal. We conduct experiments in both text-to-video (T2V) and image-to-video (I2V) settings on the VideoPhy2 [5] and Physics-IQ [29] benchmarks.

As shown in Table 3, Proprio consistently improves over the base model and outperforms WMReward, with the largest gains observed when combining search and refinement. Refinement alone already yields meaningful improvements, and initializing from better candidates (via search) and then refining them leads to consistently stronger results, indicating that Proprio benefits from both selection and optimization. On Physics-IQ, the motion variant is the strongest, meaning that motion-focused refinement is very effective for I2V physical plausibility. On VideoPhy2, the hybrid score typically yields the strongest physical consistency on the standard split, while both motion and hybrid variants substantially improve performance on the hard split. Across settings, the most consistent gains are observed on physical commonsense (PC), indicating that refinement primarily improves physical realism. Overall, these results demonstrate that Proprio provides an effective optimization signal for challenging physical reasoning tasks, with motion-aware scoring particularly beneficial for I2V and hybrid scoring offering the most robust improvements for T2V.

# 4.3 Human evaluation

To complement benchmark-based evaluation, we conduct a pairwise human evaluation study to test whether Proprio’s self-score agrees with human judgments. We sample 64 video pairs for search and refinement settings for both T2V and I2V, and collect 330 responses. In the search setting, each pair compares a low-score and highscore Proprio sample for the same condition; in refinement, each pair compares a Proprio-refined video with its unrefined baseline. As shown in Fig. 3, human preferences align well with Proprio on physical plausibility. Raters prefer the lower-score sample in 67.3% of search comparisons and the Proprio-refined sample in 66.7% of refinement comparisons. The effect is weaker for visual quality, where Proprio-selected or refined videos are preferred in roughly half of comparisons and the remaining votes are split between the baseline and no preference. This suggests that Proprio primarily improves perceived physical plausibility while largely preserving visual quality. Additional preference results and visual quality evaluation using VBench [15] are provided in Appendix A.

![](images/6759dad03584b095f19221a8f2931258edf4213de68655322159a2be5755976a.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Ours (%) | Base. (%) | No Preference (%) |
| :--- | :--- | :--- | :--- |
| Search - Physical | 67.3 | 20.1 | 12.6 |
| Search - Visual | 47.8 | 22.0 | 30.2 |
| Refinement - Physical | 66.7 | 21.6 | 11.7 |
| Refinement - Visual | 49.1 | 22.2 | 28.7 |
</details>

Figure 3: Human preference assessing visual quality and physical plausibility.

Table 3: Inference-time refinement on Turbo Wan2.2. We report Physics-IQ and the VideoPhy2 breakdown on the standard and hard subsets. Higher is better. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Physics-IQ</td><td colspan="5">VideoPhy2</td></tr><tr><td>Score</td><td>SA</td><td>Standard PC</td><td>Joint</td><td>SA</td><td>Hard PC</td><td>Joint</td></tr><tr><td>Turbo Wan2.2</td><td>32.2</td><td>65.6</td><td>67.2</td><td>55.0</td><td>31.1</td><td>45.6</td><td>19.4</td></tr><tr><td colspan="8">Refinement only</td></tr><tr><td>+ WMReward ( $\mathcal{G}$ )* [41]</td><td>32.4</td><td>67.3</td><td>67.2</td><td>55.6</td><td>32.8</td><td>45.0</td><td>21.7</td></tr><tr><td>+ Proprio ( $\nabla S_{global}$ )</td><td>34.6</td><td>65.4</td><td>68.2</td><td>54.2</td><td>29.4</td><td>43.9</td><td>20.0</td></tr><tr><td>+ Proprio ( $\nabla S_{motion}$ )</td><td>35.8</td><td>65.4</td><td>69.3</td><td>54.2</td><td>32.2</td><td>47.2</td><td>20.6</td></tr><tr><td>+ Proprio ( $\nabla S_{hybrid}$ )</td><td>34.9</td><td>65.6</td><td>71.7</td><td>54.4</td><td>32.2</td><td>50.6</td><td>21.1</td></tr><tr><td colspan="8">Search (BoN4) + refinement</td></tr><tr><td>+ WMReward ( $\mathcal{G}$ + BoN4)* [41]</td><td>34.7</td><td>64.4</td><td>67.2</td><td>51.1</td><td>32.8</td><td>48.9</td><td>19.4</td></tr><tr><td>+ Proprio ( $\nabla S_{global}$ + BoN4)</td><td>36.7</td><td>64.4</td><td>71.7</td><td>54.4</td><td>29.4</td><td>54.4</td><td>21.1</td></tr><tr><td>+ Proprio ( $\nabla S_{motion}$ + BoN4)</td><td>37.4</td><td>66.1</td><td>71.7</td><td>52.8</td><td>29.7</td><td>54.3</td><td>21.1</td></tr><tr><td>+ Proprio ( $\nabla S_{hybrid}$ + BoN4)</td><td>37.1</td><td>63.9</td><td>73.3</td><td>51.1</td><td>31.1</td><td>55.0</td><td>21.1</td></tr><tr><td colspan="8">Search (BoN8) + refinement</td></tr><tr><td>+ WMReward ( $\mathcal{G}$ + BoN8)* [41]</td><td>35.5</td><td>66.7</td><td>68.9</td><td>52.8</td><td>30.6</td><td>52.2</td><td>21.1</td></tr><tr><td>+ Proprio ( $\nabla S_{motion}$ + BoN8)</td><td>37.5</td><td>65.0</td><td>73.3</td><td>55.6</td><td>31.7</td><td>54.4</td><td>23.3</td></tr><tr><td>+ Proprio ( $\nabla S_{hybrid}$ + BoN8)</td><td>37.3</td><td>66.1</td><td>74.4</td><td>57.8</td><td>29.6</td><td>54.2</td><td>20.7</td></tr></table>

∗ self reproduced. G indicates guidance

# 5 Ablation Studies

Effect of noise levels for scoring. As shown in Fig. 4 and Table 5, the most informative scoring signal arises from an intermediate noise regime, rather than from very low or very high noise levels. On Physics-IQ, both global $S _ { \mathrm { g l o b a l } }$ and motion aware $S _ { \mathrm { m o t i o n } }$ scoring improve substantially when evaluation is restricted to mid-noise timesteps, while performance degrades at the extremes. A similar pattern holds on VideoPhy2: aggregating scores over the full noise range yields the highest semantic adherence (SA), but significantly weaker physical commonsense (PC) and joint performance. In contrast, restricting evaluation to mid-noise timesteps improves PC and produces the strongest joint scores. These results indicate that extreme noise levels yield less reliable or less informative signals, whereas mid-noise perturbations provide the most discriminative signal for physical plausibility.

Effect of variance weighting. Table 5 shows that inverse-variance weighting improves score aggregation by emphasizing timesteps with more stable residual estimates. This effect is most pronounced for the motion score, where variance weighting increases PC from 66.48 to 68.16, yielding the best overall PC result. Gains for the global score are smaller but consistent, indicating that not all timesteps contribute equally reliable information. These results highlight that effective selfscoring depends not only on selecting an appropriate noise regime, but also on weighting timesteps according to their stability. In practice, combining mid-noise with variance-weighted motion scoring provides the strongest and most robust performance. We note that this effect is significantly weaker on distilled models, where variance weighting yields only marginal improvements (see Appendix A.4).

Dynamic masking strategy. We explore different setups for the motion-based masking. Table 4 compares three dynamic masking variants for optimization. The motion mask is the default introduced in Sec. 3. The textaware mask derives motion from cross-attention, making it explicitly prompt-grounded, while the attentionbased mask uses self-attention feature changes as an internal proxy for motion, similar to [32].

Table 4: Dynamic masking strategies for optimization on Turbo Wan2.2. 

<table><tr><td rowspan="2">Masking</td><td rowspan="2">Physics-IQ Score</td><td colspan="3">VideoPhy2 Standard</td></tr><tr><td>SA</td><td>PC</td><td>Joint</td></tr><tr><td>Motion</td><td>35.8</td><td>65.36</td><td>69.27</td><td>54.19</td></tr><tr><td>Attention-based</td><td>34.10</td><td>63.33</td><td>68.33</td><td>52.22</td></tr><tr><td>Textaware</td><td>33.04</td><td>66.66</td><td>65.56</td><td>53.33</td></tr></table>

Additional details are provided in Appendix D. All three variants provide useful dynamic signals, but exhibit distinct tradeoffs. The motion mask achieves the strongest overall performance, with the best Physics-IQ score and the highest PC and joint scores on VideoPhy2, indicating that direct motion cues are the most reliable signal in this setting. The textaware mask yields the highest semantic adherence (SA), suggesting that prompt-grounded masking better preserves semantic alignment, but at some cost to physical consistency. The attention-based variant remains competitive but is slightly weaker. Overall, these results show that Proprio is compatible with multiple forms of dynamic masking, and that the choice of mask controls the tradeoff between semantic adherence and physical consistency.

![](images/e21d01916aff3ff8b661878a7b0e92ef8aafb0696b597aa2dca2c2e6cc08003d.jpg)

<details>
<summary>line</summary>

| Noise Level (ratio of sampling steps) | PhysIQ Global | PhysIQ Mtion | Random Selection |
| ------------------------------------ | ------------- | ------------ | ---------------- |
| 0.1                                  | 28.5          | 28.0         | 29.0             |
| 0.2                                  | 30.5          | 31.0         | 29.0             |
| 0.3                                  | 30.5          | 30.5         | 29.0             |
| 0.4                                  | 31.5          | 32.5         | 29.0             |
| 0.5                                  | 30.5          | 31.5         | 29.0             |
| 0.6                                  | 32.0          | 32.0         | 29.0             |
| 0.7                                  | 31.0          | 31.5         | 29.0             |
| 0.8                                  | 30.0          | 31.0         | 29.0             |
| 0.9                                  | 31.5          | 30.5         | 29.0             |
</details>

Figure 4: Effect of noise levels at different timestep ranges on Physics-IQ for Wan2.2 5b.

<table><tr><td rowspan="2">Range</td><td rowspan="2">Method</td><td colspan="3">VideoPhy2</td></tr><tr><td>SA</td><td>PC</td><td>Joint</td></tr><tr><td rowspan="2">All noise(0.2–0.8)</td><td> $S_{global}$ </td><td>71.66</td><td>60.56</td><td>51.66</td></tr><tr><td> $S_{motion}$ </td><td>69.44</td><td>61.66</td><td>51.11</td></tr><tr><td rowspan="4">Mid noise(0.3–0.6)</td><td> $S_{global}$ </td><td>67.04</td><td>64.25</td><td>51.40</td></tr><tr><td> $S_{motion}$ </td><td>67.03</td><td>66.48</td><td>56.42</td></tr><tr><td> $S_{global} + var$ </td><td>67.59</td><td>64.80</td><td>52.51</td></tr><tr><td> $S_{motion} + var$ </td><td>66.48</td><td>68.16</td><td>55.31</td></tr></table>

Table 5: Effect of aggregating scores at different noise levels and with variance weighting on scoring for VideoPhy2 (standard) with Wan2.2 5B.

# 6 Discussion

Importance of mid-level noise. Our ablations in Section 5 show that the most informative selfscoring signal arises at intermediate noise levels. At very low noise, the perturbed latent remains too close to the original sample, making the residual largely local and less sensitive to higher-level violations of physical plausibility. At very high noise, much of the sample-specific structure is lost, so predictions are driven more by the model’s generic prior and become less discriminative. Mid-level noise provides the best balance: it preserves enough structure for the score to remain sample-specific while perturbing the latent enough to reveal inconsistencies in spatiotemporal dynamics. This explains the stronger PC and joint performance of mid-noise evaluation relative to low or high-noise regimes.

Inference scaling versus inference refinement. Comparing search and refinement reveals a clear dependence on the generation setting. In T2V, search is particularly effective because samples from the same prompt exhibit substantial scene and motion diversity, making selection from a larger candidate pool highly beneficial. This aligns with prior works [13] discussing the diversityperformance tradeoff in T2V. In contrast, I2V candidates are anchored to the same input image, reducing semantic diversity and leaving motion quality as the primary source of variation. As a result, the advantage of large-scale search diminishes, and the gap between search and refinement narrows. In this regime, motion-aware refinement becomes effective even with a smaller search budget. Overall, these results indicate that scaling search is more important for T2V, whereas in I2V the marginal benefit of additional diversity is lower and refinement plays a stronger role.

# 7 Conclusion and Limitations

We introduced Proprio, a generator-native inference-time framework for improving physical plausibility in video generation. By reusing the frozen generator’s own denoising dynamics as an intrinsic self-scoring signal, Proprio enables both search and refinement without external reward models. Across text-to-video and image-to-video benchmarks, Proprio consistently improves physical plausibility and provides a strong alternative to external evaluators. Proprio is inherently limited by the knowledge encoded in the base generator: it cannot recover dynamics that the model has not learned. In addition, like other inference-time scaling methods, Proprio increases inference cost by requiring additional candidate scoring, search, or refinement steps. Our refinement experiments also rely on distilled models for computational tractability, and scaling gradient-based refinement to larger, non-distilled generators remains challenging due to the cost of backpropagation through long denoising trajectories. An important direction for future work is therefore the development of more efficient scoring and refinement. Overall, our results suggest a shift in perspective for video generation: rather than relying solely on external supervision or auxiliary reward models, future systems may increasingly exploit generator-internal signals for self-evaluation and self-alignment.

# 8 Acknowledgments

This work was supported as part of the Swiss AI Initiative by a grant from the Swiss National Supercomputing Centre (CSCS) under project ID a144. Kaouther Messaoud is supported by Hi! PARIS and ANR/France 2030 program.

# References

[1] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.   
[2] Reyhane Askari Hemmat, Melissa Hall, Alicia Sun, Candace Ross, Michal Drozdzal, and Adriana Romero-Soriano. Improving geo-diversity of generated images with contextualized vendi score guidance. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.   
[3] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.   
[4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.   
[5] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.   
[6] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024.   
[7] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.   
[8] Tim Brooks, William Peebles, Aditya Ramesh, et al. Video generation models as world simulators. OpenAI technical report, 2024. Online; accessed from OpenAI.   
[9] Yuanhao Cai, Kunpeng Li, Menglin Jia, Jialiang Wang, Junzhe Sun, Feng Liang, Weifeng Chen, Felix Juefei-Xu, Chu Wang, Ali Thabet, et al. Phygdpo: Physics-aware groupwise direct preference optimization for physically consistent text-to-video generation. arXiv preprint arXiv:2512.24551, 2025.   
[10] Nicola Dall’Asen, Xiaofeng Zhang, Reyhane Askari Hemmat, Melissa Hall, Jakob Verbeek, Adriana Romero-Soriano, and Michal Drozdzal. Increasing the utility of synthetic images through chamfer guidance. arXiv preprint arXiv:2508.10631, 2025.   
[11] Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, and Di Huang. Initno: Boosting text-toimage diffusion models via initial noise optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
[12] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2025.   
[13] Mariam Hassan, Bastien Van Delft, Wuyang Li, and Alexandre Alahi. Factorized video generation: Decoupling scene construction and temporal synthesis in text-to-video diffusion models. arXiv preprint arXiv:2512.16371, 2025.   
[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
[15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[16] Vineet Jain, Kusha Sareen, Mohammad Pedramfar, and Siamak Ravanbakhsh. Diffusion tree sampling: Scalable inference-time alignment of diffusion models. In Second Workshop on Test-Time Adaptation: Putting Updates to the Test! at ICML 2025.   
[17] Sangwon Jang, Taekyung Ki, Jaehyeong Jo, Saining Xie, Jaehong Yoon, and Sung Ju Hwang. Self-refining video sampling. arXiv preprint arXiv:2601.18577, 2026.   
[18] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. In International Conference on Machine Learning, pages 28991–29017. PMLR, 2025.   
[19] Dan Kondratyuk, Lijun Yu, Xin Zhao, Mingxing Meng, Chenlin Yan, Alexandre Drouin, Maor Yi, Lijian Luo, Liangyu Gui, Zhenyao Wang, Wilson Sean, Cordelia Schmid, David Ross, Kihyuk Sohn, and Lu Jiang. Videopoet: A large language model for zero-shot video generation. In Proceedings of the 41st International Conference on Machine Learning, pages 25105–25124, 2024.   
[20] Alexander C Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly a zero-shot classifier. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2206–2217, 2023.   
[21] Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, and Masatoshi Uehara. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. arXiv preprint arXiv:2408.08252, 2024.   
[22] Xiner Li, Masatoshi Uehara, Xingyu Su, Gabriele Scalia, Tommaso Biancalani, Aviv Regev, Sergey Levine, and Shuiwang Ji. Dynamic search for inference-time alignment in diffusion models. arXiv preprint arXiv:2503.02039, 2025.   
[23] Gongye Liu, Bo Yang, Yida Zhi, Zhizhou Zhong, Lei Ke, Didan Deng, Han Gao, Yongxiang Huang, Kaihao Zhang, Hongbo Fu, et al. Beyond vlm-based rewards: Diffusion-native latent reward modeling. arXiv preprint arXiv:2602.11146, 2026.   
[24] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physicsgrounded image-to-video generation. In European Conference on Computer Vision, pages 360–378. Springer, 2024.   
[25] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations (ICLR), 2023.   
[26] Jiaxi Lv, Yi Huang, Mingfu Yan, Jiancheng Huang, Jianzhuang Liu, Yifan Liu, Yafei Wen, Xiaoxin Chen, and Shifeng Chen. Gpt4motion: Scripting physical motions in text-to-video generation via blender-oriented gpt planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024.   
[27] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, and Saining Xie. Scaling inference time compute for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2523–2534, 2025.   
[28] Xiaoyue Mi, Wenqing Yu, Jiesong Lian, Shibo Jie, Ruizhe Zhong, Zijun Liu, Guozhen Zhang, Zixiang Zhou, Zhiyong Xu, Yuan Zhou, et al. Video generation models are good latent reward models. arXiv preprint arXiv:2511.21541, 2025.   
[29] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 948–958, 2026.   
[30] Felipe Nuti, Tim Franzmeyer, and João F Henriques. Extracting reward functions from diffusion models. Advances in Neural Information Processing Systems, 36:50196–50220, 2023.   
[31] Adam Polyak, Yossi Adi, Leshem Choshen, Tavi Halperin, Tovi Ventura, Xi Zhang, Boliang Wang, Peng Yin, Dong Xu, Shir Gur, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.   
[32] Alexander Pondaven, Aliaksandr Siarohin, Sergey Tulyakov, Philip Torr, and Fabio Pizzati. Video motion transfer with diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22911–22921, 2025.

[33] Zhiwei Tang, Jiangweizhi Peng, Jiasheng Tang, Mingyi Hong, Fan Wang, and Tsung-Hui Chang. Inferencetime alignment of diffusion models with direct noise optimization. In Forty-second International Conference on Machine Learning.   
[34] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
[35] Jing Wang, Ao Ma, Ke Cao, Jun Zheng, Jiasong Feng, Zhanjie Zhang, Wanyuan Pang, and Xiaodan Liang. Wisa: World simulator assistant for physics-aware text-to-video generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.   
[36] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.   
[37] Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, et al. Vlipp: Towards physically plausible video generation with vision and language informed physical prior. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12360–12370, 2025.   
[38] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.   
[39] Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma, James Zou, and Stefano Ermon. Tfg: Unified training-free guidance for diffusion models. In Advances in Neural Information Processing Systems, 2024.   
[40] Jianhao Yuan, Fabio Pizzati, Francesco Pinto, Lars Kunze, Ivan Laptev, Paul Newman, Philip Torr, and Daniele De Martini. Likephys: Evaluating intuitive physics understanding in video diffusion models via likelihood preference. arXiv preprint arXiv:2510.11512, 2025.   
[41] Jianhao Yuan, Xiaofeng Zhang, Felix Friedrich, Nicolas Beltran-Velez, Melissa Hall, Reyhane Askari-Hemmat, Xiaochuang Han, Nicolas Ballas, Michal Drozdzal, and Adriana Romero-Soriano. Inference-time physics alignment of video generative models with latent world models. arXiv preprint arXiv:2601.10553, 2026.   
[42] Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodiffusion: Accelerating video diffusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025.   
[43] Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. Videorepa: Learning physics for video generation through relational alignment with foundation models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, .   
[44] XiangCheng Zhang, Haowei Lin, Haotian Ye, James Zou, Jianzhu Ma, Yitao Liang, and Yilun Du. Inference-time scaling of diffusion models through classical search. In NeurIPS 2025 Workshop on Structured Probabilistic Inference {\&} Generative Modeling, .

# A Additional experiments and ablations

# A.1 Diagnostic Study: Preference for natural video dynamics

To test whether the self-score captures a useful preference for natural video dynamics, we conduct a small but highly curated diagnostic check on Physics-IQ. We select 45 distinct scenarios from Physics-IQ and use their ground-truth videos as physically plausible references. For each case, we construct a matched physical-failure example using Turbo Wan2.2, either by prompting the model toward a specific implausible behavior or by generating multiple samples and selecting one with a clear physics violation. We manually curate these pairs so that the generated counterpart remains visually and semantically close to the reference where possible, while the main intended difference is physical plausibility. The physical failures include, unrealistic collisons, rigid objects changing shapes, unrealistic shadows, liquid dynamics, unrealistic falls, wrong reaction to magnet and more. We then score both the ground-truth video and the generated failure using the same Turbo Wan2.2 model with the motion Proprio loss, and compare which video is preferred by the self-score.

The ground-truth video is preferred in 82.2% of pairs (37/45), suggesting that the generator-native residual tends to favor natural video dynamics over the model’s own physically implausible generations. This study is intended only as a diagnostic sanity check supporting our interpretation that learned-distribution consistency can serve as a useful proxy for physical plausibility. We show examples of the curated pairs in Figures 11 and 12.

# A.2 Visual Quality

To assess whether the gains in physical plausibility come at the expense of visual quality, we additionally evaluate the generated videos on VBench [15]. As shown in Table 6, Proprio does not degrade perceptual quality across either T2V or I2V. Subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality remain close to the baseline across refinement, search, and search+refinement settings. In some cases, these metrics slightly improve, suggesting that Proprio’s physical-plausibility gains are not obtained by sacrificing visual quality. Overall, these results indicate that the improvements in physical plausibility obtained by Proprio do not come at the cost of perceptual video quality, and may in some cases slightly improve it.

Table 6: VBench quality metrics for Turbo Wan2.2 under baseline sampling, Proprio refinement (∇), search (BoN), and search + refinement (∇ + BoN). 

<table><tr><td>Method</td><td>Subject Consistency</td><td>Background Consistency</td><td>Motion Smoothness</td><td>Aesthetic Quality</td><td>Imaging Quality</td></tr><tr><td colspan="6">VideoPhy2 (T2V)</td></tr><tr><td>Turbo Wan2.2</td><td>0.92</td><td>0.92</td><td>0.99</td><td>0.51</td><td>0.67</td></tr><tr><td>+ Proprio(∇)</td><td>0.93</td><td>0.93</td><td>0.98</td><td>0.51</td><td>0.68</td></tr><tr><td>+ Proprio(BoN)</td><td>0.95</td><td>0.95</td><td>0.99</td><td>0.54</td><td>0.68</td></tr><tr><td>+ Proprio(∇ + BoN)</td><td>0.93</td><td>0.93</td><td>0.98</td><td>0.52</td><td>0.68</td></tr><tr><td colspan="6">PhysicsIQ (I2V)</td></tr><tr><td>Turbo Wan2.2</td><td>0.95</td><td>0.96</td><td>0.99</td><td>0.49</td><td>0.66</td></tr><tr><td>+ Proprio(∇)</td><td>0.95</td><td>0.96</td><td>0.99</td><td>0.49</td><td>0.66</td></tr><tr><td>+ Proprio(BoN)</td><td>0.95</td><td>0.96</td><td>0.99</td><td>0.48</td><td>0.68</td></tr><tr><td>+ Proprio(∇ + BoN)</td><td>0.96</td><td>0.96</td><td>0.99</td><td>0.49</td><td>0.66</td></tr></table>

# A.3 Time Comparison

Table 7 compares the runtime overhead of Proprio and WMReward relative to baseline generation. Proprio is particularly efficient in I2V, where refinement adds only limited overhead and remains substantially cheaper than WMReward guidance, both with and without search. In T2V, the runtime gap is smaller: Proprio refinement is comparable in cost to WMReward guidance, while Proprio search+refinement becomes the most expensive setting because it combines candidate selection with gradient-based refinement. However, it is important to note that inference-time scaling in general requires an increase in compute to search for the best generation and the time required scales linearly with number of samples used for the search.

Table 7: Runtime comparison of Proprio and WMReward in I2V and T2V settings using Turbo Wan. We report average runtime in seconds and relative runtime normalized to the baseline. We use one sample generation for this experiment across 3 runs. Numbers increase linearly with samples. 

<table><tr><td>Domain</td><td>Method</td><td>Time (s)</td><td> $\times$  Time</td></tr><tr><td rowspan="5">I2V</td><td>Baseline</td><td>37.54  $\pm$  0.06</td><td>1.00</td></tr><tr><td>Ours  $\nabla$ </td><td>49.11  $\pm$  0.25</td><td>1.31</td></tr><tr><td>Ours (BON +  $\nabla$ )</td><td>67.59  $\pm$  0.11</td><td>1.80</td></tr><tr><td>WMReward [41]  $\mathcal{G}$ </td><td>87.35  $\pm$  4.46</td><td>2.33</td></tr><tr><td>WMReward [41] (BON +  $\mathcal{G}$ )</td><td>92.24  $\pm$  0.44</td><td>2.46</td></tr><tr><td rowspan="5">T2V</td><td>Baseline</td><td>14.36  $\pm$  0.18</td><td>1.00</td></tr><tr><td>Ours $\nabla$ </td><td>29.72  $\pm$  0.05</td><td>2.07</td></tr><tr><td>Ours (BON +  $\nabla$ )</td><td>49.67  $\pm$  0.06</td><td>3.46</td></tr><tr><td>WMReward [41]  $\mathcal{G}$ </td><td>30.38  $\pm$  7.48</td><td>2.12</td></tr><tr><td>WMReward [41] (BON +  $\mathcal{G}$ )</td><td>37.69  $\pm$  4.63</td><td>2.62</td></tr></table>

# A.4 Effect of variance on distilled model

Table 8 shows that variance weighting has only a limited effect on the distilled Turbo model. On Physics-IQ, using variance weighting does not improve performance and in fact leads to slightly lower scores for the global and combined variants, while leaving the dynamic variant unchanged. A similar pattern appears on VideoPhy2 where the differences are small across all three variants, and only minor shifts between semantic adherence and physical consistency. In particular, variance weighting slightly improves PC for the global and combined variants, but these gains are offset by lower SA or joint performance, and the dynamic variant is nearly unchanged. This contrasts with the non-distilled Wan model, where variance weighting provides clearer gains. One possible explanation is that distillation compresses the denoising trajectory into fewer steps, making the timestep-wise residual estimates more uniform in reliability. We therefore, do not use variance weighting for optimization setting as it’s done on the distilled model.

Table 8: Effect of variance weighting on the distilled Turbo model. 

<table><tr><td rowspan="2">Variant</td><td colspan="2">Physics-IQ</td><td colspan="6">VideoPhy2 Standard</td></tr><tr><td>w/o variance</td><td>w/ variance</td><td>SA w/o variance</td><td>PC</td><td>Joint</td><td>SA w/ variance</td><td>PC</td><td>Joint</td></tr><tr><td> $S$ </td><td>34.5</td><td>34.3</td><td>66.7</td><td>73.3</td><td>58.9</td><td>65.9</td><td>73.7</td><td>56.4</td></tr><tr><td> $S_{motion}$ </td><td>34.6</td><td>34.6</td><td>65.0</td><td>74.4</td><td>56.7</td><td>64.8</td><td>74.3</td><td>55.9</td></tr><tr><td> $S_{hybrid}$ </td><td>35.1</td><td>34.6</td><td>64.4</td><td>74.4</td><td>56.1</td><td>64.2</td><td>74.9</td><td>56.4</td></tr></table>

# A.5 Motion-stratified results

While motion is an important aspect of physical plausibility, its magnitude alone is not a proxy for physical correctness. A physically plausible video may contain little motion when the scene demands it, whereas an implausible generation may contain excessive or inconsistent motion. We therefore evaluate whether Proprio improves physical plausibility among samples with comparable motion magnitude. We compute a motion magnitude score per video and split videos into quartiles within each dataset for TurboWan2.2. The low-motion bin contains videos up to the 33rd percentile, the medium-motion bin contains videos between the 33rd and 67th percentiles, and the high-motion bin contains videos above the 67th percentile. This data-adaptive binning avoids arbitrary thresholds and enables fair within-dataset comparisons across motion regimes. As shown in Table 9, Proprio-selected samples generally improve benchmark performance within motion bins. On Physics-IQ, Proprio outperforms random selection, with particularly large gains in the low- and high-motion regimes. On VideoPhy2, Proprio improves the PC metric in the low- and medium-motion bins and remains comparable in the high-motion bin. These results indicate that Proprio’s gains are not explained solely by changes in motion magnitude; instead, the score continues to identify more physically plausible generations among videos with similar motion levels.

Table 9: Motion-stratified benchmark control. Within each motion bin, we compare Proprio samples against random samples. For Physics-IQ, we report benchmark score; for VideoPhy2, we report PC metric. Positive ∆ indicates Proprio-selected samples outperform random samples. 

<table><tr><td>Benchmark</td><td>Motion bin</td><td>Proprio</td><td>Random</td><td> $\Delta$ </td></tr><tr><td rowspan="3">Physics-IQ</td><td>Low</td><td>32.0</td><td>24.8</td><td>+7.2</td></tr><tr><td>Medium</td><td>33.0</td><td>33.0</td><td>+0.0</td></tr><tr><td>High</td><td>41.0</td><td>33.6</td><td>+7.4</td></tr><tr><td rowspan="3">VideoPhy2</td><td>Low</td><td>81.9</td><td>70.8</td><td>+11.1</td></tr><tr><td>Medium</td><td>78.9</td><td>77.4</td><td>+1.5</td></tr><tr><td>High</td><td>58.0</td><td>59.4</td><td>-1.4</td></tr></table>

# A.6 Human Evaluation

We conduct a human evaluation with overall results presented in Sec 4.3 but we provide more results in this section. Figure 5 provides the human evaluation results split by generation setting. The trend is consistent across both I2V and T2V: voters prefer Proprio for physical plausibility in the majority of comparisons, with 67.9% preference in I2V and 66.3% in T2V. This suggests that Proprio’s improvements are perceived by humans in both conditional imageto-video and text-to-video settings. For visual quality, the pattern is more mode-dependent. In I2V, most annotators report no preference, which is expected because both videos are anchored by the same input image and therefore often share very similar scene content and appearance. In T2V, visual preferences are more decisive, with 55.3% favoring Proprio, likely because generated scenes can differ more substantially in terms of composition. Overall, the human study indicates that Proprio’s main benefit is improved perceived physical plausibility, while visual quality is generally preserved and sometimes preferred.

![](images/19cf94ee83d33ccc519feaf4d2c1bd2e20c3a35c0b6a5bd6a1d742619e0782ea.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Physical | Visual |
| -------- | -------- | ------ |
| I2V - Ours | 67.9% | 39.3% |
| I2V - Base. | 17.1% | 15.7% |
| I2V - No preference | 15.0% | 45.0% |
| T2V - Ours | 66.3% | 55.3% |
| T2V - Base. | 23.7% | 26.8% |
| T2V - No preference | 10.0% | 17.9% |
</details>

Figure 5: Human preference results split by generation setting.

We additionally provide a screenshot example of the study with full instructions given to participants in figure 6. Participants volunteered to answer the questions with no compensation. No personal data were collected, only answers to the questions of study.

# B Additional Discussion

Semantic Adherence - Physical Commonsense tradeoff. Improving physical plausibility in T2V is inherently multi-objective: candidates that faithfully realize complex prompts may involve difficult interactions and therefore be more prone to physical errors, while physically safer generations may simplify the requested action or reduce interaction complexity. This has also been observed in prior works [41]. While Proprio exhibits similar tradeoff, some preserve semantic adherence better; for example $S _ { \mathrm { g l o b a l } }$ generally preserves SA better while its improvement on physics commonsense is slightly less than $S _ { \mathrm { m o t i o n } }$ as seen in Section 4. Similarly, the text-aware dynamic mask in Section 5 better preserves semantic adherence, at the cost of slightly weaker physical improvements. In practice, the choice of score can therefore be matched to the application: $S _ { \mathrm { g l o b a l } }$ is preferable when prompt fidelity is critical, $S _ { \mathrm { m o t i o n } }$ when physical dynamics are the main concern, and $S _ { \mathrm { h y b r i d } }$ as a balanced default for improving physical plausibility while maintaining semantic alignment.

![](images/3d1e968e6df65913db78c196bc3bafaed407ef2bdaa3e504843a5ec3c4ef0c35.jpg)

<details>
<summary>text_image</summary>

Physical plausibility
• Watch both videos and choose the one that is relatively better in terms of physical realism.
• Sometimes both videos may be imperfect, but please select the one with better physics than the other.
• Pay attention to object motion, interactions, and overall realism.
Visual quality
• Choose the video with better visual quality.
• Focus on overall visual appearance and rendering quality.
Participant ID: d1e9x23-194e-45cb-b2cc-d6f446b7d52
Which video is more physically plausible?
• Left  Right  No preference
Which video has better visual quality?
• Left  Right  No preference
</details>

Figure 6: A screenshot from the human evaluation study showing full instructions and layout used.

# C More Details on optimization

Algorithm 1 summarizes the self-refinement procedure used in Sec. 3. Starting from the original sampling seed, we optimize a Gaussian-affine reparameterization of the initial noise, $\zeta = \mu + \sigma \odot \zeta _ { 0 }$ , while keeping the video generator fixed. At each iteration, the current noise is used to generate a latent sample $z _ { 0 } ( \zeta ; c )$ , which is evaluated with the Proprio score together with a KL regularizer that keeps the refined noise close to the standard Gaussian prior. The optimization is performed over $( \mu , \log \sigma )$ using Adam, and the final output is the sample achieving the lowest unregularized Proprio score during refinement. This parameterization constrains the update to a structured family of perturbations, which stabilizes optimization and reduces drift away from the generator’s original sampling distribution.

Algorithm 1 Proprio Self-Refinement Algorithm   
1: Sample base noise $\zeta_0\sim \mathcal{N}(0,I)$ from the initial seed.   
2: Initialize $\mu \leftarrow \mathbf{0}$ and $\log \sigma \leftarrow \mathbf{0}$ 3: for $n = 1$ to $N$ do   
4: $\sigma \leftarrow \exp (\log \sigma)$ 5: $\zeta \leftarrow \mu +\sigma \odot \zeta_0$ 6: Generate latent sample $z_{0}(\zeta ;c)$ with the frozen generator   
7: Evaluate Proprio score $S(z_0(\zeta ;c);c)$ 8: Compute $\mathcal{L}_{\mathrm{KL}} = \frac{1}{C}\sum_{d = 1}^{C}\frac{1}{2}\left(\mu_d^2 +\sigma_d^2 -1 - 2\log \sigma_d\right)$ 9: Form the optimization objective $\mathcal{L}(\mu ,\sigma) = S(z_0(\zeta ;c);c) + \beta \mathcal{L}_{\mathrm{KL}}$ 10: Update $(\mu ,\log \sigma)$ with Adam (optional gradient clipping)   
11: end for   
12: return the sample $z_0(\zeta^\star ;c)$ achieving the lowest unregularized Proprio score $S(z_0(\zeta ;c);c)$ during optimization

# D Dynamic masking strategies

We consider three dynamic masking strategies in ablations 5, each designed to emphasize regions that are most informative for physical plausibility. All three produce a spatiotemporal saliency volume $S \in \mathbb { R } ^ { T \times H \times W }$ that is later converted into the latent dynamic mask used for scoring and refinement. The variants differ only in how this saliency volume is estimated.

Motion mask. The motion mask is derived from temporal changes in the generated samples. Given a sequence of spatiotemporal features or decoded frames $\{ Y _ { t } \} _ { t = 1 } ^ { T }$ , we compute framewise motion maps using a temporal difference operator,

$$
D _ {t} = \phi (Y _ {t}, Y _ {t - 1}), \tag {14}
$$

where ϕ measures local temporal change. The resulting maps are normalized and projected onto the latent grid to obtain a latent saliency volume $S _ { \mathrm { m o t i o n } }$ . Intuitively, this mask highlights regions with significant temporal variation, thereby emphasizing object motion, interaction changes, and potential dynamic inconsistencies.

Textaware mask. The textaware mask derives motion from cross-attention, making it explicitly prompt-grounded. For selected transformer layers and probe timesteps, we extract the cross-attention affinity between latent query tokens and prompt tokens, average over the relevant text tokens, and reshape the result into a spatiotemporal affinity map

$$
A _ {t} \in \mathbb {R} ^ {H \times W \times T}. \tag {15}
$$

We then convert this affinity into a motion-like signal by measuring its temporal variation,

$$
S _ {\text { text }} (t) = \left| A _ {t} - A _ {t - 1} \right|, \tag {16}
$$

which highlights regions whose prompt-conditioned attention changes over time. In this way, the mask focuses on dynamic regions that are both visually active and semantically relevant to the text prompt.

Attention-based mask. The attention-based mask uses internal self-attention feature changes as a proxy for motion, similar in spirit to prior work using attention dynamics as an internal motion signal [32]. For selected self-attention layers, we extract latent query and key features at consecutive timesteps and measure their temporal similarity at each spatial location. Let $q _ { t } ( h , w )$ and $k _ { t - 1 } ( h , w )$ denote the corresponding features. We define the saliency as

$$
S _ {\text { attn }} (t, h, w) = \operatorname{ReLU} \left(1 - \cos \left(q _ {t} (h, w), k _ {t - 1} (h, w)\right)\right). \tag {17}
$$

Locations whose internal features change more strongly over time therefore receive higher saliency. Unlike the other mask variants, this masking uses the model’s own internal temporal feature evolution as a learned motion proxy.

Summary. The three masking strategies reflect different notions of dynamic saliency: the motion mask uses explicit visible motion, the textaware mask emphasizes prompt-grounded temporal changes, and the attention-based mask uses internal self-attention feature changes as a model-centric proxy for motion. As shown in Table 4, all three provide useful dynamic signals, but induce different tradeoffs between semantic adherence and physical consistency.

# E More Implementation Details

Unless otherwise stated, Proprio scoring is computed in latent space over a fixed set of scheduler timesteps sampled from the mid-noise regime, using multiple perturbation samples per timestep and inverse-variance weighting across timesteps. For refinement, we backpropagate through a reduced number of denoising steps while keeping the generator frozen, and optimize only the Gaussian-affine parameters of the initial noise. Our default motion score uses the motion-based spatiotemporal mask described in Sec. 3, which is fixed during scoring and refinement unless otherwise specified. For hybrid search-and-refinement, all methods use the same candidate pool before refinement.

Table 10 summarizes the optimization hyperparameters used for Proprio self-refinement. Unless otherwise stated, we optimize the Gaussian-affine noise parameters for 10 steps using Adam with a learning rate of $2 \times 1 0 ^ { - 3 }$ . We apply gradient clipping for stability and use a small KL regularization term to keep the optimized noise close to the standard Gaussian prior. For score estimation during refinement, we average over 4 noise samples per timestep, which provides a more stable optimization signal while keeping the computational cost manageable. In all experiments, $\mu$ and $\sigma$ are calculated per channel.

Table 10: Optimization hyperparameters used for Proprio self-refinement. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimization steps (K)</td><td>10</td></tr><tr><td>Learning rate</td><td> $2 \times 10^{-3}$ </td></tr><tr><td>Gradient clipping</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>KL regularization weight ( $\beta$ )</td><td> $5 \times 10^{-3}$ </td></tr><tr><td>Number of noise samples per timestep</td><td>4</td></tr></table>

For all experiments, we use 4 mid-noise levels linearly spaced between 0.3-0.6 of the sampling steps. For non-distilled models, variance weighting is used by using 4 random iterations per noise level and calculating variance in prediction. For the hybrid loss, $\lambda = 0 . 2$ is used for all models.

# F Flow Matching Background

Flow matching formulates generation as learning a time-dependent vector field that transports a simple base distribution to the data distribution in latent space. Let $z _ { 1 }$ denote a clean video latent and let $z _ { 0 } \sim \mathcal { N } ( 0 , I )$ denote a Gaussian noise sample. A continuous path between noise and data is defined by

$$
z _ {t} = (1 - t) z _ {0} + t z _ {1}, \quad t \in [ 0, 1 ], \tag {18}
$$

with corresponding target velocity

$$
v ^ {\star} \left(z _ {0}, z _ {1}\right) = z _ {1} - z _ {0}. \tag {19}
$$

A conditional generator $\hat { v } _ { \theta } ( z _ { t } , t , c )$ is then trained to predict this transport field by minimizing

$$
\mathcal {L} _ {\mathrm{FM}} = \mathbb {E} _ {z _ {1}, z _ {0}, t} \left[ \| \hat {v} _ {\theta} (z _ {t}, t, c) - (z _ {1} - z _ {0}) \| _ {2} ^ {2} \right]. \tag {20}
$$

In practice, modern video generators implement this perturbation through a scheduler rather than an explicit analytic path, so we write the noisy latent more generally as

$$
z _ {t} = \mathcal {P} (z _ {1}, \epsilon , t), \quad \epsilon \sim \mathcal {N} (0, I), \tag {21}
$$

and the corresponding target transport under the standard velocity parameterization as

$$
v ^ {\star} (z _ {1}, \epsilon) \approx \epsilon - z _ {1}. \tag {22}
$$

At inference time, sampling starts from Gaussian noise and iteratively applies the learned vector field across timesteps to produce a latent video. In our method, we reuse this same flow-prediction objective at test time: by evaluating how well the frozen model predicts its target transport on perturbed generated samples, we obtain an intrinsic signal for ranking and refinement.

# G Extended related work

Physical plausibility in video generation. Recent video generative models [8, 6, 19, 34, 36, 38, 12, 31, 29, 18, 29] have achieved impressive visual fidelity and temporal coherence, but multiple recent studies show that strong perceptual quality does not imply strong physical understanding or physically consistent dynamics. Diagnostic works such as Physics-IQ and related physical-law evaluations highlight persistent failures in collisions, object interactions, support relations, and generalization of physical behavior, suggesting that current video generators often reproduce surface realism without reliably capturing underlying physical structure [29, 18, 29]. Existing approaches to address this problem largely fall into three groups. A first line of work improves physical fidelity during training by introducing physics-aware data, supervision, or learning objectives, for example through physics-focused pretraining, preference optimization, or curated physics-aware corpora [35, 9]. A second line of work incorporates explicit physical structure or simulation into the generation process itself. For example, PhysGen uses rigid-body simulation to ground image-to-video generation, while GPT4Motion leverages GPT-4 to script Blender-based physical motions before video synthesis [24, 26]. A third line of work addresses physical plausibility at inference time through external guidance. VLIPP uses a vision-language model as a coarse motion planner with physics-aware reasoning, and WMReward uses VJEPA-2 as an external latent world model to score candidates and guide sampling toward physically plausible videos [37, 41]. Compared with these approaches, our method does not introduce external simulators, planners, or world-model rewards; instead, it reuses the frozen generator’s own latent residual dynamics as a generator-native self-scoring signal for inference-time ranking and refinement.

Inference-time alignment for diffusion models. Inference-time alignment methods for diffusion models can be grouped into two broad families: search-based methods and optimization-based methods. Search-based methods improve alignment by spending additional compute on candidate exploration, trajectory branching, or resampling, followed by reward or value-based selection. This includes Best-of-N style inference-time scaling, value-based decoding, and dynamic search or resampling procedures that explicitly trade off exploration and exploitation during denoising [21, 22, 27]. Optimization-based methods instead directly modify the sampling trajectory or the underlying noise variables to improve a target objective. In particular, DNO optimizes injected noise online during the denoising process with respect to a reward [33], while InitNO optimizes the initial latent using attention-derived prompt-alignment criteria and a distribution-preserving regularizer [11]. More generally, training-free guidance methods steer diffusion trajectories at test time using external objectives without retraining the generator [39, 2, 10]. Collectively, these works show that large gains can be achieved at inference time alone, without changing model weights. However, they typically depend on externally defined rewards, alignment heuristics, or verifier signals. In contrast, Proprio uses a generator-native objective: it scores and refines samples directly through the frozen video generator’s own latent residual dynamics, avoiding external reward models.

Reward extraction from diffusion models. A broader line of work has explored whether diffusion models contain reward-like information that can be extracted or repurposed. In decision-making, Nuti et al. [30] define a relative reward function between a base and an expert diffusion model, and learn an auxiliary reward network by aligning its gradient to the difference in the two models’ outputs. Their setting is primarily offline RL and control, but it provides an early indication that diffusion dynamics may encode preference-relevant structure beyond standard sample generation.

Latent reward modeling for generation. More recent work has studied this question directly in image and video generation, arguing that rewards should be modeled in latent space rather than inherited from pixel-space VLM evaluators. Learning rewards directly on noisy latent states has been shown to better align with latent-space generation and reduce the overhead of external evaluation [23, 28]. Relatedly, LikePhys [40] evaluates intuitive physics understanding in video diffusion models by comparing denoising losses on curated valid–invalid simulator-generated pairs and summarizing the preference with a Plausibility Preference Error (PPE) metric. In contrast, Proprio does not train a separate reward model, does not learn from pairs of diffusion models, and does not require curated valid–invalid physics pairs. Instead, it directly reuses the frozen generator’s own residual under controlled perturbations as a training-free inference-time self-scoring signal for ranking and refinement.

# H Broader and societal impact.

This work aims to improve the physical plausibility of generated videos, which may benefit applications such as robotics, simulation, and world modeling. More physically consistent video generation could make generative models more useful for studying agent behavior, planning, training embodied systems, and building controllable environments for decision-making and reasoning. At the same time, improving the realism of synthetic video also carries risks. More plausible generated videos may be harder to distinguish from real footage and could be misused for deception, misinformation, or other forms of misleading synthetic media. While we acknowledge these risks, our work is intended purely for academic research, and any examples produced in this work will be clearly labeled as synthetic. We hope this research contributes to safer and more controllable generative systems, while encouraging continued work on safeguards, transparency, and responsible deployment.

# I Additional qualitative results

Figures 7 and 8 as well as figures 9 and 10 show additional qualitative results for T2V and I2V settings respectively. We compare baseline performance on TurboWan2.2 with the optimized version highlighting the improvement in physics plausibility of the generations. We summarize the prompts used with the videos above the figures. We additionally provide full videos comparison in our website for better comparison on the dynamics as described in Appendix ??.

![](images/95f07d3cbb27be8eab13ece6cf61455f57a025e74682906714ee920f6a887a22.jpg)

<details>
<summary>text_image</summary>

A potter applies a thin layer of slip to a clay bowl
Turbo Wan
Proprio
A person punching boxing bag
Wooden beam broken in two
A crane gently lifts a pallet of bricks
Turbo Wan
Proprio
A child's toy hammer smashes a small plastic egg
A hoverboard rider performs a 360-degree spin
Turbo Wan
Proprio
</details>

Figure 7: Overall qualitative results comparing our Proprio method against baseline TurboWan2.2. Visual examples from Proprio showing that our method produces more physically plausible generations for the T2V setting.

![](images/a80b606591b73e01a6c764cf7b40b3f3f1219ecad5c86f4c4cb3fe8e66210460.jpg)

<details>
<summary>text_image</summary>

A person uses a skateboard to perform a somersault trick
A potter uses a sponge to gently remove excess water
Turbo Wan
Proprio
A paint roller is used to apply a coat of brown paint
Two jetski race side-by-side
Turbo Wan
Proprio
The Segway travels a short distance, delivering the package
A rolling pin flattens a sphere of cookie dough into a disc
Turbo Wan
Proprio
</details>

Figure 8: Overall qualitative results comparing our Proprio method against baseline TurboWan2.2. Visual examples from Proprio showing that our method produces more physically plausible generations for the T2V setting.

![](images/373ade5dbbf6827688042f657600a5a364200cf5c9082727c85fbb6f88b6b628.jpg)

<details>
<summary>text_image</summary>

A tennis ball attached to a magnet and string is
hanging in front of a mirror
Turbo Wan
Proprio
A woven basket is hanging from a rope with a strong magnet.
An orange tennis ball is placed on a table beneath it.
Turbo Wan
Proprio
An orange ball rolls out of a black pipe that is sitting on the
table towards the right side
A teapot on a rotating display base that rotates
in front of a mirror
A yellow mug is placed on a rotating turntable
that rotates illuminated by a spotlight
A 30lb kettlebell resting on a wooden table next to a mirror.
A tennis ball rolls towards the kettlebell
Turbo Wan
Proprio
</details>

Figure 9: Overall qualitative results comparing our Proprio method against baseline TurboWan2.2. Visual examples from Proprio showing that our method produces more physically plausible generations for the I2V setting.

![](images/2954361b39ae190c26e7d4993f2c23a7665b6ad62d2c8f76a4e34c3e1b53efdd.jpg)

<details>
<summary>text_image</summary>

A teapot is placed on a wooden table. a piece of silk fabric is lowered on the teapot to cover it.
Turbo Wan
Proprio
A yellow mug and a blue wooden block are placed on a rotating turntable that rotates
Turbo Wan
Proprio
A pink block is being lowered towards a simple structure made of colorful blocks resembling a gate.
A knife is slicing through the tangerine.
Turbo Wan
Proprio
A potato is held by a grabber tool and dropped into a tall glass containing blue liquid
Air is being pumped in the balloon
</details>

Figure 10: Overall qualitative results comparing our Proprio method against baseline TurboWan2.2. Visual examples from Proprio showing that our method produces more physically plausible generations for the I2V setting.

![](images/0b58a032cecce40ce333556c063cd88cd160f3d8c2250216737adcb3e2a04f41.jpg)

<details>
<summary>text_image</summary>

Ground truth
Generated Failure
Ground truth
Generated Failure
Ground truth
Generated Failure
Ground truth
Generated Failure
Ground truth
Generated Failure
Generated Failure
</details>

Figure 11: Correctly preferred diagnostic examples. Each pair shows a ground-truth Physics-IQ video and a generated Turbo Wan2.2 sample with a visible physical failure. These examples are drawn from the subset where the dynamic Proprio self-score correctly prefers the ground-truth video, illustrating that the generator-native residual can favor natural video dynamics over physically implausible generations.

![](images/238cd8ecc3ebe0b704590f526a6bb9b37768b633ac6c759b8d26b2baca5f37fb.jpg)

<details>
<summary>text_image</summary>

Ground truth
Generated Failure
Ground truth
Generated Failure
Ground truth
Generated Failure
Ground truth
Generated Failure
Ground truth
Generated Failure
</details>

Figure 12: Correctly preferred diagnostic examples. Each pair shows a ground-truth Physics-IQ video and a generated Turbo Wan2.2 sample with a visible physical failure. These examples are drawn from the subset where the Proprio self-score correctly prefers the ground-truth video, illustrating that the generator-native residual can favor natural video dynamics over its own physically implausible generations.