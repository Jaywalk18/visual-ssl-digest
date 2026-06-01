# Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis

Hila Chefer \* 1 Patrick Esser \* 1

Dominik Lorenz 1 Dustin Podell 1 Vikash Raja 1 Vinh Tong 1 Antonio Torralba 1 2 Robin Rombach 1

https://bfl.ai/research/self-flow

![](images/6fe7b31d5079ef7fc372eb22525cee6ffd068d9ee9b6ac348db417d4ca8598f3.jpg)  
Figure 1. Results obtained by our self-supervised flow matching framework, Self-Flow. (a) On text-to-image generation, our method converges ∼ 2.8× faster than REPA (Yu et al., 2024), the predominant external-alignment method, without using any external models or supervision. Notably, REPA plateaus while our method continues to improve. (b) Compared to vanilla flow matching, our approach improves structural coherence, text rendering, and temporal consistency.

# Abstract

Strong semantic representations improve the convergence and generation quality of diffusion and flow models. Existing approaches largely rely on external models, which require separate training, operate on misaligned objectives, and exhibit unexpected scaling behavior. We argue that this dependence arises from the model’s training objective, which poses a denoising task with little incentive to learn semantic representations. We introduce Self-Flow: a self-supervised flow matching paradigm that integrates representation learning within the generative framework. Our key mechanism, Dual-Timestep Scheduling, applies heterogeneous noise levels across tokens, creating an information asymmetry that forces the model to infer missing information from corrupted inputs. This drives learning strong representations alongside generative capabilities without external supervision. Our method generalizes across modalities and enables multi-modal training while following expected scaling laws, achieving superior image, video, and audio generation.

# 1. Introduction

Modern generative models (Rombach et al., 2022a; Peebles & Xie, 2023; Ma et al., 2024; Esser et al., 2024), trained on vast data using extensive computational resources, can be dramatically improved by aligning their internal features with those of a frozen image encoder, for example, the 86M parameter model DINO (Yu et al., 2024). This encoder was trained not to generate, but to discriminate, i.e., to cluster images by semantic similarity. Its effectiveness for generative modeling exposes a gap: flow models do not learn strong representations on their own, although they help generation. External alignment offers a practical remedy: borrowing representations from a model that did learn them.

However, this approach has fundamental limitations. First, external alignment fails to uphold expected scaling laws, with stronger encoders often exhibiting diminished or even negative returns (Singh et al., 2025) (Sec. 3.2). Moreover, as we demonstrate in this work (see Sec. 4), scaling the generative model does not yield proportional improvements when relying on external alignment. Second, these methods fail to generalize across modalities: for video and audio generation, we find that alignment with most external encoders actually harms performance (Sec. 4), making external alignment less suitable for multi-modal models that must handle diverse data distributions within a single framework. Finally, it is difficult to anticipate which encoder will be effective for a given task. For example, aligning text-to-image models with SigLIP 2 (Tschannen et al., 2025) performs worse than DINOv2 (Oquab et al., 2024) (Sec. 4), despite the former being explicitly trained with text supervision and multi-aspect ratio support, properties seemingly better suited for the task.

To avoid the use of external representations, existing approaches opt to rely on the model’s natively learned features and the semantic asymmetry between different layers (Jiang et al., 2025; Haghighi et al., 2025). However, such formulations remain limited by the semantics naturally learned by the flow objective, and lag behind external alignment.

In contrast to both approaches above, we propose to directly integrate a self-supervised framework into flow matching to actively strengthen representations beyond those learned by the generative objective alone. To this end, we propose Dual-Timestep Scheduling, which applies two distinct noise levels to different subsets of input tokens, creating an information asymmetry in which some tokens are more heavily corrupted than others. We perform two forward passes: one with the mixed, heterogeneously-noised input, and one with a cleaner input where all tokens are noised at the lower of the two levels. The self-supervised objective is to predict, from the mixed input, the representations the model produces given the cleaner input. Combined with the standard flow loss on the heterogeneously-noised input, the model thus learns both dense, flow-based reconstruction and semantic feature prediction within a unified framework.

Since our formulation operates purely on the model’s internal representations without relying on external encoders, it naturally extends to both single-modality and joint multimodal training. Fig. 1 shows qualitative examples from our jointly-trained image, video, and audio model. Compared to standard flow matching, our method yields improvements in structural coherence, particularly for challenging structures like faces and hands, as well as text rendering accuracy and temporal consistency in video. Moreover, our method is agnostic to autoencoder choice: we demonstrate consistent improvements across SD (Rombach et al., 2022a), FLUX.2 (Black Forest Labs, 2025), Wan2.2 (Wan et al., 2025), Songbloom (Yang et al., 2025), and representation autoencoders (Zheng et al., 2025) (Sec. 4).

We evaluate on image, video, audio, and multi-modal generation, and show that our framework outperforms leading external alignment methods in each setting. These results suggest that joint optimization of generation and representations offers a robust, scalable, and general path forward.

# 2. Related Work

Representation Learning. Representation learning aims to learn powerful semantic representations through diverse pretraining objectives. Contrastive methods such as Sim-CLR (Chen et al., 2020a), MoCo (He et al., 2019; Chen et al., 2020b; 2021), and BYOL (Grill et al., 2020) learn by maximizing agreement between augmented views of the same image. CLIP (Radford et al., 2021) and SigLIP (Zhai et al., 2023; Tschannen et al., 2025) extend this paradigm to vision-language alignment, enabling zero-shot transfer across tasks. DINO variants (Caron et al., 2021; Oquab et al., 2024; Simeoni et al. ´ , 2025) train a student network to match a momentum-updated teacher. Masked autoencoding (MAE) (Vincent et al., 2010; Pathak et al., 2016) and its variants (He et al., 2022; Bao et al., 2021; Assran et al., 2023) offer learning representations by reconstructing masked portions of the input.

Representation Alignment for Generation. Existing methods can be categorized into those that align with external models and those that do not. Recent work has shown that aligning diffusion and flow model features with external pretrained encoders can significantly accelerate training and improve generation quality (Yu et al., 2024; Yao & Wang, 2025; Leng et al., 2025a;b; Pernias et al., 2023), with extensions to domain-specific settings such as physics in video (Zhang et al., 2025) and geometry in 3D (Wu et al., 2025b). Another line of work trains generative models directly on semantic representations rather than reconstruction-driven latents (Wu et al., 2025a; Zheng et al., 2025), though these methods remain tied to specific encoders which limits their reconstruction performance and hinders adaption across resolutions and modalities. For completeness, we show in Sec. 4 and App. G.2 that our method improves the performance of RAE (Zheng et al., 2025), demonstrating our method’s robustness to autoencoder choice. As discussed in Sec. 1, external alignment methods exhibit fundamental limitations such as unexpected scaling behavior (Sec. 3.2) and limited generalization across datasets and modalities (Sec. 4), motivating the need for a unified approach that eliminates dependence on external models entirely.

Existing methods that do not rely on external models can be broadly categorized into two groups. The first incorporates explicit self-supervised objectives at the cost of modifying the model’s training dynamics, often necessitating an additional stage of pure diffusion fine-tuning to close the train-inference gap (Zheng et al., 2023; Gao et al., 2023; Chen et al., 2025b; Zhu et al., 2024). The second preserves the diffusion framework (Jiang et al., 2025; Wang & He, 2025; Haghighi et al., 2025) and employs diffusion features to perform alignment. However, the latter methods rely on the assumption that deeper layers naturally learn strong representations, while the former obtain results that are less favorable (Wang & He, 2025; Zheng et al., 2023; Gao et al., 2023). Overall, methods that use external representations have consistently outperformed those without. Our novel Self-Flow approach closes this gap: by integrating self-supervised learning directly into flow matching, we surpass external alignment methods without requiring any external models.

# 3. Method

# 3.1. Preliminaries

Flow matching models learn to transport samples from a simple noise distribution to the data distribution by modeling a continuous-time probability path. Our approach builds on rectified flows (Liu et al., 2022; Albergo & Vanden-Eijnden, 2023; Lipman et al.), which constructs straight-line trajectories between noise and data.

Let $\mathbf { x } _ { 0 } ~ \in ~ \mathbb { R } ^ { N \times C }$ denote clean data represented as a sequence of N tokens, each of dimension C. This formulation naturally accommodates diverse input modalities, including image patches, video frames, and audio segments. We define a probability path by linearly interpolating between a noise distribution $\mathbf { x } _ { 1 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and the data distribution:

$$
\mathbf {x} _ {t} = (1 - t) \mathbf {x} _ {0} + t \mathbf {x} _ {1}, \quad t \in [ 0, 1 ], \tag {1}
$$

where t parameterizes the interpolation, with t = 1 corresponding to pure noise and t = 0 corresponding to clean data. The velocity field along this path is given by $\begin{array} { r } { \mathbf { v } _ { t } = \frac { d \mathbf { x } _ { t } } { d t } = \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } } \end{array}$ . A neural network $f _ { \theta } ( \mathbf { x } _ { t } , t )$ is trained to predict the velocity field by minimizing:

$$
\mathcal {L} _ {\text { gen }} = \mathbb {E} _ {\mathbf {x} _ {0}, \mathbf {x} _ {1}, t} \| f _ {\theta} (\mathbf {x} _ {t}, t) - (\mathbf {x} _ {1} - \mathbf {x} _ {0}) \| ^ {2}, \tag {2}
$$

where $t \sim p ( t )$ denotes the timestep sampling distribution (see App. A.2). At inference, generation proceeds by solving the ordinary differential equation (ODE) $\begin{array} { r } { \frac { d \mathbf { x } _ { t } } { d t } = f _ { \theta } ( \mathbf { x } _ { t } , t ) } \end{array}$ from pure noise $\mathbf { x } _ { 1 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ backwards in time to $t = 0$ , yielding a sample from the learned data distribution.

Recent works (Yu et al., 2024; Leng et al., 2025a; Jiang et al., 2025) demonstrate that flow matching training benefits substantially from feature alignment. Given representations from a teacher model $r _ { \phi } ,$ these methods augment training with an auxiliary objective:

$$
\mathcal {L} _ {\mathrm{rep}} = - \mathbb {E} _ {\mathbf {x} _ {0}, \mathbf {x} _ {1}, t, t ^ {\prime}} \sin \left(h _ {\theta} ^ {(l)} (\mathbf {x} _ {t}, t), r _ {\phi} ^ {(k)} (\mathbf {x} _ {t ^ {\prime}}, t ^ {\prime})\right), \tag {3}
$$

where sim denotes a similarity metric, l, k denote layer indices of the flow model and teacher, respectively, and $h _ { \theta } ^ { ( l ) } ( { \bf x } _ { t } , t ) = h _ { \theta } ( f _ { \theta } ^ { ( l ) } ( { \bf x } _ { t } , t ) )$ is an MLP projection head. This loss aligns the features at the l-th layer of the flow model $f _ { \theta }$ with the representation features $r _ { \phi } ^ { ( k ) } ( \mathbf { x } _ { t ^ { \prime } } , t ^ { \prime } )$ from layer k. Alignment methods achieve optimal performance when aligning with an external pretrained encoder, with DINOv2-B (Oquab et al., 2024) being the predominant choice (Yu et al., 2024; Leng et al., 2025a; Singh et al., 2025). However, existing evaluations focus primarily on class-conditional ImageNet generation (Deng et al., 2009), a dataset heavily represented in DINOv2 training, potentially biasing the reported results. In this work, we demonstrate that reliance on external models leads to unexpected behavior across data distributions, model scales, and modalities.

![](images/96f48d885a12526b30d030439d84860c46dc04d83eacc809d87f726cc508a423.jpg)

<details>
<summary>line</summary>

| Training Steps | DINov2-B | DINov2-L | DINov3-B | DINov3-H+ |
| -------------- | -------- | -------- | -------- | --------- |
| 200K           | 10.5     | 10.8     | 14.5     | 18.0      |
| 300K           | 8.5      | 8.8      | 11.0     | 13.5      |
| 400K           | 7.5      | 7.8      | 9.5      | 11.5      |
| 500K           | 7.0      | 7.2      | 8.5      | 10.5      |
| 600K           | 6.5      | 7.0      | 8.0      | 9.5       |
</details>

(a) REPA scaling

![](images/7b0ac584e2bd30d6404a67bf5b410d7c9c94c2f0b771ab634d45ac4c9b9e0581.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | Diffusion Forcing | Full Masking | Dual-Timestep Scheduling |
| -------------- | --------------------- | ----------------- | ------------ | ------------------------ |
| 200K           | 25                    | 53                | 45           | 22                       |
| 300K           | 20                    | 42                | 40           | 18                       |
| 400K           | 17                    | 38                | 35           | 15                       |
| 500K           | 15                    | 38                | 30           | 13                       |
| 600K           | 14                    | 37                | 28           | 12                       |
</details>

(b) Noise scheduler comparison   
Figure 2. Motivation. (a) Scaling DINO $( \mathrm { v } 2 \mathrm { - } \mathrm { B } < \mathrm { v } 2 \mathrm { - } \mathrm { L } < \mathrm { v } 3 \cdot$ H+) paradoxically degrades the generations quality using REPA. (b) Diffusion forcing and full masking create a train-inference gap which degrades generations. Our Dual-Timestep Scheduling improves generation even without a self-supervised objective.

# 3.2. Motivation

We begin by presenting experiments that motivate our unified framework by testing the scaling laws of external alignment methods. Following the setup of REPA (Yu et al., 2024) (Eq. 3), we replace the DINOv2-B backbone with increasingly stronger variants (DINOv2-L, DINOv3- B, DINOv3-H+). Figure 2a reveals an inverse correlation: stronger representation learners consistently degrade generation quality. DINOv2-B, the smallest and weakest variant, achieves the best FID, while the most capable model, DINOv3-H+, performs the worst. This suggests that external alignment creates a bottleneck: the generative model becomes dependent on a fixed external representation that may not align with the generative goal. Instead of relying on fixed representations, we want to strengthen them within the generative framework itself.

# 3.3. Dual-Timestep Scheduling

In standard flow matching, uniform noise is applied to all tokens, resulting in a denoising task that can often be solved by local correlations alone. To encourage the learning of stronger, more global representations across the model, we introduce information asymmetry: by applying different noise levels to different tokens, the model is encouraged to use cleaner tokens to infer noisy tokens. The key challenge is how to introduce such heterogeneous noise without disrupting the underlying generative dynamics.

One intuitive strategy is to randomly set t = 1 for a subset of tokens, fully masking them. Another is to sample an independent noise level for each token, similar to diffusion forcing (Chen et al., 2024). In Fig. 2b, we compare these approaches with vanilla flow matching and our proposed scheduling method. Both naive masking and diffusion forcing substantially degrade the generation quality. We attribute this to a train–inference gap: during inference, the model must denoise uniformly noised inputs at both low and high noise levels, a regime that is rarely encountered during training.

![](images/0e9273e248a24439d3e75b56c5bda424915ffde6f2607852fbba8ad2aed2e308.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image: cat image"] --> B["t ~ p(t)"]
    C["Image: cat image"] --> D["s ~ p(t)"]
    B --> E["M ~"]
    D --> F["min(t, s)"]
    E --> G["x_τ"]
    F --> H["x_τmin"]
    G --> I["f_θ (student)"]
    H --> J["f_θ' (teacher)"]
    I --> K["L_gen"]
    J --> L["L_rep"]
    K --> M["L_gen"]
    L --> N["L_rep"]
    M --> O["EMA"]
    N --> P["sg"]
```
</details>

Figure 3. Illustration of our method. Given a clean input $x _ { 0 } ,$ we draw two timesteps $t , s ,$ and a random mask M, then noise each token according to its assigned timestep. The teacher input is noised with $\tau _ { m i n } = \operatorname* { m i n } \{ t , s \}$ , creating an information asymmetry compared to the student. The student is trained to simultaneously denoise the input and reconstruct the teacher’s features given its mixed-noised view.

To address this mismatch, we propose Dual-Timestep Scheduling. The core idea is to sample two timesteps from the noise distribution (Fig. 3). The higher of the two noises effectively corrupts information, while the cleaner one serves as context. Specifically, given an input $x _ { 0 }$ we:

1. Sample two timesteps: $t , s \sim p ( t )$   
2. Sample a mask $M = \left\{ i \in \{ 1 , \dots , N \} \mid u _ { i } < \mathcal { R } _ { M } \right\}$ , with $u _ { i } \overset { \underset { \mathrm { i i d } } { } } { \sim } \mathcal { U } ( 0 , 1 )$ and a masking ratio $\mathcal { R } _ { M } \leq 0 . 5$ .   
3. Construct a Dual-Timestep $\tau \in \mathbb { R } ^ { N }$ to noise x0:

$$
\tau^ {i} = \left\{ \begin{array}{l l} s & \text { if   } i \in M \\ t & \text { otherwise } \end{array} \right. \tag {4}
$$

$$
\mathbf {x} _ {\boldsymbol {\tau}} = \operatorname{diag} (\mathbf {1} - \boldsymbol {\tau}) \mathbf {x} _ {0} + \operatorname{diag} (\boldsymbol {\tau}) \mathbf {x} _ {1} \tag {5}
$$

This approach strikes a balance between vanilla homogeneous noising, which fails to encourage strong global relations, and the fully heterogeneous approach which fails to simulate inference behavior during training, while maintaining the marginal timestep distribution per token.

Interestingly, as observed in Fig. 2b, Dual-Timestep Scheduling alone, applied to the vanilla flow matching training, is able to slightly improve the generation quality even without an explicit self-supervised objective. Intuitively, this can be attributed to the presence of cleaner information in the input, which helps the model perform the denoising task, thus the model is implicitly encouraged to consider global relations, which in turn improves its generative capabilities.

# 3.4. Self-Flow

Next, we show how to leverage the information asymmetry created by Dual-Timestep Scheduling to encourage the model to learn stronger representations. As illustrated in Fig. 3, we maintain two models: a student network $f _ { \theta }$ that learns from heterogeneously noised inputs ${ \mathbf { x } } _ { \tau } ,$ , and an EMA teacher network $f _ { \theta ^ { \prime } }$ that has the advantage of observing the cleaner $\mathbf { x } _ { \tau _ { \mathrm { m i n } } }$ which is noised by $\tau _ { \operatorname* { m i n } } = \operatorname* { m i n } ( \tau ) \in \{ t , s \}$ . Based on this setup, we can now devise a feature alignment loss where the student learns to reconstruct the teacher’s features from its partial, corrupt view of the input. Formally, our representation alignment objective is given by using the teacher network $f _ { \theta ^ { \prime } }$ as the representation network and integrating the dual timestep $\tau$ into Eq. 3, using cosine similarity as the alignment metric:

$$
\mathcal {L} _ {\text { rep }} = - \mathbb {E} _ {\mathbf {x} _ {0}, \mathbf {x} _ {1}, \boldsymbol {\tau}} \cos \left(h _ {\theta} ^ {(l)} (\mathbf {x} _ {\boldsymbol {\tau}}, \boldsymbol {\tau}), f _ {\theta^ {\prime}} ^ {(k)} (\mathbf {x} _ {\tau_ {\min}}, \tau_ {\min})\right) \tag {6}
$$

Following the insights from Yu et al. (2024); Jiang et al. (2025) on the evolution of semantic features in diffusion models, we choose $l < k$ .

To perform this task, the student is encouraged to actively leverage the cleaner tokens to infer the representations for the noisier tokens, forming global connections that transcend simple locality. Our training objective combines generation and representation learning, parametrized by a scaling factor γ:

$$
\mathcal {L} = \mathcal {L} _ {\text { gen }} + \gamma \cdot \mathcal {L} _ {\text { rep }} \tag {7}
$$

Table 1. Quantitative results on ImageNet 256×256. Our method outperforms REPA, despite its use of DINOv2 which heavily relies on ImageNet for training. Our method is beneficial in combination with representation autoencoders. 

<table><tr><td>Model</td><td>Steps</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>Pre.↑</td><td>Rec.↑</td></tr><tr><td colspan="7">Without external representations</td></tr><tr><td>SiT-XL/2</td><td>7M</td><td>8.3</td><td>6.30</td><td>130.57</td><td>0.69</td><td>0.67</td></tr><tr><td>SRA</td><td>4M</td><td>7.27</td><td>5.87</td><td>143.06</td><td>0.69</td><td>0.68</td></tr><tr><td>Ours</td><td>4M</td><td>5.70</td><td>4.97</td><td>151.40</td><td>0.72</td><td>0.67</td></tr><tr><td colspan="7">With external representations</td></tr><tr><td>REPA</td><td>4M</td><td>5.89</td><td>5.73</td><td>157.66</td><td>0.70</td><td>0.69</td></tr><tr><td colspan="7">With representation autoencoders</td></tr><tr><td>RAE</td><td>1M</td><td>3.24</td><td>6.73</td><td>218.53</td><td>0.83</td><td>0.54</td></tr><tr><td>RAE + Ours</td><td>1M</td><td>2.95</td><td>5.50</td><td>222.34</td><td>0.84</td><td>0.56</td></tr></table>

![](images/b05bcffb6657570b3dff0d9b699a44b48685f067b1f3a8abedc995d93eb20d97.jpg)

<details>
<summary>line</summary>

| Training Steps | RAE (SiT-XL) | RAE + Ours (SiT-XL) |
| -------------- | ------------ | ------------------- |
| 200K           | 5.0          | 4.3                 |
| 400K           | 4.0          | 3.6                 |
| 600K           | 3.7          | 3.3                 |
| 800K           | 3.4          | 3.1                 |
| 1.0M           | 3.2          | 2.9                 |
</details>

(a) Our method w/ RAE

![](images/b3c8b5582c5391ebdf77d3e8e63d8e4bf0f2fb8b1b1faf9245bae7a1090c3686.jpg)

<details>
<summary>line</summary>

| Layer Index | Vanilla Flow Matching | Ours |
| ----------- | --------------------- | ---- |
| 4           | 10                    | 40   |
| 8           | 15                    | 55   |
| 12          | 20                    | 60   |
| 16          | 40                    | 60   |
| 20          | 55                    | 60   |
| 24          | 50                    | 50   |
| 28          | 38                    | 38   |
</details>

(b) Linear probing on ImageNet   
Figure 4. Autoencoder generalization and improved representations. (a) Our method improves training and generation in RAE (Zheng et al., 2025), demonstrating compatibility with semantic latent spaces. (b) Linear probing confirms that our method learns stronger representations than standard flow matching.

# 4. Experiments

We evaluate our method on class-to-image (ImageNet), textto-image (T2I), text-to-video (T2V), text-to-audio (T2A), and multi-modal generation. Through quantitative results, qualitative comparisons, and scaling experiments, we demonstrate the effectiveness of our approach, its adaptability across tasks and modalities, and its scaling properties.

Implementation Details. For ImageNet experiments, we use SiT-XL (Ma et al., 2024) with the REPA setup. All other experiments use the FLUX.2 (Black Forest Labs, 2025) transformer with domain-specific autoencoders. Unless noted, all models are ∼625M parameters. For text-to-image (T2I), we use the Stable Diffusion autoencoder (following the ImageNet setup) and train on 20M text-image pairs; for text-to-video (T2V), the Wan2.2 (Wan et al., 2025) autoencoder with 6M videos; for text-to-audio (T2A), the Songbloom autoencoder (Yang et al., 2025) with FMA (Defferrard et al., 2017). All evaluations are conducted on holdout sets of the corresponding training sets. See App. A for further training and sampling details.

# 4.1. Single Modality Experiments

Baselines. We compare against vanilla flow matching and the leading methods from both categories of representation alignment: with and without external models. For each category, we select the best performing approach applicable across all tested modalities: REPA (Yu et al., 2024) for external alignment and SRA (Jiang et al., 2025) for methods without external models (see App. A.5). To ensure a thorough comparison, we additionally evaluate external encoders that are, in theory, better suited than DINO to each task: SigLIP 2 (Tschannen et al., 2025) for text-to-image (trained with text supervision and multi-aspect ratio support), V-JEPA 2 (Bardes et al., 2024) and Depth Anything 3 (Lin et al., 2025) for video, and MERT (Li et al., 2024c) for audio.

Quantitative Results. On ImageNet (Tab. 1), our method outperforms REPA (FID 5.70 vs 5.89) without external representations and despite REPA using DINOv2, itself heavily trained on ImageNet. To our knowledge, we are the first to show self-supervised learning outperforming external alignment on ImageNet. To demonstrate generalization to arbitrary latent spaces, we apply our method to RAE (Zheng et al., 2025), in the same setup as the other experiments. As shown in Fig. 4a and Tab. 1, this yields significant improvements (FID 3.24 → 2.95). See App. G.2 for further details. Finally, Fig. 4b shows linear probing results after 2M training steps. Our method significantly boosts the representation quality of early and mid layers, confirming that representations improve alongside generations.

On the T2I task (Fig. 5a,b, Tab. 2), our method achieves the best FID (3.61) among all methods, outperforming both external alignment approaches (REPA: 3.92, SigLIP 2: 3.97) and methods without external models (SRA: 3.70). Notably, we outperform REPA even when computing the Frechet ´ distance score with DINOv2 features (Fig. 5b, 167.98 vs 173.35) despite REPA explicitly aligning with DINOv2 features, a gap no other baseline closes. Our method also achieves the highest CLIP score, indicating superior textimage alignment.

Our approach shows particularly strong gains on video generation (Fig. 5c, Tab. 3), achieving the best FVD (47.81) and FID (8.92, per frame) by a significant margin; the next best method (REPA) trails by nearly 2 FVD points. Notably, external alignment with video-specific V-JEPA2 (Bardes et al., 2024) and Depth Anything 3 (Lin et al., 2025) actually harms performance relative to vanilla flow matching. We hypothesize that temporal relations are harder to learn than spatial ones, making objective misalignments harder to bridge. Moreover, video temporal redundancies allow models to exploit shortcuts by copying across frames rather than learning meaningful semantics, a behavior our masking mechanism naturally discourages.

Table 2. Quantitative results on text-to-image generation. Our method outperforms all external and internal alignment methods. 

<table><tr><td>Model</td><td>Steps</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>Pre.↑</td><td>Rec.↑</td><td>FD-DINO↓</td><td>CLIP↑</td></tr><tr><td colspan="9">Without external representations</td></tr><tr><td>Vanilla Flow</td><td>1M</td><td>4.08</td><td>8.16</td><td>20.49</td><td>0.62</td><td> $\underline{0.64}$ </td><td>204.49</td><td>30.66</td></tr><tr><td>SRA</td><td>1M</td><td> $\underline{3.70}$ </td><td> $\underline{8.05}$ </td><td>21.00</td><td> $\underline{0.63}$ </td><td> $\underline{0.64}$ </td><td>176.79</td><td> $\underline{30.78}$ </td></tr><tr><td>Ours</td><td>1M</td><td> $\underline{3.61}$ </td><td>8.14</td><td>21.19</td><td> $\underline{0.64}$ </td><td> $\underline{0.65}$ </td><td>167.98</td><td> $\underline{30.88}$ </td></tr><tr><td colspan="9">With external representations</td></tr><tr><td>REPA</td><td>1M</td><td>3.92</td><td>8.20</td><td> $\underline{21.16}$ </td><td> $\underline{0.63}$ </td><td> $\underline{0.65}$ </td><td> $\underline{173.35}$ </td><td>30.67</td></tr><tr><td>SigLIP 2</td><td>1M</td><td>3.97</td><td> $\underline{8.13}$ </td><td>20.65</td><td> $\underline{0.63}$ </td><td> $\underline{0.64}$ </td><td>196.75</td><td>30.68</td></tr></table>

Table 3. Quantitative results on video generation. Most external models harm performance. 

<table><tr><td>Model</td><td>Steps</td><td>FVD↓</td><td>FID↓</td></tr><tr><td colspan="4">Without external representations</td></tr><tr><td>Vanilla Flow</td><td>600K</td><td>50.95</td><td>9.28</td></tr><tr><td>SRA</td><td>600K</td><td>49.75</td><td>9.02</td></tr><tr><td>Ours</td><td>600K</td><td>47.81</td><td>8.92</td></tr><tr><td colspan="4">With external representations</td></tr><tr><td>w/ DINOv2</td><td>600K</td><td>49.59</td><td>9.39</td></tr><tr><td>w/ Depth Anything 3</td><td>600K</td><td>51.52</td><td>9.85</td></tr><tr><td>w/ V-JEPA2</td><td>600K</td><td>53.55</td><td>9.91</td></tr></table>

![](images/9278b485c17814d69cfe0c15a2abb1590845ccfb4b99f9981724f22d71c5b431.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | REPA DINOv2 | REPA SigLIP 2 | SRA | Ours |
| -------------- | --------------------- | ----------- | ------------- | --- | ---- |
| 200K           | 5.35                  | 4.45        | 5.00          | 4.80 | 4.25 |
| 400K           | 4.60                  | 4.10        | 4.40          | 4.10 | 3.90 |
| 600K           | 4.35                  | 3.95        | 4.15          | 3.85 | 3.75 |
| 800K           | 4.15                  | 3.95        | 4.05          | 3.75 | 3.65 |
| 1.0M           | 4.10                  | 3.95        | 3.95          | 3.70 | 3.60 |
</details>

(a) T2I FID↓

![](images/bac7b4bcc9f7912cd0b532a400c034601782e8d4aefd017aa286a199e7f1b4b9.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | REPA DINOv2 | REPA SigLIP 2 | SRA | Ours |
| -------------- | --------------------- | ----------- | ------------- | --- | ---- |
| 200K           | 330                   | 245         | 310           | 300 | 255  |
| 400K           | 265                   | 205         | 245           | 230 | 205  |
| 600K           | 235                   | 185         | 225           | 200 | 185  |
| 800K           | 215                   | 180         | 200           | 185 | 175  |
| 1.0M           | 205                   | 175         | 195           | 175 | 170  |
</details>

(b) T2I FD-DINOv2↓

![](images/d3c453b8cd4d1bc1449380877431e8b45223ddc70122c7a7dcd0fd9e43da5830.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | REPA DINov2 | REPA Depth Anything 3 | REPA V-JEPA 2 | SRA | Ours |
| -------------- | --------------------- | ---------- | --------------------- | ------------- | --- | ---- |
| 200K           | 60.0                  | 57.5       | 61.5                  | 65.0          | 56.5 | 51.5 |
| 300K           | 55.0                  | 55.0       | 57.5                  | 58.0          | 52.5 | 50.5 |
| 400K           | 52.5                  | 52.5       | 55.0                  | 56.5          | 51.0 | 49.5 |
| 500K           | 51.0                  | 51.0       | 54.5                  | 55.0          | 50.5 | 49.0 |
| 600K           | 50.5                  | 50.0       | 51.5                  | 53.5          | 49.5 | 48.0 |
</details>

(c) Video FVD↓

![](images/5038b65165d96587189f67d52b9d43052e66b3304bd1cd495a2faac714acf6ad.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | REPA MERT | SRA   | Ours  |
| -------------- | --------------------- | --------- | ----- | ----- |
| 100K           | 158.0                 | 158.0     | 158.0 | 158.0 |
| 150K           | 152.0                 | 153.0     | 152.0 | 151.0 |
| 200K           | 150.0                 | 151.0     | 149.0 | 147.0 |
| 250K           | 149.0                 | 151.0     | 148.5 | 147.0 |
| 300K           | 149.0                 | 151.0     | 148.0 | 146.0 |
| 350K           | 149.0                 | 149.0     | 147.5 | 146.0 |
</details>

(d) Audio FAD↓   
Figure 5. Quantitative results across modalities. Our method significantly outperforms all external and internal alignment methods across text-based image, video, and audio generation. Our method is the only one to outperform REPA on DINOv2 FD (despite REPA directly aligning with DINOv2). Arrows indicate whether lower (↓) or higher (↑) is better.

Table 4. Quantitative results on audio generation. Our method achieves the best FAD scores across multiple CLAP variants. 

<table><tr><td>Model</td><td>Steps</td><td>CLAP↓</td><td>CLAP-M↓</td><td>CLAP-A↓</td></tr><tr><td colspan="5">Without external representations</td></tr><tr><td>Vanilla Flow</td><td>350K</td><td>148.874</td><td>0.1695</td><td>0.1059</td></tr><tr><td>SRA</td><td>350K</td><td> $\underline{147.215}$ </td><td> $\underline{0.1664}$ </td><td> $\underline{0.1034}$ </td></tr><tr><td>Ours</td><td>350K</td><td> $\underline{145.645}$ </td><td> $\underline{0.1634}$ </td><td> $\underline{0.1001}$ </td></tr><tr><td colspan="5">With external representations</td></tr><tr><td>w/ MERT</td><td>350K</td><td>148.883</td><td>0.1677</td><td>0.1040</td></tr></table>

We observe similar trends for audio (Fig. 5d, Tab. 4): our method achieves the best FAD scores across all CLAP variants, while external alignment with MERT provides no benefit over vanilla flow matching. This is further indication that external alignment struggles to generalize.

Scaling Behavior. We evaluate the scaling behavior of our method and REPA by training text-to-image models at four scales: 290M (deph=8), 420M (depth=14), 625M (depth=21), and 1B (depth=28). Fig. 6a shows that as we scale the model, the performance gap between our method and REPA widens consistently in our favor. Notably, our 625M parameter model outperforms the 1B REPA model, demonstrating the significant performance gains from our approach at scale. Fig. 6b demonstrates that our method exhibits consistent improvements with increased compute,

![](images/33a9130e2545718ddef456604a912ade32183a92bfca14a12a5b683041afdde8.jpg)

<details>
<summary>line</summary>

| Training Steps | depth=8 (Ours) | depth=8 (REPA) | depth=14 (Ours) | depth=14 (REPA) | depth=21 (Ours) | depth=21 (REPA) | depth=28 (Ours) | depth=28 (REPA) |
| -------------- | -------------- | -------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 200K           | 28.5           | 29.0           | 29.5            | 29.7            | 29.8            | 29.9            | 30.0            | 30.1            |
| 400K           | 29.5           | 30.0           | 30.2            | 30.4            | 30.5            | 30.6            | 30.7            | 30.8            |
| 600K           | 30.0           | 30.3           | 30.5            | 30.7            | 30.8            | 30.9            | 30.9            | 31.0            |
| 800K           | 30.2           | 30.5           | 30.7            | 30.9            | 31.0            | 31.1            | 31.1            | 31.2            |
| 1.0M           | 30.5           | 30.7           | 30.9            | 31.1            | 31.2            | 31.3            | 31.3            | 31.4            |
</details>

(a) Steps

![](images/8f317c7118df3246abd9151f54a130f967ae915b4f59b26426d3cec17e712572.jpg)

<details>
<summary>line</summary>

| Training FLOPs | depth=8 (Ours) | depth=8 (REPA) | depth=14 (Ours) | depth=14 (REPA) | depth=21 (Ours) | depth=21 (REPA) | depth=28 (Ours) | depth=28 (REPA) |
| -------------- | -------------- | -------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| 1e+20          | 29.5           | 29.5           | 29.5            | 29.5            | 29.5            | 29.5            | 29.5            | 29.5            |
| 1e+21          | 30.5           | 30.5           | 30.5            | 30.5            | 30.5            | 30.5            | 30.5            | 30.5            |
</details>

(b) FLOPs   
Figure 6. Scaling behavior. As model size increases (290M → 420M → 625M → 1B parameters), the performance gap between our method and REPA widens (a). Notably, our 625M variant outperforms the 1B REPA model. Our method effectively leverages increased compute, while REPA shows diminishing returns (b).

following expected scaling laws. These results validate our hypothesis that tying the model to a fixed external encoder creates a bottleneck that limits the benefits of scaling, whereas our unified framework scales as expected.

# 4.2. Multi-Modal Experiments

Having established the benefits of our approach on individual modalities, we now explore mixed and joint multi-modal setups. The former refers to a single model trained on multiple modalities, the latter to simultaneous generation of multi-modal outputs (e.g., video and corresponding audio or actions for robotic embodiments).

Vanilla Flow Matching   
![](images/b0179317e02edfafcec477d6f3e46ffc4e0dba7ad28a6046856321f2a8b0aab4.jpg)

<details>
<summary>natural_image</summary>

Interior view of a table with three cans and an orange object, no visible text or symbols
</details>

![](images/de6477b4286fc4a263f1b0082e988aa92e167db073d3d4232b56042dd0e4059a.jpg)

<details>
<summary>natural_image</summary>

Interior scene with a table, three cans of liquid and an orange object, and a robotic arm (no visible text or symbols)
</details>

![](images/4c406c9e6b0a1204bfd2307d3042154a3d5bba5df945d344cd598ccf08eb0545.jpg)

<details>
<summary>natural_image</summary>

Interior scene with a table holding three cans and an orange object, next to a robotic arm (no visible text or symbols)
</details>

![](images/4483156483b22483c19ed0b4a66f00506362b708279df2b9cf168c0cd747e9b5.jpg)

<details>
<summary>natural_image</summary>

Top-down view of a coffee table with two cans and one cup, no visible text or symbols
</details>

![](images/a2af0bacdf5bf85a0599ad718c6c78a1156a82b6f013e5f8434f02d318bcc6c8.jpg)

<details>
<summary>natural_image</summary>

Robot arm interacting with a beverage can and a small orange container on a table (no visible text or symbols)
</details>

![](images/cecaded22cee416279fa0e546c56de2625a7d10bec384715601fce814b78c796.jpg)

<details>
<summary>natural_image</summary>

Robot interacting with a beverage can on a table (no visible text or symbols)
</details>

Action: Move Near   
![](images/5febbde0ebea20bbe3a221bdc548b1d30399f1971c59fda303227238c3a3d12e.jpg)

<details>
<summary>natural_image</summary>

A person inserting a plastic cup into a three-tiered office chair (no visible text or symbols)
</details>

![](images/04f8ff29a54e174a045d76fd3966886e121f2fa6f55c9217a5abba8596e81312.jpg)

<details>
<summary>natural_image</summary>

A robotic hand pouring liquid into a gray office drawer with a small jar on top (no visible text or symbols)
</details>

![](images/e92f5f9202b0a9abbb158bf1ed484040f5a72529a30dcfbce88bed267b5ab0c0.jpg)

<details>
<summary>natural_image</summary>

Robot interacting with a small object on a gray drawer (no visible text or symbols)
</details>

![](images/4b3ccdfb8151fef0fc3490bd2781d83322d48c21b36ab60fc6edc2831dbfb9e1.jpg)

<details>
<summary>natural_image</summary>

Person in lab coat handling a small object on an open drawer, with other items visible inside (no text or symbols)
</details>

![](images/93f7132a425785576c778f189a27c0b4e6ac385351ee8e4d619d8d5c77361f34.jpg)

<details>
<summary>natural_image</summary>

A robotic arm interacting with a small object on an open shelf, no visible text or symbols.
</details>

![](images/52660de1b2e5262c8af53bc5ce063d7b23c20bdad41b8dea12445895009b1d78.jpg)

<details>
<summary>natural_image</summary>

Robot interacting with a small object inside a rectangular tray (no text or symbols visible)
</details>

Action: Place Apple in Closed Top Drawer   
Figure 7. SIMPLER simulation rollouts. Our method significantly boosts success rate for complex actions such as Move Near (top) and Open and Place (bottom).   
![](images/5a74f5b370d7d5fd3abf298670ab148d94f4c07fba3b831937025c67f58c113b.jpg)

<details>
<summary>radar</summary>

| Category         | audio weighted: FM | audio weighted: ours | image weighted: FM | image weighted: ours | video weighted: FM | video weighted: ours |
| ---------------- | ------------------ | -------------------- | ------------------ | -------------------- | ------------------ | -------------------- |
| Image FID        | 149.1              | 161.6                | 115.8              | 3.3                  | 115.8              | 58.0                 |
| Audio FAD        | 149.1              | 161.6                | 115.8              | 3.3                  | 115.8              | 58.0                 |
| Video FVD        | 149.1              | 161.6                | 115.8              | 3.3                  | 115.8              | 58.0                 |
</details>

(a) Multi-Modal Training

![](images/9eda1dbfc662fe3aebfdcb14111a83d197947dd9f0200e5636c1f03e43794cb0.jpg)

<details>
<summary>radar</summary>

| Category           | Ours @ 30K | Ours @ 100K | FM @ 30K | FM @ 100K |
| ------------------ | ---------- | ----------- | -------- | --------- |
| Move Near          | 50%        | 50%         | 50%      | 50%       |
| Open/Close Drawer | 78%        | 78%         | 78%      | 78%       |
| Open and Place     | 29%        | 29%         | 29%      | 29%       |
| Pick Up Cake Can   | 73%        | 73%         | 73%      | 73%       |
</details>

(b) Joint Video-Action Training   
Figure 8. Multi-modal experiments. (a) We train a single model on three modalities with different weightings to control the tradeoff between them. Ours provides consistent improvements (shaded area) across all settings. Axes are inverted so that larger area indicates better performance. (b) Success rates for joint Video-Action prediction. Early on (30k), Ours outperforms FM across all tasks and achieves success in all task categories, whereas FM fails entirely on Open and Place tasks. Later (100k), performance on single-object tasks (Pick Coke Can, Open/Close Drawer) converges, while Ours maintains a significant advantage on complex multi-object and sequential tasks (Move Near, Open and Place).

For the mixed-modality experiments, we follow the singlemodality setup, except that we use the FLUX.2 autoencoder for T2I for enhanced visual quality. To systematically test our method’s impact on each modality within this mixed setting, we employ modality-specific loss weightings $w = ( w _ { I } > 0 , w _ { V } > 0 , w _ { A } > 0 )$ , taking into consideration the number of samples observed for each modality (additional details in App. A). The overall loss at each step is a weighted-linear combination of each modality’s loss, parameterized by w. A robust representation learning framework should yield improvements across all selections of w, demonstrating the method’s ability to harmonize different representations under a single backbone. This requirement

is particularly challenging due to the different nature of the modalities—while audio representations are temporal and relatively low-dimensional, video data is high-dimensional and contains significant spatial and temporal redundancies. Fig. 8a shows results in a normalized radar chart with inverted axes, using the extreme weightings that favor each modality. This allows us to test the trade-off between different modalities. Our approach consistently improves performance across all three modalities simultaneously, even under extreme setups that favor a specific modality.

Next, we consider joint video-action prediction for embodied AI, where the model jointly predicts future video frames and robot actions from a conditioning image. We initialize from the video-weighted mixed-modality model and finetune on the RT-1 robotics dataset (Brohan et al., 2023) (73.5k episodes), evaluating on the SIMPLER simulator (Li et al., 2024a). We compare Self-Flow (Ours) and vanilla flow matching (FM) initializations, both finetuned under identical conditions. Fig. 8b reports the success rate by task group. Self-Flow consistently outperforms flow matching throughout finetuning, demonstrating more efficient learning from limited robotics data. Notably, while performance on single-object tasks (Pick, Open/Close) converges between methods, Self-Flow maintains a significant advantage on complex multi-object and sequential tasks (Move Near, Open and Place, see also Fig. 7), suggesting that our approach learns representations that improve complex visual reasoning. See App. E for details on this task and App. F for experiments on joint video-audio prediction.

# 4.3. Qualitative Results.

We provide qualitative comparisons in Figs. 9, 10, with additional results in the appendices (T2I: Figs. 22–26, 32– 34; T2V: Figs. 27–31) and on our supplementary website, which contains the full video results and audio comparisons. Fig. 9 presents text rendering results from our multi-modal

Vanilla Flow Matching   
![](images/c646b635ad33fd08ad2901600dfe7c07293d2597c60b2c5c0bc1846e6fe172de.jpg)

<details>
<summary>text_image</summary>

The future is
F Lx
</details>

![](images/217db3cff508e8c26679d5ae0b521b4579aaa067bdd8092c0a1d250725bfcc9c.jpg)

<details>
<summary>text_image</summary>

FUX
was
here
</details>

![](images/ec0889440e85c6ce8558c9c8d69208c0fb478f5952437ddd0324971a9c1a38d4.jpg)

<details>
<summary>text_image</summary>

From words to
worlds:
FLUX
</details>

![](images/38bf7d295a197b708a012b3aee29e35d47dd87d9bfe625c404d11a1000614370.jpg)

<details>
<summary>text_image</summary>

FII
</details>

Ours   
![](images/a39dc7902ecca2747dee76c5fa91594b547ddf489c146b7db72f14bd7cdce5bc.jpg)

<details>
<summary>text_image</summary>

The future is
FLUX
</details>

![](images/bc5cc1a953249e1f20265f57070346e352400d74b72fdad79b69a0d687f71624.jpg)

<details>
<summary>text_image</summary>

FLUX
was
here.
</details>

![](images/7774d68be7266b4a2ddd438b18a9920bb7de1101e9effdcf3a6bfd9c83faf683.jpg)

<details>
<summary>text_image</summary>

From words
to worlds:
FLUX
</details>

![](images/3faec8b2d7d4c74a38c78f98df495ce72056d5429613b665269635b10bc6e737.jpg)

<details>
<summary>text_image</summary>

FLUX
</details>

Vanilla Flow Matching   
![](images/c82f3f55c8d364312abfdcf6e86383369d02329317d57063f8b54bb294d3bd21.jpg)

<details>
<summary>text_image</summary>

Honk if ov love
FLUX
</details>

![](images/db5baed0ebbc8d84139ed04e6ae23cf8975756c1d3c160de0a61d85e3467c7f2.jpg)

<details>
<summary>text_image</summary>

Smalal,
big draams:
4B
Startsnp
</details>

![](images/7f557125236080cbe6155660a0e61121c2ee6a9498b7e4cfa94367febc974bd3.jpg)

<details>
<summary>text_image</summary>

FLUX
bring iedes to life
</details>

![](images/836720931210fdd29886a91a80cd195e0da6ade2c95be021a0ec30e2cbed1491.jpg)

<details>
<summary>text_image</summary>

Mdo
with with
FLUX
</details>

Ours   
![](images/070ede951502ae7aab2b39407373c0e53d0259d9d2a9971a4c6170911f745110.jpg)

<details>
<summary>text_image</summary>

Honk
if you love
FLUX
</details>

![](images/e3ba605513895416faa89c739ff938fa13afc09417abe0128d6984ba5e8d63da.jpg)

<details>
<summary>text_image</summary>

Small
model,
big dreams:
4B
</details>

![](images/fe510306265648d3b91c74717969bd2bf18c5728e8491d49b0c884e480910757.jpg)

<details>
<summary>text_image</summary>

FLUX
brings ideas to life
</details>

![](images/fba12921f99af087a2e441b51c373f5c548eb4b424ede6f768fe0f2a58c74223.jpg)

<details>
<summary>text_image</summary>

Made with
FLUX
</details>

Vanilla Flow Matching   
![](images/3a98db4e070f2830c37e9458210b3462f7c8f8b60bc68fb63e1c66a28a28904a.jpg)

<details>
<summary>text_image</summary>

FUX
is nutilimolal
</details>

![](images/331113ec3977946fdecf133846e12fc7f5b0898d44ed07e492b2246217f65554.jpg)

<details>
<summary>natural_image</summary>

White baby crib with floating letters spelling 'FLU', 'I', and 'P' (no text or symbols on the cabin itself)
</details>

![](images/ab5adc5c77b5e0b028601ee69fd8edbe831e54d150fb4d979bd29b46f7ad85c7.jpg)

<details>
<summary>text_image</summary>

4B full
INDEE
</details>

![](images/c9ad2560a874437f6523b94b8aac9fc4f33c4e43cdde8246382d8555b9094e4d.jpg)

<details>
<summary>text_image</summary>

Cocking
4B x with
100K
</details>

Ours   
![](images/fcb5c6a29a5b4fbbeee6dcbc6e364ea18abc686290a19920a3a7a3ebeb0cb7be.jpg)

<details>
<summary>text_image</summary>

FLUX
is multimodal
</details>

![](images/28835ac1c707a575ec05562645200f208afb4cb16b0f3f720fb0ea326cfa22d7.jpg)

<details>
<summary>natural_image</summary>

Cylindrical baby crib with colorful 'FLUX' letters hanging on a wall (no text or symbols beyond the lettering)
</details>

![](images/caf886d787b267f3aae6dbdb582a19eb2aeebf7714f725497d34c40e5c75cce3.jpg)

<details>
<summary>text_image</summary>

4B fuel
INSIDE
</details>

![](images/f6a19356cc1fb0baf483429b94336f859f5bfa2825a0d2b27e2034e54c613f2f.jpg)

<details>
<summary>text_image</summary>

Cooking
with
4B ×100K
</details>

Figure 9. Typography comparison. Self-Flow significantly outperforms standard flow matching in accurate text rendering.

Va nilla   
REPA   
SRA   
Ours   
![](images/c8f70457a26e48a6337e46ee00c4f074ea463ed204974dd07b339ec874722ebc.jpg)

![](images/d32096660c8e834d226809c2495461a57bdebb9fbde1ab96031768fe886beca7.jpg)

![](images/b1bb0103e362abbd076599e57f5b709ddf48faba19f7e956078f69b726033acb.jpg)

![](images/88de75310647cac47f6d95a67e05959d97645c188de527c3f5ec75e97f745c55.jpg)  
a blue Vespa scooter on a gravel path beside a body of water, tall grasses, realistic photograph.

![](images/31b083e25ec6b7342a5cd5848107297d40d32933a29d811ba4891c461aa0238f.jpg)

![](images/99fe584d9586368239071c26c4981cf3b258576c2a40ae0c4caa06013d1d928f.jpg)

![](images/d1a59173ea839b101d8b237552c1e737f068d2a026ac19b6ed154f571c9bd669.jpg)

![](images/21d3c4cf3a84fae4d54ec298b5baa1d307d12e501a1d868b213316e1f226b857.jpg)  
a close-up of a wheel and brake system, featuring a polished silver alloy wheel with a blue brake.

Va nilla   
REPA   
SRA   
Ours   
![](images/be1fdf7de1ffcd6692e5f4acab38740fff57b19ffb8a867c432dc4c99c0696b3.jpg)

![](images/c29064caccf991f69435067dd0503feac11fc370c8d148ba6c7f97fd9d71f9b3.jpg)

![](images/8dda40d9ce23b53c836353ff2192a41607851d0ce9b0861192dd679a4d31a1aa.jpg)

![](images/e0c4745b1531756d5ef9db507760b7b914de619090e06ad4470e4d9147fc967f.jpg)  
a portrait of a man with light brown hair and a beard, wearing a blue and red shirt.

![](images/199b3064f89aca98dd81f4e46dbf81c09a770fd98f0ac023ed0911cc5ff0515c.jpg)

![](images/5625779df6a65ad04aabed98a5119d60279bed501a3f10d9d8d56b71d8f74bf5.jpg)

![](images/5cdec6809f1eab93b3aef437af7d856d1f88e899305579c2ef8162c4f0efd056.jpg)

![](images/68571857283cc8790d6b917ec61f32d7ed80055bb1deb3482582ec45f0e6b1e0.jpg)  
close-up of a monkey with distinctive facial markings and thick fur, against a blurred green forest.

(a) Text-to-image. Our method produces superior structure, texture, and fine-grained details across prompts of varying complexity.   
Vanilla   
![](images/786f46f3edcad00c3c33dc7270c5e9eac2306e770ed6d7146afe48c168428f20.jpg)

![](images/b9d83bb9225a3b0e3aac313ca55d9322e9969fb8458251bd9e49ae7b903874ee.jpg)

![](images/7e5b6d0de273c2e094078598a521393333a3b1891176b74ecc362cf5c64c51d6.jpg)

![](images/7997c666b98e15b37f59e5c2240e65ea4145bba17b2c622c3632a1a6185c852d.jpg)

SRA   
![](images/adc6c990f6c8b7d9e3cfba14c8cc973e6de8ff95fdac6270e7244ca92209a4d2.jpg)

![](images/509d3c35a9b65de277414031a04bb0a0d3a53b870c101595efc839241fcff6ca.jpg)

![](images/6e5b8778b75b26b36f8125e221e4433d07b55f490de62086f85ca3b4517049db.jpg)

![](images/6b480b0e9f179924d88f16b7f70b63137b619f7b4be23f0fbd4ff13c1f65a377.jpg)  
bicycle roll onto a wet concrete path, creating a dynamic splash as it passes through a large puddle. The lower half of a person's legs, clad in blue jeans, pedal the green bicycle.

![](images/7c69bb4ac39613757ca2724f9cb3795066c6c137a7cfadf50410bce80c2887c4.jpg)

![](images/84cc88bf5c19090e8fa6e6a9098774b9d08b7c9cf83dd9d2c407143cae586b6c.jpg)

![](images/a90fbc3e078f0cc7f56e577c6d55d9a6d34f44e4308ec6a373b402955d204cfc.jpg)

![](images/3393c7c5a611c0dd2ff384dacca7ddf9c230f26b947caffd86ac67eacb8391d2.jpg)

![](images/4b94b524d177b5633a36d2fd272475c156be1bde1b482a9eb1117cbad6834848.jpg)

![](images/919345b8e3e58bd09b3e04e955ff876378f80289f4cfe54e441a6330a62bc42f.jpg)

![](images/bfb83387f66735889c56a4a2e1afb360e0dcf78a8e9c11c64eccd6aee23ea2df.jpg)

![](images/88def8adf34273a9c33b3cabf226bcfe9addf917088887f6655a8f907879e1e0.jpg)  
a first-person camera view looking down at vibrant green and black skis, launches over a snow-covered drop revealing a vast, sunlit mountain range.

![](images/5aa9f087580a71511e1f0292694c0445c5c412268988d894505407134ae1c5c6.jpg)

![](images/06449f1c307f4c3c3a512e5092df61560d27827af140fcf12b96bbf59481152b.jpg)

![](images/6846ae1c84b41db38b3c0eb0375a6163d7e18e98dfada25aed08523747bd93d2.jpg)

![](images/a4f53d5898197158c2122efe986935c27c0aa5ad40a0851e51b0b9fa83d7021a.jpg)

![](images/ca31e01305bab4ec4d214a752b4bbed73969c7504d980555acf50c707738cfaf.jpg)

![](images/f1084f8964f14de3de47fb3d1aee4aa3417806992405f9ac396204756072778a.jpg)

![](images/f05ae2ff293e0bb384a459bbc20fda4b6d373d3a47c98f3ee953fb4cf71af7ec.jpg)

![](images/ff18c7f2ad452acf4fd12518f741c4a22d47948521c0068ecce8978b7ed88b9b.jpg)  
a young woman with brown hair, a white cap, denim short overalls, and white sneakers, dances in a hip-hop style, behind her a painted brick wall with a swirling yellow and green spiral.

(b) Text-to-video. Baselines exhibit structural and temporal artifacts, while our method produces coherent and temporally smooth results.   
Figure 10. Qualitative comparisons with vanilla flow matching and leading baselines: external (REPA) and internal (SRA).   
![](images/57bc39bedd951983dc03d72eb2e9e895c901344c16006b7166ea5b67d4eddf14.jpg)

<details>
<summary>line</summary>

| Training Steps | Ours  | w/o masking | x ∈ [t, t - 0.2] | w/ ℓ₁ loss | w/o ℒ_rep |
| -------------- | ----- | ----------- | ---------------- | ---------- | --------- |
| 200K           | 16.5  | 19.0        | 18.5             | 16.0       | 22.0      |
| 300K           | 12.5  | 14.0        | 13.5             | 12.0       | 17.5      |
| 400K           | 10.5  | 11.5        | 11.0             | 10.0       | 15.0      |
| 500K           | 9.5   | 10.0        | 9.5              | 9.0        | 13.5      |
| 600K           | 8.5   | 9.0         | 8.5              | 8.5        | 12.5      |
</details>

(a) Ablation study

![](images/d44a4c3c956103e22d1ab655b42dc39b532853ddffc3085754c10e58a47b322a.jpg)

<details>
<summary>line</summary>

| Training Steps | REPA (logit normal) | REPA (uniform) | Ours (logit normal) | Ours (uniform) |
| -------------- | ------------------- | -------------- | ------------------- | -------------- |
| 200K           | 6.5                 | 4.4            | 6.5                 | 4.3            |
| 400K           | 5.3                 | 4.1            | 5.2                 | 3.9            |
| 600K           | 4.9                 | 4.0            | 4.8                 | 3.8            |
| 800K           | 4.7                 | 4.0            | 4.6                 | 3.7            |
| 1.0M           | 4.5                 | 3.9            | 4.5                 | 3.6            |
</details>

(b) Impact of noise scheduler   
Figure 11. Ablations and Limitations. (a) Removing the selfsupervised loss is most detrimental. Removing Dual-Timestep Scheduling or changing the scheduling of the second timestep (s) significantly harms the results. (b) While better noise scheduling impacts both methods positively, our method benefits from it more.

4B parameter model. Self-Flow significantly improves text rendering over flow matching. The comparisons in Fig. 10 show that our method consistently produces superior visual fidelity, prompt adherence, and temporal coherence over all baselines. For images, the Vespa and portrait (1st row) examples highlight improved structural accuracy and fine details

compared to baselines. For video, the baselines exhibit significant structural and temporal artifacts. For example, the dancing woman (3rd row) displays limbs that spontaneously disappear. Conversely, our method maintains both spatial and temporal coherence across all samples. Notably, these video results are achieved with a modest model (∼625M parameters) trained on only 6M samples, demonstrating the effectiveness of our approach in low-resource settings.

# 4.4. Ablation Study

We ablate key components of our method on the ImageNet class-to-image task (Fig. 11a). Removing the representation loss (Eq. 6) results in the most significant degradation, of over 4 points, confirming that encouraging semantic feature learning is critical for generation quality. Removing the masking mechanism while retaining the representation loss also leads to substantial degradation of over 1 point, reinforcing the need for an explicit self-supervised paradigm beyond simple cross-layer feature alignment. Constraining the second timestep to be only slightly cleaner than the base timestep $( s \in [ t , t - 0 . 2 ] )$ results in degradation nearly equivalent to removing masking entirely, indicating that the formulation of masking matters: our strategy, which samples both timesteps from the full noise distribution, preserves the marginal noise distribution per token and strikes an effective balance between the generation and representation objectives. Finally, replacing the cosine similarity objective with an $\ell _ { 1 }$ loss leads to numerical instabilities as training progresses due to increasing feature norms, resulting in an increase in FID at later training steps.

# 5. Limitations and Future Work

This work challenges a common assumption: that generative models require external, domain-specific encoders to improve representations and generation quality. As we show, external alignment can exhibit unexpected scaling behavior and often struggles to generalize across modalities. For example, REPA degrades audio generation compared to vanilla flow matching (Sec. 4). Instead, we address representation deficiency at its source by unifying generation and representation learning within a single framework. This approach has trade-offs: the additional forward pass through the teacher increases training overhead, but the accelerated convergence and improved performance justify this cost (Fig. 6). Consistent with related work (Esser et al., 2024; Zheng et al., 2025), the noise scheduler p(t) requires tuning as it determines masking behavior. Fig. 11b demonstrates this on text-to-image generation in the same setup described in Sec. 4.1, where a uniform scheduler outperforms a logitnormal scheduler with a shift of α = 1.78 (App. A.2). While the better noise scheduling choice benefits both REPA and Self-Flow, the gap increases significantly in favor of the latter, which can be attributed to a more optimal timestep selection for the masking mechanism. In practice, we observe that the optimal scheduler for flow matching works well, and further tuning will likely yield additional gains.

Looking ahead, by bridging representation learning and generative modeling, our approach offers a path toward world models that harness the scalability and perceptual grounding of visual generative models without sacrificing the semantic abstraction required for planning and understanding - a direction we began to validate in Sec. 4.2. We hope this work stimulates research into consolidating generative and representation learning: two directions long pursued in isolation that may complement each other better than assumed.

# Acknowledgments

We thank the Black Forest Labs team for the codebase, architectures, and infrastructure that made this work possible. In particular, we thank Rinon Gal and Sumith Kulal for providing feedback on the manuscript and Nihanth Subramanya and Cyril Diagne for their help with the project website.

# References

Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants, 2023. URL https://arxiv.org/abs/2209.15571.   
Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619– 15629, 2023.   
Bao, H., Dong, L., and Wei, F. Beit: Bert pre-training of image transformers. ArXiv, abs/2106.08254, 2021.   
Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., and Ballas, N. V-JEPA: Latent video prediction for visual representation learning, 2024.   
Black, K., Nakamoto, M., Atreya, P., Walke, H., Finn, C., Kumar, A., and Levine, S. Zero-shot robotic manipulation with pretrained image-editing diffusion models, 2023. URL https://arxiv.org/abs/2310.10639.   
Black Forest Labs. FLUX.2: Analyzing and enhancing the latent space of FLUX – representation comparison, 2025. URL https://bfl.ai/research/ representation-comparison.   
Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jackson, T., Jesmonth, S., Joshi, N. J., Julian, R., Kalashnikov, D., Kuang, Y., Leal, I., Lee, K.-H., Levine, S., Lu, Y., Malla, U., Manjunath, D., Mordatch, I., Nachum, O., Parada, C., Peralta, J., Perez, E., Pertsch, K., Quiambao, J., Rao, K., Ryoo, M., Salazar, G., Sanketi, P., Sayed, K., Singh, J., Sontakke, S., Stone, A., Tan, C., Tran, H., Vanhoucke, V., Vega, S., Vuong, Q., Xia, F., Xiao, T., Xu, P., Xu, S., Yu, T., and Zitkovich, B. Rt-1: Robotics transformer for real-world control at scale, 2023. URL https:// arxiv.org/abs/2212.06817.   
Caron, M., Touvron, H., Misra, I., J’egou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9630–9640, 2021.   
Cen, J., Yu, C., Yuan, H., Jiang, Y., Huang, S., Guo, J., Li, X., Song, Y., Luo, H., Wang, F., Zhao, D., and Chen, H. Worldvla: Towards autoregressive action world model, 2025. URL https://arxiv.org/ abs/2506.21539.   
Cheang, C.-L., Chen, G., Jing, Y., Kong, T., Li, H., Li, Y., Liu, Y., Wu, H., Xu, J., Yang, Y., Zhang, H., and Zhu,

M. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation, 2024. URL https://arxiv.org/abs/2410.06158.   
Chen, B., Monso, D. M., Du, Y., Simchowitz, M., Tedrake, ´ R., and Sitzmann, V. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=yDo1ynArjj.   
Chen, C., Qian, R., Hu, W., Fu, T.-J., Tong, J., Wang, X., Li, L., Zhang, B., Schwing, A., Liu, W., and Yang, Y. Dit-air: Revisiting the efficiency of diffusion model architecture design in text to image generation, 2025a. URL https: //arxiv.org/abs/2503.10618.   
Chen, H., Han, Y., Chen, F., Li, X., Wang, Y., Wang, J., Wang, Z., Liu, Z., Zou, D., and Raj, B. Masked autoencoders are effective tokenizers for diffusion models. ArXiv, abs/2502.03444, 2025b.   
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning, pp. 1597–1607. PMLR, 2020a.   
Chen, X., Fan, H., Girshick, R. B., and He, K. Improved baselines with momentum contrastive learning. ArXiv, abs/2003.04297, 2020b.   
Chen, X., Xie, S., and He, K. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9640–9649, 2021.   
Collaboration, E., O’Neill, A., Rehman, A., Gupta, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., Jain, A., Tung, A., Bewley, A., Herzog, A., Irpan, A., Khazatsky, A., Rai, A., Gupta, A., Wang, A., Kolobov, A., Singh, A., Garg, A., Kembhavi, A., Xie, A., Brohan, A., Raffin, A., Sharma, A., Yavary, A., Jain, A., Balakrishna, A., Wahid, A., Burgess-Limerick, B., Kim, B., Scholkopf, B., Wulfe, B., Ichter, ¨ B., Lu, C., Xu, C., Le, C., Finn, C., Wang, C., Xu, C., Chi, C., Huang, C., Chan, C., Agia, C., Pan, C., Fu, C., Devin, C., Xu, D., Morton, D., Driess, D., Chen, D., Pathak, D., Shah, D., Buchler, D., Jayaraman, D., Kalash-¨ nikov, D., Sadigh, D., Johns, E., Foster, E., Liu, F., Ceola, F., Xia, F., Zhao, F., Frujeri, F. V., Stulp, F., Zhou, G., Sukhatme, G. S., Salhotra, G., Yan, G., Feng, G., Schiavi, G., Berseth, G., Kahn, G., Yang, G., Wang, G., Su, H., Fang, H.-S., Shi, H., Bao, H., Amor, H. B., Christensen, H. I., Furuta, H., Bharadhwaj, H., Walke, H., Fang, H., Ha, H., Mordatch, I., Radosavovic, I., Leal, I., Liang, J., Abou-Chakra, J., Kim, J., Drake, J., Peters, J., Schneider, J., Hsu, J., Vakil, J., Bohg, J., Bingham, J., Wu, J., Gao,

J., Hu, J., Wu, J., Wu, J., Sun, J., Luo, J., Gu, J., Tan, J., Oh, J., Wu, J., Lu, J., Yang, J., Malik, J., Silverio, J., ´ Hejna, J., Booher, J., Tompson, J., Yang, J., Salvador, J., Lim, J. J., Han, J., Wang, K., Rao, K., Pertsch, K., Hausman, K., Go, K., Gopalakrishnan, K., Goldberg, K., Byrne, K., Oslund, K., Kawaharazuka, K., Black, K., Lin, K., Zhang, K., Ehsani, K., Lekkala, K., Ellis, K., Rana, K., Srinivasan, K., Fang, K., Singh, K. P., Zeng, K.-H., Hatch, K., Hsu, K., Itti, L., Chen, L. Y., Pinto, L., Fei-Fei, L., Tan, L., Fan, L. J., Ott, L., Lee, L., Weihs, L., Chen, M., Lepert, M., Memmel, M., Tomizuka, M., Itkina, M., Castro, M. G., Spero, M., Du, M., Ahn, M., Yip, M. C., Zhang, M., Ding, M., Heo, M., Srirama, M. K., Sharma, M., Kim, M. J., Irshad, M. Z., Kanazawa, N., Hansen, N., Heess, N., Joshi, N. J., Suenderhauf, N., Liu, N., Palo, N. D., Shafiullah, N. M. M., Mees, O., Kroemer, O., Bastani, O., Sanketi, P. R., Miller, P. T., Yin, P., Wohlhart, P., Xu, P., Fagan, P. D., Mitrano, P., Sermanet, P., Abbeel, P., Sundaresan, P., Chen, Q., Vuong, Q., Rafailov, R., Tian, R., Doshi, R., Mart´ın-Mart´ın, R., Baijal, R., Scalise, R., Hendrix, R., Lin, R., Qian, R., Zhang, R., Mendonca, R., Shah, R., Hoque, R., Julian, R., Bustamante, S., Kirmani, S., Levine, S., Lin, S., Moore, S., Bahl, S., Dass, S., Sonawani, S., Tulsiani, S., Song, S., Xu, S., Haldar, S., Karamcheti, S., Adebola, S., Guist, S., Nasiriany, S., Schaal, S., Welker, S., Tian, S., Ramamoorthy, S., Dasari, S., Belkhale, S., Park, S., Nair, S., Mirchandani, S., Osa, T., Gupta, T., Harada, T., Matsushima, T., Xiao, T., Kollar, T., Yu, T., Ding, T., Davchev, T., Zhao, T. Z., Armstrong, T., Darrell, T., Chung, T., Jain, V., Kumar, V., Vanhoucke, V., Guizilini, V., Zhan, W., Zhou, W., Burgard, W., Chen, X., Chen, X., Wang, X., Zhu, X., Geng, X., Liu, X., Liangwei, X., Li, X., Pang, Y., Lu, Y., Ma, Y. J., Kim, Y., Chebotar, Y., Zhou, Y., Zhu, Y., Wu, Y., Xu, Y., Wang, Y., Bisk, Y., Dou, Y., Cho, Y., Lee, Y., Cui, Y., Cao, Y., Wu, Y.-H., Tang, Y., Zhu, Y., Zhang, Y., Jiang, Y., Li, Y., Li, Y., Iwasawa, Y., Matsuo, Y., Ma, Z., Xu, Z., Cui, Z. J., Zhang, Z., Fu, Z., and Lin, Z. Open x-embodiment: Robotic learning datasets and rt-x models, 2025. URL https://arxiv.org/abs/2310.08864.

Defferrard, M., Benzi, K., Vandergheynst, P., and Bresson, X. FMA: A dataset for music analysis. In 18th International Society for Music Information Retrieval Conference (ISMIR), 2017. URL https://arxiv.org/ abs/1612.01840.

Defferrard, M., Mohanty, S. P., Carroll, S. F., and Salathe,´ M. Learning to recognize musical genre from audio. In The 2018 Web Conference Companion. ACM Press, 2018. ISBN 9781450356404. doi: 10.1145/3184558.3192310. URL https://arxiv.org/abs/1803.05337.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image

database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.   
Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.   
Du, Y., Yang, M., Dai, B., Dai, H., Nachum, O., Tenenbaum, J. B., Schuurmans, D., and Abbeel, P. Learning universal policies via text-guided video generation, 2023. URL https://arxiv.org/abs/2302.00111.   
Elizalde, B., Deshmukh, S., Ismail, M. A., and Wang, H. Clap: Learning audio concepts from natural language supervision, 2022. URL https://arxiv.org/abs/ 2206.04769.   
Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, J., ¨ Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.   
Evans, Z., Parker, J. D., Carr, C., Zukowski, Z., Taylor, J., and Pons, J. Stable audio open. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2025.   
Falck, F., Pandeva, T., Zahirnia, K., Lawrence, R., Turner, R., Meeds, E., Zazo, J., and Karmalkar, S. A fourier space perspective on diffusion models, 2025. URL https: //arxiv.org/abs/2505.11278.   
Gao, S., Zhou, P., Cheng, M.-M., and Yan, S. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 23164–23173, 2023.   
Ge, S., Mahapatra, A., Parmar, G., Zhu, J.-Y., and Huang, J.-B. On the content bias in frechet video distance. In ´ Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7277–7288, 2024.   
Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P., ´ Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., Piot, B., kavukcuoglu, k., Munos, R., and Valko, M. Bootstrap your own latent - a new approach to self-supervised learning. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 21271–21284. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/file/ f3ada80d5c4ee70142b17b8192b2958e-Paper. pdf.

Gui, A., Gamper, H., Braun, S., and Emmanouilidou, D. Adapting frechet audio distance for generative music evaluation. In Proc. IEEE ICASSP 2024, 2024. URL https://arxiv.org/abs/2311.01616.   
Haghighi, Y., van Delft, B., Hassan, M., and Alahi, A. Layersync: Self-aligning intermediate layers, 2025. URL https://arxiv.org/abs/2510.12581.   
He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. B. Momentum contrast for unsupervised visual representation learning. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9726–9735, 2019.   
He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.   
Hoogeboom, E., Heek, J., and Salimans, T. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, 2023.   
Hu, Y., Guo, Y., Wang, P., Chen, X., Wang, Y.-J., Zhang, J., Sreenath, K., Lu, C., and Chen, J. Video prediction policy: A generalist robot policy with predictive visual representations, 2025. URL https://arxiv.org/ abs/2412.14803.   
Jiang, D., Wang, M., Li, L., Zhang, L., Wang, H., Wei, W., Dai, G., Zhang, Y., and Wang, J. No other representation component is needed: Diffusion transformers can provide representation guidance by themselves. arXiv preprint arXiv:2505.02831, 2025.   
Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35: 26565–26577, 2022.   
Kilgour, K., Zuluaga, M., Roblek, D., and Sharifi, M. Frechet audio distance: A metric for evaluating music en- ´ hancement algorithms, 2019. URL https://arxiv. org/abs/1812.08466.   
Kingma, D. P. and Gao, R. Understanding diffusion objectives as the elbo with simple data augmentation, 2023. URL https://arxiv.org/abs/2303.00848.   
Labs, B. F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.   
Leng, X., Singh, J., Hou, Y., Xing, Z., Xie, S., and Zheng, L. Repa-e: Unlocking vae for end-to-end tuning with latent

diffusion transformers. arXiv preprint arXiv:2504.10483, 2025a.   
Leng, X., Singh, J., Murdock, R., Smith, E., Li, R., Hou, Y., Xing, Z., Xie, S., and Zheng, L. Family of Endto-End Tuned VAEs for Supercharging T2I Diffusion Transformers. https://end2end-diffusion. github.io/repa-e-t2i/, 2025b.   
Li, S., Gao, Y., Sadigh, D., and Song, S. Unified video action model, 2025. URL https://arxiv.org/abs/ 2503.00200.   
Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., Levine, S., Wu, J., Finn, C., Su, H., Vuong, Q., and Xiao, T. Evaluating real-world robot manipulation policies in simulation, 2024a. URL https://arxiv.org/abs/ 2405.05941.   
Li, Y., Gui, A., Emmanouilidou, D., and Gamper, H. Rethinking emotion bias in music via frechet audio distance. arXiv preprint arXiv:2409.15545, 2024b.   
Li, Y., Yuan, R., Zhang, G., Ma, Y., Chen, X., Yin, H., Xiao, C., Lin, C., Ragni, A., Benetos, E., Gyenge, N., Dannenberg, R., Liu, R., Chen, W., Xia, G., Shi, Y., Huang, W., Wang, Z., Guo, Y., and Fu, J. Mert: Acoustic music understanding model with large-scale self-supervised training, 2024c. URL https://arxiv.org/abs/ 2306.00107.   
Liang, J., Tokmakov, P., Liu, R., Sudhakar, S., Shah, P., Ambrus, R., and Vondrick, C. Video generators are robot policies, 2025. URL https://arxiv.org/abs/ 2508.00795.   
Lin, H., Chen, S., Liew, J., Chen, D. Y., Li, Z., Shi, G., Feng, J., and Kang, B. Depth anything 3: Recovering the visual space from any views, 2025.   
Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations.   
Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. ArXiv, abs/2209.03003, 2022.   
Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., Vanden-Eijnden, E., and Xie, S. Sit: Exploring flow and diffusionbased generative models with scalable interpolant transformers. In European Conference on Computer Vision, pp. 23–40. Springer, 2024.   
Morales-Brotons, D., Vogels, T., and Hendrikx, H. Exponential moving average of weights in deep learning:

Dynamics and benefits, 2024. URL https://arxiv. org/abs/2411.18704.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal, pp. 1–31, 2024.   
Pai, J., Achenbach, L., Montesinos, V., Forrai, B., Mees, O., and Nava, E. mimic-video: Video-action models for generalizable robot control beyond vlas, 2025. URL https://arxiv.org/abs/2512.15692.   
Pathak, D., Krahenb ¨ uhl, P., Donahue, J., Darrell, T., and ¨ Efros, A. A. Context encoders: Feature learning by inpainting. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2536–2544, 2016.   
Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.   
Pernias, P., Rampas, D., Richter, M. L., Pal, C., and Aubreville, M. Wurstchen: An efficient architecture for large- ¨ scale text-to-image diffusion models. In The Twelfth International Conference on Learning Representations, 2023.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022a.   
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models, 2022b. URL https://arxiv. org/abs/2112.10752.   
Shazeer, N. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.   
Shen, Y., Wei, F., Du, Z., Liang, Y., Lu, Y., Yang, J., Zheng, N., and Guo, B. Videovla: Video generators can be generalizable robot manipulators, 2025. URL https: //arxiv.org/abs/2512.06963.   
Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, ´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang,

J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jegou, H., Labatut, P., and Bojanowski, P. DINOv3, 2025.´ URL https://arxiv.org/abs/2508.10104.   
Singh, J., Leng, X., Wu, Z., Zheng, L., Zhang, R., Shechtman, E., and Xie, S. What matters for representation alignment: Global information or spatial structure? 2025.   
Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.   
Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1–9, 2015.   
Tian, Y., Yang, S., Zeng, J., Wang, P., Lin, D., Dong, H., and Pang, J. Predictive inverse dynamics models are scalable learners for robotic manipulation, 2024. URL https://arxiv.org/abs/2412.15109.   
Tschannen, M., Gritsenko, A., Wang, X., Naeem, M. F., Alabdulmohsin, I. M., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., H’enaff, O., Harmsen, J., Steiner, A., and Zhai, X.-Q. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. ArXiv, abs/2502.14786, 2025.   
Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., and Gelly, S. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.   
Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., and Manzagol, P.-A. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. J. Mach. Learn. Res., 11:3371–3408, 2010.   
Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.-F., and Liu, Z. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, L., Huang, B., Zhao, Z., Tong, Z., He, Y., Wang, Y., Wang, Y., and Qiao, Y. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14549–14560, 2023.   
Wang, R. and He, K. Diffuse and disperse: Image generation with representation regularization. arXiv preprint arXiv:2506.09027, 2025.   
Wu, G., Zhang, S., Shi, R., Gao, S., Chen, Z., Wang, L., Chen, Z., Gao, H., Tang, Y., jian Yang, Cheng, M.-M., and Li, X. Representation entanglement for generation: Training diffusion transformers is much easier than you think. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https: //openreview.net/forum?id=koEALFNBj1.   
Wu, H., Jing, Y., Cheang, C., Chen, G., Xu, J., Li, X., Liu, M., Li, H., and Kong, T. Unleashing large-scale video generative pre-training for visual robot manipulation, 2023. URL https://arxiv.org/abs/2312. 13139.   
Wu, H., Wu, D., He, T., Guo, J., Ye, Y., Duan, Y., and Bian, J. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling. ArXiv, abs/2507.07982, 2025b.   
Wu, Y., Chen, K., Zhang, T., Hui, Y., Nezhurina, M., Berg-Kirkpatrick, T., and Dubnov, S. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation, 2024. URL https://arxiv.org/abs/2211.06687.   
Yang, C., Wang, S., Chen, H., Tan, W., Yu, J., and Li, H. Songbloom: Coherent song generation via interleaved autoregressive sketching and diffusion refinement. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=Fa0kehLK6s.   
Yao, J. and Wang, X. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025.   
Ye, S., Ge, Y., Zheng, K., Gao, S., Yu, S., Kurian, G., Indupuru, S., Tan, Y. L., Zhu, C., Xiang, J., Malik, A., Lee, K., Liang, W., Ranawaka, N., Gu, J., Xu, Y., Wang, G., Hu, F., Narayan, A., Bjorck, J., Wang, J., Kim, G., Niu, D., Zheng, R., Xie, Y., Wu, J., Wang, Q., Julian, R., Xu, D., Du, Y., Chebotar, Y., Reed, S., Kautz, J., Zhu, Y., Fan, L. J., and Jang, J. World action models are zero-shot policies, 2026. URL https://arxiv. org/abs/2602.15922.

Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., and Xie, S. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.   
Zeng, W. and Yan, Y. Flow matching in the low-noise regime: Pathologies and a contrastive remedy, 2025. URL https://arxiv.org/abs/2509.20952.   
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11975–11986, October 2023.   
Zhang, X., Liao, J., Zhang, S., Meng, F., Wan, X., Yan, J., and Cheng, Y. Videorepa: Learning physics for video generation through relational alignment with foundation models. arXiv preprint arXiv:2505.23656, 2025.   
Zheng, B., Ma, N., Tong, S., and Xie, S. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025. URL https:// arxiv.org/abs/2510.11690.   
Zheng, H., Nie, W., Vahdat, A., and Anandkumar, A. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023.   
Zhu, R., Pan, Y., Li, Y., Yao, T., Sun, Z., Mei, T., and Chen, C. W. Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8435–8445, 2024.

# A. Implementation Details

# A.1. Datasets

We use five different research datasets for our experiments. For comparisons with existing work, we use the ImageNet-1K dataset (Deng et al., 2009) comprising 1.28M training images and 50k validation images across 1k categories.

For image experiments, we use an internal research dataset of 200M images, each with four different captions of varying granularity. 50k images are reserved for validation. The text-to-image experiments use a curated subset of 20M images, while multi-modal experiments use the full dataset. We randomly choose among the available captions during training and evaluate on the most detailed captions.

Video experiments use an internal research dataset of 6M videos with a focus on motion. Each video comes with three different visual-only captions of varying granularity, one audio-only caption and one audio-visual caption. 5k videos are reserved for validation. We randomly choose among the available captions during training, and evaluate on medium-length, visual-only captions. In the joint audio-video modeling task, we use the audio-visual captions.

For audio experiments, we use 1M audio samples with a length of 10 seconds, obtained from CC-licensed FMA data (Defferrard et al., 2017; 2018). Each sample comes with a single caption. 20k samples are reserved for validation, without overlapping tracks between training and validation splits.

# A.2. Autoencoders and Timestep Distributions

![](images/3365bf0dc51113319b5982a1eea8a86a593b8eff1b14fc5c98d9eba6db1da647.jpg)

<details>
<summary>line</summary>

| t    | s(1.0, ) | s(1.78, ) | s(2.95, ) | s(4.63, ) | s(6.93, ) |
| ---- | -------- | --------- | --------- | --------- | --------- |
| 0.0  | 0.0      | 0.0       | 0.0       | 0.0       | 0.0       |
| 0.2  | 0.2      | 0.3       | 0.4       | 0.5       | 0.6       |
| 0.4  | 0.4      | 0.5       | 0.6       | 0.7       | 0.8       |
| 0.6  | 0.6      | 0.7       | 0.8       | 0.9       | 0.95      |
| 0.8  | 0.8      | 0.9       | 0.95      | 0.98      | 0.99      |
| 1.0  | 1.0      | 1.0       | 1.0       | 1.0       | 1.0       |
</details>

(a) Timeshift function s(α, ·).

![](images/c2779f50a46d95eba1224e0b28c05019e6faa051329e026d3ed87c6d79ca9ce7.jpg)

<details>
<summary>line</summary>

| t    | p_ts(1.0) | p_ts(1.78) | p_ts(2.95) | p_ts(4.63) | p_ts(6.93) |
| ---- | --------- | ---------- | ---------- | ---------- | ---------- |
| 0.0  | 1.0       | 0.5        | 0.3        | 0.2        | 0.1        |
| 0.2  | 1.0       | 0.6        | 0.4        | 0.3        | 0.2        |
| 0.4  | 1.0       | 0.7        | 0.5        | 0.4        | 0.3        |
| 0.6  | 1.0       | 0.8        | 0.6        | 0.5        | 0.4        |
| 0.8  | 1.0       | 1.0        | 0.8        | 0.7        | 0.6        |
| 1.0  | 1.0       | 1.5        | 1.5        | 2.5        | 7.0        |
</details>

(b) Shifted uniform distributions pts.

![](images/2db328c258e5e733e42e14669807e3450a1c3a199c65d59e7d5b7472ef2ce4a9.jpg)

<details>
<summary>line</summary>

| t    | p_lnt: 1.00 | p_lnt: 1.78 | p_lnt: 2.953 | p_lnt: 4.633 | p_lnt: 6.933 |
| ---- | ----------- | ----------- | ------------ | ------------ | ------------ |
| 0.0  | 0.0         | 0.0         | 0.0          | 0.0          | 0.0          |
| 0.2  | 0.5         | 0.3         | 0.2          | 0.1          | 0.1          |
| 0.4  | 1.0         | 0.8         | 0.5          | 0.3          | 0.2          |
| 0.6  | 1.5         | 1.2         | 1.0          | 0.8          | 0.5          |
| 0.8  | 1.2         | 1.8         | 2.5          | 2.0          | 1.5          |
| 1.0  | 0.5         | 1.0         | 2.0          | 3.5          | 5.0          |
</details>

(c) Logit-Normal distributions pln

![](images/c0903980f14e019cd8e018ea7d7653d60e7f0169abd567caef57cedad39da531.jpg)

<details>
<summary>line</summary>

| t    | p_min(1.0) | p_min(1.78) | p_min(2.95) | p_min(4.63) | p_min(6.93) |
| ---- | ---------- | ----------- | ----------- | ----------- | ----------- |
| 0.0  | 0.0        | 0.0         | 0.0         | 0.0         | 0.0         |
| 0.2  | 0.5        | 0.3         | 0.2         | 0.1         | 0.05        |
| 0.4  | 1.0        | 0.8         | 0.5         | 0.3         | 0.1         |
| 0.6  | 1.2        | 1.2         | 1.0         | 0.8         | 0.5         |
| 0.8  | 1.2        | 1.5         | 1.8         | 2.0         | 1.5         |
| 1.0  | 1.2        | 1.5         | 2.2         | 3.2         | 4.5         |
</details>

(d) Plateau-Logit-Normal ppln   
Figure 12. Shifting function and distributions used for sampling timesteps during training and shifting grids for evaluations.

Our setup generally follows the latent diffusion approach (Rombach et al., 2022b), where data is encoded through modalityspecific autoencoders, and a flow-based approach (Liu et al., 2022; Albergo & Vanden-Eijnden, 2023; Lipman et al.) is used for generative modeling in this latent space. As highlighted in previous works (Kingma & Gao, 2023; Karras et al., 2022; Zeng & Yan, 2025; Hoogeboom et al., 2023; Esser et al., 2024; Zheng et al., 2025; Falck et al., 2025), the performance can be sensitive to the choice of the timestep sampling distribution used in Eq. (2).

We use the timeshift function s(α, ·) (Esser et al., 2024; Black Forest Labs, 2025) with shifting parameter α,

$$
s (\alpha , \cdot): t \mapsto \frac {\alpha t}{1 + (\alpha - 1) t}, \tag {8}
$$

to consider various choices for sampling timesteps during training. In this context, we also refer to α as the trainshift. Similarly, when shifting timesteps from a uniform grid with the timeshift function for evaluating models, we refer to α as the sampleshift.

When sampling timesteps from a uniform distribution followed by a shift with s, we obtain a shifted uniform distribution pts,

$$
p _ {\mathrm{ts}} (t; \alpha) = \frac {\alpha}{(\alpha + (1 - \alpha) t) ^ {2}} \tag {9}
$$

Note that the selection of evaluation timesteps also corresponds to stratified sampling from the shifted uniform distribution.

When we consider the logit-normal distribution (Esser et al., 2024),

$$
p _ {\ln} (t; \mu , \sigma) = \frac {1}{\sigma \sqrt {2 \pi}} \frac {1}{t (1 - t)} \exp \left(- \frac {(\operatorname{logit} (t) - \mu) ^ {2}}{2 \sigma^ {2}}\right) \tag {10}
$$

with $\begin{array} { r } { \log \mathrm { i t } ( t ) = \log \frac { t } { 1 - t } } \end{array}$ , its samples are transformed by $s ( \alpha , \cdot )$ to samples from a logit-normal distribution with shifted parameter $\mu ^ { \prime } = \mu + \log { \alpha }$ (Black Forest Labs, 2025). Finally, the plateau-logit-normal distribution is obtained by keeping the probability density function of the logit-normal distribution constant after its mode. See Fig. 12 for a visualization.

For the ImageNet experiments and T2I experiments, we use the ${ \mathrm { S D - V A E } } ^ { 1 }$ with a uniform distribution over training timesteps to maintain comparability to previous works. To validate the applicability of our approach across different autoencoders (see Sec. G.2), we also ran experiments on RAE (Zheng et al., 2025). While Zheng et al. (2025) used a uniform distribution with shift $\alpha = 6 . 9 3$ , Black Forest Labs (2025) reported additional benefits when switching from the uniform distribution to a plateau-logit-normal distribution. In addition, we found that further increasing the shifting factor to $\alpha = 1 0 . 0$ in combination with the plateau-logit-normal distribution provides further gains.

For images in the multi-modal experiments, we use the FLUX.2 AE (Black Forest Labs, 2025) and follow their choice of a logit-normal distribution with a trainshift of $\alpha = 4 . 6 3$ . For FLUX.2 AE and SD-VAE we employ $\mathbf { a \ 2 \times 2 }$ patching resulting in a total side-length compression factor of 16 which is consistent with RAE. In this setup, images of size $2 5 6 \times 2 5 6$ are encoded to 256 tokens of dimensionality 16, 128 and 768 for SD-VAE, FLUX.2 AE and RAE, respectively.

For video experiments, we use the WAN2.2 AE (Wan et al., 2025). It uses the same spatial compression as the image autoencoders, and compresses $T$ frames into $\begin{array} { r } { \lfloor 1 + \frac { T - 1 } { 4 } \rfloor } \end{array}$ spatial latents. We train on 45 frames at a resolution of 192p, resulting in sequence lengths around 3k with dimensionality 48. To determine a suitable training timestep distribution, we train Vanilla Flow Models and search over various shift parameters α in combination with a logit-normal distribution. Fig. 15 shows the results from which we determined $\alpha = 2 . 9 5$ as a suitable value. In addition, in Sec. C, we explore mixing a logit-normal distribution with a uniform distribution on high noise levels, to counteract potential issues arising from the logit-normal distribution’s property of vanishing density at high noise levels.

Audio experiments use the Songbloom AE from Yang et al. (2025), which mostly follows the design from Evans et al. (2025). It produces 25 latents per second of audio, resulting in 250 latents of dimensionality 64 when training on our 10 second long audio samples, or 48 latents when training on the audio track of videos. We run the text-to-audio experiments over a range of different shift values to determine good shifting values. See Sec. B.

# A.3. Architecture

When employing Dual-Timestep Scheduling, we extend the timestep conditioning of the model from a single scalar $t \in \mathbb { R } ^ { 1 }$ to a vector of timesteps $t \in \mathbb { R } ^ { N }$ such that each token in the sequence is conditioned on its corresponding noising timestep. To maintain comparability with previous approaches, all our ImageNet experiments utilize a SiT-XL (Ma et al., 2024) backbone comprising around 675M parameters (the exact number depending on the dimensionality of the data representation). For all other experiments, we base the design on the FLUX architecture (Labs et al., 2025), together with changes in FLUX.22, including the use of shared modulation layers (Chen et al., 2025a) and SwiGLU (Shazeer, 2020). We adapt the configuration to roughly match the parameter count of SiT-XL (∼625M parameters). Specifically, we use a hidden size of 1152, mlp ratio 4, num heads 16, 7 double MMBlocks and 14 single Blocks. We include bias parameters on qkv layers, and use 3D RoPE (Su et al., 2024) with 24 channels per dimension. For our approach and REPA variants, we follow (Yu et al., 2024) and use lightweight projection layers that add around 10M parameters. We keep track of a copy of Exponential Moving Average (EMA) (Morales-Brotons et al., 2024) weights with a decay factor of 0.9999. This copy is used as the teacher and for evaluations. We use a fixed coefficient $\gamma = 0 . 8$ and fixed ratios for layer selection: $\ell _ { \theta } = 0 . 3 D , \ell _ { \theta ^ { \prime } } = 0 . 7 D$ , where D denotes the depth of the model. Noise schedulers $p ( t )$ are identical across all baselines and are selected prior to training for each task. The ratio of second timestep sampling is $\mathcal { R } _ { M } = 0 . 2 5$ for image, $\mathcal { R } _ { M } = 0 . 5$ for audio, and $\mathcal { R } _ { M } = 0 . 1$ for video, due to significant temporal redundancies in video data. The qualitative samples are all obtained with 50 inference steps, using a classifier-free guidance scale of 3.5 for image generation, and 5 for video and audio generation. All quantitative metrics are reported without classifier-free guidance to ensure an unbiased comparison. The number of training steps for each modality is calibrated with respect to the data size. See Sec. H for an ablation study on these layers.

# A.4. Evaluation

On ImageNet, we evaluate SD-VAE based models using 250 SDE sampling steps as in (Ma et al., 2024). For RAE based models, we use 50 steps with a sampling timeshift of 6.93 as in (Zheng et al., 2025). In both cases, we sample 50k classes randomly, without class-balanced sampling, which (Zheng et al., 2025) reports to consistently reduce FID scores by 0.1. Scores are computed using the evaluation code3 and reference batch from (Dhariwal & Nichol, 2021).

For other experiments, we similarly compute Frechet distance between multivariate Gaussian distributions estimated from ´ modality-specific representations obtained from validation data, and samples produced by a model. For video, we follow (Ge et al., 2024), and compute FVD (Unterthiner et al., 2018) scores using VideoMAEv2 (Wang et al., 2023) features. In addition, we include FID (framewise) scores computed using Inception (Szegedy et al., 2015) features. For audio data, we follow (Gui et al., 2024; Li et al., 2024b) and compute FAD (Kilgour et al., 2019) using the CLAP model from (Elizalde et al., 2022) (CLAP) and the music (CLAP-M) and audio (CLAP-A) variants from (Wu et al., 2024).

For non-ImageNet experiments, we sample with 50 ODE steps and a sampling shift adapted to the autoencoder. Unless stated otherwise, we use shifts 1.78 for SD-VAE, 6.93 for FLUX.2, 15.0 for WAN2.2 and 6.93 for the Songbloom AE.

# A.5. Baseline Selection

We compare against vanilla flow matching and the leading methods from both categories of representation alignment: with and without external models. For methods that use external encoders, we compare against REPA (Yu et al., 2024), the de facto feature alignment method applicable across modalities. REPA implements a generic principle: aligning intermediate representations with the hidden states of a pretrained external encoder. This allows us to plug in a domain-appropriate encoder for each of our experiments, making REPA a flexible choice across the experiments in the main paper.

Among the methods that do not employ an external encoder, we consider both SRA (Jiang et al., 2025) and Layer-Sync (Haghighi et al., 2025), which to our knowledge are the only published methods achieving representation alignment without external encoders or external supervision. We perform a preliminary experiment to select the leading baseline on a subset of 6M samples from our text-to-image research dataset (Fig. 13), where all methods share the same architecture (a 625M parameter FLUX.2 (Labs et al., 2025) backbone trained over the SD-VAE latent space), which is identical to the setup used in the main paper. We find that SRA outperforms LayerSync after 400K steps. We hypothesize that this is because LayerSync does not employ an EMA teacher or apply noise shifts to the teacher signal. Both choices weaken the distilled signal from the teacher over training and cause the trend reversal witnessed in our experiments. Therefore, we opt to use SRA for our main paper experiments. Importantly, unlike both SRA and LayerSync, our Dual-Timestep Scheduling mechanism formulates an explicit self-supervised objective directly within the flow matching framework, naturally encouraging the model to develop strong semantic representations alongside the generation capabilities, as is reflected by the results in Fig. 13 and in the main paper.

![](images/91ad0a853b914e5990498e85a3aa1f3bfe7b1b4a44050645ab5620180e56a6d0.jpg)

<details>
<summary>line</summary>

| Training Steps | LayerSync | SRA   | Ours  |
| -------------- | --------- | ----- | ----- |
| 200K           | 4.1       | 4.5   | 3.95  |
| 300K           | 3.85      | 3.95  | 3.7   |
| 400K           | 3.7       | 3.75  | 3.6   |
| 500K           | 3.65      | 3.6   | 3.5   |
| 600K           | 3.6       | 3.55  | 3.45  |
</details>

Figure 13. Text-to-image results for baseline selection. We compare SRA and LayerSync to select the leading baseline that does not employ an external encoder. We find that SRA outperforms LayerSync after 400K training steps, and attribute this to SRA’s use of an EMA teacher.

# B. Additional Details on Audio Experiments

For the audio experiments in Sec. 4.1, we run all approaches with a logit-normal training timestep distribution under shifts $\alpha \in \{ 0 . 7 5 , 1 . 0 , 1 . 7 8 \}$ and sampling shifts {4.62, 6.93}. For our approach, we include different masking ratios $\mathcal { R } _ { M } \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 5 , 0 . 5 \}$ . To determine the best set of hyperparameters, we compute their rankings within each approach and then choose the ones with a minimal median rank across FAD (CLAP), FAD (CLAP-M) and FAD (CLAP-A). The resulting hyperparameters together with their results are shown in Tab. 5. We observe that our approach favors a slightly higher training shift of 1.0 compared to 0.75 favored by the other approaches. A possible explanation could be that our dual-timestep noising shifts overall SNR ratios towards the mean, thus requiring slightly higher shifts to maintain sufficient coverage of low SNR regimes. Fig. 14 contains results from all hyperparameter runs, colored by approach. It shows that our approach not only outperforms baselines in the optimal hyperparameter setting, but instead compares favorably across a wide range of hyperparameters.

Table 5. Quantitative results on audio generation together with hyperparameters obtained as described in Sec. B. 

<table><tr><td>Model</td><td>Trainshift</td><td>Sampleshift</td><td>Masking Ratio</td><td>CLAP↓</td><td>CLAP-M↓</td><td>CLAP-A↓</td></tr><tr><td>Vanilla Flow Matching</td><td>0.75</td><td>6.93</td><td>-</td><td>148.874</td><td>0.1695</td><td>0.1059</td></tr><tr><td>SRA</td><td>0.75</td><td>6.93</td><td>-</td><td>147.215</td><td>0.1664</td><td>0.1034</td></tr><tr><td>Ours</td><td>1.0</td><td>6.93</td><td>0.5</td><td>145.645</td><td>0.1634</td><td>0.1001</td></tr><tr><td>REPA MERT</td><td>0.75</td><td>6.93</td><td>-</td><td>148.883</td><td>0.1677</td><td>0.1040</td></tr></table>

![](images/cf6d24d9ccaddd91fa7bb32ba2af2eda501b421876c4130147aa1ac5d9419057.jpg)

<details>
<summary>line</summary>

| Training Steps | REPA MERT | Vanilla Flow Matching | SRA | Ours |
| -------------- | --------- | --------------------- | --- | ---- |
| 100K           | 160       | 158                   | 157 | 159  |
| 150K           | 155       | 153                   | 152 | 154  |
| 200K           | 152       | 150                   | 149 | 148  |
| 250K           | 150       | 148                   | 147 | 146  |
| 300K           | 149       | 147                   | 146 | 145  |
| 350K           | 148       | 146                   | 145 | 144  |
</details>

(a) FAD (CLAP)↓

![](images/6401da4cbfb170f62490f4bec2ddbd02391a661cc213576930af5c0bfa7396a3.jpg)

<details>
<summary>line</summary>

| Training Steps | REPA MERT | Vanilla Flow Matching | SRA | Ours |
| -------------- | --------- | --------------------- | --- | ---- |
| 100K           | 0.178     | 0.179                 | 0.177 | 0.176 |
| 150K           | 0.172     | 0.174                 | 0.173 | 0.170 |
| 200K           | 0.169     | 0.171                 | 0.169 | 0.165 |
| 250K           | 0.168     | 0.170                 | 0.168 | 0.164 |
| 300K           | 0.168     | 0.169                 | 0.167 | 0.164 |
| 350K           | 0.168     | 0.169                 | 0.167 | 0.164 |
</details>

(b) FAD (CLAP-M)↓

![](images/3e0d8cd9b379d3cc2d0218e2afa132d60e5c15686373636db8b0c2260ea80a5f.jpg)

<details>
<summary>line</summary>

| Training Steps | REPA MERT | Vanilla Flow Matching | SRA | Ours |
| -------------- | --------- | --------------------- | --- | ---- |
| 100K           | 0.1125    | 0.1125                | 0.1125 | 0.1125 |
| 150K           | 0.1075    | 0.1100                | 0.1075 | 0.1075 |
| 200K           | 0.1050    | 0.1075                | 0.1050 | 0.1050 |
| 250K           | 0.1045    | 0.1075                | 0.1045 | 0.1045 |
| 300K           | 0.1045    | 0.1075                | 0.1045 | 0.1045 |
| 350K           | 0.1045    | 0.1075                | 0.1045 | 0.1045 |
</details>

(c) FAD (CLAP-A)↓   
Figure 14. Results with early-stopping from all hyperparameter variants considered in the audio experiments. The best hyperparameter runs, as determined by the selection process described in Sec. B, are shown in bold. Our approach compares favorably against all other variants across hyperparameters.

# C. Additional Details on Video Experiments

To determine the training shift parameter under a logit-normal distribution for the WAN2.2 AE, we train a Vanilla Flow Matching model with training shifts $\alpha \in \{ 1 . 0 , 1 . 7 8 , 2 . 9 5 , 4 . 6 2 , 6 . 9 3 \}$ . Based on the results shown in Fig. 15, we choose $\alpha = 2 . 9 5$ as the base training shift parameter on WAN2.2, which obtains the best FVD and framewise FID scores among the considered shifts.

To avoid potential issues arising from insufficient training in low SNR regimes (cf. Sec. B), we explore a low SNR modification, where for 5% of all sampled training timesteps, we sample from a uniform distribution on the high noise interval [0.95, 1.00]. We show results with and without this modification in Fig. 15. The performance of SRA does not change significantly under this modification, whereas we see consistent improvements for our approach and REPA DA3. REPA DINOv2 benefits in FVD but not FID and REPA V-JEPA 2’s FID performance even decreases with this change. The most significant gain is observed for our FID performance, whereas our FVD performance remains best under both settings.

![](images/70ef96907a4dd97371dd995f37807627eb3bd09a8938bc99114a70ed1f0ac2fb.jpg)

<details>
<summary>scatter</summary>

| Trainshift | FVD   | step     |
| ---------- | ----- | -------- |
| 1          | 565   | 200000   |
| 1          | 525   | 400000   |
| 1          | 518   | 600000   |
| 2          | 490   | 200000   |
| 2          | 470   | 400000   |
| 2          | 460   | 600000   |
| 3          | 462   | 200000   |
| 3          | 448   | 400000   |
| 3          | 442   | 600000   |
| 5          | 468   | 200000   |
| 5          | 452   | 400000   |
| 5          | 448   | 600000   |
| 7          | 488   | 200000   |
| 7          | 472   | 400000   |
| 7          | 462   | 600000   |
</details>

(a) FVD↓

![](images/8494c96f856fb4b2d7fe193359843558d7c9cc8ec0525cf9c71a10aef5dcfbde.jpg)

<details>
<summary>scatter</summary>

| Trainshift | FID (framework) | step     |
| ---------- | --------------- | -------- |
| 1          | 118             | 200000   |
| 1          | 108             | 400000   |
| 1          | 108             | 600000   |
| 2          | 107             | 200000   |
| 2          | 101             | 400000   |
| 2          | 98              | 600000   |
| 3          | 102             | 200000   |
| 3          | 97              | 400000   |
| 3          | 95              | 600000   |
| 4          | 104             | 200000   |
| 4          | 99              | 400000   |
| 4          | 98              | 600000   |
| 5          | 104             | 200000   |
| 5          | 99              | 400000   |
| 5          | 98              | 600000   |
| 6          | 107             | 200000   |
| 6          | 102             | 400000   |
| 6          | 101             | 600000   |
| 7          | 117             | 200000   |
| 7          | 102             | 400000   |
| 7          | 101             | 600000   |
</details>

(b) FID (framewise)↓

![](images/cbb46ea1e6795ed08776e6ccb52f17c787296f29a98679dd80dd8798e5227f69.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | Vanilla Flow Matching (w/o low SNR) | REPA DINOv2 | REPA DINOv2 (w/o low SNR) | REPA Depth Anything 3 | REPA Depth Anything 3 (w/o low SNR) | VJEPA 2 | VJEPA 2 (w/o low SNR) | SRA | SRA (w/o low SNR) | Ours (w/o low SNR) |
| -------------- | ---------------------- | ------------------------------------ | ---------- | -------------------------- | --------------------- | ----------------------------------- | ------- | --------------------- | --- | ----------------- | ------------------ |
| 200K           | 65.0                   | 60.0                                 | 62.5       | 58.0                       | 61.0                  | 57.0                                | 60.0    | 59.0                  | 57.0| 56.0              | 54.0               |
| 300K           | 62.5                   | 57.5                                 | 60.0       | 55.5                       | 57.5                  | 54.5                                | 57.5    | 56.5                  | 54.5| 53.5              | 51.5               |
| 400K           | 60.0                   | 55.0                                 | 57.5       | 53.0                       | 56.0                  | 52.5                                | 56.0    | 54.0                  | 52.5| 51.5              | 49.5               |
| 500K           | 57.5                   | 52.5                                 | 55.0       | 50.5                       | 54.5                  | 49.5                                | 54.5    | 51.5                  | 49.5| 48.5              | 47.5               |
| 600K           | 55.0                   | 50.0                                 | 52.5       | 47.5                       | 53.0                  | 46.0                                | 53.0    | 49.0                  | 47.0| 46.0              | 46.0               |
</details>

(c) FVD↓

![](images/c5d5ebac4b17aa89edaf71020717c467fc4b659f1d1d146df14cb3f3c6632004.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | Vanilla Flow Matching (w/o low SNR) | REPA DINOx2 | REPA DINOx2 (w/o low SNR) | REPA Depth Anything 3 | REPA Depth Anything 3 (w/o low SNR) | REPA VJEP2 2 | REPA VJEP2 2 (w/o low SNR) | SRA | SRA (w/o low SNR) | Ours (w/o low SNR) | Ours |
| -------------- | ---------------------- | ------------------------------------ | ----------- | -------------------------- | --------------------- | ----------------------------------- | ------------ | --------------------------- | --- | ----------------- | ------------------ | ---- |
| 200K           | 11.0                   | 11.0                                 | 10.5        | 10.5                       | 10.5                  | 10.5                              | 10.5         | 10.5                        | 10.0 | 10.0              | 9.8                | 9.5  |
| 300K           | 10.5                   | 10.5                                 | 10.0        | 10.0                       | 10.0                  | 10.0                              | 10.0         | 10.0                        | 9.8  | 9.8               | 9.5                | 9.3  |
| 400K           | 10.0                   | 10.0                                 | 9.8         | 9.8                        | 9.8                   | 9.8                               | 9.8          | 9.8                         | 9.5  | 9.5               | 9.3                | 9.1  |
| 500K           | 9.8                    | 9.8                                  | 9.6         | 9.6                        | 9.6                   | 9.6                               | 9.6          | 9.6                         | 9.3  | 9.3               | 9.1                | 8.9  |
| 600K           | 9.6                    | 9.6                                  | 9.4         | 9.4                        | 9.4                   | 9.4                               | 9.4          | 9.4                         | 9.1  | 9.1               | 8.9                | 8.7  |
</details>

(d) FID (framewise)↓   
Figure 15. Experiments on training timestep behaviors for video training. Left two figures: Performance across different shift parameters in combination with a logit-normal distribution on a Vanilla Flow Matching model trained on WAN2.2 representations. Right two figures: Performance comparison of different approaches with and without the low SNR modification described in Sec. C.

Table 6. Comparison between performance of Ours and Vanilla Flow Matching (FM) in multimodal runs. For each metric, we show the values achieved by FM and Ours as well as change of Ours relative to FM. All metrics improve across all weightings. 

<table><tr><td rowspan="2">Weights $(w_I, w_V, w_A)$ </td><td colspan="3">FID↓</td><td colspan="3">FID (framewise)↓</td><td colspan="3">FVD↓</td><td colspan="3">FAD (CLAP)↓</td><td colspan="3">FAD (CLAP-A)↓</td><td colspan="3">FAD (CLAP-M)↓</td></tr><tr><td>FM</td><td>Ours</td><td>Rel. Chg.</td><td>FM</td><td>Ours</td><td>Rel. Chg.</td><td>FM</td><td>Ours</td><td>Rel. Chg.</td><td>FM</td><td>Ours</td><td>Rel. Chg.</td><td>FM</td><td>Ours</td><td>Rel. Chg.</td><td>FM</td><td>Ours</td><td>Rel. Chg.</td></tr><tr><td>(0.067, 0.319, 0.614)</td><td>4.04</td><td>3.69</td><td>-8.55%</td><td>12.2</td><td>11.1</td><td>-9.59%</td><td>72.9</td><td>66.3</td><td>-9.09%</td><td>153</td><td>149.8</td><td>-2.13%</td><td>0.115</td><td>0.114</td><td>-0.47%</td><td>0.178</td><td>0.177</td><td>-0.22%</td></tr><tr><td>(0.105, 0.506, 0.389)</td><td>3.51</td><td>3.25</td><td>-7.34%</td><td>10.9</td><td>10.2</td><td>-6.60%</td><td>64.8</td><td>61</td><td>-5.85%</td><td>155.4</td><td>154.2</td><td>-0.75%</td><td>0.12</td><td>0.119</td><td>-0.74%</td><td>0.183</td><td>0.182</td><td>-0.35%</td></tr><tr><td>(0.131, 0.628, 0.241)</td><td>3.35</td><td>3.19</td><td>-4.60%</td><td>10.3</td><td>9.82</td><td>-5.04%</td><td>62.7</td><td>58</td><td>-7.40%</td><td>160.1</td><td>157.4</td><td>-1.66%</td><td>0.127</td><td>0.127</td><td>-0.57%</td><td>0.19</td><td>0.189</td><td>-0.38%</td></tr><tr><td>(0.196, 0.079, 0.725)</td><td>3.07</td><td>2.99</td><td>-2.85%</td><td>18.1</td><td>17.7</td><td>-2.01%</td><td>116</td><td>112</td><td>-3.72%</td><td>153.9</td><td>149.1</td><td>-3.08%</td><td>0.115</td><td>0.112</td><td>-3.12%</td><td>0.178</td><td>0.176</td><td>-1.44%</td></tr><tr><td>(0.308, 0.123, 0.569)</td><td>2.83</td><td>2.74</td><td>-3.06%</td><td>15.6</td><td>14.8</td><td>-5.10%</td><td>101</td><td>96.7</td><td>-4.33%</td><td>156.4</td><td>152.6</td><td>-2.40%</td><td>0.119</td><td>0.117</td><td>-1.86%</td><td>0.182</td><td>0.181</td><td>-0.74%</td></tr></table>

# D. Aditional Details on Multi-Modal Experiments

For the experiments of Sec. 4.2, we introduce minimal modifications to support training simultaneously on multiple modalities. Instead of a single set of input and output layers that project between the representation’s dimensionality and the hidden dimensionality of the model, we keep one set of such layers per modality. All other weights are shared between the modalities.

In our implementation, individual mini-batches always consist of a single modality. The per-modality batch sizes are chosen such that training steps take roughly the same amount of time between different modalities. The speed is mostly determined by the runtime of the modality’s autoencoder and the sequence length of the resulting latent representation. Specifically, we end up using batches of size 38, 8 and 16 for image, video and audio batches, respectively. Note that the requirement of approximately equal training step times is mostly a result of our implementation and could be avoided with other strategies such as sequence packing.

Since the modeling capacity has to be split among the modalities, the relative performance across modalities will be affected by the relative sampling frequencies and the relative weighting of losses computed on modality batches. Based on our dataset sizes and batch sizes, a single epoch consists of 5.26M, 0.75M and 0.0625M batches for image, video and audio, respectively. We moderately counteract this imbalance by sampling image batches with a probability of 57%, video with 30% and audio with 13%. With these ratios, a full image, video and audio epoch is reached after a total of 9.86M, 2.5M and 0.48M sampled batches.

![](images/6bb056130ae08f17e7a3d1c806b56115563053d5f29a46a5bfd31d7951142e65.jpg)

<details>
<summary>scatter</summary>

| image weight | FID   | Label       |
| ------------ | ----- | ----------- |
| 0.07         | 4.0   | 0.07, 0.32, 0.61 |
| 0.10         | 3.5   | 0.10, 0.51, 0.39 |
| 0.13         | 3.4   | 0.13, 0.63, 0.24 |
| 0.20         | 3.1   | 0.20, 0.08, 0.72 |
| 0.31         | 2.8   | 0.31, 0.12, 0.57 |
| Ours         | 3.2   | 0.07, 0.32, 0.61 |
| Ours         | 3.2   | 0.10, 0.51, 0.39 |
| Ours         | 3.2   | 0.13, 0.63, 0.24 |
| Ours         | 3.2   | 0.20, 0.08, 0.72 |
| Ours         | 3.2   | 0.31, 0.12, 0.57 |
</details>

(a) Image FID↓

![](images/cc9842c240bab3541a1ffc8c2169c70de03e995082f1aff06c0cd02d8bea8c29.jpg)

<details>
<summary>scatter</summary>

| video weight | FID (framework) | Method       |
| ------------ | --------------- | ------------ |
| 0.1          | 18.0            | FM           |
| 0.1          | 15.5            | Ours         |
| 0.3          | 12.0            | Ours         |
| 0.5          | 10.5            | Ours         |
| 0.6          | 10.0            | Ours         |
</details>

(b) Framewise FID↓

![](images/d29ee3dbfbda1cd38870a371651a62a5954333bef6cca158a38ce116a5eef585.jpg)

<details>
<summary>scatter</summary>

| video weight | FVD   | Label       |
| ------------ | ----- | ----------- |
| 0.1          | 112   | 0.07, 0.32, 0.61 |
| 0.1          | 102   | 0.13, 0.63, 0.24 |
| 0.1          | 98    | 0.20, 0.08, 0.72 |
| 0.1          | 105   | 0.31, 0.12, 0.57 |
| 0.3          | 72    | 0.07, 0.32, 0.61 |
| 0.3          | 66    | 0.07, 0.32, 0.61 |
| 0.5          | 64    | 0.13, 0.63, 0.24 |
| 0.5          | 62    | 0.13, 0.63, 0.24 |
| 0.6          | 60    | 0.20, 0.08, 0.72 |
| 0.6          | 58    | 0.31, 0.12, 0.57 |
The chart displays FVD as a function of video weight for different models or configurations. The legend indicates 'FM' (green triangle), 'Ours' (orange circle), and 'Ours' (purple circle). The data points are labeled with their respective parameter values.
</details>

(c) Video FVD↓

![](images/ab158c198425f468e023f0a34e27da6c8c3a3929abf44291d90ac56726f7e7c5.jpg)

<details>
<summary>scatter</summary>

| audio weight | FAD (CLAP) | Label        |
| ------------ | ---------- | ------------ |
| 0.3          | 160        | 0.07, 0.32, 0.61 |
| 0.4          | 155        | 0.10, 0.51, 0.39 |
| 0.4          | 154        | 0.13, 0.63, 0.24 |
| 0.4          | 158        | 0.20, 0.08, 0.72 |
| 0.4          | 157        | 0.20, 0.08, 0.72 |
| 0.4          | 153        | 0.31, 0.12, 0.57 |
| 0.4          | 152        | 0.31, 0.12, 0.57 |
| 0.6          | 153        | 0.07, 0.32, 0.61 |
| 0.6          | 152        | 0.31, 0.12, 0.57 |
| 0.6          | 151        | 0.31, 0.12, 0.57 |
| 0.7          | 154        | Ours         |
| 0.7          | 153        | Ours         |
| 0.7          | 152        | Ours         |
| 0.7          | 151        | Ours         |
| 0.7          | 150        | Ours         |
| 0.7          | 149        | Ours         |
| 0.7          | 148        | Ours         |
| 0.7          | 147        | Ours         |
| 0.7          | 146        | Ours         |
| 0.7          | 145        | Ours         |
| 0.7          | 144        | Ours         |
| 0.7          | 143        | Ours         |
| 0.7          | 142        | Ours         |
| 0.7          | 141        | Ours         |
| 0.7          | 140        | Ours         |
| 0.7          | 139        | Ours         |
| 0.7          | 138        | Ours         |
| 0.7          | 137        | Ours         |
| 0.7          | 136        | Ours         |
| 0.7          | 135        | Ours         |
| 0.7          | 134        | Ours         |
| 0.7          | 133        | Ours         |
| 0.7          | 132        | Ours         |
| 0.7          | 131        | Ours         |
| 0.7          | 130        | Ours         |
| 0.7          | 129        | Ours         |
| 0.7          | 128        | Ours         |
| 0.7          | 127        | Ours         |
| 0.7          | 126        | Ours         |
| 0.7          | 125        | Ours         |
| 0.7          | 124        | Ours         |
| 0.7          | 123        | Ours         |
| 0.7          | 122        | Ours         |
| 0.7          | 121        | Ours         |
| 0.7          | 120        | Ours         |
| 0.7          | 119        | Ours         |
| 0.7          | 118        | Ours         |
| 0.7          | 117        | Ours         |
| 0.7          | 116        | Ours         |
| 0.7          | 115        | Ours         |
| 0.7          | 114        | Ours         |
| 0.7          | 113        | Ours         |
| 0.7          | 112        | Ours         |
| 0.7          | 111        | Ours         |
| 0.7          | 110        | Ours         |
| 0.7          | 109        | Ours         |
| 0.7          | 108        | Ours         |
| 0.7          | 107        | Ours         |
| 0.7          | 106        | Ours         |
| 0.7          | 105        | Ours         |
| 0.7          | 104        | Ours         |
| 0.7          | 103        | Ours         |
| 0.7          | 102        | Ours         |
| 0.7          | 101        | Ours         |
| 0.7          | 100        | Ours         |
| 0.7          | -          | Ours         |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other        |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other      |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other    |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other   |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other     |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    |
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other       : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          | Other:    : 
|
| -            | -          = 'Ours' : 
|
| (Note: The 'Ours' label is not provided in the code.) since it is not explicitly provided in the image.
</details>

(d) Audio FAD↓   
Figure 16. Relationship between modality performance and weightings for multimodal runs. See Sec. D.

After having fixed also the sampling ratios of modalities, the remaining lever to control which modality gets favored during training is through modality specific loss weights. We multiply the loss in Eq. (7) using different weighting factors $w _ { I }$ for image, $w _ { V }$ for video, and $w _ { A }$ for audio batches. We explore various weighting triplets $( w _ { I } , w _ { V } , w _ { A } )$ , which can be found in Tab. 6. The image weighted, video weighted and audio weighted settings of Fig. 8 in the main paper correspond to the weight triplets (0.31, 0.12, 0.57), (0.13, 0.63, 0.24) and (0.20, 0.08, 0.72), respectively.

Tab. 6 summarizes the results across modalities for each of the weight triplets and shows that our approach improves multimodal performance consistently for all weightings and simultaneously for all involved modalities. Fig. 16 visualizes the relationship between a modality’s performance and weight. Finally, Fig. 17 shows the progress throughout training of all variants and demonstrates that the benefits of our approach are present throughout the complete training process.

![](images/31eb76a1e9202b54ce3c9a39b33f663f613124c906a75b392acc8f2ea9acf9e8.jpg)

<details>
<summary>line</summary>

| Training Steps | FM    | Ours   |
| -------------- | ----- | ------ |
| 200K           | 0.07  | 0.32   |
| 400K           | 0.10  | 0.51   |
| 600K           | 0.13  | 0.63   |
| 800K           | 0.20  | 0.08   |
| 1000K          | 0.31  | 0.12   |
| 200K           | 0.61  | 0.39   |
| 400K           | 0.39  | 0.57   |
| 600K           | 0.31  | 0.61   |
| 800K           | 0.39  | 0.51   |
| 1000K          | 0.31  | 0.57   |
| 200K           | 0.61  | 0.39   |
| 400K           | 0.39  | 0.57   |
| 600K           | 0.31  | 0.61   |
| 800K           | 0.39  | 0.51   |
| 1000K          | -     | -      |
</details>

(a) Image FID↓

![](images/8d5d47aa9952fe5af932be892f0fdacef9208c0427ffc0613cef8e9cb3e4edb8.jpg)

<details>
<summary>line</summary>

| Training Steps | FM    | Ours  |
| -------------- | ----- | ----- |
| 200K           | 0.07  | 0.13  |
| 400K           | 0.13  | 0.20  |
| 600K           | 0.20  | 0.31  |
| 800K           | 0.31  | 0.13  |
| 1000K          | 0.32  | 0.12  |
</details>

(b) Framewise FID↓

![](images/916df41ee54507aa39b68574540590ddf5fd4d10fcb11edb020fab8f18595562.jpg)

<details>
<summary>line</summary>

| Training Steps | FM    | FM    | FM    | Ours   | Ours   |
| -------------- | ----- | ----- | ----- | ------ | ------ |
| 200K           | 180   | 130   | 120   | 180    | 130    |
| 400K           | 160   | 110   | 90    | 160    | 110    |
| 600K           | 140   | 90    | 70    | 140    | 90     |
| 800K           | 120   | 70    | 60    | 120    | 70     |
| 1000K          | 100   | 60    | 50    | 100    | 60     |
</details>

(c) Video FVD↓

![](images/515b3948ae850f802a2c0a12fec853db2362a4e21de4cca1b2164db31b16b475.jpg)

<details>
<summary>line</summary>

| Training Steps | FM    | FM: 0.07, 0.32, 0.61 | FM: 0.10, 0.51, 0.39 | FM: 0.13, 0.63, 0.24 | FM: 0.20, 0.08, 0.72 | FM: 0.31, 0.12, 0.57 | Ours: 0.07, 0.32, 0.61 | Ours: 0.10, 0.51, 0.39 | Ours: 0.13, 0.63, 0.24 | Ours: 0.20, 0.08, 0.72 | Ours: 0.31, 0.12, 0.57 |
| -------------- | ----- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| 200K           | 175   | 175                   | 175                   | 175                   | 175                   | 175                   | 175                    | 175                    | 175                    | 175                    | 175                    |
| 400K           | 165   | 165                   | 165                   | 165                   | 165                   | 165                   | 165                    | 165                    | 165                    | 165                    | 165                    |
| 600K           | 160   | 160                   | 160                   | 160                   | 160                   | 160                   | 160                    | 160                    | 160                    | 160                    | 160                    |
| 800K           | 155   | 155                   | 155                   | 155                   | 155                   | 155                   | 155                    | 155                    | 155                    | 155                    | 155                    |
| 1000K          | 150   | 150                   | 150                   | 150                   | 150                   | 150                   | 150                    | 150                    | 150                    | 150                    | 150                    |
</details>

(d) Audio FAD↓   
Figure 17. Per-modality performance progress for multimodal runs. The benefits of our approach are present throughout the complete training process.

# E. Joint Video-Action Prediction

Recent works have demonstrated that video generation models can serve as effective action prediction models for embodied AI (Du et al., 2023; Wu et al., 2023; Black et al., 2023; Cheang et al., 2024; Hu et al., 2025; Shen et al., 2025; Pai et al., 2025; Tian et al., 2024; Li et al., 2025; Cen et al., 2025; Liang et al., 2025; Ye et al., 2026). We follow a simplified setup inspired by VideoVLA (Shen et al., 2025): instead of the full Open X-Embodiment (OXE) dataset (Collaboration et al., 2025), we only use the 73.5k episodes from RT-1 (Brohan et al., 2023), which is a small subset of OXE. Except for the 1D gripper action values, we normalize the remaining 6D action vector using 1% and 99% quantiles.

We add a new set of input and output layers to the model, which we initialize from the video-weighted runs (with 625M parameters) described in Sec. D, in either the FM (vanilla Flow Matching) or Ours variant. We encode the first frame using the image autoencoder (FLUX.2 (Black Forest Labs, 2025)) and 49 frames using the video autoencoder (WAN2.2 (Wan et al., 2025)). The audio input and output layers are dropped as they are not used in this experiment. We use a uniformly sampled timestep between 0 and 0.2 to apply noise augmentation to the image with the forward process. Handling of the text instruction for the task and the noise schedule for the video frames remain the same as in other experiments. In addition to the 49 video frames, we predict 6 action vectors (each 7-dimensional). We use the same noise timesteps for video frames and actions.

We train both variants for 100k steps and evaluate every 10k steps using the SIMPLER simulator (Li et al., 2024a). Given an instruction and rendering of the current state of the simulation, our model predicts 49 frames and 6 actions. We execute all 6 actions in the simulator, retrieve the updated rendering of the resulting state, and repeat the process. We run Pick Up Coke Can tasks 15 times, Open/Close Drawer tasks 18 times, Move Near tasks 10 times, and Open and Place tasks 7 times. We run every checkpoint twice over this list of tasks to get a rough estimate of the variance in success rates (shaded area in Fig. 18a).

Fig. 18a shows the success rate averaged over the four groups of tasks. Ours performs better than the baseline throughout the finetuning, showing that it learns more efficiently from a limited set of data. In Fig. 18b, we analyze the success rate by task group across an early step (30k) and the final step (100k). Early on, Ours outperforms the baseline in all tasks and achieves success in all task categories, whereas the baseline fails to perform Open and Place tasks at all. With continued training, we observe that the performance for simple tasks that involve a single object (picking a coke can, opening or closing a drawer) is similar for both methods, whereas for the more complex tasks that involve either two objects (placing one object near another) or a sequence of actions (open a drawer, then put an object into it), our method yields significantly better performance than the baseline. This suggests that our approach learns internal representations that improve complex visual reasoning capabilities.

![](images/e7534cfe89aa5e1a7c01d235478f38a9d2ed3c78a021224263367f1757874485.jpg)

<details>
<summary>line</summary>

| Training Steps | Ours | FM  |
| -------------- | ---- | --- |
| 0K             | 15   | 10  |
| 20K            | 25   | 12  |
| 40K            | 31   | 20  |
| 60K            | 37   | 28  |
| 80K            | 47   | 35  |
| 100K           | 47   | 35  |
</details>

(a) Success rate averaged over four task groups throughout training. Ours consistently outperforms the vanilla Flow Matching baseline (FM) throughout finetuning, demonstrating more efficient learning from limited data. Shaded areas indicate variance across two evaluation runs per checkpoint.

![](images/54593efa7fafe9675705bdaf128aa2c430da6be179067786221ff706524da100.jpg)

<details>
<summary>radar</summary>

| Category           | Ours @ 30K | Ours @ 100K | FM @ 30K | FM @ 100K |
| ------------------ | ---------- | ----------- | -------- | --------- |
| Move Near          | 50%        | 50%         | 40%      | 40%       |
| Open/Close Drawer  | 78%        | 78%         | 78%      | 78%       |
| Open and Place     | 29%        | 29%         | 29%      | 29%       |
| Pick Up Coke Can   | 73%        | 73%         | 73%      | 73%       |
</details>

(b) Success rate by task group. Early on (30k), Ours outperforms FM across all tasks and achieves success in all task categories, whereas FM fails entirely on Open and Place tasks. Later (100k), performance on single-object tasks (Pick Coke Can, Open/Close Drawer) converges, while Ours maintains a significant advantage on complex multi-object and sequential tasks (Move Near, Open and Place).   
Figure 18. Joint video-action prediction results on SIMPLER. Our self-supervised flow matching framework transfers effectively to embodied AI tasks, learning more efficiently from limited robotics data and showing particular advantages on complex manipulation tasks requiring multi-step reasoning.

# F. Joint Video-Audio Prediction

We consider a joint multi-modal generation task, where we train the model for audio-video prediction from a given conditioning image. We use the first frame of each video as the condition with a noise augmentation strength sampled uniformly in [0, 0.2]. We consider two initializations for this task: (1) the mixed-modality model described in Sec. D with high video-weighting, and (2) a model trained only on video data.We expect the multi-modal initialization to benefit this task substantially, since it observed all modalities during training.

For each training formulation (flow matching, Self-Flow), we consider both a video-only model and a mixed multi-modal model. Note that in the case of mixed multi-modal initialization, we can reuse all the modality specific layers from Sec. D. For video-only initializations, we initialize these weights randomly instead. Fig. 19 reports the FVD of the generated videos for each initialization given the same amount of video training samples. The results confirm our intuition: mixed-modality variants consistently outperform video-only variants. Interestingly, Self-Flow with video-only initialization outperforms the baseline with multi-modal initialization, further demonstrating the strength and generality of our learned representations.

# G. Additional Image Generation Results

# G.1. ImageNet Convergence Comparison

As mentioned in Sec. 4, Self-Flow outperforms REPA on ImageNet, despite REPA using DINOv2 for feature alignment. In Fig. 20a, we present the full convergence comparison between Self-Flow and the baselines, in the same format used in (Yu et al., 2024). Since DINOv2’s training set (LVD-142M) explicitly includes ImageNet and 142M additional web-retrieved images curated for ImageNet similarity (Oquab et al., 2024), it takes Self-Flow longer to catch up with REPA (compared to the other experiments in the main paper) and finally outperform it after ∼3M training steps.

![](images/fa31fc844d775bc47e6d53515fab145ea52f983f6fe49f9da2c74930297d93c7.jpg)

<details>
<summary>line</summary>

| Video Examples Seen | FM mixed | FM video-only | ours mixed | ours video-only |
| ------------------- | -------- | ------------- | ---------- | --------------- |
| 50M                 | 70.0     | 78.5          | 67.2       | 71.8            |
| 60M                 | 70.0     | 77.0          | 67.2       | 67.8            |
| 70M                 | 70.0     | 76.0          | 67.2       | 67.8            |
| 80M                 | 68.5     | 75.0          | 65.8       | 67.8            |
| 90M                 | 68.5     | 74.0          | 63.8       | 67.8            |
</details>

Figure 19. Joint video-audio prediction given a conditioning image, initialized from either a video-only or multi-modal (mixed) checkpoint. While multi-modal initialization benefits both formulations, even video-only Self-Flow outperforms multi-modal flow matching (FM).

# G.2. Combining Self-Supervised Flow Matching with Semantic Autoencoders

![](images/cde56602aa82b7a48738d4310890caeb7b86fbc5b25e82f776333f5ef15ccf9d.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | REPA | SRA | Ours |
| -------------- | --------------------- | ---- | --- | ---- |
| 50K            | 89.0                  | 82.0 | 89.0 | 88.0 |
| 200K           | 25.0                  | 10.0 | 24.0 | 13.0 |
| 400K           | 23.0                  | 8.0  | 12.0 | 10.0 |
| 1.0M           | 18.0                  | 6.5  | 9.0  | 8.0  |
| 2.0M           | 15.0                  | 6.0  | 8.0  | 7.0  |
| 4.0M           | 12.0                  | 5.5  | 7.0  | 6.5  |
| 7.0M           | 9.0                   | 5.5  | 6.5  | 6.0  |
</details>

(a) ImageNet convergence comparison (class-to-image)

![](images/29ed9957ae8c927b2c6a777291dd97450a0149e16361ec5cbec38203786fe75a.jpg)

<details>
<summary>line</summary>

| Training Steps | Vanilla Flow Matching | Ours |
| -------------- | --------------------- | ---- |
| 200K           | 3.3                   | 3.0  |
| 300K           | 3.05                  | 2.83 |
| 400K           | 2.93                  | 2.78 |
| 500K           | 2.82                  | 2.7  |
| 600K           | 2.79                  | 2.65 |
</details>

(b) Self-Flow with FLUX.2 AE (text-to-image)   
Figure 20. Additional image results. (a) ImageNet convergence comparison. Our method outperforms REPA after ∼3M training steps, while the experiments in the main paper show an advantage to Self-Flow from the early training steps. This can be attributed to REPA’s alignment with DINOv2, which was heavily trained on ImageNet. (b) Self-Flow with semantic autoencoders. Our self-supervised framework provides consistent improvements even when applied over semantically structured latent spaces such as RAE (Zheng et al., 2025) (on ImageNet, 4a) and FLUX.2 (Black Forest Labs, 2025) (for T2I, 20b), demonstrating complementary benefits.

Recent works suggest integrating external representations into the latent spaces of diffusion and flow models by training the model to directly denoise the representation produced by an external encoder (Zheng et al., 2025; Wu et al., 2025a), producing semantically structured latent spaces that aim to improve generation quality and convergence speed. While these approaches are not domain-agnostic by design, we evaluate whether our self-supervised flow matching framework provides additional benefits when applied over such latent spaces.

We perform experiments over RAE on ImageNet, using a plateau-logit-normal noise scheduler and a sampling shift of $\alpha = 1 0 . 0 \ : \mathrm { \Omega }$ . As shown in the main paper (Fig. 4a), our method provides consistent improvements over the RAE baseline throughout training. This demonstrates that our approach is complementary to semantic autoencoders: even when the latent space is already semantically structured, our framework further improves generation quality by encouraging the model to learn stronger representations during training. Similarly, we also observe that this observation holds in text-to-image experiments with the FLUX.2 AE (Black Forest Labs, 2025) (Fig. 20b), which uses an approach similar to REPA-E (Leng et al., 2025a) to obtain a semantically structured latent space with strong reconstruction capabilities. As discussed in Sec. 5, extending our self-supervised approach to jointly train the autoencoder in an end-to-end manner, as in REPA-E (Leng et al.,

2025a), is a promising direction for future work. The consistent improvements observed across all settings in this work make it a promising candidate to yield superior results compared to methods relying on external representations.

H. Layer Selection Ablations   
![](images/ae5e9d6a95418da75abdb755689a646488aaf3b286c2ce1efef735f35ac1f98b.jpg)

<details>
<summary>line</summary>

| Training Steps | Ours (l=8, k=20) | l=4   | l=12  | l=16  | l=20  |
| -------------- | ---------------- | ----- | ----- | ----- | ----- |
| 100K           | 32.0             | 31.5  | 31.0  | 33.0  | 39.0  |
| 150K           | 21.0             | 20.5  | 20.0  | 22.5  | 29.0  |
| 200K           | 16.5             | 16.0  | 15.5  | 18.0  | 23.5  |
| 250K           | 14.0             | 13.5  | 13.0  | 15.5  | 20.5  |
| 300K           | 12.5             | 12.0  | 11.5  | 14.0  | 18.5  |
| 350K           | 11.5             | 11.0  | 10.5  | 12.5  | 16.5  |
| 400K           | 10.5             | 10.0  | 9.5   | 11.5  | 15.5  |
</details>

(a) Student layer ablation (l)

![](images/e6334d8f5ca446fa5dfe0583853ea0c90120b1f194e417ce0f757a31dd05da3f.jpg)

<details>
<summary>line</summary>

| Training Steps | Ours (l = 8, k = 20) | k = 12 | k = 16 | k = 24 |
| -------------- | --------------------- | ------ | ------ | ------ |
| 100K           | 32.0                  | 38.0   | 36.0   | 31.0   |
| 150K           | 21.0                  | 27.0   | 25.0   | 21.0   |
| 200K           | 16.0                  | 22.0   | 20.0   | 16.0   |
| 250K           | 14.0                  | 19.0   | 17.0   | 14.0   |
| 300K           | 12.0                  | 17.0   | 15.0   | 12.0   |
| 350K           | 11.0                  | 15.0   | 13.0   | 11.0   |
| 400K           | 10.0                  | 14.0   | 12.0   | 10.0   |
</details>

(b) Teacher layer ablation (k)   
Figure 21. Effect of layer selection. We vary the student layer l (left) and teacher layer k (right) independently, with the other fixed at its default value. Performance is stable near our default selection but degrades when layers are too shallow (weak semantic signal) or when distilling to overly deep student layers (interferes with generation).

We ablate the choice of student layer l and teacher layer k for the self-supervised objective (Eq. 6) on the ImageNet class-to-image set. In all main experiments, we use $l = 0 . 3 D , k = 0 . 7 D$ where D denotes the total network depth.

Fig. 21 shows the effect of varying each layer independently. We observe that performance is stable across a range of layer choices near our default selection (e.g., varying the student layer from 8 to 4,12 has little impact on the results), suggesting the method is not overly sensitive to this hyperparameter.

However, performance degrades when layers deviate substantially from these ranges. For the teacher, using shallow layers hurts performance because semantic representations have not yet fully emerged, reducing the effectiveness of the distillation signal. For the student, distilling to deeper layers interferes with generation quality.

# I. Additional Qualitative Results

We provide additional qualitative results for text-to-image generation (Figs. 22–26) and text-to-video generation (Figs. 27– 31). Figs. 33–34 further demonstrate improved typography results from a scaled 4B parameter version of our multi-modal model, trained for just 100K steps on high-resolution images and videos. We refer the reader to the supplementary website for full videos, images from all figures, samples from the scaled 4B model, and audio results. For image generation, our method consistently produces improved structural coherence, texture fidelity, and preservation of high-frequency details across prompts of varying complexity. The video samples demonstrate our method’s ability to produce spatially and temporally coherent results, despite being trained on only 6M video samples.

Vanilla   
![](images/3dbd3d6007006e8283a8eaa2005d6ecd26caca7da26974efaa2f197676028cfc.jpg)

<details>
<summary>natural_image</summary>

Red tractor parked on a dirt path in a grassy field under a cloudy sky (no visible text or symbols)
</details>

REPA   
![](images/b6787568dc86514fb60cf9f208534ac52b69ad9de8bac9784f7017669829afc2.jpg)

<details>
<summary>natural_image</summary>

Red tractor parked on a grassy field under a cloudy sky (no visible text or symbols)
</details>

SigLIP2   
![](images/3d2b7fff0d814933627fac10399d99926283c6f287bb6eeb095fb41451db0596.jpg)

<details>
<summary>natural_image</summary>

Red tractor with black wheels parked on a grassy field under a blue sky (no visible text or symbols)
</details>

SRA   
![](images/2de014ab3220e7b16454d01223e39cf81f5a9569691a3028d21297d3439705f4.jpg)

<details>
<summary>natural_image</summary>

Red tractor with large tires parked on a gravel field under a blue sky (no visible text or symbols)
</details>

Ours   
![](images/7c7f21f3c702c8783b418cfbb746e1ec15988ac7de00de801ae9265c3f506881.jpg)

<details>
<summary>natural_image</summary>

Red agricultural tractor on a dirt path under a blue sky (no visible text or symbols)
</details>

a tractor, photographed in a vibrant red color, parked on a dirt path beside a grassy field under a partly cloudy sky, highlighting its vintage agricultural design.

![](images/31d40b067c92fa10cc2ae8a19fbacb17f6c15fef3786c7d877e96e1a116349d2.jpg)

<details>
<summary>natural_image</summary>

Furied brown fur hat with red plaid cap, no visible text or symbols
</details>

![](images/77494d4cd715ae40786cb392f23e2e6ac1faa20d3b3b3673494ec9f5b54e8d75.jpg)

<details>
<summary>natural_image</summary>

Red and black plaid fur hat with two turgers, displayed on a mannequin (no text or symbols visible)
</details>

![](images/949123f5a5a8758ee54e9b59c47b891bd1bc462ad4eea47050f0c0c08d539030.jpg)

<details>
<summary>natural_image</summary>

Mannequin wearing a plaid red-and-white fur hat with black trim (no text or symbols visible)
</details>

![](images/20cf2974bff6f43a9b7bc5fe5b127614c461013e1f38d9e3e38b6ff2b805a286.jpg)

<details>
<summary>natural_image</summary>

Red and black plaid hat with fur trim, displayed on a mannequin (no text or symbols visible)
</details>

![](images/ebfbedf4a513e4c58f6793b1bdad9842cd24301ccb4c41d28715da5822579cd9.jpg)

<details>
<summary>natural_image</summary>

Mannequin wearing a plaid hat with fur trim (no text or symbols visible)
</details>

a trapper hat, featuring a red and black plaid pattern with a thick fur around the brim and ear flaps, displayed on a mannequin head against a light background.

![](images/91e349763f5d124856081c49aa51f0a65c2b76b1da1586a1024b2cd6ce495a19.jpg)

<details>
<summary>natural_image</summary>

A large elephant standing on a dirt path surrounded by greenery (no text or symbols visible)
</details>

![](images/b637cfd0ab483c301bfce792354df4df1128c59651303b4d0ff3a18939c69989.jpg)

<details>
<summary>natural_image</summary>

A large elephant standing on a dirt road surrounded by green forest (no text or symbols visible)
</details>

![](images/5c4d6de0a18e9ff51135f947a488b011a7d017c3334160f7246b020c64541c5f.jpg)

<details>
<summary>natural_image</summary>

A large elephant standing on a dirt path surrounded by greenery (no text or symbols visible)
</details>

![](images/3e565a74469aab9b643f91927a81762ce947a692ad5413f89f31bbfdbc0d66c6.jpg)

<details>
<summary>natural_image</summary>

A large elephant walking on a dirt road surrounded by green foliage (no text or symbols visible)
</details>

![](images/31f2fbe0712d5ffad8780352275509f41e69ebdcaf7cf4040f34d52479b37312.jpg)

<details>
<summary>natural_image</summary>

A standing elephant on a dirt road surrounded by green foliage (no text or symbols visible)
</details>

an elephant standing on a dirt path in a natural, forested area with its trunk extended downward, captured in high-definition photography.

![](images/0cbf9cbd68049ce8bc9dd6f1f6fe2a02e4ed00ea71cb0c415e8a509682b95c79.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man with beard and short hair, wearing a plaid shirt (no text or symbols visible)
</details>

![](images/6fb105a32a79c5fd8300213dd6c63896f3448f32af66cd6a42296fdcfc95f9d9.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man with short hair and beard, wearing a checkered shirt (no text or symbols visible)
</details>

![](images/866a970b2169a421dc38dcc2d18843a5bf8ab0575ebb8a4910d7e8e9d3d4a1ed.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man with beard and mustache, wearing a checkered shirt (no text or symbols visible)
</details>

![](images/2812045714cd74903a72442713739dda515f8a68d9e00035117cff3c3e480d49.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man with short hair and beard, wearing a plaid shirt (no text or symbols visible)
</details>

![](images/12e8ffa88e67bb65bb68704189ffe4bc9e6dc21cbb5d7a03942f02cde9d1fecc.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man with short hair and beard, wearing a checkered shirt (no text or symbols visible)
</details>

a softly lit portrait of a man with short hair and a beard, wearing a dark shirt with a plaid collar, against a light background with blurred circular elements.

![](images/f36facffc88c97d1bedf06d10c98edc921d6693a819d0aec8c061c1736859ec2.jpg)

![](images/335ae82c88a7495b9361f95a6eefe434f4265b12d3376edb15d090450cf3b7f2.jpg)

![](images/e4cebf2594b10b11eb0b3f48910bbe79ea00ba269973a5f3a0082e911eb4a3a2.jpg)

![](images/89971a4d05f298b02c3d1bf9475678f13c94fbe0345107b8e3fcb5965df4d2a1.jpg)

![](images/9789c41c71c2d0d53f6d20eed05b7d56f933bf2f828fac91daac3bd23d5f1c40.jpg)  
a stylized, cartoon figurine of a Shiba Inu dog, depicted in a friendly pose while holding a bowl of food, standing on a wooden base.

Figure 22. Qualitative comparison (text-to-image). Our method produces superior structure, texture, and fine-grained details across prompts of varying complexity.

Vanilla   
![](images/a5fe270c6f1997b62e574b2f35cd3bdfd7649f9736d0f0174803f48e3dfc8d99.jpg)

<details>
<summary>natural_image</summary>

Black and white striped fish-shaped object mounted on a window, no text or symbols visible
</details>

REPA   
![](images/e0d867b442a3b4bf6debdf36aaabddb180785f6eb47a9cb4a28c6b3ee1f68845.jpg)

<details>
<summary>natural_image</summary>

Decorative fish-shaped object with black and white stripes displayed on a window, no text or symbols visible.
</details>

SigLIP2   
![](images/aa225c460a5bc791c1a9b3ca5d27cfec78db67abd6133b4ff1ee12f46c3be589.jpg)

<details>
<summary>natural_image</summary>

Abstract sculpture made of black and white striped fabric displayed indoors near a window with red-tiled roofs (no text or symbols visible)
</details>

SRA   
![](images/ad1706548d1582b2541bbb00a83fd4c37c31f2ddaf913927f8f851e75cdd0234.jpg)

<details>
<summary>natural_image</summary>

Black and white striped fish-shaped object hanging from a window, no text or symbols visible
</details>

Ours   
![](images/0f1c18709b81e1a8a5cb61afd8c80da81b64621f43ab0146ef242d7c7d281254.jpg)

<details>
<summary>natural_image</summary>

Decorative fish-shaped object with black and white stripes, displayed on a white surface near a window (no text or symbols visible)
</details>

a hand-painted, wooden fish-shaped decoration with horizontal white stripes, resting on a white shelf near a window.

![](images/d8da50822c6d63864b95c4ba1fe4ea041fbf5e341119123a5614251ee3b0ed92.jpg)

<details>
<summary>natural_image</summary>

Two food items: a panda-shaped rice dish and a plated dish with meat, vegetables, and sauce (no text or symbols visible)
</details>

![](images/b0320741ce1bb607d8240904c30f186d87d104b4d5e01d2f15f582e64318dc45.jpg)

<details>
<summary>natural_image</summary>

Two orange rectangular food containers: one with a panda-shaped rice topped with seaweed, the other with meat and vegetables (no text or symbols visible)
</details>

![](images/ceba9845790858e21c4c5e0e7a15d14c8904837fd39041781aafaa2848c500ab.jpg)

<details>
<summary>natural_image</summary>

Orange food tray with rice and meat, no visible text or symbols
</details>

![](images/785d34f709554a7d66fa459a79992863d20fece96d7d311c35f80906b67cb668.jpg)

<details>
<summary>natural_image</summary>

Bowl of rice with a panda face design, accompanied by a side dish of meat and greens (no text or symbols visible)
</details>

![](images/5e51a806c45073c64130f1d4a19cac918582d6ec3b8fcd0fe1d2558ee4333699.jpg)

<details>
<summary>natural_image</summary>

Orange tray with a panda-themed rice topped with a cartoon face, accompanied by a side dish of meat and vegetables (no text or symbols visible)
</details>

a bento box featuring a compartment of white rice decorated with nori into a cartoonish face, by another compartment containing grilled meat and vegetables.

![](images/180210ae7f173b4248ed9478ca2e778e844ccbb72e0b4f7ac884743a40000ea0.jpg)

<details>
<summary>natural_image</summary>

Close-up of a donkey standing in a grassy field with a metal fence in the background (no visible text or symbols)
</details>

![](images/49a071ddd9a8c10552fa22ce0e7e8fdfda5e72d31c3b2860823c826f8fa7680c.jpg)

<details>
<summary>natural_image</summary>

A donkey standing in a grassy field with a metal fence and two other animals visible in the background (no text or symbols)
</details>

![](images/49b67f1d5a4867080a2e9f8bb9c4b0dc8a1e6b967689a8fb95ac62b483456cec.jpg)

<details>
<summary>natural_image</summary>

A dark-colored goat standing in a grassy field with a metal fence in the background (no visible text or symbols)
</details>

![](images/d8efbbfe29a5b1a7774723feb699d3239453cbb2e9bd79aa08b6ff079a7332d4.jpg)

<details>
<summary>natural_image</summary>

A donkey standing in a grassy enclosure with metal fence and blurred animals in the background (no visible text or symbols)
</details>

![](images/2168d38f1d9a1c2cf50473f92e2fe65c7b40cf16c52f279be73ae4fb2f39e7c0.jpg)

<details>
<summary>natural_image</summary>

A donkey standing in a grassy field with a white fence and wooden fence in the background (no visible text or symbols)
</details>

a donkey in a grassy, fenced enclosure stands close to the metal fence. The scene is captured in natural daylight.

![](images/e90f62c36562fc637908dbf3c2de124bbe1c69018c11750eefe5a1ee31ef0efe.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with dark hair and bangs, wearing a white bow accessory (no text or symbols visible)
</details>

![](images/4b9304d969852434a17ac456b1615a318f1df8a3405c8ee48d162fd14dd7d094.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with short black hair and bangs, wearing a patterned headband (no text or symbols visible)
</details>

![](images/9e7f7aff0be4eecd4de3dbebdf5ad2ebb682820342f3953603131bb0207916f5.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with short black hair and bangs, wearing a white headband with a decorative bow (no text or symbols visible)
</details>

![](images/f6b13741bf33de706b2d5f9f71ff1e02581ed7bc1792924e0116c21a1624b137.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with short dark hair and bangs, wearing a white headband (no visible text or symbols)
</details>

![](images/cebd9c4115f007a8253ddf37fdbef1aff72309fda565a5a59aa3e6e1d54075bd.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with short dark hair and a white bow accessory (no visible text or symbols)
</details>

a professionally portrait of a woman with medium-length dark hair with bangs, a large white decorative bow with silver embellishments on the side of her head.

![](images/e6904add05aacb6f1962360f318068ca9a9fb02d1f10db413a24d7632e55b7d3.jpg)

<details>
<summary>natural_image</summary>

Close-up of a pugman's head with large blue eyes and black fur, lying on a red surface (no text or symbols visible)
</details>

![](images/08df6603b7661f38a3335613e87a9b7d776179848cf26141b124917e7d8b7d24.jpg)

<details>
<summary>natural_image</summary>

Close-up of a pug dog resting on a red surface, showing its large eye and black beak (no text or symbols visible)
</details>

![](images/ff0dae41c74ef77543c401d2945817f9f94a53b834ce4ac4442f7b9d640488e0.jpg)

<details>
<summary>natural_image</summary>

Close-up of a pug dog's head with large eyes on a red textured surface (no text or symbols visible)
</details>

![](images/0d62aef7397d109bce98233dccd2097eb3bc9104f03f2f8da465031abe239de2.jpg)

<details>
<summary>natural_image</summary>

Close-up of a pug dog's face on red carpet, showing large eyes and dark fur (no text or symbols visible)
</details>

![](images/5a192e894f5566fafa474af08ed1ffee60ec9666aab154322d00107dcef0aa4c.jpg)

<details>
<summary>natural_image</summary>

Close-up of a pug dog's face on red fabric, showing large eyes and black ears (no text or symbols visible)
</details>

a close-up, soft-lit photograph of a pug lying on its back on a red textured surface, with its large, open eyes looking directly at the camera.

Figure 23. Qualitative comparison (text-to-image). Our method produces superior structure, texture, and fine-grained details across prompts of varying complexity.

Vanilla   
![](images/1a69f63025f5f4792d0ba94d48702d5f951e42e1662e07792d32e875b75ca45e.jpg)

REPA   
![](images/997baa88cac5f90a76a4e9d5afd68d76b869fdf6a1ccd04267e70dcfdbcd7375.jpg)

SigLIP2   
![](images/21bcb7bee280ba1ba36aeba935f32033683d85095e83f24c26e14e255dfe2cf0.jpg)

SRA   
![](images/1bfdbba5e979ec4aad55fb80b81d9301b002f076591e5c21f4d70c525a18897f.jpg)

Ours   
![](images/25cb4df3a54b83262391186f49c8c5fbf62e9fc13f54bb1dcd78241766e7cfd9.jpg)  
a midcentury-style terracotta sculpture of a hippopotamus, intricately detailed, seated with a decorative headdress against a plain white background.

![](images/94001c8adfd7e587dce1ac4806618e60abc9b52fb042872d99548ead9f96db35.jpg)

<details>
<summary>natural_image</summary>

Front view of a black classic car with large headlights and grille, parked in snowy weather (no visible text or symbols)
</details>

![](images/aeb4da7346fca77dd3d38e00fe48487103e32abb7756cdcc4d2c0b764c0fb6d6.jpg)

<details>
<summary>natural_image</summary>

Front view of a black classic car with headlights on, parked outdoors in winter (no visible text or symbols)
</details>

![](images/38e4a82e4a326b32c1f7f26e63b2974125fa5f6be7640dbed027884cf0805f29.jpg)

<details>
<summary>natural_image</summary>

Front view of a black car with headlights on snow, no visible text or symbols
</details>

![](images/7420aeadafdb0c1d22e99618dcd1e9fd37f8c3884df31f9da24a89223a1b4dbf.jpg)

<details>
<summary>natural_image</summary>

Front view of a black car with large headlights and snow-covered roof (no visible text or symbols)
</details>

![](images/06e36c4addc429de0d4c8f7085cbd520c81224bc89f2ed91f62d4abff65f0cad.jpg)

<details>
<summary>natural_image</summary>

Front view of a black car with red and orange headlights, parked in snowy weather (no visible text or symbols)
</details>

a close-up of the front of a car, highlighting its glossy black paint, chrome accents,, set against a snowy outdoor background.

![](images/67b5b1c4529dfa5a609acd7f12595412d41b1a4e0896c4b0cc491de6abf45f97.jpg)

<details>
<summary>natural_image</summary>

A small donkey wearing a pink fur-lined blanket, standing indoors near wooden structures (no visible text or symbols)
</details>

![](images/8273cec8d2e8e403628139e5a88b177eb0ca876021fc21ea710ad200d13738b9.jpg)

<details>
<summary>natural_image</summary>

A donkey wearing a red quilted fur coat, standing indoors near a barn (no visible text or symbols)
</details>

![](images/271964d4b2d9708d7fb6a1e6b9c03ee1dee306a445493e9b2d67c6f5c7e6590a.jpg)

<details>
<summary>natural_image</summary>

Side profile of a gray donkey wearing a red quilted fur hat, standing indoors near wooden barn (no text or symbols visible)
</details>

![](images/66dea46f2cdc9ddf70b895b1164c14083a74c26abcca42477b19cbfe634b3d58.jpg)

<details>
<summary>natural_image</summary>

A white donkey wearing a pink quilted fur suit, standing in front of a wooden barn (no visible text or symbols)
</details>

![](images/4414a6a977086c898a363223ba2ec771ef518e4a72a82a0fe9121ab19e7411a3.jpg)

<details>
<summary>natural_image</summary>

A donkey wearing a red quilted fur coat, standing in a barn with wooden cabinetry (no visible text or symbols)
</details>

a donkey standing inside a stable, wearing a blue halter and a red quilted protective blanket over its back, with wooden walls visible in the background.

![](images/8fa946999f17d8a94649a7a86353166cdfe6e4379f8f9e8876cda4dc0cdf09f9.jpg)

<details>
<summary>natural_image</summary>

A young child holding a white plush toy, wearing a purple shirt (no text or symbols visible)
</details>

![](images/9486841a59c8386dda8e22a9eb5c0ee2bda6a3a307a11c460ee6705dfaeab194.jpg)

<details>
<summary>natural_image</summary>

A young boy holding a white stuffed toy, indoors with plain wall background (no text or symbols visible)
</details>

![](images/5ba5a59cac62755385dd5d88398b4efebad4eb68e1131c618d4bafb8736b9837.jpg)

<details>
<summary>natural_image</summary>

Child holding a white teddy bear plush toy, wearing blue shirt (no text or symbols visible)
</details>

![](images/c1c40fb71a85b117b9ba4762bec13ceac8f804dc72263619ab632cc7e925a135.jpg)

<details>
<summary>natural_image</summary>

A young boy hugging a white plush toy, both resting together (no text or symbols visible)
</details>

![](images/d921797f8328b47807e0b9c305c822f2b68fc6f8a913395286144378b0a9e13c.jpg)

<details>
<summary>natural_image</summary>

Child holding a white plush toy, smiling broadly (no text or symbols visible)
</details>

a young boy tenderly hugging a white stuffed dog toy while standing indoors against a plain wall.

![](images/57891e8b649d0b8f01266b9e4fd79d5d0b15af6cc5dbb44d176fc7911a42ffb1.jpg)

<details>
<summary>natural_image</summary>

Marble bust of a classical figure with head and beard, displayed in an oval frame against a textured orange background (no text or symbols)
</details>

![](images/f547e7eeb499cad53472fc64c4263eaa89d262d2a0d98bbbd77bdc1c0382a1ec.jpg)

<details>
<summary>natural_image</summary>

Relief sculpture of a bearded man in classical attire, framed in an oval (no text or symbols visible)
</details>

![](images/f3b698efdac2a7e49fe9da3d68121ec0d9ec5d57e0804b9bf530e9f1374e6ffd.jpg)

<details>
<summary>natural_image</summary>

Oval-shaped stone bust of a historical figure wearing traditional headwear, displayed against a solid brown background (no text or symbols visible)
</details>

![](images/17f997cf001efe832f7f6711e0f47cb7679e7eac20760eba7dbe8638f9d43f96.jpg)

<details>
<summary>natural_image</summary>

Marble bust of a classical figure within an oval frame, no visible text or symbols
</details>

![](images/07e36deac50374090c5a8f062228c6b0627243e1443774df674efada4328269c.jpg)

<details>
<summary>natural_image</summary>

Ovally bust of a classical male figure, displayed on a textured wall (no text or symbols visible)
</details>

a detailed, classical-style sculpted bust of a man in a serious expression, wearing a toga, encased in an oval frame and mounted on a textured, earthy-toned wall.

Figure 24. Qualitative comparison (text-to-image). Our method produces superior structure, texture, and fine-grained details across prompts of varying complexity.

Vanilla   
![](images/169a3120891cc611215c85f3a8fe1e85f2e80c19127f4b8ce6c981c7660ea2e7.jpg)

<details>
<summary>natural_image</summary>

Person wearing a polka-dot face mask against a green and white background (no text or symbols visible)
</details>

REPA   
![](images/a16c9715a860a1e3f5034d9070291a04811310074c81e4003c9243c77aecdc52.jpg)

<details>
<summary>natural_image</summary>

Person wearing a black floral face mask against a green and white background (no text or symbols visible)
</details>

SigLIP2   
![](images/c95d8a7d201c9942710d7f859ad104f1a35c920d6610f795657e046f34696ca6.jpg)

<details>
<summary>natural_image</summary>

Person wearing a black floral-patterned face mask against a green and white background (no text or symbols visible)
</details>

SRA   
![](images/85dd2e38fc0be37a6aac9818e9852a4d0968fe6ee0a37fb1d2bacd4a902336aa.jpg)

<details>
<summary>natural_image</summary>

Person wearing a patterned face mask against a green and white background (no text or symbols visible)
</details>

![](images/f6aeccba5838c28cfa6467086b3f64db89862c1c7a1a6596016b99ad41503d70.jpg)

<details>
<summary>natural_image</summary>

Person wearing a black-and-white patterned face mask against a green and white background (no text or symbols visible)
</details>

a person wearing a black face mask with a white pattern, standing against a green and white gradient background, captured in a softly lit, balanced composition.

![](images/eece667356dd971d3ada8c1234e3448f3ed5f3668e50ae5e86b0fc66d7985378.jpg)

<details>
<summary>natural_image</summary>

Close-up of a pink sneaker with white sole and black legs, standing on pavement (no text or symbols visible)
</details>

![](images/bdecd4c8a2ebe6e64052f6c97b551edf0d3598e47343e3e6a88d4f93233c7326.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's lower legs wearing a pink sneaker and black pants, standing on concrete ground (no text or symbols visible)
</details>

![](images/9e723857f6bef26984c9e17adc1d65a4684b5a9873a5ae08700be0e737651134.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's legs wearing pink sneakers and black pants, standing on pavement (no visible text or symbols)
</details>

![](images/2b84544c5b1fdc5e23357d5bc5c744e22a4a17167d117eafd2238febc422c50d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's lower legs wearing a pink sneaker with a logo (no visible text or symbols)
</details>

![](images/e10847250e5107715f7b80851d7674078efd0dfa7355f3e08782b190074a7da7.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's lower legs wearing pink sneakers and black pants (no text or symbols visible)
</details>

a pair of pink sneakers, worn by someone standing on a concrete surface in an urban environment, close-up, low-angle photograph.

![](images/91cc01350b25df1f6a8ecab7418303c8372f26bec45e5fc30f873e3e1e06d9d3.jpg)

<details>
<summary>natural_image</summary>

Portrait of a young woman with long brown hair wearing a beige coat over a white shirt, set against a blurred green background (no text or symbols visible)
</details>

![](images/cc23cc3dc5308afa160522511b437fd8160cfbbe40fd20b602f103c267be85a0.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with shoulder-length brown hair wearing a beige turtleneck and white shirt, outdoors with blurred green foliage background (no text or symbols visible)
</details>

![](images/1870129f188c7ef947c498be08232658a9d6b4b9d796d7542ff9343e335a836f.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with shoulder-length brown hair wearing a beige coat over a white shirt, set against a blurred green background (no text or symbols visible)
</details>

![](images/00c9bd70233d8f1acbfc5a1b3c84eff7c169e4fcf3f51dd6583e1f87633fd550.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with long brown hair wearing a beige blazer over a white shirt, against a blurred green background (no text or symbols visible)
</details>

![](images/c0fabf4d671e83666dd3891c8dde092ece98d50a03dca9546c8e9125881f10df.jpg)

<details>
<summary>natural_image</summary>

Portrait of a woman with long brown hair wearing a beige turtleneck shirt, set against a blurred green background (no text or symbols visible)
</details>

a professionally lit portrait photograph of a woman with brown hair wearing a trench coat over a white shirt captured outdoors in natural light.

![](images/0cef030748057a123ce6b31248f4593f45cab3d7510ff7ccb6123924635a6d56.jpg)

![](images/60f24cbb0d0b425ba0e86911ed9569078bea96d85c864f16dfc9efcf960d9865.jpg)

![](images/f975afde963a38c46083848a97d7fa204835797976d1b6d08b163ba7ae350263.jpg)

![](images/c3a7f3987582a61a021183f043875f57418d0c728dc6e5675c78a2c7b21db4dc.jpg)

![](images/3a8c9a184ba4c5e3d93751ca518a298f068991bb0a90f8b8bf99a6c0f1e08b52.jpg)  
a highly detailed and colorful traditional Japanese Beckoning Cat, depicted in a stylized manner with intricate patterns and vibrant colors.

![](images/461602531949b94fa59916b650172103847a5e89c935c401bed89f7eaabaa844.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a white church with a blue windmill and two spires, set against a clear sky (no signage or text visible)
</details>

![](images/b0166d54f6322e94b31576e22368f9e00dcfb171570ee99c4e0cc8356510cb0b.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a white church with blue spires and a stone pathway, surrounded by trees under a clear sky (no signage or text visible)
</details>

![](images/74745e4bee9b7e1e8b65bf9b90ea9dce37888bdd6dc2c30a20c511fb1d6b6326.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a white church with a tall spire, surrounded by hills and trees under a clear blue sky (no signage or text visible)
</details>

![](images/bf69efcb4f97941aaebd27f19f8a48e621ee2b454db9c1f5101bbb0cbaefe426.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a white church with a tall spire and blue roof, surrounded by dry hills under a clear blue sky (no signage or text visible)
</details>

![](images/608c32e2b02ec0892bb8f7cfbf665422a8553987771aec343a04dcde41df35e5.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a white church with a tall spire and blue roof, surrounded by greenery under a clear sky (no signage or text visible)
</details>

a photograph of a small, white church with blue trim and a prominent blue cross on its facade, featuring a tall steeple and situated on an elevated plot.

Figure 25. Qualitative comparison (text-to-image). Our method produces superior structure, texture, and fine-grained details across prompts of varying complexity.

Vanilla   
![](images/1d050d8c119a6dea384da1ba390bb9cb4dc2b6f389fa3a9c7cdcd1b92e3c3e4d.jpg)

<details>
<summary>natural_image</summary>

Two plush toys, one white and one green, sitting side by side against a pink textured background (no text or symbols)
</details>

REPA   
![](images/89a70494e4b9cfe46a709b39e18eb84d846e831c220107da5d0ec65a13fcd984.jpg)

<details>
<summary>natural_image</summary>

Two plush toys, one white bear and one green rabbit, displayed on a red patterned fabric background (no text or symbols visible)
</details>

SigLIP2   
![](images/bd3a51e9c12fa12b2b47cc9ef5e5123b6657ae7fdd5b9e0a05e3d4360b55201e.jpg)

<details>
<summary>natural_image</summary>

White teddy bear and green bunny plush toy against a pink textured background (no text or symbols)
</details>

SRA   
![](images/352de9b38dd2e0283144cccb3367cacfd0e56b9bead14ea07183105b32c1de68.jpg)

<details>
<summary>natural_image</summary>

Two plush toys, one white bear and one green rabbit, displayed against a pink textured background (no text or symbols visible)
</details>

![](images/7c9944598a0728276ad901b6b6a4a75d9e5c0bc9aed5e96ed4bdfb3802718344.jpg)

<details>
<summary>natural_image</summary>

Two plush toys, one white and one green, sitting together against a pink background with red pattern (no text or symbols)
</details>

a photograph of plush toys—a white teddy bear and a green rabbit—arranged together against a pink, patterned background.

![](images/3ad7fe8232961ccdcf9d4b5c9953967bac5121ed62c434495f75c6d9ecb6246c.jpg)

<details>
<summary>natural_image</summary>

Two white deer standing in front of a metal fence under a blue sky (no text or symbols visible)
</details>

![](images/ea44d0b702a63b722e0f7d18a221c9d89691d9a7894037cb3071ebf2db662560.jpg)

<details>
<summary>natural_image</summary>

White deer with large antlers standing in a fenced enclosure under a blue sky (no text or symbols visible)
</details>

![](images/5ba64da804797211f591c25afbf366468bf51b2fc6d5e3cbb4de4e9217e08061.jpg)

<details>
<summary>natural_image</summary>

Two white deer standing behind a wire fence with a thatched-roof building in the background (no visible text or symbols)
</details>

![](images/bd7376fd5e9ef3639288d6eff5a5eda52594ab8a057c06bb2f0c5986c4b58df4.jpg)

<details>
<summary>natural_image</summary>

Two white antelopes standing in front of a barn under a partly cloudy sky (no text or symbols visible)
</details>

![](images/3a01b21d490a859c9dbe2f24cdabf488db42515510a8ddece67cd2f8e8db2521.jpg)

<details>
<summary>natural_image</summary>

Two white deer with large antlers standing in front of a fence under a blue sky (no text or symbols visible)
</details>

two white reindeer inside a wire-fenced enclosure, one facing the camera and the other turned to the side; the background includes a wooden structure.

![](images/797a4ae0b0408f49622d166482278a191f9ad71fdfbc162541f3800ab2df2f7f.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern staircase with curved railings and metallic railing (no text or symbols visible)
</details>

![](images/50e1aa44ca3432388f0170be5198ac752c80e01f66fb70664ebb8c3c29f5ced6.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern staircase with exposed metal railings and skylights (no text or symbols visible)
</details>

![](images/e53c028441b610323fd2c820f02508f8c3a9903ffdd4c1d393fd7449e91e422a.jpg)

<details>
<summary>natural_image</summary>

Interior architectural detail showing a modern curved staircase with metal railings and balconies (no text or symbols visible)
</details>

![](images/b511251465d114ae243da44b78e82def3366b2473a57c937cf36dfb9822be7c0.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern building with glass railings and ceiling skylights (no visible text or symbols)
</details>

![](images/d3111caae2875d0c85d9e5d1cee93c80bbf1460a836a0e5cdf06ca96f81fc0ab.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern building with escalators and glass ceiling (no visible text or symbols)
</details>

a photograph taken from a low angle of an, highlighting its sleek, modern design with metallic railings and overhead structural elements under bright lighting.

![](images/c01bb9a6026aee82fb7e9680ef175a43e143b9031dde0b9e41426bbbeb943266.jpg)

<details>
<summary>natural_image</summary>

Black cat wearing a blue uniform with red trim, held by a hand against a blue wall (no text or symbols visible)
</details>

![](images/c3d824985845b39a1d0c4247add5a0e60c654c5f98323f21255508ddccedc689.jpg)

<details>
<summary>natural_image</summary>

Black cat wearing a blue and white sports jersey, sitting on a hand against a blue background (no text or symbols visible)
</details>

![](images/c1e55c16d4d037bad9ec80afa9b907c6e9bba37e2c6675f7c59478ae313eeb56.jpg)

<details>
<summary>natural_image</summary>

Black cat wearing a blue and white armband, sitting against a blue background (no text or symbols visible)
</details>

![](images/478d048884b2881c8ef2ac8997e03712316c7bb8e65beee761a846814dd1a1de.jpg)

<details>
<summary>natural_image</summary>

A black cat wearing a blue and red sports jersey, being held by a hand against a blue wall (no text or symbols visible)
</details>

![](images/5f35748f89738c7906171748f490412d69a98d2b6c9315bd6e0971422f77b39c.jpg)

<details>
<summary>natural_image</summary>

Black cat wearing a blue and red outfit with a heart-shaped collar, held by a hand against a blue background (no text or symbols visible)
</details>

a black cat in a blue jacket with red and white accents, wearing a tag, held by a visible arm, looking upwards with its mouth slightly open against a blue door.

![](images/a32c09a579e939282010963e4595823cf023908c0821e8a4b404ca81e54055bb.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man with beard wearing dark blue shirt (no text or symbols visible)
</details>

![](images/74cacc98e3117ac11f223f55140139578443dea61789bed797823b429b631c38.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man in a dark shirt, no visible text or symbols
</details>

![](images/107a7a550e2ea910a8dd88f410123079a68b68856b2e62c10c556aec26fd464b.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man with short hair and beard, wearing a dark blue shirt (no text or symbols visible)
</details>

![](images/08e5279f734ea20ac50344993d4508e23dac3d9e25036fb76e1b9b127f037847.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man with beard wearing dark blue shirt (no text or symbols visible)
</details>

![](images/b9aa38d1cd794aa65ebf9a8ebf4eb888abab3f5f63270f3d71c8415352466362.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man with short hair and beard, wearing a dark shirt (no text or symbols visible)
</details>

a man posing indoors with a shallow depth of field, wearing a dark blue shirt and sporting a friendly smile.

Figure 26. Qualitative comparison (text-to-image). Our method produces superior structure, texture, and fine-grained details across prompts of varying complexity.

Vanilla   
![](images/c5a83fd7fea8481d2e90e4046f0225fde44a53e371896fb89f333bbee1598144.jpg)

<details>
<summary>natural_image</summary>

Underwater photo of a sea turtle swimming near coral reefs (no text or symbols visible)
</details>

![](images/b64b3866f888d9de63780af6adf6ecae3ba1a09f905e1d6795d625408267c14d.jpg)

<details>
<summary>natural_image</summary>

Underwater photo of a sea turtle swimming near coral (no text or symbols visible)
</details>

SRA   
![](images/1f6ec30bad44240dab063fa4b95d9b783c2316b30c0c4f6be5bac60e05333cf4.jpg)

<details>
<summary>natural_image</summary>

Sea turtle swimming in deep ocean under turquoise water (no text or symbols visible)
</details>

![](images/e246488c2b62d80f9f5fac54a435d2bd48f821bedf234bcdfb9de8d949fab062.jpg)

<details>
<summary>natural_image</summary>

Underwater photo of a sea turtle swimming near coral (no text or symbols visible)
</details>

REPA   
![](images/837cc9e1a282676c0cacf3d271e42cf6ff2368d4a66ac6e206af3a7af50c0221.jpg)

<details>
<summary>natural_image</summary>

Underwater photo of a sea turtle swimming near coral and reef (no text or symbols visible)
</details>

![](images/98511e798117c4f00cbaf274536f34985d2a6dea7d16b33aaf1e047532753348.jpg)

<details>
<summary>natural_image</summary>

Underwater photo of a sea turtle swimming near coral (no text or symbols visible)
</details>

Ours   
![](images/29eee7708b9f7d508a09940cca0f8e385099c2e6ffb5374c4b352d17549cf29a.jpg)

<details>
<summary>natural_image</summary>

Sea turtle swimming in deep coral reef (no visible text or symbols)
</details>

![](images/2b747369098e852ebaa94425b99c13658969634700d695760cc48fe5970691b1.jpg)

<details>
<summary>natural_image</summary>

Sea turtle swimming in deep coral reef (no text or symbols visible)
</details>

a green sea turtle glides effortlessly through a colorful coral reef, filmed from a low angle. The serene creature moves with graceful flipper strokes, ancient eyes observing the vibrant marine life, shell patterns catching the filtered sunlight in the shallow tropical water.

Vanilla   
![](images/5f18a54688a45ffd97c9860accf7d2a09f13c773669456548617364bcd0dead1.jpg)

<details>
<summary>natural_image</summary>

Close-up of a yellow fruit with a small white spot on its surface, surrounded by green leaves (no text or symbols visible)
</details>

![](images/8be703996314e52ddc193f7b3ccc66157e338dac23bcbf26999bc2e9e141c91e.jpg)

<details>
<summary>natural_image</summary>

Cartoon illustration of a yellow fruit character with a sad expression, surrounded by green leaves (no text or symbols)
</details>

![](images/d67f85819d3072e379cf93ba67738dbc407d0430b9d6038acc7e4f0466864eea.jpg)

<details>
<summary>natural_image</summary>

Close-up of a yellow fruit with green leaves in the background (no text or symbols visible)
</details>

![](images/833e32e50eaefec113e37d5733f32b7997c01170644099a390d8ddb3abc27d11.jpg)

<details>
<summary>natural_image</summary>

Close-up of a yellow fruit with a small eye, resting on green leaves (no text or symbols visible)
</details>

REPA   
![](images/971daf87eb87a6a026b367560ada2a6261e4e82c7dab854d33df262ba20e2f7d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a yellow mango with two small white oval cutouts on its surface, surrounded by green leaves (no text or symbols)
</details>

![](images/2b422e309f677f964b74c4463369c328eaa28cb2d991d87b7beadaf473de0d98.jpg)

<details>
<summary>natural_image</summary>

Yellow mango with a small smiley face on its side, set against green leaves (no text or symbols)
</details>

Ours   
![](images/65a208b27499f2bc3d2a66a3e333f83cd17b4975587599a7864b69e46ac894bf.jpg)

<details>
<summary>natural_image</summary>

Cartoon illustration of a yellow fruit character with large eyes and a frowning expression, set against green leaves (no text or symbols)
</details>

![](images/679bcb211ad2a00b9e52d4d0352ea5012be6ebb046afa2f86142b75102adb1b7.jpg)

<details>
<summary>natural_image</summary>

Cartoon illustration of a yellow fruit with large eyes and a surprised expression, set against green foliage (no text or symbols)
</details>

a 3D animated orange tries to avoid being picked, hiding behind leaves, nervous expression.

Vanilla   
![](images/972cc2fb2b5904a2ddcc1dabb5a7622630619f65e978e94f5033812175588ac1.jpg)

<details>
<summary>natural_image</summary>

Close-up of a green snake with dark spots on its head, resting on dry ground (no text or symbols visible)
</details>

![](images/a81e1d6395ba645b4c4dba018cbf7f1afed7a52d39d99a543c62241c3a6dc82d.jpg)

<details>
<summary>natural_image</summary>

Close-up of green, striped aquatic plants on a textured surface (no visible text or symbols)
</details>

SRA   
![](images/ae3ad860449a222962cebd29766ac45f750897cc012ee153e4ee3802f7811833.jpg)

<details>
<summary>natural_image</summary>

Close-up of a yellow-tinted fish with visible eye and mouth, resting on a textured surface (no text or symbols)
</details>

![](images/f2b3b1f22607e91a5716ad64170d439aa774b190fa0d02f19af124fa52058216.jpg)

<details>
<summary>natural_image</summary>

Close-up of a green snake with detailed scales and open mouth (no text or symbols visible)
</details>

REPA   
![](images/a8c83bb046cee3196b74b03b3c9ea00d6337b76c37dbccb7a153555eee97de78.jpg)

<details>
<summary>natural_image</summary>

Close-up of a snake lying on a stone surface, showing black and white patterns (no text or symbols)
</details>

![](images/6486e0339fed92b937624ac4fa642e500db0e90acf4f1659a6f51ef68665b0d9.jpg)

<details>
<summary>natural_image</summary>

Close-up of a snake's black and white striped pattern, possibly a snake or snake with visible scales (no text or symbols)
</details>

Ours   
![](images/c0f73579b41cf9fe3be2737530e460ea94401f1b9a9134cd44f91fdadd7e9a30.jpg)

<details>
<summary>natural_image</summary>

Close-up of a snake with striped pattern, resting on a wooden surface (no text or symbols visible)
</details>

![](images/e4e74d5d1583343ec654db166cdb439d9f8268b4ed36b73075f676bf88a87217.jpg)

<details>
<summary>natural_image</summary>

Close-up of a snake with green and white patterned scales, no visible text or symbols
</details>

a snake slowly uncoils from its resting position, massive length gradually revealed. The powerful constrictor moves with undulating grace, muscular body flowing over itself as it extends.

Figure 27. Qualitative comparison (text-to-video). Baselines exhibit structural and temporal artifacts, while our method produces coherent and temporally smooth results.

Vanilla   
![](images/ec3eb5f15531d90ba71a8566cd16b8c712bffca3cb07ae17843122078ceab5dd.jpg)

<details>
<summary>natural_image</summary>

A large rhinoceros standing in a dry savanna under clear blue sky (no text or symbols visible)
</details>

![](images/80ef9acdec67f05cbe75caf6d9a36a6a38a7b1a3b647185eaca06050e2106b1a.jpg)

<details>
<summary>natural_image</summary>

A large rhinoceros standing in a dry savanna under a clear blue sky (no text or symbols visible)
</details>

REPA   
![](images/1665f4bc47dc64dfd0f6f279ec8d889376856b98890bdfccb0a98771b5e47593.jpg)

<details>
<summary>natural_image</summary>

Close-up of a large bull standing in dry grassland under clear blue sky (no text or symbols visible)
</details>

![](images/2d6b886dda2dac5535f46f35e2322414ee86a36770871be8419b89ba37a41022.jpg)

<details>
<summary>natural_image</summary>

A large dark-colored elephant standing in a grassy field under a clear blue sky (no text or symbols visible)
</details>

SRA   
![](images/b35ecf04584a15e5a46374db9ae8bda32ba556e27758160c168f022b1091a590.jpg)

<details>
<summary>natural_image</summary>

Close-up of a rhinoceros standing in a dry grassland under a clear blue sky (no text or symbols visible)
</details>

![](images/ae0c93c8b5f4d6058eb9c6fd002b2d07a93d668ac8f8f69f0ad3922a9125bb18.jpg)

<details>
<summary>natural_image</summary>

Two rhin elephants standing in a dry savanna under clear blue sky (no text or symbols visible)
</details>

Ours   
![](images/41a769d998e1971e30447af76baf81cf1e346c6523757f369b70ca03249f4b81.jpg)

<details>
<summary>natural_image</summary>

A rhinoceros standing in a dry grassland under a clear blue sky (no text or symbols visible)
</details>

![](images/c2a4c8acf61ce0b0314bd944363a1922782580a84d9cc8c1a8ed53b574656eb8.jpg)

<details>
<summary>natural_image</summary>

Rhinoceros standing in a dry grassland under a clear blue sky (no text or symbols visible)
</details>

a massive rhinoceros, seen from a low perspective, slowly walks across a vast, dry, light-colored plain. The powerful animal moves deliberately at a savanna landscape under a blue sky.

Vanilla   
![](images/8bdd06c9dbf888761994fd4f4decc065c62d87484ab928aac4ed42ec5f056c40.jpg)

<details>
<summary>natural_image</summary>

Animated character standing on a wooden path surrounded by autumn leaves (no text or symbols visible)
</details>

![](images/17532f343d4d53a82bfee5c415b415a51f6bc8a4bfdde1d5e34090b6ea79b183.jpg)

<details>
<summary>natural_image</summary>

Orange kangaroo walking on a forest path with golden leaves (no text or symbols visible)
</details>

REPA   
![](images/b81af4c74b57a3ceded1fb1ec22ac8544ab779c8ef484ecfae22ff945bcbbf77.jpg)

<details>
<summary>natural_image</summary>

Illustration of a kangaroo standing on autumn leaves with a tree in the background (no text or symbols)
</details>

![](images/479b6f174da760ec7a390647d08d5e455b970a614de6d98d568799ceb903cae4.jpg)

<details>
<summary>natural_image</summary>

Orange fox standing on a fallen autumn leaf path, surrounded by green grass and rocks (no text or symbols visible)
</details>

SRA   
![](images/d0a112a48be7e5ce93c43ece577581ed0d4745ad39e1ed42754d978388901f37.jpg)

<details>
<summary>natural_image</summary>

Animated fox standing on a forest floor with autumn leaves (no text or symbols visible)
</details>

![](images/8f3f1dfa5384ad047fa8318f946c9fd92007872d8e24f69e3ee922d85a7f9d24.jpg)

<details>
<summary>natural_image</summary>

Illustration of a fox standing on a wooden path surrounded by greenery (no text or symbols)
</details>

![](images/11b9e6ae69516ef7760a79b26f797471aa68c89d72c600448ec7bae40cb77dea.jpg)

<details>
<summary>natural_image</summary>

Illustration of a yellow fox standing on fallen leaves in a forest setting (no text or symbols)
</details>

![](images/9e38dd5690221abf46b68aa2a1c54b68b5782c0fbb196c226f4764ddde117333.jpg)

<details>
<summary>natural_image</summary>

Cartoon fox standing on fallen leaves in a forest setting (no text or symbols visible)
</details>

a cartoon 3D fox trots through fallen leaves, bushy tail swishing behind. The orange character continues on its woodland path.

Vanilla   
![](images/1835d35d046497178066d5509ec942bc003767653ab123f03a13fa9588b925cc.jpg)

<details>
<summary>natural_image</summary>

Close-up of a hand holding a sliced tomato on a wooden cutting board (no text or symbols visible)
</details>

![](images/4f78f3459922923f22fd54d7597e84dfc5c7456296490fb904b1a0487b9ad79b.jpg)

<details>
<summary>natural_image</summary>

Close-up of a hand holding a sliced tomato on a wooden cutting board (no text or symbols visible)
</details>

REPA   
![](images/0703a897aea24ddc70b824aa0b12d6a713cb77fec8764288e4cbd80ab5915be3.jpg)

<details>
<summary>natural_image</summary>

Close-up of a red tomato being cut with a knife on a wooden cutting board (no text or symbols visible)
</details>

![](images/6380b361f846a11da702d8d9394fb75a10158340a388107f43d90be462c139df.jpg)

<details>
<summary>natural_image</summary>

Close-up of a red tomato being cut with a knife on a wooden cutting board (no text or symbols visible)
</details>

SRA   
![](images/78ade7e50c45726a598a5faabd447047aadc9f6bd0bb53d460068950efb66e96.jpg)

<details>
<summary>natural_image</summary>

Close-up of a hand slicing a ripe orange tomato on a wooden cutting board (no text or symbols visible)
</details>

![](images/ebbd01c8415a84be23352d533b18ff9a906653c7cf591f1bda5f8ce30282ee87.jpg)

<details>
<summary>natural_image</summary>

Close-up of a hand holding a spoon cutting a red tomato on a wooden cutting board (no text or symbols visible)
</details>

Ours   
![](images/e12a34fd469f1c955610dd28d1055e5b29777f55ac3dfa4759ee007a32087bd4.jpg)

<details>
<summary>natural_image</summary>

Close-up of a hand slicing a bright orange tomato with a knife (no text or symbols visible)
</details>

![](images/78b57f1b18556798790114611632df0bfe9a6b241597e6262e3a3b5d98fff37e.jpg)

<details>
<summary>natural_image</summary>

Close-up of a hand slicing a ripe orange tomato with a knife (no text or symbols visible)
</details>

a sharp chef's knife slices through a ripe red tomato on a wooden cutting board, the blade moving in smooth, confident strokes. Juice pools around, the knife work demonstrating years of practiced technique.

Figure 28. Qualitative comparison (text-to-video). Baselines exhibit structural and temporal artifacts, while our method produces coherent and temporally smooth results.

Vanilla   
![](images/5086d217cdc55eee9f02464477403d338add0a7ab32becd15ee2b2e022392142.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a man wearing a blue hoodie (no visible text or symbols)
</details>

![](images/c66295c7c195a8a72b50776b09e4ccabe7b9b10db767a4bc3f4365ca7c68ca66.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a man wearing a blue hoodie (no visible text or symbols)
</details>

SRA   
![](images/efb63505e7f037717f578dec22a64b96b21967426d4fa8084c3221f5f578d2de.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a man wearing a blue hoodie (no visible text or symbols)
</details>

![](images/75c1e1bf1c5b74b38c6dcc774a15be78518d8a06d1e5131894cecc7b45417cdb.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man wearing a blue hoodie (no visible text or symbols)
</details>

REPA   
![](images/68469465923bcc75afa0b569c74ae3b4e1be37f5777ca0552607d38d32729308.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a man wearing a blue hoodie (no visible text or symbols)
</details>

![](images/8b43b042953a49300b151296fbe92911effccc97168e068247977ebd5961aef9.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a man wearing a blue hoodie (no visible text or symbols)
</details>

Ours

![](images/e9ae9370d723bf8cdc0cedcbd8b4b748665f75c13e8281e8f254aa57063961a1.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man wearing a blue hoodie (no text or symbols visible)
</details>

![](images/f25807e113e283d8a66735b3d1dc49ff8d591fda83fabad6f27badfcfdd7ead7.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man wearing a blue hoodie (no text or symbols visible)
</details>

a bald man with a trimmed beard, wearing a blue hooded sweatshirt, directs an intense and determined gaze forward, his face illuminated by soft light. His eyes are wide and mouth slightly open, conveying a focused, mid-sentence expression of direct communication.

Vanilla   
![](images/9d340be33b3f5705f46fc8f8c1d12c3d42824b29fd47ba70a97ca4aa19204519.jpg)

<details>
<summary>natural_image</summary>

Person in black suit mid-air during a dynamic pose, arms extended upward (no text or symbols visible)
</details>

![](images/4a8de93ad057adca188fb9de0c7564fdac0d61749e4786dd216ac5783e0d1172.jpg)

<details>
<summary>natural_image</summary>

Person performing a snow skating move on ice (no text or symbols visible)
</details>

REPA   
![](images/cc00780cecc8b3d351b6e8f17ed587e3a01c22dacf5f4f3cb346c6cc54afbccc.jpg)

<details>
<summary>natural_image</summary>

Person performing a dynamic pose in athletic attire against a plain background (no text or symbols visible)
</details>

![](images/226a1946f7ff590c9ad494a6f2eaee5bd55a44c14aa2d43d4667501f94193373.jpg)

<details>
<summary>natural_image</summary>

Figure skater in black uniform performing a high kick on ice (no text or symbols visible)
</details>

SRA   
![](images/de7fa05d550b7c88233f37fe53c6c6991e98d77c652a299dc5c94f16567758fc.jpg)

<details>
<summary>natural_image</summary>

Silhouette of a person standing barefoot on a plain surface (no text or symbols visible)
</details>

![](images/763c1db84f9fb773eb8d9e39a1c2e389f3f38facd3656714c8234cfb852ad381.jpg)

<details>
<summary>natural_image</summary>

Abstract black ink splatter on white background, no text or symbols visible
</details>

![](images/cc76ae2ec7a30574b1392093c43eb1247b775c07028e240ad0d4a0f0a9b5af5d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's lower body wearing black athletic shoes, standing on ice (no text or symbols visible)
</details>

![](images/a5aab893a4a327b30bf234bc04f440a36b78b8b666b96e3e56a172f7c8d8b2ac.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's feet in black and white, standing on ice with blurred background (no text or symbols visible)
</details>

zoom in on the legs of a figure skater completing a jump combination, blades touching down. The landing is secure, momentum carrying into the next element.

Vanilla   
![](images/d690032141c8f3da36c6066a178d7037395dca37db1d1c7f5498f2d0a8843cbc.jpg)

<details>
<summary>natural_image</summary>

Side profile of a large lion standing in a savanna field (no text or symbols visible)
</details>

![](images/cbe46d1131e692ebe92d356a210a5f8f1b68f649774d3d28a2360ac2ac99d3fe.jpg)

<details>
<summary>natural_image</summary>

Close-up of a lion standing in a grassy field at sunset (no text or symbols visible)
</details>

REPA   
![](images/1f3312500f655471e94cf235990733ab7578b9db7a2e2e10b2b822c6b7c75c4c.jpg)

<details>
<summary>natural_image</summary>

Close-up of a lion standing in a golden savanna setting with blurred human figures in the background (no text or symbols visible)
</details>

![](images/4bca0f269561734353cebfffa7e9109ccbe4c193c70ab1468aef7451bf0164f7.jpg)

<details>
<summary>natural_image</summary>

Close-up of a lion standing in a golden savanna setting (no text or symbols visible)
</details>

SRA   
![](images/5c1b538819d04e02a34054e0b9db7e3541dc7d5df03bb30cfe84c5a43d00c498.jpg)

<details>
<summary>natural_image</summary>

A lion standing in a golden grass field during golden hour (no text or symbols visible)
</details>

![](images/0edaa85ec4ab46af86ec2ac215bbd2d9cdac41752374a14381618449d599792b.jpg)

<details>
<summary>natural_image</summary>

Lion walking in a golden grass field during golden hour (no text or symbols visible)
</details>

![](images/1bd6bc3ccc6a3c36e7d015dde7465573faa904d45e8868e189992f1fecb07a62.jpg)

<details>
<summary>natural_image</summary>

A lion walking in a grassy field at sunset, no visible text or symbols.
</details>

![](images/a11246c541b0b4c6877c3ec4a7141c444235d632bebda9d3bf1467efefb4f5d2.jpg)

<details>
<summary>natural_image</summary>

A lion walking in a grassy field at sunset, no visible text or symbols.
</details>

a powerful male lion walks confidently across the savanna, captured from a low angle that conveys his regal bearing. The tawny predator moves with controlled power, mane flowing slightly in the breeze, amber eyes scanning the horizon as he patrols his territory under golden evening light.

Figure 29. Qualitative comparison (text-to-video). Baselines exhibit structural and temporal artifacts, while our method produces coherent and temporally smooth results.

Vanilla   
![](images/aa941312528d2f873c99baebfc9fbd8ce5e9608deff3462ae69508434a2bcf42.jpg)

<details>
<summary>natural_image</summary>

Group of gor Adults in a lush green forest, no visible text or symbols
</details>

![](images/fe31c774d61770229a2ca656f370a457d228777080958bbaae73687299b5146b.jpg)

<details>
<summary>natural_image</summary>

Two goraks in a lush green forest, one standing and one crouching (no visible text or symbols)
</details>

SRA   
![](images/8f37cb794ecaf543b8e538c74d16f354acb008d924cb104288ac7e1dcee9a624.jpg)

<details>
<summary>natural_image</summary>

Black gorilla in a dense bamboo forest, no visible text or symbols
</details>

![](images/ff13407e6c9531890836245b76c7207b60708f2702bf2b728cb5a36d20e28784.jpg)

<details>
<summary>natural_image</summary>

Black and white gorilla standing in a lush forest (no visible text or symbols)
</details>

REPA   
![](images/e515a002314750c64aad2b248439740c449e2c1e03555aec51c1c5c90303f493.jpg)

<details>
<summary>natural_image</summary>

Close-up of a gorilla standing in a dense forest (no visible text or symbols)
</details>

![](images/b1c055cee7a03a63a6e1cf45fde0c3d74ed216e3f285986dcf1e19c6f3e85399.jpg)

<details>
<summary>natural_image</summary>

Close-up of a gorilla in a lush forest setting (no visible text or symbols)
</details>

Ours   
![](images/d0b17e66af74e371a54b698d3f5ea7e758a45b83986bb48ea3dcf42e6aa0ad82.jpg)

<details>
<summary>natural_image</summary>

Gorilla standing in a lush green forest, no visible text or symbols
</details>

![](images/6bc9416e1803b134dd44d9b36a3d445c1d345875c79962fcff39af533b8c51ce.jpg)

<details>
<summary>natural_image</summary>

Close-up photo of a gorilla in a forest setting, no visible text or symbols
</details>

a silverback gorilla walks through dense jungle, seen from a low, respectful distance. The massive ape moves with quiet power, silver-gray back distinctive, intelligent eyes scanning the forest.

Vanilla   
![](images/0935d57f146a719911e4ea640421d50354ddb33f99e2b836e11b0c8e51665ad6.jpg)

<details>
<summary>natural_image</summary>

A bird perched on a tree branch at dusk, with dark foliage in the background (no visible text or symbols)
</details>

![](images/40cbd0079215ae21ff0f634856363a2fbb6ce72d0daff05d4fccf0809be39aa8.jpg)

<details>
<summary>natural_image</summary>

Close-up of a spotted bird perched on a tree branch at dusk, with no visible text or symbols.
</details>

SRA   
![](images/29e66b7acb328dfe36333a02bde37c16f76e94a191b3f3dbd4d123ccbded70f5.jpg)

<details>
<summary>natural_image</summary>

Close-up of a bird's head with yellow eyes and patterned beak against a twilight sky (no text or symbols visible)
</details>

![](images/80233c14756a21cb464da92aa96670f77cefb6527baea6aad5c19d6aa1450638.jpg)

<details>
<summary>natural_image</summary>

Close-up of an owl perched on a tree branch at night (no text or symbols visible)
</details>

REPA   
![](images/b822669e843eebfa72f939662abdc264b813f2b3232e389e0f74b36224a0f5f4.jpg)

<details>
<summary>natural_image</summary>

Close-up of an owl perched on a tree branch at dusk, with green foliage in the background (no text or symbols visible)
</details>

![](images/6b0278ee7c5a7f87497fc4fbcb3f8577d20e61bdd203bc74624ca988b409f48d.jpg)

<details>
<summary>natural_image</summary>

Owl perched on a tree branch at dusk, with green foliage in the background (no text or symbols visible)
</details>

![](images/2ba3523a1fef51b7312755247727a5cc87557a16a2fed440a4f5fe70e3018316.jpg)

<details>
<summary>natural_image</summary>

Close-up of a small bird perched on a branch against a blue sky, with green foliage in the background (no text or symbols visible)
</details>

![](images/c50dcc6d66bbc284ced0c16a7cb3cfa6666f955a042ffb54e78a1973f971e54c.jpg)

<details>
<summary>natural_image</summary>

Close-up of an owl perched on a branch against a blue sky (no text or symbols visible)
</details>

a 3D cartoon of an owl turns its head around while perched on a tree branch at dusk, worried expression, large yellow eyes blinking slowly.

Vanilla   
![](images/ebb0c95a766c23bbc94f001fba24f8e1761cb906511c447d42216a3b297be238.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of an elderly man with a beard and open mouth (no text or symbols visible)
</details>

![](images/50f6488c4e73f06902bf759a4b6d27865149e053499c3ab164c1be56965b198f.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of an elderly man with a beard and neutral expression (no text or symbols visible)
</details>

SRA   
![](images/bd42fa3ca6faafe5cabe05e8dd0bd0149acfb5bf06f10223924b0ae98a74c2a3.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of an elderly man with short hair and beard, against a dark background (no text or symbols visible)
</details>

![](images/43ec098016352eae04250902a13081c67b18a2ce210da369d1960bd4f118a8c0.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of an elderly man with curly hair and beard (no visible text or symbols)
</details>

REPA   
![](images/8137421b720bcd61fd2775e48b3d2c099948a82847465aeb59c6e52c1c03b596.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's face showing skin texture and neutral expression (no text or symbols visible)
</details>

![](images/2fcc83acb7452ff7dafb0da6f5e7031c7d1ce8c359acd2cb9b1ad9c14bc3497d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a person's face showing skin texture and facial features (no text or symbols visible)
</details>

Ours   
![](images/6ecff49ed169a8d1cce7dbb7f9f96efee32f74b841fcfadf8a2c35ae832f3fb6.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of an elderly man with a beard and serious expression (no visible text or symbols)
</details>

![](images/a1418676e88ab80ec2eb7d3dec75578080073e7cd9226a6344652141f2e9a0f8.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a man with a beard and serious expression (no visible text or symbols)
</details>

an elderly man with deep wrinkles and a neatly trimmed grey beard speaks thoughtfully to the camera. His weathered face fills the frame as his lips form each word deliberately, occasionally pausing to gather his thoughts, warm golden lamplight casting gentle shadows across his features.

Figure 30. Qualitative comparison (text-to-video). Baselines exhibit structural and temporal artifacts, while our method produces coherent and temporally smooth results.

Vanilla   
![](images/f7efb5072cbe589a12aec49294c9516dc3b10a4db29ffbe4c65edc3720e68322.jpg)

<details>
<summary>natural_image</summary>

Black equilts with large horns standing in a grassy field (no text or symbols visible)
</details>

![](images/f49bbfabaf177952bd49c76f82623b9f5501710485508e31b88fe9ff862dce8d.jpg)

<details>
<summary>natural_image</summary>

Black buffalo standing in a grassy field, no visible text or symbols
</details>

SRA   
![](images/82e7f03adfa8800138c8b09ee384731263ee3ebe149dc62d3babc30c4a9dcb26.jpg)

<details>
<summary>natural_image</summary>

A black buffalo standing in a grassy field, facing forward with large horns (no text or symbols visible)
</details>

![](images/a000ada99788041bcee15d10fcee8c13cba90fb19204b70f074ad424902cabe6.jpg)

<details>
<summary>natural_image</summary>

Black equilts standing in a grassy field with tall yellow grass (no text or symbols visible)
</details>

REPA   
![](images/f26a585ad51b5d697cf19e86549907e63b2a3e005d8f6a16560a87e13c8a3ab1.jpg)

<details>
<summary>natural_image</summary>

Black buffalo standing in a golden grass field, no visible text or symbols
</details>

![](images/d3c52bf2c9e2452a4d2d225a7119f9baee26892f61ecda9a836143ea38c68476.jpg)

<details>
<summary>natural_image</summary>

A black equilts with large white horns standing in a dry grassland field (no text or symbols visible)
</details>

Ours   
![](images/faeeddfb84e2ab6ef5f4850c9ff00fce6968b099e632209446e675b9720d685a.jpg)

<details>
<summary>natural_image</summary>

Black buffalo standing in a dry grassland field (no text or symbols visible)
</details>

![](images/8fb6e1d2fc33201fb2fa77ecdbfeed83dba1368ea6fa78fa825f024aeb1f8fe5.jpg)

<details>
<summary>natural_image</summary>

Black equatihuacup standing in a dry grassland field (no text or symbols visible)
</details>

a cape bull stands its ground, facing the camera with notorious aggression in its posture. The powerful animal's horns form a deadly boss, one of Africa's most dangerous creatures ready to charge.

Vanilla   
![](images/66eb135640804a7275d9171b3cdbe291e81926128c14bbadb0a9d45b9875a88c.jpg)

<details>
<summary>natural_image</summary>

Close-up of hands cutting a whole squash on a wooden cutting board (no text or symbols visible)
</details>

![](images/aa5e328c181222f32cdd6f08240beb39fd4ed0caf6aba23c0dfd858715fe7804.jpg)

<details>
<summary>natural_image</summary>

Close-up of hands cutting a round, textured food item with a knife on a wooden surface (no text or symbols visible)
</details>

SRA   
![](images/4429d66a9d1687a42fc80bb2c015dc682d05bd2fa68bf42a690d4cb75eb43596.jpg)

<details>
<summary>natural_image</summary>

Person cutting a yellow gourd on a wooden surface, with a spatula in use (no text or symbols visible)
</details>

![](images/ba803aa92e1501691d74a49a801968509a6f9a3f57683c04463bbbcbd051aa88.jpg)

<details>
<summary>natural_image</summary>

Person slicing a yellow melon on a wooden cutting board (no text or symbols visible)
</details>

REPA   
![](images/ff8eaef1e11217ed3b4e6c1cac399e7a95d1a66b6a0f597f93a472634fb1c6dc.jpg)

<details>
<summary>natural_image</summary>

Close-up of a hand using a knife to cut a yellow cake with visible oil, no text or symbols present.
</details>

![](images/be4e7a4f5fe26b91be163283f054a3c5055c3d7713efa06f71aef3df26f684f3.jpg)

<details>
<summary>natural_image</summary>

Close-up of hands using a knife to cut a yellow cheese-like substance (no text or symbols visible)
</details>

![](images/6b6fa672a3111a048e403ce64bbd6e24e37e9cbe6145896f46e146ef82f89b10.jpg)

<details>
<summary>natural_image</summary>

Person cutting a wooden cake on a wooden table, no visible text or symbols
</details>

![](images/aa2cf22087f5d9f6b6ebe7cf2acf6e1f41cca0da7d918575c7a65d01ebe3bc71.jpg)

<details>
<summary>natural_image</summary>

Person slicing a wooden block on a wooden cutting board (no text or symbols visible)
</details>

a cheese expert cuts into a perfectly aged wheel of Parmigiano-Reggiano.

Vanilla   
![](images/be7eee8d52fa3d5826d417c7d020e3fee89472efab87f9991065c3d908079002.jpg)

<details>
<summary>natural_image</summary>

Close-up of a blue, translucent shark head against a solid blue background (no text or symbols visible)
</details>

![](images/56ee22c3776d653647eb830620f1a9b3a09d254e0ef742f69ae8cafb58c6a5ed.jpg)

<details>
<summary>natural_image</summary>

Abstract blue-toned image with a smooth, curved shape against a dark blue background (no text or symbols)
</details>

SRA   
![](images/8860eb862a7c9ceb80ff1e6476e087a5973a04dac05915a60270305b4d0ea638.jpg)

<details>
<summary>natural_image</summary>

Underwater close-up of a translucent shark against a blue background (no text or symbols visible)
</details>

![](images/9b3b1a4b3092c2e30a5e7dfe8779ab4c62bd97888a464d98869bf554ddccb1c1.jpg)

<details>
<summary>natural_image</summary>

Underwater close-up of a dark, whale-like animal swimming against a blue background (no text or symbols visible)
</details>

REPA   
![](images/f1c2989c3f40bcdfbb68895d35dd0ca5f37d4c7118380630b767a056e1a56ad5.jpg)

<details>
<summary>natural_image</summary>

Underwater close-up of a dolphin against a solid blue background (no text or symbols visible)
</details>

![](images/0ba2bdc9595e0db77335c1c7a0e62849cea006f7f5788605bcc6f7d8a1e40924.jpg)

<details>
<summary>natural_image</summary>

Solid blue background with a faint diagonal line in the corner (no text or symbols)
</details>

Ours   
![](images/a6b08f96fd0c12b5c0fffc2d1691f3e969ea518bf9390295b3a6f16a1e4c7775.jpg)

<details>
<summary>natural_image</summary>

Underwater close-up of a shark swimming in blue water (no visible text or symbols)
</details>

![](images/df0754a0ceb42fb5958b7abfb9c636e5a178ef2f2bf98d21460e87a447f7bd26.jpg)

<details>
<summary>natural_image</summary>

Underwater close-up of a shark swimming in blue water (no visible text or symbols)
</details>

an underwater view captures a large blue shark, starting with a medium shot of its head and front body, as it gracefully turns and begins swimming away through the deep, deep blue ocean water. The camera smoothly tracks the shark as it performs a swift turn.

Figure 31. Qualitative comparison (text-to-video). Baselines exhibit structural and temporal artifacts, while our method produces coherent and temporally smooth results.

Vanilla Flow Matching   
![](images/28b442ee044e2f47ad0f9764ddf51b83d26490335ed3fb29e4052797a713365e.jpg)

<details>
<summary>text_image</summary>

From the
Black Forest
VHD LOVE
</details>

![](images/56bf5b64fe004a23bc5bb8b84247a95bf702f88ee5bf2289403c169acda06310.jpg)

<details>
<summary>text_image</summary>

Flu
wuas here
</details>

![](images/0927e3e02fdb3dc8d5bdcc8e874d2eabc99893c3513630167fe1610488e1a693.jpg)

<details>
<summary>text_image</summary>

100K
stps of talking
</details>

![](images/c5c6fc93fffbe81d39fc5a827b56b5a8c722aced5ce2a6d2407465021c132198.jpg)

<details>
<summary>text_image</summary>

BLACK
FERFST
NIGHTS
</details>

Ours   
![](images/cd669026b4fbd2a4b5506dafb045b9d34eee41ecd302cd5b8da1ad04aa8c09ba.jpg)

<details>
<summary>text_image</summary>

from the
Black Forest
of love
</details>

![](images/273a63a26b49edafc017325c58332828c99602b5c39593c989c15066f4db1fa9.jpg)

<details>
<summary>text_image</summary>

FLUX
was here
</details>

![](images/e1505a51a93bfa6e15c26f74d3b4edd5c57a87a75b5a991af0c8db1e79ad0a04.jpg)

<details>
<summary>text_image</summary>

100K steps
of training
</details>

![](images/c1e045d6d789a6b3c19e7fa31e18807e939e5c6b0365f6c8dd4b81944cd8e0d5.jpg)

<details>
<summary>text_image</summary>

BLACK
FOREST
NIGHTS
</details>

Vanilla Flow Matching   
![](images/959178099463cac47c4755fb2da45a3f9a0cedcf87eb5e40567d6d2d1584b982.jpg)

<details>
<summary>text_image</summary>

Powved
Byxra 4B
</details>

![](images/0429e4a55a5b28089ed61e62ec7561117032f7f424cba16f180d008d7d41886a.jpg)

<details>
<summary>text_image</summary>

FLUX
nvverr
sleepers
</details>

![](images/d798ed7b2d9a43fa8ba05d34e0623436afc8f5756f5e907e9a9beb28aa987223.jpg)

<details>
<summary>text_image</summary>

wewtle-va
wait just 100K steps??
</details>

![](images/b543bd93c64a055c2fdb99918f7df5d03cbd79fb45c219425d018df95e2d895e.jpg)

<details>
<summary>text_image</summary>

made
by
by
X
BFL
</details>

Ours   
![](images/a32b53c28a31a26e58ed0df7aa1315bddfd1416de2c9b9f1b572860203310e52.jpg)

<details>
<summary>text_image</summary>

Powered
by 4B
</details>

![](images/4d31429272229577da1f88030d9d7714d45665b617e22eaa2c3a9deefccd7e78.jpg)

<details>
<summary>text_image</summary>

FLUX
never sleeps
</details>

![](images/50cd97263a9a6dadccbb2a12f5fe64efcf3ae887e60a0245a19f554bd9aefcf1.jpg)

<details>
<summary>text_image</summary>

wait just 100K steps??
</details>

![](images/50a4312c646df886e82ebde8fd7620699aad5982b0aec37994f2cc6c74886033.jpg)

<details>
<summary>text_image</summary>

Made
by BFL
</details>

Vanilla Flow Matching   
![](images/e5661d978012dd6c194ad65c8e92ec7d86863123d537cf9b62b84211e278cb91.jpg)

<details>
<summary>text_image</summary>

FLUX
dreams in
multiimodal
</details>

![](images/ae0f4330623f98541f24c7ce391d53530d7906276033fff9c4a62ead877ad2eb.jpg)

<details>
<summary>text_image</summary>

FLUX
leves here
</details>

![](images/3345fb831d3b80f19ee5a495b802870ae7a827170dd2d1e82069d831c4017056.jpg)

<details>
<summary>natural_image</summary>

A hedgehog resting on a wooden surface with scattered autumn leaves and the word 'ELIX' written in the foreground (no other text or symbols visible)
</details>

![](images/5bb353f3a732bce43ae1a917a9dec8214d940db48bc8b4e19f99dbfac55f5416.jpg)

<details>
<summary>text_image</summary>

4B × 100K ≡ magsig
</details>

Ours   
![](images/b7bb390a09599f6b45bd323fa7de8c5736744c2c478605c8a0c6acbf905d7226.jpg)

<details>
<summary>text_image</summary>

FLUX
dreams in
multimodal
</details>

![](images/5ef81545818a0a2b24d21e3bc92b9d535ed4540fb50a43765ff6b6167d661897.jpg)

<details>
<summary>text_image</summary>

FLUX
lives here
</details>

![](images/526ddb1a4be3be8db5f9897a9f0042b0d6302f3d747cf6eaafd81aa5c997cc10.jpg)

<details>
<summary>natural_image</summary>

A hedgehog peeking over autumn leaves and seeds spelling 'FLUX' (no text or symbols on the image itself)
</details>

![](images/3a1fa67b7cfa67e847e2a1e9c575465a070a8f3c7a1faa3640599235849026c3.jpg)

<details>
<summary>text_image</summary>

4B × 100K =
magic
</details>

Figure 32. Improved typography in text-to-image generation. Results from our scaled 4B parameter multi-modal model after just 100K high-resolution finetuning steps. Our method produces accurate and legible text rendering compared to vanilla flow matching.

![](images/3e700019520905155f8ede581c22e6ec87d7787147b419197f2a5886f56463bc.jpg)  
Figure 33. Improved typography in text-to-image generation. Results from our scaled 4B parameter multi-modal model after just 100K high-resolution finetuning steps. Our method produces accurate and legible text rendering compared to vanilla flow matching.

![](images/56157bbbcc15b9b0f028a4ac3de3a13243f93401318d5650b1fd399d33d65914.jpg)  
Figure 34. Improved typography in text-to-image generation. Results from our scaled 4B parameter multi-modal model after just 100K high-resolution finetuning steps. Our method produces accurate and legible text rendering compared to vanilla flow matching.